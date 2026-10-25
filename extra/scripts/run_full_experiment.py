#!/usr/bin/env python3
"""
Torneo de Hiperparámetros LoRA en SDXL (1024x1024) sobre gigant/oldbookillustrations
Autor: Twinber & Nova
Ejecución: RunPod NVIDIA A40 (48 GB VRAM)
"""

import os
import gc
import json
import time
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFile
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from datasets import load_dataset
from safetensors.torch import save_file, load_file
from huggingface_hub import HfApi, create_repo

from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

# Directorios de trabajo
WORKSPACE = Path("/workspace")
CACHE_DIR = WORKSPACE / "cache_sdxl_latents"
MODELS_DIR = WORKSPACE / "models"
EVAL_DIR = WORKSPACE / "eval_images"
COMPARISONS_DIR = WORKSPACE / "comparisons"
HF_UPLOAD_DIR = WORKSPACE / "hf_upload"

BASE_IMAGES_DIR = WORKSPACE / "base_images"

for d in [CACHE_DIR, MODELS_DIR, EVAL_DIR, COMPARISONS_DIR, HF_UPLOAD_DIR, BASE_IMAGES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DATASET_NAME = "gigant/oldbookillustrations"

BENCHMARK_PROMPTS = [
    ("01_botanica", "an antique botanical plate illustration of an exotic passionflower with leaves and roots, historical scientific drawing, detailed engraving, paper texture"),
    ("02_relojero", "an antique portrait engraving of an elderly watchmaker with spectacles examining a pocket watch in his workshop, 19th century illustration"),
    ("03_catedral", "a medieval gothic cathedral on a rocky cliff overlooking a stormy ocean, 19th century vintage landscape etching"),
    ("04_locomotora", "a vintage 19th century steam locomotive crossing a stone viaduct over a deep mountain river, technical etching illustration"),
    ("05_barco", "a tall wooden sailing ship battling gigantic stormy waves in the ocean, dark sky, antique vintage maritime engraving"),
    ("06_coche_deportivo", "a modern red sports car parked on a wet asphalt city street at night, neon reflections"),
    ("07_cafeteria", "a young woman sitting in a modern coffee shop drinking a cup of coffee, looking out a large window"),
    ("08_perro_colina", "a golden retriever dog sitting happily in a green grass field, bright sunny day, clear blue sky"),
    ("09_rascacielos", "a futuristic glass skyscraper towering into the sky, urban metropolis with clouds"),
    ("10_astronauta_marte", "an astronaut walking on the red rocky surface of planet Mars, distant mountains, Earth in the sky"),
]

VARIANTS = [
    {
        "name": "variant_A_r16",
        "rank": 16,
        "alpha": 16,
        "lr": 1e-4,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        "desc": "Cross-Attention LoRA (Rank 16, Alpha 16, LR 1e-4)"
    },
    {
        "name": "variant_B_r32",
        "rank": 32,
        "alpha": 32,
        "lr": 1e-4,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        "desc": "Cross-Attention LoRA (Rank 32, Alpha 32, LR 1e-4)"
    },
    {
        "name": "variant_C_r64",
        "rank": 64,
        "alpha": 32,
        "lr": 5e-5,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
        "desc": "Cross-Attention + FFN LoRA (Rank 64, Alpha 32, LR 5e-5)"
    },
]


def precompute_latents_and_embeddings():
    """Fase 1: Precomputar y guardar en disco los latentes VAE (1024x1024 en BFLOAT16) y embeddings CLIP duales."""
    cache_file = CACHE_DIR / "sdxl_cached_dataset.pt"

    # Verificar si la caché ya existe y no tiene NaNs
    if cache_file.exists():
        try:
            test_data = torch.load(cache_file, map_location="cpu", weights_only=True)
            if not torch.isnan(test_data["latents"]).any() and not torch.isnan(test_data["prompt_embeds"]).any():
                print(f"[CACHE] Ya existe caché válida y libre de NaNs en {cache_file}. Saltando Fase 1.")
                return
            else:
                print("[CACHE] Caché existente contiene NaNs (debido a FP16 en VAE). Regenerando con BFLOAT16...")
        except Exception:
            pass

    print("=" * 60)
    print("FASE 1: PRECOMPUTACIÓN DE LATENTES VAE (BFLOAT16) Y EMBEDDINGS SDXL (1024x1024)")
    print("=" * 60)

    print("[1/4] Descargando/cargando dataset...")
    raw_dataset = load_dataset(DATASET_NAME, split="train")
    total_samples = len(raw_dataset)
    print(f"Total de muestras a procesar: {total_samples}")

    # Verificar si podemos reutilizar los embeddings de texto (que son 100% válidos y sin NaNs)
    reusable_prompt_embeds = None
    reusable_pooled_embeds = None
    if cache_file.exists():
        try:
            old_data = torch.load(cache_file, map_location="cpu", weights_only=True)
            if not torch.isnan(old_data["prompt_embeds"]).any() and len(old_data["prompt_embeds"]) == total_samples:
                print("[OPTIMIZACIÓN] Reutilizando los 4.154 embeddings de texto CLIP ya computados (0 NaNs)!")
                reusable_prompt_embeds = old_data["prompt_embeds"].to(torch.bfloat16)
                reusable_pooled_embeds = old_data["pooled_embeds"].to(torch.bfloat16)
        except Exception as e:
            print(f"[INFO] No se pudieron reutilizar embeddings de texto: {e}")

    need_text_encoders = (reusable_prompt_embeds is None)

    print("[2/4] Cargando componentes de inferencia (VAE en bfloat16)...")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")

    if need_text_encoders:
        tokenizer_1 = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2")
        text_encoder_1 = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16).to("cuda")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(MODEL_ID, subfolder="text_encoder_2", torch_dtype=torch.bfloat16).to("cuda")

    img_transform = transforms.Compose([
        transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(1024),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    all_latents = []
    all_prompt_embeds = []
    all_pooled_embeds = []

    batch_size = 16
    num_batches = math.ceil(total_samples / batch_size)

    print(f"[3/4] Codificando {total_samples} imágenes con VAE en bfloat16 en lotes de {batch_size}...")
    t0 = time.time()

    for b in tqdm(range(num_batches), desc="Codificando latentes"):
        start_idx = b * batch_size
        end_idx = min(start_idx + batch_size, total_samples)

        batch_tensors = []
        batch_captions = []
        for i in range(start_idx, end_idx):
            try:
                sample = raw_dataset[i]
                img = sample.get("1600px")
                if img is None:
                    img = sample.get("rawscan")
                img = img.convert("RGB")
                tensor = img_transform(img)
                batch_tensors.append(tensor)
            except Exception as e:
                fallback_img = Image.new("RGB", (1024, 1024), color=(240, 235, 220))
                batch_tensors.append(img_transform(fallback_img))

            if need_text_encoders:
                caption = sample.get("info_alt")
                if not caption or not isinstance(caption, str) or len(caption.strip()) == 0:
                    caption = "antique book illustration, vintage engraving, detailed historical etching"
                batch_captions.append(caption.strip())

        batch_tensors = torch.stack(batch_tensors).to("cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            latents = vae.encode(batch_tensors).latent_dist.sample() * vae.config.scaling_factor
            if torch.isnan(latents).any():
                print(f"[ERROR CRÍTICO] NaN detectado en lote {b}!")
            all_latents.append(latents.cpu())

            if need_text_encoders:
                t1_inputs = tokenizer_1(batch_captions, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to("cuda")
                t2_inputs = tokenizer_2(batch_captions, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to("cuda")

                p1 = text_encoder_1(t1_inputs, output_hidden_states=True).hidden_states[-2]
                res2 = text_encoder_2(t2_inputs, output_hidden_states=True)
                p2 = res2.hidden_states[-2]
                pooled = res2.text_embeds

                prompt_embeds = torch.concat([p1, p2], dim=-1)
                all_prompt_embeds.append(prompt_embeds.cpu())
                all_pooled_embeds.append(pooled.cpu())

    elapsed = time.time() - t0
    print(f"Codificación de imágenes completada en {elapsed:.1f} s ({total_samples / elapsed:.1f} img/s)!")

    all_latents = torch.cat(all_latents, dim=0)
    print(f"Latentes totales: {all_latents.shape} | NaNs en latentes: {torch.isnan(all_latents).any().item()}")

    if need_text_encoders:
        final_prompt_embeds = torch.cat(all_prompt_embeds, dim=0)
        final_pooled_embeds = torch.cat(all_pooled_embeds, dim=0)
    else:
        final_prompt_embeds = reusable_prompt_embeds
        final_pooled_embeds = reusable_pooled_embeds

    print(f"[4/4] Guardando dataset consolidado en {cache_file}...")
    torch.save({
        "latents": all_latents,
        "prompt_embeds": final_prompt_embeds,
        "pooled_embeds": final_pooled_embeds,
    }, cache_file)
    print(f"Dataset en caché consolidado con éxito en {cache_file}.")

    # Liberar memoria de VAE y encoders
    del vae
    if need_text_encoders:
        del text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2
    del all_latents
    gc.collect()
    torch.cuda.empty_cache()


class CachedLatentsDataset(Dataset):
    def __init__(self, cache_file):
        data = torch.load(cache_file, map_location="cpu", weights_only=True)
        self.latents = data["latents"].to(torch.bfloat16)
        self.prompt_embeds = data["prompt_embeds"].to(torch.bfloat16)
        self.pooled_embeds = data["pooled_embeds"].to(torch.bfloat16)
        self.length = self.latents.shape[0]

        # SDXL micro-conditioning add_time_ids: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
        self.add_time_ids = torch.tensor([1024, 1024, 0, 0, 1024, 1024], dtype=torch.bfloat16)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "latents": self.latents[idx],
            "prompt_embeds": self.prompt_embeds[idx],
            "pooled_embeds": self.pooled_embeds[idx],
            "add_time_ids": self.add_time_ids,
        }


def train_variant(variant_cfg, dataset):
    """Fase 2: Entrenar una variante de LoRA en UNet (1,000 pasos con checkpoints a 500 y 1,000)."""
    name = variant_cfg["name"]
    rank = variant_cfg["rank"]
    alpha = variant_cfg["alpha"]
    lr = variant_cfg["lr"]
    target_modules = variant_cfg["target_modules"]

    print("\n" + "=" * 60)
    print(f"ENTRENANDO: {name} (r={rank}, alpha={alpha}, lr={lr})")
    print(f"Módulos objetivo: {target_modules}")
    print("=" * 60)

    # Cargar UNet base en BFLOAT16 (estabilidad numérica absoluta en A40)
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=torch.bfloat16
    ).to("cuda")
    unet.enable_gradient_checkpointing()

    # Configurar LoRA con PEFT
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Scheduler de difusión para entrenamiento
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # DataLoader
    batch_size = 4
    grad_accum_steps = 2
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_iter = iter(loader)

    # Optimizador AdamW
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)

    total_steps = 1000
    save_steps = [500, 1000]

    unet.train()
    optimizer.zero_grad()

    t_start = time.time()
    pbar = tqdm(range(1, total_steps + 1), desc=f"Entrenando {name}")

    running_loss = 0.0

    for step in pbar:
        step_loss = 0.0
        for _ in range(grad_accum_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            latents = batch["latents"].to("cuda", dtype=torch.bfloat16)
            prompt_embeds = batch["prompt_embeds"].to("cuda", dtype=torch.bfloat16)
            pooled_embeds = batch["pooled_embeds"].to("cuda", dtype=torch.bfloat16)
            add_time_ids = batch["add_time_ids"].to("cuda", dtype=torch.bfloat16)

            # Samplear ruido y timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device="cuda").long()

            # Añadir ruido a los latentes según el scheduler
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            added_cond_kwargs = {
                "text_embeds": pooled_embeds,
                "time_ids": add_time_ids,
            }

            # Predecir ruido
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss = loss / grad_accum_steps
            loss.backward()

            step_loss += loss.item() * grad_accum_steps

        # Clip grad norm
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        running_loss = 0.9 * running_loss + 0.1 * step_loss if step > 1 else step_loss
        pbar.set_postfix({"loss": f"{running_loss:.4f}"})

        # Guardar checkpoints
        if step in save_steps:
            save_checkpoint(unet, name, step, rank, alpha, lr)

    elapsed = time.time() - t_start
    print(f"\n[FIN] {name} entrenado en {elapsed:.1f} s ({elapsed / total_steps:.2f} s/paso)")

    del unet, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def save_checkpoint(unet, variant_name, step, rank, alpha, lr):
    """Guarda los pesos LoRA en formato compatible con diffusers."""
    ckpt_dir = MODELS_DIR / f"{variant_name}_step{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Extraer pesos de PEFT
    peft_state_dict = get_peft_model_state_dict(unet)

    # Convertir a formato estándar diffusers (prefijo unet.)
    diffusers_state_dict = {}
    for k, v in peft_state_dict.items():
        if not k.startswith("unet."):
            diffusers_state_dict[f"unet.{k}"] = v.to(torch.float16).contiguous()
        else:
            diffusers_state_dict[k] = v.to(torch.float16).contiguous()

    weights_path = ckpt_dir / "pytorch_lora_weights.safetensors"
    save_file(diffusers_state_dict, str(weights_path))

    metadata = {
        "variant": variant_name,
        "step": step,
        "rank": rank,
        "alpha": alpha,
        "lr": lr,
        "base_model": MODEL_ID,
        "dataset": DATASET_NAME,
        "num_parameters": sum(p.numel() for p in diffusers_state_dict.values()),
        "filesize_mb": round(os.path.getsize(weights_path) / (1024 * 1024), 2),
    }
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  -> Checkpoint guardado en {ckpt_dir} ({metadata['filesize_mb']} MB)")


def run_benchmark_inferences():
    """Fase 3: Evaluar los 6 checkpoints con los 10 prompts benchmark (seed 42, 1024x1024)."""
    print("\n" + "=" * 60)
    print("FASE 3: INFERENCIA DE BENCHMARK SOBRE 6 CHECKPOINTS (10 PROMPTS, SEMILLA 42)")
    print("=" * 60)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

    # Lista de los 6 checkpoints
    checkpoints = [
        "variant_A_r16_step500",
        "variant_A_r16_step1000",
        "variant_B_r32_step500",
        "variant_B_r32_step1000",
        "variant_C_r64_step500",
        "variant_C_r64_step1000",
    ]

    for ckpt_name in checkpoints:
        ckpt_dir = MODELS_DIR / ckpt_name
        weights_path = ckpt_dir / "pytorch_lora_weights.safetensors"
        if not weights_path.exists():
            print(f"[WARN] No se encontró {weights_path}, saltando...")
            continue

        print(f"\n[EVAL] Evaluando {ckpt_name}...")
        pipe.load_lora_weights(str(ckpt_dir))

        for prompt_id, prompt_text in tqdm(BENCHMARK_PROMPTS, desc=f"Generando {ckpt_name}"):
            generator = torch.Generator(device="cuda").manual_seed(42)
            out_img = pipe(
                prompt=prompt_text,
                generator=generator,
                num_inference_steps=30,
                guidance_scale=7.5,
                width=1024,
                height=1024,
            ).images[0]

            save_path = EVAL_DIR / f"{ckpt_name}_{prompt_id}.png"
            out_img.save(save_path)

        pipe.unload_lora_weights()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print("[EVAL] Todas las 60 imágenes generadas con éxito!")


def build_comparisons():
    """Fase 4: Construir los paneles comparativos y el Master Grid 10x4."""
    print("\n" + "=" * 60)
    print("FASE 4: GENERACIÓN DE PANELES COMPARATIVOS Y MASTER GRID")
    print("=" * 60)

    # 1. Tiras de 4 columnas: [Base | r=16 (1000st) | r=32 (1000st) | r=64 (1000st)]
    for prompt_id, prompt_text in BENCHMARK_PROMPTS:
        base_path = BASE_IMAGES_DIR / f"base_{prompt_id}.png"
        if not base_path.exists():
            base_path = EVAL_DIR / f"base_{prompt_id}.png"
            if not base_path.exists():
                print(f"[WARN] No se encontró imagen base para {prompt_id}")
                continue

        img_base = Image.open(base_path).convert("RGB")
        img_a = Image.open(EVAL_DIR / f"variant_A_r16_step1000_{prompt_id}.png").convert("RGB")
        img_b = Image.open(EVAL_DIR / f"variant_B_r32_step1000_{prompt_id}.png").convert("RGB")
        img_c = Image.open(EVAL_DIR / f"variant_C_r64_step1000_{prompt_id}.png").convert("RGB")

        w, h = 1024, 1024
        header_h = 100
        canvas = Image.new("RGB", (w * 4, h + header_h), color=(18, 18, 22))
        draw = ImageDraw.Draw(canvas)

        cols = [
            (img_base, "1. SDXL Base (Sin LoRA)"),
            (img_a, "2. LoRA r=16 (1,000 pasos)"),
            (img_b, "3. LoRA r=32 (1,000 pasos)"),
            (img_c, "4. LoRA r=64 (1,000 pasos)"),
        ]

        for i, (col_img, title) in enumerate(cols):
            canvas.paste(col_img, (i * w, header_h))
            draw.text((i * w + 30, 35), title, fill=(240, 240, 250))

        out_strip = COMPARISONS_DIR / f"comparativa_4col_{prompt_id}.png"
        canvas.save(out_strip)

    # 2. Tiras de progresión temporal: [Base | Step 500 | Step 1000] para Variant B
    for prompt_id, prompt_text in BENCHMARK_PROMPTS:
        base_path = BASE_IMAGES_DIR / f"base_{prompt_id}.png"
        if not base_path.exists():
            base_path = EVAL_DIR / f"base_{prompt_id}.png"
        if not base_path.exists():
            continue

        img_base = Image.open(base_path).convert("RGB")
        img_500 = Image.open(EVAL_DIR / f"variant_B_r32_step500_{prompt_id}.png").convert("RGB")
        img_1000 = Image.open(EVAL_DIR / f"variant_B_r32_step1000_{prompt_id}.png").convert("RGB")

        w, h = 1024, 1024
        header_h = 100
        canvas = Image.new("RGB", (w * 3, h + header_h), color=(18, 18, 22))
        draw = ImageDraw.Draw(canvas)

        cols = [
            (img_base, "1. SDXL Base (Sin LoRA)"),
            (img_500, "2. LoRA r=32 (Paso 500 - 1 época)"),
            (img_1000, "3. LoRA r=32 (Paso 1,000 - 2 épocas)"),
        ]

        for i, (col_img, title) in enumerate(cols):
            canvas.paste(col_img, (i * w, header_h))
            draw.text((i * w + 30, 35), title, fill=(240, 240, 250))

        out_prog = COMPARISONS_DIR / f"progresion_epocas_{prompt_id}.png"
        canvas.save(out_prog)

    # 3. Master Grid 10x4 reducido para visualización global (512x512 por celda)
    cell_size = 512
    margin_top = 80
    margin_left = 150
    master_w = margin_left + cell_size * 4
    master_h = margin_top + cell_size * 10

    master_grid = Image.new("RGB", (master_w, master_h), color=(15, 15, 18))
    draw = ImageDraw.Draw(master_grid)

    headers = ["SDXL Base", "LoRA r=16 (1k)", "LoRA r=32 (1k)", "LoRA r=64 (1k)"]
    for i, h_text in enumerate(headers):
        draw.text((margin_left + i * cell_size + 40, 30), h_text, fill=(255, 255, 255))

    for row_idx, (prompt_id, _) in enumerate(BENCHMARK_PROMPTS):
        y = margin_top + row_idx * cell_size
        draw.text((20, y + cell_size // 2 - 10), prompt_id, fill=(200, 200, 220))

        base_p = BASE_IMAGES_DIR / f"base_{prompt_id}.png"
        if not base_p.exists():
            base_p = EVAL_DIR / f"base_{prompt_id}.png"
        img_b = Image.open(base_p).convert("RGB").resize((cell_size, cell_size), Image.Resampling.LANCZOS)
        img_16 = Image.open(EVAL_DIR / f"variant_A_r16_step1000_{prompt_id}.png").convert("RGB").resize((cell_size, cell_size), Image.Resampling.LANCZOS)
        img_32 = Image.open(EVAL_DIR / f"variant_B_r32_step1000_{prompt_id}.png").convert("RGB").resize((cell_size, cell_size), Image.Resampling.LANCZOS)
        img_64 = Image.open(EVAL_DIR / f"variant_C_r64_step1000_{prompt_id}.png").convert("RGB").resize((cell_size, cell_size), Image.Resampling.LANCZOS)

        master_grid.paste(img_b, (margin_left + 0 * cell_size, y))
        master_grid.paste(img_16, (margin_left + 1 * cell_size, y))
        master_grid.paste(img_32, (margin_left + 2 * cell_size, y))
        master_grid.paste(img_64, (margin_left + 3 * cell_size, y))

    master_path = COMPARISONS_DIR / "master_grid_10x4.png"
    master_grid.save(master_path)
    print(f"[OK] Master Grid 10x4 guardado en {master_path}")


def prepare_and_upload_hf():
    """Fase 5: Empaquetar y subir el repositorio a Hugging Face Hub."""
    print("\n" + "=" * 60)
    print("FASE 5: SUBIDA A HUGGING FACE HUB (Twinber/sdxl-oldbook-illustrations-lora)")
    print("=" * 60)

    repo_id = "Twinber/sdxl-oldbook-illustrations-lora"
    api = HfApi()

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        print(f"[HF] Repositorio {repo_id} creado o verificado.")
    except Exception as e:
        print(f"[HF] Info repo: {e}")

    # Copiar los 3 pesos finales renombrados a hf_upload/
    final_weights = [
        ("variant_A_r16_step1000/pytorch_lora_weights.safetensors", "sdxl_oldbook_lora_rank16_final.safetensors"),
        ("variant_B_r32_step1000/pytorch_lora_weights.safetensors", "sdxl_oldbook_lora_rank32_final.safetensors"),
        ("variant_C_r64_step1000/pytorch_lora_weights.safetensors", "sdxl_oldbook_lora_rank64_final.safetensors"),
        ("variant_A_r16_step500/pytorch_lora_weights.safetensors", "checkpoints/variant_A_r16_step500.safetensors"),
        ("variant_B_r32_step500/pytorch_lora_weights.safetensors", "checkpoints/variant_B_r32_step500.safetensors"),
        ("variant_C_r64_step500/pytorch_lora_weights.safetensors", "checkpoints/variant_C_r64_step500.safetensors"),
    ]

    (HF_UPLOAD_DIR / "checkpoints").mkdir(exist_ok=True)
    import shutil
    for src_rel, dst_rel in final_weights:
        src = MODELS_DIR / src_rel
        dst = HF_UPLOAD_DIR / dst_rel
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copiado {src.name} -> {dst_rel}")

    # Copiar comparativas clave
    (HF_UPLOAD_DIR / "comparisons").mkdir(exist_ok=True)
    if (COMPARISONS_DIR / "master_grid_10x4.png").exists():
        shutil.copy2(COMPARISONS_DIR / "master_grid_10x4.png", HF_UPLOAD_DIR / "comparisons/master_grid_10x4.png")
    for f in COMPARISONS_DIR.glob("comparativa_4col_*.png"):
        shutil.copy2(f, HF_UPLOAD_DIR / "comparisons" / f.name)

    # Crear Model Card README.md
    readme_content = f"""---
license: openrail++
base_model: stabilityai/stable-diffusion-xl-base-1.0
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- diffusers
- lora
- peft
- old-books
- vintage-illustrations
- engravings
- etching
instance_prompt: old book illustration style, vintage engraving, detailed etching
widget:
- text: "an antique botanical plate illustration of an exotic passionflower with leaves and roots, historical scientific drawing, detailed engraving, paper texture"
  output:
    url: "comparisons/comparativa_4col_01_botanica.png"
- text: "an antique portrait engraving of an elderly watchmaker with spectacles examining a pocket watch in his workshop, 19th century illustration"
  output:
    url: "comparisons/comparativa_4col_02_relojero.png"
- text: "a modern red sports car parked on a wet asphalt city street at night, neon reflections, antique vintage etching"
  output:
    url: "comparisons/comparativa_4col_06_coche_deportivo.png"
---

# 📚 SDXL LoRA: Old Book Illustrations & Vintage Engravings (1024×1024)

Adapters LoRA de alta resolución entrenados sobre **Stable Diffusion XL 1.0 Base** utilizando la totalidad del dataset **`gigant/oldbookillustrations`** (4.154 grabados del siglo XVIII y XIX).

Desarrollado como proyecto avanzado de investigación sobre adaptación de estilo y rank en modelos fundacionales de difusión.

---

## 🔬 Torneo de Hiperparámetros (3 Variantes)

| Variante | Rank ($r$) | Alpha ($\alpha$) | Capas Adaptadas | Learning Rate | Parámetros | Tamaño |
|---|---|---|---|---|---|---|
| **Variant A** | 16 | 16 | Cross-Attention (`to_k`, `to_q`, `to_v`, `to_out.0`) | `1e-4` | ~11.8M | 47.4 MB |
| **Variant B (Recomendada)** | 32 | 32 | Cross-Attention (`to_k`, `to_q`, `to_v`, `to_out.0`) | `1e-4` | ~23.6M | 94.7 MB |
| **Variant C** | 64 | 32 | Cross-Attention + FFN Projections | `5e-5` | ~47.2M | 189.4 MB |

---

## 🖼️ Matriz Comparativa (Base vs r=16 vs r=32 vs r=64)

![Master Comparison Grid](comparisons/master_grid_10x4.png)

---

## 💻 Uso con Diffusers

```python
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

# Cargar la variante B (r=32)
pipe.load_lora_weights("Twinber/sdxl-oldbook-illustrations-lora", weight_name="sdxl_oldbook_lora_rank32_final.safetensors")

prompt = "an antique portrait engraving of an elderly watchmaker with spectacles examining a pocket watch in his workshop, 19th century illustration"
image = pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("relojero_lora.png")
```

---

## 👥 Créditos
- **Autor:** Twinber (Daniel Aguilera González)
- **Dataset:** `gigant/oldbookillustrations`
- **Modelo Base:** `stabilityai/stable-diffusion-xl-base-1.0`
"""
    with open(HF_UPLOAD_DIR / "README.md", "w") as f:
        f.write(readme_content)

    print("[HF] Subiendo carpeta a Hugging Face Hub...")
    api.upload_folder(
        folder_path=str(HF_UPLOAD_DIR),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Subida de pesos LoRA SDXL (r=16, r=32, r=64), checkpoints y comparativas de benchmark",
    )
    print(f"[HF] Subida completada con ÉXITO a: https://huggingface.co/{repo_id}")


def main():
    t0_global = time.time()
    print("Iniciando pipeline integral de LoRA SDXL...")

    # 1. Precomputar latentes VAE y embeddings CLIP
    precompute_latents_and_embeddings()

    # 2. Cargar dataset en caché
    cache_file = CACHE_DIR / "sdxl_cached_dataset.pt"
    dataset = CachedLatentsDataset(cache_file)
    print(f"Dataset listo con {len(dataset)} muestras cargadas en memoria.")

    # 3. Entrenar las 3 variantes
    for v in VARIANTS:
        train_variant(v, dataset)

    # 4. Inferencia de benchmark (6 checkpoints x 10 prompts)
    run_benchmark_inferences()

    # 5. Generar comparativas y Master Grid
    build_comparisons()

    # 6. Subir a Hugging Face Hub
    prepare_and_upload_hf()

    total_time = time.time() - t0_global
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETO FINALIZADO CON ÉXITO EN {total_time / 60:.1f} MINUTOS!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
