"""
================================================================================
MÓDULO 11: GENERACIÓN DE IMAGEN — MÁSTER IA PONTIA
BONUS: APLICACIÓN LOCAL INTERACTIVA STREAMLIT
Comparador en tiempo real: Stable Diffusion v1.4 vs. LoRA Old Book Illustrations
Alumno: Daniel Aguilera González (@Twinber)
================================================================================
"""

import os
import time
import torch
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import LoraConfig
from safetensors.torch import load_file

# Configuración de página
st.set_page_config(
    page_title="Old Book LoRA Explorer — SD v1.4",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título y cabecera
st.title("📖 Comparador Interactivo: Stable Diffusion v1.4 vs. LoRA")
st.markdown("""
Esta aplicación permite generar y comparar **en tiempo real y en local** el comportamiento del modelo fundacional
**`CompVis/stable-diffusion-v1-4`** frente al modelo adaptado con **LoRA** sobre el dataset histórico de grabados
[`gigant/oldbookillustrations`](https://huggingface.co/datasets/gigant/oldbookillustrations).
""")

# Carga en caché del modelo
@st.cache_resource(show_spinner="⏳ Cargando Stable Diffusion v1.4 y adaptadores LoRA en memoria...")
def load_pipeline():
    torch.set_num_threads(os.cpu_count() or 8)
    model_id = "CompVis/stable-diffusion-v1-4"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )
    # Scheduler de alta eficiencia (DPM-Solver++ reduce los pasos necesarios a 15-20)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    adapter_name = "oldbook_illustrations"
    
    # Ruta de los pesos extraídos (busca en carpeta local, en padre o descarga de HF Hub)
    weights_paths = [
        os.path.join(os.path.dirname(__file__), "models", "pytorch_lora_weights.safetensors"),
        os.path.join(os.path.dirname(__file__), "..", "models", "pytorch_lora_weights.safetensors"),
    ]
    loaded = False
    for wp in weights_paths:
        if os.path.exists(wp):
            pipe.load_lora_weights(wp, adapter_name=adapter_name)
            loaded = True
            break
            
    if not loaded:
        try:
            pipe.load_lora_weights("Twinber/sd-v1-4-oldbook-illustrations-lora", adapter_name=adapter_name)
        except Exception:
            pass
    
    return pipe, adapter_name

pipe, adapter_name = load_pipeline()
st.sidebar.success("✅ SD v1.4 + LoRA cargados en RAM")

# Barra lateral con hiperparámetros
with st.sidebar:
    st.header("⚙️ Ajustes de Generación")
    
    steps = st.slider("Pasos de inferencia (Steps)", min_value=10, max_value=30, value=15, step=1,
                      help="15 pasos con DPM-Solver++ ofrecen calidad profesional en CPU en ~1 minuto.")
    
    guidance_scale = st.slider("Escala de Guiado (CFG)", min_value=1.0, max_value=15.0, value=7.5, step=0.5,
                               help="Fidelidad al prompt textual.")
    
    lora_weight = st.slider("Intensidad del LoRA (Weight)", min_value=0.0, max_value=1.5, value=1.0, step=0.1,
                            help="1.0 es el estilo completo; 0.0 desactiva LoRA.")
    
    if "seed_val" not in st.session_state:
        st.session_state["seed_val"] = 42

    col_seed1, col_seed2 = st.columns([2, 1])
    with col_seed2:
        if st.button("🎲 Random"):
            st.session_state["seed_val"] = int(torch.randint(0, 99999999, (1,)).item())
    with col_seed1:
        seed = st.number_input("Semilla (Seed)", min_value=0, max_value=99999999, key="seed_val")
    
    st.divider()
    st.markdown(r"""
    ### 💡 ¿Cómo funciona?
    El pipeline mantiene un único modelo en memoria RAM:
    * **Antes:** Se evalúa con `adapter_weight = 0.0` (matriz $\Delta W$ multiplicada por cero = modelo base puro).
    * **Después:** Se evalúa con `adapter_weight = 1.0` (inyección activa de los pesos aprendidos de grabado).
    """)

# Ejemplos rápidos
st.subheader("✍️ Introduce tu Prompt")

if "prompt_text" not in st.session_state:
    st.session_state["prompt_text"] = "a man walking his dog by the shores of a lake, tranquil landscape"

def set_prompt(text):
    st.session_state["prompt_text"] = text

preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
if preset_col1.button("🐶 Hombre con perro en lago"):
    set_prompt("a man walking his dog by the shores of a lake, tranquil landscape")
if preset_col2.button("🏰 Castillo gótico en niebla"):
    set_prompt("a medieval gothic castle on a misty cliff, antique vintage engraving")
if preset_col3.button("⛵ Barco velero en tormenta"):
    set_prompt("a 19th century tall sailing ship in a rough stormy ocean")
if preset_col4.button("🚂 Locomotora de vapor"):
    set_prompt("a vintage steam locomotive train crossing an arched bridge in the mountains")

# Caja de texto de prompt
prompt = st.text_area("Prompt en inglés (para CLIP)", key="prompt_text", height=80)

negative_prompt = st.text_input(
    "Negative Prompt (opcional)",
    value="blurry, low quality, distorted, deformed, text, watermark, photo, photorealistic, 3d render"
)

# Botón de generación
generate_btn = st.button("🚀 Generar Comparativa (Antes vs Después)", type="primary", use_container_width=True)

if generate_btn and prompt.strip():
    progress_container = st.container()
    
    with progress_container:
        st.info("🧠 Procesando con PyTorch en CPU...")
        
        # 1. Generar ANTES (Base: LoRA weight = 0.0)
        pipe.set_adapters([adapter_name], adapter_weights=[0.0])
        gen1 = torch.Generator().manual_seed(int(seed))
        
        start_t1 = time.time()
        with st.spinner("1/2 Generando imagen con Modelo Base SD v1.4..."):
            img_before = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=gen1,
            ).images[0]
        time_before = time.time() - start_t1
        
        # 2. Generar DESPUÉS (LoRA: LoRA weight = lora_weight)
        pipe.set_adapters([adapter_name], adapter_weights=[float(lora_weight)])
        gen2 = torch.Generator().manual_seed(int(seed))
        
        start_t2 = time.time()
        with st.spinner(f"2/2 Generando imagen con LoRA Old Book Illustrations (peso={lora_weight})..."):
            img_after = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=gen2,
            ).images[0]
        time_after = time.time() - start_t2

    # Mostrar resultados en dos columnas
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🏛️ ANTES: Modelo Base SD v1.4")
        st.caption(f"Semilla: {seed} | Tiempo: {time_before:.1f}s | LoRA weight: 0.0")
        st.image(img_before, use_container_width=True)
    
    with col_right:
        st.subheader(f"📖 DESPUÉS: Con LoRA (Old Book)")
        st.caption(f"Semilla: {seed} | Tiempo: {time_after:.1f}s | LoRA weight: {lora_weight}")
        st.image(img_after, use_container_width=True)
    
    # Crear panel comparativo panorámico
    w, h = img_before.size
    header_h = 45
    comparison = Image.new("RGB", (w * 2, h + header_h), (250, 250, 250))
    comparison.paste(img_before, (0, header_h))
    comparison.paste(img_after, (w, header_h))
    
    draw = ImageDraw.Draw(comparison)
    draw.line([(w, 0), (w, h + header_h)], fill=(180, 180, 180), width=2)
    draw.line([(0, header_h), (w * 2, header_h)], fill=(180, 180, 180), width=1)
    draw.text((25, 14), "MODELO BASE SD v1-4 (Antes)", fill=(40, 40, 40))
    draw.text((w + 25, 14), f"CON LORA OLDBOOK (Despues, peso={lora_weight})", fill=(10, 80, 160))
    
    # Guardar temporal
    comp_path = "/tmp/comparativa_actual.png"
    comparison.save(comp_path)
    
    st.divider()
    st.subheader("🖼️ Panel Comparativo Lado a Lado")
    st.image(comparison, use_container_width=True)
    
    with open(comp_path, "rb") as file:
        st.download_button(
            label="💾 Descargar Comparativa en PNG",
            data=file,
            file_name="comparativa_lora_oldbook.png",
            mime="image/png",
            use_container_width=True,
        )
