# 📦 Pesos LoRA SDXL 1.0 (Safetensors)

Debido al tamaño de los adaptadores de alta resolución entrenados sobre Stable Diffusion XL 1.0 (hasta 320 MB por archivo), los pesos están alojados públicamente en Hugging Face Hub:

👉 **[https://huggingface.co/Twinber/sdxl-oldbook-illustrations-lora](https://huggingface.co/Twinber/sdxl-oldbook-illustrations-lora)**

### Modelos Disponibles en el Hub:
1. `sdxl_oldbook_lora_rank16_final.safetensors` (44,5 MB - Variante A, $r=16, \alpha=16$)
2. `sdxl_oldbook_lora_rank32_final.safetensors` (88,8 MB - **Variante B Recomendada**, $r=32, \alpha=32$)
3. `sdxl_oldbook_lora_rank64_final.safetensors` (319,6 MB - Variante C con FFN, $r=64, \alpha=32$)
4. Checkpoints a 500 pasos en la subcarpeta `checkpoints/`.

### Uso Directo con Diffusers:
```python
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True
)

# Cargar la variante B (r=32)
pipe.load_lora_weights(
    "Twinber/sdxl-oldbook-illustrations-lora",
    weight_name="sdxl_oldbook_lora_rank32_final.safetensors",
)

prompt = "an antique portrait engraving of an elderly watchmaker with spectacles examining a pocket watch in his workshop, 19th century illustration"
image = pipe(
    prompt=prompt, num_inference_steps=30, guidance_scale=7.5, width=1024, height=1024
).images[0]
image.save("relojero.png")
```
