---
license: creativeml-openrail-m
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
- oldbookillustrations
- art
- 19th-century
base_model: CompVis/stable-diffusion-v1-4
instance_prompt: antique illustration
widget:
- text: "Anciano pescando en una barca en mitad de un lago"
  output:
    url: "anciano-pescando-barca.png"
- text: "Mujer bailando sobre una mesa"
  output:
    url: "mujer-bailando.png"
- text: "a man walking his dog by the shores of a lake, tranquil landscape"
  output:
    url: "comparativa_hombre_perro.png"
- text: "an antique illustration of a botanic garden with exotic flowers and trees"
  output:
    url: "comparativa_antes_despues.png"
---

# Stable Diffusion v1.4 — Old Book Illustrations LoRA

Modelo LoRA entrenado sobre `CompVis/stable-diffusion-v1-4` utilizando el dataset `gigant/oldbookillustrations` para transferir el estilo de ilustraciones, aguafuertes y grabados históricos en plancha del siglo XIX.

- **Autor:** Daniel Aguilera González ([@Twinber](https://huggingface.co/Twinber))
- **Modelo Base:** `CompVis/stable-diffusion-v1-4`
- **Dataset:** [gigant/oldbookillustrations](https://huggingface.co/datasets/gigant/oldbookillustrations)
- **Tipo de adaptación:** LoRA en proyecciones de Cross-Attention (`to_k`, `to_q`, `to_v`, `to_out.0`) de la UNet.
- **Rango ($r$):** 8, **Alpha ($\alpha$):** 8
- **Parámetros entrenables:** 1.594.368 (0,185% del total de la UNet)

---

## 🎨 Demostraciones Visuales Comparativas

### 1. Anciano pescando en barca (Sesgo fotográfico vs. Cuento de época)
> *"Anciano pescando en una barca en mitad de un lago"* (Seed: 303, Steps: 20)

![Anciano pescando](anciano-pescando-barca.png)

### 2. Mujer bailando sobre una mesa (Foto moderna vs. Ilustración Art Nouveau)
> *"Mujer bailando sobre una mesa"* (Seed: 101, Steps: 20)

![Mujer bailando](mujer-bailando.png)

### 3. Escena cotidiana: Hombre y perro en el lago
> *"a man walking his dog by the shores of a lake, tranquil landscape"* (Seed: 42)

![Hombre y perro](comparativa_hombre_perro.png)

### 4. Escena botánica histórica
> *"an antique illustration of a botanic garden with exotic flowers and trees"* (Seed: 42)

![Comparativa Botánica](comparativa_antes_despues.png)

---

## 🚀 Uso en Python con Diffusers

### Carga ligera de pesos LoRA (Solo 6.1 MB)

```python
import torch
from diffusers import StableDiffusionPipeline

# 1. Cargar pipeline base
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
).to("cuda")

# 2. Cargar adaptador LoRA directamente desde este repositorio
pipe.load_lora_weights("Twinber/sd-v1-4-oldbook-illustrations-lora", adapter_name="oldbook_illustrations")

# 3. Generar
prompt = "Anciano pescando en una barca en mitad de un lago"
generator = torch.Generator(device="cuda").manual_seed(303)
image = pipe(prompt, num_inference_steps=20, generator=generator).images[0]
image.save("pescador_oldbook.png")
```

---

## 💻 Aplicación Interactiva y Cuaderno Colab

- **Cuaderno Colab:** Incluido `probador_interactivo_lora_colab.ipynb` en `bonus/` para inferencia interactiva ultra-rápida en GPU (T4/L4/A100) en ~2 segundos por imagen.
- **Bonus Streamlit:** Aplicación local en `bonus/app.py` que permite conmutar en tiempo real el peso del LoRA ($0.0 \to 1.5$), pasos y semilla con un solo modelo en memoria RAM.
