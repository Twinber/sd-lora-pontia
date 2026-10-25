# 🎁 Contenido del Directorio Bonus

Este directorio reúne herramientas interactivas complementarias desarrolladas para explorar, evaluar y comparar de forma visual e inmediata el modelo fundacional `CompVis/stable-diffusion-v1-4` frente al adaptador LoRA `Twinber/sd-v1-4-oldbook-illustrations-lora`.

---

## 1. 📓 `probador_interactivo_lora_colab.ipynb` (Pruebas Ágiles en Google Colab con GPU)

Un cuaderno ligero diseñado específicamente para ejecutarse en la nube con aceleración por hardware (NVIDIA Tesla T4 en Google Colab):
* **Inferencia ultra-rápida:** Genera las dos imágenes (Antes vs. Después) lado a lado en apenas **2 a 3 segundos** en GPU.
* **Descarga directa desde Hugging Face Hub:** Descarga únicamente el adaptador LoRA ligero (6.12 MB) sin necesidad de descargar pesos redundantes.
* **Panel de Widgets Interactivos (`ipywidgets`):** Permite introducir prompts libres, ajustar la semilla (`seed`), los pasos de inferencia y la intensidad del LoRA ($0.0 \to 1.5$) mediante controles visuales directamente en el cuaderno.
* **Blindaje de entorno:** Incorpora resolución automática de dependencias para evitar conflictos con librerías de cuantización en Colab.

---

## 2. 💻 `app.py` (Aplicación Local Interactiva en Streamlit)

Una aplicación web completa para ejecutar en local:
* **Arquitectura de Memoria Eficiente (Single Pipeline):** Conmuta dinámicamente el adaptador con `pipe.set_adapters(["oldbook_illustrations"], adapter_weights=[w])` manteniendo un único modelo en RAM (~3.2 GB) sin duplicar memoria.
* **Evaluación en tiempo real:** Evalúa el modelo base puro (`adapter_weight = 0.0`) y el modelo LoRA (`adapter_weight = 1.0` o personalizado) bajo la misma semilla.
* **Exportación y descarga:** Genera un panel panorámico comparativo con cabeceras y botón de descarga directa en PNG.

### Cómo ejecutar la aplicación Streamlit en local:
```bash
streamlit run bonus/app.py
```
O usando el gestor `uv`:
```bash
uv run --with streamlit --with diffusers --with transformers --with torch --with torchvision --with peft --with safetensors --with accelerate --with pillow streamlit run bonus/app.py
```
