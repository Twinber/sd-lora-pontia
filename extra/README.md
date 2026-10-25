# 🚀 Experimento Avanzado: Fine-Tuning de Alta Fidelidad en SDXL (1024×1024)

Este directorio reúne los experimentos avanzados de adaptación LoRA sin las restricciones de cómputo de Google Colab ni las limitaciones de resolución de modelos fundacionales antiguos como SD v1.4.

---

## 🎯 Objetivos del Experimento

1. **Resolución Nativa 1024×1024:** Aprovechar los escaneos de alta resolución (`1600px`) del dataset `gigant/oldbookillustrations` sin recortar detalles microscópicos.
2. **Modelo Base:** `stabilityai/stable-diffusion-xl-base-1.0` (arquitectura de 2.6B parámetros con doble codificador de texto CLIP ViT-L y OpenCLIP ViT-bigG).
3. **Banco de Pruebas Dual (10 Prompts de Evaluación, Semilla fija 42):**
   * **Grupo 1 (Temática Histórica / Grabado):** Botánica, Relojero, Catedral, Locomotora, Barco.
   * **Grupo 2 (Temática Genérica / Contemporánea):** Coche deportivo, Cafetería moderna, Perro en campo, Rascacielos de cristal, Astronauta en Marte.
4. **Torneo de Hiperparámetros en RunPod:** Comparar 3 variantes de LoRA (Rank 16 vs. Rank 32 vs. Rank 64) para encontrar el equilibrio exacto entre transferencia de estilo y fidelidad semántica.

---

## 📸 Banco de Imágenes del Modelo Base (Sin LoRA — SDXL 1024px)

| Nº | Archivo Base | Temática | Prompt de Prueba (Semilla 42) | Enfoque de Validación |
|---|---|---|---|---|
| **01** | `base_01_botanica.png` | Botánica Exótica | *"an antique botanical plate illustration of an exotic passionflower with leaves and roots, historical scientific drawing, detailed engraving, paper texture"* | Grabado científico botánico en plancha de cobre |
| **02** | `base_02_relojero.png` | Retrato Oficio | *"an antique portrait engraving of an elderly watchmaker with spectacles examining a pocket watch in his workshop, 19th century illustration"* | Expresión humana, arrugas y micro-engranajes |
| **03** | `base_03_catedral.png` | Arquitectura Gótica | *"a medieval gothic cathedral on a rocky cliff overlooking a stormy ocean, 19th century vintage landscape etching"* | Piedra gótica, mar embravecido y cielo achurado |
| **04** | `base_04_locomotora.png` | Era Industrial | *"a vintage 19th century steam locomotive crossing a stone viaduct over a deep mountain river, technical etching illustration"* | Metal entintado, humo denso y viaducto de sillería |
| **05** | `base_05_barco.png` | Náutica y Mar | *"a tall wooden sailing ship battling gigantic stormy waves in the ocean, dark sky, antique vintage maritime engraving"* | Velamen, cordajes a plumilla y olas estilizadas |
| **06** | `base_06_coche_deportivo.png` | Automóvil Moderno | *"a modern red sports car parked on a wet asphalt city street at night, neon reflections"* | Transferencia extrema sobre fotorrealismo nocturno |
| **07** | `base_07_cafeteria.png` | Retrato Cotidiano | *"a young woman sitting in a modern coffee shop drinking a cup of coffee, looking out a large window"* | Transformación de foto informal contemporánea |
| **08** | `base_08_perro_colina.png` | Naturaleza y Animal | *"a golden retriever dog sitting happily in a green grass field, bright sunny day, clear blue sky"* | Pelaje y hierba bajo estilo litográfico |
| **09** | `base_09_rascacielos.png` | Metrópolis Cristal | *"a futuristic glass skyscraper towering into the sky, urban metropolis with clouds"* | Arquitectura moderna de cristal y acero a grabado |
| **10** | `base_10_astronauta_marte.png` | Ciencia Ficción | *"an astronaut walking on the red rocky surface of planet Mars, distant mountains, Earth in the sky"* | Concepto espacial anacrónico estilo Julio Verne |

---

## 🏆 Resultados del Torneo y Modelos Publicados en Hugging Face

Repositorio oficial en el Hub: **[`Twinber/sdxl-oldbook-illustrations-lora`](https://huggingface.co/Twinber/sdxl-oldbook-illustrations-lora)**

### 1. Pesos LoRA Entrenados (Safetensors)

| Variante | Rango ($r$) | Alpha ($\alpha$) | Capas Adaptadas | Parámetros | Tamaño | Archivo Local |
|---|---|---|---|---|---|---|
| **Variante A** | 16 | 16 | Cross-Attention (`to_k`, `to_q`, `to_v`, `to_out.0`) | 23.2M | 44.5 MB | [`models/sdxl_oldbook_lora_rank16_final.safetensors`](file:///home/twinber/Projects/Master-IA/11-Generacion-Imagen/extra/models/sdxl_oldbook_lora_rank16_final.safetensors) |
| **Variante B (Recomendada)** | 32 | 32 | Cross-Attention (`to_k`, `to_q`, `to_v`, `to_out.0`) | 46.4M | 88.8 MB | [`models/sdxl_oldbook_lora_rank32_final.safetensors`](file:///home/twinber/Projects/Master-IA/11-Generacion-Imagen/extra/models/sdxl_oldbook_lora_rank32_final.safetensors) |
| **Variante C** | 64 | 32 | Cross-Attention + FFN (`ff.net.0.proj`, `ff.net.2`) | 167.3M | 319.6 MB | [`models/sdxl_oldbook_lora_rank64_final.safetensors`](file:///home/twinber/Projects/Master-IA/11-Generacion-Imagen/extra/models/sdxl_oldbook_lora_rank64_final.safetensors) |

### 2. Galería de Comparativas

- **Matriz Maestra Global 10×4:** [`comparisons/master_grid_10x4.png`](file:///home/twinber/Projects/Master-IA/11-Generacion-Imagen/extra/comparisons/master_grid_10x4.png) (10 prompts en horizontal, 4 columnas en vertical: Base vs. r=16 vs. r=32 vs. r=64).
- **Tiras comparativas 4 columnas (por prompt):** Disponibles en [`comparisons/comparativa_4col_*.png`](file:///home/twinber/Projects/Master-IA/11-Generacion-Imagen/extra/comparisons/).
- **Tiras de progresión temporal (Base vs Paso 500 vs Paso 1.000):** Disponibles en [`comparisons/progresion_epocas_*.png`](file:///home/twinber/Projects/Master-IA/11-Generacion-Imagen/extra/comparisons/).

### 3. Cuaderno Probador Interactivo
- [`probador_sdxl_loras_colab.ipynb`](file:///home/twinber/Projects/Master-IA/11-Generacion-Imagen/extra/probador_sdxl_loras_colab.ipynb): Cuaderno listo con selector de adaptadores e interfaz gráfica en Google Colab.

