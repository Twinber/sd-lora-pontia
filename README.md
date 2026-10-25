# Módulo 11: Modelos de Generación de Imagen
## Adaptación Eficiente con LoRA: De Stable Diffusion v1.4 a SDXL 1.0 (1024×1024)

**Máster en Inteligencia Artificial e Inteligencia Artificial Generativa — PontIA**  
* **Autor / Estudiante:** Daniel Aguilera González (Twinber)  
* **Perfil Hugging Face:** [@Twinber](https://huggingface.co/Twinber)  
* **Repositorio GitHub:** [Twinber/sd-lora-pontia](https://github.com/Twinber/sd-lora-pontia)  

---

### 🌟 Accesos Directos a los Entregables Principales

| Recurso | Plataforma / Formato | Descripción | Enlace Directo |
|---|---|---|---|
| **Memoria de Presentación** | 📄 PDF (14 páginas A4) | Documento maestro con teoría, tablas, comparativas y enlaces activos | [`Memoria_Presentacion_Proyecto_LoRA_OldBook.pdf`](Memoria_Presentacion_Proyecto_LoRA_OldBook.pdf) |
| **Cuaderno Oficial del Máster** | 📓 Jupyter Notebook | Fine-tuning reproducible de SD v1.4 en GPU Tesla T4 (FP16) | [`entrega_finetuning_lora_oldbook.ipynb`](entrega_finetuning_lora_oldbook.ipynb) |
| **Modelo Oficial SD v1.4** | 🤗 Hugging Face Hub | Adaptador LoRA ($r=8$) publicado con Model Card oficial | [Twinber/sd-v1-4-oldbook-illustrations-lora](https://huggingface.co/Twinber/sd-v1-4-oldbook-illustrations-lora) |
| **Modelo Avanzado SDXL 1.0** | 🤗 Hugging Face Hub | 3 variantes LoRA ($r=16, 32, 64$), 6 checkpoints y Master Grid 4K | [Twinber/sdxl-oldbook-illustrations-lora](https://huggingface.co/Twinber/sdxl-oldbook-illustrations-lora) |
| **Herramientas Bonus** | 🎁 Colab & Streamlit | App interactiva local y probador ágil con widgets en GPU | [`bonus/`](bonus/) |
| **Investigación Extra SDXL** | 🚀 RunPod Cloud (A40) | Torneo de 3 variantes LoRA a 1024px con 4.154 imágenes | [`extra/`](extra/) |

---

## 1. Práctica Básica: Requisitos Oficiales del Máster PontIA

### 1.1. Objetivo y Planteamiento
El objetivo oficial del Módulo 11 es realizar un **fine-tuning completo empleando la técnica de Low-Rank Adaptation (LoRA)** sobre el modelo fundacional de difusión latente **`CompVis/stable-diffusion-v1-4`**, adaptándolo al estilo de grabados e ilustraciones de libros del siglo XIX mediante el dataset **`gigant/oldbookillustrations`**.

A diferencia del caso visto en clase (estilo Pokémon sobre imágenes cuadradas y prompts sintéticos), este dataset presenta retos técnicos de producción real:
* **Relaciones de aspecto heterogéneas:** Ilustraciones panorámicas y verticales de alta resolución (columna `1600px`). Se aplicó una transformación de `Resize(512)` sobre el lado menor seguida de `CenterCrop(512)` para evitar distorsiones anamórficas en los trazos de grabado.
* **Descripciones contextuales ricas:** Extracción y curación de leyendas históricas procedentes de la columna `info_alt`.
* **Complejidad estilística:** Captura de tramas de achurado (*cross-hatching*), técnicas de aguafuerte en plancha de cobre/madera y tonalidades de papel pergamino envejecido.

### 1.2. Entregables Oficiales de la Práctica Básica
1. **Cuaderno Jupyter Maestro:** [`entrega_finetuning_lora_oldbook.ipynb`](entrega_finetuning_lora_oldbook.ipynb), ejecutable de principio a fin, documentado y optimizado con FP16.
2. **Enlace a Hugging Face Hub en archivo de texto:** [`huggingface_repo_url.txt`](huggingface_repo_url.txt), apuntando a [Twinber/sd-v1-4-oldbook-illustrations-lora](https://huggingface.co/Twinber/sd-v1-4-oldbook-illustrations-lora).
3. **Galería de Comparativas Antes vs. Después:** Alojadas en la carpeta [`renders/`](renders/), generadas fijando exactamente el mismo prompt y la misma semilla aleatoria de ruido inicial (`seed = 42`):

#### Comparativa 1: Escena Botánica Histórica (Prueba Principal Oficial)
> **Prompt:** *"an antique illustration of a botanic garden with exotic flowers and trees"* (`seed = 42`, 20 pasos de inferencia)

![Comparativa Antes vs Después](renders/comparativa_antes_despues.png)

#### Comparativa 2: Generalización sobre Escena Cotidiana Contemporánea
Demuestra la transferencia de estilo sobre conceptos ajenos a la botánica histórica:
> **Prompt:** *"a man walking his dog by the shores of a lake, tranquil landscape"* (`seed = 42`)

![Comparativa Hombre y Perro](renders/comparativa_hombre_perro.png)

#### Comparativa 3: Anciano pescando en barca (Sesgo fotográfico vs. Cuento tradicional)
El modelo base genera una lancha a motor moderna y gafas de sol actuales; el modelo adaptado con LoRA genera una estampa litográfica de cuento tradicional con ropajes de época:
> **Prompt:** *"Anciano pescando en una barca en mitad de un lago"* (`seed = 303`)

![Anciano Pescando en Barca](renders/anciano-pescando-barca.png)

#### Comparativa 4: Mujer bailando sobre una mesa (Estudio moderno vs. Estampa victoriana)
El modelo base sesga hacia fotografía de estudio de danza contemporánea (mallas y parquet); el LoRA transforma la escena en una estampa victoriana con vestido de época, volantes y vidriera emplomada:
> **Prompt:** *"Mujer bailando sobre una mesa"* (`seed = 101`)

![Mujer Bailando](renders/mujer-bailando.png)

---

### 1.3. Fundamentos Teóricos y Mecanismo LoRA
Stable Diffusion opera en el espacio latente comprimido por el VAE ($\times 8$):

$$\mathbf{z} = \mathcal{E}(\mathbf{x}) \cdot 0.18215, \quad \mathbf{z} \in \mathbb{R}^{4 \times 64 \times 64}$$

Durante el fine-tuning, el VAE y el codificador de texto (CLIP ViT-L/14) permanecen **congelados**. En lugar de actualizar los 861 millones de parámetros de la UNet 2D, LoRA congela las matrices originales de pesos $W_0 \in \mathbb{R}^{d \times k}$ y descompone la actualización en dos matrices de bajo rango $A \in \mathbb{R}^{r \times k}$ y $B \in \mathbb{R}^{d \times r}$:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$

Las matrices de adaptación se inyectan en las proyecciones lineales de las capas de **Cross-Attention**:
* `to_k` (Keys) y `to_q` (Queries): Alineamiento espacial y condicionamiento semántico.
* `to_v` (Values) y `to_out.0` (Output Projection): Proyección de los rasgos de estilo hacia los bloques residuales.

### 1.4. Configuración Experimental e Hiperparámetros
* **Modelo Base:** `CompVis/stable-diffusion-v1-4`
* **Hardware:** GPU NVIDIA Tesla T4 (16 GB) en Google Colab
* **Rango LoRA ($r$):** `8` | **Factor de escala ($\alpha$):** `8` ($\alpha / r = 1.0$)
* **Parámetros Entrenables:** **1.594.368** de 861.115.332 (**0,185%** del total de la UNet)
* **Precisión:** `mixed_precision="fp16"` (redujo el tiempo por época de 47 min a solo 3 min en Tensor Cores)
* **Optimizador:** AdamW ($\text{LR} = 1\times 10^{-4}$, batch size = 4, 3 épocas / 750 pasos)

### 1.5. Inferencia con Diffusers desde Hugging Face
```python
import torch
from diffusers import StableDiffusionPipeline

# Cargar pipeline base en media precisión
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
).to("cuda")

# Cargar adaptador LoRA directamente desde Hugging Face Hub
pipe.load_lora_weights("Twinber/sd-v1-4-oldbook-illustrations-lora")

# Generar ilustración
prompt = "an antique vintage illustration engraving of an old lighthouse in a stormy sea"
image = pipe(prompt, num_inference_steps=25, generator=torch.Generator("cuda").manual_seed(42)).images[0]
image.save("faro_oldbook.png")
```

---

## 2. Entregables Bonus: Herramientas Interactivas y de Despliegue

La carpeta [`bonus/`](bonus/) contiene herramientas diseñadas para facilitar la interacción, evaluación y demostración del modelo tanto en local como en la nube:

1. **Cuaderno Interactivo en Google Colab:** [`bonus/probador_interactivo_lora_colab.ipynb`](bonus/probador_interactivo_lora_colab.ipynb)
   - Diseñado para cargar directamente los pesos publicados desde Hugging Face Hub en entornos Colab gratuitos (GPU T4).
   - Generación rápida (~2,1 s por imagen) con widgets interactivos de `ipywidgets` y comparador visual integrado.

2. **Aplicación Web Local en Streamlit:** [`bonus/app.py`](bonus/app.py)
   - Interfaz gráfica moderna para inferencia local en tiempo real.
   - Permite ingresar cualquier prompt de texto, controlar la escala de influencia del LoRA mediante slider ($0.0 \to 1.5$), seleccionar número de pasos de denoising y fijar semillas aleatorias.
   - Muestra la comparativa lado a lado "Modelo Base vs. Modelo con LoRA" manteniendo un único pipeline en memoria.
   - Ejecución:
     ```bash
     streamlit run bonus/app.py
     ```

3. **Documentación del Módulo Bonus:** [`bonus/README.md`](bonus/README.md) con los detalles técnicos de arquitectura e instrucciones de despliegue.

---

## 3. Proyecto Avanzado Extra: Torneo SDXL 1.0 a 1024×1024 en RunPod

Para llevar la investigación al estado del arte de la generación de imagen, se diseñó y ejecutó un **experimento a escala industrial** documentado íntegramente en la carpeta [`extra/`](extra/):

* **Repositorio Hugging Face SDXL:** [Twinber/sdxl-oldbook-illustrations-lora](https://huggingface.co/Twinber/sdxl-oldbook-illustrations-lora)
* **Memoria Técnica del Torneo SDXL:** [`extra/README.md`](extra/README.md)

### 3.1. Claves de Ingeniería y Arquitectura
* **Modelo Fundacional:** `stabilityai/stable-diffusion-xl-base-1.0` (2.600 millones de parámetros, doble text-encoder OpenCLIP ViT-bigG/14 + CLIP ViT-L/14).
* **Resolución Nativa 1024×1024:** Permite capturar la microporosidad del papel histórico y la densidad de líneas calcográficas sin degradación por interpolación.
* **Escala del Dataset:** Entrenamiento sobre la totalidad de las **4.154 imágenes** del dataset `gigant/oldbookillustrations` a 1600px.
* **Infraestructura Cloud:** Instancia NVIDIA A40 (48 GB VRAM Ampere) en RunPod Secure Cloud.
* **Estabilidad Numérica en bfloat16:** El VAE de SDXL produce desbordamientos (`NaN`) al decodificar en FP16; se implementó el pipeline integral en `torch.bfloat16`, garantizando cero artefactos numéricos.
* **Aceleración por Precomputación Desacoplada:** Precomputar latentes y embeddings duales redujo el tiempo por época de ~4 horas a solo ~30 minutos.

### 3.2. Torneo de Variantes de Hiperparámetros
Se entrenaron y compararon 3 arquitecturas LoRA diferenciadas:

| Variante | Módulos Adaptados | Rango ($r$) | Factor ($\alpha$) | Tamaño Safetensors | Tiempo / Paso | Conclusión de Rendimiento |
|---|---|---|---|---|---|---|
| **Variante A** | Cross-Attention (`to_k, to_q, to_v, to_out.0`) | `16` | `16` | 44,5 MB | 3,61 s | Adaptación ligera, preserva alta libertad semántica. |
| **Variante B (Óptima)** | Cross-Attention (`to_k, to_q, to_v, to_out.0`) | `32` | `32` | 88,8 MB | 3,63 s | **El punto dulce:** captura exquisita de líneas finas sin sobrecoste computacional (+0,02s). |
| **Variante C** | Cross-Attention + Proyecciones FFN (`ff.net.0, ff.net.2`) | `64` | `64` | 319,6 MB | 5,39 s | Máxima densidad de tinta pero +49% tiempo de entrenamiento y archivos 3,6x más pesados. |

### 3.3. Banco de Pruebas Benchmark y Resultados 4K
Se evaluaron 6 checkpoints (500 y 1.000 pasos por variante) sobre un banco de pruebas de **10 prompts exhaustivos** (5 históricos/vintage y 5 contemporáneos anacrónicos: rascacielos cyberpunk, astronauta en Marte, coche deportivo, etc.):

* **Matriz Maestra 10×4 (40 ilustraciones en alta fidelidad):** [`extra/comparisons/master_grid_10x4.png`](extra/comparisons/master_grid_10x4.png)
* **20 Tiras Comparativas Individuales a Resolución 4K (4096px):** Disponibles en [`extra/comparisons/`](extra/comparisons/).
* **Probador Interactivo SDXL en Google Colab:** [`extra/probador_sdxl_loras_colab.ipynb`](extra/probador_sdxl_loras_colab.ipynb).
* **Pipeline Maestro Automatizado:** [`extra/scripts/run_full_experiment.py`](extra/scripts/run_full_experiment.py) y script de aprovisionamiento [`extra/scripts/setup_pod.sh`](extra/scripts/setup_pod.sh).

---

## 4. Documento Maestro: Memoria Técnica de Presentación en PDF

El entregable principal del proyecto es el documento técnico en formato PDF editorial:
📄 **[`Memoria_Presentacion_Proyecto_LoRA_OldBook.pdf`](Memoria_Presentacion_Proyecto_LoRA_OldBook.pdf)**

* **Extensión y Maquetación:** 14 páginas A4 completas generadas mediante WeasyPrint con motor tipográfico moderno, tablas comparativas, cajas de alertas técnicas, formulación en LaTeX y figuras de alta resolución.
* **Interactividad y Autosuficiencia:** La portada (página 1) y la sección 9 incorporan hipervínculos activos directos al repositorio de GitHub y a los modelos publicados en Hugging Face Hub, permitiendo a cualquier evaluador auditar el código y probar los modelos con un solo clic.
* **Auditoría Financiera y de Costes:** Desglose riguroso del cómputo en RunPod (coste total de la GPU: 2,86 $ USD; saldo final verificado en 8,73 $ USD; pods eliminados por completo y coste horario residual en 0,000 $/h).
* **Script Generador del Documento:** [`extra/scripts/generate_pdf.py`](extra/scripts/generate_pdf.py).

---

## 5. Estructura Organizada del Repositorio

El repositorio está organizado en carpetas lógicas para garantizar máxima claridad técnica:

```
sd-lora-pontia/
├── Memoria_Presentacion_Proyecto_LoRA_OldBook.pdf  # 📄 Memoria técnica oficial en PDF (14 págs, interactiva)
├── entrega_finetuning_lora_oldbook.ipynb           # 📓 Cuaderno Jupyter maestro oficial ejecutable
├── huggingface_repo_url.txt                         # 🔗 Enlace oficial al modelo publicado en Hugging Face Hub
├── README.md                                       # 📖 Documentación general y guía técnica
├── .gitignore                                      # 🛡️ Filtro de binarios pesados y temporales
│
├── renders/                                        # 🖼️ Resultados visuales e inferencias de SD v1.4 + LoRA
│   ├── comparativa_antes_despues.png               # Panel comparativo botánico (seed 42)
│   ├── comparativa_hombre_perro.png                # Panel de generalización escena cotidiana (seed 42)
│   ├── anciano-pescando-barca.png                  # Comparativa: anciano pescando en barca (seed 303)
│   ├── mujer-bailando.png                          # Comparativa: mujer bailando sobre mesa (seed 101)
│   ├── imagen_antes_finetuning.png                 # Inferencia previa base
│   └── imagen_despues_finetuning.png               # Inferencia final con LoRA
│
├── bonus/                                          # 🎁 Herramientas interactivas adicionales oficiales
│   ├── probador_interactivo_lora_colab.ipynb       # Cuaderno Colab con interfaz interactiva
│   ├── app.py                                      # Aplicación interactiva en Streamlit para inferencia local
│   └── README.md                                   # Guía de despliegue del bonus
│
├── extra/                                          # 🚀 Proyecto de Investigación Avanzada: SDXL 1024×1024 (RunPod A40)
│   ├── README.md                                   # Memoria técnica del torneo de hiperparámetros
│   ├── probador_sdxl_loras_colab.ipynb             # Probador Colab para los 3 LoRAs SDXL
│   ├── base_*.png (10)                             # Imágenes base SDXL 1024px
│   ├── scripts/ (run_full_experiment.py, ...)      # Pipeline maestro distribuido en GPU
│   ├── comparisons/ (master_grid_10x4.png, ...)    # Matriz Maestra y tiras comparativas 4K
│   └── models/                                     # Pesos .safetensors (r=16, r=32, r=64)
│
├── material_clase/                                 # 📚 Material docente, apuntes y teoría de PontIA
│   ├── 1.uso_modelos_opensource.ipynb              # Cuaderno de clase: modelos open-source
│   ├── 1.finetuning_diffusion_model.ipynb          # Cuaderno de clase: difusión
│   ├── 1-2.modelo_difusion_from_scratch.ipynb      # Difusión implementada desde cero
│   ├── 2.finetuning_stable_diffusion_lora.ipynb    # Cuaderno de clase: LoRA con Pokémon
│   ├── 1-enunciado_entrega_modelos_generacion_imagen.pdf # Enunciado oficial de la entrega
│   └── *.pdf                                       # Apuntes y diapositivas de las clases en vivo
│
└── models/                                         # 🏷️ Metadatos y documentación de adaptadores locales
    ├── README.md
    └── HF_README.md
```

---

## 6. Referencias y Bibliografía Técnica

1. **Hu, E. J., et al. (2021).** *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
2. **Rombach, R., et al. (2022).** *High-Resolution Image Synthesis with Latent Diffusion Models* (Stable Diffusion). CVPR 2022.
3. **Podell, D., et al. (2023).** *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*. arXiv:2307.01952.
4. **Hugging Face Diffusers & PEFT Documentation:** [https://huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers).
5. **Old Book Illustrations Dataset:** [https://huggingface.co/datasets/gigant/oldbookillustrations](https://huggingface.co/datasets/gigant/oldbookillustrations).
