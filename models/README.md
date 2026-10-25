# 📦 Pesos LoRA Stable Diffusion v1.4

Los pesos de fine-tuning para `CompVis/stable-diffusion-v1-4` están alojados en Hugging Face Hub:

👉 **[https://huggingface.co/Twinber/sd-v1-4-oldbook-illustrations-lora](https://huggingface.co/Twinber/sd-v1-4-oldbook-illustrations-lora)**

```python
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("Twinber/sd-v1-4-oldbook-illustrations-lora")

prompt = "an antique illustration of a botanic garden with exotic flowers and trees"
image = pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("botanica.png")
```
