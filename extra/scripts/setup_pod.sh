#!/usr/bin/env bash
set -e

echo "=== [1/4] Installing Required Python Libraries ==="
pip install --no-cache-dir \
    "diffusers>=0.30.0" \
    "transformers>=4.44.0" \
    "accelerate>=0.34.0" \
    "peft>=0.12.0" \
    "datasets>=2.20.0" \
    "safetensors>=0.4.4" \
    "torchvision" \
    "huggingface_hub>=0.24.0" \
    "pillow" \
    "matplotlib"

echo "=== [2/4] Configuring Hugging Face Hub Token ==="
mkdir -p ~/.cache/huggingface
if [ -n "$HF_TOKEN" ]; then
    echo -n "$HF_TOKEN" > ~/.cache/huggingface/token
    chmod 600 ~/.cache/huggingface/token
fi
python3 -c "from huggingface_hub import HfApi; api = HfApi(); user = api.whoami(); print('Hugging Face User:', user.get('name', 'Twinber'))"

echo "=== [3/4] Verifying GPU Environment ==="
nvidia-smi
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0), '| VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))"

echo "=== [4/4] Environment Ready! ==="
