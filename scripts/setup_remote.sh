#!/bin/bash
set -e

echo "=== Installing system packages ==="
apt-get update -qq && apt-get install -y -qq git ffmpeg libsm6 libxext6 wget curl > /dev/null 2>&1
echo "System packages done"

echo "=== Upgrading PyTorch for Blackwell (RTX 5080) ==="
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
echo "PyTorch installed"

echo "=== Cloning Lux repo ==="
cd /root
if [ -d "Lux" ]; then
    cd Lux && git pull
else
    git clone https://github.com/atherion005-byte/Lux.git
    cd Lux
fi

echo "=== Installing Lux dependencies ==="
pip3 install transformers accelerate diffusers einops timm scipy pillow imageio imageio-ffmpeg decord opencv-python-headless safetensors sentencepiece protobuf huggingface_hub
echo "Deps installed"

echo "=== Creating directories ==="
mkdir -p datasets/openvid/videos checkpoints outputs

echo "=== Verifying ==="
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} {props.total_mem/1e9:.1f}GB')
print(f'RAM: {__import__(\"os\").sysconf(\"SC_PAGE_SIZE\") * __import__(\"os\").sysconf(\"SC_PHYS_PAGES\") / 1e9:.0f} GB')
"

echo "=== SETUP COMPLETE ==="
