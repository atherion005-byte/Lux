#!/bin/bash
set -e

echo "=========================================="
echo "  Lux Video Gen - Remote Setup Script"
echo "=========================================="
echo ""

# Step 1: System packages
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq git wget curl ffmpeg libsm6 libxext6 > /dev/null 2>&1
echo "  Done."

# Step 2: Upgrade PyTorch for RTX 5090 (sm_120 / Blackwell)
echo "[2/6] Upgrading PyTorch for RTX 5090 support..."
pip install --upgrade pip > /dev/null 2>&1
# Install latest PyTorch with CUDA 12.8 support (supports sm_120)
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3
echo "  Verifying PyTorch..."
python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    t = torch.zeros(1, device='cuda')
    print(f'  Tensor on GPU: OK')
"
echo "  Done."

# Step 3: Clone repo
echo "[3/6] Cloning Lux repo from GitHub..."
cd /root
if [ -d "Lux" ]; then
    echo "  Repo already exists, pulling latest..."
    cd Lux && git pull
else
    git clone https://github.com/atherion005-byte/Lux.git
    cd Lux
fi
echo "  Done."

# Step 4: Install Python dependencies
echo "[4/6] Installing Python dependencies..."
pip install -r requirements.txt 2>&1 | tail -5
echo "  Done."

# Step 5: Verify installation
echo "[5/6] Running quick verification..."
python -c "
import torch
import transformers
print(f'  torch: {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
props = torch.cuda.get_device_properties(0)
print(f'  VRAM: {props.total_memory / 1e9:.1f} GB')
"
echo "  Done."

# Step 6: Create directories
echo "[6/6] Creating directories..."
mkdir -p datasets/openvid checkpoints outputs
echo "  Done."

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "  Next: Run dataset download & training"
echo "=========================================="
