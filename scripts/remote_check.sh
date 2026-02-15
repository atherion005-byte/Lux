#!/bin/bash
echo "=== GPU CHECK ==="
nvidia-smi
echo ""
echo "=== PYTHON & PYTORCH ==="
python --version
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'CUDA version: {torch.version.cuda}')
"
echo ""
echo "=== DISK ==="
df -h / | tail -1
echo ""
echo "=== RAM ==="
free -h | head -2
echo ""
echo "=== DONE ==="
