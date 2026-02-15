#!/bin/bash
"""
Cloud Training Setup Script for Lux
====================================
Run this on a cloud GPU instance to set up and train the model.

Supports: Lambda Labs, Paperspace, Vast.ai, Modal, RunwayML

Usage:
  bash cloud_train_setup.sh
  
Or paste commands manually into the cloud terminal.
"""

# 1. CLONE REPO & INSTALL
echo "=== Setting up Lux training on cloud GPU ==="

# If repo doesn't exist, clone it
if [ ! -d "Lux" ]; then
    git clone https://github.com/YOUR_USERNAME/Lux.git
    cd Lux
else
    cd Lux
    git pull
fi

# Install dependencies
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q -r requirements.txt

# 2. DOWNLOAD DATASET
echo "=== Downloading OpenVid-1M dataset ==="
# This will download in background if on cloud with good bandwidth
python scripts/download_dataset.py --output_dir datasets/openvid --num_parts 20 --extract

# 3. MONITOR GPU
echo "Current GPU status:"
nvidia-smi

# 4. START TRAINING
echo "=== Starting Lux-B training ==="
python scripts/train.py \
    --stage dit \
    --model_variant B \
    --data_dir datasets/openvid/videos \
    --metadata_file datasets/openvid/metadata.jsonl \
    --resolution 512 \
    --num_frames 16 \
    --batch_size 4 \
    --num_epochs 10 \
    --mixed_precision fp16 \
    --use_wandb \
    --project_name "lux-cloud-training"

echo "=== Training complete! Check outputs/ for checkpoints ==="
