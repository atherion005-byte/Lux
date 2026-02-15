#!/bin/bash
set -e
cd /root/Lux

echo "=== Starting Lux-B Training ==="
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Videos: $(find datasets/openvid/videos -name '*.mp4' | wc -l)"
echo ""

# Start Part 1 download in background while training
if [ ! -f datasets/openvid/OpenVid_part1.zip ]; then
    echo "Starting Part 1 download in background..."
    wget -q -c -O datasets/openvid/OpenVid_part1.zip \
        'https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part1.zip' &
    echo "Part 1 downloading in background (PID $!)"
fi

# Train Lux-B (1.6B params) on 2x RTX A5000 (24GB each)
# DiT on GPU 0, VAE+TextEncoder on GPU 1
# bf16 + gradient checkpointing + 8-bit Adam + batch 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install bitsandbytes for 8-bit Adam
pip3 install bitsandbytes -q

python3 scripts/train.py \
    --stage dit \
    --model_variant B \
    --data_dir datasets/openvid/videos \
    --resolution 256 \
    --num_frames 16 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --save_every_n_steps 500 \
    --log_every_n_steps 1 \
    2>&1 | tee training_log.txt
