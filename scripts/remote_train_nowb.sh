#!/bin/bash
set -e

cd /root/Lux

echo "=========================================="
echo "  Lux-B Training on RTX 5090"  
echo "=========================================="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Videos: $(find datasets/openvid/videos -name '*.mp4' | wc -l)"
echo "Start: $(date)"
echo ""

# Train Lux-B (1.6B params) on RTX 5090 (32GB VRAM)
# - Resolution 512x512: good quality, fits in VRAM  
# - 16 frames: standard video length
# - Batch size 2: with gradient accumulation 8 = effective batch 16
# - BF16: RTX 5090 has excellent BF16 performance
# - Save every 500 steps for checkpointing
# - No wandb (not configured)
python scripts/train.py \
    --stage dit \
    --model_variant B \
    --data_dir datasets/openvid/videos \
    --metadata_file datasets/openvid/metadata.jsonl \
    --resolution 512 \
    --num_frames 16 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --mixed_precision bf16 \
    --save_every_n_steps 500 \
    --output_dir outputs \
    --checkpoint_dir checkpoints \
    --project_name lux-video-gen 2>&1

echo ""
echo "Training finished at $(date)"
