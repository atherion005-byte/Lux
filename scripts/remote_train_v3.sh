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
# data_dir = "." because metadata.jsonl has paths like "datasets/openvid/videos/xxx.mp4"
python scripts/train.py \
    --stage dit \
    --model_variant B \
    --data_dir . \
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
