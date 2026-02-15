#!/bin/bash
set -e

cd /root/Lux

echo "Starting dataset download at $(date)..."
echo "Instance has ~4.6 Gbps download speed"

# Download 2 parts (enough for initial training, ~80GB)
# The download script handles metadata CSV + video zips + extraction
python scripts/download_dataset.py \
    --output_dir datasets/openvid \
    --num_parts 2 \
    --start_part 0 \
    --extract \
    --num_workers 2 2>&1

echo ""
echo "Download complete at $(date)"
echo "Files:"
ls -lh datasets/openvid/ 2>/dev/null || echo "No files yet"
echo ""
echo "Video count:"
find datasets/openvid -name "*.mp4" 2>/dev/null | wc -l || echo "0"
