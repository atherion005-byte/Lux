#!/bin/bash
set -e
cd /root/Lux

echo "=== Fast dataset download via wget ==="
mkdir -p datasets/openvid/videos

# Download part 0 and 1 in parallel using wget (much faster than HF hub)
BASE="https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main"

echo "Downloading part 0..."
wget -q --show-progress -O datasets/openvid/OpenVid_part0.zip "$BASE/OpenVid_part0.zip" &
PID0=$!

echo "Downloading part 1..."
wget -q --show-progress -O datasets/openvid/OpenVid_part1.zip "$BASE/OpenVid_part1.zip" &
PID1=$!

echo "Waiting for both downloads..."
wait $PID0
echo "Part 0 done!"
wait $PID1
echo "Part 1 done!"

ls -lh datasets/openvid/OpenVid_part*.zip

echo "=== Extracting part 0 ==="
cd datasets/openvid/videos
unzip -q -o /root/Lux/datasets/openvid/OpenVid_part0.zip
echo "Part 0 extracted"

echo "=== Extracting part 1 ==="
unzip -q -o /root/Lux/datasets/openvid/OpenVid_part1.zip
echo "Part 1 extracted"

cd /root/Lux
echo "=== Cleaning up zips ==="
rm -f datasets/openvid/OpenVid_part0.zip datasets/openvid/OpenVid_part1.zip

VIDEO_COUNT=$(ls datasets/openvid/videos/*.mp4 2>/dev/null | wc -l)
echo "=== DONE: $VIDEO_COUNT videos ready ==="
