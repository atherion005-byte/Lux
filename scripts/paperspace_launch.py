#!/usr/bin/env python3
"""
Paperspace CLI Launcher for Lux Training
==========================================
Automatically launches a GPU instance on Paperspace and starts training.

Prerequisites:
  1. Install Paperspace CLI: pip install paperspace
  2. Get API key from: https://console.paperspace.com/account/api
  3. Set env var: set PAPERSPACE_API_KEY=your_api_key
  
Usage:
  python scripts/paperspace_launch.py --gpu A4000 --name lux-training-b

This will:
  1. Launch A4000 instance on Paperspace
  2. Upload your Lux code
  3. Download OpenVid-1M dataset
  4. Start training Lux-B
  5. Stream logs to your PC
"""
import argparse
import os
import sys
import subprocess
import time
import json
from pathlib import Path


def run_cmd(cmd, capture=False):
    """Execute shell command."""
    print(f"$ {cmd}")
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.returncode
    else:
        return subprocess.run(cmd, shell=True).returncode


def check_paperspace_cli():
    """Ensure Paperspace CLI is installed."""
    try:
        subprocess.run("paperspace --version", shell=True, capture_output=True, check=True)
        return True
    except:
        print("❌ Paperspace CLI not found.")
        print("Install with: pip install paperspace")
        return False


def check_api_key():
    """Check if PAPERSPACE_API_KEY is set."""
    api_key = os.environ.get("PAPERSPACE_API_KEY")
    if not api_key:
        print("❌ PAPERSPACE_API_KEY not set")
        print("Get your key from: https://console.paperspace.com/account/api")
        print("Then run: set PAPERSPACE_API_KEY=your_key")
        return False
    return True


def launch_instance(gpu="A4000", name="lux-training", region="US-West"):
    """Launch a Paperspace GPU instance."""
    print(f"\n{'='*60}")
    print(f"  Launching Paperspace Instance")
    print(f"{'='*60}\n")
    
    print(f"GPU:     {gpu}")
    print(f"Name:    {name}")
    print(f"Region:  {region}")
    print(f"Image:   Ubuntu 22.04 + PyTorch\n")
    
    cmd = f"""paperspace machines create \\
        --machineType "{gpu}" \\
        --region "{region}" \\
        --templateId t0nspur9k \\
        --name "{name}" \\
        --billingType hourly"""
    
    print("Creating instance...")
    output, rc = run_cmd(cmd, capture=True)
    
    if rc != 0:
        print(f"❌ Failed to create instance")
        print(output)
        return None
    
    # Parse machine ID from output
    try:
        machine_id = output.split("machineId")[1].split(":")[1].strip().strip('"').split(',')[0]
        print(f"✓ Instance created: {machine_id}")
        return machine_id
    except:
        print(f"Output: {output}")
        print("Could not parse machine ID. Check Paperspace console at:")
        print("https://console.paperspace.com/tkpw4hzq9c/projects")
        return None


def wait_for_instance(machine_id, timeout=300):
    """Wait for instance to be ready."""
    print(f"\nWaiting for instance to start (max {timeout}s)...")
    start = time.time()
    
    while time.time() - start < timeout:
        cmd = f"paperspace machines show --machineId {machine_id}"
        output, rc = run_cmd(cmd, capture=True)
        
        if rc == 0 and "running" in output.lower():
            print("✓ Instance is running!")
            return True
        
        print(".", end="", flush=True)
        time.sleep(10)
    
    print(f"\n❌ Timeout waiting for instance")
    return False


def get_instance_info(machine_id):
    """Get IP and SSH details."""
    cmd = f"paperspace machines show --machineId {machine_id}"
    output, rc = run_cmd(cmd, capture=True)
    
    if rc != 0:
        return None
    
    # Parse IP address
    try:
        lines = output.split('\n')
        for line in lines:
            if 'ipAddress' in line.lower() or 'ip' in line.lower():
                ip = line.split(':')[1].strip().strip('"')
                return {
                    'machine_id': machine_id,
                    'ip': ip,
                    'user': 'paperspace',
                    'key': '~/.ssh/paperspace'
                }
    except:
        pass
    
    return None


def upload_code(ip, user="paperspace"):
    """Upload Lux code to instance."""
    print(f"\nUploading Lux code to {ip}...")
    
    # Exclude large directories
    cmd = f"""rsync -av -e "ssh -i ~/.ssh/paperspace" \\
        --exclude datasets \\
        --exclude checkpoints \\
        --exclude outputs \\
        --exclude .git \\
        . {user}@{ip}:~/Lux/"""
    
    run_cmd(cmd)
    print("✓ Code uploaded")


def setup_training_script():
    """Create a training script to run on the instance."""
    script = """#!/bin/bash
set -e

cd ~/Lux

echo "=========================================="
echo "  Lux Training on Paperspace"
echo "=========================================="

# Setup
echo "[1/4] Installing dependencies..."
pip install -q -r requirements.txt

# Download dataset
echo "[2/4] Downloading OpenVid-1M dataset..."
python scripts/download_dataset.py \\
    --output_dir datasets/openvid \\
    --num_parts 20 \\
    --extract

# Login to W&B for monitoring
echo "[3/4] Setting up Weights & Biases..."
echo "Visit https://wandb.ai/authorize and copy your API key"
read -p "Paste your W&B API key: " wandb_key
export WANDB_API_KEY=$wandb_key

# Train
echo "[4/4] Starting training..."
python scripts/train.py \\
    --stage dit \\
    --model_variant B \\
    --data_dir datasets/openvid/videos \\
    --metadata_file datasets/openvid/metadata.jsonl \\
    --resolution 512 \\
    --num_frames 16 \\
    --batch_size 4 \\
    --num_epochs 10 \\
    --mixed_precision fp16 \\
    --use_wandb \\
    --project_name "lux-paperspace-training"

echo "=========================================="
echo "  Training Complete!"
echo "=========================================="
"""
    
    with open("train_on_paperspace.sh", "w") as f:
        f.write(script)
    
    return "train_on_paperspace.sh"


def main():
    parser = argparse.ArgumentParser(
        description="Launch Lux training on Paperspace"
    )
    parser.add_argument("--gpu", default="A4000", 
                       choices=["A4000", "RTX5000", "A6000", "A100", "H100"])
    parser.add_argument("--name", default="lux-training-b")
    parser.add_argument("--region", default="US-West")
    parser.add_argument("--dataset-parts", type=int, default=20)
    parser.add_argument("--model-variant", default="B", choices=["S", "B", "L"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  Lux Training - Paperspace Launcher")
    print("="*60)
    
    # Check prerequisites
    if not check_paperspace_cli():
        return 1
    
    if not check_api_key():
        return 1
    
    # Launch instance
    machine_id = launch_instance(args.gpu, args.name, args.region)
    if not machine_id:
        return 1
    
    # Wait for instance
    if not wait_for_instance(machine_id):
        return 1
    
    # Get instance info
    info = get_instance_info(machine_id)
    if not info:
        print("Could not get instance info. Check Paperspace console.")
        return 1
    
    ip = info['ip']
    print(f"\n✓ Instance ready at {ip}")
    
    # Create training script
    train_script = setup_training_script()
    print(f"✓ Created training script: {train_script}")
    
    # Upload code
    # Note: Paperspace doesn't support SSH key auth by default
    # You'll need to use the Paperspace GUI or API for file uploads
    
    print(f"\n{'='*60}")
    print("  Next Steps")
    print(f"{'='*60}\n")
    
    print(f"1. Open Paperspace console:")
    print(f"   https://console.paperspace.com/tkpw4hzq9c/projects\n")
    
    print(f"2. Find your instance: {args.name}")
    print(f"   Machine ID: {machine_id}\n")
    
    print(f"3. Click 'Open Terminal' or 'Open in Jupyter'\n")
    
    print(f"4. In the terminal, run:")
    print(f"   git clone https://github.com/YOUR_USERNAME/Lux.git")
    print(f"   cd Lux")
    print(f"   bash train_on_paperspace.sh\n")
    
    print(f"Or upload files via the web UI and start training.\n")
    
    print(f"5. Monitor training at: https://wandb.ai/YOUR_USERNAME\n")
    
    print(f"Instance billing started: ${{'A4000': 0.76, 'RTX5000': 0.82, 'A6000': 1.89, 'A100': 1.15, 'H100': 2.24}.get(args.gpu, '?')}/hour\n")
    
    print("To stop instance (save costs):")
    print(f"  paperspace machines stop --machineId {machine_id}")
    print(f"Or stop from: https://console.paperspace.com/tkpw4hzq9c/projects\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
