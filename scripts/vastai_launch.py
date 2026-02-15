#!/usr/bin/env python3
"""
Vast.ai Training Launcher for Lux
===================================
Automates GPU instance search, launch, setup, and training on Vast.ai.

Usage:
    # Search for best deals
    python scripts/vastai_launch.py search
    
    # Launch RTX 5090 instance and start training
    python scripts/vastai_launch.py launch --instance-id 31109019
    
    # Full auto: search + launch best match
    python scripts/vastai_launch.py auto --gpu RTX_5090 --max-price 0.60
    
    # Check running instances
    python scripts/vastai_launch.py status
    
    # Connect via SSH
    python scripts/vastai_launch.py ssh --instance-id <id>
    
    # Stop instance
    python scripts/vastai_launch.py stop --instance-id <id>

Requires: pip install vastai
API Key:  vastai set api-key <your_key>
"""

import argparse
import subprocess
import sys
import os
import json
import time
from pathlib import Path


# ============================================================
#  GPU Recommendations for Lux Training
# ============================================================
GPU_RECOMMENDATIONS = {
    "S": {  # Lux-S (483M params)
        "min_vram": 12,
        "recommended_gpus": ["RTX_4090", "RTX_5080", "RTX_5090", "RTX_3090"],
        "batch_size": 4,
        "est_hours": 50,
    },
    "B": {  # Lux-B (1.6B params) 
        "min_vram": 24,
        "recommended_gpus": ["RTX_5090", "RTX_4090", "A6000", "L40", "H100"],
        "batch_size": 4,
        "est_hours": 150,
    },
    "L": {  # Lux-L (2.4B params)
        "min_vram": 48,
        "recommended_gpus": ["H100", "H200", "A100", "L40S"],
        "batch_size": 2,
        "est_hours": 300,
    },
}


def run_vast(cmd, capture=True):
    """Execute vastai CLI command."""
    full_cmd = f"vastai {cmd}"
    if capture:
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    else:
        result = subprocess.run(full_cmd, shell=True)
        return "", "", result.returncode


def check_api_key():
    """Verify Vast.ai API key is configured."""
    home = Path.home()
    key_file = home / ".vast_api_key"
    
    if key_file.exists():
        key = key_file.read_text().strip()
        if key and len(key) > 10:
            return True
    
    print("=" * 60)
    print("  Vast.ai API Key Required")
    print("=" * 60)
    print()
    print("1. Go to: https://cloud.vast.ai/cli/")
    print("2. Copy your API Key")
    print("3. Run:   vastai set api-key YOUR_KEY")
    print()
    return False


# ============================================================
#  Search for GPU instances
# ============================================================
def cmd_search(args):
    """Search Vast.ai for available GPU instances."""
    print("\n" + "=" * 60)
    print("  Vast.ai GPU Search")
    print("=" * 60)
    
    model = args.model
    rec = GPU_RECOMMENDATIONS.get(model, GPU_RECOMMENDATIONS["B"])
    min_vram = rec["min_vram"]
    max_price = args.max_price
    
    print(f"\n  Model:      Lux-{model}")
    print(f"  Min VRAM:   {min_vram} GB")
    print(f"  Max Price:  ${max_price}/hr")
    print(f"  Min Reliability: {args.min_reliability}")
    
    # Build search query
    query = (
        f"gpu_ram >= {min_vram} "
        f"reliability > {args.min_reliability} "
        f"dph <= {max_price} "
        f"num_gpus >= {args.num_gpus} "
        f"cuda_vers >= 12.0 "
        f"disk_space >= {args.disk_space} "
        f"inet_down >= 100"
    )
    
    sort = args.sort or "dph"  # sort by price by default
    
    cmd = f'search offers "{query}" -o "{sort}"'
    
    print(f"\n  Searching...")
    stdout, stderr, code = run_vast(cmd)
    
    if code != 0:
        print(f"\n  Error: {stderr}")
        return False
    
    if not stdout.strip():
        print("\n  No instances found matching criteria.")
        print("  Try increasing --max-price or decreasing --min-reliability")
        return False
    
    # Parse and display results
    lines = stdout.strip().split("\n")
    
    print(f"\n  Found {len(lines)-1} matching instances:\n")
    
    # Print header + top results
    for i, line in enumerate(lines[:min(20, len(lines))]):
        print(f"  {line}")
    
    if len(lines) > 20:
        print(f"\n  ... and {len(lines) - 20} more. Use --max-price to narrow.")
    
    print(f"\n  To launch: python scripts/vastai_launch.py launch --instance-id <ID>")
    print(f"  Or auto:   python scripts/vastai_launch.py auto --model {model} --max-price {max_price}")
    
    return True


# ============================================================
#  Auto-select and launch best instance
# ============================================================
def cmd_auto(args):
    """Automatically find and launch the best instance."""
    print("\n" + "=" * 60)
    print("  Vast.ai Auto-Launch")
    print("=" * 60)
    
    model = args.model
    rec = GPU_RECOMMENDATIONS.get(model, GPU_RECOMMENDATIONS["B"])
    min_vram = rec["min_vram"]
    max_price = args.max_price
    batch_size = rec["batch_size"]
    est_hours = rec["est_hours"]
    
    print(f"\n  Model:      Lux-{model}")
    print(f"  Min VRAM:   {min_vram} GB")
    print(f"  Max Price:  ${max_price}/hr")
    print(f"  Est. Time:  {est_hours} hours")
    print(f"  Est. Cost:  ${max_price * est_hours:.0f}")
    
    # Search with JSON output
    query = (
        f"gpu_ram >= {min_vram} "
        f"reliability > {args.min_reliability} "
        f"dph <= {max_price} "
        f"num_gpus >= {args.num_gpus} "
        f"cuda_vers >= 12.0 "
        f"disk_space >= 200 "
        f"inet_down >= 100"
    )
    
    cmd = f'search offers "{query}" -o "dph" --raw'
    stdout, stderr, code = run_vast(cmd)
    
    if code != 0:
        print(f"\n  Error searching: {stderr}")
        return False
    
    try:
        offers = json.loads(stdout)
    except json.JSONDecodeError:
        print("\n  Failed to parse search results")
        print(f"  Raw output: {stdout[:500]}")
        return False
    
    if not offers:
        print("\n  No instances found. Try higher --max-price")
        return False
    
    # Pick best offer (cheapest with good reliability)
    best = offers[0]
    instance_id = best.get("id")
    gpu_name = best.get("gpu_name", "Unknown GPU")
    dph = best.get("dph_total", 0)
    vram = best.get("gpu_ram", 0)
    reliability = best.get("reliability2", 0)
    location = best.get("geolocation", "Unknown")
    
    print(f"\n  Best match found!")
    print(f"  ─────────────────────────────────────")
    print(f"  Instance ID:  {instance_id}")
    print(f"  GPU:          {gpu_name}")
    print(f"  VRAM:         {vram:.0f} GB")
    print(f"  Price:        ${dph:.3f}/hr")
    print(f"  Reliability:  {reliability:.1%}")
    print(f"  Location:     {location}")
    print(f"  Est. Total:   ${dph * est_hours:.0f}")
    print(f"  ─────────────────────────────────────")
    
    if not args.yes:
        confirm = input(f"\n  Launch this instance? [y/N]: ").strip().lower()
        if confirm != "y":
            print("  Cancelled.")
            return False
    
    # Launch it
    args.instance_id = instance_id
    args.batch_size = batch_size
    return cmd_launch(args)


# ============================================================
#  Launch instance
# ============================================================
def cmd_launch(args):
    """Launch a Vast.ai instance with training setup."""
    instance_id = args.instance_id
    model = args.model
    
    print(f"\n  Launching instance {instance_id}...")
    
    # Docker image with PyTorch + CUDA
    image = args.image or "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
    disk = args.disk or 250  # GB
    
    # The onstart script sets up everything automatically
    onstart_script = _generate_setup_script(model, args)
    
    # Write setup script to temp file
    setup_file = Path(__file__).parent / "_vastai_setup.sh"
    setup_file.write_text(onstart_script)
    
    # Create instance
    cmd = (
        f'create instance {instance_id} '
        f'--image "{image}" '
        f'--disk {disk} '
        f'--onstart-cmd "bash -c \'wget -qO /root/setup.sh https://raw.githubusercontent.com/YOUR_USERNAME/Lux/main/scripts/_vastai_setup.sh && bash /root/setup.sh\'" '
        f'--direct '
    )
    
    stdout, stderr, code = run_vast(cmd)
    
    if code != 0:
        print(f"\n  Failed to create instance!")
        print(f"  Error: {stderr}")
        print(f"\n  Manual launch alternative:")
        print(f"  vastai create instance {instance_id} --image {image} --disk {disk}")
        return False
    
    print(f"\n  Instance created!")
    
    # Try to extract instance ID from output
    new_id = None
    try:
        data = json.loads(stdout)
        new_id = data.get("new_contract")
    except:
        # Parse from text output
        for word in stdout.split():
            if word.isdigit():
                new_id = word
                break
    
    if new_id:
        print(f"  Contract ID: {new_id}")
        print(f"\n  Instance is booting up (may take 2-10 min)...")
        print(f"\n  Check status:  python scripts/vastai_launch.py status")
        print(f"  Connect SSH:   python scripts/vastai_launch.py ssh --instance-id {new_id}")
    
    print(f"\n  ═══════════════════════════════════════════")
    print(f"  Once connected, run these commands:")
    print(f"  ═══════════════════════════════════════════")
    print(f"")
    print(f"  # 1. Clone and setup")
    print(f"  git clone https://github.com/YOUR_USERNAME/Lux.git && cd Lux")
    print(f"  pip install -r requirements.txt")
    print(f"")
    print(f"  # 2. Download dataset (fast on datacenter internet)")
    print(f"  python scripts/download_dataset.py --output_dir datasets/openvid --num_parts 20")
    print(f"")
    print(f"  # 3. Start training")
    print(f"  python scripts/train.py \\")
    print(f"      --stage dit --model_variant {model} \\")
    print(f"      --data_dir datasets/openvid/videos \\")
    print(f"      --metadata_file datasets/openvid/metadata.jsonl \\")
    print(f"      --resolution 512 --num_frames 16 \\")
    print(f"      --batch_size {args.batch_size} --num_epochs 10 \\")
    print(f"      --mixed_precision fp16 --use_wandb")
    print(f"")
    print(f"  # 4. When done, download checkpoint")
    print(f"  # From your PC:")
    print(f"  python scripts/vastai_launch.py download --instance-id {new_id or '<id>'}")
    
    return True


def _generate_setup_script(model, args):
    """Generate the on-start setup script for the instance."""
    batch_size = getattr(args, "batch_size", 4)
    
    return f"""#!/bin/bash
# Lux Video Gen - Vast.ai Auto-Setup
set -e

echo "=== Lux Training Setup ==="
echo "Model: Lux-{model}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"

# Install core dependencies
pip install -q transformers diffusers accelerate einops rotary-embedding-torch
pip install -q wandb tensorboard opencv-python-headless Pillow decord
pip install -q huggingface_hub tqdm

# Clone repo
if [ ! -d "/root/Lux" ]; then
    cd /root
    git clone https://github.com/YOUR_USERNAME/Lux.git
fi
cd /root/Lux

# Install project requirements  
pip install -q -r requirements.txt 2>/dev/null || true

# Download dataset
echo "=== Downloading OpenVid-1M dataset ==="
python scripts/download_dataset.py --output_dir datasets/openvid --num_parts 20 --extract &
DOWNLOAD_PID=$!

# Wait for at least metadata + first part
echo "Waiting for initial data..."
while [ ! -f "datasets/openvid/metadata.jsonl" ]; do
    sleep 30
    echo "  Waiting for metadata..."
done

echo "=== Starting Lux-{model} Training ==="
nvidia-smi

python scripts/train.py \\
    --stage dit \\
    --model_variant {model} \\
    --data_dir datasets/openvid/videos \\
    --metadata_file datasets/openvid/metadata.jsonl \\
    --resolution 512 \\
    --num_frames 16 \\
    --batch_size {batch_size} \\
    --num_epochs 10 \\
    --mixed_precision fp16 \\
    --use_wandb \\
    --project_name "lux-{model.lower()}-vastai"

echo "=== Training Complete! ==="
"""


# ============================================================
#  Instance management
# ============================================================
def cmd_status(args):
    """Show running instances."""
    print("\n" + "=" * 60)
    print("  Vast.ai Running Instances")
    print("=" * 60)
    
    stdout, stderr, code = run_vast("show instances")
    
    if code != 0:
        print(f"\n  Error: {stderr}")
        return False
    
    if not stdout.strip() or "no instances" in stdout.lower():
        print("\n  No running instances.")
    else:
        print(f"\n{stdout}")
    
    return True


def cmd_ssh(args):
    """SSH into a running instance."""
    instance_id = args.instance_id
    
    print(f"\n  Connecting to instance {instance_id}...")
    
    # Get SSH URL
    stdout, stderr, code = run_vast(f"ssh-url {instance_id}")
    
    if code != 0:
        print(f"\n  Error: {stderr}")
        print(f"  Instance may still be loading. Check status first.")
        return False
    
    ssh_cmd = stdout.strip()
    print(f"  SSH Command: ssh {ssh_cmd}")
    print(f"\n  Connecting...")
    
    os.system(f"ssh {ssh_cmd}")
    return True


def cmd_stop(args):
    """Stop/destroy a running instance."""
    instance_id = args.instance_id
    
    print(f"\n  Stopping instance {instance_id}...")
    
    if not args.force:
        confirm = input("  Are you sure? This will delete the instance. [y/N]: ").strip().lower()
        if confirm != "y":
            print("  Cancelled.")
            return False
    
    stdout, stderr, code = run_vast(f"destroy instance {instance_id}")
    
    if code != 0:
        print(f"\n  Error: {stderr}")
        return False
    
    print(f"  Instance {instance_id} destroyed.")
    return True


def cmd_download(args):
    """Download checkpoints from a running instance."""
    instance_id = args.instance_id
    
    print(f"\n  Downloading checkpoints from instance {instance_id}...")
    
    # Get SSH info
    stdout, stderr, code = run_vast(f"ssh-url {instance_id}")
    if code != 0:
        print(f"\n  Error getting SSH URL: {stderr}")
        return False
    
    ssh_url = stdout.strip()
    
    # Use SCP to download checkpoints
    local_dir = Path("checkpoints")
    local_dir.mkdir(exist_ok=True)
    
    scp_cmd = f"scp -r {ssh_url}:~/Lux/checkpoints/* checkpoints/"
    print(f"  Running: {scp_cmd}")
    
    code = os.system(scp_cmd)
    
    if code == 0:
        print(f"\n  Checkpoints saved to ./checkpoints/")
        # List what we got
        ckpts = list(local_dir.glob("*.pt"))
        for c in ckpts:
            size_mb = c.stat().st_size / 1e6
            print(f"    {c.name} ({size_mb:.0f} MB)")
    else:
        print(f"\n  Download failed. Try manually:")
        print(f"  scp -r {ssh_url}:~/Lux/checkpoints/* checkpoints/")
    
    return code == 0


def cmd_logs(args):
    """View logs from a running instance."""
    instance_id = args.instance_id
    
    print(f"\n  Fetching logs from instance {instance_id}...")
    
    stdout, stderr, code = run_vast(f"logs {instance_id}")
    
    if code != 0:
        print(f"\n  Error: {stderr}")
        return False
    
    print(stdout)
    return True


# ============================================================
#  Cost estimator
# ============================================================
def cmd_estimate(args):
    """Estimate training cost."""
    model = args.model
    rec = GPU_RECOMMENDATIONS.get(model, GPU_RECOMMENDATIONS["B"])
    est_hours = rec["est_hours"]
    
    print("\n" + "=" * 60)
    print("  Vast.ai Cost Estimator")
    print("=" * 60)
    
    prices = [
        ("RTX 5090 (32GB)", 0.30, "Best value for Lux-B"),
        ("RTX 4090 (24GB)", 0.27, "Good for Lux-S/B"),
        ("RTX 3090 (24GB)", 0.15, "Budget option"),
        ("H100 SXM (80GB)", 1.47, "Premium, fastest"),
        ("H200 (140GB)", 2.07, "Top tier"),
        ("L40S (48GB)", 0.51, "Good mid-range"),
        ("B200 (179GB)", 2.65, "Bleeding edge"),
    ]
    
    print(f"\n  Model: Lux-{model} ({rec['min_vram']}GB+ VRAM required)")
    print(f"  Estimated training: {est_hours} hours")
    print()
    print(f"  {'GPU':<25} {'$/hr':<10} {'Total':<12} {'Notes'}")
    print(f"  {'─'*25} {'─'*10} {'─'*12} {'─'*25}")
    
    for gpu, price, note in prices:
        vram = int(gpu.split("(")[1].split("GB")[0])
        if vram >= rec["min_vram"]:
            total = price * est_hours
            marker = " ← recommended" if total < 200 and vram >= rec["min_vram"] else ""
            print(f"  {gpu:<25} ${price:<9.2f} ${total:<11.0f} {note}{marker}")
    
    print(f"\n  Note: Prices are approximate and change frequently.")
    print(f"  Run 'python scripts/vastai_launch.py search' for live pricing.")


# ============================================================
#  Main CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Vast.ai GPU Launcher for Lux Video Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for cheap GPUs
  python scripts/vastai_launch.py search --model B --max-price 0.60
  
  # Auto-find and launch best deal
  python scripts/vastai_launch.py auto --model B --max-price 0.50 --yes
  
  # Launch specific instance
  python scripts/vastai_launch.py launch --instance-id 31109019 --model B
  
  # Check running instances
  python scripts/vastai_launch.py status
  
  # SSH into instance
  python scripts/vastai_launch.py ssh --instance-id <id>
  
  # Download trained checkpoints
  python scripts/vastai_launch.py download --instance-id <id>
  
  # Estimate costs
  python scripts/vastai_launch.py estimate --model B
  
  # Stop instance when done
  python scripts/vastai_launch.py stop --instance-id <id>
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Common args
    def add_common(p):
        p.add_argument("--model", default="B", choices=["S", "B", "L"], help="Lux model variant")
    
    # Search
    search_p = subparsers.add_parser("search", help="Search for GPU instances")
    add_common(search_p)
    search_p.add_argument("--max-price", type=float, default=1.00, help="Max $/hr")
    search_p.add_argument("--min-reliability", type=float, default=0.98, help="Min reliability (0-1)")
    search_p.add_argument("--num-gpus", type=int, default=1, help="Min number of GPUs")
    search_p.add_argument("--disk-space", type=int, default=200, help="Min disk space GB")
    search_p.add_argument("--sort", default="dph", help="Sort field")
    
    # Auto
    auto_p = subparsers.add_parser("auto", help="Auto-find and launch best instance")
    add_common(auto_p)
    auto_p.add_argument("--max-price", type=float, default=0.60, help="Max $/hr")
    auto_p.add_argument("--min-reliability", type=float, default=0.98, help="Min reliability")
    auto_p.add_argument("--num-gpus", type=int, default=1, help="Min GPUs")
    auto_p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    auto_p.add_argument("--image", default=None, help="Docker image")
    auto_p.add_argument("--disk", type=int, default=250, help="Disk GB")
    auto_p.add_argument("--batch-size", type=int, default=4)
    
    # Launch
    launch_p = subparsers.add_parser("launch", help="Launch specific instance")
    add_common(launch_p)
    launch_p.add_argument("--instance-id", required=True, help="Instance type ID from search")
    launch_p.add_argument("--image", default=None, help="Docker image")
    launch_p.add_argument("--disk", type=int, default=250, help="Disk space GB")
    launch_p.add_argument("--batch-size", type=int, default=4)
    
    # Status
    subparsers.add_parser("status", help="Show running instances")
    
    # SSH
    ssh_p = subparsers.add_parser("ssh", help="SSH into instance")
    ssh_p.add_argument("--instance-id", required=True)
    
    # Stop
    stop_p = subparsers.add_parser("stop", help="Stop/destroy instance")
    stop_p.add_argument("--instance-id", required=True)
    stop_p.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    
    # Download
    dl_p = subparsers.add_parser("download", help="Download checkpoints")
    dl_p.add_argument("--instance-id", required=True)
    
    # Logs
    logs_p = subparsers.add_parser("logs", help="View instance logs")
    logs_p.add_argument("--instance-id", required=True)
    
    # Estimate
    est_p = subparsers.add_parser("estimate", help="Estimate training cost")
    add_common(est_p)
    
    args = parser.parse_args()
    
    # Check API key for commands that need it
    if args.command in ("search", "auto", "launch", "status", "ssh", "stop", "download", "logs"):
        if not check_api_key():
            return 1
    
    commands = {
        "search": cmd_search,
        "auto": cmd_auto,
        "launch": cmd_launch,
        "status": cmd_status,
        "ssh": cmd_ssh,
        "stop": cmd_stop,
        "download": cmd_download,
        "logs": cmd_logs,
        "estimate": cmd_estimate,
    }
    
    if args.command in commands:
        return 0 if commands[args.command](args) else 1
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
