"""
Cloud Training Manager for Lux
===============================
Handles data upload/download and checkpoint management between your PC and cloud GPU.

Usage:
    # Upload your local code to cloud
    python scripts/cloud_manager.py upload --remote user@lambda.io:/home/user/Lux
    
    # Download checkpoints from cloud
    python scripts/cloud_manager.py download-checkpoints --remote user@lambda.io:/home/user/Lux/checkpoints
    
    # Monitor training (pulls logs)
    python scripts/cloud_manager.py monitor --remote user@lambda.io:/home/user/Lux
"""
import argparse
import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime


def run_cmd(cmd, quiet=False):
    """Execute shell command."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=quiet)
    return result.returncode == 0


def upload_to_cloud(remote_path, local_dir=".", exclude="datasets checkpoints outputs .git"):
    """Upload local code to cloud."""
    print(f"\n{'='*60}")
    print(f"  Uploading to {remote_path}")
    print(f"{'='*60}\n")
    
    exclude_args = " ".join(f"--exclude='{ex}'" for ex in exclude.split())
    cmd = f"rsync -av --delete {exclude_args} {local_dir}/ {remote_path}/"
    
    run_cmd(cmd)
    print("\n✓ Upload complete")


def download_checkpoints(remote_path, local_dir="checkpoints"):
    """Download training checkpoints from cloud."""
    print(f"\n{'='*60}")
    print(f"  Downloading checkpoints from {remote_path}")
    print(f"{'='*60}\n")
    
    os.makedirs(local_dir, exist_ok=True)
    cmd = f"rsync -av --delete {remote_path}/ {local_dir}/"
    
    run_cmd(cmd)
    print(f"\n✓ Checkpoints saved to {local_dir}")


def monitor_training(remote_path, output_dir="cloud_logs"):
    """Pull latest training logs and display."""
    print(f"\n{'='*60}")
    print(f"  Monitoring training on {remote_path}")
    print(f"{'='*60}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Pull logs
    cmd = f"rsync -av {remote_path}/outputs/ {output_dir}/"
    run_cmd(cmd, quiet=True)
    
    # Display recent log
    log_dir = Path(output_dir)
    latest_log = max(log_dir.glob("*.log"), default=None, key=lambda p: p.stat().st_mtime)
    
    if latest_log:
        print(f"\n📊 Latest log: {latest_log.name}")
        print("-" * 60)
        with open(latest_log) as f:
            lines = f.readlines()
            # Show last 50 lines
            for line in lines[-50:]:
                print(line, end="")
    
    # Check GPU usage
    print("\n\n📈 GPU Status on cloud:")
    # This would need SSH access
    cmd = f"ssh {remote_path.split(':')[0]} nvidia-smi"
    run_cmd(cmd, quiet=False)


def estimate_cost(hours=24, gpu="A40", rate_per_hour=0.90):
    """Estimate training cost."""
    print(f"\n{'='*60}")
    print(f"  Cost Estimate")
    print(f"{'='*60}\n")
    
    cost = hours * rate_per_hour
    
    print(f"GPU:        {gpu}")
    print(f"Rate:       ${rate_per_hour}/hour")
    print(f"Duration:   {hours} hours")
    print(f"Total Cost: ${cost:.2f}\n")
    
    print("Recommendations:")
    if hours > 72:
        print("  • Consider using preemptible GPUs (50% cheaper, may interrupt)")
    if cost > 500:
        print("  • Long training - consider buying dedicated hardware or using cheaper providers")
    if cost < 50:
        print("  • Budget-friendly - good for testing!")


def setup_wandb(api_key=None):
    """Setup Weights & Biases for cloud logging."""
    print(f"\n{'='*60}")
    print(f"  Setting up W&B for remote monitoring")
    print(f"{'='*60}\n")
    
    if api_key:
        cmd = f"wandb login {api_key}"
    else:
        print("Get your API key from: https://wandb.ai/authorize")
        print("\nOn the cloud machine, run:")
        print("  wandb login")
        return
    
    run_cmd(cmd)
    print("\n✓ W&B configured - training metrics will sync automatically")


def main():
    parser = argparse.ArgumentParser(
        description="Manage Lux training on cloud GPUs"
    )
    
    subparsers = parser.add_subparsers(dest="command")
    
    # Upload
    upload_parser = subparsers.add_parser("upload", help="Upload code to cloud")
    upload_parser.add_argument("--remote", required=True, help="Remote path (user@host:/path)")
    upload_parser.add_argument("--local", default=".", help="Local directory")
    
    # Download
    download_parser = subparsers.add_parser("download-checkpoints", help="Download checkpoints")
    download_parser.add_argument("--remote", required=True, help="Remote checkpoint path")
    download_parser.add_argument("--local", default="checkpoints", help="Local directory")
    
    # Monitor
    monitor_parser = subparsers.add_parser("monitor", help="Monitor training")
    monitor_parser.add_argument("--remote", required=True, help="Remote path")
    
    # Cost
    cost_parser = subparsers.add_parser("cost", help="Estimate training cost")
    cost_parser.add_argument("--hours", type=int, default=24, help="Training hours")
    cost_parser.add_argument("--gpu", default="A40", help="GPU type")
    cost_parser.add_argument("--rate", type=float, default=0.90, help="$/hour")
    
    # W&B
    wandb_parser = subparsers.add_parser("wandb", help="Setup Weights & Biases")
    wandb_parser.add_argument("--api-key", help="W&B API key")
    
    args = parser.parse_args()
    
    if args.command == "upload":
        upload_to_cloud(args.remote, args.local)
    elif args.command == "download-checkpoints":
        download_checkpoints(args.remote, args.local)
    elif args.command == "monitor":
        while True:
            monitor_training(args.remote)
            print("\n(Refreshing every 5 minutes... Ctrl+C to stop)\n")
            time.sleep(300)
    elif args.command == "cost":
        estimate_cost(args.hours, args.gpu, args.rate)
    elif args.command == "wandb":
        setup_wandb(args.api_key)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
