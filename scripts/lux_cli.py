#!/usr/bin/env python3
"""
Lux Cloud Training CLI
======================
One command to launch, train, and manage your Lux model on cloud GPUs.

Usage:
    # See all commands
    python scripts/lux_cli.py --help
    
    # Launch on Paperspace and start training
    python scripts/lux_cli.py launch --provider paperspace --gpu A4000
    
    # Download dataset locally
    python scripts/lux_cli.py dataset download --num-parts 5
    
    # Start local training
    python scripts/lux_cli.py train local --model S --epochs 5
    
    # Monitor cloud training
    python scripts/lux_cli.py monitor --machine-id <id>
    
    # Download checkpoints from cloud
    python scripts/lux_cli.py download-checkpoints --machine-id <id>
"""
import argparse
import subprocess
import os
import sys
import json
import time
from pathlib import Path


class LuxCLI:
    """Main CLI handler for Lux training."""
    
    def __init__(self):
        self.repo_root = Path(__file__).parent.parent
        self.scripts_dir = self.repo_root / "scripts"
        self.datasets_dir = self.repo_root / "datasets"
        self.checkpoints_dir = self.repo_root / "checkpoints"
        
        os.chdir(self.repo_root)
    
    def run_cmd(self, cmd, description="", show=True):
        """Execute command with optional description."""
        if description and show:
            print(f"\n{description}...")
        if show:
            print(f"$ {cmd}\n")
        result = subprocess.run(cmd, shell=True)
        return result.returncode == 0
    
    # ========== Launch Commands ==========
    
    def cmd_launch(self, args):
        """Launch GPU instance and start training."""
        print("\n" + "="*60)
        print("  Lux Cloud Training - Launch")
        print("="*60)
        
        provider = args.provider.lower()
        gpu = args.gpu
        
        if provider == "paperspace":
            return self.launch_paperspace(gpu, args)
        elif provider == "lambda":
            return self.launch_lambda(gpu, args)
        else:
            print(f"Unknown provider: {provider}")
            return False
    
    def launch_paperspace(self, gpu, args):
        """Launch on Paperspace."""
        print(f"\nProvider:  Paperspace")
        print(f"GPU:       {gpu}")
        print(f"Region:    {args.region}")
        print(f"Name:      {args.name}")
        
        # Check API key
        api_key = os.environ.get("PAPERSPACE_API_KEY")
        if not api_key:
            print("\n❌ PAPERSPACE_API_KEY not set")
            print("Get it from: https://console.paperspace.com/account/api")
            print("Then run: set PAPERSPACE_API_KEY=your_key")
            return False
        
        # Check CLI
        result = subprocess.run("paperspace --version", shell=True, capture_output=True)
        if result.returncode != 0:
            print("\n❌ Paperspace CLI not installed")
            print("Install with: pip install paperspace")
            return False
        
        print("\n✓ Prerequisites OK")
        
        # Launch
        print("\nLaunching instance...")
        cmd = f"""paperspace machines create \\
            --machineType "{gpu}" \\
            --region "{args.region}" \\
            --templateId t0nspur9k \\
            --name "{args.name}" \\
            --billingType hourly"""
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to create instance")
            print(result.stderr)
            return False
        
        print("✓ Instance created!")
        print("\nNext steps:")
        print("1. Open: https://console.paperspace.com/tkpw4hzq9c/projects")
        print("2. Find your instance and click 'Open Terminal'")
        print("3. Run this in the terminal:")
        print(f"\n   git clone https://github.com/YOUR_USERNAME/Lux.git && cd Lux")
        print(f"   python scripts/lux_cli.py train remote --model {args.model} --epochs {args.epochs}")
        
        return True
    
    def launch_lambda(self, gpu, args):
        """Launch on Lambda Labs."""
        print(f"\nProvider:  Lambda Labs")
        print(f"GPU:       {gpu}")
        print("\nManual setup required:")
        print("1. Go to: https://lambdalabs.com/service/gpu-cloud")
        print("2. Launch instance with " + gpu)
        print("3. SSH in when ready: ssh -i key.pem ubuntu@<ip>")
        print("4. Then run: python scripts/lux_cli.py train remote --model S")
        return True
    
    # ========== Dataset Commands ==========
    
    def cmd_dataset(self, args):
        """Manage datasets."""
        print("\n" + "="*60)
        print("  Lux Dataset Manager")
        print("="*60)
        
        if args.action == "download":
            return self.dataset_download(args)
        elif args.action == "status":
            return self.dataset_status(args)
        elif args.action == "list":
            return self.dataset_list(args)
        else:
            print(f"Unknown action: {args.action}")
            return False
    
    def dataset_download(self, args):
        """Download OpenVid-1M dataset."""
        print(f"\nDataset:   OpenVid-1M")
        print(f"Parts:     {args.num_parts}")
        print(f"Output:    {args.output_dir}")
        
        self.datasets_dir.mkdir(exist_ok=True)
        
        cmd = f"""python scripts/download_dataset.py \\
            --output_dir {args.output_dir} \\
            --num_parts {args.num_parts} \\
            --extract"""
        
        return self.run_cmd(cmd, "Downloading dataset")
    
    def dataset_status(self, args):
        """Show dataset download status."""
        output_dir = Path(args.output_dir)
        
        print(f"\nDataset directory: {output_dir}")
        
        if not output_dir.exists():
            print("❌ Directory doesn't exist")
            return False
        
        videos = list((output_dir / "videos").glob("*.mp4")) if (output_dir / "videos").exists() else []
        metadata = (output_dir / "metadata.jsonl").exists()
        
        print(f"Videos:    {len(videos)}")
        print(f"Metadata:  {'✓' if metadata else '✗'}")
        print(f"Size:      {sum(f.stat().st_size for f in output_dir.rglob('*')) / 1e9:.1f} GB")
        
        return True
    
    def dataset_list(self, args):
        """List available datasets."""
        print("\nAvailable datasets:")
        print("  openvid-1m    - 1M videos, high quality (7.5TB total)")
        print("  webvid-10m    - 10M videos (need HF API)")
        print("  panda-70m     - 70M videos (massive)")
        print("\nExample: lux_cli dataset download --output-dir datasets/openvid --num-parts 5")
        return True
    
    # ========== Training Commands ==========
    
    def cmd_train(self, args):
        """Start training."""
        print("\n" + "="*60)
        print("  Lux Training")
        print("="*60)
        
        location = args.location
        
        print(f"\nLocation:  {location}")
        print(f"Model:     Lux-{args.model}")
        print(f"Epochs:    {args.epochs}")
        print(f"Batch:     {args.batch_size}")
        print(f"Resolution: {args.resolution}")
        print(f"Frames:    {args.num_frames}")
        
        if location == "local":
            return self.train_local(args)
        elif location == "remote":
            return self.train_remote(args)
        else:
            print(f"Unknown location: {location}")
            return False
    
    def train_local(self, args):
        """Train locally on your PC."""
        print("\n⚠️  Warning: Training on consumer GPU (2x 8GB)")
        print("   Lux-S only recommended. Lux-B will be slow/tight.")
        
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"\n❌ Data directory not found: {data_dir}")
            print("   Download first: lux_cli dataset download")
            return False
        
        # Find metadata
        metadata_file = data_dir.parent / "metadata.jsonl"
        if not metadata_file.exists():
            metadata_file = None
        
        cmd = f"""python scripts/train.py \\
            --stage dit \\
            --model_variant {args.model} \\
            --data_dir {data_dir / 'videos'} \\
            --metadata_file {metadata_file or 'none'} \\
            --resolution {args.resolution} \\
            --num_frames {args.num_frames} \\
            --batch_size {args.batch_size} \\
            --num_epochs {args.epochs} \\
            --mixed_precision fp16 \\
            {"--use_wandb" if args.use_wandb else ""}"""
        
        return self.run_cmd(cmd, "Starting training")
    
    def train_remote(self, args):
        """Generate training command for remote execution."""
        print("\nRun this command on the remote machine:")
        print()
        
        cmd = f"""python scripts/train.py \\
            --stage dit \\
            --model_variant {args.model} \\
            --data_dir datasets/openvid/videos \\
            --metadata_file datasets/openvid/metadata.jsonl \\
            --resolution {args.resolution} \\
            --num_frames {args.num_frames} \\
            --batch_size {args.batch_size} \\
            --num_epochs {args.epochs} \\
            --mixed_precision fp16 \\
            --use_wandb \\
            --project_name "lux-{args.model}-remote"
"""
        
        print(cmd)
        return True
    
    # ========== Monitoring Commands ==========
    
    def cmd_monitor(self, args):
        """Monitor training."""
        print("\n" + "="*60)
        print("  Lux Training Monitor")
        print("="*60)
        
        print(f"\nMonitoring: {args.machine_id}")
        print("Check: https://wandb.ai/your_username")
        print("\nRefreshing every 30 seconds...")
        
        try:
            while True:
                # Get GPU status
                cmd = f"paperspace machines show --machineId {args.machine_id}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if "running" in result.stdout.lower():
                    print("✓ Instance running")
                else:
                    print("⚠️  Instance not running")
                
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n✗ Monitoring stopped")
            return True
    
    # ========== Checkpoint Commands ==========
    
    def cmd_download_checkpoints(self, args):
        """Download checkpoints from remote."""
        print("\n" + "="*60)
        print("  Download Checkpoints")
        print("="*60)
        
        machine_id = args.machine_id
        remote_user = args.remote_user
        remote_host = args.remote_host
        
        print(f"\nFrom: {remote_user}@{remote_host}")
        print(f"To:   ./checkpoints")
        
        remote_path = f"{remote_user}@{remote_host}:~/Lux/checkpoints"
        
        cmd = f"rsync -av {remote_path}/ ./checkpoints/"
        return self.run_cmd(cmd, "Downloading checkpoints")
    
    # ========== Utility Commands ==========
    
    def cmd_status(self, args):
        """Show project status."""
        print("\n" + "="*60)
        print("  Lux Project Status")
        print("="*60)
        
        # Code
        code_files = list(self.repo_root.glob("lux_video_gen/**/*.py"))
        print(f"\n✓ Code:        {len(code_files)} files")
        
        # Datasets
        datasets = list(self.datasets_dir.glob("*/videos"))
        print(f"{'✓' if datasets else '✗'} Datasets:     {len(datasets)} downloaded")
        
        # Checkpoints
        checkpoints = list(self.checkpoints_dir.glob("*.pt"))
        print(f"{'✓' if checkpoints else '✗'} Checkpoints:  {len(checkpoints)} saved")
        
        # Environment
        import torch
        print(f"\n✓ PyTorch:     {torch.__version__}")
        print(f"✓ CUDA:        {torch.version.cuda}")
        print(f"✓ GPUs:        {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {props.name} ({props.total_memory / 1e9:.1f}GB)")
        
        return True
    
    def cmd_help_providers(self, args):
        """Show cloud provider options."""
        print("\n" + "="*60)
        print("  Cloud GPU Providers")
        print("="*60)
        
        providers = {
            "paperspace": {
                "url": "https://console.paperspace.com",
                "A4000": "$0.76/hr",
                "RTX5000": "$0.82/hr",
                "A6000": "$1.89/hr",
            },
            "lambda": {
                "url": "https://lambdalabs.com/service/gpu-cloud",
                "A40": "$0.90/hr",
                "A100": "$2.50/hr",
            },
            "modal": {
                "url": "https://modal.com",
                "A100": "$0.30-1.20/hr",
                "Info": "Cheapest, pay-per-minute",
            },
        }
        
        for name, info in providers.items():
            print(f"\n{name.upper()}")
            print(f"  URL: {info['url']}")
            for key, val in info.items():
                if key != "url":
                    print(f"  {key}: {val}")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Lux Video Generation - Cloud Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch on Paperspace
  python scripts/lux_cli.py launch --provider paperspace --gpu A4000
  
  # Download dataset locally
  python scripts/lux_cli.py dataset download --num-parts 5
  
  # Train locally
  python scripts/lux_cli.py train local --model S --epochs 5
  
  # Generate training command for cloud
  python scripts/lux_cli.py train remote --model B --epochs 10
  
  # Monitor training
  python scripts/lux_cli.py monitor --machine-id <id>
  
  # Show status
  python scripts/lux_cli.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Launch
    launch_parser = subparsers.add_parser("launch", help="Launch GPU instance and start training")
    launch_parser.add_argument("--provider", default="paperspace", choices=["paperspace", "lambda"])
    launch_parser.add_argument("--gpu", default="A4000", help="GPU type")
    launch_parser.add_argument("--region", default="US-West", help="Cloud region")
    launch_parser.add_argument("--name", default="lux-training", help="Instance name")
    launch_parser.add_argument("--model", default="B", choices=["S", "B", "L"])
    launch_parser.add_argument("--epochs", type=int, default=10)
    
    # Dataset
    dataset_parser = subparsers.add_parser("dataset", help="Manage datasets")
    dataset_parser.add_argument("action", choices=["download", "status", "list"])
    dataset_parser.add_argument("--num-parts", type=int, default=5)
    dataset_parser.add_argument("--output-dir", default="datasets/openvid")
    
    # Train
    train_parser = subparsers.add_parser("train", help="Start training")
    train_parser.add_argument("location", choices=["local", "remote"])
    train_parser.add_argument("--model", default="S", choices=["S", "B", "L"])
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--resolution", type=int, default=512)
    train_parser.add_argument("--num-frames", type=int, default=16)
    train_parser.add_argument("--data-dir", default="datasets/openvid/videos")
    train_parser.add_argument("--use-wandb", action="store_true")
    
    # Monitor
    monitor_parser = subparsers.add_parser("monitor", help="Monitor training")
    monitor_parser.add_argument("--machine-id", required=True)
    
    # Download checkpoints
    dl_parser = subparsers.add_parser("download-checkpoints", help="Download checkpoints")
    dl_parser.add_argument("--machine-id", help="Paperspace machine ID")
    dl_parser.add_argument("--remote-user", default="paperspace")
    dl_parser.add_argument("--remote-host", required=True)
    
    # Status
    subparsers.add_parser("status", help="Show project status")
    
    # Providers
    subparsers.add_parser("providers", help="Show cloud provider options")
    
    args = parser.parse_args()
    
    cli = LuxCLI()
    
    if args.command == "launch":
        return 0 if cli.cmd_launch(args) else 1
    elif args.command == "dataset":
        return 0 if cli.cmd_dataset(args) else 1
    elif args.command == "train":
        return 0 if cli.cmd_train(args) else 1
    elif args.command == "monitor":
        return 0 if cli.cmd_monitor(args) else 1
    elif args.command == "download-checkpoints":
        return 0 if cli.cmd_download_checkpoints(args) else 1
    elif args.command == "status":
        return 0 if cli.cmd_status(args) else 1
    elif args.command == "providers":
        return 0 if cli.cmd_help_providers(args) else 1
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
