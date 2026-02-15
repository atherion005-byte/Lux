Cloud Training Setup Guide for Lux
===================================

This guide walks you through training Lux on cloud GPUs from your PC.


STEP 1: Choose a Cloud GPU Provider
====================================

Recommended: Lambda Labs (reliable, good for research)
  - Website: https://lambdalabs.com
  - GPU: A40 (48GB VRAM, $0.90/hr) or A100 80GB ($2.50/hr)
  - Setup: Click "Launch Instance" → Ubuntu 22.04 → Done

Alternative options:
  - Paperspace: https://paperspace.com (easy Jupyter interface)
  - Modal: https://modal.com (cheapest, pay-per-minute)
  - Vast.ai: https://vast.ai (community GPUs, risky but cheap)


STEP 2: Connect to Your Cloud Machine
======================================

After launching, Lambda Labs gives you:
  - IP address (e.g., 123.45.67.89)
  - Username (usually "ubuntu")
  - SSH key file (downloaded to your PC)

On your PC, open PowerShell and test SSH:
  ssh -i path/to/key.pem ubuntu@123.45.67.89

You should see: ubuntu@instance-name:~$


STEP 3: Clone & Setup on Cloud
===============================

SSH into the cloud machine, then run:

  cd ~
  git clone https://github.com/YOUR_USERNAME/Lux.git
  cd Lux
  
  # Install PyTorch (CUDA 12.1 for A40)
  pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install -q -r requirements.txt
  
  # Check GPU
  nvidia-smi


STEP 4: Download Dataset on Cloud
==================================

The cloud provider usually has FAST internet (100+ Mbps). Download there!

  python scripts/download_dataset.py \
    --output_dir datasets/openvid \
    --num_parts 20 \
    --extract

This will take ~1-2 hours for 20 parts (~400GB).


STEP 5: Start Training
======================

Run training on the cloud machine:

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
    --project_name "lux-b-cloud-training"

This will run in background. Monitor with:
  tail -f outputs/training.log


STEP 6: Monitor from Your PC
=============================

Option A: Tensorboard (recommended)
  # On cloud machine
  tensorboard --logdir outputs/logs --port 6006 --no-browser
  
  # On your PC, in a new PowerShell terminal
  ssh -i path/to/key.pem -L 6006:localhost:6006 ubuntu@123.45.67.89
  
  # Open browser: http://localhost:6006
  # See training curves in real-time!

Option B: Weights & Biases (easiest)
  # On cloud machine
  pip install wandb
  wandb login  # copypaste API key from https://wandb.ai/authorize
  
  # Then training with --use_wandb flag automatically syncs
  # View at: https://wandb.ai/YOUR_USERNAME/lux-b-cloud-training


STEP 7: Download Checkpoints
=============================

While training, download intermediate checkpoints to your PC:

  # On your PC (PowerShell)
  scp -r -i path/to/key.pem ubuntu@123.45.67.89:~/Lux/checkpoints ./cloud_checkpoints
  
  # Or use the manager script
  python scripts/cloud_manager.py download-checkpoints \
    --remote ubuntu@123.45.67.89:~/Lux/checkpoints \
    --local ./cloud_checkpoints


STEP 8: Resume Training Locally (Optional)
===========================================

If training gets interrupted or you want to fine-tune on your PC:

  python scripts/train.py \
    --stage dit \
    --model_variant B \
    --data_dir datasets/openvid/videos \
    --metadata_file datasets/openvid/metadata.jsonl \
    --resume_from ./cloud_checkpoints/lux_b_step_10000.pt


COST ESTIMATION
===============

Training duration depends on model size and data:

Lux-S (483M params):
  - ~5-10 hours per epoch on A40
  - Full training (10 epochs): ~$50-100

Lux-B (1.6B params):
  - ~15-20 hours per epoch on A40
  - Full training (10 epochs): ~$150-250

Lux-L (2.4B params):
  - ~25-30 hours per epoch on A40
  - Full training (10 epochs): ~$250-400

Formula: hours * $0.90/hr (A40 rate)


IMPORTANT NOTES
===============

1. Stop the instance when not training!
   - Lambda labs charges hourly even if idle
   - Make sure to STOP (not delete) to keep your data

2. Use preemptible GPUs for 50% discount
   - Slightly risky (can be interrupted)
   - Good for non-critical training

3. Keep uploads to minimum
   - Code is small (~50MB)
   - Dataset should be downloaded ON cloud (much faster + cheaper)

4. Data persistence
   - Cloud provider keeps your data in persistent storage
   - All files remain even after instance stops
   - Can resume training anytime

5. Backup checkpoints
   - Download critical checkpoints to your PC regularly
   - Cloud storage can fail


TROUBLESHOOTING
===============

Q: "Out of memory" during training?
A: Reduce batch_size: --batch_size 2 or 1

Q: Training is slow?
A: Check GPU utilization: ssh ...  nvidia-smi -l 1

Q: Lost connection during download?
A: Use screen/tmux to detach training:
   ssh ... 
   tmux
   python scripts/train.py ...
   [Ctrl+B, then D to detach]
   [reconnect later, tmux attach]

Q: Want to use multiple GPUs?
A: Ask cloud provider for multi-GPU node, code supports distributed training


NEXT STEPS
==========

1. Pick a cloud provider
2. Launch a GPU instance
3. SSH in and follow STEP 2-7
4. Download checkpoints as training progresses
5. Test trained model locally: python scripts/generate.py --checkpoint <path>
6. Fine-tune on your PC with full dataset or specific prompts

Good luck! Questions? Check the repo README.
