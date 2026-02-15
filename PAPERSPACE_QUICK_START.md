PAPERSPACE QUICK START
======================

You're logged in at: https://console.paperspace.com/tkpw4hzq9c/projects


OPTION 1: Use Python Launcher (Automated)
==========================================

1. Install Paperspace CLI:
   pip install paperspace

2. Get your API key:
   - Go to: https://console.paperspace.com/account/api
   - Copy the API key
   - In PowerShell: set PAPERSPACE_API_KEY=your_key_here

3. Launch instance from your PC:
   python scripts/paperspace_launch.py --gpu A4000 --name lux-training-b

4. Script will:
   - Create A4000 instance
   - Wait for it to start
   - Print instructions
   - Estimate cost


OPTION 2: Manual Setup (Easier)
================================

1. Go to: https://console.paperspace.com/tkpw4hzq9c/projects

2. Click "Create New" or "Create Machine"

3. Select:
   - GPU: A4000 (24GB, $0.76/hr)
   - Image: Ubuntu 22.04 + PyTorch
   - Name: lux-training-b

4. Click Create and wait ~2 minutes for startup

5. Once running, click "Open Terminal" or "Open in Jupyter"

6. In the terminal, paste:
   
   git clone https://github.com/YOUR_USERNAME/Lux.git
   cd Lux
   pip install -q -r requirements.txt
   
   # Download dataset (this takes ~1-2 hours)
   python scripts/download_dataset.py --output_dir datasets/openvid --num_parts 20
   
   # Start training
   python scripts/train.py \
     --model_variant B \
     --data_dir datasets/openvid/videos \
     --metadata_file datasets/openvid/metadata.jsonl \
     --resolution 512 \
     --num_frames 16 \
     --batch_size 4 \
     --use_wandb \
     --project_name "lux-b-training"


OPTION 3: Using Jupyter (Easiest for Beginners)
=================================================

1. Create instance (see Option 2, step 1-4)

2. Click "Open in Jupyter"

3. Click "Terminal" tab at bottom

4. Paste the commands from Option 2, step 6

5. Watch training live in the terminal

6. Access files via Jupyter file browser on the left


MONITORING TRAINING FROM YOUR PC
=================================

Weights & Biases (Recommended):
  - When training starts, it syncs to: https://wandb.ai/YOUR_USERNAME
  - Real-time charts, loss curves, video generation quality
  - Works on Paperspace automatically (just use --use_wandb flag)

SSH into instance (if you set up SSH keys):
  ssh -i ~/.ssh/paperspace paperspace@<instance-ip>
  tail -f Lux/outputs/training.log

Download checkpoints to your PC:
  scp -i ~/.ssh/paperspace -r paperspace@<instance-ip>:~/Lux/checkpoints ./

Or use Jupyter file browser to download files one-by-one


COST TRACKING
=============

Your instance charges while RUNNING, not while stopped.

To save costs:
  1. Click the instance
  2. Click "Stop" (NOT "Terminate")
  3. Instance stays but doesn't charge
  4. Can resume later with all data intact

Cost calculator for A4000:
  - Per hour: $0.76
  - Per day (24h): $18.24
  - Per week: $127.68
  - Per month: ~$550

Example training costs:
  - Lux-B, 10 epochs: ~150 hours = ~$114
  - Lux-S, 10 epochs: ~50 hours = ~$38
  - Lux-L, 10 epochs: ~300 hours = ~$228


TROUBLESHOOTING
===============

Q: Instance won't start?
A: Paperspace sometimes has capacity issues. Try a different GPU or region.

Q: Training runs out of memory?
A: Reduce batch size: --batch_size 2

Q: Download is slow?
A: Normal on shared networks. Let it run overnight.

Q: Want to stop training early?
A: Ctrl+C in terminal, then click Stop instance.

Q: Want to resume training?
A: Stop the instance, click Resume, and same files are there.


NEXT STEPS
==========

1. Choose Option 1, 2, or 3 above
2. Follow the instructions
3. Training will start automatically
4. Checkpoint saved every 500 steps
5. Download checkpoints to your PC when done
6. Test locally: python scripts/generate.py --checkpoint <path>


SUPPORT
=======

Paperspace docs: https://docs.paperspace.com
Console: https://console.paperspace.com/tkpw4hzq9c/projects
Chat support in console

Have fun training Lux!
