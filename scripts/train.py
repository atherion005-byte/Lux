#!/usr/bin/env python3
"""
Lux Video Generation - Training Script.

Usage:
    # Train DiT on 2 GPUs (consumer setup)
    python scripts/train.py --stage dit --preset consumer_2gpu --data_dir datasets/videos

    # Train VAE first
    python scripts/train.py --stage vae --data_dir datasets/videos

    # Resume training
    python scripts/train.py --stage dit --resume_from checkpoints/step_5000

    # Custom settings
    python scripts/train.py --stage dit --model_variant S --resolution 256 --batch_size 2
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lux_video_gen.configs.training_config import get_training_config
from lux_video_gen.configs.model_config import get_model_config, VAEConfig, TextEncoderConfig
from lux_video_gen.utils.logging_utils import setup_logging
from lux_video_gen.utils.memory import log_memory_usage, estimate_model_memory


def parse_args():
    parser = argparse.ArgumentParser(description="Lux Video Generation Training")
    
    # Core
    parser.add_argument("--stage", type=str, default="dit", choices=["vae", "dit", "multimodal", "finetune"])
    parser.add_argument("--preset", type=str, default="consumer_2gpu")
    parser.add_argument("--model_variant", type=str, default=None)
    
    # Data
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--metadata_file", type=str, default=None)
    
    # Training
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    
    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--save_every_n_steps", type=int, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=None)
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--project_name", type=str, default="lux-video-gen")
    
    # Hardware
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["fp16", "bf16", "no"])
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save VRAM (slower but ~40%% less memory)")
    parser.add_argument("--offload_models", action="store_true",
                        help="Offload VAE/text encoder to CPU after encoding to save VRAM")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Use 8-bit Adam optimizer to save ~10GB VRAM on optimizer states")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(log_dir=args.output_dir)
    logger.info("=" * 60)
    logger.info("Lux Video Generation - Training")
    logger.info("=" * 60)
    
    # Build config from preset + overrides
    overrides = {}
    for key in ["model_variant", "num_epochs", "batch_size", "learning_rate",
                 "resolution", "num_frames", "gradient_accumulation_steps",
                 "output_dir", "checkpoint_dir", "resume_from",
                 "save_every_n_steps", "log_every_n_steps", "mixed_precision", "data_dir",
                 "metadata_file", "use_wandb", "project_name",
                 "gradient_checkpointing", "offload_models"]:
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val
    overrides["stage"] = args.stage
    
    config = get_training_config(preset=args.preset, **overrides)
    
    logger.info(f"Training stage: {config.stage}")
    logger.info(f"Model variant: {config.model_variant}")
    logger.info(f"Resolution: {config.resolution}")
    logger.info(f"Frames: {config.num_frames}")
    logger.info(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps} accumulation = {config.effective_batch_size}")
    logger.info(f"Mixed precision: {config.mixed_precision}")
    
    # Check hardware
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    else:
        logger.warning("No CUDA GPU detected! Training will be extremely slow.")
    
    # Initialize models
    logger.info("Initializing models...")
    
    from lux_video_gen.models.vae3d import VideoVAE
    from lux_video_gen.models.dit_video import LuxDiT
    from lux_video_gen.models.text_encoder import TextEncoderWrapper
    
    dit_config = get_model_config(config.model_variant)
    
    vae = VideoVAE()
    dit = LuxDiT.from_config(config.model_variant)
    text_encoder = TextEncoderWrapper()
    
    # Cast models to bf16 to halve memory (fp32: 6.4GB -> bf16: 3.2GB for 1.6B params)
    if config.mixed_precision == "bf16":
        dit = dit.to(torch.bfloat16)
        vae = vae.to(torch.bfloat16)
        text_encoder = text_encoder.to(torch.bfloat16)
        logger.info("Models cast to bfloat16 (halves parameter memory)")
    elif config.mixed_precision == "fp16":
        dit = dit.half()
        vae = vae.half()
        text_encoder = text_encoder.half()
        logger.info("Models cast to float16 (halves parameter memory)")
    
    # Enable gradient checkpointing for memory savings
    if getattr(config, 'gradient_checkpointing', False) or getattr(args, 'gradient_checkpointing', False):
        dit.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing ENABLED (~40% VRAM savings, ~30% slower)")
    
    # Log model sizes
    for name, model in [("VAE", vae), ("DiT", dit)]:
        info = estimate_model_memory(model)
        logger.info(f"{name}: {info['total_params']/1e6:.1f}M params, "
                     f"~{info['training_fp16_gb']:.2f} GB for training (fp16)")
    
    # Initialize noise scheduler
    from lux_video_gen.training.scheduler import create_scheduler as create_noise_scheduler
    noise_scheduler = create_noise_scheduler(config.noise_scheduler)
    
    # Initialize dataset
    logger.info("Initializing dataset...")
    from lux_video_gen.data.dataset import create_dataloader
    
    train_dataloader = create_dataloader(
        data_dir=config.data_dir,
        metadata_file=config.metadata_file,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        resolution=config.resolution,
        num_frames=config.num_frames,
        fps=config.fps,
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    from lux_video_gen.training.trainer import LuxTrainer
    
    trainer = LuxTrainer(
        dit_model=dit,
        vae_model=vae,
        text_encoder=text_encoder,
        noise_scheduler=noise_scheduler,
        train_dataloader=train_dataloader,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        num_epochs=config.num_epochs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        use_ema=config.use_ema,
        ema_decay=config.ema_decay,
        output_dir=config.output_dir,
        checkpoint_dir=config.checkpoint_dir,
        save_every_n_steps=config.save_every_n_steps,
        log_every_n_steps=config.log_every_n_steps,
        use_wandb=config.use_wandb,
        use_tensorboard=config.use_tensorboard,
        project_name=config.project_name,
        training_stage=config.stage,
        cfg_dropout_prob=config.cfg_dropout_prob,
        offload_models=getattr(args, 'offload_models', False),
        use_8bit_adam=getattr(args, 'use_8bit_adam', False),
    )
    
    # Resume if specified
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        trainer.load_checkpoint(config.resume_from)
    
    # Train
    if torch.cuda.is_available():
        log_memory_usage("Before training")
    
    trainer.train()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
