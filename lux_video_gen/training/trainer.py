"""
Main Training Loop for Lux Video Generation.

Implements a complete training pipeline with:
- Multi-stage training (VAE -> DiT -> Multi-modal -> Fine-tuning)
- Mixed precision (FP16/BF16) training
- Multi-GPU support via Accelerate
- Gradient accumulation for effective large batch sizes
- Checkpoint saving and resuming
- Logging to Wandb/Tensorboard
- EMA model maintenance
"""

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
# Using torch.amp directly (modern API)
from torch.utils.data import DataLoader

from .ema import EMAModel
from .losses import CombinedTrainingLoss, DiffusionLoss, VAELoss
from .optimizer import create_optimizer, create_scheduler
from .scheduler import FlowMatchingScheduler, create_scheduler as create_noise_scheduler

logger = logging.getLogger(__name__)


class LuxTrainer:
    """
    Main trainer for the Lux video generation model.
    
    Handles the complete training lifecycle including:
    - Mixed precision training with automatic loss scaling
    - Multi-GPU training with gradient accumulation
    - Checkpoint management and training resumption
    - Logging and visualization
    - EMA parameter maintenance
    """

    def __init__(
        self,
        dit_model: nn.Module,
        vae_model: nn.Module,
        text_encoder: nn.Module,
        noise_scheduler: Optional[Any] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        # Optimizer settings
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        # Training settings
        num_epochs: int = 100,
        gradient_accumulation_steps: int = 4,
        mixed_precision: str = "fp16",  # "fp16", "bf16", "no"
        # EMA settings
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        # Checkpoint settings
        output_dir: str = "outputs",
        checkpoint_dir: str = "checkpoints",
        save_every_n_steps: int = 5000,
        # Logging settings
        log_every_n_steps: int = 100,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        project_name: str = "lux-video-gen",
        # Training stage
        training_stage: str = "dit",  # "vae", "dit", "multimodal", "finetune"
        # Guidance
        cfg_dropout_prob: float = 0.1,
    ):
        self.dit_model = dit_model
        self.vae_model = vae_model
        self.text_encoder = text_encoder
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.use_ema = use_ema
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every_n_steps = save_every_n_steps
        self.log_every_n_steps = log_every_n_steps
        self.training_stage = training_stage
        self.cfg_dropout_prob = cfg_dropout_prob
        self.global_step = 0

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup noise scheduler
        self.noise_scheduler = noise_scheduler or FlowMatchingScheduler()

        # Setup devices
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_device_placement()

        # Setup optimizer (only for trainable model)
        if training_stage == "vae":
            self.trainable_model = self.vae_model
        else:
            self.trainable_model = self.dit_model
        
        self.optimizer = create_optimizer(
            self.trainable_model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        # Setup loss
        if training_stage == "vae":
            self.loss_fn = VAELoss()
        else:
            self.loss_fn = CombinedTrainingLoss(prediction_type="v_prediction")

        # Setup mixed precision
        self.scaler = torch.amp.GradScaler('cuda', enabled=(mixed_precision == "fp16"))
        self.autocast_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "no": torch.float32,
        }.get(mixed_precision, torch.float16)

        # Setup EMA
        if use_ema:
            self.ema = EMAModel(self.trainable_model, decay=ema_decay)
        else:
            self.ema = None

        # Setup logging
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))
            except ImportError:
                logger.warning("TensorBoard not available")

        if use_wandb:
            try:
                import wandb
                wandb.init(project=project_name, config={
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "mixed_precision": mixed_precision,
                    "training_stage": training_stage,
                })
            except ImportError:
                logger.warning("wandb not available")
                self.use_wandb = False

    def _setup_device_placement(self):
        """Place models on available GPUs."""
        num_gpus = torch.cuda.device_count()

        if num_gpus >= 2:
            # GPU 0: DiT model (primary compute)
            # GPU 1: VAE + Text encoder (auxiliary)
            logger.info(f"Using {num_gpus} GPUs for model parallelism")
            self.dit_device = torch.device("cuda:0")
            self.aux_device = torch.device("cuda:1")

            self.dit_model = self.dit_model.to(self.dit_device)
            self.vae_model = self.vae_model.to(self.aux_device)
            self.text_encoder = self.text_encoder.to(self.aux_device)
        elif num_gpus == 1:
            logger.info("Using single GPU")
            self.dit_device = self.aux_device = torch.device("cuda:0")
            self.dit_model = self.dit_model.to(self.dit_device)
            self.vae_model = self.vae_model.to(self.aux_device)
            self.text_encoder = self.text_encoder.to(self.aux_device)
        else:
            logger.info("Using CPU (training will be very slow)")
            self.dit_device = self.aux_device = torch.device("cpu")

    def train_step_dit(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single training step for DiT model.
        
        1. Encode video to latent space using VAE
        2. Encode text prompt using text encoder
        3. Sample noise and timestep
        4. Add noise to latent
        5. Predict noise/velocity with DiT
        6. Compute loss
        """
        video = batch["video"].to(self.aux_device)
        captions = batch["caption"]

        # Encode video to latent space (no grad for VAE)
        with torch.no_grad():
            latent = self.vae_model.encode_video(video)

        # Move latent to DiT device
        latent = latent.to(self.dit_device)

        # Encode text (no grad for text encoder)
        with torch.no_grad():
            text_emb, text_mask = self.text_encoder(captions, device=self.aux_device)
            text_emb = text_emb.to(self.dit_device)
            text_mask = text_mask.to(self.dit_device)

        # CFG dropout: randomly drop conditioning
        if self.training and self.cfg_dropout_prob > 0:
            drop_mask = torch.rand(latent.shape[0]) < self.cfg_dropout_prob
            if drop_mask.any():
                text_emb[drop_mask] = 0.0
                text_mask[drop_mask] = 0

        # Sample noise and timestep
        noise = torch.randn_like(latent)
        timesteps = self.noise_scheduler.sample_timesteps(
            latent.shape[0], device=self.dit_device
        )

        # Add noise
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)

        # Get target (velocity for flow matching)
        target = self.noise_scheduler.get_velocity(latent, noise, timesteps)

        # Build conditions tensor
        conditions = torch.stack([
            batch["height"],
            batch["width"],
            batch["fps"],
            batch["duration"],
        ], dim=-1).to(self.dit_device)

        # Forward pass
        with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
            output = self.dit_model(
                noisy_latent,
                timesteps,
                context=text_emb,
                context_mask=text_mask,
                conditions=conditions,
            )
            pred = output["sample"]

            # Compute loss
            losses = self.loss_fn(pred, target, timesteps)

        loss = losses["total"] if isinstance(losses, dict) else losses
        return {"loss": loss, **(losses if isinstance(losses, dict) else {})}

    def train_step_vae(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step for VAE model."""
        video = batch["video"].to(self.aux_device)

        with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
            recon, mean, log_var = self.vae_model(video)
            losses = self.loss_fn(recon, video, mean, log_var)

        return losses

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.trainable_model.train()
        epoch_losses = {}
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_dataloader):
            # Skip None batches (from collate filtering)
            if batch is None:
                continue
                
            # Select training step based on stage
            if self.training_stage == "vae":
                losses = self.train_step_vae(batch)
            else:
                losses = self.train_step_dit(batch)

            loss = losses["total"] if isinstance(losses, dict) else losses

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Update weights after accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.trainable_model.parameters(),
                        self.max_grad_norm,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Update EMA
                if self.ema is not None:
                    self.ema.update(self.trainable_model)

                self.global_step += 1

                # Logging
                if self.global_step % self.log_every_n_steps == 0:
                    self._log_metrics(losses, epoch, batch_idx)

                # Checkpoint saving
                if self.global_step % self.save_every_n_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

            # Accumulate epoch losses
            if isinstance(losses, dict):
                for k, v in losses.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    epoch_losses[k] = epoch_losses.get(k, 0) + val
            else:
                epoch_losses["total"] = epoch_losses.get("total", 0) + loss.item()
            num_batches += 1

        # Average epoch losses
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)

        return epoch_losses

    def train(self):
        """Main training loop."""
        logger.info(f"Starting {self.training_stage} training for {self.num_epochs} epochs")
        logger.info(f"  Device: {self.dit_device}")
        logger.info(f"  Mixed precision: {self.mixed_precision}")
        logger.info(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        
        # Setup LR scheduler
        if self.train_dataloader is not None:
            total_steps = len(self.train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
        else:
            total_steps = 100000
        
        lr_scheduler = create_scheduler(
            self.optimizer,
            num_warmup_steps=min(1000, total_steps // 10),
            num_training_steps=total_steps,
        )

        best_loss = float("inf")

        for epoch in range(self.num_epochs):
            epoch_start = time.time()

            # Train
            self.training = True
            train_losses = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start

            # Step LR scheduler
            lr_scheduler.step()

            # Log epoch summary
            loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in train_losses.items())
            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} | {loss_str} | "
                f"Time: {epoch_time:.1f}s | LR: {lr_scheduler.get_last_lr()[0]:.2e}"
            )

            # Save best model
            total_loss = train_losses.get("total", train_losses.get("loss", float("inf")))
            if total_loss < best_loss:
                best_loss = total_loss
                self.save_checkpoint("best")

            # Save epoch checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}")

        # Final save
        self.save_checkpoint("final")
        logger.info("Training complete!")

    def _log_metrics(self, losses: Dict, epoch: int, batch_idx: int):
        """Log training metrics."""
        if isinstance(losses, dict):
            loss_str = " | ".join(
                f"{k}: {v.item() if isinstance(v, torch.Tensor) else v:.4f}"
                for k, v in losses.items()
            )
        else:
            loss_str = f"loss: {losses.item():.4f}"
        
        logger.info(f"Step {self.global_step} (Epoch {epoch}, Batch {batch_idx}) | {loss_str}")

        # Tensorboard
        if self.tb_writer:
            if isinstance(losses, dict):
                for k, v in losses.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    self.tb_writer.add_scalar(f"train/{k}", val, self.global_step)
            self.tb_writer.add_scalar(
                "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
            )

    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{name}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        state = {
            "global_step": self.global_step,
            "model_state_dict": self.trainable_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }

        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()

        torch.save(state, checkpoint_path / "training_state.pt")

        # Save model weights separately for easy loading
        torch.save(
            self.trainable_model.state_dict(),
            checkpoint_path / "model.pt",
        )

        if self.ema is not None:
            # Save EMA model weights
            self.ema.store(self.trainable_model)
            self.ema.copy_to(self.trainable_model)
            torch.save(
                self.trainable_model.state_dict(),
                checkpoint_path / "model_ema.pt",
            )
            self.ema.restore(self.trainable_model)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        path = Path(checkpoint_path)
        state_file = path / "training_state.pt"

        if state_file.exists():
            state = torch.load(state_file, map_location="cpu", weights_only=False)
            self.global_step = state["global_step"]
            self.trainable_model.load_state_dict(state["model_state_dict"])
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.scaler.load_state_dict(state["scaler_state_dict"])

            if self.ema is not None and "ema_state_dict" in state:
                self.ema.load_state_dict(state["ema_state_dict"])

            logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")
        elif (path / "model.pt").exists():
            state_dict = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
            self.trainable_model.load_state_dict(state_dict)
            logger.info(f"Loaded model weights from {path}")
        else:
            raise FileNotFoundError(f"No checkpoint found at {path}")
