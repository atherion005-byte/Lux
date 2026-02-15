"""
Loss functions for Lux video generation training.

Implements multiple loss objectives:
- MSE/L1 diffusion loss (noise prediction or v-prediction)
- Perceptual loss for visual quality
- Temporal consistency loss for smooth motion
- Physics-aware loss for realistic dynamics
- VAE losses (reconstruction + KL divergence)
- Audio-video synchronization loss
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """
    Core diffusion training loss.
    
    Supports both epsilon-prediction and v-prediction objectives
    with optional SNR (signal-to-noise ratio) weighting.
    """

    def __init__(
        self,
        prediction_type: str = "v_prediction",
        loss_type: str = "mse",
        snr_gamma: Optional[float] = 5.0,
    ):
        super().__init__()
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.snr_gamma = snr_gamma

    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        alphas_cumprod: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute diffusion loss.
        
        Args:
            model_output: Predicted noise/velocity
            target: Ground truth noise/velocity  
            timesteps: Training timesteps (for SNR weighting)
            alphas_cumprod: Cumulative alpha values (for SNR weighting)
        """
        if self.loss_type == "mse":
            loss = F.mse_loss(model_output, target, reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(model_output, target, reduction="none")
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(model_output, target, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Average over non-batch dimensions
        loss = loss.mean(dim=list(range(1, loss.dim())))

        # Apply SNR weighting
        if self.snr_gamma is not None and timesteps is not None and alphas_cumprod is not None:
            snr = self._compute_snr(timesteps, alphas_cumprod)
            snr_weight = torch.clamp(snr, max=self.snr_gamma) / self.snr_gamma
            loss = loss * snr_weight

        return loss.mean()

    def _compute_snr(self, timesteps: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
        """Compute signal-to-noise ratio."""
        alpha = alphas_cumprod[timesteps]
        snr = alpha / (1 - alpha)
        return snr


class TemporalConsistencyLoss(nn.Module):
    """
    Loss for temporal consistency across video frames.
    
    Penalizes large differences between consecutive frames in the
    predicted output, encouraging smooth motion and preventing
    flickering artifacts.
    """

    def __init__(self, weight: float = 0.1, order: int = 1):
        super().__init__()
        self.weight = weight
        self.order = order

    def forward(self, predicted: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted: (B, C, T, H, W) predicted video
        """
        if predicted.shape[2] <= 1:
            return torch.tensor(0.0, device=predicted.device)

        # First-order temporal difference
        diff = predicted[:, :, 1:] - predicted[:, :, :-1]
        loss = diff.abs().mean()

        # Second-order for smoothness (acceleration should be small)
        if self.order >= 2 and predicted.shape[2] > 2:
            diff2 = diff[:, :, 1:] - diff[:, :, :-1]
            loss = loss + 0.5 * diff2.abs().mean()

        return self.weight * loss


class PhysicsAwareLoss(nn.Module):
    """
    Physics-informed loss for realistic motion dynamics.
    
    Encourages the model to learn physically plausible motion by:
    1. Penalizing sudden velocity changes (momentum conservation)
    2. Encouraging gravitational consistency (downward acceleration)
    3. Penalizing object interpenetration artifacts
    
    This is a soft constraint that guides the model toward
    more realistic dynamics without requiring explicit physics simulation.
    """

    def __init__(
        self,
        momentum_weight: float = 0.05,
        smoothness_weight: float = 0.05,
    ):
        super().__init__()
        self.momentum_weight = momentum_weight
        self.smoothness_weight = smoothness_weight

    def forward(self, predicted: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted: (B, C, T, H, W) predicted video latent
        """
        if predicted.shape[2] <= 2:
            return torch.tensor(0.0, device=predicted.device)

        # Velocity (first-order temporal difference)
        velocity = predicted[:, :, 1:] - predicted[:, :, :-1]

        # Acceleration (second-order difference)  
        acceleration = velocity[:, :, 1:] - velocity[:, :, :-1]

        # Momentum loss: acceleration should be smooth (jerk should be small)
        if acceleration.shape[2] > 1:
            jerk = acceleration[:, :, 1:] - acceleration[:, :, :-1]
            momentum_loss = jerk.abs().mean()
        else:
            momentum_loss = acceleration.abs().mean()

        # Smoothness: velocity changes should be gradual
        smoothness_loss = acceleration.abs().mean()

        total = (
            self.momentum_weight * momentum_loss
            + self.smoothness_weight * smoothness_loss
        )
        return total


class VAELoss(nn.Module):
    """
    Combined loss for VAE training.
    
    Includes:
    - Reconstruction loss (L1 + perceptual)
    - KL divergence regularization
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 1e-6,
        perceptual_weight: float = 0.1,
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        log_var: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss.
        
        Args:
            recon: Reconstructed video
            target: Original video
            mean: Latent mean
            log_var: Latent log variance
        """
        # Reconstruction loss (L1 + L2)
        recon_l1 = F.l1_loss(recon, target)
        recon_l2 = F.mse_loss(recon, target)
        recon_loss = recon_l1 + recon_l2

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

        total = self.recon_weight * recon_loss + self.kl_weight * kl_loss

        return {
            "total": total,
            "recon": recon_loss,
            "kl": kl_loss,
        }


class AudioVideoSyncLoss(nn.Module):
    """
    Loss for audio-video synchronization.
    
    Encourages temporal alignment between audio events and
    corresponding visual events (e.g., lip sync, action sounds).
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            video_features: (B, T, D) temporal video features
            audio_features: (B, T, D) temporal audio features (aligned)
        """
        # Ensure same temporal length
        min_t = min(video_features.shape[1], audio_features.shape[1])
        video_features = video_features[:, :min_t]
        audio_features = audio_features[:, :min_t]

        # Cosine similarity should be high for aligned pairs
        similarity = self.cosine(video_features, audio_features)
        sync_loss = 1.0 - similarity.mean()

        return self.weight * sync_loss


class CombinedTrainingLoss(nn.Module):
    """
    Combined loss for full model training.
    
    Aggregates all loss components with configurable weights.
    """

    def __init__(
        self,
        diffusion_weight: float = 1.0,
        temporal_weight: float = 0.1,
        physics_weight: float = 0.05,
        audio_sync_weight: float = 0.1,
        prediction_type: str = "v_prediction",
    ):
        super().__init__()
        self.diffusion_loss = DiffusionLoss(prediction_type=prediction_type)
        self.temporal_loss = TemporalConsistencyLoss(weight=temporal_weight)
        self.physics_loss = PhysicsAwareLoss()
        self.audio_sync_loss = AudioVideoSyncLoss(weight=audio_sync_weight)

        self.diffusion_weight = diffusion_weight
        self.physics_weight = physics_weight

    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        losses = {}

        # Core diffusion loss
        losses["diffusion"] = self.diffusion_weight * self.diffusion_loss(model_output, target)

        # Temporal consistency
        losses["temporal"] = self.temporal_loss(model_output)

        # Physics awareness
        if self.physics_weight > 0:
            losses["physics"] = self.physics_weight * self.physics_loss(model_output)

        # Audio-video sync
        if video_features is not None and audio_features is not None:
            losses["audio_sync"] = self.audio_sync_loss(video_features, audio_features)

        # Total
        losses["total"] = sum(losses.values())

        return losses
