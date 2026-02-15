"""
Noise Schedulers for the Lux diffusion model.

Implements multiple diffusion scheduling strategies:
- DDPM (Denoising Diffusion Probabilistic Models)
- DDIM (Denoising Diffusion Implicit Models)  
- Flow Matching (Rectified Flow / Conditional OT)
- v-prediction scheduler

Flow Matching is the primary scheduler, providing:
- Straighter sampling trajectories = fewer steps needed
- Better training stability
- Superior sample quality at low NFE (number of function evaluations)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class FlowMatchingScheduler:
    """
    Flow Matching / Rectified Flow scheduler.
    
    Implements conditional optimal transport flow matching where the
    forward process is a linear interpolation between data and noise:
    
        x_t = (1 - t) * x_0 + t * noise
        v = noise - x_0  (velocity)
    
    This produces straighter sampling paths than standard diffusion,
    requiring fewer steps for the same quality.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = True,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting
        self.base_shift = base_shift
        self.max_shift = max_shift

    def get_sigmas(self, timesteps: torch.Tensor, n_dim: int = 5) -> torch.Tensor:
        """Convert discrete timesteps to continuous sigma values."""
        sigmas = timesteps.float() / self.num_train_timesteps
        
        # Apply shift for better sampling distribution
        if self.shift != 1.0:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        
        # Reshape for broadcasting with n-dimensional tensors
        while sigmas.dim() < n_dim:
            sigmas = sigmas.unsqueeze(-1)
        
        return sigmas

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples using flow matching interpolation.
        
        x_t = (1 - sigma) * x_0 + sigma * noise
        """
        sigmas = self.get_sigmas(timesteps, n_dim=original_samples.dim())
        sigmas = sigmas.to(original_samples.device, original_samples.dtype)
        
        noisy_samples = (1 - sigmas) * original_samples + sigmas * noise
        return noisy_samples

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity target for training.
        
        v = noise - x_0
        """
        return noise - sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        next_timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform one denoising step.
        
        Given predicted velocity v, step from x_t to x_{t-dt}:
        x_{t-dt} = x_t - (sigma_t - sigma_{t-dt}) * v
        """
        sigma = self.get_sigmas(timestep, n_dim=sample.dim())
        sigma = sigma.to(sample.device, sample.dtype)
        
        if next_timestep is not None:
            sigma_next = self.get_sigmas(next_timestep, n_dim=sample.dim())
            sigma_next = sigma_next.to(sample.device, sample.dtype)
        else:
            sigma_next = torch.zeros_like(sigma)
        
        # Euler step: x_{t-1} = x_t - (sigma_t - sigma_{t-1}) * v
        prev_sample = sample - (sigma - sigma_next) * model_output
        return prev_sample

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for training."""
        # Logit-normal sampling for better distribution
        u = torch.randn(batch_size, device=device)
        t = torch.sigmoid(u)  # Map to (0, 1)
        t = (t * self.num_train_timesteps).long().clamp(1, self.num_train_timesteps - 1)
        return t

    def get_sampling_timesteps(
        self,
        num_steps: int = 50,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Get timestep schedule for sampling."""
        timesteps = torch.linspace(
            self.num_train_timesteps, 0, num_steps + 1,
            device=device,
        ).long()
        return timesteps


class DDPMScheduler:
    """
    DDPM noise scheduler with linear or cosine beta schedule.
    
    Standard diffusion scheduler used for comparison and compatibility.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "v_prediction",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

        # Compute beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps
            ) ** 2
        elif beta_schedule == "cosine":
            betas = self._cosine_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def _cosine_schedule(self, T: int, s: float = 0.008) -> torch.Tensor:
        """Cosine beta schedule."""
        steps = torch.arange(T + 1, dtype=torch.float64)
        alpha_bar = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(max=0.999).float()

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise."""
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].to(original_samples.device)
        
        while sqrt_alpha.dim() < original_samples.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute v-prediction target."""
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].to(sample.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].to(sample.device)
        
        while sqrt_alpha.dim() < sample.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * noise - sqrt_one_minus_alpha * sample

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps."""
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)


class DDIMScheduler:
    """
    DDIM scheduler for fast deterministic sampling.
    
    Allows reducing the number of sampling steps significantly
    while maintaining quality.
    """

    def __init__(self, ddpm_scheduler: DDPMScheduler, num_inference_steps: int = 50):
        self.ddpm = ddpm_scheduler
        self.num_inference_steps = num_inference_steps

    def get_sampling_timesteps(
        self,
        num_steps: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Get evenly spaced timesteps."""
        num_steps = num_steps or self.num_inference_steps
        step_ratio = self.ddpm.num_train_timesteps // num_steps
        timesteps = torch.arange(0, self.ddpm.num_train_timesteps, step_ratio)
        timesteps = timesteps.flip(0).to(device)
        return timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        prev_timestep: int,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM sampling step."""
        alpha_t = self.ddpm.alphas_cumprod[timestep].to(sample.device)
        alpha_prev = self.ddpm.alphas_cumprod[prev_timestep].to(sample.device) if prev_timestep >= 0 else torch.tensor(1.0, device=sample.device)

        if self.ddpm.prediction_type == "v_prediction":
            sqrt_alpha = alpha_t ** 0.5
            sqrt_one_minus_alpha = (1 - alpha_t) ** 0.5
            pred_x0 = sqrt_alpha * sample - sqrt_one_minus_alpha * model_output
            pred_noise = sqrt_alpha * model_output + sqrt_one_minus_alpha * sample
        else:
            pred_noise = model_output
            pred_x0 = (sample - (1 - alpha_t) ** 0.5 * pred_noise) / alpha_t ** 0.5

        # DDIM step
        sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)) ** 0.5

        pred_dir = (1 - alpha_prev - sigma ** 2) ** 0.5 * pred_noise
        prev_sample = alpha_prev ** 0.5 * pred_x0 + pred_dir

        if eta > 0:
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + sigma * noise

        return prev_sample


def create_scheduler(
    scheduler_type: str = "flow_matching",
    num_train_timesteps: int = 1000,
    **kwargs,
):
    """Factory function to create a noise scheduler."""
    if scheduler_type == "flow_matching":
        return FlowMatchingScheduler(num_train_timesteps=num_train_timesteps, **kwargs)
    elif scheduler_type == "ddpm":
        return DDPMScheduler(num_train_timesteps=num_train_timesteps, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
