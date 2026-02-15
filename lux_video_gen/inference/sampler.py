"""
Sampling strategies for inference.

Different sampling methods trade off speed vs quality:
- Euler: Fast, basic ODE solver
- Euler Ancestral: Adds stochasticity for diversity
- DPM-Solver++: Fast high-quality multi-step solver
- UniPC: Unified predictor-corrector
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class EulerSampler:
    """Euler method ODE solver for flow matching."""

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def get_sigmas(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Get sigma schedule."""
        ramp = torch.linspace(0, 1, num_steps + 1, device=device)
        min_inv = self.sigma_min ** (1 / 7)
        max_inv = self.sigma_max ** (1 / 7)
        sigmas = (max_inv + ramp * (min_inv - max_inv)) ** 7
        return sigmas

    @torch.no_grad()
    def step(
        self,
        model_output: torch.Tensor,
        sigma: float,
        sigma_next: float,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Single Euler step."""
        dt = sigma_next - sigma
        return sample + model_output * dt


class EulerAncestralSampler:
    """Euler ancestral sampler with stochastic noise injection."""

    def __init__(self, eta: float = 1.0):
        self.eta = eta

    @torch.no_grad()
    def step(
        self,
        model_output: torch.Tensor,
        sigma: float,
        sigma_next: float,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        sigma_up = min(
            sigma_next,
            self.eta * (sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2) ** 0.5,
        )
        sigma_down = (sigma_next**2 - sigma_up**2) ** 0.5

        dt = sigma_down - sigma
        x = sample + model_output * dt

        if sigma_up > 0:
            noise = torch.randn(
                x.shape, device=x.device, dtype=x.dtype, generator=generator
            )
            x = x + noise * sigma_up

        return x


class DPMSolverPP:
    """
    DPM-Solver++ (2nd order) for fast high-quality sampling.
    
    Based on "DPM-Solver++: Fast Solver for Guided Sampling of
    Diffusion Probabilistic Models" (Lu et al., 2022).
    """

    def __init__(self, prediction_type: str = "v_prediction"):
        self.prediction_type = prediction_type
        self.prev_output = None

    def reset(self):
        self.prev_output = None

    @torch.no_grad()
    def step(
        self,
        model_output: torch.Tensor,
        timestep: float,
        timestep_prev: float,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """One step of DPM-Solver++ (2nd order)."""
        lambda_t = -math.log(timestep) if timestep > 0 else 0
        lambda_prev = -math.log(timestep_prev) if timestep_prev > 0 else 0
        h = lambda_t - lambda_prev

        if self.prev_output is None:
            # First order (Euler-like)
            x = sample + model_output * (math.exp(h) - 1)
        else:
            # Second order
            r = h / (lambda_t - lambda_prev) if (lambda_t - lambda_prev) != 0 else 0.5
            D0 = model_output
            D1 = (model_output - self.prev_output) / (2 * r) if r != 0 else torch.zeros_like(model_output)
            x = sample + (math.exp(h) - 1) * D0 + (math.exp(h) - 1 - h) * D1

        self.prev_output = model_output
        return x


class UniPCSampler:
    """
    Unified Predictor-Corrector Sampler.
    
    Combines predictor-corrector methods for stable, high-quality
    sampling with fewer steps.
    """

    def __init__(self, order: int = 2):
        self.order = order
        self.history = []

    def reset(self):
        self.history = []

    @torch.no_grad()
    def step(
        self,
        model_output: torch.Tensor,
        timestep: float,
        timestep_next: float,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Single UniPC step."""
        dt = timestep_next - timestep

        # Predictor step
        predicted = sample + model_output * dt

        # Store for multi-step correction
        self.history.append(model_output.clone())
        if len(self.history) > self.order:
            self.history.pop(0)

        # Corrector using history
        if len(self.history) >= 2:
            correction = self.history[-1] - self.history[-2]
            predicted = predicted + 0.5 * dt * correction

        return predicted


def create_sampler(
    sampler_type: str = "euler", **kwargs
) -> object:
    """Factory function for samplers."""
    samplers = {
        "euler": EulerSampler,
        "euler_a": EulerAncestralSampler,
        "dpm_solver_pp": DPMSolverPP,
        "unipc": UniPCSampler,
    }

    if sampler_type not in samplers:
        raise ValueError(f"Unknown sampler: {sampler_type}. Available: {list(samplers.keys())}")

    return samplers[sampler_type](**kwargs)
