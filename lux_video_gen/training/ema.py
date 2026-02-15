"""
Exponential Moving Average (EMA) for model weights.

Maintains a shadow copy of model parameters that is updated
with exponential moving average of the training parameters.
The EMA model typically produces higher quality samples.
"""

from typing import Iterable, Optional, Union

import torch
import torch.nn as nn


class EMAModel:
    """
    Exponential Moving Average of model parameters.
    
    Maintains shadow parameters updated as:
        shadow = decay * shadow + (1 - decay) * param
    
    Features:
    - Warmup period with increasing decay
    - Power-based decay scheduling
    - Save/load support
    - Context manager for using EMA params
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_warmup: bool = True,
        warmup_steps: int = 1000,
        warmup_power: float = 2.0 / 3.0,
        device: Optional[torch.device] = None,
    ):
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_warmup = use_warmup
        self.warmup_steps = warmup_steps
        self.warmup_power = warmup_power
        self.step = 0

        # Create shadow parameters
        self.shadow_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                if device is not None:
                    self.shadow_params[name] = param.data.clone().to(device)
                else:
                    self.shadow_params[name] = param.data.clone()

    def get_decay(self) -> float:
        """Compute current decay value with warmup."""
        if not self.use_warmup:
            return self.decay

        step = max(0, self.step - self.update_after_step)
        if step <= 0:
            return 0.0

        if step < self.warmup_steps:
            decay = 1 - (1 - self.min_decay) * (step / self.warmup_steps) ** self.warmup_power
            return min(decay, self.decay)

        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow parameters with current model parameters."""
        self.step += 1
        decay = self.get_decay()

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name].lerp_(param.data, 1 - decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """Copy EMA parameters to model."""
        for name, param in model.named_parameters():
            if name in self.shadow_params:
                param.data.copy_(self.shadow_params[name])

    @torch.no_grad()
    def store(self, model: nn.Module):
        """Store current model parameters (for temporary swap)."""
        self.stored_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.stored_params[name] = param.data.clone()

    @torch.no_grad()
    def restore(self, model: nn.Module):
        """Restore stored model parameters."""
        if not hasattr(self, "stored_params"):
            return
        for name, param in model.named_parameters():
            if name in self.stored_params:
                param.data.copy_(self.stored_params[name])
        del self.stored_params

    def state_dict(self) -> dict:
        """Save EMA state."""
        return {
            "shadow_params": self.shadow_params,
            "decay": self.decay,
            "step": self.step,
        }

    def load_state_dict(self, state_dict: dict):
        """Load EMA state."""
        self.shadow_params = state_dict["shadow_params"]
        self.decay = state_dict["decay"]
        self.step = state_dict["step"]
