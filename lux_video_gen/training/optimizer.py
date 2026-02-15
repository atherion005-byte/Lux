"""
Optimizer configuration for Lux training.

Implements optimized training configurations for consumer GPU hardware:
- AdamW with weight decay
- Learning rate scheduling (warmup + cosine decay)
- Gradient clipping
- Parameter group separation (different LR for different components)
"""

from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
)


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    lr_scale_position: float = 0.1,
    lr_scale_embedding: float = 0.1,
    no_decay_keywords: Tuple[str, ...] = ("bias", "norm", "embedding"),
) -> List[Dict]:
    """
    Create parameter groups with different learning rates and weight decay.
    
    Separates parameters into groups:
    1. Regular parameters with weight decay
    2. Bias/norm parameters without weight decay
    3. Position/embedding parameters with reduced LR
    """
    decay_params = []
    no_decay_params = []
    position_params = []
    embedding_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(kw in name.lower() for kw in ("pos_embed", "position")):
            position_params.append(param)
        elif any(kw in name.lower() for kw in ("modality_embed", "type_embed")):
            embedding_params.append(param)
        elif any(kw in name.lower() for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay, "lr_scale": 1.0},
        {"params": no_decay_params, "weight_decay": 0.0, "lr_scale": 1.0},
    ]

    if position_params:
        groups.append({
            "params": position_params,
            "weight_decay": 0.0,
            "lr_scale": lr_scale_position,
        })

    if embedding_params:
        groups.append({
            "params": embedding_params,
            "weight_decay": 0.0,
            "lr_scale": lr_scale_embedding,
        })

    return groups


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    optimizer_type: str = "adamw",
    use_8bit: bool = False,
) -> torch.optim.Optimizer:
    """Create optimizer with parameter groups.
    
    Args:
        use_8bit: Use 8-bit Adam from bitsandbytes to save ~10GB VRAM
                  on optimizer states (1.6B model: 12.8GB -> 3.2GB).
    """
    param_groups = get_parameter_groups(model, weight_decay=weight_decay)

    # Apply LR scaling
    for group in param_groups:
        group["lr"] = learning_rate * group.pop("lr_scale", 1.0)

    if use_8bit and optimizer_type == "adamw":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                param_groups, lr=learning_rate, betas=betas, eps=eps
            )
            import logging
            logging.getLogger(__name__).info(
                "Using 8-bit AdamW (saves ~10GB VRAM on optimizer states)"
            )
        except ImportError:
            import logging
            logging.getLogger(__name__).warning(
                "bitsandbytes not installed, falling back to standard AdamW. "
                "Install with: pip install bitsandbytes"
            )
            optimizer = AdamW(param_groups, lr=learning_rate, betas=betas, eps=eps)
    elif optimizer_type == "adamw":
        optimizer = AdamW(param_groups, lr=learning_rate, betas=betas, eps=eps)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int = 1000,
    num_training_steps: int = 100000,
    scheduler_type: str = "cosine",
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler with warmup.
    
    Warmup: linear from 0 to peak LR over warmup_steps
    Main: cosine decay from peak LR to min_lr
    """
    if scheduler_type == "cosine":
        # Warmup + cosine decay
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=num_warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - num_warmup_steps,
            eta_min=optimizer.defaults["lr"] * min_lr_ratio,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[num_warmup_steps],
        )
    elif scheduler_type == "constant_with_warmup":
        def lr_lambda(step):
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

    return scheduler
