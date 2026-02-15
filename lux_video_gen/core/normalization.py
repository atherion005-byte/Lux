"""
Normalization layers for the Lux Diffusion Transformer.

Implements various normalization schemes including:
- Adaptive Layer Normalization (adaLN) with modulated scale/shift
- RMS Normalization for efficient computation  
- Adaptive Group Normalization for convolutional layers
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (adaLN-Zero) as used in DiT.
    
    Modulates the normalization with learnable scale and shift parameters
    that are conditioned on the timestep embedding. This allows the model
    to dynamically adjust normalization behavior based on the diffusion timestep.
    
    The 'Zero' variant initializes the output gate to zero, ensuring
    the block acts as an identity function at initialization.
    """

    def __init__(self, dim: int, cond_dim: Optional[int] = None):
        super().__init__()
        cond_dim = cond_dim or dim * 4
        
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # Project conditioning to 6 modulation parameters:
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim),
        )
        
        # Initialize to zero for identity at init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Args:
            x: (B, N, D) input tensor
            c: (B, cond_dim) conditioning vector (timestep embedding)
            
        Returns:
            Tuple of modulated x and 6 modulation parameters
        """
        modulation = self.adaLN_modulation(c)  # (B, 6*D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            modulation.chunk(6, dim=-1)
        )
        
        return (
            shift_msa, scale_msa, gate_msa,
            shift_mlp, scale_mlp, gate_mlp,
        )

    def modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Apply adaptive normalization with shift and scale."""
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaptiveLayerNormContinuous(nn.Module):
    """
    Continuous adaptive layer normalization for conditioning on 
    continuous signals (e.g., resolution, aspect ratio).
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) or (B, D) input
            cond: (B, cond_dim) conditioning
        """
        scale, shift = self.linear(cond).chunk(2, dim=-1)
        if x.dim() == 3:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    
    More efficient alternative to LayerNorm that normalizes by
    the RMS of the input, without mean subtraction.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class AdaptiveGroupNorm(nn.Module):
    """
    Adaptive Group Normalization for convolutional layers.
    
    Used in the VAE encoder/decoder for conditioning on timestep
    or other continuous signals.
    """

    def __init__(self, num_groups: int, num_channels: int, cond_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(cond_dim, 2 * num_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, ...) feature map
            cond: (B, cond_dim) conditioning
        """
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        
        # Reshape for broadcasting over spatial dims
        extra_dims = x.dim() - 2
        for _ in range(extra_dims):
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
        
        return self.norm(x) * (1 + scale) + shift


class QKNorm(nn.Module):
    """
    Query-Key normalization for attention stability.
    
    Normalizes Q and K independently before attention computation,
    preventing attention logits from growing too large during training.
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        return self.q_norm(q), self.k_norm(k)
