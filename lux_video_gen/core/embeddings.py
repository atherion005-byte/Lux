"""
Embedding layers for the Lux Diffusion Transformer.

Implements various positional and conditioning embeddings:
- Sinusoidal timestep embeddings for diffusion process
- Patch embeddings for converting video frames to tokens
- Learnable spatial/temporal position embeddings
- Conditioning embeddings for text, image, audio modalities
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding as used in DDPM/DiT.
    
    Converts scalar diffusion timesteps into high-dimensional vectors
    using a fixed frequency sinusoidal encoding, followed by learnable
    linear projections for dimension matching.
    """

    def __init__(self, dim: int, max_period: int = 10000, frequency_shift: float = 1.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.frequency_shift = frequency_shift

        # MLP to project sinusoidal embeddings
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,) tensor of diffusion timesteps
        Returns:
            (B, dim*4) embedding vectors
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
            / half_dim
        )
        args = timesteps[:, None].float() * freqs[None, :] * self.frequency_shift
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return self.mlp(embedding)


class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding for video data.
    
    Converts video tensor (B, C, T, H, W) into a sequence of patch tokens
    by applying a 3D convolution with stride equal to patch size.
    This tokenizes the video into spatial-temporal patches.
    """

    def __init__(
        self,
        in_channels: int = 4,  # Latent channels from VAE
        embed_dim: int = 1152,
        patch_size_t: int = 1,
        patch_size_h: int = 2,
        patch_size_w: int = 2,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = (patch_size_t, patch_size_h, patch_size_w)
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
        """
        Args:
            x: (B, C, T, H, W) video latent tensor

        Returns:
            tokens: (B, num_patches, embed_dim)
            t, h, w: Number of patches in each dimension
        """
        B, C, T, H, W = x.shape
        
        # Ensure dimensions are divisible by patch sizes
        assert T % self.patch_size[0] == 0, f"T={T} not divisible by patch_t={self.patch_size[0]}"
        assert H % self.patch_size[1] == 0, f"H={H} not divisible by patch_h={self.patch_size[1]}"
        assert W % self.patch_size[2] == 0, f"W={W} not divisible by patch_w={self.patch_size[2]}"

        x = self.proj(x)  # (B, embed_dim, t, h, w)
        t = T // self.patch_size[0]
        h = H // self.patch_size[1]
        w = W // self.patch_size[2]
        
        x = rearrange(x, "b d t h w -> b (t h w) d")
        x = self.norm(x)
        return x, t, h, w


class LearnedPositionEmbedding3D(nn.Module):
    """
    Learnable factored 3D position embedding.
    
    Uses separate embeddings for temporal and spatial dimensions,
    combined additively. This is more parameter-efficient than
    a full 3D position embedding while still capturing structure.
    """

    def __init__(
        self,
        dim: int,
        max_t: int = 64,
        max_h: int = 128,
        max_w: int = 128,
    ):
        super().__init__()
        self.temporal_embed = nn.Embedding(max_t, dim)
        self.height_embed = nn.Embedding(max_h, dim)
        self.width_embed = nn.Embedding(max_w, dim)
        
        # Learnable scale factors for each dimension
        self.scale_t = nn.Parameter(torch.ones(1) * 0.1)
        self.scale_h = nn.Parameter(torch.ones(1) * 0.1)
        self.scale_w = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, t: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            t, h, w: Number of patches in temporal, height, width dimensions
        Returns:
            (1, t*h*w, dim) position embedding
        """
        t_indices = torch.arange(t, device=device)
        h_indices = torch.arange(h, device=device)
        w_indices = torch.arange(w, device=device)

        t_emb = self.temporal_embed(t_indices) * self.scale_t  # (t, dim)
        h_emb = self.height_embed(h_indices) * self.scale_h  # (h, dim)
        w_emb = self.width_embed(w_indices) * self.scale_w  # (w, dim)

        # Create 3D grid of position embeddings
        pos = (
            t_emb[:, None, None, :]
            + h_emb[None, :, None, :]
            + w_emb[None, None, :, :]
        )  # (t, h, w, dim)
        
        return rearrange(pos, "t h w d -> 1 (t h w) d")


class ConditioningEmbedding(nn.Module):
    """
    Embedding layer for conditioning signals (resolution, aspect ratio, FPS, etc.).
    
    Encodes scalar or categorical conditioning information into
    vector representations for adaptive normalization.
    """

    def __init__(self, dim: int, num_conditions: int = 4):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            for _ in range(num_conditions)
        ])

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            conditions: (B, num_conditions) tensor of conditioning values
        Returns:
            (B, dim) combined conditioning embedding
        """
        parts = []
        for i, embed in enumerate(self.embeddings):
            parts.append(embed(conditions[:, i:i+1]))
        return sum(parts)


class ModalityEmbedding(nn.Module):
    """
    Embedding to distinguish between different input modalities.
    
    Adds a learnable modality-specific bias to tokens from different
    input types (text, image, video, audio), allowing the model to
    process mixed-modality sequences.
    """

    def __init__(self, dim: int, num_modalities: int = 5):
        super().__init__()
        # Modalities: 0=text, 1=image, 2=video, 3=audio, 4=noise
        self.embed = nn.Embedding(num_modalities, dim)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, modality_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modality_ids: (B, seq_len) integer tensor indicating modality type
        Returns:
            (B, seq_len, dim) modality embeddings
        """
        return self.embed(modality_ids) * self.scale


class FinalLayer(nn.Module):
    """
    Final layer of DiT that projects back to patch space.
    
    Uses adaptive layer norm to modulate the output based on the
    timestep embedding, then projects to the target patch dimensions.
    """

    def __init__(self, dim: int, patch_size_t: int, patch_size_h: int, 
                 patch_size_w: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim * 4, 2 * dim),
        )
        
        total_patch = patch_size_t * patch_size_h * patch_size_w
        self.linear = nn.Linear(dim, total_patch * out_channels)
        
        self.patch_size_t = patch_size_t
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) transformer output tokens
            c: (B, D*4) timestep conditioning embedding
        Returns:
            (B, N, patch_t * patch_h * patch_w * out_channels)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x

    def unpatchify(self, x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        """
        Reconstruct video tensor from patch tokens.
        
        Args:
            x: (B, t*h*w, patch_t*patch_h*patch_w*C) predicted patches
            t, h, w: Number of patches
        Returns:
            (B, C, T, H, W) reconstructed video latent
        """
        pt, ph, pw = self.patch_size_t, self.patch_size_h, self.patch_size_w
        c = self.out_channels
        
        x = rearrange(
            x, "b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)",
            t=t, h=h, w=w, pt=pt, ph=ph, pw=pw, c=c
        )
        return x
