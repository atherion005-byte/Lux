"""
3D Causal Video VAE (Variational Autoencoder)

Encodes video into a compressed latent space and decodes back.
Uses causal 3D convolutions for temporal consistency,ensuring each frame
can only attend to previous frames during encoding/decoding.

Key features:
- Causal 3D convolutions (no future frame leakage)
- Spatial downsampling 8x, temporal downsampling 4x  
- KL divergence regularization for smooth latent space
- Supports variable-length video encoding
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CausalConv3d(nn.Module):
    """
    Causal 3D convolution that only attends to current and past frames.
    Pads the temporal dimension asymmetrically to prevent future leakage.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        temporal_kernel: int = 3,
        temporal_stride: int = 1,
    ):
        super().__init__()
        self.temporal_kernel = temporal_kernel
        self.temporal_stride = temporal_stride
        # Causal temporal padding: pad only on the left side
        self.temporal_pad = temporal_kernel - 1
        
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(temporal_kernel, kernel_size, kernel_size),
            stride=(temporal_stride, stride, stride),
            padding=(0, padding, padding),  # No temporal padding in conv
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad temporal dimension causally (only past)
        x = F.pad(x, (0, 0, 0, 0, self.temporal_pad, 0))
        return self.conv(x)


class ResBlock3D(nn.Module):
    """Residual block with causal 3D convolutions."""

    def __init__(self, channels: int, out_channels: Optional[int] = None,
                 dropout: float = 0.0):
        super().__init__()
        out_channels = out_channels or channels
        
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = CausalConv3d(channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        
        if channels != out_channels:
            self.shortcut = nn.Conv3d(channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class SpatialDownsample(nn.Module):
    """Downsample spatial dimensions by 2x using strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3),
                              stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TemporalDownsample(nn.Module):
    """Downsample temporal dimension by 2x using causal strided conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = CausalConv3d(channels, channels, kernel_size=3, padding=1,
                                  temporal_kernel=3, temporal_stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SpatialUpsample(nn.Module):
    """Upsample spatial dimensions by 2x."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3),
                              stride=1, padding=(0, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        return self.conv(x)


class TemporalUpsample(nn.Module):
    """Upsample temporal dimension by 2x."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = CausalConv3d(channels, channels, kernel_size=3, padding=1,
                                  temporal_kernel=3, temporal_stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(2, 1, 1), mode="nearest")
        return self.conv(x)


class SpatialTemporalAttention3D(nn.Module):
    """Lightweight attention for the VAE bottleneck."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, T * H * W)
        q, k, v = qkv.unbind(1)
        
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(B, C, T, H, W)
        return x + self.proj(out)


class Encoder3D(nn.Module):
    """
    3D Video Encoder.
    
    Progressively downsamples video both spatially and temporally
    while increasing channel count.
    
    Compression: 8x spatial, 4x temporal -> latent is (B, latent_dim, T/4, H/8, W/8)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        latent_dim: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Initial convolution
        self.conv_in = CausalConv3d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            block = nn.ModuleList()
            
            for _ in range(num_res_blocks):
                block.append(ResBlock3D(channels, out_ch, dropout=dropout))
                channels = out_ch
            
            # Add downsampling (except last block)
            if i < len(channel_multipliers) - 1:
                block.append(SpatialDownsample(channels))
                if i < 2:  # Temporal downsampling only in first 2 levels
                    block.append(TemporalDownsample(channels))
            
            self.down_blocks.append(block)
        
        # Bottleneck
        self.mid_block = nn.ModuleList([
            ResBlock3D(channels, channels, dropout=dropout),
            SpatialTemporalAttention3D(channels),
            ResBlock3D(channels, channels, dropout=dropout),
        ])
        
        # Output
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = CausalConv3d(channels, 2 * latent_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode video to latent space.
        
        Args:
            x: (B, C, T, H, W) video tensor, pixel values in [-1, 1]
        Returns:
            (B, 2*latent_dim, T', H', W') - mean and log_var concatenated
        """
        h = self.conv_in(x)
        
        for block in self.down_blocks:
            for layer in block:
                h = layer(h)
        
        for layer in self.mid_block:
            h = layer(h)
        
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        return h


class Decoder3D(nn.Module):
    """
    3D Video Decoder.
    
    Progressively upsamples latent representation back to pixel space.
    Mirror architecture of the encoder.
    """

    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        latent_dim: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Compute initial channels (reverse of encoder)
        channels = base_channels * channel_multipliers[-1]
        
        # Initial convolution from latent space
        self.conv_in = CausalConv3d(latent_dim, channels, kernel_size=3, padding=1)
        
        # Bottleneck
        self.mid_block = nn.ModuleList([
            ResBlock3D(channels, channels, dropout=dropout),
            SpatialTemporalAttention3D(channels),
            ResBlock3D(channels, channels, dropout=dropout),
        ])
        
        # Upsampling blocks (reverse order)
        self.up_blocks = nn.ModuleList()
        reversed_mults = list(reversed(channel_multipliers))
        
        for i, mult in enumerate(reversed_mults):
            out_ch = base_channels * mult
            block = nn.ModuleList()
            
            for j in range(num_res_blocks + 1):
                in_ch = channels if j == 0 else out_ch
                block.append(ResBlock3D(in_ch, out_ch, dropout=dropout))
            channels = out_ch
            
            # Add upsampling (except last block)
            if i < len(reversed_mults) - 1:
                block.append(SpatialUpsample(channels))
                if i >= len(reversed_mults) - 3:  # Match temporal downsampling
                    block.append(TemporalUpsample(channels))
            
            self.up_blocks.append(block)
        
        # Output
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = CausalConv3d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to video.
        
        Args:
            z: (B, latent_dim, T', H', W') latent tensor
        Returns:
            (B, C, T, H, W) reconstructed video
        """
        h = self.conv_in(z)
        
        for layer in self.mid_block:
            h = layer(h)
        
        for block in self.up_blocks:
            for layer in block:
                h = layer(h)
        
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        return h


class VideoVAE(nn.Module):
    """
    Complete 3D Video VAE with encoder, decoder, and KL regularization.
    
    Compresses video from pixel space to a compact latent representation:
    - Input: (B, 3, T, H, W) with T frames at HxW resolution
    - Latent: (B, 4, T/4, H/8, W/8) 
    - Output: (B, 3, T, H, W) reconstructed video
    
    The latent space is regularized with KL divergence to ensure
    smoothness, enabling high-quality generation through diffusion.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 4,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        scaling_factor: float = 0.18215,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.scaling_factor = scaling_factor
        
        self.encoder = Encoder3D(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        
        self.decoder = Decoder3D(
            out_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        
        # Quant conv layers (match Stable Diffusion convention)
        self.quant_conv = nn.Conv3d(2 * latent_dim, 2 * latent_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(latent_dim, latent_dim, kernel_size=1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode video to latent distribution parameters.
        
        Returns:
            mean: (B, latent_dim, T', H', W')
            log_var: (B, latent_dim, T', H', W')
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, log_var = h.chunk(2, dim=1)
        return mean, log_var

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample from the latent distribution using reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to video."""
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode -> reparameterize -> decode.
        
        Returns:
            recon: Reconstructed video
            mean: Latent mean
            log_var: Latent log variance
        """
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        z_scaled = z * self.scaling_factor
        recon = self.decode(z_scaled / self.scaling_factor)
        return recon, mean, log_var

    @torch.no_grad()
    def encode_video(self, x: torch.Tensor) -> torch.Tensor:
        """Encode video to scaled latent (for diffusion training/inference)."""
        mean, _ = self.encode(x)
        return mean * self.scaling_factor

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode scaled latent to video (for inference)."""
        z = z / self.scaling_factor
        return self.decode(z)

    def kl_loss(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss."""
        return -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
