"""
Model Configuration Definitions.

Centralized configuration for all model variants with
hardware-aware defaults for consumer GPUs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class VAEConfig:
    """3D Causal VAE configuration."""
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    base_channels: int = 128
    channel_multipliers: List[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    attention_resolutions: List[int] = field(default_factory=lambda: [16])
    spatial_compression: int = 8
    temporal_compression: int = 4
    dropout: float = 0.0


@dataclass
class DiTConfig:
    """Diffusion Transformer configuration."""
    name: str = "Lux-B"
    
    # Patch embedding
    in_channels: int = 4
    patch_size: int = 2
    patch_size_t: int = 1
    
    # Transformer
    hidden_size: int = 768
    num_heads: int = 12
    depth: int = 12
    joint_depth: int = 4
    mlp_ratio: float = 4.0
    
    # Conditioning
    context_dim: int = 1024
    num_conditions: int = 4
    
    # Generation
    max_temporal_length: int = 64
    max_spatial_length: int = 128
    
    # Regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0
    
    # Memory optimization
    use_checkpoint: bool = True
    use_flash_attention: bool = True


@dataclass 
class TextEncoderConfig:
    """Text encoder configuration."""
    t5_model: str = "google/t5-v1_1-base"
    clip_model: str = "openai/clip-vit-base-patch32"
    max_length: int = 256
    output_dim: int = 1024
    freeze: bool = True


@dataclass
class AudioConfig:
    """Audio model configuration."""
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 8
    max_audio_length: int = 1024


# Pre-defined model configurations
MODEL_CONFIGS: Dict[str, DiTConfig] = {
    "S": DiTConfig(
        name="Lux-S",
        hidden_size=384,
        num_heads=6,
        depth=12,
        joint_depth=2,
    ),
    "B": DiTConfig(
        name="Lux-B",
        hidden_size=768,
        num_heads=12,
        depth=12,
        joint_depth=4,
    ),
    "L": DiTConfig(
        name="Lux-L",
        hidden_size=1024,
        num_heads=16,
        depth=24,
        joint_depth=8,
    ),
    "XL": DiTConfig(
        name="Lux-XL",
        hidden_size=1152,
        num_heads=16,
        depth=28,
        joint_depth=10,
        use_checkpoint=True,
    ),
}


def get_model_config(variant: str = "B") -> DiTConfig:
    """Get model configuration by variant name."""
    if variant not in MODEL_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[variant]
