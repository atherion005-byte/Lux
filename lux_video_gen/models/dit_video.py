"""
Diffusion Transformer (DiT) for Video Generation - The main model.

This is the core generative model that performs denoising in the latent space
of the 3D VAE. It uses a transformer architecture with:

- Factored spatial-temporal attention blocks
- Adaptive Layer Norm conditioning on diffusion timesteps
- Multi-modal cross-attention for text/image/audio/video conditioning
- Classifier-free guidance support
- Flow matching / v-prediction training objectives

Architecture inspired by DiT (Peebles & Xie 2023), PixArt-Alpha,
Open-Sora, and CogVideoX, adapted for our Lux framework.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..core.dit_block import DiTBlock, DiTBlockJoint
from ..core.embeddings import (
    ConditioningEmbedding,
    FinalLayer,
    LearnedPositionEmbedding3D,
    ModalityEmbedding,
    PatchEmbed3D,
    SinusoidalTimestepEmbedding,
)
from ..core.normalization import AdaptiveLayerNormContinuous


class LuxDiT(nn.Module):
    """
    Lux Diffusion Transformer for video generation.
    
    Takes noisy video latents and conditioning inputs, predicts the noise
    (or velocity) for the diffusion denoising process.
    
    Config sizes:
    - Lux-S: 12 layers, 768 dim, 12 heads   (~200M params)
    - Lux-B: 24 layers, 1024 dim, 16 heads  (~600M params)  
    - Lux-L: 28 layers, 1152 dim, 16 heads  (~900M params)
    - Lux-XL: 32 layers, 1536 dim, 24 heads (~2B params)
    """

    # Predefined model configurations
    CONFIGS = {
        "Lux-S": {"depth": 12, "dim": 768, "num_heads": 12, "mlp_ratio": 4.0},
        "Lux-B": {"depth": 24, "dim": 1024, "num_heads": 16, "mlp_ratio": 4.0},
        "Lux-L": {"depth": 28, "dim": 1152, "num_heads": 16, "mlp_ratio": 4.0},
        "Lux-XL": {"depth": 32, "dim": 1536, "num_heads": 24, "mlp_ratio": 4.0},
    }

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: Optional[int] = None,
        dim: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        patch_size_t: int = 1,
        patch_size_h: int = 2,
        patch_size_w: int = 2,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        cross_attention_dim: int = 4096,
        use_joint_blocks: bool = False,
        num_joint_blocks: int = 0,
        use_gradient_checkpointing: bool = False,
        max_temporal_length: int = 64,
        max_spatial_length: int = 128,
        prediction_type: str = "v_prediction",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.prediction_type = prediction_type
        self.use_joint_blocks = use_joint_blocks
        
        # === Patch Embedding ===
        self.patch_embed = PatchEmbed3D(
            in_channels=in_channels,
            embed_dim=dim,
            patch_size_t=patch_size_t,
            patch_size_h=patch_size_h,
            patch_size_w=patch_size_w,
        )
        
        # === Position Embedding ===
        self.pos_embed = LearnedPositionEmbedding3D(
            dim=dim,
            max_t=max_temporal_length,
            max_h=max_spatial_length,
            max_w=max_spatial_length,
        )
        
        # === Timestep Embedding ===
        self.time_embed = SinusoidalTimestepEmbedding(dim=dim)
        
        # === Conditioning Embeddings ===
        # For resolution, aspect ratio, FPS, duration
        self.cond_embed = ConditioningEmbedding(dim=dim * 4, num_conditions=4)
        
        # === Modality Embedding ===
        self.modality_embed = ModalityEmbedding(dim=dim, num_modalities=5)
        
        # === Context Projection (for cross-attention) ===
        if cross_attention_dim != dim:
            self.context_proj = nn.Linear(cross_attention_dim, dim)
        else:
            self.context_proj = nn.Identity()
        
        # === Transformer Blocks ===
        if use_joint_blocks and num_joint_blocks > 0:
            # Use joint blocks for first N layers, then regular blocks
            self.joint_blocks = nn.ModuleList([
                DiTBlockJoint(
                    dim=dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                )
                for _ in range(num_joint_blocks)
            ])
            remaining_depth = depth - num_joint_blocks
        else:
            self.joint_blocks = nn.ModuleList()
            remaining_depth = depth
        
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                use_cross_attention=True,
                cross_attention_dim=dim,  # Already projected
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
            for _ in range(remaining_depth)
        ])
        
        # === Final Layer ===
        self.final_layer = FinalLayer(
            dim=dim,
            patch_size_t=patch_size_t,
            patch_size_h=patch_size_h,
            patch_size_w=patch_size_w,
            out_channels=self.out_channels,
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights following DiT conventions."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.apply(_basic_init)
        
        # Zero-init the final layer
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    # Short name aliases
    CONFIG_ALIASES = {
        "S": "Lux-S",
        "B": "Lux-B",
        "L": "Lux-L",
        "XL": "Lux-XL",
    }

    @classmethod
    def from_config(cls, config_name: str, **kwargs):
        """Create model from predefined configuration.
        
        Accepts both full names ('Lux-B') and short aliases ('B').
        """
        # Resolve short aliases
        resolved = cls.CONFIG_ALIASES.get(config_name, config_name)
        if resolved not in cls.CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Choose from {list(cls.CONFIGS.keys())} or {list(cls.CONFIG_ALIASES.keys())}")
        config = cls.CONFIGS[resolved].copy()
        config.update(kwargs)
        return cls(**config)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> dict:
        """
        Forward pass of the Diffusion Transformer.
        
        Args:
            x: (B, C, T, H, W) noisy video latent
            timesteps: (B,) diffusion timesteps
            context: (B, M, D_ctx) conditioning embeddings (text/image/audio)
            context_mask: (B, M) boolean mask for padded conditioning tokens
            conditions: (B, 4) scalar conditions [height, width, fps, duration]
            
        Returns:
            Dict with 'sample': (B, C, T, H, W) predicted noise/velocity
        """
        B = x.shape[0]
        
        # === Patch Embedding ===
        x, t_patches, h_patches, w_patches = self.patch_embed(x)
        # x: (B, num_patches, dim)
        
        # === Add Position Embedding ===
        pos = self.pos_embed(t_patches, h_patches, w_patches, device=x.device)
        x = x + pos
        
        # === Timestep + Condition Embedding ===
        t_emb = self.time_embed(timesteps)  # (B, dim*4)
        if conditions is not None:
            t_emb = t_emb + self.cond_embed(conditions)
        
        # === Project Context ===
        if context is not None:
            context = self.context_proj(context)
        
        # === Joint Blocks ===
        for block in self.joint_blocks:
            x, context = block(x, t_emb, t_patches, h_patches, w_patches, context)
        
        # === Standard DiT Blocks ===
        for block in self.blocks:
            x = block(x, t_emb, t_patches, h_patches, w_patches, 
                     context=context, context_mask=context_mask)
        
        # === Final Layer ===
        x = self.final_layer(x, t_emb)
        
        # === Unpatchify ===
        x = self.final_layer.unpatchify(x, t_patches, h_patches, w_patches)
        
        if return_dict:
            return {"sample": x}
        return x

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        cfg_scale: float = 7.5,
        conditions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with Classifier-Free Guidance.
        
        Runs conditional and unconditional forward passes in parallel,
        then combines them with the guidance scale.
        """
        # Duplicate inputs for CFG
        x_combined = torch.cat([x, x], dim=0)
        t_combined = torch.cat([timesteps, timesteps], dim=0)
        
        # Create null context for unconditional
        null_context = torch.zeros_like(context)
        ctx_combined = torch.cat([context, null_context], dim=0)
        
        if conditions is not None:
            cond_combined = torch.cat([conditions, conditions], dim=0)
        else:
            cond_combined = None
        
        # Forward pass
        output = self.forward(x_combined, t_combined, ctx_combined, 
                             conditions=cond_combined, return_dict=True)
        pred = output["sample"]
        
        # Split and apply CFG
        cond_pred, uncond_pred = pred.chunk(2, dim=0)
        guided = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
        
        return guided

    def get_param_count(self) -> Dict[str, int]:
        """Get parameter counts by component."""
        counts = {}
        counts["patch_embed"] = sum(p.numel() for p in self.patch_embed.parameters())
        counts["pos_embed"] = sum(p.numel() for p in self.pos_embed.parameters())
        counts["time_embed"] = sum(p.numel() for p in self.time_embed.parameters())
        counts["joint_blocks"] = sum(p.numel() for p in self.joint_blocks.parameters())
        counts["blocks"] = sum(p.numel() for p in self.blocks.parameters())
        counts["final_layer"] = sum(p.numel() for p in self.final_layer.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for all blocks."""
        for block in self.blocks:
            block.use_gradient_checkpointing = True
        for block in self.joint_blocks:
            block.use_gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        for block in self.blocks:
            block.use_gradient_checkpointing = False
        for block in self.joint_blocks:
            block.use_gradient_checkpointing = False
