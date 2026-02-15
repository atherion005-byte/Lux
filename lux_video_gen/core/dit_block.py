"""
DiT Transformer Block - The core building block of the Lux video model.

Implements the Diffusion Transformer block with:
- Adaptive Layer Normalization (adaLN-Zero) for timestep conditioning
- Factored spatial-temporal attention (spatial + temporal)
- Optional cross-attention for multi-modal conditioning
- MLP with gated activation (SwiGLU)
- Gradient checkpointing support for memory efficiency
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import MultiHeadAttention, SpatialAttention, TemporalAttention
from .normalization import AdaptiveLayerNorm, RMSNorm


class SwiGLU(nn.Module):
    """
    SwiGLU activation function (Gated Linear Unit with SiLU gating).
    
    More expressive than standard GELU, providing better gradient flow
    and improved model capacity. Used in modern transformer architectures.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        # Round to nearest multiple of 256 for efficiency
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class DiTBlock(nn.Module):
    """
    A single Diffusion Transformer block with factored spatial-temporal attention.
    
    Architecture:
    1. Spatial self-attention (within each frame)
    2. Temporal self-attention (across frames at each position)
    3. Cross-attention to conditioning (text/image/audio embeddings)
    4. Feed-forward network (SwiGLU)
    
    All components use adaLN-Zero for timestep-adaptive normalization,
    ensuring the block acts as identity at initialization for stable training.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = True,
        use_cross_attention: bool = True,
        cross_attention_dim: Optional[int] = None,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.use_cross_attention = use_cross_attention
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Adaptive Layer Norm for timestep conditioning
        num_modulation = 9 if use_cross_attention else 6
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim * 4, num_modulation * dim),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        
        # Norms
        self.norm1_spatial = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm1_temporal = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # Spatial self-attention
        self.spatial_attn = SpatialAttention(
            dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop,
            qk_norm=qk_norm,
        )
        
        # Temporal self-attention
        self.temporal_attn = TemporalAttention(
            dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop,
            qk_norm=qk_norm,
        )
        
        # Cross-attention for conditioning
        if use_cross_attention:
            cross_attention_dim = cross_attention_dim or dim
            self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.cross_attn = MultiHeadAttention(
                dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
                qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop,
                qk_norm=qk_norm,
            )
            # Projection for cross-attention context if dimensions differ
            if cross_attention_dim != dim:
                self.context_proj = nn.Linear(cross_attention_dim, dim)
            else:
                self.context_proj = nn.Identity()
        
        # Feed-forward network
        self.mlp = SwiGLU(dim=dim, hidden_dim=int(dim * mlp_ratio))

    def _get_modulations(self, c: torch.Tensor):
        """Get all modulation parameters from the conditioning vector."""
        mods = self.adaLN(c)
        if self.use_cross_attention:
            return mods.chunk(9, dim=-1)
        return mods.chunk(6, dim=-1)

    def _forward_impl(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: int, h: int, w: int,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Internal forward without gradient checkpointing."""
        
        mods = self._get_modulations(c)
        
        if self.use_cross_attention:
            (shift_s, scale_s, gate_s,
             shift_t, scale_t, gate_t,
             shift_x, scale_x, gate_x) = mods[:9]
        else:
            (shift_s, scale_s, gate_s,
             shift_t, scale_t, gate_t) = mods[:6]
        
        # 1. Spatial self-attention
        x_norm = self.norm1_spatial(x) * (1 + scale_s.unsqueeze(1)) + shift_s.unsqueeze(1)
        x = x + gate_s.unsqueeze(1) * self.spatial_attn(x_norm, t=t, h=h, w=w)
        
        # 2. Temporal self-attention
        x_norm = self.norm1_temporal(x) * (1 + scale_t.unsqueeze(1)) + shift_t.unsqueeze(1)
        x = x + gate_t.unsqueeze(1) * self.temporal_attn(x_norm, t=t, h=h, w=w)
        
        # 3. Cross-attention (if conditioning is provided)
        if self.use_cross_attention and context is not None:
            context = self.context_proj(context)
            x_norm = self.norm_cross(x) * (1 + shift_x.unsqueeze(1)) + scale_x.unsqueeze(1)
            x = x + gate_x.unsqueeze(1) * self.cross_attn(x_norm, context=context, mask=context_mask)
        
        # 4. Feed-forward network (reuse scale/shift from last applicable mod)
        ff_shift = shift_t if not self.use_cross_attention else shift_x
        ff_scale = scale_t if not self.use_cross_attention else scale_x
        x_norm = self.norm2(x) * (1 + ff_scale.unsqueeze(1)) + ff_shift.unsqueeze(1)
        x = x + self.mlp(x_norm)
        
        return x

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: int, h: int, w: int,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for DiT block.
        
        Args:
            x: (B, T*H*W, D) sequence of video patch tokens
            c: (B, D*4) timestep conditioning embedding
            t, h, w: Number of patches in temporal, height, width
            context: (B, M, D_ctx) conditioning context (text/image/audio embeddings)
            context_mask: Optional attention mask for conditioning
        """
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(
                self._forward_impl, x, c, t, h, w, context, context_mask,
                use_reentrant=False,
            )
        return self._forward_impl(x, c, t, h, w, context, context_mask)


class DiTBlockJoint(nn.Module):
    """
    Joint DiT block that processes video and conditioning tokens together.
    
    Instead of using cross-attention, this block concatenates the video
    and conditioning tokens and processes them jointly through self-attention.
    This approach (used in SD3/Flux) often provides better conditioning.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # AdaLN for video stream (6 params) + context stream (6 params)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim * 4, 12 * dim),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        
        # Norms for both streams
        self.norm1_video = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm1_ctx = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2_video = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2_ctx = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # Joint attention for spatial
        self.spatial_attn = MultiHeadAttention(
            dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
            qkv_bias=qkv_bias, qk_norm=qk_norm,
        )
        
        # Temporal attention (video only)
        self.temporal_attn = TemporalAttention(
            dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
            qkv_bias=qkv_bias, qk_norm=qk_norm,
        )
        self.norm_temporal = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # Separate MLPs for each stream
        self.mlp_video = SwiGLU(dim=dim, hidden_dim=int(dim * mlp_ratio))
        self.mlp_ctx = SwiGLU(dim=dim, hidden_dim=int(dim * mlp_ratio))

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: int, h: int, w: int,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T*H*W, D) video tokens
            c: (B, D*4) timestep conditioning
            t, h, w: Patch dimensions
            context: (B, M, D) conditioning tokens
        """
        B = x.shape[0]
        N_vid = x.shape[1]
        N_ctx = context.shape[1]
        
        # Get modulations
        mods = self.adaLN(c).chunk(12, dim=-1)
        (shift_v1, scale_v1, gate_v1, shift_v2, scale_v2, gate_v2,
         shift_c1, scale_c1, gate_c1, shift_c2, scale_c2, gate_c2) = mods
        
        # === Joint Spatial Attention ===
        # Normalize both streams
        x_norm = self.norm1_video(x) * (1 + scale_v1.unsqueeze(1)) + shift_v1.unsqueeze(1)
        ctx_norm = self.norm1_ctx(context) * (1 + scale_c1.unsqueeze(1)) + shift_c1.unsqueeze(1)
        
        # Concat and attend jointly
        joint = torch.cat([x_norm, ctx_norm], dim=1)  # (B, N_vid + N_ctx, D)
        joint_out = self.spatial_attn(joint)
        
        # Split back
        x_attn = joint_out[:, :N_vid]
        ctx_attn = joint_out[:, N_vid:]
        
        x = x + gate_v1.unsqueeze(1) * x_attn
        context = context + gate_c1.unsqueeze(1) * ctx_attn
        
        # === Temporal Attention (video only) ===
        x_temporal = self.norm_temporal(x)
        x = x + self.temporal_attn(x_temporal, t=t, h=h, w=w)
        
        # === Feed-forward ===
        x_norm = self.norm2_video(x) * (1 + scale_v2.unsqueeze(1)) + shift_v2.unsqueeze(1)
        x = x + gate_v2.unsqueeze(1) * self.mlp_video(x_norm)
        
        ctx_norm = self.norm2_ctx(context) * (1 + scale_c2.unsqueeze(1)) + shift_c2.unsqueeze(1)
        context = context + gate_c2.unsqueeze(1) * self.mlp_ctx(ctx_norm)
        
        return x, context
