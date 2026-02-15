"""
Attention mechanisms for the Lux Diffusion Transformer.

Implements multi-head self-attention and cross-attention with support for:
- Flash Attention 2 (via xformers or native PyTorch)
- Rotary Position Embeddings (RoPE) for spatial-temporal awareness
- Memory-efficient attention for consumer GPUs
- Separate spatial and temporal attention paths
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Try to import memory-efficient attention backends
try:
    from xformers.ops import memory_efficient_attention as xformers_attn
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

try:
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    SDPA_AVAILABLE = True
except ImportError:
    SDPA_AVAILABLE = False


def get_attention_backend() -> str:
    """Determine the best available attention backend."""
    if XFORMERS_AVAILABLE:
        return "xformers"
    elif SDPA_AVAILABLE:
        return "sdpa"
    else:
        return "vanilla"


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for spatial-temporal transformers.
    
    Applies rotation-based positional encoding that naturally captures
    relative positions in both spatial and temporal dimensions.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequency bands
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute sin/cos cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build sin/cos cache for given sequence length."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the dimensions."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor.
        
        Args:
            x: Input tensor of shape (B, seq_len, heads, dim) or (B, heads, seq_len, dim)
            seq_dim: Which dimension contains the sequence
        """
        seq_len = x.shape[seq_dim]
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len

        cos = self.cos_cached[:seq_len].to(x.dtype)
        sin = self.sin_cached[:seq_len].to(x.dtype)

        if seq_dim == 1:
            cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim)
            sin = sin.unsqueeze(0).unsqueeze(2)
        elif seq_dim == 2:
            cos = cos.unsqueeze(0).unsqueeze(1)  # (1, 1, seq, dim)
            sin = sin.unsqueeze(0).unsqueeze(1)

        return x * cos + self._rotate_half(x) * sin


class SpatioTemporalRoPE(nn.Module):
    """
    Factored Rotary Position Embedding for video (3D: time, height, width).
    
    Splits the head dimension into three parts for temporal, height, and width
    positional encoding, allowing the model to learn spatial-temporal relationships.
    """

    def __init__(self, dim: int, max_t: int = 128, max_h: int = 128, max_w: int = 128,
                 theta: float = 10000.0):
        super().__init__()
        assert dim % 3 == 0 or dim % 2 == 0, "Dim should be divisible by 2 or 3"
        
        # Split dimension for temporal and spatial
        if dim % 3 == 0:
            self.dim_t = self.dim_h = self.dim_w = dim // 3
        else:
            self.dim_t = dim // 4
            self.dim_h = dim // 4
            self.dim_w = dim - self.dim_t - self.dim_h
        
        self.rope_t = RotaryPositionEmbedding(self.dim_t, max_t, theta)
        self.rope_h = RotaryPositionEmbedding(self.dim_h, max_h, theta)
        self.rope_w = RotaryPositionEmbedding(self.dim_w, max_w, theta)

    def forward(self, x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        """
        Apply factored 3D RoPE.
        
        Args:
            x: (B, T*H*W, heads, dim) 
            t, h, w: Temporal and spatial dimensions
        """
        B, seq_len, heads, dim = x.shape
        assert seq_len == t * h * w, f"Sequence length {seq_len} != {t}*{h}*{w}"
        
        # Reshape to 3D grid
        x = rearrange(x, "b (t h w) heads d -> b t h w heads d", t=t, h=h, w=w)
        
        # Split dim for each axis
        x_t, x_h, x_w = x.split([self.dim_t, self.dim_h, self.dim_w], dim=-1)
        
        # Apply RoPE along each axis
        x_t = rearrange(x_t, "b t h w heads d -> (b h w) t heads d")
        x_t = self.rope_t(x_t, seq_dim=1)
        x_t = rearrange(x_t, "(b h w) t heads d -> b t h w heads d", h=h, w=w)
        
        x_h = rearrange(x_h, "b t h w heads d -> (b t w) h heads d")
        x_h = self.rope_h(x_h, seq_dim=1)
        x_h = rearrange(x_h, "(b t w) h heads d -> b t h w heads d", t=t, w=w)
        
        x_w = rearrange(x_w, "b t h w heads d -> (b t h) w heads d")
        x_w = self.rope_w(x_w, seq_dim=1)
        x_w = rearrange(x_w, "(b t h) w heads d -> b t h w heads d", t=t, h=h)
        
        x = torch.cat([x_t, x_h, x_w], dim=-1)
        return rearrange(x, "b t h w heads d -> b (t h w) heads d")


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with support for multiple backends.
    
    Features:
    - Flash Attention 2 via xformers or PyTorch SDPA
    - Optional RoPE for positional encoding
    - QK normalization for training stability
    - Grouped-query attention (GQA) support
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = True,
        backend: Optional[str] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm
        self.backend = backend or get_attention_backend()

        # GQA ratio
        assert num_heads % self.num_kv_heads == 0
        self.kv_group_size = num_heads // self.num_kv_heads

        # Projections
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        # QK normalization layers
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope: Optional[nn.Module] = None,
        rope_args: Optional[dict] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        
        Args:
            x: Query input (B, N, D)
            context: Key/value input for cross-attention (B, M, D). If None, self-attention.
            rope: Optional RoPE module to apply
            rope_args: Additional args for RoPE (e.g., t, h, w for SpatioTemporalRoPE)
            mask: Optional attention mask
        """
        B, N, _ = x.shape
        kv_input = context if context is not None else x
        M = kv_input.shape[1]

        # Project Q, K, V
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(kv_input).reshape(B, M, self.num_kv_heads, self.head_dim)
        v = self.v_proj(kv_input).reshape(B, M, self.num_kv_heads, self.head_dim)

        # QK normalization
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE if provided (only for self-attention)
        if rope is not None and context is None:
            if rope_args:
                q = rope(q, **rope_args)
                k = rope(k, **rope_args)
            else:
                q = rope(q)
                k = rope(k)

        # Expand KV for GQA
        if self.kv_group_size > 1:
            k = repeat(k, "b m h d -> b m (h g) d", g=self.kv_group_size)
            v = repeat(v, "b m h d -> b m (h g) d", g=self.kv_group_size)

        # Choose attention backend
        if self.backend == "xformers" and XFORMERS_AVAILABLE:
            out = self._xformers_attention(q, k, v, mask)
        elif self.backend == "sdpa" and SDPA_AVAILABLE:
            out = self._sdpa_attention(q, k, v, mask)
        else:
            out = self._vanilla_attention(q, k, v, mask)

        # Output projection
        out = rearrange(out, "b n h d -> b n (h d)")
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out

    def _xformers_attention(self, q, k, v, mask):
        """Memory-efficient attention via xformers."""
        return xformers_attn(q, k, v, attn_bias=mask, p=self.attn_drop.p if self.training else 0.0)

    def _sdpa_attention(self, q, k, v, mask):
        """Scaled dot-product attention via PyTorch."""
        q = rearrange(q, "b n h d -> b h n d")
        k = rearrange(k, "b m h d -> b h m d")
        v = rearrange(v, "b m h d -> b h m d")
        # Reshape mask for SDPA broadcasting: [B, S] -> [B, 1, 1, S]
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, :]  # [B, 1, 1, S]
            if mask.dtype != torch.bool:
                mask = mask.bool()
        out = sdpa(q, k, v, attn_mask=mask, dropout_p=self.attn_drop.p if self.training else 0.0)
        return rearrange(out, "b h n d -> b n h d")

    def _vanilla_attention(self, q, k, v, mask):
        """Standard attention implementation."""
        q = rearrange(q, "b n h d -> b h n d")
        k = rearrange(k, "b m h d -> b h m d")
        v = rearrange(v, "b m h d -> b h m d")

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        return rearrange(out, "b h n d -> b n h d")


class SpatialAttention(MultiHeadAttention):
    """
    Spatial self-attention applied independently to each frame.
    
    Processes spatial tokens (H*W) at each timestep independently,
    capturing intra-frame spatial relationships.
    """

    def forward(self, x: torch.Tensor, t: int, h: int, w: int, 
                context: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        B, THW, D = x.shape
        # Reshape: treat each frame independently
        x = rearrange(x, "b (t h w) d -> (b t) (h w) d", t=t, h=h, w=w)
        if context is not None:
            context = repeat(context, "b m d -> (b t) m d", t=t)
        
        out = super().forward(x, context=context, **kwargs)
        return rearrange(out, "(b t) (h w) d -> b (t h w) d", t=t, h=h, w=w)


class TemporalAttention(MultiHeadAttention):
    """
    Temporal self-attention applied independently to each spatial position.
    
    Processes temporal tokens (T) at each spatial position independently,
    capturing inter-frame temporal relationships for motion coherence.
    """

    def forward(self, x: torch.Tensor, t: int, h: int, w: int,
                context: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        B, THW, D = x.shape
        # Reshape: treat each spatial position independently
        x = rearrange(x, "b (t h w) d -> (b h w) t d", t=t, h=h, w=w)
        
        out = super().forward(x, context=context, **kwargs)
        return rearrange(out, "(b h w) t d -> b (t h w) d", h=h, w=w)


class JointSpatioTemporalAttention(MultiHeadAttention):
    """
    Joint spatial-temporal attention using 3D RoPE.
    
    Processes all spatial-temporal tokens together with factored positional
    encoding, allowing the model to capture long-range dependencies across
    both space and time simultaneously.
    """

    def __init__(self, dim: int, num_heads: int = 8, **kwargs):
        super().__init__(dim=dim, num_heads=num_heads, **kwargs)
        self.st_rope = SpatioTemporalRoPE(
            dim=dim // num_heads,
            max_t=128,
            max_h=128,
            max_w=128,
        )

    def forward(self, x: torch.Tensor, t: int, h: int, w: int,
                context: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        return super().forward(
            x, context=context,
            rope=self.st_rope,
            rope_args={"t": t, "h": h, "w": w},
            **kwargs
        )
