"""
Text Encoder for Lux Video Generation.

Wraps pre-trained text models from Hugging Face for encoding text prompts
into conditioning embeddings. Supports:
- T5-XXL for rich semantic understanding  
- CLIP text encoder for visual-language alignment
- Dual encoder mode (T5 + CLIP concatenated)

These encoders are frozen during video model training.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from transformers import (
        AutoModel,
        AutoTokenizer,
        CLIPTextModel,
        CLIPTokenizer,
        T5EncoderModel,
        T5Tokenizer,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class T5TextEncoder(nn.Module):
    """
    T5 text encoder for semantic text understanding.
    
    Uses T5-XXL (or smaller variants) to encode text prompts into
    rich semantic embeddings. T5's encoder produces high-quality
    text representations that capture nuanced prompt instructions.
    """

    # Available T5 model variants
    VARIANTS = {
        "t5-small": "google-t5/t5-small",
        "t5-base": "google-t5/t5-base",
        "t5-large": "google-t5/t5-large",
        "t5-xl": "google-t5/t5-xl",
        "t5-xxl": "google-t5/t5-xxl",
        "flan-t5-xl": "google/flan-t5-xl",
        "flan-t5-xxl": "google/flan-t5-xxl",
    }

    def __init__(
        self,
        model_name: str = "google-t5/t5-large",
        max_length: int = 512,
        dtype: torch.dtype = torch.float16,
        device: Optional[str] = None,
    ):
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package required: pip install transformers")

        self.max_length = max_length
        self.dtype = dtype

        # Resolve model name
        if model_name in self.VARIANTS:
            model_name = self.VARIANTS[model_name]

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )

        # Freeze encoder
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.output_dim = self.model.config.d_model

    @torch.no_grad()
    def forward(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompts.
        
        Args:
            texts: List of text strings
            device: Target device
            
        Returns:
            embeddings: (B, seq_len, dim) text embeddings
            mask: (B, seq_len) attention mask
        """
        device = device or next(self.model.parameters()).device

        tokens = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        outputs = self.model(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
        )

        return outputs.last_hidden_state, tokens.attention_mask


class CLIPTextEncoder(nn.Module):
    """
    CLIP text encoder for visual-language aligned embeddings.
    
    CLIP text embeddings are naturally aligned with visual features,
    making them excellent for guiding visual generation.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        max_length: int = 77,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package required")

        self.max_length = max_length
        self.dtype = dtype

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.output_dim = self.model.config.hidden_size

    @torch.no_grad()
    def forward(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text using CLIP."""
        device = device or next(self.model.parameters()).device

        tokens = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        outputs = self.model(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
        )

        return outputs.last_hidden_state, tokens.attention_mask


class DualTextEncoder(nn.Module):
    """
    Dual text encoder combining T5 and CLIP.
    
    Concatenates embeddings from both encoders along the token dimension,
    providing both semantic richness (T5) and visual alignment (CLIP).
    Optionally projects to a common dimension.
    """

    def __init__(
        self,
        t5_model: str = "google-t5/t5-large",
        clip_model: str = "openai/clip-vit-large-patch14",
        output_dim: int = 4096,
        t5_max_length: int = 512,
        clip_max_length: int = 77,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.t5_encoder = T5TextEncoder(
            model_name=t5_model,
            max_length=t5_max_length,
            dtype=dtype,
        )
        self.clip_encoder = CLIPTextEncoder(
            model_name=clip_model,
            max_length=clip_max_length,
            dtype=dtype,
        )

        self.output_dim = output_dim

        # Project CLIP embeddings to T5 dimension space, then concat
        t5_dim = self.t5_encoder.output_dim
        clip_dim = self.clip_encoder.output_dim

        # Project both to output_dim
        self.t5_proj = nn.Linear(t5_dim, output_dim)
        self.clip_proj = nn.Linear(clip_dim, output_dim)

        # Trainable parameters (projections are trained)
        self.t5_proj.requires_grad_(True)
        self.clip_proj.requires_grad_(True)

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode texts through both encoders."""
        t5_emb, t5_mask = self.t5_encoder(texts, device=device)
        clip_emb, clip_mask = self.clip_encoder(texts, device=device)
        return (t5_emb, t5_mask), (clip_emb, clip_mask)

    def forward(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode and project text embeddings.
        
        Returns:
            embeddings: (B, t5_len + clip_len, output_dim)
            mask: (B, t5_len + clip_len)
        """
        (t5_emb, t5_mask), (clip_emb, clip_mask) = self.encode(texts, device)

        # Project to common dimension
        t5_proj = self.t5_proj(t5_emb.float()).to(t5_emb.dtype)
        clip_proj = self.clip_proj(clip_emb.float()).to(clip_emb.dtype)

        # Concatenate along token dimension
        embeddings = torch.cat([t5_proj, clip_proj], dim=1)
        mask = torch.cat([t5_mask, clip_mask], dim=1)

        return embeddings, mask


class TextEncoderWrapper(nn.Module):
    """
    Unified text encoder interface that can use T5, CLIP, or dual encoder.
    
    Provides a consistent API regardless of the underlying encoder choice,
    simplifying integration with the rest of the model.
    """

    ENCODER_TYPES = {
        "t5": "T5TextEncoder",
        "clip": "CLIPTextEncoder",
        "dual": "DualTextEncoder",
    }

    def __init__(
        self,
        encoder_type: str = "t5",
        output_dim: int = 4096,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.output_dim = output_dim

        if encoder_type == "t5":
            self.encoder = T5TextEncoder(dtype=dtype, **kwargs)
            if self.encoder.output_dim != output_dim:
                self.proj = nn.Linear(self.encoder.output_dim, output_dim)
            else:
                self.proj = nn.Identity()
        elif encoder_type == "clip":
            self.encoder = CLIPTextEncoder(dtype=dtype, **kwargs)
            if self.encoder.output_dim != output_dim:
                self.proj = nn.Linear(self.encoder.output_dim, output_dim)
            else:
                self.proj = nn.Identity()
        elif encoder_type == "dual":
            self.encoder = DualTextEncoder(
                output_dim=output_dim, dtype=dtype, **kwargs
            )
            self.proj = nn.Identity()
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def forward(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompts.
        
        Returns:
            embeddings: (B, seq_len, output_dim)
            mask: (B, seq_len)
        """
        emb, mask = self.encoder(texts, device=device)
        emb = self.proj(emb.float()).to(emb.dtype)
        return emb, mask
