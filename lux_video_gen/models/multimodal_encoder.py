"""
Multi-Modal Encoder for Lux.

Encodes and fuses conditioning inputs from multiple modalities
(text, image, video, audio) into a unified sequence of conditioning
tokens for the DiT model.

Supports:
- Text: via T5/CLIP encoders
- Images: via CLIP vision encoder (up to 9 reference images)
- Video: via 3D VAE encoder (up to 3 reference clips)
- Audio: via audio tokenizer (up to 3 audio clips)
- Reference features: motion, style, camera from reference videos
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from transformers import CLIPVisionModel, CLIPImageProcessor
    HAS_CLIP_VISION = True
except ImportError:
    HAS_CLIP_VISION = False


class ImageEncoder(nn.Module):
    """
    Image encoder using CLIP vision model.
    
    Encodes reference images into feature sequences that can be
    used as conditioning for the DiT model.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        output_dim: int = 4096,
        dtype: torch.dtype = torch.float16,
        max_images: int = 9,
    ):
        super().__init__()
        self.max_images = max_images
        self.output_dim = output_dim

        if HAS_CLIP_VISION:
            self.processor = CLIPImageProcessor.from_pretrained(model_name)
            self.model = CLIPVisionModel.from_pretrained(
                model_name, torch_dtype=dtype
            )
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            vision_dim = self.model.config.hidden_size
        else:
            vision_dim = 1024  # Default CLIP ViT-L dim

        # Projection to output dimension
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

        # Learnable token to represent "no image"
        self.null_token = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode batch of images using CLIP vision model.
        
        Args:
            images: (B, N_images, C, H, W) batch of reference images
        Returns:
            (B, N_images * num_patches, output_dim) image features
        """
        B, N, C, H, W = images.shape

        if not HAS_CLIP_VISION:
            # Fallback: simple conv-based encoding
            return self.null_token.expand(B, N, -1)

        # Flatten batch and images
        images_flat = rearrange(images, "b n c h w -> (b n) c h w")

        outputs = self.model(pixel_values=images_flat)
        features = outputs.last_hidden_state  # (B*N, num_patches, dim)

        features = rearrange(features, "(b n) p d -> b (n p) d", b=B, n=N)
        return features

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Encode images and project to output dimension.
        
        Args:
            images: (B, N_images, C, H, W) or None
        Returns:
            (B, tokens, output_dim) image conditioning tokens
        """
        if images is None:
            B = 1
            return self.null_token.expand(B, 1, -1).to(device)

        features = self.encode_images(images)
        return self.proj(features.float()).to(features.dtype)


class VideoReferenceEncoder(nn.Module):
    """
    Encodes reference videos for style/motion/camera transfer.
    
    Extracts different types of features from reference videos:
    - Motion features: temporal patterns for motion replication
    - Style features: visual appearance for style transfer
    - Camera features: camera movement patterns
    """

    def __init__(
        self,
        latent_dim: int = 4,
        output_dim: int = 4096,
        feature_types: List[str] = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.feature_types = feature_types or ["motion", "style", "camera"]

        # Feature extractors for different reference types
        self.motion_encoder = nn.Sequential(
            nn.Conv3d(latent_dim, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # Pool spatial dims
            nn.Flatten(2),
            nn.Linear(64, output_dim),
        )

        self.style_encoder = nn.Sequential(
            nn.Conv3d(latent_dim, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1, None, None)),  # Pool temporal dim
            nn.Flatten(2),
            nn.Linear(64, output_dim),
        )

        self.camera_encoder = nn.Sequential(
            nn.Conv3d(latent_dim, 64, kernel_size=(3, 3, 3), padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((None, 2, 2)),
            nn.Flatten(2),
            nn.Linear(64 * 4, output_dim),
        )

        # Feature type embedding
        self.type_embed = nn.Embedding(len(self.feature_types), output_dim)

    def forward(
        self,
        video_latent: torch.Tensor,
        reference_types: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Extract reference features from video latent.
        
        Args:
            video_latent: (B, C, T, H, W) encoded reference video
            reference_types: Which types of features to extract
            
        Returns:
            (B, num_tokens, output_dim) reference conditioning tokens
        """
        reference_types = reference_types or self.feature_types
        features = []

        for ref_type in reference_types:
            if ref_type == "motion":
                feat = self.motion_encoder(video_latent)  # (B, T, dim)
            elif ref_type == "style":
                feat = self.style_encoder(video_latent)  # (B, H*W, dim)
            elif ref_type == "camera":
                feat = self.camera_encoder(video_latent)  # (B, T, dim)
            else:
                continue

            if feat.dim() == 2:
                feat = feat.unsqueeze(1)

            # Add type embedding
            type_idx = self.feature_types.index(ref_type)
            feat = feat + self.type_embed(
                torch.tensor(type_idx, device=feat.device)
            )
            features.append(feat)

        if not features:
            return torch.zeros(
                video_latent.shape[0], 1, self.output_dim,
                device=video_latent.device, dtype=video_latent.dtype,
            )

        return torch.cat(features, dim=1)


class AudioReferenceEncoder(nn.Module):
    """
    Encodes audio references for audio-guided video generation.
    
    Processes audio inputs to extract features for:
    - Voice/speech reference (for consistent character voices)
    - Music beat synchronization
    - Sound effect reference
    """

    def __init__(self, audio_dim: int = 256, output_dim: int = 4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(audio_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.beat_detector = nn.Sequential(
            nn.Conv1d(audio_dim, 128, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, audio_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio reference.
        
        Args:
            audio_tokens: (B, audio_dim, T) encoded audio tokens
            
        Returns:
            features: (B, T, output_dim) audio conditioning tokens
            beats: (B, T) detected beat positions
        """
        features = rearrange(audio_tokens, "b d t -> b t d")
        features = self.proj(features)
        beats = self.beat_detector(audio_tokens).squeeze(1)
        return features, beats


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion module that combines conditioning from all modalities.
    
    Receives encoded features from text, image, video, and audio encoders,
    and produces a unified conditioning sequence for the DiT model.
    
    Uses cross-attention pooling to compress and align features from
    different modalities into a common representation space.
    """

    def __init__(
        self,
        dim: int = 4096,
        num_heads: int = 16,
        max_tokens: int = 1024,
        num_modalities: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.max_tokens = max_tokens

        # Modality-specific layer norms
        self.text_norm = nn.LayerNorm(dim)
        self.image_norm = nn.LayerNorm(dim)
        self.video_norm = nn.LayerNorm(dim)
        self.audio_norm = nn.LayerNorm(dim)

        # Cross-attention for fusion
        self.fusion_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        # Modality type tokens
        self.modality_tokens = nn.Embedding(num_modalities, dim)

        # Output projection
        self.output_proj = nn.Linear(dim, dim)

    def forward(
        self,
        text_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse multi-modal conditioning features.
        
        Args:
            text_features: (B, T_text, dim)
            text_mask: (B, T_text)
            image_features: (B, T_image, dim) 
            video_features: (B, T_video, dim)
            audio_features: (B, T_audio, dim)
            
        Returns:
            fused: (B, total_tokens, dim) fused conditioning
            mask: (B, total_tokens) attention mask
        """
        parts = []
        masks = []
        device = None

        # Add text features
        if text_features is not None:
            device = text_features.device
            text_features = self.text_norm(text_features)
            text_features = text_features + self.modality_tokens(
                torch.zeros(1, dtype=torch.long, device=device)
            )
            parts.append(text_features)
            if text_mask is not None:
                masks.append(text_mask)
            else:
                masks.append(torch.ones(text_features.shape[:2], device=device, dtype=torch.bool))

        # Add image features
        if image_features is not None:
            device = device or image_features.device
            image_features = self.image_norm(image_features)
            image_features = image_features + self.modality_tokens(
                torch.ones(1, dtype=torch.long, device=device)
            )
            parts.append(image_features)
            masks.append(torch.ones(image_features.shape[:2], device=device, dtype=torch.bool))

        # Add video features
        if video_features is not None:
            device = device or video_features.device
            video_features = self.video_norm(video_features)
            video_features = video_features + self.modality_tokens(
                torch.full((1,), 2, dtype=torch.long, device=device)
            )
            parts.append(video_features)
            masks.append(torch.ones(video_features.shape[:2], device=device, dtype=torch.bool))

        # Add audio features
        if audio_features is not None:
            device = device or audio_features.device
            audio_features = self.audio_norm(audio_features)
            audio_features = audio_features + self.modality_tokens(
                torch.full((1,), 3, dtype=torch.long, device=device)
            )
            parts.append(audio_features)
            masks.append(torch.ones(audio_features.shape[:2], device=device, dtype=torch.bool))

        if not parts:
            raise ValueError("At least one modality must be provided")

        # Concatenate all modalities
        fused = torch.cat(parts, dim=1)
        mask = torch.cat(masks, dim=1)

        # Self-attention fusion
        h = self.fusion_norm(fused)
        h_attn, _ = self.fusion_attn(h, h, h)
        fused = fused + h_attn
        fused = fused + self.fusion_mlp(fused)

        # Output projection
        fused = self.output_proj(fused)

        return fused, mask
