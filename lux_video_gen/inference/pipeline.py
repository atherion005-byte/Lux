"""
Main Inference Pipeline for Lux Video Generation.

Supports multiple generation modes:
- Text-to-Video (T2V)
- Image-to-Video (I2V)
- Video-to-Video (V2V) style/motion transfer
- Multi-modal generation (text + image + audio)
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for video generation."""
    # Video dimensions
    height: int = 512
    width: int = 512
    num_frames: int = 49
    fps: int = 24
    
    # Sampling
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    
    # Scheduler
    scheduler_type: str = "flow_matching"
    
    # Seeds
    seed: Optional[int] = None
    
    # Multi-modal
    image_guidance_scale: float = 1.5
    
    # Output
    output_format: str = "mp4"
    
    # Memory
    enable_vae_tiling: bool = True
    enable_vae_slicing: bool = True
    vae_tile_size: int = 256
    offload_to_cpu: bool = False


class LuxPipeline:
    """
    Main inference pipeline for Lux video generation.
    
    Orchestrates the complete generation process:
    1. Text encoding (T5 + CLIP)
    2. Initial latent preparation
    3. Iterative denoising with DiT
    4. VAE decoding to pixel space
    5. Post-processing
    """

    def __init__(
        self,
        dit_model: nn.Module,
        vae_model: nn.Module,
        text_encoder: nn.Module,
        noise_scheduler: Any,
        audio_model: Optional[nn.Module] = None,
        multimodal_encoder: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        self.dit_model = dit_model
        self.vae_model = vae_model
        self.text_encoder = text_encoder
        self.noise_scheduler = noise_scheduler
        self.audio_model = audio_model
        self.multimodal_encoder = multimodal_encoder

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._move_to_device()

    def _move_to_device(self):
        """Move models to appropriate devices."""
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            self.dit_device = torch.device("cuda:0")
            self.aux_device = torch.device("cuda:1")
        else:
            self.dit_device = self.aux_device = self.device

        self.dit_model = self.dit_model.to(self.dit_device)
        self.dit_model.eval()
        self.vae_model = self.vae_model.to(self.aux_device)
        self.vae_model.eval()
        self.text_encoder = self.text_encoder.to(self.aux_device)
        self.text_encoder.eval()

        if self.audio_model is not None:
            self.audio_model = self.audio_model.to(self.aux_device)
            self.audio_model.eval()

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        config: Optional[GenerationConfig] = None,
        reference_image: Optional[torch.Tensor] = None,
        reference_video: Optional[torch.Tensor] = None,
        reference_audio: Optional[torch.Tensor] = None,
        callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Generate video from text and optional reference inputs.
        
        Args:
            prompt: Text prompt(s) for generation
            negative_prompt: Negative prompt(s)
            config: Generation configuration
            reference_image: Optional reference image (B, C, H, W)
            reference_video: Optional reference video (B, C, T, H, W)
            reference_audio: Optional reference audio (B, T)
            callback: Optional callback(step, total_steps, latent)
            
        Returns:
            Dictionary with "video" tensor and optional "audio"
        """
        config = config or GenerationConfig()

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        elif negative_prompt is None:
            negative_prompt = [""] * batch_size

        # Set seed for reproducibility
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=self.dit_device).manual_seed(config.seed)

        logger.info(f"Generating video: {config.width}x{config.height}, {config.num_frames} frames")

        # 1. Encode text
        prompt_embeds, prompt_mask = self._encode_text(prompt)
        neg_embeds, neg_mask = self._encode_text(negative_prompt)

        # 2. Encode reference inputs (if any)
        extra_context = self._encode_references(
            reference_image, reference_video, reference_audio
        )

        # 3. Prepare initial noise
        latent_shape = self._get_latent_shape(
            batch_size, config.num_frames, config.height, config.width
        )
        latents = torch.randn(
            latent_shape, device=self.dit_device,
            dtype=torch.float16, generator=generator
        )

        # 4. Prepare conditions
        conditions = torch.tensor(
            [[config.height, config.width, config.fps, config.num_frames / config.fps]],
            device=self.dit_device,
        ).expand(batch_size, -1)

        # 5. Setup scheduler timesteps
        timesteps = self.noise_scheduler.get_sampling_timesteps(
            config.num_inference_steps, device=self.dit_device
        )

        # 6. Denoising loop
        for i, t in enumerate(timesteps):
            t_batch = t.expand(batch_size)

            # Classifier-free guidance: predict with and without conditions
            if config.guidance_scale > 1.0:
                latent_input = torch.cat([latents, latents], dim=0)
                t_input = torch.cat([t_batch, t_batch], dim=0)
                context = torch.cat([prompt_embeds, neg_embeds], dim=0)
                context_mask = torch.cat([prompt_mask, neg_mask], dim=0)
                cond_input = torch.cat([conditions, conditions], dim=0)

                with autocast(device_type="cuda", dtype=torch.float16):
                    output = self.dit_model(
                        latent_input, t_input,
                        context=context,
                        context_mask=context_mask,
                        conditions=cond_input,
                    )

                pred = output["sample"]
                pred_cond, pred_uncond = pred.chunk(2, dim=0)
                pred = pred_uncond + config.guidance_scale * (pred_cond - pred_uncond)
            else:
                with autocast(device_type="cuda", dtype=torch.float16):
                    output = self.dit_model(
                        latents, t_batch,
                        context=prompt_embeds,
                        context_mask=prompt_mask,
                        conditions=conditions,
                    )
                pred = output["sample"]

            # Step scheduler
            latents = self.noise_scheduler.step(pred, t, latents)

            # Callback
            if callback is not None:
                callback(i + 1, len(timesteps), latents)

        # 7. Decode latents to video
        video = self._decode_latents(latents, config)

        # 8. Generate audio (if model available)
        audio = None
        if self.audio_model is not None and reference_audio is not None:
            audio = self._generate_audio(video, reference_audio)

        return {
            "video": video,
            "audio": audio,
        }

    def _encode_text(
        self, texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts."""
        with torch.no_grad():
            embeddings, mask = self.text_encoder(texts, device=self.aux_device)
        return embeddings.to(self.dit_device), mask.to(self.dit_device)

    def _encode_references(
        self,
        image: Optional[torch.Tensor],
        video: Optional[torch.Tensor],
        audio: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Encode reference inputs through multi-modal encoder."""
        if self.multimodal_encoder is None:
            return None
        if image is None and video is None and audio is None:
            return None

        with torch.no_grad():
            context = self.multimodal_encoder(
                image_input=image.to(self.aux_device) if image is not None else None,
                video_input=video.to(self.aux_device) if video is not None else None,
                audio_input=audio.to(self.aux_device) if audio is not None else None,
            )
        return context.to(self.dit_device)

    def _get_latent_shape(
        self, batch_size: int, num_frames: int, height: int, width: int
    ) -> Tuple[int, ...]:
        """Calculate latent tensor shape based on VAE compression."""
        spatial_factor = 8
        temporal_factor = 4
        latent_channels = 4

        latent_t = max(1, num_frames // temporal_factor)
        latent_h = height // spatial_factor
        latent_w = width // spatial_factor

        return (batch_size, latent_channels, latent_t, latent_h, latent_w)

    def _decode_latents(
        self, latents: torch.Tensor, config: GenerationConfig
    ) -> torch.Tensor:
        """Decode latent representations to pixel-space video."""
        latents = latents.to(self.aux_device)

        if config.enable_vae_tiling and (
            latents.shape[-2] > config.vae_tile_size // 8
            or latents.shape[-1] > config.vae_tile_size // 8
        ):
            video = self._tiled_decode(latents, config.vae_tile_size // 8)
        else:
            with torch.no_grad():
                video = self.vae_model.decode(latents)

        # Clamp to valid range
        video = video.clamp(-1, 1)
        # Normalize to [0, 1]
        video = (video + 1.0) / 2.0

        return video

    def _tiled_decode(
        self, latents: torch.Tensor, tile_size: int, overlap: int = 4
    ) -> torch.Tensor:
        """Decode latents in tiles for memory efficiency."""
        B, C, T, H, W = latents.shape
        stride = tile_size - overlap
        output_spatial_factor = 8
        
        out_h = H * output_spatial_factor
        out_w = W * output_spatial_factor
        output = torch.zeros(
            B, 3, T * 4, out_h, out_w,
            device=latents.device, dtype=latents.dtype,
        )
        weight = torch.zeros_like(output)

        for h_start in range(0, H, stride):
            for w_start in range(0, W, stride):
                h_end = min(h_start + tile_size, H)
                w_end = min(w_start + tile_size, W)
                h_start_actual = max(0, h_end - tile_size)
                w_start_actual = max(0, w_end - tile_size)

                tile = latents[:, :, :, h_start_actual:h_end, w_start_actual:w_end]

                with torch.no_grad():
                    decoded = self.vae_model.decode(tile)

                oh_s = h_start_actual * output_spatial_factor
                oh_e = h_end * output_spatial_factor
                ow_s = w_start_actual * output_spatial_factor
                ow_e = w_end * output_spatial_factor

                output[:, :, :, oh_s:oh_e, ow_s:ow_e] += decoded
                weight[:, :, :, oh_s:oh_e, ow_s:ow_e] += 1

        output = output / weight.clamp(min=1)
        return output

    def _generate_audio(
        self, video: torch.Tensor, reference_audio: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Generate synchronized audio from video features."""
        if self.audio_model is None:
            return None

        with torch.no_grad():
            # Extract video features for audio conditioning
            # Use temporal average pooling as video features
            video_features = video.mean(dim=[3, 4])  # (B, C, T)
            video_features = video_features.permute(0, 2, 1)  # (B, T, C)
            
            audio = self.audio_model(
                video_features=video_features.to(self.aux_device),
                audio_input=reference_audio.to(self.aux_device),
            )

        return audio

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ) -> "LuxPipeline":
        """Load a pre-trained pipeline from disk."""
        from ..models.dit_video import LuxDiT
        from ..models.vae3d import VideoVAE
        from ..models.text_encoder import TextEncoderWrapper
        from ..training.scheduler import create_scheduler

        model_path = os.path.abspath(model_path)

        # Load configs
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}

        # Initialize models
        dit_config = config.get("dit", "B")
        dit = LuxDiT.from_config(dit_config)
        vae = VideoVAE()
        text_enc = TextEncoderWrapper()
        scheduler = create_scheduler(config.get("scheduler", "flow_matching"))

        # Load weights
        dit_path = os.path.join(model_path, "dit.pt")
        if os.path.exists(dit_path):
            dit.load_state_dict(torch.load(dit_path, map_location="cpu", weights_only=True))

        vae_path = os.path.join(model_path, "vae.pt")
        if os.path.exists(vae_path):
            vae.load_state_dict(torch.load(vae_path, map_location="cpu", weights_only=True))

        # Cast to dtype
        dit = dit.to(dtype)
        vae = vae.to(dtype)

        return cls(
            dit_model=dit,
            vae_model=vae,
            text_encoder=text_enc,
            noise_scheduler=scheduler,
            device=device,
        )

    def save_pretrained(self, save_path: str):
        """Save pipeline for later loading."""
        os.makedirs(save_path, exist_ok=True)

        torch.save(self.dit_model.state_dict(), os.path.join(save_path, "dit.pt"))
        torch.save(self.vae_model.state_dict(), os.path.join(save_path, "vae.pt"))

        config = {
            "scheduler": "flow_matching",
        }

        import json
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved pipeline to {save_path}")
