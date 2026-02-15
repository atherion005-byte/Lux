"""
Inference Configuration.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceConfig:
    """Configuration for video generation inference."""
    # Model
    model_variant: str = "B"
    model_path: Optional[str] = None
    use_ema_weights: bool = True
    
    # Generation
    height: int = 512
    width: int = 512
    num_frames: int = 49
    fps: int = 24
    
    # Sampling
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    sampler: str = "euler"  # euler, euler_a, dpm_solver_pp, unipc
    
    # Scheduler
    scheduler: str = "flow_matching"
    
    # Seeds
    seed: int = -1  # -1 = random
    
    # Output
    output_dir: str = "outputs"
    output_format: str = "mp4"
    save_frames: bool = False
    
    # Memory
    mixed_precision: str = "fp16"
    enable_vae_tiling: bool = True
    vae_tile_size: int = 256
    offload_to_cpu: bool = False
    
    # Audio
    generate_audio: bool = False
    audio_reference: Optional[str] = None
