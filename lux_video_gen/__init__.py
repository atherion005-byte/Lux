"""
Lux Video Generation Model
===========================
A next-generation AI video generation system built on Diffusion Transformer architecture.
Supports multi-modal input (text, image, video, audio) with native audio-video
synchronization, physics-aware generation, and character consistency.
"""

__version__ = "0.1.0"
__author__ = "Lux Team"

from .models.dit_video import LuxDiT
from .models.vae3d import VideoVAE
from .inference.pipeline import LuxPipeline, GenerationConfig
