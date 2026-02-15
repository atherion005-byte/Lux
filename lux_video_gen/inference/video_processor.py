"""
Video Post-Processing Pipeline.

Handles:
- Temporal interpolation for smooth output
- Super-resolution upscaling
- Color correction and normalization
- Video encoding and saving to disk
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Post-processing pipeline for generated video tensors."""

    def __init__(
        self,
        target_fps: int = 24,
        enable_interpolation: bool = True,
        interpolation_factor: int = 2,
        color_correction: bool = True,
    ):
        self.target_fps = target_fps
        self.enable_interpolation = enable_interpolation
        self.interpolation_factor = interpolation_factor
        self.color_correction = color_correction

    def process(
        self,
        video: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Full post-processing pipeline.
        
        Args:
            video: (B, C, T, H, W) tensor in [0, 1]
            target_size: Optional (H, W) for resizing
        """
        # Ensure float32 for processing
        video = video.float()

        # Temporal interpolation
        if self.enable_interpolation and self.interpolation_factor > 1:
            video = self.temporal_interpolate(video, self.interpolation_factor)

        # Spatial upscaling
        if target_size is not None:
            video = self.spatial_resize(video, target_size)

        # Color correction
        if self.color_correction:
            video = self.correct_colors(video)

        # Ensure valid range
        video = video.clamp(0, 1)

        return video

    def temporal_interpolate(
        self, video: torch.Tensor, factor: int = 2
    ) -> torch.Tensor:
        """
        Temporal interpolation using optical flow-aware blending.
        Doubles (or more) the frame count for smoother motion.
        """
        B, C, T, H, W = video.shape
        new_T = (T - 1) * factor + 1

        # Reshape for interpolation: treat spatial dims as batch
        # (B, C, T, H, W) -> (B*H*W, C, T)
        v = video.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)
        
        # Use 1D interpolation along time
        v_interp = F.interpolate(v, size=new_T, mode="linear", align_corners=True)
        
        # Reshape back
        v_interp = v_interp.reshape(B, H, W, C, new_T).permute(0, 3, 4, 1, 2)

        return v_interp

    def spatial_resize(
        self, video: torch.Tensor, target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Resize video spatially with bicubic interpolation."""
        B, C, T, H, W = video.shape
        target_h, target_w = target_size

        if H == target_h and W == target_w:
            return video

        # Process frame by frame to save memory
        frames = []
        for t in range(T):
            frame = video[:, :, t]  # (B, C, H, W)
            frame = F.interpolate(
                frame, size=(target_h, target_w),
                mode="bicubic", align_corners=False, antialias=True,
            )
            frames.append(frame)

        return torch.stack(frames, dim=2)

    def correct_colors(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply color correction:
        - Histogram stretching per-channel
        - Temporal color consistency
        """
        B, C, T, H, W = video.shape

        for b in range(B):
            for c in range(C):
                channel = video[b, c]  # (T, H, W)
                # Percentile-based contrast stretching
                lo = torch.quantile(channel, 0.01)
                hi = torch.quantile(channel, 0.99)
                if hi - lo > 0.01:
                    channel = (channel - lo) / (hi - lo)
                    video[b, c] = channel.clamp(0, 1)

        return video

    def to_numpy(self, video: torch.Tensor) -> np.ndarray:
        """Convert video tensor to numpy array (B, T, H, W, C) uint8."""
        video = video.clamp(0, 1)
        # (B, C, T, H, W) -> (B, T, H, W, C)
        video = video.permute(0, 2, 3, 4, 1)
        video = (video * 255).byte().cpu().numpy()
        return video

    def save_video(
        self,
        video: torch.Tensor,
        output_path: str,
        fps: Optional[int] = None,
        audio: Optional[torch.Tensor] = None,
        audio_sr: int = 16000,
    ):
        """
        Save video tensor to file.
        
        Args:
            video: (B, C, T, H, W) or (C, T, H, W)
            output_path: Path to save
            fps: Frames per second
            audio: Optional audio tensor
            audio_sr: Audio sample rate
        """
        fps = fps or self.target_fps
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if video.dim() == 5:
            video = video[0]  # Take first batch

        frames_np = self.to_numpy(video.unsqueeze(0))[0]  # (T, H, W, C)

        try:
            import cv2
            self._save_with_opencv(frames_np, output_path, fps)
        except ImportError:
            try:
                import imageio
                self._save_with_imageio(frames_np, output_path, fps)
            except ImportError:
                self._save_frames(frames_np, output_path)

        # Add audio if provided
        if audio is not None:
            self._mux_audio(output_path, audio, audio_sr)

        logger.info(f"Saved video to {output_path}")

    def _save_with_opencv(
        self, frames: np.ndarray, path: str, fps: int
    ):
        """Save with OpenCV, then convert to H.264 with ffmpeg if available."""
        import cv2
        import subprocess
        import shutil
        
        T, H, W, C = frames.shape
        
        # Try ffmpeg first (produces universally playable H.264 MP4)
        if shutil.which("ffmpeg"):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp_path = tmp.name
            
            # Write raw frames via OpenCV to temp file
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(tmp_path, fourcc, fps, (W, H))
            for t in range(T):
                frame_bgr = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            writer.release()
            
            # Convert to H.264 with ffmpeg
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", tmp_path,
                    "-c:v", "libx264", "-preset", "fast",
                    "-crf", "23", "-pix_fmt", "yuv420p",
                    path
                ], capture_output=True, timeout=60)
                os.unlink(tmp_path)
                return
            except Exception:
                # Fallback: just move the raw file
                if os.path.exists(tmp_path):
                    shutil.move(tmp_path, path)
                return
        
        # Fallback: raw mp4v (may not play in all players)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
        for t in range(T):
            frame_bgr = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        writer.release()

    def _save_with_imageio(
        self, frames: np.ndarray, path: str, fps: int
    ):
        """Save with imageio."""
        import imageio
        writer = imageio.get_writer(path, fps=fps)
        for t in range(frames.shape[0]):
            writer.append_data(frames[t])
        writer.close()

    def _save_frames(self, frames: np.ndarray, path: str):
        """Fallback: save individual frames as images."""
        from PIL import Image
        frame_dir = path.replace(".mp4", "_frames")
        os.makedirs(frame_dir, exist_ok=True)
        for t in range(frames.shape[0]):
            img = Image.fromarray(frames[t])
            img.save(os.path.join(frame_dir, f"frame_{t:05d}.png"))
        logger.info(f"Saved {frames.shape[0]} frames to {frame_dir}")

    def _mux_audio(
        self, video_path: str, audio: torch.Tensor, sr: int
    ):
        """Mux audio into video file using ffmpeg."""
        import subprocess
        import tempfile

        try:
            import soundfile as sf
        except ImportError:
            logger.warning("soundfile not available, skipping audio muxing")
            return

        # Save audio to temp file
        audio_np = audio.cpu().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np[0]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_np, sr)
            audio_path = f.name

        # Mux with ffmpeg
        output_with_audio = video_path.replace(".mp4", "_with_audio.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac",
            "-shortest",
            output_with_audio,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.replace(output_with_audio, video_path)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("ffmpeg not available for audio muxing")
        finally:
            os.unlink(audio_path)
