"""
Video I/O Utilities.

Handles reading and writing video files with multiple backend support.
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_video(
    path: str,
    num_frames: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
    fps: Optional[int] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Load video file to tensor.
    
    Args:
        path: Path to video file
        num_frames: Max frames to load (None = all)
        target_size: (H, W) to resize to
        fps: Target FPS (None = original)
    
    Returns:
        video: (C, T, H, W) tensor in [0, 1]
        info: dict with fps, duration, original_size
    """
    try:
        return _load_with_decord(path, num_frames, target_size, fps)
    except ImportError:
        pass

    try:
        return _load_with_opencv(path, num_frames, target_size, fps)
    except ImportError:
        raise ImportError("Need either decord or opencv-python for video loading")


def _load_with_decord(path, num_frames, target_size, fps):
    import decord
    decord.bridge.set_bridge("torch")

    vr = decord.VideoReader(path)
    orig_fps = vr.get_avg_fps()
    total_frames = len(vr)

    # Compute frame indices
    if fps and fps < orig_fps:
        step = orig_fps / fps
        indices = [int(i * step) for i in range(int(total_frames / step))]
    else:
        indices = list(range(total_frames))

    if num_frames and len(indices) > num_frames:
        # Uniform sample
        step = len(indices) / num_frames
        indices = [indices[int(i * step)] for i in range(num_frames)]

    frames = vr.get_batch(indices)  # (T, H, W, C)

    # Resize
    if target_size:
        import torch.nn.functional as F
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        frames = F.interpolate(frames, size=target_size, mode="bilinear", align_corners=False)
        video = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
    else:
        video = frames.permute(3, 0, 1, 2).float() / 255.0

    info = {
        "fps": fps or orig_fps,
        "duration": total_frames / orig_fps,
        "original_size": (vr[0].shape[0], vr[0].shape[1]),
    }

    return video, info


def _load_with_opencv(path, num_frames, target_size, fps):
    import cv2

    cap = cv2.VideoCapture(path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    step = 1
    if fps and fps < orig_fps:
        step = orig_fps / fps

    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % max(1, int(step)) == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if target_size:
                frame = cv2.resize(frame, (target_size[1], target_size[0]))
            frames.append(frame)

        if num_frames and len(frames) >= num_frames:
            break
        idx += 1

    cap.release()

    video = torch.from_numpy(np.stack(frames)).float() / 255.0
    video = video.permute(3, 0, 1, 2)  # (C, T, H, W)

    info = {
        "fps": fps or orig_fps,
        "duration": total_frames / max(orig_fps, 1),
        "original_size": (orig_h, orig_w),
    }

    return video, info


def save_video_frames(
    video: torch.Tensor,
    output_dir: str,
    prefix: str = "frame",
):
    """Save video tensor as individual frame images."""
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    if video.dim() == 5:
        video = video[0]  # (C, T, H, W)

    video = video.clamp(0, 1)
    video = (video * 255).byte()

    for t in range(video.shape[1]):
        frame = video[:, t].permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(frame)
        img.save(os.path.join(output_dir, f"{prefix}_{t:05d}.png"))


def load_image(
    path: str,
    target_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Load image to tensor (C, H, W) in [0, 1]."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if target_size:
        img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)

    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)
    return tensor


def load_audio(
    path: str,
    sr: int = 16000,
    duration: Optional[float] = None,
) -> torch.Tensor:
    """Load audio file to tensor."""
    try:
        import librosa
        audio, _ = librosa.load(path, sr=sr, duration=duration, mono=True)
        return torch.from_numpy(audio)
    except ImportError:
        try:
            import torchaudio
            waveform, orig_sr = torchaudio.load(path)
            if orig_sr != sr:
                waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if duration:
                max_samples = int(sr * duration)
                waveform = waveform[:, :max_samples]
            return waveform.squeeze(0)
        except ImportError:
            raise ImportError("Need either librosa or torchaudio for audio loading")


def video_to_gif(
    video: torch.Tensor,
    output_path: str,
    fps: int = 8,
    loop: int = 0,
):
    """Save video tensor as animated GIF."""
    from PIL import Image

    if video.dim() == 5:
        video = video[0]

    video = video.clamp(0, 1)
    frames = (video * 255).byte().permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)

    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=1000 // fps,
        loop=loop,
    )
