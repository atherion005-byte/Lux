"""
Video and image transforms for data augmentation.

Implements augmentations that maintain temporal consistency
across video frames while improving model robustness.
"""

import random
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class VideoTransform:
    """Compose multiple video transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            video = t(video)
        return video


class RandomHorizontalFlip:
    """Randomly flip all frames horizontally."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return video.flip(-1)
        return video


class RandomTemporalCrop:
    """Randomly crop temporal dimension."""

    def __init__(self, target_frames: int):
        self.target_frames = target_frames

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        T = video.shape[0]
        if T <= self.target_frames:
            return video
        start = random.randint(0, T - self.target_frames)
        return video[start:start + self.target_frames]


class ColorJitter:
    """Apply color jitter consistently across all frames."""

    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        # Apply same jitter to all frames
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            video = video * factor

        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = video.mean(dim=(-2, -1), keepdim=True)
            video = (video - mean) * factor + mean

        return video.clamp(-1, 1)


class RandomResizedCrop:
    """Random resized crop maintaining temporal consistency."""

    def __init__(
        self,
        size: Tuple[int, int],
        scale: Tuple[float, float] = (0.8, 1.0),
        ratio: Tuple[float, float] = (0.9, 1.1),
    ):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        T, C, H, W = video.shape
        
        area = H * W
        target_area = random.uniform(*self.scale) * area
        aspect_ratio = random.uniform(*self.ratio)
        
        new_w = int(round((target_area * aspect_ratio) ** 0.5))
        new_h = int(round((target_area / aspect_ratio) ** 0.5))
        new_w = min(new_w, W)
        new_h = min(new_h, H)
        
        top = random.randint(0, H - new_h)
        left = random.randint(0, W - new_w)
        
        video = video[:, :, top:top + new_h, left:left + new_w]
        video = F.interpolate(video, size=self.size, mode="bilinear", align_corners=False)
        
        return video


class Normalize:
    """Normalize video to [mean, std]."""

    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        return (video - self.mean.to(video.device)) / self.std.to(video.device)


def get_train_transforms(
    resolution: Tuple[int, int] = (512, 512),
    use_augmentation: bool = True,
) -> VideoTransform:
    """Get training transforms."""
    transforms = []
    
    if use_augmentation:
        transforms.extend([
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
        ])
    
    return VideoTransform(transforms) if transforms else None


def get_eval_transforms(resolution: Tuple[int, int] = (512, 512)) -> Optional[VideoTransform]:
    """Get evaluation transforms (minimal processing)."""
    return None
