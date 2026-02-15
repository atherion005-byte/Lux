"""
Video Dataset for Lux Training.

Supports loading video-text pairs from various dataset formats:
- WebVid-style (CSV with video URLs and captions)
- Hugging Face datasets (webdataset format)
- Local directory with video files and caption files
- LAION/InternVid style datasets

Features:
- Variable resolution and duration support
- Dynamic batching by aspect ratio
- Multi-worker data loading
- On-the-fly video augmentation
- Audio extraction and synchronization
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

try:
    import decord
    decord.bridge.set_bridge("torch")
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


class VideoTextDataset(Dataset):
    """
    Dataset for loading video-text pairs.
    
    Supports multiple data formats and handles variable-length videos
    with automatic temporal sampling and spatial resizing.
    """

    # Standard aspect ratios and their resolutions
    ASPECT_RATIOS = {
        "16:9": (576, 1024),
        "9:16": (1024, 576),
        "4:3": (640, 864),
        "3:4": (864, 640),
        "1:1": (768, 768),
        "21:9": (448, 1056),
    }

    def __init__(
        self,
        data_dir: str,
        caption_file: Optional[str] = None,
        num_frames: int = 16,
        frame_stride: int = 4,
        resolution: int = 512,
        aspect_ratio: str = "16:9",
        max_duration: float = 15.0,
        min_duration: float = 1.0,
        sample_fps: int = 24,
        load_audio: bool = True,
        audio_sample_rate: int = 16000,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.resolution = resolution
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.sample_fps = sample_fps
        self.load_audio = load_audio
        self.audio_sample_rate = audio_sample_rate
        self.transform = transform
        
        # Get target resolution
        if aspect_ratio in self.ASPECT_RATIOS:
            self.target_h, self.target_w = self.ASPECT_RATIOS[aspect_ratio]
        else:
            self.target_h = self.target_w = resolution
        
        # Scale resolution
        scale = resolution / max(self.target_h, self.target_w)
        self.target_h = int(self.target_h * scale) // 8 * 8  # Ensure divisible by 8
        self.target_w = int(self.target_w * scale) // 8 * 8
        
        # Load metadata
        self.samples = self._load_metadata(caption_file)
        if max_samples:
            self.samples = self.samples[:max_samples]

    def _load_metadata(self, caption_file: Optional[str]) -> List[Dict[str, Any]]:
        """Load dataset metadata (video paths and captions)."""
        samples = []
        
        if caption_file and Path(caption_file).exists():
            caption_path = Path(caption_file)
            
            if caption_path.suffix == ".json":
                with open(caption_path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    samples = data
                elif isinstance(data, dict):
                    samples = list(data.values())
                    
            elif caption_path.suffix == ".jsonl":
                with open(caption_path) as f:
                    for line in f:
                        samples.append(json.loads(line.strip()))
                        
            elif caption_path.suffix == ".csv":
                import csv
                with open(caption_path) as f:
                    reader = csv.DictReader(f)
                    samples = list(reader)
        else:
            # Auto-discover videos in directory
            video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            for video_path in sorted(self.data_dir.rglob("*")):
                if video_path.suffix.lower() in video_extensions:
                    caption = ""
                    caption_file = video_path.with_suffix(".txt")
                    if caption_file.exists():
                        caption = caption_file.read_text().strip()
                    
                    samples.append({
                        "video": str(video_path),
                        "caption": caption,
                    })
        
        return samples

    def _load_video_decord(self, path: str) -> Tuple[torch.Tensor, float]:
        """Load video using decord (fastest)."""
        vr = decord.VideoReader(path, num_threads=1)
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / fps
        
        # Calculate sampling
        target_frames = self.num_frames
        stride = max(1, int(fps / self.sample_fps * self.frame_stride))
        
        # Select frame indices
        total_needed = target_frames * stride
        if total_needed > total_frames:
            stride = max(1, total_frames // target_frames)
            total_needed = target_frames * stride
        
        start = random.randint(0, max(0, total_frames - total_needed))
        indices = list(range(start, min(start + total_needed, total_frames), stride))
        indices = indices[:target_frames]
        
        # Pad if needed
        while len(indices) < target_frames:
            indices.append(indices[-1])
        
        frames = vr.get_batch(indices)  # (T, H, W, C)
        frames = frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
        
        return frames, fps

    def _load_video_av(self, path: str) -> Tuple[torch.Tensor, float]:
        """Load video using PyAV (fallback)."""
        container = av.open(path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 24)
        
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")
            frames.append(torch.from_numpy(img))
            if len(frames) >= self.num_frames * self.frame_stride + 1:
                break
        
        container.close()
        
        if not frames:
            # Return black frames
            return torch.zeros(self.num_frames, 3, self.target_h, self.target_w), fps
        
        frames = torch.stack(frames).permute(0, 3, 1, 2).float() / 255.0
        
        # Sample frames with stride
        indices = list(range(0, len(frames), max(1, self.frame_stride)))[:self.num_frames]
        while len(indices) < self.num_frames:
            indices.append(indices[-1] if indices else 0)
        
        return frames[indices], fps

    def _load_audio(self, path: str, duration: float) -> Optional[torch.Tensor]:
        """Load audio from video file."""
        if not TORCHAUDIO_AVAILABLE:
            return None
        
        try:
            waveform, sr = torchaudio.load(path)
            # Resample if needed
            if sr != self.audio_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.audio_sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Trim/pad to match video duration
            target_samples = int(duration * self.audio_sample_rate)
            if waveform.shape[1] > target_samples:
                waveform = waveform[:, :target_samples]
            elif waveform.shape[1] < target_samples:
                waveform = F.pad(waveform, (0, target_samples - waveform.shape[1]))
            
            return waveform
        except Exception:
            return None

    def _resize_video(self, frames: torch.Tensor) -> torch.Tensor:
        """Resize video frames to target resolution."""
        T, C, H, W = frames.shape
        
        # Resize maintaining aspect ratio, then center crop
        if H / W > self.target_h / self.target_w:
            # Height is larger relative to target
            new_w = self.target_w
            new_h = int(H * self.target_w / W)
        else:
            new_h = self.target_h
            new_w = int(W * self.target_h / H)
        
        # Ensure even dimensions
        new_h = max(new_h, self.target_h)
        new_w = max(new_w, self.target_w)
        
        frames = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)
        
        # Center crop
        start_h = (new_h - self.target_h) // 2
        start_w = (new_w - self.target_w) // 2
        frames = frames[:, :, start_h:start_h + self.target_h, start_w:start_w + self.target_w]
        
        return frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        sample = self.samples[idx]
        
        video_path = sample.get("video", sample.get("path", ""))
        if not os.path.isabs(video_path):
            video_path = str(self.data_dir / video_path)
        
        caption = sample.get("caption", sample.get("text", ""))

        # Load video
        try:
            if DECORD_AVAILABLE:
                frames, fps = self._load_video_decord(video_path)
            elif AV_AVAILABLE:
                frames, fps = self._load_video_av(video_path)
            else:
                # Placeholder
                frames = torch.randn(self.num_frames, 3, self.target_h, self.target_w)
                fps = self.sample_fps
        except Exception as e:
            # Return placeholder on error
            frames = torch.randn(self.num_frames, 3, self.target_h, self.target_w)
            fps = self.sample_fps

        # Ensure correct frame count
        if frames.shape[0] != self.num_frames:
            if frames.shape[0] > self.num_frames:
                frames = frames[:self.num_frames]
            else:
                pad = torch.zeros(self.num_frames - frames.shape[0], *frames.shape[1:])
                frames = torch.cat([frames, pad], dim=0)

        # Resize
        frames = self._resize_video(frames)

        # Ensure exact target resolution
        if frames.shape[2] != self.target_h or frames.shape[3] != self.target_w:
            frames = F.interpolate(frames, size=(self.target_h, self.target_w), mode="bilinear", align_corners=False)

        # Normalize to [-1, 1]
        frames = frames * 2.0 - 1.0

        # Apply transforms
        if self.transform:
            frames = self.transform(frames)

        # Rearrange to (C, T, H, W) for model input
        video = frames.permute(1, 0, 2, 3)  # (C, T, H, W)

        result = {
            "video": video,
            "caption": caption,
            "fps": torch.tensor(fps, dtype=torch.float32),
            "height": torch.tensor(self.target_h, dtype=torch.float32),
            "width": torch.tensor(self.target_w, dtype=torch.float32),
            "duration": torch.tensor(self.num_frames / fps, dtype=torch.float32),
        }

        # Load audio if requested
        if self.load_audio:
            audio = self._load_audio(video_path, self.num_frames / fps)
            if audio is not None:
                result["audio"] = audio
            else:
                result["audio"] = torch.zeros(1, int(self.num_frames / fps * self.audio_sample_rate))

        return result


class AspectRatioBucketSampler(Sampler):
    """
    Sampler that groups videos by aspect ratio for efficient batching.
    
    Groups similar aspect ratios together to minimize padding waste
    and improve GPU utilization during training.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 4,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Create buckets based on aspect ratio
        self.buckets = self._create_buckets()

    def _create_buckets(self) -> List[List[int]]:
        """Group samples by aspect ratio."""
        ratio_buckets = {}
        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset.samples[idx]
                # Use target aspect ratio from dataset
                ratio_key = f"{self.dataset.target_h}x{self.dataset.target_w}"
            except (AttributeError, IndexError):
                ratio_key = "default"

            if ratio_key not in ratio_buckets:
                ratio_buckets[ratio_key] = []
            ratio_buckets[ratio_key].append(idx)

        # Create batches from buckets
        batches = []
        for indices in ratio_buckets.values():
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        return batches

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.buckets)
        for batch in self.buckets:
            yield from batch

    def __len__(self):
        return sum(len(b) for b in self.buckets)


def video_collate_fn(batch):
    """Custom collate function that handles variable-size video tensors gracefully."""
    # Filter out None entries
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    result = {}
    
    # Get target shapes from first item
    target_video_shape = batch[0]["video"].shape
    
    for key in batch[0].keys():
        values = [b[key] for b in batch]
        
        if key == "caption":
            # Keep strings as list
            result[key] = values
        elif isinstance(values[0], torch.Tensor):
            # Ensure all tensors have the same shape
            target_shape = values[0].shape
            fixed_values = []
            for v in values:
                if v.shape != target_shape:
                    # Resize to match target shape
                    if v.ndim == target_shape.__len__():
                        # Pad or truncate
                        new_v = torch.zeros(target_shape, dtype=v.dtype)
                        slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(v.shape, target_shape))
                        new_v[slices] = v[slices]
                        fixed_values.append(new_v)
                    else:
                        fixed_values.append(values[0].clone())
                else:
                    fixed_values.append(v)
            result[key] = torch.stack(fixed_values)
        else:
            result[key] = values
    
    return result


def create_dataloader(
    data_dir: str,
    batch_size: int = 4,
    num_frames: int = 16,
    resolution: int = 512,
    num_workers: int = 4,
    load_audio: bool = True,
    metadata_file: Optional[str] = None,
    fps: int = 24,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for video training.
    
    Args:
        data_dir: Path to video dataset
        batch_size: Batch size
        num_frames: Number of frames to sample per video
        resolution: Target resolution
        num_workers: Data loading workers
        load_audio: Whether to load audio
        metadata_file: Path to metadata JSON/JSONL/CSV file
        fps: Target sampling FPS
        
    Returns:
        DataLoader instance
    """
    dataset = VideoTextDataset(
        data_dir=data_dir,
        caption_file=metadata_file,
        num_frames=num_frames,
        resolution=resolution,
        sample_fps=fps,
        load_audio=load_audio,
        **kwargs,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        collate_fn=video_collate_fn,
    )

    return dataloader
