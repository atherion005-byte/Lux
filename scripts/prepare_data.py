#!/usr/bin/env python3
"""
Data Preparation Script.

Processes raw video files into a training-ready dataset:
1. Scans directory for video files
2. Extracts metadata (resolution, duration, fps)
3. Optionally generates captions using BLIP-2
4. Creates metadata JSON for the dataset loader
5. Validates and reports statistics

Usage:
    python scripts/prepare_data.py --input_dir raw_videos/ --output_dir datasets/processed/
    python scripts/prepare_data.py --input_dir raw_videos/ --output_dir datasets/processed/ --caption
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare training dataset")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--caption", action="store_true", help="Auto-caption with BLIP-2")
    parser.add_argument("--min_duration", type=float, default=0.5, help="Min video duration (seconds)")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Max video duration (seconds)")
    parser.add_argument("--min_resolution", type=int, default=128, help="Min spatial dimension")
    parser.add_argument("--extensions", nargs="+", default=[".mp4", ".avi", ".mov", ".mkv", ".webm"])
    return parser.parse_args()


def get_video_info(path: str) -> dict:
    """Extract video metadata."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        info = {
            "path": path,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        info["duration"] = info["total_frames"] / max(info["fps"], 1)
        cap.release()
        return info
    except Exception as e:
        return {"path": path, "error": str(e)}


def auto_caption(video_path: str, captioner) -> str:
    """Generate caption for a video using first frame."""
    import cv2
    from PIL import Image
    import numpy as np
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return "a video"
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    inputs = captioner["processor"](pil_image, return_tensors="pt").to(captioner["device"])
    output = captioner["model"].generate(**inputs, max_new_tokens=50)
    caption = captioner["processor"].decode(output[0], skip_special_tokens=True)
    
    return caption.strip()


def main():
    args = parse_args()
    
    print(f"Scanning {args.input_dir} for videos...")
    
    # Find all video files
    input_path = Path(args.input_dir)
    video_files = []
    for ext in args.extensions:
        video_files.extend(input_path.rglob(f"*{ext}"))
    
    print(f"Found {len(video_files)} video files")
    
    if not video_files:
        print("No videos found. Check input directory and extensions.")
        return
    
    # Setup captioner if requested
    captioner = None
    if args.caption:
        print("Loading captioning model (BLIP-2)...")
        try:
            import torch
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
            ).to(device)
            captioner = {"model": model, "processor": processor, "device": device}
            print("Captioner loaded")
        except Exception as e:
            print(f"Could not load captioner: {e}. Skipping auto-captioning.")
            captioner = None
    
    # Process videos
    os.makedirs(args.output_dir, exist_ok=True)
    metadata = []
    skipped = 0
    errors = 0
    
    for i, vf in enumerate(video_files):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Processing {i + 1}/{len(video_files)}...")
        
        info = get_video_info(str(vf))
        
        if "error" in info:
            errors += 1
            continue
        
        # Filter
        if info["duration"] < args.min_duration or info["duration"] > args.max_duration:
            skipped += 1
            continue
        if info["width"] < args.min_resolution or info["height"] < args.min_resolution:
            skipped += 1
            continue
        
        # Generate caption
        if captioner:
            caption = auto_caption(str(vf), captioner)
        else:
            # Use filename as placeholder caption
            caption = vf.stem.replace("_", " ").replace("-", " ")
        
        entry = {
            "video_path": str(vf.relative_to(input_path)),
            "caption": caption,
            "width": info["width"],
            "height": info["height"],
            "fps": info["fps"],
            "duration": info["duration"],
            "total_frames": info["total_frames"],
        }
        metadata.append(entry)
    
    # Save metadata
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Print statistics
    print(f"\n{'='*50}")
    print(f"Dataset Preparation Complete")
    print(f"{'='*50}")
    print(f"Total videos found: {len(video_files)}")
    print(f"Accepted: {len(metadata)}")
    print(f"Skipped (filters): {skipped}")
    print(f"Errors: {errors}")
    print(f"Metadata saved to: {metadata_path}")
    
    if metadata:
        durations = [m["duration"] for m in metadata]
        resolutions = [f"{m['width']}x{m['height']}" for m in metadata]
        print(f"\nDuration: {min(durations):.1f}s - {max(durations):.1f}s (avg {sum(durations)/len(durations):.1f}s)")
        print(f"Total duration: {sum(durations)/3600:.1f} hours")
        
        from collections import Counter
        top_res = Counter(resolutions).most_common(5)
        print(f"Top resolutions: {', '.join(f'{r} ({c})' for r, c in top_res)}")


if __name__ == "__main__":
    main()
