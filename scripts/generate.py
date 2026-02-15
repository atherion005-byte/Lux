#!/usr/bin/env python3
"""
Lux Video Generation - Video Generation Script.

Usage:
    # Text-to-Video
    python scripts/generate.py --prompt "A dragon flying over mountains" --output outputs/dragon.mp4

    # Image-to-Video
    python scripts/generate.py --prompt "Camera pans right" --image reference.jpg --output outputs/animated.mp4

    # High resolution
    python scripts/generate.py --prompt "Ocean waves" --width 1024 --height 576 --num_frames 49

    # Custom sampling
    python scripts/generate.py --prompt "City timelapse" --steps 100 --guidance 12.0 --seed 42
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lux_video_gen.utils.logging_utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Lux Video Generation")
    
    # Input
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, distorted, artifacts")
    parser.add_argument("--image", type=str, default=None, help="Reference image for I2V")
    parser.add_argument("--audio", type=str, default=None, help="Reference audio")
    
    # Output
    parser.add_argument("--output", type=str, default="outputs/generated.mp4")
    parser.add_argument("--save_frames", action="store_true")
    
    # Video settings
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    
    # Sampling
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--sampler", type=str, default="euler", choices=["euler", "euler_a", "dpm_solver_pp", "unipc"])
    
    # Model
    parser.add_argument("--model_path", type=str, default="checkpoints/best")
    parser.add_argument("--model_variant", type=str, default="B")
    parser.add_argument("--use_ema", action="store_true", default=True)
    
    # Hardware
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--vae_tiling", action="store_true", default=True)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Lux Video Generation")
    logger.info("=" * 60)
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Size: {args.width}x{args.height}, {args.num_frames} frames @ {args.fps} fps")
    
    # Load pipeline
    logger.info("Loading pipeline...")
    from lux_video_gen.inference.pipeline import LuxPipeline, GenerationConfig
    
    if os.path.exists(args.model_path):
        pipeline = LuxPipeline.from_pretrained(args.model_path)
    else:
        logger.warning(f"Model path {args.model_path} not found. Initializing fresh model (random weights).")
        from lux_video_gen.models.dit_video import LuxDiT
        from lux_video_gen.models.vae3d import VideoVAE
        from lux_video_gen.models.text_encoder import TextEncoderWrapper
        from lux_video_gen.training.scheduler import create_scheduler
        
        dit = LuxDiT.from_config(args.model_variant)
        vae = VideoVAE()
        text_encoder = TextEncoderWrapper()
        scheduler = create_scheduler("flow_matching")
        
        pipeline = LuxPipeline(
            dit_model=dit,
            vae_model=vae,
            text_encoder=text_encoder,
            noise_scheduler=scheduler,
        )
    
    # Generate
    config = GenerationConfig(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed if args.seed >= 0 else None,
        enable_vae_tiling=args.vae_tiling,
    )
    
    # Load reference inputs
    ref_image = None
    if args.image:
        from lux_video_gen.utils.video_utils import load_image
        ref_image = load_image(args.image, target_size=(args.height, args.width)).unsqueeze(0)
    
    ref_audio = None
    if args.audio:
        from lux_video_gen.utils.video_utils import load_audio
        ref_audio = load_audio(args.audio).unsqueeze(0)
    
    logger.info("Generating video...")
    start_time = time.time()
    
    def progress_callback(step, total, latent):
        elapsed = time.time() - start_time
        eta = elapsed / step * (total - step) if step > 0 else 0
        logger.info(f"Step {step}/{total} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
    
    result = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        config=config,
        reference_image=ref_image,
        reference_audio=ref_audio,
        callback=progress_callback,
    )
    
    gen_time = time.time() - start_time
    logger.info(f"Generation complete in {gen_time:.1f}s")
    
    # Save output
    from lux_video_gen.inference.video_processor import VideoProcessor
    processor = VideoProcessor(target_fps=args.fps)
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    processor.save_video(
        result["video"],
        args.output,
        fps=args.fps,
        audio=result.get("audio"),
    )
    
    if args.save_frames:
        from lux_video_gen.utils.video_utils import save_video_frames
        frames_dir = args.output.replace(".mp4", "_frames")
        save_video_frames(result["video"], frames_dir)
    
    logger.info(f"Saved to {args.output}")
    logger.info(f"Video: {args.width}x{args.height}, {args.num_frames} frames, {gen_time:.1f}s generation time")


if __name__ == "__main__":
    main()
