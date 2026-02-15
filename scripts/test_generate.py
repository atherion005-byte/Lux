"""
Quick video generation test using current training model.
Saves a checkpoint from the training model, then runs inference to generate a short video.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=== Lux Video Generation Test ===")
    
    # 1. Initialize models (same as training)
    from lux_video_gen.models.vae3d import VideoVAE
    from lux_video_gen.models.dit_video import LuxDiT
    from lux_video_gen.models.text_encoder import TextEncoderWrapper
    from lux_video_gen.training.scheduler import create_scheduler
    from lux_video_gen.inference.pipeline import LuxPipeline, GenerationConfig
    
    logger.info("Loading models...")
    dit = LuxDiT.from_config("B")
    vae = VideoVAE()
    text_encoder = TextEncoderWrapper()
    
    # Load training checkpoint if available
    ckpt_dirs = []
    ckpt_base = "checkpoints"
    if os.path.exists(ckpt_base):
        ckpt_dirs = sorted([d for d in os.listdir(ckpt_base) if os.path.isdir(os.path.join(ckpt_base, d))])
    
    if ckpt_dirs:
        latest_ckpt = os.path.join(ckpt_base, ckpt_dirs[-1])
        # Try EMA model first (smoother), then regular model
        ema_file = os.path.join(latest_ckpt, "model_ema.pt")
        model_file = os.path.join(latest_ckpt, "model.pt")
        if os.path.exists(ema_file):
            logger.info(f"Loading EMA checkpoint: {ema_file}")
            state = torch.load(ema_file, map_location="cpu", weights_only=True)
            dit.load_state_dict(state)
            logger.info("EMA checkpoint loaded!")
        elif os.path.exists(model_file):
            logger.info(f"Loading checkpoint: {model_file}")
            state = torch.load(model_file, map_location="cpu", weights_only=True)
            dit.load_state_dict(state)
            logger.info("Checkpoint loaded!")
        else:
            logger.info(f"No model.pt in {latest_ckpt}, using random weights")
    else:
        logger.info("No checkpoints found - using randomly initialized model")
        logger.info("(This will produce noise/abstract patterns, but proves the pipeline works)")
    
    # Cast to bf16 for memory efficiency
    dit = dit.to(torch.bfloat16)
    vae = vae.to(torch.bfloat16)
    text_encoder = text_encoder.to(torch.bfloat16)
    
    # Create noise scheduler
    noise_scheduler = create_scheduler("flow_matching")
    
    # Create pipeline
    logger.info("Creating inference pipeline...")
    pipeline = LuxPipeline(
        dit_model=dit,
        vae_model=vae,
        text_encoder=text_encoder,
        noise_scheduler=noise_scheduler,
    )
    
    # Configure for small/fast generation
    config = GenerationConfig(
        height=256,
        width=256,
        num_frames=16,
        fps=8,
        num_inference_steps=20,  # Fewer steps for speed
        guidance_scale=4.0,
        seed=42,
        enable_vae_tiling=False,
    )
    
    # Generate!
    prompt = "A beautiful sunset over the ocean with waves gently rolling"
    logger.info(f"Generating video: '{prompt}'")
    logger.info(f"  Resolution: {config.width}x{config.height}, {config.num_frames} frames")
    logger.info(f"  Steps: {config.num_inference_steps}, CFG: {config.guidance_scale}")
    
    def progress_callback(step, total, latent):
        logger.info(f"  Step {step}/{total}")
    
    result = pipeline(
        prompt=prompt,
        config=config,
        callback=progress_callback,
    )
    
    video = result["video"]
    logger.info(f"Generated video tensor: shape={video.shape}, dtype={video.dtype}")
    logger.info(f"  Value range: [{video.min().item():.3f}, {video.max().item():.3f}]")
    
    # Save as MP4
    output_path = "outputs/test_generation.mp4"
    os.makedirs("outputs", exist_ok=True)
    
    # Try saving with available method
    try:
        from lux_video_gen.inference.video_processor import VideoProcessor
        processor = VideoProcessor()
        processor.save_video(video[0], output_path=output_path, fps=config.fps)
        logger.info(f"Video saved to {output_path}")
    except Exception as e:
        logger.warning(f"VideoProcessor save failed: {e}")
        # Fallback: save frames as images
        try:
            from PIL import Image
            frames_dir = "outputs/test_frames"
            os.makedirs(frames_dir, exist_ok=True)
            vid = video[0].float().cpu()  # (C, T, H, W)
            vid = vid.clamp(0, 1)
            for t in range(vid.shape[1]):
                frame = vid[:, t]  # (C, H, W)
                frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype('uint8')
                img = Image.fromarray(frame_np)
                img.save(os.path.join(frames_dir, f"frame_{t:04d}.png"))
            logger.info(f"Saved {vid.shape[1]} frames to {frames_dir}/")
        except Exception as e2:
            logger.warning(f"Frame save also failed: {e2}")
            # Last resort: save raw tensor
            torch.save(video, "outputs/test_video_tensor.pt")
            logger.info("Saved raw tensor to outputs/test_video_tensor.pt")
    
    logger.info("=== Generation Complete! ===")
    

if __name__ == "__main__":
    main()
