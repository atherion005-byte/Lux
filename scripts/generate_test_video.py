"""
Lux Video Generation Test
==========================
Generates an actual video file using the full pipeline:
  Random noise -> DiT denoising -> VAE decode -> Save video

Note: With untrained (random) weights, the output will be abstract/noisy
patterns - this proves the full pipeline works end-to-end on GPU.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import gc


def generate_test_video():
    print("=" * 60)
    print("  Lux Video Generation - End-to-End Test")
    print("=" * 60)

    device_dit = torch.device("cuda:0")
    n_gpus = torch.cuda.device_count()
    device_vae = torch.device(f"cuda:{min(1, n_gpus-1)}")

    print(f"\nGPU 0 (DiT):  {torch.cuda.get_device_name(0)}")
    if n_gpus > 1:
        print(f"GPU 1 (VAE):  {torch.cuda.get_device_name(1)}")

    # --- Configuration ---
    num_frames = 16        # output video frames
    height = 256           # output resolution
    width = 256
    num_steps = 20         # denoising steps
    cfg_scale = 4.0        # classifier-free guidance scale
    fps = 8                # output fps

    # VAE compression factors
    temporal_factor = 4
    spatial_factor = 8
    latent_channels = 4

    latent_t = max(1, num_frames // temporal_factor)
    latent_h = height // spatial_factor
    latent_w = width // spatial_factor

    print(f"\nSettings:")
    print(f"  Output: {width}x{height}, {num_frames} frames @ {fps} fps")
    print(f"  Latent: {latent_channels}x{latent_t}x{latent_h}x{latent_w}")
    print(f"  Steps: {num_steps}, CFG: {cfg_scale}")

    # --- Load Models ---
    print("\n[1/4] Loading models...")
    t0 = time.perf_counter()

    from lux_video_gen.models.dit_video import LuxDiT
    from lux_video_gen.models.vae3d import VideoVAE
    from lux_video_gen.training.scheduler import FlowMatchingScheduler

    # DiT on GPU 0
    dit = LuxDiT.from_config("S").to(device_dit).half()
    dit.eval()
    dit_params = sum(p.numel() for p in dit.parameters()) / 1e6
    print(f"  DiT (Lux-S): {dit_params:.1f}M params -> {device_dit}")

    # VAE on GPU 1
    vae = VideoVAE(
        in_channels=3,
        latent_dim=latent_channels,
        base_channels=128,
        channel_multipliers=(1, 2, 4, 4),
        num_res_blocks=2,
    ).to(device_vae).half()
    vae.eval()
    vae_params = sum(p.numel() for p in vae.parameters()) / 1e6
    print(f"  VAE: {vae_params:.1f}M params -> {device_vae}")

    # Scheduler
    scheduler = FlowMatchingScheduler(num_train_timesteps=1000)

    print(f"  Models loaded in {time.perf_counter()-t0:.1f}s")

    # --- Prepare Conditioning ---
    print("\n[2/4] Preparing text conditioning...")

    # Simulated text embedding (in production this comes from T5+CLIP)
    prompt = "A majestic eagle soaring over snow-capped mountains at golden hour"
    print(f'  Prompt: "{prompt}"')

    # Mock text encoder output: (B, seq_len, cross_attn_dim)
    torch.manual_seed(42)  # reproducible
    text_emb = torch.randn(1, 64, 4096, device=device_dit, dtype=torch.float16)
    # "Negative" (unconditional) embedding
    uncond_emb = torch.zeros_like(text_emb)

    # Conditions tensor (resolution, fps info)
    conditions = torch.tensor(
        [[height, width, fps, num_frames / fps]], device=device_dit, dtype=torch.float16
    )

    # --- Denoising Loop ---
    print(f"\n[3/4] Denoising ({num_steps} steps with CFG={cfg_scale})...")
    t0 = time.perf_counter()

    # Start from pure noise
    latents = torch.randn(
        1, latent_channels, latent_t, latent_h, latent_w,
        device=device_dit, dtype=torch.float16,
        generator=torch.Generator(device=device_dit).manual_seed(12345),
    )

    timesteps = scheduler.get_sampling_timesteps(num_steps, device_dit)

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0)

            # Classifier-free guidance: run model twice
            latent_input = torch.cat([latents, latents], dim=0)
            t_input = torch.cat([t_batch, t_batch], dim=0)
            ctx_input = torch.cat([text_emb, uncond_emb], dim=0)
            cond_input = torch.cat([conditions, conditions], dim=0)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = dit(latent_input, t_input, context=ctx_input, conditions=cond_input)
                pred = out['sample'] if isinstance(out, dict) else out

            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)

            latents = scheduler.step(pred, t, latents)

            # Progress bar
            pct = (i + 1) / len(timesteps) * 100
            bar = "#" * int(pct // 5) + "-" * (20 - int(pct // 5))
            print(f"\r  [{bar}] {pct:5.1f}% step {i+1}/{len(timesteps)}", end="", flush=True)

    denoise_time = time.perf_counter() - t0
    print(f"\n  Denoised in {denoise_time:.1f}s ({denoise_time/num_steps*1000:.0f}ms/step)")
    print(f"  DiT peak memory: {torch.cuda.max_memory_allocated(device_dit)/1e9:.2f} GB")

    # --- VAE Decode ---
    print(f"\n[4/4] Decoding latents to video...")
    t0 = time.perf_counter()

    latents_for_vae = latents.to(device_vae)

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        video = vae.decode(latents_for_vae)

    decode_time = time.perf_counter() - t0
    print(f"  Decoded in {decode_time:.1f}s")
    print(f"  Raw video tensor: {list(video.shape)}")
    print(f"  VAE peak memory: {torch.cuda.max_memory_allocated(device_vae)/1e9:.2f} GB")

    # Normalize to [0, 1]
    video = video.float().clamp(-1, 1)
    video = (video + 1.0) / 2.0

    # --- Save Output ---
    print("\n  Saving outputs...")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "test_generation")
    os.makedirs(output_dir, exist_ok=True)

    # Save as GIF
    from lux_video_gen.utils.video_utils import video_to_gif, save_video_frames
    gif_path = os.path.join(output_dir, "test_video.gif")
    video_to_gif(video.cpu(), gif_path, fps=fps)
    print(f"  GIF saved: {gif_path}")

    # Save individual frames
    frames_dir = os.path.join(output_dir, "frames")
    save_video_frames(video.cpu(), frames_dir, prefix="frame")
    num_saved = len([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    print(f"  Frames saved: {frames_dir} ({num_saved} frames)")

    # Save as MP4 (if imageio-ffmpeg available)
    try:
        import imageio
        mp4_path = os.path.join(output_dir, "test_video.mp4")
        # (B, C, T, H, W) -> (T, H, W, C)
        vid_np = (video[0].clamp(0, 1) * 255).byte().permute(1, 2, 3, 0).cpu().numpy()
        writer = imageio.get_writer(mp4_path, fps=fps, codec='libx264', quality=8)
        for frame_idx in range(vid_np.shape[0]):
            writer.append_data(vid_np[frame_idx])
        writer.close()
        print(f"  MP4 saved: {mp4_path}")
    except Exception as e:
        print(f"  MP4 save skipped ({e})")

    # --- Summary ---
    total_time = denoise_time + decode_time
    print("\n" + "=" * 60)
    print("  Generation Complete!")
    print("=" * 60)
    print(f"  Output:     {width}x{height}, {num_frames} frames")
    print(f"  Denoise:    {denoise_time:.1f}s ({num_steps} steps)")
    print(f"  Decode:     {decode_time:.1f}s")
    print(f"  Total:      {total_time:.1f}s")
    print(f"  Output dir: {output_dir}")
    print()
    print("  NOTE: Output is abstract noise because the model has random")
    print("  (untrained) weights. Train on video data for real results.")
    print("=" * 60)

    # Cleanup
    del dit, vae, latents, video
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    generate_test_video()
