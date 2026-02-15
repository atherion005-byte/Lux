#!/usr/bin/env python3
"""Integration test for Lux Video Generation.

Tests all components without requiring external model downloads.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name} ({p.total_memory / 1e9:.1f} GB)")

# ===== 1. Model Instantiation =====
print("\n--- Model Instantiation ---")
from lux_video_gen.models.dit_video import LuxDiT
from lux_video_gen.models.vae3d import VideoVAE
from lux_video_gen.utils.memory import estimate_model_memory

for variant in ["Lux-S", "Lux-B", "Lux-L", "Lux-XL"]:
    dit = LuxDiT.from_config(variant)
    info = estimate_model_memory(dit)
    print(f"  {variant}: {info['total_params']/1e6:.1f}M params, "
          f"{info['inference_fp16_gb']:.2f} GB FP16, "
          f"{info['training_fp16_gb']:.2f} GB training")
    del dit

# Also test alias
dit = LuxDiT.from_config("S")
info = estimate_model_memory(dit)
print(f"  Alias 'S' -> {info['total_params']/1e6:.1f}M params OK")

vae = VideoVAE()
vae_info = estimate_model_memory(vae)
print(f"  VAE: {vae_info['total_params']/1e6:.1f}M params")

# ===== 2. DiT Forward Pass =====
print("\n--- DiT Forward Pass ---")
x = torch.randn(1, 4, 4, 8, 8)
ts = torch.tensor([0.5])
ctx = torch.randn(1, 16, 4096)  # cross_attention_dim=4096 by default
cond = torch.tensor([[256.0, 256.0, 24.0, 2.0]])
out = dit(x, ts, context=ctx, conditions=cond)
print(f"  Input: {x.shape} -> Output: {out['sample'].shape}")
assert out["sample"].shape == x.shape, "Output shape mismatch!"

# ===== 3. VAE Forward Pass =====
print("\n--- VAE Forward Pass ---")
video_in = torch.randn(1, 3, 8, 64, 64)
recon, mean, logvar = vae(video_in)
print(f"  Input: {video_in.shape} -> Recon: {recon.shape}")
latent = vae.encode_video(video_in)
print(f"  Encoded latent: {latent.shape}")
decoded = vae.decode(latent)
print(f"  Decoded: {decoded.shape}")

# ===== 4. Noise Scheduler =====
print("\n--- Noise Scheduler ---")
from lux_video_gen.training.scheduler import FlowMatchingScheduler, create_scheduler

sched = FlowMatchingScheduler()
t = sched.sample_timesteps(4, device="cpu")
print(f"  Timesteps: {t}")

noise = torch.randn_like(x)
noisy = sched.add_noise(x, noise, t[:1])
print(f"  Noisy: {noisy.shape}")

vel = sched.get_velocity(x, noise, t[:1])
print(f"  Velocity: {vel.shape}")

timesteps = sched.get_sampling_timesteps(10, device="cpu")
print(f"  Schedule ({len(timesteps)} steps): t0={timesteps[0].item():.3f} -> t_end={timesteps[-1].item():.3f}")

stepped = sched.step(noise, timesteps[0], noisy)
print(f"  Step result: {stepped.shape}")

# Test other schedulers
for name in ["ddpm"]:
    s = create_scheduler(name)
    print(f"  Created {name} scheduler OK")

# Test DDIM (wraps DDPM)
from lux_video_gen.training.scheduler import DDIMScheduler, DDPMScheduler
ddpm = DDPMScheduler()
ddim = DDIMScheduler(ddpm, num_inference_steps=50)
print(f"  Created DDIM scheduler OK")

# ===== 5. Loss Functions =====
print("\n--- Loss Functions ---")
from lux_video_gen.training.losses import DiffusionLoss, CombinedTrainingLoss

loss_fn = DiffusionLoss()
pred = torch.randn_like(x)
target = torch.randn_like(x)
loss = loss_fn(pred, target, t[:1])
print(f"  Diffusion loss: {loss.item():.4f}")

combined = CombinedTrainingLoss()
closs = combined(pred, target, t[:1])
print(f"  Combined loss total: {closs['total'].item():.4f}")
for k, v in closs.items():
    print(f"    {k}: {v.item():.4f}")

# ===== 6. EMA =====
print("\n--- EMA ---")
from lux_video_gen.training.ema import EMAModel
ema = EMAModel(dit, decay=0.999)
ema.update(dit)
ema.store(dit)
ema.copy_to(dit)
ema.restore(dit)
print("  EMA: update/store/copy/restore OK")

# ===== 7. Optimizer =====
print("\n--- Optimizer ---")
from lux_video_gen.training.optimizer import create_optimizer, create_scheduler as create_lr_sched

opt = create_optimizer(dit, learning_rate=1e-4)
lr_sched = create_lr_sched(opt, num_warmup_steps=10, num_training_steps=100)
print(f"  Optimizer: {type(opt).__name__}, LR scheduler: {type(lr_sched).__name__}")

# ===== 8. Configs =====
print("\n--- Configs ---")
from lux_video_gen.configs.model_config import get_model_config, MODEL_CONFIGS
from lux_video_gen.configs.training_config import get_training_config, PRESETS
from lux_video_gen.configs.inference_config import InferenceConfig

for name in MODEL_CONFIGS:
    cfg = get_model_config(name)
    print(f"  Model {name}: {cfg.name}, hidden={cfg.hidden_size}, depth={cfg.depth}")

for preset in PRESETS:
    tcfg = get_training_config(preset)
    print(f"  Training '{preset}': variant={tcfg.model_variant}, batch={tcfg.batch_size}x{tcfg.gradient_accumulation_steps}")

icfg = InferenceConfig()
print(f"  Inference: {icfg.width}x{icfg.height}, {icfg.num_frames}f, {icfg.num_inference_steps} steps")

# ===== 9. Samplers =====
print("\n--- Samplers ---")
from lux_video_gen.inference.sampler import create_sampler, EulerSampler, DPMSolverPP

for name in ["euler", "euler_a", "dpm_solver_pp", "unipc"]:
    s = create_sampler(name)
    print(f"  Created '{name}': {type(s).__name__}")

# ===== 10. Video Processor =====
print("\n--- Video Processor ---")
from lux_video_gen.inference.video_processor import VideoProcessor
processor = VideoProcessor(enable_interpolation=False)
video = torch.rand(1, 3, 8, 64, 64)
processed = processor.process(video)
print(f"  Process: {video.shape} -> {processed.shape}")
np_video = processor.to_numpy(video)
print(f"  To numpy: {np_video.shape} dtype={np_video.dtype}")

# ===== 11. Audio Sync =====
print("\n--- Audio Sync ---")
from lux_video_gen.inference.audio_sync import AudioVideoSyncProcessor
sync = AudioVideoSyncProcessor()
motion = sync.compute_motion_energy(video)
print(f"  Motion energy: {motion.shape}")

# ===== 12. Utils =====
print("\n--- Utils ---")
from lux_video_gen.utils.memory import get_gpu_memory_info, get_attention_backend, estimate_model_memory
from lux_video_gen.utils.distributed import ModelParallelPlacer
from lux_video_gen.utils.logging_utils import MetricTracker, ProgressTimer
from lux_video_gen.utils.checkpoint import CheckpointManager

backend = get_attention_backend()
print(f"  Attention backend: {backend}")

tracker = MetricTracker()
tracker.update({"loss": 0.5, "lr": 1e-4})
tracker.update({"loss": 0.3, "lr": 1e-4})
print(f"  Metrics: {tracker}")

timer = ProgressTimer(100)
timer.step()
print(f"  Timer: {timer.format()}")

placer = ModelParallelPlacer()
dmap = placer.create_device_map()
print(f"  Device map: {dmap}")

print("\n" + "=" * 60)
print("  ALL INTEGRATION TESTS PASSED!")
print("=" * 60)
