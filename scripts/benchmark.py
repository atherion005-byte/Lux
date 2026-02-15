#!/usr/bin/env python3
"""
Benchmarking Script for Lux Video Generation.

Measures:
- Model parameter counts and memory usage
- Inference speed at various resolutions
- Throughput (frames/second)
- Component-level profiling

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --model_variant S --resolutions 256 512
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Lux models")
    parser.add_argument("--model_variant", type=str, default="B", choices=["S", "B", "L", "XL"])
    parser.add_argument("--resolutions", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    return parser.parse_args()


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def benchmark_dit(model, resolution, num_frames, device, dtype, num_steps=10, warmup=2, runs=5):
    """Benchmark DiT forward pass."""
    model.eval()
    
    batch_size = 1
    latent_t = max(1, num_frames // 4)
    latent_h = resolution // 8
    latent_w = resolution // 8
    
    latent = torch.randn(batch_size, 4, latent_t, latent_h, latent_w, device=device, dtype=dtype)
    timestep = torch.tensor([0.5], device=device)
    context = torch.randn(batch_size, 77, 4096, device=device, dtype=dtype)
    conditions = torch.tensor([[resolution, resolution, 24, num_frames / 24.0]], device=device)
    
    # Warmup
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        for _ in range(warmup):
            model(latent, timestep, context=context, conditions=conditions)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(latent, timestep, context=context, conditions=conditions)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


def benchmark_vae(model, resolution, num_frames, device, dtype, warmup=2, runs=5):
    """Benchmark VAE encode/decode."""
    model.eval()
    
    video = torch.randn(1, 3, num_frames, resolution, resolution, device=device, dtype=dtype)
    
    # Warmup
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        for _ in range(warmup):
            try:
                mean, log_var = model.encode(video)
            except Exception:
                return {"encode_ms": "N/A", "decode_ms": "N/A"}
    
    torch.cuda.synchronize()
    
    # Encode benchmark
    encode_times = []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            mean, log_var = model.encode(video)
            torch.cuda.synchronize()
            encode_times.append(time.perf_counter() - start)
    
    # Reparameterize for decode input
    z = model.reparameterize(mean, log_var)
    
    # Decode benchmark
    decode_times = []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.decode(z)
            torch.cuda.synchronize()
            decode_times.append(time.perf_counter() - start)
    
    return {
        "encode_ms": sum(encode_times) / len(encode_times) * 1000,
        "decode_ms": sum(decode_times) / len(decode_times) * 1000,
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Lux Video Generation - Benchmark")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    else:
        print("WARNING: No CUDA GPU, benchmarking on CPU")
    
    # Initialize models
    print(f"\nLoading {args.model_variant} model...")
    from lux_video_gen.models.dit_video import LuxDiT
    from lux_video_gen.models.vae3d import VideoVAE
    
    dit = LuxDiT.from_config(args.model_variant).to(device).to(dtype)
    
    total, trainable = count_parameters(dit)
    print(f"\nDiT ({args.model_variant}):")
    print(f"  Total parameters: {total / 1e6:.1f}M")
    print(f"  Trainable: {trainable / 1e6:.1f}M")
    print(f"  FP16 memory: {total * 2 / 1e9:.2f} GB")
    print(f"  Training (FP16+Adam): {(total * 2 + trainable * 10) / 1e9:.2f} GB")
    
    # DiT benchmarks
    print(f"\n{'Resolution':<12} {'Frames':<8} {'Mean (ms)':<12} {'Min (ms)':<10} {'Max (ms)':<10} {'FPS':<8}")
    print("-" * 60)
    
    for res in args.resolutions:
        try:
            result = benchmark_dit(
                dit, res, args.num_frames, device, dtype,
                warmup=args.warmup, runs=args.runs
            )
            fps = args.num_frames / (result["mean_ms"] / 1000) if result["mean_ms"] > 0 else 0
            print(f"{res:<12} {args.num_frames:<8} {result['mean_ms']:<12.1f} {result['min_ms']:<10.1f} {result['max_ms']:<10.1f} {fps:<8.1f}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{res:<12} {args.num_frames:<8} OOM")
                torch.cuda.empty_cache()
            else:
                raise
    
    # VAE benchmarks
    print(f"\nVAE Benchmark:")
    try:
        vae = VideoVAE().to(device).to(dtype)
        vae_params, _ = count_parameters(vae)
        print(f"  Parameters: {vae_params / 1e6:.1f}M")
        
        for res in [256]:  # VAE is memory hungry
            result = benchmark_vae(vae, res, args.num_frames, device, dtype, warmup=args.warmup, runs=args.runs)
            print(f"  {res}x{res} x {args.num_frames}f: Encode {result['encode_ms']:.1f}ms, Decode {result['decode_ms']:.1f}ms")
        del vae
    except RuntimeError as e:
        print(f"  VAE benchmark failed: {e}")
    
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
