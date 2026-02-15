#!/usr/bin/env python3
"""
Download Pre-Trained Models from Hugging Face.

Downloads the base models needed for Lux training:
- T5 text encoder (for semantic text understanding)
- CLIP text+vision encoder (for visual-language alignment)
- Optional: Pre-trained VAE weights

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --models t5 clip --output_dir models/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Download pre-trained models")
    parser.add_argument("--models", nargs="+", default=["t5", "clip"],
                        choices=["t5", "clip", "t5-large", "clip-large", "vae"])
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--cache_dir", type=str, default=None)
    return parser.parse_args()


MODEL_REGISTRY = {
    "t5": {
        "repo": "google/t5-v1_1-base",
        "description": "T5 v1.1 Base - Text encoder for semantic understanding",
        "size_gb": 0.9,
    },
    "t5-large": {
        "repo": "google/t5-v1_1-large",
        "description": "T5 v1.1 Large - Better text understanding (more VRAM)",
        "size_gb": 3.0,
    },
    "clip": {
        "repo": "openai/clip-vit-base-patch32",
        "description": "CLIP ViT-B/32 - Visual-language alignment",
        "size_gb": 0.6,
    },
    "clip-large": {
        "repo": "openai/clip-vit-large-patch14",
        "description": "CLIP ViT-L/14 - Better visual understanding (more VRAM)",
        "size_gb": 1.7,
    },
    "vae": {
        "repo": "stabilityai/sd-vae-ft-mse",
        "description": "Stable Diffusion VAE - Pre-trained 2D VAE (can bootstrap 3D VAE)",
        "size_gb": 0.3,
    },
}


def download_model(name: str, output_dir: str, cache_dir: str = None):
    """Download a single model."""
    info = MODEL_REGISTRY[name]
    print(f"\nDownloading {name}: {info['description']}")
    print(f"  Repository: {info['repo']}")
    print(f"  Estimated size: {info['size_gb']:.1f} GB")

    try:
        from huggingface_hub import snapshot_download
        
        local_dir = os.path.join(output_dir, name)
        os.makedirs(local_dir, exist_ok=True)
        
        path = snapshot_download(
            repo_id=info["repo"],
            local_dir=local_dir,
            cache_dir=cache_dir,
        )
        print(f"  Downloaded to: {path}")
        return True
        
    except ImportError:
        print("  Falling back to transformers AutoModel...")
        try:
            if "t5" in name:
                from transformers import T5EncoderModel, T5Tokenizer
                T5Tokenizer.from_pretrained(info["repo"], cache_dir=cache_dir)
                T5EncoderModel.from_pretrained(info["repo"], cache_dir=cache_dir)
            elif "clip" in name:
                from transformers import CLIPModel, CLIPTokenizer
                CLIPTokenizer.from_pretrained(info["repo"], cache_dir=cache_dir)
                CLIPModel.from_pretrained(info["repo"], cache_dir=cache_dir)
            print(f"  Downloaded to transformers cache")
            return True
        except Exception as e:
            print(f"  Error: {e}")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    args = parse_args()
    
    print("=" * 50)
    print("Lux Video Generation - Model Download")
    print("=" * 50)
    
    total_size = sum(MODEL_REGISTRY[m]["size_gb"] for m in args.models)
    print(f"\nModels to download: {', '.join(args.models)}")
    print(f"Estimated total size: {total_size:.1f} GB")
    print(f"Output directory: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    for name in args.models:
        success = download_model(name, args.output_dir, args.cache_dir)
        results[name] = success
    
    print(f"\n{'='*50}")
    print("Download Summary:")
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")
    
    failed = [n for n, s in results.items() if not s]
    if failed:
        print(f"\nFailed downloads: {', '.join(failed)}")
        print("Check your internet connection and try again.")
        sys.exit(1)
    else:
        print("\nAll downloads complete!")


if __name__ == "__main__":
    main()
