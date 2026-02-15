#!/usr/bin/env python3
"""
Launch the Lux Video Generation Web UI.

Usage:
    python scripts/launch_ui.py
    python scripts/launch_ui.py --model_path checkpoints/best --share --port 7860
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Lux Web UI")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model")
    parser.add_argument("--model_variant", type=str, default="B")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--demo", action="store_true", help="Launch in demo mode (no model)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 50)
    print("Lux Video Generation - Web UI")
    print("=" * 50)
    
    pipeline = None
    
    if not args.demo and args.model_path:
        print(f"Loading model from {args.model_path}...")
        try:
            from lux_video_gen.inference.pipeline import LuxPipeline
            pipeline = LuxPipeline.from_pretrained(args.model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Launching in demo mode...")
    elif not args.demo:
        print("No model path specified. Launch with --model_path or --demo")
        print("Launching in demo mode...")
    
    from lux_video_gen.ui.gradio_app import create_ui
    create_ui(pipeline=pipeline, share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
