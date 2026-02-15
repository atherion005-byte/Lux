"""
Memory Optimization Utilities.

Provides tools for training and inference on consumer GPUs:
- Memory-efficient attention selection
- Gradient checkpointing helpers
- Model offloading between CPU and GPU
- VRAM monitoring and management
"""

import gc
import logging
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_gpu_memory_info(device: int = 0) -> dict:
    """Get GPU memory usage info."""
    if not torch.cuda.is_available():
        return {"total": 0, "used": 0, "free": 0}

    total = torch.cuda.get_device_properties(device).total_memory
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    free = total - reserved

    return {
        "total_gb": total / 1e9,
        "reserved_gb": reserved / 1e9,
        "allocated_gb": allocated / 1e9,
        "free_gb": free / 1e9,
    }


def log_memory_usage(prefix: str = "", device: int = 0):
    """Log current GPU memory usage."""
    info = get_gpu_memory_info(device)
    logger.info(
        f"{prefix} GPU {device}: "
        f"{info['allocated_gb']:.2f}GB allocated / "
        f"{info['reserved_gb']:.2f}GB reserved / "
        f"{info['total_gb']:.2f}GB total"
    )


def clear_memory():
    """Clear GPU memory caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def enable_gradient_checkpointing(model: nn.Module, enable: bool = True):
    """Enable gradient checkpointing on a model."""
    if hasattr(model, "gradient_checkpointing_enable"):
        if enable:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
    else:
        for module in model.modules():
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = enable


def get_attention_backend() -> str:
    """Detect best available attention implementation."""
    # Check for Flash Attention 2
    try:
        import flash_attn
        logger.info("Using Flash Attention 2")
        return "flash"
    except ImportError:
        pass

    # Check for xformers
    try:
        import xformers.ops
        logger.info("Using xformers memory-efficient attention")
        return "xformers"
    except ImportError:
        pass

    # Check for PyTorch SDPA
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        logger.info("Using PyTorch SDPA")
        return "sdpa"

    logger.info("Using vanilla attention (high memory usage)")
    return "vanilla"


class ModelOffloader:
    """
    Manages model offloading between CPU and GPU.
    
    Useful when models don't fit in VRAM simultaneously.
    Moves models to GPU on-demand and back to CPU after use.
    """

    def __init__(self, target_device: torch.device = torch.device("cuda:0")):
        self.target_device = target_device
        self.offloaded_models = {}

    def register(self, name: str, model: nn.Module):
        """Register model for offloading management."""
        self.offloaded_models[name] = {
            "model": model,
            "on_gpu": False,
        }
        model.to("cpu")

    @contextmanager
    def use(self, name: str):
        """Context manager to temporarily move model to GPU."""
        if name not in self.offloaded_models:
            yield
            return

        entry = self.offloaded_models[name]
        model = entry["model"]

        try:
            # Move to GPU
            model.to(self.target_device)
            entry["on_gpu"] = True
            yield model
        finally:
            # Move back to CPU
            model.to("cpu")
            entry["on_gpu"] = False
            clear_memory()

    def offload_all(self):
        """Move all models to CPU."""
        for entry in self.offloaded_models.values():
            if entry["on_gpu"]:
                entry["model"].to("cpu")
                entry["on_gpu"] = False
        clear_memory()


class MemoryEfficientExecution:
    """
    Context manager for memory-efficient model execution.
    
    Combines autocast, gradient checkpointing, and memory clearing.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float16,
        enable_checkpointing: bool = True,
    ):
        self.dtype = dtype
        self.enable_checkpointing = enable_checkpointing

    @contextmanager
    def __call__(self, model: nn.Module):
        """Execute model with memory optimizations."""
        if self.enable_checkpointing:
            enable_gradient_checkpointing(model, True)

        try:
            with torch.cuda.amp.autocast(dtype=self.dtype):
                yield
        finally:
            if self.enable_checkpointing:
                enable_gradient_checkpointing(model, False)
            clear_memory()


def estimate_model_memory(model: nn.Module) -> dict:
    """Estimate memory requirements for a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Memory estimates (bytes)
    fp32_mem = total_params * 4
    fp16_mem = total_params * 2
    # Training: params + gradients + optimizer states (Adam: 2 states)
    training_fp16 = total_params * 2 + trainable_params * (2 + 4 + 4)  # params + grad + states
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "inference_fp32_gb": fp32_mem / 1e9,
        "inference_fp16_gb": fp16_mem / 1e9,
        "training_fp16_gb": training_fp16 / 1e9,
    }
