"""
Training Configuration.

Pre-defined training configurations for different stages
and hardware setups.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    # Stage
    stage: str = "dit"  # "vae", "dit", "multimodal", "finetune"
    model_variant: str = "B"
    
    # Data
    data_dir: str = "datasets"
    metadata_file: Optional[str] = None
    num_workers: int = 4
    
    # Video settings
    resolution: int = 256
    num_frames: int = 17
    fps: int = 24
    
    # Batch / Accumulation
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    effective_batch_size: int = 8  # batch_size * accumulation * num_gpus
    
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Schedule
    num_epochs: int = 100
    warmup_steps: int = 1000
    lr_scheduler: str = "cosine"
    
    # Mixed precision
    mixed_precision: str = "fp16"
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    # Noise scheduler
    noise_scheduler: str = "flow_matching"
    prediction_type: str = "v_prediction"
    
    # CFG
    cfg_dropout_prob: float = 0.1
    
    # Checkpointing
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 5000
    max_checkpoints: int = 5
    
    # Logging
    log_every_n_steps: int = 100
    use_wandb: bool = False
    use_tensorboard: bool = True
    project_name: str = "lux-video-gen"
    
    # Memory optimization
    gradient_checkpointing: bool = True
    enable_xformers: bool = True
    vae_slicing: bool = True
    
    # Resume
    resume_from: Optional[str] = None


# Preset configurations for different hardware
PRESETS: Dict[str, dict] = {
    "consumer_2gpu": {
        "model_variant": "B",
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "resolution": 256,
        "num_frames": 17,
        "mixed_precision": "fp16",
        "gradient_checkpointing": True,
        "enable_xformers": True,
    },
    "consumer_1gpu": {
        "model_variant": "S",
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "resolution": 256,
        "num_frames": 9,
        "mixed_precision": "fp16",
        "gradient_checkpointing": True,
    },
    "a100_80gb": {
        "model_variant": "L",
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "resolution": 512,
        "num_frames": 49,
        "mixed_precision": "bf16",
    },
    "multi_a100": {
        "model_variant": "XL",
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "resolution": 1024,
        "num_frames": 49,
        "mixed_precision": "bf16",
    },
}


def get_training_config(preset: str = "consumer_2gpu", **overrides) -> TrainingConfig:
    """Get training config for a hardware preset with optional overrides."""
    base = TrainingConfig()
    
    if preset in PRESETS:
        for k, v in PRESETS[preset].items():
            if hasattr(base, k):
                setattr(base, k, v)
    
    for k, v in overrides.items():
        if hasattr(base, k):
            setattr(base, k, v)
    
    # Compute effective batch size
    base.effective_batch_size = base.batch_size * base.gradient_accumulation_steps
    
    return base
