"""
Distributed Training Utilities.

Provides helpers for multi-GPU training:
- Data-parallel distribution
- Model-parallel placement
- Gradient synchronization
- Process group management
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def setup_distributed(backend: str = "nccl"):
    """Initialize distributed training context."""
    if not torch.distributed.is_available():
        logger.warning("Distributed training not available")
        return False

    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.distributed.init_process_group(
            backend=backend, rank=rank, world_size=world_size
        )
        torch.cuda.set_device(local_rank)
        logger.info(f"Distributed: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return True

    return False


def cleanup_distributed():
    """Clean up distributed training."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def get_rank() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


class ModelParallelPlacer:
    """
    Places model components across multiple GPUs for model parallelism.
    
    Designed for 2-GPU setups (RTX 5060 + RTX 3070 Ti):
    - GPU 0 (primary): DiT backbone (largest model)
    - GPU 1 (auxiliary): VAE + Text Encoder + Audio Model
    """

    def __init__(self):
        self.num_gpus = torch.cuda.device_count()
        self.device_map = {}

    def create_device_map(self) -> Dict[str, torch.device]:
        """Create optimal device placement map."""
        if self.num_gpus >= 2:
            self.device_map = {
                "dit": torch.device("cuda:0"),
                "vae": torch.device("cuda:1"),
                "text_encoder": torch.device("cuda:1"),
                "audio": torch.device("cuda:1"),
                "multimodal": torch.device("cuda:1"),
            }
        elif self.num_gpus == 1:
            device = torch.device("cuda:0")
            self.device_map = {
                "dit": device,
                "vae": device,
                "text_encoder": device,
                "audio": device,
                "multimodal": device,
            }
        else:
            device = torch.device("cpu")
            self.device_map = {k: device for k in ["dit", "vae", "text_encoder", "audio", "multimodal"]}

        return self.device_map

    def place_models(self, models: Dict[str, nn.Module]):
        """Move models to their assigned devices."""
        if not self.device_map:
            self.create_device_map()

        for name, model in models.items():
            if name in self.device_map:
                device = self.device_map[name]
                model.to(device)
                logger.info(f"Placed {name} on {device}")

    def get_device(self, model_name: str) -> torch.device:
        """Get device for a model component."""
        if not self.device_map:
            self.create_device_map()
        return self.device_map.get(model_name, torch.device("cpu"))


def split_tensor_for_devices(
    tensor: torch.Tensor,
    devices: List[torch.device],
    dim: int = 0,
) -> List[torch.Tensor]:
    """Split a tensor across multiple devices."""
    chunks = tensor.chunk(len(devices), dim=dim)
    return [chunk.to(device) for chunk, device in zip(chunks, devices)]


def gather_from_devices(
    tensors: List[torch.Tensor],
    target_device: torch.device,
    dim: int = 0,
) -> torch.Tensor:
    """Gather tensors from multiple devices to one."""
    moved = [t.to(target_device) for t in tensors]
    return torch.cat(moved, dim=dim)
