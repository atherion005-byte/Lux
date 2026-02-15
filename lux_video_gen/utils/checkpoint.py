"""
Checkpoint Management Utilities.

Handles saving, loading, and managing training checkpoints:
- Automatic checkpoint rotation
- Safe checkpoint saving (atomic writes)
- Checkpoint metadata tracking
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages training checkpoints with rotation and metadata.
    
    Features:
    - Keep only the N most recent checkpoints
    - Track best checkpoints by metric
    - Atomic saves to prevent corruption
    - Metadata for checkpoint provenance
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best: bool = True,
        best_metric: str = "loss",
        best_mode: str = "min",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_mode = best_mode
        self.best_value = float("inf") if best_mode == "min" else -float("inf")
        self.saved_checkpoints = []

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a checkpoint.
        
        Returns path to saved checkpoint.
        """
        name = f"checkpoint-{step}"
        ckpt_path = self.checkpoint_dir / name
        tmp_path = self.checkpoint_dir / f".tmp_{name}"

        # Save to temp first (atomic)
        tmp_path.mkdir(parents=True, exist_ok=True)

        state = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        }

        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()

        if extra:
            state.update(extra)

        torch.save(state, tmp_path / "training_state.pt")

        # Save metadata
        metadata = {
            "step": step,
            "epoch": epoch,
            "metrics": metrics or {},
        }
        with open(tmp_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Atomic rename
        if ckpt_path.exists():
            shutil.rmtree(ckpt_path)
        tmp_path.rename(ckpt_path)

        self.saved_checkpoints.append(str(ckpt_path))
        logger.info(f"Saved checkpoint: {ckpt_path}")

        # Rotate checkpoints
        self._rotate()

        # Track best
        if self.save_best and metrics and self.best_metric in metrics:
            value = metrics[self.best_metric]
            is_better = (
                (self.best_mode == "min" and value < self.best_value) or
                (self.best_mode == "max" and value > self.best_value)
            )
            if is_better:
                self.best_value = value
                best_path = self.checkpoint_dir / "best"
                if best_path.exists():
                    shutil.rmtree(best_path)
                shutil.copytree(str(ckpt_path), str(best_path))
                logger.info(f"New best {self.best_metric}: {value:.4f}")

        return str(ckpt_path)

    def load(
        self,
        path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Any] = None,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        If path is None, loads the latest checkpoint.
        """
        if path is None:
            path = self.get_latest()
            if path is None:
                raise FileNotFoundError("No checkpoints found")

        ckpt_path = Path(path)
        state_file = ckpt_path / "training_state.pt"

        if not state_file.exists():
            # Try direct file
            state_file = ckpt_path
            if not state_file.exists():
                raise FileNotFoundError(f"No checkpoint at {path}")

        state = torch.load(state_file, map_location=map_location, weights_only=False)

        if model is not None and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        logger.info(f"Loaded checkpoint from {path} (step {state.get('step', '?')})")
        return state

    def get_latest(self) -> Optional[str]:
        """Get path to the most recent checkpoint."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        )
        if checkpoints:
            return str(checkpoints[-1])
        return None

    def _rotate(self):
        """Remove old checkpoints beyond max_checkpoints."""
        while len(self.saved_checkpoints) > self.max_checkpoints:
            old = self.saved_checkpoints.pop(0)
            if os.path.exists(old):
                shutil.rmtree(old)
                logger.debug(f"Removed old checkpoint: {old}")
