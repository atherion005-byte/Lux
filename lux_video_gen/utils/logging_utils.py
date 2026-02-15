"""
Logging Utilities.

Provides structured logging for training and inference:
- Console + file logging
- Progress bars
- Metric tracking
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    log_file: str = "lux.log",
) -> logging.Logger:
    """Setup project-wide logging."""
    root_logger = logging.getLogger("lux_video_gen")
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root_logger.addHandler(file_handler)

    return root_logger


class MetricTracker:
    """Track and average training metrics."""

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = 0.0
                self.counts[k] = 0
            self.metrics[k] += v
            self.counts[k] += 1

    def average(self) -> Dict[str, float]:
        return {
            k: self.metrics[k] / max(self.counts[k], 1)
            for k in self.metrics
        }

    def reset(self):
        self.metrics.clear()
        self.counts.clear()

    def __str__(self) -> str:
        avg = self.average()
        return " | ".join(f"{k}: {v:.4f}" for k, v in avg.items())


class ProgressTimer:
    """Track elapsed and estimated time."""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.start_time = time.time()
        self.step_count = 0

    def step(self):
        self.step_count += 1

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def eta(self) -> float:
        if self.step_count == 0:
            return 0
        rate = self.elapsed / self.step_count
        remaining = self.total_steps - self.step_count
        return rate * remaining

    def format(self) -> str:
        elapsed = self._fmt_time(self.elapsed)
        eta = self._fmt_time(self.eta)
        pct = 100 * self.step_count / max(self.total_steps, 1)
        return f"{self.step_count}/{self.total_steps} ({pct:.1f}%) [{elapsed} < {eta}]"

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
