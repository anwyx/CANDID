"""Configuration for the independent disturbance predictor."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PredictorConfig:
    obs_horizon: int = 2
    action_horizon: int = 8
    hidden: int = 256
    lr: float = 1e-3
    batch_size: int = 64
    steps: int = 10_000
    seed: int = 0
    split_seed: int = 0
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    device: str = "auto"
    log_freq: int = 100
    eval_freq: int = 500
    val_max_windows: int = 4096
    val_batch_size: int = 256
    allow_missing_event_labels: bool = False

    def __post_init__(self):
        if self.obs_horizon < 1 or self.action_horizon < 1:
            raise ValueError("Horizons must be positive")
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("Steps and batch size must be positive")
        if min(self.log_freq, self.eval_freq, self.val_max_windows, self.val_batch_size) < 1:
            raise ValueError("Logging and validation settings must be positive")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
