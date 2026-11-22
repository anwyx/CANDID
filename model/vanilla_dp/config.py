"""Hyperparameters taken from the ManiSkill state Diffusion Policy trainer."""
from __future__ import annotations

from dataclasses import dataclass
import math


MANISKILL_COMMIT = "62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3"


@dataclass
class DPConfig:
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    unet_dims: tuple[int, ...] = (64, 128, 256)
    n_groups: int = 8
    num_diffusion_iters: int = 100
    lr: float = 1e-4
    batch_size: int = 1024
    total_iters: int = 1_000_000
    seed: int = 1
    split_seed: int = 0
    cuda: bool = True
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    warmup_steps: int = 500
    log_freq: int = 1000
    eval_freq: int = 500
    val_max_windows: int = 4096
    val_batch_size: int = 256
    delta_loss_weight: float = 0.0
    overlap_loss_weight: float = 0.0
    overlap_stride: int = 4
    regularization_max_timestep: int = 20
    normalize_obs: bool = False
    max_abs_joint: float | None = None
    routes: tuple[str, ...] | None = None

    def __post_init__(self):
        self.unet_dims = tuple(int(value) for value in self.unet_dims)
        if min(self.obs_horizon, self.act_horizon, self.pred_horizon) < 1:
            raise ValueError("Horizons must be positive")
        if self.obs_horizon + self.act_horizon - 1 > self.pred_horizon:
            raise ValueError("Require obs_horizon + act_horizon - 1 <= pred_horizon")
        if self.num_diffusion_iters < 1 or self.total_iters < 1 or self.batch_size < 1:
            raise ValueError("Iteration counts and batch size must be positive")
        if min(self.log_freq, self.eval_freq, self.val_max_windows, self.val_batch_size) < 1:
            raise ValueError("Logging/validation frequency and sizes must be positive")
        if any(dim % self.n_groups != 0 for dim in self.unet_dims):
            raise ValueError("Each UNet channel width must be divisible by n_groups")
        downsamples = max(len(self.unet_dims) - 1, 0)
        factor = 2 ** downsamples
        if factor and self.pred_horizon % factor != 0:
            raise ValueError(
                f"pred_horizon must be divisible by {factor} for this UNet depth"
            )
        if not 0 <= self.val_ratio < 1 or not 0 < self.test_ratio < 1:
            raise ValueError("Invalid split ratios")
        if self.val_ratio + self.test_ratio >= 1:
            raise ValueError("Validation and test ratios must sum to less than 1")
        if any(not math.isfinite(w) or w < 0 for w in (self.delta_loss_weight, self.overlap_loss_weight)):
            raise ValueError("Temporal loss weights must be finite and nonnegative")
        if self.overlap_stride < 1:
            raise ValueError("overlap_stride must be positive")
        if self.delta_loss_weight > 0 or self.overlap_loss_weight > 0:
            if self.overlap_stride >= self.act_horizon:
                raise ValueError("overlap_stride must be smaller than act_horizon")
        if not 0 <= self.regularization_max_timestep < self.num_diffusion_iters:
            raise ValueError("regularization_max_timestep is outside diffusion schedule")
