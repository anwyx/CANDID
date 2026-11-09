"""Inference-only settings for physical-consequence guided diffusion."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GuidanceConfig:
    # rho=0 is an exact vanilla DP sampling control (including RNG use).
    strength: float = 0.005
    start_timestep: int = 30  # guide at t <= 30; negative values are invalid
    dr_mean_weight: float = 1.0
    dr_max_weight: float = 0.25
    topple_weight: float = 0.1
    fall_weight: float = 0.0
    max_grad_norm: float = 1.0  # per sample, in DP-normalized action coordinates
    max_step_norm: float = 0.02  # per step L2 bound on the guidance update

    def __post_init__(self):
        for name in (
            "strength", "dr_mean_weight", "dr_max_weight", "topple_weight",
            "fall_weight", "max_grad_norm", "max_step_norm",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.start_timestep, int) or self.start_timestep < 0:
            raise ValueError("start_timestep must be a nonnegative integer")
        if self.strength > 0 and (self.max_grad_norm == 0 or self.max_step_norm == 0):
            raise ValueError("Active guidance requires positive gradient and step norms")
        if self.strength > 0 and not any((
            self.dr_mean_weight, self.dr_max_weight, self.topple_weight, self.fall_weight
        )):
            raise ValueError("Active guidance requires at least one risk weight")
