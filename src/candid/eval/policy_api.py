"""Algorithm-independent policy protocol for the shared CANDID rollout.

A policy sees a state window and returns a chunk of environment actions. It
does not know how scenes are restored or how disturbance is scored.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class ChunkPolicy(Protocol):
    obs_horizon: int
    obs_dim: int
    action_dim: int
    action_horizon: int

    def act(self, observation_window: np.ndarray) -> np.ndarray:
        """Return ``[action_horizon, action_dim]`` commands in environment units."""

    def begin_episode(self, seed: int) -> None:
        """Optional. Seed any randomness used while acting."""
