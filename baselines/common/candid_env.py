"""Bridge ManiSkill's state_dict observations to CANDID's canonical flat vector."""
from __future__ import annotations

import importlib
import json

import gymnasium as gym
import numpy as np
import torch

from candid.data.adapter import canonical_observation, observation_vector


def register_candid_env(module: str) -> None:
    importlib.import_module(module)


def parse_env_kwargs(value: str) -> dict:
    extra = json.loads(value)
    if not isinstance(extra, dict):
        raise ValueError("--env-kwargs-json must decode to a JSON object")
    restricted = {"control_mode", "obs_mode", "reward_mode", "sim_backend",
                  "render_mode", "num_envs", "robot_uids", "reconfiguration_freq"}
    conflict = restricted.intersection(extra)
    if conflict:
        raise ValueError(f"Reserved environment keys: {sorted(conflict)}")
    return extra


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def num_distractors(base) -> int:
    for key in ("distractor_actors", "clutter_cubes", "clutter_objects"):
        if hasattr(base, key):
            return len(getattr(base, key))
    raise RuntimeError("CANDID environment has no distractor actor registry")


def canonical_live(raw_obs: dict, n: int) -> np.ndarray:
    groups = {part: {key: _to_numpy(value) for key, value in raw_obs[part].items()}
              for part in ("agent", "extra")}
    vector = observation_vector(canonical_observation(groups, n))
    if not np.isfinite(vector).all():
        raise ValueError("Nonfinite CANDID observation")
    return vector.astype(np.float32)


class CanonicalStateWrapper(gym.ObservationWrapper):
    """Preserve ManiSkill [num_envs, D] torch observations for its vector wrapper."""

    def __init__(self, env):
        super().__init__(env)
        self.n_distractors = num_distractors(env.unwrapped)
        first, _ = env.reset()
        flat = self.observation(first)
        if flat.ndim != 2 or flat.shape[0] != env.unwrapped.num_envs:
            raise ValueError(f"Expected [N,D] state, got {tuple(flat.shape)}")
        size = int(flat.shape[1])
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(size,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(env.unwrapped.num_envs, size), dtype=np.float32
        )

    def observation(self, observation):
        value = canonical_live(observation, self.n_distractors)
        return torch.as_tensor(value, dtype=torch.float32,
                               device=self.env.unwrapped.device)


def restore_cpu_canonical_space(cpu_env, canonical_space):
    """Restore the flat observation declaration overwritten by CPUGymWrapper.

    ManiSkill CPUGymWrapper exposes base_env.single_observation_space, which
    is the raw state_dict (Dict) even after CanonicalStateWrapper has converted
    observations to a flat state. Gym's SyncVectorEnv requires these to match.
    """
    if not isinstance(canonical_space, gym.spaces.Box) or len(canonical_space.shape) != 1:
        raise TypeError("CPU canonical observation space must be a flat Box")
    cpu_env.observation_space = canonical_space
    return cpu_env
