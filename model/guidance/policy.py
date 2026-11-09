"""Plug-and-play ChunkPolicy adapter for CANDID closed-loop rollout."""
from __future__ import annotations

import numpy as np
import torch

from model.guidance.config import GuidanceConfig
from model.guidance.objective import chunk_risk
from model.guidance.sampler import ConsequenceGuidedSampler


class ConsequenceGuidedPolicy:
    def __init__(self, dp_policy, predictor, predictor_normalizer, config=None):
        self.config = config if config is not None else GuidanceConfig()
        self.sampler = ConsequenceGuidedSampler(
            dp_policy, predictor, predictor_normalizer, self.config
        )
        self.dp_policy = dp_policy
        self.obs_horizon = int(dp_policy.obs_horizon)
        self.obs_dim = int(dp_policy.obs_dim)
        self.action_horizon = int(dp_policy.action_horizon)
        self.action_dim = int(dp_policy.action_dim)
        self.bounds = None
        self.decisions = []

    def set_action_space(self, action_space):
        if int(np.prod(action_space.shape)) != self.action_dim:
            raise ValueError("Environment action dimension disagrees with DP")
        low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
        if np.isnan(low).any() or np.isnan(high).any() or np.any(low > high):
            raise ValueError("Environment action bounds are invalid")
        self.bounds = (torch.from_numpy(low.copy()), torch.from_numpy(high.copy()))

    def begin_episode(self, seed):
        self.decisions = []
        self.dp_policy.begin_episode(int(seed))

    def act(self, observation_window):
        obs = np.asarray(observation_window, dtype=np.float32)
        if obs.shape != (self.obs_horizon, self.obs_dim) or not np.isfinite(obs).all():
            raise ValueError("Invalid observation history for Guidance")
        device = next(self.sampler.agent.parameters()).device
        history = torch.from_numpy(obs.copy()).to(device).unsqueeze(0)
        normalized, stats = self.sampler.sample(history, bounds=self.bounds)
        with torch.no_grad():
            physical = self.dp_policy.normalizer.denormalize(normalized)
            scored = physical
            if self.bounds is not None:
                low, high = (bound.to(device) for bound in self.bounds)
                scored = torch.maximum(torch.minimum(physical, high), low)
            if self.config.strength > 0:
                risk = chunk_risk(
                    history, scored, self.sampler.predictor,
                    self.sampler.predictor_normalizer, self.config,
                ).item()
            else:
                risk = None  # No predictor is used on the exact vanilla control.
            changed_by_clipping = bool(torch.any(scored != physical).item())
        self.decisions.append({
            "replan_index": len(self.decisions),
            **stats,
            "predicted_chunk_risk": risk,
            "action_clipping_needed": changed_by_clipping,
        })
        return physical[0].detach().cpu().numpy().astype(np.float32)

    def get_diagnostics(self):
        return {
            "mode": "gradient_guidance" if self.config.strength else "vanilla_control",
            "strength": self.config.strength,
            "start_timestep": self.config.start_timestep,
            "num_replans": len(self.decisions),
            "total_guided_steps": sum(r["guided_steps"] for r in self.decisions),
            "decisions": self.decisions,
        }
