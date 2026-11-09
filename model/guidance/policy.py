"""Plug-and-play ChunkPolicy with opt-in paired, same-noise diagnostics."""
from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from model.guidance.config import GuidanceConfig
from model.guidance.objective import chunk_risk
from model.guidance.sampler import ConsequenceGuidedSampler


class ConsequenceGuidedPolicy:
    def __init__(
        self, dp_policy, predictor, predictor_normalizer, config=None, *,
        diagnose=False, profile_timing=False,
    ):
        self.config = config if config is not None else GuidanceConfig()
        self.sampler = ConsequenceGuidedSampler(
            dp_policy, predictor, predictor_normalizer, self.config
        )
        self.dp_policy = dp_policy
        self.obs_horizon = int(dp_policy.obs_horizon)
        self.obs_dim = int(dp_policy.obs_dim)
        self.action_horizon = int(dp_policy.action_horizon)
        self.action_dim = int(dp_policy.action_dim)
        self.diagnose = bool(diagnose)
        self.profile_timing = bool(profile_timing)
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

    def _cuda_rng_devices(self):
        device = next(self.sampler.agent.parameters()).device
        if device.type != "cuda":
            return []
        return [device.index if device.index is not None else torch.cuda.current_device()]

    def _clip(self, action):
        if self.bounds is None:
            return action
        low, high = (bound.to(action.device) for bound in self.bounds)
        return torch.maximum(torch.minimum(action, high), low)

    def act(self, observation_window):
        obs = np.asarray(observation_window, dtype=np.float32)
        if obs.shape != (self.obs_horizon, self.obs_dim) or not np.isfinite(obs).all():
            raise ValueError("Invalid observation history for Guidance")
        device = next(self.sampler.agent.parameters()).device
        history = torch.from_numpy(obs.copy()).to(device).unsqueeze(0)
        # Baseline is evaluated under fork_rng first, preserving the next
        # guided sample's original RNG inputs and future replanning sequence.
        pair = None
        if self.diagnose:
            # Save baseline actions before the guided sample, but defer risk
            # comparison until after guided actions have been generated.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            begun = perf_counter()
            with torch.random.fork_rng(devices=self._cuda_rng_devices()):
                with torch.no_grad():
                    baseline_norm = self.sampler.agent.get_action(history)
                    baseline_action = self.dp_policy.normalizer.denormalize(baseline_norm)
                    baseline_scored = self._clip(baseline_action)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            baseline_ms = (perf_counter() - begun) * 1000
        normalized, stats = self.sampler.sample(
            history, bounds=self.bounds, profile_timing=self.profile_timing,
        )
        with torch.no_grad():
            physical = self.dp_policy.normalizer.denormalize(normalized)
            scored = self._clip(physical)
            if self.diagnose:
                risk_pair = chunk_risk(
                    history.expand(2, -1, -1),
                    torch.cat([baseline_scored, scored], dim=0),
                    self.sampler.predictor, self.sampler.predictor_normalizer,
                    self.config,
                )
                difference = scored - baseline_scored
                pair = {
                    "risk_before": float(risk_pair[0].item()),
                    "risk_after": float(risk_pair[1].item()),
                    "risk_delta": float((risk_pair[1] - risk_pair[0]).item()),
                    "action_abs_mean": float(difference.abs().mean().item()),
                    "action_abs_max": float(difference.abs().max().item()),
                    "action_l2": float(difference.flatten(start_dim=1).norm(dim=1).max().item()),
                    "baseline_action_clipped": bool(torch.any(baseline_action != baseline_scored).item()),
                    "baseline_sampling_wall_ms": baseline_ms,
                    "paired_comparison": "same_obs_same_sampling_rng_independent_baseline",
                }
                risk = pair["risk_after"]
            elif self.config.strength > 0:
                risk = chunk_risk(
                    history, scored, self.sampler.predictor,
                    self.sampler.predictor_normalizer, self.config,
                ).item()
            else:
                risk = None
            changed_by_clipping = bool(torch.any(scored != physical).item())

        decision = {
            "replan_index": len(self.decisions),
            **stats,
            "predicted_chunk_risk": risk,
            "action_clipping_needed": changed_by_clipping,
        }
        if pair is not None:
            decision["paired"] = pair
        self.decisions.append(decision)
        return physical[0].detach().cpu().numpy().astype(np.float32)

    def get_diagnostics(self):
        pairs = [d["paired"] for d in self.decisions if "paired" in d]
        timings = [d["timing_ms"] for d in self.decisions if "timing_ms" in d]
        keys = ("denoiser_ms", "predictor_gradient_ms", "ddpm_update_ms", "sampling_wall_ms")
        return {
            "mode": "gradient_guidance" if self.config.strength else "vanilla_control",
            "strength": self.config.strength,
            "start_timestep": self.config.start_timestep,
            "num_replans": len(self.decisions),
            "total_guided_steps": sum(r["guided_steps"] for r in self.decisions),
            "paired_count": len(pairs),
            "paired_risk_decreased_count": sum(p["risk_delta"] < 0 for p in pairs),
            "paired_risk_delta_mean": (
                float(np.mean([p["risk_delta"] for p in pairs])) if pairs else None
            ),
            "paired_action_abs_mean": (
                float(np.mean([p["action_abs_mean"] for p in pairs])) if pairs else None
            ),
            "timing_totals_ms": {
                key: sum(t[key] for t in timings) for key in keys
            } if timings else {},
            "baseline_sampling_total_ms": sum(
                p["baseline_sampling_wall_ms"] for p in pairs
            ),
            "decisions": self.decisions,
        }
