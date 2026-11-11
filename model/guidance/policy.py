"""Paired Vanilla DP and consequence-guided samples from one observation.

The guided sample is the only one that advances the real RNG. The vanilla
sample is drawn inside ``torch.random.fork_rng``, so turning the comparison
on does not change the guided actions.
"""
from __future__ import annotations

import numpy as np
import torch

from model.guidance.objective import risk_from_predictions
from model.guidance.sampler import ConsequenceGuidedSampler
from model.predictor.inference import predict


class ConsequenceGuidedPolicy:
    def __init__(
        self,
        dp_policy,
        predictor,
        predictor_normalizer,
        *,
        strength=0.0,
        start_timestep=30,
        weights=(1.0, 0.25, 0.1, 0.0),
        obs_normalizer=None,
    ):
        if strength < 0:
            raise ValueError("strength must be nonnegative")
        if start_timestep < 0:
            raise ValueError("start_timestep must be nonnegative")
        weights = tuple(float(value) for value in weights)
        if len(weights) != 4 or any(value < 0 or not np.isfinite(value) for value in weights):
            raise ValueError("weights must be four nonnegative finite values")
        for attr in ("obs_horizon", "obs_dim", "action_horizon", "action_dim", "agent", "normalizer"):
            if not hasattr(dp_policy, attr):
                raise TypeError(f"DP policy missing {attr}")
        if (
            predictor.obs_horizon != dp_policy.obs_horizon
            or predictor.obs_dim != dp_policy.obs_dim
            or predictor.action_horizon != dp_policy.action_horizon
            or predictor.action_dim != dp_policy.action_dim
        ):
            raise ValueError("DP and predictor horizons or dimensions disagree")
        self.dp_policy = dp_policy
        self.predictor = predictor.eval()
        self.predictor_normalizer = predictor_normalizer
        self.obs_horizon = int(dp_policy.obs_horizon)
        self.obs_dim = int(dp_policy.obs_dim)
        self.action_horizon = int(dp_policy.action_horizon)
        self.action_dim = int(dp_policy.action_dim)
        self.strength = float(strength)
        self.start_timestep = int(start_timestep)
        self.weights = weights
        self.obs_normalizer = obs_normalizer
        self.bounds = None
        self.record_pair = False
        self.trace = False
        self.last_comparison = None
        self.sampler = ConsequenceGuidedSampler(
            dp_policy.agent,
            dp_policy.normalizer,
            predictor,
            predictor_normalizer,
            weights,
        )

    def set_action_space(self, action_space):
        low = np.asarray(action_space.low, dtype=np.float32).reshape(self.action_dim)
        high = np.asarray(action_space.high, dtype=np.float32).reshape(self.action_dim)
        if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high):
            raise ValueError("Invalid action bounds")
        self.bounds = (low, high)

    def set_bounds(self, low, high):
        low = np.asarray(low, dtype=np.float32).reshape(self.action_dim)
        high = np.asarray(high, dtype=np.float32).reshape(self.action_dim)
        if np.any(low > high):
            raise ValueError("Invalid action bounds")
        self.bounds = (low, high)

    def begin_episode(self, seed):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def act(self, observation_window):
        if self.record_pair:
            return self.compare(observation_window)["guided_raw"]
        normalized, _rows = self._sample(observation_window, self.strength, trace=False)
        return self._physical(normalized)

    def compare(self, observation_window):
        """Vanilla and guided chunks from the same observation and RNG state."""
        devices = _cuda_devices(self.dp_policy)
        with torch.random.fork_rng(devices=devices):
            vanilla, _unused = self._sample(observation_window, 0.0, trace=False)
        guided, rows = self._sample(
            observation_window,
            self.strength,
            trace=self.trace,
        )
        vanilla_raw = self._physical(vanilla)
        guided_raw = self._physical(guided)
        vanilla_clipped = _clip(vanilla_raw, self.bounds)
        guided_clipped = _clip(guided_raw, self.bounds)
        scores = self._score(observation_window, vanilla_clipped, guided_clipped)
        delta = guided_clipped - vanilla_clipped
        result = {
            "paired_same_obs_and_rng": True,
            "vanilla_raw": vanilla_raw,
            "guided_raw": guided_raw,
            "vanilla_clipped": vanilla_clipped,
            "guided_clipped": guided_clipped,
            "scores": scores,
            "delta_dr_mean": scores["guided"]["dr_mean"] - scores["vanilla"]["dr_mean"],
            "delta_dr_max": scores["guided"]["dr_max"] - scores["vanilla"]["dr_max"],
            "delta_risk": scores["guided"]["weighted_risk"] - scores["vanilla"]["weighted_risk"],
            "action_abs_mean": float(np.mean(np.abs(delta))),
            "action_abs_max": float(np.max(np.abs(delta))),
            "action_l2": float(np.linalg.norm(delta)),
            "action_abs_mean_per_joint": np.mean(np.abs(delta), axis=0).astype(np.float64),
            "guided_diffusion_steps": sum(1 for row in rows if row["guided"]),
            "diffusion_trace": rows,
        }
        self.last_comparison = result
        return result

    def _sample(self, observation_window, strength, trace):
        window = self._window(observation_window)
        device = next(self.dp_policy.agent.parameters()).device
        obs = torch.from_numpy(window).unsqueeze(0).to(device)
        normalized, rows = self.sampler.sample(
            obs,
            strength=strength,
            start_timestep=self.start_timestep,
            trace=trace,
            obs_normalizer=self.obs_normalizer,
        )
        return normalized, rows

    def _window(self, observation_window):
        window = np.asarray(observation_window, dtype=np.float32)
        if window.shape != (self.obs_horizon, self.obs_dim) or not np.isfinite(window).all():
            raise ValueError(f"Invalid observation window {window.shape}")
        return window

    def _physical(self, normalized):
        command = self.dp_policy.normalizer.denormalize(normalized)
        value = command[0].detach().cpu().numpy().astype(np.float32)
        if value.shape != (self.action_horizon, self.action_dim) or not np.isfinite(value).all():
            raise RuntimeError(f"Invalid action chunk {value.shape}")
        return value

    def _score(self, observation_window, vanilla, guided):
        device = next(self.predictor.parameters()).device
        obs = torch.as_tensor(
            np.repeat(self._window(observation_window)[None], 2, axis=0),
            dtype=torch.float32,
            device=device,
        )
        actions = torch.as_tensor(
            np.stack((vanilla, guided)),
            dtype=torch.float32,
            device=device,
        )
        predicted = predict(self.predictor, self.predictor_normalizer, obs, actions)
        weights = np.asarray(self.weights, dtype=np.float64)
        scored = []
        for index, name in enumerate(("vanilla", "guided")):
            values = {
                "dr_mean": float(predicted["dr_mean"][index]),
                "dr_max": float(predicted["dr_max"][index]),
                "topple_score": float(predicted["topple_prob"][index]),
                "fall_score": float(predicted["fall_prob"][index]),
            }
            values["weighted_risk"] = float(risk_from_predictions(
                values["dr_mean"],
                values["dr_max"],
                values["topple_score"],
                values["fall_score"],
                weights,
            ))
            scored.append((name, values))
        return dict(scored)


def _clip(actions, bounds):
    if bounds is None:
        return actions.copy()
    return np.clip(actions, bounds[0], bounds[1]).astype(np.float32)


def _cuda_devices(dp_policy):
    try:
        device = next(dp_policy.agent.parameters()).device
    except StopIteration:
        return []
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]
