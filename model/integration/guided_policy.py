"""Composition of a frozen DP policy with a frozen disturbance predictor.

No imports in the predictor or DP training packages are needed.  The existing
rollout still sees a ChunkPolicy that returns unnormalized environment actions.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from model.predictor.inference import predict


class PredictorGuidedPolicy:
    """Score independent DP action chunks; optionally choose a safer candidate.

    Shadow mode samples exactly once and returns the unchanged DP output.
    Rerank mode generates K chunks but may replace candidate 0 only when the
    predicted improvement and absolute joint-target deviation pass the gates.

    Scores are *local chunk* heuristics, not calibrated event probabilities or
    guarantees of episode safety.
    """

    def __init__(
        self, dp_policy, predictor_model, predictor_normalizer, *,
        mode="shadow", num_candidates=1, dr_mean_weight=1.0,
        dr_max_weight=0.25, topple_weight=0.1, fall_weight=0.0,
        min_improvement=0.01, max_joint_deviation=0.15,
    ):
        if mode not in ("shadow", "rerank"):
            raise ValueError("mode must be 'shadow' or 'rerank'")
        if num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        if mode == "shadow" and num_candidates != 1:
            raise ValueError("Shadow mode must use exactly one DP candidate")
        for name, value in {
            "dr_mean_weight": dr_mean_weight,
            "dr_max_weight": dr_max_weight,
            "topple_weight": topple_weight,
            "fall_weight": fall_weight,
            "min_improvement": min_improvement,
            "max_joint_deviation": max_joint_deviation,
        }.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for attr in ("obs_horizon", "obs_dim", "action_horizon", "action_dim"):
            if not hasattr(dp_policy, attr):
                raise TypeError(f"DP policy missing {attr}")
        self.dp_policy = dp_policy
        self.predictor_model = predictor_model.eval()
        self.predictor_normalizer = predictor_normalizer
        self.obs_horizon = int(dp_policy.obs_horizon)
        self.obs_dim = int(dp_policy.obs_dim)
        self.action_horizon = int(dp_policy.action_horizon)
        self.action_dim = int(dp_policy.action_dim)
        if (
            self.obs_horizon != predictor_model.obs_horizon
            or self.obs_dim != predictor_model.obs_dim
            or self.action_horizon != predictor_model.action_horizon
            or self.action_dim != predictor_model.action_dim
        ):
            raise ValueError(
                "DP/Predictor checkpoint dimensions or horizons disagree: "
                f"DP obs=({self.obs_horizon},{self.obs_dim}) "
                f"action=({self.action_horizon},{self.action_dim}); "
                f"Predictor obs=({predictor_model.obs_horizon},{predictor_model.obs_dim}) "
                f"action=({predictor_model.action_horizon},{predictor_model.action_dim})"
            )
        self.mode = mode
        self.num_candidates = int(num_candidates)
        self.weights = np.array(
            [dr_mean_weight, dr_max_weight, topple_weight, fall_weight],
            dtype=np.float64,
        )
        self.min_improvement = float(min_improvement)
        self.max_joint_deviation = float(max_joint_deviation)
        self._bounds = None
        self.decisions = []

    def set_action_space(self, action_space):
        """Receive the actual environment bounds from the rollout, once/reset."""
        shape = tuple(action_space.shape)
        if int(np.prod(shape)) != self.action_dim:
            raise ValueError(f"Environment action dimension {shape} != {self.action_dim}")
        low = np.asarray(action_space.low, dtype=np.float32).reshape(self.action_dim)
        high = np.asarray(action_space.high, dtype=np.float32).reshape(self.action_dim)
        if np.isnan(low).any() or np.isnan(high).any() or np.any(low > high):
            raise ValueError("Invalid environment action bounds")
        self._bounds = (low, high)

    def begin_episode(self, seed):
        self.decisions = []
        if hasattr(self.dp_policy, "begin_episode"):
            self.dp_policy.begin_episode(seed)

    def _sample(self, obs_window):
        chunk = np.asarray(self.dp_policy.act(obs_window), dtype=np.float32)
        if chunk.shape != (self.action_horizon, self.action_dim):
            raise ValueError(f"DP returned {chunk.shape}; expected {(self.action_horizon, self.action_dim)}")
        if not np.isfinite(chunk).all():
            raise ValueError("DP returned nonfinite actions")
        return chunk

    def _dp_cuda_rng_devices(self):
        """Return the CUDA generator used by VanillaDPPolicy, if any."""
        agent = getattr(self.dp_policy, "agent", None)
        if agent is None:
            return []
        try:
            device = next(agent.parameters()).device
        except StopIteration:
            return []
        if device.type != "cuda":
            return []
        return [device.index if device.index is not None else torch.cuda.current_device()]

    def act(self, observation_window):
        obs = np.asarray(observation_window, dtype=np.float32)
        if obs.shape != (self.obs_horizon, self.obs_dim) or not np.isfinite(obs).all():
            raise ValueError(f"Invalid observation history shape/values: {obs.shape}")
        if self.mode == "rerank" and self._bounds is None:
            raise RuntimeError("Rerank mode requires environment bounds via set_action_space")

        # Candidate 0 advances the ordinary DP RNG exactly once, as in Vanilla DP.
        # Additional candidates are sampled with forked CPU/CUDA RNG state so the
        # *next* replan's candidate 0 is unaffected if no action was replaced.
        baseline = self._sample(obs)
        extras = []
        if self.num_candidates > 1:
            with torch.random.fork_rng(devices=self._dp_cuda_rng_devices()):
                extras = [self._sample(obs) for _ in range(self.num_candidates - 1)]
        chunks = np.stack([baseline, *extras])
        scored_chunks = chunks
        if self._bounds is not None:
            scored_chunks = np.clip(chunks, *self._bounds).astype(np.float32)
        device = next(self.predictor_model.parameters()).device
        obs_batch = torch.as_tensor(
            np.repeat(obs[None], self.num_candidates, axis=0),
            dtype=torch.float32, device=device,
        )
        action_batch = torch.as_tensor(scored_chunks, dtype=torch.float32, device=device)
        result = predict(self.predictor_model, self.predictor_normalizer, obs_batch, action_batch)
        keys = ("dr_mean", "dr_max", "topple_prob", "fall_prob")
        matrix = np.stack(
            [result[key].detach().cpu().numpy() for key in keys], axis=-1
        ).astype(np.float64)
        if matrix.shape != (self.num_candidates, 4) or not np.isfinite(matrix).all():
            raise RuntimeError("Predictor produced invalid risk scores")
        # Avoid negative or inconsistent raw DR changing the risk ordering.
        matrix[:, :2] = np.maximum(matrix[:, :2], 0)
        matrix[:, 1] = np.maximum(matrix[:, 1], matrix[:, 0])
        risk = matrix @ self.weights
        candidate_diagnostics = []
        for i in range(self.num_candidates):
            improvement = float(risk[0] - risk[i])
            joint_delta = float(np.max(np.abs(scored_chunks[i] - scored_chunks[0])))
            passes_improvement = improvement > 0 and improvement >= self.min_improvement
            passes_deviation = joint_delta <= self.max_joint_deviation
            if i == 0:
                reason = "baseline"
            elif not passes_improvement and not passes_deviation:
                reason = "insufficient_improvement_and_excessive_joint_deviation"
            elif not passes_improvement:
                reason = "insufficient_improvement"
            elif not passes_deviation:
                reason = "excessive_joint_deviation"
            else:
                reason = "eligible"
            candidate_diagnostics.append({
                "candidate_index": i,
                "risk_improvement": improvement,
                "max_joint_deviation": joint_delta,
                "passes_improvement": bool(passes_improvement),
                "passes_deviation": bool(passes_deviation),
                "eligible": bool(i != 0 and passes_improvement and passes_deviation),
                "reason": reason,
            })

        selected = 0
        if self.mode == "rerank":
            eligible = [
                item["candidate_index"] for item in candidate_diagnostics
                if item["eligible"]
            ]
            if eligible:
                selected = min(eligible, key=lambda i: (risk[i], i))
        for candidate in candidate_diagnostics:
            if candidate["candidate_index"] == selected and selected != 0:
                candidate["reason"] = "selected"
            elif candidate["eligible"]:
                candidate["reason"] = "eligible_not_selected"
        self.decisions.append({
            "replan_index": len(self.decisions),
            "mode": self.mode,
            "candidate_count": self.num_candidates,
            "selected": selected,
            "changed_from_baseline": selected != 0,
            "predicted_risk": risk.tolist(),
            "candidate_diagnostics": candidate_diagnostics,
            "predictions": [
                {key: float(matrix[i, j]) for j, key in enumerate(keys)}
                for i in range(self.num_candidates)
            ],
            "action_clipping_needed": [
                bool(np.any(scored_chunks[i] != chunks[i]))
                for i in range(self.num_candidates)
            ],
        })
        # Return raw absolute joint targets. The existing rollout clips once
        # before stepping; scoring uses the same clipped commands.
        return chunks[selected].copy()

    def get_diagnostics(self):
        reasons = Counter(
            item["reason"]
            for decision in self.decisions
            for item in decision["candidate_diagnostics"]
            if item["candidate_index"] != 0
        )
        return {
            "mode": self.mode,
            "num_candidates": self.num_candidates,
            "num_replans": len(self.decisions),
            "candidate_changed_count": sum(x["changed_from_baseline"] for x in self.decisions),
            "candidate_reason_counts": dict(reasons),
            "decisions": self.decisions,
        }
