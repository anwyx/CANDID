"""Pure sampling and action perturbation helpers for physically relabeled chunks.

Expert labels are used ONLY to choose useful probe states. Counterfactual
labels must always come from an actual new simulator rollout.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from candid.data.metrics import disturbance_from_poses


@dataclass(frozen=True)
class Probe:
    scene_id: str
    trajectory_id: str
    step: int
    stratum: str
    trajectory: dict


def stable_seed(seed: int, trajectory_id: str, step: int) -> int:
    payload = f"{seed}:{trajectory_id}:{step}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def choose_probes(dataset, allowed_scene_ids, *, count, seed=0,
                  action_horizon=8, obs_horizon=2, max_per_scene=4,
                  max_prefix_steps=None):
    """Stratify by episode-local onset, high disturbance, and safe controls.

    At most one probe is taken per source trajectory to limit correlation.
    A small deterministic subset of windows is scanned for high/low DR,
    avoiding loading all HDF5 observation/action sequences during selection.
    """
    if min(count, action_horizon, obs_horizon, max_per_scene) < 1:
        raise ValueError("count, horizons and max_per_scene must be positive")
    if max_prefix_steps is not None and max_prefix_steps < obs_horizon - 1:
        raise ValueError("max_prefix_steps is too small")
    root = Path(dataset)
    allowed = set(allowed_scene_ids)
    if not allowed:
        raise ValueError("No allowed train scenes")
    candidates = [r for r in _read_jsonl(root / "trajectories.jsonl")
                  if r["scene_id"] in allowed and r.get("saved_hdf5")
                  and int(r["num_steps"]) >= obs_horizon - 1 + action_horizon]
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    buckets = {"pre_topple": [], "high_dr": [], "low_dr": []}
    for row in candidates:
        t_min = obs_horizon - 1
        t_max = int(row["num_steps"]) - action_horizon
        if max_prefix_steps is not None:
            t_max = min(t_max, max_prefix_steps)
        if t_max < t_min:
            continue
        with np.load(root / row["disturbance_trace"], allow_pickle=False) as archive:
            poses = archive["distractor_poses"]
            sizes = archive["object_sizes_m"]
            if poses.shape[0] != int(row["num_steps"]) + 1:
                raise ValueError(f"Invalid trace length: {row['trajectory_id']}")
            if "distractor_toppled" not in archive.files or "distractor_fallen" not in archive.files:
                raise ValueError(f"Missing real event labels: {row['trajectory_id']}")
            topple = archive["distractor_toppled"]
            fall = archive["distractor_fallen"]
            onset = np.flatnonzero(np.any(~topple[:-1] & topple[1:], axis=1)) + 1
            onset = onset[(onset > t_min) & (onset <= t_max + action_horizon)]
            if len(onset):
                step = max(t_min, min(t_max, int(onset[0]) - action_horizon))
                stratum = "pre_topple"
            else:
                steps = rng.choice(np.arange(t_min, t_max + 1),
                                   size=min(16, t_max - t_min + 1), replace=False)
                risks = []
                for t in steps:
                    d = disturbance_from_poses(poses, sizes, int(t), int(t) + action_horizon,
                                               topple_flags=topple, fall_flags=fall)
                    risks.append(d["dr_mean"])
                risks = np.asarray(risks)
                if (risks >= 0.02).any():
                    step = int(steps[risks.argmax()])
                    stratum = "high_dr"
                else:
                    step = int(steps[risks.argmin()])
                    stratum = "low_dr"
        buckets[stratum].append(Probe(row["scene_id"], row["trajectory_id"],
                                      step, stratum, row))

    # Priority balances failure cases and controls, then fills empty strata.
    quotas = {"pre_topple": (count + 1) // 2,
              "high_dr": count // 4,
              "low_dr": count - (count + 1) // 2 - count // 4}
    selected = []
    scene_counts = {}
    chosen = set()

    def take(category, wanted):
        for probe in buckets[category]:
            if wanted <= 0 or len(selected) == count:
                break
            if scene_counts.get(probe.scene_id, 0) >= max_per_scene:
                continue
            if probe.trajectory_id in chosen:
                continue
            selected.append(probe)
            chosen.add(probe.trajectory_id)
            scene_counts[probe.scene_id] = scene_counts.get(probe.scene_id, 0) + 1
            wanted -= 1

    for category, quota in quotas.items():
        take(category, quota)
    # If any quota had insufficient examples, fill from any available stratum.
    for category in ("pre_topple", "high_dr", "low_dr"):
        take(category, count - len(selected))
    if len(selected) < count:
        raise ValueError(f"Requested {count} train probes, only found {len(selected)} "
                         f"under max_per_scene={max_per_scene}; reduce --states or increase the cap")
    return selected


def make_candidates(expert_actions, low, high, *, count, seed, sigma=0.02,
                    freeze_last_dimension=True):
    """Return exact expert chunk plus smooth, bounded *unlabeled* alternatives.

    All labels must be obtained by subsequently stepping the actual simulator.
    """
    actions = np.asarray(expert_actions, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32).reshape(-1)
    high = np.asarray(high, dtype=np.float32).reshape(-1)
    if actions.ndim != 2 or actions.shape[0] < 2 or actions.shape[1] != low.size:
        raise ValueError("Expected [H>=2, action_dim] expert chunk and action bounds")
    if high.shape != low.shape or np.isnan(low).any() or np.isnan(high).any() or np.any(low > high):
        raise ValueError("Invalid action bounds")
    if not np.isfinite(actions).all() or count < 2 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Require finite actions, >=2 candidates, positive sigma")
    # Expert commands are saved exactly; do not silently relabel clipped expert commands.
    if np.any(actions < low - 1e-6) or np.any(actions > high + 1e-6):
        raise ValueError("Expert actions exceed live environment bounds")
    rng = np.random.default_rng(seed)
    horizon, dim = actions.shape
    envelope = np.sin(np.pi * np.arange(1, horizon + 1) / (horizon + 1))[:, None]
    output = [actions.copy()]
    for _ in range(count - 1):
        for _attempt in range(32):
            white = rng.standard_normal((horizon, dim))
            padded = np.pad(white, ((1, 1), (0, 0)), mode="edge")
            smoothed = (padded[:-2] + 2 * padded[1:-1] + padded[2:]) / 4
            rms = np.sqrt(np.mean(smoothed ** 2, axis=0, keepdims=True))
            delta = sigma * smoothed / np.maximum(rms, 1e-12) * envelope
            if freeze_last_dimension:
                delta[:, -1] = 0
            candidate = np.clip(actions + delta, low, high).astype(np.float32)
            if np.max(np.abs(candidate - actions)) >= sigma * 0.05:
                output.append(candidate)
                break
        else:
            raise RuntimeError("Could not construct distinct bounded perturbation")
    return np.stack(output)
