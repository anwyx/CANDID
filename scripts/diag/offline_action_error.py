"""Measure DP action error on recorded observations, per joint, phase and split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from candid.data.adapter import observation_vector
from candid.data.trajectories import load_trajectories
from model.vanilla_dp.checkpoint import load_policy

PHASES = ("reach", "grasp", "after")


def phase_of(step: int, close: int) -> str:
    if step < close - 5:
        return "reach"
    return "grasp" if step <= close + 15 else "after"


def windows_for(trajectory, stride: int, act_horizon: int):
    observations = observation_vector(trajectory.observations)
    actions = trajectory.actions
    changes = np.flatnonzero(np.abs(np.diff(actions[:, 6])) > 1e-3)
    close = int(changes[0]) if len(changes) else len(actions)
    for t in range(0, len(actions) - act_horizon, stride):
        yield (np.stack([observations[max(t - 1, 0)], observations[t]]),
               actions[t:t + act_horizon], phase_of(t, close))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trajectories-per-split", type=int, default=40)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = load_policy(args.checkpoint, device=device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    split = json.loads((args.dataset / "split.json").read_text(encoding="utf-8"))
    trajectories = [t for t in load_trajectories(args.dataset) if t.task_success]
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    span = policy.normalizer._span().cpu().numpy()
    print("action span (rad):", span)
    for name in ("train", "validation"):
        allowed = set(split[name])
        pool = [t for t in trajectories if t.scene_id in allowed]
        chosen = rng.choice(len(pool), min(args.trajectories_per_split, len(pool)), replace=False)
        records = [w for i in chosen for w in windows_for(pool[i], args.stride, policy.action_horizon)]
        windows = np.stack([r[0] for r in records])
        if policy.obs_normalizer is not None:
            windows = policy.obs_normalizer.apply(windows)
        obs = torch.from_numpy(windows).to(device)
        expert = np.stack([r[1] for r in records])
        phases = np.array([r[2] for r in records])
        draws = []
        for _ in range(args.samples):
            chunks = [policy.normalizer.denormalize(policy.agent.get_action(obs[i:i + 512])).cpu().numpy()
                      for i in range(0, len(obs), 512)]
            draws.append(np.concatenate(chunks))
        draws = np.stack(draws)
        error = np.abs(draws - expert[None])
        print(f"--- {name}: {len(records)} windows from {len(chosen)} trajectories")
        for phase in PHASES:
            mask = phases == phase
            if not mask.any():
                continue
            mean = error[:, mask].mean(axis=(0, 1, 2))
            p90 = np.percentile(error[:, mask], 90, axis=(0, 1, 2))
            spread = draws[:, mask].std(axis=0).mean(axis=(0, 1))
            print(f"{phase:6s} n={mask.sum():5d} mean_abs {mean}")
            print(f"       p90_abs {p90}")
            print(f"       draw_std {spread}")
        print("first-action mean_abs", error[:, :, 0].mean(axis=(0, 1)))
        print("whole-arm max_joint_err median/p90",
              np.round(np.percentile(error[:, :, 0, :6].max(-1), [50, 90]), 4))


if __name__ == "__main__":
    main()
