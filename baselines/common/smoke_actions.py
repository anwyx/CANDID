"""Fast CPU checkpoint -> action test without another physics rollout.

This is intentionally NOT a task-success evaluation. It confirms that a
trained ACT/BC/PPO/SAC checkpoint loads and emits finite pd_joint_pos actions
when given one observation from the ring demonstration dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from baselines.common.eval_baseline import load_baseline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["act", "bc", "ppo", "sac"])
    parser.add_argument("--demo-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    with h5py.File(args.demo_path, "r") as f:
        obs = np.asarray(f["traj_0/obs"][0], dtype=np.float32)
    policy = load_baseline(args.method, args.checkpoint, device="cpu")
    if obs.shape != (policy.obs_dim,):
        raise ValueError(f"Demo obs_dim={obs.shape}, {args.method} expects {policy.obs_dim}")
    if policy.action_dim != 7:
        raise ValueError(f"Expected 7 pd_joint_pos commands, got {policy.action_dim}")
    policy.begin_episode(42)
    actions = policy.act(obs[None, :])
    if actions.shape != (policy.action_horizon, 7):
        raise RuntimeError(f"Invalid action shape: {actions.shape}")
    if not np.isfinite(actions).all():
        raise RuntimeError(f"{args.method} produced NaN/Inf actions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, actions)
    report = {
        "method": args.method,
        "checkpoint": str(args.checkpoint),
        "input": str(args.demo_path),
        "action_shape": list(actions.shape),
        "action_min": float(actions.min()),
        "action_max": float(actions.max()),
        "output": str(args.output),
        "passed": True,
        "note": "Unclipped raw policy output. Not a simulator success/safety test.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
