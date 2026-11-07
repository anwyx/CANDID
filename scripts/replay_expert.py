"""Replay recorded expert actions open-loop through the evaluation rollout."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from candid.data.adapter import observation_vector
from candid.data.trajectories import load_trajectories
from candid.eval.metrics import SafetyLimits
from candid.eval.rollout import _rollout_single


class ReplayPolicy:
    """Return one recorded action per call, then hold the last one."""

    obs_horizon = 2
    action_horizon = 1

    def __init__(self, actions: np.ndarray, obs_dim: int):
        self.actions = actions
        self.obs_dim = obs_dim
        self.action_dim = actions.shape[1]
        self.index = 0

    def act(self, window):
        action = self.actions[min(self.index, len(self.actions) - 1)]
        self.index += 1
        return action[None]


def select_ids(rows, scene_ids):
    wanted = {f"_traj_{i:03d}" for i in range(4)}
    return [row["trajectory_id"] for row in rows
            if row["scene_id"] in scene_ids and row["trajectory_id"][-9:] in wanted]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=6)
    parser.add_argument("--hold-steps", type=int, default=30)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    split = json.loads((args.dataset / "split.json").read_text(encoding="utf-8"))
    read = lambda name: [json.loads(line) for line in (args.dataset / name).open(encoding="utf-8")]
    scenes = {row["scene_id"]: row for row in read("scenes.jsonl")}
    rows = {row["trajectory_id"]: row for row in read("trajectories.jsonl")}
    groups = {scene: "test" for scene in split["test"]}
    groups.update({scene: "train" for scene in split["train"][:args.train_scenes]})
    ids = select_ids(rows.values(), set(groups))[args.shard::args.num_shards]
    loaded = {t.trajectory_id: t for t in load_trajectories(args.dataset) if t.trajectory_id in set(ids)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"results_{args.shard:02d}.jsonl"
    limits = SafetyLimits(0.2)
    for trajectory_id in ids:
        row, expert = rows[trajectory_id], loaded[trajectory_id]
        recorded = observation_vector(expert.observations)
        policy = ReplayPolicy(expert.actions, recorded.shape[-1])
        episode = args.output_dir / "episodes" / trajectory_id.replace("/", "_")
        result = _rollout_single(
            scenes[row["scene_id"]], args.dataset, policy, episode, limits,
            max_steps=len(expert.actions) + args.hold_steps, exec_horizon=1,
            eval_seed=int(row["seed"]), save_video=False, stop_on_success=True,
        )
        with np.load(episode / "trajectory.npz") as data:
            n = min(len(data["target_pose"]), len(expert.observations["target_state"]))
            qpos = np.abs(data["robot_qpos"][:n] - expert.observations["robot_qpos"][:n]).max(1)
            target = np.linalg.norm(data["target_pose"][:n, :3]
                                    - expert.observations["target_state"][:n, :3], axis=1)
        print_row = {
            "trajectory_id": trajectory_id, "group": groups[row["scene_id"]],
            "route": row["planner_config"].get("route_intent"),
            "recorded_success": bool(row["task_success"]),
            "replay_success": bool(result["task_success"]),
            "replay_success_once": bool(result["success_once"]),
            "recorded_steps": len(expert.actions), "replay_steps": int(result["num_actions"]),
            "clipped_actions": int(result["clipped_actions"]),
            "recorded_dr_mean": float(row["disturbance"]["dr_mean"]),
            "replay_dr_mean": float(result["dr_mean"]),
            "recorded_topple": int(row["disturbance"].get("topple_count", 0)),
            "replay_topple": int(result["topple_count"]),
            "qpos_err_step10": float(qpos[min(10, n - 1)]), "qpos_err_max": float(qpos.max()),
            "target_err_step50": float(target[min(50, n - 1)]), "target_err_max": float(target.max()),
        }
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(print_row) + "\n")
        print(json.dumps(print_row), flush=True)


if __name__ == "__main__":
    main()
