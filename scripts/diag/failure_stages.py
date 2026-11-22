"""Summarize closed-loop evaluation folders by the furthest task stage reached."""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np


def stage_of(result: dict, data) -> tuple[str, dict]:
    tcp, target = data["tcp_pose"][:, :3], data["target_pose"][:, :3]
    goal = data["goal_pos"][:, :2]
    reach = np.linalg.norm(tcp - target, axis=1)
    lift = target[:, 2] - target[0, 2]
    goal_xy = np.linalg.norm(target[:, :2] - goal, axis=1)
    info = {"min_reach": float(reach.min()), "max_lift": float(lift.max()), "min_goal_xy": float(goal_xy.min())}
    if result["task_success"]:
        return "success", info
    if reach.min() > 0.03:
        return "never_reached", info
    if lift.max() < 0.02:
        return "reached_no_lift", info
    if goal_xy.min() > 0.045:
        return "lifted_no_goal", info
    return "near_goal_not_finished", info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    print(f"{'run':38s} {'n':>3s} {'succ':>5s} {'safe':>5s} {'DRmed':>6s} {'clip%':>6s} {'reach_m':>8s} stages")
    for run in args.runs:
        stages, rows, reaches = collections.Counter(), [], []
        for result_path in sorted(glob.glob(str(run / "*" / "result.json"))):
            result = json.loads(Path(result_path).read_text(encoding="utf-8"))
            with np.load(Path(result_path).parent / "trajectory.npz") as data:
                stage, info = stage_of(result, data)
            stages[stage] += 1
            rows.append(result)
            reaches.append(info["min_reach"])
        if not rows:
            print(f"{run.name:38s} no episodes")
            continue
        clipped = sum(r["clipped_actions"] for r in rows) / max(1, sum(r["num_actions"] for r in rows))
        print(f"{run.name:38s} {len(rows):3d} {sum(r['task_success'] for r in rows):5d} "
              f"{sum(r['safe'] for r in rows):5d} {np.median([r['dr_mean'] for r in rows]):6.3f} "
              f"{100 * clipped:6.1f} {np.median(reaches):8.3f} {dict(stages)}")


if __name__ == "__main__":
    main()
