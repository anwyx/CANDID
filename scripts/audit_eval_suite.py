"""Audit frozen random suite geometry and actual closed-loop episode outcomes.

No ManiSkill imports; reads only the archived suite, trajectories and results.
Use this to investigate suspiciously high success rates before changing DP.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _distance_summary(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "p10": float(np.quantile(values, 0.1)),
        "p90": float(np.quantile(values, 0.9)),
    }


def audit_suite(suite_dir, result_dir=None, original_dataset=None):
    from candid.eval.scene_suite import load_suite

    suite_dir = Path(suite_dir)
    meta, scenes = load_suite(suite_dir)
    nearest_obstacle = []
    target_to_goal = []
    goal_to_obstacle = []
    initial_goal_overlap = 0
    n_obstacles = []
    seed_values = []
    for scene in scenes:
        target = np.asarray(scene["stable_initial_target_pose"], dtype=float)[:3]
        goal = np.asarray(scene["stable_initial_goal_pos"], dtype=float)[:3]
        poses = np.asarray(scene["stable_initial_distractor_poses"], dtype=float)
        nearest_obstacle.append(float(np.linalg.norm(poses[:, :2] - target[None, :2], axis=1).min()))
        goal_to_obstacle.append(float(np.linalg.norm(poses[:, :2] - goal[None, :2], axis=1).min()))
        target_to_goal.append(float(np.linalg.norm(goal[:2] - target[:2])))
        n_obstacles.append(len(poses))
        seed_values.append(int(scene["seed"]))
        spec = scene.get("scene_spec") or {}
        if np.linalg.norm(goal[:2] - target[:2]) <= spec.get("goal_xy_tolerance", 0.045):
            initial_goal_overlap += 1

    report = {
        "suite_sha256": meta["suite_content_sha256"],
        "scenes": len(scenes),
        "seed_count": len(set(seed_values)),
        "env_id": meta["env_id"],
        "env_module": meta["env_module"],
        "env_kwargs": meta["env_kwargs"],
        "num_obstacles": sorted(set(n_obstacles)),
        "nearest_target_obstacle_xy_m": _distance_summary(nearest_obstacle),
        "nearest_goal_obstacle_xy_m": _distance_summary(goal_to_obstacle),
        "target_goal_xy_m": _distance_summary(target_to_goal),
        "initial_target_within_goal_xy_tolerance": initial_goal_overlap,
    }

    if original_dataset is not None:
        original_dataset = Path(original_dataset)
        split = json.loads((original_dataset / "split.json").read_text(encoding="utf-8"))
        rows = _jsonl(original_dataset / "scenes.jsonl")
        old = [row for row in rows if row["scene_id"] in set(split["test"])]
        old_nearest = []
        old_target_goal = []
        envs = {}
        specs = {}
        for row in old:
            poses = np.asarray(row["stable_initial_distractor_poses"], dtype=float)
            target = np.asarray(row["stable_initial_target_pose"], dtype=float)[:3]
            goal = None
            layout = row.get("layout")
            if isinstance(layout, dict) and layout.get("goal_xy") is not None:
                goal = np.asarray(layout["goal_xy"], dtype=float)
            old_nearest.append(float(np.linalg.norm(poses[:, :2]-target[None, :2], axis=1).min()))
            if goal is not None:
                old_target_goal.append(float(np.linalg.norm(target[:2]-goal[:2])))
            env_key = (row["env_id"], row["env_module"], json.dumps(row.get("env_kwargs",{}),sort_keys=True))
            envs[str(env_key)] = envs.get(str(env_key), 0)+1
            spec_str = json.dumps(row.get("scene_spec"),sort_keys=True)
            specs[spec_str] = specs.get(spec_str,0)+1
        suite_spec = json.dumps(scenes[0]["scene_spec"],sort_keys=True)
        report["old_test_comparison"] = {
            "old_test_scenes": len(old),
            "old_envs": envs,
            "old_scene_spec_equals_new": len(specs)==1 and suite_spec in specs,
            "old_scene_spec_variants": len(specs),
            "nearest_target_obstacle_xy_m":_distance_summary(old_nearest),
            "target_goal_xy_m":_distance_summary(old_target_goal),
        }

    if result_dir is not None:
        result_dir = Path(result_dir)
        summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
        rows = _jsonl(result_dir / "episodes.jsonl")
        if summary.get("eval_suite_sha256") != meta["suite_content_sha256"]:
            raise ValueError("Run was NOT evaluated on this exact suite")
        scene_ids = set(scene["scene_id"] for scene in scenes)
        if any(r["scene_id"] not in scene_ids for r in rows):
            raise ValueError("Result includes episodes from another suite")
        successful = 0
        saved_steps = []
        xy_final = []
        xy_initial = []
        bad_success_xy = []
        initial_physically_placed = []
        first_state_mismatch = []
        spec_by_id = {row["scene_id"]: row["scene_spec"] for row in scenes}
        scene_by_id = {row["scene_id"]: row for row in scenes}
        for episode in rows:
            traj_path = Path(episode["trajectory_file"])
            if not traj_path.is_file():
                raise FileNotFoundError(traj_path)
            with np.load(traj_path, allow_pickle=False) as traj:
                target = traj["target_pose"]
                goals = traj["goal_pos"]
                actions = traj["actions"]
                if len(target) != len(actions)+1 or len(goals) != len(target):
                    raise ValueError(f"Trajectory steps mismatch: {episode['scene_id']}")
                initial = np.linalg.norm(target[0,:2] - goals[0,:2])
                final = np.linalg.norm(target[-1,:2] - goals[-1,:2])
                xy_final.append(float(final))
                xy_initial.append(float(initial))
                saved_steps.append(len(actions))
                tolerance = spec_by_id[episode["scene_id"]]["goal_xy_tolerance"]
                if initial <= tolerance:
                    initial_physically_placed.append(episode["scene_id"])
                stored_init = np.asarray(scene_by_id[episode["scene_id"]]["stable_initial_target_pose"])
                if not np.allclose(target[0], stored_init, atol=1e-4, rtol=0):
                    first_state_mismatch.append(episode["scene_id"])
                if episode["task_success"]:
                    successful += 1
                    if final > tolerance + 1e-4:
                        bad_success_xy.append((episode["scene_id"],float(final),tolerance))
        report["rollout"] = {
            "episodes":len(rows),
            "successful":successful,
            "success_rate":successful/len(rows),
            "summary_success_rate":summary["success_rate"],
            "num_actions":_distance_summary(saved_steps),
            "initial_target_goal_xy_m":_distance_summary(xy_initial),
            "final_target_goal_xy_m":_distance_summary(xy_final),
            "initial_target_within_goal_xy_tolerance":initial_physically_placed,
            "first_state_mismatch":first_state_mismatch,
            "success_but_outside_goal_xy":bad_success_xy,
        }
        if abs(report["rollout"]["success_rate"]-summary["success_rate"])>1e-12:
            raise ValueError("Summary success_rate disagrees with raw episode results")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--eval-suite",type=Path,required=True)
    parser.add_argument("--results",type=Path)
    parser.add_argument("--old-dataset",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    report=audit_suite(args.eval_suite,args.results,args.old_dataset)
    body=json.dumps(report,indent=2,ensure_ascii=False)
    print(body)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(body+"\n",encoding="utf-8")
    return report

if __name__=="__main__":
    main()
