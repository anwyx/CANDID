"""Canonical CANDID data collector.

Usage:
  python scripts/collect.py --scenes 50 --trajectories-per-scene 24 \\
      --output-dir demos/team_data

The production strategy is automatic. No trajectory mode selection is needed.
"""
from __future__ import annotations
import argparse
import csv
import importlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from candid.data.monitor import PhysicsTrace, as_numpy
from candid.data.storage import DatasetWriter, freeze_state
from candid.data.validate import validate_dataset
from candid.runner.collect import _scene_start, _verify_start, _status, git_sha
from candid.runner.production_sampler import (
    ProductionSampler, DEFAULT_TRAJECTORIES_PER_SCENE, WARMUP_COUNT,
)


def build_parser():
    p = argparse.ArgumentParser(
        description="Collect CANDID trajectories with feedback sampling by default")
    p.add_argument("--scenes", type=int, default=1,
                   help="Number of random scenes (default: 1)")
    p.add_argument("--trajectories-per-scene", type=int,
                   default=DEFAULT_TRAJECTORIES_PER_SCENE,
                   help="Attempts per scene (default: 24; minimum: 14)")
    p.add_argument("--seed", type=int, default=42,
                   help="First scene seed (default: 42)")
    p.add_argument("--num-distractors", type=int, default=None,
                   help="Number of distractors from the scene spec (default: all)")
    p.add_argument("--output-dir", type=Path,
                   help="Output directory (default: new timestamped demos folder)")
    p.add_argument("--save-video", action="store_true",
                   help="Save MP4s; disabled by default for large collections")
    # Advanced team-owned options. Not required for routine collection.
    advanced = p.add_argument_group("advanced options (normally leave unchanged)")
    advanced.add_argument("--variant-seed", type=int, default=2026)
    advanced.add_argument("--variant-scale", type=float, default=1.0)
    advanced.add_argument("--max-episode-steps", type=int, default=700)
    advanced.add_argument("--env-id", default="CANDIDBaseline-v1")
    advanced.add_argument("--env-module", default="envs.candid_baseline_env")
    advanced.add_argument("--env-kwargs", default="{}")
    advanced.add_argument("--verbose", action="store_true")
    return p


def parse_args(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    if args.scenes < 1:
        p.error("--scenes must be >= 1")
    if args.trajectories_per_scene <= WARMUP_COUNT:
        p.error(f"--trajectories-per-scene must be >= {WARMUP_COUNT+1} for feedback sampling")
    if args.max_episode_steps < 1:
        p.error("--max-episode-steps must be positive")
    if not (0 < args.variant_scale <= 1.5):
        p.error("--variant-scale must be in (0, 1.5]")
    try:
        args.env_kwargs = json.loads(args.env_kwargs)
        if not isinstance(args.env_kwargs, dict):
            raise ValueError("--env-kwargs must be a JSON object")
    except (ValueError, TypeError) as exc:
        p.error(str(exc))
    if args.num_distractors is not None:
        if args.num_distractors < 1:
            p.error("--num-distractors must be >= 1")
        if "num_distractors" in args.env_kwargs or "distractor_seed" in args.env_kwargs:
            p.error("Do not specify distractor selection both as CLI argument and env kwargs")
        args.env_kwargs["num_distractors"] = args.num_distractors
        args.env_kwargs["distractor_seed"] = args.seed
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("demos") / f"candid_{stamp}_{uuid4().hex[:6]}"
    # Prevent accidentally appending incompatible data into a previous run.
    if (args.output_dir / "scenes.jsonl").exists() or (args.output_dir / "trajectories.jsonl").exists():
        p.error(f"Dataset already exists: {args.output_dir}; choose a new --output-dir")
    return args


def _planner_seed(scene_seed, variant_seed, index):
    return int(np.random.SeedSequence([
        int(scene_seed) & 0xffffffff,
        int(variant_seed) & 0xffffffff,
        int(index) & 0xffffffff,
        0xC0DE,
    ]).generate_state(1)[0])


def _scene_geometry(base, trace, scene_seed, variant_seed, variant_scale):
    target = as_numpy(base.cube.pose.raw_pose).reshape(-1, 7)[0].copy()
    return {
        "scene_seed": scene_seed,
        "variant_seed": variant_seed,
        "scale": variant_scale,
        "target_xyz": target[:3],
        "distractor_poses": trace.arrays()["distractor_poses"][0].copy(),
        "distractor_ids": list(trace.names),
        "distractor_dimensions": [m.dimensions for m in base.distractor_metadata],
        "initial_tcp_xyz": as_numpy(base.agent.tcp.pose.p).reshape(-1, 3)[0].copy(),
    }


def _write_scene_summary(output_dir, entries):
    if not entries:
        return
    fields = ["seed", "scene_id", "attempts", "successes", "baseline_dr", "min_success_dr",
              "max_success_dr", "meaningful_low", "meaningful_high", "bidirectional",
              "feedback_attempts", "uniform_attempts", "fallback_attempts"]
    with (output_dir / "collection_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(entries)


def _summarize(scene_id, seed, records):
    baseline_succeeded = bool(records[0]["success"])
    base = records[0]["dr_mean"] if baseline_succeeded else None
    successful = [r["dr_mean"] for r in records if r["success"]]
    if baseline_succeeded:
        repeats = [abs(r["dr_mean"]-base) for r in records[1:3] if r["success"]]
        threshold = max(.001, .1*base, 3*max(repeats, default=0))
        lower = sum(dr < base-threshold for dr in successful)
        higher = sum(dr > base+threshold for dr in successful)
    else:
        # Without a successful baseline, relative DR coverage is undefined.
        lower = higher = 0
    return {
        "seed": seed, "scene_id": scene_id, "attempts": len(records),
        "successes": sum(bool(r["success"]) for r in records),
        "baseline_dr": base if base is not None else "",
        "min_success_dr": min(successful) if successful else "",
        "max_success_dr": max(successful) if successful else "",
        "meaningful_low": lower, "meaningful_high": higher,
        "bidirectional": int(lower > 0 and higher > 0),
        "feedback_attempts": sum(r["selection"] == "feedback" for r in records),
        "uniform_attempts": sum(r["selection"] == "uniform" for r in records),
        "fallback_attempts": sum(r["selection"] == "heuristic_no_successful_baseline" for r in records),
    }


def main(argv=None):
    args = parse_args(argv)
    import gymnasium as gym
    from mani_skill.utils.wrappers.record import RecordEpisode
    from planners.execute_baseline import solve, BaselinePlanConfig
    # A missing Round-3 planner must fail loudly before collecting any data.
    config_fields = BaselinePlanConfig.__dataclass_fields__
    required = ("strategy", "early_offset_xyz", "lift_offset_xyz", "transfer_offset_xyz")
    missing = [key for key in required if key not in config_fields]
    if missing:
        raise RuntimeError(f"Planner missing XYZ variant fields {missing}. Apply Round-3 planner first.")
    importlib.import_module(args.env_module)
    env = gym.make(args.env_id, obs_mode="state_dict", control_mode="pd_joint_pos",
                   render_mode="rgb_array", robot_uids="xarm6_robotiq", reward_mode="dense",
                   sim_backend="cpu", max_episode_steps=args.max_episode_steps, **args.env_kwargs)
    if getattr(env.unwrapped, "num_envs", 1) != 1:
        env.close()
        raise ValueError("Production collector currently requires num_envs=1")
    active = [obj.name for obj in env.unwrapped.SCENE.distractors]
    print(f"Selected distractors ({len(active)}): {', '.join(active)}", flush=True)
    trace = PhysicsTrace(env)
    writer = DatasetWriter(args.output_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    recorder = RecordEpisode(trace, output_dir=str(args.output_dir),
                             trajectory_name=f"trajectories_{run_id}",
                             save_trajectory=True, save_video=args.save_video,
                             save_on_reset=False, clean_on_close=True,
                             record_env_state=False, source_type="motionplanning",
                             source_desc="CANDID production feedback sampling")
    scene_summaries = []
    try:
        for si in range(args.scenes):
            seed = args.seed + si
            scene_id = f"{run_id}_scene_{si:05d}"
            recorder.reset(seed=seed)
            base = recorder.unwrapped
            snapshot = freeze_state(base.get_state_dict())
            stable_start = _scene_start(base, trace)
            stable = trace.arrays()
            initial_linear = stable["distractor_linear_velocities"][0].copy()
            initial_angular = stable["distractor_angular_velocities"][0].copy()
            initial_qvel = as_numpy(base.agent.robot.get_qvel())[0].copy()
            sampler_args = _scene_geometry(base, trace, seed, args.variant_seed, args.variant_scale)
            sampler = ProductionSampler(**sampler_args)
            spec = getattr(base, "SCENE", None)
            writer.save_scene(scene_id, snapshot, {
                "run_id": run_id, "seed": seed, "env_id": args.env_id,
                "env_module": args.env_module, "env_kwargs": args.env_kwargs,
                "control_mode": "pd_joint_pos", "obs_mode": "state_dict",
                "control_dt_s": float(base.control_timestep), "git_commit": git_sha(),
                "scene_spec": spec.to_dict() if spec is not None else None,
                "layout": base.get_scene_layout_for_logging() if hasattr(base, "get_scene_layout_for_logging") else None,
                "stable_initial_distractor_poses": stable_start["distractor_poses"],
                "stable_initial_target_pose": stable_start["target_pose"],
                "stable_initial_robot_qpos": stable_start["robot_qpos"],
                "stable_initial_distractor_linear_velocities": initial_linear,
                "stable_initial_distractor_angular_velocities": initial_angular,
                "distractor_ids": list(trace.names),
                "distractor_characteristic_sizes_m": list(trace.sizes),
                "distractor_dimensions_m": sampler_args["distractor_dimensions"],
                "distractor_asset_ids": [m.spec.asset_id or m.spec.kind for m in base.distractor_metadata],
                "initial_tcp_xyz": sampler_args["initial_tcp_xyz"],
                "sampling": {"algorithm": "feedback_v1", "warmup_count": WARMUP_COUNT,
                             "uniform_every": sampler.uniform_every,
                             "variant_seed": args.variant_seed, "variant_scale": args.variant_scale,
                             "trajectories_per_scene": args.trajectories_per_scene,
                             "shared_scene_snapshot": True},
            })
            records = []
            for index in range(args.trajectories_per_scene):
                if index > 0:
                    recorder.reset(seed=seed, options={"reset_to_env_states": {"env_states": snapshot}})
                    recorder.last_reset_kwargs = {"seed": seed, "options": {"restored_scene_id": scene_id}}
                    _verify_start(stable_start, _scene_start(base, trace))
                    fresh = trace.arrays()
                    if not (np.allclose(fresh["distractor_linear_velocities"][0], initial_linear,
                                            atol=1e-5, rtol=0) and
                            np.allclose(fresh["distractor_angular_velocities"][0], initial_angular,
                                            atol=1e-5, rtol=0) and
                            np.allclose(as_numpy(base.agent.robot.get_qvel())[0],
                                        initial_qvel, atol=1e-5, rtol=0)):
                        raise RuntimeError("Scene snapshot restore velocity mismatch")
                proposal = sampler.propose()
                planner_seed = _planner_seed(seed, args.variant_seed, index)
                random.seed(planner_seed)
                np.random.seed(planner_seed)
                try:
                    result = solve(recorder, seed=seed,
                                   config=BaselinePlanConfig(**proposal.config),
                                   reset_env=False, vis=False, verbose=args.verbose)
                    status = _status(result, trace.steps)
                except Exception as exc:
                    status = {"task_success": False, "stage": "exception",
                              "planner_steps": trace.steps,
                              "message": f"{type(exc).__name__}: {exc}"}
                    print(f"WARNING: seed={seed} trial={index}: {status['message']}", flush=True)
                recorder.flush_trajectory(save=True)
                if args.save_video:
                    recorder.flush_video(save=True)
                saved = trace.steps > 0
                hfile = Path(recorder._h5_file.filename).name if saved else None
                hkey = f"traj_{recorder._episode_id}" if saved else None
                result_dr = float(trace.summary()["dr_mean"])
                trajectory_id = f"{scene_id}_traj_{index:03d}"
                writer.save_trajectory(trajectory_id, scene_id, trace, {
                    "seed": seed, "trajectory_index": index,
                    "planner": "planners.execute_baseline:solve",
                    "planner_config": proposal.config,
                    "proposal": proposal.metadata,
                    "sampling_selection": proposal.metadata["selection"],
                    "planner_seed": planner_seed,
                    "control_mode": "pd_joint_pos", "obs_mode": "state_dict",
                    "control_dt_s": float(base.control_timestep),
                    "hdf5_file": hfile, "hdf5_trajectory_key": hkey,
                    "saved_hdf5": saved, **status,
                })
                sampler.observe(proposal.config, status["task_success"], result_dr)
                records.append({"success": status["task_success"], "dr_mean": result_dr,
                                "selection": proposal.metadata["selection"]})
                print(f"seed={seed} trajectory={index+1}/{args.trajectories_per_scene} "
                      f"selection={proposal.metadata['selection']} "
                      f"success={status['task_success']} steps={trace.steps} DR={result_dr:.6f}",
                      flush=True)
            summary = _summarize(scene_id, seed, records)
            scene_summaries.append(summary)
            _write_scene_summary(args.output_dir, scene_summaries)
            print(f"SCENE COMPLETE seed={seed} success={summary['successes']}/{summary['attempts']} "
                  f"low={summary['meaningful_low']} high={summary['meaningful_high']} "
                  f"bidirectional={bool(summary['bidirectional'])}", flush=True)
    finally:
        recorder.close()
    check = validate_dataset(args.output_dir)
    print(f"DATASET VALIDATED: {check}")
    print(f"Data: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
