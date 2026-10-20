"""Generic CANDID ManiSkill environment runner and dataset recorder.

Run from the CANDID project root, e.g.:

  # Original environment/planner (compatible with earlier run_baseline.py)
  python scripts/run_baseline.py --episodes 2 --obs-mode none --save-video

  # Any TEAM environment that registers a Gym/ManiSkill environment ID
  python scripts/run_baseline.py \
      --env-module envs.primitive_env --env-id CANDIDPrimitive-v0 \
      --episodes 10 --obs-mode state --output-dir demos/primitive_test

  # Pass environment-specific construction parameters in JSON
  python scripts/run_baseline.py \
      --env-module envs.ycb_env --env-id CANDIDYCB-v0 \
      --env-kwargs '{"randomize_layout": true, "num_distractors": 4}'

  # Optional custom planner implementing the shared solve() API
  python scripts/run_baseline.py --env-module envs.ycb_env \
      --env-id CANDIDYCB-v0 --planner planners.ycb_planner:solve \
      --planner-config-class planners.ycb_planner:PlanConfig \
      --plan-kwargs '{"lift_height": 0.15}'

CONTRACT:
  Each --env-module must register --env-id when imported. An environment
  compatible with the default xArm6 pick-and-place planner must provide
  .cube, .goal_site, .agent, .cube_half_size and evaluate()["success"].
  Different tasks/robots can supply a custom solve() with this signature:
      solve(env, seed: int, config: Any, vis: bool, verbose: bool)
  It MUST reset the environment before executing, using env.reset(seed=seed),
  and return an object or dict with success, stage, steps (optional details).

  Disturbance monitoring is independent of the planner and optional for
  environments without distractor logging APIs. Raw simulator truth never
  enters the policy's observation. One environment instance at a time.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Running from scripts/ would otherwise hide the project-root packages.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode

from metrics.disturbance_monitor import DisturbanceMonitor


def _json_dict(value: str, flag: str) -> dict:
    """Parse a small JSON object for environment/planner construction kwargs."""
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{flag}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError(f"{flag} expects a JSON object, e.g. '{{\"x\": 1}}'")
    return data


def _import_symbol(location: str):
    """Import a callable/class expressed as 'package.module:object_name'."""
    if ":" not in location:
        raise ValueError(f"Expected 'module:symbol', got {location!r}")
    module_name, name = location.split(":", 1)
    return getattr(importlib.import_module(module_name), name)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def _safe_path_component(value: str) -> str:
    """Prevent environment names containing slashes from changing paths."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _json_safe(value):
    """Convert common NumPy/tensor values to JSON-compatible Python values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        return _json_safe(value.detach().cpu().numpy())
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run any registered CANDID pick-and-place environment with a selected planner"
    )
    parser.add_argument("--env-id", default="CANDIDBaseline-v0",
                        help="Registered Gym/ManiSkill environment ID")
    parser.add_argument("--env-module", default="envs.candid_baseline_env",
                        help="Python module importing/registering --env-id (e.g. envs.ycb_env)")
    parser.add_argument("--env-kwargs", default="{}", metavar="JSON",
                        help='Environment-specific kwargs, e.g. \'{"num_distractors": 6}\'')
    parser.add_argument("--fixed-scene", action="store_true",
                        help="For environments supporting randomize_layout: pass randomize_layout=False")
    parser.add_argument("--robot-uid", default="xarm6_robotiq")
    parser.add_argument("--control-mode", default="pd_joint_pos",
                        help="Default official xArm6 MPlib planner requires pd_joint_pos")
    parser.add_argument("--obs-mode", default="rgbd",
                        help="none for speed; rgbd/pointcloud/state if supported by the env")
    parser.add_argument("--reward-mode", default="dense")
    parser.add_argument("--sim-backend", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument("--max-episode-steps", type=int, default=500)

    parser.add_argument("--planner", default="planners.execute_baseline:solve",
                        help="Planner function import path, MODULE:FUNCTION")
    parser.add_argument("--planner-config-class", default="planners.execute_baseline:BaselinePlanConfig",
                        help="Planner config class, MODULE:CLASS; use 'none' for no config")
    parser.add_argument("--plan-kwargs", default="{}", metavar="JSON",
                        help='Planner config kwargs, e.g. \'{"lift_height": 0.12}\'')
    parser.add_argument("--lateral-offset", type=float, default=None,
                        help="BaselinePlanConfig shortcut; omit for other planners")

    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0,
                        help="Each episode uses seed + episode_index")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default demos/<env-id>; put separate experiments in separate directories")
    parser.add_argument("--traj-name", default=None,
                        help="HDF5 basename (do not include extension)")
    parser.add_argument("--no-record", action="store_true",
                        help="Disable HDF5, but allow video/metrics")
    parser.add_argument("--save-success-only", action="store_true",
                        help="Keep only successful episodes in HDF5 and video")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--translation-weight", type=float, default=0.7)
    parser.add_argument("--rotation-weight", type=float, default=0.3)
    parser.add_argument("--table-height", type=float, default=0.0,
                        help="World Z of the tabletop, used by fall heuristic")

    args = parser.parse_args(argv)
    if args.episodes < 1 or args.max_episode_steps < 1:
        parser.error("--episodes and --max-episode-steps must be >= 1")
    if args.translation_weight < 0 or args.rotation_weight < 0:
        parser.error("disturbance weights must be nonnegative")
    try:
        args.env_kwargs = _json_dict(args.env_kwargs, "--env-kwargs")
        args.plan_kwargs = _json_dict(args.plan_kwargs, "--plan-kwargs")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.fixed_scene:
        if "randomize_layout" in args.env_kwargs:
            parser.error("Do not use --fixed-scene together with env-kwargs.randomize_layout")
        args.env_kwargs["randomize_layout"] = False
    if args.lateral_offset is not None:
        if "lateral_offset" in args.plan_kwargs:
            parser.error("Specify lateral_offset via only one of --lateral-offset/--plan-kwargs")
        if abs(args.lateral_offset) > 0.15:
            parser.error("--lateral-offset must be in [-0.15, 0.15] m")
        args.plan_kwargs["lateral_offset"] = args.lateral_offset
    if args.output_dir is None:
        args.output_dir = Path("demos") / _safe_path_component(args.env_id)
    return args


def load_planner(args):
    """Load a planner and its config without hardcoded class imports."""
    solve = _import_symbol(args.planner)
    if not callable(solve):
        raise TypeError(f"Planner {args.planner} is not callable")
    if args.planner_config_class.lower() == "none":
        if args.plan_kwargs:
            raise ValueError("--plan-kwargs requires a --planner-config-class")
        config = None
    else:
        config_cls = _import_symbol(args.planner_config_class)
        config = config_cls(**args.plan_kwargs)
    return solve, config


def build_env(args):
    """Register custom environment module, wrap it for metrics and recording."""
    importlib.import_module(args.env_module)
    # General Gym registration can also be done by external packages. ManiSkill
    # integrates into gym.make when custom env module import registers its ID.
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode="rgb_array",
        robot_uids=args.robot_uid,
        reward_mode=args.reward_mode,
        sim_backend=args.sim_backend,
        max_episode_steps=args.max_episode_steps,
        **args.env_kwargs,
    )
    if getattr(env.unwrapped, "num_envs", 1) != 1:
        env.close()
        raise ValueError("This MPlib runner and DisturbanceMonitor require num_envs=1")

    monitor = DisturbanceMonitor(
        env,
        translation_weight=args.translation_weight,
        rotation_weight=args.rotation_weight,
        table_height_m=args.table_height,
    )
    recorder = None
    if not args.no_record or args.save_video:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        name = args.traj_name or datetime.now().strftime("trajectories_%Y%m%d_%H%M%S_%f")
        recorder = RecordEpisode(
            monitor,
            output_dir=str(args.output_dir),
            trajectory_name=name,
            save_trajectory=not args.no_record,
            save_video=args.save_video,
            save_on_reset=False,
            clean_on_close=True,
            record_env_state=False,  # NEVER inject simulator truth into DP data
            source_type="motionplanning",
            source_desc=f"CANDID {args.env_id} / {args.planner}",
            video_fps=30,
        )
    return recorder or monitor, monitor, recorder


def _summarize_planner_result(result: Any, fallback_steps: int) -> dict[str, Any]:
    """Normalize return objects from custom planners to standard status keys."""
    def field(k, default=None):
        return result.get(k, default) if isinstance(result, dict) else getattr(result, k, default)
    if field("success") is None:
        raise ValueError("Planner result must have a boolean 'success' field")
    return {
        "success": bool(field("success")),
        "stage": str(field("stage", "completed")),
        "steps": int(field("steps", fallback_steps)),
        "grasped_after_lift": field("grasped_after_lift"),
        "message": str(field("message", "")),
    }


def _read_scene_layout(base_env):
    getter = getattr(base_env, "get_scene_layout_for_logging", None)
    return _json_safe(getter()) if callable(getter) else None


def main(argv=None):
    args = parse_args(argv)
    solve, config = load_planner(args)
    env, monitor, recorder = build_env(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    metadata_path = args.output_dir / "episode_metrics.jsonl"
    commit = _git_commit()
    successes = saved = 0
    warned_no_poses = warned_no_sizes = False

    try:
        with metadata_path.open("a", encoding="utf-8") as output:
            for k in range(args.episodes):
                seed = args.seed + k
                print(f"\n--- {args.env_id}: {k + 1}/{args.episodes}, seed={seed} ---")
                resets_before = monitor.reset_count
                try:
                    raw_result = solve(env, seed=seed, config=config,
                                       vis=args.vis, verbose=args.verbose)
                    status = _summarize_planner_result(raw_result, monitor.steps)
                except Exception as exc:
                    # Capture the partial episode, if the planner already reset.
                    status = {
                        "success": False,
                        "stage": "exception",
                        "steps": monitor.steps if monitor.reset_count > resets_before else 0,
                        "grasped_after_lift": None,
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"Planner exception: {status['message']}")
                episode_reset = monitor.reset_count > resets_before
                if not episode_reset:
                    # Prevent leftover labels from the previous attempt from
                    # being incorrectly associated with this failed attempt.
                    print("WARNING: planner did not call env.reset(); no current episode was recorded")
                    labels = {"has_data": False, "steps": 0, "continuous_dr": None, "d_max": None}
                else:
                    labels = monitor.summary()
                if episode_reset and not monitor.available and not warned_no_poses:
                    print("WARNING: this environment exposes no distractor pose logger; "
                          "trajectory collection works but disturbance labels are unavailable")
                    warned_no_poses = True
                if episode_reset and monitor.available and monitor.object_sizes_m is None and not warned_no_sizes:
                    print("WARNING: no per-object characteristic sizes found; raw disturbance "
                          "is saved, but continuous DR will be N/A. Implement "
                          "get_distractor_sizes_for_logging() in your environment")
                    warned_no_sizes = True
                if status["success"]:
                    successes += 1

                # Save raw per-action distractor truth independently of
                # policy observations. Unique names for each run and attempt.
                trace_path = args.output_dir / "label_traces" / f"{run_id}_attempt_{k:05d}.npz"
                trace_written = bool(episode_reset and monitor.save_pose_trace(trace_path))
                should_save = not args.save_success_only or status["success"]
                nonempty = episode_reset and monitor.steps > 0
                recorded_key = None
                recorded_path = None
                if recorder is not None and episode_reset:
                    if not args.no_record:
                        recorder.flush_trajectory(save=should_save)
                        if should_save and nonempty:
                            saved += 1
                            recorded_key = f"traj_{recorder._episode_id}"
                            recorded_path = Path(recorder._h5_file.filename).name
                    if args.save_video:
                        recorder.flush_video(save=should_save)

                row = {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "run_id": run_id,
                    "attempt_index": k,
                    "seed": seed,
                    "env_id": args.env_id,
                    "env_module": args.env_module,
                    "env_kwargs": args.env_kwargs,
                    "planner": args.planner,
                    "planner_config_class": args.planner_config_class,
                    "plan_kwargs": args.plan_kwargs,
                    "git_commit": commit,
                    "scene_layout": _read_scene_layout(env.unwrapped) if episode_reset else None,
                    "control_mode": args.control_mode,
                    "obs_mode": args.obs_mode,
                    "task_success": status["success"],
                    "last_stage": status["stage"],
                    "planner_steps": status["steps"],
                    "grasped_after_lift": status["grasped_after_lift"],
                    "message": status["message"],
                    "saved_to_hdf5": bool(recorded_key),
                    "hdf5_file": recorded_path,
                    "hdf5_trajectory_key": recorded_key,
                    # Paths are relative to output_dir so moving a completed
                    # Run to shared storage does not break metadata references.
                    "label_trace": str(trace_path.relative_to(args.output_dir)) if trace_written else None,
                    "disturbance": labels,
                }
                output.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")
                output.flush()
                dr = labels.get("continuous_dr")
                dr_text = "N/A" if dr is None else f"{dr:.4f}"
                peak = labels.get("max_peak_translation_m")
                peak_text = "N/A" if peak is None else f"{peak:.4f} m"
                print(
                    f"  success={status['success']} stage={status['stage']} "
                    f"steps={status['steps']} peak_translation={peak_text} "
                    f"continuous_dr={dr_text}"
                )
    finally:
        env.close()

    print("\n=== CANDID run summary ===")
    print(f"Environment: {args.env_id}")
    print(f"Attempts: {args.episodes}")
    print(f"Task successes: {successes} ({successes / args.episodes:.1%})")
    print(f"Saved HDF5 trajectories: {saved}")
    print(f"Episode metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
