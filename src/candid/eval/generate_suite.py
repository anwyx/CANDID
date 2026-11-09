"""Generate reproducible evaluation-only random scenes from a real CANDID env.

The frozen suite is shared across all candidate policies. No planner,
demonstration recording, or train/validation/test split is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
from pathlib import Path

import numpy as np

from candid.data.storage import freeze_state, save_state_npz, json_safe
from candid.eval.scene_suite import suite_fingerprint


def _as_np(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pose_signature(base, trace):
    return {
        "distractors": trace.arrays()["distractor_poses"][0].copy(),
        "target": _as_np(base.cube.pose.raw_pose).reshape(-1, 7)[0].copy(),
        "robot": _as_np(base.agent.robot.get_qpos()).reshape(1, -1)[0].copy(),
        "goal": _as_np(base.goal_site.pose.p).reshape(1, -1)[0].copy(),
    }


def _layout_key(sig, scene_spec):
    # Detect only *numerically identical* layouts, not all near-duplicates.
    arrays = np.concatenate((
        sig["distractors"].reshape(-1), sig["target"], sig["goal"],
    ))
    raw = json.dumps({
        "scene_spec": scene_spec,
        "layout": np.round(arrays.astype(float), 5).tolist(),
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def generate_suite(output_dir, env_id, env_module, *,
                   num_scenes=50, seed=900000, env_kwargs=None,
                   max_episode_steps=300, max_attempts=None):
    import gymnasium as gym
    from candid.data.monitor import PhysicsTrace

    if num_scenes < 1 or max_episode_steps < 1:
        raise ValueError("num_scenes and max_episode_steps must be positive")
    if max_attempts is None:
        max_attempts = num_scenes * 20
    if max_attempts < num_scenes:
        raise ValueError("max_attempts must be >= num_scenes")
    if not env_id or not env_module:
        raise ValueError("Both --env-id and --env-module are required")
    kwargs = dict(env_kwargs or {})
    if kwargs.get("randomize_layout") is False:
        raise ValueError("Random evaluation requires randomize_layout=True")
    kwargs["randomize_layout"] = True
    # Do not silently construct generic fallback physics; require original env.
    importlib.import_module(env_module)

    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Evaluation suite directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "scenes").mkdir(exist_ok=True)

    env = gym.make(
        env_id, obs_mode="state_dict", control_mode="pd_joint_pos",
        render_mode=None, render_backend="none",
        robot_uids="xarm6_robotiq", reward_mode="dense",
        sim_backend="cpu", max_episode_steps=max_episode_steps,
        **kwargs,
    )
    trace = PhysicsTrace(env)
    seen = set()
    selected = 0
    failures = 0
    try:
        if getattr(env.unwrapped, "num_envs", 1) != 1:
            raise ValueError("Random suite generation requires a single CPU env")
        for attempt in range(max_attempts):
            if selected == num_scenes:
                break
            scene_seed = seed + attempt
            try:
                trace.reset(seed=scene_seed)
            except (RuntimeError, ValueError) as error:
                failures += 1
                print(f"Skipped invalid candidate seed={scene_seed}: {error}", flush=True)
                continue
            base = env.unwrapped
            sig = _pose_signature(base, trace)
            spec = getattr(base, "SCENE", None)
            spec_dict = spec.to_dict() if spec is not None else None
            key = _layout_key(sig, spec_dict)
            if key in seen:
                failures += 1
                continue
            seen.add(key)
            scene_id = f"random_eval_{seed}_{selected:05d}"
            state = freeze_state(base.get_state_dict())
            rel = Path("scenes") / f"{scene_id}.npz"
            save_state_npz(root / rel, state)
            row = {
                "scene_id": scene_id,
                "seed": scene_seed,
                "env_id": env_id,
                "env_module": env_module,
                "env_kwargs": kwargs,
                "initial_state": str(rel),
                "control_mode": "pd_joint_pos",
                "obs_mode": "state_dict",
                "generated_eval_suite": True,
                "scene_spec": spec_dict,
                "layout": base.get_scene_layout_for_logging()
                    if hasattr(base, "get_scene_layout_for_logging") else None,
                "stable_initial_distractor_poses": sig["distractors"],
                "stable_initial_target_pose": sig["target"],
                "stable_initial_robot_qpos": sig["robot"],
                "stable_initial_goal_pos": sig["goal"],
                "distractor_ids": list(trace.names),
                "distractor_characteristic_sizes_m": list(trace.sizes),
            }
            with (root / "scenes.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
            selected += 1
            print(f"Saved evaluation scene {selected}/{num_scenes}: seed={scene_seed}", flush=True)
    finally:
        trace.close()
    if selected != num_scenes:
        raise RuntimeError(
            f"Only {selected}/{num_scenes} scenes generated in {max_attempts} attempts; "
            "suite.json was not written; increase --max-attempts or fix env"
        )
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[3],
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None
    manifest = {
        "schema_version": 1,
        "source": "random_eval_suite",
        "env_id": env_id,
        "env_module": env_module,
        "env_kwargs": kwargs,
        "seed": seed,
        "num_scenes": selected,
        "attempts": selected + failures,
        "max_episode_steps": max_episode_steps,
        "git_commit": sha,
        "suite_content_sha256": suite_fingerprint(root),
    }
    (root / "suite.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Frozen evaluation suite: {root} scenes={selected} "
          f"sha256={manifest['suite_content_sha256']}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate frozen random CANDID evaluation scenes")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--env-module", required=True)
    parser.add_argument("--env-kwargs", default="{}")
    parser.add_argument("--num-scenes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=900000)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=None)
    args = parser.parse_args(argv)
    kwargs = json.loads(args.env_kwargs)
    if not isinstance(kwargs, dict):
        parser.error("--env-kwargs must be a JSON object")
    return generate_suite(
        args.output_dir, args.env_id, args.env_module,
        num_scenes=args.num_scenes, seed=args.seed,
        env_kwargs=kwargs, max_episode_steps=args.max_episode_steps,
        max_attempts=args.max_attempts,
    )


if __name__ == "__main__":
    main()
