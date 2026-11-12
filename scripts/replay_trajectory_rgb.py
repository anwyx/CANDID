#!/usr/bin/env python3
"""Pack one CANDID eval episode and render genuine ManiSkill RGB from recorded states.

No DP/Predictor inference and no physics stepping during rendering.
Rendering requires SAPIEN/Vulkan and matching ManiSkill/robot/scene assets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import warnings
import zipfile

import numpy as np

REQUIRED = ("robot_qpos", "robot_qvel", "target_pose", "distractor_poses", "goal_pos")
BUNDLE_NAMES = {"trajectory.npz", "initial_state.npz", "scene.json", "result.json", "metadata.json"}


def get_scene(suite: Path, scene_id: str):
    meta = json.loads((suite / "suite.json").read_text(encoding="utf-8"))
    if meta.get("source") != "multi_env_eval_suite":
        raise ValueError("Expected frozen multi-environment eval suite")
    rows = [
        json.loads(line) for line in (suite / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [row for row in rows if row["scene_id"] == scene_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one scene_id={scene_id!r}, got {len(matches)}")
    return matches[0]


def pack(args):
    trajectory = args.trajectory.resolve()
    result_file = trajectory.parent / "result.json"
    if not result_file.is_file():
        raise FileNotFoundError(f"Expected episode result.json: {result_file}")
    result = json.loads(result_file.read_text(encoding="utf-8"))
    scene = get_scene(args.eval_suite, result["scene_id"])
    snapshot = args.eval_suite / scene["initial_state"]
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)
    with np.load(trajectory, allow_pickle=False) as z:
        missing = set(REQUIRED) - set(z.files)
        if missing:
            raise ValueError(f"Missing recorded fields: {sorted(missing)}")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Will not overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        z.write(trajectory, "trajectory.npz")
        z.write(snapshot, "initial_state.npz")
        z.writestr("scene.json", json.dumps(scene, indent=2, ensure_ascii=False))
        z.writestr("result.json", json.dumps(result, indent=2, ensure_ascii=False))
        z.writestr("metadata.json", json.dumps({
            "format": "candid_recorded_rgb_replay_v1",
            "source_eval_suite": str(args.eval_suite),
            "scene_id": result["scene_id"],
            "environment_module": scene["env_module"],
        }, indent=2))
    print(f"Packed {output} ({output.stat().st_size / 1048576:.1f} MiB)")


def load_track(path):
    with np.load(path, allow_pickle=False) as z:
        missing = set(REQUIRED) - set(z.files)
        if missing:
            raise ValueError(f"Missing fields {sorted(missing)}")
        data = {key: np.asarray(z[key]) for key in REQUIRED}
    n = len(data["robot_qpos"])
    if n < 2 or data["robot_qpos"].ndim != 2:
        raise ValueError("Expected robot_qpos [T,D] with T>=2")
    if data["robot_qvel"].shape != data["robot_qpos"].shape:
        raise ValueError("robot_qvel shape mismatch")
    if data["target_pose"].shape != (n, 7) or data["goal_pos"].shape != (n, 3):
        raise ValueError("target_pose [T,7] / goal_pos [T,3] shape mismatch")
    d = data["distractor_poses"]
    if d.ndim != 3 or d.shape[0] != n or d.shape[2] != 7:
        raise ValueError("distractor_poses must be [T,N,7]")
    if any(not np.isfinite(a).all() for a in data.values()):
        raise ValueError("Non-finite values in recorded trajectory")
    return data


def as_numpy(value):
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)


def check_initial(base, scene, track, atol=2e-4):
    from candid.data.storage import json_safe
    if json_safe(base.SCENE.to_dict()) != scene["scene_spec"]:
        raise ValueError("SceneSpec differs from saved evaluation scene")
    if len(base.distractor_actors) != track["distractor_poses"].shape[1]:
        raise ValueError("Distractor count does not match recorded trajectory")
    names = [m.spec.name for m in base.distractor_metadata]
    if names != scene["distractor_ids"]:
        raise ValueError(f"Distractor order mismatch: {names} != {scene['distractor_ids']}")
    compare = {
        "robot_qpos": (as_numpy(base.agent.robot.get_qpos()).reshape(-1), track["robot_qpos"][0]),
        "target_pose": (as_numpy(base.target_actor.pose.raw_pose).reshape(-1), track["target_pose"][0]),
        "goal_pos": (as_numpy(base.goal_site.pose.p).reshape(-1), track["goal_pos"][0]),
    }
    for idx, actor in enumerate(base.distractor_actors):
        compare[f"distractor_{idx}"] = (
            as_numpy(actor.pose.raw_pose).reshape(-1), track["distractor_poses"][0, idx]
        )
    for name, (live, recorded) in compare.items():
        if live.shape != recorded.shape:
            raise ValueError(f"{name}: shape differs {live.shape} != {recorded.shape}")
        ref = recorded.copy()
        if len(ref) == 7 and np.dot(live[3:], ref[3:]) < 0:
            ref[3:] *= -1  # equivalent quaternion orientation
        err = float(np.max(np.abs(live - ref)))
        if err > atol:
            raise ValueError(f"{name}: initial pose mismatch {err:.5g}; check original scene/assets")


def rgb_numpy(frame):
    frame = as_numpy(frame)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"Unexpected RGB image shape: {frame.shape}")
    frame = frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1.01:
        frame = frame * 255
    return np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))


def set_actor_pose(actor, pq):
    from mani_skill.utils.structs.pose import Pose
    actor.set_pose(Pose.create(np.asarray(pq, dtype=np.float32)))


def render(args):
    import gymnasium as gym
    import imageio.v2 as imageio
    import torch
    from candid.data.storage import load_state_npz
    from candid.eval.replay_env import resolve_scene_environment

    with tempfile.TemporaryDirectory(prefix="candid-render-") as temp:
        temp = Path(temp)
        with zipfile.ZipFile(args.bundle) as bundle:
            files = set(bundle.namelist())
            required = {"trajectory.npz", "initial_state.npz", "scene.json", "result.json"}
            if not required.issubset(files) or not files.issubset(BUNDLE_NAMES):
                raise ValueError(f"Invalid episode render bundle: {sorted(files)}")
            for member in files:
                (temp / member).write_bytes(bundle.read(member))

        scene = json.loads((temp / "scene.json").read_text(encoding="utf-8"))
        result = json.loads((temp / "result.json").read_text(encoding="utf-8"))
        if scene["scene_id"] != result["scene_id"]:
            raise ValueError("Mismatched scene_id in scene.json and result.json")
        track = load_track(temp / "trajectory.npz")
        env_id, source = resolve_scene_environment(scene)
        if source != "original_env_module":
            if not args.allow_scene_spec_fallback:
                raise RuntimeError(
                    "Original scene module is missing. Install it, or for a standard "
                    "SceneSpec-only environment retry with --allow-scene-spec-fallback."
                )
            warnings.warn("Using SceneSpec fallback; exact custom environment behavior is not guaranteed")

        base_kwargs = dict(
            obs_mode="state_dict",
            control_mode="pd_joint_pos",
            robot_uids="xarm6_robotiq",
            reward_mode="dense",
            sim_backend="cpu",
            render_backend=args.render_backend,
            render_mode="rgb_array",
            max_episode_steps=max(args.max_steps, len(track["robot_qpos"])),
            human_render_camera_configs={"width": args.width, "height": args.height},
        )
        if scene.get("generated_eval_suite"):
            base_kwargs["skip_constructor_settling"] = True
        env_kwargs = dict(scene.get("env_kwargs") or {})
        overlaps = set(base_kwargs) & set(env_kwargs)
        if overlaps:
            raise ValueError(f"Conflicting scene kwargs: {sorted(overlaps)}")

        env = gym.make(env_id, **base_kwargs, **env_kwargs)
        try:
            state = load_state_npz(temp / "initial_state.npz", device="cpu")
            env.reset(seed=int(scene["seed"]), options={"reset_to_env_states": {"env_states": state}})
            base = env.unwrapped
            check_initial(base, scene, track)
            robot = base.agent.robot
            if as_numpy(robot.get_qpos()).shape[-1] != track["robot_qpos"].shape[-1]:
                raise ValueError("Robot joint dimension mismatch")
            goal_quat = as_numpy(base.goal_site.pose.raw_pose).reshape(-1)[3:]
            indices = list(range(0, len(track["robot_qpos"]), args.stride))
            if indices[-1] != len(track["robot_qpos"]) - 1:
                indices.append(len(track["robot_qpos"]) - 1)
            if args.max_frames:
                indices = indices[:args.max_frames]
            output = args.output.resolve()
            if output.exists():
                raise FileExistsError(f"Will not overwrite {output}")
            output.parent.mkdir(parents=True, exist_ok=True)

            print(f"Rendering {len(indices)} frames of original ManiSkill scene {scene['scene_id']}", flush=True)
            with imageio.get_writer(
                output, format="FFMPEG", fps=args.fps, codec="libx264",
                pixelformat="yuv420p", macro_block_size=None, quality=8
            ) as writer:
                for j, t in enumerate(indices):
                    q = torch.as_tensor(track["robot_qpos"][t], dtype=torch.float32, device=base.device)[None]
                    v = torch.as_tensor(track["robot_qvel"][t], dtype=torch.float32, device=base.device)[None]
                    robot.set_qpos(q)
                    robot.set_qvel(v)
                    set_actor_pose(base.target_actor, track["target_pose"][t])
                    for actor, pose in zip(base.distractor_actors, track["distractor_poses"][t]):
                        set_actor_pose(actor, pose)
                    set_actor_pose(base.goal_site, np.r_[track["goal_pos"][t], goal_quat])
                    # Crucial: NO env.step(). Only archived poses are displayed.
                    writer.append_data(rgb_numpy(base.render_rgb_array()))
                    if (j + 1) % 25 == 0 or j + 1 == len(indices):
                        print(f"  {j+1}/{len(indices)} frames", flush=True)
            print(f"Saved real RGB video: {output}")
        finally:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack", help="Headless server: pack one saved episode")
    p.add_argument("--trajectory", required=True, type=Path)
    p.add_argument("--eval-suite", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.set_defaults(func=pack)
    p = sub.add_parser("render", help="Rendering machine: genuine SAPIEN/MainiSkill video")
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--render-backend", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--allow-scene-spec-fallback", action="store_true")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=300)
    p.set_defaults(func=render)
    args = parser.parse_args(argv)
    if args.command == "render":
        if (args.fps < 1 or args.stride < 1 or args.width < 2 or args.height < 2
                or args.width % 2 or args.height % 2 or args.max_frames < 0):
            parser.error("Require fps>=1, stride>=1, even width/height, max_frames>=0")
    args.func(args)


if __name__ == "__main__":
    main()
