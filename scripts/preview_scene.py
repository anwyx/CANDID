"""Save a still image of a new scene or every scene in a saved dataset."""

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import gymnasium as gym
import numpy as np
from PIL import Image


def save_frame(env, path):
    frame = env.render()
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(f"Expected one RGB image, got shape {frame.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame.astype(np.uint8)).save(path)
    print(f"Saved scene preview: {path.resolve()}")


def preview_saved_scenes(root, output_dir):
    from candid.data.storage import json_safe, load_state_npz

    root = root.resolve()
    with (root / "scenes.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"No scenes in {root / 'scenes.jsonl'}")

    first = rows[0]
    importlib.import_module(first["env_module"])
    env = gym.make(
        first["env_id"],
        obs_mode=first["obs_mode"],
        control_mode=first["control_mode"],
        robot_uids="xarm6_robotiq",
        reward_mode="dense",
        sim_backend=first.get("sim_backend", "cpu"),
        render_mode="rgb_array",
        **first["env_kwargs"],
    )
    try:
        if json_safe(env.unwrapped.SCENE.to_dict()) != first["scene_spec"]:
            raise ValueError("Saved scene spec differs from the current environment code")
        fields = ("env_module", "env_id", "env_kwargs", "obs_mode", "control_mode", "sim_backend", "scene_spec")
        for row in rows:
            if any(row.get(field) != first.get(field) for field in fields):
                raise ValueError("Saved scenes use different environment configurations; render them separately")
            scene_id = row["scene_id"]
            if Path(scene_id).name != scene_id:
                raise ValueError(f"Invalid scene ID: {scene_id}")
            state_path = (root / row["initial_state"]).resolve()
            if not state_path.is_relative_to(root):
                raise ValueError(f"Scene snapshot is outside source directory: {state_path}")
            snapshot = load_state_npz(state_path, device=env.unwrapped.device)
            env.reset(seed=int(row["seed"]), options={"reset_to_env_states": {"env_states": snapshot}})
            save_frame(env, output_dir / f"{scene_id}.png")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-module", default="envs.candid_dense_clutter_env")
    parser.add_argument("--env-id", default="CANDIDDenseClutter-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-scene", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--scene-source", type=Path,
                        help="Saved dataset directory; render every scene snapshot")
    parser.add_argument("--output-dir", type=Path,
                        help="Directory for saved scene PNGs (default: SCENE_SOURCE/previews)")
    args = parser.parse_args()

    if args.scene_source is not None:
        if args.output is not None or args.fixed_scene:
            parser.error("With --scene-source, use --output-dir for the PNG destination")
        preview_saved_scenes(args.scene_source,
                             args.output_dir or args.scene_source / "previews")
        return
    if args.output_dir is not None:
        parser.error("--output-dir requires --scene-source")
    importlib.import_module(args.env_module)
    env = gym.make(
        args.env_id,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        robot_uids="xarm6_robotiq",
        reward_mode="dense",
        sim_backend="cpu",
        render_mode="rgb_array",
        randomize_layout=not args.fixed_scene,
    )
    try:
        env.reset(seed=args.seed)
        save_frame(env, args.output or Path("demos/dense_scene.png"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
