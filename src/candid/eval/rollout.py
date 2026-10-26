"""Closed-loop episode runner shared by every CANDID policy.

Replay held-out initial Scene snapshots, feed live state_dict observations
through the canonical adapter, and execute whatever action chunk the policy
returns. This module does not load a checkpoint or define a network.
"""
from __future__ import annotations

import csv
import json
import re
from collections import deque
from pathlib import Path

import numpy as np

from candid.data.adapter import canonical_observation, observation_vector
from candid.data.splits import validate_scene_split
from candid.data.storage import load_state_npz
from candid.data.validate import validate_dataset
from candid.eval.metrics import SafetyLimits, evaluate_outcome, summarize_episodes
from candid.eval.replay_env import resolve_scene_environment


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _single(value):
    arr = _numpy(value)
    if arr.shape[0] != 1:
        raise ValueError(f"Evaluation requires batch size 1, got {arr.shape}")
    return arr[0].copy()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _safe_file_stem(scene_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", scene_id)[:160]


def _canonical_live(raw_obs: dict, n_distractors: int) -> np.ndarray:
    """Reuse training's state_dict -> feature mapping, for one live timestep."""
    if not isinstance(raw_obs, dict) or "agent" not in raw_obs or "extra" not in raw_obs:
        raise ValueError("Expected state_dict observation with agent and extra groups")
    groups = {
        key: {name: _numpy(value) for name, value in raw_obs[key].items()}
        for key in ("agent", "extra")
    }
    canonical = canonical_observation(groups, n_distractors)
    vec = observation_vector(canonical)
    if vec.shape[0] != 1 or not np.isfinite(vec).all():
        raise ValueError(f"Invalid live observation shape or values: {vec.shape}")
    return vec[0]


def _frame_record(raw_obs: dict) -> dict:
    return {
        "robot_qpos": _single(raw_obs["agent"]["qpos"]),
        "robot_qvel": _single(raw_obs["agent"]["qvel"]),
        "tcp_pose": _single(raw_obs["extra"]["tcp_pose"]),
        "target_pose": _single(raw_obs["extra"]["target_pose"]),
        "goal_pos": _single(raw_obs["extra"]["goal_pos"]),
    }


def _is_true(value) -> bool:
    return bool(_numpy(value).reshape(-1)[0])


def _check_initial_scene(scene: dict, base, trace) -> None:
    """Fail instead of silently evaluating in the wrong physical scene."""
    from candid.data.monitor import as_numpy

    if scene.get("distractor_ids") is not None and list(scene["distractor_ids"]) != list(trace.names):
        raise RuntimeError(f"Distractor order mismatch for {scene['scene_id']}")
    expected = (
        ("stable_initial_distractor_poses", trace.arrays()["distractor_poses"][0]),
        ("stable_initial_target_pose", as_numpy(base.cube.pose.raw_pose).reshape(-1, 7)[0]),
        ("stable_initial_robot_qpos", as_numpy(base.agent.robot.get_qpos()).reshape(1, -1)[0]),
    )
    for key, actual in expected:
        if scene.get(key) is None:
            continue
        target = np.asarray(scene[key], dtype=float)
        if target.shape != actual.shape or not np.allclose(actual, target, atol=1e-4, rtol=0):
            raise RuntimeError(f"Scene snapshot restore mismatch: {key} ({scene['scene_id']})")


def _clamp_action(action: np.ndarray, action_space) -> tuple[np.ndarray, bool]:
    """Bound commands to the environment's documented control action space."""
    if action.size != int(np.prod(action_space.shape)):
        raise ValueError(f"Wrong action dimension: {action.shape} vs {action_space.shape}")
    if not np.isfinite(action).all():
        raise RuntimeError("Policy produced NaN/Inf action")
    raw = action.reshape(action_space.shape)
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    bounded = np.clip(raw, low, high).astype(np.float32)
    return bounded, bool(np.any(bounded != raw))


def _rollout_single(
    scene: dict,
    dataset: Path,
    policy,
    episode_dir: Path,
    limits: SafetyLimits,
    *,
    max_steps: int,
    exec_horizon: int,
    eval_seed: int,
    save_video: bool,
    stop_on_success: bool,
) -> dict:
    """One episode from the saved scene snapshot, using the supplied policy."""
    import gymnasium as gym
    from mani_skill.utils.wrappers.record import RecordEpisode
    from candid.data.monitor import PhysicsTrace

    if scene.get("control_mode", "pd_joint_pos") != "pd_joint_pos":
        raise ValueError("Only pd_joint_pos scenes can be replayed")
    if scene.get("obs_mode", "state_dict") != "state_dict":
        raise ValueError("Only state_dict scenes can be replayed")
    env_id, environment_source = resolve_scene_environment(scene)
    episode_dir.mkdir(parents=True, exist_ok=False)
    raw_env = gym.make(
        env_id,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        render_mode="rgb_array" if save_video else None,
        robot_uids="xarm6_robotiq",
        reward_mode="dense",
        sim_backend="cpu",
        max_episode_steps=max_steps,
        **scene.get("env_kwargs", {}),
    )
    trace = PhysicsTrace(raw_env)
    recorder = None
    env = trace
    if save_video:
        recorder = RecordEpisode(
            trace,
            output_dir=str(episode_dir),
            trajectory_name="dp_rollout",
            save_trajectory=False,
            save_video=True,
            save_on_reset=False,
            clean_on_close=False,
            record_env_state=False,
        )
        env = recorder

    try:
        saved = load_state_npz(dataset / scene["initial_state"], device="cpu")
        raw_obs, info = env.reset(
            seed=int(scene["seed"]),
            options={"reset_to_env_states": {"env_states": saved}},
        )
        if getattr(env.unwrapped, "num_envs", 1) != 1:
            raise ValueError("CANDID Evaluation requires one CPU environment")
        _check_initial_scene(scene, env.unwrapped, trace)
        if int(np.prod(env.action_space.shape)) != policy.action_dim:
            raise ValueError("Policy action dimension does not match environment")
        num_distractors = len(trace.names)
        current = _canonical_live(raw_obs, num_distractors)
        if current.shape[-1] != policy.obs_dim:
            raise ValueError(f"Policy expects obs_dim={policy.obs_dim}, got {current.shape[-1]}")
        if hasattr(policy, "begin_episode"):
            policy.begin_episode(eval_seed)
        history = deque([current.copy() for _ in range(policy.obs_horizon)], maxlen=policy.obs_horizon)
        frames = [_frame_record(raw_obs)]
        actions: list[np.ndarray] = []
        clipped_actions = 0
        success_once = _is_true(info.get("success", False))
        current_success = success_once
        terminated = truncated = False
        replans = 0
        while len(actions) < max_steps:
            window = np.stack(list(history)).astype(np.float32)
            chunk = np.asarray(policy.act(window), dtype=np.float32)
            if chunk.shape != (policy.action_horizon, policy.action_dim):
                raise RuntimeError(f"Invalid predicted action chunk: {chunk.shape}")
            replans += 1
            for action in chunk[:exec_horizon]:
                command, was_clipped = _clamp_action(action, env.action_space)
                raw_obs, reward, terminated_raw, truncated_raw, info = env.step(command)
                actions.append(command.reshape(-1))
                frames.append(_frame_record(raw_obs))
                clipped_actions += int(was_clipped)
                current = _canonical_live(raw_obs, num_distractors)
                history.append(current)
                current_success = _is_true(info.get("success", False))
                success_once |= current_success
                terminated = _is_true(terminated_raw)
                truncated = _is_true(truncated_raw)
                if terminated or truncated or (stop_on_success and current_success) or len(actions) >= max_steps:
                    break
            if terminated or truncated or (stop_on_success and current_success):
                break
        if recorder is not None:
            recorder.flush_video(save=True)
        arrays = trace.arrays()
        dr = trace.summary()
        outcome = evaluate_outcome(current_success, dr, limits)
        episode_npz = episode_dir / "trajectory.npz"
        np.savez_compressed(
            episode_npz,
            actions=np.stack(actions).astype(np.float32) if actions else np.zeros((0, policy.action_dim), dtype=np.float32),
            **{key: np.stack([frame[key] for frame in frames]) for key in frames[0]},
            **arrays,
        )
        videos = [str(path) for path in sorted(episode_dir.rglob("*.mp4"))]
        if save_video and not videos:
            raise RuntimeError(f"Video was requested but no MP4 was saved in {episode_dir}")
        row = {
            "scene_id": scene["scene_id"],
            "original_env_id": scene["env_id"],
            "original_env_module": scene["env_module"],
            "environment_source": environment_source,
            "eval_seed": eval_seed,
            "num_actions": len(actions),
            "num_replans": replans,
            "clipped_actions": clipped_actions,
            "success_once": success_once,
            "terminated": terminated,
            "truncated": truncated,
            "dr_mean": dr["dr_mean"],
            "dr_max": dr["dr_max"],
            "topple_count": int(dr.get("topple_count", 0)),
            "fall_count": int(dr.get("fall_count", 0)),
            "disturbance": dr,
            **outcome,
            "video_files": videos,
            "trajectory_file": str(episode_npz),
        }
        (episode_dir / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        return row
    finally:
        env.close()


def evaluate_dataset(
    dataset_root,
    policy,
    output_dir,
    limits: SafetyLimits,
    *,
    checkpoint=None,
    max_scenes: int | None = None,
    scene_id: str | None = None,
    episodes_per_scene: int = 1,
    max_steps: int = 300,
    exec_horizon: int = 8,
    eval_seed: int = 2026,
    save_video: bool = True,
    stop_on_success: bool = True,
) -> dict:
    dataset = Path(dataset_root)
    output = Path(output_dir)
    if max_steps < 1 or episodes_per_scene < 1:
        raise ValueError("max_steps and episodes_per_scene must be positive")
    if max_scenes is not None and max_scenes < 1:
        raise ValueError("max_scenes must be positive")
    validate_dataset(dataset)
    split_path = dataset / "split.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Evaluation needs the frozen training split: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validate_scene_split(split, dataset)
    test_ids = split["test"]
    if scene_id is not None:
        if scene_id not in test_ids:
            raise ValueError(f"Requested scene is not in the held-out TEST set: {scene_id}")
        test_ids = [scene_id]
    if max_scenes is not None:
        test_ids = test_ids[:max_scenes]
    if not test_ids:
        raise ValueError("No test scenes available")
    scenes = {row["scene_id"]: row for row in _read_jsonl(dataset / "scenes.jsonl")}
    if exec_horizon < 1 or exec_horizon > policy.action_horizon:
        raise ValueError(f"exec_horizon must be in [1, {policy.action_horizon}]")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output folder must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, selected_id in enumerate(test_ids):
        scene = scenes[selected_id]
        for repetition in range(episodes_per_scene):
            seed = eval_seed + index * 100000 + repetition
            np.random.seed(seed % (2**32))
            name = f"{index:03d}_{_safe_file_stem(selected_id)}_rep{repetition:02d}"
            row = _rollout_single(
                scene, dataset, policy, output / name, limits,
                max_steps=max_steps, exec_horizon=exec_horizon,
                eval_seed=seed, save_video=save_video,
                stop_on_success=stop_on_success,
            )
            rows.append(row)
            with (output / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            csv_fields = ("scene_id", "eval_seed", "num_actions", "num_replans",
                          "clipped_actions", "task_success", "success_once", "safe",
                          "safe_success", "dr_mean", "dr_max", "topple_count", "fall_count")
            with (output / "episodes.csv").open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=csv_fields)
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow({key: row[key] for key in csv_fields})
            print(f"{selected_id} rep={repetition}: success={row['task_success']} "
                  f"safe={row['safe']} safe_success={row['safe_success']} "
                  f"DR(mean/max)={row['dr_mean']:.5f}/{row['dr_max']:.5f} "
                  f"topple={row['topple_count']} fall={row['fall_count']}", flush=True)
    report = summarize_episodes(rows, limits)
    report["clipped_action_rate"] = (sum(row["clipped_actions"] for row in rows)
                                     / max(1, sum(row["num_actions"] for row in rows)))
    report["by_scene"] = {
        selected_id: summarize_episodes([row for row in rows if row["scene_id"] == selected_id], limits)
        for selected_id in test_ids
    }
    report["metric_definitions"] = {
        "success_rate": "Fraction of episodes successful at termination",
        "success_once_rate": "Fraction of episodes successful at least once",
        "safety_rate": "No topple/fall AND real DR within all configured bounds",
        "safe_success_rate": "Task success at termination AND safe",
        "disturbance": "CANDID weighted, size-normalized peak displacement/rotation from initial scene",
    }
    report.update({
        "dataset": str(dataset), "checkpoint": None if checkpoint is None else str(checkpoint),
        "split_file": str(split_path), "test_scene_ids": test_ids,
        "episodes_per_scene": episodes_per_scene, "exec_horizon": exec_horizon,
        "max_steps": max_steps, "eval_seed": eval_seed,
        "stop_on_success": stop_on_success,
    })
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
