"""One-scene paired guidance record from a frozen dataset observation.

Closed-loop execution needs the simulator. Without it, the comparison uses
the recorded observation at the requested step. That is the same observation
vector training and evaluation build, and it is not a regenerated scene.
The vanilla chunk is a counterfactual. It is not executed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from candid.data.adapter import canonical_observation, observation_vector
from model.guidance.figures import save_comparison_figures
from model.guidance.policy import ConsequenceGuidedPolicy
from model.predictor.checkpoint import load_predictor
from model.vanilla_dp.checkpoint import load_policy


def inspect_scene(
    eval_suite,
    scene_id,
    dp_checkpoint,
    predictor_checkpoint,
    output_dir,
    *,
    strength=0.005,
    start_timestep=30,
    exec_horizon=4,
    eval_seed=2026,
    device="cpu",
    trace_replan=0,
    episode_step=None,
    max_replans=1,
    weights=(1.0, 0.25, 0.1, 0.0),
    render=False,
):
    if render:
        raise RuntimeError(
            "Simulator rendering is unavailable in this run. Use --no-render. "
            "The numerical comparison and plots do not need a rendered frame."
        )
    if max_replans != 1:
        raise ValueError(
            "Without a simulator this command compares one recorded observation. "
            "Pass --max-replans 1."
        )
    if exec_horizon < 1:
        raise ValueError("exec_horizon must be positive")
    suite = Path(eval_suite)
    scene = _scene(suite, scene_id)
    trajectory = _trajectory(suite, scene_id)
    dp_payload = torch.load(dp_checkpoint, map_location=device, weights_only=False)
    obs_normalizer = observation_normalizer_from_checkpoint(dp_payload)
    dp_policy = load_policy(dp_checkpoint, device)
    predictor, predictor_normalizer = load_predictor(predictor_checkpoint, device)
    policy = ConsequenceGuidedPolicy(
        dp_policy,
        predictor,
        predictor_normalizer,
        strength=strength,
        start_timestep=start_timestep,
        weights=weights,
        obs_normalizer=obs_normalizer,
    )
    if policy.action_horizon < exec_horizon:
        raise ValueError(
            f"exec_horizon {exec_horizon} exceeds the checkpoint action horizon {policy.action_horizon}"
        )
    low = np.asarray(dp_payload["action_min"], dtype=np.float32).reshape(-1)
    high = np.asarray(dp_payload["action_max"], dtype=np.float32).reshape(-1)
    policy.set_bounds(low, high)
    step = 0 if episode_step is None else int(episode_step)
    if episode_step is None and int(trace_replan) > 0:
        step = int(trace_replan) * int(exec_horizon)
    window = load_observation_window(
        suite,
        trajectory,
        step,
        policy.obs_horizon,
        len(trajectory["object_ids"]),
    )
    if window.shape[1] != policy.obs_dim:
        raise ValueError(
            f"Recorded observation has {window.shape[1]} features; "
            f"the DP checkpoint expects {policy.obs_dim}"
        )
    policy.record_pair = True
    policy.trace = True
    policy.begin_episode(eval_seed)
    comparison = policy.compare(window)
    comparison["start_timestep"] = int(start_timestep)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    replan_index = int(trace_replan) if episode_step is None and int(trace_replan) > 0 else 0
    record = _record(scene_id, step, replan_index, strength, comparison)
    _write_outputs(
        output,
        record,
        comparison,
        exec_horizon=exec_horizon,
        metadata=_metadata(
            suite, scene, trajectory, dp_checkpoint, predictor_checkpoint,
            strength, start_timestep, exec_horizon, eval_seed, device,
            weights, step, replan_index, obs_normalizer is not None,
        ),
    )
    figures = save_comparison_figures(
        comparison,
        output / "figures",
        scene_id=scene_id,
        episode_step=step,
        replan_index=replan_index,
        strength=strength,
        exec_horizon=exec_horizon,
    )
    return {"output_dir": str(output), "record": record, "figures": figures}


def observation_normalizer_from_checkpoint(payload):
    """Return a denoiser-only observation transform, or None.

    Current Vanilla DP checkpoints leave observations unnormalized. A
    checkpoint that says ``normalize_obs`` must carry its own statistics.
    """
    enabled = bool((payload.get("config") or {}).get("normalize_obs", False))
    if not enabled:
        return None
    if "obs_mean" not in payload or "obs_std" not in payload:
        raise ValueError(
            "DP checkpoint sets normalize_obs=True but has no obs_mean/obs_std"
        )
    mean = torch.as_tensor(payload["obs_mean"], dtype=torch.float32)
    std = torch.as_tensor(payload["obs_std"], dtype=torch.float32)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError("DP observation statistics must be finite and positive")

    def normalize(obs):
        return (obs - mean.to(obs.device)) / std.to(obs.device)

    return normalize


def load_observation_window(dataset, trajectory, episode_step, obs_horizon, num_distractors):
    import h5py

    path = Path(dataset) / trajectory["hdf5_file"]
    with h5py.File(path, "r") as handle:
        raw = _numpy_tree(handle[trajectory["hdf5_trajectory_key"]]["obs"])
    canonical = canonical_observation(raw, num_distractors)
    length = canonical["robot_qpos"].shape[0]
    if episode_step < 0 or episode_step >= length:
        raise ValueError(f"episode_step {episode_step} is outside the recorded length {length}")
    end = episode_step + 1
    start = end - obs_horizon
    window = {}
    for key, value in canonical.items():
        if start < 0:
            pad = np.repeat(value[:1], -start, axis=0)
            window[key] = np.concatenate((pad, value[:end]), axis=0)
        else:
            window[key] = value[start:end]
    vector = observation_vector(window).astype(np.float32)
    if vector.shape[0] != obs_horizon:
        raise RuntimeError("Observation window has the wrong horizon")
    return vector


def _scene(suite, scene_id):
    path = Path(suite) / "scenes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No scenes.jsonl in {suite}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("scene_id") == scene_id:
            return row
    raise ValueError(f"scene_id {scene_id!r} is not in {path}")


def _trajectory(suite, scene_id):
    path = Path(suite) / "trajectories.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No trajectories.jsonl in {suite}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("scene_id") == scene_id and row.get("hdf5_file") and row.get("hdf5_trajectory_key"):
            return row
    raise ValueError(f"No recorded HDF5 trajectory for scene {scene_id}")


def _numpy_tree(group):
    import h5py

    tree = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            tree[key] = _numpy_tree(value)
        else:
            tree[key] = np.asarray(value)
    return tree


def _record(scene_id, episode_step, replan_index, strength, comparison):
    return {
        "scene_id": scene_id,
        "episode_step": int(episode_step),
        "replan_index": int(replan_index),
        "paired_same_obs_and_rng": True,
        "observation_source": "recorded_demonstration",
        "closed_loop_executed": False,
        "guidance_strength": float(strength),
        "predicted_local_dr": {
            "vanilla": {
                "dr_mean": comparison["scores"]["vanilla"]["dr_mean"],
                "dr_max": comparison["scores"]["vanilla"]["dr_max"],
            },
            "guided": {
                "dr_mean": comparison["scores"]["guided"]["dr_mean"],
                "dr_max": comparison["scores"]["guided"]["dr_max"],
            },
        },
        "pred_topple_score": {
            "vanilla": comparison["scores"]["vanilla"]["topple_score"],
            "guided": comparison["scores"]["guided"]["topple_score"],
        },
        "pred_fall_score": {
            "vanilla": comparison["scores"]["vanilla"]["fall_score"],
            "guided": comparison["scores"]["guided"]["fall_score"],
        },
        "pred_weighted_risk": {
            "vanilla": comparison["scores"]["vanilla"]["weighted_risk"],
            "guided": comparison["scores"]["guided"]["weighted_risk"],
        },
        "delta_dr_mean_guided_minus_vanilla": comparison["delta_dr_mean"],
        "delta_dr_max_guided_minus_vanilla": comparison["delta_dr_max"],
        "delta_risk_guided_minus_vanilla": comparison["delta_risk"],
        "action_abs_mean": comparison["action_abs_mean"],
        "action_abs_max": comparison["action_abs_max"],
        "action_l2": comparison["action_l2"],
        "action_abs_mean_per_joint": comparison["action_abs_mean_per_joint"].tolist(),
        "guided_diffusion_steps": comparison["guided_diffusion_steps"],
    }


def _metadata(suite, scene, trajectory, dp_checkpoint, predictor_checkpoint,
              strength, start_timestep, exec_horizon, eval_seed, device,
              weights, episode_step, replan_index, obs_normalized):
    return {
        "eval_suite": str(suite),
        "scene_id": scene["scene_id"],
        "trajectory_id": trajectory["trajectory_id"],
        "hdf5_file": trajectory["hdf5_file"],
        "hdf5_trajectory_key": trajectory["hdf5_trajectory_key"],
        "episode_step": int(episode_step),
        "replan_index": int(replan_index),
        "observation_source": "recorded_demonstration",
        "closed_loop_executed": False,
        "dp_checkpoint": str(dp_checkpoint),
        "dp_checkpoint_sha256": _sha256(dp_checkpoint),
        "predictor_checkpoint": str(predictor_checkpoint),
        "predictor_checkpoint_sha256": _sha256(predictor_checkpoint),
        "dp_observation_normalization": bool(obs_normalized),
        "action_clip_source": "dp_training_action_range",
        "strength": float(strength),
        "start_timestep": int(start_timestep),
        "exec_horizon": int(exec_horizon),
        "eval_seed": int(eval_seed),
        "device": str(device),
        "risk_weights": {
            "dr_mean": weights[0],
            "dr_max": weights[1],
            "topple_score": weights[2],
            "fall_score": weights[3],
        },
        "notes": [
            "Vanilla and guided chunks use the same observation and the same initial diffusion noise.",
            "Predictor scores are local chunk estimates, not measured physical DR.",
            "Diffusion-step risk is the intermediate x0 estimate, not the final action chunk.",
            "The correction is subtracted from the noisy sample after DDPMScheduler.step.",
            "No action from the vanilla chunk was executed.",
        ],
    }


def _write_outputs(output, record, comparison, metadata, exec_horizon):
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (output / "replans.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    np.savez(
        output / "chunks.npz",
        vanilla_actions=comparison["vanilla_raw"][None],
        guided_actions=comparison["guided_raw"][None],
        vanilla_actions_clipped=comparison["vanilla_clipped"][None],
        guided_actions_clipped=comparison["guided_clipped"][None],
        replan_indices=np.asarray([record["replan_index"]], dtype=np.int64),
        episode_steps=np.asarray([record["episode_step"]], dtype=np.int64),
        vanilla_exec=comparison["vanilla_clipped"][None, :exec_horizon],
        guided_exec=comparison["guided_clipped"][None, :exec_horizon],
    )
    rows = comparison["diffusion_trace"]
    with (output / "diffusion_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            public = {key: value for key, value in row.items() if key not in ("x0_action_chunk", "correction")}
            public["replan_index"] = record["replan_index"]
            handle.write(json.dumps(public) + "\n")
    if rows:
        def column(name, fill=np.nan):
            values = [fill if row[name] is None else row[name] for row in rows]
            return np.asarray(values, dtype=np.float64)

        x0 = _stack_optional(rows, "x0_action_chunk")
        correction = _stack_optional(rows, "correction")
        np.savez(
            output / "diffusion_arrays.npz",
            diffusion_timestep=np.asarray([row["diffusion_timestep"] for row in rows], dtype=np.int64),
            guided=np.asarray([row["guided"] for row in rows], dtype=bool),
            pred_dr_mean=column("pred_dr_mean"),
            pred_dr_max=column("pred_dr_max"),
            pred_weighted_risk=column("pred_weighted_risk"),
            raw_gradient_l2=column("raw_gradient_l2", 0.0),
            applied_correction_l2=column("applied_correction_l2", 0.0),
            max_abs_correction=column("max_abs_correction", 0.0),
            x0_action_chunk=x0,
            correction=correction,
        )


def _stack_optional(rows, key):
    present = [row[key] for row in rows if row.get(key) is not None]
    if not present:
        return np.zeros((len(rows), 0, 0), dtype=np.float32)
    sample = np.asarray(present[0])
    stacked = np.full((len(rows),) + sample.shape[1:], np.nan, dtype=np.float32)
    for index, row in enumerate(rows):
        if row.get(key) is not None:
            stacked[index] = np.asarray(row[key], dtype=np.float32)[0]
    return stacked


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
