"""Collect physically relabeled counterfactual 8-step chunks from TRAIN expert scenes.

Each candidate starts from the archived initial state and replays ALL expert
prefix commands in a fresh CPU simulator. Never copy a label from a perturbed
expert action. This collector does not train or load a Predictor/DP checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, deque
from pathlib import Path

import h5py
import numpy as np

from candid.data.adapter import canonical_observation, observation_vector
from candid.data.counterfactual import choose_probes, make_candidates, stable_seed
from candid.data.metrics import disturbance_from_poses
from candid.data.splits import validate_scene_split
from candid.data.storage import freeze_state, json_safe, load_state_npz
from candid.eval.local_dr import _check_signature, _signature, make_env
from candid.eval.replay_env import resolve_scene_environment
from candid.eval.rollout import _canonical_live, _check_initial_scene, _clamp_action, _is_true


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_train_source(root):
    """Require the existing split; never invent a new train/test assignment."""
    root = Path(root)
    split_file = root / "split.json"
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing frozen split: {split_file}")
    split = json.loads(split_file.read_text(encoding="utf-8"))
    validate_scene_split(split, root)
    if not split.get("train"):
        raise ValueError("Training split is empty")
    scenes = {row["scene_id"]: row for row in _read_jsonl(root / "scenes.jsonl")}
    for sid in split["train"]:
        if sid not in scenes:
            raise ValueError(f"Train scene absent from manifest: {sid}")
        scene = scenes[sid]
        if (scene.get("control_mode", "pd_joint_pos") != "pd_joint_pos"
                or scene.get("obs_mode", "state_dict") != "state_dict"):
            raise ValueError(f"Wrong environment control/obs mode in {sid}")
    return split, scenes


def _load_expert(root, row):
    with h5py.File(root / row["hdf5_file"], "r") as handle:
        group = handle[row["hdf5_trajectory_key"]]
        actions = np.asarray(group["actions"], dtype=np.float32)
        obs = observation_vector(
            canonical_observation(group["obs"], len(row["object_ids"])))
    with np.load(root / row["disturbance_trace"], allow_pickle=False) as archive:
        truth = {k: archive[k].copy() for k in (
            "distractor_poses", "object_sizes_m",
            "distractor_toppled", "distractor_fallen")}
    if len(obs) != len(actions) + 1 or len(truth["distractor_poses"]) != len(obs):
        raise ValueError(f"Source trajectory length mismatch: {row['trajectory_id']}")
    return obs, actions, truth


def _simulate(scene, snapshot, prefix, candidate, *, expected_signature=None,
              source_pose=None, source_obs=None, state_atol=0.005,
              obs_atol=0.005, allow_fallback=False):
    env_id, source = resolve_scene_environment(scene)
    if source != "original_env_module" and not allow_fallback:
        raise RuntimeError("Original environment module is unavailable; refusing "
                           "to relabel physics using SceneSpec fallback")
    trace = make_env(scene, max(len(prefix) + len(candidate) + 10, 300), env_id)
    try:
        raw, _ = trace.reset(
            seed=int(scene["seed"]),
            options={"reset_to_env_states": {"env_states": freeze_state(snapshot)}})
        _check_initial_scene(scene, trace.unwrapped, trace)
        if list(trace.names) != list(scene.get("distractor_ids", trace.names)):
            raise RuntimeError("Distractor order does not match source scene")
        if int(np.prod(trace.action_space.shape)) != candidate.shape[1]:
            raise ValueError("Candidate action dimension differs from environment")
        history = deque([_canonical_live(raw, len(trace.names)).copy()] * 2,
                        maxlen=2)
        for t, action in enumerate(prefix):
            command, clipped = _clamp_action(action, trace.action_space)
            if clipped:
                raise ValueError(f"Source expert prefix action {t} was clipped")
            raw, _, terminated, truncated, _ = trace.step(command)
            history.append(_canonical_live(raw, len(trace.names)).copy())
            if _is_true(terminated) or _is_true(truncated):
                raise RuntimeError(f"Expert replay terminated before probe at {t+1}")
        start = trace.steps
        signature = _signature(trace)
        if expected_signature is not None:
            _check_signature(expected_signature, signature, atol=state_atol)
        if source_pose is not None:
            actual = trace.arrays()["distractor_poses"][start]
            if actual.shape != source_pose.shape:
                raise ValueError("Source pose shape mismatch")
            if np.max(np.abs(actual - source_pose)) > state_atol:
                raise RuntimeError("Archived expert state not reproduced by physics replay")
        obs_history = np.stack(history).astype(np.float32)
        if source_obs is not None:
            error = float(np.max(np.abs(obs_history - source_obs)))
            if error > obs_atol:
                raise RuntimeError(f"Archived expert observation replay mismatch: {error}")
        else:
            error = None
        for index, action in enumerate(candidate):
            command, clipped = _clamp_action(action, trace.action_space)
            if clipped:
                raise RuntimeError(f"Candidate action {index} unexpectedly clipped")
            _, _, _, truncated, _ = trace.step(command)
            if _is_true(truncated):
                raise RuntimeError("Candidate truncated before full 8-step outcome")
        arrays = trace.arrays()
        label = disturbance_from_poses(
            arrays["distractor_poses"], arrays["object_sizes_m"],
            start=start, stop=start + len(candidate),
            topple_flags=arrays["distractor_toppled"],
            fall_flags=arrays["distractor_fallen"])
        return {"label": label, "signature": signature,
                "observation": obs_history, "source_obs_max_abs": error}
    finally:
        trace.close()


def collect(dataset, output_dir, *, states=2, candidates=4, seed=2026,
            sigma=0.02, max_per_scene=4, max_prefix_steps=None,
            state_atol=0.005, obs_atol=0.005, dr_atol=0.02,
            allow_fallback=False, perturb_gripper=False):
    dataset = Path(dataset)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    if states < 1 or candidates < 2 or dr_atol < 0:
        raise ValueError("Require positive states, >=2 candidates, nonnegative DR tolerance")
    split, scenes = load_train_source(dataset)
    probes = choose_probes(
        dataset, split["train"], count=states, seed=seed,
        max_per_scene=max_per_scene, max_prefix_steps=max_prefix_steps)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    visited = set()
    manifest = output_dir / "states.jsonl"
    for index, probe in enumerate(probes):
        if probe.scene_id not in split["train"]:
            raise RuntimeError("Refusing any test/validation scene")
        source_row = probe.trajectory
        scene = scenes[probe.scene_id]
        observations, actions, truth = _load_expert(dataset, source_row)
        t = probe.step
        if t < 1 or t + 8 > len(actions):
            raise ValueError("Counterfactual requires obs_horizon=2, action_horizon=8")
        obs_window = observations[t-1:t+1]
        prefix, expert = actions[:t], actions[t:t+8]
        archived = disturbance_from_poses(
            truth["distractor_poses"], truth["object_sizes_m"],
            start=t, stop=t+8,
            topple_flags=truth["distractor_toppled"],
            fall_flags=truth["distractor_fallen"])
        snapshot = load_state_npz(dataset / scene["initial_state"], device="cpu")
        env_id, source = resolve_scene_environment(scene)
        if source != "original_env_module" and not allow_fallback:
            raise RuntimeError("Missing original scene environment; refusing fallback labels")
        env = make_env(scene, max(t+18, 300), env_id)
        try:
            low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        finally:
            env.close()
        actions_to_compare = make_candidates(
            expert, low, high, count=candidates,
            seed=stable_seed(seed, probe.trajectory_id, t), sigma=sigma,
            freeze_last_dimension=not perturb_gripper)
        actual = [_simulate(
            scene, snapshot, prefix, actions_to_compare[0],
            source_pose=truth["distractor_poses"][t],
            source_obs=obs_window, state_atol=state_atol,
            obs_atol=obs_atol, allow_fallback=allow_fallback)]
        for metric in ("dr_mean", "dr_max"):
            delta = abs(actual[0]["label"][metric] - archived[metric])
            if delta > dr_atol:
                raise RuntimeError(f"Expert source/replay {metric} mismatch "
                                   f"{delta} > {dr_atol}: {probe.trajectory_id}:{t}")
        for event in ("topple_count", "fall_count"):
            if actual[0]["label"][event] != archived[event]:
                raise RuntimeError(f"Expert source/replay {event} mismatch: "
                                   f"{probe.trajectory_id}:{t}")
        for alternate in actions_to_compare[1:]:
            actual.append(_simulate(
                scene, snapshot, prefix, alternate,
                expected_signature=actual[0]["signature"],
                state_atol=state_atol, allow_fallback=allow_fallback))
        targets = np.asarray([
            [r["label"]["dr_mean"], r["label"]["dr_max"],
             float(r["label"]["any_topple"]), float(r["label"]["any_fall"])]
            for r in actual], dtype=np.float32)
        obj_dr = np.asarray([r["label"]["per_object_dr"] for r in actual], np.float32)
        obj_topple = np.asarray([r["label"]["per_object_topple"] for r in actual], bool)
        obj_fall = np.asarray([r["label"]["per_object_fall"] for r in actual], bool)
        digest = hashlib.sha256(
            f"{probe.trajectory_id}:{t}".encode()).hexdigest()[:16]
        state_id = f"{index:05d}_{digest}"
        rel = f"samples/{state_id}.npz"
        np.savez_compressed(
            output_dir / rel, obs=actual[0]["observation"],
            actions=actions_to_compare, targets=targets,
            per_object_dr=obj_dr, per_object_topple=obj_topple,
            per_object_fall=obj_fall, object_sizes_m=truth["object_sizes_m"])
        meta = {"state_id": state_id, "scene_id": probe.scene_id,
                "source_trajectory_id": probe.trajectory_id, "source_t": t,
                "source_split": "train", "stratum": probe.stratum,
                "file": rel, "object_ids": source_row["object_ids"],
                "num_candidates": candidates,
                "perturbation_sigma": sigma,
                "reference_obs_replay_max_abs": actual[0]["source_obs_max_abs"],
                "expert_source_dr_mean": archived["dr_mean"],
                "expert_replayed_dr_mean": float(targets[0,0]),
                "physical_relabeling": "fresh_initial_scene_full_expert_prefix_each_candidate"}
        with manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(json_safe(meta), ensure_ascii=False) + "\n")
        counts[probe.stratum] += 1
        counts["topple_candidates"] += int((targets[:,2] > 0).sum())
        counts["states_with_distinct_topple"] += int(len(set(targets[:,2])) > 1)
        counts["physical_action_steps"] += candidates * (t+8)
        visited.add(probe.scene_id)
        print(f"[{index+1}/{states}] {probe.trajectory_id} t={t} "
              f"{probe.stratum} topples={targets[:,2].astype(int).tolist()}",
              flush=True)
    summary = {"schema_version":1, "source_dataset":str(dataset),
               "source_split_sha256":split["dataset_sha256"],
               "source_split":"train", "num_states":states,
               "num_candidates_per_state":candidates,
               "physical_rollouts":states*candidates,
               "distinct_source_scenes":len(visited),
               "strata":dict(counts),
               "topple_candidates":counts["topple_candidates"],
               "states_with_distinct_topple":counts["states_with_distinct_topple"],
               "replayed_physics_action_steps":counts["physical_action_steps"],
               "seed":seed,"sigma":sigma,"obs_horizon":2,"action_horizon":8,
               "label_names":["dr_mean","dr_max","any_topple","any_fall"],
               "labels_from_simulation":True, "trained_model":False}
    (output_dir/"summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--states",type=int,default=2)
    p.add_argument("--candidates",type=int,default=4)
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--sigma",type=float,default=0.02)
    p.add_argument("--max-per-scene",type=int,default=4)
    p.add_argument("--max-prefix-steps",type=int,default=None)
    p.add_argument("--state-atol",type=float,default=0.005)
    p.add_argument("--obs-atol",type=float,default=0.005)
    p.add_argument("--dr-atol",type=float,default=0.02)
    p.add_argument("--allow-scene-spec-fallback",action="store_true")
    p.add_argument("--perturb-gripper",action="store_true")
    args=p.parse_args(argv)
    report=collect(
        args.dataset,args.output_dir,states=args.states,
        candidates=args.candidates,seed=args.seed,sigma=args.sigma,
        max_per_scene=args.max_per_scene,max_prefix_steps=args.max_prefix_steps,
        state_atol=args.state_atol,obs_atol=args.obs_atol,dr_atol=args.dr_atol,
        allow_fallback=args.allow_scene_spec_fallback,
        perturb_gripper=args.perturb_gripper)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return report


if __name__=="__main__":
    main()
