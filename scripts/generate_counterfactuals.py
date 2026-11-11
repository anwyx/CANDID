"""Collect physically relabeled counterfactual 8-step chunks from TRAIN expert scenes.

With --consequence-horizons 4,8,16,32: label the first 4/8 candidate
steps, then observe effects while holding the final absolute joint command.
This hold protocol is NOT the policy that DP may execute in the future.

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
from candid.data.balanced_counterfactual import choose_balanced_probes, stratum_quotas
from candid.data.counterfactual import (choose_probes, make_candidates,
                                        make_multiscale_candidates, stable_seed)
from candid.data.metrics import disturbance_from_poses
from candid.data.splits import validate_scene_split
from candid.data.storage import freeze_state, json_safe, load_state_npz
from candid.eval.local_dr import _check_signature, _signature, make_env
from candid.eval.replay_env import resolve_scene_environment
from candid.eval.rollout import _canonical_live, _check_initial_scene, _clamp_action, _is_true


class ExpertReplayMismatch(RuntimeError):
    """Archived expert state/outcome cannot be reproduced faithfully."""

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = details or {}


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
              obs_atol=0.005, allow_fallback=False,
              consequence_horizons=(8,)):
    env_id, source = resolve_scene_environment(scene)
    if source != "original_env_module" and not allow_fallback:
        raise RuntimeError("Original environment module is unavailable; refusing "
                           "to relabel physics using SceneSpec fallback")
    trace = make_env(scene, max(len(prefix) + max(consequence_horizons) + 10, 300), env_id)
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
                # A recorded expert trajectory can contain actions beyond a
                # task-success terminal. Do not step past termination and
                # silently invent labels: reject this probe and use reserves.
                reason = "terminated" if _is_true(terminated) else "truncated"
                raise ExpertReplayMismatch(
                    f"Expert prefix {reason} at step {t+1}/{len(prefix)}")
        start = trace.steps
        signature = _signature(trace)
        if expected_signature is not None:
            try:
                _check_signature(expected_signature, signature, atol=state_atol)
            except RuntimeError as exc:
                raise ExpertReplayMismatch(
                    f"Alternative candidate prefix did not reproduce expert state: {exc}"
                ) from exc
        if source_pose is not None:
            actual = trace.arrays()["distractor_poses"][start]
            if actual.shape != source_pose.shape:
                raise ValueError("Source pose shape mismatch")
            if np.max(np.abs(actual - source_pose)) > state_atol:
                raise ExpertReplayMismatch("Archived expert state not reproduced by physics replay")
        obs_history = np.stack(history).astype(np.float32)
        if source_obs is not None:
            error = float(np.max(np.abs(obs_history - source_obs)))
            if error > obs_atol:
                from candid.data.replay_diagnostics import observation_error_breakdown
                breakdown = observation_error_breakdown(obs_history, source_obs)
                raise ExpertReplayMismatch(
                    f"Archived expert observation replay mismatch: {error}; "
                    f"largest_field={breakdown['largest_field']}",
                    details={"observation": breakdown})
        else:
            error = None
        for index, action in enumerate(candidate):
            command, clipped = _clamp_action(action, trace.action_space)
            if clipped:
                raise RuntimeError(f"Candidate action {index} unexpectedly clipped")
            _, _, _, truncated, _ = trace.step(command)
            if _is_true(truncated):
                raise ExpertReplayMismatch("Candidate truncated before full 8-step outcome")
        # H>8 continues with the last absolute joint target. This is a
        # fixed hold protocol, not a claim about future DP closed-loop actions.
        for _ in range(max(consequence_horizons) - len(candidate)):
            command, _ = _clamp_action(candidate[-1], trace.action_space)
            _, _, _, truncated, _ = trace.step(command)
            if _is_true(truncated):
                raise ExpertReplayMismatch("Hold continuation truncated before labeling")
        arrays = trace.arrays()
        multi = [disturbance_from_poses(
            arrays["distractor_poses"], arrays["object_sizes_m"],
            start=start, stop=start+h,
            topple_flags=arrays["distractor_toppled"],
            fall_flags=arrays["distractor_fallen"]) for h in consequence_horizons]
        label = multi[consequence_horizons.index(8)]
        return {"label": label, "multi_labels": multi, "signature": signature,
                "observation": obs_history, "source_obs_max_abs": error}
    finally:
        trace.close()


def collect(dataset, output_dir, *, states=2, candidates=4, seed=2026,
            sigma=0.02, max_per_scene=4, max_prefix_steps=None,
            state_atol=0.005, obs_atol=0.005, dr_atol=0.02,
            allow_fallback=False, perturb_gripper=False,
            candidate_mode="legacy", sigma_levels=(0.02, 0.06, 0.12),
            resume=False, reserve_probes=16,
            consequence_horizons=(8,), probe_strategy="legacy"):
    consequence_horizons = tuple(int(x) for x in consequence_horizons)
    if (8 not in consequence_horizons or sorted(set(consequence_horizons)) != list(consequence_horizons)
            or min(consequence_horizons) < 1 or max(consequence_horizons) < 8):
        raise ValueError("Consequence horizons must be sorted unique positives including 8")
    if probe_strategy not in ("legacy", "balanced"):
        raise ValueError("probe_strategy must be legacy or balanced")
    dataset = Path(dataset)
    output_dir = Path(output_dir)
    if states < 1 or candidates < 2 or dr_atol < 0 or reserve_probes < 0:
        raise ValueError("Require states>=1, candidates>=2, nonnegative tolerances/reserves")
    if candidate_mode == "multiscale" and candidates < 3:
        raise ValueError("Multiscale sampling requires at least three candidates")
    split, scenes = load_train_source(dataset)
    config = {
        "source_dataset": str(dataset), "source_split_sha256": split["dataset_sha256"],
        "states": states, "candidates": candidates, "seed": seed,
        "sigma": sigma, "candidate_mode": candidate_mode,
        "sigma_levels": [float(x) for x in sigma_levels],
        "max_per_scene": max_per_scene,
        "max_prefix_steps": max_prefix_steps,
        "state_atol": state_atol, "obs_atol": obs_atol, "dr_atol": dr_atol,
        "allow_fallback": allow_fallback, "perturb_gripper": perturb_gripper,
    }
    if probe_strategy != "legacy":
        config["probe_strategy"] = probe_strategy
    # Keep resume compatibility with old 8-step CF datasets.
    if consequence_horizons != (8,):
        config["consequence_horizons"] = list(consequence_horizons)
    manifest = output_dir / "states.jsonl"
    failed_manifest = output_dir / "skipped_probes.jsonl"
    config_path = output_dir / "collection_config.json"
    existing = output_dir.exists() and any(output_dir.iterdir())
    if existing and not resume:
        raise FileExistsError("Nonempty output directory: pass --resume to reuse verified samples")
    if resume and existing and not manifest.exists():
        raise RuntimeError("Cannot resume without states.jsonl")
    previous = _read_jsonl(manifest) if existing else []
    failed = _read_jsonl(failed_manifest) if existing and failed_manifest.exists() else []
    for row in previous:
        if row.get("source_split") != "train":
            raise RuntimeError("Resume manifest contains non-training scene")
        if row["scene_id"] not in split["train"]:
            raise RuntimeError("Resume manifest scene not in current training split")
        if row.get("num_candidates") != candidates or row.get("candidate_mode", "legacy") != candidate_mode:
            raise RuntimeError("Existing candidate settings disagree with resume options")
        if float(row.get("perturbation_sigma", sigma)) != float(sigma):
            raise RuntimeError("Existing sigma disagrees with resume options")
        expected_sigmas = ([0.] + [sigma] * (candidates - 1)
                           if candidate_mode == "legacy" else
                           [0.] + [float(sigma_levels[(i-1)//2 % len(sigma_levels)])
                                   for i in range(1, candidates)])
        if not np.allclose(row.get("nominal_sigmas", []), expected_sigmas, atol=1e-6):
            raise RuntimeError("Existing nominal perturbation scales disagree with resume")
        data_path = output_dir / row["file"]
        if not data_path.is_file():
            raise FileNotFoundError(f"Resume sample missing: {data_path}")
        with np.load(data_path, allow_pickle=False) as data:
            if data["targets"].shape != (candidates, 4):
                raise RuntimeError(f"Existing sample shape invalid: {data_path}")
            if len(consequence_horizons) > 1:
                expected = (candidates, len(consequence_horizons), 4)
                if "targets_multi" not in data or data["targets_multi"].shape != expected:
                    raise RuntimeError(f"Existing multihorizon targets invalid: {data_path}")
    if existing and config_path.exists():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        if previous_config != config:
            raise RuntimeError("Resume arguments disagree with collection_config.json")
    elif existing and previous:
        # V1.1 interrupted runs had no config file. Verify their original
        # deterministic probe selection before adopting the saved records.
        legacy_probes = choose_probes(
            dataset, split["train"], count=states, seed=seed,
            max_per_scene=max_per_scene, max_prefix_steps=max_prefix_steps)
        legacy_ids = {(p.trajectory_id, p.step) for p in legacy_probes}
        for row in previous:
            if (row["source_trajectory_id"], row["source_t"]) not in legacy_ids:
                raise RuntimeError("Existing probe does not match deterministic source selection")
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    select = choose_balanced_probes if probe_strategy == "balanced" else choose_probes
    probes = select(
        dataset, split["train"], count=states, seed=seed,
        max_per_scene=max_per_scene, max_prefix_steps=max_prefix_steps,
        reserve_count=reserve_probes)
    desired_strata = stratum_quotas(states) if probe_strategy == "balanced" else None
    counts = Counter()
    visited = set()
    completed = {(row["source_trajectory_id"], row["source_t"]) for row in previous}
    failed_keys = {(row["source_trajectory_id"], row["source_t"]) for row in failed}
    for row in previous:
        with np.load(output_dir / row["file"], allow_pickle=False) as data:
            targets = data["targets"]
        counts[row["stratum"]] += 1
        counts["topple_candidates"] += int((targets[:,2] > 0).sum())
        counts["states_with_distinct_topple"] += int(np.ptp(targets[:,2]) > 0)
        counts["states_with_dr_mean_range_gt_001"] += int(np.ptp(targets[:,0]) > .01)
        counts["states_with_dr_mean_range_gt_005"] += int(np.ptp(targets[:,0]) > .05)
        counts["physical_action_steps"] += candidates * (int(row["source_t"])+max(consequence_horizons))
        visited.add(row["scene_id"])

    def save_summary():
        actual_count = len(completed)
        summary = {
            "schema_version": 1, "source_dataset": str(dataset),
            "source_split_sha256": split["dataset_sha256"],
            "source_split": "train", "requested_states": states,
            "num_states": actual_count, "complete": actual_count == states,
            "num_candidates_per_state": candidates,
            "physical_rollouts": actual_count * candidates,
            "distinct_source_scenes": len(visited),
            "strata": dict(counts),
            "topple_candidates": counts["topple_candidates"],
            "states_with_distinct_topple": counts["states_with_distinct_topple"],
            "states_with_dr_mean_range_gt_001": counts["states_with_dr_mean_range_gt_001"],
            "states_with_dr_mean_range_gt_005": counts["states_with_dr_mean_range_gt_005"],
            "skipped_probes": len(failed),
            "replayed_physics_action_steps": counts["physical_action_steps"],
            "seed": seed, "sigma": sigma, "candidate_mode": candidate_mode,
            "sigma_levels": list(sigma_levels),
            "obs_horizon": 2, "action_horizon": 8,
            "label_names": ["dr_mean", "dr_max", "any_topple", "any_fall"],
            "labels_from_simulation": True, "trained_model": False,
            "consequence_horizons": list(consequence_horizons),
            "continuation_policy": "hold_last_joint_target_after_step_8",
            "probe_strategy": probe_strategy,
            "requested_strata": desired_strata,
            "stratum_shortfall": ({k: max(0, v-counts[k]) for k,v in desired_strata.items()}
                                  if desired_strata is not None else None),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        return summary

    save_summary()
    for probe in probes:
        if len(completed) >= states:
            break
        key = (probe.trajectory_id, probe.step)
        if key in completed or key in failed_keys:
            continue
        if desired_strata is not None and counts[probe.stratum] >= desired_strata[probe.stratum]:
            continue
        index = len(completed)
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
        if candidate_mode == "multiscale":
            actions_to_compare, candidate_sigmas = make_multiscale_candidates(
                expert, low, high, count=candidates,
                seed=stable_seed(seed, probe.trajectory_id, t),
                sigma_levels=sigma_levels,
                freeze_last_dimension=not perturb_gripper)
        elif candidate_mode == "legacy":
            actions_to_compare = make_candidates(
                expert, low, high, count=candidates,
                seed=stable_seed(seed, probe.trajectory_id, t), sigma=sigma,
                freeze_last_dimension=not perturb_gripper)
            candidate_sigmas = np.r_[0., np.repeat(sigma, candidates - 1)].astype(np.float32)
        else:
            raise ValueError(f"Unknown candidate mode: {candidate_mode}")
        try:
            actual = [_simulate(
                scene, snapshot, prefix, actions_to_compare[0],
                source_pose=truth["distractor_poses"][t],
                source_obs=obs_window, state_atol=state_atol,
                obs_atol=obs_atol, allow_fallback=allow_fallback,
                consequence_horizons=consequence_horizons)]
            for metric in ("dr_mean", "dr_max"):
                delta = abs(actual[0]["label"][metric] - archived[metric])
                if delta > dr_atol:
                    raise ExpertReplayMismatch(
                        f"Expert source/replay {metric} mismatch {delta} > {dr_atol}: "
                        f"{probe.trajectory_id}:{t}")
            for event in ("topple_count", "fall_count"):
                if actual[0]["label"][event] != archived[event]:
                    raise ExpertReplayMismatch(
                        f"Expert source/replay {event} mismatch: {probe.trajectory_id}:{t}")
            for alternate in actions_to_compare[1:]:
                actual.append(_simulate(
                    scene, snapshot, prefix, alternate,
                    expected_signature=actual[0]["signature"],
                    state_atol=state_atol, allow_fallback=allow_fallback,
                    consequence_horizons=consequence_horizons))
        except ExpertReplayMismatch as exc:
            row = {"scene_id": probe.scene_id, "source_trajectory_id": probe.trajectory_id,
                   "source_t": t, "stratum": probe.stratum,
                   "reason": str(exc), "details": exc.details}
            with failed_manifest.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            failed.append(row)
            failed_keys.add(key)
            save_summary()
            print(f"SKIP unrepeatable expert: {probe.trajectory_id} t={t}: {exc}", flush=True)
            continue
        targets = np.asarray([
            [r["label"]["dr_mean"], r["label"]["dr_max"],
             float(r["label"]["any_topple"]), float(r["label"]["any_fall"])]
            for r in actual], dtype=np.float32)
        targets_multi = np.asarray([
            [[label["dr_mean"], label["dr_max"],
              float(label["any_topple"]), float(label["any_fall"])]
             for label in r["multi_labels"]]
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
            targets_multi=targets_multi,
            consequence_horizons=np.asarray(consequence_horizons, dtype=np.int32),
            per_object_dr=obj_dr, per_object_topple=obj_topple,
            per_object_fall=obj_fall, object_sizes_m=truth["object_sizes_m"],
            nominal_sigmas=candidate_sigmas)
        meta = {"state_id": state_id, "scene_id": probe.scene_id,
                "source_trajectory_id": probe.trajectory_id, "source_t": t,
                "source_split": "train", "stratum": probe.stratum,
                "file": rel, "object_ids": source_row["object_ids"],
                "num_candidates": candidates,
                "perturbation_sigma": sigma,
                "candidate_mode": candidate_mode,
                "nominal_sigmas": candidate_sigmas.tolist(),
                "actual_action_l2_from_expert": np.linalg.norm(
                    (actions_to_compare - expert[None]).reshape(candidates, -1),
                    axis=1).tolist(),
                "dr_mean_range": float(np.ptp(targets[:, 0])),
                "dr_max_range": float(np.ptp(targets[:, 1])),
                "topple_label_distinct": bool(len(np.unique(targets[:, 2])) > 1),
                "reference_obs_replay_max_abs": actual[0]["source_obs_max_abs"],
                "expert_source_dr_mean": archived["dr_mean"],
                "expert_replayed_dr_mean": float(targets[0,0]),
                "physical_relabeling": "fresh_initial_scene_full_expert_prefix_each_candidate",
                "consequence_horizons": list(consequence_horizons),
                "continuation_policy": "hold_last_joint_target_after_step_8"}
        with manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(json_safe(meta), ensure_ascii=False) + "\n")
        counts[probe.stratum] += 1
        counts["topple_candidates"] += int((targets[:,2] > 0).sum())
        counts["states_with_distinct_topple"] += int(len(set(targets[:,2])) > 1)
        counts["states_with_dr_mean_range_gt_001"] += int(np.ptp(targets[:, 0]) > .01)
        counts["states_with_dr_mean_range_gt_005"] += int(np.ptp(targets[:, 0]) > .05)
        counts["physical_action_steps"] += candidates * (t+max(consequence_horizons))
        visited.add(probe.scene_id)
        completed.add(key)
        save_summary()
        print(f"[{len(completed)}/{states}] {probe.trajectory_id} t={t} "
              f"{probe.stratum} topples={targets[:,2].astype(int).tolist()}",
              flush=True)
    summary = save_summary()
    if not summary["complete"]:
        print(f"INCOMPLETE: collected {len(completed)}/{states} valid states; "
              f"skipped {len(failed)} unrepeatable expert probes. "
              "See skipped_probes.jsonl.", flush=True)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--states",type=int,default=2)
    p.add_argument("--candidates",type=int,default=4)
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--sigma",type=float,default=0.02)
    p.add_argument("--candidate-mode",choices=("legacy","multiscale"),default="legacy")
    p.add_argument("--sigma-levels",default="0.02,0.06,0.12",
                   help="Comma-separated action perturbation scales in multiscale mode")
    p.add_argument("--max-per-scene",type=int,default=4)
    p.add_argument("--probe-strategy",choices=("legacy", "balanced"),default="legacy",
                   help="Balanced uses five strata and same-stratum reserves")
    p.add_argument("--max-prefix-steps",type=int,default=None)
    p.add_argument("--state-atol",type=float,default=0.005)
    p.add_argument("--obs-atol",type=float,default=0.005)
    p.add_argument("--dr-atol",type=float,default=0.02)
    p.add_argument("--allow-scene-spec-fallback",action="store_true")
    p.add_argument("--perturb-gripper",action="store_true")
    p.add_argument("--consequence-horizons",default="8",
                   help="Physical horizons e.g. 4,8,16,32; H>8 holds last target")
    p.add_argument("--resume",action="store_true",
                   help="Continue an interrupted run from verified saved states")
    p.add_argument("--reserve-probes",type=int,default=16,
                   help="Additional deterministic training probes if some expert replays mismatch")
    args=p.parse_args(argv)
    try:
        levels=tuple(float(x.strip()) for x in args.sigma_levels.split(","))
    except ValueError as exc:
        p.error(f"Invalid --sigma-levels: {exc}")
    report=collect(
        args.dataset,args.output_dir,states=args.states,
        candidates=args.candidates,seed=args.seed,sigma=args.sigma,
        candidate_mode=args.candidate_mode,sigma_levels=levels,
        max_per_scene=args.max_per_scene,max_prefix_steps=args.max_prefix_steps,
        state_atol=args.state_atol,obs_atol=args.obs_atol,dr_atol=args.dr_atol,
        allow_fallback=args.allow_scene_spec_fallback,
        perturb_gripper=args.perturb_gripper,
        resume=args.resume, reserve_probes=args.reserve_probes,
        probe_strategy=args.probe_strategy,
        consequence_horizons=tuple(int(x) for x in args.consequence_horizons.split(",")))
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return report


if __name__=="__main__":
    main()
