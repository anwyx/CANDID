"""Paired fixed-state physical DR checks for Vanilla DP versus Guidance.

Follow a Vanilla DP *reference* rollout. At selected replanning boundaries,
capture the complete ManiSkill state and observation history. For that same
live state and identical Torch sampling RNG, sample a Vanilla and a Guided
8-step chunk. Replay both chunks separately from the captured state, and
calculate actual physics DR over prefixes of length 4 and 8.

No counterfactual action is ever applied to the reference rollout. This
experiment checks local disturbance; it is NOT an end-to-end success eval.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from candid.data.metrics import disturbance_from_poses
from candid.data.splits import validate_scene_split
from candid.data.storage import freeze_state, load_state_npz
from candid.data.validate import validate_dataset
from candid.eval.replay_env import resolve_scene_environment
from candid.eval.local_dr_repeat import compare_repeat
from candid.eval.rollout import (
    _canonical_live, _check_initial_scene, _clamp_action, _is_true, _read_jsonl,
)
from model.guidance.config import GuidanceConfig
from model.guidance.objective import risk_from_predictions
from model.guidance.policy import ConsequenceGuidedPolicy
from model.predictor.checkpoint import load_predictor
from model.vanilla_dp.checkpoint import load_policy


def parse_steps(csv):
    try:
        steps = sorted(set(int(s.strip()) for s in csv.split(",")))
    except ValueError as exc:
        raise ValueError("--probe-steps must be comma-separated integers") from exc
    if not steps or steps[0] < 0 or any(s % 4 for s in steps):
        raise ValueError("Probe steps must be nonnegative multiples of exec-horizon=4")
    return steps


def select_held_out(dataset, substrings):
    validate_dataset(dataset)
    split = json.loads((dataset / "split.json").read_text(encoding="utf-8"))
    validate_scene_split(split, dataset)
    test_ids = list(split["test"])
    if not substrings:
        raise ValueError("Supply --scene-substrings to select held-out scenes")
    requested = []
    for key in substrings:
        hits = [sid for sid in test_ids if key in sid]
        if len(hits) != 1:
            raise ValueError(f"{key!r} matched {len(hits)} test scenes, expected one")
        requested.append(hits[0])
    if len(set(requested)) != len(requested):
        raise ValueError("Scene selections must be unique")
    scene_map = {row["scene_id"]: row for row in _read_jsonl(dataset / "scenes.jsonl")}
    return [(test_ids.index(sid), scene_map[sid]) for sid in test_ids if sid in requested]


def cuda_rng_devices(agent):
    device = next(agent.parameters()).device
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def make_env(scene, max_steps, env_id):
    import gymnasium as gym
    from candid.data.monitor import PhysicsTrace

    return PhysicsTrace(gym.make(
        env_id, obs_mode="state_dict", control_mode="pd_joint_pos",
        render_mode=None, render_backend="none",
        robot_uids="xarm6_robotiq", reward_mode="dense",
        sim_backend="cpu", max_episode_steps=max_steps,
        **scene.get("env_kwargs", {}),
    ))


def _signature(trace):
    """Check essential state that matters for a local physical comparison."""
    from candid.data.monitor import as_numpy

    base = trace.unwrapped
    robot = base.agent.robot
    all_objects = [base.target_actor, *base.distractor_actors]
    return {
        "qpos": as_numpy(robot.get_qpos()).copy(),
        "qvel": as_numpy(robot.get_qvel()).copy(),
        "poses": np.stack(
            [as_numpy(actor.pose.raw_pose).reshape(-1, 7)[0] for actor in all_objects]
        ),
        "linear_velocities": np.stack([
            as_numpy(actor.get_linear_velocity()).reshape(-1, 3)[0]
            for actor in all_objects
        ]),
        "angular_velocities": np.stack([
            as_numpy(actor.get_angular_velocity()).reshape(-1, 3)[0]
            for actor in all_objects
        ]),
    }


def _check_signature(expected, actual, atol=1e-5):
    for key in expected:
        if expected[key].shape != actual[key].shape:
            raise RuntimeError(f"Restore changed {key} shape")
        error = float(np.max(np.abs(expected[key] - actual[key])))
        if error > atol:
            raise RuntimeError(f"Restore changed {key} (maximum absolute error={error:.7g})")


def _physical_outcomes(trace, scene, state, actions, expected):
    """Start a fresh branch and execute the same eight PD joint targets."""
    trace.reset(
        seed=int(scene["seed"]),
        options={"reset_to_env_states": {"env_states": freeze_state(state)}},
    )
    _check_signature(expected, _signature(trace))
    for command in actions:
        bounded, _ = _clamp_action(command, trace.action_space)
        _, _, terminated, truncated, _ = trace.step(bounded)
        if _is_true(terminated) or _is_true(truncated):
            break

    arrays = trace.arrays()
    steps = trace.steps
    results = {"executed_steps": steps}
    for horizon in (4, 8):
        label = f"h{horizon}"
        if steps < horizon:
            results[label] = None
            continue
        physical = disturbance_from_poses(
            arrays["distractor_poses"], arrays["object_sizes_m"],
            start=0, stop=horizon,
            topple_flags=arrays["distractor_toppled"],
            fall_flags=arrays["distractor_fallen"],
        )
        results[label] = {
            "dr_mean": physical["dr_mean"],
            "dr_max": physical["dr_max"],
            "topple_count": physical["topple_count"],
            "fall_count": physical["fall_count"],
        }
    return results, arrays["distractor_poses"]


def _same_noise_chunks(dp, guidance, window):
    """Both independent generations begin at the identical Torch RNG state.

    Each fork restores the outer CPU and active CUDA RNG state, so these
    comparisons do not alter the reference DP rollout's action sequence.
    """
    devices = cuda_rng_devices(dp.agent)
    with torch.random.fork_rng(devices=devices):
        vanilla = dp.act(window)
    with torch.random.fork_rng(devices=devices):
        guided = guidance.act(window)
    if vanilla.shape != guided.shape:
        raise RuntimeError("Vanilla and Guidance action shapes differ")
    return vanilla, guided


@torch.no_grad()
def _predict_pair(window, actions, predictor, normalizer, config):
    device = next(predictor.parameters()).device
    states = torch.as_tensor(window, dtype=torch.float32, device=device)
    states = states.unsqueeze(0).expand(2, -1, -1)
    commands = torch.as_tensor(actions, dtype=torch.float32, device=device)
    outputs = predictor(
        normalizer.normalize_obs(states),
        normalizer.normalize_action(commands),
    )
    dr = normalizer.denormalize_target(outputs["dr"])
    events = torch.sigmoid(outputs["event_logits"])
    risk = risk_from_predictions(outputs, normalizer, config)
    result = []
    for i in range(2):
        result.append({
            "risk": float(risk[i].item()),
            "dr_mean": float(dr[i, 0].item()),
            "dr_max": float(dr[i, 1].item()),
            "topple_score": float(events[i, 0].item()),
            "fall_score": float(events[i, 1].item()),
        })
    return result


def run_scene(dataset, scene, test_index, dp, guidance, predictor, normalizer,
              config, probes, max_steps, eval_seed, check_repeat, diagnostics_dir):
    env_id, source = resolve_scene_environment(scene)
    reference = make_env(scene, max_steps, env_id)
    branch = None
    rows = []
    try:
        raw, _ = reference.reset(
            seed=int(scene["seed"]),
            options={"reset_to_env_states": {
                "env_states": load_state_npz(dataset / scene["initial_state"], device="cpu")
            }},
        )
        _check_initial_scene(scene, reference.unwrapped, reference)
        branch = make_env(scene, max_steps, env_id)
        branch.reset(
            seed=int(scene["seed"]),
            options={"reset_to_env_states": {
                "env_states": load_state_npz(
                    dataset / scene["initial_state"], device="cpu"
                )
            }},
        )
        if int(np.prod(reference.action_space.shape)) != dp.action_dim:
            raise ValueError("Action dimension mismatch")
        guidance.set_action_space(reference.action_space)

        seed = eval_seed + test_index * 100000
        np.random.seed(seed % 2**32)
        dp.begin_episode(seed)
        history = deque(
            [ _canonical_live(raw, len(reference.names)).copy() ] * dp.obs_horizon,
            maxlen=dp.obs_horizon,
        )
        current_step = 0
        max_probe = max(probes)
        while current_step <= max_probe and current_step < max_steps:
            window = np.stack(history).astype(np.float32)
            if current_step in probes:
                if current_step + 8 > max_steps:
                    raise ValueError("Probe must allow 8 subsequent physical steps")
                state = freeze_state(reference.unwrapped.get_state_dict())
                signature = _signature(reference)
                vanilla, guided = _same_noise_chunks(dp, guidance, window)
                low = np.asarray(reference.action_space.low, dtype=np.float32).reshape(-1)
                high = np.asarray(reference.action_space.high, dtype=np.float32).reshape(-1)
                # Both predictor and the simulator see the action-space-clipped commands.
                candidates = np.stack([
                    np.clip(vanilla, low, high), np.clip(guided, low, high)
                ]).astype(np.float32)
                predicted = _predict_pair(
                    window, candidates, predictor, normalizer, config
                )
                physical_v, poses_v = _physical_outcomes(
                    branch, scene, state, candidates[0], signature
                )
                if check_repeat:
                    repeat, poses_repeat = _physical_outcomes(
                        branch, scene, state, candidates[0], signature
                    )
                    repeat_check = compare_repeat(
                        physical_v, repeat, poses_v, poses_repeat,
                        pose_atol=1e-5, dr_atol=1e-4,
                    )
                    if not repeat_check["passed"]:
                        failure_path = (
                            diagnostics_dir
                            / f"replay_mismatch_{test_index:03d}_step_{current_step}.json"
                        )
                        failure_path.write_text(
                            json.dumps({
                                "scene_id": scene["scene_id"],
                                "probe_step": current_step,
                                "check": repeat_check,
                                "first": physical_v,
                                "repeat": repeat,
                            }, indent=2, ensure_ascii=False) + "\\n",
                            encoding="utf-8",
                        )
                        raise RuntimeError(
                            "Baseline replay mismatch: "
                            f"max_pose_abs={repeat_check['max_pose_abs']:.8g}, "
                            f"max_dr_abs={repeat_check['max_dr_abs']:.8g}, "
                            f"events_equal={repeat_check['events_equal']}; "
                            f"details: {failure_path}. "
                            "Do not use --no-repeat-check for scientific comparison."
                        )
                    print(
                        f"Replay repeat PASS: scene={scene['scene_id']} step={current_step} "
                        f"max_pose_abs={repeat_check['max_pose_abs']:.3g} "
                        f"max_dr_abs={repeat_check['max_dr_abs']:.3g}",
                        flush=True,
                    )
                physical_g, _ = _physical_outcomes(
                    branch, scene, state, candidates[1], signature
                )
                deviations = np.abs(candidates[1] - candidates[0])
                row = {
                    "scene_id": scene["scene_id"],
                    "scene_index": test_index,
                    "reference_seed": seed,
                    "probe_step": current_step,
                    "environment_source": source,
                    "predicted": {"vanilla": predicted[0], "guidance": predicted[1]},
                    "pred_risk_delta": predicted[1]["risk"] - predicted[0]["risk"],
                    "action_abs_mean": float(deviations.mean()),
                    "action_abs_max": float(deviations.max()),
                    "real": {"vanilla": physical_v, "guidance": physical_g},
                }
                for horizon in (4, 8):
                    key = f"h{horizon}"
                    a, b = physical_v[key], physical_g[key]
                    row[f"delta_dr_mean_{key}"] = (
                        b["dr_mean"] - a["dr_mean"] if a is not None and b is not None
                        else None
                    )
                    row[f"delta_dr_max_{key}"] = (
                        b["dr_max"] - a["dr_max"] if a is not None and b is not None
                        else None
                    )
                    row[f"delta_topple_{key}"] = (
                        b["topple_count"] - a["topple_count"]
                        if a is not None and b is not None else None
                    )
                rows.append(row)
                print(
                    f"{scene['scene_id']} step={current_step}: "
                    f"pred_dRisk={row['pred_risk_delta']:+.6f} "
                    f"DR4={row['delta_dr_mean_h4']} DR8={row['delta_dr_mean_h8']}",
                    flush=True,
                )
            # Advance only Vanilla DP along the reference closed-loop path.
            chunk = dp.act(window)
            interrupted = False
            for action in chunk[:4]:
                cmd, _ = _clamp_action(action, reference.action_space)
                raw, _, terminated, truncated, info = reference.step(cmd)
                current_step += 1
                history.append(_canonical_live(raw, len(reference.names)).copy())
                if _is_true(terminated) or _is_true(truncated) or _is_true(
                    info.get("success", False)
                ):
                    interrupted = True
                    break
            if interrupted:
                break
        return rows
    finally:
        reference.close()
        if branch is not None:
            branch.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("demos/scene1"))
    parser.add_argument("--dp-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-substrings", required=True)
    parser.add_argument("--probe-steps", default="40,80,120,160")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--strength", type=float, default=0.005)
    parser.add_argument("--start-timestep", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--no-repeat-check", action="store_true",
                        help="Skip duplicated Vanilla replay determinism check")
    args = parser.parse_args(argv)
    probes = parse_steps(args.probe_steps)
    if max(probes) + 8 > args.max_steps:
        parser.error("Largest probe step plus 8 must not exceed --max-steps")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")

    folder = args.output_dir
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {folder}")
    selected = select_held_out(
        args.dataset, [s.strip() for s in args.scene_substrings.split(",") if s.strip()]
    )
    config = GuidanceConfig(
        strength=args.strength, start_timestep=args.start_timestep
    )
    dp = load_policy(args.dp_checkpoint, device=args.device)
    predictor, normalizer = load_predictor(
        args.predictor_checkpoint, device=args.device
    )
    guidance = ConsequenceGuidedPolicy(dp, predictor, normalizer, config)
    if dp.action_horizon != 8 or predictor.action_horizon != 8:
        raise ValueError("This controlled experiment requires both horizons == 8")
    if dp.obs_horizon != 2:
        raise ValueError("This scene1 experiment assumes obs_horizon == 2")
    folder.mkdir(parents=True, exist_ok=False)
    results = []
    output_path = folder / "pairs.jsonl"
    for index, scene in selected:
        rows = run_scene(
            args.dataset, scene, index, dp, guidance, predictor, normalizer,
            config, probes, args.max_steps, args.eval_seed,
            check_repeat=not args.no_repeat_check,
            diagnostics_dir=folder,
        )
        results.extend(rows)
        with output_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "scenes_requested": len(selected),
        "pairs_completed": len(results),
        "requested_probe_steps": probes,
        "strength": args.strength,
        "device": args.device,
        "reference_policy": "vanilla_dp_h4_closed_loop",
        "comparison": "same_live_state_same_diffusion_rng_separate_8step_physics_replays",
        "physical_dr_reference": "state_at_probe_step",
        "predictor_label_horizon": 8,
        "physical_prefix_horizons": [4, 8],
        "repeat_check": not args.no_repeat_check,
    }
    for h in (4, 8):
        values = [r[f"delta_dr_mean_h{h}"] for r in results
                  if r[f"delta_dr_mean_h{h}"] is not None]
        summary[f"mean_delta_dr_h{h}"] = (
            float(np.mean(values)) if values else None
        )
        summary[f"valid_pairs_h{h}"] = len(values)
    (folder / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()
