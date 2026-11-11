"""Run exactly one Scene1 comparison: original v2 (64 CF) versus v2 (574 CF).

Train v2 ordinal predictor with the original v2 recorded settings, then evaluate
using the identical frozen Scene1 test suite and SNR-capped Guidance config.
No DP training, no dataset mutation, and no new physics labels.

Run from repo root: CUDA_VISIBLE_DEVICES=4 python -u scripts/run_scene1_v2_cf574_once.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

OLD_TRAIN = Path("runs/predictor_ordinal_joint_regularized_v2/summary.json")
OLD_EVAL = Path("runs/scene1_guidance_snr_41_snr_gaussian_capped/summary.json")
CF_ROOT = Path("runs/cf_scene1_balanced_600")
TRAIN_OUT = Path("runs/predictor_scene1_v2_cf574_once")
EVAL_OUT = Path("runs/scene1_v2_cf574_snr_capped_41")
OLD_CHECKPOINT = "runs/predictor_ordinal_joint_regularized_v2/checkpoints/best_val.pt"


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required original experiment file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def append(cmd, flag, value):
    if value is not None:
        cmd.extend((flag, str(value)))


def old_training_command(old):
    """Reconstruct the original v2 settings, not a new hyperparameter sweep."""
    cfg = old["config"]
    rank = old["ranking"]
    weights = old["weights"]
    controls = old["training_controls"]
    optimizer = old["optimizer"]
    if old.get("init_checkpoint"):
        raise ValueError("Original v2 used pretrained initialization; cannot make a "
                         "leakage-free from-scratch comparison without reviewing it")
    if int(old["counterfactual_states"]) != 64:
        raise ValueError("Expected original v2 trained on exactly 64 CF states")
    if int(cfg["obs_horizon"]) != 2 or int(cfg["action_horizon"]) != 8:
        raise ValueError("Original v2 observation/action horizon mismatch")
    if cfg["dr_objective"] != "ordinal":
        raise ValueError("Original v2 was not ordinal")
    command = [sys.executable, "-u", "scripts/train_counterfactual_predictor.py",
               "--dataset", "demos/scene1",
               "--counterfactual-dataset", str(CF_ROOT),
               "--allow-incomplete-counterfactual",
               "--output-dir", str(TRAIN_OUT),
               "--device", "cuda"]
    settings = {
        "--steps": old.get("steps_completed", cfg["steps"]),
        "--batch-size": cfg["batch_size"],
        "--hidden": cfg["hidden"],
        "--lr": cfg["lr"],
        "--seed": cfg["seed"],
        "--split-seed": cfg["split_seed"],
        "--val-ratio": cfg["val_ratio"],
        "--test-ratio": cfg["test_ratio"],
        "--log-freq": cfg["log_freq"],
        "--eval-freq": cfg["eval_freq"],
        "--val-max-windows": cfg["val_max_windows"],
        "--val-batch-size": cfg["val_batch_size"],
        "--dr-thresholds": ",".join(str(t) for t in cfg["dr_thresholds"]),
        "--ordinal-pos-weight": cfg["ordinal_pos_weight"],
        "--expert-weight": weights["expert"],
        "--cf-weight": weights["counterfactual"],
        "--rank-weight": weights["ranking"],
        "--rank-temperature": rank["temperature"],
        "--topple-weight": rank["topple_weight"],
        "--fall-weight": rank["fall_weight"],
        "--dr-max-weight": rank["dr_max_weight"],
        "--min-risk-gap": rank["min_risk_gap"],
        "--weight-decay": optimizer["weight_decay"],
        "--cf-update-every": controls["cf_update_every"],
        "--cf-pos-weight-cap": controls["cf_pos_weight_cap"],
        "--early-stop-patience": controls["early_stop_patience"],
    }
    for flag, value in settings.items():
        append(command, flag, value)
    # These three training parameters were not persisted in the original v2
    # summary. Use that trainer's original defaults, and disclose this.
    print("NOTE: original summary did not record cf-batch-size (8), "
          "cf-val-ratio (0.2), expert-trajectories-per-batch (4); "
          "using original CLI defaults for these.", flush=True)
    return command


def old_evaluation_command(original):
    from candid.eval.scene_suite import load_suite

    guidance = original["guidance"]
    if guidance["strength_schedule"] != "snr_gaussian_capped":
        raise ValueError("Original evaluation was not SNR-capped")
    if Path(guidance["predictor_checkpoint"]).resolve() != Path(OLD_CHECKPOINT).resolve():
        raise ValueError("Original Guidance result does not use the expected 64-CF v2")
    suite = original.get("eval_suite")
    sha = original.get("eval_suite_sha256")
    if not suite or not sha or len(original.get("test_scene_ids", [])) != 41:
        raise ValueError("Original report has no verified 41-scene frozen suite")
    manifest, _ = load_suite(suite)
    if manifest["suite_content_sha256"] != sha:
        raise ValueError("Frozen evaluation suite content changed")
    if original["episodes_per_scene"] != 1:
        raise ValueError("Expected one episode per scene")
    if not Path(guidance["dp_checkpoint"]).is_file():
        raise FileNotFoundError(guidance["dp_checkpoint"])
    limits = original["safety_limits"]
    command = [
        sys.executable, "-u", "scripts/eval_dp_guidance.py",
        "--eval-suite", suite,
        "--dp-checkpoint", guidance["dp_checkpoint"],
        "--predictor-checkpoint", str(TRAIN_OUT / "checkpoints/best_val.pt"),
        "--output-dir", str(EVAL_OUT),
        "--max-scenes", "41", "--max-steps", str(original["max_steps"]),
        "--exec-horizon", str(original["exec_horizon"]),
        "--eval-seed", str(original["eval_seed"]),
        "--episodes-per-scene", "1",
        "--device", "cuda", "--no-video",
    ]
    for flag, key in [
        ("--strength", "strength"),
        ("--start-timestep", "start_timestep"),
        ("--guidance-parity", "guidance_parity"),
        ("--strength-schedule", "strength_schedule"),
        ("--snr-center-timestep", "snr_center_timestep"),
        ("--snr-sigma", "snr_sigma"),
        ("--dr-mean-weight", "dr_mean_weight"),
        ("--dr-max-weight", "dr_max_weight"),
        ("--topple-weight", "topple_weight"),
        ("--max-grad-norm", "max_grad_norm"),
        ("--max-step-norm", "max_step_norm"),
    ]:
        append(command, flag, guidance[key])
    append(command, "--fall-weight", 0.0)  # Existing Guidance summary omits fall_weight; original default was 0.
    append(command, "--safe-dr-mean-max", limits["dr_mean_max"])
    if limits["dr_mean_min"]:
        append(command, "--safe-dr-mean-min", limits["dr_mean_min"])
    if limits["dr_max_max"] is not None:
        append(command, "--safe-dr-max-max", limits["dr_max_max"])
        if limits["dr_max_min"]:
            append(command, "--safe-dr-max-min", limits["dr_max_min"])
    if not original.get("stop_on_success", True):
        command.append("--no-stop-on-success")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "train", "eval"), default="all",
                        help="all runs the single training then the single evaluation")
    args = parser.parse_args()
    if not Path("scripts/train_counterfactual_predictor.py").is_file():
        parser.error("Run from the CANDID repository root")
    old = read_json(OLD_TRAIN)
    original = read_json(OLD_EVAL)
    cf = read_json(CF_ROOT / "summary.json")
    if (cf.get("num_states") != 574 or cf.get("requested_states") != 600
            or cf.get("labels_from_simulation") is not True
            or cf.get("source_split") != "train"):
        parser.error("Expected verified Scene1 CF collection: 574 of 600 real states")
    train_cmd = old_training_command(old)
    eval_cmd = old_evaluation_command(original)
    if args.stage in ("all", "train"):
        if TRAIN_OUT.exists() and any(TRAIN_OUT.iterdir()):
            parser.error(f"Output already used: {TRAIN_OUT}")
        print("=== Training v2 with 574 physical CF states ===", flush=True)
        print(" ".join(train_cmd), flush=True)
        subprocess.run(train_cmd, check=True)
    if args.stage in ("all", "eval"):
        if EVAL_OUT.exists() and any(EVAL_OUT.iterdir()):
            parser.error(f"Output already used: {EVAL_OUT}")
        checkpoint = TRAIN_OUT / "checkpoints/best_val.pt"
        if not checkpoint.is_file():
            parser.error(f"Missing trained best checkpoint: {checkpoint}")
        print("=== Running paired 41-scene SNR-Capped evaluation ===", flush=True)
        print(" ".join(eval_cmd), flush=True)
        subprocess.run(eval_cmd, check=True)
        new = read_json(EVAL_OUT / "summary.json")
        print("\n=== Original v2 (64 CF) vs v2 (574 CF) ===", flush=True)
        for key in ("scenes", "success_rate", "safety_rate", "safe_success_rate",
                    "mean_dr_mean", "mean_dr_max", "total_topple", "total_fall"):
            print(f"{key:20s} {original[key]!s:>16} {new[key]!s:>16}", flush=True)
        print(f"Report: {EVAL_OUT / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
