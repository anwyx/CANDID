"""Run four v3 predictor ablations. Use the same frozen SNR-Capped control."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

EXPERIMENTS = ("h8_only", "absolute", "no_ranking", "no_difference")
SWITCHES = {
    "h8_only": ("--horizon-mode", "h8_only"),
    "absolute": ("--action-representation", "absolute"),
    "no_ranking": ("--rank-weight", "0"),
    "no_difference": ("--diff-weight", "0"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "eval"))
    parser.add_argument("--only", choices=EXPERIMENTS)
    args = parser.parse_args()
    root = Path.cwd()
    if not (root / "scripts/train_predictor_multihorizon.py").is_file():
        parser.error("Run from the CANDID repository root")
    names = (args.only,) if args.only else EXPERIMENTS

    baseline = None
    if args.phase == "eval":
        path = root / "runs/scene1_v3_snr_capped_41/summary.json"
        if not path.is_file():
            parser.error(f"Missing Full v3 baseline summary: {path}")
        baseline = json.loads(path.read_text(encoding="utf-8"))
        source = baseline.get("guidance") or {}
        suite = baseline.get("eval_suite")
        from candid.eval.scene_suite import load_suite
        if not suite or not baseline.get("eval_suite_sha256"):
            parser.error("Full v3 does not identify a frozen evaluation suite")
        suite_info, _ = load_suite(suite)
        if suite_info["suite_content_sha256"] != baseline["eval_suite_sha256"]:
            parser.error("Full v3 evaluation suite fingerprint changed")
        if len(baseline.get("test_scene_ids", [])) != 41:
            parser.error("Full v3 baseline does not have exactly 41 test scenes")
        dp = source.get("dp_checkpoint")
        if not dp or not Path(dp).is_file():
            parser.error(f"Missing Full v3 DP checkpoint: {dp}")
        if source.get("strength_schedule") != "snr_gaussian_capped":
            parser.error("Full v3 baseline was not SNR Capped Guidance")

    for name in names:
        model_dir = Path(f"runs/predictor_v3_ablation_{name}")
        output = (model_dir if args.phase == "train"
                  else Path(f"runs/scene1_v3_ablation_{name}_snr41"))
        if output.exists():
            parser.error(f"Output exists (will not overwrite): {output}")
        if args.phase == "train":
            cmd = [
                sys.executable, "-u", "scripts/train_predictor_multihorizon.py",
                "--dataset", "demos/scene1",
                "--counterfactual-dataset", "runs/cf_scene1_balanced_600",
                "--allow-incomplete-counterfactual",
                "--output-dir", str(output),
                "--steps", "5000", "--batch-size", "64",
                "--cf-batch-size", "8", "--cf-val-ratio", "0.2",
                "--eval-freq", "100", "--log-freq", "50",
                "--seed", "2026", "--device", "cuda",
                *SWITCHES[name],
            ]
        else:
            ckpt = model_dir / "checkpoints/best_val.pt"
            if not ckpt.is_file():
                parser.error(f"Missing ablation checkpoint: {ckpt}")
            cfg = baseline["guidance"]
            limits = baseline["safety_limits"]
            cmd = [
                sys.executable, "-u", "scripts/eval_dp_guidance.py",
                "--eval-suite", baseline["eval_suite"],
                "--dp-checkpoint", cfg["dp_checkpoint"],
                "--predictor-checkpoint", str(ckpt),
                "--output-dir", str(output),
                "--strength", str(cfg["strength"]),
                "--start-timestep", str(cfg["start_timestep"]),
                "--guidance-parity", cfg["guidance_parity"],
                "--strength-schedule", cfg["strength_schedule"],
                "--snr-center-timestep", str(cfg["snr_center_timestep"]),
                "--snr-sigma", str(cfg["snr_sigma"]),
                "--dr-mean-weight", str(cfg["dr_mean_weight"]),
                "--dr-max-weight", str(cfg["dr_max_weight"]),
                "--topple-weight", str(cfg["topple_weight"]),
                "--fall-weight", str(cfg["fall_weight"]),
                "--max-grad-norm", str(cfg["max_grad_norm"]),
                "--max-step-norm", str(cfg["max_step_norm"]),
                "--safe-dr-mean-max", str(limits["dr_mean_max"]),
                "--max-scenes", "41",
                "--max-steps", str(baseline["max_steps"]),
                "--exec-horizon", str(baseline["exec_horizon"]),
                "--eval-seed", str(baseline["eval_seed"]),
                "--episodes-per-scene", str(baseline["episodes_per_scene"]),
                "--device", "cuda", "--no-video",
            ]
            if not baseline.get("stop_on_success", True):
                cmd.append("--no-stop-on-success")
            if limits["dr_mean_min"]:
                cmd.extend(("--safe-dr-mean-min", str(limits["dr_mean_min"])))
            if limits["dr_max_max"] is not None:
                cmd.extend(("--safe-dr-max-max", str(limits["dr_max_max"])))
                if limits["dr_max_min"]:
                    cmd.extend(("--safe-dr-max-min", str(limits["dr_max_min"])))
        print(f"==== {args.phase} / {name} ====", flush=True)
        print(" ".join(map(str, cmd)), flush=True)
        subprocess.run(cmd, check=True)
    print("All requested experiments complete.", flush=True)


if __name__ == "__main__":
    main()
