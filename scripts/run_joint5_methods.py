"""Train one five-environment CANDID Predictor v2; evaluate Guidance/Best-of-N.

The DP and four existing baselines are NEVER retrained. For training the
joint v2 Predictor, read the proven original v2/64-CF training metadata and
reuse all recorded learning settings. For inference use the same frozen joint
DP checkpoint and combined evaluation suite for both comparisons.

Examples:
  python scripts/run_joint5_methods.py --stage train --dataset demos/combined5 \\
    --cf-dataset runs/joint5_cf64_each
  python scripts/run_joint5_methods.py --stage candid \\
    --dp-checkpoint ckpt/joint5/best_val.pt --eval-suite eval/combined250
  python scripts/run_joint5_methods.py --stage bestofn \\
    --dp-checkpoint ckpt/joint5/best_val.pt --eval-suite eval/combined250
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ORIGINAL = Path("runs/predictor_ordinal_joint_regularized_v2/summary.json")


def run(command):
    print("RUN:", " ".join(str(x) for x in command), flush=True)
    subprocess.run([str(x) for x in command], check=True)


def original_v2_command(meta, args):
    cfg = meta["config"]
    w = meta["weights"]
    rank = meta["ranking"]
    controls = meta["training_controls"]
    if meta["counterfactual_states"] != 64:
        raise ValueError("Original reference must be the known v2/64-state experiment")
    if meta.get("init_checkpoint"):
        raise ValueError("Original reference used pretrained init; refusing unsafe reuse")
    if cfg["dr_objective"] != "ordinal" or cfg["action_horizon"] != 8 or cfg["obs_horizon"] != 2:
        raise ValueError("Original reference is not v2 Ordinal H8")
    cmd = [
        sys.executable, "-u", "scripts/train_counterfactual_predictor.py",
        "--dataset", args.dataset,
        "--counterfactual-dataset", args.cf_dataset,
        "--output-dir", args.output_root / "predictor",
        "--device", "cuda",
    ]
    params = {
        "--steps": args.steps,
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
        "--dr-thresholds": ",".join(str(v) for v in cfg["dr_thresholds"]),
        "--ordinal-pos-weight": cfg["ordinal_pos_weight"],
        "--expert-weight": w["expert"],
        "--cf-weight": w["counterfactual"],
        "--rank-weight": w["ranking"],
        "--rank-temperature": rank["temperature"],
        "--topple-weight": rank["topple_weight"],
        "--fall-weight": rank["fall_weight"],
        "--dr-max-weight": rank["dr_max_weight"],
        "--min-risk-gap": rank["min_risk_gap"],
        "--weight-decay": meta["optimizer"]["weight_decay"],
        "--cf-update-every": controls["cf_update_every"],
        "--cf-pos-weight-cap": controls["cf_pos_weight_cap"],
        "--early-stop-patience": controls["early_stop_patience"],
    }
    for name, value in params.items():
        cmd.extend([name, str(value)])
    # Defaults not persisted by the original trainer; set explicitly.
    cmd.extend(["--cf-batch-size", "8", "--cf-val-ratio", "0.2",
                "--expert-trajectories-per-batch", "4"])
    return cmd


def evaluation_command(args):
    if args.dp_checkpoint is None or args.eval_suite is None:
        raise ValueError("Evaluation requires --dp-checkpoint and --eval-suite")
    from candid.eval.scene_suite import load_suite
    suite_meta, _ = load_suite(args.eval_suite)
    if suite_meta["source"] != "multi_env_eval_suite":
        raise ValueError("Use the frozen five-environment merged evaluation suite")
    if len(suite_meta["components"]) != 5:
        raise ValueError("Expected five environment groups in evaluation suite")
    for file in (args.dp_checkpoint, args.predictor_checkpoint):
        if not file.is_file():
            raise FileNotFoundError(f"Required trained checkpoint missing: {file}")
    general = [
        "--eval-suite", str(args.eval_suite),
        "--dp-checkpoint", str(args.dp_checkpoint),
        "--predictor-checkpoint", str(args.predictor_checkpoint),
        "--safe-dr-mean-max", str(args.safe_dr_mean_max),
        "--max-steps", str(args.max_steps),
        "--exec-horizon", str(args.exec_horizon),
        "--eval-seed", str(args.eval_seed),
        "--device", "cuda", "--no-video",
        "--dr-mean-weight", "1.0", "--dr-max-weight", "0.25",
        "--topple-weight", "0.5", "--fall-weight", "0",
    ]
    if args.stage == "candid":
        output = args.output_root / "candid_snr_capped"
        cmd = [sys.executable, "-u", "scripts/eval_dp_guidance.py",
               *general, "--output-dir", str(output),
               "--strength", "0.02",
               "--start-timestep", "30",
               "--guidance-parity", "odd",
               "--strength-schedule", "snr_gaussian_capped",
               "--snr-center-timestep", "15", "--snr-sigma", "1.0",
               "--max-grad-norm", "1.0", "--max-step-norm", "0.02"]
    elif args.stage == "bestofn":
        output = args.output_root / f"bestof{args.num_candidates}"
        # Pure min-predicted-risk selection among N unconditional DP samples:
        # remove progress-preserving gates to avoid calling gated rerank "Best-of-N".
        cmd = [sys.executable, "-u", "scripts/eval_dp_predictor.py",
               *general, "--output-dir", str(output),
               "--mode", "rerank",
               "--num-candidates", str(args.num_candidates),
               "--min-improvement", "0",
               "--max-joint-deviation", "1000000"]
    else:
        raise ValueError(f"Unsupported eval stage: {args.stage}")
    if output.exists():
        raise FileExistsError(f"Do not overwrite existing evaluation: {output}")
    return cmd, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("train", "candid", "bestofn"))
    parser.add_argument("--dataset", type=Path, help="Frozen merged joint EXPERT dataset")
    parser.add_argument("--cf-dataset", type=Path, help="Prepared five-source joint CF subset")
    parser.add_argument("--dp-checkpoint", type=Path, help="Already trained joint DP checkpoint")
    parser.add_argument("--eval-suite", type=Path, help="Frozen five-environment evaluation suite")
    parser.add_argument("--predictor-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs/joint5_final"))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--exec-horizon", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--safe-dr-mean-max", type=float, default=0.2)
    parser.add_argument("--eval-seed", type=int, default=2026)
    args = parser.parse_args()
    if args.exec_horizon < 1 or args.num_candidates < 2 or args.steps < 1:
        parser.error("Invalid execution horizon / N / steps")
    args.predictor_checkpoint = (
        args.predictor_checkpoint
        if args.predictor_checkpoint is not None
        else args.output_root / "predictor/checkpoints/best_val.pt")
    if args.stage == "train":
        if not args.dataset or not args.cf_dataset:
            parser.error("Training needs --dataset and --cf-dataset")
        if not (args.dataset / "split.json").is_file():
            parser.error("Merged expert dataset must retain the DP frozen split.json")
        if (args.output_root / "predictor").exists():
            parser.error("Predictor output already exists; refusing to overwrite")
        cf = json.loads((args.cf_dataset / "summary.json").read_text(encoding="utf-8"))
        if not cf.get("derived_from_existing_physical_labels"):
            parser.error("Expected a provenance-checked joint physical CF dataset")
        if Path(cf["source_dataset"]).resolve() != args.dataset.resolve():
            parser.error("Joint CF was prepared from another expert dataset")
        old = json.loads(ORIGINAL.read_text(encoding="utf-8"))
        args.output_root.mkdir(parents=True, exist_ok=True)
        run(original_v2_command(old, args))
        print("Trained joint predictor:", args.predictor_checkpoint, flush=True)
    else:
        cmd, output = evaluation_command(args)
        run(cmd)
        groups = [sys.executable, "-u", "scripts/summarize_eval_groups.py",
                  "--eval-suite", str(args.eval_suite),
                  "--results", str(output),
                  "--output", str(output / "by_environment.json")]
        run(groups)
        report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        print("\nRESULT", args.stage, "scenes", report["scenes"],
              "success", report["success_rate"],
              "safe_success", report["safe_success_rate"],
              "dr_mean", report["mean_dr_mean"],
              "topple", report["total_topple"], flush=True)


if __name__ == "__main__":
    main()
