"""Evaluate a Vanilla DP checkpoint with the shared CANDID rollout."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from candid.eval.metrics import SafetyLimits
from candid.eval.rollout import evaluate_dataset
from model.vanilla_dp.checkpoint import load_policy


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch cannot see a GPU")
    if requested not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    return requested


def _describe_device(device: str) -> str:
    if device == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return "cpu"


def evaluate_vanilla_dp(
    dataset,
    checkpoint,
    output_dir,
    limits: SafetyLimits,
    *,
    device: str = "auto",
    **rollout_kwargs,
) -> dict:
    device = resolve_device(device)
    print(f"eval device: {_describe_device(device)}", flush=True)
    policy = load_policy(checkpoint, device=device)
    return evaluate_dataset(
        dataset,
        policy,
        output_dir,
        limits,
        checkpoint=checkpoint,
        **rollout_kwargs,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate trained Vanilla DP on frozen TEST scenes")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--safe-dr-mean-max", type=float, required=True)
    parser.add_argument("--safe-dr-mean-min", type=float, default=0.0)
    parser.add_argument("--safe-dr-max-min", type=float, default=0.0)
    parser.add_argument("--safe-dr-max-max", type=float, default=None)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--episodes-per-scene", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--exec-horizon", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-stop-on-success", action="store_true")
    parser.add_argument("--scene-split", default="test", choices=("train", "validation", "test"))
    args = parser.parse_args(argv)
    limits = SafetyLimits(
        args.safe_dr_mean_max,
        args.safe_dr_max_max,
        args.safe_dr_mean_min,
        args.safe_dr_max_min,
    )
    result = evaluate_vanilla_dp(
        args.dataset,
        args.checkpoint,
        args.output_dir,
        limits,
        device=args.device,
        max_scenes=args.max_scenes,
        scene_id=args.scene_id,
        episodes_per_scene=args.episodes_per_scene,
        max_steps=args.max_steps,
        exec_horizon=args.exec_horizon,
        eval_seed=args.eval_seed,
        save_video=not args.no_video,
        stop_on_success=not args.no_stop_on_success,
        scene_split=args.scene_split,
    )
    print("CANDID Vanilla DP evaluation")
    for key in (
        "episodes",
        "scenes",
        "success_rate",
        "success_once_rate",
        "safety_rate",
        "safe_success_rate",
        "mean_dr_mean",
        "mean_dr_max",
        "total_topple",
        "total_fall",
    ):
        print(f"{key}: {result[key]}")
    print(f"Results: {args.output_dir / 'summary.json'}")
    return result


if __name__ == "__main__":
    main()
