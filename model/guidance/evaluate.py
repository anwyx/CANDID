"""Held-out CANDID closed-loop evaluation of predictor-gradient diffusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from candid.eval.metrics import SafetyLimits
from candid.eval.rollout import evaluate_dataset
from model.guidance.config import GuidanceConfig
from model.guidance.policy import ConsequenceGuidedPolicy
from model.predictor.checkpoint import load_predictor
from model.vanilla_dp.checkpoint import load_policy


def evaluate_guidance(dataset, dp_checkpoint, predictor_checkpoint, output_dir, limits, *,
                      config=None, device="cpu", max_scenes=None, scene_id=None,
                      scene_substrings=None, diagnose=False, profile_timing=False,
                      episodes_per_scene=1, max_steps=300, exec_horizon=8,
                      eval_suite=None,
                      eval_seed=2026, save_video=False, stop_on_success=True):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Invalid or unavailable device: {device}")
    settings = config if config is not None else GuidanceConfig()
    dp = load_policy(dp_checkpoint, device=device)
    predictor, normalizer = load_predictor(predictor_checkpoint, device=device)
    policy = ConsequenceGuidedPolicy(
        dp, predictor, normalizer, settings,
        diagnose=diagnose, profile_timing=profile_timing,
    )
    report = evaluate_dataset(
        dataset, policy, output_dir, limits, checkpoint=dp_checkpoint,
        eval_suite=eval_suite,
        max_scenes=max_scenes, scene_id=scene_id,
        scene_substrings=scene_substrings,
        episodes_per_scene=episodes_per_scene, max_steps=max_steps,
        exec_horizon=exec_horizon, eval_seed=eval_seed,
        save_video=save_video, stop_on_success=stop_on_success,
    )
    report["guidance"] = {
        "mode": "gradient_guidance" if settings.strength else "vanilla_control",
        "dp_checkpoint": str(dp_checkpoint),
        "predictor_checkpoint": str(predictor_checkpoint),
        "strength": settings.strength,
        "start_timestep": settings.start_timestep,
        "guidance_parity": settings.guidance_parity,
        "dr_mean_weight": settings.dr_mean_weight,
        "dr_max_weight": settings.dr_max_weight,
        "topple_weight": settings.topple_weight,
        "fall_weight": settings.fall_weight,
        "max_grad_norm": settings.max_grad_norm,
        "max_step_norm": settings.max_step_norm,
        "scoring_horizon": policy.action_horizon,
        "execution_horizon": exec_horizon,
        "risk_definition": "predicted_local_chunk_risk_not_episode_safety",
        "paired_diagnostics": diagnose,
        "profile_timing": profile_timing,
    }
    output = Path(output_dir)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="CANDID gradient-guided DP closed-loop evaluation")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Existing dataset for fixed-test mode; optional with --eval-suite")
    parser.add_argument("--eval-suite", type=Path, default=None,
                        help="Frozen random suite created by generate_eval_scenes.py")
    parser.add_argument("--dp-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--safe-dr-mean-max", type=float, required=True)
    parser.add_argument("--safe-dr-mean-min", type=float, default=0.0)
    parser.add_argument("--safe-dr-max-min", type=float, default=0.0)
    parser.add_argument("--safe-dr-max-max", type=float, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--scene-substrings", default=None,
                        help="Comma-separated unique substrings of held-out scene IDs; keeps original eval seeds")
    parser.add_argument("--diagnose", action="store_true",
                        help="Same-noise Vanilla vs Guidance chunk risk and action deviation")
    parser.add_argument("--profile-timing", action="store_true",
                        help="Optional DP/gradient/DDPM timing; CUDA event GPU times")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--episodes-per-scene", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--exec-horizon", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-stop-on-success", action="store_true")
    parser.add_argument("--strength", type=float, default=0.005)
    parser.add_argument("--start-timestep", type=int, default=30)
    parser.add_argument("--guidance-parity", choices=("all", "odd", "even"), default="all",
                        help="Guide all eligible DDPM steps (default), or only odd/even timestep indices")
    parser.add_argument("--dr-mean-weight", type=float, default=1.0)
    parser.add_argument("--dr-max-weight", type=float, default=0.25)
    parser.add_argument("--topple-weight", type=float, default=0.1)
    parser.add_argument("--fall-weight", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-step-norm", type=float, default=0.02)
    args = parser.parse_args(argv)
    if args.dataset is None and args.eval_suite is None:
        parser.error("Either --dataset (fixed tests) or --eval-suite is required")
    if args.dataset is not None and args.eval_suite is not None:
        parser.error("--dataset and --eval-suite are mutually exclusive")
    settings = GuidanceConfig(
        strength=args.strength, start_timestep=args.start_timestep,
        guidance_parity=args.guidance_parity,
        dr_mean_weight=args.dr_mean_weight, dr_max_weight=args.dr_max_weight,
        topple_weight=args.topple_weight, fall_weight=args.fall_weight,
        max_grad_norm=args.max_grad_norm, max_step_norm=args.max_step_norm,
    )
    limits = SafetyLimits(
        args.safe_dr_mean_max, args.safe_dr_max_max,
        args.safe_dr_mean_min, args.safe_dr_max_min,
    )
    report = evaluate_guidance(
        args.dataset, args.dp_checkpoint, args.predictor_checkpoint,
        args.output_dir, limits, config=settings, device=args.device,
        eval_suite=args.eval_suite,
        max_scenes=args.max_scenes, scene_id=args.scene_id,
        scene_substrings=(
            [word.strip() for word in args.scene_substrings.split(",") if word.strip()]
            if args.scene_substrings is not None else None
        ),
        diagnose=args.diagnose, profile_timing=args.profile_timing,
        episodes_per_scene=args.episodes_per_scene,
        max_steps=args.max_steps, exec_horizon=args.exec_horizon,
        eval_seed=args.eval_seed, save_video=not args.no_video,
        stop_on_success=not args.no_stop_on_success,
    )
    for key in ("episodes", "scenes", "success_rate", "safety_rate",
                "safe_success_rate", "mean_dr_mean", "mean_dr_max",
                "total_topple", "total_fall"):
        print(f"{key}: {report[key]}")
    print(f"Results: {args.output_dir / 'summary.json'}")
    return report


if __name__ == "__main__":
    main()
