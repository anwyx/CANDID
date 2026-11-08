"""Closed-loop evaluation for a frozen DP checkpoint plus frozen Predictor.

The Vanilla DP and Predictor trainers and their checkpoint formats are untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from candid.eval.metrics import SafetyLimits
from candid.eval.rollout import evaluate_dataset
from model.integration.guided_policy import PredictorGuidedPolicy
from model.predictor.checkpoint import load_predictor
from model.vanilla_dp.checkpoint import load_policy


def evaluate_dp_predictor(
    dataset, dp_checkpoint, predictor_checkpoint, output_dir, limits, *,
    mode="shadow", num_candidates=1, device="cpu", max_scenes=None,
    scene_id=None, episodes_per_scene=1, max_steps=300, exec_horizon=8,
    eval_seed=2026, save_video=False, stop_on_success=True,
    dr_mean_weight=1.0, dr_max_weight=0.25, topple_weight=0.1,
    fall_weight=0.0, min_improvement=0.01, max_joint_deviation=0.15,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Invalid or unavailable evaluation device: {device}")
    dp = load_policy(dp_checkpoint, device=device)
    predictor, normalizer = load_predictor(predictor_checkpoint, device=device)
    policy = PredictorGuidedPolicy(
        dp, predictor, normalizer, mode=mode, num_candidates=num_candidates,
        dr_mean_weight=dr_mean_weight, dr_max_weight=dr_max_weight,
        topple_weight=topple_weight, fall_weight=fall_weight,
        min_improvement=min_improvement, max_joint_deviation=max_joint_deviation,
    )
    report = evaluate_dataset(
        dataset, policy, output_dir, limits, checkpoint=dp_checkpoint,
        max_scenes=max_scenes, scene_id=scene_id,
        episodes_per_scene=episodes_per_scene, max_steps=max_steps,
        exec_horizon=exec_horizon, eval_seed=eval_seed,
        save_video=save_video, stop_on_success=stop_on_success,
    )
    # Keep the original episode-level metrics unchanged.
    report["guided_policy"] = {
        "mode": mode, "num_candidates": num_candidates,
        "predictor_checkpoint": str(predictor_checkpoint),
        "dr_mean_weight": dr_mean_weight, "dr_max_weight": dr_max_weight,
        "topple_weight": topple_weight, "fall_weight": fall_weight,
        "min_improvement": min_improvement, "max_joint_deviation": max_joint_deviation,
        "scoring_horizon": policy.action_horizon,
        "execution_horizon": exec_horizon,
        "risk_definition": "local_chunk_score_not_episode_safety",
    }
    report["candidate_changed_count"] = sum(
        int(row.get("candidate_changed_count", 0))
        for row in _read_episode_results(Path(output_dir))
    )
    (Path(output_dir) / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def _read_episode_results(folder):
    path = folder / "episodes.jsonl"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Closed-loop shadow/rerank evaluation of frozen DP and Predictor"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dp-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("shadow", "rerank"), default="shadow")
    parser.add_argument("--num-candidates", type=int, default=1)
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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-stop-on-success", action="store_true")
    parser.add_argument("--dr-mean-weight", type=float, default=1.0)
    parser.add_argument("--dr-max-weight", type=float, default=0.25)
    parser.add_argument("--topple-weight", type=float, default=0.1)
    parser.add_argument("--fall-weight", type=float, default=0.0)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--max-joint-deviation", type=float, default=0.15)
    args = parser.parse_args(argv)
    limits = SafetyLimits(
        args.safe_dr_mean_max, args.safe_dr_max_max,
        args.safe_dr_mean_min, args.safe_dr_max_min,
    )
    report = evaluate_dp_predictor(
        args.dataset, args.dp_checkpoint, args.predictor_checkpoint,
        args.output_dir, limits, mode=args.mode,
        num_candidates=args.num_candidates, device=args.device,
        max_scenes=args.max_scenes, scene_id=args.scene_id,
        episodes_per_scene=args.episodes_per_scene, max_steps=args.max_steps,
        exec_horizon=args.exec_horizon, eval_seed=args.eval_seed,
        save_video=not args.no_video,
        stop_on_success=not args.no_stop_on_success,
        dr_mean_weight=args.dr_mean_weight, dr_max_weight=args.dr_max_weight,
        topple_weight=args.topple_weight, fall_weight=args.fall_weight,
        min_improvement=args.min_improvement,
        max_joint_deviation=args.max_joint_deviation,
    )
    for key in (
        "episodes", "scenes", "success_rate", "safety_rate",
        "safe_success_rate", "mean_dr_mean", "mean_dr_max",
        "total_topple", "total_fall", "candidate_changed_count",
    ):
        print(f"{key}: {report[key]}")
    print(f"Results: {args.output_dir / 'summary.json'}")
    return report


if __name__ == "__main__":
    main()
