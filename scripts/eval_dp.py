"""Closed-loop evaluation of CANDID's state-based Vanilla DP on held-out scenes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from candid.eval.metrics import SafetyLimits
from candid.eval.rollout import evaluate_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate trained DP on frozen TEST scenes")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--safe-dr-mean-max", type=float, required=True,
                        help="Maximum acceptable episode DR Mean (inclusive)")
    parser.add_argument("--safe-dr-mean-min", type=float, default=0.0,
                        help="Minimum acceptable episode DR Mean (default 0)")
    parser.add_argument("--safe-dr-max-min", type=float, default=0.0,
                        help="Optional minimum acceptable per-object DR Max")
    parser.add_argument("--safe-dr-max-max", type=float, default=None,
                        help="Optional maximum acceptable per-object DR Max (inclusive)")
    parser.add_argument("--scene-id", default=None, help="Only evaluate this held-out test Scene")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--episodes-per-scene", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--exec-horizon", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-stop-on-success", action="store_true")
    args = parser.parse_args(argv)
    limits = SafetyLimits(args.safe_dr_mean_max, args.safe_dr_max_max,
                          args.safe_dr_mean_min, args.safe_dr_max_min)
    result = evaluate_dataset(
        args.dataset, args.checkpoint, args.output_dir, limits,
        max_scenes=args.max_scenes, scene_id=args.scene_id,
        episodes_per_scene=args.episodes_per_scene,
        max_steps=args.max_steps, exec_horizon=args.exec_horizon,
        eval_seed=args.eval_seed, device=args.device,
        save_video=not args.no_video,
        stop_on_success=not args.no_stop_on_success,
    )
    print("\nCANDID Vanilla DP Evaluation")
    for key in ("episodes", "scenes", "success_rate", "success_once_rate",
                "safety_rate", "safe_success_rate", "mean_dr_mean", "mean_dr_max",
                "total_topple", "total_fall"):
        print(f"{key}: {result[key]}")
    print(f"Results: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
