"""Paired Vanilla / CANDID guidance inspection for one frozen scene."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from model.guidance.inspect import inspect_scene


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-suite", required=True, help="Dataset directory containing scenes.jsonl")
    parser.add_argument("--scene-id", required=True, help="Exact scene_id from scenes.jsonl")
    parser.add_argument("--dp-checkpoint", required=True)
    parser.add_argument("--predictor-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strength", type=float, default=0.005)
    parser.add_argument("--start-timestep", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=300, help="Reserved for a simulator rollout")
    parser.add_argument("--exec-horizon", type=int, default=4)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trace-replan", type=int, default=0)
    parser.add_argument("--episode-step", type=int, default=None)
    parser.add_argument("--max-replans", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--dr-mean-weight", type=float, default=1.0)
    parser.add_argument("--dr-max-weight", type=float, default=0.25)
    parser.add_argument("--topple-weight", type=float, default=0.1)
    parser.add_argument("--fall-weight", type=float, default=0.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.render and args.no_render:
        raise SystemExit("Pass only one of --render and --no-render")
    if args.device == "cuda" and not _cuda_available():
        raise SystemExit("CUDA was requested but is not available")
    result = inspect_scene(
        args.eval_suite,
        args.scene_id,
        args.dp_checkpoint,
        args.predictor_checkpoint,
        args.output_dir,
        strength=args.strength,
        start_timestep=args.start_timestep,
        exec_horizon=args.exec_horizon,
        eval_seed=args.eval_seed,
        device=args.device,
        trace_replan=args.trace_replan,
        episode_step=args.episode_step,
        max_replans=args.max_replans,
        weights=(
            args.dr_mean_weight,
            args.dr_max_weight,
            args.topple_weight,
            args.fall_weight,
        ),
        render=bool(args.render and not args.no_render),
    )
    print(json.dumps({"output_dir": result["output_dir"], "record": result["record"]}, indent=2))
    return result


def _cuda_available():
    import torch
    return bool(torch.cuda.is_available())


if __name__ == "__main__":
    main()
