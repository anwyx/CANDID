"""Train the pilot diffusion policy and disturbance predictor.

The collection format is the one written by ``scripts/collect.py``:
``state_dict`` observations and ``pd_joint_pos`` actions. Video files are not read.

On the first training run a dataset's split.json is created automatically.
Later training runs reuse the same train/validation/test scene split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from candid.train.loop import run_pilot
from candid.train.synthetic import write_synthetic_dataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the CANDID pilot policy and predictor")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="Validated collection directory")
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="Write a contract-matching stand-in dataset, then train on it",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not 0 <= args.val_ratio < 1:
        parser.error("--val-ratio must be in [0, 1)")
    if not 0 < args.test_ratio < 1:
        parser.error("--test-ratio must be in (0, 1)")
    if args.val_ratio + args.test_ratio >= 1:
        parser.error("--val-ratio + --test-ratio must be less than 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset
    if args.synthetic:
        dataset = write_synthetic_dataset(args.output_dir / "synthetic_dataset")
    result = run_pilot(
        dataset,
        args.output_dir / "checkpoints",
        steps=args.steps,
        seed=args.seed,
        val_ratio=args.val_ratio,
        diffusion_steps=args.diffusion_steps,
        device=args.device,
        test_ratio=args.test_ratio,
    )
    print("CANDID train pilot")
    print(f"dataset: {dataset}")
    print(f"validation: {result['validation']}")
    print(f"split file: {result['split_file']}")
    print(f"train scenes: {result['train_scenes']}")
    print(f"val scenes: {result['val_scenes']}")
    print(f"test scenes (held out): {result['test_scenes']}")
    print(
        f"windows: policy {result['policy_train_windows']}, "
        f"predictor {result['predictor_train_windows']}"
    )
    print(
        f"policy loss: {result['policy_losses'][0]:.6f} -> {result['policy_losses'][-1]:.6f}"
    )
    print(
        f"predictor loss: {result['predictor_losses'][0]:.6f} -> "
        f"{result['predictor_losses'][-1]:.6f}"
    )
    print(f"sample action: {tuple(result['sample_action'].shape)}")
    print(f"sample prediction: {tuple(result['sample_prediction'].shape)}")
    print(f"policy checkpoint: {result['policy_checkpoint']}")
    print(f"predictor checkpoint: {result['predictor_checkpoint']}")
    return result


if __name__ == "__main__":
    main()
