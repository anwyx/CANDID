"""Train the disturbance predictor on train scenes. Test scenes stay unused."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from candid.data.splits import load_or_create_scene_split
from candid.data.trajectories import load_trajectories
from model.predictor.checkpoint import save_checkpoint
from model.predictor.config import PredictorConfig
from model.predictor.dataset import PredictorWindowDataset
from model.predictor.network import DisturbancePredictor
from model.predictor.normalizer import fit_predictor_normalizer


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this PyTorch cannot see a GPU")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be auto, cpu, or cuda")


def _describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return "cpu"


def _mean_loss(model, normalizer, dataset, device) -> float | None:
    if len(dataset) == 0:
        return None
    loader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=False)
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model.loss(
                normalizer.normalize_obs(batch["obs"]),
                normalizer.normalize_action(batch["action"]),
                normalizer.normalize_target(batch["target"]),
            )
            total += float(loss) * batch["obs"].shape[0]
            count += batch["obs"].shape[0]
    return total / count


def train_predictor(dataset_root, output_dir, config: PredictorConfig | None = None) -> dict:
    config = PredictorConfig() if config is None else config
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    torch.manual_seed(config.seed)
    device = _select_device(config.device)
    print(f"predictor device: {_describe_device(device)}", flush=True)
    split = load_or_create_scene_split(
        dataset_root,
        seed=config.split_seed,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    trajectories = load_trajectories(dataset_root)
    train_set = PredictorWindowDataset(
        trajectories, split["train"], config.obs_horizon, config.action_horizon
    )
    val_set = (
        PredictorWindowDataset(
            trajectories, split["validation"], config.obs_horizon, config.action_horizon
        )
        if split["validation"]
        else None
    )
    normalizer = fit_predictor_normalizer(train_set).to(device)
    model = DisturbancePredictor(
        train_set.obs_dim,
        train_set.action_dim,
        train_set.target_dim,
        config.obs_horizon,
        config.action_horizon,
        config.hidden,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        train_set,
        batch_size=min(config.batch_size, len(train_set)),
        shuffle=True,
        generator=generator,
    )
    losses = []
    step = 0
    model.train()
    while step < config.steps:
        for batch in loader:
            if step >= config.steps:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model.loss(
                normalizer.normalize_obs(batch["obs"]),
                normalizer.normalize_action(batch["action"]),
                normalizer.normalize_target(batch["target"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
    val_loss = None if val_set is None else _mean_loss(model, normalizer, val_set, device)
    checkpoint = save_checkpoint(
        output_dir / "checkpoints" / "disturbance_predictor.pt",
        model,
        normalizer,
        config,
    )
    return {
        "checkpoint": checkpoint,
        "train_scenes": split["train"],
        "val_scenes": split["validation"],
        "test_scenes": split["test"],
        "split_file": str(dataset_root / "split.json"),
        "losses": losses,
        "val_loss": val_loss,
        "windows": len(train_set),
        "device": str(device),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the CANDID disturbance predictor")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args(argv)
    config = PredictorConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        split_seed=args.split_seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        device=args.device,
    )
    result = train_predictor(args.dataset, args.output_dir, config)
    print(f"device: {result['device']}")
    print(f"checkpoint: {result['checkpoint']}")
    print(f"train scenes: {result['train_scenes']}")
    print(f"val scenes: {result['val_scenes']}")
    print(f"test scenes (held out): {result['test_scenes']}")
    print(f"loss: {result['losses'][0]:.6f} -> {result['losses'][-1]:.6f}")
    if result["val_loss"] is not None:
        print(f"val loss: {result['val_loss']:.6f}")
    return result


if __name__ == "__main__":
    main()
