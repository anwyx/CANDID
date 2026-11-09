"""Train the disturbance predictor on train scenes; keep test scenes held out."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset, WeightedRandomSampler

from candid.data.splits import load_or_create_scene_split
from model.predictor.checkpoint import save_checkpoint
from model.predictor.config import PredictorConfig
from model.predictor.dataset import PredictorWindowDataset, TrajectoryBatchSampler
from model.predictor.network import DisturbancePredictor
from model.predictor.policy_rollout_dataset import PolicyRolloutWindowDataset
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


def _mean_loss(model, normalizer, dataset, device, batch_size=256) -> float | None:
    if dataset is None or len(dataset) == 0:
        return None
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
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
                batch["event_mask"],
                normalizer.event_pos_weight,
            )
            total += loss.item() * batch["obs"].shape[0]
            count += batch["obs"].shape[0]
    return total / count


def _validation_subset(dataset, max_windows, seed):
    if dataset is None or len(dataset) <= max_windows:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_windows].tolist()
    return Subset(dataset, sorted(indices))


def train_predictor(dataset_root, output_dir, config: PredictorConfig | None = None,
                    policy_rollouts=None) -> dict:
    config = PredictorConfig() if config is None else config
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    device = _select_device(config.device)
    print(f"predictor device: {_describe_device(device)}", flush=True)
    split = load_or_create_scene_split(
        dataset_root,
        seed=config.split_seed,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    train_set = PredictorWindowDataset(
        dataset_root, split["train"], config.obs_horizon, config.action_horizon,
        allow_missing_event_labels=config.allow_missing_event_labels,
        label_horizon=config.label_horizon,
    )
    val_set = (
        PredictorWindowDataset(
            dataset_root, split["validation"], config.obs_horizon, config.action_horizon,
            allow_missing_event_labels=config.allow_missing_event_labels,
            label_horizon=config.label_horizon,
        )
        if split["validation"]
        else None
    )
    fixed_val_set = _validation_subset(val_set, config.val_max_windows, config.seed)
    policy_set = None
    if policy_rollouts is not None:
        policy_set = PolicyRolloutWindowDataset(
            policy_rollouts, split["train"],
            obs_horizon=config.obs_horizon,
            action_horizon=config.action_horizon,
            label_horizon=config.label_horizon,
        )
        if (policy_set.obs_dim, policy_set.action_dim) != (
            train_set.obs_dim, train_set.action_dim
        ):
            raise ValueError("Expert and policy rollout feature dimensions disagree")
    mixed_set = ConcatDataset([train_set, policy_set]) if policy_set else train_set
    normalizer = fit_predictor_normalizer(mixed_set).to(device)
    model = DisturbancePredictor(
        train_set.obs_dim,
        train_set.action_dim,
        train_set.target_dim,
        config.obs_horizon,
        config.action_horizon,
        config.hidden,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    if policy_set:
        # Each stream gets half the SGD probability regardless of its length.
        # This prevents the larger expert dataset from drowning out states
        # encountered by the learned DP. No validation or test rollouts used.
        weights = torch.cat((
            torch.full((len(train_set),), 0.5 / len(train_set)),
            torch.full((len(policy_set),), 0.5 / len(policy_set)),
        )).double()
        sampler = WeightedRandomSampler(
            weights, num_samples=max(len(train_set), len(policy_set)),
            replacement=True,
            generator=torch.Generator().manual_seed(config.seed),
        )
        loader = DataLoader(
            mixed_set, batch_size=config.batch_size, sampler=sampler,
            num_workers=0,
        )
    else:
        loader = DataLoader(
            train_set,
            batch_sampler=TrajectoryBatchSampler(
                train_set, min(config.batch_size, len(train_set)), seed=config.seed
            ),
            num_workers=0,
        )

    checkpoint_dir = output_dir / "checkpoints"
    best_path = checkpoint_dir / "best_val.pt"
    last_path = checkpoint_dir / "last.pt"
    losses = []
    best_val_loss = float("inf")
    best_step = None
    last_val_loss = None
    log_window = []
    step = 0
    log_path = output_dir / "train_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "train_loss", "val_loss"))
        writer.writeheader()
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
                    batch["event_mask"],
                    normalizer.event_pos_weight,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                value = loss.item()
                losses.append(value)
                log_window.append(value)
                step += 1

                should_log = step % config.log_freq == 0 or step == config.steps
                should_eval = (
                    fixed_val_set is not None
                    and (step % config.eval_freq == 0 or step == config.steps)
                )
                if should_eval:
                    last_val_loss = _mean_loss(
                        model, normalizer, fixed_val_set, device, config.val_batch_size
                    )
                    if last_val_loss < best_val_loss:
                        best_val_loss = last_val_loss
                        best_step = step
                        save_checkpoint(best_path, model, normalizer, config)
                    model.train()

                if should_log or should_eval:
                    train_loss = sum(log_window) / len(log_window) if log_window else None
                    writer.writerow({
                        "step": step,
                        "train_loss": train_loss,
                        "val_loss": last_val_loss if should_eval else "",
                    })
                    handle.flush()
                    if train_loss is not None:
                        message = f"step {step}/{config.steps} train_loss={train_loss:.6f}"
                        if should_eval:
                            message += f" val_loss={last_val_loss:.6f} best_step={best_step}"
                        print(message, flush=True)
                    log_window.clear()

    save_checkpoint(last_path, model, normalizer, config)
    checkpoint = best_path if best_step is not None else last_path
    summary = {
        "checkpoint": str(checkpoint),
        "best_checkpoint": str(best_path) if best_step is not None else None,
        "last_checkpoint": str(last_path),
        "best_step": best_step,
        "best_val_loss": best_val_loss if best_step is not None else None,
        "val_loss": last_val_loss,
        "train_scenes": split["train"],
        "val_scenes": split["validation"],
        "test_scenes": split["test"],
        "split_file": str(dataset_root / "split.json"),
        "train_windows": len(train_set),
        "policy_train_windows": len(policy_set) if policy_set else 0,
        "policy_rollouts": str(policy_rollouts) if policy_rollouts else None,
        "label_horizon": config.label_horizon,
        "train_missing_event_trajectories": train_set.missing_event_trajectories,
        "train_missing_event_windows": train_set.missing_event_windows,
        "train_skipped_short_trajectories": train_set.skipped_short,
        "validation_windows": len(fixed_val_set) if fixed_val_set is not None else 0,
        "validation_missing_event_trajectories": val_set.missing_event_trajectories if val_set is not None else 0,
        "steps_completed": step,
        "device": str(device),
        "config": asdict(config),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        **summary,
        "checkpoint": checkpoint,
        "best_checkpoint": best_path if best_step is not None else None,
        "last_checkpoint": last_path,
        "losses": losses,
        "windows": len(train_set),
    }


def main(argv=None):
    defaults = PredictorConfig()
    parser = argparse.ArgumentParser(description="Train the CANDID disturbance predictor")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--obs-horizon", type=int, default=defaults.obs_horizon)
    parser.add_argument("--action-horizon", type=int, default=defaults.action_horizon)
    parser.add_argument("--label-horizon", type=int, default=defaults.label_horizon)
    parser.add_argument("--policy-rollouts", type=Path, default=None,
                        help="Train-split Vanilla DP rollout directory with canonical observations")
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--split-seed", type=int, default=defaults.split_seed)
    parser.add_argument("--device", default=defaults.device, choices=("auto", "cpu", "cuda"))
    parser.add_argument("--val-ratio", type=float, default=defaults.val_ratio)
    parser.add_argument("--test-ratio", type=float, default=defaults.test_ratio)
    parser.add_argument("--log-freq", type=int, default=defaults.log_freq)
    parser.add_argument("--eval-freq", type=int, default=defaults.eval_freq)
    parser.add_argument("--val-max-windows", type=int, default=defaults.val_max_windows)
    parser.add_argument("--val-batch-size", type=int, default=defaults.val_batch_size)
    parser.add_argument("--allow-missing-event-labels", action="store_true")
    args = parser.parse_args(argv)
    config = PredictorConfig(
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        label_horizon=args.label_horizon,
        lr=args.lr,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        split_seed=args.split_seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        device=args.device,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        val_max_windows=args.val_max_windows,
        val_batch_size=args.val_batch_size,
        allow_missing_event_labels=args.allow_missing_event_labels,
    )
    result = train_predictor(args.dataset, args.output_dir, config,
                             policy_rollouts=args.policy_rollouts)
    print(f"checkpoint: {result['checkpoint']}")
    print(f"last checkpoint: {result['last_checkpoint']}")
    print(f"best step: {result['best_step']}")
    print(f"train scenes: {result['train_scenes']}")
    print(f"val scenes: {result['val_scenes']}")
    print(f"test scenes (held out): {result['test_scenes']}")
    print(f"loss: {result['losses'][0]:.6f} -> {result['losses'][-1]:.6f}")
    if result["val_loss"] is not None:
        print(f"last val loss: {result['val_loss']:.6f}")
        print(f"best val loss: {result['best_val_loss']:.6f}")
    return result


if __name__ == "__main__":
    main()
