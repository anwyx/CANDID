"""Train the state-based Vanilla Diffusion Policy.

The optimization step follows ManiSkill ``train.py`` at commit
62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3: AdamW (lr 1e-4, betas 0.95/0.999,
weight decay 1e-6), a cosine schedule with linear warmup, an EMA update, and
an iteration-counted batch sampler. In-loop simulator evaluation, TensorBoard,
wandb, tqdm, and tyro are not part of this entry point. The final EMA weights
are always written so ``evaluate.py`` can load them.

Two practical guards are not in the upstream script. Warmup is capped at the
number of training steps, because a short run cannot warm up for 500 steps.
If the collection is smaller than the requested batch, the batch shrinks to
the collection and the final partial batch is kept; upstream ``drop_last``
with batch size 1024 would otherwise yield no updates.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from candid.data.splits import load_or_create_scene_split
from candid.data.trajectories import load_trajectories
from diffusers.optimization import get_scheduler
from model.vanilla_dp.checkpoint import save_checkpoint
from model.vanilla_dp.config import DPConfig
from model.vanilla_dp.dataset import DiffusionWindowDataset
from model.vanilla_dp.paired_dataset import PairedDiffusionWindowDataset
from model.vanilla_dp.temporal_loss import compute_temporal_losses
from model.vanilla_dp.ema import EMAModel
from model.vanilla_dp.normalizer import ObservationNormalizer, fit_action_normalizer
from model.vanilla_dp.policy import Agent
from model.vanilla_dp.validation import (compute_validation_loss, make_validation_loader,
                                          plot_loss_history)


class _IterationBasedBatchSampler:
    """Copied from ManiSkill ``diffusion_policy/utils.py``."""

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        return self.num_iterations - self.start_iter


def _device(config: DPConfig) -> torch.device:
    if config.cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return "cpu"


def _select_trajectories(trajectories, dataset_root, config: DPConfig):
    """Apply the optional route and joint-range filters used by diagnostics."""
    if config.routes is None and config.max_abs_joint is None:
        return trajectories
    with (Path(dataset_root) / "trajectories.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    routes = {row["trajectory_id"]: row["planner_config"].get("route_intent") for row in rows}
    selected = []
    for trajectory in trajectories:
        if config.routes is not None and routes[trajectory.trajectory_id] not in config.routes:
            continue
        if (config.max_abs_joint is not None
                and np.abs(trajectory.actions[:, [0, 3, 5]]).max() > config.max_abs_joint):
            continue
        selected.append(trajectory)
    print(f"trajectory filter: kept {len(selected)}/{len(trajectories)}", flush=True)
    return selected


def train_vanilla_dp(dataset_root, output_dir, config: DPConfig | None = None) -> dict:
    config = DPConfig() if config is None else config
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    device = _device(config)
    print(f"vanilla_dp device: {_describe_device(device)}", flush=True)

    split = load_or_create_scene_split(
        dataset_root,
        seed=config.split_seed,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    trajectories = _select_trajectories(load_trajectories(dataset_root), dataset_root, config)
    dataset = DiffusionWindowDataset(
        trajectories, split["train"], config.obs_horizon, config.pred_horizon
    )
    validation_dataset = DiffusionWindowDataset(
        trajectories, split["validation"], config.obs_horizon, config.pred_horizon
    )
    obs_normalizer = None
    if config.normalize_obs:
        obs_normalizer = ObservationNormalizer.fit(dataset.training_observations())
        dataset.obs_normalizer = obs_normalizer
        validation_dataset.obs_normalizer = obs_normalizer
    regularized = (config.delta_loss_weight > 0 or config.overlap_loss_weight > 0)
    if regularized:
        optimization_dataset = PairedDiffusionWindowDataset(
            dataset, stride=config.overlap_stride, act_horizon=config.act_horizon
        )
        print(f'temporal training: {len(optimization_dataset)} paired windows, '
              f'delta={config.delta_loss_weight}, overlap={config.overlap_loss_weight}, '
              f'stride={config.overlap_stride}', flush=True)
    else:
        optimization_dataset = dataset
    validation_loader, validation_count = make_validation_loader(
        validation_dataset, config.val_batch_size, config.val_max_windows, config.seed + 17
    )
    if (dataset.obs_dim, dataset.action_dim) != (validation_dataset.obs_dim, validation_dataset.action_dim):
        raise ValueError("Train and validation observation/action dimensions differ")
    print(f"validation: {validation_count} windows, every {config.eval_freq} iterations", flush=True)
    normalizer = fit_action_normalizer(dataset.training_actions()).to(device)
    agent = Agent(dataset.obs_dim, dataset.action_dim, config).to(device)
    ema_agent = Agent(dataset.obs_dim, dataset.action_dim, config).to(device)
    ema = EMAModel(agent.parameters(), power=0.75)
    optimizer = torch.optim.AdamW(
        agent.parameters(), lr=config.lr, betas=(0.95, 0.999), weight_decay=1e-6
    )
    warmup = min(config.warmup_steps, config.total_iters)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup,
        num_training_steps=config.total_iters,
    )
    # Each pair contains two windows. Keep ~the same number of network
    # windows/update as the original batch_size for a fairer comparison.
    requested_batch_size = max(1, config.batch_size // 2) if regularized else config.batch_size
    batch_size = min(requested_batch_size, len(optimization_dataset))
    drop_last = len(optimization_dataset) >= requested_batch_size
    sampler = RandomSampler(optimization_dataset, replacement=False)
    batch_sampler = _IterationBasedBatchSampler(
        BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last),
        config.total_iters,
    )
    loader = DataLoader(optimization_dataset, batch_sampler=batch_sampler, num_workers=0)

    agent.train()
    ema_agent.eval()
    losses = []
    history = []
    train_loss_sum = 0.0
    train_sample_count = 0
    train_component_sums = {"noise": 0.0, "delta": 0.0, "overlap": 0.0}
    best_val_loss = float("inf")
    best_iteration = None
    best_checkpoint = output_dir / "checkpoints" / "best_val.pt"
    loss_csv = output_dir / "loss_history.csv"
    loss_png = output_dir / "loss_curve.png"
    loss_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_file = loss_csv.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=(
        "iteration", "train_loss", "train_noise_loss", "train_delta_loss",
        "train_overlap_loss", "val_loss", "lr"
    ))
    writer.writeheader()
    for iteration, batch in enumerate(loader):
        observations = batch["observations"].to(device)
        actions = normalizer.normalize(batch["actions"].to(device))
        if regularized:
            next_observations = batch["next_observations"].to(device)
            next_actions = normalizer.normalize(batch["next_actions"].to(device))
            parts = compute_temporal_losses(
                agent, observations, actions, next_observations, next_actions,
                stride=config.overlap_stride,
                max_timestep=config.regularization_max_timestep,
            )
            loss = (parts["noise"]
                    + config.delta_loss_weight * parts["delta"]
                    + config.overlap_loss_weight * parts["overlap"])
        else:
            noise_loss = agent.compute_loss(observations, actions)
            parts = {"noise": noise_loss, "delta": noise_loss * 0.0,
                     "overlap": noise_loss * 0.0}
            loss = noise_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        ema.step(agent.parameters())
        current_loss = float(loss.detach().cpu())
        losses.append(current_loss)
        batch_count = observations.shape[0]
        train_loss_sum += current_loss * batch_count
        train_sample_count += batch_count
        for name in train_component_sums:
            train_component_sums[name] += float(parts[name].detach().cpu()) * batch_count
        step = iteration + 1
        if step % config.eval_freq == 0 or step == config.total_iters:
            ema.copy_to(ema_agent.parameters())
            val_loss = compute_validation_loss(
                ema_agent, validation_loader, normalizer, device, config.seed + 10000
            )
            row = {"iteration": step, "train_loss": train_loss_sum / train_sample_count,
                   "train_noise_loss": train_component_sums["noise"] / train_sample_count,
                   "train_delta_loss": train_component_sums["delta"] / train_sample_count,
                   "train_overlap_loss": train_component_sums["overlap"] / train_sample_count,
                   "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]}
            history.append(row)
            writer.writerow(row)
            csv_file.flush()
            print(f"validation iter={step}: train={row['train_loss']:.6f} val_EMA={val_loss:.6f}", flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iteration = step
                save_checkpoint(best_checkpoint, agent, ema_agent, config, normalizer, obs_normalizer)
                print(f"new best: iteration {step}, validation={val_loss:.6f}", flush=True)
            train_loss_sum = 0.0
            train_sample_count = 0
            train_component_sums = {"noise": 0.0, "delta": 0.0, "overlap": 0.0}
        if iteration % config.log_freq == 0 or iteration + 1 == config.total_iters:
            print(
                f"vanilla_dp iter {iteration}: loss={losses[-1]:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f} device={_describe_device(device)}",
                flush=True,
            )

    csv_file.close()
    ema.copy_to(ema_agent.parameters())
    checkpoint = save_checkpoint(
        output_dir / "checkpoints" / "vanilla_dp.pt",
        agent,
        ema_agent,
        config,
        normalizer,
        obs_normalizer,
    )
    plot_loss_history(history, loss_png)
    return {
        "checkpoint": checkpoint,
        "best_checkpoint": best_checkpoint,
        "best_iteration": best_iteration,
        "best_val_loss": best_val_loss,
        "loss_history_csv": loss_csv,
        "loss_figure": loss_png,
        "loss_history": history,
        "validation_windows": validation_count,
        "train_scenes": split["train"],
        "val_scenes": split["validation"],
        "test_scenes": split["test"],
        "split_file": str(dataset_root / "split.json"),
        "losses": losses,
        "obs_horizon": config.obs_horizon,
        "act_horizon": config.act_horizon,
        "pred_horizon": config.pred_horizon,
        "device": str(device),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train CANDID's state-based Vanilla DP")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-iters", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-freq", type=int, default=500)
    parser.add_argument("--val-max-windows", type=int, default=4096)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--act-horizon", type=int, default=8)
    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--overlap-loss-weight", type=float, default=0.0)
    parser.add_argument("--overlap-stride", type=int, default=4)
    parser.add_argument("--regularization-max-timestep", type=int, default=20)
    parser.add_argument("--normalize-obs", action="store_true")
    parser.add_argument("--max-abs-joint", type=float, default=None,
                        help="Drop trajectories whose joints 1, 4 or 6 command exceeds this (rad)")
    parser.add_argument("--routes", nargs="+", default=None,
                        help="Train only on these route intents")
    args = parser.parse_args(argv)
    config = DPConfig(
        obs_horizon=args.obs_horizon,
        act_horizon=args.act_horizon,
        pred_horizon=args.pred_horizon,
        lr=args.lr,
        batch_size=args.batch_size,
        total_iters=args.total_iters,
        eval_freq=args.eval_freq,
        val_max_windows=args.val_max_windows,
        val_batch_size=args.val_batch_size,
        log_freq=args.log_freq,
        seed=args.seed,
        split_seed=args.split_seed,
        cuda=not args.no_cuda,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        delta_loss_weight=args.delta_loss_weight,
        overlap_loss_weight=args.overlap_loss_weight,
        overlap_stride=args.overlap_stride,
        regularization_max_timestep=args.regularization_max_timestep,
        normalize_obs=args.normalize_obs,
        max_abs_joint=args.max_abs_joint,
        routes=None if args.routes is None else tuple(args.routes),
    )
    result = train_vanilla_dp(args.dataset, args.output_dir, config)
    print(f"checkpoint: {result['checkpoint']}")
    print(f"best checkpoint: {result['best_checkpoint']}")
    print(f"best validation MSE: {result['best_val_loss']:.6f} at iteration {result['best_iteration']}")
    print(f"loss history: {result['loss_history_csv']}")
    print(f"loss curve: {result['loss_figure']}")
    print(f"train scenes: {result['train_scenes']}")
    print(f"val scenes: {result['val_scenes']}")
    print(f"test scenes (held out): {result['test_scenes']}")
    print(f"loss: {result['losses'][0]:.6f} -> {result['losses'][-1]:.6f}")
    return result


if __name__ == "__main__":
    main()
