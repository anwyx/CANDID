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
from model.vanilla_dp.ema import EMAModel
from model.vanilla_dp.normalizer import fit_action_normalizer
from model.vanilla_dp.policy import Agent


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
    trajectories = load_trajectories(dataset_root)
    dataset = DiffusionWindowDataset(
        trajectories, split["train"], config.obs_horizon, config.pred_horizon
    )
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
    batch_size = min(config.batch_size, len(dataset))
    drop_last = len(dataset) >= config.batch_size
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = _IterationBasedBatchSampler(
        BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last),
        config.total_iters,
    )
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)

    agent.train()
    losses = []
    for iteration, batch in enumerate(loader):
        observations = batch["observations"].to(device)
        actions = normalizer.normalize(batch["actions"].to(device))
        loss = agent.compute_loss(observations, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        ema.step(agent.parameters())
        losses.append(float(loss.detach().cpu()))
        if iteration % config.log_freq == 0 or iteration + 1 == config.total_iters:
            print(
                f"vanilla_dp iter {iteration}: loss={losses[-1]:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f} device={_describe_device(device)}",
                flush=True,
            )

    ema.copy_to(ema_agent.parameters())
    checkpoint = save_checkpoint(
        output_dir / "checkpoints" / "vanilla_dp.pt",
        agent,
        ema_agent,
        config,
        normalizer,
    )
    return {
        "checkpoint": checkpoint,
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
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--act-horizon", type=int, default=8)
    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args(argv)
    config = DPConfig(
        obs_horizon=args.obs_horizon,
        act_horizon=args.act_horizon,
        pred_horizon=args.pred_horizon,
        lr=args.lr,
        batch_size=args.batch_size,
        total_iters=args.total_iters,
        seed=args.seed,
        split_seed=args.split_seed,
        cuda=not args.no_cuda,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    result = train_vanilla_dp(args.dataset, args.output_dir, config)
    print(f"checkpoint: {result['checkpoint']}")
    print(f"train scenes: {result['train_scenes']}")
    print(f"val scenes: {result['val_scenes']}")
    print(f"test scenes (held out): {result['test_scenes']}")
    print(f"loss: {result['losses'][0]:.6f} -> {result['losses'][-1]:.6f}")
    return result


if __name__ == "__main__":
    main()
