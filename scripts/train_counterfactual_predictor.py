"""Fine-tune a frozen-architecture ordinal predictor with physical counterfactuals.

Keeps the expert BCE objective, adds physical candidate supervision, and
ranks candidates ONLY within their common replayed reference state.
Does not modify DP or predictor checkpoint format.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from candid.data.splits import load_or_create_scene_split
from model.predictor.checkpoint import load_predictor, save_checkpoint
from model.predictor.config import PredictorConfig
from model.predictor.counterfactual import (
    CounterfactualStateDataset, pairwise_physical_ranking_loss,
)
from model.predictor.dataset import PredictorWindowDataset, TrajectoryBatchSampler
from model.predictor.network import OrdinalDisturbancePredictor
from model.predictor.normalizer import fit_predictor_normalizer
from model.predictor.train import _mean_loss, _validation_subset, _select_device


def _supervised(model, normalizer, obs, actions, targets, event_mask):
    return model.loss(
        normalizer.normalize_obs(obs), normalizer.normalize_action(actions),
        targets, event_mask, normalizer.event_pos_weight,
    )


def _counterfactual_losses(model, normalizer, batch, *, temperature,
                           topple_weight, fall_weight, dr_max_weight,
                           min_risk_gap):
    obs = batch["obs"]
    actions = batch["actions"]
    labels = batch["targets"]
    n, k, h, a = actions.shape
    histories = obs[:, None].expand(-1, k, -1, -1)
    flat_obs = histories.reshape(n*k, obs.shape[1], obs.shape[2])
    flat_act = actions.reshape(n*k, h, a)
    flat_targets = labels.reshape(n*k, 4)
    supervised = _supervised(
        model, normalizer, flat_obs, flat_act, flat_targets,
        torch.ones((n*k, 2), device=obs.device, dtype=obs.dtype))
    out = model(
        normalizer.normalize_obs(flat_obs), normalizer.normalize_action(flat_act))
    grouped = {
        "dr": out["dr"].reshape(n, k, 2),
        "event_logits": out["event_logits"].reshape(n, k, 2),
    }
    rank, stats = pairwise_physical_ranking_loss(
        grouped, labels,
        temperature=temperature,
        topple_weight=topple_weight,
        fall_weight=fall_weight,
        dr_max_weight=dr_max_weight,
        min_risk_gap=min_risk_gap,
    )
    return supervised, rank, stats


def train(args):
    if (min(args.steps, args.batch_size, args.cf_batch_size, args.eval_freq) < 1
            or min(args.expert_weight, args.cf_weight, args.rank_weight) < 0
            or args.expert_weight + args.cf_weight + args.rank_weight <= 0):
        raise ValueError("Require positive steps/batch sizes and nonnegative nonzero loss weights")
    torch.manual_seed(args.seed)
    device = _select_device(args.device)
    dataset_root = Path(args.dataset)
    outdir = Path(args.output_dir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    split = load_or_create_scene_split(
        dataset_root, seed=args.split_seed, val_ratio=args.val_ratio,
        test_ratio=args.test_ratio)
    train_set = PredictorWindowDataset(
        dataset_root, split["train"], obs_horizon=2, action_horizon=8,
        allow_missing_event_labels=False, cache_size=8)
    val_set = (PredictorWindowDataset(
        dataset_root, split["validation"], obs_horizon=2, action_horizon=8,
        allow_missing_event_labels=False)
        if split["validation"] else None)
    fixed_val = _validation_subset(val_set, args.val_max_windows, args.seed)
    cf = CounterfactualStateDataset(
        args.counterfactual_dataset, dataset_root, split,
        obs_horizon=2, action_horizon=8,
        obs_dim=train_set.obs_dim, action_dim=train_set.action_dim)
    if args.init_checkpoint:
        model, normalizer = load_predictor(args.init_checkpoint, device=device)
        if not isinstance(model, OrdinalDisturbancePredictor):
            raise ValueError("Only ordinal predictor initialization is supported")
        if (model.obs_dim != train_set.obs_dim or model.action_dim != train_set.action_dim
                or model.obs_horizon != 2 or model.action_horizon != 8):
            raise ValueError("Initial Predictor architecture differs from expert and CF data")
        thresholds = tuple(float(t) for t in model.dr_thresholds.cpu().tolist())
    else:
        normalizer = fit_predictor_normalizer(train_set).to(device)
        thresholds = tuple(float(t) for t in args.dr_thresholds.split(","))
        model = OrdinalDisturbancePredictor(
            train_set.obs_dim, train_set.action_dim, 4, 2, 8,
            hidden=args.hidden,
            dr_thresholds=thresholds,
            ordinal_pos_weight=args.ordinal_pos_weight).to(device)
    # Keep save_checkpoint/load_predictor fully backward compatible.
    config = PredictorConfig(
        obs_horizon=2, action_horizon=8, hidden=model.encoder[0].out_features,
        dr_objective="ordinal", dr_thresholds=thresholds,
        ordinal_pos_weight=model.ordinal_pos_weight,
        lr=args.lr, steps=args.steps, batch_size=args.batch_size, seed=args.seed,
        split_seed=args.split_seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        device=str(device), log_freq=args.log_freq, eval_freq=args.eval_freq,
        val_max_windows=args.val_max_windows, val_batch_size=args.val_batch_size,
    )
    expert_loader = DataLoader(
        train_set,
        batch_sampler=TrajectoryBatchSampler(
            train_set, min(args.batch_size, len(train_set)), seed=args.seed),
        num_workers=0,
    )
    cf_loader = DataLoader(
        cf, batch_size=min(args.cf_batch_size, len(cf)), shuffle=True, num_workers=0)
    expert_iter = iter(expert_loader)
    cf_iter = iter(cf_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoints = outdir/"checkpoints"
    best_val = float("inf")
    best_step = None
    last_val = None
    log_path = outdir/"train_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as log:
        writer = csv.DictWriter(log, fieldnames=(
            "step", "total_loss", "expert_loss", "cf_loss", "ranking_loss",
            "ranking_accuracy", "ranking_pairs", "validation_expert_loss"))
        writer.writeheader()
        for step in range(1, args.steps+1):
            try:
                expert = next(expert_iter)
            except StopIteration:
                expert_iter = iter(expert_loader)
                expert = next(expert_iter)
            try:
                replay = next(cf_iter)
            except StopIteration:
                cf_iter = iter(cf_loader)
                replay = next(cf_iter)
            expert = {key: val.to(device) for key, val in expert.items()}
            replay = {key: val.to(device) for key, val in replay.items()}
            model.train()
            expert_loss = _supervised(
                model, normalizer, expert["obs"], expert["action"],
                expert["target"], expert["event_mask"])
            cf_loss, ranking, ranking_stats = _counterfactual_losses(
                model, normalizer, replay,
                temperature=args.rank_temperature,
                topple_weight=args.topple_weight,
                fall_weight=args.fall_weight,
                dr_max_weight=args.dr_max_weight,
                min_risk_gap=args.min_risk_gap)
            loss = (args.expert_weight*expert_loss
                    + args.cf_weight*cf_loss + args.rank_weight*ranking)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            evaluate = step % args.eval_freq == 0 or step == args.steps
            if evaluate:
                if fixed_val is not None:
                    last_val = _mean_loss(
                        model, normalizer, fixed_val, device,
                        batch_size=args.val_batch_size)
                    if last_val < best_val:
                        best_val = last_val
                        best_step = step
                        save_checkpoint(checkpoints/"best_val.pt", model, normalizer, config)
                else:
                    best_step = step
                    save_checkpoint(checkpoints/"best_val.pt", model, normalizer, config)
            if evaluate or step % args.log_freq == 0 or step == 1:
                writer.writerow({
                    "step": step,
                    "total_loss": loss.item(),
                    "expert_loss": expert_loss.item(),
                    "cf_loss": cf_loss.item(),
                    "ranking_loss": ranking.item(),
                    "ranking_accuracy": ranking_stats["accuracy"],
                    "ranking_pairs": ranking_stats["pairs"],
                    "validation_expert_loss": last_val if evaluate else "",
                })
                log.flush()
                print(
                    f"step {step}/{args.steps} "
                    f"loss={loss.item():.5f} expert={expert_loss.item():.5f} "
                    f"cf={cf_loss.item():.5f} ranking={ranking.item():.5f} "
                    f"pairs={ranking_stats['pairs']} acc={ranking_stats['accuracy']} "
                    f"val_expert={last_val if evaluate else 'NA'}", flush=True)
    save_checkpoint(checkpoints/"last.pt", model, normalizer, config)
    summary = {
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "best_checkpoint": str(checkpoints/"best_val.pt"),
        "last_checkpoint": str(checkpoints/"last.pt"),
        "best_step": best_step, "best_expert_val_loss": best_val if fixed_val else None,
        "training_expert_windows": len(train_set),
        "counterfactual_states": len(cf),
        "counterfactual_candidates_per_state": cf.num_candidates,
        "counterfactual_source": str(args.counterfactual_dataset),
        "validation_expert_windows": len(fixed_val) if fixed_val else 0,
        "train_scene_ids": split["train"], "validation_scene_ids": split["validation"],
        "test_scene_ids": split["test"], "weights": {
            "expert": args.expert_weight, "counterfactual": args.cf_weight,
            "ranking": args.rank_weight},
        "ranking": {
            "temperature": args.rank_temperature, "topple_weight": args.topple_weight,
            "fall_weight": args.fall_weight, "dr_max_weight": args.dr_max_weight,
            "min_risk_gap": args.min_risk_gap},
        "config": asdict(config), "steps_completed": args.steps,
    }
    (outdir/"summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--counterfactual-dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path)
    p.add_argument("--device", choices=("auto","cuda","cpu"), default="auto")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cf-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=.15)
    p.add_argument("--test-ratio", type=float, default=.15)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dr-thresholds", default="0.0001,0.0005,0.001,0.005,0.01")
    p.add_argument("--ordinal-pos-weight", type=float, default=5.0)
    p.add_argument("--expert-weight", type=float, default=1.0)
    p.add_argument("--cf-weight", type=float, default=0.3)
    p.add_argument("--rank-weight", type=float, default=0.5)
    p.add_argument("--rank-temperature", type=float, default=0.1)
    p.add_argument("--topple-weight", type=float, default=0.5)
    p.add_argument("--fall-weight", type=float, default=1.0)
    p.add_argument("--dr-max-weight", type=float, default=0.25)
    p.add_argument("--min-risk-gap", type=float, default=0.01)
    p.add_argument("--eval-freq", type=int, default=250)
    p.add_argument("--log-freq", type=int, default=50)
    p.add_argument("--val-max-windows", type=int, default=4096)
    p.add_argument("--val-batch-size", type=int, default=256)
    args=p.parse_args(argv)
    summary=train(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__=="__main__":
    main()
