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
from model.predictor.dataset import PredictorWindowDataset, MixedTrajectoryBatchSampler
from model.predictor.network import OrdinalDisturbancePredictor
from model.predictor.normalizer import fit_predictor_normalizer
from model.predictor.train import _mean_loss, _validation_subset, _select_device


def _supervised(model, normalizer, obs, actions, targets, event_mask,
                event_pos_weight=None):
    return model.loss(
        normalizer.normalize_obs(obs), normalizer.normalize_action(actions),
        targets, event_mask,
        normalizer.event_pos_weight if event_pos_weight is None else event_pos_weight,
    )


def _counterfactual_losses(model, normalizer, batch, *, temperature,
                           topple_weight, fall_weight, dr_max_weight,
                           min_risk_gap, cf_pos_weight_cap=None):
    obs = batch["obs"]
    actions = batch["actions"]
    labels = batch["targets"]
    n, k, h, a = actions.shape
    histories = obs[:, None].expand(-1, k, -1, -1)
    flat_obs = histories.reshape(n*k, obs.shape[1], obs.shape[2])
    flat_act = actions.reshape(n*k, h, a)
    flat_targets = labels.reshape(n*k, 4)
    pos_weight = (normalizer.event_pos_weight
                  if cf_pos_weight_cap is None else
                  normalizer.event_pos_weight.clamp(max=cf_pos_weight_cap))
    supervised = _supervised(
        model, normalizer, flat_obs, flat_act, flat_targets,
        torch.ones((n*k, 2), device=obs.device, dtype=obs.dtype),
        event_pos_weight=pos_weight)
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



@torch.no_grad()
def _cf_validation(model, normalizer, loader, device, args):
    model.eval()
    total_cf, total_rank, total_groups = 0.0, 0.0, 0
    total_pairs, weighted_correct = 0, 0.0
    for batch in loader:
        batch = {key: val.to(device) for key, val in batch.items()}
        supervised, rank, stats = _counterfactual_losses(
            model, normalizer, batch,
            temperature=args.rank_temperature,
            topple_weight=args.topple_weight,
            fall_weight=args.fall_weight,
            dr_max_weight=args.dr_max_weight,
            min_risk_gap=args.min_risk_gap,
            cf_pos_weight_cap=args.cf_pos_weight_cap)
        groups = batch["obs"].shape[0]
        total_groups += groups
        total_cf += float(supervised) * groups
        total_rank += float(rank) * groups
        total_pairs += stats["pairs"]
        if stats["accuracy"] is not None:
            weighted_correct += stats["accuracy"] * stats["pairs"]
    return {
        "cf_loss": total_cf / total_groups if total_groups else None,
        "ranking_loss": total_rank / total_groups if total_groups else None,
        "ranking_accuracy": weighted_correct / total_pairs if total_pairs else None,
        "ranking_pairs": total_pairs,
        "validation_states": total_groups,
    }


def train(args):
    if not 0 <= args.cf_val_ratio < 1:
        raise ValueError("--cf-val-ratio must be in [0,1)")
    if (args.cf_update_every < 1 or args.weight_decay < 0
            or args.cf_pos_weight_cap < 1 or args.early_stop_patience < 0):
        raise ValueError("Invalid CF cadence, positive-weight cap, weight decay, or patience")
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
    val_set = (PredictorWindowDataset(
        dataset_root, split["validation"], obs_horizon=2, action_horizon=8,
        allow_missing_event_labels=False)
        if split["validation"] else None)
    fixed_val = _validation_subset(val_set, args.val_max_windows, args.seed)
    cf = CounterfactualStateDataset(
        args.counterfactual_dataset, dataset_root, split,
        obs_horizon=2, action_horizon=8,
        allow_incomplete=args.allow_incomplete_counterfactual)

    cf_scenes = sorted({row["scene_id"] for row in cf.rows})
    permutation = torch.randperm(
        len(cf_scenes), generator=torch.Generator().manual_seed(args.seed)).tolist()
    n_cf_val = (max(1, round(len(cf_scenes) * args.cf_val_ratio))
                if args.cf_val_ratio > 0 and len(cf_scenes) >= 2 else 0)
    n_cf_val = min(n_cf_val, len(cf_scenes)-1)
    cf_val_scene_ids = {cf_scenes[i] for i in permutation[:n_cf_val]}
    cf_train_indices = [i for i, row in enumerate(cf.rows)
                        if row["scene_id"] not in cf_val_scene_ids]
    cf_val_indices = [i for i, row in enumerate(cf.rows)
                      if row["scene_id"] in cf_val_scene_ids]
    cf_train = Subset(cf, cf_train_indices)
    cf_val = Subset(cf, cf_val_indices) if cf_val_indices else None
    if not cf_train_indices:
        raise ValueError("No counterfactual training states after scene holdout")
    # A scene held out for counterfactual ranking validation must also be
    # excluded from the expert branch. Otherwise the model has already seen
    # its expert action windows during training, overstating generalization.
    expert_train_scene_ids = [
        sid for sid in split["train"] if sid not in cf_val_scene_ids
    ]
    if not expert_train_scene_ids:
        raise ValueError("Counterfactual holdout exhausted the expert training split")
    train_set = PredictorWindowDataset(
        dataset_root, expert_train_scene_ids, obs_horizon=2, action_horizon=8,
        allow_missing_event_labels=False, cache_size=8)
    cf_first = cf[0]
    if (train_set.obs_dim != cf_first["obs"].shape[-1]
            or train_set.action_dim != cf_first["actions"].shape[-1]):
        raise ValueError("Counterfactual and expert action/observation dimensions differ")
    print(f"counterfactual train={len(cf_train)} states / "
          f"validation={len(cf_val_indices)} states "
          f"({len(cf_val_scene_ids)} held-out scenes)", flush=True)
    print(f"expert train={len(expert_train_scene_ids)} scenes "
          f"excluding CF held-out scenes", flush=True)
    if args.init_checkpoint and cf_val_scene_ids:
        print("NOTE: pretrained checkpoint may have seen these CF validation "
              "scenes previously; strict unseen-scene validation requires "
              "training from scratch.", flush=True)
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
        batch_sampler=MixedTrajectoryBatchSampler(
            train_set, min(args.batch_size, len(train_set)),
            seed=args.seed, trajectories_per_batch=args.expert_trajectories_per_batch),
        num_workers=0,
    )
    cf_loader = DataLoader(
        cf_train, batch_size=min(args.cf_batch_size, len(cf_train)),
        shuffle=True, num_workers=0)
    cf_val_loader = (DataLoader(
        cf_val, batch_size=min(args.cf_batch_size, len(cf_val)),
        shuffle=False, num_workers=0) if cf_val is not None else None)
    expert_iter = iter(expert_loader)
    cf_iter = iter(cf_loader)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoints = outdir/"checkpoints"
    best_val = float("inf")
    best_step = None
    last_val = None
    last_cf_validation = None
    best_expert_validation = None
    best_cf_validation = None
    epochs_without_improvement = 0
    log_path = outdir/"train_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as log:
        writer = csv.DictWriter(log, fieldnames=(
            "step", "total_loss", "expert_loss", "cf_loss", "ranking_loss",
            "ranking_accuracy", "ranking_pairs", "validation_expert_loss",
            "validation_cf_loss", "validation_ranking_loss",
            "validation_ranking_accuracy", "validation_ranking_pairs", "selection_score"))
        writer.writeheader()
        for step in range(1, args.steps+1):
            try:
                expert = next(expert_iter)
            except StopIteration:
                expert_iter = iter(expert_loader)
                expert = next(expert_iter)
            use_cf = (step % args.cf_update_every == 0)
            if use_cf:
                try:
                    replay = next(cf_iter)
                except StopIteration:
                    cf_iter = iter(cf_loader)
                    replay = next(cf_iter)
                replay = {key: val.to(device) for key, val in replay.items()}
            expert = {key: val.to(device) for key, val in expert.items()}
            model.train()
            expert_loss = _supervised(
                model, normalizer, expert["obs"], expert["action"],
                expert["target"], expert["event_mask"])
            if use_cf:
                cf_loss, ranking, ranking_stats = _counterfactual_losses(
                    model, normalizer, replay,
                    temperature=args.rank_temperature,
                    topple_weight=args.topple_weight,
                    fall_weight=args.fall_weight,
                    dr_max_weight=args.dr_max_weight,
                    min_risk_gap=args.min_risk_gap,
                    cf_pos_weight_cap=args.cf_pos_weight_cap)
            else:
                cf_loss = expert_loss.new_zeros(())
                ranking = expert_loss.new_zeros(())
                ranking_stats = {"pairs": 0, "accuracy": None}
            loss = (args.expert_weight*expert_loss
                    + args.cf_weight*cf_loss + args.rank_weight*ranking)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            evaluate = step % args.eval_freq == 0 or step == args.steps
            selection_score = None
            if evaluate:
                if fixed_val is not None:
                    last_val = _mean_loss(
                        model, normalizer, fixed_val, device,
                        batch_size=args.val_batch_size)
                if cf_val_loader is not None:
                    last_cf_validation = _cf_validation(
                        model, normalizer, cf_val_loader, device, args)
                # Do not use training ranking accuracy for checkpoint selection.
                # Expert scene validation preserves prior calibration; separate
                # CF scene validation tests whether candidate ordering generalizes.
                selection_score = (float(last_val) * args.expert_weight
                                   if last_val is not None else 0.0)
                if last_cf_validation is not None:
                    selection_score += (
                        (args.cf_weight / args.cf_update_every)
                        * last_cf_validation["cf_loss"]
                        + (args.rank_weight / args.cf_update_every)
                        * last_cf_validation["ranking_loss"])
                if selection_score < best_val:
                    best_val = selection_score
                    best_step = step
                    best_expert_validation = last_val
                    best_cf_validation = (dict(last_cf_validation)
                                          if last_cf_validation is not None else None)
                    epochs_without_improvement = 0
                    save_checkpoint(checkpoints/"best_val.pt", model, normalizer, config)
                else:
                    epochs_without_improvement += 1
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
                    "validation_cf_loss": (
                        last_cf_validation["cf_loss"]
                        if evaluate and last_cf_validation is not None else ""),
                    "validation_ranking_loss": (
                        last_cf_validation["ranking_loss"]
                        if evaluate and last_cf_validation is not None else ""),
                    "validation_ranking_accuracy": (
                        last_cf_validation["ranking_accuracy"]
                        if evaluate and last_cf_validation is not None else ""),
                    "validation_ranking_pairs": (
                        last_cf_validation["ranking_pairs"]
                        if evaluate and last_cf_validation is not None else ""),
                    "selection_score": selection_score if evaluate else "",
                })
                log.flush()
                print(
                    f"step {step}/{args.steps} "
                    f"loss={loss.item():.5f} expert={expert_loss.item():.5f} "
                    f"cf={cf_loss.item():.5f} ranking={ranking.item():.5f} "
                    f"pairs={ranking_stats['pairs']} acc={ranking_stats['accuracy']} "
                    f"val_expert={last_val if evaluate else 'NA'} "
                    f"val_rank_acc={last_cf_validation['ranking_accuracy'] if evaluate and last_cf_validation else 'NA'}",
                    flush=True)
            if (args.early_stop_patience > 0 and evaluate
                    and epochs_without_improvement >= args.early_stop_patience):
                print(f"Early stop after {epochs_without_improvement} "
                      f"validation checks without improvement at step {step}.",
                      flush=True)
                break
    save_checkpoint(checkpoints/"last.pt", model, normalizer, config)
    summary = {
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "best_checkpoint": str(checkpoints/"best_val.pt"),
        "last_checkpoint": str(checkpoints/"last.pt"),
        "best_step": best_step, "best_validation_selection_score": best_val,
        "best_expert_validation_loss": best_expert_validation,
        "best_cf_validation": best_cf_validation,
        "training_expert_windows": len(train_set),
        "counterfactual_states": len(cf),
        "cf_collection_complete": cf.collection_complete,
        "cf_original_requested_states": cf.original_requested_states,
        "cf_original_collected_states": cf.original_collected_states,
        "cf_original_stratum_shortfall": cf.original_stratum_shortfall,
        "counterfactual_training_states": len(cf_train),
        "counterfactual_validation_states": len(cf_val_indices),
        "counterfactual_validation_scene_ids": sorted(cf_val_scene_ids),
        "counterfactual_candidates_per_state": cf.num_candidates,
        "counterfactual_source": str(args.counterfactual_dataset),
        "validation_expert_windows": len(fixed_val) if fixed_val else 0,
        "last_expert_validation_loss": last_val,
        "last_cf_validation": last_cf_validation,
        "train_scene_ids": expert_train_scene_ids,
        "counterfactual_heldout_train_scene_ids": sorted(cf_val_scene_ids),
        "validation_scene_ids": split["validation"],
        "test_scene_ids": split["test"], "weights": {
            "expert": args.expert_weight, "counterfactual": args.cf_weight,
            "ranking": args.rank_weight},
        "ranking": {
            "temperature": args.rank_temperature, "topple_weight": args.topple_weight,
            "fall_weight": args.fall_weight, "dr_max_weight": args.dr_max_weight,
            "min_risk_gap": args.min_risk_gap},
        "optimizer": {"name": "AdamW", "weight_decay": args.weight_decay},
        "training_controls": {
            "cf_update_every": args.cf_update_every,
            "cf_pos_weight_cap": args.cf_pos_weight_cap,
            "early_stop_patience": args.early_stop_patience},
        "config": asdict(config), "steps_completed": step,
    }
    (outdir/"summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--counterfactual-dataset", type=Path, required=True)
    p.add_argument("--allow-incomplete-counterfactual", action="store_true",
                   help="Explicit opt-in to verified but incompletely collected "
                        "counterfactual states; retain original shortfall metadata")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path)
    p.add_argument("--device", choices=("auto","cuda","cpu"), default="auto")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--expert-trajectories-per-batch", type=int, default=4)
    p.add_argument("--cf-batch-size", type=int, default=8)
    p.add_argument("--cf-update-every", type=int, default=1,
                   help="Use a physical CF batch every N expert training steps")
    p.add_argument("--cf-pos-weight-cap", type=float, default=50.0,
                   help="Maximum event positive class weight for CF supervision")
    p.add_argument("--cf-val-ratio", type=float, default=.2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Number of validation checks without improvement; 0 disables")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=.15)
    p.add_argument("--test-ratio", type=float, default=.15)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dr-thresholds", default="0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2,0.5,1.0")
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
