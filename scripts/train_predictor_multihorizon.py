"""Train scene1 v3 Predictor with expert H=8 + real multi-horizon CF labels.

Training labels: action input 8 steps; consequences at 4,8,16,32.
H>8 uses a fixed hold of the last joint target, *not future DP actions*.
No DP rollout data; no changed physics/DR definitions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from candid.data.splits import load_or_create_scene_split
from model.predictor.checkpoint import save_multihorizon_checkpoint
from model.predictor.counterfactual import CounterfactualStateDataset
from model.predictor.dataset import PredictorWindowDataset, MixedTrajectoryBatchSampler
from model.predictor.multihorizon import MultiHorizonConsequencePredictor
from model.predictor.normalizer import fit_predictor_normalizer
from model.predictor.train import _validation_subset, _select_device


class MultiHorizonCFDataset(Dataset):
    def __init__(self, root, dataset_root, split, *, allow_incomplete=False):
        self.base = CounterfactualStateDataset(
            root, dataset_root, split, obs_horizon=2,
            action_horizon=8, obs_dim=151, action_dim=7,
            allow_incomplete=allow_incomplete)
        self.rows = self.base.rows
        summary = json.loads((Path(root) / "summary.json").read_text())
        if (summary.get("consequence_horizons") != [4, 8, 16, 32]
                or summary.get("continuation_policy") != "hold_last_joint_target_after_step_8"):
            raise ValueError("Require scene1 v3 4,8,16,32 hold-protocol CF data")
        for row in self.rows:
            with np.load(Path(root) / row["file"], allow_pickle=False) as f:
                self._validate(f, row)

    @staticmethod
    def _validate(archive, row):
        if ("targets_multi" not in archive
                or archive["targets_multi"].shape != (row["num_candidates"], 4, 4)
                or not np.isfinite(archive["targets_multi"]).all()
                or np.any(archive["targets_multi"][..., :2] < 0)
                or not np.isin(archive["targets_multi"][..., 2:], (0., 1.)).all()
                or not np.array_equal(archive["consequence_horizons"], [4, 8, 16, 32])):
            raise ValueError(f"Invalid physical multi-horizon labels: {row['state_id']}")
        if not np.allclose(archive["targets"], archive["targets_multi"][:, 1], atol=1e-6):
            raise ValueError("H=8 multi labels differ from legacy H=8 physical labels")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        result = self.base[index]
        row = self.rows[index]
        with np.load(self.base.root / row["file"], allow_pickle=False) as archive:
            result["targets_multi"] = torch.from_numpy(
                np.asarray(archive["targets_multi"], dtype=np.float32).copy())
        return result


def score_from_physical(dr, events, horizon_weights, topple_weight=.5,
                        dr_max_weight=.25):
    mean = dr[..., 0].clamp_min(0)
    high = torch.maximum(dr[..., 1].clamp_min(0), mean)
    per_h = mean + dr_max_weight * high + topple_weight * events[..., 0]
    return (per_h * horizon_weights).sum(dim=-1)


def pair_losses(outputs, targets, horizon_weights, *,
                min_gap=.01, temperature=.1, topple_weight=.5):
    """Same-state pairwise ranking and *magnitude* difference regression.

    No ranking is computed between distinct scene states.  Each state with
    qualifying pairs contributes equally to the optimization.
    """
    n, k = targets.shape[:2]
    multi_dr = outputs["multi_dr"].reshape(n, k, 4, 2)
    probs = torch.sigmoid(outputs["multi_event_logits"]).reshape(n, k, 4, 2)
    pred = score_from_physical(
        multi_dr, probs, horizon_weights, topple_weight=topple_weight)
    truth = score_from_physical(
        targets[..., :2], targets[..., 2:], horizon_weights,
        topple_weight=topple_weight)
    truth_delta = truth[:, :, None] - truth[:, None, :]
    pred_delta = pred[:, :, None] - pred[:, None, :]
    upper = torch.triu(torch.ones(k, k, dtype=torch.bool, device=truth.device), diagonal=1)
    valid = upper.unsqueeze(0) & (truth_delta.abs() >= min_gap)
    counts = valid.sum((1, 2))
    active = counts > 0
    if not active.any():
        zero = pred.sum() * 0.
        return zero, zero, 0, None
    direction = torch.sign(truth_delta).detach()
    ranking = F.softplus(-direction * pred_delta / temperature)
    differences = F.smooth_l1_loss(
        pred_delta, truth_delta.detach(), beta=.1, reduction="none")
    rank_per = (ranking * valid).sum((1, 2)) / counts.clamp_min(1)
    diff_per = (differences * valid).sum((1, 2)) / counts.clamp_min(1)
    acc = (((pred_delta * direction > 0) & valid).sum().float()
           / valid.sum().clamp_min(1)).item()
    return rank_per[active].mean(), diff_per[active].mean(), int(valid.sum()), acc


def model_output(model, normalizer, obs, actions):
    return model(normalizer.normalize_obs(obs),
                 normalizer.normalize_action(actions, obs=obs))


def expert_loss(model, normalizer, batch):
    result = model_output(model, normalizer, batch["obs"], batch["action"])
    # The expert contains only an 8-step physical trace label.
    target = batch["target"].new_zeros((len(batch["target"]), 4, 4))
    target[:, 1] = batch["target"]
    mask = target.new_zeros((len(target), 4))
    mask[:, 1] = 1
    return model.multi_loss(result, target, horizon_mask=mask,
                            event_pos_weight=normalizer.event_pos_weight.clamp(max=3))


def cf_losses(model, normalizer, batch, args):
    obs, actions, targets = batch["obs"], batch["actions"], batch["targets_multi"]
    n, k = actions.shape[:2]
    flat_obs = obs[:, None].expand(-1, k, -1, -1).reshape(n*k, 2, 151)
    flat_actions = actions.reshape(n*k, 8, 7)
    outputs = model_output(model, normalizer, flat_obs, flat_actions)
    supervised = model.multi_loss(
        outputs, targets.reshape(n*k, 4, 4),
        event_pos_weight=normalizer.event_pos_weight.clamp(max=3))
    rank, diff, pairs, acc = pair_losses(
        outputs, targets, model.horizon_weights,
        min_gap=args.min_gap, temperature=args.temperature,
        topple_weight=args.topple_weight)
    return supervised, rank, diff, pairs, acc


@torch.no_grad()
def evaluate(model, normalizer, loader, args, device):
    model.eval()
    summary = np.zeros(3, dtype=np.float64)
    pair_count = 0
    correct = 0.
    states = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        supervised, rank, diff, pairs, acc = cf_losses(model, normalizer, batch, args)
        n = len(batch["obs"])
        summary += np.asarray([supervised.item(), rank.item(), diff.item()]) * n
        states += n
        pair_count += pairs
        if acc is not None:
            correct += acc * pairs
    return {"cf": float(summary[0] / states), "rank": float(summary[1] / states),
            "difference": float(summary[2] / states),
            "pairs": pair_count, "ranking_accuracy": correct / pair_count if pair_count else None}


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _select_device(args.device)
    root = Path(args.dataset)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output already exists: {out}")
    if min(args.steps, args.batch_size, args.cf_batch_size, args.eval_freq) < 1:
        raise ValueError("Batch sizes, steps, and eval frequency must be positive")
    split = load_or_create_scene_split(root, seed=args.split_seed)
    cf = MultiHorizonCFDataset(
        args.counterfactual_dataset, root, split,
        allow_incomplete=args.allow_incomplete_counterfactual)
    # Hold out entire CF source scenes, including their expert windows.
    cf_scenes = sorted({r["scene_id"] for r in cf.rows})
    if len(cf_scenes) < 2:
        raise ValueError("Need counterfactual data from at least two training scenes")
    rng = np.random.default_rng(args.seed)
    selected = rng.permutation(len(cf_scenes))
    n_val = min(len(cf_scenes)-1, max(1, round(len(cf_scenes)*args.cf_val_ratio)))
    held_out = {cf_scenes[int(i)] for i in selected[:n_val]}
    cf_train = Subset(cf, [i for i,r in enumerate(cf.rows)
                           if r["scene_id"] not in held_out])
    cf_val = Subset(cf, [i for i,r in enumerate(cf.rows)
                         if r["scene_id"] in held_out])
    train_scenes = [s for s in split["train"] if s not in held_out]
    if not train_scenes or not len(cf_train) or not len(cf_val):
        raise ValueError("Scene split leaves no train/validation examples")
    expert = PredictorWindowDataset(
        root, train_scenes, obs_horizon=2, action_horizon=8,
        allow_missing_event_labels=False, cache_size=8)
    normalizer = fit_predictor_normalizer(
        expert, action_representation="delta_q").to(device)
    net = MultiHorizonConsequencePredictor(
        obs_dim=151, action_dim=7, hidden=args.hidden,
        horizon_weights=tuple(float(x) for x in args.horizon_weights.split(","))).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=.01)
    expert_loader = DataLoader(
        expert, batch_sampler=MixedTrajectoryBatchSampler(
            expert, min(args.batch_size, len(expert)),
            seed=args.seed, trajectories_per_batch=4), num_workers=0)
    cf_loader = DataLoader(cf_train, batch_size=min(args.cf_batch_size, len(cf_train)),
                           shuffle=True, num_workers=0)
    val_loader = DataLoader(cf_val, batch_size=min(args.cf_batch_size, len(cf_val)),
                            shuffle=False, num_workers=0)
    val_expert = PredictorWindowDataset(
        root, split["validation"], obs_horizon=2, action_horizon=8,
        allow_missing_event_labels=False) if split["validation"] else None
    fixed_val = _validation_subset(val_expert, args.val_max_windows, args.seed)
    expert_iter, cf_iter = iter(expert_loader), iter(cf_loader)
    out.mkdir(parents=True, exist_ok=True)
    (out/"checkpoints").mkdir(parents=True, exist_ok=True)
    best, best_step = float("inf"), None
    csv_path = out/"train_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "step", "loss", "expert", "cf", "rank", "difference",
            "ranking_accuracy", "pairs", "val_score", "val_rank_acc"))
        writer.writeheader()
        for step in range(1, args.steps+1):
            try:
                expert_batch = next(expert_iter)
            except StopIteration:
                expert_iter = iter(expert_loader)
                expert_batch = next(expert_iter)
            try:
                cf_batch = next(cf_iter)
            except StopIteration:
                cf_iter = iter(cf_loader)
                cf_batch = next(cf_iter)
            expert_batch = {k:v.to(device) for k,v in expert_batch.items()}
            cf_batch = {k:v.to(device) for k,v in cf_batch.items()}
            net.train()
            e = expert_loss(net, normalizer, expert_batch)
            c, r, d, pairs, acc = cf_losses(net, normalizer, cf_batch, args)
            total = e + args.cf_weight*c + args.rank_weight*r + args.diff_weight*d
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizer.step()
            val_score, val_acc = None, None
            if step % args.eval_freq == 0 or step == args.steps:
                vals = evaluate(net, normalizer, val_loader, args, device)
                val_acc = vals["ranking_accuracy"]
                val_expert_loss = 0.
                if fixed_val:
                    ex_loader = DataLoader(fixed_val, batch_size=256,
                                           shuffle=False, num_workers=0)
                    total_v, n_v = 0., 0
                    with torch.no_grad():
                        for x in ex_loader:
                            x = {k:v.to(device) for k,v in x.items()}
                            total_v += expert_loss(net, normalizer, x).item() * len(x["obs"])
                            n_v += len(x["obs"])
                    val_expert_loss = total_v / n_v
                val_score = (val_expert_loss + args.cf_weight*vals["cf"]
                             + args.rank_weight*vals["rank"]
                             + args.diff_weight*vals["difference"])
                if val_score < best:
                    best, best_step = val_score, step
                    save_multihorizon_checkpoint(
                        out/"checkpoints"/"best_val.pt", net, normalizer)
            if step == 1 or step % args.log_freq == 0 or val_score is not None:
                writer.writerow({"step":step, "loss":total.item(), "expert":e.item(),
                                 "cf":c.item(), "rank":r.item(), "difference":d.item(),
                                 "ranking_accuracy":acc, "pairs":pairs,
                                 "val_score":val_score, "val_rank_acc":val_acc})
                handle.flush()
                print(f"step={step} loss={total.item():.5f} "
                      f"rank_acc={acc} val_score={val_score} val_rank_acc={val_acc}",
                      flush=True)
    save_multihorizon_checkpoint(out/"checkpoints"/"last.pt", net, normalizer)
    summary = {
        "dataset":str(root), "cf_dataset":str(args.counterfactual_dataset),
        "counterfactual_states":len(cf), "cf_train_states":len(cf_train),
        "cf_val_states":len(cf_val),
        "cf_collection_complete":cf.base.collection_complete,
        "cf_original_requested_states":cf.base.original_requested_states,
        "cf_original_collected_states":cf.base.original_collected_states,
        "cf_original_stratum_shortfall":cf.base.original_stratum_shortfall,
        "cf_val_scene_ids":sorted(held_out),
        "expert_train_scenes":len(train_scenes),
        "expert_train_windows":len(expert),
        "best_step":best_step, "best_val_score":best,
        "best_checkpoint":str(out/"checkpoints"/"best_val.pt"),
        "last_checkpoint":str(out/"checkpoints"/"last.pt"),
        "config":vars(args).copy(),
        "risk_horizons":[4,8,16,32],
        "hold_policy":"hold_last_joint_target_after_step_8"}
    summary["config"] = {k:str(v) if isinstance(v,Path) else v
                         for k,v in summary["config"].items()}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,default=Path("demos/scene1"))
    p.add_argument("--counterfactual-dataset",type=Path,required=True)
    p.add_argument("--allow-incomplete-counterfactual",action="store_true",
                   help="Explicitly accept incomplete but physically valid CF "
                        "data; provenance and original strata shortfall retained")
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda","auto"),default="auto")
    p.add_argument("--steps",type=int,default=2000)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--cf-batch-size",type=int,default=8)
    p.add_argument("--cf-val-ratio",type=float,default=.2)
    p.add_argument("--hidden",type=int,default=256)
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--split-seed",type=int,default=0)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--eval-freq",type=int,default=250)
    p.add_argument("--log-freq",type=int,default=50)
    p.add_argument("--val-max-windows",type=int,default=4096)
    p.add_argument("--cf-weight",type=float,default=.3)
    p.add_argument("--rank-weight",type=float,default=.5)
    p.add_argument("--diff-weight",type=float,default=.5)
    p.add_argument("--temperature",type=float,default=.1)
    p.add_argument("--min-gap",type=float,default=.01)
    p.add_argument("--topple-weight",type=float,default=.5)
    p.add_argument("--horizon-weights",default="0.35,0.25,0.25,0.15")
    args=p.parse_args()
    if not 0 < args.cf_val_ratio < 1:
        p.error("--cf-val-ratio must be between 0 and 1")
    return train(args)


if __name__=="__main__":
    main()
