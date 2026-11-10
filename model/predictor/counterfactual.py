"""Physics-labeled same-state counterfactual groups for Predictor training.

Each item is ONE state and all its candidate chunks; ranking pairs must never
cross states. Frozen source training split is validated before loading.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CounterfactualStateDataset(Dataset):
    def __init__(self, root, dataset_root, split, *, obs_horizon=2,
                 action_horizon=8, obs_dim=None, action_dim=None):
        self.root = Path(root)
        summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        if (summary.get("complete") is not True or summary.get("labels_from_simulation") is not True
                or summary.get("source_split") != "train"):
            raise ValueError("Only complete, physically relabeled TRAIN counterfactual data are allowed")
        if summary.get("source_split_sha256") != split["dataset_sha256"]:
            raise ValueError("Counterfactual source split differs from the expert dataset")
        if Path(summary.get("source_dataset", "")).resolve() != Path(dataset_root).resolve():
            raise ValueError("Counterfactual source dataset path disagrees with expert dataset")
        if summary.get("obs_horizon") != obs_horizon or summary.get("action_horizon") != action_horizon:
            raise ValueError("Counterfactual horizon differs from predictor")
        allowed = set(split["train"])
        with (self.root / "states.jsonl").open(encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        if not self.rows or len(self.rows) != summary["num_states"]:
            raise ValueError("Counterfactual state count differs from summary")
        if any(row.get("source_split") != "train" or row["scene_id"] not in allowed
               for row in self.rows):
            raise ValueError("Counterfactual data include non-training scenes")
        ids = [row["state_id"] for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate counterfactual state IDs")
        self.num_candidates = int(summary["num_candidates_per_state"])
        if self.num_candidates < 2:
            raise ValueError("Counterfactual groups need at least two candidate actions")
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # Fail before any expensive predictor training if one group is malformed.
        for row in self.rows:
            self._load(row)

    def _load(self, row):
        location = Path(row["file"])
        if location.is_absolute() or ".." in location.parts:
            raise ValueError("Counterfactual sample path must be inside dataset root")
        with np.load(self.root / location, allow_pickle=False) as archive:
            obs = np.asarray(archive["obs"], dtype=np.float32)
            actions = np.asarray(archive["actions"], dtype=np.float32)
            targets = np.asarray(archive["targets"], dtype=np.float32)
        if (obs.ndim != 2 or obs.shape[0] != self.obs_horizon
                or (self.obs_dim is not None and obs.shape[1] != self.obs_dim)
                or actions.ndim != 3
                or actions.shape[:2] != (self.num_candidates, self.action_horizon)
                or (self.action_dim is not None and actions.shape[2] != self.action_dim)
                or targets.shape != (self.num_candidates, 4)):
            raise ValueError(f"Malformed counterfactual group {row['state_id']}")
        if (not np.isfinite(obs).all() or not np.isfinite(actions).all()
                or not np.isfinite(targets).all()
                or (targets[:, :2] < 0).any()
                or not np.isin(targets[:, 2:], (0., 1.)).all()):
            raise ValueError(f"Nonphysical or missing labels in {row['state_id']}")
        return obs, actions, targets

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        obs, actions, targets = self._load(self.rows[index])
        return {
            "obs": torch.from_numpy(obs.copy()),
            "actions": torch.from_numpy(actions.copy()),
            "targets": torch.from_numpy(targets.copy()),
        }


def pairwise_physical_ranking_loss(predictions, targets, *, temperature=0.1,
                                   topple_weight=0.5, fall_weight=1.0,
                                   dr_max_weight=0.25, min_risk_gap=0.01):
    """Compare ONLY actions from the same physically replayed state.

    targets: [B,K,4] in physical units (dr_mean,dr_max,topple,fall).
    predictions: the same group's model outputs, indexed [B,K,...].
    Real labels define the direction; the differentiable *same formula* with
    predicted DR and event probabilities supplies gradient to the predictor.
    Weak/tied target pairs are excluded. Each eligible state has equal weight.
    """
    if targets.ndim != 3 or targets.shape[2] != 4 or targets.shape[1] < 2:
        raise ValueError("Ranking targets must have shape [B,K>=2,4]")
    if temperature <= 0 or min_risk_gap < 0 or min(
            topple_weight, fall_weight, dr_max_weight) < 0:
        raise ValueError("Invalid ranking scales")
    n, k, _ = targets.shape
    dr, logits = predictions["dr"], predictions["event_logits"]
    if dr.shape != (n, k, 2) or logits.shape != (n, k, 2):
        raise ValueError("Predictions must have DR/event shapes [B,K,2]")
    physical = (
        targets[..., 0] + dr_max_weight * targets[..., 1]
        + topple_weight * targets[..., 2] + fall_weight * targets[..., 3]
    )
    probs = torch.sigmoid(logits)
    predicted = (
        dr[..., 0].clamp_min(0)
        + dr_max_weight * torch.maximum(dr[..., 1].clamp_min(0), dr[..., 0].clamp_min(0))
        + topple_weight * probs[..., 0] + fall_weight * probs[..., 1]
    )
    upper = torch.triu(torch.ones(k, k, dtype=torch.bool, device=targets.device), diagonal=1)
    physical_delta = physical[:, :, None] - physical[:, None, :]
    pred_delta = predicted[:, :, None] - predicted[:, None, :]
    valid = upper.unsqueeze(0) & (physical_delta.abs() > min_risk_gap)
    truth_direction = torch.sign(physical_delta).detach()
    per_pair = F.softplus(-truth_direction * pred_delta / temperature)
    selected = torch.where(valid, per_pair, torch.zeros_like(per_pair))
    pairs_per_state = valid.sum(dim=(1, 2))
    meaningful = pairs_per_state > 0
    loss_per_state = selected.sum(dim=(1, 2)) / pairs_per_state.clamp_min(1)
    loss = (
        loss_per_state[meaningful].mean() if meaningful.any()
        else predicted.sum() * 0.0
    )
    correct = ((truth_direction * pred_delta) > 0) & valid
    return loss, {
        "pairs": int(valid.sum().item()),
        "states_with_pairs": int(meaningful.sum().item()),
        "accuracy": float(correct.sum().item() / valid.sum().item()) if valid.any() else None,
    }
