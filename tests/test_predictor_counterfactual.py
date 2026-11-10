"""Fast tests for physically labeled candidate groups and ranking gradients."""
from __future__ import annotations

import json
import numpy as np
import pytest
import torch

from model.predictor.counterfactual import (
    CounterfactualStateDataset, pairwise_physical_ranking_loss,
)
from model.predictor.network import OrdinalDisturbancePredictor


def _fixture(tmp_path):
    expert = tmp_path / "expert"
    expert.mkdir()
    root = tmp_path / "counterfactual"
    root.mkdir()
    (root / "samples").mkdir()
    split = {
        "dataset_sha256": "frozen-sha",
        "train": ["train-a", "train-b"],
        "validation": ["val-c"], "test": ["test-d"],
    }
    targets = np.asarray([
        [0.00, 0.01, 0, 0],
        [0.03, 0.07, 1, 0],
        [0.08, 0.19, 0, 0],
    ], dtype=np.float32)
    np.savez(root / "samples/0.npz",
             obs=np.zeros((2, 5), np.float32),
             actions=np.zeros((3, 8, 2), np.float32),
             targets=targets)
    (root / "states.jsonl").write_text(
        json.dumps({
            "state_id": "state-0", "scene_id": "train-a", "source_split": "train",
            "file": "samples/0.npz",
        }) + "\n")
    (root / "summary.json").write_text(json.dumps({
        "complete": True, "labels_from_simulation": True,
        "source_split": "train", "source_split_sha256": "frozen-sha",
        "source_dataset": str(expert), "obs_horizon": 2,
        "action_horizon": 8, "num_states": 1, "num_candidates_per_state": 3,
    }))
    return expert, root, split


def test_counterfactual_group_loader_and_scene_guard(tmp_path):
    expert, root, split = _fixture(tmp_path)
    dataset = CounterfactualStateDataset(
        root, expert, split, obs_dim=5, action_dim=2)
    assert len(dataset) == 1
    item = dataset[0]
    assert item["obs"].shape == (2, 5)
    assert item["actions"].shape == (3, 8, 2)
    assert item["targets"].shape == (3, 4)
    assert len(CounterfactualStateDataset(
        root, expert, split, allowed_scenes=["train-b"])) == 0
    with pytest.raises(ValueError, match="outside train"):
        CounterfactualStateDataset(root, expert, split, allowed_scenes=["test-d"])
    row_path = root / "states.jsonl"
    row = json.loads(row_path.read_text())
    row["scene_id"] = "test-d"
    row_path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="non-training"):
        CounterfactualStateDataset(root, expert, split)


def test_ranking_loss_monotone_gradient_and_tie_mask():
    model = OrdinalDisturbancePredictor(
        obs_dim=3, action_dim=2, obs_horizon=2, action_horizon=8,
        hidden=16)
    obs = torch.randn(1, 3, 2, 3).reshape(3, 2, 3)
    actions = torch.randn(3, 8, 2)
    out = model(obs, actions)
    labels = torch.tensor([[
        [.001, .002, 0., 0.],
        [.040, .080, 1., 0.],
        [.100, .200, 0., 0.],
    ]])
    predictions = {
        "dr": out["dr"].reshape(1, 3, 2),
        "event_logits": out["event_logits"].reshape(1, 3, 2),
    }
    loss, stats = pairwise_physical_ranking_loss(predictions, labels)
    assert stats["pairs"] == 3
    assert torch.isfinite(loss)
    loss.backward()
    assert model.event_head.weight.grad is not None
    assert torch.isfinite(model.event_head.weight.grad).all()
    tied = labels.clone()
    tied[:] = labels[:, 0:1, :]
    zero, empty = pairwise_physical_ranking_loss(predictions, tied)
    assert empty["pairs"] == 0
    assert float(zero) == 0.0


def test_ranking_matches_known_preference():
    labels = torch.tensor([[
        [.0, .0, 0, 0],
        [.2, .2, 1, 0],
    ]])
    ordered = {
        "dr": torch.tensor([[[0., 0.], [.1, .1]]]),
        "event_logits": torch.tensor([[[-8., -8.], [8., -8.]]]),
    }
    reversed_scores = {
        "dr": torch.flip(ordered["dr"], dims=(1,)),
        "event_logits": torch.flip(ordered["event_logits"], dims=(1,)),
    }
    good, good_stats = pairwise_physical_ranking_loss(ordered, labels)
    bad, bad_stats = pairwise_physical_ranking_loss(reversed_scores, labels)
    assert good_stats["accuracy"] == 1.
    assert bad_stats["accuracy"] == 0.
    assert good < bad
