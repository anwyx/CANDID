"""Regression tests for on-policy, executed-prefix Predictor supervision."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from model.predictor.dataset import PredictorWindowDataset
from model.predictor.network import DisturbancePredictor
from model.predictor.mixed_sampler import MixedTrajectoryBatchSampler
from model.predictor.policy_rollout_dataset import PolicyRolloutWindowDataset


def _rollout(tmp_path, split="train", scene_id="scene_train"):
    run = tmp_path / "dp_train"
    episode = run / "000_a_rep00"
    episode.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"split_name": split}))
    record = {
        "scene_id": scene_id,
        "trajectory_file": str(episode / "trajectory.npz"),
    }
    (run / "episodes.jsonl").write_text(json.dumps(record) + "\n")
    T = 12
    poses = np.zeros((T + 1, 1, 7), dtype=np.float32)
    poses[..., 3] = 1.0
    poses[5:, 0, 0] = .02
    np.savez_compressed(
        episode / "trajectory.npz",
        observations=np.zeros((T+1, 3), dtype=np.float32),
        actions=np.ones((T, 2), dtype=np.float32),
        distractor_poses=poses,
        object_sizes_m=np.asarray([.1], dtype=np.float32),
        distractor_toppled=np.zeros((T+1, 1), dtype=bool),
        distractor_fallen=np.zeros((T+1, 1), dtype=bool),
    )
    return run


def test_policy_rollouts_reject_test_and_unapproved_scenes(tmp_path):
    test_path = _rollout(tmp_path, split="test")
    with pytest.raises(ValueError, match="train"):
        PolicyRolloutWindowDataset(test_path, ["scene_train"])
    (test_path / "summary.json").write_text(json.dumps({"split_name": "train"}))
    with pytest.raises(ValueError, match="non-training"):
        PolicyRolloutWindowDataset(test_path, ["scene_other"])


def test_policy_rollout_labels_only_cover_executed_prefix(tmp_path):
    run = _rollout(tmp_path)
    d = PolicyRolloutWindowDataset(
        run, ["scene_train"], obs_horizon=2,
        action_horizon=8, label_horizon=4,
    )
    assert len(d) == 4
    x = d[0]
    assert x["action"].shape == (8, 2)
    assert x["obs"].shape == (2, 3)
    # At t=1, displacement starts at pose 5 (4 steps after t),
    # while an 8-step label would also include this displacement.
    assert x["target"][0].item() > 0


def test_predictor_prediction_ignores_unused_action_suffix():
    model = DisturbancePredictor(
        obs_dim=3, action_dim=2, action_horizon=8, label_horizon=4,
        hidden=16,
    )
    obs = torch.randn(1, 2, 3)
    actions = torch.randn(1, 8, 2)
    modified = actions.clone()
    modified[:, 4:] += 200.0
    a = model(obs, actions)
    b = model(obs, modified)
    for key in ("dr", "event_logits"):
        assert torch.equal(a[key], b[key])


def test_predictor_local_label_responds_to_first_four_actions():
    model = DisturbancePredictor(
        obs_dim=3, action_dim=2, action_horizon=8, label_horizon=4,
        hidden=16,
    )
    obs = torch.randn(1, 2, 3)
    actions = torch.randn(1, 8, 2, requires_grad=True)
    result = model(obs, actions)["dr"].sum()
    grad = torch.autograd.grad(result, actions)[0]
    assert torch.all(grad[:, 4:] == 0)
    assert torch.isfinite(grad[:, :4]).all()


def test_mixed_sampler_is_balanced_and_trajectory_local():
    class Expert:
        indices = np.asarray([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3]], dtype=np.int32)
        def __len__(self):
            return len(self.indices)

    class Policy:
        indices = [(0, 1), (0, 2), (1, 1), (1, 2)]
        def __len__(self):
            return len(self.indices)

    sampler = MixedTrajectoryBatchSampler(Expert(), Policy(), batch_size=6, seed=7)
    for batch in sampler:
        expert = [i for i in batch if i < 6]
        policy = [i for i in batch if i >= 6]
        assert len(expert) == len(policy) == 3
        assert len({0 if i < 3 else 1 for i in expert}) == 1
        assert len({0 if i < 8 else 1 for i in policy}) == 1
