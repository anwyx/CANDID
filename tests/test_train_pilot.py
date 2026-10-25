"""Run the training pipeline on a synthetic collection. No simulator required."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candid.data.validate import validate_dataset
from candid.train.dataset import WindowDataset, load_trajectories, split_scene_ids
from candid.train.loop import load_diffusion_policy, load_predictor, run_pilot
from candid.train.synthetic import ACTION_DIM, write_synthetic_dataset


def test_synthetic_collection_matches_the_training_contract(tmp_path):
    root = write_synthetic_dataset(tmp_path / "data")
    report = validate_dataset(root)
    assert report["num_scenes"] == 3
    assert report["num_trajectories"] == 6
    assert report["num_recorded_hdf5"] == 6

    trajectories = load_trajectories(root)
    assert {item.actions.shape[-1] for item in trajectories} == {ACTION_DIM}
    assert all(item.observations["robot_qpos"].shape[-1] != ACTION_DIM for item in trajectories)
    train_scenes, val_scenes = split_scene_ids([item.scene_id for item in trajectories], seed=0)
    assert set(train_scenes).isdisjoint(val_scenes)
    assert set(train_scenes) | set(val_scenes) == {item.scene_id for item in trajectories}

    policy = WindowDataset(trajectories, train_scenes, success_only=True)
    predictor = WindowDataset(trajectories, train_scenes, success_only=False)
    assert policy.samples
    assert all(policy.successes)
    assert len(predictor) > len(policy)
    assert any(not success for success in predictor.successes)
    assert policy[0]["obs"].shape[0] == 2
    assert policy[0]["action"].shape == (12, ACTION_DIM)
    assert policy[0]["target"].shape == (4,)


def test_diffusion_policy_and_predictor_train_on_synthetic_data(tmp_path):
    root = write_synthetic_dataset(tmp_path / "data")
    result = run_pilot(root, tmp_path / "run", steps=40, seed=0, diffusion_steps=8)

    assert result["sample_action"].shape == (1, 12, ACTION_DIM)
    assert result["sample_prediction"].shape == (1, 4)
    assert torch.isfinite(result["sample_action"]).all()
    assert torch.isfinite(result["sample_prediction"]).all()
    assert result["policy_losses"][-1] < result["policy_losses"][0]
    assert result["predictor_losses"][-1] < result["predictor_losses"][0]
    assert result["policy_val_loss"] is not None
    assert result["predictor_val_loss"] is not None

    policy, policy_norm = load_diffusion_policy(result["policy_checkpoint"])
    predictor, predictor_norm = load_predictor(result["predictor_checkpoint"])
    raw_obs = torch.zeros(1, 2, policy.obs_dim)
    with torch.no_grad():
        action = policy_norm.denormalize_action(policy.sample(policy_norm.normalize_obs(raw_obs)))
        prediction = predictor_norm.denormalize_target(
            predictor(
                predictor_norm.normalize_obs(raw_obs),
                predictor_norm.normalize_action(action),
            )
        )
    assert action.shape == (1, 12, ACTION_DIM)
    assert prediction.shape == (1, 4)
    assert torch.isfinite(action).all()
    assert torch.isfinite(prediction).all()
