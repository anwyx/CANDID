"""Vanilla DP and the predictor train independently on a synthetic collection."""
import ast
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from candid.data.synthetic import ACTION_DIM, write_synthetic_dataset
from candid.data.trajectories import load_trajectories
from candid.data.validate import validate_dataset
from model.predictor.checkpoint import load_predictor
from model.predictor.config import PredictorConfig
from model.predictor.inference import predict
from model.predictor.train import train_predictor
from model.vanilla_dp.checkpoint import load_policy
from model.vanilla_dp.config import DPConfig
from model.vanilla_dp.train import train_vanilla_dp


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_models_do_not_import_each_other_or_baselines():
    checks = (
        (ROOT / "model" / "vanilla_dp", ("model.predictor", "baselines")),
        (ROOT / "model" / "predictor", ("model.vanilla_dp", "baselines")),
        (ROOT / "src", ("model", "baselines")),
    )
    for directory, forbidden in checks:
        for path in directory.rglob("*.py"):
            for module in _imported_modules(path):
                for prefix in forbidden:
                    assert not module.startswith(prefix), f"{path} imports {module}"
    source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src").rglob("*.py"))
    assert "nn.Module" not in source
    assert "ConditionalUnet1D" not in source


def test_synthetic_collection_matches_the_training_contract(tmp_path):
    root = write_synthetic_dataset(tmp_path / "data")
    report = validate_dataset(root)
    assert report["num_scenes"] == 3
    assert report["num_trajectories"] == 6
    assert report["num_recorded_hdf5"] == 6
    trajectories = load_trajectories(root)
    assert {item.actions.shape[-1] for item in trajectories} == {ACTION_DIM}
    assert all(item.observations["robot_qpos"].shape[-1] != ACTION_DIM for item in trajectories)


def test_vanilla_dp_trains_and_reloads_without_the_predictor(tmp_path):
    root = write_synthetic_dataset(tmp_path / "data")
    config = DPConfig(total_iters=2, batch_size=8, cuda=False, seed=1)
    result = train_vanilla_dp(root, tmp_path / "dp", config)
    assert result["checkpoint"].is_file()
    assert set(result["train_scenes"]).isdisjoint(result["test_scenes"])
    assert set(result["val_scenes"]).isdisjoint(result["test_scenes"])
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result["losses"])
    policy = load_policy(result["checkpoint"], device="cpu")
    action = policy.act(torch.zeros(policy.obs_horizon, policy.obs_dim).numpy())
    assert action.shape == (config.act_horizon, ACTION_DIM)
    assert torch.isfinite(torch.from_numpy(action)).all()


def test_predictor_trains_and_reloads_without_vanilla_dp(tmp_path):
    root = write_synthetic_dataset(tmp_path / "data")
    config = PredictorConfig(steps=40, device="cpu")
    result = train_predictor(root, tmp_path / "predictor", config)
    assert result["losses"][-1] < result["losses"][0]
    assert result["val_loss"] is not None
    model, normalizer = load_predictor(result["checkpoint"], device="cpu")
    score = predict(
        model,
        normalizer,
        torch.zeros(1, config.obs_horizon, model.obs_dim),
        torch.zeros(1, config.action_horizon, ACTION_DIM),
    )
    assert set(score) == {'dr_mean', 'dr_max', 'topple_prob', 'fall_prob'}
    assert all(value.shape == (1,) and torch.isfinite(value).all() for value in score.values())
