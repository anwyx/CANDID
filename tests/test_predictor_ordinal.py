"""CPU-only regression contracts for the optional ordinal DR predictor."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from model.predictor.config import PredictorConfig
from model.predictor.network import DisturbancePredictor, OrdinalDisturbancePredictor
from model.predictor.normalizer import PredictorNormalizer
from model.predictor.checkpoint import save_checkpoint, load_predictor
from model.guidance.config import GuidanceConfig
from model.guidance.objective import chunk_risk


def make_model():
    return OrdinalDisturbancePredictor(
        obs_dim=3, action_dim=2, obs_horizon=2, action_horizon=8, hidden=16,
        dr_thresholds=(0.0001, 0.001, 0.005, 0.02, 0.05, 0.1),
    )


def normalizer():
    return PredictorNormalizer(
        torch.zeros(3), torch.ones(3), torch.zeros(2), torch.ones(2),
        torch.zeros(2), torch.ones(2), torch.ones(2),
    )


def test_exceedance_probabilities_are_monotone_and_dr_is_bounded():
    model = make_model()
    out = model(torch.randn(4, 2, 3), torch.randn(4, 8, 2))
    assert out["ordinal_logits"].shape == (4, 2, 6)
    probs = torch.sigmoid(out["ordinal_logits"])
    assert torch.all(probs[:, :, :-1] >= probs[:, :, 1:])
    assert torch.all(out["dr"] >= 0)
    assert torch.all(out["dr"] <= 0.1 + 1e-6)


def test_physical_thresholds_and_config_validation():
    config = PredictorConfig(dr_objective="ordinal")
    assert config.dr_thresholds[0] == 0.0001
    with pytest.raises(ValueError):
        PredictorConfig(dr_objective="ordinal", dr_thresholds=(0.01, 0.001))
    with pytest.raises(ValueError):
        PredictorConfig(dr_objective="bad")
    with pytest.raises(ValueError):
        PredictorConfig(dr_objective="ordinal", ordinal_pos_weight=0)


def test_ordinal_loss_and_guidance_are_differentiable():
    model = make_model()
    obs = torch.randn(2, 2, 3)
    act = torch.randn(2, 8, 2, requires_grad=True)
    target = torch.tensor([
        [0.0002, 0.002, 0., 0.],
        [0.03, 0.06, 1., 0.],
    ])
    loss = model.loss(obs, act, target)
    assert torch.isfinite(loss)
    loss.backward(retain_graph=True)
    assert act.grad is not None and torch.isfinite(act.grad).all()
    assert act.grad.abs().sum() > 0
    act.grad = None
    risk = chunk_risk(
        obs, act, model, normalizer(), GuidanceConfig(
            strength=0.005, start_timestep=1,
        ),
    )
    assert risk.shape == (2,) and torch.isfinite(risk).all()
    risk.sum().backward()
    assert act.grad is not None and torch.isfinite(act.grad).all()
    assert act.grad.abs().sum() > 0


def test_ordinal_checkpoint_round_trip_and_old_regression_checkpoint(tmp_path):
    norm = normalizer()
    config = PredictorConfig(
        dr_objective="ordinal", obs_horizon=2, action_horizon=8,
        hidden=16,
    )
    ordinal = make_model()
    path = tmp_path / "ordinal.pt"
    save_checkpoint(path, ordinal, norm, config)
    loaded, _ = load_predictor(path)
    assert isinstance(loaded, OrdinalDisturbancePredictor)
    obs, actions = torch.randn(2, 2, 3), torch.randn(2, 8, 2)
    expected = ordinal(obs, actions)["ordinal_logits"]
    actual = loaded(obs, actions)["ordinal_logits"]
    assert torch.equal(expected, actual)

    old = DisturbancePredictor(
        obs_dim=3, action_dim=2, obs_horizon=2, action_horizon=8, hidden=16,
    )
    old_config = PredictorConfig(obs_horizon=2, action_horizon=8, hidden=16)
    old_path = tmp_path / "regression.pt"
    save_checkpoint(old_path, old, norm, old_config)
    old_loaded, _ = load_predictor(old_path)
    assert isinstance(old_loaded, DisturbancePredictor)
    assert not isinstance(old_loaded, OrdinalDisturbancePredictor)
