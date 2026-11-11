"""Small CPU-only v3 contracts; does not depend on ManiSkill or scene data."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from model.predictor.action_features import absolute_to_delta_q, action_observation_mask
from model.predictor.multihorizon import MultiHorizonConsequencePredictor
from model.predictor.normalizer import PredictorNormalizer
from model.predictor.checkpoint import save_multihorizon_checkpoint, load_predictor
from model.guidance.config import GuidanceConfig
from model.guidance.objective import chunk_risk
from scripts.train_predictor_multihorizon import pair_losses


def make_inputs(batch=3):
    torch.manual_seed(17)
    obs = torch.randn(batch, 2, 151)
    actions = torch.randn(batch, 8, 7) * .1
    return obs, actions


def make_normalizer():
    return PredictorNormalizer(
        torch.zeros(151), torch.ones(151),
        torch.zeros(7), torch.ones(7),
        torch.zeros(2), torch.ones(2), torch.ones(2),
        action_representation="delta_q")


def test_delta_q_is_invertible_preserves_gripper_and_is_differentiable():
    obs, actions = make_inputs()
    actions.requires_grad_(True)
    transformed = absolute_to_delta_q(obs, actions)
    reconstructed = transformed.clone()
    reconstructed[:, :, :6] += obs[:, -1:, :6]
    assert torch.allclose(reconstructed, actions)
    assert torch.equal(transformed[:, :, 6], actions[:, :, 6])
    transformed.sum().backward()
    assert torch.allclose(actions.grad, torch.ones_like(actions))
    np.testing.assert_allclose(
        absolute_to_delta_q(obs.numpy(), actions.detach().numpy()),
        transformed.detach().numpy(), atol=1e-6)


def test_action_mask_only_removes_object_velocities():
    mask = action_observation_mask()
    assert len(mask) == 151
    assert (mask == 0).sum() == 42
    assert (mask[:31] == 1).all()
    assert (mask[-2:] == 1).all()


def test_multi_forward_h4_does_not_see_steps_5_to_8():
    obs, actions = make_inputs()
    model = MultiHorizonConsequencePredictor(hidden=32)
    first = model(obs, actions)
    changed = actions.clone()
    changed[:, 4:, :] += 10.
    second = model(obs, changed)
    assert torch.allclose(first["multi_dr"][:, 0],
                          second["multi_dr"][:, 0], atol=1e-7)
    assert torch.allclose(first["multi_event_logits"][:, 0],
                          second["multi_event_logits"][:, 0], atol=1e-7)
    assert not torch.allclose(first["multi_ordinal_logits"][:, 1],
                              second["multi_ordinal_logits"][:, 1])


def test_risk_guidance_gradient_is_finite_nonzero_and_action_only():
    obs, actions = make_inputs()
    model = MultiHorizonConsequencePredictor(hidden=32)
    norm = make_normalizer()
    actions.requires_grad_(True)
    risk = chunk_risk(obs, actions, model, norm,
                      GuidanceConfig(strength=.02, topple_weight=.5))
    assert risk.shape == (len(obs),)
    risk.sum().backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum().item() > 0


def test_multi_loss_and_rank_direction():
    torch.manual_seed(4)
    obs, actions = make_inputs(batch=2)
    model = MultiHorizonConsequencePredictor(hidden=32)
    outputs = model(obs, actions)
    targets = torch.zeros(2, 4, 4)
    targets[..., 0] = .01
    targets[..., 1] = .02
    loss = model.multi_loss(outputs, targets)
    assert torch.isfinite(loss)
    loss.backward()
    # For two candidates in one state, correct ordering should have
    # strictly smaller softplus ranking loss than reversed ordering.
    truth = torch.zeros(1, 2, 4, 4)
    truth[:, 0, :, :2] = .01
    truth[:, 1, :, :2] = .5
    def fake(pred_values):
        dr = torch.zeros(2, 4, 2)
        dr[:, :, 0] = pred_values[:, None]
        dr[:, :, 1] = pred_values[:, None]
        logits = torch.full((2, 4, 2), -10.)
        return {"multi_dr":dr, "multi_event_logits":logits}
    weights = model.horizon_weights
    good = pair_losses(fake(torch.tensor([.01, .5])), truth, weights)[0]
    bad = pair_losses(fake(torch.tensor([.5, .01])), truth, weights)[0]
    assert good < bad


def test_checkpoint_roundtrip_scene1_v3(tmp_path):
    model = MultiHorizonConsequencePredictor(hidden=32)
    normalizer = make_normalizer()
    path = tmp_path / "best_val.pt"
    save_multihorizon_checkpoint(path, model, normalizer)
    reloaded, norm = load_predictor(path)
    assert norm.action_representation == "delta_q"
    assert reloaded.obs_horizon == 2
    assert reloaded.action_horizon == 8
    obs, action = make_inputs()
    with torch.no_grad():
        a = model_output(model, normalizer, obs, action)
        b = model_output(reloaded, norm, obs, action)
    assert torch.allclose(a["multi_dr"], b["multi_dr"])


def model_output(model, norm, obs, actions):
    return model(norm.normalize_obs(obs),
                 norm.normalize_action(actions, obs=obs))
