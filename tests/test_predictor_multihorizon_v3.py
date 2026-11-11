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
    # Subtracting and adding FP32 qpos can introduce ~1e-7 roundoff.
    torch.testing.assert_close(reconstructed, actions, rtol=1e-5, atol=5e-7)
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

def test_ablation_checkpoint_and_risk_gradient(tmp_path):
    """H8-only and absolute-q variants must survive checkpoint roundtrip."""
    for representation in ("delta_q", "absolute"):
        for only_h8 in (False, True):
            weights = (0., 1., 0., 0.) if only_h8 else (.35, .25, .25, .15)
            model = MultiHorizonConsequencePredictor(
                hidden=32, horizon_weights=weights)
            model.trained_horizon_mode = "h8_only" if only_h8 else "multi"
            normalizer = PredictorNormalizer(
                torch.zeros(151), torch.ones(151),
                torch.zeros(7), torch.ones(7),
                torch.zeros(2), torch.ones(2), torch.ones(2),
                action_representation=representation)
            path = tmp_path / (representation + ("_h8" if only_h8 else "_multi") + ".pt")
            save_multihorizon_checkpoint(path, model, normalizer)
            restored, loaded_norm = load_predictor(path)
            assert loaded_norm.action_representation == representation
            assert restored.trained_horizon_mode == model.trained_horizon_mode
            torch.testing.assert_close(
                restored.horizon_weights, torch.tensor(weights))
            obs = torch.randn(2, 2, 151)
            actions = torch.randn(2, 8, 7, requires_grad=True)
            risk = chunk_risk(
                obs, actions, restored, loaded_norm,
                GuidanceConfig(strength=.02, topple_weight=.5))
            risk.sum().backward()
            assert actions.grad is not None
            assert torch.isfinite(actions.grad).all()
            assert actions.grad.abs().sum() > 0


def test_h8_ablation_excludes_all_other_horizons():
    """Changing H4/H16/H32 labels must not change H8-only losses."""
    model = MultiHorizonConsequencePredictor(hidden=32, horizon_weights=(0, 1, 0, 0))
    obs, actions = make_inputs(batch=2)
    out = model(obs, actions)
    truth = torch.zeros(2, 4, 4)
    truth[:, 1, :2] = .1
    mask = torch.zeros(2, 4)
    mask[:, 1] = 1
    original = model.multi_loss(out, truth, horizon_mask=mask)
    changed = truth.clone()
    changed[:, 0, :2] = .3
    changed[:, 2, :2] = .9
    changed[:, 3, :2] = .7
    changed[:, (0, 2, 3), 2:] = 1.
    torch.testing.assert_close(
        original, model.multi_loss(out, changed, horizon_mask=mask))
    pairs = torch.zeros(1, 2, 4, 4)
    pairs[:, 1, 1, :2] = .5
    pred_dr = torch.zeros(2, 4, 2)
    pred_dr[1, 1, :] = .5
    pred = {"multi_dr": pred_dr, "multi_event_logits": torch.full((2, 4, 2), -10.)}
    rank, diff, count, _ = pair_losses(pred, pairs, model.horizon_weights)
    other = pairs.clone()
    other[:, :, 0, :2] = 1.
    other[:, :, 2, :2] = .7
    other[:, :, 3, :2] = .8
    rank2, diff2, count2, _ = pair_losses(pred, other, model.horizon_weights)
    assert count == count2
    torch.testing.assert_close(rank, rank2)
    torch.testing.assert_close(diff, diff2)
