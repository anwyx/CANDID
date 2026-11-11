"""Paired guidance traces. No simulator and no training."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from model.guidance.figures import save_comparison_figures
from model.guidance.inspect import inspect_scene, observation_normalizer_from_checkpoint
from model.guidance.policy import ConsequenceGuidedPolicy
from model.predictor.normalizer import PredictorNormalizer
from model.vanilla_dp.config import DPConfig
from model.vanilla_dp.normalizer import ActionRangeNormalizer
from model.vanilla_dp.policy import Agent, VanillaDPPolicy
from scripts.inspect_candid_guidance import main


def _policy(strength):
    config = DPConfig(
        obs_horizon=1,
        act_horizon=2,
        pred_horizon=4,
        unet_dims=(8,),
        n_groups=8,
        num_diffusion_iters=4,
        total_iters=1,
        batch_size=1,
        warmup_steps=1,
        log_freq=1,
        eval_freq=1,
        val_max_windows=1,
        val_batch_size=1,
        regularization_max_timestep=0,
    )
    torch.manual_seed(0)
    agent = Agent(4, 7, config).eval()
    dp = VanillaDPPolicy(
        agent,
        ActionRangeNormalizer(torch.zeros(7), torch.ones(7)),
    )
    predictor = _SumPredictor()
    normalizer = PredictorNormalizer(
        torch.zeros(4), torch.ones(4),
        torch.zeros(7), torch.ones(7),
        torch.zeros(2), torch.ones(2),
    )
    policy = ConsequenceGuidedPolicy(
        dp, predictor, normalizer,
        strength=strength, start_timestep=1,
    )
    policy.set_bounds(np.full(7, -2, dtype=np.float32), np.full(7, 2, dtype=np.float32))
    return policy


class _SumPredictor(nn.Module):
    obs_horizon = 1
    obs_dim = 4
    action_horizon = 2
    action_dim = 7

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, obs, actions):
        total = actions.sum(dim=(1, 2)) + self.anchor.reshape(())
        return {
            "dr": torch.stack((total, total), dim=1),
            "event_logits": torch.zeros((actions.shape[0], 2), device=actions.device),
        }


def _observation():
    return np.zeros((1, 4), dtype=np.float32)


def test_cli_requires_scene_and_checkpoints():
    with pytest.raises(SystemExit):
        main([])


def test_missing_suite_names_the_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="scenes.jsonl"):
        inspect_scene(tmp_path, "missing", "dp.pt", "predictor.pt", tmp_path / "out")


def test_strength_zero_matches_and_has_no_nan():
    policy = _policy(0.0)
    policy.trace = True
    policy.begin_episode(7)
    result = policy.compare(_observation())
    assert result["vanilla_clipped"].shape == (2, 7)
    assert result["guided_clipped"].shape == (2, 7)
    assert np.isfinite(result["vanilla_clipped"]).all()
    assert np.isfinite(result["guided_clipped"]).all()
    assert np.allclose(result["vanilla_clipped"], result["guided_clipped"], atol=1e-6)
    assert abs(result["delta_dr_mean"]) < 1e-5
    assert abs(result["delta_dr_max"]) < 1e-5
    assert result["guided_diffusion_steps"] == 0


def test_strength_zero_matches_unmodified_dp_sampler():
    policy = _policy(0.0)
    obs = torch.zeros((1, 1, 4))
    torch.manual_seed(11)
    reference = policy.dp_policy.agent.get_action(obs)
    torch.manual_seed(11)
    sampled, rows = policy.sampler.sample(obs, strength=0.0, start_timestep=1, trace=True)
    expected = [int(step) for step in policy.dp_policy.agent.noise_scheduler.timesteps]
    assert [row["diffusion_timestep"] for row in rows] == expected
    assert all(
        not row["guided"] and row["pred_dr_mean"] is None and row["applied_correction_l2"] == 0.0
        for row in rows
    )
    assert torch.allclose(reference, sampled, atol=1e-6)


def test_positive_strength_records_real_corrections_and_can_change_actions():
    policy = _policy(0.2)
    policy.trace = True
    policy.begin_episode(9)
    result = policy.compare(_observation())
    timesteps = [row["diffusion_timestep"] for row in result["diffusion_trace"]]
    expected = [int(step) for step in policy.dp_policy.agent.noise_scheduler.timesteps]
    assert timesteps == expected
    guided = [row for row in result["diffusion_trace"] if row["guided"]]
    idle = [row for row in result["diffusion_trace"] if not row["guided"]]
    assert guided
    assert all(row["diffusion_timestep"] <= 1 for row in guided)
    assert all(row["pred_dr_mean"] is not None for row in guided)
    assert any(row["applied_correction_l2"] > 0 for row in guided)
    assert all(row["pred_dr_mean"] is None and row["applied_correction_l2"] == 0 for row in idle)
    assert not np.allclose(result["vanilla_clipped"], result["guided_clipped"], atol=1e-5)


def test_pair_recording_does_not_change_the_guided_chunk():
    observation = _observation()
    plain = _policy(0.2)
    plain.begin_episode(13)
    executed = plain.act(observation)
    traced = _policy(0.2)
    traced.record_pair = True
    traced.trace = True
    traced.begin_episode(13)
    recorded = traced.act(observation)
    assert np.allclose(executed, recorded, atol=1e-6)


def test_normalize_obs_without_statistics_is_rejected():
    assert observation_normalizer_from_checkpoint({"config": {}}) is None
    with pytest.raises(ValueError, match="obs_mean"):
        observation_normalizer_from_checkpoint({"config": {"normalize_obs": True}})


def test_no_render_figures_use_real_numbers(tmp_path):
    policy = _policy(0.2)
    policy.trace = True
    policy.begin_episode(2)
    result = policy.compare(_observation())
    result["start_timestep"] = 1
    paths = save_comparison_figures(
        result, tmp_path, scene_id="scene", episode_step=0,
        replan_index=0, strength=0.2, exec_horizon=1,
    )
    for path in paths.values():
        assert Path(path).stat().st_size > 1000
    record = {
        "predicted_local_dr": {
            "vanilla": {"dr_mean": result["scores"]["vanilla"]["dr_mean"]},
            "guided": {"dr_mean": result["scores"]["guided"]["dr_mean"]},
        }
    }
    (tmp_path / "replans.jsonl").write_text(json.dumps(record), encoding="utf-8")
    loaded = json.loads((tmp_path / "replans.jsonl").read_text(encoding="utf-8"))
    assert loaded["predicted_local_dr"]["guided"]["dr_mean"] == result["scores"]["guided"]["dr_mean"]
    np.savez(
        tmp_path / "chunks.npz",
        vanilla_actions=result["vanilla_raw"][None],
        guided_actions=result["guided_raw"][None],
        vanilla_actions_clipped=result["vanilla_clipped"][None],
        guided_actions_clipped=result["guided_clipped"][None],
        replan_indices=np.asarray([0]),
        episode_steps=np.asarray([0]),
    )
    with np.load(tmp_path / "chunks.npz", allow_pickle=False) as archive:
        assert archive["guided_actions_clipped"].shape == (1, 2, 7)
        assert np.isfinite(archive["guided_actions_clipped"]).all()
