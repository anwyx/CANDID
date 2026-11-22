"""CPU-only integration contract tests; no ManiSkill rollout required."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from model.integration.guided_policy import PredictorGuidedPolicy


class StubDP:
    obs_horizon = 2
    obs_dim = 3
    action_horizon = 8
    action_dim = 2

    def __init__(self, values=(0.5, 0.1)):
        self.values = values
        self.calls = 0
        self.seed = None

    def begin_episode(self, seed):
        self.calls = 0
        self.seed = seed

    def act(self, observation_window):
        value = self.values[self.calls % len(self.values)]
        self.calls += 1
        return np.full((8, 2), value, dtype=np.float32)


class StubPredictor(torch.nn.Module):
    obs_horizon = 2
    obs_dim = 3
    action_horizon = 8
    action_dim = 2

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, obs, action):
        risk = action.mean(dim=(1, 2)) + self.dummy
        dr = torch.stack((risk, risk * 2), dim=1)
        logits = torch.zeros((len(action), 2), dtype=torch.float32, device=action.device)
        return {"dr": dr, "event_logits": logits}


class StubNormalizer:
    def normalize_obs(self, value):
        return value

    def normalize_action(self, value):
        return value

    def denormalize_target(self, value):
        return value


class StubActionSpace:
    shape = (2,)
    low = np.array([-1.0, -1.0], dtype=np.float32)
    high = np.array([1.0, 1.0], dtype=np.float32)


def _policy(dp=None, **kwargs):
    return PredictorGuidedPolicy(
        dp if dp is not None else StubDP(), StubPredictor(), StubNormalizer(), **kwargs
    )


def test_shadow_returns_exact_dp_actions_without_extra_sampling():
    baseline = StubDP()
    baseline.begin_episode(123)
    expected = baseline.act(np.zeros((2, 3), dtype=np.float32))
    dp = StubDP()
    guided = _policy(dp=dp, mode="shadow", num_candidates=1)
    guided.set_action_space(StubActionSpace())
    guided.begin_episode(123)
    actual = guided.act(np.zeros((2, 3), dtype=np.float32))
    assert np.array_equal(expected, actual)
    assert dp.calls == 1
    assert dp.seed == 123
    info = guided.get_diagnostics()
    assert info["candidate_changed_count"] == 0
    assert info["decisions"][0]["selected"] == 0


def test_rerank_selects_low_risk_candidate_within_deviation_limit():
    dp = StubDP(values=(0.5, 0.1))
    guided = _policy(
        dp=dp, mode="rerank", num_candidates=2,
        min_improvement=0.05, max_joint_deviation=0.5,
    )
    guided.set_action_space(StubActionSpace())
    guided.begin_episode(5)
    actual = guided.act(np.zeros((2, 3), dtype=np.float32))
    assert np.allclose(actual, 0.1)
    assert dp.calls == 2
    assert guided.get_diagnostics()["candidate_changed_count"] == 1


def test_rerank_conservatively_keeps_first_action():
    policy = _policy(
        mode="rerank", num_candidates=2,
        min_improvement=0.05, max_joint_deviation=0.2,
    )
    policy.set_action_space(StubActionSpace())
    action = policy.act(np.zeros((2, 3), dtype=np.float32))
    assert np.allclose(action, 0.5)
    assert policy.get_diagnostics()["candidate_changed_count"] == 0


def test_bounds_are_used_for_scoring_but_shadow_returns_original_action():
    dp = StubDP(values=(2.0,))
    guided = _policy(dp=dp, mode="shadow", num_candidates=1)
    guided.set_action_space(StubActionSpace())
    action = guided.act(np.zeros((2, 3), dtype=np.float32))
    assert np.allclose(action, 2.0)
    record = guided.get_diagnostics()["decisions"][0]
    assert record["action_clipping_needed"] == [True]
    assert record["predictions"][0]["dr_mean"] == pytest.approx(1.0)


def test_dimension_horizon_bounds_and_candidate_fail_fast():
    class WrongHorizon(StubPredictor):
        action_horizon = 4

    with pytest.raises(ValueError, match="checkpoint dimensions"):
        PredictorGuidedPolicy(StubDP(), WrongHorizon(), StubNormalizer())
    with pytest.raises(ValueError, match="Shadow mode"):
        _policy(mode="shadow", num_candidates=2)
    guided = _policy(mode="rerank", num_candidates=2)
    with pytest.raises(RuntimeError, match="environment bounds"):
        guided.act(np.zeros((2, 3), dtype=np.float32))
    guided.set_action_space(StubActionSpace())
    with pytest.raises(ValueError, match="observation"):
        guided.act(np.zeros((3, 3), dtype=np.float32))


def test_real_dp_shadow_preserves_rng_sequence_without_predictor_sampling():
    from model.vanilla_dp.config import DPConfig
    from model.vanilla_dp.policy import Agent, VanillaDPPolicy
    from model.vanilla_dp.normalizer import ActionRangeNormalizer

    config = DPConfig()
    agent = Agent(3, 2, config)
    normalizer = ActionRangeNormalizer(torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]))
    dp = VanillaDPPolicy(agent, normalizer)
    fake_predictor = StubPredictor()
    guided = PredictorGuidedPolicy(dp, fake_predictor, StubNormalizer(), mode="shadow")
    obs = np.zeros((2, 3), dtype=np.float32)
    dp.begin_episode(42)
    expected = dp.act(obs)
    guided.set_action_space(StubActionSpace())
    guided.begin_episode(42)
    actual = guided.act(obs)
    assert np.array_equal(expected, actual)


class RandomTorchDP(StubDP):
    """Sample actions from the same global torch RNG used by Vanilla DP."""

    def begin_episode(self, seed):
        super().begin_episode(seed)
        torch.manual_seed(int(seed))

    def act(self, observation_window):
        self.calls += 1
        return (torch.randn(8, 2) * 0.1).numpy().astype(np.float32)


def test_rerank_no_selection_preserves_multiple_replans_and_torch_rng():
    obs = np.zeros((2, 3), dtype=np.float32)
    seed = 2026
    baseline = RandomTorchDP()
    baseline.begin_episode(seed)
    expected = [baseline.act(obs) for _ in range(7)]
    expected_rng_state = torch.get_rng_state().clone()

    dp = RandomTorchDP()
    guided = _policy(
        dp=dp, mode="rerank", num_candidates=4,
        min_improvement=1e6, max_joint_deviation=1.0,
    )
    guided.set_action_space(StubActionSpace())
    guided.begin_episode(seed)
    actual = [guided.act(obs) for _ in range(7)]
    for vanilla, selected in zip(expected, actual):
        assert np.array_equal(vanilla, selected)
    assert torch.equal(expected_rng_state, torch.get_rng_state())
    assert dp.calls == 7 * 4
    report = guided.get_diagnostics()
    assert report["num_replans"] == 7
    assert report["candidate_changed_count"] == 0
    assert len(report["decisions"]) == 7


def test_candidate_rejection_reasons_identify_each_gate():
    guided = _policy(
        dp=StubDP(values=(0.5, 0.1, 0.49)),
        mode="rerank", num_candidates=3,
        min_improvement=0.05,
        max_joint_deviation=0.15,
    )
    guided.set_action_space(StubActionSpace())
    result = guided.act(np.zeros((2, 3), dtype=np.float32))
    assert np.allclose(result, 0.5)
    report = guided.get_diagnostics()
    decision = report["decisions"][0]
    reasons = [item["reason"] for item in decision["candidate_diagnostics"]]
    assert reasons == [
        "baseline",
        "excessive_joint_deviation",
        "insufficient_improvement",
    ]
    assert decision["candidate_diagnostics"][1]["risk_improvement"] > 0.05
    assert decision["candidate_diagnostics"][1]["max_joint_deviation"] > 0.15
    assert decision["candidate_diagnostics"][2]["risk_improvement"] < 0.05
    assert decision["candidate_diagnostics"][2]["max_joint_deviation"] < 0.15
    assert report["candidate_reason_counts"] == {
        "excessive_joint_deviation": 1,
        "insufficient_improvement": 1,
    }


def test_selected_candidate_diagnostics_are_consistent():
    guided = _policy(
        dp=StubDP(values=(0.5, 0.1)),
        mode="rerank", num_candidates=2,
        min_improvement=0.05, max_joint_deviation=0.5,
    )
    guided.set_action_space(StubActionSpace())
    guided.act(np.zeros((2, 3), dtype=np.float32))
    decision = guided.get_diagnostics()["decisions"][0]
    item = decision["candidate_diagnostics"][1]
    assert item["eligible"] is True
    assert item["passes_improvement"] is True
    assert item["passes_deviation"] is True
    assert item["reason"] == "selected"
    assert decision["selected"] == 1


def test_rerank_only_uses_extra_rng_state_inside_fork(monkeypatch):
    """Extra DP samples are isolated even if they raise during generation."""
    class ExplodingDP(RandomTorchDP):
        def act(self, observation_window):
            out = super().act(observation_window)
            if self.calls == 3:
                raise RuntimeError("forced sampling failure")
            return out

    obs = np.zeros((2, 3), dtype=np.float32)
    expected = ExplodingDP()
    expected.begin_episode(21)
    expected.act(obs)
    expected_state = torch.get_rng_state().clone()

    guided = _policy(dp=ExplodingDP(), mode="rerank", num_candidates=4)
    guided.set_action_space(StubActionSpace())
    guided.begin_episode(21)
    with pytest.raises(RuntimeError, match="forced sampling failure"):
        guided.act(obs)
    assert torch.equal(expected_state, torch.get_rng_state())
