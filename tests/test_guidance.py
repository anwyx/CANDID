"""CPU-only guidance contracts; no ManiSkill or checkpoints required."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.guidance import GuidanceConfig, ConsequenceGuidedPolicy
from model.guidance.objective import chunk_risk


class TinyScheduler:
    def __init__(self):
        self.timesteps = [1, 0]
        self.alphas_cumprod = torch.ones(2)
        self.config = SimpleNamespace(prediction_type="epsilon", clip_sample=True, clip_sample_range=1.0)

    def step(self, model_output, timestep, sample):
        return SimpleNamespace(prev_sample=sample + 0.005 * torch.randn_like(sample))


class TinyDenoiser(torch.nn.Module):
    def forward(self, sample, timestep, global_cond):
        return torch.zeros_like(sample)


class TinyAgent(torch.nn.Module):
    obs_horizon, obs_dim = 2, 3
    act_horizon, act_dim = 8, 2
    pred_horizon, num_diffusion_iters = 16, 2

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.noise_pred_net = TinyDenoiser()
        self.noise_scheduler = TinyScheduler()

    def get_action(self, obs):
        with torch.no_grad():
            x = torch.randn((len(obs), self.pred_horizon, self.act_dim), device=obs.device)
            for t in self.noise_scheduler.timesteps:
                eps = self.noise_pred_net(x, t, global_cond=obs.flatten(start_dim=1))
                x = self.noise_scheduler.step(eps, t, x).prev_sample
        return x[:, 1:9]


class IdentityNormalizer:
    def denormalize(self, x):
        return x

    def denormalize_target(self, x):
        return x

    def normalize_obs(self, x):
        return x

    def normalize_action(self, x):
        return x


class TinyDP:
    obs_horizon, obs_dim = 2, 3
    action_horizon, action_dim = 8, 2

    def __init__(self):
        self.agent = TinyAgent()
        self.normalizer = IdentityNormalizer()

    def begin_episode(self, seed):
        torch.manual_seed(int(seed))

    def act(self, obs):
        return self.agent.get_action(torch.as_tensor(obs, dtype=torch.float32)[None])[0].numpy()


class TinyPredictor(torch.nn.Module):
    obs_horizon, obs_dim = 2, 3
    action_horizon, action_dim = 8, 2

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, obs, actions):
        self.calls += 1
        mean = actions.mean((1, 2))
        dr_mean = mean + 2 + self.anchor
        dr_max = dr_mean + 1
        logits = torch.stack((mean, mean), dim=-1)
        return {"dr": torch.stack((dr_mean, dr_max), dim=-1), "event_logits": logits}


class Bounds:
    shape = (2,)
    low = np.array([-5.0, -5.0], dtype=np.float32)
    high = np.array([5.0, 5.0], dtype=np.float32)


def make_policy(config, predictor=None):
    pred = predictor if predictor is not None else TinyPredictor()
    policy = ConsequenceGuidedPolicy(TinyDP(), pred, IdentityNormalizer(), config)
    policy.set_action_space(Bounds())
    return policy, pred


def test_strength_zero_exact_vanilla_actions_rng_and_skips_predictor():
    obs = np.zeros((2, 3), np.float32)
    vanilla = TinyDP()
    vanilla.begin_episode(123)
    expected = [vanilla.act(obs) for _ in range(3)]
    rng = torch.get_rng_state().clone()
    policy, predictor = make_policy(GuidanceConfig(strength=0, start_timestep=1))
    policy.begin_episode(123)
    actual = [policy.act(obs) for _ in range(3)]
    assert all(np.array_equal(a, b) for a, b in zip(expected, actual))
    assert torch.equal(rng, torch.get_rng_state())
    assert predictor.calls == 0
    assert policy.get_diagnostics()["total_guided_steps"] == 0


def test_active_guidance_changes_action_in_lower_risk_direction():
    obs = np.zeros((2, 3), np.float32)
    baseline, _ = make_policy(GuidanceConfig(strength=0, start_timestep=1))
    guided, predictor = make_policy(GuidanceConfig(
        strength=0.1, start_timestep=1, max_grad_norm=10, max_step_norm=1,
        topple_weight=0, fall_weight=0,
    ))
    baseline.begin_episode(3)
    x = baseline.act(obs)
    guided.begin_episode(3)
    y = guided.act(obs)
    assert y.shape == (8, 2) and y.mean() < x.mean()
    assert guided.get_diagnostics()["total_guided_steps"] == 2
    assert guided.get_diagnostics()["decisions"][0]["max_guidance_update_norm"] > 0
    assert predictor.calls > 0
    assert all(not p.requires_grad for p in predictor.parameters())
    assert all(not p.requires_grad for p in guided.sampler.agent.parameters())


def test_guidance_only_returns_executable_chunk():
    policy, _ = make_policy(GuidanceConfig(strength=0.02, start_timestep=1))
    policy.begin_episode(22)
    output, stats = policy.sampler.sample(torch.zeros((1, 2, 3)), policy.bounds)
    assert output.shape == (1, 8, 2)
    assert stats["guided_steps"] == 2


def test_zero_strength_returns_raw_actions_even_if_environment_would_clip():
    class NarrowBounds:
        shape = (2,)
        low = np.zeros(2, dtype=np.float32)
        high = np.zeros(2, dtype=np.float32)
    policy, predictor = make_policy(GuidanceConfig(strength=0, start_timestep=1))
    policy.set_action_space(NarrowBounds())
    policy.begin_episode(0)
    x = policy.act(np.zeros((2, 3), np.float32))
    assert np.any(x != 0)
    assert policy.get_diagnostics()["decisions"][0]["action_clipping_needed"]
    assert predictor.calls == 0


def test_horizon_bounds_and_invalid_obs_fail():
    class WrongPredictor(TinyPredictor):
        action_horizon = 4
    with pytest.raises(ValueError, match="action dimensions"):
        ConsequenceGuidedPolicy(TinyDP(), WrongPredictor(), IdentityNormalizer(),
                                GuidanceConfig(start_timestep=1))
    policy = ConsequenceGuidedPolicy(TinyDP(), TinyPredictor(), IdentityNormalizer(),
                                     GuidanceConfig(start_timestep=1))
    with pytest.raises(ValueError, match="environment action bounds"):
        policy.act(np.zeros((2, 3), np.float32))
    with pytest.raises(ValueError, match="observation history"):
        policy.act(np.zeros((3, 3), np.float32))


def test_invalid_config_and_nonfinite_input():
    for kwargs in (
        dict(strength=-1), dict(strength=float("nan")), dict(start_timestep=-1),
        dict(strength=0.5, max_grad_norm=0),
        dict(strength=0.5, dr_mean_weight=0, dr_max_weight=0,
             topple_weight=0, fall_weight=0),
    ):
        with pytest.raises(ValueError):
            GuidanceConfig(**kwargs)
    policy, _ = make_policy(GuidanceConfig(strength=0, start_timestep=1))
    with pytest.raises(ValueError):
        policy.act(np.full((2, 3), np.nan, np.float32))


def test_risk_is_differentiable_wrt_physical_actions():
    predictor = TinyPredictor()
    actions = torch.zeros((2, 8, 2), requires_grad=True)
    risk = chunk_risk(torch.zeros((2, 2, 3)), actions, predictor,
                      IdentityNormalizer(), GuidanceConfig(start_timestep=1))
    gradient = torch.autograd.grad(risk.sum(), actions)[0]
    assert torch.isfinite(gradient).all() and torch.any(gradient != 0)


def test_paired_diagnostics_do_not_change_actions_or_future_rng():
    obs = np.zeros((2, 3), np.float32)
    plain, _ = make_policy(GuidanceConfig(strength=0.05, start_timestep=1))
    paired, _ = make_policy(GuidanceConfig(strength=0.05, start_timestep=1))
    paired.diagnose = True
    plain.begin_episode(77)
    expected = [plain.act(obs) for _ in range(4)]
    expected_rng = torch.get_rng_state().clone()
    paired.begin_episode(77)
    actual = [paired.act(obs) for _ in range(4)]
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))
    assert torch.equal(expected_rng, torch.get_rng_state())
    diagnostic = paired.get_diagnostics()
    assert diagnostic["paired_count"] == 4
    assert diagnostic["paired_risk_decreased_count"] == 4
    assert diagnostic["paired_risk_delta_mean"] < 0
    assert diagnostic["paired_action_abs_mean"] > 0
    for row in diagnostic["decisions"]:
        assert row["paired"]["risk_after"] < row["paired"]["risk_before"]
        assert row["paired"]["action_abs_max"] > 0


def test_paired_zero_strength_is_noop():
    obs = np.zeros((2, 3), np.float32)
    paired, _ = make_policy(GuidanceConfig(strength=0, start_timestep=1))
    paired.diagnose = True
    paired.begin_episode(41)
    paired.act(obs)
    diagnostic = paired.get_diagnostics()
    assert diagnostic["paired_count"] == 1
    assert diagnostic["decisions"][0]["paired"]["action_abs_max"] == 0
    assert abs(diagnostic["decisions"][0]["paired"]["risk_delta"]) < 1e-7


def test_cpu_profiler_reports_nonnegative_phase_durations():
    policy, _ = make_policy(GuidanceConfig(strength=0.05, start_timestep=1))
    policy.profile_timing = True
    policy.begin_episode(17)
    policy.act(np.zeros((2, 3), np.float32))
    row = policy.get_diagnostics()["decisions"][0]["timing_ms"]
    assert row["phase_time_source"] == "cpu_wall"
    assert row["sampling_wall_ms"] > 0
    for key in ("denoiser_ms", "predictor_gradient_ms", "ddpm_update_ms"):
        assert row[key] >= 0
    totals = policy.get_diagnostics()["timing_totals_ms"]
    assert totals["sampling_wall_ms"] == row["sampling_wall_ms"]


def test_alternating_parity_timestep_selection():
    all_steps = GuidanceConfig(strength=.02, start_timestep=30)
    odd_steps = GuidanceConfig(strength=.02, start_timestep=30, guidance_parity="odd")
    even_steps = GuidanceConfig(strength=.02, start_timestep=30, guidance_parity="even")
    assert sum(all_steps.should_guide_timestep(t) for t in range(100)) == 31
    assert [t for t in range(100) if odd_steps.should_guide_timestep(t)] == list(range(1, 31, 2))
    assert [t for t in range(100) if even_steps.should_guide_timestep(t)] == list(range(0, 31, 2))
    assert not GuidanceConfig(strength=0, guidance_parity="odd").should_guide_timestep(1)
    with pytest.raises(ValueError, match="guidance_parity"):
        GuidanceConfig(guidance_parity="alternating")


def test_odd_timestep_guidance_and_zero_strength_rng_contract():
    # A linear toy risk has an action-independent gradient. Moving it from
    # t=1 to t=0 can produce EXACTLY the same final action: parity must be
    # tested by recording when Predictor is called, not by assuming outputs
    # from odd/even schedules are necessarily different.
    class TrackedDenoiser(TinyDenoiser):
        def __init__(self):
            super().__init__()
            self.last_timestep = None

        def forward(self, sample, timestep, global_cond):
            self.last_timestep = int(timestep)
            return super().forward(sample, timestep, global_cond)

    class TrackedPredictor(TinyPredictor):
        def __init__(self, denoiser):
            super().__init__()
            self.denoiser = denoiser
            self.gradient_timesteps = []
            self.scoring_timesteps = []

        def forward(self, obs, actions):
            # Guidance computes Predictor risk with autograd enabled;
            # policy.act also scores the final chunk under no_grad().
            # The latter must not be mistaken for a guidance update.
            if torch.is_grad_enabled():
                self.gradient_timesteps.append(self.denoiser.last_timestep)
            else:
                self.scoring_timesteps.append(self.denoiser.last_timestep)
            return super().forward(obs, actions)

    def tracked_policy(parity):
        denoiser = TrackedDenoiser()
        predictor = TrackedPredictor(denoiser)
        policy, _ = make_policy(GuidanceConfig(
            strength=.1, start_timestep=1, guidance_parity=parity,
            max_grad_norm=10, max_step_norm=1, topple_weight=0, fall_weight=0,
        ), predictor=predictor)
        policy.sampler.agent.noise_pred_net = denoiser
        return policy, predictor

    obs = np.zeros((2, 3), np.float32)
    odd, odd_predictor = tracked_policy("odd")
    even, even_predictor = tracked_policy("even")
    vanilla = TinyDP()
    vanilla.begin_episode(5)
    base = vanilla.act(obs)
    odd.begin_episode(5)
    odd_actions = odd.act(obs)
    even.begin_episode(5)
    even_actions = even.act(obs)

    assert odd.get_diagnostics()["total_guided_steps"] == 1
    assert even.get_diagnostics()["total_guided_steps"] == 1
    assert odd_predictor.gradient_timesteps == [1]
    assert even_predictor.gradient_timesteps == [0]
    # One extra forward per policy.act is a diagnostic score at final t=0.
    assert odd_predictor.scoring_timesteps == [0]
    assert even_predictor.scoring_timesteps == [0]
    assert odd_predictor.calls == 2
    assert even_predictor.calls == 2
    assert not np.array_equal(base, odd_actions)
    assert not np.array_equal(base, even_actions)

    zero, predictor = make_policy(GuidanceConfig(
        strength=0, start_timestep=1, guidance_parity="odd"))
    zero.begin_episode(5)
    assert np.array_equal(zero.act(obs), base)
    assert predictor.calls == 0


def test_all_parity_matches_legacy_guided_path():
    obs = np.zeros((2, 3), np.float32)
    old, _ = make_policy(GuidanceConfig(strength=.1, start_timestep=1))
    explicit, _ = make_policy(GuidanceConfig(
        strength=.1, start_timestep=1, guidance_parity="all"))
    old.begin_episode(73)
    a = [old.act(obs) for _ in range(3)]
    explicit.begin_episode(73)
    b = [explicit.act(obs) for _ in range(3)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_normalize_action_legacy_and_delta_q_interfaces():
    from model.predictor.normalizer import (
        PredictorNormalizer, normalize_predictor_actions
    )
    from model.predictor.action_features import absolute_to_delta_q

    # Old test normalizers only accept actions, not an obs keyword.
    obs = torch.zeros((1, 2, 3), dtype=torch.float32)
    actions = torch.ones((1, 8, 2), dtype=torch.float32, requires_grad=True)
    old = normalize_predictor_actions(IdentityNormalizer(), actions, obs=obs)
    assert torch.equal(old, actions)
    old.sum().backward()
    assert torch.equal(actions.grad, torch.ones_like(actions))

    # Old fitted Predictor normalizers retain absolute-action semantics.
    old_normalizer = PredictorNormalizer(
        torch.zeros(3), torch.ones(3), torch.zeros(2), torch.ones(2),
        torch.zeros(2), torch.ones(2))
    assert torch.equal(normalize_predictor_actions(
        old_normalizer, actions.detach(), obs=obs), actions.detach())

    # Scene1 v3 uses the observation to produce differentiable delta-q.
    obs_v3 = torch.randn(1, 2, 151)
    act_v3 = torch.randn(1, 8, 7, requires_grad=True)
    normalizer_v3 = PredictorNormalizer(
        torch.zeros(151), torch.ones(151), torch.zeros(7), torch.ones(7),
        torch.zeros(2), torch.ones(2),
        action_representation="delta_q")
    normalized = normalize_predictor_actions(
        normalizer_v3, act_v3, obs=obs_v3)
    assert torch.allclose(normalized, absolute_to_delta_q(obs_v3, act_v3))
    normalized.sum().backward()
    assert torch.isfinite(act_v3.grad).all()
    assert torch.all(act_v3.grad != 0)
