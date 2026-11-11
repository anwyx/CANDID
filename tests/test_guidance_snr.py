"""CPU-only tests of log-SNR strength scheduling; no simulator required."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from model.guidance.config import GuidanceConfig
from model.guidance.snr_schedule import build_strength_schedule


def _cosine_alpha(n=100):
    # Reproduces the qualitative monotonically decreasing cosine schedule.
    # The production schedule is read from the live DDPM object, not here.
    t = np.arange(n + 1, dtype=np.float64) / n
    f = np.cos(((t + .008) / 1.008) * math.pi / 2) ** 2
    return torch.as_tensor((f[1:] / f[0]).copy(), dtype=torch.float64)


def test_constant_schedule_matches_legacy_odd_steps_exactly():
    cfg = GuidanceConfig(
        strength=.02, start_timestep=30, guidance_parity="odd")
    schedule = build_strength_schedule(cfg, _cosine_alpha(), range(99, -1, -1))
    active = {t: v for t, v in schedule.items() if v > 0}
    assert set(active) == set(range(1, 31, 2))
    assert all(v == .02 for v in active.values())
    assert sum(active.values()) == pytest.approx(.30)


def test_gaussian_is_centered_and_total_dose_matches_constant():
    cfg = GuidanceConfig(
        strength=.02, start_timestep=30, guidance_parity="odd",
        strength_schedule="snr_gaussian", snr_center_timestep=15, snr_sigma=1)
    result = build_strength_schedule(cfg, _cosine_alpha(), range(99, -1, -1))
    assert sum(result.values()) == pytest.approx(.30, abs=1e-12)
    assert max(result, key=result.get) == 15
    assert result[15] > result[29] > 0
    assert result[15] > result[1] > 0
    assert result[30] == result[0] == 0


def test_monotone_schedule_increases_with_snr_and_preserves_total():
    cfg = GuidanceConfig(
        strength=.02, start_timestep=30, guidance_parity="odd",
        strength_schedule="snr_increasing", snr_center_timestep=15,
        snr_sigma=1)
    result = build_strength_schedule(cfg, _cosine_alpha(), range(99, -1, -1))
    assert sum(result.values()) == pytest.approx(.30)
    assert result[1] > result[15] > result[29] > 0


def test_gaussian_capped_never_exceeds_base_and_reduces_dose():
    cfg = GuidanceConfig(
        strength=.02, start_timestep=30, guidance_parity="odd",
        strength_schedule="snr_gaussian_capped", snr_center_timestep=15,
        snr_sigma=1)
    result = build_strength_schedule(cfg, _cosine_alpha(), range(99, -1, -1))
    assert result[15] == pytest.approx(.02)
    assert max(result.values()) <= .02 + 1e-12
    assert 0 < sum(result.values()) < .30


def test_zero_strength_skips_every_step_for_all_schedules():
    for name in ("constant", "snr_gaussian", "snr_increasing",
                 "snr_gaussian_capped"):
        cfg = GuidanceConfig(
            strength=0, guidance_parity="odd", strength_schedule=name)
        schedule = build_strength_schedule(cfg, _cosine_alpha(), range(99,-1,-1))
        assert all(v == 0 for v in schedule.values())


def test_bad_schedule_and_nonfinite_snr_parameters_fail_fast():
    with pytest.raises(ValueError, match="strength_schedule"):
        GuidanceConfig(strength_schedule="infinite")
    with pytest.raises(ValueError, match="snr_sigma"):
        GuidanceConfig(snr_sigma=0)
    with pytest.raises(ValueError, match="snr_center_timestep"):
        GuidanceConfig(snr_center_timestep=-1)
    cfg = GuidanceConfig(
        strength=.02, strength_schedule="snr_gaussian",
        snr_center_timestep=101)
    with pytest.raises(ValueError, match="SNR center"):
        build_strength_schedule(cfg, _cosine_alpha(), range(99, -1, -1))


def test_schedule_does_not_draw_rng_or_modify_scheduler_alphas():
    cfg = GuidanceConfig(
        strength=.02, strength_schedule="snr_gaussian",
        guidance_parity="odd")
    alpha = _cosine_alpha()
    before = alpha.clone()
    torch.manual_seed(314)
    rng_before = torch.random.get_rng_state().clone()
    build_strength_schedule(cfg, alpha, range(99, -1, -1))
    assert torch.equal(before, alpha)
    assert torch.equal(rng_before, torch.random.get_rng_state())
