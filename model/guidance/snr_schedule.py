"""Deterministic log-SNR guidance strengths for the frozen DDPM scheduler.

The SNR definition uses the actual scheduler.alphas_cumprod (not the
denoising loop ordinal). Strengths are precomputed once at policy creation,
never on the GPU inside denoising, so no per-step synchronization is added.
"""
from __future__ import annotations

import math


def build_strength_schedule(config, alphas_cumprod, timesteps):
    """Return {int DDPM timestep: float nominal strength}.

    snr_gaussian and snr_increasing have the SAME total nominal strength
    across the eligible DDPM steps as constant scheduling. The capped
    gaussian stays <= the fixed strength, intentionally reducing total dose.
    Actual update norms can still differ due to gradient and step clipping.
    """
    steps = [int(t) for t in timesteps]
    if len(set(steps)) != len(steps):
        raise ValueError("DDPM sampling timesteps must be unique")
    allowed = [t for t in steps if config.should_guide_timestep(t)]
    weights = {t: 0.0 for t in steps}
    if not allowed:
        return weights
    if config.strength_schedule == "constant":
        for t in allowed:
            weights[t] = float(config.strength)
        return weights

    # One host read only, during sampler initialization; no GPU sync in loop.
    if hasattr(alphas_cumprod, "detach"):
        alpha = alphas_cumprod.detach().cpu().flatten().tolist()
    else:
        alpha = list(alphas_cumprod)
    if config.snr_center_timestep >= len(alpha):
        raise ValueError("SNR center is outside DDPM training timesteps")
    if any(t < 0 or t >= len(alpha) for t in allowed):
        raise ValueError("Guidance timestep is outside scheduler alphas_cumprod")

    def log_snr(t):
        a = float(alpha[t])
        if not math.isfinite(a) or a < 0 or a > 1:
            raise ValueError("Invalid cumulative alpha in DDPM scheduler")
        return math.log(max(a, 1e-20)) - math.log(max(1-a, 1e-20))

    center = log_snr(config.snr_center_timestep)
    values = []
    for t in allowed:
        z = (log_snr(t) - center) / config.snr_sigma
        if config.strength_schedule in ("snr_gaussian", "snr_gaussian_capped"):
            weight = math.exp(-0.5 * z * z)
        elif config.strength_schedule == "snr_increasing":
            # Stable logistic: from low guidance at noisy steps to high
            # guidance at cleaner steps.
            weight = (1.0 / (1.0 + math.exp(-z))
                      if z >= 0 else math.exp(z) / (1.0 + math.exp(z)))
        else:
            raise ValueError(f"Unknown strength schedule: {config.strength_schedule}")
        values.append(weight)
    if config.strength_schedule != "snr_gaussian_capped":
        total = sum(values)
        if total <= 1e-30:
            raise ValueError("SNR weights vanished: adjust center/width")
        factor = len(allowed) / total
    else:
        factor = 1.0
    for t, value in zip(allowed, values):
        weights[t] = float(config.strength * value * factor)
    return weights
