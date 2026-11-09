"""Differentiable chunk risk, in physical rather than DP-normalized action units."""
from __future__ import annotations

import torch


def risk_from_predictions(outputs, predictor_normalizer, config):
    """Return [B] risk with the same nonnegative DR semantics as reranking.

    The regressor is unconstrained. Clamping negative DR, and requiring
    dr_max >= dr_mean, prevents nonphysical outputs from being rewarded.
    Sigmoid logits are scores, not calibrated episode event probabilities.
    """
    dr = (outputs["dr"] if "ordinal_logits" in outputs
          else predictor_normalizer.denormalize_target(outputs["dr"]))
    probabilities = torch.sigmoid(outputs["event_logits"])
    mean = dr[:, 0].clamp_min(0.0)
    maximum = torch.maximum(dr[:, 1].clamp_min(0.0), mean)
    return (
        config.dr_mean_weight * mean
        + config.dr_max_weight * maximum
        + config.topple_weight * probabilities[:, 0]
        + config.fall_weight * probabilities[:, 1]
    )


def chunk_risk(obs, absolute_actions, predictor, predictor_normalizer, config, *, check_finite=True):
    """Score [B,H_obs,O] and [B,H_act,A]; preserve gradients to actions."""
    if obs.ndim != 3 or absolute_actions.ndim != 3:
        raise ValueError("Risk inputs must be batched observation and action windows")
    if obs.shape[0] != absolute_actions.shape[0]:
        raise ValueError("Observation and action batch sizes disagree")
    outputs = predictor(
        predictor_normalizer.normalize_obs(obs),
        predictor_normalizer.normalize_action(absolute_actions),
    )
    risk = risk_from_predictions(outputs, predictor_normalizer, config)
    if risk.shape != (obs.shape[0],):
        raise ValueError("Predictor produced a risk tensor with wrong shape")
    if check_finite and not torch.isfinite(risk).all():
        raise FloatingPointError("Predictor produced nonfinite chunk risk")
    return risk
