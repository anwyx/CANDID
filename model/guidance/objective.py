"""Weighted local risk used to guide a diffusion action sample.

The scalar is a weighted sum of predicted DR and event scores. DR values are
regression outputs. Event scores are sigmoid outputs and are not calibrated
probabilities.
"""
from __future__ import annotations

import torch


def risk_from_predictions(dr_mean, dr_max, topple_score, fall_score, weights):
    """``weights`` is (dr_mean, dr_max, topple, fall)."""
    return (
        weights[0] * dr_mean
        + weights[1] * dr_max
        + weights[2] * topple_score
        + weights[3] * fall_score
    )


def chunk_predictions(model, normalizer, obs, actions):
    """Differentiable predictor forward. Actions are physical joint commands."""
    outputs = model(
        normalizer.normalize_obs(obs),
        normalizer.normalize_action(actions),
    )
    dr = normalizer.denormalize_target(outputs["dr"])
    events = torch.sigmoid(outputs["event_logits"])
    return {
        "dr_mean": dr[:, 0],
        "dr_max": dr[:, 1],
        "topple_score": events[:, 0],
        "fall_score": events[:, 1],
    }


def chunk_risk(model, normalizer, obs, actions, weights):
    predictions = chunk_predictions(model, normalizer, obs, actions)
    risk = risk_from_predictions(
        predictions["dr_mean"],
        predictions["dr_max"],
        predictions["topple_score"],
        predictions["fall_score"],
        weights,
    )
    return risk, predictions
