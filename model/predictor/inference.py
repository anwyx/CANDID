"""Predict DR values and event probabilities for a proposed action chunk."""
from __future__ import annotations

import torch


def predict(model, normalizer, obs, actions):
    """Return original-unit DR estimates and calibrated-by-sigmoid event scores."""
    device = next(model.parameters()).device
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
    if not torch.isfinite(obs).all() or not torch.isfinite(actions).all():
        raise ValueError("Predictor inputs must be finite")
    with torch.no_grad():
        outputs = model(
            normalizer.normalize_obs(obs),
            normalizer.normalize_action(actions),
        )
        dr = normalizer.denormalize_target(outputs["dr"])
        events = torch.sigmoid(outputs["event_logits"])
        return {
            "dr_mean": dr[:, 0],
            "dr_max": dr[:, 1],
            "topple_prob": events[:, 0],
            "fall_prob": events[:, 1],
        }
