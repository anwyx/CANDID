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
            normalizer.normalize_action(actions, obs=obs),
        )
        dr = (outputs["dr"] if "ordinal_logits" in outputs
              else normalizer.denormalize_target(outputs["dr"]))
        events = torch.sigmoid(outputs["event_logits"])
        result = {
            "dr_mean": dr[:, 0],
            "dr_max": dr[:, 1],
            "topple_prob": events[:, 0],
            "fall_prob": events[:, 1],
        }
        if "multi_ordinal_logits" in outputs:
            result["multi_dr"] = outputs["multi_dr"]
            result["multi_topple_prob"] = torch.sigmoid(outputs["multi_event_logits"][..., 0])
            result["multi_fall_prob"] = torch.sigmoid(outputs["multi_event_logits"][..., 1])
        elif "ordinal_logits" in outputs:
            probs = torch.sigmoid(outputs["ordinal_logits"])
            result["dr_mean_exceed_prob"] = probs[:, 0]
            result["dr_max_exceed_prob"] = probs[:, 1]
        return result
