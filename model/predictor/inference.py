"""Load a predictor and score a proposed action chunk.

The returned vector is ``[dr_mean, dr_max, topple_count, fall_count]`` in the
original label units. This module does not import Vanilla DP.
"""
from __future__ import annotations

import torch

from model.predictor.checkpoint import load_predictor


def predict(model, normalizer, obs, actions):
    with torch.no_grad():
        normalized = model(
            normalizer.normalize_obs(obs),
            normalizer.normalize_action(actions),
        )
        return normalizer.denormalize_target(normalized)
