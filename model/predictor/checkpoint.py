"""Predictor checkpoint payload."""
from __future__ import annotations

from pathlib import Path

import torch

from model.predictor.config import PredictorConfig
from model.predictor.network import DisturbancePredictor, OrdinalDisturbancePredictor
from model.predictor.normalizer import PredictorNormalizer


def save_checkpoint(path, model, normalizer: PredictorNormalizer, config: PredictorConfig):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "disturbance_predictor",
            "model": model.state_dict(),
            "format_version": 3 if config.dr_objective == 'ordinal' else 2,
            "config": {
                "obs_horizon": config.obs_horizon,
                "action_horizon": config.action_horizon,
                "hidden": config.hidden,
                "dr_objective": config.dr_objective,
                "dr_thresholds": list(config.dr_thresholds),
                "ordinal_pos_weight": config.ordinal_pos_weight,
            },
            "obs_dim": model.obs_dim,
            "action_dim": model.action_dim,
            "target_dim": model.target_dim,
            "obs_mean": normalizer.obs_mean.detach().cpu(),
            "obs_std": normalizer.obs_std.detach().cpu(),
            "action_mean": normalizer.action_mean.detach().cpu(),
            "action_std": normalizer.action_std.detach().cpu(),
            "target_mean": normalizer.target_mean.detach().cpu(),
            "target_std": normalizer.target_std.detach().cpu(),
            "event_pos_weight": normalizer.event_pos_weight.detach().cpu(),
        },
        path,
    )
    return path


def load_predictor(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("kind") != "disturbance_predictor":
        raise ValueError(f"Expected a disturbance_predictor checkpoint, found {payload.get('kind')}")
    if payload.get("format_version") not in (2, 3):
        raise ValueError("Legacy predictor checkpoint uses count regression; retrain with the new dual-head predictor")
    objective = payload['config'].get('dr_objective', 'regression')
    if objective not in ('regression', 'ordinal'):
        raise ValueError(f"Unknown predictor objective: {objective}")
    if payload.get('format_version') == 2 and objective != 'regression':
        raise ValueError("Ordinal predictor requires checkpoint format v3")
    cls = OrdinalDisturbancePredictor if objective == 'ordinal' else DisturbancePredictor
    extra = (
        {'dr_thresholds': payload['config']['dr_thresholds'],
         'ordinal_pos_weight': payload['config']['ordinal_pos_weight']}
        if objective == 'ordinal' else {}
    )
    model = cls(
        obs_dim=payload["obs_dim"],
        action_dim=payload["action_dim"],
        target_dim=payload["target_dim"],
        obs_horizon=payload["config"]["obs_horizon"],
        action_horizon=payload["config"]["action_horizon"],
        hidden=payload["config"]["hidden"],
        **extra,
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    normalizer = PredictorNormalizer(
        payload["obs_mean"],
        payload["obs_std"],
        payload["action_mean"],
        payload["action_std"],
        payload["target_mean"],
        payload["target_std"],
        payload["event_pos_weight"],
    ).to(device)
    return model, normalizer
