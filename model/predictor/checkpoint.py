"""Predictor checkpoint payload."""
from __future__ import annotations

from pathlib import Path

import torch

from model.predictor.config import PredictorConfig
from model.predictor.network import DisturbancePredictor, OrdinalDisturbancePredictor
from model.predictor.multihorizon import MultiHorizonConsequencePredictor
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


def save_multihorizon_checkpoint(path, model, normalizer):
    """Scene1 v3 format. Old v2 checkpoints remain readable and unchanged."""
    if not isinstance(model, MultiHorizonConsequencePredictor):
        raise TypeError("Expected a multi-horizon Predictor")
    if normalizer.action_representation != "delta_q":
        raise ValueError("v3 checkpoint must use delta_q normalizer")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": "disturbance_predictor", "format_version": 4,
        "model": model.state_dict(),
        "config": {
            "obs_horizon": 2, "action_horizon": 8, "hidden": model.hidden,
            "dr_objective": "multi_ordinal",
            "dr_thresholds": model.dr_thresholds.cpu().tolist(),
            "ordinal_pos_weight": model.ordinal_pos_weight,
            "consequence_horizons": [4, 8, 16, 32],
            "continuation_policy": "hold_last_joint_target_after_step_8",
            "action_representation": "delta_q",
            "horizon_weights": model.horizon_weights.cpu().tolist(),
        },
        "obs_dim": model.obs_dim, "action_dim": model.action_dim,
        "target_dim": 4,
        "obs_mean": normalizer.obs_mean.detach().cpu(),
        "obs_std": normalizer.obs_std.detach().cpu(),
        "action_mean": normalizer.action_mean.detach().cpu(),
        "action_std": normalizer.action_std.detach().cpu(),
        "target_mean": normalizer.target_mean.detach().cpu(),
        "target_std": normalizer.target_std.detach().cpu(),
        "event_pos_weight": normalizer.event_pos_weight.detach().cpu(),
    }, path)
    return path


def load_predictor(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("kind") != "disturbance_predictor":
        raise ValueError(f"Expected a disturbance_predictor checkpoint, found {payload.get('kind')}")
    if payload.get("format_version") not in (2, 3, 4):
        raise ValueError("Legacy predictor checkpoint uses count regression; retrain with the new dual-head predictor")
    objective = payload['config'].get('dr_objective', 'regression')
    if objective not in ('regression', 'ordinal', 'multi_ordinal'):
        raise ValueError(f"Unknown predictor objective: {objective}")
    if payload.get('format_version') == 2 and objective != 'regression':
        raise ValueError("Ordinal predictor requires checkpoint format v3")
    if objective == "multi_ordinal":
        if payload.get("format_version") != 4:
            raise ValueError("Multi-horizon Predictor requires checkpoint v4")
        cfg = payload["config"]
        if (tuple(cfg["consequence_horizons"]) != (4, 8, 16, 32)
                or cfg["continuation_policy"] != "hold_last_joint_target_after_step_8"
                or cfg["action_representation"] != "delta_q"):
            raise ValueError("Unknown multi-horizon Predictor physical protocol")
        model = MultiHorizonConsequencePredictor(
            obs_dim=payload["obs_dim"], action_dim=payload["action_dim"],
            hidden=cfg["hidden"], dr_thresholds=cfg["dr_thresholds"],
            ordinal_pos_weight=cfg["ordinal_pos_weight"],
            horizon_weights=cfg["horizon_weights"]).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        normalizer = PredictorNormalizer(
            payload["obs_mean"], payload["obs_std"],
            payload["action_mean"], payload["action_std"],
            payload["target_mean"], payload["target_std"],
            payload["event_pos_weight"],
            action_representation="delta_q").to(device)
        return model, normalizer
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
