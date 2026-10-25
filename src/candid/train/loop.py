"""Train and reload the pilot diffusion policy and disturbance predictor."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from candid.data.validate import validate_dataset
from candid.train.dataset import (
    ACTION_HORIZON,
    OBSERVATION_HORIZON,
    Normalizer,
    WindowDataset,
    fit_normalizer,
    load_trajectories,
    split_scene_ids,
)
from candid.train.models import DiffusionPolicy, DisturbancePredictor


def _loader(dataset, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(64, len(dataset)),
        shuffle=True,
        generator=generator,
    )


def _optimize(model, loader, loss_of, steps, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    step = 0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = loss_of(model, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
    return losses


def _mean_loss(model, dataset, loss_of, device):
    if len(dataset) == 0:
        return None
    loader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=False)
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            size = batch["obs"].shape[0]
            total += float(loss_of(model, batch).detach().cpu()) * size
            count += size
    return total / count


def _save(path, model, normalizer, kind, extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"kind": kind, "model": model.state_dict(), **normalizer.as_payload(), **extra},
        path,
    )
    return path


def train_diffusion_policy(
    train_set: WindowDataset,
    val_set: WindowDataset | None,
    output_path,
    steps: int = 40,
    lr: float = 1e-3,
    seed: int = 0,
    diffusion_steps: int = 8,
    device: str = "cpu",
):
    torch.manual_seed(seed)
    normalizer = fit_normalizer(train_set, include_targets=False).to(device)
    model = DiffusionPolicy(
        obs_dim=train_set.obs_dim,
        action_dim=train_set.action_dim,
        obs_horizon=train_set.observation_horizon,
        action_horizon=train_set.action_horizon,
        diffusion_steps=diffusion_steps,
    ).to(device)

    def loss_of(net, batch):
        return net.loss(
            normalizer.normalize_obs(batch["obs"]),
            normalizer.normalize_action(batch["action"]),
        )

    losses = _optimize(model, _loader(train_set, seed), loss_of, steps, lr, device)
    val_loss = None if val_set is None else _mean_loss(model, val_set, loss_of, device)
    checkpoint = _save(
        output_path,
        model,
        normalizer,
        "diffusion_policy",
        {
            "obs_dim": train_set.obs_dim,
            "action_dim": train_set.action_dim,
            "obs_horizon": train_set.observation_horizon,
            "action_horizon": train_set.action_horizon,
            "diffusion_steps": diffusion_steps,
        },
    )
    return {"losses": losses, "val_loss": val_loss, "checkpoint": checkpoint}


def train_predictor(
    train_set: WindowDataset,
    val_set: WindowDataset | None,
    output_path,
    steps: int = 40,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
):
    torch.manual_seed(seed)
    normalizer = fit_normalizer(train_set, include_targets=True).to(device)
    model = DisturbancePredictor(
        obs_dim=train_set.obs_dim,
        action_dim=train_set.action_dim,
        obs_horizon=train_set.observation_horizon,
        action_horizon=train_set.action_horizon,
    ).to(device)

    def loss_of(net, batch):
        return net.loss(
            normalizer.normalize_obs(batch["obs"]),
            normalizer.normalize_action(batch["action"]),
            normalizer.normalize_target(batch["target"]),
        )

    losses = _optimize(model, _loader(train_set, seed), loss_of, steps, lr, device)
    val_loss = None if val_set is None else _mean_loss(model, val_set, loss_of, device)
    checkpoint = _save(
        output_path,
        model,
        normalizer,
        "disturbance_predictor",
        {
            "obs_dim": train_set.obs_dim,
            "action_dim": train_set.action_dim,
            "target_dim": len(train_set[0]["target"]),
            "obs_horizon": train_set.observation_horizon,
            "action_horizon": train_set.action_horizon,
        },
    )
    return {"losses": losses, "val_loss": val_loss, "checkpoint": checkpoint}


def load_diffusion_policy(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["kind"] != "diffusion_policy":
        raise ValueError(f"Expected a diffusion policy checkpoint, found {payload['kind']}")
    model = DiffusionPolicy(
        obs_dim=payload["obs_dim"],
        action_dim=payload["action_dim"],
        obs_horizon=payload["obs_horizon"],
        action_horizon=payload["action_horizon"],
        diffusion_steps=payload["diffusion_steps"],
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, Normalizer.from_payload(payload).to(device)


def load_predictor(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["kind"] != "disturbance_predictor":
        raise ValueError(f"Expected a predictor checkpoint, found {payload['kind']}")
    model = DisturbancePredictor(
        obs_dim=payload["obs_dim"],
        action_dim=payload["action_dim"],
        target_dim=payload["target_dim"],
        obs_horizon=payload["obs_horizon"],
        action_horizon=payload["action_horizon"],
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, Normalizer.from_payload(payload).to(device)


def run_pilot(
    dataset_root,
    output_dir,
    steps: int = 40,
    seed: int = 0,
    val_ratio: float = 0.2,
    diffusion_steps: int = 8,
    device: str = "cpu",
):
    """Validate a collection, train both models, and sample one action chunk."""
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    report = validate_dataset(dataset_root)
    trajectories = load_trajectories(dataset_root)
    train_scenes, val_scenes = split_scene_ids(
        [item.scene_id for item in trajectories], val_ratio=val_ratio, seed=seed
    )
    policy_train = WindowDataset(trajectories, train_scenes, success_only=True)
    predictor_train = WindowDataset(trajectories, train_scenes, success_only=False)
    policy_val = WindowDataset(trajectories, val_scenes, success_only=True) if val_scenes else None
    predictor_val = (
        WindowDataset(trajectories, val_scenes, success_only=False) if val_scenes else None
    )
    policy_result = train_diffusion_policy(
        policy_train,
        policy_val,
        output_dir / "diffusion_policy.pt",
        steps=steps,
        seed=seed,
        diffusion_steps=diffusion_steps,
        device=device,
    )
    predictor_result = train_predictor(
        predictor_train,
        predictor_val,
        output_dir / "disturbance_predictor.pt",
        steps=steps,
        seed=seed,
        device=device,
    )
    policy, policy_norm = load_diffusion_policy(policy_result["checkpoint"], device)
    predictor, predictor_norm = load_predictor(predictor_result["checkpoint"], device)
    raw_obs = policy_train[0]["obs"].unsqueeze(0).to(device)
    with torch.no_grad():
        normalized_action = policy.sample(policy_norm.normalize_obs(raw_obs))
        action = policy_norm.denormalize_action(normalized_action)
        prediction = predictor_norm.denormalize_target(
            predictor(
                predictor_norm.normalize_obs(raw_obs),
                predictor_norm.normalize_action(action),
            )
        )
    if not torch.isfinite(action).all() or not torch.isfinite(prediction).all():
        raise RuntimeError("Pilot sample produced a non-finite action or prediction")
    return {
        "validation": report,
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
        "policy_train_windows": len(policy_train),
        "predictor_train_windows": len(predictor_train),
        "policy_losses": policy_result["losses"],
        "policy_val_loss": policy_result["val_loss"],
        "predictor_losses": predictor_result["losses"],
        "predictor_val_loss": predictor_result["val_loss"],
        "policy_checkpoint": policy_result["checkpoint"],
        "predictor_checkpoint": predictor_result["checkpoint"],
        "sample_action": action.detach().cpu(),
        "sample_prediction": prediction.detach().cpu(),
        "obs_horizon": OBSERVATION_HORIZON,
        "action_horizon": ACTION_HORIZON,
    }
