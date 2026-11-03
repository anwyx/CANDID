"""Deterministic validation and loss plotting for Vanilla DP."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def make_validation_loader(dataset, batch_size, max_windows, seed):
    if len(dataset) == 0:
        raise ValueError("No successful validation windows. Check the scene split.")
    n = min(len(dataset), max_windows)
    indices = np.random.default_rng(seed).choice(len(dataset), size=n, replace=False)
    loader = DataLoader(Subset(dataset, sorted(indices.tolist())),
                        batch_size=min(batch_size, n), shuffle=False, num_workers=0)
    return loader, n


@torch.inference_mode()
def compute_validation_loss(model, loader, normalizer, device, seed):
    """Compare checkpoints on the same windows and sampled diffusion noise."""
    model.eval()
    cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    total = 0.0
    count = 0
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        for batch in loader:
            obs = batch["observations"].to(device)
            actions = normalizer.normalize(batch["actions"].to(device))
            loss = model.compute_loss(obs, actions)
            n = obs.shape[0]
            total += float(loss.item()) * n
            count += n
    return total / count


def plot_loss_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [row["iteration"] for row in history]
    tr = [row.get("train_noise_loss", row["train_loss"]) for row in history]
    va = [row["val_loss"] for row in history]
    best = min(range(len(history)), key=lambda i: va[i])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, tr, label="Train DDPM noise MSE (interval mean)")
    ax.plot(x, va, label="Validation DDPM noise MSE (EMA)")
    if "train_noise_loss" in history[0]:
        ax.plot(x, [row["train_loss"] for row in history],
                label="Train total objective", linestyle="--", alpha=0.65)
    ax.scatter([x[best]], [va[best]], label="Best validation")
    ax.set(xlabel="Iteration", ylabel="Loss", title="DP training and validation losses")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
