"""Headless figures for one paired guidance comparison."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _number(value):
    return f"{float(value):.6g}"


def save_comparison_figures(result, output_dir, *, scene_id, episode_step, replan_index, strength, exec_horizon):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{replan_index:03d}"
    paths = {
        "risk": str(output / f"replan_{stem}.png"),
        "actions": str(output / f"actions_replan_{stem}.png"),
        "diffusion": str(output / f"diffusion_replan_{stem}.png"),
    }
    _risk_figure(result, paths["risk"], scene_id, episode_step, replan_index, strength)
    _action_figure(result, paths["actions"], exec_horizon)
    _diffusion_figure(result, paths["diffusion"], result.get("start_timestep"))
    return paths


def _risk_figure(result, path, scene_id, episode_step, replan_index, strength):
    vanilla = result["scores"]["vanilla"]
    guided = result["scores"]["guided"]
    labels = ["Predicted Local DR Mean", "Predicted Local DR Max"]
    vanilla_values = [vanilla["dr_mean"], vanilla["dr_max"]]
    guided_values = [guided["dr_mean"], guided["dr_max"]]
    low = min(0.0, min(vanilla_values), min(guided_values))
    high = max(max(vanilla_values), max(guided_values), 1e-6)
    span = max(high - low, 1e-6)
    y_min = low - 0.08 * span
    y_max = high + 0.18 * span
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    positions = np.arange(len(labels))
    width = 0.34
    axis.bar(positions - width / 2, vanilla_values, width, color="#2f6fed", label="Vanilla DP")
    axis.bar(positions + width / 2, guided_values, width, color="#d64545", label="CANDID Guided")
    axis.set_xticks(positions, labels)
    axis.set_ylim(y_min, y_max)
    axis.axhline(0, color="#bbbbbb", linewidth=0.6)
    axis.set_ylabel("Predicted value (shared scale)")
    axis.set_title("Same observation, same diffusion noise")
    axis.legend()
    for index, (left, right) in enumerate(zip(vanilla_values, guided_values)):
        axis.text(index - width / 2, left, _number(left), ha="center", va="bottom", fontsize=8)
        axis.text(index + width / 2, right, _number(right), ha="center", va="bottom", fontsize=8)
    note = (
        f"scene {scene_id}\n"
        f"episode step {episode_step}   replan {replan_index}   strength {strength:g}\n"
        f"shared scale [{_number(y_min)}, {_number(y_max)}]   "
        f"delta DR mean {_number(result['delta_dr_mean'])}   "
        f"delta DR max {_number(result['delta_dr_max'])}\n"
        f"action L2 {_number(result['action_l2'])}   "
        f"mean |delta| {_number(result['action_abs_mean'])}   "
        f"max |delta| {_number(result['action_abs_max'])}\n"
        "No simulator frame. Bars use one shared scale and the real predictor outputs."
    )
    figure.text(0.01, 0.01, note, fontsize=8, va="bottom")
    figure.tight_layout(rect=(0, 0.24, 1, 1))
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _action_figure(result, path, exec_horizon):
    vanilla = result["vanilla_clipped"]
    guided = result["guided_clipped"]
    horizon = vanilla.shape[0]
    joints = vanilla.shape[1]
    time = np.arange(horizon)
    figure, axes = plt.subplots(joints, 1, figsize=(8, 1.35 * joints), sharex=True)
    if joints == 1:
        axes = [axes]
    for joint, axis in enumerate(axes):
        axis.axvspan(-0.5, exec_horizon - 0.5, color="#d9e8ff", zorder=0)
        axis.plot(time, vanilla[:, joint], color="#2f6fed", label="Vanilla DP")
        axis.plot(time, guided[:, joint], color="#d64545", label="CANDID Guided")
        axis.set_ylabel(f"j{joint}")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Clipped joint commands. Blue band is the executed prefix.")
    axes[-1].set_xlabel("Action-chunk index")
    axes[-1].set_xlim(-0.5, horizon - 0.5)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _diffusion_figure(result, path, start_timestep):
    rows = result["diffusion_trace"]
    figure, axis = plt.subplots(figsize=(8, 4.2))
    if not rows:
        axis.set_title("Diffusion trace was not recorded for this replan")
        figure.savefig(path, dpi=140)
        plt.close(figure)
        return
    order = np.arange(len(rows))
    timesteps = [row["diffusion_timestep"] for row in rows]
    risk = [np.nan if row["pred_weighted_risk"] is None else row["pred_weighted_risk"] for row in rows]
    gradient = [row["raw_gradient_l2"] for row in rows]
    correction = [row["applied_correction_l2"] for row in rows]
    axis.plot(order, risk, color="#7a4eab", label="Intermediate weighted risk")
    axis.plot(order, gradient, color="#2f6fed", label="Gradient L2")
    axis.plot(order, correction, color="#d64545", label="Correction L2")
    guided_orders = [index for index, row in enumerate(rows) if row["guided"]]
    if guided_orders:
        axis.axvline(guided_orders[0], color="#888888", linestyle="--", label="Guidance starts")
    tick_index = order[:: max(1, len(order) // 8)]
    axis.set_xticks(tick_index)
    axis.set_xticklabels([str(timesteps[int(index)]) for index in tick_index])
    axis.set_xlabel("Reverse-diffusion order (tick labels are diffusion timesteps)")
    axis.set_ylabel("Value")
    axis.set_title("Intermediate x0 risk is not the final chunk DR")
    if start_timestep is not None:
        axis.text(
            0.01, 0.98,
            f"guidance when timestep <= {start_timestep}\nmax correction L2 {_number(max(correction))}",
            transform=axis.transAxes, va="top",
        )
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
