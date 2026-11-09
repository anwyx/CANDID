"""Evaluate the independent predictor on frozen held-out test scenes."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from candid.data.splits import validate_scene_split
from model.predictor.checkpoint import load_predictor
from model.predictor.dataset import PredictorWindowDataset
from model.predictor.inference import predict


def _regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if not len(actual):
        return {"count": 0, "mae": None, "rmse": None, "r2": None}
    errors = actual - predicted
    ss_tot = float(np.square(actual - actual.mean()).sum())
    ss_res = float(np.square(errors).sum())
    return {
        "count": len(actual),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
    }


def _event_metrics(actual, scores, threshold):
    actual = np.asarray(actual, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n = len(actual)
    if not n:
        return {"count": 0, "positives": 0, "precision": None, "recall": None,
                "f1": None, "auroc": None, "auprc": None, "false_negative_rate": None}
    predictions = scores >= threshold
    tp = int(np.sum((actual == 1) & predictions))
    fp = int(np.sum((actual == 0) & predictions))
    fn = int(np.sum((actual == 1) & ~predictions))
    positives = int(actual.sum())
    negatives = n - positives
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else None
    f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else (0.0 if recall is not None else None)
    auroc = auprc = None
    if positives and negatives:
        order = np.argsort(-scores, kind="stable")
        sorted_y = actual[order]
        sorted_scores = scores[order]
        distinct_end = np.r_[np.where(np.diff(sorted_scores) != 0)[0], n - 1]
        tp_curve = np.cumsum(sorted_y)[distinct_end]
        fp_curve = (distinct_end + 1) - tp_curve
        tpr = np.r_[0.0, tp_curve / positives]
        fpr = np.r_[0.0, fp_curve / negatives]
        auroc = float(np.trapz(tpr, fpr))
        precisions = tp_curve / (tp_curve + fp_curve)
        recalls = tp_curve / positives
        auprc = float(np.sum(np.diff(np.r_[0.0, recalls]) * precisions))
    return {
        "count": n, "positives": positives, "precision": precision,
        "recall": recall, "f1": f1, "auroc": auroc, "auprc": auprc,
        "false_negative_rate": fn / positives if positives else None,
    }


def _score_dataset(dataset, model, normalizer, batch_size):
    rows = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    offset = 0
    for batch in loader:
        outputs = predict(model, normalizer, batch["obs"], batch["action"])
        size = batch["obs"].shape[0]
        values = {key: tensor.detach().cpu().numpy() for key, tensor in outputs.items()}
        truth = batch["target"].numpy()
        masks = batch["event_mask"].numpy()
        for i in range(size):
            scene_id, trajectory_id, t = dataset.metadata(offset + i)
            rows.append({
                "scene_id": scene_id, "trajectory_id": trajectory_id, "t": int(t),
                "true_dr_mean": float(truth[i, 0]),
                "pred_dr_mean": float(values["dr_mean"][i]),
                "true_dr_max": float(truth[i, 1]),
                "pred_dr_max": float(values["dr_max"][i]),
                "true_topple": int(truth[i, 2]) if masks[i, 0] else None,
                "topple_prob": float(values["topple_prob"][i]),
                "true_fall": int(truth[i, 3]) if masks[i, 1] else None,
                "fall_prob": float(values["fall_prob"][i]),
            })
        offset += size
    return rows


def _choose_threshold(rows, event):
    known = [row for row in rows if row[f"true_{event}"] is not None]
    if not known:
        return 0.5
    y = [row[f"true_{event}"] for row in known]
    p = [row[f"{event}_prob"] for row in known]
    # Fixed grid, deterministic tie-breaking; test set never influences thresholds.
    candidates = [round(t / 100, 2) for t in range(5, 100, 5)]
    scored = [(float(_event_metrics(y, p, t)["f1"] or 0), -abs(t - .5), t) for t in candidates]
    return max(scored)[2]


def _summarize(rows, thresholds, high_dr_threshold=0.1):
    metrics = {}
    for name in ("dr_mean", "dr_max"):
        metrics[name] = _regression_metrics(
            [row[f"true_{name}"] for row in rows],
            [row[f"pred_{name}"] for row in rows],
        )
    high = [row for row in rows if row["true_dr_mean"] >= high_dr_threshold]
    metrics["high_dr_mean"] = {
        "threshold": high_dr_threshold,
        **_regression_metrics(
            [row["true_dr_mean"] for row in high],
            [row["pred_dr_mean"] for row in high],
        ),
    }
    for event in ("topple", "fall"):
        known = [row for row in rows if row[f"true_{event}"] is not None]
        metrics[event] = {
            "threshold": thresholds[event],
            **_event_metrics(
                [row[f"true_{event}"] for row in known],
                [row[f"{event}_prob"] for row in known],
                thresholds[event],
            ),
        }
    return metrics


def _save_scatter(rows, figures_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name in ("dr_mean", "dr_max"):
        truth = np.asarray([row[f"true_{name}"] for row in rows])
        predictions = np.asarray([row[f"pred_{name}"] for row in rows])
        fig, axis = plt.subplots(figsize=(5, 5))
        axis.scatter(truth, predictions, s=5, alpha=.4)
        extent = [min(float(truth.min()), float(predictions.min())),
                  max(float(truth.max()), float(predictions.max()))]
        axis.plot(extent, extent, linestyle="--", linewidth=1, color="black")
        axis.set(xlabel="Ground truth", ylabel="Predicted", title=name)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{name}_scatter.png", dpi=140)
        plt.close(fig)
    return True


def evaluate_predictor(dataset_root, checkpoint, output_dir, device="cpu", batch_size=256,
                       allow_missing_event_labels=False, high_dr_threshold=0.1):
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation output directory must be empty: {output_dir}")
    if batch_size < 1 or high_dr_threshold < 0:
        raise ValueError("Invalid evaluation batch size or high-DR threshold")
    model, normalizer = load_predictor(checkpoint, device=device)
    split_path = dataset_root / "split.json"
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Predictor evaluation requires the frozen training split: {split_path}"
        )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validate_scene_split(split, dataset_root)
    test_set = PredictorWindowDataset(
        dataset_root, split["test"], model.obs_horizon, model.action_horizon,
        allow_missing_event_labels=allow_missing_event_labels,
    )
    val_set = (
        PredictorWindowDataset(
            dataset_root, split["validation"], model.obs_horizon, model.action_horizon,
            allow_missing_event_labels=allow_missing_event_labels,
        ) if split["validation"] else None
    )
    validation_rows = _score_dataset(val_set, model, normalizer, batch_size) if val_set else []
    thresholds = {
        event: _choose_threshold(validation_rows, event) for event in ("topple", "fall")
    }
    test_rows = _score_dataset(test_set, model, normalizer, batch_size)
    summary = _summarize(test_rows, thresholds, high_dr_threshold)
    scene_groups = defaultdict(list)
    for row in test_rows:
        scene_groups[row["scene_id"]].append(row)
    scenes = {
        scene_id: _summarize(group, thresholds, high_dr_threshold)
        for scene_id, group in scene_groups.items()
    }
    summary.update({
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_root),
        "test_scenes": split["test"],
        "test_windows": len(test_rows),
        "validation_windows_for_threshold": len(validation_rows),
        "threshold_source": "validation" if validation_rows else "default_0.5",
        "scene_metrics": scenes,
        "test_missing_event_windows": test_set.missing_event_windows,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    with (output_dir / "scene_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene_id", "windows", "dr_mean_mae", "dr_max_mae",
                         "topple_recall", "fall_recall"])
        for scene_id, metrics in scenes.items():
            writer.writerow([
                scene_id, len(scene_groups[scene_id]),
                metrics["dr_mean"]["mae"], metrics["dr_max"]["mae"],
                metrics["topple"]["recall"], metrics["fall"]["recall"],
            ])
    summary["scatter_plots_saved"] = _save_scatter(test_rows, output_dir / "figures")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate predictor on held-out test scenes")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--allow-missing-event-labels", action="store_true")
    parser.add_argument("--high-dr-threshold", type=float, default=0.1)
    args = parser.parse_args(argv)
    report = evaluate_predictor(
        args.dataset, args.checkpoint, args.output_dir,
        device=args.device, batch_size=args.batch_size,
        allow_missing_event_labels=args.allow_missing_event_labels,
        high_dr_threshold=args.high_dr_threshold,
    )
    print(json.dumps({k: report[k] for k in ("test_windows", "dr_mean", "dr_max", "topple", "fall")},
                     indent=2))
    return report


if __name__ == "__main__":
    main()
