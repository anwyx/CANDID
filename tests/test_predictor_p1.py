"""Predictor-only regression tests: lazy windows, event labels, and held-out metrics."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from candid.data.synthetic import write_synthetic_dataset
from model.predictor.config import PredictorConfig
from model.predictor.dataset import PredictorWindowDataset
from model.predictor.evaluate import _event_metrics, evaluate_predictor
from model.predictor.train import train_predictor


def test_lazy_windows_match_in_memory_windows(tmp_path):
    from candid.data.trajectories import load_trajectories
    from candid.data.splits import load_or_create_scene_split
    root = write_synthetic_dataset(tmp_path / "data", num_scenes=4)
    split = load_or_create_scene_split(root)
    disk = PredictorWindowDataset(root, split["train"])
    memory = PredictorWindowDataset(load_trajectories(root), split["train"])
    assert len(disk) == len(memory)
    assert not hasattr(disk, "samples")
    for index in (0, len(disk) // 2, len(disk) - 1):
        on_disk = disk[index]
        in_memory = memory[index]
        for key in ("obs", "action", "target", "event_mask"):
            assert torch.allclose(on_disk[key], in_memory[key])
    assert len(disk._cache) <= disk.cache_size


def test_unknown_event_flags_are_rejected_or_masked(tmp_path):
    root = write_synthetic_dataset(tmp_path / "data", num_scenes=3)
    from candid.data.splits import load_or_create_scene_split
    split = load_or_create_scene_split(root)
    rows = [json.loads(line) for line in (root / "trajectories.jsonl").read_text().splitlines()]
    selected = next(row for row in rows if row["scene_id"] in split["train"])
    trace_path = root / selected["disturbance_trace"]
    with np.load(trace_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files if key != "distractor_toppled"}
    np.savez_compressed(trace_path, **arrays)
    with pytest.raises(ValueError, match="distractor_toppled"):
        PredictorWindowDataset(root, split["train"])
    masked = PredictorWindowDataset(root, split["train"], allow_missing_event_labels=True)
    assert masked.missing_event_trajectories >= 1
    index = next(i for i in range(len(masked)) if masked.metadata(i)[1] == selected["trajectory_id"])
    item = masked[index]
    assert item["event_mask"].tolist() == [0.0, 1.0]


def test_event_metrics_handle_class_imbalance_and_missing_class():
    metrics = _event_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 0.5)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 1
    assert metrics["auroc"] == metrics["auprc"] == 1
    missing = _event_metrics([0, 0], [0.1, 0.4], 0.5)
    assert missing["auroc"] is None
    assert missing["auprc"] is None
    assert missing["recall"] is None


def test_predictor_eval_only_uses_test_scenes(tmp_path):
    root = write_synthetic_dataset(tmp_path / "data", num_scenes=4)
    config = PredictorConfig(steps=3, batch_size=8, device="cpu", eval_freq=2, log_freq=2)
    result = train_predictor(root, tmp_path / "train", config)
    output = tmp_path / "eval"
    summary = evaluate_predictor(root, result["checkpoint"], output, batch_size=4)
    assert summary["test_windows"] > 0
    assert set(summary["scene_metrics"]) == set(result["test_scenes"])
    assert set(summary["test_scenes"]).isdisjoint(result["train_scenes"])
    assert summary["threshold_source"] == "validation"
    assert (output / "summary.json").is_file()
    with (output / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == summary["test_windows"]
    assert {row["scene_id"] for row in rows} == set(result["test_scenes"])
    assert (output / "scene_metrics.csv").is_file()


def test_trajectory_batch_sampler_covers_each_window_once(tmp_path):
    from model.predictor.dataset import TrajectoryBatchSampler
    from candid.data.splits import load_or_create_scene_split
    root = write_synthetic_dataset(tmp_path / "data", num_scenes=4)
    split = load_or_create_scene_split(root)
    dataset = PredictorWindowDataset(root, split["train"])
    sampler = TrajectoryBatchSampler(dataset, batch_size=3, seed=5)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    assert sorted(index for batch in batches for index in batch) == list(range(len(dataset)))
    assert all(len({dataset.metadata(index)[1] for index in batch}) == 1 for batch in batches)


def test_predictor_eval_requires_frozen_split(tmp_path):
    root = write_synthetic_dataset(tmp_path / "data", num_scenes=3)
    from model.predictor.network import DisturbancePredictor
    from model.predictor.normalizer import PredictorNormalizer
    from model.predictor.checkpoint import save_checkpoint
    model = DisturbancePredictor(obs_dim=12, action_dim=7)
    normalizer = PredictorNormalizer(
        np.zeros(12), np.ones(12), np.zeros(7), np.ones(7),
        np.zeros(2), np.ones(2),
    )
    path = save_checkpoint(tmp_path / "sample.pt", model, normalizer, PredictorConfig())
    with pytest.raises(FileNotFoundError, match="frozen training split"):
        evaluate_predictor(root, path, tmp_path / "eval")
