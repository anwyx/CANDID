"""Regression tests for the predictor's 8-step horizon and training loop."""
import csv
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from candid.data.synthetic import ACTION_DIM, write_synthetic_dataset
from model.predictor.checkpoint import load_predictor
from model.predictor.config import PredictorConfig
from model.predictor.inference import predict
from model.predictor.train import train_predictor


def test_predictor_default_horizon_and_validation_settings():
    config = PredictorConfig()
    assert config.obs_horizon == 2
    assert config.action_horizon == 8
    assert config.steps == 10_000
    for key in ("log_freq", "eval_freq", "val_max_windows", "val_batch_size"):
        assert getattr(config, key) > 0
    with pytest.raises(ValueError):
        PredictorConfig(eval_freq=0)


def test_predictor_validation_and_checkpoint_roundtrip(tmp_path):
    data = write_synthetic_dataset(tmp_path / "data", num_scenes=4)
    config = PredictorConfig(
        steps=6,
        batch_size=8,
        device="cpu",
        log_freq=2,
        eval_freq=2,
        val_max_windows=3,
        val_batch_size=2,
        seed=3,
    )
    output = tmp_path / "predictor"
    result = train_predictor(data, output, config)
    assert result["steps_completed"] == 6
    assert result["validation_windows"] == 3
    assert result["best_step"] in (2, 4, 6)
    assert result["best_checkpoint"].is_file()
    assert result["last_checkpoint"].is_file()
    assert result["checkpoint"] == result["best_checkpoint"]
    assert result["best_val_loss"] <= result["val_loss"]

    model, normalizer = load_predictor(result["checkpoint"], device="cpu")
    assert model.action_horizon == 8
    score = predict(
        model,
        normalizer,
        torch.zeros(1, config.obs_horizon, model.obs_dim),
        torch.zeros(1, config.action_horizon, ACTION_DIM),
    )
    assert score.shape == (1, 4)
    assert torch.isfinite(score).all()

    with (output / "train_log.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["step"]) for row in rows] == [2, 4, 6]
    assert all(row["val_loss"] for row in rows)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["action_horizon"] == 8
    assert summary["best_step"] == result["best_step"]
    assert summary["test_scenes"]
    assert set(summary["test_scenes"]).isdisjoint(summary["train_scenes"])


def test_predictor_validation_runs_on_last_step(tmp_path):
    data = write_synthetic_dataset(tmp_path / "data", num_scenes=3)
    config = PredictorConfig(
        steps=3, batch_size=8, device="cpu", log_freq=2, eval_freq=2
    )
    result = train_predictor(data, tmp_path / "out", config)
    with (tmp_path / "out" / "train_log.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["step"]) for row in rows] == [2, 3]
    assert all(row["val_loss"] for row in rows)
    assert result["best_step"] in (2, 3)


def test_predictor_without_validation_saves_last_only(tmp_path):
    data = write_synthetic_dataset(tmp_path / "data", num_scenes=3)
    config = PredictorConfig(
        steps=2, batch_size=8, device="cpu",
        log_freq=1, eval_freq=1, val_ratio=0.0, test_ratio=0.3,
    )
    result = train_predictor(data, tmp_path / "out", config)
    assert result["val_loss"] is None
    assert result["best_step"] is None
    assert result["best_checkpoint"] is None
    assert result["checkpoint"] == result["last_checkpoint"]
    assert result["checkpoint"].is_file()
    assert not (tmp_path / "out" / "checkpoints" / "best_val.pt").exists()
