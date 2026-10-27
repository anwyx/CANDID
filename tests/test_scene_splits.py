"""Tests for automatic, frozen scene-based train/validation/test splits."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candid.data.splits import load_or_create_scene_split, validate_scene_split


def _dataset(tmp_path, count=20, duplicate_pairs=False):
    root = tmp_path / "dataset"
    root.mkdir()
    with (root / "scenes.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(count):
            layout = index // 2 if duplicate_pairs else index
            row = {
                "scene_id": f"scene_{index:03d}",
                "env_id": "CANDIDBaseline-v1",
                "env_module": "envs.candid_baseline_env",
                "stable_initial_target_pose": [float(layout), 0, 0, 1, 0, 0, 0],
                "stable_initial_distractor_poses": [[float(layout), 0.1, 0, 1, 0, 0, 0]],
            }
            handle.write(json.dumps(row) + "\n")
    (root / "trajectories.jsonl").write_text("", encoding="utf-8")
    return root


def test_automatic_creation_and_reuse(tmp_path):
    root = _dataset(tmp_path)
    path = root / "split.json"
    assert not path.exists()
    split = load_or_create_scene_split(root, seed=42)
    assert path.exists()
    assert [len(split[key]) for key in ("train", "validation", "test")] == [14, 3, 3]
    saved_bytes = path.read_bytes()
    second = load_or_create_scene_split(root, seed=999, val_ratio=0.2, test_ratio=0.2)
    assert second == split
    assert path.read_bytes() == saved_bytes
    validate_scene_split(second, root)


def test_duplicate_physical_layouts_stay_together(tmp_path):
    root = _dataset(tmp_path, duplicate_pairs=True)
    split = load_or_create_scene_split(root, seed=42)
    owner = {scene: key for key in ("train", "validation", "test") for scene in split[key]}
    for i in range(0, 20, 2):
        assert owner[f"scene_{i:03d}"] == owner[f"scene_{i+1:03d}"]
    assert all(split[name] for name in ("train", "validation", "test"))


def test_stale_split_cannot_silently_reshuffle(tmp_path):
    root = _dataset(tmp_path)
    load_or_create_scene_split(root, seed=42)
    with (root / "trajectories.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"trajectory_id": "a"}\n')
    with pytest.raises(ValueError, match="Dataset changed"):
        load_or_create_scene_split(root, seed=42)


def test_overlapping_test_scene_is_rejected(tmp_path):
    root = _dataset(tmp_path)
    split = load_or_create_scene_split(root)
    split["test"].append(split["train"][0])
    (root / "split.json").write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(ValueError, match="Scene leakage"):
        load_or_create_scene_split(root)


def test_too_few_independent_layouts_are_rejected(tmp_path):
    root = _dataset(tmp_path, count=2)
    with pytest.raises(ValueError, match="at least 3 different scene layouts"):
        load_or_create_scene_split(root)
    assert not (root / "split.json").exists()


def test_disabled_validation_still_has_test_holdout(tmp_path):
    root = _dataset(tmp_path)
    split = load_or_create_scene_split(root, val_ratio=0.0, test_ratio=0.15)
    assert not split["validation"]
    assert split["train"] and split["test"]


def test_training_pipeline_uses_auto_split(tmp_path):
    sys.path.insert(0, str(ROOT))
    from candid.data.synthetic import write_synthetic_dataset
    from model.predictor.config import PredictorConfig
    from model.predictor.train import train_predictor
    from model.vanilla_dp.config import DPConfig
    from model.vanilla_dp.train import train_vanilla_dp

    root = write_synthetic_dataset(tmp_path / "synthetic")
    assert not (root / "split.json").exists()
    policy = train_vanilla_dp(
        root, tmp_path / "dp", DPConfig(total_iters=1, batch_size=8, cuda=False)
    )
    predictor = train_predictor(root, tmp_path / "predictor", PredictorConfig(steps=1))
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    assert policy["train_scenes"] == split["train"]
    assert policy["val_scenes"] == split["validation"]
    assert policy["test_scenes"] == split["test"]
    assert predictor["train_scenes"] == split["train"]
    assert set(policy["train_scenes"]).isdisjoint(policy["test_scenes"])
    assert set(policy["val_scenes"]).isdisjoint(policy["test_scenes"])
    assert policy["checkpoint"].is_file()
    assert predictor["checkpoint"].is_file()
