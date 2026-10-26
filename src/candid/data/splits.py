"""Automatically freeze a reproducible train/validation/test split by scene.

The first training run creates <dataset>/split.json. Later runs always reuse
it; changing the dataset fails loudly instead of silently reshuffling test data.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path

SPLIT_NAMES = ("train", "validation", "test")


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for filename in ("scenes.jsonl", "trajectories.jsonl"):
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / filename).read_bytes())
    return digest.hexdigest()


def _canonical(value):
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _layout_key(scene: dict) -> str:
    """Keep obvious duplicate physical layouts in the same split.

    If geometry is not recorded, treat scenes as distinct; this cannot detect
    all physically equivalent scenes, especially across differing metadata.
    """
    geometry = None
    if (scene.get("stable_initial_target_pose") is not None
            and scene.get("stable_initial_distractor_poses") is not None):
        geometry = {
            "target": scene["stable_initial_target_pose"],
            "distractors": scene["stable_initial_distractor_poses"],
            "dimensions": scene.get("distractor_dimensions_m"),
            "assets": scene.get("distractor_asset_ids"),
        }
    elif scene.get("layout") is not None:
        geometry = {"layout": scene["layout"]}
    if geometry is None:
        return "unique:" + scene["scene_id"]
    signature = {
        "env_id": scene.get("env_id"),
        "env_module": scene.get("env_module"),
        "env_kwargs": scene.get("env_kwargs"),
        "geometry": geometry,
    }
    payload = json.dumps(_canonical(signature), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ratios(val_ratio: float, test_ratio: float) -> dict[str, float]:
    val_ratio = float(val_ratio)
    test_ratio = float(test_ratio)
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1:
        raise ValueError("Validation and test ratios must be in [0, 1)")
    train_ratio = 1.0 - val_ratio - test_ratio
    if train_ratio <= 0:
        raise ValueError("Validation and test ratios must sum to less than 1")
    if test_ratio == 0:
        raise ValueError("Test ratio must be greater than zero for a held-out test set")
    return {"train": train_ratio, "validation": val_ratio, "test": test_ratio}


def _target_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    precise = {name: total * ratios[name] for name in SPLIT_NAMES}
    counts = {name: int(precise[name]) for name in SPLIT_NAMES}
    remainder = total - sum(counts.values())
    order = sorted(SPLIT_NAMES, key=lambda name: -(precise[name] - counts[name]))
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def make_scene_split(root, seed: int = 0, val_ratio: float = 0.15,
                     test_ratio: float = 0.15) -> dict:
    root = Path(root)
    ratios = _ratios(val_ratio, test_ratio)
    scenes = _rows(root / "scenes.jsonl")
    ids = [scene["scene_id"] for scene in scenes]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Dataset must contain unique, non-empty scene IDs")
    groups = defaultdict(list)
    for scene in scenes:
        groups[_layout_key(scene)].append(scene["scene_id"])
    active_splits = [name for name in SPLIT_NAMES if ratios[name] > 0]
    if len(groups) < len(active_splits):
        raise ValueError(
            f"Need at least {len(active_splits)} different scene layouts for "
            f"{', '.join(active_splits)}; found {len(groups)}"
        )

    rng = random.Random(seed)
    items = [sorted(members) for members in groups.values()]
    rng.shuffle(items)
    items.sort(key=lambda members: -len(members))
    targets = _target_counts(len(scenes), ratios)
    assigned = {name: [] for name in SPLIT_NAMES}
    assigned_groups = {name: 0 for name in SPLIT_NAMES}
    for index, members in enumerate(items):
        remaining_groups = len(items) - index
        empty = [name for name in active_splits if assigned_groups[name] == 0]
        choices = empty if remaining_groups == len(empty) else active_splits
        best = max(choices, key=lambda name: (
            targets[name] - len(assigned[name]), -SPLIT_NAMES.index(name)))
        assigned[best].extend(members)
        assigned_groups[best] += 1

    split = {
        "schema_version": 1,
        "seed": int(seed),
        "ratios": ratios,
        "dataset_sha256": _fingerprint(root),
        "num_scenes": len(scenes),
        "num_layout_groups": len(groups),
        **{name: sorted(assigned[name]) for name in SPLIT_NAMES},
    }
    validate_scene_split(split, root)
    return split


def validate_scene_split(split: dict, root) -> None:
    root = Path(root)
    if split.get("schema_version") != 1:
        raise ValueError("Unsupported scene split format")
    if split.get("dataset_sha256") != _fingerprint(root):
        raise ValueError(
            f"Dataset changed since {root / 'split.json'} was created. "
            "To intentionally make a NEW split, archive or delete split.json "
            "and rerun training. Old model evaluations will not be comparable."
        )
    ratios = split.get("ratios")
    if not isinstance(ratios, dict):
        raise ValueError("Missing ratios in split.json")
    scene_rows = _rows(root / "scenes.jsonl")
    all_ids = {scene["scene_id"] for scene in scene_rows}
    seen = set()
    ownership = {}
    for name in SPLIT_NAMES:
        ids = split.get(name)
        if not isinstance(ids, list) or any(not isinstance(s, str) for s in ids):
            raise ValueError(f"Invalid {name} scenes in split.json")
        if float(ratios.get(name, 0)) > 0 and not ids:
            raise ValueError(f"Empty {name} scenes in split.json")
        if float(ratios.get(name, 0)) == 0 and ids:
            raise ValueError(f"Unexpected {name} scenes for zero ratio")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate scene IDs in {name}")
        if seen.intersection(ids):
            raise ValueError(f"Scene leakage across train/validation/test: {seen.intersection(ids)}")
        seen.update(ids)
        for scene_id in ids:
            ownership[scene_id] = name
    if seen != all_ids:
        raise ValueError(
            f"Scene mismatch in split.json: missing={sorted(all_ids - seen)}, "
            f"unknown={sorted(seen - all_ids)}"
        )
    layout_owner = {}
    for scene in scene_rows:
        layout = _layout_key(scene)
        split_name = ownership[scene["scene_id"]]
        if layout in layout_owner and layout_owner[layout] != split_name:
            raise ValueError("Duplicate physical scene layouts leak across splits")
        layout_owner[layout] = split_name


def load_or_create_scene_split(root, seed: int = 0, val_ratio: float = 0.15,
                               test_ratio: float = 0.15) -> dict:
    """Create once while loading training data, then reuse the frozen split."""
    root = Path(root)
    path = root / "split.json"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            split = json.load(handle)
        validate_scene_split(split, root)
        return split

    split = make_scene_split(root, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
    # Stage the complete JSON, then link it into place without overwriting a
    # concurrently created split. The existing file always wins.
    fd, temporary = tempfile.mkstemp(prefix=".split-", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(split, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        try:
            os.link(temporary, path)
        except FileExistsError:
            with path.open(encoding="utf-8") as handle:
                split = json.load(handle)
            validate_scene_split(split, root)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return split
