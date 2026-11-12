"""Build a leakage-safe joint five-environment CF subset for v2 Predictor.

Requires the merged expert dataset created with scripts/merge_datasets.py
(and its frozen split.json). Takes real, previously simulated candidate labels
from each source CF collection; it NEVER simulates or re-labels physics.

Example:
  python scripts/build_joint_counterfactuals.py \\
    --dataset demos/combined5 --output-dir runs/joint5_cf64_each \\
    --source demos/scene1=runs/cf_scene1_balanced_600 \\
    --source demos/scene2=runs/cf_scene2_balanced_600_replay_consistent \\
    --source demos/scene3=runs/cf_scene3_balanced_600_replay_consistent \\
    --source demos/scene4=runs/cf_scene4_balanced_600_replay_consistent \\
    --source demos/scene5=runs/cf_scene5_balanced_600_replay_consistent
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from candid.data.balanced_counterfactual import stratum_quotas
from candid.data.splits import validate_scene_split


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def choose_balanced(rows, count, seed):
    """Prefer original five strata; fill any absent stratum from remaining states."""
    if len(rows) < count:
        raise ValueError(f"Only {len(rows)} usable train states; requested {count}")
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in rows:
        buckets[row.get("stratum", "unclassified")].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    quotas = stratum_quotas(count)
    chosen = []
    seen = set()
    for stratum, quota in quotas.items():
        for row in buckets.get(stratum, ())[:quota]:
            key = row["state_id"]
            if key not in seen:
                chosen.append(row)
                seen.add(key)
    if len(chosen) < count:
        rest = [row for row in rows if row["state_id"] not in seen]
        rng.shuffle(rest)
        chosen.extend(rest[:count - len(chosen)])
    if len(chosen) != count:
        raise RuntimeError("Subset sampling failed to produce the requested state count")
    return chosen


def build(dataset, sources, output_dir, count=64, seed=2026, mode="symlink"):
    dataset = Path(dataset).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if mode not in ("symlink", "copy") or count < 1:
        raise ValueError("Invalid --mode or --per-source")
    if output_dir.exists():
        raise FileExistsError(f"Output exists: {output_dir}")
    split = read_json(dataset / "split.json")
    validate_scene_split(split, dataset)
    joint_train = set(split["train"])
    merged_sources = read_json(dataset / "merge_sources.json")["sources"]
    mapping = {Path(x["source_dir"]).resolve(): x["name"] for x in merged_sources}
    if len(sources) != 5 or len(set(x.resolve() for x, _ in sources)) != 5:
        raise ValueError("Exactly five distinct source datasets are required")
    if len(mapping) != 5:
        raise ValueError("Expected a five-source merged expert dataset")
    if set(x.resolve() for x, _ in sources) != set(mapping):
        raise ValueError(
            "Source datasets must match the paths in merged/merge_sources.json. "
            "Use the SAME merged dataset on which the joint DP was trained.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".joint-cf-", dir=output_dir.parent))
    try:
        (stage / "samples").mkdir()
        joint_rows = []
        provenance = []
        candidate_counts = set()
        for index, (original_root, cf_root) in enumerate(sources):
            original_root = original_root.resolve(strict=True)
            cf_root = cf_root.resolve(strict=True)
            group = mapping[original_root]
            original_split = read_json(original_root / "split.json")
            validate_scene_split(original_split, original_root)
            cf = read_json(cf_root / "summary.json")
            if (cf.get("labels_from_simulation") is not True
                    or cf.get("source_split") != "train"
                    or cf.get("source_split_sha256") != original_split["dataset_sha256"]
                    or Path(cf.get("source_dataset", "")).resolve() != original_root):
                raise ValueError(f"{cf_root}: CF source split/dataset/physical labels mismatch")
            if cf.get("obs_horizon") != 2 or cf.get("action_horizon") != 8:
                raise ValueError(f"{cf_root}: CF needs obs horizon 2 and action horizon 8")
            candidates = int(cf["num_candidates_per_state"])
            candidate_counts.add(candidates)
            raw = read_rows(cf_root / "states.jsonl")
            if len(raw) != int(cf["num_states"]):
                raise ValueError(f"{cf_root}: states manifest differs from summary")
            eligible = []
            for row in raw:
                merged_scene_id = f"{group}__{row['scene_id']}"
                if row.get("source_split") != "train" or row["scene_id"] not in original_split["train"]:
                    raise ValueError(f"{cf_root}: sample outside source train split")
                if merged_scene_id in joint_train:
                    eligible.append(row)
            picked = choose_balanced(eligible, count, seed + index)
            for ordinal, row in enumerate(picked):
                original_file = Path(row["file"])
                if original_file.is_absolute() or ".." in original_file.parts:
                    raise ValueError(f"Unsafe CF sample path: {original_file}")
                source_sample = (cf_root / original_file).resolve(strict=True)
                if not source_sample.is_relative_to(cf_root):
                    raise ValueError(f"Counterfactual sample escapes source: {original_file}")
                with np.load(source_sample, allow_pickle=False) as blob:
                    if (blob["obs"].shape != (2, 151)
                            or blob["actions"].shape != (candidates, 8, 7)
                            or blob["targets"].shape != (candidates, 4)):
                        raise ValueError(f"Invalid CF sample shape: {source_sample}")
                new_id = f"{group}_{ordinal:04d}"
                target_rel = Path("samples") / f"{new_id}.npz"
                dest = stage / target_rel
                if mode == "copy":
                    shutil.copy2(source_sample, dest)
                else:
                    os.symlink(source_sample, dest)
                updated = dict(row)
                updated["state_id"] = new_id
                updated["scene_id"] = f"{group}__{row['scene_id']}"
                updated["source_trajectory_id"] = f"{group}__{row['source_trajectory_id']}"
                updated["file"] = target_rel.as_posix()
                updated["source_split"] = "train"
                updated["joint_source_cf_dir"] = str(cf_root)
                updated["joint_original_state_id"] = row["state_id"]
                joint_rows.append(updated)
            provenance.append({
                "source_group": group, "expert_dataset": str(original_root),
                "cf_dataset": str(cf_root), "available_physical_states": len(raw),
                "eligible_joint_train_states": len(eligible),
                "selected_states": len(picked),
                "requested_source_states": cf["requested_states"],
                "original_collection_complete": cf["complete"],
            })
            print(f"{group}: selected {len(picked)}/{len(eligible)} eligible CF states "
                  f"from {cf_root}", flush=True)
        if len(candidate_counts) != 1:
            raise ValueError(f"Mixed candidate group sizes: {candidate_counts}")
        requested = len(sources) * count
        if len(joint_rows) != requested:
            raise RuntimeError("Wrong joint sample count")
        with (stage / "states.jsonl").open("w", encoding="utf-8") as handle:
            for row in joint_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "schema_version": 1, "source_dataset": str(dataset),
            "source_split_sha256": split["dataset_sha256"],
            "source_split": "train", "requested_states": requested,
            "num_states": len(joint_rows), "complete": True,
            "num_candidates_per_state": next(iter(candidate_counts)),
            "obs_horizon": 2, "action_horizon": 8,
            "labels_from_simulation": True, "trained_model": False,
            "derived_from_existing_physical_labels": True,
            "no_relabeling_or_resimulation": True,
            "joint_per_source_count": count, "joint_seed": seed,
            "original_collections": provenance,
        }
        (stage / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        stage.rename(output_dir)
        return summary
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path,
                        help="Already merged five-source EXPERT dataset used for joint DP")
    parser.add_argument("--source", action="append", required=True,
                        help="SOURCE_EXPERT_DATASET=SOURCE_CF_DATASET; repeat five times")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-source", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    args = parser.parse_args()
    pairs = []
    for item in args.source:
        if "=" not in item:
            parser.error(f"Invalid source pair {item!r}")
        left, right = item.split("=", 1)
        pairs.append((Path(left), Path(right)))
    summary = build(args.dataset, pairs, args.output_dir,
                    count=args.per_source, seed=args.seed, mode=args.mode)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
