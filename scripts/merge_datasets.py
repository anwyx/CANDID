"""Merge CANDID collection directories without changing their original data.

Only files referenced by the JSONL manifests are merged. Optional videos and
collection summaries are not needed for training and are left in source folders.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from candid.data.validate import validate_dataset


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_file(root: Path, relpath: str) -> tuple[Path, Path]:
    if not isinstance(relpath, str) or not relpath:
        raise ValueError(f"Missing input file path: {relpath!r}")
    rel = Path(relpath)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe input file path: {relpath}")
    candidate = (root / rel).resolve(strict=True)
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError(f"Input file must be inside dataset root: {relpath}")
    return rel, candidate


def merge_datasets(inputs: list[Path], output_dir: Path, mode: str = "copy") -> dict:
    if mode not in ("copy", "symlink"):
        raise ValueError("mode must be copy or symlink")
    if not inputs:
        raise ValueError("Provide at least one input dataset")
    sources = [Path(path).resolve(strict=True) for path in inputs]
    if len(set(sources)) != len(sources):
        raise ValueError("The same input directory was provided more than once")
    dest = Path(output_dir).resolve()
    if dest.exists():
        raise FileExistsError(f"Output already exists: {dest}")
    for src in sources:
        if not src.is_dir():
            raise NotADirectoryError(src)
        if dest == src or dest.is_relative_to(src):
            raise ValueError(f"Output must not be inside input: {src}")
        print(f"Checking source: {src}", flush=True)
        validate_dataset(src)

    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{dest.name}.tmp-", dir=dest.parent))
    all_scenes: list[dict] = []
    all_trajectories: list[dict] = []
    sources_report: list[dict] = []
    try:
        for index, src in enumerate(sources, 1):
            label = f"team_{index:02d}"
            scenes = read_rows(src / "scenes.jsonl")
            trajectories = read_rows(src / "trajectories.jsonl")
            copied: set[Path] = set()

            def include(relpath: str) -> str:
                rel, original = source_file(src, relpath)
                target_rel = Path(label) / rel
                if target_rel not in copied:
                    target = stage / target_rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if mode == "copy":
                        shutil.copy2(original, target)
                    else:
                        target.symlink_to(original)
                    copied.add(target_rel)
                return target_rel.as_posix()

            seen_scene_ids = set()
            for scene in scenes:
                old_id = scene["scene_id"]
                seen_scene_ids.add(old_id)
                record = dict(scene)
                record["scene_id"] = f"{label}__{old_id}"
                record["initial_state"] = include(scene["initial_state"])
                record["source_dataset"] = label
                record["source_scene_id"] = old_id
                all_scenes.append(record)

            for trajectory in trajectories:
                old_scene = trajectory["scene_id"]
                if old_scene not in seen_scene_ids:
                    raise ValueError(f"Unknown source scene: {src}: {old_scene}")
                record = dict(trajectory)
                record["scene_id"] = f"{label}__{old_scene}"
                record["trajectory_id"] = f"{label}__{trajectory['trajectory_id']}"
                record["disturbance_trace"] = include(trajectory["disturbance_trace"])
                if trajectory.get("saved_hdf5"):
                    record["hdf5_file"] = include(trajectory["hdf5_file"])
                record["source_dataset"] = label
                record["source_trajectory_id"] = trajectory["trajectory_id"]
                all_trajectories.append(record)
            sources_report.append({
                "name": label, "source_dir": str(src),
                "num_scenes": len(scenes), "num_trajectories": len(trajectories),
            })
            print(f"{label}: {len(scenes)} scenes, {len(trajectories)} trajectories", flush=True)

        write_rows(stage / "scenes.jsonl", all_scenes)
        write_rows(stage / "trajectories.jsonl", all_trajectories)
        (stage / "merge_sources.json").write_text(
            json.dumps({"mode": mode, "sources": sources_report}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        report = validate_dataset(stage)
        stage.rename(dest)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Merge CANDID datasets without modifying sources")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("copy", "symlink"), default="copy",
                        help="copy (portable, default) or symlink (fast but tied to source paths)")
    args = parser.parse_args(argv)
    report = merge_datasets(args.inputs, args.output_dir, args.mode)
    print(f"MERGED DATASET VALIDATED: {report}")
    print(f"Merged data: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
