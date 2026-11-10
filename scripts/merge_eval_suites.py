"""Merge validated single-environment frozen evaluation suites without resimulation.

Usage:
    python scripts/merge_eval_suites.py --output-dir eval/combined250 \
      --suite scene1=eval/scene1 --suite scene2=eval/scene2 ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT, _ROOT / "src"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))


def _spec_hash(spec):
    return hashlib.sha256(json.dumps(
        spec, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def merge_suites(items, output_dir, expected_scenes=50):
    from candid.eval.scene_suite import load_suite, suite_fingerprint

    if len(items) < 2:
        raise ValueError("Provide at least two suites")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists; refuse overwrite: {output_dir}")
    labels = [label for label, _ in items]
    if len(labels) != len(set(labels)):
        raise ValueError("Group labels must be unique")
    validated = []
    env_registry = {}
    signatures = {}
    for label, source in items:
        source = Path(source).resolve()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", label):
            raise ValueError(f"Invalid suite group label: {label!r}")
        meta, rows = load_suite(source)
        if meta["source"] != "random_eval_suite":
            raise ValueError(f"Expected an original single-environment suite: {source}")
        if expected_scenes is not None and len(rows) != expected_scenes:
            raise ValueError(f"{label}: expected {expected_scenes} scenes, found {len(rows)}")
        specs = {_spec_hash(row["scene_spec"]) for row in rows}
        if len(specs) != 1:
            raise ValueError(f"{label}: inconsistent SceneSpec within source suite")
        n_ids = {len(row["distractor_ids"]) for row in rows}
        if any(n > 6 or n < 1 for n in n_ids):
            raise ValueError(f"{label}: invalid distractor count")
        key = meta["env_id"]
        identity = (meta["env_module"], next(iter(specs)), json.dumps(meta["env_kwargs"], sort_keys=True))
        if key in env_registry and env_registry[key] != identity:
            raise ValueError(
                f"Conflicting registrations for env ID {key!r}; "
                "give each environment a unique @register_env ID"
            )
        env_registry[key] = identity
        signatures[label] = next(iter(specs))
        validated.append((label, source, meta, rows))
    # Same observation adapter supports 1..6 distractors via padding. The
    # source manifests must nevertheless have one consistent distractor order.
    output_dir.mkdir(parents=True)
    try:
        (output_dir / "scenes").mkdir()
        components = []
        with (output_dir / "scenes.jsonl").open("w", encoding="utf-8") as handle:
            for label, source, meta, rows in validated:
                components.append({
                    "label": label,
                    "env_id": meta["env_id"],
                    "env_module": meta["env_module"],
                    "env_kwargs": meta["env_kwargs"],
                    "scene_spec_sha256": signatures[label],
                    "source_suite_sha256": meta["suite_content_sha256"],
                    "num_scenes": len(rows),
                    "source_git_commit": meta.get("git_commit"),
                })
                for row in rows:
                    src = source / row["initial_state"]
                    if not src.is_file():
                        raise FileNotFoundError(src)
                    target_id = f"{label}__{row['scene_id']}"
                    rel = Path("scenes") / f"{target_id}.npz"
                    shutil.copyfile(src, output_dir / rel)
                    merged = dict(row)
                    merged["scene_id"] = target_id
                    merged["initial_state"] = rel.as_posix()
                    merged["suite_group"] = label
                    merged["source_scene_id"] = row["scene_id"]
                    merged["source_suite_sha256"] = meta["suite_content_sha256"]
                    handle.write(json.dumps(merged, ensure_ascii=False, separators=(",", ":")) + "\n")

        manifest = {
            "schema_version": 1,
            "source": "multi_env_eval_suite",
            "num_scenes": sum(len(rows) for _, _, _, rows in validated),
            "max_episode_steps": min(meta["max_episode_steps"] for _, _, meta, _ in validated),
            "components": components,
        }
        manifest["suite_content_sha256"] = suite_fingerprint(output_dir)
        (output_dir / "suite.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8"
        )
        verified, checked = load_suite(output_dir)
        assert len(checked) == manifest["num_scenes"]
        return verified
    except BaseException:
        shutil.rmtree(output_dir)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", required=True,
                        help="LABEL=PATH (repeat for all environments)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-scenes", type=int, default=50)
    args = parser.parse_args(argv)
    sources = []
    for entry in args.suite:
        if "=" not in entry:
            parser.error(f"--suite needs LABEL=PATH, received {entry!r}")
        label, path = entry.split("=", 1)
        sources.append((label, Path(path)))
    result = merge_suites(sources, args.output_dir, args.expected_scenes)
    print(json.dumps({
        "output": str(args.output_dir),
        "scenes": result["num_scenes"],
        "groups": {c["label"]: c["num_scenes"] for c in result["components"]},
        "suite_sha256": result["suite_content_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
