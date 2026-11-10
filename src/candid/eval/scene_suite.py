"""Metadata and integrity validation for frozen random evaluation suites."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def suite_fingerprint(root):
    root = Path(root)
    manifest = root / "scenes.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    digest = hashlib.sha256()
    digest.update(manifest.read_bytes())
    for scene in [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]:
        rel = Path(scene["initial_state"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unsafe suite snapshot path: {rel}")
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(str(rel).encode())
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    return digest.hexdigest()


def load_suite(root):
    root = Path(root)
    meta_path = root / "suite.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    manifest = json.loads(meta_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("source") not in (
        "random_eval_suite", "multi_env_eval_suite"
    ):
        raise ValueError("Invalid CANDID evaluation suite")
    rows = [
        json.loads(line)
        for line in (root / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [row.get("scene_id") for row in rows]
    if not rows or len(rows) != manifest.get("num_scenes") or len(set(ids)) != len(ids):
        raise ValueError("Scene count or IDs do not match frozen suite manifest")
    if manifest["source"] == "random_eval_suite":
        for row in rows:
            if row.get("generated_eval_suite") is not True:
                raise ValueError("Scene is not tagged as random evaluation only")
            if row.get("env_id") != manifest["env_id"] or row.get("env_module") != manifest["env_module"]:
                raise ValueError("Suite mixes environment identities")
            if row.get("env_kwargs") != manifest["env_kwargs"]:
                raise ValueError("Suite mixes environment kwargs")
    else:
        import hashlib
        components = manifest.get("components", [])
        if not isinstance(components, list) or len(components) < 2:
            raise ValueError("Multi-environment suite needs two or more components")
        labels = [component.get("label") for component in components]
        if not all(isinstance(label, str) and label for label in labels) or len(set(labels)) != len(labels):
            raise ValueError("Duplicate or invalid environment group labels")
        by_label = {component["label"]: component for component in components}
        counts = {label: 0 for label in by_label}
        for row in rows:
            if row.get("generated_eval_suite") is not True:
                raise ValueError("Merged scene is not tagged as random evaluation")
            label = row.get("suite_group")
            if label not in by_label:
                raise ValueError(f"Unexpected merged scene group: {label!r}")
            component = by_label[label]
            for key in ("env_id", "env_module", "env_kwargs"):
                if row.get(key) != component.get(key):
                    raise ValueError(f"Environment mismatch in merged group {label}: {key}")
            spec = row.get("scene_spec")
            if not isinstance(spec, dict):
                raise ValueError(f"Merged scene {row['scene_id']} lacks SceneSpec")
            signature = hashlib.sha256(
                json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if signature != component.get("scene_spec_sha256"):
                raise ValueError(f"SceneSpec mismatch in merged group {label}")
            if row.get("source_suite_sha256") != component.get("source_suite_sha256"):
                raise ValueError(f"Source checksum mismatch in merged group {label}")
            if not isinstance(row.get("source_scene_id"), str) or not row["source_scene_id"]:
                raise ValueError("Merged scene missing original scene identity")
            counts[label] += 1
        for label, count in counts.items():
            if count != by_label[label].get("num_scenes"):
                raise ValueError(f"Merged scene count mismatch for group {label}")
        if sum(counts.values()) != manifest["num_scenes"]:
            raise ValueError("Merged scene total does not match manifest")
    digest = suite_fingerprint(root)
    if digest != manifest.get("suite_content_sha256"):
        raise ValueError("Frozen suite content changed; regenerate a NEW suite instead")
    return manifest, rows
