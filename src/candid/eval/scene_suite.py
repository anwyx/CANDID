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
    if manifest.get("schema_version") != 1 or manifest.get("source") != "random_eval_suite":
        raise ValueError("Invalid CANDID random evaluation suite")
    rows = [
        json.loads(line)
        for line in (root / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [row.get("scene_id") for row in rows]
    if not rows or len(rows) != manifest.get("num_scenes") or len(set(ids)) != len(ids):
        raise ValueError("Scene count or IDs do not match frozen suite manifest")
    for row in rows:
        if row.get("generated_eval_suite") is not True:
            raise ValueError("Scene is not tagged as random evaluation only")
        if row.get("env_id") != manifest["env_id"] or row.get("env_module") != manifest["env_module"]:
            raise ValueError("Suite mixes environment identities")
        if row.get("env_kwargs") != manifest["env_kwargs"]:
            raise ValueError("Suite mixes environment kwargs")
    digest = suite_fingerprint(root)
    if digest != manifest.get("suite_content_sha256"):
        raise ValueError("Frozen suite content changed; regenerate a NEW suite instead")
    return manifest, rows
