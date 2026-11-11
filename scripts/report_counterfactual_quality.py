"""Audit a saved physical counterfactual dataset without a simulator or GPU."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np


HORIZONS = (4, 8, 16, 32)


def _rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(root, *, min_gap=.01):
    root = Path(root)
    summary = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    rows = _rows(root/"states.jsonl")
    skipped = _rows(root/"skipped_probes.jsonl")
    horizons = tuple(summary.get("consequence_horizons", [8]))
    if len(rows) != summary["num_states"]:
        raise ValueError("Manifest and summary state counts differ")

    failures = Counter()
    mismatch_fields = Counter()
    for row in skipped:
        reason = row["reason"].lower()
        if "observation" in reason:
            kind = "observation_mismatch"
        elif "terminated" in reason:
            kind = "early_termination"
        elif "truncated" in reason:
            kind = "truncation"
        elif "reproduce" in reason or "signature" in reason:
            kind = "state_mismatch"
        elif "dr_" in reason or "topple" in reason or "fall" in reason:
            kind = "physical_label_mismatch"
        else:
            kind = "other"
        failures[kind] += 1
        obs = row.get("details", {}).get("observation", {})
        if obs.get("largest_field"):
            mismatch_fields[obs["largest_field"]] += 1

    all_labels = []
    eligible_pairs = 0
    stratum_counts = Counter(row["stratum"] for row in rows)
    per_scene = Counter(row["scene_id"] for row in rows)
    for row in rows:
        with np.load(root/row["file"], allow_pickle=False) as archive:
            labels = (archive["targets_multi"].copy()
                      if "targets_multi" in archive.files
                      else archive["targets"][:, None, :].copy())
        if labels.shape != (summary["num_candidates_per_state"], len(horizons), 4):
            raise ValueError(f"Malformed physical targets in {row['state_id']}")
        if not np.isfinite(labels).all():
            raise ValueError(f"Nonfinite label in {row['state_id']}")
        all_labels.append(labels)
        # Same-state pairs only. Multi-Horizon uses the training objective
        # weights to make this diagnostic directly comparable to training.
        if horizons == HORIZONS:
            weights = np.asarray([.35, .25, .25, .15])
        else:
            weights = np.ones(len(horizons)) / len(horizons)
        risk = np.sum(
            (labels[..., 0] + .25*labels[..., 1] + .5*labels[..., 2])
            * weights[None, :], axis=1)
        left, right = np.triu_indices(len(risk), 1)
        eligible_pairs += int((np.abs(risk[left] - risk[right]) >= min_gap).sum())

    per_h = {}
    delayed = {}
    if all_labels:
        labels = np.concatenate(all_labels, axis=0)
        for i, h in enumerate(horizons):
            per_h[str(h)] = {
                "topple_candidates": int(labels[:, i, 2].sum()),
                "fall_candidates": int(labels[:, i, 3].sum()),
                "mean_dr_mean": float(labels[:, i, 0].mean()),
                "mean_dr_max": float(labels[:, i, 1].mean()),
            }
        if 8 in horizons and 16 in horizons:
            safe8 = labels[:, horizons.index(8), 2] == 0
            delayed["h8_safe_to_h16_topple"] = int(
                (safe8 & (labels[:, horizons.index(16), 2] > 0)).sum())
        if 8 in horizons and 32 in horizons:
            safe8 = labels[:, horizons.index(8), 2] == 0
            delayed["h8_safe_to_h32_topple"] = int(
                (safe8 & (labels[:, horizons.index(32), 2] > 0)).sum())

    requested = summary.get("requested_strata")
    return {
        "complete": summary["complete"],
        "states": len(rows),
        "candidates_per_state": summary["num_candidates_per_state"],
        "source_scenes": len(per_scene),
        "per_scene_min_max": [min(per_scene.values()), max(per_scene.values())] if per_scene else [0, 0],
        "strata": dict(stratum_counts),
        "requested_strata": requested,
        "stratum_shortfall": (
            {k: max(0, v-stratum_counts[k]) for k, v in requested.items()}
            if requested else None),
        "skips": len(skipped),
        "skip_reasons": dict(failures),
        "observation_mismatch_largest_fields": dict(mismatch_fields),
        "eligible_ranking_pairs": eligible_pairs,
        "horizons": per_h,
        "delayed_events": delayed,
        "hold_policy": summary.get("continuation_policy"),
        "sampling_strategy": summary.get("probe_strategy", "legacy"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--min-gap", type=float, default=.01)
    args = parser.parse_args()
    report = summarize(args.root, min_gap=args.min_gap)
    path = args.root/"quality_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
