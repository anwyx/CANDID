"""Summary of each environment group in a combined evaluation."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT, _ROOT / "src"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))


def summarize_groups(suite_dir, results_dir):
    from candid.eval.scene_suite import load_suite
    from candid.eval.metrics import SafetyLimits, summarize_episodes

    meta, scenes = load_suite(suite_dir)
    if meta["source"] != "multi_env_eval_suite":
        raise ValueError("Expected a multi-environment evaluation suite")
    run = json.loads((Path(results_dir) / "summary.json").read_text(encoding="utf-8"))
    if run.get("eval_suite_sha256") != meta["suite_content_sha256"]:
        raise ValueError("Results were not generated with this exact combined suite")
    grouping = {scene["scene_id"]: scene["suite_group"] for scene in scenes}
    with (Path(results_dir) / "episodes.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    by_group = defaultdict(list)
    for row in rows:
        if row["scene_id"] not in grouping:
            raise ValueError(f"Unknown scene in evaluation: {row['scene_id']}")
        by_group[grouping[row["scene_id"]]].append(row)
    bounds = SafetyLimits(**run["safety_limits"])
    report = {}
    for component in meta["components"]:
        label = component["label"]
        subset = by_group[label]
        expected = component["num_scenes"] * int(run["episodes_per_scene"])
        if len(subset) != expected:
            raise ValueError(f"{label}: expected {expected} episodes; found {len(subset)}")
        report[label] = summarize_episodes(subset, bounds)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-suite", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = summarize_groups(args.eval_suite, args.results)
    data = json.dumps(report, indent=2, ensure_ascii=False)
    print(data)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
