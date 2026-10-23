"""Print an overview of a CANDID production dataset; no special modes."""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description="Summarize CANDID collected trajectories")
    p.add_argument("root", type=Path)
    args = p.parse_args(argv)
    scenes = [json.loads(s) for s in (args.root / "scenes.jsonl").read_text().splitlines() if s.strip()]
    rows = [json.loads(s) for s in (args.root / "trajectories.jsonl").read_text().splitlines() if s.strip()]
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene_id"]].append(row)
    if not scenes:
        raise ValueError("Empty dataset")
    print(f"Scenes: {len(scenes)}; trajectories: {len(rows)}; "
          f"successes: {sum(bool(r.get('task_success')) for r in rows)}/{len(rows)}")
    print("Selections:", dict(Counter(r.get("sampling_selection", "legacy") for r in rows)))
    print("Topple/fall episodes:",
          sum(bool(r.get("disturbance", {}).get("any_topple")) for r in rows),
          sum(bool(r.get("disturbance", {}).get("any_fall")) for r in rows))
    report = []
    for scene in scenes:
        sid = scene["scene_id"]
        rs = sorted(by_scene.get(sid, []), key=lambda r: int(r["trajectory_index"]))
        if not rs:
            continue
        baseline_succeeded = bool(rs[0].get("task_success"))
        baseline = float(rs[0]["disturbance"]["dr_mean"]) if baseline_succeeded else None
        good = [float(r["disturbance"]["dr_mean"]) for r in rs if r.get("task_success")]
        if baseline_succeeded:
            repeats = [abs(float(r["disturbance"]["dr_mean"])-baseline) for r in rs[1:3]
                       if r.get("task_success")]
            threshold = max(.001, .1*baseline, 3*max(repeats, default=0))
            low = sum(v < baseline-threshold for v in good)
            high = sum(v > baseline+threshold for v in good)
        else:
            low = high = 0
        record = {"seed": scene["seed"], "scene_id": sid,
                  "attempts": len(rs), "success": len(good), "baseline_dr": baseline if baseline is not None else "",
                  "min_success_dr": min(good) if good else "",
                  "max_success_dr": max(good) if good else "",
                  "meaningful_low": low, "meaningful_high": high,
                  "bidirectional": int(low > 0 and high > 0)}
        report.append(record)
        print(f"seed={scene['seed']} success={len(good)}/{len(rs)} "
              f"DR={record['min_success_dr']}..{record['max_success_dr']} "
              f"low={low} high={high}")
    with (args.root / "collection_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    print(f"Scenes with low/high coverage: {sum(r['bidirectional'] for r in report)}/{len(report)}")
    print("Saved:", args.root / "collection_analysis.csv")


if __name__ == "__main__":
    main()
