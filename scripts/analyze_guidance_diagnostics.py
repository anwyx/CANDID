"""Summarize paired guidance risk, action shift, timing, and physical outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_results(folder):
    episodes = folder / "episodes.jsonl"
    if not episodes.is_file():
        raise FileNotFoundError(episodes)
    return [
        json.loads(line) for line in episodes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--baseline-run", type=Path, default=None)
    args = parser.parse_args()
    rows = read_results(args.run_dir)
    baseline = {}
    if args.baseline_run is not None:
        baseline = {r["scene_id"]: r for r in read_results(args.baseline_run)}
    print("scene | risk_delta_mean | risk_decrease% | mean_action_abs | max_action_abs "
          "| gradient_ms | denoiser_ms | ddpm_ms | real_DR | delta_real_DR | delta_topple | delta_success")
    totals = {"replans": 0, "paired": 0, "risk_decreased": 0}
    for row in rows:
        diag_path = row.get("policy_diagnostics_file")
        if not diag_path:
            raise ValueError("Missing per-episode policy diagnostics; enable --diagnose")
        path = Path(diag_path)
        if not path.is_file():
            # Support moving the entire result directory after evaluation.
            path = next(args.run_dir.rglob(path.name), None)
            if path is None:
                raise FileNotFoundError(diag_path)
        diag = json.loads(path.read_text(encoding="utf-8"))
        if not diag.get("paired_count"):
            raise ValueError(f"No paired risk diagnostics for {row['scene_id']}; enable --diagnose")
        entries = [d["paired"] for d in diag["decisions"] if "paired" in d]
        count = len(entries)
        totals["replans"] += len(diag["decisions"])
        totals["paired"] += count
        totals["risk_decreased"] += sum(p["risk_delta"] < 0 for p in entries)
        avg = lambda key: sum(p[key] for p in entries) / count
        timing = diag.get("timing_totals_ms", {})
        reference = baseline.get(row["scene_id"])
        ddr = row["dr_mean"] - reference["dr_mean"] if reference else None
        dt = row["topple_count"] - reference["topple_count"] if reference else None
        ds = int(row["task_success"]) - int(reference["task_success"]) if reference else None
        fmt = lambda value: "NA" if value is None else f"{value:+.5f}"
        print(
            f"{row['scene_id']} | {avg('risk_delta'):+.6f} | "
            f"{100 * sum(p['risk_delta'] < 0 for p in entries) / count:.1f}% | "
            f"{avg('action_abs_mean'):.6f} | {max(p['action_abs_max'] for p in entries):.6f} | "
            f"{timing.get('predictor_gradient_ms', 0):.1f} | "
            f"{timing.get('denoiser_ms', 0):.1f} | {timing.get('ddpm_update_ms', 0):.1f} | "
            f"{row['dr_mean']:.5f} | {fmt(ddr)} | {fmt(dt)} | {fmt(ds)}"
        )
    print(
        f"Total: episodes={len(rows)}, replans={totals['replans']}, "
        f"paired={totals['paired']}, predicted_risk_decrease_rate="
        f"{100 * totals['risk_decreased'] / max(totals['paired'], 1):.2f}%"
    )
    print("The paired risks compare the same observation, not distinct closed-loop trajectories.")


if __name__ == "__main__":
    main()
