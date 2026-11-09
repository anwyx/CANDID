"""Compare policy performance on the exact same frozen evaluation suite.

Refuse to compare runs with different scene suite hashes, action execution
horizons, safety limits, scene order/identities or policy evaluation seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ("task_success", "safe", "safe_success", "dr_mean", "dr_max",
           "topple_count", "fall_count")
BINARY = {"task_success", "safe", "safe_success"}


def load_run(folder):
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    episodes = [
        json.loads(line)
        for line in (folder / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not episodes:
        raise ValueError(f"No episodes: {folder}")
    keys = [(r["scene_id"], r["eval_seed"]) for r in episodes]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate scene/seed results in {folder}")
    return summary, dict(zip(keys, episodes))


def compare_runs(vanilla, guided, *, bootstrap_samples=2000, seed=123):
    va, left = load_run(vanilla)
    gu, right = load_run(guided)
    if va.get("evaluation_source") != "random_frozen_suite" or gu.get("evaluation_source") != "random_frozen_suite":
        raise ValueError("Comparison requires two frozen random evaluation suite runs")
    for field in ("eval_suite_sha256", "exec_horizon", "max_steps",
                  "eval_seed", "stop_on_success", "safety_limits",
                  "episodes_per_scene"):
        if va.get(field) != gu.get(field):
            raise ValueError(f"Evaluation protocol differs: {field}")
    if not va.get("eval_suite_sha256"):
        raise ValueError("Missing frozen suite checksum")
    if left.keys() != right.keys():
        raise ValueError("Runs contain different scene/seed pairs")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")

    keys = sorted(left)
    rng = np.random.default_rng(seed)
    result = {
        "vanilla": str(vanilla), "guided": str(guided),
        "eval_suite_sha256": va["eval_suite_sha256"],
        "episodes": len(keys),
        "metrics": {},
    }
    for key in METRICS:
        baseline = np.asarray([float(left[k][key]) for k in keys])
        intervention = np.asarray([float(right[k][key]) for k in keys])
        delta = intervention - baseline
        out = {
            "vanilla_mean": float(baseline.mean()),
            "guided_mean": float(intervention.mean()),
            "delta_guided_minus_vanilla": float(delta.mean()),
            "improved_count": int(np.sum(delta > 0 if key in BINARY else delta < 0)),
            "worsened_count": int(np.sum(delta < 0 if key in BINARY else delta > 0)),
            "ties": int(np.sum(delta == 0)),
        }
        if bootstrap_samples > 0:
            indexes = rng.integers(0, len(keys), size=(bootstrap_samples, len(keys)))
            estimates = delta[indexes].mean(axis=1)
            out["paired_bootstrap_95pct_ci"] = [
                float(x) for x in np.quantile(estimates, [0.025, 0.975])
            ]
        result["metrics"][key] = out
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired CANDID random-suite outcome comparison")
    parser.add_argument("vanilla", type=Path)
    parser.add_argument("guided", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = compare_runs(
        args.vanilla, args.guided,
        bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    )
    print(f"Paired random-suite episodes: {result['episodes']}")
    for key, metric in result["metrics"].items():
        bounds = metric.get("paired_bootstrap_95pct_ci")
        ci = f" 95%CI={bounds}" if bounds is not None else ""
        print(f"{key}: {metric['vanilla_mean']:.6f} -> "
              f"{metric['guided_mean']:.6f}, "
              f"delta={metric['delta_guided_minus_vanilla']:+.6f}{ci}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return result


if __name__ == "__main__":
    main()
