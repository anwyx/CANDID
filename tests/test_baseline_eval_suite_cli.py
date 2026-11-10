"""Ensure four baseline evaluators accept the same frozen suites as DP."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]

def test_baseline_eval_suite_argument_forwarding():
    path = ROOT / "baselines/common/eval_baseline.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    tree = ast.parse(source)
    assert 'add_mutually_exclusive_group(required=True)' in source
    assert '"--eval-suite"' in source
    assert 'eval_suite=args.eval_suite' in source
    assert 'args.dataset, policy, args.output_dir' in source
    assert 'choices=("act", "bc", "ppo", "sac")' in source
