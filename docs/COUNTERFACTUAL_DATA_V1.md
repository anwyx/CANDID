# Physical counterfactual dataset collector (PR 1)

This is the **first data-only step** toward improving CANDID's actual safety.
It does not train Predictor/DP and does not modify Guidance or Rerank.

## Inputs and invariant

- Source must be an existing CANDID expert dataset with `scenes.jsonl`,
  `trajectories.jsonl`, HDF5 trajectories, `disturbance_traces/`,
  archived initial `scenes/*.npz`, and a **frozen `split.json`**.
- The collector uses **only `split["train"]`** and checks the original
  manifest fingerprint by calling `validate_scene_split`.
- `obs_horizon=2`, `action_horizon=8`. The actual guidance evaluation
  continues to use `exec_horizon=4`, unchanged.
- Sampling: near the first Topple event, high local DR, and low-risk controls.
  At most one selected probe per expert trajectory and a bounded number per
  source scene.
- Candidate 0 is the unchanged expert 8-step chunk; other candidates are
  smooth bounded perturbations (gripper coordinate stays unchanged by default).
- **Every candidate**, including the baseline, gets fresh CPU physics from the
  archived *initial* scene snapshot, with the entire expert prefix replayed.
  The resulting `dr_mean`, `dr_max`, `any_topple`, and `any_fall` are
  recomputed from actual physics traces over the selected 8-step window.
  Never copy the expert DR label onto a modified action.
- The replayed expert must agree with its archive (state/observation tolerances,
  DR tolerances, and exact event counts). A mismatch fails rather than silently
  contaminating training data. Do not loosen tolerances simply to force pass.
- This collector currently **refuses to rely on a SceneSpec-only fallback**
  unless explicitly opted in. Best practice is to restore the exact original
  environment Python module and version. The fallback can alter physics.
- `max_prefix_steps` can limit costly full-prefix replays.

## CPU-only tests

```bash
PYTHONPATH=src:. pytest -q tests/test_counterfactual.py
```

## Minimal physical smoke on server

Use the expert dataset actually used for Predictor training, not `eval/scene1`
and not `eval/results/*`. If your source is named differently, replace
`demos/scene1` accordingly.

```bash
PYTHONPATH=src:. python scripts/generate_counterfactuals.py \
  --dataset demos/scene1 \
  --output-dir runs/counterfactual_smoke_v1 \
  --states 2 --candidates 3 \
  --max-prefix-steps 120 --sigma 0.02 \
  --seed 2026
```

The default `--states 2 --candidates 4` means 8 *fresh* candidate
simulations plus expert-prefix replay cost, not just 8 simulator actions.
Use a new empty output folder on every execution.

Outputs:

- `states.jsonl`: one row per reference state, linked to its frozen training
  scene, expert trajectory and probe step.
- `samples/<state_id>.npz`: `obs[2,O]`, `actions[K,8,A]`,
  `targets[K,4]`, per-object DR and Topple/Fall flags.
- `summary.json`: source split fingerprint, sample counts, strata and number
  of candidate Topples. No synthetic labels are included.

**Acceptance**: CPU unit tests pass; physical smoke creates two state files,
three real candidates per state, finite targets, and baseline replay checks
pass without altering the expert source dataset. Report the number of
`states_with_distinct_topple`; zero is possible at small scale and is not
a sign of incorrect code.

## Next PR (not implemented here)

Build a combined Predictor dataset loader, mixed-scene BatchSampler and a
pairwise-ranking-loss ablation. Do not train on the existing held-out 50-scene
evaluation results. The current schema is intentionally separate from the
legacy HDF5 trajectory contract.
