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


## V1.1: Multi-scale counterfactual candidate coverage

The original `make_candidates` and default `--candidate-mode legacy` remain
unchanged for backward compatibility. For more contrast among physically
labeled candidate outcomes use:

```bash
PYTHONPATH=src:. python scripts/generate_counterfactuals.py \
  --dataset demos/scene1 \
  --output-dir runs/counterfactual_multiscale_smoke_v1 \
  --states 6 --candidates 7 --max-per-scene 1 \
  --candidate-mode multiscale --sigma-levels 0.02,0.06,0.12 \
  --max-prefix-steps 160 --seed 2026
```

This executes 42 physical 8-step continuations, each with its own full
expert-prefix replay. Actions 1/2, 3/4 and 5/6 are smooth +/- pairs
at progressively larger amplitudes. All are bounded by the real environment
action space; the gripper dimension is unchanged by default.

The output also records `nominal_sigmas`, per-candidate actual L2 deviations,
`dr_mean_range`, and whether the same-state Topple labels differ. The
`summary.json` includes counts of states with DR range >0.01 / >0.05,
and states with at least one Topple label disagreement.

An all-unsafe state is a legitimate physical result, not a failed replay.
Do not discard all-safe or all-unsafe groups just to make benchmark numbers
look better. If no group contains diverse outcomes, improve state selection
around the event boundary rather than assuming the Predictor can learn
action-conditioned risk from these samples.

Note: V1.1 still collects data only, without fitting any Predictor or
claiming an improvement in safety.


## V1.2: resume interrupted physical collection

Do not change `obs_atol` to suppress source/replay disagreement: it would
risk training on mismatched `(observation, action, physical label)` tuples.

An expert trajectory with inconsistent archived/replayed observation or
baseline physical DR/Topple/Fall is now **skipped and logged**, not used for
training. Only these known source-replay mismatches are skipped; unexpected
errors in environment setup, action generation, or physics still fail loudly.
Deterministic reserve probes make up the requested number of *valid* states
when available.

For an interrupted run produced before V1.2, preserving the two already
completed states, use the EXACT original collection flags and add `--resume`:

```bash
PYTHONPATH=src:. python scripts/generate_counterfactuals.py \
  --dataset demos/scene1 \
  --output-dir runs/counterfactual_multiscale_smoke_v1 \
  --states 6 --candidates 7 --max-per-scene 1 \
  --candidate-mode multiscale --sigma-levels 0.02,0.06,0.12 \
  --max-prefix-steps 160 --seed 2026 --resume
```

The legacy run's manifest records are validated against the original
deterministic probe selection. Saved NPZ candidate outcomes are retained.
A `collection_config.json` is written when resuming; subsequent resume
attempts must match this configuration, including sampling parameters and
frozen source split fingerprint.

- `skipped_probes.jsonl` records unreproducible expert chunks, including the
  exact source trajectory ID, step and reason.
- `summary.json` is updated after each committed valid group and skip.
- `num_states` is the actual count of valid saved groups, not a requested
  target; `requested_states` and `complete` distinguish shortfalls.
- `--reserve-probes 16` adds up to 16 more training-only probes without
  changing the primary six selected states. No test/validation samples are used.

This does not solve the underlying historical observation mismatch. It
prevents such mismatched source samples from contaminating Predictor training
and lets the physically verified data collection continue.
