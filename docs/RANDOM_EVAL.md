# CANDID: reproducible random-scene evaluation

The prior `--dataset demos/scene1` evaluation (frozen 18 held-out
scenes) is unchanged. The new mode samples fresh scenes from the specified
**registered environment**, settles them once, saves exact initial simulator
state files, and then replays those files for ALL policies.

**Important**: The env module must exist in your local checkout or PYTHONPATH.
For example `envs.candid_dense_clutter_env_random` may be on your server
but is NOT present in the tracked GitHub branch; use the real module and
gym env ID that you have locally. The same file and physics version should
be used for generating and evaluating the suite.

### 1. Update and test

```bash
git switch feat/predictor-ordinal-v2
git pull --ff-only origin feat/predictor-ordinal-v2
PYTHONPATH="$PWD:$PWD/src" python -m pytest \
  tests/test_random_eval_suite.py tests/test_compare_eval_runs.py \
  tests/test_eval_dp.py tests/test_guidance_scene_filter.py -q
```

### 2. Generate a tiny 2-scene smoke suite

Replace env ID and module with those corresponding to your actual env.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_eval_scenes.py \
  --env-id CANDIDDenseClutter-v1 \
  --env-module envs.candid_dense_clutter_env_random \
  --num-scenes 2 --seed 900000 --max-episode-steps 300 \
  --output-dir runs/eval_suites/smoke_2
```

This should create:
`suite.json`, `scenes.jsonl` and
`scenes/random_eval_900000_00000.npz` etc.

### 3. Evaluate both policies on the SAME smoke suite

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_dp.py \
  --eval-suite runs/eval_suites/smoke_2 \
  --checkpoint ckpt/scene1/best_val.pt \
  --output-dir runs/eval_random_smoke_vanilla \
  --safe-dr-mean-max 0.2 --max-steps 300 --exec-horizon 4 \
  --eval-seed 2026 --device cuda --no-video

CUDA_VISIBLE_DEVICES=0 python scripts/eval_dp_guidance.py \
  --eval-suite runs/eval_suites/smoke_2 \
  --dp-checkpoint ckpt/scene1/best_val.pt \
  --predictor-checkpoint runs/predictor_ordinal_expert_v2/checkpoints/best_val.pt \
  --output-dir runs/eval_random_smoke_ordinal \
  --safe-dr-mean-max 0.2 --max-steps 300 --exec-horizon 4 \
  --eval-seed 2026 --strength 0.005 --start-timestep 30 \
  --device cuda --no-video
```

### 4. Generate and evaluate the main 50-scene suite

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_eval_scenes.py \
  --env-id CANDIDDenseClutter-v1 \
  --env-module envs.candid_dense_clutter_env_random \
  --num-scenes 50 --seed 900001 --max-episode-steps 300 \
  --output-dir runs/eval_suites/scene1_random50_v1
```

Vanilla:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_dp.py \
  --eval-suite runs/eval_suites/scene1_random50_v1 \
  --checkpoint ckpt/scene1/best_val.pt \
  --output-dir runs/scene1_random50_vanilla \
  --safe-dr-mean-max 0.2 --max-steps 300 --exec-horizon 4 \
  --eval-seed 2026 --device cuda --no-video
```

Ordinal Guidance:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_dp_guidance.py \
  --eval-suite runs/eval_suites/scene1_random50_v1 \
  --dp-checkpoint ckpt/scene1/best_val.pt \
  --predictor-checkpoint runs/predictor_ordinal_expert_v2/checkpoints/best_val.pt \
  --output-dir runs/scene1_random50_ordinal \
  --safe-dr-mean-max 0.2 --max-steps 300 --exec-horizon 4 \
  --eval-seed 2026 --strength 0.005 --start-timestep 30 \
  --device cuda --no-video
```

Old regression Guidance (same suit, optional):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_dp_guidance.py \
  --eval-suite runs/eval_suites/scene1_random50_v1 \
  --dp-checkpoint ckpt/scene1/best_val.pt \
  --predictor-checkpoint ckpt_file/best_val.pt \
  --output-dir runs/scene1_random50_regression \
  --safe-dr-mean-max 0.2 --max-steps 300 --exec-horizon 4 \
  --eval-seed 2026 --strength 0.005 --start-timestep 30 \
  --device cuda --no-video
```

### 5. Paired scene-by-scene comparison

```bash
python scripts/compare_eval_runs.py \
  runs/scene1_random50_vanilla runs/scene1_random50_ordinal \
  --output runs/scene1_random50_comparison.json
```

A paired bootstrap interval is reported for each physical outcome. Fifty
independently seeded scenes are more informative than eighteen but **do not
guarantee significance**. Scene distribution, seed and evaluation protocol
must remain fixed for a fair comparison.

**NEVER regenerate a new suite between Vanilla and Guidance evaluations.**
To perform a genuinely new experiment, generate a new suite directory with
a different seed, and re-run ALL policies on that new suite.

**Training leakage**: this suite uses newly sampled random layouts rather
than the 120 recorded training/validation/test scenes; different seeds do
not mathematically guarantee geometric novelty. For publication-grade
evaluation, separately audit overlap with training layouts and evaluate
across multiple seeds and environment parameter variations.

If desired, `--max-scenes 10` selects the first ten scenes of the SAME
suite. Use distinct output folders for every evaluation run.
