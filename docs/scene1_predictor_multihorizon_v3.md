# Scene1 Predictor v3 — physically grounded multi-horizon guidance

**Branch:** feat/predictor-ordinal-v2. No new branch, no changes to DP physics
or disturbance metric. Existing v2 checkpoints and scripts remain usable.

## What is new?

- Delta-q for first six arm joint targets, seventh gripper target unchanged.
- Two feature pathways: full state including velocities, and action-conditioned
  features with object linear/angular velocities masked.
- Eight input actions; physical consequence horizons 4, 8, 16, 32.
- When H>8, simulator holds the **eighth absolute joint target**, not a
  simulated future DP policy. Interpret labels only under this hold protocol.
- H=4 neural head cannot inspect action steps 5–8.
- Mixed expert H=8 supervision, physical same-state multi-horizon ranking,
  and local risk-difference regression.
- Format-v4 checkpoints integrate through existing
  `scripts/eval_dp_guidance.py` without changing DP.

## Important

If an existing guidance evaluation is running from this working checkout,
**finish it before git pull** to avoid mixing code versions.

Always work with `demos/scene1` as the **expert training dataset**.
`eval/scene1` is a held-out evaluation suite; DO NOT use it for
counterfactual training or predictor fitting.

### 1. Checkout and CPU correctness checks

```bash
cd ~/CANDID
conda activate candid
git switch feat/predictor-ordinal-v2
git pull --ff-only origin feat/predictor-ordinal-v2
export PYTHONPATH=src:.
python -m compileall -q model/predictor model/guidance scripts/generate_counterfactuals.py scripts/train_predictor_multihorizon.py
python -m pytest -q tests/test_predictor_multihorizon_v3.py
```

### 2. Small scene1 physics-label smoke (CPU simulation)

```bash
PYTHONPATH=src:. python scripts/generate_counterfactuals.py \
  --dataset demos/scene1 \
  --output-dir runs/cf_scene1_multihorizon_smoke \
  --states 16 --candidates 7 \
  --candidate-mode multiscale \
  --sigma-levels 0.02,0.06,0.12 \
  --consequence-horizons 4,8,16,32 \
  --max-per-scene 4 --reserve-probes 64 \
  --seed 2026
```

The collector must report `"complete": true`; otherwise inspect
`skipped_probes.jsonl` and rerun to a **new** output directory or use
`--resume` with identical arguments.

### 3. GPU smoke training

```bash
PYTHONPATH=src:. python scripts/train_predictor_multihorizon.py \
  --dataset demos/scene1 \
  --counterfactual-dataset runs/cf_scene1_multihorizon_smoke \
  --output-dir runs/predictor_scene1_multihorizon_smoke \
  --steps 20 --eval-freq 10 --log-freq 5 \
  --batch-size 16 --cf-batch-size 4 \
  --device cuda
```

### 4. Integration gradient smoke (one scene, do not compare metrics)

```bash
PYTHONPATH=src:. python scripts/eval_dp_guidance.py \
  --eval-suite eval/scene1 \
  --dp-checkpoint ckpt/scene1/best_val.pt \
  --predictor-checkpoint runs/predictor_scene1_multihorizon_smoke/checkpoints/best_val.pt \
  --output-dir runs/scene1_v3_guidance_smoke \
  --safe-dr-mean-max 0.2 --max-scenes 1 --max-steps 40 \
  --exec-horizon 4 --eval-seed 2026 --device cuda \
  --strength 0.005 --start-timestep 30 \
  --dr-mean-weight 1.0 --dr-max-weight 0.25 --topple-weight 0.5 \
  --max-grad-norm 1.0 --max-step-norm 0.02 --diagnose --no-video
```

### 5. More scene1 counterfactual data after smoke passes

```bash
PYTHONPATH=src:. python scripts/generate_counterfactuals.py \
  --dataset demos/scene1 \
  --output-dir runs/cf_scene1_multihorizon_128 \
  --states 128 --candidates 7 \
  --candidate-mode multiscale \
  --sigma-levels 0.02,0.06,0.12 \
  --consequence-horizons 4,8,16,32 \
  --max-per-scene 16 --reserve-probes 128 \
  --seed 2026
```

```bash
PYTHONPATH=src:. python scripts/train_predictor_multihorizon.py \
  --dataset demos/scene1 \
  --counterfactual-dataset runs/cf_scene1_multihorizon_128 \
  --output-dir runs/predictor_scene1_multihorizon_128 \
  --steps 2000 --eval-freq 250 --log-freq 50 \
  --batch-size 64 --cf-batch-size 8 --device cuda
```

If scene diversity and ranking generalization improve, increase to 512+
**valid** states with an appropriate `--max-per-scene`. Existing collector
still uses full expert-prefix replay per candidate: collection is CPU-intensive.
We intentionally did not replace the known-correct replay with an unverified
state-restoration shortcut.

## Caveats

- These are **physical labels** for a specified action/hold protocol.
  They do not prove future closed-loop safety.
- The v3 architecture is different from v2; initialize from scratch. Do not
  load v2 model weights into v3 incompatible layers.
- The main v3 metric is held-out-scene counterfactual ranking; final validation
  is paired closed-loop Vanilla DP vs v3 Guidance on `eval/scene1`.
- GPU, full scene1 simulation and offline training are not executed by this
  repository change. The tests above must be run in the actual server conda
  environment. Do not claim physical improvement before seeing results.
