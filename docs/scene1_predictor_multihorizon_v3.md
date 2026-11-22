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

## Balanced counterfactual collection (new opt-in mode)

The original `generate_counterfactuals.py` behavior is unchanged when
`--probe-strategy` is omitted (`legacy`). The balanced mode uses archived
expert observations/traces to locate **candidate** start states; **every**
counterfactual outcome is generated by an independent fresh physics replay,
never borrowed from archive labels.

Five strict target strata: near-contact but low 8-step DR (30%),
low DR controls (25%), pre-topple (20%), slight recent object displacement
(15%), and high DR (10%). The proxy definitions are explicit in
`src/candid/data/balanced_counterfactual.py`:
near-contact = TCP-to-obstacle XY approximate clearance <=10 cm,
TCP/obstacle height distance <=14 cm, H8 DR <=.02 and no topple/fall;
low DR = H8 DR <=.005 without topple/fall;
pre-topple = source topple onset within the following 8 steps, before
the onset itself; slight-motion = previous-action obstacle translation
0.5--10 mm without existing topple/fall; high DR = H8 DR >=.02
without existing topple/fall.
These are **state-selection proxies**: not exact collision distances, and
not labels for perturbed action candidates. At most one state from each
source expert trajectory, and max-per-scene applies to both primary and
reserve probes. If a category runs out of candidates, collection may
finish **incomplete** rather than changing the desired proportions.

Run a 32-state quality pilot in a NEW output directory:

```bash
export PYTHONPATH=src:.
python -m pytest -q tests/test_counterfactual_balanced.py
python scripts/generate_counterfactuals.py \
  --dataset demos/scene1 \
  --output-dir runs/cf_scene1_balanced_32 \
  --states 32 --candidates 7 \
  --candidate-mode multiscale --sigma-levels 0.02,0.06,0.12 \
  --consequence-horizons 4,8,16,32 \
  --probe-strategy balanced --max-per-scene 6 \
  --reserve-probes 96 --seed 2026
python scripts/report_counterfactual_quality.py \
  --root runs/cf_scene1_balanced_32
```

Reuse exactly the same flags and add `--resume` to resume an interrupted
run. Do not resume the old 16-state smoke output using balanced selection:
it has different sampling/configuration constraints.

Inspect `summary.json`, `quality_report.json`, and
`skipped_probes.jsonl`. New observation mismatch skips have structured
per-field max errors (qpos, qvel, TCP, target, obstacles, mask, goal), with
time/feature indices. **Do not increase observation tolerance just to get
more examples.** If failure frequency is high, investigate the largest
field and source recording alignment first.

Only start the larger balanced run if the 32-state pilot has completed with
acceptable replay consistency, diversified physical outcomes and no
unexpected missing strata. The proposed 600-state targets are exactly
180/150/120/90/60, using a separate new output directory and a compatible
max-per-scene capacity.
