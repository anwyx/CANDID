# CANDID Command-Line Scripts

This document is the **up-to-date English CLI reference** for the repository's `scripts/` folder. It covers data collection, integrity checks, Diffusion Policy (DP) and Disturbance Predictor training, physical counterfactual replay, closed-loop CANDID Guidance, frozen multi-environment evaluation, comparisons and rendering.

For complete installation and a narrative end-to-end tutorial, read the [root README](../README.md). For code architecture and data contracts, read [`src/candid/README.md`](../src/candid/README.md).

> **Always run commands from the repository root**, with a supported ManiSkill 3 / SAPIEN / MPlib environment. The examples below are templates: `demos/`, `runs/`, `eval/` and `ckpt/` datasets/checkpoints are generated locally or provided separately. None is guaranteed to be present after cloning.

## 1. Environment and conventions

~~~bash
git clone https://github.com/anwyx/CANDID.git
cd CANDID

conda create -n candid python=3.10 -y
conda activate candid

# First install the appropriate PyTorch CUDA (or CPU) wheel from pytorch.org,
# then install the remaining dependencies:
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/src"

python -c "import torch, mani_skill, sapien, mplib; print('CUDA:', torch.cuda.is_available())"
~~~

The full [installation instructions](../README.md#2-installation-conda) cover the PyTorch installation selector and headless graphics considerations. In general use **distinct output paths** for each run. Never silently regenerate a frozen test suite or change its `split.json`.

## 2. Script map

| Script | Purpose |
|---|---|
| `collect.py` | **Canonical** teacher-data collector (feedback/uniform or direct/left/right/over route intents) |
| `validate_dataset.py` | Validate JSONL, HDF5, NPZ and episode timing contracts |
| `analyze_collection.py` | Summarise collection success and physical consequences |
| `merge_datasets.py` | Merge expert datasets with source identity preservation |
| `train_dp.py` / `eval_dp.py` | Train / closed-loop evaluate Vanilla Diffusion Policy |
| `train_predictor.py` / `eval_predictor.py` | Train / offline evaluate regression or ordinal Predictor |
| `generate_counterfactuals.py` | Replay physically perturbed action chunks from expert train scenes |
| `report_counterfactual_quality.py` | Inspect physically labelled counterfactual sample coverage |
| `train_counterfactual_predictor.py` | Train Ordinal Predictor with expert + CF + pairwise ranking losses |
| `build_joint_counterfactuals.py` | Select a provenance-checked five-source CF subset |
| `generate_eval_scenes.py` | Create one immutable random evaluation suite |
| `merge_eval_suites.py` | Merge independent suites into a multi-environment suite |
| `audit_eval_suite.py` | Inspect integrity and validity of evaluation suites |
| `eval_dp_guidance.py` | **CANDID** closed-loop risk-gradient guided DP |
| `eval_dp_predictor.py` | Shadow / rerank / Best-of-N DP and Predictor integration |
| `run_joint5_methods.py` | Historical five-environment training/evaluation wrapper; has prerequisites |
| `compare_eval_runs.py` | Paired physical metrics, protocol verification and bootstrap comparison |
| `summarize_eval_groups.py` | Per-environment metrics for merged evaluation suites |
| `analyze_guidance_diagnostics.py` | Inspect optional same-noise / gradient diagnostics |
| `render_paired_trajectory.py` | Paired TCP trajectory visualisations |
| `render_trajectory_npz.py` / `replay_trajectory_rgb.py` | Other replay/render tools |
| `run_guidance_snr_41.sh` | Historical Scene 1 guidance ablation convenience script |

Legacy / diagnostic entrypoints remain: `run_baseline.py`, `check_scene.py`, `generate_scenes.py`, `preview_scene.py`, `replay_expert.py`, `run_pick_cube.py`, `eval_local_dr.py`, `eval_dp_predictor.py`, `diag/failure_stages.py` and `diag/offline_action_error.py`. These have specialised contracts; the default pipeline is `collect.py` → `train_dp.py` / `train_counterfactual_predictor.py` → `eval_dp_guidance.py`.

Use `python scripts/<name>.py --help` for actual flags; these APIs are not interchangeable.

## 3. Collect and validate expert data

### 3.1 Standard teacher collection

~~~bash
# One-scene smoke; 13 warmup trajectories + at least one subsequent proposal.
python scripts/collect.py \
  --scenes 1 --trajectories-per-scene 14 \
  --seed 42 --no-collision-aware \
  --output-dir demos/smoke_feedback

python scripts/validate_dataset.py demos/smoke_feedback
python scripts/analyze_collection.py demos/smoke_feedback
~~~

Default mode: 13 deterministic warmup proposals followed by scene-local feedback/uniform candidate selection. The collector uses xArm6 + Robotiq, one CPU physical simulation environment, `pd_joint_pos` actions and `state_dict` observations.

**Important:** by default it also tries **one additional collision-aware reference** per scene. Use `--no-collision-aware` when you want the specified number of attempts to be exact. Failed attempts remain in manifests, even if they generated no HDF5 action frames.

### 3.2 Route-intent collection

~~~bash
python scripts/collect.py \
  --env-id CANDIDDenseClutter-v1 \
  --env-module envs.candid_dense_clutter_env_random \
  --scenes 30 --trajectories-per-scene 24 \
  --route-intents --no-collision-aware \
  --seed 1000 --output-dir demos/dense_routes
~~~

`--route-intents` explores **direct / left / right / over** transfer waypoints in groups of four; the trajectory count must be divisible by four. Each alternative starts from the **same archived initial physical scene**, not a freshly randomised layout.

Additional useful flags: `--env-kwargs '{"max_settle_steps":400}'`, `--num-distractors`, `--save-video`, `--max-episode-steps`, `--variant-seed` and `--verbose`. Use an environment ID/module pair that actually corresponds to the same scene definition.

For custom scenes or external planners, the older `run_baseline.py` entrypoint provides manual variants and additional debugging modes. Its flags are *not* the same as `collect.py`.

### 3.3 Merge collections

~~~bash
python scripts/merge_datasets.py \
  --inputs demos/team_a demos/team_b \
  --output-dir demos/combined \
  --mode copy

python scripts/validate_dataset.py demos/combined
~~~

The input datasets must already exist and be valid. `--mode symlink` reduces storage at the cost of making the result depend on the source directories' availability.

## 4. Train the Vanilla Diffusion Policy

Training creates one frozen **scene-level** `split.json` in the dataset if absent (default 70% train, 15% validation, 15% test). Reusing a dataset with a changed manifest invalidates the stored fingerprint.

~~~bash
# Quick training smoke, not a meaningful final model.
python scripts/train_dp.py \
  --dataset demos/combined \
  --output-dir runs/dp_smoke \
  --total-iters 1000 --batch-size 64 \
  --eval-freq 250

# Example full run (requires sufficient data and compute).
python scripts/train_dp.py \
  --dataset demos/combined \
  --output-dir runs/dp_joint \
  --total-iters 30000 --batch-size 1024 \
  --lr 0.0001 --eval-freq 500 \
  --obs-horizon 2 --act-horizon 8 --pred-horizon 16
~~~

This trains only on **successful recorded expert trajectories** and saves `checkpoints/best_val.pt`, `checkpoints/vanilla_dp.pt`, `loss_history.csv` and `loss_curve.png`. The DP uses 100 DDPM denoising steps by default; observation/action normalisation settings are stored in checkpoint metadata where applicable.

For CPU-only experiments append `--no-cuda`; use a smaller batch size on limited-GPU-memory servers. Temporal regularisers and observation normalisation are optional and **disabled by default**.

## 5. Train and evaluate a standalone Predictor

### 5.1 Ordinal Predictor from recorded expert windows

The ordinary Predictor entrypoint defaults to **regression**, so opt into ordinal modelling explicitly:

~~~bash
python scripts/train_predictor.py \
  --dataset demos/combined \
  --output-dir runs/predictor_ordinal \
  --dr-objective ordinal \
  --dr-thresholds 0.0001,0.001,0.005,0.02,0.05,0.1 \
  --obs-horizon 2 --action-horizon 8 \
  --steps 10000 --batch-size 128 --lr 0.0003 \
  --eval-freq 500 --device auto
~~~

The Predictor receives a history of two 151-D observations and an 8×7 action chunk. It predicts DR-threshold exceedance probabilities and topple/fall logits, **not guaranteed calibrated whole-episode DR**.

Outputs: `summary.json`, `train_log.csv`, `checkpoints/best_val.pt` and `last.pt`.

### 5.2 Offline test-set Predictor evaluation

~~~bash
python scripts/eval_predictor.py \
  --dataset demos/combined \
  --checkpoint runs/predictor_ordinal/checkpoints/best_val.pt \
  --output-dir runs/predictor_offline_test --device cpu
~~~

Outputs include prediction rows, scene-level metrics and summary statistics. Offline metrics do **not** substitute for evaluating physical robot motion in SAPIEN.

## 6. Counterfactual physical supervision and ranking

### 6.1 Generate physically relabelled action groups

The source expert dataset must have a frozen `split.json`. If you have not trained a model yet, initialise it once:

~~~bash
python - <<'PY'
from candid.data.splits import load_or_create_scene_split
root = "demos/combined"
split = load_or_create_scene_split(root)
print({k: len(split[k]) for k in ("train", "validation", "test")})
PY
~~~

Start with a small **physical replay** smoke:

~~~bash
python scripts/generate_counterfactuals.py \
  --dataset demos/combined --output-dir runs/cf_smoke \
  --states 2 --candidates 3 \
  --candidate-mode multiscale \
  --sigma-levels 0.02,0.06,0.12 \
  --seed 2026
~~~

Then request a larger collection, provided the source offers enough valid replay states:

~~~bash
python scripts/generate_counterfactuals.py \
  --dataset demos/combined --output-dir runs/cf_64 \
  --states 64 --candidates 7 --max-per-scene 8 \
  --probe-strategy balanced \
  --candidate-mode multiscale \
  --sigma-levels 0.02,0.06,0.12 --seed 2026
~~~

Candidate 0 is the original expert chunk; others are bounded perturbations. **All candidates**, including candidate 0, get fresh physical replay after restoring the initial scene and reproducing the expert prefix. This can be much slower than rolling out eight steps directly.

Inspect `runs/cf_64/summary.json`. Incomplete collection can be resumed with the **same settings** and `--resume`; replay-alignment failures should be investigated rather than ignored.

### 6.2 Train the counterfactual Ordinal Predictor

~~~bash
python scripts/train_counterfactual_predictor.py \
  --dataset demos/combined \
  --counterfactual-dataset runs/cf_64 \
  --output-dir runs/predictor_cf \
  --device auto --steps 10000 \
  --batch-size 64 --cf-batch-size 8 --lr 0.0001 \
  --expert-weight 1.0 --cf-weight 0.3 \
  --rank-weight 0.5 --cf-val-ratio 0.2 \
  --eval-freq 250 --seed 2026
~~~

The optimiser combines: (1) expert DR/event supervision, (2) physically relabelled CF supervision, and (3) within-state candidate ranking. A separate CF scene holdout protects the validation objective; the code also excludes those scenes from scratch training's expert windows.

The resulting `runs/predictor_cf/checkpoints/best_val.pt` can be loaded by the existing Guidance/Rerank scripts. For historical fine-tuning, `--init-checkpoint PATH` is available; check for possible pretraining/CF holdout overlap.

Reference: [CF generation](../docs/COUNTERFACTUAL_DATA_V1.md) and [CF Predictor training](../docs/COUNTERFACTUAL_PREDICTOR_TRAINING.md).

## 7. Frozen-suite closed-loop inference

Inference here means **simulated** evaluation with current state observations and recurrent replanning, not control of a real robot. The same initial scenes and policy evaluation protocol must be used for every method.

### 7.1 Generate a frozen suite

~~~bash
python scripts/generate_eval_scenes.py \
  --env-id CANDIDDenseClutter-v1 \
  --env-module envs.candid_dense_clutter_env_random \
  --num-scenes 20 --seed 900001 \
  --max-episode-steps 300 \
  --output-dir eval/dense20
~~~

This writes a suite fingerprint in `suite.json`. Do **not** regenerate `eval/dense20` between policies. For multi-environment tests, use `merge_eval_suites.py` with multiple independent frozen suites.

### 7.2 Vanilla DP

~~~bash
python scripts/eval_dp.py \
  --eval-suite eval/dense20 \
  --checkpoint runs/dp_joint/checkpoints/best_val.pt \
  --output-dir runs/eval_vanilla \
  --safe-dr-mean-max 0.2 \
  --max-steps 300 --exec-horizon 4 \
  --eval-seed 2026 --device auto --no-video
~~~

### 7.3 CANDID: risk-guided SNR-Gaussian Capped DDPM

**This is the final project configuration**, not the weaker generic defaults in `model/guidance/config.py`:

~~~bash
python scripts/eval_dp_guidance.py \
  --eval-suite eval/dense20 \
  --dp-checkpoint runs/dp_joint/checkpoints/best_val.pt \
  --predictor-checkpoint runs/predictor_cf/checkpoints/best_val.pt \
  --output-dir runs/eval_candid \
  --safe-dr-mean-max 0.2 \
  --max-steps 300 --exec-horizon 4 --eval-seed 2026 \
  --device auto --no-video \
  --strength 0.02 --start-timestep 30 \
  --guidance-parity odd \
  --strength-schedule snr_gaussian_capped \
  --snr-center-timestep 15 --snr-sigma 1.0 \
  --dr-mean-weight 1.0 --dr-max-weight 0.25 \
  --topple-weight 0.5 --fall-weight 0 \
  --max-grad-norm 1.0 --max-step-norm 0.02
~~~

Guidance is applied at timesteps **29, 27, ..., 1** only; the underlying DDPM still executes its entire denoising schedule. A fresh policy chunk is sampled every four executed robot actions.

Add `--diagnose` when you need additional same-noise risk/action diagnostics. On supported hardware use `--device cuda` for higher throughput.

### 7.4 Best-of-N (unguided sample selection)

~~~bash
python scripts/eval_dp_predictor.py \
  --eval-suite eval/dense20 \
  --dp-checkpoint runs/dp_joint/checkpoints/best_val.pt \
  --predictor-checkpoint runs/predictor_cf/checkpoints/best_val.pt \
  --output-dir runs/eval_bestof4 \
  --mode rerank --num-candidates 4 \
  --min-improvement 0 --max-joint-deviation 1000000 \
  --safe-dr-mean-max 0.2 \
  --max-steps 300 --exec-horizon 4 --eval-seed 2026 \
  --device auto --no-video \
  --dr-mean-weight 1.0 --dr-max-weight 0.25 \
  --topple-weight 0.5 --fall-weight 0
~~~

The very high maximum deviation deliberately disables that additional gating for a minimum-predicted-risk Best-of-4 experiment; it is not a recommended robotics safety bound.

### 7.5 Paired comparison and output files

~~~bash
python scripts/compare_eval_runs.py \
  runs/eval_vanilla runs/eval_candid \
  --output runs/vanilla_vs_candid.json
~~~

Common evaluation outputs: `summary.json`, `episodes.jsonl`, `episodes.csv`, per-scene physical traces and optional videos/diagnostics. Comparison requires matching scene/seed keys, frozen-suite hash, execution horizon, safety limits and evaluation settings. `safe_success_rate` requires **task success and physical safety**, while `success_rate` requires task success only.

## 8. Five-environment workflow helpers

### Merge five scene-suite sources

~~~bash
python scripts/merge_eval_suites.py \
  --suite dense=eval/dense \
  --suite tight=eval/tight \
  --suite large=eval/large \
  --suite diverse=eval/diverse \
  --suite mixed=eval/mixed \
  --output-dir eval/joint250

python scripts/audit_eval_suite.py --help
~~~

The `--suite NAME=PATH` arguments refer to **already generated** single-environment frozen suites, normally 50 scenes each. Their original Python environment modules must be present for physics replay.

### Joint CF subset from existing physics-labelled collections

~~~bash
python scripts/build_joint_counterfactuals.py \
  --dataset demos/joint5 \
  --source demos/dense=runs/cf_dense \
  --source demos/tight=runs/cf_tight \
  --source demos/large=runs/cf_large \
  --source demos/diverse=runs/cf_diverse \
  --source demos/mixed=runs/cf_mixed \
  --per-source 64 --mode copy \
  --output-dir runs/joint5_cf64
~~~

It verifies original-source identity and train-split consistency. You cannot run it with arbitrary CF files or sample sets from held-out scenes.

### Joint training and inference wrapper

`run_joint5_methods.py` provides `--stage train`, `--stage candid` and `--stage bestofn`. **The train stage requires an existing, historically generated** `runs/predictor_ordinal_joint_regularized_v2/summary.json`, which is **not** checked into Git. New users should call `train_counterfactual_predictor.py` directly instead of depending on that wrapper.

Once you have a trained joint DP/Predictor and `eval/joint250`, inference stages are convenient:

~~~bash
python scripts/run_joint5_methods.py \
  --stage candid \
  --dp-checkpoint runs/joint5_dp/checkpoints/best_val.pt \
  --predictor-checkpoint runs/joint5_predictor/checkpoints/best_val.pt \
  --eval-suite eval/joint250 \
  --output-root runs/joint5_eval

python scripts/run_joint5_methods.py \
  --stage bestofn \
  --num-candidates 4 \
  --dp-checkpoint runs/joint5_dp/checkpoints/best_val.pt \
  --predictor-checkpoint runs/joint5_predictor/checkpoints/best_val.pt \
  --eval-suite eval/joint250 \
  --output-root runs/joint5_eval
~~~

The wrapper saves method-specific directories under the same output root and computes per-environment summaries. Do not overwrite existing output folders; use a new root for each new experiment.

For the full five-environment collection and training recipe, see [root README Section 9](../README.md#9-five-environment-experiment-workflow).

## 9. Visualisation, legacy tools, tests

### Qualitative visualisation

- `render_paired_trajectory.py`: paired DP versus CANDID TCP traces in a shared evaluation scene.
- `render_trajectory_npz.py` and `replay_trajectory_rgb.py`: trajectory or RGB replay diagnostics.
- `analyze_guidance_diagnostics.py`: risk gradients, action differences and guidance timing when recorded.
- `report_counterfactual_quality.py`: distribution and validity of real physical counterfactual labels.
- `summarize_eval_groups.py`: per-environment scores for a merged frozen suite.

The visualisation scripts have additional file/scene/step arguments. Use their respective `--help` and the [paired-trajectory guide](../docs/paired_trajectory_visualization.md) rather than guessing screenshot paths.

### Legacy / specialised paths

`run_baseline.py` and its helper `src/candid/runner/collect.py` support manually configured planner variants, scene-only generation and replay from saved scenes; they are not the default teacher collection API. `generate_scenes.py`, `preview_scene.py`, `check_scene.py` and `replay_expert.py` help debug scene layout and teacher execution.

Other specialised scripts—including `eval_local_dr.py`, `run_predictor_v3_ablations.py`, `run_scene1_v2_cf574_once.py` and `run_guidance_snr_41.sh`—reproduce narrower historical experiments. They may expect non-versioned dataset/checkpoint directories; consult their in-file comments before running them on a new checkout.

### Test a code change

~~~bash
python -m pytest \
  tests/test_contract.py \
  tests/test_merge_datasets.py \
  tests/test_predictor_ordinal.py \
  tests/test_predictor_counterfactual.py \
  tests/test_guidance.py \
  tests/test_guidance_snr.py \
  tests/test_joint5_methods.py \
  tests/test_multi_env_suite.py -q

# Full test suite, including simulation-related tests where supported:
python -m pytest tests/ -q
~~~

CPU-only tests cover much of the data and predictor interface. **They do not prove physical sim replay or model performance**. Validate any changed environment and action/controller interface in ManiSkill before committing new results.

## 10. Common pitfalls

| Issue | Guidance |
|---|---|
| `ModuleNotFoundError: candid` | `export PYTHONPATH="$PWD:$PWD/src"` from the repository root. |
| `CUDA was requested` / slow CPU Guidance | Verify a CUDA-enabled PyTorch wheel, driver and `--device`. |
| Vulkan / SAPIEN rendering errors | Check graphics drivers; start with `--no-video` for evaluations. |
| Not enough CF states | Collect more physically verified expert training probes; do not use validation/test scenes. |
| `split.json` mismatch | Dataset manifests changed. Treat as a new experiment; preserve the original split for old checkpoints. |
| Replayed observations or poses disagree | Verify original scene code, asset versions and scene snapshot; never fake or reuse counterfactual labels. |
| `Output exists` failure | Use a fresh run directory. CF resume is allowed only with compatible original settings. |
| Different success or safety numbers | Match frozen scene checksum, seed, safety bounds, max steps and execution horizon. |
| Scene registration collision | Some modules share an environment ID; import the exact originally recorded module. |

More context: [repository README](../README.md), [core internals](../src/candid/README.md), [random-suite protocol](../docs/RANDOM_EVAL.md), [guidance](../model/guidance/README.md).
