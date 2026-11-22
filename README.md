# CANDID: Consequence-Aware Non-target Disturbance-Informed Diffusion

CANDID is a **ManiSkill 3 / SAPIEN** research codebase for low-disturbance, long-horizon robotic pick-and-place in clutter. An **xArm6 + Robotiq gripper** transports a target object while a separately trained **Disturbance Predictor** estimates the physical consequences of candidate robot action chunks. At inference time, its differentiable risk can guide a frozen **Diffusion Policy (DP)** toward actions that are less likely to disturb other objects.

The repository includes scene definitions, MPlib teacher demonstrations, physics-based disturbance labels, counterfactual replay, Vanilla DP training, ordinal Predictor training, SNR-scheduled guided sampling, frozen-scene evaluation, optional baselines, and visualisation tools.

> **Scope:** This is a simulation research implementation, **not** a real-robot controller or a safety guarantee. The repository contains source code; training datasets, final trained checkpoints, and experiment outputs must be generated or supplied separately. Run commands from the **repository root** unless explicitly stated otherwise.

## Contents

1. [Repository structure](#1-repository-structure)
2. [Installation (Conda)](#2-installation-conda)
3. [Collect expert demonstrations](#3-collect-expert-demonstrations)
4. [Inspect data and freeze splits](#4-inspect-data-and-freeze-splits)
5. [Train the Vanilla Diffusion Policy](#5-train-the-vanilla-diffusion-policy)
6. [Train a Disturbance Predictor](#6-train-a-disturbance-predictor)
7. [Generate and learn from physical counterfactuals](#7-generate-and-learn-from-physical-counterfactuals)
8. [Run inference and compare policies](#8-run-inference-and-compare-policies)
9. [Five-environment experiment workflow](#9-five-environment-experiment-workflow)
10. [Other baselines, tests, and visualisation](#10-other-baselines-tests-and-visualisation)
11. [Reproducibility and troubleshooting](#11-reproducibility-and-troubleshooting)

## 1. Repository structure

~~~text
CANDID/
├── README.md                     # This guide
├── requirements.txt              # Python dependencies (Linux / Python 3.10)
├── envs/                         # Registered CANDID tabletop environments
│   ├── candid_baseline_env.py
│   ├── candid_primitive_env.py
│   ├── candid_dense_clutter_env_random.py
│   ├── candid_tight_upright_env.py
│   ├── candid_large_clutter_env.py
│   ├── candid_tight_upright_diverse_env.py
│   └── candid_mixed_consequence_env.py
├── planners/
│   └── execute_baseline.py       # xArm6 MPlib pick/lift/transfer/place teacher
├── model/
│   ├── vanilla_dp/               # State-conditioned 1D U-Net, DDPM, training
│   ├── predictor/                # Regression / ordinal disturbance predictors
│   ├── guidance/                 # Risk objectives, gradient guidance, SNR schedule
│   └── integration/              # Shadow and Best-of-N / reranking policies
├── scripts/
│   ├── collect.py                # Canonical expert data collection
│   ├── validate_dataset.py
│   ├── analyze_collection.py
│   ├── merge_datasets.py
│   ├── train_dp.py
│   ├── train_predictor.py
│   ├── eval_dp.py
│   ├── eval_predictor.py
│   ├── generate_counterfactuals.py
│   ├── train_counterfactual_predictor.py
│   ├── build_joint_counterfactuals.py
│   ├── generate_eval_scenes.py
│   ├── merge_eval_suites.py
│   ├── eval_dp_guidance.py
│   ├── eval_dp_predictor.py
│   ├── run_joint5_methods.py
│   ├── compare_eval_runs.py
│   ├── summarize_eval_groups.py
│   └── render_paired_trajectory.py
├── src/candid/
│   ├── sim/                       # SceneSpec, object creation, reset and settling
│   ├── runner/                    # Teacher rollout and route / feedback samplers
│   ├── data/                      # PhysicsTrace, DR metrics, HDF5/NPZ, scene splits
│   └── eval/                      # Saved-scene replay, evaluation, safety metrics
├── baselines/                     # Optional BC, ACT, PPO, SAC implementations
├── docs/                          # Detailed experiment / counterfactual guides
└── tests/                         # CPU contract and simulation integration tests
~~~

Runtime paths used below (`demos/`, `runs/`, `eval/`, `ckpt/`) are **examples**; they are not pre-populated by cloning the repository.

### Central concepts

- **Scene:** a seeded tabletop layout containing a target, goal and movable distractors.
- **Expert trajectory:** a motion-planner attempt from a saved initial scene. Both successes and failures are recorded.
- **Disturbance ratio (DR):** a size-normalised combination of peak distractor translation and rotation. The evaluator also tracks toppling and falling.
- **Vanilla DP:** a pretrained state-conditioned diffusion model generating joint-position action sequences.
- **Predictor:** an independent model consuming two observation frames and an eight-action chunk. The ordinal model outputs DR-threshold exceedance scores and event logits.
- **CANDID Guidance:** inference-time risk-gradient corrections applied to selected DDPM timesteps; DP and Predictor weights stay frozen.

All policies use `state_dict` observations transformed into the canonical **151-dimensional** vector, and **7-dimensional** `pd_joint_pos` actions (six arm joints plus one gripper command). The standard DP observation/prediction/action horizons are **2 / 16 / 8**, with **100 DDPM steps**.

## 2. Installation (Conda)

### 2.1 System requirements

The repository's `requirements.txt` targets **Ubuntu 22.04/24.04, Linux x86_64, Python 3.10**. You need a working ManiSkill/SAPIEN/MPlib installation for physical data collection or rollout evaluation. An NVIDIA GPU is strongly recommended for diffusion training and guided inference, but basic unit tests and some state-based tasks can run on CPU. Rendering/video capture may additionally require a functioning Vulkan graphics driver.

~~~bash
git clone https://github.com/anwyx/CANDID.git
cd CANDID

conda create -n candid python=3.10 -y
conda activate candid

python -m pip install --upgrade pip setuptools wheel
~~~

Install a **CUDA-enabled PyTorch build compatible with your GPU/driver** using the current [official PyTorch installation selector](https://pytorch.org/get-started/locally/) before the other dependencies. For CPU-only development, use the selector's CPU build.

~~~bash
python -m pip install -r requirements.txt

# This repository uses a src/ layout rather than a pip-installable package.
export PYTHONPATH="$PWD:$PWD/src"

python -c "import torch, mani_skill, sapien, mplib, gymnasium, h5py; print('Core imports OK'); print('CUDA:', torch.cuda.is_available())"
python -m pytest tests/test_contract.py tests/test_predictor_ordinal.py -q
~~~

The main dependency pins include `mani-skill==3.0.1`, `sapien==3.0.3`, `mplib==0.1.1` and `diffusers==0.32.2`. The full set is in `requirements.txt`. For a GPU machine, verify `torch.cuda.is_available()` is true; installing a CUDA driver alone does not guarantee a CUDA-enabled PyTorch wheel.

**Use the same shell / environment for every command below**, or add the `PYTHONPATH` export to your Conda activation script. Do not run `pip install -e .`: this repository currently has no packaging configuration for an editable installation.

For video/font support on a minimal Linux image, you may also need system libraries (for example FFmpeg, Vulkan support and TrueType fonts). Try training/evaluation with `--no-video` first on headless servers. The collector always creates a ManiSkill RGB-capable environment, even when it does not save videos, so graphics-driver problems can still require environment setup.

## 3. Collect expert demonstrations

Data collection automatically creates scenes and restores the **same saved simulator state** for each teacher attempt. MPlib executes xArm6 pick-and-place trajectories; SAPIEN measures actual distractor displacement, rotation and events.

### 3.1 Basic collector (feedback/uniform route exploration)

~~~bash
# Minimal collection / environment smoke test (not enough for meaningful training).
python scripts/collect.py \
  --scenes 1 --trajectories-per-scene 14 \
  --seed 42 --no-collision-aware \
  --output-dir demos/collector_smoke

python scripts/validate_dataset.py demos/collector_smoke
python scripts/analyze_collection.py demos/collector_smoke
~~~

For a more useful sample, collect **multiple distinct scenes**:

~~~bash
python scripts/collect.py \
  --scenes 30 --trajectories-per-scene 24 \
  --seed 1000 --no-collision-aware \
  --output-dir demos/baseline_30
~~~

With no `--route-intents` flag, the production sampler begins with 13 fixed warmup proposals and then mixes history-informed feedback and uniform proposals. It is **trajectory-to-trajectory** feedback, not online control during one robot episode. The optional collision-aware reference is a **separate extra attempt**; adding `--no-collision-aware` disables it so `--trajectories-per-scene 24` means exactly 24 planned attempts, rather than 24 plus a reference.

Other useful flags: `--env-id`, `--env-module`, `--env-kwargs` (JSON), `--variant-seed`, `--max-episode-steps`, `--save-video` and `--verbose`. Run `python scripts/collect.py --help` for the authoritative options.

### 3.2 Four route intentions (the paper's demonstration style)

For Direct / Left / Right / Over transfer-route demonstrations, explicitly select `--route-intents`. The number of attempts per scene must be divisible by four.

~~~bash
python scripts/collect.py \
  --env-id CANDIDDenseClutter-v1 \
  --env-module envs.candid_dense_clutter_env_random \
  --scenes 30 --trajectories-per-scene 24 \
  --route-intents --no-collision-aware \
  --seed 2000 \
  --output-dir demos/dense_routes_30
~~~

There are six groups of four route intentions when using 24 attempts. Each trajectory begins from the same frozen scene snapshot; unsuccessful planner attempts are still indexed in the dataset. The route waypoints constrain the **gripper path**, not guaranteed contact-free whole-arm motion. For a paper-scale dataset, increase the scene count after validating a small sample.

### 3.3 Choose an environment

Supply the registered Gym ID **and** the corresponding Python module:

| Gym environment ID | Module | Purpose |
|---|---|---|
| `CANDIDBaseline-v1` | `envs.candid_baseline_env` | Small baseline cube scene (collector default) |
| `CANDIDPrimitive-v1` | `envs.candid_primitive_env` | Mixed primitive objects |
| `CANDIDDenseClutter-v1` | `envs.candid_dense_clutter_env_random` | Randomisable six-object dense clutter |
| `CANDIDTightUpright-v1` | `envs.candid_tight_upright_env` | Tall / tightly spaced clutter |
| `CANDIDLargeClutter-v1` | `envs.candid_large_clutter_env` | Larger clutter obstacles |
| `CANDIDTightUprightDiverse-v1` | `envs.candid_tight_upright_diverse_env` | Diverse tight-upright layouts |
| `CANDIDMixedConsequence-v1` | `envs.candid_mixed_consequence_env` | Mixed consequence-sensitive clutter |
| `CANDIDYCBFruit-v1` | `envs.candid_ycb_fruit_env` | Experimental YCB scene; grasping is not validated |

Several environment modules register the **same** Gym ID (e.g. the static and random DenseClutter variants). Import **only the chosen module** for a given run and preserve its name in the dataset metadata. Different environment modules or later geometry changes can invalidate saved replay states.

For independent team collections, assign distinct seed ranges and fresh output directories; never have two processes write into one dataset. **Collection may be slow**: each recorded trajectory involves motion planning and physical simulation.

## 4. Inspect data and freeze splits

### 4.1 Dataset format

A collected dataset typically contains:

~~~text
demos/example/
├── scenes.jsonl                 # Scene IDs, original environment, seeds, frozen-state references
├── trajectories.jsonl           # Trajectory IDs, outcomes, DR and HDF5/NPZ references
├── scenes/                      # Frozen initial simulator-state .npz files
├── disturbance_traces/          # Per-trajectory poses and physics events .npz
├── trajectories_<run_id>.h5     # Recorded actions and state_dict observations
├── trajectories_<run_id>.json   # Recorder metadata (when produced)
└── collection_summary.csv       # Collection summary
~~~

For `T` actions, the physical pose/event trace has **T+1** state frames. Observations are mapped to the canonical 151-D state by `src/candid/data/adapter.py`. Local Predictor targets cover an **8-step action window**; evaluation DR covers the **whole episode**.

For each non-target object `i`, the disturbance ratio is:

~~~text
DR_i = 0.7 * max_t ||p_i(t) - p_i(0)|| / size_i
     + 0.3 * max_t rotation_angle(q_i(t), q_i(0)) / pi
~~~

`DR_mean` averages objects, and `DR_max` takes the worst affected object. Topple/fall are separate physical event labels. They describe simulated consequences, not contact forces or general real-world safety.

### 4.2 Validate and analyse

~~~bash
python scripts/validate_dataset.py demos/baseline_30
python scripts/analyze_collection.py demos/baseline_30
~~~

Do not train from a partially written collection without checking its manifests and physical traces. Nonempty datasets must contain usable trajectories; planner failures can legitimately have no HDF5 action recording.

### 4.3 Frozen train/validation/test scene splits

Training automatically creates `<dataset>/split.json` the first time a split is needed (default **70% train / 15% validation / 15% test**). The split is **by scene/layout**, not by randomly shuffling individual trajectory windows. Subsequent training runs reuse and validate the stored manifest fingerprint.

If counterfactual collection must happen **before** model training, initialise the scene split directly:

~~~bash
python - <<'PY'
from candid.data.splits import load_or_create_scene_split
s = load_or_create_scene_split("demos/baseline_30", seed=0,
                               val_ratio=0.15, test_ratio=0.15)
print({k: len(s[k]) for k in ("train", "validation", "test")})
PY
~~~

**Do not regenerate or edit `split.json` after training.** A changed dataset or different split means a different experiment, and the code checks for this rather than silently accepting leakage.

### 4.4 Merge collected datasets

~~~bash
python scripts/merge_datasets.py \
  --inputs demos/team_a demos/team_b demos/team_c \
  --output-dir demos/combined \
  --mode copy

python scripts/validate_dataset.py demos/combined
~~~

The merge preserves source identities while rewriting manifests into one dataset. `--mode copy` is portable; `--mode symlink` is faster but requires original source paths to remain accessible. The merged dataset needs its **own frozen split** for joint training.

## 5. Train the Vanilla Diffusion Policy

The state-based Vanilla DP is a conditional 1D U-Net trained with DDPM noise prediction on **successful** expert trajectories. The default horizons are observation 2, prediction 16, executable actions 8.

~~~bash
# Short integration run; not an accuracy benchmark.
python scripts/train_dp.py \
  --dataset demos/baseline_30 \
  --output-dir runs/dp_smoke \
  --total-iters 1000 --batch-size 64 \
  --eval-freq 250 --log-freq 100
~~~

For a larger experiment, train on the intended collected or merged dataset:

~~~bash
python scripts/train_dp.py \
  --dataset demos/combined \
  --output-dir runs/joint_dp \
  --total-iters 30000 --batch-size 1024 \
  --lr 0.0001 --eval-freq 500 \
  --obs-horizon 2 --act-horizon 8 --pred-horizon 16
~~~

Checkpoints and logs are written to:

- `runs/joint_dp/checkpoints/best_val.pt` — best EMA weights by validation noise-prediction loss.
- `runs/joint_dp/checkpoints/vanilla_dp.pt` — final checkpoint.
- `runs/joint_dp/loss_history.csv` and `loss_curve.png`.

The batch size may need reducing for smaller GPUs. Add `--no-cuda` for a CPU-only test. Optional experimental controls include `--normalize-obs`, `--delta-loss-weight` and `--overlap-loss-weight`; the latter two are **off by default**. Keep checkpoint config, normalisers and training split together.

## 6. Train a Disturbance Predictor

The basic Predictor is an independent two-hidden-layer MLP (256 hidden units) consuming `[2,151]` observations and `[8,7]` candidate actions. It predicts local `DR_mean`, `DR_max` and topple/fall events.

### 6.1 Direct supervised Predictor (expert windows)

The original predictor defaults to continuous DR regression. To train the **ordinal** variant, specify `--dr-objective ordinal` explicitly:

~~~bash
python scripts/train_predictor.py \
  --dataset demos/baseline_30 \
  --output-dir runs/predictor_ordinal \
  --dr-objective ordinal \
  --dr-thresholds 0.0001,0.001,0.005,0.02,0.05,0.1 \
  --obs-horizon 2 --action-horizon 8 \
  --steps 10000 --batch-size 128 --lr 0.0003 \
  --eval-freq 500 --device auto
~~~

It produces `summary.json`, `train_log.csv`, `checkpoints/best_val.pt` and `checkpoints/last.pt`. The ordinal DR outputs are **threshold exceedance probabilities / a bounded risk proxy**, not exact unbounded physical DR predictions.

### 6.2 Held-out offline Predictor evaluation

~~~bash
python scripts/eval_predictor.py \
  --dataset demos/baseline_30 \
  --checkpoint runs/predictor_ordinal/checkpoints/best_val.pt \
  --output-dir runs/predictor_ordinal_test \
  --device cpu
~~~

This evaluates the held-out test split and writes metrics to `summary.json` and `predictions.csv`. Offline prediction quality alone does **not** establish that guided closed-loop robot trajectories are safer.

## 7. Generate and learn from physical counterfactuals

The final CANDID Predictor can be trained with additional **physics-labelled action alternatives**, not just recorded expert windows. All candidates for a reference state start from the **same archived scene** after replaying the expert prefix. Each candidate is actually simulated; no expert outcome label is copied onto a perturbed action.

### 7.1 Collect counterfactuals on training scenes only

First ensure the source dataset has a valid `split.json` (Section 4.3). For a small functional test:

~~~bash
python scripts/generate_counterfactuals.py \
  --dataset demos/baseline_30 \
  --output-dir runs/cf_smoke \
  --states 2 --candidates 3 \
  --candidate-mode multiscale \
  --sigma-levels 0.02,0.06,0.12 \
  --seed 2026
~~~

A more useful collection aims for 64 physically verified reference states and seven candidate chunks per state (one expert and three positive/negative perturbation pairs):

~~~bash
python scripts/generate_counterfactuals.py \
  --dataset demos/baseline_30 \
  --output-dir runs/cf_64 \
  --states 64 --candidates 7 \
  --max-per-scene 8 --probe-strategy balanced \
  --candidate-mode multiscale \
  --sigma-levels 0.02,0.06,0.12 \
  --seed 2026
~~~

This may require many more source training scenes and successful expert replays than the smoke dataset provides. The collector verifies replay consistency and can skip invalid probes. For interrupted collections, rerun **identical collection settings** with `--resume` and check `runs/cf_64/summary.json` for `complete=true`. Do not relax replay tolerances merely to force a complete collection.

Each `samples/<state_id>.npz` contains observations, candidate action chunks and actual local consequences (DR Mean/Max, topple, fall).

See [Counterfactual collection details](docs/COUNTERFACTUAL_DATA_V1.md).

### 7.2 Train the ordinal Predictor with expert + CF supervision

~~~bash
python scripts/train_counterfactual_predictor.py \
  --dataset demos/baseline_30 \
  --counterfactual-dataset runs/cf_64 \
  --output-dir runs/predictor_cf \
  --device auto --steps 10000 \
  --batch-size 64 --cf-batch-size 8 --lr 0.0001 \
  --expert-weight 1.0 --cf-weight 0.3 --rank-weight 0.5 \
  --cf-val-ratio 0.2 --eval-freq 250 --seed 2026
~~~

This entrypoint trains an **ordinal Predictor** from scratch when `--init-checkpoint` is omitted. It combines supervised expert windows, supervised physically replayed CF chunks and same-state pairwise ranking; its CF scene holdout is also excluded from expert training. For fine-tuning a compatible ordinal checkpoint, add `--init-checkpoint PATH`, noting that earlier pretraining may have already seen scenes subsequently designated for CF validation.

The model saves `runs/predictor_cf/checkpoints/best_val.pt`, `last.pt`, `summary.json` and logs. Train-time ranking accuracy and validation loss are not substitutes for a new closed-loop evaluation.

See [Counterfactual Predictor training details](docs/COUNTERFACTUAL_PREDICTOR_TRAINING.md).

## 8. Run inference and compare policies

In this codebase, **inference** means executing a trained policy in ManiSkill's **closed-loop simulator**, collecting new observations and replanning. It does not send commands to a physical robot.

Use **the same frozen evaluation suite** for Vanilla DP, CANDID and any Best-of-N comparison. Never regenerate a suite between methods, and use a **fresh output directory** for each run.

### 8.1 Generate an independent frozen evaluation suite

~~~bash
python scripts/generate_eval_scenes.py \
  --env-id CANDIDDenseClutter-v1 \
  --env-module envs.candid_dense_clutter_env_random \
  --num-scenes 20 --seed 900001 \
  --max-episode-steps 300 \
  --output-dir eval/dense20
~~~

The suite contains saved simulator states and a fingerprint in `suite.json`. Its environment source code must remain available and compatible during later replay.

### 8.2 Vanilla DP (no guidance)

~~~bash
python scripts/eval_dp.py \
  --eval-suite eval/dense20 \
  --checkpoint runs/joint_dp/checkpoints/best_val.pt \
  --output-dir runs/eval_vanilla \
  --safe-dr-mean-max 0.2 \
  --exec-horizon 4 --max-steps 300 \
  --eval-seed 2026 --device auto --no-video
~~~

### 8.3 CANDID: SNR-Gaussian Capped Guidance

Use the following **explicit settings** for the final guided-sampling configuration. Do not rely on the lower default strength/topple weight in the general CLI:

~~~bash
python scripts/eval_dp_guidance.py \
  --eval-suite eval/dense20 \
  --dp-checkpoint runs/joint_dp/checkpoints/best_val.pt \
  --predictor-checkpoint runs/predictor_cf/checkpoints/best_val.pt \
  --output-dir runs/eval_candid \
  --safe-dr-mean-max 0.2 \
  --exec-horizon 4 --max-steps 300 --eval-seed 2026 \
  --device auto --no-video \
  --strength 0.02 --start-timestep 30 \
  --guidance-parity odd \
  --strength-schedule snr_gaussian_capped \
  --snr-center-timestep 15 --snr-sigma 1.0 \
  --dr-mean-weight 1.0 --dr-max-weight 0.25 \
  --topple-weight 0.5 --fall-weight 0 \
  --max-grad-norm 1.0 --max-step-norm 0.02
~~~

With 100 reverse DDPM steps, this guides **only timesteps 29, 27, ..., 1**, while the DDPM transition still runs at every timestep. The risk uses the Predictor's 8-step local consequences; the robot executes the **first four actions** of each sampled chunk and then replans. On a GPU, use `--device cuda`; CPU guided inference is possible but substantially slower.

You can pass `--diagnose` to record optional same-noise guided/unguided diagnostics.

### 8.4 Best-of-N / reranking baseline

This evaluates several **unguided** DP samples with the same Predictor, rather than modifying DDPM denoising:

~~~bash
python scripts/eval_dp_predictor.py \
  --eval-suite eval/dense20 \
  --dp-checkpoint runs/joint_dp/checkpoints/best_val.pt \
  --predictor-checkpoint runs/predictor_cf/checkpoints/best_val.pt \
  --output-dir runs/eval_bestof4 \
  --mode rerank --num-candidates 4 \
  --safe-dr-mean-max 0.2 \
  --exec-horizon 4 --max-steps 300 --eval-seed 2026 \
  --device auto --no-video \
  --dr-mean-weight 1.0 --dr-max-weight 0.25 \
  --topple-weight 0.5 --fall-weight 0 \
  --min-improvement 0 --max-joint-deviation 1000000
~~~

`--mode shadow` is also available for prediction-only diagnostics. The two reranking thresholds above intentionally disable additional restrictions for a **pure minimum-predicted-risk Best-of-4** comparison.

### 8.5 Compare matched runs

~~~bash
python scripts/compare_eval_runs.py \
  runs/eval_vanilla runs/eval_candid \
  --output runs/vanilla_vs_candid.json
~~~

Each evaluation writes `summary.json`, `episodes.jsonl`, `episodes.csv` and individual episode artifacts. Important metrics include `success_rate`, `safe_success_rate`, `mean_dr_mean`, `mean_dr_max`, `total_topple` and `total_fall`. The comparison script rejects unequal suite hashes, safety limits, execution horizons or paired scene/seed keys.

For **existing dataset test scenes** rather than newly generated random suites, use `--dataset PATH` in place of `--eval-suite PATH`. `eval_dp.py` also accepts `--scene-split` for dataset-based diagnostics (default: `test`). Do not report training/validation-scene replays as independent held-out results.

See [Frozen random-suite evaluation](docs/RANDOM_EVAL.md) and [Guidance settings](model/guidance/README.md).

## 9. Five-environment experiment workflow

The following is an **illustrative, reproducible multi-environment recipe using five modules available in this repository**. It is not a claim that the exact paper-scale dataset, random seeds or trained checkpoints are distributed here.

### 9.1 Collect one dataset per environment

The environment pairs below are examples of five available clutter types. Use separate directories and seeds; the four-route sampler requires `--route-intents`.

~~~bash
while IFS='|' read -r label env_id env_module seed; do
  python scripts/collect.py \
    --env-id "$env_id" --env-module "$env_module" \
    --scenes 120 --trajectories-per-scene 24 \
    --route-intents --no-collision-aware \
    --seed "$seed" --output-dir "demos/$label"
  python scripts/validate_dataset.py "demos/$label"
done <<'ENVIRONMENTS'
dense|CANDIDDenseClutter-v1|envs.candid_dense_clutter_env_random|1000
tight|CANDIDTightUpright-v1|envs.candid_tight_upright_env|11000
large|CANDIDLargeClutter-v1|envs.candid_large_clutter_env|21000
diverse|CANDIDTightUprightDiverse-v1|envs.candid_tight_upright_diverse_env|31000
mixed|CANDIDMixedConsequence-v1|envs.candid_mixed_consequence_env|41000
ENVIRONMENTS
~~~

This is computationally expensive. Start with **one scene per environment** and check that the chosen modules, physics assets and planner work on your hardware before collecting all 600 scenes.

### 9.2 Merge experts, freeze splits, train one joint DP

~~~bash
python scripts/merge_datasets.py \
  --inputs demos/dense demos/tight demos/large demos/diverse demos/mixed \
  --output-dir demos/joint5 --mode copy

python scripts/train_dp.py \
  --dataset demos/joint5 \
  --output-dir runs/joint5_dp \
  --total-iters 30000 --batch-size 1024 \
  --lr 0.0001 --eval-freq 500
~~~

This creates the merged dataset's frozen `split.json` and a single shared DP, rather than five independently trained policies.

### 9.3 Build the joint physical counterfactual collection

Each **original source** needs its own frozen scene split and physics-verified CF collection. The merged joint CF builder checks original source identities, saved snapshots and consistency with the **merged train split**.

~~~bash
# Ensure each source has its own frozen split.
python - <<'PY'
from candid.data.splits import load_or_create_scene_split
for name in ("dense", "tight", "large", "diverse", "mixed"):
    root = f"demos/{name}"
    s = load_or_create_scene_split(root, seed=0, val_ratio=.15, test_ratio=.15)
    print(name, len(s["train"]), "train scenes")
PY

# Collect from training-scene sources only; use a fresh output per source.
for label in dense tight large diverse mixed; do
  python scripts/generate_counterfactuals.py \
    --dataset "demos/$label" --output-dir "runs/cf_$label" \
    --states 100 --candidates 7 --max-per-scene 8 \
    --probe-strategy balanced --candidate-mode multiscale \
    --sigma-levels 0.02,0.06,0.12 --seed 2026
done

# Select 64 compatible reference states from each source (five x 64).
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

**Important:** 100 collected CF groups per source is an example, **not a guarantee** that 64 will survive the merged training split and replay checks. If the builder reports too few eligible states, collect additional *valid train-source probes*; never move validation/test states into training or re-label physics outcomes. CF replay cost includes the full expert prefix for **every** candidate.

### 9.4 Train the joint ordinal Predictor

A portable direct command that does **not** depend on any pre-existing training `summary.json`:

~~~bash
python scripts/train_counterfactual_predictor.py \
  --dataset demos/joint5 \
  --counterfactual-dataset runs/joint5_cf64 \
  --output-dir runs/joint5_predictor \
  --device cuda --steps 10000 --lr 0.0001 \
  --batch-size 64 --cf-batch-size 8 \
  --expert-weight 1.0 --cf-weight 0.3 --rank-weight 0.5 \
  --cf-val-ratio 0.2 --eval-freq 250 --seed 2026
~~~

The existing convenience wrapper `scripts/run_joint5_methods.py --stage train` is **not** a standalone fresh-clone entrypoint: it explicitly requires the earlier Scene-1 reference metadata file `runs/predictor_ordinal_joint_regularized_v2/summary.json`. Use the direct trainer above unless you possess that exact metadata.

### 9.5 Build one common evaluation suite

~~~bash
while IFS='|' read -r label env_id env_module seed; do
  python scripts/generate_eval_scenes.py \
    --env-id "$env_id" --env-module "$env_module" \
    --num-scenes 50 --seed "$seed" --max-episode-steps 300 \
    --output-dir "eval/$label"
done <<'ENVIRONMENTS'
dense|CANDIDDenseClutter-v1|envs.candid_dense_clutter_env_random|901000
tight|CANDIDTightUpright-v1|envs.candid_tight_upright_env|911000
large|CANDIDLargeClutter-v1|envs.candid_large_clutter_env|921000
diverse|CANDIDTightUprightDiverse-v1|envs.candid_tight_upright_diverse_env|931000
mixed|CANDIDMixedConsequence-v1|envs.candid_mixed_consequence_env|941000
ENVIRONMENTS

python scripts/merge_eval_suites.py \
  --suite dense=eval/dense \
  --suite tight=eval/tight \
  --suite large=eval/large \
  --suite diverse=eval/diverse \
  --suite mixed=eval/mixed \
  --output-dir eval/joint250
~~~

Then repeat the evaluation commands from Section 8 using:

- `--eval-suite eval/joint250`
- `--dp-checkpoint runs/joint5_dp/checkpoints/best_val.pt` (for CANDID)
- `--checkpoint runs/joint5_dp/checkpoints/best_val.pt` (for Vanilla)
- `--predictor-checkpoint runs/joint5_predictor/checkpoints/best_val.pt` (for CANDID / Best-of-N)

To summarise by environment:

~~~bash
python scripts/summarize_eval_groups.py \
  --eval-suite eval/joint250 \
  --results runs/eval_candid \
  --output runs/eval_candid/by_environment.json
~~~

The combined-suites script checks source suite integrity. **Never regenerate or silently replace suites between methods.**

## 10. Other baselines, tests, and visualisation

### Optional baseline methods

`baselines/` contains the state-based **Behavior Cloning (BC)**, **ACT**, **PPO** and **SAC** implementations. BC/ACT consume exported demonstration HDF5; PPO/SAC train online in the simulator. For commands covering demonstration export, each baseline's training, checkpoint loading and common closed-loop evaluation, follow [baselines/README.md](baselines/README.md).

Main entrypoints: `baselines/common/export_demos.py`, `baselines/bc/bc.py`, `baselines/act/train.py`, `baselines/ppo/ppo.py`, `baselines/sac/sac.py` and `baselines/common/eval_baseline.py`.

### Tests

~~~bash
# Fast components / basic integration contracts (run after installation).
python -m pytest \
  tests/test_contract.py \
  tests/test_predictor_ordinal.py \
  tests/test_predictor_counterfactual.py \
  tests/test_guidance.py \
  tests/test_guidance_snr.py \
  tests/test_merge_datasets.py \
  tests/test_multi_env_suite.py -q

# Full suite (may require additional simulator assets / system libraries).
python -m pytest tests/ -q
~~~

### Diagnostics and qualitative results

- `scripts/compare_eval_runs.py` — paired policy-outcome comparison on identical frozen scenes.
- `scripts/analyze_guidance_diagnostics.py` — optional gradient/guidance diagnostics when recorded.
- `scripts/report_counterfactual_quality.py` — inspect CF replay and label diversity.
- `scripts/audit_eval_suite.py` — verify frozen evaluation suite integrity.
- `scripts/render_trajectory_npz.py` / `scripts/replay_trajectory_rgb.py` — view or replay recorded physical trajectories.
- `scripts/render_paired_trajectory.py` — compare Vanilla and CANDID TCP traces and physical disturbance in a shared scene. See [visualisation instructions](docs/paired_trajectory_visualization.md).

## 11. Reproducibility and troubleshooting

**Keep a record of:** repository commit SHA, Conda/Python/PyTorch versions, original environment modules, scene seeds, train/validation/test split, source dataset IDs, checkpoint paths, execution horizon, safety threshold, random-suite fingerprint and full command-line flags.

| Symptom | What to check |
|---|---|
| `ModuleNotFoundError: candid` | Run from repository root and export `PYTHONPATH="$PWD:$PWD/src"`. |
| CUDA not found | Verify NVIDIA driver, PyTorch wheel and `torch.cuda.is_available()`. For CPU evaluation use `--device cpu`; DP training supports `--no-cuda`. |
| Vulkan / rendering failure | For evaluation use `--no-video`; for collection check SAPIEN/Vulkan installation and GPU visibility. Headless physics and video rendering have different requirements. |
| No valid Predictor windows or split errors | Collect enough distinct **scenes**, verify successful trajectories, and keep the frozen split consistent. |
| Replay / SceneSpec mismatch | Restore the original environment module and scene snapshot; do not bypass physics/label consistency checks. |
| Existing output directory rejected | Most training/evaluation/CF entrypoints intentionally avoid overwrites. Choose a new run directory; use `--resume` only for a compatible CF collection. |
| Counterfactual sample shortage | Inspect `summary.json` and `skipped_probes.jsonl`. Increase valid source-state coverage rather than substituting held-out labels. |
| CANDID is slower than Vanilla | Risk guidance computes Predictor gradients during sampling; check GPU use and guidance frequency (odd timesteps vs every step). |
| Evaluation results differ between policies | Confirm exact `eval_suite_sha256`, environment source, seed, execution horizon and safety limits before interpreting a difference. |

### Further documentation

- [Physical counterfactual collection](docs/COUNTERFACTUAL_DATA_V1.md)
- [Counterfactual + ordinal Predictor training](docs/COUNTERFACTUAL_PREDICTOR_TRAINING.md)
- [Ordinal Predictor V2](model/predictor/ORDINAL_V2.md)
- [Frozen random evaluation](docs/RANDOM_EVAL.md)
- [Guidance](model/guidance/README.md)
- [Paired trajectory visualisation](docs/paired_trajectory_visualization.md)
- [Optional baselines](baselines/README.md)

### Attribution

This project builds on [ManiSkill](https://github.com/haosulab/ManiSkill), [SAPIEN](https://sapien.ucsd.edu/), [MPlib](https://github.com/haosulab/MPlib) and [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/). Code copied from ManiSkill's baseline examples is covered by the corresponding attribution and [Apache-2.0 notice](baselines/LICENSE-APACHE-2.0). See the repository [LICENSE](LICENSE) for the project license.
