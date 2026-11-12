# Paired CANDID trajectory visualization

Standalone entry: scripts/render_paired_trajectory.py

Creates a three-panel PNG and an MP4 by projecting two recorded TCP paths
onto REAL ManiSkill RGB frames. The background RGB shows the CANDID rollout.
The Vanilla DP path is an overlay from a different episode executed under
the same frozen initial conditions.

DEPENDENCIES

    python -m pip install opencv-python pillow imageio imageio-ffmpeg

INPUTS

Two self-contained scene3_vanilla.zip / scene3_candid.zip replay bundles made
by scripts/replay_trajectory_rgb.py pack, containing trajectory.npz,
initial_state.npz, scene.json and result.json.

OPTION 1: Reuse an existing, unmodified CANDID RGB replay MP4 (fastest):

    python scripts/render_paired_trajectory.py \
      --vanilla-bundle runs/candid_showcase/bundles/scene3_vanilla.zip \
      --guided-bundle runs/candid_showcase/bundles/scene3_candid.zip \
      --background-video Downloads/scene3_candid.mp4 \
      --output-dir Downloads/scene3_comparison \
      --steps 36 48 72

The input background MP4 must contain exactly one video frame for every
recorded Guided state (rendered with --stride 1, no trimming or intro).
The script checks frame count but cannot independently verify a video's
provenance. Supply the correct CANDID MP4. If the episode is shorter than
Step 72, omit --steps to select approximately 25%, 50%, 75% instead.

OPTION 2: Auto-render the background from the two ZIPs:

    python scripts/render_paired_trajectory.py \
      --vanilla-bundle runs/candid_showcase/bundles/scene3_vanilla.zip \
      --guided-bundle runs/candid_showcase/bundles/scene3_candid.zip \
      --output-dir Downloads/scene3_comparison \
      --render-backend cpu

This internally calls scripts/replay_trajectory_rgb.py render using archived
SceneSpec and allowed generic SceneSpec fallback. It still requires a
working SAPIEN/ManiSkill Vulkan renderer. It does NOT run any trained model
or step the simulation. Original custom scene-building overrides cannot
necessarily be reproduced by generic SceneSpec fallback.

OPTION 3: Use two raw trajectory.npz files plus the full frozen suite:

    python scripts/render_paired_trajectory.py \
      --vanilla-trajectory eval/results250/vanilla_new/135_scene3__random_eval_920001_00035_rep00/trajectory.npz \
      --guided-trajectory runs/joint5_final/candid_snr_capped/135_scene3__random_eval_920001_00035_rep00/trajectory.npz \
      --eval-suite eval/combined250 \
      --output-dir Downloads/scene3_comparison

Each raw trajectory folder must have a sibling result.json. The suite has
the archived SceneSpec and initial state. No DP/Predictor checkpoint needed.

OUTPUTS

- paired_trajectory_triptych.png: shared legend, three rendered snapshots,
  time-progressive yellow/red Vanilla and cyan/blue CANDID TCP paths, and
  physical episode DR Mean / Topple / Fall / task success.
- paired_trajectory_overlay.mp4: video with TCP traces overlaid.
- paired_trajectory_summary.json: metadata, paired-scene checks and metrics.

Use --no-overlay-video to create the triptych PNG only.

VALIDATION AND SCIENTIFIC INTERPRETATION

The script rejects mismatched scene IDs, environment metadata, eval seeds,
archived initial-state hashes, and initial robot/object/goal poses.
These are independent closed-loop trajectories with the SAME INITIAL SCENE,
NOT same-observation/same-noise counterfactual action chunks. The RGB
background follows only the CANDID rollout. Vanilla is displayed as an
overlaid path. DR is the measured episode-level physical metric, NOT
spillage probability nor predicted local DR.

The projection uses archived human_camera_eye / human_camera_look_at and
the default vertical FOV 1.0 radian. Set --fovy to match customised cameras.
Verify alignment visually for any custom camera pose or edited RGB video.

OFFLINE TESTS

    python -m pytest -q tests/test_paired_trajectory.py

These tests use synthetic frames and trajectories; a real ManiSkill end-to-end
render must be checked in your configured 3D rendering environment.
