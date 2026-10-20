"""Run and record the deterministic CANDID baseline Pick-and-Place.

Typical runs (from the CANDID project root):

    # Quick test: one fixed baseline scene, no large RGB-D dataset
    python scripts/run_baseline.py --episodes 1 --obs-mode none

    # Collect RGB-D + proprioception + executed joint actions
    python scripts/run_baseline.py --episodes 5 --obs-mode rgbd

    # Generate a simple different TRANSFER route through a waypoint
    python scripts/run_baseline.py --episodes 5 --lateral-offset 0.04

    # Visualize and save an MP4 (requires working renderer / display)
    python scripts/run_baseline.py --episodes 1 --vis --save-video

Important conventions:
* The scene is deterministic; more episodes do not mean more unique scenes.
* MPlib uses control_mode='pd_joint_pos' for this baseline.
* Both the real robot's joint states and RGB-D camera can produce the
  observations stored for future DP imitation learning.
* The planner itself may use simulator cube pose as an expert teacher.
* Simulator distractor poses are read ONLY by the disturbance label logger,
  NOT used as policy/predictor inputs.  They are NOT recorded in the HDF5
  environment-state field (record_env_state=False).
* Both successful and failed executed attempts are saved by default;
  --save-success-only filters the HDF5 demonstrations for DP training.
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

# Running 'python scripts/run_baseline.py' puts scripts/ on sys.path, not the
# project root.  Insert the root so both local modules can be imported.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import envs.candid_baseline_env  # noqa: F401; registers CANDIDBaseline-v0 in gym
from planners.execute_baseline import BaselinePlanConfig, PlanResult, solve

from mani_skill.utils.wrappers.record import RecordEpisode


class DisturbanceMonitor(gym.Wrapper):
    """Track simulator ground-truth distractor motion for LABELS ONLY.

    The monitor sees each executed physics/control step, without modifying
    actions or observations.  Later this can be replaced by a standalone
    chunk-level logger for the Disturbance Predictor.

    The reported peak motion is the maximum offset relative to the starting
    pose, while residual motion is the offset after the final action.
    Rotation is quaternion angular difference in radians.  'Topple' is an
    approximate geometric flag (local z-axis tilted > 60 degrees).
    'Fall' here only means the actor centre drops below table height; it
    is NOT a complete edge-of-table fall detector.
    """

    def __init__(self, env):
        super().__init__(env)
        self.initial_pose = None
        self.peak_translation = None
        self.peak_rotation = None
        self.ever_toppled = None
        self.ever_fell = None
        self.steps = 0
        self.pose_trace = []  # [initial pose, state after action 0, ...]

    def _poses(self) -> np.ndarray:
        """Get [N,7] ground truth ONLY from the environment label API."""
        poses = self.env.unwrapped.get_distractor_poses_for_logging()
        return poses.detach().cpu().numpy()[0].astype(np.float64)

    def reset(self, **kwargs):
        """Capture initial positions after ManiSkill resets the environment."""
        obs, info = self.env.reset(**kwargs)
        self.initial_pose = self._poses().copy()
        n = self.initial_pose.shape[0]
        self.peak_translation = np.zeros(n, dtype=np.float64)
        self.peak_rotation = np.zeros(n, dtype=np.float64)
        self.ever_toppled = np.zeros(n, dtype=bool)
        self.ever_fell = np.zeros(n, dtype=bool)
        self.steps = 0
        self.pose_trace = [self.initial_pose.copy()]
        return obs, info

    @staticmethod
    def _rotation_angle(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        """Smallest SO(3) angular difference for (w,x,y,z) quaternions."""
        q0 = q0 / np.maximum(np.linalg.norm(q0, axis=-1, keepdims=True), 1e-9)
        q1 = q1 / np.maximum(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-9)
        cos_half = np.abs(np.sum(q0 * q1, axis=-1))
        return 2.0 * np.arccos(np.clip(cos_half, 0.0, 1.0))

    def _measure(self):
        """Accumulate per-distractor peak translation, rotation and flags."""
        if self.initial_pose is None:
            return
        poses = self._poses()
        self.pose_trace.append(poses.copy())
        delta_p = poses[:, :3] - self.initial_pose[:, :3]
        displacement = np.linalg.norm(delta_p, axis=-1)
        angle = self._rotation_angle(self.initial_pose[:, 3:], poses[:, 3:])
        self.peak_translation = np.maximum(self.peak_translation, displacement)
        self.peak_rotation = np.maximum(self.peak_rotation, angle)

        # For an object initially standing upright, its local z direction
        # remains close to world +Z.  With (w,x,y,z) quaternions, the z-z
        # entry of its rotation matrix is 1 - 2(x^2 + y^2).
        quat = poses[:, 3:]
        quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-9)
        upright = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        self.ever_toppled |= upright < math.cos(math.radians(60.0))
        self.ever_fell |= poses[:, 2] < -0.03

    def step(self, action):
        """Execute unchanged action, then read resulting physical state."""
        result = self.env.step(action)
        self.steps += 1
        self._measure()
        return result

    def save_pose_trace(self, path: Path):
        """Save simulator GT poses outside the DP observation trajectory.

        Trace shape: [T+1, N_distractors, 7]; the first frame is after
        reset and each later frame follows one control action.  This lets
        the team compute *chunk-level* disturbance labels later.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            distractor_poses=np.asarray(self.pose_trace, dtype=np.float32),
            dt_steps=np.int32(self.steps),
        )

    def summary(self) -> dict[str, Any]:
        """Return small JSON-compatible episode-level labels and metadata."""
        if self.initial_pose is None:
            return {"steps": self.steps, "has_data": False}
        ending = self._poses()
        residual_translation = np.linalg.norm(
            ending[:, :3] - self.initial_pose[:, :3], axis=-1
        )
        residual_rotation = self._rotation_angle(
            self.initial_pose[:, 3:], ending[:, 3:]
        )
        # Keep raw per-object components.  Do not assign arbitrary weights
        # yet: D_max / D_mean will be computed after the team agrees on
        # translation, rotation, topple and fall weight conventions.
        return {
            "steps": self.steps,
            "has_data": True,
            "peak_translation_m": self.peak_translation.tolist(),
            "residual_translation_m": residual_translation.tolist(),
            "peak_rotation_rad": self.peak_rotation.tolist(),
            "residual_rotation_rad": residual_rotation.tolist(),
            "ever_toppled": self.ever_toppled.tolist(),
            "ever_fell_below_table": self.ever_fell.tolist(),
            "max_peak_translation_m": float(np.max(self.peak_translation)),
            "max_peak_rotation_rad": float(np.max(self.peak_rotation)),
        }


def parse_args():
    """Provide reproducible CLI defaults shared across three teammates."""
    parser = argparse.ArgumentParser(
        description="Execute ManiSkill3 xArm6 CANDID baseline with MPlib"
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of attempts")
    parser.add_argument("--seed", type=int, default=0, help="First reset seed")
    parser.add_argument(
        "--sim-backend", choices=("cpu", "gpu", "auto"), default="cpu",
        help="Use CPU for the simple single-scene MPlib baseline",
    )
    parser.add_argument(
        "--obs-mode", default="rgbd", choices=("rgbd", "pointcloud", "none"),
        help="Store deployable camera/proprio observations; 'none' is for fast debug only",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=500,
        help="Episode time limit (planning + release/settle need more than 250 steps sometimes)",
    )
    parser.add_argument(
        "--lateral-offset", type=float, default=0.0,
        help="Metre offset for an optional transfer midpoint; 0 = nominal route",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("demos/candid_baseline"),
        help="Folder for .h5/.json trajectory and episode_metrics.jsonl",
    )
    parser.add_argument(
        "--traj-name", default=None, help="Optional explicit trajectory file basename",
    )
    parser.add_argument(
        "--no-record", action="store_true", help="Run without HDF5 recording",
    )
    parser.add_argument(
        "--save-success-only", action="store_true",
        help="Discard failed attempts from HDF5, but retain failure metadata in JSONL",
    )
    parser.add_argument("--save-video", action="store_true", help="Record debugging MP4s")
    parser.add_argument("--vis", action="store_true", help="Show interactive viewer")
    parser.add_argument("--verbose", action="store_true", help="Print motion-planning stages")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    if args.max_episode_steps < 1:
        parser.error("--max-episode-steps must be >= 1")
    if abs(args.lateral_offset) > 0.15:
        parser.error("--lateral-offset must stay within [-0.15, 0.15] m")
    return args


def build_env(args):
    """Create the same exact environment/control modes for everyone.

    ``RecordEpisode`` records visual observations plus executed actions.
    The **truth-based monitor** is a separate wrapper below the recorder;
    it does not inject its pose measurements into policy observations.
    """
    env = gym.make(
        "CANDIDBaseline-v0",
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",  # Required by official MPlib XArm6 solver
        render_mode="rgb_array",
        robot_uids="xarm6_robotiq",
        reward_mode="dense",
        sim_backend=args.sim_backend,
        max_episode_steps=args.max_episode_steps,
    )
    monitor = DisturbanceMonitor(env)
    if args.no_record:
        return monitor, monitor, None

    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    name = args.traj_name or datetime.now().strftime("baseline_%Y%m%d_%H%M%S_%f")
    recorder = RecordEpisode(
        monitor,
        output_dir=str(folder),
        trajectory_name=name,
        save_trajectory=True,
        save_video=args.save_video,
        save_on_reset=False,
        clean_on_close=True,
        record_env_state=False,  # Avoid exposing simulator GT state as policy data
        source_type="motionplanning",
        source_desc="CANDID fixed-scene xArm6 MPlib baseline (task-only)",
        video_fps=30,
    )
    return recorder, monitor, recorder


def main():
    """Run all attempts and save success/failure + raw disturbance metrics."""
    args = parse_args()
    env, monitor, recorder = build_env(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "episode_metrics.jsonl"
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")

    successes = 0
    saved = 0
    plan_config = BaselinePlanConfig(lateral_offset=args.lateral_offset)

    try:
        with metadata_path.open("a", encoding="utf-8") as file:
            for k in range(args.episodes):
                seed = args.seed + k
                print(f"\n--- CANDID baseline, attempt {k + 1}/{args.episodes}, seed={seed} ---")
                try:
                    result = solve(
                        env, seed=seed, config=plan_config,
                        vis=args.vis, verbose=args.verbose,
                    )
                except Exception as exc:
                    # Do not conceal the failure.  Save the reason and keep
                    # running other attempts if the simulator remains usable.
                    print(f"Planner exception: {type(exc).__name__}: {exc}")
                    result = PlanResult(
                        success=False, stage="exception", steps=monitor.steps,
                        message=f"{type(exc).__name__}: {exc}",
                    )

                labels = monitor.summary()
                # Save dense simulator truth in a SEPARATE label folder.  The
                # recorded RGB-D/proprio HDF5 never receives these poses.
                trace_path = args.output_dir / "label_traces" / f"{run_name}_attempt_{k:04d}.npz"
                monitor.save_pose_trace(trace_path)
                if result.success:
                    successes += 1

                # With RecordEpisode the solver's executed control actions,
                # not just the requested waypoints, get written to HDF5.
                should_save = not args.save_success_only or result.success
                is_nonempty = monitor.steps > 0
                if recorder is not None:
                    recorder.flush_trajectory(save=should_save)
                    if args.save_video:
                        recorder.flush_video(save=should_save)
                    if should_save and is_nonempty:
                        saved += 1

                row = {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "seed": seed,
                    "environment": "CANDIDBaseline-v0",
                    "control_mode": "pd_joint_pos",
                    "obs_mode": args.obs_mode,
                    "lateral_offset_m": args.lateral_offset,
                    "task_success": result.success,
                    "last_stage": result.stage,
                    "planner_steps": result.steps,
                    "grasped_after_lift": result.grasped_after_lift,
                    "message": result.message,
                    "saved_to_hdf5": bool(recorder and should_save and is_nonempty),
                    "label_trace": str(trace_path),
                    "disturbance": labels,
                }
                file.write(json.dumps(row) + "\n")
                file.flush()
                print(
                    f"  success={result.success}  stage={result.stage}  "
                    f"steps={result.steps}  "
                    f"peak_clutter_translation={labels.get('max_peak_translation_m', float('nan')):.4f} m"
                )
    finally:
        env.close()

    print("\n=== CANDID baseline summary ===")
    print(f"Attempts: {args.episodes}")
    print(f"Task successes: {successes} ({successes / args.episodes:.1%})")
    print(f"Saved trajectories: {saved} (HDF5 recording {'disabled' if args.no_record else 'enabled'})")
    print(f"Episode metrics: {metadata_path.resolve()}")
    print(
        "Note: fixed baseline scene + fixed route; repetitions are NOT independent "
        "randomized environments or diverse policy demonstrations."
    )


if __name__ == "__main__":
    main()
