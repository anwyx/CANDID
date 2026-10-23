"""Unit tests for the settling checker; no ManiSkill installation needed."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from candid.sim.settling import SettlingConfig, peak_motion, settle_scene


def pose(x=0.0, yaw=0.0):
    return np.array([[x, 0, 0.02, np.cos(yaw/2), 0., 0., np.sin(yaw/2)]])


class FakeScene:
    def __init__(self, positions):
        self.positions = positions
        self.i = 0

    def step(self):
        self.i = min(self.i + 1, len(self.positions) - 1)

    def snapshot(self):
        return pose(self.positions[self.i])


class SettlingTests(unittest.TestCase):
    def test_stationary_scene_passes(self):
        scene = FakeScene([0.] * 100)
        report = settle_scene(scene.step, scene.snapshot, SettlingConfig(max_steps=100), ['cube'])
        self.assertTrue(report['idle_stable'])
        self.assertGreaterEqual(report['settle_steps'], 50)

    def test_tiny_solver_jitter_passes(self):
        # Alternating numerical vibration; may imply speed spikes, but pose
        # deviations of 0.04 mm should not invalidate a settled scene.
        scene = FakeScene([0.00004 * (-1)**i for i in range(120)])
        report = settle_scene(scene.step, scene.snapshot, SettlingConfig(max_steps=120), ['cube'])
        self.assertTrue(report['idle_stable'])

    def test_persistent_drift_is_rejected(self):
        scene = FakeScene([i*0.0003 for i in range(150)])
        with self.assertRaisesRegex(RuntimeError, 'still moving'):
            settle_scene(scene.step, scene.snapshot, SettlingConfig(max_steps=120), ['cube'])

    def test_falling_then_settled_passes(self):
        positions = [max(0., 0.01 - i*0.0005) for i in range(120)]
        scene = FakeScene(positions)
        report = settle_scene(scene.step, scene.snapshot, SettlingConfig(max_steps=120), ['cube'])
        self.assertTrue(report['idle_stable'])

    def test_quaternion_double_cover(self):
        q = pose()
        opposite = q.copy()
        opposite[0, 3:] *= -1
        d, r, _, _ = peak_motion(q, [opposite])
        self.assertAlmostEqual(d, 0.)
        self.assertAlmostEqual(r, 0.)

    def test_verification_rejects_late_motion(self):
        # Pass sliding window at t20; fail subsequent 30-step verify due to push.
        positions = [0.] * 35 + [0.02] * 200
        scene = FakeScene(positions)
        with self.assertRaisesRegex(RuntimeError, 'still moving'):
            settle_scene(scene.step, scene.snapshot,
                         SettlingConfig(max_steps=70), ['cube'])


if __name__ == '__main__':
    unittest.main()
