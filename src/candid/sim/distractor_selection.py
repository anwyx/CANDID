"""Select a reproducible subset of SceneSpec distractors before actors are built.

Selection happens once per environment instance. The selected scene is a new,
validated frozen SceneSpec; the class-level SCENE declaration is untouched.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np


def select_distractors(scene, count=None, seed=0):
    """Return a scene with exactly ``count`` selected distractors.

    ``None`` preserves the entire declared scene. Other counts select a seeded
    random subset, preserving the original relative ordering of object specs.
    This is a once-per-environment selection, not a per-episode actor rebuild.
    """
    available = len(scene.distractors)
    if count is None:
        return scene
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise ValueError("num_distractors must be an integer")
    if not 1 <= int(count) <= available:
        raise ValueError(
            f"num_distractors must be in [1, {available}] for scene {scene.name!r}"
        )
    if int(count) == available:
        return scene
    indices = sorted(np.random.default_rng(int(seed)).choice(
        available, size=int(count), replace=False
    ).tolist())
    return replace(scene, distractors=tuple(scene.distractors[i] for i in indices))
