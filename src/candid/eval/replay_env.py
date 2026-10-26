"""Resolve a CANDID environment for evaluation, optionally replaying saved SceneSpec.

A collection stores only its environment module name, not Python source code.
When that module is unavailable, a standard CANDIDTabletopEnv can instead be
reconstructed using the complete SceneSpec persisted in scenes.jsonl.

This fallback is ONLY equivalent for thin env classes whose behavior is entirely
provided by CANDIDTabletopEnv + SceneSpec. Custom overridden physics/rewards
cannot be reproduced without their source code.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import warnings
from dataclasses import fields


def _xy_region(data):
    from candid.sim.spec import XYRegion
    if not isinstance(data, dict):
        raise ValueError("Invalid saved XYRegion")
    return XYRegion(x=tuple(data["x"]), y=tuple(data["y"]))


def _object_spec(data):
    from candid.sim.spec import ObjectSpec
    if not isinstance(data, dict):
        raise ValueError("Invalid saved ObjectSpec")
    parsed = dict(data)
    parsed["region"] = _xy_region(data["region"])
    for key in ("fixed_xy", "size", "color"):
        if parsed.get(key) is not None:
            parsed[key] = tuple(parsed[key])
    return ObjectSpec(**parsed)


def scene_spec_from_dict(data):
    """Convert an archived SceneSpec.to_dict() back to a validated SceneSpec."""
    from candid.sim.spec import SceneSpec
    if not isinstance(data, dict):
        raise ValueError("Scene metadata does not contain a SceneSpec dictionary")
    parsed = dict(data)
    parsed["target"] = _object_spec(data["target"])
    parsed["distractors"] = tuple(_object_spec(item) for item in data["distractors"])
    parsed["goal_region"] = _xy_region(data["goal_region"])
    for f in fields(SceneSpec):
        if f.name in ("target", "distractors", "goal_region"):
            continue
        if f.name in parsed and isinstance(parsed[f.name], list):
            parsed[f.name] = tuple(parsed[f.name])
    return SceneSpec(**parsed)


def resolve_scene_environment(scene: dict) -> tuple[str, str]:
    """Return (gym_env_id, source_type) for evaluation.

    Normal case imports the original module. Fallback reconstructs a *standard*
    CANDIDTabletopEnv from scene_spec if, and only if, the named module itself
    (or its parent package) does not exist. Missing dependencies within a
    present module are not swallowed.
    """
    module = scene.get("env_module")
    env_id = scene.get("env_id")
    if not isinstance(module, str) or not module:
        raise ValueError("Scene record is missing env_module")
    if not isinstance(env_id, str) or not env_id:
        raise ValueError("Scene record is missing env_id")

    try:
        importlib.import_module(module)
    except ModuleNotFoundError as error:
        missing = error.name
        if not missing or not (module == missing or module.startswith(missing + ".")):
            raise  # Original module exists but one of *its dependencies* is broken.
        archived = scene.get("scene_spec")
        if not archived:
            raise ModuleNotFoundError(
                f"Cannot import {module!r}, and scene {scene.get('scene_id')!r} "
                "has no saved scene_spec. Restore the original env module "
                "to PYTHONPATH or recollect with SceneSpec metadata."
            ) from error

        # Delay simulator imports until the fallback is actually needed.
        from candid.sim.environment import CANDIDTabletopEnv
        from mani_skill.utils.registration import register_env
        spec = scene_spec_from_dict(archived)
        expected_names = scene.get("distractor_ids")
        if expected_names is not None:
            actual_names = [item.name for item in spec.distractors]
            if list(expected_names) != actual_names:
                raise ValueError(
                    f"SceneSpec distractor names/order disagree with snapshot: "
                    f"{actual_names} vs {expected_names}"
                )
        payload = json.dumps(archived, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        replay_id = f"CANDIDSceneReplay-{digest}-v0"
        from mani_skill.utils.registration import REGISTERED_ENVS
        if replay_id not in REGISTERED_ENVS:
            # A new class per saved spec; never override the original task ID.
            cls = type(f"CANDIDSceneReplay_{digest}", (CANDIDTabletopEnv,), {"SCENE": spec})
            register_env(replay_id, max_episode_steps=500)(cls)
        warnings.warn(
            f"Original env module {module!r} is unavailable; replaying "
            f"{scene.get('scene_id')!r} with archived SceneSpec and generic "
            "CANDIDTabletopEnv. Original custom env code is required for "
            "exact replay if it overrides physics/task behavior.",
            RuntimeWarning,
            stacklevel=2,
        )
        return replay_id, "scene_spec_fallback"

    return env_id, "original_env_module"
