"""Source-module fallback is tested without starting ManiSkill simulation."""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candid.eval.replay_env import resolve_scene_environment, scene_spec_from_dict
from candid.sim.spec import SceneSpec, ObjectSpec, XYRegion


def example_scene():
    spec = SceneSpec(
        name="eval_test",
        target=ObjectSpec(
            name="target", kind="box", region=XYRegion((-.12, -.10), (-.02, .02)),
            fixed_xy=(-.11, 0.0), size=(.04, .04, .04),
        ),
        distractors=(
            ObjectSpec(
                name="obstacle", kind="box", region=XYRegion((-.18, -.15), (.05, .09)),
                fixed_xy=(-.16, .07), size=(.04, .04, .04),
            ),
        ),
        goal_region=XYRegion((.08, .12), (-.02, .02)),
        goal_fixed_xy=(.10, 0.0),
    )
    return {
        "scene_id": "team_01__scene_00000", "env_id": "MissingCANDID-v1",
        "env_module": "envs.this_module_should_not_exist_abc123",
        "scene_spec": spec.to_dict(), "distractor_ids": ["obstacle"],
    }


def test_archived_scene_spec_round_trip():
    data = example_scene()["scene_spec"]
    loaded = scene_spec_from_dict(data)
    assert loaded.to_dict() == data


def test_available_module_keeps_original_env():
    item = example_scene()
    item["env_module"] = "json"
    assert resolve_scene_environment(item) == ("MissingCANDID-v1", "original_env_module")


def test_missing_module_falls_back_to_scene_spec(monkeypatch):
    fake_environment = types.ModuleType("candid.sim.environment")
    fake_environment.CANDIDTabletopEnv = type("FakeEnv", (), {})
    registrations = {}
    fake_reg = types.ModuleType("mani_skill.utils.registration")
    fake_reg.REGISTERED_ENVS = registrations
    fake_reg.register_env = lambda name, max_episode_steps: (
        lambda cls: registrations.setdefault(name, cls) or cls
    )
    monkeypatch.setitem(sys.modules, "candid.sim.environment", fake_environment)
    monkeypatch.setitem(sys.modules, "mani_skill.utils.registration", fake_reg)
    item = example_scene()
    with pytest.warns(RuntimeWarning, match="archived SceneSpec"):
        new_id, origin = resolve_scene_environment(item)
    assert origin == "scene_spec_fallback"
    assert new_id.startswith("CANDIDSceneReplay-")
    assert new_id in registrations
    assert registrations[new_id].SCENE.to_dict() == item["scene_spec"]
    with pytest.warns(RuntimeWarning):
        new_id_again, _ = resolve_scene_environment(item)
    assert new_id_again == new_id
    assert len(registrations) == 1


def test_missing_module_without_scene_spec_fails_clearly():
    item = example_scene()
    del item["scene_spec"]
    with pytest.raises(ModuleNotFoundError, match="Restore the original env module"):
        resolve_scene_environment(item)


def test_mismatched_distractor_order_rejected(monkeypatch):
    fake_environment = types.ModuleType("candid.sim.environment")
    fake_environment.CANDIDTabletopEnv = type("FakeEnv", (), {})
    fake_reg = types.ModuleType("mani_skill.utils.registration")
    fake_reg.REGISTERED_ENVS = {}
    fake_reg.register_env = lambda *args, **kwargs: lambda cls: cls
    monkeypatch.setitem(sys.modules, "candid.sim.environment", fake_environment)
    monkeypatch.setitem(sys.modules, "mani_skill.utils.registration", fake_reg)
    item = example_scene()
    item["distractor_ids"] = ["wrong_actor"]
    with pytest.raises(ValueError, match="distractor names/order"):
        resolve_scene_environment(item)
