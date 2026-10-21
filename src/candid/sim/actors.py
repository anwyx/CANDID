"""Factory for creating ManiSkill actors from ObjectSpec.

Raw collision dimensions are derived once per constructed actor and reused by
sampler, observations and disturbance normalisation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import sapien
from mani_skill.utils.building import actors


@dataclass
class BuiltObject:
    spec: object
    actor: object
    dimensions: tuple[float, float, float]
    rest_z: float
    footprint_radius: float
    characteristic_size: float


def build_actor(scene, obj_spec):
    name, kind = obj_spec.name, obj_spec.kind
    pose = sapien.Pose(p=[0, 0, 0])
    if kind == "box":
        actor = actors.build_box(scene, half_sizes=[v/2 for v in obj_spec.size],
                                 color=list(obj_spec.color), name=name, initial_pose=pose)
    elif kind == "cylinder":
        actor = actors.build_cylinder(scene, radius=obj_spec.size[0]/2,
                                      half_length=obj_spec.size[2]/2,
                                      color=list(obj_spec.color), name=name, initial_pose=pose)
    elif kind == "sphere":
        actor = actors.build_sphere(scene, radius=obj_spec.size[0]/2,
                                    color=list(obj_spec.color), name=name, initial_pose=pose)
    elif kind == "ycb":
        builder = actors.get_actor_builder(scene, id=f"ycb:{obj_spec.asset_id}")
        builder.initial_pose = pose
        actor = builder.build(name=name)
    else:
        raise ValueError(f"Unsupported actor kind: {kind}")
    return actor


def describe_actor(spec, actor):
    mesh = actor.get_first_collision_mesh(to_world_frame=False)
    if mesh is None:
        raise RuntimeError(f"Actor {spec.name} has no collision mesh")
    bounds = np.asarray(mesh.bounding_box.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise RuntimeError(f"Invalid collision bounds for {spec.name}")
    dimensions = tuple(float(x) for x in bounds[1] - bounds[0])
    corners = [(x, y) for x in (bounds[0,0], bounds[1,0])
                      for y in (bounds[0,1], bounds[1,1])]
    radius = max(float(np.hypot(x,y)) for x,y in corners)
    return BuiltObject(spec, actor, dimensions, -float(bounds[0,2]),
                       radius, max(dimensions))
