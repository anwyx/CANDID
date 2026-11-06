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
        # SAPIEN cylinders are aligned with local X by default. Rotate both
        # collision and visual geometry into local Z before placing the actor;
        # the environment later applies only a world-Z yaw to its pose.
        upright = sapien.Pose(q=[np.sqrt(0.5), 0, -np.sqrt(0.5), 0])
        builder = scene.create_actor_builder()
        builder.add_cylinder_collision(
            pose=upright, radius=obj_spec.size[0]/2,
            half_length=obj_spec.size[2]/2,
        )
        builder.add_cylinder_visual(
            pose=upright, radius=obj_spec.size[0]/2,
            half_length=obj_spec.size[2]/2,
            material=sapien.render.RenderMaterial(base_color=list(obj_spec.color)),
        )
        builder.set_initial_pose(pose)
        actor = builder.build(name=name)
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
    if spec.kind == "cylinder":
        # The collision-mesh helper reports the native cylinder mesh on a
        # different axis from SAPIEN's shape. Use the known upright dimensions
        # for spawn height, footprint checks, and dataset metadata.
        dimensions = tuple(float(v) for v in spec.size)
        radius = float(np.hypot(dimensions[0]/2, dimensions[1]/2))
        return BuiltObject(spec, actor, dimensions, dimensions[2]/2,
                           radius, max(dimensions))
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
