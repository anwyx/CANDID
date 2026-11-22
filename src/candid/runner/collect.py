"""CLI orchestration only: environments declare scenes, planners execute actions,
PhysicsTrace captures outcomes, DatasetWriter persists data and manifests.
"""
from __future__ import annotations
import argparse
import importlib
import inspect
import json
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import numpy as np
from candid.data.monitor import PhysicsTrace, as_numpy
from candid.data.storage import DatasetWriter, freeze_state, json_safe, load_state_npz


def symbol(path):
    if ":" not in path:
        raise ValueError("Expected dotted.module:Name")
    mod,name=path.split(":",1)
    return getattr(importlib.import_module(mod),name)


def parse_json(text, expected):
    result=json.loads(text)
    if not isinstance(result,expected):
        raise ValueError(f"Expected JSON {expected.__name__}")
    return result


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description="CANDID multi-scene, multi-trajectory collector")
    parser.add_argument("--env-id", default="CANDIDBaseline-v1")
    parser.add_argument("--env-module", default="envs.candid_baseline_env")
    parser.add_argument("--env-kwargs", default="{}")
    parser.add_argument("--fixed-scene", action="store_true")
    parser.add_argument("--planner", default="planners.execute_baseline:solve")
    parser.add_argument("--planner-config-class", default="planners.execute_baseline:BaselinePlanConfig")
    parser.add_argument("--plan-kwargs", default="{}")
    parser.add_argument("--plan-variants",default="[]",help='e.g. \'[{"lateral_offset":0},{"lateral_offset":0.04}]\'')
    parser.add_argument("--lateral-offset",type=float,default=None)
    parser.add_argument("--scenes",type=int,default=None)
    parser.add_argument("--episodes",type=int,default=None,help="Legacy alias of --scenes, one trajectory per scene by default")
    parser.add_argument("--trajectories-per-scene",type=int,default=None)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--robot-uid",default="xarm6_robotiq")
    parser.add_argument("--control-mode",default="pd_joint_pos")
    parser.add_argument("--obs-mode",default="state_dict")
    parser.add_argument("--sim-backend",choices=("cpu","gpu","auto"),default="cpu")
    parser.add_argument("--max-episode-steps",type=int,default=500)
    parser.add_argument("--output-dir",type=Path,default=Path("demos/candid_v1"))
    parser.add_argument("--scene-only",action="store_true",
                        help="Save settled scene snapshots without running a planner")
    parser.add_argument("--scene-source",type=Path,default=None,
                        help="Replay settled scenes from a prior output directory")
    parser.add_argument("--save-video",action="store_true")
    parser.add_argument("--save-success-only",action="store_true")
    parser.add_argument("--no-record",action="store_true")
    parser.add_argument("--vis",action="store_true")
    parser.add_argument("--verbose",action="store_true")
    parser.add_argument("--translation-weight",type=float,default=0.7)
    parser.add_argument("--rotation-weight",type=float,default=0.3)
    args=parser.parse_args(argv)
    if args.scenes is not None and args.episodes is not None:
        parser.error("Use --scenes or --episodes, not both")
    args.scenes = (args.scenes if args.scenes is not None else
                   args.episodes if args.episodes is not None else
                   None if args.scene_source is not None else 1)
    try:
        args.env_kwargs=parse_json(args.env_kwargs,dict)
        args.plan_kwargs=parse_json(args.plan_kwargs,dict)
        args.plan_variants=parse_json(args.plan_variants,list)
    except (ValueError,json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if not all(isinstance(x,dict) for x in args.plan_variants):
        parser.error("--plan-variants expects a JSON list of objects")
    if args.lateral_offset is not None:
        if "lateral_offset" in args.plan_kwargs:
            parser.error("lateral_offset specified twice")
        args.plan_kwargs["lateral_offset"]=args.lateral_offset
    if args.fixed_scene:
        if "randomize_layout" in args.env_kwargs:
            parser.error("Conflicting --fixed-scene and env-kwargs.randomize_layout")
        args.env_kwargs["randomize_layout"]=False
    if args.trajectories_per_scene is None:
        args.trajectories_per_scene=max(1,len(args.plan_variants))
    if (args.scenes is not None and args.scenes<1) or args.trajectories_per_scene<1:
        parser.error("Scene and trajectory counts must be positive")
    if args.scene_only and args.scene_source is not None:
        parser.error("Use --scene-only to generate, or --scene-source to replay")
    if args.scene_only and (args.save_video or args.plan_variants):
        parser.error("--scene-only does not execute plans or record videos")
    if args.scene_source is not None and args.fixed_scene:
        parser.error("A saved scene already has a fixed layout; omit --fixed-scene")
    if not args.no_record and args.obs_mode != "state_dict":
        parser.error("For a model-ready v1 dataset, use --obs-mode state_dict. "
                     "Other modes require --no-record (debug/video only).")
    if args.max_episode_steps<1:
        parser.error("--max-episode-steps must be positive")
    if args.trajectories_per_scene>1 and len(args.plan_variants)!=args.trajectories_per_scene:
        parser.error("Multiple trajectories per scene need the same number of --plan-variants")
    if any(x<0 for x in (args.translation_weight,args.rotation_weight)):
        parser.error("Disturbance weights must be nonnegative")
    return args


def git_sha():
    try:
        root=Path(__file__).resolve().parents[3]
        return subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,
                 stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        return None


def _status(value, fallback_steps):
    field=lambda k,d=None: value.get(k,d) if isinstance(value,dict) else getattr(value,k,d)
    if field("success") is None:
        raise RuntimeError("Planner must return a result with success field")
    return {"task_success":bool(field("success")),
            "stage":str(field("stage","completed")),
            "planner_steps":int(field("steps",fallback_steps)),
            "message":str(field("message","")),
            "route_waypoints_xyz":field("route_waypoints_xyz",())}


def _scene_start(base,trace):
    return {
        "distractor_poses":trace.arrays()["distractor_poses"][0].copy(),
        "target_pose":as_numpy(base.cube.pose.raw_pose)[0].copy(),
        "goal_pos":as_numpy(base.goal_site.pose.p)[0].copy(),
        "robot_qpos":as_numpy(base.agent.robot.get_qpos())[0].copy(),
    }


def _verify_start(first,second,atol=1e-5):
    for k in first:
        if not np.allclose(first[k],second[k],atol=atol,rtol=0):
            raise RuntimeError(f"Scene restore mismatch in {k}; refusing to mix trajectories")


def _read_scene_source(root):
    path=root/"scenes.jsonl"
    with path.open(encoding="utf-8") as f:
        rows=[json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"No scenes in {path}")
    ids=[row["scene_id"] for row in rows]
    if len(set(ids))!=len(ids):
        raise ValueError(f"Duplicate scene IDs in {path}")
    return rows


def _verify_saved_start(row,actual,atol=1e-5):
    expected={
        "distractor_poses":row["stable_initial_distractor_poses"],
        "target_pose":row["stable_initial_target_pose"],
        "robot_qpos":row["stable_initial_robot_qpos"],
    }
    if row.get("layout") is not None:
        expected["goal_pos"]=[*row["layout"]["goal_xy"],0.001]
    _verify_start(expected,actual,atol=atol)


def main(argv=None):
    args=parse_args(argv)
    source_root=args.scene_source.resolve() if args.scene_source is not None else None
    source_rows=_read_scene_source(source_root) if source_root is not None else None
    if source_rows is not None:
        if args.output_dir.resolve()==source_root:
            raise ValueError("--output-dir must differ from --scene-source")
        if args.scenes is None:
            args.scenes=len(source_rows)
        elif args.scenes>len(source_rows):
            raise ValueError(f"Requested {args.scenes} scenes, source has {len(source_rows)}")
    # Installed (src layout) or source tree (scripts shim inserts src on sys.path).
    import gymnasium as gym
    from mani_skill.utils.wrappers.record import RecordEpisode
    importlib.import_module(args.env_module)
    solver=None
    config_cls=None
    if not args.scene_only:
        solver=symbol(args.planner)
        sig=inspect.signature(solver)
        if "reset_env" not in sig.parameters:
            raise TypeError("Planner must accept reset_env=False. Runner owns Scene reset/replay. "
                            "Use supplied planners/execute_baseline.py as a template.")
        config_cls=None if args.planner_config_class.lower()=="none" else symbol(args.planner_config_class)
    env=gym.make(args.env_id,obs_mode=args.obs_mode,
                 control_mode=args.control_mode, render_mode="rgb_array",
                 robot_uids=args.robot_uid,reward_mode="dense",
                 sim_backend=args.sim_backend,max_episode_steps=args.max_episode_steps,
                 **args.env_kwargs)
    if getattr(env.unwrapped,"num_envs",1)!=1:
        env.close()
        raise ValueError("CANDID v1 collector currently requires num_envs=1")
    if source_rows is not None:
        spec=env.unwrapped.SCENE
        expected_spec=json_safe(spec.to_dict())
        for row in source_rows[:args.scenes]:
            for field,current in (("env_id",args.env_id),("env_module",args.env_module),
                                  ("env_kwargs",args.env_kwargs),("control_mode",args.control_mode),
                                  ("obs_mode",args.obs_mode),("scene_spec",expected_spec)):
                if row.get(field)!=current:
                    env.close()
                    raise ValueError(f"Saved scene {row['scene_id']} has different {field}; "
                                     "use the original environment and configuration")
            if not np.isclose(row["control_dt_s"],float(env.unwrapped.control_timestep)):
                env.close()
                raise ValueError(f"Saved scene {row['scene_id']} has different control timestep")
            if row.get("sim_backend",args.sim_backend)!=args.sim_backend:
                env.close()
                raise ValueError(f"Saved scene {row['scene_id']} has different sim backend")
    trace=PhysicsTrace(env,w_translation=args.translation_weight,
                       w_rotation=args.rotation_weight)
    dataset=DatasetWriter(args.output_dir)
    if args.scene_only:
        dataset.trajectory_manifest.touch(exist_ok=True)
    recorder=None
    run_id=datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid4().hex[:8]
    if not args.scene_only and (not args.no_record or args.save_video):
        recorder=RecordEpisode(trace,output_dir=str(args.output_dir),
            trajectory_name=f"trajectories_{run_id}",
            save_trajectory=not args.no_record,save_video=args.save_video,
            save_on_reset=False,clean_on_close=True,record_env_state=False,
            source_type="motionplanning",source_desc="CANDID shared collector")
    runner_env=recorder or trace
    sha=git_sha()
    successes=0
    attempted=0
    try:
        for scene_index in range(args.scenes):
            source_row=source_rows[scene_index] if source_rows is not None else None
            seed=int(source_row["seed"]) if source_row is not None else args.seed+scene_index
            scene_id=f"{run_id}_scene_{scene_index:05d}"
            if source_row is None:
                # ONE randomisation and physics settling, before saving/replaying.
                runner_env.reset(seed=seed)
                snapshot=freeze_state(runner_env.unwrapped.get_state_dict())
            else:
                rel=Path(source_row["initial_state"])
                state_path=(source_root/rel).resolve()
                if not state_path.is_relative_to(source_root):
                    raise ValueError(f"Scene snapshot is outside source directory: {rel}")
                snapshot=load_state_npz(state_path,device=env.unwrapped.device)
                runner_env.reset(seed=seed,options={"reset_to_env_states":{"env_states":snapshot}})
                if recorder is not None:
                    recorder.last_reset_kwargs={"seed":seed,"options":{"restored_scene_id":source_row["scene_id"]}}
            stable_start=_scene_start(runner_env.unwrapped,trace)
            if source_row is not None:
                _verify_saved_start(source_row,stable_start)
            base=runner_env.unwrapped
            spec=getattr(base,"SCENE",None)
            scene_metadata={
                "run_id":run_id,"seed":seed,"env_id":args.env_id,
                "env_module":args.env_module,"env_kwargs":args.env_kwargs,
                "control_mode":args.control_mode,"obs_mode":args.obs_mode,
                "sim_backend":args.sim_backend,
                "control_dt_s":float(base.control_timestep),
                "git_commit":sha,
                "scene_spec":spec.to_dict() if spec is not None else None,
                "layout":base.get_scene_layout_for_logging() if hasattr(base,"get_scene_layout_for_logging") else None,
                "stable_initial_distractor_poses":stable_start["distractor_poses"],
                "stable_initial_target_pose":stable_start["target_pose"],
                "stable_initial_robot_qpos":stable_start["robot_qpos"],
                "stable_initial_distractor_linear_velocities":trace.arrays()["distractor_linear_velocities"][0],
                "stable_initial_distractor_angular_velocities":trace.arrays()["distractor_angular_velocities"][0],
                "distractor_ids":list(trace.names),
                "distractor_characteristic_sizes_m":list(trace.sizes),
                "distractor_dimensions_m":[m.dimensions for m in base.distractor_metadata] if hasattr(base,"distractor_metadata") else None,
                "distractor_asset_ids":[m.spec.asset_id or m.spec.kind for m in base.distractor_metadata] if hasattr(base,"distractor_metadata") else None,
            }
            if source_row is not None:
                # Keep the source layout: a restored snapshot does not resample
                # and the environment's last_layout cache is not authoritative.
                scene_metadata["layout"]=source_row["layout"]
                scene_metadata["env_kwargs"]=source_row["env_kwargs"]
                scene_metadata["source_scene_id"]=source_row["scene_id"]
                scene_metadata["source_git_commit"]=source_row.get("git_commit")
            dataset.save_scene(scene_id,snapshot,scene_metadata)
            if args.scene_only:
                print(f"{scene_id}: saved settled scene (seed={seed}); no actions executed")
                continue
            for trajectory_index in range(args.trajectories_per_scene):
                trajectory_id=f"{scene_id}_traj_{trajectory_index:03d}"
                if trajectory_index>0:
                    # Bypass physics settling on restoring exact prior state.
                    runner_env.reset(seed=seed, options={"reset_to_env_states":{"env_states":snapshot}})
                    # ManiSkill RecordEpisode serializes its last_reset_kwargs
                    # into JSON. Do NOT embed a giant nested tensor snapshot;
                    # the canonical initial state is saved under scenes/*.npz.
                    if recorder is not None:
                        recorder.last_reset_kwargs={"seed":seed,"options":{"restored_scene_id":scene_id}}
                    _verify_start(stable_start,_scene_start(runner_env.unwrapped,trace))
                variant=(args.plan_variants[trajectory_index] if args.plan_variants else {})
                config_params={**args.plan_kwargs,**variant}
                config=config_cls(**config_params) if config_cls is not None else None
                try:
                    result=solver(runner_env,seed=seed,config=config,
                                  vis=args.vis,verbose=args.verbose,reset_env=False)
                    status=_status(result,trace.steps)
                except Exception as exc:
                    status={"task_success":False,"stage":"exception",
                            "planner_steps":trace.steps,
                            "message":f"{type(exc).__name__}: {exc}"}
                    print(f"WARNING: {trajectory_id}: {status['message']}")
                attempted+=1
                successes+=int(status["task_success"])
                saved_hdf5=False
                hdf5_file=hdf5_key=None
                if recorder is not None:
                    keep=not args.save_success_only or status["task_success"]
                    if not args.no_record:
                        recorder.flush_trajectory(save=keep)
                        if keep and trace.steps>0:
                            saved_hdf5=True
                            # ManiSkill RecordEpisode implementation fields.
                            hdf5_file=Path(recorder._h5_file.filename).name
                            hdf5_key=f"traj_{recorder._episode_id}"
                    if args.save_video:
                        recorder.flush_video(save=keep)
                dataset.save_trajectory(trajectory_id,scene_id,trace,{
                    "seed":seed,"trajectory_index":trajectory_index,
                    "planner":args.planner,"planner_config":config_params,
                    "control_mode":args.control_mode,"obs_mode":args.obs_mode,
                    "control_dt_s":float(base.control_timestep),
                    "hdf5_file":hdf5_file,"hdf5_trajectory_key":hdf5_key,
                    "saved_hdf5":saved_hdf5,**status,
                })
                print(f"{trajectory_id}: success={status['task_success']} "
                      f"steps={trace.steps} DR={trace.summary()['dr_mean']:.5f}")
    finally:
        runner_env.close()
    if args.scene_only:
        print(f"Saved {args.scenes} scene snapshots without trajectories: {args.output_dir.resolve()}")
    else:
        print(f"Collected {attempted} trajectories in {args.scenes} scenes, "
              f"task successes {successes}/{attempted}. Dataset: {args.output_dir.resolve()}")


if __name__=="__main__":
    main()
