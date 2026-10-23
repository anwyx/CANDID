from pathlib import Path
import sys
import json
import numpy as np
import h5py

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from candid.sim.spec import SceneSpec, ObjectSpec, XYRegion
from candid.sim.layout import sample_layout
from candid.data.metrics import disturbance_from_poses
from candid.data.storage import DatasetWriter
from candid.data.windows import windows, prepare_window
from candid.data.validate import validate_dataset


def make_spec():
    T=XYRegion((-.145,-.095),(-.035,.035))
    D=XYRegion((-.18,-.04),(-.095,.095))
    G=XYRegion((.08,.12),(-.04,.04))
    target=ObjectSpec("target","box",T,(-.12,0),size=(.04,.04,.04))
    distractors=tuple(ObjectSpec(f"d{i}","box",D,(-.155,.078 if i%2 else -.078),
                         size=(.04,.04,.04)) for i in range(4))
    return SceneSpec("test",target,distractors,G,(.1,0))


def test_domain_randomization_and_collision():
    spec=make_spec()
    a=sample_layout(spec,[np.sqrt(2)*.02]*5,np.random.RandomState(34))
    b=sample_layout(spec,[np.sqrt(2)*.02]*5,np.random.RandomState(34))
    assert a==b
    seen=[]
    for seed in range(100):
        layout=sample_layout(spec,[np.sqrt(2)*.02]*5,np.random.RandomState(seed))
        seen.append(layout["distractors_xy"][0])
        poses=[layout["target_xy"],*layout["distractors_xy"]]
        for i in range(len(poses)):
            for j in range(i+1,len(poses)):
                assert np.linalg.norm(np.subtract(poses[i],poses[j]))>=2*np.sqrt(2)*.02+.003-1e-8
    assert np.std(np.array(seen)[:,0])>.01


def test_chunk_reference_and_zero_quaternion_sign():
    poses=np.zeros((17,2,7),dtype=float)
    poses[:,:,3]=1.0
    poses[1:,0,0]=np.arange(1,17)*0.001
    poses[:,1,3]=-1.0  # same SO(3) rotation
    sizes=np.array([.04,.08])
    whole=disturbance_from_poses(poses,sizes)
    sub=disturbance_from_poses(poses,sizes,start=5,stop=9)
    assert whole["dr_mean"]>sub["dr_mean"]>0
    assert whole["peak_rotation_rad"]==[0,0]
    act=np.zeros((16,8),dtype=float)
    obs={"robot":np.zeros((17,7))}
    raw={"distractor_poses":poses,"object_sizes_m":sizes}
    window=prepare_window(obs,act,raw,t=5,observation_horizon=2,action_horizon=4)
    assert window["obs"]["robot"].shape==(2,7)
    assert window["actions"].shape==(4,8)
    assert abs(window["predictor_labels"]["dr_mean"]-sub["dr_mean"])<1e-10
    assert list(windows(16,2,4))==list(range(1,13))


class FakeTrace:
    steps=3
    names=["a","b"]
    def arrays(self):
        poses=np.zeros((4,2,7),dtype=np.float32)
        poses[:,:,3]=1
        return {"distractor_poses":poses,
                "distractor_linear_velocities":np.zeros((4,2,3),dtype=np.float32),
                "distractor_angular_velocities":np.zeros((4,2,3),dtype=np.float32),
                "object_sizes_m":np.array([.04,.06],dtype=np.float32)}
    def summary(self):return {"dr_mean":0.,"dr_max":0.}


def test_storage_and_validation(tmp_path):
    ds=DatasetWriter(tmp_path)
    ds.save_scene("scene_1",{"actors":{"a":np.zeros((1,13))}},
                  {"seed":42,"env_id":"demo"})
    with h5py.File(tmp_path/"samples.h5","w") as f:
        g=f.create_group("traj_0")
        g.create_dataset("actions",data=np.zeros((3,8)))
        g.create_dataset("obs",data=np.zeros((4,12)))
    ds.save_trajectory("episode_1","scene_1",FakeTrace(),{
        "hdf5_file":"samples.h5","hdf5_trajectory_key":"traj_0", "saved_hdf5":True})
    assert validate_dataset(tmp_path)=={"num_scenes":1,"num_trajectories":1,"num_recorded_hdf5":1}
    with np.load(tmp_path/"scenes/scene_1.npz",allow_pickle=False) as snap:
        assert len(snap.files)==1


def test_model_input_adapter():
    from candid.data.adapter import canonical_observation
    class G(dict):
        pass
    T1=5;N=2
    agent=G(qpos=np.ones((T1,8)),qvel=np.ones((T1,8)))
    extra=G(tcp_pose=np.zeros((T1,7)),target_pose=np.zeros((T1,7)),
            target_linear_velocity=np.zeros((T1,3)),target_angular_velocity=np.zeros((T1,3)),
            target_dimensions=np.zeros((T1,3)),
            distractor_poses=np.zeros((T1,N*7)),
            distractor_linear_velocities=np.zeros((T1,N*3)),
            distractor_angular_velocities=np.zeros((T1,N*3)),
            distractor_dimensions=np.zeros((T1,N*3)),
            goal_pos=np.zeros((T1,3)))
    obs=canonical_observation(G(agent=agent,extra=extra),N)
    assert obs['distractor_states'].shape==(T1,6,16)
    assert obs['target_state'].shape==(T1,16)
    assert np.all(obs['distractor_mask'][:,:N]==1)
    assert np.all(obs['distractor_mask'][:,N:]==0)


def test_saved_scene_snapshot_can_be_restored(tmp_path):
    from candid.data.storage import save_state_npz, load_state_npz
    orig={"actors":{"x":np.arange(13,dtype=np.float32).reshape(1,13)},
          "articulations":{"arm":np.array([[1,2,3]],dtype=np.float32)}}
    save_state_npz(tmp_path/"scene.npz",orig)
    state=load_state_npz(tmp_path/"scene.npz")
    assert state['actors']['x'].shape==(1,13)
    assert state['articulations']['arm'][0,2].item()==3
