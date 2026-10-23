from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from candid.data.metrics import event_flags_from_poses, disturbance_from_poses
from candid.data.windows import prepare_window


def upright_trace():
    poses=np.zeros((7,3,7),dtype=np.float64)
    poses[:,:,2]=0.02
    poses[:,:,3]=1
    return poses


def set_x_rotation(poses,t,obj,degrees):
    radians=np.deg2rad(degrees)/2
    poses[t,obj,3:]=[np.cos(radians),np.sin(radians),0,0]


def test_event_detection_and_sphere_exclusion():
    poses=upright_trace()
    poses[1,0,3:]=[np.sqrt(.5),0,0,np.sqrt(.5)] # yaw only
    set_x_rotation(poses,2,0,44)
    set_x_rotation(poses,3,0,60)
    set_x_rotation(poses,3,1,90)
    poses[4,0,2]=-0.02
    poses[5,1,2]=-0.02
    flags=event_flags_from_poses(poses,["box","sphere","ycb"])
    assert flags["distractor_toppled"].shape==(7,3)
    assert not flags["distractor_toppled"][1,0]
    assert not flags["distractor_toppled"][2,0]
    assert flags["distractor_toppled"][3,0]
    assert not flags["distractor_toppled"][:,1].any()
    assert flags["distractor_fallen"][4,0]
    assert flags["distractor_fallen"][5,1]
    assert not flags["distractor_fallen"][3].any()


def test_episode_and_action_chunk_events_not_recounting_prior_events():
    poses=upright_trace()
    set_x_rotation(poses,1,0,75)
    set_x_rotation(poses,2,0,75)
    set_x_rotation(poses,3,1,75)
    set_x_rotation(poses,4,1,75)
    poses[4:,2,2]=-0.02
    flags=event_flags_from_poses(poses,["box"]*3)
    all_labels=disturbance_from_poses(poses,[.04]*3,**{"topple_flags":flags["distractor_toppled"],"fall_flags":flags["distractor_fallen"]})
    assert all_labels["topple_count"]==2
    assert all_labels["fall_count"]==1
    assert all_labels["per_object_topple"]==[True,True,False]
    assert all_labels["per_object_fall"]==[False,False,True]
    chunk=disturbance_from_poses(poses,[.04]*3,start=2,stop=5,
             topple_flags=flags["distractor_toppled"],fall_flags=flags["distractor_fallen"])
    assert chunk["topple_count"]==1 # obj 0 was already tipped at state 2
    assert chunk["fall_count"]==1
    assert chunk["per_object_topple"]==[False,True,False]
    trace={"distractor_poses":poses,"object_sizes_m":np.array([.04]*3),**flags}
    sample=prepare_window({"robot":np.zeros((7,6))},np.zeros((6,8)),trace,
                          t=2,observation_horizon=2,action_horizon=3)
    assert sample["predictor_labels"]["topple_count"]==1
    assert sample["predictor_labels"]["fall_count"]==1
    assert sample["obs"]["robot"].shape==(2,6)


def test_old_traces_still_supported_and_bad_flags_rejected():
    poses=upright_trace()
    labels=disturbance_from_poses(poses,[.04]*3)
    assert "topple_count" not in labels
    with pytest.raises(ValueError,match="boolean"):
        disturbance_from_poses(poses,[.04]*3,topple_flags=np.zeros((7,3)))
    with pytest.raises(ValueError,match="kind"):
        event_flags_from_poses(poses,["box"])
    with pytest.raises(ValueError,match="quaternion"):
        bad=poses.copy()
        bad[0,0,3:]=0
        event_flags_from_poses(bad)
