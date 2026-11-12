"""CPU-only checks for five-scene training and Best-of-N CLI arguments."""
from argparse import Namespace
from pathlib import Path

from scripts.build_joint_counterfactuals import choose_balanced
from scripts.run_joint5_methods import original_v2_command


def test_joint_subset_is_deterministic_and_bounded():
    rows = [
        {"state_id": str(i), "stratum": ["near_contact", "pre_topple", "low_dr"][i % 3]}
        for i in range(100)
    ]
    a = choose_balanced(rows, 64, 2026)
    b = choose_balanced(rows, 64, 2026)
    assert len(a) == 64
    assert len({r["state_id"] for r in a}) == 64
    assert [r["state_id"] for r in a] == [r["state_id"] for r in b]


def test_original_joint_training_config():
    meta = {
        "counterfactual_states": 64,
        "init_checkpoint": None,
        "config": {
            "dr_objective": "ordinal", "obs_horizon": 2, "action_horizon": 8,
            "batch_size": 64, "hidden": 256, "lr": 0.0001, "seed": 2026,
            "split_seed": 0, "val_ratio": 0.15, "test_ratio": 0.15,
            "log_freq": 50, "eval_freq": 250, "val_max_windows": 4096,
            "val_batch_size": 256, "dr_thresholds": [0.001, 0.01],
            "ordinal_pos_weight": 5,
        },
        "weights": {"expert": 1.0, "counterfactual": 0.3, "ranking": 0.5},
        "ranking": {"temperature": 0.1, "topple_weight": 0.5,
                    "fall_weight": 0, "dr_max_weight": 0.25, "min_risk_gap": 0.01},
        "optimizer": {"weight_decay": 0},
        "training_controls": {"cf_update_every": 8,
                              "cf_pos_weight_cap": 3, "early_stop_patience": 0},
    }
    args = Namespace(
        steps=5000, dataset=Path("demos/joint"),
        cf_dataset=Path("runs/joint_cf"), output_root=Path("runs/joint"))
    cmd = original_v2_command(meta, args)
    assert cmd[cmd.index("--steps")+1] == "5000"
    assert cmd[cmd.index("--cf-update-every")+1] == "8"
    assert cmd[cmd.index("--rank-weight")+1] == "0.5"
