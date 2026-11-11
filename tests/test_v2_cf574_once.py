"""CLI contracts for one-time Scene1 v2 CF574 comparison (no GPU needed)."""
from scripts.run_scene1_v2_cf574_once import old_training_command, old_evaluation_command, TRAIN_OUT


def test_training_command_restores_old_hyperparameters():
    old = {
        "config": {
            "obs_horizon": 2, "action_horizon": 8, "dr_objective": "ordinal",
            "steps": 5000, "batch_size": 64, "hidden": 256, "lr": 1e-4,
            "seed": 2026, "split_seed": 0, "val_ratio": 0.15,
            "test_ratio": 0.15, "log_freq": 50, "eval_freq": 100,
            "val_max_windows": 4096, "val_batch_size": 256,
            "dr_thresholds": [0.0001, 0.01, 0.1],
            "ordinal_pos_weight": 5,
        },
        "counterfactual_states": 64, "steps_completed": 3000,
        "init_checkpoint": None,
        "ranking": {"temperature": 0.1, "topple_weight": 0.5,
                    "fall_weight": 0.0, "dr_max_weight": 0.25,
                    "min_risk_gap": 0.01},
        "weights": {"expert": 1, "counterfactual": 0.3, "ranking": 0.5},
        "optimizer": {"weight_decay": 0.01},
        "training_controls": {"cf_update_every": 8, "cf_pos_weight_cap": 3,
                              "early_stop_patience": 0}
    }
    cmd = old_training_command(old)
    assert cmd[cmd.index("--steps")+1] == "3000"
    assert cmd[cmd.index("--cf-update-every")+1] == "8"
    assert cmd[cmd.index("--weight-decay")+1] == "0.01"
    assert "--allow-incomplete-counterfactual" in cmd
    assert cmd[cmd.index("--output-dir")+1] == str(TRAIN_OUT)
