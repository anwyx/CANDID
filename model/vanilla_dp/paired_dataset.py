"""Two temporally aligned expert windows from the SAME trajectory."""
from __future__ import annotations

from torch.utils.data import Dataset


class PairedDiffusionWindowDataset(Dataset):
    """Pair windows that start ``stride`` steps apart (e.g. 4 steps).

    Restrict pairs to those whose executed action horizons contain real,
    non-padded expert actions. Validation keeps using the original dataset.
    """

    def __init__(self, base_dataset, *, stride: int = 4, act_horizon: int = 8):
        self.base = base_dataset
        self.stride = int(stride)
        self.act_horizon = int(act_horizon)
        if not 0 < self.stride < self.act_horizon:
            raise ValueError("Require 0 < stride < act_horizon")
        if base_dataset.obs_horizon - 1 + self.act_horizon > base_dataset.pred_horizon:
            raise ValueError("Executed actions must fit inside prediction horizon")

        lookup = {
            (trajectory_index, start): index
            for index, (trajectory_index, start, _) in enumerate(base_dataset.slices)
        }
        self.pairs = []
        for index, (trajectory_index, start, _) in enumerate(base_dataset.slices):
            other = lookup.get((trajectory_index, start + self.stride))
            if other is None:
                continue
            episode_length = len(base_dataset.sequences[trajectory_index][1])
            # Both executed 8-step slices must be entirely within real data.
            second_end = (
                start + self.stride + (base_dataset.obs_horizon - 1)
                + self.act_horizon
            )
            if second_end <= episode_length:
                self.pairs.append((index, other))
        if not self.pairs:
            raise ValueError("No valid paired diffusion windows: check trajectory lengths")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        first_index, second_index = self.pairs[index]
        first = self.base[first_index]
        second = self.base[second_index]
        return {
            "observations": first["observations"],
            "actions": first["actions"],
            "next_observations": second["observations"],
            "next_actions": second["actions"],
        }
