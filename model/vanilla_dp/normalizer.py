"""Map absolute ``pd_joint_pos`` commands onto the [-1, 1] range the scheduler clips to."""
from __future__ import annotations

import torch


class ActionRangeNormalizer:
    def __init__(self, minimum: torch.Tensor, maximum: torch.Tensor):
        if minimum.shape != maximum.shape:
            raise ValueError("Action bounds must have the same shape")
        self.minimum = minimum.detach().float()
        self.maximum = maximum.detach().float()

    def to(self, device):
        return ActionRangeNormalizer(self.minimum.to(device), self.maximum.to(device))

    def _span(self):
        return torch.clamp(self.maximum - self.minimum, min=1e-6)

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.minimum) * (2.0 / self._span()) - 1.0

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        return (action + 1.0) * 0.5 * self._span() + self.minimum


def fit_action_normalizer(actions) -> ActionRangeNormalizer:
    """``actions`` is a numpy array shaped ``[N, action_dim]``."""
    import numpy as np

    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Action statistics require a finite [N, action_dim] array")
    return ActionRangeNormalizer(
        torch.from_numpy(values.min(axis=0).copy()),
        torch.from_numpy(values.max(axis=0).copy()),
    )


class ObservationNormalizer:
    """Z-score observations and clip outliers; near-constant features keep unit scale."""

    def __init__(self, mean, std, clip: float = 5.0):
        import numpy as np

        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.clip = float(clip)

    @classmethod
    def fit(cls, observations):
        import numpy as np

        values = np.asarray(observations, dtype=np.float32)
        std = values.std(axis=0)
        return cls(values.mean(axis=0), np.where(std < 1e-4, 1.0, std))

    def apply(self, observations):
        import numpy as np

        scaled = (np.asarray(observations, dtype=np.float32) - self.mean) / self.std
        return np.clip(scaled, -self.clip, self.clip).astype(np.float32)
