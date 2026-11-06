"""Minimal, API-compatible EMA subset for ManiSkill's state-based ACT.

Adapted from diffusers v0.32.2 diffusers.training_utils.EMAModel (Apache-2.0):
https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/training_utils.py

The original ManiSkill ACT uses only EMAModel(parameters, power=0.75),
step(parameters), and copy_to(parameters). Keeping those exact operations here
avoids importing the *unrelated* UNet2D/torchvision stack through
diffusers.training_utils, which can fail with torchvision::nms on CPU.

In the original defaults, use_ema_warmup=False; 'power' does not affect the
(non-warmup) decay schedule. We intentionally preserve this behavior.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch


class EMAModel:
    """Diffusers 0.32.2 EMA behavior for the methods used by state ACT."""

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: float = 1.0,
        power: float = 2.0 / 3.0,
    ):
        parameters = list(parameters)
        self.shadow_params = [p.detach().clone() for p in parameters]
        self.decay = float(decay)
        self.min_decay = float(min_decay)
        self.update_after_step = int(update_after_step)
        self.use_ema_warmup = bool(use_ema_warmup)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.optimization_step = 0
        self.cur_decay_value = None

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        if self.use_ema_warmup:
            value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        else:
            value = (1.0 + step) / (10.0 + step)
        return max(self.min_decay, min(value, self.decay))

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        parameters = list(parameters)
        if len(parameters) != len(self.shadow_params):
            raise ValueError("EMA parameter count changed")
        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        for shadow, parameter in zip(self.shadow_params, parameters):
            if parameter.requires_grad:
                shadow.sub_((1.0 - decay) * (shadow - parameter))
            else:
                shadow.copy_(parameter)

    @torch.no_grad()
    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        parameters = list(parameters)
        if len(parameters) != len(self.shadow_params):
            raise ValueError("EMA parameter count changed")
        for shadow, parameter in zip(self.shadow_params, parameters):
            parameter.data.copy_(shadow.to(parameter.device).data)
