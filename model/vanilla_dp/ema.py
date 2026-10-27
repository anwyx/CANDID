"""EMA update used by the ManiSkill Diffusion Policy trainer.

ManiSkill constructs ``diffusers.training_utils.EMAModel`` with ``power=0.75``.
In diffusers 0.32 that power only affects the update when warmup is enabled,
and importing ``training_utils`` also imports a UNet path that fails when
torchvision cannot register its operators. This class is the warmup update
from that ``EMAModel`` (``use_ema_warmup=True``, ``power=0.75``, ``decay=0.9999``):

    decay = 1 - (1 + step) ** -power
    shadow = decay * shadow + (1 - decay) * parameter
"""
from __future__ import annotations

import torch


class EMAModel:
    def __init__(self, parameters, power: float = 0.75, decay: float = 0.9999):
        self.power = float(power)
        self.decay = float(decay)
        self.optimization_step = 0
        self.shadow_params = [parameter.detach().clone() for parameter in parameters]

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - 1)
        if step <= 0:
            return 0.0
        value = 1.0 - (1.0 + step) ** (-self.power)
        return min(value, self.decay)

    @torch.no_grad()
    def step(self, parameters):
        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)
        one_minus = 1.0 - decay
        for shadow, parameter in zip(self.shadow_params, parameters):
            if parameter.requires_grad:
                shadow.sub_(one_minus * (shadow - parameter))
            else:
                shadow.copy_(parameter)

    @torch.no_grad()
    def copy_to(self, parameters) -> None:
        for shadow, parameter in zip(self.shadow_params, parameters):
            parameter.data.copy_(shadow.to(parameter.device).data)
