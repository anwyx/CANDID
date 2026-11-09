"""Optional inference profiling.

CUDA Event durations measure GPU stream work, not CPU scheduling time.
A single synchronize per sampled chunk is used for the wall-clock total;
no per-step CUDA .item() synchronizations are introduced.
"""
from __future__ import annotations

from collections import defaultdict
from time import perf_counter

import torch


class PhaseTimer:
    def __init__(self, device, enabled=False):
        self.device = torch.device(device)
        self.enabled = bool(enabled)
        self.cuda = self.enabled and self.device.type == "cuda"
        self.records = defaultdict(list)
        self.start_wall = None
        if self.enabled:
            if self.cuda:
                torch.cuda.synchronize(self.device)
            self.start_wall = perf_counter()

    def start(self):
        if not self.enabled:
            return None
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return perf_counter()

    def stop(self, phase, start):
        if not self.enabled:
            return
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.records[phase].append((start, event))
        else:
            self.records[phase].append((perf_counter() - start) * 1000.0)

    def finish(self):
        if not self.enabled:
            return None
        if self.cuda:
            torch.cuda.synchronize(self.device)
        wall_ms = (perf_counter() - self.start_wall) * 1000.0
        results = {
            name + "_ms": (
                sum(a.elapsed_time(b) for a, b in entries)
                if self.cuda else sum(entries)
            )
            for name, entries in self.records.items()
        }
        for name in ("denoiser", "predictor_gradient", "ddpm_update"):
            results.setdefault(name + "_ms", 0.0)
        results["sampling_wall_ms"] = wall_ms
        results["phase_time_source"] = "cuda_events" if self.cuda else "cpu_wall"
        return results
