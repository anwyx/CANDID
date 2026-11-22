"""CANDID production teacher sampling: 13 warmups, then feedback/uniform mix.

No ManiSkill or Torch dependency. Each proposed config must be executed and
observed exactly once. Configuration is scene-local and deterministic given the
scene/variant seeds and previously observed successes and DR values.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from candid.runner.adaptive_sampling import AdaptiveSceneSampler, WARMUP, _rng
from candid.runner.feedback_sampler import PairedCandidateSampler
from candid.runner.route_intents import INTENTS

WARMUP_COUNT = len(WARMUP)
DEFAULT_TRAJECTORIES_PER_SCENE = 24


@dataclass(frozen=True)
class ProductionProposal:
    config: dict
    metadata: dict


class ProductionSampler:
    def __init__(self, *, scene_seed, variant_seed, scale, target_xyz,
                 distractor_poses, distractor_ids, distractor_dimensions,
                 initial_tcp_xyz, uniform_every=5):
        if not isinstance(uniform_every, int) or uniform_every < 2:
            raise ValueError("uniform_every must be an integer >= 2")
        self.arguments = dict(
            scene_seed=int(scene_seed), variant_seed=int(variant_seed), scale=float(scale),
            target_xyz=target_xyz, distractor_poses=distractor_poses,
            distractor_ids=distractor_ids, distractor_dimensions=distractor_dimensions,
            initial_tcp_xyz=initial_tcp_xyz,
        )
        self.uniform_every = uniform_every
        self.warmup_sampler = AdaptiveSceneSampler(**self.arguments)
        self.feedback_sampler = None
        self.history = []
        self.pending = None
        self._pending_source = None


    def propose(self):
        if self.pending is not None:
            raise RuntimeError("Must observe the previous proposal before another")
        index = len(self.history)
        if index < WARMUP_COUNT:
            raw = self.warmup_sampler.propose()
            source = "warmup"
            metadata = {**raw.metadata, "collection_phase": "warmup",
                        "selection": "diagnostic", "attempt_index": index}
        else:
            if self.feedback_sampler is None:
                # Successful warmup baseline is a required anchor for relative DR.
                if self.history[0]["success"]:
                    self.feedback_sampler = PairedCandidateSampler(
                        method="feedback", warmup_history=self.history, **self.arguments)
                else:
                    # Preserve failed/atypical scenes rather than crashing the
                    # whole dataset. This is a *fallback*, not feedback sampling.
                    self.feedback_sampler = False
            if self.feedback_sampler is False:
                raw = self.warmup_sampler.propose()
                source = "fallback"
                metadata = {**raw.metadata, "collection_phase": "fallback",
                            "selection": "heuristic_no_successful_baseline",
                            "attempt_index": index}
            else:
                followup_index = index - WARMUP_COUNT
                selection = ("fixed" if followup_index % self.uniform_every == self.uniform_every - 1
                             else "feedback")
                self.feedback_sampler.method = selection
                raw = self.feedback_sampler.propose()
                source = "feedback"
                metadata = {**raw.metadata, "collection_phase": "production",
                            "selection": "uniform" if selection == "fixed" else "feedback",
                            "attempt_index": index}
        self.pending = ProductionProposal(dict(raw.config), metadata)
        self._pending_source = source
        return self.pending

    def observe(self, config, success, dr_mean):
        if self.pending is None or self.pending.config != config:
            raise RuntimeError("Observation does not match the pending proposal")
        dr = float(dr_mean)
        if not math.isfinite(dr) or dr < 0:
            raise ValueError("DR must be finite and nonnegative")
        if self._pending_source in ("warmup", "fallback"):
            self.warmup_sampler.observe(config, bool(success), dr)
        else:
            self.feedback_sampler.observe(config, bool(success), dr)
        self.history.append({"config": dict(config), "success": bool(success), "dr_mean": dr})
        self.pending = None
        self._pending_source = None


class RouteIntentSampler:
    """Paired four-route experiment from one saved scene snapshot.

    Every block of four uses the same grasp/lift variation and changes only
    the transfer intent. No DR feedback biases these comparisons.
    """

    def __init__(self, *, scene_seed, variant_seed, scale, target_xyz,
                 distractor_poses, distractor_ids, distractor_dimensions,
                 initial_tcp_xyz):
        self.scene_seed = int(scene_seed)
        self.variant_seed = int(variant_seed)
        self.geometry = AdaptiveSceneSampler(
            scene_seed=scene_seed, variant_seed=variant_seed, scale=scale,
            target_xyz=target_xyz, distractor_poses=distractor_poses,
            distractor_ids=distractor_ids, distractor_dimensions=distractor_dimensions,
            initial_tcp_xyz=initial_tcp_xyz)
        self.history = []
        self.pending = None
        self.uniform_every = 4

    def propose(self):
        if self.pending is not None:
            raise RuntimeError("Must observe the previous route attempt")
        index = len(self.history)
        block, route_index = divmod(index, len(INTENTS))
        rng = _rng(self.scene_seed, self.variant_seed, block, 0xB047)
        if block == 0:
            stage, style = "baseline", ""
        else:
            stage = ("early", "lift", "combined")[(block - 1) % 3]
            style = ("random_xyz", "clutter_away", "axis_y_pos",
                     "clutter_toward")[(block - 1) % 4]
        config, _ = self.geometry._make(index, rng, stage, style,
                                        f"route_block_{block:03d}")
        config["route_intent"] = INTENTS[route_index]
        metadata = {"collection_phase": "route_intent_experiment",
                    "selection": "route_cycle", "attempt_index": index,
                    "route_intent": INTENTS[route_index],
                    "route_block": block,
                    "paired_grasp_lift_config": True}
        self.pending = ProductionProposal(config, metadata)
        return self.pending

    def observe(self, config, success, dr_mean):
        if self.pending is None or self.pending.config != config:
            raise RuntimeError("Observation does not match route proposal")
        dr = float(dr_mean)
        if not math.isfinite(dr) or dr < 0:
            raise ValueError("DR must be finite and nonnegative")
        self.history.append({"config": dict(config), "success": bool(success),
                             "dr_mean": dr})
        self.pending = None
