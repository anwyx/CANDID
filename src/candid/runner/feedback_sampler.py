"""Paired fixed-vs-feedback sampling using one identical candidate pool per slot.

No ManiSkill imports. Feedback uses only its own past successful executed
trajectories, never the fixed arm's outcomes or a geometry=disturbance assumption.
The fixed arm selects uniformly from precisely the same candidate pool.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from candid.runner.adaptive_sampling import AdaptiveSceneSampler, WARMUP, _rng, _numeric_config


@dataclass(frozen=True)
class Choice:
    config: dict
    metadata: dict


def _radius_limits(stage, scale):
    xy = (.05 if stage == "early" else .035) * scale
    z = .022 * scale
    return np.array([xy, xy, z])


class PairedCandidateSampler:
    """At attempt k>=13 choose from a deterministic pool shared by both arms.

    All pool anchors come from the *common* warmup, without looking at their DR.
    Pool IDs therefore match exactly for fixed and feedback arms. History
    affects only feedback ranking of candidates, not candidate generation.
    """

    def __init__(self, *, method, scene_seed, variant_seed, scale, target_xyz,
                 distractor_poses, distractor_ids, distractor_dimensions,
                 initial_tcp_xyz, warmup_history, candidates=96):
        if method not in ("fixed", "feedback"):
            raise ValueError("method must be fixed or feedback")
        if len(warmup_history) != len(WARMUP):
            raise ValueError(f"Expected exactly {len(WARMUP)} shared warmup results")
        self.method = method
        self.scene_seed = int(scene_seed)
        self.variant_seed = int(variant_seed)
        self.scale = float(scale)
        self.history = [dict(h) for h in warmup_history]
        self.warmup = [dict(h) for h in warmup_history]
        self.pending = None
        self.geometry = AdaptiveSceneSampler(
            scene_seed=scene_seed, variant_seed=variant_seed, scale=scale,
            target_xyz=target_xyz, distractor_poses=distractor_poses,
            distractor_ids=distractor_ids, distractor_dimensions=distractor_dimensions,
            initial_tcp_xyz=initial_tcp_xyz, candidates=candidates)
        self.candidates = int(candidates)
        base = next((h["dr_mean"] for h in self.warmup if h["config"]["strategy"] == "baseline" and h["success"]), None)
        if base is None:
            raise ValueError("Shared baseline must have a successful outcome")
        self.baseline = float(base)
        repeat = [abs(float(h["dr_mean"]) - self.baseline) for h in self.warmup
                  if h["success"] and h["config"]["strategy"].startswith("repeat_baseline")]
        self.threshold = max(.001, .10*self.baseline, 3*max(repeat, default=0.))

    def _pool(self, index):
        rng = _rng(self.scene_seed, self.variant_seed, index, 0xFA17)
        anchors = [h["config"] for h in self.warmup if
                   np.linalg.norm(_numeric_config(h["config"])) > 1e-9]
        pool = []
        for k in range(self.candidates):
            stage = str(rng.choice(("early", "lift", "combined"), p=(.70, .27, .03)))
            if k % 2 == 0 or not anchors:
                style = str(rng.choice(("random_xyz", "clutter_toward", "clutter_away",
                                        "axis_x_pos", "axis_x_neg", "axis_y_pos",
                                        "axis_y_neg", "axis_z_pos", "axis_z_neg")))
                cfg, _ = self.geometry._make(index, rng, stage, style,
                                              f"pool_{index:03d}_{k:03d}_{stage}_{style}")
            else:
                # Local perturbation around warmup waypoints, not around outcomes
                # from either evaluation arm. Both arms see exactly this pool.
                anchor = anchors[int(rng.integers(len(anchors)))]
                a = np.array(anchor["early_offset_xyz"], float)
                b = np.array(anchor["lift_offset_xyz"], float)
                source = "early" if np.linalg.norm(a) else "lift"
                lim = _radius_limits(source, self.scale)
                old = a if source == "early" else b
                noise = rng.normal(0., np.array([.012, .012, .008])*self.scale, size=3)
                vec = np.clip(old+noise, -lim, lim)
                cfg = {"strategy":f"pool_{index:03d}_{k:03d}_local_{source}",
                       "early_offset_xyz":vec.tolist() if source == "early" else [0.,0.,0.],
                       "lift_offset_xyz":vec.tolist() if source == "lift" else [0.,0.,0.],
                       "transfer_offset_xyz":[0.,0.,0.]}
            pool.append(cfg)
        return pool

    def _bucket(self, index):
        # Alternate high and low, with periodic unbiased exploration.
        # At near-zero baseline, a meaningfully lower result is impossible.
        k = index-len(WARMUP)
        if k % 4 == 3:
            return "explore"
        if self.baseline <= self.threshold:
            return "high" if k % 3 != 2 else "explore"
        return "low" if k % 2 == 0 else "high"

    def _rank(self, cfg, bucket, rng):
        # Kernel regression is used ONLY when a candidate is near observed
        # successful configurations. Far candidates get neutral exploitation
        # value instead of an unjustified global DR prediction.
        v = _numeric_config(cfg)
        good = [h for h in self.history if h["success"]]
        bad = [h for h in self.history if not h["success"]]
        ds = np.array([np.linalg.norm(v-_numeric_config(h["config"])) for h in good])
        closest = float(ds.min()) if len(ds) else float("inf")
        confidence = float(math.exp(-.5*(closest/.35)**2))
        predicted = None
        if len(ds):
            w = np.exp(-.5*(ds/.28)**2)
            if w.sum() > 1e-8:
                predicted = float(np.dot(w,[h["dr_mean"] for h in good])/w.sum())
        change = 0.0
        if predicted is not None:
            change = float(np.clip((predicted-self.baseline)/max(self.threshold,1e-8),-3,3)/3)
        preference = (change if bucket == "high" else -change if bucket == "low" else 0.)
        novelty = float(np.clip(closest/1.0,0,1))
        fail_penalty = 0.0
        if bad:
            nearest_fail = min(np.linalg.norm(v-_numeric_config(h["config"])) for h in bad)
            fail_penalty = .5*math.exp(-.5*(nearest_fail/.25)**2)
        score = .68*confidence*preference+.18*novelty+.14*float(rng.random())-fail_penalty
        if bucket == "explore":
            score = .7*novelty+.3*float(rng.random())-fail_penalty
        return float(score),predicted,confidence

    def propose(self):
        if self.pending is not None:
            raise RuntimeError("Finish previous attempt before proposing another")
        index = len(self.history)
        pool = self._pool(index)
        select_rng = _rng(self.scene_seed,self.variant_seed,index,0xF1ED)
        if self.method == "fixed":
            selected = int(select_rng.integers(len(pool)))
            target = "uniform"
            score = None
            predicted = None
            confidence = 0.
        else:
            target = self._bucket(index)
            scored = [self._rank(cfg,target,select_rng) for cfg in pool]
            # Deterministic first-index tie break.
            selected = max(range(len(pool)),key=lambda i:scored[i][0])
            score,predicted,confidence = scored[selected]
        cfg = pool[selected]
        info = {"phase":"comparison", "method":self.method,
                "target_bucket":target, "attempt_index":index,
                "candidate_pool_seed":self.variant_seed, "candidate_pool_size":len(pool),
                "pool_selected_index":selected,"score":score,
                "predicted_dr_local":predicted,"prediction_confidence":confidence,
                "baseline_dr":self.baseline,"control_threshold":self.threshold}
        self.pending = Choice(cfg,info)
        return self.pending

    def observe(self,config,success,dr_mean):
        if self.pending is None or config != self.pending.config:
            raise RuntimeError("Observed config does not match pending proposal")
        value = float(dr_mean)
        if not math.isfinite(value) or value < 0:
            raise ValueError("DR must be finite and nonnegative")
        self.history.append({"config":dict(config),"success":bool(success),"dr_mean":value})
        self.pending = None
