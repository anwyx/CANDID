"""Online, scene-local teacher-trajectory diversification for CANDID.

This is NOT online replanning inside one rollout.  Each proposal runs from the
same saved simulator snapshot, and its success/DR is fed back only AFTER the
full attempt has been recorded.  Geometry is a TCP-point proxy, not an arm-link
collision checker. No simulator, gym or torch dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

WARMUP = (
    "baseline", "repeat_baseline_1", "repeat_baseline_2",
    "early_x_pos", "early_x_neg", "early_y_pos", "early_y_neg",
    "early_z_pos", "early_z_neg", "lift_x_pos", "lift_x_neg",
    "lift_y_pos", "lift_y_neg",
)
AXES = "xyz"


def _rng(scene_seed, variant_seed, attempt, salt):
    return np.random.default_rng(np.random.SeedSequence([
        int(scene_seed) & 0xffffffff, int(variant_seed) & 0xffffffff,
        int(attempt) & 0xffffffff, int(salt) & 0xffffffff,
    ]))


def _segment_distance(point, a, b):
    d = b-a
    dd = float(np.dot(d, d))
    t = 0.0 if dd < 1e-12 else float(np.clip(np.dot(point-a,d)/dd,0,1))
    return float(np.linalg.norm(point-(a+t*d)))


def _offset_vector(rng, stage, style, scale, target, distractors, names):
    radius = (.05 if stage=="early" else .035)*scale
    zmax = (.022 if stage=="early" else .022)*scale
    chosen = ""
    if style.startswith("axis_"):
        _,axis,sign=style.split("_")
        min_r=(.020 if stage=="early" else .012)*scale if axis!="z" else (.010 if stage=="early" else .008)*scale
        max_r=radius if axis!="z" else zmax
        value=float(rng.uniform(min_r,max_r)) * (1.0 if sign=="pos" else -1.0)
        v=np.zeros(3)
        v[AXES.index(axis)]=value
    elif style.startswith("clutter_"):
        toward=style.endswith("toward")
        # Sample from ALL distractors, rather than only the closest one.
        j=int(rng.integers(len(distractors)))
        chosen=str(names[j])
        direction=distractors[j,:2]-target[:2]
        norm=float(np.linalg.norm(direction))
        if norm<1e-8:
            direction=np.array([1.,0.]); norm=1.0
        direction/=norm
        theta=float(rng.uniform(-.25,.25))
        c,s=math.cos(theta),math.sin(theta)
        rotation=np.array([[c,-s],[s,c]])
        xy=(rotation@direction)*float(rng.uniform(.40*radius,radius))
        if not toward: xy=-xy
        v=np.array([xy[0],xy[1],float(rng.uniform(-.30*zmax,zmax))])
    elif style=="random_xyz":
        r=float(rng.uniform(.30*radius,radius))
        a=float(rng.uniform(-math.pi,math.pi))
        v=np.array([r*math.cos(a),r*math.sin(a),float(rng.uniform(-zmax,zmax))])
    else:
        raise ValueError(f"Unknown style {style}")
    return v,chosen


def _numeric_config(v):
    return np.r_[np.asarray(v["early_offset_xyz"],float)/.05,
                 np.asarray(v["lift_offset_xyz"],float)/.035]


@dataclass(frozen=True)
class Proposal:
    config: dict
    metadata: dict


class AdaptiveSceneSampler:
    """Sequential candidate sampler; propose -> execute/record -> observe.

    First 13 attempts are reproducible diagnostics. Later proposals are chosen
    from a seeded candidate pool using a mix of geometry, observed DR, diversity
    and safety-of-success signals. This is a heuristic, not a trained predictor.
    """
    def __init__(self, *, scene_seed, variant_seed, scale, target_xyz,
                 distractor_poses, distractor_ids, distractor_dimensions,
                 initial_tcp_xyz, approach_height=.065, lift_height=.12,
                 candidates=96):
        self.scene_seed=int(scene_seed)
        self.variant_seed=int(variant_seed)
        self.scale=float(scale)
        if not 0<self.scale<=1.5 or not np.isfinite(self.scale):
            raise ValueError("adaptive scale must lie in (0,1.5]")
        self.target=np.asarray(target_xyz,float)
        self.distractors=np.asarray(distractor_poses,float)
        self.dims=np.asarray(distractor_dimensions,float)
        self.initial_tcp=np.asarray(initial_tcp_xyz,float)
        self.names=list(distractor_ids)
        self.approach_height=float(approach_height)
        self.lift_height=float(lift_height)
        self.candidates=int(candidates)
        if self.target.shape!=(3,) or self.initial_tcp.shape!=(3,):
            raise ValueError("target and TCP must be XYZ")
        n=len(self.names)
        if not 1<=n<=6 or self.distractors.shape not in ((n,3),(n,7)):
            raise ValueError("expected 1..6 distractors, each with XYZ or XYZ+quat")
        if self.dims.shape!=(n,3) or np.any(self.dims<=0):
            raise ValueError("one positive XYZ size per distractor required")
        if not all(np.isfinite(x).all() for x in (self.target,self.initial_tcp,self.distractors,self.dims)):
            raise ValueError("all geometry must be finite")
        if not .025<=self.approach_height<=.15 or not .05<=self.lift_height<=.25:
            raise ValueError("unusual approach/lift heights; update sampling geometry explicitly")
        if self.candidates<8:
            raise ValueError("candidate pool must have at least 8 entries")
        self.history=[]
        self.pending=None

    def _risk(self, cfg):
        # A geometric *TCP waypoint corridor proxy*, not actual robot link
        # clearance. All objects are considered. Ignore fixed final descent,
        # which is identical across variants and would mask differences.
        early=np.asarray(cfg["early_offset_xyz"],float)
        lift=np.asarray(cfg["lift_offset_xyz"],float)
        target=self.target
        pre=target+np.array([0.,0.,self.approach_height])
        lift_top=target+np.array([0.,0.,self.lift_height])
        paths=[]
        if np.linalg.norm(early)>1e-9:
            wp=pre+early
            wp[2]=max(wp[2],target[2]+.030)  # same early-Z floor as round 3
            paths.extend(((self.initial_tcp,wp),(wp,pre)))
        if np.linalg.norm(lift)>1e-9:
            wp=(target+lift_top)/2+lift
            wp[2]=max(wp[2],target[2]+.025)
            paths.extend(((target,wp),(wp,lift_top)))
        if not paths:
            paths=[(self.initial_tcp,pre)]
        best=float("inf")
        nearest=""
        for name,pose,dim in zip(self.names,self.distractors,self.dims):
            radius=float(np.linalg.norm(dim)/2)
            for a,b in paths:
                clearance=_segment_distance(pose[:3],a,b)-radius
                if clearance<best:
                    best=clearance;nearest=str(name)
        # Interpret as proximity only; no claim of collision-free validity.
        risk=math.exp(-max(best,0.0)/.035)
        return float(risk),float(best),nearest

    def _make(self,index,rng,stage,style,tag):
        cfg={"strategy":tag,"early_offset_xyz":[0.,0.,0.],
             "lift_offset_xyz":[0.,0.,0.],"transfer_offset_xyz":[0.,0.,0.]}
        if stage in ("early","lift"):
            off,name=_offset_vector(rng,stage,style,self.scale,self.target,self.distractors,self.names)
            cfg[f"{stage}_offset_xyz"]=off.tolist()
            if name:
                cfg["strategy"]+=f"@{name}"
        elif stage=="combined":
            off,name=_offset_vector(rng,"early",style,self.scale,self.target,self.distractors,self.names)
            cfg["early_offset_xyz"]=off.tolist()
            off2,name2=_offset_vector(rng,"lift","random_xyz",self.scale,self.target,self.distractors,self.names)
            cfg["lift_offset_xyz"]=off2.tolist()
            if name:
                cfg["strategy"]+=f"@{name}"
        elif stage!="baseline":
            raise ValueError(stage)
        risk,clearance,nearest=self._risk(cfg)
        return cfg,{"risk":risk,"proxy_clearance_m":clearance,"proxy_nearest_object":nearest}

    def _thresholds(self):
        base=next((h["dr_mean"] for h in self.history if h["strategy"]=="baseline" and h["success"]),None)
        if base is None:
            return None,None,0.
        spread=max((abs(h["dr_mean"]-base) for h in self.history
                    if h["success"] and h["strategy"].startswith("repeat_baseline")),default=0.)
        return float(base),float(max(.001,.10*base,3*spread)),float(spread)

    def _target_bucket(self,index):
        base,thr,_=self._thresholds()
        if base is None: return "explore"
        vals=[h["dr_mean"] for h in self.history if h["success"]]
        lower=(base>thr and any(v<base-thr for v in vals))
        higher=any(v>base+thr for v in vals)
        need=[]
        if base>thr and not lower:need.append("low")
        if not higher:need.append("high")
        if higher and (lower or base<=thr):
            low,high=min(vals),max(vals)
            mid=.5*(low+high)
            if high-low>4*thr and not any(abs(v-mid)<max(thr,.12*(high-low)) for v in vals):
                need.append("middle")
        # Preserve broad exploration; do not chase one bucket indefinitely.
        k=index-len(WARMUP)
        if k%4==3 or not need:return "explore"
        bucket_tries={name:sum(h.get("target_bucket")==name for h in self.history) for name in need}
        return sorted(need,key=lambda name:(bucket_tries[name],name))[0]

    def _candidate_pool(self,index):
        rng=_rng(self.scene_seed,self.variant_seed,index,0xAD41)
        configs=[]
        for k in range(self.candidates):
            stage=str(rng.choice(("early","lift","combined"),p=(.65,.30,.05)))
            form=int(rng.integers(5))
            if form==0:
                style="clutter_toward"
            elif form==1:
                style="clutter_away"
            elif form==2:
                style="random_xyz"
            else:
                style=f"axis_{AXES[int(rng.integers(3))]}_{'pos' if rng.random()<.5 else 'neg'}"
            cfg,geo=self._make(index,rng,stage,style,f"adaptive_{stage}_{style}_{index:03d}")
            configs.append((cfg,geo))
        return configs

    def _predict(self,cfg):
        # Local kernel of observed SUCCESSFUL outcomes; unsuccessful neighbors
        # lower proposal preference, not converted into high-DR exemplars.
        v=_numeric_config(cfg)
        successes=[h for h in self.history if h["success"]]
        if not successes:return None,0.
        d=np.array([np.linalg.norm(v-_numeric_config(h["config"])) for h in successes])
        weight=np.exp(-.5*(d/.70)**2)
        pred=float(np.dot(weight,[h["dr_mean"] for h in successes])/max(weight.sum(),1e-12))
        novelty=min(1.,float(np.min(d))/1.2)
        failed=[h for h in self.history if not h["success"]]
        fail_penalty=0.
        if failed:
            df=min(float(np.linalg.norm(v-_numeric_config(h["config"]))) for h in failed)
            fail_penalty=.30*math.exp(-.5*(df/.45)**2)
        return pred,novelty-fail_penalty

    def propose(self):
        if self.pending is not None:
            raise RuntimeError("Must observe the preceding trajectory before proposing again")
        index=len(self.history)
        base,threshold,spread=self._thresholds()
        if index<len(WARMUP):
            kind=WARMUP[index]
            rng=_rng(self.scene_seed,self.variant_seed,index,0xAD40)
            if kind=="baseline" or kind.startswith("repeat_baseline"):
                stage,style="baseline",""
            else:
                stage,axis,sign=kind.split("_")
                style=f"axis_{axis}_{sign}"
            cfg,geo=self._make(index,rng,stage,style,kind)
            bucket="baseline" if index<3 else "warmup"
            phase="warmup"
            pred=None;rank=0.
        else:
            bucket=self._target_bucket(index)
            phase="adaptive"
            rng=_rng(self.scene_seed,self.variant_seed,index,0xA551)
            records=[]
            observed=[h["dr_mean"] for h in self.history if h["success"]]
            observed_low=min(observed) if observed else 0.
            observed_high=max(observed) if observed else 0.
            for cfg,geo in self._candidate_pool(index):
                pred,nov=self._predict(cfg)
                if pred is None or observed_high-observed_low<1e-6:
                    scaled=.5
                else:
                    scaled=float(np.clip((pred-observed_low)/(observed_high-observed_low),0,1))
                risk=geo["risk"]
                if bucket=="low":
                    merit=.45*(1-risk)+.35*(1-scaled)+.20*nov
                elif bucket=="high":
                    merit=.45*risk+.35*scaled+.20*nov
                elif bucket=="middle":
                    merit=.45*(1-abs(2*risk-1))+.35*(1-abs(2*scaled-1))+.20*nov
                else:
                    merit=.6*nov+.4*float(rng.random())
                records.append((float(merit),cfg,geo,pred))
            records.sort(key=lambda x:x[0],reverse=True)
            rank,cfg,geo,pred=records[0]
        meta={"phase":phase,"target_bucket":bucket,"attempt_index":index,
              "baseline_dr":base,"control_threshold":threshold,
              "repeat_spread":spread,"risk_proxy":geo["risk"],
              "tcp_proxy_clearance_m":geo["proxy_clearance_m"],
              "tcp_proxy_nearest_object":geo["proxy_nearest_object"],
              "predicted_dr_heuristic":pred,"candidate_score":float(rank),
              "candidate_pool":self.candidates if phase=="adaptive" else 1,
              "variant_seed":self.variant_seed,"variant_scale":self.scale}
        self.pending=Proposal(cfg,meta)
        return self.pending

    def observe(self,config,success,dr_mean):
        if self.pending is None:
            raise RuntimeError("Cannot observe without a proposal")
        if config!=self.pending.config:
            raise ValueError("Observed configuration differs from pending proposal")
        dr=float(dr_mean)
        if not np.isfinite(dr) or dr<0:
            raise ValueError("DR must be finite and nonnegative")
        self.history.append({"config":dict(config),"strategy":config["strategy"],
                             "success":bool(success),"dr_mean":dr,
                             "target_bucket":self.pending.metadata["target_bucket"]})
        self.pending=None
