#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E02-R4 A/B/C: controlled Bootstrap query-cost and region-coverage ablations.

A: EXACT E02-R3 Bootstrap-to-SimpleAIF selection and adaptation.
B: A, except Bootstrap steps are excluded from adaptive query-cost updates
   (they remain in the true label budget, query counters, and STM).
C: B, plus predeclared, score-only regions with quotas, early normal-side
   audit priority, and deterministic budget pacing. Never uses unseen labels.

Common canonical v22 init, hard K labels per variant, same stream & frozen
query theta reference. Four sidecars per variant share the same STM scores
and verified labels. Original v22, R3 and all older experiments UNMODIFIED.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np


def dependencies():
    root = Path(__file__).resolve().parents[1]
    for d in (root, root / "experiments"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    import test_phase3_AIF_simplified_v22 as v22
    import e02_matched_feedback as e02
    import e02_r1_r2 as ext
    import p0_v22_equivalence as e01
    return v22, e02, ext, e01


def sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class Budget:
    """Strict K query budget with deterministic, label-blind terminal catchup.

    This is NOT the old v22 unconstrained local-qrate algorithm. Query
    decisions use genuine v22 policy primitives, then are capped/supplemented
    for a fair equal-label-budget comparison.
    """
    def __init__(self, n: int, rate: float):
        if n < 1 or not 0 <= rate <= 1:
            raise ValueError("Invalid stream size or rate")
        self.n = n
        self.K = min(n, int(math.floor(n * rate + .5)))
        self.used = 0
        self.suppressed = 0
        self.catchup = 0

    def decide(self, t: int, proposal: bool) -> tuple[bool, str]:
        if not 1 <= t <= self.n:
            raise ValueError("Invalid stream index")
        if self.used >= self.K:
            if proposal:
                self.suppressed += 1
            return False, "global_cap"
        remaining_steps = self.n - t + 1
        remaining_budget = self.K - self.used
        if remaining_steps == remaining_budget:
            self.used += 1
            if not proposal:
                self.catchup += 1
            return True, "terminal_budget_catchup"
        if proposal:
            self.used += 1
            return True, "policy"
        return False, "policy_classify"

    def check(self):
        if self.used != self.K:
            raise AssertionError(f"Query budget mismatch {self.used}!={self.K}")



class ScoreRegionBudget:
    """C: predefined query quotas over *predicted* score regions, never labels.

    Normals here mean low-anomaly-SCORE region, not true normal samples.
    Caps do not guarantee label composition. Short-run terminal catchup can
    spill beyond regional caps; such spill is counted and reported.
    """
    NAMES = ("low_score", "boundary", "high_score")

    def __init__(self, n: int, k: int, *, early_steps: int = 400,
                 low_fraction: float = .45, boundary_fraction: float = .35,
                 early_total_fraction: float = .40):
        if n < 1 or k < 0 or not 0 <= low_fraction <= 1 or not 0 <= boundary_fraction <= 1 \
                or low_fraction + boundary_fraction > 1:
            raise ValueError("Bad regional budget configuration")
        self.n = int(n)
        self.k = int(k)
        self.early_steps = min(n, max(1, int(early_steps)))
        low = int(math.floor(k * low_fraction + .5))
        band = int(math.floor(k * boundary_fraction + .5))
        self.quotas = {"low_score":low,"boundary":band,"high_score":k-low-band}
        self.counts = {name:0 for name in self.NAMES}
        self.observed = {name:0 for name in self.NAMES}
        self.overrides = {name:0 for name in self.NAMES}
        self.early_total_cap = min(k,int(math.floor(k*early_total_fraction+.5)))
        self.force_low = 0
        self.block_quota = 0
        self.block_pacing = 0

    @staticmethod
    def region(score: float, theta: float, sigma: float) -> str:
        z = (float(score)-float(theta))/max(abs(float(sigma)),1e-6)
        if z < -1.0:
            return "low_score"
        if z > 1.0:
            return "high_score"
        return "boundary"

    def allowed_total(self, t: int) -> int:
        if t <= self.early_steps:
            # Intentional bounded frontloading: no more than 40% of labels in
            # the first 400 steps (or entire smoke test when N<400).
            return self.early_total_cap
        fraction=(t-self.early_steps)/max(1,self.n-self.early_steps)
        return min(self.k, self.early_total_cap
                   + int(math.floor((self.k-self.early_total_cap)*fraction+.5)))

    def propose(self, t:int, region:str, native:bool, global_budget:Budget,
                *, target_normals:int, verified_normal_count:int) -> tuple[bool,str]:
        if region not in self.quotas:
            raise ValueError("Unknown score region")
        self.observed[region] += 1
        if self.counts[region] >= self.quotas[region]:
            self.block_quota += int(native)
            return False,"stratum_quota_exhausted"
        if global_budget.used >= self.allowed_total(t):
            self.block_pacing += int(native)
            return False,"regional_time_pace"
        if (region=="low_score" and t<=self.early_steps
              and verified_normal_count < target_normals):
            # A score-conditional intervention; verified_normal_count is
            # historical queried-only feedback, NOT current oracle label.
            self.force_low += int(not native)
            return True,"low_score_bootstrap_priority"
        # After first 400 steps ensure representation of score regions, rather
        # than allowing all budget to be spent on native boundary proposals.
        if t > self.early_steps and self.counts[region] < \
                int(math.floor(self.quotas[region]*t/self.n+.5)):
            return True,"stratum_deficit"
        return bool(native),"native_policy"

    def observe(self, region:str, asked:bool) -> None:
        if asked:
            self.counts[region] += 1
            if self.counts[region]>self.quotas[region]:
                self.overrides[region] += 1

    def snapshot(self) -> dict:
        return {"quotas":self.quotas,"queried_counts":self.counts,
                "observed_region_counts":self.observed,
                "terminal_cap_overflow":self.overrides,
                "early_total_cap":self.early_total_cap,
                "forced_low_score":self.force_low,
                "blocked_quota":self.block_quota,
                "blocked_pacing":self.block_pacing}


def record_query_cost(state:dict, asked:bool, *, bootstrap:bool) -> None:
    """A uses unmodified v22 adaptation. B/C retain ALL actual-query flags
    but adapt cost only from post-Bootstrap query decisions.
    """
    aif=state["aif"]
    aif.observe(asked)
    if state["name"]=="A":
        return
    if bootstrap:
        state["excluded_cost_steps"] += 1
        state["excluded_cost_queries"] += int(asked)
        return
    flags=state["post_bootstrap_flags"]
    flags.append(int(asked))
    window=max(1,int(getattr(aif.cfg,"local_qrate_window",256)))
    if len(flags)>window:
        del flags[:-window]
    local=sum(flags)/len(flags)
    aif.query_cost_runtime=float(np.clip(
        aif.query_cost_runtime+float(aif.cfg.query_cost_lr)*
        (local-float(aif.cfg.target_qrate)),
        float(aif.cfg.min_query_cost),float(aif.cfg.max_query_cost)))
    state["post_bootstrap_adapt_steps"]+=1


def longest_gap(steps:list[int],n:int) -> int:
    last=0
    longest=0
    for t in steps+[n+1]:
        longest=max(longest,t-last-1)
        last=t
    return longest


def stability_windows(rows:list[dict],stride:int=256) -> list[dict]:
    """Evaluation-only diagnostic. No result is fed to query or sidecar."""
    arms=("static","frozen","guarded","balanced_rescue","mistake_feedback")
    out=[]
    for lo in range(0,len(rows),stride):
        group=rows[lo:lo+stride]
        labels=[int(r["y_eval_only"]) for r in group]
        normals=sum(y==0 for y in labels)
        defects=len(labels)-normals
        entry={"start":int(group[0]["t"]),"end":int(group[-1]["t"]),
               "n":len(group),"n_normal":normals,"n_defect":defects,
               "queried":sum(int(r["queried"]) for r in group),
               "queried_normal":sum(int(r["queried"]) for r in group if r["y_eval_only"]==0),
               "queried_defect":sum(int(r["queried"]) for r in group if r["y_eval_only"]==1)}
        for arm in arms:
            key="static_score" if arm=="static" else "stm_score"
            preds=[int(r[f"{arm}_pred"]) for r in group]
            fp=sum(y==0 and pred==1 for y,pred in zip(labels,preds))
            fn=sum(y==1 and pred==0 for y,pred in zip(labels,preds))
            entry[arm]={"fpr":fp/normals if normals else None,
                        "fnr":fn/defects if defects else None,
                        "balanced_accuracy":.5*(1-fp/normals+1-fn/defects)
                            if normals and defects else None,
                        "mean_score":sum(float(r[key]) for r in group)/len(group)}
        out.append(entry)
    return out

def source_config(v22, args):
    return v22.SimpleAIFConfig(
        query_cost=args.query_cost,
        beta_entropy=args.beta_entropy,
        target_qrate=args.query_rate,
        use_local_saliency=args.local_saliency,
        bootstrap_dynamic_target=args.bootstrap_dynamic_target,
        bootstrap_max_steps=args.bootstrap_max_steps,
        bootstrap_band_min_abs=args.bootstrap_band_min_abs,
        v22_enable_warmup_budget=False,
        v22_enable_defect_audit=False,
        v22_enable_fn_audit=False,
    )


def policy_decision(v22, state, *, t, score, theta, sigma, p, amap, global_feat):
    """Native v22 primitives using same normal_dyn_only utility configuration."""
    c = state["coverage"]
    cfg = state["policy_cfg"]
    pct, novelty_info = c.novelty_percentile(global_feat)
    novelty = float(np.clip(
        (pct - cfg.coverage_ref_clip_low) /
        max(cfg.coverage_ref_clip_high - cfg.coverage_ref_clip_low, 1e-6),
        0, 1,
    ))
    state["last_novelty"] = novelty
    count = int(c.dynamic_feats.shape[0])
    a, reason, info = v22.choose_simple_aif_bootstrap_action(
        step=t, score=score, p_defect=p, theta=theta, sigmaN=sigma,
        cfg=cfg, dynamic_normal_count=count,
    )
    if a is not None:
        if info.get("bootstrap_active"):
            state["bootstrap_steps"] += 1
        if a == v22.AIFAction.QUERY_LABEL:
            state["bootstrap_proposals"] += 1
        return bool(a == v22.AIFAction.QUERY_LABEL), f"bootstrap::{reason}"
    # No use of the current sample's label. The original v22 score
    # posterior and two-channel evidence drive the AIF objective.
    evidence = v22.compute_two_channel_evidence(
        score=score, theta=theta, sigmaN=sigma, A_map_stats=amap
    )
    action, info = v22.select_simple_aif_action(
        cfg=cfg, state=state["aif"], ablation="normal_dyn_only",
        p_defect=p, score=score, theta=theta, sigmaN=sigma,
        pred_defect=score >= theta,
        novelty=novelty, novelty_info=novelty_info,
        evidence=evidence, A_map_stats=amap,
    )
    return bool(action == v22.AIFAction.QUERY_LABEL), "simple_aif"


def init_arm(v22, e02, ext, canonical, init, backbone, args, policy_name,
             fixed_sha, base_theta):
    static_bank = copy.deepcopy(canonical)
    bank = copy.deepcopy(canonical)
    for b in (static_bank, bank):
        ext.disabled_sidecars(b)
    policy_ref = copy.deepcopy(base_theta)
    arm = e02.Arm(policy_name, True, False, bank, copy.deepcopy(base_theta),
                  [], [], [])
    guarded = copy.deepcopy(base_theta)
    balanced = copy.deepcopy(base_theta)
    balanced.cfg.enable_balanced_rescue = True
    controllers = {
        "frozen": None,
        "guarded": guarded,
        "balanced_rescue": balanced,
        "mistake_feedback": ext.MistakeFeedback(
            float(base_theta.theta()), step=args.mistake_step)
    }
    support_g = []
    # v22 coverage reference uses global descriptors from the same supports.
    # Encoding original support images is read-only and never accesses labels
    # of the test stream.
    for p in init["support_paths_per_mode"]["0"]:
        image = v22.load_rgb(p)
        _, g = v22.encode_pil(backbone, image, args.feature_fuse)
        support_g.append(g)
    coverage = v22.NormalCoverageMemory.from_support_feats(
        support_g, max_dynamic=args.fixed_capacity if args.fixed_capacity < 1024 else 1024
    )
    conf = source_config(v22, args)
    # A runs original v22 cost feedback unmodified; for B/C we keep
    # observed-query counters but isolate adaptive cost history.
    if policy_name in ("B","C"):
        conf.adapt_query_cost = False
    state = {
        "name": policy_name,
        "static": static_bank,
        "arm": arm,
        "controller_ref": policy_ref,  # frozen for policy, independent of sidecars
        "controllers": controllers,
        "policy_cfg": conf, "aif": v22.SimpleAIFState(conf),
        "post_bootstrap_flags": [],
        "post_bootstrap_adapt_steps":0,
        "excluded_cost_steps":0,
        "excluded_cost_queries":0,
        "strata": ScoreRegionBudget(args.max_steps_effective,
                int(math.floor(args.max_steps_effective*args.query_rate+.5)),
                early_steps=args.strata_early_steps,
                low_fraction=args.strata_low_fraction,
                boundary_fraction=args.strata_boundary_fraction,
                early_total_fraction=args.strata_early_budget_fraction)
                if policy_name=="C" else None,
        "cost_initial":float(conf.query_cost),
        "coverage": coverage,
        "budget": Budget(args.max_steps_effective, args.query_rate),
        "last_novelty": 0.0, "bootstrap_steps": 0,
        "bootstrap_proposals": 0,
        "queried_normals": 0, "queried_defects": 0,
        "queried_positions": [], "query_reasons": {},
        "sidecar_moves": {k: [] for k in controllers},
        "scores": {"static": [], **{k: [] for k in controllers}},
        "pred": {"static": [], **{k: [] for k in controllers}},
        "thetas": {"static": [], **{k: [] for k in controllers}},
        "fixed_sha": fixed_sha,
    }
    return state


def step_state(v22, e02, state, *, t, sample, image, features, global_feat, args,
               evaluation_rows: list[dict]):
    s_static, _ = e02.frozen_score(
        v22, type("_Static", (), {"bank": state["static"]})(),
        args.class_name, features, image, args
    )
    score, amap = e02.frozen_score(
        v22, state["arm"], args.class_name, features, image, args
    )
    theta0 = float(state["controller_ref"].theta())
    sigma = float(state["controller_ref"].sigmaN())
    p = float(state["controller_ref"].posterior(score))
    if not all(map(math.isfinite,(s_static,score,theta0,sigma,p))):
        raise AssertionError("Non-finite prediction/policy state")
    proposed, reason = policy_decision(
        v22, state, t=t, score=score, theta=theta0,
        sigma=sigma, p=p, amap=amap, global_feat=global_feat)
    native_proposal=bool(proposed)
    bootstrap=reason.startswith("bootstrap::")
    score_region=ScoreRegionBudget.region(score,theta0,sigma)
    region_reason="not_applicable"
    if state["strata"] is not None:
        proposed,region_reason=state["strata"].propose(
            t,score_region,proposed,state["budget"],
            target_normals=args.bootstrap_dynamic_target,
            verified_normal_count=int(state["coverage"].dynamic_feats.shape[0]))
    asked,budget_reason=state["budget"].decide(t,proposed)
    if state["strata"] is not None:
        state["strata"].observe(score_region,asked)
    # Record EVERY prediction, for ALL threshold candidates, before reading label.
    row = {
        "t": t, "path": sample.path, "policy": state["name"],
        "queried": int(asked), "query_reason": reason,
        "budget_reason": budget_reason,
        "query_cost_pre":float(state["aif"].query_cost_runtime),
        "native_proposal":int(native_proposal),
        "score_region":score_region,
        "region_selection":region_reason,
        "bootstrap_active":int(bootstrap),
        "stm_score": score, "static_score": s_static,
        "bootstrap_coverage_normal_count": int(state["coverage"].dynamic_feats.shape[0]),
    }
    state["scores"]["static"].append(s_static)
    state["thetas"]["static"].append(theta0)
    state["pred"]["static"].append(int(s_static>=theta0))
    row.update(static_theta=theta0,static_pred=int(s_static>=theta0))
    for name, controller in state["controllers"].items():
        theta = theta0 if controller is None else float(controller.theta())
        decision = int(score >= theta)
        state["scores"][name].append(score)
        state["pred"][name].append(decision)
        state["thetas"][name].append(theta)
        row[f"{name}_theta"] = theta
        row[f"{name}_pred"] = decision
    # No query decisions above have access to the true label.
    truth = 0 if sample.is_good else 1
    row["y_eval_only"] = truth
    evaluation_rows.append(row)
    state["query_reasons"][reason] = state["query_reasons"].get(reason,0)+int(asked)
    if asked:
        state["queried_positions"].append(t)
        if truth == 0:
            state["queried_normals"] += 1
            # E02 verified-normal STM primitive. Novelty intentionally held
            # at zero for matched comparison with prior E02/R2 baseline.
            e02.write_verified_normal(
                v22, state["arm"], args.class_name,
                features, amap, score, theta0, t
            )
            state["coverage"].observe_normal(global_feat)
        else:
            state["queried_defects"] += 1
        for name, ctrl in state["controllers"].items():
            if ctrl is not None:
                old = float(ctrl.theta())
                ctrl.observe_labeled(score, truth)
                new = float(ctrl.theta())
                if abs(new-old)>1e-12:
                    state["sidecar_moves"][name].append({
                        "step":t, "before":old, "after":new,
                        "last_mode":getattr(ctrl,"last_mode",None),
                        "last_eval":getattr(ctrl,"last_eval",None)
                    })
    record_query_cost(state,asked,bootstrap=bootstrap)
    if not math.isfinite(score):
        raise AssertionError("Non-finite prequential score")


def report_state(e02, ext, e01, state, truth, args) -> dict:
    state["budget"].check()
    if len(state["queried_positions"]) != state["budget"].K:
        raise AssertionError("Mismatch between labeled positions and fixed budget")
    if state["queried_normals"]+state["queried_defects"] != state["budget"].K:
        raise AssertionError("Mismatch in verified labels")
    if e01.memory_profile(state["arm"].bank,args.class_name)["sha256"] != state["fixed_sha"]:
        raise AssertionError("Immutable Fixed Bank mutated during stream")
    if e01.memory_profile(state["static"],args.class_name)["sha256"] != state["fixed_sha"]:
        raise AssertionError("Static Fixed Bank mutated")
    if ext.stm_total(state["static"],args.class_name) != 0:
        raise AssertionError("Static model received STM updates")
    arm_metrics={}
    theta0=float(state["controller_ref"].theta())
    for name in state["scores"]:
        ctrl=state["controllers"].get(name)
        scores=state["scores"][name]
        vals=state["thetas"][name]
        if name not in {"static","frozen"} and len(scores)!=len(truth):
            raise AssertionError("Unequal length of arm results")
        if name not in {"static","frozen"} and scores!=state["scores"]["frozen"]:
            raise AssertionError("Threshold controller changed the STM score trajectory")
        if name=="frozen" and any(abs(v-theta0)>1e-10 for v in vals):
            raise AssertionError("Frozen threshold mutated")
        result=e02.metrics(truth,scores,state["pred"][name])
        arm_metrics[name]={
            **result,
            "theta0": theta0,
            "theta_final": theta0 if ctrl is None else float(ctrl.theta()),
            "theta_changes": len(state["sidecar_moves"].get(name,[])),
            "n_labeled": 0 if ctrl is None else
                (ctrl.observed if hasattr(ctrl,"observed") else
                 ctrl.n_normals_labeled+ctrl.n_defects_labeled)
        }
        if ctrl is not None and arm_metrics[name]["n_labeled"]!=state["budget"].K:
            raise AssertionError(f"{name}: queried count mismatch")
    return {
        "name":state["name"], "queries":state["budget"].used,
        "queried_normals":state["queried_normals"],
        "queried_defects":state["queried_defects"],
        "query_indices_1based":state["queried_positions"],
        "query_tape_sha256":sha(state["queried_positions"]),
        "budget_suppressed":state["budget"].suppressed,
        "budget_forced_terminal":state["budget"].catchup,
        "first_200_queries":sum(x<=200 for x in state["queried_positions"]),
        "first_400_queries":sum(x<=400 for x in state["queried_positions"]),
        "bootstrap_steps":state["bootstrap_steps"],
        "bootstrap_proposals":state["bootstrap_proposals"],
        "bootstrap_verified_normals":int(state["coverage"].dynamic_feats.shape[0]),
        "query_reasons":state["query_reasons"],
        "stm_write_calls":state["arm"].verified_normal_update_calls,
        "stm_prototypes":ext.stm_total(state["arm"].bank,args.class_name),
        "arm_metrics":arm_metrics,
        "sidecar_moves":state["sidecar_moves"],
        "fixed_sha256":state["fixed_sha"],
        "cost_final":float(state["aif"].query_cost_runtime),
        "cost_excluded_bootstrap_steps":state["excluded_cost_steps"],
        "cost_excluded_bootstrap_queries":state["excluded_cost_queries"],
        "cost_post_bootstrap_updates":state["post_bootstrap_adapt_steps"],
        "strata":state["strata"].snapshot() if state["strata"] is not None else None,
        "longest_query_free_gap":longest_gap(state["queried_positions"],len(truth)),
    }



def verify_r3_a(reference_summary: str, reference_csv: str,
                current_report: dict, current_rows: list[dict],
                args) -> dict:
    """Must be identical to original R3 Bootstrap, not merely similar final AUROC.

    This is a local-file replay audit. Do not compare nonmatching configs.
    """
    if "A" not in current_report["schedules"]:
        raise AssertionError("R3 reference audit requires variant A")
    prior=json.loads(Path(reference_summary).read_text(encoding="utf-8"))
    if prior["schema"]!="e02.r3.bootstrap.sidecar.audit.v1":
        raise AssertionError("Not a valid E02-R3 reference")
    for key in ("class_name","support_seed","legacy_seed","stream_seed",
                "query_rate","max_samples","query_cost","beta_entropy",
                "local_saliency","bootstrap_dynamic_target","bootstrap_max_steps",
                "bootstrap_band_min_abs"):
        if prior["parameters"][key]!=getattr(args,key):
            raise AssertionError(f"Incomparable R3 reference setting: {key}")
    for key in ("canonical_fixed_sha256","canonical_theta0",
                "stream_paths_sha256","support_paths","n_stream"):
        if prior[key]!=current_report[key]:
            raise AssertionError(f"Canonical mismatch with R3: {key}")
    previous=prior["schedules"]["bootstrap_simple_aif"]
    new=current_report["schedules"]["A"]
    for key in ("query_indices_1based","queried_normals","queried_defects",
                "first_200_queries","first_400_queries","budget_suppressed",
                "budget_forced_terminal","stm_prototypes"):
        if previous[key]!=new[key]:
            raise AssertionError(f"Variant A diverges from R3 at {key}")
    n=0
    with Path(reference_csv).open("r",encoding="utf-8",newline="") as fh:
        for row in csv.DictReader(fh):
            if row["policy"]!="bootstrap_simple_aif":
                continue
            if n>=len(current_rows):
                raise AssertionError("Reference contains too many Bootstrap rows")
            cur=current_rows[n]
            if cur["policy"]!="A":
                raise AssertionError("Variant A ordered CSV mismatch")
            for field in ("t","path","queried","y_eval_only","query_reason",
                          "budget_reason"):
                old=row[field]
                value=cur[field]
                if field in ("t","queried","y_eval_only"):
                    if int(old)!=int(value):
                        raise AssertionError(f"R3 mismatch at t={cur['t']}, {field}")
                elif old!=str(value):
                    raise AssertionError(f"R3 mismatch at t={cur['t']}, {field}")
            for field in ("stm_score","static_score","frozen_theta","guarded_theta",
                          "balanced_rescue_theta","mistake_feedback_theta"):
                if abs(float(row[field])-float(cur[field]))>1e-7:
                    raise AssertionError(f"R3 numeric mismatch t={cur['t']}, {field}")
            for field in ("static_pred","frozen_pred","guarded_pred",
                          "balanced_rescue_pred","mistake_feedback_pred"):
                if int(row[field])!=int(cur[field]):
                    raise AssertionError(f"R3 prediction mismatch t={cur['t']}, {field}")
            n+=1
    if n!=current_report["n_stream"]:
        raise AssertionError(f"R3 reference has {n} Bootstrap rows; expected "
                             f"{current_report['n_stream']}")
    return {"passed":True,"n_compared":n,
            "source_summary":str(reference_summary),
            "source_csv":str(reference_csv),
            "test":"prequential scores, all four sidecars, queries and predictions"}

def main_run(args):
    import torch
    v22,e02,ext,e01=dependencies()
    if args.math_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    output=Path(args.out_dir)
    output.mkdir(parents=True,exist_ok=True)
    canonical,backbone,base_bank,base_ctrl=e02.init_canonical(args,v22,e01,output)
    static_sha=e01.memory_profile(base_bank,args.class_name)["sha256"]
    stream=e02.select_stream(v22,args)
    args.max_steps_effective=len(stream)
    states={
        name:init_arm(v22,e02,ext,base_bank,canonical,backbone,args,name,
                      static_sha,base_ctrl)
        for name in args.variants.split(",")
    }
    all_rows,labels=[],[]
    for t,sample in enumerate(stream,1):
        image=v22.load_rgb(sample.path)
        patch,g=v22.encode_pil(backbone,image,args.feature_fuse)
        features=[v22.l2_normalize(x) for x in patch]
        # ONE identical encoded observation for both schedules.
        for state in states.values():
            step_state(v22,e02,state,t=t,sample=sample,image=image,
                       features=features,global_feat=g,args=args,
                       evaluation_rows=all_rows)
        # Every variant must preserve the canonical static score/threshold.
        reference=all_rows[-len(states)]
        for recent in all_rows[-len(states):]:
            if (recent["path"] != reference["path"] or
                abs(recent["static_score"]-reference["static_score"])>args.score_tolerance or
                abs(recent["static_theta"]-reference["static_theta"])>1e-12):
                raise AssertionError(f"Canonical fixed-score mismatch at t={t}")
        labels.append(0 if sample.is_good else 1)
        if args.print_every and t%args.print_every==0:
            print(f"[R4] {t}/{len(stream)} queries "+
                  ", ".join(f"{name}={s['budget'].used}" for name,s in states.items()),
                  flush=True)
    reports={name:report_state(e02,ext,e01,s,labels,args) for name,s in states.items()}
    if len(all_rows)!=len(states)*len(stream):
        raise AssertionError("Missing paired trajectory records")
    for name in states:
        report_rows=[r for r in all_rows if r["policy"]==name]
        if len(report_rows)!=len(stream):
            raise AssertionError("Variant row mismatch")
        reports[name]["stability_windows"]=stability_windows(
            report_rows,stride=args.stability_window)
    csv_path=output/"r4_prequential.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0].keys()))
        writer.writeheader();writer.writerows(all_rows)
    report={
        "schema":"e02.r4.abc.query.governance.v1",
        "class_name":args.class_name,"n_stream":len(stream),
        "total_budget_each":len(random_tape),
        "stream_paths_sha256":sha([z.path for z in stream]),
        "support_paths":canonical["support_paths_per_mode"]["0"],
        "canonical_fixed_sha256":static_sha,
        "canonical_theta0":float(base_ctrl.theta()),
        "schedules":reports,
        "parameters":{k:v for k,v in vars(args).items() if k!="max_steps_effective"},
        "scope_notes":[
            "Native v22 Bootstrap and SimpleAIF policy *functions*, with NormalCoverageMemory;",
            "controlled 3% hard cap and deterministic terminal catchup are experimental, not native v22 budget semantics;",
            "policy uses a frozen canonical theta reference to avoid feedback loops with alternative sidecars;",
            "memory uses E02 verified-normal update with novelty=0, not full v22 memory governance;",
            "query schedules lead to DIFFERENT STM and label trajectories: only within-schedule threshold effects are isolated;",
            "same 3% labels per schedule, but not the same labels across schedules;",
            "no correction, LTM, maturation, unlabeled writes, warmup budget, FN-audit or defect audit;",
            "initialization follows E01/E02 baseline_compat=False; not exact submitted historical run;",
            "not a complete SimpleAIF+normal_dyn_plus_correction reproduction;"
        ],
    }
    if args.reference_r3_summary or args.reference_r3_csv:
        if not (args.reference_r3_summary and args.reference_r3_csv):
            raise AssertionError("Provide both R3 summary and CSV for paired replay")
        a_rows=[r for r in all_rows if r["policy"]=="A"]
        report["r3_reference_replay"]=verify_r3_a(
            args.reference_r3_summary,args.reference_r3_csv,
            report,a_rows,args)
        print(f"[R4] A matches original R3 at all {len(a_rows)} steps",flush=True)
    (output/"r4_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False,
                   default=lambda x: x.item() if hasattr(x, "item") else str(x)),
        encoding="utf-8"
    )
    print(json.dumps({k:{ "queries":v["queries"],"first_200":v["first_200_queries"],
                           "queries_normal":v["queried_normals"],"queries_defect":v["queried_defects"],
                           "metrics":{a:{key:m[key] for key in ("auroc","balanced_accuracy","fpr","fnr")}
                                      for a,m in v["arm_metrics"].items()}}
                      for k,v in reports.items()},indent=2),flush=True)
    print("[R4] Saved:",output,flush=True)


def parser():
    # Reuse the exact E02 initialization arguments and defaults.
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/"experiments"))
    import e02_matched_feedback as e02
    p=e02.parser()
    p.description=__doc__
    p.add_argument("--query-cost",type=float,default=.8)
    p.add_argument("--beta-entropy",type=float,default=.3)
    p.add_argument("--local-saliency",action="store_true")
    p.add_argument("--bootstrap-dynamic-target",type=int,default=16)
    p.add_argument("--bootstrap-max-steps",type=int,default=400)
    p.add_argument("--bootstrap-band-min-abs",type=float,default=.02)
    p.add_argument("--mistake-step",type=float,default=.002)
    p.add_argument("--reference-r3-summary",default=None)
    p.add_argument("--reference-r3-csv",default=None)
    p.add_argument("--variants",default="A,B,C",
                   help="Comma separated subset of A,B,C, for lower GPU memory")
    p.add_argument("--stability-window",type=int,default=256)
    p.add_argument("--strata-early-steps",type=int,default=400)
    p.add_argument("--strata-low-fraction",type=float,default=.45)
    p.add_argument("--strata-boundary-fraction",type=float,default=.35)
    p.add_argument("--strata-early-budget-fraction",type=float,default=.40)
    return p


def self_test():
    a=Budget(3815,.03)
    for t in range(1,3816):
        asked,_=a.decide(t,t<=200 and t%4==0)
    a.check()
    assert a.K==114 and a.used==114
    b=Budget(100,.03)
    assert sum(b.decide(i,False)[0] for i in range(1,101))==3
    b.check()
    c=Budget(100,0)
    assert not any(c.decide(i,True)[0] for i in range(1,101))
    c.check()
    c=ScoreRegionBudget(3815,114)
    assert sum(c.quotas.values())==114
    assert c.region(.45,.55,.02)=="low_score"
    assert c.region(.55,.55,.02)=="boundary"
    assert c.region(.60,.55,.02)=="high_score"
    assert longest_gap([2,5,10],10)==4
    assert len(stability_windows([{"t":1,"y_eval_only":0,"queried":0,
       "static_pred":0,"frozen_pred":0,"guarded_pred":0,
       "balanced_rescue_pred":0,"mistake_feedback_pred":0,
       "static_score":.2,"stm_score":.2}]))==1
    print("E02-R4 ABC offline self-test: PASS")


if __name__=="__main__":
    args=parser().parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.dataset_root:
            raise SystemExit("Specify --dataset-root")
        if not args.json_dir:
            args.json_dir=str(Path(args.dataset_root)/"jsons"/"realiad_jsons")
        if args.max_samples<0 or not 0<args.query_rate<1:
            raise SystemExit("Need --max-samples >=0 and 0<--query-rate<1")
        variants=args.variants.split(",")
        if len(set(variants))!=len(variants) or not variants or any(v not in ("A","B","C") for v in variants):
            raise SystemExit("--variants must be comma-separated unique A,B,C")
        if args.stability_window<10:
            raise SystemExit("--stability-window must be >= 10")
        if not (0<=args.strata_low_fraction<=1 and 0<=args.strata_boundary_fraction<=1
                and args.strata_low_fraction+args.strata_boundary_fraction<=1):
            raise SystemExit("Invalid score region quotas")
        if not 0<=args.strata_early_budget_fraction<=1:
            raise SystemExit("Invalid frontloaded budget fraction")
        main_run(args)
