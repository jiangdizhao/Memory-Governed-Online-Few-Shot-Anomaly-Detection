#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E02-R3: controlled native-SimpleAIF-bootstrap-vs-random query and sidecar audit.

Uses UNMODIFIED v22 cold start (through E02), its native
choose_simple_aif_bootstrap_action -> select_simple_aif_action primitives, and
NormalCoverageMemory. A hard final label cap fixes both policies to K labels.
Within EACH query schedule, frozen/guarded/balanced/mistake theta strategies
receive precisely the same scores and queried labels. Different query schedules
are different online-learning trajectories: do NOT ascribe their gap solely to
the sidecar. Intentional E02 mechanism isolation: no correction, LTM,
maturation, self-training, FN-audit, or warmup-budget override.
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
    state = {
        "name": policy_name,
        "static": static_bank,
        "arm": arm,
        "controller_ref": policy_ref,  # frozen for policy, independent of sidecars
        "controllers": controllers,
        "policy_cfg": conf, "aif": v22.SimpleAIFState(conf),
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
               random_tape: set[int], evaluation_rows: list[dict]):
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
    if state["name"] == "random":
        proposed = t in random_tape
        reason = "seeded_uniform_random"
    else:
        proposed, reason = policy_decision(
            v22, state, t=t, score=score, theta=theta0,
            sigma=sigma, p=p, amap=amap, global_feat=global_feat)
    asked, budget_reason = state["budget"].decide(t, proposed)
    # Record EVERY prediction, for ALL threshold candidates, before reading label.
    row = {
        "t": t, "path": sample.path, "policy": state["name"],
        "queried": int(asked), "query_reason": reason,
        "budget_reason": budget_reason,
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
    state["aif"].observe(asked)
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
    }


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
    random_tape=e02.make_query_tape(len(stream),args.query_rate,args.query_seed)
    states={
        name:init_arm(v22,e02,ext,base_bank,canonical,backbone,args,name,
                      static_sha,base_ctrl)
        for name in ("random","bootstrap_simple_aif")
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
                       random_tape=random_tape,evaluation_rows=all_rows)
        labels.append(0 if sample.is_good else 1)
        if args.print_every and t%args.print_every==0:
            print(f"[R3] {t}/{len(stream)} queries random={states['random']['budget'].used} "
                  f"bootstrap={states['bootstrap_simple_aif']['budget'].used}",flush=True)
    reports={name:report_state(e02,ext,e01,s,labels,args) for name,s in states.items()}
    if reports["random"]["query_indices_1based"]!=sorted(random_tape):
        raise AssertionError("Random reference tape was not preserved")
    if reports["random"]["arm_metrics"]["frozen"]["auroc"] == \
       reports["bootstrap_simple_aif"]["arm_metrics"]["frozen"]["auroc"]:
        # Equality is possible by chance; do not assert a difference.
        pass
    if len(all_rows)!=2*len(stream):
        raise AssertionError("Missing paired trajectory records")
    csv_path=output/"r3_prequential.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0].keys()))
        writer.writeheader();writer.writerows(all_rows)
    report={
        "schema":"e02.r3.bootstrap.sidecar.audit.v1",
        "class_name":args.class_name,"n_stream":len(stream),
        "total_budget_each":len(random_tape),
        "stream_paths_sha256":sha([z.path for z in stream]),
        "support_paths":canonical["support_paths_per_mode"]["0"],
        "canonical_fixed_sha256":static_sha,
        "canonical_theta0":float(base_ctrl.theta()),
        "random_tape_sha256":sha(sorted(random_tape)),
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
    (output/"r3_summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),
                                            encoding="utf-8")
    print(json.dumps({k:{ "queries":v["queries"],"first_200":v["first_200_queries"],
                           "queries_normal":v["queried_normals"],"queries_defect":v["queried_defects"],
                           "metrics":{a:{key:m[key] for key in ("auroc","balanced_accuracy","fpr","fnr")}
                                      for a,m in v["arm_metrics"].items()}}
                      for k,v in reports.items()},indent=2),flush=True)
    print("[R3] Saved:",output,flush=True)


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
    print("E02-R3 offline self-test: PASS")


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
        main_run(args)
