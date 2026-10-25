#!/usr/bin/env python3
"""E02-R5/D: temporal pacing ONLY, compared against R4 B and C.

D = B's native Bootstrap->SimpleAIF, Bootstrap-excluded adaptive query
cost, supervised STM, and all four Sidecars, PLUS C's exact allowed_total(t)
time budget. Unlike C: NO region quotas, low-score priority, or
region-deficit querying. Current score region is diagnostic only.

Historical R4 B/C outputs are required and compared (no need to rerun them).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))
import e02_r4_abc_query_governance as r4


class TimeOnlyPacing:
    """C's time allowance, no dependence on score region or current label."""
    def __init__(self,n,k,*,early_steps=400,early_fraction=.40):
        if n<1 or k<0 or k>n or not 0<=early_fraction<=1:
            raise ValueError("Invalid D pacing parameters")
        # Reuse C's precise rounding and allowance, not an approximation.
        self.timeline=r4.ScoreRegionBudget(
            n,k,early_steps=early_steps,early_total_fraction=early_fraction)
        self.blocked=0
        self.queried=0

    def allowed_total(self,t):
        return self.timeline.allowed_total(t)

    def propose(self,t,region,native,global_budget,*,
                target_normals,verified_normal_count):
        # Adapter API only: neither score-region nor known-class counts
        # are consulted. They cannot influence D's query decision.
        del region,target_normals,verified_normal_count
        if global_budget.used>=self.allowed_total(t):
            self.blocked+=int(bool(native))
            return False,"time_only_pacing_cap"
        return bool(native),"native_policy"

    def observe(self,region,asked):
        del region
        self.queried+=int(bool(asked))

    def snapshot(self):
        return {
            "mode":"time_only",
            "early_steps":self.timeline.early_steps,
            "early_total_cap":self.timeline.early_total_cap,
            "blocked_pacing":self.blocked,
            "queried":self.queried,
            "score_quota_enforced":False,
            "low_score_priority_enabled":False,
            "stratum_deficit_enabled":False,
        }


def make_d(v22,e02,ext,e01,bank,canonical,backbone,args,sha,theta):
    d=r4.init_arm(v22,e02,ext,bank,canonical,backbone,args,"D",sha,theta)
    # R4's B/C uses no native observe() adaptation but calls
    # record_query_cost() after Bootstrap using post-Bootstrap flags.
    d["policy_cfg"].adapt_query_cost=False
    d["aif"].cfg.adapt_query_cost=False
    if d["aif"].cfg is not d["policy_cfg"]:
        raise AssertionError("Expected shared SimpleAIFConfig")
    d["strata"]=TimeOnlyPacing(args.max_steps_effective,d["budget"].K,
                                early_steps=args.strata_early_steps,
                                early_fraction=args.strata_early_budget_fraction)
    return d


REFERENCE_KEYS=(
    "support_seed","legacy_seed","stream_seed","query_seed","query_rate",
    "max_samples","fixed_capacity","fixed_per_image","augmentations",
    "include_identity","feature_fuse","image_aggregation","topk_patches",
    "knn","math_sdpa","score_tolerance","query_cost","beta_entropy",
    "local_saliency","bootstrap_dynamic_target","bootstrap_max_steps",
    "bootstrap_band_min_abs","mistake_step","stability_window",
    "strata_early_steps","strata_early_budget_fraction",
)
MATCH_FIELDS=(
    "n_stream","total_budget_each","stream_paths_sha256","support_paths",
    "canonical_fixed_sha256","canonical_theta0",
)


def read_r4_reference(args):
    if not args.reference_r4_summary:
        raise ValueError("D needs --reference-r4-summary for prior R4 B/C")
    path=Path(args.reference_r4_summary)
    old=json.loads(path.read_text(encoding="utf8"))
    if old.get("schema")!="e02.r4.abc.query.governance.v1":
        raise ValueError("Not an R4 A/B/C summary")
    if not {"B","C"}.issubset(old.get("schedules",{})):
        raise ValueError("Reference must contain both B and C")
    if old["class_name"]!=args.class_name:
        raise ValueError("Reference class mismatch")
    for key in REFERENCE_KEYS:
        expected=old["parameters"][key]
        actual=getattr(args,key)
        if expected!=actual:
            raise ValueError(
                f"Cannot compare R5/D with R4: {key}: "
                f"R4={expected!r}, R5={actual!r}")
    return old


def check_r4_match(old,new):
    for key in MATCH_FIELDS:
        if old[key]!=new[key]:
            raise AssertionError(f"R5 vs R4 canonical mismatch: {key}")
    for variant in ("B","C"):
        s=old["schedules"][variant]
        if s["queries"]!=new["total_budget_each"]:
            raise AssertionError(f"R4 {variant}: unequal label budget")
        if s["fixed_sha256"]!=new["canonical_fixed_sha256"]:
            raise AssertionError(f"R4 {variant}: Fixed Bank mismatch")


def compare(old,new):
    result={}
    for name,s in (
        ("B",old["schedules"]["B"]),
        ("D",new["schedules"]["D"]),
        ("C",old["schedules"]["C"])
    ):
        m=s["arm_metrics"]
        windows=s["stability_windows"]
        result[name]={
            "queries":s["queries"],"queried_normals":s["queried_normals"],
            "queried_defects":s["queried_defects"],
            "first_200_queries":s["first_200_queries"],
            "first_400_queries":s["first_400_queries"],
            "longest_gap":s["longest_query_free_gap"],
            "terminal_catchup":s["budget_forced_terminal"],
            "stm_prototypes":s["stm_prototypes"],
            "stm_auroc":m["frozen"]["auroc"],
            "balanced_ba":m["balanced_rescue"]["balanced_accuracy"],
            "balanced_fpr":m["balanced_rescue"]["fpr"],
            "balanced_fnr":m["balanced_rescue"]["fnr"],
            "balanced_moves":len(s["sidecar_moves"]["balanced_rescue"]),
            "zero_query_windows":sum(w["queried"]==0 for w in windows),
            "worst_window_ba":min(
                w["balanced_rescue"]["balanced_accuracy"] for w in windows
                if w["balanced_rescue"]["balanced_accuracy"] is not None),
        }
    for k in ("B","C"):
        result[f"D_minus_{k}"]={
            metric:result["D"][metric]-result[k][metric]
            for metric in ("stm_auroc","balanced_ba","balanced_fpr","balanced_fnr")
        }
    return result


def main_run(args):
    # Fail early for invalid R4 comparison rather than after a GPU job.
    old=read_r4_reference(args)
    import torch
    v22,e02,ext,e01=r4.dependencies()
    if args.math_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    canonical,backbone,bank,theta=e02.init_canonical(args,v22,e01,out)
    initial_sha=e01.memory_profile(bank,args.class_name)["sha256"]
    stream=e02.select_stream(v22,args)
    args.max_steps_effective=len(stream)
    d=make_d(v22,e02,ext,e01,bank,canonical,backbone,args,initial_sha,theta)
    rows=[]
    truth=[]
    for t,sample in enumerate(stream,1):
        img=v22.load_rgb(sample.path)
        patch,glob=v22.encode_pil(backbone,img,args.feature_fuse)
        feats=[v22.l2_normalize(x) for x in patch]
        # Reuse R4: all predictions and query decisions PRECEDE label read.
        r4.step_state(v22,e02,d,t=t,sample=sample,image=img,features=feats,
                      global_feat=glob,args=args,evaluation_rows=rows)
        truth.append(0 if sample.is_good else 1)
        if args.print_every and t%args.print_every==0:
            print(f"[R5/D] {t}/{len(stream)} queries {d['budget'].used}/"
                  f"{d['budget'].K}, cost {d['aif'].query_cost_runtime:.4f}",
                  flush=True)
    if len(rows)!=len(stream):
        raise AssertionError("D emitted wrong number of image rows")
    summary=r4.report_state(e02,ext,e01,d,truth,args)
    summary["stability_windows"]=r4.stability_windows(
        rows,stride=args.stability_window)
    summary["pacing_schedule"]={
        "first_400_cap":d["strata"].allowed_total(min(400,len(stream))),
        "allowed_at_steps":{
            str(t):d["strata"].allowed_total(t)
            for t in (1,100,200,400,600,1000,1600,2000,3000,len(stream))
            if t<=len(stream)
        }
    }
    report={
        "schema":"e02.r5.d.time_only.v1",
        "class_name":args.class_name,"n_stream":len(stream),
        "total_budget_each":r4.validated_equal_budget({"D":d}),
        "stream_paths_sha256":r4.sha([x.path for x in stream]),
        "support_paths":canonical["support_paths_per_mode"]["0"],
        "canonical_fixed_sha256":initial_sha,
        "canonical_theta0":float(theta.theta()),
        "schedules":{"D":summary},
        "parameters":{k:v for k,v in vars(args).items()
                      if k!="max_steps_effective"},
        "scope_notes":[
            "D=R4 B plus only R4 C cumulative temporal pacing.",
            "No low-score priority, no regional quotas, no region-deficit queries.",
            "R4 B/C were not rerun: comparison uses archived original results.",
            "Score region logged by R4 but never affects D query decisions.",
            "Global K-label cap and label-blind terminal catchup retained.",
            "No full v22 Signed Correction, LTM or maturation.",
        ],
    }
    check_r4_match(old,report)
    report["r4_reference_check"]={
        "passed":True,"source":str(args.reference_r4_summary),
        "matched_fields":MATCH_FIELDS}
    report["comparison"]=compare(old,report)
    # Validate and compare BEFORE emitting files; no stale variables.
    with (out/"r5_d_prequential.csv").open(
        "w",encoding="utf8",newline=""
    ) as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out/"r5_d_summary.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False,
                   default=lambda x:x.item() if hasattr(x,"item") else str(x)),
        encoding="utf8"
    )
    print(json.dumps({
        "reference_verified":True,
        "D_queries":summary["queries"],
        "D_first_200":summary["first_200_queries"],
        "D_longest_gap":summary["longest_query_free_gap"],
        "B_D_C":report["comparison"]
    },indent=2),flush=True)
    print("[R5/D] Saved:",out,flush=True)


def parser():
    p=r4.parser()
    p.description=__doc__
    p.set_defaults(variants="D",
                   out_dir="outputs/e02_r5_d_audiojack_full_seed0")
    p.add_argument("--reference-r4-summary",default=None,
                   help="Required prior R4 A/B/C r4_summary.json")
    return p


def self_test():
    for n,k in ((100,3),(3815,114)):
        budget=r4.Budget(n,.03)
        d=TimeOnlyPacing(n,budget.K,early_steps=400,early_fraction=.40)
        assert d.timeline.early_total_cap==(1 if n==100 else 46)
        if n==3815:
            assert d.allowed_total(400)==46
            assert d.allowed_total(3815)==114
        for t in range(1,n+1):
            prop,_=d.propose(t,"boundary",True,budget,
                target_normals=16,verified_normal_count=0)
            asked,_=budget.decide(t,prop)
            d.observe("boundary",asked)
        budget.check()
        assert budget.used==k
    print("E02-R5/D time-only pacing offline self-test: PASS")


if __name__=="__main__":
    args=parser().parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.dataset_root:
            raise SystemExit("--dataset-root required")
        if not args.json_dir:
            args.json_dir=str(Path(args.dataset_root)/"jsons"/"realiad_jsons")
        if args.max_samples<0 or not 0<args.query_rate<1:
            raise SystemExit("Bad sample count or query rate")
        if args.variants!="D":
            raise SystemExit("R5 D runner only accepts --variants D")
        if args.stability_window<10:
            raise SystemExit("stability_window must be >=10")
        if args.strata_early_steps<1 or not 0<=args.strata_early_budget_fraction<=1:
            raise SystemExit("Invalid temporal pacing parameters")
        main_run(args)
