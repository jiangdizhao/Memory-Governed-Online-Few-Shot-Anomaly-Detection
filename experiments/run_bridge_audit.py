#!/usr/bin/env python3
"""One-command historical-to-matched Bridge Audit runner.

DEFAULT six conditions: 3 Real-IAD classes x support seed {0,1} x STM 512.
Paired arms include old-STM-budget and total-prototype-budget flat controls.
Run --all once and re-run same command for validated resume.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT_SCHEMA="bridge.historical_matched.v1"
ARMS=(
    "flat_stm","flat_utility","flat_equal_total",
    "track_utility","shadow_utility","ltm_utility",
    "full_utility","full_no_utility",
)

def sha(values):
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode("utf-8")).hexdigest()

def conditions(a):
    cs=[s.strip() for s in a.classes.split(",") if s.strip()]
    seeds=[int(s) for s in a.seeds.split(",") if s.strip()]
    stm=[int(s) for s in a.stm_capacities.split(",") if s.strip()]
    orders=[s.strip() for s in a.orders.split(",") if s.strip()]
    for values in (cs,seeds,stm,orders):
        if not values or len(values)!=len(set(values)):
            raise ValueError("Empty or duplicated matrix dimension")
    if any(v not in (512,768) for v in stm):
        raise ValueError("Only original study's STM=512/768 is supported")
    if any(v not in ("native","normal_first","defect_first") for v in orders):
        raise ValueError("Unsupported stream order")
    return [(c,seed,k,order) for c in cs for seed in seeds
            for k in stm for order in orders]

def out_path(a,c,seed,k,order,smoke=False):
    stem=f"{c}_s{seed}_STM{k}_{order}_"+("smoke" if smoke else "full")
    return Path(a.out_root)/stem

def command(a,c,seed,k,order,smoke=False):
    cmd=[sys.executable,str(ROOT/"experiments"/"bridge_historical_matched.py"),
         "--dataset-root",a.dataset_root,
         "--class-name",c,"--support-seed",str(seed),
         "--stm-capacity",str(k),"--stream-order",order,
         "--evidence-mode",a.evidence_mode,
         "--probe-count",str(a.probe_count),
         "--probe-every",str(a.probe_every),
         "--max-samples","64" if smoke else "0",
         "--out-dir",str(out_path(a,c,seed,k,order,smoke))]
    if a.json_dir:cmd+=["--json-dir",a.json_dir]
    if a.math_sdpa:cmd.append("--math-sdpa")
    return cmd

def check(a,c,seed,k,order,smoke=False):
    directory=out_path(a,c,seed,k,order,smoke)
    report_file=directory/"summary.json"
    if not report_file.is_file():
        raise RuntimeError(f"Incomplete condition; do not overwrite: {directory}")
    j=json.loads(report_file.read_text(encoding="utf-8"))
    if (j.get("schema")!=OUT_SCHEMA or j.get("complete") is not True
        or (j.get("class_name"),j.get("support_seed"),j.get("stm_capacity"),
            j.get("stream_order"))!=(c,seed,k,order)):
        raise RuntimeError("Unexpected condition metadata")
    if (j["n_probe"]!=a.probe_count
        or j["protocol"]["evidence_mode"]!=a.evidence_mode
        or j["ltm_capacity"]!=512 or j["buffer_capacity"]!=4096):
        raise RuntimeError("Different probe/evidence/capacity protocol")
    for src,digest in j.get("source_sha256",{}).items():
        file=ROOT/src
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"Source changed since dataset run: {src}")
    if not j.get("source_sha256"):
        raise RuntimeError("Missing exact code provenance")
    p=j["protocol"]
    if (not p["four_shot_strict"] or len(set(p["supports"]))!=4
        or sha(p["supports"])!=p["support_hash"]
        or len(p["loso_scores"])!=4 or len(p["loso_folds"])!=4):
        raise RuntimeError("Support/LOSO fairness check failed")
    for fold in p["loso_folds"]:
        if fold["withheld"] in fold["training"] or (
            set(fold["training"])!=set(p["supports"])-{fold["withheld"]}):
            raise RuntimeError("LOSO self-match contamination")
    if (sha(p["query_positions"])!=p["query_hash"]
        or len(p["query_positions"])!=j["n_queries"]
        or j["queried_normal"]+j["queried_defect"]!=j["n_queries"]):
        raise RuntimeError("Query tape mismatch")
    if sha(p["probe_paths"])!=p["probe_hash"] or not p["probe_evaluation_only"]:
        raise RuntimeError("Probe audit failed")
    if not p["same_feedback_and_candidate_identity"] or not p["historical_native_retrieval"]:
        raise RuntimeError("Incomplete mechanism controls")
    if set(j["arms"])!=set(ARMS):
        raise RuntimeError("Not all memory-factor controls completed")
    for name,values in j["arms"].items():
        m=values["memory"]
        expected=k+512+4096 if name in (
            "flat_equal_total","track_utility","shadow_utility","ltm_utility",
            "full_utility","full_no_utility") else k
        if (m["capacity_total"]!=expected or m["occupied_total"]>expected
            or m["candidate_events"]!=j["queried_normal"]):
            raise RuntimeError(f"Unmatched feedback/capacity: {name}")
    if len({j["arms"][name]["memory"]["candidate_patches"]
            for name in ARMS})!=1:
        raise RuntimeError("Candidate mismatch across arms")
    if smoke and j["n_stream"]!=64:raise RuntimeError("Smoke size mismatch")
    if not smoke and j["n_stream"]<=64:raise RuntimeError("Full run too short")
    with (directory/"prequential.csv").open(encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    if (len(rows)!=j["n_stream"]
        or sha([r["path"] for r in rows])!=p["stream_hash"]
        or set(r["path"] for r in rows)&set(p["probe_paths"])):
        raise RuntimeError("Stream/probe contamination or manifest mismatch")
    observed=[int(r["t"]) for r in rows if int(r["queried"])]
    if observed!=p["query_positions"]:
        raise RuntimeError("Query positions and per-step rows disagree")
    for file in ("probe_scores.csv","probe_summary.csv",
                 "memory_snapshots.csv","retrieval_counterfactual.csv"):
        if not (directory/file).is_file():
            raise RuntimeError(f"Missing evidence file: {file}")
    final_cf=j["counterfactual_probe_final"]
    if {row["variant"] for row in final_cf}!=set(
        ("full","no_ltm","no_buffer","stm_only")):
        raise RuntimeError("Missing same-state retrieval counterfactuals")
    if not (abs(j["arms"]["shadow_utility"]["probe_aulc"]-
                j["arms"]["ltm_utility"]["probe_aulc"])<=1 or
            j["arms"]["ltm_utility"]["memory"]["ltm"]==
                j["arms"]["shadow_utility"]["memory"]["ltm"]):
        raise RuntimeError("Shadow/LTM state divergence")
    return {"class_name":c,"support_seed":seed,"stm_capacity":k,
            "stream_order":order,"n_stream":j["n_stream"],
            "normal_queries":j["queried_normal"],
            "defect_queries":j["queried_defect"],
            "result_path":str(directory),
            "arms":{name:{
                "auroc":v["auroc"],"probe_aulc":v["probe_aulc"],
                "probe_final_auroc":v["probe_final_auroc"],
                "memory":v["memory"]} for name,v in j["arms"].items()},
            "counterfactual_probe_final":final_cf}

def launch(cmd,log):
    if log.exists():
        raise FileExistsError(f"Existing log, inspect before retry: {log}")
    log.parent.mkdir(parents=True,exist_ok=True)
    print("[BRIDGE]"," ".join(cmd),flush=True)
    with log.open("w",encoding="utf-8") as f:
        p=subprocess.Popen(
            cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            encoding="utf-8",errors="replace",text=True,
            env={**os.environ,"PYTHONIOENCODING":"utf-8"})
        for line in p.stdout:
            f.write(line);f.flush()
            encoding=sys.stdout.encoding or "utf-8"
            printable=line.encode(encoding,errors="backslashreplace").decode(
                encoding,errors="replace")
            print(printable,end="",flush=True)
        rc=p.wait()
        f.write(f"\nExit code: {rc}\n")
    if rc:
        raise RuntimeError(f"Bridge subprocess exited {rc}; see {log}")

def run_condition(a,condition,smoke=False):
    path=out_path(a,*condition,smoke)
    if path.exists():
        print(f"[BRIDGE] Revalidate completed condition: {path}",flush=True)
        return check(a,*condition,smoke=smoke)
    launch(command(a,*condition,smoke),path.with_suffix(".run.log"))
    return check(a,*condition,smoke=smoke)

def save_aggregate(a,rows,complete):
    root=Path(a.out_root);root.mkdir(parents=True,exist_ok=True)
    report={"schema":"bridge.aggregate.v1","complete":complete,
            "expected":len(conditions(a)),"validated":len(rows),
            "conditions":rows,"arms":list(ARMS)}
    (root/"aggregate.json").write_text(
        json.dumps(report,indent=2,allow_nan=False,ensure_ascii=False),
        encoding="utf-8")
    with (root/"aggregate.csv").open("w",encoding="utf-8",newline="") as f:
        fields=("class_name","support_seed","stm_capacity","stream_order",
                "method","auroc","probe_aulc","probe_final_auroc",
                "stm","ltm","buffer","capacity_total",
                "consolidation_successes","allocated_feature_bytes",
                "allocated_metadata_bytes")
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for condition in rows:
            for name,v in condition["arms"].items():
                w.writerow({
                    "class_name":condition["class_name"],
                    "support_seed":condition["support_seed"],
                    "stm_capacity":condition["stm_capacity"],
                    "stream_order":condition["stream_order"],
                    "method":name,"auroc":v["auroc"],
                    "probe_aulc":v["probe_aulc"],
                    "probe_final_auroc":v["probe_final_auroc"],
                    **{field:v["memory"][field] for field in (
                        "stm","ltm","buffer","capacity_total",
                        "consolidation_successes","allocated_feature_bytes",
                        "allocated_metadata_bytes")},
                })
    print(f"[BRIDGE] {len(rows)}/{len(conditions(a))} validated",flush=True)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",action="store_true")
    group.add_argument("--validate",action="store_true")
    group.add_argument("--self-test",action="store_true")
    p.add_argument("--dataset-root",default="G:/dataset/realiad")
    p.add_argument("--json-dir",default=None)
    p.add_argument("--classes",default="audiojack,rolled_strip_base,switch")
    p.add_argument("--seeds",default="0,1")
    p.add_argument("--stm-capacities",default="512")
    p.add_argument("--orders",default="native")
    p.add_argument("--evidence-mode",choices=("neutral","loso_margin"),default="neutral")
    p.add_argument("--probe-count",type=int,default=96)
    p.add_argument("--probe-every",type=int,default=1024)
    p.add_argument("--out-root",default="outputs/bridge_historical_matched")
    p.add_argument("--math-sdpa",action="store_true")
    p.add_argument("--no-smoke",action="store_true")
    return p

if __name__=="__main__":
    a=parser().parse_args()
    if a.self_test:
        assert len(conditions(a))==6
        assert len(conditions(parser().parse_args(
            ["--all","--stm-capacities","512,768"])))==12
        assert "--max-samples" in command(a,"audiojack",0,512,"native",True)
        print("[BRIDGE] Offline runner checks PASS")
    else:
        conditions_to_run=conditions(a)
        if a.all:
            # Run actual native v22 path, fail before expensive GPU work.
            preflight=Path(a.out_root)/"cpu_mechanism_preflight.run.log"
            marker=Path(a.out_root)/"cpu_mechanism_preflight.passed"
            if not marker.is_file():
                launch([sys.executable,str(ROOT/"experiments"/"bridge_mechanism_test.py")],
                       preflight)
                marker.touch()
            if not a.no_smoke:run_condition(a,conditions_to_run[0],smoke=True)
        done=[]
        for condition in conditions_to_run:
            row=check(a,*condition) if a.validate else run_condition(a,condition)
            done.append(row)
            save_aggregate(a,done,False)
        save_aggregate(a,done,True)
