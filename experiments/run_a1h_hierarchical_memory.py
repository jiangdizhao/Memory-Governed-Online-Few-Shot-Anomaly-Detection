#!/usr/bin/env python3
"""Automate A1-H mechanism preflight, smoke, 12 GPU runs, validation, aggregation.

Single user command:
 python experiments/run_a1h_hierarchical_memory.py --all --dataset-root G:/dataset/realiad

No overwriting finished or partial condition directories. Re-run --all for resume.
Default matrix: 3 classes * two support seeds * (128,256) * Native = 12.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SCHEMA="a1h.strict4.hierarchical.attribution.v1"
ARMS=("static","reservoir","online_greedy_coreset","v22_normal_stm",
      "H1_track","H2_shadow_promotion","H3_ltm_retrieval","H4_utility")


def sha(items):
    return hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()


def conditions(a):
    classes=[x.strip() for x in a.classes.split(",") if x.strip()]
    seeds=[int(x) for x in a.seeds.split(",") if x.strip()]
    budgets=[int(x) for x in a.budgets.split(",") if x.strip()]
    orders=[x.strip() for x in a.orders.split(",") if x.strip()]
    if (not classes or not seeds or not budgets or not orders
            or any(len(set(x))!=len(x) for x in (classes,seeds,budgets,orders))
            or any(k not in (128,256) for k in budgets)
            or any(o not in ("native","normal_first","defect_first") for o in orders)):
        raise ValueError("Invalid or duplicate matrix dimensions")
    return [(c,s,b,o) for c in classes for s in seeds for b in budgets for o in orders]


def output_dir(a,c,s,b,o,smoke=False):
    return Path(a.out_root)/f"{c}_s{s}_K{b}_{o}_{'smoke' if smoke else 'full'}"


def command(a,c,s,b,o,smoke=False):
    args=[sys.executable,str(ROOT/"experiments"/"a1h_hierarchical_matched.py"),
          "--dataset-root",a.dataset_root,"--class-name",c,
          "--support-seed",str(s),"--budget",str(b),"--stream-order",o,
          "--probe-count",str(a.probe_count),
          "--probe-every",str(a.probe_every),
          "--max-samples","64" if smoke else "0",
          "--out-dir",str(output_dir(a,c,s,b,o,smoke))]
    if a.json_dir:args+=["--json-dir",a.json_dir]
    if a.math_sdpa:args+=["--math-sdpa"]
    return args


def check(a,c,s,b,o,smoke=False):
    path=output_dir(a,c,s,b,o,smoke)
    report=path/"summary.json"
    if not report.is_file():
        raise ValueError(f"Missing summary; incomplete run must be inspected: {path}")
    j=json.loads(report.read_text(encoding="utf-8"))
    if (j.get("schema")!=SCHEMA or not j.get("complete") or
        (j["class_name"],j["support_seed"],j["budget"],j["order"])!=(c,s,b,o)):
        raise ValueError(f"Schema/condition mismatch: {path}")
    # Prevent reusing a previous experiment after the memory scorer or
    # baseline code changed, even when class/seed/budget filenames match.
    if not j.get("source_sha256"):
        raise ValueError("Missing source provenance fingerprints")
    for rel,digest in j["source_sha256"].items():
        src=ROOT/rel
        if not src.is_file() or hashlib.sha256(src.read_bytes()).hexdigest()!=digest:
            raise ValueError(f"Code changed since condition was run: {rel}")
    if j["n_probe"]!=a.probe_count:
        raise ValueError("Probe configuration changed")
    p=j["protocol"]
    if (not p["strict_four_shot"] or p["support_access"]!=4
        or len(set(p["support_paths"]))!=4
        or sha(p["support_paths"])!=p["support_hash"]
        or len(p["loso_folds"])!=4 or len(p["loso_scores"])!=4):
        raise ValueError("Strict-four-shot/LOSO manifest failed")
    for f in p["loso_folds"]:
        if (f["heldout"] in f["training"] or len(f["training"])!=3
            or set(f["training"])!=set(p["support_paths"])-{f["heldout"]}):
            raise ValueError("Self match in LOSO fold")
    if (not p["probes_never_used_for_training"] or not p["labels_after_predictions"]
        or not p["candidate_from_shared_static"]
        or p["uniform_retrieval"]!="global_top5_fixed_plus_stm_plus_optional_ltm"):
        raise ValueError("Missing fairness controls")
    if p["hierarchical_allocation"]!={"stm":b//2,"ltm":b//4,"buffer":b//4}:
        raise ValueError("Hierarchy budget mismatch")
    if set(j["arms"])!=set(ARMS):
        raise ValueError("Missing arms")
    if j["n_queries"]!=round(.03*j["n_stream"]) or (
       j["queried_normal"]+j["queried_defect"]!=j["n_queries"]):
        raise ValueError("3% tape/feedback mismatch")
    if sha(p["query_positions"])!=p["query_hash"] or (
       len(p["query_positions"])!=j["n_queries"]):
        raise ValueError("Query tape fingerprint mismatch")
    if len(set(p["query_positions"]))!=j["n_queries"]:
        raise ValueError("Repeated query")
    if len(set(p["probe_paths"]))!=j["n_probe"] or (
       sha(p["probe_paths"])!=p["probe_hash"]):
        raise ValueError("Probe fingerprint mismatch")
    if smoke and j["n_stream"]!=64:
        raise ValueError("Smoke must have 64 stream samples")
    if not smoke and j["n_stream"]<=64:
        raise ValueError("Full must have more than 64 stream samples")
    if not j.get("ltm_activation"):
        raise ValueError("No actual LTM activity audit")
    for name,v in j["arms"].items():
        mem=v["memory"]
        if mem["capacity_total"]!=b or mem["occupied_total"]>b:
            raise ValueError(f"Capacity mismatch/overflow for {name}")
        if name!="static" and mem["candidate_events"]!=j["queried_normal"]:
            raise ValueError(f"Candidate feedback mismatch: {name}")
    if j["arms"]["static"]["memory"]["occupied_total"]!=0:
        raise ValueError("Static arm acquired memory")
    candidates={j["arms"][name]["memory"]["candidate_patches"]
                for name in ARMS if name!="static"}
    if len(candidates)!=1:
        raise ValueError("Arms saw different candidate patches")
    with (path/"prequential.csv").open(encoding="utf-8",newline="") as f:
        rs=list(csv.DictReader(f))
    if len(rs)!=j["n_stream"]:
        raise ValueError("Prequential length mismatch")
    if sha([r["path"] for r in rs])!=p["stream_hash"]:
        raise ValueError("Stream fingerprint mismatch")
    if {r["path"] for r in rs}&set(p["probe_paths"]):
        raise ValueError("Probe in online stream")
    if [int(r["t"]) for r in rs if int(r["queried"])]!=p["query_positions"]:
        raise ValueError("Query positions differ from per-step evidence")
    if sum(int(r["queried"]) for r in rs if int(r["label_eval_only"])==0)!=j["queried_normal"]:
        raise ValueError("Queried-normal labels disagree with rows")
    if sum(int(r["queried"]) for r in rs if int(r["label_eval_only"])==1)!=j["queried_defect"]:
        raise ValueError("Queried-defect labels disagree with rows")
    for artifact in ("probe_scores.csv","probe_summary.csv","memory_snapshots.csv"):
        if not (path/artifact).is_file():
            raise ValueError(f"Missing artifact {artifact}")
    return {"class_name":c,"seed":s,"budget":b,"order":o,
            "ltm_activation":j["ltm_activation"],
            "n_stream":j["n_stream"],"normal_queries":j["queried_normal"],
            "defect_queries":j["queried_defect"],
            "result_path":str(path),
            "arms":{name:{k:j["arms"][name][k] for k in (
                "auroc","balanced_accuracy","probe_aulc",
                "probe_final_auc","probe_worst_auc")}
                |{"memory":j["arms"][name]["memory"]} for name in ARMS}}


def subprocess_logged(cmd, logpath):
    if logpath.exists():
        raise FileExistsError(f"Previous incomplete log present: {logpath}")
    logpath.parent.mkdir(parents=True,exist_ok=True)
    print("[A1-H]", " ".join(cmd),flush=True)
    with logpath.open("w",encoding="utf-8") as out:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,errors="replace")
        for line in proc.stdout:
            out.write(line);out.flush()
            print(line,end="",flush=True)
        rc=proc.wait()
        out.write(f"\nExit code: {rc}\n")
    if rc:
        raise RuntimeError(f"Failed with code {rc}; see {logpath}")


def run_case(a,c,s,b,o,smoke=False):
    path=output_dir(a,c,s,b,o,smoke)
    if path.exists():
        print(f"[A1-H] Validating existing result: {path}",flush=True)
        return check(a,c,s,b,o,smoke)
    subprocess_logged(command(a,c,s,b,o,smoke),path.parent/(path.name+".run.log"))
    return check(a,c,s,b,o,smoke)


def write_aggregate(a,rows,complete):
    out=Path(a.out_root);out.mkdir(parents=True,exist_ok=True)
    status={"schema":"a1h.aggregate.v1","complete":complete,
            "expected":len(conditions(a)),"validated":len(rows),"conditions":rows,
            "method_count":len(ARMS),
            "notes":[
                "Only memory submodules, not full v22 controller/correction.",
                "LOSO threshold is diagnostic only. Primary endpoints are probe AULC and AUROC.",
                "Same total prototype upper bound; metadata/time costs differ.",
                "Native ordering first; no claimed real A-B-A drift protocol.",
                "Probe never used for memory/query/calibration.",
                "Hierarchy is not scientifically evaluated unless actual LTM promotion/retrieval effects are observed.",
            ]}
    (out/"aggregate.json").write_text(json.dumps(
        status,indent=2,allow_nan=False,ensure_ascii=False,default=str),encoding="utf-8")
    with (out/"aggregate.csv").open("w",newline="",encoding="utf-8") as f:
        keys=("class_name","seed","budget","order","method","auroc",
              "balanced_accuracy","probe_aulc","probe_final_auc",
              "probe_worst_auc","stm","ltm","buffer",
              "promotion_attempts","promotion_successes")
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for row in rows:
            for name,v in row["arms"].items():
                writer.writerow({"class_name":row["class_name"],"seed":row["seed"],
                    "budget":row["budget"],"order":row["order"],"method":name,
                    **{k:v[k] for k in ("auroc","balanced_accuracy","probe_aulc",
                                        "probe_final_auc","probe_worst_auc")},
                    **{k:v["memory"].get(k) for k in ("stm","ltm","buffer",
                                    "promotion_attempts","promotion_successes")}})
    print(f"[A1-H] {len(rows)}/{status['expected']} validated; {out/'aggregate.json'}",flush=True)


def self_test():
    a=parser().parse_args(["--all"])
    assert len(conditions(a))==12
    assert len(conditions(parser().parse_args(["--all","--orders",
                             "native,normal_first,defect_first"])))==36
    assert output_dir(a,"switch",1,256,"native").name=="switch_s1_K256_native_full"
    assert "--max-samples" in command(a,"audiojack",0,128,"native",smoke=True)
    assert sha([1,2])!=sha([2,1])
    print("A1-H runner self-test PASS")


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    mx=p.add_mutually_exclusive_group(required=True)
    mx.add_argument("--all",action="store_true")
    mx.add_argument("--validate",action="store_true")
    mx.add_argument("--self-test",action="store_true")
    p.add_argument("--dataset-root",default="G:/dataset/realiad")
    p.add_argument("--json-dir",default=None)
    p.add_argument("--classes",default="audiojack,rolled_strip_base,switch")
    p.add_argument("--seeds",default="0,1")
    p.add_argument("--budgets",default="128,256")
    p.add_argument("--orders",default="native")
    p.add_argument("--out-root",default="outputs/a1h_hierarchical_memory")
    p.add_argument("--probe-count",type=int,default=96)
    p.add_argument("--probe-every",type=int,default=1024)
    p.add_argument("--math-sdpa",action="store_true")
    p.add_argument("--no-smoke",action="store_true")
    p.add_argument("--skip-mechanism-test",action="store_true")
    return p


if __name__=="__main__":
    args=parser().parse_args()
    if args.self_test:self_test()
    else:
        cases=conditions(args)
        if args.all and not args.skip_mechanism_test:
            subprocess_logged(
                [sys.executable,str(ROOT/"experiments"/"a1h_mechanism_test.py")],
                Path(args.out_root)/"mechanism_test.run.log"
            ) if not (Path(args.out_root)/"mechanism_test.passed").exists() else None
            (Path(args.out_root)/"mechanism_test.passed").touch(exist_ok=True)
        if args.all and not args.no_smoke:
            run_case(args,*cases[0],smoke=True)
        result=[]
        for c,s,b,o in cases:
            if args.validate:
                result.append(check(args,c,s,b,o))
            else:
                result.append(run_case(args,c,s,b,o))
                write_aggregate(args,result,False)
        write_aggregate(args,result,True)
