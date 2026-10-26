#!/usr/bin/env python3
"""TMRA experiments I and II: fail-closed runner; immutable evidence dirs.

Quick start:
 python experiments/run_tmra.py --all --dataset-root G:/dataset/realiad --math-sdpa

For targeted local checks, run --experiment I before --experiment II.
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
OUTI="tmra.audit1.v1"
OUTII="tmra.recurrence.v1"

def cases(a):
    if a.experiment in ("I","all"):
        yield from [("I",cl,s,None) for cl,s in
                    (("switch",0),("switch",1),("audiojack",0))]
    if a.experiment in ("II","all"):
        yield from [("II",cl,s,o) for cl in ("switch","audiojack")
                    for s in (0,1) for o in ("ABA","AAA")]

def out_path(a,case,smoke=False):
    exp,cl,seed,order=case
    name=(f"I_{cl}_s{seed}" if exp=="I" else f"II_{cl}_s{seed}_{order}")
    if smoke:name+="_smoke"
    return Path(a.out_root)/name

def cmd(a,case,smoke=False):
    exp,cl,seed,order=case
    if exp=="I":
        res=[sys.executable,str(ROOT/"experiments"/"tmra_attribution.py"),
             "--dataset-root",a.dataset_root,"--class-name",cl,
             "--support-seed",str(seed),"--stm-capacity","512",
             "--probe-count","96",
             "--probe-every","1024",
             "--max-samples","64" if smoke else "0"]
    else:
        res=[sys.executable,str(ROOT/"experiments"/"tmra_recurrence.py"),
             "--dataset-root",a.dataset_root,"--class-name",cl,
             "--seed",str(seed),"--order",order,
             "--probe-count","96",
             "--phase-length","64" if smoke else "900"]
    res+=["--out-dir",str(out_path(a,case,smoke))]
    if a.json_dir:res+=["--json-dir",a.json_dir]
    if a.math_sdpa:res.append("--math-sdpa")
    return res

def sha(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def check(a,case,smoke=False):
    root=out_path(a,case,smoke)
    summary=root/"summary.json"
    if not summary.is_file():
        raise RuntimeError(f"Incomplete run; inspect/rename only this directory: {root}")
    j=json.loads(summary.read_text(encoding="utf-8"))
    exp,cl,seed,order=case
    if j.get("schema")!=(OUTI if exp=="I" else OUTII) or not j.get("complete"):
        raise RuntimeError(f"Schema/completion mismatch: {root}")
    source=j.get("source_sha256",{})
    if not source:raise RuntimeError(f"Missing exact source fingerprints: {root}")
    for path,digest in source.items():
        f=ROOT/path
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"Code changed after experiment: {path}; use a NEW out-root")
    if exp=="I":
        p=j["protocol"]
        if (j["class_name"]!=cl or j["support_seed"]!=seed
            or len(set(p["supports"]))!=4
            or sha(p["supports"])!=p["support_hash"]
            or j["n_probe"]!=96
            or j["n_queries"]!=j["queried_normal"]+j["queried_defect"]):
            raise RuntimeError("TMRA-I supervision/support manifest mismatch")
        if not j.get("attribution_summary"):
            raise RuntimeError("Missing attribution summary")
        required=("source_image_attribution.csv","source_patch_evidence.csv",
                  "retrieval_counterfactual.csv","probe_scores.csv",
                  "memory_snapshots.csv","prequential.csv")
    else:
        v=j["condition"];m=j["manifest"]
        if (v["class_name"],v["seed"],v["order"])!=(cl,seed,order):
            raise RuntimeError("TMRA-II identity mismatch")
        if v["phase_length"]!=(64 if smoke else 900):
            raise RuntimeError("TMRA-II phase-length mismatch")
        if (len(set(m["supports"]))!=4
            or sha(m["supports"])!=m["support_sha"]
            or m["query_sha"]!=sha(m["query_positions"])
            or m["q_normal"]+m["q_defect"]!=len(m["query_positions"])
            or len(j["results"])!=4
            or m["byte_ceiling"]!=3_500_000):
            raise RuntimeError("TMRA-II support/query/memory contract mismatch")
        phase=m["phase_stream_paths"]
        all_train_paths=sum(phase,[])
        probe_paths=m["A_probe_paths"]+m["B_probe_paths"]
        if (len(phase)!=3 or any(len(x)!=v["phase_length"] for x in phase)
            or len(set(all_train_paths+probe_paths))!=len(all_train_paths+probe_paths)
            or set(m["supports"])&set(all_train_paths+probe_paths)
            or len(probe_paths)!=192):
            raise RuntimeError("TMRA-II leakage/phase error")
        for method,val in j["results"].items():
            aud=val["memory"]
            if aud and (aud["allocated_state_bytes"]>m["byte_ceiling"]
                        or aud["feedback_events"]!=m["q_normal"]):
                raise RuntimeError(f"TMRA-II unmatched resources/feedback: {method}")
        required=("probes.csv","prequential.csv","memory_snapshots.csv",
                  "reacquisition.csv")
    for fname in required:
        if not (root/fname).is_file():
            raise RuntimeError(f"Missing required artifact {fname}")
    with (root/"prequential.csv").open(encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    if exp=="I":
        if len(rows)!=j["n_stream"]:
            raise RuntimeError("TMRA-I incomplete prequential rows")
    else:
        if len(rows)!=3*v["phase_length"]:
            raise RuntimeError("TMRA-II incomplete prequential rows")
        if [r["path"] for r in rows]!=all_train_paths:
            raise RuntimeError("TMRA-II stream digest mismatch")
        if [int(r["t"]) for r in rows if int(r["queried"])]!=m["query_positions"]:
            raise RuntimeError("TMRA-II actual query positions mismatch")
    return {"experiment":exp,"class_name":cl,"seed":seed,"order":order,
            "result_path":str(root),
            "result":j.get("attribution_summary",j.get("results"))}

def run(cmd,logpath):
    if logpath.exists():
        raise FileExistsError(f"Inspect partial prior log; no auto-overwrite: {logpath}")
    logpath.parent.mkdir(parents=True,exist_ok=True)
    print("[TMRA]"," ".join(cmd),flush=True)
    with logpath.open("w",encoding="utf-8") as f:
        p=subprocess.Popen(
            cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding="utf-8",errors="replace",
            env={**os.environ,"PYTHONIOENCODING":"utf-8"})
        for line in p.stdout:
            f.write(line);f.flush()
            enc=sys.stdout.encoding or "utf-8"
            print(line.encode(enc,errors="backslashreplace").decode(
                enc,errors="replace"),end="",flush=True)
        rc=p.wait()
        f.write(f"\nExit code: {rc}\n")
    if rc:raise RuntimeError(f"Child failed ({rc}), see {logpath}")

def execute(a,case,smoke=False):
    d=out_path(a,case,smoke)
    if d.exists():
        print(f"[TMRA] Validate and skip: {d}",flush=True)
        return check(a,case,smoke=smoke)
    log=d.with_name(d.name+".run.log")
    run(cmd(a,case,smoke),log)
    return check(a,case,smoke=smoke)

def aggregate(a,completed,expected):
    d=Path(a.out_root);d.mkdir(parents=True,exist_ok=True)
    name=a.experiment.lower()
    output=d/f"aggregate_{name}.json"
    output.write_text(json.dumps({
        "schema":"tmra.aggregate.v1",
        "complete":len(completed)==expected,
        "validated":len(completed),"expected":expected,
        "conditions":completed,
    },indent=2,allow_nan=False,ensure_ascii=False),encoding="utf-8")
    print(f"[TMRA] {len(completed)}/{expected} validated -> {output}",flush=True)

def check_pairs(a,completed):
    """Same raw stream/support/label tape for ABA and AAA; only B images differ."""
    if a.experiment not in ("II","all"):return
    for cl in ("switch","audiojack"):
        for seed in (0,1):
            case1=("II",cl,seed,"ABA");case2=("II",cl,seed,"AAA")
            if any(not out_path(a,cas).exists() for cas in (case1,case2)):
                continue
            data=[]
            for cas in (case1,case2):
                j=json.loads((out_path(a,cas)/"summary.json").read_text(encoding="utf-8"))
                data.append(j["manifest"])
            for k in ("supports","phase_stream_sha","A_probe_paths",
                      "B_probe_paths","query_positions","q_normal","q_defect"):
                if data[0][k]!=data[1][k]:
                    raise RuntimeError(f"ABA/AAA matched-source drift in {cl}/{seed}: {k}")

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",action="store_true")
    group.add_argument("--experiment",choices=("I","II","all"))
    group.add_argument("--validate",action="store_true")
    group.add_argument("--self-test",action="store_true")
    p.add_argument("--dataset-root",default="G:/dataset/realiad")
    p.add_argument("--json-dir",default=None)
    p.add_argument("--out-root",default="outputs/tmra")
    p.add_argument("--math-sdpa",action="store_true")
    p.add_argument("--no-smoke",action="store_true")
    return p

if __name__=="__main__":
    args=parser().parse_args()
    if args.self_test:
        args.experiment="all"
        assert len(list(cases(args)))==11
        assert len(list(cases(type("A",(),{"experiment":"I"})())))==3
        assert cmd(args,("II","switch",0,"ABA"),True)[
            cmd(args,("II","switch",0,"ABA"),True).index("--phase-length")+1
            ]=="64"
        print("[TMRA] Offline runner self-test PASS")
    else:
        if args.all or args.validate:args.experiment="all"
        tasks=list(cases(args))
        if args.all or args.experiment:
            if not args.validate and not args.no_smoke:
                # CPU operator path for both bridge and compact memory.
                smokeI=("I","switch",0,None)
                smokeII=("II","switch",0,"ABA")
                if args.experiment in ("I","all"):
                    execute(args,smokeI,True)
                if args.experiment in ("II","all"):
                    execute(args,smokeII,True)
            done=[]
            for cas in tasks:
                row=check(args,cas) if args.validate else execute(args,cas)
                done.append(row)
                aggregate(args,done,len(tasks))
            check_pairs(args,done)
