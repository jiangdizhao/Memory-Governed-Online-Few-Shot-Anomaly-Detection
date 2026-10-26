#!/usr/bin/env python3
"""Single-command resumable V0.2 cold/mature architecture study.

Preflight -> 36 paired full streams -> strict validation -> verdict.
A condition is immutable once written. On interruption rerun --all; completed
valid conditions are reused, corrupt/partial conditions fail closed with logs.
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
sys.path.insert(0,str(ROOT/"experiments"))
import v02_architecture_core as c


def cases():
    return [(cl,s,order) for cl in c.CLASSES for s in c.SEEDS for order in c.ORDERS]


def target(args,case):
    cl,s,order=case
    return Path(args.out_root)/f"{cl}_s{s}_{order}"


def command(args,case):
    cl,s,order=case
    cmd=[sys.executable,str(ROOT/"experiments"/"v02_architecture_trial.py"),
         "--dataset-root",args.dataset_root,"--class-name",cl,
         "--seed",str(s),"--order",order,"--out-dir",str(target(args,case))]
    if args.json_dir:cmd+=["--json-dir",args.json_dir]
    if args.math_sdpa:cmd+=["--math-sdpa"]
    return cmd


def metadata_preflight(args):
    """Read dataset indexes and filenames only; no DINO forward pass."""
    import p0_protocol_audit as p0
    impl=p0.require_sources()
    summary={}
    for cl in c.CLASSES:
        index=impl.RealIADJSONIndex(root=args.dataset_root,json_dir=args.json_dir,
                                    only_classes=[cl])
        train=[s for s in index.train_samples if s.cls_name==cl]
        test=[s for s in index.test_samples if s.cls_name==cl]
        probes,phases,objects=c.split_test(test)
        stream=sum(phases,[])
        tape=c.query_tape(phases)
        all_test=set(objects["A"]+objects["B"]+sum(objects["phases"],[]))
        histories=set()
        for seed in c.SEEDS:
            supports,history=c.select_history_and_support(train,seed)
            if len(history)!=64 or len(supports)!=4:
                raise AssertionError("History/support count invalid")
            if all_test & ({c.object_id(x) for x in supports} |
                           {c.object_id(x.path) for x in history}):
                raise AssertionError(f"Train/test identity overlap: {cl}/s{seed}")
            histories.add(c.digest([x.path for x in history]))
        if len(histories)!=1:
            raise AssertionError("Mature history differs across support seeds")
        if not (len(set(tape))==len(tape) and len(tape)>=1):
            raise AssertionError("Bad query tape")
        summary[cl]={"train_images":len(train),"test_images":len(test),
                     "history_objects":64,"A_probe_objects":len(objects["A"]),
                     "B_probe_objects":len(objects["B"]),
                     "stream_images":len(stream),"query_count":len(tape)}
    print(json.dumps({"stage":"V02_METADATA_PREFLIGHT_PASS","classes":summary},
                     indent=2,ensure_ascii=False),flush=True)


def checked(args,case):
    where=target(args,case)
    f=where/"summary.json"
    if not f.exists():
        raise RuntimeError(f"Partial/incomplete result. Preserve and inspect: {where}")
    x=json.loads(f.read_text(encoding="utf-8"))
    if (x.get("schema"),x.get("complete"),x.get("class_name"),
        x.get("seed"),x.get("order"))!=(c.SCHEMA,True,*case):
        raise RuntimeError(f"Bad case identity {where}")
    for path,digest in x["source_sha256"].items():
        p=ROOT/path
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"Source changed since original run: {path}")
    if len(x["support_objects"])!=4 or len(set(x["support_objects"]))!=4:
        raise RuntimeError("Invalid four-shot supports")
    if x["support_digest"]!=c.digest(x["support_paths"]):
        raise RuntimeError("Corrupt support hash")
    if len(x["history_objects"])!=c.N_HISTORY or len(set(x["history_objects"]))!=c.N_HISTORY:
        raise RuntimeError("History count/uniqueness error")
    if x["history_digest"]!=c.digest(x["history_paths"]) or x["history_normal_labels_charged"]!=64:
        raise RuntimeError("History provenance mismatch")
    if set(x["history_objects"]) & set(x["support_objects"]):
        raise RuntimeError("History leaked into fixed supports")
    ob=x["group_manifest"]
    ids=ob["A"]+ob["B"]+sum(ob["phases"],[])
    if (len(set(ids))!=len(ids) or len(ob["A"])!=48 or len(ob["B"])!=48
        or any(len(p)!=c.GROUPS_PER_PHASE for p in ob["phases"])):
        raise RuntimeError("Invalid physical-object partition")
    if set(ids) & (set(x["support_objects"])|set(x["history_objects"])):
        raise RuntimeError("Train/test object overlap")
    if x["group_digest"]!=c.digest(ob):
        raise RuntimeError("Bad object manifest hash")
    paths=x["stream_paths"]
    flat=sum(paths,[])
    if x["stream_paths_digest"]!=c.digest(paths) or len(flat)!=x["n_stream"]:
        raise RuntimeError("Stream checksum / length mismatch")
    tape=x["query_positions"]
    if (tape!=sorted(set(tape)) or not tape or min(tape)<1 or max(tape)>len(flat)
        or x["query_digest"]!=c.digest(tape)
        or len(tape)!=x["queried_normal"]+x["queried_defect"]
        or abs(len(tape)/len(flat)-c.QUERY_RATE)>.001
        or len({c.object_id(flat[t-1]) for t in tape})!=len(tape)):
        raise RuntimeError("Feedback budget/query object problem")
    if x["maximum_reserved_state_byte_gap"]>.05:
        raise RuntimeError("Maximum reserved capacity differs")
    for name in c.ARMS:
        hist=64 if name.endswith("mature") else 0
        m=x["results"][name]["memory"]
        if (m["feedback_events"]!=x["queried_normal"]+hist
            or m["allocated_state_bytes"]>m["byte_ceiling"]):
            raise RuntimeError(f"Supervision/byte limit violation: {name}")
        if x["history_state_before_online"][name]["feedback_events"]!=hist:
            raise RuntimeError(f"History label leakage: {name}")
    for name in c.REGIMES:
        cf=x["same_state_read_only"][name]
        if set(cf)!={"full","no_extra","no_ltm","no_buffer","ltm_count_end_II"}:
            raise RuntimeError("Missing same-state source audit")
    for filename in ("probe_scores.csv","stream.csv","memory_snapshots.csv","recovery.csv"):
        if not (where/filename).is_file():
            raise RuntimeError("Missing "+filename)
    with (where/"stream.csv").open(encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    if [r["path"] for r in rows]!=flat:
        raise RuntimeError("Row-level stream sample mismatch")
    if [int(r["t"]) for r in rows if int(r["queried"])]!=tape:
        raise RuntimeError("Row-level query tape mismatch")
    if any(k+"_score" not in rows[0] for k in c.ARMS):
        raise RuntimeError("Missing per-arm row-level scores")
    return x


def child(cmd,log):
    if log.exists():
        raise RuntimeError(f"Existing failed log (must inspect before rerun): {log}")
    log.parent.mkdir(parents=True,exist_ok=True)
    print("[V02] Running", " ".join(cmd),flush=True)
    env={**os.environ,"PYTHONIOENCODING":"utf-8"}
    with log.open("w",encoding="utf-8") as f:
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True,
                                 encoding="utf-8",errors="replace",env=env)
        for line in process.stdout:
            f.write(line)
            f.flush()
            enc=sys.stdout.encoding or "utf-8"
            print(line.encode(enc,errors="backslashreplace").decode(
                  enc,errors="replace"),end="",flush=True)
        rc=process.wait()
        f.write(f"\nExit code: {rc}\n")
        if rc:
            raise RuntimeError(f"V0.2 case failed (exit {rc}), see {log}")


def summarize(args):
    found={cas:checked(args,cas) for cas in cases()}
    rows=[]
    for cl in c.CLASSES:
        # These identities are shared across all support seeds by design.
        same=[found[(cl,seed,"AAA")] for seed in c.SEEDS]
        for key in ("group_digest","stream_paths_digest","query_digest","history_digest"):
            if len({s[key] for s in same})!=1:
                raise RuntimeError(f"Support-seed test/historical variance: {cl}/{key}")
        for seed in c.SEEDS:
            aba,aaa=(found[(cl,seed,order)] for order in c.ORDERS)
            for key in ("support_digest","group_digest","stream_paths_digest",
                        "query_digest","history_digest",
                        "queried_normal","queried_defect"):
                if aba[key]!=aaa[key]:
                    raise RuntimeError(f"ABA vs AAA manifest mismatch {cl}/s{seed}/{key}")
            row={"class_name":cl,"seed":seed,
                 "n_online_queries":len(aba["query_positions"]),
                 "n_mature_extra_normal_history":c.N_HISTORY}
            for regime in c.REGIMES:
                H="hierarchy_"+regime
                G="greedy_"+regime
                h1,g1=aba["results"][H],aba["results"][G]
                h0,g0=aaa["results"][H],aaa["results"][G]
                value=lambda result,checkpoint:result[checkpoint]["object_auroc"]
                row["retention_advantage_"+regime]=(
                    (value(h1,"A_end_2")-value(h0,"A_end_2"))-
                    (value(g1,"A_end_2")-value(g0,"A_end_2")))
                row["A_quality_"+regime]=value(h1,"A_end_2")-value(g1,"A_end_2")
                row["B_quality_"+regime]=value(h1,"B_end_2")-value(g1,"B_end_2")
                read=aba["same_state_read_only"][regime]
                row["source_effect_"+regime]=(
                    read["full"]["object_auroc"]-read["no_extra"]["object_auroc"])
                row["ltm_"+regime]=read["ltm_count_end_II"]
                row["hierarchy_update_seconds_"+regime]=h1["memory"]["update_seconds"]
                row["greedy_update_seconds_"+regime]=g1["memory"]["update_seconds"]
                row["hierarchy_reacquisition_"+regime]=h1["reacquisition_labels"]
                row["greedy_reacquisition_"+regime]=g1["reacquisition_labels"]
            rows.append(row)
    verdict=c.decision(rows)
    payload={"schema":"memory.v02.decision.v1","complete":True,
             "planned_conditions":len(cases()),"paired_units":len(rows),
             "paired_results":rows,"engineering_decision":verdict,
             "warning":"Designed diagnostic six-class study; synthetic photometric B shift. Historical v22 full BEA system not reproduced."}
    dest=Path(args.out_root)/"v02_final_decision.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        prior=json.loads(dest.read_text(encoding="utf-8"))
        if prior!=payload:
            raise RuntimeError("Existing decision differs; refuse overwrite")
    else:
        dest.write_text(json.dumps(payload,indent=2,ensure_ascii=False,
                                   allow_nan=False),encoding="utf-8")
    print(json.dumps({"status":"ALL_36_V02_CONDITIONS_VALIDATED",
                      "decision":verdict["decision"],"report":str(dest)},
                     indent=2,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    grp=p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--all",action="store_true")
    grp.add_argument("--preflight",action="store_true")
    grp.add_argument("--validate",action="store_true")
    grp.add_argument("--self-test",action="store_true")
    p.add_argument("--dataset-root",default="G:/dataset/realiad")
    p.add_argument("--json-dir")
    p.add_argument("--out-root",default="outputs/v02_paired_cold_mature")
    p.add_argument("--math-sdpa",action="store_true")
    args=p.parse_args()
    if args.self_test:
        assert len(cases())==36
        assert c.decision([{"class_name":cl,"seed":s,
            "retention_advantage_mature":0.0,"A_quality_mature":0.0,
            "B_quality_mature":0.0,"A_quality_cold":0.0,
            "source_effect_mature":0.0,"ltm_mature":0}
            for cl in c.CLASSES for s in c.SEEDS])["decision"]=="USE_GREEDY_WITH_RISK_SAFE_ADMISSION"
        print("[V02] CPU-only protocol self-test PASS",flush=True)
        return
    if not args.json_dir:
        args.json_dir=str(Path(args.dataset_root)/"jsons"/"realiad_jsons")
    if args.preflight or args.all:
        metadata_preflight(args)
    if args.all:
        for index,cas in enumerate(cases(),1):
            loc=target(args,cas)
            print(f"[V02] condition {index}/{len(cases())}: {cas}",flush=True)
            if not loc.exists():
                child(command(args,cas),loc.with_suffix(".run.log"))
            else:
                print(f"[V02] Revalidating existing {loc}",flush=True)
            checked(args,cas)
    if args.all or args.validate:
        summarize(args)


if __name__=="__main__":
    main()
