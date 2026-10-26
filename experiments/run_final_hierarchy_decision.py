#!/usr/bin/env python3
"""ONE FINAL TEST: 2 classes x 2 seeds x matched ABA/AAA, Greedy vs Hierarchy.

No sweep. One command writes final_decision.json after all eight validated runs.
Existing successful immutable runs are checked and reused. A broken/partial run
is never overwritten automatically. No result-based retuning allowed.
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
import final_decision_core as final

def cases():
    return [(c,s,o) for c in final.CLASSES for s in final.SEEDS for o in final.ORDERS]

def location(a,c):
    cl,seed,order=c
    return Path(a.out_root)/f"{cl}_s{seed}_{order}"

def command(a,c):
    cl,seed,order=c
    cmd=[sys.executable,str(ROOT/"experiments"/"final_hierarchy_trial.py"),
         "--dataset-root",a.dataset_root,
         "--class-name",cl,"--seed",str(seed),"--order",order,
         "--out-dir",str(location(a,c))]
    if a.json_dir:cmd+=["--json-dir",a.json_dir]
    if a.math_sdpa:cmd+=["--math-sdpa"]
    return cmd

def open_checked(a,case):
    where=location(a,case)
    p=where/"summary.json"
    if not p.is_file():raise RuntimeError(f"Incomplete, do not overwrite: {where}")
    j=json.loads(p.read_text(encoding="utf-8"))
    if (j.get("schema"),j.get("complete"),j.get("class_name"),
        j.get("seed"),j.get("order"))!=(
        final.SCHEMA,True,*case):
        raise RuntimeError("Condition identity mismatch")
    s=j.get("source_sha256",{})
    if not s:raise RuntimeError("Missing source manifest")
    for path,sha in s.items():
        f=ROOT/path
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=sha:
            raise RuntimeError(f"Code fingerprint changed since run: {path}")
    if (len(j["support_paths"])!=4 or
        len(set(j["support_objects"]))!=4 or
        j["support_digest"]!=final.digest(j["support_paths"]) or
        j["group_digest"]!=final.digest(j["group_manifest"]) or
        len(j["support_loso_scores"])!=4):
        raise RuntimeError("Four-shot / LOSO invalid")
    ob=j["group_manifest"]
    gids=ob["A"]+ob["B"]+sum(ob["phases"],[])
    if (len(set(gids))!=len(gids) or
        len(ob["A"])!=24 or len(ob["B"])!=24 or
        any(len(p)!=175 for p in ob["phases"]) or
        set(gids)&set(j["support_objects"])):
        raise RuntimeError("Physical-instance overlap or incorrect counts")
    paths=j["stream_paths"]
    if len(paths)!=3 or len(set(sum(paths,[])))!=len(sum(paths,[])):
        raise RuntimeError("Stream duplicates")
    if j["stream_paths_digest"]!=final.digest(paths):
        raise RuntimeError("Stream manifest mismatch")
    qt=j["query_positions"]
    if (j["query_digest"]!=final.digest(qt) or
        len(qt)!=j["queried_normal"]+j["queried_defect"] or
        abs(len(qt)/j["n_stream"]-.03)>.001):
        raise RuntimeError("Unmatched feedback budget")
    flat=sum(paths,[])
    if len({final.object_id(flat[t-1]) for t in qt})!=len(qt):
        raise RuntimeError("Multiple labels for one physical instance")
    if len(qt)!=len(set(qt)) or min(qt)<1 or max(qt)>len(flat):
        raise RuntimeError("Invalid query positions")
    trial_ids=set(j["support_objects"]) | set(ob["A"]) | set(ob["B"])
    if trial_ids&set(sum(ob["phases"],[]))-set(ob["A"])-set(ob["B"]):
        # the subtraction is redundant with earlier assertions,
        # retained only as defense in depth.
        raise RuntimeError("Support or probe contaminated stream")
    if j["baseline_comparability"]["original_paper_full_stack_reproduction"]:
        raise RuntimeError("Mislabelled full-system replication")
    if not j["baseline_comparability"]["equal_actual_mutable_tensor_bytes_within_5_percent"]:
        raise RuntimeError("Resource parity unverified")
    for phase in ("A_end_1","A_end_2","A_end_3"):
        if j["actual_bytes_by_phase"][phase]["relative_difference"]>.05:
            raise RuntimeError("Actual byte mismatch at phase " + phase)
    for arm in ("greedy_coreset","hierarchy"):
        m=j["results"][arm]["memory"]
        if (m["allocated_state_bytes"]>final.BYTE_CEILING or
            m["feedback_events"]!=j["queried_normal"]):
            raise RuntimeError("Dynamic memory/supervision mismatch")
    for artifact in ("stream.csv","probe_scores.csv",
                     "memory_snapshots.csv","recovery.csv"):
        if not (where/artifact).is_file():
            raise RuntimeError("Missing artifact "+artifact)
    with (where/"stream.csv").open(encoding="utf-8",newline="") as f:
        rs=list(csv.DictReader(f))
    if len(rs)!=len(flat) or [z["path"] for z in rs]!=flat:
        raise RuntimeError("Prequential stream corrupt")
    if [int(z["t"]) for z in rs if int(z["queried"])]!=qt:
        raise RuntimeError("Actual feedback positions drifted")
    return j

def run_child(cmd,log):
    if log.exists():raise FileExistsError(f"Prior failed log exists: {log}")
    log.parent.mkdir(parents=True,exist_ok=True)
    print("[FINAL]"," ".join(cmd),flush=True)
    with log.open("w",encoding="utf-8") as f:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,text=True,
                           encoding="utf-8",errors="replace",
                           env={**os.environ,"PYTHONIOENCODING":"utf-8"})
        for line in p.stdout:
            f.write(line);f.flush()
            outenc=sys.stdout.encoding or "utf-8"
            print(line.encode(outenc,errors="backslashreplace").decode(
                outenc,errors="replace"),end="",flush=True)
        code=p.wait()
        f.write(f"\nExit code: {code}\n")
        if code:raise RuntimeError(f"Trial failed with exit {code}, see {log}")

def cpu_preflight(a):
    # Reads only dataset JSON metadata and validates physical grouping,
    # positive/negative probe units and 3% group-distinct feedback.
    import final_instance_preflight as pf
    pf.validate(a.dataset_root,a.json_dir)

def make_decision(a):
    data={cas:open_checked(a,cas) for cas in cases()}
    rows=[]
    for cl in final.CLASSES:
        for seed in final.SEEDS:
            aba,aaa=(data[(cl,seed,o)] for o in final.ORDERS)
            for key in ("support_digest","group_digest","stream_paths_digest",
                        "query_digest","queried_normal","queried_defect"):
                if aba[key]!=aaa[key]:
                    raise RuntimeError(f"ABA/AAA unmatched {cl}/s{seed}: {key}")
            h1,h0=(aba["results"]["hierarchy"],aaa["results"]["hierarchy"])
            g1,g0=(aba["results"]["greedy_coreset"],aaa["results"]["greedy_coreset"])
            cf=aba["same_state_read_only"]
            effect=(cf["full"]["object_auroc"]-cf["no_extra"]["object_auroc"])
            rows.append({
                "class_name":cl,"seed":seed,
                "hierarchy_shift_advantage":(
                    (h1["A_end_2"]["object_auroc"]-h0["A_end_2"]["object_auroc"])-
                    (g1["A_end_2"]["object_auroc"]-g0["A_end_2"]["object_auroc"])),
                "hierarchy_end2_minus_greedy_end2":(
                    h1["A_end_2"]["object_auroc"]-g1["A_end_2"]["object_auroc"]),
                "hierarchy_B_end2_minus_greedy_B_end2":(
                    h1["B_end_2"]["object_auroc"]-g1["B_end_2"]["object_auroc"]),
                "A_retrieval_effect":bool(effect>=.005 and cf["ltm_count_end_II"]>0),
                "extra_retrieval_A_auc_effect":effect,
                "LTM_independent_effect":(
                    cf["full"]["object_auroc"]-cf["no_ltm"]["object_auroc"]),
                "normal_label_feedback_by_phase":aba["queried_normal_per_phase"],
                "hierarchy_ltm":h1["memory"]["ltm"],
                "hierarchy_buffer":h1["memory"]["buffer"],
                "A_AUROC_ABA":{
                    arm:aba["results"][arm]["A_end_2"]["object_auroc"]
                    for arm in ("greedy_coreset","hierarchy")},
                "A_AUROC_AAA":{
                    arm:aaa["results"][arm]["A_end_2"]["object_auroc"]
                    for arm in ("greedy_coreset","hierarchy")},
                "recovery_labels":{
                    arm:aba["results"][arm]["reacquisition_labels"]
                    for arm in ("greedy_coreset","hierarchy")},
            })
    decision=final.pass_decision(rows)
    result={
        "schema":"final.hierarchy.verdict.v1",
        "complete":True,
        "protocol_frozen":True,
        "paired_cases":rows,
        "engineering_decision":decision,
        "interpretation":"This is a bounded engineering choice, not a claim hierarchy is universally impossible.",
        "rules":"No additional tests, tuning, or alternative cohorts after reviewing this result.",
    }
    target=Path(a.out_root)/"final_decision.json"
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False,
                                 allow_nan=False),encoding="utf-8")
    print(json.dumps({"VERDICT":decision,"result_path":str(target)},
                     indent=2,ensure_ascii=False),flush=True)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",action="store_true")
    group.add_argument("--validate",action="store_true")
    group.add_argument("--preflight",action="store_true")
    group.add_argument("--self-test",action="store_true")
    p.add_argument("--dataset-root",default="G:/dataset/realiad")
    p.add_argument("--json-dir")
    p.add_argument("--out-root",default="outputs/final_hierarchy_decision")
    p.add_argument("--math-sdpa",action="store_true")
    return p

if __name__=="__main__":
    a=parser().parse_args()
    if a.self_test:
        assert len(cases())==8
        assert final.pass_decision([{
            "hierarchy_shift_advantage":-.01,
            "hierarchy_end2_minus_greedy_end2":-.02,
            "hierarchy_B_end2_minus_greedy_B_end2":-.01,
            "A_retrieval_effect":False}]*4)["keep_hierarchy"] is False
        print("[FINAL] Frozen 8-condition runner and gate PASS")
    else:
        if not a.json_dir:
            a.json_dir=str(Path(a.dataset_root)/"jsons"/"realiad_jsons")
        if a.preflight or a.all:cpu_preflight(a)
        if a.all:
            for cas in cases():
                where=location(a,cas)
                if where.exists():
                    print(f"[FINAL] Validate existing {where}",flush=True)
                else:
                    run_child(command(a,cas),where.with_suffix(".run.log"))
                open_checked(a,cas)
        if a.all or a.validate:
            make_decision(a)
