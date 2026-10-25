#!/usr/bin/env python3
"""A1-H one-condition experiment: strict4, fixed labels, 8 memory arms.

Outputs are immutable: existing result directories are refused. The outer
orchestrator handles the full class/seed/budget matrix.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import a0_a1_strict4_matched as a1
import a1h_memory as mem

SCHEMA = "a1h.strict4.hierarchical.attribution.v1"


def hash_list(xs):
    return hashlib.sha256(json.dumps(xs, sort_keys=True).encode()).hexdigest()


def auc_metrics(labels, scores, theta):
    return a1.metrics(labels, scores, theta)


def aulc(points):
    """Time-weighted, trapezoidal mean AUROC; null if one checkpoint undefined."""
    if len(points) < 2 or any(p["auroc"] is None for p in points):
        return None
    end = points[-1]["step"] - points[0]["step"]
    if end <= 0:
        return None
    area = sum((b["step"]-a["step"])*(a["auroc"]+b["auroc"])/2
               for a,b in zip(points,points[1:]))
    return area/end


def main(args):
    import numpy as np
    import torch
    import p0_protocol_audit as p0
    import e02_matched_feedback as e02
    import e02_r6_sidecar_stability as r6

    if args.math_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(f"Refusing any overwrite: {out}")
    out.mkdir(parents=True)
    impl = p0.require_sources()
    impl.set_seed(0)
    budget = mem.Budget(args.budget)
    tstart = time.perf_counter()
    ix = impl.RealIADJSONIndex(root=args.dataset_root, json_dir=args.json_dir,
                               only_classes=[args.class_name])
    normals = sorted({s.path for s in ix.train_samples
                      if s.cls_name == args.class_name and s.is_good})
    tests = sorted((s for s in ix.test_samples
                    if s.cls_name == args.class_name), key=lambda x:x.path)
    if len(tests) < args.probe_count+64:
        raise ValueError("Not enough separate test samples for stream and probe")
    supports = sorted(random.Random(args.support_seed).sample(normals,4))
    a1.verify_supports(supports,normals,[x.path for x in tests])
    cfg = p0.Config(
        device=args.device, augmentation_seed=0, bank_seed=0,
        fixed_capacity=25000, fixed_per_image=5000,
        augmentations=4, include_identity=False, multiplier=2.5,
        feature_fuse="concat", topk_patches=32,
        image_aggregation="max", image_topk=64,
        image_quantile=.95, knn=5)

    backbone = impl.DINOv2MultiLayerBackbone(device=args.device)
    refs = p0.encode_reference_features(impl, backbone, supports, cfg)
    canonical = p0.build_vmb(impl, refs, supports, cfg)
    bank_id = p0.fixed_bank_profile(canonical)
    loso = []
    loso_folds = []
    for held in supports:
        keep = [p for p in supports if p != held]
        assert len(keep)==3 and held not in keep
        fold = p0.build_vmb(impl, refs, keep, cfg)
        loso.append(p0.score_image(impl,backbone,fold,held,cfg))
        loso_folds.append({"heldout":held, "training":keep,
                           "sha256":p0.fixed_bank_profile(fold)["sha256"]})
        del fold
    theta_stats = p0.threshold_from_scores(loso,cfg.multiplier)
    theta = theta_stats["theta"]
    del refs

    shuffled = list(tests)
    random.Random(0).shuffle(shuffled)
    stream,probe = a1.select_probe(shuffled,args.probe_count,args.probe_seed)
    stream = r6.order_stream(stream,args.stream_order)
    if args.max_samples:
        stream = stream[:args.max_samples]
    if len(stream)<32:
        raise ValueError("Insufficient stream samples")
    tape = e02.make_query_tape(len(stream),.03,37)

    arms = {
        name:mem.make_arm(
            name, copy.deepcopy(canonical), budget, seed=mem.a1.stable_seed(
                args.class_name,args.support_seed,37),
            evidence_mode=args.evidence_mode)
        for name in mem.ARMS
    }
    del canonical
    for a in arms.values():
        if p0.fixed_bank_profile(a.bank)["sha256"]!=bank_id["sha256"]:
            raise AssertionError("Initial fixed-bank mismatch")
        assert a.state()["capacity_total"]==budget.total
    # Encode fixed probe ONCE; reusable frozen embeddings, no retraining or
    # query/memory access to probes. Store on CPU fp16 for bounded GPU memory.
    probe_cache=[]
    for s in probe:
        img, feats=p0.encode_query(impl,backbone,s.path,cfg)
        probe_cache.append((s.path,int(not s.is_good),
                            SimpleNamespace(height=img.height,width=img.width),
                            [f.detach().to("cpu",dtype=torch.float16) for f in feats]))
    if {x[0] for x in probe_cache}&{s.path for s in stream}:
        raise AssertionError("Probe leakage")

    def scoring(bank, img, feats, config):
        return a1.score_bank(impl,bank,img,feats,config)[0]

    rows=[]
    probe_records=[]
    probe_summary=[]
    occupancy=[]
    scores={n:[] for n in mem.ARMS}
    labels=[]
    qnormal=qdefect=0
    checkpoint_steps=[]

    def checkpoint(t):
        import torch
        # Critical causal assertions: H1/H2 can differ in shadow LTM state but
        # not STM; H2/H3 must have identical STM, LTM, buffer state. Only their
        # retrieval toggle may change scores.
        def same_bank(a, b, family):
            ca=getattr(arms[a].bank,family)["p0"][0]["id"]
            cb=getattr(arms[b].bank,family)["p0"][0]["id"]
            for x,y in zip(ca,cb):
                if x.size()!=y.size() or not torch.equal(
                        x.protos[:x.size()],y.protos[:y.size()]):
                    raise AssertionError(
                        f"Unexpected memory-state coupling between {a}/{b}: {family}")
        same_bank("H1_track","H2_shadow_promotion","normal_dyn")
        same_bank("H2_shadow_promotion","H3_ltm_retrieval","normal_dyn")
        same_bank("H2_shadow_promotion","H3_ltm_retrieval","normal_dyn_ltm")
        before_update={n:a.state() for n,a in arms.items()}
        for name,s in before_update.items():
            occupancy.append({"step":t,**s})
        snapshot={name:[] for name in mem.ARMS}
        for path,label,img,fts_cpu in probe_cache:
            fts=[z.to(args.device) for z in fts_cpu]
            for name,arm in arms.items():
                score=scoring(arm.bank,img,fts,cfg)
                snapshot[name].append((label,score))
                probe_records.append(
                    {"step":t,"path":path,"label_eval_only":label,
                     "method":name,"score":score,"theta_loso_diagnostic":theta})
            del fts
        for name,items in snapshot.items():
            y=[x[0] for x in items]; pred=[x[1] for x in items]
            p={"step":t,"method":name,
               **auc_metrics(y,pred,theta)}
            probe_summary.append(p)
        # direct LTM retrieval counterfactual on SAME state and SAME probe;
        # it affects only one-off diagnostic scoring, never replay decisions.
        for name in ("H3_ltm_retrieval","H4_utility"):
            arm=arms[name]
            deltas=[]
            for path,label,img,fts_cpu in probe_cache:
                fts=[z.to(args.device) for z in fts_cpu]
                z=mem.counterfactual_ltm_scores(scoring,arm,img,fts,cfg)
                deltas.append(z["delta"])
            occupancy[-len(arms)+list(arms).index(name)][
                "probe_mean_ltm_score_delta"]=sum(deltas)/len(deltas)
            occupancy[-len(arms)+list(arms).index(name)][
                "probe_any_ltm_score_effect"]=sum(abs(z)>1e-7 for z in deltas)
        # Same storage checkpoint, native v22 heterogeneous-k retrieval is
        # evaluated separately and NEVER used for an online decision/update.
        if t==len(stream):
            for name in ("H3_ltm_retrieval","H4_utility"):
                arm=arms[name]
                b=arm.bank
                old=b._dN_min_fixed_dyn
                try:
                    b._dN_min_fixed_dyn=type(b)._dN_min_fixed_dyn.__get__(b,type(b))
                    native=[]
                    for path,label,img,fts_cpu in probe_cache:
                        feats=[z.to(args.device) for z in fts_cpu]
                        native.append((label,scoring(b,img,feats,cfg)))
                    y=[p[0] for p in native]
                    scr=[p[1] for p in native]
                    occupancy[-len(arms)+list(arms).index(name)][
                        "native_v22_probe_auc_diagnostic"]=auc_metrics(y,scr,theta)["auroc"]
                finally:
                    b._dN_min_fixed_dyn=old
        # shadow-promotion invariance: nonretrieved LTM must not affect score.
        for x,y in zip(snapshot["H1_track"],snapshot["H2_shadow_promotion"]):
            if abs(x[1]-y[1])>args.shadow_tolerance:
                raise AssertionError("H1/H2 shadow scores diverged; possible hidden state coupling")
        checkpoint_steps.append(t)

    checkpoint(0)
    for t,sample in enumerate(stream,1):
        img,fts=p0.encode_query(impl,backbone,sample.path,cfg)
        static_score,static_map=a1.score_bank(
            impl,arms["static"].bank,img,fts,cfg)
        candidate={l:idx[:8] for l,idx in
                   impl.layer_to_topk_from_amap(static_map).items()}
        current={}
        for name,arm in arms.items():
            sc=static_score if name=="static" else scoring(arm.bank,img,fts,cfg)
            current[name]=sc
            scores[name].append(sc)
        truth=int(not sample.is_good)
        labels.append(truth)
        queried=t in tape
        rows.append({"t":t,"path":sample.path,
                     "label_eval_only":truth,"queried":int(queried),
                     **{n+"_score":v for n,v in current.items()}})
        # Labels are consumed ONLY after every arm has made its prediction.
        if queried:
            if not truth:
                qnormal+=1
                for arm in arms.values():
                    arm.observe_verified_normal(
                        impl, fts,candidate, step=t,
                        score=static_score,theta=theta,sigma=theta_stats["sigma"])
            else:qdefect+=1
        if args.probe_every and t%args.probe_every==0:
            checkpoint(t)
        if args.print_every and t%args.print_every==0:
            print(f"[A1H] {args.class_name} s{args.support_seed} "
                  f"{args.stream_order} K={budget.total} "
                  f"{t}/{len(stream)}",flush=True)
    if checkpoint_steps[-1]!=len(stream):
        checkpoint(len(stream))
    if qnormal+qdefect!=len(tape):
        raise AssertionError("Query budget mismatch")

    def write_csv(file,items):
        if not items: return
        keys=list(dict.fromkeys(k for r in items for k in r))
        with (out/file).open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(items)

    metric={name:auc_metrics(labels,scores[name],theta)
            for name in mem.ARMS}
    for name in mem.ARMS:
        marks=[x for x in probe_summary if x["method"]==name]
        metric[name]["probe_aulc"]=aulc(marks)
        metric[name]["probe_start_auc"]=marks[0]["auroc"]
        metric[name]["probe_final_auc"]=marks[-1]["auroc"]
        metric[name]["probe_worst_auc"]=(
            min(x["auroc"] for x in marks if x["auroc"] is not None)
            if any(x["auroc"] is not None for x in marks) else None)
        metric[name]["memory"]=arms[name].state()

    # Streaming data access must be identical; hierarchy can NOT query more.
    counts=[arms[n].state()["candidate_events"] for n in mem.ARMS if n!="static"]
    patches=[arms[n].state()["candidate_patches"] for n in mem.ARMS if n!="static"]
    if len(set(counts))!=1 or counts[0]!=qnormal or len(set(patches))!=1:
        raise AssertionError("Feedback/candidate workload differs across arms")
    if any(p0.fixed_bank_profile(arm.bank)["sha256"]!=bank_id["sha256"]
           for arm in arms.values()):
        raise AssertionError("Fixed support bank changed")

    write_csv("prequential.csv",rows)
    write_csv("probe_scores.csv",probe_records)
    write_csv("probe_summary.csv",probe_summary)
    write_csv("memory_snapshots.csv",occupancy)
    ltm_activation = {
        name:{
            "promotion_count":metric[name]["memory"]["promotion_successes"],
            "promoted_new_prototypes":metric[name]["memory"]["promoted_new_prototypes"],
            "final_ltm":metric[name]["memory"]["ltm"],
            "retrieval_effect_probes_at_final":next(
                (r.get("probe_any_ltm_score_effect") for r in reversed(occupancy)
                 if r["method"]==name and r["step"]==len(stream)),None),
            "interpretation":"unactivated_in_real_stream"
                if metric[name]["memory"]["ltm"]==0
                else "ltm_stored_no_retrieval" if name=="H2_shadow_promotion"
                else "ltm_stored_no_detectable_probe_score_effect"
                if not any(r.get("probe_any_ltm_score_effect",0)>0 for r in occupancy
                           if r["method"]==name)
                else "ltm_retrieval_changes_probe_scores"
        } for name in ("H2_shadow_promotion","H3_ltm_retrieval","H4_utility")
    }
    report={
        "schema":SCHEMA,"complete":True,
        "class_name":args.class_name,
        "support_seed":args.support_seed,
        "order":args.stream_order,
        "budget":budget.total,
        "n_stream":len(stream),"n_probe":len(probe),"n_queries":len(tape),
        "queried_normal":qnormal,"queried_defect":qdefect,
        "protocol":{
            "support_paths":supports,"support_hash":hash_list(supports),
            "strict_four_shot":True,
            "support_access":4,
            "loso_scores":loso,"loso_folds":loso_folds,
            "loso_theta_diagnostic_only":theta_stats,
            "fixed_bank":bank_id,
            "stream_hash":hash_list([x.path for x in stream]),
            "probe_hash":hash_list([x.path for x in probe]),
            "probe_paths":[x.path for x in probe],
            "query_positions":sorted(tape),
            "query_hash":hash_list(sorted(tape)),
            "candidate_from_shared_static":True,
            "candidate_patches_per_event_layer":8,
            "frozen_dino":True,
            "uniform_retrieval":"global_top5_fixed_plus_stm_plus_optional_ltm",
            "calibration":"LOSO diagnostic, no adaptive threshold, no reported deployment BA validity",
            "hierarchical_allocation":{
                "stm":budget.stm,"ltm":budget.ltm,"buffer":budget.buffer},
            "evidence_mode":args.evidence_mode,
            "probes_never_used_for_training":True,
            "labels_after_predictions":True,
        },
        "arms":metric,
        "ltm_activation":ltm_activation,
        "elapsed_seconds":time.perf_counter()-tstart,
        "args":vars(args),
        "source_sha256":{
            rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
            for rel in (
                "experiments/a1h_memory.py",
                "experiments/a1h_hierarchical_matched.py",
                "experiments/a0_a1_memory_methods.py",
                "experiments/a0_a1_strict4_matched.py",
                "experiments/p0_protocol_audit.py",
                "test_phase3_AIF_simplified_v22.py",
            )
        },
        "limitations":[
            "S0/S1/S2/S3/H1-H4 isolate normal-memory submodules, not complete v22.",
            "H2 stores promoted LTM prototypes but disables their retrieval, yielding a shadow state.",
            "Same max stored-prototype budget; tracker metadata and update compute additionally reported.",
            "Unified retrieval differs from native v22 min(Fixed,STM,LTM) with heterogeneous k.",
            "LOSO fixed theta is diagnostic only; BA/FNR/FPR are not deployment-calibrated.",
            "Stress orders are offline label-defined interventions only, not policy observations.",
            "Probe labels are used only in metrics, never online memory or queries.",
            "Native Real-IAD may lack enough repeated normals to produce LTM promotion.",
        ],
    }
    (out/"summary.json").write_text(
        json.dumps(report,indent=2,allow_nan=False,ensure_ascii=False,default=str),
        encoding="utf-8")
    print(json.dumps({"condition":f"{args.class_name}/s{args.support_seed}"
                      f"/{args.stream_order}/K{budget.total}",
                      "AUROC":{n:round(v["auroc"],5) if v["auroc"] is not None else None
                               for n,v in metric.items()},
                      "probe_AULC":{n:round(v["probe_aulc"],5)
                                    if v["probe_aulc"] is not None else None
                                    for n,v in metric.items()},
                      "ltm":{n:v["memory"]["ltm"] for n,v in metric.items()},
                      "output":str(out)},indent=2),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",default="G:/dataset/realiad")
    p.add_argument("--json-dir",default=None)
    p.add_argument("--class-name",default="audiojack")
    p.add_argument("--support-seed",type=int,default=0)
    p.add_argument("--stream-order",choices=("native","normal_first","defect_first"),default="native")
    p.add_argument("--budget",type=int,choices=(128,256),default=128)
    p.add_argument("--probe-count",type=int,default=96)
    p.add_argument("--probe-seed",type=int,default=212)
    p.add_argument("--probe-every",type=int,default=1024)
    p.add_argument("--max-samples",type=int,default=0)
    p.add_argument("--device",default="cuda")
    p.add_argument("--math-sdpa",action="store_true")
    p.add_argument("--evidence-mode",choices=("neutral","loso_margin"),
                   default="neutral",
                   help="neutral is primary: no dependence on defective LOSO margin")
    p.add_argument("--print-every",type=int,default=200)
    p.add_argument("--shadow-tolerance",type=float,default=1e-5)
    p.add_argument("--out-dir",required=True)
    return p


if __name__=="__main__":
    x=parser().parse_args()
    if x.json_dir is None:
        x.json_dir=str(Path(x.dataset_root)/"jsons"/"realiad_jsons")
    if not 4<=x.probe_count<=512 or x.probe_every<1 or x.max_samples<0:
        raise SystemExit("Invalid probe/sample configuration")
    main(x)
