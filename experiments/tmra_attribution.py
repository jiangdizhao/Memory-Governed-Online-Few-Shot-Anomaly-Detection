#!/usr/bin/env python3
"""Bridge-A: historically configured v22 normal-memory factorial under matched feedback.

* Historical mechanisms/capacities, NOT historical full-system reproduction:
  strict four-shot, fixed random 3% verified feedback, no sidecar/BEA/corrections.
* One common immutable Static bank selects all candidate patches.
* Native v22 min(Fixed,STM,LTM,Buffer) score; read-only checkpoint switches.
* Probe split independent of stream and never exposed to memory learner.
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
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
for d in (ROOT,HERE):
    if str(d) not in sys.path:sys.path.insert(0,str(d))
import a1h_hierarchical_matched as prev
import a1h_memory as unified_memory
import a0_a1_strict4_matched as a1
import bridge_memory as bridge
import tmra_audit_core as tmra

SCHEMA="tmra.audit1.v1"

def digest(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode("utf-8")).hexdigest()

def scores_to_metrics(y,s,theta):
    return a1.metrics(y,s,theta)

def with_retrieval(bank,score_fn,*,ltm,buffer):
    """Return read-only score; restore booleans even after failure."""
    prev_ltm,prev_buf=bank.enable_ltm_retrieval,bank.enable_maturation_buffer_retrieval
    try:
        bank.enable_ltm_retrieval=bool(ltm)
        bank.enable_maturation_buffer_retrieval=bool(buffer)
        return float(score_fn())
    finally:
        bank.enable_ltm_retrieval=prev_ltm
        bank.enable_maturation_buffer_retrieval=prev_buf

def main(args):
    import torch
    import p0_protocol_audit as p0
    import e02_matched_feedback as e02
    import e02_r6_sidecar_stability as r6

    if args.math_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    out=Path(args.out_dir)
    if out.exists():raise FileExistsError(f"Immutable output already exists: {out}")
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    cfg_br=bridge.BridgeConfig(stm=args.stm_capacity,evidence_mode=args.evidence_mode)
    impl=p0.require_sources()
    impl.set_seed(0)
    index=impl.RealIADJSONIndex(
        root=args.dataset_root,json_dir=args.json_dir,only_classes=[args.class_name])
    normals=sorted({s.path for s in index.train_samples
                    if s.cls_name==args.class_name and s.is_good})
    tests=sorted((s for s in index.test_samples if s.cls_name==args.class_name),
                 key=lambda x:x.path)
    if len(normals)<4 or len(tests)<args.probe_count+64:
        raise ValueError("Insufficient training normals or evaluation images")
    supports=sorted(random.Random(args.support_seed).sample(normals,4))
    a1.verify_supports(supports,normals,[x.path for x in tests])
    common=p0.Config(
        device=args.device,augmentation_seed=0,bank_seed=0,
        fixed_capacity=25000,fixed_per_image=5000,augmentations=4,
        include_identity=False,multiplier=2.5,feature_fuse="concat",
        topk_patches=32,image_aggregation="max",image_topk=64,
        image_quantile=.95,knn=5)
    backbone=impl.DINOv2MultiLayerBackbone(device=args.device)
    feature_cache=p0.encode_reference_features(impl,backbone,supports,common)
    fixed=p0.build_vmb(impl,feature_cache,supports,common)
    fixed_profile=p0.fixed_bank_profile(fixed)
    # Historical LOSO is a non-deployment diagnostic only.
    loso=[]
    folds=[]
    for withheld in supports:
        remainder=[z for z in supports if z!=withheld]
        fold=p0.build_vmb(impl,feature_cache,remainder,common)
        loso.append(p0.score_image(impl,backbone,fold,withheld,common))
        folds.append(dict(withheld=withheld,training=remainder,
                          bank_sha256=p0.fixed_bank_profile(fold)["sha256"]))
        del fold
    theta_stats=p0.threshold_from_scores(loso,2.5)
    del feature_cache
    source=list(tests)
    random.Random(0).shuffle(source)
    stream,probe=a1.select_probe(source,args.probe_count,args.probe_seed)
    stream=r6.order_stream(stream,args.stream_order)
    if args.max_samples:stream=stream[:args.max_samples]
    if len(stream)<32:raise ValueError("Too few stream images")
    if set(x.path for x in probe)&set(x.path for x in stream):
        raise AssertionError("Probe/stream leakage")
    tape=e02.make_query_tape(len(stream),.03,37)
    arms={name:bridge.BridgeArm(name,copy.deepcopy(fixed),cfg_br)
          for name in bridge.ARM_NAMES}
    for arm in arms.values():
        if p0.fixed_bank_profile(arm.bank)["sha256"]!=fixed_profile["sha256"]:
            raise AssertionError("Fixed reference mismatch")
    probe_cache=[]
    for s in probe:
        img,fts=p0.encode_query(impl,backbone,s.path,common)
        probe_cache.append((s.path,int(not s.is_good),
                            SimpleNamespace(height=img.height,width=img.width),
                            [v.detach().to("cpu",dtype=torch.float16) for v in fts]))

    def score(bank,img,fts):
        return float(a1.score_bank(impl,bank,img,fts,common)[0])

    labels=[]
    stream_rows=[]
    obs={name:[] for name in bridge.ARM_NAMES}
    probe_rows=[]
    probe_summary=[]
    memory_rows=[]
    intervention_rows=[]
    attribution_images=[]
    attribution_patches=[]
    checkpoints=[]
    queries_normal=queries_defect=0

    def checkpoint(t):
        bridge.assert_same_stm_and_ltm(arms)
        # Pure maturation tracking with retrieval off must not change STM
        # compared with the same-size flat utility-retention arm.
        import torch
        flat_cbs=bridge.memory_books(arms["flat_utility"].bank)[0]
        track_cbs=bridge.memory_books(arms["track_utility"].bank)[0]
        for x,y in zip(flat_cbs,track_cbs):
            if x.size()!=y.size() or not torch.equal(
                    x.protos[:x.size()],y.protos[:y.size()]):
                raise AssertionError("Maturation tracking altered STM unexpectedly")
        state_before={name:a.state() for name,a in arms.items()}
        for name,z in state_before.items():memory_rows.append({"step":t,"method":name,**z})
        p_scores={name:[] for name in bridge.ARM_NAMES}
        cf={variant:[] for variant in
            ("full","no_ltm","no_buffer","stm_only","unified_no_buffer")}
        for path,label,img,cached in probe_cache:
            feats=[v.to(args.device) for v in cached]
            for name,arm in arms.items():
                s=score(arm.bank,img,feats)
                p_scores[name].append((label,s))
                probe_rows.append(dict(step=t,path=path,label_eval_only=label,
                                       method=name,score=s))
            full=arms["full_utility"].bank
            def score_one():return score(full,img,feats)
            # Same-state unified global top-5 on fixed+STM+LTM
            # (without provisional buffer); NEVER affects online updates.
            original_dN=full._dN_min_fixed_dyn
            old_ltm,old_buffer=(
                full.enable_ltm_retrieval,full.enable_maturation_buffer_retrieval)
            try:
                full.enable_ltm_retrieval=True
                full.enable_maturation_buffer_retrieval=False
                unified_memory.install_unified_retrieval(
                    full,include_ltm=True,k=5)
                unified_score=score(full,img,feats)
            finally:
                full._dN_min_fixed_dyn=original_dN
                full.enable_ltm_retrieval=old_ltm
                full.enable_maturation_buffer_retrieval=old_buffer
            values={
                "unified_no_buffer":unified_score,
                "full":p_scores["full_utility"][-1][1],
                "no_ltm":with_retrieval(
                    full,score_one,ltm=False,buffer=True),
                "no_buffer":with_retrieval(
                    full,score_one,ltm=True,buffer=False),
                "stm_only":with_retrieval(
                    full,score_one,ltm=False,buffer=False),
            }
            for k,v in values.items():cf[k].append((label,v))
            four={"S0":values["stm_only"],"SL":values["no_buffer"],
                  "SB":values["no_ltm"],"SLB":values["full"]}
            attribution=tmra.shapley(four)
            attribution_images.append({
                "step":t,"path":path,"label_eval_only":label,
                **four,**attribution,
                "change_ltm_only":four["S0"]-four["SL"],
                "change_buffer_only":four["S0"]-four["SB"],
            })
            if abs(four["S0"]-four["SLB"])>1e-7:
                patches,sources=tmra.patch_source_details(
                    full,impl,feats,top_n=16)
                for patch in patches:
                    attribution_patches.append({
                        "step":t,"path":path,"label_eval_only":label,
                        **sources,**patch,
                    })
        for name,values in p_scores.items():
            result=scores_to_metrics(
                [z[0] for z in values],[z[1] for z in values],theta_stats["theta"])
            probe_summary.append({"step":t,"method":name,**result})
        baseline=[z[1] for z in cf["full"]]
        for variant,values in cf.items():
            y=[z[0] for z in values];s=[z[1] for z in values]
            diffs=[v-base for v,base in zip(s,baseline)]
            intervention_rows.append({
                "step":t,"variant":variant,"auroc":scores_to_metrics(
                    y,s,theta_stats["theta"])["auroc"],
                "score_changed":sum(abs(v)>1e-7 for v in diffs),
                "mean_score_delta_vs_full":sum(diffs)/len(diffs),
                "max_abs_score_delta":max(map(abs,diffs)),
            })
        checkpoints.append(t)

    checkpoint(0)
    for t,s in enumerate(stream,1):
        img,fts=p0.encode_query(impl,backbone,s.path,common)
        static_score,static_map=a1.score_bank(impl,fixed,img,fts,common)
        top=impl.layer_to_topk_from_amap(static_map)
        common_candidates={k:indices[:cfg_br.patches] for k,indices in top.items()}
        current={}
        for name,arm in arms.items():
            val=score(arm.bank,img,fts)
            current[name]=val
            obs[name].append(val)
        queried=t in tape
        truth=int(not s.is_good)
        labels.append(truth)
        stream_rows.append(dict(
            t=t,path=s.path,label_eval_only=truth,queried=int(queried),
            static_score=float(static_score),
            **{f"{name}_score":z for name,z in current.items()}))
        if queried:
            if truth==0:
                queries_normal+=1
                for name,arm in arms.items():
                    arm.observe_verified_normal(
                        impl,fts,common_candidates,step=t,score=static_score,
                        theta=theta_stats["theta"],sigma=theta_stats["sigma"])
            else:queries_defect+=1
        if args.probe_every and t%args.probe_every==0:checkpoint(t)
        if args.print_every and t%args.print_every==0:
            print(f"[BRIDGE] {args.class_name} seed={args.support_seed} "
                  f"STM={cfg_br.stm} t={t}/{len(stream)}",flush=True)
    if checkpoints[-1]!=len(stream):checkpoint(len(stream))
    if queries_normal+queries_defect!=len(tape):
        raise AssertionError("Query events mismatch")
    for name,arm in arms.items():
        if arm.events!=queries_normal:
            raise AssertionError(f"Feedback imbalance {name}")
        if p0.fixed_bank_profile(arm.bank)["sha256"]!=fixed_profile["sha256"]:
            raise AssertionError(f"Support bank modified: {name}")
    if len({a.patches_seen for a in arms.values()})!=1:
        raise AssertionError("Candidate-patch exposure mismatch")

    metrics={}
    for name,sc in obs.items():
        ps=[z for z in probe_summary if z["method"]==name]
        metrics[name]={
            **scores_to_metrics(labels,sc,theta_stats["theta"]),
            "probe_aulc":prev.aulc(ps),
            "probe_final_auroc":ps[-1]["auroc"],
            "probe_initial_auroc":ps[0]["auroc"],
            "memory":arms[name].state(),
        }
    def csv_write(filename,rows):
        if not rows:return
        columns=list(dict.fromkeys(k for z in rows for k in z))
        with (out/filename).open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=columns)
            writer.writeheader();writer.writerows(rows)
    for filename,rows in [
        ("prequential.csv",stream_rows),
        ("probe_scores.csv",probe_rows),
        ("probe_summary.csv",probe_summary),
        ("memory_snapshots.csv",memory_rows),
        ("retrieval_counterfactual.csv",intervention_rows),
        ("source_image_attribution.csv",attribution_images),
        ("source_patch_evidence.csv",attribution_patches),
    ]:csv_write(filename,rows)

    source_files=[
        "experiments/bridge_memory.py",
        "experiments/tmra_attribution.py",
        "experiments/tmra_audit_core.py",
        "experiments/a0_a1_strict4_matched.py",
        "experiments/a1h_hierarchical_matched.py",
        "experiments/p0_protocol_audit.py",
        "test_phase3_AIF_simplified_v22.py",
        "vmb_visionad_new.py",
    ]
    report={
        "attribution_summary":{
            "image_rows":len(attribution_images),
            "patch_rows":len(attribution_patches),
            "score_changed_image_rows":sum(
                abs(z["combined_score_reduction"])>1e-7
                for z in attribution_images),
            "nonzero_ltm_shapley":sum(abs(z["phi_ltm"])>1e-7
                                     for z in attribution_images),
            "nonzero_buffer_shapley":sum(abs(z["phi_buffer"])>1e-7
                                        for z in attribution_images),
        },
        "schema":SCHEMA,"complete":True,"class_name":args.class_name,
        "support_seed":args.support_seed,"stream_order":args.stream_order,
        "stm_capacity":cfg_br.stm,"ltm_capacity":cfg_br.ltm,
        "buffer_capacity":cfg_br.buffer,"total_hierarchical_capacity":cfg_br.total,
        "n_stream":len(stream),"n_probe":len(probe),
        "queried_normal":queries_normal,"queried_defect":queries_defect,
        "n_queries":len(tape),"arms":metrics,
        "counterfactual_probe_final":[
            z for z in intervention_rows if z["step"]==len(stream)],
        "protocol":{
            "four_shot_strict":True,
            "supports":supports,
            "support_hash":digest(supports),
            "loso_scores":loso,"loso_folds":folds,
            "loso_theta_diagnostic_only":theta_stats,
            "fixed_bank":fixed_profile,
            "stream_hash":digest([z.path for z in stream]),
            "probe_paths":[z.path for z in probe],
            "probe_hash":digest([z.path for z in probe]),
            "query_positions":sorted(tape),
            "query_hash":digest(sorted(tape)),
            "evidence_mode":cfg_br.evidence_mode,
            "candidate_patches_per_feedback":cfg_br.patches,
            "historical_native_retrieval":True,
            "historical_promotion_min_events":cfg_br.min_events,
            "historical_promotion_period":cfg_br.promote_every,
            "same_feedback_and_candidate_identity":True,
            "probe_evaluation_only":True,
        },
        "elapsed_seconds":time.perf_counter()-started,
        "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                         for p in source_files},
        "limits":[
            "Patch source_winner is a feature-space distance winner, not a spatial semantic explanation.",
            "A lower anomaly score rescues normals but can also conceal defects; labels are evaluation-only.",
            "Historical-to-matched bridge, NOT a recreation of original full BEA+sidecar+correction experiment.",
            "Flat_equal_total matches total candidate prototype slots, not actual bytes or compute.",
            "H_full vs H_ltm is coupled with buffer-coverage-dependent utility retention; read-only checkpoint counterfactual isolates direct retrieval.",
            "Neutral evidence bypasses historically threshold-dependent utility; optional loso_margin is sensitivity, not valid calibration.",
            "Original pool-assisted support selection is deliberately replaced with strict four-shot.",
            "No pixel-level or actual A-B-A recurrence benchmark.",
        ],
    }
    (out/"summary.json").write_text(
        json.dumps(report,indent=2,allow_nan=False,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({
        "condition":f"{args.class_name}/s{args.support_seed}/STM{cfg_br.stm}",
        "probe_AULC":{k:round(v["probe_aulc"],5)
                      if v["probe_aulc"] is not None else None
                      for k,v in metrics.items()},
        "ltm":{k:v["memory"]["ltm"] for k,v in metrics.items()},
        "buffer":{k:v["memory"]["buffer"] for k,v in metrics.items()},
        "final_retrieval_cf":report["counterfactual_probe_final"],
        "out_dir":str(out)},indent=2),flush=True)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",default="G:/dataset/realiad")
    p.add_argument("--json-dir",default=None)
    p.add_argument("--class-name",default="audiojack")
    p.add_argument("--support-seed",type=int,default=0)
    p.add_argument("--stm-capacity",type=int,choices=(512,768),default=512)
    p.add_argument("--stream-order",choices=("native","normal_first","defect_first"),default="native")
    p.add_argument("--evidence-mode",choices=("neutral","loso_margin"),default="neutral")
    p.add_argument("--probe-count",type=int,default=96)
    p.add_argument("--probe-seed",type=int,default=212)
    p.add_argument("--probe-every",type=int,default=1024)
    p.add_argument("--max-samples",type=int,default=0)
    p.add_argument("--print-every",type=int,default=250)
    p.add_argument("--math-sdpa",action="store_true")
    p.add_argument("--device",default="cuda")
    p.add_argument("--out-dir",required=True)
    return p

if __name__=="__main__":
    a=parser().parse_args()
    if a.json_dir is None:
        a.json_dir=str(Path(a.dataset_root)/"jsons"/"realiad_jsons")
    if not 8<=a.probe_count<=512 or a.probe_every<=0 or a.max_samples<0:
        raise SystemExit("Invalid probe count/interval or sample limit")
    main(a)
