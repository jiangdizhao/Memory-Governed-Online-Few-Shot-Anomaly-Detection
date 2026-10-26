#!/usr/bin/env python3
"""TMRA-II: strict-4-shot, byte-capped A-B-A vs A-A-A normal-mode return.

Controlled *synthetic color/illumination covariate shift*, NOT real plant
operating modes. Ground truth is read only for retrospective evaluation and
verified feedback at pre-committed 3% positions.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for d in (ROOT,ROOT/"experiments"):
    if str(d) not in sys.path:sys.path.insert(0,str(d))
import tmra_recurrence_core as core
import tmra_compact_memory as memory
import a0_a1_strict4_matched as a1

SCHEMA="tmra.recurrence.v1"

def main(args):
    import torch
    import numpy as np
    import p0_protocol_audit as p0
    import e02_matched_feedback as e02
    if args.math_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    config=core.Settings(phase_length=args.phase_length,
                         probe_count=args.probe_count)
    out=Path(args.out_dir)
    if out.exists():raise FileExistsError(f"Refuse to overwrite: {out}")
    out.mkdir(parents=True)
    start=time.perf_counter()
    impl=p0.require_sources()
    impl.set_seed(0)
    index=impl.RealIADJSONIndex(
        root=args.dataset_root,json_dir=args.json_dir,
        only_classes=[args.class_name])
    train=sorted({x.path for x in index.train_samples
                  if x.cls_name==args.class_name and x.is_good})
    test=sorted((x for x in index.test_samples
                 if x.cls_name==args.class_name),key=lambda x:x.path)
    supports=sorted(random.Random(args.seed).sample(train,4))
    a1.verify_supports(supports,train,[x.path for x in test])
    Aprobe,Bprobe,phases=core.label_blind_partition(test,config,
                                                   seed=args.seed+2201)
    stream=sum(phases,[])
    phase_names=core.phase_modes(args.order)
    tape=e02.make_query_tape(len(stream),.03,37)
    p_cfg=p0.Config(
        device=args.device,augmentation_seed=0,bank_seed=0,
        fixed_capacity=25000,fixed_per_image=5000,
        augmentations=4,include_identity=False,
        multiplier=2.5,feature_fuse="concat",topk_patches=32,
        image_aggregation="max",image_topk=64,image_quantile=.95,knn=5)
    backbone=impl.DINOv2MultiLayerBackbone(device=args.device)
    support_cache=p0.encode_reference_features(
        impl,backbone,supports,p_cfg)
    frozen=p0.build_vmb(impl,support_cache,supports,p_cfg)
    bank_profile=p0.fixed_bank_profile(frozen)
    del support_cache

    arms={name:memory.Arm(
        name,copy.deepcopy(frozen),config,seed=args.seed+37)
        for name in memory.NAMES if name!="static"}
    arms["static"]=None
    for arm in arms.values():
        if arm is not None and p0.fixed_bank_profile(arm.bank)["sha256"]!=bank_profile["sha256"]:
            raise AssertionError("Fixed bank mismatch")

    def encoded(path,mode):
        img=impl.load_rgb(path)
        img=core.transform_image(img,mode)
        ft,_=impl.encode_pil(backbone,img,p_cfg.feature_fuse)
        return img,ft

    def score(bank,img,feat):
        return a1.score_bank(impl,bank,img,feat,p_cfg)

    probes={}
    for mod,pool in (("A",Aprobe),("B",Bprobe)):
        cache=[]
        for item in pool:
            img,fts=encoded(item.path,mod)
            cache.append((item.path,int(not item.is_good),
                          (img.height,img.width),
                          [v.detach().to("cpu",dtype=torch.float16)
                           for v in fts]))
        if len({y for _,y,_,_ in cache})!=2:
            raise AssertionError(f"Unbalanced evaluation probe: {mod}")
        probes[mod]=cache

    probe_rows=[]
    stream_rows=[]
    memory_rows=[]
    phase_stats=[]
    recovery=[]
    qn=qd=0
    query_by_phase=[0,0,0]
    shift_deltas=[]

    def eval_probe(t,mode,checkpoint,new_A_normal_labels=0):
        result={}
        for name,arm in arms.items():
            bank=frozen if arm is None else arm.bank
            y,s=[],[]
            for path,label,hw,cached in probes[mode]:
                feats=[z.to(args.device) for z in cached]
                img=type("ImageShape",(),{"height":hw[0],"width":hw[1]})()
                val,_=score(bank,img,feats)
                y.append(label);s.append(val)
                probe_rows.append(dict(step=t,checkpoint=checkpoint,
                                       probe_mode=mode,path=path,
                                       label_eval_only=label,method=name,score=val,
                                       new_A_normal_labels=new_A_normal_labels))
            auc=p0.metrics(y,s,theta=1e9)["auroc"]
            normal_mean=float(np.mean([v for v,yy in zip(s,y) if yy==0]))
            defect_mean=float(np.mean([v for v,yy in zip(s,y) if yy==1]))
            result[name]={"auroc":auc,"normal_score_mean":normal_mean,
                          "defect_score_mean":defect_mean}
            if arm is not None:
                memory_rows.append(dict(step=t,checkpoint=checkpoint,
                                        method=name,**arm.audit()))
        return result

    initial=eval_probe(0,"A","initial")
    eval_probe(0,"B","initial")
    phase_end={}
    # All arms see the same immutable static candidate policy per input image.
    for phase_i,(phase,mode) in enumerate(zip(phases,phase_names)):
        if phase_i==2:
            # This must run *before* first Phase-III input/label.
            entry=eval_probe(2*config.phase_length,"A","return_A_pre_feedback",0)
            phase_end["pre_return_A"]=entry
            recovery.append({"labels":0,"metrics":entry,"step":2*config.phase_length})
        for local_i,item in enumerate(phase,1):
            t=phase_i*config.phase_length+local_i
            img,feat=encoded(item.path,mode)
            if phase_i==1 and mode=="B" and local_i<=6:
                _,a_feat=encoded(item.path,"A")
                xa=impl.l2_normalize(a_feat[0].float())
                xb=impl.l2_normalize(feat[0].float())
                if xa.shape!=xb.shape:
                    raise AssertionError("A/B patch geometry changed")
                delta=float((1-(xa*xb).sum(dim=1)).mean().item())
                shift_deltas.append(delta)
            static_score,static_map=score(frozen,img,feat)
            candidate={k:v[:24] for k,v in
                       impl.layer_to_topk_from_amap(static_map).items()}
            values={"static":static_score}
            for name,arm in arms.items():
                if arm is not None:
                    values[name]=score(arm.bank,img,feat)[0]
            queried=t in tape
            label=int(not item.is_good)
            stream_rows.append(dict(
                t=t,phase=phase_i+1,mode=mode,path=item.path,
                label_eval_only=label,queried=int(queried),
                **{f"{k}_score":v for k,v in values.items()}))
            if queried:
                if label==0:
                    qn+=1;query_by_phase[phase_i]+=1
                    for arm in arms.values():
                        if arm is not None:arm.update(impl,feat,candidate,t)
                    # Save *post-label* reacquisition curve. No oracle labels
                    # control any model changes or query positions.
                    if phase_i==2:
                        post=eval_probe(t,"A","A_feedback",query_by_phase[2])
                        recovery.append({"labels":query_by_phase[2],
                                         "metrics":post,"step":t})
                else:qd+=1
            if args.print_every and t%args.print_every==0:
                print(f"[TMRA-II] {args.class_name} s{args.seed} {args.order} "
                      f"{t}/{len(stream)}",flush=True)
        finish=(phase_i+1)*config.phase_length
        phase_end[f"end_{phase_i+1}"]=eval_probe(
            finish,"A",f"end_{phase_i+1}")
        # Diagnostic B probe is not used as training/selection evidence.
        if phase_i==1:
            phase_end["B_probe_end_2"]=eval_probe(
                finish,"B","B_end_2")
    if args.order=="ABA" and (not shift_deltas or max(shift_deltas)<1e-4):
        raise AssertionError(
            "Synthetic B mode produced negligible frozen-feature shift")
    if qn+qd!=len(tape):
        raise AssertionError("Precommitted tape count mismatch")
    for name,arm in arms.items():
        if arm is None:continue
        if arm.events!=qn:
            raise AssertionError(f"Unequal human feedback events {name}")
        if p0.fixed_bank_profile(arm.bank)["sha256"]!=bank_profile["sha256"]:
            raise AssertionError("Fixed bank was altered")
    if len({arm.patches for arm in arms.values() if arm is not None})!=1:
        raise AssertionError("Candidate patch mismatch")
    results={}
    for name in memory.NAMES:
        aend=phase_end["end_1"][name]
        bend=phase_end["end_2"][name]
        entry=phase_end["pre_return_A"][name]
        after=phase_end["end_3"][name]
        recover_labels=core.first_recovery(
            [c["metrics"][name]["auroc"] for c in recovery],
            [c["labels"] for c in recovery],aend["auroc"],tolerance=.01)
        results[name]={
            "A_probe_end_I_auc":aend["auroc"],
            "A_probe_end_II_auc":bend["auroc"],
            "A_probe_return_pre_feedback_auc":entry["auroc"],
            "A_probe_end_III_auc":after["auroc"],
            "retention_forgetting_auc":aend["auroc"]-bend["auroc"],
            "return_normal_mean_score_shift":
                entry["normal_score_mean"]-aend["normal_score_mean"],
            "reacquisition_normal_labels":recover_labels,
            "reacquisition_target_auc":aend["auroc"]-.01,
            "A_normal_scores":{k:phase_end[k][name]["normal_score_mean"]
                               for k in ("end_1","end_2","end_3")},
            "memory":arms[name].audit() if arms[name] else None,
        }
    def write_csv(path,rows):
        if not rows:return
        keys=list(dict.fromkeys(k for x in rows for k in x))
        with (out/path).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    for f,rows in (("probes.csv",probe_rows),
                   ("prequential.csv",stream_rows),
                   ("memory_snapshots.csv",memory_rows)):
        write_csv(f,rows)
    reacq_rows=[]
    for snap in recovery:
        for name,r in snap["metrics"].items():
            reacq_rows.append({"step":snap["step"],"method":name,
                "new_A_normal_labels":snap["labels"],**r})
    write_csv("reacquisition.csv",reacq_rows)

    sources=["experiments/tmra_recurrence.py",
             "experiments/tmra_recurrence_core.py",
             "experiments/tmra_compact_memory.py",
             "experiments/a0_a1_memory_methods.py",
             "experiments/a0_a1_strict4_matched.py",
             "experiments/p0_protocol_audit.py",
             "test_phase3_AIF_simplified_v22.py",
             "vmb_visionad_new.py"]
    report={
        "schema":SCHEMA,"complete":True,
        "condition":{"class_name":args.class_name,"seed":args.seed,
                     "order":args.order,"phase_length":config.phase_length,
                     "probe_count":config.probe_count},
        "manifest":{
            "supports":supports,"support_sha":core.hash_manifest(supports),
            "phase_modes":phase_names,
            "phase_stream_paths":[[x.path for x in phase] for phase in phases],
            "phase_stream_sha":core.hash_manifest(
                [[x.path for x in phase] for phase in phases]),
            "A_probe_paths":[x.path for x in Aprobe],
            "B_probe_paths":[x.path for x in Bprobe],
            "query_positions":sorted(tape),
            "query_sha":core.hash_manifest(sorted(tape)),
            "q_normal":qn,"q_defect":qd,
            "q_normal_by_phase":query_by_phase,
            "fixed_bank":bank_profile,
            "synthetic_shift_audit_mean_cosine_distance":(
                sum(shift_deltas)/len(shift_deltas) if shift_deltas else None),
            "synthetic_shift_mode_B":{
                "rgb_gains":[.84,.94,1.08],"rgb_offset":8.,
                "label_preserving_assumption_not_guaranteed":True},
            "byte_ceiling":config.cap_bytes,
            "equal_ceiling_not_equal_allocated_bytes":True,
            "eval_probes_disjoint_from_all_stream_images":True,
            "query_labels_only_after_pred":True,
        },
        "results":results,
        "elapsed_seconds":time.perf_counter()-start,
        "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                         for p in sources},
        "limitations":[
            "Synthetic color/illumination change is not a verified factory mode.",
            "The A-probe is reused for evaluation across checkpoints, never learning.",
            "Labels are accessed for evaluation and only queried events update normal memory.",
            "Observed recovery counts are within the fixed tape and may be censored.",
            "Memory arms have common upper physical mutable-tensor byte ceiling, not equal byte usage.",
            "Mode A equals the original image; Fixed Bank has not been domain shifted.",
            "This tests an existing hierarchical mechanism, not a novel risk-aware deletion algorithm.",
        ]}
    (out/"summary.json").write_text(json.dumps(
        report,indent=2,allow_nan=False,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({
        "condition":report["condition"],
        "results":{name:{
            "forgetting":round(v["retention_forgetting_auc"],5),
            "recovery_labels":v["reacquisition_normal_labels"],
            "ltm":v["memory"]["ltm"] if v["memory"] else None,
        } for name,v in results.items()},
        "normal_queries":qn,"defect_queries":qd,"output":str(out)
    },indent=2),flush=True)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",default="G:/dataset/realiad")
    p.add_argument("--json-dir",default=None)
    p.add_argument("--class-name",default="switch")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--order",choices=("ABA","AAA"),default="ABA")
    p.add_argument("--phase-length",type=int,default=900)
    p.add_argument("--probe-count",type=int,default=96)
    p.add_argument("--device",default="cuda")
    p.add_argument("--math-sdpa",action="store_true")
    p.add_argument("--print-every",type=int,default=300)
    p.add_argument("--out-dir",required=True)
    return p

if __name__=="__main__":
    args=parser().parse_args()
    if args.json_dir is None:
        args.json_dir=str(Path(args.dataset_root)/"jsons"/"realiad_jsons")
    main(args)
