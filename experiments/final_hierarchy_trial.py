#!/usr/bin/env python3
"""Final, frozen, two-arm hierarchy-vs-greedy decision on Real-IAD.

One class/seed/order per GPU process. No parameter sweep and no data-driven
switching of parameters. User must run through run_final_hierarchy_decision.py.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
for v in (ROOT,ROOT/"experiments"):
    if str(v) not in sys.path:sys.path.insert(0,str(v))
import final_decision_core as final
import tmra_compact_memory as memory
import tmra_recurrence_core as shift
import a0_a1_strict4_matched as a1
import tmra_audit_core as read_cf

NAMES=("greedy_coreset","hierarchy")

def main(args):
    import torch
    import numpy as np
    import p0_protocol_audit as p0
    if args.math_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    if args.class_name not in final.CLASSES or args.seed not in final.SEEDS or args.order not in final.ORDERS:
        raise ValueError("Trial identity outside frozen 2 x 2 x 2 matrix")
    out=Path(args.out_dir)
    if out.exists():raise FileExistsError(f"Immutable evidence directory exists: {out}")
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    impl=p0.require_sources();impl.set_seed(0)
    idx=impl.RealIADJSONIndex(
        root=args.dataset_root,json_dir=args.json_dir,
        only_classes=[args.class_name])
    train=[x for x in idx.train_samples if x.cls_name==args.class_name]
    test=[x for x in idx.test_samples if x.cls_name==args.class_name]
    support=final.choose_supports(train,args.seed)
    probes,phases,objects=final.make_split(test,support,args.seed)
    stream=sum(phases,[])
    assert not (set(support)&{s.path for s in stream})
    assert not (set(support)&{s.path for s in sum(probes.values(),[])})
    query=final.query_tape(phases)
    assert abs(len(query)/len(stream)-final.QUERY_RATE)<.001

    cfg=p0.Config(
        device=args.device,augmentation_seed=0,bank_seed=0,
        fixed_capacity=25000,fixed_per_image=5000,
        augmentations=4,include_identity=False,multiplier=2.5,
        feature_fuse="concat",topk_patches=32,
        image_aggregation="max",image_topk=64,image_quantile=.95,knn=5)
    backbone=impl.DINOv2MultiLayerBackbone(device=args.device)
    cache=p0.encode_reference_features(impl,backbone,support,cfg)
    fixed=p0.build_vmb(impl,cache,support,cfg)
    profile=p0.fixed_bank_profile(fixed)
    loso=[]
    for heldout in support:
        allowed=[p for p in support if p!=heldout]
        if final.object_id(heldout) in {final.object_id(p) for p in allowed}:
            raise AssertionError("LOSO same-instance self-match")
        fold=p0.build_vmb(impl,cache,allowed,cfg)
        loso.append(p0.score_image(impl,backbone,fold,heldout,cfg))
    theta=p0.threshold_from_scores(loso,2.5)["theta"]
    del cache

    conf=final.MemorySettings()
    arms={name:memory.Arm(name,copy.deepcopy(fixed),conf,seed=args.seed+37)
          for name in NAMES}
    for arm in arms.values():
        if p0.fixed_bank_profile(arm.bank)["sha256"]!=profile["sha256"]:
            raise AssertionError("Static reference mismatch")

    def encode(path,mode):
        img=shift.transform_image(impl.load_rgb(path),mode)
        feats,_=impl.encode_pil(backbone,img,cfg.feature_fuse)
        return img,feats

    def score(bank,img,feat):
        return a1.score_bank(impl,bank,img,feat,cfg)[0]

    probe_cached={}
    for mode,group in probes.items():
        cache=[]
        for s in group:
            img,feats=encode(s.path,mode)
            cache.append((s.path,final.object_id(s.path),
                          int(not s.is_good),(img.height,img.width),
                          [x.detach().to("cpu",dtype=torch.float16) for x in feats]))
        labels_by_obj={}
        for _,obj,y,_,_ in cache:
            labels_by_obj[obj]=max(labels_by_obj.get(obj,0),y)
        if set(labels_by_obj.values())!={0,1}:
            raise ValueError(f"Probe {mode} lacks both normal and anomalous physical units")
        probe_cached[mode]=cache

    probe_records=[];phase_records=[];memory_records=[]
    recovery=[]
    q_normal=0;q_defect=0
    qnormal_per_phase=[0,0,0]
    updates={}
    shift_values=[]

    def probe(t,mode,stage,store=True,flags=None):
        result={}
        for name,arm in arms.items():
            if flags is not None and name!="hierarchy":continue
            truth=[];scores=[]
            per_obj={}
            for path,obj,y,hw,cached in probe_cached[mode]:
                img=SimpleNamespace(height=hw[0],width=hw[1])
                feats=[v.to(args.device) for v in cached]
                if flags is not None:
                    with read_cf.source_flags(arm.bank,*flags):
                        val=score(arm.bank,img,feats)
                else:
                    val=score(arm.bank,img,feats)
                truth.append(y);scores.append(val)
                row=per_obj.setdefault(obj,{"y":0,"max":float("-inf")})
                row["y"]=max(row["y"],y)
                row["max"]=max(row["max"],val)
                if store:probe_records.append({
                    "step":t,"stage":stage,"mode":mode,"method":name,
                    "object_id":obj,"path":path,"label_eval_only":y,
                    "score":val,"flags":str(flags)})
            obj_y=[v["y"] for v in per_obj.values()]
            obj_score=[v["max"] for v in per_obj.values()]
            result[name]={
                "object_auroc":final.auroc(obj_y,obj_score),
                "image_auroc":final.auroc(truth,scores),
                "object_count":len(obj_y),
                "normal_image_mean":float(np.mean(
                    [s for s,y in zip(scores,truth) if not y])),
                "fpr_loso":float(np.mean(
                    [s>=theta for s,y in zip(scores,truth) if not y])),
                "fnr_loso":float(np.mean(
                    [s<theta for s,y in zip(scores,truth) if y])),
            }
            if store:memory_records.append({
                "step":t,"stage":stage,"method":name,**arm.audit()})
        return result

    initial_A=probe(0,"A","initial")
    stage={}
    global_t=0
    modes=final.phase_modes(args.order)
    for phase_idx,(samples,mode) in enumerate(zip(phases,modes)):
        if phase_idx==2:
            stage["return_pre_feedback_A"]=probe(global_t,"A","return_A_pre_feedback")
            recovery.append((0,stage["return_pre_feedback_A"]))
        for local,s in enumerate(samples,1):
            global_t+=1
            img,feats=encode(s.path,mode)
            if phase_idx==1 and mode=="B" and local<=6:
                _,alt=encode(s.path,"A")
                q=impl.l2_normalize(feats[0].float())
                p=impl.l2_normalize(alt[0].float())
                if q.shape!=p.shape:raise AssertionError("Mode changed patch geometry")
                shift_values.append(float((1-(q*p).sum(dim=1)).mean().item()))
            static_score,static_map=a1.score_bank(impl,fixed,img,feats,cfg)
            candidates={k:v[:24] for k,v in
                        impl.layer_to_topk_from_amap(static_map).items()}
            vals={name:score(arm.bank,img,feats) for name,arm in arms.items()}
            chosen=global_t in query
            label=int(not s.is_good)
            phase_records.append({
                "t":global_t,"phase":phase_idx+1,"mode":mode,
                "object_id":final.object_id(s.path),
                "path":s.path,"label_eval_only":label,
                "queried":int(chosen),"static_score":static_score,
                **{f"{n}_score":v for n,v in vals.items()},
            })
            if chosen:
                if label==0:
                    q_normal+=1;qnormal_per_phase[phase_idx]+=1
                    for arm in arms.values():
                        arm.update(impl,feats,candidates,global_t)
                    if phase_idx==2:
                        z=probe(global_t,"A","A_feedback")
                        recovery.append((qnormal_per_phase[2],z))
                else:q_defect+=1
            if args.print_every and global_t%args.print_every==0:
                print(f"[FINAL] {args.class_name}/s{args.seed}/{args.order} "
                      f"{global_t}/{len(stream)}",flush=True)
        stage[f"A_end_{phase_idx+1}"]=probe(global_t,"A",f"A_end_{phase_idx+1}")
        if phase_idx==1:
            stage["B_end_2"]=probe(global_t,"B","B_end_2")
            # Read-only, exactly same update state; intervention is NOT a
            # separate learning arm and adds no new human queries.
            stage["A_no_extra_end_2"]=probe(
                global_t,"A","A_read_no_LTM_Buffer",store=False,flags=(False,False))
            stage["A_no_LTM_end_2"]=probe(
                global_t,"A","A_read_no_LTM",store=False,flags=(False,True))
            stage["A_no_Buffer_end_2"]=probe(
                global_t,"A","A_read_no_Buffer",store=False,flags=(True,False))
        if args.order=="ABA" and phase_idx==1:
            if not shift_values or np.mean(shift_values)<.01:
                raise AssertionError("Frozen B shift failed feature-space verification")
    if len(query)!=q_normal+q_defect:raise AssertionError("Query count mismatch")
    for arm in arms.values():
        if arm.events!=q_normal:raise AssertionError("Different normal supervision")
        if p0.fixed_bank_profile(arm.bank)["sha256"]!=profile["sha256"]:
            raise AssertionError("Support bank changed")
    if arms["greedy_coreset"].patches!=arms["hierarchy"].patches:
        raise AssertionError("Candidate patch exposure differs")

    a1auc={name:stage["A_end_1"][name]["object_auroc"] for name in NAMES}
    ret={}
    for name,arm in arms.items():
        rec=[(k,z[name]["object_auroc"]) for k,z in recovery]
        ret[name]={
            "A_end_1":stage["A_end_1"][name],
            "A_end_2":stage["A_end_2"][name],
            "A_end_3":stage["A_end_3"][name],
            "B_end_2":stage["B_end_2"][name],
            "return_pre_feedback":stage["return_pre_feedback_A"][name],
            "forgetting":a1auc[name]-stage["A_end_2"][name]["object_auroc"],
            "reacquisition_labels":final.recovery_count(rec,a1auc[name]),
            "memory":arm.audit(),
        }
    delta_bytes={}
    for at in ("A_end_1","A_end_2","A_end_3"):
        a=[r for r in memory_records if r["stage"]==at]
        memory_sizes={x["method"]:x["allocated_state_bytes"] for x in a}
        b1,b2=(memory_sizes[z] for z in NAMES)
        gap=abs(b1-b2)/max(b1,b2)
        if gap>.05:
            raise AssertionError(f"Actual dynamic memory byte mismatch at {at}: {memory_sizes}")
        delta_bytes[at]={"bytes":memory_sizes,"relative_difference":gap}
    for name,arm in arms.items():
        if arm.audit()["allocated_state_bytes"]>conf.cap_bytes:
            raise AssertionError("Exceeded dynamic state byte ceiling")

    for filename,records in (
        ("probe_scores.csv",probe_records),
        ("stream.csv",phase_records),
        ("memory_snapshots.csv",memory_records),
    ):
        if not records:continue
        with (out/filename).open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]))
            w.writeheader();w.writerows(records)
    recovery_rows=[]
    for count,record in recovery:
        for name,values in record.items():
            recovery_rows.append({
                "new_phaseIII_normal_labels":count,"method":name,
                "object_auroc":values["object_auroc"],
                "image_auroc":values["image_auroc"]})
    with (out/"recovery.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(recovery_rows[0]))
        w.writeheader();w.writerows(recovery_rows)

    input_files=[
        "experiments/final_decision_core.py",
        "experiments/final_hierarchy_trial.py",
        "experiments/tmra_compact_memory.py",
        "experiments/tmra_recurrence_core.py",
        "experiments/tmra_audit_core.py",
        "experiments/a0_a1_memory_methods.py",
        "experiments/a0_a1_strict4_matched.py",
        "experiments/p0_protocol_audit.py",
        "test_phase3_AIF_simplified_v22.py",
        "vmb_visionad_new.py",
    ]
    report={
        "schema":final.SCHEMA,"complete":True,
        "class_name":args.class_name,"seed":args.seed,"order":args.order,
        "support_paths":support,"support_objects":[final.object_id(p) for p in support],
        "group_manifest":objects,
        "support_digest":final.digest(support),
        "group_digest":final.digest(objects),
        "stream_paths_digest":final.digest([[s.path for s in phase] for phase in phases]),
        "stream_paths":[[s.path for s in phase] for phase in phases],
        "probe_A_paths":[s.path for s in probes["A"]],
        "probe_B_paths":[s.path for s in probes["B"]],
        "query_positions":sorted(query),
        "query_digest":final.digest(sorted(query)),
        "queried_normal":q_normal,"queried_defect":q_defect,
        "queried_normal_per_phase":qnormal_per_phase,
        "n_stream":len(stream),"n_probe_A_images":len(probes["A"]),
        "n_probe_B_images":len(probes["B"]),
        "n_probe_A_objects":len(objects["A"]),
        "n_probe_B_objects":len(objects["B"]),
        "loso_theta_diagnostic_only":theta,"support_loso_scores":loso,
        "fixed_bank_profile":profile,
        "same_state_read_only":{
            "full":stage["A_end_2"]["hierarchy"],
            "no_extra":stage["A_no_extra_end_2"]["hierarchy"],
            "no_ltm":stage["A_no_LTM_end_2"]["hierarchy"],
            "no_buffer":stage["A_no_Buffer_end_2"]["hierarchy"],
            "ltm_count_end_II":next(
                r["ltm"] for r in memory_records
                if r["stage"]=="A_end_2" and r["method"]=="hierarchy")},
        "results":ret,"actual_bytes_by_phase":delta_bytes,
        "mean_B_feature_cosine_shift":(
            sum(shift_values)/len(shift_values) if shift_values else None),
        "baseline_comparability":{
            "strict_four_shot":True,
            "independent_physical_objects_across_all_stages":True,
            "identical_supports_stream_query_and_candidates":True,
            "fixed_native_v22_promotion_and_retrieval":True,
            "human_query_rate":final.QUERY_RATE,
            "one_query_at_most_per_physical_object":True,
            "equal_actual_mutable_tensor_bytes_within_5_percent":True,
            "original_paper_full_stack_reproduction":False,
        },
        "source_sha256":{
            p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
            for p in input_files},
        "elapsed_seconds":time.perf_counter()-started,
    }
    (out/"summary.json").write_text(
        json.dumps(report,indent=2,allow_nan=False,ensure_ascii=False),
        encoding="utf-8")
    print(json.dumps({
        "case":f"{args.class_name}/s{args.seed}/{args.order}",
        "queried_normal":q_normal,
        "hierarchy_ltm":ret["hierarchy"]["memory"]["ltm"],
        "object_end2_auc":{n:ret[n]["A_end_2"]["object_auroc"] for n in NAMES},
        "bytes_end2":delta_bytes["A_end_2"]["bytes"],
        "saved_to":str(out),
    },indent=2),flush=True)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",required=True)
    p.add_argument("--json-dir")
    p.add_argument("--class-name",choices=final.CLASSES,required=True)
    p.add_argument("--seed",type=int,choices=final.SEEDS,required=True)
    p.add_argument("--order",choices=final.ORDERS,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--math-sdpa",action="store_true")
    p.add_argument("--print-every",type=int,default=350)
    p.add_argument("--out-dir",required=True)
    return p

if __name__=="__main__":
    args=parser().parse_args()
    if args.json_dir is None:
        args.json_dir=str(Path(args.dataset_root)/"jsons"/"realiad_jsons")
    main(args)
