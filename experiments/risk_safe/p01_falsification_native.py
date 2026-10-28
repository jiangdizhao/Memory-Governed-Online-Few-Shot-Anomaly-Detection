#!/usr/bin/env python3
"""P0/P1 falsification test: normal-only recalibration vs simple memory admission.

P0: original frozen-static q90 threshold vs separately, normal-only q90
    threshold for EVERY final bank. No Probe A/B labels are used in calibration.
P1: 22-event matched, retrospective first/hash/frozen-novelty/static-score
    versus frozen R1 Beam-oracle decisions; original causal-median novelty as
    different-count deployment-feasible comparator.
All candidate patches and Greedy updates are native v22, fixed source scoring.
This is final-state Probe-A development diagnostic, NOT prequential or Probe-B.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

from . import preflight as pf
from . import d1b_protocol as d1b
from . import d1d_native_smoke as d1d
from . import d2_baselines as d2
from .d2_fixed_threshold import fit_fixed_normal_thresholds
from .native_bridge import (dynamic_codebooks, extract_v22_candidates,
                            normalized_native_candidate_rows, native_score,
                            snapshot, install_on_clone)
from .r0_opportunity import frozen_r0_tape
from .r1_oracle_core import DecisionState, transition
from .p01_falsification_core import (
    FalsificationProtocolError, select_matched, select_oracle_bits,
    metrics, paired_changes, normal_q90)
from .r1_native_oracle import ROOT, p0, v02, memory

SCHEMA = "risk_safe.p01.falsification.v1"


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",default=pf.DEFAULT_DATASET_ROOT)
    p.add_argument("--json-dir",default=None)
    p.add_argument("--class-name",choices=d1b.DEV_CLASSES,default="audiojack")
    p.add_argument("--support-seed",choices=d1b.SUPPORT_SEEDS,type=int,default=0)
    p.add_argument("--query-seed",choices=d1b.QUERY_SEEDS,type=int,default=37)
    p.add_argument("--budget",choices=tuple(sorted(set(d1b.MAIN_BUDGETS+
                               d1b.SENSITIVITY_BUDGETS))),type=int,default=128)
    p.add_argument("--capacity",type=int,default=256)
    p.add_argument("--target-admissions",type=int,default=22)
    p.add_argument("--hash-seed",type=int,default=1701)
    p.add_argument("--max-query-events",type=int,default=0,
                   help="0 full; nonzero is an INCOMPLETE smoke prefix")
    p.add_argument("--max-probe-groups-per-type",type=int,default=24,
                   help="24 full Probe A; lower is an INCOMPLETE smoke")
    p.add_argument("--oracle-summary",default=None,
                   help="R1 hindsight decisions (not an online policy)")
    p.add_argument("--r0-reference-summary",default=None,
                   help="Required for meaningful B128 parity gate")
    p.add_argument("--d1b-manifest-dir",default=d1d.DEFAULT_D1B)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--math-sdpa",action="store_true")
    p.add_argument("--out-dir",default=None)
    p.add_argument("--verify-only",action="store_true")
    return p


def output_dir(args):
    if args.out_dir:
        return Path(args.out_dir)
    suffix=(f"smokeq{args.max_query_events}_probe{args.max_probe_groups_per_type}"
            if args.max_query_events or args.max_probe_groups_per_type<24 else "full")
    return Path("outputs/p01_falsification_20261028") / (
        f"{args.class_name}_s{args.support_seed}_q{args.query_seed}"
        f"_b{args.budget}_cap{args.capacity}_k{args.target_admissions}"
        f"_h{args.hash_seed}_{suffix}")


def write_csv(path,rows):
    if not rows:
        raise FalsificationProtocolError("Refusing empty CSV artifact")
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+"\n",
                    encoding="utf-8",newline="\n")


def load_oracle(path, *, tape_sha, n_normal, target, capacity, args):
    summary=json.loads(Path(path).read_text(encoding="utf-8"))
    cfg=summary["config"]
    expected=(args.class_name,args.support_seed,args.query_seed,args.budget)
    actual=(cfg["class_name"],cfg["support_seed"],cfg["query_seed"],cfg["budget"])
    if actual!=expected or cfg["max_normal_events"]!=0:
        raise FalsificationProtocolError("R1 Oracle class/seed/budget/prefix mismatch")
    selected=select_oracle_bits(summary,tape_sha=tape_sha,
        n_normal=n_normal,target=target,capacity=capacity)
    return (selected,hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            summary["score_all_probe_a_development"]["oracle_found"],
            float(summary["threshold"]),summary["source_fixed_bank_sha256"])


def check_r0(path, *, args, tape_sha, fixed_sha, theta, metric_static,
             metric_greedy, n_probe):
    ref=json.loads(Path(path).read_text(encoding="utf-8"))
    cfg=ref["config"]
    expected=(args.class_name,args.support_seed,args.query_seed,args.budget)
    actual=(cfg["class_name"],cfg["support_seed"],cfg["query_seed"],cfg["budget"])
    if expected!=actual or args.capacity not in cfg["capacities"]:
        raise FalsificationProtocolError("R0 class/seed/budget/capacity mismatch")
    if ref["tape_sha256"]!=tape_sha or ref["fixed_bank_sha256"]!=fixed_sha:
        raise FalsificationProtocolError("R0 tape/static fingerprint mismatch")
    if abs(ref["theta0"]-theta)>1e-8:
        raise FalsificationProtocolError("R0 frozen theta mismatch")
    if n_probe!=ref["probe_metrics"]["static"]["N"]:
        raise FalsificationProtocolError("R0 reference Probe-A count mismatch")
    for name,observed in (("static",metric_static),
                          (str(args.capacity),metric_greedy)):
        original=ref["probe_metrics"][name]
        for field in ("tp","tn","fp","fn"):
            if observed[field]!=original[field]:
                raise FalsificationProtocolError(
                    f"R0 parity mismatch {name}.{field}: {observed[field]} "
                    f"!= {original[field]}")
    return dict(passed=True,reference_path=str(path),
        reference_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        scope="same Q-A/static/threshold and full Probe-A Static/Greedy confusion")


def build_native_for_states(fixed, states):
    """Install P0/P1 dynamic shadows using a correctly sized native template.

    V22 frozen banks default to K_normal_dyn=1024. P0/P1 snapshots are built
    with the requested capacity (e.g. 256); native_bridge.install_on_clone
    correctly rejects a mismatch. Configure the dynamic K on an isolated
    template, never on the source fixed bank, then use the unchanged bridge.
    """
    if not states:
        raise FalsificationProtocolError("No P0/P1 memory states to score")
    capacities = {tuple(int(x) for x in state.bank.capacities)
                  for state in states.values()}
    if len(capacities) != 1:
        raise FalsificationProtocolError("Mixed capacities across P0/P1 states")
    target_caps = next(iter(capacities))
    template = copy.deepcopy(fixed)
    dynamic = dynamic_codebooks(template)
    if len(dynamic) != len(target_caps):
        raise FalsificationProtocolError("Native/shadow layer count mismatch")
    for cb, cap in zip(dynamic, target_caps):
        if cap < 1 or len(cb.protos) != 0:
            raise FalsificationProtocolError(
                "Expected positive capacity and empty native dynamic template")
        cb.K = cap
    banks = {}
    for name, state in states.items():
        # The bridge validates feature dimensions, K, dtype and device.
        banks[name] = install_on_clone(template, state.bank)
    return banks


@torch.no_grad()
def run(args):
    if args.device=="cuda" and not torch.cuda.is_available():
        raise FalsificationProtocolError("CUDA not available")
    if args.capacity<1 or args.target_admissions<1:
        raise FalsificationProtocolError("Invalid capacity / target")
    if args.max_query_events<0 or args.max_query_events>args.budget:
        raise FalsificationProtocolError("Invalid query prefix")
    if not 1<=args.max_probe_groups_per_type<=24:
        raise FalsificationProtocolError("Invalid Probe-A subset")
    full=(args.max_query_events==0 and args.max_probe_groups_per_type==24)
    if full and (not args.r0_reference_summary or not args.oracle_summary):
        raise FalsificationProtocolError(
            "Full P0/P1 run REQUIRES original R0 and R1 reference JSON")
    if not full and (args.r0_reference_summary or args.oracle_summary):
        raise FalsificationProtocolError("No R0/R1 reference allowed on smoke prefix")
    out=output_dir(args)
    if out.exists():
        raise FileExistsError("Never overwrite prior output: "+str(out))
    if args.math_sdpa and args.device=="cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    impl=p0.require_sources()
    impl.set_seed(0)
    train,test=pf.load_class_records(Path(args.dataset_root),Path(args.json_dir),
                                     args.class_name)
    plan=pf.build_partitions(train,test,pf.SplitConfig())
    support_keys=set(plan.support_samples_by_seed[args.support_seed])
    support_paths=sorted(r.path for r in plan.support if r.sample_key in support_keys)
    if len(support_paths)!=4 or len(support_keys)!=4:
        raise FalsificationProtocolError("Strict 4-shot static reference identity violated")
    tape,n_stream,tape_provenance=frozen_r0_tape(plan,class_name=args.class_name,
        query_seed=args.query_seed,budget=args.budget,
        d1b_manifest_dir=args.d1b_manifest_dir)
    if n_stream!=2625 or len(tape.events)!=args.budget:
        raise FalsificationProtocolError("Unexpected frozen Q-A length")
    cfg=p0.Config(device=args.device,augmentation_seed=0,bank_seed=0,
        fixed_capacity=25000,fixed_per_image=5000,augmentations=4,
        include_identity=False,multiplier=2.5,feature_fuse="concat",
        topk_patches=32,image_aggregation="max",image_topk=64,
        image_quantile=.95,knn=5)
    t0=time.perf_counter()
    backbone=impl.DINOv2MultiLayerBackbone(device=args.device)
    refs=p0.encode_reference_features(impl,backbone,support_paths,cfg)
    fixed=p0.build_vmb(impl,refs,support_paths,cfg)
    fixed_sha=p0.fixed_bank_profile(fixed)["sha256"]
    del refs
    # Frozen static threshold: 64 independent training normal images only.
    history=tuple(plan.history)
    if len(history)!=64 or any(not r.is_good for r in history):
        raise FalsificationProtocolError("History must be exactly 64 training-normal images")
    original_history=[]
    for rec in history:
        img,feat=p0.encode_query(impl,backbone,rec.path,cfg)
        original_history.append(native_score(impl,fixed,img,feat,cfg)[0])
        del img,feat
    frozen_theta=float(fit_fixed_normal_thresholds(original_history)["0.90"]["theta"])
    if abs(normal_q90(original_history)-frozen_theta)>1e-7:
        raise FalsificationProtocolError("Independent q90 implementation mismatch")
    sigma=statistics.stdev(original_history)
    settings=v02.make_memory_settings(fixed)
    empty_arm=memory.MemoryArm("greedy_coreset",copy.deepcopy(fixed),settings,
                               seed=args.support_seed+37)
    for cb in dynamic_codebooks(empty_arm.bank):
        if len(cb.protos):
            raise FalsificationProtocolError("Nonempty initial dynamic memory")
        cb.K=args.capacity
    empty=DecisionState(snapshot(empty_arm.bank))
    # Only Q-A images are encoded for admission. All proposals are preselected
    # against the SAME frozen STATIC scorer (v22 prior version).
    lookup={r.path:r for r in test}
    query_rows=[]
    candidates=[]
    n_use=args.max_query_events or args.budget
    for event in tape.events[:n_use]:
        rec=lookup.get(event.path)
        if rec is None or rec.sample_key!=event.sample_key:
            raise FalsificationProtocolError("Q-A record mismatch")
        img,feat=p0.encode_query(impl,backbone,event.path,cfg)
        score,amap=native_score(impl,fixed,img,feat,cfg)
        row=dict(stream_position=int(event.position),sample_key=rec.sample_key,
                 image_id=rec.rel_path,camera_id=rec.camera_id,
                 label_revealed="normal" if rec.is_good else "defect",
                 static_score=score)
        if rec.is_good:
            allow,keep,reason,_=impl.select_informative_normal_write(
                score=score,theta=frozen_theta,sigmaN=max(sigma,.02),
                pred_defect=bool(score>=frozen_theta),novelty=0.0)
            if not allow or keep not in (8,16,24):
                raise FalsificationProtocolError("Native v22 write plan unexpectedly changed")
            raw,chosen=extract_v22_candidates(impl,amap,feat,keep=keep)
            normalized=normalized_native_candidate_rows(
                impl,empty_arm.bank,feat,chosen,keep=keep)
            novelty=d2.candidate_novelty(raw,empty_arm.bank)
            item=dict(sample_key=rec.sample_key,static_score=score,
                static_patch_novelty=novelty,keep=keep,
                raw=tuple(x.detach().clone() for x in raw),
                normalized=tuple(x.detach().clone() for x in normalized),
                stream_position=int(event.position),reason=reason)
            candidates.append(item)
            row.update(normal_candidate_index=len(candidates)-1,
                keep=keep,static_patch_novelty=novelty,write_reason=reason)
        query_rows.append(row)
        del img,feat
    n_normal=len(candidates)
    metadata=[{k:v for k,v in x.items() if k not in ("raw","normalized")} for x in candidates]
    matched=select_matched(metadata,args.target_admissions,seed=args.hash_seed)
    oracle_sha=None
    oracle_expected=None
    if args.oracle_summary:
        oracle_idxs,oracle_sha,oracle_expected,oracle_theta,oracle_fixed=load_oracle(args.oracle_summary,
            tape_sha=tape_provenance["tape_sha256"],
            n_normal=n_normal,target=args.target_admissions,
            capacity=args.capacity,args=args)
        if abs(oracle_theta-frozen_theta)>1e-8 or oracle_fixed!=fixed_sha:
            raise FalsificationProtocolError("R1 Oracle theta/fixed-bank mismatch")
        matched["r1_hindsight_beam_k"]=oracle_idxs
    # For the matched controls we intentionally choose EXACT K using all
    # already queried-normal event metadata: this is noncausal HINDSIGHT,
    # and it never accesses Probe A/B.
    states={"static":empty}
    control_rows=[]
    for name,selected in matched.items():
        state=empty
        for i,p in enumerate(candidates):
            old=state
            state=transition(state,p["normalized"],i in selected)
            control_rows.append(dict(policy=name,candidate_index=i,
                stream_position=p["stream_position"],sample_key=p["sample_key"],
                selected=i in selected,static_patch_novelty=p["static_patch_novelty"],
                static_score=p["static_score"],keep=p["keep"],
                appended=state.appended-old.appended,
                replaced=state.replaced-old.replaced,
                skipped=state.skipped-old.skipped))
        states[name]=state
    greedy=empty
    for p in candidates:
        greedy=transition(greedy,p["normalized"],True)
    states["greedy_all"]=greedy
    # Reproduce original D2 causal novelty rule with same native source and
    # sequential state-dependent memory; count is NOT forced to target K.
    novelty_state=d2.BaselineState("causal_novelty",seed=args.support_seed+37)
    causal=empty
    for i,p in enumerate(candidates):
        arm=install_on_clone(empty_arm.bank,causal.bank)
        outcome=d2.propose_baseline(novelty_state,causal.bank,p["normalized"],
            raw_candidate_layers=p["raw"],native_bank=arm)
        novelty_state.record(outcome)
        prev=causal
        admitted=int(outcome.decision=="admit")
        causal=DecisionState(outcome.bank,causal.decisions+str(admitted),
            causal.admissions+admitted,causal.effective_updates+admitted,
            causal.appended+outcome.added,causal.replaced+outcome.replaced,
            causal.skipped+outcome.skipped)
        control_rows.append(dict(policy="causal_novelty_median",candidate_index=i,
            stream_position=p["stream_position"],sample_key=p["sample_key"],
            selected=bool(admitted),static_patch_novelty=p["static_patch_novelty"],
            causal_novelty=outcome.novelty,causal_cutoff=outcome.novelty_cutoff,
            static_score=p["static_score"],keep=p["keep"],
            appended=outcome.added,replaced=outcome.replaced,skipped=outcome.skipped))
    states["causal_novelty_median"]=causal
    # Score training NORMAL calibration data and final Probe A with same
    # preselected bank states. All thresholds are calculated before Probe A.
    native_banks=build_native_for_states(fixed,states)
    calibration={name:[] for name in states}
    calibration_rows=[]
    for rec in history:
        img,features=p0.encode_query(impl,backbone,rec.path,cfg)
        row=dict(sample_key=rec.sample_key,image_id=rec.rel_path)
        for name,bank in native_banks.items():
            score=native_score(impl,bank,img,features,cfg)[0]
            calibration[name].append(score)
            row[f"score_{name}"]=score
        calibration_rows.append(row)
        del img,features
    if max(abs(a-b) for a,b in zip(original_history,calibration["static"]))>1e-5:
        raise FalsificationProtocolError("Static calibration changed after replay")
    calibrated_theta={
        name:float(fit_fixed_normal_thresholds(scores)["0.90"]["theta"])
        for name,scores in calibration.items()}
    for name,scores in calibration.items():
        if abs(calibrated_theta[name]-normal_q90(scores))>1e-7:
            raise FalsificationProtocolError("normal-q90 independent parity failed")
    if abs(calibrated_theta["static"]-frozen_theta)>1e-5:
        raise FalsificationProtocolError("Static final calibration unexpectedly changed")
    # Deterministic full Probe-A subset uses archived sample-group order.
    normal_keys=sorted(k for k in plan.probe_a_samples if all(
        x.is_good for x in plan.probe_a if x.sample_key==k))
    defect_keys=sorted(k for k in plan.probe_a_samples if any(
        not x.is_good for x in plan.probe_a if x.sample_key==k))
    if len(normal_keys)!=24 or len(defect_keys)!=24:
        raise FalsificationProtocolError("Probe-A groups not 24 normal +24 defect")
    selected_keys=set(normal_keys[:args.max_probe_groups_per_type] +
                      defect_keys[:args.max_probe_groups_per_type])
    eval_recs=tuple(x for x in plan.probe_a if x.sample_key in selected_keys)
    labels=[]
    probe_rows=[]
    predictions={name:[] for name in states}
    for rec in eval_recs:
        img,features=p0.encode_query(impl,backbone,rec.path,cfg)
        label=int(not rec.is_good)
        labels.append(label)
        row=dict(sample_key=rec.sample_key,image_id=rec.rel_path,
                 camera_id=rec.camera_id,label_offline=label)
        for name,bank in native_banks.items():
            score=native_score(impl,bank,img,features,cfg)[0]
            row[f"score_{name}"]=score
            predictions[name].append(score)
        probe_rows.append(row)
        del img,features
    summaries={}
    for name,scores in predictions.items():
        fixed_metric=metrics(labels,scores,frozen_theta)
        recal=metrics(labels,scores,calibrated_theta[name])
        summaries[name]=dict(
            n_admitted=states[name].admissions,
            n_appended=states[name].appended,n_replaced=states[name].replaced,
            n_skipped=states[name].skipped,
            n_dynamic_prototypes=sum(len(layer) for layer in states[name].bank.layers),
            final_bank_sha256=states[name].bank.sha256(),
            theta_calibrated_q90=calibrated_theta[name],
            fixed_operating_point=fixed_metric,normal_recalibrated_operating_point=recal,
            changes_vs_static_fixed_at_fixed=paired_changes(
                labels,predictions["static"],scores,frozen_theta,frozen_theta),
            changes_vs_static_fixed_at_recalibrated=paired_changes(
                labels,predictions["static"],scores,frozen_theta,calibrated_theta[name]))
    r0_parity=None
    if args.r0_reference_summary and full:
        r0_parity=check_r0(args.r0_reference_summary,args=args,
            tape_sha=tape_provenance["tape_sha256"],fixed_sha=fixed_sha,
            theta=frozen_theta,metric_static=summaries["static"]["fixed_operating_point"],
            metric_greedy=summaries["greedy_all"]["fixed_operating_point"],
            n_probe=len(eval_recs))
    if full:
        # R1 original 240-Probe-A Oracle is a direct reference, not an
        # exploratory success criterion or a target fitted from this run.
        oracle_actual=summaries["r1_hindsight_beam_k"]["fixed_operating_point"]
        for key in ("tp","tn","fp","fn"):
            if oracle_actual[key]!=oracle_expected[key]:
                raise FalsificationProtocolError(
                    f"R1 Beam final confusion parity failure for {key}")
    out.mkdir(parents=True,exist_ok=False)
    summary=dict(schema=SCHEMA,
        status="COMPLETE" if full else "PREFIX_SMOKE_INCOMPLETE",
        research_phase="P0/P1 FALSIFICATION; NOT A SAFETY-GATE SUCCESS CLAIM",
        fixed_source_four_shot=True,extra_training_normal_calibration_images=64,
        no_probe_b_access=True,
        no_probe_a_in_non_oracle_admission_or_threshold_selection=True,
        all_top_k_controls_are_retrospective_not_online=True,
        oracle_control_is_probe_a_search_label_selected=True,
        causal_novelty_count_unmatched=True,
        all_methods_share_same_v22_normal_patches_and_greedy_operator=True,
        not_full_prequential=True,
        configuration=dict(class_name=args.class_name,
            support_seed=args.support_seed,query_seed=args.query_seed,
            budget=args.budget,capacity=args.capacity,target_admissions=args.target_admissions,
            hash_seed=args.hash_seed,max_query_events=args.max_query_events,
            max_probe_groups_per_type=args.max_probe_groups_per_type,
            oracle_summary=args.oracle_summary,
            r0_reference_summary=args.r0_reference_summary),
        frozen_theta_q90=frozen_theta,
        calibration_formula="empirical 90th percentile HIGHER rank=ceil(0.9*(64-1))=57 (zero-based) of 64 TRAIN normals; bank-specific end-of-stream normal-only calibration",
        n_query_events=n_use,n_verified_normal=n_normal,
        n_queried_defect=n_use-n_normal,n_probe_images=len(eval_recs),
        n_probe_normal=sum(y==0 for y in labels),
        n_probe_defect=sum(y==1 for y in labels),
        fixed_bank_sha256=fixed_sha,
        query_tape=tape_provenance,
        r0_parity=r0_parity,r1_oracle_summary_sha256=oracle_sha,
        results=summaries,elapsed_seconds=time.perf_counter()-t0,
        interpretation_limitations=[
            "Greedy all / static and final-state Probe A only, not full-stream prequential.",
            "Calibration normals are a 64-image extra cohort, not strict four-shot-total.",
            "Retrospective top-K normal-event selectors use future queried-normal identities and are NOT deployable online policies.",
            "Original causal_novelty_median gate is online-feasible under frozen Q-A, but has an unmatched admission count.",
            "Matched admission EVENT counts do not imply matched patches or final prototype counts; report both.",
            "R1 hindsight oracle used Probe-A search labels and cannot be a confirmatory comparator.",
            "Probe A has been inspected before; all outcomes are exploratory; Probe B remains untouched.",
            "Recalibration is measured at the final memory state; no causal online threshold-policy claim.",
        ])
    write_json(out/"p01_summary.json",summary)
    write_csv(out/"p01_query_tape.csv",query_rows)
    write_csv(out/"p01_candidate_metadata.csv",metadata)
    write_csv(out/"p01_admission_decisions.csv",control_rows)
    write_csv(out/"p01_calibration_scores.csv",calibration_rows)
    write_csv(out/"p01_probe_a_scores.csv",probe_rows)
    provenance=dict(git=pf.git_state(),sources=pf.source_hashes((
        "experiments/risk_safe/p01_falsification_core.py",
        "experiments/risk_safe/p01_falsification_native.py",
        "experiments/risk_safe/r1_oracle_core.py",
        "experiments/risk_safe/r1_native_oracle.py",
        "experiments/risk_safe/r0_opportunity.py",
        "experiments/risk_safe/native_bridge.py",
        "experiments/risk_safe/d2_baselines.py",
        "experiments/risk_safe/shadow_memory.py",
        "test_phase3_AIF_simplified_v22.py")),
        dataset_json_sha256=pf.sha256_file(
            Path(args.json_dir)/(args.class_name+".json")),
        archived_query_manifest_sha256=pf.sha256_file(
            Path(args.d1b_manifest_dir)/d1b.OUTPUT_QUERIES),
        query_tape=tape_provenance,
        normal_calibration_sample_keys=[rec.sample_key for rec in history],
        probe_a_sample_keys=sorted(selected_keys))
    write_json(out/"p01_provenance.json",provenance)
    checks={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in out.iterdir() if p.is_file()}
    write_json(out/"checksums.json",checks)
    print(json.dumps(dict(status=summary["status"],out_dir=str(out),
        n_queried=n_use,n_normal=n_normal,theta_fixed=frozen_theta,
        r0_parity=r0_parity,results={k:dict(
            admitted=v["n_admitted"],prototypes=v["n_dynamic_prototypes"],
            theta_calibrated=v["theta_calibrated_q90"],
            fixed_fpr=v["fixed_operating_point"]["fpr"],
            fixed_fnr=v["fixed_operating_point"]["fnr"],
            calibrated_fpr=v["normal_recalibrated_operating_point"]["fpr"],
            calibrated_fnr=v["normal_recalibrated_operating_point"]["fnr"],
            auroc=v["fixed_operating_point"]["auroc"]) for k,v in summaries.items()}),
        indent=2))
    return 0


def verify(args):
    out=output_dir(args)
    checks=json.loads((out/"checksums.json").read_text(encoding="utf-8"))
    for name,digest in checks.items():
        if hashlib.sha256((out/name).read_bytes()).hexdigest()!=digest:
            raise FalsificationProtocolError("Result checksum mismatch: "+name)
    summary=json.loads((out/"p01_summary.json").read_text(encoding="utf-8"))
    cfg=summary["configuration"]
    for field,expected in (("budget",args.budget),("capacity",args.capacity),
        ("target_admissions",args.target_admissions),
        ("max_query_events",args.max_query_events),
        ("max_probe_groups_per_type",args.max_probe_groups_per_type)):
        if cfg[field]!=expected:
            raise FalsificationProtocolError(f"Verified configuration mismatch: {field}")
    if summary["schema"]!=SCHEMA:
        raise FalsificationProtocolError("Verified schema mismatch")
    print(json.dumps(dict(verified=True,no_gpu_work=True,
        status=summary["status"],directory=str(out),files=len(checks)),indent=2))
    return 0


def main(argv=None):
    args=parser().parse_args(argv)
    if args.json_dir is None:
        args.json_dir=str(Path(args.dataset_root)/"jsons"/"realiad_jsons")
    return verify(args) if args.verify_only else run(args)


if __name__=="__main__":
    raise SystemExit(main())
