#!/usr/bin/env python3
"""D4-A/B/C: equal native dynamic wins, causal feedback credit, shuffled risk.

Proof-of-mechanism at the POST-B16 checkpoint only (all 16 QUERY labels have
been revealed before Probe A). Not yet an incrementally updated prequential
agent. Reconstructs original unchanged native D3 Greedy 104 / Novelty 48 and
refuses to run if ANY of its 480 archived scores differ.

A: identical per-image K=min(ceil(.02*Npatches), native global winners).
B: rank these candidates by source-only normal leave-one-sample-out benefit
   and source-only defect top32 weak-local influence, versus random/spatial.
C: permute source defect evidence association, maintaining credit marginals.
Three deterministic random/shuffled seeds. Probe labels are evaluation only.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import torch

from . import conditional_retrieval_pilot as d3
from . import d3_controlled_retrieval as d3c
from . import d1d_native_smoke as d1d
from . import d2_native_replay as d2
from . import preflight as pf
from .conditional_retrieval import ScopedRetrieval, coordinates
from .d4_retrieval_rights import (
    METHODS,REPLICATES,RHO,MAX_EXAMPLES_PER_LABEL,TOP_SOURCE_PATCHES,
    Evidence,ExactWinMask,action_priority,make_evidence,
    native_winners,select_indices)
from .d2_metrics import image_metrics
from .native_bridge import NativeParityError, native_score

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/"experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0,str(p))
import p0_protocol_audit as p0

SCHEMA="risk_safe.d4.equal_win_feedback_credit_dev.v1"
BASE="outputs/risk_safe_sprint_20261027"
OUT=BASE+"/d4_audiojack_s0_q37_b16_q90_equalwin_feedback_v1"
RESULT="summary.json"
SCORES="equal_win_probe_scores.csv"
COMPARISONS="equal_win_comparison.csv"
EVIDENCE="feedback_credit_audit.json"
SOURCE="source_manifest.json"
SOURCES=(
    "experiments/risk_safe/d4_equal_win_feedback_pilot.py",
    "experiments/risk_safe/d4_retrieval_rights.py",
    "experiments/risk_safe/d3_controlled_retrieval.py",
    "experiments/risk_safe/retrieval_random_controls.py",
    "experiments/risk_safe/conditional_retrieval.py",
    "experiments/risk_safe/conditional_retrieval_pilot.py",
    "experiments/risk_safe/native_bridge.py",
    "experiments/risk_safe/d2_native_replay.py",
    "experiments/risk_safe/d2_metrics.py",
    "experiments/risk_safe/d2_baselines.py",
    "experiments/risk_safe/d1d_native_smoke.py",
    "experiments/risk_safe/preflight.py",
    "experiments/a0_a1_strict4_matched.py",
    "experiments/p0_protocol_audit.py",
    "experiments/v02_memory.py",
    "experiments/v02_architecture_core.py",
    "test_phase3_AIF_simplified_v22.py",
    "vmb_visionad_new.py",
    "tests/test_risk_safe_d4.py",
    "docs/risk_safe/D4_EQUAL_WIN_FEEDBACK_PROTOCOL.md",
)


def parser():
    p=d3c.parser()
    p.description=__doc__
    p.set_defaults(out_dir=OUT)
    p.add_argument("--win-budget-fraction",type=float,default=RHO)
    return p


def enforce(args):
    d3c.enforce(args)
    if args.win_budget_fraction!=RHO:
        raise NativeParityError("D4 fixed equal-win budget must be .02")


def config(args):
    return {
        "schema":SCHEMA,
        "d3_replay_config":d3.config(args),
        "d3_reference_dir":str(args.reference_d3_dir).replace("\\","/"),
        "source_d2a_summary":str(args.reference_d2a_summary).replace("\\","/"),
        "methods":list(METHODS),"random_shuffled_reps":list(REPLICATES),
        "image_win_fraction":RHO,
        "win_budget":"min(ceil(.02*Npatch),native_original_global_winning_patches)",
        "shared_prototypes":"Greedy 104 and Novelty 48 immutable native append outcomes",
        "native_top5":"allow all stored prototypes on chosen patches, none elsewhere",
        "feedback_time":"all frozen B16 labels revealed before offline Probe A checkpoint",
        "risk_local":"top32 static anomaly patches of DEFECT-labeled image, weak GT-free",
        "normal_LOSO":"remove all memory patches from SAME official sample key",
        "anchors":"B16 revealed single-camera images only; no Probe samples",
        "continuous_source_influence":"positive static-vs-masked native patch map delta",
        "kernel_bandwidth":.1,"spatial_bandwidth":.25,
        "risk_multiplier":1.,
        "negative_control":"shuffle defect feature/position attribution, not labels",
        "policy_tuning_with_probe":False,
        "probe_B_used":False,
        "GT_masks_used":False,
        "causal_scope":"post-B16 final checkpoint; NOT online incremental policy",
        "risk_estimate_calibrated":False,
    }


def numeric_metrics(rows,theta,reference):
    y=[int(z["label_offline_only"]) for z in rows]
    sc=[float(z["score"]) for z in rows]
    m=image_metrics(y,sc,theta)
    normal=[z for z in rows if not int(z["label_offline_only"])]
    defect=[z for z in rows if int(z["label_offline_only"])]
    gain=lambda z:max(0.,float(z["static_score"])-float(z["score"]))
    return {**m,
       "normal_mean_score_gain":sum(map(gain,normal))/len(normal),
       "defect_mean_score_erosion":sum(map(gain,defect))/len(defect),
       "total_selected_win_patches":sum(int(z["selected_win_count"]) for z in rows),
       "total_available_global_winners":sum(int(z["global_win_candidates"]) for z in rows),
       "mean_selected_win_fraction":sum(float(z["selected_win_count"])/float(z["n_patches"])
                                        for z in rows)/len(rows),
       "fpr_gain_vs_static":reference["static_metrics"]["fpr"]-m["fpr"]}


@torch.no_grad()
def build_evidence(*,args,impl,backbone,cfg,plan,bank,provenance,
                   calibrated,method):
    """Only confirmed B16 labels. No Probe anywhere in this function."""
    tape,n_stream=d1d.frozen_tape(plan,args)
    if len(tape.events)!=16 or n_stream!=2625:
        raise NativeParityError("D4 causal query tape differs from D1-B")
    meta={r.path:r for phase in plan.phases for r in phase}
    candidates=[]
    final_t=max(x.position for x in tape.events)
    for event in tape.events:
        if event.position>final_t or event.path not in meta:
            raise NativeParityError("Noncausal or missing feedback event")
        rec=meta[event.path]
        if event.sample_key!=rec.sample_key or event.rel_path!=rec.rel_path:
            raise NativeParityError("Frozen Q-A source identity changed")
        # Only HERE, after the entire B16 history has been replayed, do
        # labels become available as historical feedback.
        image,features=p0.encode_query(impl,backbone,event.path,cfg)
        candidates.append((event.position,rec,image,features))
    if len({x[1].sample_key for x in candidates})!=16:
        raise NativeParityError("Historical Q-A source labels not sample-disjoint")
    source_keys={x[1].sample_key for x in candidates}
    if source_keys & (set(plan.probe_a_samples)|set(plan.probe_b_samples)):
        raise NativeParityError("Feedback anchors contaminate held-out Probe")
    static=copy.deepcopy(bank)
    static.normal_dyn["p0"][0]["id"][0].clear()
    static_sc=ScopedRetrieval(impl,static,(),"global")
    global_sc=ScopedRetrieval(impl,copy.deepcopy(bank),provenance,"global")

    class ExcludeOrigin:
        def __init__(self):self.origin=None
        def __call__(self,provs,policy,*,query_camera,query_coords,image_id):
            if image_id!=self.origin:
                raise NativeParityError("LOSO feedback origin identity mismatch")
            eligible=torch.tensor([z["sample_key"]!=self.origin_sample_key
                                   for z in provs],dtype=torch.bool,
                                  device=query_coords.device)
            return eligible[None,:].expand(len(query_coords),len(provs))
    excl=ExcludeOrigin()
    loso=ScopedRetrieval(impl,copy.deepcopy(bank),provenance,
                         "spatial_0125",mask_provider=excl)
    normals=[]
    defects=[]
    notes=[]
    for step,rec,image,features in candidates:
        f=features[0]
        uv=coordinates(impl,len(f),(image.height,image.width))
        emb=impl.l2_normalize(f.to(bank.device)).float()
        static_map=static_sc.patch_map(
            image,features,cfg,camera_id=int(rec.camera_id))
        static_score,_=static_sc.score(
            image,features,cfg,camera_id=int(rec.camera_id))
        if rec.is_good:
            excl.origin=rec.rel_path
            excl.origin_sample_key=rec.sample_key
            dyn_map=loso.patch_map(
                image,features,cfg,camera_id=int(rec.camera_id),
                image_id=rec.rel_path)
            filtered=bool((static_map-dyn_map).max().item()>1e-7)
            delta=(static_map-dyn_map).clamp_min(0)
            # Only cross-sample corrected evidence; reward FP correction.
            factor=1.+2.*float(static_score>=calibrated["theta"])
            weights=delta*factor
            keep=min(MAX_EXAMPLES_PER_LABEL,int((weights>1e-7).sum().item()))
            # No self-fit benefit is counted when no other memory helps.
            kind="normal"
        else:
            dyn_map=global_sc.patch_map(
                image,features,cfg,camera_id=int(rec.camera_id))
            delta=(static_map-dyn_map).clamp_min(0)
            top=min(TOP_SOURCE_PATCHES,len(static_map))
            suspected=torch.zeros(len(static_map),device=static_map.device,
                                  dtype=torch.bool)
            suspected[static_map.topk(top).indices]=True
            weights=delta*suspected.float()
            keep=min(MAX_EXAMPLES_PER_LABEL,int((weights>1e-7).sum().item()))
            filtered=bool(keep)
            kind="defect"
        if keep:
            ix=weights.topk(keep).indices.to(emb.device)
            rows=(emb[ix].detach().cpu(),
                  uv[ix.detach().cpu()].clone(),
                  rec.camera_id,
                  weights[ix.to(weights.device)].detach().cpu(),
                  rec.rel_path)
            (normals if rec.is_good else defects).append(rows)
        notes.append({"t":step,"sample_key":rec.sample_key,
                      "image":rec.rel_path,"label":kind,
                      "evidence_patches":keep,
                      "has_positive_influence":filtered,
                      "static_score":static_score,
                      "self_origin_excluded":bool(rec.is_good)})
    dim=bank.normal_dyn["p0"][0]["id"][0].protos.shape[1]
    norm=make_evidence(normals,dim)
    risk=make_evidence(defects,dim)
    summary={"method":method,
             "n_historical_labelled_images":len(candidates),
             "n_normal_images":sum(z["label"]=="normal" for z in notes),
             "n_defect_images":sum(z["label"]=="defect" for z in notes),
             "normal_nonself_witness_count":len(norm.embedding),
             "defect_weak_local_witness_count":len(risk.embedding),
             "normal_credit_sum":float(norm.weight.sum()),
             "defect_risk_proxy_sum":float(risk.weight.sum()),
             "normal_source_images":sorted(set(norm.source)),
             "defect_source_images":sorted(set(risk.source)),
             "risk_identifiable":len(risk.embedding)>1 and float(risk.weight.sum())>0,
             "per_queried_event":notes}
    return norm,risk,summary


@torch.no_grad()
def controls_callback(*,args,ref_summary,ref_rows,impl,backbone,cfg,plan,
                      arms,provenance,probes,calibrated,d3_summary,d3_csvs):
    parity=d3c.verify_replayed_d3(d3_summary,d3_csvs,ref_summary,ref_rows)
    frozen={(r["method"],r["policy"],r["image"]):r for r in ref_rows}
    all_scores=[]
    comparisons=[]
    audits=[]
    # Final checkpoint occurs AFTER B16 feedback, before any heldout Probe.
    for method in ("greedy","causal_novelty"):
        bank=arms[method].bank
        proto=provenance[method]
        norm,risk,audit=build_evidence(
            args=args,impl=impl,backbone=backbone,cfg=cfg,plan=plan,
            bank=bank,provenance=proto,calibrated=calibrated,method=method)
        audits.append(audit)
        prototype_uv=torch.tensor([[x["u"],x["v"]] for x in proto],
                                   dtype=torch.float32,device=bank.device)
        mask=ExactWinMask()
        scope=ScopedRetrieval(impl,copy.deepcopy(bank),proto,
                              "spatial_0125",mask_provider=mask)
        for rec,image,features in probes:
            q,win,top5=native_winners(impl,bank,features)
            uv=coordinates(impl,len(q),(image.height,image.width)).to(q.device)
            real=frozen[(method,"global",rec.rel_path)]
            reference_global=float(real["score"])
            reference_static=float(real["static_score"])
            k=min(math.ceil(RHO*len(q)),int(win.sum().item()))
            # Distinct random seeds for random/shuffled. All other policies
            # are deterministic and evaluated exactly once per query.
            for policy in METHODS:
                repeat_ids=REPLICATES if policy in (
                    "random","shuffled_defect_feedback") else (0,)
                for replicate in repeat_ids:
                    rank=action_priority(
                        policy,q,uv,int(rec.camera_id),norm,risk,
                        top5=top5,proto_uv=prototype_uv,replicate=replicate)
                    chosen,count=select_indices(
                        win,rank,method=policy,
                        image_id=rec.rel_path,replicate=replicate)
                    if count!=k:
                        raise NativeParityError("D4 exact win budget mismatch")
                    mask.set(rec.rel_path,chosen)
                    score,_=scope.score(
                        image,features,cfg,camera_id=int(rec.camera_id),
                        image_id=rec.rel_path)
                    if not reference_global-1e-6<=score<=reference_static+1e-6:
                        raise NativeParityError(
                            f"Native D4 equality/monotonicity failed: "
                            f"{method}/{policy}/{rec.rel_path}")
                    if scope.last_query_stats["n_dynamic_winner_patches"]!=count:
                        raise NativeParityError(
                            "Exact winner count changed during native scoring")
                    all_scores.append({
                        "method":method,"policy":policy,"repeat":replicate,
                        "image":rec.rel_path,"sample_key":rec.sample_key,
                        "camera_id":rec.camera_id,
                        "label_offline_only":int(not rec.is_good),
                        "score":score,"static_score":reference_static,
                        "global_score":reference_global,
                        "selected_win_count":count,
                        "global_win_candidates":int(win.sum().item()),
                        "n_patches":len(q),
                        "budget_rho":RHO,
                        "normal_witnesses":len(norm.embedding),
                        "defect_witnesses":len(risk.embedding),
                    })
        for policy in METHODS:
            repeat_ids=REPLICATES if policy in (
                "random","shuffled_defect_feedback") else (0,)
            for rep in repeat_ids:
                rows=[z for z in all_scores if z["method"]==method
                      and z["policy"]==policy and z["repeat"]==rep]
                if len(rows)!=40:
                    raise NativeParityError("D4 incomplete Probe A pair")
                m=numeric_metrics(rows,calibrated["theta"],d3_summary)
                comparisons.append({
                    "method":method,"policy":policy,"repeat":rep,
                    **m,"memory_prototypes":len(proto),
                    "normal_feedback_witnesses":len(norm.embedding),
                    "defect_feedback_witnesses":len(risk.embedding),
                    "valid_defect_feedback_ranking":audit["risk_identifiable"],
                })
        # All source use is post-label at final checkpoint; feedback rights
        # themselves NEVER consult these Probe labels for selection.
    if len(all_scores)!=2*7*40 or len(comparisons)!=14:
        raise NativeParityError("D4 ablation matrix incomplete")
    # Same image, same native potential global winners, same exact selected
    # win count for every D4 policy / random repeat.
    groups={}
    for z in all_scores:
        groups.setdefault((z["method"],z["image"]),set()).add(
            (z["selected_win_count"],z["global_win_candidates"]))
    if len(groups)!=80 or any(len(v)!=1 for v in groups.values()):
        raise NativeParityError("D4 equal-win pairing violated")
    summary={
        "schema":SCHEMA,"completed":True,"config":config(args),
        "original_d3_replay_parity":parity,
        "q_a_labels_prior_to_probe":16,"probe_a_images":40,
        "probe_b_used":False,"probe_labels_used_in_rights":False,
        "same_per_image_native_win_budget_all_policies":True,
        "original_memory_changed":False,"N_scores":len(all_scores),
        "N_comparisons":len(comparisons),
        "methods":list(METHODS),"N_random_seeds":len(REPLICATES),
        "comparisons":comparisons,
        "limitations":[
            "post-B16 frozen-feedback rights, not event-by-event online update",
            "image labels only; defect top32 pixels are weak model hypotheses",
            "zero defect risk witness means no identifiable defect supervision",
            "random seeds share 8 official sample groups (40 camera views)",
            "Probe A is development data; radius/approaches have been inspected",
            "fixed rho=.02 is a preregistered diagnostic budget, not optimized",
            "equal native winner count does not equal image score impact",
        ],
    }
    csvs={
        SCORES:d2.csv_content((
            "method","policy","repeat","image","sample_key","camera_id",
            "label_offline_only","score","static_score","global_score",
            "selected_win_count","global_win_candidates","n_patches",
            "budget_rho","normal_witnesses","defect_witnesses"),all_scores),
        COMPARISONS:d2.csv_content((
            "method","policy","repeat","N","N_normal","N_defect",
            "tp","tn","fp","fn","fpr","fnr","auroc","auprc","theta0",
            "normal_mean_score_gain","defect_mean_score_erosion",
            "total_selected_win_patches","total_available_global_winners",
            "mean_selected_win_fraction","fpr_gain_vs_static",
            "memory_prototypes","normal_feedback_witnesses",
            "defect_feedback_witnesses",
            "valid_defect_feedback_ranking"),comparisons),
    }
    evidence={"schema":"risk_safe.d4.feedback_evidence_audit.v1",
              "q_a_revealed_before_rights":True,
              "mask_ground_truth_used":False,
              "normal_LOSO_used":True,"risk_weak_anomaly_topk":32,
              "source":audits}
    return summary,evidence,csvs


def manifest(args,result,evidence,csvs):
    return {
        "schema":"risk_safe.d4.source_manifest.v1",
        "git":pf.git_state(),
        "source_sha256":pf.source_hashes(SOURCES),
        "input_sha256":{
            "d2a_summary":pf.sha256_file(Path(args.reference_d2a_summary)),
            "d3_summary":pf.sha256_file(Path(args.reference_d3_dir)/d3.RESULT),
            "d3_scoring_csv":pf.sha256_file(Path(args.reference_d3_dir)/d3.PROBES),
            "d1b_queries":pf.sha256_file(Path(args.d1b_manifest_dir)/d2.d1b.OUTPUT_QUERIES),
            "threshold":pf.sha256_file(Path(args.threshold_registry)),
            "dataset_json":pf.sha256_file(Path(args.json_dir)/f"{args.class_name}.json"),
        },
        "config":config(args),
        "output_sha256":{
            RESULT:pf.sha256_text(pf.json_text(result)),
            EVIDENCE:pf.sha256_text(pf.json_text(evidence)),
            **{k:pf.sha256_text(v) for k,v in csvs.items()}},
        "environment":{**pf.environment_block(),
                       "device":args.device,
                       "features_extracted":True,
                       "gpu_encoder_used":args.device=="cuda"},
    }


def verify(args):
    root=Path(args.out_dir)
    result=json.loads((root/RESULT).read_text(encoding="utf-8"))
    evidence=json.loads((root/EVIDENCE).read_text(encoding="utf-8"))
    expected=json.loads((root/SOURCE).read_text(encoding="utf-8"))
    if result.get("schema")!=SCHEMA or result.get("config")!=config(args) or (
            not result.get("completed")):
        raise NativeParityError("D4 output schema/config/completion mismatch")
    if (result.get("N_scores")!=560 or result.get("N_comparisons")!=14 or
            not result.get("same_per_image_native_win_budget_all_policies") or
            not result.get("original_d3_replay_parity",{}).get("passed")):
        raise NativeParityError("D4 incomplete evidence or failed native parity")
    for file,digest in expected["output_sha256"].items():
        if pf.sha256_file(root/file)!=digest:
            raise NativeParityError("D4 immutable output tampered: "+file)
    d3c.validate_reference_d3(Path(args.reference_d3_dir))
    csvs={k:(root/k).read_text(encoding="utf-8")
          for k in (SCORES,COMPARISONS)}
    new=d3c.verify_recorded_run_manifest(
        expected,manifest(args,result,evidence,csvs),source_files=SOURCES)
    pf.verify_json_output(root/SOURCE,new)
    print(json.dumps({"D4":"verified_read_only",
                      "source_D3_native_scores":480,
                      "D4_scored_image_rows":560,
                      "same_native_wins":True,
                      "gpu_rerun":False},indent=2))
    return 0


def main(argv=None):
    args=parser().parse_args(argv)
    if args.json_dir is None:
        args.json_dir=str(Path(args.dataset_root)/pf.DEFAULT_JSON_SUBDIR)
    enforce(args)
    frozen,rows=d3c.validate_reference_d3(Path(args.reference_d3_dir))
    if args.preflight_only:
        pre=d3.preflight(args)
        print(json.dumps({"D4":"preflight_no_gpu","D3_native_reference":480,
                          "same_memory":True,"policies":list(METHODS),
                          "random_repeats":len(REPLICATES),
                          "budget_rho":RHO,"d3_metadata":pre},indent=2))
        return 0
    if args.verify_only:
        return verify(args)
    if Path(args.out_dir).exists():
        raise FileExistsError("D4 immutable outputs exist; use --verify-only")
    start=time.perf_counter()
    d3_result,d3_csvs,(result,evidence,csvs)=d3.run(
        args,controls_callback=lambda **kw:controls_callback(
            args=args,ref_summary=frozen,ref_rows=rows,**kw))
    result["elapsed_seconds"]=time.perf_counter()-start
    dest=Path(args.out_dir)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent,prefix=".d4_build_") as tmp:
        stage=Path(tmp)/dest.name
        stage.mkdir()
        pf.write_json_idempotent(stage/RESULT,result)
        pf.write_json_idempotent(stage/EVIDENCE,evidence)
        for name,content in csvs.items():
            (stage/name).write_text(content,encoding="utf-8",newline="")
        doc=manifest(args,result,evidence,csvs)
        pf.write_json_idempotent(stage/SOURCE,doc)
        for f,expected in doc["output_sha256"].items():
            if pf.sha256_file(stage/f)!=expected:
                raise NativeParityError("D4 staged output SHA mismatch: "+f)
        if dest.exists():
            raise FileExistsError("D4 output appeared before atomic publish")
        stage.rename(dest)
    print(json.dumps({"D4":"completed_development",
                      "scores":560,"comparisons":14,
                      "output":str(dest),"confirmation":False},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
