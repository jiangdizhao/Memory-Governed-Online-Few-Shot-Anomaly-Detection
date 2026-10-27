#!/usr/bin/env python3
"""D3-C: matched-random and shuffled-spatial retrieval attribution.

No new admission method is trained: replays the exact native q90 D3 Greedy and
Novelty memory, validates old 480-row D3 experiment, then evaluates RANDOMIZED
controls on the SAME 40 heldout Probe A images.

- shuffled_spatial: swap normal prototype spatial provenance (not embeddings).
- matched_count_random: EXACT native top-5 eligible-prototype count per patch.
- source_calibrated_hit_random: Bernoulli p fit using ONLY eight mature-history
  TRAIN normal views; freeze before any Probe A control scores. Probe hit rate
  need not match, and mismatch is measured explicitly.

Three independent deterministic seeds for each control and baseline. The
actual 40 Probe labels enter only the offline scorer/metric summarizer, never
the eligibility masks or training-normal hit-probability selection.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import conditional_retrieval_pilot as native
from . import d2_native_replay as d2
from . import preflight as pf
from .conditional_retrieval import ScopedRetrieval, assert_provenance
from .retrieval_random_controls import (
    CONTROL_TYPES, REPEATS, SOURCE_CALIB_N, SOURCE_CALIB_P_GRID,
    RandomEligibility, choose_probability, shuffled_spatial_provenance,
)
from .native_bridge import NativeParityError
from .d2_metrics import image_metrics

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import p0_protocol_audit as p0

SCHEMA = "risk_safe.d3c.context_randomized_attribution.v1"
SUMMARY = "summary.json"
SCORES = "random_control_probe_scores.csv"
PAIRED = "random_control_comparison.csv"
SOURCE_CALIB = "source_only_hit_calibration.json"
MANIFEST = "source_manifest.json"
FILES = (SUMMARY, SCORES, PAIRED, SOURCE_CALIB, MANIFEST)
DEFAULT_D3 = "outputs/risk_safe_sprint_20261027/d3_audiojack_s0_q37_b16_q90_probe4_fixedk"
DEFAULT_OUT = "outputs/risk_safe_sprint_20261027/d3c_audiojack_s0_q37_b16_q90_probe4_randomcontrols_v1"
SOURCES = (
    "experiments/risk_safe/d3_controlled_retrieval.py",
    "experiments/risk_safe/retrieval_random_controls.py",
    "experiments/risk_safe/conditional_retrieval.py",
    "experiments/risk_safe/conditional_retrieval_pilot.py",
    "experiments/risk_safe/d2_native_replay.py",
    "experiments/risk_safe/d2_baselines.py",
    "experiments/risk_safe/d2_metrics.py",
    "experiments/risk_safe/d2_fixed_threshold.py",
    "experiments/risk_safe/native_bridge.py",
    "experiments/risk_safe/preflight.py",
    "experiments/risk_safe/d1d_native_smoke.py",
    "experiments/a0_a1_strict4_matched.py",
    "experiments/p0_protocol_audit.py",
    "experiments/v02_memory.py",
    "experiments/v02_architecture_core.py",
    "test_phase3_AIF_simplified_v22.py",
    "vmb_visionad_new.py",
    "tests/test_risk_safe_d3.py",
    "tests/test_risk_safe_d3c.py",
    "docs/risk_safe/D3C_MATCHED_RANDOM_CONTROLS.md",
)


def parser():
    p = native.parser()
    p.set_defaults(out_dir=DEFAULT_OUT)
    p.description = __doc__
    p.add_argument("--reference-d3-dir", default=DEFAULT_D3)
    p.add_argument("--control-repeats", type=int, default=3)
    p.add_argument("--source-calibration-normal-views", type=int, default=8)
    return p


def enforce(args):
    native.enforce_protocol(args)
    if args.fixed_quantile != .90:
        raise NativeParityError("D3-C preregistration fixes q90; never select by Probe")
    if args.control_repeats != len(REPEATS) or args.source_calibration_normal_views != SOURCE_CALIB_N:
        raise NativeParityError("D3-C repeats/calibration N are frozen")
    if Path(args.reference_d3_dir).resolve() == Path(args.out_dir).resolve():
        raise NativeParityError("Reference and new output directories cannot overlap")


def config(args):
    return {
        "schema": SCHEMA,
        "parent_d3": native.config(args),
        "reference_d3_dir": str(args.reference_d3_dir).replace("\\", "/"),
        "control_seeds": list(REPEATS),
        "control_types": list(CONTROL_TYPES),
        "reference_context_policy": "spatial_0125",
        "source_hit_calibration_p_grid": list(SOURCE_CALIB_P_GRID),
        "source_normal_views": SOURCE_CALIB_N,
        "source_calibration_cohort": "first 8 sorted reserved D1-A TRAIN mature-history NORMAL images",
        "matched_count": "exact eligible PROTOTYPE count for EACH patch (not hit match)",
        "matched_hit": "probability selected on source-normal dynamic winners only; never Probe",
        "shuffled": "shuffle original per-prototype (u,v) with static embeddings unchanged",
        "native_topk": "fixed min(5,M), Static fallback if insufficient eligible",
        "mask_rng": "deterministic SHA256(arm/control/repeat/image), no labels/features",
        "probe_A_labels_used_only_in_posthoc_metrics": True,
        "probe_B_used": False,
        "sidecar": "off",
        "observational_hit_matching_not_guaranteed_on_probe": True,
        "no_algorithm_selected_after_probe_inspection": True,
    }


def parse_csv(text: str):
    return list(csv.DictReader(io.StringIO(text)))


def validate_reference_d3(folder: Path):
    summ = folder / "summary.json"
    csv_path = folder / native.PROBES
    source = folder / native.MANIFEST
    for f in (summ, csv_path, source):
        if not f.is_file():
            raise NativeParityError(f"Required frozen D3 source absent: {f}")
    doc = json.loads(summ.read_text(encoding="utf-8"))
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if doc.get("schema") != native.SCHEMA or (
            not doc.get("completed")) or (
            not doc.get("global_native_parity_pass")) or (
            not doc.get("reference_d2a_replay_parity", {}).get("passed")):
        raise NativeParityError("D3 frozen reference was not native-verified")
    for f in (summ, csv_path):
        if manifest.get("output_sha256",{}).get(f.name) != pf.sha256_file(f):
            raise NativeParityError("D3 frozen reference tampered")
    original = parse_csv(csv_path.read_text(encoding="utf-8"))
    if len(original)!=480 or len({(r["method"],r["policy"],r["image"])
                                   for r in original})!=480:
        raise NativeParityError("Frozen D3 must contain exactly 480 unique scores")
    return doc, original


def verify_replayed_d3(current: dict, current_csvs: dict, reference: dict,
                       reference_rows: list[Mapping]) -> dict:
    if current.get("schema")!=reference.get("schema") or (
            current["source_d1b_tape_sha256"]!=
            reference["source_d1b_tape_sha256"]) or (
            current["threshold_calibration_sha256"]!=
            reference["threshold_calibration_sha256"]) or (
            current["memory_prototypes"]!=reference["memory_prototypes"]) or (
            current["probe_groups"]!=reference["probe_groups"]):
        raise NativeParityError("D3-C replay does not reproduce frozen D3 conditions")
    current_rows=parse_csv(current_csvs[native.PROBES])
    indexed={(r["method"],r["policy"],r["image"]):r
             for r in reference_rows}
    if len(current_rows)!=480 or len(indexed)!=480:
        raise NativeParityError("D3-C missing native reference scores")
    max_abs=0.
    for r in current_rows:
        old=indexed.get((r["method"],r["policy"],r["image"]))
        if old is None or r["label_offline_only"]!=old["label_offline_only"]:
            raise NativeParityError("D3-C Probe/camera identity or label changed")
        diff=abs(float(r["score"])-float(old["score"]))
        if not math.isfinite(diff) or diff>1e-6:
            raise NativeParityError("D3-C v22 native reference mismatch")
        max_abs=max(max_abs,diff)
    return {"passed":True, "matched_native_scores":480,
            "max_abs_score_difference":max_abs,
            "source_results_touched":False}


def summary_metrics(rows: list[dict], *, theta: float, global_metrics: dict):
    labels=[int(r["label_offline_only"]) for r in rows]
    scores=[float(r["score"]) for r in rows]
    metrics=image_metrics(labels,scores,theta)
    norm=[r for r in rows if int(r["label_offline_only"])==0]
    defect=[r for r in rows if int(r["label_offline_only"])==1]
    def score_gain(z):return max(0.,float(z["static_score"])-float(z["score"]))
    elig=[float(r["eligible_top5_fraction"]) for r in rows]
    hits=[float(r["dynamic_win_fraction"]) for r in rows]
    return {
        **metrics,
        "normal_mean_score_gain":sum(map(score_gain,norm))/len(norm),
        "defect_mean_score_erosion":sum(map(score_gain,defect))/len(defect),
        "mean_top5_eligible_fraction":sum(elig)/len(elig),
        "mean_dynamic_winner_fraction":sum(hits)/len(hits),
        "fpr_gain_vs_static":global_metrics["static_fpr"]-metrics["fpr"],
        "fnr_delta_vs_global":metrics["fnr"]-global_metrics["global_fnr"],
    }


def manifest(args, result, calib, outputs):
    return {
        "schema": "risk_safe.d3c.source_manifest.v1",
        "git": pf.git_state(),
        "source_sha256": pf.source_hashes(SOURCES),
        "input_sha256": {
            "real_iad_json": pf.sha256_file(Path(args.json_dir) / f"{args.class_name}.json"),
            "threshold_registry": pf.sha256_file(Path(args.threshold_registry)),
            "d1b_manifest": pf.sha256_file(
                Path(args.d1b_manifest_dir) / d2.d1b.OUTPUT_QUERIES),
            "d2a_summary": pf.sha256_file(Path(args.reference_d2a_summary)),
            "d3_summary": pf.sha256_file(Path(args.reference_d3_dir) / native.RESULT),
            "d3_probe_csv": pf.sha256_file(Path(args.reference_d3_dir) / native.PROBES),
        },
        "config": config(args),
        "output_sha256": {
            SUMMARY: pf.sha256_text(pf.json_text(result)),
            SOURCE_CALIB: pf.sha256_text(pf.json_text(calib)),
            **{name:pf.sha256_text(txt) for name,txt in outputs.items()}},
        "environment": {**pf.environment_block(),"device":args.device,
                        "gpu_encoder_used":args.device=="cuda",
                        "features_extracted":True},
    }


def recorded_source_matches_git_blob(blob: bytes, recorded_sha256: str) -> tuple[bool,str]:
    """Match EXACT historical source bytes, allowing ONLY Git LF/Windows CRLF.

    pf.source_hashes() recorded SHA-256 over read_bytes() of the original
    Windows worktree. Git stores line-ending-normalized blobs when autocrlf or
    text attributes apply. A clean checkout can therefore have a *different*
    SHA-256 from 'git show COMMIT:path' while containing identical source text.

    Do NOT normalize Python tokens, whitespace, comments, trailing blank
    lines, Unicode, or any byte besides Git's LF -> CRLF checkout transform.
    Manifest digests remain unchanged. Binary / mixed-newline source cannot
    exploit normalization because we require a text blob with pure LF.
    """
    if not isinstance(recorded_sha256, str) or len(recorded_sha256) != 64:
        return False, "invalid_digest"
    try:
        int(recorded_sha256, 16)
    except ValueError:
        return False, "invalid_digest"
    if hashlib.sha256(blob).hexdigest() == recorded_sha256:
        return True, "git_blob_exact"
    if b"\x00" in blob or b"\r" in blob or b"\n" not in blob:
        return False, "not_clean_lf_text"
    windows_checkout = blob.replace(b"\n", b"\r\n")
    if hashlib.sha256(windows_checkout).hexdigest() == recorded_sha256:
        return True, "git_lf_to_windows_crlf"
    return False, "source_differs"


def verify_recorded_run_manifest(expected: dict, current: dict, *,
                                 source_files=None) -> dict:
    """Verify evidence against its ACTUAL generating git revision.

    A later verifier-only source fix changes the current source hashes.
    Instead of replacing/forging the immutable manifest, independently read
    every recorded source file from the ORIGINAL git commit, verify its
    SHA-256, require clean recorded tracked state, and then compare ALL other
    manifest fields (input/output hashes, configuration, environment).
    The generating revision must be an ancestor of the current checkout.
    """
    recorded_git=expected.get("git")
    if not isinstance(recorded_git,dict) or not recorded_git.get("git_available"):
        raise NativeParityError("Generating Git revision unavailable")
    old_head=recorded_git.get("head")
    if not isinstance(old_head,str) or len(old_head)!=40 or any(
            ch not in "0123456789abcdef" for ch in old_head.lower()):
        raise NativeParityError("Invalid recorded full commit SHA")
    if recorded_git.get("tracked_modifications") != []:
        raise NativeParityError("Recorded run had modified tracked source")
    if not pf._is_ancestor_commit(old_head,current.get("git",{}).get("head","")):
        raise NativeParityError("Generating commit is not an ancestor of HEAD")
    hashes=expected.get("source_sha256")
    expected_sources=SOURCES if source_files is None else tuple(source_files)
    if not isinstance(hashes,dict) or set(hashes)!=set(expected_sources):
        raise NativeParityError("Incomplete frozen experiment source provenance")
    for path,digest in hashes.items():
        if not isinstance(digest,str) or len(digest)!=64:
            raise NativeParityError("Malformed source SHA-256")
        fetched=subprocess.run(
            ["git","show",f"{old_head}:{path}"],cwd=str(ROOT),
            capture_output=True,check=False)
        if fetched.returncode:
            raise NativeParityError(
                f"Historical Git source could not be read: {path}")
        ok, source_type=recorded_source_matches_git_blob(fetched.stdout,digest)
        if not ok:
            raise NativeParityError(
                f"Historical source does not match generating git commit "
                f"(checked exact blob AND Windows CRLF checkout): {path}; "
                f"reason={source_type}")
    adjusted=dict(current)
    adjusted["git"]=recorded_git
    adjusted["source_sha256"]=hashes
    if adjusted!=expected:
        raise NativeParityError(
            "D3-C frozen manifest differs in inputs, outputs, config or environment")
    return adjusted


def verify(args):
    root=Path(args.out_dir)
    result=json.loads((root/SUMMARY).read_text(encoding="utf-8"))
    calib=json.loads((root/SOURCE_CALIB).read_text(encoding="utf-8"))
    if result.get("schema")!=SCHEMA or result.get("config")!=config(args) or not result.get("completed"):
        raise NativeParityError("D3-C schema/config/completion mismatch")
    expected=json.loads((root/MANIFEST).read_text(encoding="utf-8"))
    for fname,digest in expected["output_sha256"].items():
        if pf.sha256_file(root/fname)!=digest:
            raise NativeParityError(f"D3-C corrupted output: {fname}")
    baseline,_=validate_reference_d3(Path(args.reference_d3_dir))
    reference=result.get("source_reference")
    if (not isinstance(reference,dict)
            or reference.get("passed") is not True
            or reference.get("matched_native_scores")!=480
            or float(reference.get("max_abs_score_difference",float("inf")))>1e-6):
        raise NativeParityError("D3-C original 480-score native replay was not verified")
    if (result.get("N_control_scores")!=800
            or len(result.get("comparison",[]))!=20
            or result.get("no_eval_probe_fit") is not True
            or result.get("N_holdout_probe_images")!=40):
        raise NativeParityError("D3-C control experimental evidence incomplete")
    csvs={name:(root/name).read_text(encoding="utf-8")
          for name in (SCORES,PAIRED)}
    new=verify_recorded_run_manifest(expected,manifest(args,result,calib,csvs))
    pf.verify_json_output(root/MANIFEST,new)
    print(json.dumps({"D3_C":"read_only_sha_verified",
                      "native_source_replay":True,
                      "controls":list(CONTROL_TYPES),
                      "gpu_recomputed":False},indent=2))
    return 0


@torch.no_grad()
def controls_callback(*, args, ref_summary, ref_rows, impl, backbone, cfg,
                      plan, arms, provenance, probes, calibrated,
                      d3_summary, d3_csvs):
    parity=verify_replayed_d3(d3_summary,d3_csvs,ref_summary,ref_rows)
    source_records=sorted(plan.history,key=lambda r:(r.sample_key,r.camera_id,r.path))
    if len(source_records)!=64 or any(not r.is_good for r in source_records):
        raise NativeParityError("D3-C source calibration must have 64 train normals")
    # train-history views, neither heldout Probe A/B nor online Q-A.
    source_records=source_records[:SOURCE_CALIB_N]
    forbidden=set()
    for name in ("support","stream","probe_A","probe_B"):
        forbidden.update(plan.sample_sets()[name])
    if any(r.sample_key in forbidden for r in source_records):
        raise NativeParityError(
            "Random hit calibration overlaps support/stream/Probe A/B")
    all_probe_ids={r.sample_key for r,_,_ in probes}
    if any(r.sample_key in all_probe_ids for r in source_records):
        raise NativeParityError("Source-control calibration overlaps Probe")
    source_encoded=[
        (r,*p0.encode_query(impl,backbone,r.path,cfg))
        for r in source_records
    ]
    source_calib={
        "schema":"risk_safe.d3c.source_only_hit_calibration.v1",
        "source_partition":"D1-A_reserved_mature_history_train_normals",
        "train_normal_views": [r.rel_path for r in source_records],
        "no_probe_scores_or_labels_for_p_fit":True,
        "grid":list(SOURCE_CALIB_P_GRID),
        "results":{},
    }
    result_rows=[]
    comparison=[]
    frozen_image_scores = {
        (z["method"],z["policy"],z["image"]):z for z in ref_rows}
    for method in ("greedy","causal_novelty"):
        mem=arms[method].bank
        prov=provenance[method]
        assert_provenance(prov,mem.normal_dyn["p0"][0]["id"][0].size())
        scorer=ScopedRetrieval(
            impl,copy.deepcopy(mem),prov,"spatial_0125")
        reference_metrics=d3_summary["arms"][method]
        target_hits=[]
        for rec,image,features in source_encoded:
            scorer.mask_provider=None
            scorer.provenance=tuple(prov)
            _=scorer.score(image,features,cfg,camera_id=int(rec.camera_id),
                           image_id=rec.rel_path)
            target_hits.append(scorer.last_query_stats[
                "n_dynamic_winner_patches"]/scorer.last_query_stats["n_query_patches"])
        target=sum(target_hits)/len(target_hits)
        source_calib["results"][method]={}
        for replicate in REPEATS:
            observed={}
            for p in SOURCE_CALIB_P_GRID:
                scorer.mask_provider=RandomEligibility(
                    "source_calibrated_hit_random",method=method,
                    replicate=replicate,probability=p)
                measures=[]
                for rec,image,features in source_encoded:
                    _=scorer.score(image,features,cfg,camera_id=int(rec.camera_id),
                                   image_id=rec.rel_path)
                    measures.append(scorer.last_query_stats[
                        "n_dynamic_winner_patches"]/scorer.last_query_stats["n_query_patches"])
                observed[p]=sum(measures)/len(measures)
            chosen=choose_probability(target,observed)
            source_calib["results"][method][str(replicate)]=chosen
            for kind in CONTROL_TYPES:
                if kind=="true_spatial_0125" and replicate>0:
                    continue
                if kind=="shuffled_spatial":
                    scorer.provenance=tuple(shuffled_spatial_provenance(
                        prov,method=method,replicate=replicate))
                    scorer.mask_provider=None
                elif kind=="matched_count_random":
                    scorer.provenance=tuple(prov)
                    scorer.mask_provider=RandomEligibility(
                        kind,method=method,replicate=replicate)
                elif kind=="source_calibrated_hit_random":
                    scorer.provenance=tuple(prov)
                    scorer.mask_provider=RandomEligibility(
                        kind,method=method,replicate=replicate,
                        probability=chosen["selected_p"])
                elif kind=="true_spatial_0125":
                    scorer.provenance=tuple(prov)
                    scorer.mask_provider=None
                trial=[]
                for rec,image,features in probes:
                    score,_=scorer.score(
                        image,features,cfg,camera_id=int(rec.camera_id),
                        image_id=rec.rel_path)
                    base=d3_summary["arms"][method]["global"]
                    # Evaluation only: static/global are read from frozen D3.
                    original=frozen_image_scores[(method,"global",rec.rel_path)]
                    static=float(original["static_score"])
                    global_score=float(original["score"])
                    if score<global_score-1e-6 or score>static+1e-6:
                        raise NativeParityError(
                            f"Random-control native bounds violated: "
                            f"{method} {kind} rep={replicate} {rec.rel_path} "
                            f"global={global_score} control={score} static={static}")
                    stat=scorer.last_query_stats
                    row={
                        "method":method,"control":kind,"replicate":replicate,
                        "image":rec.rel_path,"sample_key":rec.sample_key,
                        "camera_id":int(rec.camera_id),
                        "label_offline_only":int(not rec.is_good),
                        "score":float(score),"static_score":static,
                        "native_global_score":global_score,
                        "eligible_top5_fraction":stat["n_native_topk_eligible_patches"]/stat["n_query_patches"],
                        "dynamic_win_fraction":stat["n_dynamic_winner_patches"]/stat["n_query_patches"],
                        "eligible_prototype_pair_fraction":stat["eligible_prototype_pair_fraction"],
                        "source_selected_p":chosen["selected_p"] if kind=="source_calibrated_hit_random" else None,
                    }
                    result_rows.append(row)
                    trial.append(row)
                if len(trial)!=40:
                    raise NativeParityError("D3-C failed full paired Probe experiment")
                metrics=summary_metrics(
                    trial,theta=calibrated["theta"],
                    global_metrics={"static_fpr":d3_summary["static_metrics"]["fpr"],
                                    "global_fnr":base["fnr"]})
                if kind=="true_spatial_0125":
                    old=d3_summary["arms"][method]["spatial_0125"]
                    for k0,k1 in (("fnr","fnr"),("fpr","fpr"),("auroc","auroc"),
                                  ("mean_dynamic_winner_fraction",
                                   "mean_dynamic_winner_patch_fraction"),
                                  ("mean_top5_eligible_fraction",
                                   "mean_native_topk_eligible_patch_fraction"),
                                  ("defect_mean_score_erosion",
                                   "mean_positive_defect_image_erosion")):
                        if abs(metrics[k0]-old[k1])>1e-6:
                            raise NativeParityError(
                                f"Real spatial reference differs: {method}/{k0}")
                comparison.append({
                    "method":method,"control":kind,"replicate":replicate,
                    **metrics,"stored_prototypes":len(prov),
                    "source_selected_p":chosen["selected_p"] if kind=="source_calibrated_hit_random" else None,
                    "source_hit_target":target,
                    "source_hit_gap":chosen["source_absolute_hit_gap"]
                        if kind=="source_calibrated_hit_random" else None,
                    "probe_hit_gap_vs_real_spatial":metrics["mean_dynamic_winner_fraction"]-
                        reference_metrics["spatial_0125"]["mean_dynamic_winner_patch_fraction"],
                })
    if len(result_rows)!=(2*(1+3*3)*40):
        raise NativeParityError("D3-C missing 800 image scores")
    if len(comparison)!=20:
        raise NativeParityError("D3-C missing 20 source/replicate summary rows")
    report={
        "schema":SCHEMA,"completed":True,"config":config(args),
        "source_reference":parity,
        "native_global_parity_checks":d3_summary["actual_native_global_parity_checks"],
        "source_only_hit_calibration":True,
        "no_eval_probe_fit":True,"N_control_scores":len(result_rows),
        "N_arms":2,"N_repeats":3,"N_holdout_probe_images":40,
        "N_normal_probe":25,"N_defect_probe":15,
        "method_arms":list(arms),
        "comparison":comparison,
        "source_only_hit_calibration_inputs":[r.rel_path for r in source_records],
        "safety_interpretation":"Development Probe A only, no Probe B or deployability claim",
        "limitations":[
            "matched_count matches eligible counts not actual dynamic hits",
            "source_hit matching can miss target on heldout Probe: report achieved hit gaps",
            "native 5-neighbor average uses Static fallback when <5 eligible",
            "repeated random replicas share same 8 official Probe sample groups",
            "true spatial radius 0.125 was inspected previously on this development Probe",
        ],
    }
    out={
        SCORES:d2.csv_content((
            "method","control","replicate","image","sample_key","camera_id",
            "label_offline_only","score","static_score","native_global_score",
            "eligible_top5_fraction","dynamic_win_fraction",
            "eligible_prototype_pair_fraction","source_selected_p"),result_rows),
        PAIRED:d2.csv_content((
            "method","control","replicate","N","N_normal","N_defect",
            "tp","tn","fp","fn","fpr","fnr","auroc","auprc","theta0",
            "normal_mean_score_gain","defect_mean_score_erosion",
            "mean_top5_eligible_fraction","mean_dynamic_winner_fraction",
            "fpr_gain_vs_static","fnr_delta_vs_global",
            "stored_prototypes","source_selected_p",
            "source_hit_target","source_hit_gap",
            "probe_hit_gap_vs_real_spatial"),comparison),
    }
    return report,source_calib,out


def main(argv:Sequence[str]|None=None):
    args=parser().parse_args(argv)
    if args.json_dir is None:
        args.json_dir=str(Path(args.dataset_root)/pf.DEFAULT_JSON_SUBDIR)
    enforce(args)
    frozen,rows=validate_reference_d3(Path(args.reference_d3_dir))
    if args.preflight_only:
        current=native.preflight(args)
        if (frozen["config"]["fixed_quantile"]!=.90 or
                frozen["probe_images"]!=40):
            raise NativeParityError("Frozen D3 source context not q90 Probe4")
        print(json.dumps({"D3_C":"preflight_pass","reference_d3_native":True,
                          "reference_d3_scores":len(rows),
                          "controls":list(CONTROL_TYPES),
                          "replicates":list(REPEATS),
                          "source_normal_calibration_views":SOURCE_CALIB_N,
                          "gpu_used":False,"d3_metadata":current},indent=2))
        return 0
    if args.verify_only:
        return verify(args)
    dest=Path(args.out_dir)
    if dest.exists():
        raise FileExistsError(
            f"D3-C immutable output exists: {dest}; use --verify-only")
    started=time.perf_counter()
    with torch.no_grad():
        d3_result,d3_csvs,(report,calib,control_csvs)=native.run(
            args,controls_callback=lambda **kw:controls_callback(
                args=args,ref_summary=frozen,ref_rows=rows,**kw))
    report["elapsed_seconds"]=time.perf_counter()-started
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(
            dir=dest.parent,prefix=".d3c_build_") as folder:
        stage=Path(folder)/dest.name
        stage.mkdir()
        pf.write_json_idempotent(stage/SUMMARY,report)
        pf.write_json_idempotent(stage/SOURCE_CALIB,calib)
        for name,txt in control_csvs.items():
            (stage/name).write_text(txt,encoding="utf-8",newline="")
        provenance=manifest(args,report,calib,control_csvs)
        pf.write_json_idempotent(stage/MANIFEST,provenance)
        for name,h in provenance["output_sha256"].items():
            if pf.sha256_file(stage/name)!=h:
                raise NativeParityError(f"Staged randomized evidence corrupted: {name}")
        if dest.exists():
            raise FileExistsError("Output appeared before atomic publish")
        stage.rename(dest)
    print(json.dumps({"D3_C":"finished","N_scores":report["N_control_scores"],
                      "native_replay_parity":True,
                      "output_dir":str(dest),"research_status":"development_only"},
                     indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
