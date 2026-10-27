#!/usr/bin/env python3
"""D3/Mechanism-1: global vs conditional retrieval on FROZEN Greedy/Novelty.

Uses the original D1-B paired Q-A tape, strict4 support, q90 normal-only
registered threshold, and Probe-A only. Replays verified-normal updates exactly
as D2-A, maintaining one provenance row PER native dynamic prototype. Changes
RETRIEVAL ONLY at evaluation: global, camera, spatial, camera+spatial, two
pre-registered radii. No new training, no Sidecar, no evaluation label leakage.
No true patch mask is claimed: defect-image patch erosion is offline diagnostic.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

import torch

from . import d2_native_replay as d2
from . import preflight as pf
from . import d1d_native_smoke as d1d
from . import d2_baselines as d2b
from . import query_protocol as qp
from .conditional_retrieval import (
    POLICIES, ScopedRetrieval, assert_provenance, coordinates,
    patch_collision_rows)
from .d2_metrics import image_metrics
from .native_bridge import (
    NativeParityError, dynamic_codebooks, extract_v22_candidates,
    normalized_native_candidate_rows, native_score, snapshot, install_on_live)

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import p0_protocol_audit as p0
import v02_memory as memory
import v02_architecture_core as architecture

SCHEMA = "risk_safe.d3.conditional_retrieval_mechanism_pilot.v2"
RESULT = "summary.json"
PROBES = "retrieval_probe_scores.csv"
COLLISIONS = "collision_patch_attribution.csv"
WRITES = "native_write_provenance.csv"
TAPE = "causal_query_ledger.csv"
MANIFEST = "source_manifest.json"
FILES = (RESULT, PROBES, COLLISIONS, WRITES, TAPE, MANIFEST)
DEFAULT_OUT = ("outputs/risk_safe_sprint_20261027/"
               "d3_audiojack_s0_q37_b16_q90_probe4_fixedk")
SOURCES = (
    "experiments/risk_safe/conditional_retrieval.py",
    "experiments/risk_safe/conditional_retrieval_pilot.py",
    "experiments/risk_safe/retrieval_collision_audit.py",
    "experiments/risk_safe/d2_native_replay.py",
    "experiments/risk_safe/d2_baselines.py",
    "experiments/risk_safe/d2_fixed_threshold.py",
    "experiments/risk_safe/d2_metrics.py",
    "experiments/risk_safe/native_bridge.py",
    "experiments/risk_safe/preflight.py",
    "experiments/risk_safe/d1b_protocol.py",
    "experiments/risk_safe/d1d_native_smoke.py",
    "experiments/v02_memory.py",
    "experiments/v02_architecture_core.py",
    "experiments/p0_protocol_audit.py",
    "experiments/a0_a1_strict4_matched.py",
    "test_phase3_AIF_simplified_v22.py",
    "vmb_visionad_new.py",
    "tests/test_risk_safe_d3.py",
    "docs/risk_safe/D3_CONDITIONAL_RETRIEVAL_PILOT.md",
)


def parser():
    p = d2.parser()
    p.description = __doc__
    p.set_defaults(max_query_events=16, probe_groups_per_type=4,
                   out_dir=DEFAULT_OUT)
    p.add_argument("--max-collision-rows", default=1000, type=int)
    p.add_argument("--reference-d2a-summary", required=True,
                   help="Original frozen D2-A q90/q95 summary.json; must match"
                        " native global end-state metrics and prototype counts")
    return p


def config(args):
    parent = d2.configuration(args)
    return {**parent, "protocol": SCHEMA,
            "reference_d2a_summary": str(Path(args.reference_d2a_summary)).replace("\\", "/"),
            "retrieval_policies": list(POLICIES),
            "conditional_k": ("fixed_native_k=min(5,global_dynamic_bank_size); "
                              "if fewer eligible, Static-only fallback"),
            "static_bank": "always global and immutable",
            "patch_grid": "native aspect-aware _grid_hw_from_npatches",
            "source_prototype_update": "native greedy append only, fail on replacement",
            "q_a": "static-view prequery ranking, verified-normal only",
            "collision_top_rows": int(args.max_collision_rows),
            "retrieval_uses_test_labels": False,
            "risk_gate": "NOT IMPLEMENTED: testing context only, no D3 learned gate",
            "evaluation_scope": "Probe A only, never Probe B",
            "camera_id": "physical Real-IAD C1-C5; v22 identity augmentation is id",
            "spatial_radius": {"small": .125, "large": .25}}


def verify_d2a_reference(reference: dict, *, calibrated: dict,
                          tape_digest: str,
                          result_arms: dict, expected_images: int) -> dict:
    """Fail closed if D3 global replay diverges from immutable D2-A results."""
    if (reference.get("schema") != "risk_safe.d2.paired_native_dev_pilot.v1"
            or not reference.get("completed") or
            not reference.get("full_budget_completed") or
            reference.get("source_d1b_tape_sha256") != tape_digest or
            reference.get("actual_query_count") != 16 or
            reference.get("probe_image_count") != expected_images or
            reference.get("config", {}).get("fixed_quantile") != float(calibrated["quantile"]) or
            reference.get("fixed_training_normal_calibration", {}).get("sha256")
            != calibrated["sha256"]):
        raise NativeParityError("Frozen D2-A reference mismatch")
    for method in ("greedy", "causal_novelty"):
        current = result_arms[method]["global"]
        old = reference["per_arm"][method]
        if old["actual_final_prototypes"] != current["n_written"]:
            raise NativeParityError(f"Prototype count differs from D2-A: {method}")
        for k_new, k_old in (
                ("fpr", "final_fpr"), ("fnr", "final_fnr")):
            if abs(float(current[k_new])-float(old[k_old]))>1e-12:
                raise NativeParityError(f"D2-A classification parity failed {method}:{k_new}")
    return {
        "passed": True,
        "methods": ["greedy", "causal_novelty"],
        "matched": ["N_prototypes", "FPR", "FNR", "frozen_QA_tape",
                    "calibration_registry", "N_probe_images"],
        "reference_is_evaluation_only": True,
    }


def enforce_protocol(args):
    if not args.threshold_registry or args.fixed_quantile not in (.90, .95):
        raise NativeParityError(
            "Require independent q90/q95 normal-only fixed registry")
    if args.max_query_events != 16 or args.budget != 16:
        raise NativeParityError("D3 pilot requires complete frozen B16")
    if args.probe_groups_per_type != 4:
        raise NativeParityError("D3 pilot requires preregistered Probe A 4+4")
    if args.class_name != "audiojack" or args.support_seed != 0 or args.query_seed != 37:
        raise NativeParityError("D3 mechanism pilot is frozen to audiojack s0 q37")
    if args.max_collision_rows < 1:
        raise NativeParityError("Collision cap must be positive")
    if not args.reference_d2a_summary:
        raise NativeParityError("Frozen D2-A reference is mandatory")


def manifest(args, result, csv_outputs):
    return {
        "schema": "risk_safe.d3.conditional_retrieval_source.v2",
        "git": pf.git_state(),
        "source_sha256": pf.source_hashes(SOURCES),
        "input_sha256": {
            "real_iad_json": pf.sha256_file(
                Path(args.json_dir) / f"{args.class_name}.json"),
            "d1b_manifest": pf.sha256_file(
                Path(args.d1b_manifest_dir) / d2.d1b.OUTPUT_QUERIES),
            "training_normal_threshold_registry": pf.sha256_file(
                Path(args.threshold_registry)),
            "reference_d2a_summary": pf.sha256_file(
                Path(args.reference_d2a_summary)),
        },
        "config": config(args),
        "output_sha256": {
            RESULT: pf.sha256_text(pf.json_text(result)),
            **{name: pf.sha256_text(s) for name, s in csv_outputs.items()}},
        "environment": pf.environment_block(),
    }


def verify_only(args):
    dest = Path(args.out_dir)
    result = json.loads((dest / RESULT).read_text(encoding="utf-8"))
    if result.get("schema") != SCHEMA or result.get("config") != config(args):
        raise NativeParityError("D3 config/schema mismatch")
    if not result.get("completed") or not result.get("global_native_parity_pass"):
        raise NativeParityError("Incomplete or native-nonparity D3 output")
    existing = json.loads((dest / MANIFEST).read_text(encoding="utf-8"))
    for name, expected in existing["output_sha256"].items():
        if pf.sha256_file(dest / name) != expected:
            raise NativeParityError(f"D3 output SHA changed: {name}")
    prepared = pf.preserve_committed_source_provenance(
        dest / MANIFEST, manifest(args, result, {
            name: (dest / name).read_text(encoding="utf-8")
            for name in (PROBES, COLLISIONS, WRITES, TAPE)}))
    pf.verify_json_output(dest / MANIFEST, prepared)
    print(json.dumps({"D3": "verified_without_gpu",
                      "files": list(FILES),
                      "native_global_parity": True}, indent=2))
    return 0


def preflight(args):
    enforce_protocol(args)
    report = d2.metadata_preflight(args)
    if report["registered_fixed_threshold"] is None:
        raise NativeParityError("Missing validated normal-only fixed threshold")
    return {"D3": "metadata_preflight_pass", "model_gpu_used": False,
            "retrieval_policies": list(POLICIES),
            "paired_methods": ["greedy", "causal_novelty"],
            "d2_a": report}


@torch.no_grad()
def run(args, *, controls_callback=None):
    enforce_protocol(args)
    start = time.perf_counter()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise NativeParityError("CUDA unavailable")
    impl = p0.require_sources()
    impl.set_seed(0)
    if args.math_sdpa and args.device == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    train, test = pf.load_class_records(
        Path(args.dataset_root), Path(args.json_dir), args.class_name)
    plan = pf.build_partitions(train, test, pf.SplitConfig())
    calibrated = d2.load_fixed_threshold(args, plan)
    if calibrated is None:
        raise NativeParityError("No calibrated fixed threshold")
    tape, count = d1d.frozen_tape(plan, args)
    if count != 2625 or len(tape.events) != 16:
        raise NativeParityError("Frozen Q-A tape changed")
    keys = set(plan.support_samples_by_seed[0])
    supports = sorted(r.path for r in plan.support if r.sample_key in keys)
    probe_recs, probe_sample_keys = d2.probe_records(
        plan, args.probe_groups_per_type)
    if {r.sample_key for r in probe_recs} & {
            e.sample_key for e in tape.events}:
        raise NativeParityError("Q-A/probe leakage")
    metadata = {r.path: r for phase in plan.phases for r in phase}
    # Keep exactly the D2-A configuration names / values, no approximations.
    cfg = p0.Config(
        device=args.device, augmentation_seed=0, bank_seed=0,
        fixed_capacity=25000, fixed_per_image=5000,
        augmentations=4, include_identity=False, multiplier=2.5,
        feature_fuse="concat", topk_patches=32, image_aggregation="max",
        image_topk=64, image_quantile=.95, knn=5)
    backbone = impl.DINOv2MultiLayerBackbone(device=args.device)
    features_cache = p0.encode_reference_features(
        impl, backbone, supports, cfg)
    fixed = p0.build_vmb(impl, features_cache, supports, cfg)
    frozen_static = p0.fixed_bank_profile(fixed)
    settings = architecture.make_memory_settings(fixed)
    arms = {
        method: memory.MemoryArm("greedy_coreset", copy.deepcopy(fixed),
                                 settings, seed=37)
        for method in ("greedy", "causal_novelty")}
    states = {
        method: d2b.BaselineState(
            method, seed=qp.derived_seed(37, "D2", "audiojack", method))
        for method in arms}
    provenance: dict[str, list[dict]] = {m: [] for m in arms}
    ledger = []
    prequery_parity = 0
    for event in tape.events:
        rec = metadata.get(event.path)
        if rec is None or rec.rel_path != event.rel_path:
            raise NativeParityError("Frozen Q-A event not present in online stream")
        image, features = p0.encode_query(impl, backbone, event.path, cfg)
        static_score, amap = native_score(impl, fixed, image, features, cfg)
        prescores = {
            m: native_score(impl, arm.bank, image, features, cfg)[0]
            for m, arm in arms.items()}
        prequery_parity += 1
        revealed = "normal" if rec.is_good else "defect"
        ledger.append({
            "stream_t": event.position, "sample_key": event.sample_key,
            "camera_id": rec.camera_id, "image": rec.rel_path,
            "label_revealed_after_scoring": revealed,
            "static_prequery_score": static_score,
            "greedy_prequery_score": prescores["greedy"],
            "novelty_prequery_score": prescores["causal_novelty"],
            "candidate_selection": "none" if not rec.is_good else "native_static_topk",
            "frozen_tape_sha256": tape.digest,
        })
        if not rec.is_good:
            continue
        allowed, keep, _, _ = impl.select_informative_normal_write(
            score=static_score, theta=calibrated["theta"],
            sigmaN=max(calibrated["sigma_normal"], .02),
            pred_defect=bool(static_score >= calibrated["theta"]),
            novelty=0.)
        if not allowed or keep not in (8, 16, 24):
            raise NativeParityError("Native candidate selection contract violated")
        raw, indices = extract_v22_candidates(
            impl, amap, features, keep=keep)
        if len(features) != 1 or 0 not in indices:
            raise NativeParityError("D3 concat layer mismatch")
        uv = coordinates(impl, len(features[0]), (image.height, image.width))
        for method, arm in arms.items():
            current = snapshot(arm.bank)
            old_sha = current.sha256()
            rows = normalized_native_candidate_rows(
                impl, arm.bank, features, indices, keep=keep)
            outcome = d2b.propose_baseline(
                states[method], current, rows,
                raw_candidate_layers=raw, native_bank=arm.bank)
            states[method].record(outcome)
            old_len = len(current.layers[0])
            if outcome.decision == "admit":
                if outcome.replaced or outcome.skipped:
                    raise NativeParityError(
                        "D3 stage-1 provenance requires append-only bank")
                if outcome.added != keep or (
                        len(outcome.bank.layers[0]) != old_len + keep):
                    raise NativeParityError("Unexpected native append count")
                install_on_live(arm.bank, outcome.bank,
                                expected_digest=old_sha, version=current.version)
                newprotos = arm.bank.normal_dyn["p0"][0]["id"][0].protos
                if not torch.equal(newprotos[old_len:], rows[0].to(
                        device=newprotos.device, dtype=newprotos.dtype)):
                    raise NativeParityError("Provenance rows differ from committed rows")
                for j, i in enumerate(indices[0][:keep]):
                    provenance[method].append({
                        "prototype_index": old_len+j, "method": method,
                        "stream_t": event.position, "camera_id": int(rec.camera_id),
                        "sample_key": rec.sample_key, "image": rec.rel_path,
                        "patch_index": int(i), "u": float(uv[i, 0]),
                        "v": float(uv[i, 1]), "native_rank": j,
                    })
            elif snapshot(arm.bank).sha256() != old_sha:
                raise NativeParityError("Rejected novelty write mutated live Memory")
        ledger[-1]["candidate_keep"] = keep
    for method, arm in arms.items():
        if p0.fixed_bank_profile(arm.bank)["sha256"] != frozen_static["sha256"]:
            raise NativeParityError("Original support was modified")
        assert_provenance(
            provenance[method], len(dynamic_codebooks(arm.bank)[0].protos))
    if len(ledger) != 16 or sum(r["label_revealed_after_scoring"] == "normal"
                              for r in ledger) != 11:
        raise NativeParityError("Expected B16 and 11 verified normal events")
    if len(provenance["greedy"]) != 104 or len(provenance["causal_novelty"]) != 48:
        raise NativeParityError(
            "Rebuilt memories differ from D2-A registered 104/48 append state")

    # Cache each Probe A image's ORIGINAL encoding just once, across policies.
    probes = []
    for rec in probe_recs:
        image, features = p0.encode_query(impl, backbone, rec.path, cfg)
        probes.append((rec, image, features))
    scorers = {}
    for method, arm in arms.items():
        scorers[method] = {
            policy: ScopedRetrieval(impl, copy.deepcopy(arm.bank),
                                    provenance[method], policy)
            for policy in POLICIES}
    static_scorer = ScopedRetrieval(impl, copy.deepcopy(fixed), (), "global")
    probe_rows = []
    collision_rows = []
    parity_count = 0
    for i, (rec, image, features) in enumerate(probes):
        static_score = native_score(impl, fixed, image, features, cfg)[0]
        untouched_scores = {
            m: native_score(impl, arm.bank, image, features, cfg)[0]
            for m, arm in arms.items()}
        scores = {"static": static_score}
        for method, methods in scorers.items():
            for policy, scorer in methods.items():
                value, _ = scorer.score(
                    image, features, cfg, camera_id=int(rec.camera_id))
                native_global = untouched_scores[method]
                if policy == "global":
                    if value != native_global:
                        raise NativeParityError(
                            f"Global D3 != original v22: {method} {rec.rel_path} "
                            f"{value} != {native_global}")
                    parity_count += 1
                elif value + 1e-6 < native_global:
                    raise NativeParityError(
                        "Scoped memory improved more than global memory")
                if value > static_score + 1e-6:
                    raise NativeParityError(
                        "Scoped memory score exceeded frozen static score")
                scores[f"{method}:{policy}"] = value
                stats = scorer.last_query_stats
                eligible_fraction = (
                    stats["n_native_topk_eligible_patches"] / stats["n_query_patches"]
                    if stats is not None else None)
                dynamic_win_fraction = (
                    stats["n_dynamic_winner_patches"] / stats["n_query_patches"]
                    if stats is not None else None)
                probe_rows.append({
                    "image": rec.rel_path, "sample_key": rec.sample_key,
                    "camera_id": rec.camera_id,
                    "label_offline_only": int(not rec.is_good),
                    "method": method, "policy": policy,
                    "score": value, "static_score": static_score,
                    "native_global_score": native_global,
                    "image_score_erosion_vs_static": max(0., static_score - value),
                    "n_stored_prototypes": len(provenance[method]),
                    "theta_q90_or_q95": calibrated["theta"],
                    "native_topk_eligible_patch_fraction": eligible_fraction,
                    "dynamic_winner_patch_fraction": dynamic_win_fraction,
                    "eligible_prototype_pair_fraction": (
                        stats["eligible_prototype_pair_fraction"]
                        if stats is not None else None),
                    "native_topk_effective_k": (
                        stats["effective_native_dynamic_k"]
                        if stats is not None else None),
                })
            collisions = patch_collision_rows(
                impl, image, features, cfg,
                global_scorer=methods["global"],
                static_scorer=static_scorer,
                provenance=provenance[method],
                query_camera=int(rec.camera_id),
                label=int(not rec.is_good),
                image_id=rec.rel_path, source=method, query_index=i)
            collision_rows.extend(collisions)

    for method in arms:
        rows = [r for r in probe_rows
                if r["method"] == method and r["policy"] == "global"]
        if len(rows) != len(probes):
            raise NativeParityError("Missing global probe rows")
    if parity_count != 2*len(probes):
        raise NativeParityError("Not all Native Global parity checks executed")
    expected_n_defects = sum(not rec.is_good for rec, _, _ in probes)
    if expected_n_defects < 1 or expected_n_defects == len(probes):
        raise NativeParityError("No paired probe class balance")

    summary_methods = {}
    labels = [int(not rec.is_good) for rec, _, _ in probes]
    static_scores = [
        next(r["static_score"] for r in probe_rows
             if r["method"] == "greedy" and r["policy"] == "global"
             and r["image"] == rec.rel_path)
        for rec, _, _ in probes]
    static_metrics = image_metrics(labels, static_scores, calibrated["theta"])
    for method in arms:
        summary_methods[method] = {}
        for policy in POLICIES:
            subset = [r for r in probe_rows
                      if r["method"] == method and r["policy"] == policy]
            if len(subset) != len(probes):
                raise NativeParityError("Policy/probe pairing incomplete")
            m = image_metrics(
                [int(r["label_offline_only"]) for r in subset],
                [float(r["score"]) for r in subset],
                calibrated["theta"])
            damage = [r["image_score_erosion_vs_static"] for r in subset
                      if int(r["label_offline_only"]) == 1]
            collision = [
                r for r in collision_rows
                if r["source"] == method and int(r["label_offline_only"]) == 1]
            summary_methods[method][policy] = {
                **m, "fpr_gain_vs_static": static_metrics["fpr"] - m["fpr"],
                "fnr_change_vs_static": m["fnr"] - static_metrics["fnr"],
                "mean_positive_defect_image_erosion": sum(damage)/len(damage),
                "n_written": len(provenance[method]),
                "mean_native_topk_eligible_patch_fraction": (
                    sum(float(r["native_topk_eligible_patch_fraction"])
                        for r in subset) / len(subset)
                    if policy != "global" else None),
                "mean_dynamic_winner_patch_fraction": (
                    sum(float(r["dynamic_winner_patch_fraction"])
                        for r in subset) / len(subset)
                    if policy != "global" else None),
                # Collision attribution is calculated for native GLOBAL
                # versus Static, NOT independently for each scoped policy.
                "global_defect_collision_patches_reference_only":
                    len(collision) if policy == "global" else None,
                "global_top1_cross_camera_fraction_reference_only":
                    ((sum(r["cross_camera_top1_diagnostic"] for r in collision) /
                      len(collision)) if collision else None)
                    if policy == "global" else None,
            }
    reference = json.loads(Path(args.reference_d2a_summary).read_text(encoding="utf-8"))
    reference_parity = verify_d2a_reference(
        reference, calibrated=calibrated, tape_digest=tape.digest,
        result_arms=summary_methods, expected_images=len(probes))
    collision_rows.sort(key=lambda r: float(r["score_erosion"]), reverse=True)
    n_collisions_all = len(collision_rows)
    collision_rows = collision_rows[:args.max_collision_rows]
    output = {
        "schema": SCHEMA, "completed": True,
        "config": config(args),
        "status": "DEVELOPMENT_ONLY_NOT_RISK_AWARE_CONTROLLER",
        "source_d1b_tape_sha256": tape.digest,
        "threshold_calibration_sha256": calibrated["sha256"],
        "q": calibrated["quantile"], "theta": calibrated["theta"],
        "sigma_normal": calibrated["sigma_normal"],
        "N_QA_events": len(tape.events), "N_ver_normal": 11, "N_ver_defect": 5,
        "probe_groups": probe_sample_keys, "probe_images": len(probes),
        "probe_defect_images": expected_n_defects,
        "actual_native_global_parity_checks": parity_count,
        "global_native_parity_pass": True,
        "reference_d2a_replay_parity": reference_parity,
        "memory_prototypes": {m: len(p) for m,p in provenance.items()},
        "source_fixed_bank_sha256": frozen_static["sha256"],
        "static_metrics": static_metrics,
        "arms": summary_methods,
        "collision_rows_total": n_collisions_all,
        "collision_rows_saved_top": len(collision_rows),
        "collision_top1_is_not_full_top5_attribution": True,
        "gt_pixel_masks_used": False,
        "probe_B_used": False,
        "no_probe_admission_feedback": True,
        "confirmatory_novelty_or_safety_claim": False,
        "elapsed_seconds": time.perf_counter() - start,
    }
    csvs = {
        PROBES: d2.csv_content((
            "image", "sample_key", "camera_id", "label_offline_only",
            "method", "policy", "score", "static_score", "native_global_score",
            "image_score_erosion_vs_static", "n_stored_prototypes",
            "theta_q90_or_q95", "native_topk_eligible_patch_fraction",
            "dynamic_winner_patch_fraction",
            "eligible_prototype_pair_fraction",
            "native_topk_effective_k"), probe_rows),
        COLLISIONS: d2.csv_content((
            "source", "image", "query_index", "label_offline_only",
            "query_camera", "patch_index", "u", "v",
            "global_patch_score", "static_patch_score", "score_erosion",
            "fixed_top5_distance", "dynamic_top5_distance",
            "winning_dynamic_top1_index_diagnostic",
            "winning_dynamic_top1_camera_diagnostic",
            "winning_dynamic_top1_spatial_linf_diagnostic",
            "cross_camera_top1_diagnostic"), collision_rows),
        WRITES: d2.csv_content((
            "prototype_index", "method", "stream_t", "camera_id",
            "sample_key", "image", "patch_index", "u", "v", "native_rank"),
            [r for ps in provenance.values() for r in ps]),
        TAPE: d2.csv_content((
            "stream_t", "sample_key", "camera_id", "image",
            "label_revealed_after_scoring", "static_prequery_score",
            "greedy_prequery_score", "novelty_prequery_score",
            "candidate_selection", "candidate_keep", "frozen_tape_sha256"),
            ledger),
    }
    if controls_callback is not None:
        # Original 12-scoring-arm frozen-prototype clones are no longer used;
        # release them before creating controlled diagnostic scorers. Avoid
        # accumulating many copies of a large native 4-shot static bank on GPU.
        del scorers, static_scorer
        controls = controls_callback(
            impl=impl, backbone=backbone, cfg=cfg, plan=plan,
            arms=arms, provenance=provenance, probes=probes,
            calibrated=calibrated, d3_summary=output, d3_csvs=csvs)
        return output, csvs, controls
    return output, csvs


def main(argv: Sequence[str] | None = None):
    args = parser().parse_args(argv)
    if args.json_dir is None:
        args.json_dir = str(Path(args.dataset_root) / pf.DEFAULT_JSON_SUBDIR)
    if args.preflight_only:
        print(json.dumps(preflight(args), indent=2))
        return 0
    if args.verify_only:
        enforce_protocol(args)
        return verify_only(args)
    dest = Path(args.out_dir)
    if dest.exists():
        raise FileExistsError(
            f"D3 immutable output exists: {dest} (use --verify-only)")
    result, csvs = run(args)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            dir=dest.parent, prefix=".d3_build_") as tmp:
        stage = Path(tmp) / dest.name
        stage.mkdir()
        pf.write_json_idempotent(stage / RESULT, result)
        for name, txt in csvs.items():
            (stage / name).write_text(txt, encoding="utf-8", newline="")
        provenance = manifest(args, result, csvs)
        pf.write_json_idempotent(stage / MANIFEST, provenance)
        for name, hash_ in provenance["output_sha256"].items():
            if pf.sha256_file(stage / name) != hash_:
                raise NativeParityError(f"D3 evidence SHA mismatch: {name}")
        if dest.exists():
            raise FileExistsError("D3 destination appeared during write")
        stage.rename(dest)
    print(json.dumps({
        "D3": "completed_native_conditional_retrieval_dev",
        "global_native_parity": result["global_native_parity_pass"],
        "probe_images": result["probe_images"],
        "arms": {k: result["memory_prototypes"][k]
                 for k in ("greedy","causal_novelty")},
        "collision_rows": result["collision_rows_total"],
        "output_dir": str(dest),
        "confirmatory": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
