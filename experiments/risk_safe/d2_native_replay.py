#!/usr/bin/env python3
"""D2 paired real-v22 online MEMORY-ADMISSION replay, bounded first GPU pilot.

All six arms share a frozen Q-A tape, support, raw images, pre-label
candidates, theta0 and held-out probes. Human labels disclosed only for Q-A
events. Unqueried stream images are not encoded because all update arms use
only queried human-verified normal images, Q-A is exogenous, and the Sidecar,
query controller adaptation and Signed Correction are disabled.

Primary output = fixed-stream-position probe trajectories, independent
defect-probe counterfactual damage and per-candidate paired causal logs.
This is a development pilot, not a confirmatory safety guarantee. Native
CDARG whole-image anchors are a declared precursor to patch-level anchors.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from . import d1b_protocol as d1b
from . import d1d_native_smoke as d1d
from . import preflight as pf
from . import query_protocol as qp
from .admission import CDARG, CDARGConfig
from .anchor_store import VerifiedImage
from .d2_baselines import BASELINES, BaselineState, propose_baseline
from .d2_metrics import (image_metrics, probe_deltas, risk_prediction_pairs,
                         online_gate_elapsed)
from .d2_fixed_threshold import (canonical_quantile_name, validate_registry,
                                 REGISTRY as FIXED_REGISTRY, ThresholdAuditError)
from .native_bridge import (
    NativeScoreAdapter, NativeParityError, dynamic_codebooks,
    extract_v22_candidates, install_on_live, native_score,
    normalized_native_candidate_rows, snapshot, verify_untouched_parity,
    verify_native_update_parity,
)

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import p0_protocol_audit as p0
import v02_architecture_core as v02
import v02_memory as v02_memory

SCHEMA = "risk_safe.d2.paired_native_dev_pilot.v1"
METHODS = (*BASELINES, "cdarg")
CHECKPOINTS = (0, 875, 1750, 2625)
DEFAULT_OUT = "outputs/risk_safe_sprint_20261027/d2_audiojack_s0_q37_b16_pilot"
RESULT = "summary.json"
BASELINES_CSV = "baseline_comparison_dev.csv"
PROBE = "probe_snapshots.csv"
DECISIONS = "candidate_decisions.csv"
EVIDENCE = "query_ledger.csv"
SOURCE = "source_manifest.json"
FILES = (RESULT, BASELINES_CSV, PROBE, DECISIONS, EVIDENCE, SOURCE)
SOURCES = (
    "experiments/risk_safe/d2_native_replay.py",
    "experiments/risk_safe/d2_baselines.py",
    "experiments/risk_safe/d2_metrics.py",
    "experiments/risk_safe/d2_fixed_threshold.py",
    "experiments/risk_safe/d2_threshold_calibration.py",
    "docs/risk_safe/D2_A_FIXED_THRESHOLD_PROTOCOL.md",
    "tests/test_risk_safe_d2a.py",
    "experiments/risk_safe/native_bridge.py",
    "experiments/risk_safe/admission.py",
    "experiments/risk_safe/anchor_store.py",
    "experiments/risk_safe/shadow_memory.py",
    "experiments/risk_safe/d1d_native_smoke.py",
    "experiments/risk_safe/d1b_protocol.py",
    "experiments/risk_safe/query_protocol.py",
    "experiments/risk_safe/preflight.py",
    "experiments/v02_memory.py",
    "experiments/v02_architecture_core.py",
    "experiments/p0_protocol_audit.py",
    "experiments/a0_a1_strict4_matched.py",
    "test_phase3_AIF_simplified_v22.py",
    "vmb_visionad_new.py",
    "tests/test_risk_safe_d2.py",
    "docs/risk_safe/D2_NATIVE_DEV_PILOT_PROTOCOL.md",
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default=pf.DEFAULT_DATASET_ROOT)
    p.add_argument("--json-dir", default=None)
    p.add_argument("--class-name", choices=d1b.INITIAL_SMOKE_CLASSES,
                   default="audiojack")
    p.add_argument("--support-seed", choices=(0,), type=int, default=0)
    p.add_argument("--query-seed", choices=d1b.INITIAL_SMOKE_QUERY_SEEDS,
                   type=int, default=37)
    p.add_argument("--budget", choices=d1b.MAIN_BUDGETS, type=int, default=16)
    p.add_argument("--max-query-events", type=int, default=8)
    p.add_argument("--probe-groups-per-type", type=int, default=2)
    p.add_argument("--threshold-registry", default=None,
                   help="Frozen D2-A training-normal threshold_registry.json")
    p.add_argument("--fixed-quantile", type=float, choices=(0.90, 0.95),
                   default=None, help="Registered q90 or q95; requires registry")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--math-sdpa", action="store_true")
    p.add_argument("--d1b-manifest-dir", default=d1d.DEFAULT_D1B)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--preflight-only", action="store_true",
                   help="metadata/tape/probe audit without loading DINOv2/GPU")
    return p


def configuration(args) -> dict:
    return {
        "class": args.class_name, "support_seed": args.support_seed,
        "query_seed": args.query_seed, "budget": args.budget,
        "max_query_events": args.max_query_events,
        "probe_groups_per_type": args.probe_groups_per_type,
        "threshold_registry": (str(Path(args.threshold_registry)).replace("\\", "/")
                               if args.threshold_registry else None),
        "fixed_quantile": args.fixed_quantile,
        "device": args.device, "math_sdpa": args.math_sdpa,
        "dataset_root": str(args.dataset_root).replace("\\", "/"),
        "json_dir": str(args.json_dir).replace("\\", "/"),
        "d1b_manifest_dir": str(args.d1b_manifest_dir).replace("\\", "/"),
        "stream_mode": "random",
        "probe": "A only; predetermined first N normal and N defect session groups; all 5 images each",
        "probe_checkpoints": list(CHECKPOINTS),
        "methods": list(METHODS),
        "threshold": ("training_normal_only_registered_fixed_quantile_no_sidecar"
                      if args.threshold_registry else "LOSO theta0 fixed, no Sidecar"),
        "correction": "disabled",
        "query": "frozen Q-A exogenous sample/camera tape",
        "candidate": "shared native v22 static scorer and 8/16/24 patch rank",
        "cdarg_anchor_scoring": "native whole-image verified-view anchors, not pixel mask",
        "cdarg_policy": {
            "epsilon_defect": 0.1, "lambda_defect": 1.0,
            "tau_utility": 0.0, "margin_sigma_fraction": 0.25,
            "sigma_normal": ("reserved 64 training-normal score std"
                             if args.threshold_registry else "support-only LOSO std"),
            "sigma_floor": 0.01, "zero_defect_anchor": "reject",
            "calibration_status": "PILOT_FIXED_UNTUNED_NOT_DEPLOYMENT_SAFE",
        },
        "normal_adaptation": "v22 native scorer before label; same candidates per arm",
        "causal_novelty": "median over only previously queried verified-normal candidate novelties, first-eligible normal admitted",
        "reservoir": "independent seeded dynamic patch reservoir; static support immutable",
        "append_at_capacity": "reject entire event, never expand capacity",
        "evaluation_only_probe_prohibited_from_online_policies": True,
    }


def probe_records(plan, n):
    """Pick sample groups as held-out evaluation ONLY; retain all camera views.

    An NG session can contain normal views: use per-image labels, not group
    propagation. Selection stratifies OFFLINE by presence of at least one
    defect-labeled view; it never affects candidate admission/queries.
    """
    grouped = pf.group_by_sample(plan.probe_a)
    normal = sorted(k for k, g in grouped.items()
                    if all(r.is_good for r in g))
    defect = sorted(k for k, g in grouped.items()
                    if any(not r.is_good for r in g))
    if len(normal) < n or len(defect) < n:
        raise NativeParityError("Insufficient fixed probe groups")
    selected = normal[:n] + defect[:n]
    records = [r for k in selected for r in
               sorted(grouped[k], key=lambda v: v.camera_id)]
    if len(records) != 10 * n:
        raise NativeParityError("Probe must have five images per selected group")
    if not any(not r.is_good for r in records) or not any(r.is_good for r in records):
        raise NativeParityError("Probe must include both image-level labels")
    return records, selected


def csv_content(columns, rows):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=columns, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in columns})
    return text.getvalue()


def sha_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def score_probe(impl, bank, probe, cfg, theta, ref=None):
    labels = [p[0] for p in probe]
    scores = [native_score(impl, bank, img, feats, cfg)[0]
              for _, img, feats in probe]
    results = image_metrics(labels, scores, theta)
    if ref is not None:
        results.update(probe_deltas(
            scores, labels, ref["scores"], results, ref["metrics"]))
    return {"scores": scores, "metrics": results}


def load_fixed_threshold(args, plan):
    """Frozen registry verified against current class partition and input JSON.

    No outcome/probe metric is ever consulted. Failure is fatal before DINOv2
    is initialised. Both quantile choices require separate admission replays.
    """
    if bool(args.threshold_registry) != (args.fixed_quantile is not None):
        raise ThresholdAuditError(
            "--threshold-registry and --fixed-quantile must be supplied together")
    if not args.threshold_registry:
        return None
    path = Path(args.threshold_registry)
    data = json.loads(path.read_text(encoding="utf-8"))
    validated = validate_registry(
        data, plan=plan, class_name=args.class_name,
        support_seed=args.support_seed,
        class_json_path=Path(args.json_dir) / (args.class_name + ".json"),
        registry_path=path, strict_manifest=True)
    quantile = canonical_quantile_name(float(args.fixed_quantile))
    result = {
        "theta": float(validated["theta_by_quantile"][quantile]["theta"]),
        "sigma_normal": float(validated["sigma_dev_normal"]),
        "quantile": quantile,
        "sha256": validated["calibration_sha256"],
        "n_dev_normal": validated["n_training_normal_views"],
        "reference_loso_theta": float(data["loso_reference_theta0"]),
        "reference_loso_sigma": float(data["loso_reference_sigma"]),
    }
    if result["sigma_normal"] <= 0:
        raise ThresholdAuditError("Calibration normal scale must be positive")
    return result


def manifest(args, result, csv_outputs):
    return {
        "schema": "risk_safe.d2.source_manifest.v1",
        "git": pf.git_state(),
        "sources_sha256": pf.source_hashes(SOURCES),
        "input_sha256": {
            "realiad_class_json": pf.sha256_file(
                Path(args.json_dir) / (args.class_name + ".json")),
            "d1b_query_manifest": pf.sha256_file(
                Path(args.d1b_manifest_dir) / d1b.OUTPUT_QUERIES),
            "registered_fixed_threshold": (pf.sha256_file(Path(args.threshold_registry))
                if args.threshold_registry else None),
        },
        "config": configuration(args),
        "output_sha256": {RESULT: pf.sha256_text(pf.json_text(result)),
                          **{k: sha_text(v) for k, v in csv_outputs.items()}},
        "environment": {**pf.environment_block(),
                        "device": args.device,
                        "cuda_initialised": args.device == "cuda",
                        "features_extracted": True},
    }


def verify_only(args):
    dest = Path(args.out_dir)
    result = json.loads((dest / RESULT).read_text(encoding="utf-8"))
    if result.get("schema") != SCHEMA or result.get("config") != configuration(args):
        raise NativeParityError("D2 result schema or configuration differs")
    if not result.get("completed"):
        raise NativeParityError("D2 is incomplete")
    existing = json.loads((dest / SOURCE).read_text(encoding="utf-8"))
    for name, expected in existing["output_sha256"].items():
        actual = pf.sha256_file(dest / name)
        if actual != expected:
            raise NativeParityError(f"D2 output tampering: {name}")
    prospective = manifest(
        args, result, {k: (dest / k).read_text(encoding="utf-8")
                       for k in (BASELINES_CSV, PROBE, DECISIONS, EVIDENCE)})
    prospective = pf.preserve_committed_source_provenance(
        dest / SOURCE, prospective)
    pf.verify_json_output(dest / SOURCE, prospective)
    print(json.dumps({"stage": "D2", "verify_only": True,
                      "outputs": {name: "verified" for name in FILES},
                      "gpu_replayed": False}, indent=2))
    return 0


@torch.no_grad()
def run(args):
    if not (1 <= args.max_query_events <= args.budget):
        raise ValueError("max-query-events must be in 1..B")
    if not (1 <= args.probe_groups_per_type <= 24):
        raise ValueError("probe-groups-per-type must be in 1..24")
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
    fixed_threshold = load_fixed_threshold(args, plan)
    chosen = set(plan.support_samples_by_seed[args.support_seed])
    supports = sorted(r.path for r in plan.support if r.sample_key in chosen)
    if len(supports) != 4:
        raise NativeParityError("Support count not equal 4")
    tape, stream_size = d1d.frozen_tape(plan, args)
    if stream_size != 2625:
        raise NativeParityError("Stream size differs from frozen D1-B")
    metadata = {r.path: r for r in test}
    selected_probes, selected_probe_ids = probe_records(
        plan, args.probe_groups_per_type)
    if set(x.sample_key for x in selected_probes) & {
            event.sample_key for event in tape.events}:
        raise NativeParityError("Probe and Q-A queried sample overlap")

    cfg = p0.Config(
        device=args.device, augmentation_seed=0, bank_seed=0,
        fixed_capacity=25000, fixed_per_image=5000,
        augmentations=4, include_identity=False, multiplier=2.5,
        feature_fuse="concat", topk_patches=32,
        image_aggregation="max", image_topk=64,
        image_quantile=.95, knn=5)
    backbone = impl.DINOv2MultiLayerBackbone(device=args.device)
    refs = p0.encode_reference_features(impl, backbone, supports, cfg)
    fixed = p0.build_vmb(impl, refs, supports, cfg)
    fixed_profile = p0.fixed_bank_profile(fixed)
    loso = []
    for held in supports:
        fold = p0.build_vmb(impl, refs, [p for p in supports if p != held], cfg)
        loso.append(p0.score_image(impl, backbone, fold, held, cfg))
        del fold
    theta_info = p0.threshold_from_scores(loso, 2.5)
    theta_loso, sigma_loso = theta_info["theta"], theta_info["sigma"]
    theta, sigma = theta_loso, sigma_loso
    if fixed_threshold is not None:
        theta = fixed_threshold["theta"]
        sigma = fixed_threshold["sigma_normal"]
    del refs
    settings = v02.make_memory_settings(fixed)

    # Keep independently preselected probe features entirely separate from
    # online admission/query methods. They are scored ONLY by probe evaluator.
    probes = []
    for r in selected_probes:
        image, features = p0.encode_query(impl, backbone, r.path, cfg)
        probes.append((int(not r.is_good), image, features))
    all_probe_labels = [x[0] for x in probes]
    defect_probes = [p for p in probes if p[0] == 1]
    if not defect_probes:
        raise NativeParityError("No image-level defect in evaluation probe")

    arms = {name: v02_memory.MemoryArm(
        "greedy_coreset", copy.deepcopy(fixed), settings,
        seed=args.support_seed + 37) for name in METHODS}
    baseline_states = {
        name: BaselineState(name, seed=qp.derived_seed(
            args.query_seed, "D2", args.class_name, name))
        for name in BASELINES}
    versions = {name: 0 for name in METHODS}
    ledger = d1d.NativeLedger()
    control = CDARG(CDARGConfig(
        epsilon_defect=.1, lambda_defect=1, tau_utility=0,
        defect_margin=.25 * sigma,
        sigma_normal=sigma, sigma_floor=.01), lambda x, bank: 0.)
    # Override the scorer per-arm only at actual evaluation time.
    probe_references = {}
    probe_rows = []
    candidate_rows = []
    query_rows = []
    initial_digests = {name: snapshot(arm.bank).sha256()
                       for name, arm in arms.items()}
    if len(set(initial_digests.values())) != 1:
        raise NativeParityError("Initial dynamic memories are not paired")
    start = time.perf_counter()
    counts = {name: {"candidate_events": 0, "admitted_events": 0,
                     "added": 0, "replaced": 0, "skipped": 0,
                     "admission_seconds": 0.,
                     "offline_probe_diagnostic_seconds": 0.}
              for name in METHODS}

    def evaluate_checkpoint(checkpoint):
        for name, arm in arms.items():
            profile = score_probe(
                impl, arm.bank, probes, cfg, theta,
                ref=probe_references.get(name))
            if name not in probe_references:
                probe_references[name] = profile
            mt = profile["metrics"]
            probe_rows.append({
                "stream_t": checkpoint, "arm": name,
                **mt, "bank_version": versions[name],
                "bank_prototypes": sum(len(cb.protos)
                         for cb in dynamic_codebooks(arm.bank)),
                "bank_sha256": snapshot(arm.bank, version=versions[name]).sha256(),
            })

    # Use the same pre-registered STREAM positions regardless of how many
    # updates an arm accepts. A bounded run is NOT a full-budget evaluation.
    events = tuple(tape.events[:args.max_query_events])
    evaluate_checkpoint(0)
    idx = 0
    for checkpoint in CHECKPOINTS[1:]:
        while idx < len(events) and events[idx].position <= checkpoint:
            item = events[idx]
            idx += 1
            rec = metadata.get(item.path)
            if rec is None or rec.rel_path != item.rel_path:
                raise NativeParityError("Q-A event does not resolve")
            image, features = p0.encode_query(impl, backbone, item.path, cfg)
            static_score, amap = native_score(impl, fixed, image, features, cfg)
            prequery_scores = {
                name: native_score(impl, arm.bank, image, features, cfg)[0]
                for name, arm in arms.items()}
            # Offline Q-A tape purchased this image's human label; reveal
            # after all prequery scores and candidate ranking.
            label = "normal" if rec.is_good else "defect"
            event = VerifiedImage(
                query_id=f"D2:{args.class_name}:{args.query_seed}:{item.position}",
                image_id=item.rel_path, sample_key=item.sample_key,
                camera_id=item.camera_id, label=label,
                query_decision_step=item.position,
                label_revealed_step=item.position)
            query_rows.append({
                "stream_t": item.position, "camera_id": item.camera_id,
                "image": item.rel_path, "label_after_query": label,
                "static_pre_score": static_score,
                "n_prior_normal": len(ledger.normal),
                "n_prior_defect": len(ledger.defect),
                "tape_sha256": tape.digest,
            })
            if label == "normal":
                allowed, keep, _, _ = impl.select_informative_normal_write(
                    score=static_score, theta=theta, sigmaN=max(sigma, .02),
                    pred_defect=bool(static_score >= theta), novelty=0.0)
                if not allowed or keep not in (8, 16, 24):
                    raise NativeParityError("Invalid native candidate plan")
                raw_candidates, candidates_idx = extract_v22_candidates(
                    impl, amap, features, keep=keep)
                for name, arm in arms.items():
                    began = time.perf_counter()
                    before = snapshot(arm.bank, version=versions[name])
                    old_sha = before.sha256()
                    offline_probe_seconds = 0.0
                    rows = normalized_native_candidate_rows(
                        impl, arm.bank, features, candidates_idx, keep=keep)
                    details = {"novelty": None, "novelty_cutoff": None,
                               "normal_utility": None, "defect_risk": None,
                               "margin_crossing": None, "probe_damage": None}
                    if name == "cdarg":
                        control.scorer = NativeScoreAdapter(impl, arm.bank, cfg)
                        proposal = control.evaluate(
                            current_verified_normal=event, step=item.position,
                            bank=before, ledger=ledger,
                            candidate_patches=rows, theta=theta,
                            candidate_already_normalized=True)
                        decision, reason = proposal.decision, proposal.reason
                        update = proposal.shadow
                        audit = proposal.greedy_audit
                        added, replaced, skipped = (
                            audit.appended, audit.replaced, audit.skipped)
                        details.update({
                            "normal_utility": proposal.normal_utility,
                            "defect_risk": proposal.defect_risk,
                            "margin_crossing": proposal.defect_margin_crossing,
                        })
                        # Independent probe damage measured AFTER the gate
                        # decision, never supplied to the gate.
                        # Entirely OFFLINE, deliberately excluded from online
                        # gate latency. This diagnostic observes heldout probe
                        # features ONLY after proposal.decision is fixed.
                        offline_probe_start = time.perf_counter()
                        rescoring = NativeScoreAdapter(impl, arm.bank, cfg)
                        harms = []
                        for _, probe_img, probe_features in defect_probes:
                            old_s = rescoring((probe_img, probe_features), before)
                            new_s = rescoring((probe_img, probe_features), update)
                            harms.append(max(0., old_s - new_s))
                        details["probe_damage"] = sum(harms) / len(harms)
                        offline_probe_seconds = (
                            time.perf_counter() - offline_probe_start)
                    else:
                        outcome = propose_baseline(
                            baseline_states[name], before, rows,
                            raw_candidate_layers=raw_candidates,
                            native_bank=arm.bank)
                        decision, reason = outcome.decision, outcome.reason
                        update = outcome.bank
                        added, replaced, skipped = (
                            outcome.added, outcome.replaced, outcome.skipped)
                        details.update({
                            "novelty": outcome.novelty,
                            "novelty_cutoff": outcome.novelty_cutoff})
                        baseline_states[name].record(outcome)
                    if decision == "admit":
                        if name == "cdarg":
                            # Validate canonical gate commit on isolated bank
                            # BEFORE writing to real native memory.
                            if not CDARG.commit_if_accepted(proposal, before):
                                raise NativeParityError("Accepted gate failed commit")
                            update = before
                        install_on_live(
                            arm.bank, update,
                            expected_digest=old_sha, version=versions[name])
                        versions[name] += 1
                        counts[name]["admitted_events"] += 1
                    elif snapshot(arm.bank, version=versions[name]).sha256() != old_sha:
                        raise NativeParityError("Rejected proposal mutated native memory")
                    counts[name]["candidate_events"] += 1
                    counts[name]["added"] += added if decision == "admit" else 0
                    counts[name]["replaced"] += replaced if decision == "admit" else 0
                    counts[name]["skipped"] += skipped
                    latency = online_gate_elapsed(
                        time.perf_counter() - began, offline_probe_seconds)
                    counts[name]["admission_seconds"] += latency
                    counts[name]["offline_probe_diagnostic_seconds"] += (
                        offline_probe_seconds)
                    candidate_rows.append({
                        "stream_t": item.position, "method": name,
                        "verified_label": label, "candidate_keep": keep,
                        "decision": decision, "reason": reason,
                        "prior_normal": len(ledger.normal),
                        "prior_defect": len(ledger.defect),
                        "prequery_score": prequery_scores[name],
                        "normal_utility": details["normal_utility"],
                        "defect_risk": details["defect_risk"],
                        "defect_margin_crossing": details["margin_crossing"],
                        "novelty": details["novelty"],
                        "novelty_cutoff": details["novelty_cutoff"],
                        "independent_probe_erosion": details["probe_damage"],
                        "added": added if decision == "admit" else 0,
                        "replaced": replaced if decision == "admit" else 0,
                        "skipped": skipped,
                        "latency_seconds": latency,
                        "offline_probe_diagnostic_seconds": offline_probe_seconds,
                        "pre_sha256": old_sha,
                        "post_sha256": snapshot(
                            arm.bank, version=versions[name]).sha256(),
                    })
            # Shared ledger with only verified image views, appended strictly
            # after decisions across ALL arms.
            ledger.add(event, image, features)
        evaluate_checkpoint(checkpoint)

    if idx != len(events):
        raise NativeParityError("Query execution did not cover eligible events")
    for name, arm in arms.items():
        if p0.fixed_bank_profile(arm.bank)["sha256"] != fixed_profile["sha256"]:
            raise NativeParityError(f"{name}: immutable 4-shot support mutated")
    if versions["static"] != 0 or snapshot(arms["static"].bank).sha256() != initial_digests["static"]:
        raise NativeParityError("Static baseline dynamically adapted")
    probe_base = {name: next(r for r in probe_rows
                             if r["arm"] == name and r["stream_t"] == 0)
                  for name in METHODS}
    results = {}
    for name in METHODS:
        curve = [r for r in probe_rows if r["arm"] == name]
        original = probe_base[name]
        for row in curve:
            row["fnr_degradation_positive"] = max(
                0., float(row["fnr"]) - float(original["fnr"]))
            row["fpr_gain"] = float(original["fpr"]) - float(row["fpr"])
        summary = counts[name].copy()
        summary.update({
            "actual_final_prototypes": curve[-1]["bank_prototypes"],
            "initial_fnr": original["fnr"], "final_fnr": curve[-1]["fnr"],
            "initial_fpr": original["fpr"], "final_fpr": curve[-1]["fpr"],
            "final_fpr_gain": curve[-1]["fpr_gain"],
            "fnr_degradation_auc_fixed_checkpoints":
                sum(r["fnr_degradation_positive"] for r in curve[1:]) / (len(curve)-1),
            "final_defect_score_erosion":
                float(curve[-1].get("defect_erosion_positive_mean", 0)),
            "N_checkpoints": len(curve),
        })
        results[name] = summary
    pred_rows = [r for r in candidate_rows if r["method"] == "cdarg"]
    payload = {
        "schema": SCHEMA, "completed": True, "stage": "D2",
        "kind": "REAL_V22_MEMORY_ADMISSION_DEVELOPMENT_PILOT",
        "config": configuration(args),
        "source_boundary": "no test/probe label available to query or gate; probe evaluation used only after decisions",
        "full_budget_completed": args.max_query_events == args.budget,
        "no_all_stream_image_inference": True,
        "checkpoint_mean_is_not_continuous_time_integral": True,
        "actual_query_count": len(events),
        "normal_label_count": sum(r["label_after_query"] == "normal" for r in query_rows),
        "defect_label_count": sum(r["label_after_query"] == "defect" for r in query_rows),
        "probe_A_sample_keys_offline_only": selected_probe_ids,
        "probe_image_count": len(probes),
        "probe_image_defect_count": len(defect_probes),
        "theta0": theta, "sigma_normal_used": sigma,
        "theta_loso_reference": theta_loso, "sigma_loso": sigma_loso,
        "fixed_training_normal_calibration": fixed_threshold,
        "source_d1b_tape_sha256": tape.digest,
        "initial_static_support": fixed_profile,
        "per_arm": results,
        "cdarg_risk_prediction": risk_prediction_pairs(
            [{"defect_risk": r["defect_risk"],
              "independent_probe_erosion": r["independent_probe_erosion"]}
             for r in pred_rows]),
        "elapsed_seconds": time.perf_counter() - start,
        "gpu_memory_peak_bytes": (torch.cuda.max_memory_allocated()
                                  if args.device == "cuda" else None),
        "latency_accounting": (
            "Admission time excludes independent offline defect-probe "
            "counterfactual rescoring; its separate timing is diagnostic only; "
            "GPU timings are exploratory host wall times"),
        "cdarg_anchor_variant": "whole-image native score; patch-anchor variant not yet implemented",
        "hyperparameters_selected_without_probe_optimisation": True,
        "confirmatory_safety_claim": False,
    }
    outputs = {
        BASELINES_CSV: csv_content((
            "method", "candidate_events", "admitted_events",
            "added", "replaced", "skipped", "admission_seconds",
            "offline_probe_diagnostic_seconds",
            "initial_fnr", "final_fnr", "initial_fpr", "final_fpr",
            "final_fpr_gain", "fnr_degradation_auc_fixed_checkpoints",
            "final_defect_score_erosion", "actual_final_prototypes",
            "N_checkpoints"),
            [{"method": k, **v} for k, v in results.items()]),
        PROBE: csv_content((
            "stream_t", "arm", "N", "N_normal", "N_defect", "tp",
            "tn", "fp", "fn", "fpr", "fnr", "auroc", "auprc",
            "theta0", "defect_erosion_positive_mean",
            "defect_erosion_positive_max", "fnr_delta",
            "fnr_degradation_positive", "fpr_gain",
            "bank_version", "bank_prototypes", "bank_sha256"), probe_rows),
        DECISIONS: csv_content((
            "stream_t", "method", "verified_label", "candidate_keep",
            "decision", "reason", "prior_normal", "prior_defect",
            "prequery_score", "normal_utility", "defect_risk",
            "defect_margin_crossing", "novelty", "novelty_cutoff",
            "independent_probe_erosion", "added", "replaced", "skipped",
            "latency_seconds", "offline_probe_diagnostic_seconds",
            "pre_sha256", "post_sha256"), candidate_rows),
        EVIDENCE: csv_content((
            "stream_t", "camera_id", "image", "label_after_query",
            "static_pre_score", "n_prior_normal", "n_prior_defect",
            "tape_sha256"), query_rows),
    }
    return payload, outputs


def metadata_preflight(args):
    train, test = pf.load_class_records(
        Path(args.dataset_root), Path(args.json_dir), args.class_name)
    plan = pf.build_partitions(train, test, pf.SplitConfig())
    calibrated = load_fixed_threshold(args, plan)
    tape, stream_count = d1d.frozen_tape(plan, args)
    selected, sample_ids = probe_records(plan, args.probe_groups_per_type)
    stream_samples = set(k for phase in plan.phase_samples for k in phase)
    if set(sample_ids) & stream_samples:
        raise NativeParityError("Probe group overlaps online stream")
    if len(tape.events) != args.budget or len(set(tape.sample_keys)) != args.budget:
        raise NativeParityError("Tape counts or unique official sample keys invalid")
    if not (1 <= args.max_query_events <= args.budget):
        raise NativeParityError("Invalid max-query-events")
    if not (1 <= args.probe_groups_per_type <= 24):
        raise NativeParityError("Invalid probe group count")
    return {
        "stage": "D2", "kind": "metadata_preflight_only",
        "class": args.class_name, "support_seed": args.support_seed,
        "query_seed": args.query_seed, "budget": args.budget,
        "replayed_query_count": args.max_query_events,
        "full_budget": args.max_query_events == args.budget,
        "stream_images": stream_count, "heldout_probe_images": len(selected),
        "heldout_probe_image_defects": sum(not r.is_good for r in selected),
        "registered_fixed_threshold": calibrated,
        "frozen_query_tape_sha256": tape.digest,
        "normal_labels_in_online_stream_not_exposed": True,
        "gpu_used": False,
    }


def main(argv: Sequence[str] | None = None):
    args = parser().parse_args(argv)
    if args.json_dir is None:
        args.json_dir = str(Path(args.dataset_root) / pf.DEFAULT_JSON_SUBDIR)
    if args.preflight_only:
        print(json.dumps(metadata_preflight(args), indent=2))
        return 0
    if args.verify_only:
        return verify_only(args)
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(
            f"D2 output directory exists; use --verify-only or NEW out-dir: {out}")
    with torch.no_grad():
        result, csv_files = run(args)
    if not result["completed"]:
        raise NativeParityError("D2 pilot incomplete")

    # Transactional output publication: all five evidence files and manifest
    # must exist and match their hashes BEFORE the final directory appears.
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            dir=out.parent, prefix=".d2_build_") as temporary:
        stage = Path(temporary) / out.name
        stage.mkdir()
        pf.write_json_idempotent(stage / RESULT, result)
        for name, value in csv_files.items():
            (stage / name).write_text(value, encoding="utf-8", newline="")
        provenance = manifest(args, result, csv_files)
        pf.write_json_idempotent(stage / SOURCE, provenance)
        for name, expected in provenance["output_sha256"].items():
            if pf.sha256_file(stage / name) != expected:
                raise NativeParityError(
                    f"D2 staged evidence SHA-256 mismatch: {name}")
        if out.exists():
            raise FileExistsError(f"D2 destination appeared concurrently: {out}")
        stage.rename(out)
    print(json.dumps({"stage": "D2",
                      "completed": True,
                      "max_queries": args.max_query_events,
                      "probes": result["probe_image_count"],
                      "baseline_arms": list(METHODS),
                      "FNR_FPR_measured": True,
                      "CDARG_proposals": result["per_arm"]["cdarg"]["candidate_events"],
                      "provenance": "saved"},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
