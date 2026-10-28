#!/usr/bin/env python3
"""R0 native-v22 mechanism activation, NOT an Admission algorithm competition.

Query-event condensed causal Q-A replay: all events are fixed label-blind before
encoding. Unqueried images cannot update any state; NOT full-stream prequential
evaluation. Normal candidates use original static-score/top-patch generation.
Each capacity runs an independent native Greedy bank. Historical image anchors
are used for read-only shadow risk audits; their current queried image is
excluded. Probe A is offline evaluation-only; Probe B remains untouched.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

from . import preflight as pf
from . import d1d_native_smoke as d1d
from . import d2_native_replay as d2
from . import d1b_protocol as d1b
from . import query_protocol as qp
from .native_bridge import (NativeParityError, install_on_clone,
    dynamic_codebooks, extract_v22_candidates, install_on_live, native_score,
    normalized_native_candidate_rows, snapshot, verify_untouched_parity)
from .d2_metrics import image_metrics, probe_deltas
from .r0_opportunity import integers, native_output_name, probe_subset

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import p0_protocol_audit as p0
import v02_architecture_core as v02
import v02_memory as memory

SCHEMA = "risk_safe.r0.native_activation.v1"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default=pf.DEFAULT_DATASET_ROOT)
    p.add_argument("--json-dir", default=None)
    p.add_argument("--class-name", default="audiojack", choices=pf.CLASSES)
    p.add_argument("--support-seed", type=int, default=0, choices=d1b.SUPPORT_SEEDS)
    p.add_argument("--query-seed", type=int, default=37, choices=d1b.QUERY_SEEDS)
    p.add_argument("--budget", type=int, default=128,
                   choices=tuple(sorted(set(d1b.MAIN_BUDGETS + d1b.SENSITIVITY_BUDGETS +
                                            d1b.LOW_BUDGET_DIAGNOSTIC))))
    p.add_argument("--max-query-events", type=int, default=0,
                   help="0 = all B; positive integer = declared incomplete smoke")
    p.add_argument("--capacities", default="64,128,256,1024")
    p.add_argument("--audit-every", type=int, default=1,
                   help="Every Nth verified-normal proposal; all decisions still counted")
    p.add_argument("--anchors-per-label", type=int, default=4,
                   help="Most recent prior verified images per label for shadow audits")
    p.add_argument("--probe-eval", choices=("none", "final"), default="final")
    p.add_argument("--probe-sample-groups-per-type", type=int, default=24)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--math-sdpa", action="store_true")
    p.add_argument("--d1b-manifest-dir", default=d1d.DEFAULT_D1B)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--verify-only", action="store_true")
    return p


def shadow_audit(impl, arm, cfg, before, shadow, recent_normal, recent_defect, theta, margin):
    """Only historical queried whole-image anchors, no probe evaluation labels."""
    before_native = install_on_clone(arm.bank, before)
    after_native = install_on_clone(arm.bank, shadow)
    def scores(image, features):
        return (native_score(impl, before_native, image, features, cfg)[0],
                native_score(impl, after_native, image, features, cfg)[0])
    result = dict(normal_available=bool(recent_normal),
                  defect_available=bool(recent_defect),
                  normal_utility=None, normal_score_gain=None,
                  defect_score_erosion=None, defect_max_erosion=None,
                  defect_crossing=None)
    if recent_normal:
        hinges, gains = [], []
        for image, features in recent_normal:
            b, a = scores(image, features)
            hinges.append(max(b-theta, 0.)-max(a-theta, 0.))
            gains.append(max(0., b-a))
        result["normal_utility"] = float(sum(hinges)/len(hinges))
        result["normal_score_gain"] = float(sum(gains)/len(gains))
    if recent_defect:
        erosion = []
        crossing = False
        for image, features in recent_defect:
            b, a = scores(image, features)
            erosion.append(max(0., b-a))
            crossing |= bool(b >= theta+margin and a < theta+margin)
        result["defect_score_erosion"] = float(sum(erosion)/len(erosion))
        result["defect_max_erosion"] = float(max(erosion))
        result["defect_crossing"] = bool(crossing)
    return result


def score_probes(impl, backbone, fixed, arms, cfg, theta, recs):
    scores = {"static": []}
    scores.update({str(cap): [] for cap in arms})
    labels = []
    records = []
    for rec in recs:
        image, features = p0.encode_query(impl, backbone, rec.path, cfg)
        labels.append(int(not rec.is_good))
        scores["static"].append(native_score(impl, fixed, image, features, cfg)[0])
        for cap, arm in arms.items():
            scores[str(cap)].append(native_score(impl, arm.bank, image, features, cfg)[0])
        records.append(dict(image_id=rec.rel_path, sample_key=rec.sample_key,
                            camera_id=rec.camera_id, label_offline=int(not rec.is_good),
                            **{f"score_{k}": v[-1] for k, v in scores.items()}))
        del image, features
    baseline = image_metrics(labels, scores["static"], theta)
    metrics = {"static": baseline}
    for cap in arms:
        key = str(cap)
        m = image_metrics(labels, scores[key], theta)
        m.update(probe_deltas(scores[key], labels, scores["static"], m, baseline))
        metrics[key] = m
    return metrics, records


def run(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; activate configured torch environment")
    if args.audit_every < 1 or args.anchors_per_label < 1:
        raise ValueError("Invalid audit cadence or anchor bound")
    if args.max_query_events < 0 or args.max_query_events > args.budget:
        raise ValueError("Invalid max-query-events")
    caps = integers(args.capacities)
    out = native_output_name(args)
    if out.exists():
        raise FileExistsError("Refusing to overwrite an existing R0 output directory: " + str(out))
    impl = p0.require_sources()
    impl.set_seed(0)
    if args.math_sdpa and args.device == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    train, test = pf.load_class_records(Path(args.dataset_root),
                                        Path(args.json_dir), args.class_name)
    plan = pf.build_partitions(train, test, pf.SplitConfig())
    supports_ids = set(plan.support_samples_by_seed[args.support_seed])
    supports = sorted(r.path for r in plan.support if r.sample_key in supports_ids)
    if len(supports) != 4 or len(supports_ids) != 4:
        raise NativeParityError("Strict 4 support identity failed")
    tape, n_stream = d1d.frozen_tape(plan, args)
    requested = tuple(tape.events[:args.max_query_events or len(tape.events)])
    if n_stream != 2625:
        raise NativeParityError("Expected frozen D1-B 2625-image stream")
    metadata = {r.path: r for r in test}
    cfg = p0.Config(device=args.device, augmentation_seed=0, bank_seed=0,
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
    for path in supports:
        fold = p0.build_vmb(impl, refs, [x for x in supports if x != path], cfg)
        loso.append(float(p0.score_image(impl, backbone, fold, path, cfg)))
        del fold
    theta_info = p0.threshold_from_scores(loso, 2.5)
    theta = float(theta_info["theta"])
    sigma = float(theta_info["sigma"])
    del refs
    settings = v02.make_memory_settings(fixed)
    arms, versions = {}, {}
    for cap in caps:
        arm = memory.MemoryArm("greedy_coreset", copy.deepcopy(fixed),
                               settings, seed=args.support_seed + 37)
        for cb in dynamic_codebooks(arm.bank):
            if len(cb.protos) != 0:
                raise RuntimeError("R0 requires an initially empty dynamic Greedy bank")
            cb.K = int(cap)
        arms[cap] = arm
        versions[cap] = 0
    events = []
    counters = {cap: dict(n_normal_events=0, n_candidate_events=0,
        n_candidate_ineligible=0, n_shadow_audits=0,
        n_defect_history_eligible_candidates=0, n_defect_history_ge4_candidates=0,
        n_normal_utility_available=0, n_normal_utility_positive=0,
        n_normal_score_gain_positive=0, n_defect_risk_available=0,
        n_defect_risk_positive=0, n_margin_crossings=0,
        n_full_before_proposal=0, n_replacement_attempts=0,
        n_replacements=0, n_skipped_patches=0, n_appended=0)
        for cap in caps}
    recent_normal, recent_defect = [], []
    total_normal = total_defect = normal_ordinal = 0
    began = time.perf_counter()
    for e in requested:
        rec = metadata.get(e.path)
        if rec is None or e.sample_key != rec.sample_key:
            raise RuntimeError("Q-A tape event identity mismatched")
        image, features = p0.encode_query(impl, backbone, e.path, cfg)
        static_score, amap = native_score(impl, fixed, image, features, cfg)
        # This label is read ONLY AFTER the pre-query static scoring and
        # event selection were performed by the frozen label-blind Q-A tape.
        is_normal = bool(rec.is_good)
        allow = False
        keep = None
        candidates_idx = None
        if is_normal:
            normal_ordinal += 1
            allow, keep, _reason, _extra = impl.select_informative_normal_write(
                score=static_score, theta=theta, sigmaN=max(sigma, .02),
                pred_defect=bool(static_score >= theta), novelty=0.0)
            if allow:
                if keep not in (8, 16, 24):
                    raise RuntimeError("Unrecognized native candidate count")
                _raw, candidates_idx = extract_v22_candidates(impl, amap, features, keep=keep)
        for cap, arm in arms.items():
            c = counters[cap]
            if not is_normal:
                continue
            c["n_normal_events"] += 1
            row = dict(query_step=int(e.position), query_image_id=rec.rel_path,
                       official_sample_key=rec.sample_key, camera_id=rec.camera_id,
                       capacity=cap, static_prequery_score=static_score,
                       n_prior_normal=total_normal, n_prior_defect=total_defect,
                       candidate_allowed=bool(allow),
                       candidate_keep=int(keep) if allow else 0,
                       shadow_audited=False, normal_utility=None,
                       normal_score_gain=None, defect_score_erosion=None,
                       defect_max_erosion=None, defect_crossing=None,
                       memory_full_before=False, replacement_attempt=False,
                       appended=0, replaced=0, skipped=0)
            if not allow:
                c["n_candidate_ineligible"] += 1
                events.append(row)
                continue
            c["n_candidate_events"] += 1
            c["n_defect_history_eligible_candidates"] += int(total_defect >= 1)
            c["n_defect_history_ge4_candidates"] += int(total_defect >= 4)
            before = snapshot(arm.bank, version=versions[cap])
            old_hash = before.sha256()
            rows = normalized_native_candidate_rows(
                impl, arm.bank, features, candidates_idx, keep=keep)
            full = any(len(layer) >= limit for layer, limit in zip(before.layers, before.capacities))
            row["memory_full_before"] = full
            c["n_full_before_proposal"] += int(full)
            proposal, audit = before.proposed(rows, already_normalized=True)
            row["appended"], row["replaced"], row["skipped"] = (
                audit.appended, audit.replaced, audit.skipped)
            row["replacement_attempt"] = full
            c["n_replacement_attempts"] += int(full)
            c["n_appended"] += audit.appended
            c["n_replacements"] += audit.replaced
            c["n_skipped_patches"] += audit.skipped
            if normal_ordinal % args.audit_every == 0:
                diag = shadow_audit(
                    impl, arm, cfg, before, proposal, recent_normal, recent_defect,
                    theta, 0.25 * sigma)
                row.update(diag)
                row["shadow_audited"] = True
                c["n_shadow_audits"] += 1
                c["n_normal_utility_available"] += int(diag["normal_available"])
                c["n_defect_risk_available"] += int(diag["defect_available"])
                c["n_normal_utility_positive"] += int(
                    diag["normal_utility"] is not None and diag["normal_utility"] > 1e-8)
                c["n_normal_score_gain_positive"] += int(
                    diag["normal_score_gain"] is not None and diag["normal_score_gain"] > 1e-8)
                c["n_defect_risk_positive"] += int(
                    diag["defect_score_erosion"] is not None and diag["defect_score_erosion"] > 1e-8)
                c["n_margin_crossings"] += int(diag["defect_crossing"] is True)
            if audit.appended or audit.replaced:
                install_on_live(arm.bank, proposal,
                                expected_digest=old_hash, version=versions[cap])
                versions[cap] += 1
            elif snapshot(arm.bank, version=versions[cap]).sha256() != old_hash:
                raise RuntimeError("Uncommitted shadow modified live bank")
            events.append(row)
        bundle = (image, tuple(f.detach().clone() for f in features))
        if is_normal:
            total_normal += 1
            recent_normal.append(bundle)
            if len(recent_normal) > args.anchors_per_label:
                recent_normal.pop(0)
        else:
            total_defect += 1
            recent_defect.append(bundle)
            if len(recent_defect) > args.anchors_per_label:
                recent_defect.pop(0)
        del image, features
    probe_metrics = None
    probe_rows = []
    if args.probe_eval == "final":
        recs = probe_subset(plan, args.probe_sample_groups_per_type)
        probe_metrics, probe_rows = score_probes(
            impl, backbone, fixed, arms, cfg, theta, recs)
    for cap, arm in arms.items():
        c = counters[cap]
        c["final_prototypes"] = sum(len(cb.protos) for cb in dynamic_codebooks(arm.bank))
        c["final_dynamic_feature_bytes"] = sum(
            cb.protos.numel() * cb.protos.element_size() for cb in dynamic_codebooks(arm.bank))
        c["capacity_reached"] = bool(c["n_full_before_proposal"] or
                                     c["final_prototypes"] >= cap * len(dynamic_codebooks(arm.bank)))
        c["activation"] = {
            "candidate_opportunity": "activated" if c["n_candidate_events"] >= 30 else "underpowered",
            "historical_defect_opportunity": "activated" if c["n_defect_history_eligible_candidates"] >= 20 else "underpowered",
            "counterfactual_risk_audited": "activated" if c["n_defect_risk_available"] >= 20 else "underpowered",
            "nonzero_defect_erosion": "activated" if c["n_defect_risk_positive"] >= 5 else "underpowered",
            "capacity_competition": "activated" if c["n_replacement_attempts"] >= 20 else "underpowered",
            "actual_replacement": "activated" if c["n_replacements"] >= 10 else "underpowered",
            "normal_utility_variation": "activated" if c["n_normal_utility_positive"] >= 5 else "underpowered",
        }
        c["activation_is_descriptive_not_statistical_power"] = True
    summary = dict(schema=SCHEMA, status="COMPLETE" if len(requested)==args.budget else "SMOKE_INCOMPLETE",
        not_full_prequential=True,
        causal_scope="shared external Q-A; only queried observations encoded; Sidecar/Correction/LTM off",
        score_source="native VisionAD v22 top5; frozen static four-shot; per-capacity native Greedy",
        threshold_scope="strict 4-shot LOSO diagnostic (n=4); NOT validated operating point",
        calibration_extra_normal_count=0,
        no_probe_b_access=True, probe_a_offline_only=True,
        config=dict(class_name=args.class_name, support_seed=args.support_seed,
                    query_seed=args.query_seed, budget=args.budget,
                    capacities=list(caps), max_query_events=args.max_query_events,
                    audit_every=args.audit_every, anchors_per_label=args.anchors_per_label,
                    probe_eval=args.probe_eval, probe_sample_groups_per_type=args.probe_sample_groups_per_type),
        n_stream_images=n_stream, n_queried=len(requested),
        n_normal_labels=total_normal, n_defect_labels=total_defect,
        tape_sha256=tape.digest, fixed_bank_sha256=fixed_profile["sha256"],
        theta0=theta, sigma_loso=sigma, loso_scores=loso,
        counters={str(k): v for k, v in counters.items()},
        probe_metrics=probe_metrics, elapsed_seconds=time.perf_counter()-began,
        limitations=["R0 is descriptive opportunity audit, not an R1 Oracle or new safe admission comparison.",
                     "Risk measured only for most recent historical image anchors, not pixel-verified defect regions.",
                     "Risk is shadow-audited every Nth normal candidate, and counters use that audited subset.",
                     "Static-derived candidate ranking and label-blind Q-A tape match D2, no adaptive query policy.",
                     "A full memory evolution under Q-A is executed but nonquery image scores are not computed.",
                     "Real-IAD official sample key is not proof of independent manufactured physical objects.",
                     "Probe A results are development-only; Probe B not read."])
    out.mkdir(parents=True, exist_ok=False)
    (out / "r0_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    def write_csv(path, rows):
        if not rows:
            return
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (out / path).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    write_csv("r0_candidate_events.csv", events)
    write_csv("r0_probe_scores.csv", probe_rows)
    sources = ("experiments/risk_safe/r0_native_activation.py",
               "experiments/risk_safe/r0_opportunity.py",
               "experiments/risk_safe/native_bridge.py",
               "experiments/risk_safe/shadow_memory.py",
               "experiments/risk_safe/d1b_protocol.py",
               "experiments/v02_memory.py",
               "test_phase3_AIF_simplified_v22.py")
    provenance = dict(git=pf.git_state(), source_hashes=pf.source_hashes(sources),
                      realiad_class_json_sha256=pf.sha256_file(
                          Path(args.json_dir)/(args.class_name+".json")),
                      d1b_manifest_sha256=pf.sha256_file(
                          Path(args.d1b_manifest_dir)/d1b.OUTPUT_QUERIES))
    (out / "r0_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in out.iterdir() if p.is_file()}
    (out / "checksums.json").write_text(
        json.dumps(digests, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(out_dir=str(out), status=summary["status"],
                          labels=total_normal+total_defect,
                          normal=total_normal, defect=total_defect,
                          capacities=summary["counters"]), indent=2))
    return 0


def verify(args):
    out = native_output_name(args)
    checksums = json.loads((out/"checksums.json").read_text(encoding="utf-8"))
    for name, value in checksums.items():
        if hashlib.sha256((out/name).read_bytes()).hexdigest() != value:
            raise RuntimeError("R0 result checksum failure: "+name)
    result = json.loads((out/"r0_summary.json").read_text(encoding="utf-8"))
    if result["schema"] != SCHEMA or result["n_queried"] != (
            args.max_query_events or args.budget):
        raise RuntimeError("R0 result schema/count mismatch")
    print(json.dumps(dict(verified=True, output=str(out),
                          status=result["status"], no_gpu_rerun=True), indent=2))
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.json_dir is None:
        args.json_dir = str(Path(args.dataset_root)/"jsons"/"realiad_jsons")
    return verify(args) if args.verify_only else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
