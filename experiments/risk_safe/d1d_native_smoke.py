#!/usr/bin/env python3
"""D1-D native v22 parity + bounded Q-A queried-event shadow audit.

Real frozen DINOv2/VisionAD scores and genuine V0.2 Greedy update are used.
No CDARG commits; no FPR/FNR/held-out probe analysis. Query-only replay
skips unqueried inference, valid for this exogenous Q-A/no-update diagnostic
only; it is NOT a full prequential stream performance experiment.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Sequence
import torch

from . import preflight as pf
from . import d1b_protocol as d1b
from . import query_protocol as qp
from .admission import CDARG, CDARGConfig
from .anchor_store import VerifiedImage
from .native_bridge import (NativeScoreAdapter, NativeParityError,
    extract_v22_candidates, native_score, snapshot, verify_untouched_parity,
    verify_native_update_parity)

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "experiments"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import p0_protocol_audit as p0
import v02_architecture_core as v02
import v02_memory as memory

SCHEMA = "risk_safe.d1d.native_smoke.v1"
RESULT = "native_smoke.json"
SOURCE = "source_manifest.json"
DEFAULT_OUT = "outputs/risk_safe_sprint_20261027/d1d_audiojack_s0_q37_b16"
DEFAULT_D1B = "outputs/risk_safe_sprint_20261027/d1b_protocol_v1"
SOURCES = (
    "experiments/risk_safe/d1d_native_smoke.py",
    "experiments/risk_safe/native_bridge.py",
    "experiments/risk_safe/admission.py",
    "experiments/risk_safe/anchor_store.py",
    "experiments/risk_safe/shadow_memory.py",
    "experiments/risk_safe/preflight.py",
    "experiments/risk_safe/d1b_protocol.py",
    "experiments/risk_safe/query_protocol.py",
    "experiments/v02_memory.py",
    "experiments/v02_architecture_core.py",
    "experiments/p0_protocol_audit.py",
    "experiments/a0_a1_strict4_matched.py",
    "test_phase3_AIF_simplified_v22.py",
    "vmb_visionad_new.py",
    "tests/test_risk_safe_d1d.py",
    "docs/risk_safe/D1_D_NATIVE_SMOKE_PROTOCOL.md",
)


@dataclass(frozen=True)
class NativeAnchor:
    event: VerifiedImage
    features: Any     # trusted (PIL image, full native feature tuple)
    weight: float = 1.0


class NativeLedger:
    """Only previously queried individual camera labels; full-image risk audit."""
    def __init__(self):
        self.normal = []
        self.defect = []
        self.query_ids = set()

    def prior(self, step):
        return (tuple(x for x in self.normal if x.event.label_revealed_step < step),
                tuple(x for x in self.defect if x.event.label_revealed_step < step))

    def add(self, event, image, features):
        if event.query_id in self.query_ids:
            raise NativeParityError("Duplicate queried image feedback")
        self.query_ids.add(event.query_id)
        anchor = NativeAnchor(event, (image, tuple(f.detach().clone()
                                                  for f in features)))
        (self.normal if event.label == "normal" else self.defect).append(anchor)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default=pf.DEFAULT_DATASET_ROOT)
    p.add_argument("--json-dir")
    p.add_argument("--class-name", default="audiojack",
                   choices=d1b.INITIAL_SMOKE_CLASSES)
    p.add_argument("--support-seed", type=int, default=0,
                   choices=d1b.INITIAL_SMOKE_SUPPORT_SEEDS)
    p.add_argument("--query-seed", type=int, default=37,
                   choices=d1b.INITIAL_SMOKE_QUERY_SEEDS)
    p.add_argument("--budget", type=int, default=16,
                   choices=d1b.MAIN_BUDGETS)
    p.add_argument("--max-query-events", type=int, default=4)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--math-sdpa", action="store_true")
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--d1b-manifest-dir", default=DEFAULT_D1B)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--verify-only", action="store_true")
    return p


def config(args):
    return {
        "class_name": args.class_name,
        "support_seed": args.support_seed,
        "query_seed": args.query_seed,
        "budget": args.budget,
        "max_query_events": args.max_query_events,
        "device": args.device, "math_sdpa": args.math_sdpa,
        "atol": args.atol, "rtol": args.rtol,
        "dataset_root": str(args.dataset_root).replace("\\", "/"),
        "json_dir": str(args.json_dir).replace("\\", "/"),
        "d1b_manifest_dir": str(args.d1b_manifest_dir).replace("\\", "/"),
        "gate": "UNFROZEN synthetic demo coefficients; audit only, NEVER commit",
        "stream": "Q-A frozen random queried-event condensed replay",
        "theta": "strict4 LOSO frozen",
        "sidecar": "off", "signed_correction": "off",
    }


def provenance(args, result):
    return {
        "schema": "risk_safe.d1d.native_source.v1",
        "git": pf.git_state(),
        "source_sha256": pf.source_hashes(SOURCES),
        "input_sha256": {
            "realiad_class_json": pf.sha256_file(
                Path(args.json_dir) / (args.class_name + ".json")),
            "d1b_query_manifest": pf.sha256_file(
                Path(args.d1b_manifest_dir) / d1b.OUTPUT_QUERIES)},
        "configuration": config(args),
        "result_sha256": pf.sha256_text(pf.json_text(result)),
        "environment": {**pf.environment_block(),
                        "device": args.device,
                        "features_extracted": True,
                        "cuda_initialised": args.device == "cuda"},
    }


def read_only_verify(args):
    out = Path(args.out_dir)
    result = json.loads((out / RESULT).read_text(encoding="utf-8"))
    if result.get("configuration") != config(args) or not result.get("passed"):
        raise NativeParityError("Invalid/mismatched frozen D1-D result")
    expected = pf.preserve_committed_source_provenance(
        out / SOURCE, provenance(args, result))
    pf.verify_json_output(out / SOURCE, expected)
    print(json.dumps({"stage": "D1-D", "verify_only": True,
                      RESULT: "sha256_verified", SOURCE: "verified",
                      "gpu_work_repeated": False}, indent=2))
    return 0


def frozen_tape(plan, args):
    stream = d1b.build_ordered_stream(plan.phases, "random", args.class_name)
    universe = tuple(sorted(set(x.sample_key for x in stream)))
    priority = d1b.shared_query_priority(universe, args.class_name,
                                          args.query_seed)
    tape = d1b.build_paired_tapes(stream, (args.budget,),
                                  args.query_seed, args.class_name,
                                  priority)[args.budget]
    p = Path(args.d1b_manifest_dir) / d1b.OUTPUT_QUERIES
    expected = json.loads(p.read_text(encoding="utf-8"))[
        "class_records"][args.class_name]["random"][str(args.query_seed)][
            str(args.budget)]
    if (tape.digest != expected["digest"] or
        tape.stream_digest != expected["stream_digest"] or
        list(tape.positions) != expected["query_positions"] or
        qp.sha256_of([e.as_manifest() for e in tape.events]) !=
            expected["events_sha256"]):
        raise NativeParityError("Query tape mismatch with frozen D1-B")
    return tape, len(stream)


@torch.no_grad()
def execute(args):
    if args.max_query_events < 1 or args.max_query_events > args.budget:
        raise NativeParityError("Invalid number of query events")
    if args.atol <= 0 or args.rtol < 0:
        raise NativeParityError("Invalid scorer parity tolerance")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise NativeParityError("CUDA unavailable; run with configured GPU environment")
    impl = p0.require_sources()
    impl.set_seed(0)
    if args.math_sdpa and args.device == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    train, test = pf.load_class_records(Path(args.dataset_root),
                                        Path(args.json_dir), args.class_name)
    plan = pf.build_partitions(train, test, pf.SplitConfig())
    sample_ids = set(plan.support_samples_by_seed[args.support_seed])
    selected = [r for r in plan.support if r.sample_key in sample_ids]
    supports = sorted(r.path for r in selected)
    if len(supports) != 4 or len({r.sample_key for r in selected}) != 4:
        raise NativeParityError("Support must be 4 distinct verified normal images")
    tape, N_stream_images = frozen_tape(plan, args)
    test_lookup = {r.path: r for r in test}

    cfg = p0.Config(
        device=args.device, augmentation_seed=0, bank_seed=0,
        fixed_capacity=25000, fixed_per_image=5000, augmentations=4,
        include_identity=False, multiplier=2.5, feature_fuse="concat",
        topk_patches=32, image_aggregation="max", image_topk=64,
        image_quantile=.95, knn=5)
    backbone = impl.DINOv2MultiLayerBackbone(device=args.device)
    refs = p0.encode_reference_features(impl, backbone, supports, cfg)
    fixed = p0.build_vmb(impl, refs, supports, cfg)
    support_fingerprint = p0.fixed_bank_profile(fixed)
    loso = []
    for held in supports:
        remaining = [p for p in supports if p != held]
        fold = p0.build_vmb(impl, refs, remaining, cfg)
        loso.append(float(p0.score_image(impl, backbone, fold, held, cfg)))
        del fold
    theta_info = p0.threshold_from_scores(loso, 2.5)
    theta, sigma = theta_info["theta"], theta_info["sigma"]
    del refs
    settings = v02.make_memory_settings(fixed)

    # Diagnostic-only parity arm. Never feed its update into replay.
    first = tape.events[0]
    first_img, first_features = p0.encode_query(
        impl, backbone, first.path, cfg)
    s_fixed, first_map = native_score(
        impl, fixed, first_img, first_features, cfg)
    parity_arm = memory.MemoryArm(
        "greedy_coreset", copy.deepcopy(fixed), settings,
        seed=args.support_seed + 37)
    read_parity = verify_untouched_parity(
        impl, parity_arm.bank, first_img, first_features, cfg,
        atol=args.atol, rtol=args.rtol)
    write_parity = verify_native_update_parity(
        impl, parity_arm, first_features, first_map,
        score=s_fixed, theta=theta, sigma=sigma, step=first.position)
    del parity_arm, first_img, first_features

    # Online audit-only arm; admission proposals NEVER reach production state.
    arm = memory.MemoryArm(
        "greedy_coreset", copy.deepcopy(fixed), settings,
        seed=args.support_seed + 37)
    before = snapshot(arm.bank).sha256()
    adapter = NativeScoreAdapter(impl, arm.bank, cfg)
    ledger = NativeLedger()
    # Demonstration ONLY; Real-IAD CDARG parameters NOT calibrated.
    gate = CDARG(CDARGConfig(
        epsilon_defect=.02, lambda_defect=1, tau_utility=0,
        defect_margin=.05, sigma_normal=1, sigma_floor=.01), adapter)
    events = []
    for item in tape.events[:args.max_query_events]:
        rec = test_lookup.get(item.path)
        if rec is None or rec.rel_path != item.rel_path:
            raise NativeParityError("Query path not in allowed test stream")
        img, features = p0.encode_query(impl, backbone, item.path, cfg)
        fixed_score, amap = native_score(impl, fixed, img, features, cfg)
        dynamic_score, _ = native_score(impl, arm.bank, img, features, cfg)
        # Label is first read here, after pre-query prediction/scoring.
        verified = VerifiedImage(
            query_id=f"qa:{args.class_name}:{args.query_seed}:{item.position}",
            image_id=item.rel_path, sample_key=item.sample_key,
            camera_id=item.camera_id, label=(
                "normal" if rec.is_good else "defect"),
            query_decision_step=item.position,
            label_revealed_step=item.position)
        record = {
            "step": item.position, "query_image": item.rel_path,
            "camera": item.camera_id, "revealed_label": verified.label,
            "prequery_fixed_score": fixed_score,
            "prequery_dynamic_score": dynamic_score,
            "prior_normal_anchors": len(ledger.normal),
            "prior_defect_anchors": len(ledger.defect),
        }
        if verified.label == "normal":
            allow, keep, _, _ = impl.select_informative_normal_write(
                score=fixed_score, theta=theta, sigmaN=max(sigma, .02),
                pred_defect=bool(fixed_score >= theta), novelty=0.0)
            if not allow:
                raise NativeParityError("Unexpected native candidate selection failure")
            patches, _ = extract_v22_candidates(
                impl, amap, features, keep=keep)
            proposal = gate.evaluate(
                current_verified_normal=verified, step=item.position,
                bank=snapshot(arm.bank), ledger=ledger,
                candidate_patches=patches, theta=theta)
            record["native_patch_keep"] = keep
            record["gate_counterfactual_audit_only"] = proposal.audit()
        else:
            record["gate_counterfactual_audit_only"] = None
        ledger.add(verified, img, features)
        events.append(record)
    if snapshot(arm.bank).sha256() != before:
        raise NativeParityError("Audit-only mode unexpectedly committed memory")
    if p0.fixed_bank_profile(arm.bank)["sha256"] != support_fingerprint["sha256"]:
        raise NativeParityError("Fixed reference memory mutated")

    return {
        "schema": SCHEMA, "stage": "D1-D",
        "configuration": config(args),
        "passed": bool(read_parity["passed"] and write_parity["passed"]),
        "native_score_parity": read_parity,
        "native_update_parity": write_parity,
        "support_profile": support_fingerprint,
        "loso_scores": loso, "theta0": theta, "sigma0": sigma,
        "frozen_tape_sha256": tape.digest,
        "frozen_stream_sha256": tape.stream_digest,
        "stream_images": N_stream_images,
        "queries_audited": len(events), "events": events,
        "production_dynamic_bank_sha256_unchanged": before,
        "native_v22_image_scorer_tested": True,
        "real_feature_greedy_parity_tested": True,
        "native_shadow_cdarg_audit_only": True,
        "admission_commits": 0,
        "heldout_probes_read": False, "FPR_FNR_AUROC_measured": False,
        "entire_stream_processed": False,
        "sidecar_and_signed_correction": "off",
        "CDARG_parameters_calibrated": False,
    }


def main(argv: Sequence[str] | None = None):
    args = parser().parse_args(argv)
    if args.json_dir is None:
        args.json_dir = str(Path(args.dataset_root) / pf.DEFAULT_JSON_SUBDIR)
    if args.verify_only:
        return read_only_verify(args)
    output = Path(args.out_dir)
    if (output / RESULT).exists() or (output / SOURCE).exists():
        raise FileExistsError("Output exists. Use --verify-only or a new folder.")
    result = execute(args)
    if not result["passed"]:
        raise NativeParityError("Native parity failed")
    a = pf.write_json_idempotent(output / RESULT, result)
    b = pf.write_json_idempotent(output / SOURCE, provenance(args, result))
    print(json.dumps({"stage": "D1-D", RESULT: a, SOURCE: b,
                      "passed": result["passed"],
                      "query_events_audited": result["queries_audited"],
                      "admission_commits": 0,
                      "FPR_FNR_AUROC_measured": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
