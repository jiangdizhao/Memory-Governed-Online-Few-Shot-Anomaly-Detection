#!/usr/bin/env python3
"""D2-A: pre-register 0.90 / 0.95 fixed thresholds from 64 disjoint train normals.

This script constructs the SAME frozen strict-4-shot v22 static bank/LOSO
reference as D2. Calibration rows are the reserved, one-view-per-official-
sample mature-history TRAIN normals, never online, support, Probe A/B or
confirmed anomalies. Threshold choices are frozen before any D2-A probe runs.

No Sidecar, no dynamic memory updates, no FNR optimisation or probe access.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from . import preflight as pf
from .d2_fixed_threshold import (
    SCHEMA, QUANTILES, REGISTRY, MANIFEST, ThresholdAuditError,
    fit_fixed_normal_thresholds, validate_registry, validate_separation,
)

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import p0_protocol_audit as p0

DEFAULT_OUT = "outputs/risk_safe_sprint_20261027/d2a_audiojack_s0_train64_fixed_thresholds"
SOURCE_FILES = (
    "experiments/risk_safe/d2_fixed_threshold.py",
    "experiments/risk_safe/d2_threshold_calibration.py",
    "experiments/risk_safe/d2_native_replay.py",
    "experiments/risk_safe/preflight.py",
    "experiments/p0_protocol_audit.py",
    "experiments/a0_a1_strict4_matched.py",
    "experiments/v02_memory.py",
    "test_phase3_AIF_simplified_v22.py",
    "vmb_visionad_new.py",
    "tests/test_risk_safe_d2a.py",
    "docs/risk_safe/D2_A_FIXED_THRESHOLD_PROTOCOL.md",
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default=pf.DEFAULT_DATASET_ROOT)
    p.add_argument("--json-dir")
    p.add_argument("--class-name", default="audiojack", choices=("audiojack",))
    p.add_argument("--support-seed", default=0, type=int, choices=(0,))
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--math-sdpa", action="store_true")
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    return p


def metadata(args):
    train, test = pf.load_class_records(
        Path(args.dataset_root), Path(args.json_dir), args.class_name)
    plan = pf.build_partitions(train, test, pf.SplitConfig())
    chosen = sorted(plan.support_samples_by_seed[args.support_seed])
    history = validate_separation(plan, chosen)
    supports = sorted(r.path for r in plan.support if r.sample_key in set(chosen))
    if len(supports) != 4:
        raise ThresholdAuditError("Strict 4-shot support not reproducible")
    return plan, supports, history


def registry_config(args):
    return {
        "class": args.class_name,
        "support_seed": args.support_seed,
        "device": args.device,
        "math_sdpa": bool(args.math_sdpa),
        "dataset_root": str(Path(args.dataset_root)).replace("\\", "/"),
        "json_dir": str(Path(args.json_dir)).replace("\\", "/"),
        "calibration_method": "reserved_mature_history_training_normal",
        "n_calibration_images": 64, "n_calibration_official_samples": 64,
        "views_per_sample": 1,
        "threshold_quantiles": list(QUANTILES),
        "quantile_interpolation": "higher",
        "v22_native_scorer": True,
        "static_memory": "strict4 frozen original VisionAD",
        "sidecar": "off", "signed_correction": "off",
        "probe_label_access": False,
        "dynamic_memory_update": False,
    }


def produce(args, *, plan, supports, history):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ThresholdAuditError("CUDA unavailable")
    impl = p0.require_sources()
    impl.set_seed(0)
    if args.math_sdpa and args.device == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    cfg = p0.Config(
        device=args.device, augmentation_seed=0, bank_seed=0,
        fixed_capacity=25000, fixed_per_image=5000,
        augmentations=4, include_identity=False, multiplier=2.5,
        feature_fuse="concat", topk_patches=32, image_aggregation="max",
        image_topk=64, image_quantile=.95, knn=5)
    backbone = impl.DINOv2MultiLayerBackbone(device=args.device)
    reference = p0.encode_reference_features(impl, backbone, supports, cfg)
    fixed = p0.build_vmb(impl, reference, supports, cfg)
    loso_scores = []
    for held in supports:
        fold = p0.build_vmb(
            impl, reference, [x for x in supports if x != held], cfg)
        loso_scores.append(float(p0.score_image(
            impl, backbone, fold, held, cfg)))
        del fold
    loso = p0.threshold_from_scores(loso_scores, 2.5)
    normal_scores = []
    for rec in history:
        score = float(p0.score_image(impl, backbone, fixed, rec.path, cfg))
        if not math.isfinite(score):
            raise ThresholdAuditError("Nonfinite calibration normal image score")
        normal_scores.append(score)
    fitted = fit_fixed_normal_thresholds(normal_scores)
    raw = np.asarray(normal_scores, dtype=np.float64)
    return {
        "schema": SCHEMA,
        "class_name": args.class_name,
        "support_seed": args.support_seed,
        "normal_calibration_partition": "reserved_mature_history_training_normal",
        "registered_quantiles": fitted,
        "calibration_sample_keys": [r.sample_key for r in history],
        "calibration_paths": [r.rel_path for r in history],
        "calibration_normal_scores": normal_scores,
        "sigma_dev_normal": float(np.std(raw, ddof=1)),
        "score_min": float(raw.min()), "score_median": float(np.median(raw)),
        "score_max": float(raw.max()),
        "loso_reference_theta0": float(loso["theta"]),
        "loso_reference_sigma": float(loso["sigma"]),
        "loso_support_scores": loso_scores,
        "support_keys": sorted(plan.support_samples_by_seed[args.support_seed]),
        "support_paths": supports,
        "realiad_class_json_sha256": pf.sha256_file(
            Path(args.json_dir) / (args.class_name + ".json")),
        "probes_used_for_calibration": False,
        "online_stream_used_for_calibration": False,
        "heldout_defect_labels_used_for_calibration": False,
        "normal_quantile_empirical_exceedance": {
            k: sum(s >= v["theta"] for s in normal_scores) / len(normal_scores)
            for k, v in fitted.items()
        },
        "config": registry_config(args),
        "claim": "normal-only development operating points, not guaranteed FNR-sensitive",
    }


def manifest(args, registry):
    return {
        "schema": "risk_safe.d2a.fixed_threshold_source.v1",
        "git": pf.git_state(),
        "source_sha256": pf.source_hashes(SOURCE_FILES),
        "input_sha256": {"realiad_class_json": pf.sha256_file(
            Path(args.json_dir) / (args.class_name + ".json"))},
        "output_sha256": {REGISTRY: pf.sha256_text(pf.json_text(registry))},
        "configuration": registry_config(args),
        "environment": {**pf.environment_block(), "device": args.device,
                        "cuda_initialised": args.device == "cuda",
                        "features_extracted": True},
    }


def verify(args, plan):
    folder = Path(args.out_dir)
    data = json.loads((folder / REGISTRY).read_text(encoding="utf-8"))
    if data.get("config") != registry_config(args):
        raise ThresholdAuditError("Frozen threshold registry config mismatch")
    validated = validate_registry(
        data, plan=plan, class_name=args.class_name,
        support_seed=args.support_seed,
        class_json_path=Path(args.json_dir) / (args.class_name + ".json"),
        registry_path=folder / REGISTRY)
    expected = pf.preserve_committed_source_provenance(
        folder / MANIFEST, manifest(args, data))
    pf.verify_json_output(folder / MANIFEST, expected)
    print(json.dumps({"D2_A": "fixed_threshold_registry_verified",
                      "normal_history_views": validated["n_training_normal_views"],
                      "thresholds": {k: v["theta"]
                          for k, v in validated["theta_by_quantile"].items()},
                      "LOSO_reference": data["loso_reference_theta0"],
                      "gpu_recomputed": False}, indent=2))
    return 0


def main(argv: Sequence[str] | None = None):
    args = parser().parse_args(argv)
    if args.json_dir is None:
        args.json_dir = str(Path(args.dataset_root) / pf.DEFAULT_JSON_SUBDIR)
    plan, supports, history = metadata(args)
    if args.preflight_only:
        print(json.dumps({
            "D2_A": "normal_only_metadata_preflight",
            "strict4_support": len(supports),
            "disjoint_training_normal_views": len(history),
            "registered_quantiles": QUANTILES,
            "heldout_probe_calibration": False,
            "gpu_used": False}, indent=2))
        return 0
    if args.verify_only:
        return verify(args, plan)
    target = Path(args.out_dir)
    if target.exists():
        raise FileExistsError(
            f"Threshold output already exists; use --verify-only: {target}")
    with torch.no_grad():
        registry = produce(args, plan=plan, supports=supports, history=history)
    # Validate independently before producing any official evidence.
    fitted = fit_fixed_normal_thresholds(registry["calibration_normal_scores"])
    if fitted != registry["registered_quantiles"]:
        raise ThresholdAuditError("Threshold recomputation mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            dir=target.parent, prefix=".d2a_cal_build_") as temp:
        staged = Path(temp) / target.name
        staged.mkdir()
        pf.write_json_idempotent(staged / REGISTRY, registry)
        pf.write_json_idempotent(staged / MANIFEST, manifest(args, registry))
        validate_registry(
            registry, plan=plan, class_name=args.class_name,
            support_seed=args.support_seed,
            class_json_path=Path(args.json_dir) / (args.class_name + ".json"),
            registry_path=staged / REGISTRY)
        if target.exists():
            raise FileExistsError("Threshold output appeared during publishing")
        staged.rename(target)
    print(json.dumps({
        "D2_A": "registered_fixed_normal_only_thresholds",
        "trained_on_N_normal_images": len(history),
        "theta_by_quantile": {k: v["theta"] for k, v in fitted.items()},
        "LOSO_reference": registry["loso_reference_theta0"],
        "evaluation_probe_used": False, "manifest": "saved"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
