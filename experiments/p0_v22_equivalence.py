#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E01 — trace ORIGINAL v22 initialization and contrast it with the P0-R2 pilot.

This is a DIAGNOSTIC, not a replacement for v22 and not a proof of equivalence.
v22 runs through its own unmodified run_phase3_streaming_aif(max_samples=0).
The P0 bank is then built separately with exactly the SAME four support paths.
Only frozen probes are scored; no stream labels, feedback, or online writes.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any
from unittest.mock import patch

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def memory_profile(bank: Any, name: str, mode: int = 0, view: str = "id") -> dict:
    """Bytewise profile of actual installed per-layer fixed prototypes."""
    if name not in bank.normal_fixed:
        raise AssertionError("Class not initialized in fixed bank")
    refs = bank.normal_fixed[name][mode][view]
    sha = hashlib.sha256()
    rows = []
    for i, cb in enumerate(refs):
        values = cb.protos.detach().cpu().contiguous().numpy()
        sha.update(values.tobytes())
        rows.append({
            "layer": i, "n_prototypes": int(values.shape[0]),
            "feature_dim": int(values.shape[1]), "dtype": str(values.dtype),
            "capacity": int(cb.K), "bytes": int(values.nbytes)
        })
    return {"layers": rows, "sha256": sha.hexdigest(),
            "total_prototypes": sum(x["n_prototypes"] for x in rows),
            "total_feature_bytes": sum(x["bytes"] for x in rows)}


def image_fingerprint(image: Any) -> str:
    h = hashlib.sha256()
    h.update(str(image.size).encode("utf-8"))
    h.update(str(image.mode).encode("utf-8"))
    h.update(image.tobytes())
    return h.hexdigest()


def summarize_aug_calls(calls: list[dict], n_support: int) -> dict:
    """Legacy v22 does concept-seeding pass then fixed-bank pass."""
    if len(calls) < 2 * n_support:
        raise AssertionError(
            f"Expected >= {2*n_support} original v22 make_support_refs calls, got {len(calls)}"
        )
    return {
        "concept_pass": calls[:n_support],
        "fixed_pass": calls[n_support:2 * n_support],
        "total_calls": len(calls),
        "notes": "Order corresponds to support iteration for single-mode, identity-only v22",
    }


def paired_score_statistics(xs: list[float], ys: list[float], tol: float) -> dict:
    if len(xs) != len(ys) or not xs:
        raise ValueError("Expected equally many nonempty paired frozen scores")
    a = np.asarray(xs, dtype=np.float64)
    b = np.asarray(ys, dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite frozen anomaly score")
    d = np.abs(a - b)
    return {
        "n": len(a), "mean_abs_delta": float(d.mean()),
        "max_abs_delta": float(d.max()),
        "n_above_tolerance": int((d > tol).sum()),
        "tolerance": float(tol),
        "within_tolerance": bool(np.all(d <= tol)),
        "score_correlation": float(np.corrcoef(a, b)[0, 1]) if
        len(a) >= 3 and float(a.std()) > 0 and float(b.std()) > 0 else None,
    }


def verdict(supports_match: bool, canonical: dict, pilot: dict,
            pair: dict, theta_delta: float, tolerance: float,
            augmentation_match: bool) -> dict:
    same_shapes = canonical["layers"] == pilot["layers"]
    same_protos = canonical["sha256"] == pilot["sha256"]
    everything = (
        supports_match and same_shapes and same_protos and
        pair["within_tolerance"] and abs(theta_delta) <= tolerance and
        augmentation_match
    )
    failed = [
        name for name, ok in {
            "native_support_selection": supports_match,
            "augmentation_fingerprints": augmentation_match,
            "prototype_profile": same_shapes,
            "prototype_bytes": same_protos,
            "frozen_probe_scores": pair["within_tolerance"],
            "initial_threshold": abs(theta_delta) <= tolerance,
        }.items() if not ok
    ]
    return {
        "status": "equivalent_on_audited_conditions" if everything else "not_equivalent",
        "failed_checks": failed, "native_support_selection_equal": supports_match,
        "matched_supports_for_bank_comparison": True,
        "support_selection_difference_separately_recorded": True,
        "prototype_shapes_equal": same_shapes,
        "prototype_bytes_equal": same_protos,
        "augmentation_fingerprints_equal": augmentation_match,
        "probe_scores_within_tolerance": pair["within_tolerance"],
        "threshold_within_tolerance": abs(theta_delta) <= tolerance,
        "explanation": (
            "Actual v22 initializer is ground truth. A NOT_EQUIVALENT result "
            "is an informative E01 finding, NOT a GPU run failure. Do not use "
            "the P0 score as a faithful copy of historical v22."
        )
    }


def load_modules():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(root / "experiments") not in sys.path:
        sys.path.insert(0, str(root / "experiments"))
    import test_phase3_AIF_simplified_v22 as impl
    import p0_protocol_audit as p0
    return impl, p0


def run_canonical(args, impl, directory: Path) -> tuple:
    """Execute the actual v22 initialization and capture its original bank."""
    artifacts = {"backbone": None, "bank": None, "augmentations": []}
    original_ctor = impl.DINOv2MultiLayerBackbone
    original_build = impl.PatchVMB.build_fixed_normal_from_support
    original_aug = impl.make_support_refs

    def capture_backbone(*a, **kw):
        obj = original_ctor(*a, **kw)
        artifacts["backbone"] = obj
        return obj

    def capture_fixed(self, *a, **kw):
        result = original_build(self, *a, **kw)
        artifacts["bank"] = self
        return result

    def capture_refs(*a, **kw):
        refs = original_aug(*a, **kw)
        artifacts["augmentations"].append({
            "count": len(refs),
            "fingerprints": [image_fingerprint(img) for img in refs],
        })
        return refs

    init_json = directory / "canonical_v22_init.json"
    # Run the v22 implementation ITSELF. No image from the test stream
    # is decoded or used before this returns (max_samples=0).
    with ExitStack() as stack:
        stack.enter_context(patch.object(impl, "DINOv2MultiLayerBackbone", capture_backbone))
        stack.enter_context(patch.object(impl.PatchVMB, "build_fixed_normal_from_support", capture_fixed))
        stack.enter_context(patch.object(impl, "make_support_refs", capture_refs))
        result = impl.run_phase3_streaming_aif(
            dataset="realiad", root=args.dataset_root, class_name=args.class_name,
            csv=args.json_dir, device=args.device,
            out_json=str(directory / "canonical_v22_run.json"),
            max_samples=0, init_dump_json=str(init_json),
            seed=args.legacy_seed, support_seed=args.support_seed,
            stream_seed=args.stream_seed, no_shuffle_test=True,
            ablation="static_baseline", policy="simple_aif",
            use_modes=False, K_modes=1, shots_per_mode=4,
            support_select="legacy",
            use_global_threshold_initialization=False,
            feature_fuse=args.feature_fuse,
            baseline_compat=False, support_include_identity=args.include_identity,
            support_aug_cfg=impl.VisionADSupportAugConfig(
                rotation_deg=args.rotation_deg, rotation_p=args.rotation_p,
                translate=args.translate, hflip_p=args.hflip_p
            ),
            aug_per_image=args.augmentations,
            views=[impl.ViewSpec(name="id", kind="id")], use_rotations=False,
            K_normal_fixed=args.fixed_capacity, K_normal_dyn=1024, K_normal_ltm=512,
            K_defect=256, K_concept=4096, knn_k_normal=args.knn,
            corr_cfg=impl.CorrectionConfig(enabled=False),
            sm_cfg=impl.SetMembershipConfig(enabled=False),
            use_defect_bank=False, disable_defect_bank=True,
            fixed_per_img_cap=args.fixed_per_image,
            concept_per_img_cap=256, concept_seed_patches=2048,
            bank_dtype="fp16", topk_patches=args.topk_patches,
            score_mode="boost", lam=0.8, tau_close=0.2, gamma=1.0,
            layer_fusion="mean", view_fusion="mean",
            img_agg=args.image_aggregation, img_topk=args.image_topk,
            img_quantile=args.image_quantile,
            theta_controller_kind="guarded_dual_anchor",
            thcfg=impl.ThresholdConfig(),
        )
    if not init_json.is_file() or not artifacts["bank"] or not artifacts["backbone"]:
        raise RuntimeError("Canonical v22 initialization did not create required audit artifacts")
    init = json.loads(init_json.read_text(encoding="utf-8"))
    modes = init["support_paths_per_mode"]
    if list(modes.keys()) != ["0"] or len(modes["0"]) != 4:
        raise RuntimeError(f"E01 only supports one mode and exactly four images; got {modes}")
    return init, artifacts, result


def get_probes(impl, args) -> list[str]:
    idx = impl.RealIADJSONIndex(root=args.dataset_root, json_dir=args.json_dir)
    samples = sorted(
        [x for x in idx.test_samples if x.cls_name == args.class_name],
        key=lambda x: x.path
    )
    if not samples:
        raise RuntimeError("No held-out test images")
    random.Random(args.stream_seed).shuffle(samples)
    return [s.path for s in samples[:args.probe_count]]


def score_shared_query(impl, bank, image, feats, cfg, cls_name: str) -> float:
    """Use the SAME query encoding for both fixed memories."""
    result, _, _ = bank.score_sample_multiview(
        cls_name=cls_name, view_to_feats={"id": feats}, mode=0,
        views={"id": impl.ViewSpec("id", "id")},
        topk_patches=cfg.topk_patches, mode_scoring="boost",
        lam=0.8, tau_close=0.2, gamma=1.0,
        layer_fusion="mean", view_fusion="mean",
        img_agg=cfg.image_aggregation, img_topk=cfg.image_topk,
        img_quantile=cfg.image_quantile, concept_assign_view="id",
        img_hw=(image.height, image.width)
    )
    return float(result)


def audit(args) -> dict:
    import torch
    impl, p0 = load_modules()
    if args.math_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("[E01] Running ORIGINAL v22 initialization (no test-stream processing).", flush=True)
    init, captured, _ = run_canonical(args, impl, out)
    canonical_supports = list(init["support_paths_per_mode"]["0"])
    baseline_model = captured["backbone"]
    canonical_bank = captured["bank"]
    canon_profile = memory_profile(canonical_bank, args.class_name)

    train_idx = impl.RealIADJSONIndex(root=args.dataset_root, json_dir=args.json_dir)
    normals = sorted({s.path for s in train_idx.train_samples
                      if s.cls_name == args.class_name and s.is_good})
    native_p0_supports, _ = p0.select_support(
        impl, baseline_model, normals, "strict4", args.support_seed, 4
    )
    native_match = (native_p0_supports == canonical_supports)
    print(f"[E01] Original v22 supports: {len(canonical_supports)}."
          f" Native P0 selection equal? {native_match}", flush=True)

    # Build legacy P0 with v22's EXACT support image identities, avoiding
    # false attribution of score differences to support selection.
    cfg = p0.Config(
        device=args.device, augmentation_seed=args.augmentation_seed,
        bank_seed=args.bank_seed, fixed_capacity=args.fixed_capacity,
        fixed_per_image=args.fixed_per_image,
        augmentations=args.augmentations, include_identity=args.include_identity,
        multiplier=args.multiplier, feature_fuse=args.feature_fuse,
        topk_patches=args.topk_patches,
        image_aggregation=args.image_aggregation, image_topk=args.image_topk,
        image_quantile=args.image_quantile, knn=args.knn
    )
    aug_p0 = []
    original_refs = impl.make_support_refs
    def capture_p0_refs(*a, **kw):
        refs = original_refs(*a, **kw)
        aug_p0.append({"count": len(refs),
                       "fingerprints": [image_fingerprint(img) for img in refs]})
        return refs
    with patch.object(impl, "make_support_refs", capture_p0_refs):
        cached = p0.encode_reference_features(impl, baseline_model, canonical_supports, cfg)
    pilot_bank = p0.build_vmb(impl, cached, canonical_supports, cfg)
    pilot_profile = memory_profile(pilot_bank, "p0")

    original_augmentation = summarize_aug_calls(
        captured["augmentations"], len(canonical_supports)
    )
    canonical_fixed_refs = original_augmentation["fixed_pass"]
    augmentation_match = (
        [item["fingerprints"] for item in canonical_fixed_refs] ==
        [item["fingerprints"] for item in aug_p0]
    )

    # No class labels loaded here: scoring held-out images is a paired,
    # frozen, non-training audit. Probe images are *never* written to memory.
    probe_paths = get_probes(impl, args)
    scores_canonical, scores_p0, probe_rows = [], [], []
    for step, path in enumerate(probe_paths):
        image, feats = p0.encode_query(impl, baseline_model, path, cfg)
        value_c = score_shared_query(impl, canonical_bank, image, feats,
                                     cfg, args.class_name)
        value_p = score_shared_query(impl, pilot_bank, image, feats, cfg, "p0")
        scores_canonical.append(value_c)
        scores_p0.append(value_p)
        probe_rows.append({
            "probe_index": step, "path": path,
            "canonical_v22_score": value_c, "p0_score": value_p,
            "p0_minus_v22": value_p - value_c,
            "abs_diff": abs(value_p - value_c)
        })
        if (step + 1) % 5 == 0:
            print(f"  [E01] paired frozen probes {step+1}/{len(probe_paths)}", flush=True)
    paired = paired_score_statistics(scores_canonical, scores_p0, args.score_tolerance)
    p0_support_scores = [
        p0.score_image(impl, baseline_model, pilot_bank, path, cfg)
        for path in canonical_supports
    ]
    pilot_th = p0.threshold_from_scores(p0_support_scores, args.multiplier)
    theta_delta = float(pilot_th["theta"] - float(init["theta0"]))
    outcome = verdict(native_match, canon_profile, pilot_profile,
                      paired, theta_delta, args.score_tolerance, augmentation_match)
    result = {
        "schema": "e01.v22.initialization.audit.v1",
        "protocol": {
            "dataset": "realiad", "class": args.class_name,
            "canonical_run": "original v22 run_phase3_streaming_aif(max_samples=0)",
            "canonical_use_modes": False,
            "canonical_support_select": "legacy",
            "canonical_num_test_stream_updates": 0,
            "candidate_normal_images_indexed": len(normals),
            "strict_four_shot_eligible": True,
            "p0_supports_forced_to_canonical": True,
            "notes": [
                "Canonical v22 uses its native _sample_paths_deterministic_or_seeded selector.",
                "P0 native uses random.sample; independent selection compared separately.",
                "P0 uses isolated augmentation/bank seeds, unlike canonical v22 sequential RNG.",
                "The P0 construction path is a pilot, not asserted to reproduce v22.",
                "Only frozen probes are scored using shared encoded features; labels are never loaded.",
                "This run is single-mode, identity-view, no correction, no set membership, no normal-pool calibration.",
                "Equivalent in this restricted scenario does not establish equivalence to historical 30-class submitted runs."
            ]
        },
        "parameters": vars(args),
        "supports": {
            "canonical_v22": canonical_supports,
            "p0_native_selection": native_p0_supports,
            "native_order_identical": native_match,
            "p0_bank_supports_used": canonical_supports
        },
        "backbone": {
            "model_name": getattr(baseline_model, "model_name", None),
            "img_size": getattr(baseline_model, "img_size", None),
            "resize_size": getattr(baseline_model, "resize_size", None),
            "layers": list(getattr(baseline_model, "layers", [])),
            "use_fp16": getattr(baseline_model, "use_fp16", None),
        },
        "augmentation": {
            "original_v22": original_augmentation,
            "p0_refs": aug_p0,
            "fixed_pass_fingerprints_equal": augmentation_match
        },
        "memory": {"canonical_v22": canon_profile, "p0": pilot_profile},
        "threshold": {
            "canonical_init_source": init["threshold_initialization"].get("source"),
            "canonical_v22_support_scores": init["support_scores"],
            "canonical_v22_theta0": init["theta0"],
            "canonical_v22_muN0": init["muN0"],
            "canonical_v22_sigmaN0": init["sigmaN0"],
            "canonical_v22_controller": init["theta_controller_kind"],
            "p0_support_scores": p0_support_scores,
            "p0_mu": pilot_th["mu"], "p0_sigma": pilot_th["sigma"],
            "p0_theta": pilot_th["theta"], "p0_minus_canonical_theta": theta_delta
        },
        "paired_probe": paired, "verdict": outcome,
        "source_sha256": {str(path): sha256_file(Path(__file__).resolve().parents[1] / path)
                          for path in ("test_phase3_AIF_simplified_v22.py",
                                       "vmb_visionad_new.py", "experiments/p0_protocol_audit.py",
                                       "experiments/p0_v22_equivalence.py")},
        "runtime": {
            "elapsed_seconds": round(time.time() - started, 2),
            "torch": torch.__version__, "numpy": np.__version__,
            "math_sdpa_only": args.math_sdpa
        },
    }
    with (out / "e01_paired_frozen_probe_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(probe_rows[0]))
        writer.writeheader()
        writer.writerows(probe_rows)
    (out / "e01_equivalence_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[E01] Verdict:", outcome["status"], flush=True)
    print("[E01] Failed checks:", outcome["failed_checks"], flush=True)
    print("[E01] Paired probe max absolute delta:", paired["max_abs_delta"], flush=True)
    print("[E01] Report:", out / "e01_equivalence_report.json", flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root")
    p.add_argument("--json-dir")
    p.add_argument("--class-name", default="audiojack")
    p.add_argument("--legacy-seed", type=int, default=0)
    p.add_argument("--support-seed", type=int, default=0)
    p.add_argument("--stream-seed", type=int, default=0)
    p.add_argument("--augmentation-seed", type=int, default=0)
    p.add_argument("--bank-seed", type=int, default=0)
    p.add_argument("--probe-count", type=int, default=12)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fixed-capacity", type=int, default=25000)
    p.add_argument("--fixed-per-image", type=int, default=5000)
    p.add_argument("--augmentations", type=int, default=4)
    p.add_argument("--include-identity", action="store_true")
    p.add_argument("--rotation-deg", type=float, default=15.)
    p.add_argument("--rotation-p", type=float, default=.7)
    p.add_argument("--translate", type=float, default=.08)
    p.add_argument("--hflip-p", type=float, default=.5)
    p.add_argument("--feature-fuse", choices=["concat","sum","none"], default="concat")
    p.add_argument("--image-aggregation", choices=["max","topk_mean","spiky_quantile"],
                   default="max")
    p.add_argument("--image-topk", type=int, default=64)
    p.add_argument("--img-quantile", dest="image_quantile", type=float, default=.95)
    p.add_argument("--topk-patches", type=int, default=32)
    p.add_argument("--knn", type=int, default=5)
    p.add_argument("--multiplier", type=float, default=2.5)
    p.add_argument("--score-tolerance", type=float, default=1e-4)
    p.add_argument("--math-sdpa", action="store_true")
    p.add_argument("--out-dir", default="outputs/e01_audiojack")
    p.add_argument("--self-test", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if args.self_test:
        a = {"layers": [{"n_prototypes":8, "feature_dim":4}], "sha256":"a"}
        b = {"layers": [{"n_prototypes":8, "feature_dim":4}], "sha256":"b"}
        p = paired_score_statistics([.1, .2], [.1, .2+1e-5], 1e-4)
        assert p["within_tolerance"] and p["n_above_tolerance"] == 0
        q = verdict(True, a, b, p, .01, 1e-4, False)
        assert q["status"] == "not_equivalent" and "prototype_bytes" in q["failed_checks"]
        print("E01 offline self-test: PASS")
        return
    if not args.dataset_root:
        raise SystemExit("--dataset-root required for GPU audit")
    if not args.json_dir:
        args.json_dir = str(Path(args.dataset_root) / "jsons" / "realiad_jsons")
    if args.probe_count < 1:
        raise SystemExit("--probe-count must be at least 1")
    if args.score_tolerance < 0:
        raise SystemExit("--score-tolerance must be nonnegative")
    audit(args)


if __name__ == "__main__":
    main()
