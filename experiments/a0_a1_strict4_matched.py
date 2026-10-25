#!/usr/bin/env python3
"""A0/A1 strict-four-shot, LOSO-calibrated, fixed-feedback memory comparison.

One GPU invocation runs ONE class/support-seed/order condition with five
identically initialized scoring banks. A second script automates the matrix.
Legacy v22, R3–R6, past results are read-only and never modified.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for d in (ROOT, HERE):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import a0_a1_memory_methods as memory

SCHEMA = "a0a1.strict4.loso.matched.v1"


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def verify_supports(supports, train_normals, tests):
    if len(supports) != 4 or len(set(supports)) != 4:
        raise AssertionError("Strict protocol requires exactly 4 unique supports")
    if not set(supports).issubset(set(train_normals)):
        raise AssertionError("Support outside eligible normal training pool")
    if set(supports).intersection(set(tests)):
        raise AssertionError("Train/test support leakage")


def select_probe(stream, n: int, seed: int):
    """Label-blind probe split, disjoint from stream and feedback."""
    stream = list(stream)
    if n < 1 or len(stream) < 2*n:
        raise ValueError("Need 1 <= probe size <= half the available test images")
    indices = set(random.Random(seed).sample(range(len(stream)), n))
    train_stream = [x for i, x in enumerate(stream) if i not in indices]
    probe = [x for i, x in enumerate(stream) if i in indices]
    if set(x.path for x in train_stream) & set(x.path for x in probe):
        raise AssertionError("Probe and stream overlap")
    return train_stream, probe


def score_bank(impl, bank, img, feat, cfg):
    value, amap, _ = bank.score_sample_multiview(
        cls_name="p0", view_to_feats={"id": feat}, mode=0,
        views={"id": impl.ViewSpec("id", "id")},
        topk_patches=cfg.topk_patches, mode_scoring="boost",
        lam=.8, tau_close=.20, gamma=1.0, layer_fusion="mean",
        view_fusion="mean", img_agg=cfg.image_aggregation,
        img_topk=cfg.image_topk, img_quantile=cfg.image_quantile,
        concept_assign_view="id", img_hw=(img.height, img.width))
    score = float(value)
    if not math.isfinite(score):
        raise AssertionError("Nonfinite anomaly score")
    return score, amap


def metrics(y, scores, theta):
    from e02_matched_feedback import metrics as e02metrics
    return e02metrics(y, scores, [int(z >= theta) for z in scores])


def main_run(args):
    import torch
    import numpy as np
    import p0_protocol_audit as p0
    import e02_matched_feedback as e02
    import e02_r6_sidecar_stability as r6
    if args.math_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    impl = p0.require_sources()
    impl.set_seed(args.legacy_seed)
    output = Path(args.out_dir)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite run (including incomplete): {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    idx = impl.RealIADJSONIndex(root=args.dataset_root, json_dir=args.json_dir)
    train_normals = sorted({s.path for s in idx.train_samples
                            if s.cls_name == args.class_name and s.is_good})
    originals = sorted([s for s in idx.test_samples
                        if s.cls_name == args.class_name], key=lambda s: s.path)
    if len(originals) < 64:
        raise RuntimeError(f"Too few test images: {len(originals)}")
    supports = sorted(random.Random(args.support_seed).sample(train_normals, 4))
    verify_supports(supports, train_normals, [x.path for x in originals])
    cfg = p0.Config(
        device=args.device, augmentation_seed=args.augmentation_seed,
        bank_seed=args.bank_seed, fixed_capacity=args.fixed_capacity,
        fixed_per_image=args.fixed_per_image,
        augmentations=args.augmentations, include_identity=args.include_identity,
        multiplier=args.multiplier, feature_fuse="concat", topk_patches=32,
        image_aggregation="max", image_topk=64, image_quantile=.95, knn=5)
    backbone = impl.DINOv2MultiLayerBackbone(device=args.device)
    feature_cache = p0.encode_reference_features(impl, backbone, supports, cfg)
    canonical = p0.build_vmb(impl, feature_cache, supports, cfg)
    fixed_hash = p0.fixed_bank_profile(canonical)
    self_scores = [p0.score_image(impl, backbone, canonical, p, cfg)
                   for p in supports]
    self_theta = p0.threshold_from_scores(self_scores, cfg.multiplier)
    loso_scores, loso_folds = [], []
    for holdout in supports:
        keep = [x for x in supports if x != holdout]
        if holdout in keep or len(keep) != 3:
            raise AssertionError("LOSO support exclusion failed")
        fold = p0.build_vmb(impl, feature_cache, keep, cfg)
        fold_hash = p0.fixed_bank_profile(fold)
        loso_scores.append(p0.score_image(impl, backbone, fold, holdout, cfg))
        loso_folds.append({"heldout": holdout, "support_paths": keep,
                           "fixed_bank_sha256": fold_hash["sha256"]})
        del fold
    theta_stats = p0.threshold_from_scores(loso_scores, cfg.multiplier)
    theta = theta_stats["theta"]
    sigma = theta_stats["sigma"]
    del feature_cache
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    all_test = list(originals)
    random.Random(args.stream_seed).shuffle(all_test)
    stream, probe = select_probe(all_test, args.probe_count, args.probe_seed)
    stream = r6.order_stream(stream, args.stream_order)
    if args.max_samples:
        stream = stream[:args.max_samples]
    if not stream or set(s.path for s in stream) & set(s.path for s in probe):
        raise AssertionError("Empty or probe-contaminated stream")
    tape = e02.make_query_tape(len(stream), args.query_rate, args.query_seed)
    K = len(tape)
    arms = {name: memory.MemoryArm(name, copy.deepcopy(canonical),
                                   dynamic_capacity=args.dynamic_capacity,
                                   seed=memory.stable_seed(args.support_seed,
                                                           args.class_name, args.query_seed))
            for name in memory.ARM_NAMES}
    del canonical
    if any(p0.fixed_bank_profile(arm.bank)["sha256"] != fixed_hash["sha256"]
           for arm in arms.values()):
        raise AssertionError("Initial fixed reference bank mismatch")
    if len({tuple(arm.capacities()) for arm in arms.values()}) != 1:
        raise AssertionError("Unmatched mutable capacities")
    records, probe_rows = [], []
    y_all = []
    scores = {name: [] for name in arms}
    normal_queries = defect_queries = 0

    def probe_checkpoint(t):
        # Ground truth is written only for evaluation; never used for updates,
        # patch admission, query selection or threshold calibration.
        for sample in probe:
            img, feat = p0.encode_query(impl, backbone, sample.path, cfg)
            for name, arm in arms.items():
                sc, _ = score_bank(impl, arm.bank, img, feat, cfg)
                probe_rows.append({"step": t, "path": sample.path,
                                   "label_eval_only": 0 if sample.is_good else 1,
                                   "method": name, "score": sc,
                                   "theta_loso_fixed": theta,
                                   "pred": int(sc >= theta)})
    probe_checkpoint(0)

    for t, sample in enumerate(stream, start=1):
        img, feats = p0.encode_query(impl, backbone, sample.path, cfg)
        # One common immutable-reference score determines identical candidates.
        static_s, static_map = score_bank(
            impl, arms["static"].bank, img, feats, cfg)
        per_layer = impl.layer_to_topk_from_amap(static_map)
        common_candidates = {layer: indices[:args.patches_per_layer]
                             for layer, indices in per_layer.items()}
        scores_t = {}
        for name, arm in arms.items():
            sc = static_s if name == "static" else score_bank(
                impl, arm.bank, img, feats, cfg)[0]
            scores_t[name] = sc
            scores[name].append(sc)
        queried = t in tape
        truth = 0 if sample.is_good else 1
        # True labels are only consumed below after ALL arms have scored.
        y_all.append(truth)
        records.append({
            "t": t, "path": sample.path, "label_eval_only": truth,
            "queried": int(queried), "theta_loso_fixed": theta,
            **{f"{name}_score": sc for name, sc in scores_t.items()},
            **{f"{name}_pred": int(sc >= theta)
               for name, sc in scores_t.items()},
        })
        if queried:
            if truth == 0:
                normal_queries += 1
                for arm in arms.values():
                    arm.observe_verified_normal(
                        impl, feats, common_candidates, step=t,
                        score=static_s, theta=theta, sigma=sigma)
            else:
                defect_queries += 1
        if args.probe_every > 0 and t % args.probe_every == 0:
            probe_checkpoint(t)
        if args.print_every and t % args.print_every == 0:
            print(f"[A0A1] {args.class_name}/seed{args.support_seed}/"
                  f"{args.stream_order}: {t}/{len(stream)}", flush=True)
    if args.probe_every <= 0 or len(stream) % args.probe_every:
        probe_checkpoint(len(stream))
    assert len(y_all) == len(stream)
    assert normal_queries + defect_queries == K
    for name in arms:
        assert len(scores[name]) == len(stream)
    if any(p0.fixed_bank_profile(arm.bank)["sha256"] != fixed_hash["sha256"]
           for arm in arms.values()):
        raise AssertionError("Immutable support bank was modified")
    if any(arm.sizes() != [0] * len(arm.sizes())
           for name, arm in arms.items() if name == "static"):
        raise AssertionError("Static control acquired mutable memory")
    out_arms = {
        name: {**metrics(y_all, scores[name], theta),
               **arm.audit(),
               "score_sha256": hashlib.sha256(
                   np.asarray(scores[name], dtype="<f8").tobytes()).hexdigest()}
        for name, arm in arms.items()
    }
    if any(v["candidate_events"] != normal_queries
           for name, v in out_arms.items() if name != "static"):
        raise AssertionError("Unmatched normal-feedback events")

    probe_summary = []
    for t in sorted({r["step"] for r in probe_rows}):
        for name in arms:
            rs = [r for r in probe_rows
                  if r["step"] == t and r["method"] == name]
            probe_summary.append({
                "step": t, "method": name,
                **metrics([r["label_eval_only"] for r in rs],
                          [r["score"] for r in rs], theta),
                "n_probe": len(rs),
            })
    with (output / "prequential.csv").open(
            "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output / "probe_scores.csv").open(
            "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(probe_rows[0]))
        writer.writeheader()
        writer.writerows(probe_rows)
    with (output / "probe_summary.csv").open(
            "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(probe_summary[0]))
        writer.writeheader()
        writer.writerows(probe_summary)
    report = {
        "schema": SCHEMA,
        "class_name": args.class_name, "support_seed": args.support_seed,
        "stream_order": args.stream_order,
        "protocol": {
            "support_selection": "strict_random_four_shot",
            "support_image_access_count": 4,
            "training_candidate_paths_indexed_not_encoded": len(train_normals),
            "support_paths": supports, "support_sha256": sha(supports),
            "calibration": "four_fold_leave_entire_support_image_and_all_augmentations_out",
            "loso_fold_lineage": loso_folds, "loso_scores": loso_scores,
            "loso_stats": theta_stats, "diagnostic_self_match_scores": self_scores,
            "diagnostic_self_match_stats": self_theta,
            "theta0_shared_across_all_methods": theta,
            "fixed_bank": fixed_hash, "feature_configuration": asdict(cfg),
            "stream_paths_sha256": sha([x.path for x in stream]),
            "probe_paths_sha256": sha([x.path for x in probe]),
            "probe_paths": [x.path for x in probe],
            "query_positions_1based": sorted(tape),
            "query_tape_sha256": sha(sorted(tape)),
            "query_rate_target": args.query_rate,
            "observed_query_normals": normal_queries,
            "observed_query_defects": defect_queries,
            "candidate_patches_from_shared_static_score": True,
            "patches_per_layer_per_normal_query": args.patches_per_layer,
            "labels_after_predictions": True,
            "probe_not_in_training_or_feedback": True,
            "no_adaptive_threshold_in_primary_experiment": True,
            "memory_capacity_matched_per_layer": True,
            "dynamic_capacity_per_layer": args.dynamic_capacity,
        },
        "n_stream": len(stream), "n_probe": len(probe), "K": K,
        "arms": out_arms, "probe_snapshots": probe_summary,
        "configuration": vars(args),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "This is a STRICT-4 and matched-label memory-only test, not full v22.",
            "LOSO estimates come from 3-support fold banks; calibration can be biased for 4-support deployed bank.",
            "Frozen theta is numerically identical across all arms; memory scores may shift under mutation.",
            "Candidates are selected from a shared STATIC reference top-k, isolating retention but not reproducing every arm's native feature selection.",
            "v22_utility_stm uses the original codebook utility retention, with LTM/maturation/correction disabled.",
            "Greedy coreset is a streaming min-pair-gap heuristic, not global optimized coreset; O(K^2) ephemeral computation/memory.",
            "Default dynamic capacity is 128 prototypes per layer (capacity pressure), unlike historical R6 1024; all arms share the same configured bound.",
            "FIFO and reservoir store all shared candidate patches; utility method novelty-gates writes; compare actual occupancy and runtime.",
            "Probe images are held-out from online stream; labels used solely for offline evaluation.",
            "AUROC over evolving prequential scores may confound memory state and stream order.",
            "No production online policy, adaptive sidecar, bootstrap, or claims of complete architecture equivalence.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False,
                   default=str), encoding="utf-8")
    print(json.dumps({
        "condition": f"{args.class_name}/s{args.support_seed}/{args.stream_order}",
        "n": len(stream), "K": K, "theta": theta,
        "memory_AUROCs": {x: round(v["auroc"], 5) if v["auroc"] is not None else None
                          for x, v in out_arms.items()},
        "BA": {x: round(v["balanced_accuracy"], 5)
                if v["balanced_accuracy"] is not None else None
                for x, v in out_arms.items()},
        "out_dir": str(output)}, ensure_ascii=False, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default="G:/dataset/realiad")
    p.add_argument("--json-dir", default=None)
    p.add_argument("--class-name", default="audiojack")
    p.add_argument("--support-seed", type=int, default=0)
    p.add_argument("--legacy-seed", type=int, default=0)
    p.add_argument("--stream-seed", type=int, default=0)
    p.add_argument("--probe-seed", type=int, default=212)
    p.add_argument("--query-seed", type=int, default=37)
    p.add_argument("--query-rate", type=float, default=.03)
    p.add_argument("--probe-count", type=int, default=32)
    p.add_argument("--probe-every", type=int, default=512)
    p.add_argument("--stream-order", choices=(
        "native", "normal_first", "defect_first", "alternating_blocks"),
        default="native")
    p.add_argument("--max-samples", type=int, default=0,
                   help="0 = full stream after fixed probe split")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fixed-capacity", type=int, default=25000)
    p.add_argument("--fixed-per-image", type=int, default=5000)
    p.add_argument("--dynamic-capacity", type=int, default=128,
                   help="Mutable normal-patch prototypes per layer, same for every arm")
    p.add_argument("--augmentations", type=int, default=4)
    p.add_argument("--augmentation-seed", type=int, default=0)
    p.add_argument("--bank-seed", type=int, default=0)
    p.add_argument("--include-identity", action="store_true")
    p.add_argument("--multiplier", type=float, default=2.5)
    p.add_argument("--patches-per-layer", type=int, default=8)
    p.add_argument("--math-sdpa", action="store_true")
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--out-dir", required=True)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.json_dir is None:
        args.json_dir = str(Path(args.dataset_root) / "jsons" / "realiad_jsons")
    if not (0 < args.query_rate <= 1 and 0 <= args.max_samples
            and 0 < args.patches_per_layer <= 32 and args.probe_count >= 4
            and 1 <= args.dynamic_capacity <= 1024):
        raise SystemExit("Invalid query rate/sample/patch/probe settings")
    main_run(args)
