#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E02: matched-stream 2x2 STM x theta experiment on canonical v22 initialization.

No edits to v22. All four arms:
- start from one *real v22* initialized fixed bank, same four support images,
  same controller theta0, same scoring settings and shared encoded query tensors;
- predict each image before seeing that image's label, and only reveal labels
  at positions chosen by a seeded, label-blind, fixed 3% query tape;
- change only verified-normal STM writes and/or queried-label theta updates.

Scope: verified-normal STM (v22 normal_dyn_only write primitive) vs v22 guarded
theta sidecar. This is NOT a reproduction of full BEA/correction/LTM, and the
query tape is a fixed experimental control rather than a learned query policy.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
from typing import Any
from unittest.mock import patch

import numpy as np


ARM_FLAGS = {
    "static": (False, False),
    "stm_only": (True, False),
    "theta_only": (False, True),
    "stm_theta": (True, True),
}


def make_query_tape(n: int, budget: float, seed: int) -> set[int]:
    """One-indexed positions; no labels or model scores are inspected."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= budget <= 1:
        raise ValueError("query rate outside [0,1]")
    k = min(n, int(np.floor(n * budget + 0.5)))
    return set(random.Random(seed).sample(range(1, n + 1), k))


def metrics(y: list[int], score: list[float], prediction: list[int]) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score
    ya = np.asarray(y, dtype=np.int64)
    sa = np.asarray(score, dtype=np.float64)
    pa = np.asarray(prediction, dtype=np.int64)
    if not (len(ya) == len(sa) == len(pa)) or not len(ya):
        raise ValueError("Invalid metric input lengths")
    tp = int(((ya == 1) & (pa == 1)).sum())
    tn = int(((ya == 0) & (pa == 0)).sum())
    fp = int(((ya == 0) & (pa == 1)).sum())
    fn = int(((ya == 1) & (pa == 0)).sum())
    return {
        "n": len(ya), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(ya),
        "balanced_accuracy": (0.5 * (tp / (tp + fn) + tn / (tn + fp))
                              if tp + fn and tn + fp else None),
        "fpr": fp / (fp + tn) if fp + tn else None,
        "fnr": fn / (tp + fn) if tp + fn else None,
        "auroc": float(roc_auc_score(ya, sa)) if len(set(y)) == 2 else None,
    }


@dataclass
class Arm:
    name: str
    stm: bool
    theta_update: bool
    bank: Any
    controller: Any
    scores: list[float]
    thresholds: list[float]
    predictions: list[int]
    verified_normal_writes: int = 0
    verified_normal_update_calls: int = 0
    label_observations: int = 0


def _bootstrap_imports():
    root = Path(__file__).resolve().parents[1]
    for path in (root, root / "experiments"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import test_phase3_AIF_simplified_v22 as impl
    import p0_v22_equivalence as e01
    return impl, e01


def init_canonical(args, impl, e01, out: Path):
    """Wrap only the original threshold initializer to retain its object."""
    config = e01.parser().parse_args([
        "--dataset-root", args.dataset_root,
        "--json-dir", args.json_dir,
        "--class-name", args.class_name,
        "--support-seed", str(args.support_seed),
        "--legacy-seed", str(args.legacy_seed),
        "--stream-seed", str(args.stream_seed),
        "--fixed-capacity", str(args.fixed_capacity),
        "--fixed-per-image", str(args.fixed_per_image),
        "--augmentations", str(args.augmentations),
        "--image-aggregation", args.image_aggregation,
        "--topk-patches", str(args.topk_patches),
        "--knn", str(args.knn),
        "--out-dir", str(out / "canonical"),
    ])
    config.math_sdpa = args.math_sdpa
    config.include_identity = args.include_identity
    config.feature_fuse = args.feature_fuse

    captured = {}
    original = impl.GuardedDualAnchorThresholdController.init_from_support

    def intercept(self, support_scores):
        result = original(self, support_scores)
        captured["controller"] = self
        return result

    with patch.object(impl.GuardedDualAnchorThresholdController,
                      "init_from_support", intercept):
        init, artifacts, _ = e01.run_canonical(config, impl, out / "canonical")
    if "controller" not in captured:
        raise AssertionError("Original v22 guarded theta initializer was not reached.")
    model = artifacts["backbone"]
    bank = artifacts["bank"]
    controller = captured["controller"]
    if not model or not bank:
        raise AssertionError("Canonical v22 model/bank not captured.")
    if abs(float(controller.theta()) - float(init["theta0"])) > 1e-7:
        raise AssertionError("Canonical theta object inconsistent with init dump.")
    return init, model, bank, controller


def create_arms(canonical_bank, canonical_controller, impl, e01, cls: str):
    baseline_hash = e01.memory_profile(canonical_bank, cls)["sha256"]
    initial_theta = float(canonical_controller.theta())
    arms = {}
    for name, (stm, adaptive_theta) in ARM_FLAGS.items():
        bank = copy.deepcopy(canonical_bank)
        ctrl = copy.deepcopy(canonical_controller)
        if e01.memory_profile(bank, cls)["sha256"] != baseline_hash:
            raise AssertionError(f"Initial bank differs for arm={name}")
        if abs(ctrl.theta() - initial_theta) > 1e-9:
            raise AssertionError(f"Initial theta differs for arm={name}")
        # Disable any indirect memory sidecars: STM-only is *just* STM.
        bank.enable_ltm_retrieval = False
        bank.enable_ltm_promotion = False
        bank.enable_maturation_buffer_retrieval = False
        bank.enable_maturation_buffer = False
        bank.enable_utility_aware_stm_retention = False
        bank.use_defect_bank = False
        bank.corr_cfg.enabled = False
        arms[name] = Arm(
            name=name, stm=stm, theta_update=adaptive_theta,
            bank=bank, controller=ctrl, scores=[], thresholds=[], predictions=[]
        )
    return arms, baseline_hash, initial_theta


def frozen_score(impl, arm, cls, features, image, args):
    value, amap, _ = arm.bank.score_sample_multiview(
        cls_name=cls, view_to_feats={"id": features}, mode=0,
        views={"id": impl.ViewSpec("id", "id")},
        topk_patches=args.topk_patches,
        mode_scoring="boost", lam=.8, tau_close=.20, gamma=1.0,
        layer_fusion="mean", view_fusion="mean",
        img_agg=args.image_aggregation, img_topk=64, img_quantile=.95,
        concept_assign_view="id", img_hw=(image.height, image.width),
    )
    return float(value), amap


def write_verified_normal(impl, arm, cls, features, amap, score, theta, t):
    """Selected v22 normal_dyn_only primitive; original source unchanged.

    No pseudo-normal self-training, correction, defect update or LTM promotion.
    The original normal_dyn_only writes all verified normal queries. Its
    select_informative_normal_write produces weights; novel/global coverage is
    held at zero here to avoid introducing an additional changing memory.
    """
    pred_defect = bool(score >= theta)
    _allow, _budget, _reason, info = impl.select_informative_normal_write(
        score=score, theta=theta, sigmaN=arm.controller.sigmaN(),
        pred_defect=pred_defect, novelty=0.0
    )
    layer_to_idx = impl.layer_to_topk_from_amap(amap)
    if not layer_to_idx or not any(layer_to_idx.values()):
        raise AssertionError("No topk patch IDs available for verified normal STM update.")
    outcome = "FP" if pred_defect else "TN"
    stats = arm.bank.update_from_topk(
        cls_name=cls, feats_per_layer=features, which="normal_dyn",
        mode=0, view="id", layer_to_patch_idx=layer_to_idx,
        tau_insert=.15, tau_replace=.25,
        event_id=t, write_value=float(info["write_value"]),
        outcome=outcome, boundary_value=float(info["boundary"]),
        coverage_gain=0.0
    )
    arm.verified_normal_update_calls += 1
    if any(int(v.get("n_added", 0)) + int(v.get("n_replaced", 0)) > 0
           for v in stats.get("layers", {}).values()):
        arm.verified_normal_writes += 1
    return stats


def select_stream(impl, args):
    idx = impl.RealIADJSONIndex(root=args.dataset_root, json_dir=args.json_dir)
    samples = sorted(
        [s for s in idx.test_samples if s.cls_name == args.class_name],
        key=lambda s: s.path
    )
    if not samples:
        raise RuntimeError(f"Empty class test stream: {args.class_name}")
    random.Random(args.stream_seed).shuffle(samples)
    if args.max_samples:
        samples = samples[:args.max_samples]
    return samples


def audit(args) -> dict:
    import torch
    impl, e01 = _bootstrap_imports()
    if args.math_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Main model gets initialized only once from original v22.
    print("[E02] Canonical v22 cold start...", flush=True)
    init, backbone, canonical_bank, canonical_controller = init_canonical(
        args, impl, e01, out
    )
    arms, fixed_sha, initial_theta = create_arms(
        canonical_bank, canonical_controller, impl, e01, args.class_name
    )
    # No future model calls mutate the authoritative canonical reference.
    del canonical_bank, canonical_controller
    stream = select_stream(impl, args)
    tape = make_query_tape(len(stream), args.query_rate, args.query_seed)
    tape_hash = sha256(json.dumps(sorted(tape)).encode()).hexdigest()
    input_paths_hash = sha256(json.dumps([x.path for x in stream]).encode()).hexdigest()
    tape_manifest = {
        "stream_length": len(stream), "query_positions_1based": sorted(tape),
        "query_rate_target": args.query_rate, "query_rate_realized": len(tape) / len(stream),
        "query_seed": args.query_seed, "stream_seed": args.stream_seed,
        "sha256": tape_hash, "stream_paths_sha256": input_paths_hash,
        "label_blind": True,
    }
    (out / "e02_query_tape.json").write_text(
        json.dumps(tape_manifest, indent=2), encoding="utf-8")
    y_all, rows = [], []
    normal_queries = defect_queries = 0
    for t, sample in enumerate(stream, start=1):
        # Exactly one DINOv2 feature encoding per time step shared by all arms.
        image = impl.load_rgb(sample.path)
        features, _global = impl.encode_pil(backbone, image, args.feature_fuse)
        features = [impl.l2_normalize(x) for x in features]

        state = {}
        for name, arm in arms.items():
            theta = float(arm.controller.theta())
            score, amap = frozen_score(impl, arm, args.class_name, features, image, args)
            arm.scores.append(score)
            arm.thresholds.append(theta)
            arm.predictions.append(int(score >= theta))
            state[name] = {"score": score, "theta": theta, "amap": amap}

        # Audit the identical start BEFORE any label or state update.
        if t == 1:
            probe_scores = [arm.scores[-1] for arm in arms.values()]
            if max(probe_scores) - min(probe_scores) > args.score_tolerance:
                raise AssertionError(
                    f"Different first-image frozen scores despite cloned banks: {probe_scores}"
                )
        # Truth enters ONLY the metrics stream and, if this step is queried,
        # the supervised update. No lookahead or oracle query selection.
        y = 0 if sample.is_good else 1
        y_all.append(y)
        asked = t in tape
        rows.append({
            "t": t, "path": sample.path, "y_eval_only": y,
            "queried": int(asked),
            **{f"{name}_{metric}": (
                    int(arm.predictions[-1]) if metric == "pred" else
                    state[name][metric]
                )
                for name, arm in arms.items()
                for metric in ("score", "theta", "pred")}
        })
        if asked:
            if y == 0:
                normal_queries += 1
            else:
                defect_queries += 1
            # Apply updates AFTER all four predictions are saved.
            for name, arm in arms.items():
                if arm.stm and y == 0:
                    write_verified_normal(
                        impl, arm, args.class_name, features,
                        state[name]["amap"], state[name]["score"],
                        state[name]["theta"], t
                    )
                if arm.theta_update:
                    arm.controller.observe_labeled(state[name]["score"], y)
                    arm.label_observations += 1
        if args.print_every and t % args.print_every == 0:
            print(f"[E02] scored={t}/{len(stream)} queried={normal_queries+defect_queries}",
                  flush=True)

    if any(len(arm.scores) != len(stream) for arm in arms.values()):
        raise AssertionError("Different observation count across arms.")
    # Crucial structural invariants: static bank/threshold never changed;
    # theta-only bank stays empty; stm-only theta stays frozen.
    for name, arm in arms.items():
        if e01.memory_profile(arm.bank, args.class_name)["sha256"] != fixed_sha:
            raise AssertionError(f"Initial fixed memory mutated for {name}.")
        if not arm.theta_update and any(
            abs(z - initial_theta) > 1e-9 for z in arm.thresholds
        ):
            raise AssertionError(f"Frozen threshold changed: {name}")
        if not arm.stm and arm.verified_normal_update_calls != 0:
            raise AssertionError(f"Unexpected STM writes: {name}")
        if arm.theta_update and arm.label_observations != len(tape):
            raise AssertionError(f"Threshold saw wrong number of labels: {name}")
    summaries = {}
    for name, arm in arms.items():
        size = sum(
            cb.size() for cb in arm.bank.normal_dyn[args.class_name][0]["id"]
        )
        summaries[name] = {
            **metrics(y_all, arm.scores, arm.predictions),
            "query_count_available": len(tape),
            "stm_write_events": arm.verified_normal_writes,
            "stm_update_calls": arm.verified_normal_update_calls,
            "dynamic_stm_prototypes": size,
            "theta_label_observations": arm.label_observations,
            "initial_theta": initial_theta,
            "final_theta": float(arm.controller.theta()),
        }
    with (out / "e02_prequential_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema": "e02.matched.feedback.factorial.v1",
        "class": args.class_name,
        "canonical_init": {
            "support_paths": init["support_paths_per_mode"]["0"],
            "fixed_bank_sha256": fixed_sha,
            "theta0": initial_theta,
            "backbone": {"name": getattr(backbone, "model_name", None),
                         "img_size": getattr(backbone, "img_size", None),
                         "resize_size": getattr(backbone, "resize_size", None),
                         "layers": list(getattr(backbone, "layers", []))},
        },
        "query_tape": tape_manifest,
        "observed_feedback": {
            "queried_normals": normal_queries, "queried_defects": defect_queries,
            "total_queried": normal_queries + defect_queries,
        },
        "arms": summaries,
        "settings": vars(args),
        "limitations": [
            "This is a matched fixed-feedback causal pilot, NOT native BEA.",
            "STM writes use v22's normal_dyn_only update primitive, with novelty held at zero.",
            "Only verified normals update STM; threshold controller receives both verified classes.",
            "No unlabeled pseudo-normal updates, correction, defect bank updates, LTM, or maturation.",
            "Threshold arm uses verified-label observe_labeled without unlabeled pseudo-observations.",
            "Canonical initial state is the ORIGINAL v22 initialization, not the P0 pilot.",
            "Prequential AUROC pools time-varying scores; additionally evaluate on an untouched fixed probe set in later work.",
            "This is only one category and one support/query seed, not publication-ready evidence.",
        ],
    }
    (out / "e02_factorial_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[E02] FINAL METRICS", json.dumps(summaries, indent=2), flush=True)
    print("[E02] Saved:", out, flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root")
    p.add_argument("--json-dir")
    p.add_argument("--class-name", default="audiojack")
    p.add_argument("--device", default="cuda")
    p.add_argument("--support-seed", type=int, default=0)
    p.add_argument("--legacy-seed", type=int, default=0)
    p.add_argument("--stream-seed", type=int, default=0)
    p.add_argument("--query-seed", type=int, default=37)
    p.add_argument("--query-rate", type=float, default=.03)
    p.add_argument("--max-samples", type=int, default=100, help="0 means all class test images")
    p.add_argument("--fixed-capacity", type=int, default=25000)
    p.add_argument("--fixed-per-image", type=int, default=5000)
    p.add_argument("--augmentations", type=int, default=4)
    p.add_argument("--include-identity", action="store_true")
    p.add_argument("--feature-fuse", default="concat", choices=["concat", "sum", "none"])
    p.add_argument("--image-aggregation", default="max",
                   choices=["max", "topk_mean", "spiky_quantile"])
    p.add_argument("--topk-patches", type=int, default=32)
    p.add_argument("--knn", type=int, default=5)
    p.add_argument("--math-sdpa", action="store_true")
    p.add_argument("--score-tolerance", type=float, default=1e-5)
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--out-dir", default="outputs/e02_audiojack_smoke")
    p.add_argument("--self-test", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if args.self_test:
        tape = make_query_tape(100, .03, 37)
        assert len(tape) == 3 and tape == make_query_tape(100, .03, 37)
        assert make_query_tape(100, 0, 37) == set()
        assert len(make_query_tape(100, 1, 37)) == 100
        m = metrics([0, 0, 1, 1], [.1, .2, .8, .9], [0, 0, 1, 1])
        assert m["accuracy"] == 1 and m["auroc"] == 1
        print("E02 offline self-test: PASS")
        return
    if not args.dataset_root:
        raise SystemExit("--dataset-root required")
    if not args.json_dir:
        args.json_dir = str(Path(args.dataset_root) / "jsons" / "realiad_jsons")
    if args.max_samples < 0:
        raise SystemExit("--max-samples must be nonnegative (0=all)")
    if args.score_tolerance < 0:
        raise SystemExit("--score-tolerance must be nonnegative")
    if not 0 <= args.query_rate <= 1:
        raise SystemExit("--query-rate must be in [0,1]")
    audit(args)


if __name__ == "__main__":
    main()
