#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0a/P0b plus static P0c pilot for the frozen v22 scoring substrate.

Important: this experiment does NOT demonstrate an online-learning gain.
It audits support access, leave-one-support-out (LOSO) threshold bias,
and the operating points of the SAME frozen bank under different theta0.
The source v22 and vmb_visionad_new modules are imported, not modified.

Example:
 python experiments/p0_protocol_audit.py --dataset-root G:/dataset/realiad \
   --json-dir G:/dataset/realiad/jsons/realiad_jsons --class-name audiojack \
   --support-policy both --seeds 0,1,2 --max-test 300 \
   --out-dir outputs/p0_audiojack
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def threshold_from_scores(values: list[float], multiplier: float = 2.5) -> dict[str, float]:
    """Match the unguarded v22 support initializer (z_prior_anom=5)."""
    xs = np.asarray(values, dtype=np.float64)
    if not len(xs) or not np.isfinite(xs).all():
        raise ValueError("Empty/nonfinite normal calibration scores.")
    mu = float(xs.mean())
    sigma = max(float(xs.std(ddof=1 if len(xs) > 1 else 0)), 1e-6)
    return dict(mu=mu, sigma=sigma, theta=mu + float(multiplier) * sigma, n=int(len(xs)))


def metrics(y: list[int], score: list[float], theta: float) -> dict[str, Any]:
    """All labels are for OFFLINE evaluation; none reach the memory scorer."""
    from sklearn.metrics import roc_auc_score
    ya = np.asarray(y, dtype=np.int64)
    sa = np.asarray(score, dtype=np.float64)
    pred = sa > theta
    tp = int(((pred == 1) & (ya == 1)).sum())
    tn = int(((pred == 0) & (ya == 0)).sum())
    fp = int(((pred == 1) & (ya == 0)).sum())
    fn = int(((pred == 0) & (ya == 1)).sum())
    return {
        "n": len(y), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(y) if y else None,
        "balanced_accuracy": 0.5 * (
            tp / (tp + fn) + tn / (tn + fp)
        ) if tp + fn and tn + fp else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "fnr": fn / (fn + tp) if fn + tp else None,
        "auroc": float(roc_auc_score(ya, sa)) if len(set(y)) == 2 else None,
        "theta": float(theta),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Config:
    device: str
    seed: int
    fixed_capacity: int
    fixed_per_image: int
    augmentations: int
    include_identity: bool
    multiplier: float
    feature_fuse: str
    topk_patches: int
    image_aggregation: str
    image_topk: int
    image_quantile: float
    knn: int


def require_sources():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import test_phase3_AIF_simplified_v22 as impl
        return impl
    except ImportError as exc:
        raise RuntimeError(
            "Install torch/torchvision/scipy/sklearn/pandas/timm and put the "
            "original v22 and vmb_visionad_new modules in the repository root."
        ) from exc


def select_support(impl, backbone, all_normals: list[str],
                   protocol: str, seed: int, shots: int) -> tuple[list[str], int]:
    """Strict selection decodes ONLY the chosen shots; pool reads all candidates."""
    if len(all_normals) < shots:
        raise ValueError(f"Need {shots} normal training images, got {len(all_normals)}")
    if protocol == "strict4":
        rng = random.Random(seed)
        return sorted(rng.sample(all_normals, shots)), shots
    if protocol == "pool_assisted4":
        centroids, labels = impl.build_modes_from_train_good(
            backbone, all_normals, K_modes=1, seed=seed
        )
        candidates = impl.select_clean_support_paths_per_mode(
            backbone, all_normals, labels, centroids,
            shots_per_mode=shots, seed=seed
        )
        chosen = candidates.get(0, [])
        if len(chosen) != shots:
            raise RuntimeError("Pool-assisted constructor did not produce exactly four supports.")
        return chosen, len(all_normals)
    raise ValueError(protocol)


def encode_reference_features(impl, backbone, paths: list[str], cfg: Config):
    """Encode reference views once; reuse the SAME features across LOSO folds.

    Caching on CPU ensures the held-out image AND all its augmentations are
    excluded from the fold bank. Scores are independent of any test labels.
    """
    import torch
    augmentation = impl.SupportAugConfig()
    cache = {}
    for path in paths:
        img = impl.load_rgb(path)
        # Deterministic per-original-image augmentation, independent of fold.
        image_seed = int(hashlib.sha256((str(cfg.seed) + "|" + path).encode()).hexdigest()[:8], 16)
        random.seed(image_seed)
        np.random.seed(image_seed % (2**32 - 1))
        torch.manual_seed(image_seed)
        refs = impl.make_support_refs(
            img, backbone, augmentation, cfg.augmentations,
            baseline_compat=False, include_identity=cfg.include_identity
        )
        entries = []
        for ref in refs:
            feats, _ = impl.encode_pil(backbone, ref, cfg.feature_fuse)
            entries.append([impl.l2_normalize(t).detach().to("cpu", dtype=torch.float16) for t in feats])
        if not entries:
            raise RuntimeError("No encoded support references; check --augmentations and --include-identity.")
        cache[path] = entries
    return cache


def build_vmb(impl, cache, chosen: list[str], cfg: Config):
    """Build a frozen, identity-view v22 PatchVMB without dynamic updates."""
    import torch
    bank = impl.PatchVMB(
        device=cfg.device,
        K_normal_fixed=cfg.fixed_capacity,
        K_normal_dyn=1024,
        K_normal_ltm=512,
        K_defect=256,
        K_concept=4096,
        knn_k_normal=cfg.knn,
        corr_cfg=impl.CorrectionConfig(enabled=False),
        use_defect_bank=False,
    )
    bank.bank_dtype = torch.float16
    # First seed concept IDs from allowed supports only. IDs do not affect dN.
    init_feat = cache[chosen[0]][0]
    bank._init_concepts_if_needed("p0", [t.to(cfg.device) for t in init_feat])
    bank.concept_cb["p0"][0].update(init_feat[0][:min(256, len(init_feat[0]))].to(cfg.device))
    rng = np.random.RandomState(cfg.seed)
    refs = []
    for path in chosen:
        for featset in cache[path]:
            slim = []
            for x in featset:
                k = min(cfg.fixed_per_image, int(x.shape[0]))
                idx = rng.choice(int(x.shape[0]), size=k, replace=False)
                slim.append(x[idx].to(cfg.device))
            refs.append(slim)
    bank.build_fixed_normal_from_support(
        cls_name="p0", support_feats=refs, mode=0,
        view="id", normal_seed_patches=8192, seed=cfg.seed,
    )
    if not bank.normal_fixed["p0"][0]["id"][0].size():
        raise RuntimeError("Fixed normal bank is empty.")
    return bank


def score_image(impl, backbone, bank, path: str, cfg: Config) -> float:
    img = impl.load_rgb(path)
    feat, _ = impl.encode_pil(backbone, img, cfg.feature_fuse)
    value, _, _ = bank.score_sample_multiview(
        cls_name="p0", view_to_feats={"id": feat}, mode=0,
        views={"id": impl.ViewSpec("id", "id")},
        topk_patches=cfg.topk_patches,
        mode_scoring="boost",
        lam=0.8, tau_close=0.20, gamma=1.0,
        layer_fusion="mean", view_fusion="mean",
        img_agg=cfg.image_aggregation,
        img_topk=cfg.image_topk, img_quantile=cfg.image_quantile,
        concept_assign_view="id", img_hw=(img.height, img.width),
    )
    return float(value)


def run_one(args, impl, backbone, index, cfg: Config,
            protocol: str, seed: int, out: Path) -> dict[str, Any]:
    import torch
    train = sorted({s.path for s in index.train_samples
                    if s.cls_name == args.class_name and s.is_good})
    tests = sorted([s for s in index.test_samples if s.cls_name == args.class_name],
                   key=lambda s: s.path)
    if not tests:
        raise RuntimeError(f"No test samples for class {args.class_name}")
    chosen, pixels_accessed = select_support(
        impl, backbone, train, protocol, seed, args.shots
    )
    if len(chosen) != args.shots or len(set(chosen)) != args.shots:
        raise AssertionError("Support not exactly four distinct images.")
    cfg.seed = seed
    cache = encode_reference_features(impl, backbone, chosen, cfg)
    bank = build_vmb(impl, cache, chosen, cfg)
    self_scores = [score_image(impl, backbone, bank, x, cfg) for x in chosen]
    self_stats = threshold_from_scores(self_scores, cfg.multiplier)

    loso_scores = []
    for heldout in chosen:
        remaining = [x for x in chosen if x != heldout]
        assert len(remaining) == args.shots - 1 and heldout not in remaining
        fold_bank = build_vmb(impl, cache, remaining, cfg)
        loso_scores.append(score_image(impl, backbone, fold_bank, heldout, cfg))
        del fold_bank
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    loso_stats = threshold_from_scores(loso_scores, cfg.multiplier)

    # Extra normal commissioning is a separate protocol, NOT strict four-shot.
    cal_stats = None
    cal_paths = []
    if args.commissioning_size:
        candidates = [x for x in train if x not in set(chosen)]
        if len(candidates) < args.commissioning_size:
            raise ValueError("Not enough held-out normal commissioning images.")
        cal_paths = sorted(random.Random(seed + 100003).sample(
            candidates, args.commissioning_size
        ))
        cal_scores = [score_image(impl, backbone, bank, x, cfg) for x in cal_paths]
        cal_stats = threshold_from_scores(cal_scores, cfg.multiplier)

    rng = random.Random(args.stream_seed)
    rng.shuffle(tests)
    if args.max_test:
        tests = tests[:args.max_test]
    y, scores = [], []
    rows = []
    for idx, sample in enumerate(tests):
        val = score_image(impl, backbone, bank, sample.path, cfg)
        truth = 0 if sample.is_good else 1
        y.append(truth)
        scores.append(val)
        rows.append({"path": sample.path, "label": truth, "score": val})
        if args.print_every and (idx + 1) % args.print_every == 0:
            print(f"  scored={idx+1}/{len(tests)}", flush=True)
    results = {
        "self": metrics(y, scores, self_stats["theta"]),
        "loso": metrics(y, scores, loso_stats["theta"]),
    }
    if cal_stats:
        results["commissioning"] = metrics(y, scores, cal_stats["theta"])
    meta = {
        "schema": "p0.protocol.audit.v1",
        "class_name": args.class_name, "protocol": protocol,
        "support_seed": seed, "stream_seed": args.stream_seed,
        "supports": chosen, "commissioning_paths": cal_paths,
        "n_available_normal_paths_indexed": len(train),
        "n_predeployment_images_decoded": (
            pixels_accessed + (args.commissioning_size or 0)
        ),
        "n_support_images": len(chosen),
        "strict_four_shot_eligible": protocol == "strict4" and not cal_paths,
        "thresholds": {
            "self": {**self_stats, "scores": self_scores},
            "loso": {**loso_stats, "scores": loso_scores},
            **({"commissioning": cal_stats} if cal_stats else {}),
        },
        "metrics": results,
        "source_sha256": {
            x: sha256_file(Path(__file__).resolve().parents[1] / x)
            for x in ("test_phase3_AIF_simplified_v22.py", "vmb_visionad_new.py")
        },
        "configuration": vars(args),
        "limits": [
            "Frozen/static detector only; online full-system P0c remains to be tested.",
            "LOSO scores are produced with three-shot reference banks and may be pessimistic for a deployed four-shot bank.",
            "This audit uses v22 scoring classes with independently constructed, cached support features; it is not a byte-for-byte reproduction of historical v22 support-bank sampling.",
            "Both static operating points share exactly the same query/test scores.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.class_name}_{protocol}_seed{seed}"
    with (out / f"{prefix}.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with (out / f"{prefix}_scores.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "score"])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"case": prefix, "theta_self": self_stats["theta"],
                      "theta_loso": loso_stats["theta"], "metrics": results}, indent=2),
          flush=True)
    del bank, cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", required=False, help="Real-IAD dataset root")
    p.add_argument("--json-dir", default=None, help="Real-IAD per-category JSON directory")
    p.add_argument("--class-name", default="audiojack")
    p.add_argument("--support-policy", choices=["strict4", "pool_assisted4", "both"], default="strict4")
    p.add_argument("--seeds", default="0", help="Comma-separated support seeds")
    p.add_argument("--stream-seed", type=int, default=0)
    p.add_argument("--shots", type=int, default=4, choices=[4])
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-test", type=int, default=0, help="0=entire class test set")
    p.add_argument("--commissioning-size", type=int, default=0, help="Extra normals: different data access protocol")
    p.add_argument("--fixed-capacity", type=int, default=25000)
    p.add_argument("--fixed-per-image", type=int, default=5000)
    p.add_argument("--augmentations", type=int, default=4)
    p.add_argument("--include-identity", action="store_true")
    p.add_argument("--feature-fuse", choices=["concat", "none", "sum"], default="concat")
    p.add_argument("--topk-patches", type=int, default=32)
    p.add_argument("--image-aggregation", choices=["max", "topk_mean", "spiky_quantile"], default="max")
    p.add_argument("--image-topk", type=int, default=64)
    p.add_argument("--image-quantile", type=float, default=0.95)
    p.add_argument("--knn", type=int, default=5)
    p.add_argument("--multiplier", type=float, default=2.5)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--out-dir", default="outputs/p0")
    p.add_argument("--self-test", action="store_true", help="Offline numerical test; no datasets/GPU")
    return p


def main():
    args = parser().parse_args()
    if args.self_test:
        vals = [0.0, 0.1, 0.2, 0.3]
        t = threshold_from_scores(vals)
        assert t["n"] == 4 and t["theta"] > 0.3
        m = metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 0.5)
        assert m["accuracy"] == 1.0 and m["fpr"] == 0.0
        print("P0 numerical self-test: PASS")
        return
    if not args.dataset_root:
        raise SystemExit("--dataset-root is required unless --self-test.")
    if args.commissioning_size < 0:
        raise SystemExit("--commissioning-size must be nonnegative.")
    import torch
    impl = require_sources()
    json_dir = args.json_dir or str(Path(args.dataset_root) / "jsons" / "realiad_jsons")
    dataset = impl.RealIADJSONIndex(root=args.dataset_root, json_dir=json_dir)
    protocols = (["strict4", "pool_assisted4"] if args.support_policy == "both"
                 else [args.support_policy])
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("No support seeds.")
    cfg = Config(
        device=args.device, seed=0, fixed_capacity=args.fixed_capacity,
        fixed_per_image=args.fixed_per_image, augmentations=args.augmentations,
        include_identity=args.include_identity, multiplier=args.multiplier,
        feature_fuse=args.feature_fuse, topk_patches=args.topk_patches,
        image_aggregation=args.image_aggregation, image_topk=args.image_topk,
        image_quantile=args.image_quantile, knn=args.knn
    )
    impl.set_seed(0)
    backbone = impl.DINOv2MultiLayerBackbone(device=args.device)
    cases = []
    try:
        for proto in protocols:
            for seed in seeds:
                print(f"\n[P0] {args.class_name} {proto} seed={seed}", flush=True)
                impl.set_seed(seed)
                cases.append(run_one(args, impl, backbone, dataset, cfg, proto, seed,
                                     Path(args.out_dir)))
    finally:
        del backbone
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {
        "schema": "p0.protocol.batch.v1",
        "cases": [
            {
                "protocol": x["protocol"],
                "seed": x["support_seed"],
                "strict_four_shot_eligible": x["strict_four_shot_eligible"],
                "support_paths": x["supports"],
                "theta_self": x["thresholds"]["self"]["theta"],
                "theta_loso": x["thresholds"]["loso"]["theta"],
                "delta_theta": x["thresholds"]["loso"]["theta"] - x["thresholds"]["self"]["theta"],
                "self": x["metrics"]["self"],
                "loso": x["metrics"]["loso"],
            } for x in cases
        ]
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p0_batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
