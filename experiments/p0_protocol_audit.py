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
import platform
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, asdict
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
    augmentation_seed: int
    bank_seed: int
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


def deterministic_subseed(seed: int, *identity: object) -> int:
    """Independent deterministic seed; independent of support order and support draw seed."""
    key = '|'.join([str(seed), *(str(v) for v in identity)])
    return int(hashlib.sha256(key.encode('utf-8')).hexdigest()[:8], 16)


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
        image_seed = deterministic_subseed(cfg.augmentation_seed, path)
        random.seed(image_seed)
        np.random.seed(image_seed % (2**32 - 1))
        torch.manual_seed(image_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(image_seed)
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
    refs = []
    for path in chosen:
        for view_idx, featset in enumerate(cache[path]):
            slim = []
            for layer_idx, x in enumerate(featset):
                k = min(cfg.fixed_per_image, int(x.shape[0]))
                # Shared support references are subsampled identically in every
                # LOSO fold, irrespective of which other support is withheld.
                local_rng = np.random.RandomState(deterministic_subseed(
                    cfg.bank_seed, path, view_idx, layer_idx
                ))
                idx = local_rng.choice(int(x.shape[0]), size=k, replace=False)
                slim.append(x[idx].to(cfg.device))
            refs.append(slim)
    bank.build_fixed_normal_from_support(
        cls_name="p0", support_feats=refs, mode=0,
        view="id", normal_seed_patches=8192, seed=cfg.bank_seed,
    )
    if not bank.normal_fixed["p0"][0]["id"][0].size():
        raise RuntimeError("Fixed normal bank is empty.")
    return bank


def fixed_bank_profile(bank) -> dict[str, Any]:
    """Actual stored reference count and fingerprint (not configured capacity)."""
    layers = bank.normal_fixed["p0"][0]["id"]
    digest = hashlib.sha256()
    info = []
    for i, cb in enumerate(layers):
        x = cb.protos.detach().to("cpu").contiguous().numpy()
        digest.update(x.tobytes())
        info.append({
            "layer": i, "n_prototypes": int(x.shape[0]),
            "feature_dim": int(x.shape[1]), "capacity": int(cb.K),
            "dtype": str(x.dtype), "bytes": int(x.nbytes)
        })
    return {
        "layers": info, "sha256": digest.hexdigest(),
        "total_prototypes": sum(t["n_prototypes"] for t in info),
        "total_feature_bytes": sum(t["bytes"] for t in info),
    }


def diagnostic_threshold_curve(y: list[int], scores: list[float],
                               named_thresholds: dict[str, float]):
    """Retrospective use of test labels ONLY for *plots*, never calibration."""
    labels = np.asarray(y, dtype=np.int8)
    xs = np.asarray(scores, dtype=np.float64)
    if not len(xs) or not np.isfinite(xs).all():
        raise ValueError("Invalid evaluation scores")
    normals = np.sort(xs[labels == 0])
    defects = np.sort(xs[labels == 1])
    if not len(normals) or not len(defects):
        raise ValueError("Need both classes for FPR/FNR diagnostic")
    thresholds = np.unique(np.concatenate((
        np.nextafter([xs.min()], -np.inf), xs,
        np.asarray(list(named_thresholds.values()), dtype=np.float64),
        np.nextafter([xs.max()], np.inf),
    )))
    n0, n1 = len(normals), len(defects)
    # Predictions use score > theta; searchsorted(..., right) matches equality.
    fpr = (n0 - np.searchsorted(normals, thresholds, side="right")) / n0
    fnr = np.searchsorted(defects, thresholds, side="right") / n1
    out = []
    for j, th in enumerate(thresholds):
        out.append({
            "threshold": float(th), "fpr": float(fpr[j]),
            "fnr": float(fnr[j]),
            "tpr": float(1.0 - fnr[j]),
            "balanced_accuracy": float(1.0 - 0.5 * (fpr[j] + fnr[j])),
            "is_self": int(th == named_thresholds.get("self")),
            "is_loso": int(th == named_thresholds.get("loso")),
            "is_commissioning": int(th == named_thresholds.get("commissioning")),
        })
    return out


def save_diagnostics(out: Path, prefix: str, y: list[int],
                     scores: list[float], named_thresholds: dict[str, float],
                     *, plot: bool = True) -> dict[str, Any]:
    curve = diagnostic_threshold_curve(y, scores, named_thresholds)
    with (out / f"{prefix}_threshold_curve.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    normal = np.asarray([s for v, s in zip(y, scores) if v == 0], dtype=float)
    defect = np.asarray([s for v, s in zip(y, scores) if v == 1], dtype=float)
    audit = {
        "score_quantiles_normal": {
            str(q): float(np.quantile(normal, q)) for q in (0, .05, .25, .5, .75, .95, 1)
        },
        "score_quantiles_defect": {
            str(q): float(np.quantile(defect, q)) for q in (0, .05, .25, .5, .75, .95, 1)
        },
        "n_curve_rows": len(curve),
        "disclaimer": "Retrospective test-label diagnostics; no test-label threshold selection.",
    }
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = np.array([r["threshold"] for r in curve])
        fp = np.array([r["fpr"] for r in curve])
        fn = np.array([r["fnr"] for r in curve])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, fp, label="FPR")
        ax.plot(xs, fn, label="FNR")
        for name, val in named_thresholds.items():
            ax.axvline(val, linestyle="--", linewidth=1, label=f"{name}: {val:.4f}")
        ax.set(xlabel="Frozen anomaly-score threshold", ylabel="Error rate",
               title="Retrospective FPR/FNR vs threshold (NOT training)")
        ax.set_ylim(-.02, 1.02)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{prefix}_threshold_sweep.png", dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fp[::-1], 1.0 - fn[::-1], label="ROC, frozen scores")
        ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1)
        for name, val in named_thresholds.items():
            close = int(np.argmin(np.abs(xs - val)))
            ax.scatter([fp[close]], [1.0 - fn[close]], s=36,
                       label=f"{name} ({fp[close]:.3f}, {1-fn[close]:.3f})")
        ax.set(xlabel="False positive rate", ylabel="True positive rate",
               title="Frozen-bank ROC with initial operating points")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{prefix}_roc.png", dpi=140)
        plt.close(fig)
    return audit


def encode_query(impl, backbone, path: str, cfg: Config):
    """Separate GPU feature extraction from bank retrieval for replay attribution."""
    img = impl.load_rgb(path)
    feat, _ = impl.encode_pil(backbone, img, cfg.feature_fuse)
    return img, feat


def score_encoded(impl, bank, img, feat, cfg: Config) -> float:
    """Score the *same* frozen feature tensors against a provided bank."""
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


def score_image(impl, backbone, bank, path: str, cfg: Config) -> float:
    img, feat = encode_query(impl, backbone, path, cfg)
    return score_encoded(impl, bank, img, feat, cfg)


def compare_repeated_case_csv(first: Path, second: Path, atol: float) -> dict[str, Any]:
    """Compare matched per-image predictions without demanding bitwise GPU identity."""
    def load(path):
        with path.open('r', encoding='utf-8', newline='') as fh:
            result = list(csv.DictReader(fh))
        return [(row['path'], int(row['label']), float(row['score'])) for row in result]
    lhs, rhs = load(first), load(second)
    if len(lhs) != len(rhs):
        raise AssertionError(f'Score CSV row-count mismatch: {len(lhs)} vs {len(rhs)}')
    if [(p, y) for p, y, _ in lhs] != [(p, y) for p, y, _ in rhs]:
        raise AssertionError('Frozen evaluation stream paths/labels differ')
    diffs = np.abs(np.asarray([z for _, _, z in lhs]) - np.asarray([z for _, _, z in rhs]))
    if not np.isfinite(diffs).all():
        raise AssertionError('Nonfinite replay differences')
    maximum = float(diffs.max()) if len(diffs) else 0.0
    return {
        'n': len(diffs), 'max_abs_delta': maximum,
        'mean_abs_delta': float(diffs.mean()) if len(diffs) else 0.0,
        'n_above_1e-6': int((diffs > 1e-6).sum()),
        'n_above_atol': int((diffs > atol).sum()),
        'tolerance': atol, 'within_tolerance': maximum <= atol,
    }


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
    cache = encode_reference_features(impl, backbone, chosen, cfg)
    bank = build_vmb(impl, cache, chosen, cfg)
    bank_audit = fixed_bank_profile(bank)
    self_scores = [score_image(impl, backbone, bank, x, cfg) for x in chosen]
    self_stats = threshold_from_scores(self_scores, cfg.multiplier)

    loso_scores = []
    loso_bank_audits = []
    for heldout in chosen:
        remaining = [x for x in chosen if x != heldout]
        assert len(remaining) == args.shots - 1 and heldout not in remaining
        fold_bank = build_vmb(impl, cache, remaining, cfg)
        loso_scores.append(score_image(impl, backbone, fold_bank, heldout, cfg))
        loso_bank_audits.append({"heldout": heldout, **fixed_bank_profile(fold_bank)})
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
        cal_paths = sorted(random.Random(args.commissioning_seed).sample(
            candidates, args.commissioning_size
        ))
        cal_scores = [score_image(impl, backbone, bank, x, cfg) for x in cal_paths]
        cal_stats = threshold_from_scores(cal_scores, cfg.multiplier)

    rng = random.Random(args.stream_seed)
    rng.shuffle(tests)
    if args.max_test:
        tests = tests[:args.max_test]
    repeatability = None
    if args.verify_replay:
        replay_bank = build_vmb(impl, cache, chosen, cfg)
        second = fixed_bank_profile(replay_bank)
        if second != bank_audit:
            raise AssertionError("Fixed bank changed on rebuild with identical inputs/seeds")
        query_image, query_feats = encode_query(impl, backbone, tests[0].path, cfg)
        initial_score = score_encoded(impl, bank, query_image, query_feats, cfg)
        replay_score = score_encoded(impl, replay_bank, query_image, query_feats, cfg)
        bank_scoring_delta = abs(initial_score - replay_score)
        if bank_scoring_delta > args.replay_tolerance:
            raise AssertionError(
                f'Bank scoring mismatch on IDENTICAL encoded features: '
                f'{initial_score} vs {replay_score}, delta={bank_scoring_delta:.3g}. '
                'This is a scoring-level difference, not a re-encoding difference.'
            )
        # Independent forward pass: CUDA memory-efficient SDPA is not guaranteed
        # bitwise reproducible. Track the drift without conflating it with bank.
        independent_score = score_image(impl, backbone, bank, tests[0].path, cfg)
        encoder_delta = abs(initial_score - independent_score)
        repeatability = {
            'bank_fingerprint_equal': True,
            'same_encoded_features_score_bank_1': initial_score,
            'same_encoded_features_score_bank_2': replay_score,
            'fixed_feature_bank_score_delta': bank_scoring_delta,
            'fixed_feature_bank_score_tolerance': args.replay_tolerance,
            'independent_forward_score': independent_score,
            'reencoding_score_delta': encoder_delta,
            'reencoding_tolerance': args.inference_tolerance,
            'reencoding_within_tolerance': encoder_delta <= args.inference_tolerance,
        }
        if encoder_delta > args.inference_tolerance:
            raise AssertionError(
                f'Independent GPU re-encoding drift {encoder_delta:.3g} exceeds '
                f'--inference-tolerance={args.inference_tolerance}. '
                'Try --math-sdpa or investigate GPU nondeterminism.'
            )
        print(f'  replay: bank_hash=equal fixed_feature_delta={bank_scoring_delta:.3g}'
              f' reencoding_delta={encoder_delta:.3g}', flush=True)
        del replay_bank
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
    out.mkdir(parents=True, exist_ok=True)
    prefix = (f"{args.class_name}_{protocol}_support{seed}"
              f"_aug{cfg.augmentation_seed}_bank{cfg.bank_seed}")
    named_thresholds = {name: stat["theta"] for name, stat in (
        [("self", self_stats), ("loso", loso_stats)] +
        ([("commissioning", cal_stats)] if cal_stats is not None else [])
    )}
    diag = save_diagnostics(out, prefix, y, scores, named_thresholds,
                            plot=not args.no_plots)
    score_digest = hashlib.sha256(np.asarray(scores, dtype="<f8").tobytes()).hexdigest()
    meta = {
        "schema": "p0.protocol.audit.v2",
        "case_id": prefix,
        "seeds": {"support": seed, "augmentation": cfg.augmentation_seed,
                  "bank": cfg.bank_seed, "stream": args.stream_seed,
                  "commissioning": args.commissioning_seed},
        "repeatability": repeatability,
        "bank_profile": bank_audit,
        "loso_bank_profiles": loso_bank_audits,
        "score_sha256": score_digest,
        "diagnostics": diag,
        "backbone": {"model_name": getattr(backbone, "model_name", None),
                     "img_size": getattr(backbone, "img_size", None),
                     "resize_size": getattr(backbone, "resize_size", None),
                     "layers": getattr(backbone, "layers", None),
                     "use_fp16": getattr(backbone, "use_fp16", None)},
        "effective_config": asdict(cfg),
        "runtime": {"python": platform.python_version(),
                    "numpy": np.__version__, "torch": torch.__version__,
                    "sdpa_math_only": args.math_sdpa,
                    "cuda_math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()) if torch.cuda.is_available() else None,
                    "cuda_flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()) if torch.cuda.is_available() else None,
                    "cuda_mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()) if torch.cuda.is_available() else None},
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
            "Supports are fixed per support_seed; augmentation and bank subsampling are fixed independently.",
            "Bitwise-identical GPU scores are NOT guaranteed with nondeterministic attention; fixed-feature bank retrieval is verified independently.",
            "LOSO folds still have only three original images and resample the final v22 bank pool; LOSO is diagnostic only.",
            "Any threshold sweep/ROC uses held-out test labels retrospectively, NEVER to choose deployment thresholds.",
        ],
    }
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
    p.add_argument("--augmentation-seed", type=int, default=0, help="Fixed across different support draws")
    p.add_argument("--bank-seed", type=int, default=0, help="Fixed across different support draws and folds")
    p.add_argument("--commissioning-seed", type=int, default=100003)
    p.add_argument("--verify-replay", action="store_true", help="Rebuild frozen bank and verify matching scores")
    p.add_argument("--replay-tolerance", type=float, default=1e-6,
                   help="Bank scoring tolerance using the IDENTICAL cached query features")
    p.add_argument("--inference-tolerance", type=float, default=1e-4,
                   help="Absolute tolerance for independent GPU feature forward passes")
    p.add_argument("--math-sdpa", action="store_true",
                   help="Disable flash/memory-efficient CUDA SDPA in this P0 process for reproducibility (slower)")
    p.add_argument("--no-plots", action="store_true", help="Still write threshold curve CSV")
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
        curve = diagnostic_threshold_curve([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], {"self": 0.5, "loso": 0.7})
        assert any(abs(z["threshold"] - 0.5) < 1e-9 and z["fpr"] == 0 and z["fnr"] == 0 for z in curve)
        assert deterministic_subseed(42, "imageA", 0) == deterministic_subseed(42, "imageA", 0)
        assert deterministic_subseed(42, "imageA", 0) != deterministic_subseed(43, "imageA", 0)
        with __import__('tempfile').TemporaryDirectory() as temp:
            a, b = Path(temp) / 'a.csv', Path(temp) / 'b.csv'
            for f, delta in [(a, 0.0), (b, 0.000011206)]:
                with f.open('w', newline='', encoding='utf-8') as fh:
                    w = csv.DictWriter(fh, fieldnames=['path','label','score'])
                    w.writeheader()
                    w.writerow({'path':'x', 'label':0, 'score':0.528495 + delta})
            c = compare_repeated_case_csv(a, b, 1e-4)
            assert c['within_tolerance'] and c['max_abs_delta'] > 1e-5
            assert not compare_repeated_case_csv(a, b, 1e-6)['within_tolerance']
        print('P0-R2 numerical/reproducibility self-test: PASS')
        return
    if not args.dataset_root:
        raise SystemExit("--dataset-root is required unless --self-test.")
    if args.commissioning_size < 0:
        raise SystemExit("--commissioning-size must be nonnegative.")
    if args.replay_tolerance < 0 or args.inference_tolerance < 0:
        raise SystemExit('Replay tolerances must be nonnegative.')
    import torch
    if args.math_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print('[P0-R2] CUDA SDPA math-only kernel enabled (slower; monitor memory).', flush=True)
    impl = require_sources()
    json_dir = args.json_dir or str(Path(args.dataset_root) / "jsons" / "realiad_jsons")
    dataset = impl.RealIADJSONIndex(root=args.dataset_root, json_dir=json_dir)
    protocols = (["strict4", "pool_assisted4"] if args.support_policy == "both"
                 else [args.support_policy])
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("No support seeds.")
    cfg = Config(
        device=args.device, augmentation_seed=args.augmentation_seed,
        bank_seed=args.bank_seed, fixed_capacity=args.fixed_capacity,
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
        "schema": "p0.protocol.batch.v2",
        "fixed_augmentation_seed": cfg.augmentation_seed,
        "fixed_bank_seed": cfg.bank_seed,
        "effective_config": asdict(cfg),
        "cases": [
            {
                "protocol": x["protocol"],
                "seed": x["support_seed"],
                "case_id": x["case_id"],
                "bank_profile": x["bank_profile"],
                "score_sha256": x["score_sha256"],
                "repeatability": x["repeatability"],
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
    support_groups = {}
    for row in summary["cases"]:
        key = (row["protocol"], tuple(row["support_paths"]))
        support_groups.setdefault(key, []).append(row)
    summary["unique_support_sets_per_protocol"] = {
        protocol: len({tuple(row["support_paths"]) for row in summary["cases"] if row["protocol"] == protocol})
        for protocol in protocols
    }
    summary['identical_support_replay'] = []
    for (proto, _), group in support_groups.items():
        if len(group) <= 1:
            continue
        base = group[0]
        checks = [compare_repeated_case_csv(
            out / f'{base["case_id"]}_scores.csv',
            out / f'{other["case_id"]}_scores.csv',
            args.inference_tolerance
        ) for other in group[1:]]
        entry = {
            'protocol': proto, 'seeds': [v['seed'] for v in group],
            'identical_bank': len({v['bank_profile']['sha256'] for v in group}) == 1,
            'bitwise_identical_scores': len({v['score_sha256'] for v in group}) == 1,
            'numeric_score_comparisons': checks,
            'within_inference_tolerance': all(c['within_tolerance'] for c in checks),
        }
        summary['identical_support_replay'].append(entry)
        if not entry['identical_bank']:
            raise AssertionError(f'Identical supports produced DIFFERENT fixed-bank data: {entry}')
        if not entry['within_inference_tolerance']:
            raise AssertionError(
                f'Identical supports produced numerically divergent GPU scores: {entry}. '
                'Try --math-sdpa or diagnose the encoder/scorer.'
            )
    if len(seeds) > 1 and summary["unique_support_sets_per_protocol"].get("pool_assisted4") == 1:
        print("NOTE: pool-assisted central selection chose identical four supports for all seeds; not independent support draws.")
    (out / "p0_batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
