"""D2-A fixed-threshold registry: leakage-resistant developmental calibration.

Only reserved 64 NORMAL *training* history views, one per official sample,
may calibrate thresholds. The original four-shot support, mature history,
online phases, and Probe A/B are non-overlapping at the official sample level.
This module never uses probe-derived scores for fitting or model selection.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from . import preflight as pf

SCHEMA = "risk_safe.d2a.fixed_normal_threshold_registry.v1"
QUANTILES = (0.90, 0.95)
N_HISTORY = 64
REGISTRY = "threshold_registry.json"
MANIFEST = "source_manifest.json"


class ThresholdAuditError(RuntimeError):
    pass


def canonical_quantile_name(q: float) -> str:
    if q not in QUANTILES:
        raise ThresholdAuditError(
            f"Unregistered quantile {q}; fixed development grid is {QUANTILES}")
    return f"{q:.2f}"


def fit_fixed_normal_thresholds(scores: Sequence[float]) -> dict:
    """Higher empirical quantile; explicit deterministic finite-sample rule."""
    xs = np.asarray(scores, dtype=np.float64)
    if xs.shape != (N_HISTORY,) or not np.isfinite(xs).all():
        raise ThresholdAuditError("Require exactly 64 finite reserved-normal scores")
    ordered = np.sort(xs)
    std = float(np.std(xs, ddof=1))
    if not math.isfinite(std):
        raise ThresholdAuditError("Nonfinite calibration scale")
    return {
        canonical_quantile_name(q): {
            "theta": float(ordered[math.ceil((len(xs) - 1) * q)]),
            "rank_zero_based": math.ceil((len(xs) - 1) * q),
            "quantile": q,
            "rule": "higher_order_statistic_of_64_disjoint_train_normal_views",
        } for q in QUANTILES
    }


def validate_separation(plan, selected_support_keys: Sequence[str]) -> list:
    """Only the fixed mature-history partition, not stream/probes/support."""
    if len(plan.history) != N_HISTORY:
        raise ThresholdAuditError("Expected exactly 64 reserved history images")
    ids = [r.sample_key for r in plan.history]
    if len(set(ids)) != N_HISTORY or any(not r.is_good for r in plan.history):
        raise ThresholdAuditError("History calibration must have 64 unique verified normals")
    other = set(selected_support_keys)
    partitions = plan.sample_sets()
    for name in ("support", "stream", "probe_A", "probe_B"):
        other.update(partitions[name])
    if other.intersection(ids):
        raise ThresholdAuditError(
            "Normal threshold calibration overlaps support/stream/probe")
    return sorted(plan.history, key=lambda r: (r.sample_key, r.camera_id, r.path))


def validate_registry(registry: Mapping, *, plan, class_name: str,
                      support_seed: int, class_json_path: Path,
                      registry_path: Path, strict_manifest: bool = True) -> dict:
    if registry.get("schema") != SCHEMA:
        raise ThresholdAuditError("Unknown fixed threshold registry schema")
    if registry.get("class_name") != class_name or registry.get("support_seed") != support_seed:
        raise ThresholdAuditError("Class/support-seed threshold mismatch")
    if registry.get("normal_calibration_partition") != "reserved_mature_history_training_normal":
        raise ThresholdAuditError("Unapproved calibration partition")
    expected_support = sorted(plan.support_samples_by_seed[support_seed])
    if registry.get("support_keys") != expected_support:
        raise ThresholdAuditError("Strict-four-shot support keys mismatch")
    history = validate_separation(plan, expected_support)
    if registry.get("calibration_sample_keys") != [r.sample_key for r in history]:
        raise ThresholdAuditError("Calibration history keys or ordering mismatch")
    if registry.get("calibration_paths") != [r.rel_path for r in history]:
        raise ThresholdAuditError("Calibration camera identities mismatch")
    if registry.get("realiad_class_json_sha256") != pf.sha256_file(class_json_path):
        raise ThresholdAuditError("Real-IAD index changed after calibration")
    if registry.get("probes_used_for_calibration") is not False:
        raise ThresholdAuditError("Registry did not assert strict probe exclusion")
    if registry.get("online_stream_used_for_calibration") is not False:
        raise ThresholdAuditError("Registry did not assert stream exclusion")
    scores = registry.get("calibration_normal_scores")
    fitted = fit_fixed_normal_thresholds(scores)
    if registry.get("registered_quantiles") != fitted:
        raise ThresholdAuditError("Recorded thresholds do not match normal-only scores")
    sigma = float(np.std(np.asarray(scores, dtype=np.float64), ddof=1))
    if not math.isclose(float(registry.get("sigma_dev_normal")), sigma,
                        abs_tol=1e-12, rel_tol=1e-12):
        raise ThresholdAuditError("Normal-score scale not reproducible")
    if strict_manifest:
        folder = registry_path.parent
        manifest_file = folder / MANIFEST
        if not manifest_file.is_file():
            raise ThresholdAuditError("Frozen calibration source manifest missing")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if (manifest.get("schema") != "risk_safe.d2a.fixed_threshold_source.v1"
                or manifest.get("output_sha256", {}).get(REGISTRY) !=
                pf.sha256_file(registry_path)
                or not manifest.get("source_sha256")
                or manifest.get("input_sha256", {}).get("realiad_class_json") !=
                pf.sha256_file(class_json_path)):
            raise ThresholdAuditError("Calibration file/source integrity mismatch")
    return {"theta_by_quantile": fitted, "sigma_dev_normal": sigma,
            "n_training_normal_views": len(scores),
            "calibration_sha256": pf.sha256_file(registry_path)}
