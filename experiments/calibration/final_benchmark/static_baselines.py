"""Frozen static reference-baseline schedule and source-bound K-Center call."""
from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


class StaticBaselineError(RuntimeError):
    pass


FIXED_PROTOTYPES = 8192
FEATURE_DIMENSION = 4096
FIXED_DTYPE = "float16"
FIXED_BYTES = FIXED_PROTOTYPES * FEATURE_DIMENSION * 2
REFERENCE_IMAGES = 36
C32_IMAGES = 32


def ensure_historical_memory_import_path() -> None:
    """Resolve the unchanged v02/tmra modules from the repository's experiments directory."""
    source_dir = str(Path(__file__).resolve().parents[2])
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)


def static36_threshold_c32(scores: Sequence[float]) -> float:
    """Use the already verified A4 higher empirical q90 order statistic."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) != C32_IMAGES or not np.isfinite(values).all():
        raise StaticBaselineError("Static-36+32 threshold requires exactly 32 finite C32 scores")
    index = int(math.ceil(0.90 * (len(values) - 1)))
    return float(np.partition(values, index)[index])


def select_static36_bank(candidates: Any, candidate_identity_hashes: Sequence[str],
                         *, expected_feature_dim: int = FEATURE_DIMENSION) -> tuple[Any, list[int], dict[str, Any]]:
    """Call the repository's frozen A3b CUDA K-Center implementation.

    This adapter does not implement a competing coreset algorithm. It is not
    invoked in Gate 1 because the authorization prohibits real-data GPU work.
    """
    import numpy as np
    matrix = np.asarray(candidates)
    if matrix.ndim != 2 or matrix.shape[1] != expected_feature_dim:
        raise StaticBaselineError("K-Center candidate matrix does not match the frozen 4096D feature contract")
    if len(matrix) < FIXED_PROTOTYPES or len(candidate_identity_hashes) != len(matrix):
        raise StaticBaselineError("K-Center candidate identities/count are incomplete")
    try:
        from experiments.calibration import a_calibration_continuation as a3
        prototypes, selected = a3.a3b_kcenter_cuda(
            matrix, candidate_identity_hashes, FIXED_PROTOTYPES,
            return_indices=True)
    except Exception as exc:
        raise StaticBaselineError(
            f"source-bound A3b K-Center could not execute ({type(exc).__name__}); no alternate algorithm is used") from None
    if tuple(prototypes.shape) != (FIXED_PROTOTYPES, FEATURE_DIMENSION):
        raise StaticBaselineError("A3b returned a prototype bank with the wrong shape")
    if str(prototypes.dtype).lower() not in {"torch.float16", "float16"}:
        raise StaticBaselineError("A3b prototype bank is not FP16")
    bank_bytes = int(prototypes.numel() * prototypes.element_size())
    if bank_bytes != FIXED_BYTES:
        raise StaticBaselineError("A3b prototype bank does not realize the registered 64 MiB target")
    return prototypes, [int(x) for x in selected], {
        "algorithm_source": "experiments.calibration.a_calibration_continuation.a3b_kcenter_cuda",
        "algorithm": "greedy max-min cosine; candidates sorted by identity hash; first-index tie break",
        "prototype_count": FIXED_PROTOTYPES, "feature_dim": FEATURE_DIMENSION,
        "dtype": "float16", "bank_bytes": bank_bytes,
        "status": "SOURCE_BOUND_ADAPTER; CUDA_NUMERICAL_PARITY_PENDING_GATE2",
    }


class FrozenNativeStatic:
    """Native fixed-bank scorer whose calibration threshold never changes."""

    def __init__(self, *, impl: Any, bank: Any, cfg: Any, threshold: float,
                 fixed_sha256: str):
        if not math.isfinite(float(threshold)):
            raise StaticBaselineError("static threshold is not finite")
        self._impl, self._bank, self._cfg = impl, bank, cfg
        self.threshold = float(threshold)
        self._digest = hashlib.sha256(json.dumps(
            {"fixed_bank_sha256": fixed_sha256, "threshold": self.threshold},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()

    def state_digest(self) -> str:
        return self._digest

    def score(self, encoded: Any):
        from experiments.calibration import r1_matched_correction_pilot as r1
        from .runner import ScorePacket
        _base_map, _top_ids, score, _summary = r1._native_dense_score(
            self._impl, self._bank, encoded.image, encoded.feature_layers, self._cfg)
        return ScorePacket(float(score), int(float(score) >= self.threshold), 0.0, 0.0,
                           0, self._digest, threshold=self.threshold)


def build_static4_native(*, category: str, impl: Any, p0: Any, r1: Any,
                         support_cache: Any, support_paths: Sequence[str], cfg: Any,
                         c64_items: Sequence[Any], expected_fixed_sha256: str) -> FrozenNativeStatic:
    """Build the native four-Support scorer and calibrate it once on C64."""
    from experiments.calibration import a4_equal_budget_allocation as a4
    from experiments.calibration import r7_shuffled_correction_4class as r7
    ensure_historical_memory_import_path()
    if len(support_paths) != 4 or len(set(support_paths)) != 4 or len(c64_items) != 64:
        raise StaticBaselineError("Static-4+64 requires four Supports and 64 C64 items")
    trajectory = a4._build_bank("STATIC-4+64", category=category, impl=impl, p0=p0,
                                r1=r1, support_cache=support_cache,
                                support_paths=list(support_paths), cfg=cfg,
                                expected_sha=expected_fixed_sha256)
    scores, _cache, _elapsed = r7._score_c64_base(impl, trajectory.bank, c64_items,
                                                  cfg, r1, category)
    if len(scores) != 64 or not np.isfinite(scores).all():
        raise StaticBaselineError("Static-4+64 commissioning scores are incomplete")
    threshold = float(np.quantile(np.asarray(scores, dtype=np.float64), .90, method="higher"))
    return FrozenNativeStatic(impl=impl, bank=trajectory.bank, cfg=cfg,
                              threshold=threshold, fixed_sha256=trajectory.fixed_sha256)


def build_static36_native(*, category: str, schedule: Any, reference_items: Sequence[Any],
                          c32_items: Sequence[Any], impl: Any, p0: Any, r1: Any,
                          backbone: Any, cfg: Any, support_cache: Any,
                          support_paths: Sequence[str]) -> tuple[FrozenNativeStatic, dict[str, Any]]:
    """Build the P0-only A4/A3b reference and frozen C32 threshold."""
    from experiments.calibration import a4_equal_budget_allocation as a4
    from experiments.calibration import r7_shuffled_correction_4class as r7
    ensure_historical_memory_import_path()
    schedule_payload = {key: value for key, value in schedule.items() if key != "schedule_sha256"}
    schedule_digest = hashlib.sha256(json.dumps(
        schedule_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()
    if (schedule.get("schedule_sha256") != schedule_digest or
            schedule.get("dataset_id") != "realiad" or
            schedule.get("category") != category or schedule.get("support_seed") != 0 or
            schedule.get("stream_seed") != 0 or len(reference_items) != 36 or
            len(c32_items) != 32 or len(support_paths) != 4):
        raise StaticBaselineError("Static-36+32 requires its registered P0 36/32 schedule")
    ref_ids = [str(item["image_identity_sha256"]) for item in reference_items]
    ref_groups = [str(item["group_sha256"]) for item in reference_items]
    c32_ids = [str(item["image_identity_sha256"]) for item in c32_items]
    c32_groups = [str(item["source_group_sha256"]) for item in c32_items]
    if (ref_ids != list(schedule.get("reference_identity_sha256", ())) or
            ref_groups != list(schedule.get("reference_group_sha256", ())) or
            c32_ids != list(schedule.get("c32_identity_sha256", ())) or
            c32_groups != list(schedule.get("c32_group_sha256", ())) or
            [str(item["image_path"]) for item in reference_items[:4]] != list(support_paths) or
            len(set(ref_ids + c32_ids)) != 68 or len(set(ref_groups + c32_groups)) != 68 or
            set(ref_ids[4:] + c32_ids) != set(schedule.get("principal_c64_identity_sha256", ()))):
        raise StaticBaselineError("Static-36+32 identities do not match the disjoint P0 schedule")
    candidates, candidate_ids, receipt = a4._kcenter_candidates(
        category, reference_items, support_cache, p0, impl, backbone, cfg, cap=a4.KC_CAP)
    prototypes, selected, coreset_receipt = select_static36_bank(candidates, candidate_ids)
    trajectory = a4._build_bank("STATIC-36+32", category=category, impl=impl, p0=p0,
                                r1=r1, support_cache=support_cache,
                                support_paths=list(support_paths), cfg=cfg,
                                expected_sha=None, prototypes=prototypes)
    scores, _cache, _elapsed = r7._score_c64_base(impl, trajectory.bank, c32_items,
                                                  cfg, r1, category)
    threshold = static36_threshold_c32(scores)
    return FrozenNativeStatic(impl=impl, bank=trajectory.bank, cfg=cfg,
                              threshold=threshold, fixed_sha256=trajectory.fixed_sha256), {
        "schedule_sha256": schedule.get("schedule_sha256"),
        "candidate_receipt": receipt, "coreset_receipt": coreset_receipt,
        "selected_count": len(selected), "threshold": threshold,
    }
