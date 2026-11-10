"""Frozen static reference-baseline schedule and source-bound K-Center call."""
from __future__ import annotations

import math
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
