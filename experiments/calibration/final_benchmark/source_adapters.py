"""Thin calls into frozen R7/R9/R10/E6 sources for a Gate 2 native backend."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def full_c1_score(*, impl: Any, bank: Any, r1: Any, exp2a: Any, cfg: Any,
                  codebook: Any, source_groups: Mapping[int, set[str]],
                  selected: Any, group_sha256: str, position: int,
                  base_map: Any, top_ids: Any) -> tuple[float, dict[str, Any]]:
    from experiments.calibration import r9_correction_mechanism_falsification as r9
    return r9._score_codebook(impl, bank, r1, exp2a, cfg, codebook,
                              source_groups, selected, group_sha256, position,
                              base_map, top_ids)


def native_common_score(*, impl: Any, bank: Any, image: Any,
                        feature_layers: Sequence[Any], cfg: Any):
    """Reuse the R1/R7 common native C0 dense map and Top-32 contract."""
    from experiments.calibration import r1_matched_correction_pilot as r1
    return r1._native_dense_score(impl, bank, image, feature_layers, cfg)


def full_c1_verified_feedback(*, impl: Any, bank: Any, r1: Any, exp2a: Any,
                              arm_id: str, cfg: Any, codebook: Any,
                              feature_layers: Sequence[Any], top_ids: Any,
                              score: float, threshold: float, prediction: int,
                              verified_label: int, position: int,
                              source_groups: dict[int, set[str]],
                              group_sha256: str) -> Any:
    """Apply the source-compatible signed R9 C1 update after verified query feedback."""
    if (isinstance(verified_label, bool) or isinstance(prediction, bool) or
            int(verified_label) not in (0, 1) or int(prediction) not in (0, 1)):
        raise ValueError("Full C1 feedback requires verified binary label and prediction")
    outcome = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[
        (int(verified_label), int(prediction))]
    return exp2a._capture_update(
        impl, bank, r1, arm_id, cfg, codebook, feature_layers, top_ids,
        1 if int(verified_label) == 1 else -1, float(score), float(threshold),
        outcome, int(position), source_groups, group_sha256)


def nshadow_normal_feedback(*, impl: Any, bank: Any, r1: Any, exp2a: Any,
                            cfg: Any, codebook: Any, feature_layers: Sequence[Any],
                            top_ids: Any, score: float, threshold: float,
                            prediction: int, verified_label: int,
                            position: int, source_groups: dict[int, set[str]],
                            group_sha256: str) -> Any | None:
    """N-shadow has an initially empty negative-only C1 update and no defect write."""
    if isinstance(verified_label, bool) or int(verified_label) not in (0, 1):
        raise ValueError("N-shadow requires verified binary feedback")
    if int(verified_label) == 1:
        return None
    return full_c1_verified_feedback(
        impl=impl, bank=bank, r1=r1, exp2a=exp2a, arm_id="N-SHADOW",
        cfg=cfg, codebook=codebook, feature_layers=feature_layers,
        top_ids=top_ids, score=score, threshold=threshold,
        prediction=prediction, verified_label=0, position=position,
        source_groups=source_groups, group_sha256=group_sha256)


def normal_memory_feedback(*, position: int, score: float, threshold: float,
                           sigma: float, prediction: int, candidate_ids: Any,
                           candidate_features: Any, feature_layers: Sequence[Any],
                           queried: bool, verified_label: int | None,
                           eligibility: Any, update: Any) -> dict[str, Any]:
    """Apply R1's common C0/verified-normal writer contract to one arm's bank."""
    from experiments.calibration import r1_matched_correction_pilot as r1
    return r1.common_c0_greedy_writer_action(
        position=position, c0_score=score, theta=threshold, sigma=sigma,
        c0_prediction=prediction, candidate_ids=candidate_ids,
        candidate_features=candidate_features, feature_layers=feature_layers,
        queried=queried, disclosed_label=verified_label,
        eligibility=eligibility, update=update)


def signed_knn_score(*, base_map: Any, top_ids: Any, selected_features: Any,
                     memory: Any, position: int,
                     group_sha256: str) -> tuple[float, Any, dict[str, Any]]:
    from experiments.calibration import r10_signed_knn_baseline as r10
    return r10._corrected_c2_score(
        base_map, top_ids, selected_features, memory,
        position=position, group_sha256=group_sha256)


def signed_knn_feedback(*, memory: Any, selected_features: Any,
                        source_group_sha256: str, position: int,
                        verified_label: int) -> dict[str, int]:
    if isinstance(verified_label, bool) or int(verified_label) not in (0, 1):
        raise ValueError("R10 feedback requires a verified binary label")
    return memory.insert(selected_features, sign=(1 if int(verified_label) == 1 else -1),
                         source_group_sha256=source_group_sha256, position=int(position))


def b3_refresh(*, controller: Any, c64_scores: Sequence[float],
               group_sha256: str, previous_state_sha256: str,
               new_state_sha256: str, previous_version: int) -> dict[str, Any] | None:
    from experiments.calibration import r7_shuffled_correction_4class as r7
    return r7._refresh_b3(controller, c64_scores, group_sha256,
                          previous_state_sha256, new_state_sha256,
                          int(previous_version))


def bea_normal_band_objective(components: Any) -> float:
    """Reuse Experiment C's source-derived D_band and exact B+N+D_band rule."""
    from experiments.calibration.exp_c_memory_evidence import d_band_utility
    value = float(components.B) + float(components.N) + float(d_band_utility(components))
    if not __import__("math").isfinite(value):
        raise ValueError("BN+Band objective is not finite")
    return value


def higher_q90(scores: Sequence[float]) -> float:
    from experiments.calibration.a4_equal_budget_allocation import _higher_q90
    return float(_higher_q90(scores))
