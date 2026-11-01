"""Synthetic CPU tests for R2-EXP1 pair, movement and amplitude diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.calibration.r2_0_label_blind_mechanism_audit import (
    corrected_dense_map,
    max_score,
)
from experiments.calibration.r2_exp1_c1_mechanism_discovery import (
    AMPLITUDES,
    _grouped_query_stats,
    _pair_margin_decomposition,
    analyze_outcomes,
    correction_movement_summary,
    exact_pairwise_attribution,
    query_interval_ids,
    source_group_cluster_bootstrap,
    _sanitized_aggregate,
)


def test_pairwise_attribution_accounts_for_strict_pairs_and_ties_exactly():
    labels = np.array([1, 1, 0, 0], dtype=np.int8)
    baseline = np.array([0.5, 0.2, 0.5, 0.3])
    candidate = np.array([0.6, 0.5, 0.5, 0.5])

    result = exact_pairwise_attribution(labels, baseline, candidate)

    assert result["pair_count"] == 4
    assert result["baseline_strict_incorrect"] == 2
    assert result["baseline_strict_correct"] == 1
    assert result["baseline_ties"] == 1
    assert result["baseline_incorrect_to_tie"] == 2
    assert result["baseline_correct_to_tie"] == 0
    assert result["baseline_tie_resolved_correctly"] == 1
    assert result["strict_incorrect_to_correct_repairs"] == 0
    assert result["strict_correct_to_incorrect_damage"] == 0
    assert result["net_concordance_change"] == pytest.approx(1.5)
    assert result["auc_difference_from_pairs"] == pytest.approx(0.375)
    assert result["candidate_auc_from_pairs"] - result["baseline_auc_from_pairs"] == pytest.approx(
        result["auc_difference_from_pairs"]
    )


def test_pairwise_attribution_counts_strict_repair_and_damage():
    labels = np.array([1, 1, 0, 0], dtype=np.int8)
    baseline = np.array([0.8, 0.2, 0.5, 0.3])
    candidate = np.array([0.4, 0.9, 0.5, 0.3])

    result = exact_pairwise_attribution(labels, baseline, candidate)

    assert result["strict_incorrect_to_correct_repairs"] >= 1
    assert result["strict_correct_to_incorrect_damage"] >= 1
    assert result["net_concordance_change"] / result["pair_count"] == pytest.approx(
        result["candidate_auc_from_pairs"] - result["baseline_auc_from_pairs"]
    )


def test_movement_summary_separates_direction_from_threshold_confusion():
    labels = [1, 1, 0, 0]
    baseline = [0.49, 0.40, 0.52, 0.40]
    candidate = [0.55, 0.35, 0.45, 0.65]

    result = correction_movement_summary(labels, baseline, candidate, theta=0.5)

    assert result["outcome_direction"]["favorable"] == 2
    assert result["outcome_direction"]["harmful"] == 2
    assert result["prediction_changes"]["flips"] == 3
    assert result["prediction_changes"]["new_fp"] == 1
    assert result["prediction_changes"]["repaired_fp"] == 1
    assert result["prediction_changes"]["repaired_fn"] == 1
    assert set(result["image_score_delta_by_outcome"]) == {"anomalies", "normals"}
    assert "absolute_magnitude_quantiles" in result["image_score_delta"]


def test_c2_amplitude_recomputes_dense_map_max_instead_of_scaling_image_delta():
    base = np.array([0.8, 0.7, 0.4, 0.3], dtype=np.float32)
    ids = np.array([0, 1], dtype=np.int64)
    historical_delta = np.array([-0.5, 0.5], dtype=np.float32)

    historical = max_score(corrected_dense_map(base, ids, historical_delta))
    scaled_delta = historical_delta * np.float32(0.1 / 0.2)
    scaled = max_score(corrected_dense_map(base, ids, scaled_delta))

    assert historical == pytest.approx(1.2)
    assert scaled == pytest.approx(0.95)
    assert scaled - float(base.max()) != pytest.approx((historical - float(base.max())) * 0.5)
    assert np.array_equal(historical_delta * np.float32(0.2 / 0.2), historical_delta)


def test_query_event_belongs_to_pre_reveal_interval():
    ids = query_interval_ids([0, 1, 2, 3, 4], [1, 3])
    assert ids.tolist() == [0, 0, 1, 1, 2]


def test_source_group_bootstrap_is_deterministic_and_paired():
    labels = [0, 1, 0, 1]
    groups = ["g1", "g1", "g2", "g2"]
    scores = {"C0": [0.1, 0.8, 0.2, 0.7], "C1": [0.2, 0.8, 0.1, 0.9]}

    first = source_group_cluster_bootstrap(labels, groups, scores, replicates=30, seed=7)
    second = source_group_cluster_bootstrap(labels, groups, scores, replicates=30, seed=7)

    assert first == second
    assert first["source_group_count"] == 2
    assert first["valid_replicates"] == 30
    assert set(first["paired_auc_difference_vs_c0_ci_95"]) == {"C1"}


def test_margin_decomposition_preserves_all_cross_class_pairs():
    labels = np.array([1, 1, 0, 0], dtype=np.int8)
    baseline = np.array([0.6, 0.4, 0.55, 0.25])
    candidate = baseline + np.array([0.05, -0.03, -0.02, 0.04])

    rows = _pair_margin_decomposition(labels, baseline, candidate, "C1")

    assert sum(row["pair_count"] for row in rows) == 4
    assert sum(row["net_concordance_change"] for row in rows) == pytest.approx(
        exact_pairwise_attribution(labels, baseline, candidate)["net_concordance_change"]
    )


def test_query_admission_aggregation_uses_only_recorded_counts():
    rows = [
        {
            "queried_outcome": "anomaly",
            "c0_prediction": 1,
            "outcome": "TP",
            "candidates_examined": 32,
            "native_gate_kept": 4,
            "inserted": 2,
            "merged": 1,
            "replaced": 0,
            "skipped": 1,
            "c2_additions": 32,
            "c2_retained_after_query": 32,
        },
        {
            "queried_outcome": "normal",
            "c0_prediction": 0,
            "outcome": "TN",
            "candidates_examined": 32,
            "native_gate_kept": 2,
            "inserted": 0,
            "merged": 0,
            "replaced": 1,
            "skipped": 1,
            "c2_additions": 32,
            "c2_retained_after_query": 64,
        },
    ]

    result = _grouped_query_stats(rows)

    assert result["total"]["candidates_examined"] == 64
    assert result["total"]["native_gate_kept"] == 6
    assert result["total"]["inserted"] == 2
    assert result["total"]["merged"] == 1
    assert result["total"]["replaced"] == 1
    assert result["total"]["skipped"] == 2
    assert result["queried_feedback_balance"] == {"anomaly": 1, "normal": 1}
    assert result["by_c0_confusion_outcome"]["TP"]["queries"] == 1


def test_integrated_analysis_is_aggregate_and_reports_undefined_intervals():
    labels = np.array([0, 1, 0, 1], dtype=np.int8)
    baseline = np.array([0.2, 0.8, 0.3, 0.7], dtype=np.float64)
    scores = {
        "C0": baseline,
        "C1": baseline + np.array([-0.01, 0.02, -0.02, 0.01]),
        "C2": baseline + np.array([0.02, -0.01, 0.03, -0.02]),
    }
    for amplitude in AMPLITUDES:
        arm = f"C2_a{amplitude:.2f}"
        scores[arm] = scores["C2"] if amplitude == 0.20 else (
            baseline + np.array([0.01, -0.005, 0.015, -0.01]) * (amplitude / 0.02)
        )
    patches = {
        name: [np.array([delta, 0.0], dtype=np.float32) for delta in (0.01, -0.02, 0.03, -0.04)]
        for name in scores
        if name != "C0"
    }
    query_rows = [
        {
            "queried_outcome": "normal",
            "queried_label": 0,
            "c0_prediction": 0,
            "outcome": "TN",
            "candidates_examined": 32,
            "native_gate_kept": 2,
            "inserted": 1,
            "merged": 0,
            "replaced": 0,
            "skipped": 1,
            "c2_additions": 32,
            "c2_retained_after_query": 32,
        },
        {
            "queried_outcome": "anomaly",
            "queried_label": 1,
            "c0_prediction": 1,
            "outcome": "TP",
            "candidates_examined": 32,
            "native_gate_kept": 3,
            "inserted": 1,
            "merged": 1,
            "replaced": 0,
            "skipped": 1,
            "c2_additions": 32,
            "c2_retained_after_query": 64,
        },
    ]

    result = analyze_outcomes(
        labels=labels,
        scores=scores,
        source_groups=["a", "a", "b", "b"],
        positions=[0, 1, 2, 3],
        query_positions=[1, 2],
        threshold=0.5,
        patch_deltas=patches,
        dense_maps=[np.array([0.8, 0.4, 0.3], dtype=np.float32) for _ in labels],
        top_patch_ids=[np.array([0, 1], dtype=np.int64) for _ in labels],
        query_details=query_rows,
        bootstrap_replicates=20,
    )

    assert result["pairwise"]["C1"]["pair_count"] == 4
    assert result["c1_vs_c2_pairwise"]["pair_count"] == 4
    assert len(result["c2_amplitude_sensitivity"]) == 4
    assert result["query_intervals"][0]["events"] == 2
    assert result["query_intervals"][0]["metrics"]["C0"]["unavailable_reason"] is None
    assert result["query_intervals"][1]["events"] == 1
    assert result["query_intervals"][1]["metrics"]["C0"]["auroc"] is None
    assert result["source_group_bootstrap"]["valid_replicates"] == 20
    assert "event_labels" not in result


def test_sanitized_commit_payload_drops_event_level_patch_values():
    analysis = {
        "outcome_counts": {"normal": 1, "anomaly": 1},
        "patch_magnitude": {
            "C1": {
                "all": {"mean_absolute": 0.1},
                "per_event_mean_absolute": [0.2, 0.3],
            }
        },
        "c1_vs_c2_magnitude_comparison": {
            "C1_top32_patch_magnitude": {
                "all": {"mean_absolute": 0.1},
                "per_event_mean_absolute": [0.2, 0.3],
            }
        },
        "nested_diagnostic": [
            {
                "event_id": "private-event",
                "offline_label": 1,
                "source_group_sha256": "private-group",
            }
        ],
    }
    inputs = {
        "input_hashes": {},
        "source_fingerprints": {},
        "tape_receipt": {"fixed_support_sha256": "fixed"},
        "r1_manifest": {"fixed_support_raw_prototype_sha256": "raw"},
    }
    run_manifest = {"run_id": "new-run", "expected_repository": {"branch": "b", "head": "h"}}

    result = _sanitized_aggregate(analysis, inputs, run_manifest, Path("private-run"))

    assert result["analysis"]["patch_magnitude"]["C1"]["all"]["mean_absolute"] == 0.1
    assert "per_event_mean_absolute" not in result["analysis"]["patch_magnitude"]["C1"]
    assert "per_event_mean_absolute" not in result["analysis"]["c1_vs_c2_magnitude_comparison"]["C1_top32_patch_magnitude"]
    assert result["analysis"]["nested_diagnostic"] == [{}]
    assert result["private_event_outcomes_and_feedback_committed"] is False
