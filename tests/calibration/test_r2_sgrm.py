"""Synthetic CPU contract tests for the frozen R2-SGRM readout."""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from dataclasses import dataclass
import math

import numpy as np
import pytest

from experiments.calibration import r2_sgrm


@dataclass
class Anchor:
    feature: object
    beta: float
    radius: float = 0.10
    kappa: float = 1.0
    position: int = 0
    source_group: str = "group-a"


def _scalar_reference(distances, beta, groups, *, tau=0.10, epsilon=1e-6):
    """Independent direct float64 implementation of the frozen equations."""
    d = np.asarray(distances, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    unique = list(dict.fromkeys(groups))
    result = []
    for query_index in range(d.shape[0]):
        group_means = []
        group_maxima = []
        for group in unique:
            indices = [i for i, identity in enumerate(groups) if identity == group]
            weights = [math.exp(-(float(d[query_index, i]) ** 2) / (2 * tau * tau))
                       for i in indices]
            mass = math.fsum(weights)
            group_means.append(math.fsum(w * float(b[i]) for w, i in zip(weights, indices)) / mass)
            group_maxima.append(max(weights))
        numerator = math.fsum(a * mu for a, mu in zip(group_maxima, group_means))
        result.append(numerator / (epsilon + math.fsum(group_maxima)))
    return np.asarray(result, dtype=np.float64)


def _anchor(beta, group, *, position=0, feature=None, radius=0.10, kappa=1.0):
    if feature is None:
        feature = np.asarray([1.0, 0.0], dtype=np.float32)
    return Anchor(feature, beta, radius, kappa, position, group)


def test_empty_memory_returns_finite_zero_and_diagnostics():
    result, diagnostics = r2_sgrm.sgrm_delta_from_distances(
        np.empty((3, 0), dtype=np.float32), np.empty((0,), dtype=np.float32), [],
        return_diagnostics=True)
    assert result.dtype == np.float32
    assert result.tolist() == [0.0, 0.0, 0.0]
    assert np.isfinite(result).all()
    assert diagnostics["eligible_patch_count"] == 0
    assert diagnostics["eligible_source_group_count"] == 0
    assert diagnostics["zero_evidence_flags"].tolist() == [True, True, True]


def test_one_group_has_analytic_epsilon_attenuated_residual():
    beta = np.array([0.2, 0.2], dtype=np.float32)
    result = r2_sgrm.sgrm_delta_from_distances(
        np.zeros((1, 2), dtype=np.float32), beta, ["one", "one"])
    assert result[0] == pytest.approx(0.2 / (1.0 + 1e-6), abs=1e-8)


def test_singleton_groups_equal_flat_gaussian_reference():
    rng = np.random.default_rng(20261101)
    distances = rng.uniform(0.0, 1.6, size=(7, 9)).astype(np.float32)
    beta = rng.uniform(-0.2, 0.2, size=9).astype(np.float32)
    groups = [f"singleton-{index}" for index in range(9)]
    result = r2_sgrm.sgrm_delta_from_distances(distances, beta, groups)
    expected = _scalar_reference(distances, beta, groups)
    assert np.allclose(result, expected, atol=2e-8, rtol=2e-7)


def test_whole_group_replication_invariance():
    distances = np.array([[0.02, 0.15, 0.04, 0.18]], dtype=np.float64)
    beta = np.array([0.08, 0.04, -0.12, -0.06], dtype=np.float64)
    groups = ["P", "P", "N", "N"]
    original = r2_sgrm.sgrm_delta_from_distances(distances, beta, groups)
    repeat = 5
    duplicated_d = np.concatenate([np.repeat(distances[:, :2], repeat, axis=1), distances[:, 2:]], axis=1)
    duplicated_b = np.concatenate([np.repeat(beta[:2], repeat), beta[2:]])
    duplicated_g = ["P"] * (2 * repeat) + ["N", "N"]
    duplicated = r2_sgrm.sgrm_delta_from_distances(duplicated_d, duplicated_b, duplicated_g)
    assert duplicated == pytest.approx(original, abs=2e-8, rel=2e-7)


def test_anchor_permutation_and_group_renaming_invariance():
    distances = np.array([[0.0, 0.03, 0.11, 0.2, 0.4]], dtype=np.float64)
    beta = np.array([0.12, 0.04, -0.08, -0.03, 0.07], dtype=np.float64)
    groups = ["A", "A", "B", "B", "C"]
    baseline = r2_sgrm.sgrm_delta_from_distances(distances, beta, groups)
    order = np.array([4, 2, 0, 3, 1])
    renamed = [{"A": "z", "B": "x", "C": "y"}[groups[i]] for i in order]
    result = r2_sgrm.sgrm_delta_from_distances(distances[:, order], beta[order], renamed)
    assert result == pytest.approx(baseline, abs=2e-8, rel=2e-7)


def test_documented_multiplicity_example_cancels_by_source_group():
    distances = np.zeros((1, 12), dtype=np.float64)
    beta = np.array([0.02] * 10 + [-0.02] * 2, dtype=np.float64)
    groups = ["P"] * 10 + ["N"] * 2
    sgrm = r2_sgrm.sgrm_delta_from_distances(distances, beta, groups)
    flat = _scalar_reference(distances, beta, [f"patch-{i}" for i in range(12)])[0]
    assert abs(float(sgrm[0])) < 2e-8
    assert flat == pytest.approx(0.0133332, abs=2e-7)


def test_mixed_group_positive_negative_evidence_cancels():
    result = r2_sgrm.sgrm_delta_from_distances(
        np.zeros((2, 2), dtype=np.float32),
        np.array([0.02, -0.02], dtype=np.float32), ["positive", "negative"])
    assert np.allclose(result, 0.0, atol=1e-8)


def test_seeded_randomized_magnitude_bound():
    rng = np.random.default_rng(75291)
    for _ in range(40):
        group_count = 1 + int(rng.integers(1, 8))
        assignments = rng.integers(0, group_count, size=35)
        signs = rng.choice(np.array([-1.0, 1.0]), size=group_count)
        beta = np.array([
            0.0 if rng.random() < 0.1 else signs[group] * rng.uniform(0.0, 0.2)
            for group in assignments], dtype=np.float32)
        groups = [f"G{group}" for group in assignments]
        distances = rng.uniform(0.0, 2.0, size=(5, len(beta))).astype(np.float32)
        delta = r2_sgrm.sgrm_delta_from_distances(distances, beta, groups)
        assert np.isfinite(delta).all()
        assert np.max(np.abs(delta)) <= np.max(np.abs(beta)) + 2e-7


def test_seeded_randomized_grouped_result_matches_independent_scalar_reference():
    rng = np.random.default_rng(90127)
    group_index = rng.integers(0, 6, size=23)
    group_sign = rng.choice(np.array([-1.0, 1.0]), size=6)
    beta = np.array([
        0.0 if rng.random() < 0.15 else group_sign[group] * rng.uniform(0.001, 0.2)
        for group in group_index], dtype=np.float32)
    groups = [f"src-{index}" for index in group_index]
    distances = rng.uniform(0.0, 2.0, size=(4, len(beta))).astype(np.float32)
    actual = r2_sgrm.sgrm_delta_from_distances(distances, beta, groups)
    expected = _scalar_reference(distances, beta, groups)
    assert np.allclose(actual, expected, atol=2e-8, rtol=2e-7)


def test_near_zero_gaussian_mass_is_stable_and_finite():
    result, diagnostics = r2_sgrm.sgrm_delta_from_distances(
        np.array([[2.0], [0.0]], dtype=np.float64), np.array([0.15]), ["far"],
        tau=1e-3, return_diagnostics=True)
    assert result[0] == 0.0
    assert result[1] == pytest.approx(0.15 / (1 + 1e-6), abs=1e-7)
    assert np.isfinite(diagnostics["group_means"]).all()
    assert np.isfinite(diagnostics["outer_group_weights"]).all()
    assert diagnostics["zero_evidence_flags"].tolist() == [True, False]


@pytest.mark.parametrize("distances,beta,groups", [
    (np.zeros(2), np.zeros(2), ["a", "a"]),
    (np.zeros((1, 2)), np.zeros(1), ["a", "a"]),
    (np.zeros((1, 1), dtype=np.int64), np.zeros(1, dtype=np.float32), ["a"]),
    (np.array([[np.nan]], dtype=np.float32), np.zeros(1, dtype=np.float32), ["a"]),
    (np.zeros((1, 1), dtype=np.float32), np.array([np.inf]), ["a"]),
    (np.array([[2.1]], dtype=np.float32), np.zeros(1, dtype=np.float32), ["a"]),
    (np.zeros((1, 1), dtype=np.float32), np.zeros(1, dtype=np.float32), [""]),
    (np.zeros((1, 1), dtype=np.float32), np.zeros(1, dtype=np.float32), "a"),
])
def test_invalid_core_inputs_fail_closed(distances, beta, groups):
    with pytest.raises(r2_sgrm.SGRMInputError):
        r2_sgrm.sgrm_delta_from_distances(distances, beta, groups)


@pytest.mark.parametrize("kwargs", [
    {"tau": 0.0}, {"tau": -0.1}, {"tau": np.inf}, {"tau": np.nan},
    {"epsilon": 0.0}, {"epsilon": -1.0}, {"epsilon": np.inf}, {"epsilon": np.nan},
])
def test_nonpositive_or_nonfinite_parameters_fail(kwargs):
    with pytest.raises(r2_sgrm.SGRMInputError):
        r2_sgrm.sgrm_delta_from_distances(
            np.zeros((1, 1), dtype=np.float32), np.ones(1, dtype=np.float32), ["g"], **kwargs)


def test_distance_roundoff_allowance_is_explicit_and_values_are_preserved():
    result, diag = r2_sgrm.sgrm_delta_from_distances(
        np.array([[-5e-7]], dtype=np.float64), np.array([0.1]), ["g"], return_diagnostics=True)
    assert result[0] == pytest.approx(0.1 / (1 + 1e-6), abs=1e-7)
    expected_relevance = math.exp(-0.5 * ((-5e-7) / 0.10) ** 2)
    assert diag["group_max_relevance"][0, 0] == pytest.approx(expected_relevance, abs=1e-15)
    with pytest.raises(r2_sgrm.SGRMInputError, match="distances"):
        r2_sgrm.sgrm_delta_from_distances(
            np.array([[-2e-6]], dtype=np.float64), np.array([0.1]), ["g"])


def test_group_sign_consistency_allows_zero_but_rejects_conflict():
    result = r2_sgrm.sgrm_delta_from_distances(
        np.zeros((1, 3), dtype=np.float32), np.array([0.1, 0.0, 0.2]), ["g", "g", "g"])
    assert result[0] > 0.0
    with pytest.raises(r2_sgrm.SGRMInputError, match="conflicting"):
        r2_sgrm.sgrm_delta_from_distances(
            np.zeros((1, 2), dtype=np.float32), np.array([0.1, -0.1]), ["g", "g"])


def test_wrapper_excludes_query_source_and_reports_only_aggregate_diagnostics():
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    anchors = [
        _anchor(0.18, "query-group", feature=np.array([1.0, 0.0], dtype=np.float32)),
        _anchor(0.06, "other-group", position=1, feature=np.array([1.0, 0.0], dtype=np.float32)),
    ]
    delta, diagnostics = r2_sgrm.sgrm_delta_from_anchors(
        query, anchors, current_position=2, current_source_group="query-group",
        return_diagnostics=True)
    assert delta[0] == pytest.approx(0.06 / (1.0 + 1e-6), abs=1e-7)
    assert diagnostics["excluded_same_source_anchor_count"] == 1
    assert diagnostics["eligible_patch_count"] == 1
    assert diagnostics["group_patch_counts"] == (1,)
    assert "query-group" not in repr(diagnostics)


@pytest.mark.parametrize("position", [3, 4])
def test_wrapper_rejects_current_and_future_anchors(position):
    with pytest.raises(r2_sgrm.SGRMInputError, match="current/future"):
        r2_sgrm.sgrm_delta_from_anchors(
            np.array([[1.0, 0.0]], dtype=np.float32), [_anchor(0.1, "other", position=position)],
            current_position=3, current_source_group="query")


def test_wrapper_rejects_negative_event_positions():
    with pytest.raises(r2_sgrm.SGRMInputError, match="nonnegative"):
        r2_sgrm.sgrm_delta_from_anchors(
            np.array([[1.0, 0.0]], dtype=np.float32), [],
            current_position=-1, current_source_group="query")
    with pytest.raises(r2_sgrm.SGRMInputError, match="nonnegative"):
        r2_sgrm.sgrm_delta_from_anchors(
            np.array([[1.0, 0.0]], dtype=np.float32), [_anchor(0.1, "old", position=-1)],
            current_position=0, current_source_group="query")


@pytest.mark.parametrize("anchor", [
    _anchor(0.1, "g", kappa=0.9),
    _anchor(0.1, "g", radius=0.11),
])
def test_wrapper_rejects_unsupported_confidence_or_radius(anchor):
    with pytest.raises(r2_sgrm.SGRMInputError, match="requires"):
        r2_sgrm.sgrm_delta_from_anchors(
            np.array([[1.0, 0.0]], dtype=np.float32), [anchor],
            current_position=2, current_source_group="query")


def test_wrapper_conflicting_signs_in_one_source_group_fail_closed():
    anchors = [_anchor(0.1, "g"), _anchor(-0.1, "g", position=1)]
    with pytest.raises(r2_sgrm.SGRMInputError, match="conflicting"):
        r2_sgrm.sgrm_delta_from_anchors(
            np.array([[1.0, 0.0]], dtype=np.float32), anchors,
            current_position=2, current_source_group="query")


@pytest.mark.parametrize("query,anchor", [
    (np.array([[1, 0]], dtype=np.int64), _anchor(0.1, "g")),
    (np.array([[np.nan, 0.0]], dtype=np.float32), _anchor(0.1, "g")),
    (np.array([1.0, 0.0], dtype=np.float32), _anchor(0.1, "g")),
    (np.array([[1.0, 0.0]], dtype=np.float32), _anchor(0.1, "g", feature=np.ones(3, dtype=np.float32))),
    (np.array([[1.0, 0.0]], dtype=np.float32), _anchor(0.1, "g", feature=np.array([np.inf, 0.0], dtype=np.float32))),
    (np.array([[1.0, 0.0]], dtype=np.float32), _anchor(0.1, " ")),
])
def test_wrapper_rejects_invalid_descriptor_shapes_types_and_values(query, anchor):
    with pytest.raises(r2_sgrm.SGRMInputError):
        r2_sgrm.sgrm_delta_from_anchors(
            query, [anchor], current_position=2, current_source_group="query")


def test_wrapper_does_not_mutate_query_anchor_features_or_objects():
    query = np.array([[3.0, 4.0]], dtype=np.float32)
    feature = np.array([6.0, 8.0], dtype=np.float32)
    anchor = _anchor(0.1, "prior", feature=feature, position=1)
    query_before = query.copy()
    feature_before = feature.copy()
    metadata_before = (anchor.beta, anchor.radius, anchor.kappa, anchor.position, anchor.source_group)
    r2_sgrm.sgrm_delta_from_anchors(
        query, [anchor], current_position=2, current_source_group="current")
    assert np.array_equal(query, query_before)
    assert np.array_equal(feature, feature_before)
    assert (anchor.beta, anchor.radius, anchor.kappa, anchor.position, anchor.source_group) == metadata_before


def test_score_before_query_update_affects_only_later_eligible_scores():
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    bank = []
    score_t = r2_sgrm.sgrm_delta_from_anchors(
        query, bank, current_position=5, current_source_group="source-t")
    # Event-t feedback is appended only after its score has been computed.
    bank.append(_anchor(0.12, "source-t", position=5, feature=query[0].copy()))
    score_t_plus_1 = r2_sgrm.sgrm_delta_from_anchors(
        query, bank, current_position=6, current_source_group="source-next")
    score_same_source = r2_sgrm.sgrm_delta_from_anchors(
        query, bank, current_position=6, current_source_group="source-t")
    assert score_t[0] == 0.0
    assert score_t_plus_1[0] == pytest.approx(0.12 / (1.0 + 1e-6), abs=1e-7)
    assert score_same_source[0] == 0.0


def test_dense_map_integration_changes_only_selected_ids_then_uses_full_map_max():
    r1 = __import__("experiments.calibration.r1_matched_correction_pilot",
                    fromlist=["corrected_dense_map", "max_score"])
    delta = r2_sgrm.sgrm_delta_from_distances(
        np.zeros((2, 2), dtype=np.float32), np.array([0.1, -0.05], dtype=np.float32),
        ["p", "n"])
    base = np.zeros((784,), dtype=np.float32)
    base[777] = 0.8
    dense = r1.corrected_dense_map(base, [5, 20], delta)
    changed = np.flatnonzero(dense != base)
    assert changed.tolist() == [5, 20]
    assert r1.max_score(dense) == pytest.approx(0.8)
    base[777] = 0.0
    dense = r1.corrected_dense_map(base, [5, 20], delta)
    assert r1.max_score(dense) == pytest.approx(float(dense.max()))


def test_cpu_4096d_interface_smoke_with_existing_torch_anchor_shape():
    import torch

    torch_query = torch.zeros((2, 4096), dtype=torch.float32, device="cpu")
    torch_anchor = torch.zeros((4096,), dtype=torch.float32, device="cpu")
    torch_query[:, 0] = 1.0
    torch_anchor[0] = 1.0
    anchor = _anchor(0.1, "old", position=0, feature=torch_anchor)
    delta = r2_sgrm.sgrm_delta_from_anchors(
        torch_query, [anchor], current_position=1, current_source_group="new")
    assert delta.shape == (2,)
    assert delta.dtype == np.float32
    assert np.allclose(delta, 0.1 / (1.0 + 1e-6), atol=1e-7)


def test_wrapper_normalization_matches_float32_cpu_feature_contract():
    query = np.array([[3.0, 4.0], [3.5, 4.2]], dtype=np.float32)
    features = [np.array([6.0, 8.0], dtype=np.float32),
                np.array([4.0, 5.0], dtype=np.float32)]
    anchors = [_anchor(0.1, "positive", position=0, feature=features[0]),
               _anchor(-0.04, "negative", position=1, feature=features[1])]
    actual = r2_sgrm.sgrm_delta_from_anchors(
        query, anchors, current_position=2, current_source_group="query")
    qnorm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    fnorm = np.stack(features) / (
        np.linalg.norm(np.stack(features), axis=1, keepdims=True) + 1e-12)
    distances = 1.0 - np.clip(qnorm @ fnorm.T, -1.0, 1.0)
    expected = _scalar_reference(distances, [0.1, -0.04], ["positive", "negative"])
    assert np.allclose(actual, expected, atol=2e-8, rtol=2e-7)


def test_selective_duplication_inside_heterogeneous_group_can_change_result():
    distances = np.array([[0.0, 0.2, 0.05]], dtype=np.float64)
    beta = np.array([0.1, 0.02, -0.03], dtype=np.float64)
    groups = ["heterogeneous", "heterogeneous", "other"]
    original = r2_sgrm.sgrm_delta_from_distances(distances, beta, groups)
    selective_d = np.array([[0.0, 0.2, 0.2, 0.05]], dtype=np.float64)
    selective_b = np.array([0.1, 0.02, 0.02, -0.03], dtype=np.float64)
    selective_g = ["heterogeneous", "heterogeneous", "heterogeneous", "other"]
    changed = r2_sgrm.sgrm_delta_from_distances(selective_d, selective_b, selective_g)
    assert abs(float(changed[0] - original[0])) > 1e-3
