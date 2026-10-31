from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.calibration.r2_l1_locality_kernels import (
    CorrectionAnchor,
    c1_compact_delta,
    c1_gated_delta,
    compact_support_phi,
    cosine_distances,
    eligible_anchor_indices,
    gated_signed_knn_delta,
    l2_normalize,
    legacy_c1_delta,
    legacy_signed_knn_delta,
)


QUERY_GROUP = "a" * 64
OTHER_GROUP = "b" * 64
THIRD_GROUP = "c" * 64


def unit_at_cosine_distance(distance: float, *, dtype=np.float64) -> np.ndarray:
    """A unit vector whose dot product with e1 is exactly 1-distance."""
    cosine = 1.0 - float(distance)
    return np.asarray([cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine))], dtype=dtype)


def make_anchor(
    feature: object = (1.0, 0.0),
    *,
    beta: float = 0.5,
    kappa: float = 1.0,
    gaussian_width: float = 0.2,
    support_radius: float = 0.3,
    position: int = 1,
    source_group: str = OTHER_GROUP,
    c2_sign: int | None = 1,
) -> CorrectionAnchor:
    return CorrectionAnchor(
        feature=np.asarray(feature), beta=beta, kappa=kappa,
        gaussian_width=gaussian_width, support_radius=support_radius,
        position=position, source_group=source_group, c2_sign=c2_sign,
    )


def call_c1(function, query, anchors, **kwargs):
    query_position = kwargs.pop("query_position", 100)
    query_source_group = kwargs.pop("query_source_group", QUERY_GROUP)
    return function(
        query, anchors, query_position=query_position,
        query_source_group=query_source_group, **kwargs)


def call_knn(function, query, anchors, **kwargs):
    return function(
        query, anchors, query_position=100, query_source_group=QUERY_GROUP, **kwargs)


def test_single_anchor_legacy_rbf_retains_beta_at_four_gaussian_widths():
    anchor = make_anchor(beta=0.5, gaussian_width=0.2, support_radius=0.3)
    query = unit_at_cosine_distance(0.8)

    old = call_c1(legacy_c1_delta, query, [anchor])
    gated = call_c1(c1_gated_delta, query, [anchor])
    compact = call_c1(c1_compact_delta, query, [anchor], evidence_regularizer=0.1)

    assert old / anchor.beta > 0.99  # exp(-8) is still much larger than 1e-6.
    assert gated == 0.0
    assert compact == 0.0


def test_legacy_c1_matches_its_normalized_gaussian_equation():
    anchor = make_anchor(beta=-0.35, kappa=0.7, gaussian_width=0.5)
    distance = 0.2
    query = unit_at_cosine_distance(distance)
    weight = anchor.kappa * math.exp(-(distance ** 2) / (2.0 * anchor.gaussian_width ** 2))
    expected = weight * anchor.beta / (weight + 1e-6)
    assert call_c1(legacy_c1_delta, query, [anchor]) == pytest.approx(expected, rel=1e-14)


def test_original_signed_knn_is_nonlocal_even_at_maximum_cosine_distance():
    anchor = make_anchor(feature=(-1.0, 0.0), beta=-0.2, c2_sign=-1,
                         support_radius=0.01)
    query = np.asarray([1.0, 0.0])
    assert cosine_distances(query, np.stack([anchor.feature]))[0] == 2.0
    assert call_knn(legacy_signed_knn_delta, query, [anchor], k=5, amplitude=0.2) == -0.2
    assert call_knn(
        gated_signed_knn_delta, query, [anchor], support_radius=0.5,
        evidence_regularizer=0.1, k=5, amplitude=0.2) == 0.0


def test_cosine_distance_semantics_and_normalization_validation():
    normalized = l2_normalize(np.asarray([[3.0, 4.0], [-2.0, 0.0]], dtype=np.float32))
    assert normalized.dtype == np.float32
    d = cosine_distances(normalized[0], normalized)
    assert d[0] == pytest.approx(0.0, abs=1e-7)
    assert d[1] == pytest.approx(1.6, abs=1e-6)
    assert cosine_distances([1.0, 0.0], [[-1.0, 0.0]])[0] == 2.0
    with pytest.raises(ValueError, match="L2-normalized"):
        cosine_distances([2.0, 0.0], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="zero-norm"):
        l2_normalize([0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        l2_normalize([np.nan, 1.0])


@pytest.mark.parametrize("normalized_distance,expected", [
    (0.0, 1.0), (0.5, 0.5625), (1.0, 0.0), (1.5, 0.0),
])
def test_compact_support_profile(normalized_distance, expected):
    assert compact_support_phi(normalized_distance) == pytest.approx(expected)


def test_compact_support_profile_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="nonnegative"):
        compact_support_phi(-0.1)
    with pytest.raises(ValueError, match="finite"):
        compact_support_phi(np.inf)


def test_c1_candidates_at_zero_distance_and_inside_support():
    anchor = make_anchor(beta=0.6, kappa=2.0, support_radius=0.5)
    query = np.asarray([1.0, 0.0])
    gated = call_c1(c1_gated_delta, query, [anchor])
    compact = call_c1(c1_compact_delta, query, [anchor], evidence_regularizer=0.5)
    assert gated == pytest.approx(0.6 * 2.0 / (2.0 + 1e-6), rel=1e-12)
    assert compact == pytest.approx(0.6 * 2.0 / 2.5)

    inside = unit_at_cosine_distance(0.25)
    assert call_c1(c1_gated_delta, inside, [anchor]) > 0.0
    assert call_c1(c1_compact_delta, inside, [anchor], evidence_regularizer=0.5) > 0.0


def test_c1_candidates_are_zero_at_and_beyond_single_anchor_support_boundary():
    radius = 0.4
    anchor = make_anchor(beta=0.7, support_radius=radius)
    boundary = unit_at_cosine_distance(radius)
    outside = unit_at_cosine_distance(radius + 1e-8)
    assert call_c1(c1_gated_delta, boundary, [anchor]) == 0.0
    assert call_c1(c1_compact_delta, boundary, [anchor], evidence_regularizer=0.1) == 0.0
    assert call_c1(c1_gated_delta, outside, [anchor]) == 0.0
    assert call_c1(c1_compact_delta, outside, [anchor], evidence_regularizer=0.1) == 0.0
    assert call_c1(c1_compact_delta, unit_at_cosine_distance(2.0), [anchor],
                   evidence_regularizer=0.1) == 0.0


def test_candidate_boundary_values_approach_zero_continuously():
    radius = 0.6
    anchor = make_anchor(beta=0.7, support_radius=radius)
    regularizer = 0.1
    at_1e_3 = unit_at_cosine_distance(radius * (1.0 - 1e-3))
    at_1e_6 = unit_at_cosine_distance(radius * (1.0 - 1e-6))
    at_boundary = unit_at_cosine_distance(radius)

    for function, kwargs in (
        (c1_gated_delta, {}),
        (c1_compact_delta, {"evidence_regularizer": regularizer}),
    ):
        v1 = call_c1(function, at_1e_3, [anchor], **kwargs)
        v2 = call_c1(function, at_1e_6, [anchor], **kwargs)
        vb = call_c1(function, at_boundary, [anchor], **kwargs)
        assert abs(v2) < abs(v1)
        assert abs(v2) < 1e-9
        assert vb == 0.0


def test_compact_weights_cancel_equal_opposite_anchors():
    positive = make_anchor(beta=0.4, c2_sign=1, position=1)
    negative = make_anchor(beta=-0.4, c2_sign=-1, position=2)
    result = call_c1(c1_compact_delta, [1.0, 0.0], [positive, negative],
                     evidence_regularizer=0.01)
    assert result == 0.0


def test_compact_weights_mix_unequal_confidences_and_signed_strengths():
    positive = make_anchor(beta=0.2, kappa=4.0, position=1)
    negative = make_anchor(beta=-0.1, kappa=1.0, position=2, c2_sign=-1)
    result = call_c1(c1_compact_delta, [1.0, 0.0], [positive, negative],
                     evidence_regularizer=1.0)
    assert result == pytest.approx((4.0 * 0.2 - 0.1) / 6.0)


def test_c1_compact_ignores_disjoint_support_anchor_but_gated_legacy_mixes_it():
    near = make_anchor(beta=0.5, support_radius=0.2, gaussian_width=2.0, position=1)
    far = make_anchor(feature=unit_at_cosine_distance(1.0), beta=-0.5,
                      support_radius=0.2, gaussian_width=2.0, position=2,
                      source_group=THIRD_GROUP, c2_sign=-1)
    query = np.asarray([1.0, 0.0])
    single_gated = call_c1(c1_gated_delta, query, [near])
    mixed_gated = call_c1(c1_gated_delta, query, [near, far])
    single_compact = call_c1(c1_compact_delta, query, [near], evidence_regularizer=0.1)
    mixed_compact = call_c1(c1_compact_delta, query, [near, far], evidence_regularizer=0.1)
    assert mixed_gated != pytest.approx(single_gated)
    assert mixed_compact == pytest.approx(single_compact)


def test_empty_and_source_excluded_anchor_memory_returns_exact_zero():
    same_source = make_anchor(source_group=QUERY_GROUP)
    for function, kwargs in (
        (legacy_c1_delta, {}), (c1_gated_delta, {}),
        (c1_compact_delta, {"evidence_regularizer": 1.0}),
    ):
        assert call_c1(function, [1.0, 0.0], [], **kwargs) == 0.0
        assert call_c1(function, [1.0, 0.0], [same_source], **kwargs) == 0.0
    assert call_knn(legacy_signed_knn_delta, [1.0, 0.0], [], k=5) == 0.0
    assert call_knn(gated_signed_knn_delta, [1.0, 0.0], [same_source],
                    support_radius=1.0, evidence_regularizer=0.1) == 0.0


def test_c1_output_is_bounded_by_max_anchor_strength_and_max_score_shift():
    anchors = [
        make_anchor(beta=(-1.0) ** i * (0.05 + i % 7 * 0.03),
                    kappa=0.2 + i, position=i,
                    source_group=(f"{i+1:064x}"[-64:]))
        for i in range(100)
    ]
    bound = max(abs(a.beta) for a in anchors)
    query = np.asarray([1.0, 0.0])
    for result in (
        call_c1(c1_gated_delta, query, anchors),
        call_c1(c1_compact_delta, query, anchors, evidence_regularizer=0.25),
    ):
        assert abs(result) <= bound + 1e-14

    base = np.linspace(-2.0, 2.0, 32)
    deltas = np.asarray([
        call_c1(c1_compact_delta, query, [a], evidence_regularizer=0.25)
        for a in anchors[:32]
    ])
    score_shift = abs(float(np.max(base + deltas)) - float(np.max(base)))
    assert score_shift <= bound + 1e-14


def test_c1_compact_is_stable_for_tiny_radii_weights_and_regularizers():
    tiny_radius = 1e-300
    tiny = make_anchor(beta=0.5, kappa=1e-300, support_radius=tiny_radius,
                       gaussian_width=tiny_radius)
    same = call_c1(c1_compact_delta, [1.0, 0.0], [tiny], evidence_regularizer=1e-200)
    far = call_c1(c1_compact_delta, [0.0, 1.0], [tiny], evidence_regularizer=1e-200)
    assert math.isfinite(same) and same > 0.0
    assert far == 0.0


def test_c1_compact_stays_bounded_with_many_huge_finite_confidences():
    anchors = [make_anchor(beta=0.3 if i % 2 else -0.3, kappa=1e300,
                           position=i + 1, source_group=f"{i+1:064x}")
               for i in range(256)]
    result = call_c1(c1_compact_delta, [1.0, 0.0], anchors,
                     query_position=300, evidence_regularizer=1e-300)
    assert math.isfinite(result)
    assert abs(result) <= 0.3


def test_float32_and_float64_inputs_agree_within_roundoff():
    feature32 = l2_normalize(np.asarray([0.3, math.sqrt(0.91)], dtype=np.float32))
    feature64 = l2_normalize(np.asarray([0.3, math.sqrt(0.91)], dtype=np.float64))
    query32 = l2_normalize(np.asarray([0.8, 0.6], dtype=np.float32))
    query64 = l2_normalize(np.asarray([0.8, 0.6], dtype=np.float64))
    a32 = make_anchor(feature32, beta=0.2, support_radius=1.0)
    a64 = make_anchor(feature64, beta=0.2, support_radius=1.0)
    d32 = cosine_distances(query32, np.stack([a32.feature]))[0]
    d64 = cosine_distances(query64, np.stack([a64.feature]))[0]
    y32 = call_c1(c1_compact_delta, query32, [a32], evidence_regularizer=0.1)
    y64 = call_c1(c1_compact_delta, query64, [a64], evidence_regularizer=0.1)
    assert d32 == pytest.approx(d64, abs=2e-7)
    assert y32 == pytest.approx(y64, abs=2e-7)


def test_eligibility_keeps_only_past_cross_source_anchors():
    legal = make_anchor(position=3, source_group=OTHER_GROUP)
    same_source = make_anchor(position=4, source_group=QUERY_GROUP)
    assert eligible_anchor_indices(
        [legal, same_source], query_position=5, query_source_group=QUERY_GROUP) == (0,)


@pytest.mark.parametrize("position", [5, 6])
def test_current_or_future_anchor_is_rejected(position):
    anchor = make_anchor(position=position)
    with pytest.raises(ValueError, match="current/future"):
        eligible_anchor_indices([anchor], query_position=5, query_source_group=QUERY_GROUP)


@pytest.mark.parametrize("bad_group", [None, "not-a-digest", "A" * 64, "0" * 63])
def test_missing_or_malformed_source_provenance_fails_closed(bad_group):
    with pytest.raises(ValueError, match="source-group"):
        make_anchor(source_group=bad_group)


def test_unvalidated_anchor_provenance_and_bad_query_provenance_fail_closed():
    with pytest.raises(ValueError, match="lacks validated provenance"):
        eligible_anchor_indices([object()], query_position=10, query_source_group=QUERY_GROUP)
    with pytest.raises(ValueError, match="query source group"):
        eligible_anchor_indices([], query_position=10, query_source_group="unknown")
    with pytest.raises(ValueError, match="query position"):
        eligible_anchor_indices([], query_position=True, query_source_group=QUERY_GROUP)


def test_anchor_arrays_are_revalidated_if_mutated_after_construction():
    anchor = make_anchor()
    anchor.feature.flags.writeable = True
    anchor.feature[0] = 2.0
    with pytest.raises(ValueError, match="L2-normalized"):
        call_c1(c1_compact_delta, [1.0, 0.0], [anchor], evidence_regularizer=0.1)


def test_invalid_anchor_numeric_fields_are_rejected():
    for overrides, expected in (
        ({"beta": np.inf}, "beta"),
        ({"kappa": 0.0}, "kappa"),
        ({"kappa": -1.0}, "kappa"),
        ({"gaussian_width": 0.0}, "gaussian_width"),
        ({"gaussian_width": -1.0}, "gaussian_width"),
        ({"support_radius": 0.0}, "support_radius"),
        ({"support_radius": -1.0}, "support_radius"),
        ({"c2_sign": 0}, "c2_sign"),
    ):
        with pytest.raises(ValueError, match=expected):
            make_anchor(**overrides)
    with pytest.raises(ValueError, match="L2-normalized"):
        make_anchor(feature=(1.0, 1.0))
    with pytest.raises(ValueError, match="nonzero norm"):
        make_anchor(feature=(0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        make_anchor(feature=(np.nan, 0.0))


def test_legacy_knn_uses_k_eff_for_small_memory_and_is_stable_on_ties():
    first = make_anchor(c2_sign=1, position=1)
    second = make_anchor(c2_sign=-1, position=2, source_group=THIRD_GROUP)
    one = call_knn(legacy_signed_knn_delta, [1.0, 0.0], [first], k=5, amplitude=0.2)
    tied = call_knn(legacy_signed_knn_delta, [1.0, 0.0], [first, second], k=1, amplitude=0.2)
    assert one == 0.2
    assert tied == 0.2  # equal distances use original anchor order
    opposite = call_knn(legacy_signed_knn_delta, [1.0, 0.0], [first, second],
                        k=2, amplitude=0.2)
    assert opposite == 0.0


def test_gated_knn_uses_nearest_local_anchors_and_amplitude_bound():
    local_positive = make_anchor(c2_sign=1, position=1)
    local_negative = make_anchor(feature=unit_at_cosine_distance(0.05), c2_sign=-1,
                                 position=2, source_group=THIRD_GROUP)
    far_positive = make_anchor(feature=(-1.0, 0.0), c2_sign=1, position=3,
                               source_group="d" * 64)
    query = np.asarray([1.0, 0.0])
    result = call_knn(
        gated_signed_knn_delta, query, [local_positive, local_negative, far_positive],
        support_radius=0.5, evidence_regularizer=0.01, k=2, amplitude=0.2)
    assert -0.2 <= result <= 0.2
    assert result > 0.0  # the nearer positive has the larger compact weight
    no_local = call_knn(
        gated_signed_knn_delta, query, [far_positive], support_radius=0.5,
        evidence_regularizer=0.01, k=5, amplitude=0.2)
    assert no_local == 0.0


def test_gated_knn_regularized_single_neighbor_matches_formula():
    anchor = make_anchor(c2_sign=-1)
    result = call_knn(
        gated_signed_knn_delta, [1.0, 0.0], [anchor], support_radius=0.5,
        evidence_regularizer=0.5, k=1, amplitude=0.2)
    assert result == pytest.approx(-0.2 / 1.5)


def test_multiple_same_sign_anchors_reinforce_inside_overlapping_support():
    first = make_anchor(beta=0.2, kappa=1.0, support_radius=0.5,
                        position=1, c2_sign=1)
    second = make_anchor(beta=0.4, kappa=2.0, support_radius=0.5,
                         position=2, source_group=THIRD_GROUP, c2_sign=1)
    value = call_c1(c1_compact_delta, [1.0, 0.0], [first, second],
                    evidence_regularizer=0.1)
    assert 0.0 < value < 0.4


def test_gated_knn_is_zero_at_support_and_tends_to_zero_from_inside():
    radius = 0.4
    anchor = make_anchor(c2_sign=-1)
    at_boundary = unit_at_cosine_distance(radius)
    just_inside = unit_at_cosine_distance(radius * (1.0 - 1e-6))
    outside = unit_at_cosine_distance(radius + 1e-7)
    kwargs = {"support_radius": radius, "evidence_regularizer": 0.2, "k": 1}
    assert call_knn(gated_signed_knn_delta, at_boundary, [anchor], **kwargs) == 0.0
    assert call_knn(gated_signed_knn_delta, outside, [anchor], **kwargs) == 0.0
    value = call_knn(gated_signed_knn_delta, just_inside, [anchor], **kwargs)
    assert value < 0.0
    assert abs(value) < 1e-8


def test_gated_knn_neighbor_set_switch_can_be_discontinuous():
    theta = 0.4
    positive = make_anchor(feature=(math.cos(theta), math.sin(theta)), c2_sign=1,
                           position=1, source_group=OTHER_GROUP)
    negative = make_anchor(feature=(math.cos(theta), -math.sin(theta)), c2_sign=-1,
                           position=2, source_group=THIRD_GROUP)
    query_at = lambda angle: np.asarray([math.cos(angle), math.sin(angle)])
    kwargs = {"support_radius": 1.0, "evidence_regularizer": 0.1, "k": 1, "amplitude": 0.2}
    below = call_knn(gated_signed_knn_delta, query_at(-1e-7), [positive, negative], **kwargs)
    tie = call_knn(gated_signed_knn_delta, query_at(0.0), [positive, negative], **kwargs)
    above = call_knn(gated_signed_knn_delta, query_at(1e-7), [positive, negative], **kwargs)
    assert below < 0.0
    assert tie > 0.0  # deterministic tie breaks toward original index zero
    assert above > 0.0
    assert abs(below - above) > 0.1


@pytest.mark.parametrize("kwargs,expected", [
    ({"support_radius": 0.0, "evidence_regularizer": 0.1}, "support_radius"),
    ({"support_radius": 1.0, "evidence_regularizer": 0.0}, "evidence_regularizer"),
    ({"support_radius": 1.0, "evidence_regularizer": -1.0}, "evidence_regularizer"),
    ({"support_radius": 1.0, "evidence_regularizer": 0.1, "k": 0}, "k"),
    ({"support_radius": 1.0, "evidence_regularizer": 0.1, "amplitude": np.inf}, "amplitude"),
])
def test_gated_knn_rejects_invalid_method_parameters(kwargs, expected):
    anchor = make_anchor()
    with pytest.raises(ValueError, match=expected):
        call_knn(gated_signed_knn_delta, [1.0, 0.0], [anchor], **kwargs)


def test_c1_compact_rejects_invalid_regularizers():
    anchor = make_anchor()
    for value in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="evidence_regularizer"):
            call_c1(c1_compact_delta, [1.0, 0.0], [anchor], evidence_regularizer=value)


def test_30_category_heterogeneity_pooled_weighting_and_four_category_pitfall():
    categories = [f"synthetic-category-{i:02d}" for i in range(30)]
    assert len(categories) == len(set(categories)) == 30

    # A deliberately imbalanced synthetic inventory: 26 large/easy categories
    # and four small/hard ones. Values represent group-level nearest distances,
    # not independent patch observations.
    per_category = [np.full(1000, 0.20) for _ in range(26)]
    per_category += [np.full(10, 0.80) for _ in range(4)]
    pooled = np.concatenate(per_category)
    category_medians = np.asarray([np.median(x) for x in per_category])
    radius = 0.30
    class_coverage = np.asarray([np.mean(x <= radius) for x in per_category])

    assert np.median(pooled) == 0.20
    assert np.quantile(pooled, 0.95) == 0.20
    assert np.mean(category_medians) == pytest.approx(0.28)
    assert np.mean(class_coverage) == pytest.approx(26 / 30)
    assert np.mean(pooled <= radius) == pytest.approx(26000 / 26040)
    assert np.min(class_coverage) == 0.0 and np.max(class_coverage) == 1.0

    # A four-category-only illustrative radius fits the selected categories
    # but misses every synthetic category with the larger scale.
    four_selected = [np.full(100, value) for value in (0.12, 0.13, 0.14, 0.15)]
    four_only_radius = float(np.quantile(np.concatenate(four_selected), 0.95))
    all_class_groups = [np.full(100, four_only_radius) for _ in range(4)]
    all_class_groups += [np.full(100, 0.60) for _ in range(26)]
    selected_coverage = np.mean(np.concatenate(four_selected) <= four_only_radius)
    all_class_category_coverage = np.mean([
        np.mean(values <= four_only_radius) for values in all_class_groups
    ])
    assert selected_coverage == 1.0
    assert all_class_category_coverage == pytest.approx(4 / 30)


def test_synthetic_same_source_self_match_inflates_coverage():
    query_feature = np.asarray([1.0, 0.0])
    same_source_feature = np.asarray([1.0, 0.0])
    cross_source_feature = np.asarray([0.0, 1.0])
    all_refs = np.stack([same_source_feature, cross_source_feature])
    distances = cosine_distances(query_feature, all_refs)
    all_ref_coverage = bool(np.any(distances < 0.1))
    source_disjoint_coverage = bool(np.any(distances[1:] < 0.1))
    assert all_ref_coverage is True
    assert source_disjoint_coverage is False
