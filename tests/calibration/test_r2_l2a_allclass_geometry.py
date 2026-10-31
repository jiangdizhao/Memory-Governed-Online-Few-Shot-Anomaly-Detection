from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from experiments.calibration.r2_l2a_allclass_inventory import (
    EXPECTED_CATEGORY_COUNT,
    OfficialCatalogReceipt,
    PartitionOverlapError,
    RoleGroupReceipt,
    build_inventory_preflight,
    partition_preflight,
)
from experiments.calibration.r2_l2a_g1_geometry import (
    DEFAULT_MAX_PAIRWISE_ELEMENTS,
    DescriptorBatch,
    FeatureCacheReceipt,
    G1B_BASELINE_MEMORY_CONTRACT,
    G1B_SELECTION_RULE,
    R1_FIXED_SUPPORT_SHA256,
    GeometryContractError,
    R1_DESCRIPTOR_PROFILE,
    Top32SelectionReceipt,
    compare_source_disjoint,
    conditional_query_group_bootstrap,
    g1a_spatial_indices,
    g1b_identifiability_status,
    select_g1b_features,
    summarize_30_category_coverage,
    summarize_category_geometry,
    validate_feature_cache_receipt,
)


REPRESENTATION = hashlib.sha256(b"synthetic-r2-l2a-representation").hexdigest()
INVENTORY = hashlib.sha256(b"synthetic-inventory").hexdigest()
CACHE = hashlib.sha256(b"synthetic-cache-receipt").hexdigest()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_batch(category, features, groups, images, **kwargs):
    return DescriptorBatch(
        category_id=category,
        features=np.asarray(features),
        source_group_sha256=tuple(groups),
        image_sha256=tuple(images),
        representation_sha256=kwargs.pop("representation_sha256", REPRESENTATION),
        inventory_sha256=kwargs.pop("inventory_sha256", INVENTORY),
        cache_receipt_sha256=kwargs.pop("cache_receipt_sha256", CACHE),
        **kwargs,
    )


def unit(angle, dimension=2):
    if dimension == 2:
        return np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
    vector = np.zeros(dimension, dtype=np.float64)
    vector[0] = np.cos(angle)
    vector[1] = np.sin(angle)
    return vector


def brute_force(query, reference, radii):
    q = query.features.astype(np.float64)
    r = reference.features.astype(np.float64)
    nearest, fifth, nearest_index, fifth_index = [], [], [], []
    eligible_counts, local_counts, boundary_counts = [], [], []
    for row, (group, image) in enumerate(zip(query.source_group_sha256,
                                             query.image_sha256)):
        eligible = [i for i, (ref_group, ref_image) in enumerate(zip(
            reference.source_group_sha256, reference.image_sha256))
                    if ref_group != group and ref_image != image]
        distances = 1.0 - np.clip(r[eligible] @ q[row], -1.0, 1.0)
        order = np.lexsort((np.asarray(eligible), distances))
        distances = distances[order]
        indices = np.asarray(eligible, dtype=np.int64)[order]
        eligible_counts.append(len(eligible))
        nearest.append(float(distances[0]) if len(distances) else np.nan)
        fifth.append(float(distances[4]) if len(distances) >= 5 else np.nan)
        nearest_index.append(int(indices[0]) if len(indices) else -1)
        fifth_index.append(int(indices[4]) if len(indices) >= 5 else -1)
        local_counts.append([int(np.count_nonzero(distances < radius)) for radius in radii])
        boundary_counts.append([int(np.count_nonzero(distances == radius)) for radius in radii])
    return {
        "nearest": np.asarray(nearest), "fifth": np.asarray(fifth),
        "nearest_index": np.asarray(nearest_index),
        "fifth_index": np.asarray(fifth_index),
        "eligible_counts": np.asarray(eligible_counts),
        "local_counts": np.asarray(local_counts, dtype=np.int32),
        "boundary_counts": np.asarray(boundary_counts, dtype=np.int32),
    }


def make_catalog(n=30):
    return OfficialCatalogReceipt(
        source_uri="https://realiad4ad.github.io/Real-IAD/",
        source_sha256=digest("official-public-catalog-projection"),
        release_id="synthetic-catalog-fixture-v1",
        categories=tuple(f"category_{i:02d}" for i in range(n)),
        official_source_verified=True,
        non_label_bearing=True,
    )


def role_receipts(categories):
    receipts = []
    for category in categories:
        for role in ("A", "B", "C"):
            group = digest(f"{category}:{role}:source-group")
            receipts.append(RoleGroupReceipt(
                category_id=category,
                role=role,
                group_sha256=group,
                receipt_sha256=digest(f"{category}:{role}:receipt"),
                authorization_id="synthetic-fixture-only",
                independent_certificate_sha256=(digest("synthetic-c-certificate")
                                                if role == "C" else None),
            ))
    return receipts


def valid_cache_receipt(**changes):
    profile = R1_DESCRIPTOR_PROFILE
    fields = dict(
        cache_sha256=digest("cache"),
        inventory_sha256=digest("inventory"),
        profile_sha256=profile.sha256,
        encoder_fingerprint_sha256=profile.encoder_fingerprint_sha256,
        preprocess_fingerprint_sha256=profile.preprocess_fingerprint_sha256,
        native_scorer_profile_sha256=profile.native_scorer_profile_sha256,
        checkpoint_sha256=profile.checkpoint_sha256,
        hub_source_tree_sha256=profile.hub_source_tree_sha256,
        feature_shape=profile.feature_shape,
        feature_dtype=profile.feature_dtype,
        view=profile.view,
        l2_normalized=True,
        role="A",
        population="authorized_normal_reference_only",
        source_group_identity_contract="realid-offical-sample-key-v2",
    )
    fields.update(changes)
    return FeatureCacheReceipt(**fields)


def test_incomplete_inventory_lists_all_slots_and_never_becomes_a_four_class_universe():
    report = build_inventory_preflight(make_catalog(4))
    assert report["expected_category_count"] == 30
    assert report["listed_category_count"] == 4
    assert report["verified_category_count"] == 0
    assert report["complete_30_category_inventory"] is False
    assert len(report["categories"]) == 30
    assert sum(item["category_id"] is None for item in report["categories"]) == 26
    assert report["status"] == "PENDING_AUTHORIZATION_INCOMPLETE_OFFICIAL_CATALOG"


def test_inventory_records_protected_and_unavailable_category_access_explicitly():
    catalog = make_catalog()
    report = build_inventory_preflight(catalog, category_access_status={
        catalog.categories[0]: "PROTECTED_SEALED",
        catalog.categories[1]: "UNAVAILABLE",
    })
    assert report["categories"][0]["access_status"] == "PROTECTED_SEALED"
    assert report["categories"][1]["access_status"] == "UNAVAILABLE"
    with pytest.raises(ValueError, match="outside the supplied catalog"):
        build_inventory_preflight(catalog, category_access_status={
            "not-a-catalog-category": "PROTECTED_SEALED"})


def test_absent_catalog_has_thirty_explicit_unresolved_slots():
    report = build_inventory_preflight()
    assert len(report["categories"]) == EXPECTED_CATEGORY_COUNT
    assert all(row["category_id"] is None for row in report["categories"])
    assert all(row["access_status"] == "PENDING_AUTHORIZATION"
               for row in report["categories"])


def test_catalog_requires_official_non_label_bearing_source_and_exact_unique_names():
    with pytest.raises(ValueError, match="official catalog source"):
        build_inventory_preflight(replace(make_catalog(), official_source_verified=False))
    with pytest.raises(ValueError, match="category names only"):
        build_inventory_preflight(replace(make_catalog(), contains_image_paths=True))
    with pytest.raises(ValueError, match="duplicate"):
        build_inventory_preflight(replace(make_catalog(), categories=("one",) * 30))
    with pytest.raises(ValueError, match="approved official HTTPS"):
        build_inventory_preflight(replace(
            make_catalog(), source_uri="https://example.net/category-list"))


def test_a_b_c_partition_receipts_certify_disjointness_only_when_complete():
    catalog = make_catalog()
    receipts = role_receipts(catalog.categories)
    report = partition_preflight(
        catalog, receipts,
        verified_identity_receipt_sha256s=[r.receipt_sha256 for r in receipts],
        verified_c_certificate_sha256=digest("synthetic-c-certificate"),
    )
    assert report["status"] == "PARTITION_CERTIFIED_DISJOINT"
    assert report["certified_disjoint"] is True
    assert report["overlap_count"] == 0
    assert report["c_certification_status"] == "EXTERNAL_CERTIFICATE_VERIFIED_BY_CALLER"
    assert report["role_group_counts"][catalog.categories[0]] == {"A": 1, "B": 1, "C": 1}

    a_b_receipts = [r for r in role_receipts(catalog.categories) if r.role != "C"]
    pending = partition_preflight(
        catalog, a_b_receipts,
        verified_identity_receipt_sha256s=[r.receipt_sha256 for r in a_b_receipts])
    assert pending["certified_disjoint"] is False
    assert pending["c_certification_status"] == "PENDING_AUTHORIZATION"
    assert len(pending["missing_category_roles"]) == 30

    unverified = partition_preflight(catalog, receipts)
    assert unverified["status"] == "PENDING_AUTHORIZATION_C_CERTIFICATE_REVIEW"
    assert unverified["certified_disjoint"] is False

    c_verified_but_receipts_unreviewed = partition_preflight(
        catalog, receipts,
        verified_c_certificate_sha256=digest("synthetic-c-certificate"))
    assert c_verified_but_receipts_unreviewed["status"] == \
        "PENDING_EXTERNAL_RECEIPT_PROVENANCE_REVIEW"


def test_partition_overlap_and_protected_receipt_projection_fail_closed():
    catalog = make_catalog()
    receipts = role_receipts(catalog.categories)
    a = next(r for r in receipts if r.category_id == catalog.categories[0] and r.role == "A")
    b_index = next(i for i, r in enumerate(receipts)
                   if r.category_id == catalog.categories[0] and r.role == "B")
    receipts[b_index] = replace(receipts[b_index], group_sha256=a.group_sha256)
    with pytest.raises(PartitionOverlapError, match="overlaps roles"):
        partition_preflight(catalog, receipts)

    c_index = next(i for i, r in enumerate(role_receipts(catalog.categories))
                   if r.category_id == catalog.categories[0] and r.role == "C")
    contaminated = role_receipts(catalog.categories)
    contaminated[c_index] = replace(contaminated[c_index], contains_image_paths=True)
    with pytest.raises(ValueError, match="identity-only projection"):
        partition_preflight(catalog, contaminated)

    missing_cert = replace(next(r for r in role_receipts(catalog.categories)
                                if r.role == "C"), independent_certificate_sha256=None)
    with pytest.raises(ValueError, match="independent identity-certificate"):
        partition_preflight(catalog, [missing_cert])


def test_r1_descriptor_profile_matches_receipt_backed_native_geometry():
    profile = R1_DESCRIPTOR_PROFILE
    assert profile.model_variant.startswith("DINOv2 ViT-L/14")
    assert profile.checkpoint_file == "dinov2_vitl14_reg4_pretrain.pth"
    assert profile.hook_layers == (5, 11, 17, 23)
    assert profile.layer_fusion.startswith("concatenate four selected 1024D")
    assert profile.feature_shape == (784, 4096)
    assert profile.feature_dtype == "float32"
    assert profile.l2_normalized is True
    assert profile.view == "id"
    assert len(profile.source_hashes) == 5


def test_cache_compatibility_is_metadata_only_and_rejects_provenance_mismatch():
    receipt = valid_cache_receipt()
    pending = validate_feature_cache_receipt(receipt)
    assert pending["status"] == "PENDING_EXTERNAL_CACHE_AND_INVENTORY_RECEIPT_REVIEW"
    validated = validate_feature_cache_receipt(
        receipt, verified_cache_sha256=receipt.cache_sha256,
        verified_inventory_sha256=receipt.inventory_sha256)
    assert validated["status"] == "COMPATIBLE_EXTERNALLY_VERIFIED_METADATA_RECEIPT"
    for field, invalid in (
        ("preprocess_fingerprint_sha256", digest("wrong-preprocess")),
        ("checkpoint_sha256", digest("wrong-checkpoint")),
        ("feature_shape", (784, 768)),
        ("feature_dtype", "float16"),
        ("view", "camera-2"),
        ("role", "C"),
        ("population", "mixed_normal_and_defect"),
        ("contains_labels_or_masks", True),
        ("contains_private_feedback", True),
    ):
        with pytest.raises(GeometryContractError, match="incompatible feature cache"):
            validate_feature_cache_receipt(replace(receipt, **{field: invalid}))


def test_g1a_spatial_sampling_is_deterministic_and_has_49_frozen_positions():
    first = g1a_spatial_indices()
    second = g1a_spatial_indices()
    assert np.array_equal(first, second)
    assert len(first) == len(set(first.tolist())) == 49
    assert first[0] == 58  # row 2, column 2 on the frozen 28x28 grid
    assert first[-1] == 754  # row 26, column 26
    assert first.flags.writeable is False


def test_g1b_stays_not_identifiable_without_a_native_top32_receipt():
    assert g1b_identifiability_status(None) == "NOT_YET_IDENTIFIABLE"
    receipt = Top32SelectionReceipt(
        query_image_sha256=digest("image"),
        scorer_profile_sha256=R1_DESCRIPTOR_PROFILE.native_scorer_profile_sha256,
        fixed_support_sha256=R1_FIXED_SUPPORT_SHA256,
        score_state_memory_sha256=digest("frozen-baseline-memory"),
        source_receipt_sha256=digest("native-c0-index-receipt"),
        baseline_memory_contract=G1B_BASELINE_MEMORY_CONTRACT,
        baseline_memory_certificate_sha256=digest("certified-r1-m0"),
        selection_rule=G1B_SELECTION_RULE,
        selected_indices=tuple(range(32)),
    )
    assert g1b_identifiability_status(receipt) == "NOT_YET_IDENTIFIABLE"
    assert g1b_identifiability_status(
        receipt, verified_receipt_sha256=receipt.source_receipt_sha256,
        verified_baseline_certificate_sha256=receipt.baseline_memory_certificate_sha256
    ) == "IDENTIFIABLE_FROM_EXTERNALLY_VERIFIED_FROZEN_NATIVE_C0_RECEIPT"
    features = np.zeros((784, 4096), dtype=np.float32)
    features[:, 0] = 1.0
    selected = select_g1b_features(
        features, receipt, verified_receipt_sha256=receipt.source_receipt_sha256,
        verified_baseline_certificate_sha256=receipt.baseline_memory_certificate_sha256)
    assert selected.shape == (32, 4096)
    assert selected.flags.writeable is False
    with pytest.raises(GeometryContractError, match="32 unique indices"):
        g1b_identifiability_status(
            replace(receipt, selected_indices=tuple(range(31))),
            verified_receipt_sha256=receipt.source_receipt_sha256,
            verified_baseline_certificate_sha256=receipt.baseline_memory_certificate_sha256)
    with pytest.raises(GeometryContractError, match="incompatible native scorer"):
        g1b_identifiability_status(
            replace(receipt, scorer_profile_sha256=digest("wrong-scorer")),
            verified_receipt_sha256=receipt.source_receipt_sha256,
            verified_baseline_certificate_sha256=receipt.baseline_memory_certificate_sha256)


def test_same_source_and_same_image_rows_are_excluded_before_neighbor_selection():
    query = make_batch("cat", [[1.0, 0.0]], [digest("g1")], [digest("img1")])
    reference = make_batch("cat", [[1.0, 0.0], [0.0, 1.0]],
                           [digest("g1"), digest("g2")],
                           [digest("img1"), digest("img2")])
    stats = compare_source_disjoint(query, reference, radii=[0.0, 0.5, 2.0],
                                    query_block_size=1, reference_block_size=1,
                                    max_pairwise_elements=1)
    assert stats.eligible_reference_count.tolist() == [1]
    assert stats.nearest_distance[0] == pytest.approx(1.0)
    assert stats.nearest_reference_index.tolist() == [1]
    assert stats.metadata["dense_query_by_reference_matrix_allocated"] is False


def test_no_eligible_references_and_fewer_than_five_have_explicit_status():
    query = make_batch("cat", [[1.0, 0.0]], [digest("g0")], [digest("q")])
    no_ref = make_batch("cat", [[1.0, 0.0]], [digest("g0")], [digest("q")])
    no_stats = compare_source_disjoint(query, no_ref, radii=[0.0, 1.0, 2.0])
    no_report = summarize_category_geometry(query, no_ref, no_stats, bootstrap_resamples=8)
    assert np.isnan(no_stats.nearest_distance[0])
    assert np.isnan(no_stats.fifth_nearest_distance[0])
    assert no_report["queries_without_eligible_references"] == 1
    assert no_report["uncertainty"]["status"] == "CI_UNDEFINED_INSUFFICIENT_QUERY_SOURCE_GROUPS"

    groups = [digest(f"r{i}") for i in range(4)]
    refs = make_batch("cat", [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
                      groups, [digest(f"ri{i}") for i in range(4)])
    few = compare_source_disjoint(query, refs, radii=[0.0, 1.0, 2.0])
    assert few.eligible_reference_count.tolist() == [4]
    assert np.isfinite(few.nearest_distance[0])
    assert np.isnan(few.fifth_nearest_distance[0])
    assert few.fifth_reference_index.tolist() == [-1]


def test_nearest_ties_are_broken_by_original_reference_index():
    query = make_batch("cat", [[1.0, 0.0]], [digest("qg")], [digest("qi")])
    refs = make_batch("cat", [[0.0, 1.0], [0.0, -1.0]],
                      [digest("r0"), digest("r1")], [digest("i0"), digest("i1")])
    stats = compare_source_disjoint(query, refs, radii=[0.0, 1.0, 2.0])
    assert stats.nearest_distance[0] == 1.0
    assert stats.nearest_reference_index[0] == 0
    assert stats.local_reference_count_by_radius[0].tolist() == [0, 0, 2]
    assert stats.boundary_equal_count_by_radius[0].tolist() == [0, 2, 0]


def test_strict_radius_boundary_excludes_equal_and_distance_two_boundary():
    query = make_batch("cat", [[1.0, 0.0]], [digest("qg")], [digest("qi")])
    refs = make_batch("cat", [[0.0, 1.0], [-1.0, 0.0]],
                      [digest("r1"), digest("r2")], [digest("i1"), digest("i2")])
    stats = compare_source_disjoint(query, refs, radii=[0.0, 1.0, 2.0])
    report = summarize_category_geometry(query, refs, stats, bootstrap_resamples=8)
    curve = report["coverage_radius_curve"]
    assert stats.nearest_distance[0] == 1.0
    assert curve[1]["nearest_distance_lt_radius_fraction"] == 0.0
    assert curve[1]["exact_boundary_equality_count"] == 1
    assert curve[2]["nearest_distance_lt_radius_fraction"] == 1.0
    assert curve[2]["exact_boundary_equality_count"] == 1
    # The antipodal distance is exactly two, so it is outside the strict R=2 support.
    assert stats.local_reference_count_by_radius[0, 2] == 1


def test_fifth_neighbor_is_exact_and_blockwise_results_match_brute_force():
    rng = np.random.default_rng(20261031)
    raw_q = rng.normal(size=(9, 12))
    raw_r = rng.normal(size=(17, 12))
    qx = raw_q / np.linalg.norm(raw_q, axis=1, keepdims=True)
    rx = raw_r / np.linalg.norm(raw_r, axis=1, keepdims=True)
    qgroups = [digest(f"qg{i}") for i in range(len(qx))]
    rgroups = [digest(f"rg{i}") for i in range(len(rx))]
    query = make_batch("cat", qx.astype(np.float32), qgroups,
                       [digest(f"qi{i}") for i in range(len(qx))])
    refs = make_batch("cat", rx.astype(np.float32), rgroups,
                      [digest(f"ri{i}") for i in range(len(rx))])
    radii = np.linspace(0.0, 2.0, 17)
    stats = compare_source_disjoint(query, refs, radii=radii,
                                    query_block_size=3, reference_block_size=4,
                                    max_pairwise_elements=12)
    expected = brute_force(query, refs, radii)
    assert np.allclose(stats.nearest_distance, expected["nearest"], atol=1e-12)
    assert np.allclose(stats.fifth_nearest_distance, expected["fifth"], atol=1e-12)
    assert np.array_equal(stats.nearest_reference_index, expected["nearest_index"])
    assert np.array_equal(stats.fifth_reference_index, expected["fifth_index"])
    assert np.array_equal(stats.eligible_reference_count, expected["eligible_counts"])
    assert np.array_equal(stats.local_reference_count_by_radius, expected["local_counts"])
    assert np.array_equal(stats.boundary_equal_count_by_radius, expected["boundary_counts"])
    assert np.all(stats.local_reference_count_by_radius[:, -1] == len(rx))
    report = summarize_category_geometry(query, refs, stats, bootstrap_resamples=16, bootstrap_seed=9)
    assert report["coverage_radius_curve"][-1]["at_least_five_local_references_fraction"] == 1.0
    assert report["coverage_radius_curve"][-1]["zero_local_references_fraction"] == 0.0
    assert stats.block_pairwise_elements == 12
    assert DEFAULT_MAX_PAIRWISE_ELEMENTS == 65_536


def test_exact_streaming_parity_on_30_heterogeneous_categories():
    rng = np.random.default_rng(44)
    categories = [f"cat_{i:02d}" for i in range(30)]
    reports = []
    for i, category in enumerate(categories):
        n_query = 1 + (i % 5)
        n_ref = 1 + (i % 8)
        base = unit(i * 0.13, dimension=8)
        offset = 0.01 + i * 0.004
        qx = np.stack([base + rng.normal(scale=offset, size=8)
                       for _ in range(n_query)])
        rx = np.stack([base + rng.normal(scale=offset * 2, size=8)
                       for _ in range(n_ref)])
        qx /= np.linalg.norm(qx, axis=1, keepdims=True)
        rx /= np.linalg.norm(rx, axis=1, keepdims=True)
        query = make_batch(category, qx.astype(np.float32),
                           [digest(f"{category}:qg:{j}") for j in range(n_query)],
                           [digest(f"{category}:qi:{j}") for j in range(n_query)])
        refs = make_batch(category, rx.astype(np.float32),
                          [digest(f"{category}:rg:{j}") for j in range(n_ref)],
                          [digest(f"{category}:ri:{j}") for j in range(n_ref)])
        radii = [0.0, 0.5, 1.0, 1.5, 2.0]
        stats = compare_source_disjoint(query, refs, radii=radii,
                                        query_block_size=2, reference_block_size=3,
                                        max_pairwise_elements=6)
        expected = brute_force(query, refs, radii)
        assert np.allclose(stats.nearest_distance, expected["nearest"], atol=1e-7)
        assert np.allclose(stats.fifth_nearest_distance, expected["fifth"], equal_nan=True)
        assert np.array_equal(stats.local_reference_count_by_radius, expected["local_counts"])
        reports.append(summarize_category_geometry(query, refs, stats, bootstrap_resamples=16,
                                                   bootstrap_seed=i))
    assert len(reports) == 30
    assert len({r["category_id"] for r in reports}) == 30


def test_conditional_source_group_bootstrap_is_deterministic_and_sparse_ci_is_undefined():
    query = make_batch("cat", [[1.0, 0.0], [0.9, 0.435889894], [0.8, 0.6]],
                       [digest("g1"), digest("g1"), digest("g2")],
                       [digest("i1"), digest("i2"), digest("i3")])
    refs = make_batch("cat", [[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.6, 0.8],
                               [-0.6, 0.8]],
                      [digest(f"r{i}") for i in range(5)],
                      [digest(f"ri{i}") for i in range(5)])
    stats = compare_source_disjoint(query, refs, radii=[0.0, 0.5, 1.0, 2.0])
    one = conditional_query_group_bootstrap(stats, n_resamples=128, seed=77)
    two = conditional_query_group_bootstrap(stats, n_resamples=128, seed=77)
    assert one == two
    assert one["status"] == "DESCRIPTIVE_CLUSTER_BOOTSTRAP_CONDITIONAL_ON_REFERENCE"
    assert one["reference_group_resampling"] is False
    assert one["independent_query_source_groups"] == 2

    sparse_query = make_batch("cat", [[1.0, 0.0], [0.0, 1.0]],
                              [digest("only-one-group")] * 2,
                              [digest("si1"), digest("si2")])
    sparse_stats = compare_source_disjoint(sparse_query, refs, radii=[0.0, 1.0, 2.0])
    sparse_ci = conditional_query_group_bootstrap(sparse_stats, n_resamples=16, seed=0)
    assert sparse_ci["status"] == "CI_UNDEFINED_INSUFFICIENT_QUERY_SOURCE_GROUPS"
    assert sparse_ci["coverage_ci"] is None


def test_macro_pooled_and_worst_category_summaries_preserve_heterogeneity():
    categories = [f"cat_{i:02d}" for i in range(30)]
    reports = []
    for i, category in enumerate(categories):
        count = 1_000 if i < 26 else 10
        coverage = 1.0 if i < 26 else 0.0
        reports.append({
            "category_id": category,
            "query_descriptor_count": count,
            "coverage_radius_curve": [{
                "radius": 0.3,
                "n_nearest_strictly_inside": int(count * coverage),
                "nearest_distance_lt_radius_fraction": coverage,
            }],
        })
    summary = summarize_30_category_coverage(reports, categories, radius=0.3)
    assert summary["category_equal_macro_coverage"] == pytest.approx(26 / 30)
    assert summary["pooled_coverage"]["value"] == pytest.approx(26_000 / 26_040)
    assert summary["pooled_coverage"]["weighting_unit"] == "query_descriptor"
    assert summary["worst_category"]["coverage"] == 0.0
    assert len(summary["lower_tail"]["category_ids"]) == 3
    assert summary["lower_tail"]["mean_coverage"] == 0.0

    incomplete = summarize_30_category_coverage(reports[:-1], categories, radius=0.3)
    assert incomplete["status"] == "INCOMPLETE_30_CATEGORY_DENOMINATOR"
    assert incomplete["pooled_coverage"] is None
    assert incomplete["category_equal_macro_coverage"] is None


@pytest.mark.parametrize("mutator, message", [
    (lambda b: replace(b, features=np.ones((1, 3), dtype=np.float32)), "normalized"),
    (lambda b: replace(b, features=np.asarray([[np.nan, 0.0]], dtype=np.float32)), "finite"),
    (lambda b: replace(b, features=np.asarray([[1.0, 0.0]], dtype=np.float16)), "dtype"),
    (lambda b: replace(b, source_group_sha256=("bad",)), "source-group"),
    (lambda b: replace(b, role="C"), "role"),
    (lambda b: replace(b, contains_labels_or_masks=True), "forbidden information"),
])
def test_invalid_feature_and_role_inputs_fail_closed(mutator, message):
    batch = make_batch("cat", [[1.0, 0.0]], [digest("g")], [digest("i")])
    with pytest.raises(GeometryContractError, match=message):
        compare_source_disjoint(mutator(batch), batch, radii=[0.0, 1.0, 2.0])


def test_dimension_profile_and_resource_bounds_are_enforced():
    q = make_batch("cat", [[1.0, 0.0]], [digest("qg")], [digest("qi")])
    r3 = make_batch("cat", [[1.0, 0.0, 0.0]], [digest("rg")], [digest("ri")])
    with pytest.raises(GeometryContractError, match="dimensions differ"):
        compare_source_disjoint(q, r3)
    with pytest.raises(GeometryContractError, match="resource cap"):
        compare_source_disjoint(q, q, query_block_size=8, reference_block_size=8,
                                max_pairwise_elements=16)
    with pytest.raises(GeometryContractError, match="radius grid"):
        compare_source_disjoint(q, q, radii=[0.0, 1.0])
