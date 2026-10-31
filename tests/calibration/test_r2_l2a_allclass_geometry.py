from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from experiments.calibration.r2_l2a_allclass_inventory import (
    AReferenceAvailabilityReceipt,
    CANONICAL_CATEGORY_IDS,
    CATALOG_EXTRACTION_ALGORITHM,
    EXPECTED_CATEGORY_COUNT,
    PINNED_COMMIT,
    PINNED_CONFIG_VERSION,
    PINNED_RAW_SOURCE_URI,
    PINNED_SOURCE_PATH,
    PINNED_SOURCE_SHA256,
    PINNED_SOURCE_URI,
    OfficialCatalogReceipt,
    PartitionOverlapError,
    RolePolicyEntry,
    RoleGroupReceipt,
    build_inventory_preflight,
    extract_category_catalog_from_config_bytes,
    make_role_policy as build_role_policy,
    official_catalog_from_pinned_source_bytes,
    partition_preflight,
)
from experiments.calibration.r2_l2a_g1_geometry import (
    DEFAULT_MAX_PAIRWISE_ELEMENTS,
    DescriptorBatch,
    FeatureCacheReceipt,
    G1B_BASELINE_MEMORY_CONTRACT,
    G1B_SELECTION_RULE,
    R1_FIXED_SUPPORT_CATEGORY_ID,
    R1_FIXED_SUPPORT_SEED,
    R1_FIXED_SUPPORT_SHA256,
    R1_FIXED_SUPPORT_DIGEST_ALGORITHM,
    GeometryContractError,
    R1_DESCRIPTOR_PROFILE,
    Top32SelectionReceipt,
    category_fixed_support_receipt_sha256,
    category_m0_scorer_state_sha256,
    compare_source_disjoint as _compare_source_disjoint,
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
A_AUTHORIZATION = hashlib.sha256(b"synthetic-a-authorization").hexdigest()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_batch(category, features, groups, images, **kwargs):
    return DescriptorBatch(
        category_id=category,
        features=np.asarray(features),
        source_group_sha256=tuple(groups),
        image_sha256=tuple(images),
        representation_sha256=kwargs.pop("representation_sha256", REPRESENTATION),
        access_status=kwargs.pop("access_status", "AUTHORIZED_FOR_G1_REFERENCE_POPULATION"),
        a_reference_availability_receipt_sha256=kwargs.pop(
            "a_reference_availability_receipt_sha256", A_AUTHORIZATION),
        inventory_sha256=kwargs.pop("inventory_sha256", INVENTORY),
        cache_receipt_sha256=kwargs.pop("cache_receipt_sha256", CACHE),
        **kwargs,
    )


def compare_source_disjoint(query, reference, **kwargs):
    """Synthetic test adapter supplies only the explicitly reviewed fixture receipt."""
    verified = {query.category_id: query.a_reference_availability_receipt_sha256}
    kwargs.setdefault("verified_a_reference_availability_receipts", verified)
    return _compare_source_disjoint(query, reference, **kwargs)


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
        source_uri=PINNED_SOURCE_URI,
        source_sha256=PINNED_SOURCE_SHA256,
        release_id=PINNED_CONFIG_VERSION,
        categories=CANONICAL_CATEGORY_IDS[:n],
        official_source_verified=True,
        non_label_bearing=True,
        source_commit=PINNED_COMMIT,
        source_path=PINNED_SOURCE_PATH,
        dataset_config_version=PINNED_CONFIG_VERSION,
        extraction_algorithm=CATALOG_EXTRACTION_ALGORITHM,
    )


def make_role_policy(categories, dispositions=None):
    dispositions = dispositions or {}
    entries = []
    for category in categories:
        for role in ("A", "B", "C"):
            disposition = dispositions.get((category, role), "REQUIRED")
            entries.append(RolePolicyEntry(
                category_id=category,
                role=role,
                disposition=disposition,
                rationale=("synthetic design rationale" if disposition == "NOT_APPLICABLE"
                           else None),
            ))
    return build_role_policy("synthetic-r2-l2a-policy", "v1", entries)


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


def valid_top32_receipt(**changes):
    fields = dict(
        query_image_sha256=digest("image"),
        scorer_profile_sha256=R1_DESCRIPTOR_PROFILE.native_scorer_profile_sha256,
        fixed_support_sha256=digest("category-specific-structured-bank"),
        score_state_memory_sha256=digest("frozen-baseline-memory"),
        source_receipt_sha256=digest("native-c0-index-receipt"),
        baseline_memory_contract=G1B_BASELINE_MEMORY_CONTRACT,
        baseline_memory_certificate_sha256=digest("certified-r1-m0"),
        selection_rule=G1B_SELECTION_RULE,
        selected_indices=tuple(range(32)),
        category_id=CANONICAL_CATEGORY_IDS[0],
        representation_sha256=R1_DESCRIPTOR_PROFILE.representation_sha256,
        fixed_support_category_id=CANONICAL_CATEGORY_IDS[0],
        fixed_support_seed=0,
        fixed_support_digest_algorithm=R1_FIXED_SUPPORT_DIGEST_ALGORITHM,
        fixed_support_receipt_sha256=digest("placeholder-bank-receipt"),
        m0_scorer_state_sha256=digest("placeholder-m0-scorer-state"),
    )
    fields.update(changes)
    if "fixed_support_receipt_sha256" not in changes:
        fields["fixed_support_receipt_sha256"] = category_fixed_support_receipt_sha256(
            fields["fixed_support_category_id"],
            fields["fixed_support_seed"],
            fields["fixed_support_digest_algorithm"],
            fields["fixed_support_sha256"],
        )
    if "m0_scorer_state_sha256" not in changes:
        fields["m0_scorer_state_sha256"] = category_m0_scorer_state_sha256(
            category_id=fields["category_id"],
            representation_sha256=fields["representation_sha256"],
            scorer_profile_sha256=fields["scorer_profile_sha256"],
            fixed_support_receipt_sha256=fields["fixed_support_receipt_sha256"],
            score_state_memory_sha256=fields["score_state_memory_sha256"],
            baseline_memory_contract=fields["baseline_memory_contract"],
            baseline_memory_certificate_sha256=fields["baseline_memory_certificate_sha256"],
        )
    return Top32SelectionReceipt(**fields)


def test_incomplete_inventory_lists_all_slots_and_never_becomes_a_four_class_universe():
    report = build_inventory_preflight()
    assert report["expected_category_count"] == 30
    assert report["listed_category_count"] == 0
    assert report["verified_category_count"] == 0
    assert report["complete_30_category_inventory"] is False
    assert len(report["categories"]) == 30
    assert sum(item["category_id"] is None for item in report["categories"]) == 30
    assert report["status"] == "PENDING_AUTHORIZATION_INCOMPLETE_OFFICIAL_CATALOG"
    with pytest.raises(ValueError, match="exactly 30 categories"):
        build_inventory_preflight(make_catalog(4))


def test_official_catalog_parser_projects_only_matching_train_test_category_ids():
    category_lines = "".join(
        f"      - data/Real-IAD/realiad_jsons/{category}.json\n"
        for category in CANONICAL_CATEGORY_IDS)
    fixture = (
        "version: v1.0.0\n"
        "dataset:\n"
        "  train:\n"
        "    meta_file:\n" + category_lines +
        "  test:\n"
        "    meta_file:\n" + category_lines +
        "    image_dir: data/Real-IAD/images\n"
    ).encode("utf-8")
    version, train, test = extract_category_catalog_from_config_bytes(fixture)
    assert version == "v1.0.0"
    assert train == test == CANONICAL_CATEGORY_IDS
    assert "data/Real-IAD" not in repr((version, train, test))


def test_official_catalog_parser_rejects_duplicate_and_inconsistent_source_lists():
    base_lines = "".join(
        f"      - data/Real-IAD/realiad_jsons/{category}.json\n"
        for category in CANONICAL_CATEGORY_IDS)
    duplicate = base_lines.replace(
        f"      - data/Real-IAD/realiad_jsons/{CANONICAL_CATEGORY_IDS[-1]}.json\n",
        f"      - data/Real-IAD/realiad_jsons/{CANONICAL_CATEGORY_IDS[0]}.json\n")
    duplicate_fixture = (
        "version: v1.0.0\ndataset:\n  train:\n    meta_file:\n" + duplicate
        + "  test:\n    meta_file:\n" + base_lines).encode("utf-8")
    with pytest.raises(ValueError, match="duplicate category"):
        extract_category_catalog_from_config_bytes(duplicate_fixture)
    different_test = base_lines.replace(
        f"      - data/Real-IAD/realiad_jsons/{CANONICAL_CATEGORY_IDS[-1]}.json\n",
        "      - data/Real-IAD/realiad_jsons/unknown_category.json\n")
    inconsistent_fixture = (
        "version: v1.0.0\ndataset:\n  train:\n    meta_file:\n" + base_lines
        + "  test:\n    meta_file:\n" + different_test).encode("utf-8")
    with pytest.raises(ValueError, match="differ|cross-check"):
        extract_category_catalog_from_config_bytes(inconsistent_fixture)


def test_pinned_catalog_factory_rejects_modified_bytes_and_unpinned_source():
    with pytest.raises(ValueError, match="pinned official Tencent"):
        official_catalog_from_pinned_source_bytes(
            b"synthetic", source_uri="https://github.com/somewhere/config.yaml")
    with pytest.raises(ValueError, match="digest mismatch"):
        official_catalog_from_pinned_source_bytes(b"synthetic")
    assert PINNED_RAW_SOURCE_URI.startswith("https://raw.githubusercontent.com/")


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


def test_complete_inventory_status_is_public_catalog_verification_only():
    catalog = make_catalog()
    report = build_inventory_preflight(catalog)
    assert report["status"] == "VERIFIED_30_CATEGORY_PUBLIC_INVENTORY"
    assert report["verified_category_count"] == 30
    assert tuple(row["category_id"] for row in report["categories"]) == CANONICAL_CATEGORY_IDS
    assert all(row["access_status"] == "PENDING_AUTHORIZATION"
               for row in report["categories"])
    assert report["catalog_source"]["source_commit"] == PINNED_COMMIT
    assert report["catalog_source"]["source_sha256"] == PINNED_SOURCE_SHA256


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
    with pytest.raises(ValueError, match="exact pinned official"):
        build_inventory_preflight(replace(
            make_catalog(), source_uri="https://example.net/category-list"))


def test_a_b_c_partition_receipts_certify_disjointness_only_when_complete():
    catalog = make_catalog()
    receipts = role_receipts(catalog.categories)
    policy = make_role_policy(catalog.categories)
    report = partition_preflight(
        catalog, receipts,
        role_policy=policy,
        verified_role_policy_sha256s=[policy.policy_sha256],
        verified_identity_receipt_sha256s=[r.receipt_sha256 for r in receipts],
        verified_c_certificate_sha256=digest("synthetic-c-certificate"),
    )
    assert report["status"] == "PARTITION_CERTIFIED_DISJOINT"
    assert report["certified_disjoint"] is True
    assert report["role_policy_declared"] is True
    assert report["role_policy_externally_verified"] is True
    assert report["role_identity_receipts_complete"] is True
    assert report["authorized_a_data_available"] is False
    assert report["g1_geometry_complete"] is False
    assert report["overlap_count"] == 0
    assert report["c_certification_status"] == "EXTERNAL_CERTIFICATE_VERIFIED_BY_CALLER"
    assert report["role_group_counts"][catalog.categories[0]] == {"A": 1, "B": 1, "C": 1}

    a_b_receipts = [r for r in role_receipts(catalog.categories) if r.role != "C"]
    pending = partition_preflight(
        catalog, a_b_receipts,
        role_policy=policy,
        verified_role_policy_sha256s=[policy.policy_sha256],
        verified_identity_receipt_sha256s=[r.receipt_sha256 for r in a_b_receipts])
    assert pending["certified_disjoint"] is False
    assert pending["c_certification_status"] == "PENDING_AUTHORIZATION"
    assert len(pending["missing_category_roles"]) == 30

    unverified = partition_preflight(catalog, receipts, role_policy=policy)
    assert unverified["status"] == "PENDING_EXTERNAL_ROLE_POLICY_REVIEW"
    assert unverified["certified_disjoint"] is False

    c_verified_but_receipts_unreviewed = partition_preflight(
        catalog, receipts,
        role_policy=policy,
        verified_role_policy_sha256s=[policy.policy_sha256],
        verified_c_certificate_sha256=digest("synthetic-c-certificate"))
    assert c_verified_but_receipts_unreviewed["status"] == \
        "PENDING_EXTERNAL_RECEIPT_PROVENANCE_REVIEW"
    assert c_verified_but_receipts_unreviewed["role_disjointness_externally_certified"] is False


def test_partition_requires_explicit_externally_reviewed_policy_for_not_applicable_roles():
    catalog = make_catalog()
    categories = catalog.categories
    disposition = {(category, "C"): "NOT_APPLICABLE" for category in categories}
    policy = make_role_policy(categories, disposition)
    a_b_receipts = [r for r in role_receipts(categories) if r.role != "C"]
    missing_policy = partition_preflight(
        catalog, a_b_receipts, role_policy=policy,
        verified_identity_receipt_sha256s=[r.receipt_sha256 for r in a_b_receipts])
    assert missing_policy["role_policy_externally_verified"] is False
    assert missing_policy["role_identity_receipts_complete"] is False
    verified = partition_preflight(
        catalog, a_b_receipts, role_policy=policy,
        verified_role_policy_sha256s=[policy.policy_sha256],
        verified_identity_receipt_sha256s=[r.receipt_sha256 for r in a_b_receipts])
    assert verified["role_identity_receipts_complete"] is True
    assert verified["c_certification_status"] == "NOT_REQUIRED_BY_EXTERNALLY_VERIFIED_POLICY"
    assert verified["a_reference_authorization_by_category"][categories[0]] == \
        "PENDING_AUTHORIZATION"
    with pytest.raises(ValueError, match="explicitly cover every category"):
        incomplete = build_role_policy("incomplete", "v1", policy.entries[:-1])
        partition_preflight(catalog, role_policy=incomplete)
    with pytest.raises(ValueError, match="despite NOT_APPLICABLE"):
        c_receipt = next(r for r in role_receipts(categories) if r.role == "C")
        partition_preflight(catalog, [c_receipt], role_policy=policy)
    with pytest.raises(ValueError, match="genuine written rationale"):
        bad = make_role_policy(categories, disposition)
        bad_entries = tuple(replace(entry, rationale=None) if entry.role == "C" else entry
                            for entry in bad.entries)
        invalid = build_role_policy("bad", "v1", bad_entries)
        partition_preflight(catalog, role_policy=invalid)


def test_absent_partition_policy_is_pending_and_never_treated_as_empty_roles():
    catalog = make_catalog()
    report = partition_preflight(catalog)
    assert report["role_policy_declared"] is False
    assert report["role_policy_externally_verified"] is False
    assert report["role_identity_receipts_complete"] is False
    assert report["role_disjointness_externally_certified"] is False
    assert report["authorized_a_data_available"] is False
    assert report["g1_geometry_complete"] is False
    assert len(report["pending_policy_category_roles"]) == 90


def test_verified_a_not_applicable_policy_keeps_category_unavailable_for_g1():
    catalog = make_catalog()
    category = catalog.categories[0]
    policy = make_role_policy(catalog.categories, {(category, "A"): "NOT_APPLICABLE"})
    receipts = [r for r in role_receipts(catalog.categories)
                if not (r.category_id == category and r.role == "A")]
    report = partition_preflight(
        catalog, receipts, role_policy=policy,
        verified_role_policy_sha256s=[policy.policy_sha256],
        verified_identity_receipt_sha256s=[r.receipt_sha256 for r in receipts],
        verified_c_certificate_sha256=digest("synthetic-c-certificate"),
    )
    assert report["a_reference_authorization_by_category"][category] == "NOT_AUTHORIZED_FOR_G1"
    assert report["authorized_a_data_available"] is False
    assert report["role_identity_receipts_complete"] is True


def test_a_reference_authorization_requires_external_count_only_receipt():
    catalog = make_catalog()
    receipts = role_receipts(catalog.categories)
    policy = make_role_policy(catalog.categories)
    category = catalog.categories[0]
    a_receipt = AReferenceAvailabilityReceipt(
        category_id=category,
        authorized_normal_reference_image_count=4,
        authorized_source_group_count=1,
        receipt_sha256=digest("a-availability"),
        authorization_id="synthetic-review",
    )
    report = partition_preflight(
        catalog, receipts, role_policy=policy,
        a_reference_receipts=[a_receipt],
        verified_role_policy_sha256s=[policy.policy_sha256],
        verified_identity_receipt_sha256s=[r.receipt_sha256 for r in receipts],
        verified_c_certificate_sha256=digest("synthetic-c-certificate"),
        verified_a_reference_receipt_sha256s=[a_receipt.receipt_sha256],
    )
    assert report["authorized_a_data_available"] is True
    assert report["a_reference_authorization_by_category"][category] == \
        "AUTHORIZED_FOR_G1_REFERENCE_POPULATION"
    assert report["g1_geometry_complete"] is False
    with pytest.raises(ValueError, match="normals only"):
        partition_preflight(
            catalog, receipts, role_policy=policy,
            a_reference_receipts=[replace(a_receipt, population="mixed")])


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
    other_class = replace(
        profile,
        profile_id="r1_other_class",
        fixed_support_sha256=digest("other-bank"),
        native_scorer_profile_sha256=digest("other-scorer-profile"),
        source_hashes=(("category_specific_runner.py", digest("runner")),),
    )
    assert other_class.representation_sha256 == profile.representation_sha256
    assert other_class.encoder_fingerprint_sha256 == profile.encoder_fingerprint_sha256
    assert other_class.preprocess_fingerprint_sha256 == profile.preprocess_fingerprint_sha256
    assert replace(profile, preprocess_fingerprint_sha256=digest("changed-preprocess")).representation_sha256 \
        != profile.representation_sha256


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
    receipt = valid_top32_receipt()
    assert g1b_identifiability_status(receipt) == "NOT_YET_IDENTIFIABLE"
    assert g1b_identifiability_status(
        receipt,
        verified_receipt_sha256=receipt.source_receipt_sha256,
        verified_baseline_certificate_sha256=receipt.baseline_memory_certificate_sha256,
    ) == "NOT_YET_IDENTIFIABLE"
    assert g1b_identifiability_status(
        receipt, verified_receipt_sha256=receipt.source_receipt_sha256,
        verified_fixed_support_receipt_sha256=receipt.fixed_support_receipt_sha256,
        verified_baseline_certificate_sha256=receipt.baseline_memory_certificate_sha256
    ) == "IDENTIFIABLE_FROM_EXTERNALLY_VERIFIED_CATEGORY_NATIVE_C0_RECEIPT"
    features = np.zeros((784, 4096), dtype=np.float32)
    features[:, 0] = 1.0
    selected = select_g1b_features(
        features, receipt, verified_receipt_sha256=receipt.source_receipt_sha256,
        verified_fixed_support_receipt_sha256=receipt.fixed_support_receipt_sha256,
        verified_baseline_certificate_sha256=receipt.baseline_memory_certificate_sha256)
    assert selected.shape == (32, 4096)
    assert selected.flags.writeable is False
    with pytest.raises(GeometryContractError, match="32 unique indices"):
        g1b_identifiability_status(
            replace(receipt, selected_indices=tuple(range(31))),
            verified_receipt_sha256=receipt.source_receipt_sha256,
            verified_fixed_support_receipt_sha256=receipt.fixed_support_receipt_sha256,
            verified_baseline_certificate_sha256=receipt.baseline_memory_certificate_sha256)
    with pytest.raises(GeometryContractError, match="incompatible native scorer"):
        g1b_identifiability_status(
            replace(receipt, scorer_profile_sha256=digest("wrong-scorer")),
            verified_receipt_sha256=receipt.source_receipt_sha256,
            verified_fixed_support_receipt_sha256=receipt.fixed_support_receipt_sha256,
            verified_baseline_certificate_sha256=receipt.baseline_memory_certificate_sha256)


def test_g1b_allows_category_specific_banks_but_requires_matching_provenance():
    other_category = CANONICAL_CATEGORY_IDS[1]
    receipt = valid_top32_receipt(
        category_id=other_category,
        fixed_support_category_id=other_category,
        fixed_support_sha256=digest("not-the-historical-switch-bank"),
        fixed_support_seed=17,
    )
    assert g1b_identifiability_status(receipt) == "NOT_YET_IDENTIFIABLE"
    assert receipt.fixed_support_sha256 != R1_FIXED_SUPPORT_SHA256
    assert g1b_identifiability_status(
        receipt,
        verified_receipt_sha256=receipt.source_receipt_sha256,
        verified_fixed_support_receipt_sha256=receipt.fixed_support_receipt_sha256,
        verified_baseline_certificate_sha256=receipt.baseline_memory_certificate_sha256,
    ) == "IDENTIFIABLE_FROM_EXTERNALLY_VERIFIED_CATEGORY_NATIVE_C0_RECEIPT"
    for change, message in (
        ({"fixed_support_category_id": CANONICAL_CATEGORY_IDS[0]}, "different category"),
        ({"fixed_support_digest_algorithm": "raw-prototype-bytes"}, "digest algorithm"),
        ({"representation_sha256": digest("wrong-representation")}, "representation identity"),
    ):
        with pytest.raises(GeometryContractError, match=message):
            g1b_identifiability_status(replace(receipt, **change))
    with pytest.raises(GeometryContractError, match="support/M0"):
        g1b_identifiability_status(
            receipt,
            verified_receipt_sha256=receipt.source_receipt_sha256,
            verified_fixed_support_receipt_sha256=digest("wrong-bank-receipt"),
            verified_baseline_certificate_sha256=receipt.baseline_memory_certificate_sha256,
        )


def test_historical_switch_seed0_bank_and_m0_state_cannot_cross_categories():
    switch_receipt = valid_top32_receipt(
        category_id="switch",
        fixed_support_category_id="switch",
        fixed_support_sha256=R1_FIXED_SUPPORT_SHA256,
        fixed_support_seed=0,
    )
    assert g1b_identifiability_status(switch_receipt) == "NOT_YET_IDENTIFIABLE"
    # Reusing the exact switch receipt under another category is rejected even
    # if the caller rewrites its category labels to look internally consistent.
    with pytest.raises(GeometryContractError, match="receipt digest is not bound|cannot be reused"):
        g1b_identifiability_status(replace(
            switch_receipt,
            category_id=CANONICAL_CATEGORY_IDS[1],
            fixed_support_category_id=CANONICAL_CATEGORY_IDS[1],
        ))

    other_category = CANONICAL_CATEGORY_IDS[1]
    other_bank = valid_top32_receipt(
        category_id=other_category,
        fixed_support_category_id=other_category,
        fixed_support_sha256=digest("other-class-bank"),
        fixed_support_seed=0,
    )
    with pytest.raises(GeometryContractError, match="category-specific M0 scorer state"):
        g1b_identifiability_status(replace(
            other_bank, m0_scorer_state_sha256=switch_receipt.m0_scorer_state_sha256))


def test_fixed_support_and_m0_receipt_digests_are_deterministic_and_category_bound():
    switch_digest = category_fixed_support_receipt_sha256(
        R1_FIXED_SUPPORT_CATEGORY_ID, R1_FIXED_SUPPORT_SEED,
        R1_FIXED_SUPPORT_DIGEST_ALGORITHM, R1_FIXED_SUPPORT_SHA256)
    assert switch_digest == category_fixed_support_receipt_sha256(
        R1_FIXED_SUPPORT_CATEGORY_ID, R1_FIXED_SUPPORT_SEED,
        R1_FIXED_SUPPORT_DIGEST_ALGORITHM, R1_FIXED_SUPPORT_SHA256)
    other_digest = category_fixed_support_receipt_sha256(
        CANONICAL_CATEGORY_IDS[0], R1_FIXED_SUPPORT_SEED,
        R1_FIXED_SUPPORT_DIGEST_ALGORITHM, digest("other-bank"))
    assert switch_digest != other_digest
    assert switch_digest != category_fixed_support_receipt_sha256(
        CANONICAL_CATEGORY_IDS[0], R1_FIXED_SUPPORT_SEED,
        R1_FIXED_SUPPORT_DIGEST_ALGORITHM, R1_FIXED_SUPPORT_SHA256)

    receipt = valid_top32_receipt()
    other_state = category_m0_scorer_state_sha256(
        category_id=CANONICAL_CATEGORY_IDS[1],
        representation_sha256=receipt.representation_sha256,
        scorer_profile_sha256=receipt.scorer_profile_sha256,
        fixed_support_receipt_sha256=receipt.fixed_support_receipt_sha256,
        score_state_memory_sha256=receipt.score_state_memory_sha256,
        baseline_memory_contract=receipt.baseline_memory_contract,
        baseline_memory_certificate_sha256=receipt.baseline_memory_certificate_sha256,
    )
    assert other_state != receipt.m0_scorer_state_sha256


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


def test_geometry_requires_authorized_a_receipt_and_external_digest_review():
    unreviewed = digest("unreviewed-a-receipt")
    query = make_batch(
        "cat", [[1.0, 0.0]], [digest("qg")], [digest("qi")],
        a_reference_availability_receipt_sha256=unreviewed)
    reference = make_batch(
        "cat", [[0.0, 1.0]], [digest("rg")], [digest("ri")],
        a_reference_availability_receipt_sha256=unreviewed)
    with pytest.raises(GeometryContractError, match="does not match the externally verified category"):
        _compare_source_disjoint(
            query, reference, radii=[0.0, 1.0, 2.0],
            verified_a_reference_availability_receipts={"cat": A_AUTHORIZATION})
    with pytest.raises(GeometryContractError, match="no externally verified A availability"):
        _compare_source_disjoint(
            query, reference, radii=[0.0, 1.0, 2.0],
            verified_a_reference_availability_receipts={"audiojack": unreviewed})
    pending = make_batch(
        "cat", [[1.0, 0.0]], [digest("pg")], [digest("pi")],
        access_status="PENDING_AUTHORIZATION")
    pending_reference = make_batch(
        "cat", [[0.0, 1.0]], [digest("prg")], [digest("pri")])
    with pytest.raises(GeometryContractError, match="not externally authorized"):
        compare_source_disjoint(pending, pending_reference, radii=[0.0, 1.0, 2.0])


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
    (lambda b: replace(b, access_status="NOT_AUTHORIZED_FOR_G1"), "not externally authorized"),
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
