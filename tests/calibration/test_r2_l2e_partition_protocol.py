from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from experiments.calibration.r2_l2a_allclass_inventory import PINNED_SOURCE_SHA256
from experiments.calibration.r2_l2b_role_certification import ResearcherApproval, SourceGroupMember
from experiments.calibration.r2_l2d_data_role_adjudication import PARTITION_RANKING_ALGORITHM
from experiments.calibration.r2_l2e_partition_protocol import (
    A_MIN_FULL_GEOMETRY_GROUPS,
    A_MIN_PROTOCOL_MATCHED_GROUPS,
    BC_MIN_GROUPS_PER_ROLE,
    CANONICAL_CATEGORY_IDS,
    EXPOSED_DEVELOPMENT_CATEGORIES,
    PARTITION_SEED,
    POLICY_PAYLOAD,
    POLICY_SHA256,
    PROTECTED_CATEGORIES,
    Q_A_TS_TIME_BINS,
    Q_A_TS_SEED,
    SCHEMA,
    TEMPORAL_ORDER_SEED,
    TEMPORAL_ORDER_RANKING_ALGORITHM,
    PartitionProtocolError,
    build_offline_evaluator_stress_order,
    canonical_sha256,
    propose_category_partition,
    researcher_approval_binding_errors,
    validate_category_proposal,
    validate_global_role_disjointness,
)


SOURCE_MANIFEST_SHA = hashlib.sha256(b"synthetic-combined-source-manifest").hexdigest()
L2B_PROPOSAL_SHA = hashlib.sha256(b"synthetic-l2b-proposal").hexdigest()


def group_digest(index: int, namespace: str = "synthetic") -> str:
    return hashlib.sha256(f"{namespace}:{index}".encode("ascii")).hexdigest()


def members(count: int, *, start: int = 0, namespace: str = "synthetic", views: int = 5):
    return tuple(SourceGroupMember(group_digest(i, namespace), views) for i in range(start, start + count))


def propose(
    category: str = "bottle_cap",
    *,
    a_count: int = 100,
    test_count: int = 600,
    test_start: int = 1000,
    prior_exposed_indices: tuple[int, ...] = (),
    **kwargs,
):
    a = members(a_count, namespace=f"a:{category}")
    test = members(test_count, start=test_start, namespace=f"test:{category}")
    exposed = tuple(group_digest(test_start + i, f"test:{category}") for i in prior_exposed_indices)
    return propose_category_partition(
        category,
        a,
        test,
        source_manifest_sha256=SOURCE_MANIFEST_SHA,
        test_identity_scope_verified=True,
        exposure_registry_verified=True,
        prior_exposed_group_sha256s=exposed,
        **kwargs,
    )


def test_policy_is_explicit_versioned_hsource_and_pins_frozen_upstream_contracts():
    assert POLICY_PAYLOAD["schema"] == "r2_l2e.h_source_partition_policy.v1"
    assert POLICY_PAYLOAD["strategy"] == "H-SOURCE"
    assert len(CANONICAL_CATEGORY_IDS) == 30
    assert POLICY_PAYLOAD["catalog_source_sha256"] == PINNED_SOURCE_SHA256
    assert POLICY_PAYLOAD["ranking_algorithm"] == PARTITION_RANKING_ALGORITHM
    assert PARTITION_SEED == 20261101
    assert Q_A_TS_TIME_BINS == 16
    assert Q_A_TS_SEED == 37
    assert TEMPORAL_ORDER_SEED == 20261102
    assert TEMPORAL_ORDER_RANKING_ALGORITHM == "sha256:r2-l2e-temporal-order.v1"
    assert A_MIN_PROTOCOL_MATCHED_GROUPS == 68
    assert A_MIN_FULL_GEOMETRY_GROUPS == 100
    assert BC_MIN_GROUPS_PER_ROLE == 200
    assert set(EXPOSED_DEVELOPMENT_CATEGORIES).isdisjoint(PROTECTED_CATEGORIES)
    assert POLICY_SHA256 == canonical_sha256(POLICY_PAYLOAD)
    policy_path = (
        Path(__file__).parents[2]
        / "docs/calibration/audit/r2_l2e_hsource_partition_protocol_20261101/"
        / "R2_L2E_HSOURCE_PARTITION_POLICY.json"
    )
    saved_policy = json.loads(policy_path.read_text(encoding="utf-8"))
    saved_digest = saved_policy.pop("policy_sha256")
    assert saved_digest == POLICY_SHA256
    assert saved_policy == POLICY_PAYLOAD


def test_repeated_hsource_split_is_deterministic_and_uses_frozen_60_40_formula():
    first = propose(prior_exposed_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    second = propose(prior_exposed_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    assert first == second
    assert first.status == "PROPOSED_REQUIRES_REVIEW"
    assert first.prior_exposed_group_count == 10
    assert first.fresh_test_group_count == 590
    assert first.b_quota_on_fresh_groups == (3 * 590) // 5
    assert first.c_quota_on_fresh_groups == 590 - (3 * 590) // 5
    assert len(first.b_group_sha256s) == 10 + first.b_quota_on_fresh_groups
    assert len(first.c_group_sha256s) == first.c_quota_on_fresh_groups
    assert not set(first.a_group_sha256s).intersection(first.b_group_sha256s)
    assert not set(first.a_group_sha256s).intersection(first.c_group_sha256s)
    assert not set(first.b_group_sha256s).intersection(first.c_group_sha256s)
    validate_category_proposal(first)


def test_a_overlap_is_excluded_before_quota_and_recorded_without_leakage():
    a = list(members(100, namespace="shared"))
    test = list(members(600, start=1000, namespace="test-overlap"))
    test[0] = SourceGroupMember(a[0].group_sha256, 5)
    proposal = propose_category_partition(
        "bottle_cap",
        tuple(a),
        tuple(test),
        source_manifest_sha256=SOURCE_MANIFEST_SHA,
        test_identity_scope_verified=True,
        exposure_registry_verified=True,
    )
    assert proposal.excluded_a_test_overlap_count == 1
    assert proposal.candidate_test_group_count == 600
    assert proposal.fresh_test_group_count == 599
    assert proposal.b_quota_on_fresh_groups == (3 * 599) // 5
    assert proposal.c_quota_on_fresh_groups == 599 - (3 * 599) // 5
    assert proposal.a_group_sha256s[0] not in proposal.b_group_sha256s
    assert proposal.a_group_sha256s[0] not in proposal.c_group_sha256s


def test_previously_exposed_groups_are_kept_in_b_and_never_assigned_to_c():
    proposal = propose(prior_exposed_indices=(0, 4, 9, 30))
    exposed = {group_digest(1000 + i, "test:bottle_cap") for i in (0, 4, 9, 30)}
    assert exposed.issubset(set(proposal.b_group_sha256s))
    assert not exposed.intersection(proposal.c_group_sha256s)


def test_hash_ranking_matches_the_frozen_l2d_domain_separated_formula():
    proposal = propose(prior_exposed_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    exposed = {group_digest(1000 + i, "test:bottle_cap") for i in range(10)}
    fresh = {
        group_digest(1000 + i, "test:bottle_cap")
        for i in range(10, 600)
    }
    ranked = sorted(
        fresh,
        key=lambda group: (
            hashlib.sha256(
                f"{PARTITION_RANKING_ALGORITHM}\0{POLICY_SHA256}\0bottle_cap\0{PARTITION_SEED}\0{group}".encode("ascii")
            ).hexdigest(),
            group,
        ),
    )
    n_b = (3 * len(fresh)) // 5
    assert set(proposal.b_group_sha256s) == exposed.union(ranked[:n_b])
    assert set(proposal.c_group_sha256s) == set(ranked[n_b:])


def test_class_level_stage05_exposure_is_sufficient_to_keep_dev_class_out_of_c():
    proposal = propose_category_partition(
        "switch",
        members(100, namespace="a:switch"),
        members(220, start=1000, namespace="test:switch"),
        source_manifest_sha256=SOURCE_MANIFEST_SHA,
        test_identity_scope_verified=True,
        exposure_registry_verified=False,
    )
    assert proposal.c_status == "PENDING_NOT_APPLICABLE_REVIEW"
    assert proposal.b_status == "PROPOSED_EXPOSED_DEVELOPMENT_REQUIRES_REVIEW"
    assert proposal.prior_exposed_group_count == 220


@pytest.mark.parametrize("category", EXPOSED_DEVELOPMENT_CATEGORIES)
def test_historically_exposed_categories_have_proposed_b_and_unapproved_c_na(category):
    proposal = propose(category=category, test_count=220)
    assert proposal.b_status == "PROPOSED_EXPOSED_DEVELOPMENT_REQUIRES_REVIEW"
    assert proposal.c_status == "PENDING_NOT_APPLICABLE_REVIEW"
    assert len(proposal.b_group_sha256s) == 220
    assert proposal.c_group_sha256s == ()
    assert not proposal.c_status.startswith("CERTIFIED")


@pytest.mark.parametrize("category", PROTECTED_CATEGORIES)
def test_mint_pcb_are_c_only_and_require_explicit_protected_scope(category):
    a = members(100, namespace=f"a:{category}")
    test = members(240, start=1000, namespace=f"test:{category}")
    with pytest.raises(PartitionProtocolError, match="protected TEST identities"):
        propose_category_partition(
            category,
            a,
            test,
            source_manifest_sha256=SOURCE_MANIFEST_SHA,
            test_identity_scope_verified=True,
            exposure_registry_verified=True,
        )
    proposal = propose_category_partition(
        category,
        a,
        test,
        source_manifest_sha256=SOURCE_MANIFEST_SHA,
        test_identity_scope_verified=True,
        protected_test_scope_verified=True,
        exposure_registry_verified=True,
    )
    assert proposal.b_status == "PENDING_NOT_APPLICABLE_REVIEW"
    assert proposal.c_status == "PROPOSED_PROTECTED_SEALED_REQUIRES_REVIEW"
    assert proposal.b_group_sha256s == ()
    assert len(proposal.c_group_sha256s) == 240


def test_protected_cannot_absorb_previously_exposed_groups_under_this_policy():
    with pytest.raises(PartitionProtocolError, match="previously exposed"):
        propose(
            category="mint",
            test_count=240,
            prior_exposed_indices=(0,),
            protected_test_scope_verified=True,
        )


def test_insufficient_group_counts_are_pending_and_never_create_empty_certificates():
    proposal = propose(test_count=400)
    assert proposal.status == "PENDING_INSUFFICIENT_GROUPS"
    assert proposal.b_status == "PENDING_INSUFFICIENT_GROUPS"
    assert proposal.c_status == "PENDING_INSUFFICIENT_GROUPS"
    assert proposal.b_group_sha256s == proposal.c_group_sha256s == ()
    assert proposal.proposed_b_group_count == 240
    assert proposal.proposed_c_group_count == 160


def test_exact_60_40_minimum_fresh_pool_boundary_is_498_groups():
    sufficient = propose(test_count=498)
    insufficient = propose(test_count=497)
    assert sufficient.status == "PROPOSED_REQUIRES_REVIEW"
    assert sufficient.proposed_c_group_count == 200
    assert insufficient.status == "PENDING_INSUFFICIENT_GROUPS"
    assert insufficient.proposed_c_group_count == 199


def test_a_protocol_matched_and_full_geometry_minimums_are_separate():
    below_protocol = propose(a_count=A_MIN_PROTOCOL_MATCHED_GROUPS - 1)
    protocol_only = propose(a_count=A_MIN_FULL_GEOMETRY_GROUPS - 1)
    full_geometry = propose(a_count=A_MIN_FULL_GEOMETRY_GROUPS)
    assert below_protocol.a_status == "PENDING_INSUFFICIENT_GROUPS_FOR_4_PLUS_64"
    assert below_protocol.a_group_sha256s == ()
    assert protocol_only.a_status == "PROPOSED_REQUIRES_INDEPENDENT_ROLE_REVIEW"
    assert protocol_only.a_full_geometry_status == "PENDING_INSUFFICIENT_GROUPS_FOR_FULL_GEOMETRY"
    assert full_geometry.a_full_geometry_status == "PROPOSED_REQUIRES_INDEPENDENT_ROLE_REVIEW"


def test_missing_source_receipts_remain_pending_and_do_not_imply_empty_roles():
    proposal = propose_category_partition(
        "bottle_cap", None, None, source_manifest_sha256=SOURCE_MANIFEST_SHA,
    )
    assert proposal.status == "PENDING_AUTHORIZED_IDENTITY_INPUT"
    assert proposal.a_status == "PENDING_AUTHORIZED_A_IDENTITY_RECEIPT"
    assert proposal.b_status == "PENDING_AUTHORIZED_TEST_IDENTITY_RECEIPT"
    assert proposal.c_status == "PENDING_AUTHORIZED_TEST_IDENTITY_RECEIPT"
    assert proposal.a_group_sha256s == proposal.b_group_sha256s == proposal.c_group_sha256s == ()


def test_explicitly_empty_test_input_is_still_not_a_certified_empty_role():
    proposal = propose_category_partition(
        "bottle_cap",
        members(100, namespace="a:bottle_cap"),
        (),
        source_manifest_sha256=SOURCE_MANIFEST_SHA,
        test_identity_scope_verified=True,
    )
    assert proposal.status == "PENDING_AUTHORIZED_IDENTITY_INPUT"
    assert proposal.c_status == "PENDING_AUTHORIZED_TEST_IDENTITY_RECEIPT"


def test_test_identity_requires_verified_scope_and_exposure_registry():
    a = members(100, namespace="a:bottle_cap")
    test = members(600, start=1000, namespace="test:bottle_cap")
    with pytest.raises(PartitionProtocolError, match="without a verified L2B"):
        propose_category_partition(
            "bottle_cap", a, test, source_manifest_sha256=SOURCE_MANIFEST_SHA,
        )
    pending = propose_category_partition(
        "bottle_cap",
        a,
        test,
        source_manifest_sha256=SOURCE_MANIFEST_SHA,
        test_identity_scope_verified=True,
        exposure_registry_verified=False,
    )
    assert pending.status == "PENDING_VERIFIED_EXPOSURE_REGISTRY"
    assert pending.b_group_sha256s == pending.c_group_sha256s == ()


def test_source_group_must_keep_all_five_camera_views_together():
    with pytest.raises(PartitionProtocolError, match="exactly 5 camera views"):
        propose_category_partition(
            "bottle_cap",
            members(100, namespace="a:bottle_cap"),
            members(600, start=1000, namespace="test:bottle_cap", views=4),
            source_manifest_sha256=SOURCE_MANIFEST_SHA,
            test_identity_scope_verified=True,
            exposure_registry_verified=True,
        )


def test_duplicate_malformed_or_cross_manifest_exposure_identities_fail_closed():
    a = members(100, namespace="a:bottle_cap")
    test = list(members(600, start=1000, namespace="test:bottle_cap"))
    test[1] = test[0]
    with pytest.raises(PartitionProtocolError, match="duplicate source groups"):
        propose_category_partition(
            "bottle_cap", a, tuple(test), source_manifest_sha256=SOURCE_MANIFEST_SHA,
            test_identity_scope_verified=True, exposure_registry_verified=True,
        )
    with pytest.raises(PartitionProtocolError, match="does not match"):
        propose_category_partition(
            "bottle_cap",
            a,
            members(600, start=1000, namespace="test:bottle_cap"),
            source_manifest_sha256=SOURCE_MANIFEST_SHA,
            test_identity_scope_verified=True,
            exposure_registry_verified=True,
            prior_exposed_group_sha256s=(group_digest(7, "wrong-manifest"),),
        )


def test_category_labels_cannot_be_an_algorithm_input():
    with pytest.raises(TypeError):
        propose_category_partition(
            "bottle_cap",
            members(100, namespace="a:bottle_cap"),
            members(600, start=1000, namespace="test:bottle_cap"),
            source_manifest_sha256=SOURCE_MANIFEST_SHA,
            labels=(0, 1),
        )


def test_global_source_group_overlap_across_categories_fails_closed():
    shared = group_digest(1, "shared-cross-category")
    empty = {"A": (), "B": (), "C": ()}
    assignments = {
        "audiojack": {**empty, "A": (shared,)},
        "pcb": {**empty, "C": (shared,)},
    }
    with pytest.raises(PartitionProtocolError, match="assigned more than once"):
        validate_global_role_disjointness(assignments)


def test_temporal_stress_orders_retain_the_same_group_multiset_and_labels_stay_offline():
    groups = tuple(group_digest(i, "stress") for i in range(240))
    labels = {group: int(i >= 120) for i, group in enumerate(groups)}
    random_order = build_offline_evaluator_stress_order(
        groups, order_name="RANDOM", labels_by_group=labels,
    )
    swapped_labels = {group: 1 - label for group, label in labels.items()}
    assert random_order == build_offline_evaluator_stress_order(
        groups, order_name="RANDOM", labels_by_group=swapped_labels,
    )
    normal_first = build_offline_evaluator_stress_order(
        groups, order_name="NORMAL_FIRST", labels_by_group=labels,
    )
    defect_first = build_offline_evaluator_stress_order(
        groups, order_name="DEFECT_FIRST", labels_by_group=labels,
    )
    bursty = build_offline_evaluator_stress_order(
        groups, order_name="BURSTY_50_GROUP", labels_by_group=labels,
    )
    for order in (random_order, normal_first, defect_first, bursty):
        assert len(order) == len(groups)
        assert set(order) == set(groups)
    assert all(labels[group] == 0 for group in normal_first[:120])
    assert all(labels[group] == 1 for group in defect_first[:120])
    assert all(labels[group] == 0 for group in bursty[:50])
    assert all(labels[group] == 1 for group in bursty[50:100])
    assert all(labels[group] == 0 for group in bursty[100:150])


def test_label_order_builders_require_a_complete_binary_evaluator_map():
    groups = tuple(group_digest(i, "stress-errors") for i in range(4))
    with pytest.raises(PartitionProtocolError, match="one evaluator label"):
        build_offline_evaluator_stress_order(groups, order_name="NORMAL_FIRST")
    with pytest.raises(PartitionProtocolError, match="binary"):
        build_offline_evaluator_stress_order(
            groups,
            order_name="BURSTY_50_GROUP",
            labels_by_group={group: 2 for group in groups},
        )
    with pytest.raises(PartitionProtocolError, match="seed"):
        build_offline_evaluator_stress_order(
            groups, order_name="RANDOM", seed=TEMPORAL_ORDER_SEED + 1,
        )


def test_policy_seed_or_payload_corruption_is_rejected():
    with pytest.raises(PartitionProtocolError, match="seed"):
        propose_category_partition(
            "bottle_cap",
            None,
            None,
            source_manifest_sha256=SOURCE_MANIFEST_SHA,
            seed=PARTITION_SEED + 1,
        )
    with pytest.raises(PartitionProtocolError, match="policy hash"):
        propose_category_partition(
            "bottle_cap",
            None,
            None,
            source_manifest_sha256=SOURCE_MANIFEST_SHA,
            policy_sha256=hashlib.sha256(b"wrong policy").hexdigest(),
        )
    proposal = propose()
    with pytest.raises(PartitionProtocolError, match="SHA-256"):
        validate_category_proposal(replace(proposal, proposal_sha256="0" * 64))
    with pytest.raises(PartitionProtocolError, match="source-manifest"):
        propose_category_partition("bottle_cap", None, None, source_manifest_sha256="bad")
    assert proposal.schema == SCHEMA


def test_l2e_policy_approval_must_bind_exact_policy_and_full_role_scope():
    good = ResearcherApproval(
        proposal_sha256=L2B_PROPOSAL_SHA,
        researcher_id="synthetic-researcher",
        decision="APPROVE_H_SOURCE",
        decision_id=f"R2-L2E-FREEZE:{POLICY_SHA256}",
        approved_categories=CANONICAL_CATEGORY_IDS,
        approved_roles=("A", "B", "C"),
        approved_protected_policy_categories=PROTECTED_CATEGORIES,
        signature_ed25519_hex="00",
        approval_sha256="00" * 32,
    )
    assert researcher_approval_binding_errors(
        good, expected_l2b_proposal_sha256=L2B_PROPOSAL_SHA,
    ) == ()
    wrong_policy = replace(good, decision_id="R2-L2E-FREEZE:" + "0" * 64)
    assert "signed decision_id does not bind the exact L2E policy SHA-256" in (
        researcher_approval_binding_errors(
            wrong_policy, expected_l2b_proposal_sha256=L2B_PROPOSAL_SHA,
        )
    )
    assert "researcher approval is missing" in researcher_approval_binding_errors(
        None, expected_l2b_proposal_sha256=L2B_PROPOSAL_SHA,
    )
