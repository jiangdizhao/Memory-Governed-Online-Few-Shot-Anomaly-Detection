from __future__ import annotations

from dataclasses import asdict, replace
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.calibration.r2_l2a_allclass_inventory import (
    AReferenceAvailabilityReceipt,
    CANONICAL_CATEGORY_IDS,
    PINNED_SOURCE_SHA256,
    PINNED_SOURCE_URI,
    OfficialCatalogReceipt,
)
from experiments.calibration.r2_l2b_role_certification import (
    G1_FULL_A,
    G1_PROTOCOL_MATCHED,
    IndependentReviewerAttestation,
    R2L2BContractError,
    ReviewedIdentityManifest,
    SourceGroupMember,
    access_scope_errors,
    build_category_readiness,
    canonical_sha256,
    certify_a_reference_availability,
    certify_role_identity,
    default_access_scope,
    detect_cross_role_overlaps,
    evaluate_policy_approval,
    g1_regimes_are_distinct,
    make_h_source_proposal,
    make_reviewed_identity_manifest,
    make_reviewer_attestation,
    make_reviewer_trust_directory,
    make_researcher_approval,
    make_role_identity_receipt,
    require_metadata_scope,
    source_group_digest_from_image_path,
    source_group_digest_from_key,
    validate_role_policy_proposal,
    _role_identity_status,
)


def catalog_receipt() -> OfficialCatalogReceipt:
    return OfficialCatalogReceipt(
        source_uri=PINNED_SOURCE_URI,
        source_sha256=PINNED_SOURCE_SHA256,
        release_id="v1.0.0",
        categories=CANONICAL_CATEGORY_IDS,
        official_source_verified=True,
        non_label_bearing=True,
        contains_image_paths=False,
        contains_labels_or_masks=False,
        source_commit="98c28ffbe95b380d17a3777d77219152368c19e9",
        source_path="experiments/RealIAD-full/config.yaml",
        dataset_config_version="v1.0.0",
        extraction_algorithm="strict_yaml_dataset_train_test_meta_file_v1",
    )


def private_key():
    return Ed25519PrivateKey.generate()


def public_key_hex(key) -> str:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()


def synthetic_receipt(role: str = "B", category: str = "audiojack", group: str | None = None):
    return make_role_identity_receipt(
        category_id=category,
        role=role,
        source_manifest_sha256="a" * 64,
        producer_id="synthetic-producer",
        group_members=(SourceGroupMember(group or ("b" * 64), 2),),
    )


def synthetic_review(receipt, *, reviewer="independent-reviewer", population="not_asserted"):
    key = private_key()
    manifest = make_reviewed_identity_manifest(
        receipt,
        population=population,
        normal_evidence_manifest_sha256=("c" * 64 if population == "authorized_normal_reference_only" else None),
    )
    attestation = make_reviewer_attestation(
        receipt, manifest, reviewer_id=reviewer, decision_id="synthetic-review-01",
        private_key=key,
    )
    trust = make_reviewer_trust_directory(
        directory_id="synthetic-trust-root",
        trusted_reviewer_public_keys={reviewer: public_key_hex(key)},
        trusted_identity_source_sha256s=("a" * 64,),
    )
    return manifest, attestation, trust


def test_complete_h_source_role_policy_is_explicit_but_unapproved():
    proposal = make_h_source_proposal(catalog_receipt())
    validate_role_policy_proposal(proposal)
    assert proposal.strategy == "H-SOURCE"
    assert len(proposal.roles) == 30 * 3
    assert {(row.category_id, row.role) for row in proposal.roles} == {
        (category, role) for category in CANONICAL_CATEGORY_IDS for role in ("A", "B", "C")
    }
    assert all(row.state == "PENDING_AUTHORIZATION_PROTECTED"
               for row in proposal.roles if row.category_id in {"mint", "pcb"})
    assert all(row.state == "PROPOSED"
               for row in proposal.roles if row.category_id not in {"mint", "pcb"})
    assert evaluate_policy_approval(
        proposal, None, None, expected_trust_directory_sha256=None
    ) == "PROPOSED_PENDING_RESEARCHER_APPROVAL"


def test_incomplete_or_self_approved_policy_fails_closed():
    proposal = make_h_source_proposal(catalog_receipt())
    with pytest.raises(R2L2BContractError, match="digest"):
        validate_role_policy_proposal(replace(proposal, proposal_sha256="0" * 64))
    approved_row = replace(proposal.roles[0], state="APPROVED")
    edited = replace(proposal, roles=(approved_row,) + proposal.roles[1:])
    edited = replace(edited, proposal_sha256=canonical_sha256({
        key: value for key, value in asdict(edited).items()
        if key != "proposal_sha256"
    }))
    with pytest.raises(R2L2BContractError, match="approve itself"):
        validate_role_policy_proposal(edited)


def test_producer_digest_cannot_approve_policy_without_external_trust():
    proposal = make_h_source_proposal(catalog_receipt())
    key = private_key()
    approval = make_researcher_approval(
        proposal,
        researcher_id="synthetic-author",
        decision_id="local-approval",
        approved_categories=CANONICAL_CATEGORY_IDS,
        approved_roles=("A", "B", "C"),
        approved_protected_policy_categories=("mint", "pcb"),
        private_key=key,
    )
    assert evaluate_policy_approval(
        proposal, approval, None, expected_trust_directory_sha256=None
    ) == "PENDING_EXTERNAL_APPROVAL_TRUST_ROOT"
    trusted_key = private_key()
    trust = make_reviewer_trust_directory(
        directory_id="policy-root",
        trusted_policy_approver_public_keys={
            "synthetic-author": public_key_hex(trusted_key),
        },
        trusted_identity_source_sha256s=("a" * 64,),
        approved_proposal_sha256s=(proposal.proposal_sha256,),
    )
    assert evaluate_policy_approval(
        proposal, approval, trust,
        expected_trust_directory_sha256=trust.directory_sha256,
    ) == "PENDING_EXTERNAL_APPROVAL_TRUST_ROOT"


def test_complete_researcher_policy_decision_does_not_grant_local_access():
    proposal = make_h_source_proposal(catalog_receipt())
    key = private_key()
    approval = make_researcher_approval(
        proposal,
        researcher_id="researcher-01",
        decision_id="synthetic-policy-decision",
        approved_categories=CANONICAL_CATEGORY_IDS,
        approved_roles=("A", "B", "C"),
        approved_protected_policy_categories=("mint", "pcb"),
        private_key=key,
    )
    trust = make_reviewer_trust_directory(
        directory_id="synthetic-policy-trust",
        trusted_reviewer_public_keys={},
        trusted_policy_approver_public_keys={"researcher-01": public_key_hex(key)},
        trusted_identity_source_sha256s=("a" * 64,),
        approved_proposal_sha256s=(proposal.proposal_sha256,),
    )
    status = evaluate_policy_approval(
        proposal, approval, trust,
        expected_trust_directory_sha256=trust.directory_sha256,
    )
    assert status == "APPROVED_POLICY_PENDING_LOCAL_ACCESS"
    report = build_category_readiness(
        catalog_receipt(), proposal, default_access_scope(),
        policy_approval=approval, trust_directory=trust,
        expected_trust_directory_sha256=trust.directory_sha256,
    )
    assert report["policy_status"] == "APPROVED_POLICY_PENDING_LOCAL_ACCESS"
    assert report["local_data_access_status"] == "PENDING_AUTHORIZATION"
    assert report["role_certified"] is False


def test_public_catalog_never_implies_local_access():
    proposal = make_h_source_proposal(catalog_receipt())
    scope = default_access_scope()
    report = build_category_readiness(catalog_receipt(), proposal, scope)
    assert report["public_inventory_status"] == "R2_L2A_R_PUBLIC_INVENTORY_PASS"
    assert report["local_data_access_status"] == "PENDING_AUTHORIZATION"
    assert report["role_certified"] is False
    assert report["g1_full_a_ready"] is False
    assert report["actual_distance_statistics"] == "NOT_RUN"
    assert len(report["category_rows"]) == 30
    assert all(row["source_group_counts"] is None for row in report["category_rows"])
    assert all(row["local_data_inventory_status"] == "PENDING_AUTHORIZATION"
               for row in report["category_rows"])
    assert all(
        row["role_identity_status"] == {
            "A": "PENDING_MISSING_RECEIPT",
            "B": "PENDING_MISSING_RECEIPT",
            "C": "PENDING_MISSING_RECEIPT",
        }
        for row in report["category_rows"]
    )


def test_default_scope_is_deny_all_and_prospective_roots_are_not_approval():
    scope = default_access_scope()
    assert scope.approved_local_root is None
    assert scope.approved_categories == ()
    assert scope.allow_physical_json_file_bytes is False
    assert scope.parsed_sections == ()
    assert scope.allow_train_normal_ok_status is False
    assert scope.allow_feature_extraction is False
    assert scope.allow_outcome_label_access is False
    assert scope.allow_mask_access is False
    assert scope.maximum_output_bytes == 0
    with pytest.raises(R2L2BContractError, match="unauthorized"):
        require_metadata_scope(
            scope, category_id="audiojack", section="train",
            fields=("official_sample_key",),
        )


def test_train_metadata_scope_requires_json_byte_and_mixed_section_acknowledgment():
    scope = replace(
        default_access_scope(),
        approval_state="APPROVED",
        approval_id="synthetic-access-approval",
        approved_local_root="G:/dataset/realiad",
        approved_categories=("audiojack",),
        allow_physical_json_file_bytes=True,
        parsed_sections=("train",),
        allowed_train_metadata_fields=("official_sample_key",),
        allow_official_image_identity_metadata=True,
        allow_official_sample_key_condition_tokens=True,
        maximum_output_bytes=1024,
    )
    assert any("mixed train/test JSON" in error for error in access_scope_errors(scope))
    with pytest.raises(R2L2BContractError, match="mixed train/test"):
        require_metadata_scope(
            scope, category_id="audiojack", section="train",
            fields=("official_sample_key",),
        )
    approved_bytes = replace(scope, acknowledge_train_and_test_may_share_physical_json=True)
    require_metadata_scope(
        approved_bytes, category_id="audiojack", section="train",
        fields=("official_sample_key",),
    )
    approved_normal_field = replace(
        approved_bytes,
        allowed_train_metadata_fields=("official_sample_key", "normal_status"),
    )
    with pytest.raises(R2L2BContractError, match="label-bearing"):
        require_metadata_scope(
            approved_normal_field, category_id="audiojack", section="train",
            fields=("normal_status",),
        )


def test_identity_field_needs_explicit_permission_even_with_train_scope():
    scope = replace(
        default_access_scope(),
        approval_state="APPROVED",
        approval_id="synthetic-access-approval",
        approved_local_root="G:/dataset/realiad",
        approved_categories=("audiojack",),
        allow_physical_json_file_bytes=True,
        acknowledge_train_and_test_may_share_physical_json=True,
        parsed_sections=("train",),
        allowed_train_metadata_fields=("official_sample_key",),
        allow_official_sample_key_condition_tokens=True,
        maximum_output_bytes=1024,
    )
    with pytest.raises(R2L2BContractError, match="identity metadata"):
        require_metadata_scope(
            scope, category_id="audiojack", section="train",
            fields=("official_sample_key",),
        )


def test_official_sample_key_condition_tokens_need_separate_permission():
    scope = replace(
        default_access_scope(),
        approval_state="APPROVED",
        approval_id="synthetic-access-approval",
        approved_local_root="G:/dataset/realiad",
        approved_categories=("audiojack",),
        allow_physical_json_file_bytes=True,
        acknowledge_train_and_test_may_share_physical_json=True,
        parsed_sections=("train",),
        allowed_train_metadata_fields=("official_sample_key",),
        allow_official_image_identity_metadata=True,
        maximum_output_bytes=1024,
    )
    with pytest.raises(R2L2BContractError, match="OK/NG/defect tokens"):
        require_metadata_scope(
            scope, category_id="audiojack", section="train",
            fields=("official_sample_key",),
        )


def test_test_label_and_protected_category_access_require_separate_approval():
    scope = replace(
        default_access_scope(),
        approval_state="APPROVED",
        approval_id="synthetic-access-approval",
        approved_local_root="G:/dataset/realiad",
        approved_categories=("audiojack",),
        allow_physical_json_file_bytes=True,
        acknowledge_train_and_test_may_share_physical_json=True,
        parsed_sections=("test",),
        allowed_test_metadata_fields=("label",),
        maximum_output_bytes=1024,
    )
    with pytest.raises(R2L2BContractError, match="outcome"):
        require_metadata_scope(
            scope, category_id="audiojack", section="test", fields=("label",)
        )
    protected_scope = replace(scope, approved_categories=("audiojack", "mint"))
    with pytest.raises(R2L2BContractError, match="protected"):
        require_metadata_scope(
            protected_scope, category_id="mint", section="test", fields=("label",), protected=True
        )


def test_test_mask_permission_is_separate_from_outcome_label_permission():
    scope = replace(
        default_access_scope(),
        approval_state="APPROVED",
        approval_id="synthetic-access-approval",
        approved_local_root="G:/dataset/realiad",
        approved_categories=("audiojack",),
        allow_physical_json_file_bytes=True,
        acknowledge_train_and_test_may_share_physical_json=True,
        parsed_sections=("test",),
        allowed_test_metadata_fields=("mask",),
        allow_outcome_label_access=True,
        maximum_output_bytes=1024,
    )
    with pytest.raises(R2L2BContractError, match="masks"):
        require_metadata_scope(
            scope, category_id="audiojack", section="test", fields=("mask",)
        )


def test_official_multiview_sample_key_maps_to_one_domain_separated_group():
    c1 = "OK/S0001/audiojack_0001_OK_C1_20200101000000.jpg"
    c2 = "OK/S0001/audiojack_0001_OK_C2_20200101000001.jpg"
    digest1 = source_group_digest_from_image_path(c1, "audiojack")
    digest2 = source_group_digest_from_image_path(c2, "audiojack")
    assert digest1 == digest2
    assert digest1 == source_group_digest_from_key("audiojack", "audiojack_0001_OK")
    assert digest1 != source_group_digest_from_key("audiojack", "audiojack_0002_OK")
    assert len(digest1) == 64


def test_identity_parser_rejects_wrong_category_and_malformed_key():
    with pytest.raises(ValueError):
        source_group_digest_from_image_path(
            "OK/S0001/audiojack_0001_OK_C1_20200101000000.jpg", "switch"
        )
    with pytest.raises(R2L2BContractError, match="official identity"):
        source_group_digest_from_key("audiojack", "audiojack_0001")


def test_role_receipt_keeps_multiview_group_once_without_raw_identity():
    digest = source_group_digest_from_key("audiojack", "audiojack_0001_OK")
    receipt = make_role_identity_receipt(
        category_id="audiojack",
        role="B",
        source_manifest_sha256="a" * 64,
        producer_id="producer",
        group_members=(SourceGroupMember(digest, 2),),
    )
    assert len(receipt.group_members) == 1
    assert receipt.image_count == 2
    assert receipt.contains_raw_sample_keys is False
    assert receipt.contains_image_paths is False
    assert receipt.contains_labels_or_masks is False
    serialized = json.dumps(asdict(receipt))
    assert "audiojack_0001_OK" not in serialized
    assert "S0001" not in serialized


def test_missing_receipt_is_not_an_empty_role():
    assert not hasattr(default_access_scope(), "group_members")
    with pytest.raises(R2L2BContractError, match="empty role requires"):
        make_role_identity_receipt(
            category_id="audiojack",
            role="B",
            source_manifest_sha256="a" * 64,
            producer_id="producer",
            group_members=(),
        )
    with pytest.raises(R2L2BContractError, match="reviewed frozen"):
        make_role_identity_receipt(
            category_id="audiojack",
            role="B",
            source_manifest_sha256="a" * 64,
            producer_id="producer",
            group_members=(),
            empty_role_policy_sha256="c" * 64,
        )


def test_explicit_empty_role_requires_frozen_policy_and_independent_review():
    policy_digest = "f" * 64
    empty = make_role_identity_receipt(
        category_id="audiojack",
        role="B",
        source_manifest_sha256="a" * 64,
        producer_id="synthetic-producer",
        group_members=(),
        empty_role_policy_sha256=policy_digest,
        empty_role_reason="Synthetic reviewed N/A rationale.",
    )
    manifest = make_reviewed_identity_manifest(empty)
    reviewer_key = private_key()
    attestation = make_reviewer_attestation(
        empty, manifest, reviewer_id="independent-reviewer", decision_id="review-empty",
        private_key=reviewer_key,
    )
    unreviewed = make_reviewer_trust_directory(
        directory_id="root",
        trusted_reviewer_public_keys={"independent-reviewer": public_key_hex(reviewer_key)},
        trusted_identity_source_sha256s=("a" * 64,),
    )
    status, reason = certify_role_identity(
        empty, manifest, attestation, unreviewed,
        expected_trust_directory_sha256=unreviewed.directory_sha256,
    )
    assert status == "PENDING_INDEPENDENT_REVIEW"
    assert "empty role policy is not independently reviewed" in reason
    reviewed = make_reviewer_trust_directory(
        directory_id="root",
        trusted_reviewer_public_keys={"independent-reviewer": public_key_hex(reviewer_key)},
        trusted_identity_source_sha256s=("a" * 64,),
        reviewed_empty_role_policy_sha256s=(policy_digest,),
    )
    status, _ = certify_role_identity(
        empty, manifest, attestation, reviewed,
        expected_trust_directory_sha256=reviewed.directory_sha256,
    )
    assert status == "CERTIFIED_FROM_SOURCE_EVIDENCE"


def test_reviewer_manifest_and_trust_root_certify_synthetic_receipt():
    receipt = synthetic_receipt()
    manifest, attestation, trust = synthetic_review(receipt)
    state, reason = certify_role_identity(
        receipt, manifest, attestation, trust,
        expected_trust_directory_sha256=trust.directory_sha256,
    )
    assert state == "CERTIFIED_FROM_SOURCE_EVIDENCE", reason


def test_digest_only_self_attestation_and_unpinned_trust_stay_pending():
    receipt = synthetic_receipt()
    manifest, attestation, trust = synthetic_review(receipt, reviewer="synthetic-producer")
    state, reason = certify_role_identity(
        receipt, manifest, attestation, trust,
        expected_trust_directory_sha256=trust.directory_sha256,
    )
    assert state == "PENDING_INDEPENDENT_REVIEW"
    assert "producer cannot" in reason
    manifest, attestation, trust = synthetic_review(receipt)
    state, reason = certify_role_identity(
        receipt, manifest, attestation, trust,
        expected_trust_directory_sha256=None,
    )
    assert state == "PENDING_INDEPENDENT_REVIEW"
    assert "trust root is absent" in reason


def test_reviewer_must_be_in_trust_directory_and_match_receipt():
    receipt = synthetic_receipt()
    manifest = make_reviewed_identity_manifest(receipt)
    untrusted_key = private_key()
    untrusted = make_reviewer_attestation(
        receipt, manifest, reviewer_id="unlisted-reviewer", decision_id="review-1",
        private_key=untrusted_key,
    )
    other_key = private_key()
    trust = make_reviewer_trust_directory(
        directory_id="root",
        trusted_reviewer_public_keys={"other-reviewer": public_key_hex(other_key)},
        trusted_identity_source_sha256s=("a" * 64,),
    )
    state, reason = certify_role_identity(
        receipt, manifest, untrusted, trust,
        expected_trust_directory_sha256=trust.directory_sha256,
    )
    assert state == "PENDING_INDEPENDENT_REVIEW"
    assert "absent from the trusted reviewer" in reason


def test_producer_cannot_forge_attestation_under_a_trusted_reviewer_name():
    receipt = synthetic_receipt()
    manifest = make_reviewed_identity_manifest(receipt)
    attacker_key = private_key()
    forged = make_reviewer_attestation(
        receipt, manifest,
        reviewer_id="independent-reviewer",
        decision_id="forged-review",
        private_key=attacker_key,
    )
    trusted_key = private_key()
    trust = make_reviewer_trust_directory(
        directory_id="root",
        trusted_reviewer_public_keys={
            "independent-reviewer": public_key_hex(trusted_key),
        },
        trusted_identity_source_sha256s=("a" * 64,),
    )
    state, reason = certify_role_identity(
        receipt, manifest, forged, trust,
        expected_trust_directory_sha256=trust.directory_sha256,
    )
    assert state == "PENDING_INDEPENDENT_REVIEW"
    assert "signature is invalid" in reason


def test_independent_manifest_must_match_producer_source_group_set():
    receipt = synthetic_receipt()
    original = make_reviewed_identity_manifest(receipt)
    partial = replace(
        original,
        group_members=(SourceGroupMember("c" * 64, 1),),
        image_count=1,
        manifest_sha256="",
    )
    manifest_payload = asdict(partial)
    manifest_payload.pop("manifest_sha256")
    manifest = replace(partial, manifest_sha256=canonical_sha256(manifest_payload))
    reviewer_key = private_key()
    attestation = make_reviewer_attestation(
        receipt, manifest,
        reviewer_id="independent-reviewer",
        decision_id="review-different-source-set",
        private_key=reviewer_key,
    )
    trust = make_reviewer_trust_directory(
        directory_id="root",
        trusted_reviewer_public_keys={
            "independent-reviewer": public_key_hex(reviewer_key),
        },
        trusted_identity_source_sha256s=("a" * 64,),
    )
    state, reason = certify_role_identity(
        receipt, manifest, attestation, trust,
        expected_trust_directory_sha256=trust.directory_sha256,
    )
    assert state == "PENDING_INDEPENDENT_REVIEW"
    assert "does not match producer receipt" in reason


def test_c_identity_requires_independent_manifest_and_unreviewed_digest_is_insufficient():
    receipt = synthetic_receipt(role="C")
    state, reason = certify_role_identity(
        receipt, None, None, None, expected_trust_directory_sha256=None
    )
    assert state == "PENDING_INDEPENDENT_REVIEW"
    assert "manifest or attestation" in reason
    manifest, attestation, trust = synthetic_review(receipt)
    state, _ = certify_role_identity(
        receipt, manifest, attestation, trust,
        expected_trust_directory_sha256="0" * 64,
    )
    assert state == "PENDING_INDEPENDENT_REVIEW"


def test_c_role_needs_explicit_external_certificate_in_access_scope():
    receipt = synthetic_receipt(role="C")
    manifest, attestation, trust = synthetic_review(receipt)
    base_scope = replace(
        default_access_scope(),
        approval_state="APPROVED",
        approval_id="synthetic-access",
        approved_local_root="G:/dataset/realiad",
        approved_categories=("audiojack",),
        allow_physical_json_file_bytes=True,
        acknowledge_train_and_test_may_share_physical_json=True,
        parsed_sections=("train",),
        allowed_train_metadata_fields=("official_sample_key",),
        allow_official_image_identity_metadata=True,
        allow_official_sample_key_condition_tokens=True,
        maximum_output_bytes=1024,
    )
    status, _ = _role_identity_status(
        receipt,
        category_id="audiojack",
        role="C",
        manifest=manifest,
        attestation=attestation,
        trust_directory=trust,
        expected_trust_directory_sha256=trust.directory_sha256,
        policy_status="APPROVED_POLICY_PENDING_LOCAL_ACCESS",
        scope=base_scope,
    )
    assert status == "PENDING_INDEPENDENT_REVIEW"
    certified_scope = replace(base_scope, independent_c_identity_certificate_supplied=True)
    status, _ = _role_identity_status(
        receipt,
        category_id="audiojack",
        role="C",
        manifest=manifest,
        attestation=attestation,
        trust_directory=trust,
        expected_trust_directory_sha256=trust.directory_sha256,
        policy_status="APPROVED_POLICY_PENDING_LOCAL_ACCESS",
        scope=certified_scope,
    )
    assert status == "CERTIFIED_FROM_SOURCE_EVIDENCE"


def test_receipt_cannot_be_substituted_into_another_category_or_role_slot():
    receipt = synthetic_receipt(role="B", category="switch")
    manifest, attestation, trust = synthetic_review(receipt)
    status, reason = _role_identity_status(
        receipt,
        category_id="audiojack",
        role="B",
        manifest=manifest,
        attestation=attestation,
        trust_directory=trust,
        expected_trust_directory_sha256=trust.directory_sha256,
        policy_status="APPROVED_POLICY_PENDING_LOCAL_ACCESS",
        scope=default_access_scope(),
    )
    assert status == "PENDING_INDEPENDENT_REVIEW"
    assert "category or role differs" in reason
    status, reason = _role_identity_status(
        receipt,
        category_id="switch",
        role="A",
        manifest=manifest,
        attestation=attestation,
        trust_directory=trust,
        expected_trust_directory_sha256=trust.directory_sha256,
        policy_status="APPROVED_POLICY_PENDING_LOCAL_ACCESS",
        scope=default_access_scope(),
    )
    assert status == "PENDING_INDEPENDENT_REVIEW"
    assert "category or role differs" in reason


def test_a_normal_reference_requires_reviewed_normal_only_evidence():
    a_receipt = synthetic_receipt(role="A")
    unchecked_manifest = make_reviewed_identity_manifest(a_receipt)
    reviewer_key = private_key()
    bad_attestation = make_reviewer_attestation(
        a_receipt, unchecked_manifest,
        reviewer_id="independent-reviewer", decision_id="review-a",
        private_key=reviewer_key,
    )
    trust = make_reviewer_trust_directory(
        directory_id="root",
        trusted_reviewer_public_keys={"independent-reviewer": public_key_hex(reviewer_key)},
        trusted_identity_source_sha256s=("a" * 64,),
    )
    status, reason = certify_role_identity(
        a_receipt, unchecked_manifest, bad_attestation, trust,
        expected_trust_directory_sha256=trust.directory_sha256,
    )
    assert status == "PENDING_INDEPENDENT_REVIEW"
    assert "normal-only evidence" in reason
    with pytest.raises(R2L2BContractError, match="normal-only evidence"):
        make_reviewed_identity_manifest(
            a_receipt, population="authorized_normal_reference_only"
        )


def test_a_reference_receipt_alone_does_not_authorize_or_certify_a_pool():
    a = AReferenceAvailabilityReceipt(
        category_id="audiojack",
        authorized_normal_reference_image_count=2,
        authorized_source_group_count=1,
        receipt_sha256="d" * 64,
        authorization_id="synthetic-only",
    )
    assert certify_a_reference_availability(
        None, expected_category="audiojack",
        independently_certified_a_identity=None, manifest=None,
        attestation=None,
        trust_directory=None, expected_trust_directory_sha256=None,
    ) == "PENDING_A_REFERENCE_RECEIPT"
    assert certify_a_reference_availability(
        a, expected_category="audiojack",
        independently_certified_a_identity=None, manifest=None,
        attestation=None,
        trust_directory=None, expected_trust_directory_sha256=None,
    ) == "PENDING_INDEPENDENT_A_REVIEW"
    trust = make_reviewer_trust_directory(
        directory_id="root",
        trusted_identity_source_sha256s=("a" * 64,),
        reviewed_a_availability_receipt_sha256s=("d" * 64,),
    )
    assert certify_a_reference_availability(
        a, expected_category="audiojack",
        independently_certified_a_identity=None, manifest=None,
        attestation=None,
        trust_directory=trust, expected_trust_directory_sha256=trust.directory_sha256,
    ) == "PENDING_A_IDENTITY_CERTIFICATION"


def test_independent_a_identity_and_normal_evidence_must_match_availability_receipt():
    receipt = synthetic_receipt(role="A")
    manifest, attestation, base_trust = synthetic_review(
        receipt, population="authorized_normal_reference_only"
    )
    availability = AReferenceAvailabilityReceipt(
        category_id="audiojack",
        authorized_normal_reference_image_count=2,
        authorized_source_group_count=1,
        receipt_sha256="d" * 64,
        authorization_id="synthetic-only",
    )
    trust = make_reviewer_trust_directory(
        directory_id="root",
        trusted_reviewer_public_keys=dict(base_trust.trusted_reviewer_public_keys),
        trusted_identity_source_sha256s=("a" * 64,),
        reviewed_a_availability_receipt_sha256s=("d" * 64,),
    )
    assert certify_a_reference_availability(
        availability,
        expected_category="audiojack", independently_certified_a_identity=receipt,
        manifest=manifest, attestation=attestation,
        trust_directory=trust, expected_trust_directory_sha256=trust.directory_sha256,
    ) == "AUTHORIZED_FOR_G1_REFERENCE_POPULATION"
    wrong_count = replace(availability, authorized_normal_reference_image_count=3)
    assert certify_a_reference_availability(
        wrong_count,
        expected_category="audiojack", independently_certified_a_identity=receipt,
        manifest=manifest, attestation=attestation,
        trust_directory=trust, expected_trust_directory_sha256=trust.directory_sha256,
    ) == "PENDING_A_NORMAL_IDENTITY_MATCH"


@pytest.mark.parametrize(
    ("role1", "role2"),
    [("A", "B"), ("A", "C"), ("B", "C")],
)
def test_source_group_overlap_fails_closed_across_roles(role1, role2):
    first = synthetic_receipt(role=role1, group="e" * 64)
    second = synthetic_receipt(role=role2, group="e" * 64)
    assert detect_cross_role_overlaps((first, second)) == ("e" * 64,)


def test_duplicate_group_in_same_role_is_not_treated_as_an_overlap_certificate():
    receipt = synthetic_receipt(role="A")
    with pytest.raises(R2L2BContractError, match="duplicate"):
        detect_cross_role_overlaps((receipt, receipt))


def test_g1_evidence_regimes_are_distinct_and_c64_is_additional():
    assert g1_regimes_are_distinct()
    assert G1_FULL_A.regime_id != G1_PROTOCOL_MATCHED.regime_id
    for regime in (G1_FULL_A, G1_PROTOCOL_MATCHED):
        assert regime.deployed_support_images == 4
        assert regime.additional_c64_calibration_normals == 64
        assert regime.c64_is_additional_to_four_supports is True
        assert regime.strict_four_shot_total is False
        assert regime.merges_into_c0_memory is False
        assert regime.radius_selected is False
