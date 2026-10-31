from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.calibration import r2_l2d_data_role_adjudication as l2d
from experiments.calibration import r2_l2c_metadata_census as l2c
from experiments.calibration.r2_l2a_allclass_inventory import (
    CANONICAL_CATEGORY_IDS,
    PINNED_SOURCE_SHA256,
    PINNED_SOURCE_URI,
    OfficialCatalogReceipt,
)
from experiments.calibration.r2_l2a_g1_geometry import (
    R1_FIXED_SUPPORT_CATEGORY_ID,
    R1_FIXED_SUPPORT_DIGEST_ALGORITHM,
    R1_FIXED_SUPPORT_SEED,
    R1_FIXED_SUPPORT_SHA256,
)
from experiments.calibration.r2_l2b_role_certification import (
    SOURCE_GROUP_DIGEST_ALGORITHM,
    SourceGroupMember,
    canonical_sha256,
    make_reviewer_attestation,
    make_reviewer_trust_directory,
    make_reviewed_identity_manifest,
    make_h_source_proposal,
    make_researcher_approval,
    make_role_identity_receipt,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _public_hex(key: Ed25519PrivateKey) -> str:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()


def _signed_json(key: Ed25519PrivateKey, payload: dict) -> str:
    return key.sign(l2c._canonical_bytes(payload)).hex()


def _trust(keys: dict[str, Ed25519PrivateKey], identity_sha: str, *, policy_key=None,
           approved_proposals=()):
    return make_reviewer_trust_directory(
        directory_id="synthetic-l2d-test-only",
        trusted_reviewer_public_keys={name: _public_hex(key) for name, key in keys.items()},
        trusted_policy_approver_public_keys=(
            {"policy-approver": _public_hex(policy_key)} if policy_key is not None else {}
        ),
        trusted_identity_source_sha256s=(identity_sha,),
        approved_proposal_sha256s=approved_proposals,
    )


def _approved_hsource_policy(policy_key: Ed25519PrivateKey):
    catalog = OfficialCatalogReceipt(
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
    proposal = make_h_source_proposal(catalog)
    approval = make_researcher_approval(
        proposal,
        researcher_id="policy-approver",
        decision_id="synthetic-policy-approval",
        approved_categories=CANONICAL_CATEGORY_IDS,
        approved_roles=("A", "B", "C"),
        approved_protected_policy_categories=proposal.protected_categories,
        private_key=policy_key,
    )
    return proposal, approval


def _role_evidence(category: str, role: str, groups: list[str], reviewer_key: Ed25519PrivateKey,
                   *, source_sha: str, reviewer_id: str = "role-reviewer"):
    receipt = make_role_identity_receipt(
        category_id=category,
        role=role,
        source_manifest_sha256=source_sha,
        producer_id=f"synthetic-{role.lower()}-producer",
        group_members=tuple(SourceGroupMember(group, 5) for group in groups),
    )
    if role == "A":
        manifest = make_reviewed_identity_manifest(
            receipt,
            population="authorized_normal_reference_only",
            normal_evidence_manifest_sha256=_sha("synthetic-normal-evidence"),
        )
    else:
        manifest = make_reviewed_identity_manifest(receipt)
    attestation = make_reviewer_attestation(
        receipt,
        manifest,
        reviewer_id=reviewer_id,
        decision_id=f"synthetic-review-{role.lower()}",
        private_key=reviewer_key,
    )
    return receipt, manifest, attestation


def _support_selection(category: str, seed: int, groups: list[str], reviewer_key: Ed25519PrivateKey):
    partial = l2c.SupportC64SelectionReceipt(
        schema="r2_l2c.support_c64_selection.v1",
        category_id=category,
        support_seed=seed,
        support_group_sha256s=tuple(groups[:4]),
        c64_group_sha256s=tuple(groups[4:68]),
        producer_id="synthetic-selection-producer",
        receipt_sha256="",
    )
    receipt = replace(
        partial,
        receipt_sha256=canonical_sha256(l2c._selection_payload(partial)),
    )
    attestation_payload = {
        "reviewer_id": "selection-reviewer",
        "decision_id": "synthetic-selection-review",
        "receipt_sha256": receipt.receipt_sha256,
    }
    attestation = l2c.SupportC64SelectionAttestation(
        reviewer_id="selection-reviewer",
        decision_id=attestation_payload["decision_id"],
        receipt_sha256=receipt.receipt_sha256,
        signature_ed25519_hex=_signed_json(reviewer_key, attestation_payload),
    )
    return receipt, attestation


def _selection_authorization(category, seed, policy, role_receipt, groups, key):
    partial = l2d.SupportC64SelectionAuthorization(
        schema=l2d.SUPPORT_SELECTION_AUTHORIZATION_SCHEMA,
        authorization_id="synthetic-selection-authorization",
        decision="AUTHORIZE_TRAIN_NORMAL_SUPPORT_C64_SELECTION",
        decision_id="synthetic-selection-authorization-decision",
        researcher_id="policy-approver",
        category_id=category,
        support_seed=seed,
        h_source_policy_proposal_sha256=policy.proposal_sha256,
        a_role_receipt_sha256=role_receipt.receipt_sha256,
        source_manifest_sha256=role_receipt.source_manifest_sha256,
        candidate_pool_sha256=l2d.candidate_pool_identity_sha256(
            category, role_receipt.source_manifest_sha256, groups,
        ),
        population="CERTIFIED_A_NORMAL_TRAIN_ONLY",
        support_count=4,
        additional_c64_count=64,
        allow_test_metadata=False,
        allow_outcome_labels=False,
        allow_masks=False,
        signature_ed25519_hex="",
        authorization_sha256="",
    )
    payload = l2d._selection_authorization_payload(partial)
    return replace(
        partial,
        signature_ed25519_hex=_signed_json(key, payload),
        authorization_sha256=canonical_sha256(payload),
    )


def _binding(category: str, seed: int, role_receipt, selection, fixed_bank_sha: str,
             selection_authorization, key: Ed25519PrivateKey, *, reviewer_id: str = "binding-reviewer"):
    receipt = l2d.make_support_c64_bank_binding_receipt(
        category_id=category,
        support_seed=seed,
        a_role_receipt_sha256=role_receipt.receipt_sha256,
        selection_receipt_sha256=selection.receipt_sha256,
        selection_authorization_sha256=selection_authorization.authorization_sha256,
        fixed_support_digest_algorithm=R1_FIXED_SUPPORT_DIGEST_ALGORITHM,
        fixed_support_sha256=fixed_bank_sha,
        producer_id="synthetic-binding-producer",
    )
    unsigned = l2d.SupportC64BankBindingAttestation(
        reviewer_id=reviewer_id,
        decision_id="synthetic-binding-review",
        receipt_sha256=receipt.receipt_sha256,
        signature_ed25519_hex="",
    )
    attestation = replace(
        unsigned,
        signature_ed25519_hex=_signed_json(
            key,
            l2d._binding_attestation_payload(unsigned),
        ),
    )
    return receipt, attestation


def test_hsource_proposal_is_deterministic_and_never_certifies_membership():
    groups = [_sha(f"group-{index}") for index in range(12)]
    args = dict(
        category_id="switch",
        source_group_sha256s=groups,
        policy_sha256=_sha("synthetic-role-policy"),
        seed=7,
        quotas={"A": 4, "B": 4, "C": 4},
    )
    first = l2d.propose_hsource_partition(**args)
    second = l2d.propose_hsource_partition(**args)
    assert first == second
    assert first.status == "PROPOSED"
    assert first.ranking_algorithm == l2d.PARTITION_RANKING_ALGORITHM
    assert dict(first.quotas) == {"A": 4, "B": 4, "C": 4}
    assert len({group for group, _role in first.group_assignments}) == 12
    assert {role for _group, role in first.group_assignments} == {"A", "B", "C"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_group_sha256s": [_sha("x"), _sha("x")]},
        {"source_group_sha256s": ["not-a-digest"]},
        {"source_population": "TEST_METADATA"},
        {"test_metadata_authorized": True},
        {"quotas": {"A": 1, "B": 1, "C": 0}},
    ],
)
def test_hsource_proposal_rejects_duplicates_bad_inputs_or_unapproved_test(overrides):
    args = dict(
        category_id="switch",
        source_group_sha256s=[_sha("group-a"), _sha("group-b")],
        policy_sha256=_sha("synthetic-policy"),
        seed=0,
        quotas={"A": 1, "B": 1, "C": 0},
    )
    args.update(overrides)
    with pytest.raises(l2d.R2L2DContractError):
        l2d.propose_hsource_partition(**args)


def test_hsource_proposal_reports_category_level_quota_shortage():
    with pytest.raises(l2d.R2L2DContractError, match="category-level candidate shortage"):
        l2d.propose_hsource_partition(
            "switch", [_sha("only-group")], policy_sha256=_sha("policy"), seed=0,
            quotas={"A": 1, "B": 1, "C": 1},
        )


def test_pairwise_group_overlap_reports_all_three_intersections_without_ids():
    a, b, c, d = (_sha(x) for x in ("a", "b", "c", "d"))
    audit = l2d.audit_pairwise_partition({
        "A": [a, b], "B": [a, c], "C": [b, c, d],
    })
    assert audit.status == "SOURCE_GROUP_OVERLAP_REJECTED"
    assert (audit.a_intersection_b, audit.a_intersection_c, audit.b_intersection_c) == (1, 1, 1)
    assert not hasattr(audit, "group_ids")


def test_pairwise_audit_rejects_duplicate_and_unknown_source_groups():
    a, b = _sha("a"), _sha("b")
    duplicate = l2d.audit_pairwise_partition({"A": [a, a], "B": [], "C": []})
    assert duplicate.status == "DUPLICATE_SOURCE_GROUP_REJECTED"
    unknown = l2d.audit_pairwise_partition(
        {"A": [a], "B": [], "C": []}, authorized_candidate_groups=[b]
    )
    assert unknown.status == "UNKNOWN_SOURCE_GROUP_REJECTED"
    assert unknown.unknown_group_count == 1


def test_missing_role_receipt_is_pending_not_certified_empty():
    result = l2d.adjudicate_category_roles(
        "switch", {"A": None, "B": None, "C": None}, None,
        expected_trust_directory_sha256=None,
    )
    assert result["partition_status"] == "A_CANDIDATES_IDENTIFIED_PARTITION_NOT_CERTIFIED"
    assert result["role_status"] == {"A": "PENDING_ROLE_RECEIPT", "B": "PENDING_ROLE_RECEIPT",
                                    "C": "PENDING_ROLE_RECEIPT"}


def test_valid_independently_signed_abc_receipts_certify_only_disjoint_groups():
    keys = {name: Ed25519PrivateKey.generate() for name in ("role-reviewer",)}
    policy_key = Ed25519PrivateKey.generate()
    policy, approval = _approved_hsource_policy(policy_key)
    source_sha = _sha("synthetic-authorized-source-manifest")
    trust = _trust(keys, source_sha, policy_key=policy_key,
                   approved_proposals=(policy.proposal_sha256,))
    group_sets = {role: [_sha(f"{role}-group")] for role in ("A", "B", "C")}
    evidence = {}
    for role, groups in group_sets.items():
        receipt, manifest, attestation = _role_evidence(
            "switch", role, groups, keys["role-reviewer"], source_sha=source_sha,
        )
        evidence[role] = l2d.RoleEvidence(receipt, manifest, attestation)
    result = l2d.adjudicate_category_roles(
        "switch", evidence, trust,
        expected_trust_directory_sha256=trust.directory_sha256,
        expected_source_manifest_sha256=source_sha,
        authorized_candidate_group_sha256s=[group for values in group_sets.values() for group in values],
        role_policy_proposal=policy,
        researcher_approval=approval,
    )
    assert result["partition_status"] == "SOURCE_GROUP_PARTITION_CERTIFIED"
    assert set(result["role_status"].values()) == {"CERTIFIED_FROM_SOURCE_EVIDENCE"}
    assert result["pairwise_overlap_counts"] == {
        "A_intersection_B": 0, "A_intersection_C": 0, "B_intersection_C": 0,
    }


def test_role_partition_needs_authorized_manifest_and_complete_candidate_universe():
    key = Ed25519PrivateKey.generate()
    policy_key = Ed25519PrivateKey.generate()
    policy, approval = _approved_hsource_policy(policy_key)
    source_sha = _sha("synthetic-role-source")
    trust = _trust({"role-reviewer": key}, source_sha, policy_key=policy_key,
                   approved_proposals=(policy.proposal_sha256,))
    groups = {role: [_sha(f"{role}-group")] for role in ("A", "B", "C")}
    evidence = {}
    for role, group_ids in groups.items():
        receipt, manifest, attestation = _role_evidence(
            "switch", role, group_ids, key, source_sha=source_sha,
        )
        evidence[role] = l2d.RoleEvidence(receipt, manifest, attestation)
    no_manifest = l2d.adjudicate_category_roles(
        "switch", evidence, trust,
        expected_trust_directory_sha256=trust.directory_sha256,
        authorized_candidate_group_sha256s=[group for values in groups.values() for group in values],
        role_policy_proposal=policy,
        researcher_approval=approval,
    )
    assert no_manifest["partition_status"] == "PENDING_AUTHORIZED_SOURCE_MANIFEST"
    no_universe = l2d.adjudicate_category_roles(
        "switch", evidence, trust,
        expected_trust_directory_sha256=trust.directory_sha256,
        expected_source_manifest_sha256=source_sha,
        role_policy_proposal=policy,
        researcher_approval=approval,
    )
    assert no_universe["partition_status"] == "PENDING_AUTHORIZED_SOURCE_GROUP_UNIVERSE"
    mismatched_source = l2d.adjudicate_category_roles(
        "switch", evidence, trust,
        expected_trust_directory_sha256=trust.directory_sha256,
        expected_source_manifest_sha256=_sha("different source"),
        authorized_candidate_group_sha256s=[group for values in groups.values() for group in values],
        role_policy_proposal=policy,
        researcher_approval=approval,
    )
    assert mismatched_source["partition_status"] == "A_CANDIDATES_IDENTIFIED_PARTITION_NOT_CERTIFIED"
    assert set(mismatched_source["role_status"].values()) == {"PENDING_SOURCE_MANIFEST_MISMATCH"}


def test_untrusted_source_or_bad_attestation_never_certifies_role():
    key = Ed25519PrivateKey.generate()
    source_sha = _sha("synthetic-role-source")
    receipt, manifest, attestation = _role_evidence(
        "switch", "A", [_sha("a-group")], key, source_sha=source_sha,
    )
    trust = _trust({}, _sha("different-source"))
    # Public adjudication composes the existing L2B trust gate.
    result = l2d.adjudicate_category_roles(
        "switch", {"A": l2d.RoleEvidence(receipt, manifest, attestation), "B": None, "C": None}, trust,
        expected_trust_directory_sha256=trust.directory_sha256,
    )
    assert result["role_status"]["A"] != "CERTIFIED_FROM_SOURCE_EVIDENCE"
    assert result["partition_status"] != "SOURCE_GROUP_PARTITION_CERTIFIED"


def _valid_support_binding():
    keys = {name: Ed25519PrivateKey.generate() for name in (
        "role-reviewer", "selection-reviewer", "binding-reviewer",
    )}
    policy_key = Ed25519PrivateKey.generate()
    policy, approval = _approved_hsource_policy(policy_key)
    source_sha = _sha("synthetic-authorized-role-manifest")
    trust = _trust(
        keys, source_sha, policy_key=policy_key,
        approved_proposals=(policy.proposal_sha256,),
    )
    category = "switch"
    seed = 0
    groups = [_sha(f"normal-group-{index}") for index in range(68)]
    role_receipt, role_manifest, role_attestation = _role_evidence(
        category, "A", groups, keys["role-reviewer"], source_sha=source_sha,
    )
    # Selection reviewer uses its own key and a synthetic signed receipt.
    selection, selection_attestation = _support_selection(
        category, seed, groups, keys["selection-reviewer"],
    )
    selection_authorization = _selection_authorization(
        category, seed, policy, role_receipt, groups, policy_key,
    )
    binding_receipt, binding_attestation = _binding(
        category, seed, role_receipt, selection, R1_FIXED_SUPPORT_SHA256,
        selection_authorization,
        keys["binding-reviewer"],
    )
    return {
        "keys": keys, "trust": trust, "groups": groups,
        "role_receipt": role_receipt, "role_manifest": role_manifest,
        "role_attestation": role_attestation,
        "selection": selection, "selection_attestation": selection_attestation,
        "selection_authorization": selection_authorization,
        "binding_receipt": binding_receipt, "binding_attestation": binding_attestation,
        "role_policy_proposal": policy, "researcher_approval": approval,
        "category": category, "seed": seed,
    }


def _assess_binding(case, **overrides):
    args = {
        "a_role_receipt": case["role_receipt"],
        "a_role_manifest": case["role_manifest"],
        "a_role_attestation": case["role_attestation"],
        "selection_receipt": case["selection"],
        "selection_attestation": case["selection_attestation"],
        "candidate_group_sha256s": case["groups"],
        "binding_receipt": case["binding_receipt"],
        "binding_attestation": case["binding_attestation"],
        "trust_directory": case["trust"],
        "expected_trust_directory_sha256": case["trust"].directory_sha256,
        "expected_category_id": case["category"],
        "expected_support_seed": case["seed"],
        "selection_authorization": case["selection_authorization"],
        "role_policy_proposal": case["role_policy_proposal"],
        "researcher_approval": case["researcher_approval"],
    }
    args.update(overrides)
    return l2d.assess_support_c64_bank_binding(**args)


def test_support_c64_and_fixed_bank_require_composed_independent_receipts():
    case = _valid_support_binding()
    assert _assess_binding(case) == "SUPPORT_C64_AND_FIXED_BANK_BOUND"
    assert case["binding_receipt"].fixed_support_sha256 == R1_FIXED_SUPPORT_SHA256
    assert case["binding_receipt"].fixed_support_receipt_sha256


def test_support_binding_rejects_missing_authorization_wrong_category_and_short_pool():
    case = _valid_support_binding()
    assert _assess_binding(case, a_role_receipt=None) == "PENDING_REQUIRED_IDENTITY_OR_AUTHORIZATION"
    assert _assess_binding(case, selection_authorization=None) == "PENDING_SELECTION_AUTHORIZATION"
    assert _assess_binding(case, researcher_approval=None) == "PENDING_H_SOURCE_POLICY_APPROVAL"
    assert _assess_binding(case, expected_category_id="cap") == "SUPPORT_CATEGORY_OR_SEED_MISMATCH"
    assert _assess_binding(case, candidate_group_sha256s=case["groups"][:67]) == (
        "CANDIDATE_POOL_INSUFFICIENT_FOR_4_SUPPORTS_PLUS_C64"
    )


def test_selection_authorization_rejects_test_scope_tampering_and_candidate_pool_swap():
    case = _valid_support_binding()
    test_scope = replace(case["selection_authorization"], allow_test_metadata=True)
    assert _assess_binding(case, selection_authorization=test_scope) == "SELECTION_AUTHORIZATION_SCOPE_INVALID"
    bad_signature = replace(
        case["selection_authorization"], signature_ed25519_hex="00",
    )
    assert _assess_binding(case, selection_authorization=bad_signature) == (
        "PENDING_SELECTION_AUTHORIZATION_SIGNATURE"
    )
    changed_pool = case["groups"][:-1] + [_sha("unauthorized-extra-group")]
    assert _assess_binding(case, candidate_group_sha256s=changed_pool) == (
        "SELECTION_AUTHORIZATION_CANDIDATE_POOL_MISMATCH"
    )


def test_support_binding_rejects_support_c64_overlap_and_invalid_signature():
    case = _valid_support_binding()
    overlap_selection = replace(
        case["selection"],
        c64_group_sha256s=(case["groups"][3],) + tuple(case["groups"][4:67]),
    )
    assert _assess_binding(case, selection_receipt=overlap_selection) == "SUPPORT_C64_OVERLAP_REJECTED"
    bad_signature = replace(case["binding_attestation"], signature_ed25519_hex="00")
    assert _assess_binding(case, binding_attestation=bad_signature) == "PENDING_INDEPENDENT_BINDING_REVIEW"
    changed_binding = replace(
        case["binding_receipt"],
        support_c64_selection_receipt_sha256=_sha("different selection"),
    )
    assert _assess_binding(case, binding_receipt=changed_binding) == "INVALID_SUPPORT_C64_FIXED_BANK_BINDING"


def test_binding_review_must_be_distinct_from_selector_and_fixed_bank_is_category_specific():
    case = _valid_support_binding()
    same_reviewer = replace(
        case["binding_attestation"],
        reviewer_id=case["selection_attestation"].reviewer_id,
    )
    assert _assess_binding(case, binding_attestation=same_reviewer) == "PENDING_INDEPENDENT_BINDING_REVIEW"
    same_a_reviewer = replace(
        case["binding_attestation"],
        reviewer_id=case["role_attestation"].reviewer_id,
    )
    assert _assess_binding(case, binding_attestation=same_a_reviewer) == "PENDING_INDEPENDENT_BINDING_REVIEW"
    with pytest.raises(l2d.R2L2DContractError, match="cross-category"):
        l2d.make_support_c64_bank_binding_receipt(
            category_id="bottle_cap", support_seed=0,
            a_role_receipt_sha256=_sha("A receipt"),
            selection_receipt_sha256=_sha("selection receipt"),
            selection_authorization_sha256=_sha("selection authorization"),
            fixed_support_digest_algorithm=R1_FIXED_SUPPORT_DIGEST_ALGORITHM,
            fixed_support_sha256=R1_FIXED_SUPPORT_SHA256,
            producer_id="synthetic-producer",
        )
    with pytest.raises(l2d.R2L2DContractError, match="switch/seed0"):
        l2d.make_support_c64_bank_binding_receipt(
            category_id=R1_FIXED_SUPPORT_CATEGORY_ID,
            support_seed=R1_FIXED_SUPPORT_SEED,
            a_role_receipt_sha256=_sha("A receipt"),
            selection_receipt_sha256=_sha("selection receipt"),
            selection_authorization_sha256=_sha("selection authorization"),
            fixed_support_digest_algorithm=R1_FIXED_SUPPORT_DIGEST_ALGORITHM,
            fixed_support_sha256=_sha("wrong switch bank"),
            producer_id="synthetic-producer",
        )


def _synthetic_census_pair(root):
    private_rows = []
    source_sha_by_category = {}
    census_source_sha = l2c._source_sha256(Path(l2c.__file__))
    identity_source_sha = l2c._source_sha256(Path(l2c.identity_contract.__file__))
    for category in CANONICAL_CATEGORY_IDS:
        group = _sha(f"synthetic-{category}-group")
        source_sha = _sha(f"synthetic-{category}-metadata")
        source_sha_by_category[category] = source_sha
        role_status = {
            "partition_status": "A_CANDIDATES_IDENTIFIED_PARTITION_NOT_CERTIFIED",
            "role_status": {"A": "PENDING_INDEPENDENT_REVIEW", "B": "PENDING_INDEPENDENT_REVIEW",
                            "C": "C_IDENTITY_CERTIFICATE_INCOMPLETE"},
            "role_reasons": {"A": "missing", "B": "missing", "C": "missing"},
            "cross_role_overlap_count": 0,
            "c_identity_certificate_status": "C_IDENTITY_CERTIFICATE_INCOMPLETE",
        }
        private_rows.append({
            "category_id": category,
            "source_json_sha256": source_sha,
            "source_group_digest_algorithm": SOURCE_GROUP_DIGEST_ALGORITHM,
            "group_members": [{"source_group_sha256": group, "camera_view_count": 5}],
            "directory_serial_overlap_count": None,
            "role_certification_status": role_status,
            "contains_raw_sample_keys": False,
            "contains_image_paths": False,
            "contains_labels_or_masks": False,
        })

    auth_sha = _sha("synthetic-census-authorization")
    public_rows = []
    for private_row in private_rows:
        category = private_row["category_id"]
        public_rows.append({
            "category_id": category,
            "source_json_sha256": source_sha_by_category[category],
            "parser_source_sha256": census_source_sha,
            "identity_parser_source_sha256": identity_source_sha,
            "metadata_availability_status": "METADATA_AVAILABLE",
            "eligible_normal_image_count": 5,
            "distinct_normal_source_group_count": 1,
            "at_least_68_legal_normal_groups": False,
            "normal_camera_view_distribution": {"5": 1},
            "malformed_identity_count": 0,
            "duplicate_image_identity_count": 0,
            "ambiguous_group_identity_count": 0,
            "directory_serial_overlap_status": "NOT_AUTHORIZED",
            "directory_serial_overlap_count": None,
            "a_role_candidate_status": "A_CANDIDATES_IDENTIFIED",
            "abc_partition_status": "A_CANDIDATES_IDENTIFIED_PARTITION_NOT_CERTIFIED",
            "support_and_c64_identity_status": "PENDING_SEPARATE_SUPPORT_C64_SELECTION_AND_REVIEW",
        })
    public = {
        "schema": "r2_l2c.metadata_census_report.v1",
        "status": "R2_L2C_METADATA_CENSUS_COMPLETE_PARTITION_PENDING",
        "operation": "R2_L2C_AUTHORIZED_TRAIN_METADATA_CENSUS_V1",
        "authorization_id": "synthetic-only",
        "authorization_sha256": auth_sha,
        "source_group_digest_algorithm": SOURCE_GROUP_DIGEST_ALGORITHM,
        "identity_contract_version": "realid-offical-sample-key-v2",
        "grouping_unit": "official_sample_key",
        "public_catalog_source_sha256": PINNED_SOURCE_SHA256,
        "public_catalog_source_uri": PINNED_SOURCE_URI,
        "target_category_count": 30,
        "authorized_category_count": 30,
        "metadata_files_opened": 30,
        "metadata_available_category_count": 30,
        "missing_or_unavailable_categories": [],
        "partition_status": "A_CANDIDATES_IDENTIFIED_PARTITION_NOT_CERTIFIED",
        "local_data_io_performed": True,
        "test_section_parsed": False,
        "test_section_bytes_encountered": True,
        "image_decoding_performed": False,
        "feature_extraction_performed": False,
        "geometry_performed": False,
        "categories": public_rows,
    }
    private = {
        "schema": "r2_l2c.private_group_receipts.v1",
        "authorization_id": "synthetic-only",
        "authorization_sha256": auth_sha,
        "source_group_digest_algorithm": SOURCE_GROUP_DIGEST_ALGORITHM,
        "receipts": private_rows,
    }
    (root / "census_report.json").write_text(json.dumps(public), encoding="utf-8")
    (root / "private_source_group_receipts.json").write_text(json.dumps(private), encoding="utf-8")
    return root


def test_artifact_pair_audits_only_aggregate_and_keeps_provenance_blocked(tmp_path):
    root = tmp_path / "l2c-output"
    root.mkdir()
    _synthetic_census_pair(root)
    result = l2d.audit_l2c_artifact_pair(
        root,
        expected_normal_image_count=150,
        expected_source_group_count=30,
        minimum_groups_per_category=1,
    )
    assert result["artifact_audit_status"] == "ARTIFACT_AUDIT_PASS"
    assert result["artifact_pair_status"] == "INTERNALLY_CONSISTENT_REPORTED"
    assert result["provenance_status"] == "BLOCKED_INPUT"
    assert result["category_count"] == 30
    assert result["source_group_count"] == 30
    assert result["role_slots_uncertified"] == 90
    assert result["private_receipt_file_sha256_emitted"] is False
    assert result["test_section_parsed"] is False


def test_artifact_pair_fails_closed_on_missing_or_contradictory_completion_marker(tmp_path):
    root = tmp_path / "l2c-output"
    root.mkdir()
    _synthetic_census_pair(root)
    public_path = root / "census_report.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    public["status"] = "R2_L2C_BLOCKED"
    public_path.write_text(json.dumps(public), encoding="utf-8")
    with pytest.raises(l2d.R2L2DContractError, match="completion marker"):
        l2d.audit_l2c_artifact_pair(
            root,
            expected_normal_image_count=150,
            expected_source_group_count=30,
            minimum_groups_per_category=1,
        )


@pytest.mark.parametrize("mutation", ("source_hash", "partial_camera_group"))
def test_artifact_pair_rejects_source_mismatch_and_split_camera_group(tmp_path, mutation):
    root = tmp_path / "l2c-output"
    root.mkdir()
    _synthetic_census_pair(root)
    public_path = root / "census_report.json"
    private_path = root / "private_source_group_receipts.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    if mutation == "source_hash":
        public["categories"][0]["source_json_sha256"] = _sha("mismatched-source")
    else:
        public["categories"][0]["eligible_normal_image_count"] = 4
        public["categories"][0]["normal_camera_view_distribution"] = {"4": 1}
        private["receipts"][0]["group_members"][0]["camera_view_count"] = 4
    public_path.write_text(json.dumps(public), encoding="utf-8")
    private_path.write_text(json.dumps(private), encoding="utf-8")
    with pytest.raises(l2d.R2L2DContractError):
        l2d.audit_l2c_artifact_pair(
            root,
            expected_normal_image_count=150,
            expected_source_group_count=30,
            minimum_groups_per_category=1,
        )
