"""R2-L2B proposed policy, access scope, and independent role certification.

This module is metadata-only. It provides no dataset discovery or extraction
entrypoint. digest_image_path accepts a caller-provided path string and delegates
identity parsing to the existing Real-IAD preflight parser; it never opens it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from experiments.calibration.r2_l2a_allclass_inventory import (
    AReferenceAvailabilityReceipt,
    CATALOG_EXTRACTION_ALGORITHM,
    CANONICAL_CATEGORY_IDS,
    EXPECTED_CATEGORY_COUNT,
    GROUPING_UNIT,
    IDENTITY_CONTRACT_VERSION,
    PINNED_CONFIG_VERSION,
    PINNED_SOURCE_URI,
    PINNED_SOURCE_SHA256,
    ROLE_NAMES,
    OfficialCatalogReceipt,
    RolePolicyEntry,
    make_role_policy,
)


SCHEMA = "r2_l2b.role_certification.v1"
PROPOSAL_ID = "r2_l2b_h_source_allclass_roles"
PROPOSAL_VERSION = "1.0.0-proposed"
STRATEGY_H_SOURCE = "H-SOURCE"
STRATEGY_H_CLASS = "H-CLASS"
PROTECTED_DEFAULT = ("mint", "pcb")
ROLE_POLICY_STATES = {
    "PROPOSED",
    "PENDING_AUTHORIZATION_PROTECTED",
    "APPROVED",
    "NOT_APPLICABLE_WITH_REVIEWED_REASON",
}
IDENTITY_METADATA_FIELDS = {
    "official_sample_key", "sample_key", "directory_serial_key",
    "camera_id", "image_id", "image_path", "file_name",
}
CONDITION_BEARING_IDENTITY_FIELDS = {
    "official_sample_key", "sample_key", "image_id", "image_path", "file_name",
}
TRAIN_LABEL_FIELDS = {
    "normal_status", "is_good", "label", "anomaly_label",
    "ground_truth", "defect", "defect_type",
}
TEST_OUTCOME_FIELDS = {
    "label", "anomaly_label", "ground_truth", "is_anomaly",
    "defect", "defect_type",
}
MASK_FIELDS = {"mask", "mask_path"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class R2L2BContractError(ValueError):
    """An R2-L2B policy, identity, or access receipt is inconsistent."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise R2L2BContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _valid_catalog(catalog: OfficialCatalogReceipt) -> tuple[str, ...]:
    categories = tuple(catalog.categories)
    if (
        categories != CANONICAL_CATEGORY_IDS
        or len(categories) != EXPECTED_CATEGORY_COUNT
        or len(set(categories)) != EXPECTED_CATEGORY_COUNT
        or catalog.source_sha256 != PINNED_SOURCE_SHA256
        or catalog.source_uri != PINNED_SOURCE_URI
        or catalog.source_commit != "98c28ffbe95b380d17a3777d77219152368c19e9"
        or catalog.source_path != "experiments/RealIAD-full/config.yaml"
        or catalog.release_id != PINNED_CONFIG_VERSION
        or catalog.dataset_config_version != PINNED_CONFIG_VERSION
        or catalog.extraction_algorithm != CATALOG_EXTRACTION_ALGORITHM
        or not catalog.official_source_verified
        or not catalog.non_label_bearing
        or catalog.contains_image_paths
        or catalog.contains_labels_or_masks
    ):
        raise R2L2BContractError(
            "proposal requires the previously verified category-only pinned public catalog")
    return categories


@dataclass(frozen=True)
class ProposedRole:
    category_id: str
    role: str
    state: str
    rationale: str | None = None
    reviewed_reason_sha256: str | None = None


@dataclass(frozen=True)
class RolePolicyProposal:
    proposal_id: str
    version: str
    strategy: str
    catalog_source_sha256: str
    identity_contract_version: str
    grouping_unit: str
    roles: tuple[ProposedRole, ...]
    frozen_role_policy_sha256: str
    protected_categories: tuple[str, ...]
    additional_protected_categories_unreviewed: bool
    proposal_sha256: str


def _proposal_payload(proposal: RolePolicyProposal) -> dict[str, Any]:
    payload = asdict(proposal)
    payload.pop("proposal_sha256", None)
    return payload


def make_h_source_proposal(
    catalog: OfficialCatalogReceipt,
    *,
    additional_protected_categories: Sequence[str] = (),
) -> RolePolicyProposal:
    """Build a complete H-SOURCE proposal without approving any local access."""
    categories = _valid_catalog(catalog)
    additional = tuple(sorted(set(additional_protected_categories)))
    if any(category not in categories for category in additional):
        raise R2L2BContractError("additional protected category is outside the public catalog")
    protected = tuple(sorted(set(PROTECTED_DEFAULT).union(additional)))
    assignments: list[ProposedRole] = []
    contract_entries: list[RolePolicyEntry] = []
    for category in categories:
        for role in ROLE_NAMES:
            protected_role = category in protected
            assignments.append(ProposedRole(
                category_id=category,
                role=role,
                state=("PENDING_AUTHORIZATION_PROTECTED" if protected_role else "PROPOSED"),
                rationale=(
                    "Retained sealed pending explicit researcher authorization."
                    if protected_role else
                    "Proposed source-group role; requires researcher approval and local authorization."
                ),
            ))
            contract_entries.append(RolePolicyEntry(
                category_id=category,
                role=role,
                disposition="REQUIRED",
            ))
    frozen_policy = make_role_policy(PROPOSAL_ID, PROPOSAL_VERSION, contract_entries)
    partial = RolePolicyProposal(
        proposal_id=PROPOSAL_ID,
        version=PROPOSAL_VERSION,
        strategy=STRATEGY_H_SOURCE,
        catalog_source_sha256=catalog.source_sha256,
        identity_contract_version=IDENTITY_CONTRACT_VERSION,
        grouping_unit=GROUPING_UNIT,
        roles=tuple(assignments),
        frozen_role_policy_sha256=frozen_policy.policy_sha256,
        protected_categories=protected,
        additional_protected_categories_unreviewed=True,
        proposal_sha256="",
    )
    return replace(partial, proposal_sha256=canonical_sha256(_proposal_payload(partial)))


def validate_role_policy_proposal(proposal: RolePolicyProposal) -> None:
    if proposal.strategy not in {STRATEGY_H_SOURCE, STRATEGY_H_CLASS}:
        raise R2L2BContractError("unknown holdout strategy")
    if proposal.strategy == STRATEGY_H_CLASS:
        raise R2L2BContractError(
            "H-CLASS requires a separately frozen category set; none is selected in this proposal")
    _require_sha256(proposal.catalog_source_sha256, "public catalog source SHA-256")
    _require_sha256(proposal.frozen_role_policy_sha256, "frozen role policy SHA-256")
    _require_sha256(proposal.proposal_sha256, "proposal SHA-256")
    expected_pairs = {(category, role) for category in CANONICAL_CATEGORY_IDS for role in ROLE_NAMES}
    actual_pairs: set[tuple[str, str]] = set()
    for item in proposal.roles:
        pair = (item.category_id, item.role)
        if pair not in expected_pairs or pair in actual_pairs:
            raise R2L2BContractError("proposal has unknown or duplicate category-role rows")
        actual_pairs.add(pair)
        if item.state not in ROLE_POLICY_STATES:
            raise R2L2BContractError("proposal has an unknown role-policy state")
        if item.state == "NOT_APPLICABLE_WITH_REVIEWED_REASON":
            if not (item.rationale or "").strip() or not item.reviewed_reason_sha256:
                raise R2L2BContractError(
                    "NOT_APPLICABLE requires a written, independently reviewed reason")
            _require_sha256(item.reviewed_reason_sha256, "reviewed N/A reason SHA-256")
        elif item.reviewed_reason_sha256 is not None:
            raise R2L2BContractError("reviewed N/A digest is only valid for an N/A row")
    if actual_pairs != expected_pairs:
        raise R2L2BContractError("proposal must cover all 30 categories and A/B/C roles")
    if tuple(proposal.protected_categories) != tuple(sorted(set(proposal.protected_categories))):
        raise R2L2BContractError("protected categories must be unique and canonically ordered")
    if not set(PROTECTED_DEFAULT).issubset(proposal.protected_categories):
        raise R2L2BContractError("Mint and PCB must remain protected by default")
    if canonical_sha256(_proposal_payload(proposal)) != proposal.proposal_sha256:
        raise R2L2BContractError("proposal digest does not match its contents")
    if any(item.state == "APPROVED" for item in proposal.roles):
        raise R2L2BContractError("a proposal cannot approve itself")
    frozen_entries = tuple(
        RolePolicyEntry(
            category_id=item.category_id,
            role=item.role,
            disposition=(
                "NOT_APPLICABLE"
                if item.state == "NOT_APPLICABLE_WITH_REVIEWED_REASON"
                else "REQUIRED"
            ),
            rationale=(
                item.rationale
                if item.state == "NOT_APPLICABLE_WITH_REVIEWED_REASON"
                else None
            ),
        )
        for item in proposal.roles
    )
    frozen_policy = make_role_policy(proposal.proposal_id, proposal.version, frozen_entries)
    if frozen_policy.policy_sha256 != proposal.frozen_role_policy_sha256:
        raise R2L2BContractError("frozen role-policy digest does not match all proposal rows")


@dataclass(frozen=True)
class ResearcherApproval:
    proposal_sha256: str
    researcher_id: str
    decision: str
    decision_id: str
    approved_categories: tuple[str, ...]
    approved_roles: tuple[str, ...]
    approved_protected_policy_categories: tuple[str, ...]
    signature_ed25519_hex: str
    approval_sha256: str


def _approval_payload(approval: ResearcherApproval) -> dict[str, Any]:
    payload = asdict(approval)
    payload.pop("signature_ed25519_hex")
    payload.pop("approval_sha256")
    return payload


def make_researcher_approval(
    proposal: RolePolicyProposal,
    *,
    researcher_id: str,
    decision_id: str,
    approved_categories: Sequence[str],
    approved_roles: Sequence[str],
    approved_protected_policy_categories: Sequence[str],
    private_key: Any,
) -> ResearcherApproval:
    """Prepare a decision receipt; this does not authorize local data access."""
    partial = ResearcherApproval(
        proposal_sha256=proposal.proposal_sha256,
        researcher_id=researcher_id,
        decision="APPROVE_H_SOURCE",
        decision_id=decision_id,
        approved_categories=tuple(approved_categories),
        approved_roles=tuple(approved_roles),
        approved_protected_policy_categories=tuple(approved_protected_policy_categories),
        signature_ed25519_hex="",
        approval_sha256="",
    )
    signature = _ed25519_sign(private_key, _approval_payload(partial))
    signed_digest = canonical_sha256({
        "payload": _approval_payload(partial),
        "signature_ed25519_hex": signature,
    })
    return replace(partial, signature_ed25519_hex=signature, approval_sha256=signed_digest)


@dataclass(frozen=True)
class ReviewerTrustDirectory:
    """Externally provisioned trust facts; caller pins its SHA out of band."""

    directory_id: str
    trusted_reviewer_ids: tuple[str, ...]
    trusted_reviewer_public_keys: tuple[tuple[str, str], ...]
    trusted_policy_approver_ids: tuple[str, ...]
    trusted_policy_approver_public_keys: tuple[tuple[str, str], ...]
    trusted_identity_source_sha256s: tuple[str, ...]
    reviewed_empty_role_policy_sha256s: tuple[str, ...]
    reviewed_a_availability_receipt_sha256s: tuple[str, ...]
    approved_proposal_sha256s: tuple[str, ...]
    directory_sha256: str


def reviewer_trust_directory_sha256(directory: ReviewerTrustDirectory) -> str:
    payload = asdict(directory)
    payload.pop("directory_sha256")
    return canonical_sha256(payload)


def make_reviewer_trust_directory(
    *,
    directory_id: str,
    trusted_reviewer_public_keys: Mapping[str, str] | None = None,
    trusted_policy_approver_public_keys: Mapping[str, str] | None = None,
    trusted_identity_source_sha256s: Sequence[str],
    reviewed_empty_role_policy_sha256s: Sequence[str] = (),
    reviewed_a_availability_receipt_sha256s: Sequence[str] = (),
    approved_proposal_sha256s: Sequence[str] = (),
) -> ReviewerTrustDirectory:
    reviewer_keys = dict(trusted_reviewer_public_keys or {})
    policy_keys = dict(trusted_policy_approver_public_keys or {})
    fields = {
        "directory_id": directory_id,
        "trusted_reviewer_ids": tuple(sorted(reviewer_keys)),
        "trusted_reviewer_public_keys": tuple(sorted(reviewer_keys.items())),
        "trusted_policy_approver_ids": tuple(sorted(policy_keys)),
        "trusted_policy_approver_public_keys": tuple(sorted(policy_keys.items())),
        "trusted_identity_source_sha256s": tuple(sorted(set(trusted_identity_source_sha256s))),
        "reviewed_empty_role_policy_sha256s": tuple(sorted(set(reviewed_empty_role_policy_sha256s))),
        "reviewed_a_availability_receipt_sha256s": tuple(sorted(set(reviewed_a_availability_receipt_sha256s))),
        "approved_proposal_sha256s": tuple(sorted(set(approved_proposal_sha256s))),
        "directory_sha256": "",
    }
    for value in fields["trusted_identity_source_sha256s"]:
        _require_sha256(value, "trusted identity source SHA-256")
    for _principal, key_hex in (
        fields["trusted_reviewer_public_keys"]
        + fields["trusted_policy_approver_public_keys"]
    ):
        try:
            if (
                not isinstance(key_hex, str)
                or re.fullmatch(r"[0-9a-f]{64}", key_hex) is None
                or len(bytes.fromhex(key_hex)) != 32
            ):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise R2L2BContractError("trusted Ed25519 public key must be 32-byte lowercase hex") from exc
    for value in fields["reviewed_empty_role_policy_sha256s"]:
        _require_sha256(value, "reviewed empty-role policy SHA-256")
    for value in fields["approved_proposal_sha256s"]:
        _require_sha256(value, "approved proposal SHA-256")
    for value in fields["reviewed_a_availability_receipt_sha256s"]:
        _require_sha256(value, "reviewed A availability receipt SHA-256")
    partial = ReviewerTrustDirectory(**fields)
    return replace(partial, directory_sha256=reviewer_trust_directory_sha256(partial))


def _ed25519_sign(private_key: Any, payload: Mapping[str, Any]) -> str:
    try:
        message = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return private_key.sign(message).hex()
    except Exception as exc:
        raise R2L2BContractError("a valid external Ed25519 signing key is required") from exc


def _ed25519_verify(public_key_hex: str, payload: Mapping[str, Any], signature_hex: str) -> bool:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        message = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        public_key.verify(bytes.fromhex(signature_hex), message)
        return True
    except Exception:
        return False


def evaluate_policy_approval(
    proposal: RolePolicyProposal,
    approval: ResearcherApproval | None,
    trust_directory: ReviewerTrustDirectory | None,
    *,
    expected_trust_directory_sha256: str | None,
) -> str:
    """A digest alone never approves a proposal; a separately pinned trust root is required."""
    validate_role_policy_proposal(proposal)
    if approval is None:
        return "PROPOSED_PENDING_RESEARCHER_APPROVAL"
    if (
        approval.decision != "APPROVE_H_SOURCE"
        or approval.proposal_sha256 != proposal.proposal_sha256
        or not approval.researcher_id.strip()
        or not approval.decision_id.strip()
        or canonical_sha256({
            "payload": _approval_payload(approval),
            "signature_ed25519_hex": approval.signature_ed25519_hex,
        }) != approval.approval_sha256
        or tuple(approval.approved_categories) != CANONICAL_CATEGORY_IDS
        or tuple(approval.approved_roles) != ROLE_NAMES
        or tuple(approval.approved_protected_policy_categories)
        != tuple(sorted(set(approval.approved_protected_policy_categories)))
        or not set(proposal.protected_categories).issubset(
            approval.approved_protected_policy_categories
        )
        or any(
            category not in CANONICAL_CATEGORY_IDS
            for category in approval.approved_protected_policy_categories
        )
    ):
        return "PENDING_INVALID_OR_MISMATCHED_APPROVAL"
    if trust_directory is None or expected_trust_directory_sha256 is None:
        return "PENDING_EXTERNAL_APPROVAL_TRUST_ROOT"
    if (
        reviewer_trust_directory_sha256(trust_directory) != trust_directory.directory_sha256
        or trust_directory.directory_sha256 != expected_trust_directory_sha256
        or approval.researcher_id not in trust_directory.trusted_policy_approver_ids
        or proposal.proposal_sha256 not in trust_directory.approved_proposal_sha256s
    ):
        return "PENDING_EXTERNAL_APPROVAL_TRUST_ROOT"
    policy_keys = dict(trust_directory.trusted_policy_approver_public_keys)
    if not _ed25519_verify(
        policy_keys[approval.researcher_id],
        _approval_payload(approval),
        approval.signature_ed25519_hex,
    ):
        return "PENDING_EXTERNAL_APPROVAL_TRUST_ROOT"
    return "APPROVED_POLICY_PENDING_LOCAL_ACCESS"


@dataclass(frozen=True)
class AccessScopeManifest:
    schema: str
    approval_state: str
    approval_id: str | None
    approved_local_root: str | None
    approved_categories: tuple[str, ...]
    prospective_roots_only_not_authorized: tuple[str, ...]
    allow_physical_json_file_bytes: bool
    acknowledge_train_and_test_may_share_physical_json: bool
    parsed_sections: tuple[str, ...]
    allowed_train_metadata_fields: tuple[str, ...]
    allowed_test_metadata_fields: tuple[str, ...]
    allow_train_normal_ok_status: bool
    allow_official_image_identity_metadata: bool
    allow_official_sample_key_condition_tokens: bool
    explicitly_authorized_protected_categories: tuple[str, ...]
    independent_c_identity_certificate_supplied: bool
    allow_feature_extraction: bool
    allow_outcome_label_access: bool
    allow_mask_access: bool
    output_disclosure_fields: tuple[str, ...]
    maximum_output_bytes: int


def default_access_scope() -> AccessScopeManifest:
    """Default-deny template. Proposed paths do not grant read permission."""
    return AccessScopeManifest(
        schema="r2_l2b.local_data_access_scope.v1",
        approval_state="PENDING_RESEARCHER_DECISION",
        approval_id=None,
        approved_local_root=None,
        approved_categories=(),
        prospective_roots_only_not_authorized=(
            "G:/dataset/realiad",
            "G:/dataset/realiad/jsons/realiad_jsons",
        ),
        allow_physical_json_file_bytes=False,
        acknowledge_train_and_test_may_share_physical_json=False,
        parsed_sections=(),
        allowed_train_metadata_fields=(),
        allowed_test_metadata_fields=(),
        allow_train_normal_ok_status=False,
        allow_official_image_identity_metadata=False,
        allow_official_sample_key_condition_tokens=False,
        explicitly_authorized_protected_categories=(),
        independent_c_identity_certificate_supplied=False,
        allow_feature_extraction=False,
        allow_outcome_label_access=False,
        allow_mask_access=False,
        output_disclosure_fields=(),
        maximum_output_bytes=0,
    )


def access_scope_errors(scope: AccessScopeManifest) -> tuple[str, ...]:
    errors: list[str] = []
    if scope.approval_state != "APPROVED":
        errors.append("researcher approval is absent")
    if not scope.approval_id:
        errors.append("approval id is absent")
    if not scope.approved_local_root:
        errors.append("approved local root is absent")
    if not scope.approved_categories:
        errors.append("approved category set is empty")
    if scope.allow_physical_json_file_bytes and not scope.acknowledge_train_and_test_may_share_physical_json:
        errors.append("mixed train/test JSON byte exposure was not acknowledged")
    if "train" in scope.parsed_sections and not scope.allow_physical_json_file_bytes:
        errors.append("train parsing is requested without physical JSON byte permission")
    if "test" in scope.parsed_sections and not scope.allow_physical_json_file_bytes:
        errors.append("test parsing is requested without physical JSON byte permission")
    if "train" in scope.parsed_sections and not scope.allowed_train_metadata_fields:
        errors.append("train section has no field allowlist")
    if "test" in scope.parsed_sections and not scope.allowed_test_metadata_fields:
        errors.append("test section has no field allowlist")
    if (
        "train" in scope.parsed_sections
        and set(scope.allowed_train_metadata_fields).intersection(TRAIN_LABEL_FIELDS)
        and not scope.allow_train_normal_ok_status
    ):
        errors.append("train label-bearing fields are not explicitly authorized")
    if (
        "test" in scope.parsed_sections
        and set(scope.allowed_test_metadata_fields).intersection(TEST_OUTCOME_FIELDS)
        and not scope.allow_outcome_label_access
    ):
        errors.append("test outcome fields are not explicitly authorized")
    if (
        "test" in scope.parsed_sections
        and set(scope.allowed_test_metadata_fields).intersection(MASK_FIELDS)
        and not scope.allow_mask_access
    ):
        errors.append("test masks are not explicitly authorized")
    if any(
        set(fields).intersection(IDENTITY_METADATA_FIELDS)
        for fields in (scope.allowed_train_metadata_fields, scope.allowed_test_metadata_fields)
    ) and not scope.allow_official_image_identity_metadata:
        errors.append("official image identity metadata is not explicitly authorized")
    if scope.allow_train_normal_ok_status and "train" not in scope.parsed_sections:
        errors.append("normal/OK status requires explicit train-section access")
    if any(
        set(fields).intersection(CONDITION_BEARING_IDENTITY_FIELDS)
        for fields in (scope.allowed_train_metadata_fields, scope.allowed_test_metadata_fields)
    ) and not scope.allow_official_sample_key_condition_tokens:
        errors.append("official sample-key OK/NG/defect tokens are not explicitly authorized")
    if scope.allow_outcome_label_access and "test" not in scope.parsed_sections:
        errors.append("outcome-label access requires explicit test-section access")
    if scope.maximum_output_bytes <= 0:
        errors.append("positive output-byte limit is absent")
    if any(category not in CANONICAL_CATEGORY_IDS for category in scope.approved_categories):
        errors.append("approved category set contains an unknown public category")
    if any(
        category in PROTECTED_DEFAULT
        and category not in scope.explicitly_authorized_protected_categories
        for category in scope.approved_categories
    ):
        errors.append("approved protected category lacks explicit authorization")
    if scope.allow_mask_access and "masks" not in scope.output_disclosure_fields:
        errors.append("mask access is not represented in the disclosure contract")
    return tuple(errors)


def require_metadata_scope(
    scope: AccessScopeManifest,
    *,
    category_id: str,
    section: str,
    fields: Sequence[str],
    protected: bool = False,
) -> None:
    """Guard a future metadata read; this function never performs the read itself."""
    errors = access_scope_errors(scope)
    if errors:
        raise R2L2BContractError(
            "local metadata access is unauthorized: " + "; ".join(errors))
    if category_id not in scope.approved_categories:
        raise R2L2BContractError("category is outside the explicit local access approval")
    if section not in scope.parsed_sections:
        raise R2L2BContractError(f"{section} section is not explicitly approved")
    allowed = (
        set(scope.allowed_train_metadata_fields) if section == "train"
        else set(scope.allowed_test_metadata_fields) if section == "test"
        else set()
    )
    if not set(fields).issubset(allowed):
        raise R2L2BContractError("requested metadata fields exceed the approved field allowlist")
    if set(fields).intersection(IDENTITY_METADATA_FIELDS) and not scope.allow_official_image_identity_metadata:
        raise R2L2BContractError("official image identity metadata is not approved")
    if (
        set(fields).intersection(CONDITION_BEARING_IDENTITY_FIELDS)
        and not scope.allow_official_sample_key_condition_tokens
    ):
        raise R2L2BContractError(
            "official sample-key OK/NG/defect tokens are not approved")
    if protected and category_id not in scope.explicitly_authorized_protected_categories:
        raise R2L2BContractError("protected category access has no explicit authorization")
    if section == "train" and set(fields).intersection(TRAIN_LABEL_FIELDS) and not scope.allow_train_normal_ok_status:
        raise R2L2BContractError("train normal/OK status is label-bearing and not approved")
    if section == "test" and set(fields).intersection(TEST_OUTCOME_FIELDS):
        if not scope.allow_outcome_label_access:
            raise R2L2BContractError("test outcome labels are not approved")
    if section == "test" and set(fields).intersection(MASK_FIELDS) and not scope.allow_mask_access:
        raise R2L2BContractError("test masks are not approved")


def source_group_digest_from_key(category_id: str, official_sample_key: str) -> str:
    """Domain-separate the established official_sample_key identity contract."""
    from experiments.risk_safe.preflight import (
        IDENTITY_VERSION,
        OFFICIAL_SAMPLE_KEY_RE,
        PRIMARY_GROUPING_UNIT,
    )

    if category_id not in CANONICAL_CATEGORY_IDS:
        raise R2L2BContractError("source identity category is outside the public catalog")
    match = OFFICIAL_SAMPLE_KEY_RE.fullmatch(f"{official_sample_key}_C1_")
    if match is None or match.group("sample_key") != official_sample_key:
        raise R2L2BContractError("sample key does not satisfy the official identity contract")
    if not official_sample_key.startswith(category_id + "_"):
        raise R2L2BContractError("sample key category does not match the supplied category")
    return canonical_sha256({
        "schema": "r2_l2b.source_group_identity.v1",
        "identity_contract_version": IDENTITY_VERSION,
        "grouping_unit": PRIMARY_GROUPING_UNIT,
        "category_id": category_id,
        "official_sample_key": official_sample_key,
    })


def source_group_digest_from_image_path(path_string: str, category_id: str) -> str:
    """Parse a caller-supplied identity string using existing logic; do not open it."""
    from experiments.risk_safe.preflight import parse_image_name

    identity = parse_image_name(path_string, expected_class=category_id)
    return source_group_digest_from_key(category_id, identity.official_sample_key)


@dataclass(frozen=True)
class SourceGroupMember:
    group_sha256: str
    image_view_count: int


@dataclass(frozen=True)
class RoleIdentityReceipt:
    schema: str
    category_id: str
    role: str
    grouping_unit: str
    identity_contract_version: str
    source_manifest_sha256: str
    producer_id: str
    group_members: tuple[SourceGroupMember, ...]
    image_count: int
    empty_role_policy_sha256: str | None
    empty_role_reason: str | None
    contains_raw_sample_keys: bool
    contains_image_paths: bool
    contains_labels_or_masks: bool
    receipt_sha256: str


def _identity_receipt_payload(receipt: RoleIdentityReceipt) -> dict[str, Any]:
    payload = asdict(receipt)
    payload.pop("receipt_sha256")
    return payload


def _identity_receipt_errors(receipt: RoleIdentityReceipt) -> tuple[str, ...]:
    errors: list[str] = []
    if receipt.schema != SCHEMA or receipt.category_id not in CANONICAL_CATEGORY_IDS:
        errors.append("unknown receipt schema or category")
    if receipt.role not in ROLE_NAMES:
        errors.append("unknown receipt role")
    if receipt.grouping_unit != GROUPING_UNIT or receipt.identity_contract_version != IDENTITY_CONTRACT_VERSION:
        errors.append("incompatible grouping or identity contract")
    try:
        _require_sha256(receipt.source_manifest_sha256, "source manifest SHA-256")
        groups = [member.group_sha256 for member in receipt.group_members]
        for group_digest in groups:
            _require_sha256(group_digest, "source-group SHA-256")
    except R2L2BContractError as exc:
        errors.append(str(exc))
        groups = []
    if len(groups) != len(set(groups)):
        errors.append("duplicate source group in receipt")
    if receipt.image_count != sum(member.image_view_count for member in receipt.group_members):
        errors.append("image count does not equal retained source-group view counts")
    if any(
        isinstance(member.image_view_count, bool) or member.image_view_count <= 0
        for member in receipt.group_members
    ):
        errors.append("source group has no image views")
    if not receipt.producer_id.strip():
        errors.append("producer identity is absent")
    if receipt.contains_raw_sample_keys or receipt.contains_image_paths or receipt.contains_labels_or_masks:
        errors.append("receipt exposes raw identities, paths, labels, or masks")
    if not receipt.group_members:
        if not receipt.empty_role_policy_sha256 or not (receipt.empty_role_reason or "").strip():
            errors.append("empty role has no explicit policy and reason")
        elif SHA256_RE.fullmatch(receipt.empty_role_policy_sha256) is None:
            errors.append("empty-role policy SHA-256 is malformed")
    elif receipt.empty_role_policy_sha256 is not None or receipt.empty_role_reason is not None:
        errors.append("nonempty role carries an empty-role policy")
    return tuple(errors)


def make_role_identity_receipt(
    *,
    category_id: str,
    role: str,
    source_manifest_sha256: str,
    producer_id: str,
    group_members: Sequence[SourceGroupMember],
    empty_role_policy_sha256: str | None = None,
    empty_role_reason: str | None = None,
) -> RoleIdentityReceipt:
    if category_id not in CANONICAL_CATEGORY_IDS or role not in ROLE_NAMES:
        raise R2L2BContractError("identity receipt category or role is outside contract")
    if not producer_id.strip():
        raise R2L2BContractError("identity receipt producer identity is required")
    _require_sha256(source_manifest_sha256, "source manifest SHA-256")
    members = tuple(sorted(group_members, key=lambda member: member.group_sha256))
    seen: set[str] = set()
    for member in members:
        _require_sha256(member.group_sha256, "source-group SHA-256")
        if member.group_sha256 in seen:
            raise R2L2BContractError("identity receipt duplicates a source group")
        if isinstance(member.image_view_count, bool) or member.image_view_count <= 0:
            raise R2L2BContractError("each source group must retain at least one image view")
        seen.add(member.group_sha256)
    if not members:
        if not empty_role_policy_sha256 or not (empty_role_reason or "").strip():
            raise R2L2BContractError(
                "an empty role requires an explicitly reviewed frozen N/A policy and reason")
        _require_sha256(empty_role_policy_sha256, "empty-role policy SHA-256")
    elif empty_role_policy_sha256 is not None or empty_role_reason is not None:
        raise R2L2BContractError("nonempty role cannot carry an empty-role rationale")
    partial = RoleIdentityReceipt(
        schema=SCHEMA,
        category_id=category_id,
        role=role,
        grouping_unit=GROUPING_UNIT,
        identity_contract_version=IDENTITY_CONTRACT_VERSION,
        source_manifest_sha256=source_manifest_sha256,
        producer_id=producer_id,
        group_members=members,
        image_count=sum(member.image_view_count for member in members),
        empty_role_policy_sha256=empty_role_policy_sha256,
        empty_role_reason=empty_role_reason,
        contains_raw_sample_keys=False,
        contains_image_paths=False,
        contains_labels_or_masks=False,
        receipt_sha256="",
    )
    return replace(partial, receipt_sha256=canonical_sha256(_identity_receipt_payload(partial)))


@dataclass(frozen=True)
class ReviewedIdentityManifest:
    category_id: str
    role: str
    grouping_unit: str
    identity_contract_version: str
    source_manifest_sha256: str
    group_members: tuple[SourceGroupMember, ...]
    image_count: int
    population: str
    normal_evidence_manifest_sha256: str | None
    empty_role_policy_sha256: str | None
    contains_raw_sample_keys: bool
    contains_image_paths: bool
    contains_labels_or_masks: bool
    manifest_sha256: str


def _reviewed_manifest_payload(manifest: ReviewedIdentityManifest) -> dict[str, Any]:
    payload = asdict(manifest)
    payload.pop("manifest_sha256")
    return payload


def make_reviewed_identity_manifest(
    receipt: RoleIdentityReceipt,
    *,
    population: str = "not_asserted",
    normal_evidence_manifest_sha256: str | None = None,
) -> ReviewedIdentityManifest:
    if receipt.role == "A" and population == "authorized_normal_reference_only":
        if not normal_evidence_manifest_sha256:
            raise R2L2BContractError("A normal-reference manifest needs normal-only evidence")
        _require_sha256(normal_evidence_manifest_sha256, "normal evidence manifest SHA-256")
    elif normal_evidence_manifest_sha256 is not None:
        raise R2L2BContractError("normal-only evidence is only valid for an A reference manifest")
    partial = ReviewedIdentityManifest(
        category_id=receipt.category_id,
        role=receipt.role,
        grouping_unit=receipt.grouping_unit,
        identity_contract_version=receipt.identity_contract_version,
        source_manifest_sha256=receipt.source_manifest_sha256,
        group_members=receipt.group_members,
        image_count=receipt.image_count,
        population=population,
        normal_evidence_manifest_sha256=normal_evidence_manifest_sha256,
        empty_role_policy_sha256=receipt.empty_role_policy_sha256,
        contains_raw_sample_keys=False,
        contains_image_paths=False,
        contains_labels_or_masks=False,
        manifest_sha256="",
    )
    return replace(
        partial,
        manifest_sha256=canonical_sha256(_reviewed_manifest_payload(partial)),
    )


@dataclass(frozen=True)
class IndependentReviewerAttestation:
    reviewer_id: str
    decision_id: str
    decision: str
    receipt_sha256: str
    reviewed_manifest_sha256: str
    signature_ed25519_hex: str
    attestation_sha256: str


def _attestation_payload(attestation: IndependentReviewerAttestation) -> dict[str, str]:
    payload = asdict(attestation)
    payload.pop("signature_ed25519_hex")
    payload.pop("attestation_sha256")
    return payload


def make_reviewer_attestation(
    receipt: RoleIdentityReceipt,
    manifest: ReviewedIdentityManifest,
    *,
    reviewer_id: str,
    decision_id: str,
    private_key: Any,
) -> IndependentReviewerAttestation:
    partial = IndependentReviewerAttestation(
        reviewer_id=reviewer_id,
        decision_id=decision_id,
        decision="CERTIFY_SOURCE_IDENTITY",
        receipt_sha256=receipt.receipt_sha256,
        reviewed_manifest_sha256=manifest.manifest_sha256,
        signature_ed25519_hex="",
        attestation_sha256="",
    )
    signature = _ed25519_sign(private_key, _attestation_payload(partial))
    signed_digest = canonical_sha256({
        "payload": _attestation_payload(partial),
        "signature_ed25519_hex": signature,
    })
    return replace(
        partial,
        signature_ed25519_hex=signature,
        attestation_sha256=signed_digest,
    )


def certify_role_identity(
    receipt: RoleIdentityReceipt,
    manifest: ReviewedIdentityManifest | None,
    attestation: IndependentReviewerAttestation | None,
    trust_directory: ReviewerTrustDirectory | None,
    *,
    expected_trust_directory_sha256: str | None,
) -> tuple[str, str]:
    """Verify producer evidence against a separate manifest and pinned trust root."""
    receipt_errors = _identity_receipt_errors(receipt)
    if receipt_errors:
        return "PENDING_INDEPENDENT_REVIEW", "; ".join(receipt_errors)
    if canonical_sha256(_identity_receipt_payload(receipt)) != receipt.receipt_sha256:
        return "PENDING_INDEPENDENT_REVIEW", "producer receipt digest is invalid"
    if manifest is None or attestation is None:
        return "PENDING_INDEPENDENT_REVIEW", "reviewer manifest or attestation is absent"
    if trust_directory is None or expected_trust_directory_sha256 is None:
        return "PENDING_INDEPENDENT_REVIEW", "external reviewer trust root is absent"
    if (
        reviewer_trust_directory_sha256(trust_directory) != trust_directory.directory_sha256
        or expected_trust_directory_sha256 != trust_directory.directory_sha256
    ):
        return "PENDING_INDEPENDENT_REVIEW", "reviewer trust root is not independently pinned"
    if attestation.reviewer_id == receipt.producer_id:
        return "PENDING_INDEPENDENT_REVIEW", "producer cannot independently certify its receipt"
    if attestation.reviewer_id not in trust_directory.trusted_reviewer_ids:
        return "PENDING_INDEPENDENT_REVIEW", "reviewer is absent from the trusted reviewer directory"
    if not attestation.reviewer_id.strip() or not attestation.decision_id.strip():
        return "PENDING_INDEPENDENT_REVIEW", "reviewer identity or decision identifier is absent"
    if receipt.source_manifest_sha256 not in trust_directory.trusted_identity_source_sha256s:
        return "PENDING_INDEPENDENT_REVIEW", "identity source is absent from the trusted directory"
    if (
        manifest.category_id != receipt.category_id
        or manifest.role != receipt.role
        or manifest.grouping_unit != receipt.grouping_unit
        or manifest.identity_contract_version != receipt.identity_contract_version
        or manifest.source_manifest_sha256 != receipt.source_manifest_sha256
        or manifest.group_members != receipt.group_members
        or manifest.image_count != receipt.image_count
        or manifest.empty_role_policy_sha256 != receipt.empty_role_policy_sha256
        or manifest.contains_raw_sample_keys
        or manifest.contains_image_paths
        or manifest.contains_labels_or_masks
        or canonical_sha256(_reviewed_manifest_payload(manifest)) != manifest.manifest_sha256
    ):
        return "PENDING_INDEPENDENT_REVIEW", "reviewer identity manifest does not match producer receipt"
    if (
        attestation.decision != "CERTIFY_SOURCE_IDENTITY"
        or attestation.receipt_sha256 != receipt.receipt_sha256
        or attestation.reviewed_manifest_sha256 != manifest.manifest_sha256
        or canonical_sha256({
            "payload": _attestation_payload(attestation),
            "signature_ed25519_hex": attestation.signature_ed25519_hex,
        }) != attestation.attestation_sha256
    ):
        return "PENDING_INDEPENDENT_REVIEW", "reviewer attestation is invalid or binds other evidence"
    reviewer_keys = dict(trust_directory.trusted_reviewer_public_keys)
    if not _ed25519_verify(
        reviewer_keys[attestation.reviewer_id],
        _attestation_payload(attestation),
        attestation.signature_ed25519_hex,
    ):
        return "PENDING_INDEPENDENT_REVIEW", "reviewer Ed25519 signature is invalid"
    if receipt.role == "A":
        if (
            manifest.population != "authorized_normal_reference_only"
            or not manifest.normal_evidence_manifest_sha256
        ):
            return "PENDING_INDEPENDENT_REVIEW", "A normal-only evidence is absent"
        try:
            _require_sha256(
                manifest.normal_evidence_manifest_sha256,
                "normal evidence manifest SHA-256",
            )
        except R2L2BContractError as exc:
            return "PENDING_INDEPENDENT_REVIEW", str(exc)
    if receipt.role == "A" and receipt.empty_role_policy_sha256 is not None:
        return "PENDING_INDEPENDENT_REVIEW", "A reference role cannot be certified empty"
    if (
        receipt.empty_role_policy_sha256 is not None
        and receipt.empty_role_policy_sha256
        not in trust_directory.reviewed_empty_role_policy_sha256s
    ):
        return "PENDING_INDEPENDENT_REVIEW", "empty role policy is not independently reviewed"
    return "CERTIFIED_FROM_SOURCE_EVIDENCE", "identity receipt matches independent review"


def detect_cross_role_overlaps(receipts: Sequence[RoleIdentityReceipt]) -> tuple[str, ...]:
    seen: dict[str, tuple[str, str]] = {}
    overlaps: set[str] = set()
    for receipt in receipts:
        for member in receipt.group_members:
            prior = seen.get(member.group_sha256)
            if prior is not None:
                if prior[0] != receipt.role or prior[1] != receipt.category_id:
                    overlaps.add(member.group_sha256)
                else:
                    raise R2L2BContractError(
                        "duplicate source-group identity receipt within a role/category")
            else:
                seen[member.group_sha256] = (receipt.role, receipt.category_id)
    return tuple(sorted(overlaps))


@dataclass(frozen=True)
class G1Regime:
    regime_id: str
    reference_population: str
    source_disjointness_requirement: str
    deployed_support_images: int
    additional_c64_calibration_normals: int
    c64_is_additional_to_four_supports: bool
    strict_four_shot_total: bool
    merges_into_c0_memory: bool
    radius_selected: bool


G1_FULL_A = G1Regime(
    regime_id="G1-FULL-A",
    reference_population=(
        "All independently authorized normal source groups assigned to A across the public catalog; "
        "broad reference-geometry regime."
    ),
    source_disjointness_requirement="A, B, and C source groups are pairwise disjoint.",
    deployed_support_images=4,
    additional_c64_calibration_normals=64,
    c64_is_additional_to_four_supports=True,
    strict_four_shot_total=False,
    merges_into_c0_memory=False,
    radius_selected=False,
)
G1_PROTOCOL_MATCHED = G1Regime(
    regime_id="G1-PROTOCOL-MATCHED",
    reference_population=(
        "Only evidence permitted by the frozen protocol: four deployed normal supports plus "
        "64 separately identified additional C64 calibration normals."
    ),
    source_disjointness_requirement="A, B, and C source groups are pairwise disjoint.",
    deployed_support_images=4,
    additional_c64_calibration_normals=64,
    c64_is_additional_to_four_supports=True,
    strict_four_shot_total=False,
    merges_into_c0_memory=False,
    radius_selected=False,
)


def g1_regimes_are_distinct() -> bool:
    return (
        G1_FULL_A.regime_id != G1_PROTOCOL_MATCHED.regime_id
        and G1_FULL_A.reference_population != G1_PROTOCOL_MATCHED.reference_population
        and not G1_FULL_A.merges_into_c0_memory
        and not G1_PROTOCOL_MATCHED.merges_into_c0_memory
        and not G1_FULL_A.strict_four_shot_total
        and not G1_PROTOCOL_MATCHED.strict_four_shot_total
    )


def certify_a_reference_availability(
    receipt: AReferenceAvailabilityReceipt | None,
    *,
    expected_category: str,
    independently_certified_a_identity: RoleIdentityReceipt | None,
    manifest: ReviewedIdentityManifest | None,
    attestation: IndependentReviewerAttestation | None,
    trust_directory: ReviewerTrustDirectory | None,
    expected_trust_directory_sha256: str | None,
) -> str:
    if receipt is None:
        return "PENDING_A_REFERENCE_RECEIPT"
    if receipt.category_id != expected_category or receipt.population != "authorized_normal_reference_only":
        return "PENDING_A_REFERENCE_RECEIPT"
    try:
        _require_sha256(receipt.receipt_sha256, "A reference receipt SHA-256")
    except R2L2BContractError:
        return "PENDING_A_REFERENCE_RECEIPT"
    if (
        receipt.authorized_normal_reference_image_count <= 0
        or receipt.authorized_source_group_count <= 0
        or not receipt.authorization_id.strip()
        or receipt.contains_image_paths
        or receipt.contains_labels_or_masks
    ):
        return "PENDING_A_REFERENCE_RECEIPT"
    if (
        trust_directory is None
        or expected_trust_directory_sha256 is None
        or reviewer_trust_directory_sha256(trust_directory) != trust_directory.directory_sha256
        or trust_directory.directory_sha256 != expected_trust_directory_sha256
        or receipt.receipt_sha256 not in trust_directory.reviewed_a_availability_receipt_sha256s
    ):
        return "PENDING_INDEPENDENT_A_REVIEW"
    if independently_certified_a_identity is None:
        return "PENDING_A_IDENTITY_CERTIFICATION"
    identity_status, _reason = certify_role_identity(
        independently_certified_a_identity,
        manifest,
        attestation,
        trust_directory,
        expected_trust_directory_sha256=expected_trust_directory_sha256,
    )
    if identity_status != "CERTIFIED_FROM_SOURCE_EVIDENCE":
        return "PENDING_A_IDENTITY_CERTIFICATION"
    if independently_certified_a_identity is None or manifest is None:
        return "PENDING_A_IDENTITY_CERTIFICATION"
    if (
        independently_certified_a_identity.category_id != expected_category
        or independently_certified_a_identity.role != "A"
        or len(independently_certified_a_identity.group_members) != receipt.authorized_source_group_count
        or independently_certified_a_identity.image_count
        != receipt.authorized_normal_reference_image_count
        or manifest.group_members != independently_certified_a_identity.group_members
        or manifest.population != "authorized_normal_reference_only"
        or not manifest.normal_evidence_manifest_sha256
    ):
        return "PENDING_A_NORMAL_IDENTITY_MATCH"
    return "AUTHORIZED_FOR_G1_REFERENCE_POPULATION"


def _role_identity_status(
    receipt: RoleIdentityReceipt | None,
    *,
    category_id: str,
    role: str,
    manifest: ReviewedIdentityManifest | None,
    attestation: IndependentReviewerAttestation | None,
    trust_directory: ReviewerTrustDirectory | None,
    expected_trust_directory_sha256: str | None,
    policy_status: str,
    scope: AccessScopeManifest,
) -> tuple[str, str | None]:
    if receipt is None:
        return "PENDING_MISSING_RECEIPT", None
    if receipt.category_id != category_id or receipt.role != role:
        return "PENDING_INDEPENDENT_REVIEW", "receipt category or role differs from its assigned slot"
    identity_status, reason = certify_role_identity(
        receipt, manifest, attestation, trust_directory,
        expected_trust_directory_sha256=expected_trust_directory_sha256,
    )
    if identity_status != "CERTIFIED_FROM_SOURCE_EVIDENCE":
        return identity_status, reason
    if policy_status != "APPROVED_POLICY_PENDING_LOCAL_ACCESS":
        return "PENDING_POLICY_APPROVAL", "role policy has not been approved for this category"
    if (
        scope.approval_state != "APPROVED"
        or access_scope_errors(scope)
        or category_id not in scope.approved_categories
        or (
            category_id in PROTECTED_DEFAULT
            and category_id not in scope.explicitly_authorized_protected_categories
        )
    ):
        return "PENDING_LOCAL_AUTHORIZATION", "local category access is not approved"
    if receipt.role == "C" and not scope.independent_c_identity_certificate_supplied:
        return "PENDING_INDEPENDENT_REVIEW", "separate C identity certificate was not supplied"
    return "CERTIFIED_FROM_SOURCE_EVIDENCE", None


def build_category_readiness(
    catalog: OfficialCatalogReceipt,
    proposal: RolePolicyProposal,
    scope: AccessScopeManifest,
    *,
    policy_approval: ResearcherApproval | None = None,
    trust_directory: ReviewerTrustDirectory | None = None,
    expected_trust_directory_sha256: str | None = None,
    role_receipts: Mapping[tuple[str, str], RoleIdentityReceipt] | None = None,
    reviewer_manifests: Mapping[tuple[str, str], ReviewedIdentityManifest] | None = None,
    reviewer_attestations: Mapping[tuple[str, str], IndependentReviewerAttestation] | None = None,
    a_reference_receipts: Mapping[str, AReferenceAvailabilityReceipt] | None = None,
) -> dict[str, Any]:
    categories = _valid_catalog(catalog)
    validate_role_policy_proposal(proposal)
    if proposal.catalog_source_sha256 != catalog.source_sha256:
        raise R2L2BContractError("proposal is not bound to the supplied public catalog")
    receipts = role_receipts or {}
    manifests = reviewer_manifests or {}
    attestations = reviewer_attestations or {}
    a_receipts = a_reference_receipts or {}
    policy_status = evaluate_policy_approval(
        proposal, policy_approval, trust_directory,
        expected_trust_directory_sha256=expected_trust_directory_sha256,
    )
    overlaps = detect_cross_role_overlaps(tuple(receipts.values()))
    if overlaps:
        raise R2L2BContractError(
            f"A/B/C source-group overlap detected for {len(overlaps)} source group(s)")

    rows: list[dict[str, Any]] = []
    for category in categories:
        role_states: dict[str, str] = {}
        for role in ROLE_NAMES:
            key = (category, role)
            state, _reason = _role_identity_status(
                receipts.get(key),
                category_id=category,
                role=role,
                manifest=manifests.get(key),
                attestation=attestations.get(key),
                trust_directory=trust_directory,
                expected_trust_directory_sha256=expected_trust_directory_sha256,
                policy_status=policy_status,
                scope=scope,
            )
            role_states[role] = state
        a_key = (category, "A")
        a_status = certify_a_reference_availability(
            a_receipts.get(category),
            expected_category=category,
            independently_certified_a_identity=receipts.get(a_key)
            if role_states["A"] == "CERTIFIED_FROM_SOURCE_EVIDENCE" else None,
            manifest=manifests.get(a_key),
            attestation=attestations.get(a_key),
            trust_directory=trust_directory,
            expected_trust_directory_sha256=expected_trust_directory_sha256,
        )
        protected = category in proposal.protected_categories
        rows.append({
            "category_id": category,
            "public_catalog_status": "VERIFIED_PUBLIC_CATEGORY_ONLY",
            "local_data_inventory_status": "PENDING_AUTHORIZATION",
            "protected_status": "PROTECTED_SEALED" if protected else "PENDING_RESEARCHER_REVIEW",
            "role_policy_status": policy_status,
            "role_identity_status": role_states,
            "a_reference_status": a_status,
            "source_group_counts": None,
        })
    all_a_ready = all(
        row["a_reference_status"] == "AUTHORIZED_FOR_G1_REFERENCE_POPULATION"
        for row in rows
    )
    all_roles_ready = all(
        all(state == "CERTIFIED_FROM_SOURCE_EVIDENCE" for state in row["role_identity_status"].values())
        for row in rows
    )
    scope_ready = not access_scope_errors(scope)
    role_certified = (
        policy_status == "APPROVED_POLICY_PENDING_LOCAL_ACCESS"
        and scope_ready and all_roles_ready
    )
    full_a_ready = role_certified and all_a_ready
    missing = [
        "researcher must choose and approve H-SOURCE or freeze an exact H-CLASS category set",
        "researcher must approve the complete versioned category-by-role policy",
        "explicit local root, category, physical JSON byte, parsed field, and output disclosure scope",
        "explicit acknowledgment that train and test sections may share physical JSON bytes",
        "normal/OK status access is label-bearing and needs a separate explicit decision",
        "Mint/PCB and any other protected categories need category-specific written authorization",
        "authorized local metadata inventory and independently reviewed A/B/C source-group identity manifests",
        "independent reviewer trust directory pinned outside producer-controlled receipts",
        "pairwise source-group disjointness certification, including independently reviewed C identities",
        "normal-only A reference availability receipts and independent source evidence",
        "separate decision on DINO/feature extraction, output location, byte limit, and disclosed fields",
        "separate G1 regime decision; neither regime merges into C0 memory and no radius is selected",
    ]
    if full_a_ready:
        missing = []
    elif role_certified:
        missing = [
            item for item in missing
            if "normal-only A reference availability receipts" in item
            or "G1 regime decision" in item
        ]
    return {
        "schema": "r2_l2b.category_readiness.v1",
        "status": "R2_L2B_PROTOCOL_READY_FOR_RESEARCHER_APPROVAL",
        "role_certified": bool(role_certified),
        "g1_full_a_ready": bool(full_a_ready),
        "g1_protocol_matched_ready": False,
        "role_certification_status": (
            "R2_L2B_ROLE_CERTIFIED" if role_certified else "PENDING"
        ),
        "policy_status": policy_status,
        "public_inventory_status": "R2_L2A_R_PUBLIC_INVENTORY_PASS",
        "local_data_access_status": "PENDING_AUTHORIZATION",
        "actual_distance_statistics": "NOT_RUN",
        "g1_status": "R2_L2B_G1_NOT_AUTHORIZED",
        "category_rows": rows,
        "missing_prerequisites": missing,
        "h_source_proposal_sha256": proposal.proposal_sha256,
        "g1_regimes": [asdict(G1_FULL_A), asdict(G1_PROTOCOL_MATCHED)],
    }
