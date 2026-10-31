"""R2-L2D provenance audit and identity-only role contracts.

The artifact auditor reads only the two existing R2-L2C output files. The
proposal, role-receipt, and Support/C64 binding helpers operate on caller
provided identities and never discover or open dataset files.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

from experiments.calibration.r2_l2a_allclass_inventory import (
    CANONICAL_CATEGORY_IDS,
    GROUPING_UNIT,
    IDENTITY_CONTRACT_VERSION,
    PINNED_SOURCE_SHA256,
    PINNED_SOURCE_URI,
    ROLE_NAMES,
)
from experiments.calibration.r2_l2a_g1_geometry import (
    R1_FIXED_SUPPORT_CATEGORY_ID,
    R1_FIXED_SUPPORT_DIGEST_ALGORITHM,
    R1_FIXED_SUPPORT_SEED,
    R1_FIXED_SUPPORT_SHA256,
    category_fixed_support_receipt_sha256,
)
from experiments.calibration.r2_l2b_role_certification import (
    IndependentReviewerAttestation,
    ResearcherApproval,
    ReviewedIdentityManifest,
    ReviewerTrustDirectory,
    RolePolicyProposal,
    RoleIdentityReceipt,
    SOURCE_GROUP_DIGEST_ALGORITHM,
    STRATEGY_H_SOURCE,
    canonical_sha256,
    certify_role_identity,
    evaluate_policy_approval,
    reviewer_trust_directory_sha256,
)
from experiments.calibration.r2_l2c_metadata_census import (
    FORBIDDEN_PRIVATE_FIELDS,
    FORBIDDEN_PUBLIC_FIELDS,
    REQUIRED_PUBLIC_OUTPUT_FIELDS,
    SupportC64SelectionAttestation,
    SupportC64SelectionReceipt,
    _source_sha256,
    _validate_private_receipt_fields,
    _canonical_bytes,
    _verify_ed25519,
    assess_support_c64_selection,
    support_c64_pool_status,
)
from experiments.calibration import r2_l2c_metadata_census as l2c_metadata_census
from experiments.risk_safe import preflight as identity_contract


SCHEMA = "r2_l2d.data_role_adjudication.v1"
PARTITION_RANKING_ALGORITHM = "sha256:r2-l2d-hsource-group-rank.v1"
SUPPORT_BANK_BINDING_SCHEMA = "r2_l2d.support_c64_fixed_bank_binding.v1"
SUPPORT_SELECTION_AUTHORIZATION_SCHEMA = "r2_l2d.support_c64_selection_authorization.v1"
EXPECTED_CENSUS_FILES = frozenset({
    "census_report.json",
    "private_source_group_receipts.json",
})
MAX_CENSUS_OUTPUT_BYTES = 64 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class R2L2DContractError(ValueError):
    """Malformed provenance or identity-only contract input."""


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise R2L2DContractError(f"{name} must be lowercase SHA-256 hex")
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _keys_recursive(value: Any) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for nested in value.values():
            result.update(_keys_recursive(nested))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for nested in value:
            result.update(_keys_recursive(nested))
        return result
    return set()


def _load_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise R2L2DContractError(f"required census artifact cannot be parsed: {path.name}") from exc
    if not isinstance(value, dict):
        raise R2L2DContractError(f"required census artifact is not a JSON object: {path.name}")
    return value


def _is_reparse_point(path: Path) -> bool:
    attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _reject_reparse_chain(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.exists() and _is_reparse_point(candidate):
            raise R2L2DContractError("census output path contains a reparse point")


def audit_l2c_artifact_pair(
    output_directory: str | Path,
    *,
    maximum_output_bytes: int = MAX_CENSUS_OUTPUT_BYTES,
    expected_normal_image_count: int = 36_465,
    expected_source_group_count: int = 7_293,
    minimum_groups_per_category: int = 68,
) -> dict[str, Any]:
    """Check the existing L2C pair without opening source metadata.

    Private group identifiers and per-source fingerprints are processed only
    in memory. The returned object contains aggregate counts and the public
    completion-report digest, never a private-receipt digest or group ID.
    """
    for value, name in (
        (maximum_output_bytes, "maximum output bytes"),
        (expected_normal_image_count, "expected normal image count"),
        (expected_source_group_count, "expected source-group count"),
        (minimum_groups_per_category, "minimum groups per category"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise R2L2DContractError(f"{name} must be a nonnegative integer")
    if maximum_output_bytes == 0 or minimum_groups_per_category == 0:
        raise R2L2DContractError("output limit and minimum group count must be positive")
    root = Path(output_directory)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise R2L2DContractError("census output path must be an existing absolute regular directory")
    _reject_reparse_chain(root)
    resolved_root = root.resolve(strict=True)
    entries = tuple(root.iterdir())
    if any(item.is_symlink() or _is_reparse_point(item) for item in entries):
        raise R2L2DContractError("census output contains a symlink or reparse point")
    if {item.name for item in entries} != EXPECTED_CENSUS_FILES:
        raise R2L2DContractError("census output file set differs from the two-file L2C contract")
    for item in entries:
        if not item.is_file() or item.resolve(strict=True).parent != resolved_root:
            raise R2L2DContractError("census output contains a non-regular or escaping file")
    actual_bytes = sum(item.stat().st_size for item in entries)
    if actual_bytes > maximum_output_bytes:
        raise R2L2DContractError("census output exceeds the approved byte limit")

    public_path = root / "census_report.json"
    private_path = root / "private_source_group_receipts.json"
    public = _load_object(public_path)
    private = _load_object(private_path)
    if public.get("schema") != "r2_l2c.metadata_census_report.v1":
        raise R2L2DContractError("public census report schema is incompatible")
    if private.get("schema") != "r2_l2c.private_group_receipts.v1":
        raise R2L2DContractError("private source-group receipt schema is incompatible")
    if set(public) - REQUIRED_PUBLIC_OUTPUT_FIELDS:
        raise R2L2DContractError("public census report contains an unapproved field")
    try:
        _validate_private_receipt_fields(private)
    except (KeyError, TypeError, ValueError) as exc:
        raise R2L2DContractError("private source-group receipt fields are incompatible") from exc
    if _keys_recursive(public).intersection(FORBIDDEN_PUBLIC_FIELDS):
        raise R2L2DContractError("public census report contains a protected field")
    if _keys_recursive(private).intersection(FORBIDDEN_PRIVATE_FIELDS):
        raise R2L2DContractError("private receipt contains a protected raw field")
    if public.get("status") != "R2_L2C_METADATA_CENSUS_COMPLETE_PARTITION_PENDING":
        raise R2L2DContractError("census completion marker is absent or contradictory")
    if public.get("operation") != "R2_L2C_AUTHORIZED_TRAIN_METADATA_CENSUS_V1":
        raise R2L2DContractError("census operation identity is incompatible")
    if public.get("local_data_io_performed") is not True:
        raise R2L2DContractError("completion report does not attest to its declared metadata census operation")
    if (public.get("authorization_id") != private.get("authorization_id")
            or public.get("authorization_sha256") != private.get("authorization_sha256")):
        raise R2L2DContractError("public/private authorization identity mismatch")
    _sha256(public.get("authorization_sha256"), "authorization SHA-256")
    if private.get("source_group_digest_algorithm") != SOURCE_GROUP_DIGEST_ALGORITHM:
        raise R2L2DContractError("private group identity algorithm differs from the frozen contract")
    if public.get("source_group_digest_algorithm") != SOURCE_GROUP_DIGEST_ALGORITHM:
        raise R2L2DContractError("public group identity algorithm differs from the frozen contract")
    if public.get("identity_contract_version") != IDENTITY_CONTRACT_VERSION:
        raise R2L2DContractError("legacy official-sample identity version is incompatible")
    if public.get("grouping_unit") != GROUPING_UNIT:
        raise R2L2DContractError("scientific independence unit is not official_sample_key")
    if (public.get("public_catalog_source_sha256") != PINNED_SOURCE_SHA256
            or public.get("public_catalog_source_uri") != PINNED_SOURCE_URI):
        raise R2L2DContractError("public category catalog is not the pinned source receipt")
    if public.get("target_category_count") != len(CANONICAL_CATEGORY_IDS):
        raise R2L2DContractError("public catalog category count is incompatible")
    if public.get("partition_status") != "A_CANDIDATES_IDENTIFIED_PARTITION_NOT_CERTIFIED":
        raise R2L2DContractError("census partition status is missing or contradictory")
    if (public.get("test_section_parsed") is not False
            or public.get("image_decoding_performed") is not False
            or public.get("feature_extraction_performed") is not False
            or public.get("geometry_performed") is not False):
        raise R2L2DContractError("census report records an out-of-scope operation")

    public_rows = public.get("categories")
    private_rows = private.get("receipts")
    if not isinstance(public_rows, list) or not isinstance(private_rows, list):
        raise R2L2DContractError("category rows are absent")
    expected_categories = tuple(CANONICAL_CATEGORY_IDS)
    if (tuple(row.get("category_id") for row in public_rows if isinstance(row, dict))
            != expected_categories
            or tuple(row.get("category_id") for row in private_rows if isinstance(row, dict))
            != expected_categories):
        raise R2L2DContractError("category coverage is incomplete, duplicated, or out of order")
    if len(public_rows) != 30 or len(private_rows) != 30:
        raise R2L2DContractError("census must contain exactly 30 public and private category rows")

    total_images = 0
    total_groups = 0
    all_groups: set[str] = set()
    role_slots_pending = 0
    category_minimum_group_count: int | None = None
    categories_with_68_groups = 0
    census_parser_sha = _source_sha256(Path(l2c_metadata_census.__file__))
    identity_parser_sha = _source_sha256(Path(identity_contract.__file__))
    for public_row, private_row in zip(public_rows, private_rows, strict=True):
        if not isinstance(public_row, dict) or not isinstance(private_row, dict):
            raise R2L2DContractError("category receipt is not an object")
        if set(public_row) - REQUIRED_PUBLIC_OUTPUT_FIELDS:
            raise R2L2DContractError("public category row contains an unapproved field")
        category = public_row["category_id"]
        source_sha = _sha256(public_row.get("source_json_sha256"), "reported source JSON SHA-256")
        if source_sha != _sha256(private_row.get("source_json_sha256"), "private source JSON SHA-256"):
            raise R2L2DContractError("public/private source fingerprint mismatch")
        if private_row.get("source_group_digest_algorithm") != SOURCE_GROUP_DIGEST_ALGORITHM:
            raise R2L2DContractError("category source-group digest algorithm mismatch")
        if (public_row.get("parser_source_sha256") != census_parser_sha
                or public_row.get("identity_parser_source_sha256") != identity_parser_sha):
            raise R2L2DContractError("recorded census parser fingerprints do not match source files")
        if (private_row.get("contains_raw_sample_keys") is not False
                or private_row.get("contains_image_paths") is not False
                or private_row.get("contains_labels_or_masks") is not False):
            raise R2L2DContractError("private receipt exposes a prohibited identity or outcome field")
        members = private_row.get("group_members")
        if not isinstance(members, list):
            raise R2L2DContractError("private group-members list is absent")
        category_groups: set[str] = set()
        view_counts: dict[str, int] = {}
        for member in members:
            if not isinstance(member, dict) or set(member) != {"source_group_sha256", "camera_view_count"}:
                raise R2L2DContractError("private group member schema is incompatible")
            group = _sha256(member.get("source_group_sha256"), "private source-group identity")
            views = member.get("camera_view_count")
            if isinstance(views, bool) or not isinstance(views, int) or views <= 0:
                raise R2L2DContractError("source group has an invalid camera-view count")
            if views != 5:
                raise R2L2DContractError("source group does not retain all five camera views")
            if group in category_groups or group in all_groups:
                raise R2L2DContractError("duplicate or ambiguous source-group identity")
            category_groups.add(group)
            all_groups.add(group)
            view_counts[str(views)] = view_counts.get(str(views), 0) + 1
        expected_group_count = public_row.get("distinct_normal_source_group_count")
        expected_image_count = public_row.get("eligible_normal_image_count")
        if (isinstance(expected_group_count, bool) or not isinstance(expected_group_count, int)
                or expected_group_count != len(category_groups)):
            raise R2L2DContractError("reported category group count does not match private identities")
        if public_row.get("at_least_68_legal_normal_groups") is not (expected_group_count >= 68):
            raise R2L2DContractError("reported 68-group capacity flag contradicts the census identities")
        if (isinstance(expected_image_count, bool) or not isinstance(expected_image_count, int)
                or expected_image_count != sum(member["camera_view_count"] for member in members)):
            raise R2L2DContractError("reported category image count does not match private view counts")
        if public_row.get("normal_camera_view_distribution") != view_counts:
            raise R2L2DContractError("camera-view aggregate does not match private group receipts")
        if (public_row.get("metadata_availability_status") != "METADATA_AVAILABLE"
                or public_row.get("a_role_candidate_status") != "A_CANDIDATES_IDENTIFIED"
                or public_row.get("abc_partition_status")
                != "A_CANDIDATES_IDENTIFIED_PARTITION_NOT_CERTIFIED"):
            raise R2L2DContractError("category availability or role-candidate status is contradictory")
        for count_field in (
            "malformed_identity_count", "duplicate_image_identity_count",
            "ambiguous_group_identity_count",
        ):
            if public_row.get(count_field) != 0:
                raise R2L2DContractError("malformed, duplicate, or ambiguous identity count is nonzero")
        serial_status = public_row.get("directory_serial_overlap_status")
        serial_count = public_row.get("directory_serial_overlap_count")
        if serial_count != private_row.get("directory_serial_overlap_count"):
            raise R2L2DContractError("public/private directory-serial sensitivity count mismatch")
        if serial_status in {"NOT_AUTHORIZED", "NOT_CHECKED_NO_LOCAL_IO"}:
            if serial_count is not None:
                raise R2L2DContractError("unauthorized directory-serial sensitivity must remain unmeasured")
        elif (serial_status == "DIRECTORY_SERIAL_OVERLAP_SENSITIVITY_REPORTED"
              and (isinstance(serial_count, bool) or not isinstance(serial_count, int) or serial_count < 0)):
            raise R2L2DContractError("directory-serial sensitivity count is malformed")
        elif serial_status != "DIRECTORY_SERIAL_OVERLAP_SENSITIVITY_REPORTED":
            raise R2L2DContractError("directory-serial sensitivity status is unrecognized")
        if public_row.get("support_and_c64_identity_status") in {
            "SUPPORT_AND_C64_IDENTITIES_CERTIFIED", "SUPPORT_C64_IDENTITIES_CERTIFIED",
        }:
            raise R2L2DContractError("census claims a Support/C64 selection that was not authorized here")
        total_images += expected_image_count
        total_groups += expected_group_count
        category_minimum_group_count = (
            expected_group_count if category_minimum_group_count is None
            else min(category_minimum_group_count, expected_group_count)
        )
        categories_with_68_groups += expected_group_count >= 68
        role = private_row.get("role_certification_status")
        if not isinstance(role, dict) or set(role.get("role_status", {})) != set(ROLE_NAMES):
            raise R2L2DContractError("private A/B/C role status is incomplete")
        if role.get("partition_status") != "A_CANDIDATES_IDENTIFIED_PARTITION_NOT_CERTIFIED":
            raise R2L2DContractError("private role partition status is contradictory")
        if role.get("cross_role_overlap_count") != 0:
            raise R2L2DContractError("reported role overlap count is nonzero")
        if any(value in {"CERTIFIED_FROM_SOURCE_EVIDENCE", "SOURCE_GROUP_PARTITION_CERTIFIED"}
               for value in role["role_status"].values()):
            raise R2L2DContractError("uncertified census contains a purported certified role")
        role_slots_pending += sum(value != "CERTIFIED_FROM_SOURCE_EVIDENCE"
                                  for value in role["role_status"].values())
    if (public.get("metadata_files_opened") != 30
            or public.get("metadata_available_category_count") != 30
            or public.get("authorized_category_count") != 30
            or public.get("missing_or_unavailable_categories") != []):
        raise R2L2DContractError("reported metadata coverage is not exactly 30/30")
    if total_images != expected_normal_image_count:
        raise R2L2DContractError("eligible normal image aggregate differs from the expected census total")
    if total_groups != expected_source_group_count:
        raise R2L2DContractError("source-group aggregate differs from the expected census total")
    if category_minimum_group_count is None or category_minimum_group_count < minimum_groups_per_category:
        raise R2L2DContractError("a category is below the minimum authorized candidate-pool size")

    return {
        "schema": SCHEMA,
        "artifact_audit_status": "ARTIFACT_AUDIT_PASS",
        "artifact_pair_status": "INTERNALLY_CONSISTENT_REPORTED",
        "provenance_status": "BLOCKED_INPUT",
        "blockers": [
            "signed authorization payload and pinned trust-root fingerprint are absent from the supplied census artifacts",
            "no independent prior 30-category source-fingerprint baseline is available; raw source JSON was not reopened",
        ],
        "public_completion_report_sha256": hashlib.sha256(public_path.read_bytes()).hexdigest(),
        "private_receipt_file_sha256_emitted": False,
        "standalone_run_manifest_present": False,
        "source_fingerprint_crossfile_matches": len(public_rows),
        "source_fingerprint_baseline_status": "BLOCKED_INPUT",
        "output_bytes": actual_bytes,
        "output_byte_limit": maximum_output_bytes,
        "category_count": len(public_rows),
        "eligible_normal_image_count": total_images,
        "source_group_count": total_groups,
        "minimum_groups_in_any_category": category_minimum_group_count,
        "categories_with_at_least_68_groups": categories_with_68_groups,
        "role_slots_uncertified": role_slots_pending,
        "directory_serial_overlap_sensitivity_status": (
            "NOT_AUTHORIZED_IN_CENSUS"
            if all(row.get("directory_serial_overlap_status") == "NOT_AUTHORIZED" for row in public_rows)
            else "REPORTED_BY_CENSUS"
        ),
        "authorization_id": public["authorization_id"],
        "authorization_sha256": public["authorization_sha256"],
        "test_section_bytes_encountered": public.get("test_section_bytes_encountered"),
        "test_section_parsed": False,
        "feature_extraction_performed": False,
        "geometry_performed": False,
    }


@dataclass(frozen=True)
class RolePartitionProposal:
    """Deterministic H-SOURCE proposal; membership is never certified here."""

    schema: str
    category_id: str
    policy_sha256: str
    seed: int
    source_population: str
    quotas: tuple[tuple[str, int], ...]
    group_assignments: tuple[tuple[str, str], ...]
    ranking_algorithm: str
    status: str
    proposal_sha256: str


def propose_hsource_partition(
    category_id: str,
    source_group_sha256s: Sequence[str],
    *,
    policy_sha256: str,
    seed: int,
    quotas: Mapping[str, int],
    source_population: str = "TRAIN_NORMAL_CANDIDATES",
    test_metadata_authorized: bool = False,
) -> RolePartitionProposal:
    """Propose a repeatable group-level partition for synthetic/planned use.

    This function does not perform dataset access and marks every result
    PROPOSED. A policy hash or deterministic output is not authorization.
    """
    if category_id not in CANONICAL_CATEGORY_IDS:
        raise R2L2DContractError("category is outside the frozen public catalog")
    _sha256(policy_sha256, "role policy SHA-256")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise R2L2DContractError("partition seed must be a nonnegative integer")
    if source_population != "TRAIN_NORMAL_CANDIDATES" or test_metadata_authorized:
        raise R2L2DContractError("Test metadata is outside this identity-only partition contract")
    if set(quotas) != set(ROLE_NAMES):
        raise R2L2DContractError("A/B/C quotas must be explicitly supplied")
    normalized_quotas: list[tuple[str, int]] = []
    for role in ROLE_NAMES:
        count = quotas[role]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise R2L2DContractError("A/B/C proposal quotas must each be positive integers")
        normalized_quotas.append((role, count))
    groups = tuple(source_group_sha256s)
    if any(_SHA256_RE.fullmatch(group) is None for group in groups):
        raise R2L2DContractError("candidate pool contains a malformed source-group identity")
    if len(set(groups)) != len(groups):
        raise R2L2DContractError("candidate pool contains duplicate source-group identities")
    requested_count = sum(count for _, count in normalized_quotas)
    if requested_count > len(groups):
        raise R2L2DContractError("category-level candidate shortage for the proposed A/B/C quotas")
    if requested_count < len(groups):
        raise R2L2DContractError("A/B/C quotas do not cover the candidate pool exactly")
    ranked = sorted(
        groups,
        key=lambda group: (
            hashlib.sha256(
                f"{PARTITION_RANKING_ALGORITHM}\0{policy_sha256}\0{category_id}\0{seed}\0{group}".encode("ascii")
            ).hexdigest(),
            group,
        ),
    )
    assignments: list[tuple[str, str]] = []
    offset = 0
    for role, count in normalized_quotas:
        assignments.extend((group, role) for group in ranked[offset:offset + count])
        offset += count
    assignments_tuple = tuple(sorted(assignments))
    body = {
        "schema": "r2_l2d.h_source_partition_proposal.v1",
        "category_id": category_id,
        "policy_sha256": policy_sha256,
        "seed": seed,
        "source_population": source_population,
        "quotas": normalized_quotas,
        "group_assignments": assignments_tuple,
        "ranking_algorithm": PARTITION_RANKING_ALGORITHM,
        "status": "PROPOSED",
    }
    return RolePartitionProposal(
        **body,
        proposal_sha256=hashlib.sha256(_json_bytes(body)).hexdigest(),
    )


@dataclass(frozen=True)
class PairwisePartitionAudit:
    status: str
    a_intersection_b: int
    a_intersection_c: int
    b_intersection_c: int
    unknown_group_count: int
    duplicate_within_role_count: int


def audit_pairwise_partition(
    assignments: Mapping[str, Sequence[str]],
    *,
    authorized_candidate_groups: Sequence[str] | None = None,
) -> PairwisePartitionAudit:
    """Check group-level membership and return counts only, never identities."""
    if set(assignments) != set(ROLE_NAMES):
        raise R2L2DContractError("partition must contain exactly A, B, and C roles")
    role_sets: dict[str, set[str]] = {}
    duplicate_count = 0
    for role in ROLE_NAMES:
        values = tuple(assignments[role])
        if any(not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None for value in values):
            raise R2L2DContractError("partition contains malformed source-group identity")
        duplicate_count += len(values) - len(set(values))
        role_sets[role] = set(values)
    if duplicate_count:
        return PairwisePartitionAudit("DUPLICATE_SOURCE_GROUP_REJECTED", 0, 0, 0, 0, duplicate_count)
    overlaps = (
        len(role_sets["A"] & role_sets["B"]),
        len(role_sets["A"] & role_sets["C"]),
        len(role_sets["B"] & role_sets["C"]),
    )
    unknown = 0
    if authorized_candidate_groups is not None:
        candidates = tuple(authorized_candidate_groups)
        if any(not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None for value in candidates):
            raise R2L2DContractError("authorized candidate pool contains malformed identity")
        if len(set(candidates)) != len(candidates):
            raise R2L2DContractError("authorized candidate pool contains duplicate identity")
        candidate_set = set(candidates)
        unknown = len(set.union(*role_sets.values()) - candidate_set)
        if unknown:
            return PairwisePartitionAudit("UNKNOWN_SOURCE_GROUP_REJECTED", *overlaps, unknown, 0)
    status = "SOURCE_GROUP_OVERLAP_REJECTED" if any(overlaps) else "DISJOINTNESS_ONLY_NOT_CERTIFIED"
    return PairwisePartitionAudit(status, *overlaps, unknown, 0)


@dataclass(frozen=True)
class RoleEvidence:
    receipt: RoleIdentityReceipt
    manifest: ReviewedIdentityManifest
    attestation: IndependentReviewerAttestation


def adjudicate_category_roles(
    category_id: str,
    evidence_by_role: Mapping[str, RoleEvidence | None],
    trust_directory: ReviewerTrustDirectory | None,
    *,
    expected_trust_directory_sha256: str | None,
    expected_source_manifest_sha256: str | None = None,
    authorized_candidate_group_sha256s: Sequence[str] | None = None,
    role_policy_proposal: RolePolicyProposal | None = None,
    researcher_approval: ResearcherApproval | None = None,
) -> dict[str, Any]:
    """Verify A/B/C role receipts with the existing L2B pinned-trust checks."""
    if category_id not in CANONICAL_CATEGORY_IDS:
        raise R2L2DContractError("category is outside the frozen public catalog")
    if set(evidence_by_role) - set(ROLE_NAMES):
        raise R2L2DContractError("role evidence contains an unknown role")
    if role_policy_proposal is None:
        policy_status = "PROPOSED_PENDING_RESEARCHER_APPROVAL"
    elif (not isinstance(role_policy_proposal, RolePolicyProposal)
          or role_policy_proposal.strategy != STRATEGY_H_SOURCE
          or role_policy_proposal.identity_contract_version != IDENTITY_CONTRACT_VERSION
          or role_policy_proposal.grouping_unit != GROUPING_UNIT):
        policy_status = "PENDING_INVALID_H_SOURCE_POLICY"
    else:
        try:
            policy_status = evaluate_policy_approval(
                role_policy_proposal,
                researcher_approval,
                trust_directory,
                expected_trust_directory_sha256=expected_trust_directory_sha256,
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            policy_status = "PENDING_INVALID_H_SOURCE_POLICY"
    statuses: dict[str, str] = {}
    groups: dict[str, set[str]] = {}
    reasons: dict[str, str] = {}
    for role in ROLE_NAMES:
        evidence = evidence_by_role.get(role)
        if evidence is None:
            statuses[role] = "PENDING_ROLE_RECEIPT"
            reasons[role] = "role receipt, independent review, or authorization evidence is absent"
            continue
        if evidence.receipt.category_id != category_id or evidence.receipt.role != role:
            statuses[role] = "PENDING_ROLE_RECEIPT"
            reasons[role] = "receipt category or role does not match its assigned slot"
            continue
        if (expected_source_manifest_sha256 is not None
                and evidence.receipt.source_manifest_sha256 != expected_source_manifest_sha256):
            statuses[role] = "PENDING_SOURCE_MANIFEST_MISMATCH"
            reasons[role] = "role receipt does not bind the authorized category source manifest"
            continue
        status, reason = certify_role_identity(
            evidence.receipt, evidence.manifest, evidence.attestation, trust_directory,
            expected_trust_directory_sha256=expected_trust_directory_sha256,
        )
        statuses[role] = status
        reasons[role] = reason
        if status == "CERTIFIED_FROM_SOURCE_EVIDENCE":
            groups[role] = {member.group_sha256 for member in evidence.receipt.group_members}
    if all(statuses[role] == "CERTIFIED_FROM_SOURCE_EVIDENCE" for role in ROLE_NAMES):
        pairwise = audit_pairwise_partition(groups)
        policy_principal = researcher_approval.researcher_id if researcher_approval is not None else None
        identity_principals = {
            principal
            for role in ROLE_NAMES
            for principal in (
                evidence_by_role[role].receipt.producer_id,
                evidence_by_role[role].attestation.reviewer_id,
            )
        }
        if policy_status != "RESEARCHER_POLICY_APPROVED":
            partition_status = "PENDING_H_SOURCE_POLICY_APPROVAL"
        elif policy_principal in identity_principals:
            partition_status = "PENDING_INDEPENDENT_POLICY_APPROVER"
        elif expected_source_manifest_sha256 is None:
            partition_status = "PENDING_AUTHORIZED_SOURCE_MANIFEST"
        elif authorized_candidate_group_sha256s is None:
            partition_status = "PENDING_AUTHORIZED_SOURCE_GROUP_UNIVERSE"
        else:
            try:
                _sha256(expected_source_manifest_sha256, "authorized source manifest SHA-256")
                candidate_groups = tuple(authorized_candidate_group_sha256s)
                if any(not isinstance(group, str) or _SHA256_RE.fullmatch(group) is None
                       for group in candidate_groups):
                    raise R2L2DContractError("authorized group universe contains malformed identities")
                if len(set(candidate_groups)) != len(candidate_groups):
                    raise R2L2DContractError("authorized group universe contains duplicate identities")
                candidate_set = set(candidate_groups)
                assigned_set = set.union(*groups.values())
                if assigned_set - candidate_set:
                    partition_status = "UNKNOWN_SOURCE_GROUP_REJECTED"
                elif candidate_set - assigned_set:
                    partition_status = "PARTITION_INCOMPLETE_REJECTED"
                elif pairwise.status == "DISJOINTNESS_ONLY_NOT_CERTIFIED":
                    partition_status = "SOURCE_GROUP_PARTITION_CERTIFIED"
                else:
                    partition_status = pairwise.status
            except R2L2DContractError:
                partition_status = "PENDING_AUTHORIZED_SOURCE_GROUP_UNIVERSE"
    else:
        pairwise = PairwisePartitionAudit("PARTITION_PENDING", 0, 0, 0, 0, 0)
        partition_status = "A_CANDIDATES_IDENTIFIED_PARTITION_NOT_CERTIFIED"
    return {
        "partition_status": partition_status,
        "role_policy_status": policy_status,
        "role_status": statuses,
        "role_reasons": reasons,
        "pairwise_overlap_counts": {
            "A_intersection_B": pairwise.a_intersection_b,
            "A_intersection_C": pairwise.a_intersection_c,
            "B_intersection_C": pairwise.b_intersection_c,
        },
    }


@dataclass(frozen=True)
class SupportC64SelectionAuthorization:
    schema: str
    authorization_id: str
    decision: str
    decision_id: str
    researcher_id: str
    category_id: str
    support_seed: int
    h_source_policy_proposal_sha256: str
    a_role_receipt_sha256: str
    source_manifest_sha256: str
    candidate_pool_sha256: str
    population: str
    support_count: int
    additional_c64_count: int
    allow_test_metadata: bool
    allow_outcome_labels: bool
    allow_masks: bool
    signature_ed25519_hex: str
    authorization_sha256: str


def candidate_pool_identity_sha256(
    category_id: str,
    source_manifest_sha256: str,
    source_group_sha256s: Sequence[str],
) -> str:
    """Hash a sorted candidate group set without emitting private identities."""
    if category_id not in CANONICAL_CATEGORY_IDS:
        raise R2L2DContractError("candidate-pool category is outside the frozen public catalog")
    _sha256(source_manifest_sha256, "candidate-pool source manifest SHA-256")
    groups = tuple(source_group_sha256s)
    if any(not isinstance(group, str) or _SHA256_RE.fullmatch(group) is None for group in groups):
        raise R2L2DContractError("candidate pool contains a malformed source-group identity")
    if not groups or len(set(groups)) != len(groups):
        raise R2L2DContractError("candidate pool is empty or contains duplicate source groups")
    return canonical_sha256({
        "schema": "r2_l2d.support_c64_candidate_pool.v1",
        "category_id": category_id,
        "source_manifest_sha256": source_manifest_sha256,
        "source_group_sha256s": tuple(sorted(groups)),
    })


def _selection_authorization_payload(
    authorization: SupportC64SelectionAuthorization,
) -> dict[str, Any]:
    payload = asdict(authorization)
    payload.pop("signature_ed25519_hex")
    payload.pop("authorization_sha256")
    return payload


def _assess_selection_authorization(
    authorization: SupportC64SelectionAuthorization,
    *,
    role_policy_proposal: RolePolicyProposal,
    a_role_receipt: RoleIdentityReceipt,
    candidate_group_sha256s: Sequence[str],
    trust_directory: ReviewerTrustDirectory,
    expected_trust_directory_sha256: str,
    expected_category_id: str,
    expected_support_seed: int,
) -> str:
    if not isinstance(authorization, SupportC64SelectionAuthorization):
        return "INVALID_SELECTION_AUTHORIZATION"
    if (authorization.schema != SUPPORT_SELECTION_AUTHORIZATION_SCHEMA
            or authorization.decision != "AUTHORIZE_TRAIN_NORMAL_SUPPORT_C64_SELECTION"
            or not isinstance(authorization.decision_id, str) or not authorization.decision_id.strip()
            or not isinstance(authorization.researcher_id, str) or not authorization.researcher_id.strip()
            or not isinstance(authorization.authorization_id, str) or not authorization.authorization_id.strip()):
        return "INVALID_SELECTION_AUTHORIZATION"
    if (isinstance(authorization.support_seed, bool)
            or not isinstance(authorization.support_seed, int)
            or authorization.support_seed < 0
            or authorization.category_id != expected_category_id
            or authorization.support_seed != expected_support_seed):
        return "SUPPORT_CATEGORY_OR_SEED_MISMATCH"
    if (authorization.h_source_policy_proposal_sha256 != role_policy_proposal.proposal_sha256
            or authorization.a_role_receipt_sha256 != a_role_receipt.receipt_sha256
            or authorization.source_manifest_sha256 != a_role_receipt.source_manifest_sha256):
        return "SELECTION_AUTHORIZATION_IDENTITY_MISMATCH"
    if (authorization.population != "CERTIFIED_A_NORMAL_TRAIN_ONLY"
            or authorization.support_count != 4
            or authorization.additional_c64_count != 64
            or authorization.allow_test_metadata is not False
            or authorization.allow_outcome_labels is not False
            or authorization.allow_masks is not False):
        return "SELECTION_AUTHORIZATION_SCOPE_INVALID"
    try:
        pool_sha = candidate_pool_identity_sha256(
            expected_category_id,
            a_role_receipt.source_manifest_sha256,
            candidate_group_sha256s,
        )
    except R2L2DContractError:
        return "INVALID_SELECTION_CANDIDATE_POOL"
    if authorization.candidate_pool_sha256 != pool_sha:
        return "SELECTION_AUTHORIZATION_CANDIDATE_POOL_MISMATCH"
    try:
        trust_is_pinned = (
            reviewer_trust_directory_sha256(trust_directory) == trust_directory.directory_sha256
            and trust_directory.directory_sha256 == expected_trust_directory_sha256
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        trust_is_pinned = False
    if not trust_is_pinned:
        return "PENDING_EXTERNAL_TRUST_ROOT"
    if (authorization.researcher_id not in trust_directory.trusted_policy_approver_ids
            or authorization.h_source_policy_proposal_sha256
            not in trust_directory.approved_proposal_sha256s):
        return "PENDING_SELECTION_AUTHORIZATION_TRUST"
    policy_keys = dict(trust_directory.trusted_policy_approver_public_keys)
    public_key = policy_keys.get(authorization.researcher_id)
    payload = _selection_authorization_payload(authorization)
    if (canonical_sha256(payload) != authorization.authorization_sha256
            or public_key is None
            or not _verify_ed25519(
                public_key, _canonical_bytes(payload), authorization.signature_ed25519_hex
            )):
        return "PENDING_SELECTION_AUTHORIZATION_SIGNATURE"
    return "SUPPORT_C64_SELECTION_AUTHORIZED"


@dataclass(frozen=True)
class SupportC64BankBindingReceipt:
    schema: str
    category_id: str
    support_seed: int
    a_role_receipt_sha256: str
    support_c64_selection_receipt_sha256: str
    selection_authorization_sha256: str
    fixed_support_digest_algorithm: str
    fixed_support_sha256: str
    fixed_support_receipt_sha256: str
    producer_id: str
    receipt_sha256: str


@dataclass(frozen=True)
class SupportC64BankBindingAttestation:
    reviewer_id: str
    decision_id: str
    receipt_sha256: str
    signature_ed25519_hex: str


def _binding_payload(receipt: SupportC64BankBindingReceipt) -> dict[str, Any]:
    payload = asdict(receipt)
    payload.pop("receipt_sha256")
    return payload


def _binding_attestation_payload(attestation: SupportC64BankBindingAttestation) -> dict[str, str]:
    return {
        "reviewer_id": attestation.reviewer_id,
        "decision_id": attestation.decision_id,
        "receipt_sha256": attestation.receipt_sha256,
    }


def make_support_c64_bank_binding_receipt(
    *,
    category_id: str,
    support_seed: int,
    a_role_receipt_sha256: str,
    selection_receipt_sha256: str,
    selection_authorization_sha256: str,
    fixed_support_digest_algorithm: str,
    fixed_support_sha256: str,
    producer_id: str,
) -> SupportC64BankBindingReceipt:
    if category_id not in CANONICAL_CATEGORY_IDS:
        raise R2L2DContractError("fixed bank category is outside the frozen public catalog")
    if isinstance(support_seed, bool) or not isinstance(support_seed, int) or support_seed < 0:
        raise R2L2DContractError("support seed must be a nonnegative integer")
    if fixed_support_digest_algorithm != R1_FIXED_SUPPORT_DIGEST_ALGORITHM:
        raise R2L2DContractError("fixed bank digest algorithm is incompatible")
    for value, name in (
        (a_role_receipt_sha256, "A role receipt SHA-256"),
        (selection_receipt_sha256, "Support/C64 selection receipt SHA-256"),
        (selection_authorization_sha256, "Support/C64 selection authorization SHA-256"),
        (fixed_support_sha256, "fixed bank SHA-256"),
    ):
        _sha256(value, name)
    if not isinstance(producer_id, str) or not producer_id.strip():
        raise R2L2DContractError("binding producer identity is required")
    if fixed_support_sha256 == R1_FIXED_SUPPORT_SHA256 and (
        category_id != R1_FIXED_SUPPORT_CATEGORY_ID or support_seed != R1_FIXED_SUPPORT_SEED
    ):
        raise R2L2DContractError("historical switch/seed0 bank cannot be reused cross-category or cross-seed")
    if (category_id == R1_FIXED_SUPPORT_CATEGORY_ID
            and support_seed == R1_FIXED_SUPPORT_SEED
            and fixed_support_sha256 != R1_FIXED_SUPPORT_SHA256):
        raise R2L2DContractError("switch/seed0 must use the retained historical structured bank digest")
    support_receipt_sha256 = category_fixed_support_receipt_sha256(
        category_id, support_seed, fixed_support_digest_algorithm, fixed_support_sha256,
    )
    partial = SupportC64BankBindingReceipt(
        schema=SUPPORT_BANK_BINDING_SCHEMA,
        category_id=category_id,
        support_seed=support_seed,
        a_role_receipt_sha256=a_role_receipt_sha256,
        support_c64_selection_receipt_sha256=selection_receipt_sha256,
        selection_authorization_sha256=selection_authorization_sha256,
        fixed_support_digest_algorithm=fixed_support_digest_algorithm,
        fixed_support_sha256=fixed_support_sha256,
        fixed_support_receipt_sha256=support_receipt_sha256,
        producer_id=producer_id,
        receipt_sha256="",
    )
    return SupportC64BankBindingReceipt(
        **_binding_payload(partial),
        receipt_sha256=canonical_sha256(_binding_payload(partial)),
    )


def assess_support_c64_bank_binding(
    *,
    a_role_receipt: RoleIdentityReceipt | None,
    a_role_manifest: ReviewedIdentityManifest | None,
    a_role_attestation: IndependentReviewerAttestation | None,
    selection_receipt: SupportC64SelectionReceipt | None,
    selection_attestation: SupportC64SelectionAttestation | None,
    candidate_group_sha256s: Sequence[str],
    binding_receipt: SupportC64BankBindingReceipt | None,
    binding_attestation: SupportC64BankBindingAttestation | None,
    trust_directory: ReviewerTrustDirectory | None,
    expected_trust_directory_sha256: str | None,
    expected_category_id: str,
    expected_support_seed: int,
    selection_authorization: SupportC64SelectionAuthorization | None = None,
    role_policy_proposal: RolePolicyProposal | None = None,
    researcher_approval: ResearcherApproval | None = None,
) -> str:
    """Require certified A, reviewed 4+64 selection, and signed bank binding."""
    if (not isinstance(role_policy_proposal, RolePolicyProposal)
            or not isinstance(researcher_approval, ResearcherApproval)):
        return "PENDING_H_SOURCE_POLICY_APPROVAL"
    if (role_policy_proposal.strategy != STRATEGY_H_SOURCE
            or role_policy_proposal.identity_contract_version != IDENTITY_CONTRACT_VERSION
            or role_policy_proposal.grouping_unit != GROUPING_UNIT):
        return "PENDING_H_SOURCE_POLICY_APPROVAL"
    try:
        policy_status = evaluate_policy_approval(
            role_policy_proposal,
            researcher_approval,
            trust_directory,
            expected_trust_directory_sha256=expected_trust_directory_sha256,
        )
    except (KeyError, TypeError, ValueError):
        return "PENDING_H_SOURCE_POLICY_APPROVAL"
    if policy_status != "RESEARCHER_POLICY_APPROVED":
        return "PENDING_H_SOURCE_POLICY_APPROVAL"
    if selection_authorization is None:
        return "PENDING_SELECTION_AUTHORIZATION"
    if any(value is None for value in (
        a_role_receipt, a_role_manifest, a_role_attestation,
        selection_receipt, selection_attestation,
        binding_receipt, binding_attestation,
    )):
        return "PENDING_REQUIRED_IDENTITY_OR_AUTHORIZATION"
    if trust_directory is None or expected_trust_directory_sha256 is None:
        return "PENDING_EXTERNAL_TRUST_ROOT"
    role_status, _reason = certify_role_identity(
        a_role_receipt, a_role_manifest, a_role_attestation, trust_directory,
        expected_trust_directory_sha256=expected_trust_directory_sha256,
    )
    if role_status != "CERTIFIED_FROM_SOURCE_EVIDENCE" or a_role_receipt.role != "A":
        return "PENDING_CERTIFIED_A_NORMAL_REFERENCE"
    if (a_role_receipt.category_id != expected_category_id
            or selection_receipt.category_id != expected_category_id
            or selection_receipt.support_seed != expected_support_seed):
        return "SUPPORT_CATEGORY_OR_SEED_MISMATCH"
    pool_status = support_c64_pool_status(candidate_group_sha256s)
    if pool_status["distinct_candidate_group_count"] < 68:
        return "CANDIDATE_POOL_INSUFFICIENT_FOR_4_SUPPORTS_PLUS_C64"
    if pool_status["candidate_pool_status"] != "CANDIDATE_POOL_SIZE_SUFFICIENT":
        return pool_status["candidate_pool_status"]
    authorization_status = _assess_selection_authorization(
        selection_authorization,
        role_policy_proposal=role_policy_proposal,
        a_role_receipt=a_role_receipt,
        candidate_group_sha256s=candidate_group_sha256s,
        trust_directory=trust_directory,
        expected_trust_directory_sha256=expected_trust_directory_sha256,
        expected_category_id=expected_category_id,
        expected_support_seed=expected_support_seed,
    )
    if authorization_status != "SUPPORT_C64_SELECTION_AUTHORIZED":
        return authorization_status
    separate_principals = {
        a_role_receipt.producer_id,
        a_role_attestation.reviewer_id,
        selection_receipt.producer_id,
        selection_attestation.reviewer_id,
        binding_receipt.producer_id,
        binding_attestation.reviewer_id,
    }
    if researcher_approval.researcher_id in separate_principals:
        return "PENDING_INDEPENDENT_SELECTION_AUTHORIZATION"
    if selection_authorization.researcher_id in separate_principals:
        return "PENDING_INDEPENDENT_SELECTION_AUTHORIZATION"
    selection_status = assess_support_c64_selection(
        candidate_group_sha256s,
        selection_receipt,
        selection_attestation,
        trust_directory,
        expected_trust_directory_sha256,
        expected_category_id=expected_category_id,
        expected_support_seed=expected_support_seed,
    )
    if selection_status != "SUPPORT_AND_C64_IDENTITIES_CERTIFIED":
        return selection_status
    a_groups = {member.group_sha256 for member in a_role_receipt.group_members}
    selected_groups = set(selection_receipt.support_group_sha256s) | set(selection_receipt.c64_group_sha256s)
    if not selected_groups.issubset(a_groups):
        return "SUPPORT_C64_OUTSIDE_CERTIFIED_A_ROLE"
    receipt = binding_receipt
    attestation = binding_attestation
    if (receipt.schema != SUPPORT_BANK_BINDING_SCHEMA
            or receipt.category_id != expected_category_id
            or receipt.support_seed != expected_support_seed
            or receipt.a_role_receipt_sha256 != a_role_receipt.receipt_sha256
            or receipt.support_c64_selection_receipt_sha256 != selection_receipt.receipt_sha256
            or receipt.selection_authorization_sha256 != selection_authorization.authorization_sha256
            or receipt.fixed_support_digest_algorithm != R1_FIXED_SUPPORT_DIGEST_ALGORITHM
            or canonical_sha256(_binding_payload(receipt)) != receipt.receipt_sha256):
        return "INVALID_SUPPORT_C64_FIXED_BANK_BINDING"
    try:
        expected_bank_receipt = category_fixed_support_receipt_sha256(
            expected_category_id,
            expected_support_seed,
            receipt.fixed_support_digest_algorithm,
            receipt.fixed_support_sha256,
        )
    except (TypeError, ValueError):
        return "INVALID_SUPPORT_C64_FIXED_BANK_BINDING"
    if receipt.fixed_support_receipt_sha256 != expected_bank_receipt:
        return "INVALID_SUPPORT_C64_FIXED_BANK_BINDING"
    if receipt.fixed_support_sha256 == R1_FIXED_SUPPORT_SHA256 and (
        expected_category_id != R1_FIXED_SUPPORT_CATEGORY_ID
        or expected_support_seed != R1_FIXED_SUPPORT_SEED
    ):
        return "HISTORICAL_FIXED_BANK_CROSS_CATEGORY_REJECTED"
    if (expected_category_id == R1_FIXED_SUPPORT_CATEGORY_ID
            and expected_support_seed == R1_FIXED_SUPPORT_SEED
            and receipt.fixed_support_sha256 != R1_FIXED_SUPPORT_SHA256):
        return "HISTORICAL_FIXED_BANK_MISMATCH"
    if (attestation.reviewer_id == receipt.producer_id
            or attestation.reviewer_id == selection_attestation.reviewer_id
            or attestation.reviewer_id == a_role_attestation.reviewer_id
            or attestation.reviewer_id == selection_authorization.researcher_id
            or attestation.reviewer_id == researcher_approval.researcher_id
            or attestation.reviewer_id not in trust_directory.trusted_reviewer_ids
            or attestation.receipt_sha256 != receipt.receipt_sha256
            or not attestation.decision_id.strip()):
        return "PENDING_INDEPENDENT_BINDING_REVIEW"
    if (reviewer_trust_directory_sha256(trust_directory) != trust_directory.directory_sha256
            or trust_directory.directory_sha256 != expected_trust_directory_sha256):
        return "PENDING_EXTERNAL_TRUST_ROOT"
    public_keys = dict(trust_directory.trusted_reviewer_public_keys)
    key = public_keys.get(attestation.reviewer_id)
    if key is None or not _verify_ed25519(
        key,
        _canonical_bytes(_binding_attestation_payload(attestation)),
        attestation.signature_ed25519_hex,
    ):
        return "PENDING_INDEPENDENT_BINDING_REVIEW"
    return "SUPPORT_C64_AND_FIXED_BANK_BOUND"
