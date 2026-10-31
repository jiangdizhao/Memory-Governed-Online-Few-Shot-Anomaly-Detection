"""Default-deny metadata-only Real-IAD TRAIN census for R2-L2C.

Mode A reads only repository protocol/authorization files. Mode B cannot touch
the dataset until a signed, pinned authorization covers the requested
projection, categories, co-located test-byte exposure, and output budget.
The reader projects only TRAIN image_path/anomaly_class and meta.normal_class/
prefix. It structurally skips the co-located test value without materializing
or interpreting its labels or masks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import threading
from typing import Any, Mapping, Sequence

from experiments.calibration.r2_l2a_allclass_inventory import (
    CANONICAL_CATEGORY_IDS,
    PINNED_SOURCE_SHA256,
    PINNED_SOURCE_URI,
    ROLE_NAMES,
)
from experiments.calibration.r2_l2b_role_certification import (
    IndependentReviewerAttestation,
    ReviewedIdentityManifest,
    ReviewerTrustDirectory,
    RoleIdentityReceipt,
    SOURCE_GROUP_DIGEST_ALGORITHM,
    canonical_sha256,
    certify_role_identity,
    reviewer_trust_directory_sha256,
    source_group_digest_from_key,
)
from experiments.risk_safe import preflight as identity_contract
from experiments.risk_safe.preflight import MalformedImageName, parse_image_name


SCHEMA = "r2_l2c.metadata_census_executor.v1"
AUTHORIZATION_SCHEMA = "r2_l2c.local_access_authorization.v1"
TRAIN_CENSUS_OPERATION = "R2_L2C_AUTHORIZED_TRAIN_METADATA_CENSUS_V1"
DEFAULT_DATASET_ROOT = r"G:\dataset\realiad"
DEFAULT_METADATA_ROOT = r"G:\dataset\realiad\jsons\realiad_jsons"
PROTECTED_CATEGORIES = frozenset({"mint", "pcb"})
REQUIRED_TRAIN_FIELDS = frozenset({"image_path", "anomaly_class"})
REQUIRED_META_FIELDS = frozenset({"normal_class", "prefix"})
REQUIRED_PUBLIC_OUTPUT_FIELDS = frozenset({
    "schema", "status", "operation", "authorization_id", "authorization_sha256",
    "source_group_digest_algorithm", "identity_contract_version", "grouping_unit",
    "public_catalog_source_uri", "public_catalog_source_sha256",
    "target_category_count", "authorized_category_count", "metadata_files_opened",
    "metadata_available_category_count", "missing_or_unavailable_categories",
    "partition_status", "local_data_io_performed", "test_section_parsed",
    "test_section_bytes_encountered", "categories", "execution_failure_type",
    "image_decoding_performed", "feature_extraction_performed", "geometry_performed",
    "category_id",
    "authorization_status",
    "metadata_availability_status",
    "authorized_train_record_count",
    "eligible_normal_image_count",
    "distinct_normal_source_group_count",
    "c64_candidate_pool_status",
    "support_and_c64_identity_status",
    "normal_camera_view_distribution",
    "malformed_identity_count",
    "duplicate_image_identity_count",
    "ambiguous_group_identity_count",
    "at_least_68_legal_normal_groups",
    "a_role_candidate_status",
    "abc_partition_status",
    "protected_category_status",
    "directory_serial_overlap_status",
    "directory_serial_overlap_count",
    "source_json_sha256",
    "parser_source_sha256",
    "identity_parser_source_sha256",
    "unresolved_metadata_limitations",
})
REQUIRED_PRIVATE_RECEIPT_FIELDS = frozenset({
    "schema", "authorization_id", "authorization_sha256",
    "source_group_digest_algorithm", "receipts", "category_id",
    "source_json_sha256", "group_members", "source_group_sha256",
    "camera_view_count", "directory_serial_overlap_count",
    "role_certification_status", "contains_raw_sample_keys",
    "contains_image_paths", "contains_labels_or_masks", "partition_status",
    "role_status", "role_reasons", "cross_role_overlap_count",
    "c_identity_certificate_status", "A", "B", "C",
})
FORBIDDEN_PUBLIC_FIELDS = frozenset({
    "image_path", "official_sample_key", "sample_key", "anomaly_class",
    "label", "mask", "mask_path", "source_group_sha256", "group_members",
})
FORBIDDEN_PRIVATE_FIELDS = frozenset({
    "image_path", "official_sample_key", "sample_key", "anomaly_class",
    "label", "mask", "mask_path",
})
_NUMBER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_AUTH_KEYS = frozenset({
    "schema", "authorization_id", "researcher_id", "decision",
    "dataset_root", "metadata_root", "approved_categories",
    "explicitly_authorized_protected_categories",
    "allow_physical_json_file_bytes", "allow_test_section_bytes_encountered",
    "allow_train_image_path", "allow_train_anomaly_class",
    "allow_official_sample_key_condition_tokens", "allow_meta_normal_class",
    "allow_meta_prefix", "allow_directory_serial_sensitivity",
    "allow_private_group_receipt_output", "allow_test_section_parsing",
    "allow_mask_access", "allow_feature_extraction", "allow_image_decoding",
    "allow_support_c64_selection", "approved_output_fields",
    "approved_private_receipt_fields", "output_directory",
    "maximum_output_bytes", "signature_ed25519_hex", "authorization_sha256",
})


class CensusContractError(ValueError):
    """Authorization, projection, identity, or output contract failed."""


class OutputBudgetError(RuntimeError):
    """A pre-write output budget or writer-accounting check failed."""


class ProjectionJSONError(ValueError):
    """Input JSON cannot be safely projected under the frozen contract."""


@dataclass(frozen=True)
class LocalAccessAuthorization:
    schema: str
    authorization_id: str
    researcher_id: str
    decision: str
    dataset_root: str
    metadata_root: str
    approved_categories: tuple[str, ...]
    explicitly_authorized_protected_categories: tuple[str, ...]
    allow_physical_json_file_bytes: bool
    allow_test_section_bytes_encountered: bool
    allow_train_image_path: bool
    allow_train_anomaly_class: bool
    allow_official_sample_key_condition_tokens: bool
    allow_meta_normal_class: bool
    allow_meta_prefix: bool
    allow_directory_serial_sensitivity: bool
    allow_private_group_receipt_output: bool
    allow_test_section_parsing: bool
    allow_mask_access: bool
    allow_feature_extraction: bool
    allow_image_decoding: bool
    allow_support_c64_selection: bool
    approved_output_fields: tuple[str, ...]
    approved_private_receipt_fields: tuple[str, ...]
    output_directory: str
    maximum_output_bytes: int
    signature_ed25519_hex: str
    authorization_sha256: str


@dataclass(frozen=True)
class RoleEvidence:
    receipt: RoleIdentityReceipt
    manifest: ReviewedIdentityManifest
    attestation: IndependentReviewerAttestation


@dataclass(frozen=True)
class SupportC64SelectionReceipt:
    schema: str
    category_id: str
    support_seed: int
    support_group_sha256s: tuple[str, ...]
    c64_group_sha256s: tuple[str, ...]
    producer_id: str
    receipt_sha256: str


@dataclass(frozen=True)
class SupportC64SelectionAttestation:
    reviewer_id: str
    decision_id: str
    receipt_sha256: str
    signature_ed25519_hex: str


@dataclass(frozen=True)
class _CensusProjection:
    meta: Mapping[str, Any]
    train_rows: tuple[Mapping[str, Any], ...]
    test_section_present: bool


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_absolute_local_path(value: str) -> bool:
    win = PureWindowsPath(value)
    posix = PurePosixPath(value)
    return win.is_absolute() or posix.is_absolute()


def _normalized_path_key(value: str) -> str:
    """Compare approved/requested paths lexically without touching the path."""
    if not isinstance(value, str) or not value.strip():
        raise CensusContractError("authorized path is empty")
    if ".." in PureWindowsPath(value).parts or ".." in PurePosixPath(value).parts:
        raise CensusContractError("authorized path contains parent traversal")
    if not _is_absolute_local_path(value):
        raise CensusContractError("authorized path must be absolute")
    return str(PureWindowsPath(value)).replace("/", "\\").rstrip("\\").casefold()


def _reject_local_approval_path_under_dataset(path: str | Path) -> None:
    """Reject approval inputs lexically before any path probe or file open."""
    candidate = os.path.abspath(os.fspath(path))
    candidate_key = _normalized_path_key(candidate)
    dataset_key = _normalized_path_key(DEFAULT_DATASET_ROOT)
    if candidate_key == dataset_key or candidate_key.startswith(dataset_key + "\\"):
        raise CensusContractError("authorization documents must be outside the dataset root")
    _reject_reparse_chain(Path(candidate))


def authorization_payload(authorization: LocalAccessAuthorization) -> dict[str, Any]:
    payload = asdict(authorization)
    payload.pop("signature_ed25519_hex")
    payload.pop("authorization_sha256")
    return payload


def authorization_from_mapping(value: Mapping[str, Any]) -> LocalAccessAuthorization:
    if not isinstance(value, Mapping):
        raise CensusContractError("authorization manifest must be a JSON object")
    supplied = set(value)
    if supplied != _EXPECTED_AUTH_KEYS:
        missing = sorted(_EXPECTED_AUTH_KEYS - supplied)
        extra = sorted(supplied - _EXPECTED_AUTH_KEYS)
        raise CensusContractError(
            f"authorization manifest fields mismatch; missing={missing}, extra={extra}")
    sequence_fields = (
        "approved_categories", "explicitly_authorized_protected_categories",
        "approved_output_fields", "approved_private_receipt_fields",
    )
    converted = dict(value)
    for name in sequence_fields:
        field_value = converted[name]
        if not isinstance(field_value, list) or any(not isinstance(item, str) for item in field_value):
            raise CensusContractError(f"{name} must be an array of strings")
        converted[name] = tuple(field_value)
    string_fields = (
        "schema", "authorization_id", "researcher_id", "decision", "dataset_root",
        "metadata_root", "output_directory", "signature_ed25519_hex",
        "authorization_sha256",
    )
    if any(not isinstance(converted[name], str) for name in string_fields):
        raise CensusContractError("authorization string field is malformed")
    for name in (
        "allow_physical_json_file_bytes", "allow_test_section_bytes_encountered",
        "allow_train_image_path", "allow_train_anomaly_class",
        "allow_official_sample_key_condition_tokens", "allow_meta_normal_class",
        "allow_meta_prefix", "allow_directory_serial_sensitivity",
        "allow_private_group_receipt_output", "allow_test_section_parsing",
        "allow_mask_access", "allow_feature_extraction", "allow_image_decoding",
        "allow_support_c64_selection",
    ):
        if type(converted[name]) is not bool:
            raise CensusContractError(f"{name} must be a JSON boolean")
    if isinstance(converted["maximum_output_bytes"], bool) or not isinstance(
        converted["maximum_output_bytes"], int
    ):
        raise CensusContractError("maximum_output_bytes must be an integer")
    try:
        return LocalAccessAuthorization(**converted)
    except TypeError as exc:
        raise CensusContractError("authorization manifest does not match its schema") from exc


def trust_directory_from_mapping(value: Mapping[str, Any]) -> ReviewerTrustDirectory:
    expected = {field.name for field in fields(ReviewerTrustDirectory)}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CensusContractError("reviewer trust directory fields do not match the pinned schema")
    converted = dict(value)
    tuple_fields = (
        "trusted_reviewer_ids", "trusted_policy_approver_ids",
        "trusted_identity_source_sha256s", "reviewed_empty_role_policy_sha256s",
        "reviewed_a_availability_receipt_sha256s", "approved_proposal_sha256s",
    )
    pair_fields = ("trusted_reviewer_public_keys", "trusted_policy_approver_public_keys")
    for name in tuple_fields:
        if not isinstance(converted[name], list) or any(not isinstance(x, str) for x in converted[name]):
            raise CensusContractError(f"trust-directory field {name} is malformed")
        converted[name] = tuple(converted[name])
    for name in pair_fields:
        pairs = converted[name]
        if not isinstance(pairs, list) or any(
            not isinstance(pair, list) or len(pair) != 2
            or not all(isinstance(x, str) for x in pair)
            for pair in pairs
        ):
            raise CensusContractError(f"trust-directory field {name} is malformed")
        converted[name] = tuple(tuple(pair) for pair in pairs)
    try:
        return ReviewerTrustDirectory(**converted)
    except TypeError as exc:
        raise CensusContractError("reviewer trust directory does not match its schema") from exc


def _verify_ed25519(public_key_hex: str, message: bytes, signature_hex: str) -> bool:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        if (re.fullmatch(r"[0-9a-f]{64}", public_key_hex) is None
                or re.fullmatch(r"[0-9a-f]{128}", signature_hex) is None):
            return False
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
            bytes.fromhex(signature_hex), message)
        return True
    except Exception:
        return False


def authorization_errors(
    authorization: LocalAccessAuthorization | None,
    trust_directory: ReviewerTrustDirectory | None,
    expected_trust_directory_sha256: str | None,
    *,
    expected_dataset_root: str = DEFAULT_DATASET_ROOT,
) -> tuple[str, ...]:
    """Validate every permission for this exact TRAIN census before data I/O."""
    errors: list[str] = []
    if authorization is None:
        return ("signed local-access authorization manifest was not supplied",)
    if authorization.schema != AUTHORIZATION_SCHEMA:
        errors.append("authorization schema is not supported")
    if authorization.decision != TRAIN_CENSUS_OPERATION:
        errors.append("authorization does not approve the R2-L2C TRAIN census operation")
    if not authorization.authorization_id.strip() or not authorization.researcher_id.strip():
        errors.append("authorization identity or decision id is absent")
    try:
        if _normalized_path_key(authorization.dataset_root) != _normalized_path_key(expected_dataset_root):
            errors.append("authorization dataset root differs from the requested root")
        expected_metadata = str(PureWindowsPath(expected_dataset_root) / "jsons" / "realiad_jsons")
        if _normalized_path_key(authorization.metadata_root) != _normalized_path_key(expected_metadata):
            errors.append("authorization metadata root differs from the requested TRAIN census root")
        output_key = _normalized_path_key(authorization.output_directory)
        dataset_key = _normalized_path_key(authorization.dataset_root)
        if output_key == dataset_key or output_key.startswith(dataset_key + "\\"):
            errors.append("output directory must be outside the dataset root")
    except CensusContractError as exc:
        errors.append(str(exc))

    categories = authorization.approved_categories
    if not categories or len(categories) != len(set(categories)):
        errors.append("approved category list is empty or contains duplicates")
    if any(category not in CANONICAL_CATEGORY_IDS for category in categories):
        errors.append("approved category list contains an unknown public category")
    protected = authorization.explicitly_authorized_protected_categories
    if len(protected) != len(set(protected)) or not set(protected).issubset(PROTECTED_CATEGORIES):
        errors.append("protected-category authorization list is malformed")
    if not set(protected).issubset(categories):
        errors.append("protected authorization names a category outside the approved set")
    if (len(authorization.approved_output_fields) != len(set(authorization.approved_output_fields))
            or len(authorization.approved_private_receipt_fields)
            != len(set(authorization.approved_private_receipt_fields))):
        errors.append("approved output field lists contain duplicates")

    required_permissions = {
        "physical JSON byte access": authorization.allow_physical_json_file_bytes,
        "co-located test-section byte acknowledgment": authorization.allow_test_section_bytes_encountered,
        "TRAIN image_path parsing": authorization.allow_train_image_path,
        "TRAIN anomaly_class parsing": authorization.allow_train_anomaly_class,
        "official sample-key condition-token interpretation": authorization.allow_official_sample_key_condition_tokens,
        "meta.normal_class parsing": authorization.allow_meta_normal_class,
        "meta.prefix parsing": authorization.allow_meta_prefix,
        "local private group-receipt output": authorization.allow_private_group_receipt_output,
    }
    errors.extend(f"{name} is not explicitly authorized" for name, allowed in required_permissions.items() if not allowed)
    if authorization.allow_test_section_parsing:
        errors.append("test-section parsing is outside this operation")
    if authorization.allow_mask_access:
        errors.append("mask access is outside this operation")
    if authorization.allow_feature_extraction:
        errors.append("feature extraction is outside this operation")
    if authorization.allow_image_decoding:
        errors.append("image decoding is outside this operation")
    if authorization.allow_support_c64_selection:
        errors.append("Support/C64 selection is outside this census operation")
    if not REQUIRED_PUBLIC_OUTPUT_FIELDS.issubset(authorization.approved_output_fields):
        errors.append("approved output fields do not cover the required aggregate report")
    if set(authorization.approved_output_fields) - REQUIRED_PUBLIC_OUTPUT_FIELDS:
        errors.append("approved output fields contain unrecognized names")
    if set(authorization.approved_output_fields).intersection(FORBIDDEN_PUBLIC_FIELDS):
        errors.append("shareable output authorization includes a raw or label-bearing field")
    if not REQUIRED_PRIVATE_RECEIPT_FIELDS.issubset(authorization.approved_private_receipt_fields):
        errors.append("private output authorization does not cover local group receipts")
    if set(authorization.approved_private_receipt_fields).intersection(FORBIDDEN_PRIVATE_FIELDS):
        errors.append("private receipt authorization includes raw paths, keys, labels, or masks")
    if set(authorization.approved_private_receipt_fields) - REQUIRED_PRIVATE_RECEIPT_FIELDS:
        errors.append("approved private receipt fields contain unrecognized names")
    if authorization.maximum_output_bytes <= 0:
        errors.append("positive output-byte budget is absent")

    if trust_directory is None or expected_trust_directory_sha256 is None:
        errors.append("independently pinned researcher trust directory is absent")
    else:
        try:
            _validate_sha256(expected_trust_directory_sha256, "pinned trust-directory SHA-256")
            if (reviewer_trust_directory_sha256(trust_directory) != trust_directory.directory_sha256
                    or trust_directory.directory_sha256 != expected_trust_directory_sha256):
                errors.append("reviewer trust directory does not match its out-of-band SHA-256 pin")
            policy_keys = dict(trust_directory.trusted_policy_approver_public_keys)
            policy_key = policy_keys.get(authorization.researcher_id)
            if (authorization.researcher_id not in trust_directory.trusted_policy_approver_ids
                    or policy_key is None):
                errors.append("researcher is absent from the pinned policy-approver trust root")
            elif not _verify_ed25519(
                policy_key,
                _canonical_bytes(authorization_payload(authorization)),
                authorization.signature_ed25519_hex,
            ):
                errors.append("researcher authorization Ed25519 signature is invalid")
        except CensusContractError as exc:
            errors.append(str(exc))
    try:
        _validate_sha256(authorization.authorization_sha256, "authorization SHA-256")
        if canonical_sha256(authorization_payload(authorization)) != authorization.authorization_sha256:
            errors.append("authorization digest does not match its contents")
    except CensusContractError as exc:
        errors.append(str(exc))
    return tuple(errors)


def _validate_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256_RE.fullmatch(value) is None:
        raise CensusContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _json_object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CensusContractError(f"duplicate JSON key in authorization document: {key}")
        result[key] = value
    return result


def _read_json_document(path: str | Path) -> Mapping[str, Any]:
    _reject_local_approval_path_under_dataset(path)
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_json_object_no_duplicates)
    if not isinstance(value, Mapping):
        raise CensusContractError("authorization/trust document must be a JSON object")
    return value


def load_authorization_and_trust(
    authorization_path: str | Path,
    trust_directory_path: str | Path,
) -> tuple[LocalAccessAuthorization, ReviewerTrustDirectory]:
    authorization = authorization_from_mapping(_read_json_document(authorization_path))
    trust_directory = trust_directory_from_mapping(_read_json_document(trust_directory_path))
    return authorization, trust_directory


def build_mode_a_report(
    authorization: LocalAccessAuthorization | None = None,
    validation_errors: Sequence[str] = (),
    *,
    authorization_valid: bool = False,
) -> dict[str, Any]:
    """Build the default-deny report without probing any dataset path."""
    rows = []
    for category in CANONICAL_CATEGORY_IDS:
        protected = category in PROTECTED_CATEGORIES
        individually_authorized = (
            authorization_valid
            and authorization is not None
            and category in authorization.approved_categories
            and (not protected or category in authorization.explicitly_authorized_protected_categories)
        )
        rows.append({
            "category_id": category,
            "authorization_status": "APPROVED_SCOPE_NOT_EXECUTED" if individually_authorized else (
                "PROTECTED_NOT_AUTHORIZED" if protected else "NOT_AUTHORIZED"),
            "metadata_availability_status": "NOT_CHECKED_NO_LOCAL_IO",
            "authorized_train_record_count": None,
            "eligible_normal_image_count": None,
            "distinct_normal_source_group_count": None,
            "c64_candidate_pool_status": "NOT_CHECKED_NO_LOCAL_IO",
            "support_and_c64_identity_status": "NOT_CERTIFIED",
            "normal_camera_view_distribution": None,
            "malformed_identity_count": None,
            "duplicate_image_identity_count": None,
            "ambiguous_group_identity_count": None,
            "at_least_68_legal_normal_groups": None,
            "a_role_candidate_status": "PENDING_LOCAL_METADATA_AUTHORIZATION",
            "abc_partition_status": "A_CANDIDATES_NOT_IDENTIFIED",
            "protected_category_status": "PROTECTED_NOT_AUTHORIZED" if protected and not individually_authorized else (
                "PROTECTED_SCOPE_APPROVED_NOT_EXECUTED" if protected else "NOT_PROTECTED"),
            "directory_serial_overlap_status": "NOT_CHECKED_NO_LOCAL_IO",
            "directory_serial_overlap_count": None,
            "source_json_sha256": None,
            "parser_source_sha256": None,
            "identity_parser_source_sha256": None,
            "unresolved_metadata_limitations": [
                "local metadata was not accessed in Mode A",
            ],
        })
    missing = list(validation_errors)
    if authorization is None:
        missing.extend([
            "signed researcher approval for the exact R2-L2C TRAIN census operation",
            "out-of-band pinned trust-directory SHA-256 and trusted researcher public key",
            "explicit approved local root and category allowlist, including separate Mint/PCB decisions",
            "physical JSON-byte access and acknowledgment that co-located test bytes may be encountered",
            "TRAIN image_path/anomaly_class and meta.normal_class/prefix permissions",
            "official sample-key OK/NG/defect-token interpretation permission",
            "approved shareable fields, local group-receipt fields, output path, and byte budget",
        ])
    return {
        "schema": SCHEMA,
        "status": "R2_L2C_EXECUTOR_READY_AWAITING_AUTHORIZATION",
        "local_access_authorization_status": "R2_L2C_LOCAL_ACCESS_NOT_AUTHORIZED",
        "mode": "preflight",
        "local_data_io_performed": False,
        "public_catalog": {
            "source_uri": PINNED_SOURCE_URI,
            "source_sha256": PINNED_SOURCE_SHA256,
            "categories": list(CANONICAL_CATEGORY_IDS),
            "category_count": len(CANONICAL_CATEGORY_IDS),
        },
        "denominators": {
            "target_category_count": len(CANONICAL_CATEGORY_IDS),
            "authorized_categories_checked": 0,
            "metadata_files_opened": 0,
        },
        "missing_categories": list(CANONICAL_CATEGORY_IDS),
        "missing_approvals": missing,
        "categories": rows,
    }


def preflight(
    authorization: LocalAccessAuthorization | None = None,
    trust_directory: ReviewerTrustDirectory | None = None,
    expected_trust_directory_sha256: str | None = None,
    *,
    expected_dataset_root: str = DEFAULT_DATASET_ROOT,
) -> dict[str, Any]:
    """Validate optional authorization artifacts; never stat the dataset root."""
    if authorization is None:
        return build_mode_a_report()
    errors = authorization_errors(
        authorization, trust_directory, expected_trust_directory_sha256,
        expected_dataset_root=expected_dataset_root,
    )
    report = build_mode_a_report(authorization, errors)
    if not errors:
        report = build_mode_a_report(authorization, (), authorization_valid=True)
        report["status"] = "R2_L2C_EXECUTOR_READY_AWAITING_AUTHORIZATION"
        report["local_access_authorization_status"] = "VALIDATED_NOT_EXECUTED_MODE_A_ONLY"
        report["missing_approvals"] = []
    return report


class _ProjectionParser:
    """Small JSON projection parser that skips unrequested values structurally."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0

    def _ws(self) -> None:
        while self.index < len(self.text) and self.text[self.index] in " \t\r\n":
            self.index += 1

    def _expect(self, char: str) -> None:
        self._ws()
        if self.index >= len(self.text) or self.text[self.index] != char:
            raise ProjectionJSONError(f"expected {char!r} in JSON projection")
        self.index += 1

    def _scan_string(self) -> None:
        if self.index >= len(self.text) or self.text[self.index] != '"':
            raise ProjectionJSONError("expected JSON string")
        self.index += 1
        while self.index < len(self.text):
            char = self.text[self.index]
            self.index += 1
            if char == '"':
                return
            if ord(char) < 0x20:
                raise ProjectionJSONError("unescaped control character in JSON string")
            if char == "\\":
                if self.index >= len(self.text):
                    raise ProjectionJSONError("truncated JSON escape")
                escape = self.text[self.index]
                self.index += 1
                if escape in '"\\/bfnrt':
                    continue
                if escape == "u":
                    digits = self.text[self.index:self.index + 4]
                    if len(digits) != 4 or re.fullmatch(r"[0-9a-fA-F]{4}", digits) is None:
                        raise ProjectionJSONError("malformed JSON unicode escape")
                    self.index += 4
                    continue
                raise ProjectionJSONError("invalid JSON string escape")
        raise ProjectionJSONError("unterminated JSON string")

    def _read_key(self) -> str:
        self._ws()
        start = self.index
        self._scan_string()
        try:
            value, end = json.JSONDecoder().raw_decode(self.text, start)
        except json.JSONDecodeError as exc:
            raise ProjectionJSONError("malformed JSON object key") from exc
        if not isinstance(value, str) or end != self.index:
            raise ProjectionJSONError("malformed JSON object key")
        return value

    def _skip_number_or_literal(self) -> None:
        start = self.index
        for literal in ("true", "false", "null"):
            if self.text.startswith(literal, self.index):
                self.index += len(literal)
                return
        match = _NUMBER_RE.match(self.text, self.index)
        if match is None:
            raise ProjectionJSONError("malformed JSON primitive")
        self.index = match.end()
        if self.index == start:
            raise ProjectionJSONError("malformed JSON primitive")

    def _skip_value(self) -> None:
        self._ws()
        if self.index >= len(self.text):
            raise ProjectionJSONError("missing JSON value")
        char = self.text[self.index]
        if char == '"':
            self._scan_string()
            return
        if char == "{":
            self.index += 1
            self._ws()
            if self.index < len(self.text) and self.text[self.index] == "}":
                self.index += 1
                return
            while True:
                self._read_key()
                self._expect(":")
                self._skip_value()
                self._ws()
                if self.index < len(self.text) and self.text[self.index] == ",":
                    self.index += 1
                    continue
                self._expect("}")
                return
        if char == "[":
            self.index += 1
            self._ws()
            if self.index < len(self.text) and self.text[self.index] == "]":
                self.index += 1
                return
            while True:
                self._skip_value()
                self._ws()
                if self.index < len(self.text) and self.text[self.index] == ",":
                    self.index += 1
                    continue
                self._expect("]")
                return
        self._skip_number_or_literal()

    def _read_selected_string(self) -> str | None:
        self._ws()
        if self.index >= len(self.text) or self.text[self.index] != '"':
            self._skip_value()
            return None
        start = self.index
        self._scan_string()
        try:
            value, end = json.JSONDecoder().raw_decode(self.text, start)
        except json.JSONDecodeError as exc:
            raise ProjectionJSONError("malformed selected JSON string") from exc
        if not isinstance(value, str) or end != self.index:
            raise ProjectionJSONError("malformed selected JSON string")
        return value

    def _project_object(self, selected_fields: frozenset[str]) -> dict[str, Any]:
        self._expect("{")
        output: dict[str, Any] = {}
        observed: set[str] = set()
        self._ws()
        if self.index < len(self.text) and self.text[self.index] == "}":
            self.index += 1
            return output
        while True:
            key = self._read_key()
            if key in observed:
                raise ProjectionJSONError("duplicate key in projected JSON object")
            observed.add(key)
            self._expect(":")
            if key in selected_fields:
                output[key] = self._read_selected_string()
            else:
                self._skip_value()
            self._ws()
            if self.index < len(self.text) and self.text[self.index] == ",":
                self.index += 1
                continue
            self._expect("}")
            return output

    def _project_train_array(self) -> tuple[Mapping[str, Any], ...]:
        self._expect("[")
        rows: list[Mapping[str, Any]] = []
        self._ws()
        if self.index < len(self.text) and self.text[self.index] == "]":
            self.index += 1
            return ()
        while True:
            self._ws()
            if self.index >= len(self.text) or self.text[self.index] != "{":
                raise ProjectionJSONError("TRAIN records must be JSON objects")
            rows.append(self._project_object(frozenset(REQUIRED_TRAIN_FIELDS)))
            self._ws()
            if self.index < len(self.text) and self.text[self.index] == ",":
                self.index += 1
                continue
            self._expect("]")
            return tuple(rows)

    def project(self) -> _CensusProjection:
        self._expect("{")
        top: set[str] = set()
        meta: Mapping[str, Any] | None = None
        train: tuple[Mapping[str, Any], ...] | None = None
        test_present = False
        self._ws()
        if self.index < len(self.text) and self.text[self.index] == "}":
            raise ProjectionJSONError("top-level JSON object is empty")
        while True:
            key = self._read_key()
            if key in top:
                raise ProjectionJSONError("duplicate top-level JSON key")
            top.add(key)
            self._expect(":")
            if key == "meta":
                meta = self._project_object(frozenset(REQUIRED_META_FIELDS))
            elif key == "train":
                train = self._project_train_array()
            elif key == "test":
                test_present = True
                self._skip_value()
            else:
                raise ProjectionJSONError("unrecognized top-level JSON section")
            self._ws()
            if self.index < len(self.text) and self.text[self.index] == ",":
                self.index += 1
                continue
            self._expect("}")
            break
        self._ws()
        if self.index != len(self.text):
            raise ProjectionJSONError("trailing content after top-level JSON object")
        if meta is None or train is None:
            raise ProjectionJSONError("required meta or TRAIN section is missing")
        if not isinstance(meta.get("normal_class"), str) or not isinstance(meta.get("prefix"), str):
            raise ProjectionJSONError("meta.normal_class or meta.prefix is missing or not a string")
        return _CensusProjection(meta, train, test_present)


def project_train_json_bytes(source_bytes: bytes) -> _CensusProjection:
    """Project selected TRAIN metadata; test values are never materialized."""
    try:
        text = source_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProjectionJSONError("metadata JSON is not valid UTF-8") from exc
    return _ProjectionParser(text).project()


def _safe_relative_parts(value: str) -> tuple[str, ...]:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise CensusContractError("image path or meta.prefix is not a safe relative path")
    return tuple(part for part in path.parts if part not in {"", "/"})


def _image_identity_path(prefix: str, image_path: str) -> str:
    parts = _safe_relative_parts(prefix) + _safe_relative_parts(image_path)
    if not parts:
        raise CensusContractError("image identity path is empty")
    return "/".join(("images",) + parts)


def _reparse_or_symlink(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _reject_reparse_chain(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    for item in (absolute, *absolute.parents):
        if _reparse_or_symlink(item):
            raise CensusContractError("symlink or reparse-point path is not allowed")


def _authorized_categories(authorization: LocalAccessAuthorization) -> frozenset[str]:
    return frozenset(
        category for category in authorization.approved_categories
        if category not in PROTECTED_CATEGORIES
        or category in authorization.explicitly_authorized_protected_categories
    )


def _empty_category_row(category: str, auth_status: str, protected_status: str) -> dict[str, Any]:
    return {
        "category_id": category,
        "authorization_status": auth_status,
        "metadata_availability_status": "NOT_CHECKED_NOT_AUTHORIZED",
        "authorized_train_record_count": None,
        "eligible_normal_image_count": None,
        "distinct_normal_source_group_count": None,
        "c64_candidate_pool_status": "NOT_CHECKED_NO_LOCAL_IO",
        "support_and_c64_identity_status": "NOT_CERTIFIED",
        "normal_camera_view_distribution": None,
        "malformed_identity_count": None,
        "duplicate_image_identity_count": None,
        "ambiguous_group_identity_count": None,
        "at_least_68_legal_normal_groups": None,
        "a_role_candidate_status": "PENDING_AUTHORIZATION",
        "abc_partition_status": "A_CANDIDATES_NOT_IDENTIFIED",
        "protected_category_status": protected_status,
        "directory_serial_overlap_status": "NOT_AUTHORIZED",
        "directory_serial_overlap_count": None,
        "source_json_sha256": None,
        "parser_source_sha256": None,
        "identity_parser_source_sha256": None,
        "unresolved_metadata_limitations": ["category was not opened"],
    }


def _category_census(
    category: str,
    source_bytes: bytes,
    authorization: LocalAccessAuthorization,
    parser_sha: str,
    identity_parser_sha: str,
    trust_directory: ReviewerTrustDirectory | None,
    expected_trust_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    projection = project_train_json_bytes(source_bytes)
    normal_class = projection.meta["normal_class"]
    prefix = projection.meta["prefix"]
    rows = projection.train_rows
    malformed = 0
    duplicate = 0
    ambiguous = 0
    unclassified = 0
    seen_images: set[str] = set()
    normal_images = 0
    group_rows: dict[str, dict[str, Any]] = {}
    serial_groups: dict[str, set[str]] = defaultdict(set)
    limitations: list[str] = []

    for row in rows:
        image_path = row.get("image_path")
        anomaly_class = row.get("anomaly_class")
        if not isinstance(image_path, str) or not image_path:
            malformed += 1
            continue
        if not isinstance(anomaly_class, str):
            unclassified += 1
            continue
        try:
            identity_path = _image_identity_path(prefix, image_path)
            identity = parse_image_name(identity_path, expected_class=category)
            group_sha = source_group_digest_from_key(category, identity.official_sample_key)
        except (CensusContractError, MalformedImageName, ValueError):
            malformed += 1
            continue
        if identity.image_id in seen_images:
            duplicate += 1
            continue
        seen_images.add(identity.image_id)
        if authorization.allow_directory_serial_sensitivity:
            serial_groups[identity.directory_serial_key].add(group_sha)
        is_normal_label = anomaly_class == normal_class
        is_ok_identity = identity.capture_session == "OK"
        if is_normal_label != is_ok_identity:
            ambiguous += 1
            continue
        if not is_normal_label:
            continue
        normal_images += 1
        existing = group_rows.get(group_sha)
        if existing is None:
            group_rows[group_sha] = {
                "sample_key": identity.official_sample_key,
                "camera_ids": {identity.camera_id},
                "image_count": 1,
                "directory_serial_key": identity.directory_serial_key,
            }
        else:
            if existing["sample_key"] != identity.official_sample_key:
                ambiguous += 1
                continue
            existing["camera_ids"].add(identity.camera_id)
            existing["image_count"] += 1

    group_views = {digest: len(item["camera_ids"]) for digest, item in group_rows.items()}
    distribution = Counter(group_views.values())
    if unclassified:
        limitations.append("some TRAIN rows lack an interpretable anomaly_class")
    if malformed:
        limitations.append("some TRAIN image identities were malformed or failed class validation")
    if duplicate:
        limitations.append("duplicate TRAIN image identities were excluded from eligible counts")
    if ambiguous:
        limitations.append("normal labels and official OK/NG identity tokens conflicted")
    if not rows:
        limitations.append("TRAIN section contained no records")

    source_sha = _sha256_bytes(source_bytes)
    candidate_groups = tuple(sorted(group_rows))
    c64_pool = support_c64_pool_status(candidate_groups)
    role_status = assess_h_source_partition(
        category,
        candidate_groups,
        {},
        trust_directory,
        expected_trust_sha256,
    )
    if candidate_groups:
        a_status = "A_CANDIDATES_IDENTIFIED"
    else:
        a_status = "NO_AUTHORIZED_NORMAL_REFERENCE_EVIDENCE"
    serial_overlap_count = None
    serial_status = "NOT_AUTHORIZED"
    if authorization.allow_directory_serial_sensitivity:
        serial_overlap_count = sum(1 for groups in serial_groups.values() if len(groups) > 1)
        serial_status = "DIRECTORY_SERIAL_OVERLAP_SENSITIVITY_REPORTED"

    public_row = {
        "category_id": category,
        "authorization_status": "AUTHORIZED_TRAIN_METADATA_ONLY",
        "metadata_availability_status": "METADATA_AVAILABLE",
        "authorized_train_record_count": len(rows),
        "eligible_normal_image_count": normal_images,
        "distinct_normal_source_group_count": len(candidate_groups),
        "c64_candidate_pool_status": c64_pool["candidate_pool_status"],
        "support_and_c64_identity_status": c64_pool["support_and_c64_identity_status"],
        "normal_camera_view_distribution": {
            str(view_count): distribution[view_count]
            for view_count in sorted(distribution)
        },
        "malformed_identity_count": malformed,
        "duplicate_image_identity_count": duplicate,
        "ambiguous_group_identity_count": ambiguous,
        "at_least_68_legal_normal_groups": len(candidate_groups) >= 68,
        "a_role_candidate_status": a_status,
        "abc_partition_status": role_status["partition_status"],
        "protected_category_status": (
            "PROTECTED_ACCESS_AUTHORIZED" if category in PROTECTED_CATEGORIES
            else "NOT_PROTECTED"
        ),
        "directory_serial_overlap_status": serial_status,
        "directory_serial_overlap_count": serial_overlap_count,
        "source_json_sha256": source_sha,
        "parser_source_sha256": parser_sha,
        "identity_parser_source_sha256": identity_parser_sha,
        "unresolved_metadata_limitations": limitations,
    }
    private_row = {
        "category_id": category,
        "source_json_sha256": source_sha,
        "source_group_digest_algorithm": SOURCE_GROUP_DIGEST_ALGORITHM,
        "group_members": [
            {
                "source_group_sha256": digest,
                "camera_view_count": group_views[digest],
            }
            for digest in candidate_groups
        ],
        "directory_serial_overlap_count": serial_overlap_count,
        "role_certification_status": role_status,
        "contains_raw_sample_keys": False,
        "contains_image_paths": False,
        "contains_labels_or_masks": False,
    }
    return public_row, private_row, projection.test_section_present


def assess_h_source_partition(
    category_id: str,
    a_candidate_group_sha256s: Sequence[str],
    role_evidence: Mapping[tuple[str, str], RoleEvidence],
    trust_directory: ReviewerTrustDirectory | None,
    expected_trust_directory_sha256: str | None,
) -> dict[str, Any]:
    """Certify A/B/C only from matching, independently verified role receipts."""
    candidate_set = set(a_candidate_group_sha256s)
    statuses: dict[str, str] = {}
    reasons: dict[str, str | None] = {}
    verified_groups: dict[str, set[str]] = {}
    for role in ROLE_NAMES:
        evidence = role_evidence.get((category_id, role))
        if evidence is None:
            statuses[role] = (
                "C_IDENTITY_CERTIFICATE_INCOMPLETE" if role == "C"
                else "PENDING_INDEPENDENT_ROLE_RECEIPT"
            )
            reasons[role] = "role evidence was not supplied"
            continue
        if evidence.receipt.category_id != category_id or evidence.receipt.role != role:
            statuses[role] = "PENDING_INDEPENDENT_ROLE_RECEIPT"
            reasons[role] = "receipt category or role does not match its assigned slot"
            continue
        try:
            state, reason = certify_role_identity(
                evidence.receipt,
                evidence.manifest,
                evidence.attestation,
                trust_directory,
                expected_trust_directory_sha256=expected_trust_directory_sha256,
            )
        except (KeyError, TypeError, ValueError):
            state, reason = "PENDING_INDEPENDENT_REVIEW", "role evidence or trust directory is malformed"
        statuses[role] = (
            "C_IDENTITY_CERTIFICATE_INCOMPLETE"
            if role == "C" and state != "CERTIFIED_FROM_SOURCE_EVIDENCE"
            else state
        )
        reasons[role] = reason
        if state == "CERTIFIED_FROM_SOURCE_EVIDENCE":
            verified_groups[role] = {member.group_sha256 for member in evidence.receipt.group_members}
    if statuses.get("A") == "CERTIFIED_FROM_SOURCE_EVIDENCE":
        if verified_groups.get("A", set()) != candidate_set:
            statuses["A"] = "A_CANDIDATE_MEMBERSHIP_MISMATCH"
            reasons["A"] = "reviewed A role membership differs from TRAIN-derived candidates"
            verified_groups.pop("A", None)
    if not candidate_set:
        partition = "NO_A_REFERENCE_CANDIDATES"
    elif any(status != "CERTIFIED_FROM_SOURCE_EVIDENCE" for status in statuses.values()):
        partition = "A_CANDIDATES_IDENTIFIED_PARTITION_NOT_CERTIFIED"
    else:
        intersection = (
            (verified_groups["A"] & verified_groups["B"])
            | (verified_groups["A"] & verified_groups["C"])
            | (verified_groups["B"] & verified_groups["C"])
        )
        partition = "SOURCE_GROUP_OVERLAP_REJECTED" if intersection else "SOURCE_GROUP_PARTITION_CERTIFIED"
    return {
        "partition_status": partition,
        "role_status": statuses,
        "role_reasons": reasons,
        "cross_role_overlap_count": len(intersection) if partition == "SOURCE_GROUP_OVERLAP_REJECTED" else 0,
        "c_identity_certificate_status": statuses.get("C", "C_IDENTITY_CERTIFICATE_INCOMPLETE"),
    }


def support_c64_pool_status(group_sha256s: Sequence[str]) -> dict[str, Any]:
    """Describe candidate-pool size without selecting Supports or C64."""
    values = tuple(group_sha256s)
    invalid = [value for value in values if not isinstance(value, str) or _HEX_SHA256_RE.fullmatch(value) is None]
    unique_count = len(set(value for value in values if isinstance(value, str)))
    if invalid:
        return {
            "candidate_pool_status": "INVALID_SOURCE_GROUP_DIGESTS",
            "distinct_candidate_group_count": unique_count,
            "at_least_68_legal_normal_groups": False,
            "support_and_c64_identity_status": "NOT_CERTIFIED",
        }
    if len(values) != unique_count:
        return {
            "candidate_pool_status": "DUPLICATE_SOURCE_GROUPS_REJECTED",
            "distinct_candidate_group_count": unique_count,
            "at_least_68_legal_normal_groups": unique_count >= 68,
            "support_and_c64_identity_status": "NOT_CERTIFIED",
        }
    return {
        "candidate_pool_status": (
            "CANDIDATE_POOL_SIZE_SUFFICIENT" if unique_count >= 64
            else "CANDIDATE_POOL_TOO_SMALL_FOR_C64"
        ),
        "distinct_candidate_group_count": unique_count,
        "at_least_68_legal_normal_groups": unique_count >= 68,
        "support_and_c64_identity_status": "PENDING_SEPARATE_SUPPORT_C64_SELECTION_AND_REVIEW",
    }


def _selection_payload(receipt: SupportC64SelectionReceipt) -> dict[str, Any]:
    payload = asdict(receipt)
    payload.pop("receipt_sha256")
    return payload


def _validate_private_receipt_fields(payload: Mapping[str, Any]) -> None:
    if set(payload) - {
        "schema", "authorization_id", "authorization_sha256",
        "source_group_digest_algorithm", "receipts",
    }:
        raise CensusContractError("private receipt contains an unapproved top-level field")
    for receipt in payload.get("receipts", ()):
        if set(receipt) - {
            "category_id", "source_json_sha256", "source_group_digest_algorithm",
            "group_members", "directory_serial_overlap_count", "role_certification_status",
            "contains_raw_sample_keys", "contains_image_paths", "contains_labels_or_masks",
        }:
            raise CensusContractError("private receipt contains an unapproved category field")
        for member in receipt.get("group_members", ()):
            if set(member) != {"source_group_sha256", "camera_view_count"}:
                raise CensusContractError("private group receipt contains an unapproved member field")
        role = receipt.get("role_certification_status", {})
        if set(role) != {
            "partition_status", "role_status", "role_reasons",
            "cross_role_overlap_count", "c_identity_certificate_status",
        }:
            raise CensusContractError("private role receipt contains an unapproved field")
        for name in ("role_status", "role_reasons"):
            if set(role[name]) != {"A", "B", "C"}:
                raise CensusContractError("private role receipt does not match the A/B/C contract")


def assess_support_c64_selection(
    candidate_group_sha256s: Sequence[str],
    receipt: SupportC64SelectionReceipt | None,
    attestation: SupportC64SelectionAttestation | None,
    trust_directory: ReviewerTrustDirectory | None,
    expected_trust_directory_sha256: str | None,
    *,
    expected_category_id: str,
    expected_support_seed: int,
) -> str:
    """Validate an externally reviewed exact four-support plus additional C64 set."""
    pool = support_c64_pool_status(candidate_group_sha256s)
    if pool["candidate_pool_status"] == "DUPLICATE_SOURCE_GROUPS_REJECTED":
        return "DUPLICATE_SOURCE_GROUPS_REJECTED"
    if pool["distinct_candidate_group_count"] < 64:
        return "CANDIDATE_POOL_TOO_SMALL_FOR_C64"
    if receipt is None or attestation is None:
        return "CANDIDATE_POOL_SIZE_SUFFICIENT"
    if pool["distinct_candidate_group_count"] < 68:
        return "CANDIDATE_POOL_INSUFFICIENT_FOR_4_SUPPORTS_PLUS_C64"
    if receipt.schema != "r2_l2c.support_c64_selection.v1":
        return "PENDING_SELECTION_RECEIPT"
    if (receipt.category_id != expected_category_id
            or isinstance(receipt.support_seed, bool)
            or not isinstance(receipt.support_seed, int)
            or isinstance(expected_support_seed, bool)
            or not isinstance(expected_support_seed, int)
            or receipt.support_seed != expected_support_seed):
        return "SUPPORT_CATEGORY_OR_SEED_MISMATCH"
    support = tuple(receipt.support_group_sha256s)
    c64 = tuple(receipt.c64_group_sha256s)
    if len(support) != 4 or len(c64) != 64:
        return "SUPPORT_C64_COUNTS_INVALID"
    if any(_HEX_SHA256_RE.fullmatch(group) is None for group in support + c64):
        return "INVALID_SOURCE_GROUP_DIGESTS"
    if len(set(support)) != 4 or len(set(c64)) != 64:
        return "DUPLICATE_SOURCE_GROUPS_REJECTED"
    if set(support).intersection(c64):
        return "SUPPORT_C64_OVERLAP_REJECTED"
    if not set(support + c64).issubset(set(candidate_group_sha256s)):
        return "SUPPORT_C64_GROUP_OUTSIDE_AUTHORIZED_NORMAL_POOL"
    if not receipt.producer_id.strip() or canonical_sha256(_selection_payload(receipt)) != receipt.receipt_sha256:
        return "PENDING_SELECTION_RECEIPT"
    try:
        trust_is_pinned = (
            trust_directory is not None
            and expected_trust_directory_sha256 is not None
            and reviewer_trust_directory_sha256(trust_directory) == trust_directory.directory_sha256
            and expected_trust_directory_sha256 == trust_directory.directory_sha256
        )
    except (KeyError, TypeError, ValueError, CensusContractError):
        trust_is_pinned = False
    if not trust_is_pinned:
        return "PENDING_EXTERNAL_SELECTION_TRUST_ROOT"
    if (attestation.reviewer_id == receipt.producer_id
            or attestation.reviewer_id not in trust_directory.trusted_reviewer_ids
            or attestation.receipt_sha256 != receipt.receipt_sha256
            or not attestation.decision_id.strip()):
        return "PENDING_INDEPENDENT_SELECTION_REVIEW"
    reviewer_keys = dict(trust_directory.trusted_reviewer_public_keys)
    reviewer_key = reviewer_keys.get(attestation.reviewer_id)
    if reviewer_key is None:
        return "PENDING_INDEPENDENT_SELECTION_REVIEW"
    signed_payload = {
        "reviewer_id": attestation.reviewer_id,
        "decision_id": attestation.decision_id,
        "receipt_sha256": attestation.receipt_sha256,
    }
    if not _verify_ed25519(
        reviewer_key,
        _canonical_bytes(signed_payload),
        attestation.signature_ed25519_hex,
    ):
        return "PENDING_INDEPENDENT_SELECTION_REVIEW"
    return "SUPPORT_AND_C64_IDENTITIES_CERTIFIED"


def _check_output_path(authorization: LocalAccessAuthorization) -> Path:
    target = Path(authorization.output_directory)
    if not target.is_absolute():
        raise CensusContractError("authorized output directory must be absolute")
    _reject_reparse_chain(target.parent)
    if target.exists() or target.is_symlink():
        raise CensusContractError("output directory already exists; historical output reuse is forbidden")
    dataset = Path(authorization.dataset_root)
    try:
        target_key = _normalized_path_key(str(target))
        dataset_key = _normalized_path_key(str(dataset))
    except CensusContractError:
        raise
    if target_key == dataset_key or target_key.startswith(dataset_key + "\\"):
        raise CensusContractError("output directory must be outside the dataset root")
    return target


class OutputBudget:
    """Exclusive, pre-write byte accountant for the two census output files."""

    def __init__(self, output_root: str | Path, maximum_bytes: int) -> None:
        if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes <= 0:
            raise OutputBudgetError("output byte budget must be a positive integer")
        self.root = Path(output_root)
        self.maximum_bytes = maximum_bytes
        self.lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=False)
        _reject_reparse_chain(self.root)
        try:
            self.lock_path = self.root / ".r2_l2c_writer.lock"
            self.lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise OutputBudgetError("concurrent or stale output writer detected") from exc
        self.allowed_files = {
            self.lock_path.name,
            "census_report.json",
            "private_source_group_receipts.json",
        }

    def _current_usage(self) -> int:
        total = 0
        for item in self.root.iterdir():
            if item.name not in self.allowed_files or _reparse_or_symlink(item) or not item.is_file():
                raise OutputBudgetError("unaccounted writer or non-regular output detected")
            if item.name != self.lock_path.name:
                total += item.stat().st_size
        return total

    def write_json(self, filename: str, value: Mapping[str, Any]) -> int:
        if filename not in {"census_report.json", "private_source_group_receipts.json"}:
            raise OutputBudgetError("output filename is outside the audited writer contract")
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        with self.lock:
            _reject_reparse_chain(self.root)
            current = self._current_usage()
            if current + len(encoded) > self.maximum_bytes:
                raise OutputBudgetError(
                    f"pre-write byte limit exceeded: {current}+{len(encoded)}>{self.maximum_bytes}")
            target = self.root / filename
            try:
                fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError as exc:
                raise OutputBudgetError("output writer refused to overwrite an existing file") from exc
            try:
                offset = 0
                while offset < len(encoded):
                    offset += os.write(fd, encoded[offset:])
                os.fsync(fd)
            finally:
                os.close(fd)
        return len(encoded)

    def close(self) -> None:
        try:
            os.close(self.lock_fd)
        except OSError:
            pass
        self.lock_path.unlink(missing_ok=True)


def execute_census(
    authorization: LocalAccessAuthorization | None,
    trust_directory: ReviewerTrustDirectory | None,
    expected_trust_directory_sha256: str | None,
    *,
    expected_dataset_root: str = DEFAULT_DATASET_ROOT,
) -> dict[str, Any]:
    """Execute only after complete signature and operation-scope validation."""
    errors = authorization_errors(
        authorization, trust_directory, expected_trust_directory_sha256,
        expected_dataset_root=expected_dataset_root,
    )
    if errors or authorization is None:
        return {
            "schema": SCHEMA,
            "status": "R2_L2C_LOCAL_ACCESS_NOT_AUTHORIZED",
            "local_data_io_performed": False,
            "authorization_errors": list(errors),
        }

    # Every authorization check above is complete before any dataset/output path
    # is probed. No directory enumeration is performed on the dataset root.
    try:
        output_root = _check_output_path(authorization)
        dataset_root = Path(authorization.dataset_root)
        metadata_root = Path(authorization.metadata_root)
        _reject_reparse_chain(dataset_root)
        _reject_reparse_chain(metadata_root)
        if not dataset_root.is_dir() or not metadata_root.is_dir():
            raise CensusContractError("authorized dataset or metadata root is unavailable")
        if not metadata_root.resolve(strict=True).is_relative_to(dataset_root.resolve(strict=True)):
            raise CensusContractError("metadata root resolves outside the authorized dataset root")
        budget = OutputBudget(output_root, authorization.maximum_output_bytes)
    except (OSError, CensusContractError, OutputBudgetError) as exc:
        return {
            "schema": SCHEMA,
            "status": "R2_L2C_BLOCKED",
            "local_data_io_performed": False,
            "authorization_errors": [],
            "execution_failure_type": type(exc).__name__,
        }

    parser_sha = _source_sha256(Path(__file__))
    identity_parser_sha = _source_sha256(Path(identity_contract.__file__))
    effective_categories = _authorized_categories(authorization)
    rows: list[dict[str, Any]] = []
    private_receipts: list[dict[str, Any]] = []
    opened_categories: list[str] = []
    test_section_presence: list[bool | None] = []
    for category in CANONICAL_CATEGORY_IDS:
        protected = category in PROTECTED_CATEGORIES
        if category not in effective_categories:
            status = "PROTECTED_NOT_AUTHORIZED" if protected else "NOT_AUTHORIZED"
            rows.append(_empty_category_row(
                category,
                status,
                "PROTECTED_NOT_AUTHORIZED" if protected else "NOT_PROTECTED",
            ))
            continue
        source = metadata_root / f"{category}.json"
        try:
            _reject_reparse_chain(source)
            if not source.is_file():
                row = _empty_category_row(category, "AUTHORIZED_SCOPE", "PROTECTED_ACCESS_AUTHORIZED" if protected else "NOT_PROTECTED")
                row["metadata_availability_status"] = "METADATA_FILE_MISSING"
                row["a_role_candidate_status"] = "A_REFERENCE_EVIDENCE_UNAVAILABLE"
                row["abc_partition_status"] = "A_CANDIDATES_NOT_IDENTIFIED"
                row["unresolved_metadata_limitations"] = [
                    "authorized category metadata file is missing; no zero count inferred",
                ]
                rows.append(row)
                continue
            if source.is_symlink() or not source.is_file():
                raise CensusContractError("category JSON is not a regular non-reparse file")
            with source.open("rb") as stream:
                source_bytes = stream.read()
            opened_categories.append(category)
            try:
                public_row, private_row, test_section_present = _category_census(
                    category,
                    source_bytes,
                    authorization,
                    parser_sha,
                    identity_parser_sha,
                    trust_directory,
                    expected_trust_directory_sha256,
                )
                test_section_presence.append(test_section_present)
            except ProjectionJSONError:
                public_row = _empty_category_row(
                    category,
                    "AUTHORIZED_SCOPE",
                    "PROTECTED_ACCESS_AUTHORIZED" if protected else "NOT_PROTECTED",
                )
                public_row["metadata_availability_status"] = "METADATA_SCHEMA_INVALID"
                public_row["unresolved_metadata_limitations"] = [
                    "TRAIN projection could not validate the metadata schema",
                ]
                private_row = None
                test_section_presence.append(None)
            if private_row is not None:
                private_receipts.append(private_row)
            rows.append(public_row)
        except (OSError, CensusContractError) as exc:
            row = _empty_category_row(
                category,
                "AUTHORIZED_SCOPE",
                "PROTECTED_ACCESS_AUTHORIZED" if protected else "NOT_PROTECTED",
            )
            row["metadata_availability_status"] = "METADATA_ACCESS_BLOCKED"
            row["unresolved_metadata_limitations"] = [
                f"authorized metadata read failed closed: {type(exc).__name__}",
            ]
            rows.append(row)

    missing_categories = [
        row["category_id"] for row in rows
        if row["metadata_availability_status"] != "METADATA_AVAILABLE"
    ]
    report = {
        "schema": "r2_l2c.metadata_census_report.v1",
        "status": "R2_L2C_METADATA_CENSUS_COMPLETE_PARTITION_PENDING",
        "operation": TRAIN_CENSUS_OPERATION,
        "authorization_id": authorization.authorization_id,
        "authorization_sha256": authorization.authorization_sha256,
        "source_group_digest_algorithm": SOURCE_GROUP_DIGEST_ALGORITHM,
        "identity_contract_version": identity_contract.IDENTITY_VERSION,
        "grouping_unit": identity_contract.PRIMARY_GROUPING_UNIT,
        "public_catalog_source_uri": PINNED_SOURCE_URI,
        "public_catalog_source_sha256": PINNED_SOURCE_SHA256,
        "target_category_count": len(CANONICAL_CATEGORY_IDS),
        "authorized_category_count": len(effective_categories),
        "metadata_files_opened": len(opened_categories),
        "metadata_available_category_count": sum(
            row["metadata_availability_status"] == "METADATA_AVAILABLE" for row in rows
        ),
        "missing_or_unavailable_categories": missing_categories,
        "partition_status": (
            "A_CANDIDATES_IDENTIFIED_PARTITION_NOT_CERTIFIED"
            if any(row["a_role_candidate_status"] == "A_CANDIDATES_IDENTIFIED" for row in rows)
            else "A_CANDIDATES_NOT_IDENTIFIED"
        ),
        "local_data_io_performed": True,
        "test_section_parsed": False,
        "test_section_bytes_encountered": (
            True if True in test_section_presence
            else None if None in test_section_presence
            else False
        ),
        "image_decoding_performed": False,
        "feature_extraction_performed": False,
        "geometry_performed": False,
        "categories": rows,
    }
    try:
        # Only allowlisted aggregate fields are emitted in the shareable report.
        for row in report["categories"]:
            for field_name in tuple(row):
                if field_name not in authorization.approved_output_fields:
                    row.pop(field_name)
        private_payload = {
            "schema": "r2_l2c.private_group_receipts.v1",
            "authorization_id": authorization.authorization_id,
            "authorization_sha256": authorization.authorization_sha256,
            "source_group_digest_algorithm": SOURCE_GROUP_DIGEST_ALGORITHM,
            "receipts": private_receipts,
        }
        if set(report) - set(authorization.approved_output_fields):
            raise CensusContractError("shareable report contains an unapproved top-level field")
        _validate_private_receipt_fields(private_payload)
        budget.write_json("private_source_group_receipts.json", private_payload)
        # The public report is the completion marker and is written last.
        budget.write_json("census_report.json", report)
    except (OSError, OutputBudgetError, CensusContractError) as exc:
        report["status"] = "R2_L2C_BLOCKED"
        report["execution_failure_type"] = type(exc).__name__
    finally:
        budget.close()
    return report


def _make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="R2-L2C default-deny TRAIN metadata census")
    parser.add_argument("--mode", choices=("preflight", "execute"), default="preflight")
    parser.add_argument("--access-manifest", type=Path)
    parser.add_argument("--trust-directory", type=Path)
    parser.add_argument("--expected-trust-directory-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _make_cli_parser().parse_args(argv)
    authorization = None
    trust_directory = None
    load_errors: list[str] = []
    if args.access_manifest is not None or args.trust_directory is not None:
        if args.access_manifest is None or args.trust_directory is None:
            load_errors.append("both --access-manifest and --trust-directory are required")
        else:
            try:
                authorization, trust_directory = load_authorization_and_trust(
                    args.access_manifest, args.trust_directory,
                )
            except (OSError, CensusContractError, json.JSONDecodeError) as exc:
                load_errors.append(f"authorization documents could not be validated: {type(exc).__name__}")
    if args.mode == "preflight":
        report = preflight(
            authorization,
            trust_directory,
            args.expected_trust_directory_sha256,
            expected_dataset_root=DEFAULT_DATASET_ROOT,
        )
        if load_errors:
            report["missing_approvals"].extend(load_errors)
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0
    if load_errors:
        report = {
            "schema": SCHEMA,
            "status": "R2_L2C_LOCAL_ACCESS_NOT_AUTHORIZED",
            "local_data_io_performed": False,
            "authorization_errors": load_errors,
        }
    else:
        report = execute_census(
            authorization,
            trust_directory,
            args.expected_trust_directory_sha256,
            expected_dataset_root=DEFAULT_DATASET_ROOT,
        )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["status"] in {
        "R2_L2C_EXECUTOR_READY_AWAITING_AUTHORIZATION",
        "R2_L2C_METADATA_CENSUS_COMPLETE_PARTITION_PENDING",
        "R2_L2C_METADATA_CENSUS_CERTIFIED",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
