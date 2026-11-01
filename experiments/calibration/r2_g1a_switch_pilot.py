"""G1-A switch image-to-descriptor bridge and read-only authorization preflight.

Importing this module does not access Real-IAD, load DINO, or initialize CUDA.
Image decoding is only reachable through ``encode_authorized_image`` after a
validated G1-A permit has been supplied by a separate preflight caller.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from experiments.calibration.r2_l2a_allclass_inventory import (
    AReferenceAvailabilityReceipt,
)
from experiments.calibration.r2_l2a_g1_geometry import (
    G1A_SAMPLING_POLICY,
    R1_DESCRIPTOR_PROFILE,
    DescriptorBatch,
    compare_source_disjoint,
    g1a_spatial_indices,
    summarize_category_geometry,
)
from experiments.calibration.r2_l2b_role_certification import (
    IndependentReviewerAttestation,
    ReviewedIdentityManifest,
    ReviewerTrustDirectory,
    RoleIdentityReceipt,
    SourceGroupMember,
    _ed25519_verify,
    canonical_sha256,
    certify_a_reference_availability,
    reviewer_trust_directory_sha256,
    source_group_digest_from_key,
)
from experiments.risk_safe.preflight import parse_image_name


G1A_AUTH_SCHEMA = "r2_g1a.switch_image_feature_authorization.v1"
G1A_OPERATION_ID = "R2_G1A_SWITCH_TRAIN_NORMAL_GEOMETRY_V1"
G1A_A_BUNDLE_SCHEMA = "r2_g1a.switch_a_reference_evidence_bundle.v1"
G1A_IMAGE_INVENTORY_SCHEMA = "r2_g1a.switch_a_image_identity_inventory.v1"
G1A_PROJECTED_RECEIPT_SCHEMA = "r2_g1a.projected_descriptor_receipt.v1"
G1A_SOURCE_FINGERPRINT_SCHEMA = "r2_g1a.source_fingerprint.v1"
G1A_SAMPLE_POLICY = "r2-g1-bounded-pilot-v1"
G1A_GROUP_DIGEST_ALGORITHM = "sha256:official-sample-group-nul-prefix-utf8.v1"
G1A_IDENTITY_CONTRACT = "realid-offical-sample-key-v2"
G1A_GROUPING_UNIT = "official_sample_key"
G1A_DEVELOPMENT_AUTHORIZATION_ID = "R2-G1A-SWITCH-DEV-RUN"
G1A_DEVELOPMENT_PARENT_COMMIT = "78e56c0820e77ee456e9c1729c3b50ece0c93a02"
G1A_DEVELOPMENT_RUN_SCHEMA = "r2_g1a.switch_development_run.v1"
G1A_DEVELOPMENT_WORKER_ENV = "R2_G1A_DEVELOPMENT_WORKER_NONCE"
DATASET_ROOT = r"G:\dataset\realiad"
METADATA_ROOT = r"G:\dataset\realiad\jsons\realiad_jsons"
OUTPUT_PARENT_RELATIVE = Path("outputs") / "calibration" / "r2_g1_pilot"
MAX_OUTPUT_BYTES = 1024**3
MIN_FREE_BYTES = 10 * 1024**3
MAX_WALL_SECONDS = 120 * 60
S1_GROUP_COUNT = 10
FAILURE_RESERVE_BYTES = 16 * 1024**2
MAX_CPU_THREADS = 4
MAX_CUDA_RESERVED_BYTES = 20 * 1024**3
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_G1A_PERMIT_SEAL = object()
_G1A_DEVELOPMENT_PERMIT_SEAL = object()


class G1APilotError(RuntimeError):
    """G1-A adapter, authorization, or preflight contract failed."""


class G1AAuthorizationError(G1APilotError):
    pass


class G1AOutputError(G1APilotError):
    pass


G1A_AUTHORIZATION_KEYS = frozenset({
    "schema", "operation_id", "authorization_id", "researcher_id", "decision",
    "dataset_root", "metadata_root", "category_id", "split", "population",
    "identity_contract", "grouping_unit", "group_digest_algorithm",
    "allowed_train_fields", "authorized_stages", "allow_train_identity_projection",
    "allow_image_decoding", "allow_dino_feature_extraction",
    "allow_projected_descriptor_output", "allow_projected_cache_reuse",
    "allow_g1a_geometry", "allow_test_metadata_parsing", "allow_test_labels",
    "allow_masks", "allow_support_selection", "allow_native_scoring",
    "allow_g1b", "allow_g2", "sample_policy", "sample_seed", "stage1_group_count",
    "maximum_source_groups", "camera_id", "sampling_projection_rule",
    "representation_sha256", "profile_source_commit", "execution_commit",
    "source_fingerprint_sha256", "trust_directory_sha256",
    "a_reference_receipt_sha256", "image_identity_receipt_sha256",
    "checkpoint_sha256", "hub_source_tree_sha256",
    "expected_loaded_model_state_sha256", "output_directory",
    "maximum_output_bytes", "minimum_free_bytes", "maximum_wall_seconds",
    "cpu_thread_limit", "cuda_device", "maximum_cuda_reserved_bytes",
    "signature_ed25519_hex", "authorization_sha256",
})

G1A_ALLOWED_TRAIN_FIELDS = (
    "image_path", "anomaly_class", "official_sample_key",
)
G1A_AUTH_BOOLEAN_FIELDS = (
    "allow_train_identity_projection", "allow_image_decoding",
    "allow_dino_feature_extraction", "allow_projected_descriptor_output",
    "allow_projected_cache_reuse", "allow_g1a_geometry",
    "allow_test_metadata_parsing", "allow_test_labels", "allow_masks",
    "allow_support_selection", "allow_native_scoring", "allow_g1b", "allow_g2",
)


def _sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise G1APilotError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _path_has_reparse_or_symlink(path: str | Path) -> bool:
    candidate = Path(os.path.abspath(os.fspath(path)))
    for node in (candidate, *candidate.parents):
        try:
            if node.is_symlink():
                return True
            info = node.lstat()
            if int(getattr(info, "st_file_attributes", 0)) & 0x400:
                return True
        except FileNotFoundError:
            continue
    return False


def read_json_object(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    try:
        from experiments.calibration.r2_l2c_metadata_census import _read_json_document
        value = _read_json_document(candidate)
    except Exception as exc:
        raise G1APilotError(f"cannot read G1 authorization evidence: {candidate.name}") from exc
    if not isinstance(value, dict):
        raise G1APilotError("authorization evidence must be one JSON object")
    return value


def authorization_payload(scope: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(scope)
    payload.pop("signature_ed25519_hex", None)
    payload.pop("authorization_sha256", None)
    return payload


def validate_authorization_scope(
    scope: Mapping[str, Any] | None,
    trust_directory: ReviewerTrustDirectory | None,
    *,
    expected_trust_directory_sha256: str | None,
    expected_source_fingerprint_sha256: str | None,
    actual_source_fingerprint_sha256: str | None,
    expected_execution_commit: str | None,
    repo_root: str | Path,
) -> tuple[str, ...]:
    """Verify a narrowly scoped signed operation without probing dataset paths."""
    if scope is None:
        return ("signed G1-A switch authorization is missing",)
    errors: list[str] = []
    if set(scope) != G1A_AUTHORIZATION_KEYS:
        errors.append("G1-A authorization fields do not match the frozen schema")
        return tuple(errors)

    if scope.get("schema") != G1A_AUTH_SCHEMA or scope.get("operation_id") != G1A_OPERATION_ID:
        errors.append("authorization does not name the frozen G1-A switch operation")
    if scope.get("decision") != "APPROVE_G1A_SWITCH_PILOT":
        errors.append("authorization decision does not approve the named G1-A operation")
    for field in ("authorization_id", "researcher_id"):
        if not isinstance(scope.get(field), str) or not scope[field].strip():
            errors.append(f"authorization {field} is missing")
    expected_scalars = {
        "dataset_root": DATASET_ROOT,
        "metadata_root": METADATA_ROOT,
        "category_id": "switch",
        "split": "TRAIN",
        "population": "authorized_normal_reference_only",
        "identity_contract": G1A_IDENTITY_CONTRACT,
        "grouping_unit": G1A_GROUPING_UNIT,
        "group_digest_algorithm": G1A_GROUP_DIGEST_ALGORITHM,
        "sample_policy": G1A_SAMPLE_POLICY,
        "sample_seed": 0,
        "stage1_group_count": S1_GROUP_COUNT,
        "maximum_source_groups": 100,
        "camera_id": 1,
        "sampling_projection_rule": G1A_SAMPLING_POLICY,
        "representation_sha256": R1_DESCRIPTOR_PROFILE.representation_sha256,
        "profile_source_commit": R1_DESCRIPTOR_PROFILE.source_commit,
        "checkpoint_sha256": R1_DESCRIPTOR_PROFILE.checkpoint_sha256,
        "hub_source_tree_sha256": R1_DESCRIPTOR_PROFILE.hub_source_tree_sha256,
        "expected_loaded_model_state_sha256": R1_DESCRIPTOR_PROFILE.loaded_model_state_sha256,
        "maximum_output_bytes": MAX_OUTPUT_BYTES,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "maximum_wall_seconds": MAX_WALL_SECONDS,
        "cpu_thread_limit": MAX_CPU_THREADS,
        "cuda_device": "cuda:0",
        "maximum_cuda_reserved_bytes": MAX_CUDA_RESERVED_BYTES,
    }
    for field, wanted in expected_scalars.items():
        if scope.get(field) != wanted or type(scope.get(field)) is not type(wanted):
            errors.append(f"authorization field {field} is incompatible")
    if scope.get("allowed_train_fields") != list(G1A_ALLOWED_TRAIN_FIELDS):
        errors.append("authorization TRAIN field allowlist is incompatible")
    if scope.get("authorized_stages") != ["S1", "S2"]:
        errors.append("authorization must explicitly cover the ordered S1 and S2 stages")
    for field in G1A_AUTH_BOOLEAN_FIELDS:
        expected = field not in {
            "allow_test_metadata_parsing", "allow_test_labels", "allow_masks",
            "allow_support_selection", "allow_native_scoring", "allow_g1b", "allow_g2",
        }
        if type(scope.get(field)) is not bool or scope.get(field) is not expected:
            errors.append(f"authorization permission {field} is not the locked value")

    for field in (
        "execution_commit", "source_fingerprint_sha256", "trust_directory_sha256",
        "a_reference_receipt_sha256", "image_identity_receipt_sha256",
        "signature_ed25519_hex", "authorization_sha256",
    ):
        if not isinstance(scope.get(field), str) or not scope[field].strip():
            errors.append(f"authorization field {field} is missing")
    try:
        if not isinstance(scope.get("execution_commit"), str) or _GIT_COMMIT_RE.fullmatch(
                scope.get("execution_commit", "")) is None:
            raise G1APilotError("execution commit must be a full lowercase Git object ID")
        _sha256(scope.get("source_fingerprint_sha256", ""), "source fingerprint")
        _sha256(scope.get("trust_directory_sha256", ""), "trust-directory pin")
        _sha256(scope.get("a_reference_receipt_sha256", ""), "A availability receipt")
        _sha256(scope.get("image_identity_receipt_sha256", ""), "A image identity receipt")
        _sha256(scope.get("authorization_sha256", ""), "authorization digest")
    except G1APilotError as exc:
        errors.append(str(exc))
    if expected_execution_commit is None or scope.get("execution_commit") != expected_execution_commit:
        errors.append("signed execution commit differs from the reviewed current commit")
    if (expected_source_fingerprint_sha256 is None
            or scope.get("source_fingerprint_sha256") != expected_source_fingerprint_sha256
            or actual_source_fingerprint_sha256 != expected_source_fingerprint_sha256):
        errors.append("independent and live source-fingerprint values are absent or mismatched")

    trust_sha = None
    if trust_directory is None or expected_trust_directory_sha256 is None:
        errors.append("fresh independently pinned G1 trust directory is absent")
    else:
        try:
            trust_sha = reviewer_trust_directory_sha256(trust_directory)
            if (trust_sha != trust_directory.directory_sha256
                    or trust_sha != expected_trust_directory_sha256
                    or scope.get("trust_directory_sha256") != expected_trust_directory_sha256):
                errors.append("G1 trust directory does not match the independent prospective pin")
            if not trust_directory.directory_id.startswith("r2-g1a-switch-access-"):
                errors.append("G1 authorization must use a fresh G1-specific trust-root identifier")
        except Exception:
            errors.append("G1 trust directory failed structural digest validation")
        keys = dict(trust_directory.trusted_policy_approver_public_keys)
        key = keys.get(str(scope.get("researcher_id", "")))
        if (scope.get("researcher_id") not in trust_directory.trusted_policy_approver_ids
                or key is None):
            errors.append("researcher is not an authorized policy approver in the G1 trust root")
        else:
            signature = scope.get("signature_ed25519_hex", "")
            if not isinstance(signature, str) or not _ed25519_verify(
                    key, authorization_payload(scope), signature):
                errors.append("G1-A authorization Ed25519 signature is invalid")
    if canonical_sha256(authorization_payload(scope)) != scope.get("authorization_sha256"):
        errors.append("G1-A authorization digest does not match its canonical payload")

    output_value = scope.get("output_directory")
    if not isinstance(output_value, str) or not output_value.strip():
        errors.append("unique output directory is missing")
    else:
        try:
            root = Path(repo_root).resolve(strict=True)
            expected_parent = Path(os.path.abspath(root / OUTPUT_PARENT_RELATIVE))
            output = Path(os.path.abspath(output_value))
            if _path_has_reparse_or_symlink(output):
                errors.append("output path contains a symlink or Windows reparse point")
            if output.parent != expected_parent:
                errors.append("output directory must be a direct child of the approved G1 pilot namespace")
            if output.exists():
                errors.append("output directory already exists; implicit resume/overwrite is forbidden")
            run_id = output.name
            if not re.fullmatch(r"r2_g1a_switch_[A-Za-z0-9_-]{6,64}", run_id):
                errors.append("output directory name does not contain a unique G1-A switch run ID")
        except OSError:
            errors.append("repository/output path cannot be resolved safely")
    return tuple(errors)


def _record_identity_error(record: "G1AImageIdentity", permit: "G1AExecutionPermit") -> None:
    if permit._authorization_seal is not _G1A_PERMIT_SEAL:
        raise G1AAuthorizationError("image operation requires a permit issued by signed-scope validation")
    if (record.category_id != "switch" or record.role != "A"
            or record.population != "authorized_normal_reference_only"
            or not record.train_normal_verified):
        raise G1AAuthorizationError("image identity is outside the authorized switch TRAIN-normal A population")
    if record.camera_id != 1:
        raise G1AAuthorizationError("image does not satisfy the approved camera-ID-1 rule")
    for value, label in ((record.source_group_sha256, "source-group identity"),
                         (record.image_sha256, "image identity")):
        _sha256(value, label)
    path = PureWindowsPath(record.image_path)
    root = PureWindowsPath(DATASET_ROOT)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise G1AAuthorizationError("image path is outside the authorized dataset root") from exc
    if not relative.parts or relative.parts[0].casefold() != "switch":
        raise G1AAuthorizationError("image path does not identify the authorized switch category")
    try:
        identity = parse_image_name(str(path), expected_class="switch")
    except Exception as exc:
        raise G1AAuthorizationError("image filename does not satisfy the frozen identity parser") from exc
    if identity.camera_id != 1 or record.camera_id != identity.camera_id:
        raise G1AAuthorizationError("camera-ID metadata disagrees with the official filename token")
    if (source_group_digest_from_key("switch", identity.official_sample_key)
            != record.source_group_sha256):
        raise G1AAuthorizationError("filename Source Group digest disagrees with the certified identity")
    image_identity_sha = hashlib.sha256(
        ("image-identity\0" + identity.image_id).encode("utf-8")
    ).hexdigest()
    if image_identity_sha != record.image_sha256:
        raise G1AAuthorizationError("filename image identity disagrees with the certified identity")
    if permit.category_id != record.category_id or permit.a_reference_receipt_sha256 == "":
        raise G1AAuthorizationError("execution permit does not bind this category and A receipt")
    identity_tuple = (record.image_sha256, record.source_group_sha256, record.camera_id)
    if identity_tuple not in permit.authorized_image_identities:
        raise G1AAuthorizationError("image identity is absent from the independently reviewed A image inventory")
    if record.source_group_sha256 not in permit.selected_source_group_sha256s:
        raise G1AAuthorizationError("Source Group was not selected by the frozen 100-group rank")


@dataclass(frozen=True)
class G1AImageIdentity:
    """One private input record; ``image_path`` must never be serialized."""

    category_id: str
    role: str
    population: str
    source_group_sha256: str
    image_sha256: str
    camera_id: int
    image_path: str
    train_normal_verified: bool


def project_switch_train_normal_json_bytes(source_bytes: bytes) -> dict[str, Any]:
    """Project only normal TRAIN image_path/anomaly_class plus meta identity fields.

    The TEST subtree is structurally skipped. Non-normal TRAIN image paths are
    syntax-checked but their string values are never decoded or retained.
    """
    from experiments.calibration.r2_l2c_metadata_census import (
        ProjectionJSONError,
        _ProjectionParser,
    )

    try:
        text_value = source_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise G1APilotError("switch metadata JSON is not valid UTF-8") from exc

    class _NormalOnlyProjectionParser(_ProjectionParser):
        def project_meta(self) -> tuple[dict[str, Any], bool]:
            self.index = 0
            self._expect("{")
            top_keys: set[str] = set()
            meta_value: dict[str, Any] | None = None
            test_present = False
            self._ws()
            while True:
                key = self._read_key()
                if key in top_keys:
                    raise ProjectionJSONError("duplicate top-level JSON key")
                top_keys.add(key)
                self._expect(":")
                if key == "meta":
                    meta_value = self._project_object(frozenset({"normal_class", "prefix"}))
                elif key == "test":
                    test_present = True
                    self._skip_value()
                elif key == "train":
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
            if (meta_value is None
                    or not isinstance(meta_value.get("normal_class"), str)
                    or not isinstance(meta_value.get("prefix"), str)):
                raise ProjectionJSONError("required meta.normal_class/prefix is missing")
            return meta_value, test_present

        def _project_normal_row(self, normal_class: str) -> tuple[dict[str, str] | None, bool]:
            self._expect("{")
            seen: set[str] = set()
            anomaly_class: str | None = None
            image_span: tuple[int, int] | None = None
            self._ws()
            while True:
                key = self._read_key()
                if key in seen:
                    raise ProjectionJSONError("duplicate key in TRAIN record")
                seen.add(key)
                self._expect(":")
                if key == "anomaly_class":
                    anomaly_class = self._read_selected_string()
                elif key == "image_path":
                    self._ws()
                    start = self.index
                    self._skip_value()
                    image_span = (start, self.index)
                else:
                    self._skip_value()
                self._ws()
                if self.index < len(self.text) and self.text[self.index] == ",":
                    self.index += 1
                    continue
                self._expect("}")
                break
            if not isinstance(anomaly_class, str) or not anomaly_class:
                raise ProjectionJSONError("TRAIN record lacks a class usable for normal-only projection")
            if anomaly_class != normal_class:
                return None, True
            if image_span is None:
                raise ProjectionJSONError("normal TRAIN record lacks image_path")
            try:
                image_path, end = json.JSONDecoder().raw_decode(self.text, image_span[0])
            except json.JSONDecodeError as exc:
                raise ProjectionJSONError("normal TRAIN image_path is malformed") from exc
            if end != image_span[1] or not isinstance(image_path, str) or not image_path:
                raise ProjectionJSONError("normal TRAIN image_path must be a nonempty string")
            return {"image_path": image_path, "anomaly_class": anomaly_class}, False

        def project_train_normals(self, normal_class: str) -> tuple[list[dict[str, str]], int, bool]:
            self.index = 0
            self._expect("{")
            top_keys: set[str] = set()
            normal_rows: list[dict[str, str]] | None = None
            test_present = False
            total_train_records = 0
            non_normal_train_records = 0
            self._ws()
            while True:
                key = self._read_key()
                if key in top_keys:
                    raise ProjectionJSONError("duplicate top-level JSON key")
                top_keys.add(key)
                self._expect(":")
                if key == "train":
                    self._expect("[")
                    normal_rows = []
                    self._ws()
                    if self.index < len(self.text) and self.text[self.index] == "]":
                        self.index += 1
                    else:
                        while True:
                            self._ws()
                            if self.index >= len(self.text) or self.text[self.index] != "{":
                                raise ProjectionJSONError("TRAIN records must be JSON objects")
                            row, non_normal = self._project_normal_row(normal_class)
                            total_train_records += 1
                            if non_normal:
                                non_normal_train_records += 1
                            elif row is not None:
                                normal_rows.append(row)
                            self._ws()
                            if self.index < len(self.text) and self.text[self.index] == ",":
                                self.index += 1
                                continue
                            self._expect("]")
                            break
                elif key == "test":
                    test_present = True
                    self._skip_value()
                elif key == "meta":
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
            if self.index != len(self.text) or normal_rows is None:
                raise ProjectionJSONError("required TRAIN section is missing or malformed")
            return normal_rows, total_train_records, test_present

    parser = _NormalOnlyProjectionParser(text_value)
    try:
        meta, meta_test_present = parser.project_meta()
        normal_rows, train_count, train_test_present = parser.project_train_normals(
            meta["normal_class"])
    except ProjectionJSONError as exc:
        raise G1APilotError(f"switch TRAIN-normal projection failed: {exc}") from exc
    return {
        "meta": meta,
        "normal_train_rows": normal_rows,
        "train_record_count": train_count,
        "non_normal_train_record_count": train_count - len(normal_rows),
        "test_section_present": bool(meta_test_present or train_test_present),
        "test_values_structurally_skipped": True,
    }


@dataclass(frozen=True)
class G1AExecutionPermit:
    authorization_id: str
    authorization_sha256: str
    category_id: str
    a_reference_receipt_sha256: str
    image_identity_receipt_sha256: str
    selected_source_group_sha256s: tuple[str, ...]
    authorized_image_identities: tuple[tuple[str, str, int], ...]
    s1_image_sha256s: tuple[str, ...]
    representation_sha256: str
    execution_commit: str
    output_directory: str
    max_output_bytes: int
    min_free_bytes: int
    max_wall_seconds: int
    max_cpu_threads: int
    max_cuda_reserved_bytes: int
    _authorization_seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class G1ADevelopmentPermit:
    """Separate researcher-authorized development permit, not third-party certification."""

    authorization_id: str
    task_scope_sha256: str
    source_fingerprint_sha256: str
    execution_commit: str
    selected_source_group_sha256s: tuple[str, ...]
    authorized_image_identities: tuple[tuple[str, str, int], ...]
    s1_image_sha256s: tuple[str, ...]
    output_directory: str
    max_output_bytes: int
    min_free_bytes: int
    max_wall_seconds: int
    max_cpu_threads: int
    max_cuda_reserved_bytes: int
    _development_seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class G1AImageIdentityMember:
    image_sha256: str
    source_group_sha256: str
    camera_id: int


@dataclass(frozen=True)
class G1AImageIdentityReceipt:
    schema: str
    category_id: str
    role: str
    population: str
    identity_contract: str
    group_digest_algorithm: str
    source_manifest_sha256: str
    a_reference_receipt_sha256: str
    producer_id: str
    image_members: tuple[G1AImageIdentityMember, ...]
    image_count: int
    contains_raw_sample_keys: bool
    contains_image_paths: bool
    contains_labels_or_masks: bool
    receipt_sha256: str


@dataclass(frozen=True)
class G1AImageIdentityAttestation:
    reviewer_id: str
    decision_id: str
    decision: str
    image_identity_receipt_sha256: str
    signature_ed25519_hex: str
    attestation_sha256: str


@dataclass(frozen=True)
class G1AProjectedDescriptorReceipt:
    schema: str
    category_id: str
    role: str
    population: str
    source_group_sha256: str
    image_sha256: str
    representation_sha256: str
    full_map_source_sha256: str
    projection_rule: str
    selected_indices_sha256: str
    projected_descriptors_sha256: str
    shape: tuple[int, int]
    dtype: str
    l2_normalized: bool
    contains_raw_sample_keys: bool
    contains_image_paths: bool
    contains_labels_or_masks: bool
    contains_private_feedback: bool
    contains_correction_anchors: bool
    receipt_sha256: str


def _array_digest(array: np.ndarray, domain: bytes) -> str:
    value = np.ascontiguousarray(array)
    return _sha256_bytes(
        domain + b"\0" + str((value.shape, value.dtype.str)).encode("ascii")
        + b"\0" + value.tobytes(order="C")
    )


def project_r1_g1a_features(
    full_map: Any,
    *,
    record: G1AImageIdentity,
) -> tuple[np.ndarray, G1AProjectedDescriptorReceipt]:
    """Validate one R1 full map, retain 49 fixed rows, and discard the rest."""
    if hasattr(full_map, "detach"):
        if str(getattr(full_map, "dtype", "")) != "torch.float32":
            raise G1APilotError("R1 feature map tensor must be float32 before CPU projection")
        array = np.asarray(full_map.detach().to(device="cpu").contiguous().numpy())
    else:
        array = np.asarray(full_map)
    if array.shape != R1_DESCRIPTOR_PROFILE.feature_shape:
        raise G1APilotError(
            f"R1 feature map must be {R1_DESCRIPTOR_PROFILE.feature_shape}, found {array.shape}")
    if array.dtype != np.dtype("float32"):
        raise G1APilotError("R1 feature map must be float32")
    if not np.isfinite(array).all():
        raise G1APilotError("R1 feature map contains a non-finite value")
    norms = np.linalg.norm(array.astype(np.float64), axis=1)
    if not np.all(np.abs(norms - 1.0) <= 1e-4):
        raise G1APilotError("R1 feature map rows are not L2-normalized")
    indices = g1a_spatial_indices()
    projected = np.ascontiguousarray(array[indices], dtype=np.float32)
    projected.flags.writeable = False
    full_sha = _array_digest(array, b"r2-g1a-r1-full-map-f32-v1")
    projected_sha = _array_digest(projected, b"r2-g1a-fixed49-projection-f32-v1")
    index_sha = _array_digest(indices.astype("<i8", copy=False), b"r2-g1a-fixed49-index-v1")
    receipt_payload = {
        "schema": G1A_PROJECTED_RECEIPT_SCHEMA,
        "category_id": record.category_id,
        "role": record.role,
        "population": record.population,
        "source_group_sha256": record.source_group_sha256,
        "image_sha256": record.image_sha256,
        "representation_sha256": R1_DESCRIPTOR_PROFILE.representation_sha256,
        "full_map_source_sha256": full_sha,
        "projection_rule": G1A_SAMPLING_POLICY,
        "selected_indices_sha256": index_sha,
        "projected_descriptors_sha256": projected_sha,
        "shape": list(projected.shape),
        "dtype": "float32",
        "l2_normalized": True,
        "contains_raw_sample_keys": False,
        "contains_image_paths": False,
        "contains_labels_or_masks": False,
        "contains_private_feedback": False,
        "contains_correction_anchors": False,
    }
    receipt = G1AProjectedDescriptorReceipt(
        schema=receipt_payload["schema"], category_id=record.category_id,
        role=record.role, population=record.population,
        source_group_sha256=record.source_group_sha256,
        image_sha256=record.image_sha256,
        representation_sha256=receipt_payload["representation_sha256"],
        full_map_source_sha256=full_sha, projection_rule=G1A_SAMPLING_POLICY,
        selected_indices_sha256=index_sha,
        projected_descriptors_sha256=projected_sha,
        shape=tuple(projected.shape), dtype="float32", l2_normalized=True,
        contains_raw_sample_keys=False, contains_image_paths=False,
        contains_labels_or_masks=False, contains_private_feedback=False,
        contains_correction_anchors=False,
        receipt_sha256=canonical_sha256(receipt_payload),
    )
    return projected, receipt


def validate_projected_receipt(
    receipt: G1AProjectedDescriptorReceipt,
    descriptors: np.ndarray,
    *,
    expected_record: G1AImageIdentity,
) -> None:
    if (receipt.schema != G1A_PROJECTED_RECEIPT_SCHEMA
            or receipt.projection_rule != G1A_SAMPLING_POLICY
            or receipt.representation_sha256 != R1_DESCRIPTOR_PROFILE.representation_sha256
            or receipt.category_id != expected_record.category_id
            or receipt.role != "A"
            or receipt.population != "authorized_normal_reference_only"
            or receipt.source_group_sha256 != expected_record.source_group_sha256
            or receipt.image_sha256 != expected_record.image_sha256
            or receipt.shape != (49, 4096) or receipt.dtype != "float32"
            or not receipt.l2_normalized or receipt.contains_raw_sample_keys
            or receipt.contains_image_paths or receipt.contains_labels_or_masks
            or receipt.contains_private_feedback or receipt.contains_correction_anchors):
        raise G1APilotError("projected descriptor receipt is incompatible with the G1-A record")
    expected_indices = _array_digest(
        g1a_spatial_indices().astype("<i8", copy=False), b"r2-g1a-fixed49-index-v1")
    if receipt.selected_indices_sha256 != expected_indices:
        raise G1APilotError("projected descriptor receipt binds another patch-index set")
    arr = np.asarray(descriptors)
    if arr.shape != (49, 4096) or arr.dtype != np.float32 or not np.isfinite(arr).all():
        raise G1APilotError("projected descriptor array is malformed")
    if _array_digest(arr, b"r2-g1a-fixed49-projection-f32-v1") != receipt.projected_descriptors_sha256:
        raise G1APilotError("projected descriptor bytes do not match their receipt")
    norms = np.linalg.norm(arr.astype(np.float64), axis=1)
    if not np.all(np.abs(norms - 1.0) <= 1e-4):
        raise G1APilotError("projected descriptors are not L2-normalized")
    payload = asdict(receipt)
    payload.pop("receipt_sha256")
    payload["shape"] = list(receipt.shape)
    if canonical_sha256(payload) != receipt.receipt_sha256:
        raise G1APilotError("projected descriptor receipt digest is invalid")


def compare_g1a_projected_cohort(
    records: Sequence[G1AImageIdentity],
    projected_descriptors: Sequence[np.ndarray],
    projected_receipts: Sequence[G1AProjectedDescriptorReceipt],
    *,
    permit: G1AExecutionPermit,
    stage: str,
) -> tuple[Any, dict[str, Any]]:
    """Run the existing exact geometry engine on a permit-matched G1-A cohort."""
    if stage not in {"S1", "S2"}:
        raise G1AAuthorizationError("G1-A geometry stage must be S1 or S2")
    if not (len(records) == len(projected_descriptors) == len(projected_receipts)):
        raise G1APilotError("record, descriptor, and projected-receipt counts differ")
    expected_count = S1_GROUP_COUNT if stage == "S1" else 100
    if len(records) != expected_count:
        raise G1APilotError(f"{stage} requires exactly {expected_count} ranked Source Groups")
    expected_identities = permit.authorized_image_identities[:expected_count]
    actual_identities = tuple(
        (record.image_sha256, record.source_group_sha256, record.camera_id)
        for record in records
    )
    if actual_identities != expected_identities:
        raise G1AAuthorizationError(f"{stage} cohort differs from the signed ranked A identity list")
    flat_descriptors: list[np.ndarray] = []
    all_groups: list[str] = []
    all_images: list[str] = []
    for record, descriptors, receipt in zip(records, projected_descriptors, projected_receipts):
        _record_identity_error(record, permit)
        values = np.asarray(descriptors)
        validate_projected_receipt(receipt, values, expected_record=record)
        flat_descriptors.append(values)
        all_groups.extend([record.source_group_sha256] * len(values))
        all_images.extend([record.image_sha256] * len(values))
    features = np.concatenate(flat_descriptors, axis=0)
    inventory_sha = canonical_sha256([
        {"source_group_sha256": record.source_group_sha256,
         "image_sha256": record.image_sha256}
        for record in records
    ])
    cache_sha = canonical_sha256([receipt.receipt_sha256 for receipt in projected_receipts])
    batch = DescriptorBatch(
        category_id="switch", features=features,
        source_group_sha256=tuple(all_groups), image_sha256=tuple(all_images),
        representation_sha256=R1_DESCRIPTOR_PROFILE.representation_sha256,
        role="A", access_status="AUTHORIZED_FOR_G1_REFERENCE_POPULATION",
        a_reference_availability_receipt_sha256=permit.a_reference_receipt_sha256,
        inventory_sha256=inventory_sha, cache_receipt_sha256=cache_sha,
    )
    stats = compare_source_disjoint(
        batch, batch,
        verified_a_reference_availability_receipts={
            "switch": permit.a_reference_receipt_sha256,
        },
    )
    summary = summarize_category_geometry(batch, batch, stats)
    summary["stage"] = stage
    summary["sampled_source_group_count"] = len(records)
    summary["sampled_image_count"] = len(records)
    summary["inventory_sha256"] = inventory_sha
    summary["projected_cache_sha256"] = cache_sha
    return stats, summary


def compare_g1a_development_cohort(
    records: Sequence[G1AImageIdentity],
    projected_descriptors: Sequence[np.ndarray],
    projected_receipts: Sequence[G1AProjectedDescriptorReceipt],
    *,
    permit: G1ADevelopmentPermit,
    identity_manifest_sha256: str,
    stage: str,
) -> tuple[Any, dict[str, Any]]:
    """Use the unchanged exact geometry engine under explicitly labelled dev scope."""
    if stage not in {"S1", "S2"}:
        raise G1AAuthorizationError("development geometry stage must be S1 or S2")
    if not (len(records) == len(projected_descriptors) == len(projected_receipts)):
        raise G1APilotError("development record/descriptor/receipt counts differ")
    expected_count = S1_GROUP_COUNT if stage == "S1" else 100
    expected_identities = permit.authorized_image_identities[:expected_count]
    identities = tuple((r.image_sha256, r.source_group_sha256, r.camera_id) for r in records)
    if len(records) != expected_count or identities != expected_identities:
        raise G1AAuthorizationError("development geometry cohort differs from frozen ranked identities")
    flat: list[np.ndarray] = []
    groups: list[str] = []
    images: list[str] = []
    for record, values, receipt in zip(records, projected_descriptors, projected_receipts):
        _validate_development_record(record, permit, require_file=False)
        array = np.asarray(values)
        validate_projected_receipt(receipt, array, expected_record=record)
        flat.append(array)
        groups.extend([record.source_group_sha256] * len(array))
        images.extend([record.image_sha256] * len(array))
    feature_matrix = np.concatenate(flat, axis=0)
    inventory_sha = canonical_sha256([
        {"source_group_sha256": record.source_group_sha256,
         "image_sha256": record.image_sha256}
        for record in records
    ])
    cache_sha = canonical_sha256([r.receipt_sha256 for r in projected_receipts])
    batch = DescriptorBatch(
        category_id="switch", features=feature_matrix,
        source_group_sha256=tuple(groups), image_sha256=tuple(images),
        representation_sha256=R1_DESCRIPTOR_PROFILE.representation_sha256,
        role="A", access_status="AUTHORIZED_FOR_G1_REFERENCE_POPULATION",
        a_reference_availability_receipt_sha256=identity_manifest_sha256,
        inventory_sha256=inventory_sha, cache_receipt_sha256=cache_sha,
    )
    # This digest is an internal TRAIN projection identity, not an independent
    # A-role certificate. The result is always labelled DEVELOPMENT_ONLY below.
    stats = compare_source_disjoint(
        batch, batch,
        verified_a_reference_availability_receipts={
            "switch": identity_manifest_sha256,
        },
    )
    summary = summarize_category_geometry(batch, batch, stats)
    summary.update({
        "stage": stage,
        "status": "DEVELOPMENT_ONLY_DESCRIPTIVE_NORMAL_GEOMETRY",
        "access_status": "DEVELOPMENT_ONLY_RESEARCHER_AUTHORIZED",
        "independent_a_role_certification": "NOT_CLAIMED",
        "independent_cache_review": "NOT_CLAIMED",
        "authorization_id": permit.authorization_id,
        "task_scope_sha256": permit.task_scope_sha256,
        "identity_manifest_sha256": identity_manifest_sha256,
        "sampled_source_group_count": len(records),
        "sampled_image_count": len(records),
        "inventory_sha256": inventory_sha,
        "projected_cache_sha256": cache_sha,
    })
    return stats, summary


def select_bounded_groups(
    group_sha256s: Sequence[str],
    *,
    category_id: str = "switch",
    profile_id: str = "G1-FULL-A",
    sample_seed: int = 0,
    maximum_groups: int = 100,
) -> tuple[str, ...]:
    """Deterministic, outcome-blind rank selection from certified A group digests."""
    if category_id != "switch" or profile_id != "G1-FULL-A":
        raise G1APilotError("the signed first pilot is restricted to switch G1-FULL-A")
    if sample_seed != 0 or maximum_groups != 100:
        raise G1APilotError("group sample seed/limit differs from the approved first pilot")
    groups = tuple(group_sha256s)
    if not groups or len(groups) != len(set(groups)):
        raise G1APilotError("A group receipt is empty or contains duplicate identities")
    if len(groups) < maximum_groups:
        raise G1APilotError("certified switch A population is smaller than the frozen 100-group pilot")
    for group in groups:
        _sha256(group, "A Source Group digest")
    ranked = sorted(
        groups,
        key=lambda group: (
            hashlib.sha256(
                f"{G1A_SAMPLE_POLICY}\0{profile_id}\0{category_id}\0{sample_seed}\0{group}"
                .encode("utf-8")
            ).hexdigest(),
            group,
        ),
    )
    return tuple(ranked[:maximum_groups])


def build_development_identity_manifest(
    projection: Mapping[str, Any],
    *,
    metadata_source_sha256: str,
    dataset_root: str | Path = DATASET_ROOT,
) -> tuple[tuple[G1AImageIdentity, ...], dict[str, Any], str, dict[str, int]]:
    """Freeze exactly 100 normal TRAIN camera-1 identities before any image read."""
    from experiments.calibration.r2_l2c_metadata_census import _image_identity_path

    _sha256(metadata_source_sha256, "switch metadata source digest")
    meta = projection.get("meta")
    rows = projection.get("normal_train_rows")
    if not isinstance(meta, Mapping) or not isinstance(rows, list):
        raise G1APilotError("switch TRAIN-normal projection has an invalid shape")
    normal_class = meta.get("normal_class")
    prefix = meta.get("prefix")
    if not isinstance(normal_class, str) or not isinstance(prefix, str):
        raise G1APilotError("switch metadata lacks normal_class or prefix")
    root = Path(dataset_root).resolve(strict=True)
    image_by_group: dict[str, dict[int, tuple[G1AImageIdentity, str]]] = {}
    seen_images: set[str] = set()
    group_sample_key: dict[str, str] = {}
    for row in rows:
        image_path_value = row.get("image_path")
        anomaly_class = row.get("anomaly_class")
        if not isinstance(image_path_value, str) or anomaly_class != normal_class:
            raise G1APilotError("normal TRAIN projection contains an unverified identity row")
        try:
            identity_path = _image_identity_path(prefix, image_path_value)
            parsed = parse_image_name(identity_path, expected_class="switch")
        except Exception as exc:
            raise G1APilotError("normal TRAIN image identity fails the frozen parser") from exc
        if parsed.capture_session != "OK":
            raise G1APilotError("normal_class disagrees with the official OK identity token")
        group_sha = source_group_digest_from_key("switch", parsed.official_sample_key)
        image_sha = hashlib.sha256(
            ("image-identity\0" + parsed.image_id).encode("utf-8")).hexdigest()
        if image_sha in seen_images:
            raise G1APilotError("switch normal TRAIN projection contains a duplicate image identity")
        seen_images.add(image_sha)
        previous_key = group_sample_key.setdefault(group_sha, parsed.official_sample_key)
        if previous_key != parsed.official_sample_key:
            raise G1APilotError("Source Group digest collision across official sample keys")
        relative = PureWindowsPath(identity_path.replace("/", "\\"))
        image_full_path = root.joinpath(*relative.parts)
        absolute = Path(os.path.abspath(image_full_path))
        try:
            absolute.relative_to(root)
        except ValueError as exc:
            raise G1APilotError("projected TRAIN image path escapes the Real-IAD root") from exc
        record = G1AImageIdentity(
            category_id="switch", role="A",
            population="authorized_normal_reference_only",
            source_group_sha256=group_sha, image_sha256=image_sha,
            camera_id=parsed.camera_id, image_path=str(absolute),
            train_normal_verified=True,
        )
        camera_map = image_by_group.setdefault(group_sha, {})
        if parsed.camera_id in camera_map:
            raise G1APilotError("one normal Source Group has duplicate camera identities")
        camera_map[parsed.camera_id] = (record, parsed.official_sample_key)

    eligible_groups = tuple(
        group for group, camera_map in image_by_group.items() if 1 in camera_map)
    if len(eligible_groups) < 100:
        raise G1APilotError(
            f"switch A candidate pool has only {len(eligible_groups)} camera-1 groups; 100 required")
    ranked_groups = select_bounded_groups(eligible_groups)
    selected: list[G1AImageIdentity] = []
    private_rows: list[dict[str, Any]] = []
    for rank, group in enumerate(ranked_groups, start=1):
        record, sample_key = image_by_group[group][1]
        selected.append(record)
        private_rows.append({
            "rank": rank,
            "official_sample_key": sample_key,
            "source_group_sha256": record.source_group_sha256,
            "image_sha256": record.image_sha256,
            "camera_id": record.camera_id,
            "image_path": record.image_path,
        })
    file_check_permit = G1ADevelopmentPermit(
        authorization_id=G1A_DEVELOPMENT_AUTHORIZATION_ID,
        task_scope_sha256="0" * 64,
        source_fingerprint_sha256="0" * 64,
        execution_commit="0" * 40,
        selected_source_group_sha256s=ranked_groups,
        authorized_image_identities=tuple(
            (r.image_sha256, r.source_group_sha256, r.camera_id) for r in selected),
        s1_image_sha256s=tuple(r.image_sha256 for r in selected[:S1_GROUP_COUNT]),
        output_directory="", max_output_bytes=MAX_OUTPUT_BYTES,
        min_free_bytes=MIN_FREE_BYTES, max_wall_seconds=MAX_WALL_SECONDS,
        max_cpu_threads=MAX_CPU_THREADS,
        max_cuda_reserved_bytes=MAX_CUDA_RESERVED_BYTES,
        _development_seal=_G1A_DEVELOPMENT_PERMIT_SEAL,
    )
    for record in selected:
        _validate_development_record(record, file_check_permit, require_file=True)
    counts = {
        "normal_train_image_count": len(rows),
        "normal_train_source_group_count": len(image_by_group),
        "camera1_eligible_source_group_count": len(eligible_groups),
        "source_groups_without_camera1_count": len(image_by_group) - len(eligible_groups),
        "selected_source_group_count": len(selected),
        "selected_image_count": len(selected),
        "non_normal_train_record_count": int(projection["non_normal_train_record_count"]),
        "train_record_count": int(projection["train_record_count"]),
    }
    private_manifest: dict[str, Any] = {
        "schema": "r2_g1a.switch_development_private_identity_manifest.v1",
        "category_id": "switch",
        "split": "TRAIN",
        "population": "verified_normal_only",
        "metadata_source_sha256": metadata_source_sha256,
        "identity_contract": G1A_IDENTITY_CONTRACT,
        "group_digest_algorithm": G1A_GROUP_DIGEST_ALGORITHM,
        "sample_policy": G1A_SAMPLE_POLICY,
        "sample_seed": 0,
        "camera_id": 1,
        "counts": counts,
        "test_section_present": bool(projection.get("test_section_present")),
        "test_values_structurally_skipped": bool(
            projection.get("test_values_structurally_skipped")),
        "test_record_count_computed": False,
        "selected_identities": private_rows,
    }
    identity_sha = canonical_sha256(private_manifest)
    return tuple(selected), private_manifest, identity_sha, counts


def make_development_permit(
    *,
    output_directory: str | Path,
    execution_commit: str,
    source_fingerprint_sha256: str,
    identity_manifest_sha256: str,
    records: Sequence[G1AImageIdentity],
) -> G1ADevelopmentPermit:
    if (len(records) != 100 or len({r.source_group_sha256 for r in records}) != 100
            or len({r.image_sha256 for r in records}) != 100):
        raise G1AAuthorizationError("development permit requires the frozen 100-group identity manifest")
    if _GIT_COMMIT_RE.fullmatch(execution_commit) is None:
        raise G1AAuthorizationError("development permit requires a committed source identity")
    _sha256(source_fingerprint_sha256, "source fingerprint")
    _sha256(identity_manifest_sha256, "private identity-manifest digest")
    scope = {
        "schema": G1A_DEVELOPMENT_RUN_SCHEMA,
        "authorization_id": G1A_DEVELOPMENT_AUTHORIZATION_ID,
        "authorization_basis": "explicit researcher instruction for switch development-only pilot",
        "branch": "calibration-stage0-contract-20261029",
        "parent_commit": G1A_DEVELOPMENT_PARENT_COMMIT,
        "execution_commit": execution_commit,
        "source_fingerprint_sha256": source_fingerprint_sha256,
        "identity_manifest_sha256": identity_manifest_sha256,
        "dataset_root": DATASET_ROOT,
        "category_id": "switch",
        "split": "TRAIN",
        "population": "verified_normal_only",
        "stages": ["S1", "S2"],
        "maximum_images": 100,
        "sample_policy": G1A_SAMPLE_POLICY,
        "sample_seed": 0,
        "camera_id": 1,
        "representation_sha256": R1_DESCRIPTOR_PROFILE.representation_sha256,
        "output_max_bytes": MAX_OUTPUT_BYTES,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "maximum_wall_seconds": MAX_WALL_SECONDS,
        "cpu_thread_limit": MAX_CPU_THREADS,
        "cuda_device": "cuda:0",
        "maximum_cuda_reserved_bytes": MAX_CUDA_RESERVED_BYTES,
        "forbidden": ["TEST records", "labels", "masks", "other categories",
                       "Mint", "PCB", "MVTec AD 2", "G1-B", "G2",
                       "Support/C64", "radius optimization"],
        "third_party_certification": False,
    }
    task_scope_sha = canonical_sha256(scope)
    return G1ADevelopmentPermit(
        authorization_id=G1A_DEVELOPMENT_AUTHORIZATION_ID,
        task_scope_sha256=task_scope_sha,
        source_fingerprint_sha256=source_fingerprint_sha256,
        execution_commit=execution_commit,
        selected_source_group_sha256s=tuple(r.source_group_sha256 for r in records),
        authorized_image_identities=tuple(
            (r.image_sha256, r.source_group_sha256, r.camera_id) for r in records),
        s1_image_sha256s=tuple(r.image_sha256 for r in records[:S1_GROUP_COUNT]),
        output_directory=str(output_directory),
        max_output_bytes=MAX_OUTPUT_BYTES,
        min_free_bytes=MIN_FREE_BYTES,
        max_wall_seconds=MAX_WALL_SECONDS,
        max_cpu_threads=MAX_CPU_THREADS,
        max_cuda_reserved_bytes=MAX_CUDA_RESERVED_BYTES,
        _development_seal=_G1A_DEVELOPMENT_PERMIT_SEAL,
    )


def verify_file_sha256(path: str | Path, expected_sha256: str) -> str:
    """Hash a model artifact without loading it into DINO or PyTorch."""
    expected = _sha256(expected_sha256, "expected file SHA-256")
    candidate = Path(path)
    if not candidate.is_file():
        raise G1APilotError(f"required fingerprinted file is missing: {candidate.name}")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise G1APilotError(f"file fingerprint mismatch: {candidate.name}")
    return actual


def verify_profile_source_hashes(repo_root: str | Path) -> dict[str, dict[str, str]]:
    """Match frozen profile hashes using canonical LF bytes and retain raw hashes."""
    root = Path(repo_root)
    checked: dict[str, dict[str, str]] = {}
    for relative, expected in R1_DESCRIPTOR_PROFILE.source_hashes:
        path = root / relative
        if not path.is_file():
            raise G1APilotError(f"frozen R1 source is missing: {relative}")
        raw = path.read_bytes()
        try:
            canonical_lf = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        except UnicodeDecodeError as exc:
            raise G1APilotError(f"frozen R1 source is not UTF-8: {relative}") from exc
        canonical_sha = _sha256_bytes(canonical_lf)
        if canonical_sha != expected:
            raise G1APilotError(f"frozen R1 source content differs from its profile: {relative}")
        checked[relative] = {
            "canonical_lf_sha256": canonical_sha,
            "working_tree_bytes_sha256": _sha256_bytes(raw),
        }
    return checked


G1A_SOURCE_FILES = tuple(sorted(set(
    [relative for relative, _expected in R1_DESCRIPTOR_PROFILE.source_hashes]
    + [
        "experiments/calibration/r2_g1a_switch_pilot.py",
        "experiments/calibration/r2_l2a_g1_geometry.py",
        "experiments/calibration/r2_l2b_role_certification.py",
        "experiments/calibration/r2_l2c_metadata_census.py",
        "experiments/calibration/r1_runtime_supervisor.py",
        "experiments/risk_safe/preflight.py",
    ]
)))


def _hub_source_fingerprint(source_root: str | Path) -> tuple[str, int]:
    root = Path(source_root)
    if not root.is_dir() or not (root / "hubconf.py").is_file():
        raise G1APilotError("cached DINOv2 hub source is missing")
    files = []
    for path in sorted(root.rglob("*.py")):
        if not path.is_file():
            continue
        files.append((path.relative_to(root).as_posix(), _sha256_bytes(path.read_bytes())))
    digest = canonical_sha256([[name, value] for name, value in files])
    return digest, len(files)


def compute_source_fingerprint(
    repo_root: str | Path,
    *,
    checkpoint_path: str | Path,
    hub_source_root: str | Path,
    execution_commit: str | None = None,
) -> dict[str, Any]:
    """Build the independently pinnable G1 source/model artifact identity.

    This hashes source and model files only; it does not instantiate the model
    or inspect dataset paths. The loaded state is verified after authorized
    model initialization and before any image decode.
    """
    root = Path(repo_root).resolve(strict=True)
    checked_profile = verify_profile_source_hashes(root)
    file_hashes: dict[str, dict[str, str]] = {}
    for relative in G1A_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise G1APilotError(f"G1 source dependency is missing: {relative}")
        raw = path.read_bytes()
        file_hashes[relative] = {
            "working_tree_bytes_sha256": _sha256_bytes(raw),
            "canonical_lf_sha256": _sha256_bytes(
                raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")),
        }
    if execution_commit is None:
        try:
            execution_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise G1APilotError("cannot establish the exact G1 execution commit") from exc
    if not isinstance(execution_commit, str) or _GIT_COMMIT_RE.fullmatch(execution_commit) is None:
        raise G1APilotError("execution commit must be a full lowercase Git object ID")
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", *G1A_SOURCE_FILES],
            cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise G1APilotError("cannot verify scoped G1 source-tree cleanliness") from exc
    if dirty:
        raise G1APilotError("scoped G1 source dependencies have uncommitted changes")
    checkpoint_sha = verify_file_sha256(
        checkpoint_path, R1_DESCRIPTOR_PROFILE.checkpoint_sha256)
    hub_sha, hub_file_count = _hub_source_fingerprint(hub_source_root)
    if hub_sha != R1_DESCRIPTOR_PROFILE.hub_source_tree_sha256:
        raise G1APilotError("cached DINOv2 hub source differs from the frozen R1 profile")
    payload = {
        "schema": G1A_SOURCE_FINGERPRINT_SCHEMA,
        "execution_commit": execution_commit,
        "r1_profile_source_commit": R1_DESCRIPTOR_PROFILE.source_commit,
        "r1_profile_source_hashes": checked_profile,
        "g1_source_files": file_hashes,
        "representation_sha256": R1_DESCRIPTOR_PROFILE.representation_sha256,
        "encoder_fingerprint_sha256": R1_DESCRIPTOR_PROFILE.encoder_fingerprint_sha256,
        "preprocess_fingerprint_sha256": R1_DESCRIPTOR_PROFILE.preprocess_fingerprint_sha256,
        "native_scorer_profile_sha256": R1_DESCRIPTOR_PROFILE.native_scorer_profile_sha256,
        "checkpoint_sha256": checkpoint_sha,
        "hub_source_tree_sha256": hub_sha,
        "hub_source_file_count": hub_file_count,
        "expected_loaded_model_state_sha256": R1_DESCRIPTOR_PROFILE.loaded_model_state_sha256,
        "loaded_model_state_verified": False,
    }
    return {"payload": payload, "source_fingerprint_sha256": canonical_sha256(payload)}


def verify_loaded_model_state(backbone: Any) -> str:
    """Verify a loaded model only after authorization and before image decode."""
    try:
        import torch
    except Exception as exc:
        raise G1APilotError("PyTorch is unavailable for the authorized model-state check") from exc
    model = getattr(backbone, "model", None)
    if model is None or not hasattr(model, "state_dict"):
        raise G1APilotError("R1 backbone has no model state to verify")
    digest = hashlib.sha256()
    try:
        for name, tensor in sorted(model.state_dict().items()):
            value = tensor.detach().to("cpu").contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str((tuple(value.shape), str(value.dtype))).encode("ascii"))
            digest.update(value.numpy().tobytes())
    except Exception as exc:
        raise G1APilotError("loaded R1 model state is incomplete or unreadable") from exc
    actual = digest.hexdigest()
    if actual != R1_DESCRIPTOR_PROFILE.loaded_model_state_sha256:
        raise G1APilotError("loaded model state differs from the frozen R1 profile")
    return actual


def apply_g1_resource_caps(torch_module: Any, *, total_cuda_bytes: int,
                           device_index: int = 0) -> float:
    """Apply signed CPU/GPU limits before constructing the DINO backbone."""
    if (isinstance(total_cuda_bytes, bool) or not isinstance(total_cuda_bytes, int)
            or total_cuda_bytes < MAX_CUDA_RESERVED_BYTES):
        raise G1APilotError("CUDA device is absent or smaller than the signed 20-GiB cap")
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index != 0:
        raise G1APilotError("the signed pilot is limited to cuda:0")
    try:
        torch_module.set_num_threads(MAX_CPU_THREADS)
        torch_module.set_num_interop_threads(1)
        if not torch_module.cuda.is_available():
            raise G1APilotError("signed G1-A pilot requires the verified CUDA device")
        fraction = MAX_CUDA_RESERVED_BYTES / total_cuda_bytes
        torch_module.cuda.set_per_process_memory_fraction(fraction, device=device_index)
    except G1APilotError:
        raise
    except Exception as exc:
        raise G1APilotError("cannot apply the signed CPU/GPU resource caps") from exc
    return fraction


def _validate_development_record(
    record: G1AImageIdentity,
    permit: G1ADevelopmentPermit,
    *,
    require_file: bool,
) -> None:
    if permit._development_seal is not _G1A_DEVELOPMENT_PERMIT_SEAL:
        raise G1AAuthorizationError("development image access requires the scoped development permit")
    if (permit.authorization_id != G1A_DEVELOPMENT_AUTHORIZATION_ID
            or record.category_id != "switch" or record.role != "A"
            or record.population != "authorized_normal_reference_only"
            or not record.train_normal_verified):
        raise G1AAuthorizationError("record is outside the explicit switch TRAIN-normal development scope")
    if record.camera_id != 1:
        raise G1AAuthorizationError("development pilot permits camera ID 1 only")
    _sha256(record.source_group_sha256, "development Source Group digest")
    _sha256(record.image_sha256, "development image identity digest")
    path = PureWindowsPath(record.image_path)
    root = PureWindowsPath(DATASET_ROOT)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise G1AAuthorizationError("development image is outside the authorized dataset root") from exc
    if not relative.parts or relative.parts[0].casefold() != "images":
        raise G1AAuthorizationError("development image path does not use the frozen images/ prefix")
    try:
        parsed = parse_image_name(str(path), expected_class="switch")
    except Exception as exc:
        raise G1AAuthorizationError("development image filename violates the frozen identity parser") from exc
    if parsed.camera_id != record.camera_id or parsed.capture_session != "OK":
        raise G1AAuthorizationError("normal TRAIN metadata and camera token disagree")
    if source_group_digest_from_key("switch", parsed.official_sample_key) != record.source_group_sha256:
        raise G1AAuthorizationError("development filename Source Group digest does not match")
    image_sha = hashlib.sha256(
        ("image-identity\0" + parsed.image_id).encode("utf-8")).hexdigest()
    if image_sha != record.image_sha256:
        raise G1AAuthorizationError("development filename image identity digest does not match")
    identity = (record.image_sha256, record.source_group_sha256, record.camera_id)
    if (identity not in permit.authorized_image_identities
            or record.source_group_sha256 not in permit.selected_source_group_sha256s):
        raise G1AAuthorizationError("development image is absent from the frozen selected identity manifest")
    if require_file:
        file_path = Path(record.image_path)
        if _path_has_reparse_or_symlink(file_path):
            raise G1AAuthorizationError("development image path contains a symlink or reparse point")
        if not file_path.is_file():
            raise G1AAuthorizationError("selected development TRAIN image is missing")


def _encode_r1_projected_image(
    record: G1AImageIdentity,
    *,
    implementation: Any,
    backbone: Any,
) -> tuple[np.ndarray, G1AProjectedDescriptorReceipt]:
    image = None
    feature_layers = None
    full_map = None
    try:
        image = implementation.load_rgb(record.image_path)
        feature_layers, _global_feature = implementation.encode_pil(
            backbone, image, mode="concat")
        if not isinstance(feature_layers, (list, tuple)) or len(feature_layers) != 1:
            raise G1APilotError("frozen R1 concat path must emit exactly one fused feature matrix")
        full_map = feature_layers[0]
        projected, receipt = project_r1_g1a_features(full_map, record=record)
        validate_projected_receipt(receipt, projected, expected_record=record)
        return projected, receipt
    finally:
        if image is not None and hasattr(image, "close"):
            image.close()
        if hasattr(backbone, "_clear_feats"):
            backbone._clear_feats()
        del full_map
        del feature_layers
        if "_global_feature" in locals():
            del _global_feature
        gc.collect()


def encode_authorized_image(
    record: G1AImageIdentity,
    *,
    permit: G1AExecutionPermit | None,
    implementation: Any,
    backbone: Any,
    stage: str = "S1",
) -> tuple[np.ndarray, G1AProjectedDescriptorReceipt]:
    """Encode one image on the exact R1 path; this is not batch/cache capable."""
    if permit is None:
        raise G1AAuthorizationError("image decode refused: no validated G1-A execution permit")
    if permit.category_id != "switch" or permit.representation_sha256 != R1_DESCRIPTOR_PROFILE.representation_sha256:
        raise G1AAuthorizationError("execution permit has an incompatible category or representation")
    if record.camera_id != 1 or not record.train_normal_verified:
        raise G1AAuthorizationError("image decode refused: record is not verified switch TRAIN-normal camera 1")
    if stage not in {"S1", "S2"}:
        raise G1AAuthorizationError("stage is outside the signed S1/S2 scope")
    _record_identity_error(record, permit)
    if stage == "S1" and record.image_sha256 not in permit.s1_image_sha256s:
        raise G1AAuthorizationError("S1 smoke scope is limited to the first 10 ranked groups")
    return _encode_r1_projected_image(record, implementation=implementation, backbone=backbone)


def encode_development_image(
    record: G1AImageIdentity,
    *,
    permit: G1ADevelopmentPermit,
    implementation: Any,
    backbone: Any,
    stage: str,
) -> tuple[np.ndarray, G1AProjectedDescriptorReceipt]:
    """Development-only image adapter; strict certification validation remains separate."""
    if stage not in {"S1", "S2"}:
        raise G1AAuthorizationError("development stage is outside S1/S2")
    _validate_development_record(record, permit, require_file=True)
    if stage == "S1" and record.image_sha256 not in permit.s1_image_sha256s:
        raise G1AAuthorizationError("S1 smoke scope is limited to the first 10 ranked groups")
    return _encode_r1_projected_image(record, implementation=implementation, backbone=backbone)


def validate_a_evidence_bundle(
    bundle: Mapping[str, Any] | None,
    trust_directory: ReviewerTrustDirectory | None,
    *,
    expected_trust_directory_sha256: str | None,
) -> tuple[str, str | None, str | None, tuple[G1AImageIdentityMember, ...]]:
    """Use the existing L2B A-identity and availability certifiers."""
    if bundle is None:
        return "PENDING_A_REFERENCE_RECEIPT", None, None, ()
    base_required = {
        "schema", "availability_receipt", "identity_receipt", "reviewed_manifest", "attestation",
    }
    if (not base_required.issubset(bundle)
            or bundle.get("schema") != G1A_A_BUNDLE_SCHEMA):
        return "PENDING_A_REFERENCE_RECEIPT", None, None, ()
    allowed_bundle_fields = base_required | {
        "image_identity_receipt", "image_identity_attestation",
    }
    if set(bundle) - allowed_bundle_fields:
        return "PENDING_A_REFERENCE_RECEIPT", None, None, ()
    image_evidence_missing = not {
        "image_identity_receipt", "image_identity_attestation",
    }.issubset(bundle)
    try:
        a_value = bundle["availability_receipt"]
        i_value = dict(bundle["identity_receipt"])
        m_value = dict(bundle["reviewed_manifest"])
        t_value = dict(bundle["attestation"])
        i_value["group_members"] = tuple(SourceGroupMember(**x) for x in i_value["group_members"])
        m_value["group_members"] = tuple(SourceGroupMember(**x) for x in m_value["group_members"])
        receipt = AReferenceAvailabilityReceipt(**a_value)
        identity_receipt = RoleIdentityReceipt(**i_value)
        reviewed_manifest = ReviewedIdentityManifest(**m_value)
        attestation = IndependentReviewerAttestation(**t_value)
        if not image_evidence_missing:
            image_value = dict(bundle["image_identity_receipt"])
            image_attestation_value = dict(bundle["image_identity_attestation"])
            image_value["image_members"] = tuple(
                G1AImageIdentityMember(**x) for x in image_value["image_members"])
            image_receipt = G1AImageIdentityReceipt(**image_value)
            image_attestation = G1AImageIdentityAttestation(**image_attestation_value)
    except (TypeError, KeyError, ValueError):
        return "PENDING_A_REFERENCE_RECEIPT", None, None, ()
    if receipt.category_id != "switch":
        return "PENDING_A_REFERENCE_RECEIPT", receipt.receipt_sha256, None, ()
    status = certify_a_reference_availability(
        receipt,
        expected_category="switch",
        independently_certified_a_identity=identity_receipt,
        manifest=reviewed_manifest,
        attestation=attestation,
        trust_directory=trust_directory,
        expected_trust_directory_sha256=expected_trust_directory_sha256,
    )
    if status != "AUTHORIZED_FOR_G1_REFERENCE_POPULATION":
        return status, receipt.receipt_sha256, None, ()
    if image_evidence_missing:
        return "PENDING_A_IMAGE_IDENTITY_REVIEW", receipt.receipt_sha256, None, ()

    image_payload = asdict(image_receipt)
    image_payload.pop("receipt_sha256")
    expected_group_views = {
        member.group_sha256: member.image_view_count
        for member in identity_receipt.group_members
    }
    observed_group_views: dict[str, int] = {}
    seen_images: set[str] = set()
    seen_group_cameras: set[tuple[str, int]] = set()
    try:
        for member in image_receipt.image_members:
            _sha256(member.image_sha256, "A image identity digest")
            _sha256(member.source_group_sha256, "A image Source Group digest")
            if isinstance(member.camera_id, bool) or member.camera_id not in (1, 2, 3, 4, 5):
                raise G1APilotError("A image inventory contains an invalid camera ID")
            if member.image_sha256 in seen_images:
                raise G1APilotError("A image inventory contains duplicate image identities")
            seen_images.add(member.image_sha256)
            pair = (member.source_group_sha256, member.camera_id)
            if pair in seen_group_cameras:
                raise G1APilotError("A image inventory duplicates a camera within one Source Group")
            seen_group_cameras.add(pair)
            if member.source_group_sha256 not in expected_group_views:
                raise G1APilotError("A image inventory contains a group outside the certified A role")
            observed_group_views[member.source_group_sha256] = (
                observed_group_views.get(member.source_group_sha256, 0) + 1)
        if observed_group_views != expected_group_views:
            raise G1APilotError("A image inventory does not cover the certified group view counts")
        if (image_receipt.schema != G1A_IMAGE_INVENTORY_SCHEMA
                or image_receipt.category_id != "switch"
                or image_receipt.role != "A"
                or image_receipt.population != "authorized_normal_reference_only"
                or image_receipt.identity_contract != G1A_IDENTITY_CONTRACT
                or image_receipt.group_digest_algorithm != G1A_GROUP_DIGEST_ALGORITHM
                or image_receipt.source_manifest_sha256 != identity_receipt.source_manifest_sha256
                or image_receipt.a_reference_receipt_sha256 != receipt.receipt_sha256
                or not image_receipt.producer_id.strip()
                or image_receipt.image_count != len(image_receipt.image_members)
                or image_receipt.image_count != receipt.authorized_normal_reference_image_count
                or image_receipt.contains_raw_sample_keys
                or image_receipt.contains_image_paths
                or image_receipt.contains_labels_or_masks
                or canonical_sha256(image_payload) != image_receipt.receipt_sha256):
            raise G1APilotError("A image identity receipt is malformed or incompatible")
        if (trust_directory is None
                or expected_trust_directory_sha256 is None
                or reviewer_trust_directory_sha256(trust_directory) != trust_directory.directory_sha256
                or expected_trust_directory_sha256 != trust_directory.directory_sha256
                or image_attestation.reviewer_id in {
                    image_receipt.producer_id, identity_receipt.producer_id,
                    attestation.reviewer_id,
                }
                or image_attestation.reviewer_id not in trust_directory.trusted_reviewer_ids
                or image_attestation.decision != "CERTIFY_G1A_IMAGE_IDENTITY_INVENTORY"
                or image_attestation.image_identity_receipt_sha256 != image_receipt.receipt_sha256
                or not image_attestation.decision_id.strip()):
            raise G1APilotError("A image identity reviewer attestation is absent or mismatched")
        attestation_payload = asdict(image_attestation)
        attestation_payload.pop("signature_ed25519_hex")
        attestation_payload.pop("attestation_sha256")
        expected_attestation_sha = canonical_sha256({
            "payload": attestation_payload,
            "signature_ed25519_hex": image_attestation.signature_ed25519_hex,
        })
        if expected_attestation_sha != image_attestation.attestation_sha256:
            raise G1APilotError("A image identity attestation digest is invalid")
        reviewer_key = dict(trust_directory.trusted_reviewer_public_keys).get(
            image_attestation.reviewer_id)
        if reviewer_key is None or not _ed25519_verify(
                reviewer_key, attestation_payload, image_attestation.signature_ed25519_hex):
            raise G1APilotError("A image identity reviewer signature is invalid")
    except G1APilotError:
        return "PENDING_A_IMAGE_IDENTITY_REVIEW", receipt.receipt_sha256, image_receipt.receipt_sha256, ()
    return (
        "AUTHORIZED_FOR_G1_REFERENCE_POPULATION", receipt.receipt_sha256,
        image_receipt.receipt_sha256, image_receipt.image_members,
    )


def make_execution_permit(
    scope: Mapping[str, Any] | None,
    trust_directory: ReviewerTrustDirectory | None,
    a_bundle: Mapping[str, Any] | None,
    *,
    expected_trust_directory_sha256: str | None,
    expected_source_fingerprint_sha256: str | None,
    actual_source_fingerprint_sha256: str | None,
    expected_execution_commit: str | None,
    repo_root: str | Path,
) -> G1AExecutionPermit:
    """Issue the in-process image-access token only after every pre-decode gate."""
    scope_errors = validate_authorization_scope(
        scope, trust_directory,
        expected_trust_directory_sha256=expected_trust_directory_sha256,
        expected_source_fingerprint_sha256=expected_source_fingerprint_sha256,
        actual_source_fingerprint_sha256=actual_source_fingerprint_sha256,
        expected_execution_commit=expected_execution_commit,
        repo_root=repo_root,
    )
    if scope_errors:
        raise G1AAuthorizationError("G1-A scope failed closed: " + "; ".join(scope_errors))
    a_status, a_receipt_sha, image_receipt_sha, image_members = validate_a_evidence_bundle(
        a_bundle, trust_directory,
        expected_trust_directory_sha256=expected_trust_directory_sha256,
    )
    if (a_status != "AUTHORIZED_FOR_G1_REFERENCE_POPULATION"
            or a_receipt_sha is None or image_receipt_sha is None):
        raise G1AAuthorizationError(f"G1-A A-identity/availability gate is {a_status}")
    if a_receipt_sha != scope.get("a_reference_receipt_sha256"):
        raise G1AAuthorizationError("signed scope binds a different A availability receipt")
    if image_receipt_sha != scope.get("image_identity_receipt_sha256"):
        raise G1AAuthorizationError("signed scope binds a different A image identity receipt")
    identity_value = a_bundle["identity_receipt"]
    candidate_groups = [member["group_sha256"] for member in identity_value["group_members"]]
    selected_groups = select_bounded_groups(candidate_groups)
    image_by_group: dict[str, list[G1AImageIdentityMember]] = {}
    for member in image_members:
        image_by_group.setdefault(member.source_group_sha256, []).append(member)
    selected_images: list[G1AImageIdentityMember] = []
    for group_digest in selected_groups:
        camera_one = [member for member in image_by_group.get(group_digest, ())
                      if member.camera_id == 1]
        if len(camera_one) != 1:
            raise G1AAuthorizationError(
                "camera-ID-1 image inventory is absent or inconsistent for a ranked A Source Group")
        selected_images.append(camera_one[0])
    return G1AExecutionPermit(
        authorization_id=str(scope["authorization_id"]),
        authorization_sha256=str(scope["authorization_sha256"]),
        category_id="switch",
        a_reference_receipt_sha256=a_receipt_sha,
        image_identity_receipt_sha256=image_receipt_sha,
        selected_source_group_sha256s=selected_groups,
        authorized_image_identities=tuple(
            (member.image_sha256, member.source_group_sha256, member.camera_id)
            for member in selected_images),
        s1_image_sha256s=tuple(member.image_sha256 for member in selected_images[:S1_GROUP_COUNT]),
        representation_sha256=R1_DESCRIPTOR_PROFILE.representation_sha256,
        execution_commit=str(scope["execution_commit"]),
        output_directory=str(scope["output_directory"]),
        max_output_bytes=MAX_OUTPUT_BYTES,
        min_free_bytes=MIN_FREE_BYTES,
        max_wall_seconds=MAX_WALL_SECONDS,
        max_cpu_threads=MAX_CPU_THREADS,
        max_cuda_reserved_bytes=MAX_CUDA_RESERVED_BYTES,
        _authorization_seal=_G1A_PERMIT_SEAL,
    )


def create_g1a_output_store(
    permit: G1AExecutionPermit,
    *,
    repo_root: str | Path,
    free_bytes: Callable[[Path], int] | None = None,
):
    """Open a new 1-GiB pre-write-budgeted G1 output store after authorization."""
    if permit._authorization_seal is not _G1A_PERMIT_SEAL:
        raise G1AAuthorizationError("output store requires a permit issued by signed-scope validation")
    output = Path(os.path.abspath(permit.output_directory))
    root = Path(repo_root).resolve(strict=True)
    expected_parent = Path(os.path.abspath(root / OUTPUT_PARENT_RELATIVE))
    if _path_has_reparse_or_symlink(output):
        raise G1AOutputError("G1 output path contains a symlink or reparse point")
    if (output.parent != expected_parent or output.exists()
            or not re.fullmatch(r"r2_g1a_switch_[A-Za-z0-9_-]{6,64}", output.name)):
        raise G1AOutputError("G1 output must be a new direct child; historical output reuse is refused")
    if permit.max_output_bytes != MAX_OUTPUT_BYTES or permit.min_free_bytes != MIN_FREE_BYTES:
        raise G1AOutputError("G1 output or free-space cap differs from the signed scope")
    available = (free_bytes or (lambda p: shutil.disk_usage(p).free))(output.parent if output.parent.exists() else root)
    if available < permit.min_free_bytes:
        raise G1AOutputError("output volume is below the signed 10-GiB free-space floor")
    from experiments.calibration.r1_matched_correction_pilot import OutputBudget
    budget = OutputBudget(
        output, max_bytes=permit.max_output_bytes,
        failure_reserve_bytes=FAILURE_RESERVE_BYTES,
    )
    try:
        budget.ensure_root()
        (output / "events").mkdir(exist_ok=False)
    except BaseException:
        budget.close()
        raise
    return G1AOutputStore(output, budget)


class G1AOutputStore:
    """Store only projected descriptors and sanitized receipts in NPZ shards."""

    def __init__(self, root: Path, budget: Any):
        self.root = root
        self.budget = budget
        self._closed = False

    def write_projected(
        self,
        ordinal: int,
        descriptors: np.ndarray,
        receipt: G1AProjectedDescriptorReceipt,
        *,
        expected_record: G1AImageIdentity,
    ) -> Path:
        if self._closed or isinstance(ordinal, bool) or not 1 <= ordinal <= 100:
            raise G1AOutputError("output store is closed or shard ordinal is outside 1..100")
        validate_projected_receipt(receipt, descriptors, expected_record=expected_record)
        payload = json.dumps(asdict(receipt), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        stream = io.BytesIO()
        np.savez(
            stream,
            projected_descriptors=np.asarray(descriptors, dtype=np.float32),
            sanitized_receipt=np.frombuffer(payload, dtype=np.uint8),
        )
        path = self.root / "events" / f"event_{ordinal:06d}.npz"
        self.budget.write_once(path, stream.getvalue())
        return path

    def update_manifest(self, updates: Mapping[str, Any], *, failure_reserved: bool = False) -> None:
        path = self.root / "run_manifest.json"
        if not path.is_file():
            raise G1AOutputError("development run manifest is missing")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise G1AOutputError("development run manifest is unreadable") from exc
        manifest.update(dict(updates))
        manifest["output_budget"] = {
            "max_bytes": MAX_OUTPUT_BYTES,
            "failure_reserve_bytes": FAILURE_RESERVE_BYTES,
            "normal_write_limit_bytes": MAX_OUTPUT_BYTES - FAILURE_RESERVE_BYTES,
            "enforcement": "OutputBudget pre-write accounting and Windows mutex",
            "os_hard_storage_isolation": False,
        }
        manifest["accounted_output_bytes_before_manifest_write"] = self.budget._audit_tree()
        self.budget.atomic_json(path, manifest, failure_reserved=failure_reserved)

    def write_private_identity_manifest(self, value: Mapping[str, Any]) -> Path:
        path = self.root / "private_identity_manifest.json"
        self.budget.write_once(path, _canonical_json_bytes(value) + b"\n")
        return path

    def write_public_summary(self, value: Mapping[str, Any]) -> Path:
        forbidden = {"image_path", "sample_key", "official_sample_key", "labels", "masks"}
        def walk(item: Any) -> bool:
            if isinstance(item, Mapping):
                return any(str(key).casefold() in forbidden or walk(child)
                           for key, child in item.items())
            if isinstance(item, (list, tuple)):
                return any(walk(child) for child in item)
            return False
        if walk(value):
            raise G1AOutputError("public G1 summary contains private identity or prohibited data")
        path = self.root / "public_summary.json"
        self.budget.write_once(path, _canonical_json_bytes(value) + b"\n")
        return path

    def close(self) -> None:
        if not self._closed:
            self.budget.close()
            self._closed = True

    def __enter__(self) -> "G1AOutputStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def G1ADevelopmentOutputBudget(*args, **kwargs):
    """Tight G1 file allowlist over the shared R1 pre-write accountant."""
    from experiments.calibration.r1_matched_correction_pilot import OutputBudget

    additional_root_files = frozenset({
        "private_identity_manifest.json", "public_summary.json",
    })

    class _Budget(OutputBudget):
        def _relative(self, path: Path) -> str:
            try:
                return super()._relative(path)
            except Exception as exc:
                candidate = Path(os.path.abspath(path))
                try:
                    relative = candidate.relative_to(self.root).as_posix()
                except ValueError:
                    raise
                if (relative in additional_root_files
                        and "audited writer allowlist" in str(exc)):
                    return relative
                raise

    return _Budget(*args, **kwargs)


def _validate_development_output_path(
    output_directory: str | Path,
    *,
    repo_root: str | Path,
    must_not_exist: bool,
) -> Path:
    root = Path(repo_root).resolve(strict=True)
    output = Path(os.path.abspath(os.fspath(output_directory)))
    expected_parent = Path(os.path.abspath(root / OUTPUT_PARENT_RELATIVE))
    if _path_has_reparse_or_symlink(output):
        raise G1AOutputError("development output path contains a symlink or reparse point")
    if output.parent != expected_parent:
        raise G1AOutputError("development output must be a direct child of the G1 pilot namespace")
    if not re.fullmatch(r"r2_g1a_switch_[A-Za-z0-9_-]{6,64}", output.name):
        raise G1AOutputError("development output directory needs a unique G1-A switch run ID")
    if must_not_exist and output.exists():
        raise G1AOutputError("development run ID already exists; restart/resume is forbidden")
    return output


def _prepare_development_output_namespace(
    output_directory: str | Path,
    *,
    repo_root: str | Path,
    initial_manifest: Mapping[str, Any],
) -> Path:
    output = _validate_development_output_path(
        output_directory, repo_root=repo_root, must_not_exist=True)
    budget = G1ADevelopmentOutputBudget(
        output, max_bytes=MAX_OUTPUT_BYTES,
        failure_reserve_bytes=FAILURE_RESERVE_BYTES,
    )
    try:
        budget.ensure_root()
        (output / "events").mkdir(exist_ok=False)
        budget.atomic_json(output / "run_manifest.json", initial_manifest)
    except BaseException:
        budget.close()
        raise
    budget.close()
    return output


def _open_development_output_store(
    output_directory: str | Path,
    *,
    repo_root: str | Path,
    worker_start: bool,
) -> G1AOutputStore:
    output = _validate_development_output_path(
        output_directory, repo_root=repo_root, must_not_exist=False)
    if not output.is_dir() or not (output / "events").is_dir():
        raise G1AOutputError("development run namespace is incomplete")
    if worker_start:
        entries = {path.name for path in output.iterdir()}
        if entries != {"run_manifest.json", "events"} or any((output / "events").iterdir()):
            raise G1AOutputError("development worker refuses restart or nonempty prelaunch outputs")
        try:
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
        except Exception as exc:
            raise G1AOutputError("prelaunch development manifest is malformed") from exc
        if manifest.get("status") != "AUTHORIZED_PREPARED":
            raise G1AOutputError("development worker lacks a fresh prepared run manifest")
    budget = G1ADevelopmentOutputBudget(
        output, max_bytes=MAX_OUTPUT_BYTES,
        failure_reserve_bytes=FAILURE_RESERVE_BYTES,
        allow_existing=True,
    )
    return G1AOutputStore(output, budget)


def default_authorization_directory() -> Path:
    return Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".r2_g1a_switch_authorization"


def read_only_artifact_status(authorization_dir: str | Path) -> dict[str, Any]:
    """Report gate artifacts without opening any dataset path or image."""
    base = Path(authorization_dir)
    expected = {
        "authorization": base / "authorization.signed.json",
        "trust_directory": base / "trust_directory.json",
        "trust_directory_pin": base / "trust_directory.sha256",
        "source_fingerprint": base / "source_fingerprint.json",
        "source_fingerprint_pin": base / "source_fingerprint.sha256",
        "a_reference_bundle": base / "a_reference_bundle.json",
    }
    missing = [str(path) for path in expected.values() if not path.is_file()]
    return {
        "schema": "r2_g1a.read_only_preflight_status.v1",
        "status": "WAITING_RESEARCHER_AUTHORIZATION" if missing else "ARTIFACTS_PRESENT_VALIDATION_REQUIRED",
        "image_access_performed": False,
        "dataset_access_performed": False,
        "expected_artifacts": {key: str(path) for key, path in expected.items()},
        "missing_artifacts": missing,
    }


def canonicalize_scope_draft(scope: Mapping[str, Any]) -> tuple[bytes, str]:
    """Return exact bytes and digest for the researcher's external Ed25519 signer."""
    if set(scope) != G1A_AUTHORIZATION_KEYS:
        raise G1AAuthorizationError("draft fields do not match the G1-A authorization schema")
    payload = authorization_payload(scope)
    encoded = _canonical_json_bytes(payload)
    return encoded, _sha256_bytes(encoded)


def _read_pin(path: Path) -> str:
    try:
        from experiments.calibration.r2_l2c_metadata_census import _reject_local_approval_path_under_dataset
        _reject_local_approval_path_under_dataset(path)
        value = path.read_text(encoding="ascii").strip()
    except Exception as exc:
        raise G1APilotError(f"cannot read independent digest pin: {path.name}") from exc
    return _sha256(value, f"{path.name} value")


def _parse_a_bundle_file(path: Path) -> dict[str, Any]:
    return read_json_object(path)


def run_read_only_preflight(
    authorization_dir: str | Path,
    *,
    expected_trust_directory_sha256: str | None,
    expected_source_fingerprint_sha256: str | None,
    repo_root: str | Path,
    checkpoint_path: str | Path,
    hub_source_root: str | Path,
) -> dict[str, Any]:
    """Validate local authorization evidence without touching dataset files."""
    base = Path(authorization_dir)
    inventory = read_only_artifact_status(base)
    if inventory["missing_artifacts"]:
        return inventory
    report: dict[str, Any] = {
        "schema": "r2_g1a.read_only_preflight_status.v1",
        "image_access_performed": False,
        "dataset_access_performed": False,
        "loaded_model_state_status": "PENDING_MODEL_INITIALIZATION_BEFORE_IMAGE_DECODE",
        "errors": [],
    }
    try:
        from experiments.calibration.r2_l2c_metadata_census import (
            trust_directory_from_mapping,
        )
        trust = trust_directory_from_mapping(read_json_object(base / "trust_directory.json"))
        trust_pin_file = _read_pin(base / "trust_directory.sha256")
        source_pin_file = _read_pin(base / "source_fingerprint.sha256")
        source_artifact = read_json_object(base / "source_fingerprint.json")
        actual_source = compute_source_fingerprint(
            repo_root, checkpoint_path=checkpoint_path,
            hub_source_root=hub_source_root,
        )
        if source_artifact != actual_source:
            raise G1APilotError("recorded source fingerprint differs from current source/model files")
        if (expected_trust_directory_sha256 is None
                or trust_pin_file != expected_trust_directory_sha256):
            raise G1APilotError("independent trust-directory pin was not supplied or does not match")
        if (expected_source_fingerprint_sha256 is None
                or source_pin_file != expected_source_fingerprint_sha256
                or actual_source["source_fingerprint_sha256"] != expected_source_fingerprint_sha256):
            raise G1APilotError("independent source-fingerprint pin was not supplied or does not match")
        scope = read_json_object(base / "authorization.signed.json")
        commit = str(actual_source["payload"]["execution_commit"])
        scope_errors = validate_authorization_scope(
            scope, trust,
            expected_trust_directory_sha256=expected_trust_directory_sha256,
            expected_source_fingerprint_sha256=expected_source_fingerprint_sha256,
            actual_source_fingerprint_sha256=actual_source["source_fingerprint_sha256"],
            expected_execution_commit=commit,
            repo_root=repo_root,
        )
        if scope_errors:
            report["errors"].extend(scope_errors)
        bundle = _parse_a_bundle_file(base / "a_reference_bundle.json")
        a_status, a_receipt_sha, image_receipt_sha, _ = validate_a_evidence_bundle(
            bundle, trust,
            expected_trust_directory_sha256=expected_trust_directory_sha256,
        )
        report["a_reference_status"] = a_status
        report["a_reference_receipt_sha256"] = a_receipt_sha
        report["image_identity_receipt_sha256"] = image_receipt_sha
        if a_status != "AUTHORIZED_FOR_G1_REFERENCE_POPULATION":
            report["errors"].append(f"independently certified A and image identity evidence is {a_status}")
        elif scope.get("a_reference_receipt_sha256") != a_receipt_sha:
            report["errors"].append("signed authorization is not bound to the certified A receipt")
        elif scope.get("image_identity_receipt_sha256") != image_receipt_sha:
            report["errors"].append("signed authorization is not bound to the certified A image inventory")
        output = Path(str(scope.get("output_directory", "")))
        free_path = output.parent if output.parent.exists() else Path(repo_root)
        free = shutil.disk_usage(free_path).free
        report["output_free_bytes"] = int(free)
        if free < MIN_FREE_BYTES:
            report["errors"].append("output volume is below the signed 10-GiB free-space floor")
        report["source_fingerprint_sha256"] = actual_source["source_fingerprint_sha256"]
        report["trust_directory_sha256"] = trust.directory_sha256
    except Exception as exc:
        # Keep identity/trust documents private; expose only the gate failure class/message.
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["status"] = (
        "PREFLIGHT_READY_NO_DATA_IO" if not report["errors"]
        else "WAITING_RESEARCHER_AUTHORIZATION"
    )
    return report


def _git_text(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo_root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise G1APilotError(f"cannot verify repository identity with git {args[0]}") from exc


def _verify_development_repository(repo_root: str | Path, execution_commit: str | None = None) -> tuple[str, str]:
    root = Path(repo_root).resolve(strict=True)
    branch = _git_text(root, "branch", "--show-current")
    commit = _git_text(root, "rev-parse", "HEAD")
    parent = _git_text(root, "rev-parse", "HEAD^")
    if branch != "calibration-stage0-contract-20261029":
        raise G1APilotError("development pilot is restricted to the reviewed calibration branch")
    if parent != G1A_DEVELOPMENT_PARENT_COMMIT:
        raise G1APilotError("development execution commit is not directly based on the authorized parent")
    if execution_commit is not None and commit != execution_commit:
        raise G1APilotError("development worker execution commit differs from its supervised parent")
    return branch, commit


def _local_source_fingerprint(repo_root: str | Path) -> dict[str, Any]:
    import torch

    hub_dir = Path(torch.hub.get_dir())
    hub_source = _cached_dinov2_source_root(hub_dir)
    checkpoint = hub_dir / "checkpoints" / R1_DESCRIPTOR_PROFILE.checkpoint_file
    return compute_source_fingerprint(
        Path(repo_root), checkpoint_path=checkpoint, hub_source_root=hub_source)


def _cached_dinov2_source_root(hub_dir: Path) -> Path:
    source_candidates = sorted(
        path for path in hub_dir.glob("facebookresearch_dinov2*")
        if path.is_dir() and (path / "hubconf.py").is_file())
    if len(source_candidates) != 1:
        raise G1APilotError("exactly one cached DINOv2 source tree is required for offline loading")
    return source_candidates[0]


def _instantiate_r1_backbone_offline(implementation: Any, torch: Any) -> Any:
    """Load only the fingerprinted local DINO tree and cached checkpoint."""
    import urllib.parse

    hub_dir = Path(torch.hub.get_dir())
    source_root = _cached_dinov2_source_root(hub_dir)
    checkpoint = hub_dir / "checkpoints" / R1_DESCRIPTOR_PROFILE.checkpoint_file
    verify_file_sha256(checkpoint, R1_DESCRIPTOR_PROFILE.checkpoint_sha256)
    original_load = torch.hub.load
    original_weights = torch.hub.load_state_dict_from_url
    original_download = torch.hub.download_url_to_file

    def local_only_load(repo_or_dir: Any, model: Any, *args: Any, **kwargs: Any) -> Any:
        if (str(repo_or_dir) != "facebookresearch/dinov2"
                or str(model) != "dinov2_vitl14_reg"):
            raise G1APilotError("DINO loader request differs from the frozen R1 model profile")
        kwargs["source"] = "local"
        return original_load(str(source_root), model, *args, **kwargs)

    def cached_weights_only(url: Any, *args: Any, **kwargs: Any) -> Any:
        basename = Path(urllib.parse.urlparse(str(url)).path).name
        if basename != R1_DESCRIPTOR_PROFILE.checkpoint_file:
            raise G1APilotError("DINO requested a checkpoint outside the frozen R1 profile")
        return original_weights(url, *args, **kwargs)

    def deny_download(*_args: Any, **_kwargs: Any) -> None:
        raise G1APilotError("network download is disabled for the development pilot")

    torch.hub.load = local_only_load
    torch.hub.load_state_dict_from_url = cached_weights_only
    torch.hub.download_url_to_file = deny_download
    try:
        return implementation.DINOv2MultiLayerBackbone(device="cuda:0")
    finally:
        torch.hub.load = original_load
        torch.hub.load_state_dict_from_url = original_weights
        torch.hub.download_url_to_file = original_download


def _peak_working_set_bytes() -> int | None:
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        process = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    except Exception:
        return None


def _prepare_development_authorization_scope(
    *,
    branch: str,
    execution_commit: str,
    source_fingerprint: Mapping[str, Any],
    run_id: str,
    output_directory: Path,
    start_utc: str,
) -> dict[str, Any]:
    return {
        "schema": G1A_DEVELOPMENT_RUN_SCHEMA,
        "mode": "DEVELOPMENT_ONLY",
        "authorization_id": G1A_DEVELOPMENT_AUTHORIZATION_ID,
        "authorization_basis": "explicit researcher task R2-G1A-SWITCH-DEV-RUN",
        "third_party_certification_claimed": False,
        "timestamp_utc": start_utc,
        "branch": branch,
        "authorized_parent_commit": G1A_DEVELOPMENT_PARENT_COMMIT,
        "execution_commit": execution_commit,
        "source_fingerprint_sha256": source_fingerprint["source_fingerprint_sha256"],
        "source_fingerprint": source_fingerprint,
        "run_id": run_id,
        "output_directory": str(output_directory),
        "dataset_root": DATASET_ROOT,
        "metadata_root": METADATA_ROOT,
        "category_id": "switch",
        "split": "TRAIN",
        "population": "verified_normal_only",
        "metadata_projection_fields": [
            "meta.normal_class", "meta.prefix", "TRAIN.anomaly_class",
            "TRAIN.image_path only when anomaly_class == meta.normal_class",
        ],
        "test_section": "structurally_skipped_without_record_materialization",
        "stages": ["S1", "S2"],
        "maximum_source_groups": 100,
        "sample_policy": G1A_SAMPLE_POLICY,
        "sample_seed": 0,
        "camera_id": 1,
        "representation_sha256": R1_DESCRIPTOR_PROFILE.representation_sha256,
        "maximum_output_bytes": MAX_OUTPUT_BYTES,
        "failure_reserve_bytes": FAILURE_RESERVE_BYTES,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "maximum_total_runtime_seconds": MAX_WALL_SECONDS,
        "cpu_thread_limit": MAX_CPU_THREADS,
        "cuda_device": "cuda:0",
        "maximum_cuda_reserved_bytes": MAX_CUDA_RESERVED_BYTES,
        "prohibited": [
            "TEST identity or outcome parsing", "labels", "masks", "non-switch categories",
            "Mint", "PCB", "MVTec AD 2", "G1-B", "G2", "Support/C64",
            "Correction-radius optimization",
        ],
    }


def run_development_pilot(
    *,
    repo_root: str | Path,
    run_id: str,
    authorization_token: str,
) -> dict[str, Any]:
    """Prepare one unique development namespace and supervise its S1->S2 worker."""
    if authorization_token != G1A_DEVELOPMENT_AUTHORIZATION_ID:
        raise G1AAuthorizationError("explicit --authorize-development-only token is required")
    root = Path(repo_root).resolve(strict=True)
    task_started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    branch, commit = _verify_development_repository(root)
    source_fingerprint = _local_source_fingerprint(root)
    output = _validate_development_output_path(
        root / OUTPUT_PARENT_RELATIVE / run_id, repo_root=root, must_not_exist=True)
    free_bytes = int(shutil.disk_usage(root).free)
    if free_bytes < MIN_FREE_BYTES:
        raise G1AOutputError("development output volume has less than 10 GiB free")
    remaining = MAX_WALL_SECONDS - (time.monotonic() - task_started)
    if remaining <= 1.0:
        raise G1APilotError("total development runtime budget expired before child launch")
    scope = _prepare_development_authorization_scope(
        branch=branch, execution_commit=commit,
        source_fingerprint=source_fingerprint, run_id=run_id,
        output_directory=output, start_utc=started_utc,
    )
    scope_sha = canonical_sha256(scope)
    deadline = task_started + MAX_WALL_SECONDS
    initial_manifest = {
        "schema": G1A_DEVELOPMENT_RUN_SCHEMA,
        "status": "AUTHORIZED_PREPARED",
        "stage": "PRELAUNCH",
        "development_only": True,
        "independent_a_role_certification": "NOT_CLAIMED",
        "authorization_scope": scope,
        "authorization_scope_sha256": scope_sha,
        "output_free_bytes_prelaunch": free_bytes,
        "automatic_restart": False,
        "resumable": False,
    }
    _prepare_development_output_namespace(
        output, repo_root=root, initial_manifest=initial_manifest)
    nonce = os.urandom(32).hex()
    child_env = dict(os.environ)
    child_env.update({
        G1A_DEVELOPMENT_WORKER_ENV: nonce,
        "R2_G1A_DEVELOPMENT_SCOPE_SHA256": scope_sha,
        "R2_G1A_DEVELOPMENT_DEADLINE_MONOTONIC": repr(deadline),
    })
    command = [
        sys.executable, "-m", "experiments.calibration.r2_g1a_switch_pilot",
        "--development-worker", "--run-id", run_id,
        "--output-directory", str(output), "--execution-commit", commit,
        "--source-fingerprint-sha256", source_fingerprint["source_fingerprint_sha256"],
    ]
    from experiments.calibration.r1_runtime_supervisor import (
        run_supervised_child,
    )
    try:
        result = run_supervised_child(
            command, timeout_seconds=min(MAX_WALL_SECONDS, remaining),
            env=child_env, cwd=root, output_dir=None,
        )
    except Exception as exc:
        # A blocking supervisor failure means the process tree may still be live;
        # do not race it with a second writer or restart the experiment.
        from experiments.calibration.r1_runtime_supervisor import SupervisorBlockingFailure
        if isinstance(exc, SupervisorBlockingFailure):
            raise
        _finalize_development_failure(
            output, repo_root=root,
            failure_type=type(exc).__name__,
            failure_message="supervisor could not complete a confirmed child run",
            supervisor_error=type(exc).__name__,
        )
        return {
            "status": "INCOMPLETE",
            "run_id": run_id,
            "execution_commit": commit,
            "source_fingerprint_sha256": source_fingerprint["source_fingerprint_sha256"],
            "failure_type": type(exc).__name__,
            "output_directory": str(output),
        }
    record = asdict(result)
    record.update({
        "schema": "r2_g1a.development_supervisor_status.v1",
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "automatic_restart": False,
        "process_tree_containment": "Windows Job Object, suspended assignment, kill-on-close",
        "output_budget_bytes": MAX_OUTPUT_BYTES,
        "output_free_bytes_prelaunch": free_bytes,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "os_hard_storage_isolation": False,
    })
    if result.status != "COMPLETED" or result.child_exit_code != 0:
        _mark_development_manifest_failed(
            output, repo_root=root, failure_type=result.status,
            failure_message="development child did not complete successfully",
        )
    total_output_bytes = _write_supervisor_status(output, repo_root=root, record=record)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "status": manifest.get("status", "INCOMPLETE"),
        "run_id": run_id,
        "execution_commit": commit,
        "source_fingerprint_sha256": source_fingerprint["source_fingerprint_sha256"],
        "supervisor": record,
        "actual_output_bytes": total_output_bytes,
        "output_directory": str(output),
    }


def _mark_development_manifest_failed(
    output: Path,
    *,
    repo_root: Path,
    failure_type: str,
    failure_message: str,
) -> None:
    store = _open_development_output_store(
        output, repo_root=repo_root, worker_start=False)
    try:
        store.update_manifest({
            "status": "INCOMPLETE",
            "failure_type": failure_type,
            "failure_message": failure_message,
            "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }, failure_reserved=True)
    finally:
        store.close()


def _finalize_development_failure(
    output: Path,
    *,
    repo_root: Path,
    failure_type: str,
    failure_message: str,
    supervisor_error: str,
) -> None:
    _mark_development_manifest_failed(
        output, repo_root=repo_root, failure_type=failure_type,
        failure_message=failure_message)
    _write_supervisor_status(output, repo_root=repo_root, record={
        "schema": "r2_g1a.development_supervisor_status.v1",
        "status": "SUPERVISOR_ERROR_NO_RESULT",
        "termination_confirmed": None,
        "automatic_restart": False,
        "supervisor_error": supervisor_error,
        "os_hard_storage_isolation": False,
    })


def _write_supervisor_status(
    output: Path,
    *,
    repo_root: Path,
    record: Mapping[str, Any],
) -> int:
    from experiments.calibration.r1_matched_correction_pilot import canonical_json

    budget = G1ADevelopmentOutputBudget(
        output, max_bytes=MAX_OUTPUT_BYTES,
        failure_reserve_bytes=FAILURE_RESERVE_BYTES,
        allow_existing=True,
    )
    try:
        value = dict(record)
        target = output / "supervisor_status.json"
        current_total = budget._audit_tree()
        old_target_bytes = target.stat().st_size if target.is_file() else 0
        other_bytes = current_total - old_target_bytes
        predicted_total = other_bytes
        for _ in range(8):
            value["output_bytes"] = predicted_total
            encoded_length = len((canonical_json(value) + "\n").encode("utf-8"))
            next_total = other_bytes + encoded_length
            if next_total == predicted_total:
                break
            predicted_total = next_total
        else:
            raise G1AOutputError("supervisor status byte count did not reach a fixed point")
        budget.write_supervisor_record(value)
        actual = budget._audit_tree()
        if actual != predicted_total:
            raise G1AOutputError("supervisor status output-byte receipt does not match actual bytes")
        return actual
    finally:
        budget.close()


def _run_development_worker(
    *,
    repo_root: str | Path,
    run_id: str,
    output_directory: str | Path,
    execution_commit: str,
    source_fingerprint_sha256: str,
) -> dict[str, Any]:
    nonce = os.environ.get(G1A_DEVELOPMENT_WORKER_ENV, "")
    scope_sha = os.environ.get("R2_G1A_DEVELOPMENT_SCOPE_SHA256", "")
    try:
        deadline = float(os.environ.get("R2_G1A_DEVELOPMENT_DEADLINE_MONOTONIC", "nan"))
    except ValueError as exc:
        raise G1AAuthorizationError("development worker deadline is malformed") from exc
    if len(nonce) != 64 or not re.fullmatch(r"[0-9a-f]{64}", nonce):
        raise G1AAuthorizationError("development worker can only be launched by its supervised parent")
    _sha256(source_fingerprint_sha256, "expected source fingerprint")
    _sha256(scope_sha, "prepared development scope digest")
    if deadline <= time.monotonic():
        raise G1AAuthorizationError("development execution deadline expired before worker start")
    root = Path(repo_root).resolve(strict=True)
    branch, actual_commit = _verify_development_repository(root, execution_commit)
    actual_source = _local_source_fingerprint(root)
    if actual_source["source_fingerprint_sha256"] != source_fingerprint_sha256:
        raise G1APilotError("development worker source/model fingerprint changed after authorization")
    output = _validate_development_output_path(
        output_directory, repo_root=root, must_not_exist=False)
    if output.name != run_id:
        raise G1AAuthorizationError("development worker run ID does not match its unique output namespace")
    store = _open_development_output_store(output, repo_root=root, worker_start=True)
    started = time.monotonic()
    cpu_started = time.process_time()
    stage = "BOOTSTRAP"
    backbone = None
    torch_module = None
    try:
        manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("authorization_scope_sha256") != scope_sha
                or manifest.get("authorization_scope", {}).get("authorization_id")
                != G1A_DEVELOPMENT_AUTHORIZATION_ID
                or manifest.get("authorization_scope", {}).get("execution_commit") != actual_commit):
            raise G1AAuthorizationError("prepared development run manifest does not match worker scope")
        free_before = int(shutil.disk_usage(root).free)
        if free_before < MIN_FREE_BYTES:
            raise G1AOutputError("development output volume fell below the 10-GiB free-space floor")
        store.update_manifest({"status": "INCOMPLETE", "stage": "TRAIN_IDENTITY_PROJECTION"})

        metadata_root = Path(METADATA_ROOT).resolve(strict=True)
        if _path_has_reparse_or_symlink(metadata_root):
            raise G1APilotError("switch metadata root contains a symlink or reparse point")
        metadata_path = metadata_root / "switch.json"
        if _path_has_reparse_or_symlink(metadata_path) or not metadata_path.is_file():
            raise G1APilotError("the single authorized switch.json metadata file is unavailable or unsafe")
        if metadata_path.parent.resolve(strict=True) != metadata_root:
            raise G1APilotError("switch metadata path escaped the authorized metadata root")
        source_bytes = metadata_path.read_bytes()
        metadata_sha = _sha256_bytes(source_bytes)
        projection = project_switch_train_normal_json_bytes(source_bytes)
        records, private_identity_manifest, identity_sha, counts = build_development_identity_manifest(
            projection, metadata_source_sha256=metadata_sha, dataset_root=DATASET_ROOT)
        store.write_private_identity_manifest(private_identity_manifest)
        permit = make_development_permit(
            output_directory=output, execution_commit=actual_commit,
            source_fingerprint_sha256=source_fingerprint_sha256,
            identity_manifest_sha256=identity_sha, records=records,
        )
        store.update_manifest({
            "stage": "IDENTITY_FROZEN",
            "source_json_sha256": metadata_sha,
            "train_identity_projection": {
                **counts,
                "test_section_present": bool(projection["test_section_present"]),
                "test_values_structurally_skipped": True,
                "test_record_count_computed": False,
            },
            "private_identity_manifest_sha256": identity_sha,
            "group_ranking_policy": G1A_SAMPLE_POLICY,
            "group_ranking_seed": 0,
            "selected_group_order_sha256": canonical_sha256(
                list(permit.selected_source_group_sha256s)),
        })
        if time.monotonic() >= deadline:
            raise G1APilotError("total 120-minute deadline reached before DINO initialization")

        import torch
        from experiments import p0_protocol_audit as p0
        from experiments.calibration.r1_matched_correction_pilot import (
            _configure_historical_runtime,
        )

        torch_module = torch
        impl = p0.require_sources()
        runtime_profile = _configure_historical_runtime(impl, torch)
        if not torch.cuda.is_available():
            raise G1APilotError("development pilot requires CUDA for frozen R1 DINO inference")
        torch.cuda.set_device(0)
        free_vram, total_vram = torch.cuda.mem_get_info(0)
        if int(free_vram) < MAX_CUDA_RESERVED_BYTES:
            raise G1APilotError("cuda:0 has less than the 20-GiB development allocation cap available")
        cuda_fraction = apply_g1_resource_caps(
            torch, total_cuda_bytes=int(total_vram), device_index=0)
        torch.cuda.reset_peak_memory_stats(0)
        stage = "MODEL_INITIALIZATION"
        store.update_manifest({
            "stage": stage,
            "runtime_profile": runtime_profile,
            "runtime_limits": {
                "cpu_threads": MAX_CPU_THREADS,
                "cuda_device": "cuda:0",
                "cuda_allocation_cap_bytes": MAX_CUDA_RESERVED_BYTES,
                "cuda_allocation_fraction": cuda_fraction,
                "cuda_free_bytes_before_model": int(free_vram),
                "cuda_total_bytes": int(total_vram),
            },
        })
        backbone = _instantiate_r1_backbone_offline(impl, torch)
        loaded_state_sha = verify_loaded_model_state(backbone)
        store.update_manifest({
            "loaded_model_state_sha256": loaded_state_sha,
            "loaded_model_state_verified": True,
        })

        projected_cache: list[np.ndarray] = []
        receipts: list[G1AProjectedDescriptorReceipt] = []
        stage = "S1_SMOKE"
        for index, record in enumerate(records[:S1_GROUP_COUNT], start=1):
            if time.monotonic() >= deadline:
                raise G1APilotError("total 120-minute deadline reached during S1")
            array, receipt = encode_development_image(
                record, permit=permit, implementation=impl,
                backbone=backbone, stage="S1")
            projected_cache.append(array)
            receipts.append(receipt)
            store.write_projected(index, array, receipt, expected_record=record)
            store.update_manifest({
                "stage": stage,
                "s1_images_projected": index,
                "s1_images_expected": S1_GROUP_COUNT,
            })
        s1_stats, s1_summary = compare_g1a_development_cohort(
            records[:S1_GROUP_COUNT], projected_cache, receipts,
            permit=permit, identity_manifest_sha256=identity_sha, stage="S1")
        if (not np.isfinite(s1_stats.nearest_distance).all()
                or not np.isfinite(s1_stats.fifth_nearest_distance).all()
                or s1_summary["queries_with_fewer_than_five_eligible_references"] != 0):
            raise G1APilotError("S1 exact geometry failed finite nearest/fifth-neighbor checks")
        torch.cuda.synchronize()
        s1_peak_reserved = int(torch.cuda.max_memory_reserved(0))
        if s1_peak_reserved > MAX_CUDA_RESERVED_BYTES:
            raise G1APilotError("S1 exceeded the authorized CUDA reserved-memory cap")
        if time.monotonic() >= deadline:
            raise G1APilotError("total 120-minute deadline reached at the S1/S2 transition")
        store.update_manifest({
            "stage": "S1_PASSED",
            "s1_status": "ENGINEERING_PASS_DEVELOPMENT_ONLY",
            "s1_geometry_summary": s1_summary,
            "s1_peak_cuda_reserved_bytes": s1_peak_reserved,
            "s1_peak_working_set_bytes": _peak_working_set_bytes(),
            "s1_output_bytes": int(store.budget._audit_tree()),
            "s1_distance_values_used_for_cohort_changes": False,
        })

        stage = "S2_FORMAL_100_GROUP_PILOT"
        for index, record in enumerate(records[S1_GROUP_COUNT:], start=S1_GROUP_COUNT + 1):
            if time.monotonic() >= deadline:
                raise G1APilotError("total 120-minute deadline reached during S2")
            array, receipt = encode_development_image(
                record, permit=permit, implementation=impl,
                backbone=backbone, stage="S2")
            projected_cache.append(array)
            receipts.append(receipt)
            store.write_projected(index, array, receipt, expected_record=record)
            store.update_manifest({
                "stage": stage,
                "s2_images_projected": index,
                "s2_images_expected": 100,
            })
        s2_stats, s2_summary = compare_g1a_development_cohort(
            records, projected_cache, receipts,
            permit=permit, identity_manifest_sha256=identity_sha, stage="S2")
        if (not np.isfinite(s2_stats.nearest_distance).all()
                or not np.isfinite(s2_stats.fifth_nearest_distance).all()
                or s2_summary["queries_with_fewer_than_five_eligible_references"] != 0):
            raise G1APilotError("S2 exact geometry failed finite nearest/fifth-neighbor checks")
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        peak_cuda_reserved = int(torch.cuda.max_memory_reserved(0))
        if peak_cuda_reserved > MAX_CUDA_RESERVED_BYTES:
            raise G1APilotError("S2 exceeded the authorized CUDA reserved-memory cap")
        resources = {
            "worker_wall_seconds": elapsed,
            "worker_cpu_seconds": time.process_time() - cpu_started,
            "peak_working_set_bytes": _peak_working_set_bytes(),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_cuda_reserved_bytes": peak_cuda_reserved,
            "output_bytes_before_final_manifest": int(store.budget._audit_tree()),
        }
        public_summary = {
            "schema": "r2_g1a.switch_development_public_summary.v1",
            "status": "DEVELOPMENT_ONLY_DESCRIPTIVE_NORMAL_GEOMETRY",
            "development_only": True,
            "independent_a_role_certification": "NOT_CLAIMED",
            "correction_or_defect_safety_claim": "NONE",
            "authorization_id": G1A_DEVELOPMENT_AUTHORIZATION_ID,
            "execution_commit": actual_commit,
            "source_fingerprint_sha256": source_fingerprint_sha256,
            "representation_sha256": R1_DESCRIPTOR_PROFILE.representation_sha256,
            "identity_manifest_sha256": identity_sha,
            "selected_group_count": 100,
            "selected_image_count": 100,
            "descriptor_count": 4900,
            "camera_id": 1,
            "sample_policy": G1A_SAMPLE_POLICY,
            "sample_seed": 0,
            "s1": s1_summary,
            "s2": s2_summary,
            "resources": resources,
        }
        store.write_public_summary(public_summary)
        store.update_manifest({
            "status": "COMPLETED_DEVELOPMENT_ONLY",
            "stage": "S2_COMPLETE",
            "s2_status": "PILOT_COMPLETE_DEVELOPMENT_ONLY",
            "s2_geometry_summary": s2_summary,
            "resources": resources,
            "public_summary_file": "public_summary.json",
            "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        })
        return {
            "status": "COMPLETED_DEVELOPMENT_ONLY",
            "actual_group_count": 100,
            "actual_image_count": 100,
            "descriptor_count": 4900,
            "resources": resources,
        }
    except BaseException as exc:
        try:
            store.update_manifest({
                "status": "INCOMPLETE",
                "stage": stage,
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }, failure_reserved=True)
        except Exception:
            pass
        raise
    finally:
        if backbone is not None:
            try:
                if hasattr(backbone, "_clear_feats"):
                    backbone._clear_feats()
            except Exception:
                pass
            del backbone
        if torch_module is not None:
            try:
                if torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()
            except Exception:
                pass
        store.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-source-fingerprint", action="store_true")
    parser.add_argument("--canonicalize-draft", type=Path)
    parser.add_argument("--canonical-payload-output", type=Path)
    parser.add_argument("--authorization-dir", type=Path, default=default_authorization_directory())
    parser.add_argument("--expected-trust-directory-sha256")
    parser.add_argument("--expected-source-fingerprint-sha256")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hub-source", type=Path)
    parser.add_argument("--run-development-pilot", action="store_true")
    parser.add_argument("--authorize-development-only")
    parser.add_argument("--run-id")
    parser.add_argument("--development-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-directory", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--execution-commit", help=argparse.SUPPRESS)
    parser.add_argument("--source-fingerprint-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.development_worker:
        try:
            if not all((args.output_directory, args.execution_commit,
                        args.source_fingerprint_sha256, args.run_id)):
                raise G1AAuthorizationError("supervised development worker arguments are incomplete")
            report = _run_development_worker(
                repo_root=args.repo_root, run_id=args.run_id,
                output_directory=args.output_directory,
                execution_commit=args.execution_commit,
                source_fingerprint_sha256=args.source_fingerprint_sha256,
            )
            print(json.dumps(report, sort_keys=True, indent=2))
            return 0 if report.get("status") == "COMPLETED_DEVELOPMENT_ONLY" else 2
        except Exception as exc:
            print(json.dumps({
                "status": "INCOMPLETE",
                "development_only": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }, sort_keys=True, indent=2))
            return 2
    if args.run_development_pilot:
        try:
            if not args.run_id:
                raise G1AAuthorizationError("--run-id is required for a unique development output namespace")
            report = run_development_pilot(
                repo_root=args.repo_root, run_id=args.run_id,
                authorization_token=args.authorize_development_only or "",
            )
            print(json.dumps(report, sort_keys=True, indent=2))
            return 0 if report.get("status") == "COMPLETED_DEVELOPMENT_ONLY" else 2
        except Exception as exc:
            print(json.dumps({
                "status": "BLOCKED_OR_INCOMPLETE",
                "development_only": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }, sort_keys=True, indent=2))
            return 2
    if args.authorize_development_only is not None:
        parser.error("--authorize-development-only requires --run-development-pilot")
    if args.canonicalize_draft is not None:
        if args.canonical_payload_output is None:
            parser.error("--canonicalize-draft requires --canonical-payload-output")
        try:
            scope = read_json_object(args.canonicalize_draft)
            payload_bytes, payload_sha = canonicalize_scope_draft(scope)
            with args.canonical_payload_output.open("xb") as stream:
                stream.write(payload_bytes)
            print(json.dumps({
                "canonical_payload_path": str(args.canonical_payload_output),
                "authorization_sha256": payload_sha,
                "signature_ed25519_hex": "RESEARCHER_SIGNS_CANONICAL_PAYLOAD_LOCALLY",
            }, sort_keys=True, indent=2))
            return 0
        except Exception as exc:
            print(json.dumps({"status": "BLOCKED_CONTRACT_MISMATCH", "error": str(exc)}))
            return 2
    if args.print_source_fingerprint:
        try:
            import torch
            hub_dir = Path(torch.hub.get_dir())
            source_candidates = sorted(
                path for path in hub_dir.glob("facebookresearch_dinov2*")
                if path.is_dir() and (path / "hubconf.py").is_file())
            checkpoint = args.checkpoint or (hub_dir / "checkpoints" / R1_DESCRIPTOR_PROFILE.checkpoint_file)
            hub_source = args.hub_source or (source_candidates[0] if source_candidates else hub_dir / "MISSING")
            print(json.dumps(compute_source_fingerprint(
                args.repo_root, checkpoint_path=checkpoint,
                hub_source_root=hub_source), sort_keys=True, indent=2))
            return 0
        except Exception as exc:
            print(json.dumps({"status": "BLOCKED_CONTRACT_MISMATCH", "error": str(exc)}))
            return 2
    inventory = read_only_artifact_status(args.authorization_dir)
    if inventory["missing_artifacts"]:
        report = inventory
    else:
        try:
            import torch
            hub_dir = Path(torch.hub.get_dir())
            source_candidates = sorted(
                path for path in hub_dir.glob("facebookresearch_dinov2*")
                if path.is_dir() and (path / "hubconf.py").is_file())
            checkpoint = args.checkpoint or (hub_dir / "checkpoints" / R1_DESCRIPTOR_PROFILE.checkpoint_file)
            hub_source = args.hub_source or (source_candidates[0] if source_candidates else hub_dir / "MISSING")
            report = run_read_only_preflight(
                args.authorization_dir,
                expected_trust_directory_sha256=args.expected_trust_directory_sha256,
                expected_source_fingerprint_sha256=args.expected_source_fingerprint_sha256,
                repo_root=args.repo_root,
                checkpoint_path=checkpoint,
                hub_source_root=hub_source,
            )
        except Exception as exc:
            report = {
                "schema": "r2_g1a.read_only_preflight_status.v1",
                "status": "WAITING_RESEARCHER_AUTHORIZATION",
                "image_access_performed": False,
                "dataset_access_performed": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["status"] == "PREFLIGHT_READY_NO_DATA_IO" else 2


if __name__ == "__main__":
    sys.exit(main())
