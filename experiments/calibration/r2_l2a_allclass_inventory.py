"""Governed inventory and A/B/C partition preflight for R2-L2A.

This module consumes explicit, non-label-bearing receipts. It never discovers
categories from image paths, feature-cache filenames, class JSONs, or labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence


EXPECTED_CATEGORY_COUNT = 30
PINNED_REPOSITORY = "Tencent/AnomalyDetection_Real-IAD"
PINNED_COMMIT = "98c28ffbe95b380d17a3777d77219152368c19e9"
PINNED_SOURCE_PATH = "experiments/RealIAD-full/config.yaml"
PINNED_SOURCE_URI = (
    "https://github.com/Tencent/AnomalyDetection_Real-IAD/blob/"
    f"{PINNED_COMMIT}/{PINNED_SOURCE_PATH}")
PINNED_RAW_SOURCE_URI = (
    "https://raw.githubusercontent.com/Tencent/AnomalyDetection_Real-IAD/"
    f"{PINNED_COMMIT}/{PINNED_SOURCE_PATH}")
PINNED_SOURCE_SHA256 = "9c17d4ba78210ad65a288fedcdd6ea8a65d0926da05a960d3ed8dd1c3210552f"
PINNED_CONFIG_VERSION = "v1.0.0"
CATALOG_EXTRACTION_ALGORITHM = "strict_yaml_dataset_train_test_meta_file_v1"
CANONICAL_CATEGORY_IDS = (
    "audiojack", "bottle_cap", "button_battery", "end_cap", "eraser",
    "fire_hood", "mint", "mounts", "pcb", "phone_battery", "plastic_nut",
    "plastic_plug", "porcelain_doll", "regulator", "rolled_strip_base",
    "sim_card_set", "switch", "tape", "terminalblock", "toothbrush", "toy",
    "toy_brick", "transistor1", "u_block", "usb", "usb_adaptor", "vcpill",
    "wooden_beads", "woodstick", "zipper",
)
ROLE_NAMES = ("A", "B", "C")
ROLE_PURPOSES = {
    "A": "parameter-design reference",
    "B": "development evaluation",
    "C": "sealed confirmation",
}
ROLE_DISPOSITIONS = {"REQUIRED", "NOT_APPLICABLE", "PENDING_AUTHORIZATION"}
ACCESS_STATUSES = {
    "PENDING_AUTHORIZATION",
    "PROTECTED_SEALED",
    "UNAVAILABLE",
    "NOT_PRESENT",
    "AUTHORIZED_METADATA_ONLY",
    "AUTHORIZED_NORMAL_REFERENCE_ONLY",
}
GROUPING_UNIT = "official_sample_key"
# Preserve the exact identifier used by experiments.risk_safe.preflight.
IDENTITY_CONTRACT_VERSION = "realid-offical-sample-key-v2"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CATEGORY_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class InventoryContractError(ValueError):
    """An inventory or identity receipt violates the versioned contract."""


class PartitionOverlapError(InventoryContractError):
    """A source-group digest appears in more than one role."""


def _sha256(value: str, field: str) -> str:
    value = str(value)
    if not SHA256_RE.fullmatch(value):
        raise InventoryContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_source_uri(source_uri: str) -> None:
    if source_uri != PINNED_SOURCE_URI:
        raise InventoryContractError(
            "catalog receipt must cite the exact pinned official Tencent config file")


@dataclass(frozen=True)
class OfficialCatalogReceipt:
    """Projected catalog evidence; category names only, with no paths or labels."""

    source_uri: str
    source_sha256: str
    release_id: str
    categories: tuple[str, ...]
    projection: str = "canonical-category-identifiers-only"
    official_source_verified: bool = False
    non_label_bearing: bool = False
    contains_image_paths: bool = False
    contains_labels_or_masks: bool = False
    source_commit: str = ""
    source_path: str = ""
    dataset_config_version: str = ""
    extraction_algorithm: str = ""


@dataclass(frozen=True)
class RoleGroupReceipt:
    """Hashed source-group membership, without raw sample keys or image paths."""

    category_id: str
    role: str
    group_sha256: str
    receipt_sha256: str
    authorization_id: str
    grouping_unit: str = GROUPING_UNIT
    identity_contract_version: str = IDENTITY_CONTRACT_VERSION
    projection_only: bool = True
    contains_raw_sample_keys: bool = False
    contains_image_paths: bool = False
    contains_labels_or_masks: bool = False
    independent_certificate_sha256: str | None = None


@dataclass(frozen=True)
class RolePolicyEntry:
    """One explicit category-role disposition in a frozen experiment policy."""

    category_id: str
    role: str
    disposition: str
    rationale: str | None = None


@dataclass(frozen=True)
class RolePolicyContract:
    policy_id: str
    policy_version: str
    entries: tuple[RolePolicyEntry, ...]
    policy_sha256: str


@dataclass(frozen=True)
class AReferenceAvailabilityReceipt:
    """Externally reviewed count-only authorization receipt for one A category."""

    category_id: str
    authorized_normal_reference_image_count: int
    authorized_source_group_count: int
    receipt_sha256: str
    authorization_id: str
    population: str = "authorized_normal_reference_only"
    projection_only: bool = True
    contains_image_paths: bool = False
    contains_labels_or_masks: bool = False


def _parse_meta_file_list(lines: Sequence[str], section: str) -> tuple[str, ...]:
    section_pattern = re.compile(rf"^  {re.escape(section)}:\s*$")
    section_indices = [i for i, line in enumerate(lines) if section_pattern.fullmatch(line)]
    if len(section_indices) != 1:
        raise InventoryContractError(
            f"expected exactly one dataset.{section} section")
    start = section_indices[0]
    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.startswith(" ") and not line.startswith("#"):
            end = i
            break
        if line.startswith("  ") and not line.startswith("    "):
            end = i
            break
    block = lines[start + 1:end]
    meta_headers = [i for i, line in enumerate(block)
                    if re.fullmatch(r"    meta_file:\s*", line)]
    if len(meta_headers) != 1:
        raise InventoryContractError(
            f"expected exactly one dataset.{section}.meta_file list")
    categories: list[str] = []
    for line in block[meta_headers[0] + 1:]:
        if not line.strip():
            continue
        if line.startswith("    ") and not line.startswith("      "):
            break
        match = re.fullmatch(r"      - ([^\s#]+)\s*", line)
        if match is None:
            raise InventoryContractError(
                f"unexpected YAML item format in dataset.{section}.meta_file")
        path_match = re.fullmatch(
            r"data/Real-IAD/realiad_jsons/([a-z][a-z0-9_]*)\.json", match.group(1))
        if path_match is None:
            raise InventoryContractError(
                f"unexpected category JSON filename format in dataset.{section}.meta_file")
        categories.append(path_match.group(1))
    if not categories:
        raise InventoryContractError(f"dataset.{section}.meta_file list is empty")
    if len(categories) != len(set(categories)):
        raise InventoryContractError(f"duplicate category in dataset.{section}.meta_file")
    return tuple(categories)


def extract_category_catalog_from_config_bytes(
    source_bytes: bytes,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Strictly project version and train/test category IDs from pinned YAML bytes."""
    if not isinstance(source_bytes, bytes):
        raise InventoryContractError("pinned config must be provided as source bytes")
    try:
        text = source_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise InventoryContractError("pinned config is not valid UTF-8") from exc
    lines = text.splitlines()
    if any(re.match(r"^[ \t]*\t", line) for line in lines):
        raise InventoryContractError("tabs are not accepted in YAML indentation")
    version_matches = [re.fullmatch(r"version:\s*([^\s#]+)\s*", line)
                       for line in lines]
    versions = [match.group(1) for match in version_matches if match is not None]
    if len(versions) != 1:
        raise InventoryContractError("expected exactly one top-level dataset config version")
    dataset_headers = [i for i, line in enumerate(lines)
                       if re.fullmatch(r"dataset:\s*", line)]
    if len(dataset_headers) != 1:
        raise InventoryContractError("expected exactly one top-level dataset mapping")
    dataset_start = dataset_headers[0]
    dataset_end = len(lines)
    for i in range(dataset_start + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.startswith(" ") and not line.startswith("#"):
            dataset_end = i
            break
    dataset_lines = lines[dataset_start + 1:dataset_end]
    train = _parse_meta_file_list(dataset_lines, "train")
    test = _parse_meta_file_list(dataset_lines, "test")
    if len(train) != EXPECTED_CATEGORY_COUNT or len(test) != EXPECTED_CATEGORY_COUNT:
        raise InventoryContractError("pinned train and test catalogs must each contain 30 categories")
    if set(train) != set(test):
        raise InventoryContractError("pinned train and test category catalogs differ")
    if set(train) != set(CANONICAL_CATEGORY_IDS):
        raise InventoryContractError("pinned source categories differ from the independent cross-check")
    return versions[0], train, test


def official_catalog_from_pinned_source_bytes(
    source_bytes: bytes,
    *,
    source_uri: str = PINNED_SOURCE_URI,
    retrieval_uri: str = PINNED_RAW_SOURCE_URI,
) -> OfficialCatalogReceipt:
    """Verify exact pinned source bytes and return a category-only receipt."""
    if not isinstance(source_bytes, bytes):
        raise InventoryContractError("pinned config must be provided as source bytes")
    _validate_source_uri(source_uri)
    if retrieval_uri != PINNED_RAW_SOURCE_URI:
        raise InventoryContractError("source must be retrieved from the exact pinned raw URL")
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha256 != PINNED_SOURCE_SHA256:
        raise InventoryContractError(
            f"pinned source byte digest mismatch: expected {PINNED_SOURCE_SHA256}, "
            f"found {actual_sha256}")
    version, train, test = extract_category_catalog_from_config_bytes(source_bytes)
    if version != PINNED_CONFIG_VERSION:
        raise InventoryContractError(
            f"unexpected dataset config version: expected {PINNED_CONFIG_VERSION}, found {version}")
    if train != test:
        raise InventoryContractError("pinned train/test category ordering differs")
    return OfficialCatalogReceipt(
        source_uri=PINNED_SOURCE_URI,
        source_sha256=actual_sha256,
        release_id=version,
        categories=train,
        official_source_verified=True,
        non_label_bearing=True,
        source_commit=PINNED_COMMIT,
        source_path=PINNED_SOURCE_PATH,
        dataset_config_version=version,
        extraction_algorithm=CATALOG_EXTRACTION_ALGORITHM,
    )


def _validate_catalog_header(receipt: OfficialCatalogReceipt) -> None:
    if not receipt.official_source_verified:
        raise InventoryContractError("official catalog source is not verified")
    if not receipt.non_label_bearing or receipt.contains_image_paths or receipt.contains_labels_or_masks:
        raise InventoryContractError("catalog projection must contain category names only")
    if receipt.projection != "canonical-category-identifiers-only":
        raise InventoryContractError("unsupported catalog projection")
    if not receipt.release_id.strip():
        raise InventoryContractError("catalog receipt requires a dataset release identifier")
    _validate_source_uri(receipt.source_uri)
    _sha256(receipt.source_sha256, "catalog source digest")
    if receipt.source_sha256 != PINNED_SOURCE_SHA256:
        raise InventoryContractError("catalog receipt does not match pinned source content bytes")
    if (receipt.source_commit != PINNED_COMMIT or receipt.source_path != PINNED_SOURCE_PATH
            or receipt.dataset_config_version != PINNED_CONFIG_VERSION
            or receipt.extraction_algorithm != CATALOG_EXTRACTION_ALGORITHM
            or receipt.release_id != PINNED_CONFIG_VERSION):
        raise InventoryContractError("catalog receipt has incompatible pinned-source provenance")
    if len(receipt.categories) != EXPECTED_CATEGORY_COUNT:
        raise InventoryContractError("pinned official catalog receipt must contain exactly 30 categories")
    for category in receipt.categories:
        if not CATEGORY_RE.fullmatch(category):
            raise InventoryContractError(f"invalid canonical category identifier: {category!r}")
    if len(set(receipt.categories)) != len(receipt.categories):
        raise InventoryContractError("catalog contains duplicate category identifiers")
    if receipt.categories != CANONICAL_CATEGORY_IDS:
        raise InventoryContractError("catalog order or names differ from pinned source extraction")


def build_inventory_preflight(
    receipt: OfficialCatalogReceipt | None = None,
    *,
    category_access_status: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return all 30 census slots; unresolved names remain explicit placeholders.

    Placeholders are not category identities and never count as a verified
    inventory. This prevents a four-class or cache-dependent list being
    mistaken for a complete universe.
    """
    if receipt is None:
        categories: tuple[str, ...] = ()
        source: dict[str, Any] = {"status": "PENDING_AUTHORIZATION"}
    else:
        _validate_catalog_header(receipt)
        categories = receipt.categories
        source = {
            "status": "OFFICIAL_PUBLIC_CATALOG_RECEIPT",
            "source_uri": receipt.source_uri,
            "source_sha256": receipt.source_sha256,
            "release_id": receipt.release_id,
            "source_commit": receipt.source_commit,
            "source_path": receipt.source_path,
            "dataset_config_version": receipt.dataset_config_version,
            "extraction_algorithm": receipt.extraction_algorithm,
            "projection": receipt.projection,
        }
    access = dict(category_access_status or {})
    if receipt is None and access:
        raise InventoryContractError("category access statuses require an official catalog")
    if set(access) - set(categories):
        raise InventoryContractError("access status names a category outside the supplied catalog")
    if any(value not in ACCESS_STATUSES for value in access.values()):
        raise InventoryContractError("unrecognized category access status")

    complete = len(categories) == EXPECTED_CATEGORY_COUNT
    records: list[dict[str, Any]] = []
    for i, category in enumerate(categories, start=1):
        records.append({
            "slot": i,
            "category_id": category,
            "inventory_status": "VERIFIED_CATALOG_NAME" if complete
            else "CATALOG_ENTRY_UNVERIFIED_INCOMPLETE_UNIVERSE",
            "access_status": access.get(category, "PENDING_AUTHORIZATION"),
            "local_group_counts": None,
            "roles": {role: "PENDING_IDENTITY_RECEIPT" for role in ROLE_NAMES},
        })
    for i in range(len(categories) + 1, EXPECTED_CATEGORY_COUNT + 1):
        records.append({
            "slot": i,
            "category_id": None,
            "inventory_status": "PENDING_AUTHORIZATION_UNRESOLVED_CATEGORY_ID",
            "access_status": "PENDING_AUTHORIZATION",
            "local_group_counts": None,
            "roles": {role: "PENDING_IDENTITY_RECEIPT" for role in ROLE_NAMES},
        })

    return {
        "schema": "r2_l2a.inventory_preflight.v1",
        "expected_category_count": EXPECTED_CATEGORY_COUNT,
        "verified_category_count": len(categories) if complete else 0,
        "listed_category_count": len(categories),
        "complete_30_category_inventory": complete,
        "status": "VERIFIED_30_CATEGORY_PUBLIC_INVENTORY" if complete
        else "PENDING_AUTHORIZATION_INCOMPLETE_OFFICIAL_CATALOG",
        "catalog_source": source,
        "categories": records,
        "geometry_coverage_status": "NOT_RUN",
    }


def _validate_group_receipt(receipt: RoleGroupReceipt,
                            category_ids: set[str]) -> None:
    if receipt.category_id not in category_ids:
        raise InventoryContractError(
            f"group receipt category is outside official catalog: {receipt.category_id}")
    if receipt.role not in ROLE_NAMES:
        raise InventoryContractError(f"unknown data role: {receipt.role}")
    _sha256(receipt.group_sha256, "source-group digest")
    _sha256(receipt.receipt_sha256, "identity receipt digest")
    if receipt.role == "C":
        if receipt.independent_certificate_sha256 is None:
            raise InventoryContractError(
                "sealed C groups require an independent identity-certificate digest")
        _sha256(receipt.independent_certificate_sha256,
                "independent C identity certificate digest")
    if (receipt.grouping_unit != GROUPING_UNIT
            or receipt.identity_contract_version != IDENTITY_CONTRACT_VERSION):
        raise InventoryContractError("incompatible official source-group identity contract")
    if (not receipt.projection_only or receipt.contains_raw_sample_keys
            or receipt.contains_image_paths or receipt.contains_labels_or_masks):
        raise InventoryContractError(
            "role receipt must be a hashed identity-only projection without protected fields")
    if not receipt.authorization_id.strip():
        raise InventoryContractError("role receipt requires an authorization identifier")


def role_policy_digest(policy_id: str, policy_version: str,
                       entries: Sequence[RolePolicyEntry]) -> str:
    """Hash a frozen category-role disposition table in canonical order."""
    if not policy_id.strip() or not policy_version.strip():
        raise InventoryContractError("role policy requires an id and version")
    canonical_entries = sorted(
        (asdict(entry) for entry in entries),
        key=lambda entry: (entry["category_id"], entry["role"]),
    )
    payload = {
        "policy_id": policy_id,
        "policy_version": policy_version,
        "entries": canonical_entries,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def make_role_policy(policy_id: str, policy_version: str,
                     entries: Sequence[RolePolicyEntry]) -> RolePolicyContract:
    frozen_entries = tuple(entries)
    return RolePolicyContract(policy_id, policy_version, frozen_entries,
                              role_policy_digest(policy_id, policy_version, frozen_entries))


def _validate_role_policy(policy: RolePolicyContract,
                          category_order: Sequence[str]) -> dict[tuple[str, str], RolePolicyEntry]:
    expected = {(category, role) for category in category_order for role in ROLE_NAMES}
    actual: dict[tuple[str, str], RolePolicyEntry] = {}
    for entry in policy.entries:
        key = (entry.category_id, entry.role)
        if key not in expected:
            raise InventoryContractError("role policy includes an unknown category or role")
        if key in actual:
            raise InventoryContractError("role policy contains a duplicate category-role entry")
        if entry.disposition not in ROLE_DISPOSITIONS:
            raise InventoryContractError("role policy contains an unknown disposition")
        if entry.disposition == "NOT_APPLICABLE" and not (entry.rationale or "").strip():
            raise InventoryContractError("NOT_APPLICABLE requires a genuine written rationale")
        if entry.disposition != "NOT_APPLICABLE" and entry.rationale is not None:
            raise InventoryContractError("rationale is only permitted for NOT_APPLICABLE")
        actual[key] = entry
    if set(actual) != expected:
        raise InventoryContractError("role policy must explicitly cover every category and A/B/C role")
    digest = role_policy_digest(policy.policy_id, policy.policy_version, policy.entries)
    if _sha256(policy.policy_sha256, "role policy digest") != digest:
        raise InventoryContractError("role policy digest does not match its frozen contents")
    return actual


def _validate_a_reference_receipt(receipt: AReferenceAvailabilityReceipt,
                                  category_ids: set[str]) -> None:
    if receipt.category_id not in category_ids:
        raise InventoryContractError("A availability receipt category is outside official catalog")
    _sha256(receipt.receipt_sha256, "A reference availability receipt digest")
    if (receipt.authorized_normal_reference_image_count <= 0
            or receipt.authorized_source_group_count <= 0):
        raise InventoryContractError("A reference availability receipt requires positive counts")
    if receipt.population != "authorized_normal_reference_only":
        raise InventoryContractError("A availability population must contain authorized normals only")
    if (not receipt.authorization_id.strip() or not receipt.projection_only
            or receipt.contains_image_paths or receipt.contains_labels_or_masks):
        raise InventoryContractError("A availability receipt must be an authorized count-only projection")


def partition_preflight(
    catalog: OfficialCatalogReceipt | None,
    receipts: Sequence[RoleGroupReceipt] = (),
    *,
    role_policy: RolePolicyContract | None = None,
    a_reference_receipts: Sequence[AReferenceAvailabilityReceipt] = (),
    verified_role_policy_sha256s: Sequence[str] = (),
    verified_identity_receipt_sha256s: Sequence[str] = (),
    verified_c_certificate_sha256: str | None = None,
    verified_a_reference_receipt_sha256s: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate frozen role policy, identity-only receipts, and A authorization.

    This preflight never treats a missing identity as an empty role. NOT_APPLICABLE
    only counts when present in a complete policy whose digest has been reviewed.
    """
    inventory = build_inventory_preflight(catalog)
    if catalog is None or len(catalog.categories) != EXPECTED_CATEGORY_COUNT:
        return {
            "schema": "r2_l2a.partition_preflight.v1",
            "status": "PENDING_AUTHORIZATION_INCOMPLETE_INVENTORY",
            "role_policy_declared": role_policy is not None,
            "role_policy_externally_verified": False,
            "role_identity_receipts_complete": False,
            "role_disjointness_externally_certified": False,
            "authorized_a_data_available": False,
            "g1_geometry_complete": False,
            "certified_disjoint": False,
            "inventory_status": inventory["status"],
            "missing_category_roles": None,
            "role_group_counts": None,
            "overlap_count": None,
        }

    policy_map: dict[tuple[str, str], RolePolicyEntry] = {}
    policy_declared = role_policy is not None
    policy_externally_verified = False
    if role_policy is not None:
        policy_map = _validate_role_policy(role_policy, catalog.categories)
        verified_policies = {_sha256(value, "externally verified role policy digest")
                             for value in verified_role_policy_sha256s}
        policy_externally_verified = role_policy.policy_sha256 in verified_policies

    category_ids = set(catalog.categories)
    seen: dict[str, tuple[str, str]] = {}
    role_counts: dict[str, dict[str, int]] = {
        category: {role: 0 for role in ROLE_NAMES} for category in catalog.categories
    }
    for receipt in receipts:
        _validate_group_receipt(receipt, category_ids)
        prior = seen.get(receipt.group_sha256)
        if prior is not None and prior[0] != receipt.role:
            raise PartitionOverlapError(
                f"source group overlaps roles {prior[0]} and {receipt.role}")
        if prior is not None:
            raise InventoryContractError("duplicate source-group identity receipt")
        seen[receipt.group_sha256] = (receipt.role, receipt.category_id)
        role_counts[receipt.category_id][receipt.role] += 1

    if role_policy is not None:
        for receipt in receipts:
            disposition = policy_map[(receipt.category_id, receipt.role)].disposition
            if disposition == "NOT_APPLICABLE":
                raise InventoryContractError("role has identity receipts despite NOT_APPLICABLE policy")

    missing: list[dict[str, str]] = []
    pending_policy: list[dict[str, str]] = []
    if role_policy is None:
        pending_policy = [
            {"category_id": category, "role": role}
            for category in catalog.categories for role in ROLE_NAMES
        ]
    else:
        for category in catalog.categories:
            for role in ROLE_NAMES:
                disposition = policy_map[(category, role)].disposition
                if disposition == "PENDING_AUTHORIZATION":
                    pending_policy.append({"category_id": category, "role": role})
                elif disposition == "REQUIRED" and role_counts[category][role] == 0:
                    missing.append({"category_id": category, "role": role})

    verified_receipts = {_sha256(value, "verified identity receipt digest")
                         for value in verified_identity_receipt_sha256s}
    unverified_receipt_count = sum(
        receipt.receipt_sha256 not in verified_receipts for receipt in receipts)
    c_receipts = [receipt for receipt in receipts if receipt.role == "C"]
    c_certificate_verified = False
    c_required = (role_policy is not None and any(
        policy_map[(category, "C")].disposition == "REQUIRED"
        for category in catalog.categories))
    if verified_c_certificate_sha256 is not None:
        _sha256(verified_c_certificate_sha256, "verified independent C certificate digest")
        if any(receipt.independent_certificate_sha256 != verified_c_certificate_sha256
               for receipt in c_receipts):
            raise InventoryContractError(
                "C receipt does not match the independently verified certificate digest")
        c_certificate_verified = bool(c_receipts)
    if not c_required and policy_externally_verified:
        c_certificate_verified = True

    availability_by_category: dict[str, AReferenceAvailabilityReceipt] = {}
    for receipt in a_reference_receipts:
        _validate_a_reference_receipt(receipt, category_ids)
        if receipt.category_id in availability_by_category:
            raise InventoryContractError("duplicate A reference availability receipt")
        if role_policy is None or policy_map[(receipt.category_id, "A")].disposition != "REQUIRED":
            raise InventoryContractError("A availability receipt requires an A REQUIRED policy disposition")
        if receipt.authorized_source_group_count != role_counts[receipt.category_id]["A"]:
            raise InventoryContractError("A availability source-group count disagrees with A identity receipts")
        availability_by_category[receipt.category_id] = receipt
    verified_a = {_sha256(value, "externally verified A availability receipt digest")
                   for value in verified_a_reference_receipt_sha256s}
    a_status: dict[str, str] = {}
    for category in catalog.categories:
        disposition = policy_map[(category, "A")].disposition if role_policy else "PENDING_AUTHORIZATION"
        a_receipt = availability_by_category.get(category)
        if disposition == "NOT_APPLICABLE" and policy_externally_verified:
            a_status[category] = "NOT_AUTHORIZED_FOR_G1"
        elif (a_receipt is not None and a_receipt.receipt_sha256 in verified_a
              and policy_externally_verified):
            a_status[category] = "AUTHORIZED_FOR_G1_REFERENCE_POPULATION"
        else:
            a_status[category] = "PENDING_AUTHORIZATION"
    authorized_a_data_available = any(
        value == "AUTHORIZED_FOR_G1_REFERENCE_POPULATION" for value in a_status.values())

    all_role_dispositions_resolved = bool(role_policy) and not pending_policy
    role_identities_complete = (
        all_role_dispositions_resolved and policy_externally_verified and not missing)
    all_receipts_verified = unverified_receipt_count == 0
    if not all_role_dispositions_resolved:
        status = "PENDING_AUTHORIZATION_ROLE_POLICY"
    elif not policy_externally_verified:
        status = "PENDING_EXTERNAL_ROLE_POLICY_REVIEW"
    elif missing:
        status = "PENDING_AUTHORIZATION_ROLE_RECEIPTS"
    elif c_required and not c_certificate_verified:
        status = "PENDING_AUTHORIZATION_C_CERTIFICATE_REVIEW"
    elif not all_receipts_verified:
        status = "PENDING_EXTERNAL_RECEIPT_PROVENANCE_REVIEW"
    else:
        status = "PARTITION_CERTIFIED_DISJOINT"
    certified = role_identities_complete and all_receipts_verified and c_certificate_verified
    g1_complete = False  # Geometry has not run in this CPU-only inventory preflight.
    return {
        "schema": "r2_l2a.partition_preflight.v1",
        "status": status,
        "role_policy_declared": policy_declared,
        "role_policy_sha256": role_policy.policy_sha256 if role_policy else None,
        "role_policy_externally_verified": policy_externally_verified,
        "role_identity_receipts_complete": role_identities_complete,
        "role_disjointness_externally_certified": certified,
        "authorized_a_data_available": authorized_a_data_available,
        "a_reference_authorization_by_category": a_status,
        "g1_geometry_complete": g1_complete,
        "certified_disjoint": certified,
        "inventory_status": inventory["status"],
        "grouping_unit": GROUPING_UNIT,
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "role_purposes": dict(ROLE_PURPOSES),
        "role_group_counts": role_counts,
        "missing_category_roles": missing,
        "pending_policy_category_roles": pending_policy,
        "overlap_count": 0,
        "c_independent_certificate_required": True,
        "c_group_count": sum(counts["C"] for counts in role_counts.values()),
        "identity_receipt_count": len(receipts),
        "externally_verified_identity_receipt_count": len(receipts) - unverified_receipt_count,
        "unverified_identity_receipt_count": unverified_receipt_count,
        "identity_receipt_authentication": "external receipt digests supplied by caller",
        "c_certification_status": "EXTERNAL_CERTIFICATE_VERIFIED_BY_CALLER"
        if c_certificate_verified and c_required else
        "NOT_REQUIRED_BY_EXTERNALLY_VERIFIED_POLICY" if c_certificate_verified else
        "PENDING_AUTHORIZATION",
    }


def as_receipt_dict(receipt: OfficialCatalogReceipt | RoleGroupReceipt) -> Mapping[str, Any]:
    """Small serialization helper for text-only audit manifests."""
    return asdict(receipt)
