"""Governed inventory and A/B/C partition preflight for R2-L2A.

This module consumes explicit, non-label-bearing receipts. It never discovers
categories from image paths, feature-cache filenames, class JSONs, or labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


EXPECTED_CATEGORY_COUNT = 30
ROLE_NAMES = ("A", "B", "C")
ROLE_PURPOSES = {
    "A": "parameter-design reference",
    "B": "development evaluation",
    "C": "sealed confirmation",
}
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
    parsed = urlparse(source_uri)
    official = (
        parsed.scheme == "https"
        and (
            parsed.hostname in {"realiad4ad.github.io", "openaccess.thecvf.com",
                                "huggingface.co"}
            or (parsed.hostname == "github.com"
                and parsed.path.rstrip("/") == "/Tencent/AnomalyDetection_Real-IAD")
        )
    )
    if not official:
        raise InventoryContractError(
            "catalog receipt must cite an approved official HTTPS source")


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
    if len(receipt.categories) > EXPECTED_CATEGORY_COUNT:
        raise InventoryContractError("catalog has more than 30 canonical categories")
    for category in receipt.categories:
        if not CATEGORY_RE.fullmatch(category):
            raise InventoryContractError(f"invalid canonical category identifier: {category!r}")
    if len(set(receipt.categories)) != len(receipt.categories):
        raise InventoryContractError("catalog contains duplicate category identifiers")


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
        "status": "VERIFIED_30_CATEGORY_INVENTORY" if complete
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


def partition_preflight(
    catalog: OfficialCatalogReceipt | None,
    receipts: Sequence[RoleGroupReceipt] = (),
    *,
    verified_identity_receipt_sha256s: Sequence[str] = (),
    verified_c_certificate_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate role digests, reject any overlap, and expose missing coverage."""
    inventory = build_inventory_preflight(catalog)
    if catalog is None or len(catalog.categories) != EXPECTED_CATEGORY_COUNT:
        return {
            "schema": "r2_l2a.partition_preflight.v1",
            "status": "PENDING_AUTHORIZATION_INCOMPLETE_INVENTORY",
            "certified_disjoint": False,
            "inventory_status": inventory["status"],
            "missing_category_roles": None,
            "role_group_counts": None,
            "overlap_count": None,
        }

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

    verified_receipts = {_sha256(value, "verified identity receipt digest")
                         for value in verified_identity_receipt_sha256s}
    unverified_receipt_count = sum(
        receipt.receipt_sha256 not in verified_receipts for receipt in receipts)
    c_receipts = [receipt for receipt in receipts if receipt.role == "C"]
    c_certificate_verified = False
    if verified_c_certificate_sha256 is not None:
        _sha256(verified_c_certificate_sha256, "verified independent C certificate digest")
        if any(receipt.independent_certificate_sha256 != verified_c_certificate_sha256
               for receipt in c_receipts):
            raise InventoryContractError(
                "C receipt does not match the independently verified certificate digest")
        c_certificate_verified = bool(c_receipts)

    missing = [
        {"category_id": category, "role": role}
        for category in catalog.categories for role in ROLE_NAMES
        if role_counts[category][role] == 0
    ]
    all_roles_present = not missing
    all_receipts_verified = bool(receipts) and unverified_receipt_count == 0
    certified = all_roles_present and all_receipts_verified and c_certificate_verified
    if not all_roles_present:
        status = "PENDING_AUTHORIZATION_ROLE_RECEIPTS"
    elif not c_certificate_verified:
        status = "PENDING_AUTHORIZATION_C_CERTIFICATE_REVIEW"
    elif not all_receipts_verified:
        status = "PENDING_EXTERNAL_RECEIPT_PROVENANCE_REVIEW"
    else:
        status = "PARTITION_CERTIFIED_DISJOINT"
    return {
        "schema": "r2_l2a.partition_preflight.v1",
        "status": status,
        "certified_disjoint": certified,
        "inventory_status": inventory["status"],
        "grouping_unit": GROUPING_UNIT,
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "role_purposes": dict(ROLE_PURPOSES),
        "role_group_counts": role_counts,
        "missing_category_roles": missing,
        "overlap_count": 0,
        "c_independent_certificate_required": True,
        "c_group_count": sum(counts["C"] for counts in role_counts.values()),
        "identity_receipt_count": len(receipts),
        "externally_verified_identity_receipt_count": len(receipts) - unverified_receipt_count,
        "unverified_identity_receipt_count": unverified_receipt_count,
        "identity_receipt_authentication": "external receipt digests supplied by caller",
        "c_certification_status": "EXTERNAL_CERTIFICATE_VERIFIED_BY_CALLER"
        if c_certificate_verified else "PENDING_AUTHORIZATION",
    }


def as_receipt_dict(receipt: OfficialCatalogReceipt | RoleGroupReceipt) -> Mapping[str, Any]:
    """Small serialization helper for text-only audit manifests."""
    return asdict(receipt)
