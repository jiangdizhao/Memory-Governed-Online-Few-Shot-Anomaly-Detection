"""Offline, label-blind FS4 registration preparation; import performs no I/O.

Only a separately authorized caller may provide real input registrations.  This
task uses synthetic images exclusively.  The private mapping must never be
supplied to the Stage A worker.
"""
from __future__ import annotations

import copy
import hashlib
import io
import secrets
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from .fs4_durable_ledger import canonical, digest
from .fs4_r2_context import _read_registered_image, validate_execution_manifest
from .fs4_r2_security import WindowsPrivateStore


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_blinded_registration(*, original: Mapping[str, Any], original_ledger: Mapping[str, Any],
                                  labels_by_position: Mapping[int, int], private_root: Path,
                                  image_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze independent opaque IDs and lossless resources in fresh stores.

    The caller's original registration supplies all events and their order;
    labels are checked only after the ordered event set has been established.
    """
    from experiments.risk_safe.preflight import parse_image_name
    original = validate_execution_manifest(original, original_ledger)
    private_root, image_root = Path(private_root).absolute(), Path(image_root).absolute()
    source_root = Path(original["dataset_root"]).absolute()
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for a, b in
           ((private_root, image_root), (private_root, source_root), (image_root, source_root))):
        raise ValueError("private mapping, source and blinded resource stores must be separate")
    ordered = [(category, item) for category in original["category_order"]
               for item in original["categories"][category]["events"]]
    if (set(labels_by_position) != {row["position"] for _, row in ordered} or
            any(type(value) is not int or value not in (0, 1) for value in labels_by_position.values())):
        raise ValueError("one binary label is required for each registered position")
    private_store = WindowsPrivateStore(private_root, create=True)
    image_store = WindowsPrivateStore(image_root, create=True)
    private_store.write("INCOMPLETE.json", canonical({"status": "INCOMPLETE"}) + b"\n")
    image_store.write("INCOMPLETE.json", canonical({"status": "INCOMPLETE"}) + b"\n")
    group_ids: dict[str, str] = {}
    seen_ids: set[str] = set()
    views: dict[tuple[str, str], set[int]] = {}
    mapped: list[dict[str, Any]] = []
    categories = {category: {"supports": copy.deepcopy(original["categories"][category]["supports"]), "events": []}
                  for category in original["category_order"]}

    def fresh() -> str:
        value = secrets.token_hex(32)
        if value in seen_ids:
            raise ValueError("opaque identifier collision")
        seen_ids.add(value)
        return value

    try:
        for category, row in ordered:
            path, image = _read_registered_image(Path(original["dataset_root"]), row, category=category)
            parsed = parse_image_name(path, expected_class=category)
            label = labels_by_position[row["position"]]
            if label != int(parsed.capture_session != "OK"):
                raise ValueError("original registration label is inconsistent")
            key = (category, parsed.official_sample_key)
            view_set = views.setdefault(key, set())
            if parsed.camera_id in view_set:
                raise ValueError("duplicate source group camera")
            view_set.add(parsed.camera_id)
            group = group_ids.setdefault(row["group"], fresh()) if row["group"] not in group_ids else group_ids[row["group"]]
            event, resource = fresh(), fresh() + ".png"
            pixels = image.tobytes()
            output = io.BytesIO()
            # Recreate RGB pixels in a fresh image so source EXIF/ICC or other
            # metadata can never be copied into the public resource.
            Image.frombytes("RGB", image.size, pixels).save(output, format="PNG")
            encoded = output.getvalue()
            with Image.open(io.BytesIO(encoded)) as decoded:
                if decoded.info or decoded.mode != "RGB" or decoded.size != image.size or decoded.tobytes() != pixels:
                    raise ValueError("blinded image pixels or metadata differ")
            image_store.write(resource, encoded)
            public = {"group": group, "event": event, "resource": resource, "position": row["position"],
                      "view": parsed.camera_id, "width": image.width, "height": image.height,
                      "content_sha256": row["content_sha256"], "resource_sha256": _sha(encoded),
                      "pixel_sha256": _sha(pixels)}
            categories[category]["events"].append(public)
            mapped.append({"category": category, "position": row["position"], "event": event, "group": group,
                           "resource": resource, "view": parsed.camera_id, "original_event": row["event"],
                           "original_group": parsed.official_sample_key, "original_path": str(path), "label": label,
                           "content_sha256": row["content_sha256"], "resource_sha256": _sha(encoded),
                           "pixel_sha256": _sha(pixels), "width": image.width, "height": image.height})
        if any(value != {1, 2, 3, 4, 5} for value in views.values()):
            raise ValueError("each source group must have exactly five camera views")
        private = {"schema": "FS4_R4_PRIVATE_MAPPING_V1", "salt": fresh(), "events": mapped}
        private_bytes = canonical(private) + b"\n"
        mapping_sha = _sha(private_bytes)
        execution = {key: copy.deepcopy(original[key]) for key in ("run_id", "protocol", "source_commit", "query_seed",
                     "model_state_sha256", "checkpoint_sha256", "hub_source_tree_sha256", "scorer_config_sha256",
                     "preflight_receipt_sha256", "dataset_root", "category_order")}
        execution.update(schema="FS4_R4_BLINDED_EXECUTION_V1", image_root=str(image_root), categories=categories,
                         source_mapping_sha256=mapping_sha)
        execution["dataset_identity_sha256"] = digest({"dataset_root": execution["dataset_root"],
            "image_root": execution["image_root"], "category_order": execution["category_order"], "categories": categories,
            "source_mapping_sha256": mapping_sha})
        receipt = {"schema": "FS4_R4_PREPARATION_RECEIPT_V1", "mapping_sha256": mapping_sha,
                   "execution_sha256": digest(execution), "event_count": len(mapped)}
        private_store.write("mapping.json", private_bytes)
        image_store.write("registration.json", canonical(execution) + b"\n")
        image_store.write("receipt.json", canonical(receipt) + b"\n")
        private_store.write("COMPLETE.json", canonical({"mapping_sha256": mapping_sha}) + b"\n")
        image_store.write("COMPLETE.json", canonical({"receipt_sha256": digest(receipt)}) + b"\n")
        return execution, receipt
    except Exception:
        # Never include a private original path or parsed defect code in an error.
        raise ValueError("blinded preparation failed; partial output remains INCOMPLETE") from None


def blinded_ledger_manifest(original_ledger: Mapping[str, Any], execution: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the public registration while retaining original Support identities."""
    keys = ("run_id", "source_commit", "source_fingerprints", "expected_events_by_category", "profile",
            "scientific_contract_sha256", "model_state_sha256", "support_groups_by_category",
            "fixed_support_sha256_by_category", "query_seed", "protocol")
    result = {key: copy.deepcopy(original_ledger[key]) for key in keys}
    if result["profile"] == "COUPLED_QUERY_ONLY_FROZEN_WRITER":
        result["controller"] = original_ledger["controller"]
    result["execution_scope"] = "TRUSTED_LOCAL_RESEARCH"
    result["research_execution_manifest"] = copy.deepcopy(dict(execution))
    result["dataset_identity_sha256"] = execution["dataset_identity_sha256"]
    result["stream_sha256_by_category"] = {category: digest([
        {"group": row["group"], "event": row["event"]} for row in execution["categories"][category]["events"]])
        for category in execution["category_order"]}
    return result
