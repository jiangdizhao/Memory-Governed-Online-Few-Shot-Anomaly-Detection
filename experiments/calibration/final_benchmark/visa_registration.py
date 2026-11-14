"""Identity-only, deterministic VisA 1cls P0 registration.

The registration never decodes TEST label or mask cells. Local locators are
written separately from the public, digest-only category bundles.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from . import identity


CATEGORIES = ("candle", "capsules", "cashew", "chewinggum", "fryum",
              "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4",
              "pipe_fryum")
EXPECTED_TEST_COUNTS = {
    "candle": 200, "capsules": 160, "cashew": 150, "chewinggum": 150,
    "fryum": 150, "macaroni1": 200, "macaroni2": 200, "pcb1": 200,
    "pcb2": 200, "pcb3": 201, "pcb4": 201, "pipe_fryum": 150,
}
EXPECTED_SNAPSHOT_SHA256 = "15535268a41c5c163b9906874fa148478f9da0bf9394659f448d320de778a7f2"
EXPECTED_SOURCE_ARTIFACTS_SHA256 = "5612ff8ec0b9ca5c1545b9f08aaba71b89f3b36bcaf8dc2e9fee82589949d029"


class VisaRegistrationError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_exclusive(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical(payload) + b"\n"
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise VisaRegistrationError(f"refusing to overwrite VisA registration artifact {path.name}") from None
    return hashlib.sha256(data).hexdigest()


def _private_locators(dataset_root: Path, identity_manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Reconstruct exact paths by identity while decoding only identity columns."""
    csv_path = dataset_root / identity.VISA_1CLS_SOURCE
    result: dict[str, dict[str, Any]] = {}
    try:
        with csv_path.open("rb") as stream:
            header_line = stream.readline().decode("utf-8-sig")
            header = next(csv.reader([header_line]))
            for original_position, line in enumerate(stream):
                if not line.strip():
                    continue
                projected = identity._parse_csv_projected_line(
                    line, header, train_label_allowed=False)
                image_key = projected.get("image", "").strip()
                category = projected.get("object", "").strip()
                split = projected.get("split", "").strip().lower()
                if split not in {"train", "test"} or not image_key or not category:
                    raise VisaRegistrationError("VisA source identity projection contains an invalid row")
                image_id = identity._image_identity_digest(identity._nfc_key(image_key))
                file_path = identity._safe_join(dataset_root, image_key, identity=image_id)
                record = next((row for row in identity_manifest["identity_records"]
                               if row["image_identity_sha256"] == image_id), None)
                if (record is None or record["category"] != category or record["split"] != split or
                        int(record["original_position"]) != original_position):
                    raise VisaRegistrationError("VisA locator row differs from the registered identity census")
                digest, size = identity._safe_file_hash(file_path, identity=image_id)
                if digest != record["image_content_sha256"] or size != record["image_bytes"]:
                    raise VisaRegistrationError("VisA image bytes differ from the registered local snapshot")
                result[image_id] = {
                    "dataset_root": str(dataset_root.resolve()),
                    "image_path": str(file_path),
                    "relative_id": file_path.resolve().relative_to(dataset_root.resolve()).as_posix(),
                    "official_image_key": identity._nfc_key(image_key),
                    "category": category, "split": split,
                    "original_position": original_position,
                }
    except OSError as exc:
        raise VisaRegistrationError(f"VisA identity table is unavailable ({type(exc).__name__})") from None
    if set(result) != {row["image_identity_sha256"] for row in identity_manifest["identity_records"]}:
        raise VisaRegistrationError("VisA private locator coverage differs from the registered identity census")
    return result


def build_visa_registration(*, dataset_root: str | Path,
                            identity_manifest_path: str | Path,
                            output_root: str | Path,
                            researcher_decision_sha256: str,
                            source_commit: str) -> dict[str, Any]:
    """Revalidate the registered snapshot and emit all 12 outcome-blind P0 schedules."""
    root, identity_path, out = Path(dataset_root).resolve(strict=True), Path(identity_manifest_path), Path(output_root)
    if out.exists():
        raise VisaRegistrationError("VisA registration output must be a fresh non-existing directory")
    try:
        registered = json.loads(identity_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VisaRegistrationError("existing identity-only VisA registration is unreadable") from exc
    if (registered.get("dataset_id") != "visa_1cls" or
            registered.get("local_snapshot_sha256") != EXPECTED_SNAPSHOT_SHA256 or
            registered.get("source_artifacts_sha256") != EXPECTED_SOURCE_ARTIFACTS_SHA256):
        raise VisaRegistrationError("existing VisA census does not match the registered local snapshot")
    identity.validate_identity_manifest(registered)
    fresh = identity.build_visa_1cls_identity_manifest(root, workers=8)
    for key in ("local_snapshot_sha256", "source_artifacts_sha256", "source_artifacts",
                "identity_records", "categories", "duplicate_content_summary"):
        if fresh.get(key) != registered.get(key):
            raise VisaRegistrationError(f"current VisA snapshot differs from registered field {key}")
    if (fresh["local_snapshot_sha256"] != EXPECTED_SNAPSHOT_SHA256 or
            fresh["source_artifacts_sha256"] != EXPECTED_SOURCE_ARTIFACTS_SHA256 or
            sum(row["test_image_event_count"] for row in fresh["categories"]) != 2162 or
            [row["category"] for row in fresh["categories"]] != list(CATEGORIES)):
        raise VisaRegistrationError("current VisA identity census differs from the 12-category contract")
    if len(fresh["duplicate_content_summary"]["exact_duplicate_content_sets"]):
        raise VisaRegistrationError("registered VisA no-duplicate condition changed; explicit clustering review required")
    if len(str(researcher_decision_sha256)) != 64 or any(
            char not in "0123456789abcdef" for char in str(researcher_decision_sha256)):
        raise VisaRegistrationError("researcher decision source digest is malformed")
    if len(str(source_commit)) != 40 or any(char not in "0123456789abcdef" for char in str(source_commit)):
        raise VisaRegistrationError("VisA registration requires the committed implementation's Git object ID")
    locators = _private_locators(root, registered)
    release_identifier = f"visa-1cls-local-snapshot-v1:{registered['local_snapshot_sha256']}"
    release_sha = hashlib.sha256(("Gate0-local-release-v1\0visa_1cls\0" + release_identifier).encode("ascii")).hexdigest()
    out.mkdir(parents=True, exist_ok=False)
    (out / "private").mkdir()
    _write_exclusive(out / "private" / "researcher_decision_binding.json", {
        "schema": "final-benchmark.visa-researcher-decision-binding.v1",
        "researcher_decision_sha256": researcher_decision_sha256,
        "authorization_source": "explicit researcher task authorization",
        "scope": ["12 category P0 metadata registrations", "cashew/P0 Stage A",
                  "cashew/P0 post-seal Stage B"],
        "publisher_authentication": False,
        "independent_review_claimed": False,
        "pixel_mask_access": False,
    })
    _write_exclusive(out / "private" / "identity_manifest.private.json", registered)
    files: list[dict[str, Any]] = []
    test_counts: dict[str, int] = {}
    for category in CATEGORIES:
        role = identity.prepare_visa_role_manifest(
            registered, dataset_release_sha256=release_sha, category=category, support_seed=0)
        if isinstance(role["support_seed"], bool) or role["support_seed"] != 0:
            raise VisaRegistrationError("VisA Support Seed must be exactly integer zero")
        rows = [row for row in registered["identity_records"] if row["category"] == category]
        selected_by_id = {row["image_identity_sha256"]: row for row in role["selected"]}
        identity_rows = []
        unit_by_image: dict[str, str] = {}
        test_rows = [row for row in rows if row["split"] == "test"]
        test_unit_by_content: dict[str, str] = {}
        content_counts: dict[str, int] = {}
        for row in test_rows:
            content_counts[row["image_content_sha256"]] = content_counts.get(row["image_content_sha256"], 0) + 1
        for row in rows:
            image_id = row["image_identity_sha256"]
            if row["split"] == "test":
                role_name = "TEST"
                if content_counts[row["image_content_sha256"]] > 1:
                    unit = test_unit_by_content.setdefault(row["image_content_sha256"], identity._sha256_text(
                        "visa-exact-duplicate-cluster-v1\0" + release_sha + "\0" + category + "\0" +
                        row["image_content_sha256"]))
                else:
                    unit = identity.visa_image_unit_sha256(release_sha, category, image_id)
            elif image_id in selected_by_id:
                role_name = selected_by_id[image_id]["role"]
                unit = selected_by_id[image_id]["source_group_sha256"]
            else:
                continue
            identity_rows.append({"image_identity_sha256": image_id,
                                  "image_content_sha256": row["image_content_sha256"],
                                  "role": role_name})
            unit_by_image[image_id] = unit
        ordered_units = sorted({unit_by_image[row["image_identity_sha256"]] for row in test_rows},
            key=lambda unit: (identity._domain_hash("Gate0-stream-v1", release_sha, category, "0", unit),
                              unit.encode("ascii")))
        candidate = identity.prepare_visa_exception_candidate(
            identity_rows=identity_rows, proposed_unit_by_image=unit_by_image,
            proposed_test_unit_order=ordered_units)
        stream_by_unit = {unit: [] for unit in ordered_units}
        for row in test_rows:
            image_id = row["image_identity_sha256"]
            unit = unit_by_image[image_id]
            stream_by_unit[unit].append(row)
        stream_records: list[dict[str, Any]] = []
        for unit in ordered_units:
            for row in sorted(stream_by_unit[unit], key=lambda value: value["image_identity_sha256"]):
                position = len(stream_records)
                image_id = row["image_identity_sha256"]
                kind = "VISA_SINGLE_IMAGE" if len(stream_by_unit[unit]) == 1 else "VISA_EXACT_DUPLICATE_CLUSTER"
                event_id = identity._domain_hash("Gate0-visa-event-v1", release_sha, category, "0", unit,
                                                 image_id, str(row["original_position"]))
                stream_records.append({
                    "dataset_id": "visa_1cls", "category": category, "role": "TEST",
                    "position": position, "original_position": int(row["original_position"]),
                    "image_identity_sha256": image_id,
                    "image_content_sha256": row["image_content_sha256"],
                    "source_group_sha256": unit, "atomic_unit_sha256": unit,
                    "atomic_unit_kind": kind, "event_id_sha256": event_id,
                })
        if len(stream_records) != EXPECTED_TEST_COUNTS[category]:
            raise VisaRegistrationError(f"VisA {category} TEST count differs from the registered 1cls count")
        stream = {"schema": "final-benchmark.visa-p0-stream.v1", "dataset_id": "visa_1cls",
                  "category": category, "support_seed": 0, "stream_seed": 0,
                  "dataset_release_sha256": release_sha,
                  "source_commit": source_commit,
                  "identity_manifest_sha256": _sha(registered),
                  "role_manifest_sha256": _sha(role),
                  "unit_candidate_sha256": candidate["candidate_sha256"],
                  "test_image_count": len(stream_records), "unit_count": len(ordered_units),
                  "stream_order": stream_records}
        stream["stream_manifest_sha256"] = _sha(stream)
        release_identity = {
            "schema": "final-benchmark.dataset-release-identity.v1",
            "dataset_id": "visa_1cls", "status": "RESEARCHER_APPROVED_LOCAL_SNAPSHOT",
            "provenance_kind": "RESEARCHER_APPROVED_LOCAL_SNAPSHOT",
            "local_snapshot_identifier": release_identifier,
            "local_snapshot_sha256": registered["local_snapshot_sha256"],
            "dataset_release_sha256": release_sha,
            "source_artifacts_sha256": registered["source_artifacts_sha256"],
            "identity_manifest_sha256": _sha(registered),
            "role_manifest_sha256": _sha(role),
            "researcher_decision_sha256": researcher_decision_sha256,
            "official_release_sha256": None,
            "publisher_authenticated": False,
            "source_commit": source_commit,
        }
        _write_exclusive(out / category / "role_manifest.json", role)
        _write_exclusive(out / category / "release_identity.json", release_identity)
        _write_exclusive(out / category / "unit_candidate.json", candidate)
        _write_exclusive(out / category / "p0_stream_manifest.json", stream)
        private_category = out / "private" / category
        private_category.mkdir()
        locator_ids = set(selected_by_id) | {row["image_identity_sha256"] for row in test_rows}
        private = {image_id: {**locators[image_id],
                    "image_content_sha256": next(r["image_content_sha256"] for r in rows
                                                  if r["image_identity_sha256"] == image_id)}
                   for image_id in sorted(locator_ids)}
        _write_exclusive(private_category / "locators.private.json", private)
        category_index = {"category": category, "test_images": len(stream_records),
            "test_units": len(ordered_units), "role_manifest_sha256": _sha(role),
            "release_identity_sha256": _sha(release_identity),
            "unit_candidate_sha256": candidate["candidate_sha256"],
            "stream_manifest_sha256": stream["stream_manifest_sha256"],
            "role_manifest_path": f"{category}/role_manifest.json",
            "release_identity_path": f"{category}/release_identity.json",
            "unit_candidate_path": f"{category}/unit_candidate.json",
            "stream_manifest_path": f"{category}/p0_stream_manifest.json",
            "private_locator_path": f"private/{category}/locators.private.json"}
        files.append(category_index)
        test_counts[category] = len(stream_records)
    index = {"schema": "final-benchmark.visa-registration-index.v1",
        "dataset_id": "visa_1cls", "source_commit": source_commit,
        "dataset_root_is_local_only": True,
        "dataset_snapshot_sha256": registered["local_snapshot_sha256"],
        "source_artifacts_sha256": registered["source_artifacts_sha256"],
        "identity_manifest_sha256": _sha(registered), "dataset_release_sha256": release_sha,
        "categories": files, "test_counts": test_counts,
        "total_test_images": sum(test_counts.values()),
        "physical_independence_claim": "NOT_ESTABLISHED",
        "publisher_authentication": False}
    index["registration_sha256"] = _sha(index)
    _write_exclusive(out / "registration_index.json", index)
    _write_exclusive(out / "private" / "locators.private.json", {
        image_id: locators[image_id] for image_id in sorted(locators)})
    return index
