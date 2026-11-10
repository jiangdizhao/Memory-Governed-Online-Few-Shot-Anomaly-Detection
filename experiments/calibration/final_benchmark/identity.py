"""Identity-only Gate 1 dataset inspection and frozen role-hash helpers.

TEST labels and masks are deliberately absent from the returned records. Image
files are opened only as opaque byte streams for SHA-256; this module never
decodes image pixels or evaluates a model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


REAL_IAD_CATEGORY_COUNT = 30
VISA_1CLS_CATEGORY_COUNT = 12
VISA_1CLS_SOURCE = "split_csv/1cls.csv"
VISA_NORMAL_TRAIN_LABEL = "normal"
_CAMERA_RE = re.compile(r"_C(?P<camera>[1-5])_\d{14}\.(?:jpg|jpeg|png)$", re.IGNORECASE)
_VISA_GROUP_FIELDS = frozenset({"official_group_id", "source_group_id", "object_group_id"})
_VISA_OUTCOME_FIELDS = frozenset({"label", "mask", "mask_path", "ground_truth", "gt"})


class IdentityAuditError(RuntimeError):
    """A fail-closed error that never includes a raw image path or TEST label."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _nfc_key(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise IdentityAuditError("empty or non-text official identity key")
    return unicodedata.normalize("NFC", value)


def _domain_hash(domain: str, *fields: str) -> str:
    parts = [domain, *(_nfc_key(x) for x in fields)]
    return _sha256_text("\0".join(parts))


def _source_group_digest(official_group_key: str) -> str:
    # Reuse the established official-sample digest domain used by Stage 05/R7.
    return _sha256_text("official-sample-group\0" + _nfc_key(official_group_key))


def _image_identity_digest(official_image_key: str) -> str:
    # The key is NFC-normalized only. Path separators and components are not
    # rewritten for the frozen selection/event hashes.
    return _sha256_text("image-identity\0" + _nfc_key(official_image_key))


def _safe_file_hash(path: Path, *, identity: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
    except OSError as exc:
        # Do not echo a potentially label-bearing path.
        raise IdentityAuditError(f"opaque file {identity} is unavailable ({type(exc).__name__})") from None
    return digest.hexdigest(), size


def _safe_join(root: Path, *relative_parts: str, identity: str) -> Path:
    parts: list[str] = []
    for raw in relative_parts:
        if not isinstance(raw, str) or not raw:
            raise IdentityAuditError(f"{identity}: missing relative image locator")
        normalized_for_io = raw.replace("\\", "/")
        if normalized_for_io.startswith("/") or re.match(r"^[A-Za-z]:", normalized_for_io):
            raise IdentityAuditError(f"{identity}: absolute image locator is forbidden")
        candidate_parts = [p for p in normalized_for_io.split("/") if p not in ("", ".")]
        if any(p == ".." for p in candidate_parts):
            raise IdentityAuditError(f"{identity}: image locator escapes the dataset root")
        parts.extend(candidate_parts)
    candidate = root.joinpath(*parts)
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise IdentityAuditError(f"{identity}: image locator is missing or escapes the dataset root") from None
    if not resolved.is_file():
        raise IdentityAuditError(f"{identity}: image locator is not a regular file")
    return resolved


def _skip_json_value_opaque(data: bytes, offset: int) -> int:
    """Find a value boundary without decoding strings or nested object keys."""
    from experiments.calibration import r7_shuffled_correction_4class as r7
    offset = r7._ws(data, offset)
    if offset >= len(data):
        raise IdentityAuditError("truncated annotation value")
    first = data[offset]
    if first == ord('"'):
        cursor, escaped = offset + 1, False
        while cursor < len(data):
            byte = data[cursor]
            if byte == ord('"') and not escaped:
                return cursor + 1
            escaped = byte == ord("\\") and not escaped
            cursor += 1
        raise IdentityAuditError("unterminated annotation string")
    if first in (ord("{"), ord("[")):
        stack, cursor, in_string, escaped = [first], offset + 1, False, False
        while cursor < len(data):
            byte = data[cursor]
            if in_string:
                if byte == ord('"') and not escaped:
                    in_string = False
                escaped = byte == ord("\\") and not escaped
            elif byte == ord('"'):
                in_string, escaped = True, False
            elif byte in (ord("{"), ord("[")):
                stack.append(byte)
            elif byte in (ord("}"), ord("]")):
                expected = ord("}") if stack[-1] == ord("{") else ord("]")
                if byte != expected:
                    raise IdentityAuditError("mismatched annotation delimiters")
                stack.pop()
                if not stack:
                    return cursor + 1
            cursor += 1
        raise IdentityAuditError("unterminated annotation container")
    end = offset
    while end < len(data) and data[end] not in b",]} \t\r\n":
        end += 1
    if end == offset:
        raise IdentityAuditError("empty annotation scalar")
    return end


def _identity_object_fields(data: bytes, start: int, stop: int) -> dict[str, tuple[int, int]]:
    from experiments.calibration import r7_shuffled_correction_4class as r7
    cursor = r7._ws(data, start)
    if cursor >= stop or data[cursor] != ord("{"):
        raise IdentityAuditError("expected an annotation object")
    cursor += 1
    fields: dict[str, tuple[int, int]] = {}
    while True:
        cursor = r7._ws(data, cursor)
        if cursor >= stop:
            raise IdentityAuditError("truncated annotation object")
        if data[cursor] == ord("}"):
            return fields
        key, cursor = r7._string(data, cursor)
        cursor = r7._ws(data, cursor)
        if cursor >= stop or data[cursor] != ord(":") or key in fields:
            raise IdentityAuditError("malformed or duplicate annotation field")
        begin = r7._ws(data, cursor + 1)
        end = _skip_json_value_opaque(data, begin)
        fields[key] = (begin, end)
        cursor = r7._ws(data, end)
        if cursor < stop and data[cursor] == ord(","):
            cursor += 1
            continue
        if cursor < stop and data[cursor] == ord("}"):
            return fields
        raise IdentityAuditError("malformed annotation object separator")


def _identity_only_realiad_sections(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Project JSON into metadata plus TRAIN normal-eligibility and TEST IDs.

    The boundary scanner skips unselected values without decoding them. TRAIN
    anomaly_class is decoded only for normal eligibility; TEST outcomes and all
    mask values are never passed to a JSON value decoder.
    """
    from experiments.calibration import r7_shuffled_correction_4class as r7
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise IdentityAuditError(f"category annotation unavailable ({type(exc).__name__})") from None
    try:
        if r7._ws(data, _skip_json_value_opaque(data, 0)) != len(data):
            raise IdentityAuditError("Real-IAD annotation has trailing data")
        top = _identity_object_fields(data, 0, len(data))
        if set(top) != {"meta", "train", "test"}:
            raise IdentityAuditError("Real-IAD annotation sections are incomplete")
        meta_bounds = top["meta"]
        meta_fields = _identity_object_fields(data, *meta_bounds)
        def decode_text(bounds: tuple[int, int]) -> str:
            if data[bounds[0]] != ord('"'):
                raise IdentityAuditError("permitted identity field must be a JSON string")
            value = r7._decode_field(data, bounds)
            if not isinstance(value, str):
                raise IdentityAuditError("permitted identity field must decode to text")
            return value

        meta = {key: decode_text(meta_fields[key])
                for key in ("prefix", "normal_class", "category", "dataset") if key in meta_fields}

        def project_array(bounds: tuple[int, int], allowed: tuple[str, ...]) -> list[dict[str, Any]]:
            start, stop = bounds
            cursor = r7._ws(data, start)
            if cursor >= stop or data[cursor] != ord("["):
                raise IdentityAuditError("Real-IAD identity section is not an array")
            cursor += 1
            rows = []
            while True:
                cursor = r7._ws(data, cursor)
                if cursor >= stop:
                    raise IdentityAuditError("Real-IAD identity array is truncated")
                if data[cursor] == ord("]"):
                    if cursor + 1 != stop:
                        raise IdentityAuditError("Real-IAD identity array has trailing data")
                    return rows
                end = _skip_json_value_opaque(data, cursor)
                fields = _identity_object_fields(data, cursor, end)
                rows.append({key: decode_text(fields[key]) for key in allowed if key in fields})
                cursor = r7._ws(data, end)
                if cursor < stop and data[cursor] == ord(","):
                    cursor += 1
                    continue
                if cursor >= stop or data[cursor] != ord("]"):
                    raise IdentityAuditError("Real-IAD identity array separator is malformed")

        train = project_array(top["train"], ("image_path", "category", "anomaly_class"))
        test = project_array(top["test"], ("image_path", "category"))
        return meta, train, test
    except IdentityAuditError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, r7.R7Error) as exc:
        raise IdentityAuditError(f"Real-IAD identity projection failed ({type(exc).__name__})") from None


def _realiad_identity_key(category: str, official_image_key: str) -> tuple[str, int]:
    """Return official sample key and camera without interpreting session text."""
    from experiments.risk_safe.preflight import OFFICIAL_SAMPLE_KEY_RE

    key = _nfc_key(official_image_key)
    basename = key.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    official = OFFICIAL_SAMPLE_KEY_RE.search(basename)
    camera = _CAMERA_RE.search(basename)
    if official is None or camera is None:
        raise IdentityAuditError(f"{category}: image identity does not match the official key/camera grammar")
    group_key = _nfc_key(official.group("sample_key"))
    if not group_key.startswith(category + "_"):
        raise IdentityAuditError(f"{category}: official key/category mismatch")
    return group_key, int(camera.group("camera"))


def _hash_many(paths: Sequence[tuple[Path, str]], *, workers: int = 4) -> dict[str, tuple[str, int]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    results: dict[str, tuple[str, int]] = {}
    batch_size = max(workers, workers * 8)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(paths), batch_size):
            batch = paths[start:start + batch_size]
            futures = [(identity, pool.submit(_safe_file_hash, path, identity=identity))
                       for path, identity in batch]
            for identity, future in futures:
                results[identity] = future.result()
    return results


def _record_for_identity(*, dataset_id: str, category: str, split: str,
                          official_image_key: str, group_key: str | None,
                          camera_id: int | None, original_position: int,
                          local_locator: str, content_sha256: str,
                          byte_count: int, train_normal_eligible: bool | None) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "category": _nfc_key(category),
        "split": split,
        "original_position": int(original_position),
        "image_identity_sha256": _image_identity_digest(official_image_key),
        "file_locator_sha256": local_locator,
        "image_content_sha256": content_sha256,
        "image_bytes": int(byte_count),
        "source_group_sha256": _source_group_digest(group_key) if group_key is not None else None,
        "camera_id": int(camera_id) if camera_id is not None else None,
        "train_normal_eligible": train_normal_eligible,
    }


def _duplicate_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_content: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_content[str(row["image_content_sha256"])].append(row)
    duplicate_sets = []
    cross_split = 0
    for digest, rows in sorted(by_content.items()):
        if len(rows) < 2:
            continue
        splits = sorted({str(row["split"]) for row in rows})
        cross_split += int(len(splits) > 1)
        duplicate_sets.append({
            "content_sha256": digest,
            "occurrences": len(rows),
            "split_names": splits,
            "image_identity_sha256": sorted(str(row["image_identity_sha256"]) for row in rows),
        })
    return {"exact_duplicate_content_sets": duplicate_sets,
            "exact_duplicate_content_set_count": len(duplicate_sets),
            "cross_split_duplicate_content_set_count": cross_split}


def _snapshot_digest(dataset_id: str, source_artifacts: Sequence[Mapping[str, Any]],
                     records: Sequence[Mapping[str, Any]]) -> str:
    identity_rows = sorted(({
        "category": row["category"], "split": row["split"],
        "image_identity_sha256": row["image_identity_sha256"],
        "image_content_sha256": row["image_content_sha256"],
        "source_group_sha256": row["source_group_sha256"],
        "camera_id": row["camera_id"],
    } for row in records), key=lambda row: (row["category"], row["split"], row["image_identity_sha256"]))
    payload = {"dataset_id": dataset_id,
               "source_artifacts": sorted(source_artifacts, key=lambda item: item["artifact_id"]),
               "identity_rows": identity_rows}
    return _sha256_bytes(_canonical_json(payload))


def _select_role_groups(*, dataset_id: str, category: str, support_seed: int,
                        dataset_release_sha256: str,
                        train_normal_rows: Sequence[Mapping[str, Any]],
                        test_group_keys: set[str]) -> dict[str, Any]:
    """Apply the exact frozen Gate0 role and image-selection hash domains."""
    digest = str(dataset_release_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise IdentityAuditError("dataset release digest is missing or malformed")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in train_normal_rows:
        group_key = str(row["_official_group_key"])
        grouped[group_key].append(row)
    if len(grouped) < 68:
        raise IdentityAuditError(f"{category}: fewer than 68 eligible official TRAIN-normal groups")
    if set(grouped) & test_group_keys:
        raise IdentityAuditError(f"{category}: TRAIN-normal and TEST official groups overlap")

    def group_rank(group_key: str) -> tuple[str, bytes]:
        value = _domain_hash("Gate0-role-v1", digest, category, str(support_seed), group_key)
        return value, _nfc_key(group_key).encode("utf-8")

    chosen_groups = sorted(grouped, key=group_rank)[:68]
    chosen: list[dict[str, Any]] = []
    for group_key in chosen_groups:
        candidates = grouped[group_key]
        def image_rank(row: Mapping[str, Any]) -> tuple[str, bytes]:
            image_key = str(row["_official_image_key"])
            value = _domain_hash("Gate0-image-v1", digest, category,
                                 str(support_seed), group_key, image_key)
            return value, _nfc_key(image_key).encode("utf-8")
        chosen_row = min(candidates, key=image_rank)
        chosen.append({
            "role": "SUPPORT" if len(chosen) < 4 else "C64",
            "image_identity_sha256": chosen_row["image_identity_sha256"],
            "image_content_sha256": chosen_row["image_content_sha256"],
            "source_group_sha256": _source_group_digest(group_key),
            "role_rank_sha256": group_rank(group_key)[0],
            "image_rank_sha256": image_rank(chosen_row)[0],
        })
    if len({row["image_content_sha256"] for row in chosen}) != 68:
        raise IdentityAuditError(f"{category}: selected TRAIN-normal roles contain exact-content duplicates")
    c64 = [row for row in chosen if row["role"] == "C64"]
    static_order = [dict(row) for row in c64]
    # Static identity partition uses the separately frozen hash domain.
    # Compute the exact static digest with the retained private source fields.
    group_by_identity = {r["image_identity_sha256"]: r["_official_group_key"]
                         for rows in grouped.values() for r in rows}
    image_by_identity = {r["image_identity_sha256"]: r["_official_image_key"]
                         for rows in grouped.values() for r in rows}
    static_order = sorted(static_order, key=lambda row: (
        _domain_hash("Gate0-static36-c64-v1", digest, category, str(support_seed),
                     group_by_identity[row["image_identity_sha256"]],
                     image_by_identity[row["image_identity_sha256"]]),
        _nfc_key(group_by_identity[row["image_identity_sha256"]]).encode("utf-8"),
        _nfc_key(image_by_identity[row["image_identity_sha256"]]).encode("utf-8")))
    for index, row in enumerate(static_order):
        row["static36_role"] = "REFERENCE" if index < 32 else "C32"
    return {"dataset_id": dataset_id, "dataset_release_sha256": digest,
            "category": category,
            "support_seed": int(support_seed), "selected": chosen,
            "static36_c64_partition": static_order}


def _stream_order(*, category: str, stream_seed: int, dataset_release_sha256: str,
                  test_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    digest = str(dataset_release_sha256).lower()
    original_positions = [row.get("original_position") for row in test_rows]
    if (any(isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in original_positions) or
            len(set(original_positions)) != len(original_positions)):
        raise IdentityAuditError("TEST original annotation positions are missing or duplicated")
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in test_rows:
        by_group[str(row["_official_group_key"])].append(row)
    ordered_groups = sorted(by_group, key=lambda group: (
        _domain_hash("Gate0-stream-v1", digest, category, str(stream_seed), group),
        _nfc_key(group).encode("utf-8")))
    result: list[dict[str, Any]] = []
    for group_key in ordered_groups:
        views = sorted(by_group[group_key], key=lambda row: int(row["camera_id"]))
        if [int(row["camera_id"]) for row in views] != [1, 2, 3, 4, 5]:
            raise IdentityAuditError(f"{category}: TEST Source Group lacks cameras 1..5 exactly once")
        group_digest = _source_group_digest(group_key)
        for ordinal, row in enumerate(views, 1):
            event_id = _domain_hash(
                "Gate0-event-v1", digest, category, str(stream_seed), group_key,
                str(int(row["camera_id"])), str(row["_official_image_key"]), str(ordinal))
            result.append({
                "position": len(result),
                "original_position": int(row["original_position"]),
                "event_id_sha256": event_id,
                "image_identity_sha256": row["image_identity_sha256"],
                "image_content_sha256": row["image_content_sha256"],
                "source_group_sha256": group_digest,
                "camera_id": int(row["camera_id"]),
                "within_group_ordinal": ordinal,
            })
    return result


def resolve_realiad_private_locators(*, dataset_root: str | Path,
                                    annotation_path: str | Path,
                                    annotation_sha256: str, category: str,
                                    split: str,
                                    selected_image_ids: Sequence[str]) -> dict[str, dict[str, str]]:
    """Resolve only registered TRAIN-normal or TEST identity metadata to local paths.

    TEST outcome and mask fields are omitted by the identity-only projection;
    file contents are checked later immediately before the authorized encoding.
    """
    from experiments.risk_safe.preflight import parse_image_name
    if split not in {"train", "test"} or not re.fullmatch(r"[0-9a-f]{64}", annotation_sha256):
        raise IdentityAuditError("private locator requires a registered split and annotation hash")
    requested = list(selected_image_ids)
    if (not requested or len(set(requested)) != len(requested) or
            any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in requested)):
        raise IdentityAuditError("private locator image identities are missing or repeated")
    actual_hash, _size = _safe_file_hash(Path(annotation_path), identity="annotation")
    if actual_hash != annotation_sha256:
        raise IdentityAuditError("official annotation hash differs from its registration")
    meta, train_rows, test_rows = _identity_only_realiad_sections(Path(annotation_path))
    normal_class, prefix = meta.get("normal_class"), meta.get("prefix", "")
    if not isinstance(normal_class, str) or not normal_class or not isinstance(prefix, str):
        raise IdentityAuditError("annotation lacks its normal class or image prefix")
    wanted, result = set(requested), {}
    for row in train_rows if split == "train" else test_rows:
        image_key = row.get("image_path")
        if not isinstance(image_key, str) or not image_key:
            raise IdentityAuditError("official image identity is missing")
        image_id = _image_identity_digest(image_key)
        if image_id not in wanted:
            continue
        if split == "train" and row.get("anomaly_class") != normal_class:
            raise IdentityAuditError("a selected TRAIN image is not verified normal")
        group, camera = _realiad_identity_key(category, image_key)
        path = _safe_join(Path(dataset_root) / "images", prefix, image_key, identity=image_id)
        parsed = parse_image_name(path, expected_class=category)
        if parsed.official_sample_key != group or parsed.camera_id != camera or image_id in result:
            raise IdentityAuditError("selected official image/group mapping is inconsistent")
        result[image_id] = {"official_image_key": image_key, "official_group_key": group,
                            "image_path": str(path), "relative_id": parsed.rel_path}
    if set(result) != wanted:
        raise IdentityAuditError("one or more registered image identities are absent from the official split")
    return result


def _official_release_status(verified_official_release_sha256: str | None) -> tuple[str | None, str]:
    if verified_official_release_sha256 is None:
        return None, "OFFICIAL_RELEASE_SHA256_UNAVAILABLE"
    digest = str(verified_official_release_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise IdentityAuditError("verified official release SHA-256 is malformed")
    return digest, "VERIFIED_OFFICIAL_RELEASE_SHA256_SUPPLIED_BY_CALLER"


def build_realiad_identity_manifest(dataset_root: str | Path,
                                    json_root: str | Path, *,
                                    verified_official_release_sha256: str | None = None,
                                    workers: int = 4) -> dict[str, Any]:
    """Inventory Real-IAD IDs, TRAIN-normal candidates and TEST groups.

    No TEST outcome fields are returned by the JSON projection. This routine
    does not decode images, inspect masks, or infer labels from file/session
    names. A caller-supplied official release digest is required before frozen
    Gate0 role or event hashes can be emitted.
    """
    root = Path(dataset_root)
    annotations = Path(json_root)
    if not root.is_dir() or not annotations.is_dir():
        raise IdentityAuditError("Real-IAD dataset or annotation root is unavailable")
    files = sorted(annotations.glob("*.json"), key=lambda p: p.name.casefold())
    if len(files) != REAL_IAD_CATEGORY_COUNT:
        raise IdentityAuditError("Real-IAD annotation inventory does not contain exactly 30 categories")

    source_artifacts: list[dict[str, Any]] = []
    pending_by_category: dict[str, dict[str, Any]] = {}
    hash_requests: dict[str, Path] = {}
    category_errors: list[dict[str, str]] = []
    for annotation in files:
        category = annotation.stem
        annotation_sha, annotation_size = _safe_file_hash(annotation, identity="realiad-annotation")
        source_artifacts.append({"artifact_id": f"annotation:{category}",
                                 "sha256": annotation_sha, "bytes": annotation_size})
        category_locators_before = set(hash_requests)
        try:
            meta, train_records, test_records = _identity_only_realiad_sections(annotation)
            prefix = str(meta.get("prefix", ""))
            normal_class = meta.get("normal_class")
            if not isinstance(normal_class, str) or not normal_class:
                raise IdentityAuditError(f"{category}: official normal-class metadata is missing")
            train_normal: list[dict[str, Any]] = []
            test_rows: list[dict[str, Any]] = []
            all_identity_rows: list[dict[str, Any]] = []
            for split, rows in (("train", train_records), ("test", test_records)):
                for position, raw in enumerate(rows):
                    allowed = {"image_path", "category"}
                    if split == "train":
                        allowed.add("anomaly_class")
                    if set(raw) - allowed:
                        raise IdentityAuditError(f"{category}: identity projection retained an unexpected {split} field")
                    image_key = raw.get("image_path")
                    if not isinstance(image_key, str) or not image_key:
                        raise IdentityAuditError(f"{category}: an official image identity is missing")
                    if raw.get("category", category) != category:
                        raise IdentityAuditError(f"{category}: annotation category identity disagrees")
                    is_normal = None
                    if split == "train":
                        # The official TRAIN annotation is used only to identify
                        # eligible normals; file/session strings never define y.
                        if "anomaly_class" not in raw:
                            raise IdentityAuditError(
                                f"{category}: TRAIN normal eligibility lacks its official image-level annotation")
                        is_normal = raw["anomaly_class"] == normal_class
                        if not is_normal:
                            continue
                    group_key, camera = _realiad_identity_key(category, image_key)
                    file_path = _safe_join(root / "images", prefix, image_key,
                                           identity=_sha256_text("image-key\0" + _nfc_key(image_key)))
                    locator = "/".join([p for p in (prefix.replace("\\", "/").strip("/"),
                                          image_key.replace("\\", "/").strip("/")) if p])
                    locator_digest = _sha256_text("local-image-locator-v1\0" + locator)
                    hash_requests.setdefault(locator_digest, file_path)
                    row = {"category": category, "split": split,
                           "original_position": position,
                           "_official_image_key": _nfc_key(image_key),
                           "_official_group_key": group_key,
                           "camera_id": camera,
                           "_locator_sha256": locator_digest}
                    if split == "train":
                        all_identity_rows.append(row)
                        train_normal.append({**row, "train_normal_eligible": True})
                    else:
                        # TEST projection must have no label or mask fields.
                        if any(key in raw for key in ("anomaly_class", "mask_path", "label", "mask")):
                            raise IdentityAuditError(f"{category}: restricted TEST field survived projection")
                        all_identity_rows.append(row)
                        test_rows.append(row)
            train_normal_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            test_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in train_normal:
                train_normal_groups[row["_official_group_key"]].append(row)
            for row in test_rows:
                test_groups[row["_official_group_key"]].append(row)
            invalid_train_groups = sum(
                int(sorted(int(row["camera_id"]) for row in rows) != [1, 2, 3, 4, 5])
                for rows in train_normal_groups.values())
            invalid_test_groups = sum(
                int(sorted(int(row["camera_id"]) for row in rows) != [1, 2, 3, 4, 5])
                for rows in test_groups.values())
            train_test_overlap = len(set(train_normal_groups) & set(test_groups))
            pending_by_category[category] = {
                "annotation_sha256": annotation_sha,
                "train_normal_rows": train_normal,
                "test_rows": test_rows,
                "all_identity_rows": all_identity_rows,
                "train_normal_group_count": len(train_normal_groups),
                "test_group_count": len(test_groups),
                "test_event_count": len(test_rows),
                "invalid_train_group_count": invalid_train_groups,
                "invalid_test_group_count": invalid_test_groups,
                "train_test_group_overlap_count": train_test_overlap,
            }
        except IdentityAuditError as exc:
            # Messages are already sanitized and contain no raw path or label.
            for locator in set(hash_requests) - category_locators_before:
                hash_requests.pop(locator, None)
            category_errors.append({"category": category, "reason": str(exc)})

    hashed = _hash_many([(path, locator) for locator, path in sorted(hash_requests.items())], workers=workers)
    category_summaries: list[dict[str, Any]] = []
    all_output_rows: list[dict[str, Any]] = []
    category_role_status: dict[str, str] = {}
    for category, detail in sorted(pending_by_category.items()):
        records: list[dict[str, Any]] = []
        normal_locators = {candidate["_locator_sha256"]
                           for candidate in detail["train_normal_rows"]}
        for row in detail["all_identity_rows"]:
            digest, size = hashed[row["_locator_sha256"]]
            is_normal = row["split"] == "train" and row["_locator_sha256"] in normal_locators
            record = _record_for_identity(
                dataset_id="realiad", category=category, split=row["split"],
                official_image_key=row["_official_image_key"],
                group_key=row["_official_group_key"], camera_id=row["camera_id"],
                original_position=row["original_position"], local_locator=row["_locator_sha256"],
                content_sha256=digest, byte_count=size,
                train_normal_eligible=True if is_normal else None)
            records.append(record)
        duplicates = _duplicate_summary(records)
        status = "LOCAL_IDENTITY_CENSUS_PASS"
        if detail["invalid_test_group_count"] or detail["invalid_train_group_count"] or detail["train_test_group_overlap_count"]:
            status = "GROUP_OR_ROLE_INTEGRITY_BLOCKED"
        elif detail["train_normal_group_count"] < 68:
            status = "INSUFFICIENT_TRAIN_NORMAL_GROUPS"
        category_role_status[category] = status
        category_summaries.append({
            "category": category,
            "train_normal_image_count": len(detail["train_normal_rows"]),
            "train_normal_group_count": detail["train_normal_group_count"],
            "test_image_event_count": detail["test_event_count"],
            "test_source_group_count": detail["test_group_count"],
            "train_normal_groups_with_invalid_camera_set": detail["invalid_train_group_count"],
            "test_groups_with_invalid_camera_set": detail["invalid_test_group_count"],
            "train_test_source_group_overlap_count": detail["train_test_group_overlap_count"],
            "duplicate_audit": duplicates,
            "role_status": status,
        })
        all_output_rows.extend(records)

    official_digest, digest_status = _official_release_status(verified_official_release_sha256)
    role_manifests: list[dict[str, Any]] = []
    stream_manifests: list[dict[str, Any]] = []
    if official_digest is not None:
        for category, detail in sorted(pending_by_category.items()):
            if category_role_status.get(category) != "LOCAL_IDENTITY_CENSUS_PASS":
                continue
            records_by_locator = {row["_locator_sha256"]: row for row in all_output_rows
                                  if row["category"] == category and row["split"] == "train"}
            normal_rows = []
            for raw in detail["train_normal_rows"]:
                record = records_by_locator[raw["_locator_sha256"]]
                normal_rows.append({**record,
                                    "_official_group_key": raw["_official_group_key"],
                                    "_official_image_key": raw["_official_image_key"]})
            test_rows = []
            test_records_by_locator = {row["file_locator_sha256"]: row for row in all_output_rows
                                       if row["category"] == category and row["split"] == "test"}
            for raw in detail["test_rows"]:
                record = test_records_by_locator[raw["_locator_sha256"]]
                test_rows.append({**record,
                                  "_official_group_key": raw["_official_group_key"],
                                  "_official_image_key": raw["_official_image_key"]})
            for support_seed in (0, 1):
                role = _select_role_groups(
                    dataset_id="realiad", category=category, support_seed=support_seed,
                    dataset_release_sha256=official_digest,
                    train_normal_rows=normal_rows,
                    test_group_keys={str(row["_official_group_key"]) for row in test_rows})
                selected_content = {row["image_content_sha256"] for row in role["selected"]}
                test_content = {row["image_content_sha256"] for row in test_rows}
                if selected_content & test_content:
                    category_role_status[category] = "EXACT_DUPLICATE_CROSS_ROLE_BLOCKED"
                    break
                role_manifests.append(role)
            if category_role_status.get(category) != "LOCAL_IDENTITY_CENSUS_PASS":
                continue
            for stream_seed in (0, 11):
                stream = _stream_order(category=category, stream_seed=stream_seed,
                                       dataset_release_sha256=official_digest,
                                       test_rows=test_rows)
                stream_manifests.append({"category": category,
                                         "stream_seed": stream_seed,
                                         "event_count": len(stream),
                                         "events": stream})

    source_artifacts_sha256 = _sha256_bytes(_canonical_json(sorted(
        source_artifacts, key=lambda item: item["artifact_id"])))
    snapshot_sha = _snapshot_digest("realiad", source_artifacts, all_output_rows)
    all_pass = (len(category_summaries) == REAL_IAD_CATEGORY_COUNT and
                all(item["role_status"] == "LOCAL_IDENTITY_CENSUS_PASS" for item in category_summaries) and
                not category_errors)
    return {
        "schema": "final-benchmark.identity-manifest.v1",
        "dataset_id": "realiad",
        "scope": "TRAIN-normal and official TEST identities; no TEST outcomes",
        "category_count": len(files),
        "categories": category_summaries,
        "source_artifacts": source_artifacts,
        "source_artifacts_sha256": source_artifacts_sha256,
        "local_snapshot_sha256": snapshot_sha,
        "official_release_sha256": official_digest,
        "official_release_status": digest_status,
        "official_provenance_status": ("REQUIRES_EXTERNAL_RELEASE_RECEIPT_REVIEW"
                                        if official_digest else
                                        "LOCAL_SOURCE_LINEAGE_NOT_EXTERNALLY_AUTHENTICATED"),
        "identity_census_status": "PASS" if all_pass else "PARTIAL",
        "role_manifest_status": "PASS" if official_digest and len(role_manifests) == 2 * REAL_IAD_CATEGORY_COUNT else "BLOCKED_OFFICIAL_RELEASE_SHA256",
        "stream_manifest_status": "PASS" if official_digest and len(stream_manifests) == 2 * REAL_IAD_CATEGORY_COUNT else "BLOCKED_OFFICIAL_RELEASE_SHA256",
        "role_manifests": role_manifests,
        "stream_manifests": stream_manifests,
        "identity_records": all_output_rows,
        "category_errors": category_errors,
        "duplicate_content_summary": _duplicate_summary(all_output_rows),
        "test_label_access": "NOT_ACCESSED; identity-only projection dropped outcome fields",
        "mask_access": "NOT_ACCESSED",
        "image_decode_or_model_inference": "NOT_PERFORMED",
    }


def _parse_csv_projected_line(line: bytes | str, header: Sequence[str], *,
                              train_label_allowed: bool) -> dict[str, str]:
    """Decode allowed CSV fields only; skip TEST outcome and mask bytes."""
    if isinstance(line, str):
        line = line.encode("utf-8")  # Synthetic callers may supply text.
    lower = [name.strip().lower() for name in header]
    required = {"object", "split", "image"}
    if not required.issubset(lower):
        raise IdentityAuditError("VisA 1cls split table lacks required identity columns")
    split_index = lower.index("split")
    image_index = lower.index("image")
    object_index = lower.index("object")
    label_index = lower.index("label") if "label" in lower else None
    if label_index is not None and split_index > label_index:
        raise IdentityAuditError("VisA split column must precede label for safe TEST projection")
    allowed = {object_index, split_index, image_index}
    group_indices = {i for i, name in enumerate(lower) if name in _VISA_GROUP_FIELDS}
    allowed |= group_indices

    output: dict[str, str] = {}
    chars = bytearray()
    index = 0
    column = 0
    in_quotes = False
    at_field_start = True
    split_value: str | None = None

    def keep_current() -> bool:
        if column in allowed:
            return True
        return bool(label_index is not None and column == label_index and
                    train_label_allowed and split_value == "train")

    def finish_field() -> None:
        nonlocal chars, split_value
        name = lower[column] if column < len(lower) else ""
        if keep_current():
            try:
                value = chars.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise IdentityAuditError("VisA permitted identity field is not UTF-8") from exc
            if name == "split":
                split_value = value.strip().lower()
            elif name == "label":
                output[name] = value.strip().lower()
            else:
                output[name] = value
        chars = bytearray()

    while index < len(line):
        char = line[index]
        if in_quotes:
            if char == ord('"'):
                if index + 1 < len(line) and line[index + 1] == ord('"'):
                    if keep_current():
                        chars.append(ord('"'))
                    index += 2
                    continue
                in_quotes = False
            elif keep_current():
                chars.append(char)
            index += 1
            continue
        if char == ord('"') and at_field_start:
            in_quotes = True
            at_field_start = False
            index += 1
            continue
        if char == ord(","):
            finish_field()
            column += 1
            at_field_start = True
            index += 1
            continue
        if char in (ord("\r"), ord("\n")):
            break
        if char == ord('"'):
            raise IdentityAuditError("VisA split table contains malformed CSV quoting")
        if keep_current():
            chars.append(char)
        at_field_start = False
        index += 1
    if in_quotes:
        raise IdentityAuditError("VisA split table contains a multiline or unterminated CSV field")
    finish_field()
    if column + 1 != len(header):
        raise IdentityAuditError("VisA split table row width differs from its header")
    if split_value is not None:
        output["split"] = split_value
    # Normalize names, but never include outcome/mask columns in returned rows.
    return output


def build_visa_1cls_identity_manifest(dataset_root: str | Path, *,
                                      verified_official_release_sha256: str | None = None,
                                      workers: int = 4) -> dict[str, Any]:
    """Inventory official VisA 1cls IDs with TEST outcome/mask cells skipped."""
    root = Path(dataset_root)
    source = root / VISA_1CLS_SOURCE
    license_file = root / "LICENSE-DATASET"
    if not root.is_dir() or not source.is_file():
        raise IdentityAuditError("VisA root or official 1cls split table is unavailable")
    try:
        csv_sha, csv_bytes = _safe_file_hash(source, identity="visa-1cls-csv")
        license_sha, license_bytes = _safe_file_hash(license_file, identity="visa-license")
        with source.open("rb") as stream:
            header_line = stream.readline().decode("utf-8-sig")
            header = next(csv.reader([header_line]))
            lower = [name.strip().lower() for name in header]
            if any(name in {"anomaly_class", "ground_truth", "gt"} for name in lower):
                raise IdentityAuditError("VisA split table contains an unregistered outcome field")
            rows: list[dict[str, Any]] = []
            for position, line in enumerate(stream):
                if not line.strip():
                    continue
                # `label` is materialized only for train rows; `mask` and
                # every TEST outcome cell are consumed by the scanner but not
                # copied, decoded, compared, logged, or stored.
                projection = _parse_csv_projected_line(
                    line, header, train_label_allowed=True)
                split = projection.get("split", "").strip().lower()
                category = projection.get("object", "").strip()
                image_key = projection.get("image", "").strip()
                if split not in {"train", "test"}:
                    raise IdentityAuditError("VisA 1cls split contains an unregistered split value")
                if not category or not image_key:
                    raise IdentityAuditError("VisA identity projection has an empty category or image key")
                train_normal = None
                if split == "train":
                    label = projection.get("label")
                    if label not in {"normal", "anomaly"}:
                        raise IdentityAuditError("VisA TRAIN label vocabulary is not the registered normal/anomaly pair")
                    train_normal = label == VISA_NORMAL_TRAIN_LABEL
                file_path = _safe_join(root, image_key,
                                       identity=_sha256_text("visa-image-key\0" + _nfc_key(image_key)))
                locator_sha = _sha256_text("local-image-locator-v1\0" + _nfc_key(image_key))
                group_value = next((projection[name] for name in projection
                                    if name.lower() in _VISA_GROUP_FIELDS), None)
                group_key = _nfc_key(group_value) if group_value else None
                rows.append({
                    "category": _nfc_key(category), "split": split,
                    "original_position": position,
                    "_official_image_key": _nfc_key(image_key),
                    "_locator_sha256": locator_sha,
                    "_official_group_key": group_key,
                    "train_normal_eligible": train_normal,
                })
    except OSError as exc:
        raise IdentityAuditError(f"VisA metadata is unavailable ({type(exc).__name__})") from None

    categories = sorted({row["category"] for row in rows})
    if len(categories) != VISA_1CLS_CATEGORY_COUNT:
        raise IdentityAuditError("VisA 1cls table does not contain exactly 12 official categories")
    hash_requests: dict[str, Path] = {}
    for row in rows:
        hash_requests.setdefault(row["_locator_sha256"], _safe_join(
            root, row["_official_image_key"], identity=row["_locator_sha256"]))
    hashed = _hash_many([(path, key) for key, path in sorted(hash_requests.items())], workers=workers)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        content_sha, byte_count = hashed[row["_locator_sha256"]]
        output_rows.append(_record_for_identity(
            dataset_id="visa_1cls", category=row["category"], split=row["split"],
            official_image_key=row["_official_image_key"],
            group_key=row["_official_group_key"], camera_id=None,
            original_position=row["original_position"],
            local_locator=row["_locator_sha256"], content_sha256=content_sha,
            byte_count=byte_count,
            train_normal_eligible=(True if row["train_normal_eligible"] is True else None)))
    categories_summary = []
    for category in categories:
        selected = [row for row in rows if row["category"] == category]
        category_output = [row for row in output_rows if row["category"] == category]
        train_normal_count = sum(1 for row in selected if row["split"] == "train" and
                                 row["train_normal_eligible"] is True)
        test_count = sum(1 for row in selected if row["split"] == "test")
        explicit_group_keys = {row["_official_group_key"] for row in selected
                               if row["_official_group_key"] is not None}
        categories_summary.append({
            "category": category,
            "train_normal_image_count": train_normal_count,
            "test_image_event_count": test_count,
            "official_group_key_column_present": bool(explicit_group_keys),
            "distinct_official_group_key_count": len(explicit_group_keys),
            "distinct_train_normal_group_count": len({row["_official_group_key"] for row in selected
                                                       if row["split"] == "train" and row["train_normal_eligible"] is True and row["_official_group_key"]}),
            "group_status": "OFFICIAL_GROUP_KEY_PRESENT_REQUIRES_PROVENANCE" if explicit_group_keys else "NO_OFFICIAL_PHYSICAL_GROUP_KEY",
            "duplicate_audit": _duplicate_summary(category_output),
        })
    source_artifacts = [
        {"artifact_id": "official-split:1cls.csv", "sha256": csv_sha, "bytes": csv_bytes},
        {"artifact_id": "dataset-license", "sha256": license_sha, "bytes": license_bytes},
    ]
    official_digest, digest_status = _official_release_status(verified_official_release_sha256)
    group_columns = [name for name in lower if name in _VISA_GROUP_FIELDS]
    group_usable = bool(group_columns) and all(
        bool(row["_official_group_key"]) for row in rows)
    role_status = "BLOCKED_OFFICIAL_GROUP_KEY"
    if group_usable and official_digest is not None:
        role_status = "ROLE_SELECTION_REQUIRES_SOURCE_GROUP_VALIDATION"
    snapshot_sha = _snapshot_digest("visa_1cls", source_artifacts, output_rows)
    return {
        "schema": "final-benchmark.identity-manifest.v1",
        "dataset_id": "visa_1cls",
        "scope": "official 1cls split; TRAIN normal eligibility and TEST identities only",
        "category_count": len(categories),
        "categories": categories_summary,
        "source_artifacts": source_artifacts,
        "source_artifacts_sha256": _sha256_bytes(_canonical_json(source_artifacts)),
        "local_snapshot_sha256": snapshot_sha,
        "official_release_sha256": official_digest,
        "official_release_status": digest_status,
        "official_provenance_status": ("REQUIRES_EXTERNAL_RELEASE_RECEIPT_REVIEW"
                                        if official_digest else
                                        "LOCAL_SOURCE_LINEAGE_NOT_EXTERNALLY_AUTHENTICATED"),
        "official_group_key_columns": group_columns,
        "identity_census_status": "PASS_LOCAL_IMAGE_MEMBERSHIP",
        "role_manifest_status": role_status,
        "identity_records": output_rows,
        "duplicate_content_summary": _duplicate_summary(output_rows),
        "test_label_access": "NOT_ACCESSED; TEST label cells were skipped by the identity-only CSV projection",
        "mask_access": "NOT_ACCESSED; mask cells and files were not opened",
        "image_decode_or_model_inference": "NOT_PERFORMED",
    }


def validate_identity_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate schema, hashes and absence of labels, masks and raw locators."""
    if manifest.get("schema") != "final-benchmark.identity-manifest.v1":
        raise IdentityAuditError("identity manifest schema mismatch")
    if manifest.get("dataset_id") not in {"realiad", "visa_1cls"}:
        raise IdentityAuditError("identity manifest dataset ID is not registered")
    records = manifest.get("identity_records")
    if not isinstance(records, list):
        raise IdentityAuditError("identity manifest records are missing")
    expected_categories = (REAL_IAD_CATEGORY_COUNT if manifest["dataset_id"] == "realiad"
                           else VISA_1CLS_CATEGORY_COUNT)
    categories = manifest.get("categories")
    if (manifest.get("category_count") != expected_categories or
            not isinstance(categories, list) or len(categories) != expected_categories):
        raise IdentityAuditError("identity manifest category inventory is incomplete")
    forbidden = {"label", "anomaly_class", "mask", "mask_path", "ground_truth", "gt",
                 "official_image_key", "official_group_key", "image_path", "local_path"}
    def reject_nested_restricted(value: Any) -> None:
        if isinstance(value, Mapping):
            if forbidden.intersection(map(str, value.keys())):
                raise IdentityAuditError("identity manifest contains a restricted nested field")
            for child in value.values():
                reject_nested_restricted(child)
        elif isinstance(value, list):
            for child in value:
                reject_nested_restricted(child)
    reject_nested_restricted(manifest)
    seen_identities: set[tuple[str, str, str]] = set()
    for record in records:
        if not isinstance(record, Mapping) or forbidden.intersection(record):
            raise IdentityAuditError("identity manifest contains a restricted field")
        if (record.get("dataset_id") != manifest["dataset_id"] or
                record.get("category") not in {item.get("category") for item in categories} or
                record.get("split") not in {"train", "test"}):
            raise IdentityAuditError("identity record is outside its registered dataset/category/split")
        for key in ("image_identity_sha256", "file_locator_sha256", "image_content_sha256"):
            value = record.get(key)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise IdentityAuditError("identity manifest contains an invalid digest")
        group = record.get("source_group_sha256")
        if group is not None and (not isinstance(group, str) or not re.fullmatch(r"[0-9a-f]{64}", group)):
            raise IdentityAuditError("identity manifest contains an invalid group digest")
        identity_key = (str(record["category"]), str(record["split"]),
                        str(record["image_identity_sha256"]))
        if identity_key in seen_identities:
            raise IdentityAuditError("identity manifest repeats an official image identity")
        seen_identities.add(identity_key)
        byte_count = record.get("image_bytes")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise IdentityAuditError("identity manifest contains an invalid image byte count")
    return {"status": "PASS_IDENTITY_ONLY_SCHEMA",
            "dataset_id": manifest["dataset_id"],
            "identity_record_count": len(records),
            "test_label_access": manifest.get("test_label_access"),
            "mask_access": manifest.get("mask_access")}


def write_manifest_exclusive(path: str | Path, manifest: Mapping[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(manifest) + b"\n"
    digest = _sha256_bytes(payload)
    try:
        with target.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise IdentityAuditError("identity manifest output already exists; refusing overwrite") from None
    return digest


def build_static36_schedule(role_manifest: Mapping[str, Any], *,
                            dataset_release_sha256: str,
                            category: str, support_seed: int) -> dict[str, Any]:
    """Return the approved baseline-only 36/32 partition without mutating roles."""
    if role_manifest.get("dataset_id") not in {"realiad", "visa_1cls"}:
        raise IdentityAuditError("Static-36+32 requires a registered dataset role manifest")
    if role_manifest.get("category") != category or int(role_manifest.get("support_seed", -1)) != int(support_seed):
        raise IdentityAuditError("Static-36+32 schedule does not match the selected P0 role manifest")
    if (not re.fullmatch(r"[0-9a-f]{64}", str(dataset_release_sha256).lower()) or
            role_manifest.get("dataset_release_sha256") != str(dataset_release_sha256).lower()):
        raise IdentityAuditError("Static-36+32 release digest differs from its principal role draw")
    selected = role_manifest.get("selected")
    ranked = role_manifest.get("static36_c64_partition")
    if not isinstance(selected, list) or len(selected) != 68 or not isinstance(ranked, list) or len(ranked) != 64:
        raise IdentityAuditError("Static-36+32 requires exactly four Supports and 64 C64 identities")
    supports = [row for row in selected if row.get("role") == "SUPPORT"]
    c64 = [row for row in selected if row.get("role") == "C64"]
    if len(supports) != 4 or len(c64) != 64:
        raise IdentityAuditError("principal role manifest does not contain the registered 4+64 allocation")
    if (len({row.get("source_group_sha256") for row in selected}) != 68 or
            len({row.get("image_content_sha256") for row in selected}) != 68):
        raise IdentityAuditError("principal role allocation repeats a Source Group or exact image content")
    ranked_hashes = [row.get("image_identity_sha256") for row in ranked]
    if len(set(ranked_hashes)) != 64 or set(ranked_hashes) != {row.get("image_identity_sha256") for row in c64}:
        raise IdentityAuditError("Static-36+32 partition does not preserve the original C64 identity set")
    ref_c64 = [row for row in ranked if row.get("static36_role") == "REFERENCE"]
    c32 = [row for row in ranked if row.get("static36_role") == "C32"]
    if len(ref_c64) != 32 or len(c32) != 32:
        raise IdentityAuditError("Static-36+32 must split P0 C64 into 32 reference and 32 C32 identities")
    schedule = {
        "schema": "final-benchmark.static36-schedule.v1",
        "dataset_id": role_manifest["dataset_id"], "category": category,
        "support_seed": int(support_seed), "stream_seed": 0,
        "dataset_release_sha256": str(dataset_release_sha256).lower(),
        "reference_identity_sha256": [row["image_identity_sha256"] for row in supports + ref_c64],
        "reference_group_sha256": [row["source_group_sha256"] for row in supports + ref_c64],
        "c32_identity_sha256": [row["image_identity_sha256"] for row in c32],
        "c32_group_sha256": [row["source_group_sha256"] for row in c32],
        "principal_c64_identity_sha256": sorted(str(row["image_identity_sha256"]) for row in c64),
        "principal_roles_mutated": False,
        "selection": "Gate0-static36-c64-v1 first 32 C64 ranks; ties by normalized UTF-8 keys",
        "status": "METADATA_SCHEDULE_ONLY_KCENTER_SOURCE_FIDELITY_PENDING",
    }
    schedule["schedule_sha256"] = _sha256_bytes(_canonical_json(schedule))
    return schedule


def prepare_visa_exception_candidate(*, identity_rows: Sequence[Mapping[str, Any]],
                                     proposed_unit_by_image: Mapping[str, str]) -> dict[str, Any]:
    """Validate a supplied identity/duplicate-cluster proposal without enabling it.

    Unit identities must be supplied by the future registered exception; this
    function does not infer physical objects or authorize online VisA feedback.
    """
    rows = list(identity_rows)
    allowed = {"image_identity_sha256", "image_content_sha256", "role"}
    if not rows or any(not isinstance(row, Mapping) or set(row) != allowed for row in rows):
        raise IdentityAuditError("VisA candidate requires outcome-free identity rows")
    images = [str(row.get("image_identity_sha256", "")) for row in rows]
    contents = [str(row.get("image_content_sha256", "")) for row in rows]
    roles = [row.get("role") for row in rows]
    if (len(set(images)) != len(images) or set(images) != set(proposed_unit_by_image) or
            roles.count("SUPPORT") != 4 or roles.count("C64") != 64 or "TEST" not in roles or
            any(role not in {"SUPPORT", "C64", "TEST"} for role in roles) or
            any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in images + contents)):
        raise IdentityAuditError("VisA candidate identities or 4+64/TEST roles are incomplete")
    units = [str(proposed_unit_by_image[image]) for image in images]
    if any(not re.fullmatch(r"[0-9a-f]{64}", unit) for unit in units):
        raise IdentityAuditError("VisA candidate unit identities must be opaque SHA-256 digests")
    unit_roles: dict[str, set[str]] = defaultdict(set)
    content_units: dict[str, set[str]] = defaultdict(set)
    for role, content, unit in zip(roles, contents, units):
        unit_roles[unit].add(str(role))
        content_units[content].add(unit)
    train_units = [unit for role, unit in zip(roles, units) if role in {"SUPPORT", "C64"}]
    if len(set(train_units)) != 68 or any(len(value) != 1 for value in unit_roles.values()):
        raise IdentityAuditError("VisA candidate unit crosses Support/C64/TEST roles")
    if any(len(value) != 1 for value in content_units.values()):
        raise IdentityAuditError("byte-identical images must share a proposed duplicate-cluster unit")
    return {"dataset_id": "visa_1cls", "query_rate_target": 0.10,
            "test_image_count": roles.count("TEST"),
            "test_unit_count": len({unit for role, unit in zip(roles, units) if role == "TEST"}),
            "image_to_unit": dict(zip(images, units)),
            "physical_independence_claim": "NOT_ESTABLISHED",
            "status": "RESEARCHER_DECISION_REQUIRED_NOT_EXECUTABLE"}

