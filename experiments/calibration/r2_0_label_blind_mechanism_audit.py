"""CPU-only, label-blind mechanism audit for the frozen R1-S4B evidence tape.

This module only opens the R1 manifest, public sealed-score tape, event NPZ
shards, and a selective structural projection of the transition tape. The
reader intentionally has no path for private feedback or correction records.
It does not load a model, image, label, or mask.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
EXPECTED_HEAD = "606844407dc5c356fafcabd2adddb0dc294bde5e"
FROZEN_R1_SOURCE_COMMIT = "c86d5b89fa0062b87d80adb324eaa6bc3c94448e"
R1_RUN_ID = "r1s4b-switch-s0-512-20261031-a1"
R1_RUN_RELATIVE = Path(
    "outputs/calibration/r1_digest_contract_v1_engineering_"
    "r1s4b-switch-s0-512-20261031-a1")
R1_REPORT_RELATIVE = Path(
    "docs/calibration/audit/R1_S4B_ENGINEERING_EXECUTION_REPORT_20261031.md")
R2_OUTPUT_NAME = "r2_0_label_blind_r1s4b_switch_s0_512_a1"
QUERY_BINDING_SHA256 = "dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183"
QUERY_VERIFICATION_SHA256 = "ffbedc32edbd8f6c9b79d0d5b0fe3663d4cfb315d20273815405b84a486cc991"
QUERY_POSITIONS_SHA256 = "7f4aedf49f3e3bffd2482bd00d9e16512553b7cef8c0af7e002b47b60ddcb17f"
FIXED_BANK_SHA256 = "8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558"
EVENT_COUNT = 512
PATCH_COUNT = 32
PATCH_VECTOR_SIZE = 784
EPS_PATCH = 1e-7
EPS_SCORE = 1e-6
PATCH_TOLERANCE_SENSITIVITY = (0.0, 1e-8, EPS_PATCH, 1e-6, 1e-5, 1e-4)
SCORE_TOLERANCE_SENSITIVITY = (0.0, 1e-8, EPS_SCORE, 1e-5, 1e-4)
C2_CAPACITY = 256
C2_PATCHES_PER_QUERY = 32

EVENT_ARRAY_FIELDS = frozenset({
    "base_patch_map", "top32_ids", "top32_features", "c1_delta", "c2_delta"})
SEALED_SCORE_FIELDS = frozenset({
    "position", "array_fields", "array_sha256", "shard_path", "shard_sha256",
    "c0_score", "c1_score", "c2_score", "official_group_sha256",
    "fixed_support_sha256", "fixed_support_raw_prototype_sha256",
    "greedy_memory_version_before", "greedy_memory_sha256_before",
    "c1_state_sha256_before", "c1_eligible_anchor_slots",
    "c1_same_source_excluded_anchor_slots", "c1_far_field_patch_count",
    "c1_far_field_abs_correction_max", "c2_state_sha256_before",
    "c2_eligible_anchor_count", "score_state_sha256"})
TRANSITION_FIELDS = frozenset({
    "position", "query_decision", "causal_order_complete",
    "greedy_memory_version_after", "greedy_memory_sha256_after",
    "greedy_memory_content_sha256_after", "c1_state_sha256_after",
    "c2_state_sha256_after"})


class R2AuditBlocked(RuntimeError):
    """The frozen evidence or audit inputs failed a fail-closed gate."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    if os.name == "nt" and path.exists():
        import ctypes
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if attrs == 0xFFFFFFFF:
            raise R2AuditBlocked(f"Cannot inspect path attributes: {path}")
        return bool(attrs & 0x400)
    return False


def _require_plain_path(path: Path, *, must_exist: bool) -> Path:
    absolute = Path(os.path.abspath(path))
    for component in reversed((absolute, *absolute.parents)):
        if component.exists() and _is_reparse_point(component):
            raise R2AuditBlocked(f"Symlink/reparse-point path is refused: {component}")
    if must_exist and not absolute.exists():
        raise R2AuditBlocked(f"Required audit input is missing: {absolute}")
    return absolute


def _skip_json_string(text: str, offset: int) -> int:
    if offset >= len(text) or text[offset] != '"':
        raise R2AuditBlocked("Malformed JSON string in structural parser")
    i = offset + 1
    while i < len(text):
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if char == '"':
            return i + 1
        i += 1
    raise R2AuditBlocked("Unterminated JSON string in structural parser")


def _skip_json_value(text: str, offset: int) -> int:
    """Skip a JSON value without decoding or exposing its contents."""
    i = offset
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text):
        raise R2AuditBlocked("Missing JSON value in structural parser")
    if text[i] == '"':
        return _skip_json_string(text, i)
    if text[i] in "[{":
        stack = ["]" if text[i] == "[" else "}"]
        i += 1
        while i < len(text) and stack:
            char = text[i]
            if char == '"':
                i = _skip_json_string(text, i)
                continue
            if char == "[":
                stack.append("]")
            elif char == "{":
                stack.append("}")
            elif char in "]}":
                if not stack or char != stack[-1]:
                    raise R2AuditBlocked("Mismatched JSON container in structural parser")
                stack.pop()
            i += 1
        if stack:
            raise R2AuditBlocked("Unterminated JSON container in structural parser")
        return i
    start = i
    while i < len(text) and text[i] not in ",}]" and not text[i].isspace():
        i += 1
    if i == start:
        raise R2AuditBlocked("Malformed JSON scalar in structural parser")
    return i


def extract_json_object_fields(raw: bytes | str,
                               allowed_fields: Iterable[str]) -> dict[str, Any]:
    """Decode only allow-listed top-level fields; discard all other values."""
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    allowed = frozenset(allowed_fields)
    def reject_constant(value: str):
        raise R2AuditBlocked(f"Nonstandard JSON constant is refused: {value}")

    decoder = json.JSONDecoder(parse_constant=reject_constant)
    i = 0
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        raise R2AuditBlocked("Expected a JSON object")
    i += 1
    result: dict[str, Any] = {}
    seen: set[str] = set()
    while True:
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            raise R2AuditBlocked("Unterminated JSON object")
        if text[i] == "}":
            i += 1
            break
        try:
            key, key_end = decoder.raw_decode(text, i)
        except (json.JSONDecodeError, TypeError) as exc:
            raise R2AuditBlocked("Malformed JSON object key") from exc
        if not isinstance(key, str) or key in seen:
            raise R2AuditBlocked("Non-string or duplicate JSON object key")
        seen.add(key)
        i = key_end
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != ":":
            raise R2AuditBlocked("Missing colon in JSON object")
        i += 1
        while i < len(text) and text[i].isspace():
            i += 1
        if key in allowed:
            try:
                value, i = decoder.raw_decode(text, i)
            except json.JSONDecodeError as exc:
                raise R2AuditBlocked(f"Malformed allow-listed JSON field: {key}") from exc
            result[key] = value
        else:
            i = _skip_json_value(text, i)
        while i < len(text) and text[i].isspace():
            i += 1
        if i < len(text) and text[i] == ",":
            i += 1
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text) or text[i] == "}":
                raise R2AuditBlocked("Trailing comma in JSON object")
            continue
        if i < len(text) and text[i] == "}":
            i += 1
            break
        raise R2AuditBlocked("Malformed JSON object delimiter")
    if text[i:].strip():
        raise R2AuditBlocked("Trailing bytes after JSON object")
    return result


class ReadOnlyR1Evidence:
    """Allow-listed read adapter; label-bearing R1 ledgers are unreachable."""

    _ROOT_FILES = frozenset({"run_manifest.json", "sealed_scores.jsonl"})
    _TRANSITIONS = "event_transitions.jsonl"

    def __init__(self, root: Path):
        self.root = _require_plain_path(Path(root), must_exist=True)
        if not self.root.is_dir():
            raise R2AuditBlocked("R1 evidence root is not a directory")

    def _authorized_path(self, relative: str, *, transition_only: bool = False) -> Path:
        normalized = str(relative).replace("\\", "/")
        pure = PurePosixPath(normalized)
        if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
            raise R2AuditBlocked("Path traversal is refused by the R1 evidence allowlist")
        allowed = normalized in self._ROOT_FILES
        allowed = allowed or (transition_only and normalized == self._TRANSITIONS)
        allowed = allowed or bool(re.fullmatch(r"events/event_[0-9]{6}\.npz", normalized))
        if not allowed:
            raise R2AuditBlocked(f"R1 evidence path is not authorized: {normalized}")
        path = self.root.joinpath(*pure.parts)
        _require_plain_path(path, must_exist=True)
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise R2AuditBlocked("Resolved evidence path escaped its root") from exc
        return path

    def read_authorized_bytes(self, relative: str) -> bytes:
        if relative == self._TRANSITIONS:
            raise R2AuditBlocked(
                "Transition tape is available only through its structural-field projection")
        path = self._authorized_path(relative)
        try:
            with path.open("rb") as stream:
                return stream.read()
        except OSError as exc:
            raise R2AuditBlocked(f"Could not read authorized evidence file {relative}: {exc}") from exc

    def read_manifest(self) -> tuple[dict[str, Any], bytes]:
        raw = self.read_authorized_bytes("run_manifest.json")
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise R2AuditBlocked("R1 run manifest is malformed") from exc
        if not isinstance(value, dict):
            raise R2AuditBlocked("R1 run manifest is not a JSON object")
        return value, raw

    def read_sealed_records(self) -> tuple[list[dict[str, Any]], bytes]:
        raw = self.read_authorized_bytes("sealed_scores.jsonl")
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            row = extract_json_object_fields(line, SEALED_SCORE_FIELDS)
            records.append(row)
        return records, raw

    def read_transition_structure(self) -> tuple[list[dict[str, Any]], str]:
        path = self._authorized_path(self._TRANSITIONS, transition_only=True)
        projection: list[dict[str, Any]] = []
        try:
            with path.open("rb") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    projection.append(extract_json_object_fields(line, TRANSITION_FIELDS))
        except OSError as exc:
            raise R2AuditBlocked(f"Could not read structural transition fields: {exc}") from exc
        return projection, _sha256_json(projection)

    def read_event_shard(self, position: int) -> tuple[dict[str, np.ndarray], bytes]:
        if isinstance(position, bool) or not isinstance(position, (int, np.integer)):
            raise R2AuditBlocked("Event position must be an integer")
        relative = f"events/event_{int(position):06d}.npz"
        raw = self.read_authorized_bytes(relative)
        try:
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                if set(archive.files) != EVENT_ARRAY_FIELDS:
                    raise R2AuditBlocked(f"Unexpected NPZ array fields at event {position}")
                arrays = {name: archive[name] for name in archive.files}
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
            raise R2AuditBlocked(f"Malformed or unreadable event shard {position}") from exc
        return arrays, raw


def validate_positions(positions: Sequence[int], expected_count: int) -> None:
    normalized = [int(value) for value in positions]
    expected = list(range(int(expected_count)))
    if normalized != expected:
        raise R2AuditBlocked(
            f"Event positions must be exactly 0..{expected_count - 1} with no gaps/duplicates")


def _arrays_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(value.shape), value.dtype.str)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def corrected_dense_map(base_map: np.ndarray, selected_ids: Sequence[int],
                        selected_delta: np.ndarray) -> np.ndarray:
    """Exact frozen R1 float32 selected-patch update."""
    base = np.asarray(base_map, dtype=np.float32)
    ids = np.asarray(selected_ids, dtype=np.int64)
    delta = np.asarray(selected_delta, dtype=np.float32)
    if base.ndim != 1 or ids.ndim != 1 or delta.shape != ids.shape:
        raise ValueError("Expected one dense map and aligned selected IDs/deltas")
    if len(set(ids.tolist())) != len(ids) or np.any(ids < 0) or np.any(ids >= len(base)):
        raise ValueError("Selected patch IDs are duplicate or out of range")
    if not np.isfinite(base).all() or not np.isfinite(delta).all():
        raise ValueError("Nonfinite base map or correction delta")
    result = base.copy()
    result[ids] = result[ids] + delta
    return result


def max_score(patch_map: np.ndarray) -> float:
    value = np.asarray(patch_map, dtype=np.float32)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ValueError("Expected a finite, nonempty dense patch map")
    return float(value.max())


def patch_metrics(delta: np.ndarray, *, epsilon: float = EPS_PATCH) -> dict[str, Any]:
    values = np.asarray(delta)
    if values.ndim != 1 or not values.size or not np.issubdtype(values.dtype, np.number):
        raise R2AuditBlocked("Correction vector must be a nonempty numeric vector")
    work = values.astype(np.float64, copy=False)
    if not np.isfinite(work).all():
        raise R2AuditBlocked("Correction vector contains a nonfinite value")
    magnitude = np.abs(work)
    exact = work != 0.0
    tolerant = magnitude > float(epsilon)
    return {
        "count": int(work.size),
        "nonzero_exact_count": int(exact.sum()),
        "nonzero_tolerance_count": int(tolerant.sum()),
        "fraction_nonzero_exact": float(exact.mean()),
        "fraction_nonzero_tolerance": float(tolerant.mean()),
        "signed_mean": float(work.mean(dtype=np.float64)),
        "mean_absolute": float(magnitude.mean(dtype=np.float64)),
        "maximum_absolute": float(magnitude.max()),
        "positive_exact_count": int((work > 0.0).sum()),
        "negative_exact_count": int((work < 0.0).sum()),
        "positive_tolerance_count": int((work > epsilon).sum()),
        "negative_tolerance_count": int((work < -epsilon).sum()),
        "entirely_zero_exact": bool(not exact.any()),
        "entirely_zero_tolerance": bool(not tolerant.any()),
        "exact_nonzero_but_below_tolerance_count": int((exact & ~tolerant).sum()),
    }


def _validate_arrays(arrays: Mapping[str, np.ndarray], position: int) -> None:
    expected = {
        "base_patch_map": (np.dtype("float32"), (PATCH_VECTOR_SIZE,)),
        "top32_ids": (np.dtype("int64"), (PATCH_COUNT,)),
        "top32_features": (np.dtype("float32"), (PATCH_COUNT, 4096)),
        "c1_delta": (np.dtype("float32"), (PATCH_COUNT,)),
        "c2_delta": (np.dtype("float32"), (PATCH_COUNT,)),
    }
    for name, (dtype, shape) in expected.items():
        value = np.asarray(arrays.get(name))
        if value.dtype != dtype or value.shape != shape:
            raise R2AuditBlocked(
                f"Malformed event array {name} at position {position}: "
                f"expected {dtype}/{shape}, got {value.dtype}/{value.shape}")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise R2AuditBlocked(f"Nonfinite event array {name} at position {position}")
    ids = arrays["top32_ids"]
    if (np.any(ids < 0) or np.any(ids >= PATCH_VECTOR_SIZE) or
            len(set(ids.tolist())) != PATCH_COUNT):
        raise R2AuditBlocked(f"Invalid selected patch IDs at position {position}")


def verify_public_tape(reader: ReadOnlyR1Evidence, *,
                       expected_count: int = EVENT_COUNT,
                       expected_query_digest: str | None = None,
                       expected_fixed_bank_digest: str | None = None
                       ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify public tape identity and reconstruct all exact frozen scores."""
    sealed, sealed_raw = reader.read_sealed_records()
    transitions, transition_projection_sha256 = reader.read_transition_structure()
    if len(sealed) != expected_count or len(transitions) != expected_count:
        raise R2AuditBlocked("Sealed-score and structural-transition counts differ from authorization")
    validate_positions([row.get("position", -1) for row in sealed], expected_count)
    validate_positions([row.get("position", -1) for row in transitions], expected_count)

    query_positions: list[int] = []
    groups: list[str] = []
    events: list[dict[str, Any]] = []
    shard_receipts: list[dict[str, Any]] = []
    fixed_digests: set[str] = set()
    prior_after: dict[str, Any] | None = None
    for position, (row, transition) in enumerate(zip(sealed, transitions)):
        required_row = SEALED_SCORE_FIELDS
        required_transition = TRANSITION_FIELDS
        if not required_row.issubset(row) or not required_transition.issubset(transition):
            missing = sorted((required_row - set(row)) | (required_transition - set(transition)))
            raise R2AuditBlocked(f"Missing required public tape fields at {position}: {missing}")
        if transition["position"] != position or row["position"] != position:
            raise R2AuditBlocked("Public event tapes are not position-aligned")
        if transition["causal_order_complete"] is not True:
            raise R2AuditBlocked(f"Causal completion marker failed at position {position}")
        if not isinstance(transition["query_decision"], bool):
            raise R2AuditBlocked(f"Query decision is not boolean at position {position}")
        if transition["query_decision"]:
            query_positions.append(position)
        group = row["official_group_sha256"]
        if not isinstance(group, str) or not re.fullmatch(r"[0-9a-f]{64}", group):
            raise R2AuditBlocked(f"Malformed source-group digest at position {position}")
        groups.append(group)
        for digest_field in ("greedy_memory_sha256_before", "c1_state_sha256_before",
                             "c2_state_sha256_before", "fixed_support_sha256"):
            if not isinstance(row[digest_field], str) or not re.fullmatch(
                    r"[0-9a-f]{64}", row[digest_field]):
                raise R2AuditBlocked(f"Malformed {digest_field} at position {position}")
        fixed_digests.add(row["fixed_support_sha256"])
        if prior_after is not None:
            for before, after in (
                    ("greedy_memory_sha256_before", "greedy_memory_sha256_after"),
                    ("c1_state_sha256_before", "c1_state_sha256_after"),
                    ("c2_state_sha256_before", "c2_state_sha256_after")):
                if row[before] != prior_after[after]:
                    raise R2AuditBlocked(
                        f"State continuity mismatch for {before} at position {position}")
            if int(row["greedy_memory_version_before"]) != int(
                    prior_after["greedy_memory_version_after"]):
                raise R2AuditBlocked(f"Greedy memory version discontinuity at {position}")

        expected_shard = f"events/event_{position:06d}.npz"
        if row["shard_path"] != expected_shard:
            raise R2AuditBlocked(f"Unexpected shard path at position {position}")
        arrays, shard_raw = reader.read_event_shard(position)
        _validate_arrays(arrays, position)
        observed_shard_sha = _sha256_bytes(shard_raw)
        if observed_shard_sha != row["shard_sha256"]:
            raise R2AuditBlocked(f"Shard SHA-256 mismatch at position {position}")
        observed_array_sha = _arrays_sha256(arrays)
        if observed_array_sha != row["array_sha256"]:
            raise R2AuditBlocked(f"Array SHA-256 mismatch at position {position}")
        if sorted(arrays) != row["array_fields"]:
            raise R2AuditBlocked(f"Array field receipt mismatch at position {position}")

        base = arrays["base_patch_map"]
        ids = arrays["top32_ids"]
        c1_map = corrected_dense_map(base, ids, arrays["c1_delta"])
        c2_map = corrected_dense_map(base, ids, arrays["c2_delta"])
        scores = {"c0": max_score(base), "c1": max_score(c1_map), "c2": max_score(c2_map)}
        for arm, score in scores.items():
            recorded = float(row[f"{arm}_score"])
            if not math.isfinite(recorded) or score != recorded:
                raise R2AuditBlocked(
                    f"Exact {arm.upper()} score reconstruction mismatch at position {position}")
        eligible = {
            "c1": int(row["c1_eligible_anchor_slots"]),
            "c2": int(row["c2_eligible_anchor_count"]),
        }
        if min(eligible.values()) < 0:
            raise R2AuditBlocked(f"Negative eligible-anchor count at position {position}")
        same_source_excluded = int(row["c1_same_source_excluded_anchor_slots"])
        if same_source_excluded < 0:
            raise R2AuditBlocked(f"Negative C1 source-exclusion count at position {position}")
        far_count = int(row["c1_far_field_patch_count"])
        far_max = float(row["c1_far_field_abs_correction_max"])
        if far_count < 0 or not math.isfinite(far_max) or far_max < 0:
            raise R2AuditBlocked(f"Invalid C1 far-field diagnostic at position {position}")

        events.append({
            "position": position,
            "source_group_sha256": group,
            "arrays": arrays,
            "scores": scores,
            "maps": {"c0": base, "c1": c1_map, "c2": c2_map},
            "eligible": eligible,
            "c1_same_source_excluded_anchor_slots": same_source_excluded,
            "c1_far_field_patch_count": far_count,
            "c1_far_field_abs_correction_max": far_max,
        })
        shard_receipts.append({
            "position": position,
            "shard_sha256": observed_shard_sha,
            "array_sha256": observed_array_sha,
            "score_reconstruction_passed": True,
        })
        prior_after = transition

    if len(fixed_digests) != 1:
        raise R2AuditBlocked("Fixed-support digest changed across sealed events")
    observed_fixed_digest = next(iter(fixed_digests))
    if expected_fixed_bank_digest and observed_fixed_digest != expected_fixed_bank_digest:
        raise R2AuditBlocked("Sealed-event fixed-bank digest differs from R1 report")
    if len(set(query_positions)) != len(query_positions):
        raise R2AuditBlocked("Duplicate query position in structural transition projection")
    query_digest = _sha256_json(query_positions)
    if expected_query_digest and query_digest != expected_query_digest:
        raise R2AuditBlocked("Frozen query-position digest mismatch")
    if len(set(groups)) == 0:
        raise R2AuditBlocked("No source groups found in sealed scores")
    receipt = {
        "event_count": expected_count,
        "sealed_count": len(sealed),
        "finalized_count": len(transitions),
        "positions_sha256": _sha256_json(list(range(expected_count))),
        "query_positions": query_positions,
        "query_positions_sha256": query_digest,
        "query_count": len(query_positions),
        "unique_source_group_count": len(set(groups)),
        "fixed_support_sha256": observed_fixed_digest,
        "sealed_scores_file_sha256": _sha256_bytes(sealed_raw),
        "transition_structural_projection_sha256": transition_projection_sha256,
        "verified_shards": shard_receipts,
        "all_recorded_shard_and_array_hashes_verified": True,
        "all_c0_c1_c2_scores_reconstructed_exactly": True,
    }
    return events, receipt


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        average = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        return None
    ac = a - a.mean()
    bc = b - b.mean()
    denominator = float(np.sqrt(np.dot(ac, ac) * np.dot(bc, bc)))
    if denominator == 0.0:
        return None
    return float(np.dot(ac, bc) / denominator)


def _pair_rank_summary(base: Sequence[float], corrected: Sequence[float],
                       groups: Sequence[str] | None = None) -> dict[str, Any]:
    x = np.asarray(base, dtype=np.float64)
    y = np.asarray(corrected, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise R2AuditBlocked("Rank analysis requires aligned finite score vectors")
    if groups is not None and len(groups) != len(x):
        raise R2AuditBlocked("Source-group vector differs from score count")
    rx, ry = _average_ranks(x), _average_ranks(y)
    pair_rx: list[float] = []
    pair_ry: list[float] = []
    total = same_group = ties_x = ties_y = ties_both = 0
    strict_comparable = reversed_pairs = newly_tied = newly_untied = 0
    concordant = 0
    participants: set[int] = set()
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if groups is not None and groups[i] == groups[j]:
                same_group += 1
                continue
            total += 1
            dx, dy = float(x[i] - x[j]), float(y[i] - y[j])
            pair_rx.append(float(rx[i] - rx[j]))
            pair_ry.append(float(ry[i] - ry[j]))
            tie_x = dx == 0.0
            tie_y = dy == 0.0
            ties_x += int(tie_x)
            ties_y += int(tie_y)
            ties_both += int(tie_x and tie_y)
            newly_tied += int(not tie_x and tie_y)
            newly_untied += int(tie_x and not tie_y)
            if not tie_x and not tie_y:
                strict_comparable += 1
                if dx * dy < 0:
                    reversed_pairs += 1
                    participants.update((i, j))
                else:
                    concordant += 1
    tau_denominator = math.sqrt(max(0, total - ties_x) * max(0, total - ties_y))
    tau_b = ((concordant - reversed_pairs) / tau_denominator
             if tau_denominator else None)
    rank_difference_corr = _pearson(pair_rx, pair_ry)
    return {
        "event_count": int(len(x)),
        "pair_scope": "all_pairs" if groups is None else "cross_source_group_pairs_only",
        "total_pairs_in_scope": int(total),
        "same_source_group_pairs_excluded": int(same_group),
        "comparable_pairs_untied_in_both": int(strict_comparable),
        "strict_reversed_pairs": int(reversed_pairs),
        "strict_reversal_fraction_of_comparable_pairs": (
            float(reversed_pairs / strict_comparable) if strict_comparable else None),
        "events_participating_in_at_least_one_strict_reversal": int(len(participants)),
        "newly_tied_pairs": int(newly_tied),
        "newly_untied_pairs": int(newly_untied),
        "ties_in_baseline": int(ties_x),
        "ties_in_corrected": int(ties_y),
        "ties_in_both": int(ties_both),
        "spearman_rank_difference_correlation": rank_difference_corr,
        "spearman_definition": (
            "Pearson correlation of global average-rank differences across in-scope pairs; "
            "equals conventional Spearman rho for the all-pair scope"),
        "kendall_tau_b": tau_b,
        "kendall_tau_b_defined": tau_b is not None,
    }


def ranking_summary(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups = [str(event["source_group_sha256"]) for event in events]
    scores0 = [float(event["scores"]["c0"]) for event in events]
    unique_groups = len(set(groups))
    same_group_pairs = sum(1 for i in range(len(groups)) for j in range(i + 1, len(groups))
                           if groups[i] == groups[j])
    total_pairs = len(groups) * (len(groups) - 1) // 2
    output: dict[str, Any] = {
        "event_count": len(events),
        "unique_source_group_count": unique_groups,
        "all_event_pairs": total_pairs,
        "same_source_group_pairs": same_group_pairs,
        "cross_source_group_pairs": total_pairs - same_group_pairs,
        "arms": {},
    }
    for arm in ("c1", "c2"):
        scores_arm = [float(event["scores"][arm]) for event in events]
        output["arms"][arm] = {
            "all_event_pairs": _pair_rank_summary(scores0, scores_arm),
            "cross_source_group_pairs": _pair_rank_summary(scores0, scores_arm, groups),
        }
    return output


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(a.size), "min": float(a.min()), "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)), "max": float(a.max()),
    }


def _score_change_summary(events: Sequence[Mapping[str, Any]], arm: str,
                          first_query_position: int) -> dict[str, Any]:
    changes = np.asarray([event["scores"][arm] - event["scores"]["c0"]
                          for event in events], dtype=np.float64)
    exact_changed = changes != 0.0
    changed = np.abs(changes) > EPS_SCORE
    before = np.asarray([int(event["position"]) <= first_query_position for event in events])
    after = ~before

    def window(mask: np.ndarray) -> dict[str, Any]:
        subset = changes[mask]
        changed_subset = np.abs(subset) > EPS_SCORE
        return {
            "events": int(mask.sum()),
            "changed_events_eps_score": int(changed_subset.sum()),
            "changed_rate_eps_score": float(changed_subset.mean()) if len(subset) else None,
            "positive_changes_eps_score": int((subset > EPS_SCORE).sum()),
            "negative_changes_eps_score": int((subset < -EPS_SCORE).sum()),
            "absolute_change_distribution": _distribution(np.abs(subset)),
        }

    return {
        "epsilon_score": EPS_SCORE,
        "events": len(events),
        "exactly_changed_events": int(exact_changed.sum()),
        "changed_events_eps_score": int(changed.sum()),
        "changed_percentage_eps_score": float(100.0 * changed.mean()),
        "positive_changes_eps_score": int((changes > EPS_SCORE).sum()),
        "negative_changes_eps_score": int((changes < -EPS_SCORE).sum()),
        "within_tolerance_events": int((~changed).sum()),
        "absolute_change_distribution": _distribution(np.abs(changes)),
        "before_first_query_feedback_score_window_positions": [0, first_query_position],
        "after_first_query_feedback_score_window_positions": [first_query_position + 1,
                                                                len(events) - 1],
        "before_first_query_feedback": window(before),
        "after_first_query_feedback": window(after),
    }


def _argmax_diagnostics(base: np.ndarray, corrected: np.ndarray,
                        ids: np.ndarray) -> dict[str, Any]:
    selected = np.zeros(base.size, dtype=bool)
    selected[ids] = True
    base_max = float(np.max(base))
    corrected_max = float(np.max(corrected))
    base_argmax = int(np.argmax(base))
    corrected_argmax = int(np.argmax(corrected))
    selected_max = float(np.max(corrected[ids]))
    unchanged_max = float(np.max(base[~selected]))
    gap = selected_max - unchanged_max
    return {
        "base_max": base_max,
        "corrected_max": corrected_max,
        "base_argmax": base_argmax,
        "corrected_argmax": corrected_argmax,
        "base_argmax_in_top32": bool(selected[base_argmax]),
        "corrected_argmax_in_top32": bool(selected[corrected_argmax]),
        "argmax_changed": bool(base_argmax != corrected_argmax),
        "base_exact_max_tie_count": int(np.count_nonzero(base == base_max)),
        "corrected_exact_max_tie_count": int(np.count_nonzero(corrected == corrected_max)),
        "base_near_max_tie_count_eps_score": int(np.count_nonzero(base_max - base <= EPS_SCORE)),
        "corrected_near_max_tie_count_eps_score": int(
            np.count_nonzero(corrected_max - corrected <= EPS_SCORE)),
        "selected_corrected_max": selected_max,
        "unchanged_patch_max": unchanged_max,
        "selected_vs_unchanged_exact_relation": (
            "selected_wins" if gap > 0 else "exact_tie" if gap == 0 else "unchanged_wins"),
        "selected_unchanged_near_tie_eps_score": bool(abs(gap) <= EPS_SCORE),
        "corrected_top32_competed_with_unchanged": bool(selected_max >= unchanged_max - EPS_SCORE),
    }


def _eligibility_category(eligible: int, patch_count: int) -> str:
    if eligible == 0:
        return "NO_ELIGIBLE_ANCHORS"
    if patch_count == 0:
        return "ELIGIBLE_ANCHORS_BUT_ZERO_PATCH_CORRECTION"
    return "NONZERO_PATCH_CORRECTION"


def _eligible_distribution(events: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    values = [int(event["eligible"][arm]) for event in events]
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return {
        "events": len(values),
        "zero_eligible_anchor_events": int(sum(value == 0 for value in values)),
        "eligible_anchor_count_distribution": _distribution(values),
        "eligible_anchor_count_histogram": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
    }


def analyze_events(events: Sequence[Mapping[str, Any]], *,
                   manifest: Mapping[str, Any],
                   query_positions: Sequence[int]
                   ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if len(events) != EVENT_COUNT:
        raise R2AuditBlocked("Mechanism analysis requires exactly 512 verified events")
    c1_patches: list[dict[str, Any]] = []
    c2_patches: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for event in events:
        pos = int(event["position"])
        arrays = event["arrays"]
        base = arrays["base_patch_map"]
        ids = arrays["top32_ids"]
        row: dict[str, Any] = {
            "position": pos,
            "source_group_sha256": event["source_group_sha256"],
            "c0_score": event["scores"]["c0"],
            "c1_score": event["scores"]["c1"],
            "c2_score": event["scores"]["c2"],
            "delta_c1": event["scores"]["c1"] - event["scores"]["c0"],
            "delta_c2": event["scores"]["c2"] - event["scores"]["c0"],
            "c1_eligible_anchor_slots": event["eligible"]["c1"],
            "c1_same_source_excluded_anchor_slots": event[
                "c1_same_source_excluded_anchor_slots"],
            "c2_eligible_anchor_count": event["eligible"]["c2"],
            "c1_far_field_patch_count": event["c1_far_field_patch_count"],
            "c1_far_field_abs_correction_max": event["c1_far_field_abs_correction_max"],
        }
        for arm in ("c1", "c2"):
            delta = arrays[f"{arm}_delta"]
            pm = patch_metrics(delta)
            category = _eligibility_category(event["eligible"][arm],
                                             pm["nonzero_tolerance_count"])
            (c1_patches if arm == "c1" else c2_patches).append(pm)
            prefix = arm
            for key, value in pm.items():
                row[f"{prefix}_{key}"] = value
            row[f"{prefix}_activation_category"] = category
            for epsilon in PATCH_TOLERANCE_SENSITIVITY:
                row[f"{prefix}_nonzero_count_eps_{epsilon:g}"] = int(
                    np.count_nonzero(np.abs(delta.astype(np.float64)) > epsilon))
            corrected = event["maps"][arm]
            argmax = _argmax_diagnostics(base, corrected, ids)
            for key, value in argmax.items():
                row[f"{prefix}_{key}"] = value
            score_delta = float(event["scores"][arm] - event["scores"]["c0"])
            patch_active = pm["nonzero_tolerance_count"] > 0
            row[f"{prefix}_score_changed_exact"] = score_delta != 0.0
            row[f"{prefix}_score_changed_eps_score"] = abs(score_delta) > EPS_SCORE
            row[f"{prefix}_patch_active_score_unchanged_eps_score"] = (
                patch_active and abs(score_delta) <= EPS_SCORE)
            row[f"{prefix}_patch_active_score_unchanged_exact"] = (
                pm["nonzero_exact_count"] > 0 and score_delta == 0.0)
        diagnostics.append(row)

    query_positions = [int(position) for position in query_positions]
    if not query_positions:
        raise R2AuditBlocked("Frozen query schedule is empty")
    first_query = int(query_positions[0])
    groups = [str(event["source_group_sha256"]) for event in events]
    ranks = ranking_summary(events)
    for arm in ("c1", "c2"):
        part_all = ranks["arms"][arm]["all_event_pairs"][
            "events_participating_in_at_least_one_strict_reversal"]
        part_cross = ranks["arms"][arm]["cross_source_group_pairs"][
            "events_participating_in_at_least_one_strict_reversal"]
        all_participants: set[int] = set()
        cross_participants: set[int] = set()
        base = [float(event["scores"]["c0"]) for event in events]
        corrected = [float(event["scores"][arm]) for event in events]
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                if (base[i] != base[j] and corrected[i] != corrected[j] and
                        (base[i] - base[j]) * (corrected[i] - corrected[j]) < 0):
                    all_participants.update((i, j))
                    if groups[i] != groups[j]:
                        cross_participants.update((i, j))
        if len(all_participants) != part_all or len(cross_participants) != part_cross:
            raise R2AuditBlocked("Rank reversal participant count is internally inconsistent")
        for index, row in enumerate(diagnostics):
            row[f"{arm}_participates_in_strict_reversal_all_pairs"] = index in all_participants
            row[f"{arm}_participates_in_strict_reversal_cross_group"] = index in cross_participants

    score_effects = {
        arm: _score_change_summary(events, arm, first_query)
        for arm in ("c1", "c2")
    }
    by_arm: dict[str, Any] = {}
    for arm, patch_rows in (("c1", c1_patches), ("c2", c2_patches)):
        count_exact = sum(row["nonzero_exact_count"] for row in patch_rows)
        count_eps = sum(row["nonzero_tolerance_count"] for row in patch_rows)
        event_nonzero = sum(row["nonzero_tolerance_count"] > 0 for row in patch_rows)
        eligible = _eligible_distribution(events, arm)
        eligible_zero_correction = sum(
            int(event["eligible"][arm] > 0 and
                patch_rows[i]["nonzero_tolerance_count"] == 0)
            for i, event in enumerate(events))
        by_arm[arm] = {
            "total_scored_events": len(events),
            "eligible_anchor_diagnostics": eligible,
            "events_with_eligible_anchors_but_zero_patch_correction_eps_patch":
                eligible_zero_correction,
            "events_with_nonzero_patch_correction_eps_patch": int(event_nonzero),
            "patch_activation": {
                "selected_patch_entries": int(len(events) * PATCH_COUNT),
                "exact_float32_nonzero_patch_entries": int(count_exact),
                "nonzero_patch_entries_eps_patch": int(count_eps),
                "events_no_eligible_anchors": eligible["zero_eligible_anchor_events"],
                "events_eligible_but_zero_patch_effect": int(eligible_zero_correction),
                "events_with_patch_effect": int(event_nonzero),
                "tolerance_sensitivity_entry_counts": {
                    f"{epsilon:g}": int(sum(np.count_nonzero(
                        np.abs(event["arrays"][f"{arm}_delta"].astype(np.float64)) > epsilon)
                        for event in events))
                    for epsilon in PATCH_TOLERANCE_SENSITIVITY
                },
            },
            "score_effects": score_effects[arm],
            "events_nonzero_patch_but_score_unchanged_eps_score": int(sum(
                row[f"{arm}_patch_active_score_unchanged_eps_score"] for row in diagnostics)),
            "events_nonzero_patch_but_score_unchanged_exact": int(sum(
                row[f"{arm}_patch_active_score_unchanged_exact"] for row in diagnostics)),
            "events_argmax_changed": int(sum(row[f"{arm}_argmax_changed"]
                                              for row in diagnostics)),
            "rank_reversal_participants_all_pairs": int(
                ranks["arms"][arm]["all_event_pairs"][
                    "events_participating_in_at_least_one_strict_reversal"]),
            "rank_reversal_participants_cross_group": int(
                ranks["arms"][arm]["cross_source_group_pairs"][
                    "events_participating_in_at_least_one_strict_reversal"]),
        }
    c1_excluded = [int(event["c1_same_source_excluded_anchor_slots"]) for event in events]
    c1_far_counts = [int(event["c1_far_field_patch_count"]) for event in events]
    c1_far_max = [float(event["c1_far_field_abs_correction_max"]) for event in events]
    capacity = int(manifest["method_identities"]["C2"]["capacity"])
    anchors_per_query = C2_PATCHES_PER_QUERY
    full_query_index = math.ceil(capacity / anchors_per_query)
    saturation_query_position = (query_positions[full_query_index - 1]
                                 if len(query_positions) >= full_query_index else None)
    first_saturated_score_position = (saturation_query_position + 1
                                      if saturation_query_position is not None else None)
    c2_pre_saturation = [row for row in diagnostics
                         if first_saturated_score_position is None or
                         int(row["position"]) < first_saturated_score_position]
    c2_post_saturation = [row for row in diagnostics
                          if first_saturated_score_position is not None and
                          int(row["position"]) >= first_saturated_score_position]
    c2_fifo = {
        "capacity": capacity,
        "retention": str(manifest["method_identities"]["C2"]["retention"]),
        "patch_anchors_added_per_query_event": anchors_per_query,
        "queries_to_reach_capacity": full_query_index,
        "query_position_that_fills_fifo": saturation_query_position,
        "first_later_score_position_with_full_history": first_saturated_score_position,
        "pre_saturation_event_count": len(c2_pre_saturation),
        "post_saturation_event_count": len(c2_post_saturation),
        "pre_saturation_nonzero_patch_events_eps_patch": sum(
            row["c2_nonzero_tolerance_count"] > 0 for row in c2_pre_saturation),
        "post_saturation_nonzero_patch_events_eps_patch": sum(
            row["c2_nonzero_tolerance_count"] > 0 for row in c2_post_saturation),
        "pre_saturation_score_change_events_eps_score": sum(
            row["c2_score_changed_eps_score"] for row in c2_pre_saturation),
        "post_saturation_score_change_events_eps_score": sum(
            row["c2_score_changed_eps_score"] for row in c2_post_saturation),
    }
    funnel = {
        arm: {
            "total_scored_events": len(events),
            "events_with_eligible_anchors": int(sum(
                int(event["eligible"][arm]) > 0 for event in events)),
            "events_with_nonzero_patch_correction_eps_patch": int(sum(
                row[f"{arm}_nonzero_tolerance_count"] > 0 for row in diagnostics)),
            "events_with_nonzero_image_score_change_eps_score": int(sum(
                row[f"{arm}_score_changed_eps_score"] for row in diagnostics)),
            "events_participating_in_at_least_one_strict_rank_reversal_all_pairs":
                by_arm[arm]["rank_reversal_participants_all_pairs"],
            "event_denominator": len(events),
            "pairwise_reversal_denominator_comparable_pairs": ranks["arms"][arm][
                "all_event_pairs"]["comparable_pairs_untied_in_both"],
        }
        for arm in ("c1", "c2")
    }
    mechanism_observable = any(
        by_arm[arm]["patch_activation"]["nonzero_patch_entries_eps_patch"] > 0 or
        by_arm[arm]["score_effects"]["changed_events_eps_score"] > 0 or
        ranks["arms"][arm]["all_event_pairs"]["strict_reversed_pairs"] > 0 or
        ranks["arms"][arm]["all_event_pairs"]["newly_tied_pairs"] > 0 or
        ranks["arms"][arm]["all_event_pairs"]["newly_untied_pairs"] > 0
        for arm in ("c1", "c2"))
    summary = {
        "classification": ("R2_0_MECHANISM_OBSERVABLE" if mechanism_observable
                           else "R2_0_MECHANISM_INACTIVE"),
        "interpretation": "Label-free mechanism observability only; not detection effectiveness.",
        "event_count": len(events),
        "patch_epsilon": EPS_PATCH,
        "score_epsilon": EPS_SCORE,
        "patch_tolerance_sensitivity": list(PATCH_TOLERANCE_SENSITIVITY),
        "score_tolerance_sensitivity": list(SCORE_TOLERANCE_SENSITIVITY),
        "first_query_position": first_query,
        "first_query_score_is_pre_disclosure": True,
        "first_feedback_can_affect_first_later_score_position": first_query + 1,
        "score_effects": score_effects,
        "arms": by_arm,
        "c1_same_source_excluded_anchor_slots_distribution": _distribution(c1_excluded),
        "c1_far_field_selected_patch_observations": int(sum(c1_far_counts)),
        "c1_far_field_selected_patch_observations_by_event": _distribution(c1_far_counts),
        "c1_far_field_abs_correction_max_by_event": _distribution(c1_far_max),
        "c1_far_field_abs_correction_max_recorded_manifest": float(
            manifest["c1_far_field_abs_correction_max"]),
        "c2_fifo_saturation_diagnostics": c2_fifo,
        "mechanism_funnel": funnel,
        "ranking_pair_scope_summary": {
            "all_event_pairs": ranks["all_event_pairs"],
            "same_source_group_pairs_excluded": ranks["same_source_group_pairs"],
            "cross_source_group_pairs": ranks["cross_source_group_pairs"],
            "unique_source_group_count": ranks["unique_source_group_count"],
        },
        "label_blind": True,
        "scientific_accuracy_metrics_computed": False,
    }
    return summary, ranks, diagnostics


def _manifest_integrity_gate(manifest: Mapping[str, Any], *,
                             report_text: str,
                             repo_root: Path) -> dict[str, Any]:
    required_identity = {
        "status": "ENGINEERING_PASS",
        "run_id": R1_RUN_ID,
        "source_commit": FROZEN_R1_SOURCE_COMMIT,
        "expected_source_commit": FROZEN_R1_SOURCE_COMMIT,
        "branch": EXPECTED_BRANCH,
        "profile_id": "r1_switch_s0_512_digest_contract_v1",
        "sealed_events": EVENT_COUNT,
        "finalized_events": EVENT_COUNT,
        "events_processed": EVENT_COUNT,
        "unique_stream_positions": EVENT_COUNT,
        "query_positions_in_prefix": 17,
        "query_positions_sha256": QUERY_POSITIONS_SHA256,
        "query_positions_decoded_sha256": QUERY_POSITIONS_SHA256,
        "query_binding_sha256": QUERY_BINDING_SHA256,
        "query_verification_sha256": QUERY_VERIFICATION_SHA256,
        "unqueried_labels_accessed": False,
        "masks_accessed": False,
        "confirmation_data_accessed": False,
        "scientific_outcome_evaluation": False,
        "auroc_computed": False,
        "auprc_computed": False,
        "classification_metrics_computed": False,
    }
    mismatches = {key: {"expected": expected, "actual": manifest.get(key)}
                  for key, expected in required_identity.items()
                  if manifest.get(key) != expected}
    if mismatches:
        raise R2AuditBlocked(f"R1 manifest identity/integrity gate failed: {mismatches}")
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise R2AuditBlocked("R1 manifest has no source hash inventory")
    source_mismatches: list[str] = []
    for relative, expected_sha in source_hashes.items():
        rel_path = PurePosixPath(str(relative))
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise R2AuditBlocked("R1 source hash inventory contains an unsafe path")
        source_path = _require_plain_path(repo_root.joinpath(*rel_path.parts), must_exist=True)
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if digest != expected_sha:
            source_mismatches.append(str(relative))
    if source_mismatches:
        raise R2AuditBlocked("R1 source SHA mismatch: " + ", ".join(source_mismatches))
    report_match = re.search(
        r"Structured digest before/after and expected receipt: `([0-9a-f]{64})`",
        report_text)
    report_raw_match = re.search(
        r"Raw-prototype diagnostic before/after: `([0-9a-f]{64})`", report_text)
    if not report_match or not report_raw_match:
        raise R2AuditBlocked("R1 execution report lacks the structured fixed-bank digest")
    report_fixed_digest = report_match.group(1)
    report_raw_digest = report_raw_match.group(1)
    if (report_fixed_digest != FIXED_BANK_SHA256 or
            manifest.get("fixed_support_sha256_expected") != report_fixed_digest or
            manifest.get("fixed_support_sha256") != report_fixed_digest or
            manifest.get("fixed_support_sha256_final") != report_fixed_digest or
            manifest.get("fixed_support_raw_prototype_sha256") != report_raw_digest or
            manifest.get("fixed_support_raw_prototype_sha256_final") != report_raw_digest):
        raise R2AuditBlocked("R1 fixed-bank digest differs from its execution report")
    if manifest.get("evidence_tape_readback", {}).get("passed") is not True:
        raise R2AuditBlocked("R1 manifest does not report successful evidence tape readback")
    return {
        "run_id": R1_RUN_ID,
        "branch": EXPECTED_BRANCH,
        "expected_repository_head_before_audit": EXPECTED_HEAD,
        "frozen_r1_source_commit": FROZEN_R1_SOURCE_COMMIT,
        "r1_manifest_status": manifest["status"],
        "source_hash_count_verified": len(source_hashes),
        "source_hashes_match_current_files": True,
        "report_fixed_bank_sha256": report_fixed_digest,
        "report_raw_prototype_sha256": report_raw_digest,
        "manifest_expected_fixed_bank_sha256": manifest["fixed_support_sha256_expected"],
        "manifest_final_fixed_bank_sha256": manifest["fixed_support_sha256_final"],
        "query_binding_sha256": manifest["query_binding_sha256"],
        "query_verification_sha256": manifest["query_verification_sha256"],
        "query_positions_sha256": manifest["query_positions_sha256"],
    }


def _verify_repository_identity(repo_root: Path) -> dict[str, str]:
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"],
                                         cwd=repo_root, text=True).strip()
        head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=repo_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise R2AuditBlocked(f"Cannot verify repository identity: {exc}") from exc
    if branch != EXPECTED_BRANCH or head != EXPECTED_HEAD:
        raise R2AuditBlocked(
            f"Repository identity mismatch: expected {EXPECTED_BRANCH}@{EXPECTED_HEAD}; "
            f"found {branch}@{head}")
    return {"branch": branch, "head": head}


def _load_report(repo_root: Path) -> tuple[str, str]:
    path = _require_plain_path(repo_root / R1_REPORT_RELATIVE, must_exist=True)
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8"), _sha256_bytes(raw)
    except UnicodeDecodeError as exc:
        raise R2AuditBlocked("R1 execution report is not UTF-8") from exc


def _write_new(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise R2AuditBlocked(f"Refusing to overwrite audit output: {path}") from exc


CSV_FIELDS = [
    "position", "query_decision", "source_group_sha256", "c0_score", "c1_score", "c2_score",
    "delta_c1", "delta_c2", "c1_eligible_anchor_slots",
    "c1_same_source_excluded_anchor_slots", "c2_eligible_anchor_count",
    "c1_far_field_patch_count", "c1_far_field_abs_correction_max",
]
for _arm in ("c1", "c2"):
    CSV_FIELDS.extend([
        f"{_arm}_count", f"{_arm}_nonzero_exact_count", f"{_arm}_nonzero_tolerance_count",
        f"{_arm}_fraction_nonzero_exact", f"{_arm}_fraction_nonzero_tolerance",
        f"{_arm}_signed_mean", f"{_arm}_mean_absolute", f"{_arm}_maximum_absolute",
        f"{_arm}_positive_exact_count", f"{_arm}_negative_exact_count",
        f"{_arm}_positive_tolerance_count", f"{_arm}_negative_tolerance_count",
        f"{_arm}_entirely_zero_exact", f"{_arm}_entirely_zero_tolerance",
        f"{_arm}_exact_nonzero_but_below_tolerance_count", f"{_arm}_activation_category",
        *[f"{_arm}_nonzero_count_eps_{epsilon:g}" for epsilon in PATCH_TOLERANCE_SENSITIVITY],
        f"{_arm}_base_max", f"{_arm}_corrected_max", f"{_arm}_base_argmax",
        f"{_arm}_corrected_argmax", f"{_arm}_base_argmax_in_top32",
        f"{_arm}_corrected_argmax_in_top32", f"{_arm}_argmax_changed",
        f"{_arm}_base_exact_max_tie_count", f"{_arm}_corrected_exact_max_tie_count",
        f"{_arm}_base_near_max_tie_count_eps_score",
        f"{_arm}_corrected_near_max_tie_count_eps_score",
        f"{_arm}_selected_corrected_max", f"{_arm}_unchanged_patch_max",
        f"{_arm}_selected_vs_unchanged_exact_relation",
        f"{_arm}_selected_unchanged_near_tie_eps_score",
        f"{_arm}_corrected_top32_competed_with_unchanged",
        f"{_arm}_score_changed_exact", f"{_arm}_score_changed_eps_score",
        f"{_arm}_patch_active_score_unchanged_eps_score",
        f"{_arm}_patch_active_score_unchanged_exact",
        f"{_arm}_participates_in_strict_reversal_all_pairs",
        f"{_arm}_participates_in_strict_reversal_cross_group",
    ])


def _serialize_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction="raise",
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def run_audit(repo_root: Path = REPOSITORY_ROOT, *,
              output_dir: Path | None = None) -> dict[str, Any]:
    repo_root = _require_plain_path(Path(repo_root), must_exist=True)
    repository_identity = _verify_repository_identity(repo_root)
    expected_run_dir = repo_root / R1_RUN_RELATIVE
    reader = ReadOnlyR1Evidence(expected_run_dir)
    default_output = repo_root / "outputs" / "calibration" / R2_OUTPUT_NAME
    output = Path(output_dir) if output_dir is not None else default_output
    output = Path(os.path.abspath(output))
    if output != default_output:
        raise R2AuditBlocked("Audit output must use the task's designated unique namespace")
    if output.exists():
        raise R2AuditBlocked(f"R2-0 output namespace already exists: {output}")
    if not output.parent.is_dir() or output.is_symlink():
        raise R2AuditBlocked("R2-0 output parent is missing or unsafe")

    manifest, manifest_raw = reader.read_manifest()
    report_text, report_sha256 = _load_report(repo_root)
    manifest_receipt = _manifest_integrity_gate(
        manifest, report_text=report_text, repo_root=repo_root)
    events, tape_receipt = verify_public_tape(
        reader, expected_count=EVENT_COUNT,
        expected_query_digest=QUERY_POSITIONS_SHA256,
        expected_fixed_bank_digest=manifest_receipt["report_fixed_bank_sha256"])
    query_positions = tape_receipt["query_positions"]
    if len(query_positions) != 17:
        raise R2AuditBlocked("Frozen public query schedule does not contain exactly 17 positions")
    if (manifest.get("query_positions_sha256") != tape_receipt["query_positions_sha256"] or
            manifest.get("query_positions_decoded_sha256") != tape_receipt["query_positions_sha256"]):
        raise R2AuditBlocked("Manifest query digest differs from public transition schedule")

    manifest_safe_fields = {
        key: manifest.get(key) for key in (
            "run_id", "status", "source_commit", "expected_source_commit", "branch",
            "profile_id", "sealed_events", "finalized_events", "events_processed",
            "query_positions_sha256", "query_positions_decoded_sha256",
            "query_binding_sha256", "query_verification_sha256", "fixed_support_sha256",
            "fixed_support_sha256_expected", "fixed_support_sha256_final",
            "fixed_support_digest_algorithm", "fixed_support_raw_digest_algorithm",
            "fixed_support_raw_prototype_sha256", "fixed_support_raw_prototype_sha256_final",
            "source_hashes", "native_scorer_profile_sha256", "query_protocol_profile_sha256",
            "encoder_fingerprint_sha256", "preprocess_fingerprint_sha256",
            "hub_source_tree_sha256", "loaded_model_state_sha256",
            "c1_far_field_selected_patch_observations", "c1_far_field_abs_correction_max",
            "method_identities")
    }
    integrity = {
        "classification": "R2_0_INTEGRITY_PASS",
        "repository_identity": repository_identity,
        "r1_identity": manifest_receipt,
        "run_manifest_sha256": _sha256_bytes(manifest_raw),
        "r1_execution_report_sha256": report_sha256,
        "public_tape": tape_receipt,
        "manifest_safe_fields": manifest_safe_fields,
        "label_bearing_files_opened": [],
        "feedback_derived_transition_fields_decoded": [],
        "integrity_passed_before_mechanism_analysis": True,
    }
    summary, ranks, diagnostics = analyze_events(
        events, manifest=manifest_safe_fields, query_positions=query_positions)
    summary["query_positions"] = query_positions
    summary["query_positions_sha256"] = tape_receipt["query_positions_sha256"]
    summary["unique_source_group_count"] = tape_receipt["unique_source_group_count"]
    summary["frozen_r1_source_commit"] = FROZEN_R1_SOURCE_COMMIT
    summary["repository_head_at_audit_start"] = EXPECTED_HEAD
    summary["r1_method_identities"] = manifest_safe_fields["method_identities"]

    # Add exact-vs-tolerance rates and query-window labels using only the public
    # schedule; the first query's score is sealed before its feedback is legal.
    for arm in ("c1", "c2"):
        for epsilon in SCORE_TOLERANCE_SENSITIVITY:
            changes = [abs(float(event["scores"][arm] - event["scores"]["c0"])) > epsilon
                       for event in events]
            summary["arms"][arm]["score_effects"].setdefault(
                "tolerance_sensitivity_changed_event_counts", {})[f"{epsilon:g}"] = int(sum(changes))
    query_set = set(query_positions)
    for row, event in zip(diagnostics, events):
        row["query_decision"] = int(int(event["position"]) in query_set)

    # Determine report classification after verifying integrity and measuring
    # effects. This is structural observability, never scientific effectiveness.
    output.mkdir(parents=False, exist_ok=False)
    _write_new(output / "integrity_receipt.json",
               (_canonical_json(integrity) + "\n").encode("utf-8"))
    _write_new(output / "mechanism_summary.json",
               (_canonical_json(summary) + "\n").encode("utf-8"))
    _write_new(output / "rank_change_summary.json",
               (_canonical_json(ranks) + "\n").encode("utf-8"))
    _write_new(output / "event_mechanism_diagnostics.csv", _serialize_csv(diagnostics))
    return {
        "classification": summary["classification"],
        "integrity": integrity["classification"],
        "output_directory": str(output),
        "events": len(diagnostics),
        "query_count": len(query_positions),
        "unique_source_group_count": tape_receipt["unique_source_group_count"],
        "mechanism_summary": summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args(argv)
    try:
        result = run_audit(args.repo_root)
    except R2AuditBlocked as exc:
        print(f"R2_0_AUDIT_BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
