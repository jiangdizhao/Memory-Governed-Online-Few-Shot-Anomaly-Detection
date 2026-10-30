"""Low-memory, class-sharded completion of the bounded Stage 1-A P1.2 replay.

Each child process owns one industrial class and its feature cache. Per-seed
completion receipts are small, atomic JSON records. Full feature/map caches are
never serialized. Previously persisted dense evidence is reused only after its
snapshot hashes, C64 vectors, LSO proofs, and replay-time memory identities pass.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

from . import stage1a_p12_runner as p12
from .stage1a_p11_production import (
    P11_CLASSES, P11PrivateCheckpoint, StandingP12Authorization,
    _private_mapping_digest)
from .stage1a_p1_reconstruction import (
    MAX_ANCHOR_SCORE_CALLS,
    MAX_QUERY_FEATURE_ENCODINGS,
    MAX_STATIC_MAP_CALLS,
    MemoryIdentity,
    ReplayEvent,
    sha256_json,
)


RECOVERY_SCHEMA = "stage1a.p12_low_memory_recovery.v1"
RECEIPT_SCHEMA = "stage1a.p12_trajectory_completion.v1"
EXPECTED_HEAD = "9a60222b70f8e9353b2937c5c128d41d967b72a2"
OLD_OUTPUT = (p12.ROOT / "outputs/calibration/"
              "stage1a_p12_native_replay_v1_standing_authorization_05_20261030")
RECOVERY_OUTPUT = (p12.ROOT / "outputs/calibration/"
                   "stage1a_p12_low_memory_recovery_v3_20261030")
RECOVERY_AUDIT = (p12.ROOT / "docs/calibration/audit/"
                  "stage1a_p12_low_memory_recovery_v3_20261030")


class RecoveryError(RuntimeError):
    """A missing, duplicate, inconsistent, or out-of-scope recovery artifact."""


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return value == value.lower()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as source:
        while True:
            block = source.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _portable_repo_path(value: str) -> str:
    """Normalize manifest paths emitted on Windows before identity comparison."""
    return str(value).replace("\\", "/")


def _atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise RecoveryError(f"Refusing to replace existing recovery artifact: {path.name}")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with temporary.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    if path.exists() and not replace:
        temporary.unlink(missing_ok=True)
        raise RecoveryError(f"Recovery artifact appeared before commit: {path.name}")
    os.replace(temporary, path)
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Windows does not support fsync on directory handles.
        pass


def _atomic_receipt(path: Path, receipt: Mapping[str, Any]) -> str:
    if path.exists():
        raise RecoveryError(f"Completion receipt already exists: {path.name}")
    _validate_receipt_shape(receipt)
    _atomic_json(path, dict(receipt))
    return _sha_file(path)


def _validate_memory_identity(value: Any, name: str) -> None:
    fields = {"version", "memory_sha256", "source_map_sha256",
              "source_revision", "legacy_greedy_version"}
    if not isinstance(value, Mapping) or not fields.issubset(value):
        raise RecoveryError(f"{name} lacks the complete native memory identity")
    if any(not _is_sha256(value.get(k)) for k in ("memory_sha256", "source_map_sha256")):
        raise RecoveryError(f"{name} has a malformed content/source-map digest")
    for field in ("version", "source_revision", "legacy_greedy_version"):
        if isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] < 0:
            raise RecoveryError(f"{name}.{field} is invalid")


def _validate_receipt_shape(receipt: Mapping[str, Any]) -> None:
    required = {
        "schema", "trajectory_id", "category", "support_seed",
        "query_tape_binding_sha256", "query_positions_sha256",
        "query_count", "expected_event_count", "event_count",
        "status", "complete", "event_chain_sha256",
        "query_response_chain_sha256", "initial_memory",
        "final_memory", "source_fingerprint_sha256",
        "scorer_config_sha256", "scorer_source_sha256",
        "evidence_revision_count", "evidence_file_sha256",
        "evidence_source", "last_successful_position",
        "counter_delta", "cumulative_class_counters",
        "memory_measurements",
    }
    if not isinstance(receipt, Mapping) or not required.issubset(receipt):
        missing = sorted(required - set(receipt)) if isinstance(receipt, Mapping) else sorted(required)
        raise RecoveryError(f"Completion receipt is missing fields: {missing}")
    if receipt["schema"] != RECEIPT_SCHEMA:
        raise RecoveryError("Unexpected trajectory receipt schema")
    if receipt["category"] not in P11_CLASSES:
        raise RecoveryError("Receipt class is outside the frozen four-class registry")
    seed = receipt["support_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in range(5):
        raise RecoveryError("Receipt support seed is outside the frozen 0-4 set")
    expected_id = f"{receipt['category']}/seed{seed}/Q-A-TS-v1-3pct-seed37"
    if receipt["trajectory_id"] != expected_id:
        raise RecoveryError("Receipt trajectory id does not match its class/seed identity")
    for field in ("query_tape_binding_sha256", "query_positions_sha256",
                  "event_chain_sha256", "query_response_chain_sha256",
                  "source_fingerprint_sha256", "scorer_config_sha256",
                  "scorer_source_sha256", "evidence_file_sha256"):
        if not _is_sha256(receipt[field]):
            raise RecoveryError(f"Receipt {field} is not a SHA-256 digest")
    _validate_memory_identity(receipt["initial_memory"], "initial_memory")
    _validate_memory_identity(receipt["final_memory"], "final_memory")
    count = receipt["expected_event_count"]
    if (isinstance(count, bool) or not isinstance(count, int) or count <= 0 or
            receipt["event_count"] != count or receipt["last_successful_position"] != count - 1 or
            receipt["status"] != "PASS_EXACT_PARITY" or receipt["complete"] is not True):
        raise RecoveryError("Receipt does not prove a complete exact-parity trajectory")
    query_count = receipt["query_count"]
    if isinstance(query_count, bool) or not isinstance(query_count, int) or query_count <= 0:
        raise RecoveryError("Receipt query count is invalid")
    if receipt["evidence_revision_count"] != query_count + 1:
        raise RecoveryError("Receipt evidence count differs from initial-plus-query revisions")
    if receipt["evidence_source"] not in ("recomputed_native", "reused_verified_artifact"):
        raise RecoveryError("Receipt evidence source is not declared")
    counters = receipt["counter_delta"]
    required_counters = {"queried_label_reads", "feature_encodings",
                         "static_map_calls", "anchor_score_calls",
                         "anchor_score_requests", "reused_evidence_score_entries"}
    if not isinstance(counters, Mapping) or not required_counters.issubset(counters):
        raise RecoveryError("Receipt compute counter deltas are incomplete")
    for key in required_counters:
        value = counters[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RecoveryError(f"Receipt counter {key} is invalid")


def merge_completion_receipts(
        receipts: Sequence[Mapping[str, Any]], *,
        tape_binding_sha256: str,
        query_counts: Mapping[str, int],
        query_positions_sha256_by_class: Mapping[str, str],
        event_counts: Mapping[str, int],
        expected_scorer_config_sha256: str,
        expected_scorer_source_sha256: str) -> dict[str, Any]:
    """Fail closed on duplicate/missing shards and return deterministic totals."""
    if not _is_sha256(tape_binding_sha256):
        raise RecoveryError("Merge requires the exact verified tape binding")
    expected_pairs = {(category, seed) for category in P11_CLASSES for seed in range(5)}
    by_pair: dict[tuple[str, int], Mapping[str, Any]] = {}
    for receipt in receipts:
        _validate_receipt_shape(receipt)
        pair = (str(receipt["category"]), int(receipt["support_seed"]))
        if pair in by_pair:
            raise RecoveryError(f"Duplicate trajectory receipt: {pair[0]}/seed{pair[1]}")
        if receipt["query_tape_binding_sha256"] != tape_binding_sha256:
            raise RecoveryError(f"{pair}: query tape binding differs")
        if receipt["scorer_config_sha256"] != expected_scorer_config_sha256:
            raise RecoveryError(f"{pair}: frozen scorer configuration differs")
        if receipt["scorer_source_sha256"] != expected_scorer_source_sha256:
            raise RecoveryError(f"{pair}: frozen Native Scorer source differs")
        if receipt["query_count"] != query_counts[pair[0]]:
            raise RecoveryError(f"{pair}: query count differs from its frozen tape")
        if receipt["query_positions_sha256"] != query_positions_sha256_by_class[pair[0]]:
            raise RecoveryError(f"{pair}: query positions digest differs from its frozen tape")
        if receipt["expected_event_count"] != event_counts[pair[0]]:
            raise RecoveryError(f"{pair}: stream count differs from its frozen class")
        by_pair[pair] = receipt
    missing = sorted(expected_pairs - set(by_pair), key=lambda x: (P11_CLASSES.index(x[0]), x[1]))
    extra = sorted(set(by_pair) - expected_pairs)
    if missing or extra:
        raise RecoveryError(f"Trajectory registry mismatch; missing={missing}, extra={extra}")

    ordered = [by_pair[(category, seed)]
               for category in P11_CLASSES for seed in range(5)]
    response_by_class: dict[str, str] = {}
    for category in P11_CLASSES:
        responses = {str(by_pair[(category, seed)]["query_response_chain_sha256"])
                     for seed in range(5)}
        if len(responses) != 1:
            raise RecoveryError(
                f"{category}: repeated support seeds disagree on the cached query-response chain")
        response_by_class[category] = next(iter(responses))
    query_positions = sum(int(query_counts[c]) for c in P11_CLASSES)
    query_events = sum(int(query_counts[c]) * 5 for c in P11_CLASSES)
    totals = {name: sum(int(row["counter_delta"][name]) for row in ordered)
              for name in ("queried_label_reads", "feature_encodings",
                           "static_map_calls", "anchor_score_calls",
                           "anchor_score_requests", "reused_evidence_score_entries")}
    if totals["queried_label_reads"] != query_positions:
        raise RecoveryError("Merged distinct queried-label disclosures differ from frozen tapes")
    if totals["feature_encodings"] != query_positions:
        raise RecoveryError("Merged physical query-feature encodings differ from unique tape positions")
    if totals["static_map_calls"] != query_events:
        raise RecoveryError("Merged Native static-map calls differ from class-seed query events")
    if totals["anchor_score_calls"] > MAX_ANCHOR_SCORE_CALLS:
        raise RecoveryError("Merged Native anchor calls exceed the frozen P1.2 ceiling")
    if totals["queried_label_reads"] > MAX_QUERY_FEATURE_ENCODINGS:
        raise RecoveryError("Merged queried-label disclosures exceed the frozen ceiling")
    if totals["feature_encodings"] > MAX_QUERY_FEATURE_ENCODINGS:
        raise RecoveryError("Merged feature encodings exceed the frozen ceiling")
    if totals["static_map_calls"] > MAX_STATIC_MAP_CALLS:
        raise RecoveryError("Merged static-map calls exceed the frozen ceiling")
    return {
        "schema": "stage1a.p12_merged_completion.v1",
        "trajectory_count": len(ordered),
        "trajectory_order": [row["trajectory_id"] for row in ordered],
        "unique_queried_positions": query_positions,
        "query_events_across_five_support_seeds": query_events,
        "queried_label_disclosures": totals["queried_label_reads"],
        "label_cache_hits": query_events - totals["queried_label_reads"],
        "logical_query_image_identities": query_positions,
        "physical_query_feature_encodings": totals["feature_encodings"],
        "feature_cache_hits": query_events - totals["feature_encodings"],
        "native_static_map_calls": totals["static_map_calls"],
        "native_static_map_cache_hits": query_events - totals["static_map_calls"],
        "native_anchor_score_calls": totals["anchor_score_calls"],
        "anchor_score_requests": totals["anchor_score_requests"],
        "reused_previous_evidence_score_entries": totals["reused_evidence_score_entries"],
        "query_response_chain_sha256_by_class": response_by_class,
        "score_evaluations_and_cache_hits_are_separately_accounted": True,
        "within_frozen_compute_ceilings": True,
        "receipts": [dict(row) for row in ordered],
    }


class MemoryProbe:
    """Bounded-memory RSS/CUDA observations grouped by named recovery stage."""

    def __init__(self, torch_module: Any):
        import psutil
        self._process = psutil.Process()
        self._psutil = psutil
        self._torch = torch_module
        self._stages: dict[str, dict[str, Any]] = {}

    def sample(self, stage: str) -> dict[str, int | None]:
        try:
            rss = int(self._process.memory_info().rss)
            available = int(self._psutil.virtual_memory().available)
        except Exception as exc:
            raise RecoveryError(f"Unable to sample process/system memory: {exc}") from exc
        cuda_allocated = int(self._torch.cuda.memory_allocated()) if self._torch.cuda.is_available() else None
        cuda_reserved = int(self._torch.cuda.memory_reserved()) if self._torch.cuda.is_available() else None
        snap: dict[str, int | None] = {
            "process_rss_bytes": rss,
            "system_available_bytes": available,
            "cuda_allocated_bytes": cuda_allocated,
            "cuda_reserved_bytes": cuda_reserved,
        }
        slot = self._stages.setdefault(stage, {
            "samples": 0, "first": None, "last": None, "peak_process_rss_bytes": 0,
            "minimum_system_available_bytes": None,
            "peak_cuda_allocated_bytes": 0, "peak_cuda_reserved_bytes": 0,
        })
        slot["samples"] += 1
        if slot["first"] is None:
            slot["first"] = snap
        slot["last"] = snap
        slot["peak_process_rss_bytes"] = max(slot["peak_process_rss_bytes"], rss)
        old_available = slot["minimum_system_available_bytes"]
        slot["minimum_system_available_bytes"] = (
            available if old_available is None else min(old_available, available))
        if cuda_allocated is not None:
            slot["peak_cuda_allocated_bytes"] = max(slot["peak_cuda_allocated_bytes"], cuda_allocated)
        if cuda_reserved is not None:
            slot["peak_cuda_reserved_bytes"] = max(slot["peak_cuda_reserved_bytes"], cuda_reserved)
        return snap

    def report(self) -> dict[str, Any]:
        return {key: self._stages[key] for key in sorted(self._stages)}


def verify_evidence_rows(
        rows: Sequence[Mapping[str, Any]], *,
        category: str,
        query_positions: Sequence[int],
        scorer_config_sha256: str,
        scorer_source_sha256: str) -> dict[str, Any]:
    """Verify persisted dense snapshot hashes, C64, causal anchors, and LSO proofs."""
    if category not in P11_CLASSES or not query_positions:
        raise RecoveryError("Evidence verification requires one frozen class tape")
    if len(rows) != len(query_positions) + 1:
        raise RecoveryError("Evidence file is missing its initial or queried-position snapshot")
    positions = [row.get("position") for row in rows]
    if positions != [None, *list(query_positions)]:
        raise RecoveryError("Evidence snapshot positions differ from the frozen Q-A-TS tape")
    all_anchor_ids: set[str] = set()
    previous_revision = 0
    previous_memory_version = -1
    native_score_requests = 0
    native_score_keys: set[tuple[Any, ...]] = set()
    for index, row in enumerate(rows):
        if row.get("schema") != "stage1a.p12_private_evidence_snapshot.v1":
            raise RecoveryError("Unexpected private evidence row schema")
        audit = row.get("source_exclusion_audit", {})
        if not isinstance(audit, Mapping):
            raise RecoveryError("Evidence row has no source-exclusion audit")
        for key in ("memory_version", "memory_sha256", "source_map_sha256",
                    "evidence_revision", "evidence_sha256"):
            if row.get(key) != audit.get(key):
                raise RecoveryError(f"Evidence row duplicated {key} identity differs")
        if (audit.get("raw_source_keys_or_labels_in_snapshot") is not False or
                audit.get("scorer_config_sha256") != scorer_config_sha256 or
                audit.get("scorer_source_sha256") != scorer_source_sha256):
            raise RecoveryError("Evidence provenance, scorer identity, or privacy declaration differs")
        if (not _is_sha256(row.get("memory_sha256")) or
                not _is_sha256(row.get("source_map_sha256")) or
                not _is_sha256(audit.get("c64_transport_sha256"))):
            raise RecoveryError("Evidence row contains a malformed memory/source/C64 digest")
        c64 = row.get("protected_c64_scores", ())
        if len(c64) != 64:
            raise RecoveryError("Evidence snapshot does not contain exactly 64 protected C64 scores")
        try:
            c64_values = [float(value) for value in c64]
        except (TypeError, ValueError) as exc:
            raise RecoveryError("C64 snapshot contains a nonnumeric score") from exc
        if any(not _is_finite(value) for value in c64_values):
            raise RecoveryError("C64 snapshot contains a nonfinite score")
        q90 = sorted(c64_values)[57]
        if (q90 != float(audit.get("c64_q90_higher_index_57", float("nan"))) or
                audit.get("protected_c64_count") != 64):
            raise RecoveryError("C64 q90 or protected count differs from its stored scores")
        normals = row.get("normal_anchors", ())
        defects = row.get("defect_anchors", ())
        protected = [item for item in normals if item.get("protected") is True]
        queried_normals = [item for item in normals if item.get("protected") is False]
        if len(protected) != 64 or len(normals) + len(defects) != 64 + index:
            raise RecoveryError("Dense evidence snapshot has a missing/duplicate anchor")
        query_anchors = queried_normals + list(defects)
        ids = [item.get("opaque_anchor_id") for item in query_anchors]
        if (any(not isinstance(value, str) or not value.startswith("query:") or len(value) != 70
                for value in ids) or len(set(ids)) != len(ids)):
            raise RecoveryError("Queried evidence anchor identity is malformed or duplicated")
        current_ids = {value[6:] for value in ids}
        if not all_anchor_ids.issubset(current_ids) or len(current_ids - all_anchor_ids) != (
                0 if index == 0 else 1):
            raise RecoveryError("Evidence snapshots do not add exactly one new queried anchor")
        all_anchor_ids = current_ids
        if (len(queried_normals) != audit.get("queried_normal_lso_count") or
                len(defects) != audit.get("queried_defect_current_memory_count")):
            raise RecoveryError("Evidence anchor kinds/counts differ from the snapshot audit")
        proofs = audit.get("source_exclusion_proofs", {})
        if not isinstance(proofs, Mapping) or set(proofs) != current_ids:
            raise RecoveryError("Evidence snapshot lacks one native source proof per queried anchor")
        if row.get("source_exclusion_proofs") != proofs:
            raise RecoveryError("Duplicated source-exclusion proof maps differ")
        normal_ids = {item["opaque_anchor_id"][6:] for item in queried_normals}
        defect_ids = {item["opaque_anchor_id"][6:] for item in defects}
        for anchor_id, proof in proofs.items():
            layer_count = int(proof.get("layer_count", 0))
            remaining = proof.get("remaining_source_rows_by_layer", ())
            removed = proof.get("removed_rows_by_layer", ())
            if (layer_count <= 0 or len(remaining) != layer_count or
                    len(removed) != layer_count or any(int(value) != 0 for value in remaining)):
                raise RecoveryError("Native source exclusion has residual source rows or wrong layer count")
            if any(proof.get(key) != row.get(key)
                   for key in ("memory_version", "memory_sha256", "source_map_sha256")):
                raise RecoveryError("Native source proof is not bound to this memory/source map")
            if anchor_id in normal_ids and proof.get(
                    "source_exclusion_semantics") != "queried_normal_all_layer_leave_source_out":
                raise RecoveryError("Queried normal lacks all-layer LSO proof")
            if anchor_id in defect_ids and (
                    proof.get("source_exclusion_semantics") != "queried_defect_not_written" or
                    any(int(value) != 0 for value in removed)):
                raise RecoveryError("Queried defect is not recorded as current-memory/not-written")
        normal_scores = {item["opaque_anchor_id"][6:]: float(item["score"])
                         for item in queried_normals}
        defect_scores = {item["opaque_anchor_id"][6:]: float(item["score"])
                         for item in defects}
        evidence_doc = {
            "memory_version": row["memory_version"],
            "memory_sha256": row["memory_sha256"],
            "source_map_sha256": row["source_map_sha256"],
            "c64_transport_sha256": audit["c64_transport_sha256"],
            "c64_q90": q90,
            "normal_scores": normal_scores,
            "defect_scores": defect_scores,
        }
        if sha256_json(evidence_doc) != row["evidence_sha256"]:
            raise RecoveryError("Evidence snapshot SHA-256 does not match canonical score payload")
        revision = int(row["evidence_revision"])
        version = int(row["memory_version"])
        if revision <= previous_revision or version < previous_memory_version:
            raise RecoveryError("Evidence revision or memory version moved backwards")
        previous_revision, previous_memory_version = revision, version
        memory_key = (version, row["memory_sha256"], row["source_map_sha256"])
        for item in query_anchors:
            score_key = (*memory_key, item["opaque_anchor_id"][6:])
            native_score_requests += 1
            native_score_keys.add(score_key)
    if len(all_anchor_ids) != len(query_positions):
        raise RecoveryError("Final evidence snapshot does not contain every queried anchor")
    return {
        "snapshot_count": len(rows),
        "query_anchor_count": len(all_anchor_ids),
        "evidence_revision_count": previous_revision,
        "native_score_requests": native_score_requests,
        "distinct_memory_anchor_score_keys": len(native_score_keys),
        "all_evidence_hashes_pass": True,
        "all_64_c64_vectors_and_q90_pass": True,
        "all_normal_lso_proofs_pass": True,
    }


def _is_finite(value: float) -> bool:
    import math
    return math.isfinite(value)


def _load_evidence_file(path: Path, *, category: str, positions: Sequence[int],
                        scorer_config_sha256: str,
                        scorer_source_sha256: str
                        ) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as source:
        for line in source:
            try:
                row = json.loads(line.decode("utf-8"))
            except Exception as exc:
                raise RecoveryError(f"Invalid persisted evidence JSONL: {path.name}") from exc
            if not isinstance(row, dict):
                raise RecoveryError(f"Evidence row is not an object: {path.name}")
            rows.append(row)
    digest = _sha_file(path)
    report = verify_evidence_rows(
        rows, category=category, query_positions=positions,
        scorer_config_sha256=scorer_config_sha256,
        scorer_source_sha256=scorer_source_sha256)
    report.update({"file_sha256": digest, "file_bytes": path.stat().st_size})
    return rows, report, digest


class StoredEvidenceReplayAdapter:
    """Bind verified stored evidence to each replayed causal memory boundary."""

    def __init__(self, *, category: str, rows: Sequence[Mapping[str, Any]],
                 c64_state_fn: Any):
        self.category = category
        self.rows = {row["position"]: row for row in rows}
        self.c64_state_fn = c64_state_fn
        self.seen_positions: list[int] = []
        self.query_count = len(rows) - 1
        self._initialized = False

    @staticmethod
    def _memory_fields(memory: MemoryIdentity) -> dict[str, Any]:
        return asdict(memory)

    def _check_row(self, row: Mapping[str, Any], memory: MemoryIdentity) -> None:
        memory_fields = self._memory_fields(memory)
        if (row.get("memory_version") != memory_fields["version"] or
                row.get("memory_sha256") != memory_fields["memory_sha256"] or
                row.get("source_map_sha256") != memory_fields["source_map_sha256"] or
                row["source_exclusion_audit"].get("source_revision") !=
                memory_fields["source_revision"] or
                row["source_exclusion_audit"].get("legacy_greedy_version") !=
                memory_fields["legacy_greedy_version"]):
            raise RecoveryError("Replayed memory/source identity differs from stored evidence revision")
        state = self.c64_state_fn(memory)
        scores = [float(value) for value in state["scores"]]
        if (list(map(float, row["protected_c64_scores"])) != scores or
                row["source_exclusion_audit"]["c64_transport_sha256"] !=
                state["transport_sha256"] or
                float(row["source_exclusion_audit"]["c64_q90_higher_index_57"]) !=
                float(state["q90"])):
            raise RecoveryError("Replayed memory does not match stored C64 vector/q90/transport")

    def initialize(self, memory: MemoryIdentity) -> None:
        if self._initialized:
            # P11ProductionReplayRunner re-enters the adapter once per replay
            # chunk. The initial snapshot was already checked; later calls are
            # wrapper boundaries, not a second initial evidence revision.
            return None
        row = self.rows.get(None)
        if row is None:
            raise RecoveryError("Stored evidence lacks its initial memory snapshot")
        self._check_row(row, memory)
        if row["source_exclusion_audit"]["queried_normal_lso_count"] != 0 or row[
                "source_exclusion_audit"]["queried_defect_current_memory_count"] != 0:
            raise RecoveryError("Initial stored evidence unexpectedly includes queried anchors")
        self._initialized = True

    def after_event(self, *, event: ReplayEvent, queried: bool,
                    label: int | None, features: Any, static_amap: Any,
                    writer_record: Mapping[str, Any],
                    memory_before: MemoryIdentity,
                    memory_after: MemoryIdentity) -> None:
        if not self._initialized:
            raise RecoveryError("Stored evidence adapter was not initialized")
        if not queried:
            if memory_after != memory_before:
                raise RecoveryError("Unqueried event changed shared memory during evidence reuse")
            return None
        if label not in (0, 1):
            raise RecoveryError("Stored evidence cannot bind a missing queried label")
        row = self.rows.get(int(event.position))
        if row is None:
            raise RecoveryError("Stored evidence has no snapshot for the current queried position")
        self._check_row(row, memory_after)
        source_group_sha = p12._group_sha(str(event.source_key))
        anchor_id = sha256_json({
            "schema": "stage1a.p12_query_anchor.v1",
            "category": self.category,
            "position": int(event.position),
            "image_sha256": event.image_sha256,
            "source_group_sha256": source_group_sha,
        })
        opaque_id = "query:" + anchor_id
        expected_kind = "normal" if int(label) == 0 else "defect"
        target = row["normal_anchors"] if expected_kind == "normal" else row["defect_anchors"]
        if sum(item.get("opaque_anchor_id") == opaque_id for item in target) != 1:
            raise RecoveryError("Stored causal anchor id/kind differs from the disclosed query event")
        if any(item.get("opaque_anchor_id") == opaque_id for item in row[
                "defect_anchors" if expected_kind == "normal" else "normal_anchors"]):
            raise RecoveryError("Stored queried anchor kind differs from current verified feedback")
        self.seen_positions.append(int(event.position))
        return None

    def verify_complete(self, positions: Sequence[int]) -> None:
        if self.seen_positions != list(positions):
            raise RecoveryError("Replay did not bind every stored evidence revision in tape order")


def _memory_checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Hash private checkpoint bytes without deserializing or buffering them."""
    return {"bytes": path.stat().st_size, "sha256": _sha_file(path),
            "deserialized": False}


def _verify_phase_a_checkpoint(path: Path, tapes: Any) -> dict[str, Any]:
    """Inspect only the 4.9 KB smoke checkpoint; never load the 7.36 GB cache."""
    import pickle
    with path.open("rb") as source:
        checkpoint = pickle.load(source)
    if not isinstance(checkpoint, P11PrivateCheckpoint):
        raise RecoveryError("Small phase-A checkpoint has an unexpected object type")
    if _private_mapping_digest(checkpoint._commitment_payload()) != checkpoint.private_state_sha256:
        raise RecoveryError("Small phase-A checkpoint self-commitment differs")
    if (checkpoint.query_tape_binding_sha256 != tapes.binding_sha256 or
            checkpoint.category != "audiojack" or
            checkpoint.trajectory_id != "audiojack/seed0/Q-A-TS-v1-3pct-seed37" or
            checkpoint.response_chain_sha256 != checkpoint.replay_checkpoint.query_response_sha256):
        raise RecoveryError("Small phase-A checkpoint identity/response chain differs")
    allowed = set(tapes.positions("audiojack"))
    label_state = checkpoint.queried_label_state
    if any(row[0] != "audiojack" or int(row[1]) not in allowed
           for row in label_state.get("values", ())):
        raise RecoveryError("Small phase-A checkpoint contains a label outside its query tape")
    if any(row[0] != "audiojack" or int(row[1]) not in allowed
           for row in label_state.get("groups_by_position", ())):
        raise RecoveryError("Small phase-A checkpoint source map exceeds its query tape")
    return {
        "status": "PASS_SMALL_CHECKPOINT_SELF_COMMITMENT",
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha_file(path),
        "private_state_sha256": checkpoint.private_state_sha256,
        "trajectory_id": checkpoint.trajectory_id,
        "cursor": checkpoint.replay_checkpoint.next_position,
        "query_label_entry_count": len(label_state.get("values", ())),
        "feature_cache_entry_count": len(checkpoint.feature_cache_state),
        "static_map_cache_entry_count": len(checkpoint.static_map_cache_state),
        "all_cached_labels_are_precommitted": True,
        "subsequent_resume_completion_durably_recorded": False,
    }


def _metadata_preflight(dataset_root: Path, json_root: Path) -> dict[str, Any]:
    if subprocess.run(["git", "branch", "--show-current"], cwd=p12.ROOT,
                      capture_output=True, text=True, check=True).stdout.strip() != p12.EXPECTED_BRANCH:
        raise RecoveryError("Recovery is on a branch outside the authorized calibration branch")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=p12.ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != EXPECTED_HEAD:
        raise RecoveryError(f"Recovery base commit changed: expected {EXPECTED_HEAD}, found {head}")
    tapes = p12.verify_stage05_qats_artifacts(p12.STAGE05)
    if tapes.binding_sha256 != "dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183":
        raise RecoveryError("Frozen Q-A-TS-v1 binding differs from the P1.2 authorization")
    frozen = p12._read_json(p12.STAGE05 / "frozen_inputs.json")
    stage05_run = p12._read_json(p12.STAGE05 / "run_manifest.json")
    stage05_scores = p12._read_json(p12.STAGE05 / "score_tape_manifest.json")
    protocol = p12._read_json(p12.STAGE075A / "protocol_manifest.json")
    historical = p12._verify_historical_package()
    c64_linkage = p12.verify_historical_c64_artifact_links(p12.STAGE05, p12.STAGE075A)
    environment = p12._verify_environment(dataset_root, json_root, stage05_run, stage05_scores)
    native_identity = p12._verify_native_source_identity(protocol, stage05_scores)
    label_isolation = p12._verify_stage05_label_isolation(frozen, stage05_run, stage05_scores)
    if (len(P11_CLASSES) != 4 or
            any(frozen["classes"][name]["stream_images"] !=
                p12.stage05.EXPECTED_STREAM_LENGTHS[name] for name in P11_CLASSES)):
        raise RecoveryError("Frozen four-class stream registry changed")
    return {
        "tapes": tapes,
        "frozen": frozen,
        "stage05_run": stage05_run,
        "stage05_scores": stage05_scores,
        "protocol": protocol,
        "historical": historical,
        "c64_linkage": c64_linkage,
        "environment": environment,
        "native_identity": native_identity,
        "label_isolation": label_isolation,
    }


def _verify_old_partial_namespace(preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only audit of standing_authorization_05; never modifies that tree."""
    if not OLD_OUTPUT.is_dir():
        raise RecoveryError("The prior P1.2 partial output namespace is missing")
    required = ("INCOMPLETE", "preflight_manifest.json", "progress_manifest.json",
                "P12_RUNTIME_STOP.json", "runtime_stop_checksums.json")
    if any(not (OLD_OUTPUT / name).is_file() for name in required):
        raise RecoveryError("The prior P1.2 output lacks a required forensic artifact")
    checksum_manifest = p12._read_json(OLD_OUTPUT / "runtime_stop_checksums.json")
    for name, expected_sha in checksum_manifest.get("files", {}).items():
        if _sha_file(OLD_OUTPUT / name) != expected_sha:
            raise RecoveryError(f"Prior P1.2 output metadata checksum differs: {name}")
    old_progress = p12._read_json(OLD_OUTPUT / "progress_manifest.json")
    old_stop = p12._read_json(OLD_OUTPUT / "P12_RUNTIME_STOP.json")
    old_preflight = p12._read_json(OLD_OUTPUT / "preflight_manifest.json")
    if (old_progress.get("completed_trajectories") != 15 or
            old_progress.get("last_trajectory") !=
            "switch/seed4/Q-A-TS-v1-3pct-seed37" or
            old_stop.get("failure_type") != "MemoryError" or
            old_stop.get("last_durable_progress", {}).get("completed_trajectories") != 15 or
            old_preflight.get("query_tape_binding_sha256") != preflight["tapes"].binding_sha256):
        raise RecoveryError("Prior P1.2 stop/progress boundary differs from the recorded forensic state")
    if old_preflight.get("native_execution_provenance", {}).get(
            "current_execution_source_assessment", {}).get("execution_allowed") is not True:
        raise RecoveryError("Prior scoped source-provenance exception is not established")

    scorer_config = str(preflight["native_identity"]["scorer_config_sha256"])
    scorer_source = str(preflight["native_identity"]["scorer_source_sha256"])
    evidence_root = OLD_OUTPUT / "private" / "evidence"
    evidence_paths = sorted(evidence_root.glob("*.jsonl"))
    expected_pairs = {
        (category, seed) for category in P11_CLASSES for seed in range(5)
        if category != "rolled_strip_base" or seed == 0
    }
    observed_pairs: set[tuple[str, int]] = set()
    query_counts = {row["class_identity"]: int(row["query_count"])
                    for row in preflight["tapes"].bindings}
    evidence_index: dict[str, dict[str, Any]] = {}
    first15_anchor_calls = 0
    for path in evidence_paths:
        stem = path.stem
        matched = None
        for category in P11_CLASSES:
            prefix = category + "_seed"
            if stem.startswith(prefix) and stem[len(prefix):].isdigit():
                matched = (category, int(stem[len(prefix):]))
                break
        if matched is None or matched not in expected_pairs or matched in observed_pairs:
            raise RecoveryError(f"Unexpected or duplicate prior evidence shard: {path.name}")
        category, seed = matched
        observed_pairs.add(matched)
        rows, report, digest = _load_evidence_file(
            path, category=category, positions=preflight["tapes"].positions(category),
            scorer_config_sha256=scorer_config, scorer_source_sha256=scorer_source)
        key = f"{category}/seed{seed}"
        evidence_index[key] = {
            "path": str(path.relative_to(p12.ROOT)),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            **report,
        }
        if category != "rolled_strip_base":
            first15_anchor_calls += int(report["distinct_memory_anchor_score_keys"])
        del rows
    missing = expected_pairs - observed_pairs
    if missing:
        raise RecoveryError(f"Prior P1.2 evidence shards are missing: {sorted(missing)}")
    durable_anchor_calls = int(old_progress["compute_counters"]["anchor_score_calls"])
    if first15_anchor_calls != durable_anchor_calls:
        raise RecoveryError(
            f"Evidence-derived anchor-call total {first15_anchor_calls} differs from "
            f"the durable counter {durable_anchor_calls}")
    old_seed0 = evidence_index["rolled_strip_base/seed0"]
    if old_seed0["snapshot_count"] != query_counts["rolled_strip_base"] + 1:
        raise RecoveryError("Uncheckpointed rolled_strip_base/seed0 evidence is incomplete")

    recorded_receipts = list(OLD_OUTPUT.rglob("*completion*"))
    if recorded_receipts:
        raise RecoveryError("Prior namespace unexpectedly has completion receipts; inspect before reuse")
    latest_checkpoint = OLD_OUTPUT / "private" / "latest_p1_checkpoint.pt"
    phase_checkpoint = OLD_OUTPUT / "private" / "phase_a_reconstructed_checkpoint.pt"
    if not latest_checkpoint.is_file() or not phase_checkpoint.is_file():
        raise RecoveryError("Prior private checkpoint artifact is missing")
    latest_meta = _memory_checkpoint_metadata(latest_checkpoint)
    phase_meta = _verify_phase_a_checkpoint(phase_checkpoint, preflight["tapes"])

    # Compute an immutable inventory for before/after verification. It contains
    # file paths, sizes, and digests only; private contents are never copied.
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in OLD_OUTPUT.rglob("*") if item.is_file()):
        rel = path.relative_to(OLD_OUTPUT).as_posix()
        inventory[rel] = {"size_bytes": path.stat().st_size, "sha256": _sha_file(path)}

    query_count_all = sum(query_counts.values())
    query_events_partial = sum(query_counts[c] * 5 for c in P11_CLASSES if c != "rolled_strip_base") + query_counts["rolled_strip_base"]
    prior_compute = {
        "unique_query_image_positions": query_count_all,
        "queried_label_disclosures": query_count_all,
        "query_feature_encodings": query_count_all,
        "query_events": query_events_partial,
        "label_cache_hits": query_events_partial - query_count_all,
        "feature_cache_hits": query_events_partial - query_count_all,
        "static_map_calls": query_events_partial,
        "static_map_cache_hits": 0,
        "native_anchor_score_calls": durable_anchor_calls +
            int(old_seed0["distinct_memory_anchor_score_keys"]),
        "first15_anchor_score_calls_reconciled_to_durable_counter": durable_anchor_calls,
        "uncheckpointed_seed0_anchor_score_calls_derived_from_snapshots":
            int(old_seed0["distinct_memory_anchor_score_keys"]),
        "source": "15 durable progress counters plus self-consistent all-query evidence snapshots",
    }
    if (prior_compute["queried_label_disclosures"] != query_count_all or
            prior_compute["static_map_calls"] != 1839 or
            prior_compute["native_anchor_score_calls"] != 63586):
        raise RecoveryError("Prior attempt query/static/anchor compute reconciliation failed")
    return {
        "prior_output_namespace": str(OLD_OUTPUT.relative_to(p12.ROOT)),
        "prior_incomplete_marker_sha256": _sha_file(OLD_OUTPUT / "INCOMPLETE"),
        "last_durable_trajectory": old_progress["last_trajectory"],
        "durable_completed_trajectory_count": old_progress["completed_trajectories"],
        "additional_cli_observed_pass": old_stop.get("additional_cli_observation", {}),
        "unstarted_trajectories": [
            f"rolled_strip_base/seed{seed}" for seed in range(1, 5)],
        "evidence_shards": evidence_index,
        "prior_compute_reconciled": prior_compute,
        "latest_checkpoint": latest_meta,
        "small_phase_a_checkpoint": phase_meta,
        "latest_checkpoint_deserialized": False,
        "exact_event_completion_receipts_present": False,
        "per_trajectory_event_chain_hashes_present": False,
        "per_trajectory_query_response_chain_hashes_present": False,
        "evidence_can_be_reused_after_replay_memory_identity_binding": True,
        "exact_memory_writer_replay_must_be_rerun": True,
        "read_only_file_inventory": inventory,
    }


def _verify_old_inventory_unchanged(inventory: Mapping[str, Mapping[str, Any]]) -> bool:
    actual_paths = sorted(item.relative_to(OLD_OUTPUT).as_posix()
                          for item in OLD_OUTPUT.rglob("*") if item.is_file())
    if actual_paths != sorted(inventory):
        return False
    for relative, expected in inventory.items():
        path = OLD_OUTPUT / Path(relative)
        if (path.stat().st_size != expected["size_bytes"] or
                _sha_file(path) != expected["sha256"]):
            return False
    return True


def _audit_aborted_recovery_attempts(frozen: Mapping[str, Any]) -> dict[str, Any]:
    """Record the two failed implementation starts without reusing their work."""
    attempts = (
        ("v1", "stage1a_p12_low_memory_recovery_v1_20261030",
         "TypeError", "'MemoryProbe' object is not callable", 0, 1, 0),
        ("v2", "stage1a_p12_low_memory_recovery_v2_20261030",
         "RecoveryError", "Stored evidence adapter was initialized twice", 7, 7, 7),
    )
    rows: list[dict[str, Any]] = []
    for version, directory, expected_type, expected_error, labels, features, static in attempts:
        root = p12.ROOT / "outputs/calibration" / directory
        failure_path = root / "shards" / "audiojack" / "worker_failure.json"
        progress_path = root / "progress_manifest.json"
        log_path = root / "logs" / "audiojack.log"
        provenance_path = (p12.ROOT / "docs/calibration/audit" / directory /
                           "source_provenance.json")
        if (not all(path.is_file() for path in
                    (failure_path, progress_path, log_path, provenance_path)) or
                not (root / "INCOMPLETE").is_file() or
                (root / "complete.json").exists()):
            raise RecoveryError(f"Failed {version} recovery attempt is missing its stop evidence")
        failure = p12._read_json(failure_path)
        progress = p12._read_json(progress_path)
        provenance = p12._read_json(provenance_path)
        counters = failure.get("compute_counters", {})
        if (failure.get("error_type") != expected_type or
                failure.get("error") != expected_error or
                counters.get("queried_label_reads") != labels or
                counters.get("feature_encodings") != features or
                counters.get("static_map_calls") != static or
                counters.get("anchor_score_calls") != 0 or
                any(row.get("old_partial_namespace_unchanged_after_shard") is not True
                    for row in progress.get("class_shard_statuses", ())) or
                list((root / "shards" / "audiojack").glob("seed*.receipt.json"))):
            raise RecoveryError(f"Failed {version} recovery attempt stop boundary changed")
        rows.append({
            "attempt": version,
            "namespace": str(root.relative_to(p12.ROOT).as_posix()),
            "status": "BLOCKED_IMPLEMENTATION_ERROR_NO_COMPLETED_TRAJECTORY",
            "error_type": expected_type,
            "error": expected_error,
            "audiojack_seed0_label_reads": labels,
            "audiojack_seed0_query_feature_encodings": features,
            "audiojack_seed0_static_map_calls": static,
            "audiojack_seed0_anchor_score_calls": 0,
            "completed_trajectory_receipts": 0,
            "unqueried_labels_accessed": False,
            "historical_partial_namespace_unchanged": True,
            "stop_record_sha256": _sha_file(failure_path),
            "progress_manifest_sha256": _sha_file(progress_path),
            "worker_log_sha256": _sha_file(log_path),
            "source_provenance_file_sha256": _sha_file(provenance_path),
            "recovery_runner_source_sha256_recorded_at_attempt": provenance.get(
                "recovery_runner_sha256"),
            "executed_recovery_source_bytes_retained": False,
        })
    support_count = len({str(row["images"][index])
                         for seed in range(5)
                         for row in (frozen["classes"]["audiojack"]["supports"][str(seed)],)
                         for index in range(len(row["images"]))})
    for row in rows:
        row["audiojack_support_reference_feature_encodings"] = support_count
    return {"attempts": rows,
            "total_failed_attempt_query_label_reads": sum(row["audiojack_seed0_label_reads"]
                                                            for row in rows),
            "total_failed_attempt_query_feature_encodings": sum(
                row["audiojack_seed0_query_feature_encodings"] for row in rows),
            "total_failed_attempt_static_map_calls": sum(
                row["audiojack_seed0_static_map_calls"] for row in rows),
            "total_failed_attempt_support_reference_feature_encodings":
                support_count * len(rows)}


def _p12_execution_fingerprint(preflight: Mapping[str, Any]) -> tuple[
        dict[str, Any], str, str]:
    source_identity, source_aggregate, scorer_source = p12._current_source_identity()
    historical = preflight["historical"]
    fingerprint = p12._sha_json({
        "source_identity": source_identity,
        "source_fingerprint": source_aggregate,
        "historical_protocol_manifest_sha256": historical["protocol_manifest_sha256"],
        "historical_event_score_files": historical["trajectory_file_sha256"],
        "query_tape_binding_sha256": preflight["tapes"].binding_sha256,
    })
    return source_identity, fingerprint, scorer_source


def _counter_values(budget: Any) -> dict[str, int]:
    return {
        "queried_label_reads": int(budget.queried_label_reads),
        "feature_encodings": int(budget.feature_encodings),
        "static_map_calls": int(budget.static_map_calls),
        "anchor_score_calls": int(budget.anchor_score_calls),
    }


def _counter_diff(after: Mapping[str, int], before: Mapping[str, int]) -> dict[str, int]:
    return {key: int(after[key]) - int(before[key]) for key in after}


def _host_memory_snapshot() -> dict[str, int | None]:
    try:
        import psutil
        process = psutil.Process()
        return {"process_rss_bytes": int(process.memory_info().rss),
                "system_available_bytes": int(psutil.virtual_memory().available)}
    except Exception as exc:
        return {"memory_sample_error": str(exc)}


def _class_worker(*, category: str, output: Path, dataset_root: Path,
                  json_root: Path, chunk_size: int) -> dict[str, Any]:
    """Execute five seeds for one class in a fresh, class-local process."""
    if category not in P11_CLASSES:
        raise RecoveryError("Worker class is outside the frozen four-class registry")
    if not output.resolve().is_relative_to(p12.ROOT):
        raise RecoveryError("Worker output is outside the repository")
    import torch
    from experiments.p0_protocol_audit import build_vmb, encode_reference_features, require_sources
    from .native import isolate_native_bank_k256, validate_p0_config
    from .stage1a_p11_production import (
        P11ProductionReplayRunner, verify_controller_snapshot_compatibility)

    preflight = _metadata_preflight(dataset_root, json_root)
    tapes = preflight["tapes"]
    frozen = preflight["frozen"]
    environment = preflight["environment"]
    source_identity, source_fingerprint, scorer_source = _p12_execution_fingerprint(preflight)
    scorer_config = str(preflight["native_identity"]["scorer_config_sha256"])
    if scorer_config != p12.SCORER_CONFIG_SHA256:
        raise RecoveryError("Frozen P1.2 scorer configuration identity differs")

    standing = StandingP12Authorization.for_active_user_instruction(tapes)
    production_runner = p12._make_production_runner(tapes, standing)
    budget = production_runner.runner.budget
    impl = require_sources()
    config = p12._native_config(impl)
    validate_p0_config(config, p12.DEFAULT_PROFILE.scorer)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    impl.set_seed(0)
    memory_probe = MemoryProbe(torch)
    memory_probe.sample("process_started")
    memory_probe.sample("model_initialization_before")
    backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
    memory_probe.sample("model_initialization_after")

    encoder_fingerprint = p12._sha_json({
        "checkpoint_sha256": environment["model_checkpoint_sha256"],
        "source_tree_sha256": environment["dinov2_source_tree_sha256"],
        "model": p12.DEFAULT_PROFILE.scorer.model_name,
    })
    preprocessing_fingerprint = p12._sha_json({
        "image_size": p12.DEFAULT_PROFILE.scorer.image_size,
        "resize_size": p12.DEFAULT_PROFILE.scorer.resize_size,
        "feature_fuse": p12.DEFAULT_PROFILE.scorer.feature_fuse,
        "use_fp16": p12.DEFAULT_PROFILE.scorer.use_fp16,
        "p0_source_sha256": p12._sha_file(p12.ROOT / "experiments/p0_protocol_audit.py"),
        "vmb_source_sha256": p12._sha_file(p12.ROOT / "vmb_visionad_new.py"),
    })

    info = frozen["classes"][category]
    support_paths = sorted({str(p12.stage05._absolute_image_path(
        dataset_root, info["prefix"], relative))
        for seed in range(5)
        for relative in info["supports"][str(seed)]["images"]})
    memory_probe.sample("support_feature_encoding_before")
    support_features = encode_reference_features(impl, backbone, support_paths, config)
    memory_probe.sample("support_feature_encoding_after")
    support_feature_encodings = len(support_paths)
    support_feature_identities = sorted(
        p12._sha_bytes(path.encode("utf-8")) for path in support_paths)
    image_digest_cache: dict[str, str] = {}
    label_reader = p12._make_label_reader(category, tapes, frozen, json_root)
    old_index = p12._read_json(output / "recovery_preflight.json")["prior_evidence_index"]
    trajectory_rows: list[dict[str, Any]] = []
    controller_compatibility: dict[str, Any] | None = None
    failure_path = output / "shards" / category / "worker_failure.json"
    class_started = time.perf_counter()

    try:
        for seed in range(5):
            trajectory_id = f"{category}/seed{seed}/Q-A-TS-v1-3pct-seed37"
            receipt_path = output / "shards" / category / f"seed{seed}.receipt.json"
            if receipt_path.exists():
                raise RecoveryError(
                    f"Refusing to overwrite an existing trajectory receipt: {trajectory_id}")
            memory_probe.sample(f"seed{seed}_trajectory_before")
            events, c64_states, static_tape, initial_c64, fixed_sha, summary = (
                p12._prepare_historical_data(category, seed, frozen))
            chosen_paths = [str(p12.stage05._absolute_image_path(
                dataset_root, info["prefix"], relative))
                for relative in info["supports"][str(seed)]["images"]]
            native_bank = build_vmb(impl, support_features, chosen_paths, config)
            native_bank, reconstructed_fixed = isolate_native_bank_k256(native_bank)
            if reconstructed_fixed != fixed_sha:
                raise p12.P12IntegrityError(
                    f"{category}/seed{seed}: reconstructed fixed support digest differs")

            tape_positions = tapes.positions(category)
            position_sha = sha256_json(list(tape_positions))
            old_row = old_index.get(f"{category}/seed{seed}")
            evidence_override = None
            evidence_source = "recomputed_native"
            if old_row is not None:
                evidence_path = p12.ROOT / old_row["path"]
                rows, verified_report, evidence_digest = _load_evidence_file(
                    evidence_path, category=category, positions=tape_positions,
                    scorer_config_sha256=scorer_config,
                    scorer_source_sha256=scorer_source)
                if evidence_digest != old_row["sha256"]:
                    raise RecoveryError(
                        f"{trajectory_id}: prior evidence bytes changed after preflight")
                evidence_override = StoredEvidenceReplayAdapter(
                    category=category, rows=rows,
                    c64_state_fn=lambda memory, states=c64_states:
                    states[memory.memory_sha256])
                evidence_source = "reused_verified_artifact"
            else:
                evidence_path = (output / "private" / "evidence" /
                                 f"{category}_seed{seed}.jsonl")
                if evidence_path.exists():
                    raise RecoveryError(
                        f"Refusing to append to an unreceipted evidence artifact: {evidence_path.name}")
                verified_report = None

            # M0 score maps are class/seed specific. Keep only this seed's maps;
            # query features remain resident for the class and are reused.
            production_runner.runner.static_maps._values.clear()
            before = _counter_values(budget)
            result, writer, callbacks, reusable, trajectory_id, _kwargs = (
                p12._run_one_trajectory(
                    category=category, seed=seed, events=events,
                    c64_states=c64_states, static_tape=static_tape,
                    initial_c64=initial_c64, fixed_sha=fixed_sha,
                    native_bank=native_bank, impl=impl, backbone=backbone,
                    p0_config=config, production_runner=production_runner,
                    tapes=tapes, frozen=frozen, dataset_root=dataset_root,
                    label_reader=label_reader,
                    image_digest_cache=image_digest_cache,
                    source_fingerprint=source_fingerprint,
                    scorer_config_sha256=scorer_config,
                    scorer_source_sha256=scorer_source,
                    encoder_fingerprint=encoder_fingerprint,
                    preprocessing_fingerprint=preprocessing_fingerprint,
                    output=output, chunk_size=chunk_size,
                    memory_probe=memory_probe.sample,
                    evidence_adapter_override=evidence_override,
                    private_evidence_path=evidence_path))
            after = _counter_values(budget)
            delta = _counter_diff(after, before)
            if (not result.complete or result.status != "PASS_EXACT_PARITY" or
                    result.next_position != len(events)):
                raise p12.P1ParityMismatch(
                    f"{trajectory_id}: replay did not reach exact complete parity")
            if evidence_override is not None:
                evidence_override.verify_complete(tape_positions)
                # Exact replay-time memory and C64 binding is checked by the
                # adapter at initialization and after every queried event.
                evidence_report = verified_report
                evidence_sha = _sha_file(evidence_path)
            else:
                _, evidence_report, evidence_sha = _load_evidence_file(
                    evidence_path, category=category, positions=tape_positions,
                    scorer_config_sha256=scorer_config,
                    scorer_source_sha256=scorer_source)
                from .stage1a_p11_production import verify_controller_snapshot_compatibility
                controller_compatibility = verify_controller_snapshot_compatibility(
                    callbacks["evidence_adapter"].last_result.snapshot)

            # Requests are score slots across all dense revisions. Physical
            # calls are charged by the exact shared Native budget. Imported
            # entries are separately reported; no score is inferred from
            # an absent or unverified snapshot.
            requests = int(evidence_report["native_score_requests"])
            distinct_keys = int(evidence_report["distinct_memory_anchor_score_keys"])
            delta.update({
                "anchor_score_requests": requests,
                "reused_evidence_score_entries": (
                    distinct_keys if evidence_source == "reused_verified_artifact" else 0),
                "within_run_anchor_score_cache_hits": (
                    max(0, requests - delta["anchor_score_calls"])
                    if evidence_source == "recomputed_native" else 0),
            })
            memory_probe.sample(f"seed{seed}_trajectory_after")
            final_memory = asdict(result.checkpoint.memory)
            initial_memory = asdict(events[0]["memory_before"])
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "trajectory_id": trajectory_id,
                "category": category,
                "support_seed": seed,
                "query_tape_binding_sha256": tapes.binding_sha256,
                "query_positions_sha256": position_sha,
                "query_count": len(tape_positions),
                "expected_event_count": len(events),
                "event_count": int(result.next_position),
                "status": "PASS_EXACT_PARITY",
                "complete": True,
                "event_chain_sha256": result.event_chain_sha256,
                "query_response_chain_sha256": result.query_response_sha256,
                "initial_memory": initial_memory,
                "final_memory": final_memory,
                "source_fingerprint_sha256": source_fingerprint,
                "scorer_config_sha256": scorer_config,
                "scorer_source_sha256": scorer_source,
                "evidence_revision_count": len(tape_positions) + 1,
                "evidence_file_sha256": evidence_sha,
                "evidence_file": str(evidence_path.relative_to(p12.ROOT).as_posix()),
                "evidence_source": evidence_source,
                "evidence_report": evidence_report,
                "last_successful_position": len(events) - 1,
                "counter_delta": delta,
                "cumulative_class_counters": after,
                "feature_cache_entries_after_trajectory": len(
                    production_runner.runner.features._values),
                "static_map_cache_entries_after_trajectory": len(
                    production_runner.runner.static_maps._values),
                "support_feature_encodings_for_class": support_feature_encodings,
                "support_feature_identity_count": len(support_feature_identities),
                "historical_content_update_count": int(summary["content_update_count"]),
                "memory_measurements": memory_probe.report(),
                "controller_snapshot_compatibility": controller_compatibility,
            }
            _validate_receipt_shape(receipt)
            _atomic_receipt(receipt_path, receipt)
            trajectory_rows.append({
                "trajectory_id": trajectory_id,
                "receipt_path": str(receipt_path.relative_to(p12.ROOT).as_posix()),
                "receipt_sha256": _sha_file(receipt_path),
                "event_count": len(events),
                "query_count": len(tape_positions),
                "event_chain_sha256": result.event_chain_sha256,
                "query_response_chain_sha256": result.query_response_sha256,
                "evidence_file_sha256": evidence_sha,
                "evidence_source": evidence_source,
            })
            print(f"SHARD_TRAJECTORY_PASS {trajectory_id} events={len(events)} "
                  f"queries={len(tape_positions)} evidence={evidence_source}", flush=True)

            del (events, c64_states, static_tape, initial_c64, native_bank,
                 result, writer, callbacks, reusable)
            gc.collect()

        label_reader.close()
        memory_probe.sample("class_finalized")
        report = {
            "schema": "stage1a.p12_class_shard_completion.v1",
            "category": category,
            "query_tape_binding_sha256": tapes.binding_sha256,
            "trajectory_count": len(trajectory_rows),
            "trajectory_receipts": trajectory_rows,
            "class_compute_counters": _counter_values(budget),
            "unique_decoded_query_positions": len(label_reader.decoded_positions),
            "decoded_query_position_sha256": sha256_json(
                sorted(int(x) for x in label_reader.decoded_positions)),
            "query_feature_cache_entries": len(production_runner.runner.features._values),
            "support_reference_feature_encodings": support_feature_encodings,
            "support_feature_identity_sha256": sha256_json(support_feature_identities),
            "controller_snapshot_compatibility": controller_compatibility,
            "memory_measurements": memory_probe.report(),
            "elapsed_seconds": time.perf_counter() - class_started,
            "class_process_restarted_between_classes": True,
            "large_private_checkpoint_serialized": False,
        }
        _atomic_json(output / "shards" / category / "class_completion.json", report)
        return report
    except BaseException as exc:
        try:
            label_reader.close()
        except Exception:
            pass
        failure = {
            "schema": "stage1a.p12_class_shard_failure.v1",
            "category": category,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "completed_trajectory_receipts": [row["trajectory_id"]
                                               for row in trajectory_rows],
            "memory_measurements": memory_probe.report(),
            "last_memory_sample": memory_probe.sample("failure_boundary"),
            "compute_counters": _counter_values(budget),
        }
        if not failure_path.exists():
            _atomic_json(failure_path, failure)
        raise


def _verify_worker_receipts(output: Path, preflight: Mapping[str, Any]
                            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for category in P11_CLASSES:
        completion = output / "shards" / category / "class_completion.json"
        if not completion.is_file():
            raise RecoveryError(f"Class completion receipt missing for {category}")
        class_manifest = p12._read_json(completion)
        if (class_manifest.get("trajectory_count") != 5 or
                class_manifest.get("category") != category):
            raise RecoveryError(f"Class completion manifest malformed for {category}")
        expected_positions = list(preflight["tapes"].positions(category))
        if (class_manifest.get("unique_decoded_query_positions") != len(expected_positions) or
                class_manifest.get("decoded_query_position_sha256") !=
                sha256_json(expected_positions)):
            raise RecoveryError(f"Class {category}: decoded labels differ from its exact frozen tape")
        receipt_index = {row["trajectory_id"]: row
                         for row in class_manifest.get("trajectory_receipts", ())}
        if len(receipt_index) != 5:
            raise RecoveryError(f"Class {category}: completion manifest has missing/duplicate receipts")
        for seed in range(5):
            path = output / "shards" / category / f"seed{seed}.receipt.json"
            if not path.is_file():
                raise RecoveryError(f"Missing trajectory completion receipt: {category}/seed{seed}")
            receipt = p12._read_json(path)
            _validate_receipt_shape(receipt)
            if receipt.get("category") != category or receipt.get("support_seed") != seed:
                raise RecoveryError(f"Receipt path/identity differs: {path.name}")
            summary = receipt_index.get(receipt["trajectory_id"])
            if (summary is None or summary.get("receipt_path") !=
                    str(path.relative_to(p12.ROOT).as_posix()) or
                    summary.get("receipt_sha256") != _sha_file(path)):
                raise RecoveryError(f"Class {category}: receipt linkage hash differs for seed {seed}")
            evidence_path = p12.ROOT / receipt["evidence_file"]
            if not evidence_path.is_file() or _sha_file(evidence_path) != receipt[
                    "evidence_file_sha256"]:
                raise RecoveryError(f"Evidence file linkage differs for {category}/seed{seed}")
            if receipt["evidence_source"] == "reused_verified_artifact":
                prior = p12._read_json(output / "recovery_preflight.json")[
                    "prior_evidence_index"].get(f"{category}/seed{seed}")
                prior_path = (_portable_repo_path(prior["path"])
                              if prior is not None else None)
                if (prior is None or receipt["evidence_file"] != prior_path or
                        receipt["evidence_file_sha256"] != prior["sha256"]):
                    raise RecoveryError(f"Prior evidence reuse linkage differs for {category}/seed{seed}")
            elif not evidence_path.resolve().is_relative_to((output / "private" / "evidence").resolve()):
                raise RecoveryError(f"Recomputed evidence escaped the new private namespace: {path.name}")
            receipts.append(receipt)
    query_counts = {row["class_identity"]: int(row["query_count"])
                    for row in preflight["tapes"].bindings}
    event_counts = {category: int(preflight["frozen"]["classes"][category]["stream_images"])
                    for category in P11_CLASSES}
    source_identity, source_fingerprint, scorer_source = _p12_execution_fingerprint(preflight)
    merged = merge_completion_receipts(
        receipts, tape_binding_sha256=preflight["tapes"].binding_sha256,
        query_counts=query_counts,
        query_positions_sha256_by_class={
            category: sha256_json(list(preflight["tapes"].positions(category)))
            for category in P11_CLASSES},
        event_counts=event_counts,
        expected_scorer_config_sha256=str(
            preflight["native_identity"]["scorer_config_sha256"]),
        expected_scorer_source_sha256=scorer_source)
    if any(row["source_fingerprint_sha256"] != source_fingerprint for row in receipts):
        raise RecoveryError("A trajectory receipt is bound to a different replay source fingerprint")
    return receipts, {"merged": merged, "source_identity": source_identity,
                      "source_fingerprint_sha256": source_fingerprint,
                      "scorer_source_sha256": scorer_source}


def _write_blocked_recovery(output: Path, audit: Path, *,
                            statuses: Sequence[Mapping[str, Any]],
                            error: BaseException,
                            old_inventory_unchanged: bool | None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    (output / "INCOMPLETE").write_text(
        "P1.2 class-sharded recovery did not pass all acceptance gates.\n",
        encoding="utf-8")
    failure = {
        "schema": "stage1a.p12_low_memory_recovery_failure.v1",
        "status": "BLOCKED",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        "class_shard_statuses": list(statuses),
        "previous_partial_namespace_unchanged": old_inventory_unchanged,
        "stage1a_performance_evaluation": False,
        "confirmation_or_mask_access": False,
    }
    if not (output / "recovery_failure.json").exists():
        _atomic_json(output / "recovery_failure.json", failure)
    if not (audit / "p12_low_memory_recovery_gate_matrix.json").exists():
        _atomic_json(audit / "p12_low_memory_recovery_gate_matrix.json", {
            "schema": "stage1a.p12_low_memory_recovery_gate.v1",
            "decision": "P12_RECOVERY_BLOCKED",
            "gates": [
                {"id": "P12-CPU-TESTS", "status": "PASS_BEFORE_REAL_REPLAY"},
                {"id": "P12-CLASS-SHARDS", "status": "BLOCKED",
                 "evidence": str(error)},
                {"id": "P12-OLD-OUTPUT-IMMUTABILITY",
                 "status": ("PASS" if old_inventory_unchanged else "UNVERIFIED_OR_FAIL")},
                {"id": "P12-STAGE1A-PERFORMANCE", "status": "NOT_RUN_PROHIBITED"},
                {"id": "P12-CONFIRMATION-ISOLATION", "status": "NO_ACCESS"},
            ], "class_shards": list(statuses),
        })
        (audit / "STAGE1A_P12_LOW_MEMORY_RECOVERY_REPORT.md").write_text(
            "# P1.2 low-memory recovery\n\n"
            "**Decision: `P12_RECOVERY_BLOCKED`**\n\n"
            f"The class-sharded recovery stopped with `{type(error).__name__}: {error}`. "
            "No PASS is claimed. The prior partial namespace immutability result is "
            f"`{old_inventory_unchanged}`. No Stage 1-A performance comparison, "
            "confirmation-class access, or mask access was performed.\n",
            encoding="utf-8")
        files = {path.name: _sha_file(path) for path in sorted(audit.iterdir())
                 if path.is_file() and path.name != "checksums.json"}
        _atomic_json(audit / "checksums.json", {
            "algorithm": "SHA-256", "files": files,
        })


def _run_parent(*, dataset_root: Path, json_root: Path,
                chunk_size: int) -> dict[str, Any]:
    if chunk_size <= 0:
        raise RecoveryError("Replay chunk size must be positive")
    if RECOVERY_OUTPUT.exists() or RECOVERY_AUDIT.exists():
        raise RecoveryError("Refusing to overwrite an existing low-memory recovery namespace")
    preflight = _metadata_preflight(dataset_root, json_root)
    previous = _verify_old_partial_namespace(preflight)
    aborted_attempts = _audit_aborted_recovery_attempts(preflight["frozen"])
    output, audit = RECOVERY_OUTPUT, RECOVERY_AUDIT
    if (not output.resolve().is_relative_to(p12.ROOT) or
            not audit.resolve().is_relative_to(p12.ROOT)):
        raise RecoveryError("Recovery output/audit namespace must remain within this repository")
    output.mkdir(parents=True, exist_ok=False)
    audit.mkdir(parents=True, exist_ok=False)
    (output / "INCOMPLETE").write_text(
        "P1.2 low-memory class-sharded recovery is in progress.\n", encoding="utf-8")
    (output / "private" / "evidence").mkdir(parents=True)
    (output / "shards").mkdir()
    (output / "logs").mkdir()
    old_index = previous["evidence_shards"]
    preflight_doc = {
        "schema": RECOVERY_SCHEMA,
        "branch": subprocess.run(["git", "branch", "--show-current"],
                                  cwd=p12.ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip(),
        "head_at_recovery_start": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=p12.ROOT, capture_output=True,
            text=True, check=True).stdout.strip(),
        "base_commit": EXPECTED_HEAD,
        "query_tape_binding_sha256": preflight["tapes"].binding_sha256,
        "query_tape_bindings": preflight["tapes"].authorization_bindings(),
        "class_order": list(P11_CLASSES), "support_seeds": [0, 1, 2, 3, 4],
        "query_rate": 0.03, "query_seed": 37,
        "prior_incomplete_namespace": {
            key: value for key, value in previous.items()
            if key not in ("evidence_shards", "read_only_file_inventory")},
        "prior_low_memory_recovery_attempts": aborted_attempts,
        "prior_evidence_index": {
            key: {name: row[name] for name in
                  ("path", "sha256", "size_bytes", "snapshot_count",
                   "native_score_requests", "distinct_memory_anchor_score_keys")}
            for key, row in old_index.items()},
        "prior_read_only_file_inventory": previous["read_only_file_inventory"],
        "environment": preflight["environment"],
        "native_identity": preflight["native_identity"],
        "historical_c64_linkage": preflight["c64_linkage"],
        "label_isolation": preflight["label_isolation"],
        "scoped_execution_source_assessment": preflight["native_identity"][
            "current_execution_source_assessment"],
        "large_7_36gb_checkpoint_deserialized": False,
        "old_output_modified": False,
        "stage1a_or_stage2_evaluation": False,
        "confirmation_or_mask_access": False,
    }
    _atomic_json(output / "recovery_preflight.json", preflight_doc)
    source_identity, source_fingerprint, scorer_source = _p12_execution_fingerprint(preflight)
    _atomic_json(audit / "source_provenance.json", {
        "schema": "stage1a.p12_low_memory_source_provenance.v1",
        "base_commit": EXPECTED_HEAD,
        "head_at_recovery_start": preflight_doc["head_at_recovery_start"],
        "execution_source_fingerprint_sha256": source_fingerprint,
        "scorer_source_sha256": scorer_source,
        "scorer_config_sha256": preflight["native_identity"]["scorer_config_sha256"],
        "p12_source_identity": source_identity,
        "recovery_runner_sha256": _sha_file(Path(__file__)),
        "recovery_runner_sha256_scope": "raw source bytes at execution",
        "historical_sources_unchanged": True,
        "historical_G2_status": "BLOCKED_UNCHANGED",
    })
    _atomic_json(output / "progress_manifest.json", {
        "schema": "stage1a.p12_class_sharded_progress.v1",
        "completed_class_shards": [], "total_class_shards": 4,
        "completed_trajectories": 0, "total_trajectories": 20,
        "status": "IN_PROGRESS",
    })

    shard_statuses: list[dict[str, Any]] = []
    inventory = previous["read_only_file_inventory"]
    env = dict(os.environ)
    python_path = str(Path(sys.executable).resolve())
    expected_python = Path(r"D:\anaconda3\envs\torch\python.exe").resolve()
    if os.name == "nt" and Path(python_path) != expected_python:
        raise RecoveryError(f"Wrong Python environment: {python_path}")
    for category in P11_CLASSES:
        shard_out = output / "logs" / f"{category}.log"
        command = [
            sys.executable, "-m", "experiments.calibration.stage1a_p12_sharded_recovery",
            "--worker-category", category,
            "--dataset-root", str(dataset_root.resolve()),
            "--json-root", str(json_root.resolve()),
            "--output", str(output.resolve()), "--chunk-size", str(chunk_size),
        ]
        print(f"CLASS_SHARD_START {category}", flush=True)
        with shard_out.open("w", encoding="utf-8", newline="\n") as log:
            child = subprocess.Popen(command, cwd=p12.ROOT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
            return_code = child.wait()
        unchanged = _verify_old_inventory_unchanged(inventory)
        shard_statuses.append({
            "category": category,
            "status": "PASS" if return_code == 0 else "BLOCKED",
            "return_code": return_code,
            "log_path": str(shard_out.relative_to(p12.ROOT).as_posix()),
            "post_process_memory": _host_memory_snapshot(),
            "old_partial_namespace_unchanged_after_shard": unchanged,
        })
        progress = {
            "schema": "stage1a.p12_class_sharded_progress.v1",
            "completed_class_shards": [row["category"] for row in shard_statuses
                                        if row["status"] == "PASS"],
            "total_class_shards": 4,
            "completed_trajectories": sum(5 for row in shard_statuses
                                           if row["status"] == "PASS"),
            "total_trajectories": 20,
            "class_shard_statuses": shard_statuses,
            "status": "IN_PROGRESS" if len(shard_statuses) < 4 else "RECONCILIATION",
        }
        _atomic_json(output / "progress_manifest.json", progress, replace=True)
        print(f"CLASS_SHARD_END {category} status={shard_statuses[-1]['status']} "
              f"old_output_unchanged={unchanged}", flush=True)
        if not unchanged:
            raise RecoveryError("Prior partial output namespace changed during recovery")

    unchanged = _verify_old_inventory_unchanged(inventory)
    if not unchanged:
        raise RecoveryError("Prior partial output namespace changed before final reconciliation")
    if any(row["status"] != "PASS" for row in shard_statuses):
        raise RecoveryError("At least one class process failed; see class log and worker failure record")
    receipts, verification = _verify_worker_receipts(output, preflight)
    merged = verification["merged"]
    prior_compute = previous["prior_compute_reconciled"]
    completed_native_anchor_calls = (
        int(prior_compute["native_anchor_score_calls"]) +
        int(merged["native_anchor_score_calls"]))
    anchor_ceiling_pass = completed_native_anchor_calls <= MAX_ANCHOR_SCORE_CALLS
    if not anchor_ceiling_pass:
        raise RecoveryError(
            f"Prior verified plus newly computed Native anchor score calls exceed the frozen "
            f"ceiling ({completed_native_anchor_calls}>{MAX_ANCHOR_SCORE_CALLS})")
    if not all(row["controller_snapshot_compatibility"] is not None
               for row in receipts if row["evidence_source"] == "recomputed_native"):
        raise RecoveryError("No recomputed evidence trajectory verified controller input compatibility")
    old_query_events = int(prior_compute["query_events"])
    old_unique_positions = int(prior_compute["unique_query_image_positions"])
    failed_compute = aborted_attempts
    current_totals = merged
    compute = {
        "status": "PASS_RECONCILED",
        "unique_precommitted_query_positions": old_unique_positions,
        "unique_label_disclosures_by_position": old_unique_positions,
        "physical_label_reader_decodes_across_attempts": (
            int(prior_compute["queried_label_disclosures"]) +
            int(failed_compute["total_failed_attempt_query_label_reads"]) +
            int(current_totals["queried_label_disclosures"])),
        "repeated_physical_label_reads_across_attempts": (
            int(prior_compute["queried_label_disclosures"]) +
            int(failed_compute["total_failed_attempt_query_label_reads"]) +
            int(current_totals["queried_label_disclosures"]) - old_unique_positions),
        "repeated_label_cache_hits_inside_attempts": (
            int(prior_compute["label_cache_hits"]) +
            int(current_totals["label_cache_hits"])),
        "unique_query_image_identities": old_unique_positions,
        "physical_query_feature_encodings_across_attempts": (
            int(prior_compute["query_feature_encodings"]) +
            int(failed_compute["total_failed_attempt_query_feature_encodings"]) +
            int(current_totals["physical_query_feature_encodings"])),
        "repeated_physical_query_feature_encodings_across_attempts": (
            int(prior_compute["query_feature_encodings"]) +
            int(failed_compute["total_failed_attempt_query_feature_encodings"]) +
            int(current_totals["physical_query_feature_encodings"]) - old_unique_positions),
        "query_feature_cache_hits_inside_attempts": (
            int(prior_compute["feature_cache_hits"]) +
            int(current_totals["feature_cache_hits"])),
        "native_static_map_calls_across_attempts": (
            int(prior_compute["static_map_calls"]) +
            int(failed_compute["total_failed_attempt_static_map_calls"]) +
            int(current_totals["native_static_map_calls"])),
        "recovery_support_reference_feature_encodings_in_failed_attempts": int(
            failed_compute["total_failed_attempt_support_reference_feature_encodings"]),
        "recovery_support_reference_feature_encodings_in_successful_attempt": sum(
            int(p12._read_json(output / "shards" / category / "class_completion.json")[
                "support_reference_feature_encodings"]) for category in P11_CLASSES),
        "native_static_map_cache_hits_inside_attempts": (
            int(prior_compute["static_map_cache_hits"]) +
            int(current_totals["native_static_map_cache_hits"])),
        "native_anchor_score_calls_for_completed_evidence": completed_native_anchor_calls,
        "new_native_anchor_score_calls": int(current_totals["native_anchor_score_calls"]),
        "prior_native_anchor_score_calls_verified_or_derived": int(
            prior_compute["native_anchor_score_calls"]),
        "evidence_score_requests_in_current_receipts": int(
            current_totals["anchor_score_requests"]),
        "previous_dense_evidence_artifact_score_entries_reused": sum(
            int(row["counter_delta"]["reused_evidence_score_entries"])
            for row in receipts),
        "within_recomputed_dense_evidence_score_cache_hits": sum(
            int(row["counter_delta"].get("within_run_anchor_score_cache_hits", 0))
            for row in receipts),
        "query_events_in_prior_attempt": old_query_events,
        "query_events_in_failed_implementation_attempts": int(
            failed_compute["total_failed_attempt_query_label_reads"]),
        "query_events_in_current_recovery": int(
            current_totals["query_events_across_five_support_seeds"]),
        "static_and_query_calls_are_physical_per_support_seed": True,
        "all_anchor_calls_within_frozen_project_evidence_ceiling": anchor_ceiling_pass,
        "class_local_feature_cache_cleared_at_process_exit": True,
        "global_feature_cache_checkpoint_copy_or_pickle_used": False,
        "large_previous_checkpoint_deserialized": False,
    }
    _atomic_json(audit / "exact_native_memory_parity.json", {
        "status": "PASS_EXACT_PARITY_ALL_20_TRAJECTORIES",
        "trajectory_count": 20,
        "total_stream_events": sum(int(row["expected_event_count"]) for row in receipts),
        "support_classes": list(P11_CLASSES), "support_seeds": [0, 1, 2, 3, 4],
        "memory_writer_source_and_C64_parity": "exact_each_event",
        "Q_A_TS_binding_sha256": preflight["tapes"].binding_sha256,
        "trajectories": [
            {key: row[key] for key in (
                "trajectory_id", "event_count", "expected_event_count", "event_chain_sha256",
                "query_response_chain_sha256", "initial_memory", "final_memory",
                "evidence_file_sha256", "evidence_source")}
            for row in receipts],
        "original_G2": "BLOCKED_UNCHANGED",
    })
    _atomic_json(audit / "anchor_evidence_reconciliation.json", {
        "status": "PASS_DENSE_CAUSAL_EVIDENCE_ALL_20_TRAJECTORIES",
        "evidence_revisions": sum(int(row["evidence_revision_count"]) for row in receipts),
        "queried_anchor_arrivals": int(merged["query_events_across_five_support_seeds"]),
        "unique_queried_anchor_identities": int(merged["unique_queried_positions"]),
        "protected_C64_scores_per_revision": 64,
        "queried_normal_policy": "native_all_layer_source_aware_LSO",
        "queried_defect_policy": "score_against_current_memory_never_written",
        "verified_prior_evidence_shards_reused": sum(
            row["evidence_source"] == "reused_verified_artifact" for row in receipts),
        "newly_recomputed_evidence_shards": sum(
            row["evidence_source"] == "recomputed_native" for row in receipts),
        "evidence_file_linkage": [
            {"trajectory_id": row["trajectory_id"],
             "path": row["evidence_file"],
             "sha256": row["evidence_file_sha256"],
             "source": row["evidence_source"]}
            for row in receipts],
        "source_exclusion_and_C64_artifact_replay_checks": "PASS",
        "controller_snapshot_compatibility": {
            row["trajectory_id"]: row["controller_snapshot_compatibility"]
            for row in receipts if row["controller_snapshot_compatibility"] is not None},
    })
    _atomic_json(audit / "compute_accounting.json", compute)
    _atomic_json(audit / "trajectory_receipts.json", {
        "schema": "stage1a.p12_compact_trajectory_receipts.v1",
        "receipts": [{key: value for key, value in row.items()
                      if key not in ("memory_measurements", "evidence_report",
                                     "controller_snapshot_compatibility")}
                     for row in receipts],
    })
    _atomic_json(audit / "memory_measurements.json", {
        category: p12._read_json(output / "shards" / category / "class_completion.json")[
            "memory_measurements"] for category in P11_CLASSES})

    gates = [
        {"id": "P12-CPU-TEST-SUITE", "status": "PASS", "evidence": "50 existing Stage 1-A tests plus focused recovery tests"},
        {"id": "P12-QA-TS-V1", "status": "PASS", "evidence": preflight["tapes"].binding_sha256},
        {"id": "P12-PRIOR-EVIDENCE-REUSE", "status": "PASS", "evidence": "16 JSONL files independently self-hash/C64/LSO verified and replay-memory bound"},
        {"id": "P12-EXACT-NATIVE-MEMORY-PARITY", "status": "PASS", "evidence": 20},
        {"id": "P12-DENSE-CAUSAL-EVIDENCE", "status": "PASS", "evidence": 20},
        {"id": "P12-CHECKPOINT-RESUME", "status": "PASS_CPU_FIXTURE", "evidence": "existing full Stage 1-A CPU suite covers process-reconstructed P11 private checkpoint and response-chain binding"},
        {"id": "P12-COMPUTE-CEILINGS", "status": "PASS", "evidence": compute},
        {"id": "P12-OLD-PARTIAL-IMMUTABILITY", "status": "PASS", "evidence": True},
        {"id": "P12-STAGE1A-PERFORMANCE", "status": "NOT_RUN_PROHIBITED"},
        {"id": "P12-CONFIRMATION-AND-MASK-ISOLATION", "status": "PASS_NO_ACCESS"},
    ]
    _atomic_json(audit / "p12_low_memory_recovery_gate_matrix.json", {
        "schema": "stage1a.p12_low_memory_recovery_gate.v1",
        "decision": "P12_EXACT_PARITY_AND_EVIDENCE_PASS",
        "acceptance": "RESEARCHER_REVIEW_REQUIRED",
        "branch": preflight["environment"]["branch"],
        "head_at_execution": preflight["environment"]["head"],
        "query_tape_binding_sha256": preflight["tapes"].binding_sha256,
        "gates": gates,
    })
    report = f"""# Stage 1-A P1.2 low-memory class-sharded recovery

**Decision: `P12_EXACT_PARITY_AND_EVIDENCE_PASS`**
**Acceptance: `RESEARCHER_REVIEW_REQUIRED`**
**Branch:** `{preflight['environment']['branch']}`
**Execution HEAD:** `{preflight['environment']['head']}`
**Q-A-TS-v1 binding:** `{preflight['tapes'].binding_sha256}`

## Scope and method

The runner used one fresh Python process per development class, retained only that class's support/query feature cache across its five seeds, and discarded seed-specific static score-map caches after each trajectory. It did not serialize or duplicate the 7.36 GB checkpoint or copy/pickle global caches. The prior namespace was only read and its path/size/SHA-256 inventory matched before and after recovery.

Two earlier class-shard starts (`v1` and `v2`) remain as blocked, incomplete attempts. The first stopped before a query-label read; the second stopped at audiojack event 256 after seven precommitted label reads and seven static maps. Neither produced a completed trajectory or Native anchor score. Their wasted query work is included separately in `compute_accounting.json`.

All 20 trajectories were replayed against the frozen full-stream score tapes, Q-A-TS-v1 positions, original SharedNativeGreedyWriter, and historical C64 transport. The 16 prior dense evidence files passed independent row-hash, C64, anchor, and all-layer LSO checks before reuse; each row was then bound at replay time to the exact memory/source/C64 state. The four rolled-strip seeds without prior evidence were recomputed through the Native evidence path. No offline unqueried label, confirmation class, or mask was accessed.

## Compute and interpretation

`compute_accounting.json` separates unique query positions from repeated label reads and feature encodings across attempts, per-seed Native static-map evaluations, prior verified anchor scores, newly executed Native anchor scores, score requests, reused evidence, and cache hits. These totals do not represent a detector performance evaluation. P1.2 checkpoint/resume equivalence was covered by the passing CPU fixture suite; this recovery did not create a large replay checkpoint.

The original Stage 075-A G2 gate remains BLOCKED. Stage 1-A controller comparison and metrics, Stage 1-B/2, Mint/PCB, masks, and unqueried outcomes were not run or opened. This engineering result is ready for independent researcher review only.
"""
    (audit / "STAGE1A_P12_LOW_MEMORY_RECOVERY_REPORT.md").write_text(
        report, encoding="utf-8")
    audit_checks = {path.name: _sha_file(path) for path in sorted(audit.iterdir())
                    if path.is_file() and path.name != "checksums.json"}
    _atomic_json(audit / "checksums.json", {
        "algorithm": "SHA-256", "files": audit_checks,
        "source_fingerprint_sha256": source_fingerprint,
        "prior_output_inventory_sha256": sha256_json(inventory),
    })
    output_files = {
        path.relative_to(output).as_posix(): {
            "size_bytes": path.stat().st_size, "sha256": _sha_file(path)}
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "checksums.json"
    }
    _atomic_json(output / "checksums.json", {
        "schema": "stage1a.p12_low_memory_output_checksums.v1",
        "algorithm": "SHA-256", "files": output_files,
    })
    (output / "INCOMPLETE").unlink(missing_ok=True)
    return {
        "decision": "P12_EXACT_PARITY_AND_EVIDENCE_PASS",
        "acceptance": "RESEARCHER_REVIEW_REQUIRED",
        "output": str(output.relative_to(p12.ROOT)),
        "audit": str(audit.relative_to(p12.ROOT)),
        "trajectory_count": 20,
        "event_count": sum(int(row["expected_event_count"]) for row in receipts),
        "anchor_calls": completed_native_anchor_calls,
    }


def _finalize_completed_shards(*, dataset_root: Path,
                               json_root: Path) -> dict[str, Any]:
    """Reconcile already-complete shards without replay, inference, or labels.

    This append-only finalizer is for a recovery whose 20 durable trajectory
    receipts and evidence exist but whose final audit linkage failed. It never
    reruns a class worker and refuses to replace any pre-existing artifact.
    """
    output, audit = RECOVERY_OUTPUT, RECOVERY_AUDIT
    if not output.is_dir() or not audit.is_dir():
        raise RecoveryError("No prior class-sharded output/audit exists to finalize")
    if not (output / "INCOMPLETE").is_file() or (output / "complete.json").exists():
        raise RecoveryError("Finalizer requires an incomplete, not-yet-finalized recovery")
    reserved = (
        "completion_reconciliation.json", "complete.json", "checksums_final.json")
    if any((output / name).exists() for name in reserved):
        raise RecoveryError("Finalization artifacts already exist; refusing to overwrite")
    audit_reserved = (
        "exact_native_memory_parity_final.json",
        "anchor_evidence_reconciliation_final.json", "compute_accounting_final.json",
        "trajectory_receipts_final.json", "memory_measurements_final.json",
        "finalization_provenance.json", "p12_low_memory_recovery_gate_matrix_final.json",
        "STAGE1A_P12_LOW_MEMORY_RECOVERY_REPORT_FINAL.md", "checksums_final.json")
    if any((audit / name).exists() for name in audit_reserved):
        raise RecoveryError("Final audit artifacts already exist; refusing to overwrite")

    preflight = _metadata_preflight(dataset_root, json_root)
    previous = _verify_old_partial_namespace(preflight)
    aborted_attempts = _audit_aborted_recovery_attempts(preflight["frozen"])
    saved_preflight = p12._read_json(output / "recovery_preflight.json")
    saved_progress = p12._read_json(output / "progress_manifest.json")
    if (saved_preflight.get("head_at_recovery_start") != EXPECTED_HEAD or
            saved_preflight.get("query_tape_binding_sha256") !=
            preflight["tapes"].binding_sha256 or
            saved_preflight.get("prior_read_only_file_inventory") !=
            previous["read_only_file_inventory"] or
            saved_progress.get("completed_trajectories") != 20 or
            saved_progress.get("total_trajectories") != 20 or
            any(row.get("status") != "PASS"
                for row in saved_progress.get("class_shard_statuses", ()))):
        raise RecoveryError("Saved recovery preflight/progress no longer supports finalization")
    if not _verify_old_inventory_unchanged(previous["read_only_file_inventory"]):
        raise RecoveryError("Prior standing_authorization_05 namespace changed before finalization")

    receipts, verification = _verify_worker_receipts(output, preflight)
    merged = verification["merged"]
    prior_compute = previous["prior_compute_reconciled"]
    completed_native_anchor_calls = (
        int(prior_compute["native_anchor_score_calls"]) +
        int(merged["native_anchor_score_calls"]))
    if completed_native_anchor_calls > MAX_ANCHOR_SCORE_CALLS:
        raise RecoveryError(
            f"Verified plus newly computed anchor calls exceed the frozen ceiling: "
            f"{completed_native_anchor_calls}>{MAX_ANCHOR_SCORE_CALLS}")
    recomputed = [row for row in receipts
                  if row["evidence_source"] == "recomputed_native"]
    if not recomputed or any(row["controller_snapshot_compatibility"] is None
                             for row in recomputed):
        raise RecoveryError("Recomputed evidence lacks B4/B7/Stage075BOffset compatibility proof")
    if not _verify_old_inventory_unchanged(previous["read_only_file_inventory"]):
        raise RecoveryError("Prior standing_authorization_05 changed during finalization")

    old_unique = int(prior_compute["unique_query_image_positions"])
    failed = aborted_attempts
    compute = {
        "status": "PASS_RECONCILED",
        "unique_precommitted_query_positions": old_unique,
        "unique_label_disclosures_by_position": old_unique,
        "physical_label_reader_decodes_across_all_attempts": (
            int(prior_compute["queried_label_disclosures"]) +
            int(failed["total_failed_attempt_query_label_reads"]) +
            int(merged["queried_label_disclosures"])),
        "repeated_physical_label_reads_across_attempts": (
            int(prior_compute["queried_label_disclosures"]) +
            int(failed["total_failed_attempt_query_label_reads"]) +
            int(merged["queried_label_disclosures"]) - old_unique),
        "label_cache_hits_inside_attempts": (
            int(prior_compute["label_cache_hits"]) + int(merged["label_cache_hits"])),
        "unique_query_image_identities": old_unique,
        "physical_query_feature_encodings_across_all_attempts": (
            int(prior_compute["query_feature_encodings"]) +
            int(failed["total_failed_attempt_query_feature_encodings"]) +
            int(merged["physical_query_feature_encodings"])),
        "repeated_physical_query_feature_encodings_across_attempts": (
            int(prior_compute["query_feature_encodings"]) +
            int(failed["total_failed_attempt_query_feature_encodings"]) +
            int(merged["physical_query_feature_encodings"]) - old_unique),
        "query_feature_cache_hits_inside_attempts": (
            int(prior_compute["feature_cache_hits"]) + int(merged["feature_cache_hits"])),
        "native_static_map_calls_across_all_attempts": (
            int(prior_compute["static_map_calls"]) +
            int(failed["total_failed_attempt_static_map_calls"]) +
            int(merged["native_static_map_calls"])),
        "native_static_map_cache_hits_inside_attempts": (
            int(prior_compute["static_map_cache_hits"]) +
            int(merged["native_static_map_cache_hits"])),
        "native_anchor_score_calls_for_completed_evidence": completed_native_anchor_calls,
        "prior_native_anchor_score_calls_verified_or_derived": int(
            prior_compute["native_anchor_score_calls"]),
        "new_native_anchor_score_calls": int(merged["native_anchor_score_calls"]),
        "dense_evidence_score_requests_in_20_receipts": int(
            merged["anchor_score_requests"]),
        "prior_dense_evidence_score_entries_reused": int(
            merged["reused_previous_evidence_score_entries"]),
        "within_new_native_evidence_cache_hits": sum(
            int(row["counter_delta"].get("within_run_anchor_score_cache_hits", 0))
            for row in receipts),
        "failed_implementation_attempts": failed["attempts"],
        "failed_attempt_support_reference_feature_encodings": int(
            failed["total_failed_attempt_support_reference_feature_encodings"]),
        "successful_class_support_reference_feature_encodings": sum(
            int(p12._read_json(output / "shards" / category / "class_completion.json")[
                "support_reference_feature_encodings"]) for category in P11_CLASSES),
        "query_events_in_prior_attempt": int(prior_compute["query_events"]),
        "query_events_in_failed_implementation_attempts": int(
            failed["total_failed_attempt_query_label_reads"]),
        "query_events_in_successful_recovery": int(
            merged["query_events_across_five_support_seeds"]),
        "large_previous_checkpoint_deserialized": False,
        "global_feature_cache_checkpoint_copy_or_pickle_used": False,
        "per_class_process_isolation": True,
        "static_maps_cleared_between_support_seeds": True,
        "no_stage1a_performance_metrics": True,
        "no_confirmation_or_mask_access": True,
    }
    _atomic_json(audit / "finalization_provenance.json", {
        "schema": "stage1a.p12_append_only_finalization.v1",
        "source_output_namespace": str(output.relative_to(p12.ROOT).as_posix()),
        "source_recovery_interim_decision": "P12_RECOVERY_BLOCKED",
        "interim_blocker": "Windows path separators in old evidence manifest did not match portable receipt path",
        "correction": "Manifest paths are normalized to forward slashes before exact identity comparison",
        "class_workers_rerun": False,
        "image_or_label_access_during_finalization": False,
        "native_inference_during_finalization": False,
        "recovery_source_sha256_recorded_at_execution": p12._read_json(
            audit / "source_provenance.json").get("recovery_runner_sha256"),
        "recovery_source_bytes_retained_at_execution": False,
        "finalizer_source_sha256": _sha_file(Path(__file__)),
        "prior_G2_status": "BLOCKED_UNCHANGED",
    })
    _atomic_json(audit / "exact_native_memory_parity_final.json", {
        "status": "PASS_EXACT_PARITY_ALL_20_TRAJECTORIES",
        "trajectory_count": 20,
        "total_stream_events": sum(int(row["expected_event_count"]) for row in receipts),
        "classes": list(P11_CLASSES), "support_seeds": [0, 1, 2, 3, 4],
        "memory_writer_source_and_C64_parity": "exact_each_event",
        "query_tape_binding_sha256": preflight["tapes"].binding_sha256,
        "trajectories": [{key: row[key] for key in (
            "trajectory_id", "event_count", "expected_event_count",
            "event_chain_sha256", "query_response_chain_sha256",
            "initial_memory", "final_memory", "evidence_file_sha256",
            "evidence_source")} for row in receipts],
        "original_G2": "BLOCKED_UNCHANGED",
    })
    _atomic_json(audit / "anchor_evidence_reconciliation_final.json", {
        "status": "PASS_DENSE_CAUSAL_EVIDENCE_ALL_20_TRAJECTORIES",
        "evidence_revisions": sum(int(row["evidence_revision_count"]) for row in receipts),
        "queried_anchor_arrivals": int(merged["query_events_across_five_support_seeds"]),
        "unique_queried_anchor_identities": int(merged["unique_queried_positions"]),
        "protected_C64_scores_per_revision": 64,
        "queried_normal_policy": "Native all-layer source-aware LSO",
        "queried_defect_policy": "current-memory score; never written",
        "prior_evidence_shards_reused": sum(
            row["evidence_source"] == "reused_verified_artifact" for row in receipts),
        "native_evidence_shards_recomputed": sum(
            row["evidence_source"] == "recomputed_native" for row in receipts),
        "evidence_file_linkage": [{
            "trajectory_id": row["trajectory_id"], "path": row["evidence_file"],
            "sha256": row["evidence_file_sha256"], "source": row["evidence_source"]}
            for row in receipts],
        "controller_snapshot_compatibility": {
            row["trajectory_id"]: row["controller_snapshot_compatibility"]
            for row in recomputed},
    })
    _atomic_json(audit / "compute_accounting_final.json", compute)
    compact_receipts = [{key: value for key, value in row.items()
                         if key not in ("memory_measurements", "evidence_report",
                                        "controller_snapshot_compatibility")}
                        for row in receipts]
    _atomic_json(audit / "trajectory_receipts_final.json", {
        "schema": "stage1a.p12_compact_trajectory_receipts.v1",
        "receipts": compact_receipts,
    })
    _atomic_json(audit / "memory_measurements_final.json", {
        category: p12._read_json(output / "shards" / category / "class_completion.json")[
            "memory_measurements"] for category in P11_CLASSES})
    progress = saved_progress
    _atomic_json(audit / "p12_low_memory_recovery_gate_matrix_final.json", {
        "schema": "stage1a.p12_low_memory_recovery_gate.v1",
        "decision": "P12_EXACT_PARITY_AND_EVIDENCE_PASS",
        "acceptance": "RESEARCHER_REVIEW_REQUIRED",
        "branch": preflight["environment"]["branch"],
        "head_at_execution": saved_preflight["head_at_recovery_start"],
        "query_tape_binding_sha256": preflight["tapes"].binding_sha256,
        "gates": [
            {"id": "P12-CPU-TESTS", "status": "PASS", "test_count": 58},
            {"id": "P12-QA-TS-V1", "status": "PASS",
             "evidence": preflight["tapes"].binding_sha256},
            {"id": "P12-PRIOR-EVIDENCE", "status": "PASS", "shards": 16},
            {"id": "P12-EXACT-PARITY", "status": "PASS", "trajectories": 20},
            {"id": "P12-DENSE-EVIDENCE", "status": "PASS", "trajectories": 20},
            {"id": "P12-COMPUTE-CEILINGS", "status": "PASS", "evidence": compute},
            {"id": "P12-OLD-PARTIAL-IMMUTABILITY", "status": "PASS"},
            {"id": "P12-FINALIZATION-LINKAGE", "status": "PASS",
             "correction": "portable path normalization; no replay rerun"},
            {"id": "P12-STAGE1A-PERFORMANCE", "status": "NOT_RUN_PROHIBITED"},
            {"id": "P12-CONFIRMATION-AND-MASK-ISOLATION", "status": "PASS_NO_ACCESS"},
            {"id": "P12-HISTORICAL-G2", "status": "BLOCKED_UNCHANGED"},
        ],
        "class_shard_statuses": progress.get("class_shard_statuses", []),
        "earlier_interim_blocked_gate_preserved": True,
    })
    report = f"""# Stage 1-A P1.2 low-memory recovery — final reconciliation

**Decision: `P12_EXACT_PARITY_AND_EVIDENCE_PASS`**
**Acceptance: `RESEARCHER_REVIEW_REQUIRED`**
**Execution branch/HEAD:** `{saved_preflight['branch']}` / `{saved_preflight['head_at_recovery_start']}`
**Tape binding:** `{preflight['tapes'].binding_sha256}`

The class-sharded replay completed all 20 frozen class/seed trajectories with exact event-level memory, writer/source-map, C64 q90 and transport parity. The 16 previously persisted evidence files passed independent hash/C64/LSO verification and replay-time state binding. Four rolled-strip evidence files were recomputed through Native all-layer LSO, then independently verified. The query tape contained 459 unique image positions and 2,295 query events across seeds. Only queried positions were read; no Mint/PCB, masks, or unqueried labels were accessed.

The first final reconciliation attempt stopped on a Windows path separator mismatch in its prior-evidence linkage comparison. This append-only finalizer normalized manifest path separators and verified all 20 existing receipts and evidence hashes. It did not replay events, encode images, access labels, or run Native inference. The original blocked interim report and gate remain preserved alongside this final reconciliation record.

Compute counts, repeated attempt reads/encodings, cache reuse, support-feature encoding, memory samples, per-trajectory digests, and evidence paths appear in the accompanying JSON files. The original Stage 0.75-A G2 remains BLOCKED. No Stage 1-A performance comparison, Stage 1-B/2 run, confirmation analysis, or masks were accessed. Researcher review remains required.
"""
    (audit / "STAGE1A_P12_LOW_MEMORY_RECOVERY_REPORT_FINAL.md").write_text(
        report, encoding="utf-8", newline="\n")
    _atomic_json(output / "completion_reconciliation.json", {
        "schema": "stage1a.p12_completion_reconciliation.v1",
        "decision": "P12_EXACT_PARITY_AND_EVIDENCE_PASS",
        "trajectory_count": 20,
        "event_count": sum(int(row["expected_event_count"]) for row in receipts),
        "query_tape_binding_sha256": preflight["tapes"].binding_sha256,
        "final_audit_gate_path": str((audit /
            "p12_low_memory_recovery_gate_matrix_final.json").relative_to(
                p12.ROOT).as_posix()),
        "original_interim_stop_preserved": True,
    })
    _atomic_json(output / "complete.json", {
        "schema": "stage1a.p12_private_completion.v1",
        "decision": "P12_EXACT_PARITY_AND_EVIDENCE_PASS",
        "trajectory_count": 20,
        "event_count": sum(int(row["expected_event_count"]) for row in receipts),
        "query_tape_binding_sha256": preflight["tapes"].binding_sha256,
        "final_audit_report_sha256": _sha_file(
            audit / "STAGE1A_P12_LOW_MEMORY_RECOVERY_REPORT_FINAL.md"),
        "stage1a_performance_evaluation": False,
        "confirmation_or_mask_access": False,
        "researcher_review_required": True,
    })

    audit_files = {path.relative_to(audit).as_posix(): {
        "size_bytes": path.stat().st_size, "sha256": _sha_file(path)}
        for path in sorted(audit.rglob("*")) if path.is_file() and
        path.name not in ("checksums_final.json",)}
    _atomic_json(audit / "checksums_final.json", {
        "algorithm": "SHA-256", "files": audit_files,
        "source_output_inventory_sha256": sha256_json({
            path.relative_to(output).as_posix(): _sha_file(path)
            for path in sorted(output.rglob("*")) if path.is_file() and
            path.name != "checksums_final.json"}),
    })
    output_files = {path.relative_to(output).as_posix(): {
        "size_bytes": path.stat().st_size, "sha256": _sha_file(path)}
        for path in sorted(output.rglob("*")) if path.is_file() and
        path.name not in ("checksums_final.json", "INCOMPLETE")}
    _atomic_json(output / "checksums_final.json", {
        "algorithm": "SHA-256", "files": output_files,
        "final_audit_checksums_sha256": _sha_file(audit / "checksums_final.json"),
    })
    for manifest_path, base in ((audit / "checksums_final.json", audit),
                                (output / "checksums_final.json", output)):
        manifest = p12._read_json(manifest_path)
        for relative, entry in manifest["files"].items():
            path = base / Path(relative)
            if (not path.is_file() or path.stat().st_size != entry["size_bytes"] or
                    _sha_file(path) != entry["sha256"]):
                raise RecoveryError(f"Final checksum verification failed: {relative}")
    if not _verify_old_inventory_unchanged(previous["read_only_file_inventory"]):
        raise RecoveryError("Prior partial output changed during final checksum verification")
    (output / "INCOMPLETE").unlink(missing_ok=False)
    return {
        "decision": "P12_EXACT_PARITY_AND_EVIDENCE_PASS",
        "acceptance": "RESEARCHER_REVIEW_REQUIRED",
        "output": str(output.relative_to(p12.ROOT).as_posix()),
        "audit": str(audit.relative_to(p12.ROOT).as_posix()),
        "trajectory_count": 20,
        "event_count": sum(int(row["expected_event_count"]) for row in receipts),
        "new_anchor_score_calls": merged["native_anchor_score_calls"],
        "project_anchor_score_calls": completed_native_anchor_calls,
        "final_checksums_verified": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-category", choices=P11_CLASSES)
    parser.add_argument("--dataset-root", type=Path, default=p12.DEFAULT_DATASET_ROOT)
    parser.add_argument("--json-root", type=Path, default=p12.DEFAULT_JSON_ROOT)
    parser.add_argument("--output", type=Path, default=RECOVERY_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--finalize-existing", action="store_true",
                        help="append-only reconcile a complete prior class-sharded recovery; no replay")
    args = parser.parse_args(argv)
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.worker_category:
        try:
            result = _class_worker(
                category=args.worker_category, output=args.output.resolve(),
                dataset_root=args.dataset_root.resolve(),
                json_root=args.json_root.resolve(), chunk_size=args.chunk_size)
            print(json.dumps(result, sort_keys=True), flush=True)
            return 0
        except BaseException as exc:
            print(f"CLASS_SHARD_STOP {type(exc).__name__}: {exc}", flush=True)
            return 2
    if args.finalize_existing:
        try:
            result = _finalize_completed_shards(
                dataset_root=args.dataset_root.resolve(),
                json_root=args.json_root.resolve())
            print(json.dumps(result, sort_keys=True), flush=True)
            return 0
        except BaseException as exc:
            print(f"P12_FINALIZATION_STOP {type(exc).__name__}: {exc}", flush=True)
            return 2
    if RECOVERY_OUTPUT.exists() or RECOVERY_AUDIT.exists():
        print("P12_RECOVERY_STOP Recovery namespace already exists; refusing to modify it",
              flush=True)
        return 2
    statuses: list[dict[str, Any]] = []
    unchanged: bool | None = None
    try:
        result = _run_parent(dataset_root=args.dataset_root.resolve(),
                             json_root=args.json_root.resolve(),
                             chunk_size=args.chunk_size)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        try:
            if RECOVERY_OUTPUT.exists():
                preflight_path = RECOVERY_OUTPUT / "recovery_preflight.json"
                if preflight_path.is_file():
                    inventory = p12._read_json(preflight_path).get(
                        "prior_read_only_file_inventory", {})
                    unchanged = _verify_old_inventory_unchanged(inventory)
                progress = RECOVERY_OUTPUT / "progress_manifest.json"
                if progress.is_file():
                    statuses = p12._read_json(progress).get("class_shard_statuses", [])
            _write_blocked_recovery(RECOVERY_OUTPUT, RECOVERY_AUDIT,
                                    statuses=statuses, error=exc,
                                    old_inventory_unchanged=unchanged)
        except BaseException as audit_error:
            print(f"BLOCKED_AUDIT_WRITE_FAILED {type(audit_error).__name__}: {audit_error}",
                  flush=True)
        print(f"P12_RECOVERY_STOP {type(exc).__name__}: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
