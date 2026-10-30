"""Bounded R1 matched-trajectory Local Correction engineering pilot.

This module is intentionally isolated from historical scorer, writer,
controller, and experiment sources. It imports their native functions but
does not patch them. The pilot records engineering diagnostics only; it does
not evaluate anomaly-detection outcomes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
PARENT_COMMIT = "acc241299373d2f93d8c85ada8d2dbba2caf4f9f"
STAGE05 = ROOT / "outputs/calibration/stage05_exploratory_20261029"
DEFAULT_DATASET_ROOT = Path(r"G:\dataset\realiad")
DEFAULT_JSON_ROOT = DEFAULT_DATASET_ROOT / "jsons" / "realiad_jsons"
DEFAULT_OUTPUT = ROOT / "outputs/calibration/r1_switch_s0_512_engineering_20261031"
MAX_EVENTS = 512
MAX_OUTPUT_BYTES = 2 * 1024**3
MAX_GPU_SECONDS = 120 * 60
SMOKE_EVENTS = 16
QUERY_BINDING_SHA256 = "dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183"
QUERY_PROTOCOL_SHA256 = "6f1d8cf40e5e7b2d98dcfabe5836b49c6cd57069d3c8377122dba64e3fc2da1c"


class R1Error(RuntimeError):
    """Base error for a fail-closed R1 gate."""


class R1BlockedInput(R1Error):
    pass


class R1ParityError(R1Error):
    pass


class R1CausalityError(R1Error):
    pass


class R1ResourceLimit(R1Error):
    pass


class TapeIntegrityError(R1Error):
    pass


@dataclass(frozen=True)
class PilotConfig:
    category: str = "switch"
    support_seed: int = 0
    start_position: int = 0
    max_events: int = MAX_EVENTS
    query_rate: float = 0.03
    query_seed: int = 37
    device: str = "cuda"
    max_output_bytes: int = MAX_OUTPUT_BYTES
    max_gpu_seconds: int = MAX_GPU_SECONDS
    smoke_events: int = SMOKE_EVENTS
    c2_neighbors: int = 5
    c2_interface_scale: float = 0.20


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def opaque_sample_group(sample_key: str) -> str:
    return hashlib.sha256(
        ("official-sample-group\0" + str(sample_key)).encode("utf-8")).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to("cpu").contiguous()
    return sha256_bytes(str((tuple(value.shape), str(value.dtype))).encode("ascii") +
                        value.numpy().tobytes())


def arrays_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(value.shape), value.dtype.str)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    out = io.BytesIO()
    np.save(out, np.ascontiguousarray(array), allow_pickle=False)
    return out.getvalue()


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Stable, uncompressed NPZ bytes with fixed ZIP timestamps and key order."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(arrays[name]))
    return out.getvalue()


def corrected_dense_map(base_map: np.ndarray, selected_ids: Sequence[int],
                        selected_delta: np.ndarray) -> np.ndarray:
    """Apply a correction only at native selected IDs and return a new map."""
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


def source_disjoint_anchor_indices(anchor_positions: Sequence[int],
                                   anchor_groups: Sequence[str], *,
                                   event_position: int,
                                   event_group: str) -> list[int]:
    if len(anchor_positions) != len(anchor_groups):
        raise ValueError("Anchor positions and source groups differ in length")
    if not event_group:
        raise ValueError("Current official source group is missing")
    keep: list[int] = []
    for index, (position, group) in enumerate(zip(anchor_positions, anchor_groups)):
        if int(position) >= int(event_position):
            continue
        if str(group) != str(event_group):
            keep.append(index)
    return keep


def signed_knn_delta(query_rows: np.ndarray, anchors: Sequence[Mapping[str, Any]], *,
                     event_position: int, event_group: str,
                     neighbors: int = 5, amplitude: float = 0.20) -> tuple[np.ndarray, int]:
    """Deterministic unit-testable signed kNN interface; no fitting/tuning."""
    q = np.asarray(query_rows, dtype=np.float32)
    if q.ndim != 2 or not np.isfinite(q).all() or neighbors < 1:
        raise ValueError("Invalid query feature matrix or neighbor count")
    eligible = [a for a in anchors
                if int(a["position"]) < int(event_position) and
                str(a["group_sha256"]) != str(event_group)]
    if not eligible:
        return np.zeros((len(q),), dtype=np.float32), 0
    bank = np.stack([np.asarray(a["feature"], dtype=np.float32) for a in eligible])
    signs = np.asarray([int(a["sign"]) for a in eligible], dtype=np.float32)
    if bank.ndim != 2 or bank.shape[1] != q.shape[1] or not np.isfinite(bank).all():
        raise ValueError("Invalid signed kNN anchor features")
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    pn = bank / np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-12)
    distances = 1.0 - np.clip(qn @ pn.T, -1.0, 1.0)
    result = np.zeros((len(q),), dtype=np.float32)
    k = min(int(neighbors), len(eligible))
    anchor_order = np.arange(len(eligible), dtype=np.int64)
    for row in range(len(q)):
        nearest = np.lexsort((anchor_order, distances[row]))[:k]
        result[row] = np.float32(float(amplitude) * float(signs[nearest].mean()))
    return result, len(eligible)


class CausalEventOrder:
    """Runtime guard for score -> reveal -> correction -> common write order."""
    def __init__(self, position: int, queried: bool):
        self.position = int(position)
        self.queried = bool(queried)
        self.phase = "observed"

    def seal_scores(self) -> None:
        if self.phase != "observed":
            raise R1CausalityError("Scores were sealed in an invalid event phase")
        self.phase = "scores_sealed"

    def disclose(self, provider: Callable[[], int]) -> int:
        if self.phase != "scores_sealed" or not self.queried:
            raise R1CausalityError("Queried label requested before score seal or on an unqueried event")
        value = int(provider())
        if value not in (0, 1):
            raise R1CausalityError("Queried label provider returned a nonbinary value")
        self.phase = "label_disclosed"
        return value

    def corrections_updated(self) -> None:
        expected = "label_disclosed" if self.queried else "scores_sealed"
        if self.phase != expected:
            raise R1CausalityError("Correction update occurred before legal disclosure")
        self.phase = "corrections_updated"

    def memory_updated(self) -> None:
        if self.phase != "corrections_updated":
            raise R1CausalityError("Greedy write occurred before correction phase")
        self.phase = "memory_updated"

    def finish(self) -> None:
        if self.phase != "memory_updated":
            raise R1CausalityError("Event did not complete the required causal phases")
        self.phase = "complete"


def _contains_forbidden_label_payload(value: Any) -> bool:
    forbidden = {"label", "raw_label", "true_label", "anomaly_class", "is_good",
                 "ground_truth", "mask", "defect_mask"}
    if isinstance(value, Mapping):
        return any(str(key).lower() in forbidden or _contains_forbidden_label_payload(item)
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_label_payload(item) for item in value)
    return False


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    payload = (canonical_json(record) + "\n").encode("utf-8")
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _write_once(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(value) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as stream:
        temp_path = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def _directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


class EvidenceTapeWriter:
    """Append-only event shards; restart is refused without a full state checkpoint."""
    def __init__(self, root: Path, *, max_bytes: int = MAX_OUTPUT_BYTES,
                 resume_empty_bootstrap: bool = False):
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        self.next_sealed = 0
        self.next_finalized = 0
        if self.root.exists():
            if not resume_empty_bootstrap:
                raise FileExistsError(f"R1 output namespace already exists: {self.root}")
            entries = {path.name for path in self.root.iterdir()}
            manifest_path = self.root / "run_manifest.json"
            if entries != {"events", "run_manifest.json"} or not (self.root / "events").is_dir():
                raise TapeIntegrityError("Explicit resume found unexpected R1 output files")
            if any((self.root / "events").iterdir()):
                raise TapeIntegrityError("Explicit empty-bootstrap resume found event shards")
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (previous.get("schema") != "calibration.r1_evidence_tape.v1" or
                    previous.get("status") != "INCOMPLETE" or
                    previous.get("sealed_events") != 0 or
                    previous.get("finalized_events") != 0 or
                    previous.get("next_position") != 0 or
                    previous.get("failure_type") != "ModuleNotFoundError" or
                    "experiments" not in str(previous.get("failure_message", ""))):
                raise TapeIntegrityError("Existing R1 state is not the known empty import-bootstrap failure")
            history_path = self.root / "attempt_history.jsonl"
            _append_jsonl(history_path, {
                "attempt": 1,
                "outcome": "EMPTY_BOOTSTRAP_FAILURE_RETAINED",
                "failure_type": previous["failure_type"],
                "failure_message": previous["failure_message"],
                "sealed_events": 0,
                "finalized_events": 0,
            })
            self.manifest = dict(previous)
            for key in ("failure_type", "failure_message", "events_processed",
                        "parity_failures", "causal_state_mismatches"):
                self.manifest.pop(key, None)
            self.manifest.update({"status": "INCOMPLETE", "retry_count": 1,
                                  "resume_mode": "explicit_empty_bootstrap_only"})
            self.update_manifest({})
        else:
            self.root.mkdir(parents=True)
            (self.root / "events").mkdir()
            self.manifest: dict[str, Any] = {
                "schema": "calibration.r1_evidence_tape.v1",
                "status": "INCOMPLETE",
                "sealed_events": 0,
                "finalized_events": 0,
                "next_position": 0,
                "output_bytes": 0,
            }
            _write_once(self.root / "run_manifest.json",
                        (canonical_json(self.manifest) + "\n").encode("utf-8"))

    def update_manifest(self, updates: Mapping[str, Any]) -> None:
        self.manifest.update(dict(updates))
        self.manifest["output_bytes"] = _directory_bytes(self.root)
        if self.manifest["output_bytes"] > self.max_bytes:
            raise R1ResourceLimit("R1 output exceeded the 2 GiB authorized ceiling")
        _atomic_json(self.root / "run_manifest.json", self.manifest)

    def seal_event(self, position: int, arrays: Mapping[str, np.ndarray],
                   metadata: Mapping[str, Any]) -> dict[str, Any]:
        position = int(position)
        if position != self.next_sealed:
            raise TapeIntegrityError(
                f"Expected next sealed position {self.next_sealed}, received {position}")
        if _contains_forbidden_label_payload(metadata):
            raise TapeIntegrityError("Online score tape contains a label/mask payload")
        required = {"base_patch_map", "top32_ids", "top32_features",
                    "c1_delta", "c2_delta"}
        if set(arrays) != required:
            raise TapeIntegrityError("Event shard fields differ from the frozen tape schema")
        base = np.asarray(arrays["base_patch_map"], dtype=np.float32)
        ids = np.asarray(arrays["top32_ids"], dtype=np.int64)
        features = np.asarray(arrays["top32_features"], dtype=np.float32)
        d1 = np.asarray(arrays["c1_delta"], dtype=np.float32)
        d2 = np.asarray(arrays["c2_delta"], dtype=np.float32)
        if (base.ndim != 1 or len(base) != 784 or ids.shape != (32,) or
                features.ndim != 2 or features.shape[0] != 32 or
                d1.shape != (32,) or d2.shape != (32,)):
            raise TapeIntegrityError("Event shard has an unexpected tensor shape")
        if (np.any(ids < 0) or np.any(ids >= len(base)) or len(set(ids.tolist())) != 32 or
                not all(np.isfinite(x).all() for x in (base, features, d1, d2))):
            raise TapeIntegrityError("Event shard has invalid IDs or nonfinite values")
        expected_scores = {
            "c0_score": max_score(base),
            "c1_score": max_score(corrected_dense_map(base, ids, d1)),
            "c2_score": max_score(corrected_dense_map(base, ids, d2)),
        }
        for key, expected in expected_scores.items():
            if key not in metadata or float(metadata[key]) != expected:
                raise TapeIntegrityError(f"{key} does not reconstruct exactly from the dense map")
        array_payload = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
        shard_bytes = deterministic_npz_bytes(array_payload)
        shard_name = f"events/event_{position:06d}.npz"
        _write_once(self.root / shard_name, shard_bytes)
        record = dict(metadata)
        record.update({
            "position": position,
            "array_fields": sorted(array_payload),
            "array_sha256": arrays_sha256(array_payload),
            "shard_path": shard_name,
            "shard_sha256": sha256_bytes(shard_bytes),
        })
        if _contains_forbidden_label_payload(record):
            raise TapeIntegrityError("Sealed event metadata contains a label/mask payload")
        _append_jsonl(self.root / "sealed_scores.jsonl", record)
        self.next_sealed += 1
        self.update_manifest({"sealed_events": self.next_sealed,
                              "next_position": self.next_sealed})
        return record

    def finalize_event(self, position: int, metadata: Mapping[str, Any]) -> dict[str, Any]:
        position = int(position)
        if position >= self.next_sealed:
            raise TapeIntegrityError("Cannot finalize an event before its scores are sealed")
        if position != self.next_finalized:
            raise TapeIntegrityError(
                f"Expected next finalized position {self.next_finalized}, received {position}")
        if _contains_forbidden_label_payload(metadata):
            raise TapeIntegrityError("Evidence tape contains a label/mask payload")
        record = dict(metadata)
        record["position"] = position
        _append_jsonl(self.root / "event_transitions.jsonl", record)
        self.next_finalized += 1
        self.update_manifest({"finalized_events": self.next_finalized})
        return record

    def verify(self, *, require_complete: bool = True) -> dict[str, Any]:
        sealed_path = self.root / "sealed_scores.jsonl"
        final_path = self.root / "event_transitions.jsonl"
        sealed = ([json.loads(line) for line in sealed_path.read_text(encoding="utf-8").splitlines()]
                  if sealed_path.exists() else [])
        finalized = ([json.loads(line) for line in final_path.read_text(encoding="utf-8").splitlines()]
                     if final_path.exists() else [])
        sealed_positions = [int(row["position"]) for row in sealed]
        finalized_positions = [int(row["position"]) for row in finalized]
        if sealed_positions != list(range(len(sealed_positions))):
            raise TapeIntegrityError("Sealed event positions have a duplicate or gap")
        if finalized_positions != list(range(len(finalized_positions))):
            raise TapeIntegrityError("Final transition positions have a duplicate or gap")
        if len(finalized) > len(sealed):
            raise TapeIntegrityError("More transitions than sealed score events")
        if require_complete and len(sealed) != len(finalized):
            raise TapeIntegrityError("A sealed score event lacks a post-feedback transition")
        for row in sealed:
            shard = self.root / str(row["shard_path"])
            if not shard.is_file() or sha256_file(shard) != row["shard_sha256"]:
                raise TapeIntegrityError("Event shard missing or checksum mismatch")
            with np.load(shard, allow_pickle=False) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
            if arrays_sha256(arrays) != row["array_sha256"]:
                raise TapeIntegrityError("Event array checksum mismatch")
            ids = arrays["top32_ids"].astype(np.int64, copy=False)
            if (float(row["c0_score"]) != max_score(arrays["base_patch_map"]) or
                    float(row["c1_score"]) != max_score(corrected_dense_map(
                        arrays["base_patch_map"], ids, arrays["c1_delta"])) or
                    float(row["c2_score"]) != max_score(corrected_dense_map(
                        arrays["base_patch_map"], ids, arrays["c2_delta"]))):
                raise TapeIntegrityError("Deterministic score replay does not match sealed scores")
        return {"sealed_events": len(sealed), "finalized_events": len(finalized),
                "sealed_positions_sha256": sha256_json(sealed_positions),
                "transition_positions_sha256": sha256_json(finalized_positions),
                "passed": len(sealed) == len(finalized) if require_complete else True}


def _validate_branch_and_start() -> dict[str, str]:
    branch = subprocess.check_output(["git", "branch", "--show-current"],
                                     cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=ROOT, text=True).strip()
    if branch != EXPECTED_BRANCH or head != PARENT_COMMIT:
        raise R1BlockedInput(
            f"Expected {EXPECTED_BRANCH}@{PARENT_COMMIT}; found {branch}@{head}")
    return {"branch": branch, "parent_commit": head}


def _preflight(dataset_root: Path, json_root: Path) -> dict[str, Any]:
    git_identity = _validate_branch_and_start()
    from experiments.calibration.profiles import NativeScorerConfig, QueryProtocolConfig
    from experiments.calibration.stage05 import _absolute_image_path
    from experiments.calibration.stage1a_p11_production import verify_stage05_qats_artifacts

    scorer_profile = NativeScorerConfig()
    query_profile = QueryProtocolConfig()
    tapes = verify_stage05_qats_artifacts(STAGE05)
    if tapes.binding_sha256 != QUERY_BINDING_SHA256:
        raise R1BlockedInput("Frozen Q-A-TS binding differs from the authorization")
    binding = next((dict(row) for row in tapes.bindings
                    if row.get("class_identity") == "switch"), None)
    if not binding or binding.get("query_protocol_sha256") != QUERY_PROTOCOL_SHA256:
        raise R1BlockedInput("Switch Query Tape identity is missing or incompatible")

    frozen_path = STAGE05 / "frozen_inputs.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("test_labels_accessed") is not False or frozen.get("test_masks_accessed") is not False:
        raise R1BlockedInput("Frozen Stage 0.5 provenance reports prior test-label or mask access")
    info = frozen["classes"]["switch"]
    stream = info["stream"]
    if len(stream) < MAX_EVENTS or len(stream) != int(binding["n_images"]):
        raise R1BlockedInput("Frozen switch stream is shorter than the authorized prefix")
    if len(info["supports"]["0"]["images"]) != 4:
        raise R1BlockedInput("Frozen support seed 0 does not contain exactly four supports")
    if int(info["c64"]["count"]) != 64:
        raise R1BlockedInput("Frozen protected C64 manifest is not exactly 64 references")
    if not info.get("support_c64_disjoint") or not info.get("train_test_groups_disjoint"):
        raise R1BlockedInput("Frozen support/C64 group-disjointness receipt failed")

    prefix = stream[:MAX_EVENTS]
    for position, row in enumerate(prefix):
        if int(row.get("position", -1)) != position:
            raise R1BlockedInput("Frozen stream position sequence is not the authorized prefix")
        if not isinstance(row.get("timestamp"), str) or not row["timestamp"]:
            raise R1BlockedInput(f"Missing source timestamp at event position {position}")
        sample_key = str(row["sample_key"])
        group = opaque_sample_group(sample_key)
        expected_group = tapes.group_identity_by_class_position["switch"].get(position)
        if row.get("opaque_group") != group or (expected_group is not None and expected_group != group):
            raise R1BlockedInput(f"Frozen stream/query group mismatch at event {position}")

    query_positions = tuple(position for position in tapes.positions("switch")
                            if 0 <= int(position) < MAX_EVENTS)
    query_set = frozenset(int(position) for position in query_positions)
    if any(position not in range(MAX_EVENTS) for position in query_set):
        raise R1BlockedInput("Query position falls outside the authorized event prefix")
    group_map = {position: str(tapes.group_identity_by_class_position["switch"][position])
                 for position in query_set}
    if len(set(group_map.values())) != len(group_map):
        raise R1BlockedInput("Query Tape contains duplicate official source groups in the prefix")

    support_groups = set(opaque_sample_group(x) for x in info["supports"]["0"]["groups"])
    c64_groups = set(opaque_sample_group(x) for x in info["c64"]["groups"])
    stream_groups = set(str(row["opaque_group"]) for row in prefix)
    if (support_groups & stream_groups or c64_groups & stream_groups or
            support_groups & c64_groups):
        raise R1BlockedInput("Support/C64/query-stream source-group overlap detected")

    support_paths = [Path(_absolute_image_path(dataset_root, info["prefix"], str(relative)))
                     for relative in info["supports"]["0"]["images"]]
    path_records: dict[str, Path] = {}
    for relative, path in zip(info["supports"]["0"]["images"], support_paths):
        path_records["support:" + sha256_bytes(str(relative).encode())] = path
    for position, row in enumerate(prefix):
        path_records[f"stream:{position}"] = Path(
            _absolute_image_path(dataset_root, info["prefix"], str(row["relative_id"])))
    for role, path in path_records.items():
        if not path.is_file():
            raise R1BlockedInput(f"Authorized local image file is missing for {role}")
    label_json = json_root / "switch.json"
    if not label_json.is_file():
        raise R1BlockedInput("Authorized queried-position label source is missing")

    score_manifest = json.loads((STAGE05 / "score_tape_manifest.json").read_text(
        encoding="utf-8"))
    c64_row = score_manifest["classes"]["switch"]["score_tapes"]["0"]
    c64_path = STAGE05 / str(c64_row["file"])
    if (c64_row.get("c64_score_count") != 64 or
            not c64_row.get("native_parity", {}).get("passed") or
            sha256_file(c64_path) != c64_row["sha256"]):
        raise R1BlockedInput("Frozen C64 score receipt is incomplete or checksum-invalid")
    with np.load(c64_path, allow_pickle=False) as archive:
        c64_scores = archive["c64_scores"].astype(np.float64, copy=True)
        c64_identity = str(archive["c64_identity_sha256"].item())
    if (len(c64_scores) != 64 or not np.isfinite(c64_scores).all() or
            c64_identity != info["c64"]["image_sha256"]):
        raise R1BlockedInput("Frozen C64 score/identity count differs from the protected manifest")
    if c64_row["c64_group_sha256"] != info["c64"]["group_sha256"]:
        raise R1BlockedInput("Frozen C64 score receipt has a different group identity")
    from experiments.calibration.controllers import theta0_from_c64
    theta0 = float(theta0_from_c64(c64_scores.tolist()))
    sigma0 = max(float(c64_scores.std(ddof=1)), 0.02)

    return {
        "git": git_identity,
        "scorer_profile": asdict(scorer_profile),
        "scorer_profile_sha256": scorer_profile.sha256,
        "query_profile": asdict(query_profile),
        "query_profile_config_sha256": query_profile.sha256,
        "query_binding_sha256": tapes.binding_sha256,
        "query_verification_sha256": tapes.verification_sha256,
        "query_binding": binding,
        "query_positions": sorted(query_set),
        "query_positions_sha256": sha256_json(sorted(query_set)),
        "query_positions_prefix_count": len(query_set),
        "query_group_sha256_by_position": group_map,
        "frozen_inputs_sha256": sha256_file(frozen_path),
        "class_json_sha256": str(info["class_json_sha256"]),
        "stream_count": len(stream),
        "stream_prefix_group_count": len(stream_groups),
        "stream_prefix_timestamp_policy": "frozen event position is causal order; preserve and verify source timestamps, do not require monotonic timestamps",
        "support_paths": support_paths,
        "stream_metadata": prefix,
        "prefix": str(info["prefix"]),
        "support_group_sha256": sha256_json(sorted(support_groups)),
        "c64_group_sha256": str(info["c64"]["group_sha256"]),
        "c64_image_sha256": str(info["c64"]["image_sha256"]),
        "c64_score_tape_sha256": str(c64_row["sha256"]),
        "c64_score_identity_sha256": c64_identity,
        "c64_theta0_higher_q90": theta0,
        "c64_sigma0_with_floor_0_02": sigma0,
        "fixed_support_sha256_expected": str(c64_row["fixed_support_sha256"]),
        "dataset_root": str(dataset_root),
        "json_root": str(json_root),
        "label_json_path": label_json,
        "label_json_expected_sha256": str(info["class_json_sha256"]),
    }


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(value.shape), str(value.dtype))).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _hub_source_sha256(torch_hub_dir: Path) -> tuple[str, str]:
    candidates = sorted(path for path in torch_hub_dir.glob("facebookresearch_dinov2*")
                        if path.is_dir() and (path / "hubconf.py").is_file())
    if not candidates:
        raise R1BlockedInput("Cached DINOv2 torch.hub source is unavailable; network fetch is not authorized")
    source = candidates[0]
    entries = []
    for path in sorted(source.rglob("*.py")):
        entries.append((str(path.relative_to(source)).replace("\\", "/"), sha256_file(path)))
    return str(source), sha256_json(entries)


def _native_dense_score(impl: Any, bank: Any, image: Any,
                        features: Sequence[torch.Tensor], cfg: Any
                        ) -> tuple[np.ndarray, np.ndarray, float, Mapping[str, Any]]:
    import torch
    if bank.corr_cfg.enabled or bank.use_defect_bank:
        raise R1ParityError("Common C0 scorer requires correction and defect banks disabled")
    s_maps, summary = bank.score_single_view(
        cls_name="p0", feats_per_layer=list(features), mode=0, view="id",
        topk_patches=cfg.topk_patches, mode_scoring="boost", lam=0.8,
        tau_close=0.20, gamma=1.0)
    if len(s_maps) != 1:
        raise R1ParityError("Locked concat profile must expose exactly one fused feature layer")
    ids = np.asarray(summary["layer0"]["topk_patch_idx"], dtype=np.int64)
    base_map = s_maps[0].detach().float().to("cpu").contiguous().numpy().astype(
        np.float32, copy=True)
    if base_map.shape != (784,) or ids.shape != (32,):
        raise R1ParityError(f"Unexpected native dense map/top-k shapes: {base_map.shape}/{ids.shape}")
    if not np.isfinite(base_map).all() or len(set(ids.tolist())) != 32:
        raise R1ParityError("Native dense map is nonfinite or Top-32 IDs are invalid")
    aspect = float(image.height) / float(image.width) if image.width else None
    h, w = impl._grid_hw_from_npatches(784, aspect=aspect)
    fused = bank.fuse_layers(s_maps, fusion="mean")
    aligned = impl.align_map_to_identity(
        fused, impl.ViewSpec("id", "id"), int(h), int(w))
    score = float(bank.aggregate_map(aligned, agg=cfg.image_aggregation,
                                     topk=cfg.image_topk,
                                     quantile=cfg.image_quantile))
    if not math.isfinite(score):
        raise R1ParityError("Native adapter produced a nonfinite C0 score")
    if score != max_score(base_map):
        raise R1ParityError(
            "Native image-level max does not reconstruct exactly from the dense patch map")
    return base_map, ids, score, summary


def _native_reference_and_debug(impl: Any, bank: Any, image: Any,
                                features: Sequence[torch.Tensor], cfg: Any
                                ) -> tuple[float, np.ndarray, np.ndarray]:
    from experiments.a0_a1_strict4_matched import score_bank
    score, amap = score_bank(impl, bank, image, list(features), cfg)
    candidate_map = impl.layer_to_topk_from_amap(amap)
    ids = np.asarray(candidate_map.get(0, []), dtype=np.int64)
    debug = bank.score_sample_correction_debug_maps(
        cls_name="p0", view_to_feats={"id": list(features)}, mode=0,
        views={"id": impl.ViewSpec("id", "id")},
        topk_patches=cfg.topk_patches, mode_scoring="boost",
        lam=0.8, tau_close=0.20, gamma=1.0,
        layer_fusion="mean", view_fusion="mean",
        img_agg=cfg.image_aggregation, img_topk=cfg.image_topk,
        img_quantile=cfg.image_quantile, concept_assign_view="id",
        img_hw=(image.height, image.width))
    dense = np.asarray(debug["A_pre"], dtype=np.float32).reshape(-1)
    return float(score), ids, dense.copy()


def _memory_content_sha256(bank: Any) -> str:
    from experiments.risk_safe.native_bridge import snapshot
    return snapshot(bank, version=0).sha256()


def _memory_state_sha256(bank: Any, version: int) -> str:
    from experiments.risk_safe.native_bridge import snapshot
    return snapshot(bank, version=int(version)).sha256()


def _correction_state_sha256(bank: Any,
                             source_groups: Mapping[int, set[str]]) -> str:
    cb = bank.corr_cb["p0"]["id"][0]
    n = int(cb.size())
    digest = hashlib.sha256()
    for name in ("protos", "beta", "radius", "kappa", "count",
                 "outcome_fp", "outcome_fn", "outcome_tn", "outcome_tp", "last_step"):
        value = getattr(cb, name)[:n].detach().to("cpu").contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str((tuple(value.shape), str(value.dtype))).encode("ascii"))
        digest.update(value.numpy().tobytes())
    digest.update(canonical_json({str(k): sorted(v) for k, v in sorted(source_groups.items())}).encode())
    return digest.hexdigest()


def _filtered_c1_codebook(impl: Any, bank: Any, source_groups: Mapping[int, set[str]],
                          current_group: str) -> tuple[Any, list[int]]:
    import torch
    live = bank.corr_cb["p0"]["id"][0]
    n = int(live.size())
    missing = [slot for slot in range(n) if slot not in source_groups or not source_groups[slot]]
    if missing:
        raise R1CausalityError("Active C1 anchor lacks source-group provenance")
    eligible = [slot for slot in range(n) if current_group not in source_groups[slot]]
    if any(current_group in source_groups[slot] for slot in eligible):
        raise R1CausalityError("C1 source-group exclusion failed")
    cfg = bank.corr_cfg
    cfg_copy = type(cfg)(**asdict(cfg))
    cfg_copy.enabled = True
    filtered = impl.SignedCorrectionCodebook(
        max(1, len(eligible)), int(live.protos.shape[1]), live.device, cfg_copy)
    filtered.n = len(eligible)
    if eligible:
        slots = torch.as_tensor(eligible, device=live.device, dtype=torch.long)
        for name in ("protos", "beta", "radius", "kappa", "count",
                     "outcome_fp", "outcome_fn", "outcome_tn", "outcome_tp", "last_step"):
            getattr(filtered, name)[:len(eligible)].copy_(getattr(live, name)[slots])
    return filtered, eligible


def _c1_score_delta(impl: Any, bank: Any, selected_features: torch.Tensor,
                    source_groups: Mapping[int, set[str]], current_group: str
                    ) -> tuple[np.ndarray, dict[str, Any]]:
    filtered, eligible = _filtered_c1_codebook(impl, bank, source_groups, current_group)
    if not eligible:
        return np.zeros((32,), dtype=np.float32), {
            "eligible_anchor_slots": 0, "same_source_excluded_anchor_slots": int(bank.corr_cb["p0"]["id"][0].size()),
            "far_field_patch_count": 0,
            "far_field_abs_correction_max": 0.0,
        }
    q = impl.l2_normalize(selected_features.to(bank.device))
    delta = filtered.apply(q).detach().float().to("cpu").numpy().astype(np.float32, copy=True)
    dmat = filtered._cos_dist(q, filtered.protos[:filtered.n])
    nearest_dist, nearest_slot = dmat.min(dim=1)
    nearest_radius = filtered.radius[:filtered.n][nearest_slot].clamp_min(1e-6)
    far = nearest_dist > (3.0 * nearest_radius)
    far_values = np.abs(delta[far.detach().to("cpu").numpy()]) if bool(far.any()) else np.zeros(0)
    return delta, {
        "eligible_anchor_slots": len(eligible),
        "same_source_excluded_anchor_slots": int(bank.corr_cb["p0"]["id"][0].size()) - len(eligible),
        "far_field_patch_count": int(far.sum().item()),
        "far_field_abs_correction_max": float(far_values.max()) if len(far_values) else 0.0,
    }


def _c2_state_sha256(anchors: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json([
        {"position": int(anchor["position"]),
         "group_sha256": str(anchor["group_sha256"]),
         "patch_id": int(anchor["patch_id"]), "sign": int(anchor["sign"]),
         "feature_sha256": str(anchor["feature_sha256"])}
        for anchor in anchors])


def _source_hashes() -> dict[str, str]:
    files = [
        "test_phase3_AIF_simplified_v22.py", "vmb_visionad_new.py",
        "experiments/v02_memory.py", "experiments/risk_safe/native_bridge.py",
        "experiments/a0_a1_strict4_matched.py", "experiments/p0_protocol_audit.py",
        "experiments/calibration/profiles.py",
        "experiments/calibration/stage1a_formal_comparison.py",
        "experiments/calibration/stage05.py",
        "experiments/calibration/stage1a_p11_production.py",
        "experiments/calibration/stage1a_p12_label_provider.py",
    ]
    return {path: sha256_file(ROOT / path) for path in files}


def _process_rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _append_private_feedback(root: Path, record: Mapping[str, Any]) -> str:
    """Restricted untracked response ledger; never referenced in the online tape payload."""
    path = root / "private_feedback.jsonl"
    payload = (canonical_json(record) + "\n").encode("utf-8")
    _append_jsonl(path, record)
    return sha256_bytes(payload)


def run_pilot(*, dataset_root: Path = DEFAULT_DATASET_ROOT,
              json_root: Path = DEFAULT_JSON_ROOT,
              output: Path = DEFAULT_OUTPUT,
              resume_empty_bootstrap: bool = False) -> dict[str, Any]:
    config = PilotConfig()
    if output.exists() and not resume_empty_bootstrap:
        raise FileExistsError(f"R1 output namespace already exists: {output}")
    tape = EvidenceTapeWriter(output, max_bytes=config.max_output_bytes,
                              resume_empty_bootstrap=resume_empty_bootstrap)
    preflight: dict[str, Any] = {}
    started = time.monotonic()
    pilot_execution_started: float | None = None
    parity_count = 0
    query_count = 0
    query_labels_decoded = 0
    positive_anchor_writes = 0
    negative_anchor_writes = 0
    c1_active_positive = c1_active_negative = 0
    c2_active_anchors = 0
    normal_memory_writes = 0
    causal_mismatches = 0
    c1_far_field_count = 0
    c1_far_field_abs_max = 0.0
    source_exclusion_checks = 0
    encode_seconds = native_seconds = correction_score_seconds = 0.0
    correction_update_seconds = writer_seconds = 0.0
    peak_rss = 0
    peak_cuda_allocated = peak_cuda_reserved = 0
    c1_source_groups: dict[int, set[str]] = {}
    c2_anchors: list[dict[str, Any]] = []
    memory_version = 0
    response_chain = "0" * 64
    try:
        import torch
        from experiments import p0_protocol_audit as p0
        from experiments.calibration import stage05
        from experiments.calibration.profiles import NativeScorerConfig
        from experiments.calibration.stage1a_p12_label_provider import QueriedPositionLabelReader
        from experiments.risk_safe.native_bridge import dynamic_codebooks
        from v02_memory import MemoryArm

        preflight = _preflight(dataset_root, json_root)
        torch_hub_dir = Path(torch.hub.get_dir())
        hub_source_path, hub_source_sha256 = _hub_source_sha256(torch_hub_dir)
        if not torch.cuda.is_available():
            raise R1BlockedInput("Authorized CUDA device is unavailable")
        torch.cuda.reset_peak_memory_stats()
        impl = p0.require_sources()
        scorer_profile = NativeScorerConfig()
        cfg = p0.Config(
            device=config.device, augmentation_seed=0, bank_seed=0,
            fixed_capacity=25000, fixed_per_image=5000, augmentations=4,
            include_identity=False, multiplier=2.5, feature_fuse="concat",
            topk_patches=32, image_aggregation="max", image_topk=64,
            image_quantile=0.95, knn=5)

        torch.cuda.synchronize()
        dino_started = time.perf_counter()
        pilot_execution_started = time.monotonic()
        backbone = impl.DINOv2MultiLayerBackbone(device=config.device)
        weight_state_sha256 = _model_state_sha256(backbone.model)
        source_hashes = _source_hashes()
        encoder_fingerprint = sha256_json({
            "model_name": scorer_profile.model_name,
            "loaded_model_state_sha256": weight_state_sha256,
            "hub_source_path": Path(hub_source_path).name,
            "hub_source_tree_sha256": hub_source_sha256,
            "vmb_source_sha256": source_hashes["vmb_visionad_new.py"],
            "v22_encoder_source_sha256": source_hashes["test_phase3_AIF_simplified_v22.py"],
            "hook_layers": list(backbone.layers),
        })
        preprocess_fingerprint = sha256_json({
            "profile": asdict(scorer_profile), "rgb": "PIL.convert(RGB)",
            "resize": [448, 448], "crop": [392, 392],
            "interpolation": "PIL.BICUBIC", "normalization": "model mean/std",
            "fusion": "concat_then_l2_normalize", "view": "id",
            "pillow_version": __import__("PIL").__version__,
        })
        support_features = p0.encode_reference_features(
            impl, backbone, [str(path) for path in preflight["support_paths"]], cfg)
        chosen = [str(path) for path in preflight["support_paths"]]
        bank = p0.build_vmb(impl, support_features, chosen, cfg)
        fixed_profile = p0.fixed_bank_profile(bank)
        if fixed_profile["sha256"] != preflight["fixed_support_sha256_expected"]:
            raise R1BlockedInput("Rebuilt fixed support bank differs from frozen C64 score receipt")
        settings = SimpleNamespace(
            flat=256, stm=256, ltm=1, buffer=1, min_events=2,
            promote_every=16, cap_bytes=config.max_output_bytes)
        writer = MemoryArm("greedy_coreset", bank, settings, seed=0)
        if any(int(cb.size()) for cb in dynamic_codebooks(bank)):
            raise R1BlockedInput("Flat Greedy dynamic memory did not start empty")
        if any(int(cb.K) != 256 for cb in dynamic_codebooks(bank)):
            raise R1BlockedInput("Flat Greedy capacity differs from locked K=256")
        bank.corr_cfg.enabled = False
        bank.use_defect_bank = False
        c1_cb = bank.corr_cb["p0"]["id"][0]
        if c1_cb.size() != 0:
            raise R1BlockedInput("C1 correction state was not empty at the pilot start")
        if c1_cb.K != 256:
            raise R1BlockedInput("C1 correction capacity differs from the existing bounded default")
        torch.cuda.synchronize()
        support_and_model_seconds = time.perf_counter() - dino_started
        memory_content_digest = _memory_content_sha256(bank)
        memory_state_digest = _memory_state_sha256(bank, memory_version)
        fixed_digest = str(fixed_profile["sha256"])
        feature_dims = int(support_features[chosen[0]][0][0].shape[-1])
        feature_grid = int(support_features[chosen[0]][0][0].shape[0])

        tape.update_manifest({
            "task": "R1", "authorization_scope": {
                "category": "switch", "support_seed": 0,
                "positions_inclusive": [0, 511], "max_events": 512,
                "query_protocol": "Q-A-TS-v1", "query_seed": 37,
                "query_rate": 0.03, "gpu_seconds_limit": MAX_GPU_SECONDS,
                "output_bytes_limit": MAX_OUTPUT_BYTES,
                "scientific_outcome_evaluation": False,
            },
            "branch": preflight["git"]["branch"],
            "parent_commit": preflight["git"]["parent_commit"],
            "source_hashes": source_hashes,
            "scorer_profile_sha256": preflight["scorer_profile_sha256"],
            "query_binding_sha256": preflight["query_binding_sha256"],
            "query_verification_sha256": preflight["query_verification_sha256"],
            "query_positions_sha256": preflight["query_positions_sha256"],
            "query_positions_in_prefix": preflight["query_positions_prefix_count"],
            "frozen_inputs_sha256": preflight["frozen_inputs_sha256"],
            "class_json_sha256": preflight["class_json_sha256"],
            "support_group_sha256": preflight["support_group_sha256"],
            "c64_group_sha256": preflight["c64_group_sha256"],
            "c64_image_sha256": preflight["c64_image_sha256"],
            "c64_score_tape_sha256": preflight["c64_score_tape_sha256"],
            "c64_theta0_higher_q90": preflight["c64_theta0_higher_q90"],
            "c64_sigma0_with_floor_0_02": preflight["c64_sigma0_with_floor_0_02"],
            "fixed_support_sha256": fixed_digest,
            "encoder_fingerprint_sha256": encoder_fingerprint,
            "loaded_model_state_sha256": weight_state_sha256,
            "hub_source_tree_sha256": hub_source_sha256,
            "preprocess_fingerprint_sha256": preprocess_fingerprint,
            "feature_tensor_expected_shape": [feature_grid, feature_dims],
            "feature_tensor_expected_dtype": "float32",
            "fixed_theta": preflight["c64_theta0_higher_q90"],
            "writer_sigma": preflight["c64_sigma0_with_floor_0_02"],
            "method_identities": {
                "C0": {"name": "native baseline, no correction",
                       "score_source": "shared native dense patch map"},
                "C1": {"name": "C1-Greedy-adapted",
                       "historical_v22_faithful_replay": False,
                       "write_source": "existing native SignedCorrectionCodebook and update_corrections_from_topk",
                       "strength_scale": 1.0,
                       "outcome_reference": "common C0 score/prediction/fixed C64 threshold",
                       "source_group_exclusion": "filter active anchors by queried source-group provenance",
                       "reasoningbank_dependent_route": False},
                "C2": {"name": "signed cosine kNN interface fixture",
                       "neighbors": config.c2_neighbors,
                       "amplitude": config.c2_interface_scale,
                       "distance": "unit-normalized cosine distance; deterministic index tie break",
                       "capacity": 256, "retention": "FIFO",
                       "scientific_hyperparameters_approved": False,
                       "future_preregistration_required": [
                           "neighbor count", "correction amplitude", "distance normalization/locality",
                           "anchor capacity/retention", "equal supervision budget",
                           "simple baseline identity"]},
            },
            "output_payloads_include_private_feedback": True,
            "unqueried_labels_accessed": False,
            "masks_accessed": False,
            "confirmation_data_accessed": False,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        label_reader = QueriedPositionLabelReader(
            category="switch", path=preflight["label_json_path"],
            expected_source_sha256=preflight["label_json_expected_sha256"],
            positions=preflight["query_positions"],
            group_sha256_by_position=preflight["query_group_sha256_by_position"],
        )
        records = preflight["stream_metadata"]
        support_and_model_elapsed = support_and_model_seconds
        started = time.monotonic()
        tape.update_manifest({"status": "INCOMPLETE",
                              "support_and_model_seconds": support_and_model_elapsed})

        for position, meta in enumerate(records):
            elapsed = (time.monotonic() - pilot_execution_started
                       if pilot_execution_started is not None else time.monotonic() - started)
            if elapsed >= config.max_gpu_seconds:
                raise R1ResourceLimit("120-minute GPU execution wall-time ceiling reached")
            group_sha = str(meta["opaque_group"])
            image_path = Path(stage05._absolute_image_path(
                dataset_root, preflight["prefix"], str(meta["relative_id"])))
            image_sha = sha256_file(image_path)
            encode_started = time.perf_counter()
            torch.cuda.synchronize()
            image, features = p0.encode_query(impl, backbone, str(image_path), cfg)
            torch.cuda.synchronize()
            encode_seconds += time.perf_counter() - encode_started
            if len(features) != 1 or tuple(features[0].shape) != (784, 4096):
                raise R1ParityError(f"Unexpected feature shape at event {position}: "
                                    f"{[tuple(x.shape) for x in features]}")
            if features[0].dtype != torch.float32 or not torch.isfinite(features[0]).all().item():
                raise R1ParityError("Feature dtype or finiteness differs from the locked profile")
            feature_digest = tensor_sha256(features[0])
            feature_object_id = id(features[0])
            if position == 0:
                tape.update_manifest({
                    "feature_tensor_actual_shape": list(features[0].shape),
                    "feature_tensor_actual_dtype": str(features[0].dtype),
                })
            feature_key = sha256_json({
                "image_sha256": image_sha, "encoder_fingerprint_sha256": encoder_fingerprint,
                "preprocess_fingerprint_sha256": preprocess_fingerprint,
                "view": "id", "feature_shape": list(features[0].shape),
                "feature_dtype": str(features[0].dtype), "feature_sha256": feature_digest,
            })

            native_before = _memory_content_sha256(bank)
            native_started = time.perf_counter()
            base_map, top_ids, c0_score, summary = _native_dense_score(
                impl, bank, image, features, cfg)
            torch.cuda.synchronize()
            native_seconds += time.perf_counter() - native_started
            if _memory_content_sha256(bank) != native_before:
                raise R1CausalityError("Native dense-map extraction mutated Greedy memory")
            if not np.array_equal(top_ids, np.asarray(
                    summary["layer0"]["topk_patch_idx"], dtype=np.int64)):
                raise R1ParityError("Native Top-32 identity differs inside the adapter")

            if position < config.smoke_events:
                before_parity = _memory_content_sha256(bank)
                fixed_before_parity = p0.fixed_bank_profile(bank)["sha256"]
                torch.cuda.synchronize()
                ref_started = time.perf_counter()
                reference_score, reference_ids, reference_map = _native_reference_and_debug(
                    impl, bank, image, features, cfg)
                torch.cuda.synchronize()
                native_seconds += time.perf_counter() - ref_started
                if _memory_content_sha256(bank) != before_parity:
                    raise R1CausalityError("Native reference/debug score mutated Greedy memory")
                if p0.fixed_bank_profile(bank)["sha256"] != fixed_before_parity or \
                        fixed_before_parity != fixed_digest:
                    raise R1CausalityError("Native parity replay mutated frozen fixed memory")
                if id(features[0]) != feature_object_id or tensor_sha256(features[0]) != feature_digest:
                    raise R1ParityError("Native parity replay changed the shared feature tensor")
                tolerance = 1e-5 + 1e-4 * abs(reference_score)
                if not np.array_equal(reference_ids, top_ids):
                    raise R1ParityError(f"Top-32 IDs differ at event {position}")
                if not np.array_equal(reference_map, base_map):
                    raise R1ParityError(f"Dense native map differs at event {position}")
                if abs(reference_score - c0_score) > tolerance:
                    raise R1ParityError(f"Native C0 score exceeds bridge tolerance at event {position}")
                parity_count += 1

            correction_score_started = time.perf_counter()
            normalized_selected = impl.l2_normalize(
                features[0].to(bank.device))[torch.as_tensor(
                    top_ids, device=bank.device, dtype=torch.long)]
            c1_delta, c1_diag = _c1_score_delta(
                impl, bank, normalized_selected, c1_source_groups, group_sha)
            c1_far_field_count += int(c1_diag["far_field_patch_count"])
            c1_far_field_abs_max = max(
                c1_far_field_abs_max, float(c1_diag["far_field_abs_correction_max"]))
            c2_delta, c2_eligible = signed_knn_delta(
                normalized_selected.detach().float().to("cpu").numpy(), c2_anchors,
                event_position=position, event_group=group_sha,
                neighbors=config.c2_neighbors, amplitude=config.c2_interface_scale)
            c2_expected_indices = source_disjoint_anchor_indices(
                [int(anchor["position"]) for anchor in c2_anchors],
                [str(anchor["group_sha256"]) for anchor in c2_anchors],
                event_position=position, event_group=group_sha)
            if len(c2_expected_indices) != c2_eligible:
                raise R1CausalityError("C2 source-group or future-anchor exclusion failed")
            source_exclusion_checks += 1
            correction_score_seconds += time.perf_counter() - correction_score_started
            corrected1 = corrected_dense_map(base_map, top_ids, c1_delta)
            corrected2 = corrected_dense_map(base_map, top_ids, c2_delta)
            c1_score, c2_score = max_score(corrected1), max_score(corrected2)
            if not (math.isfinite(c1_score) and math.isfinite(c2_score)):
                raise R1ParityError("Correction replay produced nonfinite image scores")
            theta = float(preflight["c64_theta0_higher_q90"])
            preds = {"c0": int(c0_score >= theta),
                     "c1": int(c1_score >= theta),
                     "c2": int(c2_score >= theta)}
            queried = position in preflight["query_positions"]
            memory_before_version = memory_version
            memory_before_digest = memory_state_digest
            c1_before_digest = _correction_state_sha256(bank, c1_source_groups)
            c2_before_digest = _c2_state_sha256(c2_anchors)
            score_state_digest = sha256_json({
                "fixed_support_sha256": fixed_digest,
                "greedy_memory_version": memory_before_version,
                "greedy_memory_sha256": memory_before_digest,
                "scorer_profile_sha256": preflight["scorer_profile_sha256"],
            })
            event_id = sha256_json({"category": "switch", "position": position,
                                    "image_sha256": image_sha, "group_sha256": group_sha})
            arrays = {
                "base_patch_map": base_map.astype(np.float32, copy=False),
                "top32_ids": top_ids.astype(np.int64, copy=False),
                "top32_features": normalized_selected.detach().float().to("cpu").contiguous().numpy()
                .astype(np.float32, copy=True),
                "c1_delta": c1_delta.astype(np.float32, copy=False),
                "c2_delta": c2_delta.astype(np.float32, copy=False),
            }
            if arrays["top32_features"].shape != (32, 4096):
                raise R1ParityError("Stored selected feature rows have unexpected shape")
            metadata = {
                "event_id": event_id, "category": "switch", "support_seed": 0,
                "timestamp": str(meta["timestamp"]),
                "source_image_identity_sha256": sha256_bytes(str(meta["relative_id"]).encode()),
                "source_image_content_sha256": image_sha,
                "official_group_sha256": group_sha,
                "camera_id": int(meta["camera_id"]),
                "feature_key_sha256": feature_key,
                "feature_tensor_sha256": feature_digest,
                "encoder_fingerprint_sha256": encoder_fingerprint,
                "preprocess_fingerprint_sha256": preprocess_fingerprint,
                "greedy_memory_version_before": memory_before_version,
                "greedy_memory_sha256_before": memory_before_digest,
                "fixed_support_sha256": fixed_digest,
                "score_state_sha256": score_state_digest,
                "c0_score": c0_score, "c1_score": c1_score, "c2_score": c2_score,
                "c0_prediction": preds["c0"], "c1_prediction": preds["c1"],
                "c2_prediction": preds["c2"],
                "query_decision": bool(queried),
                "query_reference": (f"switch:{position}" if queried else None),
                "c1_state_sha256_before": c1_before_digest,
                "c2_state_sha256_before": c2_before_digest,
                "c1_eligible_anchor_slots": int(c1_diag["eligible_anchor_slots"]),
                "c1_same_source_excluded_anchor_slots": int(c1_diag["same_source_excluded_anchor_slots"]),
                "c2_eligible_anchor_count": int(c2_eligible),
                "c1_far_field_patch_count": int(c1_diag["far_field_patch_count"]),
                "c1_far_field_abs_correction_max": float(c1_diag["far_field_abs_correction_max"]),
            }
            gate = CausalEventOrder(position, queried)
            gate.seal_scores()
            sealed_record = tape.seal_event(position, arrays, metadata)

            label: int | None = None
            feedback_ref: str | None = None
            correction_write_summary: dict[str, Any] = {"c1": None, "c2_added": 0}
            if queried:
                query_count += 1
                label = gate.disclose(lambda: label_reader(
                    "switch", position, str(meta["sample_key"])))
                query_labels_decoded += 1
                feedback_payload = {
                    "schema": "calibration.r1_private_feedback.v1",
                    "position": position, "event_id": event_id,
                    "source_image_identity_sha256": metadata["source_image_identity_sha256"],
                    "official_group_sha256": group_sha,
                    "disclosure_after_score": True,
                    "disclosure_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "label": int(label),
                }
                feedback_sha = _append_private_feedback(output, feedback_payload)
                feedback_ref = f"private_feedback.jsonl:{position}:{feedback_sha}"
                response_chain = sha256_bytes(
                    bytes.fromhex(response_chain) +
                    canonical_json({"position": position, "group_sha256": group_sha,
                                    "label": int(label)}).encode("utf-8"))
                pred0 = preds["c0"]
                outcome = ("TN" if label == 0 and pred0 == 0 else
                           "FP" if label == 0 and pred0 == 1 else
                           "FN" if label == 1 and pred0 == 0 else "TP")
                sign = 1 if label == 1 else -1
                c1_update_started = time.perf_counter()
                old_n = int(c1_cb.size())
                bank.corr_cfg.enabled = True
                try:
                    c1_result = bank.update_corrections_from_topk(
                        cls_name="p0", feats_per_layer=list(features), mode=0, view="id",
                        layer_to_patch_idx={0: top_ids.tolist()}, sign=sign,
                        score_img=float(c0_score), theta=theta, strength_scale=1.0,
                        outcome=outcome, step=position)
                finally:
                    bank.corr_cfg.enabled = False
                c1_layer = c1_result.get("layers", {}).get("0", {})
                new_n = int(c1_cb.size())
                for slot in range(new_n):
                    if int(c1_cb.last_step[slot].item()) != position:
                        continue
                    current_count = int(c1_cb.count[slot].item())
                    # Union even when native replacement metadata is ambiguous:
                    # this may over-exclude a source, but cannot leak it back in.
                    c1_source_groups.setdefault(slot, set()).add(group_sha)
                for slot in list(c1_source_groups):
                    if slot >= new_n:
                        del c1_source_groups[slot]
                kept = int(c1_layer.get("kept", 0))
                admitted = int(c1_layer.get("inserted", 0)) + int(c1_layer.get("merged", 0)) + int(c1_layer.get("replaced", 0))
                if sign > 0:
                    positive_anchor_writes += admitted
                else:
                    negative_anchor_writes += admitted
                c1_active_positive = int((c1_cb.beta[:new_n] > 0).sum().item())
                c1_active_negative = int((c1_cb.beta[:new_n] < 0).sum().item())
                c2_added = 0
                for patch_row, patch_id in enumerate(top_ids.tolist()):
                    feature_row = normalized_selected[patch_row].detach().float().to("cpu").numpy().copy()
                    feature_sha = arrays_sha256({"feature": feature_row})
                    c2_anchors.append({
                        "position": position, "group_sha256": group_sha,
                        "patch_id": int(patch_id), "sign": sign,
                        "feature": feature_row, "feature_sha256": feature_sha,
                    })
                    c2_added += 1
                if len(c2_anchors) > 256:
                    del c2_anchors[:len(c2_anchors) - 256]
                c2_active_anchors = len(c2_anchors)
                correction_update_seconds += time.perf_counter() - c1_update_started
                _append_jsonl(output / "correction_anchor_events.jsonl", {
                    "position": position, "event_id": event_id,
                    "feedback_ref": feedback_ref, "source_group_sha256": group_sha,
                    "c0_prediction": pred0, "c0_score": c0_score,
                    "c0_threshold": theta, "outcome": outcome, "sign": sign,
                    "c1_kept_by_native_gate": kept,
                    "c1_stats": {k: int(v) for k, v in c1_layer.items()
                                 if isinstance(v, (int, np.integer))},
                    "c2_added_patch_rows": c2_added,
                    "c2_retention": "FIFO, max 256 patch anchors",
                })
                correction_write_summary = {
                    "c1": {k: int(v) for k, v in c1_layer.items()
                           if isinstance(v, (int, np.integer))},
                    "c2_added": c2_added,
                }
            gate.corrections_updated()

            writer_action: dict[str, Any] = {
                "input_score_source": "common_C0",
                "score": float(c0_score), "theta": theta,
                "sigma": float(preflight["c64_sigma0_with_floor_0_02"]),
                "candidate_ids_sha256": sha256_json(top_ids.tolist()),
                "action": "no_normal_memory_write",
                "memory_version_after": memory_version,
            }
            writer_started = time.perf_counter()
            memory_content_before_write = _memory_content_sha256(bank)
            if queried and label == 0:
                allow, keep, reason, _info = impl.select_informative_normal_write(
                    score=float(c0_score), theta=theta,
                    sigmaN=float(max(preflight["c64_sigma0_with_floor_0_02"], 0.02)),
                    pred_defect=bool(c0_score >= theta), novelty=0.0)
                if allow:
                    audit_before = writer.audit()
                    writer.update(impl, list(features), {0: top_ids.tolist()}, position,
                                  score=float(c0_score), theta=theta,
                                  sigma=float(preflight["c64_sigma0_with_floor_0_02"]))
                    audit_after = writer.audit()
                    writer_action.update({
                        "action": "native_greedy_normal_update",
                        "eligibility_reason": str(reason),
                        "keep_count": int(keep),
                        "accepted_patches_delta": int(audit_after["accepted_patches"] - audit_before["accepted_patches"]),
                        "skipped_patches_delta": int(audit_after["skipped_patches"] - audit_before["skipped_patches"]),
                    })
                    normal_memory_writes += 1
                else:
                    writer_action.update({"action": "native_gate_no_write",
                                          "eligibility_reason": str(reason),
                                          "keep_count": int(keep)})
            writer_seconds += time.perf_counter() - writer_started
            memory_content_after_write = _memory_content_sha256(bank)
            if memory_content_after_write != memory_content_before_write:
                memory_version += 1
                memory_content_digest = memory_content_after_write
                memory_state_digest = _memory_state_sha256(bank, memory_version)
            writer_action["memory_version_after"] = memory_version
            writer_action["memory_digest_after"] = memory_state_digest
            gate.memory_updated()
            gate.finish()

            c1_after_digest = _correction_state_sha256(bank, c1_source_groups)
            c2_after_digest = _c2_state_sha256(c2_anchors)
            if queried:
                c1_group_positions = [a["position"] for a in c2_anchors]
                c1_group_values = [a["group_sha256"] for a in c2_anchors]
                if any(pos >= position for pos in c1_group_positions):
                    causal_mismatches += 1
                    raise R1CausalityError("C2 anchor bank contains a current/future event")
                _ = source_disjoint_anchor_indices(
                    c1_group_positions, c1_group_values,
                    event_position=position, event_group=group_sha)

            peak_rss = max(peak_rss, _process_rss_bytes() or 0)
            if torch.cuda.is_available():
                peak_cuda_allocated = max(peak_cuda_allocated,
                                          int(torch.cuda.max_memory_allocated()))
                peak_cuda_reserved = max(peak_cuda_reserved,
                                         int(torch.cuda.max_memory_reserved()))
            transition = {
                "writer_inputs_sha256": sha256_json({
                    "score": c0_score, "theta": theta,
                    "sigma": preflight["c64_sigma0_with_floor_0_02"],
                    "prediction": preds["c0"], "top32_ids": top_ids.tolist(),
                    "queried": queried,
                }),
                "writer_action": writer_action,
                "feedback_reference": feedback_ref,
                "greedy_memory_version_after": memory_version,
                "greedy_memory_sha256_after": memory_state_digest,
                "greedy_memory_content_sha256_after": memory_content_digest,
                "c1_state_sha256_after": c1_after_digest,
                "c2_state_sha256_after": c2_after_digest,
                "causal_order_complete": True,
            }
            tape.finalize_event(position, transition)
            if queried and label_reader.decoded_positions[-1] != position:
                causal_mismatches += 1
                raise R1CausalityError("Queried label reader disclosed a different position")
            if tape.manifest["output_bytes"] > config.max_output_bytes:
                raise R1ResourceLimit("R1 output exceeded the authorized storage ceiling")
            if (position + 1) % 32 == 0:
                print(f"R1_PROGRESS events={position + 1} queries={query_count} "
                      f"gpu_wall_seconds={elapsed:.1f} output_bytes={_directory_bytes(output)}",
                      flush=True)

        if query_labels_decoded != len(preflight["query_positions"]):
            raise R1CausalityError("Queried label count differs from the frozen prefix tape")
        if list(label_reader.decoded_positions) != list(preflight["query_positions"]):
            raise R1CausalityError("Queried label disclosure order differs from the fixed tape")
        final_fixed = p0.fixed_bank_profile(bank)
        if final_fixed["sha256"] != fixed_digest:
            raise R1CausalityError("Shared Greedy writer mutated the frozen fixed support bank")
        readback = tape.verify(require_complete=True)
        observed_anchors_both_signs = c1_active_positive > 0 and c1_active_negative > 0
        status = "ENGINEERING_PASS" if observed_anchors_both_signs else "INSUFFICIENT_ANCHOR_EVIDENCE"
        final_summary = {
            "status": status, "events_processed": tape.next_finalized,
            "unique_stream_positions": tape.next_finalized,
            "smoke_parity_events": parity_count,
            "query_events": query_count,
            "queried_labels_decoded": query_labels_decoded,
            "query_positions_decoded_sha256": sha256_json(label_reader.decoded_positions),
            "c1_positive_anchor_writes": positive_anchor_writes,
            "c1_negative_anchor_writes": negative_anchor_writes,
            "c1_active_positive_anchors": c1_active_positive,
            "c1_active_negative_anchors": c1_active_negative,
            "c2_active_anchors": c2_active_anchors,
            "c1_far_field_selected_patch_observations": c1_far_field_count,
            "c1_far_field_abs_correction_max": c1_far_field_abs_max,
            "source_group_exclusion_checks": source_exclusion_checks,
            "source_group_exclusion_checks_passed": source_exclusion_checks == tape.next_sealed,
            "normal_memory_write_events": normal_memory_writes,
            "greedy_memory_version_final": memory_version,
            "greedy_memory_sha256_final": memory_state_digest,
            "fixed_support_sha256_final": final_fixed["sha256"],
            "causal_state_mismatches": causal_mismatches,
            "parity_failures": 0,
            "evidence_tape_readback": readback,
            "encoding_seconds": encode_seconds,
            "native_scoring_seconds": native_seconds,
            "support_and_model_seconds": support_and_model_elapsed,
            "correction_scoring_seconds": correction_score_seconds,
            "correction_update_seconds": correction_update_seconds,
            "greedy_writer_seconds": writer_seconds,
            "gpu_wall_seconds": (time.monotonic() - pilot_execution_started
                                 if pilot_execution_started is not None else 0.0),
            "peak_cpu_rss_bytes": peak_rss or None,
            "peak_cuda_allocated_bytes": peak_cuda_allocated,
            "peak_cuda_reserved_bytes": peak_cuda_reserved,
            "output_bytes": _directory_bytes(output),
            "response_chain_sha256": response_chain,
            "scientific_outcome_evaluation": False,
            "auroc_computed": False, "auprc_computed": False,
            "classification_metrics_computed": False,
            "masks_accessed": False, "unqueried_labels_accessed": False,
            "confirmation_data_accessed": False,
        }
        tape.update_manifest(final_summary)
        tape.manifest["status"] = status
        tape.update_manifest({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        label_reader.close()
        backbone.cleanup()
        return final_summary
    except Exception as exc:
        if output.exists():
            if isinstance(exc, R1ParityError):
                status = "PARITY_FAILED"
            elif isinstance(exc, R1CausalityError):
                status = "CAUSALITY_FAILED"
            elif isinstance(exc, R1ResourceLimit):
                status = "RESOURCE_LIMIT_REACHED"
            elif isinstance(exc, R1BlockedInput):
                status = "BLOCKED_INPUT"
            else:
                status = "INCOMPLETE"
            try:
                current = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
                current.update({"status": status, "failure_type": type(exc).__name__,
                                "failure_message": str(exc)[:1000],
                                "events_processed": current.get("finalized_events", 0),
                                "query_events": query_count,
                                "queried_labels_decoded": query_labels_decoded,
                                "parity_failures": int(isinstance(exc, R1ParityError)),
                                "causal_state_mismatches": causal_mismatches,
                                "gpu_wall_seconds": (time.monotonic() - pilot_execution_started
                                                     if pilot_execution_started is not None else 0.0),
                                "peak_cpu_rss_bytes": peak_rss or None,
                                "peak_cuda_allocated_bytes": peak_cuda_allocated,
                                "peak_cuda_reserved_bytes": peak_cuda_reserved,
                                "output_bytes": _directory_bytes(output)})
                _atomic_json(output / "run_manifest.json", current)
            except Exception:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--json-root", type=Path, default=DEFAULT_JSON_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-empty-bootstrap-failure", action="store_true",
                        help="Explicitly continue only the known empty import-bootstrap failure")
    args = parser.parse_args(argv)
    try:
        result = run_pilot(dataset_root=args.dataset_root, json_root=args.json_root,
                           output=args.output,
                           resume_empty_bootstrap=args.resume_empty_bootstrap_failure)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0 if result["status"] in {"ENGINEERING_PASS", "INSUFFICIENT_ANCHOR_EVIDENCE"} else 1
    except Exception as exc:
        print(f"R1_PILOT_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
