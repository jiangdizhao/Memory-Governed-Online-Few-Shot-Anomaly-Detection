"""Durable launch receipts and fail-closed resource guardrails.

This module contains execution bookkeeping only. It does not score images,
load labels, or change any scientific algorithm.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence


class ExecutionBoundaryError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_exclusive(path: str | Path, value: Mapping[str, Any]) -> str:
    target = Path(path)
    payload = canonical_json(dict(value)) + b"\n"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_bytes(payload)


def atomic_json_replace(path: str | Path, value: Mapping[str, Any]) -> str:
    target = Path(path)
    payload = canonical_json(dict(value)) + b"\n"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return sha256_bytes(payload)


def append_jsonl_fsync(path: str | Path, record: Mapping[str, Any], *, create: bool = False) -> str:
    target = Path(path)
    flags = os.O_WRONLY | os.O_APPEND
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    descriptor = os.open(target, flags, 0o600)
    payload = canonical_json(dict(record)) + b"\n"
    with os.fdopen(descriptor, "ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_bytes(payload)


def append_jsonl_records_fsync(path: str | Path, records: Sequence[Mapping[str, Any]]) -> str:
    target = Path(path)
    payload = b"".join(canonical_json(dict(record)) + b"\n" for record in records)
    descriptor = os.open(target, os.O_WRONLY | os.O_APPEND)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_bytes(payload)


class AttemptStatus:
    """Exclusive attempt directory with atomically updated durable status."""

    def __init__(self, root: str | Path, *, identity: Mapping[str, Any],
                 expected_events: int, execution_kind: str):
        self.root = Path(root)
        if self.root.exists():
            raise ExecutionBoundaryError(f"attempt output already exists: {self.root}")
        self.root.mkdir(parents=True, exist_ok=False)
        self.path = self.root / "attempt_status.json"
        self.state: dict[str, Any] = {
            "schema": "final-benchmark.attempt-status.v1",
            "status": "RUNNING",
            "execution_kind": execution_kind,
            "identity": dict(identity),
            "expected_event_count": int(expected_events),
            "completed_event_count": 0,
            "failing_phase": None,
            "error_type": None,
            "error_message": None,
            "started_utc": utc_now(),
            "updated_utc": utc_now(),
            "completion_seal_path": None,
            "completion_seal_sha256": None,
        }
        self.status_sha256 = write_json_exclusive(self.path, self.state)

    def update(self, *, phase: str, completed_events: int) -> None:
        if self.state["status"] != "RUNNING":
            raise ExecutionBoundaryError("only RUNNING attempts can receive progress updates")
        if not 0 <= int(completed_events) <= self.state["expected_event_count"]:
            raise ExecutionBoundaryError("attempt progress is outside the registered event range")
        if int(completed_events) < self.state["completed_event_count"]:
            raise ExecutionBoundaryError("attempt progress cannot move backwards")
        self.state.update(failing_phase=str(phase), completed_event_count=int(completed_events),
                          updated_utc=utc_now())
        self.status_sha256 = atomic_json_replace(self.path, self.state)

    def incomplete(self, *, phase: str, completed_events: int,
                   exception: BaseException) -> None:
        if self.state["status"] != "RUNNING":
            return
        completed = max(int(self.state["completed_event_count"]), int(completed_events))
        self.state.update(status="INCOMPLETE", failing_phase=str(phase),
                          completed_event_count=completed,
                          error_type=type(exception).__name__,
                          error_message=str(exception), updated_utc=utc_now())
        self.status_sha256 = atomic_json_replace(self.path, self.state)

    def sealed_complete(self, *, phase: str, seal_path: str | Path,
                        completed_events: int) -> None:
        if self.state["status"] != "RUNNING":
            raise ExecutionBoundaryError("only a RUNNING attempt can be sealed complete")
        seal = Path(seal_path)
        if not seal.is_file() or int(completed_events) != self.state["expected_event_count"]:
            raise ExecutionBoundaryError("cannot mark attempt complete without a full-stream durable seal")
        self.state.update(status="SEALED_COMPLETE", failing_phase=str(phase),
                          completed_event_count=int(completed_events),
                          completion_seal_path=seal.name,
                          completion_seal_sha256=sha256_file(seal), updated_utc=utc_now())
        self.status_sha256 = atomic_json_replace(self.path, self.state)


def verify_attempt_receipt(status_path: str | Path, *, expected_identity: Mapping[str, Any],
                           expected_kind: str | None = None) -> dict[str, Any]:
    path = Path(status_path)
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExecutionBoundaryError("attempt status receipt is unreadable") from exc
    if (status.get("schema") != "final-benchmark.attempt-status.v1" or
            status.get("status") != "SEALED_COMPLETE" or
            status.get("identity") != dict(expected_identity) or
            (expected_kind is not None and status.get("execution_kind") != expected_kind) or
            status.get("completed_event_count") != status.get("expected_event_count")):
        raise ExecutionBoundaryError("Stage B requires a matching SEALED_COMPLETE attempt receipt")
    seal_path = path.parent / str(status.get("completion_seal_path") or "")
    if (seal_path.name != status.get("completion_seal_path") or not seal_path.is_file() or
            sha256_file(seal_path) != status.get("completion_seal_sha256")):
        raise ExecutionBoundaryError("attempt completion seal is missing or differs from its receipt")
    try:
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExecutionBoundaryError("completion seal is unreadable") from exc
    if seal.get("status") != "SEALED_COMPLETE":
        raise ExecutionBoundaryError("completion seal is not marked SEALED_COMPLETE")
    if seal.get("expected_event_count", seal.get("scored_images")) != status.get("expected_event_count"):
        raise ExecutionBoundaryError("completion seal event count differs from the attempt receipt")
    return {"status": status, "seal": seal, "seal_path": seal_path}


# Guardrails are deliberately stop limits based on bounded prior observations,
# not predictions or guarantees of full-campaign runtime/resource use.
MEASURED_RESOURCE_BASELINES = {
    "principal": {
        "observed_bytes_per_event": 5989,
        "peak_host_rss_bytes": 6708342784,
        "peak_gpu_reserved_bytes": 17263755264,
    },
    "foads": {
        "observed_bytes_per_event": 1731,
        "peak_host_rss_bytes": 11010330624,
        "peak_gpu_reserved_bytes": 1411383296,
    },
}
OUTPUT_EXTRAPOLATION_MULTIPLIER = 10
PER_ATTEMPT_FIXED_RESERVE_BYTES = 10 * 1024 * 1024
SYSTEM_DISK_RESERVE_BYTES = 10 * 1024 * 1024 * 1024
HOST_MEMORY_MARGIN = 1.25
GPU_FREE_MARGIN_BYTES = 1 * 1024 * 1024 * 1024


def estimate_attempt_output_bytes(kind: str, event_count: int) -> int:
    if kind not in MEASURED_RESOURCE_BASELINES or event_count < 0:
        raise ExecutionBoundaryError("unknown execution kind or invalid event count for resource estimate")
    per_event = MEASURED_RESOURCE_BASELINES[kind]["observed_bytes_per_event"]
    return PER_ATTEMPT_FIXED_RESERVE_BYTES + event_count * per_event * OUTPUT_EXTRAPOLATION_MULTIPLIER


def resource_snapshot(output_path: str | Path) -> dict[str, Any]:
    try:
        import psutil
        host_available = int(psutil.virtual_memory().available)
        process_rss = int(psutil.Process().memory_info().rss)
    except Exception as exc:
        raise ExecutionBoundaryError("psutil host-memory telemetry is unavailable") from exc
    disk_root = Path(output_path).resolve().parent
    while not disk_root.exists() and disk_root != disk_root.parent:
        disk_root = disk_root.parent
    disk_free = int(shutil.disk_usage(disk_root).free)
    gpu: dict[str, Any]
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10)
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(rows) != 1:
            raise ExecutionBoundaryError("resource guard requires exactly one registered NVIDIA GPU")
        name, total_mib, free_mib = [part.strip() for part in rows[0].split(",")]
        gpu = {"name": name, "total_bytes": int(total_mib) * 1024 * 1024,
               "free_bytes": int(free_mib) * 1024 * 1024}
    except ExecutionBoundaryError:
        raise
    except Exception as exc:
        raise ExecutionBoundaryError("nvidia-smi GPU availability telemetry is unavailable") from exc
    return {"host_available_bytes": host_available, "process_rss_bytes": process_rss,
            "disk_free_bytes": disk_free, "gpu": gpu}


def runtime_resource_snapshot(output_path: str | Path) -> dict[str, int | None]:
    """Low-overhead per-group telemetry; avoids spawning nvidia-smi per group."""
    try:
        import psutil
        host_available = int(psutil.virtual_memory().available)
        process_rss = int(psutil.Process().memory_info().rss)
    except Exception as exc:
        raise ExecutionBoundaryError("psutil host-memory telemetry is unavailable") from exc
    disk_root = Path(output_path).resolve().parent
    while not disk_root.exists() and disk_root != disk_root.parent:
        disk_root = disk_root.parent
    disk_free = int(shutil.disk_usage(disk_root).free)
    gpu_reserved = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_reserved = int(torch.cuda.max_memory_reserved())
    except Exception as exc:
        raise ExecutionBoundaryError("runtime CUDA memory telemetry is unavailable") from exc
    return {"host_available_bytes": host_available, "process_rss_bytes": process_rss,
            "disk_free_bytes": disk_free, "gpu_reserved_bytes": gpu_reserved}


def enforce_resource_limits(kind: str, snapshot: Mapping[str, Any], *,
                            required_disk_bytes: int,
                            runtime_gpu_reserved_bytes: int | None = None) -> None:
    if kind not in MEASURED_RESOURCE_BASELINES:
        raise ExecutionBoundaryError("unknown execution kind for resource limits")
    baseline = MEASURED_RESOURCE_BASELINES[kind]
    minimum_host = int(baseline["peak_host_rss_bytes"] * HOST_MEMORY_MARGIN)
    if int(snapshot["host_available_bytes"]) < minimum_host:
        raise ExecutionBoundaryError("available host RAM is below the measured-peak stop limit")
    if int(snapshot["disk_free_bytes"]) < int(required_disk_bytes) + SYSTEM_DISK_RESERVE_BYTES:
        raise ExecutionBoundaryError("free disk is below the per-attempt budget plus 10 GiB reserve")
    gpu = snapshot.get("gpu")
    if not isinstance(gpu, Mapping) or "3090 Ti" not in str(gpu.get("name", "")):
        raise ExecutionBoundaryError("registered RTX 3090 Ti is not the sole visible GPU")
    if runtime_gpu_reserved_bytes is None:
        if int(gpu["free_bytes"]) < int(baseline["peak_gpu_reserved_bytes"]) + GPU_FREE_MARGIN_BYTES:
            raise ExecutionBoundaryError("GPU free VRAM is below the measured-peak stop limit plus 1 GiB")
    elif int(runtime_gpu_reserved_bytes) > int(baseline["peak_gpu_reserved_bytes"] * HOST_MEMORY_MARGIN):
        raise ExecutionBoundaryError("runtime GPU reservation exceeded the 1.25x measured-peak stop limit")


def enforce_runtime_limits(kind: str, snapshot: Mapping[str, Any], *,
                           required_disk_bytes: int) -> None:
    if kind not in MEASURED_RESOURCE_BASELINES:
        raise ExecutionBoundaryError("unknown execution kind for runtime limits")
    baseline = MEASURED_RESOURCE_BASELINES[kind]
    if int(snapshot["host_available_bytes"]) < int(baseline["peak_host_rss_bytes"] * HOST_MEMORY_MARGIN):
        raise ExecutionBoundaryError("available host RAM is below the measured-peak stop limit")
    if int(snapshot["disk_free_bytes"]) < int(required_disk_bytes) + SYSTEM_DISK_RESERVE_BYTES:
        raise ExecutionBoundaryError("free disk fell below remaining attempt budget plus 10 GiB reserve")
    gpu_reserved = snapshot.get("gpu_reserved_bytes")
    if gpu_reserved is not None and int(gpu_reserved) > int(baseline["peak_gpu_reserved_bytes"] * HOST_MEMORY_MARGIN):
        raise ExecutionBoundaryError("runtime GPU reservation exceeded the 1.25x measured-peak stop limit")

