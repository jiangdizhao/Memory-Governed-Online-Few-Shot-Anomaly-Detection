"""Strictly bounded, separately sealed E6 engineering GPU smoke.

Importing this module is inert. The only real execution path requires a
reviewed exact commit, an explicit researcher authorization, audiojack, and
the fixed 20-event prefix contract. The smoke uses the shared E6 coordinator
and never emits a scientific Stage A seal.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

MAX_SMOKE_TEST_EVENTS = 20
MAX_WALL_SECONDS = 3600
MAX_GPU_MEMORY_BYTES = 22 * 1024**3
MAX_OUTPUT_BYTES = 2 * 1024**3
FAILURE_RESERVE_BYTES = 64 * 1024
MIN_FREE_DISK_BYTES = 10 * 1024**3
MAX_OUTPUT_FILES = 13
EXPECTED_PYTHON_EXECUTABLE = Path(r"D:\anaconda3\envs\torch\python.exe")
GPU_MEMORY_FALLBACK_POLICY = (
    "When process VRAM is unavailable, count all device-level used VRAM as smoke usage "
    "and enforce min(22 GiB, device total); the child PyTorch allocator remains capped "
    "at 22 GiB. This overcounts desktop/other non-compute allocations by design."
)
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_OUTPUT_NAMES = frozenset({
    ".output_budget.lock", "smoke_manifest.json", "smoke_scores.jsonl",
    *{f"smoke_feedback_Q{i}.jsonl" for i in range(7)},
    "smoke_seal.json", "smoke_failure.json", "smoke_gpu_monitoring.json",
})
R1_SUPERVISOR_SOURCE_SHA256 = "e6254a82d8dd24156cea65380b41b04cb1372744da64a085043aa6dd74df5bab"


class SmokeError(RuntimeError):
    """Bounded smoke preflight, storage, or execution failure."""


class SmokeBudgetError(SmokeError):
    """A pre-write output reservation failed closed."""


class SmokeTerminationError(SmokeError):
    """The dedicated Windows Job Object process tree could not be confirmed dead."""


@dataclass(frozen=True)
class SmokeSupervisorResult:
    status: str
    child_pid: int
    child_exit_code: int | None
    elapsed_seconds: float
    timeout_seconds: float
    termination_confirmed: bool
    active_job_processes: int
    peak_process_gpu_bytes: int | None
    supervisor_pid: int
    authorized_worker_pid: int
    observed_compute_pids: tuple[int, ...]
    foreign_compute_pids: tuple[int, ...]
    process_specific_vram_available: bool | None
    process_specific_vram_unavailable_reasons: tuple[str, ...]
    monitoring_mode: str
    monitor_sample_count: int
    peak_device_used_bytes: int | None
    configured_gpu_memory_limit_bytes: int
    termination_reason: str | None = None
    monitor_error: str | None = None


@dataclass(frozen=True)
class GPUProcessInventory:
    """Compute PIDs plus optional per-process VRAM; unknown is never zero."""

    process_ids: tuple[int, ...]
    memory_bytes_by_pid: Mapping[int, int] | None
    memory_unavailable_reason: str | None = None

    @property
    def process_memory_available(self) -> bool:
        return self.memory_bytes_by_pid is not None


@dataclass(frozen=True)
class GPUDeviceMemory:
    used_bytes: int
    free_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class GPUMonitorSample:
    processes: GPUProcessInventory
    device_memory: GPUDeviceMemory


def _reject_reparse_ancestors(path: Path) -> None:
    for candidate in (Path(path), *Path(path).parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        attrs = int(getattr(info, "st_file_attributes", 0))
        reparse = bool(attrs & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)))
        if stat.S_ISLNK(info.st_mode) or reparse:
            raise SmokeError(f"smoke path contains a symlink/reparse point: {candidate}")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_pinned_supervisor_source(repository_root: Path) -> str:
    path = Path(repository_root) / "experiments" / "calibration" / "r1_runtime_supervisor.py"
    if not path.is_file() or _sha256_file(path) != R1_SUPERVISOR_SOURCE_SHA256:
        raise SmokeError("Windows Job Object supervisor source fingerprint changed")
    from experiments.calibration import r1_runtime_supervisor
    if Path(r1_runtime_supervisor.__file__).resolve(strict=True) != path.resolve(strict=True):
        raise SmokeError("Windows Job Object supervisor was imported from an unpinned path")
    return R1_SUPERVISOR_SOURCE_SHA256


class SmokeOutputBudget:
    """Single-run, non-overwriting byte accountant for every smoke output write."""

    def __init__(self, root: Path, *, initialize: bool = False):
        raw_root = Path(root)
        _reject_reparse_ancestors(raw_root)
        self.root = raw_root.resolve(strict=True)
        parent = self.root.parent.resolve(strict=True)
        if (parent.name != "calibration" or
                not re.fullmatch(r"e6_gpu_smoke_[A-Za-z0-9][A-Za-z0-9_-]{0,63}", self.root.name) or
                not self.root.is_dir()):
            raise SmokeBudgetError("OutputBudget root is not a direct E6 smoke run directory")
        self.lock_path = self.root / ".output_budget.lock"
        if initialize:
            if any(self.root.iterdir()):
                raise SmokeBudgetError("new smoke output directory is not empty")
            with self.lock_path.open("xb") as stream:
                stream.write(b"\0")
                stream.flush()
                os.fsync(stream.fileno())
        self.require_registered()

    @contextlib.contextmanager
    def _exclusive_lock(self):
        """Fail on contention instead of allowing concurrent budget writers."""
        _reject_reparse_ancestors(self.lock_path)
        stream = self.lock_path.open("r+b")
        locked = False
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                except OSError as exc:
                    raise SmokeBudgetError("concurrent smoke output writer detected") from exc
            else:
                import fcntl
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except OSError as exc:
                    raise SmokeBudgetError("concurrent smoke output writer detected") from exc
            yield
        finally:
            if locked:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()

    def require_registered(self) -> None:
        with self._exclusive_lock():
            _reject_reparse_ancestors(self.root)
            if not self.root.is_dir() or self.lock_path.is_symlink() or not self.lock_path.is_file():
                raise SmokeBudgetError("active smoke OutputBudget registration is missing")
            if self.lock_path.stat().st_size != 1:
                raise SmokeBudgetError("smoke OutputBudget lock identity is malformed")
            self._inventory()

    def _inventory(self) -> dict[str, int]:
        files: dict[str, int] = {}
        for entry in self.root.iterdir():
            _reject_reparse_ancestors(entry)
            info = entry.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise SmokeBudgetError("smoke output namespace contains a non-regular file")
            if entry.name not in _OUTPUT_NAMES:
                raise SmokeBudgetError(f"unaccounted smoke output writer/file: {entry.name}")
            files[entry.name] = int(info.st_size)
        if len(files) > MAX_OUTPUT_FILES:
            raise SmokeBudgetError("smoke output file-count ceiling exceeded")
        return files

    def _prewrite(self, name: str, payload_size: int, *, failure: bool = False,
                  exclusive: bool = False) -> None:
        if (not isinstance(name, str) or name not in _OUTPUT_NAMES or
                name in {".output_budget.lock", "smoke_failure.json"} and not failure):
            raise SmokeBudgetError("writer is not registered for this smoke OutputBudget")
        if failure and name != "smoke_failure.json":
            raise SmokeBudgetError("reserved failure capacity is restricted to smoke_failure.json")
        files = self._inventory()
        if exclusive and name in files:
            raise FileExistsError(f"refusing to overwrite smoke output: {name}")
        projected_count = len(files) + (0 if name in files else 1)
        if projected_count > MAX_OUTPUT_FILES:
            raise SmokeBudgetError("smoke output file-count ceiling would be exceeded")
        used = sum(files.values())
        ceiling = MAX_OUTPUT_BYTES if failure else MAX_OUTPUT_BYTES - FAILURE_RESERVE_BYTES
        if failure and payload_size > FAILURE_RESERVE_BYTES:
            raise SmokeBudgetError("failure metadata exceeds its reserved capacity")
        if used + int(payload_size) > ceiling:
            raise SmokeBudgetError("smoke output byte ceiling would be exceeded before write")

    def create_empty(self, name: str) -> None:
        with self._exclusive_lock():
            self._prewrite(name, 0, exclusive=True)
            path = self.root / name
            _reject_reparse_ancestors(path)
            with path.open("xb") as stream:
                stream.flush()
                os.fsync(stream.fileno())

    def append_bytes(self, name: str, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise SmokeBudgetError("budgeted append requires bytes")
        with self._exclusive_lock():
            self._prewrite(name, len(payload))
            path = self.root / name
            _reject_reparse_ancestors(path)
            if not path.is_file():
                raise SmokeBudgetError("append-only smoke output was not registered before use")
            with path.open("ab") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())

    def write_json_once(self, name: str, value: Mapping[str, Any]) -> str:
        payload = _canonical_json(value)
        is_failure = name == "smoke_failure.json"
        with self._exclusive_lock():
            self._prewrite(name, len(payload), failure=is_failure, exclusive=True)
            path = self.root / name
            _reject_reparse_ancestors(path)
            with path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        return hashlib.sha256(payload).hexdigest()

    def write_failure(self, value: Mapping[str, Any]) -> str | None:
        try:
            return self.write_json_once("smoke_failure.json", value)
        except FileExistsError:
            return None

    def total_bytes(self) -> int:
        with self._exclusive_lock():
            return sum(self._inventory().values())


def create_smoke_output_namespace(run_id: str, *, repository_root: Path) -> tuple[Path, SmokeOutputBudget]:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise SmokeError("run ID must be a filesystem-safe 1-64 character token")
    root = Path(repository_root).resolve(strict=True)
    output_parent_unresolved = root / "outputs" / "calibration"
    _reject_reparse_ancestors(output_parent_unresolved)
    output_parent = output_parent_unresolved.resolve(strict=True)
    run_dir = output_parent / f"e6_gpu_smoke_{run_id}"
    _reject_reparse_ancestors(run_dir)
    run_dir.mkdir(exist_ok=False)
    return run_dir, SmokeOutputBudget(run_dir, initialize=True)


def _run_nvidia_smi_query(arguments: Sequence[str], *,
                          runner: Callable[..., Any]) -> Any:
    try:
        return runner(
            ["nvidia-smi", *arguments],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SmokeError("nvidia-smi GPU monitoring query failed") from exc


def _query_gpu_processes(*, runner: Callable[..., Any] = subprocess.run) -> GPUProcessInventory:
    """Query compute PIDs separately from optional per-process memory telemetry."""
    pid_result = _run_nvidia_smi_query(
        ["--query-compute-apps=pid", "--format=csv,noheader,nounits"], runner=runner)
    if int(pid_result.returncode) != 0:
        raise SmokeError("nvidia-smi could not verify compute-process PIDs")
    process_ids: set[int] = set()
    for line in str(pid_result.stdout).splitlines():
        value = line.strip()
        if not value:
            continue
        if not value.isdigit() or int(value) <= 0:
            raise SmokeError("nvidia-smi returned an unaccountable compute-process PID")
        process_ids.add(int(value))

    try:
        memory_result = _run_nvidia_smi_query(
            ["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            runner=runner)
    except SmokeError:
        return GPUProcessInventory(tuple(sorted(process_ids)), None,
                                   "per_process_memory_query_failed")
    if int(memory_result.returncode) != 0:
        return GPUProcessInventory(tuple(sorted(process_ids)), None,
                                   "per_process_memory_query_failed")

    memory_rows: dict[int, int] = {}
    observed_memory_pids: set[int] = set()
    unavailable_reason: str | None = None
    for line in str(memory_result.stdout).splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) != 2 or not fields[0].isdigit() or int(fields[0]) <= 0:
            unavailable_reason = "per_process_memory_response_unparseable"
            continue
        pid = int(fields[0])
        observed_memory_pids.add(pid)
        value = fields[1]
        normalized_value = value.strip().strip("[]").upper()
        if normalized_value in {"N/A", "NA", "NOT SUPPORTED", "NOT AVAILABLE", "UNKNOWN"}:
            unavailable_reason = "driver_reports_per_process_memory_NA"
            continue
        if not value.isdigit():
            unavailable_reason = "per_process_memory_response_unparseable"
            continue
        memory_rows[pid] = memory_rows.get(pid, 0) + int(value) * 1024**2

    all_observed_pids = process_ids | observed_memory_pids
    if observed_memory_pids != process_ids:
        unavailable_reason = unavailable_reason or "process_list_changed_between_queries"
    if unavailable_reason is not None:
        # Never map N/A (or a missing row) to zero. The caller must use the
        # device-level conservative fallback and record this status.
        return GPUProcessInventory(tuple(sorted(all_observed_pids)), None, unavailable_reason)
    return GPUProcessInventory(tuple(sorted(process_ids)), memory_rows, None)


def _query_gpu_device_memory(*, torch_module: Any = None,
                             runner: Callable[..., Any] = subprocess.run) -> GPUDeviceMemory:
    """Require device-wide used/free/total telemetry for process-memory fallback.

    Prefer CUDA's device-level memory API when available so a single visible
    CUDA device is unambiguous even when the host has other physical adapters.
    The nvidia-smi device query is retained as a testable fallback.
    """
    if torch_module is not None:
        try:
            free_bytes, total_bytes = torch_module.cuda.mem_get_info(0)
            free_bytes, total_bytes = int(free_bytes), int(total_bytes)
        except Exception as exc:
            raise SmokeError("CUDA device-level GPU memory telemetry is unavailable") from exc
        used_bytes = total_bytes - free_bytes
        if total_bytes <= 0 or free_bytes < 0 or used_bytes < 0 or free_bytes > total_bytes:
            raise SmokeError("CUDA device-level memory telemetry is outside valid bounds")
        return GPUDeviceMemory(used_bytes, free_bytes, total_bytes)
    result = _run_nvidia_smi_query(
        ["--query-gpu=memory.used,memory.free,memory.total",
         "--format=csv,noheader,nounits"], runner=runner)
    if int(result.returncode) != 0:
        raise SmokeError("nvidia-smi could not verify device-level GPU memory")
    rows = [line.strip() for line in str(result.stdout).splitlines() if line.strip()]
    if len(rows) != 1:
        raise SmokeError("GPU device-level memory telemetry is missing or ambiguous")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 3 or any(not field.isdigit() for field in fields):
        raise SmokeError("GPU device-level used/free/total memory is unavailable")
    used_mib, free_mib, total_mib = map(int, fields)
    if (total_mib <= 0 or used_mib > total_mib or free_mib > total_mib):
        raise SmokeError("GPU device-level memory telemetry is outside valid bounds")
    return GPUDeviceMemory(used_mib * 1024**2, free_mib * 1024**2,
                           total_mib * 1024**2)


def _query_gpu_monitor_sample(*, torch_module: Any = None,
                              runner: Callable[..., Any] = subprocess.run) -> GPUMonitorSample:
    return GPUMonitorSample(
        processes=_query_gpu_processes(runner=runner),
        device_memory=_query_gpu_device_memory(torch_module=torch_module, runner=runner))


def _foreign_compute_pids(process_ids: Sequence[int], *, supervisor_pid: int,
                          authorized_worker_pid: int | None) -> tuple[int, ...]:
    allowed = {int(supervisor_pid)}
    if authorized_worker_pid is not None:
        allowed.add(int(authorized_worker_pid))
    return tuple(sorted({int(pid) for pid in process_ids} - allowed))


def _gpu_monitoring_profile(sample: GPUMonitorSample, *, supervisor_pid: int,
                            authorized_worker_pid: int | None) -> dict[str, Any]:
    foreign = _foreign_compute_pids(
        sample.processes.process_ids, supervisor_pid=supervisor_pid,
        authorized_worker_pid=authorized_worker_pid)
    memory_available = sample.processes.process_memory_available
    if authorized_worker_pid is not None:
        memory_available = bool(
            memory_available and
            int(authorized_worker_pid) in (sample.processes.memory_bytes_by_pid or {}))
    return {
        "supervisor_pid": int(supervisor_pid),
        "authorized_worker_pid": (None if authorized_worker_pid is None
                                  else int(authorized_worker_pid)),
        "compute_process_ids": list(sample.processes.process_ids),
        "foreign_compute_pids": list(foreign),
        "process_memory_query_available": sample.processes.process_memory_available,
        "process_specific_vram_available": (
            None if authorized_worker_pid is None else bool(memory_available)),
        "process_specific_vram_unavailable_reason": (
            sample.processes.memory_unavailable_reason if not memory_available else None),
        "device_memory_used_bytes": sample.device_memory.used_bytes,
        "device_memory_free_bytes": sample.device_memory.free_bytes,
        "device_memory_total_bytes": sample.device_memory.total_bytes,
        "configured_process_limit_bytes": MAX_GPU_MEMORY_BYTES,
        "fallback_policy": GPU_MEMORY_FALLBACK_POLICY,
    }


def preflight_gpu_smoke(*, category: str, max_test_events: int, run_id: str,
                        approved_commit: str, authorization_id: str,
                        human_authorized: bool, repository_root: Path,
                        torch_module: Any = None,
                        subprocess_runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Read-only exact-identity, prefix, model, GPU-process, and disk preflight."""
    if human_authorized is not True or not authorization_id:
        raise SmokeError("GPU smoke requires explicit researcher authorization and ID")
    if category != "audiojack":
        raise SmokeError("bounded smoke is authorized for audiojack only")
    if isinstance(max_test_events, bool) or int(max_test_events) != MAX_SMOKE_TEST_EVENTS:
        raise SmokeError("bounded smoke requires exactly the fixed 20-event maximum")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise SmokeError("run ID must be a filesystem-safe 1-64 character token")
    root = Path(repository_root).resolve(strict=True)
    try:
        actual_python = Path(sys.executable).resolve(strict=True)
        expected_python = EXPECTED_PYTHON_EXECUTABLE.resolve(strict=True)
    except OSError as exc:
        raise SmokeError("the registered E6 Python interpreter is unavailable") from exc
    if actual_python != expected_python:
        raise SmokeError("GPU smoke requires the registered D:\\anaconda3\\envs\\torch Python interpreter")

    from experiments.calibration.e6_bea_closed_loop import verify_e6_execution_identity
    identity = verify_e6_execution_identity(approved_commit, root=root)
    supervisor_sha = _verify_pinned_supervisor_source(root)
    from experiments.calibration.e6_bea_runner import load_e6_stage05_category
    info = load_e6_stage05_category(
        "audiojack", repository_root=root, verify_image_paths=True,
        smoke_prefix_events=MAX_SMOKE_TEST_EVENTS)
    if (len(info.get("events", ())) != MAX_SMOKE_TEST_EVENTS or
            len({event.source_group_sha256 for event in info["events"]}) != 4):
        raise SmokeError("frozen R7 metadata does not provide four complete five-view prefix groups")

    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration import r1_matched_correction_pilot as r1
    model_receipt = r3._get_verified_model_receipt()
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SmokeError("preflight requires exactly one visible CUDA device")
    device = torch.cuda.get_device_properties(0)
    device_name = str(device.name)
    total_gpu_bytes = int(device.total_memory)
    if device_name != model_receipt.get("runtime_profile", {}).get("cuda_device_name"):
        raise SmokeError("visible CUDA device differs from the verified local model receipt")
    if str(torch.__version__) != model_receipt.get("runtime_profile", {}).get("torch_version"):
        raise SmokeError("PyTorch version differs from the verified local model receipt")
    processes = _query_gpu_processes(runner=subprocess_runner)
    device_memory = _query_gpu_device_memory(torch_module=torch, runner=subprocess_runner)
    supervisor_pid = os.getpid()
    foreign = _foreign_compute_pids(
        processes.process_ids, supervisor_pid=supervisor_pid,
        authorized_worker_pid=None)
    if foreign:
        raise SmokeError(f"foreign CUDA compute processes are present: {list(foreign)}")
    effective_total = min(total_gpu_bytes, device_memory.total_bytes)
    required_free = min(MAX_GPU_MEMORY_BYTES, effective_total)
    conservative_free = device_memory.free_bytes
    if conservative_free < required_free:
        raise SmokeError("available GPU memory is below the registered 22 GiB process ceiling")
    hub_root = Path(torch.hub.get_dir())
    hub_path, hub_sha = r1._hub_source_sha256(hub_root)
    if (hub_sha != r3.KNOWN_R1_HUB_TREE_SHA256 or
            hub_sha != model_receipt.get("hub_source_tree_sha256")):
        raise SmokeError("local DINOv2 source tree fingerprint differs from the verified receipt")
    checkpoint = hub_root / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
    checkpoint_sha = r3.sha256_file(checkpoint) if checkpoint.is_file() else None
    if (checkpoint_sha != r3.KNOWN_CHECKPOINT_SHA256 or
            checkpoint_sha != model_receipt.get("checkpoint_sha256")):
        raise SmokeError("verified local DINOv2 checkpoint is absent or has changed")

    output_parent_unresolved = root / "outputs" / "calibration"
    _reject_reparse_ancestors(output_parent_unresolved)
    output_parent = output_parent_unresolved.resolve(strict=True)
    output_path = output_parent / f"e6_gpu_smoke_{run_id}"
    _reject_reparse_ancestors(output_path)
    if output_path.exists():
        raise SmokeError("smoke output run ID already exists; resuming and overwrite are forbidden")
    free_disk = int(shutil.disk_usage(output_parent).free)
    if free_disk < MIN_FREE_DISK_BYTES:
        raise SmokeError("preflight requires at least 10 GiB free disk space")
    return {
        "branch": identity["branch"], "approved_commit": identity["head"],
        "authorization_id": str(authorization_id), "category": "audiojack",
        "event_count": MAX_SMOKE_TEST_EVENTS,
        "stream_prefix_sha256": hashlib.sha256(
            __import__("experiments.calibration.e6_bea_closed_loop", fromlist=["_canonical_json"])
            ._canonical_json([{
                "position": event.position,
                "original_json_position": event.original_position,
                "event_id_sha256": event.event_id_sha256,
                "image_identity_sha256": event.image_identity_sha256,
                "source_group_sha256": event.source_group_sha256,
                "camera_id": event.camera_id,
            } for event in info["events"]])).hexdigest(),
        "category_json_sha256": info["contract"]["category_json_sha256"],
        "stage05_receipts": dict(info["stage05_receipts"]),
        "model_checkpoint_sha256": checkpoint_sha, "model_hub_source_sha256": hub_sha,
        "model_loaded_state_sha256": model_receipt.get("loaded_model_state_sha256"),
        "windows_job_supervisor_source_sha256": supervisor_sha,
        "cuda_device_name": device_name, "cuda_device_total_bytes": total_gpu_bytes,
        "cuda_free_bytes": conservative_free,
        "gpu_monitoring": _gpu_monitoring_profile(
            GPUMonitorSample(processes, device_memory), supervisor_pid=supervisor_pid,
            authorized_worker_pid=None),
        "free_disk_bytes": free_disk, "output_dir": str(output_path),
        "model_source_location": str(hub_path),
        "python_executable": str(actual_python),
    }


def _gpu_process_memory_for_child(pid: int) -> GPUMonitorSample:
    import torch
    return _query_gpu_monitor_sample(torch_module=torch)


def supervise_smoke_child(command: Sequence[str], *, environment: Mapping[str, str],
                          working_directory: Path,
                          gpu_process_probe: Callable[[int], GPUMonitorSample] =
                          _gpu_process_memory_for_child,
                          timeout_seconds: float = MAX_WALL_SECONDS,
                          poll_interval_seconds: float = 1.0) -> SmokeSupervisorResult:
    """Run the worker suspended inside a kill-on-close Windows Job Object."""
    from experiments.calibration import r1_runtime_supervisor as r1_supervisor

    r1_supervisor._require_windows()
    if not command or any(not isinstance(value, str) or not value for value in command):
        raise SmokeError("smoke worker command is invalid")
    if not 0 < float(timeout_seconds) <= MAX_WALL_SECONDS:
        raise SmokeError("smoke deadline must not exceed 3,600 seconds")
    if not 0 < float(poll_interval_seconds) <= 1.0:
        raise SmokeError("smoke watchdog polling interval is invalid")
    job = r1_supervisor._WindowsJob()
    process_handle = None
    child_pid = -1
    started = time.monotonic()
    deadline = started + float(timeout_seconds)
    termination_reason = None
    monitor_error = None
    termination_confirmed = False
    active = 0
    root_exited_at = None
    exit_code = None
    supervisor_pid = os.getpid()
    peak_process_gpu_bytes: int | None = None
    peak_device_used_bytes: int | None = None
    monitor_sample_count = 0
    process_specific_sample_count = 0
    process_memory_unavailable_reasons: set[str] = set()
    observed_compute_pids: set[int] = set()
    foreign_compute_pids: set[int] = set()
    monitoring_modes: set[str] = set()
    try:
        process_handle, child_pid = r1_supervisor._create_child_suspended(
            command, dict(environment), Path(working_directory), job)
        kernel32 = job.kernel32
        while True:
            now = time.monotonic()
            if termination_reason is None and now >= deadline:
                termination_reason = "wall_clock_deadline"
                job.terminate()
            wait_ms = max(1, min(1000, int(float(poll_interval_seconds) * 1000),
                                int(max(0.0, deadline - now) * 1000)))
            wait_state = kernel32.WaitForSingleObject(process_handle, wait_ms)
            active = job.active_processes()
            if wait_state not in (r1_supervisor.WAIT_OBJECT_0, r1_supervisor.WAIT_TIMEOUT):
                termination_reason = termination_reason or "process_wait_failed"
                monitor_error = f"unexpected Windows wait status 0x{wait_state:08x}"
                job.terminate()
            if wait_state == r1_supervisor.WAIT_OBJECT_0:
                root_exited_at = root_exited_at or time.monotonic()
            elif termination_reason is None:
                try:
                    sample = gpu_process_probe(child_pid)
                    if not isinstance(sample, GPUMonitorSample):
                        raise SmokeError("GPU monitor did not provide PID and device-memory telemetry")
                    monitor_sample_count += 1
                    process_ids = set(sample.processes.process_ids)
                    observed_compute_pids.update(process_ids)
                    foreign = set(_foreign_compute_pids(
                        process_ids, supervisor_pid=supervisor_pid,
                        authorized_worker_pid=child_pid))
                    foreign_compute_pids.update(foreign)
                    device_memory = sample.device_memory
                    if (device_memory.total_bytes <= 0 or device_memory.used_bytes < 0 or
                            device_memory.free_bytes < 0 or
                            device_memory.used_bytes > device_memory.total_bytes or
                            device_memory.free_bytes > device_memory.total_bytes):
                        raise SmokeError("device-level GPU memory telemetry is invalid")
                    peak_device_used_bytes = max(
                        peak_device_used_bytes or 0, int(device_memory.used_bytes))
                    if foreign:
                        termination_reason = "competing_gpu_process_detected"
                    memory_by_pid = sample.processes.memory_bytes_by_pid
                    worker_memory_available = bool(
                        memory_by_pid is not None and child_pid in process_ids and
                        child_pid in memory_by_pid)
                    if worker_memory_available:
                        process_specific_sample_count += 1
                        child_gpu_bytes = int(memory_by_pid[child_pid])
                        if child_gpu_bytes < 0:
                            raise SmokeError("per-process GPU memory telemetry is invalid")
                        peak_process_gpu_bytes = max(
                            peak_process_gpu_bytes or 0, child_gpu_bytes)
                        monitoring_modes.add("process_specific")
                        if child_gpu_bytes > MAX_GPU_MEMORY_BYTES:
                            termination_reason = "gpu_memory_ceiling_exceeded"
                    else:
                        reason = (sample.processes.memory_unavailable_reason or
                                  "authorized_worker_pid_not_reported")
                        process_memory_unavailable_reasons.add(reason)
                        monitoring_modes.add("device_level_conservative_fallback")
                        conservative_limit = min(
                            MAX_GPU_MEMORY_BYTES, int(device_memory.total_bytes))
                        # With no child-specific value, count all used device VRAM
                        # against the smoke. This intentionally overcounts display
                        # and other non-compute usage and never treats N/A as zero.
                        if int(device_memory.used_bytes) > conservative_limit:
                            termination_reason = "device_memory_ceiling_exceeded"
                except Exception as exc:
                    termination_reason = "gpu_memory_monitor_failed"
                    monitor_error = f"{type(exc).__name__}: {str(exc)[:500]}"
                if termination_reason is None and time.monotonic() >= deadline:
                    termination_reason = "wall_clock_deadline"
            if termination_reason and active > 0:
                job.terminate()
            if wait_state == r1_supervisor.WAIT_OBJECT_0 and active == 0:
                termination_confirmed = True
                break
            if root_exited_at is not None and active > 0 and time.monotonic() - root_exited_at >= 1.0:
                termination_reason = termination_reason or "descendant_process_remained"
                job.terminate()
            if termination_reason is not None:
                if time.monotonic() - max(deadline, started) > 30.0 and active > 0:
                    break
            elif time.monotonic() >= deadline:
                termination_reason = "wall_clock_deadline"
        if not termination_confirmed:
            grace_deadline = time.monotonic() + 30.0
            while time.monotonic() < grace_deadline:
                process_state = kernel32.WaitForSingleObject(process_handle, 0)
                active = job.active_processes()
                if process_state == r1_supervisor.WAIT_OBJECT_0 and active == 0:
                    termination_confirmed = True
                    break
                time.sleep(min(0.05, max(0.0, grace_deadline - time.monotonic())))
        if not termination_confirmed:
            raise SmokeTerminationError(
                f"could not confirm E6 smoke process tree exit; active={active}")
        code = ctypes.c_uint32(0)
        if not kernel32.GetExitCodeProcess(process_handle, ctypes.byref(code)):
            raise SmokeTerminationError("GetExitCodeProcess failed after confirmed smoke exit")
        exit_code = int(code.value)
    finally:
        job.close()
        if process_handle is not None:
            kernel32 = job.kernel32
            kernel32.WaitForSingleObject(process_handle, 30_000)
            kernel32.CloseHandle(process_handle)

    elapsed = time.monotonic() - started
    if exit_code == 0 and monitor_sample_count == 0 and termination_reason is None:
        termination_reason = "gpu_monitoring_unavailable"
        monitor_error = "completed child had no runtime GPU process/device telemetry samples"
    if monitor_sample_count == 0:
        process_specific_available: bool | None = None
    else:
        process_specific_available = process_specific_sample_count == monitor_sample_count
    if not monitoring_modes:
        monitoring_mode = "unavailable"
    elif len(monitoring_modes) == 1:
        monitoring_mode = next(iter(monitoring_modes))
    else:
        monitoring_mode = "mixed"
    if termination_reason == "wall_clock_deadline":
        status = "TIMED_OUT_TERMINATED"
    elif termination_reason is not None:
        status = "RESOURCE_LIMIT_TERMINATED"
    elif exit_code == 0:
        status = "COMPLETED"
    else:
        status = "FAILED_NONZERO_EXIT"
    return SmokeSupervisorResult(
        status=status, child_pid=child_pid, child_exit_code=exit_code,
        elapsed_seconds=elapsed, timeout_seconds=float(timeout_seconds),
        termination_confirmed=termination_confirmed,
        active_job_processes=active, peak_process_gpu_bytes=peak_process_gpu_bytes,
        supervisor_pid=supervisor_pid, authorized_worker_pid=child_pid,
        observed_compute_pids=tuple(sorted(observed_compute_pids)),
        foreign_compute_pids=tuple(sorted(foreign_compute_pids)),
        process_specific_vram_available=process_specific_available,
        process_specific_vram_unavailable_reasons=tuple(sorted(process_memory_unavailable_reasons)),
        monitoring_mode=monitoring_mode, monitor_sample_count=monitor_sample_count,
        peak_device_used_bytes=peak_device_used_bytes,
        configured_gpu_memory_limit_bytes=MAX_GPU_MEMORY_BYTES,
        termination_reason=termination_reason, monitor_error=monitor_error)


def _validate_smoke_outputs(budget: SmokeOutputBudget, run_id: str) -> dict[str, Any]:
    budget.require_registered()
    root = budget.root
    if root.name != f"e6_gpu_smoke_{run_id}":
        raise SmokeError("smoke output directory identity changed during execution")
    if any((root / name).exists() for name in ("run_manifest.json", "stage_a_seal.json", "sealed_scores.jsonl")):
        raise SmokeError("smoke produced a scientific Stage A artifact")
    manifest = json.loads((root / "smoke_manifest.json").read_text(encoding="utf-8"))
    seal = json.loads((root / "smoke_seal.json").read_text(encoding="utf-8"))
    monitoring = json.loads((root / "smoke_gpu_monitoring.json").read_text(encoding="utf-8"))
    if (manifest.get("artifact_kind") != "bounded_engineering_smoke" or
            manifest.get("stage_b_enabled") is not False or
            seal.get("artifact_kind") != "bounded_engineering_smoke" or
            seal.get("scientific_stage_a_complete") is not False or
            seal.get("stage_b_enabled") is not False or
            seal.get("status") not in {"SMOKE_PASS", "SMOKE_INCOMPLETE_COVERAGE"} or
            seal.get("event_count") != MAX_SMOKE_TEST_EVENTS or
            monitoring.get("schema") != "e6.bea_v2.smoke_gpu_monitoring.v1" or
            monitoring.get("status") != "MONITORING_COMPLETE" or
            monitoring.get("configured_process_limit_bytes") != MAX_GPU_MEMORY_BYTES or
            monitoring.get("process_specific_vram_available") not in (True, False) or
            int(monitoring.get("monitor_sample_count", 0)) <= 0 or
            monitoring.get("foreign_compute_pids") != []):
        raise SmokeError("smoke-specific durable receipt is missing or malformed")
    if monitoring["process_specific_vram_available"] is False:
        mode = monitoring.get("monitoring_mode")
        if (mode not in {"device_level_conservative_fallback", "mixed"} or
                (mode == "device_level_conservative_fallback" and
                 monitoring.get("peak_process_gpu_bytes") is not None) or
                monitoring.get("peak_device_used_bytes") is None or
                monitoring.get("fallback_policy") != GPU_MEMORY_FALLBACK_POLICY):
            raise SmokeError("WDDM fallback receipt misreports unavailable per-process VRAM")
    elif (monitoring.get("monitoring_mode") != "process_specific" or
          monitoring.get("peak_process_gpu_bytes") is None):
        raise SmokeError("per-process VRAM receipt omits its measured peak or mode")
    score_rows = (root / "smoke_scores.jsonl").read_bytes().splitlines()
    if len(score_rows) != MAX_SMOKE_TEST_EVENTS:
        raise SmokeError("smoke score ledger does not contain exactly 20 sealed events")
    for position, raw in enumerate(score_rows):
        row = json.loads(raw)
        if (row.get("schema") != "e6.bea_v2.bounded_smoke_score.v1" or
                row.get("position") != position or set(row.get("arm_rows", {})) !=
                {f"Q{i}" for i in range(7)}):
            raise SmokeError("smoke score ledger identity/arm contract failed")
    return seal


def _worker_main(*, run_id: str, category: str, max_test_events: int,
                 approved_commit: str, authorization_id: str,
                 human_authorized: bool, output_dir: Path,
                 repository_root: Path) -> int:
    if human_authorized is not True or not authorization_id:
        raise SmokeError("smoke child requires explicit researcher authorization")
    if category != "audiojack" or int(max_test_events) != MAX_SMOKE_TEST_EVENTS:
        raise SmokeError("smoke child received an incompatible category or event cap")
    from experiments.calibration import r1_runtime_supervisor as r1_supervisor
    if not r1_supervisor.current_process_is_in_job():
        raise SmokeError("smoke worker must run inside the dedicated Windows Job Object")
    root = Path(repository_root).resolve(strict=True)
    expected_raw = root / "outputs" / "calibration" / f"e6_gpu_smoke_{run_id}"
    output_raw = Path(output_dir)
    _reject_reparse_ancestors(expected_raw)
    _reject_reparse_ancestors(output_raw)
    expected_dir = expected_raw.resolve(strict=True)
    output_dir = output_raw.resolve(strict=True)
    if output_dir != expected_dir:
        raise SmokeError("smoke child output path does not match its unique run ID")
    budget = SmokeOutputBudget(output_dir)
    from experiments.calibration.e6_bea_runner import execute_bounded_smoke_stage_a
    result = execute_bounded_smoke_stage_a(
        run_id=run_id, approved_commit=approved_commit,
        authorization_id=authorization_id, repository_root=root,
        output_budget=budget)
    if result.artifact_kind != "bounded_engineering_smoke" or result.complete:
        raise SmokeError("bounded smoke coordinator returned a scientific completion receipt")
    return 0


def execute_gpu_smoke(*, category: str, max_test_events: int, run_id: str,
                      approved_commit: str, authorization_id: str,
                      human_authorized: bool, repository_root: Path) -> dict[str, Any]:
    """Preflight, create one unique run, and supervise one worker without retry."""
    root = Path(repository_root).resolve(strict=True)
    preflight = preflight_gpu_smoke(
        category=category, max_test_events=max_test_events, run_id=run_id,
        approved_commit=approved_commit, authorization_id=authorization_id,
        human_authorized=human_authorized, repository_root=root)
    import torch as torch_module
    output_dir, budget = create_smoke_output_namespace(run_id, repository_root=root)
    command = [
        sys.executable, "-m", "experiments.calibration.e6_bea_smoke", "--_worker",
        "--researcher-authorized", "--authorization-id", authorization_id,
        "--approved-commit", approved_commit, "--category", "audiojack",
        "--run-id", run_id, "--max-test-events", str(MAX_SMOKE_TEST_EVENTS),
        "--_output-dir", str(output_dir),
    ]
    try:
        result = supervise_smoke_child(
            command, environment=os.environ.copy(), working_directory=root,
            timeout_seconds=MAX_WALL_SECONDS,
            gpu_process_probe=lambda _pid: _query_gpu_monitor_sample(
                torch_module=torch_module))
        budget.write_json_once("smoke_gpu_monitoring.json", {
            "schema": "e6.bea_v2.smoke_gpu_monitoring.v1",
            "status": "MONITORING_COMPLETE" if result.monitor_sample_count else "MONITORING_UNAVAILABLE",
            "preflight": preflight["gpu_monitoring"],
            "supervisor_pid": result.supervisor_pid,
            "authorized_worker_pid": result.authorized_worker_pid,
            "observed_compute_pids": list(result.observed_compute_pids),
            "foreign_compute_pids": list(result.foreign_compute_pids),
            "process_specific_vram_available": result.process_specific_vram_available,
            "process_specific_vram_unavailable_reasons": list(
                result.process_specific_vram_unavailable_reasons),
            "monitoring_mode": result.monitoring_mode,
            "monitor_sample_count": result.monitor_sample_count,
            "peak_process_gpu_bytes": result.peak_process_gpu_bytes,
            "peak_device_used_bytes": result.peak_device_used_bytes,
            "configured_process_limit_bytes": result.configured_gpu_memory_limit_bytes,
            "fallback_policy": GPU_MEMORY_FALLBACK_POLICY,
            "monitor_error": result.monitor_error,
        })
        if result.status != "COMPLETED" or not result.termination_confirmed:
            budget.write_failure({
                "schema": "e6.bea_v2.smoke_failure.v1", "status": "SMOKE_FAILED",
                "artifact_kind": "bounded_engineering_smoke", "stage": "supervisor",
                "failure_type": result.status,
                "message": result.termination_reason or "smoke child did not complete successfully",
                "supervisor": asdict(result), "automatic_restart": False,
                "stage_b_enabled": False,
            })
            raise SmokeError(f"bounded smoke child failed: {result.status}")
        seal = _validate_smoke_outputs(budget, run_id)
        return {"status": seal["status"], "output_dir": str(output_dir),
                "event_count": MAX_SMOKE_TEST_EVENTS, "supervisor": asdict(result),
                "preflight": preflight}
    except BaseException as exc:
        budget.write_failure({
            "schema": "e6.bea_v2.smoke_failure.v1", "status": "SMOKE_FAILED",
            "artifact_kind": "bounded_engineering_smoke", "stage": "parent_supervisor",
            "failure_type": type(exc).__name__, "message": str(exc)[:2000],
            "automatic_restart": False, "stage_b_enabled": False,
        })
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-gpu-smoke", action="store_true")
    parser.add_argument("--researcher-authorized", action="store_true")
    parser.add_argument("--authorization-id")
    parser.add_argument("--approved-commit")
    parser.add_argument("--category")
    parser.add_argument("--run-id")
    parser.add_argument("--max-test-events", type=int, default=MAX_SMOKE_TEST_EVENTS)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_output-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        try:
            if not args._output_dir:
                raise SmokeError("smoke worker output directory is required")
            _worker_main(
                run_id=args.run_id, category=args.category,
                max_test_events=args.max_test_events, approved_commit=args.approved_commit,
                authorization_id=args.authorization_id,
                human_authorized=args.researcher_authorized,
                output_dir=args._output_dir,
                repository_root=Path(__file__).resolve().parents[2])
        except Exception as exc:
            print(f"E6 smoke worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        return 0
    if not args.execute_gpu_smoke:
        print("E6 GPU smoke is disabled; no dataset, model, labels, or CUDA were accessed.")
        return 0
    try:
        if not args.category or not args.run_id or not args.approved_commit:
            raise SmokeError("--category, --run-id, and --approved-commit are required")
        result = execute_gpu_smoke(
            category=args.category, max_test_events=args.max_test_events,
            run_id=args.run_id, approved_commit=args.approved_commit,
            authorization_id=args.authorization_id or "",
            human_authorized=args.researcher_authorized,
            repository_root=Path(__file__).resolve().parents[2])
        print(json.dumps({key: value for key, value in result.items()
                          if key != "preflight"}, sort_keys=True))
    except Exception as exc:
        print(f"E6 GPU smoke stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
