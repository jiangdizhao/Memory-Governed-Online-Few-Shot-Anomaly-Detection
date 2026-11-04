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
MAX_OUTPUT_FILES = 12
EXPECTED_PYTHON_EXECUTABLE = Path(r"D:\anaconda3\envs\torch\python.exe")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_OUTPUT_NAMES = frozenset({
    ".output_budget.lock", "smoke_manifest.json", "smoke_scores.jsonl",
    *{f"smoke_feedback_Q{i}.jsonl" for i in range(7)},
    "smoke_seal.json", "smoke_failure.json",
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
    peak_process_gpu_bytes: int
    termination_reason: str | None = None
    monitor_error: str | None = None


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


def _query_gpu_processes(*, runner: Callable[..., Any] = subprocess.run) -> dict[int, int]:
    try:
        result = runner(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SmokeError("nvidia-smi compute-process query failed") from exc
    if int(result.returncode) != 0:
        raise SmokeError("nvidia-smi could not verify active compute processes")
    processes: dict[int, int] = {}
    for line in str(result.stdout).splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1].isdigit():
            raise SmokeError("nvidia-smi returned unaccountable GPU process memory")
        pid, memory_mib = int(fields[0]), int(fields[1])
        processes[pid] = processes.get(pid, 0) + memory_mib * 1024**2
    return processes


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
    if processes:
        raise SmokeError("a competing CUDA compute process is present; preflight fails closed")
    free_gpu_bytes, _total = torch.cuda.mem_get_info(0)
    if int(free_gpu_bytes) < min(MAX_GPU_MEMORY_BYTES, total_gpu_bytes):
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
        "cuda_free_bytes": int(free_gpu_bytes), "gpu_processes": processes,
        "free_disk_bytes": free_disk, "output_dir": str(output_path),
        "model_source_location": str(hub_path),
        "python_executable": str(actual_python),
    }


def _gpu_process_memory_for_child(pid: int) -> Mapping[int, int]:
    return _query_gpu_processes()


def supervise_smoke_child(command: Sequence[str], *, environment: Mapping[str, str],
                          working_directory: Path,
                          gpu_process_probe: Callable[[int], Mapping[int, int]] =
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
    peak_gpu_bytes = 0
    try:
        process_handle, child_pid = r1_supervisor._create_child_suspended(
            command, dict(environment), Path(working_directory), job)
        kernel32 = job.kernel32
        while True:
            now = time.monotonic()
            if termination_reason is None and now >= deadline:
                termination_reason = "wall_clock_deadline"
                job.terminate()
            wait_ms = max(1, min(1000, int(max(0.0, deadline - now) * 1000)))
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
                    processes = dict(gpu_process_probe(child_pid))
                    foreign_pids = set(processes) - {child_pid}
                    if foreign_pids:
                        termination_reason = "competing_gpu_process_detected"
                    child_gpu_bytes = int(processes.get(child_pid, 0))
                    peak_gpu_bytes = max(peak_gpu_bytes, child_gpu_bytes)
                    if child_gpu_bytes > MAX_GPU_MEMORY_BYTES:
                        termination_reason = "gpu_memory_ceiling_exceeded"
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
        active_job_processes=active, peak_process_gpu_bytes=peak_gpu_bytes,
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
    if (manifest.get("artifact_kind") != "bounded_engineering_smoke" or
            manifest.get("stage_b_enabled") is not False or
            seal.get("artifact_kind") != "bounded_engineering_smoke" or
            seal.get("scientific_stage_a_complete") is not False or
            seal.get("stage_b_enabled") is not False or
            seal.get("status") not in {"SMOKE_PASS", "SMOKE_INCOMPLETE_COVERAGE"} or
            seal.get("event_count") != MAX_SMOKE_TEST_EVENTS):
        raise SmokeError("smoke-specific durable receipt is missing or malformed")
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
            timeout_seconds=MAX_WALL_SECONDS)
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
