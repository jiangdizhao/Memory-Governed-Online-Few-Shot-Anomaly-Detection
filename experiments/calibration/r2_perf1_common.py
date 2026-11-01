"""Shared contracts for the bounded R2-PERF-1 switch evaluation.

This module deliberately contains no outcome reader and no correction policy.
Stage A and Stage B run in distinct child processes and exchange only a sealed
score table in the unique output namespace.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Iterator, Mapping


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PARENT = ROOT / "outputs" / "calibration"
OUTPUT_PREFIX = "r2_perf1_switch_s0_512_"
MAX_OUTPUT_BYTES = 2 * 1024**3
FAILURE_RESERVE_BYTES = 64 * 1024
MIN_FREE_BYTES = 10 * 1024**3
MAX_RUNTIME_SECONDS = 120 * 60
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
EXPECTED_HEAD = "31e20db54eef22b12232915e7f147497914f1362"
EVIDENCE_RELATIVE = Path(
    "outputs/calibration/"
    "r1_digest_contract_v1_engineering_r1s4b-switch-s0-512-20261031-a1")
EVIDENCE_RUN_ID = "r1s4b-switch-s0-512-20261031-a1"
EVIDENCE_AUTHORIZATION_ID = "R1-S4B-20261031-USER-GO"
EVENT_COUNT = 512
QUERY_COUNT = 17
PATCH_COUNT = 32
PATCH_WIDTH = 4096
PATCH_VECTOR_SIZE = 784
AMPLITUDE = 0.20
K_NEIGHBORS = 5
ANCHOR_CAPACITY = 256
PRIMARY_RADIUS = 0.10
PRIMARY_LAMBDA = 0.10
RADII = (0.02, 0.05, 0.10, 0.20, 0.40)
LAMBDAS = (0.10, 1.00)
CONFIGURATION_SCHEMA = "calibration.r2_perf1.frozen_method_grid.v1"
SCORE_TABLE_SCHEMA = "calibration.r2_perf1.stage_a_scores.v1"
SEALED_SCORE_SCHEMA = "calibration.r2_perf1.stage_a_seal.v1"

OUTPUT_FILES = frozenset({
    "run_manifest.json",
    "stage_a_scores.jsonl",
    "stage_a_c2_parity.json",
    "stage_a_diagnostics.json",
    "stage_a_sealed.json",
    "stage_b_metrics.json",
    "stage_b_sensitivity.json",
    "stage_b_diagnostics.json",
    "stage_b_report.json",
    "orchestrator_complete.json",
    "supervisor_status.json",
    "failure.json",
})

SOURCE_FILES = (
    "experiments/calibration/r2_perf1_common.py",
    "experiments/calibration/r2_perf1_local_correction_eval.py",
    "experiments/calibration/r2_perf1_stage_a.py",
    "experiments/calibration/r2_perf1_stage_b.py",
    "experiments/calibration/r1_matched_correction_pilot.py",
    "experiments/calibration/r1_runtime_supervisor.py",
    "experiments/calibration/r2_0_label_blind_mechanism_audit.py",
    "experiments/calibration/r2_l1_locality_kernels.py",
    "experiments/calibration/query_tape.py",
    "experiments/calibration/stage05.py",
    "experiments/risk_safe/preflight.py",
)


class Perf1Error(RuntimeError):
    """The R2-PERF-1 execution contract failed closed."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def source_fingerprints() -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise Perf1Error(f"Required source file is missing: {relative}")
        output[relative] = sha256_file(path)
    return output


def verify_source_fingerprints(expected: Mapping[str, str]) -> None:
    observed = source_fingerprints()
    if dict(expected) != observed:
        changed = sorted(set(expected) | set(observed))
        changed = [name for name in changed if expected.get(name) != observed.get(name)]
        raise Perf1Error("Source fingerprints changed during the run: " + ", ".join(changed))


def git_identity() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True,
                stderr=subprocess.STDOUT).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise Perf1Error(f"Cannot verify repository identity ({args[0]}): {exc}") from exc

    return {"branch": run("branch", "--show-current"), "head": run("rev-parse", "HEAD")}


def require_expected_repository() -> dict[str, str]:
    identity = git_identity()
    if identity["branch"] != EXPECTED_BRANCH or identity["head"] != EXPECTED_HEAD:
        raise Perf1Error(
            f"Repository identity mismatch: expected {EXPECTED_BRANCH}@{EXPECTED_HEAD}; "
            f"found {identity['branch']}@{identity['head']}")
    return identity


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    if os.name == "nt" and path.exists():
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        attrs = kernel32.GetFileAttributesW(str(path))
        if attrs == 0xFFFFFFFF:
            raise Perf1Error(f"Cannot inspect path attributes: {path}")
        return bool(attrs & 0x400)
    return False


def require_plain_path(path: Path, *, must_exist: bool) -> Path:
    absolute = Path(os.path.abspath(path))
    for component in reversed((absolute, *absolute.parents)):
        if component.exists() and _is_reparse_point(component):
            raise Perf1Error(f"Symlink/reparse-point path is refused: {component}")
    if must_exist and not absolute.exists():
        raise Perf1Error(f"Required path is missing: {absolute}")
    return absolute


def new_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "_" + os.urandom(4).hex()


def output_path(run_id: str) -> Path:
    if re.fullmatch(r"[0-9]{8}T[0-9]{6}_[0-9a-f]{8}", run_id) is None:
        raise Perf1Error("Run ID is not a unique timestamp/random-token identity")
    return OUTPUT_PARENT / f"{OUTPUT_PREFIX}{run_id}"


def create_output_namespace(run_id: str) -> Path:
    parent = require_plain_path(OUTPUT_PARENT, must_exist=True)
    if not parent.is_dir():
        raise Perf1Error("Output parent is not a directory")
    free_bytes = int(shutil.disk_usage(parent).free)
    if free_bytes < MIN_FREE_BYTES:
        raise Perf1Error(
            f"Output preflight has {free_bytes} free bytes; {MIN_FREE_BYTES} are required")
    root = output_path(run_id)
    require_plain_path(root, must_exist=False)
    root.mkdir(exist_ok=False)
    return root


def _mutex_name(root: Path) -> str:
    key = os.path.normcase(str(root.resolve(strict=True)))
    return "Local\\R2Perf1OutputBudget-" + sha256_bytes(key.encode("utf-8"))


@contextmanager
def _cross_process_write_guard(root: Path) -> Iterator[None]:
    if os.name != "nt":
        raise Perf1Error("The real R2-PERF-1 output writer requires the Windows mutex backend")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.CreateMutexW(None, False, _mutex_name(root))
    if not handle:
        raise Perf1Error("CreateMutexW failed for the output budget")
    acquired = False
    try:
        status = kernel32.WaitForSingleObject(handle, 0)
        if status not in (0x00000000, 0x00000080):
            raise Perf1Error("Concurrent output writer detected")
        acquired = True
        yield
    finally:
        if handle and acquired:
            try:
                kernel32.ReleaseMutex(handle)
            finally:
                kernel32.CloseHandle(handle)
        elif handle:
            kernel32.CloseHandle(handle)


class OutputBudget:
    """Flat, fail-closed, pre-write byte accountant for one unique run."""

    def __init__(self, root: Path, *, allow_existing: bool,
                 max_bytes: int = MAX_OUTPUT_BYTES,
                 failure_reserve_bytes: int = FAILURE_RESERVE_BYTES):
        self.root = require_plain_path(Path(root), must_exist=allow_existing)
        self.max_bytes = int(max_bytes)
        self.failure_reserve_bytes = int(failure_reserve_bytes)
        if (self.root.parent.resolve(strict=True) != OUTPUT_PARENT.resolve(strict=True)
                or not self.root.name.startswith(OUTPUT_PREFIX)):
            raise Perf1Error("Output root is outside its exact versioned namespace")
        if not 0 < self.failure_reserve_bytes < self.max_bytes:
            raise Perf1Error("Output byte bound/reserve is invalid")
        if self.root.exists() and not allow_existing:
            raise FileExistsError(f"Refusing to reuse output namespace: {self.root}")
        if not self.root.exists():
            self.root.mkdir(exist_ok=False)
        if not self.root.is_dir():
            raise Perf1Error("Output namespace is not a plain directory")
        self._thread_lock = threading.Lock()

    @property
    def normal_limit(self) -> int:
        return self.max_bytes - self.failure_reserve_bytes

    def _audit(self) -> int:
        require_plain_path(self.root, must_exist=True)
        total = 0
        for entry in os.scandir(self.root):
            path = Path(entry.path)
            if _is_reparse_point(path):
                raise Perf1Error(f"Output contains a symlink/reparse point: {path.name}")
            if not entry.is_file(follow_symlinks=False):
                raise Perf1Error(f"Output contains an unaccounted directory or device: {path.name}")
            if path.name not in OUTPUT_FILES:
                raise Perf1Error(f"Output contains an unaccounted file: {path.name}")
            stat = entry.stat(follow_symlinks=False)
            if getattr(stat, "st_nlink", 1) > 1:
                raise Perf1Error(f"Output contains a multiply-linked file: {path.name}")
            total += int(stat.st_size)
        return total

    def write(self, name: str, payload: bytes, *, append: bool = False,
              failure_reserved: bool = False) -> int:
        if name not in OUTPUT_FILES or Path(name).name != name:
            raise Perf1Error(f"Output filename is outside the audited allowlist: {name}")
        if not isinstance(payload, bytes):
            raise TypeError("Budgeted output payload must be bytes")
        if not self._thread_lock.acquire(blocking=False):
            raise Perf1Error("Concurrent output writer detected in this process")
        try:
            with _cross_process_write_guard(self.root):
                current = self._audit()
                if current > self.max_bytes:
                    raise Perf1Error("Existing output exceeds its authorized byte limit")
                target = self.root / name
                if _is_reparse_point(target):
                    raise Perf1Error("Output target cannot be a reparse point")
                exists = target.exists()
                if exists and not append:
                    raise FileExistsError(f"Refusing to overwrite output file: {name}")
                if append and exists and not target.is_file():
                    raise Perf1Error("Append target is not a regular file")
                limit = self.max_bytes if failure_reserved else self.normal_limit
                if current + len(payload) > limit:
                    raise Perf1Error(f"Write refused before I/O: {name} exceeds byte budget")
                flags = os.O_CREAT | os.O_WRONLY | (os.O_APPEND if append else os.O_EXCL)
                if hasattr(os, "O_BINARY"):
                    flags |= os.O_BINARY
                fd = os.open(target, flags, 0o600)
                try:
                    view = memoryview(payload)
                    offset = 0
                    while offset < len(view):
                        written = os.write(fd, view[offset:])
                        if written <= 0:
                            raise OSError("short write to budgeted output")
                        offset += written
                    os.fsync(fd)
                finally:
                    os.close(fd)
                return current + len(payload)
        finally:
            self._thread_lock.release()

    def write_json(self, name: str, value: Any, *, append: bool = False,
                   failure_reserved: bool = False) -> int:
        payload = (canonical_json(value) + "\n").encode("utf-8")
        return self.write(name, payload, append=append,
                          failure_reserved=failure_reserved)


def require_worker(stage: str) -> None:
    """Require the exact interpreter, CPU visibility, and supervised job."""
    expected_python = Path(r"D:\anaconda3\envs\torch\python.exe")
    if os.path.normcase(str(Path(sys.executable).resolve())) != os.path.normcase(
            str(expected_python.resolve())):
        raise Perf1Error(f"Wrong Python interpreter: {sys.executable}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise Perf1Error("Worker must start with CUDA_VISIBLE_DEVICES set to an empty string")
    if os.environ.get("R2_PERF1_WORKER") != stage:
        raise Perf1Error(f"Worker entry point is not authorized for stage {stage}")
    nonce = os.environ.get("R2_PERF1_SUPERVISED_NONCE", "")
    if re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        raise Perf1Error("Worker lacks its supervisor-bound launch nonce")
    require_expected_repository()
    if os.name != "nt":
        raise Perf1Error("R2-PERF-1 workers require the Windows Job Object supervisor")
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job
    if not current_process_is_in_job():
        raise Perf1Error("Worker is not contained in the 120-minute Windows Job Object")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
