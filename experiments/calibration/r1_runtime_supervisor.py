"""Windows Job Object supervisor for explicitly authorized R1 child runs.

The generic child-process entry points are used by synthetic CPU tests. The
R1-specific CLI requires an explicit authorization switch and authorization ID.
No experiment is launched at import time.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Mapping, Sequence


MAX_RUNTIME_SECONDS = 120 * 60
TERMINATION_GRACE_SECONDS = 30.0
SUPERVISOR_SCHEMA = "calibration.r1_runtime_supervisor.v1"
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
CREATE_SUSPENDED = 0x00000004
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_UNICODE_ENVIRONMENT = 0x00000400
CREATE_NO_WINDOW = 0x08000000
STARTF_USESHOWWINDOW = 0x00000001
STARTF_USESTDHANDLES = 0x00000100
WAIT_OBJECT_0 = 0x00000000
WAIT_ABANDONED = 0x00000080
WAIT_TIMEOUT = 0x00000102
STILL_ACTIVE = 259


class SupervisorError(RuntimeError):
    pass


class SupervisorBlockingFailure(SupervisorError):
    """The supervisor could not confirm that its child process tree exited."""


@dataclass(frozen=True)
class SupervisorResult:
    status: str
    child_pid: int
    child_exit_code: int | None
    elapsed_seconds: float
    timeout_seconds: float
    termination_confirmed: bool
    active_job_processes: int


def _require_windows() -> None:
    if os.name != "nt":
        raise SupervisorError("R1 process-tree containment requires Windows Job Objects")


def current_process_is_in_job() -> bool:
    """Return whether this process belongs to a Windows Job Object."""
    _require_windows()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.POINTER(ctypes.c_int)]
    kernel32.IsProcessInJob.restype = ctypes.c_int
    value = ctypes.c_int(0)
    current = kernel32.GetCurrentProcess()
    if not kernel32.IsProcessInJob(current, None, ctypes.byref(value)):
        raise SupervisorError("IsProcessInJob failed")
    return bool(value.value)


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", ctypes.c_uint32),
        ("TotalProcesses", ctypes.c_uint32),
        ("ActiveProcesses", ctypes.c_uint32),
        ("TotalTerminatedProcesses", ctypes.c_uint32),
    ]


class _STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("lpReserved", ctypes.c_wchar_p),
        ("lpDesktop", ctypes.c_wchar_p),
        ("lpTitle", ctypes.c_wchar_p),
        ("dwX", ctypes.c_uint32), ("dwY", ctypes.c_uint32),
        ("dwXSize", ctypes.c_uint32), ("dwYSize", ctypes.c_uint32),
        ("dwXCountChars", ctypes.c_uint32), ("dwYCountChars", ctypes.c_uint32),
        ("dwFillAttribute", ctypes.c_uint32), ("dwFlags", ctypes.c_uint32),
        ("wShowWindow", ctypes.c_uint16), ("cbReserved2", ctypes.c_uint16),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", ctypes.c_void_p), ("hStdOutput", ctypes.c_void_p),
        ("hStdError", ctypes.c_void_p),
    ]


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", ctypes.c_void_p), ("hThread", ctypes.c_void_p),
                ("dwProcessId", ctypes.c_uint32), ("dwThreadId", ctypes.c_uint32)]


class _SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("nLength", ctypes.c_uint32), ("lpSecurityDescriptor", ctypes.c_void_p),
                ("bInheritHandle", ctypes.c_int)]


class _WindowsJob:
    def __init__(self):
        _require_windows()
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        self.kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        self.kernel32.SetInformationJobObject.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        self.kernel32.SetInformationJobObject.restype = ctypes.c_int
        self.kernel32.QueryInformationJobObject.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32)]
        self.kernel32.QueryInformationJobObject.restype = ctypes.c_int
        self.kernel32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.kernel32.TerminateJobObject.restype = ctypes.c_int
        self.kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self.kernel32.CloseHandle.restype = ctypes.c_int
        self.handle = self.kernel32.CreateJobObjectW(None, None)
        if not self.handle:
            raise SupervisorError("CreateJobObjectW failed")
        limits = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel32.SetInformationJobObject(
                self.handle, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise SupervisorError("Could not set KILL_ON_JOB_CLOSE on the child job")

    def assign(self, process_handle) -> None:
        self.kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.kernel32.AssignProcessToJobObject.restype = ctypes.c_int
        if not self.kernel32.AssignProcessToJobObject(self.handle, process_handle):
            raise SupervisorError("AssignProcessToJobObject failed; child was not resumed")

    def active_processes(self) -> int:
        info = _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        returned = ctypes.c_uint32(0)
        if not self.kernel32.QueryInformationJobObject(
                self.handle, JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
                ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(returned)):
            raise SupervisorError("QueryInformationJobObject failed")
        return int(info.ActiveProcesses)

    def terminate(self, exit_code: int = 0xE0000001) -> None:
        if not self.kernel32.TerminateJobObject(self.handle, exit_code):
            raise SupervisorBlockingFailure("TerminateJobObject failed")

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.kernel32.CloseHandle(self.handle)
            self.handle = None


def _create_child_suspended(command: Sequence[str], env: Mapping[str, str],
                             cwd: Path | None, job: _WindowsJob):
    kernel32 = job.kernel32
    kernel32.CreateProcessW.argtypes = [
        ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_wchar_p,
        ctypes.POINTER(_STARTUPINFOW), ctypes.POINTER(_PROCESS_INFORMATION)]
    kernel32.CreateProcessW.restype = ctypes.c_int
    kernel32.CreateFileW.argtypes = [
        ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.POINTER(_SECURITY_ATTRIBUTES), ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_void_p]
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.ResumeThread.argtypes = [ctypes.c_void_p]
    kernel32.ResumeThread.restype = ctypes.c_uint32
    kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.TerminateProcess.restype = ctypes.c_int
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]

    security = _SECURITY_ATTRIBUTES(ctypes.sizeof(_SECURITY_ATTRIBUTES), None, 1)
    null_handle = kernel32.CreateFileW(
        "NUL", 0xC0000000, 0x00000003, ctypes.byref(security), 3, 0x00000080, None)
    if not null_handle or null_handle == ctypes.c_void_p(-1).value:
        raise SupervisorError("Could not open inherited NUL handles for child output")
    startup = _STARTUPINFOW()
    startup.cb = ctypes.sizeof(startup)
    startup.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES
    startup.wShowWindow = 0
    startup.hStdInput = null_handle
    startup.hStdOutput = null_handle
    startup.hStdError = null_handle
    info = _PROCESS_INFORMATION()
    env_block = "\0".join(f"{key}={value}" for key, value in sorted(env.items(), key=lambda kv: kv[0].upper())) + "\0\0"
    env_buffer = ctypes.create_unicode_buffer(env_block)
    command_buffer = ctypes.create_unicode_buffer(subprocess.list2cmdline(list(command)))
    creation_flags = (CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP |
                      CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW)
    try:
        ok = kernel32.CreateProcessW(
            None, command_buffer, None, None, True, creation_flags,
            ctypes.cast(env_buffer, ctypes.c_void_p), str(cwd) if cwd else None,
            ctypes.byref(startup), ctypes.byref(info))
    finally:
        kernel32.CloseHandle(null_handle)
    if not ok:
        raise SupervisorError("CreateProcessW failed")
    try:
        job.assign(info.hProcess)
        if kernel32.ResumeThread(info.hThread) == 0xFFFFFFFF:
            raise SupervisorError("ResumeThread failed")
    except BaseException:
        kernel32.TerminateProcess(info.hProcess, 0xE0000002)
        kernel32.WaitForSingleObject(info.hProcess, 5000)
        kernel32.CloseHandle(info.hThread)
        kernel32.CloseHandle(info.hProcess)
        raise
    kernel32.CloseHandle(info.hThread)
    return info.hProcess, int(info.dwProcessId)


def run_supervised_child(command: Sequence[str], *,
                         timeout_seconds: float = MAX_RUNTIME_SECONDS,
                         env: Mapping[str, str] | None = None,
                         cwd: Path | None = None,
                         output_dir: Path | None = None,
                         max_output_bytes: int = 2 * 1024**3,
                         failure_reserve_bytes: int = 64 * 1024,
                         allowed_output_parent: Path | None = None) -> SupervisorResult:
    """Run a command suspended-in-job and kill its complete tree at deadline."""
    _require_windows()
    if not command or any(not isinstance(value, str) or not value for value in command):
        raise SupervisorError("Child command must be a nonempty argument list")
    if not 0 < float(timeout_seconds) <= MAX_RUNTIME_SECONDS:
        raise SupervisorError("Child deadline must be positive and no greater than 120 minutes")
    if output_dir is not None:
        output_dir = Path(os.path.abspath(output_dir))
        if output_dir.exists():
            raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
        if allowed_output_parent is not None:
            parent = Path(allowed_output_parent).resolve(strict=True)
            if output_dir.parent.resolve(strict=True) != parent:
                raise SupervisorError("Output directory is outside its authorized parent")
    child_env = dict(os.environ if env is None else env)
    job = _WindowsJob()
    process_handle = None
    pid = -1
    start = time.monotonic()
    timed_out = False
    descendants_terminated = False
    termination_confirmed = False
    blocking_failure: str | None = None
    exit_code: int | None = None
    active = 0
    try:
        process_handle, pid = _create_child_suspended(
            command, child_env, Path(cwd) if cwd else None, job)
        kernel32 = job.kernel32
        remaining = max(0.0, float(timeout_seconds) - (time.monotonic() - start))
        wait_ms = min(0xFFFFFFFE, int(remaining * 1000))
        wait_result = kernel32.WaitForSingleObject(process_handle, wait_ms)
        if wait_result == WAIT_TIMEOUT:
            timed_out = True
            try:
                job.terminate()
            except SupervisorBlockingFailure:
                # Keep checking until the bounded grace ends. The job's
                # KILL_ON_JOB_CLOSE setting remains a final termination guard.
                pass
        elif wait_result != WAIT_OBJECT_0:
            raise SupervisorError(f"Unexpected child wait result 0x{wait_result:08x}")

        deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
        exited_with_children_deadline = None
        while True:
            process_state = kernel32.WaitForSingleObject(process_handle, 0)
            active = job.active_processes()
            if process_state == WAIT_OBJECT_0 and active == 0:
                termination_confirmed = True
                break
            if not timed_out and process_state == WAIT_OBJECT_0 and active > 0:
                if exited_with_children_deadline is None:
                    exited_with_children_deadline = time.monotonic() + 1.0
                elif time.monotonic() >= exited_with_children_deadline:
                    descendants_terminated = True
                    job.terminate()
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        if not termination_confirmed:
            blocking_failure = (
                f"Could not confirm child tree exit (process wait={process_state}, active={active})")
        else:
            code = ctypes.c_uint32(0)
            if not kernel32.GetExitCodeProcess(process_handle, ctypes.byref(code)):
                raise SupervisorError("GetExitCodeProcess failed after confirmed exit")
            exit_code = int(code.value)
    finally:
        # Closing the final Job handle enforces KILL_ON_JOB_CLOSE. Keep the
        # process handle open long enough to observe a final root-process exit.
        job.close()
        if process_handle is not None:
            if not termination_confirmed:
                job.kernel32.WaitForSingleObject(
                    process_handle, int(TERMINATION_GRACE_SECONDS * 1000))
            job.kernel32.CloseHandle(process_handle)

    elapsed = time.monotonic() - start
    if blocking_failure is not None:
        status = "BLOCKING_TERMINATION_FAILURE"
    elif timed_out:
        status = "TIMED_OUT_TERMINATED"
    elif descendants_terminated:
        status = "FAILED_DESCENDANTS_TERMINATED"
    elif exit_code == 0:
        status = "COMPLETED"
    else:
        status = "FAILED_NONZERO_EXIT"
    result = SupervisorResult(status, pid, exit_code, elapsed,
                              float(timeout_seconds), termination_confirmed, active)
    persist_error: Exception | None = None
    if output_dir is not None:
        from experiments.calibration.r1_matched_correction_pilot import OutputBudget
        try:
            budget = OutputBudget(
                output_dir, max_bytes=max_output_bytes,
                failure_reserve_bytes=failure_reserve_bytes,
                allowed_parent=allowed_output_parent, allow_existing=output_dir.exists())
            try:
                budget.write_supervisor_record({
                    "schema": SUPERVISOR_SCHEMA,
                    **asdict(result),
                    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "automatic_restart": False,
                    "process_tree_containment": "Windows Job Object, suspended assignment, kill-on-close",
                })
            finally:
                budget.close()
        except Exception as exc:
            persist_error = exc
    if blocking_failure is not None:
        if persist_error is not None:
            blocking_failure += f"; supervisor failure metadata could not be persisted: {persist_error}"
        raise SupervisorBlockingFailure(blocking_failure)
    if persist_error is not None:
        raise SupervisorError(f"Could not persist supervisor result: {persist_error}") from persist_error
    return result


def supervise_r1_pilot(*, expected_source_commit: str, profile_id: str,
                       run_id: str, authorization_id: str,
                       authorized: bool,
                       support_only: bool = False,
                       dataset_root: Path | None = None,
                       json_root: Path | None = None) -> SupervisorResult:
    """Perform source/branch checks, then launch one explicitly authorized R1 run."""
    if authorized is not True:
        raise SupervisorError("Explicit --authorize-real-execution is required")
    from experiments.calibration.r1_matched_correction_pilot import (
        DEFAULT_DATASET_ROOT, DEFAULT_JSON_ROOT, OUTPUT_NAMESPACE_PREFIX,
        ROOT, SUPPORT_ONLY_PROFILE_ID, R1_PROFILE_ID,
        _output_namespace, _validate_branch_and_start,
        _validate_run_identity_fields,
    )
    _validate_run_identity_fields(run_id, authorization_id)
    expected_profile = SUPPORT_ONLY_PROFILE_ID if support_only else R1_PROFILE_ID
    if profile_id != expected_profile:
        raise SupervisorError(f"Incompatible R1 profile for selected mode: {profile_id}")
    _validate_branch_and_start(expected_source_commit=expected_source_commit,
                               profile_id=profile_id, support_only=support_only)
    output = _output_namespace(run_id, support_only=support_only)
    if output.exists():
        raise FileExistsError(f"R1 output namespace exists; automatic restart is forbidden: {output}")
    run_module = ROOT / "experiments" / "calibration" / "r1_matched_correction_pilot.py"
    command = [sys.executable, str(run_module),
               "--dataset-root", str(dataset_root or DEFAULT_DATASET_ROOT),
               "--json-root", str(json_root or DEFAULT_JSON_ROOT),
               "--expected-source-commit", expected_source_commit,
               "--profile-id", profile_id, "--run-id", run_id,
               "--authorization-id", authorization_id]
    if support_only:
        command.append("--support-only")
    child_env = dict(os.environ)
    child_env.update({
        "R1_PILOT_SUPERVISED_CHILD": "1",
        "R1_PILOT_SUPERVISOR_AUTHORIZATION_ID": authorization_id,
        "R1_PILOT_SUPERVISOR_NONCE": os.urandom(32).hex(),
    })
    return run_supervised_child(
        command, timeout_seconds=MAX_RUNTIME_SECONDS, env=child_env, cwd=ROOT,
        output_dir=output, max_output_bytes=2 * 1024**3,
        allowed_output_parent=ROOT / "outputs" / "calibration")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--authorize-real-execution", action="store_true")
    parser.add_argument("--support-only", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--json-root", type=Path)
    args = parser.parse_args(argv)
    try:
        result = supervise_r1_pilot(
            expected_source_commit=args.expected_source_commit,
            profile_id=args.profile_id, run_id=args.run_id,
            authorization_id=args.authorization_id,
            authorized=args.authorize_real_execution,
            support_only=args.support_only, dataset_root=args.dataset_root,
            json_root=args.json_root)
        print(json.dumps(asdict(result), sort_keys=True, indent=2))
        return 0 if result.status == "COMPLETED" else 1
    except Exception as exc:
        print(f"R1_SUPERVISOR_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
