"""Synthetic CPU checks for R1 runtime and output containment."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest
import torch

from experiments.calibration.r1_matched_correction_pilot import (
    EvidenceTapeWriter,
    MAX_OUTPUT_BYTES,
    MIN_OUTPUT_FREE_BYTES,
    OUTPUT_FAILURE_RESERVE_BYTES,
    OutputBudget,
    OutputBudgetError,
    R1BlockedInput,
    R1ResourceLimit,
    _atomic_json,
    _append_jsonl,
    _require_minimum_output_free_space,
    _write_once,
    _require_supervised_authorization,
    common_c0_greedy_writer_action,
)
from experiments.calibration.r1_runtime_supervisor import (
    MAX_RUNTIME_SECONDS,
    SupervisorBlockingFailure,
    SupervisorError,
    SupervisorResult,
    run_supervised_child,
    supervise_r1_pilot,
)
import experiments.calibration.r1_runtime_supervisor as runtime_supervisor


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects are required")
ROOT = Path(__file__).resolve().parents[1]


def _child_with_budget(body: str) -> str:
    prefix = textwrap.dedent(f"""
        import os, sys
        from pathlib import Path
        sys.path.insert(0, {str(ROOT)!r})
        from experiments.calibration.r1_matched_correction_pilot import OutputBudget
        root = Path(os.environ['R1_TEST_OUTPUT_DIR'])
        budget = OutputBudget(root, max_bytes=1024, failure_reserve_bytes=256)
        budget.ensure_root()
    """).strip()
    return prefix + "\n" + textwrap.dedent(body).strip() + "\nbudget.close()\n"


def _supervise(command: list[str], output: Path, timeout: float = 5.0):
    env = dict(os.environ)
    env["R1_TEST_OUTPUT_DIR"] = str(output)
    return run_supervised_child(
        command, timeout_seconds=timeout, env=env, cwd=ROOT,
        output_dir=output, max_output_bytes=1024,
        failure_reserve_bytes=256)


def _pid_is_alive(pid: int) -> bool:
    import ctypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(0x00100000 | 0x1000, False, pid)
    if not handle:
        return ctypes.get_last_error() not in (87, 1168)
    try:
        return kernel32.WaitForSingleObject(handle, 0) == 0x00000102
    finally:
        kernel32.CloseHandle(handle)


def test_output_budget_refuses_before_write_and_preserves_failure_reserve(tmp_path):
    root = tmp_path / "bounded"
    budget = OutputBudget(root, max_bytes=256, failure_reserve_bytes=64)
    budget.ensure_root()
    manifest = root / "run_manifest.json"
    budget.write_once(manifest, b"x" * 160)
    with pytest.raises(OutputBudgetError, match="before I/O"):
        budget.append(manifest, b"y" * 33)
    assert manifest.stat().st_size == 160
    assert sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) == 160
    budget.write_supervisor_record({"status": "synthetic failure"})
    total = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    assert total <= 256
    assert json.loads((root / "supervisor_status.json").read_text())[
        "status"] == "synthetic failure"
    budget.close()


def test_event_npz_and_all_jsonl_classes_are_preflighted(tmp_path):
    root = tmp_path / "npz-budget"
    budget = OutputBudget(root, max_bytes=2048, failure_reserve_bytes=256)
    tape = EvidenceTapeWriter(root, max_bytes=2048, budget=budget)
    arrays = {
        "base_patch_map": np.linspace(0, 1, 784, dtype=np.float32),
        "top32_ids": np.arange(32, dtype=np.int64),
        "top32_features": np.ones((32, 2), dtype=np.float32),
        "c1_delta": np.zeros(32, dtype=np.float32),
        "c2_delta": np.zeros(32, dtype=np.float32),
    }
    expected_score = float(arrays["base_patch_map"].max())
    metadata = {"event_id": "synthetic", "category": "synthetic",
                "official_group_sha256": "a" * 64,
                "c0_score": expected_score, "c1_score": expected_score,
                "c2_score": expected_score}
    before = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    with pytest.raises(OutputBudgetError, match="before I/O"):
        tape.seal_event(0, arrays, metadata)
    assert not (root / "events" / "event_000000.npz").exists()
    assert not (root / "sealed_scores.jsonl").exists()
    after = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    assert after == before <= budget.normal_limit
    budget.close()

    small_root = tmp_path / "jsonl-budget"
    small = OutputBudget(small_root, max_bytes=1024, failure_reserve_bytes=128)
    small.ensure_root()
    _append_jsonl(small_root / "sealed_scores.jsonl", {"kind": "public-score"})
    _append_jsonl(small_root / "event_transitions.jsonl", {"kind": "transition"})
    _append_jsonl(small_root / "private_feedback.jsonl", {"kind": "private-feedback"})
    _append_jsonl(small_root / "correction_anchor_events.jsonl", {"kind": "anchor"})
    total = sum(path.stat().st_size for path in small_root.rglob("*") if path.is_file())
    assert total <= small.normal_limit
    small.close()


def test_output_budget_fails_closed_on_unaccounted_invalid_and_concurrent_writes(tmp_path):
    root = tmp_path / "guarded"
    budget = OutputBudget(root, max_bytes=512, failure_reserve_bytes=64)
    budget.ensure_root()
    (root / "unaccounted.bin").write_bytes(b"rogue")
    with pytest.raises(OutputBudgetError, match="allowlist"):
        budget.write_once(root / "run_manifest.json", b"{}")
    (root / "unaccounted.bin").unlink()

    with pytest.raises(OutputBudgetError, match="outside"):
        budget.write_once(tmp_path / "outside.bin", b"x")
    assert not (tmp_path / "outside.bin").exists()

    assert budget._write_lock.acquire(blocking=False)
    try:
        with pytest.raises(OutputBudgetError, match="Concurrent writer"):
            budget.write_once(root / "run_manifest.json", b"{}")
    finally:
        budget._write_lock.release()
    assert not (root / "run_manifest.json").exists()
    budget.close()


def test_output_budget_mutex_rejects_a_second_process_writer(tmp_path):
    root = tmp_path / "cross-process"
    budget = OutputBudget(root, max_bytes=512, failure_reserve_bytes=64)
    budget.ensure_root()
    release = budget._acquire_os_mutex()
    try:
        script = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from pathlib import Path
from experiments.calibration.r1_matched_correction_pilot import OutputBudget, OutputBudgetError
root = Path({str(root)!r})
budget = OutputBudget(root, max_bytes=512, failure_reserve_bytes=64, allow_existing=True)
try:
    budget.write_once(root / 'run_manifest.json', b'{{}}')
except OutputBudgetError:
    raise SystemExit(42)
raise SystemExit(0)
"""
        completed = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                   capture_output=True, check=False, timeout=15)
    finally:
        release()
    assert completed.returncode == 42
    assert not (root / "run_manifest.json").exists()
    budget.close()


def test_output_budget_rejects_reused_and_wrong_namespace(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    with pytest.raises(FileExistsError):
        OutputBudget(root, max_bytes=256, failure_reserve_bytes=64)
    allowed = tmp_path / "not-r1"
    with pytest.raises(OutputBudgetError, match="namespace prefix"):
        OutputBudget(allowed, max_bytes=256, failure_reserve_bytes=64,
                     allowed_parent=tmp_path)


def test_budgeted_r1_write_wrappers_fail_closed_without_matching_budget(tmp_path, monkeypatch):
    import experiments.calibration.r1_matched_correction_pilot as pilot

    monkeypatch.setattr(pilot, "_ACTIVE_OUTPUT_BUDGETS", {})
    targets = [tmp_path / "no-budget" / name for name in (
        "public.jsonl", "private.bin", "run_manifest.json")]
    with pytest.raises(OutputBudgetError, match="without an active OutputBudget"):
        _append_jsonl(targets[0], {"synthetic": True})
    with pytest.raises(OutputBudgetError, match="without an active OutputBudget"):
        _write_once(targets[1], b"synthetic")
    with pytest.raises(OutputBudgetError, match="without an active OutputBudget"):
        _atomic_json(targets[2], {"synthetic": True})
    assert not (tmp_path / "no-budget").exists()

    root = tmp_path / "active-budget"
    budget = OutputBudget(root, max_bytes=1024, failure_reserve_bytes=128)
    budget.ensure_root()
    outside_targets = [tmp_path / "escape" / name for name in (
        "public.jsonl", "private.bin", "run_manifest.json")]
    with pytest.raises(OutputBudgetError, match="outside every active OutputBudget"):
        _append_jsonl(outside_targets[0], {"synthetic": True})
    with pytest.raises(OutputBudgetError, match="outside every active OutputBudget"):
        _write_once(outside_targets[1], b"synthetic")
    with pytest.raises(OutputBudgetError, match="outside every active OutputBudget"):
        _atomic_json(outside_targets[2], {"synthetic": True})
    assert not (tmp_path / "escape").exists()
    budget.close()


def test_output_free_space_preflight_uses_ten_gib_threshold_and_fails_closed(tmp_path):
    enough = lambda _path: SimpleNamespace(free=MIN_OUTPUT_FREE_BYTES)
    too_little = lambda _path: SimpleNamespace(free=MIN_OUTPUT_FREE_BYTES - 1)
    assert _require_minimum_output_free_space(
        tmp_path, disk_usage_fn=enough) == MIN_OUTPUT_FREE_BYTES
    with pytest.raises(R1ResourceLimit, match="at least"):
        _require_minimum_output_free_space(tmp_path, disk_usage_fn=too_little)

    def unavailable(_path):
        raise OSError("synthetic disk query failure")

    with pytest.raises(R1BlockedInput, match="Cannot verify free space"):
        _require_minimum_output_free_space(tmp_path, disk_usage_fn=unavailable)


def test_application_storage_limits_and_runtime_deadline_remain_fixed():
    assert MAX_RUNTIME_SECONDS == 120 * 60
    assert MAX_OUTPUT_BYTES == 2 * 1024**3
    assert OUTPUT_FAILURE_RESERVE_BYTES == 64 * 1024


def test_output_budget_and_free_space_preflight_reject_reparse_ancestors(
        tmp_path, monkeypatch):
    import experiments.calibration.r1_matched_correction_pilot as pilot

    root = tmp_path / "r1_digest_contract_v1_engineering_reparse-test"
    monkeypatch.setattr(pilot, "_path_is_reparse_point",
                        lambda path: Path(path) == tmp_path)
    with pytest.raises(OutputBudgetError, match="reparse point"):
        OutputBudget(root, max_bytes=1024, failure_reserve_bytes=128)
    with pytest.raises(R1BlockedInput, match="symlink/reparse path"):
        _require_minimum_output_free_space(tmp_path, disk_usage_fn=lambda _path: None)


def test_supervisor_checks_disk_space_before_child_launch_and_rejects_reuse(
        tmp_path, monkeypatch):
    import experiments.calibration.r1_matched_correction_pilot as pilot

    output = tmp_path / "r1_digest_contract_v1_engineering_unique-test"
    monkeypatch.setattr(pilot, "_validate_run_identity_fields", lambda *_args: None)
    monkeypatch.setattr(pilot, "_validate_branch_and_start", lambda **_kwargs: {})
    monkeypatch.setattr(pilot, "_output_namespace", lambda *_args, **_kwargs: output)
    monkeypatch.setattr(
        pilot.shutil, "disk_usage",
        lambda _path: SimpleNamespace(free=MIN_OUTPUT_FREE_BYTES - 1))
    launched = []
    monkeypatch.setattr(runtime_supervisor, "run_supervised_child",
                        lambda *args, **kwargs: launched.append((args, kwargs)))

    with pytest.raises(R1ResourceLimit, match="Insufficient output-volume space"):
        supervise_r1_pilot(
            expected_source_commit="a" * 40,
            profile_id=pilot.R1_PROFILE_ID,
            run_id="synthetic-low-space", authorization_id="synthetic-authorization",
            authorized=True)
    assert launched == []
    assert not output.exists()

    output.mkdir()
    with pytest.raises(FileExistsError, match="automatic restart is forbidden"):
        supervise_r1_pilot(
            expected_source_commit="a" * 40,
            profile_id=pilot.R1_PROFILE_ID,
            run_id="synthetic-low-space", authorization_id="synthetic-authorization",
            authorized=True)
    assert launched == []


def test_job_object_contains_child_and_returns_on_normal_exit(tmp_path):
    output = tmp_path / "normal-child"
    script = "from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job; assert current_process_is_in_job()"
    result = _supervise([sys.executable, "-c", script], output)
    assert result.status == "COMPLETED"
    assert result.child_exit_code == 0
    assert result.termination_confirmed
    assert result.active_job_processes == 0
    status = json.loads((output / "supervisor_status.json").read_text())
    assert status["automatic_restart"] is False
    assert status["process_tree_containment"].startswith("Windows Job Object")


def test_job_timeout_terminates_sleeping_child_and_preserves_metadata(tmp_path):
    output = tmp_path / "sleeping-child"
    result = _supervise([sys.executable, "-c", "import time; time.sleep(60)"],
                        output, timeout=0.25)
    assert result.status == "TIMED_OUT_TERMINATED"
    assert result.termination_confirmed
    assert result.active_job_processes == 0
    assert json.loads((output / "supervisor_status.json").read_text())["status"] == result.status


def test_job_timeout_kills_spawned_descendant_and_keeps_child_partial_output(tmp_path):
    output = tmp_path / "tree-child"
    pid_file = tmp_path / "descendant.pid"
    body = f"""
        budget.write_once(root / 'run_manifest.json', b'{{\"status\":\"partial\"}}')
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        Path({str(pid_file)!r}).write_text(str(child.pid))
        time.sleep(60)
    """
    script = "import subprocess, time\n" + _child_with_budget(body)
    result = _supervise([sys.executable, "-c", script], output, timeout=2.5)
    assert result.status == "TIMED_OUT_TERMINATED"
    assert result.termination_confirmed and result.active_job_processes == 0
    descendant_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 3.0
    while _pid_is_alive(descendant_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _pid_is_alive(descendant_pid)
    assert json.loads((output / "run_manifest.json").read_text())["status"] == "partial"
    assert (output / "supervisor_status.json").is_file()


def test_unconfirmed_termination_fails_closed_and_records_failure(tmp_path, monkeypatch):
    output = tmp_path / "unconfirmed-child"

    def refuse_termination(_job, exit_code=0xE0000001):
        raise SupervisorBlockingFailure("synthetic termination API failure")

    monkeypatch.setattr(runtime_supervisor._WindowsJob, "terminate", refuse_termination)
    monkeypatch.setattr(runtime_supervisor, "TERMINATION_GRACE_SECONDS", 0.05)
    with pytest.raises(SupervisorBlockingFailure, match="Could not confirm child tree exit"):
        _supervise([sys.executable, "-c", "import time; time.sleep(60)"],
                   output, timeout=0.25)
    status = json.loads((output / "supervisor_status.json").read_text())
    assert status["status"] == "BLOCKING_TERMINATION_FAILURE"
    assert status["termination_confirmed"] is False
    assert status["automatic_restart"] is False
    assert not _pid_is_alive(status["child_pid"])


def test_unexpected_child_failure_is_recorded_without_restart(tmp_path):
    output = tmp_path / "failing-child"
    body = "budget.write_once(root / 'run_manifest.json', b'{\"status\":\"partial\"}')\nraise SystemExit(7)"
    result = _supervise([sys.executable, "-c", _child_with_budget(body)], output)
    assert result.status == "FAILED_NONZERO_EXIT"
    assert result.child_exit_code == 7
    status = json.loads((output / "supervisor_status.json").read_text())
    assert status["automatic_restart"] is False
    assert (output / "run_manifest.json").is_file()


def test_real_pilot_requires_explicit_authorization_before_launch():
    with pytest.raises(SupervisorError, match="Explicit"):
        supervise_r1_pilot(expected_source_commit="0" * 40,
                           profile_id="r1_switch_s0_512_digest_contract_v1",
                           run_id="synthetic-never-launched",
                           authorization_id="missing-approval", authorized=False)
    with pytest.raises(R1BlockedInput, match="explicit-authorization supervisor"):
        _require_supervised_authorization("missing-approval")


def test_actual_memory_arm_adapter_has_deterministic_c0_only_shared_state():
    from types import SimpleNamespace
    sys.path.insert(0, str(ROOT / "experiments"))
    from v02_memory import MemoryArm

    class TinyCodebook:
        def __init__(self, values, capacity):
            self.protos = values.clone()
            self.K = int(capacity)
            self.device = values.device
            self.dtype = values.dtype

        def size(self):
            return int(len(self.protos))

    class FixtureImpl:
        @staticmethod
        def l2_normalize(values):
            return torch.nn.functional.normalize(values, dim=-1)

        @staticmethod
        def select_informative_normal_write(**kwargs):
            return True, 32, "c0_fixture_allowed", dict(kwargs)

    ids = np.arange(752, 784, dtype=np.int64)

    def make_arm():
        dim = 4
        dynamic = TinyCodebook(torch.empty((0, dim), dtype=torch.float32), 256)
        bank = SimpleNamespace(
            normal_dyn={"p0": [{"id": [dynamic]}]},
            normal_dyn_ltm={"p0": [{"id": []}]},
            normal_dyn_candidates={"p0": [{"id": []}]},
            corr_cfg=SimpleNamespace(enabled=False),
        )
        settings = SimpleNamespace(flat=256, stm=256, ltm=1, buffer=1,
                                    min_events=2, promote_every=16,
                                    cap_bytes=2 * 1024**3)
        arm = MemoryArm("greedy_coreset", bank, settings, seed=0)
        return bank, arm

    def features(offset: float):
        values = torch.arange(784, dtype=torch.float32) + 1.0 + offset
        rows = torch.stack((torch.sin(values * 0.013),
                            torch.cos(values * 0.017),
                            torch.sin(values * 0.023),
                            torch.cos(values * 0.029)), dim=1)
        return torch.nn.functional.normalize(rows, dim=1)

    def state_digest(bank):
        dynamic = bank.normal_dyn["p0"][0]["id"][0].protos.detach().cpu().numpy()
        return (tuple(dynamic.shape), dynamic.tobytes())

    def run_trajectory(c1_values, c2_values):
        bank, arm = make_arm()
        impl = FixtureImpl()
        states, actions = [], []
        for position, (c1_value, c2_value, offset) in enumerate(
                zip(c1_values, c2_values, (0.0, 0.2))):
            _ = c1_value, c2_value  # correction outputs are deliberately unused
            full_features = features(offset)
            common_candidates = full_features[torch.as_tensor(ids)].detach().cpu().numpy().astype(np.float32)

            def update(*, position, feature_layers, candidate_ids, score, theta, sigma):
                before = arm.audit()
                arm.update(impl, list(feature_layers),
                           {0: candidate_ids.tolist()}, position,
                           score=score, theta=theta, sigma=sigma)
                after = arm.audit()
                return {"accepted_patches_delta": after["accepted_patches"] - before["accepted_patches"],
                        "skipped_patches_delta": after["skipped_patches"] - before["skipped_patches"]}

            action = common_c0_greedy_writer_action(
                position=position, c0_score=0.25, theta=0.5, sigma=0.1,
                c0_prediction=0, candidate_ids=ids,
                candidate_features=common_candidates,
                feature_layers=[full_features], queried=True, disclosed_label=0,
                eligibility=impl.select_informative_normal_write, update=update)
            actions.append(action)
            states.append(state_digest(bank))
        return bank, arm, actions, states

    first = run_trajectory([0.0, -50.0], [0.0, 50.0])
    second = run_trajectory([1e20, 1e30], [-1e20, -1e30])
    assert first[2:] == second[2:]
    assert first[1].name == "greedy_coreset"
    assert first[1].bank.K_normal_dyn == 256
    assert first[1].audit()["stm"] == 64
    assert all(action["input_score_source"] == "common_C0" for action in first[2])
    assert first[2][0]["candidate_ids_sha256"] == second[2][0]["candidate_ids_sha256"]


def test_actual_memory_arm_does_not_write_for_unqueried_or_queried_defect():
    from types import SimpleNamespace
    sys.path.insert(0, str(ROOT / "experiments"))
    from v02_memory import MemoryArm

    class TinyCodebook:
        def __init__(self, values, capacity):
            self.protos, self.K = values.clone(), int(capacity)
            self.device, self.dtype = values.device, values.dtype

        def size(self):
            return int(len(self.protos))

    class FixtureImpl:
        @staticmethod
        def l2_normalize(values):
            return torch.nn.functional.normalize(values, dim=-1)

        @staticmethod
        def select_informative_normal_write(**kwargs):
            pytest.fail("C0 normal eligibility must not run without queried-normal feedback")

    dynamic = TinyCodebook(torch.empty((0, 4), dtype=torch.float32), 256)
    bank = SimpleNamespace(normal_dyn={"p0": [{"id": [dynamic]}]},
                           normal_dyn_ltm={"p0": [{"id": []}]},
                           normal_dyn_candidates={"p0": [{"id": []}]},
                           corr_cfg=SimpleNamespace(enabled=False))
    settings = SimpleNamespace(flat=256, stm=256, ltm=1, buffer=1,
                                min_events=2, promote_every=16,
                                cap_bytes=2 * 1024**3)
    arm = MemoryArm("greedy_coreset", bank, settings, seed=0)
    before = dynamic.protos.clone()
    ids = np.arange(752, 784, dtype=np.int64)
    rows = torch.nn.functional.normalize(torch.ones((784, 4)), dim=1)
    for queried, label in ((False, None), (True, 1)):
        def forbidden_update(**kwargs):
            pytest.fail("Native Greedy adapter wrote without queried-normal authorization")

        common_c0_greedy_writer_action(
            position=0, c0_score=0.8, theta=0.5, sigma=0.1,
            c0_prediction=1, candidate_ids=ids,
            candidate_features=rows[torch.as_tensor(ids)].numpy().astype(np.float32),
            feature_layers=[rows], queried=queried, disclosed_label=label,
            eligibility=FixtureImpl.select_informative_normal_write,
            update=forbidden_update)
    assert torch.equal(dynamic.protos, before)
