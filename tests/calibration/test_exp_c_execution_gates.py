from __future__ import annotations

import pytest

from experiments.calibration import exp_c_full_run
from experiments.calibration import exp_c_memory_evidence
from experiments.calibration.e6_bea_d2 import D2_ARM_IDS
from experiments.calibration.r1_runtime_supervisor import MAX_RUNTIME_SECONDS


def test_stage_deadlines_are_bound_to_the_supervisor_limit():
    assert 0 < exp_c_full_run.CATEGORY_DEADLINE_SECONDS <= MAX_RUNTIME_SECONDS
    assert 0 < exp_c_full_run.STAGE_B_DEADLINE_SECONDS <= MAX_RUNTIME_SECONDS
    assert exp_c_full_run.CATEGORY_DEADLINE_SECONDS == MAX_RUNTIME_SECONDS
    assert exp_c_full_run.STAGE_B_DEADLINE_SECONDS == MAX_RUNTIME_SECONDS


def test_launcher_imports_the_original_five_arm_d2_constant():
    assert exp_c_full_run.UNCHANGED_D2_ARMS is exp_c_memory_evidence.UNCHANGED_D2_ARMS
    assert set(exp_c_full_run.UNCHANGED_D2_ARMS) == set(
        arm for arm in exp_c_memory_evidence.ARM_IDS if arm in D2_ARM_IDS)
    assert len(exp_c_full_run.UNCHANGED_D2_ARMS) == 5


@pytest.mark.parametrize(
    "mode_args",
    [
        ["--stage-a-child", "--category", "audiojack"],
        ["--stage-b-child"],
    ],
)
def test_direct_child_modes_require_job_object_containment(monkeypatch, capsys, mode_args):
    calls = []

    def outside_job():
        calls.append(True)
        return False

    monkeypatch.setattr(exp_c_full_run.supervisor, "current_process_is_in_job", outside_job)
    result = exp_c_full_run.main([
        *mode_args, "--run-id", "exp_c_guard_test01",
        "--authorization-file", "unused-auth.json",
    ])

    assert result == 1
    assert calls == [True]
    assert "authorized Windows Job Object" in capsys.readouterr().out


def test_registered_child_job_guard_accepts_contained_child(monkeypatch):
    monkeypatch.setattr(exp_c_full_run.supervisor, "current_process_is_in_job", lambda: True)
    exp_c_full_run._require_job_object_child("STAGE_A")
    exp_c_full_run._require_job_object_child("STAGE_B")
    with pytest.raises(exp_c_full_run.ExperimentCError, match="unregistered child stage"):
        exp_c_full_run._require_job_object_child("STAGE_C")
