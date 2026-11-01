from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.calibration import r2_sgrm_quick_eval as quick_eval


def test_output_budget_preflights_bytes_and_reserves_failure_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(quick_eval, "MAX_OUTPUT_BYTES", 12)
    monkeypatch.setattr(quick_eval, "FAILURE_RESERVE_BYTES", 4)
    monkeypatch.setattr(quick_eval, "MAX_OUTPUT_FILES", 2)
    budget = quick_eval.OutputBudget(tmp_path / "run", create=True)
    budget.write("run_manifest.json", b"12345678")
    with pytest.raises(quick_eval.ValidationError, match="byte budget"):
        budget.append_json("run_manifest.json", {"extra": "x"})
    assert (tmp_path / "run" / "run_manifest.json").read_bytes() == b"12345678"
    budget.write("failure.json", b"fail")
    assert budget.used_bytes == 12
    with pytest.raises(quick_eval.ValidationError, match="overwrite"):
        budget.write("run_manifest.json", b"replace")


def test_output_budget_reserves_one_file_for_failure_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(quick_eval, "MAX_OUTPUT_FILES", 2)
    budget = quick_eval.OutputBudget(tmp_path / "run", create=True)
    budget.write("run_manifest.json", b"ok")
    with pytest.raises(quick_eval.ValidationError, match="file budget"):
        budget.write("stage_a_seal.json", b"not allowed")
    budget.write("failure.json", b"failed")
    assert sorted(path.name for path in (tmp_path / "run").iterdir()) == [
        "failure.json", "run_manifest.json"]


def test_output_budget_rejects_reparse_point_parent(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable in this environment")
    with pytest.raises(quick_eval.ValidationError, match="symlink or reparse point"):
        quick_eval.OutputBudget(link / "run", create=True)
    assert not (outside / "run").exists()


def test_causal_jsonl_reader_is_lazy_and_never_reads_ahead(tmp_path):
    first = b'{"position":0}\n'
    second = b'{"position":1,"label":1}\n'
    path = tmp_path / "feedback.jsonl"
    path.write_bytes(first + second)
    with quick_eval._CausalJsonlReader(path, "synthetic feedback") as reader:
        assert reader.fd is None
        assert reader.read_record() == {"position": 0}
        assert os.lseek(reader.fd, 0, os.SEEK_CUR) == len(first)
        assert reader.digest.digest() == hashlib.sha256(first).digest()
        assert reader.read_record() == {"position": 1, "label": 1}
        assert reader.digest.digest() == hashlib.sha256(first + second).digest()


def test_event_score_is_sealed_before_current_feedback_callback():
    events = []

    def make_score():
        events.append("score")
        return {"score": 0.25}

    def seal(record):
        assert record == {"score": 0.25}
        events.append("fsynced")

    def disclose():
        assert events[-1] == "fsynced"
        events.append("feedback")

    result = quick_eval._score_seal_record(make_score, seal, disclose, queried=True)
    assert result == {"score": 0.25}
    assert events == ["score", "fsynced", "feedback"]


def test_fractional_frozen_beta_cap_is_validated_without_integer_truncation():
    assert quick_eval._validated_beta_cap(0.2) == pytest.approx(0.2)
    with pytest.raises(quick_eval.ValidationError, match="beta cap"):
        quick_eval._validated_beta_cap(0.0)


def test_eligibility_excludes_same_source_and_future_anchors():
    anchors = [
        SimpleNamespace(position=2, source_group="old-a"),
        SimpleNamespace(position=3, source_group="current"),
        SimpleNamespace(position=4, source_group="future"),
    ]
    assert quick_eval._eligible(anchors, 4, "current") == [anchors[0]]


def test_paired_readouts_share_one_distance_matrix_and_do_not_mutate_memory(monkeypatch):
    distances = np.asarray([[0.0, 0.2, 0.4], [0.1, 0.3, 0.5]], dtype=np.float32)
    beta = np.asarray([0.2, 0.1, -0.05], dtype=np.float32)
    groups = ["g1", "g1", "g2"]
    torch = pytest.importorskip("torch")
    anchors = [SimpleNamespace(
        feature=torch.tensor([float(i), 1.0]), beta=float(beta[i]), radius=0.2,
        kappa=0.5, position=i, source_group=groups[i]) for i in range(3)]
    before = quick_eval._memory_digest(anchors)
    original = quick_eval.r2_sgrm.sgrm_delta_from_distances
    seen = []

    def capture(matrix, values, source_groups, **kwargs):
        seen.append(matrix is distances)
        return original(matrix, values, source_groups, **kwargs)

    monkeypatch.setattr(quick_eval.r2_sgrm, "sgrm_delta_from_distances", capture)
    flat, sgrm, _diagnostics = quick_eval._paired_deltas(distances, beta, groups)
    assert seen == [True]
    assert flat.shape == sgrm.shape == (2,)
    assert np.isfinite(flat).all() and np.isfinite(sgrm).all()
    assert quick_eval._memory_digest(anchors) == before


def test_c1_common_candidate_gate_preserves_original_clip_and_floor():
    from experiments.calibration import r2_exp2b_memory_readout as exp2b

    cfg = SimpleNamespace(beta_max=0.2, beta_min=0.01)
    result = exp2b._common_candidate_eligibility(
        np.asarray([True, False, True, True]),
        np.asarray([0.5, 0.1, 0.005, -0.1], dtype=np.float32), cfg)
    np.testing.assert_allclose(result["clipped_beta"], [0.2, 0.1, 0.005, -0.1], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(result["common_mask"], [True, False, False, True])
    assert result["raw_selected_count"] == 4
    assert result["dN_gate_rejects"] == 1
    assert result["beta_floor_rejects"] == 1
    assert result["common_eligible_count"] == 2


def test_event_identity_mismatch_fails_closed():
    identity = "a" * 64
    score = {
        "category": "switch", "position": 0, "event_number": 0,
        "event_id": "b" * 64, "class_json_sha256": "c" * 64,
        "source_identity_sha256": "d" * 64, "source_group_sha256": identity,
    }
    reference = dict(score)
    quick_eval._event_identity(score, reference, "switch", 0)
    reference["source_group_sha256"] = "e" * 64
    with pytest.raises(quick_eval.ValidationError, match="source_group_sha256"):
        quick_eval._event_identity(score, reference, "switch", 0)


def test_current_query_validator_accepts_diagnostic_schema_without_event_id():
    score = {"category": "switch", "position": 35, "event_id": "a" * 64,
             "source_identity_sha256": "b" * 64, "source_group_sha256": "c" * 64}
    diagnostic = {"category": "switch", "position": 35,
                  "source_group_sha256": "c" * 64, "label": 1}
    feedback = {**score, "disclosure_after_score_seal": True, "label": 1}
    assert quick_eval._validate_current_query_records(
        score, diagnostic, feedback, "switch", 35) == 1
    diagnostic["position"] = 36
    with pytest.raises(quick_eval.ValidationError, match="diagnostic identity"):
        quick_eval._validate_current_query_records(score, diagnostic, feedback, "switch", 35)


def test_stage_b_refuses_score_payload_tampering_before_outcome_access(tmp_path, monkeypatch):
    monkeypatch.setattr(quick_eval, "OUTPUT_ROOT", tmp_path)
    path = tmp_path / "stage_a_scores.jsonl"
    digest = hashlib.sha256()
    with path.open("wb") as stream:
        for category in quick_eval.CATEGORIES:
            for position in range(quick_eval.EVENTS):
                row = {"category": category, "position": position,
                       "source_group_sha256": "f" * 64,
                       "scores": {"SGRM": 0.0}}
                encoded = quick_eval._canonical(row)
                digest.update(encoded)
                stream.write(encoded)
    (tmp_path / "stage_a_seal.json").write_bytes(quick_eval._canonical({
        "status": "SEALED_COMPLETE", "total_events": 2048,
        "stage_a_scores_sha256": digest.hexdigest()}))
    seal, rows = quick_eval._stage_b_rows()
    assert seal["status"] == "SEALED_COMPLETE"
    assert all(len(rows[category]) == 512 for category in quick_eval.CATEGORIES)
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(quick_eval.ValidationError, match="digest mismatch"):
        quick_eval._stage_b_rows()


def test_normal_only_metrics_remain_undefined_for_auc_ap_fnr_and_ba():
    from experiments.calibration import r2_perf1_stage_b as metrics

    result = metrics._metric_record(np.zeros(4, dtype=np.int8),
                                    np.asarray([0.1, 0.2, 0.3, 0.4]), 0.25)
    assert result["image_auroc"] is None
    assert result["image_auprc_average_precision"] is None
    assert result["fixed_threshold"]["fpr"] == 0.5
    assert result["fixed_threshold"]["fnr"] is None
    assert result["fixed_threshold"]["balanced_accuracy"] is None


def test_pair_reversal_diagnostic_excludes_same_source_group_pairs():
    result = quick_eval._pair_reversals(
        np.asarray([0, 1, 0, 1]), ["a", "a", "b", "c"],
        np.asarray([0.7, 0.6, 0.8, 0.8]),
        np.asarray([0.7, 0.9, 0.2, 0.1]))
    assert result["same_source_group_pairs_excluded"] == 1
    assert result["helpful_pair_reversals"] == 1
    assert result["damaging_pair_reversals"] == 1


def test_source_group_bootstrap_is_paired_and_deterministic(monkeypatch):
    from experiments.calibration import r2_perf1_stage_b as metrics

    monkeypatch.setattr(quick_eval, "BOOTSTRAP_REPLICATES", 20)
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    groups = ["a", "a", "b", "c", "d", "d", "e", "f"]
    arms = {
        "C0": np.asarray([0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.9]),
        "SGRM": np.asarray([0.2, 0.9, 0.1, 0.6, 0.4, 0.5, 0.3, 0.8]),
        "matched_flat_Gaussian": np.asarray([0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.9]),
        "C1_NoBoth": np.asarray([0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.9]),
        "historical_C2_a0.02": np.asarray([0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.9]),
    }
    first = quick_eval._paired_group_bootstrap(labels, groups, arms, 0.5, metrics)
    second = quick_eval._paired_group_bootstrap(labels, groups, arms, 0.5, metrics)
    assert first == second
    assert first["unique_source_groups"] == len(set(groups))
    assert first["conditional_on_existing_development_trajectories"] is True
    assert first["intervals"]["SGRM_minus_matched_flat_Gaussian:image_auroc"]["valid_replicates"] > 0


def test_supervisor_terminates_synthetic_child_tree(tmp_path):
    child_pid_file = tmp_path / "descendant.pid"
    descendant_code = "import time; time.sleep(60)"
    parent_code = (
        "import pathlib,subprocess,sys,time;"
        f"p=subprocess.Popen([sys.executable,'-c',{descendant_code!r}]);"
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(p.pid));"
        "time.sleep(60)"
    )
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code], stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=creationflags, start_new_session=(os.name != "nt"))
    deadline = time.monotonic() + 5
    while not child_pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not child_pid_file.is_file():
        quick_eval._terminate_process_tree(parent)
        pytest.fail("Synthetic child process did not report its descendant PID")
    descendant_pid = int(child_pid_file.read_text(encoding="utf-8"))
    quick_eval._terminate_process_tree(parent)
    assert parent.poll() is not None
    if os.name == "nt":
        kernel = ctypes.windll.kernel32
        from ctypes import wintypes
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, descendant_pid)  # SYNCHRONIZE
        if handle:
            try:
                assert kernel.WaitForSingleObject(handle, 5000) == 0
            finally:
                kernel.CloseHandle(handle)
        else:
            listing = subprocess.check_output(
                ["tasklist.exe", "/FI", f"PID eq {descendant_pid}", "/NH"], text=True)
            assert str(descendant_pid) not in listing
    else:
        proc_stat = Path(f"/proc/{descendant_pid}/stat")
        deadline = time.monotonic() + 5
        while proc_stat.exists() and time.monotonic() < deadline:
            state = proc_stat.read_text(encoding="ascii").split()[2]
            if state == "Z":
                break
            time.sleep(0.02)
        assert not proc_stat.exists() or proc_stat.read_text(encoding="ascii").split()[2] == "Z"
