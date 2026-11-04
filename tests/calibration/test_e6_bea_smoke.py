from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.calibration import e6_bea_smoke as smoke
from experiments.calibration.e6_bea_closed_loop import (
    ARM_IDS,
    ArmRuntime,
    EncodedEvent,
    EventIdentity,
    FeedbackUpdate,
    NativePreEventScore,
    NormalCoverageMemory,
    PrivateFeedbackSink,
    ScoreLedger,
    StageAClosedLoop,
    StageASeal,
    StageBLabelSource,
    E6ClosedLoopError,
    stage_b_evaluate,
    validate_smoke_event_prefix,
)
from experiments.calibration.e6_bea_policy import QueryController, P_RANDOM_FIVE_VIEW


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _events(group_count: int = 4) -> tuple[EventIdentity, ...]:
    rows = []
    for group_index in range(group_count):
        group = _sha(f"smoke-group-{group_index}")
        for camera in range(1, 6):
            position = len(rows)
            relative = f"audiojack/{group_index}/{camera}.png"
            rows.append(EventIdentity(
                position=position, original_position=position,
                event_id_sha256=_sha(f"event-{position}"),
                image_identity_sha256=_sha(relative),
                source_group_sha256=group, relative_id=relative,
                camera_id=str(camera), timestamp=f"202611040000{position:02d}",
                image_path=f"not-opened/{position}.png"))
    return tuple(rows)


def _new_budget(tmp_path: Path, run_id: str = "unit-a"):
    parent = tmp_path / "outputs" / "calibration"
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = parent / f"e6_gpu_smoke_{run_id}"
    run_dir.mkdir()
    return run_dir, smoke.SmokeOutputBudget(run_dir, initialize=True)


def _gpu_sample(worker_pid: int, *, process_memory: dict[int, int] | None = None,
                process_ids: tuple[int, ...] | None = None,
                used_bytes: int = 3 * 1024**3,
                free_bytes: int = 21 * 1024**3,
                total_bytes: int = 24 * 1024**3,
                unavailable_reason: str | None = None):
    if process_ids is None:
        process_ids = (worker_pid,) if process_memory is not None else ()
    return smoke.GPUMonitorSample(
        processes=smoke.GPUProcessInventory(
            process_ids=process_ids, memory_bytes_by_pid=process_memory,
            memory_unavailable_reason=unavailable_reason),
        device_memory=smoke.GPUDeviceMemory(
            used_bytes=used_bytes, free_bytes=free_bytes, total_bytes=total_bytes))


class _Reader:
    prefix_only = True

    def __init__(self, arm_id: str, ledger: ScoreLedger, label: int = 0):
        self.arm_id = arm_id
        self.ledger = ledger
        self.label = label
        self.events = ()
        self.category = "audiojack"
        self.source_sha256 = "a" * 64
        self.accessed: list[int] = []

    def label_for(self, event, receipt):
        assert receipt.arm_id == self.arm_id
        assert self.ledger.is_sealed(event.position)
        rows = [json.loads(line) for line in self.ledger.path.read_text().splitlines()]
        sealed = next(row for row in rows if row["position"] == event.position)
        assert set(sealed["arm_rows"]) == set(ARM_IDS)
        assert all("label" not in value for value in sealed["arm_rows"].values())
        self.accessed.append(event.position)
        return self.label

    def close(self):
        return None


class _Engine:
    def __init__(self):
        self.initial_state_sha256 = "c" * 64
        self.state_digest = self.initial_state_sha256
        self.state_version = 0
        self.current_threshold = 0.5
        self.input_identity = {"c64_score_tolerance": {"atol": 1e-5, "rtol": 1e-4}}
        self.score_calls = 0
        self.unqueried_calls = 0
        self.feedback_calls = []

    def score(self, event, encoded):
        self.score_calls += 1
        return NativePreEventScore(
            gc_score=0.25, c0_score=0.2, c64_scores=tuple(np.linspace(0, 1, 64)),
            correction_top32=tuple(np.zeros(32)), correction_bank_size=0,
            topk_dN_max=0.8, topk_dN_mean=0.2,
            scorer_state_version=0, scorer_state_sha256=self.initial_state_sha256)

    def observe_unqueried(self, event, encoded, score):
        self.unqueried_calls += 1
        return {"action": "no_normal_memory_write"}

    def apply_feedback(self, event, encoded, score, label):
        self.feedback_calls.append((event.position, label))
        return FeedbackUpdate(
            label=label, c1_changed=True, greedy_changed=(label == 0),
            scorer_state_version=0, scorer_state_sha256=self.initial_state_sha256,
            diagnostics={"synthetic_cpu_test": True}, scorer_state_changed=False)


def _synthetic_coordinator(tmp_path: Path, events=None):
    events = tuple(events or _events())
    run_dir, budget = _new_budget(tmp_path)
    ledger = ScoreLedger(
        run_dir / "smoke_scores.jsonl", output_budget=budget,
        artifact_kind="bounded_engineering_smoke")
    support = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    arms = {}
    for arm_id in ARM_IDS:
        reader = _Reader(arm_id, ledger)
        reader.events = events
        sink = PrivateFeedbackSink(
            run_dir / f"smoke_feedback_{arm_id}.jsonl", arm_id, output_budget=budget)
        arms[arm_id] = ArmRuntime(
            arm_id=arm_id, engine=_Engine(), controller=QueryController(arm_id, seed=37),
            coverage=NormalCoverageMemory(support), label_reader=reader,
            private_feedback=sink)
    encoded_ids = []

    def encode(event):
        encoded = EncodedEvent(
            image=None, feature_layers=(), global_cls=np.asarray([1.0, 0.0], dtype=np.float32),
            encoding_seconds=0.001)
        encoded_ids.append((event.position, id(encoded)))
        return encoded

    coordinator = StageAClosedLoop(
        category="audiojack", events=events, feature_encoder=encode,
        arms=arms, ledger=ledger, synthetic_test_only=True,
        smoke_mode=True, output_budget=budget)
    return coordinator, arms, encoded_ids, budget


def test_default_invocation_is_inert(monkeypatch, capsys):
    monkeypatch.setattr(smoke, "preflight_gpu_smoke",
                        lambda **_kwargs: pytest.fail("default invocation reached GPU preflight"))
    assert smoke.main([]) == 0
    assert "no dataset, model, labels, or CUDA" in capsys.readouterr().out


def test_smoke_requires_explicit_authorization_and_exact_reviewed_commit(tmp_path, monkeypatch):
    with pytest.raises(smoke.SmokeError, match="explicit researcher authorization"):
        smoke.preflight_gpu_smoke(
            category="audiojack", max_test_events=20, run_id="a1",
            approved_commit="0" * 40, authorization_id="", human_authorized=False,
            repository_root=tmp_path)
    with pytest.raises(E6ClosedLoopError, match="separately reviewed E6 identity"):
        smoke.preflight_gpu_smoke(
            category="audiojack", max_test_events=20, run_id="a1",
            approved_commit="0" * 40, authorization_id="auth-1", human_authorized=True,
            repository_root=Path(__file__).resolve().parents[2])


@pytest.mark.parametrize("category,cap", [("switch", 20), ("audiojack", 21), ("audiojack", 19)])
def test_preflight_rejects_wrong_category_or_event_cap_before_source_access(
        tmp_path, monkeypatch, category, cap):
    monkeypatch.setattr(smoke, "_verify_pinned_supervisor_source",
                        lambda *_args: pytest.fail("invalid configuration reached source loading"))
    with pytest.raises(smoke.SmokeError):
        smoke.preflight_gpu_smoke(
            category=category, max_test_events=cap, run_id="a1",
            approved_commit="a" * 40, authorization_id="auth-1", human_authorized=True,
            repository_root=tmp_path)


def test_preflight_requires_the_registered_python_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke.sys, "executable", os.environ["COMSPEC"])
    with pytest.raises(smoke.SmokeError, match="registered D:\\\\anaconda3"):
        smoke.preflight_gpu_smoke(
            category="audiojack", max_test_events=20, run_id="a1",
            approved_commit="a" * 40, authorization_id="auth-1", human_authorized=True,
            repository_root=tmp_path)


def test_wddm_na_keeps_compute_pid_and_marks_process_memory_unavailable():
    commands = []

    def runner(command, **_kwargs):
        commands.append(command)
        if "--query-compute-apps=pid" in command:
            return SimpleNamespace(returncode=0, stdout="1234\n")
        if "--query-compute-apps=pid,used_memory" in command:
            return SimpleNamespace(returncode=0, stdout="1234, [N/A]\n")
        pytest.fail(f"unexpected nvidia-smi query: {command}")

    inventory = smoke._query_gpu_processes(runner=runner)
    assert commands[0][1] == "--query-compute-apps=pid"
    assert inventory.process_ids == (1234,)
    assert inventory.memory_bytes_by_pid is None
    assert inventory.process_memory_available is False
    assert inventory.memory_unavailable_reason == "driver_reports_per_process_memory_NA"


def test_numeric_per_process_memory_is_parsed_in_mib():
    def runner(command, **_kwargs):
        if "--query-compute-apps=pid" in command:
            return SimpleNamespace(returncode=0, stdout="1234\n")
        return SimpleNamespace(returncode=0, stdout="1234, 768\n")

    inventory = smoke._query_gpu_processes(runner=runner)
    assert inventory.process_ids == (1234,)
    assert inventory.memory_bytes_by_pid == {1234: 768 * 1024**2}
    assert inventory.process_memory_available is True


def test_compute_pid_classification_allows_supervisor_and_worker_only():
    foreign = smoke._foreign_compute_pids(
        (os.getpid(), 4321, 8765), supervisor_pid=os.getpid(),
        authorized_worker_pid=4321)
    assert foreign == (8765,)
    assert smoke._foreign_compute_pids(
        (os.getpid(), 4321), supervisor_pid=os.getpid(),
        authorized_worker_pid=4321) == ()


def test_required_gpu_monitoring_failures_fail_closed():
    with pytest.raises(smoke.SmokeError, match="compute-process PIDs"):
        smoke._query_gpu_processes(runner=lambda *_a, **_k:
                                   SimpleNamespace(returncode=1, stdout=""))

    def invalid_device_query(command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="N/A, N/A, N/A\n")

    with pytest.raises(smoke.SmokeError, match="device-level used/free/total"):
        smoke._query_gpu_device_memory(runner=invalid_device_query)


def test_device_level_gpu_memory_telemetry_is_parsed_in_mib():
    sample = smoke._query_gpu_device_memory(runner=lambda *_a, **_k:
        SimpleNamespace(returncode=0, stdout="4096, 20480, 24576\n"))
    assert sample == smoke.GPUDeviceMemory(
        used_bytes=4096 * 1024**2, free_bytes=20480 * 1024**2,
        total_bytes=24576 * 1024**2)


def test_prefix_is_exactly_four_complete_five_view_groups():
    selected = _events()
    assert validate_smoke_event_prefix(selected) == selected
    with pytest.raises(E6ClosedLoopError, match="exactly 20"):
        validate_smoke_event_prefix(selected[:-1])
    with pytest.raises(E6ClosedLoopError, match="exactly 20"):
        validate_smoke_event_prefix((*selected, _events(5)[-1]))
    with pytest.raises(E6ClosedLoopError, match="contiguous"):
        validate_smoke_event_prefix((selected[1], selected[0], *selected[2:]))


def test_prefix_projection_decodes_only_selected_image_paths(tmp_path, monkeypatch):
    from experiments.calibration import r7_shuffled_correction_4class as r7
    import experiments.risk_safe.preflight as preflight

    events = []
    rows = []
    for position, event in enumerate(_events(5)):
        relative = event.relative_id
        events.append(EventIdentity(
            position=position, original_position=position,
            event_id_sha256=event.event_id_sha256,
            image_identity_sha256=r7._identity_hash(relative),
            source_group_sha256=hashlib.sha256(
                f"official-sample-group\0key-{position // 5}".encode()).hexdigest(),
            relative_id=relative, camera_id=event.camera_id,
            timestamp=event.timestamp, image_path="unused"))
        rows.append({"image_path": f"view-{position}",
                     "anomaly_class": {"not": "decoded by prefix projection"},
                     "mask": ["not decoded"]})
    rows[-1]["image_path"] = 123456  # outside the selected prefix and intentionally invalid.
    path = tmp_path / "audiojack.json"
    path.write_text(json.dumps({"meta": {"normal_class": "good"}, "test": rows}))
    calls = []

    def parse(value, expected_class=None):
        index = int(value.split("-")[1])
        selected = events[index]
        calls.append(index)
        return SimpleNamespace(
            rel_path=selected.relative_id, camera_id=int(selected.camera_id),
            timestamp=selected.timestamp,
            official_sample_key=f"key-{index // 5}")

    monkeypatch.setattr(preflight, "parse_image_name", parse)
    from experiments.calibration.e6_bea_closed_loop import validate_test_image_prefix_projection
    validate_test_image_prefix_projection(path, "audiojack", events[:20], full_event_count=25)
    assert calls == list(range(20))


def test_coordinator_rejects_full_stream_before_encoding(tmp_path):
    called = []
    with pytest.raises(E6ClosedLoopError, match="exactly 20"):
        StageAClosedLoop(
            category="audiojack", events=_events(5),
            feature_encoder=lambda event: called.append(event), arms={}, ledger=None,
            synthetic_test_only=True, smoke_mode=True)
    assert called == []


def test_output_budget_checks_before_write_and_preserves_failure_reserve(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "MAX_OUTPUT_BYTES", 2048)
    monkeypatch.setattr(smoke, "FAILURE_RESERVE_BYTES", 256)
    run_dir, budget = _new_budget(tmp_path)
    budget.write_json_once("smoke_manifest.json", {"padding": "x" * 1600})
    budget.create_empty("smoke_scores.jsonl")
    with pytest.raises(smoke.SmokeBudgetError, match="before write"):
        budget.append_bytes("smoke_scores.jsonl", b"x" * 200)
    assert (run_dir / "smoke_scores.jsonl").stat().st_size == 0
    budget.write_failure({"status": "SMOKE_FAILED", "message": "short"})
    assert (run_dir / "smoke_failure.json").is_file()
    assert budget.total_bytes() <= 2048


def test_output_budget_rejects_unregistered_writer_and_never_overwrites(tmp_path):
    run_dir, budget = _new_budget(tmp_path)
    with pytest.raises(smoke.SmokeBudgetError, match="not registered"):
        budget.create_empty("escape.jsonl")
    budget.write_json_once("smoke_manifest.json", {"status_at_creation": "INCOMPLETE"})
    with pytest.raises(FileExistsError):
        budget.write_json_once("smoke_manifest.json", {"status_at_creation": "COMPLETE"})
    (run_dir / "unaccounted.bin").write_bytes(b"x")
    with pytest.raises(smoke.SmokeBudgetError, match="unaccounted"):
        budget.require_registered()


def test_smoke_namespace_is_unique_and_non_overwriting(tmp_path):
    (tmp_path / "outputs" / "calibration").mkdir(parents=True)
    run_dir, budget = smoke.create_smoke_output_namespace("unique1", repository_root=tmp_path)
    assert run_dir.name == "e6_gpu_smoke_unique1"
    assert budget.total_bytes() == 1
    with pytest.raises(FileExistsError):
        smoke.create_smoke_output_namespace("unique1", repository_root=tmp_path)


def test_worker_calls_only_bounded_coordinator(monkeypatch, tmp_path):
    from experiments.calibration import e6_bea_runner as runner
    from experiments.calibration import r1_runtime_supervisor as r1_supervisor
    monkeypatch.setattr(r1_supervisor, "current_process_is_in_job", lambda: True)
    output_dir, budget = _new_budget(tmp_path, "worker1")
    called = []

    def bounded(**kwargs):
        called.append(kwargs["run_id"])
        return SimpleNamespace(artifact_kind="bounded_engineering_smoke", complete=False)

    monkeypatch.setattr(runner, "execute_bounded_smoke_stage_a", bounded)
    monkeypatch.setattr(runner, "execute_real_stage_a",
                        lambda **_kwargs: pytest.fail("smoke called full-stream Stage A"))
    result = smoke._worker_main(
        run_id="worker1", category="audiojack", max_test_events=20,
        approved_commit="a" * 40, authorization_id="auth-1", human_authorized=True,
        output_dir=output_dir, repository_root=tmp_path)
    assert result == 0
    assert called == ["worker1"]


def test_worker_rejects_direct_uncontained_invocation(monkeypatch, tmp_path):
    from experiments.calibration import r1_runtime_supervisor as r1_supervisor
    monkeypatch.setattr(r1_supervisor, "current_process_is_in_job", lambda: False)
    with pytest.raises(smoke.SmokeError, match="dedicated Windows Job Object"):
        smoke._worker_main(
            run_id="uncontained", category="audiojack", max_test_events=20,
            approved_commit="a" * 40, authorization_id="auth-1", human_authorized=True,
            output_dir=tmp_path / "never-opened", repository_root=tmp_path)


def test_stage_b_rejects_smoke_receipt_before_reading_any_label(tmp_path):
    seals = {
        category: StageASeal(
            category=category, event_count=20 if category == "audiojack" else 3815,
            score_log_path=str(tmp_path / "missing-score-file.jsonl"),
            score_log_sha256="a" * 64, private_feedback_sha256={},
            arm_query_counts={}, arm_controller_diagnostics={}, complete=(category != "audiojack"),
            artifact_kind=("bounded_engineering_smoke" if category == "audiojack"
                           else "scientific_stage_a"))
        for category in ("audiojack", "regulator", "switch", "rolled_strip_base")
    }
    sources = {category: StageBLabelSource(Path("missing.json"), "b" * 64)
               for category in seals}
    with pytest.raises(E6ClosedLoopError, match="cannot be promoted"):
        stage_b_evaluate(
            stage_a_seals=seals,
            expected_categories=("audiojack", "regulator", "switch", "rolled_strip_base"),
            label_sources=sources, stage_b_authorized=True, authorization_id="stage-b-test")


def test_cpu_coordinator_seals_all_arms_before_disclosure_and_encodes_once(tmp_path):
    coordinator, arms, encoded_ids, budget = _synthetic_coordinator(tmp_path)
    receipt = coordinator.run()
    assert receipt.artifact_kind == "bounded_engineering_smoke"
    assert receipt.complete is False
    assert receipt.smoke_status in {"SMOKE_PASS", "SMOKE_INCOMPLETE_COVERAGE"}
    assert len(encoded_ids) == 20
    assert [position for position, _identity in encoded_ids] == list(range(20))
    assert all(arm.engine.score_calls == 20 for arm in arms.values())
    assert sum(arm.engine.score_calls for arm in arms.values()) == 20 * len(ARM_IDS)
    rows = [json.loads(line) for line in (budget.root / "smoke_scores.jsonl").read_text().splitlines()]
    assert len(rows) == 20
    assert all(row["schema"] == "e6.bea_v2.bounded_smoke_score.v1" for row in rows)
    for arm_id, arm in arms.items():
        queried = [row["position"] for row in rows if row["arm_rows"][arm_id]["query_executed"]]
        assert arm.label_reader.accessed == queried
        assert len({rows[position]["source_group_sha256"] for position in queried}) == len(queried)
    assert not (budget.root / "stage_a_seal.json").exists()
    assert not (budget.root / "run_manifest.json").exists()


def test_score_policy_has_no_smoke_horizon_parameter():
    import inspect
    signature = inspect.signature(QueryController.propose)
    assert "stream_length" not in signature.parameters
    assert "event_limit" not in signature.parameters
    assert "max_test_events" not in signature.parameters
    assert P_RANDOM_FIVE_VIEW == 0.0319812149975186


def test_truncated_prefix_does_not_change_registered_query_decisions():
    def decisions(stream_length):
        outcomes = {}
        for arm_id in ARM_IDS:
            controller = QueryController(arm_id, seed=37)
            arm_rows = []
            for position in range(stream_length):
                event_id = _sha(f"policy-event-{position}")
                components = SimpleNamespace(
                    psi=(position % 7) / 7,
                    novelty=(position % 11) / 11,
                    objective=lambda requested_arm, position=position:
                        ((position % 5) + 1) / 10,
                )
                proposal = controller.propose(
                    position=position, event_id_sha256=event_id, components=components)
                arm_rows.append((proposal.proposed, proposal.objective,
                                 proposal.threshold, proposal.reason))
                suppressed = proposal.proposed and position // 5 in {1, 3}
                executed = proposal.proposed and not suppressed
                controller.observe_final_action(
                    position=position, proposal=proposal,
                    executed_query=executed, group_suppressed=suppressed)
            outcomes[arm_id] = tuple(arm_rows)
        return outcomes

    # The same first 20 observations produce identical policy decisions whether
    # the caller stops there or has a longer stream available.
    assert decisions(20) == {
        arm_id: rows[:20] for arm_id, rows in decisions(25).items()
    }


def test_support_augmentation_batching_obeys_eight_forward_cap(monkeypatch):
    import torch
    import torch.nn.functional as torch_functional
    from experiments.calibration import e6_bea_runner as runner

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class FakeImpl:
        @staticmethod
        def SupportAugConfig():
            return object()

        @staticmethod
        def load_rgb(path):
            return f"rgb:{path}"

        @staticmethod
        def make_support_refs(image, _backbone, _augmentation, count, **kwargs):
            assert image.startswith("rgb:") and count == 4
            assert kwargs == {"baseline_compat": False, "include_identity": False}
            return [f"{image}:aug-{index}" for index in range(count)]

        @staticmethod
        def fuse_patch_features(per_layer, _mode):
            return [torch.cat(per_layer, dim=-1)]

        @staticmethod
        def l2_normalize(tensor):
            return torch_functional.normalize(tensor, dim=-1)

    class FakeP0:
        @staticmethod
        def deterministic_subseed(seed, path):
            return seed + len(path)

    class FakeBackbone:
        def __init__(self):
            self.batch_sizes = []

        def encode_batch(self, refs):
            self.batch_sizes.append(len(refs))
            batch = len(refs)
            first = torch.arange(batch * 6, dtype=torch.float32).reshape(batch, 2, 3) + 1
            second = first + 1
            return [first, second], torch.ones((batch, 3), dtype=torch.float32)

    paths = [f"support-{index}" for index in range(4)]
    cfg = SimpleNamespace(
        augmentation_seed=0, augmentations=4, include_identity=False,
        feature_fuse="concat")
    backbone = FakeBackbone()
    cache, receipt = runner._support_features_batched(
        FakeImpl, FakeP0, backbone, paths, cfg)

    assert backbone.batch_sizes == [4, 4, 4, 4]
    assert len(cache) == 4
    assert all(len(cache[path]) == 4 for path in paths)
    assert receipt["support_batch_forwards"] == 4
    assert receipt["support_augmentation_view_inputs"] == 16
    assert receipt["support_cls_forwards"] == 4
    assert receipt["support_cls_image_inputs"] == 4
    assert receipt["support_fresh_backbone_invocations"] == 8
    assert receipt["support_fresh_backbone_invocations"] <= receipt["support_invocation_limit"]


def test_wrong_supervisor_source_fingerprint_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "experiments" / "calibration" / "r1_runtime_supervisor.py"
    path.parent.mkdir(parents=True)
    path.write_text("pinned source")
    monkeypatch.setattr(smoke, "_sha256_file", lambda _path: "0" * 64)
    with pytest.raises(smoke.SmokeError, match="fingerprint changed"):
        smoke._verify_pinned_supervisor_source(tmp_path)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object process-tree containment")
def test_windows_job_watchdog_completes_and_records_child_failure():
    ok = smoke.supervise_smoke_child(
        [sys.executable, "-c", "import time; time.sleep(0.2)"], environment=os.environ.copy(),
        working_directory=Path.cwd(),
        gpu_process_probe=lambda pid: _gpu_sample(pid, process_memory={pid: 32 * 1024**2}),
        timeout_seconds=5, poll_interval_seconds=0.05)
    assert ok.status == "COMPLETED" and ok.termination_confirmed
    assert ok.process_specific_vram_available is True
    assert ok.monitoring_mode == "process_specific"
    failure = smoke.supervise_smoke_child(
        [sys.executable, "-c", "import time; time.sleep(0.2); raise SystemExit(7)"],
        environment=os.environ.copy(), working_directory=Path.cwd(),
        gpu_process_probe=lambda pid: _gpu_sample(pid, process_memory={pid: 32 * 1024**2}),
        timeout_seconds=5, poll_interval_seconds=0.05)
    assert failure.status == "FAILED_NONZERO_EXIT" and failure.child_exit_code == 7
    assert failure.termination_confirmed


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object process-tree containment")
def test_windows_job_watchdog_kills_timeout_and_spawned_descendant(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    code = (
        "import subprocess,sys,time; "
        f"child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"open({str(pid_file)!r},'w').write(str(child.pid)); time.sleep(30)"
    )
    result = smoke.supervise_smoke_child(
        [sys.executable, "-c", code], environment=os.environ.copy(),
        working_directory=Path.cwd(),
        gpu_process_probe=lambda pid: _gpu_sample(pid, process_memory={pid: 32 * 1024**2}),
        timeout_seconds=0.5, poll_interval_seconds=0.05)
    assert result.status == "TIMED_OUT_TERMINATED"
    assert result.termination_confirmed and result.active_job_processes == 0
    child_pid = int(pid_file.read_text())
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(0x00100000, False, child_pid)  # SYNCHRONIZE
    if handle:
        try:
            assert kernel32.WaitForSingleObject(handle, 0) == 0  # WAIT_OBJECT_0
        finally:
            kernel32.CloseHandle(handle)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object process-tree containment")
@pytest.mark.parametrize("probe,expected", [
    (lambda pid: _gpu_sample(pid, process_memory={pid: smoke.MAX_GPU_MEMORY_BYTES + 1}),
     "gpu_memory_ceiling_exceeded"),
    (lambda pid: _gpu_sample(
        pid, process_memory={pid: 1024, pid + 100000: 1024},
        process_ids=(pid, pid + 100000)), "competing_gpu_process_detected"),
    (lambda pid: _gpu_sample(
        pid, process_memory=None, process_ids=(pid,),
        used_bytes=smoke.MAX_GPU_MEMORY_BYTES + 1,
        free_bytes=1024**3,
        unavailable_reason="driver_reports_per_process_memory_NA"),
     "device_memory_ceiling_exceeded"),
])
def test_windows_job_watchdog_kills_gpu_resource_or_competing_pid(probe, expected):
    result = smoke.supervise_smoke_child(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        environment=os.environ.copy(), working_directory=Path.cwd(),
        gpu_process_probe=probe, timeout_seconds=5, poll_interval_seconds=0.05)
    assert result.status == "RESOURCE_LIMIT_TERMINATED"
    assert result.termination_reason == expected
    assert result.termination_confirmed


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object process-tree containment")
def test_windows_job_watchdog_fails_closed_when_gpu_monitoring_is_unavailable():
    result = smoke.supervise_smoke_child(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        environment=os.environ.copy(), working_directory=Path.cwd(),
        gpu_process_probe=lambda _pid: (_ for _ in ()).throw(
            smoke.SmokeError("device telemetry unavailable")),
        timeout_seconds=5, poll_interval_seconds=0.05)
    assert result.status == "RESOURCE_LIMIT_TERMINATED"
    assert result.termination_reason == "gpu_memory_monitor_failed"
    assert result.monitor_error and "device telemetry unavailable" in result.monitor_error
    assert result.termination_confirmed


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object process-tree containment")
def test_windows_job_watchdog_uses_conservative_device_fallback_for_wddm_na():
    result = smoke.supervise_smoke_child(
        [sys.executable, "-c", "import time; time.sleep(0.2)"],
        environment=os.environ.copy(), working_directory=Path.cwd(),
        gpu_process_probe=lambda pid: _gpu_sample(
            pid, process_memory=None, process_ids=(pid,),
            used_bytes=8 * 1024**3, free_bytes=16 * 1024**3,
            unavailable_reason="driver_reports_per_process_memory_NA"),
        timeout_seconds=5, poll_interval_seconds=0.05)
    assert result.status == "COMPLETED" and result.termination_confirmed
    assert result.process_specific_vram_available is False
    assert result.peak_process_gpu_bytes is None
    assert result.peak_device_used_bytes == 8 * 1024**3
    assert result.monitoring_mode == "device_level_conservative_fallback"
    assert result.configured_gpu_memory_limit_bytes == smoke.MAX_GPU_MEMORY_BYTES
    assert result.process_specific_vram_unavailable_reasons == (
        "driver_reports_per_process_memory_NA",)


def test_preflight_rejects_wrong_visible_gpu_count_before_launch(tmp_path, monkeypatch):
    import experiments.calibration.e6_bea_closed_loop as closed_loop
    import experiments.calibration.e6_bea_runner as runner
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r3_fullstream_2x2 as r3

    (tmp_path / "outputs" / "calibration").mkdir(parents=True)
    monkeypatch.setattr(closed_loop, "verify_e6_execution_identity", lambda *_a, **_k: {
        "branch": closed_loop.EXPECTED_BRANCH, "head": "a" * 40,
        "r7_helper_source_sha256": {}})
    monkeypatch.setattr(smoke, "_verify_pinned_supervisor_source", lambda _root: "b" * 64)
    monkeypatch.setattr(runner, "load_e6_stage05_category", lambda *_a, **_k: {
        "events": _events(), "contract": {"category_json_sha256": "c" * 64},
        "stage05_receipts": {}})
    monkeypatch.setattr(r3, "_get_verified_model_receipt", lambda: {})

    class FakeCuda:
        @staticmethod
        def is_available(): return True
        @staticmethod
        def device_count(): return 2

    with pytest.raises(smoke.SmokeError, match="exactly one visible CUDA"):
        smoke.preflight_gpu_smoke(
            category="audiojack", max_test_events=20, run_id="preflight1",
            approved_commit="a" * 40, authorization_id="auth-1", human_authorized=True,
            repository_root=tmp_path, torch_module=SimpleNamespace(cuda=FakeCuda()))


def test_preflight_rejects_model_source_or_checkpoint_sha_mismatch(tmp_path, monkeypatch):
    import experiments.calibration.e6_bea_closed_loop as closed_loop
    import experiments.calibration.e6_bea_runner as runner
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r3_fullstream_2x2 as r3

    (tmp_path / "outputs" / "calibration").mkdir(parents=True)
    monkeypatch.setattr(closed_loop, "verify_e6_execution_identity", lambda *_a, **_k: {
        "branch": closed_loop.EXPECTED_BRANCH, "head": "a" * 40,
        "r7_helper_source_sha256": {}})
    monkeypatch.setattr(smoke, "_verify_pinned_supervisor_source", lambda _root: "b" * 64)
    monkeypatch.setattr(runner, "load_e6_stage05_category", lambda *_a, **_k: {
        "events": _events(), "contract": {"category_json_sha256": "c" * 64},
        "stage05_receipts": {}})
    receipt = {"runtime_profile": {"cuda_device_name": "GPU", "torch_version": "2.0"},
               "hub_source_tree_sha256": "d" * 64, "checkpoint_sha256": "e" * 64}
    monkeypatch.setattr(r3, "_get_verified_model_receipt", lambda: receipt)
    monkeypatch.setattr(r3, "KNOWN_R1_HUB_TREE_SHA256", "d" * 64)
    monkeypatch.setattr(r3, "KNOWN_CHECKPOINT_SHA256", "e" * 64)
    monkeypatch.setattr(r3, "sha256_file", lambda _path: "wrong")
    monkeypatch.setattr(r1, "_hub_source_sha256", lambda _root: ("local", "d" * 64))
    monkeypatch.setattr(smoke, "_query_gpu_processes", lambda **_kwargs:
                        smoke.GPUProcessInventory((), {}, None))
    monkeypatch.setattr(smoke, "_query_gpu_device_memory", lambda **_kwargs:
                        smoke.GPUDeviceMemory(0, 24 * 1024**3, 24 * 1024**3))

    class FakeCuda:
        @staticmethod
        def is_available(): return True
        @staticmethod
        def device_count(): return 1
        @staticmethod
        def get_device_properties(_device): return SimpleNamespace(name="GPU", total_memory=24 * 1024**3)
        @staticmethod
        def mem_get_info(_device): return (24 * 1024**3, 24 * 1024**3)

    hub = tmp_path / "hub"
    (hub / "checkpoints").mkdir(parents=True)
    checkpoint = hub / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
    checkpoint.write_bytes(b"weights")
    fake_torch = SimpleNamespace(
        cuda=FakeCuda(), __version__="2.0", hub=SimpleNamespace(get_dir=lambda: str(hub)))
    monkeypatch.setattr(smoke.shutil, "disk_usage", lambda _path: SimpleNamespace(free=20 * 1024**3))
    with pytest.raises(smoke.SmokeError, match="checkpoint"):
        smoke.preflight_gpu_smoke(
            category="audiojack", max_test_events=20, run_id="preflight2",
            approved_commit="a" * 40, authorization_id="auth-1", human_authorized=True,
            repository_root=tmp_path, torch_module=fake_torch,
            subprocess_runner=lambda *_a, **_k: SimpleNamespace(returncode=0, stdout=""))


def test_preflight_rejects_less_than_ten_gibibytes_free_disk(tmp_path, monkeypatch):
    import experiments.calibration.e6_bea_closed_loop as closed_loop
    import experiments.calibration.e6_bea_runner as runner
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r3_fullstream_2x2 as r3

    output_parent = tmp_path / "outputs" / "calibration"
    output_parent.mkdir(parents=True)
    monkeypatch.setattr(closed_loop, "verify_e6_execution_identity", lambda *_a, **_k: {
        "branch": closed_loop.EXPECTED_BRANCH, "head": "a" * 40,
        "r7_helper_source_sha256": {}})
    monkeypatch.setattr(smoke, "_verify_pinned_supervisor_source", lambda _root: "b" * 64)
    monkeypatch.setattr(runner, "load_e6_stage05_category", lambda *_a, **_k: {
        "events": _events(), "contract": {"category_json_sha256": "c" * 64},
        "stage05_receipts": {}})
    receipt = {"runtime_profile": {"cuda_device_name": "GPU", "torch_version": "2.0"},
               "hub_source_tree_sha256": "d" * 64, "checkpoint_sha256": "e" * 64}
    monkeypatch.setattr(r3, "_get_verified_model_receipt", lambda: receipt)
    monkeypatch.setattr(r3, "KNOWN_R1_HUB_TREE_SHA256", "d" * 64)
    monkeypatch.setattr(r3, "KNOWN_CHECKPOINT_SHA256", "e" * 64)
    monkeypatch.setattr(r3, "sha256_file", lambda _path: "e" * 64)
    monkeypatch.setattr(r1, "_hub_source_sha256", lambda _root: ("local", "d" * 64))
    monkeypatch.setattr(smoke, "_query_gpu_processes", lambda **_kwargs:
                        smoke.GPUProcessInventory((), {}, None))
    monkeypatch.setattr(smoke, "_query_gpu_device_memory", lambda **_kwargs:
                        smoke.GPUDeviceMemory(0, 24 * 1024**3, 24 * 1024**3))
    monkeypatch.setattr(smoke.shutil, "disk_usage",
                        lambda _path: SimpleNamespace(free=9 * 1024**3))

    class FakeCuda:
        @staticmethod
        def is_available(): return True
        @staticmethod
        def device_count(): return 1
        @staticmethod
        def get_device_properties(_device):
            return SimpleNamespace(name="GPU", total_memory=24 * 1024**3)
        @staticmethod
        def mem_get_info(_device): return (24 * 1024**3, 24 * 1024**3)

    hub = tmp_path / "hub"
    checkpoint = hub / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"local test checkpoint")
    fake_torch = SimpleNamespace(
        cuda=FakeCuda(), __version__="2.0", hub=SimpleNamespace(get_dir=lambda: str(hub)))
    with pytest.raises(smoke.SmokeError, match="at least 10 GiB"):
        smoke.preflight_gpu_smoke(
            category="audiojack", max_test_events=20, run_id="lowdisk",
            approved_commit="a" * 40, authorization_id="auth-1", human_authorized=True,
            repository_root=tmp_path, torch_module=fake_torch,
            subprocess_runner=lambda *_a, **_k: SimpleNamespace(returncode=0, stdout=""))
    assert not (output_parent / "e6_gpu_smoke_lowdisk").exists()


def test_failed_smoke_has_failure_receipt_and_no_success_seal(tmp_path):
    run_dir, budget = _new_budget(tmp_path, "failed1")
    budget.write_failure({"schema": "e6.bea_v2.smoke_failure.v1", "status": "SMOKE_FAILED",
                          "automatic_restart": False})
    assert (run_dir / "smoke_failure.json").exists()
    assert not (run_dir / "smoke_seal.json").exists()
