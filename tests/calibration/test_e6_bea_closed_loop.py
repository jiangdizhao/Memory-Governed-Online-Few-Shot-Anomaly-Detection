import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import numpy as np
import pytest
import torch

from experiments.calibration.e6_bea_closed_loop import (
    ARM_IDS,
    E6_REGISTERED_CATEGORIES,
    ArmRuntime,
    EncodedEvent,
    E6ClosedLoopError,
    EventIdentity,
    FeedbackUpdate,
    NativePreEventScore,
    NormalCoverageMemory,
    OnDemandLabelReader,
    PrivateFeedbackSink,
    R7FrozenFeatureEncoder,
    ScoreLedger,
    StageAClosedLoop,
    StageASeal,
    stage_b_evaluate,
    make_e6_event_stream,
    validate_five_view_groups,
    validate_data_role_group_disjointness,
    validate_test_image_projection,
    R7_HELPER_SOURCE_SHA256,
    verify_e6_execution_identity,
    _write_json_once,
    _read_sealed_score_log,
    create_e6_output_namespace,
)
from experiments.calibration.e6_bea_policy import (
    P_RANDOM_FIVE_VIEW,
    QueryController,
    _stable_uniform01,
)


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _events(group_count=4):
    rows = []
    position = 0
    for group_index in range(group_count):
        group = _sha(f"group-{group_index}")
        for camera in range(1, 6):
            image = _sha(f"image-{group_index}-{camera}")
            event = _sha(f"event-{position}-{image}-{group}")
            rows.append(EventIdentity(
                position=position, original_position=position,
                event_id_sha256=event, image_identity_sha256=image,
                source_group_sha256=group,
                relative_id=f"synthetic/{group_index}/{camera}.png",
                camera_id=str(camera), timestamp=f"202611040000{position:02d}",
                image_path=f"synthetic/{group_index}/{camera}.png"))
            position += 1
    return tuple(rows)


class FakeReader:
    def __init__(self, arm_id, label=0):
        self.arm_id = arm_id
        self.label = label
        self.accessed = []

    def label_for(self, event, receipt):
        assert receipt.arm_id == self.arm_id
        self.accessed.append(event.position)
        return self.label


class FakeEngine:
    def __init__(self, *, initial="a" * 64, label_threshold_shift=0.0):
        self.initial_state_sha256 = initial
        self.current_threshold = 0.0
        self.label_threshold_shift = label_threshold_shift
        self.normal_writes = 0
        self.c1_updates = 0
        self.unqueried_observations = 0
        self.state = 0
        self.seen_encoded_ids = []

    def score(self, event, encoded):
        self.seen_encoded_ids.append(encoded)
        return NativePreEventScore(
            gc_score=self.current_threshold - 0.001,
            c0_score=0.25,
            c64_scores=tuple(np.linspace(0.0, 1.0, 64)),
            correction_top32=tuple(np.zeros(32)), correction_bank_size=0,
            topk_dN_max=0.8, topk_dN_mean=0.2,
            scorer_state_version=self.state,
            scorer_state_sha256=self.initial_state_sha256)

    def observe_unqueried(self, event, encoded, score):
        self.unqueried_observations += 1
        return {"action": "no_normal_memory_write"}

    def apply_feedback(self, event, encoded, score, label):
        self.c1_updates += 1
        if label == 0:
            self.normal_writes += 1
        self.state += 1
        if self.label_threshold_shift:
            self.current_threshold += self.label_threshold_shift
        return FeedbackUpdate(
            label=label, c1_changed=True, greedy_changed=(label == 0),
            scorer_state_version=self.state, scorer_state_sha256=self.initial_state_sha256,
            diagnostics={"normal_write": label == 0})


def _coverage():
    support = np.eye(4, dtype=np.float32)
    return NormalCoverageMemory(support)


def _choose_q0_seed(events):
    return next(seed for seed in range(10000) if all(
        _stable_uniform01(seed, "Q0", event.event_id_sha256, "random") >= P_RANDOM_FIVE_VIEW
        for event in events))


def _make_runtime(tmp_path, events, *, label=0, threshold_shift=0.0, category="audiojack"):
    arms = {}
    q0_seed = _choose_q0_seed(events)
    for arm_id in ARM_IDS:
        engine = FakeEngine(label_threshold_shift=threshold_shift)
        reader = FakeReader(arm_id, label=label)
        arms[arm_id] = ArmRuntime(
            arm_id=arm_id, engine=engine,
            controller=QueryController(arm_id, seed=q0_seed),
            coverage=_coverage(), label_reader=reader)
    encoded_ids = []

    def encode(_event):
        value = EncodedEvent(image=None, feature_layers=(), global_cls=np.asarray([1, 0, 0, 0],
                                                                               dtype=np.float32))
        encoded_ids.append(id(value))
        return value

    ledger = ScoreLedger(tmp_path / "sealed_scores.jsonl")
    runner = StageAClosedLoop(category=category, events=events,
                              feature_encoder=encode, arms=arms, ledger=ledger,
                              synthetic_test_only=True)
    return runner, arms, encoded_ids


def test_five_view_preflight_rejects_wrong_group_size_and_duplicate_camera():
    events = _events(1)
    assert validate_five_view_groups(events) == {"source_groups": 1, "events": 5,
                                                  "views_per_group": 5}
    with pytest.raises(E6ClosedLoopError, match="five-distinct-view|camera-1-through-5"):
        validate_five_view_groups(events[:4])
    duplicate = list(events)
    duplicate[1] = EventIdentity(**{**duplicate[1].__dict__, "camera_id": "1"})
    with pytest.raises(E6ClosedLoopError, match="camera-1-through-5"):
        validate_five_view_groups(duplicate)


def test_support_c64_and_test_group_identities_must_be_disjoint():
    events = _events(1)
    support = [_sha(f"support-{index}") for index in range(4)]
    c64 = [_sha(f"c64-{index}") for index in range(64)]
    validate_data_role_group_disjointness(events, support, c64)
    with pytest.raises(E6ClosedLoopError, match="must be disjoint"):
        validate_data_role_group_disjointness(events, [events[0].source_group_sha256, *support[1:]], c64)
    with pytest.raises(E6ClosedLoopError, match="must be disjoint"):
        validate_data_role_group_disjointness(events, support, [events[0].source_group_sha256, *c64[1:]])
    with pytest.raises(E6ClosedLoopError, match="identities are incomplete"):
        validate_data_role_group_disjointness(events, support, c64[:-1])


def test_manifest_writer_is_exclusive_and_rejects_public_label_fields(tmp_path):
    path = tmp_path / "run_manifest.json"
    digest = _write_json_once(path, {"schema": "synthetic", "status": "INCOMPLETE"})
    assert len(digest) == 64
    with pytest.raises(FileExistsError):
        _write_json_once(path, {"schema": "replacement"})
    forbidden = tmp_path / "forbidden.json"
    with pytest.raises(E6ClosedLoopError, match="forbidden field"):
        _write_json_once(forbidden, {"label": 1})
    assert not forbidden.exists()


def test_output_namespace_creation_is_unique_and_never_reuses_existing_directory(tmp_path):
    output_parent = tmp_path / "outputs" / "calibration"
    output_parent.mkdir(parents=True)
    run_dir = create_e6_output_namespace("cpu-test-a1", repository_root=tmp_path)
    assert run_dir == output_parent / "e6_bea_v2_cpu-test-a1"
    marker = json.loads((run_dir / "run_namespace.json").read_text(encoding="utf-8"))
    assert marker == {"schema": "e6.bea_v2.run_namespace.v1",
                      "run_id": "cpu-test-a1", "status_at_creation": "INCOMPLETE"}
    with pytest.raises(FileExistsError):
        create_e6_output_namespace("cpu-test-a1", repository_root=tmp_path)
    with pytest.raises(E6ClosedLoopError, match="filesystem-safe"):
        create_e6_output_namespace("../bad", repository_root=tmp_path)


def test_e6_event_builder_reuses_r7_shuffle_but_never_fixed_query_tape(monkeypatch):
    from experiments.calibration import r7_shuffled_correction_4class as r7

    def no_fixed_tape(*_args, **_kwargs):
        raise AssertionError("E6 must not build or consult the historical R7 fixed query tape")

    monkeypatch.setattr(r7, "_make_query_tape", no_fixed_tape)
    rows = []
    for group_index in range(2):
        group = _sha(f"source-{group_index}")
        for camera in range(1, 6):
            rows.append({"opaque_group": group, "position": len(rows),
                         "relative_id": f"synthetic/{group_index}/{camera}.png",
                         "camera_id": str(camera), "timestamp": str(camera),
                         "image_path": f"synthetic/{group_index}/{camera}.png"})
    first = make_e6_event_stream(rows)
    second = make_e6_event_stream(rows)
    assert first == second
    assert len(first) == 10
    assert validate_five_view_groups(first)["source_groups"] == 2


def test_r7_feature_encoder_returns_patch_features_and_global_cls_once():
    calls = {"load": 0, "encode": 0}

    class FakeImpl:
        @staticmethod
        def load_rgb(_path):
            calls["load"] += 1
            return object()

        @staticmethod
        def encode_pil(_backbone, _image, mode):
            calls["encode"] += 1
            assert mode == "concat"
            return [torch.ones((784, 4096), dtype=torch.float32)], torch.ones(1024)

    encoded = R7FrozenFeatureEncoder(impl=FakeImpl(), backbone=object(),
                                     feature_fuse="concat")(_events(1)[0])
    assert calls == {"load": 1, "encode": 1}
    assert encoded.feature_layers[0].shape == (784, 4096)
    assert encoded.global_cls.shape == (1024,)
    assert np.isfinite(encoded.global_cls).all()


def test_stage_a_seals_all_arms_before_feedback_and_encodes_once_per_event(tmp_path):
    events = _events(4)
    runner, arms, encoded_ids = _make_runtime(tmp_path, events)
    seal = runner.run()
    assert seal.complete and seal.event_count == len(events)
    assert len(encoded_ids) == len(events)
    assert all(len(arms[arm].engine.seen_encoded_ids) == len(events) for arm in ARM_IDS)
    for position in range(len(events)):
        ids = {id(arms[arm].engine.seen_encoded_ids[position]) for arm in ARM_IDS}
        assert len(ids) == 1
    rows = [json.loads(line) for line in Path(seal.score_log_path).read_text().splitlines()]
    assert len(rows) == len(events)
    assert all("label" not in json.dumps(row) and "anomaly_class" not in json.dumps(row)
               for row in rows)
    for arm_id in ARM_IDS:
        reader = arms[arm_id].label_reader
        assert reader.accessed == [row["position"] for row in rows
                                   if row["arm_rows"][arm_id]["query_executed"]]
        assert len(reader.accessed) == seal.arm_query_counts[arm_id]
    assert sum(seal.arm_query_counts.values()) > 0
    assert seal.arm_query_counts["Q0"] == 0
    assert seal.arm_query_counts["Q4"] == 4
    assert all(arms[arm].engine.c1_updates == seal.arm_query_counts[arm]
               for arm in ARM_IDS)
    assert seal.arm_controller_diagnostics["Q4"]["group_suppressed"] == 16
    assert rows[0]["arm_rows"]["Q5"]["components"] == rows[0]["arm_rows"]["Q6"]["components"]
    assert all(len({row["source_group_sha256"] for row in rows
                    if row["arm_rows"][arm]["query_executed"]}) ==
               seal.arm_query_counts[arm] for arm in ARM_IDS)


def test_partial_stage_a_failure_closes_writers_and_cannot_be_resumed(tmp_path):
    events = _events(1)
    runner, arms, _ = _make_runtime(tmp_path, events)
    query_seed = next(seed for seed in range(10000) if
                      _stable_uniform01(seed, "Q0", events[0].event_id_sha256, "random") <
                      P_RANDOM_FIVE_VIEW)
    arms["Q0"].controller = QueryController("Q0", seed=query_seed)

    def fail_after_query_seal(_event, _receipt):
        raise RuntimeError("synthetic disclosure-provider failure")

    arms["Q0"].label_reader.label_for = fail_after_query_seal
    with pytest.raises(RuntimeError, match="synthetic disclosure-provider failure"):
        runner.run()
    assert runner.ledger._stream.closed
    assert len(Path(runner.ledger.path).read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(FileExistsError):
        ScoreLedger(runner.ledger.path)


def test_seven_independent_engines_start_equal_then_diverge_by_query_policy(tmp_path):
    runner, arms, _ = _make_runtime(tmp_path, _events(3))
    initial = {arms[arm].engine.initial_state_sha256 for arm in ARM_IDS}
    assert initial == {"a" * 64}
    assert len({id(arms[arm].engine) for arm in ARM_IDS}) == 7
    assert len({id(arms[arm].coverage) for arm in ARM_IDS}) == 7
    runner.run()
    assert arms["Q4"].engine.normal_writes == 3
    assert arms["Q0"].engine.normal_writes == 0
    assert arms["Q4"].engine.c1_updates == 3
    assert arms["Q0"].engine.c1_updates == 0


def test_real_stage_a_rejects_synthetic_engines_unless_test_only_is_explicit(tmp_path):
    events = _events(1)
    runner, arms, _ = _make_runtime(tmp_path, events)
    with pytest.raises(E6ClosedLoopError, match="real Stage A requires"):
        StageAClosedLoop(category="audiojack", events=events,
                         feature_encoder=runner.feature_encoder,
                         arms=arms, ledger=runner.ledger)


def test_query_group_exclusion_suppresses_other_cameras_without_rescheduling(tmp_path):
    runner, arms, _ = _make_runtime(tmp_path, _events(2))
    seal = runner.run()
    rows = [json.loads(line) for line in Path(seal.score_log_path).read_text().splitlines()]
    q4 = [row["arm_rows"]["Q4"] for row in rows]
    assert [index for index, row in enumerate(q4) if row["query_executed"]] == [0, 5]
    assert sum(row["group_suppressed"] for row in q4) == 8
    assert sum(row["query_proposed"] for row in q4) == 10
    assert seal.arm_query_counts["Q4"] == 2


def test_queried_defect_updates_correction_but_never_normal_memory_or_coverage(tmp_path):
    runner, arms, _ = _make_runtime(tmp_path, _events(1), label=1)
    runner.run()
    for arm_id in ("Q4", "Q5", "Q6"):
        assert arms[arm_id].engine.c1_updates == 1
        assert arms[arm_id].engine.normal_writes == 0
        assert arms[arm_id].coverage.dynamic_size == 0


def test_unqueried_events_do_not_disclose_labels_or_supervised_write(tmp_path):
    runner, arms, _ = _make_runtime(tmp_path, _events(1))
    seal = runner.run()
    for arm_id in ("Q0", "Q1"):
        assert arms[arm_id].label_reader.accessed == []
        assert arms[arm_id].engine.c1_updates == 0
        assert arms[arm_id].engine.normal_writes == 0
        assert arms[arm_id].engine.unqueried_observations == seal.event_count


def test_queried_labels_are_written_only_to_separate_per_arm_private_files(tmp_path):
    events = _events(1)
    runner, arms, _ = _make_runtime(tmp_path, events)
    sinks = {}
    for arm_id in ARM_IDS:
        sink = PrivateFeedbackSink(tmp_path / f"private_{arm_id}.jsonl", arm_id)
        arms[arm_id].private_feedback = sink
        sinks[arm_id] = sink
    seal = runner.run()
    for sink in sinks.values():
        sink.close()
    public = Path(seal.score_log_path).read_text(encoding="utf-8")
    assert '"label"' not in public
    assert set(seal.private_feedback_sha256) == set(ARM_IDS)
    for arm_id, sink in sinks.items():
        lines = Path(sink.path).read_text(encoding="utf-8").splitlines()
        assert _sha_file(sink.path) == seal.private_feedback_sha256[arm_id]
        assert len(lines) == seal.arm_query_counts[arm_id]
        for line in lines:
            row = json.loads(line)
            assert row["arm_id"] == arm_id
            assert row["label"] == 0


def test_normal_feedback_updates_only_that_arms_coverage_and_next_threshold(tmp_path):
    runner, arms, _ = _make_runtime(tmp_path, _events(2), label=0, threshold_shift=10.0)
    seal = runner.run()
    assert arms["Q4"].coverage.dynamic_size == seal.arm_query_counts["Q4"]
    assert arms["Q0"].coverage.dynamic_size == 0
    rows = [json.loads(line) for line in Path(seal.score_log_path).read_text().splitlines()]
    assert rows[0]["arm_rows"]["Q4"]["b3_threshold"] == 0.0
    assert rows[1]["arm_rows"]["Q4"]["b3_threshold"] == 10.0
    assert rows[0]["arm_rows"]["Q4"]["prediction"] == 0


def test_stage_b_requires_all_complete_seals_before_touching_outcomes(tmp_path):
    runner, _arms, _ = _make_runtime(tmp_path, _events(1))
    seal = runner.run()
    touched = []
    with pytest.raises(E6ClosedLoopError, match="all four registered categories"):
        stage_b_evaluate(stage_a_seals={"audiojack": seal},
                         expected_categories=E6_REGISTERED_CATEGORIES,
                         labels_for_category=lambda category: touched.append(category) or [],
                         stage_b_authorized=True, authorization_id="synthetic-only")
    assert touched == []


def test_stage_b_disabled_by_default_and_never_receives_online_state(tmp_path):
    arms = None
    seals = {}
    for category in E6_REGISTERED_CATEGORIES:
        category_dir = tmp_path / category
        category_dir.mkdir()
        runner, current_arms, _ = _make_runtime(
            category_dir, _events(2), category=category)
        seals[category] = runner.run()
        if category == "audiojack":
            arms = current_arms
    touched = []
    with pytest.raises(E6ClosedLoopError, match="disabled"):
        stage_b_evaluate(stage_a_seals=seals,
                         expected_categories=E6_REGISTERED_CATEGORIES,
                         labels_for_category=lambda category: touched.append(category) or [],
                         stage_b_authorized=False)
    assert touched == []
    with pytest.raises(E6ClosedLoopError, match="private feedback seals are incomplete"):
        stage_b_evaluate(stage_a_seals=seals,
                         expected_categories=E6_REGISTERED_CATEGORIES,
                         labels_for_category=lambda category: touched.append(category) or [],
                         stage_b_authorized=True,
                         authorization_id="synthetic-test-only")
    assert touched == []
    assert arms["Q4"].engine.normal_writes == seals["audiojack"].arm_query_counts["Q4"]


def test_stage_b_rejects_complete_in_memory_receipt_without_persisted_manifest_or_seal(tmp_path):
    score_path = tmp_path / "sealed_scores.jsonl"
    score_path.write_bytes(b"")
    feedback = {}
    for arm in ARM_IDS:
        feedback_path = tmp_path / f"private_feedback_{arm}.jsonl"
        feedback_path.write_bytes(b"")
        feedback[arm] = _sha("")
    diagnostics = {arm: {"arm_id": arm, "events_observed": 0} for arm in ARM_IDS}
    receipt = StageASeal(
        category="audiojack", event_count=0, score_log_path=str(score_path),
        score_log_sha256=_sha(""), private_feedback_sha256=feedback,
        arm_query_counts={arm: 0 for arm in ARM_IDS},
        arm_controller_diagnostics=diagnostics, complete=True)
    with pytest.raises(E6ClosedLoopError, match="durable Stage A manifest and seal are required"):
        _read_sealed_score_log(score_path, receipt)


def test_stage_b_tamper_or_inconsistent_query_count_fails_before_labels(tmp_path):
    seals = {}
    for category in E6_REGISTERED_CATEGORIES:
        category_dir = tmp_path / category
        category_dir.mkdir()
        runner, _arms, _ = _make_runtime(category_dir, _events(1), category=category)
        seals[category] = runner.run()
    audiojack_path = Path(seals["audiojack"].score_log_path)
    audiojack_path.write_text(audiojack_path.read_text() + " ")
    touched = []
    with pytest.raises(E6ClosedLoopError, match="digest"):
        stage_b_evaluate(stage_a_seals=seals,
                         expected_categories=E6_REGISTERED_CATEGORIES,
                         labels_for_category=lambda category: touched.append(category) or [],
                         stage_b_authorized=True,
                         authorization_id="synthetic-only")
    assert touched == []


def _real_iad_synthetic_events():
    from experiments.calibration import r7_shuffled_correction_4class as r7
    group_key = "audiojack_0001_OK"
    group_sha = hashlib.sha256(("official-sample-group\0" + group_key).encode()).hexdigest()
    events, records = [], []
    for position, camera in enumerate(range(1, 6)):
        timestamp = f"202611040000{position:02d}"
        filename = f"audiojack_0001_OK_C{camera}_{timestamp}.jpg"
        relative = f"OK/S0001/{filename}"
        image_id = r7._identity_hash(relative)
        event_id = hashlib.sha256(
            f"r7-event\0{position}\0{image_id}\0{group_sha}".encode()).hexdigest()
        events.append(EventIdentity(
            position=position, original_position=position, event_id_sha256=event_id,
            image_identity_sha256=image_id, source_group_sha256=group_sha,
            relative_id=relative, camera_id=str(camera), timestamp=timestamp,
            image_path=relative))
        records.append({"image_path": relative, "anomaly_class": "OK",
                        "mask_path": {"this": "is skipped, never decoded"}})
    return tuple(events), {"meta": {"normal_class": "OK"}, "test": records}


def test_on_demand_label_gate_requires_seal_identity_and_discloses_once(tmp_path):
    events, test_json = _real_iad_synthetic_events()
    json_path = tmp_path / "synthetic_test.json"
    json_path.write_text(json.dumps(test_json), encoding="utf-8")
    validate_test_image_projection(json_path, "audiojack", events)
    ledger = ScoreLedger(tmp_path / "scores.jsonl")
    arms = {arm: {"query_executed": arm == "Q4"} for arm in ARM_IDS}
    with pytest.raises(E6ClosedLoopError, match="no durably sealed"):
        ledger.query_receipt("Q4", events[0])
    ledger.seal_event(events[0], arms)
    reader = OnDemandLabelReader(json_path, "audiojack", _sha_file(json_path),
                                 events, "Q4", ledger)
    with pytest.raises(E6ClosedLoopError, match="no durably sealed"):
        ledger.query_receipt("Q0", events[0])
    wrong_event = EventIdentity(**{**events[0].__dict__, "source_group_sha256": "f" * 64})
    with pytest.raises(E6ClosedLoopError, match="identity"):
        ledger.query_receipt("Q4", wrong_event)
    wrong_original = EventIdentity(**{**events[0].__dict__, "original_position": 1})
    with pytest.raises(E6ClosedLoopError, match="identity"):
        ledger.query_receipt("Q4", wrong_original)
    receipt = ledger.query_receipt("Q4", events[0])
    assert reader.label_for(events[0], receipt) == 0
    assert reader.decoded_positions == [0]
    with pytest.raises(E6ClosedLoopError, match="duplicate label"):
        reader.label_for(events[0], receipt)
    reader.close()
    ledger.close()


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_normal_coverage_uses_four_supports_then_only_queried_normals():
    coverage = _coverage()
    initial_sha = coverage.state_sha256()
    pct, info = coverage.novelty_percentile(np.asarray([1, 0, 0, 0], dtype=np.float32))
    assert 0.0 <= pct <= 1.0 and info["support_size"] == 4
    assert info["dynamic_size"] == 0
    coverage.observe_normal(np.asarray([1, 1, 0, 0], dtype=np.float32))
    assert coverage.dynamic_size == 1
    assert coverage.state_sha256() != initial_sha


def test_coverage_percentile_matches_frozen_v22_reference():
    v22 = pytest.importorskip("test_phase3_AIF_simplified_v22")
    support = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0],
                          [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    ours = NormalCoverageMemory(support)
    reference = v22.NormalCoverageMemory.from_support_feats(
        [torch.from_numpy(row.copy()) for row in support])
    for query in (np.asarray([1, 0.1, 0, 0], dtype=np.float32),
                  np.asarray([0.1, 0.1, 0.9, 0], dtype=np.float32)):
        ours_pct, _ = ours.novelty_percentile(query)
        ref_pct, _ = reference.novelty_percentile(torch.from_numpy(query.copy()))
        assert ours_pct == pytest.approx(ref_pct, abs=1e-7)
    normal = np.asarray([0.7, 0.7, 0.0, 0.0], dtype=np.float32)
    ours.observe_normal(normal)
    reference.observe_normal(torch.from_numpy(normal.copy()))
    ours_pct, _ = ours.novelty_percentile(np.asarray([0.6, 0.8, 0, 0], dtype=np.float32))
    ref_pct, _ = reference.novelty_percentile(torch.tensor([0.6, 0.8, 0, 0]))
    assert ours_pct == pytest.approx(ref_pct, abs=1e-7)


def test_execution_identity_requires_reviewed_commit_branch_and_pinned_sources(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    expected = "2a911dda634a7fe8a73347d4b2016d20b7980461"
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[1:] == ["branch", "--show-current"]:
            output = "calibration-stage0-contract-20261029\n"
        elif args[1:] == ["rev-parse", "HEAD"]:
            output = expected + "\n"
        else:
            output = ""
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(subprocess, "run", fake_run)
    receipt = verify_e6_execution_identity(expected, root=root)
    assert receipt["approved_e6_commit"] == expected
    assert receipt["r7_helper_source_sha256"] == dict(R7_HELPER_SOURCE_SHA256)
    assert len(calls) == 3

    with pytest.raises(E6ClosedLoopError, match="branch/HEAD"):
        verify_e6_execution_identity("f" * 40, root=root)

    import experiments.calibration.e6_bea_closed_loop as e6_module
    original_sha_file = e6_module._sha256_file
    monkeypatch.setattr(e6_module, "_sha256_file", lambda _path: "0" * 64)
    with pytest.raises(E6ClosedLoopError, match="source SHA changed"):
        verify_e6_execution_identity(expected, root=root)
    monkeypatch.setattr(e6_module, "_sha256_file", original_sha_file)

    def wrong_branch(args, **kwargs):
        if args[1:] == ["branch", "--show-current"]:
            output = "main\n"
        elif args[1:] == ["rev-parse", "HEAD"]:
            output = expected + "\n"
        else:
            output = ""
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(subprocess, "run", wrong_branch)
    with pytest.raises(E6ClosedLoopError, match="branch/HEAD"):
        verify_e6_execution_identity(expected, root=root)

    def dirty_source(args, **kwargs):
        if args[1:] == ["branch", "--show-current"]:
            output = "calibration-stage0-contract-20261029\n"
        elif args[1:] == ["rev-parse", "HEAD"]:
            output = expected + "\n"
        else:
            output = " M experiments/calibration/e6_bea_closed_loop.py\n"
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(subprocess, "run", dirty_source)
    with pytest.raises(E6ClosedLoopError, match="uncommitted changes"):
        verify_e6_execution_identity(expected, root=root)


def test_native_dense_score_and_common_greedy_writer_use_r7_helpers_on_synthetic_inputs():
    from experiments.calibration import r1_matched_correction_pilot as r1

    class SyntheticBank:
        corr_cfg = SimpleNamespace(enabled=False)
        use_defect_bank = False

        def score_single_view(self, **kwargs):
            assert kwargs["cls_name"] == "p0" and kwargs["view"] == "id"
            amap = torch.arange(784, dtype=torch.float32)
            summary = {"layer0": {"topk_patch_idx": np.arange(32, dtype=np.int64),
                                  "topk_dN_max": 0.8, "topk_dN_mean": 0.2}}
            return [amap], summary

        def fuse_layers(self, maps, fusion):
            assert fusion == "mean"
            return maps[0]

        def aggregate_map(self, amap, **kwargs):
            assert kwargs["agg"] == "max"
            return float(torch.max(amap))

    class SyntheticImpl:
        @staticmethod
        def _grid_hw_from_npatches(count, aspect=None):
            assert count == 784
            return 28, 28

        @staticmethod
        def ViewSpec(source, target):
            return (source, target)

        @staticmethod
        def align_map_to_identity(amap, view_spec, height, width):
            assert view_spec == ("id", "id") and (height, width) == (28, 28)
            return amap

    cfg = SimpleNamespace(topk_patches=32, image_aggregation="max",
                          image_topk=1, image_quantile=0.99)
    base_map, top_ids, score, summary = r1._native_dense_score(
        SyntheticImpl(), SyntheticBank(), SimpleNamespace(height=28, width=28),
        [torch.zeros((784, 3), dtype=torch.float32)], cfg)
    assert base_map.shape == (784,) and np.array_equal(top_ids, np.arange(32))
    assert score == float(np.max(base_map)) == 783.0
    assert summary["layer0"]["topk_dN_max"] == 0.8

    candidate_ids = np.arange(32, dtype=np.int64)
    candidate_features = np.ones((32, 3), dtype=np.float32)
    calls = []

    def eligibility(**kwargs):
        calls.append(("eligibility", kwargs["score"], kwargs["theta"]))
        return True, 32, "synthetic-eligible", {}

    def update(**kwargs):
        calls.append(("update", kwargs["position"], kwargs["score"]))
        return {"accepted_patches_delta": 1}

    action = r1.common_c0_greedy_writer_action(
        position=0, c0_score=0.25, theta=0.30, sigma=0.10,
        c0_prediction=0, candidate_ids=candidate_ids,
        candidate_features=candidate_features,
        feature_layers=[torch.zeros((784, 3), dtype=torch.float32)],
        queried=True, disclosed_label=0, eligibility=eligibility, update=update)
    assert action["action"] == "native_greedy_normal_update"
    assert [item[0] for item in calls] == ["eligibility", "update"]
    calls.clear()
    defect = r1.common_c0_greedy_writer_action(
        position=1, c0_score=0.25, theta=0.30, sigma=0.10,
        c0_prediction=0, candidate_ids=candidate_ids,
        candidate_features=candidate_features,
        feature_layers=[torch.zeros((784, 3), dtype=torch.float32)],
        queried=True, disclosed_label=1, eligibility=eligibility, update=update)
    assert defect["action"] == "verified_defect_no_normal_write"
    assert calls == []
    unqueried = r1.common_c0_greedy_writer_action(
        position=2, c0_score=0.25, theta=0.30, sigma=0.10,
        c0_prediction=0, candidate_ids=candidate_ids,
        candidate_features=candidate_features,
        feature_layers=[torch.zeros((784, 3), dtype=torch.float32)],
        queried=False, disclosed_label=None, eligibility=eligibility, update=update)
    assert unqueried["action"] == "no_normal_memory_write"
    assert calls == []


def test_native_c1_residuals_use_r7_causal_source_exclusion_on_cpu():
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r7_shuffled_correction_4class as r7
    v22 = pytest.importorskip("test_phase3_AIF_simplified_v22")

    cfg = v22.CorrectionConfig(enabled=False, K_correction=1024)
    codebook = v22.SignedCorrectionCodebook(1024, 4, torch.device("cpu"), cfg)
    codebook.n = 1
    codebook.protos[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    codebook.beta[0] = 0.10
    codebook.radius[0] = 0.10
    codebook.kappa[0] = 1.0
    codebook.count[0] = 1
    codebook.last_step[0] = 0
    bank = SimpleNamespace(corr_cb={"p0": {"id": [codebook]}},
                           corr_cfg=cfg, device=torch.device("cpu"))
    current_group = "b" * 64
    anchor_group = "a" * 64
    selected = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 32)

    delta, diagnostics = r1._c1_score_delta(
        v22, bank, selected, {0: {anchor_group}}, current_group, event_position=1)
    assert delta.shape == (32,) and np.all(delta > 0.09)
    assert diagnostics["eligible_anchor_slots"] == 1
    assert diagnostics["same_source_excluded_anchor_slots"] == 0

    excluded, diagnostics = r1._c1_score_delta(
        v22, bank, selected, {0: {current_group}}, current_group, event_position=1)
    assert np.array_equal(excluded, np.zeros(32, dtype=np.float32))
    assert diagnostics["eligible_anchor_slots"] == 0
    assert diagnostics["same_source_excluded_anchor_slots"] == 1
