from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys

import pytest
import torch

from experiments.calibration import e123_recovery as e123
from experiments.calibration.protocol import EvidenceAnchor, EvidenceSnapshot, Feedback, Observation


class FakeCodebook:
    def __init__(self, capacity: int = 256):
        self.K = capacity
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.protos = torch.empty((0, 2), dtype=self.dtype)
        self.meta_count = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        self.update_calls = []

    def size(self):
        return int(self.protos.shape[0])

    def update(self, features, **kwargs):
        """Spy implementation for the native Utility STM call contract."""
        self.update_calls.append((features.clone(), dict(kwargs)))
        self.protos = torch.cat((self.protos, features.to(self.dtype)), dim=0)
        return {"n_added": int(features.shape[0]), "n_replaced": 0}


class FakeBank:
    def __init__(self, capacity: int = 256):
        self.normal_fixed = {"support": torch.tensor([[0.25, 0.75]], dtype=torch.float32)}
        self.normal_dyn = {"p0": [{"id": [FakeCodebook(capacity)]}]}
        self.corr_cfg = type("FakeCorrectionConfig", (), {"enabled": True})()
        self.use_defect_bank = True


class FakeImpl:
    @staticmethod
    def l2_normalize(value):
        return torch.nn.functional.normalize(value.float(), dim=1)


def _candidate_stream(n=300):
    ids = list(range(n))
    rows = torch.tensor([[float(i), float(i) + 0.25] for i in ids], dtype=torch.float32)
    return {0: (ids, rows)}


def _reference_reservoir(values, capacity, seed):
    rng = random.Random(seed)
    slots = []
    for seen, value in enumerate(values, start=1):
        if len(slots) < capacity:
            slots.append(value)
        else:
            slot = rng.randrange(seen)
            if slot < capacity:
                slots[slot] = value
    return slots


def _reference_random_replacement(values, capacity, seed):
    rng = random.Random(seed)
    slots = []
    for value in values:
        if len(slots) < capacity:
            slots.append(value)
        else:
            slots[rng.randrange(capacity)] = value
    return slots


def _valid_global_seal():
    import math

    condition_seals = [{
        "condition_id": f"{category}_s{seed}_{order}",
        "seal_sha256": f"{index:064x}",
        "score_rows_sha256": f"{index + 100:064x}",
    } for index, (category, seed, order) in enumerate(
        (c, s, o) for c in e123.CATEGORIES
        for s in e123.SUPPORT_SEEDS for o in e123.STREAM_ORDERS)]
    seal = {"schema": e123.GLOBAL_SEAL_SCHEMA, "status": "SEALED_COMPLETE",
            "run_id": "synthetic", "condition_count": 16,
            "event_count": sum(e123.EXPECTED_COUNTS.values()) * 4,
            "query_count": sum(
                math.floor(0.03 * n + 0.5) * 4
                for n in e123.EXPECTED_COUNTS.values()),
            "condition_seals": condition_seals}
    seal["seal_sha256"] = e123.sha256_json(seal)
    return seal


def _observe(state, candidates):
    return state.observe(
        impl=FakeImpl(), common_candidates=candidates,
        event_position=17, static_score=0.2, theta0=0.5, sigma0=0.1,
        eligibility_info={"write_value": 0.0, "boundary": 0.0})


def test_regime_b_roles_require_exact_counts_and_disjoint_groups():
    supports = [f"s{i}" for i in range(4)]
    c64 = [f"c{i}" for i in range(64)]
    test = [f"t{i}" for i in range(20)]
    e123.validate_role_disjointness(supports, c64, test)
    with pytest.raises(e123.E123Error):
        e123.validate_role_disjointness(supports, c64, [*test, supports[0]])
    with pytest.raises(e123.E123Error):
        e123.validate_role_disjointness(supports[:-1], c64, test)


def test_query_tape_and_arm_registry_are_fixed_and_deterministic():
    from experiments.calibration.profiles import QueryProtocolConfig
    from experiments.calibration.protocol import TapeRecord
    from experiments.calibration.query_tape import build_qa_ts, verify_tape

    records = [TapeRecord(i, f"group-{i}", 1 + i % 5, f"image-{i}",
                          f"2026-01-{i % 28 + 1:02d}") for i in range(512)]
    first = build_qa_ts(records, QueryProtocolConfig())
    second = build_qa_ts(records, QueryProtocolConfig())
    verify_tape(first, records, config=QueryProtocolConfig())
    assert first.digest == second.digest
    assert first.positions == second.positions
    assert len(first.positions) == 15
    assert len(first.queried_groups) == 15
    expected_arms = ("STATIC", *(f"{method}_K{capacity}"
                                 for method in e123.MUTABLE_METHODS
                                 for capacity in e123.CAPACITIES))
    assert e123.CONFIGURATION_IDS == expected_arms
    # One condition owns one immutable event order and one tape; every arm uses them.
    per_arm_tapes = {arm: first.digest for arm in e123.CONFIGURATION_IDS}
    per_arm_orders = {arm: tuple(record.position for record in records)
                      for arm in e123.CONFIGURATION_IDS}
    assert len(set(per_arm_tapes.values())) == 1
    assert len(set(per_arm_orders.values())) == 1


def test_common_candidate_adapter_gives_identical_ids_and_feature_object_to_arms():
    raw_features = [torch.tensor([[1.0, 0.0], [0.0, 1.0],
                                  [1.0, 1.0], [-1.0, 1.0]])]
    candidates = e123.prepare_common_candidates(
        FakeImpl(), raw_features, {0: [1, 3]}, device="cpu")
    ids, feature_tensor = candidates[0]
    assert ids == [1, 3]
    assert feature_tensor.shape == (2, 2)

    canonical_bank = FakeBank()
    static_before = canonical_bank.normal_fixed["support"].clone()
    states = {
        name: e123.E123MemoryState(
            method=name, capacity=256,
            bank=e123._clone_bank_with_capacity(canonical_bank, 256), seed=42)
        for name in ("FIFO", "RESERVOIR", "RANDOM_REPLACEMENT")
    }
    result = e123.dispatch_common_write(
        states, eligible=True,
        observe_kwargs={
            "impl": FakeImpl(), "common_candidates": candidates,
            "event_position": 1, "static_score": 0.1, "theta0": 0.5,
            "sigma0": 0.1, "eligibility_info": {},
        })
    assert set(result) == set(states)
    assert all(changed for changed, _ in result.values())
    assert all(state.candidate_patches == 2 for state in states.values())
    assert len({id(e123._bank_dynamic_codebooks(state.bank)[0].protos)
                for state in states.values()}) == 3
    assert torch.equal(canonical_bank.normal_fixed["support"], static_before)
    assert all(torch.equal(state.bank.normal_fixed["support"], static_before)
               for state in states.values())
    assert all(state.bank.normal_fixed is canonical_bank.normal_fixed
               for state in states.values())
    assert "STATIC" not in states


def test_fifo_reservoir_and_random_replacement_have_distinct_correct_semantics():
    candidates = _candidate_stream(300)
    states = {
        name: e123.E123MemoryState(
            method=name, capacity=256, bank=FakeBank(), seed=7)
        for name in ("FIFO", "RESERVOIR", "RANDOM_REPLACEMENT")
    }
    for state in states.values():
        _observe(state, candidates)
    actual = {
        name: e123._bank_dynamic_codebooks(state.bank)[0].protos[:, 0].tolist()
        for name, state in states.items()
    }
    values = [float(i) for i in range(300)]
    # FIFO uses a circular physical layout: cursor=44 is the oldest slot.
    assert actual["FIFO"] == values[256:] + values[44:256]
    assert states["FIFO"].fifo_cursor == [44]
    assert actual["RESERVOIR"] == _reference_reservoir(values, 256, 7)
    assert actual["RANDOM_REPLACEMENT"] == _reference_random_replacement(values, 256, 7)
    assert actual["RESERVOIR"] != actual["RANDOM_REPLACEMENT"]
    audits = {name: state.audit() for name, state in states.items()}
    assert all(audit["layers"][0]["occupied_slots"] == 256
               for audit in audits.values())
    assert all(audit["layers"][0]["capacity"] == 256
               for audit in audits.values())
    assert all(audit["prototype_tensor_bytes"] == 256 * 2 * 4
               for audit in audits.values())
    assert all(audit["allocated_persistent_tensor_bytes"] ==
               audit["prototype_tensor_bytes"] + audit["metadata_tensor_bytes"]
               for audit in audits.values())
    assert all(audit["metadata_runtime_bytes_estimate"] > 0
               for audit in audits.values())
    assert all(audit["metadata_tensor_bytes"] >=
               e123._tensor_storage_bytes(
                   e123._bank_dynamic_codebooks(states[name].bank)[0].meta_count)
               for name, audit in audits.items())
    assert audits["RESERVOIR"]["rejected_patches"] > 0
    assert audits["RANDOM_REPLACEMENT"]["rejected_patches"] == 0


def test_random_policy_transitions_repeat_deterministically():
    candidates = _candidate_stream(300)
    for method in ("FIFO", "RESERVOIR", "RANDOM_REPLACEMENT"):
        left = e123.E123MemoryState(method=method, capacity=256,
                                    bank=FakeBank(), seed=101)
        right = e123.E123MemoryState(method=method, capacity=256,
                                     bank=FakeBank(), seed=101)
        _observe(left, candidates)
        _observe(right, candidates)
        assert left.state_digest == right.state_digest
        assert torch.equal(e123._bank_dynamic_codebooks(left.bank)[0].protos,
                           e123._bank_dynamic_codebooks(right.bank)[0].protos)


def test_c0_eligibility_excludes_unqueried_and_defect_feedback():
    calls = []

    def gate(**kwargs):
        calls.append(kwargs)
        return True, 8, "C0_NORMAL_GATE", {"write_value": 1.0, "boundary": 0.2}

    assert e123.resolve_common_eligibility(
        queried=False, label=None, c0_score=0.1, theta0=0.5,
        sigma0=0.1, eligibility=gate) == (False, 0, "unqueried_event", None)
    assert e123.resolve_common_eligibility(
        queried=True, label=1, c0_score=0.1, theta0=0.5,
        sigma0=0.1, eligibility=gate) == (False, 0, "verified_defect", None)
    assert calls == []
    allowed = e123.resolve_common_eligibility(
        queried=True, label=0, c0_score=0.7, theta0=0.5,
        sigma0=0.1, eligibility=gate)
    assert allowed[:3] == (True, 8, "C0_NORMAL_GATE")
    assert calls[0]["score"] == 0.7
    assert calls[0]["pred_defect"] is True

    state = e123.E123MemoryState(method="FIFO", capacity=256,
                                 bank=FakeBank(), seed=0)
    before = state.audit()
    assert e123.dispatch_common_write(
        {"FIFO_K256": state}, eligible=False,
        observe_kwargs={"impl": FakeImpl(), "common_candidates": _candidate_stream(1),
                        "event_position": 1, "static_score": 0.1,
                        "theta0": 0.5, "sigma0": 0.1,
                        "eligibility_info": {}}) == {}
    assert state.audit()["candidate_events"] == before["candidate_events"] == 0


def test_all_arms_receive_same_ordered_candidate_payload_and_static_is_not_written():
    candidates = _candidate_stream(3)
    original_ids = list(candidates[0][0])
    original_features = candidates[0][1].clone()
    states = {
        name: e123.E123MemoryState(method=name, capacity=256,
                                   bank=FakeBank(), seed=4)
        for name in e123.MUTABLE_METHODS
    }
    payload = {"impl": FakeImpl(), "common_candidates": candidates,
               "event_position": 5, "static_score": 0.2, "theta0": 0.5,
               "sigma0": 0.1,
               "eligibility_info": {"write_value": 0.0, "boundary": 0.0}}
    e123.dispatch_common_write(states, eligible=True, observe_kwargs=payload)
    assert candidates[0][0] == original_ids
    assert torch.equal(candidates[0][1], original_features)
    assert all(state.candidate_patches == len(original_ids)
               for state in states.values())
    utility_codebook = e123._bank_dynamic_codebooks(states["UTILITY_STM"].bank)[0]
    assert len(utility_codebook.update_calls) == 1
    assert torch.equal(utility_codebook.update_calls[0][0], original_features)
    assert "STATIC" not in e123.MUTABLE_METHODS


def test_score_seal_precedes_query_label_decode_and_unqueried_never_calls_reader():
    timeline = []

    class Scores:
        def flush(self, *, sync):
            assert sync is True
            timeline.append("score_fsync")

    def reader(category, position, source_key):
        timeline.append(f"decode:{position}")
        return 0

    label = e123.disclose_after_score_seal(
        Scores(), queried=True, label_reader=reader, category="audiojack",
        original_position=7, source_key="sample-7")
    assert label == 0
    assert timeline == ["score_fsync", "decode:7"]
    timeline.clear()
    assert e123.disclose_after_score_seal(
        Scores(), queried=False, label_reader=reader, category="audiojack",
        original_position=8, source_key="sample-8") is None
    assert timeline == []


def test_bounded_queried_reader_decodes_only_selected_test_record(tmp_path, monkeypatch):
    import experiments.calibration.stage1a_p12_label_provider as provider

    path = tmp_path / "audiojack.json"
    path.write_text(
        '{"meta":{"normal_class":"good"},"train":[],"test":['
        '{"image_path":"one","anomaly_class":"good"}, '
        '{"image_path":"two","anomaly_class":"scratch"}   ]}',
        encoding="utf-8")
    group_hash = hashlib.sha256(b"official-sample-group\0sample-two").hexdigest()
    original_decoder = provider._decode_selected_label
    decoded = []

    def counting_decoder(data, start, stop, normal_class):
        decoded.append((start, stop))
        return original_decoder(data, start, stop, normal_class)

    monkeypatch.setattr(provider, "_decode_selected_label", counting_decoder)
    reader = e123.RandomAccessQueriedLabelReader(
        category="audiojack", path=path, expected_source_sha256=e123.sha256_file(path),
        allowed_positions=[1], group_sha256_by_position={1: group_hash})
    try:
        assert reader("audiojack", 1, "sample-two") == 1
        assert reader.decoded_positions == [1]
        assert len(decoded) == 1
        with pytest.raises(e123.E123Error):
            reader("audiojack", 0, "sample-one")
    finally:
        reader.close()


def test_historical_loco_label_reader_bounds_final_record_before_trailing_whitespace(
        tmp_path, monkeypatch):
    import experiments.calibration.stage1a_p12_label_provider as provider

    path = tmp_path / "audiojack.json"
    path.write_text(
        '{"meta":{"normal_class":"good"},"train":[],"test":['
        '{"image_path":"one","anomaly_class":"good"}, '
        '{"image_path":"two","anomaly_class":"scratch"}   ]}',
        encoding="utf-8")
    source = provider._RestrictedTestArray(path)
    try:
        first_start = provider._ws(source._data, source._cursor)
        cursor = provider._ws(source._data,
                              provider._skip_value(source._data, first_start) + 1)
        stop = provider._skip_value(source._data, cursor)
        with pytest.raises(provider.QueriedLabelAccessError,
                           match="Unexpected bytes after queried test record"):
            provider._decode_selected_label(
                source._data, cursor, stop, source.normal_class)
    finally:
        source.close()

    original_decoder = provider._decode_selected_label
    decoder_inputs = []

    def recording_decoder(data, start, stop, normal_class):
        decoder_inputs.append(type(data))
        return original_decoder(data, start, stop, normal_class)

    monkeypatch.setattr(provider, "_decode_selected_label", recording_decoder)
    labels = e123._read_historical_development_labels(
        "audiojack", path, expected_source_sha256=e123.sha256_file(path),
        expected_count=2)
    assert labels == [0, 1]
    assert decoder_inputs == [bytes, bytes]
    with pytest.raises(e123.E123Error, match="source changed"):
        e123._read_historical_development_labels(
            "audiojack", path, expected_source_sha256="0" * 64, expected_count=2)


def test_stage_b_reader_requires_global_stage_a_seal(tmp_path):
    path = tmp_path / "audiojack.json"
    path.write_text(
        '{"meta":{"normal_class":"good"},"train":[],"test":['
        '{"image_path":"one","anomaly_class":"good"}, '
        '{"image_path":"two","anomaly_class":"scratch"}   ]}',
        encoding="utf-8")
    groups = {
        position: hashlib.sha256(
            f"official-sample-group\0sample-{position}".encode()).hexdigest()
        for position in range(2)
    }
    reader = e123.RandomAccessQueriedLabelReader(
        category="audiojack", path=path, expected_source_sha256=e123.sha256_file(path),
        allowed_positions=[0, 1], group_sha256_by_position=groups)
    try:
        with pytest.raises(e123.E123Error, match="global Stage A seal"):
            reader.stage_b_all_labels(global_seal={"schema": e123.GLOBAL_SEAL_SCHEMA,
                                                   "status": "INCOMPLETE",
                                                   "condition_count": 1})
        valid = _valid_global_seal()
        assert reader.stage_b_all_labels(global_seal=valid) == [0, 1]
    finally:
        reader.close()


def test_fixed_offset_is_constant_and_threshold_does_not_change_auc():
    coefficient = -0.25
    values = [e123.fixed_offset_threshold(0.5, 0.1, coefficient) for _ in range(100)]
    assert len(set(values)) == 1
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.6, 0.4, 0.9]
    threshold_a = e123._binary_metrics(labels, scores, 0.5)
    threshold_b = e123._binary_metrics(labels, scores, 0.3)
    assert threshold_a["auroc"] == threshold_b["auroc"]
    assert threshold_a["fpr"] != threshold_b["fpr"] or threshold_a["fnr"] != threshold_b["fnr"]


def test_e1_frozen_and_b3_cells_keep_the_same_static_and_greedy_score_vectors():
    labels = [0, 1, 0, 1, 1, 0]
    static_scores = [0.1, 0.8, 0.2, 0.7, 0.6, 0.4]
    greedy_scores = [0.2, 0.9, 0.1, 0.25, 0.55, 0.3]
    theta0 = 0.5
    static_frozen = e123._binary_metrics(labels, static_scores, theta0)
    static_b3 = e123._metrics_from_predictions(
        labels, static_scores, [int(score >= theta0) for score in static_scores])
    greedy_frozen = e123._binary_metrics(labels, greedy_scores, theta0)
    greedy_b3 = e123._metrics_from_predictions(
        labels, greedy_scores, [int(score >= threshold)
                                for score, threshold in zip(
                                    greedy_scores, [0.45, 0.55, 0.45, 0.55, 0.45, 0.55])])
    assert static_b3 == static_frozen
    assert greedy_b3["auroc"] == greedy_frozen["auroc"]
    assert static_frozen["auroc"] != greedy_frozen["auroc"]


def test_e3_controllers_replay_one_greedy_score_vector_with_query_only_feedback():
    events = tuple(e123.E123Event(
        position=i, original_position=i, relative_id=f"image-{i}",
        image_identity_sha256=f"{i:064x}",
        source_group_sha256=f"{i + 100:064x}", camera_id=0,
        timestamp="", image_path="", sample_key=f"sample-{i}")
        for i in range(6))
    condition = e123.Condition(
        "audiojack", 0, "json", events, frozenset({1, 3, 5}),
        "a" * 64, "b" * 64, (), (), (), (), "c" * 64)
    greedy_scores = [0.1, 0.8, 0.2, 0.7, 0.6, 0.4]
    rows = [{
        "event_id_sha256": f"{i + 200:064x}", "camera_id": 0,
        "scores": {"GREEDY_K1024": score},
        "greedy1024_memory_version_before": 0,
        "greedy1024_transition_sha256_before": "state0",
        "b3_greedy1024_threshold_before": 0.5,
    } for i, score in enumerate(greedy_scores)]
    feedback = [{"position": position, "label": label}
                for position, label in ((1, 1), (3, 1), (5, 0))]
    labels = [0, 1, 0, 1, 1, 0]
    c64 = [0.2 + i * 0.005 for i in range(64)]
    metrics, predictions = e123._replay_threshold_controllers(
        condition, rows, c64, theta0=0.5, sigma0=0.1,
        fixed_offset=-0.25, feedback=feedback, labels=labels)
    aucs = {row["auroc"] for row in metrics.values()}
    assert len(aucs) == 1
    assert metrics["FIXED_OFFSET"]["threshold_update_count"] == 0
    assert metrics["FROZEN_C64"]["threshold_update_count"] == 0
    assert all(len(values) == len(greedy_scores) for values in predictions.values())


def test_output_budget_prewrite_reserve_and_nonoverwrite(tmp_path, monkeypatch):
    base = tmp_path / "calibration"
    base.mkdir()
    monkeypatch.setattr(e123, "OUTPUT_BASE", base)
    root = base / "recovery_e123_synthetic"
    budget = e123.OutputBudget(root, create=True, max_bytes=8192,
                               failure_reserve_bytes=4096)
    budget.write_once("E123_contract.json", b"x" * 4095)
    with pytest.raises(e123.E123Error, match="pre-write byte limit"):
        budget.write_once("E123_report.json", b"y")
    second_root = base / "recovery_e123_nonoverwrite"
    second = e123.OutputBudget(second_root, create=True, max_bytes=8192,
                               failure_reserve_bytes=4096)
    second.write_once("E123_contract.json", b"first")
    with pytest.raises(FileExistsError):
        second.write_once("E123_contract.json", b"replacement")
    budget.failure("E123_failure.json", {"status": "synthetic_failure"})
    assert budget.verify_complete_namespace()["file_bytes"] <= 8192
    (root / "unregistered.bin").write_bytes(b"escape")
    with pytest.raises(e123.E123Error, match="Unaccounted output file"):
        budget.verify_complete_namespace()


def test_b1_ignores_unqueried_feedback_and_b3_updates_only_from_protected_c64():
    from experiments.calibration.controllers import (
        OriginalGuardedSidecar, RollingQ90QueriedNormal, StateConsistentQuantile)

    b1 = RollingQ90QueriedNormal(0.5, window=4, min_count=2)
    observation = Observation(0, "e0", 1, 0.7, 0, "state0")
    pred = b1.predict(observation)
    b1.observe_feedback(observation, pred, Feedback(0, queried=False, label=None))
    assert list(b1.history) == []
    b1.observe_feedback(observation, pred, Feedback(0, queried=True, label=0))
    assert list(b1.history) == [0.7]

    b2 = OriginalGuardedSidecar([0.2 + i * 0.001 for i in range(64)], 0.5)
    b2_before = b2.theta
    unqueried = Observation(1, "e1", 1, 0.99, 0, "state0")
    unqueried_prediction = b2.predict(unqueried)
    b2.observe_score(unqueried)
    b2.observe_feedback(unqueried, unqueried_prediction,
                        Feedback(1, queried=False, label=None))
    assert b2.finalize_step() == b2_before
    assert b2.adaptation["score_only_normal_fallback"].startswith("disabled")

    b3 = StateConsistentQuantile(0.5, delta=0.01, min_normals=16)
    anchors = tuple(EvidenceAnchor(f"c{i}", 0.4 + i * 0.001, 1, True)
                    for i in range(64))
    b3.on_memory_change(EvidenceSnapshot(1, anchors, (), {"protected_c64": True}, 1))
    theta_after = b3.finalize_step()
    assert abs(theta_after - 0.5) <= 0.0100001
    assert b3.status == "CURRENT_C64_Q90"


def test_incomplete_or_modified_global_seal_cannot_promote_stage_b():
    with pytest.raises(e123.E123Error):
        e123.require_complete_stage_a({"schema": e123.GLOBAL_SEAL_SCHEMA,
                                       "status": "INCOMPLETE",
                                       "condition_count": 16,
                                       "condition_seals": [{} for _ in range(16)]})
    payload = _valid_global_seal()
    e123.require_complete_stage_a(payload)
    payload["query_count"] = 15
    with pytest.raises(e123.E123Error):
        e123.require_complete_stage_a(payload)


@pytest.mark.parametrize("load_behavior,digest_behavior", [
    ("success", "match"), ("success", "mismatch"), ("failure", "unused"),
])
def test_native_profile_hashes_underlying_dino_model_and_restores_hub(
        monkeypatch, load_behavior, digest_behavior):
    from dataclasses import dataclass
    from types import SimpleNamespace

    import experiments.p0_protocol_audit as p0
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r3_fullstream_2x2 as r3

    expected_state = "1f26464ba4055403e550f1d5c1333fb012631d98661df6993cabea730d17e3f9"
    expected_hub = "b7c0037f90671c4932393bd56de8c8a205b29a6eb22dea563b6c8a88c471da15"
    expected_checkpoint = "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
    fake_torch_version = str(torch.__version__)
    expected_receipt = {
        "status": "ENGINEERING_PASS", "task": "R1",
        "profile_id": "r1_switch_s0_512_digest_contract_v1",
        "hub_source_tree_sha256": expected_hub,
        "checkpoint_sha256": expected_checkpoint,
        "loaded_model_state_sha256": expected_state,
        "runtime_profile": {"cuda_device_name": "mock-verified-device",
                             "torch_version": fake_torch_version},
    }
    model_source = {
        "runtime_profile": {"cuda_device_name": "mock-verified-device",
                             "torch_version": fake_torch_version},
        "hub_source_tree_sha256": expected_hub,
        "checkpoint_sha256": expected_checkpoint,
    }
    model = object()
    wrapper = SimpleNamespace(model=model)
    events = []
    hash_targets = []

    class FakeImpl:
        def set_seed(self, seed):
            events.append(("seed", seed))

        def DINOv2MultiLayerBackbone(self, *, device):
            events.append(("load", device))
            if load_behavior == "failure":
                raise RuntimeError("synthetic model construction failure")
            return wrapper

    @dataclass
    class FakeNativeConfig:
        identity: str = "synthetic"

    impl = FakeImpl()
    restored = []
    monkeypatch.setattr(p0, "require_sources", lambda: impl)
    monkeypatch.setattr(r3, "_get_verified_model_receipt", lambda: expected_receipt)
    monkeypatch.setattr(r3, "_runtime_and_native_setup", lambda *_args: (
        lambda: restored.append(True), model_source))
    monkeypatch.setattr(r3, "_native_config", lambda *_args: (
        FakeNativeConfig(), FakeNativeConfig()))
    monkeypatch.setattr(e123, "sha256_file", lambda _path: "a" * 64)
    monkeypatch.setattr(r1, "_model_state_sha256", lambda value: (
        hash_targets.append(value) or
        (expected_state if digest_behavior == "match" else "0" * 64)))

    monkeypatch.setattr(torch.backends.cuda, "enable_flash_sdp", lambda _enabled: None)
    monkeypatch.setattr(torch.backends.cuda, "enable_mem_efficient_sdp", lambda _enabled: None)
    monkeypatch.setattr(torch.backends.cuda, "enable_math_sdp", lambda _enabled: None)
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: False)
    monkeypatch.setattr(torch.backends.cuda, "mem_efficient_sdp_enabled", lambda: False)
    monkeypatch.setattr(torch.backends.cuda, "math_sdp_enabled", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "mock-verified-device")

    if load_behavior == "failure":
        with pytest.raises(RuntimeError, match="synthetic model construction failure"):
            e123._native_profile()
        assert hash_targets == []
    elif digest_behavior == "mismatch":
        with pytest.raises(e123.E123Error, match="fingerprint differs") as error:
            e123._native_profile()
        assert hash_targets == [model]
        assert f"expected={expected_state}" in str(error.value)
        assert f"actual={'0' * 64}" in str(error.value)
    else:
        _fake_torch, _fake_impl, loaded_wrapper, native = e123._native_profile()
        assert loaded_wrapper is wrapper
        assert native["runtime"]["loaded_model_state_sha256"] == expected_state
        assert native["runtime"]["hub_source_tree_sha256"] == expected_hub
        assert native["runtime"]["checkpoint_sha256"] == expected_checkpoint
        assert hash_targets == [model]
    assert all(target is model for target in hash_targets)
    assert expected_receipt["loaded_model_state_sha256"] == expected_state
    assert restored == [True]


def _synthetic_native_candidate_inputs():
    import numpy as np

    # Deliberately non-sorted native order to catch accidental re-ranking.
    ids = np.asarray([(index * 73) % 784 for index in range(32)], dtype=np.int64)
    dense_map = np.linspace(-1.0, 1.0, 784, dtype=np.float32)
    summary = {"layer0": {"topk_patch_idx": ids.tolist()}}
    features = [torch.arange(784 * 4, dtype=torch.float32).reshape(784, 4)]
    return dense_map, ids, summary, features


def test_native_candidate_helper_uses_ordered_top8_without_dense_map_conversion(
        monkeypatch):
    import numpy as np

    import experiments.p0_protocol_audit as p0

    dense_map, ids, summary, features = _synthetic_native_candidate_inputs()
    dense_before = dense_map.copy()
    ids_before = ids.copy()
    summary_before = {"layer0": {"topk_patch_idx": list(summary["layer0"]["topk_patch_idx"])}}
    features_before = features[0].clone()
    impl = p0.require_sources()
    original = impl.layer_to_topk_from_amap
    converter_inputs = []

    def checked_converter(value):
        assert isinstance(value, dict), "dense NumPy maps must not reach the summary converter"
        converter_inputs.append(value)
        return original(value)

    monkeypatch.setattr(impl, "layer_to_topk_from_amap", checked_converter)
    selected = e123._native_shared_top8_candidates(dense_map, ids, summary, features)
    assert selected == {0: [int(value) for value in ids[:8]]}
    assert converter_inputs == []

    # The historical conversion agrees when it receives its required A_map_stats mapping.
    structured = {"per_layer_summary": {"layer0": {
        "topk_patch_idx": ids.tolist()}}}
    reference = impl.layer_to_topk_from_amap(structured)
    assert selected[0] == reference[0][:8]
    assert converter_inputs == [structured]
    assert np.array_equal(dense_map, dense_before)
    assert np.array_equal(ids, ids_before)
    assert summary == summary_before
    assert torch.equal(features[0], features_before)


@pytest.mark.parametrize("case", [
    "duplicate", "missing", "inconsistent", "negative", "out_of_range",
    "missing_summary", "wrong_feature_count",
])
def test_native_candidate_helper_rejects_invalid_top32_contract(case):
    import numpy as np

    dense_map, ids, summary, features = _synthetic_native_candidate_inputs()
    if case == "duplicate":
        ids[1] = ids[0]
        summary["layer0"]["topk_patch_idx"] = ids.tolist()
    elif case == "missing":
        ids = ids[:31]
        summary["layer0"]["topk_patch_idx"] = ids.tolist()
    elif case == "inconsistent":
        summary["layer0"]["topk_patch_idx"] = ids[::-1].tolist()
    elif case == "negative":
        ids[0] = -1
        summary["layer0"]["topk_patch_idx"] = ids.tolist()
    elif case == "out_of_range":
        ids[-1] = 784
        summary["layer0"]["topk_patch_idx"] = ids.tolist()
    elif case == "missing_summary":
        summary = {}
    elif case == "wrong_feature_count":
        features = [torch.zeros((783, 4), dtype=torch.float32)]
    assert isinstance(dense_map, np.ndarray)
    with pytest.raises(e123.E123Error):
        e123._native_shared_top8_candidates(dense_map, ids, summary, features)


def test_native_top8_candidates_are_shared_across_mutable_arms_only():
    import numpy as np

    dense_map, ids, summary, features = _synthetic_native_candidate_inputs()
    selected = e123._native_shared_top8_candidates(dense_map, ids, summary, features)
    static_feature_before = features[0].clone()
    common = e123.prepare_common_candidates(
        FakeImpl, features, selected, device="cpu")
    seen = []

    class WriterSpy:
        def observe(self, **kwargs):
            seen.append(kwargs["common_candidates"])
            return False

    arm_ids = e123.CONFIGURATION_IDS[1:]
    states = {arm_id: WriterSpy() for arm_id in arm_ids}
    e123.dispatch_common_write(
        states, eligible=True,
        observe_kwargs={"common_candidates": common},
    )
    assert tuple(states) == arm_ids
    assert "STATIC" not in states
    assert len(seen) == len(arm_ids)
    assert all(candidate is common for candidate in seen)
    assert common[0][0] == [int(value) for value in ids[:8]]
    expected = torch.nn.functional.normalize(
        static_feature_before[ids[:8]].float(), dim=1)
    assert torch.equal(common[0][1], expected)
    assert torch.equal(features[0], static_feature_before)
    assert np.array_equal(ids, np.asarray(summary["layer0"]["topk_patch_idx"]))


def test_condition_failure_receipt_preserves_traceback_and_event_context():
    def synthetic_failure():
        raise RuntimeError("synthetic traceback sentinel")

    with pytest.raises(RuntimeError) as caught:
        synthetic_failure()
    receipt = e123._condition_failure_receipt(
        caught.value, run_id="synthetic_run", condition_id="audiojack_s0_json",
        source_commit_sha="a" * 40, last_completed_event_position=12,
        failing_event_position=13)
    assert receipt["error_type"] == "RuntimeError"
    assert receipt["error"] == "synthetic traceback sentinel"
    assert "synthetic_failure" in receipt["traceback"]
    assert "synthetic traceback sentinel" in receipt["traceback"]
    assert receipt["source_file"].endswith("test_e123_recovery.py")
    assert receipt["source_line"] > 0
    assert "raise RuntimeError" in receipt["source_text"]
    assert receipt["last_completed_event_position"] == 12
    assert receipt["failing_event_position"] == 13


def test_supervisor_result_success_requires_all_tree_completion_fields():
    result = e123.SupervisorResult(
        status="COMPLETED", child_pid=4321, child_exit_code=0,
        elapsed_seconds=1.25, timeout_seconds=7200.0,
        termination_confirmed=True, active_job_processes=0)
    verified = e123._verify_supervisor_result(result)
    assert verified.status == "COMPLETED"
    assert verified.child_exit_code == 0
    assert verified.termination_confirmed is True
    assert verified.active_job_processes == 0
    assert e123._supervisor_result_payload(result) == {
        "status": "COMPLETED", "child_pid": 4321, "child_exit_code": 0,
        "elapsed_seconds": 1.25, "timeout_seconds": 7200.0,
        "termination_confirmed": True, "active_job_processes": 0,
    }


@pytest.mark.parametrize("result, message", [
    (e123.SupervisorResult("FAILED_NONZERO_EXIT", 4321, 9, 1.0, 7200.0,
                           True, 0), "did not complete successfully"),
    (e123.SupervisorResult("TIMED_OUT_TERMINATED", 4321, 1, 7200.0, 7200.0,
                           True, 0), "did not complete successfully"),
    (e123.SupervisorResult("FAILED_DESCENDANTS_TERMINATED", 4321, 0, 2.0,
                           7200.0, True, 0), "did not complete successfully"),
])
def test_supervisor_result_failures_are_rejected_after_coherence_check(result, message):
    with pytest.raises(e123.E123Error, match=message):
        e123._verify_supervisor_result(result)


def test_supervisor_result_missing_fields_and_contradictions_fail_closed():
    payload = e123._supervisor_result_payload(e123.SupervisorResult(
        "COMPLETED", 4321, 0, 1.0, 7200.0, True, 0))
    del payload["child_exit_code"]
    with pytest.raises(e123.E123Error, match="missing=.*child_exit_code"):
        e123._verify_supervisor_result(payload)

    contradictory = e123.SupervisorResult(
        "COMPLETED", 4321, 7, 1.0, 7200.0, True, 0)
    with pytest.raises(e123.E123Error, match="Contradictory SupervisorResult"):
        e123._verify_supervisor_result(contradictory)

    malformed = e123.SupervisorResult(
        "COMPLETED", 4321, 0, float("nan"), 7200.0, True, 0)
    with pytest.raises(e123.E123Error, match="Malformed SupervisorResult"):
        e123._verify_supervisor_result(malformed)


def test_module_execution_defines_final_report_helpers_before_entrypoint():
    source = (e123.ROOT / "experiments/calibration/e123_recovery.py").read_text(
        encoding="utf-8")
    assert source.rfind("def _scientific_claims(") < source.rfind(
        'if __name__ == "__main__":')
    assert source.rfind("def _render_unresolved_claims(") < source.rfind(
        'if __name__ == "__main__":')
    assert source.rfind("def _render_combined_report(") < source.rfind(
        'if __name__ == "__main__":')
    probe = """
import argparse, inspect, runpy, sys
original = argparse.ArgumentParser.parse_args
def checking_parse_args(self, *args, **kwargs):
    module_globals = inspect.currentframe().f_back.f_globals
    required = ("_scientific_claims", "_render_unresolved_claims",
                "_render_combined_report")
    assert all(callable(module_globals.get(name)) for name in required)
    return original(self, *args, **kwargs)
argparse.ArgumentParser.parse_args = checking_parse_args
sys.argv = ["e123_recovery", "--help"]
try:
    runpy.run_module("experiments.calibration.e123_recovery", run_name="__main__")
except SystemExit as exc:
    assert exc.code == 0
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe], cwd=e123.ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    assert completed.returncode == 0, completed.stdout
    assert "--stage-a-child" in completed.stdout


def test_synthetic_campaign_exercises_parent_global_seal_and_real_stage_b(
        tmp_path, monkeypatch):
    from experiments.calibration import r1_runtime_supervisor as supervisor
    from experiments.calibration import r3_fullstream_2x2 as r3

    output_base = tmp_path / "outputs" / "calibration"
    output_base.mkdir(parents=True)
    json_root = tmp_path / "synthetic-json"
    json_root.mkdir()
    source_digests = {}
    for category in e123.CATEGORIES:
        category_outcomes = [int(position % 2 == 1) for position in range(17)]
        document = {
            "meta": {"normal_class": "good"}, "train": [],
            "test": [{"image_path": f"{category}-{position}.png",
                      "anomaly_class": "scratch" if label else "good"}
                     for position, label in enumerate(category_outcomes)],
        }
        path = json_root / f"{category}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        source_digests[category] = e123.sha256_file(path)

    conditions = []
    for category in e123.CATEGORIES:
        for seed in e123.SUPPORT_SEEDS:
            for order in e123.STREAM_ORDERS:
                condition_id = f"{category}_s{seed}_{order}"
                events = []
                for position in range(17):
                    original_position = (position if order == "json"
                                         else 16 - position)
                    sample_key = f"{category}-sample-{original_position}"
                    group = e123.sha256_bytes(
                        ("official-sample-group\0" + sample_key).encode("utf-8"))
                    events.append(e123.E123Event(
                        position=position, original_position=original_position,
                        relative_id=f"{category}/{original_position}.png",
                        image_identity_sha256=e123.sha256_bytes(
                            f"image:{category}:{original_position}".encode()),
                        source_group_sha256=group, camera_id=1,
                        timestamp="synthetic", image_path="synthetic-no-image-read",
                        sample_key=sample_key))
                conditions.append(e123.Condition(
                    category=category, support_seed=seed, order=order,
                    events=tuple(events), query_positions=frozenset({0}),
                    query_tape_sha256=e123.sha256_bytes(
                        f"tape:{condition_id}".encode()),
                    stream_identity_sha256=e123.sha256_bytes(
                        f"stream:{condition_id}".encode()),
                    support_images=tuple(f"support-{i}" for i in range(4)),
                    support_groups=tuple(f"support-group-{i}" for i in range(4)),
                    c64_images=tuple(f"c64-{i}" for i in range(64)),
                    c64_groups=tuple(f"c64-group-{i}" for i in range(64)),
                    class_json_sha256=source_digests[category]))

    synthetic_source = {
        "branch": e123.EXPECTED_BRANCH, "commit_sha": "a" * 40,
        "requested_parent_sha": e123.REQUESTED_PARENT,
        "actual_parent_sha": "b" * 40,
        "requested_parent_is_ancestor": True,
        "source_sha256": {"synthetic": "c" * 64},
        "source_fingerprint_sha256": e123.sha256_json({"synthetic": "c" * 64}),
    }
    fixed_offsets = {
        "schema": "recovery.e123.fixed_offset_selection.v1",
        "selection_by_target_category": {
            category: {"coefficient": -1.0,
                       "selection_status": "SYNTHETIC_ONLY"}
            for category in e123.CATEGORIES
        },
    }
    monkeypatch.setattr(e123, "OUTPUT_BASE", output_base)
    monkeypatch.setattr(e123, "EXPECTED_COUNTS",
                        {category: 17 for category in e123.CATEGORIES})
    monkeypatch.setattr(e123, "source_identity", lambda: dict(synthetic_source))
    monkeypatch.setattr(e123, "_build_conditions", lambda: (conditions, {
        "synthetic_identity_only": True}))
    monkeypatch.setattr(e123, "select_fixed_offsets", lambda: fixed_offsets)
    monkeypatch.setattr(e123, "_historical_evidence_receipts", lambda: [])
    monkeypatch.setattr(e123, "_preflight_disk", lambda _path: {
        "free_bytes": e123.MIN_FREE_BYTES + 1, "required_free_bytes": e123.MIN_FREE_BYTES,
        "total_bytes": e123.MIN_FREE_BYTES + 1,
    })
    monkeypatch.setattr(e123, "_run_focused_cpu_tests", lambda: {
        "status": "PASS", "passed_test_count": 1,
        "dino_loaded": False, "cuda_executed": False,
        "real_dataset_read": False,
    })
    monkeypatch.setattr(r3, "JSON_ROOT", json_root)
    monkeypatch.setattr(r3, "_get_verified_model_receipt", lambda: {
        "status": "SYNTHETIC_MODEL_RECEIPT", "dino_loaded": False,
    })
    monkeypatch.setattr(e123, "_stage1a_k256_evidence", lambda: {
        "classification": "SYNTHETIC_ONLY", "trajectory_count": 0,
        "event_count": 0, "native_or_gpu_rerun": False,
        "per_method_macro": {}, "per_category_macro": {},
        "source_sha256": "d" * 64, "source_file": "synthetic",
        "labels_already_exposed_historically": True,
    })

    probe_contract, probe_conditions = e123.build_contract("synthetic_contract")
    e123._verify_contract(probe_contract, expected_run_id="synthetic_contract")
    assert len(probe_conditions) == 16
    assert all(len(condition.events) == 17 for condition in probe_conditions)

    stage_a_decodes = []
    stage_b_label_reads = []
    original_label_call = e123.RandomAccessQueriedLabelReader.__call__
    original_stage_b_read = e123.RandomAccessQueriedLabelReader.stage_b_all_labels

    def record_stage_a_decode(reader, category, position, source_key):
        stage_a_decodes.append((category, position))
        return original_label_call(reader, category, position, source_key)

    global_seal_barrier = {"passed": False, "stage_b_calls": 0}

    def record_stage_b_read(reader, *, global_seal):
        assert global_seal_barrier["passed"] is True
        assert global_seal["status"] == "SEALED_COMPLETE"
        stage_b_label_reads.append(reader.category)
        return original_stage_b_read(reader, global_seal=global_seal)

    monkeypatch.setattr(e123.RandomAccessQueriedLabelReader, "__call__",
                        record_stage_a_decode)
    monkeypatch.setattr(e123.RandomAccessQueriedLabelReader, "stage_b_all_labels",
                        record_stage_b_read)

    original_verify_condition = e123._verify_condition_seal
    validated_condition_ids = []

    def record_condition_validation(*args, **kwargs):
        seal = original_verify_condition(*args, **kwargs)
        validated_condition_ids.append(seal["condition_id"])
        return seal

    monkeypatch.setattr(e123, "_verify_condition_seal", record_condition_validation)

    report_calls = []
    original_claims = e123._scientific_claims
    original_unresolved = e123._render_unresolved_claims
    original_combined = e123._render_combined_report

    def record_claims(*args, **kwargs):
        report_calls.append("claims")
        return original_claims(*args, **kwargs)

    def record_unresolved(*args, **kwargs):
        report_calls.append("unresolved")
        return original_unresolved(*args, **kwargs)

    def record_combined(*args, **kwargs):
        report_calls.append("combined")
        return original_combined(*args, **kwargs)

    monkeypatch.setattr(e123, "_scientific_claims", record_claims)
    monkeypatch.setattr(e123, "_render_unresolved_claims", record_unresolved)
    monkeypatch.setattr(e123, "_render_combined_report", record_combined)
    original_run_stage_b = e123._run_stage_b

    def require_global_before_stage_b(run_path, run_id, contract, run_conditions,
                                      global_seal):
        seal_path = run_path / "stage_a" / "global_seal.json"
        assert seal_path.is_file()
        e123.require_complete_stage_a(e123.read_json(seal_path))
        assert global_seal_barrier["passed"] is False
        global_seal_barrier["passed"] = True
        global_seal_barrier["stage_b_calls"] += 1
        return original_run_stage_b(run_path, run_id, contract, run_conditions,
                                    global_seal)

    monkeypatch.setattr(e123, "_run_stage_b", require_global_before_stage_b)

    def write_synthetic_condition(run_id, condition, env):
        run_path = output_base / f"recovery_e123_{run_id}"
        contract = e123.read_json(run_path / "E123_contract.json")
        e123._verify_contract(contract, expected_run_id=run_id)
        launch_token = env["E123_SUPERVISOR_LAUNCH_TOKEN"]
        assert e123.sha256_bytes(launch_token.encode("ascii")) == contract[
            "supervisor_launch_token_sha256"]
        budget = e123.OutputBudget(run_path)
        e123._condition_manifest(budget, condition, contract, source={
            "synthetic_only": True, "official_group_disjointness": True,
        })
        condition_dir = f"stage_a/conditions/{condition.condition_id}"
        scores_rel = f"{condition_dir}/scores.jsonl"
        snapshots_rel = f"{condition_dir}/c64_snapshots.jsonl"
        feedback_rel = f"{condition_dir}/query_feedback.private.jsonl"
        progress_rel = f"{condition_dir}/progress.jsonl"
        scores_out = e123.BufferedJsonl(budget, scores_rel, flush_every=64)
        snapshots_out = e123.BufferedJsonl(budget, snapshots_rel, flush_every=1)
        feedback_out = e123.BufferedJsonl(budget, feedback_rel, flush_every=1)
        progress_out = e123.BufferedJsonl(budget, progress_rel, flush_every=1)
        memory_digest = e123.sha256_bytes(
            f"memory:{condition.condition_id}".encode())
        c64_scores = [0.40 + index * 0.001 for index in range(64)]
        snapshots_out.add({
            "schema": "recovery.e123.c64_snapshot.v1", "memory_version": 0,
            "memory_state_sha256": memory_digest, "c64_scores": c64_scores,
            "theta_before": 0.5, "theta_after": 0.5,
            "snapshot_source": "C64 scored against fixed support bank at M0",
        })
        snapshots_out.flush(sync=True)
        query_original_positions = [condition.events[position].original_position
                                    for position in condition.query_positions]
        query_groups = {
            event.original_position: event.source_group_sha256
            for event in condition.events if event.position in condition.query_positions
        }
        reader = e123.RandomAccessQueriedLabelReader(
            category=condition.category,
            path=json_root / f"{condition.category}.json",
            expected_source_sha256=condition.class_json_sha256,
            allowed_positions=query_original_positions,
            group_sha256_by_position=query_groups)
        query_label = None
        try:
            for position, event in enumerate(condition.events):
                queried = position in condition.query_positions
                event_id = e123.sha256_json({
                    "condition_id": condition.condition_id, "position": position,
                    "image_identity_sha256": event.image_identity_sha256,
                    "source_group_sha256": event.source_group_sha256,
                })
                scores = {
                    arm: 0.20 + position * 0.001 + arm_index * 0.0001
                    for arm_index, arm in enumerate(e123.CONFIGURATION_IDS)
                }
                scores_out.add({
                    "schema": e123.SCORE_SCHEMA,
                    "condition_id": condition.condition_id,
                    "position": position,
                    "original_position": event.original_position,
                    "event_id_sha256": event_id,
                    "image_identity_sha256": event.image_identity_sha256,
                    "source_group_sha256": event.source_group_sha256,
                    "camera_id": event.camera_id,
                    "query_tape_decision": queried,
                    "query_tape_sha256": condition.query_tape_sha256,
                    "scores": scores, "theta0": 0.5, "sigma0": 0.1,
                    "b3_greedy1024_threshold_before": 0.5,
                    "greedy1024_memory_version_before": 0,
                    "greedy1024_transition_sha256_before": memory_digest,
                })
                if queried:
                    query_label = e123.disclose_after_score_seal(
                        scores_out, queried=True, label_reader=reader,
                        category=condition.category,
                        original_position=event.original_position,
                        source_key=event.sample_key)
                    feedback_out.add({
                        "schema": "recovery.e123.private_query_feedback.v1",
                        "condition_id": condition.condition_id,
                        "position": position, "original_position": event.original_position,
                        "event_id_sha256": event_id,
                        "source_group_sha256": event.source_group_sha256,
                        "label": query_label,
                    })
                    feedback_out.flush(sync=True)
                progress_out.add({"position": position, "sealed": True})
            scores_out.flush(sync=True)
            snapshots_out.flush(sync=True)
            feedback_out.flush(sync=True)
            progress_out.flush(sync=True)
        finally:
            reader.close()
        mutable = {}
        for arm in e123.CONFIGURATION_IDS[1:]:
            capacity = int(arm.rsplit("K", 1)[1])
            mutable[arm] = {
                "prototype_tensor_bytes": capacity * 2 * 4,
                "metadata_runtime_bytes_estimate": 16,
                "metadata_tensor_bytes": 8,
                "allocated_persistent_tensor_bytes": capacity * 2 * 4 + 8,
                "python_metadata_bytes_estimate": 16,
                "persistent_mutable_bytes_estimate": capacity * 2 * 4 + 24,
                "layers": [{"occupied_slots": 1, "capacity": capacity}],
                "dtype": "float32", "accepted_patches": 1,
                "rejected_patches": 0, "replaced_patches": 0,
                "candidate_patches": 1,
            }
        resource_summary = {"mutable_arms": mutable}
        feedback_rows = e123._read_jsonl(run_path / f"{feedback_rel}")
        seal = {
            "schema": e123.CONDITION_SEAL_SCHEMA,
            "status": "SEALED_COMPLETE", "run_id": run_id,
            "condition_id": condition.condition_id,
            "category": condition.category,
            "support_seed": condition.support_seed,
            "stream_order": condition.order,
            "event_count": len(condition.events),
            "query_count": len(condition.query_positions),
            "query_normal_count": int(query_label == 0),
            "query_defect_count": int(query_label == 1),
            "score_configurations": list(e123.CONFIGURATION_IDS),
            "score_rows_sha256": e123.sha256_file(run_path / scores_rel),
            "score_row_count": len(condition.events),
            "c64_snapshots_sha256": e123.sha256_file(run_path / snapshots_rel),
            "private_query_feedback_sha256": e123.sha256_file(
                run_path / feedback_rel),
            "feedback_count": len(feedback_rows),
            "query_tape_sha256": condition.query_tape_sha256,
            "resource_summary": resource_summary,
            "support_self_scores": [{"score": 0.55 + index * 0.01}
                                    for index in range(4)],
            "static_native_reference_parity": {"status": "SYNTHETIC_PASS"},
            "fixed_bank": {"historical_match": True},
            "c64": {"max_abs_error_to_historical_receipt": 0.0},
        }
        seal["seal_sha256"] = e123.sha256_json(seal)
        budget.write_once(
            f"{condition_dir}/condition_seal.json", e123.canonical_bytes(seal))
        budget.verify_complete_namespace()

    def fake_supervisor(command, *, timeout_seconds, env, cwd, output_dir=None):
        assert timeout_seconds == e123.SUPERVISOR_TIMEOUT_SECONDS
        run_id = command[command.index("--run-id") + 1]
        condition_id = command[command.index("--condition-id") + 1]
        assert cwd == e123.ROOT
        if run_id == "synthetic_fail_01":
            return e123.SupervisorResult(
                "FAILED_NONZERO_EXIT", 4321, 7, 1.0, timeout_seconds, True, 0)
        condition = next(row for row in conditions if row.condition_id == condition_id)
        write_synthetic_condition(run_id, condition, env)
        return e123.SupervisorResult(
            "COMPLETED", 4321, 0, 1.0, timeout_seconds, True, 0)

    monkeypatch.setattr(supervisor, "run_supervised_child", fake_supervisor)

    assert e123._run_campaign("synthetic_success_01") == 0
    success_root = output_base / "recovery_e123_synthetic_success_01"
    assert global_seal_barrier == {"passed": True, "stage_b_calls": 1}
    assert len(stage_a_decodes) == 16
    for category in e123.CATEGORIES:
        assert sorted(position for decoded_category, position in stage_a_decodes
                      if decoded_category == category) == [0, 0, 16, 16]
    assert len(stage_b_label_reads) == 16
    assert len(validated_condition_ids) >= 16
    assert set(report_calls) == {"claims", "unresolved", "combined"}
    assert len(report_calls) == 3
    assert len(conditions) == 16
    for name in (
            "stage_a/global_seal.json", "stage_b/aggregate.json",
            "E1_failure_decomposition.json", "E2_memory_comparison.json",
            "E3_calibration_comparison.json", "E123_protocol_parity.json",
            "E123_unresolved_claims.md", "E123_combined_report.md",
            "E123_provenance.json"):
        assert (success_root / name).is_file(), name
    global_seal = e123.read_json(success_root / "stage_a/global_seal.json")
    e123.require_complete_stage_a(global_seal)
    aggregate = e123.read_json(success_root / "stage_b/aggregate.json")
    assert aggregate["status"] == "COMPLETE_AFTER_GLOBAL_STAGE_A_SEAL"
    assert aggregate["image_level_outcome_label_vectors_written"] is False
    assert aggregate["e1"]["by_condition"]
    assert aggregate["e2"]["resource_aggregate"]
    assert aggregate["e3"]["category_equal_metrics"]
    for name in ("stage_b/aggregate.json", "E1_failure_decomposition.json",
                 "E2_memory_comparison.json", "E3_calibration_comparison.json",
                 "E123_protocol_parity.json", "E123_provenance.json"):
        artifact = e123.read_json(success_root / name)
        claimed = artifact.pop("artifact_sha256")
        assert claimed == e123.sha256_json(artifact), name
    assert e123.OutputBudget(success_root).verify_complete_namespace()[
        "unaccounted_files"] == 0

    assert e123._run_campaign("synthetic_fail_01") == 1
    failure_root = output_base / "recovery_e123_synthetic_fail_01"
    failed_receipt = e123.read_json(
        failure_root / "stage_a/conditions/audiojack_s0_json/supervisor.json")
    assert failed_receipt["result"] == {
        "status": "FAILED_NONZERO_EXIT", "child_pid": 4321,
        "child_exit_code": 7, "elapsed_seconds": 1.0,
        "timeout_seconds": float(e123.SUPERVISOR_TIMEOUT_SECONDS),
        "termination_confirmed": True, "active_job_processes": 0,
    }
    assert (failure_root / "stage_a/failure.json").is_file()
    assert not (failure_root / "stage_a/global_seal.json").exists()
    assert global_seal_barrier["stage_b_calls"] == 1
