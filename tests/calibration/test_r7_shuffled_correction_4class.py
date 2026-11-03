from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.calibration import r7_shuffled_correction_4class as r7
from experiments.calibration.profiles import QueryProtocolConfig
from experiments.calibration.protocol import TapeRecord
from experiments.calibration.query_tape import build_qa_ts, verify_tape


def _group(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fake_rows():
    rows = []
    for group_index, group_name in enumerate(("group-a", "group-b", "group-c", "group-d")):
        for view in range(2):
            rows.append(SimpleNamespace(
                position=len(rows), opaque_group=_group(group_name),
                relative_id=f"OK/S{group_index + 1:04d}/audiojack_{group_index + 1:04d}_OK_C{view + 1}_20231021130235.jpg",
                camera_id=view + 1, timestamp="20231021130235",
                image_path=f"G:/dataset/realiad/images/audiojack/OK/S{group_index + 1:04d}/image.jpg"))
    return rows


def _parse_fixture_row(sample: int, *, defect: str | None = None) -> tuple[str, str, str]:
    timestamp = f"202310211302{sample:02d}"
    if defect is None:
        sample_key = f"audiojack_{sample:04d}_OK"
        relative = f"OK/S{sample:04d}/audiojack_{sample:04d}_OK_C1_{timestamp}.jpg"
        label = "OK"
    else:
        sample_key = f"audiojack__{sample:04d}_NG_{defect}"
        relative = (f"NG/{defect}/S{sample:04d}/"
                    f"audiojack__{sample:04d}_NG_{defect}_C1_{timestamp}.jpg")
        label = defect
    return relative, sample_key, label


def _label_fixture(tmp_path: Path):
    source_rows = [_parse_fixture_row(1), _parse_fixture_row(2, defect="SC"),
                   _parse_fixture_row(3), _parse_fixture_row(4, defect="DT")]
    records = [{"image_path": relative, "anomaly_class": label,
                "mask_path": f"private-mask-{index}.png"}
               for index, (relative, _sample_key, label) in enumerate(source_rows)]
    payload = json.dumps({"meta": {"normal_class": "OK", "category": "audiojack"},
                          "train": [], "test": records}, separators=(",", ":")).encode("utf-8")
    path = tmp_path / "audiojack.json"
    path.write_bytes(payload)
    # Query shuffled positions 0 and 1, which map to original JSON positions 3 and 0.
    mapping = (3, 0, 2, 1)
    events = []
    for position, original in enumerate(mapping):
        relative, sample_key, _label = source_rows[original]
        parsed_group = hashlib.sha256(("official-sample-group\0" + sample_key).encode()).hexdigest()
        events.append(r7.ShuffledEvent(
            position=position, original_position=original,
            relative_id=relative, image_identity_sha256=r7._identity_hash(relative),
            source_group_sha256=parsed_group, camera_id="1",
            timestamp=f"202310211302{original + 1:02d}", image_path=relative))
    return path, payload, events


def test_group_shuffle_is_deterministic_group_bounded_and_preserves_view_order():
    rows = _fake_rows()
    one = r7.deterministic_group_shuffle(rows, seed=0)
    two = r7.deterministic_group_shuffle(rows, seed=0)
    assert one == two
    assert [event.position for event in one] == list(range(len(rows)))
    assert sorted(event.original_position for event in one) == list(range(len(rows)))
    shuffled_group_order = []
    for event in one:
        if not shuffled_group_order or event.source_group_sha256 != shuffled_group_order[-1]:
            shuffled_group_order.append(event.source_group_sha256)
    assert len(shuffled_group_order) == 4
    assert shuffled_group_order != [rows[0].opaque_group, rows[2].opaque_group,
                                    rows[4].opaque_group, rows[6].opaque_group]
    for group in shuffled_group_order:
        members = [event.original_position for event in one
                   if event.source_group_sha256 == group]
        assert members == sorted(members)
    assert r7._identity_receipt(one)["mapping_sha256"] == r7._identity_receipt(two)["mapping_sha256"]


def test_regenerated_qats_tape_is_deterministic_and_queries_one_view_per_group():
    events = []
    for position in range(320):
        group_sha = _group(f"physical-group-{position}")
        events.append(r7.ShuffledEvent(
            position=position, original_position=position,
            relative_id=f"sample-{position:04d}.jpg",
            image_identity_sha256=_group(f"image-{position}"),
            source_group_sha256=group_sha, camera_id=str(position % 5 + 1),
            timestamp=f"2023102113{position % 60:02d}35", image_path="unused"))
    records = [TapeRecord(position=event.position,
                          official_sample_key=event.source_group_sha256,
                          camera_id=int(event.camera_id),
                          relative_id=event.image_identity_sha256,
                          timestamp=event.timestamp, phase=0)
               for event in events]
    first = build_qa_ts(records, config=QueryProtocolConfig())
    second = build_qa_ts(records, config=QueryProtocolConfig())
    verify_tape(first, records, config=QueryProtocolConfig())
    verify_tape(second, records, config=QueryProtocolConfig())
    assert first.digest == second.digest
    assert sorted(first.positions) == sorted(second.positions)
    assert len(first.positions) == 10
    assert len(first.queried_groups) == len(first.positions)
    assert r7._tape_receipt(first)["tape_sha256"] == r7._tape_receipt(second)["tape_sha256"]


def test_label_reader_maps_nonmonotone_shuffled_queries_and_discloses_only_tape_positions(tmp_path):
    path, payload, events = _label_fixture(tmp_path)
    reader = r7.IndexedQueriedLabelReader(
        path, "audiojack", hashlib.sha256(payload).hexdigest(), events, (0, 1))
    try:
        with pytest.raises(r7.R7Error, match="before the score row"):
            reader.label_for(0, sealed_and_fsynced=False)
        assert reader.label_for(0, sealed_and_fsynced=True) == 1
        assert reader.label_for(1, sealed_and_fsynced=True) == 0
        assert reader.decoded_positions == [0, 1]
        with pytest.raises(r7.R7Error, match="unqueried"):
            reader.label_for(2, sealed_and_fsynced=True)
        reader.finish()
    finally:
        reader.close()


def test_label_reader_rejects_out_of_order_duplicate_and_wrong_source_revision(tmp_path):
    path, payload, events = _label_fixture(tmp_path)
    with pytest.raises(r7.R7Error, match="source hash changed"):
        r7.IndexedQueriedLabelReader(path, "audiojack", "0" * 64, events, (0, 1))
    reader = r7.IndexedQueriedLabelReader(
        path, "audiojack", hashlib.sha256(payload).hexdigest(), events, (0, 1))
    try:
        with pytest.raises(r7.R7Error, match="out of tape order"):
            reader.label_for(1, sealed_and_fsynced=True)
        assert reader.label_for(0, sealed_and_fsynced=True) == 1
        with pytest.raises(r7.R7Error, match="out of tape order"):
            reader.label_for(0, sealed_and_fsynced=True)
    finally:
        reader.close()


def test_score_disclosure_gate_completes_durable_write_before_label_access(tmp_path):
    writes = []
    path = tmp_path / "sealed.jsonl"
    def write_and_fsync(row):
        payload = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
        with path.open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        writes.append(dict(row))
    gate = r7.ScoreDisclosureGate(write_and_fsync)
    row = {"position": 0, "scores": {"G": 0.2, "GC": 0.1},
           "decisions": {arm: {"prediction": 0} for arm in r7.ARMS},
           "query_tape_decision": True}
    gate.seal(0, row)
    assert writes == [row]
    assert path.read_text(encoding="utf-8").strip() == json.dumps(row, sort_keys=True)
    assert gate.is_sealed(0)
    with pytest.raises(r7.R7Error, match="in shuffled event order"):
        gate.seal(2, {"position": 2})
    with pytest.raises(r7.R7Error, match="invalid score row"):
        gate.seal(1, {"position": 1, "label": 1})


def test_private_history_writer_registers_and_accounts_first_jsonl_append(tmp_path):
    budget = r7._budget(tmp_path / "private", create=True)
    first = {"record_type": "queried_feedback_and_updates", "position": 3}
    second = {"record_type": "progress", "position": 4}
    r7._append_history(budget, first)
    r7._append_history(budget, second)
    rows = [json.loads(line) for line in
            (tmp_path / "private" / "attempt_history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == [first, second]
    assert set(budget.audit()["files"]) == {"attempt_history.jsonl", "budget.lock"}


def test_source_group_bootstrap_and_worst_512_window_are_cpu_only():
    n = 640
    labels = np.asarray([index % 2 for index in range(n)], dtype=np.int8)
    groups = np.asarray([f"{index // 2:064x}" for index in range(n)])
    scores = {"G": labels.astype(float) * 0.6 + np.linspace(0, 0.2, n),
              "GC": labels.astype(float) * 0.6 + np.linspace(0, 0.2, n) + 0.01}
    predictions = {
        "G-Frozen": (scores["G"] >= 0.5).astype(np.int8),
        "G-B1": (scores["G"] >= 0.52).astype(np.int8),
        "G-B3": (scores["G"] >= 0.48).astype(np.int8),
        "GC-Frozen": (scores["GC"] >= 0.5).astype(np.int8),
        "GC-B1": (scores["GC"] >= 0.52).astype(np.int8),
        "GC-B3": (scores["GC"] >= 0.48).astype(np.int8),
    }
    result = r7._paired_group_bootstrap(scores, predictions, labels, groups,
                                       seed=123, replicates=30)
    assert result["source_group_count"] == n // 2
    assert result["same_sampled_groups_for_all_scorers_and_controllers"] is True
    assert result["contrasts"]["auroc_GC_minus_G"]["valid_replicates"] > 0
    assert r7._worst_512_metrics(labels, predictions["G-Frozen"])["complete_windows"] == n - 512 + 1


def test_current_state_c64_rescore_uses_next_event_position():
    observed_positions = []
    def score_delta(_impl, _bank, _selected, _groups, _group, *, event_position):
        observed_positions.append(event_position)
        return np.zeros(32, dtype=np.float32), {"same_source_excluded_anchor_slots": 0}
    feature = np.zeros((784, 4096), dtype=np.float32)
    item = {"features": [feature], "source_group_sha256": _group("protected-c64")}
    base_map = np.zeros((28, 28), dtype=np.float32)
    base_cache = [(base_map, np.arange(32, dtype=np.int64), 0.0)]
    fake_r1 = SimpleNamespace(
        _c1_score_delta=score_delta,
        corrected_dense_map=lambda score_map, _ids, _delta: score_map,
        max_score=lambda _score_map: 0.25)
    impl = SimpleNamespace(l2_normalize=lambda values: values)
    bank = SimpleNamespace(device="cpu")
    scores, _elapsed = r7._score_c64_corrected(
        impl, bank, [item], base_cache, None, fake_r1, {}, "audiojack",
        event_position=36)
    assert observed_positions == [36]
    assert scores == [0.25]


def test_run_identity_rejects_wrong_branch_head_and_nonunique_output_namespace():
    assert r7.EXPECTED_BRANCH == "calibration-stage0-contract-20261029"
    assert len(r7.EXPECTED_PARENT) == 40
    assert r7.OUTPUT_LIMIT == 4 * 1024**3
    assert r7.FAILURE_RESERVE < r7.OUTPUT_LIMIT
    assert r7.MIN_FREE_BYTES == 10 * 1024**3
    assert r7.FULL_DEADLINE_SECONDS == 120 * 60
    assert r7.SMOKE_DEADLINE_SECONDS == 15 * 60
