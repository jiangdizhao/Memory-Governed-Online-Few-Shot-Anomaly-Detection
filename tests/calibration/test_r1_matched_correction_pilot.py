"""CPU-only synthetic gates for the isolated R1 correction pilot."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from experiments.calibration.r1_matched_correction_pilot import (
    CausalEventOrder,
    EvidenceTapeWriter,
    R1CausalityError,
    TapeIntegrityError,
    arrays_sha256,
    corrected_dense_map,
    deterministic_npz_bytes,
    max_score,
    sha256_bytes,
    signed_knn_delta,
    source_disjoint_anchor_indices,
)


def _arrays() -> dict[str, np.ndarray]:
    base = np.linspace(0.0, 0.783, 784, dtype=np.float32)
    ids = np.arange(752, 784, dtype=np.int64)
    features = np.zeros((32, 8), dtype=np.float32)
    features[:, 0] = 1.0
    c1 = np.zeros(32, dtype=np.float32)
    c1[31] = -0.5
    c2 = np.zeros(32, dtype=np.float32)
    return {
        "base_patch_map": base,
        "top32_ids": ids,
        "top32_features": features,
        "c1_delta": c1,
        "c2_delta": c2,
    }


def _metadata(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    base = arrays["base_patch_map"]
    ids = arrays["top32_ids"]
    c1 = arrays["c1_delta"]
    c2 = arrays["c2_delta"]
    return {
        "event_id": "synthetic-event-0",
        "category": "synthetic",
        "official_group_sha256": "synthetic-group-a",
        "c0_score": max_score(base),
        "c1_score": max_score(corrected_dense_map(base, ids, c1)),
        "c2_score": max_score(corrected_dense_map(base, ids, c2)),
        "query_decision": False,
    }


def test_dense_reconstruction_preserves_native_topk_and_exact_max():
    arrays = _arrays()
    before = arrays_sha256({"base_patch_map": arrays["base_patch_map"]})
    restored = corrected_dense_map(arrays["base_patch_map"],
                                   arrays["top32_ids"], arrays["c1_delta"])
    assert np.array_equal(arrays["top32_ids"], np.arange(752, 784, dtype=np.int64))
    assert restored[783] == pytest.approx(float(arrays["base_patch_map"][783]) - 0.5)
    assert max_score(restored) == float(restored[782])
    zero = corrected_dense_map(arrays["base_patch_map"], arrays["top32_ids"],
                               np.zeros(32, dtype=np.float32))
    assert np.array_equal(zero, arrays["base_patch_map"])
    assert max_score(zero) == max_score(arrays["base_patch_map"])
    assert arrays_sha256({"base_patch_map": arrays["base_patch_map"]}) == before


def test_serialized_dense_map_and_selected_ids_replay_exactly(tmp_path):
    arrays = _arrays()
    writer = EvidenceTapeWriter(tmp_path / "tape")
    writer.seal_event(0, arrays, _metadata(arrays))
    writer.finalize_event(0, {"writer_action": {"action": "no_write"},
                              "causal_order_complete": True})
    result = writer.verify()
    assert result["passed"]
    with np.load(tmp_path / "tape" / "events" / "event_000000.npz",
                 allow_pickle=False) as restored:
        assert np.array_equal(restored["top32_ids"], arrays["top32_ids"])
        assert np.array_equal(restored["base_patch_map"], arrays["base_patch_map"])
        replay = corrected_dense_map(restored["base_patch_map"], restored["top32_ids"],
                                     restored["c1_delta"])
        assert max_score(replay) == _metadata(arrays)["c1_score"]


def test_checksums_and_npz_bytes_are_deterministic():
    arrays = _arrays()
    first = deterministic_npz_bytes(arrays)
    second = deterministic_npz_bytes({key: value.copy()
                                      for key, value in reversed(list(arrays.items()))})
    assert first == second
    assert sha256_bytes(first) == sha256_bytes(second)
    assert arrays_sha256(arrays) == arrays_sha256(dict(reversed(list(arrays.items()))))


def test_query_label_provider_is_unavailable_until_score_seal():
    calls: list[str] = []
    gate = CausalEventOrder(9, queried=True)
    with pytest.raises(R1CausalityError):
        gate.disclose(lambda: calls.append("read") or 0)
    assert calls == []
    gate.seal_scores()
    assert gate.disclose(lambda: calls.append("read") or 1) == 1
    gate.corrections_updated()
    gate.memory_updated()
    gate.finish()
    assert calls == ["read"]
    assert gate.phase == "complete"


def test_unqueried_event_never_requests_a_label():
    gate = CausalEventOrder(2, queried=False)
    gate.seal_scores()
    with pytest.raises(R1CausalityError):
        gate.disclose(lambda: pytest.fail("unqueried label provider was called"))
    gate.corrections_updated()
    gate.memory_updated()
    gate.finish()


def test_source_and_future_anchors_are_excluded():
    positions = [1, 2, 4]
    groups = ["same", "other", "future"]
    kept = source_disjoint_anchor_indices(positions, groups,
                                         event_position=4, event_group="same")
    assert kept == [1]
    anchors = [
        {"position": 1, "group_sha256": "same", "sign": 1,
         "feature": np.array([1.0, 0.0], dtype=np.float32)},
        {"position": 2, "group_sha256": "other", "sign": -1,
         "feature": np.array([1.0, 0.0], dtype=np.float32)},
        {"position": 5, "group_sha256": "future", "sign": 1,
         "feature": np.array([1.0, 0.0], dtype=np.float32)},
    ]
    delta, count = signed_knn_delta(
        np.array([[1.0, 0.0]], dtype=np.float32), anchors,
        event_position=4, event_group="same", neighbors=5, amplitude=0.20)
    assert count == 1
    assert delta.tolist() == pytest.approx([-0.20])


def test_tape_refuses_duplicate_gapped_resume_and_label_payloads(tmp_path):
    arrays = _arrays()
    writer = EvidenceTapeWriter(tmp_path / "tape")
    with pytest.raises(TapeIntegrityError):
        writer.seal_event(1, arrays, _metadata(arrays))
    with pytest.raises(TapeIntegrityError):
        writer.seal_event(0, arrays, {**_metadata(arrays), "label": 1})
    writer.seal_event(0, arrays, _metadata(arrays))
    with pytest.raises(TapeIntegrityError):
        writer.seal_event(0, arrays, _metadata(arrays))
    with pytest.raises(FileExistsError):
        EvidenceTapeWriter(tmp_path / "tape")
    writer.finalize_event(0, {"writer_action": {"action": "no_write"}})
    assert writer.verify()["passed"]


def test_score_replay_does_not_mutate_baseline_map_or_digest():
    arrays = _arrays()
    baseline = arrays["base_patch_map"].copy()
    digest_before = hashlib.sha256(baseline.tobytes()).hexdigest()
    _ = corrected_dense_map(baseline, arrays["top32_ids"], arrays["c1_delta"])
    _ = corrected_dense_map(baseline, arrays["top32_ids"], arrays["c2_delta"])
    assert np.array_equal(baseline, arrays["base_patch_map"])
    assert hashlib.sha256(baseline.tobytes()).hexdigest() == digest_before


def test_tape_readback_rejects_modified_shard(tmp_path):
    arrays = _arrays()
    writer = EvidenceTapeWriter(tmp_path / "tape")
    writer.seal_event(0, arrays, _metadata(arrays))
    writer.finalize_event(0, {"writer_action": {"action": "no_write"}})
    shard = tmp_path / "tape" / "events" / "event_000000.npz"
    shard.write_bytes(shard.read_bytes() + b"corruption")
    with pytest.raises(TapeIntegrityError):
        writer.verify()


def test_explicit_resume_accepts_only_the_empty_bootstrap_failure(tmp_path):
    root = tmp_path / "bootstrap-tape"
    (root / "events").mkdir(parents=True)
    (root / "run_manifest.json").write_text(json.dumps({
        "schema": "calibration.r1_evidence_tape.v1",
        "status": "INCOMPLETE", "sealed_events": 0,
        "finalized_events": 0, "next_position": 0,
        "failure_type": "ModuleNotFoundError",
        "failure_message": "No module named 'experiments'",
    }), encoding="utf-8")
    with pytest.raises(FileExistsError):
        EvidenceTapeWriter(root)
    resumed = EvidenceTapeWriter(root, resume_empty_bootstrap=True)
    assert resumed.next_sealed == resumed.next_finalized == 0
    history = json.loads((root / "attempt_history.jsonl").read_text(encoding="utf-8"))
    assert history["outcome"] == "EMPTY_BOOTSTRAP_FAILURE_RETAINED"
    arrays = _arrays()
    resumed.seal_event(0, arrays, _metadata(arrays))
    resumed.finalize_event(0, {"writer_action": {"action": "no_write"}})
    assert resumed.verify()["passed"]

    contaminated = tmp_path / "contaminated-tape"
    (contaminated / "events").mkdir(parents=True)
    (contaminated / "events" / "event_000000.npz").write_bytes(b"partial")
    (contaminated / "run_manifest.json").write_text(json.dumps({
        "schema": "calibration.r1_evidence_tape.v1",
        "status": "INCOMPLETE", "sealed_events": 0,
        "finalized_events": 0, "next_position": 0,
        "failure_type": "ModuleNotFoundError",
        "failure_message": "No module named 'experiments'",
    }), encoding="utf-8")
    with pytest.raises(TapeIntegrityError):
        EvidenceTapeWriter(contaminated, resume_empty_bootstrap=True)
