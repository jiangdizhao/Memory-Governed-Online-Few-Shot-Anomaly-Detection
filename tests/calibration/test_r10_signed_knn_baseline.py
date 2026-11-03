from __future__ import annotations

import hashlib

import numpy as np
import pytest

from experiments.calibration import r1_matched_correction_pilot as r1
from experiments.calibration.r10_signed_knn_baseline import (
    AMPLITUDE,
    K_NEIGHBORS,
    SignedKNNFIFO,
    _corrected_c2_score,
)


def _group(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _make_memory(capacity: int, rows: np.ndarray, signs: list[int],
                 groups: list[str] | None = None,
                 positions: list[int] | None = None) -> SignedKNNFIFO:
    memory = SignedKNNFIFO(capacity)
    groups = groups or [_group("anchor-source")]
    positions = positions or [0]
    for i, feature in enumerate(rows):
        memory.insert(feature[None, :], sign=signs[i],
                      source_group_sha256=groups[i] if len(groups) > 1 else groups[0],
                      position=positions[i] if len(positions) > 1 else positions[0])
    return memory


def _historical_anchors(rows: np.ndarray, signs: list[int],
                        groups: list[str], positions: list[int]) -> list[dict]:
    anchors = []
    for i, feature in enumerate(rows):
        value = np.asarray(feature, dtype=np.float32)
        anchors.append({
            "position": int(positions[i]),
            "group_sha256": groups[i],
            "patch_id": i % 784,
            "sign": int(signs[i]),
            "feature": value,
            "feature_sha256": r1.arrays_sha256({"feature": value}),
        })
    return anchors


def test_empty_fifo_returns_zero_delta():
    memory = SignedKNNFIFO(256)
    query = np.ones((32, 4096), dtype=np.float32)
    delta, audit = memory.score(query, event_position=1,
                                event_group_sha256=_group("query"))
    assert np.array_equal(delta, np.zeros(32, dtype=np.float32))
    assert audit == {"eligible_anchor_count": 0, "source_excluded_anchor_count": 0}
    assert memory.count == 0


def test_float32_formula_matches_historical_signed_knn_for_fewer_than_five():
    rng = np.random.default_rng(814)
    anchors = rng.normal(size=(3, 4096)).astype(np.float32)
    query = rng.normal(size=(7, 4096)).astype(np.float32)
    groups = [_group(f"anchor-{i}") for i in range(3)]
    signs = [-1, 1, 1]
    positions = [1, 2, 3]
    memory = _make_memory(256, anchors, signs, groups, positions)
    actual, audit = memory.score(query, event_position=4,
                                 event_group_sha256=_group("query"))
    expected, eligible = r1.signed_knn_delta(
        query, _historical_anchors(anchors, signs, groups, positions),
        event_position=4, event_group=_group("query"),
        neighbors=K_NEIGHBORS, amplitude=AMPLITUDE)
    assert eligible == 3
    assert audit["eligible_anchor_count"] == 3
    np.testing.assert_array_equal(actual, expected)


def test_equal_distance_ties_use_fifo_anchor_order_and_mixed_signs():
    feature = np.zeros((4096,), dtype=np.float32)
    feature[0] = 3.0
    anchors = np.stack([feature.copy() for _ in range(6)])
    signs = [-1, 1, -1, 1, -1, 1]
    groups = [_group(f"tie-{i}") for i in range(6)]
    positions = [0] * 6
    memory = _make_memory(256, anchors, signs, groups, positions)
    query = feature[None, :]
    actual, _ = memory.score(query, event_position=1,
                             event_group_sha256=_group("query"))
    expected, _ = r1.signed_knn_delta(
        query, _historical_anchors(anchors, signs, groups, positions),
        event_position=1, event_group=_group("query"),
        neighbors=5, amplitude=0.02)
    np.testing.assert_array_equal(actual, expected)
    assert actual[0] == np.float32(-0.004)


def test_same_source_is_excluded_and_current_or_future_anchors_fail_closed():
    rows = np.zeros((2, 4096), dtype=np.float32)
    rows[:, 0] = 1.0
    group = _group("same-source")
    memory = _make_memory(256, rows, [-1, 1], [group, group], [1, 2])
    delta, audit = memory.score(rows[:1], event_position=3,
                                event_group_sha256=group)
    assert np.array_equal(delta, np.zeros(1, dtype=np.float32))
    assert audit == {"eligible_anchor_count": 0, "source_excluded_anchor_count": 2}
    with pytest.raises(ValueError, match="current or future"):
        memory.score(rows[:1], event_position=2,
                     event_group_sha256=_group("other-source"))


def test_fifo_evicts_oldest_global_entries_for_both_registered_capacities():
    rng = np.random.default_rng(17)
    rows = rng.normal(size=(300, 4096)).astype(np.float32)
    groups = [_group(f"fifo-{i}") for i in range(len(rows))]
    memory_256 = _make_memory(256, rows, [1] * len(rows), groups, [0] * len(rows))
    memory_1024 = _make_memory(1024, rows, [1] * len(rows), groups, [0] * len(rows))
    idx_256 = memory_256._active_indices()
    idx_1024 = memory_1024._active_indices()
    assert memory_256.count == 256
    assert memory_256.evictions == 44
    assert memory_1024.count == 300
    assert memory_1024.evictions == 0
    assert [bytes(value).hex() for value in memory_256.source_groups[idx_256]] == groups[-256:]
    assert [bytes(value).hex() for value in memory_1024.source_groups[idx_1024]] == groups
    assert memory_256.first_saturation_position == 0
    assert memory_1024.first_saturation_position is None


def test_fifo_saturation_occurs_at_the_registered_256_and_1024_row_limits():
    feature = np.zeros((32, 4096), dtype=np.float32)
    feature[:, 0] = 1.0
    m256 = SignedKNNFIFO(256)
    m1024 = SignedKNNFIFO(1024)
    for position in range(9):
        m256.insert(feature, sign=(-1 if position % 2 else 1),
                    source_group_sha256=_group(f"source-256-{position}"),
                    position=position)
    for position in range(33):
        m1024.insert(feature, sign=(-1 if position % 2 else 1),
                     source_group_sha256=_group(f"source-1024-{position}"),
                     position=position)
    assert m256.count == 256
    assert m256.evictions == 32
    assert m256.first_saturation_position == 7
    assert m1024.count == 1024
    assert m1024.evictions == 32
    assert m1024.first_saturation_position == 31


def test_capacity_states_are_independent_and_digest_is_deterministic():
    rng = np.random.default_rng(32)
    rows = rng.normal(size=(40, 4096)).astype(np.float32)
    m1 = _make_memory(256, rows, [-1 if i % 2 else 1 for i in range(40)])
    m2 = _make_memory(256, rows, [-1 if i % 2 else 1 for i in range(40)])
    assert m1.state_sha256() == m2.state_sha256()
    before = m1.state_sha256()
    rows[0, 11] += np.float32(0.25)
    mutated = _make_memory(256, rows, [-1 if i % 2 else 1 for i in range(40)])
    assert mutated.state_sha256() != before
    other_capacity = _make_memory(1024, rows, [-1 if i % 2 else 1 for i in range(40)])
    assert other_capacity.count == m1.count
    assert other_capacity.state_sha256() != mutated.state_sha256()


def test_native_top32_dense_map_adapter_and_image_max_are_used():
    rng = np.random.default_rng(99)
    base = rng.normal(size=(784,)).astype(np.float32)
    ids = np.arange(32, dtype=np.int64)
    query = rng.normal(size=(32, 4096)).astype(np.float32)
    memory = SignedKNNFIFO(256)
    first, _, _ = _corrected_c2_score(
        base, ids, query, memory, position=1, group_sha256=_group("query"))
    assert first == float(base.max())
    memory.insert(query, sign=-1, source_group_sha256=_group("anchor"), position=0)
    score, delta, _ = _corrected_c2_score(
        base, ids, query, memory, position=1, group_sha256=_group("query"))
    expected_map = r1.corrected_dense_map(base, ids, delta)
    assert score == float(expected_map.max())
    assert np.array_equal(expected_map[32:], base[32:])


def test_incompatible_capacity_k_or_amplitude_is_rejected():
    with pytest.raises(ValueError):
        SignedKNNFIFO(512)
    memory = SignedKNNFIFO(256)
    with pytest.raises(ValueError, match="unregistered"):
        memory.score(np.ones((1, 4096), dtype=np.float32), event_position=0,
                     event_group_sha256=_group("query"), amplitude=0.2)
    with pytest.raises(ValueError, match="unregistered"):
        memory.score(np.ones((1, 4096), dtype=np.float32), event_position=0,
                     event_group_sha256=_group("query"), neighbors=3)
