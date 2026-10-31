"""Synthetic CPU tests for the label-blind R1 mechanism audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.calibration.r2_0_label_blind_mechanism_audit import (
    EPS_PATCH,
    ReadOnlyR1Evidence,
    R2AuditBlocked,
    _argmax_diagnostics,
    _eligibility_category,
    _pair_rank_summary,
    _sha256_json,
    corrected_dense_map,
    extract_json_object_fields,
    max_score,
    patch_metrics,
    validate_positions,
    verify_public_tape,
)


def _synthetic_arrays(*, c1_delta=None, c2_delta=None, base=None, ids=None):
    ids = np.arange(32, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
    base = np.zeros(784, dtype=np.float32) if base is None else np.asarray(base, dtype=np.float32)
    c1 = np.zeros(32, dtype=np.float32) if c1_delta is None else np.asarray(c1_delta, dtype=np.float32)
    c2 = np.zeros(32, dtype=np.float32) if c2_delta is None else np.asarray(c2_delta, dtype=np.float32)
    return {
        "base_patch_map": base,
        "top32_ids": ids,
        "top32_features": np.zeros((32, 4096), dtype=np.float32),
        "c1_delta": c1,
        "c2_delta": c2,
    }


def _array_sha256(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(value.shape), value.dtype.str)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _write_synthetic_tape(root: Path, *, corrupt_shard=False, malformed_arrays=False):
    (root / "events").mkdir(parents=True)
    arrays = _synthetic_arrays()
    if malformed_arrays:
        arrays["c1_delta"] = np.zeros(31, dtype=np.float32)
    shard_path = root / "events" / "event_000000.npz"
    np.savez(shard_path, **arrays)
    raw = shard_path.read_bytes()
    shard_sha = hashlib.sha256(raw).hexdigest()
    if corrupt_shard:
        shard_path.write_bytes(raw + b"corruption")
    c1_for_score = (arrays["c1_delta"] if arrays["c1_delta"].shape == (32,)
                    else np.zeros(32, dtype=np.float32))
    row = {
        "position": 0,
        "array_fields": sorted(arrays),
        "array_sha256": _array_sha256(arrays),
        "shard_path": "events/event_000000.npz",
        "shard_sha256": shard_sha,
        "c0_score": max_score(arrays["base_patch_map"]),
        "c1_score": max_score(corrected_dense_map(
            arrays["base_patch_map"], arrays["top32_ids"], c1_for_score)),
        "c2_score": max_score(corrected_dense_map(
            arrays["base_patch_map"], arrays["top32_ids"], arrays["c2_delta"])),
        "official_group_sha256": "a" * 64,
        "fixed_support_sha256": "b" * 64,
        "fixed_support_raw_prototype_sha256": "c" * 64,
        "greedy_memory_version_before": 0,
        "greedy_memory_sha256_before": "d" * 64,
        "c1_state_sha256_before": "e" * 64,
        "c1_eligible_anchor_slots": 0,
        "c1_same_source_excluded_anchor_slots": 0,
        "c1_far_field_patch_count": 0,
        "c1_far_field_abs_correction_max": 0.0,
        "c2_state_sha256_before": "f" * 64,
        "c2_eligible_anchor_count": 0,
        "score_state_sha256": "1" * 64,
    }
    transition = {
        "position": 0,
        "query_decision": True,
        "causal_order_complete": True,
        "greedy_memory_version_after": 0,
        "greedy_memory_sha256_after": "d" * 64,
        "greedy_memory_content_sha256_after": "2" * 64,
        "c1_state_sha256_after": "e" * 64,
        "c2_state_sha256_after": "f" * 64,
        # The reader must skip these fields without returning or interpreting them.
        "writer_action": {"classification": "label-derived-do-not-read"},
        "feedback_reference": "private-ledger-reference-do-not-read",
    }
    (root / "sealed_scores.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (root / "event_transitions.jsonl").write_text(
        json.dumps(transition) + "\n", encoding="utf-8")


def test_exact_dense_map_score_reconstruction():
    base = np.zeros(784, dtype=np.float32)
    base[4] = 0.75
    base[7] = 0.80
    ids = np.array([4, 7], dtype=np.int64)
    delta = np.array([0.10, -0.20], dtype=np.float32)
    corrected = corrected_dense_map(base, ids, delta)
    assert corrected[4] == np.float32(0.85)
    assert corrected[7] == np.float32(0.60)
    assert max_score(base) == float(np.float32(0.8))
    assert max_score(corrected) == float(corrected[4])


def test_nonzero_patch_correction_can_be_masked_by_unchanged_image_max():
    base = np.zeros(784, dtype=np.float32)
    base[0] = 0.9
    base[1] = 0.4
    ids = np.array([1, 2], dtype=np.int64)
    delta = np.array([0.2, -0.1], dtype=np.float32)
    corrected = corrected_dense_map(base, ids, delta)
    assert np.count_nonzero(delta) == 2
    assert max_score(corrected) == max_score(base)
    diagnostic = _argmax_diagnostics(base, corrected, ids)
    assert diagnostic["base_argmax"] == diagnostic["corrected_argmax"] == 0
    assert not diagnostic["base_argmax_in_top32"]
    assert diagnostic["selected_vs_unchanged_exact_relation"] == "unchanged_wins"


def test_correction_can_change_maximizing_patch_and_score_signs():
    base = np.zeros(784, dtype=np.float32)
    base[0] = 0.9
    base[1] = 0.8
    ids = np.array([1], dtype=np.int64)
    positive = corrected_dense_map(base, ids, np.array([0.2], dtype=np.float32))
    assert max_score(positive) > max_score(base)
    assert _argmax_diagnostics(base, positive, ids)["argmax_changed"]
    negative = corrected_dense_map(base, np.array([0]), np.array([-0.2], dtype=np.float32))
    assert max_score(negative) < max_score(base)


def test_patch_tolerance_uses_strict_boundary_and_reports_exact_nonzero():
    values = np.array([
        EPS_PATCH,
        np.nextafter(EPS_PATCH, np.inf),
        -EPS_PATCH,
        np.nextafter(-EPS_PATCH, -np.inf),
        0.0,
    ], dtype=np.float64)
    result = patch_metrics(values)
    assert result["nonzero_exact_count"] == 4
    assert result["nonzero_tolerance_count"] == 2
    assert result["exact_nonzero_but_below_tolerance_count"] == 2
    assert result["positive_tolerance_count"] == 1
    assert result["negative_tolerance_count"] == 1


def test_eligibility_funnel_distinguishes_no_anchors_zero_and_active():
    assert _eligibility_category(0, 0) == "NO_ELIGIBLE_ANCHORS"
    assert _eligibility_category(3, 0) == "ELIGIBLE_ANCHORS_BUT_ZERO_PATCH_CORRECTION"
    assert _eligibility_category(3, 2) == "NONZERO_PATCH_CORRECTION"


def test_rank_ties_and_strict_reversals_are_counted_correctly():
    reversed_summary = _pair_rank_summary([1, 2, 3], [3, 2, 1])
    assert reversed_summary["strict_reversed_pairs"] == 3
    assert reversed_summary["comparable_pairs_untied_in_both"] == 3
    assert reversed_summary["strict_reversal_fraction_of_comparable_pairs"] == 1.0
    assert reversed_summary["spearman_rank_difference_correlation"] == pytest.approx(-1.0)
    assert reversed_summary["kendall_tau_b"] == pytest.approx(-1.0)

    tie_summary = _pair_rank_summary([1, 1, 2], [1, 2, 2])
    assert tie_summary["ties_in_baseline"] == 1
    assert tie_summary["ties_in_corrected"] == 1
    assert tie_summary["newly_untied_pairs"] == 1
    assert tie_summary["newly_tied_pairs"] == 1
    assert tie_summary["strict_reversed_pairs"] == 0


def test_cross_source_group_pair_filter_excludes_same_group_pair():
    result = _pair_rank_summary([1, 2, 3], [3, 2, 1], ["same", "same", "other"])
    assert result["total_pairs_in_scope"] == 2
    assert result["same_source_group_pairs_excluded"] == 1
    assert result["comparable_pairs_untied_in_both"] == 2
    assert result["strict_reversed_pairs"] == 2


def test_constant_rank_inputs_report_undefined_correlations():
    result = _pair_rank_summary([1, 1, 1], [2, 2, 2])
    assert result["spearman_rank_difference_correlation"] is None
    assert result["kendall_tau_b"] is None
    assert not result["kendall_tau_b_defined"]


def test_transition_projection_skips_feedback_derived_fields():
    raw = json.dumps({
        "position": 4,
        "query_decision": True,
        "causal_order_complete": True,
        "writer_action": {"label": "secret-value", "nested": [1, 2, {"x": 3}]},
        "feedback_reference": "private-id",
    }).encode("utf-8")
    projected = extract_json_object_fields(raw, {
        "position", "query_decision", "causal_order_complete"})
    assert projected == {
        "position": 4,
        "query_decision": True,
        "causal_order_complete": True,
    }
    assert "writer_action" not in projected
    assert "feedback_reference" not in projected


def test_reader_refuses_label_bearing_paths_before_opening(tmp_path, monkeypatch):
    root = tmp_path / "r1-evidence"
    root.mkdir()
    (root / "private_feedback.jsonl").write_text('{"label":1}\n', encoding="utf-8")
    (root / "correction_anchor_events.jsonl").write_text('{"sign":1}\n', encoding="utf-8")
    reader = ReadOnlyR1Evidence(root)
    opened = []
    original_open = Path.open

    def spy_open(path, *args, **kwargs):
        opened.append(Path(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy_open)
    for forbidden in ("private_feedback.jsonl", "correction_anchor_events.jsonl"):
        with pytest.raises(R2AuditBlocked, match="not authorized"):
            reader.read_authorized_bytes(forbidden)
    with pytest.raises(R2AuditBlocked, match="structural-field projection"):
        reader.read_authorized_bytes("event_transitions.jsonl")
    assert opened == []


def test_integrity_verifier_reconstructs_public_synthetic_tape(tmp_path):
    _write_synthetic_tape(tmp_path)
    reader = ReadOnlyR1Evidence(tmp_path)
    events, receipt = verify_public_tape(
        reader, expected_count=1,
        expected_query_digest=_sha256_json([0]),
        expected_fixed_bank_digest="b" * 64)
    assert len(events) == 1
    assert events[0]["scores"] == {"c0": 0.0, "c1": 0.0, "c2": 0.0}
    assert receipt["all_recorded_shard_and_array_hashes_verified"]
    assert receipt["all_c0_c1_c2_scores_reconstructed_exactly"]
    assert receipt["query_positions"] == [0]


def test_integrity_verifier_rejects_corrupted_shard_hash(tmp_path):
    _write_synthetic_tape(tmp_path, corrupt_shard=True)
    reader = ReadOnlyR1Evidence(tmp_path)
    with pytest.raises(R2AuditBlocked, match="Shard SHA-256 mismatch"):
        verify_public_tape(reader, expected_count=1)


def test_integrity_verifier_rejects_malformed_event_arrays(tmp_path):
    _write_synthetic_tape(tmp_path, malformed_arrays=True)
    reader = ReadOnlyR1Evidence(tmp_path)
    with pytest.raises(R2AuditBlocked, match="Malformed event array c1_delta"):
        verify_public_tape(reader, expected_count=1)


def test_incomplete_duplicate_and_missing_event_positions_fail_closed():
    with pytest.raises(R2AuditBlocked, match="exactly 0..1"):
        validate_positions([0], 2)
    with pytest.raises(R2AuditBlocked, match="exactly 0..1"):
        validate_positions([0, 0], 2)
    with pytest.raises(R2AuditBlocked, match="exactly 0..2"):
        validate_positions([0, 1, 3], 3)


def test_missing_shard_fails_closed(tmp_path):
    (tmp_path / "events").mkdir()
    reader = ReadOnlyR1Evidence(tmp_path)
    with pytest.raises(R2AuditBlocked, match="missing"):
        reader.read_event_shard(0)


def test_json_projection_rejects_duplicate_keys_and_trailing_commas():
    with pytest.raises(R2AuditBlocked, match="duplicate"):
        extract_json_object_fields('{"position":1,"position":2}', {"position"})
    with pytest.raises(R2AuditBlocked, match="Trailing comma"):
        extract_json_object_fields('{"position":1,}', {"position"})
