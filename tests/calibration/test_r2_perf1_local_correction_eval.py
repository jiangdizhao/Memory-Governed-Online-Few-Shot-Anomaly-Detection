from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from experiments.calibration import r2_perf1_stage_a as stage_a
from experiments.calibration import r2_perf1_stage_b as stage_b
from experiments.calibration.r1_matched_correction_pilot import (
    _validate_c2_anchor_state, arrays_sha256, signed_knn_delta)
from experiments.calibration.r2_0_label_blind_mechanism_audit import (
    corrected_dense_map, max_score)
from experiments.calibration.r2_perf1_common import (
    MAX_OUTPUT_BYTES, OUTPUT_PARENT, OutputBudget, new_run_id, output_path)


def _anchor(position: int, group: str, sign: int, feature: np.ndarray):
    value = np.asarray(feature, dtype=np.float32)
    return {
        "position": position,
        "group_sha256": group,
        "patch_id": position % 784,
        "sign": sign,
        "feature": value,
        "feature_sha256": arrays_sha256({"feature": value}),
    }


def test_vectorized_cosine_distance_matches_scalar_reference():
    rng = np.random.default_rng(41)
    queries = rng.normal(size=(7, 13)).astype(np.float32)
    anchors = rng.normal(size=(11, 13)).astype(np.float32)
    observed = stage_a.vectorized_cosine_distances(queries, anchors)
    from experiments.calibration.r2_l1_locality_kernels import cosine_distances
    normalized_queries = queries / np.maximum(
        np.linalg.norm(queries, axis=1, keepdims=True), np.float32(1e-12))
    normalized_anchors = anchors / np.maximum(
        np.linalg.norm(anchors, axis=1, keepdims=True), np.float32(1e-12))
    expected = np.stack([cosine_distances(row, normalized_anchors)
                         for row in normalized_queries])
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-6)


def test_c2_gated_and_c3_are_identical_when_they_use_the_same_top_k():
    distances = np.asarray([[0.01, 0.02, 0.03, 0.04], [0.04, 0.01, 0.02, 0.03]],
                           dtype=np.float32)
    signs = [1, -1, 1, -1]
    order = [3, 1, 2, 0]
    gated, _ = stage_a.candidate_deltas(
        distances, signs, anchor_order=order, radius=0.10,
        regularizer=0.10, k=5)
    compact_top_k, _ = stage_a.candidate_deltas(
        distances, signs, anchor_order=order, radius=0.10,
        regularizer=0.10, k=None)
    np.testing.assert_array_equal(gated, compact_top_k)


def test_all_local_c3_can_differ_from_nearest_five_gated_baseline():
    distances = np.asarray([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]], dtype=np.float32)
    signs = [1, 1, 1, 1, 1, -1]
    gated, _ = stage_a.candidate_deltas(
        distances, signs, anchor_order=range(6), radius=0.10,
        regularizer=0.10, k=5)
    all_local, _ = stage_a.candidate_deltas(
        distances, signs, anchor_order=range(6), radius=0.10,
        regularizer=0.10, k=None)
    assert gated[0] != all_local[0]


def test_strict_radius_excludes_anchor_exactly_at_radius():
    distances = np.asarray([[0.0, 0.10]], dtype=np.float32)
    delta, stats = stage_a.candidate_deltas(
        distances, [1, -1], anchor_order=[0, 1], radius=0.10,
        regularizer=0.10, k=None)
    assert stats["local_anchor_count_per_patch"].tolist() == [1]
    assert delta[0] > 0


def test_sign_conflict_and_regularization_are_bounded():
    distances = np.asarray([[0.0, 0.0]], dtype=np.float32)
    delta, stats = stage_a.candidate_deltas(
        distances, [1, -1], anchor_order=[0, 1], radius=0.10,
        regularizer=0.10, k=None)
    assert stats["local_positive_anchor_count_per_patch"].tolist() == [1]
    assert stats["local_negative_anchor_count_per_patch"].tolist() == [1]
    assert delta[0] == 0.0


def test_queried_c2_writer_appends_32_and_retains_last_256_fifo_anchors():
    rng = np.random.default_rng(3)
    existing = []
    ids = np.arange(32, dtype=np.int64)
    features = rng.normal(size=(32, 4096)).astype(np.float32)
    for position in range(17):
        existing = stage_a.append_queried_c2_anchors(
            existing, event_position=position,
            event_group=f"{position + 1:064x}", top32_ids=ids,
            top32_features=features, sign=(-1 if position % 2 == 0 else 1),
            capacity=256)
    assert len(existing) == 256
    assert {a["position"] for a in existing} == set(range(9, 17))
    assert all(a["sign"] == (-1 if a["position"] % 2 == 0 else 1) for a in existing)
    assert all(len(a["feature_sha256"]) == 64 for a in existing)


def test_c2_sign_convention_source_group_exclusion_and_current_future_rejection():
    group_a, group_b = "a" * 64, "b" * 64
    feature = np.asarray([1.0, 0.0], dtype=np.float32)
    positive = _anchor(2, group_a, 1, feature)
    negative = _anchor(3, group_a, -1, feature)
    pos, count = signed_knn_delta(
        np.asarray([[1.0, 0.0]], dtype=np.float32), [positive],
        event_position=5, event_group=group_b)
    neg, count_neg = signed_knn_delta(
        np.asarray([[1.0, 0.0]], dtype=np.float32), [negative],
        event_position=5, event_group=group_b)
    assert count == count_neg == 1
    assert pos.tolist() == [np.float32(0.20)]
    assert neg.tolist() == [np.float32(-0.20)]
    same_source = _anchor(2, group_b, 1, feature)
    _, excluded_count = signed_knn_delta(
        np.asarray([[1.0, 0.0]], dtype=np.float32), [same_source],
        event_position=5, event_group=group_b)
    assert excluded_count == 0
    for invalid_position in (5, 6):
        invalid = _anchor(invalid_position, group_a, 1, feature)
        with pytest.raises(Exception, match="current/future"):
            _validate_c2_anchor_state(
                [invalid], event_position=5, phase="pre_score", event_group=group_b)


def test_correction_is_applied_only_to_top32_then_max_aggregated():
    base = np.zeros(784, dtype=np.float32)
    base[5] = 0.5
    base[6] = 0.7
    corrected = corrected_dense_map(base, [5, 7], np.asarray([0.3, -0.2], dtype=np.float32))
    assert corrected[5] == pytest.approx(0.8)
    assert corrected[7] == pytest.approx(-0.2)
    assert corrected[6] == base[6]
    assert max_score(corrected) == pytest.approx(0.8)


def test_score_is_flushed_before_query_disclosure_and_unqueried_reader_is_never_called():
    calls = []
    stage_a.seal_then_disclose(
        {"position": 7}, queried=True,
        seal_score=lambda row: calls.append(("seal", row["position"])),
        disclose=lambda: calls.append(("reveal", 7)) or "label")
    assert calls == [("seal", 7), ("reveal", 7)]
    calls.clear()
    stage_a.seal_then_disclose(
        {"position": 8}, queried=False,
        seal_score=lambda row: calls.append(("seal", row["position"])),
        disclose=lambda: pytest.fail("unqueried disclosure attempted"))
    assert calls == [("seal", 8)]


def test_stage_a_source_has_no_test_json_path_or_outcome_parser():
    source = Path(stage_a.__file__).read_text(encoding="utf-8")
    assert "switch.json" not in source
    assert "anomaly_class" not in source
    assert "mask_path" not in source


def test_stage_b_decodes_only_target_prefix_after_identity_callback():
    text = json.dumps({
        "meta": {"normal_class": "OK", "prefix": "switch"},
        "train": [{"anomaly_class": "TRAIN_SECRET", "image_path": "train.jpg"}],
        "test": [
            {"anomaly_class": "D1", "image_path": "one.jpg", "mask_path": {"private": "m1"}},
            {"image_path": "two.jpg", "mask_path": "m2", "anomaly_class": "OK"},
            {"anomaly_class": "OUTSIDE_PREFIX_SECRET", "image_path": "later.jpg"},
        ],
    })
    projected = []

    def callback(position, image_path, raw_record):
        assert image_path in {"one.jpg", "two.jpg"}
        label = stage_b._matched_test_label(
            raw_record, expected_identity=f"identity-{position}",
            observed_identity=f"identity-{position}")
        projected.append((position, label))

    count = stage_b._TestPrefixProjection(text).project([{}, {}], callback)
    assert count == 2
    assert projected == [(0, "D1"), (1, "OK")]


def test_misaligned_stage_b_identity_fails_before_label_projection(monkeypatch):
    def must_not_project(_raw):
        pytest.fail("outcome decoder ran before identity validation")

    monkeypatch.setattr(stage_b, "_project_test_label_for_matched_record", must_not_project)
    with pytest.raises(stage_b.StageBError, match="not aligned"):
        stage_b._matched_test_label("{\"anomaly_class\":\"SECRET\"}",
                                   expected_identity="a", observed_identity="b")


def test_auc_auprc_and_frozen_threshold_metrics():
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.4, 0.35, 0.8], dtype=np.float64)
    auc, ap, reason = stage_b._auc_ap(labels, scores)
    assert auc == pytest.approx(0.75)
    assert ap == pytest.approx(5.0 / 6.0)
    assert reason is None
    threshold = stage_b._fixed_threshold_metrics(labels, scores, 0.4)
    assert threshold["tp"] == 1
    assert threshold["fp"] == 1
    assert threshold["fn"] == 1
    assert threshold["tn"] == 1
    assert threshold["fpr"] == pytest.approx(0.5)
    assert threshold["fnr"] == pytest.approx(0.5)
    assert threshold["balanced_accuracy"] == pytest.approx(0.5)


def test_one_class_metric_is_explicitly_undefined():
    auc, ap, reason = stage_b._auc_ap([1, 1, 1], [0.2, 0.3, 0.4])
    assert auc is None and ap is None
    assert reason == "AUROC and AUPRC require both classes"


def test_group_cluster_bootstrap_is_reproducible_and_paired():
    labels = [0, 1, 0, 1, 0, 1]
    groups = ["a", "a", "b", "b", "c", "c"]
    scores = {"c0": [0.1, 0.8, 0.2, 0.7, 0.3, 0.6],
             "c3_r0.10_l0.10": [0.2, 0.9, 0.2, 0.7, 0.1, 0.8],
             "c2gated_r0.10_l0.10": [0.1, 0.7, 0.2, 0.8, 0.3, 0.6]}
    first = stage_b._cluster_bootstrap(labels, groups, scores, iterations=40, seed=77)
    second = stage_b._cluster_bootstrap(labels, groups, scores, iterations=40, seed=77)
    assert first == second
    assert first["unique_source_groups"] == 3
    assert first["conditional_on_fixed_historical_causal_trajectory"] is True


def test_stage_a_grid_matches_launch_grid_and_is_frozen():
    from experiments.calibration.r2_perf1_local_correction_eval import _configuration_record
    config = _configuration_record()
    grid = stage_a._method_grid()
    assert config["radii"] == [0.02, 0.05, 0.1, 0.2, 0.4]
    assert config["regularizers"] == [0.1, 1.0]
    assert len(grid) == 20
    assert {(row["method"], row["radius"], row["lambda"]) for row in grid} == {
        (method, radius, regularizer)
        for radius in config["radii"] for regularizer in config["regularizers"]
        for method in ("C2-Gated", "C3")}


@pytest.fixture
def synthetic_output_root():
    root = output_path(new_run_id())
    root.mkdir(exist_ok=False)
    try:
        yield root
    finally:
        resolved = root.resolve(strict=True)
        assert resolved.parent == OUTPUT_PARENT.resolve(strict=True)
        shutil.rmtree(resolved)


def test_output_budget_refuses_prewrite_overflow_and_preserves_partial_output(synthetic_output_root):
    budget = OutputBudget(synthetic_output_root, allow_existing=True,
                          max_bytes=32, failure_reserve_bytes=8)
    budget.write("run_manifest.json", b"m" * 24)
    with pytest.raises(Exception, match="before I/O"):
        budget.write("stage_a_scores.jsonl", b"s")
    assert (synthetic_output_root / "run_manifest.json").stat().st_size == 24
    budget.write("failure.json", b"f" * 8, failure_reserved=True)
    assert budget._audit() == 32
    assert (synthetic_output_root / "run_manifest.json").read_bytes() == b"m" * 24


def test_output_budget_refuses_overwrite_unknown_files_and_historical_reuse(synthetic_output_root):
    budget = OutputBudget(synthetic_output_root, allow_existing=True,
                          max_bytes=1024, failure_reserve_bytes=64)
    budget.write("run_manifest.json", b"one")
    with pytest.raises(FileExistsError):
        budget.write("run_manifest.json", b"two")
    with pytest.raises(Exception, match="allowlist"):
        budget.write("private_feedback.jsonl", b"unaccounted")
    with pytest.raises(FileExistsError):
        OutputBudget(synthetic_output_root, allow_existing=False,
                     max_bytes=1024, failure_reserve_bytes=64)


def test_score_loader_rejects_misalignment_and_tampered_seal(tmp_path):
    rows = []
    for position in range(512):
        rows.append({
            "schema": "calibration.r2_perf1.score_row.v1", "position": position,
            "source_image_identity_sha256": hashlib.sha256(str(position).encode()).hexdigest(),
            "source_group_sha256": hashlib.sha256(f"g{position}".encode()).hexdigest(),
            "scores": {"c0": 0.1, "c1": 0.1, "c2": 0.1,
                       "c2gated_r0.10_l0.10": 0.1, "c3_r0.10_l0.10": 0.1},
            "argmax_patch": {"c0": 0, "c1": 0, "c2": 0,
                             "c2gated_r0.10_l0.10": 0, "c3_r0.10_l0.10": 0},
        })
    path = tmp_path / "stage_a_scores.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    seal = {"status": "SEALED", "event_count": 512, "outcome_labels_read": False,
            "score_table_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    loaded = stage_b._load_and_verify_sealed_scores(tmp_path, seal)
    assert len(loaded) == 512
    tampered = dict(seal, score_table_sha256="0" * 64)
    with pytest.raises(stage_b.StageBError, match="SHA-256"):
        stage_b._load_and_verify_sealed_scores(tmp_path, tampered)
    rows[7]["position"] = 8
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    seal["score_table_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(stage_b.StageBError, match="misaligned"):
        stage_b._load_and_verify_sealed_scores(tmp_path, seal)
