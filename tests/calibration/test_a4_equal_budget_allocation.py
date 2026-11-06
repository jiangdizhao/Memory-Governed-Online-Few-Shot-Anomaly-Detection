from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.calibration import a4_equal_budget_allocation as a4
from experiments.calibration import a4_independent_verifier as a4_verifier


@pytest.mark.parametrize("count", [32, 64])
def test_higher_q90_uses_registered_higher_order_statistic(count: int) -> None:
    values = np.arange(count, dtype=np.float64)
    expected = int(np.ceil(0.90 * (count - 1)))
    assert a4._higher_q90(values) == expected


def test_b3_refresh_is_versioned_clipped_and_uses_sample_scale() -> None:
    initial = np.linspace(0.0, 0.01, 32)
    state = a4.B3State(initial)
    old_theta = state.theta
    assert state.sigma == pytest.approx(max(float(initial.std(ddof=1)), 0.02))
    assert state.refresh(np.full(32, 100.0), 0) is False
    assert state.theta == old_theta
    assert state.refresh(np.full(32, 100.0), 1) is True
    assert state.theta == pytest.approx(old_theta + 0.01)
    assert state.sigma == pytest.approx(0.02)
    assert state.history[-1]["version"] == 1
    with pytest.raises(a4.A4Error):
        state.refresh(np.zeros(32), 0)


def test_fixed_bank_profile_fails_closed_on_any_capacity_difference() -> None:
    valid = {"total_prototypes": a4.FIXED_COUNT,
             "total_feature_bytes": a4.FIXED_BYTES,
             "layers": [{"n_prototypes": a4.FIXED_COUNT,
                         "feature_dim": a4.FEATURE_DIM,
                         "dtype": "float16", "bytes": a4.FIXED_BYTES}]}
    a4._validate_fixed_profile(valid)
    invalid = json.loads(json.dumps(valid))
    invalid["layers"][0]["dtype"] = "float32"
    with pytest.raises(a4.A4Error):
        a4._validate_fixed_profile(invalid)


def test_stage_b_projects_every_exact_arm_and_mode_key() -> None:
    row = {"scores": {}, "predictions": {}, "thresholds": {}}
    for number, arm in enumerate(a4.ARMS):
        for mode in ("static", "online"):
            key = a4.SCORE_KEYS[arm][mode]
            row["scores"][key] = number + (0.1 if mode == "static" else 0.2)
            row["predictions"][key] = number % 2
            row["thresholds"][key] = number + 1.0
    for number, arm in enumerate(a4.ARMS):
        for mode in ("static", "online"):
            scores, predictions, thresholds = a4._stage_b_vectors([row], arm, mode)
            assert scores.tolist() == [number + (0.1 if mode == "static" else 0.2)]
            assert predictions.tolist() == [number % 2]
            assert thresholds.tolist() == [number + 1.0]
    with pytest.raises(a4.A4Error):
        a4._stage_b_vectors([row], "A", "prequential")


def test_single_class_window_is_explicitly_undefined() -> None:
    labels = np.zeros(512, dtype=np.int8)
    predictions = np.zeros(512, dtype=np.int8)
    result = a4._window_metric_row(labels, predictions, 0)
    assert result == {"start": 0, "end_exclusive": 512,
                      "status": "INVALID_SINGLE_CLASS_WINDOW",
                      "fpr": None, "fnr": None, "ber": None}


def test_valid_window_computes_fpr_fnr_and_ber() -> None:
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 256)
    predictions = np.zeros(512, dtype=np.int8)
    result = a4._window_metric_row(labels, predictions, 0)
    assert result["status"] == "VALID"
    assert result["fpr"] == 0.0
    assert result["fnr"] == 1.0
    assert result["ber"] == 0.5


def test_native_causal_gate_rejects_early_and_unqueried_disclosure() -> None:
    from experiments.calibration.r1_matched_correction_pilot import (
        CausalEventOrder, R1CausalityError)

    early = CausalEventOrder(4)
    with pytest.raises(R1CausalityError):
        early.disclose(lambda: 0)
    early.seal_scores()
    early.decide_query(False)
    with pytest.raises(R1CausalityError):
        early.disclose(lambda: 0)
    early.corrections_updated()
    early.memory_updated()
    early.finish()

    queried = CausalEventOrder(5)
    queried.seal_scores()
    queried.decide_query(True)
    assert queried.disclose(lambda: 0) == 0
    queried.corrections_updated()
    queried.memory_updated()
    queried.finish()


def test_feedback_loader_rejects_queried_defect_normal_write(tmp_path) -> None:
    path = tmp_path / "feedback.jsonl"
    record = {"schema": "test", "category": "audiojack", "position": 1,
              "original_json_position": 2, "event_id_sha256": "e",
              "image_identity_sha256": "i", "source_group_sha256": "g",
              "label": 1, "score_row_fsynced_before_disclosure": True,
              "writer_actions": {"A": {"action": {
                  "action": "native_greedy_normal_update"}}},
              "state_transitions": {}, "threshold_refreshes": []}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(a4.A4Error):
        a4._load_feedback(path)


def test_a3b_quota_balance_and_kcenter_tie_order_are_deterministic() -> None:
    from experiments.calibration import a_calibration_continuation as a3

    counts = {f"g{i}": 100 for i in range(4)}
    quotas = a3.a3b_candidate_quotas(list(counts), 8, counts, "A4_test")
    assert quotas == {group: 2 for group in counts}
    candidates = np.asarray([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float32)
    identities = ["d", "c", "b", "a"]
    selected = a3.a3b_kcenter_numpy_reference(candidates, identities, 3)
    selected_again = a3.a3b_kcenter_numpy_reference(candidates, identities, 3)
    assert selected == selected_again
    assert identities[selected[0]] == "a"
    assert len(set(selected)) == 3


@pytest.mark.parametrize("category", a4.CATEGORIES)
def test_frozen_a1_roles_and_a2_query_tape(category: str) -> None:
    from experiments.calibration import stage05

    train, test, json_sha = stage05.read_class_stream(
        a4.DATASET_ROOT, a4.JSON_ROOT, category, test_labels=False)
    roles = a4._category_roles(category, train, test, json_sha)
    assert len(roles["support_items"]) == 4
    assert len(roles["c64_items"]) == 64
    assert len(roles["c32_items"]) == len(roles["r32_items"]) == 32
    assert not (set(roles["c32_groups"]) & set(roles["r32_groups"]))
    assert set(roles["c32_groups"]) | set(roles["r32_groups"]) == {
        item["group"] for item in roles["c64_items"]}
    assert roles["no_test_overlap"] is True
    receipt = a4._query_identity_receipt(category, test)
    assert receipt["query_count"] == len(receipt["positions"])
    assert receipt["group_count"] == receipt["query_count"]


def test_write_once_refuses_different_existing_content(tmp_path) -> None:
    path = tmp_path / "immutable.bin"
    a4._write_once(path, b"first")
    with pytest.raises(FileExistsError):
        a4._write_once(path, b"second")
    assert path.read_bytes() == b"first"


def test_run_state_uses_writable_fsync_handle(tmp_path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    a4._write_run_state(run_root, "RUNNING", {"campaign_started_epoch": 1})
    saved = json.loads((run_root / "RUN_STATE.json").read_text(encoding="utf-8"))
    assert saved["status"] == "RUNNING"
    assert saved["campaign_started_epoch"] == 1


def test_source_allowlist_excludes_historical_experiment_files() -> None:
    assert "experiments/calibration/a4_equal_budget_allocation.py" in a4.ALLOWED_A4_SOURCE_PATHS
    assert "experiments/calibration/a_calibration_scale.py" not in a4.ALLOWED_A4_SOURCE_PATHS
    assert not any(path.startswith("outputs/") for path in a4.ALLOWED_A4_SOURCE_PATHS)


def test_source_hash_normalizes_windows_checkout_line_endings() -> None:
    assert a4._sha_source_bytes(b"first\r\nsecond\r\n") == a4._sha_source_bytes(b"first\nsecond\n")


def test_protected_file_audit_preserves_pre_task_untracked_inventory() -> None:
    audit = a4.protected_file_audit()
    assert audit["status"] == "PASS"
    assert audit["preexisting_untracked_inventory"]["matches_pre_task_inventory"] is True


def test_independent_inventory_check_allows_a4_output_counts_to_grow() -> None:
    preflight = {
        "baseline_count": 5,
        "baseline_paths_sha256": "frozen-paths",
        "observed_preexisting_count": 5,
        "observed_preexisting_paths_sha256": "frozen-paths",
        "matches_pre_task_inventory": True,
        "task_output_untracked_count": 1,
        "task_report_untracked_count": 0,
    }
    after_run = {
        **preflight,
        "task_output_untracked_count": 42,
        "task_report_untracked_count": 8,
    }
    assert a4_verifier._same_preexisting_untracked_inventory(after_run, preflight) is True

    changed = {**after_run, "observed_preexisting_count": 6}
    assert a4_verifier._same_preexisting_untracked_inventory(changed, preflight) is False


def test_independent_model_scorer_identity_requires_frozen_receipts() -> None:
    model_receipt = {
        "loaded_model_state_sha256": "loaded-state",
        "checkpoint_sha256": "checkpoint",
        "hub_source_tree_sha256": "hub-tree",
    }
    model = {
        "loaded_model_state_sha256": "loaded-state",
        "runtime_profile": {
            "checkpoint_sha256": "checkpoint",
            "hub_source_tree_sha256": "hub-tree",
            "runtime_profile": {"cuda_device_name": "registered-gpu"},
            "layer_indices": [5, 11, 17, 23],
            "image_size": 392,
            "resize_size": 448,
        },
        "scorer_profile": {
            "model_name": "dinov2_vitl14_reg",
            "image_size": 392,
            "resize_size": 448,
            "use_fp16": True,
            "view_names": ["id"],
            "knn": 5,
            "image_aggregation": "max",
            "feature_fuse": "concat",
            "fixed_support_count": 4,
        },
    }
    assert a4_verifier._frozen_model_scorer_identity_matches(
        model, model_receipt, "registered-gpu") is True
    assert a4_verifier._frozen_model_scorer_identity_matches(
        model, model_receipt, "unexpected-gpu") is False
    assert a4_verifier._frozen_model_scorer_identity_matches(
        None, model_receipt, "registered-gpu") is False
