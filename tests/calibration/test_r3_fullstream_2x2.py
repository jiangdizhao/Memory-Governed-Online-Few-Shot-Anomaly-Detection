"""Synthetic CPU contract tests for the R3 full-stream runner; no dataset access."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from experiments.calibration import r3_fullstream_2x2 as r3
from experiments.calibration.r2_feedback_repair import OutputBudget, RepairError


def test_canonical_bytes_are_stable_and_reject_nan():
    assert r3.canonical_bytes({"b": 2, "a": 1}) == r3.canonical_bytes({"a": 1, "b": 2})
    with pytest.raises(ValueError):
        r3.canonical_bytes({"not_finite": float("nan")})


def test_query_label_is_not_read_for_unqueried_event():
    calls = []

    class Writer:
        def flush(self):
            calls.append("flush")

    def reader(*args):
        calls.append(("read", args))
        return 0

    assert r3._disclose_query_label_after_seal(
        Writer(), query_decision=False, category="switch", position=4,
        sample_key="opaque", label_reader=reader) is None
    assert calls == []


def test_query_scores_are_flushed_before_feedback_read():
    calls = []

    class Writer:
        def flush(self):
            calls.append("flush")

    def reader(category, position, sample_key):
        calls.append(("read", category, position, sample_key))
        return 1

    value = r3._disclose_query_label_after_seal(
        Writer(), query_decision=True, category="switch", position=7,
        sample_key="opaque", label_reader=reader)
    assert value == 1
    assert calls == ["flush", ("read", "switch", 7, "opaque")]


def test_query_decision_without_reader_fails_closed():
    with pytest.raises(r3.R3Error, match="restricted label reader"):
        r3._disclose_query_label_after_seal(
            object(), query_decision=True, category="switch", position=1,
            sample_key="opaque", label_reader=None)


def test_stage_a_child_environment_restores_authorized_gpu_visibility():
    base = {"CUDA_VISIBLE_DEVICES": "", "R3_STAGE_B_CPU_ONLY": "1"}
    env = r3._child_environment(base, "STAGE_A", None)
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "R3_STAGE_B_CPU_ONLY" not in env
    assert r3._child_environment(base, "STAGE_A", "0")["CUDA_VISIBLE_DEVICES"] == "0"


def test_stage_b_child_environment_remains_cpu_only():
    env = r3._child_environment({"CUDA_VISIBLE_DEVICES": "0"}, "STAGE_B", "0")
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["R3_STAGE_B_CPU_ONLY"] == "1"


def test_r2_budget_import_cuda_side_effect_can_be_reversed(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    r3._restore_cuda_visibility(None)
    assert "CUDA_VISIBLE_DEVICES" not in r3.os.environ
    r3._restore_cuda_visibility("0")
    assert r3.os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_stage_b_projection_reads_only_labels_with_trailing_array_whitespace(tmp_path):
    path = tmp_path / "synthetic_category.json"
    path.write_text(
        '{"meta":{"normal_class":"good"},"test":['
        '{"anomaly_class":"good","mask_path":"missing-mask-a.png"},'
        '{"anomaly_class":"scratch","mask_path":"missing-mask-b.png"}   ]}',
        encoding="utf-8",
    )
    labels = r3._read_stage_b_label_projection(path, 2)
    assert labels.tolist() == [0, 1]
    assert not (tmp_path / "missing-mask-a.png").exists()
    assert not (tmp_path / "missing-mask-b.png").exists()


def test_stage_b_projection_rejects_wrong_frozen_count(tmp_path):
    path = tmp_path / "synthetic_category.json"
    path.write_text(
        '{"meta":{"normal_class":"good"},"test":[{"anomaly_class":"good"}]}',
        encoding="utf-8",
    )
    with pytest.raises(r3.R3Error, match="array ended before the frozen event count"):
        r3._read_stage_b_label_projection(path, 2)


def test_stage_a_reuse_compatibility_is_exact_and_input_bound(monkeypatch):
    monkeypatch.setattr(r3, "_check_repository_identity", lambda: {})
    monkeypatch.setattr(r3, "_source_fingerprints", lambda: {"source.py": "same"})
    monkeypatch.setattr(r3, "sha256_file", lambda _path: "current-runner")
    accepted_sha = next(iter(r3.STAGE_A_REUSE_COMPATIBLE_RUNNERS))
    manifest = {
        "source_parent_commit": r3.EXPECTED_BASELINE,
        "source_fingerprints": {"source.py": "same"},
        "runner_source_sha256": accepted_sha,
    }
    assert r3._verify_reusable_stage_a_binding(manifest) == \
        "stage-b-projection-and-parent-reporting-repairs-only"
    manifest["runner_source_sha256"] = "unreviewed-runner"
    with pytest.raises(r3.R3Error, match="explicit reuse compatibility"):
        r3._verify_reusable_stage_a_binding(manifest)


def test_stage_a_reuse_identity_allows_only_the_versioned_runner_change():
    old_runner = next(iter(r3.STAGE_A_REUSE_COMPATIBLE_RUNNERS))
    current = {"runner_source_sha256": "current", "config": {"arms": ["S", "SC", "G", "GC"]}}
    old_identity_sha = r3._identity_sha_with_runner(current, old_runner)
    assert old_identity_sha == r3.sha256_json({
        "runner_source_sha256": old_runner,
        "config": {"arms": ["S", "SC", "G", "GC"]},
    })
    changed_experiment = {**current, "config": {"arms": ["S", "SC"]}}
    assert r3._identity_sha_with_runner(changed_experiment, old_runner) != old_identity_sha


def test_total_wall_includes_reused_category_compute(monkeypatch):
    monkeypatch.setattr(r3.time, "monotonic", lambda: 120.0)
    assert r3._overall_wall_seconds(100.0, 900.0) == 920.0


def test_collected_test_case_count_counts_functions_and_class_methods(tmp_path):
    source = tmp_path / "synthetic_test_module.py"
    source.write_text(
        "def test_alpha(): pass\n"
        "def helper(): pass\n"
        "class TestGroup:\n"
        "    def test_beta(self): pass\n"
        "    def helper(self): pass\n",
        encoding="utf-8",
    )
    assert r3._collected_test_case_count(source) == 2


def test_context_test_result_reflects_supervised_exit_and_count():
    assert r3.context_test_result(
        {"status": "COMPLETED", "child_exit_code": 0}, 28) == \
        "28 passed, 0 failed, 0 skipped"
    assert r3.context_test_result(
        {"status": "FAILED_NONZERO_EXIT", "child_exit_code": 1}, 28).startswith("failed:")


def test_scientific_conclusion_reports_negative_interaction_and_fpr_tradeoff():
    categories = {}
    for category in r3.CATEGORIES:
        categories[category] = {
            "factorial_contrasts": {"auroc": {
                "GC_minus_G": -0.1, "GC_minus_SC": 0.05, "interaction": -0.05}},
            "paired_source_group_bootstrap": {"contrasts": {"auroc": {
                "GC_minus_G": {"ci_95_percentile": [-0.2, -0.01]},
                "GC_minus_SC": {"ci_95_percentile": [0.01, 0.1]},
                "interaction": {"ci_95_percentile": [-0.1, -0.01]},
            }}},
            "arms": {"GC": {"fpr": 0.1}, "G": {"fpr": 0.01}},
        }
    conclusion = r3._scientific_conclusion({"categories": categories})
    assert "factorial interaction was negative" in conclusion
    assert "higher FPR than G" in conclusion
    assert "; regulator:" in conclusion


def test_output_budget_refuses_overflow_before_write(tmp_path):
    root = tmp_path / "run"
    budget = OutputBudget(root, create=True, max_bytes=1024,
                          failure_reserve_bytes=128, max_files=4)
    budget.write_once("run_manifest.json", b"x" * 800)
    before = (root / "run_manifest.json").stat().st_size
    with pytest.raises(RepairError, match="byte budget"):
        budget.write_once("stage_a_seal.json", b"y" * 200)
    assert (root / "run_manifest.json").stat().st_size == before
    assert not (root / "stage_a_seal.json").exists()


def test_output_budget_reserves_failure_metadata(tmp_path):
    root = tmp_path / "run"
    budget = OutputBudget(root, create=True, max_bytes=1024,
                          failure_reserve_bytes=128, max_files=4)
    budget.write_once("run_manifest.json", b"x" * 895)
    with pytest.raises(RepairError, match="byte budget"):
        budget.write_once("stage_a_seal.json", b"y")
    budget.write_json("failure.json", {"status": "failed"}, failure_reserved=True)
    assert json.loads((root / "failure.json").read_text()) == {"status": "failed"}
    assert budget.audit()["bytes"] <= 1024


def test_output_budget_rejects_unaccounted_path(tmp_path):
    root = tmp_path / "run"
    budget = OutputBudget(root, create=True, max_bytes=1024,
                          failure_reserve_bytes=128, max_files=4)
    (root / "foreign.bin").write_bytes(b"outside writer")
    with pytest.raises(RepairError, match="Unaccounted output path"):
        budget.audit()


def test_output_budget_refuses_overwrite(tmp_path):
    root = tmp_path / "run"
    budget = OutputBudget(root, create=True, max_bytes=1024,
                          failure_reserve_bytes=128, max_files=4)
    budget.write_once("run_manifest.json", b"original")
    with pytest.raises(RepairError, match="overwrite"):
        budget.write_once("run_manifest.json", b"replacement")
    assert (root / "run_manifest.json").read_bytes() == b"original"


def test_weighted_rank_metrics_use_half_credit_for_ties():
    result = r3._weighted_rank_metrics(
        np.array([0.0, 1.0, 1.0, 2.0]), np.array([0, 0, 1, 1]))
    assert result["auroc"] == pytest.approx(0.875)
    assert result["average_precision"] == pytest.approx(5 / 6)


def test_single_class_stream_has_undefined_auroc_but_defined_ap():
    result = r3._weighted_rank_metrics(
        np.array([0.1, 0.9, 0.2]), np.array([1, 1, 1]))
    assert result["auroc"] is None
    assert result["average_precision"] == pytest.approx(1.0)


def test_factorial_contrasts_follow_frozen_2x2_order():
    by_arm = {
        "S": {"auroc": 0.40}, "SC": {"auroc": 0.50},
        "G": {"auroc": 0.60}, "GC": {"auroc": 0.90},
    }
    result = r3._factorial_contrasts(by_arm, "auroc")
    assert result == {"GC_minus_G": pytest.approx(0.30),
                      "GC_minus_SC": pytest.approx(0.40),
                      "SC_minus_S": pytest.approx(0.10),
                      "G_minus_S": pytest.approx(0.20),
                      "interaction": pytest.approx(0.20)}


def test_transition_ledger_reconstructs_confusion_matrix():
    labels = np.array([0, 0, 1, 1, 1])
    base = np.array([0, 1, 0, 1, 0])
    candidate = np.array([1, 0, 1, 1, 0])
    result = r3._confusion_transitions(base, candidate, labels)
    assert result["reconstruction_verified"] is True
    assert result["FN_to_TP_repaired"] == 1
    assert result["FP_to_TN_repaired"] == 1
    assert result["TN_to_FP_introduced"] == 1
    assert result["TP_to_FN_introduced"] == 0
    assert result["candidate_confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}


def test_transition_ledger_detects_misaligned_prediction_shapes():
    with pytest.raises((r3.R3Error, ValueError, IndexError)):
        r3._confusion_transitions(np.array([0]), np.array([0, 1]), np.array([0]))


def test_source_group_pair_diagnostics_report_disjoint_denominator():
    result = r3._pair_credit_changes(
        np.array([0.1, 0.9, 0.2, 0.8]),
        np.array([0.2, 0.8, 0.3, 0.7]),
        np.array([0, 1, 0, 1]),
        np.array(["a", "a", "b", "c"]))
    assert result["all_pairs"]["normal_defect_pair_denominator"] == 4
    assert result["source_group_disjoint"]["normal_defect_pair_denominator"] == 3
    assert result["all_pairs"]["ties_weighted_as_half"] is True


def test_paired_source_group_bootstrap_is_deterministic_and_shared():
    scores = {
        "S": np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9]),
        "SC": np.array([0.2, 0.3, 0.6, 0.9, 0.4, 0.8]),
        "G": np.array([0.1, 0.4, 0.8, 0.7, 0.2, 0.9]),
        "GC": np.array([0.3, 0.4, 0.8, 0.9, 0.5, 1.0]),
    }
    labels = np.array([0, 0, 1, 1, 0, 1])
    groups = np.array(["g1", "g2", "g3", "g4", "g5", "g6"])
    first = r3._paired_group_bootstrap(scores, labels, groups, 0.5,
                                       replicates=80, seed=311)
    second = r3._paired_group_bootstrap(scores, labels, groups, 0.5,
                                        replicates=80, seed=311)
    assert first == second
    assert first["same_sampled_groups_for_all_arms"] is True
    assert first["factorial_interaction_computed_inside_each_replicate"] is True
    assert first["contrasts"]["auroc"]["interaction"]["valid_replicates"] > 0


def test_run_namespace_is_new_only_and_rejects_existing_path(tmp_path, monkeypatch):
    monkeypatch.setattr(r3, "OUTPUT_BASE", tmp_path)
    run_id = "r3_fullstream_2x2_test_unique"
    target = r3._output_path(run_id)
    target.mkdir()
    with pytest.raises(FileExistsError, match="Refusing output reuse"):
        r3._output_path(run_id)
    assert r3._existing_output_path(run_id) == target


def test_stage_b_cannot_start_without_full_stage_a_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(r3, "_existing_output_path", lambda _run_id: tmp_path)
    monkeypatch.setattr(r3, "_verify_source_binding", lambda _manifest: None)
    import torch
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"mode": "FULL_STAGE_A"}), encoding="utf-8")
    (tmp_path / "stage_a_scores.jsonl").write_text("", encoding="utf-8")
    called = []

    class ForbiddenLabelReader:
        def __init__(self, *_args, **_kwargs):
            called.append("opened")
            raise AssertionError("label reader must not open before score seal")

    import experiments.calibration.stage1a_p12_label_provider as provider
    monkeypatch.setattr(provider, "_RestrictedTestArray", ForbiddenLabelReader)
    with pytest.raises(r3.R3Error, match="requires a completed"):
        r3._run_stage_b_child("r3_fullstream_2x2_test_unique")
    assert called == []


def _synthetic_stage_a(tmp_path, monkeypatch, missing_category=None):
    categories = r3.CATEGORIES
    counts = {category: 4 for category in categories}
    monkeypatch.setattr(r3, "EXPECTED_COUNTS", counts)
    root = tmp_path / "synthetic_run"
    budget = OutputBudget(root, create=True, max_bytes=1024 * 1024,
                          failure_reserve_bytes=4096, max_files=12)
    category_meta = {}
    rows = []
    history_rows = []
    category_score_rows = {category: [] for category in categories}
    query_receipt_rows = {category: [] for category in categories}
    for category in categories:
        query_positions = [1]
        category_meta[category] = {
            "query_positions": query_positions,
            "query_tape_sha256": hashlib.sha256(category.encode()).hexdigest(),
            "query_positions_sha256": r3.sha256_json(query_positions),
        }
        for position in range(4):
            row = {"schema": r3.SCORE_SCHEMA, "category": category,
                   "position": position, "query_tape_decision": position in query_positions,
                   "event_id_sha256": "e" * 64,
                   "source_group_sha256": "g" * 64,
                   "predictions_at_frozen_threshold": {arm: 1 for arm in r3.ARMS},
                   "scores": {arm: float(position) for arm in r3.ARMS}}
            rows.append(row)
            category_score_rows[category].append(r3.canonical_bytes(row))
        feedback = {"schema": r3.HISTORY_SCHEMA,
                    "record_type": "queried_feedback_and_updates",
                    "category": category, "position": 1,
                    "event_id_sha256": "e" * 64,
                    "source_group_sha256": "g" * 64,
                    "label": 0, "disclosure_after_four_score_seal": True,
                    "correction_updates": {
                        "SC": {"base_score": 1.0, "base_prediction": 1, "outcome": "FP"},
                        "GC": {"base_score": 1.0, "base_prediction": 1, "outcome": "FP"}},
                    "greedy_writer_action": {"input_score_source": "common_C0",
                                              "prediction": 1, "action": "native_gate_no_write"}}
        history_rows.append(feedback)
        query_receipt_rows[category].append(r3.canonical_bytes(feedback))
    budget.write_once("stage_a_scores.jsonl", b"".join(r3.canonical_bytes(x) for x in rows))
    for row in history_rows:
        r3._append_history(budget, row)
    for category in categories:
        if category == missing_category:
            continue
        category_receipt = {
            "schema": r3.HISTORY_SCHEMA, "record_type": "category_complete",
            "category": category, "events_processed": 4,
            "score_rows_sha256": hashlib.sha256(
                b"".join(category_score_rows[category])).hexdigest(),
            "query_receipts_sha256": hashlib.sha256(
                b"".join(query_receipt_rows[category])).hexdigest(),
            "fixed_support_sha256": "f" * 64,
            "first512_exp2a_parity": {"status": "SYNTHETIC"},
            "consistency_checks": {}, "actual_stream_image_encodings": 4,
            "greedy_write_events": 0,
            "correction_final_occupancy": {"SC": 1, "GC": 1},
            "correction_stats": {},
        }
        r3._append_history(budget, category_receipt)
    return root, {"classes": category_meta}


def test_stage_a_seal_requires_all_categories_and_exact_event_order(tmp_path, monkeypatch):
    root, contract = _synthetic_stage_a(tmp_path, monkeypatch)
    seal = r3._seal_stage_a(root, contract)
    assert seal["status"] == "SEALED_COMPLETE"
    assert seal["event_count"] == 16
    assert seal["query_count"] == 4
    assert r3.sha256_file(root / "stage_a_scores.jsonl") == seal["score_rows_sha256"]


def test_stage_a_refuses_missing_category_receipt(tmp_path, monkeypatch):
    root, contract = _synthetic_stage_a(
        tmp_path, monkeypatch, missing_category="audiojack")
    with pytest.raises(r3.R3Error, match="all four completed category receipts"):
        r3._seal_stage_a(root, contract)
    assert not (root / "stage_a_seal.json").exists()
