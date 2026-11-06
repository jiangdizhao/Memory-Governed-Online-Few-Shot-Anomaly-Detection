from __future__ import annotations

import json
import numpy as np
import pytest

from experiments.calibration.a_calibration_continuation import (
    B3State,
    ContinuationError,
    _b3_stability_metrics,
    _integrated_performance_summaries,
    _reuse_or_write_integrated_aggregate,
    _append_bounded_private_jsonl,
    _record_static_c64_prediction,
    _condition_seal_is_valid,
    _private_ledger_path,
    _IndexedFeatureCache,
    _load_jsonl,
    _nonoverlap_window_metrics,
    _public_a3b_receipt,
    _runner_coverage_audit,
    _validate_resume_identity,
    _experiment_contract,
    _file_fingerprints,
    _json_bytes,
    _sha_bytes,
    _sha_file,
    PROTOCOL_ID,
    AUTHORIZATION_ID,
    SOURCE_PARENT,
    a3b_kcenter_cuda,
    a3b_candidate_identity,
    a3b_candidate_quotas,
    a3b_kcenter_numpy_reference,
    higher_q90,
)
from experiments.calibration import r1_matched_correction_pilot as r1
from experiments.calibration import r7_shuffled_correction_4class as r7


def test_higher_q90_uses_registered_higher_order_statistic():
    for values in ([1.0, 2.0], [0.1, 0.2, 0.3, 0.4, 0.5],
                   list(range(64)), list(range(128))):
        assert higher_q90(values) == float(np.quantile(values, 0.9, method="higher"))


def test_higher_q90_rejects_invalid_inputs():
    for values in ([], [1.0, np.nan], [[1.0, 2.0]]):
        with pytest.raises(ValueError):
            higher_q90(values)


def test_b3_initializes_sample_scale_and_refreshes_once_per_version():
    state = B3State.initialize([0.0, 1.0, 2.0, 3.0])
    assert state.theta == 3.0
    assert state.sigma == pytest.approx(np.std([0.0, 1.0, 2.0, 3.0], ddof=1))
    assert state.refresh([10.0, 11.0, 12.0, 13.0], version=1)
    assert state.theta == pytest.approx(3.01)
    theta, sigma = state.theta, state.sigma
    assert not state.refresh([0.0, 0.0, 0.0, 0.0], version=1)
    assert (state.theta, state.sigma) == (theta, sigma)
    with pytest.raises(ContinuationError):
        state.refresh([0.0, 1.0], version=0)


def test_b3_threshold_and_scale_are_separate_updates():
    state = B3State.initialize([0.0, 0.0, 0.0, 1.0])
    initial = state.theta
    assert state.refresh([0.1, 0.1, 0.1, 0.1], version=1)
    assert state.theta == pytest.approx(initial - 0.01)
    assert state.sigma == 0.02


def test_static_reference_decision_is_recorded_in_event_decisions():
    from types import SimpleNamespace
    state = B3State.initialize([0.0, 0.1, 0.2, 0.3])
    trajectory = SimpleNamespace(name="F4-static", g_digest="abc", g_version=0)
    decisions = {}
    _record_static_c64_prediction(decisions, trajectory, state, score=1.0)
    row = decisions["F4-static:STATIC:C64"]
    assert row["prediction"] == 1
    assert row["scorer_state_version"] == 0
    assert row["scorer_state_sha256"] == "abc"
    assert row["threshold"] == state.theta


def test_a3b_candidate_quotas_are_balanced_bounded_and_deterministic():
    groups = [f"g{i}" for i in range(7)]
    counts = {g: 10_000 for g in groups}
    first = a3b_candidate_quotas(groups, 31, counts, "cat-v1")
    second = a3b_candidate_quotas(list(reversed(groups)), 31, counts, "cat-v1")
    assert first == second
    assert sum(first.values()) == 31
    assert set(first.values()) == {4, 5}
    capped = a3b_candidate_quotas(groups, 31, {**counts, "g0": 1}, "cat-v1")
    assert capped["g0"] == 1
    assert sum(capped.values()) == 31


def test_a3b_candidate_identity_is_namespaced_and_stable():
    a = a3b_candidate_identity("cat", "group", "image", 0, 7)
    assert a == a3b_candidate_identity("cat", "group", "image", 0, 7)
    assert len(a) == 64
    assert a != a3b_candidate_identity("cat2", "group", "image", 0, 7)


def test_kcenter_first_center_and_ties_use_identity_hash_order():
    # Orthogonal candidates tie after the antipodal pair has been selected.
    x = np.asarray([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float64)
    identities = ["f" * 64, "a" * 64, "c" * 64, "b" * 64]
    selected = a3b_kcenter_numpy_reference(x, identities, 3)
    assert selected[0] == 1  # minimum identity hash, regardless of row order
    assert selected[1] == 0
    assert selected[2] == 3  # tied distance resolved by minimum identity hash
    assert len(set(selected)) == 3


def test_kcenter_rejects_zero_descriptors_and_bad_budgets():
    with pytest.raises(ValueError):
        a3b_kcenter_numpy_reference(np.zeros((2, 3)), ["a", "b"], 1)
    with pytest.raises(ValueError):
        a3b_kcenter_numpy_reference(np.ones((2, 3)), ["a"], 1)


def test_cuda_kcenter_matches_numpy_reference_and_identity_ties():
    if not __import__("torch").cuda.is_available():
        pytest.skip("registered CUDA implementation requires CUDA")
    candidates = np.asarray([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                             [0, -1, 0], [0, 0, 1]], dtype=np.float16)
    identities = ["f" * 64, "a" * 64, "c" * 64, "b" * 64, "d" * 64]
    expected = a3b_kcenter_numpy_reference(candidates, identities, 4)
    prototypes, selected = a3b_kcenter_cuda(
        candidates, identities, 4, block_rows=2, return_indices=True)
    assert selected == expected
    assert tuple(prototypes.shape) == (4, 3)
    assert str(prototypes.dtype) == "torch.float16"


def test_score_disclosure_gate_fsync_callback_precedes_query_label():
    events = []
    gate = r7.ScoreDisclosureGate(lambda row: events.append(("fsync", dict(row))))
    row = {"position": 0, "scores": {"G": 0.1}}
    gate.seal(0, row)
    causal = r1.CausalEventOrder(0)
    causal.seal_scores()
    causal.decide_query(True)
    label = causal.disclose(lambda: events.append(("label", 0)) or 0)
    assert label == 0
    assert [event[0] for event in events] == ["fsync", "label"]
    assert gate.is_sealed(0)
    assert "label" not in events[0][1]


def test_score_gate_rejects_unqueried_label_fields_and_out_of_order_rows():
    gate = r7.ScoreDisclosureGate(lambda _row: None)
    with pytest.raises(Exception):
        gate.seal(0, {"position": 0, "label": 0})
    with pytest.raises(Exception):
        gate.seal(1, {"position": 1, "scores": {}})


def test_unqueried_causal_event_cannot_disclose_any_label():
    causal = r1.CausalEventOrder(4)
    causal.seal_scores()
    causal.decide_query(False)
    with pytest.raises(r1.R1CausalityError):
        causal.disclose(lambda: pytest.fail("unqueried test label was read"))
    causal.corrections_updated()
    causal.memory_updated()
    causal.finish()


def test_native_writer_admits_only_queried_verified_normal_feedback():
    torch = pytest.importorskip("torch")
    calls = []
    def eligible(**_kwargs):
        return True, 32, "synthetic", {}
    def update(**kwargs):
        calls.append(kwargs["position"])
        return {"accepted_patches_delta": 1}
    kwargs = {"c0_score": 0.1, "theta": 0.2, "sigma": 0.02,
        "c0_prediction": 0, "candidate_ids": np.arange(32, dtype=np.int64),
        "candidate_features": np.zeros((32, 4), dtype=np.float32),
        "feature_layers": [torch.zeros((784, 4), dtype=torch.float32)],
        "eligibility": eligible, "update": update}
    defect = r1.common_c0_greedy_writer_action(
        position=0, queried=True, disclosed_label=1, **kwargs)
    unqueried = r1.common_c0_greedy_writer_action(
        position=1, queried=False, disclosed_label=None, **kwargs)
    normal = r1.common_c0_greedy_writer_action(
        position=2, queried=True, disclosed_label=0, **kwargs)
    assert defect["action"] == "verified_defect_no_normal_write"
    assert unqueried["action"] == "no_normal_memory_write"
    assert normal["action"] == "native_greedy_normal_update"
    assert calls == [2]


def test_stage_a_score_tape_rejects_labels_and_unsealed_jsonl(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps({"position": 0, "label": 1}) + "\n", encoding="utf-8")
    with pytest.raises(ContinuationError, match="contains a label"):
        _load_jsonl(path)
    path.write_text('{"position":0}', encoding="utf-8")
    with pytest.raises(ContinuationError, match="unsealed"):
        _load_jsonl(path)


def test_private_feedback_ledger_is_fsynced_and_byte_bounded(tmp_path):
    path = tmp_path / "stage_a_feedback_private.jsonl"
    _append_bounded_private_jsonl(path, {"position": 0, "label": 1},
                                  maximum_bytes=64)
    assert path.read_text(encoding="utf-8") == '{"label":1,"position":0}\n'
    with pytest.raises(ContinuationError, match="size cap"):
        _append_bounded_private_jsonl(path, {"payload": "x" * 100}, maximum_bytes=64)


def test_integrated_report_summaries_are_ready_before_aggregate_serialization():
    arms = {
        "A3b-F4-STATIC:STATIC:C64": 0.1,
        "A3b-FEXP-KC-STATIC:STATIC:C64": 0.2,
        "A3b-F4:G": 0.3,
        "A3b-FEXP-KC:G": 0.4,
        "P0-A3a-C64:G": 0.5,
        "P0-A3a-C128:G": 0.6,
    }
    decision_metrics = {
        arm: {"ber": ber, "fpr": ber / 2, "fnr": ber / 2,
              "ranking": {"auroc": 1 - ber, "average_precision": 0.9 - ber}}
        for arm, ber in arms.items()}
    metrics = {
        "categories": {"synthetic": {"P0": {"decision_metrics": decision_metrics}}},
        "resource_diagnostics": {
            "synthetic": {
                "test_feature_cache": {"encoding_seconds": 2.0},
                "a3b_reference_construction": {
                    "candidate_feature_extraction_seconds": 1.0,
                    "candidate_selection_seconds": 2.0,
                    "native_reference_bank_build_seconds": 3.0,
                    "kcenter_seconds": 4.0,
                },
                "conditions": {"P0": {"cmax_G_resource_profile": {
                    "measured_full_reference_rescore_seconds": 5.0}}},
            }
        },
    }

    summary = _integrated_performance_summaries(metrics, categories=("synthetic",))

    assert summary["a3a_g_ber_means_p0"] == {"C64": 0.5, "C128": 0.6}
    assert summary["a3b_means_p0"]["F4_static"]["ber"] == 0.1
    assert summary["a3b_means_p0"]["FEXP_KC_prequential"]["auroc"] == pytest.approx(0.6)
    assert summary["mean_test_encoding_seconds_per_category"] == 2.0
    assert summary["mean_a3b_reference_construction_seconds_per_category"] == 10.0
    assert summary["mean_measured_cmax_rescore_profile_seconds"] == 5.0


def test_integrated_aggregate_reuse_preserves_existing_bytes_and_rejects_metric_changes(tmp_path):
    path = tmp_path / "aggregate.json"
    original = {"value": 1, "report_assembly_sha256": "a" * 64}
    original_bytes = _json_bytes(original) + b"\n"
    path.write_bytes(original_bytes)

    reused = _reuse_or_write_integrated_aggregate(
        path, {"value": 1, "report_assembly_sha256": "b" * 64})

    assert reused == original_bytes
    assert path.read_bytes() == original_bytes
    with pytest.raises(ContinuationError, match="differs from sealed metrics"):
        _reuse_or_write_integrated_aggregate(
            path, {"value": 2, "report_assembly_sha256": "b" * 64})


def test_condition_seal_requires_complete_status_and_matching_checksums(tmp_path):
    condition = tmp_path / "P0"
    condition.mkdir()
    scores = condition / "stage_a_scores.jsonl"
    feedback = _private_ledger_path(condition, "feedback")
    feedback.parent.mkdir()
    scores.write_text("{}\n", encoding="utf-8")
    feedback.write_text("{}\n", encoding="utf-8")
    from experiments.calibration.a_calibration_continuation import _sha_file
    seal = {"status": "SEALED_COMPLETE", "score_rows_sha256": _sha_file(scores),
            "feedback_rows_sha256": _sha_file(feedback)}
    (condition / "stage_a_seal.json").write_text(json.dumps(seal), encoding="utf-8")
    assert _condition_seal_is_valid(condition)
    assert not _condition_seal_is_valid(condition, allow_smoke=True)
    scores.write_text("changed\n", encoding="utf-8")
    assert not _condition_seal_is_valid(condition)


def test_smoke_condition_is_never_accepted_as_a_full_seal(tmp_path):
    condition = tmp_path / "P0"
    condition.mkdir()
    scores = condition / "stage_a_scores.jsonl"
    feedback = _private_ledger_path(condition, "feedback")
    feedback.parent.mkdir()
    scores.write_text("{}\n", encoding="utf-8")
    feedback.write_text("{}\n", encoding="utf-8")
    from experiments.calibration.a_calibration_continuation import _sha_file
    seal = {"status": "INCOMPLETE", "score_rows_sha256": _sha_file(scores),
            "feedback_rows_sha256": _sha_file(feedback)}
    (condition / "stage_a_seal.json").write_text(json.dumps(seal), encoding="utf-8")
    assert _condition_seal_is_valid(condition, allow_smoke=True)
    assert not _condition_seal_is_valid(condition)


def test_fixed_nonoverlapping_windows_exclude_partial_and_single_class_windows():
    labels = np.asarray([0, 1] * 256 + [0] * 512 + [0, 1], dtype=np.int8)
    predictions = np.zeros_like(labels)
    result = _nonoverlap_window_metrics(labels, predictions, window=512)
    assert result["complete_windows"] == 2
    assert result["valid_windows"] == 1
    assert result["windows"][1]["status"] == "INVALID_SINGLE_CLASS_WINDOW"
    assert result["worst_valid_window"]["start"] == 0


def test_b3_stability_summary_reports_variation_not_duplicate_versions():
    seal = {"final_b3_history": {"A2": {"G:C64": [
        {"version": 0, "theta": 1.0, "sigma": 0.3},
        {"version": 1, "theta": 1.01, "sigma": 0.4},
        {"version": 2, "theta": 1.01, "sigma": 0.2},
    ]}}}
    summary = _b3_stability_metrics(seal)["A2"]["G:C64"]
    assert summary["threshold_total_variation"] == pytest.approx(0.01)
    assert summary["effective_threshold_changes"] == 1
    assert summary["threshold_update_count"] == 2
    assert summary["state_version_count"] == 3


def test_public_a3b_receipt_redacts_group_and_raw_score_identities():
    receipt = {"reference_pool_groups": 145,
        "candidate_count_by_group": {"private-group-sha": 452},
        "candidate_identity_sha256": "private-candidate-hash",
        "native_score_read_compatibility": {"F4": {
            "adapter_score": 0.123, "native_bridge_score": 0.123,
            "absolute_difference": 0.0, "passed": True,
            "tolerance": {"atol": 1e-5, "rtol": 1e-4}}}}
    public = _public_a3b_receipt(receipt)
    assert public["reference_pool_groups"] == 145
    assert "candidate_count_by_group" not in public
    assert "candidate_identity_sha256" not in public
    assert public["native_score_read_compatibility"]["F4"] == {
        "passed": True, "absolute_difference": 0.0,
        "tolerance": {"atol": 1e-5, "rtol": 1e-4}}


def test_compact_feature_cache_uses_original_json_position():
    rows = np.arange(12).reshape(3, 4)
    cache = _IndexedFeatureCache(rows, [29, 4, 17])
    assert np.array_equal(cache[29], rows[0])
    assert np.array_equal(cache[4], rows[1])
    assert np.array_equal(cache[17], rows[2])
    with pytest.raises(ContinuationError, match="does not contain"):
        cache[99]


def test_resume_identity_requires_exact_source_and_contract_hashes():
    current = {"branch": "calibration-stage0-contract-20261029",
               "head": SOURCE_PARENT}
    preflight = {"git": current, "source_fingerprints": _file_fingerprints()}
    manifest = {"run_id": "a23_20261106_test01", "git": current,
        "runner_sha256": _sha_file(__import__(
            "experiments.calibration.a_calibration_continuation",
            fromlist=["__file__"]).__file__),
        "experiment_contract_sha256": _sha_bytes(_json_bytes(_experiment_contract())),
        "source_fingerprints": preflight["source_fingerprints"],
        "protocol_id": PROTOCOL_ID, "authorization_id": AUTHORIZATION_ID}
    _validate_resume_identity(manifest, "a23_20261106_test01", preflight)
    manifest["source_fingerprints"] = {"changed": "sha256"}
    with pytest.raises(ContinuationError, match="fingerprints"):
        _validate_resume_identity(manifest, "a23_20261106_test01", preflight)


def test_retained_output_budget_excludes_transient_scratch(tmp_path):
    from experiments.calibration.a_calibration_continuation import _retained_output_bytes
    (tmp_path / "stage_a").mkdir()
    (tmp_path / "stage_a" / "score.jsonl").write_bytes(b"small")
    (tmp_path / ".scratch" / "audiojack").mkdir(parents=True)
    (tmp_path / ".scratch" / "audiojack" / "features.npy").write_bytes(b"x" * 1024)
    assert _retained_output_bytes(tmp_path) == 5


def test_coverage_audit_contains_every_registered_scientific_branch():
    audit = _runner_coverage_audit()
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert audit["condition_matrix"]["A2_P1_P2"] == ["G", "GC", "C64", "C128"]


@pytest.mark.parametrize("label", [0, 1])
def test_static_c64_prediction_is_a_valid_binary_b3_readout(label):
    state = B3State.initialize([0.1, 0.2, 0.3, 0.4])
    score = 0.5 if label else 0.1
    from experiments.calibration.a_calibration_continuation import _b3_prediction
    prediction = _b3_prediction(state, score=score, state_digest="f" * 64,
                                state_version=0)
    assert prediction["prediction"] == label
