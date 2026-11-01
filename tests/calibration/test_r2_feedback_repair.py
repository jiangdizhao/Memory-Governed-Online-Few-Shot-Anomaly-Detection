from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys

import numpy as np
import pytest
from types import SimpleNamespace
import torch

from experiments.calibration import r2_feedback_repair as repair


torch.set_num_threads(1)
GROUP_A = "a" * 64
GROUP_B = "b" * 64
GROUP_C = "c" * 64


def _banks(capacity: int = 8, dimension: int = 4):
    import test_phase3_AIF_simplified_v22 as v22
    return {
        name: repair._make_bank(name, v22, capacity=capacity, feature_dim=dimension)
        for name in ("G_N", "G_A", "G_NA")
    }


def _unit_features(count: int = 4, dimension: int = 4):
    result = np.zeros((count, dimension), dtype=np.float32)
    for index in range(count):
        result[index, index % dimension] = 1.0
    return result


def test_native_gaussian_readout_matches_signed_codebook_apply():
    import test_phase3_AIF_simplified_v22 as v22
    bank = repair._make_bank("G_NA", v22, capacity=4, feature_dim=4)
    features = torch.from_numpy(_unit_features(3, 4))
    beta = torch.tensor([0.1, -0.12, 0.08], dtype=torch.float32)
    bank.codebook.update(features, beta, outcome="TP", step=0)
    query = torch.from_numpy(_unit_features(2, 4))
    distances = 1.0 - torch.clamp(query @ bank.codebook.protos[:3].t(), -1.0, 1.0)
    actual, pos, neg = repair._gaussian_readout(
        distances, bank.codebook.beta[:3], bank.codebook.radius[:3],
        bank.codebook.kappa[:3], components=True)
    expected = bank.codebook.apply(query)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert torch.allclose(actual, pos + neg, atol=1e-7, rtol=1e-6)


def test_compact_keep_beta_alignment_and_class_conditional_banks():
    banks = _banks()
    assert len({id(bank.codebook) for bank in banks.values()}) == 3
    features = _unit_features(4, 4)
    keep = np.array([True, False, True, False])
    beta_normal = np.array([-0.1, -0.12], dtype=np.float32)
    before = repair._apply_class_conditional_feedback(
        banks, features, keep, beta_normal, label=0, outcome="TN",
        position=1, group=GROUP_A, queried=True, score_sealed=True)
    assert before["G_N"] is not None and before["G_NA"] is not None
    assert before["G_A"] is None
    assert banks["G_N"].occupancy == banks["G_NA"].occupancy == 2
    assert banks["G_A"].occupancy == 0
    normal_state = banks["G_N"].codebook.protos[:2].clone()

    beta_anomaly = np.array([0.11, 0.13], dtype=np.float32)
    after = repair._apply_class_conditional_feedback(
        banks, features, keep, beta_anomaly, label=1, outcome="FN",
        position=2, group=GROUP_B, queried=True, score_sealed=True)
    assert after["G_N"] is None and after["G_A"] is not None and after["G_NA"] is not None
    assert banks["G_N"].occupancy == 2
    assert banks["G_A"].occupancy == 2
    assert banks["G_NA"].occupancy == 4
    assert torch.equal(banks["G_N"].codebook.protos[:2], normal_state)


def test_update_fails_before_durable_score_and_unqueried_feedback_does_not_write():
    banks = _banks()
    features = _unit_features(4, 4)
    keep = np.array([True, False, True, False])
    beta = np.array([-0.1, -0.12], dtype=np.float32)
    with pytest.raises(repair.RepairError, match="before the score"):
        repair._apply_class_conditional_feedback(
            banks, features, keep, beta, label=0, outcome="TN",
            position=0, group=GROUP_A, queried=True, score_sealed=False)
    assert all(bank.occupancy == 0 for bank in banks.values())
    result = repair._apply_class_conditional_feedback(
        banks, features, keep, beta, label=0, outcome="TN",
        position=0, group=GROUP_A, queried=False, score_sealed=False)
    assert all(value is None for value in result.values())
    assert all(bank.occupancy == 0 for bank in banks.values())


def test_diagnostic_identity_uses_category_position_and_group_receipt():
    current = {"category": "switch", "position": 35,
               "source_group_sha256": "group-hash"}
    assert repair._diagnostic_identity_matches(
        current, category="switch", position=35, source_group="group-hash")
    assert not repair._diagnostic_identity_matches(
        current, category="switch", position=36, source_group="group-hash")
    assert not repair._diagnostic_identity_matches(
        current, category="switch", position=35, source_group="other-group")


def test_same_source_exclusion_is_conservative_and_temporal():
    import test_phase3_AIF_simplified_v22 as v22
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r2_exp2b_memory_readout as exp2b
    bank = repair._make_bank("G_NA", v22, capacity=4, feature_dim=4)
    features = _unit_features(4, 4)
    keep = np.array([True, False, True, False])
    beta = np.array([-0.1, -0.12], dtype=np.float32)
    repair._apply_class_conditional_feedback(
        {"G_NA": bank}, features, keep, beta, label=0, outcome="TN",
        position=1, group=GROUP_A, queried=True, score_sealed=True)
    query = torch.from_numpy(_unit_features(4, 4))
    same_delta, _, same_detail = repair._score_bank(
        v22, r1, exp2b, bank, query, GROUP_A, 2)
    other_delta, _, other_detail = repair._score_bank(
        v22, r1, exp2b, bank, query, GROUP_B, 2)
    assert same_detail["eligible"] == 0 and same_detail["same_source_excluded"] == 2
    assert torch.count_nonzero(same_delta) == 0
    assert other_detail["eligible"] == 2
    assert torch.any(other_delta < 0)
    with pytest.raises(Exception, match="current/future"):
        r1._validate_c1_source_groups(
            bank.source_groups, bank.codebook.last_step[:2].tolist(),
            event_position=1, phase="pre_score", event_group=GROUP_B)


def test_global_control_uses_eligible_bank_scalar_without_a_writer():
    import test_phase3_AIF_simplified_v22 as v22
    bank = repair._make_bank("G_NA", v22, capacity=4, feature_dim=4)
    features = _unit_features(4, 4)
    keep = np.array([True, False, True, False])
    repair._apply_feedback_update(bank, features, keep,
                                  np.array([-0.1, -0.12], dtype=np.float32),
                                  label=0, outcome="TN", position=1, group=GROUP_A)
    repair._apply_feedback_update(bank, features, keep,
                                  np.array([0.1, 0.12], dtype=np.float32),
                                  label=1, outcome="FN", position=2, group=GROUP_B)
    count = bank.occupancy
    expected = float((bank.codebook.kappa[:count] * bank.codebook.beta[:count]).sum().item() /
                     (1e-6 + bank.codebook.kappa[:count].sum().item()))
    state_before = bank.codebook.protos[:count].clone()
    actual = repair._global_delta(bank, list(range(count)))
    assert actual == pytest.approx(expected, abs=1e-8)
    assert bank.occupancy == count and torch.equal(bank.codebook.protos[:count], state_before)


def test_positive_negative_components_keep_the_full_shared_denominator():
    distances = torch.zeros((2, 2), dtype=torch.float32)
    beta = torch.tensor([0.2, -0.1], dtype=torch.float32)
    radius = torch.ones(2, dtype=torch.float32)
    kappa = torch.ones(2, dtype=torch.float32)
    full, positive, negative = repair._gaussian_readout(
        distances, beta, radius, kappa, components=True)
    assert torch.allclose(full, positive + negative, atol=1e-7, rtol=1e-6)
    assert positive[0].item() == pytest.approx(0.2 / 2.000001, abs=1e-7)
    assert negative[0].item() == pytest.approx(-0.1 / 2.000001, abs=1e-7)
    assert full[0].item() == pytest.approx(0.1 / 2.000001, abs=1e-7)


def test_dense_map_maximum_is_recomputed_after_patch_correction():
    from experiments.calibration import r1_matched_correction_pilot as r1
    base = np.zeros(784, dtype=np.float32)
    base[0] = 0.5
    base[783] = 0.7
    ids = np.arange(32, dtype=np.int64)
    negative = np.full(32, -0.2, dtype=np.float32)
    positive = np.full(32, 0.2, dtype=np.float32)
    assert repair._map_score(base, ids, negative, r1) == pytest.approx(0.7)
    assert repair._map_score(base, ids, positive, r1) == pytest.approx(0.7)


def test_one_sign_score_monotonicity_holds_after_native_readout():
    from experiments.calibration import r1_matched_correction_pilot as r1
    base = np.linspace(0.1, 0.4, 784, dtype=np.float32)
    ids = np.arange(32, dtype=np.int64)
    g = r1.max_score(base)
    assert repair._map_score(base, ids, np.full(32, -0.05, dtype=np.float32), r1) <= g
    assert repair._map_score(base, ids, np.full(32, 0.05, dtype=np.float32), r1) >= g


def test_repair_ledger_conserves_confusion_and_undefined_denominators():
    labels = np.array([1, 1, 0, 0], dtype=np.int8)
    scores = {
        "G": np.array([0.1, 0.9, 0.9, 0.1]),
        "G_NA": np.array([0.9, 0.1, 0.1, 0.9]),
    }
    result = repair._repair_ledger(labels, scores, 0.5)["G_NA"]
    assert result["fn_to_tp_repaired"] == 1
    assert result["tp_to_fn_introduced"] == 1
    assert result["fp_to_tn_repaired"] == 1
    assert result["tn_to_fp_introduced"] == 1
    assert result["confusion_reconstruction_verified"] is True
    only_normal = repair._repair_ledger(
        np.array([0, 0]), {"G": np.array([0.1, 0.1]),
                           "G_NA": np.array([0.1, 0.9])}, 0.5)["G_NA"]
    assert only_normal["rates"]["fn_repair_among_reference_fn"] is None
    assert only_normal["rates"]["new_fn_among_reference_tp"] is None


def test_ties_and_cross_source_group_pair_denominator_are_explicit():
    labels = np.array([1, 0, 1, 0], dtype=np.int8)
    groups = [GROUP_A, GROUP_A, GROUP_B, GROUP_C]
    base = np.ones(4, dtype=np.float32)
    candidate = np.ones(4, dtype=np.float32)
    record = repair._pair_accounting(labels, groups, base, candidate)
    assert record["defined"] is True
    assert record["all_pairs"]["pair_count"] == 4
    assert record["all_pairs"]["reference_auc"] == pytest.approx(0.5)
    assert record["all_pairs"]["net_auc_difference"] == pytest.approx(0.0)
    assert record["cross_source_group_pairs"]["pair_count"] == 3
    assert record["cross_source_group_denominator_is_own_pair_count"] is True
    normal_only = repair._pair_accounting(
        np.zeros(3, dtype=np.int8), [GROUP_A, GROUP_B, GROUP_C],
        np.arange(3), np.arange(3))
    assert normal_only["defined"] is False


def test_unqueried_subset_mask_and_normal_only_ranking_state():
    rows = [{"query_tape_decision": False}, {"query_tape_decision": True},
            {"query_tape_decision": False}]
    assert repair._unqueried_mask(rows).tolist() == [True, False, True]
    from experiments.calibration import r2_perf1_stage_b as perf
    pack = repair._metric_pack(perf, np.zeros(3, dtype=np.int8),
                               np.array([0.1, 0.2, 0.3]), 0.2)
    assert pack["auroc"] is None and pack["average_precision"] is None
    assert pack["fixed_threshold"]["fnr"] is None


def test_stage_b_refuses_to_project_outcomes_without_a_complete_score_seal(tmp_path):
    root = tmp_path / "r2_feedback_repair_20261102_a1"
    budget = repair.OutputBudget(root, create=True)
    budget.write_json("run_manifest.json", {
        "schema": repair.MANIFEST_SCHEMA, "run_id": root.name,
        "stage_a_source_run_id": root.name,
        "stage_a_algorithm_sha256": "0" * 64,
    })
    with pytest.raises(Exception, match="stage_a_seal.json"):
        repair._stage_b_results({}, root, budget)


def test_output_budget_prevents_overrun_overwrite_and_concurrent_mutation(tmp_path):
    root = tmp_path / "bounded"
    budget = repair.OutputBudget(root, create=True, max_bytes=16,
                                 failure_reserve_bytes=4, max_files=4)
    budget.write_once("stage_a_scores.jsonl", b"x" * 10)
    with pytest.raises(repair.RepairError, match="before write"):
        budget.append("stage_a_scores.jsonl", b"yy")
    with pytest.raises(repair.RepairError, match="overwrite"):
        budget.write_once("stage_a_scores.jsonl", b"z")
    budget.write_once("failure.json", b"1234", failure_reserved=True)
    audit = budget.audit()
    assert audit["bytes"] <= audit["budget_bytes"] == 16

    other_root = tmp_path / "concurrent"
    first = repair.OutputBudget(other_root, create=True, max_bytes=128,
                                failure_reserve_bytes=8, max_files=5)
    other = repair.OutputBudget(other_root, max_bytes=128,
                                failure_reserve_bytes=8, max_files=5)
    first.write_once("stage_a_scores.jsonl", b"changed")
    # Another budget instance holding a stale inventory must fail before writing.
    with pytest.raises(repair.RepairError, match="Concurrent or unaccounted"):
        other.write_once("complete_seal.json", b"not allowed")

    jsonl_root = tmp_path / "jsonl"
    jsonl_budget = repair.OutputBudget(jsonl_root, create=True, max_bytes=256,
                                       failure_reserve_bytes=16, max_files=6)
    jsonl_budget.append_jsonl("attempt_history.jsonl", {"phase": "STARTED"})
    jsonl_budget.append_jsonl("attempt_history.jsonl", {"phase": "FAILED"})
    assert len((jsonl_root / "attempt_history.jsonl").read_bytes().splitlines()) == 2

    # A failed write may account for a partial change only to its own target;
    # a simultaneous mutation of another registered file must still fail.
    (jsonl_root / "attempt_history.jsonl").write_bytes(b"external mutation")
    with pytest.raises(repair.RepairError, match="Concurrent or unaccounted"):
        jsonl_budget._refresh_inventory(allowed_change="stage_a_scores.jsonl")


def test_run_identity_binds_source_fingerprints_and_600_second_deadline():
    assert repair.SUPERVISED_DEADLINE_SECONDS == 600
    assert repair.EXPECTED_BRANCH == "calibration-stage0-contract-20261029"
    assert str(repair.ROOT) in sys.path
    with pytest.raises(repair.RepairError, match="versioned"):
        repair._validate_run_id("old-r1-run")


def test_stage_a_reuse_contract_ignores_only_stage_b_or_reporting_hash_changes():
    source = {name: f"native-{index}" for index, name in
              enumerate(repair.STAGE_A_EXTERNAL_SOURCE_PATHS)}
    current = dict(source)
    current["experiments/calibration/r2_feedback_repair.py"] = "new-stage-b-report-code"
    assert repair._stage_a_reuse_source_mismatches(source, current) == {}
    current[repair.STAGE_A_EXTERNAL_SOURCE_PATHS[0]] = "changed-native-source"
    mismatch = repair._stage_a_reuse_source_mismatches(source, current)
    assert set(mismatch) == {repair.STAGE_A_EXTERNAL_SOURCE_PATHS[0]}


def test_cpu_guard_uses_visible_device_count_and_initialized_context():
    hidden_but_compiled = SimpleNamespace(cuda=SimpleNamespace(
        device_count=lambda: 0, is_initialized=lambda: False,
        is_available=lambda: True))
    visible = SimpleNamespace(cuda=SimpleNamespace(
        device_count=lambda: 1, is_initialized=lambda: False,
        is_available=lambda: True))
    initialized = SimpleNamespace(cuda=SimpleNamespace(
        device_count=lambda: 0, is_initialized=lambda: True,
        is_available=lambda: True))
    assert repair._cuda_context_is_hidden(hidden_but_compiled)
    assert not repair._cuda_context_is_hidden(visible)
    assert not repair._cuda_context_is_hidden(initialized)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object containment is Windows-specific")
def test_existing_job_object_supervisor_terminates_hanging_process_tree():
    import subprocess
    from experiments.calibration.r1_runtime_supervisor import run_supervised_child
    child = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
        "time.sleep(30)"
    )
    result = run_supervised_child([sys.executable, "-c", child], timeout_seconds=0.5)
    assert result.status == "TIMED_OUT_TERMINATED"
    assert result.termination_confirmed is True
    assert result.active_job_processes == 0
