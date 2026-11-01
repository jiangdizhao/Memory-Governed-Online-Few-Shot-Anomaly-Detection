"""Synthetic CPU tests for R2-EXP2B memory/readout attribution."""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pytest

from experiments.calibration import r2_exp2b_memory_readout as exp


def _native(capacity: int = 8, feature_dim: int = 2):
    torch = exp._torch()
    import test_phase3_AIF_simplified_v22 as v22
    cfg = v22.CorrectionConfig()
    cfg.enabled = True
    cfg.merge_eps = 0.001
    cb = v22.SignedCorrectionCodebook(
        K=capacity, feat_dim=feature_dim, device=torch.device("cpu"), cfg=cfg)
    return v22, cfg, cb


def test_common_candidate_replay_uses_keep_mask_clip_and_beta_floor():
    _v22, cfg, _cb = _native(capacity=4)
    keep = np.array([True, False, True, True], dtype=bool)
    beta = np.array([0.10, 0.20, 0.01998, 0.50], dtype=np.float32)
    result = exp._common_candidate_eligibility(keep, beta, cfg)
    assert result["raw_selected_count"] == 4
    assert result["attempted_after_dN"] == 3
    assert result["dN_gate_rejects"] == 1
    assert result["beta_floor_rejects"] == 1
    assert result["common_eligible_count"] == 2
    assert result["clipped_beta"].tolist() == pytest.approx([0.10, 0.20, 0.01998, 0.20])
    assert result["common_mask"].tolist() == [True, False, False, True]


def test_replay_native_update_uses_selected_features_and_aligned_beta():
    torch = exp._torch()
    _v22, _cfg, cb = _native(capacity=4)
    features = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    keep = np.array([True, False, True])
    beta = np.array([0.10, 0.20, -0.12], dtype=np.float32)
    stats = exp._replay_native_update(cb, features, keep, beta, outcome="FN", position=7)
    assert stats["inserted"] == 2
    assert stats["merged"] == stats["replaced"] == 0
    assert int(cb.size()) == 2
    assert cb.last_step[:2].tolist() == [7, 7]
    assert cb.beta[:2].tolist() == pytest.approx([0.10, -0.12])
    assert cb.protos.device.type == "cpu"
    assert cb.protos.dtype == torch.float32


def test_beta_floor_skip_is_distinct_from_capacity_rejection():
    torch = exp._torch()
    _v22, _cfg, floor_cb = _native(capacity=1)
    floor = floor_cb.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0.01998]),
                            outcome="TN", step=0)
    assert floor["skipped"] == 1
    assert floor["inserted"] == 0

    _v22, _cfg, full_cb = _native(capacity=1)
    full_cb.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0.20]),
                   outcome="FP", step=0)
    rejected = full_cb.update(torch.tensor([[0.0, 1.0]]), torch.tensor([0.02]),
                              outcome="TN", step=1)
    assert rejected["skipped"] == 1
    assert rejected["inserted"] == rejected["merged"] == rejected["replaced"] == 0
    assert int(full_cb.size()) == 1


def test_native_codebook_known_merge_and_replacement_examples():
    torch = exp._torch()
    _v22, _cfg, merge_cb = _native(capacity=2)
    first = merge_cb.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0.10]),
                            outcome="FP", step=0)
    second = merge_cb.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0.14]),
                             outcome="TN", step=1)
    assert first["inserted"] == 1
    assert second["merged"] == 1
    assert int(merge_cb.size()) == 1
    assert int(merge_cb.count[0]) == 2

    _v22, _cfg, replace_cb = _native(capacity=1)
    replace_cb.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0.02]),
                      outcome="TN", step=0)
    replaced = replace_cb.update(torch.tensor([[0.0, 1.0]]), torch.tensor([0.20]),
                                 outcome="FP", step=1)
    assert replaced["replaced"] == 1
    assert int(replace_cb.last_step[0]) == 1
    assert replace_cb.beta[0].item() == pytest.approx(0.20)


def test_fifo_matches_native_codebook_when_policy_has_no_merge_or_capacity_action():
    torch = exp._torch()
    v22, cfg, cb = _native(capacity=8)
    fifo = []
    for pos, (feature, beta) in enumerate((([1, 0], 0.10), ([0, 1], -0.12))):
        q = torch.tensor([feature], dtype=torch.float32)
        stats = cb.update(q, torch.tensor([beta]), outcome="FP", step=pos)
        assert stats["inserted"] == 1
        assert stats["merged"] == stats["replaced"] == stats["skipped"] == 0
        normalized = v22.l2_normalize(q)[0]
        fifo.append(exp.Anchor(normalized.clone(), float(np.float32(beta)),
                               float(np.float32(cfg.radius0)), float(np.float32(cfg.kappa0)),
                               pos, f"{pos + 1:x}" * 64))
        assert exp._memory_equal(fifo, cb)


def test_source_group_exclusion_and_temporal_eligibility_use_native_contract():
    torch = exp._torch()
    v22, _cfg, cb = _native(capacity=4, feature_dim=4096)
    r1 = __import__("experiments.calibration.r1_matched_correction_pilot",
                    fromlist=["_record_c1_source_group_updates"])
    source_group = "a" * 64
    other_group = "b" * 64
    feature = torch.zeros((1, 4096), dtype=torch.float32)
    feature[0, 0] = 1.0
    cb.update(feature, torch.tensor([0.10]), outcome="FP", step=0)
    source_groups: dict[int, set[str]] = {}
    r1._record_c1_source_group_updates(source_groups,
        last_steps=cb.last_step[:cb.size()].tolist(), event_position=0,
        event_group=source_group)
    fifo = [exp.Anchor(cb.protos[0].clone(), 0.10, 0.10, 1.0, 0, source_group)]
    query = np.zeros((32, 4096), dtype=np.float32)
    query[:, 0] = 1.0
    same = exp._read_memory(cb, source_groups, fifo, query, source_group, 1, v22, r1)
    assert same[-1]["codebook_live"] == 1
    assert same[-1]["codebook_same_source_or_ineligible"] == 1
    assert same[-1]["fifo_same_source_or_temporal"] == 1
    assert np.count_nonzero(same[0]) == np.count_nonzero(same[1]) == 0

    other = exp._read_memory(cb, source_groups, fifo, query, other_group, 1, v22, r1)
    assert other[-1]["codebook_same_source_or_ineligible"] == 0
    assert other[-1]["fifo_same_source_or_temporal"] == 0
    assert np.all(other[0] != 0)


def test_gaussian_and_matched_top5_formulas_and_stable_distance_ties():
    torch = exp._torch()
    distances = torch.tensor([[0.0, 0.0, 0.2, 0.3, 0.4, 0.5]], dtype=torch.float32)
    beta = torch.tensor([0.10, -0.10, 0.20, -0.20, 0.30, 0.40], dtype=torch.float32)
    radius = torch.full((6,), 0.10, dtype=torch.float32)
    kappa = torch.tensor([1.0, 2.0, 1.0, 1.0, 2.0, 1.0], dtype=torch.float32)
    gaussian, top5, selected = exp._two_readouts_from_distances(distances, beta, radius, kappa)
    weights = torch.exp(-(distances * distances) / (2 * radius[None, :] ** 2)) * kappa[None, :]
    expected_g = (weights * beta[None, :]).sum(1) / (weights.sum(1) + exp.GAUSSIAN_EPS)
    chosen = torch.tensor([0, 1, 2, 3, 4])
    expected_k = (kappa[chosen] * beta[chosen]).sum() / (kappa[chosen].sum() + exp.TOP5_EPS)
    assert gaussian[0] == pytest.approx(expected_g.item(), abs=1e-7)
    assert top5[0] == pytest.approx(expected_k.item(), abs=1e-7)
    assert selected.tolist() == [[0, 1, 2, 3, 4]]


def test_empty_memory_returns_zero_for_both_readouts():
    torch = exp._torch()
    empty = torch.empty((3, 0), dtype=torch.float32)
    zeros, top5, selected = exp._two_readouts_from_distances(
        empty, torch.empty((0,)), torch.empty((0,)), torch.empty((0,)))
    assert zeros.tolist() == [0.0, 0.0, 0.0]
    assert top5.tolist() == zeros.tolist()
    assert selected.shape == (3, 0)


def test_original_dense_map_max_aggregation_uses_all_784_patches():
    r1 = __import__("experiments.calibration.r1_matched_correction_pilot",
                    fromlist=["corrected_dense_map", "max_score"])
    base = np.zeros((784,), dtype=np.float32)
    base[100] = 0.8
    delta = np.array([0.3, -0.1], dtype=np.float32)
    score = exp._score_dense(base, np.array([2, 5], dtype=np.int64), delta, r1)
    assert score == pytest.approx(0.8)
    base[5] = 0.7
    assert exp._score_dense(base, np.array([2, 5]), delta, r1) == pytest.approx(0.8)


def test_d_reference_parity_uses_frozen_tolerance_and_reports_nonbitwise_values():
    actual = np.array([0.1, 0.2], dtype=np.float32)
    expected = actual.copy()
    expected[0] = np.float32(expected[0] + 4e-6)
    max_error, nonbitwise, accepted = exp._compare_reference(actual, expected)
    assert max_error > 0
    assert nonbitwise == 1
    assert accepted
    _error, _count, rejected = exp._compare_reference(
        actual, np.array([0.12, 0.2], dtype=np.float32))
    assert not rejected


def test_readouts_do_not_change_native_state_or_future_update():
    torch = exp._torch()
    v22, _cfg, cb = _native(capacity=4, feature_dim=4096)
    r1 = __import__("experiments.calibration.r1_matched_correction_pilot",
                    fromlist=["_record_c1_source_group_updates", "_correction_state_sha256"])
    group = "c" * 64
    q0 = torch.zeros((1, 4096), dtype=torch.float32)
    q0[0, 0] = 1.0
    cb.update(q0, torch.tensor([0.10]), outcome="FP", step=0)
    provenance: dict[int, set[str]] = {}
    r1._record_c1_source_group_updates(
        provenance, last_steps=cb.last_step[:cb.size()].tolist(), event_position=0,
        event_group=group)
    proxy = type("Proxy", (), {"corr_cb": {"p0": {"id": [cb]}}})()
    before = r1._correction_state_sha256(proxy, provenance)
    query = np.zeros((32, 4096), dtype=np.float32)
    query[:, 0] = 1.0
    exp._read_memory(cb, provenance, [], query, "d" * 64, 1, v22, r1)
    after = r1._correction_state_sha256(proxy, provenance)
    assert before == after


def test_magnitude_normalization_is_retrospective_and_zero_scale_is_unavailable():
    gaussian = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    knn = np.ones_like(gaussian)
    result = exp._trajectory_alpha(gaussian, knn)
    assert result["status"] == "AVAILABLE"
    assert result["alpha"] == pytest.approx(np.sqrt(2.0))
    assert result["is_causal"] is False
    zero = exp._trajectory_alpha(gaussian, np.zeros_like(knn))
    assert zero["status"] == "ZERO_KNN_RMS_UNAVAILABLE"
    assert zero["alpha"] is None


def test_audiojack_style_single_class_has_defined_normal_fpr_only():
    evaluator = __import__("experiments.calibration.r2_perf1_stage_b", fromlist=["_metric_record"])
    labels = np.zeros((16,), dtype=np.int8)
    scores = np.linspace(0.0, 1.0, len(labels))
    record = evaluator._metric_record(labels, scores, 0.75)
    assert record["image_auroc"] is None
    assert record["image_auprc_average_precision"] is None
    assert record["fixed_threshold"]["fpr"] == pytest.approx(0.25)
    assert record["fixed_threshold"]["fnr"] is None
    assert record["fixed_threshold"]["balanced_accuracy"] is None


def test_stage_b_binds_exp2a_identity_projection_to_shared_perf1_metrics():
    label_provider = __import__(
        "experiments.calibration.r2_exp2a_c1_strength", fromlist=["_stage_b_labels"])
    metric_provider = __import__(
        "experiments.calibration.r2_perf1_stage_b", fromlist=["_metric_record"])
    evaluator = exp._stage_b_evaluator()
    assert evaluator._stage_b_labels is label_provider._stage_b_labels
    assert evaluator._metric_record is metric_provider._metric_record
    assert evaluator.DATASET_ROOT == label_provider.DATASET_ROOT
    assert evaluator.JSON_ROOT == label_provider.JSON_ROOT


def test_stage_b_refuses_missing_score_seal_before_evaluator_access(tmp_path):
    with pytest.raises(exp.Exp2BIntegrityError, match="refused evaluator access"):
        exp._verify_score_seal(tmp_path)


def test_output_ledger_enforces_synthetic_budget_before_each_write(tmp_path):
    root = tmp_path / "budget"
    with exp.OutputLedger(root, create=True, test_only=True,
                          max_bytes=128, failure_reserve_bytes=16) as ledger:
        ledger.write("run_manifest.json", b"x" * 60)
        ledger.write("INCOMPLETE", b"y" * 50)
        assert ledger.used_bytes == 111  # 1-byte lock + 60 + 50
        with pytest.raises(exp.OutputBudgetError, match="Pre-write"):
            ledger.write("event_refs.jsonl", b"zz")
        ledger.write("failure.json", b"f" * 16, failure_reserved=True)
        assert ledger.used_bytes == 127 <= 128
        ledger.checkpoint(durable=False)


def test_output_ledger_has_no_hot_loop_tree_scan_and_fails_on_unaccounted_writer(tmp_path):
    root = tmp_path / "ledger"
    with exp.OutputLedger(root, create=True, test_only=True,
                          max_bytes=4096, failure_reserve_bytes=128) as ledger:
        start_audits = ledger.audit_count
        ledger.write("run_manifest.json", b"1")
        ledger.write("INCOMPLETE", b"2")
        ledger.write("event_refs.jsonl", b"3")
        ledger.write("stage_a_seal.json", b"4")
        assert ledger.audit_count == start_audits
        ledger.checkpoint(durable=False)
        assert ledger.audit_count == start_audits + 1
        (root / "unexpected.bin").write_bytes(b"unaccounted")
        with pytest.raises(exp.OutputBudgetError, match="Unaccounted output path"):
            ledger.checkpoint(durable=False)
