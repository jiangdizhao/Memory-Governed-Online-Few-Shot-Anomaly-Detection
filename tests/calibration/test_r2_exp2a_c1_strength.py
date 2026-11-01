from __future__ import annotations

import hashlib
import json
from pathlib import Path
import types

import numpy as np
import pytest
import torch

from experiments import p0_protocol_audit as p0
from experiments.calibration import r1_matched_correction_pilot as r1
from experiments.calibration import r2_exp2a_c1_strength as exp2a


@pytest.fixture(scope="module")
def impl():
    return p0.require_sources()


def _fake_bank(impl, *, distances=(0.13, 0.20, 0.40, 0.30),
               beta_scale=0.5, weights=None, capacity=8):
    cfg = impl.CorrectionConfig(enabled=True, beta_scale=beta_scale)
    weights = weights or {"FP": 1.30, "FN": 1.20, "TN": 0.60, "TP": 0.80}
    cfg.outcome_weight_fp = weights["FP"]
    cfg.outcome_weight_fn = weights["FN"]
    cfg.outcome_weight_tn = weights["TN"]
    cfg.outcome_weight_tp = weights["TP"]
    cb = impl.SignedCorrectionCodebook(capacity, 4, torch.device("cpu"), cfg)

    class FakeVMB:
        pass

    bank = FakeVMB()
    bank.device = torch.device("cpu")
    bank.corr_cfg = cfg
    bank.corr_cb = {"p0": {"id": [cb]}}
    bank._ensure_mode_view_normals = types.MethodType(
        lambda self, *args, **kwargs: None, bank)
    bank._dN_min_fixed_dyn = types.MethodType(
        lambda self, cls, mode, view, layer, q: (
            torch.as_tensor(distances, dtype=torch.float32), None), bank)
    bank.update_corrections_from_topk = types.MethodType(
        impl.PatchVMB.update_corrections_from_topk, bank)
    return bank, cb, cfg


def _original_update(impl, bank, *, outcome="FP", score=0.3, theta=0.2,
                     patch_ids=(0, 1, 2), step=1, sign=None):
    label = 1 if outcome in ("FN", "TP") else 0
    if sign is None:
        sign = 1 if label else -1
    return impl.PatchVMB.update_corrections_from_topk(
        bank, cls_name="p0", feats_per_layer=[torch.eye(4, dtype=torch.float32)],
        mode=0, view="id", layer_to_patch_idx={0: list(patch_ids)},
        sign=sign, score_img=score, theta=theta, strength_scale=1.0,
        outcome=outcome, step=step)


def test_arm_definitions_match_exact_c1_defaults():
    parameters = exp2a.arm_parameters()
    assert parameters["c1_full"] == {
        "beta_scale": 0.5,
        "outcome_weights": {"FP": 1.30, "FN": 1.20, "TN": 0.60, "TP": 0.80}}
    assert parameters["c1_no_outcome_weight"]["beta_scale"] == 0.5
    assert set(parameters["c1_no_outcome_weight"]["outcome_weights"].values()) == {1.0}
    assert parameters["c1_no_margin"]["beta_scale"] == 0.0
    assert parameters["c1_no_margin"]["outcome_weights"] == parameters["c1_full"]["outcome_weights"]
    assert parameters["c1_no_both"]["beta_scale"] == 0.0
    assert set(parameters["c1_no_both"]["outcome_weights"].values()) == {1.0}


def test_native_arm_configuration_changes_only_the_registered_strength_fields(impl):
    bank, reference, base = _fake_bank(impl)
    configs, codebooks = exp2a._make_arm_configs(bank, impl)
    assert configs["c1_full"] is base
    assert codebooks["c1_full"] is reference
    full = vars(configs["c1_full"]).copy()
    for arm in exp2a.ARMS[1:]:
        candidate = vars(configs[arm]).copy()
        allowed = {"beta_scale", "outcome_weight_fp", "outcome_weight_fn",
                   "outcome_weight_tn", "outcome_weight_tp", "enabled"}
        assert {key for key in full if full[key] != candidate[key]} <= allowed
        assert codebooks[arm] is not reference
        assert codebooks[arm].size() == 0
    assert configs["c1_no_outcome_weight"].beta_scale == base.beta_scale
    assert configs["c1_no_outcome_weight"].enable_sign_aware_write
    assert configs["c1_no_outcome_weight"].enable_sign_aware_retention
    assert configs["c1_no_margin"].beta_scale == 0.0
    assert configs["c1_no_margin"].outcome_weight_fp == base.outcome_weight_fp
    assert configs["c1_no_both"].beta_scale == 0.0
    assert configs["c1_no_both"].outcome_weight_fp == 1.0


def test_runtime_tracer_observes_native_dn_and_all_four_arm_beta_inputs(impl):
    bank, _reference, _base = _fake_bank(impl)
    configs, codebooks = exp2a._make_arm_configs(bank, impl)
    group = hashlib.sha256(b"query-source-group").hexdigest()
    states = {arm: {} for arm in exp2a.ARMS}
    diagnostics = {}
    for arm in exp2a.ARMS:
        _result, layer, diagnostics[arm] = exp2a._capture_update(
            impl, bank, r1, arm, configs[arm], codebooks[arm],
            [torch.eye(4, dtype=torch.float32)],
            np.asarray([0, 1, 2], dtype=np.int64), -1, 0.3, 0.2,
            "FP", 1, states[arm], group)
        assert diagnostics[arm]["diagnostic_trace_complete"]
        assert diagnostics[arm]["selected_dN_before_gate"] == pytest.approx(
            [0.13, 0.20, 0.40])
        assert diagnostics[arm]["native_gate_keep_mask"] == [True, True, True]
        assert len(diagnostics[arm]["patch_multiplier"]) == 3
        assert len(diagnostics[arm]["beta_patch_float32_before_codebook_clip"]) == 3
        assert diagnostics[arm]["outcome"] == "FP"
        stored = (int(layer.get("inserted", 0)) + int(layer.get("merged", 0)) +
                  int(layer.get("replaced", 0)))
        assert codebooks[arm].outcome_fp[:codebooks[arm].size()].sum().item() == stored
    assert diagnostics["c1_full"]["outcome_weight"] == 1.30
    assert diagnostics["c1_no_outcome_weight"]["outcome_weight"] == 1.0
    assert diagnostics["c1_no_margin"]["post_base_clip_beta"] == 0.02
    assert diagnostics["c1_no_both"]["outcome_weight"] == 1.0
    assert len({diagnostics[arm]["codebook_state_sha256_after"]
                for arm in exp2a.ARMS}) == 4
    assert bank.corr_cfg is configs["c1_full"]
    assert bank.corr_cfg.enabled is False


@pytest.mark.parametrize(
    "outcome,weight,sign",
    [("FP", 1.30, -1), ("FN", 1.20, 1),
     ("TN", 0.60, -1), ("TP", 0.80, 1)])
def test_original_v22_outcome_weights_and_patch_beta_are_preserved(
        impl, outcome, weight, sign):
    bank, cb, cfg = _fake_bank(impl)
    result = _original_update(impl, bank, outcome=outcome, sign=sign)
    layer = result["layers"]["0"]
    assert layer["outcome_weight"] == weight
    assert layer["beta_abs_base"] == pytest.approx(0.05)
    assert layer["beta_abs_after_outcome"] == pytest.approx(
        np.clip(0.05 * weight, cfg.beta_min, cfg.beta_max))
    assert layer["kept"] == 3
    assert layer["inserted"] == 3
    d = np.asarray([0.13, 0.20, 0.40], dtype=np.float64)
    z = (d - d.min()) / (d.max() - d.min() + 1e-12)
    multiplier = np.clip(cfg.patch_beta_floor +
                         (cfg.patch_beta_cap - cfg.patch_beta_floor) * z,
                         cfg.patch_beta_floor, cfg.patch_beta_cap)
    expected = np.asarray(
        [sign * layer["beta_abs_after_outcome"] * m for m in multiplier],
        dtype=np.float32)
    np.testing.assert_allclose(cb.beta[:3].numpy(), expected, rtol=0, atol=1e-7)


def test_no_margin_uses_source_beta_min_clip_and_saturation_order(impl):
    bank, cb, cfg = _fake_bank(impl, beta_scale=0.0)
    row = _original_update(impl, bank, outcome="FN", score=0.2, theta=0.2)
    layer = row["layers"]["0"]
    assert layer["beta_abs_base"] == cfg.beta_min
    assert layer["beta_abs_after_outcome"] == pytest.approx(0.024)
    assert layer["beta_abs_min"] > 0
    bank2, cb2, cfg2 = _fake_bank(impl, beta_scale=0.5)
    saturated = _original_update(
        impl, bank2, outcome="FP", score=0.0, theta=0.5)
    assert saturated["layers"]["0"]["beta_abs_base"] == cfg2.beta_max
    assert saturated["layers"]["0"]["beta_abs_after_outcome"] == cfg2.beta_max
    assert float(cb2.beta[:3].abs().max()) == pytest.approx(cfg2.beta_max)


def test_factorial_contrasts_and_category_equal_aggregation():
    effects = exp2a.factorial_effects(0.80, 0.70, 0.65, 0.60)
    assert effects["outcome_effect_margin_enabled"] == pytest.approx(0.10)
    assert effects["outcome_effect_margin_disabled"] == pytest.approx(0.05)
    assert effects["margin_effect_outcome_enabled"] == pytest.approx(0.15)
    assert effects["margin_effect_outcome_disabled"] == pytest.approx(0.10)
    assert effects["interaction"] == pytest.approx(0.05)
    data = {category: {"effect": index / 10}
            for index, category in enumerate(exp2a.CATEGORIES, 1)}
    aggregate = exp2a.category_equal_aggregate(data, ("effect",))
    assert aggregate["effect"]["category_equal_macro"] == pytest.approx(0.25)
    assert aggregate["effect"]["worst_category"] == pytest.approx(0.1)
    assert aggregate["effect"]["positive_category_count"] == 4

    partial = {category: {"effect": value}
               for category, value in zip(exp2a.CATEGORIES, (0.1, None, 0.2, 0.3))}
    partial_summary = exp2a.category_equal_aggregate(
        partial, ("effect",))["effect"]
    assert partial_summary["category_equal_macro"] is None
    assert partial_summary["category_equal_macro_defined"] == pytest.approx(0.2)
    assert partial_summary["defined_category_count"] == 3
    assert partial_summary["defined_categories"] == [
        exp2a.CATEGORIES[0], exp2a.CATEGORIES[2], exp2a.CATEGORIES[3]]
    assert partial_summary["undefined_categories"] == [exp2a.CATEGORIES[1]]


def test_coverage_aware_verdict_does_not_confuse_missing_class_with_weak_activation():
    effects = {
        "margin_effect_outcome_enabled": {
            "defined_category_count": 3,
            "category_equal_macro_defined": -0.01,
            "positive_category_count": 1,
        },
        "outcome_effect_margin_enabled": {
            "defined_category_count": 3,
            "category_equal_macro_defined": -0.001,
            "positive_category_count": 2,
        },
    }
    assert exp2a._coverage_aware_development_verdict(
        "PASS", effects, [0.4, 0.7, 0.8]) == "NO_CLEAR_INCREMENTAL_BENEFIT"
    assert exp2a._coverage_aware_development_verdict(
        "MISMATCH", effects, [0.4, 0.7, 0.8]) == "REFERENCE_PARITY_UNVERIFIED"
    assert exp2a._coverage_aware_development_verdict(
        "PASS", effects, [0.0, None, 0.0]) == "INSUFFICIENT_ACTIVATION"
    effects["margin_effect_outcome_enabled"]["defined_category_count"] = 2
    assert exp2a._coverage_aware_development_verdict(
        "PASS", effects, [0.4, 0.7, 0.8]) == "EXPERIMENT_INCOMPLETE"


def test_codebook_states_are_independent_and_sign_aware_retention_works(impl):
    cfg_a = impl.CorrectionConfig(enabled=True, K_correction=2)
    cfg_b = impl.CorrectionConfig(enabled=True, K_correction=2)
    a = impl.SignedCorrectionCodebook(2, 4, torch.device("cpu"), cfg_a)
    b = impl.SignedCorrectionCodebook(2, 4, torch.device("cpu"), cfg_b)
    q = torch.eye(4, dtype=torch.float32)
    a.update(q[:1], torch.tensor([0.04]), outcome="TN", step=1)
    b.update(q[:1], torch.tensor([0.08]), outcome="TN", step=1)
    assert not torch.equal(a.beta[:1], b.beta[:1])
    a.update(q[1:2], torch.tensor([-0.20]), outcome="FN", step=2)
    stats = a.update(q[2:3], torch.tensor([0.20]), outcome="FP", step=3)
    assert stats["replaced"] == 1
    assert stats["same_sign_replaced"] == 1
    assert stats["cross_sign_replaced"] == 0
    assert a.size() == 2
    assert int(a.last_step[:a.size()].max()) == 3
    assert b.size() == 1
    assert int(b.last_step[:b.size()].max()) == 1


def test_source_group_exclusion_uses_active_anchor_provenance(impl):
    group_a = hashlib.sha256(b"group-a").hexdigest()
    group_b = hashlib.sha256(b"group-b").hexdigest()
    cfg = impl.CorrectionConfig(enabled=True)
    cb = impl.SignedCorrectionCodebook(4, 4, torch.device("cpu"), cfg)
    cb.update(torch.eye(4)[:2], torch.tensor([-0.04, 0.04]),
              outcome="FP", step=0)
    class Bank:
        pass
    bank = Bank()
    bank.corr_cfg = cfg
    bank.corr_cb = {"p0": {"id": [cb]}}
    groups = {0: {group_a}, 1: {group_b}}
    eligible = r1._validate_c1_source_groups(
        groups, [0, 0], event_position=1, phase="pre_score", event_group=group_a)
    assert eligible == [1]
    filtered, copied = r1._filtered_c1_codebook(
        impl, bank, groups, group_a, event_position=1)
    assert copied == [1]
    assert filtered.size() == 1
    assert r1.source_disjoint_anchor_indices(
        [0], [group_a], event_position=1, event_group=group_a) == []
    assert r1.source_disjoint_anchor_indices(
        [0], [group_b], event_position=1, event_group=group_a) == [0]
    with pytest.raises(r1.R1CausalityError):
        r1._validate_c1_source_groups(
            groups, [0, 2], event_position=1, phase="pre_score",
            event_group=group_a)


def test_query_tape_causality_seals_scores_before_disclosure():
    calls = []
    gate = r1.CausalEventOrder(4)
    with pytest.raises(r1.R1CausalityError):
        gate.disclose(lambda: calls.append("label") or 1)
    gate.seal_scores()
    assert gate.decide_query(True)
    assert gate.disclose(lambda: calls.append("label") or 1) == 1
    gate.corrections_updated()
    gate.memory_updated()
    gate.finish()
    assert calls == ["label"]
    unqueried = r1.CausalEventOrder(5)
    unqueried.seal_scores()
    assert unqueried.decide_query(False) is False
    with pytest.raises(r1.R1CausalityError):
        unqueried.disclose(lambda: calls.append("forbidden") or 0)


def test_corrected_dense_map_uses_true_maximum():
    base = np.zeros(784, dtype=np.float32)
    base[700] = 0.9
    ids = np.asarray([1, 5, 10], dtype=np.int64)
    delta = np.asarray([0.1, 0.8, -0.2], dtype=np.float32)
    corrected = r1.corrected_dense_map(base, ids, delta)
    assert r1.max_score(corrected) == pytest.approx(0.9)
    assert corrected[5] == pytest.approx(0.8)


def test_cross_group_pair_summary_preserves_undefined_reason_and_denominator():
    scores = {
        "c0": np.asarray([0.9, 0.1], dtype=np.float32),
        "c1_full": np.asarray([0.1, 0.9], dtype=np.float32),
    }
    undefined = exp2a._cross_group_ranking_diagnostics(
        np.asarray([0, 0], dtype=np.int8), scores, ["g0", "g1"])
    summary = exp2a._cross_group_pair_harm_for_arm(undefined, "c1_full")
    assert summary == {
        "status": "UNDEFINED",
        "reason": "requires at least one normal and one defect",
        "cross_source_group_normal_defect_pair_denominator": 0,
        "same_source_group_pairs_excluded": 0,
    }

    defined = exp2a._cross_group_ranking_diagnostics(
        np.asarray([0, 1], dtype=np.int8), scores, ["normal-group", "defect-group"])
    summary = exp2a._cross_group_pair_harm_for_arm(defined, "c1_full")
    assert summary["status"] == "DEFINED"
    assert summary["cross_source_group_normal_defect_pair_denominator"] == 1
    assert summary["strict_wrong_to_strict_correct_repaired"] == 1
    assert summary["strict_correct_to_strict_wrong_damaged"] == 0


def test_common_writer_uses_only_c0_and_never_writes_unqueried_or_defect(
        monkeypatch):
    calls = []
    def eligible(**_kwargs):
        return True, 32, "native", {}
    def update(**kwargs):
        calls.append(kwargs["position"])
        return {"accepted": 32}
    common = r1.common_c0_greedy_writer_action
    ids = np.arange(32, dtype=np.int64)
    features = np.ones((32, 4), dtype=np.float32)
    base_inputs = {
        "position": 0, "c0_score": 0.1, "theta": 0.2, "sigma": 0.1,
        "c0_prediction": 0, "candidate_ids": ids,
        "candidate_features": features, "feature_layers": [torch.ones((32, 4))],
        "eligibility": eligible, "update": update,
    }
    unqueried = common(**base_inputs, queried=False, disclosed_label=None)
    defect = common(**{**base_inputs, "position": 1}, queried=True, disclosed_label=1)
    assert unqueried["action"] == "no_normal_memory_write"
    assert defect["action"] == "verified_defect_no_normal_write"
    assert calls == []
    normal = common(**{**base_inputs, "position": 2}, queried=True, disclosed_label=0)
    assert normal["action"] == "native_greedy_normal_update"
    assert calls == [2]
    # Candidate C1 scores and Codebook states are absent from this adapter's
    # inputs, so their values cannot select or suppress the shared write.
    assert normal["input_score_source"] == "common_C0"


def test_output_budget_refuses_overflow_before_write_and_reserves_failure_bytes(tmp_path):
    Budget = exp2a._output_budget_class(r1)
    root = tmp_path / "budget"
    budget = Budget(root, max_bytes=128, failure_reserve_bytes=32)
    try:
        budget.ensure_root()
        (root / "events").mkdir()
        budget.write_once(root / "run_manifest.json", b"x" * 30)
        budget.append(root / "event_scores.jsonl", b"y" * 60)
        prior = (root / "event_scores.jsonl").stat().st_size
        with pytest.raises(r1.OutputBudgetError, match="refused before I/O"):
            budget.append(root / "event_scores.jsonl", b"z" * 20)
        assert (root / "event_scores.jsonl").stat().st_size == prior
        budget.write_once(root / "failure.json", b"f" * 35, failure_reserved=True)
        assert budget._audit_tree() == 125
    finally:
        budget.close()
    with pytest.raises(FileExistsError):
        Budget(root, max_bytes=128, failure_reserve_bytes=32)


def test_stage_b_refuses_to_open_outcomes_without_complete_seal(
        tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    (root / "run_manifest.json").write_text("{}", encoding="utf-8")
    (root / "stage_a_seal.json").write_text(json.dumps({
        "schema": "r2.exp2a.stage_a_seal.v1",
        "status": "SEALED_INCOMPLETE",
        "category_event_counts": {name: 1 for name in exp2a.CATEGORIES},
        "metadata": {},
    }), encoding="utf-8")
    monkeypatch.setattr(
        exp2a, "_stage_b_labels",
        lambda *args, **kwargs: pytest.fail("outcomes must remain unopened"))
    with pytest.raises(exp2a.Exp2AIntegrityError, match="Stage A is incomplete"):
        exp2a._stage_b_worker(
            output=root, dataset_root=tmp_path, json_root=tmp_path,
            bootstrap_iterations=1)


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


def test_switch_reference_parity_compares_full_trajectory(tmp_path, monkeypatch):
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    (old_root / "events").mkdir(parents=True)
    (new_root / "events").mkdir(parents=True)
    (old_root / "run_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(exp2a, "R1_SWITCH", old_root)
    old_scores, old_transitions, old_feedback = [], [], []
    new_scores, new_transitions, new_feedback, new_diagnostics = [], [], [], []
    group = hashlib.sha256(b"shared-group").hexdigest()
    state = hashlib.sha256(b"state").hexdigest()
    memory_before = hashlib.sha256(b"memory-before").hexdigest()
    memory_after = hashlib.sha256(b"memory-after").hexdigest()
    fixed_support = hashlib.sha256(b"fixed-support").hexdigest()
    for position in range(exp2a.MAX_EVENTS):
        base = np.zeros(784, dtype=np.float32)
        base[30] = np.float32(0.5)
        ids = np.arange(32, dtype=np.int64)
        features = np.zeros((32, 4), dtype=np.float32)
        old_arrays = {
            "base_patch_map": base, "top32_ids": ids,
            "top32_features": features,
            "c1_delta": np.zeros(32, dtype=np.float32),
            "c2_delta": np.zeros(32, dtype=np.float32),
        }
        new_arrays = {
            "base_patch_map": base, "top32_ids": ids,
            "top32_features": features,
            "c1_delta_full": np.zeros(32, dtype=np.float32),
            "c1_delta_no_outcome_weight": np.zeros(32, dtype=np.float32),
            "c1_delta_no_margin": np.zeros(32, dtype=np.float32),
            "c1_delta_no_both": np.zeros(32, dtype=np.float32),
            "c2_delta_a002": np.zeros(32, dtype=np.float32),
            "c2_delta_a020": np.zeros(32, dtype=np.float32),
        }
        old_bytes = r1.deterministic_npz_bytes(old_arrays)
        new_bytes = r1.deterministic_npz_bytes(new_arrays)
        old_name = f"events/event_{position:06d}.npz"
        new_name = f"events/event_{position:06d}.npz"
        (old_root / old_name).write_bytes(old_bytes)
        (new_root / new_name).write_bytes(new_bytes)
        queried = position == 0
        old_scores.append({
            "position": position, "category": "switch",
            "shard_path": old_name,
            "shard_sha256": r1.sha256_bytes(old_bytes),
            "array_sha256": r1.arrays_sha256(old_arrays),
            "c0_score": 0.5, "c1_score": 0.5,
            "c1_state_sha256_before": state,
            "greedy_memory_sha256_before": memory_before,
                "c0_prediction": 1, "official_group_sha256": group,
                "fixed_support_sha256": fixed_support,
        })
        new_scores.append({
            "position": position, "category": "switch",
            "shard_path": new_name,
            "shard_sha256": r1.sha256_bytes(new_bytes),
            "arrays_sha256": r1.arrays_sha256(new_arrays),
            "scores": {arm: 0.5 for arm in exp2a.SCORE_ARMS},
                "predictions_at_frozen_c64_theta": {arm: 1 for arm in exp2a.SCORE_ARMS},
            "c1_state_sha256_before": {arm: state for arm in exp2a.ARMS},
            "c0_memory_sha256_before": memory_before,
                "source_group_sha256": group,
                "fixed_support_sha256": fixed_support,
        })
        old_transitions.append({
            "position": position, "query_decision": queried,
            "c1_state_sha256_after": state,
            "greedy_memory_sha256_after": memory_after,
        })
        new_transitions.append({
            "position": position, "category": "switch",
            "query_decision": queried,
            "c1_state_sha256_after": {arm: state for arm in exp2a.ARMS},
            "c0_memory_sha256_after": memory_after,
        })
        if queried:
            old_feedback.append({
                    "category": "switch", "position": position, "outcome": "TN",
                "c0_prediction": 0, "source_group_sha256": group,
                "c1_stats": {"inserted": 1, "merged": 0, "replaced": 0,
                             "skipped": 0, "same_sign_replaced": 0,
                             "cross_sign_replaced": 0},
            })
            new_feedback.append({
                    "category": "switch", "position": position, "label": 0,
                "source_group_sha256": group,
            })
            new_diagnostics.append({
                "category": "switch",
                "position": position,
                "arms": {"c1_full": {"native_layer_stats": {
                    "inserted": 1, "merged": 0, "replaced": 0,
                    "skipped": 0, "same_sign_replaced": 0,
                    "cross_sign_replaced": 0}}},
            })
    _write_jsonl(old_root / "sealed_scores.jsonl", old_scores)
    _write_jsonl(old_root / "event_transitions.jsonl", old_transitions)
    _write_jsonl(old_root / "correction_anchor_events.jsonl", old_feedback)
    _write_jsonl(new_root / "event_scores.jsonl", new_scores)
    _write_jsonl(new_root / "event_transitions.jsonl", new_transitions)
    _write_jsonl(new_root / "private_query_feedback.jsonl", new_feedback)
    _write_jsonl(new_root / "strength_diagnostics.jsonl", new_diagnostics)
    result = exp2a._verify_switch_reference(new_root)
    assert result["status"] == "PASS"
    assert result["event_checks"]["events"] == 512
    assert result["event_checks"]["query_events"] == 1


def test_task_specific_180_minute_job_limit_is_scoped_and_restored(monkeypatch):
    from experiments.calibration import r1_runtime_supervisor as supervisor

    original = supervisor.MAX_RUNTIME_SECONDS
    assert original == 120 * 60
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["timeout"] = kwargs["timeout_seconds"]
        seen["guard_during_call"] = supervisor.MAX_RUNTIME_SECONDS
        return "completed"

    monkeypatch.setattr(supervisor, "run_supervised_child", fake_run)
    result = exp2a._run_exp2a_supervised_child(
        ["python.exe", "worker.py"], timeout_seconds=180 * 60,
        env={}, cwd=Path.cwd())
    assert result == "completed"
    assert seen == {
        "command": ["python.exe", "worker.py"],
        "timeout": 180 * 60,
        "guard_during_call": 180 * 60,
    }
    assert supervisor.MAX_RUNTIME_SECONDS == original


def test_task_specific_job_limit_restores_guard_after_supervisor_error(monkeypatch):
    from experiments.calibration import r1_runtime_supervisor as supervisor

    original = supervisor.MAX_RUNTIME_SECONDS

    def fail_run(*_args, **_kwargs):
        assert supervisor.MAX_RUNTIME_SECONDS == 180 * 60
        raise RuntimeError("synthetic supervisor failure")

    monkeypatch.setattr(supervisor, "run_supervised_child", fail_run)
    with pytest.raises(RuntimeError, match="synthetic supervisor failure"):
        exp2a._run_exp2a_supervised_child(
            ["python.exe", "worker.py"], timeout_seconds=180 * 60,
            env={}, cwd=Path.cwd())
    assert supervisor.MAX_RUNTIME_SECONDS == original
