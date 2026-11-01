"""Deterministic CPU-only tests for the isolated R4 FPSA Codebook."""
from __future__ import annotations

import hashlib
import inspect

import pytest
import torch

import test_phase3_AIF_simplified_v22 as native
from experiments.calibration.r4_fpsa_codebook import (
    FPSAContractError,
    FPSASignedCorrectionCodebook,
)


def _group(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _pair(k: int = 4, dim: int = 12):
    cfg_native = native.CorrectionConfig(enabled=True, K_correction=k)
    cfg_fpsa = native.CorrectionConfig(enabled=True, K_correction=k)
    device = torch.device("cpu")
    return (
        native.SignedCorrectionCodebook(k, dim, device, cfg_native),
        FPSASignedCorrectionCodebook(k, dim, device, cfg_fpsa),
    )


def _feature(index: int, dim: int = 12) -> torch.Tensor:
    row = torch.zeros((1, dim), dtype=torch.float32, device="cpu")
    row[0, index] = 1.0
    return row


def _write(cb, feature_index: int, beta: float, step: int, outcome: str = "TN"):
    return cb.update(
        _feature(feature_index), torch.tensor([beta], dtype=torch.float32),
        outcome=outcome, step=step,
    )


def _assert_native_state_equal(left, right):
    assert left.n == right.n
    for name in (
        "protos", "beta", "radius", "kappa", "count", "outcome_fp",
        "outcome_fn", "outcome_tn", "outcome_tp", "last_step",
    ):
        torch.testing.assert_close(getattr(left, name), getattr(right, name), rtol=0, atol=0)
    assert left.counters == right.counters
    for name in left.counters:
        assert left.summary()[name] == right.summary()[name]


def test_no_quota_pressure_matches_native_updates_state_and_readout():
    native_cb, fpsa_cb = _pair()
    events = [
        (0, -0.10, "TN"), (1, -0.15, "FP"), (2, 0.10, "FN"),
        (3, 0.12, "TP"), (4, 0.20, "FN"), (2, 0.08, "TP"),
        (5, -0.09, "TN"),
    ]
    for step, (feature, beta, outcome) in enumerate(events):
        expected = _write(native_cb, feature, beta, step, outcome)
        actual = _write(fpsa_cb, feature, beta, step, outcome)
        assert actual == expected
        _assert_native_state_equal(native_cb, fpsa_cb)
        assert fpsa_cb.last_update_diagnostics["quota_borrow_opportunities"] == 0

    assert FPSASignedCorrectionCodebook.apply is native.SignedCorrectionCodebook.apply
    query = torch.eye(8, 12, dtype=torch.float32)
    torch.testing.assert_close(native_cb.apply(query), fpsa_cb.apply(query), rtol=0, atol=0)


def test_normal_only_feedback_keeps_zero_positive_quota_and_native_behavior():
    native_cb, fpsa_cb = _pair(k=3)
    for step in range(3):
        fpsa_cb.observe_verified_feedback(0, _group(f"normal-{step}"), step, score_sealed=True)
        assert _write(native_cb, step, -0.12, step) == _write(fpsa_cb, step, -0.12, step)
    assert fpsa_cb.diagnostics()["target_positive_capacity"] == 0
    assert fpsa_cb.diagnostics()["target_negative_capacity"] == 3

    expected = _write(native_cb, 4, -0.16, 4, "FP")
    actual = _write(fpsa_cb, 4, -0.16, 4, "FP")
    assert actual == expected
    _assert_native_state_equal(native_cb, fpsa_cb)
    assert fpsa_cb.diagnostics()["actual_positive_occupancy"] == 0


def test_delayed_anomalous_feedback_reclaims_slots_while_native_sign_lock_stays_at_one():
    native_cb, fpsa_cb = _pair(k=4)
    for step in range(4):
        assert _write(native_cb, step, -0.10, step) == _write(fpsa_cb, step, -0.10, step)

    for offset, outcome in enumerate(("FN", "TP", "TP")):
        step = 10 + offset
        fpsa_cb.observe_verified_feedback(1, _group(f"delayed-defect-{offset}"), step,
                                          score_sealed=True)
        _write(native_cb, 4 + offset, 0.20, step, outcome)
        result = _write(fpsa_cb, 4 + offset, 0.20, step, outcome)
        if offset > 0:
            assert result["cross_sign_replaced"] == 1

    native_summary = native_cb.summary()
    fpsa_summary = fpsa_cb.diagnostics()
    assert native_summary["positive"] == 1  # native can replace its first + anchor, not add another
    assert fpsa_summary["target_positive_capacity"] == 4
    assert fpsa_summary["actual_positive_occupancy"] == 3
    assert fpsa_summary["actual_negative_occupancy"] == 1
    assert fpsa_summary["quota_borrow_opportunities"] == 3
    assert fpsa_summary["quota_cross_sign_replacements"] == 3
    assert fpsa_summary["quota_borrow_rejected"] == 0
    assert int(fpsa_cb.last_step[:fpsa_cb.n].max().item()) == 12


def test_quota_borrow_selects_weakest_opposite_sign_by_native_retention():
    cb = FPSASignedCorrectionCodebook(3, 12, torch.device("cpu"),
                                      native.CorrectionConfig(K_correction=3))
    for step in range(3):
        _write(cb, step, -0.10, step)
    # Create unequal native counts through legitimate same-sign merges.
    _write(cb, 0, -0.10, 3)
    _write(cb, 0, -0.10, 4)
    _write(cb, 2, -0.10, 5)
    counts = cb.count[:cb.n].float()
    mistakes = (cb.outcome_fp[:cb.n] + cb.outcome_fn[:cb.n]).float()
    retention = (
        counts + 1.25 * mistakes
        + 0.50 * torch.clamp(cb.beta[:cb.n].abs() / cb.cfg.beta_max, 0.0, 1.0)
        + 0.25 * torch.clamp(cb.kappa[:cb.n] / cb.cfg.kappa_max, 0.0, 1.0)
    )
    weakest_slot = int(torch.argmin(retention).item())
    assert weakest_slot == 1

    cb.observe_verified_feedback(1, _group("retention-test-defect"), 10)
    result = _write(cb, 3, 0.20, 10, "FN")
    assert result["cross_sign_replaced"] == 1
    assert int(cb.last_step[weakest_slot].item()) == 10
    assert cb.beta[weakest_slot].item() > 0
    assert cb.diagnostics()["quota_cross_sign_replacements"] == 1


def test_feedback_quota_semantics_half_up_rounding_and_no_forced_eviction():
    cb = FPSASignedCorrectionCodebook(5, 12, torch.device("cpu"),
                                      native.CorrectionConfig(K_correction=5))
    assert cb.diagnostics()["target_positive_capacity"] is None
    assert cb.diagnostics()["target_negative_capacity"] is None

    cb.observe_verified_feedback(0, _group("normal-a"), 0)
    assert (cb.diagnostics()["target_positive_capacity"],
            cb.diagnostics()["target_negative_capacity"]) == (0, 5)
    cb.observe_verified_feedback(1, _group("defect-a"), 1)
    # 5 * 1 / 2 = 2.5, rounded half-up to 3 rather than ties-to-even 2.
    assert (cb.diagnostics()["target_positive_capacity"],
            cb.diagnostics()["target_negative_capacity"]) == (3, 2)
    for step in range(2, 12):
        cb.observe_verified_feedback(0, _group(f"normal-{step}"), step)
    assert cb.diagnostics()["target_positive_capacity"] == 1  # lower clamp

    # A changed target does not evict or relabel existing live anchors.
    full = FPSASignedCorrectionCodebook(4, 12, torch.device("cpu"),
                                        native.CorrectionConfig(K_correction=4))
    for step in range(4):
        _write(full, step, -0.10, step)
    before = full.protos.clone()
    full.observe_verified_feedback(1, _group("late-defect"), 10)
    assert full.diagnostics()["target_positive_capacity"] == 4
    assert full.diagnostics()["actual_positive_occupancy"] == 0
    torch.testing.assert_close(full.protos, before, rtol=0, atol=0)

    anomaly_only = FPSASignedCorrectionCodebook(4, 12, torch.device("cpu"),
                                               native.CorrectionConfig(K_correction=4))
    anomaly_only.observe_verified_feedback(1, _group("only-defect"), 0)
    assert (anomaly_only.diagnostics()["target_positive_capacity"],
            anomaly_only.diagnostics()["target_negative_capacity"]) == (4, 0)


def test_feedback_registration_fails_closed_and_is_transactional():
    cb = FPSASignedCorrectionCodebook(4, 12, torch.device("cpu"),
                                      native.CorrectionConfig(K_correction=4))
    good = _group("group-a")
    with pytest.raises(FPSAContractError, match="score seal"):
        cb.observe_verified_feedback(0, good, 0, score_sealed=False)
    with pytest.raises(FPSAContractError, match="label"):
        cb.observe_verified_feedback(True, good, 0)
    with pytest.raises(FPSAContractError, match="identity"):
        cb.observe_verified_feedback(0, "not-a-digest", 0)
    assert cb.diagnostics()["queried_source_groups_total"] == 0

    cb.observe_verified_feedback(0, good, 3)
    with pytest.raises(FPSAContractError, match="duplicate"):
        cb.observe_verified_feedback(1, good, 4)
    with pytest.raises(FPSAContractError, match="increasing"):
        cb.observe_verified_feedback(1, _group("group-b"), 3)
    assert cb.diagnostics()["queried_normal_source_groups"] == 1
    assert cb.diagnostics()["queried_anomalous_source_groups"] == 0
    with pytest.raises(FPSAContractError, match="increasing"):
        cb.observe_verified_feedback(1, _group("group-c"), 2)


def test_rejected_borrow_is_not_forced_and_beta_floor_remains_native():
    cb = FPSASignedCorrectionCodebook(2, 12, torch.device("cpu"),
                                      native.CorrectionConfig(K_correction=2))
    # Fill both negative slots and strengthen their native retention values by
    # same-sign merges; this is a deterministic valid native state.
    _write(cb, 0, -0.10, 0)
    _write(cb, 1, -0.10, 1)
    for step in range(2, 14):
        _write(cb, 0 if step % 2 == 0 else 1, -0.10, step)
    before = cb.protos.clone()
    cb.observe_verified_feedback(1, _group("queried-defect"), 20)

    result = cb.update(torch.cat((_feature(2), _feature(3))),
                       torch.tensor([0.01, 0.02]), outcome="TP", step=20)
    assert result["skipped"] == 2
    assert cb.last_update_diagnostics["quota_borrow_opportunities"] == 1
    assert cb.last_update_diagnostics["quota_cross_sign_replacements"] == 0
    assert cb.last_update_diagnostics["quota_borrow_rejected"] == 1
    assert cb.diagnostics()["quota_borrow_rejected"] == 1
    assert cb.diagnostics()["actual_positive_occupancy"] == 0
    torch.testing.assert_close(cb.protos, before, rtol=0, atol=0)


def test_mixed_feedback_never_exceeds_capacity_or_creates_zero_sign_slots():
    cb = FPSASignedCorrectionCodebook(3, 12, torch.device("cpu"),
                                      native.CorrectionConfig(K_correction=3))
    labels = (0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0)
    for step, label in enumerate(labels):
        cb.observe_verified_feedback(label, _group(f"mixed-{step}"), step)
        sign = 1.0 if label else -1.0
        outcome = "FN" if label else "FP"
        result = _write(cb, step % 10, sign * 0.16, step, outcome)
        assert sum(result[key] for key in ("inserted", "merged", "replaced", "skipped")) == 1
        state = cb.diagnostics()
        assert state["actual_positive_occupancy"] + state["actual_negative_occupancy"] == cb.size()
        assert cb.size() <= cb.K
        assert state["zero_sign_occupancy"] == 0
        assert int(torch.isfinite(cb.protos[:cb.n]).all().item()) == 1


def test_cross_sign_replacement_preserves_step_metadata_for_source_group_lso():
    from experiments.calibration import r1_matched_correction_pilot as r1

    cb = FPSASignedCorrectionCodebook(3, 12, torch.device("cpu"),
                                      native.CorrectionConfig(K_correction=3))
    provenance: dict[int, set[str]] = {}
    for step in range(3):
        _write(cb, step, -0.10, step)
        group = _group(f"original-{step}")
        r1._record_c1_source_group_updates(
            provenance, last_steps=cb.last_step[:cb.n].tolist(),
            event_position=step, event_group=group)

    queried_group = _group("replacement-event")
    cb.observe_verified_feedback(1, queried_group, 9, score_sealed=True)
    _write(cb, 3, 0.20, 9, "FN")
    r1._record_c1_source_group_updates(
        provenance, last_steps=cb.last_step[:cb.n].tolist(),
        event_position=9, event_group=queried_group)

    replaced_slots = torch.nonzero(cb.last_step[:cb.n] == 9, as_tuple=False).view(-1).tolist()
    assert len(replaced_slots) == 1
    replaced_slot = replaced_slots[0]
    assert queried_group in provenance[replaced_slot]
    eligible_same_source = r1._validate_c1_source_groups(
        provenance, cb.last_step[:cb.n].tolist(), event_position=10,
        phase="pre_score", event_group=queried_group)
    assert replaced_slot not in eligible_same_source
    eligible_other_source = r1._validate_c1_source_groups(
        provenance, cb.last_step[:cb.n].tolist(), event_position=10,
        phase="pre_score", event_group=_group("next-event"))
    assert replaced_slot in eligible_other_source
    with pytest.raises(r1.R1CausalityError, match="current/future last_step"):
        r1._validate_c1_source_groups(
            provenance, cb.last_step[:cb.n].tolist(), event_position=9,
            phase="pre_score", event_group=_group("same-step"))


def test_native_c1_codebook_replacement_signature_and_inherited_readout_contract():
    assert issubclass(FPSASignedCorrectionCodebook, native.SignedCorrectionCodebook)
    assert inspect.signature(FPSASignedCorrectionCodebook.update) == \
        inspect.signature(native.SignedCorrectionCodebook.update)
    assert inspect.signature(FPSASignedCorrectionCodebook.apply) == \
        inspect.signature(native.SignedCorrectionCodebook.apply)
    assert FPSASignedCorrectionCodebook.apply is native.SignedCorrectionCodebook.apply
    _, cb = _pair(k=2)
    assert cb.size() == 0
    assert cb.summary()["inserted_total"] == 0
    assert set(cb.update(_feature(0), torch.tensor([-0.10]), outcome="TN", step=0)) == {
        "inserted", "merged", "replaced", "skipped", "same_sign_replaced",
        "cross_sign_replaced", "same_sign_merged",
    }
