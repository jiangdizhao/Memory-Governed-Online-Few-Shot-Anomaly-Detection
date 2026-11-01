"""CPU-only causal and native-writer tests for R4-FPSA audiojack."""
from __future__ import annotations

import hashlib

import pytest
import torch

import test_phase3_AIF_simplified_v22 as native
from experiments.calibration.r4_fpsa_codebook import FPSASignedCorrectionCodebook
from experiments.calibration.r4_fpsa_audiojack_runner import _json_projection


def _group(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _assert_native_state_equal(left, right) -> None:
    assert left.n == right.n
    for name in ("protos", "beta", "radius", "kappa", "count", "outcome_fp",
                 "outcome_fn", "outcome_tn", "outcome_tp", "last_step"):
        torch.testing.assert_close(getattr(left, name), getattr(right, name), rtol=0, atol=0)
    assert left.counters == right.counters


def _event_features(generator: torch.Generator, *, dim: int = 512) -> torch.Tensor:
    values = torch.randn((4, dim), generator=generator, dtype=torch.float32)
    return torch.nn.functional.normalize(values, dim=1)


def test_audiojack_74_normal_then_40_anomalous_feedback_preflight():
    """Exercise the frozen sign-history shape, not just a short quota toy case."""
    torch.set_num_threads(1)
    capacity, dim = 256, 512
    native_cfg = native.CorrectionConfig(enabled=True, K_correction=capacity)
    fpsa_cfg = native.CorrectionConfig(enabled=True, K_correction=capacity)
    native_cb = native.SignedCorrectionCodebook(capacity, dim, torch.device("cpu"), native_cfg)
    fpsa_cb = FPSASignedCorrectionCodebook(capacity, dim, torch.device("cpu"), fpsa_cfg)
    generator = torch.Generator(device="cpu").manual_seed(20261102)
    normal_only_occupancy = []

    # The actual 74 distinct queried-normal groups arrive first. Four eligible
    # patches per feedback create 296 writes, enough to fill K=256.
    for step in range(74):
        group = _group(f"audiojack-normal-{step}")
        fpsa_cb.observe_verified_feedback(0, group, step, score_sealed=True)
        features = _event_features(generator, dim=dim)
        beta = torch.full((4,), -0.10, dtype=torch.float32)
        left = native_cb.update(features, beta, outcome="TN", step=step)
        right = fpsa_cb.update(features, beta, outcome="TN", step=step)
        assert right == left
        _assert_native_state_equal(native_cb, fpsa_cb)
        diag = fpsa_cb.diagnostics()
        assert diag["target_positive_capacity"] == 0
        assert diag["actual_positive_occupancy"] == 0
        assert diag["quota_borrow_opportunities"] == 0
        normal_only_occupancy.append(fpsa_cb.size())

    assert max(normal_only_occupancy) == capacity
    assert fpsa_cb.diagnostics()["queried_normal_source_groups"] == 74
    assert fpsa_cb.diagnostics()["queried_anomalous_source_groups"] == 0

    # The first anomalous feedback uses both histories: half-up(256 * 1 / 75)=3,
    # rather than an anomaly-only target of 256. Registration follows the seal.
    for anomaly_index in range(40):
        step = 74 + anomaly_index
        group = _group(f"audiojack-anomaly-{anomaly_index}")
        feedback = fpsa_cb.observe_verified_feedback(1, group, step, score_sealed=True)
        if anomaly_index == 0:
            assert feedback["queried_normal_source_groups"] == 74
            assert feedback["queried_anomalous_source_groups"] == 1
            assert feedback["target_positive_capacity"] == 3
            assert feedback["target_negative_capacity"] == 253
        features = _event_features(generator, dim=dim)
        beta = torch.full((4,), 0.20, dtype=torch.float32)
        native_cb.update(features, beta, outcome="FN", step=step)
        fpsa_cb.update(features, beta, outcome="FN", step=step)
        assert native_cb.size() <= capacity and fpsa_cb.size() <= capacity
        assert fpsa_cb.diagnostics()["actual_positive_occupancy"] + \
            fpsa_cb.diagnostics()["actual_negative_occupancy"] == fpsa_cb.size()
        if anomaly_index >= 1:
            # Native sign-aware retention can replace within an admitted sign,
            # but does not progressively allocate more slots to the new sign.
            assert native_cb.summary()["positive"] == 1

    native_state = native_cb.summary()
    fpsa_state = fpsa_cb.diagnostics()
    assert (fpsa_state["queried_normal_source_groups"],
            fpsa_state["queried_anomalous_source_groups"],
            fpsa_state["queried_source_groups_total"]) == (74, 40, 114)
    assert fpsa_state["target_positive_capacity"] == 90
    assert fpsa_state["target_negative_capacity"] == 166
    assert fpsa_state["actual_positive_occupancy"] + fpsa_state["actual_negative_occupancy"] <= capacity
    assert native_state["positive"] == 1
    assert fpsa_state["quota_borrow_opportunities"] > 0
    assert fpsa_state["quota_cross_sign_replacements"] > 0
    assert fpsa_state["quota_borrow_rejected"] >= 0
    assert fpsa_state["native_cross_sign_replaced_total"] >= \
        fpsa_state["quota_cross_sign_replacements"]


def test_native_patchvmb_writer_delivers_original_gated_fpsa_update_inputs():
    torch.set_num_threads(1)
    capacity, dim = 8, 64
    cfg = native.CorrectionConfig(enabled=True, K_correction=capacity)

    class RecordingFPSACodebook(FPSASignedCorrectionCodebook):
        def update(self, q, beta, **kwargs):
            self.recorded = {
                "q": q.detach().clone(), "beta": beta.detach().clone(),
                "outcome": kwargs.get("outcome"), "step": kwargs.get("step"),
            }
            return super().update(q, beta, **kwargs)

    bank = native.PatchVMB(
        device="cpu", K_normal_fixed=2, K_normal_dyn=256, K_normal_ltm=1,
        K_defect=1, K_concept=1, knn_k_normal=1, corr_cfg=cfg,
        use_defect_bank=False)
    features = torch.zeros((32, dim), dtype=torch.float32)
    for row in range(32):
        features[row, row + 1] = 1.0
    bank._ensure_mode_view_normals("p0", 0, "id", [features])
    fixed = bank.normal_fixed["p0"][0]["id"][0]
    fixed.set_prototypes(torch.nn.functional.one_hot(torch.tensor([0]), dim).float())
    # Replace only the correction Codebook slot with the recording FPSA subtype;
    # the actual frozen PatchVMB writer computes the gate and beta values.
    recording = RecordingFPSACodebook(capacity, dim, torch.device("cpu"), cfg)
    bank.corr_cb["p0"]["id"][0] = recording
    ids = list(range(32))
    outcome = "FP"
    step = 7
    result = bank.update_corrections_from_topk(
        "p0", [features], mode=0, view="id", layer_to_patch_idx={0: ids},
        sign=-1, score_img=1.2, theta=1.0, outcome=outcome, step=step)

    assert result["layers"]["0"]["kept"] == 32
    assert recording.recorded["outcome"] == outcome
    assert recording.recorded["step"] == step
    assert recording.recorded["q"].shape == (32, dim)
    torch.testing.assert_close(recording.recorded["q"], features, rtol=0, atol=0)
    torch.testing.assert_close(recording.recorded["beta"],
                               torch.full((32,), -0.13), rtol=0, atol=1e-6)
    assert recording.size() <= capacity


def test_reference_writer_projection_skips_label_value():
    line = (b'{"record_type":"queried_feedback_and_updates","category":"audiojack",'
            b'"position":4,"label":"must-not-be-decoded",'
            b'"greedy_writer_action":{"action":"verified_defect_no_normal_write"}}')
    projection = _json_projection(
        line, {"record_type", "category", "position", "greedy_writer_action"})
    assert projection == {
        "record_type": "queried_feedback_and_updates", "category": "audiojack",
        "position": 4,
        "greedy_writer_action": {"action": "verified_defect_no_normal_write"},
    }


@pytest.mark.parametrize("mutation", [None])
def test_full_run_identity_constants_are_single_category_and_fixed(mutation):
    from experiments.calibration import r4_fpsa_audiojack_runner as runner
    assert runner.CATEGORY == "audiojack"
    assert runner.EVENTS == 3815
    assert runner.QUERIES == 114
    assert runner.FIRST_ANOMALOUS_QUERY == 1305
    assert runner.RUNTIME_SECONDS == 120 * 60


def test_orchestrator_requires_explicit_authorization():
    from experiments.calibration import r4_fpsa_audiojack_runner as runner
    with pytest.raises(runner.R4AudiojackError, match="Explicit --authorize"):
        runner.main(["--run-orchestrator"])


def test_authorized_orchestrator_dispatches_without_child_token(monkeypatch):
    from experiments.calibration import r4_fpsa_audiojack_runner as runner
    calls = []
    monkeypatch.setattr(runner, "run_experiment", lambda: calls.append("run"))
    assert runner.main(["--authorize-r4-audiojack", "--run-orchestrator"]) == 0
    assert calls == ["run"]
