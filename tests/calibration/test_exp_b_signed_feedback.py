from __future__ import annotations

from dataclasses import dataclass
import json

import pytest
import torch

from experiments.calibration import exp_b_signed_feedback as expb
from experiments.calibration.r1_matched_correction_pilot import CausalEventOrder, R1CausalityError


@dataclass
class FakeCfg:
    enabled: bool = False


class FakeCodebook:
    def __init__(self, capacity: int, feature_dim: int, device, cfg):
        self.K = capacity
        self.device = torch.device(device)
        self.cfg = cfg
        self.n = 0
        self.protos = torch.zeros((capacity, feature_dim), dtype=torch.float32, device=self.device)
        self.beta = torch.zeros((capacity,), dtype=torch.float32, device=self.device)
        self.radius = torch.ones((capacity,), dtype=torch.float32, device=self.device)
        self.kappa = torch.ones((capacity,), dtype=torch.float32, device=self.device)
        self.count = torch.zeros((capacity,), dtype=torch.int64, device=self.device)
        self.outcome_fp = torch.zeros((capacity,), dtype=torch.int64, device=self.device)
        self.outcome_fn = torch.zeros((capacity,), dtype=torch.int64, device=self.device)
        self.outcome_tn = torch.zeros((capacity,), dtype=torch.int64, device=self.device)
        self.outcome_tp = torch.zeros((capacity,), dtype=torch.int64, device=self.device)
        self.last_step = torch.full((capacity,), -1, dtype=torch.int64, device=self.device)

    def size(self):
        return self.n

    def apply(self, query):
        if not self.n:
            return torch.zeros((query.shape[0],), dtype=query.dtype, device=query.device)
        sim = torch.clamp(query @ self.protos[:self.n].T, -1.0, 1.0)
        weight = torch.exp(-((1.0 - sim) ** 2) / 0.02)
        return (weight * self.beta[:self.n]).sum(dim=1) / (weight.sum(dim=1) + 1e-12)


class FakeImpl:
    SignedCorrectionCodebook = FakeCodebook


def _make_codebook(beta, capacity=None):
    cb = FakeCodebook(capacity or len(beta), 4, "cpu", FakeCfg(enabled=True))
    cb.n = len(beta)
    cb.protos[:cb.n] = torch.eye(4)[:cb.n]
    cb.beta[:cb.n] = torch.tensor(beta, dtype=torch.float32)
    return cb


def test_factorial_sign_policy_and_independent_native_states():
    assert expb.accepted_c1_signs("G", 0) == ()
    assert expb.accepted_c1_signs("G", 1) == ()
    assert expb.accepted_c1_signs("N", 0) == (-1,)
    assert expb.accepted_c1_signs("N", 1) == ()
    assert expb.accepted_c1_signs("D", 0) == ()
    assert expb.accepted_c1_signs("D", 1) == (1,)
    assert expb.accepted_c1_signs("ND", 0) == (-1,)
    assert expb.accepted_c1_signs("ND", 1) == (1,)

    base = FakeCodebook(1024, 4, "cpu", FakeCfg())
    states = expb._make_native_c1_states(object(), FakeImpl, FakeCfg(), base)
    assert states["N"]["cb"] is not states["D"]["cb"]
    assert states["D"]["cb"] is not states["ND"]["cb"]
    states["N"]["cb"].beta[0] = -0.25
    assert float(states["ND"]["cb"].beta[0]) == 0.0
    assert states["N"]["cb"].size() == states["D"]["cb"].size() == states["ND"]["cb"].size() == 0


def test_sign_masked_clone_excludes_unselected_sign_from_readout():
    live = _make_codebook([0.2, -0.1, 0.15])
    cfg = FakeCfg(enabled=False)
    positive = expb.sign_masked_c1_codebook(FakeImpl, live, [0, 1, 2], 1, cfg)
    negative = expb.sign_masked_c1_codebook(FakeImpl, live, [0, 1, 2], -1, cfg)
    assert positive.size() == 2
    assert negative.size() == 1
    assert positive.beta[:positive.size()].tolist() == pytest.approx([0.2, 0.15])
    assert negative.beta[:negative.size()].tolist() == pytest.approx([-0.1])
    assert live.size() == 3
    assert live.beta[:live.size()].tolist() == pytest.approx([0.2, -0.1, 0.15])


def test_stage_a_score_validation_rejects_labels_and_query_drift():
    row = {"schema": expb.SCORE_SCHEMA, "category": "audiojack", "position": 0,
           "scores": {arm: 0.0 for arm in expb.MAIN_ARMS}, "query_tape_decision": False}
    expb.validate_score_rows([row], category="audiojack", expected_count=1,
                             expected_queries=())
    with pytest.raises(expb.ExperimentBError, match="label field"):
        expb.validate_score_rows([{**row, "label": 0}], category="audiojack",
                                 expected_count=1, expected_queries=())
    with pytest.raises(expb.ExperimentBError, match="Query Tape"):
        expb.validate_score_rows([{**row, "query_tape_decision": True}],
                                 category="audiojack", expected_count=1,
                                 expected_queries=())


def test_real_r10_corrected_c2_score_accepts_group_identity_and_returns_finite_score():
    import numpy as np
    from experiments.calibration import r10_signed_knn_baseline as r10

    memory = r10.SignedKNNFIFO(capacity=1024)
    base_map = np.linspace(0.1, 0.9, 784, dtype=np.float32)
    top_ids = np.arange(32, dtype=np.int64)
    selected = np.eye(32, 4096, dtype=np.float32)

    score, delta, diagnostics = r10._corrected_c2_score(
        base_map, top_ids, selected, memory,
        position=7, group_sha256="a" * 64)

    assert np.isfinite(score)
    assert score == pytest.approx(float(np.max(base_map)))
    assert delta.shape == (32,)
    assert diagnostics["eligible_anchor_count"] == 0


def test_state_change_threshold_refresh_updates_real_b3_threshold_and_version():
    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration.controllers import StateConsistentQuantile

    initial_scores = [float(value) for value in range(64)]
    updated_scores = [value + 0.005 for value in initial_scores]
    old_digest, new_digest = "a" * 64, "b" * 64
    group_digest = "c" * 64
    controller = StateConsistentQuantile(57.0, delta=0.01, min_normals=16, name="test-B3")
    r7._initialize_b3(controller, initial_scores, group_digest, old_digest, 57.0)

    transition = expb._refresh_threshold(
        controller, updated_scores, group_digest, old_digest, new_digest, old_version=0)

    assert transition is not None
    assert transition["version_before"] == 0
    assert transition["version_after"] == 1
    assert transition["new_theta"] == pytest.approx(57.005)
    assert controller.theta == pytest.approx(57.005)
    assert controller.evidence_version == 1
    assert controller._r7_state_digest == new_digest


def test_stage_a_startup_failure_receipt_has_traceback_and_is_non_overwriting(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments.calibration import r1_runtime_supervisor as supervisor

    monkeypatch.setattr(expb, "RUN_ROOT", tmp_path)
    stage_root = tmp_path / "expb_test_0001" / "stage_a"
    stage_root.mkdir(parents=True)
    monkeypatch.setattr(supervisor, "current_process_is_in_job", lambda: True)

    def fail_authorization(*_args, **_kwargs):
        raise RuntimeError("synthetic pre-category startup failure")

    monkeypatch.setattr(expb, "_check_run_authorization", fail_authorization)
    args = SimpleNamespace(category="audiojack", run_id="expb_test_0001",
                           authorization_file=tmp_path / "auth.json", experiment_seed=0)
    with pytest.raises(RuntimeError, match="synthetic pre-category startup failure"):
        expb._run_stage_a_child(args, authorized_cuda_visibility=None)

    path = stage_root / "audiojack.startup_failure.json"
    assert path.is_file()
    first = path.read_text(encoding="utf-8")
    receipt = json.loads(first)
    assert receipt["exception_type"] == "RuntimeError"
    assert receipt["message"] == "synthetic pre-category startup failure"
    assert "RuntimeError: synthetic pre-category startup failure" in receipt["traceback"]

    expb._record_stage_a_child_failure(
        "expb_test_0001", "audiojack", ValueError("later error must not replace first"))
    assert path.read_text(encoding="utf-8") == first


def test_stage_b_gate_requires_all_category_seals_before_any_label_projection(tmp_path):
    with pytest.raises(expb.ExperimentBError, match="refuses labels"):
        expb._stage_b_global(tmp_path, {"status": "SEALED_COMPLETE", "categories": {}}, {})


def test_causal_guard_requires_score_seal_and_query_before_reveal_and_update():
    gate = CausalEventOrder(3)
    with pytest.raises(R1CausalityError, match="before score seal"):
        gate.disclose(lambda: 1)
    gate.seal_scores()
    assert gate.decide_query(True)
    assert gate.disclose(lambda: 1) == 1
    gate.corrections_updated()
    gate.memory_updated()
    gate.finish()
    assert gate.phase == "complete"


def test_source_group_bootstrap_is_seeded_and_resamples_whole_groups():
    rows = []
    labels = []
    for group in range(8):
        label = int(group >= 4)
        for view in range(5):
            value = float(group) + view * 0.001
            rows.append({"source_group_sha256": f"{group:064x}",
                         "scores": {"G": value, "N": value + 0.01,
                                    "D": value - 0.01, "ND": value + 0.02}})
            labels.append(label)
    left = expb._source_group_bootstrap(labels, rows, seed=17, replicates=40)
    right = expb._source_group_bootstrap(labels, rows, seed=17, replicates=40)
    assert left == right
    assert left["valid_replicates"] > 0
    assert left["method"].startswith("paired bootstrap of whole official Source Groups")
