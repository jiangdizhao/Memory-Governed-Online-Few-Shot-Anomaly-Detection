"""Synthetic CPU contracts for the frozen R5 audiojack mechanism runner."""
from __future__ import annotations

import hashlib

import pytest

from experiments.calibration.controllers import StateConsistentQuantile
from experiments.calibration.protocol import Observation
from experiments.calibration import r5_audiojack_b3_fpsa_runner as r5


H0 = "0" * 64
H1 = "1" * 64
GROUP = "a" * 64


def _initial_scores(value: float = 0.5) -> list[float]:
    # The higher q90 order statistic at ceil(.9 * (64 - 1)) is index 57.
    return [0.1] * 57 + [value] * 7


def _controller(theta: float = 0.5, name: str = "test") -> StateConsistentQuantile:
    controller = StateConsistentQuantile(
        theta=theta, delta=0.01, min_normals=16, name=name)
    r5._initialize_b3(controller, theta, _initial_scores(theta), GROUP, H0)
    return controller


def test_original_b3_version_zero_initialization_does_not_move_threshold():
    controller = _controller()
    assert controller.theta == 0.5
    assert controller.evidence_version == 0
    assert controller.status == "CURRENT_C64_Q90"
    assert controller.normal_scores == tuple(_initial_scores())


def test_b3_higher_quantile_ties_and_clipping_are_exact():
    controller = _controller()
    tied = [0.1] * 57 + [0.8] * 7
    transition = r5._refresh_b3(
        controller, previous_digest=H0, new_digest=H1, previous_version=0,
        new_scores=tied, c64_group_sha256=GROUP, new_state_digest=H1)
    assert transition["raw_c64_q90"] == 0.8
    assert transition["new_theta"] == pytest.approx(0.51)
    assert transition["movement"] == pytest.approx(0.01)

    transition_down = r5._refresh_b3(
        controller, previous_digest=H1, new_digest="2" * 64, previous_version=1,
        new_scores=[0.2] * 64, c64_group_sha256=GROUP, new_state_digest="2" * 64)
    assert transition_down["new_theta"] == pytest.approx(0.50)
    assert transition_down["movement"] == pytest.approx(-0.01)


def test_no_state_change_means_no_threshold_refresh():
    before = {"G": H0, "GC": H0, "GC-FPSA": H0}
    assert r5._changed_scorers(before, dict(before)) == ()
    controller = _controller()
    state = controller.state_dict()
    assert r5._refresh_b3(
        controller, previous_digest=H0, new_digest=H0, previous_version=0,
        new_scores=[0.9] * 64, c64_group_sha256=GROUP,
        new_state_digest=H0) is None
    assert controller.state_dict() == state


def test_correction_only_change_refreshes_only_corrected_state_streams():
    before = {"G": H0, "GC": H0, "GC-FPSA": H0}
    after = {"G": H0, "GC": H1, "GC-FPSA": "2" * 64}
    assert r5._changed_scorers(before, after) == ("GC", "GC-FPSA")
    gc = _controller(name="GC")
    fpsa = _controller(name="FPSA")
    gc_update = r5._refresh_b3(
        gc, previous_digest=H0, new_digest=H1, previous_version=0,
        new_scores=[0.1] * 57 + [0.9] * 7,
        c64_group_sha256=GROUP, new_state_digest=H1)
    fpsa_update = r5._refresh_b3(
        fpsa, previous_digest=H0, new_digest="2" * 64, previous_version=0,
        new_scores=[0.1] * 57 + [0.1] * 7,
        c64_group_sha256=GROUP, new_state_digest="2" * 64)
    assert gc_update["new_theta"] == pytest.approx(0.51)
    assert fpsa_update["new_theta"] == pytest.approx(0.49)
    assert gc.theta != fpsa.theta
    # G's digest did not change, so its controller receives no update.
    assert _controller().evidence_version == 0


def test_changed_scorer_identity_requires_complete_sha256_digests():
    with pytest.raises(r5.R5AudiojackError):
        r5._changed_scorers({"G": H0, "GC": "bad", "GC-FPSA": H0},
                            {"G": H0, "GC": H1, "GC-FPSA": H0})


def test_snapshot_fails_closed_on_wrong_state_fingerprint_or_c64_group():
    with pytest.raises(r5.R5AudiojackError, match="64 finite current-state"):
        r5._c64_evidence([0.2] * 64, 0, GROUP, "wrong-state")
    with pytest.raises(r5.R5AudiojackError, match="64 finite current-state"):
        r5._c64_evidence([0.2] * 64, 0, "not-a-group", H0)


def test_prediction_requires_matching_version_and_fingerprint_and_has_no_label():
    controller = _controller()
    assert "label" not in Observation.__dataclass_fields__
    prediction = r5._predict_b3(
        controller, position=0, event_id_sha256=H1, camera_id=1,
        score=0.6, version=0, state_digest=H0)
    assert prediction["prediction"] == 1
    assert prediction["threshold"] == 0.5
    with pytest.raises(r5.R5AudiojackError, match="version/fingerprint"):
        r5._predict_b3(
            controller, position=1, event_id_sha256=H1, camera_id=1,
            score=0.6, version=0, state_digest=H1)
    with pytest.raises(TypeError):
        r5._predict_b3(
            controller, position=1, event_id_sha256=H1, camera_id=1,
            score=0.6, version=0, state_digest=H0, label=1)


def test_query_disclosure_gate_requires_current_score_flush():
    gate = r5._ScoreDisclosureGate(7)
    with pytest.raises(r5.R5AudiojackError, match="before score seal"):
        gate.authorize_disclosure(7)
    gate.mark_score_flushed(7)
    gate.authorize_disclosure(7)
    with pytest.raises(r5.R5AudiojackError, match="more than once"):
        gate.authorize_disclosure(7)


def test_corrected_c64_rescore_rejects_same_source_group():
    provenance = {"GC": {1: {GROUP}}, "GC-FPSA": {}}
    with pytest.raises(r5.R5AudiojackError, match="entered GC correction provenance"):
        r5._assert_c64_source_groups_excluded(GROUP, provenance, "GC")
    r5._assert_c64_source_groups_excluded(GROUP, provenance, "GC-FPSA")


def test_frozen_writer_gate_is_independent_of_b3_threshold_and_prediction():
    import numpy as np
    from experiments.calibration.r1_matched_correction_pilot import (
        common_c0_greedy_writer_action,
    )

    ids = np.arange(32, dtype=np.int64)
    candidate_features = np.ones((32, 4), dtype=np.float32)

    def execute(_b3_theta: float, _b3_prediction: int):
        state = []

        def eligibility(**_kwargs):
            return True, 32, "synthetic_allowed", None

        def update(**kwargs):
            state.append((kwargs["position"], kwargs["score"], kwargs["theta"]))
            return {"accepted_patches_delta": 1}

        action = common_c0_greedy_writer_action(
            position=3, c0_score=0.6, theta=0.5, sigma=0.1,
            c0_prediction=r5._frozen_writer_gate(0.6, 0.5),
            candidate_ids=ids, candidate_features=candidate_features,
            feature_layers=(), queried=True, disclosed_label=0,
            eligibility=eligibility, update=update)
        return action, state

    result_a = execute(0.4, 1)
    result_b = execute(0.8, 0)
    assert result_a == result_b
    assert result_a[0]["input_score_source"] == "common_C0"
    assert result_a[1] == [(3, 0.6, 0.5)]

    state = []
    noop = lambda **_kwargs: pytest.fail("defect/unqueried event must not update")
    args = dict(position=4, c0_score=0.6, theta=0.5, sigma=0.1,
                c0_prediction=1, candidate_ids=ids,
                candidate_features=candidate_features, feature_layers=(),
                eligibility=lambda **_kwargs: pytest.fail("defect skips eligibility"),
                update=noop)
    defect = common_c0_greedy_writer_action(
        **args, queried=True, disclosed_label=1)
    unqueried = common_c0_greedy_writer_action(
        **args, queried=False, disclosed_label=None)
    assert defect["action"] == "verified_defect_no_normal_write"
    assert unqueried["eligibility_reason"] == "unqueried_event"


def test_smoke_seal_transition_is_deferred_to_next_event():
    transition = r5._defer_b3_transition(12, "GC", H1, {
        "new_theta": 0.51, "evidence_fingerprint": H0})
    assert transition["apply_from_position"] == 12 + 1


def test_output_budget_refuses_over_budget_write_before_file_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(r5, "OUTPUT_LIMIT", 16)
    monkeypatch.setattr(r5, "FAILURE_RESERVE", 4)
    output = tmp_path / "r5_synthetic"
    budget = r5.R5OutputBudget(output, create=True)
    # The one-byte budget lock plus this payload exactly consumes normal space.
    budget.write_once("attempt_history.jsonl", b"12345678901")
    path = output / "stage_a_scores.jsonl"
    with pytest.raises(r5.R5AudiojackError, match="budget exhausted before write"):
        budget.write_once("stage_a_scores.jsonl", b"x")
    assert not path.exists()
    assert budget.audit()["used_bytes"] == 12


def test_output_path_rejects_existing_historical_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(r5, "OUTPUT_BASE", tmp_path)
    (tmp_path / r5.RUN_ID).mkdir()
    with pytest.raises(FileExistsError, match="restart/resume is forbidden"):
        r5._output_path(r5.RUN_ID)


def test_synthetic_c64_feature_cache_round_trip():
    import numpy as np
    import torch
    from PIL import Image

    image = Image.new("RGB", (3, 2), color=(4, 5, 6))
    features = [torch.arange(12, dtype=torch.float32).reshape(3, 4)]
    payload, receipt = r5._cache_encode(image, features)
    decoded = r5._cache_decode(payload)
    assert decoded["rgb"].shape == (2, 3, 3)
    assert np.array_equal(decoded["features"][0], features[0].numpy())
    assert receipt["payload_sha256"] == hashlib.sha256(payload).hexdigest()


def test_binary_metrics_and_source_group_bootstrap_are_deterministic():
    import numpy as np

    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    predictions = {"a": np.asarray([0, 1, 1, 0], dtype=np.int8),
                   "b": np.asarray([0, 0, 1, 1], dtype=np.int8)}
    metrics = r5._binary_metrics(labels, predictions["a"])
    assert metrics["confusion"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    assert metrics["balanced_accuracy"] == 0.5
    comparisons = {"b-a": ("b", "a")}
    groups = np.asarray(["g1", "g2", "g3", "g4"])
    first = r5._bootstrap_contrasts(predictions, labels, groups, comparisons)
    second = r5._bootstrap_contrasts(predictions, labels, groups, comparisons)
    assert first == second
    assert first["replicates"] == 1000
