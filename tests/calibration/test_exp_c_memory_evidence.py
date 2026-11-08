from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.calibration.e6_bea_closed_loop import (
    ArmRuntime,
    EventIdentity,
    FeedbackUpdate,
    NativePreEventScore,
)
from experiments.calibration.e6_bea_d2 import D2PolicyController
from experiments.calibration.e6_bea_policy import compute_bea_components
from experiments.calibration.exp_c_memory_evidence import (
    ARM_IDS,
    AverageGate,
    CPrivateFeedbackSink,
    CScoreLedger,
    CStageAClosedLoop,
    CBEAPriceController,
    analyze_metrics,
    c_objective,
    classify_budget,
    build_stage_b_metrics,
    gate_decomposition,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _components(score: float = -0.01):
    return compute_bea_components(
        score=score, threshold=0.0,
        c64_scores=np.linspace(-0.1, 0.1, 64),
        novelty_percentile=0.5,
        correction_top32=np.full(32, 0.01), correction_bank_size=1,
        topk_dN_max=0.2, topk_dN_mean=0.1)


class _Coverage:
    def __init__(self):
        self.observed = 0

    def novelty_percentile(self, _feature):
        return 0.5, {"distance_nn": 0.2, "synthetic": True}

    def state_sha256(self):
        return _sha(f"coverage-{self.observed}")

    def observe_normal(self, _feature):
        self.observed += 1


class _Engine:
    def __init__(self):
        self.current_threshold = 0.0
        self.state_version = 0
        self.state_digest = _sha("engine-0")
        self.initial_state_sha256 = self.state_digest
        self.correction_cb = type("Correction", (), {"size": lambda _self: 1})()
        self.writer = type("Writer", (), {"audit": lambda _self: {"accepted_patches": 1}})()
        self.input_identity = {}

    def score(self, event, _encoded):
        value = -0.01 + 0.00001 * event.position
        return NativePreEventScore(
            gc_score=value, c0_score=value, c64_scores=tuple(np.linspace(-0.1, 0.1, 64)),
            correction_top32=tuple(np.full(32, 0.01)), correction_bank_size=1,
            topk_dN_max=0.2, topk_dN_mean=0.1,
            scorer_state_version=self.state_version,
            scorer_state_sha256=self.state_digest)

    def observe_unqueried(self, _event, _encoded, _pre):
        return {}

    def apply_feedback(self, _event, _encoded, _pre, label):
        self.state_version += 1
        self.state_digest = _sha(f"engine-{self.state_version}")
        return FeedbackUpdate(label=label, c1_changed=True, greedy_changed=True,
                              scorer_state_version=self.state_version,
                              scorer_state_sha256=self.state_digest,
                              diagnostics={"synthetic": True}, scorer_state_changed=True)


class _Reader:
    def __init__(self, arm_id, ledger, events):
        self.arm_id, self.ledger, self.events = arm_id, ledger, tuple(events)
        self.calls = []

    def label_for(self, event, receipt):
        assert self.ledger.is_sealed(event.position)
        assert self.ledger.authorize_receipt(receipt, self.arm_id, event)
        self.calls.append(event.position)
        return 0

    def close(self):
        pass


def test_frozen_D_reconstructs_from_band_and_current_evidence():
    components = _components()
    values = gate_decomposition(components)
    assert np.isclose(values["D_reconstructed"], components.D, rtol=0, atol=1e-12)
    assert np.isclose(values["g_sample"], 1.0 - components.correction_evidence)
    assert c_objective("BN_BAND", components, values["D_band"], 0.4) == (
        components.B + components.N + values["D_band"])
    assert c_objective("BN_AVG_GATE", components, values["D_band"], 0.4) == (
        components.B + components.N + 0.4 * values["D_band"])
    assert c_objective("BND", components, values["D_band"], 0.4) == (
        components.B + components.N + components.D)


def test_average_gate_is_empty_lagged_weighted_and_bounded_to_256():
    gate = AverageGate()
    assert gate.value() == 1.0
    decision_gate = gate.value()
    gate.observe_after_feedback(0.25, 0.2)
    assert decision_gate == 1.0
    assert np.isclose(gate.value(), 0.8)
    gate.observe_after_feedback(0.75, 0.6)
    assert len(gate.history) == 2
    assert np.isclose(gate.value(), (0.25 * 0.8 + 0.75 * 0.4) / 1.0)
    for i in range(300):
        gate.observe_after_feedback(1.0, (i % 2) / 2.0)
    assert len(gate.history) == 256
    zero = AverageGate()
    for _ in range(4):
        zero.observe_after_feedback(0.0, 0.5)
    assert zero.value() == 1.0
    left, right = AverageGate(), AverageGate()
    left.observe_after_feedback(0.5, 0.0)
    right.observe_after_feedback(0.5, 1.0)
    assert left.value() == 1.0 and right.value() == 0.0


def test_existing_d2_controllers_keep_their_registered_proposals():
    components = _components()
    for arm in ("RAND37", "MARGIN37", "B", "BD", "BND"):
        left, right = D2PolicyController(arm, seed=37), D2PolicyController(arm, seed=37)
        for position in range(6):
            event_id = _sha(f"event-{position}")
            p1 = left.propose(position=position, event_id_sha256=event_id,
                              components=components, novelty_distance_nn=0.2)
            p2 = right.propose(position=position, event_id_sha256=event_id,
                               components=components, novelty_distance_nn=0.2)
            assert p1 == p2
            left.observe_final_action(position=position, proposal=p1,
                                      executed_query=p1.proposed, group_suppressed=False)
            right.observe_final_action(position=position, proposal=p2,
                                       executed_query=p2.proposed, group_suppressed=False)
        assert left.diagnostics() == right.diagnostics()


def test_synthetic_seven_arm_event_loop_seals_before_disclosure_and_lags_gate(tmp_path: Path):
    group = _sha("one-official-source-group")
    events = tuple(EventIdentity(
        position=i, original_position=100 + i,
        event_id_sha256=_sha(f"event-{i}"), image_identity_sha256=_sha(f"image-{i}"),
        source_group_sha256=group, relative_id=f"audiojack/{i}.png", camera_id=str(i),
        timestamp=f"t{i}", image_path=f"synthetic/{i}.png") for i in range(5))
    run_dir = tmp_path / "category"
    run_dir.mkdir()
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text('{"synthetic":true}\n', encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    ledger = CScoreLedger(run_dir / "sealed_scores.jsonl")
    arms = {}
    for arm_id in ARM_IDS:
        reader = _Reader(arm_id, ledger, events)
        sink = CPrivateFeedbackSink(run_dir / f"private_feedback_{arm_id}.jsonl", arm_id)
        controller = (CBEAPriceController(arm_id) if arm_id in {"BN_BAND", "BN_AVG_GATE"}
                      else D2PolicyController(arm_id, seed=37))
        arms[arm_id] = ArmRuntime(arm_id, _Engine(), controller, _Coverage(), reader, sink)

    coordinator = CStageAClosedLoop(
        category="audiojack", run_id="synthetic", events=events,
        feature_encoder=lambda event: type("Encoded", (), {
            "global_cls": np.asarray([float(event.position + 1), 1.0]),
            "encoding_seconds": 0.0})(),
        arms=arms, ledger=ledger, run_dir=run_dir,
        manifest_path=manifest_path, manifest_sha256=manifest_sha,
        authorization_id="synthetic-only", expected_event_count=5)
    seal = coordinator.run()
    rows = [json.loads(line) for line in seal.score_log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    assert rows[0]["arm_rows"]["BN_AVG_GATE"]["g_bar_lagged"] == 1.0
    assert rows[1]["arm_rows"]["BN_AVG_GATE"]["g_bar_lagged"] < 1.0
    assert len({id(value) for value in coordinator.average_gates.values()}) == len(ARM_IDS)
    for arm_id in ARM_IDS:
        executed = sum(bool(row["arm_rows"][arm_id]["query_executed"]) for row in rows)
        assert len(arms[arm_id].label_reader.calls) == executed
    assert seal.arm_query_counts["B"] == 1
    assert seal.arm_controller_diagnostics["B"]["group_suppressed"] >= 1
    assert rows[1]["arm_rows"]["B"]["query_executed"] is False
    assert rows[0]["arm_rows"]["BND"]["within_bnd_state_diagnostic"]["read_only"] is True
    assert not any("label" in row for row in rows)
    assert all({"prediction", "gc_score", "query_executed"}.issubset(row["arm_rows"][arm])
               for row in rows for arm in ARM_IDS)
    replay = {arm: D2PolicyController(arm, seed=37)
              for arm in ("RAND37", "MARGIN37", "B", "BD", "BND")}
    for row in rows:
        event = events[row["position"]]
        for arm, controller in replay.items():
            arm_row = row["arm_rows"][arm]
            components = compute_bea_components(
                score=arm_row["components"]["score"],
                threshold=arm_row["components"]["threshold"],
                c64_scores=np.linspace(-0.1, 0.1, 64), novelty_percentile=0.5,
                correction_top32=np.full(32, 0.01), correction_bank_size=1,
                topk_dN_max=0.2, topk_dN_mean=0.1)
            proposal = controller.propose(
                position=event.position, event_id_sha256=event.event_id_sha256,
                components=components,
                novelty_distance_nn=float(arm_row["novelty_info"]["distance_nn"]))
            assert (proposal.proposed, proposal.reason, proposal.objective,
                    proposal.query_cost, proposal.random_u, proposal.tie_u) == (
                arm_row["query_proposed"], arm_row["query_reason"], arm_row["query_objective"],
                arm_row["query_cost_runtime"], arm_row["query_random_u"], arm_row["query_tie_u"])
            controller.observe_final_action(
                position=event.position, proposal=proposal,
                executed_query=arm_row["query_executed"],
                group_suppressed=arm_row["group_suppressed"])


def test_contrast_sign_budget_classification_and_rate_tolerance():
    metrics = {}
    for category in ("audiojack", "regulator", "switch", "rolled_strip_base"):
        metrics[category] = {}
        for arm in ARM_IDS:
            ber = {"B": 0.20, "BD": 0.18, "BN_BAND": 0.17,
                   "BN_AVG_GATE": 0.16, "BND": 0.15,
                   "RAND37": 0.22, "MARGIN37": 0.21}[arm]
            rate = 0.03 if arm in {"B", "BD", "BN_BAND", "BN_AVG_GATE", "BND"} else 0.037
            metrics[category][arm] = {"ber": ber, "query_rate": rate, "ba": 1 - ber,
                                      "fpr": ber, "fnr": ber, "auroc": 0.7,
                                      "ap": 0.5, "query_count": 10}
    result = analyze_metrics(metrics)
    contrast = result["contrasts"]["CATEGORY_EQUAL_MACRO"]["BN_AVG_GATE_minus_BND"]
    assert np.isclose(contrast["ber_difference_left_minus_right"], 0.01)
    assert contrast["budget_classification"] == "APPROXIMATELY_BUDGET_MATCHED"
    assert classify_budget(rate_left=0.02, rate_right=0.03,
                           ber_left=0.14, ber_right=0.15) == "LEFT_DOMINATES_QUALITY_AND_EXPENDITURE"


def test_synthetic_stage_b_metrics_and_paired_source_group_bootstrap():
    labels, rows_by_category, query_counts = {}, {}, {}
    for category in ("audiojack", "regulator", "switch", "rolled_strip_base"):
        y = np.asarray([0, 1] * 4, dtype=np.int8)
        rows = []
        query_counts[category] = {arm: 4 for arm in ARM_IDS}
        for position, label in enumerate(y):
            arm_rows = {}
            for arm in ARM_IDS:
                pred = int(label if arm != "B" else 1 - label)
                score = 1.0 if pred else -1.0
                arm_rows[arm] = {
                    "gc_score": score, "b3_threshold": 0.0, "prediction": pred,
                    "query_executed": position % 2 == 0,
                    "group_suppressed": False, "scorer_state_version": 0,
                    "correction_bank_size": 0, "D_band": 0.2,
                    "g_bar_lagged": 0.8,
                    "components": {"pred_defect": bool(pred), "correction_evidence": 0.5},
                    "within_bnd_state_diagnostic": {
                        "read_only": True, "price": 0.8,
                        "sample_specific_objective": 0.7,
                        "band_objective": 0.8,
                        "average_gate_objective": 0.75,
                        "sample_specific_proposal": False,
                        "band_proposal": False, "average_gate_proposal": False,
                        "band_minus_sample": 0.1, "average_minus_sample": 0.05},
                }
            rows.append({"position": position,
                         "source_group_sha256": _sha(f"{category}-group-{position // 2}"),
                         "arm_rows": arm_rows})
        labels[category] = y
        rows_by_category[category] = rows
    result = build_stage_b_metrics(labels, rows_by_category, query_counts, bootstrap_draws=100)
    assert result["per_category"]["audiojack"]["B"]["ber"] == 1.0
    assert result["per_category"]["audiojack"]["BND"]["queried_normal_count"] == 4
    assert result["mechanism_diagnostics"]["counterfactual_queries_executed"] is False
    interval = result["paired_source_group_bootstrap"][
        "category_equal_macro_contrasts"]["BN_AVG_GATE_minus_BND"]
    assert interval["valid_draws"] == 100
    assert result["paired_source_group_bootstrap"]["independent_online_histories_claimed"] is False
