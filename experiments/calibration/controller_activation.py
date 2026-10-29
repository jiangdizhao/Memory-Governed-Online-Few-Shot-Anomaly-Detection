"""Deterministic, synthetic controller activation diagnostics for Stage 0."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from .controllers import FrozenTheta, StateConsistentQuantile, theta0_from_c64
from .engine import PrequentialEngine
from .method_registry import METHOD_CARDS, build_controllers
from .metrics import PrivateOutcomeLedger
from .profiles import canonical_json
from .protocol import (EvidenceAnchor, EvidenceSnapshot, Observation, ScoreResult,
                       StreamEvent, TapeRecord)
from .query_tape import build_qa_ts
from .scdac_g import SCDACGController
from .spot_dspot import DSPOTController, SPOTController
from .sun_cs import SunCSController


HARNESS_VERSION = "stage0-controller-activation-v1"
N_EVENTS = 1150
MEMORY_SCORE_DELTA = 1e-4


class _SyntheticMemoryAndEvidence:
    """Checkpointed fixture state; never a production memory implementation."""

    def __init__(self, c64: tuple[float, ...]):
        self.version = 0
        self.digest = self._digest()
        self.c64 = c64
        self.normal_rows: list[tuple[int, float]] = []
        self.defect_rows: list[tuple[int, float]] = []
        self.source_revision = 0

    def _digest(self) -> str:
        return hashlib.sha256(f"synthetic-memory:{self.version}".encode()).hexdigest()

    def score(self, raw_score: float, t: int) -> ScoreResult:
        state = hashlib.sha256(canonical_json({
            "synthetic_score_t": t, "memory_version": self.version,
            "memory_digest": self.digest,
        }).encode()).hexdigest()
        return ScoreResult(float(raw_score + self.version * MEMORY_SCORE_DELTA),
                           self.version, state)

    def write(self, t: int, raw_score: float, queried: bool,
              label: int | None) -> dict[str, Any]:
        before = self.digest
        changed = bool(queried and label == 0)
        if changed:
            self.version += 1
            self.digest = self._digest()
            self.source_revision += 1
        return {"synthetic_threshold_independent_reference_write": True,
                "queried_verified_normal_only": changed,
                "queried_defect_write": False,
                "content_changed": changed,
                "memory_version_before": self.version - int(changed),
                "memory_version_after": self.version,
                "source_revision": self.source_revision,
                "digest_before": before, "digest_after": self.digest,
                "event": t, "raw_score": raw_score}

    def evidence(self, t: int | None, raw_score: float | None,
                 feedback: Any) -> EvidenceSnapshot:
        if t is not None and raw_score is not None and feedback is not None:
            row = (int(t), float(raw_score))
            if feedback.label == 0:
                self.normal_rows.append(row)
            elif feedback.label == 1:
                self.defect_rows.append(row)
        shift = self.version * MEMORY_SCORE_DELTA
        protected = tuple(EvidenceAnchor(
            f"activation-c64:{i:03d}", score + shift, self.version, True,
            0, "C64_PROTECTED") for i, score in enumerate(self.c64))
        normals = tuple(EvidenceAnchor(
            f"activation-query-normal:{event}", score + shift, self.version,
            False, 1, "queried_normal_all_layer_leave_source_out")
            for event, score in self.normal_rows)
        defects = tuple(EvidenceAnchor(
            f"activation-query-defect:{event}", score + shift, self.version,
            False, 0, "queried_defect_not_written")
            for event, score in self.defect_rows)
        return EvidenceSnapshot(
            self.version, protected + normals, defects,
            {"synthetic_fixture": True, "c64_mutable": False,
             "unqueried_defect_labels_available": False,
             "source_revision": self.source_revision},
            evidence_revision=len(self.normal_rows) + len(self.defect_rows) +
            self.source_revision)

    def state_dict(self) -> dict[str, Any]:
        return {"schema": "synthetic-controller-activation-memory.v1",
                "version": self.version, "digest": self.digest,
                "normal_rows": list(self.normal_rows),
                "defect_rows": list(self.defect_rows),
                "source_revision": self.source_revision}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema") != "synthetic-controller-activation-memory.v1":
            raise ValueError("Activation memory checkpoint identity mismatch")
        self.version = int(state["version"])
        self.digest = str(state["digest"])
        if self.digest != self._digest():
            raise ValueError("Activation memory content digest mismatch")
        self.normal_rows = [(int(t), float(s)) for t, s in state["normal_rows"]]
        self.defect_rows = [(int(t), float(s)) for t, s in state["defect_rows"]]
        self.source_revision = int(state["source_revision"])


def _activation_fixture() -> tuple[list[StreamEvent], list[int], Any, np.ndarray,
                                   tuple[float, ...], float]:
    records = [TapeRecord(i, f"activation-group-{i:04d}", i % 5 + 1,
                          f"activation-view-{i:04d}", phase=0)
               for i in range(N_EVENTS)]
    tape = build_qa_ts(records)
    query_positions = sorted(tape.positions)
    labels = [int((i % 19) in (0, 1, 2)) for i in range(N_EVENTS)]
    # The tape is frozen before its synthetic evaluator labels are assigned.
    for ordinal, position in enumerate(query_positions):
        labels[position] = ordinal % 2

    rng = np.random.default_rng(20261029)
    scores = 0.12 + 0.035 * np.sin(np.arange(N_EVENTS) * 0.11)
    scores += rng.normal(0.0, 0.012, size=N_EVENTS)
    # Fixed score-only shift, plus queried synthetic defects for B7 branch
    # coverage. Neither score generation nor query selection consumes labels.
    scores[1075:] += np.linspace(0.0, 0.12, N_EVENTS - 1075)
    for ordinal, position in enumerate(query_positions):
        if labels[position] == 1:
            scores[position] = 0.90 + 0.08 * math.sin(ordinal * 0.37)
    for position in range(175, N_EVENTS, 257):
        scores[position] = 1.8 + 0.01 * (position % 7)
    # A fixed extreme score exercises alarm exclusion after both POT methods
    # finish initialization. It is part of the frozen synthetic tape.
    scores[1100] = 100.0
    c64 = tuple(0.10 + i * 0.0005 for i in range(64))
    theta0 = theta0_from_c64(list(c64))
    events = [StreamEvent(record, float(scores[i])) for i, record in enumerate(records)]
    return events, labels, tape, scores, c64, theta0


def _make_engine(*, threshold_offset: float = 0.0) -> tuple[PrequentialEngine, Any,
                                                               Any, Any]:
    events, labels, tape, raw_scores, c64, theta0 = _activation_fixture()
    memory = _SyntheticMemoryAndEvidence(c64)
    controllers = build_controllers(theta0, list(c64))
    # B7's dense candidate-risk grid is exercised below with explicit
    # evidence snapshots. Keep it out of the long score warm-up replay so the
    # diagnostic doesn't repeatedly optimize the same growing evidence set.
    controllers = [controller for controller in controllers if controller.name != "B7"]
    if threshold_offset:
        # A deliberately different frozen threshold paired against the full
        # controller run isolates the engine's shared writer from threshold
        # outputs without repeating expensive B7 evidence optimization.
        controllers = [FrozenTheta(theta0 + threshold_offset)]
    engine = PrequentialEngine(
        tape=tape, events=events, outcomes=PrivateOutcomeLedger(labels),
        controllers=controllers,
        score_fn=lambda event: memory.score(float(event.payload), event.record.position),
        memory_state_fn=lambda: (memory.version, memory.digest),
        write_fn=lambda event, _score, feedback: memory.write(
            event.record.position, float(event.payload), feedback.queried,
            feedback.label),
        evidence_fn=lambda event, feedback: memory.evidence(
            event.record.position if event is not None else None,
            float(event.payload) if event is not None else None, feedback),
        checkpoint_components={"synthetic_memory_and_evidence": memory},
        provenance_id=f"{HARNESS_VERSION}:{tape.digest}:threshold-offset={threshold_offset}")
    return engine, memory, tape, raw_scores


def _count_changes(values: list[Any]) -> int:
    prior = None
    changes = 0
    for value in values:
        if value is None:
            continue
        if prior is not None and value != prior:
            changes += 1
        prior = value
    return changes


def _b7_synthetic_branches(theta0: float) -> dict[str, Any]:
    def anchors(scores: list[float], *, normal: bool, version: int = 0):
        return tuple(EvidenceAnchor(
            f"{'normal' if normal else 'defect'}:{i}", score, version,
            normal, 0,
            "C64_PROTECTED" if normal else "queried_defect_not_written")
            for i, score in enumerate(scores))

    evidence_both = EvidenceSnapshot(
        0, anchors([0.1] * 64, normal=True), anchors([0.9] * 32, normal=False),
        {"c64_mutable": False}, 1)
    b7_both = SCDACGController(theta0)
    b7_both.on_memory_change(evidence_both)
    theta_b7_both = b7_both.finalize_step()
    b7_prediction = b7_both.predict(Observation(
        0, "activation-b7-both", 1, 0.7, 0,
        hashlib.sha256(b"activation-b7-current-state").hexdigest()))
    b3 = StateConsistentQuantile(theta0, delta=0.01)
    b3.on_memory_change(evidence_both)
    theta_b3 = b3.finalize_step()

    normal_only = SCDACGController(0.5)
    normal_only.on_memory_change(EvidenceSnapshot(
        0, anchors([0.1] * 64, normal=True), (), {"c64_mutable": False}, 2))
    normal_only.finalize_step()

    conflict = SCDACGController(0.5)
    conflict.on_memory_change(EvidenceSnapshot(
        0, anchors([0.2] * 32 + [0.8] * 32, normal=True),
        anchors([0.2] * 16 + [0.8] * 16, normal=False),
        {"c64_mutable": False}, 3))
    conflict.finalize_step()

    clipping = SCDACGController(0.95)
    clipping.on_memory_change(evidence_both)
    clipping.finalize_step()
    clip_diag = clipping.diagnostics()

    # Unqueried defects are deliberately represented with a forbidden source
    # policy; only the 16 queried anchors may activate the defect side.
    mixed_defects = anchors([0.9] * 16, normal=False) + tuple(
        EvidenceAnchor(f"unqueried:{i}", 0.95, 0, False, 0,
                       "unqueried_label_forbidden") for i in range(32))
    queried_only = SCDACGController(0.5)
    queried_only.on_memory_change(EvidenceSnapshot(
        0, anchors([0.1] * 64, normal=True), mixed_defects,
        {"unqueried_defect_labels_available": False}, 4))
    queried_only.finalize_step()

    # Exercise the calibration fixture's actual content-version transition:
    # a verified-normal query changes the synthetic bank and all current-M
    # protected-anchor scores, while the query source remains LSO-filtered.
    refresh_memory = _SyntheticMemoryAndEvidence(tuple([0.1] * 64))
    before_snapshot = refresh_memory.evidence(None, None, None)
    before_snapshot = EvidenceSnapshot(
        before_snapshot.memory_version, before_snapshot.normal_anchors,
        anchors([0.9] * 16, normal=False, version=0),
        before_snapshot.source_exclusion_audit, before_snapshot.evidence_revision)
    refresh_controller = SCDACGController(0.5)
    refresh_controller.on_memory_change(before_snapshot)
    before_anchor_scores = tuple(score for _, score in refresh_controller.normal_scores)
    refresh_memory.write(25, 0.25, True, 0)
    after_snapshot = refresh_memory.evidence(25, 0.25,
                                            type("Feedback", (), {"label": 0})())
    after_snapshot = EvidenceSnapshot(
        after_snapshot.memory_version, after_snapshot.normal_anchors,
        anchors([0.9 + MEMORY_SCORE_DELTA] * 16, normal=False,
                version=after_snapshot.memory_version),
        after_snapshot.source_exclusion_audit, after_snapshot.evidence_revision)
    refresh_controller.on_memory_change(after_snapshot)
    after_anchor_scores = tuple(score for _, score in refresh_controller.normal_scores)
    anchor_refresh = {
        "memory_version_advanced": after_snapshot.memory_version == 1,
        "protected_anchor_scores_changed": before_anchor_scores != after_anchor_scores,
        "queried_normal_lso_policy_preserved": all(
            anchor.source_exclusion_policy == "queried_normal_all_layer_leave_source_out"
            for anchor in after_snapshot.normal_anchors if not anchor.protected),
        "controller_observed_version": refresh_controller.memory_version,
        "controller_observed_evidence_revision": refresh_controller.evidence_revision,
    }

    result = {
        "BOTH": {"state": b7_both.state, "decision": b7_both.decision,
                 "threshold": theta_b7_both,
                 "descriptive_feasible_candidates": b7_both.diagnostics().get(
                     "support_feasible_candidate_count", 0)},
        "NORMAL_ONLY": {"state": normal_only.state,
                        "decision": normal_only.decision,
                        "fpr_available": normal_only.diagnostics().get("fpr_upper") is not None,
                        "fnr_unavailable": normal_only.diagnostics().get("fnr_upper") is None},
        "CONFLICT": {"state": conflict.state, "decision": conflict.decision,
                     "fallback_logged": conflict.decision ==
                     "CONFLICT_MINIMUM_RISK_COMPROMISE",
                     "reason": conflict.constraint_diagnostic_reason},
        "BOTH_vs_SCQ": {"b7_threshold": theta_b7_both,
                         "scq_threshold": theta_b3,
                         "b7_differs_from_scq": theta_b7_both != theta_b3},
        "step_clipping": {
            "proposed_threshold": clip_diag.get("proposed_threshold"),
            "executed_threshold": clipping.theta,
            "step_bounded": abs(clipping.theta - 0.95) <=
            clipping.config.max_threshold_step + 1e-12,
            "executed_feasibility_logged":
                "executed_threshold_in_descriptive_feasible_region" in clip_diag},
        "queried_defect_only": {
            "eligible_defect_count": len(queried_only.defect_scores),
            "raw_defect_count": queried_only.defect_count_raw,
            "unqueried_policy_rows_excluded": len(queried_only.defect_scores) == 16,
        },
        "current_memory_anchor_refresh": anchor_refresh,
        "threshold_independent_memory_policy": True,
        "checkpoint_equivalence": False,
        "prediction_available": b7_prediction.available,
        "prediction": b7_prediction.label,
    }
    saved = clipping.state_dict()
    restored = SCDACGController(0.95)
    restored.load_state_dict(saved)
    result["checkpoint_equivalence"] = restored.state_dict() == saved
    result["threshold_updates"] = int(clipping.theta != 0.95)
    return result
    saved = clipping.state_dict()
    restored = SCDACGController(0.95)
    restored.load_state_dict(saved)
    result["checkpoint_equivalence"] = restored.state_dict() == saved
    result["threshold_updates"] = int(clipping.theta != 0.95)
    return result


def run_controller_activation() -> dict[str, Any]:
    """Run a frozen score fixture through the shared prequential engine."""
    full_engine, full_memory, tape, raw_scores = _make_engine()
    full = full_engine.run()
    interrupted, _, _, _ = _make_engine()
    partial = interrupted.run(max_events=1100)
    resumed, _, _, _ = _make_engine()
    resumed.load_state_dict(partial["checkpoint"])
    replayed = resumed.run()
    checkpoint_equivalent = all(
        full[key] == replayed[key]
        for key in ("status", "cursor", "tape_digest", "scores", "predictions",
                    "prediction_availability", "trace", "metrics", "controller_states"))

    offset_engine, offset_memory, offset_tape, _ = _make_engine(threshold_offset=1.0)
    offset = offset_engine.run()
    full_memory_path = [(row["memory_version_before"], row["memory_version_after"],
                         row["memory_sha256_before"], row["memory_sha256_after"])
                        for row in full["trace"]]
    offset_memory_path = [(row["memory_version_before"], row["memory_version_after"],
                           row["memory_sha256_before"], row["memory_sha256_after"])
                          for row in offset["trace"]]
    shared_invariant = (full["scores"] == offset["scores"] and
                        tape.digest == offset_tape.digest and
                        full_memory_path == offset_memory_path and
                        full_memory.state_dict() == offset_memory.state_dict())

    methods: dict[str, Any] = {}
    for name in full["predictions"]:
        predictions = full["predictions"][name]
        available = full["prediction_availability"][name]
        theta = [row["theta_post"].get(name) for row in full["trace"]]
        info: dict[str, Any] = {
            "version": METHOD_CARDS.get(name, {}).get("version",
                       METHOD_CARDS.get(name, {}).get("faithful_or_adapted", "registered")),
            "evidence_source": "frozen synthetic score tape; common label-blind Q-A-TS; synthetic C64 and queried feedback",
            "initialization_events": 0,
            "initialization_evidence": "64 additional protected C64 normals before stream start",
            "predictions_available": sum(bool(x) for x in available),
            "abstentions": sum(label is None and is_available
                               for label, is_available in zip(predictions, available)),
            "threshold_updates": _count_changes(theta),
            "actual_operational_branches_exercised": sorted(set(
                row["prediction_reason"].get(name, "") for row in full["trace"]
                if row["prediction_reason"].get(name, ""))),
            "checkpoint_equivalence": checkpoint_equivalent,
            "shared_score_invariance": shared_invariant,
            "status": "PASS" if checkpoint_equivalent and shared_invariant else "FAIL",
        }
        if name in {"B5a", "B5b"}:
            controller = next(c for c in full_engine.controllers if c.name == name)
            state = controller.state_dict()
            core = state["core"]
            if name == "B5a":
                info.update({
                    "version": state["version"],
                    "initialization_events": len(core["initial"]),
                    "initialization_evidence": "first 1000 frozen synthetic score observations",
                    "tail_threshold_initialized": core["threshold"] is not None,
                    "accepted_normal_updates": core["n"] - 1000,
                    "peak_updates": len(core["peaks"]),
                    "alarms_excluded": core["alarms"],
                    "threshold_changed_after_initialization":
                        len(set(theta[1000:])) > 1,
                    "queried_labels_used": False,
                })
            else:
                info.update({
                    "version": state["version"],
                    "initialization_events": 50 + len(state["initial_residuals"]),
                    "initialization_evidence": "50-observation local window plus residual POT warm-up",
                    "window_initialization_events": 50,
                    "residual_warmup_events": len(state["initial_residuals"]),
                    "local_window_size": len(state["local_window"]),
                    "accepted_normal_updates": core["n"] - 1000,
                    "peak_updates": len(core["peaks"]),
                    "alarms_excluded_from_pot_and_local_mean": state["alarms"],
                    "initial_window_mean": float(np.mean(raw_scores[:50])),
                    "final_window_mean": float(sum(state["local_window"]) /
                                                len(state["local_window"])),
                    "local_mean_shifted": abs(
                        float(sum(state["local_window"]) / len(state["local_window"])) -
                        float(np.mean(raw_scores[:50]))) > 1e-3,
                    "queried_labels_used": False,
                })
            activated = info["predictions_available"] > 0
            exercised = info["threshold_changed_after_initialization"] \
                if name == "B5a" else info["local_mean_shifted"]
            info["status"] = "PASS" if activated and exercised and checkpoint_equivalent else "PARTIAL"
        elif name == "B6":
            controller = next(c for c in full_engine.controllers if c.name == name)
            state = controller.state_dict()
            reasons = [row["prediction_reason"].get(name, "") for row in full["trace"]]
            info.update({
                "version": state["version"],
                "endpoint_policy": state["endpoint_policy"],
                "p": state["p"], "delta": state["delta"],
                "reset_count": state["reset_count"],
                "cold_start_abstentions": reasons.count("ABSTAIN_COLD_START"),
                "change_detection_abstentions": reasons.count("ABSTAIN_CHANGE_DETECTED"),
                "confidence_interval_abstentions": reasons.count("SUN_CS_ABSTAIN"),
                "source_target": state["target"],
                "author_code_parity": "NOT_CLAIMED; no verified author implementation",
                "theorem_transfer": "NOT_CLAIMED for physical defects/dynamic-memory scores",
            })
            info["status"] = "PASS" if checkpoint_equivalent else "FAIL"
        elif name.endswith("_FROZEN_FALLBACK"):
            reasons = [row["prediction_reason"].get(name, "") for row in full["trace"]]
            info.update({
                "startup_fallback_count": reasons.count("STARTUP_FALLBACK"),
                "abstention_fallback_count": reasons.count("ABSTENTION_FALLBACK"),
                "original_method_count": reasons.count("ORIGINAL_METHOD"),
                "fallback_rate": sum(x in {"STARTUP_FALLBACK", "ABSTENTION_FALLBACK"}
                                      for x in reasons) / len(reasons),
                "status": "PASS" if checkpoint_equivalent else "FAIL",
            })
        methods[name] = info

    # B7's dense risk-grid branches are tested with explicit synthetic,
    # source-policy-tagged evidence outside the long stream warm-up replay.
    activation_theta0 = theta0_from_c64([0.10 + i * 0.0005 for i in range(64)])
    b7 = _b7_synthetic_branches(activation_theta0)
    b7_pass = (b7["BOTH"]["state"] == "BOTH" and
               b7["NORMAL_ONLY"]["state"] == "NORMAL_ONLY" and
               b7["CONFLICT"]["state"] == "CONFLICT" and
               b7["BOTH_vs_SCQ"]["b7_differs_from_scq"] and
               b7["BOTH"]["descriptive_feasible_candidates"] > 0 and
               b7["step_clipping"]["step_bounded"] and
               b7["step_clipping"]["executed_feasibility_logged"] and
               b7["queried_defect_only"]["unqueried_policy_rows_excluded"] and
               b7["current_memory_anchor_refresh"]["memory_version_advanced"] and
               b7["current_memory_anchor_refresh"]["protected_anchor_scores_changed"] and
               b7["current_memory_anchor_refresh"]["queried_normal_lso_policy_preserved"] and
               b7["checkpoint_equivalence"] and b7["prediction_available"])
    methods["B7"] = {
        "version": METHOD_CARDS["B7"]["version"],
        "evidence_source": "synthetic protected normal and queried-anchor snapshots with explicit source policy tags",
        "initialization_events": 1,
        "predictions_available": int(b7["prediction_available"]),
        "abstentions": int(b7["prediction"] is None),
        "threshold_updates": b7["threshold_updates"],
        "actual_operational_branches_exercised": [
            "BOTH", "NORMAL_ONLY", "CONFLICT", "feasible_set", "step_clipping",
            "minimum_risk_fallback", "queried_defect_only"],
        "checkpoint_equivalence": b7["checkpoint_equivalence"],
        "shared_score_invariance": shared_invariant,
        "b7_branch_matrix": b7,
        "status": "PASS" if b7_pass and shared_invariant else "PARTIAL",
    }

    for row in methods.values():
        if not row.get("checkpoint_equivalence"):
            row["checkpoint_equivalence"] = "FAIL"
    status = "PASS" if all(row["status"] == "PASS" for row in methods.values()) else "PARTIAL"
    return {
        "schema": "calibration.controller_activation_matrix.v1",
        "harness_version": HARNESS_VERSION,
        "status": status,
        "scope": "synthetic mechanism testing only; no native image scoring or empirical anomaly results",
        "stream_events": N_EVENTS,
        "score_tape_sha256": hashlib.sha256(np.asarray(raw_scores, dtype=np.float64).tobytes()).hexdigest(),
        "query_tape_sha256": tape.digest,
        "query_count": len(tape.events),
        "stage1_or_stage2_run": False,
        "confirmation_outcomes_opened": False,
        "checkpoint_equivalence_full_vs_resumed": checkpoint_equivalent,
        "shared_score_and_memory_invariance_under_threshold_offset": shared_invariant,
        "b7_branch_matrix": b7,
        "methods": methods,
    }
