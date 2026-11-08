"""Experiment C: lagged and sample-specific Correction Evidence acquisition.

This module is the seven-arm adapter around the frozen E6/D2 substrate.  It
does not execute a campaign on import.  Real-data execution is owned by
``exp_c_full_run`` and requires a separate, source-bound authorization.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from experiments.calibration.e6_bea_policy import (
    BEAComponents,
    INITIAL_QUERY_COST,
    QUERY_COST_LEARNING_RATE,
    QUERY_COST_MAX,
    QUERY_COST_MIN,
    QUERY_COST_WINDOW,
    TARGET_QUERY_RATE,
    compute_bea_components,
)
from experiments.calibration.e6_bea_d2 import (
    D2_EXPECTED_COUNTS,
    D2PolicyController,
    D2Proposal,
    _stable_uniform01,
)
from experiments.calibration.e6_bea_closed_loop import (
    ArmRuntime,
    E6_REGISTERED_CATEGORIES,
    EventIdentity,
    FeedbackUpdate,
    NormalCoverageMemory,
    OnDemandLabelReader,
    ScoreLedger,
    SealedQueryReceipt,
    _binary_metrics,
    _canonical_json,
    _reject_public_label_fields,
    _reject_reparse_ancestors,
    _sha256_file,
    _worst_window_ber,
)


CATEGORIES = tuple(E6_REGISTERED_CATEGORIES)
ARM_IDS = ("RAND37", "MARGIN37", "B", "BD", "BN_BAND", "BN_AVG_GATE", "BND")
UNCHANGED_D2_ARMS = ("RAND37", "MARGIN37", "B", "BD", "BND")
AVERAGE_WINDOW = 256
BUDGET_MATCH_TOLERANCE = 0.003
CONTRASTS = (
    ("BN_AVG_GATE_minus_BND", "BN_AVG_GATE", "BND"),
    ("BN_BAND_minus_BND", "BN_BAND", "BND"),
    ("BD_minus_BND", "BD", "BND"),
    ("MARGIN37_minus_BND", "MARGIN37", "BND"),
    ("B_minus_BND", "B", "BND"),
)
C_SCORE_SCHEMA = "exp-c.memory-evidence.stage-a-score.v1"
C_SEAL_SCHEMA = "exp-c.memory-evidence.stage-a-seal.v1"
C_MANIFEST_SCHEMA = "exp-c.memory-evidence.category-manifest.v1"
MAX_CATEGORY_OUTPUT_BYTES = 2 * 1024**3
MIN_FREE_DISK_BYTES = 20 * 1024**3


class ExperimentCError(RuntimeError):
    """Experiment C protocol, identity, or artifact failure."""


class COutputBudget:
    """Explicit category-local append budget for seven score/feedback streams."""

    def __init__(self, root: Path, max_bytes: int = MAX_CATEGORY_OUTPUT_BYTES):
        self.root = Path(root).resolve(strict=True)
        self.max_bytes = int(max_bytes)
        self.bytes_written = 0
        self.files: set[str] = set()
        self.allowed = {"sealed_scores.jsonl", *[f"private_feedback_{arm}.jsonl" for arm in ARM_IDS]}

    def register(self, path: Path) -> None:
        path = Path(path)
        if (path.parent.resolve(strict=True) != self.root or path.name not in self.allowed or
                path.name in self.files or path.exists() or path.is_symlink()):
            raise ExperimentCError("C output budget rejected an unregistered or existing artifact")
        self.files.add(path.name)

    def append(self, stream: Any, name: str, payload: bytes) -> None:
        if name not in self.files or self.bytes_written + len(payload) > self.max_bytes:
            raise ExperimentCError("C category scientific streams exceed their registered output budget")
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        self.bytes_written += len(payload)


def c_objective(arm_id: str, components: BEAComponents,
                d_band: float | None = None, g_bar: float = 1.0) -> float:
    """Return a registered C utility, using the frozen D2 components."""
    if (not math.isfinite(float(g_bar)) or not 0.0 <= float(g_bar) <= 1.0 or
            (d_band is not None and
             (not math.isfinite(float(d_band)) or float(d_band) < 0.0))):
        raise ExperimentCError("C gate or D_band is outside its registered range")
    if arm_id == "B":
        value = components.B
    elif arm_id == "BD":
        value = components.B + components.D
    elif arm_id == "BN_BAND":
        value = components.B + components.N + float(d_band if d_band is not None else d_band_utility(components))
    elif arm_id == "BN_AVG_GATE":
        value = components.B + components.N + float(g_bar) * float(
            d_band if d_band is not None else d_band_utility(components))
    elif arm_id == "BND":
        value = components.B + components.N + components.D
    else:
        raise ExperimentCError(f"{arm_id}: no C BEA objective is registered")
    if not math.isfinite(float(value)):
        raise ExperimentCError(f"{arm_id}: objective is not finite")
    return float(value)


def d_band_utility(components: BEAComponents) -> float:
    """Unmasked asymmetric predicted-normal band from historical fields."""
    return float(0.35 * (0.0 if components.pred_defect else
                         components.p_defect * components.gate_far * components.gate_near))


def gate_decomposition(components: BEAComponents) -> dict[str, float]:
    band = d_band_utility(components)
    sample_gate = float(1.0 - components.correction_evidence)
    reconstructed = band * sample_gate
    if not math.isclose(reconstructed, float(components.D), rel_tol=0.0, abs_tol=1e-12):
        raise ExperimentCError("D_band * current correction gate differs from frozen D")
    return {"D_band": band, "g_sample": sample_gate,
            "D_reconstructed": reconstructed, "D_historical": float(components.D)}


@dataclass
class AverageGate:
    """D_band-weighted, lagged gate over the previous 256 arm events."""

    window: int = AVERAGE_WINDOW
    history: deque[tuple[float, float]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.window != AVERAGE_WINDOW:
            raise ExperimentCError("AverageGate window is frozen at 256 events")
        self.history = deque(self.history, maxlen=self.window)

    def value(self) -> float:
        denominator = math.fsum(band for band, _ in self.history)
        if not self.history or denominator == 0.0:
            return 1.0
        numerator = math.fsum(band * gate for band, gate in self.history)
        return float(numerator / denominator)

    def observe_after_feedback(self, d_band: float, correction_evidence: float) -> None:
        if (not math.isfinite(float(d_band)) or float(d_band) < 0.0 or
                not math.isfinite(float(correction_evidence)) or
                not 0.0 <= float(correction_evidence) <= 1.0):
            raise ExperimentCError("AverageGate input is outside the registered component range")
        self.history.append((float(d_band), float(1.0 - correction_evidence)))

    def diagnostics(self) -> dict[str, Any]:
        den = math.fsum(band for band, _ in self.history)
        return {"gate": self.value(), "history_events": len(self.history),
                "history_weight": den}


@dataclass
class CBEAPriceController:
    """D2 BEA price rule for the two new registered C objective terms."""

    arm_id: str
    seed: int = 37
    query_cost_runtime: float = INITIAL_QUERY_COST
    executed_flags: deque[int] = field(default_factory=lambda: deque(maxlen=QUERY_COST_WINDOW))
    events_observed: int = 0
    proposals: int = 0
    executed_queries: int = 0
    group_suppressed: int = 0
    query_positions: list[int] = field(default_factory=list)
    query_gaps: list[int] = field(default_factory=list)
    last_observed_position: int = -1

    def __post_init__(self) -> None:
        if self.arm_id not in {"BN_BAND", "BN_AVG_GATE"}:
            raise ExperimentCError("C price controller is reserved for new BEA arms")
        if float(self.query_cost_runtime) != INITIAL_QUERY_COST:
            raise ExperimentCError("new C BEA arm must start at D2 price 0.80")

    def propose_objective(self, position: int, objective: float) -> D2Proposal:
        if int(position) != self.last_observed_position + 1 or not math.isfinite(float(objective)):
            raise ExperimentCError("new C controller received an unordered or nonfinite objective")
        cost = float(self.query_cost_runtime)
        proposed = float(objective) > cost
        return D2Proposal(self.arm_id, proposed, float(objective), cost,
                          float(objective), cost,
                          "bea_priority_above_runtime_cost" if proposed else
                          "bea_priority_not_above_runtime_cost")

    def observe_final_action(self, *, position: int, proposal: D2Proposal,
                             executed_query: bool, group_suppressed: bool) -> float:
        if (int(position) != self.last_observed_position + 1 or proposal.arm_id != self.arm_id or
                not isinstance(executed_query, bool) or not isinstance(group_suppressed, bool) or
                (group_suppressed and (not proposal.proposed or executed_query)) or
                (executed_query and not proposal.proposed) or
                (proposal.proposed and not executed_query and not group_suppressed)):
            raise ExperimentCError("new C final action does not match its sealed proposal")
        self.events_observed += 1
        self.proposals += int(proposal.proposed)
        self.group_suppressed += int(group_suppressed)
        self.executed_flags.append(int(executed_query))
        rate = sum(self.executed_flags) / len(self.executed_flags)
        self.query_cost_runtime = min(QUERY_COST_MAX, max(
            QUERY_COST_MIN, float(self.query_cost_runtime) +
            QUERY_COST_LEARNING_RATE * (rate - TARGET_QUERY_RATE)))
        if executed_query:
            if self.query_positions:
                self.query_gaps.append(int(position) - self.query_positions[-1])
            self.query_positions.append(int(position))
            self.executed_queries += 1
        self.last_observed_position = int(position)
        return float(self.query_cost_runtime)

    def diagnostics(self) -> dict[str, Any]:
        positions = list(self.query_positions)
        gaps = list(self.query_gaps)
        intervals = ([positions[0]] if positions else [])
        intervals.extend(max(0, gap - 1) for gap in gaps)
        intervals.append(max(0, self.events_observed - 1 - positions[-1])
                         if positions else self.events_observed)
        return {"arm_id": self.arm_id, "events_observed": self.events_observed,
                "proposals": self.proposals, "executed_queries": self.executed_queries,
                "realised_query_rate": self.executed_queries / self.events_observed
                if self.events_observed else 0.0,
                "group_suppressed": self.group_suppressed,
                "query_cost_runtime": float(self.query_cost_runtime),
                "query_positions": positions, "query_gaps": gaps,
                "query_free_interval_lengths": intervals,
                "max_query_gap": max(gaps, default=None),
                "max_query_free_interval_events": max(intervals, default=0),
                "utility_history_count": 0}


class CScoreLedger(ScoreLedger):
    """Joint seven-arm, outcome-free fsynced ledger with inherited seals."""

    def __init__(self, path: Path, budget: COutputBudget | None = None):
        self.arm_ids = ARM_IDS
        self.budget = budget
        if budget is not None:
            budget.register(Path(path))
        super().__init__(path, artifact_kind="scientific_stage_a")

    def seal_event(self, event: EventIdentity,
                   arm_rows: Mapping[str, Mapping[str, Any]]) -> str:
        if self._closed or event.position != self._last_position + 1 or set(arm_rows) != set(ARM_IDS):
            raise ExperimentCError("C event requires one contiguous joint record for exactly seven arms")
        row = {"schema": C_SCORE_SCHEMA, "position": event.position,
               "original_json_position": event.original_position,
               "event_id_sha256": event.event_id_sha256,
               "image_identity_sha256": event.image_identity_sha256,
               "source_group_sha256": event.source_group_sha256,
               "arm_rows": {arm: dict(arm_rows[arm]) for arm in ARM_IDS}}
        _reject_public_label_fields(row)
        payload = _canonical_json(row)
        if self.budget is None:
            self._stream.write(payload)
            self._stream.flush()
            os.fsync(self._stream.fileno())
        else:
            self.budget.append(self._stream, self.path.name, payload)
        self._last_position = event.position
        for arm, data in row["arm_rows"].items():
            if data.get("query_executed") is True:
                self._receipts[(event.position, arm)] = SealedQueryReceipt(
                    arm, event.position, event.original_position,
                    event.event_id_sha256, event.image_identity_sha256,
                    event.source_group_sha256, self._token)
        return hashlib.sha256(payload).hexdigest()


class CPrivateFeedbackSink:
    """Per-arm queried labels stored apart from the public score ledger."""

    def __init__(self, path: Path, arm_id: str, budget: COutputBudget | None = None):
        if arm_id not in ARM_IDS:
            raise ExperimentCError("C private feedback arm is unregistered")
        self.path, self.arm_id = Path(path), arm_id
        self.budget = budget
        _reject_reparse_ancestors(self.path)
        if budget is not None:
            budget.register(self.path)
        self._stream = self.path.open("xb")

    def write(self, event: EventIdentity, label: int, update: FeedbackUpdate) -> None:
        if label not in (0, 1):
            raise ExperimentCError("private feedback label is not binary")
        row = {"schema": "exp-c.memory-evidence.private-feedback.v1",
               "arm_id": self.arm_id, "position": event.position,
               "original_json_position": event.original_position,
               "event_id_sha256": event.event_id_sha256,
               "image_identity_sha256": event.image_identity_sha256,
               "source_group_sha256": event.source_group_sha256,
               "label": int(label), "update": dict(update.diagnostics)}
        payload = _canonical_json(row)
        if self.budget is None:
            self._stream.write(payload)
            self._stream.flush()
            os.fsync(self._stream.fileno())
        else:
            self.budget.append(self._stream, self.path.name, payload)

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()


@dataclass(frozen=True)
class CStageASeal:
    category: str
    run_id: str
    event_count: int
    run_dir: Path
    score_log_path: Path
    score_log_sha256: str
    private_feedback_sha256: Mapping[str, str]
    arm_query_counts: Mapping[str, int]
    arm_controller_diagnostics: Mapping[str, Mapping[str, Any]]
    manifest_path: Path
    manifest_sha256: str
    seal_path: Path
    seal_sha256: str
    complete: bool = True


class CStageAClosedLoop:
    """New seven-arm causal loop; scoring and queried updates are D2 native."""

    def __init__(self, *, category: str, run_id: str,
                 events: Sequence[EventIdentity],
                 feature_encoder: Callable[[EventIdentity], Any],
                 arms: Mapping[str, ArmRuntime], ledger: CScoreLedger,
                 run_dir: Path, manifest_path: Path, manifest_sha256: str,
                 authorization_id: str, expected_event_count: int | None = None):
        self.category, self.run_id = str(category), str(run_id)
        if self.category not in CATEGORIES or not authorization_id:
            raise ExperimentCError("C Stage A category or execution authorization is invalid")
        self.events = tuple(events)
        if (expected_event_count is not None and len(self.events) != expected_event_count) or any(
                event.position != i for i, event in enumerate(self.events)):
            raise ExperimentCError("C Stage A requires the complete contiguous registered stream")
        if set(arms) != set(ARM_IDS) or tuple(ledger.arm_ids) != ARM_IDS:
            raise ExperimentCError("C coordinator and seven-arm score ledger registries differ")
        if len({id(arm.engine) for arm in arms.values()}) != len(ARM_IDS):
            raise ExperimentCError("C arms must own independent native scorer states")
        if len({id(arm.coverage) for arm in arms.values()}) != len(ARM_IDS):
            raise ExperimentCError("C arms must own independent Coverage Memory states")
        if len({id(arm.controller) for arm in arms.values()}) != len(ARM_IDS):
            raise ExperimentCError("C arms must own independent controller states")
        for arm_id in ARM_IDS:
            arm = arms[arm_id]
            if arm.arm_id != arm_id or arm.controller.arm_id != arm_id or arm.label_reader.arm_id != arm_id:
                raise ExperimentCError("C arm, controller and private label reader identities differ")
            if arm.label_reader.ledger is not ledger or arm.label_reader.events != self.events:
                raise ExperimentCError("C label reader is not bound to the exact event stream and ledger")
            if arm.private_feedback is None or arm.private_feedback.arm_id != arm_id:
                raise ExperimentCError("C requires one private feedback sink per arm")
        self.feature_encoder = feature_encoder
        self.arms, self.ledger = dict(arms), ledger
        self.run_dir, self.manifest_path = Path(run_dir), Path(manifest_path)
        self.manifest_sha256, self.authorization_id = manifest_sha256, authorization_id
        self.average_gates = {arm: AverageGate() for arm in ARM_IDS}

    def run(self) -> CStageASeal:
        query_counts = {arm: 0 for arm in ARM_IDS}
        state_changes = {arm: {"c1_changed": 0, "greedy_changed": 0,
                               "scorer_state_changed": 0,
                               "label_disclosure_seconds": 0.0,
                               "native_feedback_update_seconds": 0.0,
                               "private_feedback_write_seconds": 0.0}
                         for arm in ARM_IDS}
        last_position = -1
        start = time.perf_counter()
        try:
            if _sha256_file(self.manifest_path) != self.manifest_sha256:
                raise ExperimentCError("C category manifest changed after initialization")
            for event in self.events:
                if event.position != last_position + 1:
                    raise ExperimentCError("C event stream skipped or duplicated a position")
                encode_started = time.perf_counter()
                encoded = self.feature_encoder(event)
                encode_seconds = max(float(getattr(encoded, "encoding_seconds", 0.0)),
                                     time.perf_counter() - encode_started)
                scored: dict[str, tuple[Any, BEAComponents, Mapping[str, Any], float,
                                         dict[str, float], float]] = {}
                proposals: dict[str, D2Proposal] = {}
                suppressed: dict[str, bool] = {}
                arm_rows: dict[str, dict[str, Any]] = {}

                for arm_id in ARM_IDS:
                    arm = self.arms[arm_id]
                    score_started = time.perf_counter()
                    pre = arm.engine.score(event, encoded)
                    score_seconds = time.perf_counter() - score_started
                    novelty_percentile, novelty_info = arm.coverage.novelty_percentile(encoded.global_cls)
                    components = compute_bea_components(
                        score=pre.gc_score, threshold=float(arm.engine.current_threshold),
                        c64_scores=pre.c64_scores, novelty_percentile=novelty_percentile,
                        correction_top32=pre.correction_top32,
                        correction_bank_size=pre.correction_bank_size,
                        topk_dN_max=pre.topk_dN_max, topk_dN_mean=pre.topk_dN_mean)
                    gates = gate_decomposition(components)
                    historical_gate = self.average_gates[arm_id].value()
                    scored[arm_id] = (pre, components, dict(novelty_info), score_seconds,
                                      gates, historical_gate)

                for arm_id in ARM_IDS:
                    arm = self.arms[arm_id]
                    _pre, components, novelty_info, _seconds, gates, historical_gate = scored[arm_id]
                    if arm_id in {"RAND37", "MARGIN37"}:
                        proposal = arm.controller.propose(
                            position=event.position, event_id_sha256=event.event_id_sha256,
                            components=components,
                            novelty_distance_nn=float(novelty_info["distance_nn"]))
                    elif arm_id in {"B", "BD", "BND"}:
                        proposal = arm.controller.propose(
                            position=event.position, event_id_sha256=event.event_id_sha256,
                            components=components,
                            novelty_distance_nn=float(novelty_info["distance_nn"]))
                        expected = c_objective(arm_id, components, gates["D_band"], historical_gate)
                        if not math.isclose(float(proposal.objective), expected,
                                            rel_tol=0.0, abs_tol=1e-12):
                            raise ExperimentCError(f"{arm_id}: reused D2 proposal differs from C objective")
                    else:
                        objective = c_objective(arm_id, components, gates["D_band"], historical_gate)
                        proposal = arm.controller.propose_objective(event.position, objective)
                    proposals[arm_id] = proposal

                for arm_id in ARM_IDS:
                    arm = self.arms[arm_id]
                    pre, components, novelty_info, score_seconds, gates, historical_gate = scored[arm_id]
                    proposal = proposals[arm_id]
                    group_suppressed = bool(proposal.proposed and
                                            event.source_group_sha256 in arm.queried_groups)
                    executed = bool(proposal.proposed and not group_suppressed)
                    suppressed[arm_id] = group_suppressed
                    objective = (None if proposal.objective is None else float(proposal.objective))
                    arm_rows[arm_id] = {
                        "gc_score": float(pre.gc_score), "c0_score": float(pre.c0_score),
                        "b3_threshold": float(arm.engine.current_threshold),
                        "c64_sigma": float(components.sigma),
                        "prediction": int(components.pred_defect),
                        "scorer_state_version": int(pre.scorer_state_version),
                        "scorer_state_sha256": str(pre.scorer_state_sha256),
                        "correction_bank_size": int(pre.correction_bank_size),
                        "coverage_state_sha256": arm.coverage.state_sha256(),
                        "query_proposed": bool(proposal.proposed), "query_executed": executed,
                        "group_suppressed": group_suppressed,
                        "suppression_reason": "source_group_already_queried" if group_suppressed else None,
                        "query_reason": proposal.reason, "query_utility": proposal.utility,
                        "query_threshold": proposal.threshold, "query_objective": objective,
                        "query_cost_runtime": proposal.query_cost,
                        "query_random_u": proposal.random_u,
                        "query_tie_probability": proposal.tie_probability,
                        "query_tie_u": proposal.tie_u, "query_n_gt": proposal.n_gt,
                        "query_n_eq": proposal.n_eq, "utility_history_count": proposal.history_count,
                        "encoding_seconds": encode_seconds, "native_score_seconds": score_seconds,
                        "novelty_info": dict(novelty_info), "components": components.as_dict(),
                        "D_band": gates["D_band"], "g_sample": gates["g_sample"],
                        "g_bar_lagged": historical_gate,
                        "objective_reconstructed": objective,
                    }

                # Read-only same-state proposals use BND's score, history and price.
                bnd_components = scored["BND"][1]
                bnd_band = scored["BND"][4]["D_band"]
                bnd_gate = scored["BND"][5]
                bnd_price = float(proposals["BND"].query_cost)
                same_state_band = c_objective("BN_BAND", bnd_components, bnd_band, bnd_gate)
                same_state_average = c_objective("BN_AVG_GATE", bnd_components, bnd_band, bnd_gate)
                same_state_sample = c_objective("BND", bnd_components, bnd_band, bnd_gate)
                arm_rows["BND"]["within_bnd_state_diagnostic"] = {
                    "read_only": True, "price": bnd_price,
                    "sample_specific_objective": same_state_sample,
                    "band_objective": same_state_band,
                    "average_gate_objective": same_state_average,
                    "sample_specific_proposal": same_state_sample > bnd_price,
                    "band_proposal": same_state_band > bnd_price,
                    "average_gate_proposal": same_state_average > bnd_price,
                    "band_minus_sample": same_state_band - same_state_sample,
                    "average_minus_sample": same_state_average - same_state_sample,
                }

                # The public seven-arm decision record is durable before any label read.
                self.ledger.seal_event(event, arm_rows)
                for arm_id in ARM_IDS:
                    arm = self.arms[arm_id]
                    pre = scored[arm_id][0]
                    if arm_rows[arm_id]["query_executed"]:
                        arm.queried_groups.add(event.source_group_sha256)
                        receipt = self.ledger.query_receipt(arm_id, event)
                        disclosure_started = time.perf_counter()
                        label = arm.label_reader.label_for(event, receipt)
                        state_changes[arm_id]["label_disclosure_seconds"] += (
                            time.perf_counter() - disclosure_started)
                        feedback_started = time.perf_counter()
                        update = arm.engine.apply_feedback(event, encoded, pre, label)
                        state_changes[arm_id]["native_feedback_update_seconds"] += (
                            time.perf_counter() - feedback_started)
                        if update.label != label:
                            raise ExperimentCError("native feedback update changed the authorized label")
                        state_changes[arm_id]["c1_changed"] += int(update.c1_changed)
                        state_changes[arm_id]["greedy_changed"] += int(update.greedy_changed)
                        state_changes[arm_id]["scorer_state_changed"] += int(update.scorer_state_changed)
                        if label == 0:
                            arm.coverage.observe_normal(encoded.global_cls)
                        write_started = time.perf_counter()
                        arm.private_feedback.write(event, label, update)
                        state_changes[arm_id]["private_feedback_write_seconds"] += (
                            time.perf_counter() - write_started)
                        query_counts[arm_id] += 1
                    else:
                        arm.engine.observe_unqueried(event, encoded, pre)
                    arm.controller.observe_final_action(
                        position=event.position, proposal=proposals[arm_id],
                        executed_query=bool(arm_rows[arm_id]["query_executed"]),
                        group_suppressed=suppressed[arm_id])

                # Current evidence becomes history only after all feedback actions finish.
                for arm_id in ARM_IDS:
                    components = scored[arm_id][1]
                    self.average_gates[arm_id].observe_after_feedback(
                        scored[arm_id][4]["D_band"], components.correction_evidence)
                last_position = event.position

            if last_position + 1 != len(self.events):
                raise ExperimentCError("C stream did not finish its registered event set")
            self.ledger.close()
            for arm in self.arms.values():
                arm.label_reader.close()
                arm.private_feedback.close()
            score_sha = _sha256_file(self.ledger.path)
            feedback_sha = {arm: _sha256_file(self.arms[arm].private_feedback.path) for arm in ARM_IDS}
            diagnostics = {arm: self.arms[arm].controller.diagnostics() for arm in ARM_IDS}
            seal = {"schema": C_SEAL_SCHEMA, "status": "COMPLETE",
                    "run_id": self.run_id, "category": self.category,
                    "event_count": len(self.events), "authorization_id": self.authorization_id,
                    "manifest_sha256": self.manifest_sha256,
                    "score_log": {"file": self.ledger.path.name, "sha256": score_sha},
                    "feedback_artifacts": {
                        arm: {"file": Path(self.arms[arm].private_feedback.path).name,
                              "sha256": feedback_sha[arm]} for arm in ARM_IDS},
                    "arm_query_counts": query_counts,
                    "arm_controller_diagnostics": diagnostics,
                    "final_arm_state": {arm: {
                        "scorer_state_version": int(self.arms[arm].engine.state_version),
                        "scorer_state_sha256": str(self.arms[arm].engine.state_digest),
                        "coverage_state_sha256": self.arms[arm].coverage.state_sha256(),
                        "queried_source_group_count": len(self.arms[arm].queried_groups)}
                        for arm in ARM_IDS},
                    "average_gate_final": {arm: self.average_gates[arm].diagnostics()
                                           for arm in ARM_IDS},
                    "state_change_and_feedback_cost": state_changes,
                    "greedy_writer_final": {arm: self.arms[arm].engine.writer.audit()
                                             for arm in ARM_IDS},
                    "correction_bank_final_size": {
                        arm: int(self.arms[arm].engine.correction_cb.size()) for arm in ARM_IDS},
                    "runtime_seconds": float(time.perf_counter() - start),
                    "score_and_feedback_bytes": (self.ledger.budget.bytes_written
                                                  if self.ledger.budget is not None else None),
                    "label_free_score_ledger": True}
            seal_path = self.run_dir / "stage_a_seal.json"
            payload = _canonical_json(seal)
            with seal_path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            return CStageASeal(
                self.category, self.run_id, len(self.events), self.run_dir,
                self.ledger.path, score_sha, feedback_sha, query_counts, diagnostics,
                self.manifest_path, self.manifest_sha256, seal_path,
                hashlib.sha256(payload).hexdigest())
        except BaseException:
            self.ledger.close()
            for arm in self.arms.values():
                try:
                    arm.label_reader.close()
                    arm.private_feedback.close()
                except Exception:
                    pass
            raise


def classify_budget(*, rate_left: float, rate_right: float, ber_left: float,
                    ber_right: float, tolerance: float = BUDGET_MATCH_TOLERANCE) -> str:
    if abs(float(rate_left) - float(rate_right)) <= tolerance:
        return "APPROXIMATELY_BUDGET_MATCHED"
    if rate_left < rate_right and ber_left <= ber_right:
        return "LEFT_DOMINATES_QUALITY_AND_EXPENDITURE"
    if rate_right < rate_left and ber_right <= ber_left:
        return "RIGHT_DOMINATES_QUALITY_AND_EXPENDITURE"
    return "BUDGET_CONFOUNDED"


def analyze_metrics(per_category: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    if set(per_category) != set(CATEGORIES):
        raise ExperimentCError("Stage B metrics require exactly the four registered categories")
    macro: dict[str, dict[str, float]] = {}
    for arm in ARM_IDS:
        macro[arm] = {metric: float(np.mean([per_category[cat][arm][metric] for cat in CATEGORIES]))
                      for metric in ("ber", "ba", "fpr", "fnr", "auroc", "ap",
                                     "worst_256_window_ber", "query_rate")}
        macro[arm]["query_count_sum"] = int(sum(per_category[cat][arm]["query_count"] for cat in CATEGORIES))
    contrasts: dict[str, Any] = {}
    for scope in (*CATEGORIES, "CATEGORY_EQUAL_MACRO"):
        source = macro if scope == "CATEGORY_EQUAL_MACRO" else per_category[scope]
        rows: dict[str, Any] = {}
        for name, left, right in CONTRASTS:
            rate_difference = float(source[left]["query_rate"] - source[right]["query_rate"])
            rows[name] = {"ber_difference_left_minus_right": float(source[left]["ber"] - source[right]["ber"]),
                          "query_rate_difference_left_minus_right": rate_difference,
                          "budget_match_tolerance": BUDGET_MATCH_TOLERANCE,
                          "budget_classification": classify_budget(
                              rate_left=source[left]["query_rate"], rate_right=source[right]["query_rate"],
                              ber_left=source[left]["ber"], ber_right=source[right]["ber"])}
        contrasts[scope] = rows
    return {"primary_metric": "category_equal_macro_prequential_ber",
            "per_category": dict(per_category), "category_equal_macro": macro,
            "contrasts": contrasts,
            "budget_match_rule": "absolute query-rate difference <= 0.003, category and macro separately"}


def paired_source_group_bootstrap(labels: Mapping[str, np.ndarray],
                                  score_rows: Mapping[str, Sequence[Mapping[str, Any]]],
                                  *, draws: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Paired cluster intervals; descriptive conditional-trajectory summaries only."""
    if draws < 100:
        raise ExperimentCError("paired bootstrap requires at least 100 registered resamples")
    rng = np.random.default_rng(seed)
    category_draws: dict[str, dict[str, list[float]]] = {cat: {} for cat in CATEGORIES}
    macro_draws: dict[str, list[float]] = {name: [] for name, _, _ in CONTRASTS}
    for category in CATEGORIES:
        rows, y = score_rows[category], np.asarray(labels[category], dtype=np.int8)
        if len(rows) != len(y):
            raise ExperimentCError(f"{category}: bootstrap inputs are not paired")
        by_group: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            by_group.setdefault(str(row["source_group_sha256"]), []).append(index)
        groups = tuple(sorted(by_group))
        if not groups:
            raise ExperimentCError(f"{category}: no Source Groups for paired bootstrap")
        group_stats: dict[str, dict[str, np.ndarray]] = {}
        for group in groups:
            idx = np.asarray(by_group[group], dtype=np.int64)
            stats = {}
            for arm in ARM_IDS:
                ar = [rows[i]["arm_rows"][arm] for i in idx]
                pred = np.asarray([int(item["prediction"]) for item in ar], dtype=np.int8)
                tn = np.count_nonzero((y[idx] == 0) & (pred == 0))
                fp = np.count_nonzero((y[idx] == 0) & (pred == 1))
                fn = np.count_nonzero((y[idx] == 1) & (pred == 0))
                tp = np.count_nonzero((y[idx] == 1) & (pred == 1))
                stats[arm] = np.asarray([tn, fp, fn, tp], dtype=np.float64)
            group_stats[group] = stats
        local: dict[str, list[float]] = {name: [] for name, _, _ in CONTRASTS}
        attempts = 0
        while min((len(values) for values in local.values()), default=0) < draws and attempts < draws * 5:
            attempts += 1
            sample = rng.integers(0, len(groups), size=len(groups))
            sums = {arm: np.sum([group_stats[groups[i]][arm] for i in sample], axis=0)
                    for arm in ARM_IDS}
            losses: dict[str, float] = {}
            valid = True
            for arm, (tn, fp, fn, tp) in sums.items():
                if tn + fp == 0 or fn + tp == 0:
                    valid = False
                    break
                losses[arm] = 0.5 * (fp / (tn + fp) + fn / (fn + tp))
            if not valid:
                continue
            for name, left, right in CONTRASTS:
                local[name].append(float(losses[left] - losses[right]))
        category_draws[category] = local
    for draw in range(draws):
        for name, _left, _right in CONTRASTS:
            values = [category_draws[cat][name][draw] for cat in CATEGORIES
                      if len(category_draws[cat][name]) > draw]
            if len(values) == len(CATEGORIES):
                macro_draws[name].append(float(np.mean(values)))
    return {"method": "paired Source Group cluster bootstrap on sealed conditional trajectories",
            "draws_requested": draws, "seed": seed,
            "independent_online_histories_claimed": False,
            "category_contrasts": {
                category: {name: _interval(values) for name, values in category_draws[category].items()}
                for category in CATEGORIES},
            "category_equal_macro_contrasts": {
                name: _interval(values) for name, values in macro_draws.items()}}


def _interval(values: Sequence[float]) -> dict[str, Any]:
    if len(values) < 100:
        return {"lower_2_5": None, "median": None, "upper_97_5": None,
                "valid_draws": len(values), "status": "INSUFFICIENT_VALID_RESAMPLES"}
    q = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.5, 0.975])
    return {"lower_2_5": float(q[0]), "median": float(q[1]),
            "upper_97_5": float(q[2]), "valid_draws": len(values),
            "status": "DESCRIPTIVE_CONDITIONAL_TRAJECTORY_INTERVAL"}


def build_stage_b_metrics(labels: Mapping[str, np.ndarray],
                          rows_by_category: Mapping[str, Sequence[Mapping[str, Any]]],
                          query_counts: Mapping[str, Mapping[str, int]], *,
                          bootstrap_draws: int = 1000) -> dict[str, Any]:
    per_category: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        rows = rows_by_category[category]
        y = np.asarray(labels[category], dtype=np.int8)
        per_category[category] = {}
        for arm in ARM_IDS:
            arm_rows = [row["arm_rows"][arm] for row in rows]
            score = np.asarray([float(row["gc_score"]) for row in arm_rows], dtype=np.float64)
            threshold = np.asarray([float(row["b3_threshold"]) for row in arm_rows], dtype=np.float64)
            pred = np.asarray([int(row["prediction"]) for row in arm_rows], dtype=np.int8)
            if not np.array_equal(pred, (score >= threshold).astype(np.int8)):
                raise ExperimentCError(f"{category}/{arm}: sealed prediction differs from score >= B3")
            metric = _binary_metrics(y, pred, score)
            count = int(query_counts[category][arm])
            query_mask = np.asarray([bool(item["query_executed"]) for item in arm_rows], dtype=bool)
            metric.update({"query_count": count, "query_rate": count / len(y),
                           "queried_defect_count": int(np.count_nonzero(y[query_mask] == 1)),
                           "queried_normal_count": int(np.count_nonzero(y[query_mask] == 0)),
                           "first_queried_defect_position": next((int(i) for i in np.flatnonzero(
                               query_mask & (y == 1))), None),
                           "worst_256_window_ber": _worst_window_ber(y, pred, width=256),
                           "source_group_suppressed": int(sum(
                               bool(item["group_suppressed"]) for item in arm_rows)),
                           "query_positions": [i for i, item in enumerate(arm_rows)
                                               if item["query_executed"]],
                           "segment_query_rates": _segment_query_rates(arm_rows)})
            positions = metric["query_positions"]
            metric["maximum_query_free_interval_events"] = max(
                ([positions[0]] if positions else []) +
                [max(0, b - a - 1) for a, b in zip(positions, positions[1:])] +
                [max(0, len(y) - 1 - positions[-1]) if positions else len(y)])
            metric["c1_state_version_max"] = max(
                int(item["scorer_state_version"]) for item in arm_rows)
            metric["correction_bank_size_max"] = max(
                int(item["correction_bank_size"]) for item in arm_rows)
            per_category[category][arm] = metric
    result = analyze_metrics(per_category)
    result["mechanism_diagnostics"] = _mechanism_diagnostics(rows_by_category)
    result["paired_source_group_bootstrap"] = paired_source_group_bootstrap(
        labels, rows_by_category, draws=bootstrap_draws)
    return result


def _mechanism_diagnostics(rows_by_category: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    combined = [row for category in CATEGORIES for row in rows_by_category[category]]
    bnd_rows = [row["arm_rows"]["BND"] for row in combined]
    eligible = [row for row in bnd_rows if not bool(row["components"]["pred_defect"])]
    bands = [float(row["D_band"]) for row in bnd_rows]
    evidence = [float(row["components"]["correction_evidence"]) for row in eligible]
    divergences = {"sample_vs_band": 0, "sample_vs_average": 0,
                   "sample_differs_from_both": 0}
    objective_diffs = {"band_minus_sample": [], "average_minus_sample": []}
    for row in bnd_rows:
        info = row.get("within_bnd_state_diagnostic")
        if not isinstance(info, Mapping) or info.get("read_only") is not True:
            raise ExperimentCError("BND state-matched read-only policy diagnostic is missing")
        sample, band, average = (bool(info["sample_specific_proposal"]),
                                 bool(info["band_proposal"]),
                                 bool(info["average_gate_proposal"]))
        divergences["sample_vs_band"] += int(sample != band)
        divergences["sample_vs_average"] += int(sample != average)
        divergences["sample_differs_from_both"] += int(sample != band and sample != average)
        objective_diffs["band_minus_sample"].append(float(info["band_minus_sample"]))
        objective_diffs["average_minus_sample"].append(float(info["average_minus_sample"]))
    return {
        "band_positive_frequency": float(np.mean(np.asarray(bands) > 0.0)),
        "band_positive_event_count": int(np.count_nonzero(np.asarray(bands) > 0.0)),
        "predicted_normal_eligible_events": len(eligible),
        "eligible_fraction_with_partial_correction_evidence": (
            float(np.mean([(0.0 < value < 1.0) for value in evidence]) if evidence else 0.0)),
        "current_gate_distribution": _distribution([1.0 - float(row["components"]["correction_evidence"])
                                                     for row in bnd_rows]),
        "lagged_average_gate_distribution": _distribution([float(row["g_bar_lagged"])
                                                            for row in
                                                            (r["arm_rows"]["BN_AVG_GATE"] for r in combined)]),
        "objective_difference_distribution": {name: _distribution(values)
                                              for name, values in objective_diffs.items()},
        "same_bnd_state_price_query_divergence_counts": divergences,
        "counterfactual_queries_executed": False,
        "state_updates_from_diagnostic": False,
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ExperimentCError("cannot summarize empty or nonfinite numeric values")
    q = np.quantile(array, [0.05, 0.5, 0.95])
    return {"mean": float(np.mean(array)), "median": float(q[1]),
            "p05": float(q[0]), "p95": float(q[2]),
            "min": float(np.min(array)), "max": float(np.max(array))}


def _segment_query_rates(arm_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    length = len(arm_rows)
    result = []
    for i in range(4):
        lo, hi = ((i * length + 3) // 4, ((i + 1) * length + 3) // 4)
        count = sum(bool(row["query_executed"]) for row in arm_rows[lo:hi])
        result.append({"segment": i + 1, "start_position": lo,
                       "stop_position_exclusive": hi,
                       "query_count": int(count), "query_rate": count / (hi - lo)})
    return result
