"""D2 ten-arm matched-supervision BEA experiment primitives.

This module supplies D2-specific policy state, joint event sealing, Stage A
validation, Stage B metrics, and the label-blind E6 trajectory parity audit.
Native feature extraction and scoring remain delegated to the frozen R7/E6
adapters. Importing this module does not access dataset files or CUDA.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass
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
    UTILITY_COLD_START_EVENTS,
    UTILITY_HISTORY_WINDOW,
    _stable_uniform01,
    compute_bea_components,
)
from experiments.calibration.e6_bea_closed_loop import (
    ARM_IDS as E6_ARM_IDS,
    E6ClosedLoopError,
    E6_REGISTERED_CATEGORIES,
    ArmRuntime,
    EncodedEvent,
    EventIdentity,
    FeedbackUpdate,
    NativePreEventScore,
    NormalCoverageMemory,
    OnDemandLabelReader,
    ScoreLedger,
    SealedQueryReceipt,
    StageASeal,
    StageBLabelSource,
    _canonical_json,
    _reject_public_label_fields,
    _reject_reparse_ancestors,
    _sha256_file,
    _validate_stage_b_identity_records,
    _load_stage_b_identity_records,
    _binary_metrics,
    _worst_window_ber,
)


D2_ARM_IDS = (
    "B", "BN", "BD", "BND", "BNCD",
    "RAND37", "PERIOD27", "MARGIN37", "ENTROPY37", "NOVELTY_NN37",
)
D2_BEA_TERMS: Mapping[str, tuple[str, ...]] = {
    "B": ("B",),
    "BN": ("B", "N"),
    "BD": ("B", "D"),
    "BND": ("B", "N", "D"),
    "BNCD": ("B", "N", "C", "D"),
}
D2_SIMPLE_ARMS = ("RAND37", "PERIOD27", "MARGIN37", "ENTROPY37", "NOVELTY_NN37")
D2_E6_PARITY_MAP = {"B": "Q4", "BND": "Q5", "BNCD": "Q6"}
D2_STAGE_B_AUTHORIZATION_ID = "E6-BEA-D2-STAGEB-20261104-A1"
D2_TARGET_RATE = 0.037
D2_PERCENTILE = 0.963
D2_RANDOM_GROUP_VIEWS = 5
D2_RANDOM_VIEW_PROBABILITY = 1.0 - (
    1.0 - D2_RANDOM_GROUP_VIEWS * D2_TARGET_RATE
) ** (1.0 / D2_RANDOM_GROUP_VIEWS)
D2_EXPECTED_COUNTS = {"audiojack": 3815, "regulator": 3855,
                      "switch": 3820, "rolled_strip_base": 3805}
D2_STAGE_A_SCHEMA = "e6.bea_v2.d2.stage_a_score.v1"
D2_SEAL_SCHEMA = "e6.bea_v2.d2.stage_a_seal.v1"
D2_MANIFEST_SCHEMA = "e6.bea_v2.d2.run_manifest.v1"
D2_CONTRACT_SCHEMA = "e6.bea_v2.d2.contract.v1"
D2_EVALUATION_SCHEMA = "e6.bea_v2.d2.evaluation.v1"


class D2Error(RuntimeError):
    """D2 protocol or artifact validation failure."""


@dataclass(frozen=True)
class D2Proposal:
    arm_id: str
    proposed: bool
    utility: float | None
    threshold: float | None
    objective: float | None
    query_cost: float | None
    reason: str
    random_u: float | None = None
    tie_probability: float | None = None
    tie_u: float | None = None
    n_gt: int = 0
    n_eq: int = 0
    history_count: int = 0


def _objective(components: BEAComponents, arm_id: str) -> float:
    try:
        terms = D2_BEA_TERMS[arm_id]
    except KeyError as exc:
        raise D2Error(f"{arm_id}: no additive BEA objective is registered") from exc
    return float(sum(float(getattr(components, term)) for term in terms))


@dataclass
class D2PolicyController:
    """Independent online controller for one registered D2 arm."""

    arm_id: str
    seed: int = 37
    query_cost_runtime: float = INITIAL_QUERY_COST
    executed_flags: deque[int] | None = None
    utility_history: deque[float] | None = None
    events_observed: int = 0
    proposals: int = 0
    executed_queries: int = 0
    group_suppressed: int = 0
    quantile_comparisons: int = 0
    quantile_ties: int = 0
    tie_lottery_calls: int = 0
    query_positions: list[int] | None = None
    query_gaps: list[int] | None = None
    last_observed_position: int = -1

    def __post_init__(self) -> None:
        if self.arm_id not in D2_ARM_IDS:
            raise D2Error(f"unregistered D2 arm: {self.arm_id}")
        if (not math.isfinite(float(self.query_cost_runtime)) or
                not QUERY_COST_MIN <= float(self.query_cost_runtime) <= QUERY_COST_MAX):
            raise D2Error("D2 BEA query price is outside the frozen E6 bounds")
        self.executed_flags = deque(maxlen=QUERY_COST_WINDOW)
        self.utility_history = deque(maxlen=UTILITY_HISTORY_WINDOW)
        self.query_positions = []
        self.query_gaps = []

    def propose(self, *, position: int, event_id_sha256: str,
                components: BEAComponents, novelty_distance_nn: float) -> D2Proposal:
        if (isinstance(position, bool) or int(position) != position or
                int(position) != self.last_observed_position + 1):
            raise D2Error("D2 policy positions must be observed once in order")
        if self.arm_id in D2_BEA_TERMS:
            objective = _objective(components, self.arm_id)
            cost = float(self.query_cost_runtime)
            return D2Proposal(
                self.arm_id, objective > cost, objective, cost, objective, cost,
                "bea_priority_above_runtime_cost" if objective > cost else
                "bea_priority_not_above_runtime_cost")
        if self.arm_id == "RAND37":
            u = _stable_uniform01(self.seed, self.arm_id, event_id_sha256, "random")
            return D2Proposal(self.arm_id, u < D2_RANDOM_VIEW_PROBABILITY,
                              D2_RANDOM_VIEW_PROBABILITY, None, None, None,
                              "identity_hashed_five_view_bernoulli", random_u=u)
        if self.arm_id == "PERIOD27":
            proposed = ((int(position) + 1) % 27) == 0
            return D2Proposal(self.arm_id, proposed, None, None, None, None,
                              "position_plus_one_mod_27")

        if self.arm_id == "MARGIN37":
            utility = float(components.psi)
        elif self.arm_id == "ENTROPY37":
            utility = float(components.entropy)
        elif self.arm_id == "NOVELTY_NN37":
            utility = float(novelty_distance_nn)
            if utility < 0.0:
                raise D2Error("raw nearest-neighbor novelty distance cannot be negative")
        else:  # pragma: no cover - guarded by D2_ARM_IDS and __post_init__
            raise D2Error(f"unknown D2 policy: {self.arm_id}")
        if not math.isfinite(utility):
            raise D2Error("percentile query utility must be finite")
        prior = tuple(self.utility_history or ())
        if len(prior) < UTILITY_COLD_START_EVENTS:
            # A shared hash identity makes the margin and entropy cold-start
            # draws identical, so differences arise from their utilities.
            u = _stable_uniform01(self.seed, "D2_PERCENTILE_SHARED",
                                  event_id_sha256, "cold_start_bernoulli")
            return D2Proposal(self.arm_id, u < D2_TARGET_RATE, utility, None,
                              utility, None, "seeded_3p7_percent_cold_start",
                              random_u=u, history_count=len(prior))

        threshold = float(np.quantile(np.asarray(prior, dtype=np.float64), D2_PERCENTILE))
        n_gt = int(np.count_nonzero(np.asarray(prior, dtype=np.float64) > threshold))
        n_eq = int(np.count_nonzero(np.asarray(prior, dtype=np.float64) == threshold))
        self.quantile_comparisons += 1
        if utility > threshold:
            return D2Proposal(self.arm_id, True, utility, threshold, utility, None,
                              "past_256_q963_above", n_gt=n_gt, n_eq=n_eq,
                              history_count=len(prior))
        if utility < threshold:
            return D2Proposal(self.arm_id, False, utility, threshold, utility, None,
                              "past_256_q963_below", n_gt=n_gt, n_eq=n_eq,
                              history_count=len(prior))
        self.quantile_ties += 1
        self.tie_lottery_calls += 1
        tie_probability = (min(1.0, max(0.0,
            (D2_TARGET_RATE * len(prior) - n_gt) / n_eq)) if n_eq else 0.0)
        tie_u = _stable_uniform01(self.seed, "D2_PERCENTILE_SHARED",
                                  event_id_sha256, "percentile_tie_lottery")
        return D2Proposal(
            self.arm_id, tie_u < tie_probability, utility, threshold, utility, None,
            "q963_identity_hashed_tie_lottery", tie_probability=tie_probability,
            tie_u=tie_u, n_gt=n_gt, n_eq=n_eq, history_count=len(prior))

    def observe_final_action(self, *, position: int, proposal: D2Proposal,
                             executed_query: bool, group_suppressed: bool) -> float | None:
        if (isinstance(position, bool) or int(position) != position or
                int(position) != self.last_observed_position + 1 or
                proposal.arm_id != self.arm_id or
                not isinstance(executed_query, bool) or
                not isinstance(group_suppressed, bool)):
            raise D2Error("D2 final action does not match its controller/event")
        if group_suppressed and (not proposal.proposed or executed_query):
            raise D2Error("invalid D2 Source Group suppression state")
        if executed_query and not proposal.proposed:
            raise D2Error("D2 execution cannot occur without a proposal")
        if proposal.proposed and not executed_query and not group_suppressed:
            raise D2Error("D2 proposal was not executed or Source-Group-suppressed")
        self.events_observed += 1
        self.proposals += int(proposal.proposed)
        self.group_suppressed += int(group_suppressed)
        if self.arm_id in {"MARGIN37", "ENTROPY37", "NOVELTY_NN37"}:
            if proposal.utility is None or not math.isfinite(float(proposal.utility)):
                raise D2Error("D2 percentile policy failed to record its utility")
            assert self.utility_history is not None
            self.utility_history.append(float(proposal.utility))
        next_cost: float | None = None
        if self.arm_id in D2_BEA_TERMS:
            assert self.executed_flags is not None
            self.executed_flags.append(int(executed_query))
            rho_local = sum(self.executed_flags) / len(self.executed_flags)
            next_cost = min(QUERY_COST_MAX, max(
                QUERY_COST_MIN, float(self.query_cost_runtime) +
                QUERY_COST_LEARNING_RATE * (rho_local - TARGET_QUERY_RATE)))
            self.query_cost_runtime = next_cost
        if executed_query:
            assert self.query_positions is not None and self.query_gaps is not None
            if self.query_positions:
                self.query_gaps.append(int(position) - self.query_positions[-1])
            self.query_positions.append(int(position))
            self.executed_queries += 1
        self.last_observed_position = int(position)
        return next_cost

    def diagnostics(self) -> dict[str, Any]:
        positions = list(self.query_positions or ())
        gaps = list(self.query_gaps or ())
        intervals = ([positions[0]] if positions else [])
        intervals.extend(max(0, gap - 1) for gap in gaps)
        intervals.append(max(0, self.events_observed - 1 - positions[-1])
                         if positions else self.events_observed)
        return {
            "arm_id": self.arm_id,
            "events_observed": self.events_observed,
            "proposals": self.proposals,
            "executed_queries": self.executed_queries,
            "realised_query_rate": (self.executed_queries / self.events_observed
                                     if self.events_observed else 0.0),
            "group_suppressed": self.group_suppressed,
            "quantile_comparisons": self.quantile_comparisons,
            "quantile_ties": self.quantile_ties,
            "tie_lottery_calls": self.tie_lottery_calls,
            "query_cost_runtime": (float(self.query_cost_runtime)
                                    if self.arm_id in D2_BEA_TERMS else None),
            "query_positions": positions,
            "query_gaps": gaps,
            "query_free_interval_lengths": intervals,
            "max_query_gap": max(gaps, default=None),
            "max_query_free_interval_events": max(intervals, default=0),
            "utility_history_count": len(self.utility_history or ()),
        }


class D2ScoreLedger(ScoreLedger):
    """D2-specific joint append-only ledger with the inherited receipt gate."""

    def __init__(self, path: Path, arm_ids: Sequence[str] = D2_ARM_IDS):
        self.arm_ids = tuple(arm_ids)
        if (not self.arm_ids or len(set(self.arm_ids)) != len(self.arm_ids) or
                set(self.arm_ids) != set(D2_ARM_IDS)):
            raise D2Error("D2 ledger must use the exact registered ten-arm set")
        super().__init__(path, artifact_kind="scientific_stage_a")

    def seal_event(self, event: EventIdentity,
                   arm_rows: Mapping[str, Mapping[str, Any]]) -> str:
        if self._closed or event.position != self._last_position + 1:
            raise D2Error("D2 score ledger is closed or event order is not contiguous")
        if set(arm_rows) != set(self.arm_ids):
            raise D2Error("every D2 event must seal all ten arms together")
        row = {
            "schema": D2_STAGE_A_SCHEMA,
            "position": event.position,
            "original_json_position": event.original_position,
            "event_id_sha256": event.event_id_sha256,
            "image_identity_sha256": event.image_identity_sha256,
            "source_group_sha256": event.source_group_sha256,
            "arm_rows": {arm: dict(arm_rows[arm]) for arm in self.arm_ids},
        }
        _reject_public_label_fields(row)
        payload = _canonical_json(row)
        self._stream.write(payload)
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._last_position = event.position
        for arm_id, arm_row in row["arm_rows"].items():
            if arm_row.get("query_executed") is True:
                self._receipts[(event.position, arm_id)] = SealedQueryReceipt(
                    arm_id, event.position, event.original_position,
                    event.event_id_sha256, event.image_identity_sha256,
                    event.source_group_sha256, self._token)
        return hashlib.sha256(payload).hexdigest()


class D2PrivateFeedbackSink:
    """D2 arm-private feedback file; labels never enter the public ledger."""

    def __init__(self, path: Path, arm_id: str):
        if arm_id not in D2_ARM_IDS:
            raise D2Error("D2 feedback sink arm identity is not registered")
        self.arm_id = arm_id
        self.path = Path(path)
        _reject_reparse_ancestors(self.path)
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError(f"D2 feedback file already exists: {self.path}")
        self._stream = self.path.open("xb")

    def write(self, event: EventIdentity, label: int, update: FeedbackUpdate) -> None:
        if label not in (0, 1):
            raise D2Error("D2 private feedback label must be binary")
        row = {
            "schema": "e6.bea_v2.d2.private_feedback.v1",
            "arm_id": self.arm_id,
            "position": event.position,
            "original_json_position": event.original_position,
            "event_id_sha256": event.event_id_sha256,
            "image_identity_sha256": event.image_identity_sha256,
            "source_group_sha256": event.source_group_sha256,
            "label": int(label),
            "update": dict(update.diagnostics),
        }
        payload = _canonical_json(row)
        self._stream.write(payload)
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()


@dataclass(frozen=True)
class D2StageASeal:
    category: str
    run_id: str
    event_count: int
    score_log_path: str
    score_log_sha256: str
    private_feedback_sha256: Mapping[str, str]
    arm_query_counts: Mapping[str, int]
    arm_controller_diagnostics: Mapping[str, Mapping[str, Any]]
    manifest_path: str
    manifest_sha256: str
    seal_path: str
    seal_sha256: str
    category_json_sha256: str
    complete: bool = True


def _write_json_once(path: Path, value: Mapping[str, Any]) -> str:
    path = Path(path)
    _reject_reparse_ancestors(path)
    payload = _canonical_json(dict(value))
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def event_stream_identity_sha256(events: Sequence[EventIdentity]) -> str:
    record = [{
        "position": event.position,
        "original_json_position": event.original_position,
        "event_id_sha256": event.event_id_sha256,
        "image_identity_sha256": event.image_identity_sha256,
        "source_group_sha256": event.source_group_sha256,
        "camera_id": event.camera_id,
    } for event in events]
    return hashlib.sha256(_canonical_json(record)).hexdigest()


class D2StageAClosedLoop:
    """Ten-arm, one-encoding-per-event D2 causal coordinator."""

    def __init__(self, *, category: str, run_id: str,
                 events: Sequence[EventIdentity],
                 feature_encoder: Callable[[EventIdentity], EncodedEvent],
                 arms: Mapping[str, ArmRuntime], ledger: D2ScoreLedger,
                 output_dir: Path, manifest_path: Path, manifest_sha256: str,
                 execution_authorization_id: str):
        self.category, self.run_id = str(category), str(run_id)
        if self.category not in D2_EXPECTED_COUNTS:
            raise D2Error("D2 Stage A category is not registered")
        self.events = tuple(events)
        if len(self.events) != D2_EXPECTED_COUNTS[self.category]:
            raise D2Error("D2 requires the full frozen TEST stream for each category")
        if any(event.position != i for i, event in enumerate(self.events)):
            raise D2Error("D2 event identities are not contiguous in shuffled order")
        if set(arms) != set(D2_ARM_IDS) or tuple(ledger.arm_ids) != D2_ARM_IDS:
            raise D2Error("D2 coordinator arm registry differs from its joint ledger")
        if not execution_authorization_id:
            raise D2Error("D2 Stage A requires the authorized experiment ID")
        self.feature_encoder = feature_encoder
        self.arms = dict(arms)
        self.ledger = ledger
        self.output_dir = Path(output_dir)
        self.manifest_path = Path(manifest_path)
        self.manifest_sha256 = str(manifest_sha256)
        self.execution_authorization_id = str(execution_authorization_id)
        self.last_event = -1

    def run(self) -> D2StageASeal:
        query_counts = {arm: 0 for arm in D2_ARM_IDS}
        closed = False
        try:
            if _sha256_file(self.manifest_path) != self.manifest_sha256:
                raise D2Error("D2 run manifest changed after Stage A initialization")
            for event in self.events:
                if event.position != self.last_event + 1:
                    raise D2Error("D2 Stage A event stream skipped or duplicated a position")
                start = time.perf_counter()
                encoded = self.feature_encoder(event)
                if not isinstance(encoded, EncodedEvent):
                    raise D2Error("D2 feature encoder must return one EncodedEvent")
                encode_seconds = max(float(encoded.encoding_seconds),
                                     time.perf_counter() - start)
                scored: dict[str, tuple[
                    NativePreEventScore, BEAComponents, Mapping[str, Any], float
                ]] = {}
                proposals: dict[str, D2Proposal] = {}
                suppressed: dict[str, bool] = {}
                arm_rows: dict[str, dict[str, Any]] = {}
                # First compute all ten pre-event native scores/components.
                for arm_id in D2_ARM_IDS:
                    arm = self.arms[arm_id]
                    score_started = time.perf_counter()
                    pre = arm.engine.score(event, encoded)
                    score_seconds = time.perf_counter() - score_started
                    novelty_percentile, novelty_info = arm.coverage.novelty_percentile(
                        encoded.global_cls)
                    if "distance_nn" not in novelty_info:
                        raise D2Error("Coverage Memory omitted the raw distance_nn receipt")
                    components = compute_bea_components(
                        score=pre.gc_score,
                        threshold=float(arm.engine.current_threshold),
                        c64_scores=pre.c64_scores,
                        novelty_percentile=novelty_percentile,
                        correction_top32=pre.correction_top32,
                        correction_bank_size=pre.correction_bank_size,
                        topk_dN_max=pre.topk_dN_max,
                        topk_dN_mean=pre.topk_dN_mean)
                    scored[arm_id] = (pre, components, dict(novelty_info),
                                      score_seconds)

                # Next decide every policy proposal from its own past state.
                for arm_id in D2_ARM_IDS:
                    arm = self.arms[arm_id]
                    pre, components, novelty_info, _score_seconds = scored[arm_id]
                    proposal = arm.controller.propose(
                        position=event.position,
                        event_id_sha256=event.event_id_sha256,
                        components=components,
                        novelty_distance_nn=float(novelty_info["distance_nn"]))
                    if arm_id in D2_BEA_TERMS and not math.isclose(
                            float(proposal.objective), _objective(components, arm_id),
                            rel_tol=0.0, abs_tol=1e-12):
                        raise D2Error(f"{arm_id}: BEA proposal differs from frozen components")
                    proposals[arm_id] = proposal

                # Apply the per-arm Source Group exclusion only after all ten
                # proposals have been fixed for this event.
                for arm_id in D2_ARM_IDS:
                    arm = self.arms[arm_id]
                    pre, components, novelty_info, score_seconds = scored[arm_id]
                    proposal = proposals[arm_id]
                    group_suppressed = bool(
                        proposal.proposed and event.source_group_sha256 in arm.queried_groups)
                    executed = bool(proposal.proposed and not group_suppressed)
                    suppressed[arm_id] = group_suppressed
                    arm_rows[arm_id] = {
                        "gc_score": float(pre.gc_score),
                        "c0_score": float(pre.c0_score),
                        "b3_threshold": float(arm.engine.current_threshold),
                        "c64_sigma": float(components.sigma),
                        "prediction": int(components.pred_defect),
                        "scorer_state_version": int(pre.scorer_state_version),
                        "scorer_state_sha256": str(pre.scorer_state_sha256),
                        "correction_bank_size": int(pre.correction_bank_size),
                        "query_proposed": bool(proposal.proposed),
                        "query_executed": executed,
                        "group_suppressed": group_suppressed,
                        "suppression_reason": ("source_group_already_queried"
                                                if group_suppressed else None),
                        "query_reason": proposal.reason,
                        "query_utility": proposal.utility,
                        "query_threshold": proposal.threshold,
                        "query_objective": proposal.objective,
                        "query_cost_runtime": proposal.query_cost,
                        "query_random_u": proposal.random_u,
                        "query_tie_probability": proposal.tie_probability,
                        "query_tie_u": proposal.tie_u,
                        "query_n_gt": proposal.n_gt,
                        "query_n_eq": proposal.n_eq,
                        "utility_history_count": proposal.history_count,
                        "encoding_seconds": encode_seconds,
                        "native_score_seconds": score_seconds,
                        "novelty_info": dict(novelty_info),
                        "components": components.as_dict(),
                    }
                # This fsync is the common disclosure barrier for all ten arms.
                self.ledger.seal_event(event, arm_rows)
                for arm_id in D2_ARM_IDS:
                    arm = self.arms[arm_id]
                    pre, _components, _novelty_info, _score_seconds = scored[arm_id]
                    if arm_rows[arm_id]["query_executed"]:
                        arm.queried_groups.add(event.source_group_sha256)
                        receipt = self.ledger.query_receipt(arm_id, event)
                        label = arm.label_reader.label_for(event, receipt)
                        update = arm.engine.apply_feedback(event, encoded, pre, label)
                        if update.label != label:
                            raise D2Error("native arm feedback receipt changed the authorized label")
                        if label == 0:
                            arm.coverage.observe_normal(encoded.global_cls)
                        if arm.private_feedback is None:
                            raise D2Error("D2 queried arm has no private feedback sink")
                        arm.private_feedback.write(event, label, update)
                        query_counts[arm_id] += 1
                    else:
                        arm.engine.observe_unqueried(event, encoded, pre)
                    arm.controller.observe_final_action(
                        position=event.position,
                        proposal=proposals[arm_id],
                        executed_query=bool(arm_rows[arm_id]["query_executed"]),
                        group_suppressed=suppressed[arm_id])
                self.last_event = event.position
            if self.last_event + 1 != len(self.events):
                raise D2Error("D2 stream ended without all registered events")
            self.ledger.close()
            for arm in self.arms.values():
                arm.label_reader.close()
                if arm.private_feedback is not None:
                    arm.private_feedback.close()
            closed = True
            score_sha = _sha256_file(self.ledger.path)
            feedback_sha = {
                arm: _sha256_file(Path(self.arms[arm].private_feedback.path))
                for arm in D2_ARM_IDS}
            controller = {arm: self.arms[arm].controller.diagnostics()
                          for arm in D2_ARM_IDS}
            final_state = {
                arm: {
                    "scorer_state_version": int(self.arms[arm].engine.state_version),
                    "scorer_state_sha256": str(self.arms[arm].engine.state_digest),
                    "coverage_state_sha256": self.arms[arm].coverage.state_sha256(),
                    "queried_source_group_count": len(self.arms[arm].queried_groups),
                } for arm in D2_ARM_IDS}
            manifest_sha = _sha256_file(self.manifest_path)
            if manifest_sha != self.manifest_sha256:
                raise D2Error("D2 run manifest changed during Stage A execution")
            seal_record = {
                "schema": D2_SEAL_SCHEMA,
                "status": "COMPLETE",
                "category": self.category,
                "run_id": self.run_id,
                "event_count": len(self.events),
                "category_json_sha256": str(
                    self.arms["B"].engine.input_identity["category_json_sha256"]),
                "manifest_sha256": manifest_sha,
                "score_log": {"file": self.ledger.path.name, "sha256": score_sha},
                "feedback_artifacts": {
                    arm: {"file": Path(self.arms[arm].private_feedback.path).name,
                          "sha256": feedback_sha[arm]} for arm in D2_ARM_IDS},
                "arm_query_counts": query_counts,
                "arm_controller_diagnostics": controller,
                "final_arm_state": final_state,
            }
            seal_path = self.output_dir / "stage_a_seal.json"
            seal_sha = _write_json_once(seal_path, seal_record)
            return D2StageASeal(
                self.category, self.run_id, len(self.events), str(self.ledger.path),
                score_sha, feedback_sha, query_counts, controller,
                str(self.manifest_path), manifest_sha, str(seal_path), seal_sha,
                str(seal_record["category_json_sha256"]))
        finally:
            if not closed:
                try:
                    self.ledger.close()
                finally:
                    for arm in self.arms.values():
                        try:
                            arm.label_reader.close()
                        except Exception:
                            pass
                        if arm.private_feedback is not None:
                            try:
                                arm.private_feedback.close()
                            except Exception:
                                pass


@dataclass(frozen=True)
class D2ValidatedRun:
    category: str
    run_id: str
    run_dir: Path
    manifest: Mapping[str, Any]
    seal: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    manifest_sha256: str
    seal_sha256: str
    score_log_sha256: str


def validate_d2_stage_a(*, run_dir: Path, category: str, run_id: str,
                        expected_commit: str, contract_sha256: str,
                        expected_events: Sequence[EventIdentity] | None = None
                        ) -> D2ValidatedRun:
    """Fail-closed verification of one full D2 Stage A output namespace."""
    run_dir = Path(run_dir).resolve(strict=True)
    _reject_reparse_ancestors(run_dir)
    if category not in D2_EXPECTED_COUNTS:
        raise D2Error("D2 seal category is not registered")
    if (run_dir.name != f"e6_bea_d2_{run_id}_{category}" or
            (run_dir / "failure.json").exists()):
        raise D2Error(f"{category}: failed or misnamed D2 run cannot be promoted")
    namespace_path = run_dir / "run_namespace.json"
    manifest_path = run_dir / "run_manifest.json"
    seal_path = run_dir / "stage_a_seal.json"
    score_path = run_dir / "sealed_scores.jsonl"
    for path in (namespace_path, manifest_path, seal_path, score_path):
        _reject_reparse_ancestors(path)
    try:
        namespace = json.loads(namespace_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise D2Error(f"{category}: D2 Stage A namespace, manifest, or seal is unreadable") from exc
    expected_files = {
        "run_namespace.json", "run_manifest.json", "sealed_scores.jsonl",
        "stage_a_seal.json", *[f"private_feedback_{arm}.jsonl" for arm in D2_ARM_IDS]}
    if set(manifest.get("output_files", ())) != expected_files:
        raise D2Error(f"{category}: D2 output manifest file contract is incomplete")
    if {item.name for item in run_dir.iterdir()} != expected_files:
        raise D2Error(f"{category}: D2 output contains missing or unaccounted files")
    manifest_sha = _sha256_file(manifest_path)
    seal_sha = _sha256_file(seal_path)
    score_sha = _sha256_file(score_path)
    if (manifest.get("schema") != D2_MANIFEST_SCHEMA or
            manifest.get("status_at_creation") != "INCOMPLETE" or
            manifest.get("stage_b_enabled") is not False or
            manifest.get("run_id") != run_id or manifest.get("category") != category or
            manifest.get("approved_commit") != expected_commit or
            manifest.get("contract_sha256") != contract_sha256 or
            manifest.get("stream_event_count") != D2_EXPECTED_COUNTS[category] or
            manifest.get("registered_arm_ids") != list(D2_ARM_IDS) or
            namespace.get("run_id") != run_id or namespace.get("category") != category or
            namespace.get("status_at_creation") != "INCOMPLETE" or
            manifest.get("namespace_record_sha256") != _sha256_file(namespace_path)):
        raise D2Error(f"{category}: D2 manifest or namespace identity is invalid")
    if (seal.get("schema") != D2_SEAL_SCHEMA or seal.get("status") != "COMPLETE" or
            seal.get("category") != category or seal.get("run_id") != run_id or
            seal.get("event_count") != D2_EXPECTED_COUNTS[category] or
            seal.get("manifest_sha256") != manifest_sha or
            seal.get("score_log") != {"file": score_path.name, "sha256": score_sha} or
            seal.get("category_json_sha256") !=
            manifest.get("input_identity", {}).get("category_json_sha256")):
        raise D2Error(f"{category}: D2 completion seal does not match its inputs")
    if (seal.get("feedback_artifacts", {}).keys() != set(D2_ARM_IDS) or
            seal.get("arm_query_counts", {}).keys() != set(D2_ARM_IDS) or
            seal.get("arm_controller_diagnostics", {}).keys() != set(D2_ARM_IDS) or
            seal.get("final_arm_state", {}).keys() != set(D2_ARM_IDS)):
        raise D2Error(f"{category}: D2 seal does not include all ten arms")
    feedback_sha: dict[str, str] = {}
    for arm in D2_ARM_IDS:
        record = seal["feedback_artifacts"][arm]
        path = run_dir / f"private_feedback_{arm}.jsonl"
        _reject_reparse_ancestors(path)
        if (record != {"file": path.name, "sha256": _sha256_file(path)} or
                len(str(record["sha256"])) != 64):
            raise D2Error(f"{category}: private feedback hash failed for {arm}")
        feedback_sha[arm] = str(record["sha256"])
    rows: list[dict[str, Any]] = []
    with score_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise D2Error(f"{category}: D2 score row lacks its durable newline")
            row = json.loads(line)
            _reject_public_label_fields(row)
            if (row.get("schema") != D2_STAGE_A_SCHEMA or
                    row.get("position") != len(rows) or
                    set(row.get("arm_rows", {})) != set(D2_ARM_IDS)):
                raise D2Error(f"{category}: D2 ledger positions or arm identities are inconsistent")
            rows.append(row)
    expected_count = D2_EXPECTED_COUNTS[category]
    if len(rows) != expected_count:
        raise D2Error(f"{category}: D2 score ledger is incomplete")
    actual_counts = {arm: 0 for arm in D2_ARM_IDS}
    for position, row in enumerate(rows):
        if expected_events is not None:
            event = expected_events[position]
            for field, actual in (
                    ("position", event.position),
                    ("original_json_position", event.original_position),
                    ("event_id_sha256", event.event_id_sha256),
                    ("image_identity_sha256", event.image_identity_sha256),
                    ("source_group_sha256", event.source_group_sha256)):
                if row.get(field) != actual:
                    raise D2Error(f"{category}: D2 score row {position} identity differs at {field}")
        for arm in D2_ARM_IDS:
            arm_row = row["arm_rows"][arm]
            if not isinstance(arm_row.get("query_executed"), bool):
                raise D2Error(f"{category}: {arm} query execution flag is invalid")
            if arm_row["query_executed"]:
                actual_counts[arm] += 1
    if (actual_counts != seal["arm_query_counts"] or
            actual_counts != {arm: int(seal["arm_controller_diagnostics"][arm]["executed_queries"])
                              for arm in D2_ARM_IDS}):
        raise D2Error(f"{category}: D2 query counts disagree across the ledger and seal")
    return D2ValidatedRun(category, run_id, run_dir, manifest, seal, tuple(rows),
                          manifest_sha, seal_sha, score_sha)


def require_four_complete_d2_runs(runs: Mapping[str, D2ValidatedRun]) -> None:
    """Stage B label access gate: no outcome loader is called before all seals pass."""
    if set(runs) != set(E6_REGISTERED_CATEGORIES):
        raise D2Error("D2 Stage B requires all four registered complete Stage A seals")
    for category, run in runs.items():
        if (not run.seal or run.seal.get("status") != "COMPLETE" or
                run.category != category or len(run.rows) != D2_EXPECTED_COUNTS[category] or
                not re.fullmatch(r"[0-9a-f]{64}", run.manifest_sha256) or
                not re.fullmatch(r"[0-9a-f]{64}", run.seal_sha256) or
                not re.fullmatch(r"[0-9a-f]{64}", run.score_log_sha256)):
            raise D2Error("D2 Stage B rejected an incomplete or mismatched Stage A seal")


def authorize_d2_stage_b(*, runs: Mapping[str, D2ValidatedRun],
                         authorization_id: str) -> None:
    """Require the separately registered Stage B authorization and four seals."""
    if authorization_id != D2_STAGE_B_AUTHORIZATION_ID:
        raise D2Error("D2 Stage B requires its exact registered authorization ID")
    require_four_complete_d2_runs(runs)


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise D2Error("metric distribution requires finite nonempty values")
    quantiles = np.quantile(array, [0.05, 0.5, 0.95])
    return {"count": int(len(array)), "mean": float(np.mean(array)),
            "median": float(quantiles[1]), "p05": float(quantiles[0]),
            "p95": float(quantiles[2]), "min": float(np.min(array)),
            "max": float(np.max(array))}


def _segment_bounds(length: int) -> list[tuple[int, int]]:
    # ceil boundaries yield four contiguous segments whose sizes differ by at most one.
    return [((i * length + 3) // 4, ((i + 1) * length + 3) // 4)
            for i in range(4)]


def d2_stage_b_evaluate(*, runs: Mapping[str, D2ValidatedRun],
                        labels_by_category: Mapping[str, np.ndarray],
                        authorization_id: str | None = None) -> dict[str, Any]:
    """Evaluate sealed D2 scores with the original E6 binary/ranking/window metrics."""
    authorize_d2_stage_b(runs=runs, authorization_id=str(authorization_id or ""))
    if set(labels_by_category) != set(E6_REGISTERED_CATEGORIES):
        raise D2Error("D2 Stage B label mapping must contain exactly the four development categories")
    from experiments.calibration import e6_bea_closed_loop as e6

    category_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for category in E6_REGISTERED_CATEGORIES:
        run = runs[category]
        y = np.asarray(labels_by_category[category], dtype=np.int8)
        if len(y) != len(run.rows) or not np.isin(y, (0, 1)).all():
            raise D2Error(f"{category}: authenticated Stage B labels differ from the sealed stream")
        if not np.any(y == 0) or not np.any(y == 1):
            raise D2Error(f"{category}: Stage B requires both true classes")
        arm_results: dict[str, dict[str, Any]] = {}
        for arm in D2_ARM_IDS:
            arm_rows = [row["arm_rows"][arm] for row in run.rows]
            score = np.asarray([float(row["gc_score"]) for row in arm_rows], dtype=np.float64)
            threshold = np.asarray([float(row["b3_threshold"]) for row in arm_rows], dtype=np.float64)
            pred = np.asarray([int(row["prediction"]) for row in arm_rows], dtype=np.int8)
            recomputed_pred = (score >= threshold).astype(np.int8)
            if not np.array_equal(pred, recomputed_pred):
                raise D2Error(f"{category}/{arm}: stored prequential predictions differ from score >= B3")
            metrics = e6._binary_metrics(y, pred, score)
            qcount = int(run.seal["arm_query_counts"][arm])
            controller = run.seal["arm_controller_diagnostics"][arm]
            metrics.update({
                "query_count": qcount,
                "query_rate": qcount / len(y),
                "proposal_count": int(controller["proposals"]),
                "source_group_suppressed": int(controller["group_suppressed"]),
                "quantile_comparisons": int(controller["quantile_comparisons"]),
                "quantile_ties": int(controller["quantile_ties"]),
                "tie_lottery_calls": int(controller["tie_lottery_calls"]),
                "max_query_free_interval_events": int(
                    controller["max_query_free_interval_events"]),
                "max_query_gap": controller["max_query_gap"],
                "query_free_interval_lengths": list(
                    controller["query_free_interval_lengths"]),
                "query_cost_final": controller["query_cost_runtime"],
                "worst_256_window_ber": e6._worst_window_ber(y, pred, width=256),
                "b3_threshold": _distribution(threshold),
                "correction_bank_size": _distribution([
                    float(row["correction_bank_size"]) for row in arm_rows]),
                "native_score_seconds_total": float(sum(
                    float(row.get("native_score_seconds") or 0.0) for row in arm_rows)),
                "query_utility": (_distribution([
                    float(row["query_utility"]) for row in arm_rows
                    if row.get("query_utility") is not None])
                    if any(row.get("query_utility") is not None for row in arm_rows) else None),
                "query_threshold": (_distribution([
                    float(row["query_threshold"]) for row in arm_rows
                    if row.get("query_threshold") is not None])
                    if any(row.get("query_threshold") is not None for row in arm_rows) else None),
            })
            segments = []
            for seg, (lo, hi) in enumerate(_segment_bounds(len(y)), start=1):
                local_y, local_pred = y[lo:hi], pred[lo:hi]
                local = None
                if np.any(local_y == 0) and np.any(local_y == 1):
                    local = e6._binary_metrics(local_y, local_pred,
                                               score[lo:hi])
                local_q = sum(bool(row["query_executed"])
                              for row in arm_rows[lo:hi])
                segments.append({
                    "segment": seg, "start_position": lo,
                    "stop_position_exclusive": hi,
                    "query_count": int(local_q),
                    "query_rate": float(local_q / (hi - lo)),
                    "mean_b3_threshold": float(np.mean(threshold[lo:hi])),
                    "score_margin": _distribution(score[lo:hi] - threshold[lo:hi]),
                    "metrics": local,
                })
            metrics["four_segment_temporal"] = segments
            arm_results[arm] = metrics
        category_metrics[category] = arm_results

    macro: dict[str, dict[str, float]] = {}
    for arm in D2_ARM_IDS:
        macro[arm] = {
            metric: float(np.mean([category_metrics[cat][arm][metric]
                                   for cat in E6_REGISTERED_CATEGORIES]))
            for metric in ("ber", "fpr", "fnr", "ba", "auroc", "ap",
                           "worst_256_window_ber", "query_rate")}
        macro[arm]["query_count_sum"] = int(sum(
            category_metrics[cat][arm]["query_count"] for cat in E6_REGISTERED_CATEGORIES))
    return {
        "schema": D2_EVALUATION_SCHEMA,
        "primary_metric": "category_equal_macro_prequential_ber",
        "classification": "DEVELOPMENT_EXPLORATORY",
        "stage_b_authorization_id": authorization_id,
        "category_metrics": category_metrics,
        "category_equal_macro": macro,
        "category_event_counts": {cat: len(runs[cat].rows)
                                  for cat in E6_REGISTERED_CATEGORIES},
    }


def classify_budget(*, rate_a: float, ber_a: float,
                    rate_b: float, ber_b: float,
                    tolerance: float = 0.003,
                    a_name: str = "A", b_name: str = "B") -> str:
    """Classify realized feedback budgets without normalizing the outcome metric."""
    if abs(float(rate_a) - float(rate_b)) <= tolerance:
        return "APPROXIMATELY_BUDGET_MATCHED"
    if rate_b > rate_a and ber_a < ber_b:
        return "BUDGET_DOMINANCE_EVIDENCE_IN_A_FAVOR"
    if rate_a < rate_b and ber_a <= ber_b:
        return f"{a_name}_DOMINATES_QUALITY_AND_EXPENDITURE"
    if rate_b < rate_a and ber_b <= ber_a:
        return f"{b_name}_DOMINATES_QUALITY_AND_EXPENDITURE"
    return "BUDGET_CONFOUNDED"


def factorial_and_policy_analysis(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Calculate registered D2 contrasts and expenditure classifications."""
    categories = evaluation["category_metrics"]
    macro = evaluation["category_equal_macro"]
    contrasts: dict[str, Any] = {}
    for scope in (*E6_REGISTERED_CATEGORIES, "CATEGORY_EQUAL_MACRO"):
        source = macro if scope == "CATEGORY_EQUAL_MACRO" else categories[scope]
        bea_losses = {arm: source[arm]["ber"] for arm in ("B", "BN", "BD", "BND", "BNCD")}
        n_effect = 0.5 * ((bea_losses["B"] - bea_losses["BN"]) +
                          (bea_losses["BD"] - bea_losses["BND"]))
        d_effect = 0.5 * ((bea_losses["B"] - bea_losses["BD"]) +
                          (bea_losses["BN"] - bea_losses["BND"]))
        interaction = bea_losses["BND"] - bea_losses["BN"] - bea_losses["BD"] + bea_losses["B"]
        contrasts[scope] = {
            "N_main_improvement": float(n_effect),
            "D_main_improvement": float(d_effect),
            "N_by_D_interaction": float(interaction),
            "C_delta_BER_BNCD_minus_BND": float(bea_losses["BNCD"] - bea_losses["BND"]),
        }
        rates = {arm: source[arm]["query_rate"] for arm in D2_ARM_IDS}
        for name, left, right in (
                ("B_vs_BN", "B", "BN"), ("BD_vs_BND", "BD", "BND"),
                ("B_vs_BD", "B", "BD"), ("BN_vs_BND", "BN", "BND"),
                ("BND_vs_BNCD", "BND", "BNCD")):
            count_key = "query_count_sum" if scope == "CATEGORY_EQUAL_MACRO" else "query_count"
            contrasts[scope].setdefault("budget_match", {})[name] = {
                "query_rate_difference": float(rates[left] - rates[right]),
                "label_count_difference": int(source[left][count_key] - source[right][count_key]),
                "classification": classify_budget(
                    rate_a=rates[left], ber_a=bea_losses[left],
                    rate_b=rates[right], ber_b=bea_losses[right],
                    a_name=left, b_name=right),
            }

    simple_arms = list(D2_SIMPLE_ARMS)
    strongest_simple = min(simple_arms, key=lambda arm: macro[arm]["ber"])
    primary_comparisons: dict[str, Any] = {}
    for arm in simple_arms:
        cat_rows = {}
        for cat in E6_REGISTERED_CATEGORIES:
            bea, simple = categories[cat]["BND"], categories[cat][arm]
            cat_rows[cat] = {
                "ber_difference_BND_minus_simple": float(bea["ber"] - simple["ber"]),
                "query_rate_difference_BND_minus_simple": float(
                    bea["query_rate"] - simple["query_rate"]),
                "label_count_difference_BND_minus_simple": int(
                    bea["query_count"] - simple["query_count"]),
                "classification": classify_budget(
                    rate_a=bea["query_rate"], ber_a=bea["ber"],
                    rate_b=simple["query_rate"], ber_b=simple["ber"],
                    a_name="BND", b_name=arm),
                "BND_fpr": float(bea["fpr"]), "BND_fnr": float(bea["fnr"]),
                "simple_fpr": float(simple["fpr"]), "simple_fnr": float(simple["fnr"]),
            }
        primary_comparisons[arm] = {
            "macro_ber_difference_BND_minus_simple": float(
                macro["BND"]["ber"] - macro[arm]["ber"]),
            "macro_query_rate_difference_BND_minus_simple": float(
                macro["BND"]["query_rate"] - macro[arm]["query_rate"]),
            "macro_label_count_difference_BND_minus_simple": int(
                macro["BND"]["query_count_sum"] - macro[arm]["query_count_sum"]),
            "category_wins_for_BND": int(sum(
                categories[cat]["BND"]["ber"] < categories[cat][arm]["ber"]
                for cat in E6_REGISTERED_CATEGORIES)),
            "category_losses_for_BND": int(sum(
                categories[cat]["BND"]["ber"] > categories[cat][arm]["ber"]
                for cat in E6_REGISTERED_CATEGORIES)),
            "category_classifications": cat_rows,
        }

    simple_margin = macro["BND"]["ber"] - macro[strongest_simple]["ber"]
    return {
        "primary_BND_vs_simple": {
            "strongest_simple_by_macro_ber": strongest_simple,
            "strongest_simple_macro_ber": float(macro[strongest_simple]["ber"]),
            "BND_macro_ber": float(macro["BND"]["ber"]),
            "macro_ber_difference_BND_minus_strongest_simple": float(simple_margin),
            "macro_query_rate_difference_BND_minus_strongest_simple": float(
                macro["BND"]["query_rate"] - macro[strongest_simple]["query_rate"]),
            "macro_label_count_difference_BND_minus_strongest_simple": int(
                macro["BND"]["query_count_sum"] - macro[strongest_simple]["query_count_sum"]),
            "macro_budget_classification": classify_budget(
                rate_a=macro["BND"]["query_rate"], ber_a=macro["BND"]["ber"],
                rate_b=macro[strongest_simple]["query_rate"],
                ber_b=macro[strongest_simple]["ber"], a_name="BND",
                b_name=strongest_simple),
            "candidate_practical_effect_threshold_absolute_ber": 0.01,
            "candidate_threshold_met": bool(simple_margin <= -0.01),
            "interpretation": "best-of-five-simple-baselines; not an unadjusted confirmatory contrast",
            "comparisons": primary_comparisons,
        },
        "N_by_D_factorial": contrasts,
        "C_on_vs_C_off": {
            "macro_delta_BER_BNCD_minus_BND": float(
                macro["BNCD"]["ber"] - macro["BND"]["ber"]),
            "macro_delta_AUROC_BNCD_minus_BND": float(
                macro["BNCD"]["auroc"] - macro["BND"]["auroc"]),
            "macro_delta_AP_BNCD_minus_BND": float(
                macro["BNCD"]["ap"] - macro["BND"]["ap"]),
            "macro_label_count_difference_BNCD_minus_BND": int(
                macro["BNCD"]["query_count_sum"] - macro["BND"]["query_count_sum"]),
            "macro_query_rate_difference_BNCD_minus_BND": float(
                macro["BNCD"]["query_rate"] - macro["BND"]["query_rate"]),
            "macro_budget_classification": classify_budget(
                rate_a=macro["BND"]["query_rate"], ber_a=macro["BND"]["ber"],
                rate_b=macro["BNCD"]["query_rate"], ber_b=macro["BNCD"]["ber"],
                a_name="BND", b_name="BNCD"),
            "resource_controlled_benefit_established": (
                abs(macro["BNCD"]["query_rate"] - macro["BND"]["query_rate"]) <= 0.003),
            "category_results": {
                cat: {
                    "delta_BER": float(categories[cat]["BNCD"]["ber"] -
                                        categories[cat]["BND"]["ber"]),
                    "delta_AUROC": float(categories[cat]["BNCD"]["auroc"] -
                                          categories[cat]["BND"]["auroc"]),
                    "delta_AP": float(categories[cat]["BNCD"]["ap"] -
                                       categories[cat]["BND"]["ap"]),
                    "BND_query_count": int(categories[cat]["BND"]["query_count"]),
                    "BNCD_query_count": int(categories[cat]["BNCD"]["query_count"]),
                    "BND_FPR": float(categories[cat]["BND"]["fpr"]),
                    "BND_FNR": float(categories[cat]["BND"]["fnr"]),
                    "BNCD_FPR": float(categories[cat]["BNCD"]["fpr"]),
                    "BNCD_FNR": float(categories[cat]["BNCD"]["fnr"]),
                    "budget_classification": classify_budget(
                        rate_a=categories[cat]["BND"]["query_rate"],
                        ber_a=categories[cat]["BND"]["ber"],
                        rate_b=categories[cat]["BNCD"]["query_rate"],
                        ber_b=categories[cat]["BNCD"]["ber"],
                        a_name="BND", b_name="BNCD"),
                } for cat in E6_REGISTERED_CATEGORIES},
        },
        "margin_entropy_equivalence": {},
        "novelty_nn_diagnostics": {
            cat: {
                "query_count": int(categories[cat]["NOVELTY_NN37"]["query_count"]),
                "query_rate": float(categories[cat]["NOVELTY_NN37"]["query_rate"]),
                "proposal_count": int(categories[cat]["NOVELTY_NN37"]["proposal_count"]),
                "source_group_suppressed": int(
                    categories[cat]["NOVELTY_NN37"]["source_group_suppressed"]),
                "quantile_ties": int(categories[cat]["NOVELTY_NN37"]["quantile_ties"]),
                "max_query_free_interval_events": int(
                    categories[cat]["NOVELTY_NN37"]["max_query_free_interval_events"]),
                "query_free_interval_lengths": list(
                    categories[cat]["NOVELTY_NN37"]["query_free_interval_lengths"]),
                "ber": float(categories[cat]["NOVELTY_NN37"]["ber"]),
                "auroc": float(categories[cat]["NOVELTY_NN37"]["auroc"]),
                "ap": float(categories[cat]["NOVELTY_NN37"]["ap"]),
            } for cat in E6_REGISTERED_CATEGORIES},
    }


def compare_public_trajectories(run: D2ValidatedRun, arm_a: str,
                                arm_b: str) -> dict[str, Any]:
    """Compare query and native state trajectories using only public score rows."""
    fields = ("gc_score", "c0_score", "b3_threshold", "scorer_state_version",
              "scorer_state_sha256", "correction_bank_size", "query_proposed",
              "query_executed", "group_suppressed")
    mismatch_counts = {field: 0 for field in fields}
    for row in run.rows:
        left, right = row["arm_rows"][arm_a], row["arm_rows"][arm_b]
        for field in fields:
            mismatch_counts[field] += int(left.get(field) != right.get(field))
    final_state = run.seal["final_arm_state"]
    final_state_equal = final_state[arm_a] == final_state[arm_b]
    query_identical = all(mismatch_counts[name] == 0 for name in (
        "query_proposed", "query_executed", "group_suppressed"))
    scorer_trajectory_identical = all(mismatch_counts[name] == 0 for name in (
        "gc_score", "b3_threshold", "scorer_state_version", "scorer_state_sha256",
        "correction_bank_size"))
    return {
        "arms": [arm_a, arm_b],
        "query_proposals_and_executions_identical": query_identical,
        "native_scorer_trajectory_identical": scorer_trajectory_identical,
        "final_mutable_state_receipts_identical": final_state_equal,
        "complete_logged_trajectory_identical": (
            query_identical and scorer_trajectory_identical and final_state_equal),
        "per_field_mismatch_counts": mismatch_counts,
    }


def compare_d2_to_frozen_e6(*, d2_runs: Mapping[str, D2ValidatedRun],
                            e6_root: Path) -> dict[str, Any]:
    """Label-blind parity comparison of D2 B/BND/BNCD against frozen E6 Q4/Q5/Q6."""
    from experiments.calibration import e6_bea_closed_loop as e6
    result: dict[str, Any] = {}
    for category in E6_REGISTERED_CATEGORIES:
        d2 = d2_runs[category]
        source_dir = Path(e6_root) / f"e6_bea_v2_E6FULL_20261104_{category}_fix1"
        try:
            old_manifest_path = source_dir / "run_manifest.json"
            old_seal_path = source_dir / "stage_a_seal.json"
            old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
            old_record = json.loads(old_seal_path.read_text(encoding="utf-8"))
            old_feedback = {arm: str(old_record["feedback_artifacts"][arm]["sha256"])
                            for arm in E6_ARM_IDS}
            old_seal = StageASeal(
                category=category, event_count=int(old_record["event_count"]),
                score_log_path=str(source_dir / "sealed_scores.jsonl"),
                score_log_sha256=str(old_record["score_log"]["sha256"]),
                private_feedback_sha256=old_feedback,
                arm_query_counts=old_record["arm_query_counts"],
                arm_controller_diagnostics=old_record["arm_controller_diagnostics"],
                complete=old_record.get("status") == "COMPLETE",
                manifest_path=str(old_manifest_path),
                manifest_sha256=str(old_record["manifest_sha256"]),
                seal_path=str(old_seal_path), seal_sha256=_sha256_file(old_seal_path),
                category_json_sha256=str(old_record["category_json_sha256"]))
            old_rows = e6._read_sealed_score_log(source_dir / "sealed_scores.jsonl", old_seal)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise D2Error(f"{category}: frozen E6 parity reference failed validation") from exc
        if len(old_rows) != len(d2.rows):
            raise D2Error(f"{category}: E6 parity reference event count differs from D2")
        arm_checks: dict[str, Any] = {}
        for d2_arm, e6_arm in D2_E6_PARITY_MAP.items():
            score_diffs: list[float] = []
            threshold_diffs: list[float] = []
            exact_fields = {
                "query_proposed": 0, "query_executed": 0,
                "scorer_state_version": 0, "scorer_state_sha256": 0,
            "position": 0, "original_json_position": 0,
            "event_id_sha256": 0, "image_identity_sha256": 0,
            "source_group_sha256": 0,
            }
            score_mismatch = threshold_mismatch = 0
            for drow, erow in zip(d2.rows, old_rows):
                for field in ("position", "original_json_position", "event_id_sha256",
                              "image_identity_sha256", "source_group_sha256"):
                    exact_fields[field] += int(drow.get(field) != erow.get(field))
                da, ea = drow["arm_rows"][d2_arm], erow["arm_rows"][e6_arm]
                for name in ("query_proposed", "query_executed",
                             "scorer_state_version", "scorer_state_sha256"):
                    exact_fields[name] += int(da.get(name) != ea.get(name))
                ds, es = float(da["gc_score"]), float(ea["gc_score"])
                dt, et = float(da["b3_threshold"]), float(ea["b3_threshold"])
                score_diffs.append(abs(ds - es))
                threshold_diffs.append(abs(dt - et))
                tol = d2.manifest.get("input_identity", {}).get("c64_score_tolerance", {})
                atol, rtol = float(tol.get("atol", 0.0)), float(tol.get("rtol", 0.0))
                score_mismatch += int(not math.isclose(ds, es, rel_tol=rtol, abs_tol=atol))
                threshold_mismatch += int(not math.isclose(dt, et, rel_tol=rtol, abs_tol=atol))
            d2_initial = d2.manifest.get("initial_arm_states", {}).get(d2_arm, {})
            d2_inputs = d2.manifest.get("support_and_c64_identity", {})
            old_inputs = old_manifest.get("support_and_c64_identity", {})
            identity_match = {
                "model_state_sha256": (
                    d2.manifest.get("model_identity", {}).get("loaded_model_state_sha256") ==
                    old_manifest.get("model_state_sha256")),
                "support_image_sha256": (
                    d2_inputs.get("support_image_sha256") ==
                    old_inputs.get("support_image_sha256")),
                "support_group_sha256": (
                    d2_inputs.get("support_group_sha256") ==
                    old_inputs.get("support_group_sha256")),
                "c64_image_sha256": (
                    d2_inputs.get("c64_image_sha256") ==
                    old_inputs.get("c64_image_sha256")),
                "c64_group_sha256": (
                    d2_inputs.get("c64_group_sha256") ==
                    old_inputs.get("c64_group_sha256")),
                "c64_score_archive_sha256": (
                    d2_inputs.get("c64_score_archive_sha256") ==
                    old_inputs.get("c64_score_archive_sha256")),
                "c64_initial_score_vector_sha256": (
                    d2_initial.get("c64_initial_score_vector_sha256") ==
                    old_inputs.get("c64_expected_score_vector_sha256")),
                "fixed_bank_sha256": (
                    d2_initial.get("fixed_bank_sha256") ==
                    old_inputs.get("fixed_bank_sha256")),
                "initial_scorer_state_sha256": (
                    d2_initial.get("scorer_state_sha256") ==
                    old_manifest.get("initial_complete_scorer_state_sha256")),
            }
            arm_checks[d2_arm] = {
                "compared_to_e6_arm": e6_arm,
                "initial_identities": identity_match,
                "initial_identities_all_match": all(identity_match.values()),
                "max_absolute_score_difference": max(score_diffs, default=0.0),
                "score_values_outside_registered_c64_tolerance": score_mismatch,
                "max_absolute_threshold_difference": max(threshold_diffs, default=0.0),
                "threshold_values_outside_registered_tolerance": threshold_mismatch,
                "exact_field_mismatch_counts": exact_fields,
                "realized_query_count_D2": int(d2.seal["arm_query_counts"][d2_arm]),
                "realized_query_count_E6": int(old_seal.arm_query_counts[e6_arm]),
            }
        result[category] = {
            "e6_reference_run_dir": str(source_dir),
            "e6_reference_score_sha256": old_seal.score_log_sha256,
            "e6_reference_manifest_sha256": old_seal.manifest_sha256,
            "arms": arm_checks,
        }
    return result
