"""Group-atomic Gate 1 orchestration with injected historical scoring bindings.

The module owns event order, query-cap decisions, outcome isolation and seals.
Numerical image scoring/update bindings are deliberately injected so this
orchestrator can be validated on synthetic CPU data without real inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import numbers
import re
import time
from typing import Any, Mapping, Protocol, Sequence

from .query import (BEA_ARMS, MARGIN_ARMS, PRINCIPAL_ARMS, BeaPrice, CategoryCap,
                    MarginRho, Proposal, QueryProtocolError)
from .telemetry import TelemetryLedger


class CampaignProtocolError(RuntimeError):
    """The category stream or causal state transition is invalid."""


STATIC_ARMS = ("STATIC-4+64", "STATIC-36+32")
_FORBIDDEN = frozenset({"label", "anomaly_class", "mask", "mask_path", "ground_truth", "gt"})


def static_arms_for_condition(condition_id: str) -> tuple[str, ...]:
    if condition_id == "P0":
        return STATIC_ARMS
    if condition_id in ("P1", "P2"):
        return ("STATIC-4+64",)
    raise CampaignProtocolError("unregistered paired condition")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_outcome_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN:
                raise CampaignProtocolError("Stage A object contains an outcome-only field")
            _reject_outcome_fields(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _reject_outcome_fields(child)


@dataclass(frozen=True)
class TestEvent:
    position: int
    event_id_sha256: str
    image_identity_sha256: str
    group_sha256: str
    camera_id: int
    feature_ref: Any = None
    role: str = "TEST"
    encoding_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("event_id_sha256", "image_identity_sha256", "group_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(getattr(self, name))):
                raise CampaignProtocolError(f"{name} must be a lowercase SHA-256 digest")
        if self.role != "TEST":
            raise CampaignProtocolError("only registered TEST observations may enter a stream")
        if (isinstance(self.position, bool) or not isinstance(self.position, numbers.Integral) or
                isinstance(self.camera_id, bool) or not isinstance(self.camera_id, numbers.Integral)):
            raise CampaignProtocolError("event position and camera ID must be integers")
        if self.encoding_seconds is not None and (
                not math.isfinite(float(self.encoding_seconds)) or float(self.encoding_seconds) < 0):
            raise CampaignProtocolError("image encoding telemetry is invalid")


@dataclass(frozen=True)
class ScorePacket:
    score: float
    prediction: int
    psi: float
    bea_objective: float
    state_version: int = 0
    state_sha256: str = ""
    diagnostics: Mapping[str, Any] | None = None
    threshold: float | None = None

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(x)) for x in (self.score, self.psi, self.bea_objective)):
            raise CampaignProtocolError("scorer returned a nonfinite prequential value")
        if (isinstance(self.prediction, bool) or not isinstance(self.prediction, numbers.Integral) or
                int(self.prediction) not in (0, 1)):
            raise CampaignProtocolError("prediction must be binary")
        if self.threshold is not None:
            if not math.isfinite(float(self.threshold)):
                raise CampaignProtocolError("prediction threshold must be finite")
            if int(self.prediction) != int(float(self.score) >= float(self.threshold)):
                raise CampaignProtocolError("prediction does not follow the packet's prequential threshold")
        if not 0.0 <= float(self.psi) <= 1.0:
            raise CampaignProtocolError("standardized-margin utility must be in [0,1]")
        if self.state_sha256 and not re.fullmatch(r"[0-9a-f]{64}", self.state_sha256):
            raise CampaignProtocolError("score state identity is malformed")
        if self.diagnostics is not None:
            _reject_outcome_fields(self.diagnostics)


class LabelBroker(Protocol):
    """A query-only interface. No Stage A scorer receives this object."""
    def disclose(self, *, arm_id: str, event: TestEvent,
                 query_receipt: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]: ...


class ArmBackend(Protocol):
    """Source-bound numerical adapters; every mutable arm is independent."""
    def snapshot(self, arm_id: str) -> Any: ...
    def score(self, arm_id: str, event: TestEvent, snapshot: Any) -> ScorePacket: ...
    def state_digest(self, arm_id: str) -> str: ...
    def commit_feedback(self, arm_id: str, event: TestEvent, packet: ScorePacket,
                        label: int, query_receipt: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def shadow_snapshot(self, fb_snapshot: Any) -> Any: ...
    def score_shadow(self, event: TestEvent, fb_snapshot: Any,
                     shadow_snapshot: Any) -> ScorePacket: ...
    def commit_shadow_feedback(self, event: TestEvent, packet: ScorePacket,
                               label: int, fb_query_receipt: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def shadow_state_digest(self) -> str: ...
    def static_score(self, baseline_id: str, event: TestEvent) -> ScorePacket: ...
    def static_state_digest(self, baseline_id: str) -> str: ...


@dataclass(frozen=True)
class StageASeal:
    dataset_id: str
    category: str
    run_condition: str
    expected_event_count: int
    completed_event_count: int
    group_count: int
    score_rows_sha256: str
    group_seals_sha256: str
    durable_seal_sha256: str
    status: str


class GroupAtomicCategoryRunner:
    """One category/condition trajectory with a mandatory five-view barrier."""

    def __init__(self, *, dataset_id: str, category: str, condition_id: str,
                 support_seed: int, stream_seed: int, expected_event_count: int,
                 expected_group_order: Sequence[str], rho: float,
                 backend: ArmBackend, broker: LabelBroker, sink: Any):
        expected_rho = {"realiad": 0.03, "visa_1cls": 0.10}.get(dataset_id)
        if expected_rho is None or float(rho) != expected_rho:
            raise CampaignProtocolError("dataset-specific query target differs from Gate 0")
        if dataset_id != "realiad":
            raise CampaignProtocolError("VisA online execution is blocked pending certified physical-group unit")
        if (isinstance(expected_event_count, bool) or
                not isinstance(expected_event_count, numbers.Integral) or
                int(expected_event_count) <= 0 or int(expected_event_count) % 5):
            raise CampaignProtocolError("category event count must be positive")
        if (condition_id not in {"P0", "P1", "P2"} or
                isinstance(support_seed, bool) or isinstance(stream_seed, bool) or
                not isinstance(support_seed, numbers.Integral) or
                not isinstance(stream_seed, numbers.Integral) or
                {"P0": (0, 0), "P1": (0, 11), "P2": (1, 0)}[condition_id] !=
                (int(support_seed), int(stream_seed))):
            raise CampaignProtocolError("paired condition or Support/Stream seed differs from Gate 0")
        if (len(expected_group_order) * 5 != int(expected_event_count) or
                len(set(expected_group_order)) != len(expected_group_order) or
                any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in expected_group_order)):
            raise CampaignProtocolError("registered Source Group order is incomplete or malformed")
        self.dataset_id, self.category = dataset_id, category
        self.condition_id = condition_id
        self.static_arms = static_arms_for_condition(condition_id)
        self.support_seed, self.stream_seed = int(support_seed), int(stream_seed)
        self.expected_event_count, self.rho = int(expected_event_count), float(rho)
        self.expected_group_order = tuple(expected_group_order)
        self.backend, self.broker = backend, broker
        if not callable(getattr(sink, "append_group", None)) or not callable(getattr(sink, "seal_category", None)):
            raise CampaignProtocolError("Stage A requires a durable fsync group sink")
        self.sink = sink
        self.telemetry = TelemetryLedger()
        self.caps = {arm: CategoryCap(self.expected_event_count, self.rho) for arm in PRINCIPAL_ARMS}
        self.bea = {arm: BeaPrice(arm, self.rho) for arm in BEA_ARMS}
        self.margin = {
            arm: MarginRho(dataset_id, category, arm, self.rho, self.expected_event_count,
                           self.support_seed, self.stream_seed)
            for arm in MARGIN_ARMS
        }
        self.rows: list[dict[str, Any]] = []
        self.group_seals: list[dict[str, Any]] = []
        self.event_count = 0
        self.group_count = 0
        self._seen_groups: set[str] = set()
        self._seen_events: set[str] = set()
        self._sealed = False
        self._query_receipt_hashes: dict[str, list[str]] = {arm: [] for arm in PRINCIPAL_ARMS}
        self._updates: dict[str, list[Mapping[str, Any]]] = {arm: [] for arm in PRINCIPAL_ARMS}
        self._executed_positions: dict[str, list[int]] = {arm: [] for arm in PRINCIPAL_ARMS}
        self._first_cap_exhaustion: dict[str, int | None] = {
            arm: (0 if self.caps[arm].limit == 0 else None) for arm in PRINCIPAL_ARMS}

    def process_group(self, events: Sequence[TestEvent]) -> tuple[Mapping[str, Any], ...]:
        if self._sealed:
            raise CampaignProtocolError("category has already been sealed")
        if len(events) != 5:
            raise CampaignProtocolError("Real-IAD Source Group must contain exactly five views")
        expected_positions = list(range(self.event_count, self.event_count + 5))
        if [event.position for event in events] != expected_positions:
            raise CampaignProtocolError("group event positions are missing, duplicated or noncontiguous")
        if [event.camera_id for event in events] != [1, 2, 3, 4, 5]:
            raise CampaignProtocolError("Real-IAD group camera order must be exactly 1 through 5")
        if any(event.feature_ref is None for event in events):
            raise CampaignProtocolError("group lacks the one shared immutable encoded-feature reference per event")
        group_ids = {event.group_sha256 for event in events}
        if len(group_ids) != 1 or next(iter(group_ids)) in self._seen_groups:
            raise CampaignProtocolError("Source Group identity is inconsistent or repeated")
        if len({event.event_id_sha256 for event in events}) != 5 or any(
                event.event_id_sha256 in self._seen_events for event in events):
            raise CampaignProtocolError("group contains duplicate event identity")
        if self.event_count + 5 > self.expected_event_count:
            raise CampaignProtocolError("group exceeds the registered category event count")

        group_sha = next(iter(group_ids))
        if group_sha != self.expected_group_order[self.group_count]:
            raise CampaignProtocolError("Source Group order differs from the frozen stream manifest")
        before_digest = {arm: self.backend.state_digest(arm) for arm in PRINCIPAL_ARMS}
        shadow_before = self.backend.shadow_state_digest()
        static_before = {arm: self.backend.static_state_digest(arm) for arm in self.static_arms}
        snapshots = {arm: self.backend.snapshot(arm) for arm in PRINCIPAL_ARMS}
        shadow_snapshot = self.backend.shadow_snapshot(snapshots["F-B"])

        packets: dict[tuple[str, int], ScorePacket] = {}
        static_packets: dict[tuple[str, int], ScorePacket] = {}
        for event_index, event in enumerate(events):
            _reject_outcome_fields(event.__dict__)
            if event.encoding_seconds is not None:
                self.telemetry.record_duration("image_encoding", event.encoding_seconds)
            self.telemetry.increment("test_image_events_scored")
            self.telemetry.increment("shared_feature_references")
            for arm in PRINCIPAL_ARMS:
                timer = time.perf_counter()
                packet = self.backend.score(arm, event, snapshots[arm])
                if packet.threshold is None:
                    raise CampaignProtocolError("principal score packet lacks its current prequential threshold")
                self.telemetry.record_duration(f"native_scoring:{arm}", time.perf_counter() - timer)
                if packet.state_sha256 and packet.state_sha256 != before_digest[arm]:
                    raise CampaignProtocolError("score was produced from an unregistered arm state")
                packets[(arm, event_index)] = packet
            timer = time.perf_counter()
            packet = self.backend.score_shadow(event, snapshots["F-B"], shadow_snapshot)
            self.telemetry.record_duration("native_scoring:N-shadow", time.perf_counter() - timer)
            if packet.state_sha256 and packet.state_sha256 != shadow_before:
                raise CampaignProtocolError("N-shadow score did not use its frozen pre-group state")
            packets[("N-shadow", event_index)] = packet
            for baseline in self.static_arms:
                timer = time.perf_counter()
                static_packet = self.backend.static_score(baseline, event)
                static_packets[(baseline, event_index)] = static_packet
                if static_packet.threshold is None:
                    raise CampaignProtocolError("static score packet lacks its frozen C32 threshold")
                self.telemetry.record_duration(f"static_scoring:{baseline}", time.perf_counter() - timer)
                if static_packet.state_sha256 and static_packet.state_sha256 != static_before[baseline]:
                    raise CampaignProtocolError("static score was produced from a changed baseline state")

        if ({arm: self.backend.state_digest(arm) for arm in PRINCIPAL_ARMS} != before_digest or
                self.backend.shadow_state_digest() != shadow_before or
                {arm: self.backend.static_state_digest(arm) for arm in self.static_arms} != static_before):
            raise CampaignProtocolError("a scorer mutated detector state before group sealing")

        # Freeze the controller state for the whole group. Every view sees the
        # same BEA price and pre-group Margin history.
        group_price = {arm: self.bea[arm].price for arm in BEA_ARMS}
        group_starts_cold = events[0].position < self.margin["F-M"].cold_start_events
        proposals: dict[tuple[str, int], Proposal] = {}
        for arm in PRINCIPAL_ARMS:
            for event_index, event in enumerate(events):
                packet = packets[(arm, event_index)]
                if arm in BEA_ARMS:
                    proposals[(arm, event_index)] = self.bea[arm].propose(
                        packet.bea_objective, group_price=group_price[arm])
                else:
                    proposals[(arm, event_index)] = self.margin[arm].propose_group_view(
                        position=event.position, event_id_sha256=event.event_id_sha256,
                        psi=packet.psi, group_starts_cold=group_starts_cold)

        action_rows: dict[tuple[str, int], dict[str, bool]] = {}
        for arm in PRINCIPAL_ARMS:
            group_used = False
            cap = self.caps[arm]
            for event_index in range(5):
                proposal = proposals[(arm, event_index)]
                cap.proposals += int(proposal.proposed)
                executed = group_suppressed = cap_suppressed = False
                if proposal.proposed:
                    if group_used:
                        group_suppressed = True
                    elif cap.remaining() <= 0:
                        cap_suppressed = True
                    else:
                        executed = True
                        group_used = True
                        cap.executed += 1
                        self._executed_positions[arm].append(events[event_index].position)
                        if cap.executed == cap.limit:
                            self._first_cap_exhaustion[arm] = events[event_index].position
                cap.group_suppressed += int(group_suppressed)
                cap.cap_suppressed += int(cap_suppressed)
                action_rows[(arm, event_index)] = {
                    "proposed": bool(proposal.proposed), "executed": executed,
                    "group_suppressed": group_suppressed,
                    "cap_suppressed": cap_suppressed,
                }

        # Query receipt identities are knowable from metadata and are fixed in
        # the public seal before any label disclosure occurs.
        query_receipts: dict[tuple[str, int], Mapping[str, Any]] = {}
        for arm in PRINCIPAL_ARMS:
            for event_index, event in enumerate(events):
                if action_rows[(arm, event_index)]["executed"]:
                    query_receipts[(arm, event_index)] = {
                        "schema": "final-benchmark.query-receipt.v1",
                        "arm_id": arm, "category": self.category,
                        "condition_id": self.condition_id,
                        "event_id_sha256": event.event_id_sha256,
                        "source_group_sha256": event.group_sha256,
                        "position": event.position,
                    }

        # Public score rows are constructed and hashed before label access.
        group_rows: list[dict[str, Any]] = []
        for event_index, event in enumerate(events):
            row: dict[str, Any] = {
                "position": event.position,
                "event_id_sha256": event.event_id_sha256,
                "image_identity_sha256": event.image_identity_sha256,
                "source_group_sha256": event.group_sha256,
                "camera_id": event.camera_id,
                "arm_scores": {}, "arm_predictions": {}, "arm_thresholds": {}, "query_actions": {},
                "static_scores": {}, "n_shadow_score": packets[("N-shadow", event_index)].score,
                "n_shadow_state_version": int(packets[("N-shadow", event_index)].state_version),
                "n_shadow_state_sha256": packets[("N-shadow", event_index)].state_sha256 or shadow_before,
            }
            for arm in PRINCIPAL_ARMS:
                packet = packets[(arm, event_index)]
                proposal = proposals[(arm, event_index)]
                row["arm_scores"][arm] = float(packet.score)
                row["arm_predictions"][arm] = int(packet.prediction)
                row["arm_thresholds"][arm] = float(packet.threshold)
                row.setdefault("arm_state_versions", {})[arm] = int(packet.state_version)
                row.setdefault("arm_state_sha256", {})[arm] = packet.state_sha256 or before_digest[arm]
                row["query_actions"][arm] = {
                    **action_rows[(arm, event_index)],
                    "proposal_reason": proposal.reason,
                    "proposal_utility": proposal.utility,
                    "proposal_threshold": proposal.threshold,
                    "proposal_objective": proposal.objective,
                    "proposal_price": proposal.price,
                    "query_receipt_sha256": (_sha(_canonical(query_receipts[(arm, event_index)]))
                                              if action_rows[(arm, event_index)]["executed"] else None),
                    "state_version": int(packet.state_version),
                    "state_sha256": packet.state_sha256 or before_digest[arm],
                }
            for baseline in self.static_arms:
                row["static_scores"][baseline] = float(static_packets[(baseline, event_index)].score)
                row.setdefault("static_predictions", {})[baseline] = int(static_packets[(baseline, event_index)].prediction)
                row.setdefault("static_thresholds", {})[baseline] = float(static_packets[(baseline, event_index)].threshold)
                row.setdefault("static_state_sha256", {})[baseline] = static_packets[(baseline, event_index)].state_sha256 or static_before[baseline]
            _reject_outcome_fields(row)
            group_rows.append(row)
        group_rows_sha = _sha(_canonical(group_rows))
        group_record = {
            "group_index": self.group_count,
            "source_group_sha256": group_sha,
            "start_position": events[0].position,
            "end_position_exclusive": events[-1].position + 1,
            "event_order_sha256": _sha(_canonical([event.event_id_sha256 for event in events])),
            "score_rows_sha256": group_rows_sha,
            "status": "SEALED_BEFORE_LABEL_DISCLOSURE",
        }
        timer = time.perf_counter()
        durable_group_receipt_sha = self.sink.append_group(
            group_seal=group_record, score_rows=group_rows)
        self.telemetry.record_duration("group_journal_fsync", time.perf_counter() - timer)
        if not re.fullmatch(r"[0-9a-f]{64}", str(durable_group_receipt_sha)):
            raise CampaignProtocolError("durable group sink returned a malformed commit receipt")
        self.group_seals.append({**group_record, "durable_group_receipt_sha256": durable_group_receipt_sha})

        # Query policy state is advanced only after all five score rows are
        # sealed; all proposals used the pre-group state captured above.
        for arm in MARGIN_ARMS:
            self.margin[arm].observe_group([packets[(arm, i)].psi for i in range(5)])
        for arm in BEA_ARMS:
            actions = [bool(action_rows[(arm, i)]["executed"]) for i in range(5)]
            self.bea[arm].proposals += sum(int(proposals[(arm, i)].proposed) for i in range(5))
            self.bea[arm].group_suppressed += sum(int(action_rows[(arm, i)]["group_suppressed"]) for i in range(5))
            self.bea[arm].cap_suppressed += sum(int(action_rows[(arm, i)]["cap_suppressed"]) for i in range(5))
            self.bea[arm].observe_final_actions(actions)

        # Only executed query receipts may cross the private broker boundary.
        fb_disclosed = False
        arms_with_feedback: set[str] = set()
        for arm in PRINCIPAL_ARMS:
            for event_index, event in enumerate(events):
                action = action_rows[(arm, event_index)]
                if not action["executed"]:
                    continue
                query_receipt = query_receipts[(arm, event_index)]
                timer = time.perf_counter()
                label, private_receipt = self.broker.disclose(
                    arm_id=arm, event=event, query_receipt=query_receipt)
                self.telemetry.record_duration("query_broker_disclosure", time.perf_counter() - timer)
                if (isinstance(label, bool) or not isinstance(label, numbers.Integral) or
                        int(label) not in (0, 1) or not isinstance(private_receipt, Mapping)):
                    raise CampaignProtocolError("query broker returned an invalid verified label")
                private_digest = _sha(_canonical({"receipt": private_receipt, "label": int(label)}))
                if arm == "F-B":
                    # The shadow's C1 writer must consume the same pre-feedback
                    # C0/Greedy evidence as F-B, before F-B can update Greedy.
                    principal_before_shadow = {name: self.backend.state_digest(name) for name in PRINCIPAL_ARMS}
                    shadow_state_before_feedback = self.backend.shadow_state_digest()
                    timer = time.perf_counter()
                    shadow_update = self.backend.commit_shadow_feedback(
                        event, packets[(arm, event_index)], int(label), private_receipt)
                    self.telemetry.record_duration("feedback_update:N-shadow", time.perf_counter() - timer)
                    _reject_outcome_fields(shadow_update)
                    if ({name: self.backend.state_digest(name) for name in PRINCIPAL_ARMS} != principal_before_shadow or
                            {name: self.backend.static_state_digest(name) for name in self.static_arms} != static_before):
                        raise CampaignProtocolError("N-shadow update changed a principal or static trajectory")
                    if int(label) == 1 and self.backend.shadow_state_digest() != shadow_state_before_feedback:
                        raise CampaignProtocolError("N-shadow accepted a defect label into its normal-only correction")
                    fb_disclosed = True
                other_digests = {other: self.backend.state_digest(other)
                                 for other in PRINCIPAL_ARMS if other != arm}
                timer = time.perf_counter()
                update = self.backend.commit_feedback(arm, event, packets[(arm, event_index)],
                                                      int(label), private_receipt)
                self.telemetry.record_duration(f"feedback_update:{arm}", time.perf_counter() - timer)
                if ({other: self.backend.state_digest(other) for other in other_digests} != other_digests or
                        {baseline: self.backend.static_state_digest(baseline) for baseline in self.static_arms} != static_before):
                    raise CampaignProtocolError("feedback update escaped its arm or mutated a static baseline")
                _reject_outcome_fields(update)
                self._updates[arm].append(dict(update))
                arms_with_feedback.add(arm)
                self._query_receipt_hashes[arm].append(private_digest)

        if not fb_disclosed and self.backend.shadow_state_digest() != shadow_before:
            raise CampaignProtocolError("N-shadow changed state without inheriting F-B feedback")

        for arm in PRINCIPAL_ARMS:
            if arm not in arms_with_feedback and self.backend.state_digest(arm) != before_digest[arm]:
                raise CampaignProtocolError("an unqueried arm changed state at the group boundary")

        # State writes are legal only after every row in this Source Group was
        # sealed and the query broker returned the purchased event's label.
        for event in events:
            self._seen_events.add(event.event_id_sha256)
        self._seen_groups.add(group_sha)
        self.rows.extend(group_rows)
        self.event_count += 5
        self.group_count += 1
        return tuple(group_rows)

    def seal(self) -> StageASeal:
        if self.event_count != self.expected_event_count:
            raise CampaignProtocolError("Stage A category cannot seal before every registered event is complete")
        if self.event_count % 5 or self.group_count != len(self.expected_group_order):
            raise CampaignProtocolError("Real-IAD category ends with an incomplete Source Group")
        self._sealed = True
        digest = _sha(_canonical(self.rows))
        group_seal_digest = _sha(_canonical(self.group_seals))
        seal_payload = {
            "dataset_id": self.dataset_id, "category": self.category,
            "run_condition": self.condition_id,
            "expected_event_count": self.expected_event_count,
            "completed_event_count": self.event_count,
            "group_count": self.group_count,
            "score_rows_sha256": digest,
            "group_seals_sha256": group_seal_digest,
            "status": "SEALED_COMPLETE",
        }
        timer = time.perf_counter()
        durable_sha = self.sink.seal_category(seal_payload)
        self.telemetry.record_duration("category_seal_fsync", time.perf_counter() - timer)
        return StageASeal(self.dataset_id, self.category, self.condition_id,
                          self.expected_event_count, self.event_count, self.group_count,
                          digest, group_seal_digest, durable_sha, "SEALED_COMPLETE")

    def accounting(self) -> dict[str, Any]:
        def longest_no_query_gap(positions: Sequence[int]) -> int:
            if not positions:
                return self.expected_event_count
            gaps = [int(positions[0])]
            gaps.extend(int(right) - int(left) - 1 for left, right in zip(positions, positions[1:]))
            gaps.append(self.expected_event_count - 1 - int(positions[-1]))
            return max(gaps, default=0)

        result = {
            arm: {
                "cap": self.caps[arm].limit,
                "unspent_cap": self.caps[arm].limit - self.caps[arm].executed,
                "executed": self.caps[arm].executed,
                "proposal_count": self.caps[arm].proposals,
                "group_suppressed": self.caps[arm].group_suppressed,
                "cap_suppressed": self.caps[arm].cap_suppressed,
                "realized_rate": self.caps[arm].executed / self.expected_event_count,
                "query_receipt_count": len(self._query_receipt_hashes[arm]),
                "first_cap_exhaustion_position": self._first_cap_exhaustion[arm],
                "cap_exhausted": self.caps[arm].executed >= self.caps[arm].limit,
                "max_no_query_gap_events": longest_no_query_gap(self._executed_positions[arm]),
                "final_bea_query_price": self.bea[arm].price if arm in BEA_ARMS else None,
                "bea_executed_action_history_events": len(self.bea[arm].flags) if arm in BEA_ARMS else None,
                "margin_utility_history_events": len(self.margin[arm].history) if arm in MARGIN_ARMS else None,
                "margin_cold_start_events": self.margin[arm].cold_start_events if arm in MARGIN_ARMS else None,
                "margin_quantile_comparisons": self.margin[arm].quantile_count if arm in MARGIN_ARMS else None,
                "margin_quantile_ties": self.margin[arm].tie_count if arm in MARGIN_ARMS else None,
            }
            for arm in PRINCIPAL_ARMS
        }
        return result

    def resource_report(self) -> dict[str, Any]:
        return self.telemetry.report()
