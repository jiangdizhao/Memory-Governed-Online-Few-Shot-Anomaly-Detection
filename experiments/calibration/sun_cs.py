"""Sun et al. ICML 2024 Algorithm 3 with declared finite-sample conventions.

The online target is the score-stream quantile event, not a physical-defect
label. The paper's theorem is not asserted for dependent scores produced by a
changing Greedy memory. No offline score matching or feedback adaptation is
implemented here.

This module separates a deliberately simple reference implementation from an
exact coordinate-compressed persistent order-statistic implementation. Both
use the equations in the paper. The finite-sample choices needed to execute
them are versioned by ``SunCSEndpointPolicy``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
import tracemalloc
from typing import Any, Iterable, Literal, Sequence

from .protocol import Observation, Prediction


class SunCSReferenceError(ValueError):
    """A reference calculation is invalid."""


class SunCSUndefinedSourceBehavior(SunCSReferenceError):
    """A requested operation needs an explicitly selected convention."""


Interval = tuple[float, float]
EndpointPolicy = Literal["empirical_extrema", "conservative_extended_real"]
SUN_CS_VERSION = "B6-SUN-CS-A3-PAPER-EQUATIONS-DECLARED-CONVENTIONS-v1.0.0"


@dataclass(frozen=True)
class SunCSEndpointPolicy:
    name: EndpointPolicy = "empirical_extrema"
    status: str = "PAPER_EQUATIONS_WITH_DECLARED_ENDPOINT_CONVENTIONS"

    def __post_init__(self) -> None:
        expected = ("PAPER_EQUATIONS_WITH_DECLARED_ENDPOINT_CONVENTIONS"
                    if self.name == "empirical_extrema" else
                    "CONSERVATIVE_ENDPOINT_SENSITIVITY")
        if self.name not in ("empirical_extrema", "conservative_extended_real"):
            raise ValueError("Unknown Sun-CS endpoint policy")
        if self.status != expected:
            raise ValueError("Sun-CS endpoint policy status/version mismatch")


PRIMARY_ENDPOINT_POLICY = SunCSEndpointPolicy()
EXTENDED_REAL_ENDPOINT_POLICY = SunCSEndpointPolicy(
    "conservative_extended_real", "CONSERVATIVE_ENDPOINT_SENSITIVITY")


def confidence_radius(t: int, delta: float) -> float:
    """Equation (2), preserving its raw radius (including values above one)."""
    if not 0.0 < float(delta) <= 1.0:
        raise SunCSReferenceError("delta must be in (0,1]")
    if int(t) != t or t < 0:
        raise SunCSReferenceError("t must be a nonnegative integer")
    if t == 0:
        return 1.0
    raw = (0.85 / math.sqrt(t) * math.sqrt(
        math.log(math.log(math.e * t)) + 0.8 * math.log(1612.0 / delta)))
    if not math.isfinite(raw):
        raise SunCSReferenceError("Equation (2) produced a non-finite radius")
    return raw


def empirical_quantile(q: float, values: Iterable[float], *,
                       endpoint_policy: SunCSEndpointPolicy = PRIMARY_ENDPOINT_POLICY
                       ) -> float:
    """Paper midpoint quantile with the registered endpoint/rank convention.

    The paper specifies midpoint order statistics at floor(q*n), ceil(q*n),
    one-based. At q=0 the primary variant selects the minimum; q=1 selects the
    maximum. For a strictly interior q with floor(q*n)=0, both rank lookups
    are extended to the first order statistic. The secondary diagnostic uses
    -inf/+inf at exact q endpoints and is not author-code parity.
    """
    q = float(q)
    ordered = sorted(float(value) for value in values)
    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise SunCSReferenceError("q must be finite and in [0,1]")
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise SunCSUndefinedSourceBehavior(
            "Empirical quantiles require a nonempty finite sample")
    if q == 0.0:
        if endpoint_policy.name == "conservative_extended_real":
            return -math.inf
        return ordered[0]
    if q == 1.0:
        if endpoint_policy.name == "conservative_extended_real":
            return math.inf
        return ordered[-1]
    rank = q * len(ordered)
    lower_rank = max(1, math.floor(rank))
    upper_rank = max(1, math.ceil(rank))
    lower, upper = ordered[lower_rank - 1], ordered[upper_rank - 1]
    return lower / 2.0 + upper / 2.0


def _levels(p: float, delta: float, n: int) -> tuple[float, float]:
    if not 0.0 < float(p) < 1.0:
        raise SunCSReferenceError("p must be in (0,1)")
    radius = confidence_radius(n, delta)
    return max(float(p) - 2.0 * radius, 0.0), min(float(p) + 2.0 * radius, 1.0)


def quantile_confidence_set(p: float, delta: float, values: Sequence[float], *,
                            endpoint_policy: SunCSEndpointPolicy = PRIMARY_ENDPOINT_POLICY
                            ) -> Interval:
    """Equation (1) for a nonempty prefix, with a selected endpoint policy."""
    if not values:
        raise SunCSUndefinedSourceBehavior(
            "The empty historical prefix is handled as ABSTAIN_COLD_START")
    low_q, high_q = _levels(p, delta, len(values))
    return (empirical_quantile(low_q, values, endpoint_policy=endpoint_policy),
            empirical_quantile(high_q, values, endpoint_policy=endpoint_policy))


def classify_score(score: float, confidence_set: Interval) -> int | None:
    """Algorithm 1: strict upper anomaly, lower benign, inclusive abstention."""
    value = float(score)
    lower, upper = map(float, confidence_set)
    if (not math.isfinite(value) or math.isnan(lower) or math.isnan(upper) or
            lower > upper):
        raise SunCSReferenceError("Invalid score or confidence set")
    if value > upper:
        return 1
    if value < lower:
        return 0
    return None


def intersect_intervals(intervals: Sequence[Interval]) -> Interval | None:
    if not intervals:
        raise SunCSUndefinedSourceBehavior(
            "Algorithm 2 requires nonempty prefix and suffix families")
    lower = max(float(interval[0]) for interval in intervals)
    upper = min(float(interval[1]) for interval in intervals)
    if any(math.isnan(value) for interval in intervals for value in interval):
        raise SunCSReferenceError("NaN interval endpoint")
    return None if lower > upper else (lower, upper)


def change_detected(forward_prefix_sets: Sequence[Interval],
                    backward_suffix_sets: Sequence[Interval]) -> bool:
    """Algorithm 2 predicate on already constructed confidence sets."""
    forward = intersect_intervals(forward_prefix_sets)
    backward = intersect_intervals(backward_suffix_sets)
    if forward is None or backward is None:
        return True
    return max(forward[0], backward[0]) > min(forward[1], backward[1])


def _intersection_or_empty(intervals: Sequence[Interval]) -> Interval | None:
    lower = max(x[0] for x in intervals)
    upper = min(x[1] for x in intervals)
    return None if lower > upper else (lower, upper)


def algorithm2_reference(values: Sequence[float], p: float = 0.99,
                          delta: float = 0.01, *,
                          endpoint_policy: SunCSEndpointPolicy = PRIMARY_ENDPOINT_POLICY
                          ) -> dict[str, Any]:
    """Independent O(n^2 log n) Algorithm 2 reference for small sequences."""
    if not values:
        raise SunCSUndefinedSourceBehavior("Algorithm 2 requires at least one score")
    data = tuple(float(x) for x in values)
    if any(not math.isfinite(x) for x in data):
        raise SunCSReferenceError("Algorithm 2 scores must be finite")
    forward = [quantile_confidence_set(p, delta, data[:stop],
                                      endpoint_policy=endpoint_policy)
               for stop in range(1, len(data) + 1)]
    backward = [quantile_confidence_set(p, delta, data[start:],
                                       endpoint_policy=endpoint_policy)
                for start in range(len(data))]
    f_intersection = _intersection_or_empty(forward)
    b_intersection = _intersection_or_empty(backward)
    detected = (f_intersection is None or b_intersection is None or
                max(f_intersection[0], b_intersection[0]) >
                min(f_intersection[1], b_intersection[1]))
    return {"forward_sets": forward, "backward_sets": backward,
            "forward_intersection": f_intersection,
            "backward_intersection": b_intersection,
            "change_detected": detected}


class _PersistentOrderStatistics:
    """Persistent exact count tree over a coordinate-compressed score set."""
    def __init__(self, values: Sequence[float]):
        self.keys = sorted(set(float(x) for x in values))
        if not self.keys:
            raise SunCSUndefinedSourceBehavior("Order-statistic tree needs observed scores")
        self.left, self.right, self.count = [0], [0], [0]
        self.roots = [0]
        for value in values:
            self.roots.append(self._insert(
                self.roots[-1], 0, len(self.keys) - 1,
                self._key_position(float(value))))

    def _key_position(self, value: float) -> int:
        import bisect
        return bisect.bisect_left(self.keys, value)

    def _copy_node(self, node: int) -> int:
        self.left.append(self.left[node])
        self.right.append(self.right[node])
        self.count.append(self.count[node] + 1)
        return len(self.count) - 1

    def _insert(self, old: int, lo: int, hi: int, position: int) -> int:
        new = self._copy_node(old)
        if lo != hi:
            mid = (lo + hi) // 2
            if position <= mid:
                self.left[new] = self._insert(self.left[old], lo, mid, position)
            else:
                self.right[new] = self._insert(self.right[old], mid + 1, hi,
                                               position)
        return new

    def kth_difference(self, after: int, before: int, rank: int) -> float:
        """Return one-based rank in multiset(after)-multiset(before)."""
        if not 1 <= rank <= self.count[after] - self.count[before]:
            raise SunCSReferenceError("Order-statistic rank outside range")
        a, b = after, before
        lo, hi = 0, len(self.keys) - 1
        while lo != hi:
            left_count = self.count[self.left[a]] - self.count[self.left[b]]
            mid = (lo + hi) // 2
            if rank <= left_count:
                a, b, hi = self.left[a], self.left[b], mid
            else:
                rank -= left_count
                a, b, lo = self.right[a], self.right[b], mid + 1
        return self.keys[lo]

    def _quantile(self, after: int, before: int, n: int, q: float,
                  policy: SunCSEndpointPolicy) -> float:
        if q == 0.0:
            return (-math.inf if policy.name == "conservative_extended_real" else
                    self.kth_difference(after, before, 1))
        if q == 1.0:
            return (math.inf if policy.name == "conservative_extended_real" else
                    self.kth_difference(after, before, n))
        raw = q * n
        lower_rank = max(1, math.floor(raw))
        upper_rank = max(1, math.ceil(raw))
        lower = self.kth_difference(after, before, lower_rank)
        upper = self.kth_difference(after, before, upper_rank)
        return lower / 2.0 + upper / 2.0

    def confidence_set(self, after: int, before: int, n: int, p: float,
                       delta: float, policy: SunCSEndpointPolicy) -> Interval:
        low_q, high_q = _levels(p, delta, n)
        return (self._quantile(after, before, n, low_q, policy),
                self._quantile(after, before, n, high_q, policy))


def algorithm2_optimized(values: Sequence[float], p: float = 0.99,
                         delta: float = 0.01, *,
                         endpoint_policy: SunCSEndpointPolicy = PRIMARY_ENDPOINT_POLICY
                         ) -> dict[str, Any]:
    """Exact Algorithm 2 using persistent order statistics (O(n log n))."""
    if not values:
        raise SunCSUndefinedSourceBehavior("Algorithm 2 requires at least one score")
    data = tuple(float(x) for x in values)
    if any(not math.isfinite(x) for x in data):
        raise SunCSReferenceError("Algorithm 2 scores must be finite")
    tree = _PersistentOrderStatistics(data)
    n = len(data)
    forward = [tree.confidence_set(tree.roots[stop], 0, stop, p, delta,
                                   endpoint_policy)
               for stop in range(1, n + 1)]
    backward = [tree.confidence_set(tree.roots[n], tree.roots[start], n - start,
                                    p, delta, endpoint_policy)
                for start in range(n)]
    f_intersection = _intersection_or_empty(forward)
    b_intersection = _intersection_or_empty(backward)
    detected = (f_intersection is None or b_intersection is None or
                max(f_intersection[0], b_intersection[0]) >
                min(f_intersection[1], b_intersection[1]))
    return {"forward_sets": forward, "backward_sets": backward,
            "forward_intersection": f_intersection,
            "backward_intersection": b_intersection,
            "change_detected": detected}


def _encoded_endpoint(value: float) -> float | dict[str, str]:
    if math.isinf(value):
        return {"kind": "positive_infinity" if value > 0 else "negative_infinity"}
    return float(value)


def encode_interval(interval: Interval | None) -> list[Any] | None:
    if interval is None:
        return None
    return [_encoded_endpoint(interval[0]), _encoded_endpoint(interval[1])]


def algorithm3_reference(scores: Sequence[float], p: float = 0.99,
                         delta: float = 0.01, *,
                         endpoint_policy: SunCSEndpointPolicy = PRIMARY_ENDPOINT_POLICY
                         ) -> list[dict[str, Any]]:
    """Deliberately simple causal Algorithm 3 event trace for parity tests.

    The first event abstains cold. At a detected-change event, that event is
    also abstained, tau becomes t-1 in the paper's one-based indexing, and its
    score becomes the first observation in the new segment after prediction.
    """
    history: list[float] = []
    output: list[dict[str, Any]] = []
    segment_start = 0
    for t, raw in enumerate(scores):
        score = float(raw)
        if not math.isfinite(score):
            raise SunCSReferenceError("Algorithm 3 scores must be finite")
        before = tuple(history)
        if not before:
            row = {"t": t, "prediction": None, "reason": "ABSTAIN_COLD_START",
                   "confidence_set": None, "change_detected": False,
                   "segment_start": segment_start, "history_size": 0}
        else:
            change = algorithm2_reference(before, p, delta,
                                          endpoint_policy=endpoint_policy)["change_detected"]
            if change:
                row = {"t": t, "prediction": None,
                       "reason": "ABSTAIN_CHANGE_DETECTED",
                       "confidence_set": None, "change_detected": True,
                       "segment_start": t, "history_size": len(before)}
                history.clear()
                segment_start = t
            else:
                confidence_set = quantile_confidence_set(
                    p, delta, before, endpoint_policy=endpoint_policy)
                prediction = classify_score(score, confidence_set)
                row = {"t": t, "prediction": prediction,
                       "reason": ("SUN_CS_ANOMALY" if prediction == 1 else
                                  "SUN_CS_BENIGN" if prediction == 0 else
                                  "SUN_CS_ABSTAIN"),
                       "confidence_set": encode_interval(confidence_set),
                       "change_detected": False, "segment_start": segment_start,
                       "history_size": len(before)}
        history.append(score)
        output.append(row)
    return output


class SunCSController:
    """Causal Algorithm 3 controller; physical labels are deliberately ignored."""
    name = "B6"

    def __init__(self, p: float = 0.99, delta: float = 0.01,
                 endpoint_policy: SunCSEndpointPolicy = PRIMARY_ENDPOINT_POLICY):
        if not 0.0 < p < 1.0 or not 0.0 < delta < 1.0:
            raise ValueError("Sun-CS requires p, delta in (0,1)")
        if not isinstance(endpoint_policy, SunCSEndpointPolicy):
            raise TypeError("Endpoint policy must be versioned")
        self.p, self.delta = float(p), float(delta)
        self.endpoint_policy = endpoint_policy
        self.history: list[float] = []
        self.segment_start = 0
        self.reset_count = 0
        self.theta: float | None = None
        self._next_interval: Interval | None = None
        self._last_t = -1
        self._pending_change: bool | None = None

    def _decision(self, observation: Observation) -> tuple[int | None, str,
                                                            Interval | None,
                                                            bool]:
        if not self.history:
            return None, "ABSTAIN_COLD_START", None, False
        change = algorithm2_optimized(
            self.history, self.p, self.delta,
            endpoint_policy=self.endpoint_policy)["change_detected"]
        if change:
            return None, "ABSTAIN_CHANGE_DETECTED", None, True
        confidence_set = quantile_confidence_set(
            self.p, self.delta, self.history,
            endpoint_policy=self.endpoint_policy)
        prediction = classify_score(observation.score, confidence_set)
        reason = ("SUN_CS_ANOMALY" if prediction == 1 else
                  "SUN_CS_BENIGN" if prediction == 0 else "SUN_CS_ABSTAIN")
        return prediction, reason, confidence_set, False

    def predict(self, observation: Observation) -> Prediction:
        label, reason, confidence_set, change = self._decision(observation)
        upper = confidence_set[1] if confidence_set else None
        threshold = upper if upper is not None and math.isfinite(upper) else None
        details = {
            "method_version": SUN_CS_VERSION,
            "endpoint_policy": self.endpoint_policy.name,
            "source_target": "1[S_t > Q_Ft(p)]",
            "physical_defect_guarantee": False,
            "change_detected": change,
            "segment_start": observation.t if change else self.segment_start,
            "history_size": len(self.history),
            "confidence_set": encode_interval(confidence_set),
        }
        return Prediction(label, threshold, reason, details, available=True)

    def observe_score(self, observation: Observation) -> None:
        if observation.t != self._last_t + 1:
            raise SunCSReferenceError("Sun-CS requires contiguous causal observations")
        # Engine execution supplies the already computed past-only decision in
        # observe_feedback. Direct controller use can omit it and recompute.
        change = self._pending_change
        if change is None:
            change = bool(self.history and algorithm2_optimized(
                self.history, self.p, self.delta,
                endpoint_policy=self.endpoint_policy)["change_detected"])
        self._pending_change = None
        if self.history and change:
            self.history.clear()
            self.segment_start = observation.t
            self.reset_count += 1
        elif not self.history:
            self.segment_start = observation.t
        self.history.append(float(observation.score))
        self._next_interval = quantile_confidence_set(
            self.p, self.delta, self.history,
            endpoint_policy=self.endpoint_policy)
        self._last_t = observation.t

    def observe_feedback(self, observation: Observation, prediction: Prediction,
                         feedback: Any) -> None:
        # The method reads only its own past-only change decision. Physical
        # feedback content is intentionally ignored.
        value = prediction.details.get("change_detected")
        if value is not None:
            self._pending_change = bool(value)

    def on_memory_change(self, evidence: Any) -> None:
        return

    def finalize_step(self) -> float | None:
        if self._next_interval is None or not math.isfinite(self._next_interval[1]):
            return None
        return float(self._next_interval[1])

    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": SUN_CS_VERSION,
                "p": self.p, "delta": self.delta,
                "endpoint_policy": self.endpoint_policy.name,
                "endpoint_status": self.endpoint_policy.status,
                "history": list(self.history), "segment_start": self.segment_start,
                "reset_count": self.reset_count, "last_t": self._last_t,
                "next_interval": encode_interval(self._next_interval),
                "pending_change": self._pending_change,
                "target": "score_quantile_indicator_not_physical_defect"}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state.get("name") != self.name or state.get("version") != SUN_CS_VERSION or
                state.get("p") != self.p or state.get("delta") != self.delta or
                state.get("endpoint_policy") != self.endpoint_policy.name or
                state.get("endpoint_status") != self.endpoint_policy.status or
                state.get("target") != "score_quantile_indicator_not_physical_defect"):
            raise SunCSReferenceError("Sun-CS checkpoint identity mismatch")
        history = [float(x) for x in state["history"]]
        if any(not math.isfinite(x) for x in history):
            raise SunCSReferenceError("Nonfinite Sun-CS checkpoint history")
        self.history = history
        self.segment_start = int(state["segment_start"])
        self.reset_count = int(state["reset_count"])
        self._last_t = int(state["last_t"])
        self._next_interval = (quantile_confidence_set(
            self.p, self.delta, self.history,
            endpoint_policy=self.endpoint_policy) if self.history else None)
        if encode_interval(self._next_interval) != state.get("next_interval"):
            raise SunCSReferenceError("Sun-CS checkpoint interval mismatch")
        self._pending_change = state.get("pending_change")


def unique_offline_match(dataset_sets: Sequence[Interval],
                         online_set: Interval) -> int | None:
    """Algorithm 4 selection helper retained for reference; not registered."""
    matches = [index for index, interval in enumerate(dataset_sets)
               if max(float(interval[0]), float(online_set[0])) <= min(
                   float(interval[1]), float(online_set[1]))]
    return matches[0] if len(matches) == 1 else None


def run_class_scale_synthetic_diagnostic(n: int = 3815) -> dict[str, Any]:
    """Measure the exact causal controller on a stationary class-length fixture.

    This is a computational stress diagnostic only. The constant synthetic
    score stream intentionally avoids change resets, exposing the long-segment
    worst-case A3 work without using any image, label, or dataset outcome.
    """
    if not isinstance(n, int) or n < 2:
        raise ValueError("Class-scale diagnostic requires n >= 2")
    from .protocol import Feedback, Observation

    controller = SunCSController(p=0.99, delta=0.01,
                                 endpoint_policy=PRIMARY_ENDPOINT_POLICY)
    counts = {"SUN_CS_ANOMALY": 0, "SUN_CS_BENIGN": 0, "SUN_CS_ABSTAIN": 0,
              "ABSTAIN_COLD_START": 0, "ABSTAIN_CHANGE_DETECTED": 0}
    tracemalloc.start()
    started = time.perf_counter()
    try:
        for t in range(n):
            observation = Observation(
                t=t, observation_id=f"sun-cs-synthetic-{t:05d}", camera_id=1,
                score=0.5, memory_version=0,
                score_state_digest="synthetic-stationary-fixed-state")
            prediction = controller.predict(observation)
            if prediction.reason in counts:
                counts[prediction.reason] += 1
            controller.observe_feedback(
                observation, prediction, Feedback(t=t, queried=False, label=None))
            controller.observe_score(observation)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "schema": "calibration.sun_cs_class_scale_synthetic_diagnostic.v1",
        "status": "PASS_SYNTHETIC_COMPLETED",
        "n_events": n,
        "fixture": "constant_score_0.5_stationary_unlabeled_stream_no_change_resets",
        "prediction_counts": counts,
        "binary_coverage": (counts["SUN_CS_ANOMALY"] +
                            counts["SUN_CS_BENIGN"]) / n,
        "abstention_fraction": (counts["SUN_CS_ABSTAIN"] +
                                counts["ABSTAIN_COLD_START"] +
                                counts["ABSTAIN_CHANGE_DETECTED"]) / n,
        "reset_count": controller.reset_count,
        "segment_history_size": len(controller.history),
        "runtime_seconds_with_tracemalloc": elapsed,
        "peak_python_traced_bytes": peak,
        "memory_scope": "Python allocations only; native process RSS is not measured",
        "time_complexity": "O(T^2 log T) worst case across a length-T no-reset Algorithm 3 segment",
        "peak_auxiliary_space": "O(T log T) for the largest per-event Algorithm 2 persistent order-statistic structure",
        "stage1_or_stage2_run": False,
        "real_iad_data_or_outcomes_used": False,
    }
