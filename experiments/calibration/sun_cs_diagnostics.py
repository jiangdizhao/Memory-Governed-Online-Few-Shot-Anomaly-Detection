"""Deterministic, non-Real-IAD diagnostics for Sun-CS endpoint policies.

The generator and offline evaluator are deliberately separate from the
controller. Only the current scalar score is placed in ``Observation``; the
known synthetic quantile and score-target label are retained by this module
for post-decision diagnostic accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import math
import os
from pathlib import Path
import threading
import time
import tracemalloc
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np

from .protocol import Feedback, Observation
from .sun_cs import (
    EMPIRICAL_EXTREMA_SENSITIVITY_POLICY,
    PRIMARY_ENDPOINT_POLICY,
    SUN_CS_ENDPOINT_POLICY_SCHEMA,
    SUN_CS_VERSION,
    SunCSEndpointPolicy,
    SunCSController,
    confidence_radius,
)


P = 0.99
DELTA = 0.01
SYNTHETIC_DIAGNOSTIC_SCHEMA = "calibration.sun_cs_continuous_diagnostics.v1"
SCENARIO_SEEDS = {
    "stationary_uniform": 2026102901,
    "stationary_lognormal": 2026102902,
    "piecewise_uniform_shift": 2026102903,
    "class_scale_uniform": 2026102904,
}


@dataclass(frozen=True)
class SyntheticStream:
    name: str
    values: tuple[float, ...]
    stationary: bool
    seed: int
    shift_event: int | None

    def true_quantile_at(self, t: int) -> float:
        if self.name == "stationary_uniform" or self.name == "class_scale_uniform":
            return P
        if self.name == "stationary_lognormal":
            return math.exp(NormalDist().inv_cdf(P))
        if self.name == "piecewise_uniform_shift":
            return 2.0 + P if self.shift_event is not None and t >= self.shift_event else P
        raise ValueError(f"Unknown synthetic scenario: {self.name}")


def make_stream(name: str, n: int, *, seed: int | None = None) -> SyntheticStream:
    if n < 2:
        raise ValueError("Synthetic diagnostic requires at least two observations")
    if seed is None:
        try:
            seed = SCENARIO_SEEDS[name]
        except KeyError as exc:
            raise ValueError(f"Unknown synthetic scenario: {name}") from exc
    rng = np.random.default_rng(seed)
    if name in ("stationary_uniform", "class_scale_uniform"):
        values = rng.uniform(0.0, 1.0, size=n)
        stationary, shift_event = True, None
    elif name == "stationary_lognormal":
        values = rng.lognormal(mean=0.0, sigma=1.0, size=n)
        stationary, shift_event = True, None
    elif name == "piecewise_uniform_shift":
        shift_event = n // 2
        values = np.concatenate((
            rng.uniform(0.0, 1.0, size=shift_event),
            rng.uniform(2.0, 3.0, size=n - shift_event),
        ))
        stationary = False
    else:
        raise ValueError(f"Unknown synthetic scenario: {name}")
    finite_values = tuple(float(x) for x in values)
    if any(not math.isfinite(x) for x in finite_values):
        raise RuntimeError("Synthetic generator produced a nonfinite score")
    return SyntheticStream(name, finite_values, stationary, int(seed), shift_event)


class _PeakRSSSampler:
    """Sample process RSS on the untraced timing run when psutil is available."""

    def __init__(self) -> None:
        try:
            import psutil  # existing environment dependency; no installation
            self._process = psutil.Process(os.getpid())
            self.peak = int(self._process.memory_info().rss)
            self.available = True
        except Exception:
            self._process = None
            self.peak = None
            self.available = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_PeakRSSSampler":
        if self.available:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(int(self.peak), int(self._process.memory_info().rss))
            except Exception:
                pass
            self._stop.wait(0.02)

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def _decode_endpoint(value: Any) -> float:
    if isinstance(value, dict):
        kind = value.get("kind")
        if kind == "negative_infinity":
            return -math.inf
        if kind == "positive_infinity":
            return math.inf
        raise ValueError(f"Unknown structured Sun-CS endpoint: {value}")
    return float(value)


def _run_once(stream: SyntheticStream,
              policy: SunCSEndpointPolicy) -> dict[str, Any]:
    controller = SunCSController(P, DELTA, endpoint_policy=policy)
    reasons = {
        "SUN_CS_ANOMALY": 0,
        "SUN_CS_BENIGN": 0,
        "SUN_CS_ABSTAIN": 0,
        "ABSTAIN_COLD_START": 0,
        "ABSTAIN_CHANGE_DETECTED": 0,
    }
    target_counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    target_positive_count = 0
    target_positive_covered = 0
    target_negative_covered = 0
    interval_available = 0
    interval_contains_true_quantile = 0
    first_binary: int | None = None
    first_change: int | None = None
    first_change_after_shift: int | None = None

    for t, score in enumerate(stream.values):
        obs = Observation(
            t=t, observation_id=f"synthetic-{stream.name}-{t:05d}",
            camera_id=(t % 5) + 1, score=score, memory_version=0,
            score_state_digest=f"synthetic-fixed-score-state-{stream.name}",
        )
        prediction = controller.predict(obs)
        reason = prediction.reason
        reasons[reason] += 1
        if reason == "ABSTAIN_CHANGE_DETECTED":
            if first_change is None:
                first_change = t
            if (stream.shift_event is not None and t >= stream.shift_event and
                    first_change_after_shift is None):
                first_change_after_shift = t
        target_positive = score > stream.true_quantile_at(t)
        target_positive_count += int(target_positive)
        if prediction.label is not None:
            if first_binary is None:
                first_binary = t
            if target_positive:
                target_positive_covered += 1
            else:
                target_negative_covered += 1
            if target_positive and prediction.label == 1:
                target_counts["TP"] += 1
            elif target_positive and prediction.label == 0:
                target_counts["FN"] += 1
            elif not target_positive and prediction.label == 1:
                target_counts["FP"] += 1
            else:
                target_counts["TN"] += 1

        encoded_interval = prediction.details.get("confidence_set")
        if encoded_interval is not None:
            lower, upper = map(_decode_endpoint, encoded_interval)
            interval_available += 1
            q_true = stream.true_quantile_at(t)
            interval_contains_true_quantile += int(lower <= q_true <= upper)

        # The online controller receives no synthetic target label or future
        # data. Its feedback handler is passed only the current unqueried token.
        controller.observe_feedback(
            obs, prediction, Feedback(t=t, queried=False, label=None))
        controller.observe_score(obs)

    n = len(stream.values)
    binary = reasons["SUN_CS_ANOMALY"] + reasons["SUN_CS_BENIGN"]
    target_negative_count = n - target_positive_count
    positive_binary = target_counts["TP"] + target_counts["FN"]
    negative_binary = target_counts["TN"] + target_counts["FP"]
    confusion_n = sum(target_counts.values())
    return {
        "n_events": n,
        "prediction_counts": reasons,
        "first_binary_prediction_position_zero_based": first_binary,
        "binary_coverage": binary / n,
        "abstention_fraction": (n - binary) / n,
        "reset_count": controller.reset_count,
        "first_change_detection_position_zero_based": first_change,
        "first_change_detection_after_shift_position_zero_based": first_change_after_shift,
        "false_reset_count_stationary": controller.reset_count if stream.stationary else None,
        "shift_event_zero_based": stream.shift_event,
        "post_shift_change_detection_latency": (
            None if stream.shift_event is None or first_change_after_shift is None
            else first_change_after_shift - stream.shift_event),
        "post_shift_change_detected": (
            None if stream.shift_event is None else
            bool(first_change_after_shift is not None)),
        "source_target": "1[score > true synthetic Q_Ft(0.99)]",
        "offline_target_positive_count": target_positive_count,
        "source_target_binary_confusion_on_covered_events": target_counts,
        "source_target_binary_coverage": binary / n,
        "source_target_positive_event_coverage": (
            None if target_positive_count == 0 else
            target_positive_covered / target_positive_count),
        "source_target_negative_event_coverage": (
            None if target_negative_count == 0 else
            target_negative_covered / target_negative_count),
        "source_target_positive_event_abstention_rate": (
            None if target_positive_count == 0 else
            1.0 - target_positive_covered / target_positive_count),
        "source_target_negative_event_abstention_rate": (
            None if target_negative_count == 0 else
            1.0 - target_negative_covered / target_negative_count),
        "source_target_selective_accuracy": (
            None if confusion_n == 0 else
            (target_counts["TP"] + target_counts["TN"]) / confusion_n),
        "source_target_selective_fpr": (
            None if negative_binary == 0 else target_counts["FP"] / negative_binary),
        "source_target_selective_fnr": (
            None if positive_binary == 0 else target_counts["FN"] / positive_binary),
        "offline_true_quantile_interval_evaluations": interval_available,
        "offline_true_quantile_interval_coverage": (
            None if interval_available == 0 else
            interval_contains_true_quantile / interval_available),
        "offline_true_quantile_interval_contains_count": interval_contains_true_quantile,
        "score_history_final": len(controller.history),
    }


def run_policy_diagnostic(stream: SyntheticStream,
                          policy: SunCSEndpointPolicy, *,
                          measure_python_peak: bool = True) -> dict[str, Any]:
    """Run once untraced for latency/RSS, then separately for Python peak RAM."""
    with _PeakRSSSampler() as rss:
        started = time.perf_counter()
        result = _run_once(stream, policy)
        runtime = time.perf_counter() - started
    result.update({
        "scenario": stream.name,
        "stationarity": "STATIONARY" if stream.stationary else "PIECEWISE_STATIONARY",
        "seed": stream.seed,
        "endpoint_policy": policy.name,
        "endpoint_policy_status": policy.status,
        "method_version": policy.method_version,
        "endpoint_policy_schema": SUN_CS_ENDPOINT_POLICY_SCHEMA,
        "source_fingerprints": source_fingerprints(),
        "runtime_seconds_untraced": runtime,
        "amortized_runtime_seconds_per_observation": runtime / len(stream.values),
        "process_rss_peak_bytes_sampled": rss.peak,
        "process_rss_sampling_status": "SAMPLED_PSUTIL" if rss.available else "UNAVAILABLE",
    })
    if measure_python_peak:
        tracemalloc.start()
        try:
            _run_once(stream, policy)
            _, python_peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        result["peak_python_bytes_tracemalloc_separate_run"] = int(python_peak)
        result["python_peak_measurement_status"] = "MEASURED_SEPARATE_TRACED_RUN"
    else:
        result["peak_python_bytes_tracemalloc_separate_run"] = None
        result["python_peak_measurement_status"] = "SKIPPED_BY_DIAGNOSTIC_PROFILE"
    return result


def endpoint_identifiability_table(
        ns: Iterable[int] = (64, 256, 1050, 3815)) -> list[dict[str, Any]]:
    """Exact confidence levels and interval endpoints for sample [1,...,n]."""
    rows = []
    for n in ns:
        radius = confidence_radius(n, DELTA)
        low_q = max(P - 2.0 * radius, 0.0)
        high_q = min(P + 2.0 * radius, 1.0)
        # For x_i=i, the midpoint order statistic is calculable exactly.
        raw_rank = low_q * n
        low_floor = max(1, math.floor(raw_rank))
        low_ceil = max(1, math.ceil(raw_rank))
        lower = low_floor / 2.0 + low_ceil / 2.0
        if high_q == 1.0:
            empirical_upper: int = n
            conservative_upper: str = "+infinity"
            finite_score_anomaly_possible = False
        else:
            upper_rank = high_q * n
            hi_floor, hi_ceil = math.floor(upper_rank), math.ceil(upper_rank)
            empirical_upper = hi_floor / 2.0 + hi_ceil / 2.0
            conservative_upper = empirical_upper
            finite_score_anomaly_possible = True
        rows.append({
            "n": n,
            "radius": radius,
            "lower_requested_level": low_q,
            "upper_requested_level": high_q,
            "lower_order_statistic_midpoint_for_sample_1_to_n": lower,
            "empirical_extrema_interval_for_sample_1_to_n": [lower, empirical_upper],
            "conservative_extended_real_interval_for_sample_1_to_n": [
                lower, conservative_upper],
            "finite_score_anomaly_possible_under_primary_endpoint": finite_score_anomaly_possible,
        })
    return rows


def first_unclipped_upper_level_n() -> dict[str, Any]:
    """Find the first prefix length where p+2u_n(delta) is strictly below 1."""
    low, high = 1, 1
    while P + 2.0 * confidence_radius(high, DELTA) >= 1.0:
        high *= 2
    low = high // 2
    while low + 1 < high:
        mid = (low + high) // 2
        if P + 2.0 * confidence_radius(mid, DELTA) < 1.0:
            high = mid
        else:
            low = mid
    return {
        "first_prefix_n_with_unclipped_upper_level": high,
        "upper_level_at_n_minus_1": min(
            P + 2.0 * confidence_radius(high - 1, DELTA), 1.0),
        "upper_level_at_n": P + 2.0 * confidence_radius(high, DELTA),
        "interpretation": (
            "A conservative primary anomaly decision is impossible before this "
            "prefix length because the upper confidence endpoint is +infinity."),
    }


def clopper_pearson_zero_error_minimum(delta: float = 0.05,
                                       target_upper: float = 0.20) -> dict[str, Any]:
    if not 0.0 < delta < 1.0 or not 0.0 < target_upper < 1.0:
        raise ValueError("delta and target_upper must be in (0,1)")
    n_min = math.ceil(math.log(delta) / math.log1p(-target_upper))
    bound = lambda n: 1.0 - delta ** (1.0 / n)
    return {
        "formula": "U(0,n)=1-delta^(1/n)",
        "delta": delta,
        "target_upper": target_upper,
        "minimum_zero_error_n": n_min,
        "bound_at_n_minus_1": bound(n_min - 1),
        "bound_at_n": bound(n_min),
        "minimum_queried_defects_for_two_sided_activation": 8,
        "eight_defects_satisfy_descriptive_target": bound(8) <= target_upper,
        "status": "DESCRIPTIVE_FIXED_TIME_SAMPLE_SIZE_ONLY",
        "online_guarantee_claimed": False,
    }


def _binomial_tail_at_least(n: int, p: float, k: int) -> float:
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, x) * p ** x * (1.0 - p) ** (n - x)
               for x in range(k, n + 1))


def b7_query_budget_feasibility(
        per_class_query_budgets: dict[str, int],
        defect_fractions: Iterable[float] = (0.05, 0.10, 0.20, 0.30),
        minimum_evidence: int = 8,
        zero_error_precision_n: int = 14) -> dict[str, Any]:
    """Hypothetical Bernoulli calculations over fixed metadata query budgets."""
    if not per_class_query_budgets or any(n < 0 for n in per_class_query_budgets.values()):
        raise ValueError("Per-class fixed query budgets must be nonnegative")
    rows = []
    for p in defect_fractions:
        if not 0.0 < p < 1.0:
            raise ValueError("Hypothetical defect fractions must be in (0,1)")
        details = {}
        p8, p14 = [], []
        for class_name, n_queries in per_class_query_budgets.items():
            reach_8 = _binomial_tail_at_least(n_queries, p, minimum_evidence)
            reach_14 = _binomial_tail_at_least(n_queries, p, zero_error_precision_n)
            p8.append(reach_8)
            p14.append(reach_14)
            details[class_name] = {
                "query_count": n_queries,
                "probability_queried_defects_at_least_8": reach_8,
                "probability_queried_defects_at_least_14": reach_14,
                "probability_below_8_and_remain_normal_only_hold": 1.0 - reach_8,
            }
        rows.append({
            "hypothetical_queried_defect_fraction": p,
            "independent_bernoulli_assumption": True,
            "classes": details,
            "expected_development_classes_reaching_8": sum(p8),
            "expected_development_classes_reaching_14": sum(p14),
            "probability_at_least_one_class_reaches_8": 1.0 - math.prod(1.0 - x for x in p8),
            "probability_all_classes_reach_8": math.prod(p8),
            "probability_at_least_one_class_reaches_14": 1.0 - math.prod(1.0 - x for x in p14),
            "probability_all_classes_reach_14": math.prod(p14),
        })
    return {
        "query_budget_rule": "half_up(0.03*N_images), one label-blind query per official sample group",
        "per_class_query_budgets": dict(per_class_query_budgets),
        "total_query_budget": sum(per_class_query_budgets.values()),
        "minimum_defects_for_nominal_two_sided_evidence": minimum_evidence,
        "zero_error_defects_for_descriptive_fnr_upper_le_0_20": zero_error_precision_n,
        "assumption": (
            "Within each hypothetical scenario, queried labels are independent "
            "Bernoulli(p) draws with the same p for each development class. "
            "The fractions are not Real-IAD prevalence estimates."),
        "results": rows,
        "stage1_or_stage2_executed": False,
        "real_iad_labels_or_prevalence_inspected": False,
    }


def query_budgets_from_metadata_preflight(path: str | Path) -> dict[str, Any]:
    """Read only preflight identity/image/query-count fields, never label rows."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (payload.get("status") != "PASS" or
            payload.get("confirmation_classes_opened") is not False or
            payload.get("stage1_or_stage2_run") is not False):
        raise ValueError("Metadata preflight is not an eligible Stage 0 source")
    summaries: dict[str, Any] = {}
    for row in payload.get("classes", []):
        name = row.get("class")
        stream = row.get("test_stream", {})
        qa = row.get("Q-A-TS", {})
        if (name not in {"audiojack", "regulator", "switch", "rolled_strip_base"} or
                row.get("status") != "PASS" or row.get("metadata_only") is not True or
                row.get("image_or_mask_opened") is not False or
                row.get("query_priorities_used_test_label") is not False):
            raise ValueError("Preflight class row is not a privacy-safe development summary")
        n_images, query_count = int(stream["images"]), int(qa["query_count"])
        budget = math.floor(0.03 * n_images + 0.5)
        if query_count != budget:
            raise ValueError(f"Recorded Q-A-TS budget mismatch for {name}")
        summaries[name] = {
            "images": n_images,
            "official_sample_groups": int(stream["official_sample_groups"]),
            "query_count": query_count,
            "image_query_fraction": float(qa["image_query_fraction"]),
            "metadata_only": True,
        }
    expected_names = {"audiojack", "regulator", "switch", "rolled_strip_base"}
    if set(summaries) != expected_names:
        raise ValueError("Preflight must cover exactly the four development classes")
    return {
        "source_artifact": source.as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "classes": summaries,
        "per_class_query_budgets": {
            name: row["query_count"] for name, row in summaries.items()},
        "total_images": sum(row["images"] for row in summaries.values()),
        "total_official_sample_groups": sum(
            row["official_sample_groups"] for row in summaries.values()),
        "total_queries": sum(row["query_count"] for row in summaries.values()),
        "total_image_query_fraction": (
            sum(row["query_count"] for row in summaries.values()) /
            sum(row["images"] for row in summaries.values())),
        "total_official_sample_group_query_fraction": (
            sum(row["query_count"] for row in summaries.values()) /
            sum(row["official_sample_groups"] for row in summaries.values())),
        "labels_or_hidden_prevalence_read": False,
        "confirmation_classes_opened": False,
    }


def source_fingerprints() -> dict[str, str]:
    module_paths = {
        "sun_cs_source": Path(__file__).with_name("sun_cs.py"),
        "diagnostic_source": Path(__file__),
    }
    return {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in module_paths.items()
    }


def build_diagnostic_suite(*, n: int = 512,
                           include_class_scale: bool = False,
                           class_scale_memory_trace: bool = True) -> dict[str, Any]:
    if n < 128:
        raise ValueError("Use at least 128 events to expose initialization behavior")
    policies = (PRIMARY_ENDPOINT_POLICY,
                EMPIRICAL_EXTREMA_SENSITIVITY_POLICY)
    scenarios = ("stationary_uniform", "stationary_lognormal",
                 "piecewise_uniform_shift")
    results = []
    for scenario_name in scenarios:
        stream = make_stream(scenario_name, n)
        for policy in policies:
            results.append(run_policy_diagnostic(stream, policy))
    if include_class_scale:
        stream = make_stream("class_scale_uniform", 3815)
        results.append(run_policy_diagnostic(
            stream, PRIMARY_ENDPOINT_POLICY,
            measure_python_peak=class_scale_memory_trace))
    return {
        "schema": SYNTHETIC_DIAGNOSTIC_SCHEMA,
        "status": "PASS_SYNTHETIC_ONLY",
        "method_version": SUN_CS_VERSION,
        "endpoint_policy_schema": SUN_CS_ENDPOINT_POLICY_SCHEMA,
        "p": P,
        "delta": DELTA,
        "finite_stream_identifiability": endpoint_identifiability_table(),
        "upper_endpoint_identifiability_horizon": first_unclipped_upper_level_n(),
        "b7_zero_error_descriptive_feasibility": clopper_pearson_zero_error_minimum(),
        "source_fingerprints": source_fingerprints(),
        "real_iad_data_or_outcomes_used": False,
        "stage1_or_stage2_executed": False,
        "controller_information_boundary": (
            "Only current causal score; synthetic true quantile and source-target "
            "labels are evaluated offline after each prediction."),
        "runs": results,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {target}")
    target.write_text(data, encoding="utf-8")
