"""Auditable Sun-CS paper-equation primitives; not a registered controller.

The source paper leaves empty-prefix and endpoint empirical-quantile behavior
unresolved.  This module therefore implements only operations whose input is
defined by the paper and raises on those unresolved cases.  It must not be
presented as a B6 controller or used to make predictions.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


class SunCSReferenceError(ValueError):
    """A reference calculation is invalid or needs an unresolved convention."""


class SunCSUndefinedSourceBehavior(SunCSReferenceError):
    """The paper does not define the empty-prefix or endpoint operation."""


Interval = tuple[float, float]


def confidence_radius(t: int, delta: float) -> float:
    """Equation (2), with the paper's stated range ``u_t in [0,1]``."""
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
    return min(1.0, max(0.0, raw))


def empirical_quantile(q: float, values: Iterable[float]) -> float:
    """Paper's midpoint empirical quantile for a defined interior rank.

    The paper gives midpoint order statistics at ``floor(q*n)`` and
    ``ceil(q*n)`` (one-based).  It does not define ranks 0 or n+1, so those
    cases fail closed rather than silently selecting sample extrema.
    """
    q = float(q)
    ordered = sorted(float(value) for value in values)
    if not math.isfinite(q) or not 0.0 < q < 1.0:
        raise SunCSUndefinedSourceBehavior(
            "Paper endpoint empirical quantile convention is unspecified")
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise SunCSUndefinedSourceBehavior(
            "Paper does not define an empirical quantile for an empty/nonfinite sample")
    rank = q * len(ordered)
    lower_rank, upper_rank = math.floor(rank), math.ceil(rank)
    if lower_rank < 1 or upper_rank > len(ordered):
        raise SunCSUndefinedSourceBehavior(
            "Requested order-statistic endpoint is undefined in the paper")
    return (ordered[lower_rank - 1] + ordered[upper_rank - 1]) / 2.0


def quantile_confidence_set(p: float, delta: float,
                            values: Sequence[float]) -> Interval:
    """Equation (1), only when both clipped empirical levels are defined."""
    if not 0.0 < float(p) < 1.0:
        raise SunCSReferenceError("p must be in (0,1)")
    if not values:
        raise SunCSUndefinedSourceBehavior(
            "Sun-CS has no source-defined confidence set for an empty prefix")
    radius = confidence_radius(len(values), delta)
    low_q = max(float(p) - 2.0 * radius, 0.0)
    high_q = min(float(p) + 2.0 * radius, 1.0)
    if low_q <= 0.0 or high_q >= 1.0:
        raise SunCSUndefinedSourceBehavior(
            "Equation (1) clips to an endpoint with no source-defined empirical quantile")
    low = empirical_quantile(low_q, values)
    high = empirical_quantile(high_q, values)
    if low > high:
        raise SunCSReferenceError("Empirical confidence-set endpoints are reversed")
    return low, high


def classify_score(score: float, confidence_set: Interval) -> int | None:
    """Algorithm 1: anomaly=1, benign=0, abstain=None."""
    value = float(score)
    lower, upper = (float(confidence_set[0]), float(confidence_set[1]))
    if (math.isnan(value) or math.isnan(lower) or math.isnan(upper) or
            lower > upper):
        raise SunCSReferenceError("Invalid score or confidence set")
    if value > upper:
        return 1
    if lower <= value <= upper:
        return None
    return 0


def intersect_intervals(intervals: Sequence[Interval]) -> Interval | None:
    """Set intersection for the intervals supplied to Algorithms 2 or 4."""
    if not intervals:
        raise SunCSUndefinedSourceBehavior(
            "The paper's empty prefix/suffix intersection is undefined")
    lower = max(float(interval[0]) for interval in intervals)
    upper = min(float(interval[1]) for interval in intervals)
    if any(math.isnan(value) for interval in intervals for value in interval):
        raise SunCSReferenceError("NaN interval endpoint")
    if lower > upper:
        return None
    return lower, upper


def change_detected(forward_prefix_sets: Sequence[Interval],
                    backward_suffix_sets: Sequence[Interval]) -> bool:
    """Algorithm 2's disjoint-intersection test on defined confidence sets."""
    forward = intersect_intervals(forward_prefix_sets)
    backward = intersect_intervals(backward_suffix_sets)
    if forward is None or backward is None:
        return True
    return max(forward[0], backward[0]) > min(forward[1], backward[1])


def unique_offline_match(dataset_sets: Sequence[Interval],
                         online_set: Interval) -> int | None:
    """Algorithm 4 selection: return the sole intersecting dataset index."""
    matches = []
    for index, interval in enumerate(dataset_sets):
        if max(float(interval[0]), float(online_set[0])) <= min(
                float(interval[1]), float(online_set[1])):
            matches.append(index)
    return matches[0] if len(matches) == 1 else None
