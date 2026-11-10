"""Frozen Gate 0 per-category query controllers for the final benchmark."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import math
import numbers
import re
from typing import Sequence

import numpy as np


class QueryProtocolError(RuntimeError):
    """A query proposal or accounting transition violates the frozen contract."""


PRINCIPAL_ARMS = ("F-B", "K-B", "F-M", "K-M")
BEA_ARMS = frozenset(("F-B", "K-B"))
MARGIN_ARMS = frozenset(("F-M", "K-M"))
UTILITY_WINDOW = 256
PRICE_WINDOW = 256
PINNED_NUMPY_VERSION = "1.24.1"


def _uniform(domain: str, *, dataset_id: str, category: str, arm_id: str,
             support_seed: int, stream_seed: int, event_id_sha256: str) -> float:
    if not re.fullmatch(r"[0-9a-f]{64}", event_id_sha256):
        raise QueryProtocolError("event identity must be a lowercase SHA-256 digest")
    if domain not in {"cold-start", "tie"}:
        raise QueryProtocolError("Margin_rho hash domain is not registered")
    payload = ("Gate0-Margin-rho-v1\0" + dataset_id + "\0" + category + "\0" +
               arm_id + "\0" + str(int(support_seed)) + "\0" + str(int(stream_seed)) +
               "\0" + event_id_sha256 + "\0" + domain).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / float(1 << 64)


def _numpy_linear_quantile(values: Sequence[float], q: float) -> float:
    if np.__version__ != PINNED_NUMPY_VERSION:
        raise QueryProtocolError(
            f"Margin_rho requires pinned NumPy {PINNED_NUMPY_VERSION}; found {np.__version__}")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise QueryProtocolError("Margin_rho quantile history must be finite and nonempty")
    # Gate 0 pins the equivalent Hyndman-Fan type 7 / NumPy linear rule.
    return float(np.quantile(array, q, method="linear"))


@dataclass(frozen=True)
class Proposal:
    arm_id: str
    proposed: bool
    utility: float | None
    threshold: float | None
    objective: float | None
    price: float | None
    reason: str


@dataclass
class MarginRho:
    dataset_id: str
    category: str
    arm_id: str
    rho: float
    total_events: int
    support_seed: int
    stream_seed: int
    history: deque[float] = field(default_factory=lambda: deque(maxlen=UTILITY_WINDOW))
    cold_start_events: int = field(init=False)
    quantile_count: int = 0
    tie_count: int = 0

    def __post_init__(self) -> None:
        if np.__version__ != PINNED_NUMPY_VERSION:
            raise QueryProtocolError(
                f"Margin_rho requires pinned NumPy {PINNED_NUMPY_VERSION}; found {np.__version__}")
        if self.arm_id not in MARGIN_ARMS:
            raise QueryProtocolError("Margin_rho is registered only for F-M and K-M")
        if self.dataset_id not in {"realiad", "visa_1cls"}:
            raise QueryProtocolError("dataset ID is not registered")
        expected = {"realiad": 0.03, "visa_1cls": 0.10}[self.dataset_id]
        if (float(self.rho) != expected or isinstance(self.total_events, bool) or
                not isinstance(self.total_events, numbers.Integral) or int(self.total_events) < 1):
            raise QueryProtocolError("dataset target rate or stream length differs from Gate 0")
        if (isinstance(self.support_seed, bool) or isinstance(self.stream_seed, bool) or
                not isinstance(self.support_seed, numbers.Integral) or
                not isinstance(self.stream_seed, numbers.Integral) or
                min(int(self.support_seed), int(self.stream_seed)) < 0):
            raise QueryProtocolError("seed fields must be unsigned decimals")
        self.history = deque((float(x) for x in self.history), maxlen=UTILITY_WINDOW)
        self.cold_start_events = min(64, max(16, math.floor(0.2 * self.total_events)))

    def propose_group_view(self, *, position: int, event_id_sha256: str,
                           psi: float, group_starts_cold: bool) -> Proposal:
        value = float(psi)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise QueryProtocolError("Margin_rho standardized-margin utility is outside [0,1]")
        # The entire group containing L remains cold-start; only the next group
        # may consult the past-only empirical quantile.
        if group_starts_cold:
            u = _uniform("cold-start", dataset_id=self.dataset_id,
                         category=self.category, arm_id=self.arm_id,
                         support_seed=self.support_seed, stream_seed=self.stream_seed,
                         event_id_sha256=event_id_sha256)
            return Proposal(self.arm_id, u < self.rho, value, None, value, None,
                            "identity_hashed_cold_start")
        if len(self.history) == 0:
            raise QueryProtocolError("quantile mode cannot start without past utilities")
        prior = np.asarray(self.history, dtype=np.float64)
        q = _numpy_linear_quantile(prior, 1.0 - self.rho)
        n_gt = int(np.count_nonzero(prior > q))
        n_eq = int(np.count_nonzero(prior == q))
        self.quantile_count += 1
        if value > q:
            proposed, reason = True, "past_quantile_strictly_exceeded"
        elif value < q:
            proposed, reason = False, "past_quantile_not_reached"
        else:
            self.tie_count += 1
            p_tie = min(1.0, max(0.0, (self.rho * len(prior) - n_gt) / n_eq)) if n_eq else 0.0
            u = _uniform("tie", dataset_id=self.dataset_id,
                         category=self.category, arm_id=self.arm_id,
                         support_seed=self.support_seed, stream_seed=self.stream_seed,
                         event_id_sha256=event_id_sha256)
            proposed, reason = u < p_tie, "identity_hashed_quantile_tie"
        return Proposal(self.arm_id, proposed, value, q, value, None, reason)

    def observe_group(self, utilities: Sequence[float]) -> None:
        for item in utilities:
            value = float(item)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise QueryProtocolError("Margin_rho observed utility is outside [0,1]")
            self.history.append(value)


@dataclass
class BeaPrice:
    arm_id: str
    rho: float
    price: float = 0.8
    flags: deque[int] = field(default_factory=lambda: deque(maxlen=PRICE_WINDOW))
    proposals: int = 0
    executed: int = 0
    group_suppressed: int = 0
    cap_suppressed: int = 0

    def __post_init__(self) -> None:
        if self.arm_id not in BEA_ARMS or not 0.0 < float(self.rho) < 1.0:
            raise QueryProtocolError("BEA price controller registration is invalid")
        if float(self.price) != 0.8:
            raise QueryProtocolError("BEA initial price must be 0.8")
        self.flags = deque((int(x) for x in self.flags), maxlen=PRICE_WINDOW)
        if any(x not in (0, 1) for x in self.flags):
            raise QueryProtocolError("BEA history must contain final executed action flags")

    def propose(self, objective: float, *, group_price: float | None = None) -> Proposal:
        value = float(objective)
        cutoff = float(self.price if group_price is None else group_price)
        if not math.isfinite(value) or not math.isfinite(cutoff):
            raise QueryProtocolError("BEA objective and price must be finite")
        return Proposal(self.arm_id, value > cutoff, None, cutoff, value, cutoff,
                        "objective_above_group_price" if value > cutoff else "objective_at_or_below_group_price")

    def observe_final_actions(self, actions_in_view_order: Sequence[bool]) -> list[float]:
        prices: list[float] = []
        for action in actions_in_view_order:
            if not isinstance(action, bool):
                raise QueryProtocolError("final action must be explicit boolean")
            self.flags.append(int(action))
            rate = sum(self.flags) / len(self.flags)
            self.price = min(3.0, max(0.05, self.price + 0.02 * (rate - self.rho)))
            self.executed += int(action)
            prices.append(self.price)
        return prices


@dataclass
class CategoryCap:
    total_events: int
    rho: float
    executed: int = 0
    proposals: int = 0
    group_suppressed: int = 0
    cap_suppressed: int = 0

    def __post_init__(self) -> None:
        if (isinstance(self.total_events, bool) or not isinstance(self.total_events, numbers.Integral) or
                int(self.total_events) <= 0 or not 0.0 < float(self.rho) < 1.0):
            raise QueryProtocolError("category cap inputs are invalid")
        self.limit = math.floor(self.rho * self.total_events)

    def remaining(self) -> int:
        return self.limit - self.executed
