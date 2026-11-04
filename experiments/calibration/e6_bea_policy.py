"""Pure E6-BEA-v2 utilities and online query policies.

This module has no dataset, model, CUDA, or experiment-runner side effects.  Its
policy registry is immutable so the registered seven arms cannot drift at run
time.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence


ARM_IDS = ("Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6")
TARGET_QUERY_RATE = 0.03
INITIAL_QUERY_COST = 0.80
QUERY_COST_LEARNING_RATE = 0.02
QUERY_COST_WINDOW = 256
QUERY_COST_MIN = 0.05
QUERY_COST_MAX = 3.00
UTILITY_HISTORY_WINDOW = 256
UTILITY_COLD_START_EVENTS = 64
QUERY_PERIOD = 33
NOVELTY_CLIP_LOW = 0.05
NOVELTY_CLIP_HIGH = 0.95
CORRECTION_SCALE = 0.05
SIGMA_FLOOR = 0.02
RANDOM_GROUP_VIEW_COUNT = 5
P_RANDOM_FIVE_VIEW = 1.0 - (1.0 - RANDOM_GROUP_VIEW_COUNT * TARGET_QUERY_RATE) ** (
    1.0 / RANDOM_GROUP_VIEW_COUNT
)


class E6PolicyError(ValueError):
    """Raised when policy input or state violates the frozen E6 contract."""


@dataclass(frozen=True)
class ArmPolicy:
    arm_id: str
    kind: str
    objective_terms: tuple[str, ...] = ()
    adaptive_price: bool = False


ARM_POLICIES: Mapping[str, ArmPolicy] = MappingProxyType({
    "Q0": ArmPolicy("Q0", "random"),
    "Q1": ArmPolicy("Q1", "periodic"),
    "Q2": ArmPolicy("Q2", "standardized_margin"),
    "Q3": ArmPolicy("Q3", "novelty"),
    "Q4": ArmPolicy("Q4", "bea", ("B",), True),
    "Q5": ArmPolicy("Q5", "bea", ("B", "N", "D"), True),
    "Q6": ArmPolicy("Q6", "bea", ("B", "N", "C", "D"), True),
})


@dataclass(frozen=True)
class BEAComponents:
    score: float
    threshold: float
    sigma: float
    margin_z: float
    p_defect: float
    pred_defect: bool
    entropy: float
    psi: float
    novelty_percentile: float
    novelty: float
    correction_top_mean: float
    correction_top_max: float
    correction_top_min: float
    correction_bank_size: int
    corr_def: float
    corr_norm: float
    corr_mag: float
    correction_evidence: float
    local_saliency: float
    saliency_multiplier: float
    gate_far: float
    gate_near: float
    weak_correction: float
    R: float
    H: float
    U_N: float
    U_C: float
    U_D: float
    B: float
    N: float
    C: float
    D: float

    def as_dict(self) -> dict[str, float | int | bool]:
        values = dict(self.__dict__)
        values["pred_defect"] = bool(self.pred_defect)
        return values

    def objective(self, arm_id: str) -> float:
        try:
            terms = ARM_POLICIES[arm_id].objective_terms
        except KeyError as exc:
            raise E6PolicyError(f"Unknown E6 arm: {arm_id}") from exc
        values = {"B": self.B, "N": self.N, "C": self.C, "D": self.D}
        return float(sum(values[name] for name in terms))


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise E6PolicyError(f"{name} must be finite")
    return result


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _entropy_bern_v22(probability: float) -> float:
    """Mirror v22 ``_entropy_bern`` including its 1e-8 clipping."""
    p = _clip(probability, 0.0, 1.0)
    p = min(max(p, 1e-8), 1.0 - 1e-8)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def normalize_novelty_percentile(percentile: float) -> float:
    """Apply the original NormalCoverageMemory [0.05, 0.95] normalization."""
    value = _finite(percentile, "novelty percentile")
    return _clip((value - NOVELTY_CLIP_LOW) /
                 (NOVELTY_CLIP_HIGH - NOVELTY_CLIP_LOW), 0.0, 1.0)


def local_saliency_from_native_stats(topk_dN_max: float,
                                     topk_dN_mean: float) -> float:
    """Reproduce v22 ``compute_local_saliency_simple`` for its fused layer."""
    dmax = _finite(topk_dN_max, "Top-32 normal-distance maximum")
    dmean = _finite(topk_dN_mean, "Top-32 normal-distance mean")
    denominator = max(abs(dmax), abs(dmean), 1e-6)
    return _clip(max(0.0, (dmax - dmean) / denominator), 0.0, 1.0)


def compute_bea_components(
    *,
    score: float,
    threshold: float,
    c64_scores: Sequence[float],
    novelty_percentile: float,
    correction_top32: Sequence[float],
    correction_bank_size: int,
    topk_dN_max: float,
    topk_dN_mean: float,
) -> BEAComponents:
    """Compute the exact raw and weighted E6-BEA-v2 components.

    ``correction_top32`` must be the native C1 pre-event residual vector in the
    same Top-32 order used by the native corrected map.  An empty bank is
    represented by its actual all-zero residual vector, never by a fallback.
    """
    import numpy as np

    s = _finite(score, "GC score")
    theta = _finite(threshold, "current GC-B3 threshold")
    c64 = np.asarray(c64_scores, dtype=np.float64)
    corrections = np.asarray(correction_top32, dtype=np.float64)
    if c64.ndim != 1 or c64.size != 64 or not np.isfinite(c64).all():
        raise E6PolicyError("current GC C64 state must contain 64 finite scores")
    if corrections.shape != (32,) or not np.isfinite(corrections).all():
        raise E6PolicyError("native C1 evidence must contain 32 finite Top-32 residuals")
    if (isinstance(correction_bank_size, bool) or
            int(correction_bank_size) != correction_bank_size or
            int(correction_bank_size) < 0):
        raise E6PolicyError("correction bank size must be a nonnegative integer")
    if int(correction_bank_size) == 0 and np.any(corrections != 0.0):
        raise E6PolicyError("an empty native C1 bank must produce an all-zero residual vector")

    sigma = max(float(np.std(c64, ddof=1)), SIGMA_FLOOR)
    margin = s - theta
    m = margin / sigma
    p = _sigmoid(m)
    pred_defect = bool(s >= theta)
    entropy = _entropy_bern_v22(p)
    psi = math.exp(-0.5 * m * m)
    raw_novelty = _clip(_finite(novelty_percentile, "novelty percentile"), 0.0, 1.0)
    novelty = normalize_novelty_percentile(raw_novelty)

    corr_mean = float(np.mean(corrections))
    corr_top_max = float(np.max(corrections))
    corr_top_min = float(np.min(corrections))
    corr_def = max(0.0, corr_top_max)
    corr_norm = max(0.0, -corr_top_min)
    corr_mag = max(corr_def, corr_norm)
    evidence = _clip(corr_mag / CORRECTION_SCALE, 0.0, 1.0)

    saliency = local_saliency_from_native_stats(topk_dN_max, topk_dN_mean)
    saliency_multiplier = 1.0 + max(0.0, saliency - 0.5)
    u_n = psi * novelty
    u_c = psi * evidence * saliency_multiplier

    near = max(0.005, 0.5 * sigma)
    far = max(0.02, 3.0 * sigma)
    tau = max(0.005, 0.5 * sigma)
    gate_far = _sigmoid((margin + far) / tau)
    gate_near = _sigmoid((-margin - near) / tau)
    weak_corr = 1.0 - evidence
    u_d = 0.0 if pred_defect else p * gate_far * gate_near * weak_corr

    risk = (1.0 - p) if pred_defect else (2.5 * p)
    b = risk + 0.30 * entropy
    n = 0.35 * u_n
    c = 0.15 * u_c
    d = 0.35 * u_d
    return BEAComponents(
        score=s, threshold=theta, sigma=sigma, margin_z=m,
        p_defect=p, pred_defect=pred_defect, entropy=entropy, psi=psi,
        novelty_percentile=raw_novelty, novelty=novelty,
        correction_top_mean=corr_mean, correction_top_max=corr_top_max,
        correction_top_min=corr_top_min,
        correction_bank_size=int(correction_bank_size), corr_def=corr_def,
        corr_norm=corr_norm, corr_mag=corr_mag, correction_evidence=evidence,
        local_saliency=saliency, saliency_multiplier=saliency_multiplier,
        gate_far=gate_far, gate_near=gate_near, weak_correction=weak_corr,
        R=risk, H=entropy, U_N=u_n, U_C=u_c, U_D=u_d,
        B=b, N=n, C=c, D=d,
    )


def _stable_uniform01(seed: int, arm_id: str, event_id_sha256: str,
                      domain: str) -> float:
    if len(event_id_sha256) != 64:
        raise E6PolicyError("event identity must be a SHA-256 hex digest")
    try:
        bytes.fromhex(event_id_sha256)
    except ValueError as exc:
        raise E6PolicyError("event identity must be a SHA-256 hex digest") from exc
    payload = f"E6-BEA-v2\0{int(seed)}\0{arm_id}\0{domain}\0{event_id_sha256}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(1 << 64)


@dataclass(frozen=True)
class QueryProposal:
    arm_id: str
    proposed: bool
    utility: float | None
    threshold: float | None
    objective: float | None
    query_cost: float | None
    reason: str
    random_u: float | None = None
    history_count: int = 0


@dataclass
class QueryController:
    """One arm's online proposal, price, and realised-query accounting."""

    arm_id: str
    seed: int = 0
    query_cost_runtime: float = INITIAL_QUERY_COST
    executed_flags: deque[int] = field(default_factory=lambda: deque(maxlen=QUERY_COST_WINDOW))
    utility_history: deque[float] = field(default_factory=lambda: deque(maxlen=UTILITY_HISTORY_WINDOW))
    events_observed: int = 0
    proposals: int = 0
    executed_queries: int = 0
    group_suppressed: int = 0
    last_observed_position: int = -1
    query_positions: list[int] = field(default_factory=list)
    query_gaps: list[int] = field(default_factory=list)
    quantile_comparisons: int = 0
    quantile_ties: int = 0

    def __post_init__(self) -> None:
        if self.arm_id not in ARM_POLICIES:
            raise E6PolicyError(f"Unknown E6 arm: {self.arm_id}")
        if (not math.isfinite(float(self.query_cost_runtime)) or
                not QUERY_COST_MIN <= float(self.query_cost_runtime) <= QUERY_COST_MAX):
            raise E6PolicyError("query cost must be finite and inside the registered bounds")

    def propose(self, *, position: int, event_id_sha256: str,
                components: BEAComponents) -> QueryProposal:
        """Propose from current and previous observations only; no stream length."""
        if isinstance(position, bool) or int(position) != self.last_observed_position + 1:
            raise E6PolicyError("query policy positions must be observed in order")
        spec = ARM_POLICIES[self.arm_id]
        if spec.kind == "random":
            u = _stable_uniform01(self.seed, self.arm_id, event_id_sha256, "random")
            return QueryProposal(self.arm_id, u < P_RANDOM_FIVE_VIEW,
                                 P_RANDOM_FIVE_VIEW, None, None, None,
                                 "seeded_five_view_bernoulli", u, 0)
        if spec.kind == "periodic":
            proposed = ((int(position) + 1) % QUERY_PERIOD) == 0
            return QueryProposal(self.arm_id, proposed, None, None, None, None,
                                 "observed_event_period_33", None, 0)
        if spec.kind in ("standardized_margin", "novelty"):
            utility = components.psi if spec.kind == "standardized_margin" else components.novelty
            prior = tuple(self.utility_history)
            if len(prior) < UTILITY_COLD_START_EVENTS:
                u = _stable_uniform01(self.seed, self.arm_id, event_id_sha256,
                                      "cold_start_bernoulli")
                return QueryProposal(self.arm_id, u < TARGET_QUERY_RATE,
                                     utility, None, utility, None,
                                     "seeded_cold_start_bernoulli", u, len(prior))
            import numpy as np
            threshold = float(np.quantile(np.asarray(prior, dtype=np.float64), 0.97))
            # Strict comparison is the frozen tie rule: values tied at the
            # previous-history percentile do not create a tie-driven query burst.
            self.quantile_comparisons += 1
            self.quantile_ties += int(float(utility) == threshold)
            proposed = float(utility) > threshold
            return QueryProposal(self.arm_id, proposed, float(utility), threshold,
                                 float(utility), None,
                                 "previous_256_q97_strictly_above" if proposed else
                                 "previous_256_q97_at_or_below", None, len(prior))
        objective = components.objective(self.arm_id)
        cost = float(self.query_cost_runtime)
        return QueryProposal(self.arm_id, objective > cost, objective, cost,
                             objective, cost,
                             "bea_priority_above_runtime_cost" if objective > cost else
                             "bea_priority_not_above_runtime_cost", None, 0)

    def observe_final_action(self, *, position: int, proposal: QueryProposal,
                             executed_query: bool, group_suppressed: bool) -> float | None:
        """Update future-only policy state using the final executed action."""
        if (isinstance(position, bool) or int(position) != position or
                int(position) != self.last_observed_position + 1):
            raise E6PolicyError("query controller observed a duplicated or skipped event")
        if proposal.arm_id != self.arm_id:
            raise E6PolicyError("proposal belongs to a different arm")
        if not isinstance(executed_query, bool) or not isinstance(group_suppressed, bool):
            raise E6PolicyError("final query and suppression actions must be explicit booleans")
        if group_suppressed and not proposal.proposed:
            raise E6PolicyError("a Source Group cannot suppress an absent proposal")
        executed = bool(executed_query)
        if executed and not proposal.proposed:
            raise E6PolicyError("a non-proposed event cannot execute a query")
        if group_suppressed and executed:
            raise E6PolicyError("a Source Group-suppressed proposal cannot execute")
        if proposal.proposed and not executed and not group_suppressed:
            raise E6PolicyError("a proposed query can be suppressed only by Source Group exclusion")
        self.events_observed += 1
        self.proposals += int(proposal.proposed)
        self.group_suppressed += int(bool(group_suppressed))
        if proposal.arm_id in ("Q2", "Q3"):
            if proposal.utility is None or not math.isfinite(float(proposal.utility)):
                raise E6PolicyError("quantile policy must record its observed utility")
            self.utility_history.append(float(proposal.utility))
        next_cost: float | None = None
        if ARM_POLICIES[self.arm_id].adaptive_price:
            self.executed_flags.append(int(executed))
            rho_local = sum(self.executed_flags) / len(self.executed_flags)
            next_cost = min(QUERY_COST_MAX, max(
                QUERY_COST_MIN,
                float(self.query_cost_runtime) + QUERY_COST_LEARNING_RATE *
                (rho_local - TARGET_QUERY_RATE),
            ))
            self.query_cost_runtime = next_cost
        if executed:
            if self.query_positions:
                self.query_gaps.append(int(position) - self.query_positions[-1])
            self.query_positions.append(int(position))
            self.executed_queries += 1
        self.last_observed_position = int(position)
        return next_cost

    def diagnostics(self) -> dict[str, Any]:
        inactivity_intervals = ([self.query_positions[0]] if self.query_positions else [])
        inactivity_intervals.extend(max(0, gap - 1) for gap in self.query_gaps)
        if self.query_positions:
            inactivity_intervals.append(max(
                0, self.events_observed - 1 - self.query_positions[-1]))
        else:
            inactivity_intervals.append(self.events_observed)
        return {
            "arm_id": self.arm_id,
            "events_observed": self.events_observed,
            "proposals": self.proposals,
            "executed_queries": self.executed_queries,
            "realised_query_rate": self.executed_queries / self.events_observed
            if self.events_observed else 0.0,
            "group_suppressed": self.group_suppressed,
            "quantile_comparisons": self.quantile_comparisons,
            "quantile_ties": self.quantile_ties,
            "quantile_tie_rate": (self.quantile_ties / self.quantile_comparisons
                                  if self.quantile_comparisons else None),
            "query_cost_runtime": self.query_cost_runtime
            if ARM_POLICIES[self.arm_id].adaptive_price else None,
            "query_gaps": list(self.query_gaps),
            "max_query_gap": max(self.query_gaps, default=None),
            "max_inactivity_events": max(inactivity_intervals, default=0),
        }
