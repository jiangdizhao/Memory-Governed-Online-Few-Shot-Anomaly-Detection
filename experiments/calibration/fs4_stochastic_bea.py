"""C64-free, label-blind stochastic BEA group-slot design.

No dataset access or model loading is performed in this module.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import re
from typing import Sequence

import numpy as np

POLICY = "STOCHASTIC_BEA_GROUP_SLOT_V1"
TARGET_RATE = 0.037
EXPLORATION = 0.05


def _finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("finite value required")
    return value


def _hash(seed: int, domain: str, identity: str, counter: int = 0) -> int:
    if not identity:
        raise ValueError("opaque identity is required")
    payload = f"{POLICY}|{int(seed)}|{domain}|{identity}|{counter}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def candidate_ordinal(seed: int, group: str) -> int:
    # Rejection avoids modulo bias under the uniform 256-bit hash model.
    limit = (1 << 256) - ((1 << 256) % 5)
    counter = 0
    while True:
        value = _hash(seed, "candidate-slot", group, counter)
        if value < limit:
            return value % 5 + 1
        counter += 1


def action_uniform(seed: int, group: str, event: str) -> float:
    # Separate hash domain from the candidate lottery; use 53 bits in [0,1).
    return (_hash(seed, "query-action", group + "|" + event) >> 203) / (1 << 53)


def bea_components(*, score: float, threshold: float, scale: float,
                   novelty_percentile: float, correction_top32: Sequence[float],
                   correction_bank_size: int, topk_dN_max: float,
                   topk_dN_mean: float) -> dict[str, float]:
    """E6 B/N/D equations with the explicitly replaced lagged score scale."""
    s, theta, sigma = _finite(score), _finite(threshold), _finite(scale)
    if sigma < 0.02:
        raise ValueError("operational score scale below its registered floor")
    corr = np.asarray(correction_top32, dtype=np.float64)
    if corr.shape != (32,) or not np.isfinite(corr).all():
        raise ValueError("native correction residual must contain 32 finite values")
    if isinstance(correction_bank_size, bool) or int(correction_bank_size) != correction_bank_size or correction_bank_size < 0:
        raise ValueError("invalid correction-bank size")
    if correction_bank_size == 0 and np.any(corr != 0):
        raise ValueError("empty correction bank must have zero residuals")
    margin = s - theta
    z = margin / sigma
    sigmoid = lambda x: 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
    p = sigmoid(z)
    pc = min(max(p, 1e-8), 1.0 - 1e-8)
    entropy = -(pc * math.log(pc) + (1.0 - pc) * math.log(1.0 - pc))
    psi = math.exp(-0.5 * z * z)
    novelty = min(1.0, max(0.0, (_finite(novelty_percentile) - 0.05) / 0.90))
    evidence = min(1.0, max(0.0, max(max(0.0, float(corr.max())), max(0.0, -float(corr.min()))) / 0.05))
    dmax, dmean = _finite(topk_dN_max), _finite(topk_dN_mean)
    saliency = min(1.0, max(0.0, (dmax - dmean) / max(abs(dmax), abs(dmean), 1e-6)))
    pred_defect = s >= theta
    risk = 1.0 - p if pred_defect else 2.5 * p
    near, far, tau = max(0.005, 0.5 * sigma), max(0.02, 3.0 * sigma), max(0.005, 0.5 * sigma)
    u_d = 0.0 if pred_defect else p * sigmoid((margin + far) / tau) * sigmoid((-margin - near) / tau) * (1.0 - evidence)
    b, n, d = risk + 0.30 * entropy, 0.35 * psi * novelty, 0.35 * u_d
    return {"B": b, "N": n, "D": d, "utility": b + n + d,
            "sigma_query": sigma, "novelty": novelty, "correction_evidence": evidence,
            "saliency": saliency, "margin_z": z}


@dataclass(frozen=True)
class SealedPrediction:
    position: int
    group: str
    event: str
    score_g: float
    score_gc: float
    scorer_g: str
    scorer_gc: str
    threshold_state: str
    seal: str

    def __post_init__(self) -> None:
        if (self.position < 0 or any(not isinstance(x, str) or not re.fullmatch(r"[0-9a-f]{64}", x)
                for x in (self.group, self.event, self.scorer_g, self.scorer_gc, self.threshold_state, self.seal))):
            raise ValueError("sealed prediction identity is incomplete")
        _finite(self.score_g); _finite(self.score_gc)


@dataclass(frozen=True)
class QueryReceipt:
    prediction: SealedPrediction
    ordinal: int
    candidate: int
    seed: int
    policy: str
    utility: float
    reference_threshold: float
    query_scale: float
    cost: float
    exploitation: bool
    exploration_probability: float
    slot_probability: float
    gate_probability: float | None
    design_probability: float | None
    random_draw: float | None
    executed: bool
    query_seal: str | None = None
    disclosed: bool = False
    label: int | None = None


def receipt_digest(receipt: QueryReceipt) -> str:
    payload = json.dumps(asdict(receipt), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class StochasticBEAGroupSlot:
    """One label-blind sequential policy; feedback disclosure is one-shot."""

    def __init__(self, seed: int = 37) -> None:
        self.seed = int(seed)
        self.price = 0.80
        self.group_counts: dict[str, int] = {}
        self.candidates: dict[str, int] = {}
        self.history: deque[tuple[float, str]] = deque(maxlen=256)
        self.flags: deque[int] = deque(maxlen=256)
        self.receipts: dict[int, QueryReceipt] = {}
        self.events: set[str] = set()
        self.last_position = -1

    def prior_scale(self) -> float:
        values = np.asarray([x[0] for x in self.history], dtype=np.float64)
        if len(values) < 2:
            return 0.02
        median = float(np.median(values))
        return max(1.4826 * float(np.median(np.abs(values - median))), 0.02)

    def decide(self, prediction: SealedPrediction, *, reference_threshold: float,
               novelty_percentile: float, correction_top32: Sequence[float],
               correction_bank_size: int, topk_dN_max: float,
               topk_dN_mean: float, pre_score_scale: float) -> QueryReceipt:
        if prediction.position != self.last_position + 1:
            raise ValueError("events must be sealed once in stream order")
        if self.last_position >= 0:
            previous = self.receipts[self.last_position]
            if not previous.query_seal or (previous.executed and not previous.disclosed):
                raise ValueError("previous query decision or feedback was not completed")
        if prediction.event in self.events:
            raise ValueError("event identity cannot be repeated")
        if _finite(pre_score_scale) != self.prior_scale():
            raise ValueError("query scale must be obtained before this event")
        group = prediction.group
        ordinal = self.group_counts.get(group, 0) + 1
        if ordinal > 5:
            raise ValueError("source group has more than five views")
        candidate = self.candidates.setdefault(group, candidate_ordinal(self.seed, group))
        components = bea_components(score=prediction.score_gc, threshold=reference_threshold,
            scale=pre_score_scale, novelty_percentile=novelty_percentile,
            correction_top32=correction_top32, correction_bank_size=correction_bank_size,
            topk_dN_max=topk_dN_max, topk_dN_mean=topk_dN_mean)
        utility = components["utility"]
        exploit = utility > self.price
        eligible = ordinal == candidate
        gate = (1.0 if exploit else EXPLORATION) if eligible else None
        draw = action_uniform(self.seed, group, prediction.event) if eligible else None
        executed = bool(eligible and draw is not None and gate is not None and draw < gate)
        receipt = QueryReceipt(prediction, ordinal, candidate, self.seed, POLICY, utility,
            float(reference_threshold), float(pre_score_scale), self.price, exploit, EXPLORATION, 0.2, gate,
            0.2 * gate if gate is not None else None, draw, executed)
        self.group_counts[group] = ordinal
        self.events.add(prediction.event)
        self.receipts[prediction.position] = receipt
        self.flags.append(int(executed))
        rho = sum(self.flags) / len(self.flags)
        self.price = min(3.0, max(0.05, self.price + 0.02 * (rho - TARGET_RATE)))
        # All current scores enter the unlabeled history only after the decision.
        self.history.append((prediction.score_gc, group))
        self.last_position = prediction.position
        return receipt

    def seal_query(self, receipt: QueryReceipt, seal: str) -> QueryReceipt:
        if (self.receipts.get(receipt.prediction.position) != receipt or receipt.query_seal or
                not isinstance(seal, str) or seal != receipt_digest(receipt)):
            raise ValueError("query decision must be sealed once by the event ledger")
        sealed = replace(receipt, query_seal=seal)
        self.receipts[receipt.prediction.position] = sealed
        return sealed

    def disclose(self, receipt: QueryReceipt, label: int) -> QueryReceipt:
        saved = self.receipts.get(receipt.prediction.position)
        if saved != receipt or not receipt.query_seal or not receipt.executed or receipt.disclosed or label not in (0, 1) or isinstance(label, bool):
            raise ValueError("feedback requires a genuine, executed, undisclosed sealed query")
        disclosed = replace(receipt, disclosed=True, label=int(label))
        self.receipts[receipt.prediction.position] = disclosed
        return disclosed

    def validate_complete_stream(self) -> None:
        """Check group geometry and causal completion before a global stream seal."""
        if not self.receipts or any(count != 5 for count in self.group_counts.values()):
            raise ValueError("complete stream requires exactly five views per group")
        if any(not r.query_seal or (r.executed and not r.disclosed) for r in self.receipts.values()):
            raise ValueError("complete stream requires all query decisions and feedback to be sealed")

    def diagnostics(self) -> dict[str, float | int | None]:
        receipts = list(self.receipts.values())
        executed = [r for r in receipts if r.executed]
        exploration = sum(not r.exploitation for r in executed)
        return {"events": len(receipts), "queries": len(executed),
                "realized_rate": len(executed) / len(receipts) if receipts else None,
                "rate_deviation": len(executed) / len(receipts) - TARGET_RATE if receipts else None,
                "exploration_queries": exploration,
                "exploration_fraction": exploration / len(executed) if executed else None,
                "target_rate": TARGET_RATE, "price": self.price,
                "history_size": len(self.history)}
