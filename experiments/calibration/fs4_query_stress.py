"""Deterministic, synthetic-only label-blind query-policy stress diagnostics."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

import numpy as np

from .fs4_stochastic_bea import (SealedPrediction, StochasticBEAGroupSlot,
                                 action_uniform, candidate_ordinal, receipt_digest)

SCENARIOS = ("high", "low", "stable", "shift_up", "shift_down", "drift",
             "no_exploitation", "interleaved", "imbalance", "ties")


def _id(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def run_scenario(name: str, *, groups: int = 400, seed: int = 37) -> dict:
    if name not in SCENARIOS or groups < 1:
        raise ValueError("invalid synthetic stress scenario")
    policy = StochasticBEAGroupSlot(seed)
    count = 5 * groups
    sequence = [(g, v) for v in range(5) for g in range(groups)] if name == "interleaved" else [(g, v) for g in range(groups) for v in range(5)]
    rates, probabilities, regions, price_samples = [], Counter(), {}, []
    query_positions: list[int] = []
    candidates = exploration = normals = 0
    for pos, (group_index, ordinal) in enumerate(sequence):
        x = pos / max(1, count - 1)
        stable = 0.50 + 0.03 * math.sin(pos * 0.13)
        score = {"high": 0.499, "low": 0.10, "stable": stable,
                 "shift_up": stable + (0.20 if x >= 0.5 else 0),
                 "shift_down": stable - (0.20 if x >= 0.5 else 0),
                 "drift": stable + 0.30 * x, "no_exploitation": 0.10,
                 "interleaved": stable, "imbalance": 0.499 if x < 0.2 else 0.10,
                 "ties": 0.50}[name]
        group, event = _id(f"stress-group-{group_index}"), _id(f"stress-event-{group_index}-{ordinal}")
        p = SealedPrediction(pos, group, event, score, score, _id("g"), _id("gc"), _id("theta"), _id("sealed"))
        r = policy.decide(p, reference_threshold=0.5, novelty_percentile=0.50000001 if name == "ties" else 0.5,
            correction_top32=np.zeros(32), correction_bank_size=0, topk_dN_max=1.0,
            topk_dN_mean=0.5, pre_score_scale=policy.prior_scale())
        assert r.candidate == candidate_ordinal(seed, group)
        if r.ordinal == r.candidate:
            candidates += 1
            assert r.random_draw == action_uniform(seed, group, event)
            assert r.design_probability == 0.2 * r.gate_probability
            assert r.executed == (r.random_draw < r.gate_probability)
            probabilities[str(r.design_probability)] += 1
        else:
            assert r.design_probability is None and r.gate_probability is None and r.random_draw is None
        r = policy.seal_query(r, receipt_digest(r))
        region = "below" if score < 0.48 else "above" if score > 0.52 else "near"
        row = regions.setdefault(region, {"events": 0, "queries": 0})
        row["events"] += 1
        if r.executed:
            query_positions.append(pos); row["queries"] += 1
            exploration += int(not r.exploitation)
            # A synthetic feedback oracle supplies only the queried outcome;
            # its label never affects the generated score or the query policy.
            label = 1 if name == "no_exploitation" else 0
            normals += int(label == 0)
            policy.disclose(r, label)
        if pos % 100 == 0 or pos == count - 1:
            price_samples.append({"event": pos, "price": policy.price})
        rates.append(int(r.executed))
    policy.validate_complete_stream()
    gaps = [b - a - 1 for a, b in zip([-1] + query_positions, query_positions + [count])]
    queries = len(query_positions)
    return {"scenario": name, "events": count, "groups": groups, "candidate_slots": candidates,
            "queries": queries, "image_query_rate": queries / count, "group_query_fraction": queries / groups,
            "target_deviation": queries / count - 0.037, "exploration_queries": exploration,
            "exploration_fraction": exploration / queries if queries else None,
            "verified_synthetic_normals": normals, "price_trajectory_every_100_events": price_samples,
            "longest_query_free_interval": max(gaps), "eligible_design_probabilities": dict(probabilities),
            "score_regions": regions, "uncovered_observed_regions": [k for k, v in regions.items() if not v["queries"]],
            "unobserved_regions": [k for k in ("below", "near", "above") if k not in regions]}


def main() -> None:
    print(json.dumps([run_scenario(name) for name in SCENARIOS], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
