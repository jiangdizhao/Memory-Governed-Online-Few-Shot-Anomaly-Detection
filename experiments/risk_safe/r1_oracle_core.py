"""R1 offline Oracle search over state-dependent verified-normal admissions.

This is a retrospective DEVELOPER diagnostic, not a deployable controller:
search-set ground-truth labels are used in selecting memory states. Only exact
mode with all 2^N leaves evaluated is a finite-protocol global optimum.
Beam mode reports ONLY the best feasible configuration FOUND.
No test/probe B access, no claim of full-stream prequential performance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence


class R1ProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class DecisionState:
    bank: Any
    decisions: str = ""
    admissions: int = 0
    effective_updates: int = 0
    appended: int = 0
    replaced: int = 0
    skipped: int = 0


def transition(state: DecisionState, candidate: Any, take: bool) -> DecisionState:
    """Reject never mutates state. Accept replays exact native Greedy proposal."""
    if not take:
        return DecisionState(state.bank, state.decisions + "0", state.admissions,
                             state.effective_updates, state.appended,
                             state.replaced, state.skipped)
    shadow, audit = state.bank.proposed(candidate, already_normalized=True)
    if shadow.version < state.bank.version:
        raise R1ProtocolError("Non-monotone shadow memory version")
    changed = int(bool(audit.appended or audit.replaced))
    if changed and shadow.version != state.bank.version + 1:
        raise R1ProtocolError("Changed shadow must advance version exactly once")
    if not changed and shadow.sha256() != state.bank.sha256():
        raise R1ProtocolError("No-change proposal mutated bank")
    return DecisionState(shadow, state.decisions + "1",
                         state.admissions + 1, state.effective_updates + changed,
                         state.appended + audit.appended,
                         state.replaced + audit.replaced,
                         state.skipped + audit.skipped)


def evaluate_scores(labels: Sequence[int], static_scores: Sequence[float],
                    scores: Sequence[float], theta: float, *,
                    lambda_defect: float = 1.0) -> dict:
    if not (len(labels) == len(static_scores) == len(scores)):
        raise R1ProtocolError("Probe shape mismatch")
    if not labels or set(labels) != {0, 1}:
        raise R1ProtocolError("Oracle probe needs real normal and defect images")
    if not math.isfinite(theta) or lambda_defect < 0 or not math.isfinite(lambda_defect):
        raise R1ProtocolError("Invalid threshold/risk penalty")
    if any(not math.isfinite(float(x)) for x in (*static_scores, *scores)):
        raise R1ProtocolError("Nonfinite scores")
    n0 = sum(y == 0 for y in labels)
    n1 = len(labels) - n0
    tp = tn = fp = fn = new_fn = rescued_fn = rescued_fp = new_fp = 0
    normal_relief = defect_erosion = normal_hinge = 0.0
    for y, baseline, current in zip(labels, static_scores, scores):
        old = baseline >= theta
        pred = current >= theta
        if y == 0:
            fp += int(pred)
            tn += int(not pred)
            rescued_fp += int(old and not pred)
            new_fp += int(not old and pred)
            normal_relief += max(0., baseline - current)
            normal_hinge += max(0., baseline - theta) - max(0., current - theta)
        else:
            tp += int(pred)
            fn += int(not pred)
            new_fn += int(old and not pred)
            rescued_fn += int(not old and pred)
            defect_erosion += max(0., baseline - current)
    normal_relief /= n0
    defect_erosion /= n1
    return {
        "n_images": len(labels), "n_normal": n0, "n_defect": n1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "fpr": fp / n0, "fnr": fn / n1,
        "balanced_accuracy": .5 * (tn / n0 + tp / n1),
        "new_false_negatives": new_fn,
        "rescued_false_negatives": rescued_fn,
        "rescued_false_positives": rescued_fp,
        "new_false_positives": new_fp,
        "mean_normal_score_relief": normal_relief,
        "mean_defect_score_erosion": defect_erosion,
        "mean_normal_hinge_difference_signed": normal_hinge / n0,
        "risk_adjusted_relief": normal_relief - lambda_defect * defect_erosion,
        "safe_no_new_fn": new_fn == 0,
    }


def search_rank(m: dict, s: DecisionState) -> tuple:
    """Predeclare lexicographic deployment-risk FIRST, then normal rescue.

    Outcome on the searched Probe-A subset is retrospective/optimistic.
    Uses group-level selectivity nowhere; all outcomes remain image-level.
    """
    return (
        int(m["safe_no_new_fn"]),
        -int(m["new_false_negatives"]),
        int(m["rescued_false_positives"]),
        float(m["risk_adjusted_relief"]),
        float(m["balanced_accuracy"]),
        -int(s.admissions), -int(s.effective_updates),
        -int(s.decisions or "0", 2),
    )


def record(state: DecisionState, metrics: dict) -> dict:
    return {
        "decisions": state.decisions,
        "n_admitted": state.admissions,
        "n_effective_updates": state.effective_updates,
        "n_appended": state.appended,
        "n_replaced": state.replaced,
        "n_skipped": state.skipped,
        "bank_sha256": state.bank.sha256(),
        "n_dynamic_prototypes": sum(len(x) for x in state.bank.layers),
        **metrics,
    }


def run_exact(candidates: Sequence[Any], start: DecisionState,
              evaluate: Callable[[DecisionState], dict], *,
              max_leaves: int = 4096) -> dict:
    n = len(candidates)
    if n == 0 or n > 20 or (1 << n) > max_leaves:
        raise R1ProtocolError("Exact mode refuses partial 2^N search: lower N or increase hard max_leaves")
    rows, best_state, best_metric, best_rank = [], None, None, None
    evals = 0
    def dfs(i, state):
        nonlocal best_state, best_metric, best_rank, evals
        if i == n:
            metric = evaluate(state)
            evals += 1
            rows.append(record(state, metric))
            rank = search_rank(metric, state)
            if best_rank is None or rank > best_rank:
                best_state, best_metric, best_rank = state, metric, rank
            return
        dfs(i + 1, transition(state, candidates[i], False))
        dfs(i + 1, transition(state, candidates[i], True))
    dfs(0, start)
    if evals != (1 << n):
        raise R1ProtocolError("Exact leaf count mismatch")
    return dict(status="EXACT_FINITE_PROTOCOL_OPTIMUM", evaluated_leaves=evals,
                total_possible_leaves=1 << n, globally_exhaustive=True,
                best_state=best_state, best_metrics=best_metric, leaf_rows=rows,
                search_evaluations=evals)


def run_beam(candidates: Sequence[Any], start: DecisionState,
             evaluate: Callable[[DecisionState], dict], *,
             beam_width: int = 4) -> dict:
    if not candidates or beam_width < 1 or beam_width > 256:
        raise R1ProtocolError("Invalid beam candidate length or width")
    beam = [start]
    n_eval = 0
    stage_rows = []
    for depth, candidate in enumerate(candidates, 1):
        scored = []
        for state in beam:
            for take in (False, True):
                child = transition(state, candidate, take)
                metric = evaluate(child)
                n_eval += 1
                scored.append((search_rank(metric, child), child, metric))
        scored.sort(key=lambda x: x[0], reverse=True)
        new_beam, seen = [], set()
        for _rank, state, metric in scored:
            digest = state.bank.sha256()
            # Different sequences reaching identical states cannot improve
            # the current objective; keep deterministic best rank per digest.
            if digest in seen:
                continue
            seen.add(digest)
            new_beam.append(state)
            if len(new_beam) >= beam_width:
                break
        beam = new_beam
        if not beam:
            raise R1ProtocolError("Beam became empty")
        stage_rows.append(dict(depth=depth, explored=len(scored),
                               surviving=len(beam), best_decisions=scored[0][1].decisions,
                               **{f"best_{k}":v for k,v in scored[0][2].items()
                                  if k in ("balanced_accuracy", "new_false_negatives",
                                           "rescued_false_positives", "risk_adjusted_relief")}))
    scored_final = [(search_rank(m, s), s, m)
                    for s in beam for m in (evaluate(s),)]
    n_eval += len(scored_final)
    scored_final.sort(key=lambda x: x[0], reverse=True)
    _, best_state, best_metrics = scored_final[0]
    return dict(status="BEAM_BEST_FOUND_NOT_AN_UPPER_BOUND",
                globally_exhaustive=False, evaluated_leaves=len(beam),
                total_possible_leaves=2 ** len(candidates),
                best_state=best_state, best_metrics=best_metrics,
                leaf_rows=[record(s,m) for _,s,m in scored_final],
                stage_rows=stage_rows, search_evaluations=n_eval)


def frontier_by_admission_count(rows: Sequence[dict]) -> list[dict]:
    """Best balanced accuracy in every feasible admission-count slice."""
    out = {}
    for row in rows:
        k = row["n_admitted"]
        if k not in out or (row["balanced_accuracy"], -row["new_false_negatives"],
                            -row["mean_defect_score_erosion"]) > (
                            out[k]["balanced_accuracy"], -out[k]["new_false_negatives"],
                            -out[k]["mean_defect_score_erosion"]):
            out[k] = row
    return [out[k] for k in sorted(out)]
