"""Two-scenario synthetic full-engine B7 activation diagnostic.

Uses the actual PrequentialEngine, AnchorLedger, CalibrationGreedyState, and
historical GreedyBank operator with synthetic CPU features. The cosine scorer
is the explicitly synthetic fixture; this is not native-image or Real-IAD
validation. No raw event trace or synthetic feature tensor is exported.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from collections.abc import Mapping
import math
import time
import tracemalloc
from typing import Any

import numpy as np
import torch

from experiments.risk_safe.shadow_memory import CosinePatchScorer, GreedyBank
from .anchor_ledger import AnchorLedger
from .controllers import theta0_from_c64
from .engine import PrequentialEngine
from .memory_state import CalibrationGreedyState
from .method_registry import build_controllers
from .metrics import PrivateOutcomeLedger
from .profiles import QueryProtocolConfig
from .protocol import ScoreResult, StreamEvent, TapeRecord
from .query_tape import build_qa_ts


HARNESS_VERSION = "B7_SYNTHETIC_FULL_ENGINE_ACTIVATION_PASS-v1.0.0"
N_EVENTS = 1024
SPLIT = 517
MEMORY_DELTA = 0.0001


class _SyntheticNativeMemory:
    """Controller-independent native Greedy operator fixture, with source map."""

    def __init__(self, dimensions: int = 4, layers: int = 2):
        self.bank = CalibrationGreedyState(
            GreedyBank([torch.empty((0, dimensions), dtype=torch.float32)
                        for _ in range(layers)], [256] * layers), 0)
        self.source_rows: list[list[str]] = [[] for _ in range(layers)]
        self.source_revision = 0
        self.write_count = 0

    @property
    def version(self) -> int:
        return self.bank.version

    @property
    def digest(self) -> str:
        return self.bank.sha256()

    def write(self, event: StreamEvent, score: ScoreResult,
              feedback: Any) -> dict[str, Any]:
        before_version, before_digest = self.version, self.digest
        if not feedback.queried or feedback.label != 0:
            return {"eligible": False, "reason": "feedback_not_verified_normal",
                    "version_before": before_version,
                    "version_after": before_version,
                    "digest_before": before_digest,
                    "digest_after": before_digest}
        candidate = event.anchor_features
        candidate_layers = [row.detach().to(dtype=torch.float32, device="cpu")
                            for row in candidate]
        proposed, audit, content_changed = self.bank.proposed(
            candidate_layers, already_normalized=False)
        if content_changed:
            next_sources = []
            for old, layer_rows in zip(self.source_rows, candidate_layers):
                rows = list(old)
                rows.extend([event.record.official_sample_key] * len(layer_rows))
                next_sources.append(rows)
            if any(len(rows) != len(layer) for rows, layer in
                   zip(next_sources, proposed.layers)):
                raise RuntimeError("Synthetic Greedy source attribution diverged")
            self.bank = proposed
            self.source_rows = next_sources
            self.source_revision += 1
            self.write_count += 1
        return {"eligible": True, "content_changed": content_changed,
                "reason": "queried_verified_normal_synthetic_greedy_append",
                "appended": audit.appended, "replaced": audit.replaced,
                "skipped": audit.skipped,
                "version_before": before_version,
                "version_after": self.version,
                "digest_before": before_digest, "digest_after": self.digest,
                "source_revision": self.source_revision,
                "fixed_reference_policy": "query-normal only; no controller threshold"}

    def state_dict(self) -> dict[str, Any]:
        return {"schema": "synthetic-native-greedy-memory.v1",
                "capacities": list(self.bank.capacities),
                "legacy_version": self.bank.legacy_greedy_version,
                "memory_content_version": self.bank.memory_content_version,
                "layers": [layer.detach().cpu().contiguous().clone()
                           for layer in self.bank.layers],
                "source_rows": [list(rows) for rows in self.source_rows],
                "source_revision": self.source_revision,
                "write_count": self.write_count,
                "digest": self.digest}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state.get("schema") != "synthetic-native-greedy-memory.v1" or
                state.get("capacities") != list(self.bank.capacities)):
            raise RuntimeError("Synthetic Greedy checkpoint identity mismatch")
        layers = [torch.as_tensor(value).detach().cpu().contiguous().clone()
                  for value in state["layers"]]
        legacy = GreedyBank(layers, self.bank.capacities,
                            version=int(state["legacy_version"]))
        restored = CalibrationGreedyState(
            legacy, int(state["memory_content_version"]))
        sources = [list(rows) for rows in state["source_rows"]]
        if len(sources) != len(layers) or any(
                len(source) != len(layer) for source, layer in zip(sources, layers)):
            raise RuntimeError("Synthetic Greedy checkpoint source map mismatch")
        if restored.sha256() != state["digest"]:
            raise RuntimeError("Synthetic Greedy checkpoint digest mismatch")
        self.bank, self.source_rows = restored, sources
        self.source_revision = int(state["source_revision"])
        self.write_count = int(state["write_count"])


def _unit_feature(label: int, event_index: int, layer: int) -> torch.Tensor:
    rng = np.random.default_rng((100_000 + event_index * 13 + layer) % (2 ** 32))
    base = np.array([1.0, 0.0, 0.0, 0.0] if label == 0 else
                    [-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    value = base + rng.normal(0.0, 0.025, size=4).astype(np.float32)
    value /= np.linalg.norm(value)
    return torch.tensor(value[None, :], dtype=torch.float32)


def _exact_state_equal(left: Any, right: Any) -> bool:
    """Compare nested checkpoints without tensor/array truth-value coercion."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and
                left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and
                torch.equal(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and
                left.dtype == right.dtype and left.shape == right.shape and
                np.array_equal(left, right, equal_nan=True))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping) and
                left.keys() == right.keys() and
                all(_exact_state_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right) and
                all(_exact_state_equal(a, b) for a, b in zip(left, right)))
    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()
    try:
        return bool(left == right)
    except (TypeError, ValueError, RuntimeError):
        return False


def _scenario_inputs(name: str) -> tuple[list[StreamEvent], list[int], Any,
                                          list[tuple[str, int, Any]],
                                          CosinePatchScorer]:
    records = [TapeRecord(i, f"b7-{name}-group-{i:04d}", i % 5 + 1,
                          f"b7-{name}-view-{i:04d}", phase=i * 16 // N_EVENTS)
               for i in range(N_EVENTS)]
    tape = build_qa_ts(records, QueryProtocolConfig())
    if name == "sparse_feedback_mechanism":
        rng = np.random.default_rng(20261029)
        labels = [int(value < 0.08) for value in rng.random(N_EVENTS)]
    elif name == "two_sided_evidence_stress":
        labels = [int(((i * 7 + 3) % 13) < 8) for i in range(N_EVENTS)]
    else:
        raise ValueError("Unknown B7 activation scenario")

    fixed_layers = []
    for _layer in range(2):
        support = torch.stack([_unit_feature(0, -10 - j, _layer)[0]
                               for j in range(4)])
        fixed_layers.append(support)
    scorer = CosinePatchScorer(fixed_layers)
    c64 = []
    for i in range(64):
        features = tuple(_unit_feature(0, -1000 - i, layer)
                         for layer in range(2))
        c64.append((f"b7-{name}-c64-{i:03d}", i % 5 + 1, features))

    events = []
    for i, record in enumerate(records):
        features = tuple(_unit_feature(labels[i], i, layer) for layer in range(2))
        events.append(StreamEvent(record, None, features))
    return events, labels, tape, c64, scorer


def _make_engine(name: str, *, b7_threshold_offset: float = 0.0):
    events, labels, tape, c64, scorer = _scenario_inputs(name)
    memory = _SyntheticNativeMemory()
    ledger = AnchorLedger(c64, deployed_support_keys=[f"support-{i}" for i in range(4)])
    c64_initial = ledger.refresh(memory=memory.bank.legacy_bank,
                                 source_rows=memory.source_rows, scorer=scorer)
    theta0 = theta0_from_c64([row.score for row in c64_initial.normal_anchors])
    controllers = build_controllers(theta0,
                                    [row.score for row in c64_initial.normal_anchors])
    if b7_threshold_offset:
        next(controller for controller in controllers if controller.name == "B7").theta += b7_threshold_offset

    def score_fn(event: StreamEvent) -> ScoreResult:
        score = scorer(event.anchor_features, memory.bank.legacy_bank)
        digest = hashlib.sha256(
            f"synthetic-cosine:{memory.version}:{memory.digest}".encode()).hexdigest()
        return ScoreResult(float(score), memory.version, digest)

    def evidence_fn(event: StreamEvent | None, feedback: Any):
        if event is not None and feedback is not None and feedback.queried:
            ledger.add_queried(
                source_key=event.record.official_sample_key,
                camera_id=event.record.camera_id,
                label=int(feedback.label), features=event.anchor_features,
                t=event.record.position, memory_version=memory.version)
        snapshot = ledger.refresh(memory=memory.bank.legacy_bank,
                                  source_rows=memory.source_rows, scorer=scorer)
        ledger.assert_current(snapshot, memory.version)
        return snapshot

    engine = PrequentialEngine(
        tape=tape, events=events, outcomes=PrivateOutcomeLedger(labels),
        controllers=controllers, score_fn=score_fn,
        memory_state_fn=lambda: (memory.version, memory.digest),
        write_fn=memory.write, evidence_fn=evidence_fn,
        query_config=QueryProtocolConfig(),
        checkpoint_components={"synthetic_greedy_memory": memory,
                               "synthetic_anchor_ledger": ledger},
        provenance_id=f"{HARNESS_VERSION}:{name}:{tape.digest}:b7-offset={b7_threshold_offset}")
    return engine, memory, ledger, tape, theta0


def _summarize(name: str, engine: PrequentialEngine, memory: _SyntheticNativeMemory,
               ledger: AnchorLedger, tape: Any, theta0: float,
               run: dict[str, Any], *, elapsed: float, peak_bytes: int) -> dict[str, Any]:
    b7 = next(c for c in engine.controllers if c.name == "B7")
    state = b7.state_dict()
    trace = run["trace"]
    theta = [row["theta_post"].get("B7") for row in trace]
    threshold_updates = [
        {"t": i, "threshold": float(value)}
        for i, value in enumerate(theta)
        if value is not None and (i == 0 or theta[i - 1] != value)]
    deltas = [abs(float(after) - float(before))
              for before, after in zip(theta, theta[1:])
              if before is not None and after is not None]
    diagnostics = [row["controller_diagnostics"].get("B7", {}) for row in trace]
    statuses = [str(row.get("status", "")) for row in diagnostics]
    candidate_increments = []
    optimization_revisions = []
    previous_candidates = 0
    for diagnostic in diagnostics:
        current = int(diagnostic.get("total_candidate_evaluations", previous_candidates))
        if current > previous_candidates:
            candidate_increments.append(current - previous_candidates)
            optimization_revisions.append(int(diagnostic.get("evidence_revision", -1)))
        previous_candidates = current
    optimization_revisions_unique = (
        len(optimization_revisions) == len(set(optimization_revisions)) and
        all(revision >= 0 for revision in optimization_revisions))
    anchor_state = ledger.state_dict()
    normal_lso_rows = sum(
        int(row.get("excluded_rows", 0)) for row in anchor_state["anchors"][64:]
        if row.get("label") == 0)
    c64_rescored_current = all(
        row["score_version"] == memory.version
        for row in anchor_state["anchors"][:64])
    return {
        "scenario": name, "status": "PASS",
        "scenario_design_note": (
            "Predetermined low-defect synthetic labels under a label-blind 3% Q-A-TS tape; queried feedback is disclosed only through the private evaluator after prediction."
            if name == "sparse_feedback_mechanism" else
            "Mechanism stress test: 21 of 31 queried events are synthetic defects (67.7%) to exercise two-sided constraints; this is not a Real-IAD prevalence model. Labels remain hidden from controllers."),
        "run_scope": "synthetic_cpu_full_engine_mechanism_test_only",
        "n_events": len(trace), "query_count": len(tape.events),
        "query_fraction_images": len(tape.events) / N_EVENTS,
        "synthetic_evaluator_queried_normal_count": sum(
            engine.outcomes._labels[t] == 0 for t in tape.positions),
        "synthetic_evaluator_queried_defect_count": sum(
            engine.outcomes._labels[t] == 1 for t in tape.positions),
        "query_tape_sha256": tape.digest,
        "score_vector_sha256": hashlib.sha256(
            np.asarray(run["scores"], dtype=np.float64).tobytes()).hexdigest(),
        "query_positions_sha256": hashlib.sha256(
            ",".join(map(str, sorted(tape.positions))).encode()).hexdigest(),
        "theta0_synthetic_c64": theta0,
        "memory_version": memory.version,
        "memory_sha256": memory.digest,
        "memory_write_count": memory.write_count,
        "anchor_evidence_revision": anchor_state["evidence_revision"],
        "c64_current_memory_version_all_64": c64_rescored_current,
        "queried_normal_lso_source_rows_excluded": normal_lso_rows,
        "queried_anchor_count": anchor_state["queried_count"],
        "B7": {
            "state_final": state["state"],
            "decision_final": state["decision"],
            "state_event_counts": {key: statuses.count(key) for key in
                                    ("NONE", "NORMAL_ONLY", "BOTH", "CONFLICT")},
            "evidence_revision_final": state["evidence_revision"],
            "memory_version_final": state["memory_version"],
            "candidate_evaluations": state["total_candidate_evaluations"],
            "bound_evaluations": state["total_bound_evaluations"],
            "optimization_event_count": len(candidate_increments),
            "optimization_evidence_revisions": optimization_revisions,
            "optimization_revisions_unique": optimization_revisions_unique,
            "candidate_evaluations_per_optimization": candidate_increments,
            "threshold_update_count": sum(a != b for a, b in zip(theta, theta[1:])),
            "threshold_updates": threshold_updates,
            "max_observed_step": max(deltas, default=0.0),
            "configured_max_step": b7.config.max_threshold_step,
            "hold_event_count": sum(str(d.get("decision", "")).startswith("HOLD")
                                    for d in diagnostics),
            "conflict_event_count": statuses.count("CONFLICT"),
            "score_stream_consumed_by_B7": False,
            "queried_only_feedback": True,
        },
        "runtime_seconds": elapsed,
        "python_peak_allocated_bytes": peak_bytes,
        "greedy_bank_prototype_bytes": memory.bank.legacy_bank.bytes(),
        "controller_names": [c.name for c in engine.controllers],
        "synthetic_features_and_event_trace_exported": False,
    }


def _run_b7_engine_activation_impl() -> dict[str, Any]:
    reports: dict[str, Any] = {}
    invariance: dict[str, bool] = {}
    for scenario in ("sparse_feedback_mechanism", "two_sided_evidence_stress"):
        engine, memory, ledger, tape, theta0 = _make_engine(scenario)
        tracemalloc.start()
        start = time.perf_counter()
        full = engine.run()
        full_checkpoint = engine.state_dict()
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        interrupted, _, _, interrupted_tape, _ = _make_engine(scenario)
        partial = interrupted.run(max_events=SPLIT)
        if partial["status"] != "INCOMPLETE":
            raise RuntimeError("Expected an event-boundary partial engine run")
        resumed, resumed_memory, resumed_ledger, resumed_tape, _ = _make_engine(scenario)
        resumed.load_state_dict(partial["checkpoint"])
        replayed = resumed.run()
        keys = ("status", "cursor", "tape_digest", "scores", "predictions",
                "prediction_availability", "trace", "metrics", "controller_states")
        resume_equal = all(_exact_state_equal(full[key], replayed[key])
                           for key in keys)
        resume_equal = (resume_equal and _exact_state_equal(
                            full_checkpoint, resumed.state_dict()) and
                        _exact_state_equal(memory.state_dict(),
                                           resumed_memory.state_dict()) and
                        _exact_state_equal(ledger.state_dict(),
                                           resumed_ledger.state_dict())
                        and tape.digest == resumed_tape.digest == interrupted_tape.digest)

        invariant = None
        if scenario == "two_sided_evidence_stress":
            paired, paired_memory, _, paired_tape, _ = _make_engine(
                scenario, b7_threshold_offset=.37)
            counterfactual = paired.run()
            shared_memory_path = lambda result: [
                (row["memory_version_before"], row["memory_version_after"],
                 row["memory_sha256_before"], row["memory_sha256_after"])
                for row in result["trace"]]
            invariant = (_exact_state_equal(full["scores"], counterfactual["scores"]) and
                         _exact_state_equal(shared_memory_path(full),
                                            shared_memory_path(counterfactual)) and
                         _exact_state_equal(memory.state_dict(),
                                            paired_memory.state_dict()) and
                         tape.digest == paired_tape.digest and
                         all(row["query_event"] == other["query_event"]
                             for row, other in zip(full["trace"], counterfactual["trace"])))
            invariance[scenario] = invariant
        summary = _summarize(scenario, engine, memory, ledger, tape, theta0,
                             full, elapsed=elapsed, peak_bytes=peak)
        summary["checkpoint_resume_identical"] = resume_equal
        summary["threshold_counterfactual_score_memory_query_invariance"] = (
            invariant if invariant is not None else "TESTED_IN_TWO_SIDED_STRESS_SCENARIO")
        if not resume_equal or invariant is False:
            summary["status"] = "FAIL"
        if not summary["B7"]["optimization_revisions_unique"]:
            summary["status"] = "FAIL"
        if summary["B7"]["max_observed_step"] > summary["B7"]["configured_max_step"] + 1e-12:
            summary["status"] = "FAIL"
        reports[scenario] = summary

    return {
        "schema": "calibration.b7_synthetic_full_engine_activation.v1",
        "pass_name": HARNESS_VERSION,
        "status": "PASS" if (all(row["status"] == "PASS" for row in reports.values()) and
                              invariance.get("two_sided_evidence_stress") is True)
        else "FAIL",
        "scope": "synthetic mechanism activation; not native image parity or Real-IAD validation",
        "stage1_or_stage2_run": False,
        "confirmation_outcomes_opened": False,
        "synthetic_evaluator_labels_exposed_to_controllers": False,
        "query_protocol": {"name": "Q-A-TS-v1", "fraction": .03,
                           "seed": 37, "tape_label_blind": True},
        "native_greedy_operator": "experiments.risk_safe.shadow_memory.GreedyBank via CalibrationGreedyState; synthetic CPU candidates",
        "scorer": "CosinePatchScorer synthetic fixture only; no Real-IAD or v22 score claim",
        "scenarios": reports,
        "shared_invariance": invariance,
    }


@lru_cache(maxsize=1)
def run_b7_engine_activation() -> dict[str, Any]:
    """Run exact synthetic checks with a temporary one-thread CPU setting."""
    original_threads = torch.get_num_threads()
    try:
        # The fixture performs many tiny tensor rescoring calls; a single CPU
        # worker avoids large thread-pool startup overhead without changing
        # scorer or memory equations.
        if original_threads != 1:
            torch.set_num_threads(1)
        return _run_b7_engine_activation_impl()
    finally:
        if torch.get_num_threads() != original_threads:
            torch.set_num_threads(original_threads)
