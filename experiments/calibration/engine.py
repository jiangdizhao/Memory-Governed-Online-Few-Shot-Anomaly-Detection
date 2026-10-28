"""Single-trajectory, predict-before-disclosure prequential engine."""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .metrics import PrivateOutcomeLedger
from .protocol import (EvidenceSnapshot, Feedback, Observation, Prediction,
                       ScoreResult, StreamEvent)
from .query_tape import QueryTape, verify_tape
from .profiles import QueryProtocolConfig, canonical_json


class EngineError(RuntimeError): pass


class PrequentialEngine:
    def __init__(self, *, tape: QueryTape, events: Sequence[StreamEvent],
                 outcomes: PrivateOutcomeLedger, controllers: Sequence[Any],
                 score_fn: Callable[[StreamEvent], ScoreResult],
                 memory_state_fn: Callable[[], tuple[int, str]],
                 write_fn: Callable[[StreamEvent, ScoreResult, Feedback], Any],
                 evidence_fn: Callable[[StreamEvent | None, Feedback | None],
                                       EvidenceSnapshot] | None = None,
                 query_config: QueryProtocolConfig = QueryProtocolConfig(),
                 checkpoint_components: dict[str, Any] | None = None,
                 provenance_id: str):
        if len(events) != len(outcomes):
            raise EngineError("Event/evaluator stream lengths differ")
        self.events, self.outcomes = tuple(events), outcomes
        self.tape, self.controllers = tape, tuple(controllers)
        if len({c.name for c in controllers}) != len(controllers):
            raise EngineError("Controller names must be unique")
        self.score_fn, self.memory_state_fn = score_fn, memory_state_fn
        self.write_fn, self.evidence_fn = write_fn, evidence_fn
        self.query_config = query_config
        self.checkpoint_components = dict(checkpoint_components or {})
        if any(not callable(getattr(component, "state_dict", None)) or
               not callable(getattr(component, "load_state_dict", None))
               for component in self.checkpoint_components.values()):
            raise EngineError("Checkpoint components require state_dict/load_state_dict")
        if not provenance_id or not str(provenance_id).strip():
            raise EngineError("Checkpoint provenance_id must be explicit and nonempty")
        self.provenance_id = str(provenance_id)
        verify_tape(tape, [event.record for event in events], query_config)
        labels = bytes(outcomes._labels)
        self._identity = {
            "schema": "calibration.prequential_checkpoint.v1",
            "tape_digest": tape.digest,
            "stream_digest": tape.stream_digest,
            "provenance_id": self.provenance_id,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "n_events": len(events),
            "query_config_sha256": query_config.sha256,
            "outcome_sha256": hashlib.sha256(labels).hexdigest(),
            "controller_names": [c.name for c in self.controllers],
            "component_names": sorted(self.checkpoint_components),
        }
        self._cursor = 0
        self._scores: list[float] = []
        self._predictions: dict[str, list[int | None]] = {
            c.name: [] for c in self.controllers}
        self._prediction_availability: dict[str, list[bool]] = {
            c.name: [] for c in self.controllers}
        self._trace: list[dict[str, Any]] = []
        self._queried_labels: list[tuple[int, int]] = []
        self._last_evidence: EvidenceSnapshot | None = None

    def state_dict(self) -> dict[str, Any]:
        """Capture a private event-boundary checkpoint, including RNG state.

        Evaluator labels are represented only by a digest. They never enter a
        controller checkpoint. Checkpoints may contain image features in named
        private components (for example AnchorLedger) and must not be published.
        """
        return copy.deepcopy({
            "identity": self._identity,
            "cursor": self._cursor,
            "scores": self._scores,
            "predictions": self._predictions,
            "prediction_availability": self._prediction_availability,
            "trace": self._trace,
            "queried_labels": self._queried_labels,
            "last_evidence": self._last_evidence,
            "controllers": {c.name: c.state_dict() for c in self.controllers},
            "components": {name: obj.state_dict()
                           for name, obj in self.checkpoint_components.items()},
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None,
            },
        })

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore only when stream, evaluator, code-facing components match."""
        if state.get("identity") != self._identity:
            raise EngineError("Checkpoint tape/source/evaluator/controller identity mismatch")
        cursor = state.get("cursor")
        if not isinstance(cursor, int) or not 0 <= cursor <= len(self.events):
            raise EngineError("Checkpoint event cursor is invalid")
        scores = list(state.get("scores", ()))
        trace = list(state.get("trace", ()))
        predictions = state.get("predictions", {})
        availability = state.get("prediction_availability", {})
        if len(scores) != cursor or len(trace) != cursor or any(
                len(predictions.get(c.name, ())) != cursor or
                len(availability.get(c.name, ())) != cursor
                for c in self.controllers):
            raise EngineError("Checkpoint traces do not match the event cursor")
        controller_states = state.get("controllers", {})
        component_states = state.get("components", {})
        if set(controller_states) != set(self._identity["controller_names"]):
            raise EngineError("Checkpoint controller set differs")
        if set(component_states) != set(self.checkpoint_components):
            raise EngineError("Checkpoint component set differs")
        for controller in self.controllers:
            loader = getattr(controller, "load_state_dict", None)
            if not callable(loader):
                raise EngineError(f"{controller.name} does not support state restoration")
        for controller in self.controllers:
            controller.load_state_dict(copy.deepcopy(controller_states[controller.name]))
        for name, component in self.checkpoint_components.items():
            component.load_state_dict(copy.deepcopy(component_states[name]))
        self._cursor = cursor
        self._scores = scores
        self._predictions = {name: list(values)
                             for name, values in predictions.items()}
        self._prediction_availability = {
            name: [bool(value) for value in values]
            for name, values in availability.items()}
        self._trace = trace
        self._queried_labels = [tuple(x) for x in state.get("queried_labels", ())]
        self._last_evidence = copy.deepcopy(state.get("last_evidence"))
        rng = state.get("rng", {})
        try:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch_cpu"])
            if rng.get("torch_cuda") is not None:
                if not torch.cuda.is_available():
                    raise EngineError("Checkpoint requires unavailable CUDA RNG state")
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EngineError(f"Checkpoint RNG state is invalid: {exc}") from exc

    def save_checkpoint(self, path: str | Path) -> None:
        """Atomically persist a private checkpoint for local resumable runs."""
        target = Path(path)
        temporary = target.with_name(target.name + ".tmp")
        torch.save(self.state_dict(), temporary)
        os.replace(temporary, target)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a local checkpoint and apply all identity guards in restore."""
        self.load_state_dict(torch.load(Path(path), map_location="cpu"))

    def run(self, *, max_events: int | None = None) -> dict[str, Any]:
        if max_events is not None and max_events < 0:
            raise ValueError("max_events must be nonnegative or None")
        queried = self.tape.positions
        stop = len(self.events) if max_events is None else min(
            len(self.events), self._cursor + max_events)
        for t in range(self._cursor, stop):
            event = self.events[t]
            version_before, digest_before = self.memory_state_fn()
            result = self.score_fn(event)
            if result.memory_version != version_before:
                raise EngineError("Score used a stale or future memory version")
            self._scores.append(float(result.score))
            obs_id = hashlib.sha256(
                f"{self.query_config.version}:{t}:{event.record.camera_id}".encode("utf-8")
            ).hexdigest()[:24]
            observation = Observation(t, obs_id, event.record.camera_id,
                                      result.score, version_before,
                                      result.score_state_digest)
            pre_states, current_preds = {}, {}
            for controller in self.controllers:
                before = canonical_json(controller.state_dict())
                prediction = controller.predict(observation)
                after = canonical_json(controller.state_dict())
                if before != after:
                    raise EngineError(f"{controller.name}.predict mutated controller state")
                current_preds[controller.name] = prediction
                self._predictions[controller.name].append(prediction.label)
                self._prediction_availability[controller.name].append(
                    bool(prediction.available))
                pre_states[controller.name] = controller.state_dict()

            is_query = t in queried
            label = self.outcomes._disclose_to_engine(t, is_query)
            feedback = Feedback(t, is_query, label)
            if is_query:
                self._queried_labels.append((t, int(label)))
            for controller in self.controllers:
                controller.observe_feedback(observation,
                    current_preds[controller.name], feedback)
                controller.observe_score(observation)

            write_audit = self.write_fn(event, result, feedback)
            version_after, digest_after = self.memory_state_fn()
            if version_after < version_before:
                raise EngineError("Memory version regressed")
            if version_after > version_before + 1:
                raise EngineError("A single event advanced more than one memory version")
            if (version_after == version_before) != (digest_after == digest_before):
                raise EngineError("Memory content digest/version transition is inconsistent")
            evidence = None
            if (self.evidence_fn is not None and
                    (self._last_evidence is None or is_query or version_after != version_before)):
                evidence = self.evidence_fn(event if is_query else None,
                                            feedback if is_query else None)
                if evidence.memory_version != version_after:
                    raise EngineError("Evidence ledger returned a stale memory version")
                self._last_evidence = evidence
                for controller in self.controllers:
                    controller.on_memory_change(evidence)
            elif self._last_evidence is not None:
                evidence = self._last_evidence

            post = {c.name: c.finalize_step() for c in self.controllers}
            self._trace.append({
                "t": t,
                "sample_key": event.record.official_sample_key,
                "camera_id": event.record.camera_id,
                "query_event": is_query,
                "label_disclosed": is_query,
                "memory_version_before": version_before,
                "memory_version_after": version_after,
                "memory_sha256_before": digest_before,
                "memory_sha256_after": digest_after,
                "score": float(result.score),
                "score_state_digest": result.score_state_digest,
                "theta_pre": {name: p.threshold for name, p in current_preds.items()},
                "prediction_pre": {name: p.label for name, p in current_preds.items()},
                "prediction_available": {name: p.available
                                         for name, p in current_preds.items()},
                "prediction_reason": {name: p.reason for name, p in current_preds.items()},
                "theta_post": post,
                "evidence_state": evidence.state if evidence else "UNAVAILABLE",
                "write": _jsonable(write_audit),
            })
            self._cursor = t + 1
        if self._cursor < len(self.events):
            return {"status": "INCOMPLETE", "cursor": self._cursor,
                    "n_events": len(self.events), "tape_digest": self.tape.digest,
                    "scores": list(self._scores),
                    "predictions": copy.deepcopy(self._predictions),
                    "prediction_availability": copy.deepcopy(
                        self._prediction_availability),
                    "trace": copy.deepcopy(self._trace),
                    "checkpoint": self.state_dict()}
        metrics = self.outcomes.evaluate(
            self._scores, self._predictions,
            prediction_availability=self._prediction_availability)
        seen_query_classes: set[int] = set()
        first_both = None
        for t, label in self._queried_labels:
            seen_query_classes.add(label)
            if len(seen_query_classes) == 2:
                first_both = t
                break
        query_audit = {
            "n_queried_images": len(self.tape.events),
            "n_queried_groups": len(self.tape.queried_groups),
            "query_fraction_images": len(self.tape.events) / len(self.events),
            "query_fraction_groups": len(self.tape.queried_groups) / self.tape.n_groups,
            "first_queried_normal_t": next((t for t, y in self._queried_labels if y == 0), None),
            "first_queried_defect_t": next((t for t, y in self._queried_labels if y == 1), None),
            "first_both_class_evidence_t": first_both,
            "early_queries_first_400": sum(t < 400 for t, _ in self._queried_labels),
        }
        return {"status": "COMPLETE", "cursor": self._cursor,
                "tape_digest": self.tape.digest,
                "scores": list(self._scores), "predictions": copy.deepcopy(self._predictions),
                "prediction_availability": copy.deepcopy(
                    self._prediction_availability),
                "trace": copy.deepcopy(self._trace), "metrics": metrics,
                "controller_states": {c.name: c.state_dict() for c in self.controllers},
                "query_count": len(self.tape.events), "query_audit": query_audit}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dataclass_fields__"):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return repr(value)
