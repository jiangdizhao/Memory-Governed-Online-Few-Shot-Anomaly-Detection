"""Single-trajectory, predict-before-disclosure prequential engine."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Any, Callable, Sequence

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
                 query_config: QueryProtocolConfig = QueryProtocolConfig()):
        if len(events) != len(outcomes):
            raise EngineError("Event/evaluator stream lengths differ")
        self.events, self.outcomes = tuple(events), outcomes
        self.tape, self.controllers = tape, tuple(controllers)
        if len({c.name for c in controllers}) != len(controllers):
            raise EngineError("Controller names must be unique")
        self.score_fn, self.memory_state_fn = score_fn, memory_state_fn
        self.write_fn, self.evidence_fn = write_fn, evidence_fn
        self.query_config = query_config
        verify_tape(tape, [event.record for event in events], query_config)

    def run(self) -> dict[str, Any]:
        scores: list[float] = []
        preds: dict[str, list[int | None]] = {c.name: [] for c in self.controllers}
        trace: list[dict[str, Any]] = []
        queried_labels: list[tuple[int, int]] = []
        last_evidence: EvidenceSnapshot | None = None
        queried = self.tape.positions
        for t, event in enumerate(self.events):
            version_before, digest_before = self.memory_state_fn()
            result = self.score_fn(event)
            if result.memory_version != version_before:
                raise EngineError("Score used a stale or future memory version")
            scores.append(float(result.score))
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
                preds[controller.name].append(prediction.label)
                pre_states[controller.name] = controller.state_dict()

            is_query = t in queried
            label = self.outcomes._disclose_to_engine(t, is_query)
            feedback = Feedback(t, is_query, label)
            if is_query:
                queried_labels.append((t, int(label)))
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
                    (last_evidence is None or is_query or version_after != version_before)):
                evidence = self.evidence_fn(event if is_query else None,
                                            feedback if is_query else None)
                if evidence.memory_version != version_after:
                    raise EngineError("Evidence ledger returned a stale memory version")
                last_evidence = evidence
                for controller in self.controllers:
                    controller.on_memory_change(evidence)
            elif last_evidence is not None:
                evidence = last_evidence

            post = {c.name: c.finalize_step() for c in self.controllers}
            trace.append({
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
                "prediction_reason": {name: p.reason for name, p in current_preds.items()},
                "theta_post": post,
                "evidence_state": evidence.state if evidence else "UNAVAILABLE",
                "write": _jsonable(write_audit),
            })
        metrics = self.outcomes.evaluate(scores, preds)
        seen_query_classes: set[int] = set()
        first_both = None
        for t, label in queried_labels:
            seen_query_classes.add(label)
            if len(seen_query_classes) == 2:
                first_both = t
                break
        query_audit = {
            "n_queried_images": len(self.tape.events),
            "n_queried_groups": len(self.tape.queried_groups),
            "query_fraction_images": len(self.tape.events) / len(self.events),
            "query_fraction_groups": len(self.tape.queried_groups) / self.tape.n_groups,
            "first_queried_normal_t": next((t for t, y in queried_labels if y == 0), None),
            "first_queried_defect_t": next((t for t, y in queried_labels if y == 1), None),
            "first_both_class_evidence_t": first_both,
            "early_queries_first_400": sum(t < 400 for t, _ in queried_labels),
        }
        return {"tape_digest": self.tape.digest,
                "scores": scores, "predictions": preds,
                "trace": trace, "metrics": metrics,
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
