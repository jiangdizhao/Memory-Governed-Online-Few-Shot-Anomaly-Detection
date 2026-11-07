"""Bind an FS4 trajectory to durable prediction, feedback and transition writes."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Any, Callable

from .fs4_checkpoint import RecoveryLedger
from .fs4_durable_ledger import DurableLedger, digest


def _array_digest(value: Any) -> str:
    x = value.copy(order="C")
    h = hashlib.sha256(f"{x.dtype}|{x.shape}".encode())
    h.update(x.tobytes())
    return h.hexdigest()


def post_state_digest(trajectory: Any) -> str:
    """Bind all decision-relevant components, not just native tensor state."""
    controllers = ({"L": trajectory.L, "E": trajectory.E} if hasattr(trajectory, "L") else
                   {trajectory.kind: trajectory.controller})
    items = {}
    for name, c in controllers.items():
        items[name] = {"theta": c.engine.theta, "tokens": c.engine.tokens, "revision": c.revision,
            "status": c.engine.status, "evidence": [{"position": e.position, "group": e.group,
                "event": e.event, "normal_version": e.normal_version, "correction_version": e.correction_version,
                "query_seal": e.receipt.query_seal} for e in c.evidence]}
        if hasattr(c, "cache"):
            items[name]["cache"] = {group: record.feature_sha256 for group, record in c.cache.records.items()}
            items[name]["successful_tokens"] = c.successful_tokens
    p = trajectory.policy
    state = {"profile": trajectory.mode, "native": trajectory.runtime.states(),
             "normal_version": trajectory.runtime.normal_version, "correction_version": trajectory.runtime.correction_version,
             "query": {"seed": p.seed, "price": p.price, "groups": p.group_counts, "candidates": p.candidates,
                       "history": list(p.history), "flags": list(p.flags), "last_position": p.last_position},
             "coverage": {name: _array_digest(getattr(trajectory.coverage, name))
                          for name in ("support", "support_distances", "dynamic", "distances")},
             "controllers": items}
    return digest(state)


class DurableExecutor:
    """Only this adapter wires actual fsync ledger methods into a trajectory."""
    def __init__(self, trajectory: Any, ledger: DurableLedger | RecoveryLedger,
                 label_reader: Callable[[str, int], int] | None = None) -> None:
        self.trajectory, self.ledger = trajectory, ledger
        trajectory.seal_prediction = ledger.seal_prediction
        trajectory.seal_query = ledger.seal_query
        if isinstance(ledger, RecoveryLedger):
            trajectory.label_provider = ledger.disclose
        elif label_reader is not None:
            trajectory.label_provider = lambda event, position: ledger.disclose(event, position, label_reader)
        else:
            raise ValueError("an explicit query-only label reader is required")

    def step(self, **event: Any) -> dict[str, Any]:
        row = self.trajectory.step(**event)
        transition = {"profile": row["mode"], "query": asdict(row["query"]),
                      "feedback_update": row["feedback_update"], "next_thresholds": row["next_thresholds"]}
        self.ledger.complete_event(position=event["position"], post_state_digest=post_state_digest(self.trajectory),
                                   transition=transition)
        return row

    def seal_category(self, category: str) -> str:
        if isinstance(self.ledger, RecoveryLedger):
            self.ledger.assert_caught_up()
            underlying = self.ledger.ledger
        else:
            underlying = self.ledger
        self.trajectory.policy.validate_complete_stream()
        return underlying.seal_category(category)
