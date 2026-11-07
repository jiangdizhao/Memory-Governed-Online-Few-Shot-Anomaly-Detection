"""Bind an FS4 trajectory to durable prediction, feedback and transition writes."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Any, Callable

from .fs4_checkpoint import RecoveryLedger
from .fs4_durable_ledger import DurableLedger, digest
from .fs4_stochastic_bea import receipt_digest


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
                "query_seal": e.receipt.query_seal, "receipt_sha256": receipt_digest(e.receipt),
                "pre_scores": dict(e.pre_scores), "scorer_states": dict(e.scorer_states),
                "feature_handle": e.feature_handle} for e in c.evidence]}
        if hasattr(c, "cache"):
            items[name]["cache"] = {group: record.feature_sha256 for group, record in c.cache.records.items()}
            items[name]["successful_tokens"] = c.successful_tokens
    p = trajectory.policy
    state = {"profile": trajectory.mode, "native": trajectory.runtime.states(),
             "normal_version": trajectory.runtime.normal_version, "correction_version": trajectory.runtime.correction_version,
             "query": {"seed": p.seed, "start_position": p.start_position, "price": p.price, "groups": p.group_counts, "candidates": p.candidates,
                       "history": list(p.history), "flags": list(p.flags), "last_position": p.last_position},
             "coverage": {name: _array_digest(getattr(trajectory.coverage, name))
                          for name in ("support", "support_distances", "dynamic", "distances")},
             "controllers": items}
    return digest(state)


class _DurableCore:
    """Internal ledger adapter; input authority is supplied by its wrapper."""
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
        audit_snapshot = getattr(self.trajectory, "audit_snapshot", None)
        if callable(audit_snapshot):
            transition["audit"] = audit_snapshot()
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


class SyntheticTestExecutor(_DurableCore):
    """Synthetic-only adapter accepting test doubles and test feature tensors."""
    def __init__(self, trajectory: Any, ledger: Any, label_reader: Any = None) -> None:
        underlying = ledger.ledger if isinstance(ledger, RecoveryLedger) else ledger
        if isinstance(underlying, DurableLedger):
            raise ValueError("synthetic executor cannot write a production ledger")
        super().__init__(trajectory, ledger, label_reader)


class DurableExecutor:
    """Production executor. Callers provide only an already registered position."""
    def __init__(self, *, context: Any, ledger: DurableLedger,
                 label_reader: Callable[[str, int], int], checkpoint_name: str | None = None) -> None:
        from .fs4_r2_context import ProductionContext
        from .fs4_r2_security import WindowsPrivateStore, validate_private_acl
        from .fs4_checkpoint import begin_recovery
        if type(context) is not ProductionContext or type(ledger) is not DurableLedger or context._ledger is not ledger:
            raise ValueError("production executor requires its own trusted context and production ledger")
        if not callable(label_reader) or context._trajectory.policy.seed != ledger.manifest["query_seed"]:
            raise ValueError("query reader or registered query seed is invalid")
        if context._trajectory.runtime is not context._runtime or context._category != ledger.current_category:
            raise ValueError("production executor received an unrelated native runtime")
        ledger._bind_context(context)
        self._context, self._ledger = context, ledger
        if checkpoint_name is None:
            if ledger.position != context._trajectory.policy.start_position or ledger.phase != "READY":
                raise ValueError("advanced production ledger requires trusted checkpoint recovery")
            selected_ledger: DurableLedger | RecoveryLedger = ledger
        else:
            checkpoints = WindowsPrivateStore(ledger.root / "checkpoints", create=False)
            checkpoint = checkpoints._child(checkpoint_name, must_exist=True)
            validate_private_acl(checkpoint, directory=True)
            for name in ("state.json", "tensors.npz", "COMMITTED.json"):
                validate_private_acl(checkpoint / name, directory=False)
            selected_ledger = begin_recovery(checkpoint, trajectory=context._trajectory, ledger=ledger,
                                             label_reader=label_reader, trusted_context=context)
        self._adapter = _DurableCore(context._trajectory, selected_ledger, label_reader)

    def step(self, position: int) -> dict[str, Any]:
        """Encode the signed event from the bound dataset; no tensors are accepted."""
        event = self._context._encode_event(position)
        self._ledger._active_provenance = dict(event["event_provenance"])
        try:
            return self._adapter.step(**event)
        finally:
            self._ledger._active_provenance = None

    def checkpoint(self, name: str) -> dict[str, Any]:
        from .fs4_checkpoint import save_checkpoint
        from .fs4_r2_security import WindowsPrivateStore
        store_path = self._ledger.root / "checkpoints"
        store = WindowsPrivateStore(store_path, create=not store_path.exists())
        path = store._child(name, must_exist=False)
        selected = self._adapter.ledger
        if isinstance(selected, RecoveryLedger):
            selected.assert_caught_up()
        return save_checkpoint(path, trajectory=self._context._trajectory, ledger=self._ledger, private_store=store)

    def seal_category(self) -> str:
        return self._adapter.seal_category(self._context._category)
