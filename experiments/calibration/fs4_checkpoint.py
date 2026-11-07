"""Explicit, checksummed FS4 event-boundary checkpoint; no object-graph pickle.

A committed checkpoint is immutable and never overwrites a prior one. Advanced
ledgers are accepted only by the explicit checked recovery API.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .fs4_durable_ledger import canonical, digest
from .fs4_greedy_provenance import _native_modules
from .fs4_stochastic_bea import QueryReceipt, SealedPrediction, TARGET_RATE, receipt_digest
from .fs4_threshold_controllers import SelectionAwareSourceExcludedThreshold
from .fs4_threshold_core import NormalEvidence


class CheckpointError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _controllers(trajectory: Any) -> dict[str, Any]:
    if hasattr(trajectory, "L") and hasattr(trajectory, "E"):
        return {"L": trajectory.L, "E": trajectory.E}
    return {trajectory.kind: trajectory.controller}


def _tokens(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _validate_c1(runtime: Any, next_position: int) -> None:
    from experiments.calibration import r1_matched_correction_pilot as r1
    cb = runtime.correction_cb
    try:
        r1._validate_c1_source_groups(runtime.sources, cb.last_step[:cb.size()].detach().cpu().tolist(),
                                      event_position=next_position, phase="pre_score", event_group="0" * 64)
    except Exception as exc:
        raise CheckpointError("invalid C1 provenance at checkpoint boundary") from exc


def save_checkpoint(path: Path, *, trajectory: Any, ledger: Any) -> dict[str, Any]:
    """Save a completely applied event, bound to the exact durable ledger offset."""
    path = Path(path)
    offset = ledger.offset()
    if offset["phase"] != "READY" or offset["next_position"] != trajectory.policy.last_position + 1 or getattr(trajectory, "failed", False):
        raise CheckpointError("checkpoint requires a complete ledger/trajectory event boundary")
    runtime = trajectory.runtime
    if not all(hasattr(runtime, n) for n in ("bank", "correction_cb", "provenance", "writer", "fixed_sha")):
        raise CheckpointError("checkpoint requires actual native FS4 runtime")
    states = runtime.states()
    _validate_c1(runtime, offset["next_position"])
    bridge, _ = _native_modules()
    arrays: dict[str, np.ndarray] = {}
    def tensor(name: str, value: torch.Tensor) -> None:
        arrays[name] = value.detach().to("cpu").contiguous().numpy().copy()
    dynamic_names = []
    for i, book in enumerate(bridge.dynamic_codebooks(runtime.bank)):
        names = []
        for name, value in vars(book).items():
            if isinstance(value, torch.Tensor):
                tensor(f"dyn_{i}_{name}", value); names.append(name)
        dynamic_names.append(names)
    correction_names = []
    for name, value in vars(runtime.correction_cb).items():
        if isinstance(value, torch.Tensor):
            tensor(f"c1_{name}", value); correction_names.append(name)
    for name in ("support", "support_distances", "dynamic", "distances"):
        arrays[f"coverage_{name}"] = np.asarray(getattr(trajectory.coverage, name)).copy()
    cdata: dict[str, Any] = {}
    for kind, controller in _controllers(trajectory).items():
        if getattr(controller, "fatal_error", None) is not None:
            raise CheckpointError("cannot checkpoint a failed controller")
        e = controller.engine
        data: dict[str, Any] = {"theta": e.theta, "tokens": e.tokens, "status": e.status,
            "suppression": e.suppression, "revision": controller.revision,
            "evidence": [{"position": item.position, "normal_version": item.normal_version,
                          "correction_version": item.correction_version, "feature_handle": item.feature_handle}
                         for item in controller.evidence]}
        if isinstance(controller, SelectionAwareSourceExcludedThreshold):
            data.update({"attempt_tokens": controller.attempt_tokens, "successful_tokens": controller.successful_tokens,
                         "current_scores": controller.current_scores, "rescore_calls": dict(controller.rescore_calls),
                         "rescore_invocations": controller.rescore_invocations, "rescore_seconds": controller.rescore_seconds,
                         "cache": []})
            for i, item in enumerate(controller.evidence):
                record = controller.cache.records[item.group]
                tensor(f"cache_{kind}_{i}", record.features)
                data["cache"].append({"group": item.group, "height": record.height, "width": record.width,
                                      "normalization": record.normalization, "sha256": record.feature_sha256})
        cdata[kind] = data
    policy = trajectory.policy
    state = {"schema": "FS4_CHECKPOINT_BOUNDARY_V1", "manifest_sha256": digest(ledger.manifest),
        "ledger_offset": offset, "profile": trajectory.mode, "fixed_sha": runtime.fixed_sha,
        "writer_reference": asdict(runtime._writer_reference), "scorer_states": states,
        "normal_version": runtime.normal_version, "correction_version": runtime.correction_version,
        "dynamic_tensor_names": dynamic_names, "correction_tensor_names": correction_names,
        "correction_n": runtime.correction_cb.n, "correction_counters": runtime.correction_cb.counters,
        "correction_sources": {str(k): sorted(v) for k, v in runtime.sources.items()},
        "greedy_rows": runtime.provenance.rows, "greedy_digest": runtime.provenance.memory_digest,
        "greedy_revision": runtime.provenance.revision, "greedy_valid": runtime.provenance.valid,
        "writer": {name: getattr(runtime.writer, name) for name in
                   ("events", "patches", "promotions", "accepted", "skipped", "update_seconds")},
        "policy": {"seed": policy.seed, "start_position": policy.start_position, "price": policy.price, "group_counts": policy.group_counts,
                   "candidates": policy.candidates, "history": list(policy.history), "flags": list(policy.flags),
                   "receipts": [asdict(policy.receipts[p]) for p in sorted(policy.receipts)],
                   "events": sorted(policy.events), "last_position": policy.last_position},
        "controllers": cdata}
    path.mkdir(parents=False, exist_ok=False)
    tensor_path = path / "tensors.npz"
    with tensor_path.open("xb") as stream:
        np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    state["tensor_sha256"] = _sha(tensor_path)
    state_path = path / "state.json"
    with state_path.open("xb") as stream:
        stream.write(canonical(state) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    marker = {"state_sha256": _sha(state_path), "tensor_sha256": state["tensor_sha256"]}
    with (path / "COMMITTED.json").open("xb") as stream:
        stream.write(canonical(marker) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    return {**marker, "persistent_bytes": sum(x.stat().st_size for x in (tensor_path, state_path, path / "COMMITTED.json")),
            "ledger_offset": offset}


def _verify_prefix(saved: dict[str, Any], ledger: Any) -> None:
    count = saved["records"]
    if not isinstance(count, int) or count < 0 or count > len(ledger.records):
        raise CheckpointError("checkpoint ledger record count mismatch")
    prefix = ledger.records[:count]
    prefix_hash = prefix[-1]["sha256"] if prefix else digest(ledger.manifest)
    prefix_bytes = sum(len(canonical(row)) + 1 for row in prefix)
    if saved["phase"] != "READY" or prefix_hash != saved["sha256"] or prefix_bytes != saved["bytes"]:
        raise CheckpointError("checkpoint ledger prefix mismatch")


def _validate_saved_policy(saved: dict[str, Any], ledger: Any, record_count: int) -> None:
    """Bind restored receipts and adaptive policy history to durable records."""
    rows = ledger.records[:record_count]
    start = saved["start_position"]
    queries = [row["payload"]["receipt"] for row in rows if row["kind"] == "QUERY" and
               row["payload"]["receipt"]["prediction"]["position"] >= start]
    feedback = {row["payload"]["position"]: row["payload"]["label"] for row in rows if row["kind"] == "FEEDBACK"}
    if len(queries) != len(saved["receipts"]) or len(queries) != saved["last_position"] - start + 1:
        raise CheckpointError("checkpoint query receipt count differs from ledger")
    flags: deque[int] = deque(maxlen=256)
    history: deque[tuple[float, str]] = deque(maxlen=256)
    group_counts: dict[str, int] = {}; candidates: dict[str, int] = {}; events = set(); price = .8
    for pos, (query, row) in enumerate(zip(queries, saved["receipts"]), start=start):
        p = query["prediction"]
        expected = dict(row); expected.update({"query_seal": None, "disclosed": False, "label": None})
        base = dict(query); base["prediction"] = SealedPrediction(**base["prediction"])
        expected_seal = receipt_digest(QueryReceipt(**base))
        if (query != expected or p["position"] != pos or query["seed"] != saved["seed"] or
                row["query_seal"] != expected_seal or query["cost"] != price or
                row["disclosed"] != query["executed"] or
                row["label"] != (feedback.get(pos) if query["executed"] else None) or
                (query["executed"] and pos not in feedback)):
            raise CheckpointError("checkpoint receipt does not match durable legal query/feedback")
        group_counts[p["group"]] = query["ordinal"]; candidates[p["group"]] = query["candidate"]
        events.add(p["event"]); history.append((p["score_gc"], p["group"]))
        flags.append(int(query["executed"]))
        price = min(3.0, max(.05, price + .02 * (sum(flags) / len(flags) - TARGET_RATE)))
    if (saved["group_counts"] != group_counts or saved["candidates"] != candidates or
            set(saved["events"]) != events or [tuple(x) for x in saved["history"]] != list(history) or
            saved["flags"] != list(flags) or saved["price"] != price):
        raise CheckpointError("checkpoint adaptive query history differs from ledger")


def restore_checkpoint(path: Path, *, trajectory: Any, ledger: Any, allow_advanced: bool = False) -> dict[str, Any]:
    """Restore into a fresh native trajectory at an identical durable boundary."""
    path = Path(path)
    try:
        marker = json.loads((path / "COMMITTED.json").read_text(encoding="utf-8"))
        state_path, tensor_path = path / "state.json", path / "tensors.npz"
        if _sha(state_path) != marker["state_sha256"] or _sha(tensor_path) != marker["tensor_sha256"]:
            raise CheckpointError("checkpoint checksum mismatch")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (state["schema"] != "FS4_CHECKPOINT_BOUNDARY_V1" or state["manifest_sha256"] != digest(ledger.manifest) or
                state["tensor_sha256"] != marker["tensor_sha256"] or
                state["profile"] != trajectory.mode):
            raise CheckpointError("checkpoint source, profile or ledger boundary mismatch")
        _verify_prefix(state["ledger_offset"], ledger)
        _validate_saved_policy(state["policy"], ledger, state["ledger_offset"]["records"])
        if not allow_advanced and state["ledger_offset"] != ledger.offset():
            raise CheckpointError("advanced ledger requires explicit replay recovery")
        runtime, policy = trajectory.runtime, trajectory.policy
        if runtime.fixed_sha != state["fixed_sha"] or asdict(runtime._writer_reference) != state["writer_reference"]:
            raise CheckpointError("fixed Support bank or learning reference mismatch")
        if runtime.normal_version or runtime.correction_version or policy.last_position != policy.start_position - 1:
            raise CheckpointError("restore target must be fresh")
        if runtime.states() is None:
            raise CheckpointError("invalid native runtime")
        bridge, _ = _native_modules()
        with np.load(tensor_path, allow_pickle=False) as arrays:
            if (not np.array_equal(trajectory.coverage.support, arrays["coverage_support"]) or
                    not np.array_equal(trajectory.coverage.support_distances, arrays["coverage_support_distances"])):
                raise CheckpointError("frozen Support coverage geometry mismatch")
            for i, book in enumerate(bridge.dynamic_codebooks(runtime.bank)):
                for name in state["dynamic_tensor_names"][i]:
                    previous = getattr(book, name)
                    setattr(book, name, torch.from_numpy(arrays[f"dyn_{i}_{name}"].copy()).to(previous.device, dtype=previous.dtype))
            cb = runtime.correction_cb
            for name in state["correction_tensor_names"]:
                previous = getattr(cb, name)
                restored = torch.from_numpy(arrays[f"c1_{name}"].copy()).to(previous.device, dtype=previous.dtype)
                if restored.shape != previous.shape:
                    raise CheckpointError("correction tensor geometry mismatch")
                setattr(cb, name, restored)
            cb.n = int(state["correction_n"]); cb.counters = dict(state["correction_counters"])
            runtime.sources.clear(); runtime.sources.update({int(k): set(v) for k, v in state["correction_sources"].items()})
            runtime.provenance.rows = state["greedy_rows"]
            runtime.provenance.memory_digest = state["greedy_digest"]
            runtime.provenance.revision = state["greedy_revision"]; runtime.provenance.valid = state["greedy_valid"]
            runtime.normal_version = state["normal_version"]; runtime.correction_version = state["correction_version"]
            for name, value in state["writer"].items(): setattr(runtime.writer, name, value)
            for name in ("support", "support_distances", "dynamic", "distances"):
                setattr(trajectory.coverage, name, arrays[f"coverage_{name}"].copy())
            p = state["policy"]
            if policy.seed != p["seed"] or policy.start_position != p["start_position"]:
                raise CheckpointError("query seed or category start position mismatch")
            policy.price = p["price"]; policy.group_counts = p["group_counts"]; policy.candidates = p["candidates"]
            policy.history = deque((tuple(x) for x in p["history"]), maxlen=256)
            policy.flags = deque(p["flags"], maxlen=256); policy.events = set(p["events"])
            policy.last_position = p["last_position"]; policy.receipts = {}
            for row in p["receipts"]:
                row = dict(row); row["prediction"] = SealedPrediction(**row["prediction"])
                receipt = QueryReceipt(**row); policy.receipts[receipt.prediction.position] = receipt
            for kind, controller in _controllers(trajectory).items():
                data = state["controllers"][kind]; engine = controller.engine
                engine.theta = data["theta"]; engine.tokens = {k: _tokens(v) for k, v in data["tokens"].items()}
                engine.status = data["status"]; engine.suppression = data["suppression"]
                controller.revision = data["revision"]; controller.evidence = []
                for saved in data["evidence"]:
                    receipt = policy.receipts[saved["position"]]
                    controller.evidence.append(NormalEvidence.from_disclosure(policy, receipt,
                        normal_version=saved["normal_version"], correction_version=saved["correction_version"],
                        feature_handle=saved["feature_handle"]))
                if isinstance(controller, SelectionAwareSourceExcludedThreshold):
                    for i, saved in enumerate(data["cache"]):
                        item = controller.evidence[i]
                        if saved["group"] != item.group:
                            raise CheckpointError("cached feature/source association mismatch")
                        record = controller.cache.add(item, torch.from_numpy(arrays[f"cache_{kind}_{i}"].copy()),
                            height=saved["height"], width=saved["width"], normalization=saved["normalization"])
                        if record.feature_sha256 != saved["sha256"]:
                            raise CheckpointError("cached feature checksum mismatch")
                    controller.attempt_tokens = {k: _tokens(v) for k, v in data["attempt_tokens"].items()}
                    controller.successful_tokens = {k: _tokens(v) for k, v in data["successful_tokens"].items()}
                    controller.current_scores = data["current_scores"]; controller.rescore_calls = Counter(data["rescore_calls"])
                    controller.rescore_invocations = data["rescore_invocations"]; controller.rescore_seconds = data["rescore_seconds"]
                else:
                    controller.groups = {item.group for item in controller.evidence}
        _validate_c1(runtime, policy.last_position + 1)
        if runtime.states() != state["scorer_states"] or policy.last_position + 1 != state["ledger_offset"]["next_position"]:
            raise CheckpointError("restored native state digest or position mismatch")
        return state["ledger_offset"]
    except Exception:
        trajectory.failed = True
        raise


class RecoveryLedger:
    """Replay recorded legal actions, appending only after the existing tail.

    The input provider must supply the exact originally encoded event again.
    Prediction and query records and completed post-state digests are compared
    byte-for-byte. A recorded feedback value is replayed without a second read;
    an intent lacking a recorded value is ambiguous and permanently fails closed.
    """
    def __init__(self, ledger: Any, offset: dict[str, Any], reader: Any) -> None:
        _verify_prefix(offset, ledger)
        self.ledger, self.reader = ledger, reader
        self.pending = list(ledger.records[offset["records"]:])
        self.index = 0

    def _peek(self) -> dict[str, Any] | None:
        return self.pending[self.index] if self.index < len(self.pending) else None

    def _take(self, kind: str, payload: dict[str, Any]) -> bool:
        record = self._peek()
        if record is None:
            return False
        if record["kind"] != kind or canonical(record["payload"]) != canonical(payload):
            raise CheckpointError(f"replay differs from durable {kind} record")
        self.index += 1
        return True

    def seal_prediction(self, event: dict[str, Any]) -> str:
        from .fs4_threshold_integration import prediction_digest
        value = prediction_digest(event)
        if not self._take("PREDICTION", {"event": event, "prediction_sha256": value}):
            return self.ledger.seal_prediction(event)
        return value

    def seal_query(self, receipt: Any) -> str:
        from .fs4_stochastic_bea import receipt_digest
        value = receipt_digest(receipt)
        if not self._take("QUERY", {"receipt": asdict(receipt), "query_sha256": value}):
            return self.ledger.seal_query(receipt)
        return value

    def disclose(self, event: str, position: int) -> int:
        record = self._peek()
        if record is None:
            return self.ledger.disclose(event, position, self.reader)
        if not self._take("LABEL_INTENT", {"position": position}):
            raise CheckpointError("missing durable label intent")
        feedback = self._peek()
        if feedback is None:
            raise CheckpointError("ambiguous prior label disclosure; recovery is forbidden")
        if feedback["kind"] != "FEEDBACK" or feedback["payload"].get("position") != position:
            raise CheckpointError("invalid durable feedback during replay")
        label = feedback["payload"].get("label")
        if label not in (0, 1) or isinstance(label, bool):
            raise CheckpointError("invalid durable feedback label")
        self.index += 1
        return label

    def complete_event(self, *, position: int, post_state_digest: str, transition: dict[str, Any]) -> None:
        payload = {"position": position, "post_state_digest": post_state_digest, "transition": transition}
        if self._take("TRANSITION", payload):
            if not self._take("EVENT_COMPLETE", {"position": position}):
                self.ledger._append("EVENT_COMPLETE", {"position": position})
        else:
            self.ledger.complete_event(position=position, post_state_digest=post_state_digest, transition=transition)

    def assert_caught_up(self) -> None:
        if self._peek() is not None:
            raise CheckpointError("durable tail has not been fully replayed")


def begin_recovery(path: Path, *, trajectory: Any, ledger: Any, label_reader: Any) -> RecoveryLedger:
    offset = restore_checkpoint(path, trajectory=trajectory, ledger=ledger, allow_advanced=True)
    recovery = RecoveryLedger(ledger, offset, label_reader)
    trajectory.seal_prediction = recovery.seal_prediction
    trajectory.seal_query = recovery.seal_query
    trajectory.label_provider = recovery.disclose
    return recovery
