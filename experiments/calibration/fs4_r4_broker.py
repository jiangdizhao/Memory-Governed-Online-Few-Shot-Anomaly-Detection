"""Separate query-only feedback broker for the local research trust model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .fs4_durable_ledger import LedgerIntegrityError, canonical, digest
from .fs4_r2_security import WindowsPrivateStore
from .fs4_research_ledger import ResearchLedger


def verified_mapping(private_root: Path, execution: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Verify private-to-public identity without returning it to Stage A."""
    store = WindowsPrivateStore(Path(private_root), create=False)
    raw = store.read("mapping.json")
    complete = json.loads(store.read("COMPLETE.json"))
    sha = hashlib.sha256(raw).hexdigest()
    if sha != execution["source_mapping_sha256"] or complete != {"mapping_sha256": sha}:
        raise LedgerIntegrityError("private mapping commitment differs")
    mapping = json.loads(raw)
    if set(mapping) != {"schema", "salt", "events"} or mapping["schema"] != "FS4_R4_PRIVATE_MAPPING_V1":
        raise LedgerIntegrityError("invalid private mapping")
    import re
    if not isinstance(mapping["salt"], str) or re.fullmatch(r"[0-9a-f]{64}", mapping["salt"]) is None:
        raise LedgerIntegrityError("private mapping salt is invalid")
    expected = {item["event"]: (category, item) for category in execution["category_order"]
                for item in execution["categories"][category]["events"]}
    from experiments.risk_safe.preflight import parse_image_name
    result, original_paths, original_events, group_map, reverse_groups = {}, set(), set(), {}, {}
    support_groups = {item["group"] for category in execution["category_order"]
                      for item in execution["categories"][category]["supports"]}
    for row in mapping["events"]:
        event = row.get("event")
        if event in result or event not in expected:
            raise LedgerIntegrityError("private mapping event identity differs")
        category, public = expected[event]
        keys = ("position", "event", "group", "resource", "view", "content_sha256", "resource_sha256",
                "pixel_sha256", "width", "height")
        if (row.get("category") != category or any(row.get(k) != public[k] for k in keys) or
                type(row.get("label")) is not int or row["label"] not in (0, 1)):
            raise LedgerIntegrityError("private/public event mapping differs")
        try:
            original_path = row["original_path"]
            parsed = parse_image_name(original_path, expected_class=category)
            original_group = parsed.official_sample_key
            support_digest = hashlib.sha256(f"official-sample-group\0{original_group}".encode()).hexdigest()
            if (original_path in original_paths or row["original_group"] != original_group or
                    not isinstance(row["original_event"], str) or re.fullmatch(r"[0-9a-f]{64}", row["original_event"]) is None or
                    row["original_event"] in original_events or
                    parsed.camera_id != row["view"] or int(parsed.capture_session != "OK") != row["label"] or
                    support_digest in support_groups or group_map.setdefault(original_group, row["group"]) != row["group"] or
                    reverse_groups.setdefault(row["group"], original_group) != original_group):
                raise ValueError("inconsistent private original identity")
            original_paths.add(original_path)
            original_events.add(row["original_event"])
        except Exception:
            raise LedgerIntegrityError("private original registration differs") from None
        result[event] = row
    if set(result) != set(expected):
        raise LedgerIntegrityError("private mapping is incomplete")
    return result


def _snapshot(run_root: Path) -> ResearchLedger:
    """Replay the durable hash chain read-only, independently of the writer."""
    from .fs4_r4_blinded import validate_blinded_execution
    store = WindowsPrivateStore(run_root, create=False)
    manifest = json.loads(store.read("manifest.json"))
    execution = validate_blinded_execution(manifest["research_execution_manifest"], manifest)
    obj = object.__new__(ResearchLedger)
    obj.manifest = manifest; obj.execution_manifest = execution; obj._execution_manifest = execution
    obj._reading_existing = True; obj._active_provenance = None; obj._trusted_context = None
    obj.records = []; obj.last_hash = digest(manifest); obj.phase = "READY"; obj.position = 0
    obj.event_ids = set(); obj.group_counts = {}; obj.category_counts = {c: 0 for c in manifest["expected_events_by_category"]}
    obj.category_streams = {c: [] for c in obj.category_counts}; obj.category_seals = {}; obj.current_category = None
    obj.query_price = .8; obj.query_history = []; obj.query_flags = []; obj.prediction = None; obj.query = None
    data = store.read("events.jsonl")
    if data and not data.endswith(b"\n"):
        raise LedgerIntegrityError("incomplete durable ledger")
    for line in data.splitlines():
        row = json.loads(line)
        body = {key: row[key] for key in ("index", "previous", "kind", "payload")}
        if (canonical(row) != line or row["index"] != len(obj.records) or row["previous"] != obj.last_hash or
                row["sha256"] != digest(body)):
            raise LedgerIntegrityError("durable ledger chain differs")
        obj._apply(row["kind"], row["payload"])
        obj.records.append(row); obj.last_hash = row["sha256"]
    return obj


class QueryOnlyBroker:
    """Private process: releases a label once after an independently checked intent."""

    def __init__(self, *, private_root: Path, run_root: Path, disclosures_root: Path) -> None:
        self.private_root = Path(private_root).absolute()
        self.run_root = Path(run_root).absolute()
        self.disclosures_root = Path(disclosures_root).absolute()
        if any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for a, b in
               ((self.private_root, self.run_root), (self.private_root, self.disclosures_root),
                (self.run_root, self.disclosures_root))):
            raise LedgerIntegrityError("broker stores must be separate")
        self.disclosures = WindowsPrivateStore(self.disclosures_root, create=not self.disclosures_root.exists())

    def disclose(self, *, event: str, group: str, position: int, query_sha256: str) -> int:
        import re
        if (type(position) is not int or position < 0 or any(not isinstance(value, str) or
                re.fullmatch(r"[0-9a-f]{64}", value) is None for value in (event, group, query_sha256))):
            raise LedgerIntegrityError("invalid broker request identity")
        journal = _snapshot(self.run_root)
        if journal.phase != "LABEL_INTENT" or not journal.records or journal.records[-1]["kind"] != "LABEL_INTENT":
            raise LedgerIntegrityError("broker requires a durable executed-query label intent")
        prediction, query = journal.prediction, journal.query
        if (prediction is None or query is None or not query["executed"] or prediction["event"] != event or
                prediction["group"] != group or prediction["position"] != position or journal.position != position or
                len(journal.records) < 2 or journal.records[-2]["kind"] != "QUERY" or
                journal.records[-2]["payload"]["query_sha256"] != query_sha256 or digest(query) != query_sha256):
            raise LedgerIntegrityError("broker request does not match the sealed query")
        mapping = verified_mapping(self.private_root, journal.execution_manifest)
        row = mapping.get(event)
        if row is None or row["group"] != group or row["position"] != position:
            raise LedgerIntegrityError("broker request does not match the registered event")
        # Create-before-release makes a duplicate or ambiguous crash fail closed.
        proof = {"event": event, "group": group, "position": position, "query_sha256": query_sha256}
        try:
            self.disclosures.write(event + ".json", canonical(proof) + b"\n")
        except FileExistsError:
            raise LedgerIntegrityError("broker refuses duplicate disclosure") from None
        return row["label"]


def broker_main(connection: Any, *, private_root: str, run_root: str, disclosures_root: str, manifest_path: str) -> None:
    """Serve a narrow JSON protocol; errors contain no private path or label."""
    try:
        from .fs4_r4_blinded import validate_blinded_execution
        if Path(manifest_path).absolute().is_relative_to(Path(private_root).absolute()):
            raise LedgerIntegrityError("public manifest must be outside private mapping storage")
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        execution = validate_blinded_execution(manifest["research_execution_manifest"], manifest)
        verified_mapping(Path(private_root), execution)
        broker = QueryOnlyBroker(private_root=Path(private_root), run_root=Path(run_root),
                                 disclosures_root=Path(disclosures_root))
        connection.send_bytes(canonical({"status": "READY", "manifest_sha256": digest(manifest)}))
        while True:
            request = json.loads(connection.recv_bytes(4096))
            if request == {"command": "close"}:
                connection.send_bytes(canonical({"status": "CLOSED"})); break
            if set(request) != {"event", "group", "position", "query_sha256"}:
                raise LedgerIntegrityError("invalid broker request")
            value = broker.disclose(**request)
            connection.send_bytes(canonical({"status": "OK", "label": value}))
    except Exception:
        try: connection.send_bytes(canonical({"status": "ERROR", "message": "broker rejected disclosure"}))
        except Exception: pass
    finally:
        connection.close()
