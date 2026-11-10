"""Non-overwriting fsync group journal for outcome-free Stage A records."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .query import PRINCIPAL_ARMS


class SinkProtocolError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


class FsyncGroupSink:
    """Append one complete five-view envelope and fsync before label disclosure."""

    def __init__(self, journal_path: str | Path, *, identity: Mapping[str, Any]):
        self.journal_path = Path(journal_path)
        self.seal_path = self.journal_path.with_name("stage_a_category_seal.json")
        if not self.journal_path.parent.is_dir() or self.journal_path.exists() or self.seal_path.exists():
            raise SinkProtocolError("Stage A sink requires an existing parent and fresh non-overwriting files")
        payload = {"schema": "final-benchmark.stage-a-journal-header.v1", **dict(identity)}
        self._reject_outcomes(payload)
        self._stream = self.journal_path.open("xb")
        self._write_payload(payload)
        self.group_count = 0
        self.event_count = 0
        self.closed = False
        self._sealed_queries: dict[tuple[str, int], Mapping[str, Any]] = {}
        self._visa_units: set[str] = set()

    @staticmethod
    def _reject_outcomes(value: Any) -> None:
        forbidden = {"label", "anomaly_class", "mask", "mask_path", "ground_truth", "gt"}
        if isinstance(value, Mapping):
            if forbidden.intersection(map(str.lower, map(str, value))):
                raise SinkProtocolError("Stage A journal rejected an outcome-bearing field")
            for child in value.values():
                FsyncGroupSink._reject_outcomes(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                FsyncGroupSink._reject_outcomes(child)

    def _write_payload(self, value: Any) -> str:
        payload = _canonical(value) + b"\n"
        self._stream.write(payload)
        self._stream.flush()
        os.fsync(self._stream.fileno())
        return hashlib.sha256(payload).hexdigest()

    def append_group(self, *, group_seal: Mapping[str, Any],
                     score_rows: Sequence[Mapping[str, Any]]) -> str:
        if self.closed or len(score_rows) != 5:
            raise SinkProtocolError("Stage A group journal is closed or group is incomplete")
        self._reject_outcomes(group_seal)
        self._reject_outcomes(score_rows)
        if hashlib.sha256(_canonical(list(score_rows))).hexdigest() != group_seal.get("score_rows_sha256"):
            raise SinkProtocolError("group score-row digest differs from the sealed rows")
        positions = [int(row.get("position", -1)) for row in score_rows]
        cameras = [int(row.get("camera_id", -1)) for row in score_rows]
        groups = {row.get("source_group_sha256") for row in score_rows}
        if (positions != list(range(int(group_seal.get("start_position", -2)),
                                    int(group_seal.get("end_position_exclusive", -2)))) or
                len(groups) != 1 or next(iter(groups)) != group_seal.get("source_group_sha256") or
                cameras != [1, 2, 3, 4, 5] or
                group_seal.get("status") != "SEALED_BEFORE_LABEL_DISCLOSURE"):
            raise SinkProtocolError("group journal rejected an invalid or unsealed five-view group")
        queries: dict[tuple[str, int], Mapping[str, Any]] = {}
        queried_arms: set[str] = set()
        for row in score_rows:
            for arm, action in row.get("query_actions", {}).items():
                if (arm not in PRINCIPAL_ARMS or not isinstance(action, Mapping) or
                        not isinstance(action.get("executed"), bool)):
                    raise SinkProtocolError("group query action is malformed")
                receipt_sha = action.get("query_receipt_sha256")
                if action["executed"]:
                    if arm in queried_arms:
                        raise SinkProtocolError("a Source Group cannot purchase two labels for one arm")
                    queried_arms.add(arm)
                    if (not isinstance(receipt_sha, str) or len(receipt_sha) != 64 or
                            any(char not in "0123456789abcdef" for char in receipt_sha)):
                        raise SinkProtocolError("executed query lacks its receipt digest")
                    key = (str(arm), int(row["position"]))
                    if key in self._sealed_queries or key in queries:
                        raise SinkProtocolError("duplicate sealed query identity")
                    queries[key] = {"event_id_sha256": row["event_id_sha256"],
                                    "image_identity_sha256": row["image_identity_sha256"],
                                    "source_group_sha256": row["source_group_sha256"],
                                    "query_receipt_sha256": receipt_sha}
                elif receipt_sha is not None:
                    raise SinkProtocolError("unexecuted query carries a receipt digest")
        envelope = {"schema": "final-benchmark.stage-a-group.v1",
                    "group_seal": dict(group_seal), "score_rows": list(score_rows)}
        digest = self._write_payload(envelope)
        self._sealed_queries.update(queries)
        self.group_count += 1
        self.event_count += 5
        return digest

    def append_unit(self, *, group_seal: Mapping[str, Any],
                    score_rows: Sequence[Mapping[str, Any]]) -> str:
        """Durably seal one VisA image/duplicate-cluster unit before feedback."""
        if self.closed or not score_rows:
            raise SinkProtocolError("atomic-unit journal is closed or unit is empty")
        self._reject_outcomes(group_seal)
        self._reject_outcomes(score_rows)
        unit_sha = group_seal.get("atomic_unit_sha256")
        unit_kind = group_seal.get("atomic_unit_kind")
        if (unit_kind not in {"VISA_SINGLE_IMAGE", "VISA_EXACT_DUPLICATE_CLUSTER"} or
                not isinstance(unit_sha, str) or len(unit_sha) != 64 or
                any(char not in "0123456789abcdef" for char in unit_sha) or
                group_seal.get("unit_candidate_sha256") is None or
                group_seal.get("unit_approval_sha256") is None or
                group_seal.get("status") != "SEALED_BEFORE_LABEL_DISCLOSURE"):
            raise SinkProtocolError("VisA unit seal lacks its candidate-bound authorization")
        if hashlib.sha256(_canonical(list(score_rows))).hexdigest() != group_seal.get("score_rows_sha256"):
            raise SinkProtocolError("atomic-unit score-row digest differs from its sealed rows")
        start = int(group_seal.get("start_position", -1))
        stop = int(group_seal.get("end_position_exclusive", -1))
        positions = [int(row.get("position", -1)) for row in score_rows]
        event_ids = [str(row.get("event_id_sha256", "")) for row in score_rows]
        image_ids = [str(row.get("image_identity_sha256", "")) for row in score_rows]
        if (positions != list(range(start, stop)) or stop - start != len(score_rows) or
                start != self.event_count or unit_sha in self._visa_units or
                len(set(event_ids)) != len(event_ids) or len(set(image_ids)) != len(image_ids) or
                hashlib.sha256(_canonical(event_ids)).hexdigest() != group_seal.get("event_order_sha256") or
                any(row.get("atomic_unit_sha256") != unit_sha or
                    row.get("atomic_unit_kind") != unit_kind for row in score_rows) or
                group_seal.get("source_group_sha256") != unit_sha):
            raise SinkProtocolError("VisA rows do not match one ordered, registered atomic unit")
        queries: dict[tuple[str, int], Mapping[str, Any]] = {}
        queried_arms: set[str] = set()
        for row in score_rows:
            if not isinstance(row.get("query_actions"), Mapping):
                raise SinkProtocolError("VisA score row has no query-action map")
            for arm, action in row["query_actions"].items():
                if (arm not in PRINCIPAL_ARMS or not isinstance(action, Mapping) or
                        not isinstance(action.get("executed"), bool)):
                    raise SinkProtocolError("atomic-unit query action is malformed")
                receipt_sha = action.get("query_receipt_sha256")
                if action["executed"]:
                    if arm in queried_arms:
                        raise SinkProtocolError("an atomic unit cannot purchase two labels for one arm")
                    queried_arms.add(arm)
                    if (not isinstance(receipt_sha, str) or len(receipt_sha) != 64 or
                            any(char not in "0123456789abcdef" for char in receipt_sha)):
                        raise SinkProtocolError("executed atomic-unit query lacks its receipt digest")
                    key = (str(arm), int(row["position"]))
                    if key in self._sealed_queries or key in queries:
                        raise SinkProtocolError("duplicate sealed atomic-unit query identity")
                    queries[key] = {"event_id_sha256": row["event_id_sha256"],
                        "image_identity_sha256": row["image_identity_sha256"],
                        "source_group_sha256": unit_sha,
                        "query_receipt_sha256": receipt_sha}
                elif receipt_sha is not None:
                    raise SinkProtocolError("unexecuted VisA action carries a query receipt")
        envelope = {"schema": "final-benchmark.stage-a-atomic-unit.v1",
                    "unit_seal": dict(group_seal), "score_rows": list(score_rows)}
        digest = self._write_payload(envelope)
        self._sealed_queries.update(queries)
        self._visa_units.add(unit_sha)
        self.group_count += 1
        self.event_count += len(score_rows)
        return digest

    def query_is_sealed(self, *, arm_id: str, event: Any,
                        query_receipt: Mapping[str, Any]) -> bool:
        """Match an executed action to the already-fsynced group envelope."""
        saved = self._sealed_queries.get((str(arm_id), int(event.position)))
        return bool(saved and saved["event_id_sha256"] == event.event_id_sha256 and
                    saved["image_identity_sha256"] == event.image_identity_sha256 and
                    saved["source_group_sha256"] == event.group_sha256 and
                    saved["query_receipt_sha256"] == hashlib.sha256(_canonical(query_receipt)).hexdigest())

    def consume_sealed_query(self, *, arm_id: str, event: Any,
                             query_receipt: Mapping[str, Any]) -> bool:
        if not self.query_is_sealed(arm_id=arm_id, event=event, query_receipt=query_receipt):
            return False
        self._sealed_queries.pop((str(arm_id), int(event.position)))
        return True

    def seal_category(self, seal: Mapping[str, Any]) -> str:
        if self.closed or int(seal.get("completed_event_count", -1)) != self.event_count or int(
                seal.get("group_count", -1)) != self.group_count:
            raise SinkProtocolError("Stage A category seal does not match durable group journal")
        self._reject_outcomes(seal)
        if (seal.get("status") != "SEALED_COMPLETE" or
                not isinstance(seal.get("score_rows_sha256"), str) or
                not isinstance(seal.get("group_seals_sha256"), str)):
            raise SinkProtocolError("Stage A category completion receipt is incomplete")
        payload = _canonical({"schema": "final-benchmark.stage-a-category-seal.v1",
                              **dict(seal)}) + b"\n"
        try:
            with self.seal_path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            raise SinkProtocolError("Stage A category seal already exists; refusing overwrite") from None
        self._stream.flush()
        os.fsync(self._stream.fileno())
        return hashlib.sha256(payload).hexdigest()

    def close(self) -> None:
        if not self.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            self.closed = True

    def __enter__(self) -> "FsyncGroupSink":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
