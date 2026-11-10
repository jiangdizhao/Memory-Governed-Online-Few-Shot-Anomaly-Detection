"""Non-overwriting fsync group journal for outcome-free Stage A records."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


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
        envelope = {"schema": "final-benchmark.stage-a-group.v1",
                    "group_seal": dict(group_seal), "score_rows": list(score_rows)}
        digest = self._write_payload(envelope)
        self.group_count += 1
        self.event_count += 5
        return digest

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
