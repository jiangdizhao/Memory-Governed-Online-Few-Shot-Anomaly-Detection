"""Real-IAD query-only label broker bound to durable group score seals."""
from __future__ import annotations

import hashlib
import json
import mmap
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from . import identity
from .runner import PRINCIPAL_ARMS


class LabelBrokerError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _test_record_bounds(data: Any, bounds: tuple[int, int]) -> list[tuple[int, int]]:
    """Index TEST records without decoding their values or nested keys."""
    from experiments.calibration import r7_shuffled_correction_4class as r7
    start, stop = bounds
    cursor = r7._ws(data, start)
    if cursor >= stop or data[cursor] != ord("["):
        raise LabelBrokerError("registered TEST section is not an array")
    cursor += 1
    result = []
    while True:
        cursor = r7._ws(data, cursor)
        if cursor >= stop:
            raise LabelBrokerError("registered TEST array is truncated")
        if data[cursor] == ord("]"):
            if cursor + 1 != stop:
                raise LabelBrokerError("registered TEST array has trailing data")
            return result
        end = identity._skip_json_value_opaque(data, cursor)
        result.append((cursor, end))
        cursor = r7._ws(data, end)
        if cursor < stop and data[cursor] == ord(","):
            cursor += 1
            continue
        if cursor >= stop or data[cursor] != ord("]"):
            raise LabelBrokerError("registered TEST array separator is malformed")


class RealIADQueryLabelBroker:
    """Decode only a purchased image's official image-level JSON label.

    The E6/R7 normal-class comparison is preserved. Object boundaries are
    scanned without decoding any unqueried outcome or mask value. This class
    is for a separately authorized future run, not a Gate 1 evaluation.
    """

    def __init__(self, *, category: str, condition_id: str, stream_seed: int,
                 dataset_release_sha256: str, annotation_path: str | Path,
                 annotation_sha256: str, stream_records: Sequence[Mapping[str, Any]],
                 private_locators: Mapping[str, Mapping[str, str]],
                 sink: Any, private_feedback_path: str | Path):
        from experiments.calibration import r7_shuffled_correction_4class as r7
        from experiments.risk_safe.preflight import parse_image_name

        if (condition_id not in {"P0", "P1", "P2"} or
                stream_seed != {"P0": 0, "P1": 11, "P2": 0}[condition_id] or
                not re.fullmatch(r"[0-9a-f]{64}", dataset_release_sha256) or
                not re.fullmatch(r"[0-9a-f]{64}", annotation_sha256) or
                not callable(getattr(sink, "query_is_sealed", None)) or
                not callable(getattr(sink, "consume_sealed_query", None))):
            raise LabelBrokerError("broker lacks the registered stream or durable sink")
        records = list(stream_records)
        original_positions = [row.get("original_position") for row in records]
        if (not records or [row.get("position") for row in records] != list(range(len(records))) or
                any(isinstance(value, bool) or not isinstance(value, int) for value in original_positions) or
                sorted(original_positions) != list(range(len(records))) or
                set(private_locators) != {row.get("image_identity_sha256") for row in records}):
            raise LabelBrokerError("private locator set does not match the registered TEST stream")
        for row in records:
            image_id = str(row["image_identity_sha256"])
            loc = private_locators[image_id]
            image_key, group_key = loc["official_image_key"], loc["official_group_key"]
            group, camera = identity._realiad_identity_key(category, image_key)
            parsed = parse_image_name(loc["image_path"], expected_class=category)
            ordinal = row.get("within_group_ordinal")
            event_id = identity._domain_hash("Gate0-event-v1", dataset_release_sha256, category,
                                             str(stream_seed), group_key, str(camera), image_key, str(ordinal))
            if (group != group_key or parsed.official_sample_key != group_key or
                    parsed.camera_id != camera or ordinal != camera or row.get("camera_id") != camera or
                    identity._image_identity_digest(image_key) != image_id or
                    identity._source_group_digest(group_key) != row.get("source_group_sha256") or
                    event_id != row.get("event_id_sha256")):
                raise LabelBrokerError("private TEST identity differs from its registered event")
        self.category, self.condition_id = category, condition_id
        self.annotation_sha256 = annotation_sha256
        self.records, self.sink = tuple(records), sink
        self.decoded_queries: list[tuple[str, int]] = []
        self._decoded: set[tuple[str, int]] = set()
        self._source = self._mapping = self._feedback = None
        try:
            self._source = Path(annotation_path).open("rb")
            digest = hashlib.sha256()
            for block in iter(lambda: self._source.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != annotation_sha256:
                raise LabelBrokerError("official annotation hash differs from its registration")
            self._mapping = mmap.mmap(self._source.fileno(), 0, access=mmap.ACCESS_READ)
            fields = identity._identity_object_fields(self._mapping, 0, len(self._mapping))
            if "meta" not in fields or "test" not in fields:
                raise LabelBrokerError("official annotation lacks metadata or TEST records")
            meta_fields = identity._identity_object_fields(self._mapping, *fields["meta"])
            if ("normal_class" not in meta_fields or
                    self._mapping[meta_fields["normal_class"][0]] != ord('"')):
                raise LabelBrokerError("official annotation lacks normal_class metadata")
            self._normal_class = r7._decode_field(self._mapping, meta_fields["normal_class"])
            if not isinstance(self._normal_class, str) or not self._normal_class:
                raise LabelBrokerError("official normal_class metadata is malformed")
            self._bounds = _test_record_bounds(self._mapping, fields["test"])
            if len(self._bounds) != len(records):
                raise LabelBrokerError("official TEST record count differs from the registered stream")
            self._feedback = Path(private_feedback_path).open("xb")
        except BaseException:
            self.close()
            raise

    def disclose(self, *, arm_id: str, event: Any,
                 query_receipt: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
        from experiments.calibration import r7_shuffled_correction_4class as r7
        if (arm_id not in PRINCIPAL_ARMS or not 0 <= event.position < len(self.records) or
                (arm_id, event.position) in self._decoded):
            raise LabelBrokerError("query arm or event position is not available")
        row = self.records[event.position]
        expected_receipt = {"schema": "final-benchmark.query-receipt.v1", "arm_id": arm_id,
            "category": self.category, "condition_id": self.condition_id,
            "event_id_sha256": event.event_id_sha256, "source_group_sha256": event.group_sha256,
            "position": event.position}
        if (row["event_id_sha256"] != event.event_id_sha256 or
                row["image_identity_sha256"] != event.image_identity_sha256 or
                row["source_group_sha256"] != event.group_sha256 or
                dict(query_receipt) != expected_receipt or
                not self.sink.query_is_sealed(arm_id=arm_id, event=event, query_receipt=query_receipt)):
            raise LabelBrokerError("label request lacks its matching fsynced executed-query receipt")
        start, stop = self._bounds[int(row["original_position"])]
        fields = identity._identity_object_fields(self._mapping, start, stop)
        if ("image_path" not in fields or "anomaly_class" not in fields or
                self._mapping[fields["image_path"][0]] != ord('"') or
                self._mapping[fields["anomaly_class"][0]] != ord('"')):
            raise LabelBrokerError("selected annotation lacks its official identity or class")
        image_key = r7._decode_field(self._mapping, fields["image_path"])
        if (not isinstance(image_key, str) or
                identity._image_identity_digest(image_key) != event.image_identity_sha256):
            raise LabelBrokerError("selected annotation image identity differs from the sealed event")
        if not self.sink.consume_sealed_query(arm_id=arm_id, event=event, query_receipt=query_receipt):
            raise LabelBrokerError("executed-query receipt was already consumed")
        self._decoded.add((arm_id, event.position))
        label_class = r7._decode_field(self._mapping, fields["anomaly_class"])
        if not isinstance(label_class, str):
            raise LabelBrokerError("selected official image-level anomaly_class is malformed")
        label = int(label_class != self._normal_class)
        query_sha = hashlib.sha256(_canonical(query_receipt)).hexdigest()
        payload = _canonical({"schema": "final-benchmark.private-feedback.v1",
            "arm_id": arm_id, "position": event.position, "event_id_sha256": event.event_id_sha256,
            "query_receipt_sha256": query_sha, "label": label}) + b"\n"
        self._feedback.write(payload)
        self._feedback.flush()
        os.fsync(self._feedback.fileno())
        self.decoded_queries.append((arm_id, event.position))
        return label, {"private_feedback_sha256": hashlib.sha256(payload).hexdigest(),
                       "query_receipt_sha256": query_sha,
                       "annotation_sha256": self.annotation_sha256}

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._source is not None:
            self._source.close()
            self._source = None
        if self._feedback is not None:
            self._feedback.close()
            self._feedback = None
