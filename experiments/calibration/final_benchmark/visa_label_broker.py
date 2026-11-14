"""Query-only broker for official VisA image-level TEST labels."""
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


class VisaLabelBrokerError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


class VisaQueryLabelBroker:
    """Decode one official TEST label only after its exact query unit is fsynced."""

    def __init__(self, *, category: str, condition_id: str,
                 dataset_root: str | Path, dataset_release_sha256: str,
                 source_csv_sha256: str, stream_records: Sequence[Mapping[str, Any]],
                 private_locators: Mapping[str, Mapping[str, Any]],
                 unit_candidate: Mapping[str, Any], approval: Mapping[str, Any],
                 sink: Any, private_feedback_path: str | Path):
        if (condition_id != "P0" or not re.fullmatch(r"[0-9a-f]{64}", dataset_release_sha256) or
                not re.fullmatch(r"[0-9a-f]{64}", source_csv_sha256) or
                not callable(getattr(sink, "query_is_sealed", None)) or
                not callable(getattr(sink, "consume_sealed_query", None))):
            raise VisaLabelBrokerError("VisA broker lacks its P0 release, source or durable-sink binding")
        self.root = Path(dataset_root).resolve(strict=True)
        self.category, self.condition_id = category, condition_id
        self.release_sha256, self.csv_sha256 = dataset_release_sha256, source_csv_sha256
        self.records = tuple(dict(row) for row in stream_records)
        self.locators = dict(private_locators)
        self.candidate, self.approval, self.sink = dict(unit_candidate), dict(approval), sink
        expected_ids = {str(row["image_identity_sha256"]) for row in self.records}
        if (not self.records or [row.get("position") for row in self.records] != list(range(len(self.records))) or
                set(self.locators).intersection(expected_ids) != expected_ids or
                self.candidate.get("dataset_id") != "visa_1cls" or
                self.candidate.get("category") not in (None, category) or
                self.candidate.get("candidate_sha256") != approval.get("unit_candidate_sha256")):
            raise VisaLabelBrokerError("VisA broker identities differ from the registered unit candidate")
        self._rows_by_original: dict[int, dict[str, Any]] = {}
        self._source = self._mapping = self._feedback = None
        self.decoded_queries: list[tuple[str, int]] = []
        self._decoded: set[tuple[str, int]] = set()
        csv_path = self.root / identity.VISA_1CLS_SOURCE
        try:
            self._source = csv_path.open("rb")
            digest = hashlib.sha256()
            for block in iter(lambda: self._source.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != source_csv_sha256:
                raise VisaLabelBrokerError("official VisA split-table bytes differ from the identity registration")
            self._mapping = mmap.mmap(self._source.fileno(), 0, access=mmap.ACCESS_READ)
            header_end = self._mapping.find(b"\n")
            if header_end < 0:
                raise VisaLabelBrokerError("official VisA split table has no header")
            header = next(__import__("csv").reader([self._mapping[:header_end].decode("utf-8-sig")]))
            self._header = tuple(header)
            self._lines: dict[int, bytes] = {}
            cursor = header_end + 1
            original_position = 0
            by_pos = {int(row["original_position"]): row for row in self.records}
            while cursor < len(self._mapping):
                end = self._mapping.find(b"\n", cursor)
                if end < 0:
                    end = len(self._mapping)
                line = self._mapping[cursor:end]
                if line.strip():
                    projected = identity._parse_csv_projected_line(
                        line, self._header, train_label_allowed=False)
                    if projected.get("split", "").lower() == "test":
                        image_key = projected.get("image", "").strip()
                        category_value = projected.get("object", "").strip()
                        image_id = identity._image_identity_digest(identity._nfc_key(image_key))
                        registered = by_pos.get(original_position)
                        if (registered is None or category_value != category or
                                image_id != registered.get("image_identity_sha256")):
                            if registered is not None or category_value == category:
                                raise VisaLabelBrokerError("VisA TEST identity/order differs from the category stream")
                        else:
                            locator = self.locators[image_id]
                            path = identity._safe_join(self.root, image_key, identity=image_id)
                            if (int(locator.get("original_position", -1)) != original_position or
                                    locator.get("official_image_key") != identity._nfc_key(image_key) or
                                    Path(locator.get("image_path", "")).resolve(strict=True) != path):
                                raise VisaLabelBrokerError("VisA TEST locator differs from the official CSV identity")
                            self._lines[original_position] = line
                            self._rows_by_original[original_position] = registered
                original_position += 1
                cursor = end + 1
            if set(self._rows_by_original) != set(by_pos):
                raise VisaLabelBrokerError("VisA category TEST stream does not cover its exact source rows")
            self._feedback = Path(private_feedback_path).open("xb")
        except BaseException:
            self.close()
            raise

    def disclose(self, *, arm_id: str, event: Any,
                 query_receipt: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
        if arm_id not in PRINCIPAL_ARMS or not 0 <= int(event.position) < len(self.records):
            raise VisaLabelBrokerError("VisA query arm or event position is not registered")
        key = (arm_id, int(event.position))
        if key in self._decoded:
            raise VisaLabelBrokerError("VisA query label was already disclosed to this arm/event")
        row = self.records[event.position]
        expected_unit = str(row["atomic_unit_sha256"])
        expected_kind = str(row["atomic_unit_kind"])
        expected_receipt = {
            "schema": "final-benchmark.query-receipt.atomic-unit.v1",
            "arm_id": arm_id, "category": self.category, "condition_id": self.condition_id,
            "event_id_sha256": event.event_id_sha256,
            "atomic_unit_sha256": expected_unit, "atomic_unit_kind": expected_kind,
            "position": event.position,
        }
        if (row.get("event_id_sha256") != event.event_id_sha256 or
                row.get("image_identity_sha256") != event.image_identity_sha256 or
                row.get("source_group_sha256") != event.group_sha256 or
                event.atomic_unit_sha256 != expected_unit or dict(query_receipt) != expected_receipt or
                not self.sink.query_is_sealed(arm_id=arm_id, event=event, query_receipt=query_receipt)):
            raise VisaLabelBrokerError("VisA label request lacks the matching fsynced executed-query receipt")
        original = int(row["original_position"])
        line = self._lines[original]
        projected = identity._parse_csv_projected_line(
            line, self._header, train_label_allowed=False, queried_test_label_allowed=True)
        if (projected.get("split") != "test" or projected.get("object") != self.category or
                identity._image_identity_digest(identity._nfc_key(projected.get("image", ""))) !=
                event.image_identity_sha256 or projected.get("label") not in {"normal", "anomaly"}):
            raise VisaLabelBrokerError("selected official VisA label row differs from the sealed event identity")
        if not self.sink.consume_sealed_query(arm_id=arm_id, event=event, query_receipt=query_receipt):
            raise VisaLabelBrokerError("VisA executed-query receipt was already consumed")
        label = int(projected["label"] == "anomaly")
        receipt_sha = hashlib.sha256(_canonical(query_receipt)).hexdigest()
        payload = _canonical({"schema": "final-benchmark.visa-private-feedback.v1",
            "arm_id": arm_id, "position": event.position,
            "event_id_sha256": event.event_id_sha256,
            "image_identity_sha256": event.image_identity_sha256,
            "atomic_unit_sha256": expected_unit,
            "query_receipt_sha256": receipt_sha, "label": label}) + b"\n"
        self._feedback.write(payload)
        self._feedback.flush()
        os.fsync(self._feedback.fileno())
        self._decoded.add(key)
        self.decoded_queries.append(key)
        return label, {"private_feedback_sha256": hashlib.sha256(payload).hexdigest(),
                       "query_receipt_sha256": receipt_sha,
                       "source_csv_sha256": self.csv_sha256}

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
