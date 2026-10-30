"""Position-restricted access to Real-IAD test labels for P1.2.

The provider scans JSON structure without decoding test records outside the
verified Q-A-TS tape. For a requested row it extracts only ``anomaly_class``;
mask paths, unqueried labels, and image fields are skipped lexically.
"""
from __future__ import annotations

import hashlib
import json
import mmap
from pathlib import Path
from typing import Callable, Iterable


class QueriedLabelAccessError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _ws(data: mmap.mmap, index: int) -> int:
    while index < len(data) and data[index] in b" \t\r\n":
        index += 1
    return index


def _string(data: mmap.mmap, index: int) -> tuple[str, int]:
    index = _ws(data, index)
    if index >= len(data) or data[index] != ord('"'):
        raise QueriedLabelAccessError("Expected a JSON string token")
    start = index
    index += 1
    escaped = False
    while index < len(data):
        value = data[index]
        if escaped:
            escaped = False
        elif value == ord("\\"):
            escaped = True
        elif value == ord('"'):
            raw = data[start:index + 1]
            try:
                return json.loads(raw.decode("utf-8")), index + 1
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise QueriedLabelAccessError("Malformed JSON string token") from exc
        index += 1
    raise QueriedLabelAccessError("Unterminated JSON string token")


def _skip_value(data: mmap.mmap, index: int) -> int:
    """Skip a JSON value without decoding any strings or scalar values."""
    index = _ws(data, index)
    if index >= len(data):
        raise QueriedLabelAccessError("Missing JSON value")
    first = data[index]
    if first == ord('"'):
        index += 1
        escaped = False
        while index < len(data):
            value = data[index]
            if escaped:
                escaped = False
            elif value == ord("\\"):
                escaped = True
            elif value == ord('"'):
                return index + 1
            index += 1
        raise QueriedLabelAccessError("Unterminated skipped JSON string")
    if first in (ord("{"), ord("[")):
        stack = [ord("}") if first == ord("{") else ord("]")]
        index += 1
        in_string = False
        escaped = False
        while index < len(data) and stack:
            value = data[index]
            if in_string:
                if escaped:
                    escaped = False
                elif value == ord("\\"):
                    escaped = True
                elif value == ord('"'):
                    in_string = False
            elif value == ord('"'):
                in_string = True
            elif value == ord("{"):
                stack.append(ord("}"))
            elif value == ord("["):
                stack.append(ord("]"))
            elif value in (ord("}"), ord("]")):
                if not stack or value != stack[-1]:
                    raise QueriedLabelAccessError("Mismatched JSON container while skipping")
                stack.pop()
            index += 1
        if stack or in_string:
            raise QueriedLabelAccessError("Unterminated skipped JSON container")
        return index
    start = index
    while index < len(data) and data[index] not in b",]} \t\r\n":
        index += 1
    if index == start:
        raise QueriedLabelAccessError("Malformed JSON scalar")
    return index


def _decode_meta(data: mmap.mmap, start: int, stop: int) -> dict:
    try:
        value = json.loads(data[start:stop].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QueriedLabelAccessError("Malformed label-blind category metadata") from exc
    if not isinstance(value, dict):
        raise QueriedLabelAccessError("Category metadata must be a JSON object")
    return value


def _decode_selected_label(data: mmap.mmap, start: int, stop: int,
                           normal_class: str) -> int:
    index = _ws(data, start)
    if index >= stop or data[index] != ord("{"):
        raise QueriedLabelAccessError("Queried test record is not a JSON object")
    index += 1
    anomaly_class = None
    found = False
    while True:
        index = _ws(data, index)
        if index >= stop:
            raise QueriedLabelAccessError("Truncated queried test record")
        if data[index] == ord("}"):
            index += 1
            break
        key, index = _string(data, index)
        index = _ws(data, index)
        if index >= stop or data[index] != ord(":"):
            raise QueriedLabelAccessError("Malformed queried test record member")
        value_start = _ws(data, index + 1)
        if key == "anomaly_class":
            if found:
                raise QueriedLabelAccessError("Duplicate anomaly_class field in queried row")
            anomaly_class, index = _string(data, value_start)
            found = True
        else:
            index = _skip_value(data, value_start)
        index = _ws(data, index)
        if index < stop and data[index] == ord(","):
            index += 1
            continue
        if index < stop and data[index] == ord("}"):
            index += 1
            break
        raise QueriedLabelAccessError("Malformed queried test record separator")
    if _ws(data, index) != stop:
        raise QueriedLabelAccessError("Unexpected bytes after queried test record")
    if not found:
        raise QueriedLabelAccessError("Queried test record has no anomaly_class label")
    return 0 if anomaly_class == normal_class else 1


class _RestrictedTestArray:
    """Incremental test-array cursor that decodes one requested label at a time."""

    def __init__(self, path: Path):
        self._file = path.open("rb")
        self._data = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._cursor = self._locate_test_array()
        self._position = 0

    def _locate_test_array(self) -> int:
        data = self._data
        index = _ws(data, 0)
        if index >= len(data) or data[index] != ord("{"):
            raise QueriedLabelAccessError("Category JSON must start with an object")
        index += 1
        meta = None
        while True:
            index = _ws(data, index)
            if index >= len(data) or data[index] == ord("}"):
                break
            key, index = _string(data, index)
            index = _ws(data, index)
            if index >= len(data) or data[index] != ord(":"):
                raise QueriedLabelAccessError("Malformed category JSON member")
            value_start = _ws(data, index + 1)
            value_stop = _skip_value(data, value_start)
            if key == "meta":
                meta = _decode_meta(data, value_start, value_stop)
            elif key == "test":
                if meta is None or not isinstance(meta.get("normal_class"), str):
                    raise QueriedLabelAccessError(
                        "Label-blind normal_class metadata must precede test")
                if data[value_start] != ord("["):
                    raise QueriedLabelAccessError("Category test section must be an array")
                self.normal_class = meta["normal_class"]
                return _ws(data, value_start + 1)
            index = _ws(data, value_stop)
            if index < len(data) and data[index] == ord(","):
                index += 1
                continue
            if index < len(data) and data[index] == ord("}"):
                break
            raise QueriedLabelAccessError("Malformed category JSON separator")
        raise QueriedLabelAccessError("Category JSON has no test array")

    def label_at(self, requested_position: int) -> int:
        if requested_position < self._position:
            raise QueriedLabelAccessError("Label provider cannot move backward")
        data = self._data
        while self._position <= requested_position:
            start = _ws(data, self._cursor)
            if start >= len(data) or data[start] == ord("]"):
                raise QueriedLabelAccessError("A queried position falls outside the test array")
            stop = _skip_value(data, start)
            is_target = self._position == requested_position
            label = (_decode_selected_label(data, start, stop, self.normal_class)
                     if is_target else None)
            cursor = _ws(data, stop)
            if cursor < len(data) and data[cursor] == ord(","):
                self._cursor = _ws(data, cursor + 1)
            elif cursor < len(data) and data[cursor] == ord("]"):
                self._cursor = cursor
            else:
                raise QueriedLabelAccessError("Malformed test-array separator")
            self._position += 1
        assert label in (0, 1)
        return int(label)

    def close(self) -> None:
        self._data.close()
        self._file.close()


def _selected_test_labels(data: mmap.mmap, positions: set[int]) -> tuple[str, dict[int, int]]:
    """Legacy private helper retained for CPU tests of lexical skip behavior."""
    index = _ws(data, 0)
    if index >= len(data) or data[index] != ord("{"):
        raise QueriedLabelAccessError("Category JSON must start with an object")
    index += 1
    meta: dict | None = None
    labels: dict[int, int] = {}
    while True:
        index = _ws(data, index)
        if index >= len(data):
            raise QueriedLabelAccessError("Truncated category JSON")
        if data[index] == ord("}"):
            index += 1
            break
        key, index = _string(data, index)
        index = _ws(data, index)
        if index >= len(data) or data[index] != ord(":"):
            raise QueriedLabelAccessError("Malformed category JSON member")
        value_start = _ws(data, index + 1)
        if key == "meta":
            value_stop = _skip_value(data, value_start)
            meta = _decode_meta(data, value_start, value_stop)
        elif key == "test":
            if meta is None or not isinstance(meta.get("normal_class"), str):
                raise QueriedLabelAccessError("Label-blind normal_class metadata must precede test")
            if data[value_start] != ord("["):
                raise QueriedLabelAccessError("Category test section must be an array")
            cursor = _ws(data, value_start + 1)
            position = 0
            while cursor < len(data) and data[cursor] != ord("]"):
                start = cursor
                stop = _skip_value(data, start)
                if position in positions:
                    labels[position] = _decode_selected_label(
                        data, start, stop, meta["normal_class"])
                cursor = _ws(data, stop)
                if cursor < len(data) and data[cursor] == ord(","):
                    cursor = _ws(data, cursor + 1)
                elif cursor >= len(data) or data[cursor] != ord("]"):
                    raise QueriedLabelAccessError("Malformed test-array separator")
                position += 1
            if position <= max(positions, default=-1):
                raise QueriedLabelAccessError("A queried position falls outside the test array")
            index = cursor + 1
        else:
            index = _skip_value(data, value_start)
        index = _ws(data, index)
        if index < len(data) and data[index] == ord(","):
            index += 1
            continue
        if index < len(data) and data[index] == ord("}"):
            index += 1
            break
        raise QueriedLabelAccessError("Malformed category JSON separator")
    if meta is None or set(labels) != positions:
        raise QueriedLabelAccessError("Category JSON omitted metadata or a requested queried label")
    return str(meta["normal_class"]), labels


class QueriedPositionLabelReader:
    """Monotone oracle over exactly one verified Q-A-TS position set."""

    def __init__(self, *, category: str, path: Path,
                 expected_source_sha256: str, positions: Iterable[int],
                 group_sha256_by_position: dict[int, str]):
        self.category = category
        self.path = path.resolve()
        self.expected_source_sha256 = expected_source_sha256
        raw_positions = tuple(int(value) for value in positions)
        self.positions = tuple(sorted(set(raw_positions)))
        self.group_sha256_by_position = dict(group_sha256_by_position)
        if (not self.positions or len(self.positions) != len(raw_positions) or
                tuple(sorted(self.group_sha256_by_position)) != self.positions):
            raise QueriedLabelAccessError("Queried label allowlist is invalid or incomplete")
        self._cursor = 0
        self._array: _RestrictedTestArray | None = None
        self.decoded_positions: list[int] = []

    def _prepare(self) -> _RestrictedTestArray:
        if self._array is not None:
            return self._array
        if not self.path.is_file() or sha256_file(self.path) != self.expected_source_sha256:
            raise QueriedLabelAccessError("Category source hash differs from frozen metadata")
        try:
            self._array = _RestrictedTestArray(self.path)
        except Exception:
            if self._array is not None:
                self._array.close()
                self._array = None
            raise
        return self._array

    def close(self) -> None:
        if self._array is not None:
            self._array.close()
            self._array = None

    def _read_one(self, position: int) -> int:
        array = self._prepare()
        return array.label_at(position)

    def __call__(self, category: str, position: int, source_key: str) -> int:
        if category != self.category or self._cursor >= len(self.positions):
            raise QueriedLabelAccessError("Label request is outside this class tape")
        try:
            allow_index = self.positions.index(position)
        except ValueError as exc:
            raise QueriedLabelAccessError("Label request is not on the verified query tape") from exc
        if allow_index < self._cursor:
            raise QueriedLabelAccessError("Label request moved backward on the frozen tape")
        expected_position = self.positions[allow_index]
        group_digest = hashlib.sha256(
            ("official-sample-group\0" + source_key).encode("utf-8")).hexdigest()
        if self.group_sha256_by_position[position] != group_digest:
            raise QueriedLabelAccessError("Queried event source group differs from the verified tape")
        # A restored private checkpoint can resume after earlier queried
        # labels. The array cursor skips those rows lexically and decodes only
        # this request's anomaly_class field.
        label = self._read_one(expected_position)
        self.decoded_positions.append(position)
        self._cursor = allow_index + 1
        return label
