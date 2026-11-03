"""R7 fixed-shuffle Correction necessity experiment.

The module keeps Stage A label access behind a shuffled-position query gate.
Stage B is a separate CPU child and will only open outcomes after the four
full-stream Stage A receipts have been sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
EXPECTED_PARENT = "276746b01fba47a7d41fa3ceb01b40e961fee341"
R5_EXECUTED_RUNNER_SHA256 = "69d34f74dd81e43834a08028407211ae48e27451c2c388da88fc86618608c939"
R5_COMMITTED_RUNNER_SHA256 = "bb07c3a9bef677c3321a4c384af04d459185a5ee801d91f51aabe9058d6d73cb"
R5_REPORT_SHA256 = "b3888fe53316dd7beaf169b18476a2ac03b77c9425a9e980e56ff589b583f9f3"
RUN_ID = "r7_shuffled_correction_4class_20261103_a4"
OUTPUT_BASE = ROOT / "outputs" / "calibration"
RUN_NAMESPACE = OUTPUT_BASE / RUN_ID
REPORT_DIR = ROOT / "docs" / "calibration" / "audit" / "r7_shuffled_correction_4class_20261103"
REPORT_PATH = REPORT_DIR / "REPORT.md"
AGGREGATE_PATH = REPORT_DIR / "aggregate_results.json"
JSON_ROOT = Path(r"G:\dataset\realiad\jsons\realiad_jsons")
IMAGE_ROOT = Path(r"G:\dataset\realiad\images")
CATEGORIES = ("audiojack", "regulator", "switch", "rolled_strip_base")
EXPECTED_COUNTS = {"audiojack": 3815, "regulator": 3855,
                   "switch": 3820, "rolled_strip_base": 3805}
EXPECTED_QUERIES = {"audiojack": 114, "regulator": 116,
                    "switch": 115, "rolled_strip_base": 114}
TOTAL_EVENTS = sum(EXPECTED_COUNTS.values())
TOTAL_QUERIES = sum(EXPECTED_QUERIES.values())
MEMORY_CAPACITY = 1024
OUTPUT_LIMIT = 4 * 1024**3
FAILURE_RESERVE = 64 * 1024
MIN_FREE_BYTES = 10 * 1024**3
FULL_DEADLINE_SECONDS = 120 * 60
SMOKE_DEADLINE_SECONDS = 15 * 60
SMOKE_EVENTS = 512
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 73191
SMOKE_CATEGORY = "audiojack"
SCORE_SCHEMA = "r7.fixed_shuffle.score.v1"
STAGE_A_SCHEMA = "r7.fixed_shuffle.stage_a_seal.v1"
STAGE_B_SCHEMA = "r7.fixed_shuffle.stage_b_seal.v1"
RUN_SCHEMA = "r7.fixed_shuffle.run.v1"
IDENTITY_SCHEMA = "r7.fixed_shuffle.identity.v1"
AUTHORIZATION = "R7 four-class fixed-shuffle retrospective development experiment"
ARMS = ("G-Frozen", "G-B1", "G-B3", "GC-Frozen", "GC-B1", "GC-B3")
SCORE_SOURCE_PATHS = (
    "experiments/calibration/r3_fullstream_2x2.py",
    "experiments/calibration/r6_capacity1024_audiojack_runner.py",
    "experiments/calibration/r5_audiojack_b3_fpsa_runner.py",
    "experiments/calibration/query_tape.py",
    "experiments/calibration/stage1a_p12_label_provider.py",
    "experiments/calibration/controllers.py",
    "experiments/calibration/stage05.py",
    "experiments/calibration/profiles.py",
    "experiments/calibration/native.py",
    "experiments/calibration/r1_matched_correction_pilot.py",
    "experiments/calibration/r2_feedback_repair.py",
    "experiments/calibration/r2_exp2a_c1_strength.py",
    "experiments/calibration/r1_runtime_supervisor.py",
    "experiments/risk_safe/native_bridge.py",
    "experiments/p0_protocol_audit.py",
    "experiments/v02_memory.py",
    "vmb_visionad_new.py",
    "test_phase3_AIF_simplified_v22.py",
)


class R7Error(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_read(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _git(args: Sequence[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8")
    return result.stdout.strip()


def _assert_repository_identity() -> None:
    branch = _git(("branch", "--show-current"))
    head = _git(("rev-parse", "HEAD"))
    if branch != EXPECTED_BRANCH or head != EXPECTED_PARENT:
        raise R7Error(f"source identity mismatch: branch={branch!r}, HEAD={head}")
    dirty = _git(("status", "--porcelain", "--", *SCORE_SOURCE_PATHS))
    if dirty:
        raise R7Error("a pinned experiment dependency has local changes")


def _source_fingerprints() -> dict[str, str]:
    values = {}
    for relative in SCORE_SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise R7Error(f"required source is missing: {relative}")
        values[relative] = _sha256_file(path)
    return values


@dataclass(frozen=True)
class ShuffledEvent:
    position: int
    original_position: int
    relative_id: str
    image_identity_sha256: str
    source_group_sha256: str
    camera_id: str
    timestamp: str
    image_path: str

    @property
    def event_id_sha256(self) -> str:
        token = (f"r7-event\0{self.position}\0{self.image_identity_sha256}\0"
                 f"{self.source_group_sha256}").encode("utf-8")
        return hashlib.sha256(token).hexdigest()


def _identity_hash(relative_id: str) -> str:
    return hashlib.sha256(("r7-image-identity\0" + relative_id).encode("utf-8")).hexdigest()


def deterministic_group_shuffle(rows: Sequence[Any], seed: int = 0) -> list[ShuffledEvent]:
    """Shuffle first-seen official groups; preserve each group's original view order."""
    def field(row: Any, name: str) -> Any:
        return row[name] if isinstance(row, Mapping) else getattr(row, name)

    buckets: dict[str, list[Any]] = {}
    group_order: list[str] = []
    for row in rows:
        group = str(field(row, "opaque_group"))
        if not re.fullmatch(r"[0-9a-f]{64}", group):
            raise R7Error("Stage 05 supplied a malformed opaque Source Group identity")
        if group not in buckets:
            group_order.append(group)
            buckets[group] = []
        buckets[group].append(row)
    for group in group_order:
        buckets[group].sort(key=lambda row: int(field(row, "position")))
    shuffled_group_order = list(group_order)
    random.Random(int(seed)).shuffle(shuffled_group_order)
    output: list[ShuffledEvent] = []
    for group in shuffled_group_order:
        for row in buckets[group]:
            relative_id = str(field(row, "relative_id")).replace("\\", "/")
            image_hash = _identity_hash(relative_id)
            output.append(ShuffledEvent(
                position=len(output), original_position=int(field(row, "position")),
                relative_id=relative_id, image_identity_sha256=image_hash,
                source_group_sha256=group, camera_id=str(field(row, "camera_id")),
                timestamp=str(field(row, "timestamp")), image_path=str(field(row, "image_path"))))
    if len(output) != len(rows) or len({row.original_position for row in output}) != len(rows):
        raise R7Error("group shuffle is not a bijection over the frozen stream")
    return output


def _identity_receipt(events: Sequence[ShuffledEvent]) -> dict[str, Any]:
    rows = [{"position": row.position, "original_position": row.original_position,
             "image_identity_sha256": row.image_identity_sha256,
             "source_group_sha256": row.source_group_sha256,
             "camera_id": row.camera_id}
            for row in events]
    return {"schema": "r7.shuffled_identity_map.v1", "event_count": len(rows),
            "mapping_sha256": _sha256_json(rows), "rows": rows}


def _make_query_tape(category: str, events: Sequence[ShuffledEvent]) -> Any:
    from experiments.calibration import query_tape
    from experiments.calibration.protocol import TapeRecord
    from experiments.calibration.profiles import QueryProtocolConfig

    records = [TapeRecord(
        position=event.position, official_sample_key=event.source_group_sha256,
        camera_id=int(event.camera_id), relative_id=event.image_identity_sha256,
        timestamp=event.timestamp, phase=0) for event in events]
    tape = query_tape.build_qa_ts(records, config=QueryProtocolConfig())
    query_tape.verify_tape(tape, records, config=QueryProtocolConfig())
    if len(tape.events) != EXPECTED_QUERIES[category]:
        raise R7Error(f"{category}: shuffled Q-A-TS query budget changed")
    positions = tuple(sorted(int(item.position) for item in tape.events))
    groups = [events[position].source_group_sha256 for position in positions]
    if len(set(positions)) != len(positions) or len(set(groups)) != len(groups):
        raise R7Error(f"{category}: Q-A-TS did not enforce one query per Source Group")
    if any(not 0 <= position < len(events) for position in positions):
        raise R7Error(f"{category}: query tape position lies outside the shuffled stream")
    return tape


def _tape_receipt(tape: Any) -> dict[str, Any]:
    positions = tuple(sorted(int(value) for value in tape.positions))
    return {"schema": "r7.shuffled_query_tape.v1", "seed": 37,
            "query_fraction": 0.03, "bins": 16, "rounding": "half_up",
            "query_count": len(positions), "positions": list(positions),
            "tape_sha256": str(tape.digest),
            "positions_sha256": _sha256_json(list(positions))}


def _ws(data: bytes, offset: int) -> int:
    while offset < len(data) and data[offset] in b" \t\r\n":
        offset += 1
    return offset


def _string(data: bytes, offset: int) -> tuple[str, int]:
    if offset >= len(data) or data[offset] != ord('"'):
        raise R7Error("malformed JSON string boundary")
    end = offset + 1
    escaped = False
    while end < len(data):
        byte = data[end]
        if byte == ord('"') and not escaped:
            token = data[offset:end + 1]
            try:
                return json.loads(token.decode("utf-8")), end + 1
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise R7Error("invalid UTF-8 JSON string") from exc
        if byte == ord("\\") and not escaped:
            escaped = True
        else:
            escaped = False
        end += 1
    raise R7Error("unterminated JSON string")


def _skip_value(data: bytes, offset: int) -> int:
    offset = _ws(data, offset)
    if offset >= len(data):
        raise R7Error("truncated JSON value")
    if data[offset] == ord('"'):
        end = offset + 1
        escaped = False
        while end < len(data):
            byte = data[end]
            if byte == ord('"') and not escaped:
                return end + 1
            if byte == ord("\\") and not escaped:
                escaped = True
            else:
                escaped = False
            end += 1
        raise R7Error("unterminated JSON string while indexing unselected value")
    if data[offset] in (ord('{'), ord('[')):
        opening = data[offset]
        closing = ord('}') if opening == ord('{') else ord(']')
        offset += 1
        while True:
            offset = _ws(data, offset)
            if offset >= len(data):
                raise R7Error("unterminated JSON object or array")
            if data[offset] == closing:
                return offset + 1
            if opening == ord('{'):
                _, offset = _string(data, offset)
                offset = _ws(data, offset)
                if offset >= len(data) or data[offset] != ord(':'):
                    raise R7Error("malformed JSON object member")
                offset = _skip_value(data, offset + 1)
            else:
                offset = _skip_value(data, offset)
            offset = _ws(data, offset)
            if offset < len(data) and data[offset] == ord(','):
                offset += 1
                continue
            if offset < len(data) and data[offset] == closing:
                return offset + 1
            raise R7Error("malformed JSON delimiter")
    end = offset
    while end < len(data) and data[end] not in b",]} \t\r\n":
        end += 1
    if end == offset:
        raise R7Error("empty JSON scalar")
    return end


def _object_fields(data: bytes, start: int, stop: int) -> dict[str, tuple[int, int]]:
    pos = _ws(data, start)
    if pos >= stop or data[pos] != ord('{'):
        raise R7Error("expected JSON object")
    pos += 1
    fields: dict[str, tuple[int, int]] = {}
    while True:
        pos = _ws(data, pos)
        if pos >= stop:
            raise R7Error("truncated JSON object")
        if data[pos] == ord('}'):
            return fields
        key, pos = _string(data, pos)
        pos = _ws(data, pos)
        if pos >= stop or data[pos] != ord(':'):
            raise R7Error("malformed JSON object field")
        value_start = _ws(data, pos + 1)
        value_stop = _skip_value(data, value_start)
        if key in fields:
            raise R7Error("duplicate JSON object key")
        fields[key] = (value_start, value_stop)
        pos = _ws(data, value_stop)
        if pos < stop and data[pos] == ord(','):
            pos += 1
            continue
        if pos < stop and data[pos] == ord('}'):
            return fields
        raise R7Error("malformed JSON object end")


def _test_array_boundaries(data: bytes) -> list[tuple[int, int]]:
    """Index records without decoding outcome or mask values."""
    pos = _ws(data, 0)
    if pos >= len(data) or data[pos] != ord('{'):
        raise R7Error("expected top-level Real-IAD JSON object")
    pos += 1
    test_bounds = None
    while True:
        pos = _ws(data, pos)
        if pos >= len(data):
            raise R7Error("truncated top-level JSON object")
        if data[pos] == ord('}'):
            break
        key, pos = _string(data, pos)
        pos = _ws(data, pos)
        if pos >= len(data) or data[pos] != ord(':'):
            raise R7Error("malformed top-level JSON field")
        value_start = _ws(data, pos + 1)
        value_stop = _skip_value(data, value_start)
        if key == "test":
            test_bounds = (value_start, value_stop)
        pos = _ws(data, value_stop)
        if pos < len(data) and data[pos] == ord(','):
            pos += 1
            continue
        if pos < len(data) and data[pos] == ord('}'):
            break
        raise R7Error("malformed top-level object end")
    if test_bounds is None:
        raise R7Error("TEST array is missing")
    start, stop = test_bounds
    pos = _ws(data, start)
    if pos >= stop or data[pos] != ord('['):
        raise R7Error("TEST field is not an array")
    pos += 1
    out = []
    while True:
        pos = _ws(data, pos)
        if pos >= stop:
            raise R7Error("truncated TEST array")
        if data[pos] == ord(']'):
            break
        record_start = pos
        record_stop = _skip_value(data, record_start)
        out.append((record_start, record_stop))
        pos = _ws(data, record_stop)
        if pos < stop and data[pos] == ord(','):
            pos += 1
            continue
        if pos < stop and data[pos] == ord(']'):
            break
        raise R7Error("malformed TEST array separator")
    return out


def _decode_field(data: bytes, bounds: tuple[int, int]) -> Any:
    start, stop = bounds
    try:
        return json.loads(data[start:stop].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R7Error("selected JSON field could not be decoded") from exc


class IndexedQueriedLabelReader:
    """Decode only an authorized queried record's outcome at its mapped source index."""
    def __init__(self, path: Path, category: str, expected_sha256: str,
                 events: Sequence[ShuffledEvent], query_positions: Sequence[int]):
        self.path = path
        self.category = category
        self.events = tuple(events)
        self.allowed = tuple(int(value) for value in query_positions)
        self.allowed_set = set(self.allowed)
        self.cursor = 0
        self.decoded_positions: list[int] = []
        self.data_file = path.open("rb")
        if _sha256_file(path) != expected_sha256:
            self.data_file.close()
            raise R7Error(f"{category}: TEST JSON source hash changed after preflight")
        self.mapping = mmap.mmap(self.data_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.bounds = _test_array_boundaries(self.mapping)
        if len(self.bounds) != len(self.events):
            self.close()
            raise R7Error(f"{category}: TEST JSON record count differs from frozen metadata")
        top_fields = _object_fields(self.mapping, 0, len(self.mapping))
        meta_bounds = top_fields.get("meta")
        if meta_bounds is None:
            self.close()
            raise R7Error(f"{category}: top-level label-blind category metadata is missing")
        metadata = _decode_field(self.mapping, meta_bounds)
        self.normal_class = metadata.get("normal_class") if isinstance(metadata, dict) else None
        if not isinstance(self.normal_class, str) or not self.normal_class:
            self.close()
            raise R7Error(f"{category}: top-level normal-class identity is invalid")

    def label_for(self, position: int, *, sealed_and_fsynced: bool) -> int:
        position = int(position)
        if sealed_and_fsynced is not True:
            raise R7Error("query feedback requested before the score row was fsynced")
        if position not in self.allowed_set:
            raise R7Error("attempt to disclose an unqueried TEST outcome")
        if self.cursor >= len(self.allowed) or position != self.allowed[self.cursor]:
            raise R7Error("queried TEST outcomes requested out of tape order or more than once")
        event = self.events[position]
        original_position = event.original_position
        if not 0 <= original_position < len(self.bounds):
            raise R7Error("shuffled position maps outside the original JSON array")
        start, stop = self.bounds[original_position]
        fields = _object_fields(self.mapping, start, stop)
        image_field = fields.get("image_path")
        if image_field is None or "anomaly_class" not in fields:
            raise R7Error("queried JSON record lacks required image/outcome fields")
        raw_image_path = _decode_field(self.mapping, image_field)
        if not isinstance(raw_image_path, str):
            raise R7Error("queried record image_path is not a string")
        from experiments.risk_safe.preflight import parse_image_name
        parsed = parse_image_name(raw_image_path, expected_class=self.category)
        if (str(parsed.rel_path).replace("\\", "/") != event.relative_id or
                _identity_hash(str(parsed.rel_path).replace("\\", "/")) !=
                event.image_identity_sha256 or int(parsed.camera_id) != int(event.camera_id) or
                str(parsed.timestamp) != event.timestamp or
                hashlib.sha256(("official-sample-group\0" +
                                str(parsed.official_sample_key)).encode("utf-8")).hexdigest() !=
                event.source_group_sha256):
            raise R7Error("queried outcome record does not match the frozen image identity")
        from experiments.calibration.stage1a_p12_label_provider import _decode_selected_label
        try:
            label = int(_decode_selected_label(self.mapping, start, stop, self.normal_class))
        except Exception as exc:
            raise R7Error("queried record outcome could not be decoded under the restricted label reader") from exc
        self.cursor += 1
        self.decoded_positions.append(position)
        return label

    def finish(self) -> None:
        if tuple(self.decoded_positions) != self.allowed:
            raise R7Error("Stage A did not consume the exact shuffled query allowlist")

    def close(self) -> None:
        mapping = getattr(self, "mapping", None)
        if mapping is not None:
            mapping.close()
            self.mapping = None
        file = getattr(self, "data_file", None)
        if file is not None:
            file.close()
            self.data_file = None

    def __enter__(self) -> "IndexedQueriedLabelReader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class ScoreDisclosureGate:
    def __init__(self, write_and_fsync: Any):
        self._write_and_fsync = write_and_fsync
        self.last_position = -1
        self.sealed_positions: set[int] = set()

    def seal(self, position: int, row: Mapping[str, Any]) -> None:
        position = int(position)
        if position != self.last_position + 1:
            raise R7Error("score rows must be sealed in shuffled event order")
        if row.get("position") != position or "label" in row or "anomaly_class" in row:
            raise R7Error("invalid score row at the Stage A disclosure boundary")
        self._write_and_fsync(row)
        self.last_position = position
        self.sealed_positions.add(position)

    def is_sealed(self, position: int) -> bool:
        return int(position) in self.sealed_positions


def validate_image_projection(path: Path, category: str, events: Sequence[ShuffledEvent]) -> None:
    """Compare TEST image_path identities to frozen Stage 05 rows without labels/masks."""
    raw = path.read_bytes()
    bounds = _test_array_boundaries(raw)
    if len(bounds) != len(events):
        raise R7Error(f"{category}: image metadata array length differs from frozen event list")
    for event in events:
        start, stop = bounds[event.original_position]
        fields = _object_fields(raw, start, stop)
        image_field = fields.get("image_path")
        if image_field is None:
            raise R7Error(f"{category}: a TEST record is missing image_path metadata")
        image_path = _decode_field(raw, image_field)
        if not isinstance(image_path, str):
            raise R7Error(f"{category}: TEST image_path metadata is not a string")
        from experiments.risk_safe.preflight import parse_image_name
        parsed = parse_image_name(image_path, expected_class=category)
        relative_id = str(parsed.rel_path).replace("\\", "/")
        group_sha = hashlib.sha256(("official-sample-group\0" +
                                     str(parsed.official_sample_key)).encode("utf-8")).hexdigest()
        if (relative_id != event.relative_id or _identity_hash(relative_id) != event.image_identity_sha256 or
                int(parsed.camera_id) != int(event.camera_id) or
                str(parsed.timestamp) != event.timestamp or
                group_sha != event.source_group_sha256):
            raise R7Error(f"{category}: TEST JSON image identity differs at a frozen position")


def _budget(root: Path, *, create: bool = False) -> Any:
    # Importing R2's OutputBudget hides CUDA at module import. Stage A restores
    # the inherited visibility before constructing any native model.
    from experiments.calibration.r2_feedback_repair import OutputBudget
    return OutputBudget(root, create=create, max_bytes=OUTPUT_LIMIT,
                        failure_reserve_bytes=FAILURE_RESERVE, max_files=20)


class BudgetJsonl:
    def __init__(self, budget: Any, name: str):
        self.budget, self.name = budget, name
        self.pending: list[Mapping[str, Any]] = []

    def write(self, row: Mapping[str, Any], *, flush: bool = False) -> None:
        self.pending.append(dict(row))
        if flush:
            self.flush()

    def flush(self) -> None:
        for row in self.pending:
            self.budget.append_jsonl(self.name, row)
        self.pending.clear()


def _ensure_no_reparse_chain(path: Path, stop: Path) -> None:
    from experiments.calibration.r2_feedback_repair import _is_reparse
    current = path.absolute()
    stop = stop.absolute()
    while True:
        if current.exists() and _is_reparse(current):
            raise R7Error(f"reparse point in R7 output chain: {current}")
        if current == stop:
            return
        parent = current.parent
        if parent == current or stop not in parent.parents and parent != stop:
            raise R7Error("output path is not contained by its authorized root")
        current = parent


def _check_disk_space(path: Path) -> int:
    import shutil
    usage = shutil.disk_usage(path)
    if usage.free < MIN_FREE_BYTES:
        raise R7Error(f"R7 requires at least {MIN_FREE_BYTES} bytes free on the output volume")
    return int(usage.free)


def _frozen_contract() -> dict[str, Any]:
    from experiments.calibration import r3_fullstream_2x2 as r3
    contract = r3.load_frozen_contract(verify_image_paths=True)
    if set(contract["classes"]) != set(CATEGORIES):
        raise R7Error("frozen Stage 05 categories do not match the authorized four-class set")
    frozen = _json_read(r3.STAGE05 / "frozen_inputs.json")
    result: dict[str, Any] = {
        "r3": r3, "metadata": {}, "source_receipts": {
            "stage05_run_manifest_sha256": contract["stage05_run_manifest_sha256"],
            "frozen_inputs_sha256": contract["frozen_inputs_sha256"],
            "score_tape_manifest_sha256": contract["score_tape_manifest_sha256"],
            "query_tape_archive_sha256": contract["query_tape_archive_sha256"],
        },
    }
    score_manifest = _json_read(r3.STAGE05 / "score_tape_manifest.json")
    for category in CATEGORIES:
        frozen_info = frozen["classes"][category]
        class_contract = contract["classes"][category]
        rows = class_contract["stream_rows"]
        if len(rows) != EXPECTED_COUNTS[category]:
            raise R7Error(f"{category}: unexpected frozen event count")
        events = deterministic_group_shuffle(rows, seed=0)
        if len(events) != EXPECTED_COUNTS[category]:
            raise R7Error(f"{category}: shuffled stream length changed")
        json_path = r3.JSON_ROOT / f"{category}.json"
        if _sha256_file(json_path) != class_contract["category_json_sha256"]:
            raise R7Error(f"{category}: public TEST JSON hash changed after frozen contract validation")
        validate_image_projection(json_path, category, events)
        tape = _make_query_tape(category, events)
        tape_receipt = _tape_receipt(tape)
        if tape_receipt["query_count"] != EXPECTED_QUERIES[category]:
            raise R7Error(f"{category}: Q-A-TS query budget mismatch")
        c64 = frozen_info["c64"]
        supports = frozen_info["supports"]["0"]
        c64_groups = [hashlib.sha256(("official-sample-group\0" + str(group)).encode("utf-8")).hexdigest()
                      for group in c64["groups"]]
        if (len(c64["images"]) != 64 or len(set(c64_groups)) != 64 or
                class_contract["c64_group_sha256"] != c64["group_sha256"]):
            raise R7Error(f"{category}: C64 group/image identity receipt is inconsistent")
        c64_paths = [r3.DATASET_ROOT / "images" / frozen_info["prefix"] / rel
                     for rel in c64["images"]]
        if any(not path.is_file() for path in c64_paths):
            raise R7Error(f"{category}: frozen C64 image is missing")
        score_row = score_manifest["classes"][category]["score_tapes"]["0"]
        score_path = (r3.STAGE05 / str(score_row["file"])).resolve(strict=True)
        if r3.STAGE05.resolve(strict=True) not in score_path.parents:
            raise R7Error(f"{category}: C64 score path escaped the frozen Stage 05 directory")
        if _sha256_file(score_path) != score_row.get("sha256"):
            raise R7Error(f"{category}: C64 score archive digest mismatch")
        import numpy as np
        with np.load(score_path, allow_pickle=False) as archive:
            c64_scores = np.asarray(archive["c64_scores"], dtype=np.float64)
            c64_identity = str(archive["c64_identity_sha256"].item())
        if (c64_scores.shape != (64,) or not np.isfinite(c64_scores).all() or
                c64_identity != c64["image_sha256"]):
            raise R7Error(f"{category}: C64 score archive identity or shape is invalid")
        result["metadata"][category] = {
            "events": events, "tape": tape, "tape_receipt": tape_receipt,
            "contract": class_contract, "frozen_info": frozen_info,
            "json_path": json_path, "c64_paths": c64_paths,
            "c64_groups": c64_groups, "c64_scores": c64_scores.tolist(),
            "c64_score_archive_sha256": score_row["sha256"],
            "support_group_sha256": class_contract["support_group_sha256"],
            "support_image_sha256": class_contract["support_image_sha256"],
            "fixed_support_sha256_expected": class_contract["fixed_support_sha256_expected"],
            "theta_c64_q90": float(class_contract["theta_c64_q90"]),
            "sigma_c64_floor_002": float(class_contract["sigma_c64_floor_002"]),
            "identity_receipt": _identity_receipt(events),
        }
    return result


def _experiment_identity(contract: Mapping[str, Any], sources: Mapping[str, str],
                          runner_sha256: str) -> dict[str, Any]:
    from dataclasses import asdict
    from experiments.calibration.profiles import NativeScorerConfig
    native_profile = NativeScorerConfig()
    p0_config = {
        "device": "cuda", "augmentation_seed": 0, "bank_seed": 0,
        "fixed_capacity": 25000, "fixed_per_image": 5000,
        "augmentations": 4, "include_identity": False, "multiplier": 2.5,
        "feature_fuse": "concat", "topk_patches": 32,
        "image_aggregation": "max", "image_topk": 64,
        "image_quantile": 0.95, "knn": 5,
    }
    correction_config = dict(contract["r3"].CORRECTION_CONFIG)
    correction_config["capacity"] = MEMORY_CAPACITY
    classes = {}
    for category in CATEGORIES:
        item = contract["metadata"][category]
        classes[category] = {
            "event_count": EXPECTED_COUNTS[category],
            "query_count": EXPECTED_QUERIES[category],
            "shuffled_identity_map_sha256": item["identity_receipt"]["mapping_sha256"],
            "shuffled_identity_map": item["identity_receipt"]["rows"],
            "shuffled_query_tape": item["tape_receipt"],
            "fixed_support_sha256_expected": item["fixed_support_sha256_expected"],
            "c64_image_sha256": item["frozen_info"]["c64"]["image_sha256"],
            "c64_group_sha256": item["frozen_info"]["c64"]["group_sha256"],
            "c64_score_archive_sha256": item["c64_score_archive_sha256"],
            "category_json_sha256": item["contract"]["category_json_sha256"],
            "theta_c64_q90": item["theta_c64_q90"],
            "sigma_c64_floor_002": item["sigma_c64_floor_002"],
        }
    identity = {
        "schema": IDENTITY_SCHEMA,
        "classification": "RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC",
        "source_parent_commit": EXPECTED_PARENT,
        "branch": EXPECTED_BRANCH,
        "source_fingerprints": dict(sources),
        "runner_source_sha256": runner_sha256,
        "configuration": {
            "categories": list(CATEGORIES), "events": dict(EXPECTED_COUNTS),
            "group_shuffle": {"seed": 0, "unit": "opaque official Source Group",
                              "group_order": "first occurrence in frozen JSON stream",
                              "within_group_order": "original JSON view order"},
            "query_protocol": "Q-A-TS-v1; rate=0.03; seed=37; bins=16; half_up; max one camera per group",
            "query_count_by_category": dict(EXPECTED_QUERIES),
            "greedy_capacity": MEMORY_CAPACITY,
            "correction": {"recipe": "frozen original C1-Full SignedCorrection",
                           "capacity": MEMORY_CAPACITY, "initial_state": "empty"},
            "decision_arms": list(ARMS),
            "controller_arms_affect_only_decisions": True,
            "frozen_writer_controls": {"source": "G pre-event native C0 score",
                                       "threshold": "frozen C64 q90",
                                       "feedback": "queried verified outcomes only"},
            "b1": {"window": 256, "min_count": 16, "quantile": "higher q90"},
            "b3": {"method": "StateConsistentQuantile", "delta": 0.01,
                   "min_count": 16, "effective_from_next_event": True},
            "feature_encoding_per_test_event": 1,
            "stage_a_query_label_only": True,
            "stage_b_after_all_four_stage_a_seals": True,
            "stage_b_labels_only": True, "test_masks_opened": False,
            "runtime_profile": "stage05_set_seed_0_math_sdp_v1",
            "native_p0_config": p0_config,
            "native_scorer_profile": asdict(native_profile),
            "native_scorer_profile_sha256": native_profile.sha256,
            "greedy_writer": {
                "policy": "greedy_coreset", "capacity_per_layer": MEMORY_CAPACITY,
                "settings": {"flat": MEMORY_CAPACITY, "stm": MEMORY_CAPACITY,
                             "ltm": 0, "buffer": 0, "min_events": 2,
                             "promote_every": 16, "cap_bytes": 2 * 1024**3},
                "single_common_state_for_G_and_GC": True,
            },
            "native_correction_config_capacity_1024": correction_config,
            "output_limit_bytes_per_directory": OUTPUT_LIMIT,
            "failure_reserve_bytes": FAILURE_RESERVE,
            "minimum_prelaunch_free_bytes": MIN_FREE_BYTES,
            "full_deadline_seconds_per_category": FULL_DEADLINE_SECONDS,
            "smoke_deadline_seconds": SMOKE_DEADLINE_SECONDS,
            "automatic_restart": False,
            "resume_incomplete_run": False,
        },
        "stage05_source_receipts": dict(contract["source_receipts"]),
        "classes": classes,
    }
    identity["identity_sha256"] = _sha256_json(identity)
    return identity


def _create_output_directories() -> tuple[Path, Path]:
    from experiments.calibration.r2_feedback_repair import _is_reparse
    if RUN_NAMESPACE.exists():
        raise R7Error("R7 output namespace already exists; incomplete output is never resumed")
    if REPORT_DIR.exists():
        raise R7Error("R7 report namespace already exists; refusing to overwrite")
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    _ensure_no_reparse_chain(OUTPUT_BASE, ROOT)
    if _is_reparse(OUTPUT_BASE):
        raise R7Error("R7 output base is a reparse point")
    RUN_NAMESPACE.mkdir(parents=False, exist_ok=False)
    if _is_reparse(RUN_NAMESPACE) or RUN_NAMESPACE.resolve(strict=True).parent != OUTPUT_BASE.resolve(strict=True):
        raise R7Error("R7 unique output namespace was redirected")
    smoke = RUN_NAMESPACE / "smoke"
    full = RUN_NAMESPACE / "full"
    _budget(smoke, create=True)
    _budget(full, create=True)
    if set(child.name for child in RUN_NAMESPACE.iterdir()) != {"smoke", "full"}:
        raise R7Error("R7 namespace contains unexpected output paths")
    return smoke, full


def _read_history(output: Path) -> list[dict[str, Any]]:
    path = output / "attempt_history.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_history(budget: Any, row: Mapping[str, Any]) -> None:
    budget.append_jsonl("attempt_history.jsonl", dict(row))


def _verify_child_identity(output: Path, mode: str, category: str,
                           smoke_events: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if mode not in {"SMOKE_ONLY", "FULL_STAGE_A"} or category not in CATEGORIES:
        raise R7Error("invalid R7 child mode or category")
    expected_root = RUN_NAMESPACE / ("smoke" if mode == "SMOKE_ONLY" else "full")
    if output.resolve(strict=True) != expected_root.resolve(strict=True):
        raise R7Error("R7 child output directory is not the manifest-bound unique namespace")
    _ensure_no_reparse_chain(output, OUTPUT_BASE)
    manifest = _json_read(output / "run_manifest.json")
    token = os.environ.get("R7_CHILD_TOKEN", "")
    if (not token or hashlib.sha256(token.encode("utf-8")).hexdigest() !=
            manifest.get("authorization_token_sha256")):
        raise R7Error("real R7 pilot cannot start without a parent-issued authorization token")
    if (manifest.get("schema") != RUN_SCHEMA or manifest.get("run_id") != RUN_ID or
            manifest.get("mode") != mode or manifest.get("authorization") != AUTHORIZATION or
            manifest.get("source_parent_commit") != EXPECTED_PARENT or
            manifest.get("branch") != EXPECTED_BRANCH or
            manifest.get("source_fingerprints") != _source_fingerprints() or
            manifest.get("runner_source_sha256") != _sha256_file(Path(__file__).resolve())):
        raise R7Error("R7 child source identity, branch, commit or authorization mismatch")
    _assert_repository_identity()
    if mode == "SMOKE_ONLY" and not 0 < int(smoke_events) <= 512:
        raise R7Error("R7 integration smoke must contain 1..512 shuffled events")
    if mode == "FULL_STAGE_A" and int(smoke_events) != 0:
        raise R7Error("R7 full Stage A does not accept a partial event limit")
    if _check_disk_space(OUTPUT_BASE) < MIN_FREE_BYTES:
        raise R7Error("R7 output volume fell below the prelaunch free-space threshold")
    contract = _frozen_contract()
    identity = _experiment_identity(contract, _source_fingerprints(),
                                     _sha256_file(Path(__file__).resolve()))
    if manifest.get("identity_sha256") != identity["identity_sha256"]:
        raise R7Error("R7 child regenerated a different shuffled/tape identity")
    category_item = contract["metadata"][category]
    if (manifest.get("class_json_sha256_by_category", {}).get(category) !=
            category_item["contract"]["category_json_sha256"]):
        raise R7Error("R7 category JSON source identity mismatch")
    return contract, manifest


def _load_r5_audiojack_c64_cache(contract: Mapping[str, Any], category: str) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """Read only the sealed R5 C64 feature cache; never parse event scores or labels."""
    if category != "audiojack":
        return None
    import numpy as np
    from PIL import Image
    r5_root = OUTPUT_BASE / "r5_audiojack_b3_fpsa_20261102_a1"
    required = ("run_manifest.json", "stage_a_seal.json", "stage_b_seal.json",
                "complete_seal.json", "c64_cache_manifest.json")
    if any(not (r5_root / name).is_file() for name in required):
        raise R7Error("R5 audiojack C64 cache is missing required sealed receipt files")
    _ensure_no_reparse_chain(r5_root, OUTPUT_BASE)
    r5_run = _json_read(r5_root / "run_manifest.json")
    r5_a_path = r5_root / "stage_a_seal.json"
    r5_b_path = r5_root / "stage_b_seal.json"
    r5_complete_path = r5_root / "complete_seal.json"
    r5_a = _json_read(r5_a_path)
    r5_b = _json_read(r5_b_path)
    r5_complete = _json_read(r5_complete_path)
    cache_path = r5_root / "c64_cache_manifest.json"
    cache_manifest = _json_read(cache_path)
    category_item = contract["metadata"][category]
    if (r5_run.get("run_id") != "r5_audiojack_b3_fpsa_20261102_a1" or
            r5_run.get("category") != category or r5_run.get("mode") != "FULL_STAGE_A" or
            r5_run.get("event_count") != EXPECTED_COUNTS[category] or
            r5_run.get("class_json_sha256") != category_item["contract"]["category_json_sha256"] or
            r5_run.get("fixed_support_sha256") != category_item["fixed_support_sha256_expected"] or
            r5_run.get("c64_score_tape_sha256") != category_item["c64_score_archive_sha256"] or
            r5_run.get("runner_source_sha256") != R5_EXECUTED_RUNNER_SHA256 or
            r5_a.get("status") != "SEALED_COMPLETE" or r5_a.get("category") != category or
            r5_a.get("event_count") != EXPECTED_COUNTS[category] or
            r5_b.get("status") != "SEALED_COMPLETE" or r5_b.get("test_masks_opened") is not False or
            r5_b.get("stage_a_seal_sha256") != _sha256_file(r5_a_path) or
            r5_complete.get("status") != "COMPLETE" or
            r5_complete.get("stage_a_seal_sha256") != _sha256_file(r5_a_path) or
            r5_complete.get("stage_b_seal_sha256") != _sha256_file(r5_b_path) or
            r5_a.get("c64_cache_manifest_sha256") != _sha256_file(cache_path) or
            cache_manifest.get("schema") != "r5.audiojack.c64_feature_cache.v1" or
            cache_manifest.get("category") != category or
            cache_manifest.get("encoded_once_per_image") is not True or
            cache_manifest.get("protected_train_normal_count") != 64 or
            cache_manifest.get("test_labels_or_masks_accessed") is not False or
            cache_manifest.get("c64_group_sha256") != category_item["frozen_info"]["c64"]["group_sha256"] or
            cache_manifest.get("c64_image_sha256") != category_item["frozen_info"]["c64"]["image_sha256"]):
        raise R7Error("R5 C64 cache source run, Stage A/B seal or frozen identity is invalid")
    r5_report_path = ROOT / "docs/calibration/audit/r5_audiojack_b3_fpsa_20261102/REPORT.md"
    r5_report_text = r5_report_path.read_text(encoding="utf-8") if r5_report_path.is_file() else ""
    r5_runner_path = ROOT / "experiments/calibration/r5_audiojack_b3_fpsa_runner.py"
    if (not r5_report_path.is_file() or _sha256_file(r5_report_path) != R5_REPORT_SHA256 or
            R5_EXECUTED_RUNNER_SHA256 not in r5_report_text or
            R5_COMMITTED_RUNNER_SHA256 not in r5_report_text or
            "only the extra blank line at EOF was removed" not in r5_report_text or
            "no executable code changed" not in r5_report_text or
            _sha256_file(r5_runner_path) != R5_COMMITTED_RUNNER_SHA256 or
            subprocess.run(["git", "diff", "--quiet", EXPECTED_PARENT, "--",
                            str(r5_runner_path.relative_to(ROOT))], cwd=ROOT).returncode != 0 or
            not isinstance(r5_run.get("source_fingerprints"), dict)):
        raise R7Error("R5 C64 cache runner identities do not match the sealed execution and reviewed source")
    for relative, digest in r5_run["source_fingerprints"].items():
        source = ROOT / relative
        if not source.is_file() or _sha256_file(source) != digest:
            raise R7Error(f"R5 C64 cache dependency changed: {relative}")
    receipts = cache_manifest.get("cache_receipts")
    expected_groups = category_item["c64_groups"]
    if not isinstance(receipts, list) or len(receipts) != 64:
        raise R7Error("R5 C64 cache does not contain exactly 64 file receipts")
    cache: list[dict[str, Any]] = []
    for index, receipt in enumerate(receipts):
        expected_name = f"c64_cache_{index:03d}.npz"
        if (receipt.get("index") != index or receipt.get("file") != expected_name or
                receipt.get("source_group_sha256") != expected_groups[index]):
            raise R7Error(f"R5 C64 cache receipt identity mismatch at {index}")
        source = r5_root / expected_name
        if source.is_symlink() or not source.is_file():
            raise R7Error(f"R5 C64 feature cache file is absent or redirected at {index}")
        payload = source.read_bytes()
        if (len(payload) != receipt.get("file_bytes") or
                hashlib.sha256(payload).hexdigest() != receipt.get("file_sha256")):
            raise R7Error(f"R5 C64 feature cache checksum mismatch at {index}")
        with np.load(source, allow_pickle=False) as archive:
            rgb = np.asarray(archive["rgb_uint8"], dtype=np.uint8).copy()
            keys = sorted((key for key in archive.files if key.startswith("feature_")),
                          key=lambda key: int(key.rsplit("_", 1)[1]))
            features = [np.asarray(archive[key]).copy() for key in keys]
        if (rgb.ndim != 3 or rgb.shape[2] != 3 or len(features) != 1 or
                features[0].shape != (784, 4096) or features[0].dtype != np.float32 or
                not np.isfinite(features[0]).all()):
            raise R7Error(f"R5 C64 feature representation is incompatible at {index}")
        with Image.open(category_item["c64_paths"][index]) as image:
            source_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape != source_rgb.shape or not np.array_equal(rgb, source_rgb):
            raise R7Error(f"R5 C64 cache RGB bytes differ from the frozen image at {index}")
        cache.append({"rgb": rgb, "features": features,
                      "source_group_sha256": expected_groups[index],
                      "cache_sha256": receipt["file_sha256"]})
    receipt = {
        "reused": True, "source_run_id": r5_run["run_id"],
        "source_runner_executed_sha256": R5_EXECUTED_RUNNER_SHA256,
        "source_runner_committed_sha256": R5_COMMITTED_RUNNER_SHA256,
        "source_runner_change_classification": "only an extra blank line at EOF removed; no executable code changed",
        "source_runner_report_sha256": R5_REPORT_SHA256,
        "source_run_manifest_sha256": _sha256_file(r5_root / "run_manifest.json"),
        "source_stage_a_seal_sha256": _sha256_file(r5_a_path),
        "source_stage_b_seal_sha256": _sha256_file(r5_b_path),
        "source_complete_seal_sha256": _sha256_file(r5_complete_path),
        "source_cache_manifest_sha256": _sha256_file(cache_path),
        "count": len(cache), "c64_group_sha256": cache_manifest["c64_group_sha256"],
        "c64_image_sha256": cache_manifest["c64_image_sha256"],
        "all_cache_file_receipts_verified": True,
        "all_rgb_pixels_match_frozen_c64_images": True,
        "feature_contract": {"shape": [784, 4096], "dtype": "float32"},
    }
    return cache, receipt

def _make_c64_items(category: str, info: Mapping[str, Any], p0: Any,
                    impl: Any, backbone: Any, cfg: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cached = _load_r5_audiojack_c64_cache({"metadata": {category: info}}, category)
    if cached is not None:
        return cached
    import numpy as np
    import torch
    items = []
    started = time.perf_counter()
    for index, path in enumerate(info["c64_paths"]):
        image, features = p0.encode_query(impl, backbone, str(path), cfg)
        if (len(features) != 1 or tuple(features[0].shape) != (784, 4096) or
                features[0].dtype != torch.float32 or not torch.isfinite(features[0]).all().item()):
            raise R7Error(f"{category}: C64 native feature contract failed at index {index}")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        feature = features[0].detach().to("cpu").contiguous().numpy().copy()
        items.append({"rgb": rgb, "features": [feature],
                      "source_group_sha256": info["c64_groups"][index]})
    return items, {"reused": False, "source_run_id": None,
                   "count": len(items), "c64_group_sha256": info["frozen_info"]["c64"]["group_sha256"],
                   "c64_image_sha256": info["frozen_info"]["c64"]["image_sha256"],
                   "new_c64_dino_encodings": len(items),
                   "encoding_seconds": time.perf_counter() - started,
                   "feature_contract": {"shape": [784, 4096], "dtype": "float32"}}


def _state_digests(bank: Any, source_groups: Mapping[int, set[str]],
                   fixed_digest: str, r1: Any) -> dict[str, str]:
    greedy = r1._memory_content_sha256(bank)
    correction = r1._correction_state_sha256(bank, source_groups)
    return {
        "G": _sha256_json({"fixed_support_sha256": fixed_digest,
                           "greedy_memory_sha256": greedy}),
        "GC": _sha256_json({"fixed_support_sha256": fixed_digest,
                            "greedy_memory_sha256": greedy,
                            "correction_state_sha256": correction}),
    }


def _c64_evidence(scores: Sequence[float], version: int, group_sha256: str,
                  state_digest: str) -> Any:
    from experiments.calibration.protocol import EvidenceAnchor, EvidenceSnapshot
    values = tuple(float(value) for value in scores)
    if (len(values) != 64 or any(not math.isfinite(value) for value in values) or
            not re.fullmatch(r"[0-9a-f]{64}", group_sha256) or
            not re.fullmatch(r"[0-9a-f]{64}", state_digest)):
        raise R7Error("B3 C64 evidence is incomplete or unbound to current scorer state")
    anchors = tuple(EvidenceAnchor(
        opaque_anchor_id=hashlib.sha256(f"r7-c64-{i}-{group_sha256}".encode()).hexdigest(),
        score=value, memory_version=int(version), protected=True,
        source_exclusion_policy="protected_no_selfwrite")
        for i, value in enumerate(values))
    return EvidenceSnapshot(
        memory_version=int(version), normal_anchors=anchors, defect_anchors=(),
        source_exclusion_audit={"protected_c64_only": True,
                                "c64_group_sha256": group_sha256,
                                "scorer_state_sha256": state_digest,
                                "c64_score_vector_sha256": _sha256_json(list(values))},
        evidence_revision=int(version))


def _initialize_b3(controller: Any, scores: Sequence[float], group_sha256: str,
                   state_digest: str, theta0: float) -> dict[str, Any]:
    before = float(controller.theta)
    evidence = _c64_evidence(scores, 0, group_sha256, state_digest)
    controller.on_memory_change(evidence)
    after = float(controller.finalize_step())
    if (before != theta0 or after != theta0 or controller.evidence_version != 0 or
            controller.status != "CURRENT_C64_Q90" or
            tuple(controller.normal_scores) != tuple(float(value) for value in scores)):
        raise R7Error("B3 initial complete-state C64 evidence failed its version-zero contract")
    controller._r7_state_digest = state_digest
    return {"version": 0, "theta": after, "status": controller.status,
            "movement": after - before, "c64_score_vector_sha256": _sha256_json(list(scores))}


def _refresh_b3(controller: Any, scores: Sequence[float], group_sha256: str,
                previous_digest: str, new_digest: str, previous_version: int) -> dict[str, Any] | None:
    if previous_digest == new_digest:
        return None
    if (controller.evidence_version != previous_version or
            len(scores) != 64 or getattr(controller, "_r7_state_digest", None) != previous_digest):
        raise R7Error("B3 refresh has stale version or scorer-state identity")
    old_theta = float(controller.theta)
    new_version = previous_version + 1
    evidence = _c64_evidence(scores, new_version, group_sha256, new_digest)
    controller.on_memory_change(evidence)
    new_theta = float(controller.finalize_step())
    if (controller.evidence_version != new_version or not math.isfinite(new_theta) or
            tuple(controller.normal_scores) != tuple(float(value) for value in scores) or
            abs(new_theta - old_theta) > 0.01 + 1e-12):
        raise R7Error("B3 current-state threshold update violates state version or delta contract")
    controller._r7_state_digest = new_digest
    raw = sorted(float(value) for value in scores)[math.ceil(0.90 * 63)]
    return {"version_before": previous_version, "version_after": new_version,
            "old_theta": old_theta, "raw_c64_q90": raw,
            "new_theta": new_theta, "movement": new_theta - old_theta,
            "c64_score_vector_sha256": _sha256_json(list(scores))}


def _predict_b3(controller: Any, *, position: int, event: ShuffledEvent,
                score: float, version: int, state_digest: str) -> dict[str, Any]:
    from experiments.calibration.protocol import Observation
    if (controller.evidence_version != version or
            getattr(controller, "_r7_state_digest", None) != state_digest):
        raise R7Error("B3 decision does not match the active complete scorer state")
    observation = Observation(
        t=position, observation_id=event.event_id_sha256,
        camera_id=int(event.camera_id), score=float(score),
        memory_version=version, score_state_digest=state_digest)
    result = controller.predict(observation)
    if (result.label != int(score >= controller.theta) or
            result.threshold != controller.theta or not result.available):
        raise R7Error("B3 prediction does not follow score >= current threshold")
    return {"prediction": int(result.label), "threshold": float(result.threshold),
            "status": str(result.reason or controller.status),
            "scorer_state_version": version, "scorer_state_sha256": state_digest}


def _score_c64_state(impl: Any, bank: Any, items: Sequence[Mapping[str, Any]], cfg: Any,
                     r1: Any, source_groups: Mapping[int, set[str]], category: str,
                     c64_group_sha256: str) -> tuple[list[float], list[tuple[Any, Any, float]], float]:
    import numpy as np
    import torch
    from PIL import Image
    output_scores: list[float] = []
    base_cache: list[tuple[Any, Any, float]] = []
    started = time.perf_counter()
    for position, item in enumerate(items):
        image = Image.fromarray(np.asarray(item["rgb"], dtype=np.uint8), mode="RGB")
        features = [torch.from_numpy(np.ascontiguousarray(value)).to(bank.device)
                    for value in item["features"]]
        base_map, top_ids, base_score, summary = r1._native_dense_score(
            impl, bank, image, features, cfg)
        if (not np.array_equal(top_ids, np.asarray(summary["layer0"]["topk_patch_idx"], dtype=np.int64)) or
                len(top_ids) != 32):
            raise R7Error(f"{category}: native C64 Top-32 receipt changed")
        base_cache.append((base_map, top_ids, float(base_score)))
        selected = impl.l2_normalize(features[0].to(bank.device))[
            torch.as_tensor(top_ids, device=bank.device, dtype=torch.long)]
        delta, diagnostics = r1._c1_score_delta(
            impl, bank, selected, source_groups, str(item["source_group_sha256"]),
            event_position=position)
        if int(diagnostics.get("same_source_excluded_anchor_slots", 0)):
            # C64 are frozen, disjoint references; a matching source indicates
            # a broken identity partition and is not silently tolerated.
            raise R7Error("C64 source-group exclusion found a colliding correction anchor")
        corrected = r1.corrected_dense_map(base_map, top_ids, delta)
        output_scores.append(float(r1.max_score(corrected)))
    return output_scores, base_cache, time.perf_counter() - started


def _score_c64_base(impl: Any, bank: Any, items: Sequence[Mapping[str, Any]],
                    cfg: Any, r1: Any, category: str
                    ) -> tuple[list[float], list[tuple[Any, Any, float]], float]:
    import numpy as np
    import torch
    from PIL import Image
    scores: list[float] = []
    cache: list[tuple[Any, Any, float]] = []
    started = time.perf_counter()
    for item in items:
        image = Image.fromarray(np.asarray(item["rgb"], dtype=np.uint8), mode="RGB")
        features = [torch.from_numpy(np.ascontiguousarray(value)).to(bank.device)
                    for value in item["features"]]
        base_map, top_ids, base_score, summary = r1._native_dense_score(
            impl, bank, image, features, cfg)
        if (len(top_ids) != 32 or not np.array_equal(
                top_ids, np.asarray(summary["layer0"]["topk_patch_idx"], dtype=np.int64))):
            raise R7Error(f"{category}: native C64 base Top-32 receipt changed")
        cache.append((base_map, top_ids, float(base_score)))
        scores.append(float(base_score))
    return scores, cache, time.perf_counter() - started


def _score_c64_corrected(impl: Any, bank: Any, items: Sequence[Mapping[str, Any]],
                         base_cache: Sequence[tuple[Any, Any, float]], cfg: Any,
                         r1: Any, source_groups: Mapping[int, set[str]], category: str,
                         *, event_position: int
                         ) -> tuple[list[float], float]:
    import numpy as np
    import torch
    scores = []
    started = time.perf_counter()
    for index, (item, base) in enumerate(zip(items, base_cache)):
        base_map, top_ids, _ = base
        feature = torch.from_numpy(np.ascontiguousarray(item["features"][0])).to(bank.device)
        selected = impl.l2_normalize(feature)[
            torch.as_tensor(top_ids, device=bank.device, dtype=torch.long)]
        delta, diagnostics = r1._c1_score_delta(
            impl, bank, selected, source_groups, str(item["source_group_sha256"]),
            event_position=event_position)
        if int(diagnostics.get("same_source_excluded_anchor_slots", 0)):
            raise R7Error("C64 source-group exclusion found a colliding correction anchor")
        scores.append(float(r1.max_score(r1.corrected_dense_map(base_map, top_ids, delta))))
    return scores, time.perf_counter() - started


def _tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _mutable_state_bytes(bank: Any, correction_cb: Any, dynamic_codebooks_fn: Any) -> dict[str, Any]:
    import torch
    greedy = dynamic_codebooks_fn(bank)
    greedy_prototype_used = sum(_tensor_bytes(cb.protos) for cb in greedy)
    greedy_metadata = sum(
        _tensor_bytes(value) for cb in greedy for name, value in vars(cb).items()
        if name != "protos" and isinstance(value, torch.Tensor))
    correction_arrays = ("protos", "beta", "radius", "kappa", "count",
                         "outcome_fp", "outcome_fn", "outcome_tn", "outcome_tp", "last_step")
    correction_prototypes = _tensor_bytes(correction_cb.protos)
    correction_metadata = sum(_tensor_bytes(getattr(correction_cb, name))
                              for name in correction_arrays if name != "protos")
    return {
        "greedy_used_prototype_bytes": greedy_prototype_used,
        "greedy_mutable_metadata_bytes": greedy_metadata,
        "correction_allocated_prototype_bytes": correction_prototypes,
        "correction_allocated_metadata_bytes": correction_metadata,
        "combined_mutable_tensor_bytes": (greedy_prototype_used + greedy_metadata +
                                          correction_prototypes + correction_metadata),
        "greedy_occupancy_by_layer": [int(cb.size()) for cb in greedy],
        "greedy_capacity_by_layer": [int(cb.K) for cb in greedy],
        "correction_occupancy": int(correction_cb.size()),
        "correction_capacity": int(correction_cb.K),
        "correction_positive_occupancy": int((correction_cb.beta[:correction_cb.size()] > 0).sum().item()),
        "correction_negative_occupancy": int((correction_cb.beta[:correction_cb.size()] < 0).sum().item()),
        "correction_zero_sign_occupancy": int((correction_cb.beta[:correction_cb.size()] == 0).sum().item()),
    }


def _stage_a_child(category: str, mode: str, smoke_events: int = 0) -> dict[str, Any]:
    cuda_visibility_before_imports = os.environ.get("CUDA_VISIBLE_DEVICES")
    import numpy as np
    import torch
    from types import SimpleNamespace
    from collections import Counter
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r2_exp2a_c1_strength as exp2a
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration import r6_capacity1024_audiojack_runner as r6
    from experiments.calibration.controllers import RollingQ90QueriedNormal, StateConsistentQuantile
    from experiments.calibration.protocol import Feedback, Observation
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.risk_safe.native_bridge import dynamic_codebooks
    from v02_memory import MemoryArm
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job

    if not current_process_is_in_job():
        raise R7Error("R7 Stage A child is outside the Windows Job Object")
    output = Path(os.environ.get("R7_CHILD_OUTPUT", ""))
    contract, manifest = _verify_child_identity(output, mode, category, smoke_events)
    budget = _budget(output)
    r3._restore_cuda_visibility(cuda_visibility_before_imports)
    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise R7Error("R7 Stage A requires exactly one authorized CUDA device")
    torch.set_num_threads(1)
    impl = p0.require_sources()
    model_receipt = r3._get_verified_model_receipt()
    restore_hub, runtime_info = r3._runtime_and_native_setup(torch, impl, model_receipt)
    cfg, scorer_profile = r3._native_config(
        p0, __import__("experiments.calibration.profiles", fromlist=["NativeScorerConfig"]), r1)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    backbone = None
    reader = None
    try:
        setup_started = time.perf_counter()
        try:
            backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
        finally:
            restore_hub()
        loaded_model_sha = r1._model_state_sha256(backbone.model)
        if loaded_model_sha != model_receipt.get("loaded_model_state_sha256"):
            raise R7Error("R7 loaded DINO weights differ from the independently verified receipt")
        runtime_info["loaded_model_state_sha256"] = loaded_model_sha
        info = contract["metadata"][category]
        source = info["contract"]
        support_paths = [str(path) for path in source["support_paths"]]
        support_features = p0.encode_reference_features(impl, backbone, support_paths, cfg)
        support_encoder_invocations = sum(len(support_features[path]) for path in support_paths)
        impl.set_seed(0)
        bank = p0.build_vmb(impl, support_features, support_paths, cfg)
        fixed_receipt = r3._check_bank_receipt(
            bank, p0, info["fixed_support_sha256_expected"], r1)
        r3._validate_correction_config(bank)
        correction_cfg, correction_banks = r6._configure_r6_correction_capacity(bank)
        correction_cb = bank.corr_cb["p0"]["id"][0]
        if (len(correction_banks) != 1 or correction_cb is not correction_banks[0] or
                correction_cb.size() != 0 or correction_cb.K != MEMORY_CAPACITY or
                correction_cfg.K_correction != MEMORY_CAPACITY):
            raise R7Error("R7 native C1-Full Correction did not start empty at K=1024")
        settings = SimpleNamespace(flat=MEMORY_CAPACITY, stm=MEMORY_CAPACITY,
                                   ltm=0, buffer=0, min_events=2, promote_every=16,
                                   cap_bytes=2 * 1024**3)
        writer = MemoryArm("greedy_coreset", bank, settings, seed=0)
        if any(cb.size() != 0 or cb.K != MEMORY_CAPACITY for cb in dynamic_codebooks(bank)):
            raise R7Error("R7 shared Greedy normal bank did not start empty at K=1024")
        bank.corr_cfg.enabled = False
        bank.use_defect_bank = False
        frozen_digest = _fixed_bank_digest(bank)
        if frozen_digest != info["fixed_support_sha256_expected"]:
            raise R7Error("R7 structured fixed Support digest differs from the frozen receipt")

        c64_items, c64_cache_receipt = _make_c64_items(
            category, info, p0, impl, backbone, cfg)
        if len(c64_items) != 64:
            raise R7Error("R7 did not obtain all 64 protected C64 feature vectors")
        for index, item in enumerate(c64_items):
            if item.get("source_group_sha256") != info["c64_groups"][index]:
                raise R7Error("R7 C64 input ordering differs from its frozen Source Group identity")
        initial_g_c64, base_cache, c64_init_seconds = _score_c64_base(
            impl, bank, c64_items, cfg, r1, category)
        c64_reference = np.asarray(info["c64_scores"], dtype=np.float64)
        if any(not r3._score_tolerance(actual, expected)
               for actual, expected in zip(initial_g_c64, c64_reference)):
            raise R7Error("R7 native initial G C64 scores differ from the frozen score tape")
        source_groups: dict[int, set[str]] = {}
        initial_gc_c64, c64_gc_init_seconds = _score_c64_corrected(
            impl, bank, c64_items, base_cache, cfg, r1, source_groups, category,
            event_position=0)
        if any(not r3._score_tolerance(actual, expected)
               for actual, expected in zip(initial_gc_c64, initial_g_c64)):
            raise R7Error("R7 empty Correction initial GC C64 scores differ from G")
        if not math.isclose(float(np.quantile(initial_g_c64, 0.90, method="higher")),
                            float(info["theta_c64_q90"]), rel_tol=0.0, abs_tol=1e-12):
            raise R7Error("R7 initial C64 q90 threshold differs from the frozen receipt")
        theta0 = float(info["theta_c64_q90"])
        measured_sigma0 = max(float(np.std(c64_reference, ddof=1)), 0.02)
        sigma0 = float(info["sigma_c64_floor_002"])
        if not math.isclose(measured_sigma0, sigma0, rel_tol=0.0, abs_tol=1e-12):
            raise R7Error("R7 initial normal scale differs from the frozen C64 sigma receipt")
        state_digests = _state_digests(bank, source_groups, frozen_digest, r1)
        state_versions = {"G": 0, "GC": 0}
        c64_state_scores = {"G": list(initial_g_c64), "GC": list(initial_gc_c64)}
        controllers: dict[str, Any] = {
            "G-B1": RollingQ90QueriedNormal(theta0, window=256, min_count=16, name="G-B1"),
            "GC-B1": RollingQ90QueriedNormal(theta0, window=256, min_count=16, name="GC-B1"),
            "G-B3": StateConsistentQuantile(theta0, delta=0.01, min_normals=16, name="G-B3"),
            "GC-B3": StateConsistentQuantile(theta0, delta=0.01, min_normals=16, name="GC-B3"),
        }
        initial_b3 = {
            "G-B3": _initialize_b3(controllers["G-B3"], c64_state_scores["G"],
                                   info["frozen_info"]["c64"]["group_sha256"],
                                   state_digests["G"], theta0),
            "GC-B3": _initialize_b3(controllers["GC-B3"], c64_state_scores["GC"],
                                    info["frozen_info"]["c64"]["group_sha256"],
                                    state_digests["GC"], theta0),
        }
        tape_receipt = info["tape_receipt"]
        all_query_positions = tuple(int(value) for value in tape_receipt["positions"])
        event_limit = (min(EXPECTED_COUNTS[category], int(smoke_events))
                       if mode == "SMOKE_ONLY" else EXPECTED_COUNTS[category])
        allowed_queries = tuple(value for value in all_query_positions if value < event_limit)
        query_set = set(allowed_queries)
        reader = IndexedQueriedLabelReader(
            info["json_path"], category, source["category_json_sha256"],
            info["events"], allowed_queries)
        scores_out = BudgetJsonl(budget, "stage_a_scores.jsonl")
        score_digest = hashlib.sha256()
        feedback_digest = hashlib.sha256()
        query_group_digests: set[str] = set()
        score_gate = ScoreDisclosureGate(lambda row: scores_out.write(row, flush=True))
        counters = Counter()
        correction_stats: Counter[str] = Counter()
        correction_changes = 0
        first_greedy_change = None
        first_correction_change = None
        state_change_positions = {"G": [], "GC": []}
        threshold_transitions: list[dict[str, Any]] = []
        b1_threshold_history = {"G-B1": [], "GC-B1": []}
        b3_threshold_history = {"G-B3": [initial_b3["G-B3"]],
                               "GC-B3": [initial_b3["GC-B3"]]}
        c64_rounds = {"G": 1, "GC": 2}
        c64_score_calls = {"G": 64, "GC": 128}
        c64_seconds = {"G": c64_init_seconds, "GC": c64_gc_init_seconds}
        correction_score_differences: list[float] = []
        correction_update_diagnostics: list[dict[str, int]] = []
        correction_occupancy_history: list[dict[str, int]] = []
        greedy_occupancy_history: list[list[int]] = []
        correction_capacity_first_reached: int | None = None
        parity_checks = {"direct_native_reference": False,
                         "initial_g_c64_count": len(initial_g_c64),
                         "initial_gc_c64_count": len(initial_gc_c64),
                         "max_abs_g_c64_error": max(abs(a - b) for a, b in
                                                      zip(initial_g_c64, c64_reference)),
                         "max_abs_empty_gc_g_c64_error": max(abs(a - b) for a, b in
                                                               zip(initial_gc_c64, initial_g_c64))}
        torch.cuda.synchronize()
        setup_seconds = time.perf_counter() - setup_started
        for position, event in enumerate(info["events"][:event_limit]):
            causal = r1.CausalEventOrder(position)
            image_path = Path(event.image_path)
            if not image_path.is_file():
                raise R7Error(f"{category}:{position}: frozen event image disappeared")
            torch.cuda.synchronize()
            encoding_started = time.perf_counter()
            image, features = p0.encode_query(impl, backbone, str(image_path), cfg)
            torch.cuda.synchronize()
            counters["encoding_seconds"] += time.perf_counter() - encoding_started
            counters["actual_stream_image_encodings"] += 1
            if (len(features) != 1 or tuple(features[0].shape) != (784, 4096) or
                    features[0].dtype != torch.float32 or not torch.isfinite(features[0]).all().item()):
                raise R7Error(f"{category}:{position}: native fused descriptor contract failed")
            score_started = time.perf_counter()
            base_map, top_ids, base_score, summary = r1._native_dense_score(
                impl, bank, image, features, cfg)
            if (len(top_ids) != 32 or not np.array_equal(
                    top_ids, np.asarray(summary["layer0"]["topk_patch_idx"], dtype=np.int64))):
                raise R7Error(f"{category}:{position}: native Top-32 receipt mismatch")
            group_sha = event.source_group_sha256
            selected = impl.l2_normalize(features[0].to(bank.device))[
                torch.as_tensor(top_ids, device=bank.device, dtype=torch.long)]
            delta, correction_diag = r1._c1_score_delta(
                impl, bank, selected, source_groups, group_sha, event_position=position)
            corrected_map = r1.corrected_dense_map(base_map, top_ids, delta)
            corrected_score = float(r1.max_score(corrected_map))
            torch.cuda.synchronize()
            counters["scoring_seconds"] += time.perf_counter() - score_started
            base_score = float(base_score)
            correction_score_differences.append(corrected_score - base_score)
            counters["correction_active_events"] += int(corrected_score != base_score)
            counters["correction_positive_same_source_exclusions"] += int(
                correction_diag.get("same_source_excluded_anchor_slots", 0))
            if position == 0:
                reference_score, reference_ids, reference_map = r1._native_reference_and_debug(
                    impl, bank, image, features, cfg)
                if (not np.array_equal(reference_ids, top_ids) or
                        not np.array_equal(reference_map, base_map) or
                        not r3._score_tolerance(reference_score, base_score)):
                    raise R7Error("R7 shuffled event native map adapter failed direct reference parity")
                parity_checks["direct_native_reference"] = True
            score_map = {"G": base_score, "GC": corrected_score}
            state_version_by_scorer = {"G": state_versions["G"], "GC": state_versions["GC"]}
            state_digest_by_scorer = dict(state_digests)
            query_decision = position in query_set
            decision_predictions: dict[str, Any] = {}
            decisions: dict[str, dict[str, Any]] = {}
            for scorer, arm_frozen, arm_b1, arm_b3 in (
                    ("G", "G-Frozen", "G-B1", "G-B3"),
                    ("GC", "GC-Frozen", "GC-B1", "GC-B3")):
                frozen_prediction = int(score_map[scorer] >= theta0)
                decisions[arm_frozen] = {"prediction": frozen_prediction,
                                         "threshold": theta0, "controller_status": "FROZEN_C64_Q90",
                                         "scorer_state_version": state_versions[scorer],
                                         "scorer_state_sha256": state_digests[scorer]}
                observation = Observation(
                    t=position, observation_id=event.event_id_sha256,
                    camera_id=int(event.camera_id), score=score_map[scorer],
                    memory_version=state_versions[scorer],
                    score_state_digest=state_digests[scorer])
                b1 = controllers[arm_b1]
                b1.observe_score(observation)
                b1_prediction = b1.predict(observation)
                if b1_prediction.label is None or not b1_prediction.available:
                    raise R7Error("R7 B1 unexpectedly abstained on an available event")
                decisions[arm_b1] = {"prediction": int(b1_prediction.label),
                                     "threshold": float(b1_prediction.threshold),
                                     "controller_status": "ROLLING_QUERIED_NORMAL_Q90",
                                     "scorer_state_version": state_versions[scorer],
                                     "scorer_state_sha256": state_digests[scorer]}
                decision_predictions[arm_b1] = (observation, b1_prediction)
                b3 = controllers[arm_b3]
                b3_record = _predict_b3(
                    b3, position=position, event=event, score=score_map[scorer],
                    version=state_versions[scorer], state_digest=state_digests[scorer])
                decisions[arm_b3] = b3_record
                decision_predictions[arm_b3] = (observation, b3.predict(observation))
                decision_predictions[arm_frozen] = (observation, None)
            if set(decisions) != set(ARMS):
                raise R7Error("R7 did not create all six current event decisions")
            score_row = {
                "schema": SCORE_SCHEMA, "category": category,
                "position": position, "original_json_position": event.original_position,
                "event_id_sha256": event.event_id_sha256,
                "image_identity_sha256": event.image_identity_sha256,
                "source_group_sha256": group_sha,
                "camera_id": int(event.camera_id),
                "query_tape_decision": bool(query_decision),
                "query_tape_sha256": tape_receipt["tape_sha256"],
                "query_tape_positions_sha256": tape_receipt["positions_sha256"],
                "scores": {"G": base_score, "GC": corrected_score},
                "correction_score_delta": corrected_score - base_score,
                "decisions": decisions,
                "score_state_versions": state_version_by_scorer,
                "score_state_sha256": state_digest_by_scorer,
                "structured_fixed_support_sha256": frozen_digest,
                "native_c1_formula": "frozen C1-Full SignedCorrection; original native update/readout math",
            }
            writer_before = r1._memory_content_sha256(bank)
            audit_before = writer.audit()
            mutation = {"changed": False}

            def writer_update(*, position: int, feature_layers: Sequence[Any],
                              candidate_ids: Any, score: float, theta: float,
                              sigma: float) -> Mapping[str, Any]:
                before = r1._memory_content_sha256(bank)
                writer.update(impl, list(feature_layers), {0: candidate_ids.tolist()},
                              position, score=score, theta=theta, sigma=sigma)
                after = r1._memory_content_sha256(bank)
                mutation["changed"] = before != after
                after_audit = writer.audit()
                return {"accepted_patches_delta": int(after_audit["accepted_patches"] -
                                                       audit_before["accepted_patches"]),
                        "skipped_patches_delta": int(after_audit["skipped_patches"] -
                                                      audit_before["skipped_patches"]),
                        "_memory_content_changed": mutation["changed"]}

            candidate_features = selected.detach().float().cpu().numpy()
            causal.seal_scores()
            causal.decide_query(query_decision)
            score_gate.seal(position, score_row)
            score_digest.update(_canonical_bytes(score_row))
            label = None
            if query_decision:
                label = causal.disclose(lambda: reader.label_for(
                    position, sealed_and_fsynced=score_gate.is_sealed(position)))
                if group_sha in query_group_digests:
                    raise R7Error("R7 query tape repeated a Source Group")
                query_group_digests.add(group_sha)
                counters["query_events"] += 1
                before_digests = _state_digests(bank, source_groups, frozen_digest, r1)
                if before_digests != state_digests:
                    raise R7Error("R7 scorer state changed between event sealing and legal feedback")
                outcome = {(0, 0): "TN", (0, 1): "FP",
                           (1, 0): "FN", (1, 1): "TP"}[(label, int(base_score >= theta0))]
                sign = 1 if label == 1 else -1
                corr_started = time.perf_counter()
                _result, layer, correction_trace = exp2a._capture_update(
                    impl, bank, r1, "GC", bank.corr_cfg, correction_cb,
                    features, np.asarray(top_ids, dtype=np.int64), sign,
                    base_score, theta0, outcome, position, source_groups, group_sha)
                integer_stats = {key: int(value) for key, value in layer.items()
                                 if isinstance(value, (int, np.integer))}
                correction_stats.update(integer_stats)
                correction_update_diagnostics.append(integer_stats)
                correction_occupancy_history.append({
                    "position": position, "occupancy": int(correction_cb.size()),
                    "positive": int((correction_cb.beta[:correction_cb.size()] > 0).sum().item()),
                    "negative": int((correction_cb.beta[:correction_cb.size()] < 0).sum().item()),
                })
                if (int(correction_cb.size()) == MEMORY_CAPACITY and
                        correction_capacity_first_reached is None):
                    correction_capacity_first_reached = position
                counters["correction_updates"] += 1
                correction_changed = before_digests["GC"] != _state_digests(
                    bank, source_groups, frozen_digest, r1)["GC"]
                if correction_changed:
                    counters["correction_state_changes"] += 1
                    if first_correction_change is None:
                        first_correction_change = position
                counters["correction_update_seconds"] += time.perf_counter() - corr_started
                causal.corrections_updated()
                writer_action = r1.common_c0_greedy_writer_action(
                    position=position, c0_score=base_score, theta=theta0,
                    sigma=sigma0, c0_prediction=int(base_score >= theta0),
                    candidate_ids=np.asarray(top_ids, dtype=np.int64),
                    candidate_features=candidate_features,
                    feature_layers=features, queried=True, disclosed_label=label,
                    eligibility=impl.select_informative_normal_write,
                    update=writer_update)
                writer_action.pop("_memory_content_changed", None)
                counters["greedy_write_events"] += int(
                    writer_action.get("action") == "native_greedy_normal_update")
                writer_after = r1._memory_content_sha256(bank)
                changed_greedy = writer_before != writer_after
                if changed_greedy:
                    counters["greedy_memory_changes"] += 1
                    if first_greedy_change is None:
                        first_greedy_change = position
                writer_action["normal_memory_content_changed"] = changed_greedy
                if label == 1 and writer_action.get("action") != "verified_defect_no_normal_write":
                    raise R7Error("Queried defect entered normal memory or was not rejected by the common writer")
                if not query_decision and writer_action.get("action") != "no_normal_memory_write":
                    raise R7Error("Unqueried event caused a supervised Greedy write")
                after_digests = _state_digests(bank, source_groups, frozen_digest, r1)
                changed_scorers = [name for name in ("G", "GC")
                                   if after_digests[name] != state_digests[name]]
                if changed_greedy and "G" not in changed_scorers:
                    raise R7Error("Greedy memory mutation did not change G scorer identity")
                if correction_changed and "GC" not in changed_scorers:
                    raise R7Error("Correction mutation did not change GC scorer identity")
                if "G" in changed_scorers:
                    state_change_positions["G"].append(position)
                    state_versions["G"] += 1
                    c64_state_scores["G"], base_cache, elapsed = _score_c64_base(
                        impl, bank, c64_items, cfg, r1, category)
                    c64_seconds["G"] += elapsed
                    c64_rounds["G"] += 1
                    c64_score_calls["G"] += 64
                if "GC" in changed_scorers:
                    state_change_positions["GC"].append(position)
                    state_versions["GC"] += 1
                    if "G" in changed_scorers:
                        # The freshly rescored base-cache binds GC to the same
                        # common Greedy state as G before adding C1 readout.
                        pass
                    c64_state_scores["GC"], elapsed = _score_c64_corrected(
                        impl, bank, c64_items, base_cache, cfg, r1,
                        source_groups, category, event_position=position + 1)
                    c64_seconds["GC"] += elapsed
                    c64_rounds["GC"] += 1
                    c64_score_calls["GC"] += 64
                transitions = {}
                for scorer, controller_key in (("G", "G-B3"), ("GC", "GC-B3")):
                    if scorer in changed_scorers:
                        transition = _refresh_b3(
                            controllers[controller_key], c64_state_scores[scorer],
                            info["frozen_info"]["c64"]["group_sha256"],
                            state_digests[scorer], after_digests[scorer],
                            state_versions[scorer] - 1)
                        if transition is None:
                            raise R7Error("R7 B3 missed a changed complete scorer state")
                        transition["apply_from_position"] = position + 1
                        transitions[scorer] = transition
                        b3_threshold_history[controller_key].append(transition)
                state_digests = after_digests
                for scorer, controller_key in (("G", "G-B1"), ("GC", "GC-B1")):
                    observation, prediction = decision_predictions[controller_key]
                    controllers[controller_key].observe_feedback(
                        observation, prediction, Feedback(t=position, queried=True, label=label))
                    theta_next = float(controllers[controller_key].finalize_step())
                    b1_threshold_history[controller_key].append({
                        "position": position, "apply_from_position": position + 1,
                        "threshold": theta_next,
                        "normal_query_count": len(controllers[controller_key].history)})
                for scorer, controller_key in (("G", "G-B3"), ("GC", "GC-B3")):
                    observation, prediction = decision_predictions[controller_key]
                    controllers[controller_key].observe_feedback(
                        observation, prediction, Feedback(t=position, queried=True, label=label))
                memory_audit = writer.audit()
                counters["greedy_accepted_patches"] += int(writer_action.get("accepted_patches_delta", 0))
                counters["greedy_skipped_patches"] += int(writer_action.get("skipped_patches_delta", 0))
                greedy_occupancy_history.append([int(cb.size()) for cb in dynamic_codebooks(bank)])
                feedback_row = {
                    "schema": "r7.fixed_shuffle.private_feedback.v1",
                    "record_type": "queried_feedback_and_updates",
                    "category": category, "position": position,
                    "original_json_position": event.original_position,
                    "event_id_sha256": event.event_id_sha256,
                    "source_group_sha256": group_sha,
                    "score_row_fsynced_before_disclosure": score_gate.is_sealed(position),
                    "label": int(label),
                    "frozen_writer_outcome_from_G_C0": outcome,
                    "correction_update": {"sign": sign, "native_stats": integer_stats,
                                          "occupancy_after": int(correction_cb.size()),
                                          "trace": {key: int(value) for key, value in
                                                    correction_trace.items()
                                                    if isinstance(value, (int, np.integer))}},
                    "common_greedy_writer_action": writer_action,
                    "greedy_state_changed": changed_greedy,
                    "correction_state_changed": correction_changed,
                    "scorer_states_changed": changed_scorers,
                    "state_versions_after": dict(state_versions),
                    "state_digests_after": dict(state_digests),
                    "b1_thresholds_for_next_event": {
                        key: float(controllers[key].theta) for key in ("G-B1", "GC-B1")},
                    "b3_transitions_for_next_event": transitions,
                }
                payload = _canonical_bytes(feedback_row)
                budget.append_jsonl("attempt_history.jsonl", feedback_row)
                feedback_digest.update(payload)
                causal.memory_updated()
                causal.finish()
                counters["query_group_unique"] = len(query_group_digests)
            else:
                # No feedback is supplied to either supervised writer.
                causal.corrections_updated()
                writer_action = r1.common_c0_greedy_writer_action(
                    position=position, c0_score=base_score, theta=theta0,
                    sigma=sigma0, c0_prediction=int(base_score >= theta0),
                    candidate_ids=np.asarray(top_ids, dtype=np.int64),
                    candidate_features=candidate_features, feature_layers=features,
                    queried=False, disclosed_label=None,
                    eligibility=impl.select_informative_normal_write,
                    update=writer_update)
                writer_action["normal_memory_content_changed"] = False
                if writer_action.get("action") != "no_normal_memory_write":
                    raise R7Error("Unqueried R7 event performed a supervised Greedy writer action")
                causal.memory_updated()
                causal.finish()
            scores_out.pending.clear()
            if query_decision:
                thresholds = {key: float(value.theta) for key, value in controllers.items()}
                transition_receipts = []
                # Controller state changes are already in the feedback record;
                # only the next-event scalar thresholds are retained here.
                for key in ("G-B1", "GC-B1", "G-B3", "GC-B3"):
                    transition_receipts.append({"controller": key, "theta_next": thresholds[key]})
                _append_history(budget, {
                    "schema": "r7.fixed_shuffle.history.v1",
                    "record_type": "post_feedback_progress",
                    "category": category, "position": position,
                    "thresholds_for_next_event": transition_receipts,
                    "scorer_state_versions": dict(state_versions),
                    "automatic_restart": False,
                })
            if (position + 1) % 256 == 0:
                _append_history(budget, {
                    "schema": "r7.fixed_shuffle.history.v1",
                    "record_type": "diagnostic_progress_checkpoint",
                    "category": category, "sealed_through_position": position,
                    "score_rows_written": position + 1,
                    "query_feedback_count": int(counters["query_events"]),
                    "scorer_state_versions": dict(state_versions),
                    "scorer_state_sha256": dict(state_digests),
                    "automatic_restart": False,
                })
        scores_out.flush()
        reader.finish()
        reader.close()
        reader = None
        if int(counters["actual_stream_image_encodings"]) != event_limit:
            raise R7Error("R7 Stage A did not encode each scored event exactly once")
        actual_queries = int(counters["query_events"])
        if actual_queries != len(allowed_queries):
            raise R7Error("R7 Stage A queried feedback count differs from its fixed tape prefix")
        if mode == "FULL_STAGE_A" and actual_queries != EXPECTED_QUERIES[category]:
            raise R7Error("R7 full Stage A did not consume the exact shuffled Q-A-TS tape")
        final_fixed_digest = _fixed_bank_digest(bank)
        if final_fixed_digest != info["fixed_support_sha256_expected"]:
            raise R7Error("R7 mutable writer altered the frozen fixed Support bank")
        writer_audit = writer.audit()
        correction_profile = _mutable_state_bytes(bank, correction_cb, dynamic_codebooks)
        b1_update_summary = {}
        for key in ("G-B1", "GC-B1"):
            history = b1_threshold_history[key]
            prior_theta = theta0
            threshold_movements = 0
            for item in history:
                next_theta = float(item["threshold"])
                threshold_movements += int(next_theta != prior_theta)
                prior_theta = next_theta
            b1_update_summary[key] = {
                "queried_events_observed": len(history),
                "verified_normal_updates": int(history[-1]["normal_query_count"]) if history else 0,
                "threshold_movements": threshold_movements,
                "final_threshold": prior_theta,
            }
        b3_update_summary = {
            key: {
                "complete_scorer_state_updates": max(0, len(history) - 1),
                "threshold_movements": sum(
                    float(item.get("movement", 0.0)) != 0.0 for item in history[1:]),
                "final_threshold": float(history[-1]["theta"] if "theta" in history[-1]
                                         else history[-1]["new_theta"]),
            }
            for key, history in b3_threshold_history.items()
        }
        cat_receipt = {
            "schema": "r7.fixed_shuffle.category_complete.v1",
            "record_type": "category_complete", "category": category,
            "mode": mode, "status": "SEALED_COMPLETE",
            "events_processed": event_limit, "query_events_processed": actual_queries,
            "query_positions": list(allowed_queries),
            "query_tape_sha256": tape_receipt["tape_sha256"],
            "query_positions_sha256": tape_receipt["positions_sha256"],
            "canonical_score_rows_sha256": score_digest.hexdigest(),
            "query_feedback_receipts_sha256": feedback_digest.hexdigest(),
            "structured_fixed_support_sha256": final_fixed_digest,
            "fixed_support_receipt": fixed_receipt,
            "actual_stream_image_encodings": int(counters["actual_stream_image_encodings"]),
            "first_greedy_memory_change_position": first_greedy_change,
            "first_correction_state_change_position": first_correction_change,
            "greedy_write_events": int(counters["greedy_write_events"]),
            "greedy_memory_changes": int(counters["greedy_memory_changes"]),
            "greedy_accepted_patches": int(counters["greedy_accepted_patches"]),
            "greedy_skipped_patches": int(counters["greedy_skipped_patches"]),
            "greedy_writer_audit": writer_audit,
            "correction_update_count": int(counters["correction_updates"]),
            "correction_state_change_count": int(counters["correction_state_changes"]),
            "correction_update_stats": dict(correction_stats),
            "correction_final_occupancy": int(correction_cb.size()),
            "correction_capacity_first_reached_position": correction_capacity_first_reached,
            "correction_score_activation_count": int(counters["correction_active_events"]),
            "correction_score_difference_summary": _distribution_summary(correction_score_differences),
            "correction_state_occupancy_history": correction_occupancy_history,
            "greedy_occupancy_history": greedy_occupancy_history,
            "mutable_state_bytes": correction_profile,
            "scorer_state_versions_final": dict(state_versions),
            "scorer_state_sha256_final": dict(state_digests),
            "state_change_positions": state_change_positions,
            "b1_threshold_history": b1_threshold_history,
            "b3_threshold_history": b3_threshold_history,
            "b1_update_summary": b1_update_summary,
            "b3_update_summary": b3_update_summary,
            "initial_b3": initial_b3,
            "c64_rescore_rounds": c64_rounds,
            "c64_rescore_score_calls": c64_score_calls,
            "c64_rescore_seconds": c64_seconds,
            "c64_cache_receipt": c64_cache_receipt,
            "c64_initial_native_score_parity": parity_checks,
            "runtime_profile": runtime_info.get("runtime_profile"),
            "hub_source_tree_sha256": runtime_info.get("hub_source_tree_sha256"),
            "checkpoint_sha256": runtime_info.get("checkpoint_sha256"),
            "loaded_model_state_sha256": loaded_model_sha,
            "feature_encoding_per_test_event": 1,
            "stream_encoding_seconds": float(counters["encoding_seconds"]),
            "scoring_seconds": float(counters["scoring_seconds"]),
            "correction_update_seconds": float(counters["correction_update_seconds"]),
            "setup_seconds": setup_seconds,
            "wall_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_cpu_rss_bytes": r1._process_rss_bytes(),
            "stage_a_label_access": {"only_queried_shuffled_positions": True,
                                      "decoded_shuffled_positions": list(reader.decoded_positions)
                                      if reader else list(allowed_queries),
                                      "unqueried_test_labels_accessed": False,
                                      "offline_outcome_vectors_accessed": False,
                                      "test_masks_opened": False},
            "stage_a_score_rows_have_no_outcomes": True,
            "automatic_restart": False,
        }
        _append_history(budget, cat_receipt)
        return cat_receipt
    finally:
        if reader is not None:
            reader.close()
        if backbone is not None:
            del backbone
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _distribution_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    import numpy as np
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "median": None,
                "q05": None, "q95": None, "min": None, "max": None,
                "positive_fraction": None, "negative_fraction": None,
                "nonzero_fraction": None}
    return {"count": int(len(array)), "mean": float(array.mean()),
            "median": float(np.median(array)),
            "q05": float(np.quantile(array, .05)),
            "q95": float(np.quantile(array, .95)),
            "min": float(array.min()), "max": float(array.max()),
            "positive_fraction": float(np.mean(array > 0)),
            "negative_fraction": float(np.mean(array < 0)),
            "nonzero_fraction": float(np.mean(array != 0))}


def _rows_by_category(output: Path, *, require_full: bool = True) -> dict[str, list[dict[str, Any]]]:
    rows_by_category = {category: [] for category in CATEGORIES}
    scores_path = output / "stage_a_scores.jsonl"
    if not scores_path.is_file():
        raise R7Error("R7 Stage A score trajectory is missing")
    with scores_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get("schema") != SCORE_SCHEMA or row.get("category") not in rows_by_category or
                    any(key in row for key in ("label", "anomaly_class", "mask_path", "ground_truth"))):
                raise R7Error(f"R7 Stage A score row {line_no} violates its schema or label firewall")
            rows_by_category[str(row["category"])].append(row)
    for category, rows in rows_by_category.items():
        positions = [int(row.get("position", -1)) for row in rows]
        expected = EXPECTED_COUNTS[category] if require_full else len(rows)
        if positions != list(range(expected)):
            raise R7Error(f"{category}: R7 Stage A trajectory is not contiguous in shuffled order")
        if any(row.get("category") != category for row in rows):
            raise R7Error(f"{category}: R7 Stage A category identity failed")
    return rows_by_category


def _verify_category_receipt(category: str, rows: Sequence[Mapping[str, Any]],
                             completion: Mapping[str, Any], feedback: Sequence[Mapping[str, Any]],
                             info: Mapping[str, Any]) -> dict[str, Any]:
    if (completion.get("record_type") != "category_complete" or
            completion.get("status") != "SEALED_COMPLETE" or
            completion.get("category") != category or
            completion.get("mode") != "FULL_STAGE_A" or
            len(rows) != EXPECTED_COUNTS[category] or
            completion.get("events_processed") != EXPECTED_COUNTS[category] or
            completion.get("actual_stream_image_encodings") != EXPECTED_COUNTS[category]):
        raise R7Error(f"{category}: Stage A category completion receipt is incomplete")
    expected_queries = tuple(int(value) for value in info["tape_receipt"]["positions"])
    actual_queries = tuple(int(row["position"]) for row in rows if row["query_tape_decision"] is True)
    if (actual_queries != expected_queries or
            tuple(int(x) for x in completion.get("query_positions", [])) != expected_queries or
            len(expected_queries) != EXPECTED_QUERIES[category]):
        raise R7Error(f"{category}: Stage A did not execute the regenerated shuffled query tape")
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_bytes(row))
        if (row.get("query_tape_sha256") != info["tape_receipt"]["tape_sha256"] or
                row.get("query_tape_positions_sha256") != info["tape_receipt"]["positions_sha256"] or
                row.get("structured_fixed_support_sha256") != info["fixed_support_sha256_expected"] or
                set(row.get("scores", {})) != {"G", "GC"} or
                set(row.get("decisions", {})) != set(ARMS)):
            raise R7Error(f"{category}:{row.get('position')}: score row identity is incomplete")
        for arm, decision in row["decisions"].items():
            score = float(row["scores"]["G" if arm.startswith("G-") else "GC"])
            threshold = float(decision["threshold"])
            if int(decision["prediction"]) != int(score >= threshold):
                raise R7Error(f"{category}:{row['position']}:{arm}: sealed prediction/threshold mismatch")
    if digest.hexdigest() != completion.get("canonical_score_rows_sha256"):
        raise R7Error(f"{category}: score-row digest differs from its category completion receipt")
    feedback_rows = [row for row in feedback
                     if row.get("record_type") == "queried_feedback_and_updates" and
                     row.get("category") == category]
    if tuple(int(row.get("position", -1)) for row in feedback_rows) != expected_queries:
        raise R7Error(f"{category}: queried feedback receipt order differs from Q-A-TS")
    feedback_digest = hashlib.sha256()
    seen_groups = set()
    for receipt in feedback_rows:
        position = int(receipt["position"])
        score_row = rows[position]
        label = int(receipt.get("label", -1))
        if (label not in (0, 1) or receipt.get("score_row_fsynced_before_disclosure") is not True or
                receipt.get("event_id_sha256") != score_row.get("event_id_sha256") or
                receipt.get("source_group_sha256") != score_row.get("source_group_sha256") or
                score_row.get("query_tape_decision") is not True or
                receipt.get("source_group_sha256") in seen_groups):
            raise R7Error(f"{category}:{position}: Stage A feedback was not legal or identity-bound")
        seen_groups.add(receipt["source_group_sha256"])
        prediction = int(score_row["decisions"]["G-Frozen"]["prediction"])
        expected_outcome = {(0, 0): "TN", (0, 1): "FP",
                            (1, 0): "FN", (1, 1): "TP"}[(label, prediction)]
        if (receipt.get("frozen_writer_outcome_from_G_C0") != expected_outcome or
                receipt.get("correction_update", {}).get("sign") != (1 if label else -1)):
            raise R7Error(f"{category}:{position}: C1 update did not use legal feedback and frozen C0 outcome")
        writer = receipt.get("common_greedy_writer_action", {})
        if (writer.get("input_score_source") != "common_C0" or
                float(writer.get("score", math.nan)) != float(score_row["scores"]["G"]) or
                float(writer.get("theta", math.nan)) != float(info["theta_c64_q90"]) or
                int(writer.get("prediction", -1)) != prediction or
                (label == 1 and (writer.get("action") != "verified_defect_no_normal_write" or
                                 writer.get("normal_memory_content_changed") is True))):
            raise R7Error(f"{category}:{position}: Greedy writer was not controlled by common frozen C0")
        feedback_digest.update(_canonical_bytes(receipt))
    if (feedback_digest.hexdigest() != completion.get("query_feedback_receipts_sha256") or
            len(feedback_rows) != EXPECTED_QUERIES[category] or
            completion.get("query_events_processed") != len(feedback_rows) or
            completion.get("stage_a_label_access", {}).get("unqueried_test_labels_accessed") is not False or
            completion.get("stage_a_label_access", {}).get("test_masks_opened") is not False):
        raise R7Error(f"{category}: Stage A feedback digest or information-flow receipt failed")
    return {
        "event_count": len(rows), "query_count": len(feedback_rows),
        "query_tape_sha256": info["tape_receipt"]["tape_sha256"],
        "query_positions_sha256": info["tape_receipt"]["positions_sha256"],
        "canonical_score_rows_sha256": digest.hexdigest(),
        "query_feedback_receipts_sha256": feedback_digest.hexdigest(),
        "structured_fixed_support_sha256": completion["structured_fixed_support_sha256"],
        "greedy_write_events": completion["greedy_write_events"],
        "greedy_accepted_patches": completion["greedy_accepted_patches"],
        "greedy_skipped_patches": completion["greedy_skipped_patches"],
        "correction_final_occupancy": completion["correction_final_occupancy"],
        "correction_update_stats": completion["correction_update_stats"],
        "actual_stream_image_encodings": completion["actual_stream_image_encodings"],
        "runtime_receipt_sha256": _sha256_json({
            key: completion.get(key) for key in (
                "runtime_profile", "hub_source_tree_sha256", "checkpoint_sha256",
                "loaded_model_state_sha256", "peak_cuda_allocated_bytes",
                "peak_cuda_reserved_bytes", "peak_cpu_rss_bytes", "wall_seconds")}),
    }


def _seal_stage_a(output: Path, contract: Mapping[str, Any],
                  identity: Mapping[str, Any]) -> dict[str, Any]:
    rows_by_category = _rows_by_category(output, require_full=True)
    history = _read_history(output)
    completions = [row for row in history if row.get("record_type") == "category_complete"]
    by_category: dict[str, Mapping[str, Any]] = {}
    for row in completions:
        category = str(row.get("category"))
        if category in by_category:
            raise R7Error(f"{category}: duplicate Stage A completion receipt")
        by_category[category] = row
    if set(by_category) != set(CATEGORIES):
        raise R7Error("R7 Stage A requires exactly four completed category receipts")
    category_seals = {}
    total_queries = 0
    for category in CATEGORIES:
        category_seals[category] = _verify_category_receipt(
            category, rows_by_category[category], by_category[category], history,
            contract["metadata"][category])
        total_queries += int(category_seals[category]["query_count"])
    score_path = output / "stage_a_scores.jsonl"
    seal = {
        "schema": STAGE_A_SCHEMA, "status": "SEALED_COMPLETE",
        "run_id": RUN_ID, "source_parent_commit": EXPECTED_PARENT,
        "identity_sha256": identity["identity_sha256"],
        "category_event_counts": dict(EXPECTED_COUNTS),
        "category_query_counts": dict(EXPECTED_QUERIES),
        "category_seals": category_seals,
        "event_count": sum(len(rows_by_category[name]) for name in CATEGORIES),
        "query_count": total_queries,
        "score_rows_sha256": _sha256_file(score_path),
        "feedback_disclosure_order_verified": True,
        "every_event_score_row_fsynced_before_any_query_disclosure": True,
        "unqueried_test_labels_accessed": False,
        "stage_b_outcome_vectors_accessed": False,
        "test_masks_opened": False,
        "classification": "RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC",
        "automatic_restart": False,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if seal["event_count"] != TOTAL_EVENTS or total_queries != TOTAL_QUERIES:
        raise R7Error("R7 Stage A total event or query budget differs from the frozen four-class contract")
    _budget(output).write_json("stage_a_seal.json", seal)
    return seal


def _binary_metrics(labels: Any, predictions: Any, weights: Any | None = None) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(predictions, dtype=np.int8)
    w = np.ones(len(y), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if (y.ndim != 1 or p.shape != y.shape or w.shape != y.shape or
            np.any((y != 0) & (y != 1)) or np.any((p != 0) & (p != 1)) or
            not np.isfinite(w).all() or np.any(w < 0)):
        raise R7Error("Stage B binary metric inputs are malformed")
    tn = float(w[(y == 0) & (p == 0)].sum())
    fp = float(w[(y == 0) & (p == 1)].sum())
    fn = float(w[(y == 1) & (p == 0)].sum())
    tp = float(w[(y == 1) & (p == 1)].sum())
    fpr = fp / (fp + tn) if fp + tn else None
    fnr = fn / (fn + tp) if fn + tp else None
    ba = 1 - (fpr + fnr) / 2 if fpr is not None and fnr is not None else None
    total = tn + fp + fn + tp
    return {"confusion": {"tn": int(round(tn)), "fp": int(round(fp)),
                          "fn": int(round(fn)), "tp": int(round(tp))},
            "weighted_confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "fpr": fpr, "fnr": fnr, "balanced_accuracy": ba,
            "accuracy": (tn + tp) / total if total else None}


def _full_metrics(scores: Any, labels: Any, predictions: Any) -> dict[str, Any]:
    import numpy as np
    from experiments.calibration import r3_fullstream_2x2 as r3
    score = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    pred = np.asarray(predictions, dtype=np.int8)
    if (score.shape != y.shape or pred.shape != y.shape or
            not np.isfinite(score).all()):
        raise R7Error("Stage B score, outcome and prediction arrays are not aligned")
    return {**r3._weighted_rank_metrics(score, y),
            **_binary_metrics(y, pred),
            "prediction_coverage": float(np.mean(np.isin(pred, (0, 1))))}


def _worst_512_metrics(labels: Any, predictions: Any, window: int = 512) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(predictions, dtype=np.int8)
    if y.shape != p.shape or len(y) < window:
        return {"window_events": window, "stride": 1, "complete_windows": 0,
                "worst_balanced_accuracy": None, "maximum_fpr": None,
                "maximum_fnr": None}
    prefix = {}
    for name, values in {"tn": ((y == 0) & (p == 0)),
                         "fp": ((y == 0) & (p == 1)),
                         "fn": ((y == 1) & (p == 0)),
                         "tp": ((y == 1) & (p == 1))}.items():
        prefix[name] = np.r_[0, np.cumsum(values, dtype=np.int64)]
    totals = {name: arr[window:] - arr[:-window] for name, arr in prefix.items()}
    neg = totals["tn"] + totals["fp"]
    pos = totals["fn"] + totals["tp"]
    fpr = np.divide(totals["fp"], neg, out=np.full(len(neg), np.nan), where=neg > 0)
    fnr = np.divide(totals["fn"], pos, out=np.full(len(pos), np.nan), where=pos > 0)
    valid = np.isfinite(fpr) & np.isfinite(fnr)
    ba = 1.0 - (fpr[valid] + fnr[valid]) / 2.0
    return {"window_events": window, "stride": 1,
            "complete_windows": int(len(neg)), "windows_with_both_classes": int(valid.sum()),
            "worst_balanced_accuracy": float(ba.min()) if len(ba) else None,
            "maximum_fpr": float(np.nanmax(fpr)) if np.isfinite(fpr).any() else None,
            "maximum_fnr": float(np.nanmax(fnr)) if np.isfinite(fnr).any() else None,
            "minimum_tnr": float(1 - np.nanmax(fpr)) if np.isfinite(fpr).any() else None,
            "minimum_tpr": float(1 - np.nanmax(fnr)) if np.isfinite(fnr).any() else None}


def _paired_group_bootstrap(scores: Mapping[str, Any], predictions: Mapping[str, Any],
                            labels: Any, groups: Any, *, seed: int,
                            replicates: int = BOOTSTRAP_REPLICATES
                            ) -> dict[str, Any]:
    import numpy as np
    from experiments.calibration import r3_fullstream_2x2 as r3
    y = np.asarray(labels, dtype=np.int8)
    group_values, inverse = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    if len(group_values) < 2:
        raise R7Error("Source Group bootstrap requires at least two independent groups")
    rng = np.random.default_rng(int(seed))
    effects = ("auroc_GC_minus_G", "average_precision_GC_minus_G",
               "balanced_accuracy_GC_minus_G_Frozen",
               "balanced_accuracy_GC_minus_G_B1",
               "balanced_accuracy_GC_minus_G_B3",
               "fpr_GC_minus_G_Frozen", "fpr_GC_minus_G_B1", "fpr_GC_minus_G_B3",
               "fnr_GC_minus_G_Frozen", "fnr_GC_minus_G_B1", "fnr_GC_minus_G_B3")
    draws = {name: np.full(int(replicates), np.nan, dtype=np.float64) for name in effects}
    for rep in range(int(replicates)):
        sampled = rng.integers(0, len(group_values), size=len(group_values))
        cluster_counts = np.bincount(sampled, minlength=len(group_values)).astype(np.float64)
        weights = cluster_counts[inverse]
        rank_g = r3._weighted_rank_metrics(scores["G"], y, weights)
        rank_gc = r3._weighted_rank_metrics(scores["GC"], y, weights)
        if rank_g["auroc"] is not None and rank_gc["auroc"] is not None:
            draws["auroc_GC_minus_G"][rep] = float(rank_gc["auroc"] - rank_g["auroc"])
        if rank_g["average_precision"] is not None and rank_gc["average_precision"] is not None:
            draws["average_precision_GC_minus_G"][rep] = float(
                rank_gc["average_precision"] - rank_g["average_precision"])
        for calibration, suffix in (("Frozen", "Frozen"), ("B1", "B1"), ("B3", "B3")):
            g = _binary_metrics(y, predictions[f"G-{calibration}"], weights)
            gc = _binary_metrics(y, predictions[f"GC-{calibration}"], weights)
            for metric in ("balanced_accuracy", "fpr", "fnr"):
                left, right = gc[metric], g[metric]
                if left is not None and right is not None:
                    draws[f"{metric}_GC_minus_G_{suffix}"][rep] = float(left - right)
    output = {name: {"valid_replicates": int(np.isfinite(values).sum()),
                     "ci_95_percentile": ([float(np.nanquantile(values, .025)),
                                           float(np.nanquantile(values, .975))]
                                          if np.isfinite(values).any() else None)}
              for name, values in draws.items()}
    return {"method": "paired Source Group cluster bootstrap; percentile 95% interval",
            "replicates_requested": int(replicates), "seed": int(seed),
            "source_group_count": int(len(group_values)),
            "same_sampled_groups_for_all_scorers_and_controllers": True,
            "contrasts": output, "_draws": draws}


def _point_contrasts(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for metric, name in (("auroc", "auroc_GC_minus_G"),
                         ("average_precision", "average_precision_GC_minus_G")):
        left, right = metrics["GC-Frozen"].get(metric), metrics["G-Frozen"].get(metric)
        output[name] = None if left is None or right is None else float(left - right)
    for calibration in ("Frozen", "B1", "B3"):
        for metric in ("balanced_accuracy", "fpr", "fnr"):
            left = metrics[f"GC-{calibration}"].get(metric)
            right = metrics[f"G-{calibration}"].get(metric)
            output[f"{metric}_GC_minus_G_{calibration}"] = (
                None if left is None or right is None else float(left - right))
    return output


def _third_slices(n: int) -> list[tuple[str, int, int]]:
    bounds = [0, n // 3, (2 * n) // 3, n]
    return [("early", bounds[0], bounds[1]),
            ("middle", bounds[1], bounds[2]),
            ("late", bounds[2], bounds[3])]


def _category_stage_b(category: str, rows: Sequence[Mapping[str, Any]], labels: Any,
                      events: Sequence[ShuffledEvent], tape_receipt: Mapping[str, Any],
                      completion: Mapping[str, Any], history: Sequence[Mapping[str, Any]],
                      seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    n = EXPECTED_COUNTS[category]
    if (len(rows) != n or y.shape != (n,) or
            [int(row["position"]) for row in rows] != list(range(n)) or
            np.any((y != 0) & (y != 1))):
        raise R7Error(f"{category}: Stage B outcome, shuffled map or score trajectory is invalid")
    positions = tuple(int(value) for value in tape_receipt["positions"])
    if len(positions) != EXPECTED_QUERIES[category]:
        raise R7Error(f"{category}: Stage B tape budget differs from Stage A")
    feedback = [row for row in history
                if row.get("record_type") == "queried_feedback_and_updates" and
                row.get("category") == category]
    if tuple(int(row["position"]) for row in feedback) != positions:
        raise R7Error(f"{category}: Stage B query feedback does not follow the sealed tape")
    for row in rows:
        position = int(row["position"])
        event = events[position]
        if (int(row.get("original_json_position", -1)) != event.original_position or
                row.get("event_id_sha256") != event.event_id_sha256 or
                row.get("image_identity_sha256") != event.image_identity_sha256 or
                row.get("source_group_sha256") != event.source_group_sha256 or
                bool(row.get("query_tape_decision")) != (position in set(positions))):
            raise R7Error(f"{category}:{position}: Stage B shuffled event identity is not bound to Stage A")
    for record in feedback:
        position = int(record["position"])
        if int(record["label"]) != int(y[position]):
            raise R7Error(f"{category}:{position}: queried label differs from sealed Stage B outcome")
    scores = {
        scorer: np.asarray([float(row["scores"][scorer]) for row in rows], dtype=np.float64)
        for scorer in ("G", "GC")}
    predictions = {
        arm: np.asarray([int(row["decisions"][arm]["prediction"]) for row in rows], dtype=np.int8)
        for arm in ARMS}
    thresholds = {
        arm: np.asarray([float(row["decisions"][arm]["threshold"]) for row in rows], dtype=np.float64)
        for arm in ARMS}
    for arm in ARMS:
        if not np.array_equal(predictions[arm], (scores["G" if arm.startswith("G-") else "GC"] >=
                                                thresholds[arm]).astype(np.int8)):
            raise R7Error(f"{category}:{arm}: Stage A decision is not the sealed threshold comparison")
    metrics = {}
    for arm in ARMS:
        scorer = "G" if arm.startswith("G-") else "GC"
        metrics[arm] = _full_metrics(scores[scorer], y, predictions[arm])
        metrics[arm]["threshold_summary"] = _distribution_summary(thresholds[arm])
        metrics[arm]["query_count"] = len(positions)
        metrics[arm]["query_rate"] = len(positions) / n
        metrics[arm]["worst_complete_512_event_window"] = _worst_512_metrics(y, predictions[arm])
    for scorer in ("G", "GC"):
        reference = metrics[f"{scorer}-Frozen"]
        for arm in (f"{scorer}-B1", f"{scorer}-B3"):
            for key in ("auroc", "average_precision"):
                a, b = metrics[arm][key], reference[key]
                if a != b and (a is None or b is None or abs(float(a) - float(b)) > 1e-15):
                    raise R7Error(f"{category}:{arm}: controller changed the sealed ranking score")
    contrasts = _point_contrasts(metrics)
    groups = np.asarray([event.source_group_sha256 for event in events], dtype=str)
    bootstrap = _paired_group_bootstrap(scores, predictions, y, groups, seed=seed)
    query_labels = y[np.asarray(positions, dtype=np.int64)]
    temporal = []
    for name, begin, end in _third_slices(n):
        section_y = y[begin:end]
        section_scores = {key: value[begin:end] for key, value in scores.items()}
        section_predictions = {key: value[begin:end] for key, value in predictions.items()}
        section_metrics = {}
        for arm in ARMS:
            scorer = "G" if arm.startswith("G-") else "GC"
            section_metrics[arm] = _full_metrics(
                section_scores[scorer], section_y, section_predictions[arm])
        q_in = [int(pos) for pos in positions if begin <= pos < end]
        q_labels = [int(y[pos]) for pos in q_in]
        temporal.append({
            "shuffled_stream_region": name,
            "start_position_inclusive": begin, "end_position_exclusive": end,
            "event_count": end - begin,
            "normal_events": int((section_y == 0).sum()),
            "defect_events": int((section_y == 1).sum()),
            "query_count": len(q_in), "query_normal_feedback": q_labels.count(0),
            "query_defect_feedback": q_labels.count(1),
            "metrics_by_arm": section_metrics,
            "GC_minus_G": _point_contrasts(section_metrics),
        })
    query_coverage = []
    for fraction in (0.25, 0.50, 0.75, 1.0):
        stop = int(math.ceil(n * fraction))
        count = sum(position < stop for position in positions)
        query_coverage.append({"shuffled_stream_fraction": fraction,
                               "events_seen": stop,
                               "queries_seen": count,
                               "query_coverage": count / stop if stop else None})
    correction_deltas = [float(row["correction_score_delta"]) for row in rows]
    class_result = {
        "event_count": n,
        "source_group_count": len(set(groups.tolist())),
        "outcomes": {"normal": int((y == 0).sum()), "defect": int((y == 1).sum()),
                     "defect_prevalence": float((y == 1).mean())},
        "query_feedback": {"count": len(positions), "normal": int((query_labels == 0).sum()),
                           "defect": int((query_labels == 1).sum()),
                           "rate": len(positions) / n,
                           "one_query_per_source_group": len(set(groups[list(positions)].tolist())) == len(positions)},
        "first_defect_position_in_shuffled_order": int(np.flatnonzero(y == 1)[0]) if np.any(y == 1) else None,
        "first_queried_defect_position_in_shuffled_order": (
            min((position for position in positions if int(y[position]) == 1), default=None)),
        "metrics_by_arm": metrics,
        "paired_contrasts_GC_minus_G": contrasts,
        "paired_source_group_bootstrap": bootstrap["contrasts"],
        "bootstrap_method": {key: value for key, value in bootstrap.items() if key != "_draws"},
        "shuffled_stream_early_middle_late": temporal,
        "cumulative_query_coverage": query_coverage,
        "correction_score_delta_distribution_GC_minus_G": _distribution_summary(correction_deltas),
        "correction_activation_fraction": float(np.mean(np.asarray(correction_deltas) != 0)),
        "mechanism_receipt": {
            "greedy_write_events": int(completion["greedy_write_events"]),
            "greedy_memory_changes": int(completion["greedy_memory_changes"]),
            "greedy_accepted_patches": int(completion["greedy_accepted_patches"]),
            "greedy_skipped_patches": int(completion["greedy_skipped_patches"]),
            "greedy_writer_audit": completion["greedy_writer_audit"],
            "greedy_occupancy_history": completion["greedy_occupancy_history"],
            "correction_update_count": int(completion["correction_update_count"]),
            "correction_state_change_count": int(completion["correction_state_change_count"]),
            "correction_update_stats": completion["correction_update_stats"],
            "correction_final_occupancy": int(completion["correction_final_occupancy"]),
            "correction_occupancy_history": completion["correction_state_occupancy_history"],
            "correction_capacity_first_reached_position": completion["correction_capacity_first_reached_position"],
            "final_mutable_state_bytes": completion["mutable_state_bytes"],
            "state_change_positions": completion["state_change_positions"],
            "b1_threshold_history": completion["b1_threshold_history"],
            "b3_threshold_history": completion["b3_threshold_history"],
            "b1_update_summary": completion["b1_update_summary"],
            "b3_update_summary": completion["b3_update_summary"],
            "c64_rescore_rounds": completion["c64_rescore_rounds"],
            "c64_rescore_score_calls": completion["c64_rescore_score_calls"],
            "c64_rescore_seconds": completion["c64_rescore_seconds"],
            "c64_cache_receipt": completion["c64_cache_receipt"],
        },
        "resources": {"wall_seconds": completion["wall_seconds"],
                      "stream_image_encodings": completion["actual_stream_image_encodings"],
                      "peak_cuda_allocated_bytes": completion["peak_cuda_allocated_bytes"],
                      "peak_cuda_reserved_bytes": completion["peak_cuda_reserved_bytes"],
                      "peak_cpu_rss_bytes": completion["peak_cpu_rss_bytes"],
                      "runtime_profile": completion["runtime_profile"],
                      "hub_source_tree_sha256": completion["hub_source_tree_sha256"],
                      "checkpoint_sha256": completion["checkpoint_sha256"],
                      "loaded_model_state_sha256": completion["loaded_model_state_sha256"]},
    }
    return class_result, bootstrap["_draws"]


def _mean_ci(values: Any) -> list[float] | None:
    import numpy as np
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return ([float(np.quantile(array, .025)), float(np.quantile(array, .975))]
            if len(array) else None)


def _aggregate_macro(categories: Mapping[str, Mapping[str, Any]],
                     bootstrap_draws: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    metric_keys = ("auroc", "average_precision", "balanced_accuracy", "fpr", "fnr",
                   "accuracy", "prediction_coverage")
    macro_arms = {}
    for arm in ARMS:
        result: dict[str, Any] = {}
        for key in metric_keys:
            values = [categories[category]["metrics_by_arm"][arm].get(key)
                      for category in CATEGORIES]
            finite = [float(value) for value in values if value is not None]
            result[key] = sum(finite) / len(finite) if finite else None
        confusion_keys = ("tn", "fp", "fn", "tp")
        result["confusion_sum"] = {
            key: sum(int(categories[category]["metrics_by_arm"][arm]["confusion"][key])
                     for category in CATEGORIES) for key in confusion_keys}
        result["query_count"] = TOTAL_QUERIES
        result["query_rate"] = TOTAL_QUERIES / TOTAL_EVENTS
        result["threshold_mean_equal_category"] = sum(
            float(categories[category]["metrics_by_arm"][arm]["threshold_summary"]["mean"])
            for category in CATEGORIES) / len(CATEGORIES)
        for key in ("worst_balanced_accuracy", "maximum_fpr", "maximum_fnr"):
            values = [categories[category]["metrics_by_arm"][arm][
                "worst_complete_512_event_window"].get(key) for category in CATEGORIES]
            finite = [float(value) for value in values if value is not None]
            result.setdefault("mean_of_category_worst_512_windows", {})[key] = (
                sum(finite) / len(finite) if finite else None)
        macro_arms[arm] = result
    contrast_names = sorted(set.intersection(*[
        set(categories[category]["paired_contrasts_GC_minus_G"]) for category in CATEGORIES]))
    macro_contrasts = {}
    sign_consistency = {}
    for name in contrast_names:
        values = [categories[category]["paired_contrasts_GC_minus_G"][name]
                  for category in CATEGORIES]
        finite = [float(value) for value in values if value is not None]
        macro_contrasts[name] = sum(finite) / len(finite) if finite else None
        sign_consistency[name] = {
            "positive_categories": sum(value > 0 for value in finite),
            "negative_categories": sum(value < 0 for value in finite),
            "zero_categories": sum(value == 0 for value in finite),
            "available_categories": len(finite),
            "category_count": len(CATEGORIES),
        }
    boot_macro: dict[str, Any] = {}
    contrast_ids = sorted(set.intersection(*[
        set(bootstrap_draws[category]) for category in CATEGORIES]))
    for name in contrast_ids:
        draws_by_category = [bootstrap_draws[category][name] for category in CATEGORIES]
        arrays = []
        for value in draws_by_category:
            array = value
            if len(array) != BOOTSTRAP_REPLICATES:
                raise R7Error("Category bootstrap draw lengths differ")
            arrays.append(array)
        stack = __import__("numpy").stack(arrays, axis=0)
        valid = __import__("numpy").isfinite(stack).all(axis=0)
        macro_draws = stack[:, valid].mean(axis=0)
        boot_macro[name] = {"valid_replicates": int(valid.sum()),
                            "ci_95_percentile": _mean_ci(macro_draws)}
    query_normal = sum(int(categories[c]["query_feedback"]["normal"]) for c in CATEGORIES)
    query_defect = sum(int(categories[c]["query_feedback"]["defect"]) for c in CATEGORIES)
    return {
        "category_count": len(CATEGORIES),
        "events": TOTAL_EVENTS, "query_count": TOTAL_QUERIES,
        "query_rate": TOTAL_QUERIES / TOTAL_EVENTS,
        "query_feedback": {"normal": query_normal, "defect": query_defect},
        "outcomes": {"normal": sum(int(categories[c]["outcomes"]["normal"]) for c in CATEGORIES),
                     "defect": sum(int(categories[c]["outcomes"]["defect"]) for c in CATEGORIES)},
        "metrics_by_arm_equal_category_macro": macro_arms,
        "paired_contrasts_GC_minus_G_equal_category_macro": macro_contrasts,
        "category_sign_consistency_GC_minus_G": sign_consistency,
        "paired_source_group_bootstrap_equal_category_macro": boot_macro,
        "bootstrap_method": "within each category, resample whole Source Groups; average category contrasts per replicate",
    }


def _stage_b_child() -> dict[str, Any]:
    import numpy as np
    import torch
    if torch.cuda.device_count() != 0 or torch.cuda.is_initialized():
        raise R7Error("R7 Stage B must run CPU-only with CUDA hidden")
    output = Path(os.environ.get("R7_CHILD_OUTPUT", ""))
    contract, manifest = _verify_child_identity(output, "FULL_STAGE_A", CATEGORIES[0], 0)
    stage_a_path = output / "stage_a_seal.json"
    scores_path = output / "stage_a_scores.jsonl"
    if not stage_a_path.is_file() or not scores_path.is_file():
        raise R7Error("R7 Stage B requires a sealed full Stage A trajectory")
    stage_a = _json_read(stage_a_path)
    if (stage_a.get("schema") != STAGE_A_SCHEMA or stage_a.get("status") != "SEALED_COMPLETE" or
            stage_a.get("run_id") != RUN_ID or stage_a.get("event_count") != TOTAL_EVENTS or
            stage_a.get("query_count") != TOTAL_QUERIES or
            stage_a.get("identity_sha256") != manifest.get("identity_sha256") or
            stage_a.get("score_rows_sha256") != _sha256_file(scores_path) or
            set(stage_a.get("category_seals", {})) != set(CATEGORIES)):
        raise R7Error("R7 Stage A global seal is incomplete or its score file changed")
    rows_by_category = _rows_by_category(output, require_full=True)
    history = _read_history(output)
    completion_rows = [row for row in history
                       if row.get("record_type") == "category_complete" and
                       row.get("mode") == "FULL_STAGE_A"]
    completions = {row["category"]: row for row in completion_rows}
    if len(completion_rows) != len(CATEGORIES) or len(completions) != len(CATEGORIES):
        raise R7Error("R7 Stage B found duplicate or extraneous category completion receipts")
    if set(completions) != set(CATEGORIES):
        raise R7Error("R7 Stage B found missing or duplicate category completion receipts")
    # The first outcome field is not opened until all four Stage A category
    # completion receipts and the global seal above have passed verification.
    from experiments.calibration import r3_fullstream_2x2 as r3
    category_results: dict[str, Any] = {}
    bootstrap_draws: dict[str, Any] = {}
    outcome_vectors: dict[str, dict[str, int]] = {}
    for cat_index, category in enumerate(CATEGORIES):
        class_info = contract["metadata"][category]
        checked = _verify_category_receipt(
            category, rows_by_category[category], completions[category], history, class_info)
        if checked["canonical_score_rows_sha256"] != stage_a["category_seals"][category][
                "canonical_score_rows_sha256"]:
            raise R7Error(f"{category}: category seal digest differs from Stage A global seal")
        original_labels = r3._read_stage_b_label_projection(
            class_info["json_path"], EXPECTED_COUNTS[category])
        y = np.asarray([original_labels[event.original_position]
                        for event in class_info["events"]], dtype=np.int8)
        category_results[category], bootstrap_draws[category] = _category_stage_b(
            category, rows_by_category[category], y, class_info["events"],
            class_info["tape_receipt"], completions[category], history,
            BOOTSTRAP_SEED + cat_index * 1009)
        outcome_vectors[category] = {"normal": int((y == 0).sum()),
                                     "defect": int((y == 1).sum())}
    macro = _aggregate_macro(category_results, bootstrap_draws)
    aggregate = {
        "schema": "r7.fixed_shuffle.aggregate.v1",
        "run_id": RUN_ID,
        "classification": "RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC",
        "interpretation_limits": {
            "independent_confirmation": False,
            "blinded_evaluation": False,
            "physical_factory_chronology_claimed": False,
            "results_use_only_the_four_authorized_development_categories": True,
            "all_stage_a_scores_sealed_before_outcome_projection": True,
            "test_masks_opened": False,
        },
        "design_identity_sha256": manifest["identity_sha256"],
        "stage_a_seal_sha256": _sha256_file(stage_a_path),
        "stage_a_score_rows_sha256": _sha256_file(scores_path),
        "class_results": category_results,
        "equal_category_macro": macro,
        "outcome_vectors_read_after_seal": outcome_vectors,
        "paired_bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "stage_b_label_projection": "image-level anomaly_class only, reordered through sealed shuffled-to-original identity map",
        "test_masks_opened": False,
    }
    budget = _budget(output)
    budget.write_json("aggregate_results.json", aggregate)
    stage_b_seal = {
        "schema": STAGE_B_SCHEMA, "status": "SEALED_COMPLETE",
        "run_id": RUN_ID,
        "stage_a_seal_sha256": _sha256_file(stage_a_path),
        "stage_a_score_rows_sha256": _sha256_file(scores_path),
        "aggregate_results_sha256": _sha256_file(output / "aggregate_results.json"),
        "category_event_counts": dict(EXPECTED_COUNTS),
        "outcome_labels_opened_after_all_four_stage_a_seals": True,
        "outcome_projection": "anomaly_class only",
        "test_masks_opened": False,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    budget.write_json("stage_b_seal.json", stage_b_seal)
    budget.write_json("complete_seal.json", {
        "schema": "r7.fixed_shuffle.complete.v1", "status": "COMPLETE",
        "run_id": RUN_ID,
        "stage_a_seal_sha256": _sha256_file(stage_a_path),
        "stage_b_seal_sha256": _sha256_file(output / "stage_b_seal.json"),
        "aggregate_results_sha256": _sha256_file(output / "aggregate_results.json"),
        "event_count": TOTAL_EVENTS, "query_count": TOTAL_QUERIES,
        "automatic_restart": False,
    })
    return aggregate


def _initialize_output(output: Path, mode: str, identity: Mapping[str, Any],
                       sources: Mapping[str, str], token: str,
                       free_space: int) -> None:
    budget = _budget(output)
    manifest = {
        "schema": RUN_SCHEMA, "run_id": RUN_ID, "mode": mode,
        "authorization": AUTHORIZATION,
        "authorization_token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        "status": "INCOMPLETE",
        "classification": "RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC",
        "source_parent_commit": EXPECTED_PARENT, "branch": EXPECTED_BRANCH,
        "source_fingerprints": dict(sources),
        "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        "identity_sha256": identity["identity_sha256"],
        "experiment_identity": identity,
        "class_json_sha256_by_category": {
            category: identity["classes"][category]["category_json_sha256"]
            for category in CATEGORIES},
        "stage05_source_receipts": identity["stage05_source_receipts"],
        "output_storage": {"per_directory_byte_limit": OUTPUT_LIMIT,
                           "failure_metadata_reserve_bytes": FAILURE_RESERVE,
                           "max_files": 20, "prewrite_accountant": "R2 OutputBudget",
                           "os_hard_storage_isolation": False,
                           "unique_non_overwriting_namespace": True},
        "prelaunch_free_space_bytes": int(free_space),
        "minimum_free_space_bytes": MIN_FREE_BYTES,
        "windows_job_object_required": True,
        "full_deadline_seconds_per_category": FULL_DEADLINE_SECONDS,
        "smoke_deadline_seconds": SMOKE_DEADLINE_SECONDS,
        "automatic_restart": False, "resume_incomplete_run": False,
        "test_masks_accessed": False, "unqueried_test_labels_accessed": False,
        "offline_outcome_vectors_accessed": False,
    }
    budget.write_json("run_manifest.json", manifest)


def _supervisor_row(stage: str, category: str | None, result: Any) -> dict[str, Any]:
    from dataclasses import asdict
    return {"schema": "r7.fixed_shuffle.supervisor.v1",
            "record_type": "supervisor_result", "stage": stage,
            "category": category, "result": asdict(result),
            "process_tree_containment": "Windows Job Object, suspended child assignment, kill-on-close",
            "automatic_restart": False}


def _run_child(stage: str, output: Path, token: str, *,
               authorized_cuda_visibility: str | None,
               category: str | None = None, mode: str | None = None,
               smoke_events: int = 0) -> Any:
    from experiments.calibration import r1_runtime_supervisor as supervisor
    from experiments.calibration import r3_fullstream_2x2 as r3
    base = dict(os.environ)
    env = r3._child_environment(
        base, "STAGE_A" if stage in {"SMOKE", "STAGE_A"} else "STAGE_B",
        authorized_cuda_visibility)
    env["R7_CHILD_OUTPUT"] = str(output)
    env["R7_CHILD_TOKEN"] = token
    command = [sys.executable, "-m", "experiments.calibration.r7_shuffled_correction_4class"]
    if stage in {"SMOKE", "STAGE_A"}:
        command.extend(["--stage-a-child", "--category", str(category),
                        "--mode", str(mode), "--smoke-events", str(smoke_events)])
        timeout = SMOKE_DEADLINE_SECONDS if stage == "SMOKE" else FULL_DEADLINE_SECONDS
    elif stage == "STAGE_B":
        command.append("--stage-b-child")
        timeout = FULL_DEADLINE_SECONDS
    else:
        raise R7Error(f"unsupported R7 supervised stage: {stage}")
    return supervisor.run_supervised_child(
        command, timeout_seconds=timeout, env=env, cwd=ROOT)


def _append_parent_failure(output: Path, stage: str, error: BaseException) -> None:
    if not output.exists() or (output / "failure.json").exists():
        return
    try:
        import traceback
        _budget(output).write_json("failure.json", {
            "schema": "r7.fixed_shuffle.failure.v1", "run_id": RUN_ID,
            "stage": stage, "error_type": type(error).__name__,
            "message": str(error)[:4000],
            "traceback": "".join(traceback.format_exception(error))[-12000:],
            "partial_outputs_preserved": True,
            "automatic_restart": False, "finished_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, failure_reserved=True)
    except Exception:
        # Preserve the primary failure if even reserved metadata cannot be written.
        pass


def _write_child_failure(stage: str, error: BaseException) -> None:
    output_value = os.environ.get("R7_CHILD_OUTPUT")
    if not output_value:
        return
    output = Path(output_value)
    if not output.is_dir() or (output / "failure.json").exists():
        return
    import traceback
    _budget(output).write_json("failure.json", {
        "schema": "r7.fixed_shuffle.failure.v1", "run_id": RUN_ID,
        "stage": stage, "error_type": type(error).__name__,
        "message": str(error)[:4000],
        "traceback": "".join(traceback.format_exception(error))[-12000:],
        "partial_outputs_preserved": True, "automatic_restart": False,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, failure_reserved=True)


def _write_report_and_aggregate(aggregate: Mapping[str, Any],
                                identity: Mapping[str, Any],
                                stage_a: Mapping[str, Any],
                                smoke_receipt: Mapping[str, Any],
                                stage_b_supervisor: Any,
                                source_fingerprints: Mapping[str, str]) -> None:
    if REPORT_DIR.exists() or REPORT_PATH.exists() or AGGREGATE_PATH.exists():
        raise R7Error("R7 documentation output exists; refusing to overwrite")
    REPORT_DIR.mkdir(parents=False, exist_ok=False)
    aggregate_bytes = _canonical_bytes(dict(aggregate))
    AGGREGATE_PATH.write_bytes(aggregate_bytes)
    if hashlib.sha256(AGGREGATE_PATH.read_bytes()).hexdigest() != hashlib.sha256(aggregate_bytes).hexdigest():
        raise R7Error("R7 public aggregate JSON failed its post-write digest check")
    lines = [
        "# R7 Four-Class Fixed-Shuffle Correction Necessity Experiment",
        "",
        "## Classification and protocol",
        "",
        "**Classification:** `RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC`. This is not independent confirmation or a blind evaluation. Historical development outcomes may have been observed before this experiment. The four category streams were permuted by Source Group (seed 0); original camera-view order was kept within each group. Reported positions are positions in that shuffled stream, not factory chronology.",
        "",
        "All six decisions at an event and its fixed Q-A-TS query decision were durably sealed before any queried label was disclosed. Stage A read only the exact query labels. Stage B opened image-level outcomes only after all four full Stage A seals and did not open masks. Greedy and C1 updates used the same legal queried feedback. Threshold variants changed decisions only; the shared Greedy writer and Correction eligibility used the pre-event G/C0 score and frozen C64 q90.",
        "",
        "C1-Full used the frozen native SignedCorrection formula and state with K=1024; Greedy used K=1024. No FPSA, BEA, hierarchy, parameter search, or formula changes were used. Q-A-TS-v1 used rate 3%, Seed 37, 16 bins, half-up rounding, with at most one queried view per Source Group.",
        "The exact manifest-bound P0 scoring values, profile digest, C1-Full configuration, Greedy writer settings and capacity overrides are recorded in the private experiment identity.",
        "",
        "## Execution receipts",
        "",
        f"- Branch and parent commit: `{EXPECTED_BRANCH}` / `{EXPECTED_PARENT}`.",
        f"- Design identity SHA-256: `{identity['identity_sha256']}`.",
        f"- Stage A status: `{stage_a['status']}`; {stage_a['event_count']:,} shuffled events and {stage_a['query_count']} legal queries.",
        f"- GPU smoke: `{smoke_receipt.get('status', 'PASS')}` on audiojack, {smoke_receipt.get('events_processed')} events and {smoke_receipt.get('query_events_processed')} queried events; Greedy state changes {smoke_receipt.get('greedy_memory_changes')}, C1 state changes {smoke_receipt.get('correction_state_change_count')}. The smoke is excluded from Stage B metrics.",
        f"- Stage B status: `SEALED_COMPLETE`; outcome projection opened after the four category seals. Windows Job Object result: `{getattr(stage_b_supervisor, 'status', 'UNKNOWN')}`.",
        "- Every full category encoded each scored TEST event once. Audiojack reused the checksum-verified sealed R5 C64 cache directly; regulator, switch, and rolled_strip_base encoded each of their 64 C64 references once.",
        f"- R5 cache lineage: the pinned R5 report records executed runner `{R5_EXECUTED_RUNNER_SHA256}` and committed runner `{R5_COMMITTED_RUNNER_SHA256}`; it records that only a trailing blank line changed. R7 verified both hashes, the report digest `{R5_REPORT_SHA256}`, R5 seals, cache manifest/file checksums, C64 group/image identities, and pixel equality. The later R6 cache report contains an invalid 63-character executed-runner digest, so R7 does not rely on the R6 copy.",
        "- Runtime ceiling was 120 minutes per full category child and 15 minutes for smoke. Each child was supervised in a Windows Job Object. No restart or resume was used.",
        "- Output controls: fresh private run namespace, 4 GiB pre-write OutputBudget per smoke/full directory, 64 KiB reserved failure metadata, and at least 10 GiB free space before launch. This is an application-level bound, not an OS-level hard storage quota.",
        "",
        "## Results by category and arm",
        "",
        "AUROC/AP rank the sealed scorer trajectory; controller thresholds do not alter those scores. Confusion is TN/FP/FN/TP. Worst-window values use every stride-1 complete 512-event window in shuffled order. Query count/rate is shared across the six arms.",
        "",
    ]
    for category in CATEGORIES:
        result = aggregate["class_results"][category]
        lines.extend([f"### {category}", "",
                      f"Events {result['event_count']:,}; Source Groups {result['source_group_count']:,}; outcomes normal {result['outcomes']['normal']:,}, defect {result['outcomes']['defect']:,}; queries {result['query_feedback']['count']} ({result['query_feedback']['rate']:.3%}; normal feedback {result['query_feedback']['normal']}, defect feedback {result['query_feedback']['defect']}).",
                      "",
                      "| Arm | AUROC | AP | BA | FPR | FNR | Accuracy | TN / FP / FN / TP | Coverage | Queries | Worst 512 BA | Worst 512 FPR | Worst 512 FNR |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for arm in ARMS:
            metric = result["metrics_by_arm"][arm]
            worst = metric["worst_complete_512_event_window"]
            confusion = metric["confusion"]
            def fmt(value: Any) -> str:
                return "NA" if value is None else f"{float(value):.4f}"
            lines.append(
                f"| {arm} | {fmt(metric['auroc'])} | {fmt(metric['average_precision'])} | {fmt(metric['balanced_accuracy'])} | {fmt(metric['fpr'])} | {fmt(metric['fnr'])} | {fmt(metric['accuracy'])} | {confusion['tn']} / {confusion['fp']} / {confusion['fn']} / {confusion['tp']} | {fmt(metric['prediction_coverage'])} | {metric['query_count']} | {fmt(worst['worst_balanced_accuracy'])} | {fmt(worst['maximum_fpr'])} | {fmt(worst['maximum_fnr'])} |")
        lines.extend(["", "Paired GC−G contrasts and Source Group bootstrap intervals:", "",
                      "| Contrast | Estimate | 95% Source Group bootstrap interval |",
                      "|---|---:|---:|"])
        for name, value in result["paired_contrasts_GC_minus_G"].items():
            ci = result["paired_source_group_bootstrap"][name]["ci_95_percentile"]
            ci_text = "NA" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            lines.append(f"| {name} | {'NA' if value is None else f'{value:.4f}'} | {ci_text} |")
        lines.extend(["", "Mechanism and shuffled-region diagnostics:", "",
                      f"- First defect position: {result['first_defect_position_in_shuffled_order']}; first queried defect position: {result['first_queried_defect_position_in_shuffled_order']}.",
                      f"- Correction score delta GC−G: `{json.dumps(result['correction_score_delta_distribution_GC_minus_G'], sort_keys=True)}`; activation fraction {result['correction_activation_fraction']:.4f}.",
                      f"- Greedy native update events: {result['mechanism_receipt']['greedy_write_events']}; actual shared-bank mutations: {result['mechanism_receipt']['greedy_memory_changes']}; accepted patches {result['mechanism_receipt']['greedy_accepted_patches']}; skipped patches {result['mechanism_receipt']['greedy_skipped_patches']}.",
                      f"- C1 queried updates: {result['mechanism_receipt']['correction_update_count']}; scorer-state changes {result['mechanism_receipt']['correction_state_change_count']}; final occupancy {result['mechanism_receipt']['correction_final_occupancy']}/1024; first full capacity position {result['mechanism_receipt']['correction_capacity_first_reached_position']}.",
                      f"- B1 updates/threshold movements: `{json.dumps(result['mechanism_receipt']['b1_update_summary'], sort_keys=True)}`; B3 complete-state updates/threshold movements: `{json.dumps(result['mechanism_receipt']['b3_update_summary'], sort_keys=True)}`.",
                      f"- Query-time Greedy and Correction occupancy traces are included as anonymous shuffled positions and counts in the aggregate; no sample identity is included.",
                      f"- Mutable tensor bytes at completion: `{json.dumps(result['mechanism_receipt']['final_mutable_state_bytes'], sort_keys=True)}`.",
                      f"- C64 rescoring rounds: `{json.dumps(result['mechanism_receipt']['c64_rescore_rounds'], sort_keys=True)}`; score calls: `{json.dumps(result['mechanism_receipt']['c64_rescore_score_calls'], sort_keys=True)}`; seconds: `{json.dumps(result['mechanism_receipt']['c64_rescore_seconds'], sort_keys=True)}`.",
                      "",
                      "| Shuffled region | Events | Queries (normal / defect) | G-Frozen AUROC | GC-Frozen AUROC | GC−G AUROC | G-Frozen BA | GC-Frozen BA | GC−G BA |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for region in result["shuffled_stream_early_middle_late"]:
            mm = region["metrics_by_arm"]
            delta = region["GC_minus_G"]
            def fmt_region(value: Any) -> str:
                return "NA" if value is None else f"{float(value):.4f}"
            lines.append(
                f"| {region['shuffled_stream_region']} | {region['event_count']} | {region['query_count']} ({region['query_normal_feedback']} / {region['query_defect_feedback']}) | {fmt_region(mm['G-Frozen']['auroc'])} | {fmt_region(mm['GC-Frozen']['auroc'])} | {fmt_region(delta['auroc_GC_minus_G'])} | {fmt_region(mm['G-Frozen']['balanced_accuracy'])} | {fmt_region(mm['GC-Frozen']['balanced_accuracy'])} | {fmt_region(delta['balanced_accuracy_GC_minus_G_Frozen'])} |")
        lines.extend(["", ""])
    macro = aggregate["equal_category_macro"]
    lines.extend(["## Equal-category macro results", "",
                  "| Arm | AUROC | AP | BA | FPR | FNR | Accuracy | Queries |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for arm in ARMS:
        item = macro["metrics_by_arm_equal_category_macro"][arm]
        def fmt_macro(value: Any) -> str:
            return "NA" if value is None else f"{float(value):.4f}"
        lines.append(f"| {arm} | {fmt_macro(item['auroc'])} | {fmt_macro(item['average_precision'])} | {fmt_macro(item['balanced_accuracy'])} | {fmt_macro(item['fpr'])} | {fmt_macro(item['fnr'])} | {fmt_macro(item['accuracy'])} | {item['query_count']} |")
    lines.extend(["", "Paired macro GC−G effects and intervals:", "",
                  "| Contrast | Estimate | 95% Source Group bootstrap interval |",
                  "|---|---:|---:|"])
    for name, value in macro["paired_contrasts_GC_minus_G_equal_category_macro"].items():
        ci = macro["paired_source_group_bootstrap_equal_category_macro"][name]["ci_95_percentile"]
        ci_text = "NA" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        lines.append(f"| {name} | {'NA' if value is None else f'{value:.4f}'} | {ci_text} |")
    lines.extend(["", "Category sign counts for each GC−G contrast:", "",
                  "| Contrast | Positive categories | Negative categories | Zero | Available |",
                  "|---|---:|---:|---:|---:|"])
    for name, result in macro["category_sign_consistency_GC_minus_G"].items():
        lines.append(f"| {name} | {result['positive_categories']} | {result['negative_categories']} | {result['zero_categories']} | {result['available_categories']} |")
    lines.extend(["", "## Runtime fingerprints and audit limits", "",
                  "The exact runtime profile and DINO hub source, checkpoint, and loaded-state fingerprints are recorded in the aggregate by category. The source code fingerprint set is recorded below. No source fingerprint was used to change the treatment.",
                  "",
                  "| Source | SHA-256 |", "|---|---|"])
    for relative, digest in source_fingerprints.items():
        lines.append(f"| `{relative}` | `{digest}` |")
    lines.extend(["", "The R7 runner was bound to the parent source commit and its SHA-256 is recorded in the run manifest and aggregate identity. Fixed Support digests were independently checked against the frozen structured receipt for every category. C64 support is additional to the four fixed Support images.",
                  "",
                  "This is retrospective evidence from four previously exposed development categories. It cannot establish independent generalization, blind confirmation, or physical time-order behavior. Per-event identities, labels and scores remain only in the private run directory; this report and its aggregate contain no sample identities, feature weights, event-level labels, or raw event-level score vectors.",
                  "",
                  "## Reproducibility",
                  "",
                  f"Private output: `outputs/calibration/{RUN_ID}/` (smoke and full children are separate, non-resumable directories). Public tracked deliverables are this report and `aggregate_results.json`. Stage A seal SHA-256: `{_sha256_file(RUN_NAMESPACE / 'full' / 'stage_a_seal.json')}`. Stage B aggregate SHA-256: `{hashlib.sha256(aggregate_bytes).hexdigest()}`.",
                  ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _parent_run() -> dict[str, Any]:
    import secrets
    from experiments.calibration import r1_runtime_supervisor as supervisor
    from experiments.calibration import r3_fullstream_2x2 as r3
    authorized_cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    _assert_repository_identity()
    source_fingerprints = _source_fingerprints()
    free_before = _check_disk_space(OUTPUT_BASE)
    contract = _frozen_contract()
    smoke_info = contract["metadata"][SMOKE_CATEGORY]
    smoke_query_count = sum(int(position) < SMOKE_EVENTS
                            for position in smoke_info["tape_receipt"]["positions"])
    if smoke_query_count < 1:
        raise R7Error("pre-registered GPU smoke prefix contains no legal query event")
    identity = _experiment_identity(contract, source_fingerprints,
                                    _sha256_file(Path(__file__).resolve()))
    token = secrets.token_hex(32)
    smoke_output, full_output = _create_output_directories()
    _initialize_output(smoke_output, "SMOKE_ONLY", identity,
                       source_fingerprints, token, free_before)
    _initialize_output(full_output, "FULL_STAGE_A", identity,
                       source_fingerprints, token, free_before)
    smoke_receipt = None
    try:
        smoke_result = _run_child(
            "SMOKE", smoke_output, token,
            authorized_cuda_visibility=authorized_cuda_visibility,
            category=SMOKE_CATEGORY, mode="SMOKE_ONLY", smoke_events=SMOKE_EVENTS)
        _append_history(_budget(smoke_output), _supervisor_row("GPU_SMOKE", SMOKE_CATEGORY, smoke_result))
        if (smoke_result.status != "COMPLETED" or smoke_result.child_exit_code != 0 or
                smoke_result.termination_confirmed is not True or
                smoke_result.active_job_processes != 0):
            raise R7Error(f"supervised R7 GPU smoke failed: {smoke_result}")
        smoke_history = _read_history(smoke_output)
        smoke_rows = [row for row in smoke_history
                      if row.get("record_type") == "category_complete" and
                      row.get("category") == SMOKE_CATEGORY]
        if len(smoke_rows) != 1:
            raise R7Error("R7 GPU smoke did not emit one complete category receipt")
        smoke_receipt = smoke_rows[0]
        if (smoke_receipt.get("events_processed") != SMOKE_EVENTS or
                smoke_receipt.get("query_events_processed", 0) < 1 or
                smoke_receipt.get("greedy_memory_changes", 0) +
                smoke_receipt.get("correction_state_change_count", 0) < 1):
            raise R7Error("R7 GPU smoke did not exercise a legal queried scorer-state transition")
        _append_history(_budget(full_output), {
            "schema": "r7.fixed_shuffle.history.v1", "record_type": "smoke_receipt",
            "smoke_output_directory_id": "separate_smoke_child",
            "category": SMOKE_CATEGORY, "events_processed": SMOKE_EVENTS,
            "query_count": smoke_receipt["query_events_processed"],
            "legal_feedback_state_transition_verified": True,
            "smoke_receipt_sha256": _sha256_json(smoke_receipt),
            "supervisor_result": {"status": smoke_result.status,
                                  "child_exit_code": smoke_result.child_exit_code,
                                  "elapsed_seconds": smoke_result.elapsed_seconds,
                                  "termination_confirmed": smoke_result.termination_confirmed,
                                  "active_job_processes": smoke_result.active_job_processes},
        })
        for category in CATEGORIES:
            child_result = _run_child(
                "STAGE_A", full_output, token,
                authorized_cuda_visibility=authorized_cuda_visibility,
                category=category, mode="FULL_STAGE_A", smoke_events=0)
            _append_history(_budget(full_output), _supervisor_row("FULL_STAGE_A", category, child_result))
            if (child_result.status != "COMPLETED" or child_result.child_exit_code != 0 or
                    child_result.termination_confirmed is not True or
                    child_result.active_job_processes != 0):
                raise R7Error(f"supervised R7 full Stage A failed for {category}: {child_result}")
        stage_a = _seal_stage_a(full_output, contract, identity)
        stage_b_result = _run_child(
            "STAGE_B", full_output, token,
            authorized_cuda_visibility=authorized_cuda_visibility)
        if (stage_b_result.status != "COMPLETED" or stage_b_result.child_exit_code != 0 or
                stage_b_result.termination_confirmed is not True or
                stage_b_result.active_job_processes != 0):
            raise R7Error(f"supervised R7 CPU Stage B failed: {stage_b_result}")
        _append_history(_budget(full_output), _supervisor_row("STAGE_B_CPU", None, stage_b_result))
        _budget(full_output).write_json("supervisor_status.json", {
            "schema": "r7.fixed_shuffle.supervisor_summary.v1",
            "smoke": {"status": "COMPLETED", "elapsed_seconds": smoke_result.elapsed_seconds},
            "stage_a_categories": list(CATEGORIES),
            "stage_b": {"status": stage_b_result.status,
                        "elapsed_seconds": stage_b_result.elapsed_seconds,
                        "termination_confirmed": stage_b_result.termination_confirmed},
            "windows_job_object": "verified termination and empty job for every child",
            "automatic_restart": False,
        })
        complete = _json_read(full_output / "complete_seal.json")
        if complete.get("status") != "COMPLETE":
            raise R7Error("R7 Stage B returned without the full completion seal")
        aggregate = _json_read(full_output / "aggregate_results.json")
        _write_report_and_aggregate(aggregate, identity, stage_a,
                                    {**smoke_receipt, "status": "PASS"},
                                    stage_b_result, source_fingerprints)
        return {"status": "COMPLETE", "identity": identity,
                "stage_a": stage_a, "aggregate": aggregate,
                "smoke_receipt": smoke_receipt,
                "smoke_supervisor": smoke_result,
                "stage_b_supervisor": stage_b_result,
                "output": str(full_output)}
    except BaseException as exc:
        _append_parent_failure(full_output, "PARENT_OR_FULL_RUN", exc)
        _append_parent_failure(smoke_output, "SMOKE_OR_PARENT", exc)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-child", action="store_true")
    parser.add_argument("--stage-b-child", action="store_true")
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--mode", choices=("SMOKE_ONLY", "FULL_STAGE_A"))
    parser.add_argument("--smoke-events", type=int, default=0)
    args = parser.parse_args(argv)
    if args.stage_a_child and args.stage_b_child:
        parser.error("choose at most one child mode")
    if args.stage_a_child:
        if not args.category or not args.mode:
            parser.error("Stage A child needs --category and --mode")
        try:
            _stage_a_child(args.category, args.mode, args.smoke_events)
            return 0
        except BaseException as exc:
            _write_child_failure("STAGE_A", exc)
            raise
    if args.stage_b_child:
        try:
            _stage_b_child()
            return 0
        except BaseException as exc:
            _write_child_failure("STAGE_B", exc)
            raise
    if any((args.category, args.mode, args.smoke_events)):
        parser.error("child-only arguments cannot be used without a child mode")
    result = _parent_run()
    print(f"R7_COMPLETE run={RUN_ID} events={TOTAL_EVENTS} queries={TOTAL_QUERIES} "
          f"output={result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
