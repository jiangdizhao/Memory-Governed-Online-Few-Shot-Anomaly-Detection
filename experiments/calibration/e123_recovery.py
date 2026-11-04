"""E123 integrated development campaign.

This runner is intentionally separate from historical experiment runners.  It
uses the registered Stage 0.5/R7 native inputs, writes into one new immutable
namespace, seals all Stage A scores before offline Stage B label evaluation,
and never resumes a partial namespace.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import deque
from dataclasses import asdict, dataclass
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import shutil
import stat
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = ROOT / "outputs" / "calibration"
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
REQUESTED_PARENT = "46a9cd634015dd190ee1a2910beeedceb378ea30"
EXPECTED_COUNTS = {
    "audiojack": 3815,
    "regulator": 3855,
    "switch": 3820,
    "rolled_strip_base": 3805,
}
CATEGORIES = tuple(EXPECTED_COUNTS)
SUPPORT_SEEDS = (0, 1)
STREAM_ORDERS = ("json", "group_shuffle_seed0")
CAPACITIES = (256, 512, 1024)
MUTABLE_METHODS = ("FIFO", "RESERVOIR", "RANDOM_REPLACEMENT", "GREEDY", "UTILITY_STM")
CONFIGURATION_IDS = (
    "STATIC",
    *(f"{method}_K{capacity}" for method in MUTABLE_METHODS
      for capacity in CAPACITIES),
)
FIXED_OFFSET_GRID = tuple(round(-1.0 + i * 0.05, 2) for i in range(41))
FIXED_OFFSET_TIE_RULE = "minimum_macro_BER; then smallest_abs_coefficient; then smaller_signed_coefficient"
BYTE_TOLERANCE = 0.05
MAX_OUTPUT_BYTES = 4 * 1024**3
FAILURE_RESERVE_BYTES = 1024**2
MIN_FREE_BYTES = 10 * 1024**3
SUPERVISOR_TIMEOUT_SECONDS = 120 * 60
SOURCE_PATHS = (
    "experiments/calibration/e123_recovery.py",
    "experiments/calibration/r3_fullstream_2x2.py",
    "experiments/calibration/r7_shuffled_correction_4class.py",
    "experiments/calibration/native.py",
    "experiments/calibration/profiles.py",
    "experiments/calibration/query_tape.py",
    "experiments/calibration/protocol.py",
    "experiments/calibration/controllers.py",
    "experiments/calibration/stage05.py",
    "experiments/calibration/stage1a_final_readiness.py",
    "experiments/calibration/stage1a_p12_label_provider.py",
    "experiments/calibration/r1_matched_correction_pilot.py",
    "experiments/calibration/r1_runtime_supervisor.py",
    "experiments/a0_a1_memory_methods.py",
    "experiments/p0_protocol_audit.py",
    "experiments/v02_memory.py",
    "test_phase3_AIF_simplified_v22.py",
    "vmb_visionad_new.py",
    "tests/calibration/test_e123_recovery.py",
)
STAGE1A_ROOT = ROOT / "outputs/calibration/stage1a_formal_controller_comparison_v4_20261030"
STAGE1A_P12_ROOT = ROOT / "outputs/calibration/stage1a_p12_low_memory_recovery_v3_20261030"
STAGE05_ROOT = ROOT / "outputs/calibration/stage05_exploratory_20261029"
HISTORICAL_EVIDENCE_PATHS = (
    "outputs/a0_a1_strict4_memory/aggregate.json",
    "docs/A0_A1_STRICT4_MATCHED_MEMORY_PROTOCOL.md",
    "docs/A1H_HIERARCHICAL_MEMORY_ATTRIBUTION_PROTOCOL.md",
    "docs/BRIDGE_HISTORICAL_TO_MATCHED_AUDIT_PROTOCOL.md",
    "docs/calibration/STAGE075A_GREEDY_RANKING_FALSIFICATION_REPORT.md",
    "docs/calibration/STAGE075A_ACTUAL_B3_REPLAY_REPORT.md",
    "docs/calibration/audit/b3_state_lag_formal_v1_20261030/FORMAL_B3_MECHANISM_RESULTS.md",
    "docs/calibration/audit/external_baseline_b0_20261030/B0_EXTERNAL_BASELINE_AUDIT.md",
    "docs/calibration/audit/r3_fullstream_2x2_20261102/REPORT.md",
    "docs/calibration/audit/r7_shuffled_correction_4class_20261103/REPORT.md",
)
RUN_SCHEMA = "recovery.e123.run.v1"
CONTRACT_SCHEMA = "recovery.e123.contract.v1"
SCORE_SCHEMA = "recovery.e123.stage_a_score.v1"
CONDITION_SEAL_SCHEMA = "recovery.e123.condition_seal.v1"
GLOBAL_SEAL_SCHEMA = "recovery.e123.stage_a_global_seal.v1"


class E123Error(RuntimeError):
    """E123 input, causal protocol, or execution integrity failure."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise E123Error(f"Expected JSON object: {path}")
    return value


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8")
    return result.stdout.strip()


def source_identity() -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    commit = _git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        raise E123Error(f"Wrong branch: expected {EXPECTED_BRANCH}, got {branch}")
    tracked = _git("status", "--porcelain", "--", *SOURCE_PATHS)
    if tracked:
        raise E123Error("An E123 source dependency has tracked working-tree changes")
    missing = [path for path in SOURCE_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise E123Error(f"Pinned E123 sources are missing: {missing}")
    fingerprints = {path: sha256_file(ROOT / path) for path in SOURCE_PATHS}
    return {
        "branch": branch,
        "commit_sha": commit,
        "requested_parent_sha": REQUESTED_PARENT,
        "actual_parent_sha": _git("rev-parse", "HEAD^"),
        "requested_parent_is_ancestor": _git(
            "merge-base", "--is-ancestor", REQUESTED_PARENT, "HEAD") == "",
        "source_sha256": fingerprints,
        "source_fingerprint_sha256": sha256_json(fingerprints),
    }


def _historical_evidence_receipts() -> list[dict[str, Any]]:
    receipts = []
    for relative in HISTORICAL_EVIDENCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise E123Error(f"Registered historical evidence is missing: {relative}")
        receipts.append({"path": relative, "size_bytes": path.stat().st_size,
                         "sha256": sha256_file(path),
                         "use": "historical context only; not pooled into Regime B E123"})
    return receipts


def _safe_output_root(path: Path, *, must_exist: bool) -> Path:
    path = Path(os.path.abspath(path))
    parent = OUTPUT_BASE.resolve(strict=True)
    if OUTPUT_BASE.is_symlink() or _is_reparse_point(OUTPUT_BASE):
        raise E123Error("E123 output base cannot be a symlink or reparse point")
    if path.parent.resolve(strict=True) != parent:
        raise E123Error("E123 namespace must be a direct child of outputs/calibration")
    if path.exists() and (path.is_symlink() or _is_reparse_point(path)):
        raise E123Error("E123 namespace cannot be a symlink or reparse point")
    if must_exist and not path.is_dir():
        raise E123Error("E123 namespace is missing or not a directory")
    if path.exists() and path.resolve(strict=True).parent != parent:
        raise E123Error("E123 namespace resolved outside outputs/calibration")
    return path


class OutputBudget:
    """Pre-write byte accounting for one new, non-resumable output namespace."""

    _ALLOWED_RE = re.compile(
        r"^(?:E123_[A-Za-z0-9_.-]+\.json|E123_[A-Za-z0-9_.-]+\.md|"
        r"budget\.lock|stage_a/conditions/[a-z0-9_]+/"
        r"(?:condition_manifest\.json|scores\.jsonl|c64_snapshots\.jsonl|query_feedback\.private\.jsonl|"
        r"progress\.jsonl|condition_seal\.json|failure\.json|supervisor\.json)|"
        r"stage_a/(?:global_seal\.json|failure\.json|progress\.jsonl)|"
        r"stage_b/(?:aggregate\.json|failure\.json)|"
        r"logs/[a-z0-9_]+\.log)$")

    def __init__(self, root: Path, *, create: bool = False,
                 max_bytes: int = MAX_OUTPUT_BYTES,
                 failure_reserve_bytes: int = FAILURE_RESERVE_BYTES):
        self.root = _safe_output_root(root, must_exist=not create)
        self.max_bytes = int(max_bytes)
        self.failure_reserve_bytes = int(failure_reserve_bytes)
        if self.max_bytes <= self.failure_reserve_bytes or self.failure_reserve_bytes < 4096:
            raise E123Error("Invalid E123 output budget or failure reserve")
        if create:
            self.root.mkdir(mode=0o700, parents=False, exist_ok=False)
            self.root = _safe_output_root(self.root, must_exist=True)
            self._make_lock()
        self._thread_lock = threading.RLock()
        self._write_count = 0

    def _make_lock(self) -> None:
        lock_path = self.root / "budget.lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(fd, b"\0")
            os.fsync(fd)
        finally:
            os.close(fd)

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            lock_path = self.root / "budget.lock"
            if (not lock_path.is_file() or lock_path.is_symlink() or
                    _is_reparse_point(lock_path)):
                raise E123Error("Output budget lock is missing or unsafe")
            with lock_path.open("r+b", buffering=0) as lock_file:
                if os.name == "nt":
                    import msvcrt
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                    try:
                        yield
                    finally:
                        lock_file.seek(0)
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _relative(self, relative: str) -> tuple[str, Path]:
        rel = Path(relative).as_posix()
        if (not self._ALLOWED_RE.fullmatch(rel) or rel.startswith("/") or
                ".." in Path(rel).parts):
            raise E123Error(f"Output path is outside the E123 write registry: {relative}")
        path = self.root / Path(rel)
        if self.root.is_symlink() or _is_reparse_point(self.root):
            raise E123Error("E123 root became a symlink or reparse point")
        root_resolved = self.root.resolve(strict=True)
        parent = self.root
        for part in Path(rel).parts[:-1]:
            parent = parent / part
            try:
                parent.mkdir()
            except FileExistsError:
                pass
            if (parent.is_symlink() or _is_reparse_point(parent) or
                    not parent.is_dir()):
                raise E123Error("Symlink/reparse-point output parent rejected")
            resolved_parent = parent.resolve(strict=True)
            if root_resolved not in (resolved_parent, *resolved_parent.parents):
                raise E123Error("Output parent escaped the E123 namespace")
        if path.exists() and (path.is_symlink() or _is_reparse_point(path)):
            raise E123Error("Symlink/reparse-point output file rejected")
        return rel, path

    def _audit_and_size(self) -> int:
        size = 0
        for current, dirs, files in os.walk(self.root, followlinks=False):
            current_path = Path(current)
            for name in list(dirs):
                item = current_path / name
                if item.is_symlink() or _is_reparse_point(item):
                    raise E123Error("Unaccounted symlink/reparse point in output namespace")
            for name in files:
                item = current_path / name
                rel = item.relative_to(self.root).as_posix()
                if not self._ALLOWED_RE.fullmatch(rel):
                    raise E123Error(f"Unaccounted output file detected: {rel}")
                if item.is_symlink() or _is_reparse_point(item) or not item.is_file():
                    raise E123Error(f"Unsafe output file detected: {rel}")
                size += item.stat().st_size
        return size

    def _prewrite(self, nbytes: int, *, failure: bool) -> None:
        if nbytes < 0:
            raise E123Error("Negative output reservation")
        current = self._audit_and_size()
        allowed = self.max_bytes if failure else self.max_bytes - self.failure_reserve_bytes
        if current + nbytes > allowed:
            raise E123Error(
                f"E123 pre-write byte limit refused {nbytes} bytes: "
                f"used={current}, authorized={allowed}")

    def write_once(self, relative: str, payload: bytes, *, failure: bool = False) -> None:
        with self._locked():
            rel, path = self._relative(relative)
            self._prewrite(len(payload), failure=failure)
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | nofollow, 0o600)
            try:
                with os.fdopen(fd, "wb", closefd=False) as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                os.close(fd)
            self._write_count += 1

    def append(self, relative: str, payload: bytes, *, sync: bool = False,
               failure: bool = False) -> None:
        with self._locked():
            _rel, path = self._relative(relative)
            self._prewrite(len(payload), failure=failure)
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | nofollow, 0o600)
            try:
                with os.fdopen(fd, "ab", closefd=False) as stream:
                    stream.write(payload)
                    stream.flush()
                    if sync:
                        os.fsync(stream.fileno())
            finally:
                os.close(fd)
            self._write_count += 1

    def verify_complete_namespace(self) -> dict[str, Any]:
        size = self._audit_and_size()
        if size > self.max_bytes:
            raise E123Error("Output namespace exceeds its authorized byte budget")
        return {"file_bytes": size, "max_bytes": self.max_bytes,
                "failure_reserve_bytes": self.failure_reserve_bytes,
                "write_operations": self._write_count,
                "unaccounted_files": 0}

    def failure(self, relative: str, payload: Mapping[str, Any]) -> None:
        try:
            _rel, path = self._relative(relative)
            if path.exists():
                return
            self.write_once(relative, canonical_bytes(dict(payload)), failure=True)
        except Exception:
            # The caller still reports the original failure. Reserved bytes are
            # deliberately not consumed by ordinary experiment artifacts.
            pass


def _is_reparse_point(path: Path) -> bool:
    try:
        attrs = getattr(path.lstat(), "st_file_attributes", 0)
        return bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    except OSError:
        return False


@dataclass(frozen=True)
class E123Event:
    position: int
    original_position: int
    relative_id: str
    image_identity_sha256: str
    source_group_sha256: str
    camera_id: int
    timestamp: str
    image_path: str
    sample_key: str


@dataclass(frozen=True)
class Condition:
    category: str
    support_seed: int
    order: str
    events: tuple[E123Event, ...]
    query_positions: frozenset[int]
    query_tape_sha256: str
    stream_identity_sha256: str
    support_images: tuple[str, ...]
    support_groups: tuple[str, ...]
    c64_images: tuple[str, ...]
    c64_groups: tuple[str, ...]
    class_json_sha256: str

    @property
    def condition_id(self) -> str:
        return f"{self.category}_s{self.support_seed}_{self.order}"


def _load_stage05_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    from experiments.calibration.r3_fullstream_2x2 import (
        STAGE05, _read_json as r3_read_json, load_frozen_contract)

    r3_frozen = load_frozen_contract(verify_image_paths=True)
    frozen = r3_read_json(STAGE05 / "frozen_inputs.json")
    return frozen, r3_frozen


def _opaque_group(raw_group: str) -> str:
    return sha256_bytes(("official-sample-group\0" + str(raw_group)).encode("utf-8"))


def _build_conditions() -> tuple[list[Condition], dict[str, Any]]:
    from experiments.calibration import query_tape
    from experiments.calibration.profiles import QueryProtocolConfig
    from experiments.calibration.protocol import TapeRecord
    from experiments.calibration.r3_fullstream_2x2 import (
        DATASET_ROOT, JSON_ROOT)
    from experiments.calibration.r3_fullstream_2x2 import STAGE05
    from experiments.calibration.stage05 import _absolute_image_path
    from experiments.calibration.r7_shuffled_correction_4class import deterministic_group_shuffle

    frozen, checked = _load_stage05_inputs()
    score_manifest = read_json(STAGE05 / "score_tape_manifest.json")
    config = QueryProtocolConfig()
    if (frozen.get("schema") != "stage05.frozen_inputs.v1" or
            frozen.get("freeze_verification_status") != "PASS" or
            set(frozen.get("classes", {})) != set(CATEGORIES)):
        raise E123Error("Frozen Stage 0.5 input receipt is incompatible")
    conditions: list[Condition] = []
    identities: dict[str, Any] = {}
    for category in CATEGORIES:
        base = frozen["classes"][category]
        checked_info = checked["classes"][category]
        expected_n = EXPECTED_COUNTS[category]
        class_json = JSON_ROOT / f"{category}.json"
        class_hash = sha256_file(class_json)
        if (class_hash != base.get("class_json_sha256") or
                len(base.get("stream", ())) != expected_n or
                checked_info["stream_count"] != expected_n):
            raise E123Error(f"{category}: public event inventory/hash differs from Stage 0.5")
        source_rows = []
        for position, row in enumerate(base["stream"]):
            if int(row.get("position", -1)) != position:
                raise E123Error(f"{category}: Stage 0.5 stream order is not contiguous")
            source_rows.append({
                "position": position,
                "relative_id": str(row["relative_id"]),
                "opaque_group": str(row["opaque_group"]),
                "camera_id": int(row["camera_id"]),
                "timestamp": str(row.get("timestamp", "")),
                "sample_key": str(row["sample_key"]),
                "image_path": str(_absolute_image_path(
                    DATASET_ROOT, base["prefix"], str(row["relative_id"]))),
            })
        if tuple(str(row["relative_id"]) for row in source_rows) == ():
            raise E123Error(f"{category}: empty frozen stream")
        category_inventory: dict[str, Any] = {
            "category_json_sha256": class_hash,
            "event_count": expected_n,
            "support_seeds": {},
        }
        c64 = base["c64"]
        c64_groups = tuple(sorted(_opaque_group(str(g)) for g in c64["groups"]))
        c64_images = tuple(str(v) for v in c64["images"])
        if len(c64_groups) != 64 or len(c64_images) != 64 or len(set(c64_groups)) != 64:
            raise E123Error(f"{category}: C64 receipt is incomplete or duplicated")
        event_groups = {str(row["opaque_group"]) for row in source_rows}
        if set(c64_groups) & event_groups:
            raise E123Error(f"{category}: C64 intersects TEST Source Groups")
        for seed in SUPPORT_SEEDS:
            support = base["supports"].get(str(seed))
            if not isinstance(support, dict) or support.get("all_train_normal") is not True:
                raise E123Error(f"{category}: Stage 0.5 support seed {seed} is not verified")
            support_groups = tuple(sorted(_opaque_group(str(g)) for g in support["groups"]))
            support_images = tuple(str(v) for v in support["images"])
            if (len(support_groups) != 4 or len(set(support_groups)) != 4 or
                    len(support_images) != 4 or len(set(support_images)) != 4 or
                    set(support_groups) & set(c64_groups) or
                    set(support_groups) & event_groups):
                raise E123Error(f"{category}: seed {seed} Support/C64/TEST roles overlap")
            conditions_for_support = []
            for order in STREAM_ORDERS:
                if order == "json":
                    ordered_rows = source_rows
                    events = tuple(E123Event(
                        position=i, original_position=i,
                        relative_id=str(row["relative_id"]),
                        image_identity_sha256=sha256_bytes(
                            ("e123-image-identity\0" + str(row["relative_id"])).encode("utf-8")),
                        source_group_sha256=str(row["opaque_group"]),
                        camera_id=int(row["camera_id"]), timestamp=str(row.get("timestamp", "")),
                        image_path=str(row["image_path"]), sample_key=str(row["sample_key"]))
                        for i, row in enumerate(ordered_rows))
                else:
                    shuffled = deterministic_group_shuffle(source_rows, seed=0)
                    events = tuple(E123Event(
                        position=i, original_position=int(row.original_position),
                        relative_id=str(row.relative_id),
                        image_identity_sha256=str(row.image_identity_sha256),
                        source_group_sha256=str(row.source_group_sha256),
                        camera_id=int(row.camera_id), timestamp=str(row.timestamp),
                        image_path=str(row.image_path), sample_key=str(
                            source_rows[int(row.original_position)]["sample_key"]))
                        for i, row in enumerate(shuffled))
                if len(events) != expected_n or len({e.relative_id for e in events}) != expected_n:
                    raise E123Error(f"{category}/{order}: stream order is not a bijection")
                records = [TapeRecord(
                    position=e.position,
                    official_sample_key=e.source_group_sha256,
                    camera_id=e.camera_id,
                    relative_id=e.image_identity_sha256,
                    timestamp=e.timestamp,
                    phase=0,
                ) for e in events]
                tape = query_tape.build_qa_ts(records, config=config)
                query_tape.verify_tape(tape, records, config=config)
                if (tape.seed != 37 or tape.policy != "Q-A-TS" or
                        len(tape.positions) != int(0.03 * expected_n + 0.5) or
                        len(tape.queried_groups) != len(tape.positions)):
                    raise E123Error(f"{category}/{order}: Q-A-TS contract mismatch")
                condition = Condition(
                    category, seed, order, events, tape.positions, tape.digest,
                    tape.stream_digest, support_images, support_groups,
                    c64_images, c64_groups, class_hash)
                conditions.append(condition)
                conditions_for_support.append({
                    "condition_id": condition.condition_id,
                    "stream_identity_sha256": condition.stream_identity_sha256,
                    "query_tape_sha256": condition.query_tape_sha256,
                    "query_count": len(condition.query_positions),
                    "event_count": len(condition.events),
                })
            category_inventory["support_seeds"][str(seed)] = {
                "support_group_sha256": sha256_json(list(support_groups)),
                "support_image_identity_sha256": sha256_json(list(support_images)),
                "support_group_count": 4,
                "c64_group_sha256": sha256_json(list(c64_groups)),
                "c64_image_identity_sha256": sha256_json(list(c64_images)),
                "c64_count": 64,
                "support_c64_test_disjoint": True,
                "conditions": conditions_for_support,
            }
        identities[category] = category_inventory
    if len(conditions) != 16 or len({c.condition_id for c in conditions}) != 16:
        raise E123Error("E123 condition registry must contain exactly sixteen pairs")
    # Same stream/order condition shares one tape across both Support seeds.
    for category in CATEGORIES:
        for order in STREAM_ORDERS:
            pair = [c for c in conditions if c.category == category and c.order == order]
            if len(pair) != 2 or pair[0].query_tape_sha256 != pair[1].query_tape_sha256:
                raise E123Error(f"{category}/{order}: paired Supports lack one common query tape")
    return conditions, identities


def _binary_metrics(labels: Sequence[int], scores: Sequence[float],
                    threshold: float) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or not len(y):
        raise E123Error("Metric inputs must be aligned nonempty vectors")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(s).all():
        raise E123Error("Metric input labels/scores are invalid")
    pred = (s >= float(threshold)).astype(np.int8)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fpr = fp / (fp + tn) if fp + tn else None
    fnr = fn / (fn + tp) if fn + tp else None
    ber = (fpr + fnr) / 2 if fpr is not None and fnr is not None else None
    return {
        "n": int(len(y)), "normal_count": int((y == 0).sum()),
        "defect_count": int((y == 1).sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fpr": fpr, "fnr": fnr, "ber": ber,
        "auroc": _auroc(y, s), "average_precision": _average_precision(y, s),
    }


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if not n_pos or not n_neg:
        return None
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    left = 0
    while left < len(s):
        right = left + 1
        while right < len(s) and sorted_scores[right] == sorted_scores[left]:
            right += 1
        ranks[order[left:right]] = (left + 1 + right) / 2
        left = right
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    if not positives:
        return None
    order = np.argsort(-s, kind="mergesort")
    sorted_y = y[order]
    true_positive = np.cumsum(sorted_y == 1)
    ranks = np.arange(1, len(y) + 1)
    precision = true_positive / ranks
    return float(precision[sorted_y == 1].sum() / positives)


def _fixed_offset_metric(labels: Sequence[int], scores: Sequence[float],
                         theta0: float, sigma0: float, coefficient: float) -> float:
    metrics = _binary_metrics(labels, scores, theta0 + coefficient * sigma0)
    if metrics["ber"] is None:
        raise E123Error("Fixed-offset selection has a category without both label classes")
    return float(metrics["ber"])


def _read_historical_development_labels(category: str, path: Path, *,
                                       expected_source_sha256: str,
                                       expected_count: int) -> list[int]:
    """Read only an authorized historical LOCO training fold, record-bounded.

    The historical decoder's mmap boundary check can see JSON whitespace after
    the final array record. E123 supplies each already-indexed selected record
    as a bounded byte slice and leaves the historical decoder unchanged.
    """
    from experiments.calibration.stage1a_p12_label_provider import (
        QueriedLabelAccessError, _RestrictedTestArray, _decode_selected_label,
        _skip_value, _ws)

    path = Path(path).resolve(strict=True)
    if sha256_file(path) != expected_source_sha256:
        raise E123Error(f"Historical development label source changed: {category}")
    source = _RestrictedTestArray(path)
    labels: list[int] = []
    try:
        data = source._data
        cursor = source._cursor
        while True:
            start = _ws(data, cursor)
            if start >= len(data) or data[start] == ord("]"):
                break
            stop = _skip_value(data, start)
            record = data[start:stop]
            labels.append(int(_decode_selected_label(
                record, 0, len(record), source.normal_class)))
            cursor = _ws(data, stop)
            if cursor < len(data) and data[cursor] == ord(","):
                cursor = _ws(data, cursor + 1)
            elif cursor < len(data) and data[cursor] == ord("]"):
                break
            else:
                raise E123Error("Malformed TEST array during historical LOCO label read")
    except QueriedLabelAccessError as exc:
        raise E123Error(
            f"Could not read the authorized historical {category} LOCO labels: {exc}") from exc
    finally:
        source.close()
    if len(labels) != int(expected_count):
        raise E123Error(
            f"Historical {category} label count differs: {len(labels)} != {expected_count}")
    return labels


def _load_retrospective_stage1a() -> tuple[dict[tuple[str, int], dict[str, Any]],
                                            dict[str, Any]]:
    """Read only sealed historical Stage 1-A scores and the permitted dev labels.

    This is retrospective development-only selection before E123 Stage A.  The
    target fold's category is skipped; no current E123 outcome or confirmation
    category is opened.
    """
    import numpy as np
    from experiments.calibration.stage1a_final_readiness import (
        CLASSES, P12_OUTPUT, load_frozen_trajectory_input)
    from experiments.calibration.controllers import theta0_from_c64
    from experiments.calibration.r3_fullstream_2x2 import JSON_ROOT

    complete = read_json(STAGE1A_ROOT / "prediction_phase_complete.json")
    receipts = read_json(STAGE1A_ROOT / "prediction_receipts.json")
    if (complete.get("status") != "PREDICTION_PHASE_COMPLETE_LABEL_FIREWALL_INTACT" or
            complete.get("event_count") != 76475 or complete.get("trajectory_count") != 20 or
            complete.get("outcome_labels_accessed") is not False or
            receipts.get("status") != "PASS_ALL_20_CAUSAL_REPLAYS" or
            receipts.get("native_or_gpu_execution") is not False):
        raise E123Error("Historical Stage 1-A prediction phase is not sealed and compatible")
    if sha256_file(STAGE1A_ROOT / "prediction_receipts.json") != complete.get(
            "prediction_receipts_sha256"):
        raise E123Error("Historical Stage 1-A receipt checksum mismatch")
    with (STAGE1A_ROOT / "checksums.json").open("r", encoding="utf-8") as stream:
        checksums = json.load(stream).get("files", {})
    trajectories: dict[tuple[str, int], dict[str, Any]] = {}
    category_sources: dict[str, Any] = {}
    receipt_rows = {(str(row["category"]), int(row["support_seed"])): row
                    for row in receipts["trajectories"]}
    for category in CLASSES:
        if category not in CATEGORIES:
            continue
        score_path = JSON_ROOT / f"{category}.json"
        category_sources[category] = {
            "source_json_sha256": sha256_file(score_path),
            "expected_event_count": EXPECTED_COUNTS[category],
            "source": "retrospective Stage1-A development evaluator after sealed predictions; decoded only when a category is in a training fold",
        }
        for seed in range(5):
            key = (category, seed)
            receipt = receipt_rows.get(key)
            if receipt is None:
                raise E123Error(f"Missing Stage 1-A trajectory receipt: {key}")
            relative = str(receipt["prediction_file"])
            path = STAGE1A_ROOT / relative
            digest = sha256_file(path)
            if digest != receipt.get("prediction_file_sha256") or checksums.get(relative) != digest:
                raise E123Error(f"Stage 1-A score trajectory hash mismatch: {relative}")
            p12_receipt_path = P12_OUTPUT / "shards" / category / f"seed{seed}.receipt.json"
            p12_receipt = read_json(p12_receipt_path)
            trajectory_input = load_frozen_trajectory_input(
                category=category, support_seed=seed, receipt=p12_receipt)
            c64_scores = [anchor.score for anchor in trajectory_input.initial_evidence.normal_anchors
                          if anchor.protected]
            if len(c64_scores) != 64:
                raise E123Error(f"Stage 1-A C64 anchor count is not 64: {key}")
            theta0 = float(theta0_from_c64(c64_scores))
            sigma0 = max(float(np.std(c64_scores, ddof=1)), 0.02)
            score_rows: list[float] = []
            time_index_base: int | None = None
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for position, line in enumerate(stream):
                    row = json.loads(line)
                    t_value = int(row.get("t", -1))
                    if time_index_base is None:
                        time_index_base = t_value - position
                    if time_index_base not in (0, 1) or t_value != position + time_index_base:
                        raise E123Error(f"Stage 1-A score positions are not ordered: {key}")
                    value = float(row["dynamic_score"])
                    if not math.isfinite(value):
                        raise E123Error(f"Stage 1-A has a nonfinite dynamic score: {key}")
                    score_rows.append(value)
            if len(score_rows) != EXPECTED_COUNTS[category]:
                raise E123Error(f"Stage 1-A event count differs: {key}")
            trajectories[key] = {
                "scores": score_rows, "theta0": theta0, "sigma0": sigma0,
                "score_sha256": digest, "receipt_sha256": sha256_file(p12_receipt_path),
                "query_tape_binding_sha256": str(receipt.get("query_chain_verified_against_p12_receipt")),
            }
    return trajectories, {
        "prediction_phase_complete_sha256": sha256_file(
            STAGE1A_ROOT / "prediction_phase_complete.json"),
        "prediction_receipts_sha256": sha256_file(STAGE1A_ROOT / "prediction_receipts.json"),
        "category_sources": category_sources,
        "scope": "four registered development categories only; per target fold, target labels excluded",
    }


def select_fixed_offsets() -> dict[str, Any]:
    trajectories, provenance = _load_retrospective_stage1a()
    selections: dict[str, Any] = {}
    all_candidates: dict[str, Any] = {}
    for target in CATEGORIES:
        training_categories = [category for category in CATEGORIES if category != target]
        if target in training_categories:
            raise E123Error("Fixed-offset target leakage in leave-one-category-out selection")
        labels: dict[str, list[int]] = {}
        from experiments.calibration.r3_fullstream_2x2 import JSON_ROOT
        for category in training_categories:
            path = JSON_ROOT / f"{category}.json"
            labels[category] = _read_historical_development_labels(
                category, path,
                expected_source_sha256=provenance["category_sources"][category][
                    "source_json_sha256"],
                expected_count=EXPECTED_COUNTS[category])
        candidate_summary = []
        for coefficient in FIXED_OFFSET_GRID:
            category_ber = []
            for category in training_categories:
                seed_values = []
                for seed in range(5):
                    trajectory = trajectories[(category, seed)]
                    seed_values.append(_fixed_offset_metric(
                        labels[category], trajectory["scores"], trajectory["theta0"],
                        trajectory["sigma0"], coefficient))
                category_ber.append(sum(seed_values) / len(seed_values))
            candidate_summary.append({
                "coefficient": coefficient,
                "training_categories": training_categories,
                "category_equal_mean_ber": sum(category_ber) / len(category_ber),
                "category_mean_ber": dict(zip(training_categories, category_ber)),
            })
        winner = min(candidate_summary, key=lambda row: (
            row["category_equal_mean_ber"], abs(row["coefficient"]), row["coefficient"]))
        selections[target] = {
            "coefficient": float(winner["coefficient"]),
            "selection_status": "SELECTED_RETROSPECTIVELY_ON_OTHER_THREE_DEVELOPMENT_CATEGORIES",
            "training_categories": training_categories,
            "target_category_outcomes_accessed_for_selection": False,
            "training_category_mean_ber": winner["category_mean_ber"],
            "training_macro_ber": winner["category_equal_mean_ber"],
            "tie_rule": FIXED_OFFSET_TIE_RULE,
        }
        all_candidates[target] = candidate_summary
    return {
        "schema": "recovery.e123.fixed_offset_selection.v1",
        "candidate_grid": list(FIXED_OFFSET_GRID),
        "selection_rule": "minimize mean of three category BERs; per-category BER averages five archived K256 support-seed trajectories",
        "threshold_formula": "theta_fixed = theta0 + coefficient * sigma0",
        "sigma0_formula": "max(sample_std(C64 M0 scores, ddof=1), 0.02)",
        "tie_rule": FIXED_OFFSET_TIE_RULE,
        "historical_exposure_disclosure": "All four categories were previously exposed development data; selection is retrospective and is not blind confirmation.",
        "provenance": provenance,
        "selection_by_target_category": selections,
        "candidate_results_by_target_category": all_candidates,
    }


def build_contract(run_id: str) -> tuple[dict[str, Any], list[Condition]]:
    identity = source_identity()
    conditions, identities = _build_conditions()
    fixed_offsets = select_fixed_offsets()
    try:
        from experiments.calibration.r3_fullstream_2x2 import _get_verified_model_receipt
        model_receipt = _get_verified_model_receipt()
    except Exception as exc:
        raise E123Error(f"Verified native DINO/model receipt is unavailable: {exc}") from exc
    contract = {
        "schema": CONTRACT_SCHEMA,
        "run_id": run_id,
        "authorization": "Researcher-authorized E123 Regime B development campaign; E1/E2/E3 only; no D3 or confirmation data.",
        "source_identity": identity,
        "historical_evidence_separate_from_regime_b": {
            "receipts": _historical_evidence_receipts(),
            "strict4_results_numerically_pooled": False,
            "a0_a1_dynamic_capacity_per_layer": 128,
            "a0_a1_aggregate_K_means": "number of queries, not dynamic memory capacity",
            "full_hierarchy_arm_rerun": False,
            "historical_hierarchy_decision": "retained as separate bounded negative evidence; not reopened by E123",
            "historical_stage1a_k256": "used only for preregistered leave-one-category-out Fixed Offset selection and reported separately from K1024",
        },
        "requested_starting_parent": REQUESTED_PARENT,
        "actual_starting_parent": identity["actual_parent_sha"],
        "starting_parent_reconciliation": (
            "Actual HEAD is the direct D3-P0 documentation-only child of requested parent; "
            "D3 paths remain excluded and E123 is bound to actual post-D3 HEAD."),
        "model_receipt": model_receipt,
        "dataset": {
            "development_categories": dict(EXPECTED_COUNTS),
            "dataset_root_fingerprint": "Stage05 verified local Real-IAD paths; no dataset files copied to outputs",
            "identity_basis": "official Real-IAD sample key as frozen/opaque Stage05 Source Group; physical identity beyond official key is not independently established",
        },
        "design": {
            "support_seeds": list(SUPPORT_SEEDS),
            "stream_orders": list(STREAM_ORDERS),
            "paired_conditions": len(conditions),
            "condition_rule": "4 categories x 2 supports x 2 orders = 16; methods share a condition's one query tape",
            "regime": "four eligible training-normal Support images plus 64 additional fixed labelled training-normal C64 references",
            "support_c64_test_disjointness": identities,
            "query_protocol": {
                "name": "Q-A-TS-v1", "target_rate": 0.03, "seed": 37,
                "bins": 16, "rounding": "half-up floor(0.03*N + 0.5)",
                "group_constraint": "one query per official Source Group",
                "forced_initial_queries": False,
                "same_tape_across_support_seed_pair": True,
                "tape_bindings": [
                    {"condition_id": c.condition_id,
                     "query_count": len(c.query_positions),
                     "query_tape_sha256": c.query_tape_sha256}
                    for c in conditions
                ],
            },
            "native_scorer": {
                "configuration_id": "R7/E6 frozen native P0 profile",
                "feature_fuse": "concat", "augmentations": 4,
                "fixed_capacity": 25000, "fixed_per_image": 5000,
                "include_identity": False, "topk_patches": 32,
                "image_aggregation": "max", "image_topk": 64,
                "image_quantile": 0.95, "knn": 5,
                "runtime": "impl.set_seed(0); Flash SDP off; memory-efficient SDP off; math SDP on; verified cached DINOv2 ViT-L/14 registers",
            },
            "memory_comparison": {
                "configurations": list(CONFIGURATION_IDS),
                "candidate_source": "one shared static-support native map; top 8 patch IDs per fused layer per queried normal",
                "candidate_order": "ascending native anomaly-map rank order returned by layer_to_topk_from_amap",
                "eligibility": "queried and verified normal label only; same eligible event set for every mutable method; no threshold-controller gating",
                "mutable_updates": "all methods receive identical candidate IDs/features in identical order; only mutable retention rule differs",
                "greedy_fidelity": "common static-candidate Greedy coreset; not exact R7/E6 native Greedy candidate-selection parity",
                "utility_stm": "original v22 utility-aware FixedCapacityCodebook.update; LTM, maturation and correction disabled",
                "fixed_threshold": "C64 higher empirical q90 at M0, shared by all configurations in a paired condition",
                "registered_persistent_bytes_tolerance": BYTE_TOLERANCE,
                "capacity_primary": 1024,
                "capacity_sensitivity": [256, 512],
                "primary_metric": "category-equal prequential AUROC",
            },
            "e1": {
                "matrix": ["STATIC/FROZEN_C64", "STATIC/B3_CURRENT_STATE_C64",
                           "GREEDY_K1024/FROZEN_C64", "GREEDY_K1024/B3_CURRENT_STATE_C64"],
                "support_self_matching_diagnostic": (
                    "compare higher empirical q90 from the four Support self-scores with the legal C64 higher-q90 theta0; "
                    "diagnostic only, never used for online predictions or memory eligibility"),
                "interaction": "(Greedy B3 BER - Greedy Frozen BER) - (Static B3 BER - Static Frozen BER); observed score-by-threshold interaction only",
            },
            "e3": {
                "controllers": ["FROZEN_C64", "FIXED_OFFSET", "B1_ROLLING_QUERIED_NORMAL",
                                "B2_ORIGINAL_SIDECAR_ADAPTED", "B3_CURRENT_STATE_C64"],
                "fixed_offset_selection": fixed_offsets,
                "b1": "rolling q90 of last 256 queried verified-normal stale scores; minimum 16",
                "b2": "original v22 Sidecar with unqueried normal fallback disabled",
                "b3": "protected C64 rescored against current Greedy-K1024 bank; delta 0.01 per memory version; update active next event",
                "b3_snapshots": "C64 score snapshots persisted after each actual Greedy memory version change; no extra DINO inference in Stage B",
                "same_score_auc_invariant": True,
            },
            "excluded": ["D3", "confirmation categories", "VisA", "MVTec AD",
                         "R1-R10 reruns", "E6 rerun", "D2 rerun", "BEA", "Signed Correction",
                         "defect memory", "pseudo-label writes"],
        },
        "output_controls": {
            "max_bytes": MAX_OUTPUT_BYTES,
            "failure_reserve_bytes": FAILURE_RESERVE_BYTES,
            "minimum_free_bytes_before_launch": MIN_FREE_BYTES,
            "prewrite_accounting": True,
            "unique_non_overwriting_namespace": True,
            "resume": False,
            "supervisor": "Windows Job Object; 120 minutes per Stage A condition; no auto-restart",
        },
        "protocol_parity_claims": {
            "historical_static_native_source": "R7/E6 native v22 VisionAD scorer; full manuscript baseline parity must be tested from receipts and reported separately",
            "native_greedy_exact_parity": False,
            "historical_fixed_bank_equivalence": "verified per support-0 condition against structured Stage 0.5 digest at runtime; support-1 has no historical same-bank receipt",
            "historical_negative_results_preserved": ["R3 correction-negative", "R7 correction-positive", "FPSA stop decision"],
        },
        "created_utc_epoch": time.time(),
        "stage_a_started": True,
    }
    contract["contract_sha256"] = sha256_json(contract)
    return contract, conditions


def _verify_contract(contract: Mapping[str, Any], *, expected_run_id: str | None = None) -> None:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise E123Error("E123 contract schema mismatch")
    if expected_run_id is not None and contract.get("run_id") != expected_run_id:
        raise E123Error("E123 contract run ID mismatch")
    copy_contract = dict(contract)
    digest = str(copy_contract.pop("contract_sha256", ""))
    if not digest or sha256_json(copy_contract) != digest:
        raise E123Error("E123 contract checksum mismatch")
    identity = source_identity()
    if (identity["commit_sha"] != contract["source_identity"]["commit_sha"] or
            identity["source_sha256"] != contract["source_identity"]["source_sha256"]):
        raise E123Error("E123 execution source identity changed after contract freeze")


def _preflight_disk(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path.parent)
    if usage.free < MIN_FREE_BYTES:
        raise E123Error(f"Insufficient free space: {usage.free} < {MIN_FREE_BYTES}")
    return {"free_bytes": usage.free, "required_free_bytes": MIN_FREE_BYTES,
            "total_bytes": usage.total}


def validate_role_disjointness(support_groups: Sequence[str], c64_groups: Sequence[str],
                               test_groups: Sequence[str]) -> None:
    support, c64, test = set(support_groups), set(c64_groups), set(test_groups)
    if (len(support) != 4 or len(c64) != 64 or
            support & c64 or support & test or c64 & test):
        raise E123Error("Regime B Support/C64/TEST roles are incomplete or overlap")


def common_write_eligible(*, queried: bool, label: int | None) -> bool:
    """Frozen E2 eligibility: only a queried, verified normal can write."""
    if not queried:
        if label is not None:
            raise E123Error("Unqueried event carries a label")
        return False
    if label not in (0, 1):
        raise E123Error("Queried feedback is not a verified binary label")
    return label == 0


def resolve_common_eligibility(*, queried: bool, label: int | None,
                               c0_score: float, theta0: float, sigma0: float,
                               eligibility: Any) -> tuple[bool, int, str, Mapping[str, Any] | None]:
    """Resolve one frozen C0 gate shared by all mutable E2 arms."""
    if not common_write_eligible(queried=queried, label=label):
        reason = "unqueried_event" if not queried else "verified_defect"
        return False, 0, reason, None
    allow, keep, reason, info = eligibility(
        score=float(c0_score), theta=float(theta0), sigmaN=max(float(sigma0), 0.02),
        pred_defect=bool(float(c0_score) >= float(theta0)), novelty=0.0)
    return bool(allow), int(keep), str(reason), dict(info)


def fixed_offset_threshold(theta0: float, sigma0: float, coefficient: float) -> float:
    value = float(theta0) + float(coefficient) * float(sigma0)
    if not math.isfinite(value):
        raise E123Error("Fixed-offset threshold is nonfinite")
    return value


def prepare_common_candidates(impl: Any, features: Sequence[Any],
                              candidate_ids: Mapping[int, Sequence[int]],
                              *, device: str) -> dict[int, tuple[list[int], Any]]:
    """Build one shared normalized candidate tensor per fused layer."""
    import torch
    result: dict[int, tuple[list[int], Any]] = {}
    for layer, raw_ids in candidate_ids.items():
        ids = [int(value) for value in raw_ids]
        if not ids or layer < 0 or layer >= len(features):
            raise E123Error("Common candidate layer/index list is invalid")
        normalized = impl.l2_normalize(features[layer].to(device))
        if min(ids) < 0 or max(ids) >= int(normalized.shape[0]):
            raise E123Error("Common candidate patch ID is outside the fused map")
        index = torch.as_tensor(ids, device=normalized.device, dtype=torch.long)
        result[int(layer)] = (ids, normalized.index_select(0, index))
    if not result:
        raise E123Error("Common candidate set cannot be empty")
    return result


def dispatch_common_write(states: Mapping[str, Any], *, eligible: bool,
                          observe_kwargs: Mapping[str, Any],
                          synchronize: Any = None) -> dict[str, tuple[bool, float]]:
    """Use one eligibility decision and one candidate object for every arm."""
    if not eligible:
        return {}
    common_candidates = observe_kwargs.get("common_candidates")
    if not isinstance(common_candidates, Mapping):
        raise E123Error("Shared writer needs one common candidate mapping")
    results = {}
    for name, state in states.items():
        started = time.perf_counter()
        changed = bool(state.observe(**dict(observe_kwargs)))
        if synchronize is not None:
            synchronize()
        results[name] = (changed, time.perf_counter() - started)
    return results


def disclose_after_score_seal(scores_out: Any, *, queried: bool,
                              label_reader: Any, category: str,
                              original_position: int, source_key: str) -> int | None:
    """Disclose no label unless the current complete score row is durable."""
    if not queried:
        return None
    scores_out.flush(sync=True)
    return int(label_reader(category, original_position, source_key))


class RandomAccessQueriedLabelReader:
    """Index TEST record boundaries label-blindly; decode only authorized rows.

    Existing p12 lexical helpers skip strings/scalars without decoding them.
    Selected rows are passed as a bounded bytes slice to the historical decoder,
    preventing its mmap-wide trailing-whitespace boundary bug.
    """

    def __init__(self, *, category: str, path: Path, expected_source_sha256: str,
                 allowed_positions: Iterable[int],
                 group_sha256_by_position: Mapping[int, str]):
        import mmap
        from experiments.calibration.stage1a_p12_label_provider import (
            QueriedLabelAccessError, _RestrictedTestArray, _skip_value, _ws)

        self.category = category
        self.path = Path(path).resolve(strict=True)
        self.expected_source_sha256 = expected_source_sha256
        if sha256_file(self.path) != expected_source_sha256:
            raise E123Error("Category JSON source hash changed before label-reader setup")
        raw = tuple(int(p) for p in allowed_positions)
        self.allowed = set(raw)
        self.group_sha256_by_position = {int(k): str(v)
                                        for k, v in group_sha256_by_position.items()}
        if len(self.allowed) != len(raw) or self.allowed != set(self.group_sha256_by_position):
            raise E123Error("Query-label allowlist and Source Group receipt differ")
        self._p12 = __import__(
            "experiments.calibration.stage1a_p12_label_provider",
            fromlist=["_decode_selected_label", "_RestrictedTestArray"])
        self._array = _RestrictedTestArray(self.path)
        self._data = self._array._data
        self.normal_class = self._array.normal_class
        cursor = self._array._cursor
        offsets: list[tuple[int, int]] = []
        try:
            while True:
                start = _ws(self._data, cursor)
                if start >= len(self._data) or self._data[start] == ord("]"):
                    break
                stop = _skip_value(self._data, start)
                offsets.append((start, stop))
                cursor = _ws(self._data, stop)
                if cursor < len(self._data) and self._data[cursor] == ord(","):
                    cursor = _ws(self._data, cursor + 1)
                elif cursor < len(self._data) and self._data[cursor] == ord("]"):
                    break
                else:
                    raise E123Error("Malformed test array while building label-blind offsets")
        except QueriedLabelAccessError as exc:
            self._array.close()
            raise E123Error(f"Could not index TEST array label-blindly: {exc}") from exc
        self.offsets = tuple(offsets)
        if not self.allowed.issubset(range(len(self.offsets))):
            self._array.close()
            raise E123Error("Authorized query position falls outside the frozen TEST array")
        self.decoded_positions: list[int] = []
        self._decoded: set[int] = set()

    def __call__(self, category: str, position: int, source_key: str) -> int:
        if category != self.category or position not in self.allowed or position in self._decoded:
            raise E123Error("Queried label request is outside its one-use authorization")
        from experiments.calibration.stage1a_p12_label_provider import _decode_selected_label
        expected_group = sha256_bytes(
            ("official-sample-group\0" + str(source_key)).encode("utf-8"))
        if self.group_sha256_by_position.get(position) != expected_group:
            raise E123Error("Queried label Source Group differs from the frozen tape")
        start, stop = self.offsets[position]
        record = self._data[start:stop]
        label = int(_decode_selected_label(record, 0, len(record), self.normal_class))
        self._decoded.add(position)
        self.decoded_positions.append(position)
        return label

    def stage_b_all_labels(self, *, global_seal: Mapping[str, Any]) -> list[int]:
        try:
            require_complete_stage_a(global_seal)
        except E123Error as exc:
            raise E123Error(
                "Stage B label access requires a valid complete global Stage A seal") from exc
        from experiments.calibration.stage1a_p12_label_provider import _decode_selected_label
        labels = []
        for position, (start, stop) in enumerate(self.offsets):
            record = self._data[start:stop]
            labels.append(int(_decode_selected_label(
                record, 0, len(record), self.normal_class)))
        return labels

    def close(self) -> None:
        self._array.close()


def _bank_dynamic_codebooks(bank: Any) -> list[Any]:
    return list(bank.normal_dyn["p0"][0]["id"])


def _clone_bank_with_capacity(bank: Any, capacity: int) -> Any:
    import torch

    clone = copy.copy(bank)
    clone.normal_fixed = bank.normal_fixed
    clone.normal_dyn = copy.deepcopy(bank.normal_dyn)
    clone.corr_cfg = copy.copy(bank.corr_cfg)
    clone.corr_cfg.enabled = False
    clone.use_defect_bank = False
    for attr in ("enable_ltm_retrieval", "enable_ltm_promotion",
                 "enable_maturation_buffer_retrieval", "enable_maturation_buffer",
                 "enable_utility_aware_stm_retention"):
        if hasattr(clone, attr):
            setattr(clone, attr, False)
    for codebook in _bank_dynamic_codebooks(clone):
        codebook.K = int(capacity)
        device = getattr(codebook, "device", getattr(bank, "device", "cuda"))
        dtype = getattr(codebook, "dtype", torch.float16)
        feature_dim = int(codebook.protos.shape[1]) if codebook.protos.ndim == 2 else 4096
        codebook.protos = torch.empty((0, feature_dim), device=device, dtype=dtype)
    return clone


def _tensor_storage_bytes(tensor: Any) -> int:
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:
        return int(tensor.numel() * tensor.element_size())


def _python_metadata_bytes(value: Any, seen: set[int] | None = None) -> int:
    import sys
    import torch

    if seen is None:
        seen = set()
    ident = id(value)
    if ident in seen:
        return 0
    seen.add(ident)
    if isinstance(value, torch.Tensor):
        return sys.getsizeof(value) + _tensor_storage_bytes(value)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_python_metadata_bytes(k, seen) + _python_metadata_bytes(v, seen)
                    for k, v in value.items())
    elif isinstance(value, (list, tuple, set, frozenset, deque)):
        size += sum(_python_metadata_bytes(item, seen) for item in value)
    elif hasattr(value, "__dict__"):
        size += _python_metadata_bytes(vars(value), seen)
    return size


class E123MemoryState:
    """One native bank and one retention policy with shared-candidate inputs."""

    def __init__(self, *, method: str, capacity: int, bank: Any, seed: int):
        import random as _random

        if method not in MUTABLE_METHODS or capacity not in CAPACITIES:
            raise E123Error("Unregistered E123 memory arm")
        self.method = method
        self.capacity = int(capacity)
        self.bank = bank
        self.rng = _random.Random(int(seed))
        self.seen_by_layer = [0 for _ in _bank_dynamic_codebooks(bank)]
        self.fifo_cursor = [0 for _ in self.seen_by_layer]
        self.candidate_events = 0
        self.candidate_patches = 0
        self.accepted_patches = 0
        self.rejected_patches = 0
        self.replaced_patches = 0
        self.memory_version = 0
        self.last_actions: list[dict[str, Any]] = []
        self.state_digest = sha256_bytes(
            f"e123-memory-empty|{method}|{capacity}|{seed}".encode("utf-8"))

    def occupancies(self) -> list[int]:
        return [int(cb.size()) for cb in _bank_dynamic_codebooks(self.bank)]

    def _append(self, codebook: Any, feature: Any) -> None:
        import torch
        row = feature.reshape(1, -1).to(codebook.device, dtype=codebook.dtype)
        codebook.protos = torch.cat((codebook.protos, row), dim=0)

    def observe(self, *, impl: Any, common_candidates: Mapping[int, tuple[list[int], Any]],
                event_position: int, static_score: float, theta0: float,
                sigma0: float, eligibility_info: Mapping[str, Any]) -> bool:
        """Apply the same precomputed IDs/features to this arm after a legal write."""
        if self.method not in MUTABLE_METHODS:
            raise E123Error("A nonmutable arm was sent a write")
        self.candidate_events += 1
        self.last_actions = []
        changed = False
        for layer, codebook in enumerate(_bank_dynamic_codebooks(self.bank)):
            candidate = common_candidates.get(layer)
            if candidate is None:
                continue
            patch_ids, features = candidate
            self.candidate_patches += len(patch_ids)
            if len(patch_ids) != len(features):
                raise E123Error("Candidate patch IDs/features are not aligned")
            if self.method == "UTILITY_STM":
                prior_size = int(codebook.size())
                report = codebook.update(
                    features, tau_insert=.15, tau_replace=.25, allow_replace=True,
                    use_utility_retention=True,
                    utility_score=float(eligibility_info["write_value"]),
                    outcome="FP" if static_score >= theta0 else "TN",
                    boundary_value=float(eligibility_info["boundary"]), event_id=int(event_position),
                    step=int(event_position), coverage_gain=0.0)
                added = int(report.get("n_added", 0))
                replaced = int(report.get("n_replaced", 0))
                accepted = added + replaced
                self.accepted_patches += accepted
                self.replaced_patches += replaced
                self.rejected_patches += max(0, len(patch_ids) - accepted)
                changed = changed or accepted > 0 or int(codebook.size()) != prior_size
                self.last_actions.append({
                    "layer": layer, "patch_ids": list(patch_ids),
                    "common_eligibility": True, "added": added,
                    "replaced": replaced, "rejected": max(0, len(patch_ids) - accepted),
                })
                continue
            for patch_id, feature in zip(patch_ids, features):
                self.seen_by_layer[layer] += 1
                n = int(codebook.size())
                K = int(codebook.K)
                action = "append"
                if n < K:
                    self._append(codebook, feature)
                    self.accepted_patches += 1
                    changed = True
                elif self.method == "FIFO":
                    slot = self.fifo_cursor[layer] % K
                    codebook.protos[slot] = feature.to(
                        device=codebook.device, dtype=codebook.dtype)
                    self.fifo_cursor[layer] = (slot + 1) % K
                    self.accepted_patches += 1
                    self.replaced_patches += 1
                    action = f"fifo_replace:{slot}"
                    changed = True
                elif self.method == "RESERVOIR":
                    slot = self.rng.randrange(self.seen_by_layer[layer])
                    if slot < K:
                        codebook.protos[slot] = feature.to(
                            device=codebook.device, dtype=codebook.dtype)
                        self.accepted_patches += 1
                        self.replaced_patches += 1
                        action = f"reservoir_replace:{slot}"
                        changed = True
                    else:
                        self.rejected_patches += 1
                        action = "reservoir_reject"
                elif self.method == "RANDOM_REPLACEMENT":
                    slot = self.rng.randrange(K)
                    codebook.protos[slot] = feature.to(
                        device=codebook.device, dtype=codebook.dtype)
                    self.accepted_patches += 1
                    self.replaced_patches += 1
                    action = f"random_replace:{slot}"
                    changed = True
                elif self.method == "GREEDY":
                    from experiments.a0_a1_memory_methods import kcenter_choice
                    slot = kcenter_choice(codebook.protos, feature)
                    if slot is None:
                        self.rejected_patches += 1
                        action = "greedy_reject"
                    else:
                        codebook.protos[int(slot)] = feature.to(
                            device=codebook.device, dtype=codebook.dtype)
                        self.accepted_patches += 1
                        self.replaced_patches += 1
                        action = f"greedy_replace:{int(slot)}"
                        changed = True
                self.last_actions.append({"layer": layer, "patch_id": int(patch_id),
                                          "action": action})
            if int(codebook.size()) > int(codebook.K):
                raise E123Error("Mutable memory exceeded its registered capacity")
        if changed:
            self.memory_version += 1
            self.state_digest = sha256_bytes(canonical_bytes({
                "previous": self.state_digest, "memory_version": self.memory_version,
                "event_position": int(event_position), "method": self.method,
                "capacity": self.capacity, "actions": self.last_actions,
                "candidate_ids": {str(k): v[0] for k, v in common_candidates.items()},
            }))
        return changed

    def audit(self) -> dict[str, Any]:
        import torch
        codebooks = _bank_dynamic_codebooks(self.bank)
        prototype_bytes = sum(_tensor_storage_bytes(cb.protos) for cb in codebooks)
        metadata_tensors: dict[int, Any] = {}
        for codebook in codebooks:
            for name, value in vars(codebook).items():
                if name != "protos" and isinstance(value, torch.Tensor):
                    try:
                        storage_key = int(value.untyped_storage()._cdata)
                    except Exception:
                        storage_key = id(value)
                    metadata_tensors[storage_key] = value
        metadata_tensor_bytes = sum(_tensor_storage_bytes(value)
                                    for value in metadata_tensors.values())
        metadata_bytes = _python_metadata_bytes({
            "rng": self.rng.getstate(), "seen_by_layer": self.seen_by_layer,
            "fifo_cursor": self.fifo_cursor, "candidate_events": self.candidate_events,
            "candidate_patches": self.candidate_patches,
            "accepted_patches": self.accepted_patches,
            "rejected_patches": self.rejected_patches,
            "replaced_patches": self.replaced_patches,
            "memory_version": self.memory_version,
            "last_actions": self.last_actions,
            "codebook_metadata": [
                {k: v for k, v in vars(cb).items() if k != "protos"}
                for cb in codebooks],
        })
        layers = []
        for cb in codebooks:
            layers.append({
                "occupied_slots": int(cb.size()), "capacity": int(cb.K),
                "prototype_shape": list(cb.protos.shape),
                "dtype": str(cb.protos.dtype),
                "prototype_tensor_bytes": int(cb.protos.numel() * cb.protos.element_size()),
                "allocated_storage_bytes": _tensor_storage_bytes(cb.protos),
            })
        return {
            "method": self.method, "capacity_per_layer": self.capacity,
            "layers": layers, "prototype_tensor_bytes": prototype_bytes,
            "metadata_tensor_bytes": metadata_tensor_bytes,
            "allocated_persistent_tensor_bytes": prototype_bytes + metadata_tensor_bytes,
            "metadata_runtime_bytes_estimate": metadata_bytes,
            "python_metadata_bytes_estimate": max(0, metadata_bytes - metadata_tensor_bytes),
            "persistent_mutable_bytes_estimate": prototype_bytes + metadata_bytes,
            "candidate_events": self.candidate_events,
            "candidate_patches": self.candidate_patches,
            "accepted_patches": self.accepted_patches,
            "rejected_patches": self.rejected_patches,
            "replaced_patches": self.replaced_patches,
            "memory_version": self.memory_version,
            "state_transition_sha256": self.state_digest,
            "dtype": str(codebooks[0].protos.dtype) if codebooks else str(torch.float16),
        }


def _stable_seed(*values: Any) -> int:
    return int(sha256_bytes("|".join(map(str, values)).encode("utf-8"))[:16], 16)


def _common_candidate_digest(candidates: Mapping[int, tuple[list[int], Any]]) -> str:
    return sha256_json({str(layer): [int(value) for value in ids]
                        for layer, (ids, _features) in candidates.items()})


def _candidate_feature_digest(candidates: Mapping[int, tuple[list[int], Any]]) -> str:
    digest = hashlib.sha256()
    for layer in sorted(candidates):
        ids, tensor = candidates[layer]
        cpu = tensor.detach().to("cpu").contiguous()
        digest.update(canonical_bytes({"layer": layer, "ids": ids,
                                       "shape": list(cpu.shape), "dtype": str(cpu.dtype)}))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _fixed_digest_and_profile(bank: Any, p0: Any) -> tuple[str, dict[str, Any]]:
    from experiments.calibration.native import _fixed_bank_digest
    profile = p0.fixed_bank_profile(bank)
    return _fixed_bank_digest(bank), profile


class BufferedJsonl:
    def __init__(self, budget: OutputBudget, relative: str, *, flush_every: int = 128):
        self.budget, self.relative = budget, relative
        self.flush_every = max(1, int(flush_every))
        self.pending: list[bytes] = []

    def add(self, value: Mapping[str, Any]) -> bytes:
        payload = canonical_bytes(dict(value))
        self.pending.append(payload)
        if len(self.pending) >= self.flush_every:
            self.flush(sync=False)
        return payload

    def flush(self, *, sync: bool) -> None:
        if self.pending:
            self.budget.append(self.relative, b"".join(self.pending), sync=sync)
            self.pending.clear()


class BudgetedTextLog:
    """Text stream adapter that sends child logs through the byte accountant."""

    def __init__(self, budget: OutputBudget, relative: str):
        self.budget, self.relative = budget, relative
        self.pending = bytearray()

    def write(self, value: str) -> int:
        self.pending.extend(str(value).encode("utf-8", errors="replace"))
        if len(self.pending) >= 4096:
            self.flush()
        return len(value)

    def flush(self) -> None:
        if self.pending:
            self.budget.append(self.relative, bytes(self.pending), sync=False)
            self.pending.clear()

    def isatty(self) -> bool:
        return False


def _condition_by_id(conditions: Sequence[Condition], condition_id: str) -> Condition:
    matches = [condition for condition in conditions
               if condition.condition_id == condition_id]
    if len(matches) != 1:
        raise E123Error(f"Unknown or ambiguous condition ID: {condition_id}")
    return matches[0]


def _native_profile() -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    import experiments.p0_protocol_audit as p0
    from experiments.calibration import profiles
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r3_fullstream_2x2 as r3

    impl = p0.require_sources()
    impl.set_seed(0)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    expected_model = r3._get_verified_model_receipt()
    restore_hub, model_source = r3._runtime_and_native_setup(
        torch, impl, expected_model)
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise E123Error("E123 requires the verified single CUDA device")
        backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
    finally:
        restore_hub()
    actual_model_state = r1._model_state_sha256(backbone.model)
    if actual_model_state != expected_model.get("loaded_model_state_sha256"):
        raise E123Error(
            "Loaded DINO state fingerprint differs from the verified model receipt: "
            f"expected={expected_model.get('loaded_model_state_sha256')}; "
            f"actual={actual_model_state}")
    cfg, scorer = r3._native_config(p0, profiles, r1)
    runtime = {
        **model_source,
        "model_receipt_sha256": sha256_file(r3.MODEL_RECEIPT / "run_manifest.json"),
        "loaded_model_state_sha256": actual_model_state,
        "profile_id": "R7/E6_stage05_set_seed_0_math_sdp_v1",
        "seed": 0,
        "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "memory_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": str(torch.version.cuda),
        "native_config": asdict(cfg),
        "native_scorer_config": asdict(scorer),
    }
    if (runtime["flash_sdp_enabled"] or runtime["memory_efficient_sdp_enabled"] or
            not runtime["math_sdp_enabled"]):
        raise E123Error("Historical explicit math-only SDP runtime is not active")
    return torch, impl, backbone, {
        "p0": p0, "r1": r1, "r3": r3, "cfg": cfg, "runtime": runtime,
    }


def _stage05_paths(condition: Condition, frozen: Mapping[str, Any]
                   ) -> tuple[list[Path], list[Path], Path]:
    from experiments.calibration.r3_fullstream_2x2 import DATASET_ROOT, JSON_ROOT
    from experiments.calibration.stage05 import _absolute_image_path

    record = frozen["classes"][condition.category]
    support = record["supports"][str(condition.support_seed)]
    if tuple(str(x) for x in support["images"]) != condition.support_images:
        raise E123Error("Support image identity differs from the Stage 0.5 condition")
    if tuple(sorted(_opaque_group(str(g)) for g in support["groups"])) != condition.support_groups:
        raise E123Error("Support Source Group identity differs from Stage 0.5")
    prefix = str(record["prefix"])
    support_paths = [Path(_absolute_image_path(DATASET_ROOT, prefix, image))
                     for image in condition.support_images]
    c64_paths = [Path(_absolute_image_path(DATASET_ROOT, prefix, image))
                 for image in condition.c64_images]
    label_path = JSON_ROOT / f"{condition.category}.json"
    for path in (*support_paths, *c64_paths,
                 *(Path(event.image_path) for event in condition.events)):
        if not path.is_file():
            raise E123Error(f"Authorized experiment image is missing: {path}")
    if sha256_file(label_path) != condition.class_json_sha256:
        raise E123Error("TEST source JSON changed after Stage 0.5 authorization")
    return support_paths, c64_paths, label_path


def _encode_c64(impl: Any, backbone: Any, paths: Sequence[Path], cfg: Any
                ) -> tuple[list[dict[str, Any]], float]:
    import experiments.p0_protocol_audit as p0

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, path in enumerate(paths):
        image, features = p0.encode_query(impl, backbone, str(path), cfg)
        rows.append({
            "image_hw": [int(image.height), int(image.width)],
            "features_cpu": [feature.detach().to("cpu").contiguous() for feature in features],
            "image_sha256": sha256_file(path), "reference_index": index,
        })
    return rows, time.perf_counter() - started


def _score_c64(torch: Any, impl: Any, bank: Any,
               c64_rows: Sequence[Mapping[str, Any]], r1: Any, cfg: Any) -> list[float]:
    from types import SimpleNamespace

    scores: list[float] = []
    for row in c64_rows:
        height, width = row["image_hw"]
        image = SimpleNamespace(height=height, width=width)
        features = [feature.to("cuda", non_blocking=False)
                    for feature in row["features_cpu"]]
        _amap, _ids, score, _summary = r1._native_dense_score(
            impl, bank, image, features, cfg)
        scores.append(float(score))
    if len(scores) != 64 or any(not math.isfinite(value) for value in scores):
        raise E123Error("Native current-state C64 scoring did not produce 64 finite values")
    return scores


def _stage05_c64_receipt(category: str) -> tuple[dict[str, Any], list[float]]:
    import numpy as np
    from experiments.calibration.r3_fullstream_2x2 import STAGE05

    manifest = read_json(STAGE05 / "score_tape_manifest.json")
    entry = manifest["classes"][category]["score_tapes"]["0"]
    path = (STAGE05 / str(entry["file"])).resolve(strict=True)
    if STAGE05.resolve(strict=True) not in path.parents or sha256_file(path) != entry.get("sha256"):
        raise E123Error("Stage 0.5 C64 score receipt is path-unsafe or hash-invalid")
    with np.load(path, allow_pickle=False) as archive:
        scores = np.asarray(archive["c64_scores"], dtype=np.float64)
        identity = str(archive["c64_identity_sha256"].item())
    if scores.shape != (64,) or not np.isfinite(scores).all():
        raise E123Error("Historical Stage 0.5 C64 score receipt is malformed")
    return ({"path": path.relative_to(ROOT).as_posix(), "sha256": entry["sha256"],
             "fixed_support_sha256": entry["fixed_support_sha256"],
             "c64_identity_sha256": identity}, scores.tolist())


def _condition_manifest(budget: OutputBudget, condition: Condition,
                        contract: Mapping[str, Any], *, source: Mapping[str, Any]) -> None:
    relative = f"stage_a/conditions/{condition.condition_id}/condition_manifest.json"
    if (budget.root / relative).exists():
        raise E123Error("This condition has already started; automatic resume is forbidden")
    budget.write_once(relative, canonical_bytes({
        "schema": "recovery.e123.condition_manifest.v1",
        "condition_id": condition.condition_id, "category": condition.category,
        "support_seed": condition.support_seed, "stream_order": condition.order,
        "event_count": len(condition.events), "query_count": len(condition.query_positions),
        "query_tape_sha256": condition.query_tape_sha256,
        "stream_identity_sha256": condition.stream_identity_sha256,
        "source_identity_sha256": contract["source_identity"]["source_fingerprint_sha256"],
        "class_json_sha256": condition.class_json_sha256,
        "role_identity": dict(source), "status": "STARTED_NO_RESUME",
    }))


def _run_stage_a_condition(run_id: str, condition_id: str) -> int:
    run_path = _safe_output_root(OUTPUT_BASE / f"recovery_e123_{run_id}", must_exist=True)
    budget = OutputBudget(run_path)
    contract = read_json(run_path / "E123_contract.json")
    _verify_contract(contract, expected_run_id=run_id)
    if contract.get("stage_a_started") is not True:
        raise E123Error("E123 Stage A is not authorized by its frozen run contract")
    launch_token = os.environ.get("E123_SUPERVISOR_LAUNCH_TOKEN", "")
    if (not launch_token or sha256_bytes(launch_token.encode("ascii")) !=
            contract.get("supervisor_launch_token_sha256")):
        raise E123Error("Stage A child is missing its single-use supervised-launch capability")
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job
    if not current_process_is_in_job():
        raise E123Error("Stage A child must run inside the Windows Job Object supervisor")
    conditions, _identity = _build_conditions()
    condition = _condition_by_id(conditions, condition_id)
    condition_dir = run_path / "stage_a" / "conditions" / condition.condition_id
    if condition_dir.exists() and any(condition_dir.iterdir()):
        raise E123Error("Condition namespace already contains files; resume is forbidden")
    registered = {item["condition_id"]: item
                  for item in contract["design"]["query_protocol"]["tape_bindings"]}.get(condition_id)
    if (registered is None or registered.get("query_tape_sha256") != condition.query_tape_sha256 or
            int(registered.get("query_count", -1)) != len(condition.query_positions)):
        raise E123Error("Reconstructed condition differs from its frozen query tape")
    disk = _preflight_disk(run_path)
    frozen, checked = _load_stage05_inputs()
    support_paths, c64_paths, label_path = _stage05_paths(condition, frozen)
    support_record = frozen["classes"][condition.category]["supports"][str(condition.support_seed)]
    role_receipt = {
        "support_group_sha256": sha256_json(list(condition.support_groups)),
        "support_image_identity_sha256": sha256_json(list(condition.support_images)),
        "c64_group_sha256": sha256_json(list(condition.c64_groups)),
        "c64_image_identity_sha256": sha256_json(list(condition.c64_images)),
        "test_group_set_sha256": sha256_json(sorted({e.source_group_sha256 for e in condition.events})),
        "support_train_normal_verified": bool(support_record.get("all_train_normal")),
        "official_group_disjointness": True,
    }
    validate_role_disjointness(condition.support_groups, condition.c64_groups,
                               [e.source_group_sha256 for e in condition.events])
    _condition_manifest(budget, condition, contract, source=role_receipt)
    score_rel = f"stage_a/conditions/{condition_id}/scores.jsonl"
    snapshot_rel = f"stage_a/conditions/{condition_id}/c64_snapshots.jsonl"
    feedback_rel = f"stage_a/conditions/{condition_id}/query_feedback.private.jsonl"
    progress_rel = f"stage_a/conditions/{condition_id}/progress.jsonl"
    scores_out = BufferedJsonl(budget, score_rel, flush_every=64)
    snapshots_out = BufferedJsonl(budget, snapshot_rel, flush_every=1)
    feedback_out = BufferedJsonl(budget, feedback_rel, flush_every=1)
    progress_out = BufferedJsonl(budget, progress_rel, flush_every=1)
    log_sink = BudgetedTextLog(budget, f"logs/{condition_id}.log")
    from contextlib import redirect_stderr, redirect_stdout
    started = time.perf_counter()
    try:
        with redirect_stdout(log_sink), redirect_stderr(log_sink):
            torch, impl, backbone, native = _native_profile()
            p0, r1, r3, cfg = native["p0"], native["r1"], native["r3"], native["cfg"]
            torch.cuda.synchronize()
            model_baseline_cuda_bytes = int(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            support_cache = p0.encode_reference_features(
                impl, backbone, [str(path) for path in support_paths], cfg)
            canonical_bank = p0.build_vmb(
                impl, support_cache, [str(path) for path in support_paths], cfg)
            structured_digest, fixed_profile = _fixed_digest_and_profile(canonical_bank, p0)
            raw_digest = p0.fixed_bank_profile(canonical_bank)["sha256"]
            expected_structured = None
            if condition.support_seed == 0:
                expected_structured = checked["classes"][condition.category][
                    "fixed_support_sha256_expected"]
                if structured_digest != expected_structured:
                    raise E123Error(
                        "Historical structured fixed-bank digest mismatch: "
                        f"expected={expected_structured}; actual={structured_digest}; "
                        f"raw={raw_digest}; layers={fixed_profile['layers']}")
            c64_rows, c64_encode_seconds = _encode_c64(impl, backbone, c64_paths, cfg)
            c64_initial = _score_c64(torch, impl, canonical_bank, c64_rows, r1, cfg)
            import numpy as np
            theta0 = float(np.sort(np.asarray(c64_initial, dtype=np.float64), kind="mergesort")[
                math.ceil(.90 * (len(c64_initial) - 1))])
            sigma0 = max(float(np.std(c64_initial, ddof=1)), 0.02)
            historical_c64_receipt = None
            historical_c64_error = None
            if condition.support_seed == 0:
                historical_c64_receipt, historical_c64 = _stage05_c64_receipt(
                    condition.category)
                historical_c64_error = float(np.max(np.abs(
                    np.asarray(c64_initial, dtype=np.float64) -
                    np.asarray(historical_c64, dtype=np.float64))))
                if any(not r3._score_tolerance(actual, expected)
                       for actual, expected in zip(c64_initial, historical_c64)):
                    raise E123Error(
                        "Native seed-0 C64 scores differ from Stage 0.5 receipt: "
                        f"max_abs={historical_c64_error}")
            support_self_scores = []
            for index, path in enumerate(support_paths):
                image, features = p0.encode_query(impl, backbone, str(path), cfg)
                _amap, _ids, score, _summary = r1._native_dense_score(
                    impl, canonical_bank, image, features, cfg)
                support_self_scores.append({
                    "support_index": index,
                    "image_identity_sha256": sha256_json(condition.support_images[index]),
                    "image_content_sha256": sha256_file(path),
                    "score": float(score), "self_matching": True,
                })
            arms: dict[str, Any] = {"STATIC": canonical_bank}
            states: dict[str, E123MemoryState] = {}
            for method in MUTABLE_METHODS:
                for capacity in CAPACITIES:
                    arm_id = f"{method}_K{capacity}"
                    clone = _clone_bank_with_capacity(canonical_bank, capacity)
                    state = E123MemoryState(
                        method=method, capacity=capacity, bank=clone,
                        seed=_stable_seed(run_id, condition_id, arm_id))
                    states[arm_id] = state
                    arms[arm_id] = clone
            dynamic_tensor_ids = [id(codebook.protos) for state in states.values()
                                  for codebook in _bank_dynamic_codebooks(state.bank)]
            if len(set(dynamic_tensor_ids)) != len(dynamic_tensor_ids):
                raise E123Error("Mutable arm tensors alias across E2 configurations")
            from experiments.calibration.controllers import StateConsistentQuantile
            from experiments.calibration.protocol import EvidenceAnchor, EvidenceSnapshot
            b3 = StateConsistentQuantile(theta0, delta=0.01, min_normals=16, name="B3")
            initial_evidence = EvidenceSnapshot(
                memory_version=0,
                normal_anchors=tuple(EvidenceAnchor(
                    opaque_anchor_id=f"e123-c64-{index}", score=float(score),
                    memory_version=0, protected=True)
                    for index, score in enumerate(c64_initial)),
                defect_anchors=(),
                source_exclusion_audit={"protected_c64": True,
                                        "source_group_disjointness_verified": True},
                evidence_revision=0)
            b3.on_memory_change(initial_evidence)
            b3.finalize_step()
            snapshots_out.add({
                "schema": "recovery.e123.c64_snapshot.v1", "memory_version": 0,
                "memory_state_sha256": states["GREEDY_K1024"].state_digest,
                "c64_scores": c64_initial,
                "theta_before": theta0, "theta_after": float(b3.theta),
                "c64_image_content_sha256": [row["image_sha256"] for row in c64_rows],
                "snapshot_source": "C64 scored against fixed support bank at M0"})
            snapshots_out.flush(sync=True)
            query_original_positions = [condition.events[position].original_position
                                        for position in condition.query_positions]
            query_group_by_original = {
                event.original_position: event.source_group_sha256
                for event in condition.events if event.position in condition.query_positions}
            label_reader = RandomAccessQueriedLabelReader(
                category=condition.category, path=label_path,
                expected_source_sha256=condition.class_json_sha256,
                allowed_positions=query_original_positions,
                group_sha256_by_position=query_group_by_original)
            score_chain = hashlib.sha256()
            candidate_id_hashes: set[str] = set()
            candidate_feature_hashes: set[str] = set()
            query_normal_count = query_defect_count = 0
            eligible_normal_writes = normal_write_rejects = 0
            encode_seconds = c64_encode_seconds
            score_seconds = {name: 0.0 for name in CONFIGURATION_IDS}
            update_seconds = {name: 0.0 for name in CONFIGURATION_IDS if name != "STATIC"}
            c64_rescore_seconds = 0.0
            fixed_offset = float(contract["design"]["e3"]["fixed_offset_selection"]
                                 ["selection_by_target_category"][condition.category]
                                 ["coefficient"])
            initial_reference_parity: dict[str, Any] | None = None
            try:
                for position, event in enumerate(condition.events):
                    event_start = time.perf_counter()
                    image_path = Path(event.image_path)
                    image_hash = sha256_file(image_path)
                    torch.cuda.synchronize()
                    timer = time.perf_counter()
                    image, features = p0.encode_query(impl, backbone, str(image_path), cfg)
                    torch.cuda.synchronize()
                    encode_seconds += time.perf_counter() - timer
                    if len(features) != 1 or tuple(features[0].shape) != (784, 4096):
                        raise E123Error(f"{condition_id}:{position}: fused native feature shape changed")
                    timer = time.perf_counter()
                    static_map, static_ids, static_score, _static_summary = r1._native_dense_score(
                        impl, canonical_bank, image, features, cfg)
                    torch.cuda.synchronize()
                    score_seconds["STATIC"] += time.perf_counter() - timer
                    if position == 0:
                        ref_score, ref_ids, ref_map = r1._native_reference_and_debug(
                            impl, canonical_bank, image, features, cfg)
                        if (not np.array_equal(static_ids, ref_ids) or
                                not np.array_equal(static_map, ref_map) or
                                not r3._score_tolerance(static_score, ref_score)):
                            raise E123Error("Initial Static native reference parity failed")
                        initial_reference_parity = {
                            "status": "PASS",
                            "score_max_abs_error": abs(float(static_score) - float(ref_score)),
                            "map_max_abs_error": float(np.max(np.abs(static_map - ref_map))),
                            "top32_ids_equal": True,
                        }
                    patch_map = impl.layer_to_topk_from_amap(static_map)
                    candidate_ids = {
                        int(layer): [int(value) for value in values[:8]]
                        for layer, values in patch_map.items() if values[:8]
                    }
                    if not candidate_ids:
                        raise E123Error("Static map produced no shared candidate patches")
                    candidate_digest = sha256_json({str(k): v for k, v in candidate_ids.items()})
                    candidate_id_hashes.add(candidate_digest)
                    scores: dict[str, float] = {"STATIC": float(static_score)}
                    arm_state_digests: dict[str, str] = {"STATIC": "M0"}
                    for arm_id in CONFIGURATION_IDS[1:]:
                        timer = time.perf_counter()
                        _amap, _ids, score, _summary = r1._native_dense_score(
                            impl, arms[arm_id], image, features, cfg)
                        scores[arm_id] = float(score)
                        score_seconds[arm_id] += time.perf_counter() - timer
                        arm_state_digests[arm_id] = states[arm_id].state_digest
                    b3_theta_before = float(b3.theta)
                    greedy_state = states["GREEDY_K1024"]
                    event_id = sha256_json({
                        "condition_id": condition_id, "position": position,
                        "image_identity_sha256": event.image_identity_sha256,
                        "source_group_sha256": event.source_group_sha256,
                    })
                    queried = position in condition.query_positions
                    row = {
                        "schema": SCORE_SCHEMA, "condition_id": condition_id,
                        "position": position, "original_position": event.original_position,
                        "event_id_sha256": event_id,
                        "image_identity_sha256": event.image_identity_sha256,
                        "source_group_sha256": event.source_group_sha256,
                        "image_content_sha256": image_hash,
                        "camera_id": event.camera_id,
                        "timestamp_semantics": "metadata only; not asserted physical chronology",
                        "query_tape_decision": queried,
                        "query_tape_sha256": condition.query_tape_sha256,
                        "scores": scores, "theta0": theta0, "sigma0": sigma0,
                        "fixed_offset_coefficient": fixed_offset,
                        "theta_fixed_offset": fixed_offset_threshold(
                            theta0, sigma0, fixed_offset),
                        "b3_greedy1024_threshold_before": b3_theta_before,
                        "greedy1024_memory_version_before": greedy_state.memory_version,
                        "greedy1024_transition_sha256_before": greedy_state.state_digest,
                        "candidate_patch_ids_sha256": candidate_digest,
                        "candidate_patch_ids": {str(k): v for k, v in candidate_ids.items()},
                        "memory_state_transition_sha256": arm_state_digests,
                    }
                    encoded = scores_out.add(row)
                    score_chain.update(encoded)
                    if queried:
                        # Every score in this row is flushed and fsynced before
                        # the current queried label is decoded.
                        label = disclose_after_score_seal(
                            scores_out, queried=True, label_reader=label_reader,
                            category=condition.category,
                            original_position=event.original_position,
                            source_key=event.sample_key)
                        feedback_out.add({
                            "schema": "recovery.e123.private_query_feedback.v1",
                            "condition_id": condition_id, "position": position,
                            "original_position": event.original_position,
                            "event_id_sha256": event_id,
                            "source_group_sha256": event.source_group_sha256,
                            "label": label,
                        })
                        feedback_out.flush(sync=True)
                        if label == 0:
                            query_normal_count += 1
                        else:
                            query_defect_count += 1
                        allow, keep_count, reason, eligibility_info = resolve_common_eligibility(
                            queried=True, label=label, c0_score=static_score,
                            theta0=theta0, sigma0=sigma0,
                            eligibility=impl.select_informative_normal_write)
                        progress_out.add({
                            "event": "feedback_resolved", "position": position,
                            "query_label_class": "normal" if label == 0 else "defect",
                            "common_c0_eligibility": allow,
                            "common_c0_eligibility_reason": reason,
                            "common_keep_count": keep_count,
                            "candidate_patch_ids_sha256": candidate_digest,
                        })
                        progress_out.flush(sync=True)
                        if label == 0 and not allow:
                            normal_write_rejects += 1
                        if allow:
                            eligible_normal_writes += 1
                            common_candidates = prepare_common_candidates(
                                impl, features, candidate_ids, device="cuda")
                            feature_digest = _candidate_feature_digest(common_candidates)
                            candidate_feature_hashes.add(feature_digest)
                            greedy_before = greedy_state.memory_version
                            update_results = dispatch_common_write(
                                states, eligible=True,
                                observe_kwargs={
                                    "impl": impl, "common_candidates": common_candidates,
                                    "event_position": position, "static_score": static_score,
                                    "theta0": theta0, "sigma0": sigma0,
                                    "eligibility_info": eligibility_info or {},
                                }, synchronize=torch.cuda.synchronize)
                            for arm_id, (_changed, elapsed) in update_results.items():
                                update_seconds[arm_id] += elapsed
                            if greedy_state.memory_version != greedy_before:
                                torch.cuda.synchronize()
                                timer = time.perf_counter()
                                current_c64 = _score_c64(
                                    torch, impl, greedy_state.bank, c64_rows, r1, cfg)
                                torch.cuda.synchronize()
                                c64_rescore_seconds += time.perf_counter() - timer
                                version = greedy_state.memory_version
                                evidence = EvidenceSnapshot(
                                    memory_version=version,
                                    normal_anchors=tuple(EvidenceAnchor(
                                        opaque_anchor_id=f"e123-c64-{index}",
                                        score=float(score), memory_version=version,
                                        protected=True)
                                        for index, score in enumerate(current_c64)),
                                    defect_anchors=(), source_exclusion_audit={
                                        "c64_recomputed_after_memory_change": True,
                                        "roles_disjoint": True}, evidence_revision=version)
                                b3.on_memory_change(evidence)
                                b3.finalize_step()
                                snapshots_out.add({
                                    "schema": "recovery.e123.c64_snapshot.v1",
                                    "memory_version": version,
                                    "memory_state_sha256": greedy_state.state_digest,
                                    "c64_scores": current_c64,
                                    "theta_before": b3_theta_before,
                                    "theta_after": float(b3.theta),
                                    "snapshot_source": "C64 rescored after actual Greedy-K1024 state change"})
                                snapshots_out.flush(sync=True)
                    if ((position + 1) % 64 == 0 or position + 1 == len(condition.events)):
                        progress_out.add({
                            "event": "progress", "position": position,
                            "completed_events": position + 1,
                            "event_count": len(condition.events),
                            "elapsed_seconds": time.perf_counter() - event_start,
                            "query_count_so_far": query_normal_count + query_defect_count,
                        })
                        progress_out.flush(sync=False)
                    del image, features, static_map
                scores_out.flush(sync=True)
                snapshots_out.flush(sync=True)
                feedback_out.flush(sync=True)
                progress_out.flush(sync=True)
            finally:
                label_reader.close()
            if len(label_reader.decoded_positions) != len(condition.query_positions):
                raise E123Error("Stage A decoded-label count differs from the frozen tape")
            if query_normal_count + query_defect_count != len(condition.query_positions):
                raise E123Error("Stage A feedback count differs from Q-A-TS")
            arm_audits = {arm_id: state.audit() for arm_id, state in states.items()}
            for arm_id, audit in arm_audits.items():
                expected_capacity = int(arm_id.rsplit("K", 1)[1])
                if any(int(layer["capacity"]) != expected_capacity
                       for layer in audit["layers"]):
                    raise E123Error(f"{arm_id}: dynamic layer capacity differs")
            resource_summary: dict[str, Any] = {
                "shared_fixed_prototype_bytes": int(fixed_profile["total_feature_bytes"]),
                "shared_fixed_bank_profile": fixed_profile,
                "mutable_arms": arm_audits,
                "registered_persistent_byte_tolerance": BYTE_TOLERANCE,
                "persistent_byte_comparison_by_capacity": {},
                "test_event_image_encodings": len(condition.events),
                "c64_reference_image_encodings": 64,
                "support_image_self_score_encodings": len(support_paths),
                "total_image_encodings": len(condition.events) + 64 + len(support_paths),
                "encode_seconds": encode_seconds,
                "score_seconds_by_configuration": score_seconds,
                "update_seconds_by_configuration": update_seconds,
                "c64_rescore_seconds": c64_rescore_seconds,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "eligible_queried_normal_events": eligible_normal_writes,
                "queried_normal_events_rejected_by_common_c0_gate": normal_write_rejects,
                "candidate_id_digest_cardinality": len(candidate_id_hashes),
                "candidate_feature_digest_cardinality": len(candidate_feature_hashes),
                "shared_feature_candidate_fingerprint_algorithm": "sha256(layer,patch IDs,normalized tensor shape/dtype/bytes)",
                "model_baseline_cuda_allocated_bytes": model_baseline_cuda_bytes,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "peak_cuda_allocated_above_model_baseline_bytes": max(
                    0, int(torch.cuda.max_memory_allocated()) - model_baseline_cuda_bytes),
                "peak_memory_semantics": (
                    "device high-water mark includes backbone, banks, persistent mutable tensors, and temporary scoring/update allocations; "
                    "the above-baseline value is not a pure temporary-only estimate"),
            }
            for capacity in CAPACITIES:
                comparable = [state.audit() for arm_id, state in states.items()
                              if arm_id.endswith(f"K{capacity}")]
                byte_values = [int(item["persistent_mutable_bytes_estimate"])
                               for item in comparable]
                spread = ((max(byte_values) - min(byte_values)) / max(byte_values)
                          if byte_values and max(byte_values) else 0.0)
                prototype_values = [int(item["prototype_tensor_bytes"])
                                    for item in comparable]
                resource_summary["persistent_byte_comparison_by_capacity"][str(capacity)] = {
                    "methods": len(comparable), "max_relative_spread": spread,
                    "within_tolerance": spread <= BYTE_TOLERANCE,
                    "prototype_bytes_equal": len(set(prototype_values)) == 1,
                }
            if initial_reference_parity is None:
                raise E123Error("Static native map parity was not performed")
            score_file = run_path / score_rel
            if not score_file.is_file():
                raise E123Error("Stage A score shard is missing")
            seal = {
                "schema": CONDITION_SEAL_SCHEMA, "status": "SEALED_COMPLETE",
                "run_id": run_id, "condition_id": condition_id,
                "category": condition.category, "support_seed": condition.support_seed,
                "stream_order": condition.order, "event_count": len(condition.events),
                "query_count": len(condition.query_positions),
                "query_normal_count": query_normal_count,
                "query_defect_count": query_defect_count,
                "score_configurations": list(CONFIGURATION_IDS),
                "score_rows_sha256": sha256_file(score_file),
                "score_rows_chain_sha256": score_chain.hexdigest(),
                "score_row_count": len(condition.events),
                "c64_snapshots_sha256": sha256_file(run_path / snapshot_rel),
                "private_query_feedback_sha256": sha256_file(run_path / feedback_rel),
                "feedback_count": len(label_reader.decoded_positions),
                "query_tape_sha256": condition.query_tape_sha256,
                "fixed_bank": {
                    "structured_sha256": structured_digest,
                    "structured_algorithm": "experiments.calibration.native._fixed_bank_digest",
                    "raw_prototype_sha256": raw_digest,
                    "raw_algorithm": "p0.fixed_bank_profile.raw_prototype_bytes",
                    "expected_historical_structured_sha256": expected_structured,
                    "historical_match": (condition.support_seed == 0),
                    "layers": fixed_profile["layers"],
                },
                "c64": {
                    "theta0_higher_q90": theta0, "sigma0_floor_0_02": sigma0,
                    "scores": c64_initial,
                    "image_content_sha256": [row["image_sha256"] for row in c64_rows],
                    "identity_sha256": sha256_json(list(condition.c64_images)),
                    "historical_score_receipt": historical_c64_receipt,
                    "max_abs_error_to_historical_receipt": historical_c64_error,
                },
                "support_self_scores": support_self_scores,
                "static_native_reference_parity": initial_reference_parity,
                "resource_summary": resource_summary,
                "candidate_selection": "common static-support map; first 8 patch IDs per fused layer",
                "all_mutable_states_independent": True,
                "all_arms_receive_identical_candidate_ids_and_feature_tensor": True,
                "normal_write_eligibility": "queried verified normal AND common C0 informative-normal gate at fixed theta0/sigma0",
                "unqueried_labels_decoded": False, "masks_accessed": False,
                "model_runtime": native["runtime"], "disk_preflight": disk,
                "elapsed_seconds": time.perf_counter() - started,
            }
            seal["seal_sha256"] = sha256_json(seal)
            budget.write_once(
                f"stage_a/conditions/{condition_id}/condition_seal.json",
                canonical_bytes(seal))
            budget.verify_complete_namespace()
        log_sink.flush()
        return 0
    except Exception as exc:
        try:
            scores_out.flush(sync=True)
            snapshots_out.flush(sync=True)
            feedback_out.flush(sync=True)
            progress_out.flush(sync=True)
        except Exception:
            pass
        log_sink.flush()
        budget.failure(f"stage_a/conditions/{condition_id}/failure.json", {
            "schema": "recovery.e123.condition_failure.v1",
            "run_id": run_id, "condition_id": condition_id,
            "error_type": type(exc).__name__, "error": str(exc),
            "source_commit_sha": contract["source_identity"]["commit_sha"],
            "stage_a_status": "INCOMPLETE_PRESERVE_PARTIAL_DO_NOT_RESUME",
        })
        return 1


def _verify_condition_seal(run_path: Path, condition: Condition,
                           run_id: str) -> dict[str, Any]:
    seal_path = (run_path / "stage_a" / "conditions" /
                 condition.condition_id / "condition_seal.json")
    if not seal_path.is_file():
        raise E123Error(f"Missing complete Stage A seal: {condition.condition_id}")
    seal = read_json(seal_path)
    digest = str(seal.pop("seal_sha256", ""))
    if (seal.get("schema") != CONDITION_SEAL_SCHEMA or
            seal.get("status") != "SEALED_COMPLETE" or
            seal.get("run_id") != run_id or
            seal.get("condition_id") != condition.condition_id or
            seal.get("event_count") != len(condition.events) or
            seal.get("score_row_count") != len(condition.events) or
            seal.get("query_count") != len(condition.query_positions) or
            seal.get("query_tape_sha256") != condition.query_tape_sha256 or
            seal.get("score_configurations") != list(CONFIGURATION_IDS) or
            sha256_json(seal) != digest):
        raise E123Error(f"Stage A condition seal identity/checksum failed: {condition.condition_id}")
    folder = seal_path.parent
    if (sha256_file(folder / "scores.jsonl") != seal.get("score_rows_sha256") or
            sha256_file(folder / "c64_snapshots.jsonl") != seal.get("c64_snapshots_sha256") or
            sha256_file(folder / "query_feedback.private.jsonl") !=
            seal.get("private_query_feedback_sha256")):
        raise E123Error(f"Stage A sealed file changed: {condition.condition_id}")
    return {**seal, "seal_sha256": digest}


def _seal_all_stage_a(run_path: Path, run_id: str,
                      conditions: Sequence[Condition]) -> dict[str, Any]:
    if len(conditions) != 16:
        raise E123Error("Stage B requires exactly the registered sixteen conditions")
    seals = [_verify_condition_seal(run_path, condition, run_id)
             for condition in conditions]
    if len({seal["condition_id"] for seal in seals}) != 16:
        raise E123Error("Stage A condition seals are not unique")
    event_count = sum(int(seal["event_count"]) for seal in seals)
    query_count = sum(int(seal["query_count"]) for seal in seals)
    if event_count != sum(EXPECTED_COUNTS.values()) * 4:
        raise E123Error("Global Stage A event count differs from the registered 16 conditions")
    return {
        "schema": GLOBAL_SEAL_SCHEMA,
        "status": "SEALED_COMPLETE",
        "run_id": run_id,
        "condition_count": len(seals),
        "event_count": event_count,
        "query_count": query_count,
        "condition_seals": [
            {"condition_id": row["condition_id"], "seal_sha256": row["seal_sha256"],
             "score_rows_sha256": row["score_rows_sha256"]}
            for row in seals],
    }


def require_complete_stage_a(global_seal: Mapping[str, Any], *,
                             expected_conditions: int = 16) -> None:
    seal = dict(global_seal)
    digest = str(seal.pop("seal_sha256", ""))
    condition_seals = seal.get("condition_seals", ())
    expected_ids = {
        f"{category}_s{seed}_{order}"
        for category in CATEGORIES for seed in SUPPORT_SEEDS for order in STREAM_ORDERS
    }
    actual_ids = [str(row.get("condition_id", ""))
                  for row in condition_seals if isinstance(row, Mapping)]
    expected_events = sum(EXPECTED_COUNTS.values()) * len(SUPPORT_SEEDS) * len(STREAM_ORDERS)
    expected_queries = sum(
        math.floor(0.03 * count + 0.5) * len(SUPPORT_SEEDS) * len(STREAM_ORDERS)
        for count in EXPECTED_COUNTS.values())
    if (seal.get("schema") != GLOBAL_SEAL_SCHEMA or
            seal.get("status") != "SEALED_COMPLETE" or
            int(seal.get("condition_count", -1)) != expected_conditions or
            expected_conditions != len(expected_ids) or
            len(condition_seals) != expected_conditions or
            len(actual_ids) != expected_conditions or set(actual_ids) != expected_ids or
            len(set(actual_ids)) != expected_conditions or
            any(not re.fullmatch(r"[0-9a-f]{64}", str(row.get("seal_sha256", ""))) or
                not re.fullmatch(r"[0-9a-f]{64}", str(row.get("score_rows_sha256", "")))
                for row in condition_seals) or
            int(seal.get("event_count", -1)) != expected_events or
            int(seal.get("query_count", -1)) != expected_queries or
            not digest or sha256_json(seal) != digest):
        raise E123Error("Incomplete Stage A artifacts cannot be promoted to Stage B")


def _metrics_from_predictions(labels: Sequence[int], scores: Sequence[float],
                              predictions: Sequence[int]) -> dict[str, Any]:
    if len(labels) != len(scores) or len(labels) != len(predictions) or not labels:
        raise E123Error("Labels, score trajectory and predictions must be aligned")
    if any(int(value) not in (0, 1) for value in predictions):
        raise E123Error("Predictions must be binary")
    tp = sum(int(y == 1 and p == 1) for y, p in zip(labels, predictions))
    fp = sum(int(y == 0 and p == 1) for y, p in zip(labels, predictions))
    fn = sum(int(y == 1 and p == 0) for y, p in zip(labels, predictions))
    tn = sum(int(y == 0 and p == 0) for y, p in zip(labels, predictions))
    fpr = fp / (fp + tn) if fp + tn else None
    fnr = fn / (fn + tp) if fn + tp else None
    return {"n": len(labels), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "normal_count": sum(int(y == 0) for y in labels),
            "defect_count": sum(int(y == 1) for y in labels),
            "fpr": fpr, "fnr": fnr,
            "ber": (fpr + fnr) / 2 if fpr is not None and fnr is not None else None,
            "auroc": _auroc(labels, scores),
            "average_precision": _average_precision(labels, scores)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise E123Error(f"Malformed JSONL at {path.name}:{line_number}") from exc
            if not isinstance(value, dict):
                raise E123Error(f"JSONL row is not an object at {path.name}:{line_number}")
            rows.append(value)
    return rows


def _read_condition_stage_a(run_path: Path, condition: Condition,
                            seal: Mapping[str, Any]
                            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    folder = run_path / "stage_a" / "conditions" / condition.condition_id
    scores = _read_jsonl(folder / "scores.jsonl")
    snapshots = _read_jsonl(folder / "c64_snapshots.jsonl")
    feedback = _read_jsonl(folder / "query_feedback.private.jsonl")
    if len(scores) != len(condition.events) or len(snapshots) < 1:
        raise E123Error(f"{condition.condition_id}: incomplete Stage A rows/snapshots")
    if len(feedback) != len(condition.query_positions):
        raise E123Error(f"{condition.condition_id}: private feedback count mismatch")
    for position, row in enumerate(scores):
        if (row.get("schema") != SCORE_SCHEMA or
                row.get("condition_id") != condition.condition_id or
                int(row.get("position", -1)) != position or
                int(row.get("original_position", -1)) != condition.events[position].original_position or
                row.get("query_tape_sha256") != condition.query_tape_sha256 or
                bool(row.get("query_tape_decision")) != (position in condition.query_positions) or
                set(row.get("scores", {})) != set(CONFIGURATION_IDS)):
            raise E123Error(f"{condition.condition_id}:{position}: sealed score identity mismatch")
    feedback_by_position = {}
    for row in feedback:
        position = int(row.get("position", -1))
        if (row.get("schema") != "recovery.e123.private_query_feedback.v1" or
                row.get("condition_id") != condition.condition_id or
                position not in condition.query_positions or position in feedback_by_position or
                row.get("source_group_sha256") != condition.events[position].source_group_sha256 or
                row.get("label") not in (0, 1)):
            raise E123Error(f"{condition.condition_id}: invalid private query feedback row")
        feedback_by_position[position] = row
    if set(feedback_by_position) != set(condition.query_positions):
        raise E123Error(f"{condition.condition_id}: private feedback positions differ from Q-A-TS")
    previous_version = -1
    snapshot_by_version = {}
    for row in snapshots:
        version = int(row.get("memory_version", -1))
        if (row.get("schema") != "recovery.e123.c64_snapshot.v1" or
                version <= previous_version or len(row.get("c64_scores", ())) != 64 or
                row.get("snapshot_source") not in {
                    "C64 scored against fixed support bank at M0",
                    "C64 rescored after actual Greedy-K1024 state change"}):
            raise E123Error(f"{condition.condition_id}: malformed or unordered C64 snapshot")
        previous_version = version
        snapshot_by_version[version] = row
    if 0 not in snapshot_by_version:
        raise E123Error(f"{condition.condition_id}: initial C64 M0 snapshot missing")
    for row in scores:
        version = int(row["greedy1024_memory_version_before"])
        snapshot = snapshot_by_version.get(version)
        if snapshot is None or snapshot.get("memory_state_sha256") != row.get(
                "greedy1024_transition_sha256_before"):
            raise E123Error(f"{condition.condition_id}: Greedy/C64 state snapshot mismatch")
        if not math.isclose(float(snapshot["theta_after"]),
                            float(row["b3_greedy1024_threshold_before"]),
                            rel_tol=0.0, abs_tol=1e-12):
            raise E123Error(f"{condition.condition_id}: B3 pre-event threshold mismatch")
    return scores, snapshots, list(feedback_by_position.values())


def _worst_window_ber(labels: Sequence[int], predictions: Sequence[int],
                      *, window: int = 256) -> dict[str, Any]:
    values = []
    for start in range(max(0, len(labels) - window + 1)):
        stop = start + window
        metrics = _binary_metrics(labels[start:stop], [float(v) for v in predictions[start:stop]],
                                  0.5)
        if metrics["ber"] is not None:
            values.append((float(metrics["ber"]), start, stop))
    return {
        "window": window,
        "defined_window_count": len(values),
        "worst_ber": max((item[0] for item in values), default=None),
        "worst_window_start": max(values, default=(None, None, None))[1],
        "worst_window_stop_exclusive": max(values, default=(None, None, None))[2],
    }


def _replay_threshold_controllers(condition: Condition, score_rows: Sequence[Mapping[str, Any]],
                                  c64_initial: Sequence[float], theta0: float,
                                  sigma0: float, fixed_offset: float,
                                  feedback: Sequence[Mapping[str, Any]],
                                  labels: Sequence[int]
                                  ) -> tuple[dict[str, dict[str, Any]], dict[str, list[int]]]:
    from experiments.calibration.controllers import (
        OriginalGuardedSidecar, RollingQ90QueriedNormal)
    from experiments.calibration.protocol import Feedback, Observation

    query_by_position = {int(row["position"]): int(row["label"]) for row in feedback}
    if set(query_by_position) != set(condition.query_positions):
        raise E123Error("Threshold replay feedback allowlist differs from the frozen tape")
    greedy_scores = [float(row["scores"]["GREEDY_K1024"]) for row in score_rows]
    b1 = RollingQ90QueriedNormal(float(theta0), window=256, min_count=16,
                                 name="B1_ROLLING_QUERIED_NORMAL")
    b2 = OriginalGuardedSidecar(list(c64_initial), float(theta0))
    controller_instances = {"B1_ROLLING_QUERIED_NORMAL": b1,
                            "B2_ORIGINAL_SIDECAR_ADAPTED": b2}
    threshold_before: dict[str, list[float]] = {
        "FROZEN_C64": [float(theta0)] * len(score_rows),
        "FIXED_OFFSET": [float(theta0 + fixed_offset * sigma0)] * len(score_rows),
        "B1_ROLLING_QUERIED_NORMAL": [],
        "B2_ORIGINAL_SIDECAR_ADAPTED": [],
        "B3_CURRENT_STATE_C64": [float(row["b3_greedy1024_threshold_before"])
                                  for row in score_rows],
    }
    predictions: dict[str, list[int]] = {
        name: [] for name in threshold_before
    }
    startup_available = {"B1_ROLLING_QUERIED_NORMAL": 0}
    for position, (row, score, label) in enumerate(zip(score_rows, greedy_scores, labels)):
        event_id = str(row["event_id_sha256"])
        version = int(row["greedy1024_memory_version_before"])
        state_digest = str(row["greedy1024_transition_sha256_before"])
        observation = Observation(
            t=position, observation_id=event_id,
            camera_id=int(row["camera_id"]), score=score,
            memory_version=version, score_state_digest=state_digest)
        for name, controller in controller_instances.items():
            pred = controller.predict(observation)
            if pred.label not in (0, 1) or pred.threshold is None:
                raise E123Error(f"{name} produced unavailable E3 prediction at {position}")
            threshold_before[name].append(float(pred.threshold))
            predictions[name].append(int(pred.label))
            if name == "B1_ROLLING_QUERIED_NORMAL" and len(controller.history) >= controller.min_count:
                startup_available[name] += 1
            controller.observe_score(observation)
            queried = position in query_by_position
            feedback_row = Feedback(
                t=position, queried=queried,
                label=query_by_position.get(position) if queried else None)
            if queried and query_by_position[position] != int(label):
                raise E123Error("Stage A query label does not match Stage B sealed outcome")
            controller.observe_feedback(observation, pred, feedback_row)
            controller.finalize_step()
        for name in ("FROZEN_C64", "FIXED_OFFSET", "B3_CURRENT_STATE_C64"):
            predictions[name].append(int(score >= threshold_before[name][position]))
    score_auc = _auroc(labels, greedy_scores)
    result: dict[str, dict[str, Any]] = {}
    for name, thresholds in threshold_before.items():
        metric = _binary_metrics(labels, greedy_scores, 0.0)
        pred = predictions[name]
        tp = sum(int(y == 1 and p == 1) for y, p in zip(labels, pred))
        fp = sum(int(y == 0 and p == 1) for y, p in zip(labels, pred))
        fn = sum(int(y == 1 and p == 0) for y, p in zip(labels, pred))
        tn = sum(int(y == 0 and p == 0) for y, p in zip(labels, pred))
        fpr = fp / (fp + tn) if fp + tn else None
        fnr = fn / (fn + tp) if fn + tp else None
        ber = (fpr + fnr) / 2 if fpr is not None and fnr is not None else None
        movements = [abs(right - left) for left, right in zip(thresholds, thresholds[1:])]
        result[name] = {
            "n": len(labels), "normal_count": sum(y == 0 for y in labels),
            "defect_count": sum(y == 1 for y in labels),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "fpr": fpr, "fnr": fnr, "ber": ber,
            "auroc": score_auc,
            "threshold_first": thresholds[0], "threshold_last": thresholds[-1],
            "threshold_total_variation": sum(movements),
            "threshold_update_count": sum(value > 1e-12 for value in movements),
            "threshold_startup_coverage": (
                startup_available.get(name, len(labels)) / len(labels)),
            "worst_256_ber": _worst_window_ber(labels, pred),
            "query_count": len(condition.query_positions),
            "query_rate": len(condition.query_positions) / len(condition.events),
        }
    aucs = {value["auroc"] for value in result.values()}
    if len(aucs) != 1:
        raise E123Error("E3 AUROC changed across controllers replaying the same score vector")
    return result, predictions


def _stage1a_k256_evidence() -> dict[str, Any]:
    complete = read_json(STAGE1A_ROOT / "prediction_phase_complete.json")
    source = STAGE1A_ROOT / "per_method_metrics.json"
    checksum = read_json(STAGE1A_ROOT / "checksums.json").get("files", {})
    relative = "per_method_metrics.json"
    digest = sha256_file(source)
    if (complete.get("status") != "PREDICTION_PHASE_COMPLETE_LABEL_FIREWALL_INTACT" or
            checksum.get(relative) != digest):
        raise E123Error("Historical Stage 1-A K256 aggregate lacks a valid source checksum")
    artifact = read_json(source)
    macro = artifact["descriptive_macro_summaries"]["macro_over_20_trajectories"]
    by_category = artifact["descriptive_macro_summaries"]["by_class_support_seed_macro"]
    selected = {}
    for method in ("B0", "B3"):
        metrics = macro[method]
        selected[method] = {
            key: value.get("macro_mean_over_20_class_seed_trajectories")
            for key, value in metrics.items()
            if key in ("balanced_accuracy", "fpr", "fnr",
                       "worst_defined_full_window_balanced_accuracy")
        }
    class_summary = {}
    for method in ("B0", "B3"):
        class_summary[method] = {}
        for category in CATEGORIES:
            group = by_category[method].get(category)
            if not isinstance(group, Mapping):
                continue
            class_summary[method][category] = {
                key: value.get("macro_mean_over_support_seeds")
                for key, value in group.items()
                if key in ("balanced_accuracy", "fpr", "fnr") and isinstance(value, Mapping)
            }
    return {
        "classification": "HISTORICAL_STAGE1A_K256_DEVELOPMENT_EVIDENCE; NOT POOLED WITH E123 K1024",
        "trajectory_count": int(complete["trajectory_count"]),
        "event_count": int(complete["event_count"]),
        "native_or_gpu_rerun": False,
        "per_method_macro": selected,
        "per_category_macro": class_summary,
        "source_sha256": digest,
        "source_file": source.relative_to(ROOT).as_posix(),
        "labels_already_exposed_historically": True,
    }


def _write_report_artifact(budget: OutputBudget, name: str, value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload["artifact_sha256"] = sha256_json(payload)
    budget.write_once(name, canonical_bytes(payload))
    return payload["artifact_sha256"]


def _run_stage_b(run_path: Path, run_id: str, contract: Mapping[str, Any],
                 conditions: Sequence[Condition], global_seal: Mapping[str, Any]
                 ) -> dict[str, Any]:
    require_complete_stage_a(global_seal)
    budget = OutputBudget(run_path)
    import numpy as np
    import os as _os
    original_cuda = _os.environ.get("CUDA_VISIBLE_DEVICES")
    _os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        e2_conditions: dict[str, Any] = {}
        e1_conditions: dict[str, Any] = {}
        e3_conditions: dict[str, Any] = {}
        resource_rows: dict[str, Any] = {}
        scores_by_condition: dict[str, dict[str, list[float]]] = {}
        metrics_by_condition: dict[str, dict[str, dict[str, Any]]] = {}
        e3_predictions: dict[str, dict[str, list[int]]] = {}
        category_label_counts: dict[str, dict[str, int]] = {}
        seals_by_id = {row["condition_id"]: row for row in global_seal["condition_seals"]}
        for condition in conditions:
            condition_seal = _verify_condition_seal(run_path, condition, run_id)
            if seals_by_id[condition.condition_id]["seal_sha256"] != condition_seal["seal_sha256"]:
                raise E123Error("Global Stage A seal does not bind each condition seal")
            scores_rows, snapshots, feedback = _read_condition_stage_a(
                run_path, condition, condition_seal)
            from experiments.calibration.r3_fullstream_2x2 import JSON_ROOT
            all_groups = {event.original_position: event.source_group_sha256
                          for event in condition.events}
            reader = RandomAccessQueriedLabelReader(
                category=condition.category,
                path=JSON_ROOT / f"{condition.category}.json",
                expected_source_sha256=condition.class_json_sha256,
                allowed_positions=range(len(condition.events)),
                group_sha256_by_position=all_groups)
            try:
                labels_original = reader.stage_b_all_labels(global_seal=global_seal)
            finally:
                reader.close()
            labels_order = [int(labels_original[event.original_position])
                            for event in condition.events]
            category_label_counts[condition.condition_id] = {
                "normal": sum(label == 0 for label in labels_order),
                "defect": sum(label == 1 for label in labels_order),
            }
            feedback_by_position = {int(row["position"]): row for row in feedback}
            for position, feedback_row in feedback_by_position.items():
                event = condition.events[position]
                if feedback_row["label"] != labels_original[event.original_position]:
                    raise E123Error("Sealed query feedback differs from Stage B source outcome")
            per_arm_scores = {
                arm: [float(row["scores"][arm]) for row in scores_rows]
                for arm in CONFIGURATION_IDS
            }
            scores_by_condition[condition.condition_id] = per_arm_scores
            theta0 = float(scores_rows[0]["theta0"])
            sigma0 = float(scores_rows[0]["sigma0"])
            per_arm_metrics = {
                arm: _binary_metrics(labels_order, values, theta0)
                for arm, values in per_arm_scores.items()
            }
            metrics_by_condition[condition.condition_id] = per_arm_metrics
            condition_seal_path = (run_path / "stage_a" / "conditions" /
                                   condition.condition_id / "condition_seal.json")
            full_seal = read_json(condition_seal_path)
            resource_rows[condition.condition_id] = full_seal["resource_summary"]
            e2_conditions[condition.condition_id] = {
                "category": condition.category,
                "support_seed": condition.support_seed,
                "stream_order": condition.order,
                "event_count": len(labels_order),
                "query_count": len(condition.query_positions),
                "query_rate": len(condition.query_positions) / len(labels_order),
                "query_label_counts": {
                    "normal": condition_seal["query_normal_count"],
                    "defect": condition_seal["query_defect_count"],
                    "total": condition_seal["query_normal_count"] +
                    condition_seal["query_defect_count"],
                },
                "metrics_by_configuration": per_arm_metrics,
            }
            fixed_offset = float(contract["design"]["e3"]["fixed_offset_selection"]
                                 ["selection_by_target_category"][condition.category]
                                 ["coefficient"])
            c64_initial = list(snapshots[0]["c64_scores"])
            e3_metric, predictions = _replay_threshold_controllers(
                condition, scores_rows, c64_initial, theta0, sigma0,
                fixed_offset, feedback, labels_order)
            e3_predictions[condition.condition_id] = predictions
            e3_conditions[condition.condition_id] = {
                "category": condition.category, "support_seed": condition.support_seed,
                "stream_order": condition.order, "metrics_by_controller": e3_metric,
                "fixed_offset_coefficient": fixed_offset,
                "fixed_offset_threshold": theta0 + fixed_offset * sigma0,
                "b2_initialization": "original Sidecar native C64 quantile; adapted with unqueried fallback disabled",
                "b3_snapshot_count": len(snapshots),
                "b3_memory_versions_replayed": [int(row["memory_version"]) for row in snapshots],
            }
            # Static has no memory updates, so its B3 row equals Frozen exactly.
            static_scores = per_arm_scores["STATIC"]
            greedy_scores = per_arm_scores["GREEDY_K1024"]
            support_self_values = [float(row["score"])
                                   for row in condition_seal["support_self_scores"]]
            if len(support_self_values) != 4 or not all(
                    math.isfinite(value) for value in support_self_values):
                raise E123Error("E1 support self-score diagnostic is incomplete")
            support_self_theta = float(np.quantile(support_self_values, .90, method="higher"))
            support_static_metrics = _binary_metrics(
                labels_order, static_scores, support_self_theta)
            support_greedy_metrics = _binary_metrics(
                labels_order, greedy_scores, support_self_theta)
            b3_greedy_thresholds = [
                float(row["b3_greedy1024_threshold_before"]) for row in scores_rows]
            static_frozen = _binary_metrics(labels_order, static_scores, theta0)
            static_b3 = _binary_metrics(labels_order, static_scores, theta0)
            greedy_frozen = _binary_metrics(labels_order, greedy_scores, theta0)
            greedy_b3_pred = [int(score >= threshold)
                              for score, threshold in zip(greedy_scores, b3_greedy_thresholds)]
            greedy_b3_metrics = _metrics_from_predictions(
                labels_order, greedy_scores, greedy_b3_pred)
            e1_conditions[condition.condition_id] = {
                "category": condition.category, "support_seed": condition.support_seed,
                "stream_order": condition.order,
                "static_frozen": static_frozen,
                "static_b3_no_update_equivalent": static_b3,
                "greedy_k1024_frozen": greedy_frozen,
                "greedy_k1024_b3": greedy_b3_metrics,
                "ranking_effect_static_minus_greedy": {
                    "auroc_delta_greedy_minus_static":
                        per_arm_metrics["GREEDY_K1024"]["auroc"] - per_arm_metrics["STATIC"]["auroc"],
                    "ap_delta_greedy_minus_static":
                        per_arm_metrics["GREEDY_K1024"]["average_precision"] -
                        per_arm_metrics["STATIC"]["average_precision"],
                },
                "threshold_effect_at_static_score": 0.0,
                "threshold_effect_at_greedy_score_ber_delta":
                    greedy_b3_metrics["ber"] - greedy_frozen["ber"],
                "support_self_scores": condition_seal["support_self_scores"],
                "support_self_matching_threshold_diagnostic": {
                    "method": "higher empirical q90 of four support self-scores",
                    "online_use": False,
                    "support_self_threshold": support_self_theta,
                    "c64_nonself_threshold": theta0,
                    "support_minus_c64_threshold": support_self_theta - theta0,
                    "static_metrics_at_support_self_threshold": support_static_metrics,
                    "static_error_count_delta_vs_c64": {
                        key: support_static_metrics[key] - static_frozen[key]
                        for key in ("tp", "fp", "fn", "tn")
                    },
                    "greedy_k1024_metrics_at_support_self_threshold": support_greedy_metrics,
                    "greedy_k1024_error_count_delta_vs_c64": {
                        key: support_greedy_metrics[key] - greedy_frozen[key]
                        for key in ("tp", "fp", "fn", "tn")
                    },
                },
                "c64_theta0": theta0,
                "support_self_score_mean": sum(
                    float(row["score"]) for row in condition_seal["support_self_scores"]) / 4,
                "c64_initial_score_mean": sum(c64_initial) / 64,
                "c64_initial_score_q90": theta0,
                "difference_in_differences_ber": (
                    (greedy_b3_metrics["ber"] - greedy_frozen["ber"]) -
                    (static_b3["ber"] - static_frozen["ber"])),
            }
        # Category-equal summaries average the four paired conditions per class.
        def category_equal(values: Mapping[str, Mapping[str, Any]],
                           field: str, categories: Sequence[str] = CATEGORIES) -> dict[str, Any]:
            by_category: dict[str, float | None] = {}
            for category in categories:
                observations = [float(row[field]) for cid, row in values.items()
                                if row.get("category") == category and row.get(field) is not None]
                by_category[category] = sum(observations) / len(observations) if observations else None
            defined = [value for value in by_category.values() if value is not None]
            return {"by_category": by_category,
                    "category_equal_mean": sum(defined) / len(defined) if defined else None,
                    "defined_category_count": len(defined)}

        def paired_condition_differences(values: Mapping[str, float]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for category in CATEGORIES:
                rows = {(seed, order): float(values[
                    f"{category}_s{seed}_{order}"])
                        for seed in SUPPORT_SEEDS for order in STREAM_ORDERS}
                result[category] = {
                    "support_seed_1_minus_0_by_stream_order": {
                        order: rows[(1, order)] - rows[(0, order)]
                        for order in STREAM_ORDERS},
                    "shuffle_minus_json_by_support_seed": {
                        str(seed): rows[(seed, "group_shuffle_seed0")] - rows[(seed, "json")]
                        for seed in SUPPORT_SEEDS},
                }
            return result

        e2_summary: dict[str, Any] = {"by_condition": e2_conditions,
                                      "category_equal_metrics": {}}
        for arm in CONFIGURATION_IDS:
            e2_summary["category_equal_metrics"][arm] = {
                metric: category_equal({
                    cid: {"category": e2_conditions[cid]["category"], metric: row[metric]}
                    for cid, rows in metrics_by_condition.items()
                    for row in [rows[arm]]}, metric)
                for metric in ("auroc", "average_precision", "ber", "fpr", "fnr")
            }
        resource_aggregate: dict[str, Any] = {}
        for arm in CONFIGURATION_IDS[1:]:
            vals = [resource["mutable_arms"][arm]
                    for resource in resource_rows.values()]
            resource_aggregate[arm] = {
                "prototype_tensor_bytes_by_condition": [v["prototype_tensor_bytes"] for v in vals],
                "metadata_runtime_bytes_estimate_by_condition": [
                    v["metadata_runtime_bytes_estimate"] for v in vals],
                "metadata_tensor_bytes_by_condition": [
                    v["metadata_tensor_bytes"] for v in vals],
                "allocated_persistent_tensor_bytes_by_condition": [
                    v["allocated_persistent_tensor_bytes"] for v in vals],
                "python_metadata_bytes_estimate_by_condition": [
                    v["python_metadata_bytes_estimate"] for v in vals],
                "persistent_mutable_bytes_estimate_by_condition": [
                    v["persistent_mutable_bytes_estimate"] for v in vals],
                "final_occupied_slots_by_condition": [
                    [layer["occupied_slots"] for layer in v["layers"]] for v in vals],
                "dtype_by_condition": [v["dtype"] for v in vals],
                "accepted_patches": sum(int(v["accepted_patches"]) for v in vals),
                "rejected_patches": sum(int(v["rejected_patches"]) for v in vals),
                "replaced_patches": sum(int(v["replaced_patches"]) for v in vals),
                "candidate_patches": sum(int(v["candidate_patches"]) for v in vals),
            }
        e2_summary["resource_aggregate"] = resource_aggregate
        e2_summary["resource_by_condition"] = resource_rows
        e2_summary["capacity_sensitivity"] = {
            str(capacity): {
                "category_equal_auroc_by_configuration": {
                    f"{method}_K{capacity}": e2_summary["category_equal_metrics"][
                        f"{method}_K{capacity}"]["auroc"]["category_equal_mean"]
                    for method in MUTABLE_METHODS
                },
                "greedy_minus_static_category_equal_auroc": (
                    e2_summary["category_equal_metrics"][f"GREEDY_K{capacity}"][
                        "auroc"]["category_equal_mean"] -
                    e2_summary["category_equal_metrics"]["STATIC"][
                        "auroc"]["category_equal_mean"]),
            } for capacity in CAPACITIES
        }
        e2_summary["interpretation"] = {
            "primary_metric": "category-equal prequential AUROC",
            "same_capacity_tolerance": BYTE_TOLERANCE,
            "equal_K_not_claimed_equal_total_bytes": True,
            "greedy_semantics": "common-candidate online coreset; differs from R7/E6 native current-state candidate selection",
        }
        e2_summary["paired_support_and_stream_sensitivity"] = {
            arm: paired_condition_differences({
                condition_id: float(rows[arm]["auroc"])
                for condition_id, rows in metrics_by_condition.items()})
            for arm in CONFIGURATION_IDS
        }
        e1_summary = {"by_condition": e1_conditions}
        for field in ("ranking_effect_static_minus_greedy",):
            e1_summary[field] = {
                metric: category_equal({
                    cid: {"category": row["category"], metric: row[field][metric]}
                    for cid, row in e1_conditions.items()}, metric)
                for metric in ("auroc_delta_greedy_minus_static", "ap_delta_greedy_minus_static")
            }
        e1_summary["mean_difference_in_differences_ber"] = category_equal({
            cid: {"category": row["category"],
                  "value": row["difference_in_differences_ber"]}
            for cid, row in e1_conditions.items()}, "value")
        e1_summary["paired_support_and_stream_sensitivity"] = {
            "greedy_minus_static_auroc": paired_condition_differences({
                condition_id: float(row["ranking_effect_static_minus_greedy"][
                    "auroc_delta_greedy_minus_static"])
                for condition_id, row in e1_conditions.items()}),
            "greedy_b3_minus_frozen_ber": paired_condition_differences({
                condition_id: float(row["threshold_effect_at_greedy_score_ber_delta"])
                for condition_id, row in e1_conditions.items()}),
        }
        e1_summary["interpretation"] = {
            "static_b3_equals_frozen": True,
            "difference_in_differences_is_observed_interaction_only": True,
            "original_manuscript_static_parity": "NOT_ESTABLISHED",
        }
        e3_summary: dict[str, Any] = {"by_condition": e3_conditions,
                                      "category_equal_metrics": {}}
        controller_names = (
            "FROZEN_C64", "FIXED_OFFSET", "B1_ROLLING_QUERIED_NORMAL",
            "B2_ORIGINAL_SIDECAR_ADAPTED", "B3_CURRENT_STATE_C64")
        for controller in controller_names:
            controller_rows = {
                cid: {"category": e3_conditions[cid]["category"],
                      **e3_conditions[cid]["metrics_by_controller"][controller]}
                for cid in e3_conditions}
            e3_summary["category_equal_metrics"][controller] = {
                metric: category_equal(controller_rows, metric)
                for metric in ("ber", "fpr", "fnr", "auroc", "threshold_total_variation",
                               "threshold_update_count", "threshold_startup_coverage")
            }
        b3_ber_by_category = e3_summary["category_equal_metrics"][
            "B3_CURRENT_STATE_C64"]["ber"]["by_category"]
        e3_summary["b3_minus_comparator_ber"] = {}
        for comparator in controller_names:
            if comparator == "B3_CURRENT_STATE_C64":
                continue
            comparator_values = e3_summary["category_equal_metrics"][comparator][
                "ber"]["by_category"]
            deltas = {
                category: (b3_ber_by_category[category] - comparator_values[category]
                           if b3_ber_by_category[category] is not None and
                           comparator_values[category] is not None else None)
                for category in CATEGORIES
            }
            defined = [value for value in deltas.values() if value is not None]
            e3_summary["b3_minus_comparator_ber"][comparator] = {
                "by_category": deltas,
                "category_equal_mean_delta": sum(defined) / len(defined) if defined else None,
                "status_is_descriptive_not_equivalence": True,
            }
        e3_summary["paired_support_and_stream_sensitivity"] = {
            f"B3_minus_{comparator}_ber": paired_condition_differences({
                condition_id: float(row["metrics_by_controller"][
                    "B3_CURRENT_STATE_C64"]["ber"] - row["metrics_by_controller"][
                        comparator]["ber"])
                for condition_id, row in e3_conditions.items()})
            for comparator in controller_names if comparator != "B3_CURRENT_STATE_C64"
        }
        e3_summary["stage1a_k256_historical_evidence"] = _stage1a_k256_evidence()
        e3_summary["fixed_offset_selection"] = contract["design"]["e3"]["fixed_offset_selection"]
        e3_summary["same_score_auroc_invariant_verified"] = True
        e3_summary["no_unqueried_online_controller_labels"] = True
        e3_summary["interpretation"] = {
            "categories_are_prior_exposed_development_data": True,
            "not_independent_confirmation": True,
            "k256_and_k1024_not_pooled": True,
        }
        global_seal_copy = dict(global_seal)
        require_complete_stage_a(global_seal_copy)
        # Seal Stage B's data use as aggregate-only; no label vector is written.
        stage_b_aggregate = {
            "schema": "recovery.e123.stage_b_aggregate.v1",
            "status": "COMPLETE_AFTER_GLOBAL_STAGE_A_SEAL",
            "run_id": run_id,
            "global_stage_a_seal_sha256": global_seal["seal_sha256"],
            "category_label_denominators": category_label_counts,
            "image_level_outcome_label_vectors_written": False,
            "e1": e1_summary, "e2": e2_summary, "e3": e3_summary,
        }
        aggregate_sha = _write_report_artifact(budget, "stage_b/aggregate.json", stage_b_aggregate)
        e1_sha = _write_report_artifact(budget, "E1_failure_decomposition.json", e1_summary)
        e2_sha = _write_report_artifact(budget, "E2_memory_comparison.json", e2_summary)
        e3_sha = _write_report_artifact(budget, "E3_calibration_comparison.json", e3_summary)
        protocol_parity = {
            "schema": "recovery.e123.protocol_parity.v1",
            "status": "PASS_WITH_DECLARED_DIFFERENCES",
            "native_static_reference_parity": {
                cid: row["static_native_reference_parity"]
                for cid, row in ((c.condition_id,
                                  read_json(run_path / "stage_a" / "conditions" /
                                            c.condition_id / "condition_seal.json"))
                                 for c in conditions)},
            "stage05_seed0_fixed_bank_receipt_matches": {
                cid: row["fixed_bank"]["historical_match"]
                for cid, row in ((c.condition_id,
                                  read_json(run_path / "stage_a" / "conditions" /
                                            c.condition_id / "condition_seal.json"))
                                 for c in conditions)},
            "stage05_seed0_c64_score_receipt_matches": {
                cid: row["c64"]["max_abs_error_to_historical_receipt"]
                for cid, row in ((c.condition_id,
                                  read_json(run_path / "stage_a" / "conditions" /
                                            c.condition_id / "condition_seal.json"))
                                 for c in conditions)},
            "historical_manuscript_baseline_parity": "ORIGINAL_BASELINE_PARITY_NOT_ESTABLISHED",
            "greedy_writer_parity": "COMMON_STATIC_CANDIDATE_GREEDY_IS_NOT_EXACT_R7_E6_NATIVE_GREEDY",
            "query_tape_source": "experiments.calibration.query_tape Q-A-TS-v1 seed37, rebuilt identically across paired Support seeds",
            "stage_a_global_seal_sha256": global_seal["seal_sha256"],
        }
        parity_sha = _write_report_artifact(budget, "E123_protocol_parity.json", protocol_parity)
        claims = _scientific_claims(e1_summary, e2_summary, e3_summary)
        unresolved_md = _render_unresolved_claims(claims, protocol_parity)
        budget.write_once("E123_unresolved_claims.md", unresolved_md.encode("utf-8"))
        report = _render_combined_report(
            run_id, contract, e1_summary, e2_summary, e3_summary,
            claims, global_seal, protocol_parity)
        budget.write_once("E123_combined_report.md", report.encode("utf-8"))
        provenance = {
            "schema": "recovery.e123.provenance.v1", "status": "COMPLETE",
            "run_id": run_id,
            "source_identity": contract["source_identity"],
            "contract_sha256": contract["contract_sha256"],
            "global_stage_a_seal_sha256": global_seal["seal_sha256"],
            "stage_b_aggregate_sha256": aggregate_sha,
            "e1_sha256": e1_sha, "e2_sha256": e2_sha, "e3_sha256": e3_sha,
            "protocol_parity_sha256": parity_sha,
            "condition_count": len(conditions),
            "event_count": global_seal["event_count"],
            "query_count": global_seal["query_count"],
            "stage_a_unqueried_labels_decoded": False,
            "stage_b_labels_accessed_after_global_stage_a_seal": True,
            "confirmation_categories_accessed": False,
            "d3_outputs_accessed_or_modified": False,
        }
        _write_report_artifact(budget, "E123_provenance.json", provenance)
        budget.verify_complete_namespace()
        return stage_b_aggregate
    finally:
        if original_cuda is None:
            _os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            _os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda


def _run_focused_cpu_tests() -> dict[str, Any]:
    test_path = ROOT / "tests/calibration/test_e123_recovery.py"
    if not test_path.is_file():
        raise E123Error("Focused E123 CPU regression test file is missing")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(test_path)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", timeout=300)
    output = result.stdout or ""
    match = re.search(r"(\d+)\s+passed", output)
    record = {
        "schema": "recovery.e123.cpu_verification.v1",
        "command": [sys.executable, "-m", "pytest", "-q",
                    test_path.relative_to(ROOT).as_posix()],
        "exit_code": int(result.returncode),
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "passed_test_count": int(match.group(1)) if match else None,
        "stdout_stderr_sha256": sha256_bytes(output.encode("utf-8")),
        "output_tail": output[-10000:],
        "test_file_sha256": sha256_file(test_path),
        "dino_loaded": False,
        "cuda_executed": False,
        "real_dataset_read": False,
    }
    if result.returncode != 0 or record["passed_test_count"] is None:
        raise E123Error(f"Focused E123 CPU tests failed:\n{output[-8000:]}")
    return record


def _write_incomplete_reports(budget: OutputBudget, run_id: str,
                              contract: Mapping[str, Any], error: BaseException,
                              completed_conditions: Sequence[str]) -> None:
    parity_path = budget.root / "E123_protocol_parity.json"
    if not parity_path.exists():
        budget.failure("E123_protocol_parity.json", {
            "schema": "recovery.e123.protocol_parity.v1",
            "status": "INCOMPLETE_STAGE_A",
            "completed_conditions": list(completed_conditions),
            "historical_manuscript_baseline_parity": "NOT_ESTABLISHED",
            "failure": f"{type(error).__name__}: {error}",
        })
    unresolved_path = budget.root / "E123_unresolved_claims.md"
    if not unresolved_path.exists():
        budget.write_once(
            "E123_unresolved_claims.md",
            ("# E123 unresolved claims\n\n"
             "Stage A did not seal all sixteen paired conditions. Stage B was not run.\n")
            .encode("utf-8"), failure=True)
    provenance_path = budget.root / "E123_provenance.json"
    if not provenance_path.exists():
        budget.failure("E123_provenance.json", {
            "schema": "recovery.e123.provenance.v1", "status": "INCOMPLETE",
            "run_id": run_id, "source_identity": contract.get("source_identity"),
            "contract_sha256": contract.get("contract_sha256"),
            "completed_conditions": list(completed_conditions),
            "error_type": type(error).__name__, "error": str(error),
            "stage_b_started": (budget.root / "stage_a" / "global_seal.json").is_file(),
            "confirmation_categories_accessed": False,
            "d3_outputs_accessed_or_modified": False,
        })


def _run_campaign(run_id: str) -> int:
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,60}", run_id):
        raise E123Error("Run ID must be unique alphanumeric text")
    run_path = OUTPUT_BASE / f"recovery_e123_{run_id}"
    _safe_output_root(run_path, must_exist=False)
    if run_path.exists():
        raise E123Error("Refusing existing E123 output namespace; partial runs are never resumed")
    disk_preflight = _preflight_disk(OUTPUT_BASE)
    cpu_verification = _run_focused_cpu_tests()
    if cpu_verification.get("status") != "PASS":
        raise E123Error("Focused E123 CPU verification failed; real execution is forbidden")
    contract, conditions = build_contract(run_id)
    launch_token = secrets.token_hex(32)
    contract["supervisor_launch_token_sha256"] = sha256_bytes(launch_token.encode("ascii"))
    contract["initial_disk_preflight"] = disk_preflight
    contract.pop("contract_sha256", None)
    contract["contract_sha256"] = sha256_json(contract)
    budget = OutputBudget(run_path, create=True)
    completed: list[str] = []
    try:
        budget.write_once("E123_contract.json", canonical_bytes(contract))
        budget.write_once("E123_cpu_verification.json", canonical_bytes(cpu_verification))
        from experiments.calibration.r1_runtime_supervisor import run_supervised_child
        for condition in conditions:
            if _preflight_disk(run_path)["free_bytes"] < MIN_FREE_BYTES:
                raise E123Error("Free disk space fell below the 10 GiB launch threshold")
            child_env = dict(os.environ)
            child_env.update({
                "PYTHONHASHSEED": "0",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "E123_SUPERVISOR_LAUNCH_TOKEN": launch_token,
            })
            command = [sys.executable, "-m", "experiments.calibration.e123_recovery",
                       "--stage-a-child", "--run-id", run_id,
                       "--condition-id", condition.condition_id]
            supervisor_result = run_supervised_child(
                command, timeout_seconds=SUPERVISOR_TIMEOUT_SECONDS,
                env=child_env, cwd=ROOT, output_dir=None)
            result_data = (asdict(supervisor_result)
                           if hasattr(supervisor_result, "__dataclass_fields__")
                           else dict(vars(supervisor_result)))
            supervisor_rel = (
                f"stage_a/conditions/{condition.condition_id}/supervisor.json")
            budget.write_once(supervisor_rel, canonical_bytes({
                "schema": "recovery.e123.supervisor_receipt.v1",
                "condition_id": condition.condition_id,
                "timeout_seconds": SUPERVISOR_TIMEOUT_SECONDS,
                "automatic_restart": False,
                "result": result_data,
            }))
            exit_code = result_data.get("exit_code")
            timed_out = bool(result_data.get("timed_out", False))
            termination_confirmed = bool(result_data.get("termination_confirmed", False))
            blocking_failure = result_data.get("blocking_failure")
            if exit_code != 0 or timed_out or not termination_confirmed or blocking_failure:
                detail = {
                    "condition_id": condition.condition_id,
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                    "termination_confirmed": termination_confirmed,
                    "blocking_failure": blocking_failure,
                    "supervisor_result": result_data,
                }
                raise E123Error(f"Stage A child failed or exceeded supervision: {detail}")
            _verify_condition_seal(run_path, condition, run_id)
            completed.append(condition.condition_id)
            budget.append("stage_a/progress.jsonl", canonical_bytes({
                "completed_condition_count": len(completed),
                "condition_count": len(conditions),
                "condition_id": condition.condition_id,
                "completed_event_count": sum(len(c.events) for c in conditions
                                              if c.condition_id in completed),
            }), sync=True)
        global_seal = _seal_all_stage_a(run_path, run_id, conditions)
        global_seal["seal_sha256"] = sha256_json(global_seal)
        budget.write_once("stage_a/global_seal.json", canonical_bytes(global_seal))
        global_file = read_json(run_path / "stage_a/global_seal.json")
        require_complete_stage_a(global_file)
        _run_stage_b(run_path, run_id, contract, conditions, global_file)
        return 0
    except Exception as exc:
        budget.failure("stage_a/failure.json", {
            "schema": "recovery.e123.run_failure.v1", "run_id": run_id,
            "error_type": type(exc).__name__, "error": str(exc),
            "completed_conditions": list(completed),
            "stage_b_authorized": (run_path / "stage_a" / "global_seal.json").is_file(),
            "automatic_restart": False,
        })
        _write_incomplete_reports(budget, run_id, contract, exc, completed)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stage-a-child", action="store_true")
    parser.add_argument("--condition-id", default=None)
    args = parser.parse_args(argv)
    if args.stage_a_child:
        if not args.run_id or not args.condition_id:
            parser.error("--stage-a-child requires --run-id and --condition-id")
        return _run_stage_a_condition(args.run_id, args.condition_id)
    if args.condition_id is not None:
        parser.error("--condition-id is valid only with --stage-a-child")
    run_id = args.run_id or (
        time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "_" + uuid.uuid4().hex[:8])
    try:
        return _run_campaign(run_id)
    except Exception as exc:
        print(f"E123 failed before or during supervised campaign: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


def _scientific_claims(e1: Mapping[str, Any], e2: Mapping[str, Any],
                       e3: Mapping[str, Any]) -> list[dict[str, Any]]:
    def signs(by_category: Mapping[str, Any], field: str) -> tuple[float, int, int]:
        values = [float(v) for v in by_category.values() if v is not None]
        return (sum(values) / len(values) if values else 0.0,
                sum(v > 0 for v in values), sum(v < 0 for v in values))

    greedy = e2["category_equal_metrics"]["GREEDY_K1024"]["auroc"]["by_category"]
    static = e2["category_equal_metrics"]["STATIC"]["auroc"]["by_category"]
    rank_deltas = {cat: (greedy[cat] - static[cat])
                   for cat in CATEGORIES if greedy.get(cat) is not None and static.get(cat) is not None}
    rank_mean, rank_positive, rank_negative = signs(rank_deltas, "")
    controllers = e3["category_equal_metrics"]
    b3_ber = controllers["B3_CURRENT_STATE_C64"]["ber"]["by_category"]
    frozen_ber = controllers["FROZEN_C64"]["ber"]["by_category"]
    b3_delta = {cat: (b3_ber[cat] - frozen_ber[cat])
                for cat in CATEGORIES if b3_ber.get(cat) is not None and frozen_ber.get(cat) is not None}
    b3_mean, b3_pos, b3_neg = signs(b3_delta, "")
    def classify_delta(mean: float, positive: int, negative: int) -> str:
        if mean < 0 and positive == 0:
            return "SUPPORTED"
        if mean > 0 and negative == 0:
            return "NOT_SUPPORTED"
        return "MIXED"
    support_diagnostics = [row["support_self_matching_threshold_diagnostic"]
                           for row in e1["by_condition"].values()]
    threshold_changed_count = sum(
        abs(float(row["support_minus_c64_threshold"])) > 1e-12
        for row in support_diagnostics)
    error_count_changed = sum(any(int(value) != 0 for value in
                                  row["static_error_count_delta_vs_c64"].values()) or
                              any(int(value) != 0 for value in
                                  row["greedy_k1024_error_count_delta_vs_c64"].values())
                              for row in support_diagnostics)
    capacity_deltas = [float(e2["capacity_sensitivity"][str(capacity)][
        "greedy_minus_static_category_equal_auroc"]) for capacity in CAPACITIES]
    capacity_signs = {0 if abs(value) <= 1e-12 else (1 if value > 0 else -1)
                      for value in capacity_deltas}
    claims = [
        {"claim": "Normal-memory updating improves ranking over Static",
         "status": "SUPPORTED" if rank_mean > 0 and rank_positive >= 3 else
                   "NOT_SUPPORTED" if rank_mean <= 0 else "MIXED",
         "evidence": {"category_equal_auroc_delta": rank_mean,
                      "categories_positive": rank_positive,
                      "categories_negative": rank_negative}},
        {"claim": "The common-candidate Greedy retention rule beats FIFO, Reservoir and Random Replacement",
         "status": "SUPPORTED" if all(
             e2["category_equal_metrics"]["GREEDY_K1024"]["auroc"]["category_equal_mean"] >
             e2["category_equal_metrics"][f"{method}_K1024"]["auroc"]["category_equal_mean"]
             for method in ("FIFO", "RESERVOIR", "RANDOM_REPLACEMENT")) else "MIXED",
         "evidence": {"category_equal_auroc_by_method": {
             method: e2["category_equal_metrics"][f"{method}_K1024"]["auroc"]["category_equal_mean"]
             for method in ("GREEDY", "FIFO", "RESERVOIR", "RANDOM_REPLACEMENT")}}},
        {"claim": "Original v22 utility-aware STM is necessary",
         "status": "NOT_SUPPORTED" if e2["category_equal_metrics"]["UTILITY_STM_K1024"]["auroc"][
             "category_equal_mean"] <= e2["category_equal_metrics"]["GREEDY_K1024"]["auroc"][
             "category_equal_mean"] else "MIXED",
         "evidence": {"utility_stm_auroc": e2["category_equal_metrics"]["UTILITY_STM_K1024"][
             "auroc"]["category_equal_mean"],
             "greedy_auroc": e2["category_equal_metrics"]["GREEDY_K1024"][
                 "auroc"]["category_equal_mean"]}},
        {"claim": "The direction of Greedy-versus-Static ranking change is invariant across K256/K512/K1024",
         "status": "SUPPORTED" if len(capacity_signs) == 1 and 0 not in capacity_signs else
                   "MIXED" if len(capacity_signs) > 1 else "NOT_SUPPORTED",
         "evidence": {"greedy_minus_static_category_equal_auroc_by_capacity":
                      dict(zip(map(str, CAPACITIES), capacity_deltas))}},
        {"claim": "Support self-matching changes the initial operating threshold and at least one error count",
         "status": "SUPPORTED" if threshold_changed_count and error_count_changed else "NOT_SUPPORTED",
         "evidence": {"conditions_with_threshold_difference": threshold_changed_count,
                      "conditions_with_error_count_difference": error_count_changed,
                      "condition_count": len(support_diagnostics),
                      "diagnostic_is_not_used_online": True}},
        {"claim": "B3 reduces BER versus frozen C64 on identical Greedy-K1024 scores",
         "status": classify_delta(b3_mean, b3_pos, b3_neg),
         "evidence": {"category_equal_ber_delta_b3_minus_frozen": b3_mean,
                      "categories_b3_better": b3_neg,
                      "categories_b3_worse": b3_pos}},
        {"claim": "Greedy AUROC gain uniquely identifies normal-reference undercoverage",
         "status": "NOT_IDENTIFIABLE",
         "evidence": {"ranking_gain_is_not_a_unique_causal_diagnosis": True}},
        {"claim": "Historical full Hierarchy retirement should be reopened",
         "status": "NOT_IDENTIFIABLE",
         "evidence": {"E123_did_not_run_a_full_hierarchy_arm": True,
                      "historical_negative_result_preserved": True}},
    ]
    for comparator, comparison in e3["b3_minus_comparator_ber"].items():
        deltas = comparison["by_category"]
        values = [float(value) for value in deltas.values() if value is not None]
        mean = sum(values) / len(values) if values else 0.0
        positive = sum(value > 0 for value in values)
        negative = sum(value < 0 for value in values)
        claims.append({
            "claim": f"B3 has lower category BER than {comparator} on identical Greedy-K1024 scores",
            "status": classify_delta(mean, positive, negative) if values else "BLOCKED",
            "evidence": {"b3_minus_comparator_ber_by_category": deltas,
                         "category_equal_mean_delta": comparison["category_equal_mean_delta"],
                         "equivalence_margin_registered": False},
        })
    claims.append({
        "claim": "Threshold controllers are statistically interchangeable",
        "status": "NOT_IDENTIFIABLE",
        "evidence": {"no_equivalence_margin_or_independent_confirmation_was_registered": True},
    })
    return claims


def _render_unresolved_claims(claims: Sequence[Mapping[str, Any]],
                              parity: Mapping[str, Any]) -> str:
    lines = ["# E123 unresolved claims", "",
             "This is retrospective development evidence, not blind confirmation.", "",
             "## Claim classifications", ""]
    for claim in claims:
        lines.append(f"- **{claim['status']}** — {claim['claim']}: "
                     f"`{json.dumps(claim['evidence'], sort_keys=True)}`")
    lines += ["", "## Remaining limits", "",
              "- Original manuscript static-score parity is not established by this run.",
              "- E123 common-candidate Greedy is not exact R7/E6 native Greedy writer parity.",
              "- Full Hierarchy is not rerun; its historical negative evidence remains separate.",
              "- No independent confirmation category was accessed.",
              "- Official Source Group identity does not prove physical identity beyond the official key.",
              "- No significance claims treat images, support seeds, or orders as independent samples.",
              f"- Protocol parity status: `{parity.get('status')}`.", ""]
    return "\n".join(lines)
def _render_combined_report(run_id: str, contract: Mapping[str, Any],
                            e1: Mapping[str, Any], e2: Mapping[str, Any],
                            e3: Mapping[str, Any], claims: Sequence[Mapping[str, Any]],
                            global_seal: Mapping[str, Any],
                            parity: Mapping[str, Any]) -> str:
    cat_auc = {
        arm: {cat: e2["category_equal_metrics"][arm]["auroc"]["by_category"][cat]
              for cat in CATEGORIES}
        for arm in CONFIGURATION_IDS
    }
    cat_ber = {
        controller: {cat: e3["category_equal_metrics"][controller]["ber"]["by_category"][cat]
                     for cat in CATEGORIES}
        for controller in e3["category_equal_metrics"]
    }
    lines = [
        "# Recovery E123 integrated development experiment", "",
        f"Run ID: `{run_id}`", "",
        "## Scope and completion", "",
        f"Stage A sealed {global_seal['condition_count']}/16 paired conditions "
        f"({global_seal['event_count']} events; {global_seal['query_count']} queried event appearances).",
        "Stage B labels were opened only after the global Stage A seal. No confirmation data or D3 outputs were accessed.",
        "All four categories were previously exposed development data; these results are not independent confirmation.",
        "", "## E1: ranking and operating point", "",
        f"Category-equal Greedy K1024 minus Static AUROC: "
        f"`{e1['ranking_effect_static_minus_greedy']['auroc_delta_greedy_minus_static']['category_equal_mean']}`.",
        f"Category-equal observed score-by-threshold BER difference-in-differences: "
        f"`{e1['mean_difference_in_differences_ber']['category_equal_mean']}`.",
        "Static+B3 is an exact no-update equivalent to Static+Frozen in this design.",
        "Per-condition support self-score q90 versus legal C64 q90 threshold diagnostics, error-count deltas, and paired support/stream differences are in `E1_failure_decomposition.json`.",
        "", "## E2: normal-memory methods", "",
        "Category-equal prequential AUROC by configuration:", "",
        "| Configuration | Audiojack | Regulator | Switch | Rolled strip base | Macro |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in CONFIGURATION_IDS:
        macro = e2["category_equal_metrics"][arm]["auroc"]["category_equal_mean"]
        lines.append("| " + arm + " | " + " | ".join(
            str(cat_auc[arm][cat]) for cat in CATEGORIES) + f" | {macro} |")
    lines += ["", "Prototype tensor storage, metadata tensor storage, and Python metadata estimates are reported separately "
              "in `E2_memory_comparison.json`; equal K is not treated as equal total bytes.",
              "Support-seed and stream-order paired differences are reported by category.",
              "The common-candidate Greedy rule differs from R7/E6's current-state candidate selection.",
              "", "## E3: same-score threshold replay", "",
              "Category-equal BER by controller:", "",
              "| Controller | Audiojack | Regulator | Switch | Rolled strip base | Macro |",
              "|---|---:|---:|---:|---:|---:|"]
    for controller in cat_ber:
        macro = e3["category_equal_metrics"][controller]["ber"]["category_equal_mean"]
        lines.append("| " + controller + " | " + " | ".join(
            str(cat_ber[controller][cat]) for cat in CATEGORIES) + f" | {macro} |")
    lines += ["", "B3-minus-comparator category BER differences and support/stream paired differences are in `E3_calibration_comparison.json`.",
              "Historical Stage 1-A K256 results are reported separately and were not pooled with K1024.",
              "Fixed Offset used a preregistered finite grid and leave-one-category-out selection on archived Stage 1-A development trajectories.",
              "B2 is the original Sidecar adapted with unqueried-normal fallback disabled. B3 used C64 score snapshots captured during Stage A.",
              "", "## Parity, claims and limitations", "",
              f"Native protocol parity: `{parity['status']}`. Original manuscript baseline parity: "
              f"`{parity['historical_manuscript_baseline_parity']}`.",
              "Scientific claim statuses:", ""]
    for claim in claims:
        lines.append(f"- **{claim['status']}** — {claim['claim']}")
    lines += ["", "The run does not complete the broader paper recovery or establish independent generalization.", ""]
    return "\n".join(lines)
