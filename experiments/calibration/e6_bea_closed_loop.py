"""E6-BEA-v2 closed-loop orchestration and restricted Stage A/B boundaries.

Importing this module does not load a model, dataset, or CUDA runtime. The
coordinator is exercised with synthetic adapters in CPU tests. The optional
R7 adapter delegates scoring and writing to the already audited R1/R7 helpers.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import re
import stat
import subprocess
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from experiments.calibration.e6_bea_policy import (
    ARM_IDS,
    ARM_POLICIES,
    BEAComponents,
    QueryController,
    QueryProposal,
    compute_bea_components,
)


class E6ClosedLoopError(RuntimeError):
    """Raised when E6 identity, causal order, or sealing contracts fail."""


_HEX64 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_PUBLIC_KEYS = frozenset({
    "label", "anomaly_class", "mask", "mask_path", "ground_truth",
    "outcome", "private_feedback",
})
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
E6_REGISTERED_CATEGORIES = ("audiojack", "regulator", "switch", "rolled_strip_base")
E6_EXECUTION_SOURCE_PATHS = (
    "experiments/calibration/e6_bea_policy.py",
    "experiments/calibration/e6_bea_closed_loop.py",
    "tests/calibration/test_e6_bea_policy.py",
    "tests/calibration/test_e6_bea_closed_loop.py",
)
R7_HELPER_SOURCE_SHA256 = MappingProxyType({
    "experiments/calibration/r1_matched_correction_pilot.py":
        "3208e2c52277ad76351eb86968e362aa315effe9e3c734bfe2b6ff7bab0db448",
    "experiments/calibration/r7_shuffled_correction_4class.py":
        "0e2b0bc468cdcc1a8d72082487b1618817a45342fe93b99a00052ecb50f85b3a",
    "experiments/calibration/r3_fullstream_2x2.py":
        "0a67d893df6fe73399726f4cc60caa5967a6e3c2eb93535f23f7089350986741",
    "experiments/calibration/r6_capacity1024_audiojack_runner.py":
        "f9176d50420a6fc5474ebccb0cb2dd640cf0f3152911ea6c4c9736bad85d0165",
    "experiments/calibration/native.py":
        "3a347d65aaf4bfd6b41eff2ea7339d878065bb2e08ed64795f57b0458135dd45",
    "experiments/p0_protocol_audit.py":
        "b83918ec5aceb3ef63cca942fc9fb0e241559d9672d20bed422b15fcd48135e7",
    "experiments/calibration/profiles.py":
        "7ff59b6a3783bf4061ab75c6b0782b23a0f6e456b40457a022e5d0de13d6f7a5",
    "experiments/tmra_compact_memory.py":
        "2a76351e68aacf443e29e428255eba2e46eec3547351bcb721725662d0469644",
    "experiments/calibration/r2_exp2a_c1_strength.py":
        "4b798dcb1472e6dd630f58b3e62f3782dfb302f7579e942fcdb399885a5f33d3",
    "experiments/calibration/protocol.py":
        "16b39ead3975bf442047737bc5746814ef9697cd0a9962464ee388bbfc52da60",
    "experiments/calibration/controllers.py":
        "25fbdfc9a92e8bfb6fe983185dca3cdd5ed24d804c28838b4797e0ae8670cf7d",
    "experiments/calibration/stage1a_p12_label_provider.py":
        "4e51255ae894c2d6ffcc7142e8fd59d9714df94b18ed26ac996528adb163ceaf",
    "experiments/risk_safe/preflight.py":
        "2070ba770ac36af06f688b0450c711d67b7bca526e3939178d559216535c25ba",
    "experiments/risk_safe/native_bridge.py":
        "9575214ea8d14ebe32a537fa15afa332446b41d66c1c8c85f890d2251eaef8f7",
    "test_phase3_AIF_simplified_v22.py":
        "bfc871d5a9a32b2f5bdb9d7aab5177e702dec2e1110a6632f20bd13c20fcba30",
    "vmb_visionad_new.py":
        "bd6545bf5ff63acce01792e5d42aa964c88e73fb59670febeb55f2d1139a409a",
    "experiments/v02_memory.py":
        "3c4e247e3a41dc9acc8147a35aa892cf3f3571aca85beb4169809bd70ad48ae0",
})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_reparse_ancestors(path: Path) -> None:
    for candidate in (Path(path), *Path(path).parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        attrs = int(getattr(info, "st_file_attributes", 0))
        reparse = bool(attrs & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)))
        if stat.S_ISLNK(info.st_mode) or reparse:
            raise E6ClosedLoopError(f"E6 output path contains a symlink/reparse point: {candidate}")


def verify_e6_execution_identity(approved_commit: str, *, root: Path) -> dict[str, Any]:
    """Require a separately reviewed exact E6 commit and frozen native sources.

    R7's standalone runner pins its historical HEAD and therefore cannot be
    called after a new E6 commit. E6 instead binds direct helper reuse to this
    exact reviewed commit plus the independently frozen helper file hashes.
    """
    if (not isinstance(approved_commit, str) or
            not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", approved_commit)):
        raise E6ClosedLoopError("real execution requires the independently reviewed E6 commit SHA")
    root = Path(root).resolve(strict=True)
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8").stdout.strip()
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8").stdout.strip().lower()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise E6ClosedLoopError("cannot verify E6 branch and commit identity") from exc
    if branch != EXPECTED_BRANCH or head != approved_commit.lower():
        raise E6ClosedLoopError("current branch/HEAD do not match the separately reviewed E6 identity")
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", *E6_EXECUTION_SOURCE_PATHS],
            cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8").stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise E6ClosedLoopError("cannot verify that reviewed E6 source paths are clean") from exc
    if dirty.strip():
        raise E6ClosedLoopError("reviewed E6 source/test paths have uncommitted changes")
    fingerprints = {}
    for relative, expected in R7_HELPER_SOURCE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise E6ClosedLoopError(f"frozen R7 helper source is missing: {relative}")
        actual = _sha256_file(path)
        if actual != expected:
            raise E6ClosedLoopError(f"frozen R7 helper source SHA changed: {relative}")
        fingerprints[relative] = actual
    return {"branch": branch, "head": head, "approved_e6_commit": approved_commit.lower(),
            "r7_helper_source_sha256": fingerprints}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _config_record(value: Any, name: str) -> dict[str, Any]:
    if is_dataclass(value):
        record = asdict(value)
    elif isinstance(value, Mapping):
        record = dict(value)
    elif hasattr(value, "__dict__"):
        record = dict(vars(value))
    else:
        raise E6ClosedLoopError(f"{name} cannot be represented as a configuration receipt")
    _canonical_json(record)
    return record


def _write_json_once(path: Path, value: Mapping[str, Any]) -> str:
    path = Path(path)
    _reject_reparse_ancestors(path)
    _reject_public_label_fields(value)
    payload = _canonical_json(value)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def create_e6_output_namespace(run_id: str, *, repository_root: Path) -> Path:
    """Create a never-reused E6 output directory and its durable identity marker."""
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise E6ClosedLoopError("E6 run ID must be a 1-64 character filesystem-safe token")
    root = Path(repository_root).resolve(strict=True)
    output_parent = (root / "outputs" / "calibration").resolve(strict=True)
    _reject_reparse_ancestors(output_parent)
    run_dir = output_parent / f"e6_bea_v2_{run_id}"
    _reject_reparse_ancestors(run_dir)
    run_dir.mkdir(exist_ok=False)
    _write_json_once(run_dir / "run_namespace.json", {
        "schema": "e6.bea_v2.run_namespace.v1",
        "run_id": run_id,
        "status_at_creation": "INCOMPLETE",
    })
    return run_dir


def _reject_public_label_fields(value: Any, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_PUBLIC_KEYS:
                raise E6ClosedLoopError(f"public Stage A score contains forbidden field: {path}.{key}")
            _reject_public_label_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_public_label_fields(item, f"{path}[{index}]")


@dataclass(frozen=True)
class EventIdentity:
    position: int
    original_position: int
    event_id_sha256: str
    image_identity_sha256: str
    source_group_sha256: str
    relative_id: str
    camera_id: str
    timestamp: str
    image_path: str

    def __post_init__(self) -> None:
        if isinstance(self.position, bool) or int(self.position) < 0:
            raise E6ClosedLoopError("event position must be a nonnegative integer")
        if isinstance(self.original_position, bool) or int(self.original_position) < 0:
            raise E6ClosedLoopError("original JSON position must be nonnegative")
        for name in ("event_id_sha256", "image_identity_sha256", "source_group_sha256"):
            if not _HEX64.fullmatch(str(getattr(self, name))):
                raise E6ClosedLoopError(f"{name} must be a lowercase SHA-256 digest")
        if not self.relative_id or not self.camera_id or not self.image_path:
            raise E6ClosedLoopError("event image identity is incomplete")

    @classmethod
    def from_r7(cls, event: Any) -> "EventIdentity":
        return cls(
            position=int(event.position), original_position=int(event.original_position),
            event_id_sha256=str(event.event_id_sha256),
            image_identity_sha256=str(event.image_identity_sha256),
            source_group_sha256=str(event.source_group_sha256),
            relative_id=str(event.relative_id), camera_id=str(event.camera_id),
            timestamp=str(event.timestamp), image_path=str(event.image_path))


@dataclass(frozen=True)
class EncodedEvent:
    """One frozen encoder result shared by all seven arms for this event."""

    image: Any
    feature_layers: Sequence[Any]
    global_cls: Any
    encoding_seconds: float = 0.0


@dataclass(frozen=True)
class NativePreEventScore:
    """Per-arm native score and pre-event C1 evidence; context stays in memory."""

    gc_score: float
    c0_score: float
    c64_scores: Sequence[float]
    correction_top32: Sequence[float]
    correction_bank_size: int
    topk_dN_max: float
    topk_dN_mean: float
    scorer_state_version: int
    scorer_state_sha256: str
    context: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class FeedbackUpdate:
    label: int
    c1_changed: bool
    greedy_changed: bool
    scorer_state_version: int
    scorer_state_sha256: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class ArmEngine(Protocol):
    initial_state_sha256: str
    current_threshold: float

    def score(self, event: EventIdentity, encoded: EncodedEvent) -> NativePreEventScore: ...

    def observe_unqueried(self, event: EventIdentity, encoded: EncodedEvent,
                          score: NativePreEventScore) -> Mapping[str, Any]: ...

    def apply_feedback(self, event: EventIdentity, encoded: EncodedEvent,
                       score: NativePreEventScore, label: int) -> FeedbackUpdate: ...


class NormalCoverageMemory:
    """CPU representation of v22 NormalCoverageMemory's support/dynamic geometry."""

    def __init__(self, support_features: Any, *, max_dynamic: int = 1024):
        rows = np.asarray(support_features, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[0] != 4 or rows.shape[1] == 0:
            raise E6ClosedLoopError("Coverage Memory requires four deployed support CLS features")
        if not np.isfinite(rows).all():
            raise E6ClosedLoopError("support CLS features must be finite")
        self.support_features = self._normalize_rows(rows)
        similarities = self.support_features @ self.support_features.T
        np.fill_diagonal(similarities, -1.0)
        self.support_reference_distances = np.maximum(0.0, 1.0 - similarities.max(axis=1))
        self.dynamic_features = np.empty((0, rows.shape[1]), dtype=np.float32)
        self.dynamic_reference_distances = np.empty((0,), dtype=np.float32)
        self.max_dynamic = int(max_dynamic)
        if self.max_dynamic < 0:
            raise E6ClosedLoopError("Coverage Memory dynamic capacity cannot be negative")

    @staticmethod
    def _normalize_rows(rows: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return rows / np.maximum(norms, 1e-12)

    @property
    def dynamic_size(self) -> int:
        return int(self.dynamic_features.shape[0])

    def novelty_percentile(self, feature: Any) -> tuple[float, dict[str, Any]]:
        query = np.asarray(feature, dtype=np.float32).reshape(-1)
        if query.shape != (self.support_features.shape[1],) or not np.isfinite(query).all():
            raise E6ClosedLoopError("global CLS feature has an invalid shape or value")
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        bank = (self.support_features if self.dynamic_size == 0 else
                np.concatenate((self.support_features, self.dynamic_features), axis=0))
        nn_distance = max(0.0, 1.0 - float(np.max(bank @ query)))
        refs = np.concatenate((self.support_reference_distances,
                               self.dynamic_reference_distances))
        if refs.size == 0:
            pct = 1.0
        else:
            # v22 counts reference distances <= the current nearest distance.
            pct = float(np.count_nonzero(refs <= nn_distance) / refs.size)
        return pct, {"distance_nn": nn_distance, "novelty_percentile": pct,
                     "support_size": int(self.support_features.shape[0]),
                     "dynamic_size": self.dynamic_size, "reference_size": int(refs.size)}

    def observe_normal(self, feature: Any) -> None:
        query = np.asarray(feature, dtype=np.float32).reshape(-1)
        if query.shape != (self.support_features.shape[1],) or not np.isfinite(query).all():
            raise E6ClosedLoopError("queried normal CLS feature has an invalid shape or value")
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        bank = (self.support_features if self.dynamic_size == 0 else
                np.concatenate((self.support_features, self.dynamic_features), axis=0))
        reference_distance = max(0.0, 1.0 - float(np.max(bank @ query)))
        self.dynamic_features = np.concatenate((self.dynamic_features, query[None, :]), axis=0)
        self.dynamic_reference_distances = np.append(
            self.dynamic_reference_distances, np.float32(reference_distance))
        if self.dynamic_size > self.max_dynamic:
            keep = self.max_dynamic
            self.dynamic_features = self.dynamic_features[-keep:] if keep else self.dynamic_features[:0]
            self.dynamic_reference_distances = (
                self.dynamic_reference_distances[-keep:] if keep else
                self.dynamic_reference_distances[:0])

    def state_sha256(self) -> str:
        digest = hashlib.sha256()
        for array in (self.support_features, self.support_reference_distances,
                      self.dynamic_features, self.dynamic_reference_distances):
            digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()


def validate_five_view_groups(events: Sequence[EventIdentity]) -> dict[str, int]:
    """Preflight the registered five-camera Source Group assumption for Q0."""
    groups: dict[str, list[EventIdentity]] = {}
    for expected_position, event in enumerate(events):
        if event.position != expected_position:
            raise E6ClosedLoopError("event metadata is not a contiguous shuffled-position sequence")
        groups.setdefault(event.source_group_sha256, []).append(event)
    for group, views in groups.items():
        cameras = [view.camera_id for view in views]
        positions = [view.position for view in views]
        if len(views) != 5 or set(cameras) != {"1", "2", "3", "4", "5"}:
            raise E6ClosedLoopError(
                f"Source Group {group} violates the registered camera-1-through-5 view assumption")
        if positions != list(range(positions[0], positions[0] + 5)):
            raise E6ClosedLoopError("five camera views must remain contiguous within each shuffled group")
    return {"source_groups": len(groups), "events": len(events), "views_per_group": 5}


def validate_data_role_group_disjointness(
    events: Sequence[EventIdentity], support_groups: Sequence[str],
    c64_groups: Sequence[str],
) -> None:
    """Require four Support, 64 C64, and TEST Source Groups to be disjoint."""
    support = tuple(str(value) for value in support_groups)
    c64 = tuple(str(value) for value in c64_groups)
    if (len(support) != 4 or len(set(support)) != 4 or
            len(c64) != 64 or len(set(c64)) != 64 or
            any(not _HEX64.fullmatch(value) for value in (*support, *c64))):
        raise E6ClosedLoopError("Support/C64 Source Group identities are incomplete or malformed")
    test = {event.source_group_sha256 for event in events}
    if set(support) & set(c64) or set(support) & test or set(c64) & test:
        raise E6ClosedLoopError("Support, C64, and TEST Source Groups must be disjoint")


def make_e6_event_stream(stage05_rows: Sequence[Any]) -> tuple[EventIdentity, ...]:
    """Reuse only R7's deterministic group shuffle; never build its fixed tape."""
    from experiments.calibration import r7_shuffled_correction_4class as r7

    shuffled = r7.deterministic_group_shuffle(stage05_rows, seed=0)
    events = tuple(EventIdentity.from_r7(event) for event in shuffled)
    validate_five_view_groups(events)
    return events


def validate_test_image_projection(path: Path, category: str,
                                   events: Sequence[EventIdentity]) -> None:
    """Validate only TEST image identity metadata, without decoding outcomes/masks."""
    from experiments.calibration import r7_shuffled_correction_4class as r7

    r7.validate_image_projection(Path(path), str(category), events)


@dataclass(frozen=True)
class SealedQueryReceipt:
    arm_id: str
    position: int
    original_position: int
    event_id_sha256: str
    image_identity_sha256: str
    source_group_sha256: str
    _seal_token: object = field(repr=False, compare=False)


class ScoreLedger:
    """Append-only, fsynced Stage A score ledger with no-overwrite creation."""

    def __init__(self, path: Path):
        self.path = Path(path)
        _reject_reparse_ancestors(self.path)
        if not self.path.parent.is_dir() or self.path.parent.is_symlink():
            raise E6ClosedLoopError("score ledger parent must be an existing plain directory")
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError(f"E6 score ledger already exists: {self.path}")
        self._stream = self.path.open("xb")
        self._token = object()
        self._last_position = -1
        self._receipts: dict[tuple[int, str], SealedQueryReceipt] = {}
        self._closed = False

    def seal_event(self, event: EventIdentity,
                   arm_rows: Mapping[str, Mapping[str, Any]]) -> str:
        if self._closed:
            raise E6ClosedLoopError("score ledger is closed")
        if event.position != self._last_position + 1:
            raise E6ClosedLoopError("Stage A score events must seal in shuffled order")
        if set(arm_rows) != set(ARM_IDS):
            raise E6ClosedLoopError("every event must seal all seven arm decisions together")
        row = {
            "schema": "e6.bea_v2.stage_a_score.v1",
            "position": event.position,
            "original_json_position": event.original_position,
            "event_id_sha256": event.event_id_sha256,
            "image_identity_sha256": event.image_identity_sha256,
            "source_group_sha256": event.source_group_sha256,
            "arm_rows": {arm: dict(arm_rows[arm]) for arm in ARM_IDS},
        }
        _reject_public_label_fields(row)
        payload = _canonical_json(row)
        self._stream.write(payload)
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._last_position = event.position
        for arm_id, arm_row in row["arm_rows"].items():
            if arm_row.get("query_executed") is True:
                receipt = SealedQueryReceipt(
                    arm_id, event.position, event.original_position, event.event_id_sha256,
                    event.image_identity_sha256, event.source_group_sha256, self._token)
                self._receipts[(event.position, arm_id)] = receipt
        return hashlib.sha256(payload).hexdigest()

    def query_receipt(self, arm_id: str, event: EventIdentity) -> SealedQueryReceipt:
        receipt = self._receipts.get((event.position, arm_id))
        if receipt is None:
            raise E6ClosedLoopError("event has no durably sealed executed-query receipt for this arm")
        if (receipt.original_position != event.original_position or
                receipt.event_id_sha256 != event.event_id_sha256 or
                receipt.image_identity_sha256 != event.image_identity_sha256 or
                receipt.source_group_sha256 != event.source_group_sha256):
            raise E6ClosedLoopError("sealed query receipt does not match the current event identity")
        return receipt

    def authorize_receipt(self, receipt: SealedQueryReceipt, arm_id: str,
                          event: EventIdentity) -> bool:
        key = (event.position, arm_id)
        stored = self._receipts.get(key)
        valid = bool(
            stored is receipt and receipt._seal_token is self._token and
            receipt.arm_id == arm_id and receipt.position == event.position and
            receipt.original_position == event.original_position and
            receipt.event_id_sha256 == event.event_id_sha256 and
            receipt.image_identity_sha256 == event.image_identity_sha256 and
            receipt.source_group_sha256 == event.source_group_sha256)
        if valid:
            self._receipts.pop(key, None)
        return valid

    def is_sealed(self, position: int) -> bool:
        return 0 <= int(position) <= self._last_position

    def close(self) -> None:
        if not self._closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "ScoreLedger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class OnDemandLabelReader:
    """Decode only a requested arm's label after its matching score seal.

    JSON array boundaries and top-level normal-class metadata are indexed at
    construction. Outcome values are decoded only for the selected queried
    record. R7's fixed query allowlist is intentionally not used.
    """

    def __init__(self, path: Path, category: str, expected_sha256: str,
                 events: Sequence[EventIdentity], arm_id: str, ledger: ScoreLedger):
        if arm_id not in ARM_IDS:
            raise E6ClosedLoopError("label reader arm identity is not registered")
        self.path = Path(path)
        self.category = str(category)
        self.source_sha256 = str(expected_sha256)
        self.events = tuple(events)
        self.arm_id = arm_id
        self.ledger = ledger
        self.decoded_positions: list[int] = []
        self._decoded: set[int] = set()
        self._data_file = self.path.open("rb")
        try:
            if _sha256_file(self.path) != expected_sha256:
                raise E6ClosedLoopError("TEST JSON fingerprint changed after preflight")
            from experiments.calibration import r7_shuffled_correction_4class as r7
            self._r7 = r7
            self._mapping = mmap.mmap(self._data_file.fileno(), 0, access=mmap.ACCESS_READ)
            self._bounds = r7._test_array_boundaries(self._mapping)
            if len(self._bounds) != len(self.events):
                raise E6ClosedLoopError("TEST JSON record count differs from frozen event identities")
            top_fields = r7._object_fields(self._mapping, 0, len(self._mapping))
            meta_bounds = top_fields.get("meta")
            if meta_bounds is None:
                raise E6ClosedLoopError("TEST JSON lacks label-blind normal-class metadata")
            metadata = r7._decode_field(self._mapping, meta_bounds)
            self._normal_class = metadata.get("normal_class") if isinstance(metadata, dict) else None
            if not isinstance(self._normal_class, str) or not self._normal_class:
                raise E6ClosedLoopError("TEST JSON normal-class metadata is malformed")
        except BaseException:
            self.close()
            raise

    def label_for(self, event: EventIdentity, receipt: SealedQueryReceipt) -> int:
        if event.position in self._decoded:
            raise E6ClosedLoopError("duplicate label disclosure for this arm/event")
        if not self.ledger.authorize_receipt(receipt, self.arm_id, event):
            raise E6ClosedLoopError("label request lacks an authorized durable query seal")
        if event.position >= len(self.events) or self.events[event.position] != event:
            raise E6ClosedLoopError("label request event differs from frozen event identity")
        if not self.ledger.is_sealed(event.position):
            raise E6ClosedLoopError("label request preceded score-ledger fsync")
        if not 0 <= event.original_position < len(self._bounds):
            raise E6ClosedLoopError("event maps outside the frozen TEST JSON array")
        start, stop = self._bounds[event.original_position]
        fields = self._r7._object_fields(self._mapping, start, stop)
        image_bounds = fields.get("image_path")
        if image_bounds is None or "anomaly_class" not in fields:
            raise E6ClosedLoopError("selected TEST record lacks image or outcome fields")
        image_path = self._r7._decode_field(self._mapping, image_bounds)
        if not isinstance(image_path, str):
            raise E6ClosedLoopError("selected TEST image_path is malformed")
        from experiments.risk_safe.preflight import parse_image_name
        parsed = parse_image_name(image_path, expected_class=self.category)
        relative_id = str(parsed.rel_path).replace("\\", "/")
        image_digest = self._r7._identity_hash(relative_id)
        group_digest = hashlib.sha256((
            "official-sample-group\0" + str(parsed.official_sample_key)).encode("utf-8")).hexdigest()
        if (relative_id != event.relative_id or image_digest != event.image_identity_sha256 or
                int(parsed.camera_id) != int(event.camera_id) or
                str(parsed.timestamp) != event.timestamp or
                group_digest != event.source_group_sha256):
            raise E6ClosedLoopError("selected outcome record does not match the sealed event identity")
        from experiments.calibration.stage1a_p12_label_provider import _decode_selected_label
        try:
            label = int(_decode_selected_label(self._mapping, start, stop, self._normal_class))
        except Exception as exc:
            raise E6ClosedLoopError("selected queried anomaly_class could not be decoded") from exc
        if label not in (0, 1):
            raise E6ClosedLoopError("selected outcome is not a binary normal/defect value")
        self._decoded.add(event.position)
        self.decoded_positions.append(event.position)
        return label

    def close(self) -> None:
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            mapping.close()
            self._mapping = None
        data_file = getattr(self, "_data_file", None)
        if data_file is not None:
            data_file.close()
            self._data_file = None

    def __enter__(self) -> "OnDemandLabelReader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class PrivateFeedbackSink:
    """Separate append-only per-arm queried feedback; never part of score JSONL."""

    def __init__(self, path: Path, arm_id: str):
        if arm_id not in ARM_IDS:
            raise E6ClosedLoopError("private feedback sink arm is not registered")
        self.arm_id = arm_id
        self.path = Path(path)
        _reject_reparse_ancestors(self.path)
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError(f"private feedback file already exists: {self.path}")
        self._stream = self.path.open("xb")

    def write(self, event: EventIdentity, label: int, update: FeedbackUpdate) -> None:
        if label not in (0, 1):
            raise E6ClosedLoopError("private feedback label must be binary")
        row = {
            "schema": "e6.bea_v2.private_feedback.v1", "arm_id": self.arm_id,
            "position": event.position, "original_json_position": event.original_position,
            "event_id_sha256": event.event_id_sha256,
            "image_identity_sha256": event.image_identity_sha256,
            "source_group_sha256": event.source_group_sha256,
            "label": int(label), "update": dict(update.diagnostics),
        }
        payload = _canonical_json(row)
        self._stream.write(payload)
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()


@dataclass
class ArmRuntime:
    arm_id: str
    engine: ArmEngine
    controller: QueryController
    coverage: NormalCoverageMemory
    label_reader: Any
    private_feedback: Any = None
    queried_groups: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class StageASeal:
    category: str
    event_count: int
    score_log_path: str
    score_log_sha256: str
    private_feedback_sha256: Mapping[str, str]
    arm_query_counts: Mapping[str, int]
    arm_controller_diagnostics: Mapping[str, Mapping[str, Any]]
    complete: bool
    manifest_path: str | None = None
    manifest_sha256: str | None = None
    seal_path: str | None = None
    seal_sha256: str | None = None


class StageAClosedLoop:
    """Seven-arm prequential engine; query decisions never share identities."""

    def __init__(self, *, category: str, events: Sequence[EventIdentity],
                 feature_encoder: Callable[[EventIdentity], EncodedEvent],
                 arms: Mapping[str, ArmRuntime], ledger: ScoreLedger,
                 synthetic_test_only: bool = False):
        self.category = str(category)
        if self.category not in E6_REGISTERED_CATEGORIES:
            raise E6ClosedLoopError("Stage A category is outside the registered development set")
        self.events = tuple(events)
        self.feature_encoder = feature_encoder
        self.arms = dict(arms)
        self.ledger = ledger
        if not isinstance(synthetic_test_only, bool):
            raise E6ClosedLoopError("synthetic_test_only must be an explicit boolean")
        self.synthetic_test_only = synthetic_test_only
        if not synthetic_test_only:
            if not isinstance(self.ledger, ScoreLedger):
                raise E6ClosedLoopError("real Stage A requires the append-only fsynced score ledger")
            repo_root = Path(__file__).resolve().parents[2]
            output_parent = (repo_root / "outputs" / "calibration").resolve(strict=True)
            run_dir = self.ledger.path.parent.resolve(strict=True)
            if (run_dir.parent != output_parent or
                    not re.fullmatch(r"e6_bea_v2_[A-Za-z0-9_-]+", run_dir.name) or
                    self.ledger.path.name != "sealed_scores.jsonl"):
                raise E6ClosedLoopError("real Stage A requires a unique e6_bea_v2_ output namespace")
            namespace_path = run_dir / "run_namespace.json"
            _reject_reparse_ancestors(namespace_path)
            try:
                namespace = json.loads(namespace_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise E6ClosedLoopError("real Stage A requires its exclusive namespace identity marker") from exc
            run_id = run_dir.name.removeprefix("e6_bea_v2_")
            if (namespace.get("schema") != "e6.bea_v2.run_namespace.v1" or
                    namespace.get("status_at_creation") != "INCOMPLETE" or
                    namespace.get("run_id") != run_id):
                raise E6ClosedLoopError("real Stage A namespace marker does not match its output directory")
            self.run_id = run_id
            self.namespace_path = namespace_path
            self.namespace_sha256 = _sha256_file(namespace_path)
            if not isinstance(self.feature_encoder, R7FrozenFeatureEncoder):
                raise E6ClosedLoopError("real Stage A requires the pinned R7 frozen feature encoder")
            if not self.feature_encoder.verified_model_identity:
                raise E6ClosedLoopError("real Stage A requires a verified frozen DINO weight fingerprint")
            if any(not isinstance(arm.engine, R7NativeArmEngine) for arm in self.arms.values()):
                raise E6ClosedLoopError("real Stage A requires seven independently initialized R7 native engines")
            input_identities = [getattr(arm.engine, "input_identity", None)
                                for arm in self.arms.values()]
            if (any(not isinstance(identity, Mapping) for identity in input_identities) or
                    len({_canonical_json(identity) for identity in input_identities}) != 1):
                raise E6ClosedLoopError("all native arms must bind the same verified input identities")
            config_identities = [
                _canonical_json({
                    "p0": arm.engine.native_scorer_config,
                    "r7_profile": arm.engine.native_profile_config,
                    "c1": arm.engine.correction_config_record,
                    "greedy": arm.engine.greedy_settings_record,
                }) for arm in self.arms.values()]
            if len(set(config_identities)) != 1:
                raise E6ClosedLoopError("all native arms must use identical frozen scorer and writer configs")
            validate_data_role_group_disjointness(
                self.events,
                input_identities[0].get("support_source_group_sha256", ()),
                input_identities[0].get("c64_source_group_sha256", ()))
            if any(arm.engine.impl is not self.feature_encoder.impl for arm in self.arms.values()):
                raise E6ClosedLoopError("feature encoder and native scorers must share the pinned implementation")
            if any(not isinstance(arm.label_reader, OnDemandLabelReader)
                   for arm in self.arms.values()):
                raise E6ClosedLoopError("real Stage A requires the restricted per-arm on-demand label reader")
            if any(arm.label_reader.ledger is not self.ledger for arm in self.arms.values()):
                raise E6ClosedLoopError("real per-arm label readers must bind to the active score ledger")
            if any(arm.label_reader.category != self.category or
                   arm.label_reader.events != self.events for arm in self.arms.values()):
                raise E6ClosedLoopError("real per-arm label readers must bind the exact Stage A event identity")
            expected_test_sha = str(self.arms["Q0"].engine.input_identity["category_json_sha256"])
            if any(arm.label_reader.source_sha256 != expected_test_sha
                   for arm in self.arms.values()):
                raise E6ClosedLoopError("on-demand label readers differ from the frozen category JSON digest")
            validate_test_image_projection(
                self.arms["Q0"].label_reader.path, self.category, self.events)
            if any(not Path(event.image_path).is_file() for event in self.events):
                raise E6ClosedLoopError("a frozen TEST image path is missing before Stage A")
            if any(not isinstance(arm.private_feedback, PrivateFeedbackSink)
                   for arm in self.arms.values()):
                raise E6ClosedLoopError("real Stage A requires one separate private feedback file per arm")
            for arm_id, arm in self.arms.items():
                controller = arm.controller
                if (controller.seed != 0 or controller.events_observed != 0 or
                        controller.proposals != 0 or controller.executed_queries != 0 or
                        controller.group_suppressed != 0 or controller.last_observed_position != -1 or
                        controller.executed_flags or controller.utility_history or
                        controller.query_positions or controller.query_gaps):
                    raise E6ClosedLoopError(f"{arm_id}: query controller is not in its registered initial state")
                if ARM_POLICIES[arm_id].adaptive_price and controller.query_cost_runtime != 0.80:
                    raise E6ClosedLoopError(f"{arm_id}: adaptive query price must start at 0.80")
                if arm.coverage.dynamic_size != 0:
                    raise E6ClosedLoopError(f"{arm_id}: Coverage Memory must start from four Supports only")
                engine = arm.engine
                if (engine.state_version != 0 or engine.state_digest != engine.initial_state_sha256 or
                        engine.correction_cb.size() != 0 or engine.current_threshold != engine.theta0 or
                        len(engine.c64_scores) != 64):
                    raise E6ClosedLoopError(f"{arm_id}: native Greedy/C1/B3 state is not at its verified initial state")
        if set(self.arms) != set(ARM_IDS):
            raise E6ClosedLoopError("Stage A requires the seven registered Q0-Q6 arms")
        validate_five_view_groups(self.events)
        if len({id(arm.engine) for arm in self.arms.values()}) != len(ARM_IDS):
            raise E6ClosedLoopError("each query arm must own an independent scorer/memory engine")
        if len({id(arm.controller) for arm in self.arms.values()}) != len(ARM_IDS):
            raise E6ClosedLoopError("each query arm must own independent controller state")
        if len({id(arm.coverage) for arm in self.arms.values()}) != len(ARM_IDS):
            raise E6ClosedLoopError("each query arm must own independent Coverage Memory")
        state_objects = [getattr(arm.engine, "owned_mutable_state_ids", frozenset())
                         for arm in self.arms.values()]
        for left in range(len(state_objects)):
            for right in range(left + 1, len(state_objects)):
                if state_objects[left] & state_objects[right]:
                    raise E6ClosedLoopError("query arms share a mutable scorer or calibration object")
        state_ids = {str(arm.engine.initial_state_sha256) for arm in self.arms.values()}
        if len(state_ids) != 1:
            raise E6ClosedLoopError("seven arms must begin from the same verified initial memory state")
        coverage_ids = {arm.coverage.state_sha256() for arm in self.arms.values()}
        if len(coverage_ids) != 1:
            raise E6ClosedLoopError("seven arms must begin from identical support-only coverage state")
        for arm_id, arm in self.arms.items():
            if arm.arm_id != arm_id or arm.controller.arm_id != arm_id:
                raise E6ClosedLoopError("arm registry, runtime, and controller identities differ")
            if arm.label_reader.arm_id != arm_id:
                raise E6ClosedLoopError("label reader is not isolated to its registered arm")
            if arm.private_feedback is not None and arm.private_feedback.arm_id != arm_id:
                raise E6ClosedLoopError("private feedback sink is not isolated to its registered arm")
            if (arm.private_feedback is not None and
                    Path(arm.private_feedback.path).resolve(strict=False) ==
                    self.ledger.path.resolve(strict=False)):
                raise E6ClosedLoopError("queried labels cannot be written into the public score ledger")
        private_sinks = [arm.private_feedback for arm in self.arms.values()
                         if arm.private_feedback is not None]
        if len({id(sink) for sink in private_sinks}) != len(private_sinks):
            raise E6ClosedLoopError("queried feedback sinks must be independent per arm")
        private_paths = [Path(sink.path).resolve(strict=False) for sink in private_sinks]
        if len(set(private_paths)) != len(private_paths):
            raise E6ClosedLoopError("queried feedback files must not be shared across arms")
        if (not synthetic_test_only and any(
                path.parent != self.ledger.path.resolve(strict=False).parent
                for path in private_paths)):
            raise E6ClosedLoopError("all E6 outputs must stay directly inside the unique run directory")
        if not synthetic_test_only:
            expected_names = {"run_namespace.json", "sealed_scores.jsonl"} | {
                f"private_feedback_{arm}.jsonl" for arm in ARM_IDS}
            if {path.name for path in private_paths} != expected_names - {"sealed_scores.jsonl"}:
                raise E6ClosedLoopError("private feedback output filenames do not match arm identities")
            actual_names = {path.name for path in self.ledger.path.parent.iterdir()}
            if actual_names != expected_names:
                raise E6ClosedLoopError("real E6 output namespace contains unaccounted files")
            self.manifest_path = self.ledger.path.parent / "run_manifest.json"
            self.seal_path = self.ledger.path.parent / "stage_a_seal.json"
            self.manifest_sha256 = _write_json_once(
                self.manifest_path, self._manifest_record())
        else:
            self.manifest_path = None
            self.seal_path = None
            self.manifest_sha256 = None

    def _manifest_record(self) -> dict[str, Any]:
        engine = self.arms["Q0"].engine
        identities = [{
            "position": event.position,
            "original_json_position": event.original_position,
            "event_id_sha256": event.event_id_sha256,
            "image_identity_sha256": event.image_identity_sha256,
            "source_group_sha256": event.source_group_sha256,
            "camera_id": event.camera_id,
        } for event in self.events]
        source_identity = dict(engine.execution_identity)
        return {
            "schema": "e6.bea_v2.run_manifest.v1",
            "run_id": getattr(self, "run_id", "synthetic-test-only"),
            "namespace_record_sha256": getattr(self, "namespace_sha256", None),
            "category": self.category,
            "registered_categories": list(E6_REGISTERED_CATEGORIES),
            "status_at_creation": "INCOMPLETE",
            "dataset_protocol": "Real-IAD development; Regime B; five camera views per Source Group",
            "stream_shuffle": {"source": "R7 deterministic_group_shuffle", "seed": 0,
                               "group_order": "R7 deterministic group shuffle"},
            "stream_event_count": len(self.events),
            "stream_identity_sha256": hashlib.sha256(_canonical_json(identities)).hexdigest(),
            "source_identity": source_identity,
            "model_state_sha256": self.feature_encoder.runtime_info.get(
                "loaded_model_state_sha256"),
            "runtime_profile": self.feature_encoder.runtime_info.get("runtime_profile"),
            "native_scoring_configuration": {
                "p0": dict(engine.native_scorer_config),
                "r7_profile": dict(engine.native_profile_config),
                "greedy_writer": dict(engine.greedy_settings_record),
                "c1_correction": dict(engine.correction_config_record),
            },
            "support_and_c64_identity": dict(getattr(engine, "input_identity", {})),
            "initial_complete_scorer_state_sha256": engine.initial_state_sha256,
            "greedy_capacity": 1024,
            "c1_full_capacity": 1024,
            "coverage_reference": "four deployed support global CLS features; queried verified normals only",
            "query_policies": {
                arm: {"kind": ARM_POLICIES[arm].kind,
                      "objective_terms": list(ARM_POLICIES[arm].objective_terms),
                      "adaptive_price": ARM_POLICIES[arm].adaptive_price}
                for arm in ARM_IDS},
            "output_files": ["run_namespace.json", "run_manifest.json", "sealed_scores.jsonl",
                             *[f"private_feedback_{arm}.jsonl" for arm in ARM_IDS],
                             "stage_a_seal.json"],
            "stage_b_enabled": False,
        }

    def run(self) -> StageASeal:
        """Run Stage A and close all durable writers even after an execution error."""
        try:
            return self._run_body()
        finally:
            self.ledger.close()
            for arm in self.arms.values():
                close_reader = getattr(arm.label_reader, "close", None)
                if callable(close_reader):
                    close_reader()
                if arm.private_feedback is not None:
                    arm.private_feedback.close()

    def _run_body(self) -> StageASeal:
        query_counts = {arm: 0 for arm in ARM_IDS}
        last_event = -1
        for event in self.events:
            if event.position != last_event + 1:
                raise E6ClosedLoopError("Stage A event stream is not contiguous")
            encode_started = time.perf_counter()
            encoded = self.feature_encoder(event)
            if not isinstance(encoded, EncodedEvent):
                raise E6ClosedLoopError("feature encoder must return one EncodedEvent per image")
            encode_elapsed = max(float(encoded.encoding_seconds), time.perf_counter() - encode_started)
            scored: dict[str, tuple[NativePreEventScore, BEAComponents]] = {}
            proposals: dict[str, QueryProposal] = {}
            suppressed: dict[str, bool] = {}
            arm_rows: dict[str, dict[str, Any]] = {}
            score_seconds: dict[str, float] = {}
            for arm_id in ARM_IDS:
                arm = self.arms[arm_id]
                score_started = time.perf_counter()
                pre = arm.engine.score(event, encoded)
                score_seconds[arm_id] = time.perf_counter() - score_started
                novelty_percentile, novelty_info = arm.coverage.novelty_percentile(encoded.global_cls)
                components = compute_bea_components(
                    score=pre.gc_score, threshold=float(arm.engine.current_threshold),
                    c64_scores=pre.c64_scores, novelty_percentile=novelty_percentile,
                    correction_top32=pre.correction_top32,
                    correction_bank_size=pre.correction_bank_size,
                    topk_dN_max=pre.topk_dN_max, topk_dN_mean=pre.topk_dN_mean)
                proposal = arm.controller.propose(
                    position=event.position, event_id_sha256=event.event_id_sha256,
                    components=components)
                group_suppressed = bool(
                    proposal.proposed and event.source_group_sha256 in arm.queried_groups)
                executed = bool(proposal.proposed and not group_suppressed)
                scored[arm_id] = (pre, components)
                proposals[arm_id] = proposal
                suppressed[arm_id] = group_suppressed
                arm_rows[arm_id] = {
                    "gc_score": float(pre.gc_score), "c0_score": float(pre.c0_score),
                    "b3_threshold": float(arm.engine.current_threshold),
                    "c64_sigma": float(components.sigma),
                    "prediction": int(components.pred_defect),
                    "scorer_state_version": int(pre.scorer_state_version),
                    "scorer_state_sha256": pre.scorer_state_sha256,
                    "correction_bank_size": int(pre.correction_bank_size),
                    "query_proposed": bool(proposal.proposed),
                    "query_executed": executed,
                    "group_suppressed": group_suppressed,
                    "suppression_reason": "source_group_already_queried" if group_suppressed else None,
                    "query_reason": proposal.reason,
                    "query_utility": proposal.utility,
                    "query_threshold": proposal.threshold,
                    "query_objective": proposal.objective,
                    "query_cost_runtime": proposal.query_cost,
                    "query_random_u": proposal.random_u,
                    "utility_history_count": proposal.history_count,
                    "encoding_seconds": encode_elapsed,
                    "native_score_seconds": score_seconds[arm_id],
                    "novelty_info": novelty_info,
                    "components": components.as_dict(),
                }
            self.ledger.seal_event(event, arm_rows)

            # No outcome is requested until the joint seven-arm score row is fsynced.
            for arm_id in ARM_IDS:
                arm = self.arms[arm_id]
                pre, _components = scored[arm_id]
                if arm_rows[arm_id]["query_executed"]:
                    arm.queried_groups.add(event.source_group_sha256)
                    receipt = self.ledger.query_receipt(arm_id, event)
                    label = arm.label_reader.label_for(event, receipt)
                    update = arm.engine.apply_feedback(event, encoded, pre, label)
                    if update.label != label:
                        raise E6ClosedLoopError("engine feedback receipt differs from authorized label")
                    if label == 0:
                        arm.coverage.observe_normal(encoded.global_cls)
                    if arm.private_feedback is not None:
                        arm.private_feedback.write(event, label, update)
                    query_counts[arm_id] += 1
                else:
                    arm.engine.observe_unqueried(event, encoded, pre)
                arm.controller.observe_final_action(
                    position=event.position, proposal=proposals[arm_id],
                    executed_query=bool(arm_rows[arm_id]["query_executed"]),
                    group_suppressed=suppressed[arm_id])
            last_event = event.position
        if last_event < 0:
            raise E6ClosedLoopError("Stage A cannot seal an empty trajectory")
        self.ledger.close()
        for arm in self.arms.values():
            close_reader = getattr(arm.label_reader, "close", None)
            if callable(close_reader):
                close_reader()
            if arm.private_feedback is not None:
                arm.private_feedback.close()
        score_log_sha256 = _sha256_file(self.ledger.path)
        private_feedback_sha256 = {
            arm: _sha256_file(self.arms[arm].private_feedback.path)
            for arm in ARM_IDS if self.arms[arm].private_feedback is not None}
        controller_diagnostics = {
            arm: self.arms[arm].controller.diagnostics() for arm in ARM_IDS}
        seal_sha256 = None
        if not self.synthetic_test_only:
            final_arm_state = {
                arm: {
                    "scorer_state_version": int(self.arms[arm].engine.state_version),
                    "scorer_state_sha256": str(self.arms[arm].engine.state_digest),
                    "coverage_state_sha256": self.arms[arm].coverage.state_sha256(),
                    "queried_source_group_count": len(self.arms[arm].queried_groups),
                }
                for arm in ARM_IDS
            }
            seal_record = {
                "schema": "e6.bea_v2.stage_a_seal.v1",
                "status": "COMPLETE",
                "category": self.category,
                "event_count": last_event + 1,
                "manifest_sha256": self.manifest_sha256,
                "score_log": {"file": self.ledger.path.name,
                              "sha256": score_log_sha256},
                "feedback_artifacts": {
                    arm: {"file": Path(self.arms[arm].private_feedback.path).name,
                          "sha256": private_feedback_sha256[arm]}
                    for arm in ARM_IDS
                },
                "arm_query_counts": query_counts,
                "arm_controller_diagnostics": controller_diagnostics,
                "final_arm_state": final_arm_state,
            }
            seal_sha256 = _write_json_once(self.seal_path, seal_record)
        return StageASeal(
            category=self.category, event_count=last_event + 1,
            score_log_path=str(self.ledger.path), score_log_sha256=score_log_sha256,
            private_feedback_sha256=private_feedback_sha256,
            arm_query_counts=query_counts,
            arm_controller_diagnostics=controller_diagnostics,
            complete=True,
            manifest_path=str(self.manifest_path) if self.manifest_path else None,
            manifest_sha256=self.manifest_sha256,
            seal_path=str(self.seal_path) if self.seal_path else None,
            seal_sha256=seal_sha256)


class R7FrozenFeatureEncoder:
    """Encode one identity-view image once and retain both R7 patches and CLS."""

    def __init__(self, *, impl: Any, backbone: Any, feature_fuse: str,
                 r1: Any = None, expected_model_state_sha256: str | None = None,
                 runtime_info: Mapping[str, Any] | None = None):
        self.impl = impl
        self.backbone = backbone
        self.feature_fuse = str(feature_fuse)
        self.runtime_info = dict(runtime_info or {})
        self.verified_model_identity = False
        if expected_model_state_sha256 is not None:
            if r1 is None or not _HEX64.fullmatch(str(expected_model_state_sha256)):
                raise E6ClosedLoopError("model identity requires R1 verifier and a SHA-256 receipt")
            actual = str(r1._model_state_sha256(backbone.model))
            runtime_expected = self.runtime_info.get("expected_model_state_sha256")
            if actual != expected_model_state_sha256 or runtime_expected != expected_model_state_sha256:
                raise E6ClosedLoopError("loaded DINO state differs from the verified R7 model receipt")
            self.verified_model_identity = True

    def __call__(self, event: EventIdentity) -> EncodedEvent:
        import torch

        started = time.perf_counter()
        image = self.impl.load_rgb(event.image_path)
        feature_layers, global_cls = self.impl.encode_pil(
            self.backbone, image, self.feature_fuse)
        if (len(feature_layers) != 1 or tuple(feature_layers[0].shape) != (784, 4096) or
                feature_layers[0].dtype != torch.float32 or
                not bool(torch.isfinite(feature_layers[0]).all().item())):
            raise E6ClosedLoopError("R7 frozen identity-view feature contract must be (784,4096) float32")
        if feature_layers[0].device.type == "cuda":
            torch.cuda.synchronize(feature_layers[0].device)
        cls = torch.as_tensor(global_cls).detach().float().reshape(-1).cpu().numpy().copy()
        if cls.ndim != 1 or cls.size == 0 or not np.isfinite(cls).all():
            raise E6ClosedLoopError("R7 global CLS feature is missing or nonfinite")
        return EncodedEvent(image=image, feature_layers=feature_layers,
                            global_cls=cls, encoding_seconds=time.perf_counter() - started)


def load_r7_frozen_backbone(*, impl: Any, r1: Any, r3: Any,
                            approved_e6_commit: str, repository_root: Path
                            ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load only the pinned local DINO source/weights after E6 review identity.

    This function is never called by CPU tests. It is the future real-run setup
    gate and refuses source downloads, changed checkpoints, or a changed model.
    """
    identity = verify_e6_execution_identity(approved_e6_commit, root=repository_root)
    root = Path(repository_root).resolve(strict=True)
    for module, relative in ((impl, "test_phase3_AIF_simplified_v22.py"),
                             (r1, "experiments/calibration/r1_matched_correction_pilot.py"),
                             (r3, "experiments/calibration/r3_fullstream_2x2.py")):
        module_path = getattr(module, "__file__", None)
        if module_path is None or Path(module_path).resolve(strict=True) != (root / relative).resolve(strict=True):
            raise E6ClosedLoopError(f"DINO setup helper is not loaded from pinned source: {relative}")
    import torch

    model_receipt = dict(r3._get_verified_model_receipt())
    restore_hub, runtime_info = r3._runtime_and_native_setup(torch, impl, model_receipt)
    try:
        backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
    finally:
        restore_hub()
    actual_model_sha256 = str(r1._model_state_sha256(backbone.model))
    expected_model_sha256 = str(model_receipt.get("loaded_model_state_sha256", ""))
    if (not _HEX64.fullmatch(expected_model_sha256) or
            actual_model_sha256 != expected_model_sha256 or
            runtime_info.get("expected_model_state_sha256") != expected_model_sha256):
        raise E6ClosedLoopError("loaded frozen DINO model differs from the verified local receipt")
    runtime_info = dict(runtime_info)
    runtime_info["loaded_model_state_sha256"] = actual_model_sha256
    runtime_info["approved_e6_commit"] = identity["head"]
    return backbone, model_receipt, runtime_info


class R7NativeArmEngine:
    """Per-arm R7 implementation adapter using only verified native helpers.

    The caller must construct a separate native bank, Greedy writer, correction
    codebook, C64 cache, and B3 controller for each arm from the same immutable
    support/C64 receipts. This adapter does not load inputs or a model.
    """

    def __init__(self, *, impl: Any, bank: Any, r1: Any, r7: Any, exp2a: Any,
                 cfg: Any, writer: Any, correction_cfg: Any, correction_cb: Any,
                 source_groups: dict[int, set[str]], fixed_support_sha256: str,
                 theta0: float, sigma0: float, c64_items: Sequence[Mapping[str, Any]],
                 c64_base_cache: Sequence[Any], c64_gc_scores: Sequence[float],
                 c64_group_sha256: str, input_identity: Mapping[str, Any],
                 native_scorer_config: Mapping[str, Any],
                 native_profile_config: Mapping[str, Any],
                 correction_config_record: Mapping[str, Any],
                 greedy_settings_record: Mapping[str, Any],
                 b3_controller: Any,
                 approved_e6_commit: str, repository_root: Path):
        self.execution_identity = verify_e6_execution_identity(
            approved_e6_commit, root=repository_root)
        for module, relative in (
                (impl, "vmb_visionad_new.py"),
                (r1, "experiments/calibration/r1_matched_correction_pilot.py"),
                (r7, "experiments/calibration/r7_shuffled_correction_4class.py"),
                (exp2a, "experiments/calibration/r2_exp2a_c1_strength.py")):
            module_path = getattr(module, "__file__", None)
            expected_path = (Path(repository_root).resolve(strict=True) / relative).resolve(strict=True)
            if module_path is None or Path(module_path).resolve(strict=True) != expected_path:
                raise E6ClosedLoopError(f"native helper module is not loaded from pinned source: {relative}")
        self.impl, self.bank, self.r1, self.r7, self.exp2a = impl, bank, r1, r7, exp2a
        self.cfg, self.writer = cfg, writer
        self.correction_cfg, self.correction_cb = correction_cfg, correction_cb
        self.source_groups = source_groups
        self.fixed_support_sha256 = fixed_support_sha256
        self.theta0, self.sigma0 = float(theta0), float(sigma0)
        self.c64_items = tuple(c64_items)
        self.c64_base_cache = list(c64_base_cache)
        self.c64_scores = [float(value) for value in c64_gc_scores]
        self.c64_group_sha256 = str(c64_group_sha256)
        self.input_identity = dict(input_identity)
        self.native_scorer_config = dict(native_scorer_config)
        self.native_profile_config = dict(native_profile_config)
        self.correction_config_record = dict(correction_config_record)
        self.greedy_settings_record = dict(greedy_settings_record)
        self.b3_controller = b3_controller
        if len(self.c64_items) != 64 or len(self.c64_scores) != 64:
            raise E6ClosedLoopError("R7 adapter requires the 64 registered C64 commissioning items")
        if len(self.c64_base_cache) != 64:
            raise E6ClosedLoopError("R7 adapter requires the matching 64-item native base cache")
        if (int(getattr(self.correction_cb, "K", -1)) != 1024 or
                int(getattr(self.bank.corr_cfg, "K_correction", -1)) != 1024 or
                int(self.correction_cb.size()) != 0 or self.source_groups):
            raise E6ClosedLoopError("R7 E6 correction codebook must have capacity K=1024")
        if getattr(self.writer, "name", None) != "greedy_coreset":
            raise E6ClosedLoopError("R7 E6 normal writer must use the frozen greedy_coreset policy")
        from experiments.risk_safe.native_bridge import dynamic_codebooks
        greedy_codebooks = dynamic_codebooks(self.bank)
        if (not greedy_codebooks or any(int(codebook.K) != 1024 or
                                         int(codebook.size()) != 0
                                         for codebook in greedy_codebooks)):
            raise E6ClosedLoopError("R7 E6 Greedy normal-memory layers must each have capacity K=1024")
        if (self.fixed_support_sha256 != self.input_identity.get("fixed_bank_sha256") or
                self.c64_group_sha256 != self.input_identity.get("c64_group_sha256") or
                hashlib.sha256(_canonical_json(self.c64_scores)).hexdigest() !=
                self.input_identity.get("c64_score_vector_sha256")):
            raise E6ClosedLoopError("R7 E6 arm inputs differ from their fixed-support/C64 receipts")
        if (not math.isclose(max(float(np.std(self.c64_scores, ddof=1)), 0.02), self.sigma0,
                            rel_tol=0.0, abs_tol=1e-12) or
                not math.isclose(float(np.quantile(self.c64_scores, 0.90, method="higher")),
                                 self.theta0, rel_tol=0.0, abs_tol=1e-12)):
            raise E6ClosedLoopError("R7 E6 C64 scores differ from frozen theta0/sigma0 receipts")
        self.state_digest = self._digests()["GC"]
        self.state_version = int(getattr(b3_controller, "evidence_version", 0))
        self.r7._initialize_b3(
            self.b3_controller, self.c64_scores, self.c64_group_sha256,
            self.state_digest, self.theta0)
        self.current_threshold = float(self.b3_controller.theta)
        if self.b3_controller.evidence_version != 0:
            raise E6ClosedLoopError("R7 initial C64 calibration did not establish state version zero")
        self.state_version = 0
        self.initial_state_sha256 = self.state_digest
        self.owned_mutable_state_ids = frozenset({
            id(self.bank), id(self.writer), id(self.correction_cb),
            id(self.source_groups), id(self.b3_controller),
            id(self.c64_base_cache), id(self.c64_scores)})

    def _digests(self) -> dict[str, str]:
        return self.r7._state_digests(
            self.bank, self.source_groups, self.fixed_support_sha256, self.r1)

    def score(self, event: EventIdentity, encoded: EncodedEvent) -> NativePreEventScore:
        import torch

        if (self.b3_controller.evidence_version != self.state_version or
                getattr(self.b3_controller, "_r7_state_digest", None) != self.state_digest):
            raise E6ClosedLoopError("R7 arm B3 is stale for its complete scorer state")
        features = [value if isinstance(value, torch.Tensor) else
                    torch.as_tensor(value, device=self.bank.device) for value in encoded.feature_layers]
        base_map, top_ids, base_score, summary = self.r1._native_dense_score(
            self.impl, self.bank, encoded.image, features, self.cfg)
        selected = self.impl.l2_normalize(features[0].to(self.bank.device))[
            torch.as_tensor(top_ids, device=self.bank.device, dtype=torch.long)]
        delta, _diagnostics = self.r1._c1_score_delta(
            self.impl, self.bank, selected, self.source_groups,
            event.source_group_sha256, event_position=event.position)
        corrected = self.r1.corrected_dense_map(base_map, top_ids, delta)
        gc_score = float(self.r1.max_score(corrected))
        layer = summary.get("layer0", {})
        try:
            dmax = float(layer["topk_dN_max"])
            dmean = float(layer["topk_dN_mean"])
        except (KeyError, TypeError, ValueError) as exc:
            raise E6ClosedLoopError("R7 native summary lacks Top-32 normal-distance statistics") from exc
        return NativePreEventScore(
            gc_score=gc_score, c0_score=float(base_score), c64_scores=tuple(self.c64_scores),
            correction_top32=tuple(float(value) for value in delta),
            correction_bank_size=int(self.correction_cb.size()),
            topk_dN_max=dmax, topk_dN_mean=dmean,
            scorer_state_version=self.state_version,
            scorer_state_sha256=self.state_digest,
            context={"features": features, "base_map": base_map,
                     "top_ids": np.asarray(top_ids, dtype=np.int64),
                     "candidate_features": selected.detach().float().cpu().numpy().astype(
                         np.float32, copy=True), "base_score": float(base_score)})

    def observe_unqueried(self, event: EventIdentity, encoded: EncodedEvent,
                          score: NativePreEventScore) -> Mapping[str, Any]:
        return self._common_writer(event, score, queried=False, label=None)

    def apply_feedback(self, event: EventIdentity, encoded: EncodedEvent,
                       score: NativePreEventScore, label: int) -> FeedbackUpdate:
        if label not in (0, 1):
            raise E6ClosedLoopError("R7 adapter accepts only verified binary queried feedback")
        before = self._digests()
        if score.scorer_state_sha256 != before["GC"] or score.scorer_state_version != self.state_version:
            raise E6ClosedLoopError("feedback arrived after this arm's pre-event state changed")
        c0_prediction = int(score.c0_score >= self.theta0)
        outcome = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[
            (label, c0_prediction)]
        sign = 1 if label == 1 else -1
        context = score.context
        _result, _layer, correction_trace = self.exp2a._capture_update(
            self.impl, self.bank, self.r1, "GC", self.correction_cfg,
            self.correction_cb, context["features"], context["top_ids"], sign,
            score.c0_score, self.theta0, outcome, event.position,
            self.source_groups, event.source_group_sha256)
        writer_receipt = self._common_writer(event, score, queried=True, label=label)
        after = self._digests()
        c1_changed = before["GC"] != after["GC"]
        greedy_changed = before["G"] != after["G"]
        if c1_changed or greedy_changed:
            previous_version = self.state_version
            new_digest = after["GC"]
            if greedy_changed:
                self.c64_scores, self.c64_base_cache, _ = self.r7._score_c64_base(
                    self.impl, self.bank, self.c64_items, self.cfg, self.r1,
                    "e6_per_arm")
            self.c64_scores, _ = self.r7._score_c64_corrected(
                self.impl, self.bank, self.c64_items, self.c64_base_cache,
                self.cfg, self.r1, self.source_groups, "e6_per_arm",
                event_position=event.position + 1)
            self.state_version += 1
            transition = self.r7._refresh_b3(
                self.b3_controller, self.c64_scores, self.c64_group_sha256,
                self.state_digest, new_digest, previous_version)
            if transition is None:
                raise E6ClosedLoopError("R7 B3 refresh failed to follow a changed complete scorer state")
            self.state_digest = new_digest
            self.current_threshold = float(self.b3_controller.theta)
        else:
            transition = None
        return FeedbackUpdate(
            label=label, c1_changed=c1_changed, greedy_changed=greedy_changed,
            scorer_state_version=self.state_version,
            scorer_state_sha256=self.state_digest,
            diagnostics={"frozen_c0_outcome": outcome, "c1_sign": sign,
                         "correction_trace": correction_trace,
                         "greedy_writer": writer_receipt,
                         "b3_transition_for_next_event": transition})

    def _common_writer(self, event: EventIdentity, score: NativePreEventScore,
                       *, queried: bool, label: int | None) -> Mapping[str, Any]:
        context = score.context
        candidate_features = context["candidate_features"]
        audit_before = self.writer.audit()

        def update(*, position: int, feature_layers: Sequence[Any], candidate_ids: Any,
                   score: float, theta: float, sigma: float) -> Mapping[str, Any]:
            before = self.r1._memory_content_sha256(self.bank)
            self.writer.update(self.impl, list(feature_layers), {0: candidate_ids.tolist()},
                               position, score=score, theta=theta, sigma=sigma)
            changed = before != self.r1._memory_content_sha256(self.bank)
            after_audit = self.writer.audit()
            return {"accepted_patches_delta": int(
                        after_audit["accepted_patches"] - audit_before["accepted_patches"]),
                    "skipped_patches_delta": int(
                        after_audit["skipped_patches"] - audit_before["skipped_patches"]),
                    "_memory_content_changed": changed}

        receipt = self.r1.common_c0_greedy_writer_action(
            position=event.position, c0_score=score.c0_score,
            theta=self.theta0, sigma=self.sigma0,
            c0_prediction=int(score.c0_score >= self.theta0),
            candidate_ids=context["top_ids"], candidate_features=candidate_features,
            feature_layers=context["features"], queried=queried,
            disclosed_label=label, eligibility=self.impl.select_informative_normal_write,
            update=update)
        receipt.pop("_memory_content_changed", None)
        if not queried and receipt.get("action") != "no_normal_memory_write":
            raise E6ClosedLoopError("unqueried event entered the native Greedy writer")
        if queried and label == 1 and receipt.get("action") != "verified_defect_no_normal_write":
            raise E6ClosedLoopError("queried defect entered the native normal memory")
        return receipt


def build_r7_native_arm_engines(*, category: str, info: Mapping[str, Any],
                                impl: Any, p0: Any, r1: Any, r3: Any, r6: Any,
                                r7: Any, exp2a: Any, native_module: Any,
                                cfg: Any, support_features: Mapping[str, Any],
                                support_paths: Sequence[str],
                                c64_items: Sequence[Mapping[str, Any]],
                                approved_e6_commit: str, repository_root: Path
                                ) -> dict[str, R7NativeArmEngine]:
    """Create seven independent same-support R7 Greedy+C1+B3 states.

    Support and C64 features are immutable shared inputs and must already be
    encoded from their verified identities. This factory never reads TEST
    outcomes and never constructs or consults a fixed query tape.
    """
    if category not in E6_REGISTERED_CATEGORIES:
        raise E6ClosedLoopError("native E6 arm factory received an unregistered category")
    execution_identity = verify_e6_execution_identity(
        approved_e6_commit, root=repository_root)
    root = Path(repository_root).resolve(strict=True)
    modules = (
        (impl, "test_phase3_AIF_simplified_v22.py"),
        (p0, "experiments/p0_protocol_audit.py"),
        (r1, "experiments/calibration/r1_matched_correction_pilot.py"),
        (r3, "experiments/calibration/r3_fullstream_2x2.py"),
        (r6, "experiments/calibration/r6_capacity1024_audiojack_runner.py"),
        (r7, "experiments/calibration/r7_shuffled_correction_4class.py"),
        (exp2a, "experiments/calibration/r2_exp2a_c1_strength.py"),
        (native_module, "experiments/calibration/native.py"),
    )
    for module, relative in modules:
        module_path = getattr(module, "__file__", None)
        if module_path is None or Path(module_path).resolve(strict=True) != (root / relative).resolve(strict=True):
            raise E6ClosedLoopError(f"native E6 setup module is not loaded from pinned source: {relative}")
    expected_support_paths = tuple(str(value) for value in info["contract"]["support_paths"])
    if (len(expected_support_paths) != 4 or len(set(expected_support_paths)) != 4 or
            tuple(str(value) for value in support_paths) != expected_support_paths or
            set(str(value) for value in support_features) != set(expected_support_paths)):
        raise E6ClosedLoopError("support feature inputs differ from the authenticated four-support identity")
    if len(c64_items) != 64:
        raise E6ClosedLoopError("E6 native arm factory requires 64 frozen C64 items")
    support_groups_raw = tuple(info["frozen_info"]["supports"]["0"]["groups"])
    support_images_raw = tuple(info["frozen_info"]["supports"]["0"]["images"])
    support_group_sha256 = tuple(sorted({hashlib.sha256((
        "official-sample-group\0" + str(group)).encode("utf-8")).hexdigest()
        for group in support_groups_raw}))
    support_group_order = tuple(hashlib.sha256((
        "official-sample-group\0" + str(group)).encode("utf-8")).hexdigest()
        for group in support_groups_raw)
    c64_groups = tuple(str(value) for value in info["c64_groups"])
    if (len(support_group_sha256) != 4 or len(support_group_order) != 4 or
            len(set(support_group_order)) != 4 or len(support_images_raw) != 4 or
            len(c64_groups) != 64 or len(set(c64_groups)) != 64):
        raise E6ClosedLoopError("authenticated Support/C64 group receipts have invalid cardinality")
    for index, item in enumerate(c64_items):
        if item.get("source_group_sha256") != c64_groups[index]:
            raise E6ClosedLoopError("shared C64 features do not match the frozen Source Group order")
    c64_info = info["frozen_info"]["c64"]
    c64_images_raw = tuple(str(value) for value in c64_info["images"])
    if len(c64_images_raw) != 64 or len(set(c64_images_raw)) != 64:
        raise E6ClosedLoopError("authenticated C64 image identity list has invalid cardinality")
    c64_group_sha = str(c64_info["group_sha256"])
    if c64_group_sha != str(info["contract"]["c64_group_sha256"]):
        raise E6ClosedLoopError("C64 group digest differs from the frozen category contract")
    category_json_sha256 = str(info["contract"]["category_json_sha256"])
    support_image_sha256 = str(info["support_image_sha256"])
    support_group_digest = str(info["support_group_sha256"])
    c64_image_sha256 = str(c64_info["image_sha256"])
    c64_score_archive_sha256 = str(info["c64_score_archive_sha256"])
    for name, value in (
            ("TEST category JSON", category_json_sha256),
            ("support image", support_image_sha256),
            ("support group", support_group_digest),
            ("C64 image", c64_image_sha256),
            ("C64 group", c64_group_sha),
            ("C64 score archive", c64_score_archive_sha256)):
        if not _HEX64.fullmatch(value):
            raise E6ClosedLoopError(f"{name} identity is not a SHA-256 digest")
    input_identity = {
        "category_json_sha256": category_json_sha256,
        "support_seed": 0,
        "support_count": len(expected_support_paths),
        "support_image_sha256": support_image_sha256,
        "support_ordered_image_list_sha256": hashlib.sha256(
            _canonical_json([str(value) for value in support_images_raw])).hexdigest(),
        "support_group_sha256": support_group_digest,
        "support_source_group_sha256": list(support_group_sha256),
        "support_ordered_group_list_sha256": hashlib.sha256(
            _canonical_json(list(support_group_order))).hexdigest(),
        "fixed_bank_sha256": str(info["fixed_support_sha256_expected"]),
        "c64_count": len(c64_items),
        "c64_image_sha256": c64_image_sha256,
        "c64_ordered_image_list_sha256": hashlib.sha256(
            _canonical_json(list(c64_images_raw))).hexdigest(),
        "c64_group_sha256": c64_group_sha,
        "c64_source_group_sha256": list(c64_groups),
        "c64_ordered_group_list_sha256": hashlib.sha256(_canonical_json(list(c64_groups))).hexdigest(),
        "c64_score_archive_sha256": c64_score_archive_sha256,
        "c64_score_vector_sha256": hashlib.sha256(_canonical_json(
            [float(value) for value in info["c64_scores"]])).hexdigest(),
    }
    if not _HEX64.fullmatch(input_identity["fixed_bank_sha256"]):
        raise E6ClosedLoopError("fixed-support bank identity is not a SHA-256 digest")
    if (set(support_group_sha256) & set(c64_groups) or
            set(support_group_sha256) & set(str(event.get("opaque_group", ""))
                                            for event in info["contract"]["stream_rows"]) or
            set(c64_groups) & set(str(event.get("opaque_group", ""))
                                  for event in info["contract"]["stream_rows"])):
        raise E6ClosedLoopError("authenticated Support, C64, and TEST groups overlap")

    from types import SimpleNamespace
    from experiments.calibration import controllers
    from experiments.calibration import profiles
    from experiments.calibration.controllers import StateConsistentQuantile
    from experiments.risk_safe.native_bridge import dynamic_codebooks
    from experiments.v02_memory import MemoryArm
    from experiments import v02_memory
    for module, relative in (
            (controllers, "experiments/calibration/controllers.py"),
            (profiles, "experiments/calibration/profiles.py"),
            (v02_memory, "experiments/v02_memory.py")):
        module_path = getattr(module, "__file__", None)
        if module_path is None or Path(module_path).resolve(strict=True) != (root / relative).resolve(strict=True):
            raise E6ClosedLoopError(f"native E6 memory/calibration source is not pinned: {relative}")

    expected_cfg, native_profile = r3._native_config(p0, profiles, r1)
    scorer_config_record = _config_record(cfg, "P0 native scorer")
    if scorer_config_record != _config_record(expected_cfg, "registered P0 native scorer"):
        raise E6ClosedLoopError("native scorer config differs from the frozen R7 configuration")
    native_profile_record = _config_record(native_profile, "R7 native scorer profile")

    theta0 = float(info["theta_c64_q90"])
    sigma0 = float(info["sigma_c64_floor_002"])
    expected_fixed_sha = str(info["fixed_support_sha256_expected"])
    engines: dict[str, R7NativeArmEngine] = {}
    settings = SimpleNamespace(flat=1024, stm=1024, ltm=0, buffer=0,
                               min_events=2, promote_every=16,
                               cap_bytes=2 * 1024**3)
    for arm_id in ARM_IDS:
        impl.set_seed(0)
        bank = p0.build_vmb(impl, support_features, list(support_paths), cfg)
        r3._validate_correction_config(bank)
        correction_cfg, correction_banks = r6._configure_r6_correction_capacity(bank)
        correction_cb = bank.corr_cb["p0"]["id"][0]
        if (len(correction_banks) != 1 or correction_cb is not correction_banks[0] or
                correction_cb.size() != 0 or correction_cb.K != 1024 or
                int(correction_cfg.K_correction) != 1024):
            raise E6ClosedLoopError(f"{arm_id}: native C1 bank did not start empty at K=1024")
        writer = MemoryArm("greedy_coreset", bank, settings, seed=0)
        correction_config_record = _config_record(correction_cfg, "R7 C1 correction")
        greedy_settings_record = _config_record(settings, "R7 Greedy writer")
        if any(codebook.size() != 0 or codebook.K != 1024
               for codebook in dynamic_codebooks(bank)):
            raise E6ClosedLoopError(f"{arm_id}: native Greedy memory did not start empty at K=1024")
        bank.corr_cfg.enabled = False
        bank.use_defect_bank = False
        fixed_sha = native_module._fixed_bank_digest(bank)
        if fixed_sha != expected_fixed_sha:
            raise E6ClosedLoopError(f"{arm_id}: structured fixed-support identity changed")

        base_scores, base_cache, _ = r7._score_c64_base(
            impl, bank, c64_items, cfg, r1, category)
        expected_scores = [float(value) for value in info["c64_scores"]]
        if len(base_scores) != 64 or len(expected_scores) != 64 or any(
               not r3._score_tolerance(actual, expected)
               for actual, expected in zip(base_scores, expected_scores)):
            raise E6ClosedLoopError(f"{arm_id}: native frozen C64 base scores changed")
        source_groups: dict[int, set[str]] = {}
        gc_scores, _ = r7._score_c64_corrected(
            impl, bank, c64_items, base_cache, cfg, r1, source_groups,
            category, event_position=0)
        if any(not r3._score_tolerance(actual, base)
               for actual, base in zip(gc_scores, base_scores)):
            raise E6ClosedLoopError(f"{arm_id}: empty C1 GC C64 scores differ from G")
        measured_sigma0 = max(float(np.std(expected_scores, ddof=1)), 0.02)
        if not math.isclose(measured_sigma0, sigma0, rel_tol=0.0, abs_tol=1e-12):
            raise E6ClosedLoopError(f"{arm_id}: frozen C0 writer sigma receipt is inconsistent")
        if not math.isclose(float(np.quantile(expected_scores, 0.90, method="higher")),
                            theta0, rel_tol=0.0, abs_tol=1e-12):
            raise E6ClosedLoopError(f"{arm_id}: frozen C0 B3 threshold receipt is inconsistent")
        b3 = StateConsistentQuantile(theta0, delta=0.01, min_normals=16,
                                     name=f"E6-{arm_id}-GC-B3")
        engines[arm_id] = R7NativeArmEngine(
            impl=impl, bank=bank, r1=r1, r7=r7, exp2a=exp2a, cfg=cfg,
            writer=writer, correction_cfg=correction_cfg,
            correction_cb=correction_cb, source_groups=source_groups,
            fixed_support_sha256=fixed_sha, theta0=theta0, sigma0=sigma0,
            c64_items=c64_items, c64_base_cache=base_cache,
            c64_gc_scores=gc_scores, c64_group_sha256=c64_group_sha,
            input_identity=input_identity,
            native_scorer_config=scorer_config_record,
            native_profile_config=native_profile_record,
            correction_config_record=correction_config_record,
            greedy_settings_record=greedy_settings_record,
            b3_controller=b3, approved_e6_commit=approved_e6_commit,
            repository_root=root)
    if len({engine.initial_state_sha256 for engine in engines.values()}) != 1:
        raise E6ClosedLoopError("seven independently built E6 native arms do not share initial state")
    if len({id(engine.bank) for engine in engines.values()}) != len(ARM_IDS):
        raise E6ClosedLoopError("E6 native arm factory reused a mutable native bank")
    return engines


def _read_sealed_score_log(path: Path, seal: StageASeal) -> list[dict[str, Any]]:
    if not seal.complete or not _HEX64.fullmatch(seal.score_log_sha256):
        raise E6ClosedLoopError(f"{seal.category}: Stage A seal is incomplete or malformed")
    score_path = Path(path)
    _reject_reparse_ancestors(score_path)
    if _sha256_file(score_path) != seal.score_log_sha256:
        raise E6ClosedLoopError(f"{seal.category}: Stage A score log does not match its seal digest")
    if set(seal.private_feedback_sha256) != set(ARM_IDS):
        raise E6ClosedLoopError(f"{seal.category}: Stage A private feedback seals are incomplete")
    for arm in ARM_IDS:
        private_path = score_path.parent / f"private_feedback_{arm}.jsonl"
        _reject_reparse_ancestors(private_path)
        expected_private_sha = seal.private_feedback_sha256[arm]
        if (not _HEX64.fullmatch(str(expected_private_sha)) or
                not private_path.is_file() or _sha256_file(private_path) != expected_private_sha):
            raise E6ClosedLoopError(f"{seal.category}: Stage A private feedback seal failed for {arm}")
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            _reject_public_label_fields(row)
            if row.get("position") != len(rows) or set(row.get("arm_rows", {})) != set(ARM_IDS):
                raise E6ClosedLoopError(f"{seal.category}: score log positions/arms are inconsistent")
            rows.append(row)
    if len(rows) != seal.event_count:
        raise E6ClosedLoopError(f"{seal.category}: sealed event count does not match the score log")
    if set(seal.arm_query_counts) != set(ARM_IDS):
        raise E6ClosedLoopError(f"{seal.category}: seal lacks all seven arm query counts")
    if set(seal.arm_controller_diagnostics) != set(ARM_IDS):
        raise E6ClosedLoopError(f"{seal.category}: seal lacks all seven controller diagnostics")
    run_dir = score_path.parent.resolve(strict=True)
    expected_manifest = run_dir / "run_manifest.json"
    expected_seal = run_dir / "stage_a_seal.json"
    if (score_path.name != "sealed_scores.jsonl" or seal.manifest_path is None or
            seal.seal_path is None or seal.manifest_sha256 is None or
            seal.seal_sha256 is None):
        raise E6ClosedLoopError(f"{seal.category}: durable Stage A manifest and seal are required")
    manifest_path = Path(seal.manifest_path)
    seal_path = Path(seal.seal_path)
    _reject_reparse_ancestors(score_path)
    _reject_reparse_ancestors(manifest_path)
    _reject_reparse_ancestors(seal_path)
    if (manifest_path.resolve(strict=True) != expected_manifest or
            seal_path.resolve(strict=True) != expected_seal):
        raise E6ClosedLoopError(f"{seal.category}: Stage A seal paths escaped their run directory")
    if (_sha256_file(manifest_path) != seal.manifest_sha256 or
            not _HEX64.fullmatch(seal.manifest_sha256)):
        raise E6ClosedLoopError(f"{seal.category}: run manifest digest does not match its Stage A seal")
    if (_sha256_file(seal_path) != seal.seal_sha256 or
            not _HEX64.fullmatch(seal.seal_sha256)):
        raise E6ClosedLoopError(f"{seal.category}: durable Stage A seal digest is invalid")
    try:
        manifest_record = json.loads(manifest_path.read_text(encoding="utf-8"))
        seal_record = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E6ClosedLoopError(f"{seal.category}: durable Stage A records are unreadable") from exc
    if (manifest_record.get("schema") != "e6.bea_v2.run_manifest.v1" or
            manifest_record.get("category") != seal.category or
            manifest_record.get("registered_categories") != list(E6_REGISTERED_CATEGORIES) or
            manifest_record.get("stream_event_count") != seal.event_count or
            manifest_record.get("status_at_creation") != "INCOMPLETE" or
            manifest_record.get("stage_b_enabled") is not False):
        raise E6ClosedLoopError(f"{seal.category}: run manifest identity/status is invalid")
    namespace_path = run_dir / "run_namespace.json"
    _reject_reparse_ancestors(namespace_path)
    namespace_sha256 = manifest_record.get("namespace_record_sha256")
    try:
        namespace_record = json.loads(namespace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E6ClosedLoopError(f"{seal.category}: exclusive run namespace marker is unreadable") from exc
    expected_run_id = run_dir.name.removeprefix("e6_bea_v2_")
    expected_outputs = {"run_namespace.json", "run_manifest.json", "sealed_scores.jsonl",
                        "stage_a_seal.json", *[f"private_feedback_{arm}.jsonl" for arm in ARM_IDS]}
    if (not isinstance(namespace_sha256, str) or not _HEX64.fullmatch(namespace_sha256) or
            _sha256_file(namespace_path) != namespace_sha256 or
            namespace_record.get("schema") != "e6.bea_v2.run_namespace.v1" or
            namespace_record.get("status_at_creation") != "INCOMPLETE" or
            namespace_record.get("run_id") != expected_run_id or
            manifest_record.get("run_id") != expected_run_id or
            set(manifest_record.get("output_files", ())) != expected_outputs):
        raise E6ClosedLoopError(f"{seal.category}: run namespace identity is invalid")
    expected_feedback = {
        arm: {"file": f"private_feedback_{arm}.jsonl",
              "sha256": seal.private_feedback_sha256[arm]}
        for arm in ARM_IDS
    }
    if (seal_record.get("schema") != "e6.bea_v2.stage_a_seal.v1" or
            seal_record.get("status") != "COMPLETE" or
            seal_record.get("category") != seal.category or
            seal_record.get("event_count") != seal.event_count or
            seal_record.get("manifest_sha256") != seal.manifest_sha256 or
            seal_record.get("score_log") != {
                "file": "sealed_scores.jsonl", "sha256": seal.score_log_sha256} or
            seal_record.get("feedback_artifacts") != expected_feedback or
            seal_record.get("arm_query_counts") != dict(seal.arm_query_counts) or
            seal_record.get("arm_controller_diagnostics") !=
            dict(seal.arm_controller_diagnostics)):
        raise E6ClosedLoopError(f"{seal.category}: durable Stage A seal fields disagree with its receipt")
    actual_counts = {arm: 0 for arm in ARM_IDS}
    for row in rows:
        for arm in ARM_IDS:
            actual_counts[arm] += int(bool(row["arm_rows"][arm].get("query_executed")))
    if actual_counts != dict(seal.arm_query_counts):
        raise E6ClosedLoopError(f"{seal.category}: sealed query counts disagree with score rows")
    return rows


def stage_b_evaluate(*, stage_a_seals: Mapping[str, StageASeal],
                     expected_categories: Sequence[str],
                     labels_for_category: Callable[[str], Sequence[int]],
                     stage_b_authorized: bool = False,
                     authorization_id: str | None = None) -> dict[str, Any]:
    """Evaluate only after every registered Stage A category/arm seal verifies.

    No online engine/state is accepted by this API, so Stage B outcomes cannot
    flow back into online scoring or updates. The default call is disabled.
    """
    if stage_b_authorized is not True or not authorization_id:
        raise E6ClosedLoopError("Stage B is disabled without separate explicit authorization")
    categories = tuple(expected_categories)
    if (set(categories) != set(E6_REGISTERED_CATEGORIES) or
            len(categories) != len(E6_REGISTERED_CATEGORIES) or
            set(categories) != set(stage_a_seals)):
        raise E6ClosedLoopError("Stage B requires seals for all four registered categories")
    logs: dict[str, list[dict[str, Any]]] = {}
    for category in categories:
        seal = stage_a_seals[category]
        if seal.category != category:
            raise E6ClosedLoopError("Stage A category key and seal identity disagree")
        logs[category] = _read_sealed_score_log(Path(seal.score_log_path), seal)
    # Outcome access begins only after all category logs and seals passed.
    outcomes = {category: np.asarray(labels_for_category(category), dtype=np.int8)
                for category in categories}
    for category, rows in logs.items():
        y = outcomes[category]
        if y.shape != (len(rows),) or not np.isin(y, (0, 1)).all():
            raise E6ClosedLoopError(f"{category}: Stage B outcomes are not one binary value per event")
    per_category: dict[str, dict[str, Any]] = {}
    for category, rows in logs.items():
        y = outcomes[category]
        per_category[category] = {}
        for arm in ARM_IDS:
            score = np.asarray([row["arm_rows"][arm]["gc_score"] for row in rows], dtype=np.float64)
            theta = np.asarray([row["arm_rows"][arm]["b3_threshold"] for row in rows], dtype=np.float64)
            pred = score >= theta
            per_category[category][arm] = _binary_metrics(y, pred.astype(np.int8), score)
            per_category[category][arm]["realised_query_rate"] = float(np.mean([
                bool(row["arm_rows"][arm]["query_executed"]) for row in rows]))
            per_category[category][arm]["query_count"] = int(sum(
                bool(row["arm_rows"][arm]["query_executed"]) for row in rows))
            per_category[category][arm]["worst_256_window_ber"] = _worst_window_ber(y, pred)
    macro: dict[str, Any] = {}
    for arm in ARM_IDS:
        metrics = [per_category[category][arm] for category in categories]
        macro[arm] = {
            "macro_ber": float(np.mean([item["ber"] for item in metrics])),
            "macro_ba": float(np.mean([item["ba"] for item in metrics])),
            "macro_fpr": float(np.mean([item["fpr"] for item in metrics])),
            "macro_fnr": float(np.mean([item["fnr"] for item in metrics])),
            "macro_auroc": float(np.mean([item["auroc"] for item in metrics])),
            "macro_ap": float(np.mean([item["ap"] for item in metrics])),
            "realised_query_rate": float(np.mean([item["realised_query_rate"] for item in metrics])),
        }
    return {"schema": "e6.bea_v2.stage_b_metrics.v1", "authorization_id": authorization_id,
            "primary_metric": "category-equal MacroBER", "per_category": per_category,
            "category_equal_macro": macro}


def _binary_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if y.ndim != 1 or len(y) == 0 or not np.isin(y, (0, 1)).all():
        raise E6ClosedLoopError("binary evaluation requires a nonempty 0/1 outcome vector")
    negatives, positives = y == 0, y == 1
    if not negatives.any() or not positives.any():
        raise E6ClosedLoopError("BER/AUROC/AP require both outcome classes in this category")
    fpr = float(np.mean(pred[negatives] == 1))
    fnr = float(np.mean(pred[positives] == 0))
    return {"fpr": fpr, "fnr": fnr, "ber": (fpr + fnr) / 2.0,
            "ba": 1.0 - (fpr + fnr) / 2.0,
            "auroc": _auroc(y, score), "ap": _average_precision(y, score)}


def _auroc(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    sorted_score = score[order]
    first = 0
    while first < len(order):
        stop = first + 1
        while stop < len(order) and sorted_score[stop] == sorted_score[first]:
            stop += 1
        ranks[order[first:stop]] = (first + 1 + stop) / 2.0
        first = stop
    n_pos = int(np.count_nonzero(y == 1))
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    sorted_y = y[order]
    positives = int(np.count_nonzero(y == 1))
    true_positive = 0
    total = 0.0
    first = 0
    while first < len(order):
        stop = first + 1
        while stop < len(order) and score[order[stop]] == score[order[first]]:
            stop += 1
        group_positive = int(np.count_nonzero(sorted_y[first:stop] == 1))
        true_positive += group_positive
        total += group_positive * (true_positive / stop)
        first = stop
    return float(total / positives)


def _worst_window_ber(y: np.ndarray, pred: np.ndarray, width: int = 256) -> float:
    if len(y) != len(pred) or not len(y):
        raise E6ClosedLoopError("worst-window metric requires paired nonempty vectors")
    best = 0.0
    for start in range(0, len(y), width):
        stop = min(len(y), start + width)
        local_y, local_pred = y[start:stop], pred[start:stop]
        if np.any(local_y == 0) and np.any(local_y == 1):
            fpr = float(np.mean(local_pred[local_y == 0] == 1))
            fnr = float(np.mean(local_pred[local_y == 1] == 0))
            best = max(best, (fpr + fnr) / 2.0)
    return best
