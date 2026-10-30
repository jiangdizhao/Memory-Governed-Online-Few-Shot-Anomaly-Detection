"""Fail-closed primitives for bounded Stage 1-A P1 reconstruction.

This module contains no dataset loader and performs no inference at import
time. An authorization record is validated before a lazy event source, image
provider, or queried-label provider is touched. Real features, labels, scores,
and memory tensors are runtime-private and are never serialized by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence


AUTHORIZATION_SCHEMA = "calibration.stage1a_p1_bounded_authorization.v1"
QUERY_FEEDBACK_AUTHORIZATION_SCHEMA = "calibration.stage1a_queried_feedback_access.v1"
REPLAY_CHECKPOINT_SCHEMA = "calibration.stage1a_p1_replay_checkpoint.v1"
P1_PROTOCOL_SHA256 = "6f1d8cf40e5e7b2d98dcfabe5836b49c6cd57069d3c8377122dba64e3fc2da1c"
P1_CLASSES = ("audiojack", "regulator", "switch", "rolled_strip_base")
MAX_QUERY_FEATURE_ENCODINGS = 459
MAX_STATIC_MAP_CALLS = 2295
MAX_ANCHOR_SCORE_CALLS = 75406
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class P1Blocked(RuntimeError):
    """A missing or out-of-scope P1 authorization blocks data access."""


class P1ParityMismatch(RuntimeError):
    """An exact source/protocol comparison failed; the trajectory must stop."""


class P1ComputeLimitExceeded(RuntimeError):
    """A bounded P1 authorization's explicit compute cap was reached."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _require_sha(value: str, field: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class BoundedAuthorization:
    authorization_id: str
    query_feedback_authorization_id: str
    allowed_classes: tuple[str, ...]
    query_protocol_sha256: str
    max_queried_label_reads: int
    max_feature_encodings: int
    max_static_map_calls: int
    max_anchor_score_calls: int

    @classmethod
    def from_records(cls, record: Mapping[str, Any] | None,
                     query_feedback_record: Mapping[str, Any] | None
                     ) -> "BoundedAuthorization":
        if not isinstance(record, Mapping):
            raise P1Blocked("No researcher-approved bounded P1 authorization record")
        if not isinstance(query_feedback_record, Mapping):
            raise P1Blocked("No separate researcher-approved queried-feedback access decision")
        access = record.get("access")
        bounds = record.get("bounds")
        classes = tuple(record.get("allowed_classes", ()))
        if (record.get("schema") != AUTHORIZATION_SCHEMA or
                record.get("decision") != "APPROVED" or
                record.get("scope") != "STAGE1A_P1_NATIVE_REPLAY_AND_ANCHOR_EVIDENCE" or
                not str(record.get("authorization_id", "")).strip() or
                not str(record.get("approved_by", "")).strip() or
                not str(record.get("approved_at_utc", "")).strip()):
            raise P1Blocked("P1 authorization is absent, incomplete, or for another scope")
        if classes != P1_CLASSES:
            raise P1Blocked("P1 authorization must be limited to the four development classes")
        if record.get("query_protocol_sha256") != P1_PROTOCOL_SHA256:
            raise P1Blocked("P1 authorization query-protocol identity does not match the frozen tape")
        if (not isinstance(access, Mapping) or
                access.get("native_scorer_calls") is not True or
                access.get("unqueried_labels") is not False or
                access.get("confirmation_classes") is not False or
                access.get("masks") is not False):
            raise P1Blocked("P1 authorization must bound queried labels/scoring and forbid other data")
        if not isinstance(bounds, Mapping):
            raise P1Blocked("P1 authorization lacks explicit compute bounds")
        label_classes = tuple(query_feedback_record.get("allowed_classes", ()))
        if (query_feedback_record.get("schema") != QUERY_FEEDBACK_AUTHORIZATION_SCHEMA or
                query_feedback_record.get("decision") != "APPROVED" or
                query_feedback_record.get("scope") != "STAGE1A_PRECOMMITTED_QUERIED_POSITIONS_ONLY" or
                not str(query_feedback_record.get("authorization_id", "")).strip() or
                not str(query_feedback_record.get("approved_by", "")).strip() or
                not str(query_feedback_record.get("approved_at_utc", "")).strip() or
                label_classes != P1_CLASSES or
                query_feedback_record.get("query_protocol_sha256") != P1_PROTOCOL_SHA256 or
                query_feedback_record.get("unqueried_labels") is not False or
                query_feedback_record.get("confirmation_classes") is not False or
                query_feedback_record.get("masks") is not False):
            raise P1Blocked("Separate queried-feedback approval is absent or outside the frozen tape")
        maxima = {
            "max_feature_encodings": MAX_QUERY_FEATURE_ENCODINGS,
            "max_static_map_calls": MAX_STATIC_MAP_CALLS,
            "max_anchor_score_calls": MAX_ANCHOR_SCORE_CALLS,
        }
        checked: dict[str, int] = {}
        for key, hard_max in maxima.items():
            value = bounds.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= hard_max:
                raise P1Blocked(f"P1 authorization has invalid bound: {key}")
            checked[key] = value
        max_labels = query_feedback_record.get("max_queried_label_reads")
        if isinstance(max_labels, bool) or not isinstance(max_labels, int) or not 0 <= max_labels <= MAX_QUERY_FEATURE_ENCODINGS:
            raise P1Blocked("Queried-feedback authorization has an invalid label-read bound")
        return cls(str(record["authorization_id"]),
                   str(query_feedback_record["authorization_id"]), classes,
                   _require_sha(str(record["query_protocol_sha256"]),
                                "query_protocol_sha256"), max_labels, **checked)


_P12_STANDING_CONTEXT_TOKEN = object()


@dataclass
class ComputeBudget:
    authorization: BoundedAuthorization
    queried_label_reads: int = 0
    feature_encodings: int = 0
    static_map_calls: int = 0
    anchor_score_calls: int = 0

    def charge(self, kind: str, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("Compute charge must be nonnegative")
        fields = {
            "queried_label": ("queried_label_reads", self.authorization.max_queried_label_reads),
            "feature_encoding": ("feature_encodings", self.authorization.max_feature_encodings),
            "static_map": ("static_map_calls", self.authorization.max_static_map_calls),
            "anchor_score": ("anchor_score_calls", self.authorization.max_anchor_score_calls),
        }
        if kind not in fields:
            raise ValueError(f"Unknown P1 compute counter: {kind}")
        attribute, maximum = fields[kind]
        new_value = getattr(self, attribute) + amount
        if new_value > maximum:
            raise P1ComputeLimitExceeded(
                f"P1 {kind} bound reached ({new_value}>{maximum}); stop without retry")
        setattr(self, attribute, new_value)


def source_fingerprints(root: Path, relative_paths: Sequence[str]) -> dict[str, Any]:
    """Hash allowlisted UTF-8 source text with newline forms canonicalized to LF."""
    root = root.resolve()
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("Source fingerprint allowlist contains duplicate paths")
    entries: dict[str, str] = {}
    for relative in sorted(relative_paths):
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Source fingerprint path must remain repository-relative")
        path = (root / rel).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Approved source fingerprint file missing: {relative}")
        source_text = path.read_bytes().decode("utf-8")
        canonical_source = source_text.replace("\r\n", "\n").replace("\r", "\n")
        entries[rel.as_posix()] = hashlib.sha256(
            canonical_source.encode("utf-8")).hexdigest()
    return {"files": entries, "aggregate_sha256": sha256_json(entries)}


@dataclass(frozen=True)
class FeatureKey:
    image_sha256: str
    encoder_fingerprint_sha256: str
    preprocessing_fingerprint_sha256: str

    @property
    def sha256(self) -> str:
        _require_sha(self.image_sha256, "image_sha256")
        _require_sha(self.encoder_fingerprint_sha256, "encoder_fingerprint_sha256")
        _require_sha(self.preprocessing_fingerprint_sha256,
                     "preprocessing_fingerprint_sha256")
        return sha256_json(asdict(self))


class FeatureCache:
    """In-memory, private feature reuse keyed by exact image/encoder/preprocess identity."""

    def __init__(self, budget: ComputeBudget):
        self._budget = budget
        self._values: dict[str, Any] = {}

    def get_or_encode(self, key: FeatureKey, encode: Callable[[], Any]) -> Any:
        cache_key = key.sha256
        if cache_key not in self._values:
            self._budget.charge("feature_encoding")
            self._values[cache_key] = encode()
        return self._values[cache_key]

    def provenance(self) -> dict[str, Any]:
        return {"cached_feature_count": len(self._values),
                "feature_encoding_calls": self._budget.feature_encodings,
                "feature_payload_serialized": False}


@dataclass(frozen=True)
class StaticMapKey:
    image_sha256: str
    static_memory_sha256: str
    scorer_config_sha256: str
    scorer_source_sha256: str

    @property
    def sha256(self) -> str:
        for field_name, value in asdict(self).items():
            _require_sha(value, field_name)
        return sha256_json(asdict(self))


class StaticMapCache:
    """Private static M0 score-map reuse under exact per-seed memory identity."""

    def __init__(self, budget: ComputeBudget):
        self._budget = budget
        self._values: dict[str, tuple[float, Any]] = {}

    def get_or_score(self, key: StaticMapKey,
                     score_map: Callable[[], tuple[float, Any]]) -> tuple[float, Any]:
        cache_key = key.sha256
        if cache_key not in self._values:
            self._budget.charge("static_map")
            score, amap = score_map()
            if not math.isfinite(float(score)):
                raise P1ParityMismatch("Regenerated static scalar score is nonfinite")
            self._values[cache_key] = (float(score), amap)
        return self._values[cache_key]


@dataclass(frozen=True)
class MemoryIdentity:
    version: int
    memory_sha256: str
    source_map_sha256: str
    source_revision: int
    legacy_greedy_version: int

    def __post_init__(self) -> None:
        if min(self.version, self.source_revision, self.legacy_greedy_version) < 0:
            raise ValueError("Memory identity counters must be nonnegative")
        _require_sha(self.memory_sha256, "memory_sha256")
        _require_sha(self.source_map_sha256, "source_map_sha256")


@dataclass(frozen=True)
class C64Identity:
    memory_version: int
    memory_sha256: str
    q90: float
    transport_sha256: str

    def __post_init__(self) -> None:
        if self.memory_version < 0 or not math.isfinite(float(self.q90)):
            raise ValueError("Invalid C64 state")
        _require_sha(self.memory_sha256, "memory_sha256")
        _require_sha(self.transport_sha256, "transport_sha256")


@dataclass(frozen=True)
class ReplayEvent:
    position: int
    queried: bool
    source_key: str
    image_sha256: str | None
    static_score: float
    dynamic_score: float
    memory_before: MemoryIdentity
    memory_after: MemoryIdentity
    c64_before: C64Identity
    c64_after: C64Identity
    expected_writer_signature: Mapping[str, Any]
    static_memory_sha256: str

    def __post_init__(self) -> None:
        if self.position < 0 or not self.source_key:
            raise ValueError("Replay event lacks position/source identity")
        if not math.isfinite(float(self.static_score)) or not math.isfinite(float(self.dynamic_score)):
            raise ValueError("Replay event score tape contains nonfinite values")
        _require_sha(self.static_memory_sha256, "static_memory_sha256")
        if self.queried and self.image_sha256 is None:
            raise ValueError("Queried replay event requires a private image identity hash")
        if self.image_sha256 is not None:
            _require_sha(self.image_sha256, "image_sha256")

    @property
    def source_identity_sha256(self) -> str:
        return hashlib.sha256(self.source_key.encode("utf-8")).hexdigest()


def shared_writer_memory_identity(writer: Any) -> MemoryIdentity:
    """Read exact public identity fields from the existing SharedNativeGreedyWriter."""
    bank = writer.bank
    return MemoryIdentity(
        int(bank.memory_content_version), str(bank.sha256()),
        str(writer.source_map_sha256()), int(writer.source_revision),
        int(bank.legacy_greedy_version))


def apply_shared_native_writer(writer: Any, event: ReplayEvent, queried: bool,
                               label: int | None, features: Any,
                               static_amap: Any) -> Any:
    """Adapt a replay event to the frozen writer without cloning or changing it."""
    return writer.apply(
        queried=queried, label=label, source_key=event.source_key,
        score=float(event.static_score), static_amap=static_amap,
        full_features=() if features is None else features,
        step=int(event.position))


class StoredC64StateLookup:
    """Exact in-memory lookup over the historical C64 transport tape."""
    def __init__(self, states: Mapping[tuple[int, str], C64Identity]):
        self._states = dict(states)
        for (version, memory_sha), state in self._states.items():
            _require_sha(memory_sha, "memory_sha256")
            if state.memory_version != version or state.memory_sha256 != memory_sha:
                raise ValueError("C64 lookup key differs from its stored identity")

    def __call__(self, memory: MemoryIdentity) -> C64Identity:
        key = (memory.version, memory.memory_sha256)
        try:
            state = self._states[key]
        except KeyError as exc:
            raise P1ParityMismatch(
                f"no stored C64 transport state for memory version {memory.version}") from exc
        if state.memory_sha256 != memory.memory_sha256:
            raise P1ParityMismatch("stored C64 state is bound to another memory digest")
        return state


@dataclass(frozen=True)
class ReplayCheckpoint:
    schema: str
    authorization_id: str
    trajectory_id: str
    source_fingerprint_sha256: str
    next_position: int
    memory: MemoryIdentity
    c64: C64Identity
    writer_state_sha256: str
    event_chain_sha256: str
    query_response_sha256: str

    def __post_init__(self) -> None:
        if self.schema != REPLAY_CHECKPOINT_SCHEMA or self.next_position < 0:
            raise ValueError("Invalid P1 replay checkpoint identity/cursor")
        for name in ("source_fingerprint_sha256", "writer_state_sha256",
                     "event_chain_sha256", "query_response_sha256"):
            _require_sha(getattr(self, name), name)

    def to_public_dict(self) -> dict[str, Any]:
        """The checkpoint contains digests/state identities only, never labels or tensors."""
        return asdict(self)


@dataclass(frozen=True)
class ReplayResult:
    status: str
    complete: bool
    next_position: int
    events_checked: int
    query_events: int
    memory_parity_events: int
    writer_parity_events: int
    c64_parity_events: int
    query_response_sha256: str
    event_chain_sha256: str
    checkpoint: ReplayCheckpoint


class QueriedLabelCache:
    """Private, once-per-class/query-position label disclosure cache.

    It has no API for unqueried positions and charges the authorization budget
    before invoking the private label reader. Raw labels never leave this object
    except transiently to the replay writer/controller adapter.
    """

    def __init__(self, budget: ComputeBudget,
                 query_positions_by_class: Mapping[str, Sequence[int]]):
        self._budget = budget
        self._positions = {category: frozenset(int(x) for x in positions)
                           for category, positions in query_positions_by_class.items()}
        if set(self._positions) != set(budget.authorization.allowed_classes):
            raise P1Blocked("Queried label schedule must cover exactly the authorized classes")
        self._values: dict[tuple[str, int, str], int] = {}
        self._groups_by_position: dict[tuple[str, int], str] = {}
        self._position_by_group: dict[tuple[str, str], int] = {}

    def get(self, category: str, position: int, source_key: str,
            reader: Callable[[str, int, str], int]) -> int:
        if category not in self._positions or position not in self._positions[category]:
            raise P1Blocked("Queried-label access attempted outside the precommitted query tape")
        group_sha = hashlib.sha256(source_key.encode("utf-8")).hexdigest()
        position_key = (category, position)
        old_group = self._groups_by_position.get(position_key)
        if old_group is not None and old_group != group_sha:
            raise P1ParityMismatch("query position changed official sample group across support seeds")
        group_key = (category, group_sha)
        old_position = self._position_by_group.get(group_key)
        if old_position is not None and old_position != position:
            raise P1ParityMismatch("official sample group appears at multiple queried positions")
        self._groups_by_position[position_key] = group_sha
        self._position_by_group[group_key] = position
        key = (category, position, group_sha)
        if key not in self._values:
            self._budget.charge("queried_label")
            label = reader(category, position, source_key)
            if label not in (0, 1):
                raise P1ParityMismatch("queried label reader returned a nonbinary/unverified value")
            self._values[key] = int(label)
        return self._values[key]

    @property
    def disclosure_count(self) -> int:
        return len(self._values)

    def is_precommitted_query(self, category: str, position: int) -> bool:
        return category in self._positions and position in self._positions[category]

    @property
    def response_sha256(self) -> str:
        rows = [{"category": category, "position": position,
                 "source_identity_sha256": source_sha, "label": label}
                for (category, position, source_sha), label in sorted(self._values.items())]
        return sha256_json(rows)

    def private_state_dict(self) -> dict[str, Any]:
        """Private event-boundary state; callers must not publish this payload."""
        return {"values": [[category, position, source_sha, label]
                           for (category, position, source_sha), label
                           in sorted(self._values.items())],
                "groups_by_position": [[category, position, source_sha]
                                       for (category, position), source_sha
                                       in sorted(self._groups_by_position.items())]}

    def load_private_state_dict(self, state: Mapping[str, Any]) -> None:
        values = {}
        for row in state.get("values", ()):
            if len(row) != 4:
                raise P1ParityMismatch("private queried-label checkpoint row is malformed")
            category, position, source_sha, label = row
            if (category not in self._positions or position not in self._positions[category] or
                    label not in (0, 1)):
                raise P1ParityMismatch("private queried-label checkpoint exceeds authorization")
            _require_sha(source_sha, "source_identity_sha256")
            values[(category, int(position), source_sha)] = int(label)
        if len(values) > self._budget.authorization.max_queried_label_reads:
            raise P1Blocked("private queried-label checkpoint exceeds the approved read bound")
        self._budget.queried_label_reads = len(values)
        groups = {}
        for row in state.get("groups_by_position", ()):
            if len(row) != 3:
                raise P1ParityMismatch("private query-position checkpoint row is malformed")
            category, position, source_sha = row
            if category not in self._positions or position not in self._positions[category]:
                raise P1Blocked("private query-position checkpoint exceeds the approved tape")
            _require_sha(source_sha, "source_identity_sha256")
            groups[(category, int(position))] = source_sha
        by_group: dict[tuple[str, str], int] = {}
        for (category, position), source_sha in groups.items():
            old_position = by_group.get((category, source_sha))
            if old_position is not None and old_position != position:
                raise P1ParityMismatch("private query checkpoint repeats a source group")
            by_group[(category, source_sha)] = position
        if any(groups.get((category, position)) != source_sha
               for category, position, source_sha in values):
            raise P1ParityMismatch("private query-label checkpoint group linkage differs")
        self._values, self._groups_by_position = values, groups
        self._position_by_group = by_group


def _writer_signature(result: Any, source_key: str) -> dict[str, Any]:
    if hasattr(result, "__dataclass_fields__"):
        from dataclasses import asdict as _asdict
        result = _asdict(result)
    if not isinstance(result, Mapping):
        raise TypeError("Writer adapter must return a mapping/dataclass signature")
    signature = dict(result)
    raw_source = signature.pop("source_key", source_key)
    signature["source_identity_sha256"] = hashlib.sha256(
        str(raw_source).encode("utf-8")).hexdigest()
    return signature


def _same(label: str, expected: Any, actual: Any, position: int) -> None:
    if canonical_json(expected) != canonical_json(actual):
        raise P1ParityMismatch(f"event {position}: exact {label} mismatch; stop trajectory")


class P1NativeReplayRunner:
    """Replay writer events over query positions while reusing full-stream score tapes."""

    def __init__(self, authorization_record: Mapping[str, Any] | None,
                 query_feedback_authorization_record: Mapping[str, Any] | None, *,
                 authorization: BoundedAuthorization | None = None,
                 _standing_p12_context_token: object | None = None,
                 budget: ComputeBudget | None = None,
                 feature_cache: FeatureCache | None = None,
                 static_map_cache: StaticMapCache | None = None):
        # This is deliberately the first validation. Callers must supply lazy
        # event/image/label providers so no restricted input is opened earlier.
        if authorization is None:
            if _standing_p12_context_token is not None:
                raise P1Blocked("Standing P1.2 context token was supplied without a bounded context")
            self.authorization = BoundedAuthorization.from_records(
                authorization_record, query_feedback_authorization_record)
        else:
            # P1.2 can also receive a bounded authorization context established
            # by the active standing researcher instruction. This is an
            # in-memory scope/budget object, not an approval record. The
            # P1.1 production wrapper must bind it to all four verified tapes
            # before constructing this runner.
            if authorization_record is not None or query_feedback_authorization_record is not None:
                raise P1Blocked("Use either recorded approvals or a standing authorization context, not both")
            maxima = {
                "max_queried_label_reads": MAX_QUERY_FEATURE_ENCODINGS,
                "max_feature_encodings": MAX_QUERY_FEATURE_ENCODINGS,
                "max_static_map_calls": MAX_STATIC_MAP_CALLS,
                "max_anchor_score_calls": MAX_ANCHOR_SCORE_CALLS,
            }
            if (_standing_p12_context_token is not _P12_STANDING_CONTEXT_TOKEN or
                    authorization.allowed_classes != P1_CLASSES or
                    authorization.query_protocol_sha256 != P1_PROTOCOL_SHA256 or
                    not authorization.authorization_id or
                    not authorization.query_feedback_authorization_id or
                    any(isinstance(getattr(authorization, field), bool) or
                        not isinstance(getattr(authorization, field), int) or
                        not 0 <= getattr(authorization, field) <= maximum
                        for field, maximum in maxima.items())):
                raise P1Blocked("Standing authorization context is outside the immutable P1 scope or quotas")
            self.authorization = authorization
        self.budget = budget or ComputeBudget(self.authorization)
        if self.budget.authorization != self.authorization:
            raise P1Blocked("Compute budget belongs to a different authorization")
        self.features = feature_cache or FeatureCache(self.budget)
        self.static_maps = static_map_cache or StaticMapCache(self.budget)
        if self.features._budget is not self.budget or self.static_maps._budget is not self.budget:
            raise P1Blocked("Shared feature/static-map caches must use the same bounded compute budget")

    def run(self, *, events_factory: Callable[[int], Iterable[ReplayEvent]],
            total_events: int, trajectory_id: str, category: str,
            source_fingerprint_sha256: str,
            scorer_config_sha256: str,
            scorer_source_sha256: str,
            state_fn: Callable[[], MemoryIdentity],
            c64_fn: Callable[[MemoryIdentity], C64Identity],
            writer_state_sha256_fn: Callable[[], str],
            feature_key_fn: Callable[[ReplayEvent], FeatureKey],
            encode_feature_fn: Callable[[ReplayEvent], Any],
            queried_labels: QueriedLabelCache,
            label_reader: Callable[[str, int, str], int],
            static_map_fn: Callable[[ReplayEvent, Any], tuple[float, Any]],
            writer_fn: Callable[[ReplayEvent, bool, int | None, Any, Any], Any],
            checkpoint: ReplayCheckpoint | None = None,
            max_events: int | None = None) -> ReplayResult:
        _require_sha(source_fingerprint_sha256, "source_fingerprint_sha256")
        _require_sha(scorer_config_sha256, "scorer_config_sha256")
        _require_sha(scorer_source_sha256, "scorer_source_sha256")
        if total_events < 0 or not trajectory_id:
            raise ValueError("Replay trajectory identity/length is invalid")
        if category not in self.authorization.allowed_classes:
            raise P1Blocked("Replay category is outside the approved four-class scope")
        if queried_labels._budget is not self.budget:
            raise P1Blocked("Queried-label cache belongs to another P1 compute budget")
        if max_events is not None and max_events < 0:
            raise ValueError("max_events must be nonnegative")
        if checkpoint is None:
            cursor = 0
            event_chain = hashlib.sha256(b"stage1a-p1-event-chain-v1").hexdigest()
            response_chain = hashlib.sha256(b"stage1a-p1-query-response-v1").hexdigest()
        else:
            if (checkpoint.authorization_id != self.authorization.authorization_id or
                    checkpoint.trajectory_id != trajectory_id or
                    checkpoint.source_fingerprint_sha256 != source_fingerprint_sha256):
                raise P1ParityMismatch("checkpoint identity differs; stop trajectory")
            cursor = checkpoint.next_position
            event_chain = checkpoint.event_chain_sha256
            response_chain = checkpoint.query_response_sha256
            _same("resume memory identity", asdict(checkpoint.memory),
                  asdict(state_fn()), cursor)
            _same("resume C64 transport identity", asdict(checkpoint.c64),
                  asdict(c64_fn(checkpoint.memory)), cursor)
            _same("resume private writer state digest", checkpoint.writer_state_sha256,
                  _require_sha(writer_state_sha256_fn(), "writer_state_sha256"), cursor)
        if cursor > total_events:
            raise P1ParityMismatch("checkpoint cursor exceeds trajectory length")

        event_count = query_count = memory_checks = writer_checks = c64_checks = 0
        stop_at = total_events if max_events is None else min(total_events, cursor + max_events)
        source = iter(events_factory(cursor))
        for position in range(cursor, stop_at):
            try:
                event = next(source)
            except StopIteration as exc:
                raise P1ParityMismatch(f"event source ended before position {position}") from exc
            if event.position != position:
                raise P1ParityMismatch(f"expected event {position}, got {event.position}; stop trajectory")
            if event.queried != queried_labels.is_precommitted_query(category, position):
                raise P1ParityMismatch(
                    f"event {position}: query flag differs from the fixed tape; stop trajectory")

            before = state_fn()
            _same("pre-event memory/source identity", asdict(event.memory_before),
                  asdict(before), position)
            c64_before = c64_fn(before)
            _same("pre-event C64 q90/transport identity", asdict(event.c64_before),
                  asdict(c64_before), position)
            memory_checks += 1
            c64_checks += 1

            features = static_amap = None
            label: int | None = None
            if event.queried:
                key = feature_key_fn(event)
                if key.image_sha256 != event.image_sha256:
                    raise P1ParityMismatch(f"event {position}: feature key image mismatch")
                features = self.features.get_or_encode(
                    key, lambda event=event: encode_feature_fn(event))
                # Match historical causal ordering: regenerate the queried
                # event's static scalar/map before label disclosure. The scalar
                # tape is reused for every event and checked against this score.
                map_key = StaticMapKey(
                    event.image_sha256, event.static_memory_sha256,
                    scorer_config_sha256, scorer_source_sha256)
                static_value, static_amap = self.static_maps.get_or_score(
                    map_key, lambda event=event, features=features:
                    static_map_fn(event, features))
                if static_value != event.static_score:
                    raise P1ParityMismatch(
                        f"event {position}: regenerated M0 scalar differs from saved score tape")
                label = queried_labels.get(category, position, event.source_key,
                                           label_reader)
                if label not in (0, 1):
                    raise P1ParityMismatch(f"event {position}: queried label is not verified binary")
                query_record = {"position": position,
                                "source_identity_sha256": event.source_identity_sha256,
                                "label": int(label)}
                response_chain = hashlib.sha256(
                    bytes.fromhex(response_chain) +
                    canonical_json(query_record).encode("ascii")).hexdigest()
                query_count += 1

            actual_writer = _writer_signature(
                writer_fn(event, event.queried, label, features, static_amap),
                event.source_key)
            _same("writer action/source attribution", dict(event.expected_writer_signature),
                  actual_writer, position)
            writer_checks += 1

            after = state_fn()
            _same("post-event memory/source identity", asdict(event.memory_after),
                  asdict(after), position)
            c64_after = c64_fn(after)
            _same("post-event C64 q90/transport identity", asdict(event.c64_after),
                  asdict(c64_after), position)
            memory_checks += 1
            c64_checks += 1

            event_row = {
                "position": position,
                "queried": event.queried,
                "static_score_tape": event.static_score,
                "dynamic_score_tape": event.dynamic_score,
                "memory_before": asdict(before),
                "memory_after": asdict(after),
                "writer_signature": actual_writer,
                "c64_before": asdict(c64_before),
                "c64_after": asdict(c64_after),
            }
            event_chain = hashlib.sha256(
                bytes.fromhex(event_chain) + canonical_json(event_row).encode("ascii")
            ).hexdigest()
            cursor += 1
            event_count += 1

        final_memory = state_fn()
        final_c64 = c64_fn(final_memory)
        checkpoint_out = ReplayCheckpoint(
            REPLAY_CHECKPOINT_SCHEMA, self.authorization.authorization_id,
            trajectory_id, source_fingerprint_sha256, cursor,
            final_memory, final_c64,
            _require_sha(writer_state_sha256_fn(), "writer_state_sha256"),
            event_chain, response_chain)
        complete = cursor == total_events
        return ReplayResult(
            "PASS_EXACT_PARITY" if complete else "INCOMPLETE_CHECKPOINT",
            complete, cursor, event_count, query_count, memory_checks,
            writer_checks, c64_checks, response_chain, event_chain,
            checkpoint_out)


@dataclass(frozen=True)
class QueryAnchor:
    anchor_identity_sha256: str
    source_group_sha256: str
    kind: str
    feature_ref: Any

    def __post_init__(self) -> None:
        _require_sha(self.anchor_identity_sha256, "anchor_identity_sha256")
        _require_sha(self.source_group_sha256, "source_group_sha256")
        if self.kind not in ("normal", "defect"):
            raise ValueError("Only queried verified-normal/defect anchors are eligible")


def native_score_cache_key(*, memory: MemoryIdentity,
                           anchor: QueryAnchor,
                           scorer_config_sha256: str,
                           scorer_source_sha256: str) -> str:
    """Exact score cache identity, including LSO policy and source attribution."""
    policy = ("queried_normal_all_layer_leave_source_out" if anchor.kind == "normal"
              else "queried_defect_not_written")
    return sha256_json({
        "memory_version": memory.version,
        "memory_content_sha256": memory.memory_sha256,
        "source_map_sha256": memory.source_map_sha256,
        "anchor_identity_sha256": anchor.anchor_identity_sha256,
        "anchor_source_group_sha256": anchor.source_group_sha256,
        "scorer_config_sha256": _require_sha(scorer_config_sha256,
                                              "scorer_config_sha256"),
        "scorer_source_sha256": _require_sha(scorer_source_sha256,
                                              "scorer_source_sha256"),
        "source_exclusion_semantics": policy,
    })


@dataclass(frozen=True)
class EvidenceRefresh:
    memory_version: int
    memory_sha256: str
    evidence_revision: int
    evidence_sha256: str
    c64_q90: float
    normal_scores: Mapping[str, float]
    defect_scores: Mapping[str, float]
    native_score_calls: int
    reused_query_scores: int


@dataclass(frozen=True)
class NativeAnchorScoreProof:
    """Adapter attestation for the exact memory/anchor/exclusion score call."""
    score: float
    memory_version: int
    memory_sha256: str
    source_map_sha256: str
    anchor_identity_sha256: str
    source_group_sha256: str
    source_exclusion_semantics: str
    source_group_remaining_by_layer: tuple[int, ...]


class VersionedAnchorEvidenceCache:
    """Version-addressed queried-anchor scores with exact C64 transport reuse."""

    def __init__(self, budget: ComputeBudget):
        self._budget = budget
        self._scores: dict[str, float] = {}
        self._evidence_sha256: str | None = None
        self._revision = 0
        self._seen_groups: dict[str, str] = {}
        self._seen_ids: dict[str, str] = {}

    def refresh(self, *, memory: MemoryIdentity,
                anchors: Sequence[QueryAnchor],
                protected_c64_scores: Sequence[float],
                c64_memory_sha256: str,
                expected_c64_q90: float,
                c64_transport_sha256: str,
                scorer_config_sha256: str,
                scorer_source_sha256: str,
                protected_group_sha256s: Sequence[str],
                deployed_support_group_sha256s: Sequence[str],
                score_anchor: Callable[[QueryAnchor, MemoryIdentity, str],
                                       NativeAnchorScoreProof]
                ) -> EvidenceRefresh:
        _require_sha(c64_memory_sha256, "c64_memory_sha256")
        _require_sha(c64_transport_sha256, "c64_transport_sha256")
        if c64_memory_sha256 != memory.memory_sha256:
            raise P1ParityMismatch("C64 score tape is bound to another memory state")
        if len(protected_c64_scores) != 64 or any(
                not math.isfinite(float(x)) for x in protected_c64_scores):
            raise P1ParityMismatch("exactly 64 finite protected C64 scores are required")
        c64_q90 = sorted(float(x) for x in protected_c64_scores)[57]
        if c64_q90 != expected_c64_q90:
            raise P1ParityMismatch("stored C64 q90 differs from exact higher-quantile replay")
        protected = set(protected_group_sha256s)
        supports = set(deployed_support_group_sha256s)
        seen_groups: set[str] = set()
        seen_ids: set[str] = set()
        current: dict[str, float] = {}
        calls_before = self._budget.anchor_score_calls
        reuse_count = 0
        for anchor in anchors:
            if anchor.anchor_identity_sha256 in seen_ids:
                raise P1ParityMismatch("duplicate queried anchor identity")
            if anchor.source_group_sha256 in seen_groups:
                raise P1ParityMismatch("more than one queried anchor from an official sample group")
            previous_anchor = self._seen_groups.get(anchor.source_group_sha256)
            if previous_anchor is not None and previous_anchor != anchor.anchor_identity_sha256:
                raise P1ParityMismatch("official sample group already supplied a queried anchor")
            previous_group = self._seen_ids.get(anchor.anchor_identity_sha256)
            if previous_group is not None and previous_group != anchor.source_group_sha256:
                raise P1ParityMismatch("queried anchor identity changed source group")
            if anchor.source_group_sha256 in protected or anchor.source_group_sha256 in supports:
                raise P1ParityMismatch("queried anchor overlaps protected C64/support")
            seen_ids.add(anchor.anchor_identity_sha256)
            seen_groups.add(anchor.source_group_sha256)
            self._seen_groups[anchor.source_group_sha256] = anchor.anchor_identity_sha256
            self._seen_ids[anchor.anchor_identity_sha256] = anchor.source_group_sha256
            key = native_score_cache_key(
                memory=memory, anchor=anchor,
                scorer_config_sha256=scorer_config_sha256,
                scorer_source_sha256=scorer_source_sha256)
            if key in self._scores:
                score = self._scores[key]
                reuse_count += 1
            else:
                policy = ("queried_normal_all_layer_leave_source_out" if anchor.kind == "normal"
                          else "queried_defect_not_written")
                self._budget.charge("anchor_score")
                proof = score_anchor(anchor, memory, policy)
                if not isinstance(proof, NativeAnchorScoreProof):
                    raise P1ParityMismatch("native scorer did not return an anchor-score proof")
                if (proof.memory_version != memory.version or
                        proof.memory_sha256 != memory.memory_sha256 or
                        proof.source_map_sha256 != memory.source_map_sha256 or
                        proof.anchor_identity_sha256 != anchor.anchor_identity_sha256 or
                        proof.source_group_sha256 != anchor.source_group_sha256 or
                        proof.source_exclusion_semantics != policy or
                        not proof.source_group_remaining_by_layer or
                        any(count != 0 for count in proof.source_group_remaining_by_layer)):
                    raise P1ParityMismatch(
                        "native anchor-score proof failed memory/source/LSO identity")
                score = float(proof.score)
                if not math.isfinite(score):
                    raise P1ParityMismatch("queried anchor native score is nonfinite")
                self._scores[key] = score
            current[anchor.anchor_identity_sha256] = score
        normal = {a.anchor_identity_sha256: current[a.anchor_identity_sha256]
                  for a in anchors if a.kind == "normal"}
        defect = {a.anchor_identity_sha256: current[a.anchor_identity_sha256]
                  for a in anchors if a.kind == "defect"}
        evidence_doc = {
            "memory_version": memory.version,
            "memory_sha256": memory.memory_sha256,
            "source_map_sha256": memory.source_map_sha256,
            "c64_transport_sha256": c64_transport_sha256,
            "c64_q90": c64_q90,
            "normal_scores": normal,
            "defect_scores": defect,
        }
        evidence_sha = sha256_json(evidence_doc)
        if evidence_sha != self._evidence_sha256:
            self._revision += 1
            self._evidence_sha256 = evidence_sha
        return EvidenceRefresh(memory.version, memory.memory_sha256,
                               self._revision, evidence_sha, c64_q90,
                               normal, defect,
                               self._budget.anchor_score_calls - calls_before,
                               reuse_count)


def make_p1_authorization_record(*, authorization_id: str, approved_by: str,
                                 approved_at_utc: str,
                                 max_feature_encodings: int = MAX_QUERY_FEATURE_ENCODINGS,
                                 max_static_map_calls: int = MAX_STATIC_MAP_CALLS,
                                 max_anchor_score_calls: int = MAX_ANCHOR_SCORE_CALLS
                                 ) -> dict[str, Any]:
    """Build the required record shape for a researcher; this does not approve it."""
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "decision": "PENDING_RESEARCHER_DECISION",
        "authorization_id": authorization_id,
        "approved_by": approved_by,
        "approved_at_utc": approved_at_utc,
        "scope": "STAGE1A_P1_NATIVE_REPLAY_AND_ANCHOR_EVIDENCE",
        "allowed_classes": list(P1_CLASSES),
        "query_protocol_sha256": P1_PROTOCOL_SHA256,
        "access": {"native_scorer_calls": True,
                   "unqueried_labels": False, "confirmation_classes": False,
                   "masks": False},
        "bounds": {"max_feature_encodings": max_feature_encodings,
                   "max_static_map_calls": max_static_map_calls,
                   "max_anchor_score_calls": max_anchor_score_calls},
    }


def make_queried_feedback_authorization_record(*, authorization_id: str,
                                               approved_by: str,
                                               approved_at_utc: str,
                                               max_queried_label_reads: int = MAX_QUERY_FEATURE_ENCODINGS
                                               ) -> dict[str, Any]:
    """Build the separate feedback-access record shape; this does not approve it."""
    return {
        "schema": QUERY_FEEDBACK_AUTHORIZATION_SCHEMA,
        "decision": "PENDING_RESEARCHER_DECISION",
        "authorization_id": authorization_id,
        "approved_by": approved_by,
        "approved_at_utc": approved_at_utc,
        "scope": "STAGE1A_PRECOMMITTED_QUERIED_POSITIONS_ONLY",
        "allowed_classes": list(P1_CLASSES),
        "query_protocol_sha256": P1_PROTOCOL_SHA256,
        "unqueried_labels": False,
        "confirmation_classes": False,
        "masks": False,
        "max_queried_label_reads": max_queried_label_reads,
    }
