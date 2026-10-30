"""CPU-testable production bindings for Stage 1-A P1 reconstruction.

The module reads no dataset images or labels at import time. Its artifact
preflights are deliberately allowlisted to the metadata-only frozen Q-A-TS
tapes and already stored score/transport arrays. Native image scoring remains
a caller-injected frozen ``NativeScoreService`` call and requires separate
authorization before any real-data provider is constructed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
from typing import Any, Callable, Mapping, Sequence

from .stage1a_p1_reconstruction import (
    AUTHORIZATION_SCHEMA, QUERY_FEEDBACK_AUTHORIZATION_SCHEMA,
    BoundedAuthorization, C64Identity, ComputeBudget, MemoryIdentity,
    NativeAnchorScoreProof, P1Blocked, P1NativeReplayRunner,
    P1ParityMismatch, QueryAnchor, QueriedLabelCache, ReplayResult,
    VersionedAnchorEvidenceCache, apply_shared_native_writer, canonical_json,
    sha256_json, shared_writer_memory_identity,
)


P11_AUTHORIZATION_SCHEMA = "calibration.stage1a_p11_bounded_authorization.v1"
P11_FEEDBACK_AUTHORIZATION_SCHEMA = "calibration.stage1a_p11_queried_feedback_access.v1"
P11_PRIVATE_CHECKPOINT_SCHEMA = "calibration.stage1a_p11_private_checkpoint.v1"
P11_CLASSES = ("audiojack", "regulator", "switch", "rolled_strip_base")
P11_QUERY_PROTOCOL_VERSION = "Q-A-TS-v1"
P11_QUERY_SEED = 37
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class P11IntegrityError(RuntimeError):
    """Frozen tape, historical record, C64, or private-state parity failure."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: str, name: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise P11IntegrityError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _p11_digest(value: Any) -> str:
    return _sha_bytes(canonical_json(value).encode("ascii"))


@dataclass(frozen=True)
class VerifiedQATapes:
    """Digest-only public identities plus private, label-blind query positions."""

    bindings: tuple[Mapping[str, Any], ...]
    positions_by_class: Mapping[str, tuple[int, ...]]
    group_identity_by_class_position: Mapping[str, Mapping[int, str]]
    source_artifact_sha256: Mapping[str, str]
    verification_sha256: str

    def __post_init__(self) -> None:
        if tuple(row.get("class_identity") for row in self.bindings) != P11_CLASSES:
            raise P11IntegrityError("Verified Q-A-TS set must contain the four classes in protocol order")
        if set(self.positions_by_class) != set(P11_CLASSES):
            raise P11IntegrityError("Verified Q-A-TS positions must cover exactly four classes")
        if set(self.group_identity_by_class_position) != set(P11_CLASSES):
            raise P11IntegrityError("Verified group identities must cover exactly four classes")
        for row in self.bindings:
            _require_sha(str(row.get("tape_sha256", "")), "tape_sha256")
            _require_sha(str(row.get("positions_sha256", "")), "positions_sha256")
            positions = self.positions_by_class[str(row["class_identity"])]
            if tuple(sorted(set(positions))) != positions:
                raise P11IntegrityError("Verified query positions must be unique and increasing")
            if len(positions) != row.get("query_count"):
                raise P11IntegrityError("Tape query count differs from its private position set")
            if _p11_digest(list(positions)) != row.get("positions_sha256"):
                raise P11IntegrityError("Query-position identity digest mismatch")
            group_map = self.group_identity_by_class_position[str(row["class_identity"])]
            if tuple(sorted(group_map)) != positions or len(set(group_map.values())) != len(group_map):
                raise P11IntegrityError("Verified query groups are missing, duplicated, or mispositioned")
            if any(not _HEX64.fullmatch(str(value)) for value in group_map.values()):
                raise P11IntegrityError("Verified query group identities must be opaque SHA-256 values")
            if _p11_digest({str(k): v for k, v in group_map.items()}) != row.get("query_group_positions_sha256"):
                raise P11IntegrityError("Query group-position identity digest mismatch")
        _require_sha(self.verification_sha256, "verification_sha256")
        expected = _p11_digest({
            "bindings": [dict(row) for row in self.bindings],
            "positions_by_class": {k: list(v) for k, v in self.positions_by_class.items()},
            "group_identity_by_class_position": {
                k: {str(pos): group for pos, group in value.items()}
                for k, value in self.group_identity_by_class_position.items()},
            "source_artifact_sha256": dict(self.source_artifact_sha256),
        })
        if expected != self.verification_sha256:
            raise P11IntegrityError("Verified Q-A-TS set self-identity mismatch")

    def authorization_bindings(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.bindings]

    def positions(self, class_identity: str) -> tuple[int, ...]:
        try:
            return self.positions_by_class[class_identity]
        except KeyError as exc:
            raise P1Blocked("Class is outside the four verified Q-A-TS tapes") from exc

    @property
    def binding_sha256(self) -> str:
        return _p11_digest(self.authorization_bindings())


def verify_qats_objects(frozen: Mapping[str, Any], tape_archive: Mapping[str, Any], *,
                        frozen_sha256: str, archive_sha256: str,
                        checksums_sha256: str) -> VerifiedQATapes:
    """Verify only the four frozen 3%/seed-37 tape objects and metadata streams.

    Stream records are projected to the exact label-blind fields used by the
    historical Stage 0.5 tape builder. No labels, masks, or image bytes are
    accepted or returned by this function.
    """
    from . import stage05
    from .profiles import canonical_json
    from .protocol import TapeRecord

    if frozen.get("schema") != "stage05.frozen_inputs.v1":
        raise P11IntegrityError("Unexpected frozen Stage 0.5 input schema")
    classes = frozen.get("classes")
    if not isinstance(classes, Mapping) or set(classes) != set(P11_CLASSES):
        raise P11IntegrityError("Frozen stream metadata does not match the four development classes")
    if frozen.get("query_seeds") is None or P11_QUERY_SEED not in frozen["query_seeds"]:
        raise P11IntegrityError("Frozen query registry omits seed 37")
    config = stage05.Stage05QueryConfig(seed=P11_QUERY_SEED)
    rows: list[dict[str, Any]] = []
    positions_by_class: dict[str, tuple[int, ...]] = {}
    groups_by_class_position: dict[str, dict[int, str]] = {}
    for category in P11_CLASSES:
        class_info = classes[category]
        stream = class_info.get("stream")
        if not isinstance(stream, list) or len(stream) != class_info.get("stream_images"):
            raise P11IntegrityError(f"{category}: frozen metadata stream count mismatch")
        records = []
        for row in stream:
            # Deliberately project only fields required for Q-A-TS-v1. In
            # particular, image_path and every label/mask field are ignored.
            rel_digest = _sha_bytes(
                ("image-identity\0" + str(row["relative_id"])).encode("utf-8"))
            records.append(TapeRecord(
                int(row["position"]), str(row["opaque_group"]),
                int(row["camera_id"]), rel_digest,
                str(row.get("timestamp", "")), 0))
        class_tapes = tape_archive.get(category)
        if not isinstance(class_tapes, Mapping):
            raise P11IntegrityError(f"{category}: frozen query tape is absent")
        stored_by_rate = {float(rate): seeds.get("37")
                          for rate, seeds in class_tapes.items()
                          if isinstance(seeds, Mapping)}
        stored = stored_by_rate.get(0.03)
        if not isinstance(stored, Mapping):
            raise P11IntegrityError(f"{category}: exact 3% seed-37 Q-A-TS tape is absent")
        try:
            stage05.verify_nested_tapes(stored_by_rate, records)
            expected_by_rate = stage05.build_nested_tapes(records, P11_QUERY_SEED)
        except Exception as exc:
            raise P11IntegrityError(f"{category}: historical Q-A-TS verification failed") from exc
        expected = expected_by_rate[0.03]
        expected_events = [stage05._event_payload(event)
                           for event in stage05.build_qa_ts(records, config).events]
        positions = tuple(sorted(int(event["position"]) for event in stored.get("events", ())))
        groups = [str(event["official_sample_key"]) for event in stored.get("events", ())]
        if (stored.get("version") != P11_QUERY_PROTOCOL_VERSION or
                float(stored.get("rate", -1)) != 0.03 or
                int(stored.get("seed", -1)) != P11_QUERY_SEED or
                stored != expected or stored.get("events") != expected_events or
                len(groups) != len(set(groups)) or
                len(positions) != int(stored.get("requested_budget", -1)) or
                len(records) != class_info.get("stream_images") or
                len(set(row.official_sample_key for row in records)) !=
                class_info.get("stream_official_groups")):
            raise P11IntegrityError(f"{category}: tape policy, dimensions, or group uniqueness mismatch")
        stored_digest = frozen.get("tape_digests_by_seed37", {}).get(category, {}).get("0.03")
        if stored["digest"] != stored_digest:
            raise P11IntegrityError(f"{category}: frozen manifest tape digest differs")
        if any(position < 0 or position >= len(records) for position in positions):
            raise P11IntegrityError(f"{category}: query position is outside its metadata stream")
        positions_by_class[category] = positions
        groups_by_class_position[category] = {
            int(event["position"]): str(event["official_sample_key"])
            for event in stored["events"]}
        rows.append({
            "class_identity": category,
            "policy": P11_QUERY_PROTOCOL_VERSION,
            "seed": P11_QUERY_SEED,
            "query_protocol_sha256": "6f1d8cf40e5e7b2d98dcfabe5836b49c6cd57069d3c8377122dba64e3fc2da1c",
            "query_config_sha256": _sha_bytes(canonical_json(asdict(config)).encode("utf-8")),
            "tape_sha256": stored["digest"],
            "stream_sha256": stored["stream_digest"],
            "positions_sha256": _p11_digest(list(positions)),
            "query_group_positions_sha256": _p11_digest({
                str(pos): group for pos, group in groups_by_class_position[category].items()}),
            "n_images": len(records),
            "n_official_groups": int(class_info["stream_official_groups"]),
            "query_count": len(positions),
            "official_group_unique": True,
        })
    source_artifacts = {
        "frozen_inputs_sha256": _require_sha(frozen_sha256, "frozen_inputs_sha256"),
        "query_tapes_archive_sha256": _require_sha(archive_sha256, "query_tapes_archive_sha256"),
        "stage05_checksums_manifest_sha256": _require_sha(checksums_sha256, "checksums_sha256"),
    }
    verify_sha = _p11_digest({
        "bindings": rows,
        "positions_by_class": {k: list(v) for k, v in positions_by_class.items()},
        "group_identity_by_class_position": {
            k: {str(pos): group for pos, group in value.items()}
            for k, value in groups_by_class_position.items()},
        "source_artifact_sha256": source_artifacts,
    })
    return VerifiedQATapes(tuple(rows), positions_by_class, groups_by_class_position,
                           source_artifacts, verify_sha)


def verify_stage05_qats_artifacts(stage05_output: Path) -> VerifiedQATapes:
    """Read only checksums, frozen metadata, and the label-blind Q-A-TS archive."""
    stage05_output = stage05_output.resolve()
    checks_path = stage05_output / "checksums.json"
    frozen_path = stage05_output / "frozen_inputs.json"
    tapes_path = stage05_output / "query_tapes.json.gz"
    for path in (checks_path, frozen_path, tapes_path):
        if not path.is_file():
            raise P11IntegrityError(f"Missing approved metadata-only Q-A-TS artifact: {path.name}")
    checks_bytes = checks_path.read_bytes()
    checks = json.loads(checks_bytes.decode("utf-8"))
    files = checks.get("files")
    if not isinstance(files, Mapping):
        raise P11IntegrityError("Stage 0.5 checksums manifest lacks file entries")
    frozen_bytes = frozen_path.read_bytes()
    tape_bytes = tapes_path.read_bytes()
    for name, payload in (("frozen_inputs.json", frozen_bytes),
                          ("query_tapes.json.gz", tape_bytes)):
        if files.get(name) != _sha_bytes(payload):
            raise P11IntegrityError(f"Stage 0.5 checksum entry does not match {name}")
    frozen = json.loads(frozen_bytes.decode("utf-8"))
    if frozen.get("query_tapes_file_sha256") != _sha_bytes(tape_bytes):
        raise P11IntegrityError("Frozen manifest query-tape archive hash differs")
    tape_archive = json.loads(gzip.decompress(tape_bytes).decode("utf-8"))
    return verify_qats_objects(
        frozen, tape_archive, frozen_sha256=_sha_bytes(frozen_bytes),
        archive_sha256=_sha_bytes(tape_bytes), checksums_sha256=_sha_bytes(checks_bytes))


def validate_p11_authorizations(native_record: Mapping[str, Any] | None,
                                feedback_record: Mapping[str, Any] | None,
                                verified_tapes: VerifiedQATapes
                                ) -> BoundedAuthorization:
    """Require both approvals to bind the exact four verified tape identities."""
    if not isinstance(native_record, Mapping) or not isinstance(feedback_record, Mapping):
        raise P1Blocked("P1.1 requires separate native/data and queried-feedback approvals")
    expected = verified_tapes.authorization_bindings()
    for record, schema, scope in (
            (native_record, P11_AUTHORIZATION_SCHEMA,
             "STAGE1A_P1_NATIVE_REPLAY_AND_ANCHOR_EVIDENCE"),
            (feedback_record, P11_FEEDBACK_AUTHORIZATION_SCHEMA,
             "STAGE1A_PRECOMMITTED_QUERIED_POSITIONS_ONLY")):
        if (record.get("schema") != schema or record.get("decision") != "APPROVED" or
                record.get("scope") != scope or
                tuple(record.get("allowed_classes", ())) != P11_CLASSES or
                record.get("query_protocol_sha256") != expected[0]["query_protocol_sha256"] or
                canonical_json(record.get("query_tape_bindings")) != canonical_json(expected)):
            raise P1Blocked("P1.1 approval is missing, out of scope, or not bound to verified Q-A-TS tape identities")
    # Use the prior validated authorization implementation after P1.1 has
    # established exact tape binding. This preserves all previous bounds.
    native_legacy = dict(native_record)
    native_legacy["schema"] = AUTHORIZATION_SCHEMA
    feedback_legacy = dict(feedback_record)
    feedback_legacy["schema"] = QUERY_FEEDBACK_AUTHORIZATION_SCHEMA
    return BoundedAuthorization.from_records(native_legacy, feedback_legacy)


def make_p11_authorization_templates(*, authorization_id: str,
                                     feedback_authorization_id: str,
                                     verified_tapes: VerifiedQATapes,
                                     approved_by: str = "",
                                     approved_at_utc: str = "",
                                     max_queried_label_reads: int = 459,
                                     max_feature_encodings: int = 459,
                                     max_static_map_calls: int = 2295,
                                     max_anchor_score_calls: int = 75406
                                     ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create pending records with verified tape identities; never self-approve."""
    common = {
        "allowed_classes": list(P11_CLASSES),
        "query_protocol_sha256": verified_tapes.bindings[0]["query_protocol_sha256"],
        "query_tape_bindings": verified_tapes.authorization_bindings(),
        "approved_by": approved_by,
        "approved_at_utc": approved_at_utc,
    }
    native = {
        **common, "schema": P11_AUTHORIZATION_SCHEMA,
        "decision": "PENDING_RESEARCHER_DECISION",
        "authorization_id": authorization_id,
        "scope": "STAGE1A_P1_NATIVE_REPLAY_AND_ANCHOR_EVIDENCE",
        "access": {"native_scorer_calls": True, "unqueried_labels": False,
                   "confirmation_classes": False, "masks": False},
        "bounds": {"max_queried_label_reads": max_queried_label_reads,
                   "max_feature_encodings": max_feature_encodings,
                   "max_static_map_calls": max_static_map_calls,
                   "max_anchor_score_calls": max_anchor_score_calls},
    }
    feedback = {
        **common, "schema": P11_FEEDBACK_AUTHORIZATION_SCHEMA,
        "decision": "PENDING_RESEARCHER_DECISION",
        "authorization_id": feedback_authorization_id,
        "scope": "STAGE1A_PRECOMMITTED_QUERIED_POSITIONS_ONLY",
        "unqueried_labels": False, "confirmation_classes": False, "masks": False,
        "max_queried_label_reads": max_queried_label_reads,
    }
    return native, feedback


class P11ProductionReplayRunner:
    """Authorization/tape-gated front door to the P1 exact event runner."""

    def __init__(self, native_authorization: Mapping[str, Any] | None,
                 feedback_authorization: Mapping[str, Any] | None,
                 verified_tapes: VerifiedQATapes, **runner_kwargs: Any):
        self.verified_tapes = verified_tapes
        self.authorization = validate_p11_authorizations(
            native_authorization, feedback_authorization, verified_tapes)
        native_legacy = dict(native_authorization or {})
        native_legacy["schema"] = AUTHORIZATION_SCHEMA
        feedback_legacy = dict(feedback_authorization or {})
        feedback_legacy["schema"] = QUERY_FEEDBACK_AUTHORIZATION_SCHEMA
        self.runner = P1NativeReplayRunner(native_legacy, feedback_legacy,
                                           **runner_kwargs)
        self.queried_labels = QueriedLabelCache(
            self.runner.budget,
            {name: verified_tapes.positions(name) for name in P11_CLASSES})

    def run(self, **kwargs: Any) -> ReplayResult:
        category = kwargs.get("category")
        if category not in P11_CLASSES:
            raise P1Blocked("P1.1 replay category is outside the verified four-class tape set")
        if self.queried_labels._positions[category] != frozenset(
                self.verified_tapes.positions(category)):
            raise P11IntegrityError("Label cache schedule changed after tape verification")
        factory = kwargs.get("events_factory")
        if not callable(factory):
            raise P11IntegrityError("P1.1 replay requires a lazy event metadata provider")
        expected_groups = self.verified_tapes.group_identity_by_class_position[category]
        def checked_events(start: int):
            for event in factory(start):
                expected_group = expected_groups.get(int(event.position))
                if expected_group is not None:
                    actual_group = _sha_bytes(
                        ("official-sample-group\0" + str(event.source_key)).encode("utf-8"))
                    if actual_group != expected_group:
                        raise P11IntegrityError(
                            f"event {event.position}: official-group identity differs from verified Q-A-TS tape")
                yield event
        kwargs["events_factory"] = checked_events
        kwargs["queried_labels"] = self.queried_labels
        evidence_revision_adapter = kwargs.pop("evidence_revision_adapter", None)
        if evidence_revision_adapter is not None:
            writer_fn = kwargs.get("writer_fn")
            state_fn = kwargs.get("state_fn")
            c64_fn = kwargs.get("c64_fn")
            if (not callable(writer_fn) or not callable(state_fn) or
                    not callable(c64_fn)):
                raise P11IntegrityError(
                    "Causal evidence integration requires writer, memory, and C64-state providers")
            evidence_revision_adapter.initialize(state_fn())

            def writer_then_refresh(event: Any, queried: bool, label: int | None,
                                    features: Any, static_amap: Any) -> Any:
                memory_before = state_fn()
                writer_record = writer_fn(event, queried, label, features, static_amap)
                memory_after = state_fn()
                from .stage1a_p1_reconstruction import _same, _writer_signature
                _same("writer action/source attribution",
                      dict(event.expected_writer_signature),
                      _writer_signature(writer_record, event.source_key), event.position)
                _same("post-event memory/source identity", asdict(event.memory_after),
                      asdict(memory_after), event.position)
                _same("post-event C64 q90/transport identity", asdict(event.c64_after),
                      asdict(c64_fn(memory_after)), event.position)
                evidence_revision_adapter.after_event(
                    event=event, queried=queried, label=label, features=features,
                    static_amap=static_amap, writer_record=writer_record,
                    memory_before=memory_before, memory_after=memory_after)
                return writer_record

            kwargs["writer_fn"] = writer_then_refresh
        return self.runner.run(**kwargs)


def historical_writer_record(result: Any, writer: Any) -> dict[str, Any]:
    """Call the historical schema constructor directly; preserve every field."""
    from .stage075a_runner import _writer_record
    row = _writer_record(result, writer)
    if not isinstance(row, dict) or not row:
        raise P11IntegrityError("Historical _writer_record returned no mapping")
    return row


def apply_historical_shared_writer(writer: Any, event: Any, queried: bool,
                                   label: int | None, features: Any,
                                   static_amap: Any) -> dict[str, Any]:
    """Apply the real shared writer and return its exact historical record."""
    result = apply_shared_native_writer(writer, event, queried, label,
                                        features, static_amap)
    return historical_writer_record(result, writer)


def expected_historical_replay_signature(record: Mapping[str, Any],
                                         source_key: str) -> dict[str, Any]:
    """Add the P1 event-chain source identity without altering the source row."""
    result = dict(record)
    result["source_identity_sha256"] = _sha_bytes(source_key.encode("utf-8"))
    return result


class NativeAllLayerLSOAdapter:
    """Use AnchorLedger's actual all-layer source filter before native scoring."""

    def __init__(self, writer: Any,
                 score_fn: Callable[[Any, Any], Any]):
        self.writer = writer
        self.score_fn = score_fn
        self.proofs: dict[str, dict[str, Any]] = {}

    def __call__(self, anchor: QueryAnchor, memory: MemoryIdentity,
                 policy: str) -> NativeAnchorScoreProof:
        from .anchor_ledger import AnchorLedger
        from .memory_state import CalibrationGreedyState
        live = shared_writer_memory_identity(self.writer)
        if live != memory:
            raise P1ParityMismatch("Native LSO adapter received stale memory/source identity")
        bank = self.writer.bank
        source_rows = [list(layer) for layer in self.writer.sources]
        if len(source_rows) != len(bank.layers) or any(
                len(src) != len(layer) for src, layer in zip(source_rows, bank.layers)):
            raise P11IntegrityError("Native source map does not align with actual prototype rows")
        layers = len(bank.layers)
        source_key = getattr(anchor, "source_key", None)
        if not isinstance(source_key, str):
            source_key = None
        # QueryAnchor stores only a source-group digest in the cache-facing
        # interface. Resolve the exact private source key from writer rows via
        # the caller-provided private feature reference tuple when available.
        if isinstance(anchor.feature_ref, Mapping):
            source_key = anchor.feature_ref.get("source_key", source_key)
        if not isinstance(source_key, str) or not source_key:
            raise P11IntegrityError("Native LSO requires the private exact source key")
        source_key_sha = _sha_bytes(source_key.encode("utf-8"))
        if source_key_sha != anchor.source_group_sha256:
            raise P11IntegrityError("Private source key differs from queried anchor source identity")
        removed_by_layer = tuple(
            sum(source == source_key for source in layer) for layer in source_rows)
        remaining_by_layer: tuple[int, ...]
        score_bank = bank
        excluded_sha = bank.sha256()
        if anchor.kind == "normal":
            if policy != "queried_normal_all_layer_leave_source_out":
                raise P11IntegrityError("Queried normal did not request historical all-layer LSO")
            filtered, removed_total = AnchorLedger._without_source(
                bank.legacy_bank, source_rows, source_key)
            filtered_sources = [tuple(row for row in layer if row != source_key)
                                for layer in source_rows]
            actual_removed = tuple(len(a) - len(b)
                                   for a, b in zip(source_rows, filtered_sources))
            if removed_total != sum(actual_removed):
                raise P11IntegrityError("AnchorLedger LSO row count differs from source-map inspection")
            if len(filtered.layers) != layers or len(filtered_sources) != layers:
                raise P11IntegrityError("Native LSO changed the number of layers")
            if any(len(tensor) != len(rows)
                   for tensor, rows in zip(filtered.layers, filtered_sources)):
                raise P11IntegrityError("Native LSO retained or removed rows inconsistently in a layer")
            remaining_by_layer = tuple(row.count(source_key) for row in filtered_sources)
            if any(remaining_by_layer):
                raise P11IntegrityError("Native LSO left source rows in one or more layers")
            score_bank = CalibrationGreedyState(filtered, bank.memory_content_version)
            excluded_sha = score_bank.sha256()
            removed_by_layer = actual_removed
        elif anchor.kind == "defect":
            if policy != "queried_defect_not_written":
                raise P11IntegrityError("Queried defect requested an unexpected source policy")
            if any(removed_by_layer):
                raise P11IntegrityError("Queried defect source is unexpectedly resident in the normal memory")
            remaining_by_layer = tuple(0 for _ in range(layers))
        else:  # QueryAnchor validates this, retain fail-closed behavior at the boundary.
            raise P11IntegrityError("Unsupported query anchor kind")
        result = self.score_fn(anchor.feature_ref, score_bank)
        score = float(result.score if hasattr(result, "score") else result)
        if (not math.isfinite(score) or
                (hasattr(result, "memory_version") and result.memory_version != memory.version)):
            raise P1ParityMismatch("Native anchor score is nonfinite or bound to another content version")
        proof = NativeAnchorScoreProof(
            score, memory.version, memory.memory_sha256, memory.source_map_sha256,
            anchor.anchor_identity_sha256, anchor.source_group_sha256,
            policy, remaining_by_layer)
        self.proofs[anchor.anchor_identity_sha256] = {
            "kind": anchor.kind,
            "memory_version": memory.version,
            "memory_sha256": memory.memory_sha256,
            "source_map_sha256": memory.source_map_sha256,
            "source_revision": memory.source_revision,
            "source_group_identity_sha256": anchor.source_group_sha256,
            "layer_count": layers,
            "removed_rows_by_layer": list(removed_by_layer),
            "remaining_source_rows_by_layer": list(remaining_by_layer),
            "scored_bank_sha256": excluded_sha,
            "source_exclusion_semantics": policy,
        }
        return proof


@dataclass(frozen=True)
class P11EvidenceResult:
    snapshot: Any
    protected_c64_scores: tuple[float, ...]
    evidence_sha256: str
    source_exclusion_proofs: Mapping[str, Mapping[str, Any]]


def refresh_complete_evidence(*, cache: VersionedAnchorEvidenceCache,
                              scorer: NativeAllLayerLSOAdapter,
                              memory: MemoryIdentity,
                              anchors: Sequence[QueryAnchor],
                              protected_c64_scores: Sequence[float],
                              c64_transport_sha256: str,
                              expected_c64_q90: float,
                              scorer_config_sha256: str,
                              scorer_source_sha256: str,
                              protected_group_sha256s: Sequence[str],
                              deployed_support_group_sha256s: Sequence[str]
                              ) -> P11EvidenceResult:
    """Create controller-compatible evidence from a full current state."""
    from .protocol import EvidenceAnchor, EvidenceSnapshot
    refreshed = cache.refresh(
        memory=memory, anchors=anchors,
        protected_c64_scores=protected_c64_scores,
        c64_memory_sha256=memory.memory_sha256,
        expected_c64_q90=expected_c64_q90,
        c64_transport_sha256=c64_transport_sha256,
        scorer_config_sha256=scorer_config_sha256,
        scorer_source_sha256=scorer_source_sha256,
        protected_group_sha256s=protected_group_sha256s,
        deployed_support_group_sha256s=deployed_support_group_sha256s,
        score_anchor=scorer)
    if len(protected_c64_scores) != 64:
        raise P11IntegrityError("A complete evidence snapshot requires all 64 protected C64 scores")
    normals = [EvidenceAnchor(f"c64:{index:03d}", float(value), memory.version,
                              True, 0, "C64_PROTECTED")
               for index, value in enumerate(protected_c64_scores)]
    defects = []
    for anchor in anchors:
        identity = anchor.anchor_identity_sha256
        proof = scorer.proofs.get(identity)
        if not proof or proof["memory_sha256"] != memory.memory_sha256:
            raise P11IntegrityError("Evidence refresh lacks current source-aware proof provenance")
        score = (refreshed.normal_scores if anchor.kind == "normal"
                 else refreshed.defect_scores)[identity]
        evidence_anchor = EvidenceAnchor(
            f"query:{identity}", float(score), memory.version, False,
            int(sum(proof["removed_rows_by_layer"])),
            "queried_normal_all_layer_leave_source_out" if anchor.kind == "normal"
            else "queried_defect_not_written")
        (normals if anchor.kind == "normal" else defects).append(evidence_anchor)
    audit = {
        "provenance_schema": "calibration.stage1a_p11_evidence_provenance.v1",
        "memory_version": memory.version,
        "memory_sha256": memory.memory_sha256,
        "source_map_sha256": memory.source_map_sha256,
        "source_revision": memory.source_revision,
        "legacy_greedy_version": memory.legacy_greedy_version,
        "evidence_revision": refreshed.evidence_revision,
        "evidence_sha256": refreshed.evidence_sha256,
        "protected_c64_count": 64,
        "queried_normal_lso_count": sum(a.kind == "normal" for a in anchors),
        "queried_defect_current_memory_count": sum(a.kind == "defect" for a in anchors),
        "c64_q90_higher_index_57": refreshed.c64_q90,
        "c64_transport_sha256": c64_transport_sha256,
        "scorer_config_sha256": scorer_config_sha256,
        "scorer_source_sha256": scorer_source_sha256,
        "source_exclusion_proofs": {
            key: dict(scorer.proofs[key]) for key in sorted(scorer.proofs)
            if key in {anchor.anchor_identity_sha256 for anchor in anchors}},
        "raw_source_keys_or_labels_in_snapshot": False,
    }
    snapshot = EvidenceSnapshot(memory.version, tuple(normals), tuple(defects),
                                audit, refreshed.evidence_revision)
    return P11EvidenceResult(snapshot, tuple(float(x) for x in protected_c64_scores),
                             refreshed.evidence_sha256,
                             audit["source_exclusion_proofs"])


class CausalEvidenceRevisionAdapter:
    """Refresh a complete snapshot at the P1 replay's post-write causal boundary.

    ``anchor_factory`` receives only the current queried event, its disclosed
    binary label, and its already encoded feature. ``protected_c64_state_fn``
    must provide the verified 64-score state for the exact post-event memory.
    Neither callback is invoked for an unqueried event unless the writer has
    unexpectedly changed memory, which fails closed.
    """

    def __init__(self, *, writer: Any, cache: VersionedAnchorEvidenceCache,
                 scorer: NativeAllLayerLSOAdapter,
                 anchor_factory: Callable[[Any, int, Any], QueryAnchor],
                 protected_c64_state_fn: Callable[[MemoryIdentity], Mapping[str, Any]],
                 scorer_config_sha256: str, scorer_source_sha256: str,
                 protected_group_sha256s: Sequence[str],
                 deployed_support_group_sha256s: Sequence[str]):
        self.writer = writer
        self.cache = cache
        self.scorer = scorer
        self.anchor_factory = anchor_factory
        self.protected_c64_state_fn = protected_c64_state_fn
        self.scorer_config_sha256 = _require_sha(scorer_config_sha256,
                                                 "scorer_config_sha256")
        self.scorer_source_sha256 = _require_sha(scorer_source_sha256,
                                                 "scorer_source_sha256")
        self.protected_group_sha256s = tuple(protected_group_sha256s)
        self.deployed_support_group_sha256s = tuple(deployed_support_group_sha256s)
        self.anchors: dict[str, QueryAnchor] = {}
        self.last_result: P11EvidenceResult | None = None
        self.revisions: list[dict[str, Any]] = []

    def initialize(self, memory: MemoryIdentity) -> P11EvidenceResult:
        """Publish the 64-anchor C64 snapshot for the initial memory before events."""
        if memory != shared_writer_memory_identity(self.writer):
            raise P1ParityMismatch("Initial evidence identity differs from the live native writer")
        c64 = self.protected_c64_state_fn(memory)
        if not isinstance(c64, Mapping) or not {
                "scores", "transport_sha256", "q90"}.issubset(c64):
            raise P11IntegrityError(
                "Initial evidence needs verified current-memory C64 scores, q90 and transport identity")
        result = refresh_complete_evidence(
            cache=self.cache, scorer=self.scorer, memory=memory, anchors=tuple(self.anchors.values()),
            protected_c64_scores=c64["scores"],
            c64_transport_sha256=str(c64["transport_sha256"]),
            expected_c64_q90=float(c64["q90"]),
            scorer_config_sha256=self.scorer_config_sha256,
            scorer_source_sha256=self.scorer_source_sha256,
            protected_group_sha256s=self.protected_group_sha256s,
            deployed_support_group_sha256s=self.deployed_support_group_sha256s)
        self.last_result = result
        if not self.revisions:
            self.revisions.append({
                "position": None, "queried": False, "writer_changed_memory": False,
                "memory_version": memory.version, "memory_sha256": memory.memory_sha256,
                "source_map_sha256": memory.source_map_sha256,
                "evidence_revision": result.snapshot.evidence_revision,
                "evidence_sha256": result.evidence_sha256,
                "query_anchor_count": len(self.anchors), "initial_snapshot": True,
            })
        return result

    def after_event(self, *, event: Any, queried: bool, label: int | None,
                    features: Any, static_amap: Any, writer_record: Any,
                    memory_before: MemoryIdentity,
                    memory_after: MemoryIdentity) -> P11EvidenceResult | None:
        if memory_after != shared_writer_memory_identity(self.writer):
            raise P1ParityMismatch("Evidence callback observed stale post-write memory identity")
        if not queried and memory_after != memory_before:
            raise P11IntegrityError(
                "Shared normal memory changed without a verified queried-normal event")
        if not queried and memory_after == memory_before:
            return None
        if queried:
            if label not in (0, 1):
                raise P11IntegrityError("Causal evidence refresh requires the disclosed binary query label")
            anchor = self.anchor_factory(event, int(label), features)
            expected_kind = "normal" if int(label) == 0 else "defect"
            expected_source = _sha_bytes(str(event.source_key).encode("utf-8"))
            if (not isinstance(anchor, QueryAnchor) or anchor.kind != expected_kind or
                    anchor.source_group_sha256 != expected_source):
                raise P11IntegrityError(
                    "Queried anchor identity/kind differs from the current disclosed event")
            if anchor.anchor_identity_sha256 in self.anchors:
                raise P11IntegrityError("Causal event attempted to add a duplicate queried anchor")
            self.anchors[anchor.anchor_identity_sha256] = anchor
        current_memory = shared_writer_memory_identity(self.writer)
        c64 = self.protected_c64_state_fn(current_memory)
        if not isinstance(c64, Mapping) or not {
                "scores", "transport_sha256", "q90"}.issubset(c64):
            raise P11IntegrityError(
                "Causal evidence needs verified current-memory C64 scores, q90 and transport identity")
        result = refresh_complete_evidence(
            cache=self.cache, scorer=self.scorer, memory=current_memory,
            anchors=tuple(self.anchors.values()),
            protected_c64_scores=c64["scores"],
            c64_transport_sha256=str(c64["transport_sha256"]),
            expected_c64_q90=float(c64["q90"]),
            scorer_config_sha256=self.scorer_config_sha256,
            scorer_source_sha256=self.scorer_source_sha256,
            protected_group_sha256s=self.protected_group_sha256s,
            deployed_support_group_sha256s=self.deployed_support_group_sha256s)
        if (result.snapshot.memory_version != memory_after.version or
                result.snapshot.source_exclusion_audit["memory_sha256"] != memory_after.memory_sha256 or
                result.snapshot.source_exclusion_audit["source_map_sha256"] != memory_after.source_map_sha256):
            raise P1ParityMismatch("Causal evidence revision does not bind the post-event memory/source state")
        self.last_result = result
        self.revisions.append({
            "position": int(event.position),
            "queried": bool(queried),
            "writer_changed_memory": memory_before != memory_after,
            "memory_version": memory_after.version,
            "memory_sha256": memory_after.memory_sha256,
            "source_map_sha256": memory_after.source_map_sha256,
            "evidence_revision": result.snapshot.evidence_revision,
            "evidence_sha256": result.evidence_sha256,
            "query_anchor_count": len(self.anchors),
            "writer_record_sha256": _p11_digest(writer_record),
            "feature_present": features is not None,
            "static_map_present": static_amap is not None,
        })
        return result


def verify_controller_snapshot_compatibility(snapshot: Any) -> dict[str, Any]:
    """Smoke the unchanged B4, B7 and Stage075BOffset EvidenceSnapshot inputs."""
    from .controllers import AdaptedDualACI
    from .scdac_g import SCDACGController
    from .stage075b_controllers import Stage075BOffset
    controllers = (AdaptedDualACI(0.5), SCDACGController(0.5),
                   Stage075BOffset(0.5))
    for controller in controllers:
        before = controller.state_dict()
        controller.on_memory_change(snapshot)
        if controller.state_dict() == before:
            # A duplicate version/revision is a no-op; fresh controllers must
            # observe a nonzero complete evidence revision in this smoke.
            if snapshot.evidence_revision > 0:
                raise P11IntegrityError(f"{controller.name} did not consume the fresh evidence revision")
        controller.finalize_step()
    return {"B4": "PASS_SYNTHETIC_EVIDENCE_INPUT",
            "B7": "PASS_SYNTHETIC_EVIDENCE_INPUT",
            "Stage075BOffset": "PASS_SYNTHETIC_EVIDENCE_INPUT"}


def query_response_chain_from_private_cache(labels: QueriedLabelCache,
                                            verified_tapes: VerifiedQATapes,
                                            category: str, next_position: int) -> str:
    """Recompute a trajectory's response chain from private cached disclosures."""
    state = labels.private_state_dict()
    values = {(str(cat), int(pos), str(source_sha)): int(label)
              for cat, pos, source_sha, label in state["values"]}
    groups = {(str(cat), int(pos)): str(source_sha)
              for cat, pos, source_sha in state["groups_by_position"]}
    chain = hashlib.sha256(b"stage1a-p1-query-response-v1").hexdigest()
    for position in verified_tapes.positions(category):
        if position >= next_position:
            break
        source_sha = groups.get((category, position))
        if source_sha is None:
            raise P11IntegrityError("Private label checkpoint omits a queried position before its cursor")
        label = values.get((category, position, source_sha))
        if label not in (0, 1):
            raise P11IntegrityError("Private label checkpoint has no verified binary response")
        response = {"position": position,
                    "source_identity_sha256": source_sha,
                    "label": label}
        chain = hashlib.sha256(bytes.fromhex(chain) +
                               canonical_json(response).encode("ascii")).hexdigest()
    return chain


def _private_mapping_digest(values: Mapping[str, Any]) -> str:
    return _p11_digest(values)


@dataclass
class P11PrivateCheckpoint:
    """Private resumable state. Never put this object or its bytes in Git."""

    schema: str
    authorization_id: str
    trajectory_id: str
    category: str
    source_fingerprint_sha256: str
    scorer_config_sha256: str
    scorer_source_sha256: str
    query_tape_binding_sha256: str
    replay_checkpoint: Any
    memory_identity: MemoryIdentity
    writer_state: Mapping[str, Any]
    queried_label_state: Mapping[str, Any]
    feature_cache_state: Mapping[str, Any]
    static_map_cache_state: Mapping[str, Any]
    evidence_cache_state: Mapping[str, Any] | None
    compute_counters: Mapping[str, int]
    response_chain_sha256: str
    private_state_sha256: str

    @classmethod
    def capture(cls, *, result: ReplayResult,
                production_runner: P11ProductionReplayRunner,
                writer: Any, trajectory_id: str, category: str,
                scorer_config_sha256: str, scorer_source_sha256: str,
                evidence_cache: VersionedAnchorEvidenceCache | None = None,
                causal_evidence_adapter: CausalEvidenceRevisionAdapter | None = None
                ) -> "P11PrivateCheckpoint":
        public = result.checkpoint
        response = query_response_chain_from_private_cache(
            production_runner.queried_labels, production_runner.verified_tapes,
            category, public.next_position)
        if response != public.query_response_sha256:
            raise P11IntegrityError("Queried-label state does not reproduce replay response-chain digest")
        writer_state = copy.deepcopy(writer.state_dict())
        label_state = copy.deepcopy(production_runner.queried_labels.private_state_dict())
        feature_state = copy.deepcopy(production_runner.runner.features._values)
        static_state = copy.deepcopy(production_runner.runner.static_maps._values)
        if (causal_evidence_adapter is not None and
                evidence_cache is not None and
                causal_evidence_adapter.cache is not evidence_cache):
            raise P11IntegrityError("Checkpoint cache differs from causal evidence adapter cache")
        effective_evidence_cache = (causal_evidence_adapter.cache
                                    if causal_evidence_adapter is not None
                                    else evidence_cache)
        evidence_state = None if effective_evidence_cache is None else {
            "scores": copy.deepcopy(effective_evidence_cache._scores),
            "evidence_sha256": effective_evidence_cache._evidence_sha256,
            "revision": effective_evidence_cache._revision,
            "seen_groups": copy.deepcopy(effective_evidence_cache._seen_groups),
            "seen_ids": copy.deepcopy(effective_evidence_cache._seen_ids),
            "causal_anchors": (copy.deepcopy(causal_evidence_adapter.anchors)
                               if causal_evidence_adapter is not None else None),
            "causal_proofs": (copy.deepcopy(causal_evidence_adapter.scorer.proofs)
                              if causal_evidence_adapter is not None else None),
            "causal_revisions": (copy.deepcopy(causal_evidence_adapter.revisions)
                                 if causal_evidence_adapter is not None else None),
        }
        counters = {
            "queried_label_reads": production_runner.runner.budget.queried_label_reads,
            "feature_encodings": production_runner.runner.budget.feature_encodings,
            "static_map_calls": production_runner.runner.budget.static_map_calls,
            "anchor_score_calls": production_runner.runner.budget.anchor_score_calls,
        }
        memory = shared_writer_memory_identity(writer)
        checkpoint = cls(
            P11_PRIVATE_CHECKPOINT_SCHEMA,
            production_runner.authorization.authorization_id,
            trajectory_id, category, public.source_fingerprint_sha256,
            scorer_config_sha256, scorer_source_sha256,
            production_runner.verified_tapes.binding_sha256,
            public, memory, writer_state, label_state, feature_state,
            static_state, evidence_state, counters, response, sha256_json({"pending": True}))
        checkpoint.private_state_sha256 = _private_mapping_digest(
            checkpoint._commitment_payload())
        return checkpoint

    def _commitment_payload(self) -> dict[str, Any]:
        writer_fields = ("schema", "fixed_support_sha256", "theta_ref", "sigma_ref",
                         "protected_source_keys", "capacities", "memory_content_version",
                         "legacy_greedy_version", "source_revision", "source_map_sha256",
                         "sha256", "content_sha256", "legacy_greedy_sha256")
        writer_profile = {key: self.writer_state.get(key) for key in writer_fields}
        return {
            "schema": self.schema,
            "authorization_id": self.authorization_id,
            "trajectory_id": self.trajectory_id,
            "category": self.category,
            "source_fingerprint_sha256": self.source_fingerprint_sha256,
            "scorer_config_sha256": self.scorer_config_sha256,
            "scorer_source_sha256": self.scorer_source_sha256,
            "query_tape_binding_sha256": self.query_tape_binding_sha256,
            "next_position": self.replay_checkpoint.next_position,
            "event_chain_sha256": self.replay_checkpoint.event_chain_sha256,
            "response_chain_sha256": self.response_chain_sha256,
            "memory_identity": asdict(self.memory_identity),
            "writer_profile": writer_profile,
            "queried_label_state_sha256": _private_mapping_digest(self.queried_label_state),
            "feature_cache_keys": sorted(self.feature_cache_state),
            "feature_payload_sha256": {
                key: _sha_bytes(pickle.dumps(value, protocol=5))
                for key, value in sorted(self.feature_cache_state.items())},
            "static_map_keys": sorted(self.static_map_cache_state),
            "static_payload_sha256": {
                key: _sha_bytes(pickle.dumps(value, protocol=5))
                for key, value in sorted(self.static_map_cache_state.items())},
            "evidence_cache_sha256": (
                _sha_bytes(pickle.dumps(self.evidence_cache_state, protocol=5))
                if self.evidence_cache_state is not None else None),
            "compute_counters": dict(self.compute_counters),
        }

    def public_summary(self) -> dict[str, Any]:
        """Safe linkage only; private tensors/features/labels are excluded."""
        return {
            "schema": self.schema,
            "authorization_id": self.authorization_id,
            "trajectory_id": self.trajectory_id,
            "category": self.category,
            "next_position": self.replay_checkpoint.next_position,
            "query_tape_binding_sha256": self.query_tape_binding_sha256,
            "memory_identity": asdict(self.memory_identity),
            "source_map_sha256": self.writer_state.get("source_map_sha256"),
            "private_query_label_count": len(self.queried_label_state.get("values", ())),
            "private_feature_cache_entry_count": len(self.feature_cache_state),
            "private_static_map_entry_count": len(self.static_map_cache_state),
            "private_evidence_score_entry_count": (
                len(self.evidence_cache_state.get("scores", {}))
                if self.evidence_cache_state is not None else 0),
            "compute_counters": dict(self.compute_counters),
            "response_chain_sha256": self.response_chain_sha256,
            "private_state_sha256": self.private_state_sha256,
            "private_payload_in_public_summary": False,
        }

    def restore(self, *, production_runner: P11ProductionReplayRunner,
                writer: Any,
                scorer_config_sha256: str, scorer_source_sha256: str,
                evidence_cache: VersionedAnchorEvidenceCache | None = None,
                causal_evidence_adapter: CausalEvidenceRevisionAdapter | None = None) -> None:
        if (self.schema != P11_PRIVATE_CHECKPOINT_SCHEMA or
                self.authorization_id != production_runner.authorization.authorization_id or
                self.query_tape_binding_sha256 != production_runner.verified_tapes.binding_sha256 or
                self.category not in P11_CLASSES or
                self.scorer_config_sha256 != scorer_config_sha256 or
                self.scorer_source_sha256 != scorer_source_sha256):
            raise P11IntegrityError("Private checkpoint authorization/tape/scorer identity mismatch")
        labels = production_runner.queried_labels
        labels.load_private_state_dict(copy.deepcopy(dict(self.queried_label_state)))
        if query_response_chain_from_private_cache(
                labels, production_runner.verified_tapes, self.category,
                self.replay_checkpoint.next_position) != self.response_chain_sha256:
            raise P11IntegrityError("Restored queried-label state does not match its response-chain digest")
        if _private_mapping_digest(self._commitment_payload()) != self.private_state_sha256:
            raise P11IntegrityError("Private checkpoint content/cache/state commitment mismatch")
        writer.load_state_dict(copy.deepcopy(dict(self.writer_state)))
        if shared_writer_memory_identity(writer) != self.memory_identity:
            raise P11IntegrityError("Restored native writer/source-map identity differs")
        p1runner = production_runner.runner
        p1runner.features._values = copy.deepcopy(dict(self.feature_cache_state))
        p1runner.static_maps._values = copy.deepcopy(dict(self.static_map_cache_state))
        for key in p1runner.features._values:
            _require_sha(key, "private feature-cache key")
        for key in p1runner.static_maps._values:
            _require_sha(key, "private static-map key")
        counters = dict(self.compute_counters)
        expected_counter_names = {"queried_label_reads", "feature_encodings",
                                  "static_map_calls", "anchor_score_calls"}
        if set(counters) != expected_counter_names:
            raise P11IntegrityError("Private checkpoint compute-counter schema mismatch")
        p1runner.budget.queried_label_reads = int(counters["queried_label_reads"])
        p1runner.budget.feature_encodings = int(counters["feature_encodings"])
        p1runner.budget.static_map_calls = int(counters["static_map_calls"])
        p1runner.budget.anchor_score_calls = int(counters["anchor_score_calls"])
        auth = p1runner.budget.authorization
        for field, maximum in (
                ("queried_label_reads", auth.max_queried_label_reads),
                ("feature_encodings", auth.max_feature_encodings),
                ("static_map_calls", auth.max_static_map_calls),
                ("anchor_score_calls", auth.max_anchor_score_calls)):
            if counters[field] < 0 or counters[field] > maximum:
                raise P11IntegrityError(f"Restored compute counter exceeds authorization: {field}")
        if counters["queried_label_reads"] != labels.disclosure_count:
            raise P11IntegrityError("Restored query-label count differs from its budget counter")
        if (counters["feature_encodings"] != len(p1runner.features._values) or
                counters["static_map_calls"] != len(p1runner.static_maps._values)):
            raise P11IntegrityError("Restored feature/map cache identities differ from compute counters")
        if self.evidence_cache_state is not None:
            if (causal_evidence_adapter is not None and evidence_cache is not None and
                    causal_evidence_adapter.cache is not evidence_cache):
                raise P11IntegrityError("Restore cache differs from causal evidence adapter cache")
            restore_cache = (causal_evidence_adapter.cache
                             if causal_evidence_adapter is not None else evidence_cache)
            if restore_cache is None:
                raise P11IntegrityError("Checkpoint has evidence cache state but no restore target")
            if restore_cache._budget is not p1runner.budget:
                raise P11IntegrityError("Evidence cache and production runner use different compute budgets")
            restore_cache._scores = copy.deepcopy(self.evidence_cache_state["scores"])
            restore_cache._evidence_sha256 = self.evidence_cache_state["evidence_sha256"]
            restore_cache._revision = int(self.evidence_cache_state["revision"])
            restore_cache._seen_groups = copy.deepcopy(self.evidence_cache_state["seen_groups"])
            restore_cache._seen_ids = copy.deepcopy(self.evidence_cache_state["seen_ids"])
            if counters["anchor_score_calls"] != restore_cache._budget.anchor_score_calls:
                restore_cache._budget.anchor_score_calls = counters["anchor_score_calls"]
            causal_anchor_state = self.evidence_cache_state.get("causal_anchors")
            if causal_evidence_adapter is not None and causal_anchor_state is None:
                raise P11IntegrityError("Checkpoint lacks causal anchor state for the restore adapter")
            if causal_anchor_state is not None:
                if causal_evidence_adapter is None:
                    raise P11IntegrityError("Checkpoint contains causal anchors but no adapter restore target")
                causal_evidence_adapter.anchors = copy.deepcopy(
                    causal_anchor_state)
                causal_evidence_adapter.scorer.proofs = copy.deepcopy(
                    self.evidence_cache_state["causal_proofs"])
                causal_evidence_adapter.revisions = copy.deepcopy(
                    self.evidence_cache_state["causal_revisions"])
                causal_evidence_adapter.last_result = None
        elif evidence_cache is not None or causal_evidence_adapter is not None:
            raise P11IntegrityError("Restore target expects evidence cache state absent from checkpoint")


def verify_historical_c64_artifact_links(stage05_output: Path,
                                         stage075a_output: Path) -> dict[str, Any]:
    """Verify stored C64 score vectors and transport hashes without rescoring.

    This checks serialized artifact linkage only. It does not encode images or
    independently recompute any Native C64 score.
    """
    import numpy as np
    from .stage075a import higher_quantile
    from .stage075a_runner import _sha_json

    stage05_output, stage075a_output = stage05_output.resolve(), stage075a_output.resolve()
    stage05_checks = json.loads((stage05_output / "checksums.json").read_text(encoding="utf-8"))
    stage075a_checks = json.loads((stage075a_output / "checksums.json").read_text(encoding="utf-8"))
    score_manifest_path = stage05_output / "score_tape_manifest.json"
    score_manifest_bytes = score_manifest_path.read_bytes()
    score_manifest = json.loads(score_manifest_bytes.decode("utf-8"))
    stage05_manifest_sha = _sha_bytes(score_manifest_bytes)
    if stage05_checks.get("files", {}).get("score_tape_manifest.json") != stage05_manifest_sha:
        raise P11IntegrityError("Stage 0.5 score-tape manifest checksum mismatch")
    linkage_path = stage075a_output / "audit_linkage.json"
    protocol_path = stage075a_output / "protocol_manifest.json"
    linkage_bytes = linkage_path.read_bytes()
    protocol_bytes = protocol_path.read_bytes()
    linkage = json.loads(linkage_bytes.decode("utf-8"))
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    linkage_sha = _sha_bytes(linkage_bytes)
    protocol_sha = _sha_bytes(protocol_bytes)
    root_files = stage075a_checks.get("files", {})
    if (root_files.get("audit_linkage.json") != linkage_sha or
            root_files.get("protocol_manifest.json") != protocol_sha):
        raise P11IntegrityError("Stage 0.75-A root manifest does not link audit linkage/protocol bytes")
    if linkage.get("protocol_manifest_sha256") != protocol_sha:
        raise P11IntegrityError("Stage 0.75-A audit linkage protocol digest mismatch")
    if linkage.get("run_manifest_sha256") != root_files.get("run_manifest.json"):
        raise P11IntegrityError("Stage 0.75-A audit linkage run-manifest digest mismatch")
    c64_rows = 0
    initial_vectors = 0
    update_vectors = 0
    verified_files: list[str] = []
    for category in P11_CLASSES:
        trace_link = linkage.get("source_trace_hashes_by_class", {}).get(category)
        if not isinstance(trace_link, Mapping):
            raise P11IntegrityError(f"{category}: historical audit linkage is missing")
        trajectory = stage075a_output / "trajectories" / category
        arrays_path = trajectory / "scores.npz"
        transport_path = trajectory / "c64_transport.json"
        per_checks = json.loads((trajectory / "checksums.json").read_text(encoding="utf-8"))
        arrays_sha, transport_sha = _sha_bytes(arrays_path.read_bytes()), _sha_bytes(transport_path.read_bytes())
        if (trace_link.get("scores_sha256") != arrays_sha or
                trace_link.get("transport_sha256") != transport_sha or
                per_checks.get("files", {}).get("scores.npz") != arrays_sha or
                per_checks.get("files", {}).get("c64_transport.json") != transport_sha):
            raise P11IntegrityError(f"{category}: stored C64 vector/transport file linkage mismatch")
        transport_by_seed = json.loads(transport_path.read_text(encoding="utf-8"))
        seeds = protocol.get("support_seeds")
        if seeds != [0, 1, 2, 3, 4]:
            raise P11IntegrityError("Historical Stage 0.75-A support-seed registry changed")
        score_entries = score_manifest.get("classes", {}).get(category, {}).get("score_tapes", {})
        with np.load(arrays_path, allow_pickle=False) as arrays:
            for seed in seeds:
                seed_key = str(seed)
                entry = score_entries.get(seed_key)
                if not isinstance(entry, Mapping):
                    raise P11IntegrityError(f"{category}/seed{seed}: Stage 0.5 C64 score entry missing")
                score_rel = entry["file"]
                score_path = stage05_output / score_rel
                score_sha = _sha_bytes(score_path.read_bytes())
                if (score_sha != entry.get("sha256") or
                        stage05_checks.get("files", {}).get(score_rel) != score_sha):
                    raise P11IntegrityError(f"{category}/seed{seed}: Stage 0.5 score tape checksum mismatch")
                with np.load(score_path, allow_pickle=False) as c64_archive:
                    initial = np.asarray(c64_archive["c64_scores"], dtype=np.float64)
                if initial.shape != (64,) or not np.isfinite(initial).all():
                    raise P11IntegrityError(f"{category}/seed{seed}: initial C64 vector invalid")
                before_key, after_key = f"c64_pairs_before_s{seed}", f"c64_pairs_after_s{seed}"
                if before_key not in arrays or after_key not in arrays:
                    raise P11IntegrityError(f"{category}/seed{seed}: stored C64 pair vectors are missing")
                before = np.asarray(arrays[before_key], dtype=np.float64)
                after = np.asarray(arrays[after_key], dtype=np.float64)
                rows = transport_by_seed.get(seed_key)
                if (before.ndim != 2 or after.shape != before.shape or before.shape[1:] != (64,) or
                        not isinstance(rows, list) or len(rows) != len(before)):
                    raise P11IntegrityError(f"{category}/seed{seed}: C64 transport vector dimensions differ")
                if not len(before) or not np.array_equal(initial, before[0]):
                    raise P11IntegrityError(f"{category}/seed{seed}: Stage 0.5 initial C64 differs from first linked transport vector")
                if float(higher_quantile(initial.tolist(), .90)) != float(higher_quantile(before[0].tolist(), .90)):
                    raise P11IntegrityError(f"{category}/seed{seed}: linked initial C64 q90 differs")
                initial_vectors += 1
                for index, (pre, post, row) in enumerate(zip(before, after, rows)):
                    if (_sha_json(pre.tolist()) != row.get("c64_before_sha256") or
                            _sha_json(post.tolist()) != row.get("c64_after_sha256")):
                        raise P11IntegrityError(f"{category}/seed{seed}/update{index}: C64 vector digest mismatch")
                    if float(higher_quantile(post.tolist(), .90)) != float(row.get("theta_SCQ_candidate_higher_q90")):
                        raise P11IntegrityError(f"{category}/seed{seed}/update{index}: stored post-C64 q90 mismatch")
                    if row.get("n_c64") != 64 or row.get("memory_version_after") != row.get("memory_version_before", -1) + 1:
                        raise P11IntegrityError(f"{category}/seed{seed}/update{index}: stored memory/C64 transition metadata invalid")
                    c64_rows += 1
                update_vectors += len(before)
        verified_files.extend([f"{category}/scores.npz", f"{category}/c64_transport.json"])
    return {
        "status": "PASS_HISTORICAL_ARTIFACT_LINKAGE_ONLY",
        "historical_native_c64_rescoring_performed": False,
        "stage05_score_tape_manifest_sha256": stage05_manifest_sha,
        "stage075a_audit_linkage_sha256": linkage_sha,
        "stage075a_protocol_manifest_sha256": protocol_sha,
        "class_seed_initial_c64_vectors_verified": initial_vectors,
        "postwrite_c64_vectors_verified": update_vectors,
        "c64_transport_rows_verified": c64_rows,
        "verified_file_roles": verified_files,
        "historical_stage075a_incomplete_marker_present": (stage075a_output / "INCOMPLETE").exists(),
        "original_g2_closed": False,
        "independent_numerical_c64_parity": "NOT_ESTABLISHED; no new Native Scorer call was made",
    }


def p11_source_identity(root: Path) -> dict[str, Any]:
    """Compact source provenance for the adapter and unchanged dependencies."""
    from .stage1a_p1_reconstruction import source_fingerprints
    allowlist = [
        "experiments/calibration/stage1a_p11_production.py",
        "experiments/calibration/stage1a_p1_reconstruction.py",
        "experiments/calibration/stage05.py",
        "experiments/calibration/stage05_runner.py",
        "experiments/calibration/stage075a_runner.py",
        "experiments/calibration/native.py",
        "experiments/calibration/anchor_ledger.py",
        "experiments/calibration/memory_state.py",
        "experiments/calibration/query_tape.py",
        "experiments/calibration/protocol.py",
        "experiments/calibration/profiles.py",
        "experiments/calibration/engine.py",
        "experiments/calibration/controllers.py",
        "experiments/calibration/scdac_g.py",
        "experiments/calibration/stage075b_controllers.py",
        "experiments/risk_safe/native_bridge.py",
        "experiments/risk_safe/shadow_memory.py",
        "experiments/v02_memory.py",
        "test_phase3_AIF_simplified_v22.py",
        "vmb_visionad_new.py",
    ]
    return source_fingerprints(root, allowlist)
