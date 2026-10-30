"""Stage 1-A controller-only replay preparation and readiness audit.

The adapter reads frozen Stage 0.75-A scores/transitions and verified P1.2
queried-anchor evidence; it never initializes a Native Scorer or recomputes
features, scores, or memory. Readiness validation uses deterministic
synthetic fixtures plus a bounded first-query real-input smoke. It does not
produce full-stream predictions or performance metrics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .controllers import (
    AdaptedDualACI, FrozenTheta, FrozenThresholdFallback,
    OriginalGuardedSidecar, RollingQ90QueriedNormal,
    StateConsistentQuantile,
)
from .metrics import PrivateOutcomeLedger, summarize_prediction_modes
from .protocol import EvidenceAnchor, EvidenceSnapshot, Feedback, Observation, Prediction
from .profiles import canonical_json
from .scdac_g import SCDACGConfig, SCDACGController
from .spot_dspot import DSPOTController, DSPOT_VERSION, POTConfig, SPOTController
from .stage075b_controllers import Stage075BOffset


ROOT = Path(__file__).resolve().parents[2]
BRANCH = "calibration-stage0-contract-20261029"
BASE_COMMIT = "89d5f9c3cf28c51ca8808cfc978b847b01fca922"
CLASSES = ("audiojack", "regulator", "switch", "rolled_strip_base")
SEEDS = (0, 1, 2, 3, 4)
TAPE_BINDING_SHA256 = "dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183"
STAGE075A = ROOT / "outputs/calibration/stage075a_greedy_ranking_repaired3_20261029"
P12_OUTPUT = ROOT / "outputs/calibration/stage1a_p12_low_memory_recovery_v3_20261030"
P12_AUDIT = ROOT / "docs/calibration/audit/stage1a_p12_low_memory_recovery_v3_20261030"
STAGE05 = ROOT / "outputs/calibration/stage05_exploratory_20261029"
AUDIT_DIR = ROOT / "docs/calibration/audit/stage1a_final_readiness_v6_20261030"


class ReadinessError(RuntimeError):
    """Frozen input or causal replay did not satisfy the registered contract."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReadinessError(f"Could not read JSON artifact {path.name}: {exc}") from exc


def _assert_hash(path: Path, expected: str, name: str) -> str:
    if not path.is_file():
        raise ReadinessError(f"Required frozen artifact is missing: {name}")
    actual = sha256_file(path)
    if actual != expected:
        raise ReadinessError(f"Frozen artifact checksum mismatch: {name}")
    return actual


@dataclass(frozen=True)
class MemoryIdentity:
    version: int
    memory_sha256: str
    source_map_sha256: str
    source_revision: int
    legacy_greedy_version: int


@dataclass(frozen=True)
class ReplayEvent:
    """One immutable frozen event; label is present only at a query position."""

    t: int
    observation: Observation
    queried: bool
    queried_label: int | None
    memory_before: MemoryIdentity
    memory_after: MemoryIdentity
    evidence_after: EvidenceSnapshot | None = None
    expected_evidence_sha256: str | None = None
    expected_c64_transport_sha256: str | None = None
    category: str | None = None
    support_seed: int | None = None
    expected_query_anchor_id: str | None = None
    query_source_identity_sha256: str | None = None
    static_score: float | None = None
    writer_record: Mapping[str, Any] | None = None


_DIRECT_FEEDBACK = frozenset(("B1", "B2", "B4"))
_FULL_EVIDENCE = frozenset(("B4", "B7", "Stage075BOffset"))
_C64_ONLY_EVIDENCE = frozenset(("B3",))
_SCORE_ONLY = frozenset(("B5a", "B5a_FROZEN_FALLBACK",
                         "B5b", "B5b_FROZEN_FALLBACK"))


def _filtered_evidence(evidence: EvidenceSnapshot, method_id: str) -> EvidenceSnapshot | None:
    if method_id in _FULL_EVIDENCE:
        return copy.deepcopy(evidence)
    if method_id not in _C64_ONLY_EVIDENCE:
        return None
    protected = tuple(copy.deepcopy(anchor) for anchor in evidence.normal_anchors
                      if anchor.protected)
    audit = copy.deepcopy(dict(evidence.source_exclusion_audit))
    audit["consumer_view"] = "protected_c64_only"
    return EvidenceSnapshot(evidence.memory_version, protected, (), audit,
                            evidence.evidence_revision)


def _check_evidence(evidence: EvidenceSnapshot, memory: MemoryIdentity, *,
                    event_position: int | None, expected_sha256: str | None,
                    expected_c64_sha256: str | None,
                    category: str | None = None, support_seed: int | None = None) -> None:
    audit = evidence.source_exclusion_audit
    required = ("memory_sha256", "source_map_sha256", "c64_transport_sha256")
    if any(not isinstance(audit.get(field), str) for field in required):
        raise ReadinessError("Evidence is missing memory/source/C64 identity")
    if (evidence.memory_version != memory.version or
            audit.get("memory_version", evidence.memory_version) != memory.version or
            audit["memory_sha256"] != memory.memory_sha256 or
            audit["source_map_sha256"] != memory.source_map_sha256 or
            audit.get("source_revision", memory.source_revision) != memory.source_revision or
            audit.get("legacy_greedy_version", memory.legacy_greedy_version) !=
            memory.legacy_greedy_version):
        raise ReadinessError("Evidence identity does not match the recorded post-event memory")
    if expected_sha256 is not None and audit.get("evidence_sha256") != expected_sha256:
        raise ReadinessError("Evidence content identity differs from its frozen revision")
    if (expected_c64_sha256 is not None and
            audit["c64_transport_sha256"] != expected_c64_sha256):
        raise ReadinessError("Evidence C64 transport differs from the frozen revision")
    if event_position is not None and audit.get("event_position") != event_position:
        raise ReadinessError("Evidence was delivered at the wrong event boundary")
    if category is not None and audit.get("category") != category:
        raise ReadinessError("Evidence category identity differs from this trajectory")
    if support_seed is not None and audit.get("support_seed") != support_seed:
        raise ReadinessError("Evidence support-seed identity differs from this trajectory")
    c64 = [anchor for anchor in evidence.normal_anchors if anchor.protected]
    if c64 and len(c64) != 64:
        raise ReadinessError("Evidence has an incomplete protected C64 vector")
    if len({anchor.opaque_anchor_id for anchor in evidence.normal_anchors +
            evidence.defect_anchors}) != len(evidence.normal_anchors + evidence.defect_anchors):
        raise ReadinessError("Evidence contains duplicate anchor identities")


def verify_recorded_writer_transition(event: ReplayEvent) -> None:
    """Check a frozen writer record against adjacent memory identities."""
    record = event.writer_record
    if record is None:
        return
    required = {
        "eligible", "changed", "keep", "reason", "appended", "replaced", "skipped",
        "source_map_changed", "source_revision_before", "source_revision_after",
        "memory_content_version_before", "memory_content_version_after",
        "legacy_greedy_version_before", "legacy_greedy_version_after",
        "source_map_sha256", "dynamic_rows_by_layer", "capacity_by_layer",
        "resident_source_groups_by_layer_sha256",
    }
    if set(record) != required:
        raise ReadinessError("Frozen writer record does not match the historical schema")
    if (not isinstance(record["eligible"], bool) or
            not isinstance(record["changed"], bool) or
            not isinstance(record["source_map_changed"], bool) or
            not isinstance(record["reason"], str) or not record["reason"] or
            not re.fullmatch(r"[0-9a-f]{64}", str(record["source_map_sha256"]))):
        raise ReadinessError("Frozen writer action or source-map identity is malformed")
    before, after = event.memory_before, event.memory_after
    identity_fields = (
        ("memory_content_version_before", before.version),
        ("memory_content_version_after", after.version),
        ("legacy_greedy_version_before", before.legacy_greedy_version),
        ("legacy_greedy_version_after", after.legacy_greedy_version),
        ("source_revision_before", before.source_revision),
        ("source_revision_after", after.source_revision),
        ("source_map_sha256", after.source_map_sha256),
    )
    if any(record.get(field) != expected for field, expected in identity_fields):
        raise ReadinessError("Frozen writer record differs from the recorded memory transition")
    if record["source_map_changed"] != (before.source_map_sha256 != after.source_map_sha256):
        raise ReadinessError("Frozen writer source-map action differs from adjacent identities")
    if bool(record["changed"]) != (before.memory_sha256 != after.memory_sha256):
        raise ReadinessError("Frozen writer action and memory-content transition disagree")
    for field in ("keep", "appended", "replaced", "skipped"):
        if isinstance(record[field], bool) or not isinstance(record[field], int) or record[field] < 0:
            raise ReadinessError(f"Frozen writer count {field} is invalid")
    rows = record["dynamic_rows_by_layer"]
    capacities = record["capacity_by_layer"]
    resident = record["resident_source_groups_by_layer_sha256"]
    if (not isinstance(rows, list) or not isinstance(capacities, list) or
            not isinstance(resident, list) or not rows or
            len(rows) != len(capacities) or len(rows) != len(resident)):
        raise ReadinessError("Frozen writer layer statistics have inconsistent shapes")
    for row_count, capacity, groups in zip(rows, capacities, resident):
        if (isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0 or
                isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0 or
                not isinstance(groups, Mapping)):
            raise ReadinessError("Frozen writer layer statistics are malformed")
        if any(not re.fullmatch(r"[0-9a-f]{64}", str(group_sha)) or
               isinstance(count, bool) or not isinstance(count, int) or count <= 0
               for group_sha, count in groups.items()):
            raise ReadinessError("Frozen writer resident-source statistics are malformed")


@dataclass(frozen=True)
class FrozenTrajectoryInput:
    """Verified, lazy source for one frozen class/support-seed trajectory."""

    category: str
    support_seed: int
    event_count: int
    query_positions: tuple[int, ...]
    query_response_chain_sha256: str
    initial_memory: MemoryIdentity
    initial_evidence: EvidenceSnapshot
    _event_factory: Any

    def events(self):
        return self._event_factory()


def load_frozen_trajectory_input(*, category: str, support_seed: int,
                                 receipt: Mapping[str, Any],
                                 root: Path = ROOT) -> FrozenTrajectoryInput:
    """Open only frozen scores/transitions and P1.2 queried-anchor evidence.

    No Native Scorer, DINOv2, Greedy writer, unqueried label, mask, or outcome
    artifact is accessed. P1.2 evidence is checked with its original verifier;
    labels are derived only from the newly added queried anchor at a taped event.
    """
    import gzip
    import numpy as np

    from .stage1a_p11_production import verify_stage05_qats_artifacts
    from .stage1a_p12_sharded_recovery import verify_evidence_rows

    if category not in CLASSES or support_seed not in SEEDS:
        raise ReadinessError("Frozen trajectory identity is outside the four-class/seed registry")
    if (receipt.get("category") != category or
            receipt.get("support_seed") != support_seed or
            receipt.get("status") != "PASS_EXACT_PARITY" or
            receipt.get("complete") is not True or
            receipt.get("query_tape_binding_sha256") != TAPE_BINDING_SHA256):
        raise ReadinessError("Trajectory receipt identity or completion status differs")

    tapes = verify_stage05_qats_artifacts(STAGE05)
    positions = tuple(int(position) for position in tapes.positions(category))
    if (len(set(positions)) != len(positions) or tuple(sorted(positions)) != positions or
            receipt.get("query_count") != len(positions) or
            receipt.get("query_positions_sha256") != sha256_json(list(positions))):
        raise ReadinessError("Trajectory receipt does not bind to the verified Q-A tape positions")
    query_groups = tapes.group_identity_by_class_position[category]

    evidence_path = root / str(receipt.get("evidence_file", ""))
    evidence_sha = _assert_hash(evidence_path,
                                str(receipt.get("evidence_file_sha256", "")),
                                "P1.2 queried-anchor evidence")
    evidence_rows = []
    with evidence_path.open("rb") as stream:
        for line in stream:
            row = json.loads(line.decode("utf-8"))
            if not isinstance(row, dict):
                raise ReadinessError("P1.2 evidence row is not an object")
            evidence_rows.append(row)
    verification = verify_evidence_rows(
        evidence_rows, category=category, query_positions=positions,
        scorer_config_sha256=str(receipt["scorer_config_sha256"]),
        scorer_source_sha256=str(receipt["scorer_source_sha256"]))
    if (verification.get("all_evidence_hashes_pass") is not True or
            len(evidence_rows) != len(positions) + 1 or
            evidence_sha != receipt.get("evidence_file_sha256")):
        raise ReadinessError("P1.2 evidence shard failed its independent verifier")
    if [row.get("position") for row in evidence_rows] != [None, *positions]:
        raise ReadinessError("P1.2 evidence revisions are missing, duplicate, or out of query order")

    folder = STAGE075A / "trajectories" / category
    checksum_entries = _read_json(folder / "checksums.json").get("files", {})
    safe_names = ("trajectory_manifest.json", "scores.npz", "event_trace.jsonl.gz",
                  "c64_transport.json")
    for name in safe_names:
        _assert_hash(folder / name, str(checksum_entries.get(name, "")),
                     f"Stage 0.75-A {category}/{name}")
    trajectory_meta = _read_json(folder / "trajectory_manifest.json")
    event_count = int(trajectory_meta.get("positions_scored", -1))
    if (trajectory_meta.get("category") != category or
            trajectory_meta.get("smoke_only") is not False or
            trajectory_meta.get("stage1_or_stage2_run") is not False or
            receipt.get("event_count") != event_count or
            receipt.get("expected_event_count") != event_count):
        raise ReadinessError("Frozen trajectory length differs from its P1.2 receipt")

    first_row = evidence_rows[0]
    initial_memory = MemoryIdentity(
        int(first_row["memory_version"]), str(first_row["memory_sha256"]),
        str(first_row["source_map_sha256"]),
        int(first_row["source_exclusion_audit"]["source_revision"]),
        int(first_row["source_exclusion_audit"]["legacy_greedy_version"]))
    receipt_initial = MemoryIdentity(**receipt["initial_memory"])
    if initial_memory != receipt_initial:
        raise ReadinessError("Initial P1.2 evidence identity differs from trajectory receipt")
    initial_evidence = p12_row_to_evidence_snapshot(
        first_row, category=category, support_seed=support_seed,
        expected_position=None)
    _check_evidence(initial_evidence, initial_memory, event_position=None,
                    expected_sha256=first_row["evidence_sha256"],
                    expected_c64_sha256=first_row["source_exclusion_audit"][
                        "c64_transport_sha256"], category=category,
                    support_seed=support_seed)
    del evidence_rows

    stage05_score_manifest = _read_json(STAGE05 / "score_tape_manifest.json")
    score_tape = stage05_score_manifest["classes"][category]["score_tapes"][str(support_seed)]
    fixed_support_sha = str(score_tape["fixed_support_sha256"])
    scorer_config_sha = str(receipt["scorer_config_sha256"])

    def event_iterator():
        current = initial_memory
        previous_anchor_ids = {
            item["opaque_anchor_id"] for item in
            first_row.get("normal_anchors", ()) + first_row.get("defect_anchors", ())
            if str(item.get("opaque_anchor_id", "")).startswith("query:")}
        seen_queries: list[int] = []
        seen_events = 0
        query_position_set = set(positions)
        names = {
            "static": f"static_s{support_seed}",
            "dynamic": f"dynamic_s{support_seed}",
            "memory_version": f"memory_version_before_s{support_seed}",
            "memory_sha": f"memory_sha_before_s{support_seed}",
            "q90": f"current_q90_s{support_seed}",
        }
        _assert_hash(evidence_path, evidence_sha, "P1.2 queried-anchor evidence before replay")
        with evidence_path.open("rb") as evidence_stream:
            def evidence_iter():
                for line in evidence_stream:
                    try:
                        item = json.loads(line.decode("utf-8"))
                    except Exception as exc:
                        raise ReadinessError("P1.2 evidence changed to invalid JSON during replay") from exc
                    if not isinstance(item, dict):
                        raise ReadinessError("P1.2 evidence changed to a non-object row")
                    yield item
            evidence_rows_live = iter(evidence_iter())
            live_initial = next(evidence_rows_live, None)
            if live_initial != first_row:
                raise ReadinessError("P1.2 initial evidence changed after read-only verification")
            with np.load(folder / "scores.npz", allow_pickle=False) as score_arrays:
                if any(name not in score_arrays.files for name in names.values()):
                    raise ReadinessError("Frozen score tape lacks a required seed-specific array")
                with gzip.open(folder / "event_trace.jsonl.gz", "rt", encoding="utf-8") as trace:
                    for expected_t, line in enumerate(trace):
                        try:
                            raw = json.loads(line)
                        except Exception as exc:
                            raise ReadinessError("Frozen event trace contains invalid JSON") from exc
                        t = int(raw.get("position", -1))
                        if t != expected_t or t >= event_count:
                            raise ReadinessError("Frozen event trace positions are missing or out of order")
                        queried = t in query_position_set
                        if (bool(raw.get("query_tape_selected")) != queried or
                                bool(raw.get("queried_label_disclosed")) != queried or
                                raw.get("oracle_label_request_position") != (t if queried else None)):
                            raise ReadinessError("Frozen event query boundary differs from verified tape")
                        if queried and raw.get("official_group_sha256") != query_groups.get(t):
                            raise ReadinessError("Frozen event official group differs from verified Q-A tape")
                        seed_row = raw.get("support_seeds", {}).get(str(support_seed))
                        if not isinstance(seed_row, Mapping):
                            raise ReadinessError("Frozen event lacks the requested support-seed row")
                        static_score = float(score_arrays[names["static"]][t])
                        dynamic_score = float(score_arrays[names["dynamic"]][t])
                        if (static_score != float(seed_row["static_score"]) or
                                dynamic_score != float(seed_row["dynamic_score"]) or
                                float(score_arrays[names["q90"]][t]) !=
                                float(seed_row["current_C64_q90_pre_event"])):
                            raise ReadinessError("Frozen NPZ scores differ from the event trace")
                        pre = MemoryIdentity(
                            int(score_arrays[names["memory_version"]][t]),
                            str(score_arrays[names["memory_sha"]][t]),
                            current.source_map_sha256, current.source_revision,
                            current.legacy_greedy_version)
                        if (int(seed_row["memory_version_before"]) != pre.version or
                                str(seed_row["memory_sha256_before"]) != pre.memory_sha256 or
                                pre != current):
                            raise ReadinessError("Frozen pre-event score/memory identity is discontinuous")
                        writer = seed_row.get("writer")
                        if not isinstance(writer, Mapping):
                            raise ReadinessError("Frozen event is missing its historical writer record")
                        post = MemoryIdentity(
                            int(seed_row["memory_version_after"]),
                            str(seed_row["memory_sha256_after"]),
                            str(writer["source_map_sha256"]),
                            int(writer["source_revision_after"]),
                            int(writer["legacy_greedy_version_after"]))
                        obs_id = hashlib.sha256(
                            f"Q-A-TS-v1:{t}:{int(raw['camera_id'])}".encode("utf-8")
                        ).hexdigest()[:24]
                        score_digest = sha256_json({
                            "memory": pre.memory_sha256, "scorer": scorer_config_sha,
                            "fixed_support": fixed_support_sha,
                            "kind": "raw_native_score_bank"})
                        observation = Observation(t, obs_id, int(raw["camera_id"]),
                                                  dynamic_score, pre.version, score_digest)
                        query_label = None
                        evidence = None
                        expected_evidence_sha = None
                        expected_c64_sha = None
                        query_anchor_id = None
                        if queried:
                            row = next(evidence_rows_live, None)
                            if row is None or row.get("position") != t:
                                raise ReadinessError("P1.2 evidence revision missing at the taped query")
                            ids_now = {item["opaque_anchor_id"] for item in
                                       row.get("normal_anchors", ()) + row.get("defect_anchors", ())
                                       if str(item.get("opaque_anchor_id", "")).startswith("query:")}
                            added = ids_now - previous_anchor_ids
                            if len(added) != 1 or not previous_anchor_ids.issubset(ids_now):
                                raise ReadinessError("Queried evidence does not add exactly one anchor")
                            query_anchor_id = next(iter(added))
                            normal_ids = {item["opaque_anchor_id"]
                                          for item in row.get("normal_anchors", ())}
                            defect_ids = {item["opaque_anchor_id"]
                                          for item in row.get("defect_anchors", ())}
                            if query_anchor_id in normal_ids and query_anchor_id not in defect_ids:
                                query_label = 0
                            elif query_anchor_id in defect_ids and query_anchor_id not in normal_ids:
                                query_label = 1
                            else:
                                raise ReadinessError("Queried anchor has no unique normal/defect side")
                            evidence = p12_row_to_evidence_snapshot(
                                row, category=category, support_seed=support_seed,
                                expected_position=t)
                            expected_evidence_sha = str(row["evidence_sha256"])
                            expected_c64_sha = str(row["source_exclusion_audit"][
                                "c64_transport_sha256"])
                            previous_anchor_ids = ids_now
                            seen_queries.append(t)
                        event = ReplayEvent(
                            t, observation, queried, query_label, pre, post,
                            evidence, expected_evidence_sha, expected_c64_sha,
                            category, support_seed, query_anchor_id,
                            (str(raw["official_group_sha256"]) if queried else None),
                            static_score, copy.deepcopy(dict(writer)))
                        verify_recorded_writer_transition(event)
                        current = post
                        seen_events += 1
                        yield event
            if next(evidence_rows_live, None) is not None:
                raise ReadinessError("P1.2 evidence shard contains an extra revision")
        if seen_events != event_count or seen_queries != list(positions):
            raise ReadinessError("Frozen trajectory ended before all taped events/evidence revisions")
        _assert_hash(evidence_path, evidence_sha, "P1.2 queried-anchor evidence after replay")
        if current != MemoryIdentity(**receipt["final_memory"]):
            raise ReadinessError("Frozen final memory differs from P1.2 trajectory receipt")

    return FrozenTrajectoryInput(
        category, support_seed, event_count, positions,
        str(receipt["query_response_chain_sha256"]), initial_memory,
        initial_evidence, event_iterator)


def p12_row_to_evidence_snapshot(row: Mapping[str, Any], *, category: str,
                                 support_seed: int,
                                 expected_position: int | None) -> EvidenceSnapshot:
    """Convert one previously verified P1.2 JSONL row into a controller view.

    The caller must first validate the complete JSONL shard with the P1.2
    `verify_evidence_rows` implementation and bind its file SHA to the signed
    trajectory receipt. This conversion performs boundary checks and adds the
    category/seed/event identity required by the controller-only adapter.
    """
    if row.get("schema") != "stage1a.p12_private_evidence_snapshot.v1":
        raise ReadinessError("Unexpected P1.2 evidence snapshot schema")
    if row.get("position") != expected_position:
        raise ReadinessError("P1.2 evidence row position differs from its query boundary")
    audit = row.get("source_exclusion_audit")
    if not isinstance(audit, Mapping):
        raise ReadinessError("P1.2 evidence row has no source-exclusion audit")
    if (row.get("memory_version") != audit.get("memory_version") or
            row.get("memory_sha256") != audit.get("memory_sha256") or
            row.get("source_map_sha256") != audit.get("source_map_sha256") or
            row.get("evidence_revision") != audit.get("evidence_revision") or
            row.get("evidence_sha256") != audit.get("evidence_sha256")):
        raise ReadinessError("P1.2 evidence duplicate identity fields disagree")
    required_audit = ("source_revision", "legacy_greedy_version", "c64_transport_sha256")
    if any(key not in audit for key in required_audit):
        raise ReadinessError("P1.2 evidence lacks source/C64 provenance")

    def anchors(items: Sequence[Mapping[str, Any]]) -> tuple[EvidenceAnchor, ...]:
        output = []
        for item in items:
            try:
                output.append(EvidenceAnchor(
                    opaque_anchor_id=str(item["opaque_anchor_id"]),
                    score=float(item["score"]),
                    memory_version=int(item["memory_version"]),
                    protected=bool(item["protected"]),
                    source_excluded_rows=int(item.get("source_excluded_rows", 0)),
                    source_exclusion_policy=str(item.get(
                        "source_exclusion_policy", "protected_no_selfwrite"))))
            except (KeyError, TypeError, ValueError) as exc:
                raise ReadinessError("P1.2 evidence anchor record is malformed") from exc
        return tuple(output)

    normal = anchors(row.get("normal_anchors", ()))
    defect = anchors(row.get("defect_anchors", ()))
    if sum(anchor.protected for anchor in normal) != 64:
        raise ReadinessError("P1.2 evidence must carry the full protected C64 anchor set")
    bound_audit = copy.deepcopy(dict(audit))
    bound_audit.update({"category": category, "support_seed": int(support_seed),
                        "event_position": expected_position,
                        "evidence_sha256": str(row["evidence_sha256"])})
    return EvidenceSnapshot(int(row["memory_version"]), normal, defect,
                            bound_audit, int(row["evidence_revision"]))


class ControllerOnlyReplay:
    """Causal, no-writer replay adapter over precomputed event records.

    It calls predict before disclosure, provides direct queried feedback only
    to registered consumers, applies no memory operation, and accepts only the
    recorded post-event memory identity/evidence revision.
    """

    def __init__(self, *, controllers: Mapping[str, Any],
                 query_positions: Iterable[int], input_binding_sha256: str,
                 initial_memory: MemoryIdentity,
                 initial_evidence: EvidenceSnapshot | None = None):
        if not controllers or len(set(controllers)) != len(controllers):
            raise ReadinessError("Controller registry must be nonempty and unique")
        if not re.fullmatch(r"[0-9a-f]{64}", input_binding_sha256):
            raise ReadinessError("Replay requires a SHA-256 input binding")
        self.controllers = dict(controllers)
        self.query_positions = frozenset(int(t) for t in query_positions)
        self.input_binding_sha256 = input_binding_sha256
        self.cursor = 0
        self.last_memory = initial_memory
        self.last_evidence_revision = -1
        self.last_evidence_sha256: str | None = None
        self.query_response_chain_sha256 = hashlib.sha256(
            b"stage1a-p1-query-response-v1").hexdigest()
        self.predictions: dict[str, list[Prediction]] = {key: [] for key in controllers}
        self.trace: list[dict[str, Any]] = []
        if initial_evidence is not None:
            _check_evidence(initial_evidence, initial_memory, event_position=None,
                            expected_sha256=None, expected_c64_sha256=None)
            self._deliver_evidence(initial_evidence)
            self.last_evidence_revision = initial_evidence.evidence_revision
            self.last_evidence_sha256 = str(
                initial_evidence.source_exclusion_audit.get("evidence_sha256", ""))

    def _deliver_evidence(self, evidence: EvidenceSnapshot) -> None:
        for method_id, controller in self.controllers.items():
            selected = _filtered_evidence(evidence, method_id)
            if selected is not None:
                controller.on_memory_change(selected)

    def step(self, event: ReplayEvent) -> dict[str, Prediction]:
        t = self.cursor
        if event.t != t or event.observation.t != t:
            raise ReadinessError("Replay events must be contiguous, zero-based, and ordered")
        if event.memory_before != self.last_memory:
            raise ReadinessError("Pre-event memory identity differs from recorded trajectory")
        if event.observation.memory_version != event.memory_before.version:
            raise ReadinessError("Observation score was not computed from pre-event memory")
        if event.queried != (t in self.query_positions):
            raise ReadinessError("Query flag differs from the frozen Q-A-TS tape")
        if event.queried and event.queried_label not in (0, 1):
            raise ReadinessError("Frozen query event is missing its queried response")
        if not event.queried and event.queried_label is not None:
            raise ReadinessError("An unqueried label entered the controller replay")
        before_predictions: dict[str, Prediction] = {}
        for method_id, controller in self.controllers.items():
            before_state = canonical_json(controller.state_dict())
            prediction = controller.predict(copy.deepcopy(event.observation))
            if canonical_json(controller.state_dict()) != before_state:
                raise ReadinessError(f"{method_id}.predict changed controller state")
            before_predictions[method_id] = prediction
            self.predictions[method_id].append(prediction)

        before, after = event.memory_before, event.memory_after
        verify_recorded_writer_transition(event)
        if after.version < before.version or after.version > before.version + 1:
            raise ReadinessError("Recorded memory version transition is invalid")
        if ((after.version == before.version) !=
                (after.memory_sha256 == before.memory_sha256)):
            raise ReadinessError("Recorded memory version/content transition is inconsistent")
        if (event.evidence_after is None) != (event.expected_evidence_sha256 is None):
            raise ReadinessError("Evidence revision and its identity must be supplied together")
        if event.evidence_after is not None:
            evidence = event.evidence_after
            if not event.queried and after == before:
                raise ReadinessError("Evidence revision appeared without a query/write boundary")
            if evidence.evidence_revision <= self.last_evidence_revision:
                raise ReadinessError("Evidence revision is duplicate or out of order")
            _check_evidence(evidence, after, event_position=t,
                            expected_sha256=event.expected_evidence_sha256,
                            expected_c64_sha256=event.expected_c64_transport_sha256,
                            category=event.category, support_seed=event.support_seed)
            if event.queried:
                if not event.expected_query_anchor_id:
                    raise ReadinessError("Queried evidence lacks its exact anchor identity")
                observed_ids = [anchor.opaque_anchor_id
                                for anchor in evidence.normal_anchors + evidence.defect_anchors]
                if observed_ids.count(event.expected_query_anchor_id) != 1:
                    raise ReadinessError("Evidence does not contain exactly one matching query anchor")
                if (event.queried_label == 0 and
                        event.expected_query_anchor_id not in
                        {anchor.opaque_anchor_id for anchor in evidence.normal_anchors}):
                    raise ReadinessError("Queried normal anchor is absent from the normal evidence side")
                if (event.queried_label == 1 and
                        event.expected_query_anchor_id not in
                        {anchor.opaque_anchor_id for anchor in evidence.defect_anchors}):
                    raise ReadinessError("Queried defect anchor is absent from the defect evidence side")
            self._deliver_evidence(evidence)
            self.last_evidence_revision = evidence.evidence_revision
            self.last_evidence_sha256 = event.expected_evidence_sha256
        elif event.queried:
            raise ReadinessError("Queried event lacks its causal evidence revision")
        elif after != before:
            raise ReadinessError("Recorded memory transition lacks its causal evidence revision")

        # Feedback and the current unlabeled score update controller state only
        # after the shared writer transition and matching evidence revision.
        for method_id, controller in self.controllers.items():
            feedback = (Feedback(t, True, int(event.queried_label))
                        if event.queried and method_id in _DIRECT_FEEDBACK
                        else Feedback(t, False, None))
            controller.observe_feedback(copy.deepcopy(event.observation),
                                        copy.deepcopy(before_predictions[method_id]), feedback)
            controller.observe_score(copy.deepcopy(event.observation))

        post = {method_id: controller.finalize_step()
                for method_id, controller in self.controllers.items()}
        self.last_memory = after
        if event.queried:
            source_identity = (event.query_source_identity_sha256 or hashlib.sha256(
                event.observation.observation_id.encode("utf-8")).hexdigest())
            if not re.fullmatch(r"[0-9a-f]{64}", source_identity):
                raise ReadinessError("Queried response lacks its official source identity")
            response = {"position": t,
                        "source_identity_sha256": source_identity,
                        "label": int(event.queried_label)}
            self.query_response_chain_sha256 = sha256_bytes(
                bytes.fromhex(self.query_response_chain_sha256) +
                canonical_json(response).encode("ascii"))
        row = {
            "t": t,
            "query_event": event.queried,
            "memory_version_before": before.version,
            "memory_version_after": after.version,
            "memory_sha256_before": before.memory_sha256,
            "memory_sha256_after": after.memory_sha256,
            "prediction_pre": {k: v.label for k, v in before_predictions.items()},
            "prediction_available": {k: v.available for k, v in before_predictions.items()},
            "prediction_reason": {k: v.reason for k, v in before_predictions.items()},
            "threshold_post": post,
            "evidence_revision": self.last_evidence_revision,
        }
        self.trace.append(row)
        self.cursor += 1
        return before_predictions

    def state_dict(self) -> dict[str, Any]:
        """Small checkpoint: controller state and digests, never response labels."""
        return copy.deepcopy({
            "schema": "stage1a.controller_only_replay_checkpoint.v1",
            "input_binding_sha256": self.input_binding_sha256,
            "query_positions": sorted(self.query_positions),
            "cursor": self.cursor,
            "last_memory": asdict(self.last_memory),
            "last_evidence_revision": self.last_evidence_revision,
            "last_evidence_sha256": self.last_evidence_sha256,
            "query_response_chain_sha256": self.query_response_chain_sha256,
            "predictions": {k: [asdict(x) for x in v]
                            for k, v in self.predictions.items()},
            "controllers": {k: v.state_dict() for k, v in self.controllers.items()},
            "trace": self.trace,
        })

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (state.get("schema") != "stage1a.controller_only_replay_checkpoint.v1" or
                state.get("input_binding_sha256") != self.input_binding_sha256 or
                state.get("query_positions") != sorted(self.query_positions) or
                set(state.get("controllers", {})) != set(self.controllers)):
            raise ReadinessError("Checkpoint input/controller identity differs")
        cursor = state.get("cursor")
        if not isinstance(cursor, int) or cursor < 0 or len(state.get("trace", ())) != cursor:
            raise ReadinessError("Checkpoint cursor/trace is inconsistent")
        if any(len(state.get("predictions", {}).get(k, ())) != cursor
               for k in self.controllers):
            raise ReadinessError("Checkpoint prediction history differs from cursor")
        for key, controller in self.controllers.items():
            controller.load_state_dict(copy.deepcopy(state["controllers"][key]))
        self.cursor = cursor
        self.last_memory = MemoryIdentity(**state["last_memory"])
        self.last_evidence_revision = int(state["last_evidence_revision"])
        self.last_evidence_sha256 = state["last_evidence_sha256"]
        self.query_response_chain_sha256 = str(state["query_response_chain_sha256"])
        self.predictions = {
            key: [Prediction(**row) for row in values]
            for key, values in state["predictions"].items()}
        self.trace = copy.deepcopy(list(state["trace"]))

    def verify_query_response_chain(self, expected_sha256: str) -> str:
        """Verify a completed causal disclosure chain against its P1.2 receipt."""
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ReadinessError("Expected query-response chain is not a SHA-256 identity")
        if self.query_positions and self.cursor <= max(self.query_positions):
            raise ReadinessError("Cannot verify a query-response chain before all taped events")
        if self.query_response_chain_sha256 != expected_sha256:
            raise ReadinessError("Causal query responses differ from the P1.2 receipt chain")
        return self.query_response_chain_sha256


def build_controller_suite(theta0: float, c64_scores: Sequence[float]) -> dict[str, Any]:
    """Instantiate the frozen executable registry; B6/B8 remain nonexecuted."""
    if len(c64_scores) != 64:
        raise ReadinessError("The frozen registry requires exactly 64 C64 scores")
    c64 = [float(value) for value in c64_scores]
    spot = SPOTController(POTConfig())
    spot_fallback_source = SPOTController(POTConfig())
    dspot = DSPOTController(POTConfig(version=DSPOT_VERSION), window=50)
    dspot_fallback_source = DSPOTController(
        POTConfig(version=DSPOT_VERSION), window=50)
    return {
        "B0": FrozenTheta(theta0),
        "B1": RollingQ90QueriedNormal(theta0, window=256, min_count=16),
        "B2": OriginalGuardedSidecar(c64, theta0),
        "B3": StateConsistentQuantile(theta0, delta=0.01, min_normals=16),
        "B4": AdaptedDualACI(theta0, alpha_target=0.10, beta_target=0.10,
                             gamma_normal=0.005, gamma_defect=0.005),
        "B5a": spot,
        "B5a_FROZEN_FALLBACK": FrozenThresholdFallback(
            spot_fallback_source, theta0, name="B5a_FROZEN_FALLBACK"),
        "B5b": dspot,
        "B5b_FROZEN_FALLBACK": FrozenThresholdFallback(
            dspot_fallback_source, theta0, name="B5b_FROZEN_FALLBACK"),
        "B7": SCDACGController(theta0, SCDACGConfig()),
        "Stage075BOffset": Stage075BOffset(theta0, name="Stage075BOffset"),
    }


def synthetic_metric_contract(labels: Sequence[int], scores: Sequence[float],
                              predictions: Mapping[str, Sequence[int | None]], *,
                              availability: Mapping[str, Sequence[bool]] | None = None,
                              reasons: Mapping[str, Sequence[str]] | None = None,
                              horizon_start: int = 1050) -> dict[str, Any]:
    """Offline-only metric adapter; callers must pass synthetic fixtures here."""
    ledger = PrivateOutcomeLedger(labels)
    base = ledger.evaluate(scores, dict(predictions),
                           prediction_availability=dict(availability or {}))
    methods: dict[str, Any] = {}
    for name, stream in predictions.items():
        row = copy.deepcopy(base["methods"][name])
        ys = list(labels)
        av = list((availability or {}).get(name, [True] * len(stream)))
        if len(ys) != len(stream) or len(av) != len(stream):
            raise ReadinessError("Synthetic metric vectors differ in length")
        normal = [i for i, y in enumerate(ys) if y == 0]
        defect = [i for i, y in enumerate(ys) if y == 1]
        answered_normal = [i for i in normal if av[i] and stream[i] is not None]
        answered_defect = [i for i in defect if av[i] and stream[i] is not None]
        row["normal_coverage"] = (len(answered_normal) / len(normal) if normal else None)
        row["defect_coverage"] = (len(answered_defect) / len(defect) if defect else None)
        tn = sum(stream[i] == 0 for i in answered_normal)
        fp = sum(stream[i] == 1 for i in answered_normal)
        fn = sum(stream[i] == 0 for i in answered_defect)
        tp = sum(stream[i] == 1 for i in answered_defect)
        row["selective_fpr"] = fp / (tn + fp) if tn + fp else None
        row["selective_fnr"] = fn / (fn + tp) if fn + tp else None
        row["defect_recall"] = (None if row["fnr"] is None else 1.0 - row["fnr"])
        row["selective_balanced_accuracy"] = (
            1.0 - (row["selective_fpr"] + row["selective_fnr"]) / 2.0
            if row["selective_fpr"] is not None and row["selective_fnr"] is not None
            else None)
        row["full_256_image_windows_only"] = [
            item for item in row["windows"]["windows"] if item["n"] == 256]
        row["worst_defined_full_window_balanced_accuracy"] = min(
            (item["balanced_accuracy"] for item in row["full_256_image_windows_only"]
             if item["balanced_accuracy"] is not None), default=None)
        row["worst_defined_full_window_fpr"] = max(
            (item["fpr"] for item in row["full_256_image_windows_only"]
             if item["fpr"] is not None), default=None)
        row["worst_defined_full_window_fnr"] = max(
            (item["fnr"] for item in row["full_256_image_windows_only"]
             if item["fnr"] is not None), default=None)
        reason_rows = list((reasons or {}).get(name, [""] * len(stream)))
        if len(reason_rows) != len(stream):
            raise ReadinessError("Prediction reason vector differs in length")
        row["prediction_modes"] = summarize_prediction_modes(reason_rows)
        methods[name] = row
    horizon = (ledger.evaluate_common_horizon(
        scores, dict(predictions), start=horizon_start,
        prediction_availability=dict(availability or {}))
        if horizon_start <= len(labels) else {
            "status": "UNAVAILABLE_SHORT_SYNTHETIC_STREAM",
            "start": horizon_start, "stop_exclusive": len(labels),
            "n": 0, "metrics": None, "prediction_modes": {}})
    return {
        "n": base["n"], "n_normal": base["n_normal"], "n_defect": base["n_defect"],
        "methods": methods,
        "shared_score_auroc": base["shared_score_auroc"],
        "shared_score_auprc": base["shared_score_auprc"],
        "ranking_status": base["ranking_status"],
        "common_horizon": {"start_zero_based": horizon_start,
                           "result": horizon},
    }


def _method_registry() -> dict[str, Any]:
    from .method_registry import METHOD_CARDS

    rows = {name: copy.deepcopy(card) for name, card in METHOD_CARDS.items()
            if name in ("B0", "B1", "B2", "B3", "B4", "B5a", "B5b",
                        "B5a_FROZEN_FALLBACK", "B5b_FROZEN_FALLBACK", "B7")}
    rows["Stage075BOffset"] = {
        "name": "Stage075BOffset raw-median-offset control",
        "version": "stage075b-b3-raw-median-offset-v1",
        "status": "REGISTERED_DISTINCT_CONTROL",
        "equation": "desired=(median(current queried-normal scores)+median(current queried-defect scores))/2; if either side is unavailable, desired=own current theta; clip movement to +/-0.01",
        "configuration": {"max_step": 0.01,
                          "missing_class_side": "hold its own threshold",
                          "not_additive_b3_offset": True},
        "source": "experiments/calibration/stage075b_controllers.py::Stage075BOffset",
    }
    rows["B6"] = copy.deepcopy(METHOD_CARDS["B6"])
    rows["B6"].update({
        "name": "Sun-CS source-native applicability-only reference",
        "status": "APPLICABILITY_ONLY_NOT_EXECUTED_IN_STAGE1A_METHOD_COMPARISON",
        "source_target": "score-history quantile indicator, not physical-defect label",
        "reason": "Frozen protocol retains the reference without presenting it as a full-coverage binary detector.",
    })
    rows["B8"] = {
        "name": "Stage 0.75-B historical negative control",
        "status": "HISTORICAL_NEGATIVE_CONTROL_NOT_REEXECUTED",
        "source": "Existing Stage 0.75-B artifacts; no new controller run.",
    }
    return {
        "schema": "stage1a.frozen_method_registry.v1",
        "immutable": True,
        "execution_scope": ["B0", "B1", "B2", "B3", "B4", "B5a",
                            "B5a_FROZEN_FALLBACK", "B5b",
                            "B5b_FROZEN_FALLBACK", "B7", "Stage075BOffset"],
        "applicability_only": ["B6"],
        "historical_only": ["B8"],
        "methods": rows,
    }



P12_V2_REQUIRED_AUDIT_FILES = frozenset({
    "STAGE1A_P12_LOW_MEMORY_RECOVERY_REPORT.md",
    "STAGE1A_P12_LOW_MEMORY_RECOVERY_REPORT_FINAL.md",
    "anchor_evidence_reconciliation_final.json",
    "checkpoint_resume_tests.log",
    "checkpoint_resume_verification.json",
    "checksums.json",
    "checksums_final.json",
    "compute_accounting_final.json",
    "controller_input_validation_note.json",
    "exact_native_memory_parity_final.json",
    "finalization_provenance.json",
    "memory_measurements_final.json",
    "p12_low_memory_recovery_gate_matrix.json",
    "p12_low_memory_recovery_gate_matrix_final.json",
    "post_execution_source_normalization.json",
    "source_provenance.json",
    "trajectory_receipts_final.json",
})


def verify_p12_canonical_audit_linkage(
        audit_dir: Path, complete: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the entire v2 LF-normalized inventory; retain legacy raw mismatches.

    Historical raw pointers remain immutable and may be unresolved. They do not
    override a verified complete canonical-LF v2 inventory and the separately
    checked 20 event/evidence receipts.
    """
    manifest_path = audit_dir / "checksums_final_v2.json"
    manifest = _read_json(manifest_path)
    entries = manifest.get("files", {})
    if (manifest.get("schema") != "stage1a.p12_low_memory_recovery_audit_checksums.v2" or
            manifest.get("file_count") != 17 or
            manifest.get("self_excluded_manifest") != "checksums_final_v2.json" or
            not isinstance(entries, dict) or
            set(entries) != P12_V2_REQUIRED_AUDIT_FILES or
            "replace CRLF and bare CR with LF" not in str(manifest.get("normalization", ""))):
        raise ReadinessError("P1.2 canonical-LF v2 manifest schema/inventory differs")
    verified: dict[str, str] = {}
    for filename in sorted(P12_V2_REQUIRED_AUDIT_FILES):
        row = entries[filename]
        if (not isinstance(row, dict) or
                row.get("hash_mode") != "canonical_lf_text" or
                not isinstance(row.get("size_bytes"), int) or
                isinstance(row.get("size_bytes"), bool) or row["size_bytes"] < 0 or
                not isinstance(row.get("sha256"), str) or
                not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
            raise ReadinessError(f"P1.2 v2 canonical-LF entry is invalid: {filename}")
        path = audit_dir / filename
        if not path.is_file():
            raise ReadinessError(f"P1.2 v2 canonical-LF audit file missing: {filename}")
        try:
            raw = path.read_bytes().decode("utf-8")
        except UnicodeError as exc:
            raise ReadinessError(f"P1.2 v2 audit file is not UTF-8: {filename}") from exc
        canonical = raw.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        actual = sha256_bytes(canonical)
        if len(canonical) != row["size_bytes"] or actual != row["sha256"]:
            raise ReadinessError(f"P1.2 v2 canonical-LF checksum differs: {filename}")
        verified[filename] = actual

    final_name = "STAGE1A_P12_LOW_MEMORY_RECOVERY_REPORT_FINAL.md"
    legacy = _read_json(audit_dir / "checksums_final.json")
    legacy_entries = legacy.get("files", {})
    if not isinstance(legacy_entries, dict):
        raise ReadinessError("P1.2 legacy checksum inventory is malformed")
    final_report_path = audit_dir / final_name
    current_raw_sha = sha256_file(final_report_path)
    legacy_pointers = [
        {"source": "complete.json", "expected_raw_sha256":
         complete.get("final_audit_report_sha256"), "current_raw_sha256": current_raw_sha},
    ]
    for filename in sorted(P12_V2_REQUIRED_AUDIT_FILES):
        legacy_row = legacy_entries.get(filename)
        if legacy_row is None:
            continue
        expected = legacy_row.get("sha256") if isinstance(legacy_row, dict) else legacy_row
        legacy_pointers.append({
            "source": "checksums_final.json:" + filename,
            "expected_raw_sha256": expected,
            "current_raw_sha256": sha256_file(audit_dir / filename),
        })
    unresolved = [item for item in legacy_pointers
                  if item["expected_raw_sha256"] != item["current_raw_sha256"]]
    return {
        "status": "PASS_CANONICAL_LF_V2_COMPLETE",
        "manifest": "checksums_final_v2.json",
        "manifest_sha256": sha256_file(manifest_path),
        "verified_file_count": len(verified),
        "verified_canonical_sha256": verified,
        "final_report_canonical_sha256": verified[final_name],
        "legacy_raw_byte_discrepancies": unresolved,
        "legacy_raw_byte_discrepancies_nonblocking": True,
        "legacy_bytes_recovered": not unresolved,
        "historical_g2_status": "BLOCKED_UNCHANGED",
        "historical_manifests_modified": False,
    }


def _input_inventory(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash frozen input artifacts without loading scores, traces or outcomes."""
    from .stage1a_p11_production import verify_stage05_qats_artifacts

    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root,
                            capture_output=True, text=True, check=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True, check=True).stdout.strip()
    if branch != BRANCH or head != BASE_COMMIT:
        raise ReadinessError(f"Expected clean readiness base {BRANCH}@{BASE_COMMIT}; found {branch}@{head}")

    tapes = verify_stage05_qats_artifacts(STAGE05)
    bindings = [copy.deepcopy(dict(row)) for row in tapes.bindings]
    if ([row["class_identity"] for row in bindings] != list(CLASSES) or
            sum(int(row["query_count"]) for row in bindings) != 459 or
            any(row["seed"] != 37 or row["policy"] != "Q-A-TS-v1" or
                row["official_group_unique"] is not True for row in bindings)):
        raise ReadinessError("Verified Q-A-TS tapes differ from the frozen Stage 1-A focus")
    if tapes.binding_sha256 != TAPE_BINDING_SHA256:
        raise ReadinessError("Q-A-TS aggregate tape binding differs from approved identity")
    tape_by_class = {row["class_identity"]: row for row in bindings}

    p12_complete = _read_json(P12_OUTPUT / "complete.json")
    if (p12_complete.get("decision") != "P12_EXACT_PARITY_AND_EVIDENCE_PASS" or
            p12_complete.get("trajectory_count") != 20 or
            p12_complete.get("event_count") != 76475 or
            p12_complete.get("query_tape_binding_sha256") != TAPE_BINDING_SHA256 or
            p12_complete.get("stage1a_performance_evaluation") is not False or
            p12_complete.get("confirmation_or_mask_access") is not False):
        raise ReadinessError("P1.2 completion record is outside the frozen readiness scope")
    final_report = P12_AUDIT / "STAGE1A_P12_LOW_MEMORY_RECOVERY_REPORT_FINAL.md"
    p12_audit_integrity = verify_p12_canonical_audit_linkage(P12_AUDIT, p12_complete)

    audit_receipts_path = P12_AUDIT / "trajectory_receipts_final.json"
    audit_receipts = _read_json(audit_receipts_path).get("receipts", ())
    audit_receipt_by_pair = {(row.get("category"), row.get("support_seed")): row
                             for row in audit_receipts}
    if len(audit_receipt_by_pair) != 20:
        raise ReadinessError("P1.2 audit receipt summary does not contain 20 unique trajectories")

    stage_root_checks = _read_json(STAGE075A / "checksums.json").get("files", {})
    root_artifacts: dict[str, str] = {}
    for name in ("protocol_manifest.json", "run_manifest.json", "audit_linkage.json"):
        root_artifacts[name] = _assert_hash(STAGE075A / name, stage_root_checks.get(name, ""),
                                            f"Stage 0.75-A/{name}")
    protocol = _read_json(STAGE075A / "protocol_manifest.json")
    run = _read_json(STAGE075A / "run_manifest.json")
    if (run.get("online_trajectory_count") != 20 or
            run.get("stage1_or_stage2_run") is not False or
            protocol.get("support_seeds") != list(SEEDS) or
            protocol.get("query_protocol", {}).get("rate") != 0.03 or
            protocol.get("query_protocol", {}).get("seed") != 37):
        raise ReadinessError("Stage 0.75-A source score/memory run does not match frozen protocol")

    stage05_checks = _read_json(STAGE05 / "checksums.json").get("files", {})
    stage05_score_manifest_path = STAGE05 / "score_tape_manifest.json"
    stage05_score_manifest_sha = _assert_hash(
        stage05_score_manifest_path, stage05_checks.get("score_tape_manifest.json", ""),
        "Stage05/score_tape_manifest.json")
    stage05_score_manifest = _read_json(stage05_score_manifest_path)
    stage05_score_files: dict[str, Any] = {}
    for category in CLASSES:
        category_manifest = stage05_score_manifest.get("classes", {}).get(category, {})
        if category_manifest.get("status") != "PASS":
            raise ReadinessError(f"Stage 0.5 score tape is not complete: {category}")
        seed_rows = category_manifest.get("score_tapes", {})
        for seed in SEEDS:
            entry = seed_rows.get(str(seed), {})
            relative = str(entry.get("file", ""))
            expected = str(entry.get("sha256", ""))
            checksum_expected = stage05_checks.get(relative)
            if not relative.startswith(f"score_tapes/{category}_support") or expected != checksum_expected:
                raise ReadinessError(f"Stage 0.5 static score identity differs: {category}/seed{seed}")
            stage05_score_files[f"{category}/seed{seed}"] = {
                "path": (STAGE05 / relative).relative_to(root).as_posix(),
                "sha256": _assert_hash(STAGE05 / relative, expected,
                                       f"Stage05/{category}/seed{seed}"),
                "fixed_support_sha256": entry.get("fixed_support_sha256"),
            }

    receipts: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    score_sources: dict[str, Any] = {}
    for category in CLASSES:
        folder = STAGE075A / "trajectories" / category
        manifest = _read_json(folder / "checksums.json").get("files", {})
        trajectory_artifacts = {}
        for name in ("trajectory_manifest.json", "scores.npz",
                     "event_trace.jsonl.gz", "c64_transport.json"):
            trajectory_artifacts[name] = _assert_hash(
                folder / name, manifest.get(name, ""), f"Stage075A/{category}/{name}")
        trajectory_meta = _read_json(folder / "trajectory_manifest.json")
        expected_stream_count = int(tape_by_class[category]["n_images"])
        if (trajectory_meta.get("category") != category or
                trajectory_meta.get("smoke_only") is not False or
                trajectory_meta.get("positions_scored") != expected_stream_count or
                trajectory_meta.get("stage1_or_stage2_run") is not False):
            raise ReadinessError(f"Stage 0.75-A trajectory metadata differs: {category}")
        score_sources[category] = trajectory_artifacts
        for seed in SEEDS:
            receipt_path = P12_OUTPUT / "shards" / category / f"seed{seed}.receipt.json"
            receipt = _read_json(receipt_path)
            evidence_path = root / str(receipt.get("evidence_file", ""))
            evidence_sha = _assert_hash(evidence_path,
                                        str(receipt.get("evidence_file_sha256", "")),
                                        f"P1.2 evidence {category}/seed{seed}")
            tape_identity = tape_by_class[category]
            if (receipt.get("category") != category or receipt.get("support_seed") != seed or
                    receipt.get("status") != "PASS_EXACT_PARITY" or
                    receipt.get("complete") is not True or
                    receipt.get("query_tape_binding_sha256") != TAPE_BINDING_SHA256 or
                    receipt.get("query_count") != tape_identity["query_count"] or
                    receipt.get("query_positions_sha256") != tape_identity["positions_sha256"] or
                    receipt.get("expected_event_count") != expected_stream_count or
                    receipt.get("event_count") != expected_stream_count or
                    receipt.get("scorer_config_sha256") != protocol.get("scorer_config_sha256") or
                    receipt.get("evidence_revision_count") != receipt.get("query_count", -1) + 1 or
                    receipt.get("evidence_file_sha256") != evidence_sha):
                raise ReadinessError(f"P1.2 trajectory receipt failed: {category}/seed{seed}")
            linked_receipt = audit_receipt_by_pair.get((category, seed))
            if (not isinstance(linked_receipt, Mapping) or
                    any(receipt.get(key) != value
                        for key, value in linked_receipt.items())):
                raise ReadinessError(f"P1.2 final receipt summary differs: {category}/seed{seed}")
            receipt_hash = sha256_file(receipt_path)
            row = {
                "trajectory_id": receipt.get("trajectory_id"),
                "category": category,
                "support_seed": seed,
                "query_tape_identity": tape_identity,
                "receipt_path": receipt_path.relative_to(root).as_posix(),
                "receipt_sha256": receipt_hash,
                "event_count": receipt.get("event_count"),
                "query_count": receipt.get("query_count"),
                "query_positions_sha256": receipt.get("query_positions_sha256"),
                "query_response_chain_sha256": receipt.get("query_response_chain_sha256"),
                "event_chain_sha256": receipt.get("event_chain_sha256"),
                "source_fingerprint_sha256": receipt.get("source_fingerprint_sha256"),
                "scorer_config_sha256": receipt.get("scorer_config_sha256"),
                "scorer_source_sha256": receipt.get("scorer_source_sha256"),
                "evidence_source": receipt.get("evidence_source"),
                "evidence_path": evidence_path.relative_to(root).as_posix(),
                "evidence_sha256": evidence_sha,
                "evidence_revision_count": receipt.get("evidence_revision_count"),
                "initial_memory": receipt.get("initial_memory"),
                "final_memory": receipt.get("final_memory"),
                "stage075a_artifacts": trajectory_artifacts,
                "controller_score_semantics": "frozen pre-event score from recorded memory version",
            }
            receipts.append(row)
            trajectory_rows.append(row)
    if len(trajectory_rows) != 20:
        raise ReadinessError("Readiness input inventory does not contain exactly 20 trajectories")

    source_paths = (
        "experiments/calibration/stage1a_final_readiness.py",
        "experiments/calibration/controllers.py",
        "experiments/calibration/spot_dspot.py",
        "experiments/calibration/scdac_g.py",
        "experiments/calibration/stage075b_controllers.py",
        "experiments/calibration/metrics.py",
        "experiments/calibration/protocol.py",
        "experiments/calibration/profiles.py",
        "experiments/calibration/method_registry.py",
        "experiments/calibration/stage1a_p11_production.py",
        "experiments/calibration/stage1a_p12_sharded_recovery.py",
        "experiments/calibration/stage075a_runner.py",
        "experiments/calibration/stage075a.py",
        "experiments/calibration/stage05.py",
        "experiments/calibration/sun_cs.py",
        "tests/calibration/test_stage1a_final_readiness.py",
        "test_phase3_AIF_simplified_v22.py",
        "docs/calibration/STAGE1_EVALUATION_PROTOCOL_APPROVED_DESIGN_v3.md",
        "docs/calibration/STAGE1A_FOCUSED_PROTOCOL_ADDENDUM_CANDIDATE_v1_2026-10-30.md",
        "docs/calibration/STAGE1A_P0_READINESS_REPORT_v1_2026-10-30.md",
        "docs/calibration/P12_SCOPED_SOURCE_RECONSTRUCTION_DECISION_2026-10-30.md",
        "docs/calibration/audit/stage1a_p12_low_memory_recovery_v3_20261030/STAGE1A_P12_LOW_MEMORY_RECOVERY_REPORT_FINAL.md",
    )
    current_sources = {name: sha256_file(root / name) for name in source_paths}
    inputs = {
        "schema": "stage1a.readiness.execution_manifest.v1",
        "repository": "jiangdizhao/Memory-Governed-Online-Few-Shot-Anomaly-Detection",
        "branch": branch,
        "base_commit": head,
        "focused_protocol": {
            "decision_id": "FOCUSED_STAGE1A_READINESS_SPEC_v1_2026-10-30",
            "classes": list(CLASSES), "support_seeds": list(SEEDS),
            "query_protocol": "Q-A-TS-v1", "query_rate": 0.03,
            "query_seed": 37, "query_tape_binding_sha256": TAPE_BINDING_SHA256,
            "new_qb_or_rate_variants": False,
            "approved_design_v3_modified": False,
            "focused_addendum_candidate_modified": False,
        },
        "verified_query_tape_bindings": bindings,
        "query_tape_verifier_sha256": tapes.verification_sha256,
        "stage05_source_artifacts": {
            name: sha256_file(STAGE05 / name)
            for name in ("checksums.json", "frozen_inputs.json", "query_tapes.json.gz",
                         "run_manifest.json")
            if (STAGE05 / name).is_file()},
        "stage05_score_tape_manifest_sha256": stage05_score_manifest_sha,
        "stage05_static_score_tape_artifacts": stage05_score_files,
        "stage075a_root_artifacts": root_artifacts,
        "stage075a_protocol_sha256": sha256_file(STAGE075A / "protocol_manifest.json"),
        "stage075a_trajectory_artifacts_by_class": score_sources,
        "p12_completion": {
            "complete_json_sha256": sha256_file(P12_OUTPUT / "complete.json"),
            "final_report_sha256": sha256_file(final_report),
            "audit_receipts_sha256": sha256_file(audit_receipts_path),
            "audit_integrity": p12_audit_integrity,
            "trajectory_count": 20, "event_count": 76475,
            "performance_evaluation_executed": False,
        },
        "trajectories": trajectory_rows,
        "historical_g2_status": "BLOCKED_UNCHANGED",
        "p12_source_provenance_caveat": {
            "scoped_current_source_reconstruction_used": True,
            "exact_historical_executed_source_bytes_recovered": False,
            "query_response_digests_recovered_from_historical_g2": False,
            "historical_g2_closed": False,
        },
    }
    provenance = {
        "schema": "stage1a.readiness.source_provenance.v1",
        "current_readiness_source_sha256": current_sources,
        "stage075a_historical_source_sha256": protocol.get("source_sha256", {}),
        "stage075a_native_source_sha256": protocol.get("native_source_sha256", {}),
        "p12_scoped_source_provenance_artifact_sha256": sha256_file(
            P12_AUDIT / "source_provenance.json"),
        "source_provenance_caveats": inputs["p12_source_provenance_caveat"],
        "offline_evaluation_metrics_opened": False,
        "full_stream_labels_or_masks_opened": False,
        "gpu_or_native_inference_executed": False,
    }
    return inputs, provenance


def _controller_input_matrix(registry: Mapping[str, Any]) -> dict[str, Any]:
    rows = {
        "B0": {"score": True, "queried_label": False, "evidence": "none"},
        "B1": {"score": True, "queried_label": "queried verified labels; only normal labels update W=256 buffer", "evidence": "none"},
        "B2": {"score": True, "queried_label": "queried verified labels", "evidence": "additional C64 scalar initialization only"},
        "B3": {"score": True, "queried_label": False, "evidence": "protected current-memory C64 only"},
        "B4": {"score": True, "queried_label": "queried verified labels", "evidence": "causal current-memory queried anchors"},
        "B5a": {"score": True, "queried_label": False, "evidence": "none"},
        "B5a_FROZEN_FALLBACK": {"score": True, "queried_label": False, "evidence": "none; distinct theta0 fallback wrapper"},
        "B5b": {"score": True, "queried_label": False, "evidence": "none"},
        "B5b_FROZEN_FALLBACK": {"score": True, "queried_label": False, "evidence": "none; distinct theta0 fallback wrapper"},
        "B7": {"score": True, "queried_label": False, "evidence": "protected C64 plus eligible queried normal LSO and queried defect anchors"},
        "Stage075BOffset": {"score": True, "queried_label": False, "evidence": "causal queried anchor evidence; holds own threshold if either class side is unavailable"},
        "B6": {"score": "applicability reference only", "queried_label": False, "evidence": "none; not run in readiness"},
        "B8": {"score": "historical negative control only", "queried_label": False, "evidence": "historical artifacts only"},
    }
    if set(rows) != set(registry["methods"]):
        raise ReadinessError("Controller information matrix and registry do not match")
    return {"schema": "stage1a.controller_input_matrix.v1", "methods": rows,
            "shared_score_memory_query_writer_path": "controller-independent frozen input path"}


def _metric_contract() -> dict[str, Any]:
    return {
        "schema": "stage1a.metric_contract.v1",
        "implementation": "experiments/calibration/metrics.py::PrivateOutcomeLedger plus readiness adapter",
        "evaluation_target": "physical defect; labels remain in post-prediction evaluator",
        "full_stream": ["TP", "TN", "FP", "FN", "balanced_accuracy", "FPR", "FNR", "defect_recall"],
        "undefined_denominators": "JSON null; never coerce to zero",
        "worst_window": {"size": 256, "stride": 256,
                         "report_only_full_nonoverlap_windows": True,
                         "undefined_windows_excluded_from_worst_defined": True},
        "common_horizon": {"start_zero_based": 1050, "stop_exclusive": "stream end"},
        "coverage": ["overall", "normal_class_conditional", "defect_class_conditional", "selective_balanced_accuracy"],
        "missingness": {"source_startup_unavailable": "separate from available abstention",
                        "fallback_variants": "separate rows with original/fallback decision fractions"},
        "ranking": "shared-score AUROC/AUPRC computed once offline and invariant to threshold-only controllers",
        "unsupervised_applicability": {
            "methods": ["B5a", "B5b", "B6"],
            "report_actual_and_class_conditional_coverage": True,
            "post_outcome_numeric_coverage_cutoff": None,
            "inappropriate_metrics": "report undefined rather than suppressing the comparator",
            "fallback_variants": "report fraction of decisions made by source-native method and by fallback separately",
            "fallback_is_deployment_hybrid": True,
        },
        "source_native_spot_dspot_full_stream_ba": "null when startup-unavailable predictions occur; never ordinary full-stream BA",
        "real_stream_metrics_created": False,
    }


def _run_unit_suite(root: Path = ROOT) -> dict[str, Any]:
    modules = (
        "tests.calibration.test_stage1a_final_readiness",
        "tests.calibration.test_engine_and_ledger",
        "tests.calibration.test_controllers",
        "tests.calibration.test_spot_dspot",
        "tests.calibration.test_scdac_g",
        "tests.calibration.test_stage075b",
    )
    command = [sys.executable, "-m", "unittest", *modules, "-v"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    combined = result.stdout + result.stderr
    ran = re.search(r"Ran (\d+) tests?", combined)
    test_names: list[str] = []
    import ast
    for module in modules:
        path = root / (module.replace(".", "/") + ".py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [node.name for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                 node.name.startswith("test_")]
        test_names.extend(f"{module}.{name}" for name in sorted(names))
    count = int(ran.group(1)) if ran else None
    return {
        "schema": "stage1a.readiness.test_results.v1",
        "command": " ".join(command),
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "exit_code": result.returncode,
        "test_count": count,
        "tests": [{"id": name, "status": "PASS" if result.returncode == 0 else "UNVERIFIED"}
                  for name in test_names],
        "test_inventory_matches_unittest_count": count == len(test_names),
        "stdout_stderr_sha256": sha256_bytes(combined.encode("utf-8")),
        "summary": combined[-8000:],
        "scope": "synthetic CPU controller/metric fixtures plus existing causal/controller unit suites; bounded first-query frozen-input smoke is recorded separately; no real performance evaluation",
    }


def _run_bounded_real_input_smoke() -> dict[str, Any]:
    """Exercise the production adapter only through audiojack/seed0's first query."""
    from .controllers import theta0_from_c64

    category, seed = "audiojack", 0
    receipt_path = P12_OUTPUT / "shards" / category / f"seed{seed}.receipt.json"
    receipt = _read_json(receipt_path)
    trajectory = load_frozen_trajectory_input(
        category=category, support_seed=seed, receipt=receipt)
    c64 = [anchor.score for anchor in trajectory.initial_evidence.normal_anchors
           if anchor.protected]
    suite = build_controller_suite(theta0_from_c64(c64), c64)
    input_binding = sha256_json({
        "category": category, "support_seed": seed,
        "receipt_sha256": sha256_file(receipt_path),
        "query_tape_binding_sha256": TAPE_BINDING_SHA256,
    })
    replay = ControllerOnlyReplay(
        controllers=suite, query_positions=trajectory.query_positions,
        input_binding_sha256=input_binding,
        initial_memory=trajectory.initial_memory,
        initial_evidence=trajectory.initial_evidence)
    first_query = trajectory.query_positions[0]
    events = trajectory.events()
    event_count = query_count = 0
    try:
        for item in events:
            replay.step(item)
            event_count += 1
            query_count += int(item.queried)
            if item.t == first_query:
                break
    finally:
        events.close()
    if (event_count != first_query + 1 or query_count != 1 or
            replay.cursor != event_count):
        raise ReadinessError("Bounded real-input smoke did not stop at the first taped query")
    return {
        "status": "PASS",
        "trajectory": f"{category}/seed{seed}",
        "receipt_sha256": sha256_file(receipt_path),
        "evidence_sha256": receipt["evidence_file_sha256"],
        "query_tape_binding_sha256": TAPE_BINDING_SHA256,
        "events_through_first_query": event_count,
        "precommitted_query_events_consumed": query_count,
        "first_query_position": first_query,
        "registered_controllers_instantiated": len(suite),
        "full_stream_replayed": False,
        "performance_metrics_generated": False,
        "native_or_gpu_execution": False,
        "unqueried_labels_accessed": False,
        "confirmation_classes_or_masks_accessed": False,
    }


def build_audit_package(root: Path = ROOT, target: Path = AUDIT_DIR) -> Path:
    """Create a new immutable, compact readiness package; never overwrite."""
    if target.exists():
        raise ReadinessError(f"Immutable audit namespace already exists: {target}")
    inputs, provenance = _input_inventory(root)
    registry = _method_registry()
    method_source_paths = {
        "B0": ["experiments/calibration/controllers.py"],
        "B1": ["experiments/calibration/controllers.py"],
        "B2": ["experiments/calibration/controllers.py", "test_phase3_AIF_simplified_v22.py"],
        "B3": ["experiments/calibration/controllers.py"],
        "B4": ["experiments/calibration/controllers.py"],
        "B5a": ["experiments/calibration/spot_dspot.py"],
        "B5a_FROZEN_FALLBACK": ["experiments/calibration/spot_dspot.py", "experiments/calibration/controllers.py"],
        "B5b": ["experiments/calibration/spot_dspot.py"],
        "B5b_FROZEN_FALLBACK": ["experiments/calibration/spot_dspot.py", "experiments/calibration/controllers.py"],
        "B7": ["experiments/calibration/scdac_g.py"],
        "Stage075BOffset": ["experiments/calibration/stage075b_controllers.py"],
        "B6": ["experiments/calibration/sun_cs.py"],
        "B8": ["experiments/calibration/stage075b_controllers.py"],
    }
    all_current_source_hashes = provenance["current_readiness_source_sha256"]
    for method_id, method in registry["methods"].items():
        source_files = method_source_paths[method_id]
        source_identity = {path: (all_current_source_hashes.get(path) or
                                  sha256_file(root / path)) for path in source_files}
        config = {key: method.get(key) for key in (
            "version", "status", "equation", "equations", "defaults",
            "configuration", "initialization", "adaptations", "missing_data_behavior")
                  if key in method}
        method["source_files"] = source_files
        method["source_sha256"] = source_identity
        method["configuration_sha256"] = sha256_json(config)
    inputs["controller_config_identities"] = {
        method_id: {
            "configuration_sha256": method["configuration_sha256"],
            "source_files": method["source_files"],
            "source_sha256": method["source_sha256"],
        } for method_id, method in registry["methods"].items()}
    tests = _run_unit_suite(root)
    if (tests["status"] != "PASS" or
            tests["test_inventory_matches_unittest_count"] is not True):
        raise ReadinessError("Readiness synthetic unit suite failed; audit package not emitted")
    tests["bounded_real_input_smoke"] = _run_bounded_real_input_smoke()
    input_matrix = _controller_input_matrix(registry)
    metric_contract = _metric_contract()
    # The complete P1.2 v2 canonical-LF manifest is verified (17/17) by
    # _input_inventory. Legacy raw-byte pointers are archival caveats, not an
    # additional gate on the already-verified 20 trajectories and evidence.
    p12_integrity = inputs["p12_completion"]["audit_integrity"]
    if p12_integrity["status"] != "PASS_CANONICAL_LF_V2_COMPLETE":
        raise ReadinessError("P1.2 canonical-LF v2 audit inventory did not pass")
    decision = "STAGE1A_READY_FOR_EXECUTION"
    gates = {
        "schema": "stage1a.readiness.gate_matrix.v1",
        "decision": decision,
        "formal_evaluation_executed": False,
        "gates": [
            {"id": "R-INPUTS", "status": "PASS", "detail": "20/20 receipts and their evidence SHA-256 identities verified; four frozen score/writer/memory artifact sets linked."},
            {"id": "R-TAPE", "status": "PASS", "detail": "Q-A-TS-v1 tapes verified metadata-only against four class identities and frozen aggregate binding."},
            {"id": "R-REGISTRY", "status": "PASS", "detail": "All executable methods registered; B6 applicability-only and B8 historical-only are explicitly excluded from execution."},
            {"id": "R-CAUSAL", "status": "PASS", "detail": "Synthetic causal tests pass; bounded audiojack/seed0 real-input adapter smoke consumes 36 events through only the first taped query, with no metrics."},
            {"id": "R-METRICS", "status": "PASS", "detail": "Synthetic fixtures verify the frozen metric, null-denominator, horizon, coverage, window, startup, fallback and ranking contracts."},
            {"id": "R-P12-AUDIT-LINKAGE", "status": "PASS_WITH_LEGACY_RAW_BYTE_CAVEAT" if p12_integrity["legacy_raw_byte_discrepancies"] else "PASS", "detail": "All 17 P1.2 v2 canonical-LF audit files match exactly; 20 receipt/evidence pairs independently match. Old raw-byte pointers remain archived as nonblocking discrepancies."},
            {"id": "R-SOURCE-CAVEATS", "status": "PASS_WITH_DECLARED_CAVEAT", "detail": "Historical G2 remains BLOCKED; exact executed P1.2 source bytes were not recovered."},
            {"id": "R-STAGE1-EVALUATION", "status": "NOT_RUN", "detail": "No formal Stage 1-A comparison or Real-IAD performance metrics were generated; only the 36-event first-query readiness smoke ran."},
        ],
        "historical_g2_status": "BLOCKED_UNCHANGED",
        "researcher_review_required_for_acceptance": True,
    }
    report = """# Stage 1-A Final Readiness Report\n\n"""
    report += f"**Decision: `{decision}`**\n\n"
    report += ("This package freezes a focused, controller-only Stage 1-A execution specification. ")
    report += ("It does not execute the algorithm comparison, report Real-IAD outcomes, or evaluate performance.\n\n")
    report += "## Frozen scope\n\n"
    report += "- Classes: audiojack, regulator, switch, rolled_strip_base.\n"
    report += "- Support seeds: 0–4.\n"
    report += "- Query tape: Q-A-TS-v1, 3%, seed 37; binding `" + TAPE_BINDING_SHA256 + "`.\n"
    report += "- No Q-B, 5%, or 7% runs. Protocol v3 and the focused addendum candidate are unchanged; this is `FOCUSED_STAGE1A_READINESS_SPEC_v1_2026-10-30`.\n\n"
    report += "## Readiness evidence\n\n"
    report += "The adapter consumes precomputed scores and recorded memory transitions. It calls each prediction before queried feedback, supplies feedback only to registered consumers, applies the recorded writer/memory transition, delivers the corresponding evidence, then updates controller state for the next event. It never invokes Native Scorer, DINOv2, or Greedy writer. Deterministic synthetic CPU tests passed. A separate bounded production-input smoke processed audiojack/seed0 through its first taped query at t=35 (36 events, one query), produced no metrics, and stopped; it did not replay the stream.\n\n"
    report += "The execution manifest links all 20 P1.2 receipts and evidence hashes, and Stage 0.75-A score/event/C64 provenance by class. The query tape verifier reads only frozen metadata and the label-blind tape archive. Offline performance outputs, full-stream labels, masks, and confirmation outcomes were not opened.\n\n"
    report += "## Method registry and metrics\n\n"
    report += "Executable methods: B0–B5b, B7, and Stage075BOffset; B5a/B5b source-native methods and their Frozen-Theta hybrids remain separate. B6 is retained as applicability-only and B8 as an existing historical negative control. Stage075BOffset preserves its own raw-median-offset behavior and holds its own threshold if either class-side evidence is unavailable; it is not an additive B3 offset.\n\n"
    report += "Metrics are specified for a later private post-prediction evaluator. Undefined denominators remain null; source-native SPOT/dSPOT startup-unavailable predictions do not receive ordinary full-stream BA; common horizon begins at zero-based t=1050; worst-window reporting uses defined full nonoverlapping 256-image windows. B5a/B5b/B6 applicability reporting retains actual and class-conditional coverage, makes no post-outcome cutoff, and leaves inappropriate metrics undefined; fallback decisions are separately attributed.\n\n"
    report += "## Source caveats and limits\n\n"
    report += "The historical Stage 0.75-A G2 gate remains BLOCKED. P1.2 uses scoped current-source reconstruction; exact historical executed source bytes and G2 query-response digests were not recovered. This package preserves that caveat and does not close G2. The readiness result is an engineering readiness decision only, not a scientific validation or risk guarantee.\n\n"
    report += "## P1.2 audit integrity resolution\n\n"
    report += ("The complete 17-file P1.2 v2 canonical-LF audit inventory is verified, "
               "including the final report, receipts, parity/evidence and compute records. "
               "Earlier raw-byte report pointers remain unresolved and are retained as "
               "historical provenance caveats, not treated as a numerical or causal failure. "
               "No historical file was rewritten. See the append-only v7 integrity closure.\n\n")
    report += "## Gate decision\n\n"
    report += "**`STAGE1A_READY_FOR_EXECUTION`** — all readiness gates pass with the declared source and legacy raw-byte caveats. The next task may conduct the frozen Stage 1-A comparison; no evaluation was started by readiness.\n"

    files: dict[str, bytes] = {
        "STAGE1A_FINAL_READINESS_REPORT.md": report.encode("utf-8"),
        "stage1a_frozen_method_registry.json": (json.dumps(registry, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(),
        "stage1a_execution_manifest.json": (json.dumps(inputs, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(),
        "stage1a_controller_input_matrix.json": (json.dumps(input_matrix, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(),
        "stage1a_metric_contract.json": (json.dumps(metric_contract, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(),
        "stage1a_readiness_gate_matrix.json": (json.dumps(gates, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(),
        "test_results.json": (json.dumps(tests, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(),
        "source_provenance.json": (json.dumps(provenance, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(),
    }
    target.mkdir(parents=True, exist_ok=False)
    for name, payload in files.items():
        (target / name).write_bytes(payload)
    checksums = {name: sha256_file(target / name) for name in sorted(files)}
    (target / "checksums.json").write_bytes(
        (json.dumps({"schema": "stage1a.readiness.checksums.v1", "files": checksums},
                    sort_keys=True, indent=2) + "\n").encode("utf-8"))
    return target


def verify_audit_package(target: Path = AUDIT_DIR) -> dict[str, Any]:
    checks = _read_json(target / "checksums.json").get("files", {})
    expected_names = {
        "STAGE1A_FINAL_READINESS_REPORT.md", "stage1a_frozen_method_registry.json",
        "stage1a_execution_manifest.json", "stage1a_controller_input_matrix.json",
        "stage1a_metric_contract.json", "stage1a_readiness_gate_matrix.json",
        "test_results.json", "source_provenance.json",
    }
    if set(checks) != expected_names:
        raise ReadinessError("Readiness checksum inventory is incomplete or contains extras")
    for name, expected in checks.items():
        _assert_hash(target / name, expected, name)
    manifest = _read_json(target / "stage1a_execution_manifest.json")
    if len(manifest.get("trajectories", ())) != 20:
        raise ReadinessError("Readiness package lost one or more trajectories")
    test_results = _read_json(target / "test_results.json")
    smoke = test_results.get("bounded_real_input_smoke", {})
    gate = _read_json(target / "stage1a_readiness_gate_matrix.json")
    if (test_results.get("status") != "PASS" or
            test_results.get("test_inventory_matches_unittest_count") is not True or
            smoke.get("status") != "PASS" or
            smoke.get("trajectory") != "audiojack/seed0" or
            smoke.get("events_through_first_query") != 36 or
            smoke.get("precommitted_query_events_consumed") != 1 or
            smoke.get("full_stream_replayed") is not False or
            smoke.get("performance_metrics_generated") is not False or
            smoke.get("unqueried_labels_accessed") is not False or
            gate.get("decision") not in {"STAGE1A_READY_FOR_EXECUTION",
                                         "STAGE1A_READINESS_PARTIAL",
                                         "STAGE1A_READINESS_BLOCKED"}):
        raise ReadinessError("Readiness test status or final decision is malformed")
    return {"status": "PASS", "checksummed_files": len(checks),
            "trajectory_identities": len(manifest["trajectories"]),
            "checksums_manifest_sha256": sha256_file(target / "checksums.json")}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-audit", action="store_true")
    parser.add_argument("--verify-audit", action="store_true")
    args = parser.parse_args()
    if args.build_audit:
        print(build_audit_package())
    elif args.verify_audit:
        print(json.dumps(verify_audit_package(), sort_keys=True))
    else:
        parser.error("Choose --build-audit or --verify-audit")
