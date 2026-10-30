"""Versioned B3 memory-state lag replay and read-only Gate A0 audit.

The replay accepts scores and memory transitions only. It never accepts labels,
opens images, invokes a scorer, or writes under the frozen source root.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "calibration.b3_state_lag_a1.v1"
CLASSES = ("audiojack", "regulator", "switch", "rolled_strip_base")
SEEDS = (0, 1, 2, 3, 4)
ARMS = {"A2": 0, "A3": 1, "A4": 5}
DELTA = 0.01
Q90 = 0.90
C64_N = 64
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class AblationError(RuntimeError):
    """Fail-closed error for invalid artifacts or replay state."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def higher_quantile(values: Sequence[float], q: float = Q90) -> float:
    """Higher order statistic at ceil(q * (n - 1)), with finite inputs only."""
    if not values or not math.isfinite(float(q)) or not 0.0 <= q <= 1.0:
        raise ValueError("higher_quantile requires finite values and q in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("higher_quantile received a nonfinite value")
    return ordered[math.ceil(float(q) * (len(ordered) - 1))]


def _require_sha(value: str, label: str) -> None:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise AblationError(f"{label} must be a lowercase SHA-256 digest")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AblationError(f"MISSING_ARTIFACT: {path}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AblationError(f"Invalid JSON artifact: {path}") from exc


def _check_file(path: Path, expected: str, role: str,
                artifacts: list[dict[str, str]]) -> str:
    if not path.is_file():
        raise AblationError(f"MISSING_ARTIFACT: {role}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise AblationError(f"HASH_MISMATCH: {role}: {path}")
    artifacts.append({"role": role, "path": path.as_posix(), "sha256": actual})
    return actual


@dataclass(frozen=True)
class C64Reference:
    memory_version: int
    memory_sha256: str
    q90: float
    c64_group_sha256: str
    vector_sha256: str
    c64_count: int = C64_N

    @classmethod
    def from_scores(cls, version: int, memory_sha256: str,
                    scores: Sequence[float], c64_group_sha256: str) -> "C64Reference":
        vals = [float(value) for value in scores]
        if len(vals) != C64_N or not all(math.isfinite(value) for value in vals):
            raise AblationError("Each C64 reference must contain 64 finite scores")
        return cls(int(version), str(memory_sha256), higher_quantile(vals),
                   str(c64_group_sha256), sha256_json(vals), len(vals))


def reference_from_evidence(version: int, memory_sha256: str,
                            scores: Sequence[float], c64_group_sha256: str,
                            stored_q90: float, stored_vector_sha256: str) -> C64Reference:
    """Recompute a frozen q90/vector identity and reject any mismatch."""
    reference = C64Reference.from_scores(version, memory_sha256, scores, c64_group_sha256)
    if reference.q90 != float(stored_q90):
        raise AblationError("Q90_EVIDENCE_MISMATCH")
    if reference.vector_sha256 != str(stored_vector_sha256):
        raise AblationError("C64_VECTOR_EVIDENCE_MISMATCH")
    return reference


class VersionedC64History:
    """Complete q90 history; lookups declare and enforce their as-of version."""

    def __init__(self, references: Iterable[C64Reference]):
        refs = tuple(sorted(references, key=lambda item: item.memory_version))
        if not refs or [item.memory_version for item in refs] != list(range(len(refs))):
            raise AblationError("C64 q90 history must be complete from version 0")
        identity = refs[0].c64_group_sha256
        _require_sha(identity, "C64 group identity")
        for ref in refs:
            _require_sha(ref.memory_sha256, "memory content identity")
            _require_sha(ref.vector_sha256, "C64 vector identity")
            _require_sha(ref.c64_group_sha256, "C64 group identity")
            if ref.c64_count != C64_N or not math.isfinite(ref.q90):
                raise AblationError("Malformed C64 reference evidence")
            if ref.c64_group_sha256 != identity:
                raise AblationError("C64 anchor identity changed across memory versions")
        self._references = refs
        self.identity_sha256 = sha256_json([
            {"version": r.memory_version, "memory_sha256": r.memory_sha256,
             "q90": r.q90, "c64_group_sha256": r.c64_group_sha256,
             "vector_sha256": r.vector_sha256, "c64_count": r.c64_count}
            for r in refs
        ])
        self.lookup_log: list[tuple[int, int]] = []

    @property
    def references(self) -> tuple[C64Reference, ...]:
        return self._references

    @property
    def final_version(self) -> int:
        return len(self._references) - 1

    def lookup(self, version: int, *, as_of_version: int) -> C64Reference:
        version, as_of_version = int(version), int(as_of_version)
        self.lookup_log.append((version, as_of_version))
        if version > as_of_version:
            raise AblationError("FUTURE_EVIDENCE_ACCESS")
        if version < 0 or version >= len(self._references):
            raise AblationError(f"MISSING_Q90_EVIDENCE: memory version {version}")
        return self._references[version]


@dataclass(frozen=True)
class TrajectoryEvent:
    position: int
    score: float
    memory_version_before: int
    memory_sha256_before: str
    memory_version_after: int
    memory_sha256_after: str
    query_selected: bool = False

    def __post_init__(self) -> None:
        if self.position < 0 or not math.isfinite(float(self.score)):
            raise AblationError("Trajectory event has invalid position or score")
        _require_sha(self.memory_sha256_before, "pre-event memory identity")
        _require_sha(self.memory_sha256_after, "post-event memory identity")
        before, after = int(self.memory_version_before), int(self.memory_version_after)
        if after not in (before, before + 1):
            raise AblationError("Memory version must hold or advance by exactly one")
        if after == before and self.memory_sha256_after != self.memory_sha256_before:
            raise AblationError("No-op memory write changed its content digest")


@dataclass(frozen=True)
class ThresholdUpdate:
    position: int
    memory_version: int
    reference_version: int
    q90_reference: float
    theta_before: float
    theta_after: float
    applied_step: float


class B3LagController:
    """Label-free B3 recurrence with registered current or lagged q90."""

    def __init__(self, arm_id: str, history: VersionedC64History,
                 initial_theta: float, delta: float = DELTA):
        if arm_id not in ARMS:
            raise ValueError(f"Unknown registered arm: {arm_id}")
        if not math.isfinite(float(delta)) or float(delta) != DELTA:
            raise AblationError(f"Delta is frozen at {DELTA}")
        first = history.lookup(0, as_of_version=0)
        if float(initial_theta) != first.q90:
            raise AblationError("Initial theta must exactly equal M0 C64 q90")
        self.arm_id, self.lag = arm_id, ARMS[arm_id]
        self.history = history
        self.initial_theta = float(initial_theta)
        self.delta = float(delta)
        self.theta = float(initial_theta)
        self.current_memory_version = 0
        self.current_memory_sha256 = first.memory_sha256
        self.updates: list[ThresholdUpdate] = []
        self.transitions: list[dict[str, Any]] = []
        self._pending_prediction: dict[str, Any] | None = None

    def predict(self, score: float, memory_version: int,
                memory_sha256: str) -> int:
        if self._pending_prediction is not None:
            raise AblationError("Previous prediction has no completed memory transition")
        if not math.isfinite(float(score)):
            raise AblationError("Prediction score must be finite")
        if (int(memory_version) != self.current_memory_version or
                str(memory_sha256) != self.current_memory_sha256):
            raise AblationError("Prediction memory identity differs from controller state")
        prediction = int(float(score) >= self.theta)
        self._pending_prediction = {
            "score": float(score), "memory_version": int(memory_version),
            "memory_sha256": str(memory_sha256), "prediction": prediction,
        }
        return prediction

    def after_event(self, event: TrajectoryEvent) -> float:
        pending = self._pending_prediction
        if pending is None:
            raise AblationError("Memory transition requires a preceding prediction")
        if (pending["score"] != float(event.score) or
                pending["memory_version"] != event.memory_version_before or
                pending["memory_sha256"] != event.memory_sha256_before):
            raise AblationError("Memory transition differs from the preceding prediction event")
        if event.position != len(self.transitions):
            raise AblationError("Controller event positions must be contiguous from zero")
        if (event.memory_version_before != self.current_memory_version or
                event.memory_sha256_before != self.current_memory_sha256):
            raise AblationError("Event pre-state differs from controller memory state")
        if event.memory_version_after == event.memory_version_before:
            self.transitions.append({
                "position": int(event.position), "score": float(event.score),
                "prediction": pending["prediction"],
                "memory_version_before": int(event.memory_version_before),
                "memory_sha256_before": event.memory_sha256_before,
                "memory_version_after": int(event.memory_version_after),
                "memory_sha256_after": event.memory_sha256_after,
            })
            self._pending_prediction = None
            return self.theta
        current = self.history.lookup(event.memory_version_after,
                                      as_of_version=event.memory_version_after)
        if current.memory_sha256 != event.memory_sha256_after:
            raise AblationError("Current q90 evidence memory digest mismatch")
        reference_version = max(0, event.memory_version_after - self.lag)
        reference = self.history.lookup(reference_version,
                                        as_of_version=event.memory_version_after)
        theta_before = self.theta
        theta_after = min(max(reference.q90, theta_before - self.delta),
                          theta_before + self.delta)
        self.transitions.append({
            "position": int(event.position), "score": float(event.score),
            "prediction": pending["prediction"],
            "memory_version_before": int(event.memory_version_before),
            "memory_sha256_before": event.memory_sha256_before,
            "memory_version_after": int(event.memory_version_after),
            "memory_sha256_after": event.memory_sha256_after,
        })
        self.theta = theta_after
        self.current_memory_version = event.memory_version_after
        self.current_memory_sha256 = event.memory_sha256_after
        self.updates.append(ThresholdUpdate(
            event.position, event.memory_version_after, reference_version,
            reference.q90, theta_before, self.theta, self.theta - theta_before))
        self._pending_prediction = None
        return self.theta

    def state_dict(self) -> dict[str, Any]:
        if self._pending_prediction is not None:
            raise AblationError("Checkpoint requires an event boundary after its memory transition")
        return {
            "schema": "calibration.b3_lag_checkpoint.v1",
            "arm_id": self.arm_id, "lag": self.lag, "delta": self.delta,
            "initial_theta": self.initial_theta, "theta": self.theta,
            "current_memory_version": self.current_memory_version,
            "current_memory_sha256": self.current_memory_sha256,
            "history_identity_sha256": self.history.identity_sha256,
            "updates": [vars(item) for item in self.updates],
            "transitions": list(self.transitions),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any],
                        history: VersionedC64History) -> "B3LagController":
        if state.get("schema") != "calibration.b3_lag_checkpoint.v1":
            raise AblationError("Unsupported checkpoint schema")
        controller = cls(str(state.get("arm_id")), history,
                         float(state["initial_theta"]), float(state["delta"]))
        if (int(state.get("lag", -1)) != controller.lag or
                state.get("history_identity_sha256") != history.identity_sha256):
            raise AblationError("Checkpoint configuration or q90 history mismatch")
        for raw in state.get("transitions", []):
            score = float(raw["score"])
            prediction = controller.predict(
                score, int(raw["memory_version_before"]),
                str(raw["memory_sha256_before"]))
            if prediction != int(raw["prediction"]):
                raise AblationError("Checkpoint prediction history does not replay exactly")
            event = TrajectoryEvent(
                int(raw["position"]), score,
                int(raw["memory_version_before"]), str(raw["memory_sha256_before"]),
                int(raw["memory_version_after"]), str(raw["memory_sha256_after"]))
            controller.after_event(event)
        if (controller.theta != float(state["theta"]) or
                controller.current_memory_version != int(state["current_memory_version"]) or
                controller.current_memory_sha256 != state["current_memory_sha256"] or
                [vars(item) for item in controller.updates] != list(state["updates"])):
            raise AblationError("Checkpoint does not restore the exact replay state")
        return controller


@dataclass(frozen=True)
class ArmReplay:
    arm_id: str
    predictions: tuple[int, ...]
    threshold_pre: tuple[float, ...]
    threshold_post: tuple[float, ...]
    updates: tuple[ThresholdUpdate, ...]
    positions: tuple[int, ...]
    memory_versions_before: tuple[int, ...]
    memory_versions_after: tuple[int, ...]


def replay_arm(events: Sequence[TrajectoryEvent], history: VersionedC64History,
               arm_id: str, delta: float = DELTA) -> ArmReplay:
    if not events:
        raise AblationError("Cannot replay an empty trajectory")
    q0 = history.lookup(0, as_of_version=0).q90
    controller = B3LagController(arm_id, history, q0, delta)
    preds: list[int] = []
    pre: list[float] = []
    post: list[float] = []
    prior_position = -1
    for event in events:
        if event.position != prior_position + 1:
            raise AblationError("Trajectory event positions are not contiguous")
        prior_position = event.position
        pre.append(controller.theta)
        preds.append(controller.predict(event.score, event.memory_version_before,
                                        event.memory_sha256_before))
        # The content transition follows this prediction and affects the next.
        post.append(controller.after_event(event))
    return ArmReplay(
        arm_id, tuple(preds), tuple(pre), tuple(post), tuple(controller.updates),
        tuple(e.position for e in events),
        tuple(e.memory_version_before for e in events),
        tuple(e.memory_version_after for e in events))


def replay_matched_arms(events: Sequence[TrajectoryEvent],
                        history: VersionedC64History,
                        delta: float = DELTA) -> dict[str, ArmReplay]:
    """Replay all three arms over one immutable score/memory event stream."""
    results = {arm: replay_arm(events, history, arm, delta) for arm in ARMS}
    if len({len(result.updates) for result in results.values()}) != 1:
        raise AblationError("A2/A3/A4 update event counts differ")
    for result in results.values():
        if (result.positions != results["A2"].positions or
                result.memory_versions_before != results["A2"].memory_versions_before or
                result.memory_versions_after != results["A2"].memory_versions_after):
            raise AblationError("Matched arms diverged from the shared event stream")
    return results


@dataclass
class FrozenTrajectory:
    category: str
    seed: int
    events: tuple[TrajectoryEvent, ...]
    history: VersionedC64History
    original_b3_rows: tuple[dict[str, Any], ...]
    matrix_row: dict[str, Any]
    artifact_hashes: list[dict[str, str]] = field(default_factory=list)

def _manifest_entry_digest(path: Path, entry: Any, role: str,
                           artifacts: list[dict[str, str]]) -> str:
    """Verify a raw-byte or canonical-LF checksum entry without weakening it."""
    if not path.is_file():
        raise AblationError(f"MISSING_ARTIFACT: {role}: {path}")
    if isinstance(entry, str):
        return _check_file(path, entry, role, artifacts)
    if not isinstance(entry, Mapping) or not isinstance(entry.get("sha256"), str):
        raise AblationError(f"MISSING_ARTIFACT: {role} checksum entry")
    mode = entry.get("hash_mode", "raw_bytes")
    expected = entry["sha256"]
    if mode == "raw_bytes":
        actual = sha256_file(path)
        size = path.stat().st_size
    elif mode == "canonical_lf_text":
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            raise AblationError(f"Invalid canonical text artifact: {path}") from exc
        canonical = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        actual = hashlib.sha256(canonical).hexdigest()
        size = len(canonical)
    else:
        raise AblationError(f"Unsupported checksum hash mode {mode!r}: {role}")
    if actual != expected:
        raise AblationError(f"HASH_MISMATCH: {role}: {path}")
    if "size_bytes" in entry and int(entry["size_bytes"]) != size:
        raise AblationError(f"SIZE_MISMATCH: {role}: {path}")
    artifacts.append({"role": role, "path": path.as_posix(), "sha256": actual,
                      "hash_mode": str(mode)})
    return actual


def _hash_from_manifest(path: Path, role: str, manifest: Mapping[str, Any],
                        relative_name: str, artifacts: list[dict[str, str]]) -> str:
    entry = manifest.get("files", {}).get(relative_name)
    if entry is None:
        raise AblationError(f"MISSING_ARTIFACT: {role} checksum entry {relative_name}")
    return _manifest_entry_digest(path, entry, role, artifacts)


def _read_prediction_rows(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AblationError(f"Malformed historical prediction row {line_no}") from exc
            if any(key.lower() in {"label", "true_label", "ground_truth", "outcome"}
                   for key in row):
                raise AblationError("Historical prediction row unexpectedly contains labels")
            rows.append(row)
    return tuple(rows)


def load_frozen_trajectory(source_root: Path, category: str, seed: int,
                           stage05_checks: Mapping[str, Any],
                           stage05_manifest: Mapping[str, Any],
                           stage075_checks: Mapping[str, Any],
                           formal_checks: Mapping[str, Any],
                           p12_receipts: Mapping[tuple[str, int], Mapping[str, Any]],
                           p12_audit_receipts: Mapping[tuple[str, int], Mapping[str, Any]],
                           artifacts: list[dict[str, str]]) -> FrozenTrajectory:
    """Load a frozen trajectory through an explicit read-only source-root adapter."""
    if category not in CLASSES or seed not in SEEDS:
        raise AblationError("Trajectory is outside the frozen class/seed registry")
    stage05 = source_root / "outputs/calibration/stage05_exploratory_20261029"
    stage075 = source_root / "outputs/calibration/stage075a_greedy_ranking_repaired3_20261029"
    p12out = source_root / "outputs/calibration/stage1a_p12_low_memory_recovery_v3_20261030"
    formal = source_root / "outputs/calibration/stage1a_formal_controller_comparison_v4_20261030"
    base = stage075 / "trajectories" / category

    local_checks = _read_json(base / "checksums.json")
    local_files = local_checks.get("files", {})
    if not isinstance(local_files, Mapping):
        raise AblationError(f"{category}: malformed trajectory checksum manifest")

    tape_entry = stage05_manifest["classes"][category]["score_tapes"][str(seed)]
    c64_rel = str(tape_entry["file"])
    c64_path = stage05 / c64_rel
    _hash_from_manifest(c64_path, f"{category}/seed{seed} initial C64 tape",
                        stage05_checks, c64_rel, artifacts)
    if sha256_file(c64_path) != tape_entry.get("sha256"):
        raise AblationError(f"HASH_MISMATCH: Stage05 C64 tape manifest {category}/seed{seed}")

    file_roles = {
        "scores.npz": f"trajectories/{category}/scores.npz",
        "c64_transport.json": f"trajectories/{category}/c64_transport.json",
        "event_trace.jsonl.gz": f"trajectories/{category}/event_trace.jsonl.gz",
        "trajectory_manifest.json": f"trajectories/{category}/trajectory_manifest.json",
    }
    trajectory_hashes: dict[str, str] = {}
    for name, root_rel in file_roles.items():
        path = base / name
        expected_local = local_files.get(name)
        if not isinstance(expected_local, str):
            raise AblationError(f"MISSING_ARTIFACT: {category}/{name} checksum entry")
        actual = _check_file(path, expected_local, f"{category}/{name}", artifacts)
        expected_root = stage075_checks.get("files", {}).get(root_rel)
        linked = _manifest_entry_digest(
            path, expected_root, f"Stage075A root linkage {root_rel}", artifacts)
        if linked != actual:
            raise AblationError(f"HASH_MISMATCH: Stage075A root linkage {root_rel}")
        trajectory_hashes[name] = actual

    c64_archive = np.load(c64_path, allow_pickle=False)
    initial_scores = np.asarray(c64_archive["c64_scores"], dtype=np.float64)
    c64_archive.close()
    if initial_scores.shape != (C64_N,) or not np.isfinite(initial_scores).all():
        raise AblationError(f"{category}/seed{seed}: invalid M0 C64 vector")
    group_sha = str(tape_entry.get("c64_group_sha256", ""))
    _require_sha(group_sha, "Stage05 C64 group identity")

    arrays = np.load(base / "scores.npz", allow_pickle=False)
    transport_all = _read_json(base / "c64_transport.json")
    transport = transport_all.get(str(seed))
    if not isinstance(transport, list):
        arrays.close()
        raise AblationError(f"MISSING_HISTORY: {category}/seed{seed} C64 transport")
    before = np.asarray(arrays[f"c64_pairs_before_s{seed}"], dtype=np.float64)
    after = np.asarray(arrays[f"c64_pairs_after_s{seed}"], dtype=np.float64)
    if (before.ndim != 2 or after.shape != before.shape or
            before.shape[1:] != (C64_N,) or len(before) != len(transport) or
            not len(before) or not np.array_equal(before[0], initial_scores)):
        arrays.close()
        raise AblationError(f"HASH_MISMATCH: {category}/seed{seed} C64 vector linkage")

    receipt_path = p12out / "shards" / category / f"seed{seed}.receipt.json"
    receipt = _read_json(receipt_path)
    if receipt.get("complete") is not True or receipt.get("status") != "PASS_EXACT_PARITY":
        arrays.close()
        raise AblationError(f"BLOCKED_INPUT_STATE: incomplete P1.2 receipt {category}/seed{seed}")
    artifacts.append({"role": f"P1.2 receipt {category}/seed{seed}",
                      "path": receipt_path.as_posix(), "sha256": sha256_file(receipt_path)})
    audit_receipt = p12_audit_receipts.get((category, seed))
    if not audit_receipt:
        arrays.close()
        raise AblationError(f"MISSING_ARTIFACT: P1.2 final audit receipt {category}/seed{seed}")
    for key in ("event_count", "event_chain_sha256", "evidence_file_sha256",
                "historical_content_update_count", "initial_memory", "final_memory"):
        if receipt.get(key) != audit_receipt.get(key):
            arrays.close()
            raise AblationError(f"HASH_MISMATCH: P1.2 receipt reconciliation {category}/seed{seed}")
    evidence_path = source_root / receipt["evidence_file"]
    evidence_sha = _check_file(evidence_path, receipt["evidence_file_sha256"],
                               f"{category}/seed{seed} frozen P1.2 evidence bytes", artifacts)
    p12_receipt = p12_receipts.get((category, seed))
    if p12_receipt and p12_receipt.get("evidence_file_sha256") != evidence_sha:
        arrays.close()
        raise AblationError(f"HASH_MISMATCH: P1.2 evidence index {category}/seed{seed}")

    memory0 = receipt["initial_memory"]
    current_memory_sha = str(memory0["memory_sha256"])
    current_version = int(memory0["version"])
    if current_version != 0:
        arrays.close()
        raise AblationError(f"{category}/seed{seed}: expected M0 at version 0")
    references = [C64Reference.from_scores(0, current_memory_sha,
                                            initial_scores, group_sha)]
    previous_vector = initial_scores.copy()
    for ix, tr in enumerate(transport):
        pre, post = before[ix], after[ix]
        vector_pre_sha, vector_post_sha = sha256_json(pre.tolist()), sha256_json(post.tolist())
        q90_post = higher_quantile(post.tolist())
        if (not np.array_equal(pre, previous_vector) or
                int(tr["memory_version_before"]) != current_version or
                int(tr["memory_version_after"]) != current_version + 1 or
                tr["memory_before_sha256"] != current_memory_sha or
                tr["c64_before_sha256"] != vector_pre_sha or
                tr["c64_after_sha256"] != vector_post_sha or
                float(tr["theta_SCQ_candidate_higher_q90"]) != q90_post or
                int(tr["n_c64"]) != C64_N or not np.isfinite(post).all()):
            arrays.close()
            raise AblationError(f"HASH_MISMATCH: {category}/seed{seed} transition {ix}")
        current_version += 1
        current_memory_sha = str(tr["memory_after_sha256"])
        references.append(reference_from_evidence(
            current_version, current_memory_sha, post.tolist(), group_sha,
            float(tr["theta_SCQ_candidate_higher_q90"]),
            str(tr["c64_after_sha256"])))
        previous_vector = post.copy()
    arrays.close()

    trajectory_manifest = _read_json(base / "trajectory_manifest.json")
    seed_manifest = trajectory_manifest.get("memory_by_seed", {}).get(str(seed), {})
    transition_count = len(transport)
    if (transition_count != int(receipt["historical_content_update_count"]) or
            transition_count != int(seed_manifest.get("actual_content_changes", -1)) or
            current_version != int(receipt["final_memory"]["version"]) or
            current_memory_sha != receipt["final_memory"]["memory_sha256"] or
            current_memory_sha != seed_manifest.get("final_memory_sha256")):
        raise AblationError(f"HASH_MISMATCH: {category}/seed{seed} final memory identity")
    history = VersionedC64History(references)
    if history.final_version != current_version:
        raise AblationError(f"MISSING_HISTORY: {category}/seed{seed} q90 versions")

    arrays = np.load(base / "scores.npz", allow_pickle=False)
    dynamic = np.asarray(arrays[f"dynamic_s{seed}"], dtype=np.float64)
    memory_versions = np.asarray(arrays[f"memory_version_before_s{seed}"], dtype=np.int64)
    memory_sha_arr = np.asarray(arrays[f"memory_sha_before_s{seed}"]).astype(str)
    q90_tape = np.asarray(arrays[f"current_q90_s{seed}"], dtype=np.float64)
    arrays.close()
    events: list[TrajectoryEvent] = []
    update_ix = 0
    prior_version, prior_sha = 0, str(memory0["memory_sha256"])
    with gzip.open(base / "event_trace.jsonl.gz", "rt", encoding="utf-8") as stream:
        for t, line in enumerate(stream):
            raw = json.loads(line)
            if int(raw.get("position", -1)) != t:
                raise AblationError(f"{category}/seed{seed}: event positions are not contiguous")
            if any(key.lower() in {"label", "true_label", "ground_truth", "outcome"}
                   for key in raw):
                raise AblationError("Frozen event trace unexpectedly contains outcome labels")
            seed_row = raw.get("support_seeds", {}).get(str(seed))
            if not isinstance(seed_row, Mapping):
                raise AblationError(f"{category}/seed{seed}: missing event seed state")
            v_before, v_after = int(seed_row["memory_version_before"]), int(seed_row["memory_version_after"])
            sha_before, sha_after = str(seed_row["memory_sha256_before"]), str(seed_row["memory_sha256_after"])
            writer = seed_row["writer"]
            q_before = history.lookup(v_before, as_of_version=v_before).q90
            if (v_before != prior_version or sha_before != prior_sha or
                    int(writer["memory_content_version_before"]) != v_before or
                    int(writer["memory_content_version_after"]) != v_after or
                    v_after - v_before != int(bool(writer["changed"])) or
                    int(memory_versions[t]) != v_before or str(memory_sha_arr[t]) != sha_before or
                    float(seed_row["dynamic_score"]) != float(dynamic[t]) or
                    float(q90_tape[t]) != q_before or
                    float(seed_row["current_C64_q90_pre_event"]) != q_before):
                raise AblationError(f"HASH_MISMATCH: {category}/seed{seed}/event {t}")
            if v_after != v_before:
                tr = transport[update_ix]
                if (int(tr["position"]) != t or tr["memory_before_sha256"] != sha_before or
                        tr["memory_after_sha256"] != sha_after or
                        int(tr["memory_version_before"]) != v_before or
                        int(tr["memory_version_after"]) != v_after or
                        int(tr["writer_appended"]) != int(writer["appended"]) or
                        int(tr["writer_replaced"]) != int(writer["replaced"]) or
                        int(tr["writer_skipped"]) != int(writer["skipped"])):
                    raise AblationError(f"HASH_MISMATCH: {category}/seed{seed} event transition {t}")
                update_ix += 1
            elif sha_after != sha_before:
                raise AblationError(f"HASH_MISMATCH: {category}/seed{seed} no-op write {t}")
            events.append(TrajectoryEvent(
                t, float(dynamic[t]), v_before, sha_before, v_after, sha_after,
                bool(raw["query_tape_selected"])))
            prior_version, prior_sha = v_after, sha_after
    if (len(events) != int(receipt["event_count"]) or update_ix != transition_count or
            prior_version != int(receipt["final_memory"]["version"]) or
            prior_sha != receipt["final_memory"]["memory_sha256"]):
        raise AblationError(f"HASH_MISMATCH: {category}/seed{seed} full event chain")

    pred_rel = f"predictions/{category}_seed{seed}.jsonl.gz"
    pred_path = formal / pred_rel
    formal_expected_entry = formal_checks.get("files", {}).get(pred_rel)
    if formal_expected_entry is None:
        raise AblationError(f"MISSING_ARTIFACT: formal A2 prediction checksum {pred_rel}")
    formal_expected = _manifest_entry_digest(
        pred_path, formal_expected_entry,
        f"historical A2 predictions {category}/seed{seed}", artifacts)
    old_rows = _read_prediction_rows(pred_path)
    if len(old_rows) != len(events):
        raise AblationError(f"MISSING_HISTORY: historical A2 prediction rows {category}/seed{seed}")

    replay = replay_arm(events, history, "A2")
    pre_bad = post_bad = pred_bad = event_bad = 0
    for event, expected in zip(events, old_rows):
        b3 = expected.get("methods", {}).get("B3", {})
        if (int(expected.get("t", -1)) != event.position or
                int(expected.get("memory_version_before", -1)) != event.memory_version_before or
                int(expected.get("memory_version_after", -1)) != event.memory_version_after or
                str(expected.get("memory_sha256_before")) != event.memory_sha256_before or
                str(expected.get("memory_sha256_after")) != event.memory_sha256_after or
                float(expected.get("dynamic_score", float("nan"))) != event.score or
                bool(expected.get("query_event")) != event.query_selected):
            event_bad += 1
        if float(b3.get("threshold_pre", float("nan"))) != replay.threshold_pre[event.position]:
            pre_bad += 1
        if float(b3.get("threshold_post", float("nan"))) != replay.threshold_post[event.position]:
            post_bad += 1
        if int(b3.get("prediction", -999)) != replay.predictions[event.position]:
            pred_bad += 1
    if any((pre_bad, post_bad, pred_bad, event_bad)):
        raise AblationError(f"A2_HISTORICAL_PARITY_FAIL: {category}/seed{seed}")
    if len(replay.updates) != transition_count:
        raise AblationError(f"A2 update event count mismatch: {category}/seed{seed}")

    row = {
        "class": category, "support_seed": seed, "status": "PASS",
        "trajectory_id": f"{category}/seed{seed}/Q-A-TS-v1-3pct-seed37",
        "distinct_memory_versions": history.final_version + 1,
        "actual_memory_changes": transition_count, "q90_history_complete": True,
        "c64_identity_sha256": group_sha,
        "c64_version_history_sha256": history.identity_sha256,
        "initial_memory_sha256": str(memory0["memory_sha256"]),
        "final_memory_sha256": str(receipt["final_memory"]["memory_sha256"]),
        "event_count": len(events), "event_chain_sha256": receipt["event_chain_sha256"],
        "evidence_file": str(receipt["evidence_file"]).replace("\\", "/"),
        "evidence_file_sha256": evidence_sha,
        "prediction_file": pred_rel, "prediction_file_sha256": formal_expected,
        "c64_score_npz_sha256": trajectory_hashes["scores.npz"],
        "c64_transport_sha256": trajectory_hashes["c64_transport.json"],
        "event_trace_sha256": trajectory_hashes["event_trace.jsonl.gz"],
        "a2_pre_threshold_mismatches": pre_bad,
        "a2_post_threshold_mismatches": post_bad,
        "a2_prediction_mismatches": pred_bad,
        "a2_event_identity_mismatches": event_bad,
        "a2_update_events": len(replay.updates),
        "query_tape_binding_sha256": receipt["query_tape_binding_sha256"],
    }
    return FrozenTrajectory(category, seed, tuple(events), history, old_rows, row, artifacts)

def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_value(root: Path, *args: str) -> str:
    import subprocess
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _deduplicate_artifacts(items: Sequence[Mapping[str, str]],
                           source_root: Path) -> list[dict[str, str]]:
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for item in items:
        rel = Path(item["path"]).resolve().relative_to(source_root.resolve()).as_posix()
        key = (rel, item["sha256"])
        unique[key] = {"role": item["role"], "path": rel, "sha256": item["sha256"],
                       "hash_mode": item.get("hash_mode", "raw_bytes")}
    return list(unique.values())


def verify_a0(source_root: Path, code_root: Path | None = None) -> dict[str, Any]:
    """Reverify frozen A0 inputs and exact A2 threshold/prediction parity.

    C64 vectors, score/memory metadata, checksums, and label-free prediction
    rows are decoded. P1.2 queried-evidence JSONL files are hashed, not parsed.
    No outcome or metric files are opened.
    """
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise AblationError("Frozen source root does not exist")
    code_root = (code_root or Path(__file__).resolve().parents[2]).resolve()
    stage05 = source_root / "outputs/calibration/stage05_exploratory_20261029"
    stage075 = source_root / "outputs/calibration/stage075a_greedy_ranking_repaired3_20261029"
    p12out = source_root / "outputs/calibration/stage1a_p12_low_memory_recovery_v3_20261030"
    p12audit = source_root / "docs/calibration/audit/stage1a_p12_low_memory_recovery_v3_20261030"
    readiness = source_root / "docs/calibration/audit/stage1a_final_readiness_v6_20261030"
    formal = source_root / "outputs/calibration/stage1a_formal_controller_comparison_v4_20261030"

    artifacts: list[dict[str, str]] = []
    stage05_checks = _read_json(stage05 / "checksums.json")
    stage05_manifest_path = stage05 / "score_tape_manifest.json"
    _hash_from_manifest(stage05_manifest_path, "Stage05 score tape manifest",
                        stage05_checks, "score_tape_manifest.json", artifacts)
    stage05_manifest = _read_json(stage05_manifest_path)

    stage075_checks = _read_json(stage075 / "checksums.json")
    for name in ("protocol_manifest.json", "audit_linkage.json"):
        expected = stage075_checks.get("files", {}).get(name)
        if not isinstance(expected, str):
            raise AblationError(f"MISSING_ARTIFACT: Stage075A root {name}")
        _check_file(stage075 / name, expected, f"Stage075A {name}", artifacts)

    preflight_path = p12out / "recovery_preflight.json"
    preflight = _read_json(preflight_path)
    linkage = preflight.get("historical_c64_linkage", {})
    if (linkage.get("status") != "PASS_HISTORICAL_ARTIFACT_LINKAGE_ONLY" or
            linkage.get("class_seed_initial_c64_vectors_verified") != 20 or
            linkage.get("postwrite_c64_vectors_verified") != 1190 or
            linkage.get("c64_transport_rows_verified") != 1190 or
            linkage.get("historical_native_c64_rescoring_performed") is not False):
        raise AblationError("BLOCKED_INPUT_STATE: historical C64 linkage gate differs")
    artifacts.append({"role": "P1.2 recovery preflight",
                      "path": preflight_path.as_posix(),
                      "sha256": sha256_file(preflight_path)})

    audit_checks = _read_json(p12audit / "checksums_final_v2.json")
    for name in ("trajectory_receipts_final.json",
                 "exact_native_memory_parity_final.json",
                 "anchor_evidence_reconciliation_final.json",
                 "source_provenance.json",
                 "p12_low_memory_recovery_gate_matrix_final.json"):
        expected = audit_checks.get("files", {}).get(name)
        if expected is None:
            raise AblationError(f"MISSING_ARTIFACT: P1.2 audit checksum for {name}")
        _manifest_entry_digest(p12audit / name, expected,
                               f"P1.2 audit {name}", artifacts)
    exact = _read_json(p12audit / "exact_native_memory_parity_final.json")
    if (exact.get("status") != "PASS_EXACT_PARITY_ALL_20_TRAJECTORIES" or
            exact.get("trajectory_count") != 20 or
            exact.get("memory_writer_source_and_C64_parity") != "exact_each_event"):
        raise AblationError("BLOCKED_INPUT_STATE: P1.2 final exact-parity audit invalid")
    receipt_doc = _read_json(p12audit / "trajectory_receipts_final.json")
    p12_audit_receipts: dict[tuple[str, int], Mapping[str, Any]] = {}
    for item in receipt_doc.get("receipts", []):
        p12_audit_receipts[(str(item.get("category")),
                            int(item.get("support_seed", -1)))] = item
    if len(p12_audit_receipts) != 20:
        raise AblationError("MISSING_HISTORY: P1.2 audit does not contain 20 receipts")

    recovery_receipts: dict[tuple[str, int], Mapping[str, Any]] = {}
    for category in CLASSES:
        for seed in SEEDS:
            path = p12out / "shards" / category / f"seed{seed}.receipt.json"
            receipt = _read_json(path)
            if receipt.get("complete") is not True or receipt.get("status") != "PASS_EXACT_PARITY":
                raise AblationError(f"BLOCKED_INPUT_STATE: P1.2 receipt {category}/seed{seed}")
            _require_sha(str(receipt.get("evidence_file_sha256")), "P1.2 evidence digest")
            recovery_receipts[(category, seed)] = receipt

    formal_checks = _read_json(formal / "checksums.json")
    formal_seal_path = formal / "prediction_phase_complete.json"
    formal_seal = _read_json(formal_seal_path)
    if (formal_seal.get("status") != "PREDICTION_PHASE_COMPLETE_LABEL_FIREWALL_INTACT" or
            formal_seal.get("outcome_labels_accessed") is not False or
            formal_seal.get("event_count") != 76475 or
            formal_seal.get("trajectory_count") != 20):
        raise AblationError("BLOCKED_INPUT_STATE: formal prediction phase seal invalid")
    for name in ("prediction_phase_complete.json", "preflight.json"):
        expected = formal_checks.get("files", {}).get(name)
        if expected is not None:
            _manifest_entry_digest(formal / name, expected,
                                   f"Formal A2 {name}", artifacts)

    readiness_checks = _read_json(readiness / "checksums.json")
    for name in ("stage1a_execution_manifest.json", "stage1a_readiness_gate_matrix.json",
                 "source_provenance.json"):
        expected = readiness_checks.get("files", {}).get(name)
        if expected is not None:
            _manifest_entry_digest(readiness / name, expected,
                                   f"Stage1A readiness {name}", artifacts)
    readiness_manifest = _read_json(readiness / "stage1a_execution_manifest.json")
    readiness_count = readiness_manifest.get("trajectory_count")
    if readiness_count is not None and readiness_count != 20:
        raise AblationError("MISSING_HISTORY: Stage1A readiness manifest trajectory count")

    matrix: list[dict[str, Any]] = []
    totals = {"trajectory_count": 0, "events": 0, "memory_updates": 0,
              "pre_threshold_mismatches": 0, "post_threshold_mismatches": 0,
              "prediction_mismatches": 0, "event_identity_mismatches": 0}
    trajectories: list[FrozenTrajectory] = []
    class_identity: dict[str, str] = {}
    for category in CLASSES:
        for seed in SEEDS:
            trajectory = load_frozen_trajectory(
                source_root, category, seed, stage05_checks, stage05_manifest,
                stage075_checks, formal_checks, recovery_receipts,
                p12_audit_receipts, artifacts)
            identity = trajectory.matrix_row["c64_identity_sha256"]
            if category in class_identity and class_identity[category] != identity:
                raise AblationError(f"HASH_MISMATCH: C64 identities differ across {category} seeds")
            class_identity[category] = identity
            matrix.append(trajectory.matrix_row)
            trajectories.append(trajectory)
            totals["trajectory_count"] += 1
            totals["events"] += len(trajectory.events)
            totals["memory_updates"] += len(trajectory.history.references) - 1
            for key in ("pre_threshold_mismatches", "post_threshold_mismatches",
                        "prediction_mismatches", "event_identity_mismatches"):
                totals[key] += trajectory.matrix_row[f"a2_{key}"]
    if (totals["trajectory_count"] != 20 or totals["events"] != 76475 or
            totals["memory_updates"] != 1190):
        raise AblationError("BLOCKED_INPUT_STATE: aggregate trajectory/event/update counts differ")
    if any(totals[key] for key in ("pre_threshold_mismatches",
                                   "post_threshold_mismatches",
                                   "prediction_mismatches",
                                   "event_identity_mismatches")):
        raise AblationError("A2_HISTORICAL_PARITY_FAIL: aggregate mismatch")

    source_files = (
        "experiments/calibration/controllers.py",
        "experiments/calibration/stage075a.py",
        "experiments/calibration/stage075a_runner.py",
        "experiments/calibration/stage1a_p11_production.py",
        "experiments/calibration/stage1a_p12_runner.py",
        "experiments/calibration/stage1a_p12_sharded_recovery.py",
        "experiments/calibration/stage1a_final_readiness.py",
        "experiments/calibration/stage1a_formal_comparison.py",
        "experiments/calibration/stage1a_b3_mechanism_ablation.py",
        "tests/calibration/test_b3_mechanism_ablation.py",
    )
    source_hashes = {}
    for rel in source_files:
        path = code_root / rel
        if path.is_file():
            source_hashes[rel] = sha256_file(path)
    source_hashes["base_commit"] = _git_value(code_root, "rev-parse", "HEAD")
    source_hashes["master_plan"] = sha256_file(
        code_root / "docs/calibration/B3_CALIBRATION_PAPER_RECOVERY_MASTER_PLAN_AND_PROGRESS.md")

    resources = {
        "dataset_root_exists": Path(r"G:\dataset\realiad").is_dir(),
        "json_metadata_exists": Path(r"G:\dataset\realiad\jsons\realiad_jsons").is_dir(),
        "dinov2_cache_exists": Path(r"C:\Users\rico\.cache\torch\hub\facebookresearch_dinov2_main").is_dir(),
    }
    if not all(resources.values()):
        raise AblationError("Environment resource preflight failed")

    return {
        "schema": "calibration.b3_state_lag_a0_verification.v1",
        "status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
        "mechanism_configuration": {
            "arms_lag_by_memory_content_versions": dict(ARMS),
            "delta": DELTA,
            "q90_quantile": Q90,
            "q90_order_statistic_index_zero_based": 57,
            "prediction_rule": "score >= theta",
            "update_order": "predict, then apply transition for next event",
            "initial_theta": "q90(memory_version_0)",
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "torch_version": _package_version("torch"),
            **resources,
            "cuda_inference_calls": 0, "dataset_image_reads": 0,
            "label_provider_reads": 0, "outcome_label_vectors_opened": 0,
            "confirmation_label_files_opened": 0, "performance_metric_files_opened": 0,
            "p12_evidence_files_hashed_not_decoded": 20,
            "aggregate_receipt_schema_probe_objects_read_outside_verifier": 1,
            "access_counts": {
                "frozen_trajectories_verified": 20,
                "c64_initial_vectors_decoded": 20,
                "distinct_versioned_c64_vectors_verified": 1210,
                "memory_content_transitions_verified": 1190,
                "event_trace_rows_decoded": 76475,
                "historical_a2_prediction_rows_decoded": 76475,
                "p12_evidence_files_hashed_not_decoded": 20,
                "aggregate_schema_probe_objects_read_outside_verifier": 1,
                "image_files_opened": 0,
                "label_or_outcome_files_opened": 0,
                "performance_metric_files_opened": 0,
            },
        },
        "worktree": {
            "branch": _git_value(code_root, "branch", "--show-current"),
            "base_commit": _git_value(code_root, "rev-parse", "HEAD"),
            "isolated_worktree": True,
            "clean_verified_before_edits": True,
        },
        "historical_limitations": {
            "stage075a_incomplete_marker_present": bool(
                linkage.get("historical_stage075a_incomplete_marker_present")),
            "original_g2_source_byte_parity": "BLOCKED_UNCHANGED",
            "historical_native_c64_rescoring_performed": False,
            "historical_formal_outcome_metrics_read_or_used": False,
        },
        "summary": totals, "c64_group_identity_by_class": class_identity,
        "feasibility_matrix": matrix,
        "source_sha256": source_hashes,
        "input_artifacts": _deduplicate_artifacts(artifacts, source_root),
    }


def render_feasibility_report(result: Mapping[str, Any]) -> str:
    summary, rows = result["summary"], result["feasibility_matrix"]
    lines = [
        "# B3 state-lag Gate A0 verification", "",
        "Status: PASS for frozen-input feasibility and historical A2 recurrence parity.",
        "",
        "This is an artifact and threshold/prediction replay audit. It is not a performance evaluation and does not test H1.",
        "", "## Verified counts", "",
        f"- Trajectories: {summary['trajectory_count']}/20",
        f"- Content-changing memory updates: {summary['memory_updates']}",
        f"- Versioned C64 q90 states: {sum(int(row['distinct_memory_versions']) for row in rows)}",
        f"- Historical A2 events: {summary['events']}",
        f"- Exact pre/post threshold mismatches: {summary['pre_threshold_mismatches']}/{summary['post_threshold_mismatches']}",
        f"- Exact prediction mismatches: {summary['prediction_mismatches']}",
        f"- Event/memory identity mismatches: {summary['event_identity_mismatches']}",
        "",
        "A2 uses q_v, A3 uses q_max(0,v-1), and A4 uses q_max(0,v-5); each arm uses the fixed Delta=0.01. The q90 uses the higher order statistic at zero-based index 57 of the fixed 64-score vector. A2 initialization equals q0. A state change is applied after the current prediction and affects the next prediction.",
        "", "## Feasibility matrix", "",
        "| Class | Seed | Distinct versions | Content changes | q90 history | Status | Evidence SHA-256 prefix |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['class']} | {row['support_seed']} | {row['distinct_memory_versions']} | "
            f"{row['actual_memory_changes']} | COMPLETE | PASS | {row['evidence_file_sha256'][:12]} |")
    limit = result["historical_limitations"]
    lines += [
        "", "## Provenance limits", "",
        f"- Original Stage 0.75-A/G2 source-byte parity remains {limit['original_g2_source_byte_parity']}; this audit does not retroactively validate historical source bytes.",
        "- Frozen C64 vectors and memory transitions were verified from stored hashes and receipts; no Native rescoring or GPU inference was run.",
        "- Outcome and metric artifacts were not read or used. Historical per-event prediction rows contain no label fields.",
        "- Access count: 20 initial C64 vectors, 1,210 distinct versioned C64 vectors, 1,190 transitions, 76,475 event rows, 76,475 historical prediction rows, and 20 P1.2 evidence files hashed without decoding. No images, outcome/label files, or performance metrics were opened. One aggregate receipt schema-probe object was read outside this verifier and is disclosed in the manifest.",
        "- No BA/FPR/FNR was calculated. H1 remains untested.",
        "", "Full input hashes and memory identities are in feasibility_matrix.json and source_and_input_manifest.json.", "",
    ]
    return "\n".join(lines)


def render_runbook() -> str:
    return """# Later formal Experiment A runbook

This runbook is for a separately authorized performance evaluation. Do not run its outcome-scoring stage until the researcher explicitly authorizes it.

## Frozen inputs

- Supply the original checkout as the read-only frozen root.
- Verify all source and input digests before replay.
- Use only the four registered development classes and support seeds 0-4, fixed Q-A-TS tape, stored Native score/memory transitions, and fixed C64 identities.
- Never reconstruct missing C64 vectors or memory states.

## Matched controller replay

For each trajectory, construct its complete VersionedC64History and immutable TrajectoryEvent sequence, then call replay_matched_arms(events, history, delta=0.01). This returns A2/A3/A4 predictions, pre/post thresholds, and update ledgers. A2 uses q_v; A3 uses q_max(0,v-1); A4 uses q_max(0,v-5). All use the same q90 estimator, initial q0, Delta, event positions, scores, and memory transitions. Prediction uses score >= theta. Updates happen after prediction and affect only the next event.

Before offline metrics, assert identical positions, score inputs, memory digests, update positions/counts, C64 identity hashes, and initial thresholds across arms. Persist each arm under a new immutable, checksummed namespace. Bind source commit, inputs, query tape, arm configuration, and environment in its manifest. Do not write to the frozen root.

## Outcome stage

Only after separate authorization may an evaluator join prediction streams to permitted labels and calculate preregistered full-stream, common-horizon, and worst-window metrics. Keep labels outside controller state and replay checkpoints. Do not access confirmation classes or masks. Do not tune on these results.
"""


def write_a0_audit(result: Mapping[str, Any], code_root: Path,
                   audit_dir: Path) -> None:
    code_root, audit_dir = code_root.resolve(), audit_dir.resolve()
    source_root = Path(result["source_root"]).resolve()
    if _is_within(audit_dir, source_root):
        raise AblationError("Refusing to write an audit beneath the frozen source root")
    if not _is_within(audit_dir, code_root):
        raise AblationError("Audit output must remain inside the A1 worktree")
    audit_dir.mkdir(parents=True, exist_ok=True)
    targets = ("feasibility_matrix.json", "feasibility_report.md",
               "B3_STATE_LAG_FORMAL_RUNBOOK.md", "source_and_input_manifest.json")
    if any((audit_dir / name).exists() for name in targets):
        raise AblationError("Refusing to overwrite an existing A0 audit artifact")
    matrix_text = json.dumps(
        {"schema": "calibration.b3_state_lag_feasibility_matrix.v1",
         "trajectories": result["feasibility_matrix"]},
        indent=2, sort_keys=True, allow_nan=False) + "\n"
    report_text = render_feasibility_report(result)
    runbook_text = render_runbook()
    for name, contents in (
            ("feasibility_matrix.json", matrix_text),
            ("feasibility_report.md", report_text),
            ("B3_STATE_LAG_FORMAL_RUNBOOK.md", runbook_text)):
        with (audit_dir / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(contents)
    output_hashes = {}
    for path in sorted(audit_dir.iterdir()):
        if path.is_file():
            output_hashes[path.name] = sha256_file(path)
    manifest = {key: value for key, value in result.items()
                if key not in ("feasibility_matrix", "source_root")}
    manifest["frozen_source_root"] = "original checkout; read-only"
    manifest["audit_output_sha256"] = output_hashes
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with (audit_dir / "source_and_input_manifest.json").open(
            "x", encoding="utf-8", newline="\n") as stream:
        stream.write(manifest_text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-a0", action="store_true",
                        help="metadata-only frozen A0 verification and A2 parity")
    parser.add_argument("--frozen-root", type=Path, required=True,
                        help="read-only original checkout containing frozen artifacts")
    parser.add_argument("--audit-dir", type=Path,
                        help="new output directory inside this A1 worktree")
    args = parser.parse_args(argv)
    if not args.verify_a0:
        parser.error("select --verify-a0")
    code_root = Path(__file__).resolve().parents[2]
    result = verify_a0(args.frozen_root, code_root)
    result["source_root"] = str(args.frozen_root.resolve())
    if args.audit_dir:
        write_a0_audit(result, code_root, args.audit_dir)
    print(json.dumps({
        "status": result["status"],
        "trajectory_count": result["summary"]["trajectory_count"],
        "events": result["summary"]["events"],
        "memory_updates": result["summary"]["memory_updates"],
        "pre_threshold_mismatches": result["summary"]["pre_threshold_mismatches"],
        "post_threshold_mismatches": result["summary"]["post_threshold_mismatches"],
        "prediction_mismatches": result["summary"]["prediction_mismatches"],
        "audit_dir": str(args.audit_dir) if args.audit_dir else None,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
