"""Gated launcher for the frozen FS4 S/CL/CE scientific campaign.

The no-argument operation is inert. Real-data staging, Support parity, Stage A
and Stage B each require a separate campaign-bound authorization record.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Mapping

from .fs4_durable_ledger import canonical, digest
from .fs4_joint_campaign_contract import (ARM_IDS, EXPECTED_CATEGORIES, EXPECTED_EVENTS,
    CampaignContractError, build_trajectory_manifests, load_spec, spec_identity,
    validate_registration, validate_spec, validate_support_preflight)

ROOT = Path(__file__).resolve().parents[2]
STAGE05_FROZEN_INPUTS = ROOT / "outputs/calibration/stage05_exploratory_20261029/frozen_inputs.json"
AUTH_SCHEMA = "FS4_JOINT_CAMPAIGN_AUTHORIZATION_V1"
CAMPAIGN_SCHEMA = "FS4_JOINT_CAMPAIGN_REGISTRATION_V1"
SCOPES = {"STAGING", "SUPPORT_PREFLIGHT", "RESOURCE_PREFLIGHT", "STAGE_A", "STAGE_B"}
TRAJECTORY_WALL_LIMIT_SECONDS = 604800
CAMPAIGN_WALL_LIMIT_SECONDS = 1814400


class CampaignDeadlineExceeded(CampaignContractError):
    def __init__(self, status: str) -> None:
        self.status = status
        super().__init__(status)


class RecoverableExecutionInterruption(RuntimeError):
    """Explicit test/infrastructure interruption at a recoverable event boundary."""


class CampaignTimingLedger:
    """Append-only campaign wall clock; restart gaps are charged conservatively."""

    SCHEMA = "FS4_CAMPAIGN_TIMING_LEDGER_V1"

    def __init__(self, path: Path, *, campaign_id: str, source_commit: str,
                 spec_sha256: str | None = None, clock: Any = time) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_stream = self.path.with_suffix(self.path.suffix + ".lock").open("a+b")
        if self._lock_stream.seek(0, os.SEEK_END) == 0:
            self._lock_stream.write(b"\0")
            self._lock_stream.flush()
        self._lock_stream.seek(0)
        try:
            import msvcrt
            msvcrt.locking(self._lock_stream.fileno(), msvcrt.LK_NBLCK, 1)
            self._unlock = lambda: msvcrt.locking(self._lock_stream.fileno(), msvcrt.LK_UNLCK, 1)
        except ImportError:
            import fcntl
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._unlock = lambda: fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            self._lock_stream.close()
            raise CampaignContractError("another Stage A process owns the timing ledger") from exc
        self._closed = False
        self.campaign_id = campaign_id
        self.source_commit = source_commit
        self.spec_sha256 = spec_sha256 or spec_identity()
        self.clock = clock
        self.pid = os.getpid()
        frozen_spec = load_spec()
        if (_spec_value(frozen_spec, "trajectory_wall_limit_seconds") != TRAJECTORY_WALL_LIMIT_SECONDS or
                _spec_value(frozen_spec, "campaign_wall_limit_seconds") != CAMPAIGN_WALL_LIMIT_SECONDS):
            raise CampaignContractError("frozen wall-time limits differ from the executor constants")
        self._mono = int(clock.monotonic_ns())
        self._wall = int(clock.time_ns())
        self.records = self._read_records()
        self.completed_trajectories = {row.get("payload", {}).get("trajectory_id") for row in self.records
            if row.get("kind") == "TRAJECTORY_COMPLETE"}
        self.deadline_status = next((row.get("deadline_status") or
            (row.get("payload", {}).get("status") if row.get("kind") == "DEADLINE_EXCEEDED" else None)
            for row in reversed(self.records) if row.get("deadline_status") or row.get("kind") == "DEADLINE_EXCEEDED"), None)
        self.unclean_previous_process = bool(self.records and self.records[-1].get("process_open"))
        if self.records:
            last = self.records[-1]
            if (last.get("campaign_id") != campaign_id or last.get("source_commit") != source_commit or
                    last.get("spec_sha256") != self.spec_sha256):
                raise CampaignContractError("timing ledger belongs to a different campaign/source/spec")
            gap = self._wall - last["unix_ns"]
            if gap < 0:
                raise CampaignContractError("timing ledger wall clock moved backwards")
            campaign_elapsed = last["campaign_elapsed_ns"] + gap
            trajectory_elapsed = dict(last["trajectory_elapsed_ns"])
            active = last.get("active_trajectory")
            if active is not None:
                trajectory_elapsed[active] = trajectory_elapsed.get(active, 0) + gap
            if self.deadline_status:
                raise CampaignDeadlineExceeded(self.deadline_status)
            self.active_trajectory = active
            self.campaign_elapsed_ns = campaign_elapsed
            self.trajectory_elapsed_ns = trajectory_elapsed
            self._append("PROCESS_RESUME", {"previous_process_id": last.get("process_id"),
                "interrupted_gap_ns": gap, "interrupted_period_charged_to": active})
        else:
            self.active_trajectory = None
            self.campaign_elapsed_ns = 0
            self.trajectory_elapsed_ns = {name: 0 for name in ARM_IDS}
            self._append("CAMPAIGN_ORIGIN", {"origin_unix_ns": self._wall,
                "origin_process_monotonic_ns": self._mono})
            self._append("PROCESS_START", {"process_start_unix_ns": self._wall})

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        previous = "0" * 64
        records = []
        for index, line in enumerate(self.path.read_bytes().splitlines()):
            try:
                row = json.loads(line)
                body = {key: value for key, value in row.items() if key != "record_sha256"}
                expected = _sha(canonical(body))
                if (row.get("schema") != self.SCHEMA or row.get("sequence") != index or
                        row.get("previous_sha256") != previous or row.get("record_sha256") != expected or
                        type(row.get("campaign_elapsed_ns")) is not int or row["campaign_elapsed_ns"] < 0 or
                        not isinstance(row.get("trajectory_elapsed_ns"), dict) or
                        type(row.get("unix_ns")) is not int):
                    raise ValueError
                records.append(row)
                previous = expected
            except Exception as exc:
                raise CampaignContractError("timing ledger is malformed, truncated or tampered") from exc
        if not records:
            raise CampaignContractError("timing ledger exists without a complete origin record")
        if records[0].get("kind") != "CAMPAIGN_ORIGIN":
            raise CampaignContractError("timing ledger origin record is missing")
        prior_campaign, prior_trajectory, prior_wall, process_mono = -1, {}, -1, {}
        exceeded_without_record = False
        recorded_deadline_status = None
        valid_statuses = {"TRAJECTORY_WALL_LIMIT_EXCEEDED", "CAMPAIGN_WALL_LIMIT_EXCEEDED"}
        valid_kinds = {"CAMPAIGN_ORIGIN", "PROCESS_START", "PROCESS_RESUME", "PROCESS_STOP",
            "TRAJECTORY_ORIGIN", "TRAJECTORY_START", "TRAJECTORY_COMPLETE", "TRAJECTORY_STOP",
            "DEADLINE_EXCEEDED", "CHECKPOINT", "RECOVERY_CHECKPOINT_SELECTED",
            "RECOVERY_REPLAY_COMPLETED", "RECOVERY_LEDGER_VERIFIED", "CAMPAIGN_COMPLETE", "CAMPAIGN_FAILURE"}
        for row in records:
            campaign_elapsed = row["campaign_elapsed_ns"]
            by_trajectory = row["trajectory_elapsed_ns"]
            if (campaign_elapsed < prior_campaign or row["unix_ns"] < prior_wall or
                    set(by_trajectory) != set(ARM_IDS) or
                    any(type(value) is not int or value < prior_trajectory.get(arm, 0)
                        for arm, value in by_trajectory.items()) or
                    sum(by_trajectory.values()) > campaign_elapsed or
                    row.get("active_trajectory") not in (None, *ARM_IDS) or
                    type(row.get("process_open")) is not bool or
                    row["process_open"] != (row.get("kind") != "PROCESS_STOP") or
                    row.get("kind") not in valid_kinds or
                    not isinstance(row.get("payload"), dict)):
                raise CampaignContractError("timing ledger totals or process state are inconsistent")
            pid = row.get("process_id")
            monotonic_value = row.get("process_monotonic_ns")
            if type(pid) is not int or type(monotonic_value) is not int or monotonic_value < process_mono.get(pid, -1):
                raise CampaignContractError("timing ledger process clock is inconsistent")
            process_mono[pid] = monotonic_value
            status = row.get("deadline_status")
            if status is not None and status not in valid_statuses:
                raise CampaignContractError("timing ledger contains an unknown deadline status")
            if row.get("kind") == "DEADLINE_EXCEEDED":
                declared = row.get("payload", {}).get("status")
                if declared not in valid_statuses or status != declared or recorded_deadline_status not in (None, declared):
                    raise CampaignContractError("timing ledger deadline status is inconsistent")
                recorded_deadline_status = declared
            elif recorded_deadline_status is not None and status != recorded_deadline_status:
                raise CampaignContractError("timing ledger dropped a prior deadline failure")
            elif recorded_deadline_status is None and status is not None:
                raise CampaignContractError("timing ledger has an unrecorded deadline status")
            if (campaign_elapsed >= CAMPAIGN_WALL_LIMIT_SECONDS * 1_000_000_000 or
                    any(value >= TRAJECTORY_WALL_LIMIT_SECONDS * 1_000_000_000
                        for value in by_trajectory.values())) and status is None and row.get("kind") != "DEADLINE_EXCEEDED":
                exceeded_without_record = True
            prior_campaign, prior_trajectory, prior_wall = campaign_elapsed, by_trajectory, row["unix_ns"]
        if exceeded_without_record and not any(row.get("kind") == "DEADLINE_EXCEEDED" for row in records):
            raise CampaignContractError("timing ledger passed a frozen deadline without recording failure")
        return records

    def _current_elapsed(self) -> tuple[int, dict[str, int]]:
        delta = int(self.clock.monotonic_ns()) - self._mono
        if delta < 0:
            raise CampaignContractError("process monotonic clock moved backwards")
        trajectory = dict(self.trajectory_elapsed_ns)
        if self.active_trajectory is not None:
            trajectory[self.active_trajectory] = trajectory.get(self.active_trajectory, 0) + delta
        return self.campaign_elapsed_ns + delta, trajectory

    @property
    def prefix_sha256(self) -> str:
        return self.records[-1]["record_sha256"] if self.records else "0" * 64

    def _append(self, kind: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        now_mono, now_wall = int(self.clock.monotonic_ns()), int(self.clock.time_ns())
        delta = now_mono - self._mono
        if delta < 0 or now_wall < self._wall:
            raise CampaignContractError("process clock moved backwards")
        self.campaign_elapsed_ns += delta
        if self.active_trajectory is not None:
            self.trajectory_elapsed_ns[self.active_trajectory] = (
                self.trajectory_elapsed_ns.get(self.active_trajectory, 0) + delta)
        self._mono, self._wall = now_mono, now_wall
        body = {"schema": self.SCHEMA, "sequence": len(self.records),
            "previous_sha256": self.prefix_sha256, "kind": kind, "payload": dict(payload or {}),
            "campaign_id": self.campaign_id, "spec_sha256": self.spec_sha256,
            "source_commit": self.source_commit, "process_id": self.pid, "process_open": kind != "PROCESS_STOP",
            "unix_ns": now_wall, "process_monotonic_ns": now_mono,
            "campaign_elapsed_ns": self.campaign_elapsed_ns,
            "trajectory_elapsed_ns": dict(self.trajectory_elapsed_ns),
            "active_trajectory": self.active_trajectory,
            "deadline_status": self.deadline_status}
        if self.records and self.records[-1].get("deadline_status"):
            body["deadline_status"] = self.records[-1]["deadline_status"]
        row = {**body, "record_sha256": _sha(canonical(body))}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = canonical(row) + b"\n"
        if not self.path.exists():
            _write_new(self.path, data)
        else:
            with self.path.open("ab", buffering=0) as stream:
                if stream.write(data) != len(data):
                    raise OSError("short campaign timing-ledger append")
                os.fsync(stream.fileno())
        self.records.append(row)
        return row

    def record(self, kind: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self._append(kind, payload)

    def start_trajectory(self, trajectory: str) -> None:
        if trajectory not in ARM_IDS:
            raise CampaignContractError("unknown timed trajectory")
        if trajectory in self.completed_trajectories:
            self.check(trajectory)
            return
        if self.active_trajectory not in (None, trajectory):
            raise CampaignContractError("timing ledger has a different active trajectory")
        if self.active_trajectory is None:
            self._append("TRAJECTORY_ORIGIN", {"trajectory_id": trajectory,
                "origin_unix_ns": int(self.clock.time_ns()),
                "origin_process_monotonic_ns": int(self.clock.monotonic_ns())})
            self.active_trajectory = trajectory
            self._append("TRAJECTORY_START", {"trajectory_id": trajectory})

    def finish_trajectory(self, trajectory: str) -> None:
        if trajectory in self.completed_trajectories and self.active_trajectory != trajectory:
            self.check(trajectory)
            return
        if self.active_trajectory != trajectory:
            raise CampaignContractError("timing ledger trajectory completion is out of order")
        self.check(trajectory)
        self._append("TRAJECTORY_COMPLETE", {"trajectory_id": trajectory,
            "elapsed_ns": self.trajectory_elapsed_ns.get(trajectory, 0)})
        self.completed_trajectories.add(trajectory)
        self.active_trajectory = None
        self._append("TRAJECTORY_STOP", {"trajectory_id": trajectory})

    def check(self, trajectory: str | None = None) -> dict[str, Any]:
        campaign, by_trajectory = self._current_elapsed()
        if campaign >= CAMPAIGN_WALL_LIMIT_SECONDS * 1_000_000_000:
            self.deadline_status = "CAMPAIGN_WALL_LIMIT_EXCEEDED"
            self._append("DEADLINE_EXCEEDED", {"status": "CAMPAIGN_WALL_LIMIT_EXCEEDED"})
            raise CampaignDeadlineExceeded("CAMPAIGN_WALL_LIMIT_EXCEEDED")
        if trajectory is not None and by_trajectory.get(trajectory, 0) >= TRAJECTORY_WALL_LIMIT_SECONDS * 1_000_000_000:
            self.deadline_status = "TRAJECTORY_WALL_LIMIT_EXCEEDED"
            self._append("DEADLINE_EXCEEDED", {"status": "TRAJECTORY_WALL_LIMIT_EXCEEDED"})
            raise CampaignDeadlineExceeded("TRAJECTORY_WALL_LIMIT_EXCEEDED")
        return {"campaign_elapsed_ns": campaign, "trajectory_elapsed_ns": by_trajectory,
            "campaign_remaining_ns": CAMPAIGN_WALL_LIMIT_SECONDS * 1_000_000_000 - campaign,
            "trajectory_remaining_ns": (TRAJECTORY_WALL_LIMIT_SECONDS * 1_000_000_000 -
                by_trajectory.get(trajectory, 0)) if trajectory else None}

    def timeout_seconds(self, trajectory: str | None = None) -> float:
        budget = self.check(trajectory)
        remaining = [budget["campaign_remaining_ns"]]
        if trajectory is not None:
            remaining.append(budget["trajectory_remaining_ns"])
        return max(0.001, min(remaining) / 1_000_000_000)

    def close(self) -> None:
        if self._closed:
            return
        self._append("PROCESS_STOP", {"process_stop_unix_ns": int(self.clock.time_ns())})
        self._unlock()
        self._lock_stream.close()
        self._closed = True

    def __del__(self) -> None:
        if getattr(self, "_closed", True):
            return
        try:
            self._unlock()
            self._lock_stream.close()
            self._closed = True
        except Exception:
            pass


def _spec_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for child in value.values():
            found = _spec_value(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _spec_value(child, key)
            if found is not None:
                return found
    return None


def _append_failure_history(execution_root: Path, record: Mapping[str, Any]) -> None:
    path = execution_root / "INCOMPLETE_FAILURE_HISTORY.jsonl"
    previous = "0" * 64
    sequence = 0
    if path.exists():
        for line in path.read_bytes().splitlines():
            row = json.loads(line)
            body = {key: value for key, value in row.items() if key != "record_sha256"}
            if (row.get("sequence") != sequence or row.get("previous_sha256") != previous or
                    row.get("record_sha256") != _sha(canonical(body))):
                raise CampaignContractError("incomplete failure history is tampered")
            previous = row["record_sha256"]
            sequence += 1
    body = {"schema": "FS4_JOINT_CAMPAIGN_FAILURE_HISTORY_V1", "sequence": sequence,
        "previous_sha256": previous, **dict(record)}
    row = {**body, "record_sha256": _sha(canonical(body))}
    _append_telemetry(path, row)


def _record_incomplete(path: Path, reason_code: str, *, campaign_id: str, source_commit: str,
                       classification: str) -> dict[str, Any]:
    execution_root = Path(path)
    marker = execution_root / "INCOMPLETE.json"
    if not marker.exists():
        _write_json_new(marker, {"schema": "FS4_JOINT_CAMPAIGN_INCOMPLETE_V2",
            "status": "INCOMPLETE", "campaign_id": campaign_id, "spec_sha256": spec_identity(),
            "source_freeze_commit": source_commit, "reason_code": reason_code,
            "classification": classification, "recorded_unix_ns": time.time_ns()})
    marker_digest = _sha(marker.read_bytes())
    _append_failure_history(execution_root, {"campaign_id": campaign_id,
        "spec_sha256": spec_identity(), "source_freeze_commit": source_commit,
        "previous_incomplete_record_sha256": marker_digest,
        "reason_code": reason_code, "classification": classification,
        "recorded_unix_ns": time.time_ns()})
    return _read_json(marker)


def _read_failure_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    previous, records = "0" * 64, []
    for sequence, line in enumerate(path.read_bytes().splitlines()):
        try:
            row = json.loads(line)
            body = {key: value for key, value in row.items() if key != "record_sha256"}
            if (row.get("schema") != "FS4_JOINT_CAMPAIGN_FAILURE_HISTORY_V1" or
                    row.get("sequence") != sequence or row.get("previous_sha256") != previous or
                    row.get("record_sha256") != _sha(canonical(body))):
                raise ValueError
            previous = row["record_sha256"]
            records.append(row)
        except Exception as exc:
            raise CampaignContractError("incomplete failure history is malformed or tampered") from exc
    return records


def _verified_timing_records(execution_root: Path, binding: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = execution_root / "TIMING_LEDGER.jsonl"
    parser = object.__new__(CampaignTimingLedger)
    parser.path = path
    records = parser._read_records()
    if any(row.get("campaign_id") != binding["campaign_id"] or
           row.get("spec_sha256") != spec_identity() or
           row.get("source_commit") != binding["source_freeze_commit"] for row in records):
        raise CampaignContractError("timing ledger campaign/source/spec binding differs")
    if records[-1].get("process_open"):
        raise CampaignContractError("timing ledger has an unclosed orchestration process")
    if any(row.get("kind") == "DEADLINE_EXCEEDED" for row in records):
        raise CampaignContractError("deadline-exhausted campaign cannot be verified or recovered")
    return records


def _validate_terminal_state(campaign_root: Path, binding: Mapping[str, Any],
                             projections: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    execution_root = Path(campaign_root) / "execution"
    seal_path = execution_root / "STAGE_A_CAMPAIGN_SEAL.json"
    if not seal_path.is_file():
        raise CampaignContractError("Stage A completion seal is missing")
    seal = _read_json(seal_path)
    expected_trajectory_seals = {key: {"run_id": projections[key]["run_id"],
        "manifest_sha256": projections[key]["manifest_sha256"],
        "stage_a_sha256": projections[key]["stage_a_sha256"]} for key in ARM_IDS}
    if (seal.get("schema") != "FS4_FINAL_JOINT_STAGE_A_CAMPAIGN_SEAL_V1" or
            seal.get("status") != "STAGE_A_COMPLETE" or
            seal.get("campaign_id") != binding["campaign_id"] or
            seal.get("scientific_spec_sha256") != spec_identity() or
            seal.get("campaign_registration_sha256") != binding["campaign_registration_sha256"] or
            seal.get("trajectory_seals") != expected_trajectory_seals):
        raise CampaignContractError("Stage A completion seal differs from the verified trajectories")
    marker_path = execution_root / "INCOMPLETE.json"
    receipt_path = execution_root / "RECOVERY_COMPLETION.json"
    history = _read_failure_history(execution_root / "INCOMPLETE_FAILURE_HISTORY.jsonl")
    timing_records = _verified_timing_records(execution_root, binding)
    if not marker_path.exists():
        if history or receipt_path.exists():
            raise CampaignContractError("failure evidence or recovery receipt exists without its incomplete marker")
        completion = next((row for row in timing_records if row.get("kind") == "CAMPAIGN_COMPLETE"), None)
        if (completion is None or seal.get("timing_completion_record_sha256") != completion.get("record_sha256") or
                seal.get("execution_time_totals_ns") != completion.get("payload", {}).get("elapsed_totals_ns")):
            raise CampaignContractError("timing ledger lacks a successful campaign completion record")
        return {"terminal_state": "UNINTERRUPTED_COMPLETE", "seal": seal}
    marker_bytes = marker_path.read_bytes()
    marker_digest = _sha(marker_bytes)
    marker = _read_json(marker_path)
    if (marker.get("schema") != "FS4_JOINT_CAMPAIGN_INCOMPLETE_V2" or
            marker.get("status") != "INCOMPLETE" or marker.get("campaign_id") != binding["campaign_id"] or
            marker.get("spec_sha256") != spec_identity() or
            marker.get("source_freeze_commit") != binding["source_freeze_commit"] or not history):
        raise CampaignContractError("incomplete marker identity or failure history is invalid")
    if marker.get("classification") != "RECOVERABLE_EXECUTION_INTERRUPTION" or any(
            row.get("classification") != "RECOVERABLE_EXECUTION_INTERRUPTION" or
            row.get("previous_incomplete_record_sha256") != marker_digest or
            row.get("campaign_id") != binding["campaign_id"] or
            row.get("spec_sha256") != spec_identity() or
            row.get("source_freeze_commit") != binding["source_freeze_commit"] for row in history):
        raise CampaignContractError("fatal, ambiguous or deadline failure is not eligible for recovery")
    if not receipt_path.is_file():
        raise CampaignContractError("recovered Stage A has no recovery-completion receipt")
    receipt = _read_json(receipt_path)
    ledger_identities = {key: {"run_id": projections[key]["run_id"],
        "manifest_sha256": projections[key]["manifest_sha256"],
        "stage_a_sha256": projections[key]["stage_a_sha256"],
        "category_seals": projections[key]["category_seals"]} for key in ARM_IDS}
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    expected_receipt_sha = _sha(canonical(receipt_body) + b"\n")
    required_kinds = {"RECOVERY_CHECKPOINT_SELECTED", "RECOVERY_REPLAY_COMPLETED",
                      "RECOVERY_LEDGER_VERIFIED", "CAMPAIGN_COMPLETE"}
    if (receipt.get("schema") != "FS4_JOINT_CAMPAIGN_RECOVERY_COMPLETION_V1" or
            receipt.get("status") != "VERIFIED_RECOVERY_COMPLETION" or
            receipt.get("campaign_id") != binding["campaign_id"] or
            receipt.get("spec_sha256") != spec_identity() or
            receipt.get("source_freeze_commit") != binding["source_freeze_commit"] or
            receipt.get("previous_incomplete_record_sha256") != marker_digest or
            receipt.get("verified_complete_stage_a_ledger_identities") != ledger_identities or
            receipt.get("final_trajectory_stage_a_seals") != seal["trajectory_seals"] or
            receipt.get("recovery_classification") != "VERIFIED_RECOVERY_COMPLETION" or
            not isinstance(receipt.get("recovery_checkpoint_identities"), list) or
            not receipt["recovery_checkpoint_identities"] or
            not isinstance(receipt.get("replayed_ledger_prefix_identities"), list) or
            not receipt["replayed_ledger_prefix_identities"] or
            receipt.get("receipt_sha256") != expected_receipt_sha):
        raise CampaignContractError("recovery-completion receipt is forged, incomplete or bound to different evidence")
    timing_kinds = {row.get("kind") for row in timing_records}
    logged_checkpoints = [row.get("payload") for row in timing_records
        if row.get("kind") == "RECOVERY_CHECKPOINT_SELECTED"]
    logged_prefixes = [row.get("payload") for row in timing_records
        if row.get("kind") == "RECOVERY_REPLAY_COMPLETED"]
    verified_record = next((row for row in timing_records
        if row.get("kind") == "RECOVERY_LEDGER_VERIFIED"), None)
    completion_record = next((row for row in timing_records
        if row.get("kind") == "CAMPAIGN_COMPLETE"), None)
    if (completion_record is None or seal.get("timing_completion_record_sha256") != completion_record.get("record_sha256") or
            seal.get("execution_time_totals_ns") != completion_record.get("payload", {}).get("elapsed_totals_ns")):
        raise CampaignContractError("recovery seal does not bind the final timing totals")
    if (not required_kinds <= timing_kinds or completion_record is None or
            receipt.get("timing_ledger_prefix_sha256") != completion_record.get("record_sha256") or
            receipt.get("recovery_checkpoint_identities") != logged_checkpoints or
            receipt.get("replayed_ledger_prefix_identities") != logged_prefixes or
            verified_record is None or verified_record.get("payload", {}).get(
                "verified_complete_stage_a_ledger_identities") != ledger_identities):
        raise CampaignContractError("recovery timing audit is incomplete or not bound to the timing ledger")
    return {"terminal_state": "VERIFIED_RECOVERY_COMPLETION", "seal": seal, "receipt": receipt}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_new(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_new(path: Path, value: Any) -> str:
    data = canonical(value) + b"\n"
    _write_new(path, data)
    return _sha(data)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_bytes())
    except Exception as exc:
        raise ValueError(f"required campaign JSON is unreadable: {Path(path).name}") from exc


def _validate_authorization(path: Path, *, scope: str, campaign_id: str,
                            bindings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    auth = _read_json(path)
    binding = dict(bindings or {})
    required = {"schema", "authorization_id", "authorized_by", "authorized_at", "scope",
                "campaign_id", "scientific_spec_sha256", "authorization_statement", *binding}
    if (not isinstance(auth, dict) or set(auth) != required or auth.get("schema") != AUTH_SCHEMA or
            auth.get("scope") != scope or scope not in SCOPES or auth.get("campaign_id") != campaign_id or
            auth.get("scientific_spec_sha256") != spec_identity() or
            any(auth.get(key) != value for key, value in binding.items()) or
            any(not isinstance(auth.get(key), str) or not auth[key].strip()
                for key in ("authorization_id", "authorized_by", "authorized_at", "authorization_statement"))):
        raise CampaignContractError(f"separate {scope} authorization is missing, malformed or bound to different inputs")
    return auth


def _load_campaign(campaign_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign_root = Path(campaign_root).absolute()
    registration = _read_json(campaign_root / "registration.json")
    metadata = _read_json(campaign_root / "campaign.json")
    if (metadata.get("schema") != CAMPAIGN_SCHEMA or metadata.get("campaign_id") is None or
            metadata.get("scientific_spec_sha256") != spec_identity() or
            metadata.get("campaign_registration_sha256") != digest(registration)):
        raise CampaignContractError("campaign registration receipt or frozen specification digest differs")
    validate_registration(registration)
    return metadata, registration


def _check_committed_sources(source_commit: str) -> dict[str, str]:
    from .fs4_joint_campaign_contract import source_fingerprints
    result = source_fingerprints(ROOT)
    for relative, expected in result.items():
        committed = subprocess.run(["git", "show", f"{source_commit}:{relative}"], cwd=ROOT,
                                   capture_output=True)
        if committed.returncode or _sha(committed.stdout.replace(b"\r\n", b"\n")) != _sha((ROOT / relative).read_bytes().replace(b"\r\n", b"\n")):
            raise CampaignContractError(f"campaign source is not frozen at the registered commit: {relative}")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, "HEAD"], cwd=ROOT,
                              capture_output=True)
    if ancestor.returncode:
        raise CampaignContractError("campaign source-freeze commit is not an ancestor of HEAD")
    return result


def freeze_only() -> dict[str, Any]:
    spec = load_spec()
    return {"schema": "FS4_FINAL_JOINT_CAMPAIGN_FREEZE_CHECK_V1", "status": "FROZEN_SPEC_VALID",
            "scientific_execution_authorized": False, "scientific_spec_sha256": spec_identity(spec),
            "events": sum(EXPECTED_EVENTS.values()), "source_groups": sum(n // 5 for n in EXPECTED_EVENTS.values()),
            "trajectories": list(ARM_IDS)}


def validate_contract() -> dict[str, Any]:
    spec = load_spec()
    arms = _read_json(ROOT / "docs/calibration/fs4_joint_campaign/ARM_REGISTRY.json")
    if arms.get("schema") != "FS4_FINAL_JOINT_CAMPAIGN_ARM_REGISTRY_V1" or len(arms.get("arms", [])) != 3:
        raise CampaignContractError("human-readable arm registry is incomplete")
    actual = [(row.get("trajectory_id"), row.get("profile"), row.get("controller")) for row in arms["arms"]]
    expected = [(row["trajectory_id"], row["profile"], row["controller"]) for row in spec["arms"]]
    if actual != expected or arms.get("shared_registration", {}).get("registration_count") != 1:
        raise CampaignContractError("human and machine-readable trajectory registrations differ")
    return {"status": "PASS", "scientific_spec_sha256": spec_identity(spec),
            "category_events": EXPECTED_EVENTS, "total_events": sum(EXPECTED_EVENTS.values()),
            "arm_registry": actual, "execution_enabled": False}


def _r7_reordered_registration(original: Mapping[str, Any], original_ledger: Mapping[str, Any],
                               labels_by_original_position: Mapping[int, int]) -> tuple[dict[str, Any], dict[str, Any], dict[int, int], str, str]:
    from experiments.risk_safe.preflight import parse_image_name
    from .r7_shuffled_correction_4class import deterministic_group_shuffle

    if (original.get("schema") != "FS4_TRUSTED_EXECUTION_V1" or
            original.get("category_order") != list(EXPECTED_CATEGORIES) or
            original.get("query_seed") != 37 or original.get("model_state_sha256") != load_spec()["source_model_support"]["model"]["model_state_sha256"]):
        raise CampaignContractError("input trusted registration differs from the frozen four-category R4 population")
    if original_ledger.get("expected_events_by_category") != EXPECTED_EVENTS:
        raise CampaignContractError("input source manifest population differs from the frozen event counts")
    if set(labels_by_original_position) != set(range(sum(EXPECTED_EVENTS.values()))):
        raise CampaignContractError("private staging labels do not cover the exact original registration positions")
    categories = copy.deepcopy(original["categories"])
    result_events: dict[str, list[dict[str, Any]]] = {}
    order_receipt = []
    labels_by_new_position: dict[int, int] = {}
    global_position = 0
    for category in EXPECTED_CATEGORIES:
        rows = categories[category]["events"]
        if len(rows) != EXPECTED_EVENTS[category]:
            raise CampaignContractError(f"{category}: original registration event count differs")
        prepared = []
        view_by_group: dict[str, list[int]] = {}
        row_by_position = {}
        for row in rows:
            absolute = Path(original["dataset_root"]) / row["relative_path"]
            parsed = parse_image_name(absolute, expected_class=category)
            if row["group"] != hashlib.sha256(f"official-sample-group\0{parsed.official_sample_key}".encode()).hexdigest():
                raise CampaignContractError(f"{category}: original group identity differs from the image name")
            view_by_group.setdefault(row["group"], []).append(parsed.camera_id)
            row_by_position[row["position"]] = row
            prepared.append({"opaque_group": row["group"], "position": row["position"],
                "relative_id": row["relative_path"], "camera_id": str(parsed.camera_id),
                "timestamp": parsed.timestamp, "image_path": str(absolute)})
        if any(sorted(views) != [1, 2, 3, 4, 5] or views != sorted(views) for views in view_by_group.values()):
            raise CampaignContractError(f"{category}: a registered five-view Source Group is incomplete or reordered")
        shuffled = deterministic_group_shuffle(prepared, seed=0)
        if len(shuffled) != len(rows) or {event.original_position for event in shuffled} != set(row_by_position):
            raise CampaignContractError(f"{category}: R7 group shuffle is not a bijection")
        ordered_rows = []
        for event in shuffled:
            row = copy.deepcopy(row_by_position[event.original_position])
            row["position"] = global_position
            ordered_rows.append(row)
            labels_by_new_position[global_position] = labels_by_original_position[event.original_position]
            order_receipt.append({"category": category, "position": global_position,
                "original_position": event.original_position, "group_sha256": event.source_group_sha256,
                "view": int(event.camera_id), "image_identity_sha256": event.image_identity_sha256})
            global_position += 1
        result_events[category] = ordered_rows
    reordered = copy.deepcopy(dict(original))
    reordered_categories = {}
    for category in EXPECTED_CATEGORIES:
        reordered_categories[category] = {"supports": copy.deepcopy(categories[category]["supports"]),
                                          "events": result_events[category]}
    reordered["categories"] = reordered_categories
    reordered["dataset_identity_sha256"] = digest({"dataset_root": reordered["dataset_root"],
        "category_order": reordered["category_order"], "categories": reordered_categories})
    ledger = copy.deepcopy(dict(original_ledger))
    ledger["dataset_identity_sha256"] = reordered["dataset_identity_sha256"]
    ledger["stream_sha256_by_category"] = {category: digest([
        {"group": row["group"], "event": row["event"]} for row in result_events[category]])
        for category in EXPECTED_CATEGORIES}
    return reordered, ledger, labels_by_new_position, digest(order_receipt), digest({"registration": original, "ledger": original_ledger})


def _validate_staging_labels_from_official_json(original: Mapping[str, Any],
                                                original_ledger: Mapping[str, Any],
                                                labels_by_position: Mapping[int, int]) -> dict[str, Any]:
    """Bind private staging labels to the registered image-level JSON labels.

    Capture-session tokens identify a source acquisition. They are not an
    image-level label: a normal camera view can occur in an NG sample. The
    frozen Stage 05 registration and each registered JSON hash anchor the
    authoritative ``anomaly_class == meta.normal_class`` labels.
    """
    from pathlib import PurePosixPath
    from experiments.risk_safe.preflight import parse_image_name

    identity = original_ledger.get("frozen_source_identity")
    if not isinstance(identity, Mapping):
        raise CampaignContractError("registered Stage 05 JSON source identity is missing")
    try:
        frozen_bytes = STAGE05_FROZEN_INPUTS.read_bytes()
        frozen = json.loads(frozen_bytes)
    except Exception as exc:
        raise CampaignContractError("registered Stage 05 source input is unavailable") from exc
    frozen_sha = _sha(frozen_bytes)
    if (frozen.get("schema") != "stage05.frozen_inputs.v1" or
            frozen.get("freeze_verification_status") != "PASS" or
            frozen.get("stage1_or_stage2_run") is not False or
            identity.get("stage05_frozen_inputs_sha256") != frozen_sha):
        raise CampaignContractError("Stage 05 frozen source identity differs")

    categories = original.get("category_order")
    if categories != list(EXPECTED_CATEGORIES) or set(frozen.get("classes", {})) != set(EXPECTED_CATEGORIES):
        raise CampaignContractError("registered source category set differs from the frozen population")
    json_hashes = identity.get("real_iad_json_sha256_by_category")
    if not isinstance(json_hashes, Mapping) or set(json_hashes) != set(EXPECTED_CATEGORIES):
        raise CampaignContractError("registered official JSON hashes are incomplete")
    positions = [row.get("position") for category in categories
                 for row in original["categories"][category]["events"]]
    if (set(labels_by_position) != set(positions) or
            any(type(value) is not int or value not in (0, 1) for value in labels_by_position.values())):
        raise CampaignContractError("private labels do not cover the exact registered positions")

    dataset_root = Path(original["dataset_root"]).resolve(strict=True)
    json_root = dataset_root.parent / "jsons" / "realiad_jsons"
    verified_json_hashes: dict[str, str] = {}
    global_position = 0
    verified_count = 0
    for category in categories:
        source_path = json_root / f"{category}.json"
        try:
            source_bytes = source_path.read_bytes()
            source = json.loads(source_bytes)
        except Exception as exc:
            raise CampaignContractError(f"{category}: registered official JSON is unavailable") from exc
        source_sha = _sha(source_bytes)
        frozen_class = frozen["classes"][category]
        if (source_sha != json_hashes.get(category) or
                source_sha != frozen_class.get("class_json_sha256")):
            raise CampaignContractError(f"{category}: official JSON hash differs from its frozen registration")
        metadata = source.get("meta")
        if (not isinstance(metadata, Mapping) or not isinstance(metadata.get("normal_class"), str) or
                not isinstance(metadata.get("prefix"), str) or
                metadata["prefix"].replace("\\", "/") != f"{category}/"):
            raise CampaignContractError(f"{category}: official JSON label metadata or prefix is invalid")
        rows = source.get("test")
        events = original["categories"][category]["events"]
        stream = frozen_class.get("stream")
        if (not isinstance(rows, list) or not isinstance(stream, list) or
                len(rows) != EXPECTED_EVENTS[category] or len(events) != len(rows) or len(stream) != len(rows)):
            raise CampaignContractError(f"{category}: official JSON, frozen stream and registration counts differ")

        group_views: dict[str, list[int]] = {}
        seen_paths: set[str] = set()
        for local_position, (record, event, frozen_row) in enumerate(zip(rows, events, stream, strict=True)):
            position = global_position + local_position
            if (event.get("position") != position or frozen_row.get("position") != local_position or
                    "anomaly_class" not in record or not isinstance(record["anomaly_class"], str) or
                    not isinstance(record.get("image_path"), str) or
                    ("category" in record and record["category"] != category)):
                raise CampaignContractError(f"{category}: official annotation or source position is invalid")
            relative = PurePosixPath(metadata["prefix"]) / PurePosixPath(record["image_path"].replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or str(relative) != event.get("relative_path", "").replace("\\", "/"):
                raise CampaignContractError(f"{category}: registered path differs from the official JSON order")
            image_path = (dataset_root.joinpath(*relative.parts)).resolve(strict=True)
            if dataset_root not in image_path.parents:
                raise CampaignContractError(f"{category}: official JSON image path escaped the dataset root")
            parsed = parse_image_name(image_path, expected_class=category)
            group_sha = hashlib.sha256(
                f"official-sample-group\0{parsed.official_sample_key}".encode("utf-8")).hexdigest()
            if (event.get("group") != group_sha or
                    frozen_row.get("sample_key") != parsed.official_sample_key or
                    frozen_row.get("opaque_group") != group_sha or
                    int(frozen_row.get("camera_id", -1)) != parsed.camera_id or
                    str(frozen_row.get("timestamp")) != parsed.timestamp or
                    str(frozen_row.get("relative_id", "")).replace("\\", "/") != parsed.rel_path or
                    Path(str(frozen_row.get("image_path", ""))).resolve() != image_path):
                raise CampaignContractError(f"{category}: frozen source image/group/view identity differs")
            if str(image_path) in seen_paths:
                raise CampaignContractError(f"{category}: duplicate official JSON image path")
            seen_paths.add(str(image_path))
            group_views.setdefault(parsed.official_sample_key, []).append(parsed.camera_id)

            official_label = int(record["anomaly_class"] != metadata["normal_class"])
            if parsed.capture_session == "OK" and official_label != 0:
                raise CampaignContractError(f"{category}: official JSON labels an OK-session image as defective")
            if labels_by_position[position] != official_label:
                raise CampaignContractError(f"{category}: private input label differs from official JSON at position {position}")
            verified_count += 1

        if any(views != [1, 2, 3, 4, 5] for views in group_views.values()):
            raise CampaignContractError(f"{category}: official Source Groups are not ordered complete five-view groups")
        if len(group_views) != EXPECTED_EVENTS[category] // 5:
            raise CampaignContractError(f"{category}: official Source Group count differs")
        global_position += len(rows)
        verified_json_hashes[category] = source_sha

    return {"schema": "FS4_OFFICIAL_JSON_LABEL_VERIFICATION_V1",
            "stage05_frozen_inputs_sha256": frozen_sha,
            "official_json_sha256_by_category": verified_json_hashes,
            "verified_private_label_count": verified_count,
            "label_semantics": "anomaly_class equals meta.normal_class per image"}


def prepare_real_data(*, campaign_id: str, campaign_root: Path, private_root: Path, image_root: Path,
                      original_registration_path: Path, original_ledger_path: Path,
                      labels_path: Path, authorization_path: Path) -> dict[str, Any]:
    _validate_authorization(authorization_path, scope="STAGING", campaign_id=campaign_id)
    campaign_root, private_root, image_root = (Path(p).absolute() for p in
        (campaign_root, private_root, image_root))
    spec = load_spec()
    for left, right in ((campaign_root, private_root), (campaign_root, image_root), (private_root, image_root)):
        if left == right or left.is_relative_to(right) or right.is_relative_to(left):
            raise CampaignContractError("campaign, mapping and image stores must be non-nested separate roots")
    if campaign_root.exists() or private_root.exists() or image_root.exists():
        raise FileExistsError("real-data campaign and blinded stores must use fresh non-overwriting roots")
    original = _read_json(original_registration_path)
    original_ledger = _read_json(original_ledger_path)
    current_source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
        check=True, capture_output=True, text=True).stdout.strip()
    if (original.get("source_commit") != current_source_commit or
            original_ledger.get("source_commit") != current_source_commit):
        raise CampaignContractError("original registration and ledger must use the corrected source commit")
    labels_document = _read_json(labels_path)
    if set(labels_document) != {"schema", "labels_by_position"} or labels_document["schema"] != "FS4_CAMPAIGN_PRIVATE_LABEL_INPUT_V1":
        raise CampaignContractError("staging labels input has an invalid explicit schema")
    raw_labels = labels_document["labels_by_position"]
    if (not isinstance(raw_labels, dict) or
            any(not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key for key in raw_labels)):
        raise CampaignContractError("staging label positions are malformed")
    labels_by_original = {int(key): value for key, value in raw_labels.items()}
    label_source_verification = _validate_staging_labels_from_official_json(
        original, original_ledger, labels_by_original)
    reordered, reordered_ledger, labels_by_new, order_sha, source_identity = _r7_reordered_registration(
        original, original_ledger, labels_by_original)
    from .fs4_r4_preparation import prepare_blinded_registration
    blinded, preparation_receipt = prepare_blinded_registration(original=reordered,
        original_ledger=reordered_ledger, labels_by_position=labels_by_new,
        private_root=private_root, image_root=image_root)
    validate_registration(blinded, spec=spec)
    campaign_root.mkdir(parents=False, exist_ok=False)
    campaign_meta = {"schema": CAMPAIGN_SCHEMA, "campaign_id": campaign_id,
        "scientific_spec_sha256": spec_identity(spec), "source_freeze_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True,
            text=True).stdout.strip(), "campaign_registration_sha256": digest(blinded),
        "source_mapping_sha256": blinded["source_mapping_sha256"], "image_root": str(image_root),
        "private_root": str(private_root), "dataset_root": blinded["dataset_root"],
        "r7_seed0_order_sha256": order_sha, "source_manifest_identity_sha256": source_identity,
        "preparation_receipt": preparation_receipt, "trajectory_manifests": {},
        "private_label_source_verification": label_source_verification,
        "status": "STAGING_COMPLETE_SUPPORT_PREFLIGHT_REQUIRED"}
    _write_json_new(campaign_root / "registration.json", blinded)
    _write_json_new(campaign_root / "campaign.json", campaign_meta)
    return campaign_meta


def verify_support_preflight(*, campaign_root: Path, receipt_path: Path,
                             authorization_path: Path) -> dict[str, Any]:
    campaign_root = Path(campaign_root).absolute()
    metadata, registration = _load_campaign(campaign_root)
    _validate_authorization(authorization_path, scope="SUPPORT_PREFLIGHT", campaign_id=metadata["campaign_id"],
        bindings={"campaign_registration_sha256": metadata["campaign_registration_sha256"]})
    receipt_bytes = Path(receipt_path).read_bytes()
    receipt = validate_support_preflight(json.loads(receipt_bytes), campaign_id=metadata["campaign_id"])
    from .fs4_r2_security import WindowsPrivateStore
    support_root = campaign_root / "support-preflight"
    support_store = WindowsPrivateStore(support_root, create=True)
    support_store.write("receipt.json", receipt_bytes)
    pinned_path = support_root / "receipt.json"
    from .fs4_joint_campaign_contract import build_trajectory_manifests
    source_commit = metadata["source_freeze_commit"]
    _check_committed_sources(source_commit)
    manifests = build_trajectory_manifests(registration=registration, campaign_id=metadata["campaign_id"],
        campaign_registration_sha256=metadata["campaign_registration_sha256"],
        support_preflight_receipt=receipt, support_preflight_receipt_path=pinned_path,
        r7_seed0_order_sha256=metadata["r7_seed0_order_sha256"],
        source_manifest_identity_sha256=metadata["source_manifest_identity_sha256"],
        source_commit=source_commit)
    manifest_dir = campaign_root / "manifests"
    manifest_dir.mkdir(parents=False, exist_ok=False)
    manifest_hashes = {}
    for trajectory in ARM_IDS:
        manifest_hashes[trajectory] = _write_json_new(manifest_dir / f"{trajectory}.json", manifests[trajectory])
    metadata.update({"support_preflight_receipt_sha256": _sha(receipt_bytes),
        "support_preflight_receipt_path": str(pinned_path), "trajectory_manifests": manifest_hashes,
        "status": "SUPPORT_PREFLIGHT_VERIFIED_RESOURCE_PREFLIGHT_REQUIRED"})
    # campaign.json is an append-only receipt; the update is captured separately.
    _write_json_new(campaign_root / "support-preflight-binding.json", metadata)
    return {"status": "PASS", "campaign_id": metadata["campaign_id"],
            "campaign_registration_sha256": metadata["campaign_registration_sha256"],
            "support_preflight_receipt_sha256": _sha(receipt_bytes), "trajectory_manifest_sha256": manifest_hashes}


def _campaign_with_support(campaign_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata, registration = _load_campaign(campaign_root)
    binding_path = Path(campaign_root) / "support-preflight-binding.json"
    if not binding_path.is_file():
        raise CampaignContractError("campaign-bound four-Support preflight is not verified")
    binding = _read_json(binding_path)
    if binding.get("campaign_registration_sha256") != metadata["campaign_registration_sha256"]:
        raise CampaignContractError("Support preflight binding belongs to another registration")
    return binding, registration


def resource_preflight(*, campaign_root: Path, benchmark_receipt_path: Path | None,
                       authorization_path: Path) -> dict[str, Any]:
    campaign_root = Path(campaign_root).absolute()
    metadata, registration = _campaign_with_support(campaign_root)
    _validate_authorization(authorization_path, scope="RESOURCE_PREFLIGHT", campaign_id=metadata["campaign_id"],
        bindings={"campaign_registration_sha256": metadata["campaign_registration_sha256"],
                  "support_preflight_receipt_sha256": metadata["support_preflight_receipt_sha256"]})
    benchmark = _read_json(benchmark_receipt_path) if benchmark_receipt_path else None
    from .fs4_joint_campaign_resources import resource_preflight as check_resources
    report = check_resources(registration=registration, image_root=Path(metadata["image_root"]),
        output_root=campaign_root / "execution", private_root=Path(metadata["private_root"]),
        benchmark_receipt=benchmark)
    report["campaign_id"] = metadata["campaign_id"]
    report["support_preflight_receipt_sha256"] = metadata["support_preflight_receipt_sha256"]
    path = campaign_root / "resource-preflight.json"
    _write_json_new(path, report)
    return report


def _load_trajectory_manifest(campaign_root: Path, trajectory: str) -> dict[str, Any]:
    binding, _registration = _campaign_with_support(campaign_root)
    path = Path(campaign_root) / "manifests" / f"{trajectory}.json"
    manifest = _read_json(path)
    if _sha(canonical(manifest) + b"\n") != binding["trajectory_manifests"].get(trajectory):
        raise CampaignContractError(f"{trajectory}: bound trajectory manifest digest differs")
    return manifest


def _inspect_ledger(run_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    if not run_root.exists():
        return None
    from .fs4_research_ledger import ResearchLedger
    ledger = ResearchLedger(run_root, manifest, create=False)
    try:
        prefix = {"run_id": manifest.get("run_id"), "manifest_sha256": digest(manifest),
            "position": ledger.position, "phase": ledger.phase, "last_hash": ledger.last_hash}
        return {"position": ledger.position, "phase": ledger.phase, "current_category": ledger.current_category,
                "category_seals": dict(ledger.category_seals), "last_hash": ledger.last_hash,
                "prefix_identity": prefix}
    finally:
        ledger.close()


def _checkpoint_candidates(run_root: Path, category: str, manifest: Mapping[str, Any],
                           category_start: int, category_end: int) -> list[tuple[int, str]]:
    checkpoint_root = run_root / "checkpoints"
    if not checkpoint_root.exists():
        return []
    from .fs4_r2_security import WindowsPrivateStore, validate_private_acl
    store = WindowsPrivateStore(checkpoint_root, create=False)
    candidates = []
    for child in checkpoint_root.iterdir():
        validate_private_acl(child, directory=True)
        if not child.is_dir():
            raise CampaignContractError("checkpoint store contains an unexpected object")
        for filename in ("state.json", "tensors.npz", "COMMITTED.json"):
            validate_private_acl(child / filename, directory=False)
        marker = _read_json(child / "COMMITTED.json")
        state_bytes = (child / "state.json").read_bytes()
        tensor_bytes = (child / "tensors.npz").read_bytes()
        state = json.loads(state_bytes)
        if (marker != {"state_sha256": _sha(state_bytes), "tensor_sha256": _sha(tensor_bytes)} or
                state.get("manifest_sha256") != digest(manifest) or
                state.get("tensor_sha256") != marker["tensor_sha256"]):
            raise CampaignContractError("checkpoint inventory has a corrupt or wrong-manifest entry")
        next_position = state.get("ledger_offset", {}).get("next_position")
        if not isinstance(next_position, int) or child.name.startswith(category + "-") is False:
            continue
        if category_start <= next_position <= category_end:
            candidates.append((next_position, child.name))
    return sorted(candidates)


def _append_telemetry(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(dict(row)) + b"\n"
    if not path.exists():
        _write_new(path, data)
        return
    with path.open("ab", buffering=0) as stream:
        if stream.write(data) != len(data):
            raise OSError("short runtime telemetry append")
        os.fsync(stream.fileno())


def _start_worker(*, campaign_root: Path, metadata: Mapping[str, Any], trajectory: str,
                  manifest_path: Path, manifest: Mapping[str, Any], category: str,
                  checkpoint: str | None, create: bool, timeout_seconds: float | None = None):
    from .fs4_r4_worker import BrokerProcess, BlindedResearchWorker
    run_root = campaign_root / "execution" / "runs" / trajectory
    disclosure_root = campaign_root / "execution" / "disclosures" / trajectory
    run_root.parent.mkdir(parents=True, exist_ok=True)
    disclosure_root.parent.mkdir(parents=True, exist_ok=True)
    broker = BrokerProcess(private_root=Path(metadata["private_root"]), run_root=run_root,
        disclosures_root=disclosure_root, manifest_path=manifest_path, timeout_seconds=timeout_seconds)
    config = {"manifest_path": str(manifest_path.absolute()), "run_root": str(run_root.absolute()),
        "category": category, "create": create, "checkpoint": checkpoint}
    config_dir = campaign_root / "execution" / "runtime-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{trajectory}-{category}-{uuid.uuid4().hex}.json"
    _write_json_new(config_path, config)
    try:
        worker = BlindedResearchWorker(config_path, broker, timeout_seconds=timeout_seconds)
    except Exception:
        try:
            config_path.unlink()
        except OSError:
            pass
        raise
    return worker, config_path, run_root, disclosure_root


def _checkpoint_identity(run_root: Path, trajectory: str, category: str, name: str) -> dict[str, Any]:
    path = Path(run_root) / "checkpoints" / name
    try:
        committed = (path / "COMMITTED.json").read_bytes()
        state = (path / "state.json").read_bytes()
        tensors = (path / "tensors.npz").read_bytes()
        marker = json.loads(committed)
        if marker != {"state_sha256": _sha(state), "tensor_sha256": _sha(tensors)}:
            raise ValueError
    except Exception as exc:
        raise CampaignContractError("recovery checkpoint identity is missing or corrupt") from exc
    return {"trajectory_id": trajectory, "category": category, "checkpoint_name": name,
        "committed_sha256": _sha(committed), "state_sha256": _sha(state),
        "tensor_sha256": _sha(tensors)}


def _recovery_eligibility(execution_root: Path, binding: Mapping[str, Any], *,
                          unclean_previous_process: bool) -> bool:
    marker_path = execution_root / "INCOMPLETE.json"
    if marker_path.exists():
        marker_digest = _sha(marker_path.read_bytes())
        marker = _read_json(marker_path)
        history = _read_failure_history(execution_root / "INCOMPLETE_FAILURE_HISTORY.jsonl")
        if (marker.get("schema") != "FS4_JOINT_CAMPAIGN_INCOMPLETE_V2" or
                marker.get("campaign_id") != binding["campaign_id"] or
                marker.get("spec_sha256") != spec_identity() or
                marker.get("source_freeze_commit") != binding["source_freeze_commit"] or
                marker.get("classification") != "RECOVERABLE_EXECUTION_INTERRUPTION" or not history or
                any(row.get("classification") != "RECOVERABLE_EXECUTION_INTERRUPTION" or
                    row.get("previous_incomplete_record_sha256") != marker_digest or
                    row.get("campaign_id") != binding["campaign_id"] or
                    row.get("spec_sha256") != spec_identity() or
                    row.get("source_freeze_commit") != binding["source_freeze_commit"] for row in history)):
            raise CampaignContractError("existing incomplete campaign is fatal, ambiguous or deadline-exhausted")
    elif not unclean_previous_process:
        raise CampaignContractError("resume has no recoverable incomplete marker or unclean prior process")

    ambiguous = False
    for trajectory in ARM_IDS:
        manifest_path = Path(execution_root).parent / "manifests" / f"{trajectory}.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        ledger_state = _inspect_ledger(execution_root / "runs" / trajectory, manifest)
        if ledger_state and ledger_state.get("phase") == "LABEL_INTENT":
            ambiguous = True
    if ambiguous:
        _record_incomplete(execution_root, "LABEL_DISCLOSURE_INTENT_UNRESOLVED",
            campaign_id=binding["campaign_id"], source_commit=binding["source_freeze_commit"],
            classification="AMBIGUOUS_LABEL_DISCLOSURE")
        raise CampaignContractError("ambiguous label intent makes this campaign permanently ineligible for resume")
    if marker_path.exists():
        return True
    _record_incomplete(execution_root, "UNCLEAN_ORCHESTRATION_PROCESS_INTERRUPTION",
        campaign_id=binding["campaign_id"], source_commit=binding["source_freeze_commit"],
        classification="RECOVERABLE_EXECUTION_INTERRUPTION")
    return True


def run_stage_a(*, campaign_root: Path, authorization_path: Path, resume: bool = False) -> dict[str, Any]:
    campaign_root = Path(campaign_root).absolute()
    binding, registration = _campaign_with_support(campaign_root)
    resource_path = campaign_root / "resource-preflight.json"
    if not resource_path.is_file():
        raise CampaignContractError("resource preflight receipt is required before Stage A")
    resource = _read_json(resource_path)
    if resource.get("status") != "PASS" or resource.get("spec_sha256") != spec_identity() or resource.get(
            "registration_sha256") != digest(registration):
        raise CampaignContractError("BLOCKED_RESOURCE_PREFLIGHT: immutable resource receipt does not pass")
    manifest_hashes = binding["trajectory_manifests"]
    _validate_authorization(authorization_path, scope="STAGE_A", campaign_id=binding["campaign_id"],
        bindings={"campaign_registration_sha256": binding["campaign_registration_sha256"],
                  "support_preflight_receipt_sha256": binding["support_preflight_receipt_sha256"],
                  "resource_preflight_receipt_sha256": _sha(resource_path.read_bytes()),
                  "trajectory_manifests": manifest_hashes})
    execution_root = campaign_root / "execution"
    if not execution_root.exists():
        execution_root.mkdir(parents=False, exist_ok=False)
    elif not resume:
        raise FileExistsError("execution namespace exists; pass explicit resume authorization to continue")
    timing: CampaignTimingLedger | None = None
    recovery = False
    recovery_checkpoints: list[dict[str, Any]] = []
    recovery_prefixes: list[dict[str, Any]] = []
    worker = None
    config_path: Path | None = None
    try:
        existing_seal = execution_root / "STAGE_A_CAMPAIGN_SEAL.json"
        if existing_seal.exists():
            marker = execution_root / "INCOMPLETE.json"
            receipt = execution_root / "RECOVERY_COMPLETION.json"
            if not marker.exists() or receipt.exists():
                return verify_stage_a(campaign_root=campaign_root)
            if not resume:
                raise FileExistsError("incomplete recovery seal requires explicit resume authorization")
        timing = CampaignTimingLedger(execution_root / "TIMING_LEDGER.jsonl",
            campaign_id=binding["campaign_id"], source_commit=binding["source_freeze_commit"])
        if resume:
            recovery = _recovery_eligibility(execution_root, binding,
                unclean_previous_process=timing.unclean_previous_process)
        elif timing.records and len(timing.records) > 2:
            raise CampaignContractError("existing timing history requires explicit resume")
        for trajectory in ARM_IDS:
            timing.start_trajectory(trajectory)
            timing.check(trajectory)
            manifest_path = campaign_root / "manifests" / f"{trajectory}.json"
            manifest = _load_trajectory_manifest(campaign_root, trajectory)
            if manifest["source_commit"] != binding["source_freeze_commit"]:
                raise CampaignContractError("trajectory source freeze commit differs from campaign staging")
            run_root = execution_root / "runs" / trajectory
            existing = _inspect_ledger(run_root, manifest)
            if existing and existing["phase"] == "GLOBAL_A":
                if recovery:
                    for category in EXPECTED_CATEGORIES:
                        events = registration["categories"][category]["events"]
                        candidates = _checkpoint_candidates(run_root, category, manifest,
                            events[0]["position"], events[-1]["position"] + 1)
                        if not candidates:
                            raise CampaignContractError("sealed trajectory has no verified recovery checkpoint")
                        checkpoint_name = candidates[-1][1]
                        identity = _checkpoint_identity(run_root, trajectory, category, checkpoint_name)
                        recovery_checkpoints.append(identity)
                        timing.record("RECOVERY_CHECKPOINT_SELECTED", identity)
                    prefix = {"trajectory_id": trajectory, "category": "GLOBAL_A",
                        "pre": existing.get("prefix_identity"), "post": existing.get("prefix_identity")}
                    recovery_prefixes.append(prefix)
                    timing.record("RECOVERY_REPLAY_COMPLETED", prefix)
                timing.finish_trajectory(trajectory)
                continue
            parent_ledger = existing
            for category in EXPECTED_CATEGORIES:
                if parent_ledger and category in parent_ledger["category_seals"]:
                    if recovery:
                        events = registration["categories"][category]["events"]
                        candidates = _checkpoint_candidates(run_root, category, manifest,
                            events[0]["position"], events[-1]["position"] + 1)
                        if not candidates:
                            raise CampaignContractError("sealed category has no verified recovery checkpoint")
                        name = candidates[-1][1]
                        identity = _checkpoint_identity(run_root, trajectory, category, name)
                        recovery_checkpoints.append(identity)
                        timing.record("RECOVERY_CHECKPOINT_SELECTED", identity)
                        prefix = {"trajectory_id": trajectory, "category": category,
                            "pre": parent_ledger.get("prefix_identity"), "post": parent_ledger.get("prefix_identity")}
                        recovery_prefixes.append(prefix)
                        timing.record("RECOVERY_REPLAY_COMPLETED", prefix)
                    continue
                timing.check(trajectory)
                events = registration["categories"][category]["events"]
                category_start, category_end = events[0]["position"], events[-1]["position"] + 1
                current = _inspect_ledger(run_root, manifest)
                if current and current["current_category"] not in (None, category):
                    raise CampaignContractError("ledger is active in a different category; refusing to switch trajectories")
                candidates = _checkpoint_candidates(run_root, category, manifest, category_start, category_end)
                checkpoint = candidates[-1][1] if candidates else None
                if checkpoint and current and current["position"] < candidates[-1][0]:
                    raise CampaignContractError("checkpoint is ahead of its durable ledger")
                if current and current["position"] > category_start and not checkpoint:
                    raise CampaignContractError("advanced ledger has no committed category checkpoint")
                pre_prefix = current.get("prefix_identity") if current else None
                create = current is None
                worker, config_path, run_root, disclosure_root = _start_worker(
                    campaign_root=campaign_root, metadata=binding, trajectory=trajectory,
                    manifest_path=manifest_path, manifest=manifest, category=category,
                    checkpoint=checkpoint, create=create,
                    timeout_seconds=timing.timeout_seconds(trajectory))
                telemetry_path = execution_root / "telemetry" / f"{trajectory}.jsonl"
                try:
                    if checkpoint is None:
                        timing.check(trajectory)
                        worker.checkpoint(f"{category}-start", timeout_seconds=timing.timeout_seconds(trajectory))
                        checkpoint_name = f"{category}-start"
                    else:
                        checkpoint_name = checkpoint
                    if recovery:
                        identity = _checkpoint_identity(run_root, trajectory, category, checkpoint_name)
                        recovery_checkpoints.append(identity)
                        timing.record("RECOVERY_CHECKPOINT_SELECTED", identity)
                    if checkpoint is None:
                        position = category_start
                    else:
                        checkpoint_state = _read_json(run_root / "checkpoints" / checkpoint / "state.json")
                        position = checkpoint_state["ledger_offset"]["next_position"]
                    for event_position in range(position, category_end):
                        timing.check(trajectory)
                        result = worker.step(event_position, timeout_seconds=timing.timeout_seconds(trajectory))
                        expected = events[event_position - category_start]
                        prediction = result.get("prediction", {})
                        if (prediction.get("position") != event_position or prediction.get("event") != expected["event"] or
                                prediction.get("group") != expected["group"]):
                            raise CampaignContractError("worker returned a different registered event identity")
                        timing.check(trajectory)
                        completed = event_position + 1
                        if completed % 1000 == 0:
                            name = f"{category}-event-{completed:05d}"
                            timing.check(trajectory)
                            if name not in {item[1] for item in _checkpoint_candidates(
                                    run_root, category, manifest, category_start, category_end)}:
                                worker.checkpoint(name, timeout_seconds=timing.timeout_seconds(trajectory))
                            timing.record("CHECKPOINT", {"trajectory_id": trajectory, "category": category,
                                "checkpoint_name": name, "completed_events": completed})
                            timing.check(trajectory)
                            _append_telemetry(telemetry_path, {"schema": "FS4_CAMPAIGN_RUNTIME_TELEMETRY_V1",
                                "trajectory_id": trajectory, "category": category, "completed_global_events": completed,
                                "captured_unix_time": time.time(), **worker.stats(
                                    timeout_seconds=timing.timeout_seconds(trajectory))})
                        del result
                    end_name = f"{category}-end"
                    timing.check(trajectory)
                    if end_name not in {item[1] for item in _checkpoint_candidates(
                            run_root, category, manifest, category_start, category_end)}:
                        worker.checkpoint(end_name, timeout_seconds=timing.timeout_seconds(trajectory))
                    timing.record("CHECKPOINT", {"trajectory_id": trajectory, "category": category,
                        "checkpoint_name": end_name, "completed_events": category_end})
                    timing.check(trajectory)
                    worker.seal_category(timeout_seconds=timing.timeout_seconds(trajectory))
                    if category == EXPECTED_CATEGORIES[-1]:
                        timing.check(trajectory)
                        worker.seal_global_a(timeout_seconds=timing.timeout_seconds(trajectory))
                    _append_telemetry(telemetry_path, {"schema": "FS4_CAMPAIGN_RUNTIME_TELEMETRY_V1",
                        "trajectory_id": trajectory, "category": category, "completed_global_events": category_end,
                        "captured_unix_time": time.time(), **worker.stats(
                            timeout_seconds=timing.timeout_seconds(trajectory))})
                finally:
                    worker.close()
                    worker = None
                    try:
                        config_path.unlink()
                    except OSError:
                        pass
                    config_path = None
                parent_ledger = _inspect_ledger(run_root, manifest)
                if recovery:
                    identity = _checkpoint_identity(run_root, trajectory, category, end_name)
                    if identity not in recovery_checkpoints:
                        recovery_checkpoints.append(identity)
                        timing.record("RECOVERY_CHECKPOINT_SELECTED", identity)
                    prefix = {"trajectory_id": trajectory, "category": category,
                        "pre": pre_prefix, "post": parent_ledger.get("prefix_identity")}
                    recovery_prefixes.append(prefix)
                    timing.record("RECOVERY_REPLAY_COMPLETED", prefix)
            timing.finish_trajectory(trajectory)
        projections = {}
        for trajectory in ARM_IDS:
            timing.check()
            projections[trajectory] = _verify_trajectory_stage_a(campaign_root, trajectory,
                write_projection=True, timeout_seconds=timing.timeout_seconds())
        _validate_complete_stage_a(campaign_root, binding, registration, projections, require_terminal=False)
        if recovery:
            timing.record("RECOVERY_LEDGER_VERIFIED", {"verified_complete_stage_a_ledger_identities": {
                key: {"run_id": projections[key]["run_id"], "manifest_sha256": projections[key]["manifest_sha256"],
                    "stage_a_sha256": projections[key]["stage_a_sha256"],
                    "category_seals": projections[key]["category_seals"]} for key in ARM_IDS}})
        timing.check()
        elapsed_totals = timing._current_elapsed()
        completion_record = timing.record("CAMPAIGN_COMPLETE", {"status": "STAGE_A_COMPLETE",
            "elapsed_totals_ns": {"campaign": elapsed_totals[0],
                "trajectories": elapsed_totals[1]}})
        seal = {"schema": "FS4_FINAL_JOINT_STAGE_A_CAMPAIGN_SEAL_V1", "status": "STAGE_A_COMPLETE",
            "campaign_id": binding["campaign_id"], "scientific_spec_sha256": spec_identity(),
            "campaign_registration_sha256": binding["campaign_registration_sha256"],
            "timing_completion_record_sha256": completion_record["record_sha256"],
            "execution_time_totals_ns": completion_record["payload"]["elapsed_totals_ns"],
            "trajectory_seals": {key: {"run_id": projections[key]["run_id"],
                "manifest_sha256": projections[key]["manifest_sha256"], "stage_a_sha256": projections[key]["stage_a_sha256"]}
                for key in ARM_IDS}}
        seal_path = execution_root / "STAGE_A_CAMPAIGN_SEAL.json"
        if seal_path.exists():
            if _read_json(seal_path) != seal:
                raise CampaignContractError("existing Stage A seal differs; refusing overwrite")
        else:
            _write_json_new(seal_path, seal)
        if recovery:
            marker_digest = _sha((execution_root / "INCOMPLETE.json").read_bytes())
            ledger_identities = {key: {"run_id": projections[key]["run_id"],
                "manifest_sha256": projections[key]["manifest_sha256"],
                "stage_a_sha256": projections[key]["stage_a_sha256"],
                "category_seals": projections[key]["category_seals"]} for key in ARM_IDS}
            receipt = {"schema": "FS4_JOINT_CAMPAIGN_RECOVERY_COMPLETION_V1",
                "status": "VERIFIED_RECOVERY_COMPLETION", "campaign_id": binding["campaign_id"],
                "spec_sha256": spec_identity(), "source_freeze_commit": binding["source_freeze_commit"],
                "previous_incomplete_record_sha256": marker_digest,
                "recovery_checkpoint_identities": recovery_checkpoints,
                "replayed_ledger_prefix_identities": recovery_prefixes,
                "verified_complete_stage_a_ledger_identities": ledger_identities,
                "final_trajectory_stage_a_seals": seal["trajectory_seals"],
                "recovery_classification": "VERIFIED_RECOVERY_COMPLETION",
                "timing_ledger_prefix_sha256": completion_record["record_sha256"]}
            receipt["receipt_sha256"] = _sha(canonical(receipt) + b"\n")
            receipt_path = execution_root / "RECOVERY_COMPLETION.json"
            if receipt_path.exists():
                if _read_json(receipt_path) != receipt:
                    raise CampaignContractError("existing recovery completion receipt differs; refusing overwrite")
            else:
                _write_json_new(receipt_path, receipt)
        if timing:
            timing.close()
            timing = None
        _validate_terminal_state(campaign_root, binding, projections)
        return seal
    except Exception as exc:
        if worker is not None:
            try: worker.close()
            except Exception: pass
        try:
            from .fs4_r4_worker import ResearchWorkerTimeout
            worker_timeout = isinstance(exc, ResearchWorkerTimeout)
        except ImportError:
            worker_timeout = False
        if isinstance(exc, CampaignDeadlineExceeded):
            classification = "DEADLINE_EXHAUSTION"
            reason = exc.status
        elif worker_timeout:
            ambiguous = False
            for trajectory in ARM_IDS:
                try:
                    manifest = _load_trajectory_manifest(campaign_root, trajectory)
                    state = _inspect_ledger(execution_root / "runs" / trajectory, manifest)
                    ambiguous = ambiguous or bool(state and state.get("phase") == "LABEL_INTENT")
                except Exception:
                    ambiguous = True
            classification = "AMBIGUOUS_LABEL_DISCLOSURE" if ambiguous else "RECOVERABLE_EXECUTION_INTERRUPTION"
            reason = "WORKER_TIMEOUT_WITH_LABEL_INTENT" if ambiguous else "SUPERVISED_WORKER_TIMEOUT_AT_SAFE_BOUNDARY"
        elif isinstance(exc, RecoverableExecutionInterruption):
            classification, reason = "RECOVERABLE_EXECUTION_INTERRUPTION", "INJECTED_RECOVERABLE_EVENT_BOUNDARY_INTERRUPTION"
        else:
            classification, reason = "UNRECOVERABLE_INTEGRITY_FAILURE", "STAGE_A_INTEGRITY_OR_EXECUTION_FAILURE"
        _record_incomplete(execution_root, reason, campaign_id=binding["campaign_id"],
            source_commit=binding["source_freeze_commit"], classification=classification)
        if timing is not None:
            try:
                timing.record("CAMPAIGN_FAILURE", {"classification": classification, "reason_code": reason})
                timing.close()
            except Exception:
                pass
        raise
    finally:
        if worker is not None:
            try: worker.close()
            except Exception: pass


def _verify_trajectory_stage_a(campaign_root: Path, trajectory: str, *, write_projection: bool,
                               timeout_seconds: float | None = None) -> dict[str, Any]:
    run_root = Path(campaign_root) / "execution" / "runs" / trajectory
    command = [sys.executable, "-m", "experiments.calibration.fs4_research_stage_a_verify", str(run_root)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, check=True,
        timeout=timeout_seconds)
    projection = json.loads(completed.stdout)
    if projection.get("trajectory_id") != trajectory:
        raise CampaignContractError("Stage A verifier returned another trajectory")
    if write_projection:
        path = Path(campaign_root) / "execution" / "sealed-projections" / f"{trajectory}.json"
        if not path.exists():
            _write_json_new(path, projection)
        elif _read_json(path) != projection:
            raise CampaignContractError("existing sealed projection differs; refusing overwrite")
    return projection


def _validate_complete_stage_a(campaign_root: Path, binding: Mapping[str, Any], registration: Mapping[str, Any],
                               projections: Mapping[str, Mapping[str, Any]], *, require_terminal: bool = True) -> None:
    if set(projections) != set(ARM_IDS):
        raise CampaignContractError("all three Stage A trajectories are mandatory")
    reference = None
    for trajectory in ARM_IDS:
        item = projections[trajectory]
        if (item.get("campaign_id") != binding["campaign_id"] or
                item.get("campaign_registration_sha256") != binding["campaign_registration_sha256"] or
                item.get("scientific_spec_sha256") != spec_identity() or
                item.get("source_commit") != binding["source_freeze_commit"] or
                item.get("support_preflight_receipt_sha256") != binding["support_preflight_receipt_sha256"] or
                item.get("r7_seed0_order_sha256") != binding["r7_seed0_order_sha256"] or
                item.get("source_mapping_sha256") != binding["source_mapping_sha256"] or
                item.get("model_state_sha256") != load_spec()["source_model_support"]["model"]["model_state_sha256"]):
            raise CampaignContractError(f"{trajectory}: Stage A campaign identity differs")
        frozen_model = load_spec()["source_model_support"]["model"]
        if (item.get("checkpoint_sha256") != frozen_model["checkpoint_sha256"] or
                item.get("hub_source_tree_sha256") != frozen_model["hub_source_tree_sha256"] or
                item.get("scorer_config_sha256") != frozen_model["native_config_sha256"] or
                item.get("preflight_receipt_sha256") != load_spec()["source_model_support"]["historical_receipt_sha256"]):
            raise CampaignContractError(f"{trajectory}: frozen model or native Support receipt differs")
        if item.get("fixed_support_sha256_by_category") != {
                category: load_spec()["source_model_support"]["support_records_by_category"][category]["fixed_bank_sha256"]
                for category in EXPECTED_CATEGORIES}:
            raise CampaignContractError(f"{trajectory}: four-Support fixed bank identity differs")
        category_seals = item.get("category_seals")
        if (not isinstance(category_seals, Mapping) or set(category_seals) != set(EXPECTED_CATEGORIES) or
                any(not isinstance(value, str) or len(value) != 64 for value in category_seals.values())):
            raise CampaignContractError(f"{trajectory}: all four verified category seals are mandatory")
        if item.get("stream_sha256_by_category") != {
                category: digest([{"group": row["group"], "event": row["event"]}
                                 for row in registration["categories"][category]["events"]])
                for category in EXPECTED_CATEGORIES}:
            raise CampaignContractError(f"{trajectory}: Stage A stream identity differs")
        rows = [(row["event"], row["group"], row["position"], row["view"]) for row in item["events"]]
        if reference is None:
            reference = rows
        elif rows != reference:
            raise CampaignContractError("Stage A trajectories do not share the same opaque event sequence")
        if len(item["events"]) != sum(EXPECTED_EVENTS.values()):
            raise CampaignContractError(f"{trajectory}: Stage A event count is incomplete")
    if require_terminal:
        _validate_terminal_state(campaign_root, binding, projections)


def verify_stage_a(*, campaign_root: Path) -> dict[str, Any]:
    binding, registration = _campaign_with_support(campaign_root)
    projections = {trajectory: _verify_trajectory_stage_a(campaign_root, trajectory, write_projection=True)
                   for trajectory in ARM_IDS}
    _validate_complete_stage_a(campaign_root, binding, registration, projections)
    terminal = _validate_terminal_state(campaign_root, binding, projections)
    return {"status": "PASS", "campaign_id": binding["campaign_id"],
        "terminal_state": terminal["terminal_state"],
        "trajectory_seals": {key: {"stage_a_sha256": projections[key]["stage_a_sha256"],
            "manifest_sha256": projections[key]["manifest_sha256"]} for key in ARM_IDS}}


def _validate_stage_b_grant(grant_path: Path, *, binding: Mapping[str, Any],
                            projections: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    seals = {trajectory: {"manifest_sha256": projections[trajectory]["manifest_sha256"],
        "stage_a_sha256": projections[trajectory]["stage_a_sha256"],
        "run_id": projections[trajectory]["run_id"]} for trajectory in ARM_IDS}
    grant = _validate_authorization(grant_path, scope="STAGE_B", campaign_id=binding["campaign_id"],
        bindings={"campaign_registration_sha256": binding["campaign_registration_sha256"],
                  "source_mapping_sha256": binding["source_mapping_sha256"], "trajectory_seals": seals})
    return grant


def run_stage_b(*, campaign_root: Path, grant_path: Path, result_root: Path) -> dict[str, Any]:
    from .fs4_joint_campaign_evaluator import evaluate_campaign, canonical_bytes, sha256_bytes
    from .fs4_r2_security import WindowsPrivateStore
    campaign_root, result_root = Path(campaign_root).absolute(), Path(result_root).absolute()
    binding, registration = _campaign_with_support(campaign_root)
    if result_root.exists():
        raise FileExistsError("Stage B result root must be fresh and non-overwriting")
    projections = {trajectory: _verify_trajectory_stage_a(campaign_root, trajectory, write_projection=False)
                   for trajectory in ARM_IDS}
    _validate_complete_stage_a(campaign_root, binding, registration, projections)
    _validate_stage_b_grant(grant_path, binding=binding, projections=projections)
    # The full outcome map is opened only after the exact twelve sealed category
    # streams, common registration and separate Stage B authorization pass.
    stage_s = projections["S"]
    run_root = campaign_root / "execution" / "runs" / "S"
    projected = subprocess.run([sys.executable, "-m", "experiments.calibration.fs4_r4_mapping_verify",
        str(binding["private_root"]), str(run_root), stage_s["manifest_sha256"]],
        cwd=ROOT, capture_output=True, check=True)
    outcome_projection = json.loads(projected.stdout)
    labels = outcome_projection.get("labels")
    if (outcome_projection.get("source_mapping_sha256") != binding["source_mapping_sha256"] or
            not isinstance(labels, dict) or
            any(any(labels.get(event) != label for event, label in projections[trajectory]["queried_feedback"].items())
                for trajectory in ARM_IDS)):
        raise CampaignContractError("complete outcome mapping or one branch's queried feedback differs")
    telemetry_by_arm = {}
    for trajectory in ARM_IDS:
        telemetry_path = campaign_root / "execution" / "telemetry" / f"{trajectory}.jsonl"
        if not telemetry_path.is_file():
            raise CampaignContractError(f"{trajectory}: runtime/resource telemetry is missing")
        rows = [json.loads(line) for line in telemetry_path.read_bytes().splitlines()]
        if {row.get("category") for row in rows} != set(EXPECTED_CATEGORIES):
            raise CampaignContractError(f"{trajectory}: per-category runtime telemetry is incomplete")
        telemetry_by_arm[trajectory] = rows
    for other in (campaign_root, Path(binding["private_root"]).absolute(), Path(binding["image_root"]).absolute()):
        if result_root == other or result_root.is_relative_to(other) or other.is_relative_to(result_root):
            raise CampaignContractError("Stage B results must be isolated from campaign, mapping and image stores")
    result_store = WindowsPrivateStore(result_root, create=True)
    result_store.write("INCOMPLETE.json", canonical({"schema": "FS4_FINAL_JOINT_STAGE_B_INCOMPLETE_V1",
        "status": "INCOMPLETE"}) + b"\n")
    files: dict[str, str] = {}
    for trajectory in ARM_IDS:
        data = canonical_bytes(projections[trajectory]) + b"\n"
        result_store.write(f"stage_a_{trajectory}.json", data)
        files[f"stage_a_{trajectory}.json"] = sha256_bytes(data)
        telemetry_data = canonical_bytes(telemetry_by_arm[trajectory]) + b"\n"
        result_store.write(f"telemetry_{trajectory}.json", telemetry_data)
        files[f"telemetry_{trajectory}.json"] = sha256_bytes(telemetry_data)
    labels_data = canonical_bytes(outcome_projection) + b"\n"
    result_store.write("outcome_projection.json", labels_data)
    files["outcome_projection.json"] = sha256_bytes(labels_data)
    report = evaluate_campaign(projections, labels, telemetry_by_arm)
    report_data = canonical_bytes(report) + b"\n"
    result_store.write("metrics.json", report_data)
    files["metrics.json"] = sha256_bytes(report_data)
    grant_data = Path(grant_path).read_bytes()
    result_store.write("stage_b_authorization.json", grant_data)
    files["stage_b_authorization.json"] = sha256_bytes(grant_data)
    campaign_result = {"schema": "FS4_FINAL_JOINT_CAMPAIGN_RESULT_V1", "status": "COMPLETE",
        "campaign_id": binding["campaign_id"], "scientific_spec_sha256": spec_identity(),
        "campaign_registration_sha256": binding["campaign_registration_sha256"],
        "source_mapping_sha256": binding["source_mapping_sha256"],
        "trajectory_seals": {key: {"manifest_sha256": projections[key]["manifest_sha256"],
            "stage_a_sha256": projections[key]["stage_a_sha256"], "run_id": projections[key]["run_id"]}
            for key in ARM_IDS}, "metrics_sha256": sha256_bytes(report_data)}
    campaign_result_data = canonical_bytes(campaign_result) + b"\n"
    result_store.write("campaign_result.json", campaign_result_data)
    files["campaign_result.json"] = sha256_bytes(campaign_result_data)
    complete = {"schema": "FS4_FINAL_JOINT_CAMPAIGN_COMPLETE_V1", "status": "COMPLETE",
        "campaign_result_sha256": sha256_bytes(campaign_result_data), "files_sha256": files}
    result_store.write("COMPLETE.json", canonical_bytes(complete) + b"\n")
    from .fs4_r2_security import validate_private_acl
    incomplete_marker = result_root / "INCOMPLETE.json"
    validate_private_acl(incomplete_marker, directory=False)
    incomplete_marker.unlink()
    return verify_complete(result_root)


def verify_complete(result_root: Path) -> dict[str, Any]:
    from .fs4_joint_campaign_evaluator import evaluate_campaign, canonical_bytes, sha256_bytes
    from .fs4_r2_security import WindowsPrivateStore
    store = WindowsPrivateStore(Path(result_root), create=False)
    complete = json.loads(store.read("COMPLETE.json"))
    if complete.get("schema") != "FS4_FINAL_JOINT_CAMPAIGN_COMPLETE_V1" or complete.get("status") != "COMPLETE":
        raise CampaignContractError("Stage B completion seal is missing")
    expected_files = {*(f"stage_a_{arm}.json" for arm in ARM_IDS),
        *(f"telemetry_{arm}.json" for arm in ARM_IDS), "outcome_projection.json", "metrics.json",
        "stage_b_authorization.json", "campaign_result.json"}
    if set(complete.get("files_sha256", {})) != expected_files:
        raise CampaignContractError("Stage B completion seal has an unexpected file inventory")
    loaded = {}
    for name, expected in complete["files_sha256"].items():
        data = store.read(name)
        if _sha(data) != expected:
            raise CampaignContractError(f"Stage B file checksum differs: {name}")
        loaded[name] = json.loads(data)
    result = loaded["campaign_result.json"]
    if _sha(store.read("campaign_result.json")) != complete["campaign_result_sha256"]:
        raise CampaignContractError("campaign result receipt checksum differs")
    stage_a = {trajectory: loaded[f"stage_a_{trajectory}.json"] for trajectory in ARM_IDS}
    telemetry = {trajectory: loaded[f"telemetry_{trajectory}.json"] for trajectory in ARM_IDS}
    outcome_projection = loaded["outcome_projection.json"]
    labels = outcome_projection.get("labels")
    expected_seals = {trajectory: {"manifest_sha256": stage_a[trajectory]["manifest_sha256"],
        "stage_a_sha256": stage_a[trajectory]["stage_a_sha256"], "run_id": stage_a[trajectory]["run_id"]}
        for trajectory in ARM_IDS}
    if (result.get("trajectory_seals") != expected_seals or
            result.get("source_mapping_sha256") != outcome_projection.get("source_mapping_sha256") or
            outcome_projection.get("manifest_sha256") != stage_a["S"].get("manifest_sha256") or
            any(stage_a[trajectory].get("source_mapping_sha256") != result.get("source_mapping_sha256") or
                stage_a[trajectory].get("campaign_id") != result.get("campaign_id") or
                stage_a[trajectory].get("scientific_spec_sha256") != result.get("scientific_spec_sha256") or
                stage_a[trajectory].get("campaign_registration_sha256") != result.get("campaign_registration_sha256")
                for trajectory in ARM_IDS)):
        raise CampaignContractError("final outcome projection, campaign registration or Stage A seals do not agree")
    grant = loaded["stage_b_authorization.json"]
    expected_grant_keys = {"schema", "authorization_id", "authorized_by", "authorized_at", "scope",
        "campaign_id", "scientific_spec_sha256", "authorization_statement", "campaign_registration_sha256",
        "source_mapping_sha256", "trajectory_seals"}
    if (set(grant) != expected_grant_keys or grant.get("schema") != AUTH_SCHEMA or grant.get("scope") != "STAGE_B" or
            grant.get("campaign_id") != result.get("campaign_id") or
            grant.get("scientific_spec_sha256") != result.get("scientific_spec_sha256") or
            grant.get("campaign_registration_sha256") != result.get("campaign_registration_sha256") or
            grant.get("source_mapping_sha256") != result.get("source_mapping_sha256") or
            grant.get("trajectory_seals") != expected_seals or
            any(not isinstance(grant.get(key), str) or not grant[key].strip()
                for key in ("authorization_id", "authorized_by", "authorized_at", "authorization_statement"))):
        raise CampaignContractError("stored Stage B authorization no longer matches the completed campaign")
    if (not isinstance(labels, dict) or
            any(any(labels.get(event) != label for event, label in stage_a[trajectory]["queried_feedback"].items())
                for trajectory in ARM_IDS)):
        raise CampaignContractError("final outcome labels disagree with a trajectory's verified feedback")
    recomputed = evaluate_campaign(stage_a, labels, telemetry)
    if canonical_bytes(recomputed) + b"\n" != store.read("metrics.json"):
        raise CampaignContractError("stored metrics do not reproduce from the sealed Stage A projections and labels")
    if result.get("metrics_sha256") != _sha(store.read("metrics.json")):
        raise CampaignContractError("campaign result metric checksum differs")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", nargs="?", default="freeze-only", choices=("freeze-only", "validate-contract",
        "prepare-real-data", "verify-support-preflight", "resource-preflight", "run-stage-a",
        "verify-stage-a", "run-stage-b", "verify-complete"))
    parser.add_argument("--campaign-id")
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--private-root", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--original-registration", type=Path)
    parser.add_argument("--original-ledger", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--support-receipt", type=Path)
    parser.add_argument("--benchmark-receipt", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--grant", type=Path)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "freeze-only": result = freeze_only()
        elif args.operation == "validate-contract": result = validate_contract()
        elif args.operation == "prepare-real-data":
            required = (args.campaign_id, args.campaign_root, args.private_root, args.image_root,
                        args.original_registration, args.original_ledger, args.labels, args.authorization)
            if any(item is None for item in required): raise ValueError("prepare-real-data requires all staging inputs and authorization")
            result = prepare_real_data(campaign_id=args.campaign_id, campaign_root=args.campaign_root,
                private_root=args.private_root, image_root=args.image_root,
                original_registration_path=args.original_registration, original_ledger_path=args.original_ledger,
                labels_path=args.labels, authorization_path=args.authorization)
        elif args.operation == "verify-support-preflight":
            if any(item is None for item in (args.campaign_root, args.support_receipt, args.authorization)):
                raise ValueError("verify-support-preflight requires campaign root, receipt and authorization")
            result = verify_support_preflight(campaign_root=args.campaign_root, receipt_path=args.support_receipt,
                                              authorization_path=args.authorization)
        elif args.operation == "resource-preflight":
            if args.campaign_root is None or args.authorization is None: raise ValueError("resource-preflight requires campaign root and authorization")
            result = resource_preflight(campaign_root=args.campaign_root, benchmark_receipt_path=args.benchmark_receipt,
                                        authorization_path=args.authorization)
        elif args.operation == "run-stage-a":
            if args.campaign_root is None or args.authorization is None: raise ValueError("run-stage-a requires campaign root and authorization")
            result = run_stage_a(campaign_root=args.campaign_root, authorization_path=args.authorization, resume=args.resume)
        elif args.operation == "verify-stage-a":
            if args.campaign_root is None: raise ValueError("verify-stage-a requires campaign root")
            result = verify_stage_a(campaign_root=args.campaign_root)
        elif args.operation == "run-stage-b":
            if any(item is None for item in (args.campaign_root, args.grant, args.result_root)):
                raise ValueError("run-stage-b requires campaign root, separate grant and fresh result root")
            result = run_stage_b(campaign_root=args.campaign_root, grant_path=args.grant, result_root=args.result_root)
        else:
            if args.result_root is None: raise ValueError("verify-complete requires result root")
            result = verify_complete(args.result_root)
        sys.stdout.buffer.write(canonical(result) + b"\n")
        return 0
    except Exception as exc:
        print(f"FS4_CAMPAIGN_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
