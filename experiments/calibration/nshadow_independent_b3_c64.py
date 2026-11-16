"""N-shadow + independent current-state B3/C64 supplementary experiment.

This is a post-hoc, query-event-only replay over sealed final-benchmark
trajectories.  It reuses the registered native F-B/Greedy and N-shadow C1
updates, while adding a separate R7 B3 controller calibrated on protected C64.
The original Stage A and Stage B artifacts are read-only.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import shutil
import threading
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "outputs/calibration/nshadow_independent_b3_c64_20261117_v1"
REAL_PLANS = (
    ROOT / "outputs/calibration/final_benchmark_launch/private_binding/realiad-final-20261111-r1-40d61c3/final_bound_plan.json",
    ROOT / "outputs/calibration/final_benchmark_launch/private_binding/realiad-final-20261113-remaining54-a3688e87ac16/final_bound_plan.json",
)
REAL_RUNS = (
    "realiad-final-20261111-r1-40d61c3",
    "realiad-final-20261113-remaining54-a3688e87ac16",
)
REAL_STAGE_B_ROOT = ROOT / "outputs/calibration/final_benchmark_stage_b/realiad-stageb-90-20261114-v1"
VISA_REG_ROOT = ROOT / "outputs/calibration/final_benchmark/visa_1cls/registration_01"
VISA_STAGE_A_ROOT = ROOT / "outputs/calibration/final_benchmark/visa_1cls"
REAL_DATA_ROOT = Path(r"G:\dataset\realiad")
VISA_DATA_ROOT = Path(r"G:\dataset\VisA")
VISA_CATEGORIES = (
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
)
PRINCIPAL_ARMS = ("F-B", "K-B", "F-M", "K-M")
INHERITED_FEEDBACK_ORDER = PRINCIPAL_ARMS
STATE_REPLAY_TOLERANCE = 1e-5
BOOTSTRAP_REPLICATES = 10_000
REAL_BOOTSTRAP_SEED = 20_261_117
VISA_BOOTSTRAP_SEED = 20_261_118


class SupplementaryError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_object(value: Any) -> str:
    return sha256_bytes(canonical(value))


def _original_unit_schema(dataset_id: str) -> str:
    schemas = {"realiad": "final-benchmark.stage-a-group.v1",
               "visa_1cls": "final-benchmark.stage-a-atomic-unit.v1"}
    try:
        return schemas[dataset_id]
    except KeyError as exc:
        raise SupplementaryError(f"unsupported dataset unit schema: {dataset_id}") from exc


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise SupplementaryError(f"cannot read registered JSON metadata: {Path(path).name}") from exc
    if not isinstance(value, dict):
        raise SupplementaryError(f"registered JSON root is not an object: {Path(path).name}")
    return value


def write_exclusive(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_bytes(payload)


def write_text_exclusive(path: Path, value: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_bytes(payload)


def append_fsync(path: Path, value: Any) -> str:
    payload = canonical(value) + b"\n"
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_bytes(payload)


def write_progress(path: Path, value: Mapping[str, Any]) -> None:
    """Small, replaceable progress state; scientific journals remain append-only."""
    temporary = path.with_name(path.name + ".tmp")
    payload = canonical(dict(value)) + b"\n"
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@dataclass
class Trajectory:
    dataset_id: str
    category: str
    condition: str
    source_commit: str
    attempt_root: Path
    private_feedback_path: Path
    plan_path: Path
    plan_sha256: str
    bundle: dict[str, Any]
    identities: dict[str, Any]
    role_manifest: dict[str, Any]
    stream_manifest: dict[str, Any]
    release_identity: dict[str, Any]
    records: list[dict[str, Any]]
    stream_seed: int
    support_seed: int
    registration: dict[str, Any]
    original_stage_b: dict[str, Any]
    input_audit: dict[str, Any] | None = None

    @property
    def key(self) -> str:
        return f"{self.dataset_id}/{self.category}/{self.condition}"

    @property
    def event_count(self) -> int:
        return len(self.records)


def _attempt_seal_and_rows(trajectory: Trajectory, *, verify_all: bool = True
                           ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Verify immutable original status, seal and journal; does not open outcomes."""
    from experiments.calibration.final_benchmark.execution_support import verify_attempt_receipt

    status_path = trajectory.attempt_root / "attempt_status.json"
    status = read_json(status_path)
    expected_identity = status.get("identity")
    if not isinstance(expected_identity, dict):
        raise SupplementaryError(f"{trajectory.key}: historical status lacks identity binding")
    expected_count = trajectory.event_count
    if (status.get("schema") != "final-benchmark.attempt-status.v1" or
            status.get("status") != "SEALED_COMPLETE" or
            status.get("execution_kind") != "principal_stage_a" or
            status.get("expected_event_count") != expected_count or
            status.get("completed_event_count") != expected_count):
        raise SupplementaryError(f"{trajectory.key}: original Stage A attempt is not complete")
    try:
        verified = verify_attempt_receipt(status_path, expected_identity=expected_identity,
                                          expected_kind="principal_stage_a")
    except Exception as exc:
        raise SupplementaryError(f"{trajectory.key}: original completion receipt failed verification") from exc
    seal = verified["seal"]
    expected_seal_fields = {
        "dataset_id": trajectory.dataset_id, "category": trajectory.category,
        "run_condition": trajectory.condition, "expected_event_count": expected_count,
        "completed_event_count": expected_count, "status": "SEALED_COMPLETE",
    }
    if any(seal.get(key) != value for key, value in expected_seal_fields.items()):
        raise SupplementaryError(f"{trajectory.key}: original seal differs from registered trajectory")
    journal_path = trajectory.attempt_root / "stage_a_journal.jsonl"
    rows: list[dict[str, Any]] = []
    group_seals: list[dict[str, Any]] = []
    cursor = 0
    with journal_path.open("rb") as stream:
        header_line = stream.readline()
        try:
            header = json.loads(header_line)
        except Exception as exc:
            raise SupplementaryError(f"{trajectory.key}: malformed original journal header") from exc
        if header.get("schema") != "final-benchmark.stage-a-journal-header.v1":
            raise SupplementaryError(f"{trajectory.key}: wrong original journal schema")
        for field in ("dataset_id", "category", "condition_id", "dataset_release_sha256",
                      "role_manifest_sha256", "source_commit"):
            expected_value = {
                "dataset_id": trajectory.dataset_id, "category": trajectory.category,
                "condition_id": trajectory.condition,
                "dataset_release_sha256": trajectory.release_identity.get("dataset_release_sha256"),
                "role_manifest_sha256": sha256_object(trajectory.role_manifest),
                "source_commit": trajectory.source_commit,
            }[field]
            if header.get(field) != expected_value:
                raise SupplementaryError(f"{trajectory.key}: original journal header mismatch: {field}")
        for raw_line in stream:
            try:
                envelope = json.loads(raw_line)
            except Exception as exc:
                raise SupplementaryError(f"{trajectory.key}: malformed original unit envelope") from exc
            seal_key = "group_seal" if trajectory.dataset_id == "realiad" else "unit_seal"
            unit = envelope.get(seal_key)
            unit_rows = envelope.get("score_rows")
            expected_envelope_schema = _original_unit_schema(trajectory.dataset_id)
            if (envelope.get("schema") != expected_envelope_schema or
                    not isinstance(unit, dict) or not isinstance(unit_rows, list) or not unit_rows):
                raise SupplementaryError(f"{trajectory.key}: invalid original atomic-unit envelope")
            start, stop = int(unit.get("start_position", -1)), int(unit.get("end_position_exclusive", -1))
            if (start != cursor or stop != start + len(unit_rows) or
                    unit.get("group_index") != len(group_seals) or
                    unit.get("status") != "SEALED_BEFORE_LABEL_DISCLOSURE" or
                    unit.get("score_rows_sha256") != sha256_object(unit_rows) or
                    unit.get("event_order_sha256") != sha256_object(
                        [row.get("event_id_sha256") for row in unit_rows])):
                raise SupplementaryError(f"{trajectory.key}: original unit seal/order/hash mismatch")
            if trajectory.dataset_id == "realiad":
                if stop - start != 5 or unit.get("source_group_sha256") != unit_rows[0].get("source_group_sha256"):
                    raise SupplementaryError(f"{trajectory.key}: Real-IAD unit is not a complete five-view group")
                if [int(row.get("camera_id", -1)) for row in unit_rows] != [1, 2, 3, 4, 5]:
                    raise SupplementaryError(f"{trajectory.key}: Real-IAD camera order is invalid")
            else:
                if (unit.get("atomic_unit_sha256") != unit.get("source_group_sha256") or
                        unit.get("unit_candidate_sha256") != trajectory.registration.get("unit_candidate_sha256") or
                        unit.get("unit_approval_sha256") != trajectory.registration.get("unit_approval_sha256")):
                    raise SupplementaryError(f"{trajectory.key}: VisA atomic-unit authorization identity mismatch")
                for arm in PRINCIPAL_ARMS:
                    if sum(bool(row.get("query_actions", {}).get(arm, {}).get("executed"))
                           for row in unit_rows) > 1:
                        raise SupplementaryError(f"{trajectory.key}: multiple queries in one registered VisA unit")
            for offset, row in enumerate(unit_rows):
                position = start + offset
                registered = trajectory.records[position]
                for key in ("position", "event_id_sha256", "image_identity_sha256", "source_group_sha256"):
                    if row.get(key) != registered.get(key):
                        raise SupplementaryError(f"{trajectory.key}:{position}: original event identity/order mismatch")
                if trajectory.dataset_id == "realiad":
                    if row.get("camera_id") != registered.get("camera_id"):
                        raise SupplementaryError(f"{trajectory.key}:{position}: original camera identity mismatch")
                    if row.get("source_group_sha256") != unit.get("source_group_sha256"):
                        raise SupplementaryError(f"{trajectory.key}:{position}: source group split across seal")
                else:
                    for key in ("atomic_unit_sha256", "atomic_unit_kind"):
                        if row.get(key) != registered.get(key):
                            raise SupplementaryError(f"{trajectory.key}:{position}: VisA unit identity mismatch")
                if (not math.isfinite(float(row.get("n_shadow_score", float("nan")))) or
                        not isinstance(row.get("n_shadow_state_sha256"), str) or
                        not isinstance(row.get("n_shadow_state_version"), int)):
                    raise SupplementaryError(f"{trajectory.key}:{position}: original N-shadow state record is malformed")
                actions = row.get("query_actions")
                if not isinstance(actions, dict) or set(actions) != set(PRINCIPAL_ARMS):
                    raise SupplementaryError(f"{trajectory.key}:{position}: original query-action schema is invalid")
            durable = {**unit, "durable_group_receipt_sha256": sha256_bytes(raw_line)}
            group_seals.append(durable)
            rows.extend(dict(row) for row in unit_rows)
            cursor = stop
    if (cursor != expected_count or len(rows) != expected_count or
            len(group_seals) != seal.get("group_count") or
            sha256_object(rows) != seal.get("score_rows_sha256") or
            sha256_object(group_seals) != seal.get("group_seals_sha256")):
        raise SupplementaryError(f"{trajectory.key}: original complete-stream digest differs from its seal")
    if verify_all:
        query_counts = {arm: sum(bool(row["query_actions"][arm].get("executed")) for row in rows)
                        for arm in PRINCIPAL_ARMS}
        expected_feedback_lines = sum(query_counts.values())
        feedback_path = trajectory.private_feedback_path
        line_count = sum(1 for _ in feedback_path.open("rb"))
        if line_count != expected_feedback_lines:
            raise SupplementaryError(f"{trajectory.key}: private query ledger count differs from sealed queries")
        if trajectory.input_audit is not None:
            trajectory.input_audit.update({
                "original_status_sha256": sha256_file(status_path),
                "original_seal_sha256": sha256_file(verified["seal_path"]),
                "original_journal_sha256": sha256_file(journal_path),
                "original_private_feedback_sha256": sha256_file(feedback_path),
                "event_count": expected_count, "atomic_unit_count": len(group_seals),
                "executed_query_count_by_arm": query_counts,
                "private_feedback_line_count": line_count,
                "original_stage_a_status": "SEALED_COMPLETE",
                "labels_or_masks_read_during_asset_audit": False,
            })
    return seal, rows, group_seals


def _real_trajectories() -> list[Trajectory]:
    from experiments.calibration.final_benchmark import launch
    candidates: list[Trajectory] = []
    plan_by_run: dict[str, tuple[Path, dict[str, Any], str]] = {}
    for plan_path, run_id in zip(REAL_PLANS, REAL_RUNS):
        config = launch.read_json(plan_path)
        # Historical source commits are data provenance here; the registered
        # plan and schedule remain strictly validated without binding to HEAD.
        launch.validate_config(config, current_head=None, require_source_binding=True)
        plan_sha = sha256_file(plan_path)
        if config.get("source_identity", {}).get("campaign_run_id") != run_id:
            raise SupplementaryError(f"Real-IAD plan does not match its immutable campaign ID: {run_id}")
        plan_by_run[run_id] = (plan_path, config, plan_sha)
    stage_b_auth = read_json(REAL_STAGE_B_ROOT / "stage_b_authorization.json")
    stage_b_summary = read_json(REAL_STAGE_B_ROOT / "stage_b_metrics_90.json")
    if (stage_b_auth.get("status") != "AUTHORIZED_POST_STAGE_A_LABEL_PROJECTION" or
            stage_b_auth.get("dataset_id") != "realiad" or
            stage_b_auth.get("registered_trajectory_count") != 90 or
            stage_b_auth.get("registered_test_image_events") != 343755 or
            stage_b_summary.get("schema") != "realiad.stage-b-final-metrics.v1" or
            stage_b_summary.get("status") != "STAGE_B_ALL_90_CPU_EVALUATED" or
            stage_b_summary.get("stage_b_authorization_sha256") != sha256_file(REAL_STAGE_B_ROOT / "stage_b_authorization.json")):
        raise SupplementaryError("combined Real-IAD Stage B provenance does not match its registered 90 trajectories")
    registered_stage_b = stage_b_summary.get("scientific_results")
    if not isinstance(registered_stage_b, list) or len(registered_stage_b) != 90:
        raise SupplementaryError("combined Real-IAD Stage B summary lacks its exact 90 result records")
    stage_b_by_key = {(row["category"], row["condition"]): row for row in registered_stage_b}
    if len(stage_b_by_key) != 90:
        raise SupplementaryError("combined Real-IAD Stage B trajectory identities are not unique")
    # The 151k-row identity and locator files are shared by all 90 leaves.
    # Cache only these immutable registration objects during discovery.
    uncached_read_json = launch.read_json
    read_cache: dict[str, dict[str, Any]] = {}
    def cached_read_json(path: str | Path) -> dict[str, Any]:
        resolved = str(Path(path).resolve())
        if Path(path).name in {"realiad_identity_manifest.private.json",
                               "realiad_private_locators.json"}:
            if resolved not in read_cache:
                read_cache[resolved] = uncached_read_json(path)
            return read_cache[resolved]
        return uncached_read_json(path)
    launch.read_json = cached_read_json
    for run_id, (plan_path, config, plan_sha) in plan_by_run.items():
        bundle_root = plan_path.parent
        for row in config["trajectories"]:
            category, condition = str(row["category"]), str(row["condition"])
            campaign_root = ROOT / "outputs/calibration/final_benchmark" / run_id / "stage_a" / category / condition
            preliminary = []
            for attempt_root in sorted(campaign_root.glob("attempt_*")):
                status_path = attempt_root / "attempt_status.json"
                if not status_path.is_file():
                    continue
                candidate_status = read_json(status_path)
                identity = candidate_status.get("identity", {})
                if (candidate_status.get("status") == "SEALED_COMPLETE" and
                        candidate_status.get("expected_event_count") == row["test_event_count"] and
                        identity.get("campaign_run_id") == run_id and
                        identity.get("category") == category and
                        identity.get("condition") == condition and
                        identity.get("source_commit") == config["source_identity"]["final_source_commit"]):
                    preliminary.append(attempt_root)
            if len(preliminary) != 1:
                continue
            bundle_path = bundle_root / f"{category}_{condition}_private_bundle.json"
            if not bundle_path.is_file():
                continue
            payload, identities = launch.load_principal_bundle(bundle_path, config=config,
                category=category, condition=condition)
            identities["execution_plan_sha256"] = plan_sha
            valid: list[Path] = []
            for attempt_root in preliminary:
                status = read_json(attempt_root / "attempt_status.json")
                identity = status.get("identity", {})
                expected_values = {
                    "campaign_run_id": run_id,
                    "source_commit": config["source_identity"]["final_source_commit"],
                    "category": category, "condition": condition,
                    "execution_plan_sha256": plan_sha,
                    "execution_plan_file_sha256": plan_sha,
                    "identity_manifest_sha256": identities["identity_manifest_sha256"],
                    "role_manifest_sha256": identities["role_manifest_sha256"],
                    "stream_manifest_sha256": identities["stream_manifest_sha256"],
                    "dataset_release_sha256": identities["dataset_release_sha256"],
                    "annotation_sha256": identities["annotation_sha256"],
                    "category_bundle_file_sha256": identities["category_bundle_file_sha256"],
                    "private_locator_manifest_file_sha256": identities["private_locator_manifest_file_sha256"],
                }
                if (status.get("status") == "SEALED_COMPLETE" and
                        status.get("expected_event_count") == row["test_event_count"] and
                        all(identity.get(key) == value for key, value in expected_values.items())):
                    valid.append(attempt_root)
            if len(valid) != 1:
                continue
            category_bundle = payload["category_bundle"]
            trajectory = Trajectory(
                dataset_id="realiad", category=category, condition=condition,
                source_commit=str(config["source_identity"]["final_source_commit"]),
                attempt_root=valid[0], private_feedback_path=valid[0] / "private_feedback.jsonl",
                plan_path=plan_path, plan_sha256=plan_sha, bundle=payload,
                identities=identities, role_manifest=dict(category_bundle["role_manifest"]),
                stream_manifest=dict(category_bundle["stream_manifest"]),
                release_identity=dict(category_bundle["release_identity"]),
                records=list(category_bundle["stream_manifest"]["events"]),
                stream_seed=int(category_bundle["stream_seed"]),
                support_seed=int(category_bundle["support_seed"]),
                registration={"trajectory_plan_sha256": plan_sha,
                    "role_manifest_sha256": identities["role_manifest_sha256"],
                    "stream_manifest_sha256": identities["stream_manifest_sha256"],
                    "annotation_sha256": identities["annotation_sha256"],
                    "category_bundle_file_sha256": identities["category_bundle_file_sha256"]},
                original_stage_b=stage_b_by_key[(category, condition)])
            trajectory.input_audit = {
                "dataset_id": "realiad", "category": category, "condition": condition,
                "source_commit": trajectory.source_commit, "campaign_run_id": run_id,
                "plan_sha256": plan_sha, "role_manifest_sha256": identities["role_manifest_sha256"],
                "stream_manifest_sha256": identities["stream_manifest_sha256"],
                "dataset_release_sha256": identities["dataset_release_sha256"],
                "annotation_sha256": identities["annotation_sha256"],
                "attempt_root_relative": str(valid[0].relative_to(ROOT)),
                "original_stage_b_record_identity_matches": (
                    trajectory.original_stage_b.get("category") == category and
                    trajectory.original_stage_b.get("condition") == condition and
                    trajectory.original_stage_b.get("stage_a_category_seal_sha256") ==
                    read_json(valid[0] / "attempt_status.json").get("completion_seal_sha256")),
                "original_stage_b_authorization_sha256": sha256_file(REAL_STAGE_B_ROOT / "stage_b_authorization.json"),
            }
            candidates.append(trajectory)
    launch.read_json = uncached_read_json
    by_key = {(t.category, t.condition): t for t in candidates}
    expected = {(str(row["category"]), str(row["condition"]))
                for plan_path, _config, _sha in plan_by_run.values()
                for row in _config["trajectories"]}
    if len(expected) != 90 or set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        raise SupplementaryError(f"Real-IAD sealed input census is {len(by_key)}/90; missing={missing[:8]}")
    return sorted(candidates, key=lambda t: (t.category, {"P0": 0, "P1": 1, "P2": 2}[t.condition]))


def _visa_trajectories() -> list[Trajectory]:
    from experiments.calibration.final_benchmark import launch
    index = read_json(VISA_REG_ROOT / "registration_index.json")
    if (index.get("schema") != "final-benchmark.visa-registration-index.v1" or
            index.get("dataset_id") != "visa_1cls" or
            not isinstance(index.get("categories"), list) or not index["categories"]):
        raise SupplementaryError("VisA registration index is missing or incompatible")
    category_rows = {row["category"]: row for row in index["categories"]}
    if set(category_rows) != set(VISA_CATEGORIES) or index.get("total_test_images") != 2162:
        raise SupplementaryError("VisA P0 registration does not contain the exact twelve-category census")
    results: list[Trajectory] = []
    for category in VISA_CATEGORIES:
        entry = category_rows[category]
        role = read_json(VISA_REG_ROOT / entry["role_manifest_path"])
        stream = read_json(VISA_REG_ROOT / entry["stream_manifest_path"])
        release = read_json(VISA_REG_ROOT / entry["release_identity_path"])
        locators = read_json(VISA_REG_ROOT / "private" / category / "locators.private.json")
        candidate = read_json(VISA_REG_ROOT / entry["unit_candidate_path"])
        private = VISA_REG_ROOT / "private" / category
        plan_path = private / "execution_plan.json"
        auth_path = private / "execution_authorization.json"
        stage_b_auth_path = private / "stage_b_attempt_01_authorization.json"
        plan = read_json(plan_path)
        auth = read_json(auth_path)
        stage_b_auth = read_json(stage_b_auth_path)
        plan_file_sha, auth_sha = sha256_file(plan_path), sha256_file(auth_path)
        plan_identity_sha = str(plan.get("execution_plan_sha256", ""))
        candidate_sha = str(candidate.get("candidate_sha256", ""))
        stream_unsigned = dict(stream)
        stream_sha = stream_unsigned.pop("stream_manifest_sha256", None)
        if (role.get("category") != category or stream.get("category") != category or
                stream.get("dataset_id") != "visa_1cls" or
                sha256_object(role) != entry["role_manifest_sha256"] or
                stream_sha != entry["stream_manifest_sha256"] or
                sha256_object(stream_unsigned) != stream_sha or
                release.get("dataset_release_sha256") != index.get("dataset_release_sha256") or
                candidate_sha != entry["unit_candidate_sha256"] or
                not plan_identity_sha or
                auth.get("execution_plan_file_sha256") != plan_file_sha or
                auth.get("status") != "RESEARCHER_APPROVED_VISA_ATOMIC_UNIT_EXECUTION" or
                auth.get("stage_b_authorized") is not False):
            raise SupplementaryError(f"VisA {category}: registered role/stream/auth identity mismatch")
        attempt_root = VISA_STAGE_A_ROOT / category / "P0" / f"{category}_P0_attempt_01"
        status_path = attempt_root / "attempt_status.json"
        status = read_json(status_path)
        identity = status.get("identity", {})
        records = list(stream.get("stream_order", []))
        expected_identity = {
            "source_commit": index["source_commit"], "dataset_id": "visa_1cls",
            "category": category, "condition_id": "P0",
            "dataset_release_sha256": index["dataset_release_sha256"],
            "identity_manifest_sha256": index["identity_manifest_sha256"],
            "role_manifest_sha256": entry["role_manifest_sha256"],
            "stream_manifest_sha256": entry["stream_manifest_sha256"],
            "execution_plan_sha256": plan_identity_sha,
            "execution_authorization_sha256": auth_sha,
            "unit_candidate_sha256": candidate_sha,
        }
        if (status.get("status") != "SEALED_COMPLETE" or
                status.get("expected_event_count") != entry["test_images"] or
                status.get("completed_event_count") != entry["test_images"] or
                any(identity.get(key) != value for key, value in expected_identity.items()) or
                not attempt_root.joinpath("stage_a_journal.jsonl").is_file() or
                not attempt_root.joinpath("private_feedback.jsonl").is_file()):
            raise SupplementaryError(f"VisA {category}: original P0 attempt is not the registered sealed attempt")
        approval_sha = str(identity.get("unit_approval_sha256", ""))
        if (stage_b_auth.get("status") != "RESEARCHER_AUTHORIZED_POST_SEAL_VISA_STAGE_B" or
                stage_b_auth.get("scope") != f"{category}_P0_post_seal_only" or
                stage_b_auth.get("unit_candidate_sha256") != candidate_sha or
                stage_b_auth.get("unit_approval_sha256") != approval_sha):
            raise SupplementaryError(f"VisA {category}: existing Stage B authorization metadata mismatch")
        original_stage_b_path = (VISA_STAGE_A_ROOT / category / "P0" /
            f"{category}_P0_attempt_01_stage_b_attempt_01" / "stage_b_metrics.json")
        original_stage_b = read_json(original_stage_b_path)
        if (original_stage_b.get("stage_a_seal_sha256") != status.get("completion_seal_sha256") or
                original_stage_b.get("stage_a_journal_sha256") != sha256_file(attempt_root / "stage_a_journal.jsonl") or
                original_stage_b.get("stage_b_authorization_sha256") != sha256_object(stage_b_auth) or
                original_stage_b.get("dataset_id") != "visa_1cls" or
                original_stage_b.get("category") != category):
            raise SupplementaryError(f"VisA {category}: existing Stage B result does not bind to original seals")
        event_locators = locators
        if isinstance(locators.get("locators"), dict):
            event_locators = locators["locators"]
        if not {str(record["image_identity_sha256"]) for record in records}.issubset(set(event_locators)):
            raise SupplementaryError(f"VisA {category}: private TEST locators do not match registered P0 identities")
        dataset_root = next(iter(event_locators.values())).get("dataset_root") if event_locators else None
        if not isinstance(dataset_root, str) or not Path(dataset_root).is_dir():
            raise SupplementaryError(f"VisA {category}: registered local dataset root is unavailable")
        results.append(Trajectory(
            dataset_id="visa_1cls", category=category, condition="P0",
            source_commit=str(index["source_commit"]), attempt_root=attempt_root,
            private_feedback_path=attempt_root / "private_feedback.jsonl",
            plan_path=plan_path, plan_sha256=plan_file_sha,
            bundle={"identity_manifest_path": str(VISA_REG_ROOT / "private/identity_manifest.private.json"),
                    "identity_manifest_sha256": index["identity_manifest_sha256"],
                    "locators": event_locators, "dataset_root": dataset_root,
                    "stage_b_authorization_path": str(stage_b_auth_path),
                    "stage_b_authorization": stage_b_auth,
                    "stage_b_authorization_sha256": sha256_object(stage_b_auth),
                    "stage_b_authorization_file_sha256": sha256_file(stage_b_auth_path),
                    "candidate": candidate},
            identities=identity, role_manifest=role, stream_manifest=stream,
            release_identity=release, records=records, stream_seed=0, support_seed=0,
            registration={"unit_candidate_sha256": candidate_sha,
                "unit_approval_sha256": approval_sha,
                "dataset_release_sha256": index["dataset_release_sha256"],
                "identity_manifest_sha256": index["identity_manifest_sha256"],
                "source_csv_sha256": stage_b_auth["source_csv_sha256"],
                "registration_index_sha256": sha256_file(VISA_REG_ROOT / "registration_index.json"),
                "role_manifest_sha256": entry["role_manifest_sha256"],
                "stream_manifest_sha256": entry["stream_manifest_sha256"]},
            original_stage_b=original_stage_b))
        results[-1].input_audit = {
            "dataset_id": "visa_1cls", "category": category, "condition": "P0",
            "source_commit": index["source_commit"], "registration_index_sha256": sha256_file(VISA_REG_ROOT / "registration_index.json"),
            "identity_manifest_sha256": index["identity_manifest_sha256"],
            "dataset_release_sha256": index["dataset_release_sha256"],
            "role_manifest_sha256": entry["role_manifest_sha256"],
            "stream_manifest_sha256": entry["stream_manifest_sha256"],
            "unit_candidate_sha256": candidate_sha, "unit_approval_sha256": approval_sha,
            "stage_b_authorization_sha256": sha256_file(stage_b_auth_path),
            "original_stage_b_result_sha256": sha256_file(original_stage_b_path),
            "attempt_root_relative": str(attempt_root.relative_to(ROOT)),
            "original_stage_b_record_identity_matches": True,
        }
    return results


def discover_trajectories() -> list[Trajectory]:
    real = _real_trajectories()
    visa = _visa_trajectories()
    trajectories = real + visa
    if (len(real) != 90 or len(visa) != 12 or len(trajectories) != 102 or
            sum(t.event_count for t in real) != 343755 or
            sum(t.event_count for t in visa) != 2162):
        raise SupplementaryError("registered supplementary scope does not equal 90 Real-IAD + 12 VisA trajectories")
    # The two mandated engineering-smoke examples run first, before expansion.
    audio = next(t for t in trajectories if (t.dataset_id, t.category, t.condition) ==
                 ("realiad", "audiojack", "P0"))
    cashew = next(t for t in trajectories if (t.dataset_id, t.category, t.condition) ==
                  ("visa_1cls", "cashew", "P0"))
    ordered = [audio, cashew] + [t for t in trajectories if t not in (audio, cashew)]
    if len({t.key for t in ordered}) != 102:
        raise SupplementaryError("supplementary trajectory identities are not unique")
    return ordered


def preflight_assets(trajectories: Sequence[Trajectory]) -> dict[str, Any]:
    summaries = []
    for index, trajectory in enumerate(trajectories, 1):
        trajectory.input_audit = trajectory.input_audit or {}
        seal, rows, group_seals = _attempt_seal_and_rows(trajectory)
        original_metrics = trajectory.original_stage_b
        if trajectory.dataset_id == "realiad":
            auth_rows = read_json(REAL_STAGE_B_ROOT / "stage_b_authorization.json")["authorizations"]
            auth = [a for a in auth_rows if a.get("category") == trajectory.category and
                    a.get("condition") == trajectory.condition]
            if (len(auth) != 1 or auth[0].get("source_commit") != trajectory.source_commit or
                    auth[0].get("stage_a_category_seal_sha256") != sha256_file(
                        trajectory.attempt_root / "stage_a_category_seal.json")):
                # Stage B binds the parsed seal object digest as `durable_seal_sha256`
                # in the status, which is also the file SHA checked here.
                raise SupplementaryError(f"{trajectory.key}: combined Stage B authorization does not bind its seal")
            if (original_metrics.get("source_commit") != trajectory.source_commit or
                    original_metrics.get("annotation_sha256") != trajectory.registration["annotation_sha256"]):
                raise SupplementaryError(f"{trajectory.key}: original Stage B result source identity mismatch")
        query_counts = {arm: sum(bool(row["query_actions"][arm].get("executed")) for row in rows)
                        for arm in PRINCIPAL_ARMS}
        summaries.append({**trajectory.input_audit,
            "original_seal_status": seal.get("status"),
            "original_score_rows_sha256": seal["score_rows_sha256"],
            "original_group_seals_sha256": seal["group_seals_sha256"],
            "original_stage_a_group_count": len(group_seals),
            "stage_a_query_count_by_arm": query_counts,
            "original_stage_b_available": bool(original_metrics),
            "registered_event_count": trajectory.event_count})
        if index % 10 == 0 or index == len(trajectories):
            print(f"[asset-preflight] {index}/{len(trajectories)} sealed inputs verified", flush=True)
    return {
        "schema": "nshadow-b3-c64.asset-and-replay-audit.v1",
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "current_git_head": _git_head(),
        "historical_source_commits_are_provenance_only": True,
        "realiad_trajectory_count": 90, "realiad_image_event_count": 343755,
        "visa_trajectory_count": 12, "visa_image_event_count": 2162,
        "total_trajectory_count": 102, "total_image_event_count": 345917,
        "real_iad_combined_stage_b_authorization_sha256": sha256_file(
            REAL_STAGE_B_ROOT / "stage_b_authorization.json"),
        "real_iad_stage_b_metrics_sha256": sha256_file(REAL_STAGE_B_ROOT / "stage_b_metrics_90.json"),
        "real_iad_campaign_plan_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in REAL_PLANS},
        "visa_registration_index_sha256": sha256_file(VISA_REG_ROOT / "registration_index.json"),
        "trajectories": summaries,
        "test_outcomes_or_masks_read": False,
        "historical_artifacts_modified": False,
    }


def _git_head() -> str:
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
        check=True, capture_output=True, text=True).stdout.strip()


class ResourceSampler:
    def __init__(self, interval_seconds: float = 1.0):
        self.interval_seconds = float(interval_seconds)
        self.peak_rss_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import psutil
        process = psutil.Process()
        def sample() -> None:
            while not self._stop.is_set():
                try:
                    rss = int(process.memory_info().rss)
                    self.peak_rss_bytes = max(self.peak_rss_bytes or 0, rss)
                except Exception:
                    pass
                self._stop.wait(self.interval_seconds)
        self._thread = threading.Thread(target=sample, name="nshadow-rss-sampler", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


class ReplayEngine:
    """Minimal exact F-B/Greedy + normal-only C1 reconstruction for one leaf."""

    def __init__(self, trajectory: Trajectory, encoder: Any, commissioning: Mapping[str, Any]):
        import torch
        from types import SimpleNamespace
        from dataclasses import asdict
        from v02_memory import MemoryArm
        from experiments.calibration import r1_matched_correction_pilot as r1
        from experiments.calibration import r2_exp2a_c1_strength as exp2a
        from experiments.calibration import r3_fullstream_2x2 as r3
        from experiments.calibration import r6_capacity1024_audiojack_runner as r6
        from experiments.calibration import r7_shuffled_correction_4class as r7
        from experiments.calibration import r9_correction_mechanism_falsification as r9
        from experiments.calibration import profiles
        from experiments.calibration.controllers import StateConsistentQuantile
        from experiments.calibration.native import _fixed_bank_digest
        from experiments.risk_safe.native_bridge import dynamic_codebooks

        inherited_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
        encoder_impl, p0 = encoder.impl, encoder.p0
        r3._restore_cuda_visibility(inherited_visibility)
        cfg, _profile = r3._native_config(p0, profiles, r1)
        if asdict(cfg) != asdict(encoder.cfg):
            raise SupplementaryError("loaded native encoder configuration differs from the frozen profile")
        self.impl, self.p0, self.r1, self.exp2a = encoder_impl, p0, r1, exp2a
        self.r3, self.r6, self.r7, self.r9 = r3, r6, r7, r9
        self.cfg, self.encoder, self.trajectory = cfg, encoder, trajectory
        self.c64_items = tuple(commissioning["c64_items"])
        self.c64_group_sha256 = sha256_object([
            str(item["source_group_sha256"]) for item in self.c64_items])
        if len(self.c64_items) != 64:
            raise SupplementaryError(f"{trajectory.key}: expected exactly 64 registered C64 images")

        self.impl.set_seed(0)
        self.bank = self.p0.build_vmb(
            self.impl, commissioning["support_cache"],
            list(commissioning["support_paths"]), self.cfg)
        self.r3._validate_correction_config(self.bank)
        self.fb_cfg, codebooks = self.r6._configure_r6_correction_capacity(self.bank)
        if len(codebooks) != 1 or int(codebooks[0].size()) != 0:
            raise SupplementaryError("F-B must begin with one empty native K=1024 Correction bank")
        self.fb_cb = codebooks[0]
        self.fb_sources: dict[int, set[str]] = {}
        self.writer = MemoryArm("greedy_coreset", self.bank,
            SimpleNamespace(flat=1024, stm=1024, ltm=0, buffer=0, min_events=2,
                            promote_every=16, cap_bytes=2 * 1024**3), seed=0)
        if any(int(cb.K) != 1024 or cb.size() != 0 for cb in dynamic_codebooks(self.bank)):
            raise SupplementaryError("shared Greedy normal Memory must start empty at K=1024")
        self.bank.use_defect_bank = False
        self.bank.corr_cfg.enabled = False
        self.fixed_sha = _fixed_bank_digest(self.bank)
        self.base_c64, self.base_cache, _base_seconds = self.r7._score_c64_base(
            self.impl, self.bank, self.c64_items, self.cfg, self.r1, trajectory.category)
        if (len(self.base_c64) != 64 or not np.isfinite(np.asarray(self.base_c64)).all() or
                any(tuple(cache[0].shape) != (784,) for cache in self.base_cache)):
            raise SupplementaryError("initial native C64 base scores or dense-map cache are invalid")
        theta0 = float(np.quantile(np.asarray(self.base_c64, dtype=np.float64), .90,
                                   method="higher"))
        self.theta0 = theta0
        self.initial_theta = theta0
        self.sigma0 = max(float(np.std(np.asarray(self.base_c64, dtype=np.float64), ddof=1)), .02)

        self.shadow_cfg = type(self.fb_cfg)(**asdict(self.fb_cfg))
        self.shadow_cfg.enabled = False
        self.shadow_cb = type(self.fb_cb)(1024, int(self.fb_cb.protos.shape[1]),
                                          self.fb_cb.device, self.shadow_cfg)
        self.shadow_sources: dict[int, set[str]] = {}
        self.shadow_version = 0
        self.controller = StateConsistentQuantile(theta0, delta=.01, min_normals=16,
                                                   name="N-shadow-independent-B3")
        self.shadow_sha = self._shadow_digest()
        self.c64_scores, _elapsed = self._score_current_c64(event_position=0)
        if (len(self.c64_scores) != 64 or not np.isfinite(np.asarray(self.c64_scores)).all() or
                not np.allclose(self.c64_scores, self.base_c64, rtol=0.0, atol=1e-7)):
            raise SupplementaryError("empty N-shadow C1 does not reproduce the native initial C64 vector")
        initial = self.r7._initialize_b3(
            self.controller, self.c64_scores, self.c64_group_sha256,
            self.shadow_sha, theta0)
        if (self.controller.evidence_version != 0 or self.controller.theta != theta0 or
                initial.get("status") != "CURRENT_C64_Q90"):
            raise SupplementaryError("independent B3 initial C64 evidence failed the source contract")
        self.b3_refresh_count = 0
        self.c64_encoding_seconds = float(_base_seconds)
        self.c64_scoring_seconds = float(_elapsed)
        self.max_abs_fb_parity = 0.0
        self.max_abs_shadow_parity = 0.0
        self.query_parity_count = 0
        self.transition_count = 0
        self.threshold_movements: list[float] = []
        self.initial_state_sha256 = self.shadow_sha
        self.fixed_support_sha256 = self.fixed_sha

    def _shadow_digest(self) -> str:
        return self.r9._state_digest_for_codebook(
            self.bank, self.fixed_sha, self.shadow_sources, self.shadow_cfg,
            self.shadow_cb, self.exp2a, self.r1)

    def shadow_state_digest(self) -> str:
        """Expose the same digest interface used by the historical backend."""
        return self._shadow_digest()

    def _fb_digest(self) -> str:
        return self.r9._state_digest_for_codebook(
            self.bank, self.fixed_sha, self.fb_sources, self.fb_cfg,
            self.fb_cb, self.exp2a, self.r1)

    def _score_current_c64(self, *, event_position: int) -> tuple[list[float], float]:
        values, elapsed = self.r9._score_c64_for_codebook(
            self.impl, self.bank, self.r7, self.r1, self.exp2a,
            self.c64_items, self.base_cache, self.cfg, self.shadow_cfg,
            self.shadow_cb, self.shadow_sources, self.trajectory.category,
            int(event_position))
        return [float(value) for value in values], float(elapsed)

    def refresh_if_changed(self, *, previous_digest: str, position: int,
                           greedy_changed: bool) -> dict[str, Any] | None:
        current = self._shadow_digest()
        if current == previous_digest:
            return None
        if self.shadow_sha != previous_digest:
            raise SupplementaryError("N-shadow transition previous digest is stale")
        if greedy_changed:
            self.base_c64, self.base_cache, base_elapsed = self.r7._score_c64_base(
                self.impl, self.bank, self.c64_items, self.cfg, self.r1,
                self.trajectory.category)
            self.c64_encoding_seconds += float(base_elapsed)
        c64, elapsed = self._score_current_c64(event_position=position + 1)
        if len(c64) != 64 or not np.isfinite(np.asarray(c64)).all():
            raise SupplementaryError("current-state N-shadow C64 vector is invalid")
        transition = self.r7._refresh_b3(
            self.controller, c64, self.c64_group_sha256, previous_digest,
            current, self.controller.evidence_version)
        if transition is None:
            raise SupplementaryError("changed N-shadow state did not produce a registered B3 refresh")
        self.shadow_version += 1
        self.b3_refresh_count += 1
        self.shadow_sha = current
        self.c64_scores = c64
        self.c64_scoring_seconds += elapsed
        self.threshold_movements.append(float(transition["movement"]))
        self.transition_count += 1
        transition_record = {
            "position_after_group": int(position),
            "source_state_version_before": self.shadow_version - 1,
            "source_state_version_after": self.shadow_version,
            "previous_source_state_sha256": previous_digest,
            "source_state_sha256": current,
            "b3_evidence_version": int(self.controller.evidence_version),
            "c64_score_vector_sha256": transition["c64_score_vector_sha256"],
            "old_theta": float(transition["old_theta"]),
            "raw_c64_q90": float(transition["raw_c64_q90"]),
            "new_theta": float(transition["new_theta"]),
            "movement": float(transition["movement"]),
            "greedy_cache_rebuilt": bool(greedy_changed),
        }
        return transition_record

    def score_query_event(self, event: Any) -> dict[str, Any]:
        import torch
        features = [value.to(self.bank.device) for value in event.feature_ref.feature_layers]
        if (len(features) != 1 or tuple(features[0].shape) != (784, 4096) or
                features[0].dtype != torch.float32 or not bool(torch.isfinite(features[0]).all().item())):
            raise SupplementaryError("query-event feature differs from native (784,4096) float32 contract")
        base_map, top_ids, c0_score, summary = self.r1._native_dense_score(
            self.impl, self.bank, event.feature_ref.image, features, self.cfg)
        top_ids = np.asarray(top_ids, dtype=np.int64)
        if (top_ids.shape != (32,) or not np.array_equal(
                top_ids, np.asarray(summary["layer0"]["topk_patch_idx"], dtype=np.int64))):
            raise SupplementaryError("native query-event C0 Top-32 identity differs from its receipt")
        selected = self.impl.l2_normalize(features[0])[torch.as_tensor(
            top_ids, device=self.bank.device, dtype=torch.long)]
        selected_np = selected.detach().float().cpu().numpy()
        fb_delta, _ = self.exp2a._score_arm(
            self.impl, self.bank, "F-B", self.fb_cfg, self.fb_cb,
            self.fb_sources, selected, event.group_sha256, event.position)
        fb_score = float(self.r1.max_score(
            self.r1.corrected_dense_map(base_map, top_ids, fb_delta)))
        shadow_delta, _ = self.exp2a._score_arm(
            self.impl, self.bank, "N-SHADOW", self.shadow_cfg, self.shadow_cb,
            self.shadow_sources, selected, event.group_sha256, event.position)
        shadow_score = float(self.r1.max_score(
            self.r1.corrected_dense_map(base_map, top_ids, shadow_delta)))
        return {"event": event, "features": features, "top_ids": top_ids,
            "selected": selected, "selected_np": selected_np,
            "base_map": base_map, "c0_score": float(c0_score),
            "fb_score": fb_score, "shadow_score": shadow_score,
            "state_sha256": self._shadow_digest()}

    def apply_f_b_feedback(self, context: Mapping[str, Any], label: int,
                           *, query_position: int) -> list[dict[str, Any]]:
        if label not in (0, 1) or isinstance(label, bool):
            raise SupplementaryError("purchased F-B feedback must be binary")
        event = context["event"]
        position = int(event.position)
        if position != int(query_position):
            raise SupplementaryError("F-B purchased label is misaligned to its event position")
        transitions: list[dict[str, Any]] = []
        shadow_before = self._shadow_digest()
        fb_before = self._fb_digest()
        greedy_before = self.r1._memory_content_sha256(self.bank)
        c0_score = float(context["c0_score"])
        c0_prediction = int(c0_score >= self.theta0)
        c0_outcome = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[
            (int(label), c0_prediction)]

        if label == 0:
            self.exp2a._capture_update(
                self.impl, self.bank, self.r1, "N-SHADOW", self.shadow_cfg,
                self.shadow_cb, context["features"], context["top_ids"], -1,
                c0_score, self.theta0,
                "FP" if c0_score >= self.theta0 else "TN", position,
                self.shadow_sources, event.group_sha256)
        after_shadow_c1 = self._shadow_digest()
        if label == 1 and after_shadow_c1 != shadow_before:
            raise SupplementaryError("N-shadow accepted a verified defect into normal-only C1")
        if after_shadow_c1 != shadow_before:
            self.shadow_sha = shadow_before
            transition = self.refresh_if_changed(
                previous_digest=shadow_before, position=position, greedy_changed=False)
            if transition is None:
                raise SupplementaryError("normal-only C1 state transition was not calibrated")
            transitions.append(transition)

        fb_after_shadow = self._fb_digest()
        if fb_after_shadow != fb_before:
            raise SupplementaryError("N-shadow correction update changed the F-B correction trajectory")
        self.exp2a._capture_update(
            self.impl, self.bank, self.r1, "F-B", self.fb_cfg, self.fb_cb,
            context["features"], context["top_ids"], 1 if label else -1,
            c0_score, self.theta0, c0_outcome, position, self.fb_sources,
            event.group_sha256)

        def update(**kwargs: Any) -> dict[str, Any]:
            self.writer.update(self.impl, list(kwargs["feature_layers"]),
                {0: kwargs["candidate_ids"].tolist()}, kwargs["position"],
                score=kwargs["score"], theta=kwargs["theta"], sigma=kwargs["sigma"])
            return {}

        writer_receipt = self.r1.common_c0_greedy_writer_action(
            position=position, c0_score=c0_score, theta=self.theta0,
            sigma=self.sigma0, c0_prediction=c0_prediction,
            candidate_ids=context["top_ids"], candidate_features=context["selected_np"],
            feature_layers=context["features"], queried=True, disclosed_label=int(label),
            eligibility=self.impl.select_informative_normal_write, update=update)
        greedy_after = self.r1._memory_content_sha256(self.bank)
        greedy_changed = greedy_after != greedy_before
        if label == 1 and greedy_changed:
            raise SupplementaryError("verified defect changed shared Greedy normal memory")
        # The F-B signed correction is intentionally excluded from N-shadow's
        # digest. Only a Greedy change can cause a second shadow transition.
        after_all = self._shadow_digest()
        if after_all != self.shadow_sha:
            previous = self.shadow_sha
            transition = self.refresh_if_changed(previous_digest=previous,
                position=position, greedy_changed=greedy_changed)
            if transition is None:
                raise SupplementaryError("shared-memory state change lacked its B3 refresh")
            transitions.append(transition)
        elif label == 1 and after_all != shadow_before:
            raise SupplementaryError("defect feedback changed the N-shadow scoring state")
        if self._fb_digest() == "":
            raise SupplementaryError("F-B detector digest unexpectedly became empty")
        return transitions


def _registered_event(trajectory: Trajectory, encoder: Any, record: Mapping[str, Any]) -> Any:
    image_id = str(record["image_identity_sha256"])
    locator = trajectory.bundle.get("locators", {}).get(image_id) if trajectory.dataset_id == "visa_1cls" else \
              trajectory.bundle["test_private_locators"].get(image_id)
    if not isinstance(locator, Mapping):
        raise SupplementaryError(f"{trajectory.key}: registered query-event locator is missing")
    path = Path(str(locator.get("image_path", ""))).resolve(strict=True)
    allowed_root = REAL_DATA_ROOT if trajectory.dataset_id == "realiad" else VISA_DATA_ROOT
    try:
        path.relative_to(allowed_root.resolve(strict=True))
    except ValueError as exc:
        raise SupplementaryError(f"{trajectory.key}: image locator escapes the researcher-specified dataset root") from exc
    if trajectory.dataset_id == "realiad":
        return encoder.encode_registered_realiad_event(
            record=record, locator=locator, category=trajectory.category,
            dataset_release_sha256=str(trajectory.release_identity["dataset_release_sha256"]),
            stream_seed=trajectory.stream_seed,
            annotation_sha256=str(trajectory.registration["annotation_sha256"]),
            identity_manifest_sha256=str(trajectory.registration.get("identity_manifest_sha256",
                trajectory.identities["identity_manifest_sha256"])))
    return encoder.encode_registered_visa_event(
        record=record, locator=locator, category=trajectory.category,
        dataset_release_sha256=str(trajectory.release_identity["dataset_release_sha256"]))


def _query_receipt(trajectory: Trajectory, row: Mapping[str, Any]) -> dict[str, Any]:
    common = {"arm_id": "F-B", "category": trajectory.category,
        "condition_id": trajectory.condition,
        "event_id_sha256": str(row["event_id_sha256"]),
        "position": int(row["position"])}
    if trajectory.dataset_id == "realiad":
        return {"schema": "final-benchmark.query-receipt.v1", **common,
            "source_group_sha256": str(row["source_group_sha256"])}
    if trajectory.dataset_id == "visa_1cls":
        return {"schema": "final-benchmark.query-receipt.atomic-unit.v1", **common,
            "atomic_unit_sha256": str(row["atomic_unit_sha256"]),
            "atomic_unit_kind": str(row["atomic_unit_kind"])}
    raise SupplementaryError(f"unsupported query receipt dataset: {trajectory.dataset_id}")


def _purchased_feedback_matches(trajectory: Trajectory, arm: str,
                                source_row: Mapping[str, Any],
                                action: Mapping[str, Any],
                                item: Mapping[str, Any]) -> bool:
    """Validate the native private-ledger schema without changing its label."""
    expected_schema = ("final-benchmark.private-feedback.v1"
        if trajectory.dataset_id == "realiad" else
        "final-benchmark.visa-private-feedback.v1"
        if trajectory.dataset_id == "visa_1cls" else None)
    common_match = (
        expected_schema is not None and item.get("schema") == expected_schema and
        item.get("arm_id") == arm and
        item.get("position") == int(source_row["position"]) and
        item.get("event_id_sha256") == source_row["event_id_sha256"] and
        item.get("query_receipt_sha256") == action.get("query_receipt_sha256") and
        isinstance(item.get("label"), int) and not isinstance(item.get("label"), bool) and
        item.get("label") in (0, 1))
    if not common_match:
        return False
    if trajectory.dataset_id == "realiad":
        # The Real-IAD private ledger binds event identity through the receipt
        # hash; source_group_sha256 is in that public query receipt, not duplicated
        # in the private feedback row.
        return True
    return (item.get("atomic_unit_sha256") == source_row.get("atomic_unit_sha256") and
            item.get("image_identity_sha256") == source_row.get("image_identity_sha256"))


def _atomic_unit_prediction_rows(local_rows: Sequence[Mapping[str, Any]], *,
                                 threshold: float, state_sha256: str,
                                 state_version: int, b3_evidence_version: int
                                 ) -> list[dict[str, Any]]:
    """Build a label-blind atomic unit under one pre-unit state and threshold."""
    if (not local_rows or not math.isfinite(float(threshold)) or
            not isinstance(state_sha256, str) or len(state_sha256) != 64 or
            state_version < 0 or b3_evidence_version < 0):
        raise SupplementaryError("atomic unit lacks a valid pre-unit scorer/controller identity")
    output: list[dict[str, Any]] = []
    for row in local_rows:
        score = float(row["n_shadow_score"])
        if not math.isfinite(score):
            raise SupplementaryError("atomic unit contains a nonfinite inherited N-shadow score")
        output.append({
            "position": int(row["position"]),
            "event_id_sha256": str(row["event_id_sha256"]),
            "image_identity_sha256": str(row["image_identity_sha256"]),
            "source_group_sha256": str(row["source_group_sha256"]),
            "camera_id": row.get("camera_id"),
            "atomic_unit_sha256": row.get("atomic_unit_sha256"),
            "atomic_unit_kind": row.get("atomic_unit_kind"),
            "n_shadow_score": score,
            "independent_b3_threshold": float(threshold),
            "prediction": int(score >= threshold),
            "source_state_version": int(state_version),
            "source_state_sha256": state_sha256,
            "b3_evidence_version": int(b3_evidence_version),
            "inherited_f_b_query_executed": bool(
                row["query_actions"]["F-B"].get("executed")),
        })
    return output


def _trajectory_output_directory(root: Path, trajectory: Trajectory) -> Path:
    if trajectory.dataset_id == "realiad":
        return root / "stage_a" / "realiad" / trajectory.category / trajectory.condition
    return root / "stage_a" / "visa_1cls" / trajectory.category / trajectory.condition


def _run_stage_a_trajectory(trajectory: Trajectory, encoder: Any,
                            commissioning: Mapping[str, Any], output_root: Path,
                            campaign_progress: Path) -> dict[str, Any]:
    started = time.perf_counter()
    original_seal, original_rows, original_groups = _attempt_seal_and_rows(trajectory)
    output_dir = _trajectory_output_directory(output_root, trajectory)
    output_dir.mkdir(parents=True, exist_ok=False)
    journal_path = output_dir / "shadow_b3_prediction_journal.jsonl"
    transition_path = output_dir / "shadow_c64_b3_transitions.jsonl"
    private_path = output_dir / "private_feedback_transition_receipts.jsonl"
    receipt_path = output_dir / "stage_a_completion_receipt.json"
    parity_path = output_dir / "replay_parity.json"
    status_path = output_dir / "attempt_status.json"
    status = {"schema": "nshadow-independent-b3-c64.attempt-status.v1",
        "dataset_id": trajectory.dataset_id, "category": trajectory.category,
        "condition": trajectory.condition, "status": "RUNNING",
        "expected_event_count": trajectory.event_count, "completed_event_count": 0,
        "original_source_commit": trajectory.source_commit,
        "started_unix": time.time(), "stage_a_phase": "INITIALIZING_NATIVE_REPLAY"}
    write_exclusive(status_path, status)
    engine: ReplayEngine | None = None
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        engine = ReplayEngine(trajectory, encoder, commissioning)
        initial_sha = engine.shadow_state_digest()
        initial_version = engine.shadow_version
        if (original_rows[0].get("n_shadow_state_sha256") != initial_sha or
                int(original_rows[0].get("n_shadow_state_version", -1)) != initial_version):
            raise SupplementaryError(f"{trajectory.key}: reconstructed initial N-shadow identity differs from sealed journal")
        expected_feedback_lines = 0
        fb_query_count = 0
        max_fb_parity = max_shadow_parity = 0.0
        query_parity_count = 0
        query_encoding_seconds = 0.0
        group_count = 0
        event_cursor = 0
        c64_start_seconds = engine.c64_encoding_seconds + engine.c64_scoring_seconds
        with trajectory.private_feedback_path.open("rb") as feedback_stream:
            for unit_index, original_group in enumerate(original_groups):
                start = int(original_group["start_position"])
                stop = int(original_group["end_position_exclusive"])
                local_rows = original_rows[start:stop]
                if start != event_cursor or not local_rows:
                    raise SupplementaryError(f"{trajectory.key}: original atomic-unit cursor is noncontiguous")
                current_digest = engine.shadow_state_digest()
                current_version = engine.shadow_version
                if any(row.get("n_shadow_state_sha256") != current_digest or
                       int(row.get("n_shadow_state_version", -1)) != current_version
                       for row in local_rows):
                    raise SupplementaryError(f"{trajectory.key}:{start}: reconstructed N-shadow state/version mismatch")
                f_b_queries = [row for row in local_rows
                               if bool(row["query_actions"]["F-B"].get("executed"))]
                if len(f_b_queries) > 1:
                    raise SupplementaryError(f"{trajectory.key}:{start}: original F-B has multiple queries in one atomic unit")
                query_context: dict[int, dict[str, Any]] = {}
                if f_b_queries:
                    query_row = f_b_queries[0]
                    position = int(query_row["position"])
                    event = _registered_event(trajectory, encoder, trajectory.records[position])
                    query_encoding_seconds += float(event.encoding_seconds or 0.0)
                    context = engine.score_query_event(event)
                    fb_error = abs(float(context["fb_score"]) - float(query_row["arm_scores"]["F-B"]))
                    shadow_error = abs(float(context["shadow_score"]) - float(query_row["n_shadow_score"]))
                    max_fb_parity = max(max_fb_parity, fb_error)
                    max_shadow_parity = max(max_shadow_parity, shadow_error)
                    if fb_error > STATE_REPLAY_TOLERANCE or shadow_error > STATE_REPLAY_TOLERANCE:
                        raise SupplementaryError(
                            f"{trajectory.key}:{position}: native query score replay mismatch "
                            f"(F-B={fb_error:.9g}, N-shadow={shadow_error:.9g})")
                    action = query_row["query_actions"]["F-B"]
                    receipt = _query_receipt(trajectory, query_row)
                    if sha256_object(receipt) != action.get("query_receipt_sha256"):
                        raise SupplementaryError(f"{trajectory.key}:{position}: F-B query receipt identity mismatch")
                    query_context[position] = context
                    query_parity_count += 1

                threshold = float(engine.controller.theta)
                source_digest = current_digest
                source_version = current_version
                b3_version = int(engine.controller.evidence_version)
                sidecar_rows = _atomic_unit_prediction_rows(
                    local_rows, threshold=threshold, state_sha256=source_digest,
                    state_version=source_version, b3_evidence_version=b3_version)
                group_record = {
                    "schema": "nshadow-independent-b3-c64.atomic-unit.v1",
                    "atomic_unit_index": unit_index,
                    "source_group_sha256": str(original_group["source_group_sha256"]),
                    "start_position": start, "end_position_exclusive": stop,
                    "event_order_sha256": sha256_object(
                        [row["event_id_sha256"] for row in sidecar_rows]),
                    "source_state_sha256": source_digest,
                    "source_state_version": source_version,
                    "b3_evidence_version": b3_version,
                    "score_rows_sha256": sha256_object(sidecar_rows),
                    "status": "SEALED_BEFORE_INHERITED_FEEDBACK_DISCLOSURE",
                    "score_rows": sidecar_rows,
                }
                append_fsync(journal_path, group_record)
                # Only now may the private original query ledger be advanced.
                expected_disclosures: list[tuple[str, dict[str, Any]]] = []
                for arm in INHERITED_FEEDBACK_ORDER:
                    for row in local_rows:
                        if bool(row["query_actions"][arm].get("executed")):
                            expected_disclosures.append((arm, row))
                expected_feedback_lines += len(expected_disclosures)
                for arm, source_row in expected_disclosures:
                    raw = feedback_stream.readline()
                    if not raw:
                        raise SupplementaryError(f"{trajectory.key}: inherited private feedback ledger ended early")
                    try:
                        item = json.loads(raw)
                    except Exception as exc:
                        raise SupplementaryError(f"{trajectory.key}: malformed purchased feedback record") from exc
                    action = source_row["query_actions"][arm]
                    if not _purchased_feedback_matches(trajectory, arm, source_row, action, item):
                        raise SupplementaryError(f"{trajectory.key}:{source_row['position']}: purchased feedback receipt mismatch")
                    if arm != "F-B":
                        continue
                    position = int(source_row["position"])
                    context = query_context.get(position)
                    if context is None:
                        raise SupplementaryError(f"{trajectory.key}:{position}: F-B query lacks its sealed score context")
                    label = int(item["label"])
                    old_transitions = engine.transition_count
                    transitions = engine.apply_f_b_feedback(context, label, query_position=position)
                    for transition in transitions:
                        append_fsync(transition_path, transition)
                    private_receipt = {
                        "schema": "nshadow-independent-b3-c64.private-feedback-transition.v1",
                        "position": position, "event_id_sha256": str(source_row["event_id_sha256"]),
                        "query_receipt_sha256": str(action["query_receipt_sha256"]),
                        "purchased_label": label,
                        "original_feedback_line_sha256": sha256_bytes(raw),
                        "pre_state_sha256": str(context["state_sha256"]),
                        "post_state_sha256": engine.shadow_state_digest(),
                        "state_transition_count": engine.transition_count - old_transitions,
                    }
                    append_fsync(private_path, private_receipt)
                    fb_query_count += 1
                    del context
                group_count += 1
                event_cursor = stop
                status.update(completed_event_count=event_cursor,
                              b3_refresh_count=engine.b3_refresh_count,
                              stage_a_phase="REPLAYING_SEALED_SOURCE_GROUPS")
                if group_count % 25 == 0 or event_cursor == trajectory.event_count:
                    write_progress(status_path, status)
                if group_count % 50 == 0 or event_cursor == trajectory.event_count:
                    append_fsync(campaign_progress, {
                        "event": "TRAJECTORY_PROGRESS", "trajectory": trajectory.key,
                        "completed_trajectory_count": None, "completed_event_count": event_cursor,
                        "expected_event_count": trajectory.event_count,
                        "completed_atomic_unit_count": group_count,
                        "expected_atomic_unit_count": len(original_groups),
                        "b3_refresh_count": engine.b3_refresh_count,
                        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    })
                if group_count % 128 == 0:
                    print(f"[{trajectory.key}] {event_cursor}/{trajectory.event_count} events; "
                          f"{engine.b3_refresh_count} independent B3 refreshes", flush=True)
            if feedback_stream.read(1) != b"":
                raise SupplementaryError(f"{trajectory.key}: unconsumed purchased-feedback rows remain")
        if (event_cursor != trajectory.event_count or group_count != len(original_groups) or
                expected_feedback_lines != sum(int(value) for value in (
                    trajectory.input_audit or {}).get("executed_query_count_by_arm", {}).values()) or
                fb_query_count != sum(bool(row["query_actions"]["F-B"].get("executed"))
                                      for row in original_rows)):
            raise SupplementaryError(f"{trajectory.key}: replay coverage or query accounting is incomplete")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak_gpu_allocated = int(torch.cuda.max_memory_allocated())
        peak_gpu_reserved = int(torch.cuda.max_memory_reserved())
        try:
            import psutil
            peak_host_rss = int(psutil.Process().memory_info().rss)
        except Exception:
            peak_host_rss = None
        journal_sha = sha256_file(journal_path)
        transition_sha = sha256_file(transition_path) if transition_path.exists() else sha256_bytes(b"")
        private_feedback_sha = sha256_file(private_path) if private_path.exists() else sha256_bytes(b"")
        seal_payload = {
            "schema": "nshadow-independent-b3-c64.stage-a-seal.v1",
            "status": "SEALED_COMPLETE", "dataset_id": trajectory.dataset_id,
            "category": trajectory.category, "condition": trajectory.condition,
            "expected_event_count": trajectory.event_count,
            "completed_event_count": event_cursor,
            "atomic_unit_count": group_count,
            "inherited_f_b_query_count": fb_query_count,
            "original_source_commit": trajectory.source_commit,
            "original_stage_a_seal_sha256": sha256_file(
                trajectory.attempt_root / "stage_a_category_seal.json"),
            "original_stage_a_journal_sha256": sha256_file(
                trajectory.attempt_root / "stage_a_journal.jsonl"),
            "prediction_journal_sha256": journal_sha,
            "c64_b3_transitions_sha256": transition_sha,
            "private_feedback_transition_receipts_sha256": private_feedback_sha,
            "initial_n_shadow_state_sha256": engine.initial_state_sha256,
            "final_n_shadow_state_sha256": engine.shadow_state_digest(),
            "independent_b3_refresh_count": engine.b3_refresh_count,
            "initial_b3_threshold": float(engine.initial_theta),
            "final_b3_threshold": float(engine.controller.theta),
            "outcome_access_phase": "NO_UNQUERIED_OUTCOME_ACCESS_STAGE_A",
        }
        write_exclusive(receipt_path, seal_payload)
        parity = {
            "schema": "nshadow-independent-b3-c64.replay-parity.v1",
            "trajectory": trajectory.key,
            "status": "PASS",
            "original_source_state_identity_checks": event_cursor,
            "original_source_state_digest_exact_matches": event_cursor,
            "original_source_state_version_exact_matches": event_cursor,
            "query_native_score_parity_count": query_parity_count,
            "maximum_abs_f_b_score_error": max_fb_parity,
            "maximum_abs_n_shadow_score_error": max_shadow_parity,
            "declared_float_tolerance": STATE_REPLAY_TOLERANCE,
            "inherited_f_b_query_identity_count": fb_query_count,
            "c64_initial_vector_count": 64,
            "independent_b3_refresh_count": engine.b3_refresh_count,
            "normal_only_shadow_state_version": engine.shadow_version,
            "original_stage_a_rows_and_seals_unchanged": True,
            "test_outcomes_or_masks_read": False,
            "private_feedback_read_only_after_sidecar_group_fsync": True,
        }
        write_exclusive(parity_path, parity)
        status.update(status="SEALED_COMPLETE", completed_event_count=event_cursor,
            completion_receipt_sha256=sha256_file(receipt_path),
            replay_parity_sha256=sha256_file(parity_path),
            prediction_journal_sha256=journal_sha,
            c64_b3_transitions_sha256=transition_sha,
            private_feedback_transition_receipts_sha256=private_feedback_sha,
            independent_b3_refresh_count=engine.b3_refresh_count,
            query_score_parity_count=query_parity_count,
            max_f_b_score_error=max_fb_parity,
            max_n_shadow_score_error=max_shadow_parity,
            query_event_encoding_seconds=query_encoding_seconds,
            runtime_seconds=elapsed, peak_gpu_allocated_bytes=peak_gpu_allocated,
            peak_gpu_reserved_bytes=peak_gpu_reserved, process_rss_bytes=peak_host_rss,
            c64_scoring_seconds=engine.c64_scoring_seconds,
            initial_c64_runtime_seconds=c64_start_seconds,
            completed_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_progress(status_path, status)
        result = {**status, "directory": str(output_dir.relative_to(output_root)),
            "event_count": trajectory.event_count, "atomic_unit_count": group_count,
            "initial_b3_threshold": float(np.quantile(engine.base_c64, .90, method="higher")),
            "final_b3_threshold": float(engine.controller.theta),
            "threshold_movements": list(engine.threshold_movements),
            "transition_count": engine.b3_refresh_count,
            "output_bytes": sum(path.stat().st_size for path in output_dir.iterdir() if path.is_file())}
        append_fsync(campaign_progress, {"event": "TRAJECTORY_SEALED_COMPLETE",
            "trajectory": trajectory.key, "completed_event_count": trajectory.event_count,
            "b3_refresh_count": engine.b3_refresh_count,
            "runtime_seconds": elapsed, "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        return result
    except BaseException as exc:
        status.update(status="INCOMPLETE", completed_event_count=int(status.get("completed_event_count", 0)),
            error_type=type(exc).__name__, error_message=str(exc), traceback=traceback.format_exc(),
            failed_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_progress(status_path, status)
        append_fsync(campaign_progress, {"event": "TRAJECTORY_INCOMPLETE",
            "trajectory": trajectory.key, "completed_event_count": status["completed_event_count"],
            "error_type": type(exc).__name__, "error_message": str(exc),
            "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        raise
    finally:
        if engine is not None:
            del engine
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def _load_sidecar_rows(trajectory: Trajectory, output_root: Path
                       ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    directory = _trajectory_output_directory(output_root, trajectory)
    receipt = read_json(directory / "stage_a_completion_receipt.json")
    status = read_json(directory / "attempt_status.json")
    if (receipt.get("status") != "SEALED_COMPLETE" or
            status.get("status") != "SEALED_COMPLETE" or
            receipt.get("completed_event_count") != trajectory.event_count or
            status.get("completed_event_count") != trajectory.event_count or
            receipt.get("prediction_journal_sha256") != sha256_file(
                directory / "shadow_b3_prediction_journal.jsonl") or
            receipt.get("c64_b3_transitions_sha256") != (
                sha256_file(directory / "shadow_c64_b3_transitions.jsonl")
                if (directory / "shadow_c64_b3_transitions.jsonl").exists() else sha256_bytes(b""))):
        raise SupplementaryError(f"{trajectory.key}: supplementary Stage A receipt failed integrity verification")
    original_seal, original_rows, original_groups = _attempt_seal_and_rows(trajectory)
    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    cursor = 0
    with (directory / "shadow_b3_prediction_journal.jsonl").open("rb") as stream:
        for raw in stream:
            envelope = json.loads(raw)
            local = envelope.get("score_rows")
            start, stop = int(envelope.get("start_position", -1)), int(envelope.get("end_position_exclusive", -1))
            if (envelope.get("schema") != "nshadow-independent-b3-c64.atomic-unit.v1" or
                    not isinstance(local, list) or start != cursor or stop != start + len(local) or
                    envelope.get("status") != "SEALED_BEFORE_INHERITED_FEEDBACK_DISCLOSURE" or
                    envelope.get("score_rows_sha256") != sha256_object(local) or
                    envelope.get("event_order_sha256") != sha256_object(
                        [row.get("event_id_sha256") for row in local])):
                raise SupplementaryError(f"{trajectory.key}: supplementary prediction journal has invalid unit order/hash")
            source = original_groups[len(groups)]
            source_local = original_rows[start:stop]
            if (source.get("source_group_sha256") != envelope.get("source_group_sha256") or
                    len(source_local) != len(local) or
                    envelope.get("source_state_sha256") != source_local[0].get("n_shadow_state_sha256") or
                    envelope.get("source_state_version") != source_local[0].get("n_shadow_state_version")):
                raise SupplementaryError(f"{trajectory.key}:{start}: supplementary unit is not bound to original source state")
            for offset, row in enumerate(local):
                registered = trajectory.records[start + offset]
                original = source_local[offset]
                if (row.get("position") != start + offset or
                        row.get("event_id_sha256") != registered.get("event_id_sha256") or
                        row.get("image_identity_sha256") != registered.get("image_identity_sha256") or
                        row.get("source_group_sha256") != registered.get("source_group_sha256") or
                        row.get("n_shadow_score") != original.get("n_shadow_score") or
                        row.get("source_state_sha256") != original.get("n_shadow_state_sha256") or
                        row.get("source_state_version") != original.get("n_shadow_state_version") or
                        int(row.get("prediction", -1)) != int(
                            float(row["n_shadow_score"]) >= float(row["independent_b3_threshold"]))):
                    raise SupplementaryError(f"{trajectory.key}:{start + offset}: supplementary row identity/prediction mismatch")
            groups.append(dict(envelope))
            rows.extend(dict(row) for row in local)
            cursor = stop
    if (cursor != trajectory.event_count or len(rows) != trajectory.event_count or
            len(groups) != len(original_groups)):
        raise SupplementaryError(f"{trajectory.key}: supplementary sidecar does not cover the complete original stream")
    if sha256_object([row["event_id_sha256"] for row in rows]) != \
            sha256_object([row["event_id_sha256"] for row in original_rows]):
        raise SupplementaryError(f"{trajectory.key}: supplementary event order differs from original Stage A")
    return receipt, rows, groups


def _verify_metric_parity(current: Mapping[str, Any], historical: Mapping[str, Any],
                          *, keys: Sequence[str], context: str) -> None:
    for key in keys:
        left, right = current.get(key), historical.get(key)
        if left is None or right is None:
            if left is not right:
                raise SupplementaryError(f"{context}: historical metric parity mismatch for {key}")
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12):
                raise SupplementaryError(f"{context}: historical metric parity mismatch for {key}")
        elif left != right:
            raise SupplementaryError(f"{context}: historical metric parity mismatch for {key}")


def _metric_slices(labels: Sequence[int], scores: Sequence[float],
                   predictions: Sequence[int], dataset_id: str) -> dict[str, Any]:
    from experiments.calibration.final_benchmark.metrics import binary_metrics
    n = len(labels)
    spans = {
        "early": (0, n // 3), "middle": (n // 3, (2 * n) // 3),
        "late": ((2 * n) // 3, n),
        "final_30_percent": (int(math.floor(n * .70)), n),
        "final_10_percent": (int(math.floor(n * .90)), n),
    }
    result = {}
    for name, (start, stop) in spans.items():
        result[name] = {"start_position": start, "end_position_exclusive": stop,
            "metrics": binary_metrics(labels[start:stop], scores[start:stop], predictions[start:stop])}
    return result


def _stage_b_one(trajectory: Trajectory, output_root: Path) -> dict[str, Any]:
    """Project official outcomes only after all 102 supplementary seals exist."""
    from experiments.calibration.final_benchmark.metrics import binary_metrics, worst_256_window_ber
    from experiments.calibration.final_benchmark.foads_evaluation import _project_labels
    from experiments.calibration.final_benchmark.visa_stage_b import _project_all_test_labels

    receipt, shadow_rows, _groups = _load_sidecar_rows(trajectory, output_root)
    original_seal, original_rows, _original_groups = _attempt_seal_and_rows(trajectory)
    if trajectory.dataset_id == "realiad":
        annotation = Path(trajectory.bundle["annotation_path"])
        labels = _project_labels(annotation.read_bytes(),
            expected_sha256=str(trajectory.registration["annotation_sha256"]),
            category=trajectory.category,
            dataset_release_sha256=str(trajectory.release_identity["dataset_release_sha256"]),
            stream_seed=trajectory.stream_seed,
            events=trajectory.records)
        source_sha = str(trajectory.registration["annotation_sha256"])
    else:
        identity_manifest = read_json(trajectory.bundle["identity_manifest_path"])
        source_sha = str(trajectory.registration["source_csv_sha256"])
        identity_sources = [item for item in identity_manifest.get("source_artifacts", [])
                            if item.get("artifact_id") == "official-split:1cls.csv"]
        if len(identity_sources) != 1 or identity_sources[0].get("sha256") != source_sha:
            raise SupplementaryError(f"{trajectory.key}: official VisA outcome source hash differs from Gate 1")
        labels_by_event = _project_all_test_labels(
            dataset_root=Path(trajectory.bundle["dataset_root"]),
            source_csv_sha256=source_sha, category=trajectory.category,
            records=trajectory.records, rows=original_rows)
        labels = [int(labels_by_event[str(row["event_id_sha256"])]) for row in original_rows]
    if len(labels) != trajectory.event_count or any(value not in (0, 1) for value in labels):
        raise SupplementaryError(f"{trajectory.key}: post-seal image-level outcome projection is incomplete")

    # Validate every previously purchased F-B disclosure against the official
    # image-level outcome only now, after the supplementary full-stream seal.
    leaf = _trajectory_output_directory(output_root, trajectory)
    private_receipts: dict[int, dict[str, Any]] = {}
    private_file = leaf / "private_feedback_transition_receipts.jsonl"
    if private_file.exists():
        with private_file.open("r", encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                position = int(item.get("position", -1))
                if (position in private_receipts or not 0 <= position < trajectory.event_count or
                        item.get("event_id_sha256") != original_rows[position].get("event_id_sha256") or
                        item.get("purchased_label") != int(labels[position])):
                    raise SupplementaryError(f"{trajectory.key}:{position}: purchased label differs from post-seal official outcome")
                private_receipts[position] = item
    f_b_queries = [i for i, row in enumerate(original_rows)
                   if bool(row["query_actions"]["F-B"].get("executed"))]
    if set(private_receipts) != set(f_b_queries):
        raise SupplementaryError(f"{trajectory.key}: private F-B feedback receipts do not cover inherited queries")

    label_vector = np.asarray(labels, dtype=np.int8)
    shadow_scores = [float(row["n_shadow_score"]) for row in shadow_rows]
    shadow_predictions = [int(row["prediction"]) for row in shadow_rows]
    fb_scores = [float(row["arm_scores"]["F-B"]) for row in original_rows]
    fb_predictions = [int(row["arm_predictions"]["F-B"]) for row in original_rows]
    fm_scores = [float(row["arm_scores"]["F-M"]) for row in original_rows]
    fm_predictions = [int(row["arm_predictions"]["F-M"]) for row in original_rows]
    shadow_metric = binary_metrics(labels, shadow_scores, shadow_predictions)
    fb_metric = binary_metrics(labels, fb_scores, fb_predictions)
    fm_metric = binary_metrics(labels, fm_scores, fm_predictions)
    original_metrics = trajectory.original_stage_b.get("metrics", trajectory.original_stage_b.get("arm_metrics", {}))
    historical_shadow = original_metrics.get("N-shadow")
    historical_fb = original_metrics.get("F-B")
    historical_fm = original_metrics.get("F-M")
    if not all(isinstance(value, Mapping) for value in (historical_shadow, historical_fb, historical_fm)):
        raise SupplementaryError(f"{trajectory.key}: original Stage B lacks N-shadow/F-B/F-M metrics")
    _verify_metric_parity(shadow_metric, historical_shadow,
        keys=("n", "positive_count", "negative_count", "auroc", "ap"),
        context=f"{trajectory.key}: original N-shadow ranking")
    _verify_metric_parity(fb_metric, historical_fb,
        keys=("n", "positive_count", "negative_count", "auroc", "ap",
              "balanced_accuracy", "ber", "fpr", "fnr"),
        context=f"{trajectory.key}: original F-B operating metric")
    _verify_metric_parity(fm_metric, historical_fm,
        keys=("n", "positive_count", "negative_count", "auroc", "ap",
              "balanced_accuracy", "ber", "fpr", "fnr"),
        context=f"{trajectory.key}: original F-M operating metric")

    query_count = {arm: sum(bool(row["query_actions"][arm].get("executed"))
                            for row in original_rows) for arm in PRINCIPAL_ARMS}
    queried_composition = {}
    first_queried_defect = {}
    for arm in PRINCIPAL_ARMS:
        queried = [i for i, row in enumerate(original_rows)
                   if bool(row["query_actions"][arm].get("executed"))]
        queried_composition[arm] = {
            "normal": sum(int(labels[i]) == 0 for i in queried),
            "defect": sum(int(labels[i]) == 1 for i in queried),
        }
        defect_pos = [int(original_rows[i]["position"]) for i in queried if int(labels[i]) == 1]
        first_queried_defect[arm] = min(defect_pos) if defect_pos else None
    if trajectory.dataset_id == "realiad":
        registered_query_count = trajectory.original_stage_b.get("query_count")
        registered_composition = trajectory.original_stage_b.get("queried_label_composition")
        registered_first = trajectory.original_stage_b.get("first_queried_defect_position")
    else:
        registered_query_count = trajectory.original_stage_b.get("query_count")
        registered_composition = trajectory.original_stage_b.get("queried_label_composition")
        registered_first = trajectory.original_stage_b.get("first_queried_defect_position")
    if (query_count != registered_query_count or queried_composition != registered_composition or
            first_queried_defect != registered_first):
        raise SupplementaryError(f"{trajectory.key}: Stage B query counts/composition/first defect fail historical parity")

    contrast = {}
    for metric in ("ber", "fpr", "fnr"):
        left, right = shadow_metric.get(metric), fb_metric.get(metric)
        contrast[f"n_shadow_b3_minus_f_b_{metric}"] = (
            None if left is None or right is None else float(left) - float(right))
    fm_contrast = {key: (None if shadow_metric.get(key) is None or fm_metric.get(key) is None
                         else float(shadow_metric[key]) - float(fm_metric[key]))
                   for key in ("ber", "fpr", "fnr")}
    b3_thresholds = [float(row["independent_b3_threshold"]) for row in shadow_rows]
    b3_refresh_count = int(receipt["independent_b3_refresh_count"])
    temporal = {
        "N-shadow+B3": _metric_slices(labels, shadow_scores, shadow_predictions, trajectory.dataset_id),
        "F-B": _metric_slices(labels, fb_scores, fb_predictions, trajectory.dataset_id),
        "F-M": _metric_slices(labels, fm_scores, fm_predictions, trajectory.dataset_id),
    }
    result = {
        "schema": "nshadow-independent-b3-c64.stage-b-metrics.v1",
        "status": "STAGE_B_COMPLETE",
        "dataset_id": trajectory.dataset_id, "category": trajectory.category,
        "condition": trajectory.condition,
        "event_count": trajectory.event_count,
        "positive_count": int(label_vector.sum()),
        "negative_count": int(len(label_vector) - label_vector.sum()),
        "N-shadow+B3": shadow_metric,
        "F-B": fb_metric,
        "F-M": fm_metric,
        "N-shadow+B3_minus_F-B": contrast,
        "N-shadow+B3_minus_F-M_descriptive": fm_contrast,
        "query_count": query_count,
        "queried_label_composition": queried_composition,
        "first_queried_defect_position": first_queried_defect,
        "inherited_f_b_query_count": len(f_b_queries),
        "independent_b3_refresh_count": b3_refresh_count,
        "independent_b3_threshold": {
            "initial": float(receipt["initial_b3_threshold"]),
            "final": float(receipt["final_b3_threshold"]),
            "minimum": min(b3_thresholds), "maximum": max(b3_thresholds),
            "range": max(b3_thresholds) - min(b3_thresholds),
            "maximum_absolute_refresh_movement": max(
                (abs(float(value["movement"])) for value in _read_jsonl(
                    leaf / "shadow_c64_b3_transitions.jsonl")), default=0.0),
        },
        "temporal_metrics": temporal,
        "original_n_shadow_ranking_parity": True,
        "original_f_b_stage_b_metric_parity": True,
        "original_query_accounting_parity": True,
        "purchased_f_b_feedback_matches_official_image_labels": True,
        "official_label_source_sha256": source_sha,
        "supplementary_stage_a_seal_sha256": sha256_file(leaf / "stage_a_completion_receipt.json"),
        "original_stage_a_seal_sha256": sha256_file(
            trajectory.attempt_root / "stage_a_category_seal.json"),
        "outcome_access_phase": "STAGE_B_AFTER_ALL_102_SUPPLEMENTARY_STAGE_A_SEALS",
        "pixel_masks_accessed": False,
    }
    target = leaf / "stage_b_metrics.json"
    write_exclusive(target, result)
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _paired_bootstrap(values: Mapping[str, float], *, seed: int) -> dict[str, Any]:
    names = sorted(values)
    array = np.asarray([float(values[name]) for name in names], dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        return {"status": "UNDEFINED", "category_count": len(array),
                "mean": None, "ci_95_percentile": None,
                "replicates": BOOTSTRAP_REPLICATES, "seed": seed}
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for begin in range(0, BOOTSTRAP_REPLICATES, 1000):
        size = min(1000, BOOTSTRAP_REPLICATES - begin)
        indices = rng.integers(0, len(array), size=(size, len(array)))
        draws[begin:begin + size] = array[indices].mean(axis=1)
    return {"status": "DEFINED", "category_count": len(array),
        "category_ids": names, "mean": float(array.mean()),
        "ci_95_percentile": [float(np.quantile(draws, .025)),
                             float(np.quantile(draws, .975))],
        "replicates": BOOTSTRAP_REPLICATES, "seed": seed,
        "resampling_unit": "category; conditions averaged within category before resampling"}


def _average(values: Sequence[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _collect_stage_b(trajectories: Sequence[Trajectory], output_root: Path
                     ) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    result: dict[str, dict[tuple[str, str], dict[str, Any]]] = {"realiad": {}, "visa_1cls": {}}
    for trajectory in trajectories:
        path = _trajectory_output_directory(output_root, trajectory) / "stage_b_metrics.json"
        metric = read_json(path)
        if metric.get("status") != "STAGE_B_COMPLETE":
            raise SupplementaryError(f"{trajectory.key}: missing completed supplementary Stage B result")
        result[trajectory.dataset_id][(trajectory.category, trajectory.condition)] = metric
    if len(result["realiad"]) != 90 or len(result["visa_1cls"]) != 12:
        raise SupplementaryError("aggregate input census does not contain 90 Real-IAD and 12 VisA Stage B results")
    return result


def _category_metrics(rows: Mapping[tuple[str, str], Mapping[str, Any]], categories: Sequence[str],
                      conditions: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    names = ("auroc", "ap", "balanced_accuracy", "ber", "fpr", "fnr")
    per_category: dict[str, Any] = {}
    macro: dict[str, Any] = {"category_count": len(categories), "category_weighting": "equal"}
    for category in categories:
        condition_values = [rows[(category, condition)] for condition in conditions]
        cat_data: dict[str, Any] = {}
        for arm in ("N-shadow+B3", "F-B", "F-M"):
            metrics = [item[arm] for item in condition_values]
            cat_data[arm] = {key: _average([metric.get(key) for metric in metrics])
                             for key in names}
        cat_data["N-shadow+B3_minus_F-B"] = {
            key: _average([item["N-shadow+B3_minus_F-B"].get(f"n_shadow_b3_minus_f_b_{key}")
                           for item in condition_values]) for key in ("ber", "fpr", "fnr")}
        cat_data["N-shadow+B3_minus_F-M_descriptive"] = {
            key: _average([item["N-shadow+B3_minus_F-M_descriptive"].get(key)
                           for item in condition_values]) for key in ("ber", "fpr", "fnr")}
        cat_data["trajectory_count"] = len(conditions)
        per_category[category] = cat_data
    for arm in ("N-shadow+B3", "F-B", "F-M"):
        macro[arm] = {metric: _average([per_category[cat][arm][metric] for cat in categories])
                      for metric in names}
    macro["N-shadow+B3_minus_F-B"] = {
        metric: _average([per_category[cat]["N-shadow+B3_minus_F-B"][metric]
                          for cat in categories]) for metric in ("ber", "fpr", "fnr")}
    macro["N-shadow+B3_minus_F-M_descriptive"] = {
        metric: _average([per_category[cat]["N-shadow+B3_minus_F-M_descriptive"][metric]
                          for cat in categories]) for metric in ("ber", "fpr", "fnr")}
    return per_category, macro


def _aggregate_and_report(trajectories: Sequence[Trajectory], output_root: Path,
                          audit: Mapping[str, Any], total_runtime_seconds: float,
                          max_host_rss_bytes: int | None) -> dict[str, Any]:
    stage_b = _collect_stage_b(trajectories, output_root)
    real_by_key = stage_b["realiad"]
    visa_by_key = stage_b["visa_1cls"]
    real_categories = sorted({key[0] for key in real_by_key})
    visa_categories = sorted({key[0] for key in visa_by_key})
    real_per_category, real_macro = _category_metrics(
        real_by_key, real_categories, ("P0", "P1", "P2"))
    visa_per_category, visa_macro = _category_metrics(
        visa_by_key, visa_categories, ("P0",))
    if len(real_categories) != 30 or len(visa_categories) != 12:
        raise SupplementaryError("category-level aggregate census differs from 30 Real-IAD and 12 VisA")

    real_bootstrap = {}
    for metric, seed_offset in (("ber", 0), ("fpr", 1), ("fnr", 2)):
        real_bootstrap[f"n_shadow_b3_minus_f_b_{metric}"] = _paired_bootstrap(
            {category: real_per_category[category]["N-shadow+B3_minus_F-B"][metric]
             for category in real_categories}, seed=REAL_BOOTSTRAP_SEED + seed_offset)
    visa_bootstrap = {}
    for metric, seed_offset in (("ber", 0), ("fpr", 1), ("fnr", 2)):
        visa_bootstrap[f"n_shadow_b3_minus_f_b_{metric}"] = _paired_bootstrap(
            {category: visa_per_category[category]["N-shadow+B3_minus_F-B"][metric]
             for category in visa_categories}, seed=VISA_BOOTSTRAP_SEED + seed_offset)
    real_contrasts = {"schema": "nshadow-independent-b3-c64.realiad-category-contrasts.v1",
        "status": "COMPLETE", "hypothesis_status": "POST_HOC_EXPLORATORY",
        "categories": real_per_category, "category_equal_macro": real_macro,
        "paired_category_bootstrap": real_bootstrap,
        "negative_category_effects": {
            "ber_n_shadow_b3_lower_than_f_b": [cat for cat in real_categories
                if real_per_category[cat]["N-shadow+B3_minus_F-B"]["ber"] is not None and
                real_per_category[cat]["N-shadow+B3_minus_F-B"]["ber"] < 0],
        }, "conditions_averaged_within_category": ["P0", "P1", "P2"],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "real_iad_categories_are_the_resampling_unit": True}
    visa_contrasts = {"schema": "nshadow-independent-b3-c64.visa-category-contrasts.v1",
        "status": "COMPLETE", "hypothesis_status": "POST_HOC_EXPLORATORY",
        "categories": visa_per_category, "category_equal_macro": visa_macro,
        "paired_category_bootstrap": visa_bootstrap,
        "negative_category_effects": {
            "ber_n_shadow_b3_lower_than_f_b": [cat for cat in visa_categories
                if visa_per_category[cat]["N-shadow+B3_minus_F-B"]["ber"] is not None and
                visa_per_category[cat]["N-shadow+B3_minus_F-B"]["ber"] < 0],
        }, "conditions_averaged_within_category": ["P0"],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "visa_categories_are_the_resampling_unit": True,
        "physical_independence_claim": "NOT_ESTABLISHED"}
    write_exclusive(output_root / "realiad_30_category_contrasts.json", real_contrasts)
    write_exclusive(output_root / "visa_12_category_contrasts.json", visa_contrasts)

    real_rows = []
    temporal_rows = []
    for (category, condition), item in sorted(real_by_key.items()):
        row = {"category": category, "condition": condition,
            "event_count": item["event_count"],
            "query_count_f_b": item["query_count"]["F-B"],
            "query_normal_f_b": item["queried_label_composition"]["F-B"]["normal"],
            "query_defect_f_b": item["queried_label_composition"]["F-B"]["defect"],
            "n_shadow_b3_refresh_count": item["independent_b3_refresh_count"],
            "threshold_initial": item["independent_b3_threshold"]["initial"],
            "threshold_final": item["independent_b3_threshold"]["final"],
            "threshold_minimum": item["independent_b3_threshold"]["minimum"],
            "threshold_maximum": item["independent_b3_threshold"]["maximum"],
            "threshold_max_abs_movement": item["independent_b3_threshold"]["maximum_absolute_refresh_movement"]}
        for arm in ("N-shadow+B3", "F-B", "F-M"):
            for metric in ("auroc", "ap", "ber", "fpr", "fnr", "balanced_accuracy"):
                row[f"{arm.replace('-', '_').replace('+', '_')}_{metric}"] = item[arm].get(metric)
        for metric in ("ber", "fpr", "fnr"):
            row[f"delta_n_shadow_b3_minus_f_b_{metric}"] = item["N-shadow+B3_minus_F-B"][
                f"n_shadow_b3_minus_f_b_{metric}"]
        real_rows.append(row)
        for arm, segment_data in item["temporal_metrics"].items():
            for segment, values in segment_data.items():
                temporal_rows.append({"dataset_id": "realiad", "category": category,
                    "condition": condition, "arm": arm, "segment": segment,
                    "start_position": values["start_position"],
                    "end_position_exclusive": values["end_position_exclusive"],
                    **values["metrics"]})
    _write_csv(output_root / "realiad_90_metrics.csv", real_rows)
    visa_rows = []
    for (category, condition), item in sorted(visa_by_key.items()):
        row = {"category": category, "condition": condition,
            "event_count": item["event_count"],
            "query_count_f_b": item["query_count"]["F-B"],
            "query_normal_f_b": item["queried_label_composition"]["F-B"]["normal"],
            "query_defect_f_b": item["queried_label_composition"]["F-B"]["defect"],
            "n_shadow_b3_refresh_count": item["independent_b3_refresh_count"],
            "threshold_initial": item["independent_b3_threshold"]["initial"],
            "threshold_final": item["independent_b3_threshold"]["final"],
            "threshold_minimum": item["independent_b3_threshold"]["minimum"],
            "threshold_maximum": item["independent_b3_threshold"]["maximum"],
            "threshold_max_abs_movement": item["independent_b3_threshold"]["maximum_absolute_refresh_movement"]}
        for arm in ("N-shadow+B3", "F-B", "F-M"):
            for metric in ("auroc", "ap", "ber", "fpr", "fnr", "balanced_accuracy"):
                row[f"{arm.replace('-', '_').replace('+', '_')}_{metric}"] = item[arm].get(metric)
        for metric in ("ber", "fpr", "fnr"):
            row[f"delta_n_shadow_b3_minus_f_b_{metric}"] = item["N-shadow+B3_minus_F-B"][
                f"n_shadow_b3_minus_f_b_{metric}"]
        visa_rows.append(row)
        for arm, segment_data in item["temporal_metrics"].items():
            for segment, values in segment_data.items():
                temporal_rows.append({"dataset_id": "visa_1cls", "category": category,
                    "condition": condition, "arm": arm, "segment": segment,
                    "start_position": values["start_position"],
                    "end_position_exclusive": values["end_position_exclusive"],
                    **values["metrics"]})
    _write_csv(output_root / "visa_12_metrics.csv", visa_rows)
    _write_csv(output_root / "temporal_aggregate.csv", temporal_rows)

    status_paths = [_trajectory_output_directory(output_root, trajectory) / "attempt_status.json"
                    for trajectory in trajectories]
    stage_a_statuses = [read_json(path) for path in status_paths]
    runtimes = [float(status.get("runtime_seconds", 0)) for status in stage_a_statuses]
    peaks_gpu_alloc = [int(status.get("peak_gpu_allocated_bytes", 0)) for status in stage_a_statuses]
    peaks_gpu_reserved = [int(status.get("peak_gpu_reserved_bytes", 0)) for status in stage_a_statuses]
    total_output_bytes = sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file())
    resource = {
        "schema": "nshadow-independent-b3-c64.resource-and-fidelity.v1",
        "status": "COMPLETE", "trajectory_count": len(stage_a_statuses),
        "total_measured_stage_a_seconds": float(sum(runtimes)),
        "wall_clock_total_seconds_including_stage_b": float(total_runtime_seconds),
        "mean_trajectory_runtime_seconds": float(np.mean(runtimes)),
        "max_trajectory_runtime_seconds": max(runtimes, default=0.0),
        "process_peak_rss_sampled_bytes": max_host_rss_bytes,
        "process_rss_sampling_period_seconds": 1,
        "gpu_peak_allocated_bytes_max_over_trajectories": max(peaks_gpu_alloc, default=0),
        "gpu_peak_reserved_bytes_max_over_trajectories": max(peaks_gpu_reserved, default=0),
        "gpu_sampling": "PyTorch CUDA allocator high-water marks, reset per trajectory; does not include non-PyTorch driver allocations",
        "output_tree_bytes": int(total_output_bytes),
        "n_shadow_native_score_replay_events": sum(int(s.get("query_score_parity_count", 0)) for s in stage_a_statuses),
        "state_digest_exact_match_count": sum(int(s.get("expected_event_count", 0)) for s in stage_a_statuses),
        "trajectory_counts": {"realiad": 90, "visa_1cls": 12},
        "image_event_counts": {"realiad": 343755, "visa_1cls": 2162},
        "query_event_only_native_replay": True,
        "new_test_outcomes_read_before_all_102_seals": False,
        "historical_source_commit_sha_differences_blocking": False,
        "asset_audit_sha256": sha256_file(output_root / "ASSET_AND_REPLAY_AUDIT.json"),
    }
    write_exclusive(output_root / "RESOURCE_AND_FIDELITY_REPORT.json", resource)
    _write_resource_report(output_root / "RESOURCE_AND_FIDELITY_REPORT.md", resource)
    _write_figures(output_root, real_per_category, visa_per_category)
    report = _write_final_report(output_root, real_macro, visa_macro,
                                 real_bootstrap, visa_bootstrap, real_per_category,
                                 visa_per_category, resource, audit)
    return {"real_iad": real_macro, "visa": visa_macro,
            "real_iad_bootstrap": real_bootstrap,
            "visa_bootstrap": visa_bootstrap,
            "report_path": str((output_root / "FINAL_SUPPLEMENTARY_RESEARCH_REPORT.md").relative_to(ROOT)),
            "resource": resource, "final_report": report}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise SupplementaryError(f"refusing to write an empty aggregate table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _write_resource_report(path: Path, resource: Mapping[str, Any]) -> None:
    body = ["# Resource and fidelity report", "",
        f"- Supplementary trajectories: {resource['trajectory_count']}.",
        f"- Measured Stage A runtime total: {resource['total_measured_stage_a_seconds']:.1f} s.",
        f"- End-to-end wall time through Stage B and aggregation: {resource['wall_clock_total_seconds_including_stage_b']:.1f} s.",
        f"- Maximum sampled process RSS: {resource['process_peak_rss_sampled_bytes']} bytes; process sampled every {resource['process_rss_sampling_period_seconds']} s.",
        f"- Maximum PyTorch GPU allocated/reserved high-water marks: {resource['gpu_peak_allocated_bytes_max_over_trajectories']} / {resource['gpu_peak_reserved_bytes_max_over_trajectories']} bytes.",
        f"- Output tree size: {resource['output_tree_bytes']} bytes.",
        "- Native scoring parity was recomputed only for inherited F-B query events; the sealed raw N-shadow score was reused for all other image positions.",
        "- The reconstructed normal-memory/C1 state digest and version matched the original sealed record at every image event.",
        "- Stage B opened official image-level outcomes after all 102 supplementary Stage A seals; no pixel masks were read.", ""]
    write_text_exclusive(path, "\n".join(body))


def _write_figures(root: Path, real: Mapping[str, Any], visa: Mapping[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for dataset, data, name in (("Real-IAD", real, "realiad_category_ber_contrasts.png"),
                                ("VisA", visa, "visa_category_ber_contrasts.png")):
        categories = sorted(data)
        values = [data[cat]["N-shadow+B3_minus_F-B"]["ber"] for cat in categories]
        colors = ["#2b6cb0" if value is not None and value < 0 else "#c05621" for value in values]
        fig, ax = plt.subplots(figsize=(max(8, len(categories) * .35), 4.8))
        ax.bar(range(len(categories)), [0.0 if value is None else value for value in values], color=colors)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_ylabel("BER(N-shadow+B3) − BER(F-B)")
        ax.set_title(f"{dataset}: category BER contrasts; negative favors N-shadow+B3")
        ax.set_xticks(range(len(categories)), categories, rotation=70, ha="right", fontsize=8)
        fig.tight_layout()
        fig.savefig(root / name, dpi=160)
        plt.close(fig)


def _write_final_report(root: Path, real: Mapping[str, Any], visa: Mapping[str, Any],
                         real_bootstrap: Mapping[str, Any], visa_bootstrap: Mapping[str, Any],
                         real_categories: Mapping[str, Any], visa_categories: Mapping[str, Any],
                         resource: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    def describe(label: str, macro: Mapping[str, Any], boot: Mapping[str, Any],
                 categories: Mapping[str, Any]) -> list[str]:
        effect = macro["N-shadow+B3_minus_F-B"]
        lines = [f"### {label}", "",
            "| Method | AUROC | AP | BER | FPR | FNR | BA |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for arm in ("N-shadow+B3", "F-B", "F-M"):
            m = macro[arm]
            lines.append(f"| {arm} | {_fmt(m['auroc'])} | {_fmt(m['ap'])} | {_fmt(m['ber'])} | {_fmt(m['fpr'])} | {_fmt(m['fnr'])} | {_fmt(m['balanced_accuracy'])} |")
        lines.extend(["", "| Contrast | Mean difference | Paired category-bootstrap 95% CI |",
            "|---|---:|---:|"])
        for metric in ("ber", "fpr", "fnr"):
            key = f"n_shadow_b3_minus_f_b_{metric}"
            interval = boot[key].get("ci_95_percentile")
            ci = "undefined" if interval is None else f"[{interval[0]:.4f}, {interval[1]:.4f}]"
            lines.append(f"| Δ {metric.upper()} (N-shadow+B3 − F-B) | {_fmt(effect[metric])} | {ci} |")
        negative = [cat for cat in categories if categories[cat]["N-shadow+B3_minus_F-B"]["ber"] is not None
                    and categories[cat]["N-shadow+B3_minus_F-B"]["ber"] < 0]
        positive = [cat for cat in categories if categories[cat]["N-shadow+B3_minus_F-B"]["ber"] is not None
                    and categories[cat]["N-shadow+B3_minus_F-B"]["ber"] > 0]
        lines.extend(["", f"Categories where N-shadow+B3 has lower BER than F-B: {len(negative)}/{len(categories)}.",
                      f"Categories where F-B has lower BER than N-shadow+B3: {len(positive)}/{len(categories)}.", ""])
        return lines

    real_delta = real["N-shadow+B3_minus_F-B"]["ber"]
    visa_delta = visa["N-shadow+B3_minus_F-B"]["ber"]
    real_ci = real_bootstrap["n_shadow_b3_minus_f_b_ber"].get("ci_95_percentile")
    visa_ci = visa_bootstrap["n_shadow_b3_minus_f_b_ber"].get("ci_95_percentile")
    lines = ["# N-shadow with independent B3/C64: supplementary results", "",
        "## Status and method", "",
        "Completed as a post-hoc exploratory supplementary analysis over the existing sealed streams: 90 Real-IAD category-condition trajectories (P0/P1/P2) and 12 VisA P0 trajectories, 345,917 image events total.", "",
        "The N-shadow detector remains the original shared F-B Greedy memory plus its independent normal-only C1 correction memory. It inherits only F-B's already executed query events, updates C1 only from verified-normal feedback, ignores defects for shadow C1, and receives no independent query decisions. The supplementary controller is a separate R7 B3 StateConsistentQuantile initialized and refreshed from the 64 registered protected C64 images, q90 with method `higher`, minimum 16 normal anchors, and a maximum 0.01 threshold movement per source-state transition.", "",
        "The sidecar reuses original sealed N-shadow scores for non-query events and reconstructs mutable state from the exact purchased F-B feedback records. At each inherited F-B query event, a native F-B and N-shadow score replay was computed and compared with the original sealed score within a predeclared 1e-5 tolerance. State digests and versions were checked on every event. Each Real-IAD five-view group (and each registered VisA atomic unit) was scored and fsynced before its corresponding private purchased feedback was consumed. Official image-level outcomes were opened only after all 102 supplementary seals; masks were not accessed.", "",
        "The central exploratory contrast is Δ BER = BER(N-shadow+B3) − BER(F-B); positive values favor F-B. This is post hoc and does not replace the original AUROC/AP-only N-shadow analysis or establish causal attribution beyond this registered feedback history.", ""]
    lines.extend(describe("Real-IAD: 30 categories, conditions averaged within category", real, real_bootstrap, real_categories))
    lines.extend(describe("VisA: 12 categories, P0", visa, visa_bootstrap, visa_categories))
    lines.extend(["## Interpretation", "",
        f"Real-IAD mean Δ BER was {_fmt(real_delta)}" +
        (f" (paired 30-category bootstrap 95% CI [{real_ci[0]:.4f}, {real_ci[1]:.4f}])." if real_ci else " (interval undefined)."),
        f"VisA mean Δ BER was {_fmt(visa_delta)}" +
        (f" (paired 12-category bootstrap 95% CI [{visa_ci[0]:.4f}, {visa_ci[1]:.4f}])." if visa_ci else " (interval undefined)."), "",
        "A positive Δ BER indicates the independent B3 threshold leaves N-shadow with higher operating BER than F-B; a negative value indicates lower N-shadow+B3 BER. FPR and FNR differences are reported separately because a BER change can conceal an operating-point trade-off. F-M is descriptive only because it follows a different query/feedback history.", "",
        "This analysis identifies the realized post-hoc behavior of the independently calibrated normal-only shadow on these streams. It does not identify counterfactual learning value for unqueried observations, prove that defect-side correction alone caused any performance difference, establish a universal threshold policy, or establish VisA physical independence from Real-IAD.", "",
        "## Computation and limitations", "",
        f"Measured Stage A replay time: {resource['total_measured_stage_a_seconds']:.1f} s across 102 trajectories. End-to-end execution time through Stage B and aggregation: {resource['wall_clock_total_seconds_including_stage_b']:.1f} s. The sidecar avoided recomputing the four-arm stream: only the already purchased F-B query images were re-encoded for native score parity. Current-state C64 scoring ran when N-shadow's complete scoring-state digest changed.", "",
        "Per-category results, temporal segments, state-transition records, Stage B metrics, confidence intervals, resource measures and fidelity checks are saved beside this report. Real-IAD and VisA were analyzed separately with categories as the bootstrap unit.", ""])
    write_text_exclusive(root / "FINAL_SUPPLEMENTARY_RESEARCH_REPORT.md", "\n".join(lines))
    return {"realiad_delta_ber": real_delta, "visa_delta_ber": visa_delta,
            "realiad_ci": real_ci, "visa_ci": visa_ci}


def _fresh_output_root(base: Path) -> Path:
    """Reserve a fresh campaign root without touching any prior attempt."""
    base.parent.mkdir(parents=True, exist_ok=True)
    candidates = [base] + [base.with_name(f"{base.name}_attempt_{index:02d}")
                           for index in range(2, 100)]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise SupplementaryError("no fresh supplementary campaign root is available")


def _commissioning_for(runtime: Any, trajectory: Trajectory,
                       cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = sha256_object({"dataset_id": trajectory.dataset_id,
        "category": trajectory.category, "role_manifest_sha256": sha256_object(trajectory.role_manifest),
        "dataset_release_sha256": trajectory.release_identity["dataset_release_sha256"]})
    if key in cache:
        return cache[key]
    selected_ids = {str(row["image_identity_sha256"])
                    for row in trajectory.role_manifest["selected"]}
    if trajectory.dataset_id == "realiad":
        locators = trajectory.bundle.get("role_private_locators")
    else:
        all_locators = trajectory.bundle.get("locators", {})
        locators = {key: value for key, value in all_locators.items() if key in selected_ids}
    if not isinstance(locators, Mapping) or set(locators) != selected_ids:
        raise SupplementaryError(f"{trajectory.key}: registered Support/C64 locator set is incomplete")
    result = runtime.prepare_commissioning(
        role_manifest=trajectory.role_manifest,
        private_locators=locators,
        condition_id=trajectory.condition,
        dataset_root=trajectory.bundle.get("dataset_root"))
    cache[key] = result
    return result


def _all_stage_a_seals_valid(trajectories: Sequence[Trajectory], output_root: Path) -> None:
    """Fail closed before any Stage B projector can open official outcomes."""
    for index, trajectory in enumerate(trajectories, 1):
        _receipt, rows, _groups = _load_sidecar_rows(trajectory, output_root)
        if len(rows) != trajectory.event_count:
            raise SupplementaryError(f"{trajectory.key}: supplementary Stage A coverage is incomplete")
        if index % 10 == 0 or index == len(trajectories):
            print(f"[stage-a-seal-gate] {index}/{len(trajectories)} supplementary seals verified",
                  flush=True)


def run_full_experiment(*, output_base: Path = OUT_ROOT) -> dict[str, Any]:
    """Execute all registered sidecars, then post-seal image-level evaluation."""
    wall_started = time.perf_counter()
    trajectories = discover_trajectories()
    output_root = _fresh_output_root(output_base)
    progress_path = output_root / "campaign_progress.jsonl"
    state_path = output_root / "RUN_STATE.json"
    sampler = ResourceSampler()
    sampler.start()
    run_state: dict[str, Any] = {
        "schema": "nshadow-independent-b3-c64.run-state.v1",
        "status": "RUNNING", "phase": "METADATA_PREFLIGHT",
        "output_root": str(output_root.relative_to(ROOT)),
        "source_git_head": _git_head(),
        "runner_source_sha256": sha256_file(__file__),
        "dataset_roots": {"realiad": str(REAL_DATA_ROOT), "visa_1cls": str(VISA_DATA_ROOT)},
        "registered_trajectory_count": 102, "registered_image_event_count": 345917,
        "stage_a_completed_trajectories": 0, "stage_b_completed_trajectories": 0,
        "test_outcomes_or_masks_read": False,
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_exclusive(state_path, run_state)
    append_fsync(progress_path, {"event": "SUPPLEMENTARY_CAMPAIGN_STARTED",
        "source_git_head": run_state["source_git_head"], "trajectory_count": 102,
        "image_event_count": 345917, "timestamp_local": run_state["created_local"]})
    try:
        if not REAL_DATA_ROOT.is_dir() or not VISA_DATA_ROOT.is_dir():
            raise SupplementaryError("researcher-specified Real-IAD or VisA dataset root is unavailable")
        run_state["phase"] = "VERIFYING_ALL_ORIGINAL_SEALS_AND_FEEDBACK_LEDGER_METADATA"
        write_progress(state_path, run_state)
        audit = preflight_assets(trajectories)
        audit_sha = write_exclusive(output_root / "ASSET_AND_REPLAY_AUDIT.json", audit)
        write_text_exclusive(output_root / "SUPPLEMENTARY_METHOD_CONTRACT.md",
            (ROOT / "docs/calibration/nshadow_independent_b3_c64/SUPPLEMENTARY_METHOD_CONTRACT.md")
            .read_text(encoding="utf-8"))
        append_fsync(progress_path, {"event": "ALL_ORIGINAL_INPUTS_VERIFIED",
            "asset_audit_sha256": audit_sha, "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

        from experiments.calibration.final_benchmark.native_backend import NativeInputEncoder
        runtime = NativeInputEncoder.load_authorized_runtime()
        runtime_identity = {
            "schema": "nshadow-independent-b3-c64.runtime-identity.v1",
            "source_git_head": _git_head(), "runner_source_sha256": sha256_file(__file__),
            "model_receipt": runtime.model_receipt,
            "native_config": runtime.cfg.__dict__ if hasattr(runtime.cfg, "__dict__") else repr(runtime.cfg),
            "dataset_roots": {"realiad": str(REAL_DATA_ROOT), "visa_1cls": str(VISA_DATA_ROOT)},
            "test_outcomes_or_masks_read": False,
        }
        write_exclusive(output_root / "RUNTIME_IDENTITY.json", runtime_identity)

        run_state.update(phase="SUPPLEMENTARY_STAGE_A_QUERY_EVENT_RECONSTRUCTION")
        write_progress(state_path, run_state)
        commissioning_cache: dict[str, dict[str, Any]] = {}
        active_dataset_category: tuple[str, str] | None = None
        completed_stage_a: list[dict[str, Any]] = []
        for index, trajectory in enumerate(trajectories, 1):
            current_category = (trajectory.dataset_id, trajectory.category)
            if active_dataset_category is not None and current_category != active_dataset_category:
                commissioning_cache.clear()
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            active_dataset_category = current_category
            commissioning = _commissioning_for(runtime, trajectory, commissioning_cache)
            result = _run_stage_a_trajectory(
                trajectory, runtime, commissioning, output_root, progress_path)
            completed_stage_a.append(result)
            run_state.update(stage_a_completed_trajectories=index,
                stage_a_completed_events=sum(int(item["event_count"]) for item in completed_stage_a),
                current_trajectory=trajectory.key,
                current_trajectory_status="SEALED_COMPLETE")
            write_progress(state_path, run_state)
            print(f"[stage-a] {index}/102 sealed {trajectory.key}: "
                  f"{trajectory.event_count} events, {result['runtime_seconds']:.1f}s", flush=True)
        commissioning_cache.clear()
        del runtime
        gc.collect()

        run_state.update(phase="VERIFYING_ALL_102_STAGE_A_SEALS_BEFORE_STAGE_B",
                         current_trajectory=None)
        write_progress(state_path, run_state)
        _all_stage_a_seals_valid(trajectories, output_root)
        append_fsync(progress_path, {"event": "ALL_102_STAGE_A_SEALS_VALIDATED",
            "official_outcomes_read": False, "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

        run_state.update(phase="POST_SEAL_STAGE_B_IMAGE_LEVEL_EVALUATION")
        write_progress(state_path, run_state)
        stage_b_results = []
        for index, trajectory in enumerate(trajectories, 1):
            result = _stage_b_one(trajectory, output_root)
            stage_b_results.append(result)
            run_state.update(stage_b_completed_trajectories=index,
                test_outcomes_or_masks_read=True,
                outcomes_read_only_after_all_stage_a_seals=True,
                current_trajectory=trajectory.key)
            write_progress(state_path, run_state)
            if index % 5 == 0 or index == len(trajectories):
                print(f"[stage-b] {index}/102 completed image-level projection/evaluation",
                      flush=True)
        elapsed = time.perf_counter() - wall_started
        resource = {"process_peak_rss_sampled_bytes": sampler.peak_rss_bytes}
        integrated = _aggregate_and_report(trajectories, output_root, audit, elapsed,
                                           sampler.peak_rss_bytes)
        run_state.update(status="SEALED_COMPLETE", phase="COMPLETE",
            stage_a_completed_trajectories=102, stage_a_completed_events=345917,
            stage_b_completed_trajectories=102,
            end_to_end_wall_clock_seconds=time.perf_counter() - wall_started,
            test_outcomes_or_masks_read=True, pixel_masks_accessed=False,
            asset_audit_sha256=audit_sha,
            aggregate_report_path=integrated["report_path"],
            completed_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_progress(state_path, run_state)
        append_fsync(progress_path, {"event": "SUPPLEMENTARY_CAMPAIGN_SEALED_COMPLETE",
            "trajectory_count": 102, "image_event_count": 345917,
            "test_outcomes_opened_after_all_stage_a_seals": True,
            "pixel_masks_accessed": False,
            "end_to_end_wall_clock_seconds": run_state["end_to_end_wall_clock_seconds"],
            "timestamp_local": run_state["completed_local"]})
        return {"status": "SEALED_COMPLETE", "output_root": str(output_root),
            "run_state": run_state, "integrated_results": integrated,
            "stage_a_trajectory_count": len(completed_stage_a),
            "stage_b_trajectory_count": len(stage_b_results), "resource": resource}
    except BaseException as exc:
        run_state.update(status="INCOMPLETE", error_type=type(exc).__name__,
            error_message=str(exc), traceback=traceback.format_exc(),
            failed_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            process_peak_rss_sampled_bytes=sampler.peak_rss_bytes)
        write_progress(state_path, run_state)
        append_fsync(progress_path, {"event": "SUPPLEMENTARY_CAMPAIGN_INCOMPLETE",
            "phase": run_state.get("phase"), "error_type": type(exc).__name__,
            "error_message": str(exc), "timestamp_local": run_state["failed_local"]})
        raise
    finally:
        sampler.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", action="store_true",
        help="execute all 102 registered supplementary trajectories and post-seal Stage B")
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT,
        help="base path for a new output tree; existing paths are never overwritten")
    args = parser.parse_args(argv)
    if not args.full_run:
        parser.error("--full-run is required; this module does not run partial scientific scopes")
    try:
        result = run_full_experiment(output_base=args.output_root)
    except BaseException as exc:
        print(f"SUPPLEMENTARY_EXPERIMENT_INCOMPLETE: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
    print(json.dumps({"status": result["status"], "output_root": result["output_root"],
        "report_path": result["integrated_results"]["report_path"],
        "realiad": result["integrated_results"]["real_iad"],
        "visa": result["integrated_results"]["visa"]}, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
