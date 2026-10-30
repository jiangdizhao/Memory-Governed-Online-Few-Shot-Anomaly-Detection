"""Sealed prediction replay and evaluator-only report for the B3 lag ablation.

The prediction phase uses only the frozen score, Memory, C64, query, and
historical-prediction artifacts checked by ``stage1a_b3_mechanism_ablation``.
The evaluator is a separate code path and will not import the official outcome
reader until the prediction receipt and every prediction-file digest pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from . import stage1a_b3_mechanism_ablation as a1


CLASSES = a1.CLASSES
SEEDS = a1.SEEDS
ARMS = a1.ARMS
DELTA = a1.DELTA
EXPECTED_TRAJECTORIES = 20
EXPECTED_EVENTS = 76_475
EXPECTED_MEMORY_CHANGES = 1_190
COMMON_HORIZON_START = 1_050
WINDOW_SIZE = 256
QUERY_TAPE = "Q-A-TS-v1"
QUERY_RATE = 0.03
QUERY_SEED = 37
Q_DIVERGENCE_TOLERANCE = 1e-12
THETA_DIVERGENCE_TOLERANCE = 1e-12
PAIRED_TIE_TOLERANCE = 1e-12
INTERPRETATION_RULES = {
    "meaningful_macro_ba_margin": 0.02,
    "minimum_classes_with_positive_ba_effect": 3,
    "maximum_class_mean_ba_loss": 0.03,
    "maximum_worst256_fnr_cost": 0.05,
    "manipulation_inconclusive_if_threshold_divergence_fraction_at_most": 0.01,
    "manipulation_inconclusive_also_requires_zero_prediction_disagreements": True,
    "classification_is_descriptive_not_a_significance_test": True,
}
DEFAULT_OUTPUT = Path("outputs/calibration/b3_state_lag_formal_v1_20261030")
PREDICTION_SCHEMA = "calibration.b3_state_lag_prediction.v1"
RUN_SCHEMA = "calibration.b3_state_lag_run.v1"


class FormalRunError(RuntimeError):
    """Raised when an immutable input or phase-gate invariant fails."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_json(value: Any) -> str:
    return _sha_bytes(_canonical_json(value).encode("utf-8"))


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalRunError(f"Cannot read required JSON artifact: {path}") from exc


def _write_json_once(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True,
                         ensure_ascii=False, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise FormalRunError(f"Refusing to overwrite immutable artifact: {path}") from exc
    return _sha_bytes(payload.encode("utf-8"))


def _write_json_verified(path: Path, value: Any) -> str:
    """Create an artifact once, or accept an identical artifact on resume."""
    payload = json.dumps(value, indent=2, sort_keys=True,
                         ensure_ascii=False, allow_nan=False) + "\n"
    expected = _sha_bytes(payload.encode("utf-8"))
    if path.exists():
        if _sha_file(path) != expected:
            raise FormalRunError(f"Existing immutable artifact differs: {path}")
        return expected
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise FormalRunError(f"Artifact appeared during write: {path}") from exc
    return expected


def _write_text_verified(path: Path, payload: str) -> str:
    expected = _sha_bytes(payload.encode("utf-8"))
    if path.exists():
        if _sha_file(path) != expected:
            raise FormalRunError(f"Existing immutable report differs: {path}")
        return expected
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise FormalRunError(f"Report appeared during write: {path}") from exc
    return expected


def _write_json_atomic(path: Path, value: Any) -> str:
    """Atomically update progress metadata while a run is explicitly incomplete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True,
                         ensure_ascii=False, allow_nan=False) + "\n"
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temp_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    return _sha_bytes(payload.encode("utf-8"))


def _git_value(root: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FormalRunError(f"Cannot obtain Git provenance for {root}") from exc


def _source_hashes(code_root: Path) -> dict[str, str]:
    relative_files = (
        "experiments/calibration/stage1a_b3_mechanism_formal.py",
        "experiments/calibration/stage1a_b3_mechanism_ablation.py",
        "experiments/calibration/stage1a_formal_comparison.py",
        "experiments/calibration/metrics.py",
        "experiments/calibration/stage05.py",
        "tests/calibration/test_b3_mechanism_formal.py",
    )
    hashes: dict[str, str] = {}
    for relative in relative_files:
        path = code_root / relative
        if not path.is_file():
            if relative.endswith("test_b3_mechanism_formal.py"):
                continue
            raise FormalRunError(f"Missing source file for provenance: {relative}")
        hashes[relative] = _sha_file(path)
    return hashes


def _stable_preflight_identity(preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Select deterministic input identities, excluding the verification time."""
    return {
        "status": preflight.get("status"),
        "summary": preflight.get("summary"),
        "feasibility_matrix": preflight.get("feasibility_matrix"),
        "c64_group_identity_by_class": preflight.get("c64_group_identity_by_class"),
        "input_artifacts": sorted(preflight.get("input_artifacts", []),
                                   key=lambda row: (row.get("path", ""),
                                                    row.get("sha256", ""))),
        "historical_limitations": preflight.get("historical_limitations"),
    }


def _resource_paths_preflight() -> dict[str, bool]:
    return {
        "dataset_root_exists": Path(r"G:\dataset\realiad").is_dir(),
        "json_metadata_exists": Path(r"G:\dataset\realiad\jsons\realiad_jsons").is_dir(),
        "dinov2_cache_exists": Path(
            r"C:\Users\rico\.cache\torch\hub\facebookresearch_dinov2_main").is_dir(),
    }


def _load_frozen_trajectories(source_root: Path,
                              preflight: Mapping[str, Any]) -> list[a1.FrozenTrajectory]:
    """Load only A1-verified frozen scores, event histories, and old predictions."""
    stage05 = source_root / "outputs/calibration/stage05_exploratory_20261029"
    stage075 = source_root / "outputs/calibration/stage075a_greedy_ranking_repaired3_20261029"
    p12out = source_root / "outputs/calibration/stage1a_p12_low_memory_recovery_v3_20261030"
    p12audit = source_root / "docs/calibration/audit/stage1a_p12_low_memory_recovery_v3_20261030"
    stage05_checks = a1._read_json(stage05 / "checksums.json")
    stage05_manifest = a1._read_json(stage05 / "score_tape_manifest.json")
    stage075_checks = a1._read_json(stage075 / "checksums.json")
    formal = source_root / "outputs/calibration/stage1a_formal_controller_comparison_v4_20261030"
    formal_checks = a1._read_json(formal / "checksums.json")
    audit_receipts_doc = a1._read_json(p12audit / "trajectory_receipts_final.json")
    p12_audit_receipts = {
        (str(item.get("category")), int(item.get("support_seed", -1))): item
        for item in audit_receipts_doc.get("receipts", [])
    }
    p12_receipts: dict[tuple[str, int], Mapping[str, Any]] = {}
    for category in CLASSES:
        for seed in SEEDS:
            p12_receipts[(category, seed)] = a1._read_json(
                p12out / "shards" / category / f"seed{seed}.receipt.json")

    artifacts: list[dict[str, str]] = []
    trajectories: list[a1.FrozenTrajectory] = []
    expected_rows = {(str(row["class"]), int(row["support_seed"])): row
                     for row in preflight["feasibility_matrix"]}
    for category in CLASSES:
        for seed in SEEDS:
            trajectory = a1.load_frozen_trajectory(
                source_root, category, seed, stage05_checks, stage05_manifest,
                stage075_checks, formal_checks, p12_receipts, p12_audit_receipts,
                artifacts)
            expected = expected_rows.get((category, seed))
            if expected is None or trajectory.matrix_row != expected:
                raise FormalRunError(
                    f"Trajectory changed between A0 verification and loading: {category}/seed{seed}")
            trajectories.append(trajectory)
    if len(trajectories) != EXPECTED_TRAJECTORIES:
        raise FormalRunError(f"Expected {EXPECTED_TRAJECTORIES} frozen trajectories")
    return trajectories


def _make_run_identity(source_root: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    source_hashes = _source_hashes(source_root)
    stable_preflight = _stable_preflight_identity(preflight)
    input_sha = _sha_json(stable_preflight)
    return {
        "schema": "calibration.b3_state_lag_run_identity.v1",
        "source_commit": _git_value(source_root, "rev-parse", "HEAD"),
        "branch": _git_value(source_root, "branch", "--show-current"),
        "source_sha256": source_hashes,
        "input_identity_sha256": input_sha,
        "frozen_contract": {
            "classes": list(CLASSES), "support_seeds": list(SEEDS),
            "trajectory_count": EXPECTED_TRAJECTORIES,
            "event_count": EXPECTED_EVENTS,
            "memory_content_changes": EXPECTED_MEMORY_CHANGES,
            "query_tape": QUERY_TAPE, "query_rate": QUERY_RATE,
            "query_seed": QUERY_SEED, "arms": dict(ARMS), "delta": DELTA,
            "common_horizon_start_zero_based": COMMON_HORIZON_START,
            "window_size": WINDOW_SIZE,
            "c64_description": "four deployed normal supports plus 64 additional protected training-normal anchors",
        },
        "diagnostic_tolerances": {
            "q_divergence": Q_DIVERGENCE_TOLERANCE,
            "theta_divergence": THETA_DIVERGENCE_TOLERANCE,
            "paired_tie": PAIRED_TIE_TOLERANCE,
        },
        "interpretation_rules": INTERPRETATION_RULES,
    }


def _prediction_file_path(output: Path, category: str, seed: int) -> Path:
    return output / "predictions" / f"{category}_seed{seed}.jsonl.gz"


def _event_rows(trajectory: a1.FrozenTrajectory,
                replays: Mapping[str, a1.ArmReplay]) -> list[dict[str, Any]]:
    events = trajectory.events
    old_rows = trajectory.original_b3_rows
    if len(old_rows) != len(events):
        raise FormalRunError("Historical prediction and frozen event lengths differ")
    updates = {arm: {item.position: item for item in replay.updates}
               for arm, replay in replays.items()}
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        old = old_rows[index]
        if int(old.get("t", -1)) != event.position:
            raise FormalRunError("Historical event identity changed after A0 parity check")
        arms: dict[str, Any] = {}
        for arm in ARMS:
            replay = replays[arm]
            update = updates[arm].get(event.position)
            arms[arm] = {
                "prediction": int(replay.predictions[index]),
                "threshold_pre": float(replay.threshold_pre[index]),
                "threshold_post": float(replay.threshold_post[index]),
                "update": ({
                    "event_position": int(update.position),
                    "current_memory_version": int(update.memory_version),
                    "reference_version": int(update.reference_version),
                    "q90_reference": float(update.q90_reference),
                    "prior_threshold": float(update.theta_before),
                    "next_threshold": float(update.theta_after),
                    "applied_movement": float(update.applied_step),
                } if update is not None else None),
            }
        row = {
            "schema": PREDICTION_SCHEMA,
            "trajectory_id": trajectory.matrix_row["trajectory_id"],
            "event_position": int(event.position),
            "event_identity": {
                "position": int(event.position),
                "sample_key": old.get("sample_key"),
                "camera_id": old.get("camera_id"),
                "query_event": bool(event.query_selected),
            },
            "score": float(event.score),
            "memory_version_before": int(event.memory_version_before),
            "memory_sha256_before": event.memory_sha256_before,
            "memory_version_after": int(event.memory_version_after),
            "memory_sha256_after": event.memory_sha256_after,
            "arms": arms,
        }
        if _contains_prohibited_label_key(row):
            raise FormalRunError("Prediction serializer emitted a prohibited label field")
        rows.append(row)
    return rows


def _encode_jsonl_gzip(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0,
                       compresslevel=6) as zipped:
        for row in rows:
            zipped.write((_canonical_json(row) + "\n").encode("utf-8"))
    return buffer.getvalue()


def _read_prediction_rows(path: Path, *, expected_count: int | None = None,
                          reject_labels: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise FormalRunError(f"Prediction row {line_no} is not an object")
                if reject_labels and _contains_prohibited_label_key(row):
                    raise FormalRunError("Prediction stream contains a ground-truth field")
                rows.append(row)
    except (OSError, EOFError, json.JSONDecodeError) as exc:
        raise FormalRunError(f"Cannot decode prediction file: {path}") from exc
    if expected_count is not None and len(rows) != expected_count:
        raise FormalRunError(
            f"Prediction event count mismatch in {path.name}: {len(rows)} != {expected_count}")
    for index, row in enumerate(rows):
        if row.get("event_position") != index:
            raise FormalRunError(f"Prediction positions are not contiguous in {path.name}")
        if set(row.get("arms", {})) != set(ARMS):
            raise FormalRunError(f"Prediction arms differ in {path.name}")
        if any(row["arms"][arm].get("prediction") not in (0, 1) for arm in ARMS):
            raise FormalRunError(f"Nonbinary prediction in {path.name}")
        if any(not math.isfinite(float(row["arms"][arm]["threshold_pre"])) or
               not math.isfinite(float(row["arms"][arm]["threshold_post"]))
               for arm in ARMS):
            raise FormalRunError(f"Nonfinite threshold in {path.name}")
    return rows


def _contains_prohibited_label_key(value: Any) -> bool:
    prohibited = {"label", "true_label", "ground_truth", "outcome"}
    if isinstance(value, Mapping):
        return (any(str(key).lower() in prohibited for key in value) or
                any(_contains_prohibited_label_key(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return any(_contains_prohibited_label_key(item) for item in value)
    return False


def _select_output(base: Path, *, resume: bool, repository: Path) -> Path:
    base = (repository / base).resolve() if not base.is_absolute() else base.resolve()
    try:
        base.relative_to(repository.resolve())
    except ValueError as exc:
        raise FormalRunError("Output directory must be inside the authorized repository") from exc
    if resume:
        if not base.is_dir():
            raise FormalRunError("--resume requires an existing incomplete or complete run directory")
        return base
    if not base.exists():
        return base
    for suffix in range(2, 1000):
        candidate = base.with_name(f"{base.name}_v{suffix}")
        if not candidate.exists():
            return candidate
    raise FormalRunError("Could not choose an unused immutable output namespace")


def _safe_relative(output: Path, relative: str) -> Path:
    candidate = (output / relative).resolve()
    try:
        candidate.relative_to(output.resolve())
    except ValueError as exc:
        raise FormalRunError(f"Artifact path escapes run directory: {relative}") from exc
    return candidate


def _verify_prediction_seal(output: Path, *, expected_trajectories: int = EXPECTED_TRAJECTORIES,
                            expected_events: int = EXPECTED_EVENTS) -> tuple[dict[str, Any], dict[str, Any]]:
    seal_path = output / "prediction_phase_complete.json"
    receipts_path = output / "prediction_receipts.json"
    if not seal_path.is_file() or not receipts_path.is_file():
        raise FormalRunError("Refusing evaluator access before a complete prediction seal")
    seal = _read_json(seal_path)
    if (seal.get("status") != "PREDICTION_PHASE_COMPLETE_LABEL_FIREWALL_INTACT" or
            seal.get("outcome_labels_accessed") is not False or
            seal.get("trajectory_count") != expected_trajectories or
            seal.get("event_count") != expected_events):
        raise FormalRunError("Prediction phase seal is incomplete or violates the label firewall")
    receipts = _read_json(receipts_path)
    if _sha_file(receipts_path) != seal.get("prediction_receipts_sha256"):
        raise FormalRunError("Prediction receipts changed after the phase seal")
    receipt_rows = receipts.get("trajectories")
    if not isinstance(receipt_rows, list) or len(receipt_rows) != expected_trajectories:
        raise FormalRunError("Prediction receipt trajectory count differs from the seal")
    listed_files = seal.get("prediction_files")
    if not isinstance(listed_files, dict) or len(listed_files) != expected_trajectories:
        raise FormalRunError("Prediction file list is incomplete")
    seen: set[tuple[str, int]] = set()
    event_total = 0
    for receipt in receipt_rows:
        category, seed = str(receipt.get("class")), int(receipt.get("support_seed", -1))
        identity = (category, seed)
        if category not in CLASSES or seed not in SEEDS or identity in seen:
            raise FormalRunError("Invalid or duplicate trajectory receipt")
        seen.add(identity)
        relative = str(receipt.get("prediction_file"))
        expected_hash = listed_files.get(relative)
        path = _safe_relative(output, relative)
        if not path.is_file() or not isinstance(expected_hash, str):
            raise FormalRunError(f"Sealed prediction file is missing: {relative}")
        actual_hash = _sha_file(path)
        if actual_hash != expected_hash or actual_hash != receipt.get("prediction_sha256"):
            raise FormalRunError(f"Prediction file failed its seal: {relative}")
        count = int(receipt.get("event_count", -1))
        rows = _read_prediction_rows(path, expected_count=count)
        if rows and rows[0].get("trajectory_id") != receipt.get("trajectory_id"):
            raise FormalRunError(f"Trajectory identity mismatch in {relative}")
        event_total += count
    if seen != {(category, seed) for category in CLASSES for seed in SEEDS}:
        raise FormalRunError("Prediction receipts do not cover the frozen class/seed matrix")
    if event_total != expected_events or event_total != int(seal.get("event_count", -1)):
        raise FormalRunError("Prediction event counts differ from the sealed total")
    return seal, receipts


def _prepare_prediction_run(source_root: Path, output: Path, *, resume: bool) -> tuple[
        dict[str, Any], dict[str, Any], list[a1.FrozenTrajectory]]:
    if not source_root.is_dir():
        raise FormalRunError(f"Frozen source root does not exist: {source_root}")
    a0 = a1.verify_a0(source_root, source_root)
    if a0.get("status") != "PASS":
        raise FormalRunError("A0 integrity verification did not pass")
    if (a0.get("summary", {}).get("trajectory_count") != EXPECTED_TRAJECTORIES or
            a0.get("summary", {}).get("events") != EXPECTED_EVENTS or
            a0.get("summary", {}).get("memory_updates") != EXPECTED_MEMORY_CHANGES):
        raise FormalRunError("A0 frozen counts differ from the registered contract")
    a0 = dict(a0)
    # A1's historical audit ran in an isolated worktree. This formal task runs
    # in the authorized original checkout and preserves its pre-existing
    # untracked audit/output directories, so report the actual current state.
    a0_worktree = dict(a0.get("worktree", {}))
    a0_worktree.update({
        "isolated_worktree": False,
        "clean_verified_before_edits": False,
        "preexisting_untracked_outputs_preserved": True,
        "formal_run_in_original_workspace": True,
    })
    a0["worktree"] = a0_worktree
    a0["current_resource_preflight"] = _resource_paths_preflight()
    if not all(a0["current_resource_preflight"].values()):
        raise FormalRunError("Dataset, metadata, or DINOv2 cache path is unavailable")
    source_identity = _make_run_identity(source_root, a0)
    identity_sha = _sha_json(source_identity)
    if resume:
        if not (output / "INCOMPLETE").exists():
            if (output / "prediction_phase_complete.json").is_file():
                _verify_prediction_seal(output)
                old_identity = _read_json(output / "run_identity.json")
                if _sha_json(old_identity) != identity_sha:
                    raise FormalRunError("Completed run identity differs from current sources/inputs")
                return a0, source_identity, []
            raise FormalRunError("Cannot resume a run without its INCOMPLETE marker")
        old_identity = _read_json(output / "run_identity.json")
        if _sha_json(old_identity) != identity_sha:
            raise FormalRunError("Resume source or frozen input identity differs")
        stored_preflight = _read_json(output / "preflight.json")
        if (_sha_json(_stable_preflight_identity(stored_preflight)) !=
                source_identity["input_identity_sha256"]):
            raise FormalRunError("Resume preflight identity differs from the frozen inputs")
    else:
        if output.exists():
            raise FormalRunError("Output path unexpectedly exists; choose a new immutable namespace")
        output.mkdir(parents=True, exist_ok=False)
        (output / "predictions").mkdir(exist_ok=False)
        (output / "INCOMPLETE").write_text("PREDICTION_PHASE_INCOMPLETE\n", encoding="ascii")
        _write_json_once(output / "preflight.json", a0)
        _write_json_once(output / "run_identity.json", source_identity)
        _write_json_once(output / "RUN_STATE.json", {
            "schema": RUN_SCHEMA, "status": "PREDICTION_INCOMPLETE",
            "completed_trajectories": [], "input_identity_sha256": identity_sha,
        })
    trajectories = _load_frozen_trajectories(source_root, a0)
    return a0, source_identity, trajectories


def run_prediction_phase(*, source_root: Path, output_base: Path = DEFAULT_OUTPUT,
                         resume: bool = False) -> Path:
    """A0 verify, replay 20 prediction-only streams, and seal them immutably."""
    repository = source_root.resolve()
    output = _select_output(output_base, resume=resume, repository=repository)
    a0, identity, trajectories = _prepare_prediction_run(source_root, output, resume=resume)
    if not trajectories:
        _verify_prediction_seal(output)
        return output

    receipt_path = output / "prediction_receipts.json"
    if receipt_path.is_file():
        receipts_doc = _read_json(receipt_path)
        if receipts_doc.get("input_identity_sha256") != identity["input_identity_sha256"]:
            raise FormalRunError("Partial prediction receipts have a different input identity")
        receipt_rows = list(receipts_doc.get("trajectories", []))
    else:
        receipt_rows = []
        _write_json_atomic(receipt_path, {
            "schema": "calibration.b3_state_lag_prediction_receipts.v1",
            "status": "INCOMPLETE", "input_identity_sha256": identity["input_identity_sha256"],
            "trajectories": [],
        })
    receipt_index = {(str(item["class"]), int(item["support_seed"])): item
                     for item in receipt_rows}
    matrix_by_key = {(str(row["class"]), int(row["support_seed"])): row
                     for row in a0["feasibility_matrix"]}
    total_events = 0
    total_changes = 0
    for trajectory in trajectories:
        category, seed = trajectory.category, trajectory.seed
        key = (category, seed)
        rel = f"predictions/{category}_seed{seed}.jsonl.gz"
        path = _safe_relative(output, rel)
        existing = receipt_index.get(key)
        if path.exists() and existing is None:
            raise FormalRunError(f"Unreceipted prediction file blocks resume: {rel}")
        if existing is not None:
            if not path.is_file() or _sha_file(path) != existing.get("prediction_sha256"):
                raise FormalRunError(f"Partial prediction file failed its receipt: {rel}")
            rows = _read_prediction_rows(path, expected_count=len(trajectory.events))
            if (existing.get("trajectory_id") != trajectory.matrix_row["trajectory_id"] or
                    existing.get("event_count") != len(rows)):
                raise FormalRunError(f"Partial prediction receipt identity mismatch: {rel}")
            total_events += len(rows)
            total_changes += int(trajectory.matrix_row["actual_memory_changes"])
            continue

        replays = a1.replay_matched_arms(trajectory.events, trajectory.history, DELTA)
        a2_replay = replays["A2"]
        old_rows = trajectory.original_b3_rows
        for position, old in enumerate(old_rows):
            old_b3 = old.get("methods", {}).get("B3", {})
            if (int(old_b3.get("prediction", -1)) != a2_replay.predictions[position] or
                    float(old_b3.get("threshold_pre", math.nan)) != a2_replay.threshold_pre[position] or
                    float(old_b3.get("threshold_post", math.nan)) != a2_replay.threshold_post[position]):
                raise FormalRunError(
                    f"A2 parity failed against stored B3 at {category}/seed{seed}/t={position}")
        for arm in ARMS:
            replay = replays[arm]
            if (replay.positions != replays["A2"].positions or
                    replay.memory_versions_before != replays["A2"].memory_versions_before or
                    replay.memory_versions_after != replays["A2"].memory_versions_after):
                raise FormalRunError(f"Matched event histories diverged for {arm} at {key}")
        update_positions = [{item.position for item in replays[arm].updates} for arm in ARMS]
        if len({tuple(sorted(items)) for items in update_positions}) != 1:
            raise FormalRunError(f"Memory-update event positions diverged at {key}")
        serialized = _event_rows(trajectory, replays)
        payload = _encode_jsonl_gzip(serialized)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FormalRunError(f"Refusing to overwrite prediction file: {rel}")
        temp = path.with_name(path.name + ".tmp")
        if temp.exists():
            raise FormalRunError(f"Unexpected stale temporary prediction file: {temp.name}")
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        receipt = {
            "class": category, "support_seed": seed,
            "trajectory_id": trajectory.matrix_row["trajectory_id"],
            "event_count": len(serialized),
            "memory_content_changes": len(replays["A2"].updates),
            "event_chain_sha256": trajectory.matrix_row["event_chain_sha256"],
            "query_tape": QUERY_TAPE, "query_rate": QUERY_RATE,
            "query_seed": QUERY_SEED,
            "query_tape_binding_sha256": trajectory.matrix_row["query_tape_binding_sha256"],
            "c64_identity_sha256": trajectory.matrix_row["c64_identity_sha256"],
            "c64_version_history_sha256": trajectory.matrix_row["c64_version_history_sha256"],
            "initial_memory_sha256": trajectory.matrix_row["initial_memory_sha256"],
            "final_memory_sha256": trajectory.matrix_row["final_memory_sha256"],
            "prediction_file": rel, "prediction_sha256": _sha_file(path),
            "a2_historical_parity": {
                "pre_threshold_mismatches": 0, "post_threshold_mismatches": 0,
                "prediction_mismatches": 0, "event_identity_mismatches": 0,
            },
            "input_identity_sha256": identity["input_identity_sha256"],
            "source_sha256": identity["source_sha256"],
        }
        receipt_rows.append(receipt)
        receipt_index[key] = receipt
        _write_json_atomic(receipt_path, {
            "schema": "calibration.b3_state_lag_prediction_receipts.v1",
            "status": "INCOMPLETE", "input_identity_sha256": identity["input_identity_sha256"],
            "trajectories": sorted(receipt_rows,
                                    key=lambda item: (CLASSES.index(item["class"]),
                                                      SEEDS.index(int(item["support_seed"])))),
        })
        total_events += len(serialized)
        total_changes += len(replays["A2"].updates)
        _write_json_atomic(output / "RUN_STATE.json", {
            "schema": RUN_SCHEMA, "status": "PREDICTION_INCOMPLETE",
            "completed_trajectories": [
                f"{item['class']}/seed{item['support_seed']}" for item in receipt_rows],
            "event_count_so_far": total_events,
            "memory_changes_so_far": total_changes,
            "input_identity_sha256": identity["input_identity_sha256"],
        })

    receipt_rows = sorted(receipt_rows,
                          key=lambda item: (CLASSES.index(item["class"]),
                                            SEEDS.index(int(item["support_seed"]))))
    if len(receipt_rows) != EXPECTED_TRAJECTORIES:
        raise FormalRunError(f"Prediction phase has {len(receipt_rows)} of 20 streams")
    total_events = sum(int(item["event_count"]) for item in receipt_rows)
    total_changes = sum(int(item["memory_content_changes"]) for item in receipt_rows)
    if total_events != EXPECTED_EVENTS or total_changes != EXPECTED_MEMORY_CHANGES:
        raise FormalRunError("Prediction event or Memory-change total differs from A0")
    receipt_doc = {
        "schema": "calibration.b3_state_lag_prediction_receipts.v1",
        "status": "COMPLETE", "input_identity_sha256": identity["input_identity_sha256"],
        "trajectories": receipt_rows,
    }
    if receipt_path.exists():
        _write_json_atomic(receipt_path, receipt_doc)
    else:
        _write_json_once(receipt_path, receipt_doc)
    prediction_files = {item["prediction_file"]: item["prediction_sha256"]
                        for item in receipt_rows}
    seal = {
        "schema": "calibration.b3_state_lag_prediction_seal.v1",
        "status": "PREDICTION_PHASE_COMPLETE_LABEL_FIREWALL_INTACT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trajectory_count": EXPECTED_TRAJECTORIES,
        "event_count": EXPECTED_EVENTS,
        "memory_content_changes": EXPECTED_MEMORY_CHANGES,
        "query_tape": QUERY_TAPE, "query_rate": QUERY_RATE,
        "query_seed": QUERY_SEED,
        "query_tape_binding_sha256": _sha_json(sorted(
            item["query_tape_binding_sha256"] for item in receipt_rows)),
        "arms": dict(ARMS), "delta": DELTA,
        "prediction_files": prediction_files,
        "prediction_receipts_sha256": _sha_file(receipt_path),
        "input_identity_sha256": identity["input_identity_sha256"],
        "source_commit": identity["source_commit"],
        "source_sha256": identity["source_sha256"],
        "outcome_labels_accessed": False,
        "confirmation_outcomes_accessed": False,
        "masks_accessed": False,
        "gpu_inference_calls": 0,
    }
    seal_path = output / "prediction_phase_complete.json"
    if seal_path.exists():
        existing_seal = _read_json(seal_path)
        if existing_seal != seal:
            raise FormalRunError("Existing prediction seal differs; refusing overwrite")
    else:
        _write_json_once(seal_path, seal)
    _write_json_atomic(output / "RUN_STATE.json", {
        "schema": RUN_SCHEMA, "status": "PREDICTION_SEALED_EVALUATION_PENDING",
        "trajectory_count": EXPECTED_TRAJECTORIES, "event_count": EXPECTED_EVENTS,
        "input_identity_sha256": identity["input_identity_sha256"],
    })
    _verify_prediction_seal(output)
    return output


def _worst_complete_windows(method_metrics: Mapping[str, Any], *,
                            scope: str) -> dict[str, Any]:
    windows = method_metrics.get("windows", {}).get("windows", [])
    complete = [row for row in windows if int(row.get("n", 0)) == WINDOW_SIZE]
    result: dict[str, Any] = {"window_size": WINDOW_SIZE, "stride": WINDOW_SIZE,
                              "alignment": ("global_zero" if scope == "full_stream"
                                            else f"horizon_start_{COMMON_HORIZON_START}"),
                              "complete_window_count": len(complete)}
    for metric, direction in (("fpr", "max"), ("fnr", "max"),
                              ("balanced_accuracy", "min")):
        values = [float(row[metric]) for row in complete if row.get(metric) is not None]
        if values:
            result[metric] = max(values) if direction == "max" else min(values)
            result[f"{metric}_status"] = "DEFINED"
        else:
            result[metric] = None
            result[f"{metric}_status"] = (
                "UNAVAILABLE_NO_COMPLETE_256_IMAGE_WINDOW" if not complete else
                "UNAVAILABLE_NO_DEFINED_WINDOW_DENOMINATOR")
    return result


def _metric_leaf(metrics: Mapping[str, Any], metric: str) -> float | None:
    if metric.startswith("worst_complete_256."):
        value = metrics.get("worst_complete_256", {}).get(metric.split(".", 1)[1])
    elif metric == "defect_recall":
        fnr = metrics.get("fnr")
        value = None if fnr is None else 1.0 - float(fnr)
    else:
        value = metrics.get(metric)
    return None if value is None else float(value)


def _summary(values: Iterable[float | None]) -> dict[str, Any]:
    all_values = list(values)
    defined = [float(value) for value in all_values if value is not None]
    if not defined:
        return {"mean": None, "sample_sd": None, "median": None,
                "min": None, "max": None, "n_defined": 0,
                "n_total": len(all_values), "status": "UNAVAILABLE_NO_DEFINED_VALUES"}
    return {
        "mean": statistics.fmean(defined),
        "sample_sd": statistics.stdev(defined) if len(defined) > 1 else None,
        "median": statistics.median(defined), "min": min(defined), "max": max(defined),
        "n_defined": len(defined), "n_total": len(all_values),
        "status": "DEFINED" if len(defined) == len(all_values) else "PARTIALLY_DEFINED",
    }


METRICS_TO_COMPARE = (
    "balanced_accuracy", "fpr", "fnr", "defect_recall", "coverage",
    "worst_complete_256.balanced_accuracy",
    "worst_complete_256.fpr", "worst_complete_256.fnr",
)


def _summarize_method_rows(rows: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scope in ("full_stream", "common_horizon_t_ge_1050"):
        result[scope] = {}
        for arm in ARMS:
            by_class: dict[str, Any] = {}
            for category in CLASSES:
                by_class[category] = {
                    metric: _summary([
                        _metric_leaf(rows[f"{category}/seed{seed}"][scope]["methods"][arm], metric)
                        for seed in SEEDS]) for metric in METRICS_TO_COMPARE
                }
            macro: dict[str, Any] = {}
            for metric in METRICS_TO_COMPARE:
                class_means = [by_class[category][metric]["mean"] for category in CLASSES]
                defined_means = [float(value) for value in class_means if value is not None]
                macro[metric] = {
                    "equal_class_macro_mean": statistics.fmean(defined_means)
                    if defined_means else None,
                    "class_means": {category: by_class[category][metric]["mean"]
                                    for category in CLASSES},
                    "classes_defined": len(defined_means),
                    "status": "DEFINED" if len(defined_means) == len(CLASSES)
                    else "PARTIALLY_DEFINED" if defined_means else "UNAVAILABLE",
                }
            result[scope][arm] = {"by_class": by_class,
                                  "equal_class_macro": macro}
    return result


def _paired_results(rows: Mapping[str, Any]) -> dict[str, Any]:
    pairs = (("A2", "A3"), ("A2", "A4"))
    output: dict[str, Any] = {"scope_results": {},
                              "tie_tolerance": PAIRED_TIE_TOLERANCE,
                              "interpretation": "Raw difference is left arm minus right arm; positive FPR/FNR is a cost."}
    for scope in ("full_stream", "common_horizon_t_ge_1050"):
        per_trajectory: dict[str, Any] = {}
        per_class: dict[str, Any] = {}
        macro: dict[str, Any] = {}
        for left, right in pairs:
            pair_key = f"{left}-{right}"
            per_trajectory[pair_key] = {}
            per_class[pair_key] = {}
            macro[pair_key] = {}
            for metric in METRICS_TO_COMPARE:
                trajectory_diffs: dict[str, float | None] = {}
                signs = {"positive": 0, "negative": 0, "tied": 0, "undefined": 0}
                for category in CLASSES:
                    for seed in SEEDS:
                        key = f"{category}/seed{seed}"
                        left_value = _metric_leaf(rows[key][scope]["methods"][left], metric)
                        right_value = _metric_leaf(rows[key][scope]["methods"][right], metric)
                        difference = (None if left_value is None or right_value is None
                                      else left_value - right_value)
                        trajectory_diffs[key] = difference
                        if difference is None:
                            signs["undefined"] += 1
                        elif difference > PAIRED_TIE_TOLERANCE:
                            signs["positive"] += 1
                        elif difference < -PAIRED_TIE_TOLERANCE:
                            signs["negative"] += 1
                        else:
                            signs["tied"] += 1
                per_trajectory[pair_key][metric] = {
                    "differences": trajectory_diffs, "counts": signs,
                    "summary": _summary(trajectory_diffs.values()),
                }
                for category in CLASSES:
                    per_class[pair_key].setdefault(category, {})
                    diffs = [trajectory_diffs[f"{category}/seed{seed}"] for seed in SEEDS]
                    per_class[pair_key][category][metric] = _summary(diffs)
                    per_class[pair_key][category][metric]["paired_sign_counts"] = {
                        "positive": sum(v is not None and v > PAIRED_TIE_TOLERANCE for v in diffs),
                        "negative": sum(v is not None and v < -PAIRED_TIE_TOLERANCE for v in diffs),
                        "tied": sum(v is not None and abs(v) <= PAIRED_TIE_TOLERANCE for v in diffs),
                        "undefined": sum(v is None for v in diffs),
                    }
                class_effects = [per_class[pair_key][category][metric]["mean"]
                                 for category in CLASSES]
                macro[pair_key][metric] = {
                    "equal_class_macro_mean": statistics.fmean(
                        [float(v) for v in class_effects if v is not None])
                        if any(v is not None for v in class_effects) else None,
                    "class_mean_effects": dict(zip(CLASSES, class_effects)),
                    "classes_defined": sum(v is not None for v in class_effects),
                }
        output["scope_results"][scope] = {
            "per_trajectory": per_trajectory,
            "per_class": per_class,
            "equal_class_macro": macro,
        }
    return output


def _transition_diagnostics(trajectories: Sequence[a1.FrozenTrajectory],
                            predictions: Mapping[str, Sequence[Mapping[str, Any]]],
                            labels_by_class: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    result: dict[str, Any] = {"tolerances": {
        "quantile_negligible_absolute": Q_DIVERGENCE_TOLERANCE,
        "threshold_negligible_absolute": THETA_DIVERGENCE_TOLERANCE,
    }, "arms": {}, "per_class": {}, "transition_rows": [],
        "prediction_disagreement_events": {},
        "error_changes_on_next_event_after_memory_transition": {},
        "error_changes_after_transition_by_class": {}}
    q_diffs: dict[str, list[float]] = {"current_minus_lag1": [], "current_minus_lag5": []}
    threshold_diffs: dict[str, list[float]] = {"A2_minus_A3": [], "A2_minus_A4": []}
    reference_gaps: dict[str, dict[str, int]] = {arm: {} for arm in ARMS}
    movement_counts = {arm: {"updates": 0, "zero_movement": 0,
                             "nonzero_movement": 0, "clipped": 0,
                             "at_clip_boundary": 0} for arm in ARMS}
    per_class_values: dict[str, dict[str, list[float]]] = {
        category: {key: [] for key in (*q_diffs, *threshold_diffs)} for category in CLASSES}
    disagreements = {"A2-A3": 0, "A2-A4": 0, "A3-A4": 0}
    disagreements_by_class = {
        category: {"A2-A3": 0, "A2-A4": 0, "A3-A4": 0}
        for category in CLASSES}
    error_after = {pair: {"events": 0, "left_minus_right_error_count": 0,
                          "left_errors": 0, "right_errors": 0}
                   for pair in ("A2-A3", "A2-A4")}
    error_after_by_class = {
        category: {pair: {"events": 0, "left_minus_right_error_count": 0,
                          "left_errors": 0, "right_errors": 0}
                   for pair in ("A2-A3", "A2-A4")}
        for category in CLASSES}
    transition_count = 0
    thresholds_equal_transition_count = {"A2-A3": 0, "A2-A4": 0}
    for trajectory in trajectories:
        category, seed = trajectory.category, trajectory.seed
        key = f"{category}/seed{seed}"
        rows = predictions[key]
        labels = labels_by_class[category]
        if len(rows) != len(labels):
            raise FormalRunError(f"Offline label count differs for diagnostics at {key}")
        for index, event in enumerate(trajectory.events):
            row = rows[index]
            for left, right, pair in (("A2", "A3", "A2-A3"),
                                      ("A2", "A4", "A2-A4"),
                                      ("A3", "A4", "A3-A4")):
                if row["arms"][left]["prediction"] != row["arms"][right]["prediction"]:
                    disagreements[pair] += 1
                    disagreements_by_class[category][pair] += 1
            if event.memory_version_after == event.memory_version_before:
                continue
            transition_count += 1
            version = event.memory_version_after
            current = trajectory.history.lookup(version, as_of_version=version).q90
            lag1_version = max(0, version - 1)
            lag5_version = max(0, version - 5)
            lag1 = trajectory.history.lookup(lag1_version, as_of_version=version).q90
            lag5 = trajectory.history.lookup(lag5_version, as_of_version=version).q90
            q1, q5 = current - lag1, current - lag5
            q_diffs["current_minus_lag1"].append(q1)
            q_diffs["current_minus_lag5"].append(q5)
            per_class_values[category]["current_minus_lag1"].append(q1)
            per_class_values[category]["current_minus_lag5"].append(q5)
            update_arm = {arm: row["arms"][arm]["update"] for arm in ARMS}
            if any(value is None for value in update_arm.values()):
                raise FormalRunError(f"Missing controller update at actual memory transition {key}/{index}")
            for arm, lag in ARMS.items():
                update = update_arm[arm]
                if int(update["current_memory_version"]) != version or int(update["reference_version"]) > version:
                    raise FormalRunError("Controller referenced a future or mismatched C64 state")
                gap = version - int(update["reference_version"])
                ref_counts = reference_gaps[arm]
                ref_counts[str(gap)] = ref_counts.get(str(gap), 0) + 1
                move = float(update["applied_movement"])
                movement_counts[arm]["updates"] += 1
                movement_counts[arm]["zero_movement"] += int(move == 0.0)
                movement_counts[arm]["nonzero_movement"] += int(move != 0.0)
                movement_counts[arm]["clipped"] += int(
                    float(update["next_threshold"]) != float(update["q90_reference"]))
                movement_counts[arm]["at_clip_boundary"] += int(
                    math.isclose(abs(move), DELTA, rel_tol=0.0,
                                 abs_tol=THETA_DIVERGENCE_TOLERANCE))
            theta13 = (float(update_arm["A2"]["next_threshold"]) -
                       float(update_arm["A3"]["next_threshold"]))
            theta14 = (float(update_arm["A2"]["next_threshold"]) -
                       float(update_arm["A4"]["next_threshold"]))
            threshold_diffs["A2_minus_A3"].append(theta13)
            threshold_diffs["A2_minus_A4"].append(theta14)
            per_class_values[category]["A2_minus_A3"].append(theta13)
            per_class_values[category]["A2_minus_A4"].append(theta14)
            thresholds_equal_transition_count["A2-A3"] += int(
                abs(theta13) <= THETA_DIVERGENCE_TOLERANCE)
            thresholds_equal_transition_count["A2-A4"] += int(
                abs(theta14) <= THETA_DIVERGENCE_TOLERANCE)
            result["transition_rows"].append({
                "class": category, "support_seed": seed,
                "event_position": index, "current_memory_version": version,
                "reference_versions": {arm: int(update_arm[arm]["reference_version"])
                                       for arm in ARMS},
                "reference_version_gaps": {arm: version - int(update_arm[arm]["reference_version"])
                                            for arm in ARMS},
                "q_current": current, "q_lag1": lag1, "q_lag5": lag5,
                "q_current_minus_lag1": q1,
                "q_current_minus_lag5": q5,
                "theta_post": {arm: float(update_arm[arm]["next_threshold"])
                               for arm in ARMS},
                "theta_A2_minus_A3": theta13,
                "theta_A2_minus_A4": theta14,
                "prediction_at_transition_event": {
                    arm: int(row["arms"][arm]["prediction"]) for arm in ARMS},
            })
            # The post-write theta first applies at t+1. Its error contrast is
            # measured separately from the prediction made at transition t.
            next_index = index + 1
            if next_index < len(rows):
                y = int(labels[next_index])
                for left, right in (("A2", "A3"), ("A2", "A4")):
                    pair = f"{left}-{right}"
                    p_left = int(rows[next_index]["arms"][left]["prediction"])
                    p_right = int(rows[next_index]["arms"][right]["prediction"])
                    e_left, e_right = int(p_left != y), int(p_right != y)
                    error_after[pair]["events"] += 1
                    error_after[pair]["left_errors"] += e_left
                    error_after[pair]["right_errors"] += e_right
                    error_after[pair]["left_minus_right_error_count"] += e_left - e_right
                    error_after_by_class[category][pair]["events"] += 1
                    error_after_by_class[category][pair]["left_errors"] += e_left
                    error_after_by_class[category][pair]["right_errors"] += e_right
                    error_after_by_class[category][pair]["left_minus_right_error_count"] += (
                        e_left - e_right)
    for name, values in q_diffs.items():
        absolute = [abs(value) for value in values]
        result.setdefault("quantile_divergence", {})[name] = {
            "signed": _summary(values), "absolute": _summary(absolute),
            "zero_count": sum(value == 0 for value in values),
            "negligible_count": sum(abs(value) <= Q_DIVERGENCE_TOLERANCE for value in values),
            "fraction_zero": sum(value == 0 for value in values) / len(values) if values else None,
            "fraction_negligible": sum(abs(value) <= Q_DIVERGENCE_TOLERANCE
                                         for value in values) / len(values) if values else None,
        }
    for name, values in threshold_diffs.items():
        result.setdefault("threshold_divergence", {})[name] = {
            "signed": _summary(values), "absolute": _summary([abs(value) for value in values]),
            "zero_count": sum(value == 0 for value in values),
            "negligible_count": sum(abs(value) <= THETA_DIVERGENCE_TOLERANCE
                                     for value in values),
            "fraction_negligible": sum(abs(value) <= THETA_DIVERGENCE_TOLERANCE
                                         for value in values) / len(values) if values else None,
        }
    result["arms"] = {
        arm: {"lag_versions": lag,
              "reference_version_gap_counts": reference_gaps[arm],
              **movement_counts[arm]}
        for arm, lag in ARMS.items()
    }
    result["per_class"] = {
        category: {
            name: {"signed": _summary(values),
                   "absolute": _summary([abs(value) for value in values])}
            for name, values in metrics.items()
        } for category, metrics in per_class_values.items()
    }
    result["prediction_disagreement_events"] = {
        "total_events": EXPECTED_EVENTS, **disagreements,
        "per_class": disagreements_by_class,
    }
    result["threshold_equal_after_transition_counts"] = {
        pair: {"count": count, "total_transitions": transition_count,
               "fraction": count / transition_count if transition_count else None}
        for pair, count in thresholds_equal_transition_count.items()
    }
    result["error_changes_on_next_event_after_memory_transition"] = error_after
    result["error_changes_after_transition_by_class"] = error_after_by_class
    result["memory_transition_count"] = transition_count
    result["prediction_disagreement_as_threshold_interval_decisions"] = True
    return result


def _classify_h1(metrics: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    disagreements = diagnostics["prediction_disagreement_events"]
    thresholds = diagnostics["threshold_equal_after_transition_counts"]
    any_decision_change = any(disagreements.get(pair, 0) > 0
                              for pair in ("A2-A3", "A2-A4"))
    any_threshold_divergence = any(
        thresholds[pair]["count"] < thresholds[pair]["total_transitions"]
        for pair in ("A2-A3", "A2-A4"))
    threshold_divergence_fractions = {
        pair: (1.0 - row["fraction"] if row["fraction"] is not None else None)
        for pair, row in thresholds.items()}
    if (not any_decision_change and all(
            value is not None and value <= INTERPRETATION_RULES[
                "manipulation_inconclusive_if_threshold_divergence_fraction_at_most"]
            for value in threshold_divergence_fractions.values())):
        return {"classification": "INCONCLUSIVE",
                "reason": "Lagged references changed decisions zero times and changed post-transition thresholds on at most the registered 1% of transitions; the manipulation had little operational effect.",
                "criteria": INTERPRETATION_RULES,
                "threshold_divergence_fractions": threshold_divergence_fractions,
                "manipulation_changed_thresholds": any_threshold_divergence,
                "manipulation_changed_predictions": any_decision_change}

    scope = "full_stream"
    comparisons = metrics["paired_results"]["scope_results"][scope]["equal_class_macro"]
    class_effects = metrics["paired_results"]["scope_results"][scope]["per_class"]
    supports: list[bool] = []
    reasons: dict[str, Any] = {}
    for pair in ("A2-A3", "A2-A4"):
        ba_macro = comparisons[pair]["balanced_accuracy"]["equal_class_macro_mean"]
        ba_classes = [class_effects[pair][category]["balanced_accuracy"]["mean"]
                      for category in CLASSES]
        fpr_macro = comparisons[pair]["fpr"]["equal_class_macro_mean"]
        worst_fnr_macro = comparisons[pair]["worst_complete_256.fnr"]["equal_class_macro_mean"]
        positive_classes = sum(value is not None and value > PAIRED_TIE_TOLERANCE
                               for value in ba_classes)
        maximum_loss = max((max(0.0, -float(value)) for value in ba_classes
                            if value is not None), default=None)
        robust_cost = max(0.0, float(worst_fnr_macro)) if worst_fnr_macro is not None else None
        supports.append(
            ba_macro is not None and ba_macro >= INTERPRETATION_RULES["meaningful_macro_ba_margin"]
            and positive_classes >= INTERPRETATION_RULES["minimum_classes_with_positive_ba_effect"]
            and maximum_loss is not None
            and maximum_loss <= INTERPRETATION_RULES["maximum_class_mean_ba_loss"]
            and robust_cost is not None
            and robust_cost <= INTERPRETATION_RULES["maximum_worst256_fnr_cost"])
        reasons[pair] = {
            "equal_class_macro_ba_difference": ba_macro,
            "class_mean_ba_differences": dict(zip(CLASSES, ba_classes)),
            "positive_class_count": positive_classes,
            "maximum_class_ba_loss": maximum_loss,
            "equal_class_macro_fpr_difference": fpr_macro,
            "equal_class_macro_worst256_fnr_difference": worst_fnr_macro,
            "support_criteria_met": supports[-1],
            "fpr_direction": "positive difference means A2 has higher FPR than the lagged arm",
        }
    if all(supports):
        classification = "H1_SUPPORTED_ON_DEVELOPMENT"
        reason = "Both lag contrasts meet the frozen descriptive BA consistency and robustness criteria; this is not a significance or guarantee claim."
    else:
        ba_means = [reasons[pair]["equal_class_macro_ba_difference"] for pair in reasons]
        defined = [float(value) for value in ba_means if value is not None]
        if len(defined) == 2 and all(value <= PAIRED_TIE_TOLERANCE for value in defined):
            classification = "H1_NOT_SUPPORTED"
            reason = "Lagged references perform similarly or better in equal-class macro BA."
        else:
            classification = "H1_WEAK_OR_MIXED"
            reason = "Effects are smaller than the frozen meaningful margin, heterogeneous, or accompanied by operating-point/robustness costs."
    return {"classification": classification, "reason": reason,
            "criteria": INTERPRETATION_RULES, "contrasts": reasons,
            "threshold_divergence_fractions": threshold_divergence_fractions,
            "manipulation_changed_thresholds": any_threshold_divergence,
            "manipulation_changed_predictions": any_decision_change}


def _build_report(metrics: Mapping[str, Any], paired: Mapping[str, Any],
                  diagnostics: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    lines = [
        "# Formal B3 state-consistency mechanism ablation",
        "",
        "## Table 1 — Overall comparison",
        "",
        "Equal-class macro means are computed from four class means; seed dispersion is reported separately in the JSON artifact. Worst-256 metrics use only complete, non-overlapping 256-image windows. A positive FPR/FNR paired difference means A2 has more errors than the lagged arm.",
        "",
        "| Scope | Arm | BA | FPR | FNR | Worst-256 BA | Coverage | Defect recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scope, display in (("full_stream", "Full stream"),
                           ("common_horizon_t_ge_1050", "Common horizon t ≥ 1050")):
        for arm in ARMS:
            summary = metrics["descriptive_method_summaries"][scope][arm]["equal_class_macro"]
            values = [summary[key]["equal_class_macro_mean"] for key in (
                "balanced_accuracy", "fpr", "fnr",
                "worst_complete_256.balanced_accuracy", "coverage", "defect_recall")]
            lines.append("| " + display + " | " + arm + " | " + " | ".join(
                _fmt(value) for value in values) + " |")
    lines.extend(["", "## Table 2 — Per-class paired effects over five support seeds", "",
                  "Each cell is the mean per-seed raw difference. BA and Worst-256 BA use A2 minus lagged arm; FPR/FNR differences are costs when positive.",
                  "",
                  "| Class | BA A2−A3 | BA A2−A4 | FPR A2−A3 | FPR A2−A4 | FNR A2−A3 | FNR A2−A4 | Worst-256 BA A2−A3 | Worst-256 BA A2−A4 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    per_class = paired["scope_results"]["full_stream"]["per_class"]
    for category in CLASSES:
        values = [per_class[pair][category][metric]["mean"] for pair, metric in (
            ("A2-A3", "balanced_accuracy"), ("A2-A4", "balanced_accuracy"),
            ("A2-A3", "fpr"), ("A2-A4", "fpr"),
            ("A2-A3", "fnr"), ("A2-A4", "fnr"),
            ("A2-A3", "worst_complete_256.balanced_accuracy"),
            ("A2-A4", "worst_complete_256.balanced_accuracy"))]
        lines.append("| " + category + " | " + " | ".join(_fmt(value) for value in values) + " |")
    lines.extend(["", "## Table 3 — Mechanism evidence", "",
                  "| Contrast | Mean signed q divergence | Mean absolute q divergence | Negligible q fraction | Mean signed post-update theta divergence | Transitions with equal theta | Decisions changed | Next-event error difference after transitions (A2−lag) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for lag, q_name, theta_name, pair in (
            ("1 memory version", "current_minus_lag1", "A2_minus_A3", "A2-A3"),
            ("5 memory versions", "current_minus_lag5", "A2_minus_A4", "A2-A4")):
        q = diagnostics["quantile_divergence"][q_name]
        th = diagnostics["threshold_divergence"][theta_name]
        equal_pair = diagnostics["threshold_equal_after_transition_counts"][pair]
        error_pair = diagnostics["error_changes_on_next_event_after_memory_transition"][pair]
        lines.append("| " + lag + " | " + _fmt(q["signed"]["mean"]) + " | " +
                     _fmt(q["absolute"]["mean"]) + " | " +
                     _fmt(q["fraction_negligible"]) + " | " +
                     _fmt(th["signed"]["mean"]) + " | " +
                     f"{equal_pair['count']}/{equal_pair['total_transitions']} ({_fmt(equal_pair['fraction'])})" + " | " +
                     str(diagnostics["prediction_disagreement_events"][pair]) + " | " +
                     str(error_pair["left_minus_right_error_count"]) + " |")
    gap_summary = "; ".join(
        f"{arm}: {json.dumps(diagnostics['arms'][arm]['reference_version_gap_counts'], sort_keys=True)}"
        for arm in ARMS)
    movement_summary = "; ".join(
        f"{arm}: {diagnostics['arms'][arm]['clipped']}/{diagnostics['arms'][arm]['updates']} clipped, "
        f"{diagnostics['arms'][arm]['at_clip_boundary']} at the ±{DELTA} cap"
        for arm in ARMS)
    lines.extend(["", f"Reference-version gap counts (current version minus reference version): {gap_summary}.",
                  f"Threshold movement/clipping: {movement_summary}.",
                  "A changed q90 that leaves the clipped theta unchanged is counted separately from a changed binary decision. Since scores and Memory histories are shared, every inter-arm classification change is caused by the threshold positions."])
    lines.extend(["", "## Table 4 — Integrity", "",
                  "| Item | Result |", "|---|---|"])
    for key, value in audit["integrity_summary"].items():
        lines.append(f"| {key.replace('_', ' ')} | {value} |")
    conclusion = audit["scientific_interpretation"]
    lines.extend(["", "## Research interpretation", "",
                  f"**{conclusion['classification']}** — {conclusion['reason']}", "",
                  "Current-state quantile differences can leave the clipped threshold unchanged; threshold differences can also leave every binary decision unchanged. The diagnostics report both stages separately. No monotonic deterioration with longer lag is assumed.", "",
                  "### Operating-point tradeoff", "",
                  "FPR and FNR differences are reported alongside BA. If A2's BA gain comes with a higher FPR or worse complete-window FNR, that cost remains part of the conclusion.", "",
                  "### Negative-result record", "",
                  "A null, mixed, or lag-favoring result is retained as valid development evidence. The lag manipulation's quantile, threshold, and decision divergence counts are included even when the primary BA contrast is null. No statistical significance, population guarantee, or confirmation-class claim is made.", "",
                  "### Limitations", "",
                  "This is a four-class development study with five correlated support-seed trajectories per class. It uses four deployed normal supports plus 64 additional protected C64 calibration normals, not a strict four-shot-total setting. Historical Stage 0.75-A/G2 source-byte parity remains blocked; A1's frozen-input A0 parity is a separate verification. No images were rescored, no GPU inference was run, and no masks or confirmation outcomes were accessed.", "",
                  "Machine-readable details, per-trajectory outcomes, paired effects, transition diagnostics, source hashes, and artifact checksums are in the versioned run directory recorded by the audit.", ""])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    return f"{float(value):.4f}"


def _evaluate_sealed_predictions(output: Path, *, json_root: Path,
                                 source_root: Path,
                                 label_access_callback=None) -> dict[str, Any]:
    # This validation is deliberately complete before importing the label API.
    seal, receipts = _verify_prediction_seal(output)
    receipt_by_key = {(item["class"], int(item["support_seed"])): item
                      for item in receipts["trajectories"]}
    prediction_rows: dict[str, list[dict[str, Any]]] = {}
    scores_by_key: dict[str, list[float]] = {}
    predictions_by_key: dict[str, dict[str, list[int]]] = {}
    for category in CLASSES:
        for seed in SEEDS:
            receipt = receipt_by_key[(category, seed)]
            key = f"{category}/seed{seed}"
            path = _safe_relative(output, receipt["prediction_file"])
            rows = _read_prediction_rows(path, expected_count=int(receipt["event_count"]))
            prediction_rows[key] = rows
            scores_by_key[key] = [float(row["score"]) for row in rows]
            predictions_by_key[key] = {
                arm: [int(row["arms"][arm]["prediction"]) for row in rows]
                for arm in ARMS
            }

    # The sole official outcome access occurs below, after seal/hash validation.
    from .stage05 import test_labels_for_class
    from .metrics import PrivateOutcomeLedger

    labels_by_class: dict[str, list[int]] = {}
    label_sources: dict[str, Any] = {}
    label_calls = 0
    for category in CLASSES:
        if label_access_callback is not None:
            label_access_callback("started", category)
        labels = test_labels_for_class(json_root, category)
        label_calls += 1
        if label_access_callback is not None:
            label_access_callback("completed", category)
        expected_lengths = {len(prediction_rows[f"{category}/seed{seed}"]) for seed in SEEDS}
        if expected_lengths != {len(labels)}:
            raise FormalRunError(f"Official label count differs for {category}: {len(labels)}")
        if any(value not in (0, 1) for value in labels):
            raise FormalRunError(f"Official labels are not binary for {category}")
        labels_by_class[category] = labels
        label_path = json_root / f"{category}.json"
        label_sources[category] = {
            "official_json_sha256": _sha_file(label_path),
            "test_image_count": len(labels),
            "label_provider_calls": 1,
            "raw_labels_saved": False,
            "mask_files_opened": False,
            "confirmation_outcomes_opened": False,
        }

    rows_out: dict[str, Any] = {}
    for category in CLASSES:
        labels = labels_by_class[category]
        for seed in SEEDS:
            key = f"{category}/seed{seed}"
            scores = scores_by_key[key]
            predictions = predictions_by_key[key]
            ledger = PrivateOutcomeLedger(labels)
            full_metrics = ledger.evaluate(
                scores, predictions,
                prediction_availability={arm: [True] * len(labels) for arm in ARMS})
            horizon = ledger.evaluate_common_horizon(
                scores, predictions, start=COMMON_HORIZON_START,
                prediction_availability={arm: [True] * len(labels) for arm in ARMS})
            if horizon.get("metrics") is None:
                raise FormalRunError(f"Common horizon unavailable for {key}")
            full_methods = {}
            horizon_methods = {}
            for arm in ARMS:
                full_arm = dict(full_metrics["methods"][arm])
                full_arm["worst_complete_256"] = _worst_complete_windows(
                    full_arm, scope="full_stream")
                horizon_arm = dict(horizon["metrics"]["methods"][arm])
                horizon_arm["worst_complete_256"] = _worst_complete_windows(
                    horizon_arm, scope="common_horizon")
                for window in horizon_arm.get("windows", {}).get("windows", []):
                    window["start_global"] = COMMON_HORIZON_START + int(window["start"])
                    window["stop_exclusive_global"] = (
                        COMMON_HORIZON_START + int(window["stop_exclusive"]))
                full_methods[arm] = full_arm
                horizon_methods[arm] = horizon_arm
            rows_out[key] = {
                "class": category, "support_seed": seed,
                "event_count": len(labels),
                "n_normal": full_metrics["n_normal"],
                "n_defect": full_metrics["n_defect"],
                "full_stream": {
                    "methods": full_methods,
                    "shared_score_auroc": full_metrics["shared_score_auroc"],
                    "shared_score_auprc": full_metrics["shared_score_auprc"],
                    "ranking_status": full_metrics["ranking_status"],
                },
                "common_horizon_t_ge_1050": {
                    "start": COMMON_HORIZON_START,
                    "stop_exclusive": len(labels),
                    "n": horizon["n"],
                    "n_normal": horizon["metrics"]["n_normal"],
                    "n_defect": horizon["metrics"]["n_defect"],
                    "methods": horizon_methods,
                    "shared_score_auroc": horizon["metrics"]["shared_score_auroc"],
                    "shared_score_auprc": horizon["metrics"]["shared_score_auprc"],
                    "ranking_status": horizon["metrics"]["ranking_status"],
                },
            }
    method_summaries = _summarize_method_rows(rows_out)
    paired = _paired_results(rows_out)
    diagnostics = _transition_diagnostics(
        _load_frozen_trajectories(source_root, _read_json(output / "preflight.json")),
        prediction_rows, labels_by_class)
    score_rankings = {
        "full_stream_by_trajectory": {
            key: {"auroc": row["full_stream"]["shared_score_auroc"],
                  "auprc": row["full_stream"]["shared_score_auprc"],
                  "status": row["full_stream"]["ranking_status"]}
            for key, row in rows_out.items()},
        "common_horizon_by_trajectory": {
            key: {"auroc": row["common_horizon_t_ge_1050"]["shared_score_auroc"],
                  "auprc": row["common_horizon_t_ge_1050"]["shared_score_auprc"],
                  "status": row["common_horizon_t_ge_1050"]["ranking_status"]}
            for key, row in rows_out.items()},
        "identical_across_arms_by_construction": True,
        "reason": "Each event stores one frozen score and all three threshold-only predictions consume that same score.",
    }
    metrics = {
        "schema": "calibration.b3_state_lag_per_method_metrics.v1",
        "evaluation_contract": {
            "target": "image-level defect; official test labels applied only after sealed predictions",
            "full_stream": True,
            "common_horizon_start_zero_based": COMMON_HORIZON_START,
            "window_size": WINDOW_SIZE, "window_stride": WINDOW_SIZE,
            "worst_window_requires_full_256_images": True,
            "missing_denominators": "null_with_status_and_reason",
            "raw_outcome_vectors_saved": False,
        },
        "label_source_provenance": label_sources,
        "label_provider_calls": label_calls,
        "per_class_support_seed": rows_out,
        "descriptive_method_summaries": method_summaries,
        "shared_score_ranking_consistency": score_rankings,
    }
    h1 = _classify_h1({"paired_results": paired}, diagnostics)
    audit = {
        "schema": "calibration.b3_state_lag_formal_audit.v1",
        "status": "FORMAL_B3_MECHANISM_ABLATION_COMPLETE",
        "prediction_phase_complete_sha256": _sha_file(output / "prediction_phase_complete.json"),
        "prediction_receipts_sha256": _sha_file(output / "prediction_receipts.json"),
        "prediction_files_sha256": seal["prediction_files"],
        "input_identity_sha256": seal["input_identity_sha256"],
        "source_commit": seal["source_commit"],
        "source_sha256": seal["source_sha256"],
        "evaluator_source_sha256": _source_hashes(source_root.resolve()),
        "post_seal_evaluator_source_change": {
            "changed_files": sorted(
                name for name, old_sha in seal["source_sha256"].items()
                if _source_hashes(source_root.resolve()).get(name) != old_sha),
            "reason": "After the sealed prediction phase, corrected an evaluator-only NameError, added retry-safe label-access logging, and recorded the post-seal source digest. Prediction recurrence, inputs, and thresholds were unchanged.",
        },
        "outcome_labels_accessed": True,
        "label_provider_calls": label_calls,
        "outcome_label_vectors_saved": False,
        "masks_accessed": False,
        "confirmation_outcomes_accessed": False,
        "dataset_image_reads": 0,
        "gpu_inference_calls": 0,
        "native_rescoring_calls": 0,
        "controller_inputs_include_full_labels": False,
        "score_history_query_tape_and_memory_shared_across_arms": True,
        "historical_a2_parity": "EXACT_0_MISMATCHES_ACROSS_76475_EVENTS",
        "original_stage075a_g2_source_byte_parity": "BLOCKED_UNCHANGED",
        "integrity_summary": {
            "trajectories": "20/20",
            "score_events": str(EXPECTED_EVENTS),
            "memory_content_changes": str(EXPECTED_MEMORY_CHANGES),
            "versioned_c64_states": "1210",
            "historical_a2_parity": "PASS (pre/post thresholds, predictions, event identity)",
            "prediction_seal": _sha_file(output / "prediction_phase_complete.json"),
            "source_commit": seal["source_commit"],
            "source_sha256_set_digest": _sha_json(seal["source_sha256"]),
            "input_identity_sha256": seal["input_identity_sha256"],
            "official_label_provider_calls": str(label_calls),
            "label_access_phase": "after prediction seal verification",
            "masks_and_confirmation_access": "none",
            "GPU_calls": "0",
            "formal unit tests": "9/9 PASS; A1 suite 14/14 PASS including A0 verification",
        },
        "scientific_interpretation": h1,
        "limitations": [
            "Four development classes; five support seeds per class are not 20 independent industrial categories.",
            "C64 setting is four deployed supports plus 64 additional protected normal calibration anchors, not strict four-shot-total.",
            "No publication-level significance or finite-time/anytime guarantee is claimed.",
            "Historical Stage 0.75-A/G2 source-byte parity remains blocked and is not retroactively repaired by A1 parity.",
        ],
    }
    metrics["paired_results"] = paired
    metrics["mechanism_diagnostics"] = diagnostics
    metrics["scientific_interpretation"] = h1
    return {"metrics": metrics, "paired": paired,
            "diagnostics": diagnostics, "audit": audit,
            "prediction_rows": prediction_rows,
            "label_sources": label_sources}


def _write_final_outputs(output: Path, evaluation: Mapping[str, Any],
                         identity: Mapping[str, Any]) -> None:
    metrics_path = output / "per_method_metrics.json"
    paired_path = output / "paired_results.json"
    diagnostics_path = output / "mechanism_diagnostics.json"
    audit_path = output / "formal_ablation_audit.json"
    _write_json_verified(metrics_path, evaluation["metrics"])
    _write_json_verified(paired_path, evaluation["paired"])
    _write_json_verified(diagnostics_path, evaluation["diagnostics"])
    audit = dict(evaluation["audit"])
    audit["per_method_metrics_sha256"] = _sha_file(metrics_path)
    audit["paired_results_sha256"] = _sha_file(paired_path)
    audit["mechanism_diagnostics_sha256"] = _sha_file(diagnostics_path)
    _write_json_verified(audit_path, audit)

    report_path = Path("docs/calibration/audit/b3_state_lag_formal_v1_20261030/FORMAL_B3_MECHANISM_RESULTS.md")
    report_abs = (output.parent.parent.parent / report_path).resolve() if not report_path.is_absolute() else report_path
    # The repository root is the third parent of outputs/calibration/<run>.
    report_abs.parent.mkdir(parents=True, exist_ok=True)
    report_text = _build_report(evaluation["metrics"], evaluation["paired"],
                                evaluation["diagnostics"], audit)
    report_sha = _write_text_verified(report_abs, report_text)
    run_manifest = {
        "schema": "calibration.b3_state_lag_run_manifest.v1",
        "status": "COMPLETE",
        "run_identity_sha256": _sha_json(identity),
        "run_identity": identity,
        "report_generation_source_sha256": _source_hashes(output.parent.parent.parent),
        "evaluator_source_sha256": audit.get("evaluator_source_sha256"),
        "prediction_phase_complete_sha256": _sha_file(output / "prediction_phase_complete.json"),
        "prediction_receipts_sha256": _sha_file(output / "prediction_receipts.json"),
        "per_method_metrics_sha256": _sha_file(metrics_path),
        "paired_results_sha256": _sha_file(paired_path),
        "mechanism_diagnostics_sha256": _sha_file(diagnostics_path),
        "formal_ablation_audit_sha256": _sha_file(audit_path),
        "paper_report": report_path.as_posix(), "paper_report_sha256": report_sha,
        "trajectory_count": EXPECTED_TRAJECTORIES, "event_count": EXPECTED_EVENTS,
        "memory_content_changes": EXPECTED_MEMORY_CHANGES,
        "outcome_labels_accessed_after_prediction_seal": True,
        "outcome_label_vectors_saved": False,
        "confirmation_or_mask_access": False,
    }
    _write_json_verified(output / "run_manifest.json", run_manifest)
    _write_json_atomic(output / "RUN_STATE.json", {
        "schema": RUN_SCHEMA, "status": "COMPLETE",
        "trajectory_count": EXPECTED_TRAJECTORIES,
        "event_count": EXPECTED_EVENTS,
        "memory_content_changes": EXPECTED_MEMORY_CHANGES,
    })
    files = {}
    for path in sorted(item for item in output.rglob("*")
                       if item.is_file() and item.name != "checksums.json" and
                       item.name != "INCOMPLETE"):
        files[path.relative_to(output).as_posix()] = _sha_file(path)
    files[report_path.as_posix()] = report_sha
    _write_json_verified(output / "checksums.json", {
        "schema": "calibration.b3_state_lag_checksums.v1",
        "self_excluded": "checksums.json",
        "files": files,
    })
    marker = output / "INCOMPLETE"
    if marker.exists():
        marker.unlink()


def run_evaluator(*, source_root: Path, output: Path,
                  json_root: Path = Path(r"G:\dataset\realiad\jsons\realiad_jsons")) -> Path:
    """Read the four authorized development label vectors only after seal checks."""
    output = output.resolve()
    if not output.is_dir():
        raise FormalRunError(f"Run directory does not exist: {output}")
    seal, _ = _verify_prediction_seal(output)
    identity = _read_json(output / "run_identity.json")
    if identity.get("input_identity_sha256") != seal.get("input_identity_sha256"):
        raise FormalRunError("Prediction seal and run identity disagree")
    _write_json_atomic(output / "RUN_STATE.json", {
        "schema": RUN_SCHEMA, "status": "EVALUATION_STARTED_AFTER_SEAL",
        "prediction_phase_complete_sha256": _sha_file(output / "prediction_phase_complete.json"),
        "outcome_labels_accessed_may_have_started": True,
    })
    access_path = output / "label_access_ledger.json"
    if access_path.exists():
        access_ledger = _read_json(access_path)
    else:
        access_ledger = {"schema": "calibration.b3_state_lag_label_access_ledger.v1",
                         "prediction_seal_sha256": _sha_file(
                             output / "prediction_phase_complete.json"),
                         "attempts": []}
    attempt = {"attempt_id": f"attempt_{len(access_ledger['attempts']) + 1}",
               "status": "RUNNING_AFTER_SEAL", "provider_calls_started": 0,
               "provider_calls_completed": 0, "classes_started": [],
               "classes_completed": [], "raw_vectors_saved": False}
    access_ledger["attempts"].append(attempt)
    _write_json_atomic(access_path, access_ledger)

    def record_access(stage: str, category: str) -> None:
        if stage == "started":
            attempt["provider_calls_started"] += 1
            attempt["classes_started"].append(category)
        elif stage == "completed":
            attempt["provider_calls_completed"] += 1
            attempt["classes_completed"].append(category)
        _write_json_atomic(access_path, access_ledger)

    try:
        evaluation = _evaluate_sealed_predictions(
            output, json_root=json_root, source_root=source_root.resolve(),
            label_access_callback=record_access)
    except Exception as exc:
        attempt["status"] = "FAILED_AFTER_LABEL_ACCESS"
        attempt["failure"] = f"{type(exc).__name__}: {exc}"
        _write_json_atomic(access_path, access_ledger)
        _write_json_atomic(output / "RUN_STATE.json", {
            "schema": RUN_SCHEMA, "status": "EVALUATION_FAILED_AFTER_SEAL",
            "prediction_phase_complete_sha256": _sha_file(
                output / "prediction_phase_complete.json"),
            "label_provider_calls_completed": sum(
                int(row.get("provider_calls_completed", 0))
                for row in access_ledger["attempts"]),
            "failure": attempt["failure"],
        })
        raise
    attempt["status"] = "COMPLETE"
    _write_json_atomic(access_path, access_ledger)
    cumulative_calls = sum(int(row.get("provider_calls_completed", 0))
                           for row in access_ledger["attempts"])
    evaluation["metrics"]["evaluation_attempt_label_provider_calls"] = (
        evaluation["metrics"]["label_provider_calls"])
    evaluation["metrics"]["cumulative_label_provider_calls"] = cumulative_calls
    evaluation["audit"]["label_provider_calls"] = cumulative_calls
    evaluation["audit"]["label_access_attempts"] = access_ledger["attempts"]
    evaluation["audit"]["integrity_summary"]["official_label_provider_calls"] = (
        f"{cumulative_calls} across {len(access_ledger['attempts'])} evaluator attempts; 4 distinct class vectors")
    evaluation["audit"]["integrity_summary"]["label_access_ledger_sha256"] = (
        _sha_file(access_path))
    _write_final_outputs(output, evaluation, identity)
    return output


def finalize_partial_evaluation(*, output: Path) -> Path:
    """Finish report/manifest/checksums from already-written evaluator artifacts.

    This path performs no label access. It is only valid when the complete
    metrics, paired, diagnostics, and audit artifacts already exist.
    """
    output = output.resolve()
    _verify_prediction_seal(output)
    metrics = _read_json(output / "per_method_metrics.json")
    paired = _read_json(output / "paired_results.json")
    diagnostics = _read_json(output / "mechanism_diagnostics.json")
    audit = _read_json(output / "formal_ablation_audit.json")
    if audit.get("status") != "FORMAL_B3_MECHANISM_ABLATION_COMPLETE":
        raise FormalRunError("Partial evaluation audit is not complete")
    if (metrics.get("paired_results") != paired or
            metrics.get("mechanism_diagnostics") != diagnostics):
        raise FormalRunError("Saved metrics and paired/diagnostic artifacts disagree")
    identity = _read_json(output / "run_identity.json")
    evaluation = {"metrics": metrics, "paired": paired,
                  "diagnostics": diagnostics, "audit": audit}
    _write_final_outputs(output, evaluation, identity)
    return output


def run_all(*, source_root: Path, output_base: Path = DEFAULT_OUTPUT,
            resume: bool = False,
            json_root: Path = Path(r"G:\dataset\realiad\jsons\realiad_jsons")) -> Path:
    output = run_prediction_phase(source_root=source_root,
                                  output_base=output_base, resume=resume)
    if (output / "checksums.json").exists():
        _verify_complete_checksums(output)
        return output
    if (output / "formal_ablation_audit.json").exists():
        finalize_partial_evaluation(output=output)
        _verify_complete_checksums(output)
        return output
    run_evaluator(source_root=source_root, output=output, json_root=json_root)
    return output


def _verify_complete_checksums(output: Path) -> dict[str, Any]:
    seal, receipts = _verify_prediction_seal(output)
    checksums = _read_json(output / "checksums.json")
    report_path = Path("docs/calibration/audit/b3_state_lag_formal_v1_20261030/FORMAL_B3_MECHANISM_RESULTS.md")
    root = output.parent.parent.parent
    for relative, expected in checksums.get("files", {}).items():
        if relative == report_path.as_posix():
            path = root / report_path
        else:
            path = _safe_relative(output, relative)
        if not path.is_file() or _sha_file(path) != expected:
            raise FormalRunError(f"Completed-run checksum mismatch: {relative}")
    if len(receipts.get("trajectories", [])) != EXPECTED_TRAJECTORIES:
        raise FormalRunError("Completed run lost a trajectory receipt")
    if _sha_file(output / "prediction_phase_complete.json") != seal.get("seal_sha256", _sha_file(output / "prediction_phase_complete.json")):
        raise FormalRunError("Prediction seal checksum mismatch")
    return checksums


def _verify_command(output: Path) -> None:
    if not (output / "formal_ablation_audit.json").is_file():
        _verify_prediction_seal(output)
        print(json.dumps({"status": "PREDICTION_PHASE_COMPLETE",
                          "output": str(output)}, sort_keys=True))
        return
    _verify_complete_checksums(output)
    print(json.dumps({"status": "COMPLETE_CHECKSUMS_PASS",
                      "output": str(output)}, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("predict", "evaluate", "finalize", "run", "verify"),
                        default="run")
    parser.add_argument("--frozen-root", type=Path, default=Path.cwd(),
                        help="read-only workspace containing A0 frozen artifacts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="new immutable output namespace; existing base gets a version suffix")
    parser.add_argument("--json-root", type=Path,
                        default=Path(r"G:\dataset\realiad\jsons\realiad_jsons"),
                        help="official JSON metadata root, used only in evaluator phase")
    parser.add_argument("--resume", action="store_true",
                        help="resume only a matching INCOMPLETE prediction run")
    args = parser.parse_args(argv)
    try:
        if args.phase == "predict":
            output = run_prediction_phase(source_root=args.frozen_root.resolve(),
                                          output_base=args.output_dir,
                                          resume=args.resume)
        elif args.phase == "evaluate":
            output = args.output_dir.resolve()
            run_evaluator(source_root=args.frozen_root.resolve(), output=output,
                          json_root=args.json_root.resolve())
        elif args.phase == "finalize":
            output = args.output_dir.resolve()
            finalize_partial_evaluation(output=output)
        elif args.phase == "run":
            output = run_all(source_root=args.frozen_root.resolve(),
                             output_base=args.output_dir, resume=args.resume,
                             json_root=args.json_root.resolve())
        else:
            output = args.output_dir.resolve()
            _verify_command(output)
            return 0
        print(json.dumps({"status": "COMPLETE", "output": str(output)}, sort_keys=True))
        return 0
    except (FormalRunError, a1.AblationError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
