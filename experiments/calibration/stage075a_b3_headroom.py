"""Offline actual-B3 replay and local operating-point headroom audit.

This module reads the frozen Stage 0.75-A scores, event logs, C64 transports,
and snapshot arrays.  It never invokes the scorer, Greedy writer, backbone, or
label oracle.  Evaluation labels are loaded only after the label-free B3
replay has completed.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .controllers import StateConsistentQuantile, theta0_from_c64
from .protocol import EvidenceAnchor, EvidenceSnapshot, Observation
from .profiles import canonical_json
from .stage075a import DEVELOPMENT_CLASSES, SNAPSHOT_NEXT_WIDTH, SNAPSHOT_POSITIONS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "outputs/calibration/stage075a_greedy_ranking_repaired3_20261029"
DEFAULT_STAGE05 = ROOT / "outputs/calibration/stage05_exploratory_20261029"
DEFAULT_JSON_ROOT = Path("G:/dataset/realiad/jsons/realiad_jsons")
SUPPORT_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_SOURCE_HEAD = "4d715a9b27027ee0c52dab4552d7296153ca8633"
DELTA = 0.01
MIN_NORMALS = 16
LAG_ZERO_TOL = 1e-12
METRIC_TOL = 1e-12
# Frozen before evaluator labels are loaded. This is a diagnostic grid only.
LOCAL_OFFSETS = (0.0, -0.0025, 0.0025, -0.005, 0.005, -0.01, 0.01,
                 -0.02, 0.02, -0.04, 0.04)
LOCAL_GRID_RULE = "theta_candidate = replayed_actual_B3_theta + h; score >= threshold"
LOCAL_ACTION_TIE_RULE = "maximize BA; minimize abs(h); then choose lower raw threshold"


class HeadroomError(RuntimeError):
    """A verified source artifact cannot support the registered diagnostic."""


@dataclass(frozen=True)
class CategoryInput:
    category: str
    directory: Path
    manifest: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]
    transport: Mapping[str, tuple[Mapping[str, Any], ...]]
    arrays: Mapping[str, np.ndarray]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_hash_manifest(folder: Path, manifest_path: Path,
                          *, allowed_unlisted: set[str] | None = None) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    files = manifest.get("files", manifest)
    if not isinstance(files, dict):
        raise HeadroomError(f"Invalid checksum schema: {manifest_path}")
    missing, mismatched = [], []
    for relative, expected in files.items():
        path = folder / relative
        if not path.is_file():
            missing.append(relative)
        elif sha256_file(path) != expected:
            mismatched.append(relative)
    allowed = allowed_unlisted or set()
    actual = {path.relative_to(folder).as_posix()
              for path in folder.rglob("*") if path.is_file() and
              path.name != "checksums.json"}
    unlisted = sorted(actual - set(files) - allowed)
    if missing or mismatched or unlisted:
        raise HeadroomError(
            f"Checksum verification failed for {folder}: missing={missing}, "
            f"mismatched={mismatched}, unlisted={unlisted}")
    return {"status": "PASS", "checked_files": len(files),
            "missing": missing, "mismatched": mismatched,
            "unlisted_expected": sorted(actual - set(files))}


def _git_blob(path: str) -> str:
    proc = subprocess.run(["git", "hash-object", f"--path={path}", path], cwd=ROOT,
                          capture_output=True, text=True, check=False)
    if proc.returncode:
        raise HeadroomError(f"Cannot verify Git blob for protected source {path}: {proc.stderr}")
    return proc.stdout.strip()


def _current_source_hashes(protocol: Mapping[str, Any]) -> dict[str, bool]:
    declared = protocol.get("source_sha256", {})
    statuses: dict[str, bool] = {}
    for relative, expected in declared.items():
        path = ROOT / relative
        statuses[relative] = path.is_file() and sha256_file(path) == expected
    if not statuses or not all(statuses.values()):
        failures = [name for name, ok in statuses.items() if not ok]
        raise HeadroomError(f"Stage 0.75-A source identity mismatch: {failures}")
    return statuses


def verify_stage05_inputs(stage05_dir: Path) -> dict[str, Any]:
    checks = _verify_hash_manifest(stage05_dir, stage05_dir / "checksums.json")
    run_manifest = _read_json(stage05_dir / "run_manifest.json")
    score_manifest = _read_json(stage05_dir / "score_tape_manifest.json")
    frozen_inputs = stage05_dir / "frozen_inputs.json"
    if not frozen_inputs.is_file():
        raise HeadroomError("Stage 0.5 frozen input manifest is missing")
    return {
        "status": "PASS", "checksum_verification": checks,
        "run_manifest_sha256": sha256_file(stage05_dir / "run_manifest.json"),
        "score_manifest_sha256": sha256_file(stage05_dir / "score_tape_manifest.json"),
        "frozen_inputs_sha256": sha256_file(frozen_inputs),
        "run_schema": run_manifest.get("schema"),
        "score_schema": score_manifest.get("schema"),
    }


def verify_source_artifacts(source_dir: Path, stage05_dir: Path) -> dict[str, Any]:
    """Verify raw, category, protocol, source, and Stage 0.5 provenance."""
    allowed_root_extras = {"INCOMPLETE"}
    allowed_root_extras.update(
        path.relative_to(source_dir).as_posix()
        for path in (source_dir / "private_checkpoints").rglob("*") if path.is_file())
    root_checks = _verify_hash_manifest(
        source_dir, source_dir / "checksums.json", allowed_unlisted=allowed_root_extras)
    run = _read_json(source_dir / "run_manifest.json")
    protocol_path = source_dir / "protocol_manifest.json"
    protocol = _read_json(protocol_path)
    online = _read_json(source_dir / "online_complete.json")
    linkage = _read_json(source_dir / "audit_linkage.json")
    if run.get("phase_status") != {"evaluate": "PASS", "freeze": "PASS",
                                    "publish": "PASS", "run": "PASS", "smoke": "PASS"}:
        raise HeadroomError(f"Original Stage 0.75-A phase manifest is not complete: {run.get('phase_status')}")
    if (online.get("status") != "PASS" or online.get("trajectory_count") != 20 or
            run.get("online_trajectory_count") != 20):
        raise HeadroomError("Original Stage 0.75-A online completion is not exactly 20/20")
    raw_protocol_hash = sha256_file(protocol_path)
    if linkage.get("protocol_manifest_sha256") != raw_protocol_hash:
        raise HeadroomError("Stage 0.75-A audit linkage does not match protocol bytes")
    if linkage.get("online_completion_sha256") != sha256_file(source_dir / "online_complete.json"):
        raise HeadroomError("Stage 0.75-A audit linkage does not match online completion")
    if linkage.get("evaluation_metrics_sha256") != sha256_file(source_dir / "evaluation_metrics.json"):
        raise HeadroomError("Stage 0.75-A audit linkage does not match evaluation metrics")
    if run.get("online_completion_sha256") != sha256_file(source_dir / "online_complete.json"):
        raise HeadroomError("Stage 0.75-A run manifest completion digest mismatch")
    if run.get("offline_evaluation_sha256") != sha256_file(source_dir / "evaluation_metrics.json"):
        raise HeadroomError("Stage 0.75-A run manifest evaluation digest mismatch")
    if protocol.get("source_head") != EXPECTED_SOURCE_HEAD or run.get("source_head") != EXPECTED_SOURCE_HEAD:
        raise HeadroomError("Stage 0.75-A source head differs from its recorded parent")
    if protocol.get("parent_sha") != EXPECTED_SOURCE_HEAD or run.get("parent_sha") != EXPECTED_SOURCE_HEAD:
        raise HeadroomError("Stage 0.75-A parent provenance mismatch")
    if set(online.get("categories", [])) != set(DEVELOPMENT_CLASSES):
        raise HeadroomError("Stage 0.75-A category set differs from the frozen four classes")
    if tuple(online.get("support_seeds", [])) != SUPPORT_SEEDS:
        raise HeadroomError("Stage 0.75-A support seed list is not exactly 0..4")
    if protocol.get("scope", {}).get("stage1_or_stage2") or protocol.get("scope", {}).get("confirmation_classes"):
        raise HeadroomError("Stage 0.75-A scope provenance is inconsistent with this task")
    if protocol.get("scope", {}).get("masks"):
        raise HeadroomError("Original protocol indicates masks were accessed")
    source_status = _current_source_hashes(protocol)
    stage05 = verify_stage05_inputs(stage05_dir)
    if stage05["run_manifest_sha256"] != protocol.get("stage05_run_manifest_sha256"):
        raise HeadroomError("Stage 0.75-A no longer matches its Stage 0.5 run manifest")
    if stage05["score_manifest_sha256"] != protocol.get("stage05_score_manifest_sha256"):
        raise HeadroomError("Stage 0.75-A no longer matches its Stage 0.5 score manifest")
    # The old runner leaves this marker on every successful run; preserve it and
    # disclose it. Its own verifier excludes it, while all phase manifests pass.
    stale_marker = ((source_dir / "INCOMPLETE").read_text(encoding="utf-8")
                    if (source_dir / "INCOMPLETE").is_file() else None)
    canonical_protocol_hash = hashlib.sha256(canonical_json(protocol).encode("utf-8")).hexdigest()
    if run.get("protocol_sha256") != canonical_protocol_hash:
        raise HeadroomError("Stage 0.75-A canonical protocol digest mismatch")
    online_rows = {row["category"]: row for row in online["trajectories"]}
    if set(online_rows) != set(DEVELOPMENT_CLASSES):
        raise HeadroomError("Online completion has missing or duplicate category rows")
    return {
        "status": "PASS", "source_directory": source_dir.relative_to(ROOT).as_posix(),
        "stage075a_parent_commit": EXPECTED_SOURCE_HEAD,
        "run_manifest_sha256": sha256_file(source_dir / "run_manifest.json"),
        "protocol_manifest_sha256": raw_protocol_hash,
        "protocol_manifest_canonical_sha256": canonical_protocol_hash,
        "online_complete_sha256": sha256_file(source_dir / "online_complete.json"),
        "evaluation_metrics_sha256": sha256_file(source_dir / "evaluation_metrics.json"),
        "audit_linkage_sha256": sha256_file(source_dir / "audit_linkage.json"),
        "root_checksum_verification": root_checks,
        "stage05_provenance": stage05,
        "source_hashes_match": source_status,
        "category_completion": {
            category: {"status": online_rows[category].get("status"),
                       "positions_scored": online_rows[category].get("positions_scored"),
                       "support_seeds": online_rows[category].get("support_seeds"),
                       "smoke_only": online_rows[category].get("smoke_only")}
            for category in DEVELOPMENT_CLASSES},
        "stale_incomplete_marker": stale_marker,
        "stale_marker_interpretation": (
            "The historical runner leaves this marker after successful publish. "
            "Run phases, category manifests and its verifier report complete; "
            "marker is excluded from the original checksum manifest and was not modified."),
    }


def _anchor_snapshot(c64: Sequence[float], version: int) -> EvidenceSnapshot:
    anchors = tuple(EvidenceAnchor(
        opaque_anchor_id=f"stage075a-c64-{index}", score=float(score),
        memory_version=version, protected=True, source_exclusion_policy="protected_no_selfwrite")
        for index, score in enumerate(c64))
    return EvidenceSnapshot(memory_version=version, normal_anchors=anchors,
                            defect_anchors=(),
                            source_exclusion_audit={"protected_c64_only": True},
                            evidence_revision=version)


def _q90(c64: Sequence[float]) -> float:
    if len(c64) != 64 or not all(math.isfinite(float(value)) for value in c64):
        raise HeadroomError("C64 replay state must contain 64 finite values")
    return sorted(float(value) for value in c64)[math.ceil(0.90 * 63)]


def _lag_statistics(values: Sequence[float], positions: Sequence[int]) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return {"n_events": 0, "signed_mean": None, "mean_absolute": None,
                "max_absolute": None, "integrated_absolute": None,
                "fraction_nonzero": None, "quantiles_absolute": None}
    nz = np.abs(x) > LAG_ZERO_TOL
    qs = {str(q): float(np.quantile(np.abs(x), q, method="linear"))
          for q in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)}
    return {"n_events": int(len(x)), "signed_mean": float(x.mean()),
            "mean_absolute": float(np.abs(x).mean()),
            "max_absolute": float(np.abs(x).max()),
            "integrated_absolute": float(np.abs(x).sum()),
            "fraction_nonzero": float(nz.mean()),
            "nonzero_events": int(nz.sum()), "quantiles_absolute": qs}


def _nonzero_episodes(values: Sequence[float], positions: Sequence[int]) -> list[dict[str, int]]:
    episodes: list[dict[str, int]] = []
    start = None
    last = None
    count = 0
    for value, position in zip(values, positions):
        if abs(float(value)) > LAG_ZERO_TOL:
            if start is None or last is None or position != last + 1:
                if start is not None:
                    episodes.append({"start_event": start, "end_event": last,
                                     "event_count": count})
                start, count = int(position), 0
            last, count = int(position), count + 1
        elif start is not None:
            episodes.append({"start_event": start, "end_event": last,
                             "event_count": count})
            start, last, count = None, None, 0
    if start is not None:
        episodes.append({"start_event": start, "end_event": last,
                         "event_count": count})
    return episodes


def _cap_episodes(updates: Sequence[Mapping[str, Any]]) -> list[dict[str, int]]:
    episodes: list[dict[str, int]] = []
    active: list[Mapping[str, Any]] = []
    for row in updates:
        if row["binding"] in ("LOWER", "UPPER"):
            active.append(row)
        elif active:
            episodes.append({"start_event": int(active[0]["position"]),
                             "end_event": int(active[-1]["position"]),
                             "bound_update_count": len(active),
                             "event_span": int(active[-1]["position"] - active[0]["position"] + 1)})
            active = []
    if active:
        episodes.append({"start_event": int(active[0]["position"]),
                         "end_event": int(active[-1]["position"]),
                         "bound_update_count": len(active),
                         "event_span": int(active[-1]["position"] - active[0]["position"] + 1)})
    return episodes


def _replay_one(category: str, seed: int, events: Sequence[Mapping[str, Any]],
                arrays: Mapping[str, np.ndarray],
                transport: Sequence[Mapping[str, Any]],
                category_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Replay real B3; this function has no label argument or label access."""
    dynamic = np.asarray(arrays[f"dynamic_s{seed}"], dtype=np.float64)
    q90_recorded = np.asarray(arrays[f"current_q90_s{seed}"], dtype=np.float64)
    versions_recorded = np.asarray(arrays[f"memory_version_before_s{seed}"], dtype=np.int64)
    digests_recorded = np.asarray(arrays[f"memory_sha_before_s{seed}"])
    n = len(dynamic)
    if len(events) != n or not np.isfinite(dynamic).all() or not np.isfinite(q90_recorded).all():
        raise HeadroomError(f"{category}/seed{seed}: trace and finite score arrays do not align")
    if [int(row["position"]) for row in events] != list(range(n)):
        raise HeadroomError(f"{category}/seed{seed}: event trace positions are not complete/ordered")
    if not transport or int(transport[0]["memory_version_before"]) != 0:
        raise HeadroomError(f"{category}/seed{seed}: no version-0 C64 pair to initialize B3")
    before_pairs = np.asarray(arrays[f"c64_pairs_before_s{seed}"], dtype=np.float64)
    after_pairs = np.asarray(arrays[f"c64_pairs_after_s{seed}"], dtype=np.float64)
    if before_pairs.shape != (len(transport), 64) or after_pairs.shape != (len(transport), 64):
        raise HeadroomError(f"{category}/seed{seed}: paired C64 arrays do not align with records")
    if not (np.isfinite(before_pairs).all() and np.isfinite(after_pairs).all()):
        raise HeadroomError(f"{category}/seed{seed}: C64 transport contains nonfinite scores")

    transport_at: dict[int, tuple[int, Mapping[str, Any]]] = {}
    for index, row in enumerate(transport):
        pos = int(row["position"])
        if pos in transport_at:
            raise HeadroomError(f"{category}/seed{seed}: duplicate C64 update position {pos}")
        transport_at[pos] = (index, row)
        if sha256_json(before_pairs[index].tolist()) != row.get("c64_before_sha256"):
            raise HeadroomError(f"{category}/seed{seed}/event{pos}: before-C64 pair hash mismatch")
        if sha256_json(after_pairs[index].tolist()) != row.get("c64_after_sha256"):
            raise HeadroomError(f"{category}/seed{seed}/event{pos}: after-C64 pair hash mismatch")

    current_c64 = before_pairs[0].copy()
    if sha256_json(current_c64.tolist()) != transport[0].get("c64_before_sha256"):
        raise HeadroomError(f"{category}/seed{seed}: initial C64 state provenance mismatch")
    theta0 = float(category_manifest["memory_by_seed"][str(seed)]["theta0"])
    if theta0 != _q90(current_c64) or theta0 != theta0_from_c64(current_c64.tolist()):
        raise HeadroomError(f"{category}/seed{seed}: theta0 differs from version-0 C64 higher-q90")
    controller = StateConsistentQuantile(theta=theta0, delta=DELTA, min_normals=MIN_NORMALS)
    controller.on_memory_change(_anchor_snapshot(current_c64, 0))
    init_theta = controller.finalize_step()
    if init_theta != theta0 or controller.evidence_version != 0:
        raise HeadroomError(f"{category}/seed{seed}: B3 initialization caused an extra threshold step")

    current_version = 0
    current_digest = str(events[0]["support_seeds"][str(seed)]["memory_sha256_before"])
    q90_pre: list[float] = []
    theta_pre: list[float] = []
    predictions: list[int] = []
    update_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for position, event in enumerate(events):
        original = event["support_seeds"][str(seed)]
        before_version = int(original["memory_version_before"])
        after_version = int(original["memory_version_after"])
        before_sha = str(original["memory_sha256_before"])
        after_sha = str(original["memory_sha256_after"])
        if (before_version != current_version or before_sha != current_digest or
                int(versions_recorded[position]) != before_version or
                str(digests_recorded[position]) != before_sha):
            raise HeadroomError(f"{category}/seed{seed}/event{position}: memory pre-state mismatch")
        q90 = _q90(current_c64)
        if (q90 != float(q90_recorded[position]) or
                q90 != float(original["current_C64_q90_pre_event"])):
            raise HeadroomError(f"{category}/seed{seed}/event{position}: pre-event C64 q90 mismatch")
        if float(dynamic[position]) != float(original["dynamic_score"]):
            raise HeadroomError(f"{category}/seed{seed}/event{position}: dynamic score mismatch")
        obs = Observation(position, f"stage075a-position-{position}",
                          int(event["camera_id"]), float(dynamic[position]),
                          before_version, before_sha)
        prediction = controller.predict(obs)
        theta = float(prediction.threshold)
        label = int(prediction.label)
        if label != int(float(dynamic[position]) >= theta):
            raise HeadroomError("Actual B3 prediction does not follow score >= theta")
        q90_pre.append(q90)
        theta_pre.append(theta)
        predictions.append(label)

        changed = after_version != before_version
        writer_changed = bool(original["writer"].get("changed", False))
        if changed != writer_changed or after_version not in (before_version, before_version + 1):
            raise HeadroomError(f"{category}/seed{seed}/event{position}: invalid writer/version transition")
        if changed:
            if position not in transport_at:
                raise HeadroomError(f"{category}/seed{seed}/event{position}: missing paired C64 update")
            pair_index, record = transport_at[position]
            if (int(record["memory_version_before"]) != before_version or
                    int(record["memory_version_after"]) != after_version or
                    record["memory_before_sha256"] != before_sha or
                    record["memory_after_sha256"] != after_sha or
                    not np.array_equal(before_pairs[pair_index], current_c64)):
                raise HeadroomError(f"{category}/seed{seed}/event{position}: C64/memory transition mismatch")
            old_theta = theta
            controller.on_memory_change(_anchor_snapshot(after_pairs[pair_index], after_version))
            theta_after = float(controller.finalize_step())
            movement = theta_after - old_theta
            if abs(movement) > DELTA + 1e-12:
                raise HeadroomError(f"{category}/seed{seed}/event{position}: B3 exceeded 0.01 step cap")
            raw_move = _q90(after_pairs[pair_index]) - old_theta
            binding = ("LOWER" if raw_move < -DELTA - 1e-12 else
                       "UPPER" if raw_move > DELTA + 1e-12 else "NONE")
            update_rows.append({"position": position,
                                "version_before": before_version,
                                "version_after": after_version,
                                "theta_before": old_theta,
                                "theta_after": theta_after,
                                "raw_q90_target": _q90(after_pairs[pair_index]),
                                "raw_target_delta": raw_move,
                                "actual_movement": movement,
                                "binding": binding,
                                "q90_changed": _q90(after_pairs[pair_index]) != q90})
            current_c64 = after_pairs[pair_index].copy()
            current_version = after_version
            current_digest = after_sha
        else:
            if position in transport_at:
                raise HeadroomError(f"{category}/seed{seed}/event{position}: transport without memory change")
            if after_sha != before_sha or int(original["writer"]["memory_content_version_after"]) != before_version:
                raise HeadroomError(f"{category}/seed{seed}/event{position}: no-change event altered state")
            # Deliberately do not call B3: queried defects or zero-content normal
            # writes cannot fabricate a memory-version transition.
            theta_after = theta
        if int(controller.evidence_version) != current_version:
            raise HeadroomError(f"{category}/seed{seed}/event{position}: B3 evidence version is stale")
        trace_rows.append({"position": position, "memory_version_before": before_version,
                           "memory_version_after": after_version,
                           "memory_sha256_before": before_sha,
                           "memory_sha256_after": after_sha,
                           "dynamic_score": float(dynamic[position]),
                           "theta_pre": theta, "prediction_pre": label,
                           "q90_pre": q90, "theta_post_for_next_event": theta_after,
                           "content_changed": changed})

    if set(transport_at) != {int(row["position"]) for row in events
                             if int(row["support_seeds"][str(seed)]["memory_version_after"]) !=
                             int(row["support_seeds"][str(seed)]["memory_version_before"])}:
        raise HeadroomError(f"{category}/seed{seed}: C64 transport count differs from content writes")
    lag = np.asarray(theta_pre, dtype=np.float64) - np.asarray(q90_pre, dtype=np.float64)
    event_positions = list(range(n))
    changed_positions = [i for i, row in enumerate(events)
                         if int(row["support_seeds"][str(seed)]["memory_version_after"]) !=
                         int(row["support_seeds"][str(seed)]["memory_version_before"])]
    lag_on_change = [float(lag[i]) for i in changed_positions]
    cap_episodes = _cap_episodes(update_rows)
    threshold_array = np.asarray(theta_pre, dtype=np.float64)
    boundaries = sorted(set([0, *[int(t) for t in SNAPSHOT_POSITIONS if t < n], n]))
    lag_intervals = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        selected_updates = [row for row in update_rows if start <= int(row["position"]) < stop]
        lag_intervals.append({"start_inclusive": start, "end_exclusive": stop,
                              "event_lag": _lag_statistics(lag[start:stop].tolist(),
                                                            event_positions[start:stop]),
                              "memory_change_count": len(selected_updates),
                              "lower_cap_binding_updates": sum(row["binding"] == "LOWER" for row in selected_updates),
                              "upper_cap_binding_updates": sum(row["binding"] == "UPPER" for row in selected_updates)})
    return {
        "category": category, "seed": seed, "n_events": n, "theta0": theta0,
        "theta_pre": theta_pre, "q90_pre": q90_pre, "predictions_pre": predictions,
        "replay_trace": trace_rows, "updates": update_rows,
        "threshold_total_variation": float(np.abs(np.diff(threshold_array)).sum()),
        "threshold_update_count": int(np.count_nonzero(np.abs(np.diff(threshold_array)) > 0.0)),
        "memory_change_count": len(update_rows),
        "movement_distribution": {
            "n_memory_change_updates": len(update_rows),
            "signed_values": [row["actual_movement"] for row in update_rows],
            "absolute_values": [abs(row["actual_movement"]) for row in update_rows],
            "raw_target_delta_values": [row["raw_target_delta"] for row in update_rows]},
        "lag": {"all_events": _lag_statistics(lag.tolist(), event_positions),
                "memory_change_pre_event": _lag_statistics(lag_on_change, changed_positions),
                "stream_intervals": lag_intervals,
                "persistent_episodes": _nonzero_episodes(lag.tolist(), event_positions),
                "longest_persistent_episode_events": max(
                    (row["event_count"] for row in _nonzero_episodes(lag.tolist(), event_positions)),
                    default=0)},
        "movement_cap": {
            "lower_binding_update_count": sum(row["binding"] == "LOWER" for row in update_rows),
            "upper_binding_update_count": sum(row["binding"] == "UPPER" for row in update_rows),
            "unbound_update_count": sum(row["binding"] == "NONE" for row in update_rows),
            "binding_episodes": cap_episodes,
            "longest_episode_event_span": max((row["event_span"] for row in cap_episodes), default=0),
            "q90_unchanged_after_write_count": sum(not row["q90_changed"] for row in update_rows)},
    }


def _snapshot_mask_check(category: str, seed: int, events: Sequence[Mapping[str, Any]],
                         arrays: Mapping[str, np.ndarray],
                         category_manifest: Mapping[str, Any], n: int) -> dict[str, Any]:
    """Reconstruct prior-source exclusions from trace writer events, no labels."""
    per_snapshot: dict[str, Any] = {}
    for t in SNAPSHOT_POSITIONS:
        key = f"snapshot_s{seed}_t{t}"
        if key not in arrays:
            raise HeadroomError(f"{category}/seed{seed}: missing fixed snapshot {t}")
        values = np.asarray(arrays[key], dtype=np.float64)
        if values.shape != (n,) or np.isinf(values).any():
            raise HeadroomError(f"{category}/seed{seed}/snapshot{t}: invalid snapshot score shape/inf")
        prior_groups: set[str] = set()
        for position in range(t):
            event = events[position]
            state = event["support_seeds"][str(seed)]
            if (state.get("newly_resident_source_group") or
                    int(state["writer"].get("appended", 0)) > 0 or
                    int(state["writer"].get("replaced", 0)) > 0):
                prior_groups.add(str(event["official_group_sha256"]))
        expected = np.zeros(n, dtype=bool)
        for position in range(t, n):
            expected[position] = str(events[position]["official_group_sha256"]) not in prior_groups
        finite = np.isfinite(values)
        snapshot_manifest = category_manifest["snapshot_by_seed"][str(seed)][str(t)]
        snapshot_theta = np.asarray(arrays[f"snapshot_theta_s{seed}_t{t}"])
        boundary = events[t]["support_seeds"][str(seed)]
        if snapshot_theta.shape != () or not math.isfinite(float(snapshot_theta)):
            raise HeadroomError(f"{category}/seed{seed}/snapshot{t}: invalid stored snapshot q90")
        if (int(snapshot_manifest["memory_content_version"]) !=
                int(boundary["memory_version_before"]) or
                snapshot_manifest["memory_sha256"] != boundary["memory_sha256_before"] or
                float(snapshot_manifest["c64_q90"]) !=
                float(boundary["current_C64_q90_pre_event"]) or
                float(snapshot_theta) != float(snapshot_manifest["c64_q90"])):
            raise HeadroomError(f"{category}/seed{seed}/snapshot{t}: fixed-memory boundary provenance mismatch")
        expected_next = int(expected[t:min(n, t + SNAPSHOT_NEXT_WIDTH)].sum())
        expected_all = int(expected.sum())
        if not np.array_equal(finite, expected):
            raise HeadroomError(f"{category}/seed{seed}/snapshot{t}: finite-score mask differs from prior-group rule")
        if (int(snapshot_manifest["cohort_all_remaining_count"]) != expected_all or
                int(snapshot_manifest["cohort_next256_count"]) != expected_next or
                int(snapshot_manifest["prior_admitted_group_count"]) != len(prior_groups)):
            raise HeadroomError(f"{category}/seed{seed}/snapshot{t}: frozen cohort manifest mismatch")
        per_snapshot[str(t)] = {"all_remaining_count": expected_all,
                                "next256_count": expected_next,
                                "prior_admitted_group_count": len(prior_groups),
                                "finite_mask_sha256": hashlib.sha256(
                                    np.packbits(finite).tobytes()).hexdigest(),
                                "snapshot_memory_sha256": snapshot_manifest["memory_sha256"],
                                "snapshot_memory_content_sha256": snapshot_manifest["memory_content_sha256"],
                                "snapshot_c64_q90": float(snapshot_manifest["c64_q90"])}
    return per_snapshot


def load_verified_inputs(source_dir: Path, stage05_dir: Path) -> tuple[dict[str, Any], dict[str, CategoryInput]]:
    provenance = verify_source_artifacts(source_dir, stage05_dir)
    online = _read_json(source_dir / "online_complete.json")
    category_manifest = {row["category"]: row for row in online["trajectories"]}
    categories: dict[str, CategoryInput] = {}
    root_checks = _read_json(source_dir / "checksums.json")["files"]
    stage05_manifest = _read_json(stage05_dir / "checksums.json")["files"]
    stage05_scores = _read_json(stage05_dir / "score_tape_manifest.json")
    stage05_c64_provenance: dict[str, Any] = {}
    provenance["source_input_hashes"] = {
        "root_checksums_manifest_sha256": sha256_file(source_dir / "checksums.json"),
        "stage05_checksums_manifest_sha256": sha256_file(stage05_dir / "checksums.json"),
        "stage05_frozen_inputs_sha256": stage05_manifest.get("frozen_inputs.json"),
        "root_files_sha256": {relative: digest for relative, digest in root_checks.items()
                               if relative in {"run_manifest.json", "protocol_manifest.json",
                                               "online_complete.json", "evaluation_metrics.json",
                                               "audit_linkage.json"}},
    }
    protocol = _read_json(source_dir / "protocol_manifest.json")
    for category in DEVELOPMENT_CLASSES:
        directory = source_dir / "trajectories" / category
        _verify_hash_manifest(directory, directory / "checksums.json")
        manifest = _read_json(directory / "trajectory_manifest.json")
        online_row = category_manifest[category]
        if (manifest.get("status") != "ONLINE_COMPLETE" or
                online_row.get("status") != "ONLINE_COMPLETE" or
                manifest.get("smoke_only") is not False or
                int(manifest.get("positions_scored", -1)) != int(online_row.get("positions_scored", -2))):
            raise HeadroomError(f"{category}: full trajectory manifest is absent or marked smoke")
        if set(manifest.get("support_seeds", [])) != set(SUPPORT_SEEDS):
            raise HeadroomError(f"{category}: trajectory seed set differs from 0..4")
        for key, expected in (("scores.npz", manifest.get("score_artifact_sha256")),
                              ("event_trace.jsonl.gz", manifest.get("event_trace_sha256"))):
            if not expected or sha256_file(directory / key) != expected:
                raise HeadroomError(f"{category}: trajectory {key} hash does not match its manifest")
        events = tuple(json.loads(line) for line in gzip.open(
            directory / "event_trace.jsonl.gz", "rt", encoding="utf-8"))
        transport_obj = _read_json(directory / "c64_transport.json")
        transport = {str(seed): tuple(transport_obj[str(seed)]) for seed in SUPPORT_SEEDS}
        archive = np.load(directory / "scores.npz", allow_pickle=False)
        arrays = {key: archive[key].copy() for key in archive.files}
        archive.close()
        n = int(manifest["positions_scored"])
        if len(events) != n:
            raise HeadroomError(f"{category}: event trace length differs from manifest")
        if int(protocol["class_inputs"][category]["stream_images"]) != n:
            raise HeadroomError(f"{category}: event count differs from frozen protocol")
        expected_names = {f"{prefix}_s{seed}" for seed in SUPPORT_SEEDS
                          for prefix in ("static", "dynamic", "current_q90", "ever_group_clean",
                                         "resident_group_clean", "memory_version_before",
                                         "memory_sha_before")}
        expected_names.update({f"c64_pairs_{side}_s{seed}" for seed in SUPPORT_SEEDS
                               for side in ("before", "after")})
        for seed in SUPPORT_SEEDS:
            expected_names.update({f"snapshot_s{seed}_t{t}" for t in SNAPSHOT_POSITIONS})
            expected_names.update({f"snapshot_theta_s{seed}_t{t}" for t in SNAPSHOT_POSITIONS})
        if set(arrays) != expected_names:
            raise HeadroomError(f"{category}: NPZ key set mismatch; missing={sorted(expected_names-set(arrays))}, extra={sorted(set(arrays)-expected_names)}")
        for seed in SUPPORT_SEEDS:
            for prefix in ("static", "dynamic", "current_q90"):
                arr = np.asarray(arrays[f"{prefix}_s{seed}"])
                if arr.shape != (n,) or not np.isfinite(arr).all():
                    raise HeadroomError(f"{category}/seed{seed}: {prefix} score array invalid")
            for prefix in ("ever_group_clean", "resident_group_clean", "memory_version_before", "memory_sha_before"):
                if np.asarray(arrays[f"{prefix}_s{seed}"]).shape != (n,):
                    raise HeadroomError(f"{category}/seed{seed}: {prefix} array misaligned")
            if len(transport[str(seed)]) != int(manifest["memory_by_seed"][str(seed)]["actual_content_changes"]):
                raise HeadroomError(f"{category}/seed{seed}: C64 transport count does not equal writer changes")
            score_entry = stage05_scores["classes"][category]["score_tapes"][str(seed)]
            score_path = stage05_dir / score_entry["file"]
            relative_score_path = score_path.relative_to(stage05_dir).as_posix()
            if relative_score_path not in stage05_manifest:
                raise HeadroomError(f"{category}/seed{seed}: Stage 0.5 C64 score tape is not checksum-listed")
            with np.load(score_path, allow_pickle=False) as stage05_tape:
                stage05_c64 = np.asarray(stage05_tape["c64_scores"], dtype=np.float64).copy()
            raw_initial_c64 = np.asarray(arrays[f"c64_pairs_before_s{seed}"][0], dtype=np.float64)
            theta0_stage05 = theta0_from_c64(stage05_c64.tolist())
            theta0_recorded = float(manifest["memory_by_seed"][str(seed)]["theta0"])
            if (stage05_c64.shape != (64,) or not np.isfinite(stage05_c64).all() or
                    not np.array_equal(stage05_c64, raw_initial_c64) or
                    theta0_stage05 != theta0_recorded):
                raise HeadroomError(f"{category}/seed{seed}: Stage 0.5 C64 scores do not reproduce raw Stage 0.75-A theta0")
            stage05_c64_provenance[f"{category}/seed{seed}"] = {
                "score_tape_file": relative_score_path,
                "score_tape_file_sha256": sha256_file(score_path),
                "c64_scores_sha256": sha256_json(stage05_c64.tolist()),
                "theta0_higher_q90": theta0_stage05,
                "exactly_matches_stage075a_initial_c64": True}
            _snapshot_mask_check(category, seed, events, arrays,
                                 manifest, n)
            for pos, event in enumerate(events):
                state = event["support_seeds"][str(seed)]
                if (float(arrays[f"dynamic_s{seed}"][pos]) != float(state["dynamic_score"]) or
                        int(arrays[f"memory_version_before_s{seed}"][pos]) != int(state["memory_version_before"]) or
                        str(arrays[f"memory_sha_before_s{seed}"][pos]) != str(state["memory_sha256_before"])):
                    raise HeadroomError(f"{category}/seed{seed}/event{pos}: trace and NPZ state disagree")
                if bool(arrays[f"ever_group_clean_s{seed}"][pos]) == bool(state["pre_event_previously_admitted_group"]):
                    raise HeadroomError(f"{category}/seed{seed}/event{pos}: original ever-group-clean mask mismatch")
                if bool(arrays[f"resident_group_clean_s{seed}"][pos]) == bool(state["pre_event_currently_resident_group"]):
                    raise HeadroomError(f"{category}/seed{seed}/event{pos}: original resident-group-clean mask mismatch")
        categories[category] = CategoryInput(category, directory, manifest,
                                             events, transport, arrays)
    provenance["trajectory_count"] = sum(len(SUPPORT_SEEDS) for _ in categories)
    provenance["category_event_counts"] = {
        category: len(categories[category].events) for category in DEVELOPMENT_CLASSES}
    provenance["smoke_artifacts_used"] = False
    provenance["stage05_c64_theta0_reconciliation"] = stage05_c64_provenance
    return provenance, categories


def replay_all(categories: Mapping[str, CategoryInput]) -> dict[str, Any]:
    """Perform all label-free source-faithful B3 replays."""
    trajectories: dict[str, dict[str, Any]] = {}
    for category in DEVELOPMENT_CLASSES:
        source = categories[category]
        runs = {}
        for seed in SUPPORT_SEEDS:
            run = _replay_one(category, seed, source.events, source.arrays,
                              source.transport[str(seed)], source.manifest)
            run["snapshot_states"] = _snapshot_mask_check(
                category, seed, source.events, source.arrays, source.manifest,
                run["n_events"])
            runs[str(seed)] = run
        trajectories[category] = runs
    return {"status": "PASS", "controller_import":
            "experiments.calibration.controllers.StateConsistentQuantile",
            "controller_configuration": {"delta": DELTA, "min_normals": MIN_NORMALS,
                                          "prediction": "score >= theta",
                                          "source_file_sha256": sha256_file(ROOT / "experiments/calibration/controllers.py")},
            "labels_supplied_to_replay": False,
            "trajectories": trajectories}


def _reference_recurrence(theta0: float, q90_before: Sequence[float],
                          q90_after: Sequence[float], change: Sequence[bool]) -> list[float]:
    """Independent test oracle for the actual B3 clipping recurrence."""
    theta = float(theta0)
    output = []
    for q_before, q_after, changed in zip(q90_before, q90_after, change):
        output.append(theta)
        if changed:
            theta = min(max(float(q_after), theta - DELTA), theta + DELTA)
        elif float(q_after) != float(q_before):
            raise ValueError("No-content-change recurrence cannot change C64 q90")
    return output


def _confusion(labels: Sequence[int], scores: Sequence[float],
               threshold: float | Sequence[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    th = np.asarray(threshold, dtype=np.float64)
    if (y.ndim != 1 or s.shape != y.shape or th.shape not in ((), y.shape) or
            not np.isin(y, (0, 1)).all() or not np.isfinite(s).all() or
            not np.isfinite(th).all()):
        raise ValueError("Confusion metrics require aligned finite binary inputs")
    pred = s >= th
    pos, neg = y == 1, y == 0
    tp, fp = int(np.count_nonzero(pred & pos)), int(np.count_nonzero(pred & neg))
    fn, tn = int(np.count_nonzero(~pred & pos)), int(np.count_nonzero(~pred & neg))
    fpr = fp / (fp + tn) if fp + tn else None
    fnr = fn / (fn + tp) if fn + tp else None
    ba = 0.5 * ((1.0 - fpr) + (1.0 - fnr)) if fpr is not None and fnr is not None else None
    return {"n": int(len(y)), "positive": int(pos.sum()), "negative": int(neg.sum()),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "fpr": fpr, "fnr": fnr,
            "balanced_accuracy": ba,
            "threshold_kind": "per_event" if th.ndim else "scalar"}


def _ranking(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    from .stage075a import ranking_metrics
    return ranking_metrics(labels, scores)


def load_verified_labels(json_root: Path, source_dir: Path,
                         protocol: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Open only previously exposed development image labels, after replay."""
    from .stage05 import test_labels_for_class
    labels: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for category in DEVELOPMENT_CLASSES:
        source = json_root / f"{category}.json"
        digest = sha256_file(source)
        if digest != protocol["metadata_sha256_by_class"][category]:
            raise HeadroomError(f"{category}: evaluator metadata hash differs from frozen Stage 0.75-A protocol")
        values = np.asarray(test_labels_for_class(json_root, category), dtype=np.int8)
        if values.ndim != 1 or not np.isin(values, (0, 1)).all():
            raise HeadroomError(f"{category}: invalid evaluator label vector")
        labels[category] = values
        hashes[category] = digest
    return labels, hashes


def _macro_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, dict[str, Any]] = {}
    for category in DEVELOPMENT_CLASSES:
        selected = [row for row in rows if row["category"] == category]
        by_class[category] = {key: (float(np.mean([row[key] for row in selected
                                                   if row.get(key) is not None]))
                                    if any(row.get(key) is not None for row in selected) else None)
                              for key in ("balanced_accuracy", "fpr", "fnr")}
        by_class[category]["seed_count"] = len(selected)
    macro = {}
    for key in ("balanced_accuracy", "fpr", "fnr"):
        values = [by_class[category][key] for category in DEVELOPMENT_CLASSES
                  if by_class[category][key] is not None]
        macro[key] = float(np.mean(values)) if values else None
    return {"aggregation": "mean seeds within class, then equal-class macro mean",
            "category_means": by_class, "equal_class_macro_mean": macro}


def query_feasibility(categories: Mapping[str, CategoryInput],
                       labels: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Offline query tape audit; only earlier queried labels are counted."""
    rows = []
    for category in DEVELOPMENT_CLASSES:
        source = categories[category]
        y = np.asarray(labels[category], dtype=np.int8)
        if len(y) != len(source.events):
            raise HeadroomError(f"{category}: labels and query trace have different lengths")
        selected_positions = []
        for pos, event in enumerate(source.events):
            selected = bool(event["query_tape_selected"])
            disclosed = bool(event["queried_label_disclosed"])
            if selected != disclosed:
                raise HeadroomError(f"{category}/event{pos}: selected query lacks exactly one disclosed label")
            if selected:
                selected_positions.append(pos)
        for t in SNAPSHOT_POSITIONS:
            prior = [pos for pos in selected_positions if pos < t]
            normal_groups: set[str] = set()
            defect_groups: set[str] = set()
            first_defect = None
            group_labels: dict[str, int] = {}
            for pos in prior:
                group = str(source.events[pos]["official_group_sha256"])
                label = int(y[pos])
                previous = group_labels.setdefault(group, label)
                if previous != label:
                    raise HeadroomError(f"{category}: one official source group has conflicting queried labels")
                if label == 0:
                    normal_groups.add(group)
                else:
                    defect_groups.add(group)
                    if first_defect is None:
                        first_defect = pos
            rows.append({"category": category, "snapshot_position": int(t),
                         "queries_strictly_before_t": len(prior),
                         "distinct_queried_normal_groups": len(normal_groups),
                         "distinct_queried_defect_groups": len(defect_groups),
                         "first_queried_defect_arrival_before_t": first_defect,
                         "two_class_queried_feedback_available": bool(normal_groups and defect_groups),
                         "queried_anchor_arrival_scores_in_trace": True,
                         "queried_anchor_current_snapshot_scores_available": False,
                         "source_group_identity_available": True,
                         "legal_source_aware_lso_can_be_established": False,
                         "online_local_gain_identifiable": False,
                         "status": "ONLINE_LOCAL_GAIN_NOT_IDENTIFIABLE_FROM_EXISTING_ARTIFACTS",
                         "missing_evidence": [
                             "per-query-anchor score rescored under the exact snapshot memory state",
                             "source-excluded current-memory score for each queried anchor",
                             "versioned provenance linking those rescored anchors to snapshot t"],
                         "reason": (
                             "Stored trace contains each query's arrival score and memory version, "
                             "but no later fixed-snapshot query-anchor scores. Reusing arrival "
                             "scores would be stale; rescoring would require image/features and "
                             "is prohibited by this read-only task.")})
    return rows


def evaluate_all(categories: Mapping[str, CategoryInput],
                 replays: Mapping[str, Any], labels: Mapping[str, np.ndarray],
                 original_evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate after all controller replays have completed; labels stay evaluator-only."""
    per_trajectory: list[dict[str, Any]] = []
    historical_checks: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    for category in DEVELOPMENT_CLASSES:
        source = categories[category]
        y = np.asarray(labels[category], dtype=np.int8)
        if len(y) != len(source.events):
            raise HeadroomError(f"{category}: evaluator labels do not align with event stream")
        groups = [str(event["official_group_sha256"]) for event in source.events]
        for seed in SUPPORT_SEEDS:
            key = str(seed)
            run = replays["trajectories"][category][key]
            arrays = source.arrays
            static = np.asarray(arrays[f"static_s{seed}"], dtype=np.float64)
            dynamic = np.asarray(arrays[f"dynamic_s{seed}"], dtype=np.float64)
            q90 = np.asarray(run["q90_pre"], dtype=np.float64)
            b3 = np.asarray(run["theta_pre"], dtype=np.float64)
            theta0 = float(run["theta0"])
            if not (len(y) == len(static) == len(dynamic) == len(q90) == len(b3)):
                raise HeadroomError(f"{category}/seed{seed}: evaluator and score vectors misalign")
            masks = {
                "as_operated_full_stream": np.ones(len(y), dtype=bool),
                "conservative_ever_group_clean": np.asarray(arrays[f"ever_group_clean_s{seed}"], dtype=bool),
                "resident_source_clean_sensitivity": np.asarray(arrays[f"resident_group_clean_s{seed}"], dtype=bool),
            }
            scope_results: dict[str, Any] = {}
            for scope, mask in masks.items():
                yy = y[mask]
                score_static, score_dynamic = static[mask], dynamic[mask]
                thresholds_q90, thresholds_b3 = q90[mask], b3[mask]
                metrics = {
                    "M0_Frozen_theta0": _confusion(yy, score_static, theta0),
                    "Greedy_Frozen_theta0": _confusion(yy, score_dynamic, theta0),
                    "Greedy_ReadOnly_current_C64_q90": _confusion(yy, score_dynamic, thresholds_q90),
                    "Greedy_Actual_B3": _confusion(yy, score_dynamic, thresholds_b3),
                }
                metrics["M0_Frozen_theta0"]["ranking"] = _ranking(yy, score_static)
                greedy_ranking = _ranking(yy, score_dynamic)
                for method in ("Greedy_Frozen_theta0", "Greedy_ReadOnly_current_C64_q90",
                               "Greedy_Actual_B3"):
                    metrics[method]["ranking"] = greedy_ranking
                scope_results[scope] = {"n_total": len(y), "n_included": int(mask.sum()),
                                        "n_excluded": int((~mask).sum()), "methods": metrics}
            historical = original_evaluation["classes"][category]["support_seeds"][key]
            expected_by_scope = {
                "as_operated_full_stream": historical["as_operated_all_events"],
                "conservative_ever_group_clean": historical["conservative_group_clean"],
                "resident_source_clean_sensitivity": historical["resident_source_clean_sensitivity"],
            }
            checks = []
            mapping = {
                "M0_Frozen_theta0": "frozen_theta0_static",
                "Greedy_Frozen_theta0": "frozen_theta0_dynamic",
                "Greedy_ReadOnly_current_C64_q90": "current_C64_q90_dynamic",
            }
            for scope, expected in expected_by_scope.items():
                got = scope_results[scope]["methods"]
                for method, hist_key in mapping.items():
                    ok = _compare_confusion(got[method], expected[hist_key])
                    checks.append({"scope": scope, "method": method,
                                   "status": "PASS" if ok else "FAIL",
                                   "max_tolerance": METRIC_TOL})
            historical_checks.extend([{"category": category, "seed": seed, **row}
                                      for row in checks])
            per_trajectory.append({"category": category, "seed": seed,
                                   "stream_count": len(y), "normal_count": int((y == 0).sum()),
                                   "defect_count": int((y == 1).sum()),
                                   "theta0": theta0, "scopes": scope_results,
                                   "b3_threshold_total_variation": run["threshold_total_variation"],
                                   "b3_threshold_update_count": run["threshold_update_count"],
                                   "b3_memory_change_count": run["memory_change_count"],
                                   "b3_movement_distribution": run["movement_distribution"],
                                   "b3_lag": run["lag"], "b3_movement_cap": run["movement_cap"],
                                   "b3_vs_q90_full_stream_ba_difference": (
                                       scope_results["as_operated_full_stream"]["methods"]["Greedy_Actual_B3"]["balanced_accuracy"] -
                                       scope_results["as_operated_full_stream"]["methods"]["Greedy_ReadOnly_current_C64_q90"]["balanced_accuracy"]),
                                   "b3_vs_frozen_full_stream_ba_difference": (
                                       scope_results["as_operated_full_stream"]["methods"]["Greedy_Actual_B3"]["balanced_accuracy"] -
                                       scope_results["as_operated_full_stream"]["methods"]["Greedy_Frozen_theta0"]["balanced_accuracy"]),
                                   "historical_metric_reproduction": checks})
            for t in SNAPSHOT_POSITIONS:
                if t >= len(y):
                    continue
                theta_b3, theta_q90 = float(b3[t]), float(q90[t])
                for cohort in ("next256_group_clean", "all_remaining_group_clean"):
                    start = t
                    stop = min(len(y), t + SNAPSHOT_NEXT_WIDTH) if cohort.startswith("next256") else len(y)
                    positions = np.arange(start, stop)
                    scores_all = np.asarray(arrays[f"snapshot_s{seed}_t{t}"], dtype=np.float64)
                    eligible = np.isfinite(scores_all[positions])
                    selected = positions[eligible]
                    if len(selected) == 0:
                        raise HeadroomError(f"{category}/seed{seed}/snapshot{t}/{cohort}: no eligible images")
                    snapshot_result = _snapshot_cohort(y[selected], scores_all[selected],
                                                       [groups[i] for i in selected],
                                                       theta_b3, theta_q90)
                    snapshot_rows.append({"category": category, "seed": seed,
                                          "snapshot_position": int(t), "cohort": cohort,
                                          "eligible_count": len(selected),
                                          "cohort_index_sha256": hashlib.sha256(
                                              np.asarray(selected, dtype=np.int64).tobytes()).hexdigest(),
                                          **snapshot_result})
    if len(per_trajectory) != 20:
        raise HeadroomError(f"Expected 20 trajectories, evaluated {len(per_trajectory)}")
    if not historical_checks or any(row["status"] != "PASS" for row in historical_checks):
        failures = [row for row in historical_checks if row["status"] != "PASS"]
        raise HeadroomError(f"Historical frozen/q90 metric reproduction failed: {failures[:10]}")
    macro: dict[str, Any] = {}
    for scope in ("as_operated_full_stream", "conservative_ever_group_clean",
                  "resident_source_clean_sensitivity"):
        macro[scope] = {}
        for method in ("M0_Frozen_theta0", "Greedy_Frozen_theta0",
                       "Greedy_ReadOnly_current_C64_q90", "Greedy_Actual_B3"):
            rows = [{"category": row["category"], **row["scopes"][scope]["methods"][method]}
                    for row in per_trajectory]
            macro[scope][method] = _macro_summary(rows)
    queries = query_feasibility(categories, labels)
    return {"status": "PASS", "labels_loaded_after_all_replays": True,
            "labels_used_by_controller": False,
            "per_trajectory_metrics": per_trajectory,
            "equal_class_macro_metrics": macro,
            "historical_metric_reproduction": {
                "status": "PASS", "checks": historical_checks,
                "check_count": len(historical_checks), "tolerance": METRIC_TOL},
            "snapshot_headroom": snapshot_rows,
            "snapshot_cohort_count": len(snapshot_rows),
            "local_grid": list(LOCAL_OFFSETS),
            "local_grid_tie_rule": LOCAL_ACTION_TIE_RULE,
            "global_oracle_tie_rule": "maximize BA; minimize absolute movement from actual B3; then higher threshold",
            "query_feasibility": queries,
            "online_identifiability_status": "ONLINE_LOCAL_GAIN_NOT_IDENTIFIABLE_FROM_EXISTING_ARTIFACTS",
            "no_defect_masks_accessed": True,
            "no_new_scoring_performed": True,
            "retrospective_oracle_warning": "All threshold oracles and best local actions use evaluator labels on the same fixed cohort; they are retrospective, in-sample, and not achievable online estimates."}


def _compare_confusion(observed: Mapping[str, Any], expected: Mapping[str, Any],
                       *, tolerance: float = METRIC_TOL) -> bool:
    keys = ("n", "positive", "negative", "tp", "fp", "tn", "fn")
    if any(observed.get(key) != expected.get(key) for key in keys):
        return False
    for key in ("balanced_accuracy", "fpr", "fnr"):
        left, right = observed.get(key), expected.get(key)
        if left is None or right is None:
            if left is not right:
                return False
        elif abs(float(left) - float(right)) > tolerance:
            return False
    return True


def _global_oracle(labels: np.ndarray, scores: np.ndarray,
                   anchor_threshold: float) -> dict[str, Any]:
    if len(np.unique(labels)) != 2:
        return {"status": "UNDEFINED_SINGLE_CLASS", "threshold": None,
                "balanced_accuracy": None, "metrics": _confusion(labels, scores, anchor_threshold),
                "tie_break": "not_applicable"}
    unique, inverse = np.unique(scores, return_inverse=True)
    positive_by_score = np.bincount(inverse, weights=(labels == 1).astype(np.int64),
                                    minlength=len(unique)).astype(np.int64)
    negative_by_score = np.bincount(inverse, weights=(labels == 0).astype(np.int64),
                                    minlength=len(unique)).astype(np.int64)
    tp_tail = np.cumsum(positive_by_score[::-1])[::-1]
    fp_tail = np.cumsum(negative_by_score[::-1])[::-1]
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    candidates: dict[float, tuple[int, int]] = {
        float(th): (int(tp_tail[i]), int(fp_tail[i])) for i, th in enumerate(unique)}
    anchor_index = int(np.searchsorted(unique, float(anchor_threshold), side="left"))
    candidates[float(anchor_threshold)] = (
        int(tp_tail[anchor_index]) if anchor_index < len(unique) else 0,
        int(fp_tail[anchor_index]) if anchor_index < len(unique) else 0)
    candidates[float(np.nextafter(float(scores.max()), math.inf))] = (0, 0)
    rows = []
    for threshold, (tp, fp) in candidates.items():
        ba = 0.5 * ((n_neg - fp) / n_neg + tp / n_pos)
        rows.append((threshold, ba))
    best_ba = max(row[1] for row in rows)
    tied = [row for row in rows if abs(row[1] - best_ba) <= 1e-15]
    threshold, _ = min(tied, key=lambda row: (abs(row[0] - anchor_threshold), -row[0]))
    metrics = _confusion(labels, scores, threshold)
    return {"status": "DEFINED_RETROSPECTIVE_IN_SAMPLE_DIAGNOSTIC",
            "threshold": threshold, "balanced_accuracy": best_ba,
            "metrics": metrics, "candidate_threshold_count": len(candidates),
            "anchor_threshold_in_search": float(anchor_threshold) in candidates,
            "prediction_convention": "score >= threshold",
            "tie_break": "maximize BA; minimize absolute movement from actual B3; then higher threshold",
            "oracle_is_online_or_unbiased": False}


def _band_rows(labels: np.ndarray, scores: np.ndarray, threshold: float,
               group_ids: Sequence[str], offsets: Sequence[float]) -> list[dict[str, Any]]:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(group_ids, dtype=object)
    if groups.shape != y.shape:
        raise ValueError("Band source-group identities must align with cohort")
    base = _confusion(y, s, threshold)
    rows = []
    n0, n1 = base["negative"], base["positive"]
    for offset in offsets:
        if offset == 0:
            continue
        direction = "LOWER" if offset < 0 else "RAISE"
        step = abs(float(offset))
        if offset < 0:
            mask = (s >= threshold - step) & (s < threshold)
            delta_j = ((int(np.count_nonzero(mask & (y == 1))) / n1 if n1 else None) -
                       (int(np.count_nonzero(mask & (y == 0))) / n0 if n0 else None)
                       if n0 and n1 else None)
        else:
            mask = (s >= threshold) & (s < threshold + step)
            delta_j = ((int(np.count_nonzero(mask & (y == 0))) / n0 if n0 else None) -
                       (int(np.count_nonzero(mask & (y == 1))) / n1 if n1 else None)
                       if n0 and n1 else None)
        norm_n = int(np.count_nonzero(mask & (y == 0)))
        defect_n = int(np.count_nonzero(mask & (y == 1)))
        groups_in_band = set(str(x) for x in groups[mask])
        changed = _confusion(y, s, threshold + offset)
        direct_delta = (changed["balanced_accuracy"] - base["balanced_accuracy"]
                        if changed["balanced_accuracy"] is not None and
                        base["balanced_accuracy"] is not None else None)
        identity_delta = 0.5 * float(delta_j) if delta_j is not None else None
        if direct_delta is not None and abs(direct_delta - identity_delta) > 1e-12:
            raise HeadroomError("Decision-band Youden identity does not match direct BA difference")
        rows.append({"direction": direction, "offset": float(offset),
                     "threshold_from": float(threshold), "threshold_to": float(threshold + offset),
                     "normal_count": norm_n, "defect_count": defect_n,
                     "normal_proportion": norm_n / n0 if n0 else None,
                     "defect_proportion": defect_n / n1 if n1 else None,
                     "delta_fpr": norm_n / n0 if n0 and offset < 0 else
                                 (-norm_n / n0 if n0 else None) if offset > 0 else None,
                     "delta_fnr": (-defect_n / n1 if n1 else None) if offset < 0 else
                                 (defect_n / n1 if n1 else None),
                     "delta_youden": delta_j, "delta_balanced_accuracy": direct_delta,
                     "band_identity_delta_balanced_accuracy": identity_delta,
                     "band_identity_pass": direct_delta is not None and
                         abs(direct_delta - identity_delta) <= 1e-12,
                     "empty": not bool(mask.any()),
                     "source_group_count": len(groups_in_band),
                     "source_group_count_by_label": {
                         "normal": len(set(str(x) for x in groups[mask & (y == 0)])),
                         "defect": len(set(str(x) for x in groups[mask & (y == 1)]))},
                     "group_dependence_caveat": "Images from one official source group are dependent; counts are not independent units."})
    return rows


def _snapshot_cohort(labels: np.ndarray, scores: np.ndarray, group_ids: Sequence[str],
                     theta_b3: float, theta_q90: float) -> dict[str, Any]:
    b3 = _confusion(labels, scores, theta_b3)
    q90 = _confusion(labels, scores, theta_q90)
    oracle = _global_oracle(labels, scores, theta_b3)
    grid = []
    for offset in LOCAL_OFFSETS:
        threshold = float(theta_b3 + offset)
        metrics = _confusion(labels, scores, threshold)
        grid.append({"offset": float(offset), "threshold": threshold,
                     "scope": "IMMEDIATE_WITHIN_ONE_STEP" if abs(offset) <= DELTA else "BROADER_DIAGNOSTIC",
                     "metrics": metrics,
                     "delta_balanced_accuracy_vs_b3": (
                         metrics["balanced_accuracy"] - b3["balanced_accuracy"]
                         if metrics["balanced_accuracy"] is not None and
                         b3["balanced_accuracy"] is not None else None)})
    candidates = [row for row in grid if row["metrics"]["balanced_accuracy"] is not None]
    def choose(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
        if not rows:
            return None
        max_ba = max(float(row["metrics"]["balanced_accuracy"]) for row in rows)
        tied = [row for row in rows if abs(float(row["metrics"]["balanced_accuracy"]) - max_ba) <= 1e-15]
        return min(tied, key=lambda row: (abs(float(row["offset"]),), float(row["threshold"])))
    immediate = choose([row for row in candidates if abs(float(row["offset"])) <= DELTA])
    full = choose(candidates)
    oracle_ba = oracle["balanced_accuracy"]
    b3_ba, q90_ba = b3["balanced_accuracy"], q90["balanced_accuracy"]
    h_total = (float(oracle_ba) - float(b3_ba)
               if oracle_ba is not None and b3_ba is not None else None)
    if h_total is not None and h_total < -1e-12:
        raise HeadroomError("Global oracle failed to include actual B3 threshold")
    h_q90 = (float(q90_ba) - float(b3_ba)
             if q90_ba is not None and b3_ba is not None else None)
    h_beyond = (float(oracle_ba) - float(q90_ba)
                if oracle_ba is not None and q90_ba is not None else None)
    def local_summary(best: Mapping[str, Any] | None) -> dict[str, Any]:
        if best is None or b3_ba is None or best["metrics"]["balanced_accuracy"] is None:
            return {"best_offset": None, "best_action": "UNDEFINED_SINGLE_CLASS",
                    "h_local": None, "capture_ratio": None}
        gain = float(best["metrics"]["balanced_accuracy"]) - float(b3_ba)
        action = "KEEP" if float(best["offset"]) == 0 else "LOWER" if float(best["offset"]) < 0 else "RAISE"
        return {"best_offset": float(best["offset"]), "best_action": action,
                "best_threshold": float(best["threshold"]), "best_metrics": best["metrics"],
                "h_local": gain,
                "capture_ratio": gain / h_total if h_total is not None and h_total > 0 else None}
    return {"cohort_count": len(labels), "normal_count": int(np.count_nonzero(labels == 0)),
            "defect_count": int(np.count_nonzero(labels == 1)),
            "theta_b3": float(theta_b3), "theta_q90": float(theta_q90),
            "actual_b3": b3, "readonly_q90": q90,
            "retrospective_global_oracle": oracle,
            "h_total_oracle_minus_b3": h_total,
            "h_q90_minus_b3": h_q90,
            "h_beyond_q90_oracle_minus_q90": h_beyond,
            "local_grid": grid,
            "within_one_step": local_summary(immediate),
            "full_local_grid": local_summary(full),
            "best_local_action": local_summary(full),
            "decision_bands": _band_rows(labels, scores, theta_b3, group_ids, LOCAL_OFFSETS)}
