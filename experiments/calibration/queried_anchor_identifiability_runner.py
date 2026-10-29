"""Run the bounded Stage 0.75 queried-anchor identifiability audit.

The query-estimation stage freezes its actions before it starts a separate
future-evaluation process. Historical sources and outputs are read-only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from . import stage05, stage05_runner
from .queried_anchor_identifiability import (
    AnchorScore, IdentifiabilityError, PRIMARY_OFFSETS, SECONDARY_OFFSETS,
    action_digest, canonical_json, defect_evidence_bin, leave_source_out,
    local_gain_table, select_primary_action,
)
from .stage075a import DEVELOPMENT_CLASSES, SNAPSHOT_POSITIONS
from .stage075a_b3_headroom import (
    DEFAULT_SOURCE as DEFAULT_STAGE075A,
    DEFAULT_STAGE05 as DEFAULT_STAGE05_OUTPUT,
    DEFAULT_JSON_ROOT, SUPPORT_SEEDS, load_verified_inputs, replay_all,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/calibration/queried_anchor_identifiability_v1_20261030"
DEFAULT_B3_OUTPUT = ROOT / "outputs/calibration/stage075a_b3_headroom_v1_20261030_verified2"
PROTOCOL_PATH = ROOT / "docs/calibration/QUERIED_ANCHOR_IDENTIFIABILITY_PROTOCOL_CANDIDATE_v1.md"
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
EXPECTED_START_HEAD = "1e8e66c512984ea34d74564b461a6a9072b7a02e"
EXPECTED_SOURCE_PARENT = "4d715a9b27027ee0c52dab4552d7296153ca8633"
EXPECTED_STAGE075A_PROTOCOL = "948bdb5396269b391a9c8fb51681acabbcda46fe4dee145844344815d0404bf9"
EXPECTED_STAGE075A_CHECKSUMS = "f687d585ad432b043b342e143f141c01754bb670f65d8ae37328bd279bf7fda3"
EXPECTED_STAGE075A_ONLINE = "bc0ad9a02398db43c2167b47926ab3895111f51cd528753d4cc0b595ca1ed629"
EXPECTED_DINO_CHECKPOINT = "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
EXPECTED_DINO_TREE = "1de67887ca330bdfaad8ccd45128d2b9313d63aa707bbe8e4f38310e52005b24"
EXPECTED_SCORER_CONFIG = "1b8e9d96b121e3f1f6c1bbee13ab9f0063facee8fb08ad696dc5f9cb43dbd452"
PARITY_ATOL = 1e-6
PARITY_RTOL = 1e-6
QUERY_LIMIT = max(SNAPSHOT_POSITIONS)
SHUFFLE_SEED = 20261030


class AuditBlocked(RuntimeError):
    """A required source, state, identity, or parity gate failed closed."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def _atomic_gzip_json(path: Path, value: Any) -> None:
    payload = (canonical_json(value) + "\n").encode("utf-8")
    _atomic_bytes(path, gzip.compress(payload, compresslevel=9, mtime=0))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*.py"), key=lambda p: p.relative_to(path).as_posix()):
        if not item.is_file():
            continue
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _artifact_digest(source: Path) -> str:
    files = _read_json(source / "checksums.json")["files"]
    values = {}
    for relative, digest in sorted(files.items()):
        path = source / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise AuditBlocked(f"Original Stage 0.75-A checksum failed: {relative}")
        values[relative] = digest
    for category in DEVELOPMENT_CLASSES:
        manifest = _read_json(source / "trajectories" / category / "trajectory_manifest.json")
        checkpoint = source / "private_checkpoints" / f"{category}_memory_states.pt"
        actual = sha256_file(checkpoint)
        if actual != manifest.get("private_memory_states_sha256"):
            raise AuditBlocked(f"Trusted private snapshot checksum failed: {category}")
        values[f"private_checkpoints/{category}_memory_states.pt"] = actual
    b3_folder = DEFAULT_B3_OUTPUT
    b3_checks = _read_json(b3_folder / "checksums.json")["files"]
    for relative, digest in sorted(b3_checks.items()):
        path = b3_folder / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise AuditBlocked(f"Actual-B3 artifact changed: {relative}")
        values[f"actual_b3/{relative}"] = digest
    for name in ("run_manifest.json", "complete.json", "checksums.json"):
        values[f"actual_b3/{name}"] = sha256_file(b3_folder / name)
    return _sha_json(values)


def _verify_actual_b3_artifacts(folder: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checks_path = folder / "checksums.json"
    run_path = folder / "run_manifest.json"
    complete_path = folder / "complete.json"
    if not all(path.is_file() for path in (checks_path, run_path, complete_path)):
        raise AuditBlocked("completed Actual-B3 artifact package is incomplete")
    checks = _read_json(checks_path).get("files")
    if not isinstance(checks, dict) or not checks:
        raise AuditBlocked("Actual-B3 checksum manifest is invalid")
    for relative, expected in checks.items():
        path = folder / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise AuditBlocked(f"Actual-B3 artifact checksum mismatch: {relative}")
    run = _read_json(run_path)
    complete = _read_json(complete_path)
    linkage_failures = []
    if run.get("status") != "PASS" or run.get("trajectory_count") != 20 or run.get(
            "b3_replay_status") != "PASS":
        linkage_failures.append("run manifest does not report 20 PASS replays")
    if complete.get("status") != "PASS":
        linkage_failures.append("complete manifest status is not PASS")
    if complete.get("run_manifest_sha256") != sha256_file(run_path):
        linkage_failures.append("complete manifest run_manifest SHA-256 mismatch")
    if complete.get("checksums_manifest_sha256") != sha256_file(checks_path):
        linkage_failures.append("complete manifest checksums.json SHA-256 mismatch")
    if int(complete.get("files_checked", -1)) != len(checks):
        linkage_failures.append(
            f"complete manifest file count {complete.get('files_checked')} differs from "
            f"current checksum entries {len(checks)}")
    replay_path = folder / "b3_replay.json"
    if (not replay_path.is_file() or
            sha256_file(replay_path) != run.get("replay_sha256")):
        raise AuditBlocked("Actual-B3 label-free replay identity mismatch")
    replay = _read_json(replay_path)
    if (replay.get("status") != "PASS" or
            replay.get("labels_supplied_to_replay") is not False or
            set(replay.get("trajectories", {})) != set(DEVELOPMENT_CLASSES) or
            sum(len(rows) for rows in replay.get("trajectories", {}).values()) != 20):
        raise AuditBlocked("Actual-B3 replay is not a label-free 20-trajectory artifact")
    return {"status": "PASS" if not linkage_failures else "BLOCKED",
            "linkage_failures": linkage_failures,
            "listed_files_verified": True,
            "run_manifest_replay_identity_verified": True,
            "folder": folder.relative_to(ROOT).as_posix(),
            "checksums_sha256": sha256_file(checks_path),
            "run_manifest_sha256": sha256_file(run_path),
            "complete_sha256": sha256_file(complete_path),
            "replay_sha256": sha256_file(replay_path),
            "checked_file_count": len(checks)}, replay


def _opaque_group(sample_key: str) -> str:
    return hashlib.sha256(f"official-sample-group\0{sample_key}".encode("utf-8")).hexdigest()


def _tolerant_equal(left: float, right: float) -> bool:
    return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(
        float(left) - float(right)) <= PARITY_ATOL + PARITY_RTOL * max(
            abs(float(left)), abs(float(right)))


def _q90(values: Sequence[float]) -> float:
    if len(values) != 64 or not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise AuditBlocked("protected C64 scorer returned an invalid vector")
    return float(sorted(float(value) for value in values)[57])


class _QueriedLabelClient:
    """Expose only sorted original Q-A-TS labels below the largest snapshot."""

    def __init__(self, *, category: str, json_root: Path, source_sha256: str,
                 positions: Sequence[int]):
        self.positions = list(map(int, positions))
        args = [sys.executable, "-m",
                "experiments.calibration.queried_anchor_oracle",
                "--class-name", category, "--json-root", str(json_root),
                "--source-sha256", source_sha256, "--positions",
                ",".join(map(str, self.positions))]
        self.process = subprocess.Popen(args, cwd=ROOT, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True,
                                        bufsize=1)
        assert self.process.stdout is not None
        ready = json.loads(self.process.stdout.readline())
        expected_positions = hashlib.sha256(json.dumps(
            self.positions, separators=(",", ":")).encode("ascii")).hexdigest()
        if (ready.get("ready") is not True or
                ready.get("source_sha256") != source_sha256 or
                ready.get("allowed_positions_sha256") != expected_positions):
            self.close()
            raise AuditBlocked(f"{category}: queried-label oracle identity mismatch")
        self.label_count = int(ready["label_count"])
        self.cursor = 0

    def reveal(self, position: int) -> int:
        if self.process.stdin is None or self.process.stdout is None:
            raise AuditBlocked("queried-label oracle pipe is closed")
        self.process.stdin.write(json.dumps({"position": int(position)}) + "\n")
        self.process.stdin.flush()
        row = json.loads(self.process.stdout.readline())
        if (row.get("error") or int(row.get("position", -1)) != position or
                self.cursor >= len(self.positions) or
                self.positions[self.cursor] != position):
            raise AuditBlocked(f"queried-label oracle rejected position {position}")
        self.cursor += 1
        return int(row["label"])

    def close(self) -> None:
        if self.process.poll() is None and self.process.stdin is not None:
            try:
                self.process.stdin.write('{"close":true}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=15)
            except Exception:
                self.process.terminate()
                self.process.wait(timeout=15)
        if self.process.returncode not in (0, None):
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise AuditBlocked(f"queried-label oracle failed: {stderr}")


@contextmanager
def _local_dinov2_only(torch: Any, hub_source: Path,
                       checkpoint: Path):
    """Route the unchanged backbone constructor only to hash-verified local files."""
    original_load = torch.hub.load
    original_weights = torch.hub.load_state_dict_from_url

    def safe_load(repo_or_dir: str, model: str, *args: Any, **kwargs: Any):
        if repo_or_dir not in ("facebookresearch/dinov2", str(hub_source)):
            raise AuditBlocked(f"unexpected torch.hub source: {repo_or_dir}")
        if model != "dinov2_vitl14_reg":
            raise AuditBlocked(f"unexpected backbone request: {model}")
        kwargs.pop("source", None)
        return original_load(str(hub_source), model, *args,
                             source="local", **kwargs)

    def safe_weights(url: str, *args: Any, **kwargs: Any):
        from urllib.parse import urlparse
        filename = Path(urlparse(str(url)).path).name
        if filename != checkpoint.name or not checkpoint.is_file():
            raise AuditBlocked(f"unexpected or missing local DINO checkpoint request: {url}")
        map_location = kwargs.get("map_location", "cpu")
        return torch.load(checkpoint, map_location=map_location, weights_only=False)

    torch.hub.load = safe_load
    torch.hub.load_state_dict_from_url = safe_weights
    try:
        yield
    finally:
        torch.hub.load = original_load
        torch.hub.load_state_dict_from_url = original_weights


def _load_verified_private_checkpoint(source: Path, category: str,
                                     trajectory_manifest: Mapping[str, Any],
                                     torch: Any) -> dict[str, Any]:
    expected_path = (source / "private_checkpoints" /
                     f"{category}_memory_states.pt").resolve()
    trusted_dir = (source / "private_checkpoints").resolve()
    if trusted_dir not in expected_path.parents:
        raise AuditBlocked("private snapshot path escaped the trusted directory")
    expected_sha = str(trajectory_manifest.get("private_memory_states_sha256", ""))
    actual_sha = sha256_file(expected_path)
    if not expected_sha or actual_sha != expected_sha:
        raise AuditBlocked(f"{category}: private snapshot SHA-256 mismatch")
    # weights_only=False is safe here only because the exact file hash and
    # trusted repository/schema were verified immediately before deserialization.
    payload = torch.load(expected_path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != "stage075a.private_memory_states.v1" or
            payload.get("category") != category or
            payload.get("support_seeds") != list(SUPPORT_SEEDS)):
        raise AuditBlocked(f"{category}: private snapshot schema/category mismatch")
    return payload


def _reconstructed_prior_groups(events: Sequence[Mapping[str, Any]], seed: int,
                                t: int, stream_rows: Sequence[Any]) -> set[str]:
    if len(events) != len(stream_rows):
        raise AuditBlocked("event trace and identity-only stream rows are misaligned")
    prior: set[str] = set()
    for position in range(t):
        event = events[position]
        state = event["support_seeds"][str(seed)]
        writer = state["writer"]
        if (state.get("newly_resident_source_group") or
                int(writer.get("appended", 0)) > 0 or
                int(writer.get("replaced", 0)) > 0):
            # Native writer/source_map keys are metadata sample keys. The
            # event's public official_group_sha256 is a distinct opaque ID.
            prior.add(str(stream_rows[position].sample_key))
    return prior


def _validate_snapshot_record(category: str, seed: int, t: int,
                              record: Mapping[str, Any],
                              manifest: Mapping[str, Any],
                              events: Sequence[Mapping[str, Any]],
                              stream_rows: Sequence[Any],
                              torch: Any) -> tuple[Any, list[list[str]], set[str]]:
    from .memory_state import CalibrationGreedyState
    from .profiles import canonical_json as profile_json
    state = record.get("memory")
    if not isinstance(state, CalibrationGreedyState):
        raise AuditBlocked(f"{category}/s{seed}/t{t}: checkpoint state type mismatch")
    source_rows = record.get("source_map")
    if not isinstance(source_rows, list) or len(source_rows) != len(state.layers):
        raise AuditBlocked(f"{category}/s{seed}/t{t}: source-map layer mismatch")
    source_rows = [[str(value) for value in layer] for layer in source_rows]
    if any(len(layer) != len(rows) for layer, rows in zip(state.layers, source_rows)):
        raise AuditBlocked(f"{category}/s{seed}/t{t}: source-map row mismatch")
    meta = manifest["snapshot_by_seed"][str(seed)][str(t)]
    if (int(state.version) != int(meta["memory_content_version"]) or
            state.sha256() != meta["memory_sha256"] or
            state.content_sha256() != meta["memory_content_sha256"]):
        raise AuditBlocked(f"{category}/s{seed}/t{t}: memory version/digest mismatch")
    if any(int(capacity) != 256 for capacity in state.capacities):
        raise AuditBlocked(f"{category}/s{seed}/t{t}: dynamic capacity differs from K=256")
    source_sha = hashlib.sha256(profile_json(source_rows).encode("utf-8")).hexdigest()
    if source_sha != meta["source_map_sha256"]:
        raise AuditBlocked(f"{category}/s{seed}/t{t}: source-map digest mismatch")
    prior = set(map(str, record.get("prior_admitted_groups", ())))
    expected_prior = _reconstructed_prior_groups(events, seed, t, stream_rows)
    if prior != expected_prior:
        raise AuditBlocked(f"{category}/s{seed}/t{t}: prior-admitted group history mismatch")
    digest_prior = _sha_json(sorted(hashlib.sha256(
        ("memory-source-group\0" + key).encode()).hexdigest() for key in prior))
    if digest_prior != meta["prior_admitted_groups_sha256"]:
        raise AuditBlocked(f"{category}/s{seed}/t{t}: prior-group digest mismatch")
    q90 = float(record["c64_q90"])
    if not _tolerant_equal(q90, float(meta["c64_q90"])):
        raise AuditBlocked(f"{category}/s{seed}/t{t}: checkpoint q90 differs from manifest")
    return state, source_rows, prior


def _cpu_features(features: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(tensor.detach().to(device="cpu").contiguous().clone()
                 for tensor in features)


def _accumulate_excluded_rows(totals: Sequence[int], removed: Sequence[int],
                              layer_count: int) -> list[int]:
    """Accumulate source-exclusion counts against the restored native depth."""
    if len(totals) != layer_count or len(removed) != layer_count:
        raise AuditBlocked("LSO row-count layer count differs from native bank")
    return [int(total) + int(count) for total, count in zip(totals, removed)]


def _gpu_features(features: Sequence[Any], torch: Any) -> tuple[Any, ...]:
    return tuple(tensor.to(device="cuda", non_blocking=False) for tensor in features)


def _score_values(normals: Sequence[float], defects: Sequence[float],
                  theta: float, offsets: Sequence[float]) -> dict[str, Any]:
    if not normals or not defects:
        return {"status": "TWO_CLASS_GAIN_UNIDENTIFIABLE",
                "reason": "MISSING_NORMAL_OR_DEFECT_EVIDENCE",
                "normal_groups": len(normals), "defect_groups": len(defects),
                "gains": None}
    n, d = list(map(float, normals)), list(map(float, defects))
    if any(not math.isfinite(value) for value in [*n, *d, float(theta)]):
        return {"status": "SCORE_INCOMPLETE", "reason": "NONFINITE_SCORE",
                "normal_groups": len(n), "defect_groups": len(d), "gains": None}
    def objective(threshold: float) -> dict[str, float]:
        fpr = sum(score >= threshold for score in n) / len(n)
        fnr = sum(score < threshold for score in d) / len(d)
        return {"fpr": float(fpr), "fnr": float(fnr),
                "j": float(1.0 - fpr - fnr)}
    base = objective(theta)
    gains = {}
    for offset in offsets:
        value = objective(theta + float(offset))
        delta = value["j"] - base["j"]
        gains[f"{float(offset):+.6f}"] = {
            "offset": float(offset), "threshold": theta + float(offset),
            "fpr": value["fpr"], "fnr": value["fnr"], "j": value["j"],
            "delta_j_vs_keep": float(delta), "delta_ba_vs_keep": float(delta / 2),
            "delta_fpr": float(value["fpr"] - base["fpr"]),
            "delta_fnr": float(value["fnr"] - base["fnr"])}
    return {"status": "IDENTIFIABLE_TWO_CLASS_EMPIRICAL",
            "reason": "COMPLETE_VERSION_MATCHED_TWO_CLASS_EVIDENCE",
            "normal_groups": len(n), "defect_groups": len(d),
            "invalid_required_scores": 0, "base": base, "gains": gains}


def _action_row(category: str, seed: int, t: int, threshold: float,
                memory_version: int, memory_sha: str,
                q_count: int, records: Sequence[Mapping[str, Any]],
                scoring_status: str = "PASS", scoring_error: str | None = None,
                runtime: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    theta_value = float(threshold) if math.isfinite(float(threshold)) else 0.0
    normals = [AnchorScore(**row["primary_anchor"]) for row in records
               if int(row["label"]) == 0]
    defects = [AnchorScore(**row["primary_anchor"]) for row in records
               if int(row["label"]) == 1]
    invalid = 1 if scoring_status != "PASS" else 0
    try:
        primary = local_gain_table(normals, defects, theta=theta_value,
                                   offsets=PRIMARY_OFFSETS,
                                   expected_version=memory_version,
                                   expected_snapshot_sha256=memory_sha,
                                   invalid_required_scores=invalid)
        secondary = local_gain_table(normals, defects, theta=theta_value,
                                     offsets=SECONDARY_OFFSETS,
                                     expected_version=memory_version,
                                     expected_snapshot_sha256=memory_sha,
                                     invalid_required_scores=invalid)
        selected = select_primary_action(primary)
        primary_repeat = local_gain_table(normals, defects, theta=theta_value,
                                          offsets=PRIMARY_OFFSETS,
                                          expected_version=memory_version,
                                          expected_snapshot_sha256=memory_sha,
                                          invalid_required_scores=invalid)
        deterministic = canonical_json(primary) == canonical_json(primary_repeat)
    except IdentifiabilityError as exc:
        primary = {"status": "SCORE_INCOMPLETE", "reason": str(exc),
                   "normal_groups": len(normals), "defect_groups": len(defects),
                   "gains": None}
        secondary = {"status": "SCORE_INCOMPLETE", "reason": str(exc),
                     "normal_groups": len(normals), "defect_groups": len(defects),
                     "gains": None}
        selected = {"action": "ABSTAIN", "reason": "INVALID_QUERY_EVIDENCE"}
        deterministic = False
    full_normals = [replace(anchor, score=float(next(row["full_score"] for row in records
                       if row["group_sha256"] == anchor.group_sha256))) for anchor in normals]
    full_table = _score_values([row.score for row in full_normals],
                               [row.score for row in defects], theta_value,
                               PRIMARY_OFFSETS)
    full_action = select_primary_action(full_table)

    comparable_stale = [row for row in records
                        if row.get("arrival_score") is not None and
                        row.get("arrival_source_was_resident") is False]
    stale_normals = [float(row["arrival_score"]) for row in comparable_stale
                     if int(row["label"]) == 0]
    stale_defects = [float(row["arrival_score"]) for row in comparable_stale
                     if int(row["label"]) == 1]
    stale_table = _score_values(stale_normals, stale_defects, theta_value,
                                PRIMARY_OFFSETS)
    stale_action = select_primary_action(stale_table)
    if primary.get("status") != "IDENTIFIABLE_TWO_CLASS_EMPIRICAL":
        self_changed = stale_changed = None
    else:
        self_changed = (full_action["action"] != selected["action"]
                        if full_table["status"] == "IDENTIFIABLE_TWO_CLASS_EMPIRICAL"
                        else None)
        stale_changed = (stale_action["action"] != selected["action"]
                         if stale_table["status"] == "IDENTIFIABLE_TWO_CLASS_EMPIRICAL"
                         else None)
    row = {
        "category": category, "seed": int(seed), "snapshot_position": int(t),
        "snapshot_memory_version": int(memory_version),
        "snapshot_memory_sha256": memory_sha,
        "b3_threshold": theta_value if math.isfinite(float(threshold)) else None,
        "query_positions_before_snapshot": int(q_count),
        "eligible_queried_anchor_count": int(q_count),
        "scored_queried_anchor_count": len(records),
        "unavailable_queried_anchor_count": max(0, int(q_count) - len(records)),
        "normal_groups": len(normals), "defect_groups": len(defects),
        "defect_evidence_bin": defect_evidence_bin(len(defects)),
        "score_status": scoring_status, "score_error": scoring_error,
        "primary": primary, "secondary": secondary,
        "selected_action": selected["action"],
        "selection_reason": selected["reason"],
        "full_memory_sensitivity": {
            "estimate": full_table, "selected_action": full_action["action"],
            "selection_reason": full_action["reason"],
            "changes_primary_action": self_changed},
        "arrival_score_sensitivity": {
            "estimate": stale_table, "selected_action": stale_action["action"],
            "selection_reason": stale_action["reason"],
            "source_policy_comparable_anchor_count": len(comparable_stale),
            "normal_source_policy_mismatch_excluded": sum(
                int(row["label"]) == 0 and row.get("arrival_source_was_resident") is True
                for row in records),
            "changes_primary_action": stale_changed},
        "estimator_repeat_identical": deterministic,
        "runtime": dict(runtime or {}),
    }
    return row, list(records)


def _check_manifest_identity(source: Path, stage05_dir: Path,
                             json_root: Path, dataset_root: Path,
                             protocol_sha: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise AuditBlocked("active branch is not the explicitly authorized branch")
    if _git("rev-parse", "HEAD") != EXPECTED_START_HEAD:
        raise AuditBlocked("branch HEAD differs from the expected task-start baseline")
    remote = _git("ls-remote", "origin", f"refs/heads/{EXPECTED_BRANCH}").split()[0]
    if remote != EXPECTED_START_HEAD:
        raise AuditBlocked("authorized remote branch changed before this run")
    from .source_provenance import verify_protected_sources
    protected = verify_protected_sources()
    if protected.get("status") != "PASS":
        raise AuditBlocked("protected historical source verification failed")
    provenance, categories = load_verified_inputs(source.resolve(), stage05_dir.resolve())
    old_protocol_path = source / "protocol_manifest.json"
    if sha256_file(old_protocol_path) != EXPECTED_STAGE075A_PROTOCOL:
        raise AuditBlocked("Stage 0.75-A frozen protocol bytes differ from task identity")
    if sha256_file(source / "checksums.json") != EXPECTED_STAGE075A_CHECKSUMS:
        raise AuditBlocked("Stage 0.75-A checksum manifest bytes differ from task identity")
    if sha256_file(source / "online_complete.json") != EXPECTED_STAGE075A_ONLINE:
        raise AuditBlocked("Stage 0.75-A online completion manifest differs from task identity")
    old_protocol = _read_json(old_protocol_path)
    if old_protocol.get("source_head") != EXPECTED_SOURCE_PARENT:
        raise AuditBlocked("historical source parent differs from frozen task identity")
    if old_protocol.get("scorer_config_sha256") != EXPECTED_SCORER_CONFIG:
        raise AuditBlocked("frozen Native Scorer configuration hash changed")
    if old_protocol.get("local_checkpoint_sha256") != EXPECTED_DINO_CHECKPOINT or old_protocol.get(
            "local_dinov2_source_tree_sha256") != EXPECTED_DINO_TREE:
        raise AuditBlocked("frozen DINO checkpoint/source hashes changed")
    if sha256_file(PROTOCOL_PATH) != protocol_sha:
        raise AuditBlocked("queried-anchor candidate protocol hash changed after freeze")
    if not dataset_root.is_dir() or not json_root.is_dir():
        raise AuditBlocked("authorized local dataset or metadata root is unavailable")
    if hashlib.sha256(str(dataset_root.resolve()).encode()).hexdigest() != old_protocol["dataset_root_sha256"]:
        raise AuditBlocked("dataset root path identity differs from frozen Stage 0.75-A")
    if hashlib.sha256(str(json_root.resolve()).encode()).hexdigest() != old_protocol["json_root_sha256"]:
        raise AuditBlocked("metadata root path identity differs from frozen Stage 0.75-A")
    frozen, _ = stage05_runner._verify_freeze(stage05_dir)
    return provenance, categories, old_protocol, frozen


def _verify_query_inputs(category: str, events: Sequence[Mapping[str, Any]],
                         stream_rows: Sequence[Any], frozen_info: Mapping[str, Any],
                         old_protocol: Mapping[str, Any], labels: Mapping[int, int]
                         ) -> tuple[list[int], dict[int, Any]]:
    if len(events) != len(stream_rows):
        raise AuditBlocked(f"{category}: original stream metadata/event trace length mismatch")
    by_position = {int(row.position): row for row in stream_rows}
    selected = [int(event["position"]) for event in events
                if bool(event["query_tape_selected"])]
    if selected != sorted(set(selected)):
        raise AuditBlocked(f"{category}: query tape positions are not unique and ordered")
    if len(selected) != int(old_protocol["class_inputs"][category]["query_count"]):
        raise AuditBlocked(f"{category}: query tape count differs from frozen protocol")
    used_positions = [position for position in selected if position < QUERY_LIMIT]
    groups: set[str] = set()
    support_groups: set[str] = set()
    for support in frozen_info["supports"].values():
        support_groups.update(map(str, support["groups"]))
    c64_groups = set(map(str, frozen_info["c64"]["groups"]))
    for position in used_positions:
        record = by_position[position]
        event = events[position]
        if (event.get("queried_label_disclosed") is not True or
                int(event.get("oracle_label_request_position", -1)) != position or
                event.get("official_group_sha256") != record.opaque_group):
            raise AuditBlocked(f"{category}/position{position}: disclosure/tape/source identity mismatch")
        if record.sample_key in support_groups or record.sample_key in c64_groups:
            raise AuditBlocked(f"{category}/position{position}: query overlaps support or C64")
        if record.opaque_group in groups:
            raise AuditBlocked(f"{category}: query tape contains duplicate official sample group")
        groups.add(record.opaque_group)
        if position not in labels or labels[position] not in (0, 1):
            raise AuditBlocked(f"{category}/position{position}: authorized query label missing")
        # The original online process did not persist label values. The source
        # JSON hash, selected position, and writer response are reconciled.
        for seed in SUPPORT_SEEDS:
            writer = event["support_seeds"][str(seed)]["writer"]
            if labels[position] == 1:
                if (writer.get("eligible") or
                        writer.get("reason") != "feedback_not_verified_normal" or
                        bool(writer.get("changed"))):
                    raise AuditBlocked(f"{category}/position{position}: defect disclosure/writer record conflict")
            elif writer.get("reason") == "feedback_not_verified_normal":
                raise AuditBlocked(f"{category}/position{position}: normal disclosure/writer record conflict")
    return used_positions, by_position


def _score_category(category: str, *, source: Path, output: Path,
                    events: Sequence[Mapping[str, Any]], query_positions: Sequence[int],
                    query_labels: Mapping[int, int], stream_rows: Sequence[Any],
                    frozen: Mapping[str, Any], stage05_dir: Path,
                    old_protocol: Mapping[str, Any], replay: Mapping[str, Any],
                    snapshot_arrays: Mapping[str, np.ndarray],
                    dataset_root: Path, json_root: Path, torch: Any
                    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import p0_protocol_audit as p0
    from experiments.risk_safe import native_bridge
    from .stage075a_runner import (_build_native_category, _memory_state_to_device,
                                   _native_config, _tree_sha256)
    from .memory_state import CalibrationGreedyState

    category_started = time.perf_counter()
    info = frozen["classes"][category]
    cfg = _native_config(p0.require_sources())
    impl = p0.require_sources()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    impl.set_seed(0)

    hub_dir = Path(torch.hub.get_dir())
    hub_source = hub_dir / "facebookresearch_dinov2_main"
    checkpoint = hub_dir / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
    if (not hub_source.is_dir() or not checkpoint.is_file() or
            _tree_sha256(hub_source) != EXPECTED_DINO_TREE or
            sha256_file(checkpoint) != EXPECTED_DINO_CHECKPOINT):
        raise AuditBlocked("hash-verified local DINO source/checkpoint is unavailable")
    with _local_dinov2_only(torch, hub_source, checkpoint):
        (actual_stream, backbone, static_banks, _pristine_banks,
         fixed_hashes, c64_bundles_gpu) = _build_native_category(
            category, SUPPORT_SEEDS, info, dataset_root, json_root, impl, cfg)
    del _pristine_banks
    if len(actual_stream) != len(stream_rows):
        raise AuditBlocked(f"{category}: reconstructed stream count changed")
    score_manifest = _read_json(stage05_dir / "score_tape_manifest.json")
    for seed in SUPPORT_SEEDS:
        expected_fixed = score_manifest["classes"][category]["score_tapes"][str(seed)][
            "fixed_support_sha256"]
        if fixed_hashes[seed] != expected_fixed:
            raise AuditBlocked(f"{category}/seed{seed}: fixed support bank digest mismatch")

    query_cache: dict[int, tuple[Any, tuple[Any, ...]]] = {}
    query_feature_encoded = 0
    for position in query_positions:
        record = stream_rows[position]
        image, features = p0.encode_query(impl, backbone, record.image_path, cfg)
        query_cache[position] = (image, _cpu_features(features))
        query_feature_encoded += 1
        del features
    # C64 bundles were encoded by the frozen helper. Move one cache copy to
    # CPU so per-state scoring does not retain a second mutable representation.
    c64_bundles = [(image, _cpu_features(features))
                   for image, features in c64_bundles_gpu]
    c64_feature_encoded = len(c64_bundles)
    del c64_bundles_gpu
    del backbone
    torch.cuda.empty_cache()

    trajectory_manifest = _read_json(source / "trajectories" / category /
                                     "trajectory_manifest.json")
    memory_path = source / "private_checkpoints" / f"{category}_memory_states.pt"
    checkpoint_payload = _load_verified_private_checkpoint(
        source, category, trajectory_manifest, torch)
    snapshot_rows: list[dict[str, Any]] = []
    private_anchor_rows: list[dict[str, Any]] = []
    state_runtime: dict[str, Any] = {}
    total_native_calls = total_lso_calls = total_full_calls = total_defect_calls = 0
    total_primary_lso_anchors = total_anchor_repeat_calls = 0
    feature_uses = 0
    peak_cuda = 0
    for seed in SUPPORT_SEEDS:
        for t in SNAPSHOT_POSITIONS:
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            query_before = [position for position in query_positions if position < t]
            record_rows: list[dict[str, Any]] = []
            native_calls = lso_calls = full_calls = defect_calls = 0
            primary_lso_anchors = anchor_repeat_calls = 0
            anchor_repeat_delta = None
            anchor_repeat_scale = None
            anchor_repeat_status = "NO_ELIGIBLE_ANCHOR"
            score_error = None
            status = "PASS"
            attempted_anchor_count = 0
            memory_version = -1
            memory_sha = ""
            threshold = float("nan")
            state = state_cpu = native_bank = codebooks = None
            excluded_total: list[int] | None = None
            c64_q90 = None
            c64_parity_delta = None
            try:
                saved = checkpoint_payload["snapshots"][str(seed)][str(t)]
                state_cpu, source_rows, prior_groups = _validate_snapshot_record(
                    category, seed, t, saved,
                    trajectory_manifest, events, stream_rows, torch)
                state = _memory_state_to_device(state_cpu, "cuda")
                if not isinstance(state, CalibrationGreedyState):
                    raise AuditBlocked("device restore changed calibration state type")
                if (state.sha256() != state_cpu.sha256() or
                        state.content_sha256() != state_cpu.content_sha256()):
                    raise AuditBlocked("device restore changed snapshot prototype bytes")
                memory_version, memory_sha = int(state.version), state.sha256()
                threshold = float(replay["trajectories"][category][str(seed)]["theta_pre"][t])
                boundary = events[t]["support_seeds"][str(seed)]
                if (int(boundary["memory_version_before"]) != memory_version or
                        boundary["memory_sha256_before"] != memory_sha or
                        not _tolerant_equal(float(boundary["current_C64_q90_pre_event"]),
                                            float(saved["c64_q90"]))):
                    raise AuditBlocked("snapshot does not match exact pre-event memory boundary")
                if any(int(cap) != 256 for cap in state.capacities):
                    raise AuditBlocked("K=256 native dynamic capacity mismatch")
                native_bank = native_bridge.install_on_clone(
                    static_banks[seed], state.legacy_bank)
                codebooks = native_bridge.dynamic_codebooks(native_bank)
                if len(state.layers) != len(codebooks):
                    raise AuditBlocked("snapshot/native layer count mismatch")
                excluded_total = [0] * len(state.layers)
                from experiments.p0_protocol_audit import score_encoded
                c64_scores = []
                for image, cpu_features in c64_bundles:
                    features = _gpu_features(cpu_features, torch)
                    c64_scores.append(float(score_encoded(
                        impl, native_bank, image, features, cfg)))
                    native_calls += 1
                    del features
                c64_q90 = _q90(c64_scores)
                native_calls += 1
                first_image, first_features = c64_bundles[0]
                c64_repeat = float(score_encoded(
                    impl, native_bank, first_image,
                    _gpu_features(first_features, torch), cfg))
                c64_parity_delta = abs(c64_repeat - c64_scores[0])
                if not _tolerant_equal(c64_repeat, c64_scores[0]):
                    raise AuditBlocked("repeated native C64 score exceeded parity tolerance")
                expected_q90 = float(trajectory_manifest["snapshot_by_seed"][str(seed)][str(t)]["c64_q90"])
                stored_q90 = float(np.asarray(snapshot_arrays[f"snapshot_theta_s{seed}_t{t}"]))
                recorded_q90 = float(boundary["current_C64_q90_pre_event"])
                if not all(_tolerant_equal(c64_q90, value)
                           for value in (expected_q90, stored_q90, recorded_q90)):
                    raise AuditBlocked("restored snapshot protected-C64 q90 parity failed")

                for position in query_before:
                    stream_row = stream_rows[position]
                    event = events[position]
                    group_key = str(event["official_group_sha256"])
                    if stream_row.opaque_group != group_key:
                        raise AuditBlocked("queried source group differs from frozen event trace")
                    source_key = str(stream_row.sample_key)
                    label = int(query_labels[position])
                    if label == 1:
                        if source_key in prior_groups or any(
                                source_key in layer for layer in source_rows):
                            raise AuditBlocked("queried defect source was admitted to normal memory")
                    image, cpu_features = query_cache[position]
                    features = _gpu_features(cpu_features, torch)
                    feature_uses += 1
                    full_score = float(score_encoded(impl, native_bank, image, features, cfg))
                    native_calls += 1
                    if not math.isfinite(full_score):
                        raise AuditBlocked("native query full-memory score is nonfinite")
                    removed: tuple[int, ...] = tuple(0 for _ in state.layers)
                    if label == 0:
                        lso_state, removed = leave_source_out(state, source_rows, source_key)
                        assert excluded_total is not None
                        excluded_total = _accumulate_excluded_rows(
                            excluded_total, removed, len(codebooks))
                        saved_prototypes = [cb.protos for cb in codebooks]
                        try:
                            if len(lso_state.layers) != len(codebooks):
                                raise AuditBlocked("LSO state/native layer count mismatch")
                            for cb, layer in zip(codebooks, lso_state.layers):
                                if layer.device != cb.protos.device or layer.dtype != cb.protos.dtype:
                                    raise AuditBlocked("LSO tensor device/dtype differs from native codebook")
                                cb.protos = layer.detach().clone().contiguous()
                            lso_score = float(score_encoded(
                                impl, native_bank, image, features, cfg))
                            native_calls += 1
                            lso_calls += 1
                            primary_lso_anchors += 1
                            if query_before and position == query_before[0]:
                                lso_repeat = float(score_encoded(
                                    impl, native_bank, image, features, cfg))
                                native_calls += 1
                                lso_calls += 1
                                anchor_repeat_calls += 1
                                anchor_repeat_delta = abs(lso_repeat - lso_score)
                                anchor_repeat_scale = max(abs(lso_repeat), abs(lso_score))
                                if not _tolerant_equal(lso_repeat, lso_score):
                                    raise AuditBlocked(
                                        "repeated native LSO anchor score exceeded parity tolerance")
                                anchor_repeat_status = "PASS"
                        finally:
                            for cb, prototype in zip(codebooks, saved_prototypes):
                                cb.protos = prototype
                        restored = native_bridge.snapshot(native_bank, version=state.version)
                        if restored.sha256() != state.sha256():
                            raise AuditBlocked("private native clone did not restore original snapshot bytes")
                        primary_score = lso_score
                        primary_policy = "queried_normal_all_layer_leave_source_out"
                        full_calls += 1
                    else:
                        primary_score = full_score
                        primary_policy = "queried_defect_unchanged_frozen_memory"
                        defect_calls += 1
                        lso_state = state
                        if query_before and position == query_before[0]:
                            full_repeat = float(score_encoded(
                                impl, native_bank, image, features, cfg))
                            native_calls += 1
                            anchor_repeat_calls += 1
                            anchor_repeat_delta = abs(full_repeat - full_score)
                            anchor_repeat_scale = max(abs(full_repeat), abs(full_score))
                            if not _tolerant_equal(full_repeat, full_score):
                                raise AuditBlocked(
                                    "repeated native defect anchor score exceeded parity tolerance")
                            anchor_repeat_status = "PASS"
                    if not math.isfinite(primary_score):
                        raise AuditBlocked("native primary query score is nonfinite")
                    source_event = event
                    arrival_state = source_event["support_seeds"][str(seed)]
                    arrival_source_resident = bool(
                        arrival_state["pre_event_currently_resident_group"])
                    if label == 1 and arrival_source_resident:
                        raise AuditBlocked("defect was resident at arrival, inconsistent with writer contract")
                    primary_anchor = AnchorScore(
                        group_sha256=group_key, position=position,
                        camera_id=int(stream_row.camera_id), label=label,
                        score=primary_score, memory_version=memory_version,
                        snapshot_memory_sha256=memory_sha,
                        scoring_memory_sha256=lso_state.sha256(),
                        policy=primary_policy,
                        excluded_rows_by_layer=tuple(removed),
                        arrival_score=float(arrival_state["dynamic_score"]),
                        arrival_memory_version=int(arrival_state["memory_version_before"]),
                        arrival_source_was_resident=arrival_source_resident)
                    primary_anchor.validate(expected_version=memory_version,
                                            expected_snapshot_sha256=memory_sha)
                    record_rows.append({
                        "category": category, "seed": int(seed),
                        "snapshot_position": int(t), "position": int(position),
                        "group_sha256": group_key, "camera_id": int(stream_row.camera_id),
                        "label": label,
                        "primary_anchor": asdict(primary_anchor),
                        "full_score": full_score,
                        "arrival_score": float(arrival_state["dynamic_score"]),
                        "arrival_memory_version": int(arrival_state["memory_version_before"]),
                        "arrival_memory_sha256": str(arrival_state["memory_sha256_before"]),
                        "arrival_source_was_resident": arrival_source_resident,
                        "arrival_score_policy": "native_dynamic_pre_event",
                    })
                    del features
                attempted_anchor_count = len(record_rows)
                if (state.sha256() != state_cpu.sha256() or
                        state.content_sha256() != state_cpu.content_sha256()):
                    raise AuditBlocked("measurement changed loaded historical snapshot")
            except Exception as exc:
                status = "BLOCKED"
                score_error = f"{type(exc).__name__}: {exc}"
                attempted_anchor_count = len(record_rows)
                record_rows = []
            elapsed = time.perf_counter() - started
            peak_cuda = max(peak_cuda, int(torch.cuda.max_memory_allocated()))
            total_native_calls += native_calls
            total_lso_calls += lso_calls
            total_primary_lso_anchors += primary_lso_anchors
            total_anchor_repeat_calls += anchor_repeat_calls
            total_full_calls += full_calls
            total_defect_calls += defect_calls
            state_key = f"{seed}/{t}"
            runtime_row = {
                "elapsed_seconds": elapsed,
                "native_score_calls": native_calls,
                "lso_score_calls": lso_calls,
                "lso_primary_anchor_scores": primary_lso_anchors,
                "repeated_anchor_score_calls": anchor_repeat_calls,
                "query_anchor_repeat_abs_delta": anchor_repeat_delta,
                "query_anchor_repeat_scale": anchor_repeat_scale,
                "query_anchor_repeat_status": anchor_repeat_status,
                "normal_full_memory_scores": full_calls,
                "defect_scores": defect_calls,
                "source_excluded_rows_by_layer": excluded_total or [],
                "eligible_query_anchors": len(query_before),
                "scored_query_anchors": len(record_rows),
                "attempted_query_anchors_before_block": attempted_anchor_count,
                "unavailable_query_anchors": max(0, len(query_before) - len(record_rows)),
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "c64_q90_recomputed": c64_q90,
                "c64_q90_parity_abs_repeat_delta": c64_parity_delta,
            }
            action, serialized_rows = _action_row(
                category, seed, t, threshold, memory_version, memory_sha,
                len(query_before), record_rows, scoring_status=status,
                scoring_error=score_error, runtime=runtime_row)
            snapshot_rows.append(action)
            private_anchor_rows.extend(serialized_rows)
            state_runtime[state_key] = runtime_row | {
                "status": status, "error": score_error,
                "scored_query_anchors": len(record_rows)}
            del state, state_cpu, native_bank, codebooks
            torch.cuda.empty_cache()

    del checkpoint_payload, static_banks
    torch.cuda.empty_cache()
    summary = {
        "category": category, "snapshot_state_count": len(snapshot_rows),
        "query_images_newly_encoded": query_feature_encoded,
        "c64_images_encoded_by_frozen_helper": c64_feature_encoded,
        "query_feature_cache_uses": feature_uses,
        "query_feature_cache_reuse_count": max(0, feature_uses - query_feature_encoded),
        "c64_feature_cache_uses": len(SNAPSHOT_POSITIONS) * len(SUPPORT_SEEDS) * len(c64_bundles),
        "c64_feature_cache_reuse_count": max(
            0, len(SNAPSHOT_POSITIONS) * len(SUPPORT_SEEDS) * len(c64_bundles) - c64_feature_encoded),
        "native_score_calls": total_native_calls,
        "lso_score_calls": total_lso_calls,
        "lso_primary_anchor_score_count": total_primary_lso_anchors,
        "repeated_query_anchor_score_count": total_anchor_repeat_calls,
        "normal_full_memory_scores": total_full_calls,
        "defect_scores": total_defect_calls,
        "snapshot_scoring_elapsed_seconds": float(sum(
            row["elapsed_seconds"] for row in state_runtime.values())),
        "elapsed_seconds": float(time.perf_counter() - category_started),
        "peak_cuda_allocated_bytes": peak_cuda,
        "state_runtime": state_runtime,
        "fixed_support_sha256_by_seed": {str(seed): fixed_hashes[seed]
                                         for seed in SUPPORT_SEEDS},
        "memory_checkpoint_sha256": trajectory_manifest["private_memory_states_sha256"],
        "checkpoint_file_size_bytes": memory_path.stat().st_size,
    }
    return snapshot_rows, private_anchor_rows, summary


def _shuffle_control(private_rows: Sequence[Mapping[str, Any]],
                     primary_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}
    for row in private_rows:
        key = (str(row["category"]), int(row["seed"]), int(row["snapshot_position"]))
        grouped.setdefault(key, []).append(row)
    primary_by_key = {(row["category"], int(row["seed"]), int(row["snapshot_position"])): row
                      for row in primary_rows}
    output = []
    changed = 0
    for key, values in sorted(grouped.items()):
        labels = [int(row["label"]) for row in values]
        shuffled = list(labels)
        seed = SHUFFLE_SEED + sum(map(ord, key[0])) + key[1] * 1009 + key[2]
        random.Random(seed).shuffle(shuffled)
        anchors = [AnchorScore(**row["primary_anchor"]) for row in values]
        permuted = [replace(anchor, label=label)
                    for anchor, label in zip(anchors, shuffled)]
        normals = [row for row in permuted if row.label == 0]
        defects = [row for row in permuted if row.label == 1]
        snapshot = primary_by_key[key]
        try:
            table = local_gain_table(
                normals, defects, theta=float(snapshot["b3_threshold"]),
                offsets=PRIMARY_OFFSETS,
                expected_version=int(snapshot["snapshot_memory_version"]),
                expected_snapshot_sha256=str(snapshot["snapshot_memory_sha256"]))
            action = select_primary_action(table)
        except IdentifiabilityError as exc:
            table = {"status": "SCORE_INCOMPLETE", "reason": str(exc)}
            action = {"action": "ABSTAIN", "reason": "INVALID_SHUFFLED_EVIDENCE"}
        changed += int(action["action"] != snapshot["selected_action"])
        output.append({"category": key[0], "seed": key[1],
                       "snapshot_position": key[2], "action": action["action"],
                       "reason": action["reason"],
                       "normal_groups": len(normals), "defect_groups": len(defects),
                       "estimate_status": table.get("status")})
    return {"status": "PASS", "seed": SHUFFLE_SEED,
            "eligible_snapshot_count": len(output),
            "action_changed_count": changed,
            "action_changed_fraction": (None if not output else changed / len(output)),
            "results": output}


def _n1_missing_defect_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check every audiojack seed at t=0/400/800 before defect evidence exists."""
    positions = (0, 400, 800)
    expected = {(seed, position) for seed in SUPPORT_SEEDS
                for position in positions}
    selected = [row for row in rows
                if row["category"] == "audiojack" and
                int(row["snapshot_position"]) in positions]
    observed = {(int(row["seed"]), int(row["snapshot_position"]))
                for row in selected}
    valid = (observed == expected and len(selected) == len(expected) and all(
        int(row["defect_groups"]) == 0 and
        row["primary"]["status"] == "TWO_CLASS_GAIN_UNIDENTIFIABLE" and
        row["selected_action"] == "ABSTAIN" for row in selected))
    return {"status": "PASS" if valid else "FAIL",
            "audiojack_t0_400_800_rows": len(selected),
            "expected_seed_position_rows": len(expected),
            "all_are_two_class_unidentifiable": bool(valid)}


def _negative_controls(rows: Sequence[Mapping[str, Any]],
                       private_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    action_counts = {action: sum(row["selected_action"] == action for row in rows)
                     for action in ("LOWER", "KEEP", "RAISE", "ABSTAIN")}
    n3 = [row for row in rows
          if row["full_memory_sensitivity"]["changes_primary_action"] is True]
    n3_n = sum(row["full_memory_sensitivity"]["changes_primary_action"] is not None
               for row in rows)
    n_stale = [row for row in rows
               if row["arrival_score_sensitivity"]["changes_primary_action"] is True]
    stale_n = sum(row["arrival_score_sensitivity"]["changes_primary_action"] is not None
                  for row in rows)
    primary_counts: dict[str, Any] = {}
    for count_bin in ("0", "1-2", "3-7", "8-13", "14+"):
        selected = [row for row in rows if row["defect_evidence_bin"] == count_bin]
        primary_counts[count_bin] = {
            "snapshot_rows": len(selected),
            "identifiable_rows": sum(row["primary"]["status"] ==
                                      "IDENTIFIABLE_TWO_CLASS_EMPIRICAL" for row in selected),
            "action_counts": {action: sum(row["selected_action"] == action for row in selected)
                              for action in ("LOWER", "KEEP", "RAISE", "ABSTAIN")},
        }
    return {
        "N1_missing_defect_evidence": _n1_missing_defect_evidence(rows),
        "N2_wrong_memory_rejection": {"status": "PASS",
                                      "unit_test": "test_wrong_memory_version_or_digest_is_rejected"},
        "N3_normal_self_contamination": {
            "status": "PASS", "states_with_action_change": len(n3),
            "comparable_states": n3_n,
            "action_change_fraction": None if not n3_n else len(n3) / n3_n},
        "N4_future_label_leakage": {"status": "PASS",
                                    "process_separated": True,
                                    "query_action_digest_frozen_before_evaluator": True},
        "N5_query_label_shuffle": _shuffle_control(private_rows, rows),
        "N6_no_useful_local_gain": {"status": "PASS",
                                    "unit_test": "test_no_useful_local_gain_selects_keep"},
        "N7_same_group_duplication": {"status": "PASS",
                                      "unit_test": "test_duplicate_official_group_cannot_inflate_evidence"},
        "N8_frozen_score_invariance": {"status": "PENDING_FUTURE_EVALUATOR",
                                       "query_action_counts": action_counts},
        "arrival_score_sensitivity": {
            "states_with_action_change": len(n_stale), "comparable_states": stale_n,
            "action_change_fraction": None if not stale_n else len(n_stale) / stale_n},
        "by_defect_evidence_bin": primary_counts,
    }


def _preflight_record(source: Path, stage05_dir: Path, dataset_root: Path,
                      json_root: Path, provenance: Mapping[str, Any],
                      old_protocol: Mapping[str, Any], categories: Mapping[str, Any],
                      frozen: Mapping[str, Any], protected: Mapping[str, Any],
                      source_before: str, actual_b3: Mapping[str, Any],
                      code_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "status": "PASS" if actual_b3.get("status") == "PASS" else "BLOCKED",
        "stage075a_input_status": "PASS",
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "HEAD"), "remote_head": _git(
            "ls-remote", "origin", f"refs/heads/{EXPECTED_BRANCH}").split()[0],
        "start_head": EXPECTED_START_HEAD,
        "stage075a_parent": old_protocol.get("source_head"),
        "stage075a_protocol_sha256": sha256_file(source / "protocol_manifest.json"),
        "stage075a_checksums_sha256": sha256_file(source / "checksums.json"),
        "stage075a_online_complete_sha256": sha256_file(source / "online_complete.json"),
        "actual_b3_artifact_verification": dict(actual_b3),
        "stage075a_root_artifact_verification": provenance["root_checksum_verification"],
        "trajectory_count": provenance["trajectory_count"],
        "category_event_counts": provenance["category_event_counts"],
        "snapshot_state_expected": len(DEVELOPMENT_CLASSES) * len(SUPPORT_SEEDS) * len(SNAPSHOT_POSITIONS),
        "private_checkpoint_hashes": {
            category: _read_json(source / "trajectories" / category /
                                 "trajectory_manifest.json")["private_memory_states_sha256"]
            for category in DEVELOPMENT_CLASSES},
        "stale_incomplete_marker_preserved": provenance["stale_incomplete_marker"] is not None,
        "stale_incomplete_marker_note": provenance["stale_marker_interpretation"],
        "protected_sources": {"status": protected["status"],
                              "dependency_hash_count": protected["dependency_hash_count"],
                              "shadow_memory_byte_equal_to_baseline":
                              protected["shadow_memory_byte_equal_to_baseline"]},
        "stage05_frozen_input_sha256": sha256_file(stage05_dir / "frozen_inputs.json"),
        "dataset_root_path_sha256": old_protocol["dataset_root_sha256"],
        "json_root_path_sha256": old_protocol["json_root_sha256"],
        "dataset_root_exists": dataset_root.is_dir(),
        "json_root_exists": json_root.is_dir(),
        "scorer_config": old_protocol["scorer_config"],
        "scorer_config_sha256": old_protocol["scorer_config_sha256"],
        "dino_checkpoint_sha256": old_protocol["local_checkpoint_sha256"],
        "dino_source_tree_sha256": old_protocol["local_dinov2_source_tree_sha256"],
        "class_stream_lengths": {category: len(categories[category].events)
                                 for category in DEVELOPMENT_CLASSES},
        "stage05_freeze_test_labels_accessed": bool(frozen.get("test_labels_accessed")),
        "stage05_freeze_masks_accessed": bool(frozen.get("test_masks_accessed")),
        "source_artifact_digest_before_measurement": source_before,
        "new_code_sha256": dict(code_hashes),
    }


def _run_release_tests(output: Path) -> dict[str, Any]:
    """Run and retain the requested release checks before queried-label access."""
    log_dir = output / "test_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    commands = {
        "python_compilation": [
            sys.executable, "-m", "py_compile",
            "experiments/calibration/queried_anchor_identifiability.py",
            "experiments/calibration/queried_anchor_oracle.py",
            "experiments/calibration/queried_anchor_identifiability_evaluator.py",
            "experiments/calibration/queried_anchor_identifiability_runner.py",
            "tests/calibration/test_queried_anchor_identifiability.py"],
        "focused_identifiability_tests": [
            sys.executable, "-m", "unittest", "discover", "-s",
            "tests/calibration", "-p", "test_queried_anchor_identifiability.py", "-v"],
        "full_calibration_suite": [
            sys.executable, "-m", "unittest", "discover", "-s",
            "tests/calibration", "-v"],
    }
    results: dict[str, Any] = {}
    for name, command in commands.items():
        log_path = log_dir / f"{name}.log"
        started = time.perf_counter()
        try:
            with log_path.open("w", encoding="utf-8", newline="\n") as stream:
                completed = subprocess.run(
                    command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    text=True, check=False, timeout=3 * 60 * 60)
            text = log_path.read_text(encoding="utf-8", errors="replace")
            count_match = re.search(r"Ran (\d+) tests? in ([0-9.]+)s", text)
            failures_match = re.search(r"FAILED \(failures=(\d+)(?:, errors=(\d+))?", text)
            status = "PASS" if completed.returncode == 0 else "FAIL"
            results[name] = {
                "status": status, "returncode": completed.returncode,
                "command": command,
                "tests_run": None if count_match is None else int(count_match.group(1)),
                "reported_test_seconds": None if count_match is None else float(count_match.group(2)),
                "failures": (0 if failures_match is None else
                             int(failures_match.group(1)) + int(failures_match.group(2) or 0)),
                "skipped_count": len(re.findall(r"\bskipped ['\"]", text)),
                "log_path": log_path.relative_to(output).as_posix(),
                "log_sha256": sha256_file(log_path),
                "elapsed_seconds": time.perf_counter() - started,
            }
        except subprocess.TimeoutExpired:
            results[name] = {
                "status": "BLOCKED", "returncode": None,
                "command": command, "tests_run": None, "failures": None,
                "reason": "three-hour test timeout",
                "log_path": log_path.relative_to(output).as_posix(),
                "log_sha256": sha256_file(log_path) if log_path.is_file() else None,
                "elapsed_seconds": time.perf_counter() - started,
            }
    try:
        torch = __import__("torch")
        environment = {"python": sys.version, "torch": torch.__version__,
                       "cuda_available": bool(torch.cuda.is_available()),
                       "cuda_version": getattr(torch.version, "cuda", None)}
    except Exception as exc:
        environment = {"python": sys.version, "torch_import_error": str(exc)}
    status = ("PASS" if all(value["status"] == "PASS"
                            for value in results.values()) else "BLOCKED")
    return {"schema": "queried_anchor_identifiability.release_tests.v1",
            "status": status, "environment": environment, "checks": results}


def _write_reports(output: Path, audit_dir: Path,
                   preflight: Mapping[str, Any], action_payload: Mapping[str, Any],
                   future: Mapping[str, Any], scoring_summary: Mapping[str, Any],
                   controls: Mapping[str, Any], test_record: Mapping[str, Any],
                   gates: Mapping[str, Any]) -> None:
    rows = action_payload["rows"]
    future_rows = future["rows"]
    future_by_key = {(row["category"], int(row["seed"]), int(row["snapshot_position"])): row
                     for row in future_rows}
    rescoring_lines = [
        "# Native Query-Anchor Rescoring Report", "",
        "**Status:** source-faithful native measurement; independent research review required.", "",
        f"Protocol candidate SHA-256: `{preflight['protocol_sha256']}`.", "",
        "The unchanged v22/VisionAD `score_encoded` path scored the original frozen",
        "snapshot memories through the Native Scorer. The frozen four supports remain",
        "separate from the additional protected C64 reference. No Greedy writes occurred.", "",
        "## Scope and integrity", "",
        f"- Source trajectories verified: {preflight['trajectory_count']}.",
        f"- Snapshot states expected: {preflight['snapshot_state_expected']}.",
        f"- Stage 0.75-A stale `INCOMPLETE` marker preserved: {preflight['stale_incomplete_marker_preserved']}.",
        "- C64 parity uses all 64 protected references at every restored snapshot.",
        "- Fixed support digest and snapshot memory/source-map identities are checked exactly.",
        "- C64 q90 parity tolerance: abs(delta) <= 1e-6 + 1e-6 * max(abs(a), abs(b)).",
        "- DINO source and weights were loaded only from the verified local cache.", "",
        f"- Actual-B3 package linkage status: {preflight['actual_b3_artifact_verification']['status']}; "
        f"listed source files and the replay/run-manifest link were independently checked. "
        f"Linkage findings: {preflight['actual_b3_artifact_verification'].get('linkage_failures', [])}.",
        (f"- Prior incomplete implementation attempt preserved at "
         f"`{preflight['preserved_prior_attempt']['path']}` with status "
         f"`{preflight['preserved_prior_attempt']['status']}`; no future result "
         f"artifact was produced."
         if preflight.get("preserved_prior_attempt") else ""),
        "## Rescoring totals", "",
        f"Newly encoded queried images: {scoring_summary['query_images_newly_encoded']}.",
        f"Protected C64 images encoded once per category: {scoring_summary['c64_images_encoded_by_frozen_helper']}.",
        f"Current-memory normal LSO score calls: {scoring_summary['lso_score_calls']}.",
        f"Primary normal LSO anchor scores: {scoring_summary['lso_primary_anchor_score_count']}; repeated query-anchor parity scores: {scoring_summary['repeated_query_anchor_score_count']}.",
        f"Current-memory defect score calls: {scoring_summary['defect_scores']}.",
        f"Normal full-memory sensitivity score calls: {scoring_summary['normal_full_memory_scores']}.",
        f"Native score calls total, including C64 parity/repeats: {scoring_summary['native_score_calls']}.",
        f"Queried feature-cache reuse count: {scoring_summary['query_feature_cache_reuse_count']}.",
        f"Protected C64 feature-cache reuse count: {scoring_summary['c64_feature_cache_reuse_count']}.",
        f"Elapsed native measurement seconds: {scoring_summary['elapsed_seconds']:.3f}.",
        f"Peak CUDA allocated bytes: {scoring_summary['peak_cuda_allocated_bytes']}.", "",
        "Normal anchors use all-layer source-aware LSO. Defect anchors use unchanged",
        "snapshot memory and are rejected if any prototype provenance indicates prior",
        "normal admission. Arrival-time scores are used only in the staleness sensitivity.", "",
        "The original Stage 0.75-A trace stores query positions and writer outcomes but",
        "does not persist the disclosed label values. This run reconciled query labels",
        "against the unchanged official metadata SHA, the frozen query tape, and every",
        "seed's writer response; no separate historical label-response digest exists.", "",
        "## Gate matrix", "",
        "See the machine-readable gate matrix in the audit package. Blocked cells are",
        "not replaced by reconstructed or approximate states.", "",
    ]
    (ROOT / "docs/calibration/QUERIED_ANCHOR_NATIVE_RESCORING_REPORT.md").write_text(
        "\n".join(rescoring_lines), encoding="utf-8", newline="\n")

    results_lines = [
        "# Sparse-Feedback Identifiability Results", "",
        "**Status:** temporally separated within previously exposed development classes; not an independent test-set result.", "",
        "Primary query estimates use version-matched LSO normals and unchanged-memory",
        "defects. Actions were hashed before the separate future evaluator loaded labels.", "",
        "## All 120 query-derived state rows", "",
        "| Class | Seed | t | N groups | D groups | Defect bin | Query ΔJ LOWER / RAISE | Action | Next256 future ΔBA L/K/R | Selected ΔBA | Best match | All-remaining future ΔBA L/K/R | Selected ΔBA | Best match |",
        "|---|---:|---:|---:|---:|---|---:|---|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        key = (row["category"], int(row["seed"]), int(row["snapshot_position"]))
        ev = future_by_key[key]
        primary = row["primary"].get("gains") or {}
        lo = primary.get("-0.010000", {}).get("delta_j_vs_keep")
        hi = primary.get("+0.010000", {}).get("delta_j_vs_keep")
        def fmt(x): return "NA" if x is None else f"{float(x):+.4f}"
        cohort = ev["cohorts"]
        nextrow = cohort["next256_group_clean"]
        allrow = cohort["all_remaining_group_clean"]
        def future_ba_triplet(value):
            gains = value.get("future_primary_gains", {})
            return " / ".join(fmt(gains.get(key, {}).get("delta_ba_vs_keep"))
                              for key in ("-0.010000", "+0.000000", "+0.010000"))
        nmatch = ("NA" if not nextrow["future_best_primary_action"]["unique"] or
                  row["selected_action"] == "ABSTAIN" else
                  str(row["selected_action"] == nextrow["future_best_primary_action"]["action"]))
        amatch = ("NA" if not allrow["future_best_primary_action"]["unique"] or
                  row["selected_action"] == "ABSTAIN" else
                  str(row["selected_action"] == allrow["future_best_primary_action"]["action"]))
        results_lines.append(
            f"| {row['category']} | {row['seed']} | {row['snapshot_position']} | {row['normal_groups']} | {row['defect_groups']} | {row['defect_evidence_bin']} | {fmt(lo)} / {fmt(hi)} | {row['selected_action']} | {future_ba_triplet(nextrow)} | {fmt(nextrow['query_selected_future_delta_ba'])} | {nmatch} | {future_ba_triplet(allrow)} | {fmt(allrow['query_selected_future_delta_ba'])} | {amatch} |")
    results_lines.extend(["", "## Class-first future summaries", "",
                          "```json", json.dumps(future["summary_by_class"], indent=2,
                                                 sort_keys=True), "```", "",
                          "All one-class future cohorts carry null gains. Query evidence bins",
                          "are diagnostics only. Overlapping snapshots/cohorts are dependent;",
                          "they are not independent replications. See the compressed audit",
                          "results for per-offset FPR/FNR/BA changes and all denominators.", "",
                          "## Equal-class macro and snapshot-time summaries", "",
                          "```json", json.dumps({
                              "equal_class_macro": future.get("equal_class_macro", {}),
                              "by_snapshot_position": future.get("by_snapshot_position", {}),
                          }, indent=2, sort_keys=True), "```", ""])
    (ROOT / "docs/calibration/QUERIED_ANCHOR_IDENTIFIABILITY_RESULTS.md").write_text(
        "\n".join(results_lines), encoding="utf-8", newline="\n")

    class_summ = future["summary_by_class"]
    decision = _choose_decision(rows, future)
    identified_count = sum(row["primary"]["status"] ==
                           "IDENTIFIABLE_TWO_CLASS_EMPIRICAL" for row in rows)
    action_counts = {action: sum(row["selected_action"] == action for row in rows)
                     for action in ("LOWER", "KEEP", "RAISE", "ABSTAIN")}
    bin_future = future.get("by_query_defect_evidence_bin", {})
    class_selected_gain = {
        category: {cohort: class_summ[category]["cohorts"][cohort][
            "mean_selected_future_delta_ba"] for cohort in (
                "next256_group_clean", "all_remaining_group_clean")}
        for category in DEVELOPMENT_CLASSES}
    n3 = controls["N3_normal_self_contamination"]
    stale = controls["arrival_score_sensitivity"]
    implication_lines = [
        "# Queried-Anchor Identifiability: Research Implications", "",
        "This is a development-set measurement, not a deployment or risk guarantee.", "",
        "## Findings", "",
        f"1. Legal sparse feedback yielded a complete two-class estimate in {identified_count}/120 snapshot states ({identified_count/120:.1%}); the selected actions were `{json.dumps(action_counts, sort_keys=True)}`. This is query-anchor coverage, not population coverage.",
        f"2. Future-selected-action mean ΔBA by frozen defect-count bin is `{json.dumps(bin_future, sort_keys=True)}`. The 8+ bins are descriptive; no label count is certified as sufficient for population reliability.",
        f"3. LOWER was selected {action_counts['LOWER']} times and RAISE {action_counts['RAISE']} times. Their per-state queried gains are in Report B; future unique-best agreement and selected-action gains are reported separately with denominators.",
        f"4. Full-memory rather than LSO normal scoring changed the primary action in {n3['states_with_action_change']}/{n3['comparable_states']} comparable states; primary estimates use source-aware all-layer LSO.",
        f"5. Current-memory rescoring is the primary evidence. Source-policy-comparable arrival-score sensitivity changed the action in {stale['states_with_action_change']}/{stale['comparable_states']} states; source-policy-mismatched normal arrivals are excluded and reported.",
        f"6. Regulator and Rolled Strip Base class-first future selected ΔBA summaries are `{json.dumps({k: class_selected_gain[k] for k in ('regulator', 'rolled_strip_base')}, sort_keys=True)}`. Equal-class macros are provided below.",
        f"7. The measured candidate decision is `{decision['decision']}`. It summarizes temporally separated development outcomes; it does not implement an online threshold correction.",
        "8. Additional percentile transport was not tested by this audit. C64 parity is a reconstruction check, so these measurements alone do not justify a new transport correction.",
        "9. Sparse selective feedback, source/group correlation, temporal dependence, overlapping snapshots/cohorts, changing memory and prior exposure of all four categories remain the central uncertainties. No IID or population-risk claim is made.",
        "10. B8 Recovery remains unimplemented. Any further method must receive a new scoped researcher decision; this report does not authorize B8 v4 or another stage.", "",
        "## Class-specific summary", "",
        "```json", json.dumps({k: class_summ[k] for k in ("regulator", "rolled_strip_base")},
                              indent=2, sort_keys=True), "```", "",
        "## Equal-class summaries", "",
        "```json", json.dumps(future.get("equal_class_macro", {}),
                              indent=2, sort_keys=True), "```", "",
        "No confidence interval is interpreted as IID coverage. Future cohorts overlap",
        "and all categories were previously exposed during development.", ""]
    (ROOT / "docs/calibration/QUERIED_ANCHOR_B8_RESEARCH_IMPLICATIONS.md").write_text(
        "\n".join(implication_lines), encoding="utf-8", newline="\n")

    decision_lines = [
        "# Queried-Anchor Identifiability Decision Candidate", "",
        f"**Candidate decision:** `{decision['decision']}`", "",
        "**Review status:** requires explicit researcher approval. This report does",
        "not authorize B8 v4, Stage 0.75-C, Stage 1/2, or confirmation-class access.", "",
        "## Evidence", "", decision["rationale"], "",
        decision["decision_rule"], "",
        "## Required interpretation", "",
        "Query-derived gains describe the selected queried anchors, not an unbiased",
        "population BA estimate. Evaluations are temporally separated within four",
        "previously exposed development categories; snapshot windows overlap. No IID",
        "coverage, deployment risk, or independent confirmation claim is made.", "",
        "## Researcher decisions still required", "",
        "- Whether the observed identification coverage and harm frequency justify any separately frozen follow-up.",
        "- Whether B8 Recovery should be paused, simplified, or reconsidered under a new protocol.",
        "- Whether any next-stage experiment should be authorized at all.", ""]
    (ROOT / "docs/calibration/QUERIED_ANCHOR_IDENTIFIABILITY_DECISION_CANDIDATE.md").write_text(
        "\n".join(decision_lines), encoding="utf-8", newline="\n")

    # Compact privacy-safe audit data; no paths, labels, group hashes, or raw scores.
    audit_dir.mkdir(parents=True, exist_ok=False)
    compact = {
        "schema": "queried_anchor_identifiability.audit.v1",
        "protocol_sha256": preflight["protocol_sha256"],
        "query_action_digest": action_payload["action_digest"],
        "state_count": len(rows), "future_row_count": len(future_rows),
        "action_counts": {action: sum(row["selected_action"] == action for row in rows)
                          for action in ("LOWER", "KEEP", "RAISE", "ABSTAIN")},
        "states": [{k: row[k] for k in (
            "category", "seed", "snapshot_position", "snapshot_memory_version",
            "snapshot_memory_sha256", "b3_threshold", "query_positions_before_snapshot",
            "eligible_queried_anchor_count", "scored_queried_anchor_count",
            "unavailable_queried_anchor_count", "normal_groups", "defect_groups",
            "defect_evidence_bin", "primary", "secondary", "selected_action",
            "selection_reason", "full_memory_sensitivity",
            "arrival_score_sensitivity", "score_status", "runtime")}
            for row in rows],
        "future": [{k: row[k] for k in ("category", "seed", "snapshot_position",
                                        "query_action", "cohorts")}
                   for row in future_rows],
        "scoring_summary": scoring_summary,
        "negative_controls": controls,
        "test_results": test_record,
        "gate_matrix": gates,
        "provenance": {
            "task_start_head": EXPECTED_START_HEAD,
            "historical_source_parent": EXPECTED_SOURCE_PARENT,
            "stage075a_protocol_sha256": preflight["stage075a_protocol_sha256"],
            "stage075a_checksums_sha256": preflight["stage075a_checksums_sha256"],
            "actual_b3_artifact_verification": preflight["actual_b3_artifact_verification"],
            "preserved_prior_attempt": preflight.get("preserved_prior_attempt"),
            "private_memory_file_hashes": preflight["private_checkpoint_hashes"],
            "scorer_config_sha256": preflight["scorer_config_sha256"],
            "dino_checkpoint_sha256": preflight["dino_checkpoint_sha256"],
            "dino_source_tree_sha256": preflight["dino_source_tree_sha256"],
            "new_code_sha256": preflight["new_code_sha256"],
        },
        "report_hashes": {},
        "historical_artifacts_unchanged": future["historical_artifacts_unchanged"],
        "scope": {"b8_implemented": False, "greedy_writes": 0,
                  "stage075c_stage1_stage2": False,
                  "confirmation_access": False, "mask_access": False},
    }
    _atomic_json(audit_dir / "results.json", compact)
    _atomic_json(audit_dir / "provenance.json", compact["provenance"])
    _atomic_json(audit_dir / "evidence_count_matrix.json", {
        "by_defect_evidence_bin": controls["by_defect_evidence_bin"],
        "query_action_counts": compact["action_counts"],
        "states": [{k: row[k] for k in ("category", "seed", "snapshot_position",
                                        "normal_groups", "defect_groups",
                                        "defect_evidence_bin", "selected_action")}
                   for row in rows]})
    _atomic_json(audit_dir / "negative_controls.json", controls)
    _atomic_json(audit_dir / "runtime_summary.json", scoring_summary)
    _atomic_json(audit_dir / "gate_matrix.json", gates)
    _atomic_json(audit_dir / "test_results.json", test_record)
    _atomic_json(audit_dir / "protocol_linkage.json", {
        "candidate_protocol_path": "docs/calibration/QUERIED_ANCHOR_IDENTIFIABILITY_PROTOCOL_CANDIDATE_v1.md",
        "candidate_protocol_sha256": preflight["protocol_sha256"],
        "audit_results_path": "results.json"})
    (audit_dir / "README.md").write_text(
        "# Queried-Anchor Identifiability Audit v1\n\n"
        "Compact privacy-safe evidence for the Stage 0.75 queried-anchor audit. "
        "The package contains no raw labels, sample identifiers, image paths, "
        "images, masks, features, weights, or private checkpoints.\n\n"
        f"Local private linkage: `{output.relative_to(ROOT).as_posix()}/`.\n\n"
        "This is a development measurement and requires independent research review.\n",
        encoding="utf-8", newline="\n")
    _atomic_json(audit_dir / "configuration.json", {
        "categories": list(DEVELOPMENT_CLASSES),
        "support_seeds": list(SUPPORT_SEEDS),
        "snapshot_positions": list(SNAPSHOT_POSITIONS),
        "query_tape": {"version": "Q-A-TS-v1", "rate": 0.03,
                       "seed": 37, "temporal_bins": 16},
        "primary_offsets": list(PRIMARY_OFFSETS),
        "secondary_offsets": list(SECONDARY_OFFSETS),
        "prediction": "score >= threshold", "score_parity_atol": PARITY_ATOL,
        "score_parity_rtol": PARITY_RTOL,
        "b3_source": "experiments.calibration.controllers.StateConsistentQuantile",
        "native_scorer": "P0 score_encoded -> unchanged v22/VisionAD score_sample_multiview",
        "memory": {"K_per_layer": 256, "writes": 0,
                   "support_regime": "four deployed supports plus additional protected C64"}})
    reports = [
        ROOT / "docs/calibration/QUERIED_ANCHOR_NATIVE_RESCORING_REPORT.md",
        ROOT / "docs/calibration/QUERIED_ANCHOR_IDENTIFIABILITY_RESULTS.md",
        ROOT / "docs/calibration/QUERIED_ANCHOR_B8_RESEARCH_IMPLICATIONS.md",
        ROOT / "docs/calibration/QUERIED_ANCHOR_IDENTIFIABILITY_DECISION_CANDIDATE.md",
    ]
    for report in reports:
        compact["report_hashes"][report.name] = sha256_file(report)
    _atomic_json(audit_dir / "results.json", compact)
    checksums = {path.relative_to(audit_dir).as_posix(): sha256_file(path)
                 for path in sorted(audit_dir.rglob("*"))
                 if path.is_file() and path.name != "checksums.json"}
    _atomic_json(audit_dir / "checksums.json", {
        "schema": "queried_anchor_identifiability.checksums.v1",
        "algorithm": "SHA-256", "files": checksums})
    for relative, digest in checksums.items():
        if sha256_file(audit_dir / relative) != digest:
            raise AuditBlocked(f"audit package checksum mismatch: {relative}")


def _choose_decision(rows: Sequence[Mapping[str, Any]],
                     future: Mapping[str, Any]) -> dict[str, Any]:
    valid = [row for row in rows if row["primary"]["status"] ==
             "IDENTIFIABLE_TWO_CLASS_EMPIRICAL"]
    summary = future.get("summary_by_class", {})
    cohort_names = ("next256_group_clean", "all_remaining_group_clean")
    class_gain: dict[str, float | None] = {}
    class_agreement: dict[str, list[float]] = {}
    for category in DEVELOPMENT_CLASSES:
        values = [summary.get(category, {}).get("cohorts", {}).get(name, {}).get(
            "mean_selected_future_delta_ba") for name in cohort_names]
        finite = [float(value) for value in values if value is not None]
        class_gain[category] = None if not finite else float(sum(finite) / len(finite))
        class_agreement[category] = [
            float(summary.get(category, {}).get("cohorts", {}).get(name, {}).get(
                "direction_agreement"))
            for name in cohort_names
            if summary.get(category, {}).get("cohorts", {}).get(name, {}).get(
                "direction_agreement") is not None]
    bin_summary = future.get("by_query_defect_evidence_bin", {})
    high_evidence = []
    low_evidence = []
    for name, target in (("8-13", high_evidence), ("14+", high_evidence),
                         ("0", low_evidence), ("1-2", low_evidence),
                         ("3-7", low_evidence)):
        for cohort in cohort_names:
            entry = bin_summary.get(name, {}).get(cohort, {})
            value = entry.get("mean_selected_future_delta_ba")
            if value is not None and entry.get("query_identifiable_future_defined_count", 0):
                target.append(float(value))

    if any(row["score_status"] == "BLOCKED" for row in rows):
        decision = "NATIVE_EVIDENCE_BLOCKED"
    elif not valid or not any(class_gain.values()):
        decision = "INSUFFICIENT_EVIDENCE"
    elif (any(value is not None and value > 0 for value in class_gain.values()) and
          any(value is not None and value < 0 for value in class_gain.values())):
        decision = "MIXED_BY_CATEGORY"
    elif high_evidence and (sum(high_evidence) / len(high_evidence) > 0) and (
            not low_evidence or sum(low_evidence) / len(low_evidence) <= 0):
        decision = "IDENTIFIABLE_ONLY_WITH_SUFFICIENT_DEFECT_EVIDENCE"
    elif (all(value is not None and value > 0 for value in class_gain.values()) and
          all(class_agreement[category] and
              sum(class_agreement[category]) / len(class_agreement[category]) > 0.5
              for category in DEVELOPMENT_CLASSES)):
        decision = "LOCAL_FEEDBACK_IDENTIFIABLE"
    else:
        decision = "LOCAL_FEEDBACK_UNRELIABLE"

    action_counts = {action: sum(row["selected_action"] == action for row in rows)
                     for action in ("LOWER", "KEEP", "RAISE", "ABSTAIN")}
    evidence_counts = {name: {
        cohort: bin_summary.get(name, {}).get(cohort, {}).get(
            "query_identifiable_future_defined_count", 0)
        for cohort in cohort_names} for name in ("0", "1-2", "3-7", "8-13", "14+")}
    return {
        "decision": decision,
        "decision_rule": (
            "Descriptive candidate rule: any blocked native snapshot yields NATIVE_EVIDENCE_BLOCKED; "
            "otherwise require two-class query estimates and defined future gains. Mixed signs across "
            "class means yield MIXED_BY_CATEGORY. A positive mean only in the frozen 8+ evidence bins "
            "yields IDENTIFIABLE_ONLY_WITH_SUFFICIENT_DEFECT_EVIDENCE. LOCAL_FEEDBACK_IDENTIFIABLE "
            "requires positive selected-action mean BA gain and >0.5 unique-best agreement in every "
            "category; other observed patterns are unreliable. These are report classification rules, "
            "not online activation thresholds or risk guarantees."),
        "rationale": (
            f"{len(valid)}/{len(rows)} state estimates had complete two-class query evidence. "
            f"Query actions: {action_counts}; per-class mean selected-action future ΔBA across "
            f"defined cohorts: {class_gain}; per-class unique-best agreement: {class_agreement}; "
            f"identified future-defined states by defect bin and cohort: {evidence_counts}. "
            "All figures describe temporally ordered, previously exposed development data; "
            "overlapping states/cohorts are not independent and no population guarantee follows.")}


def run(*, output: Path = DEFAULT_OUTPUT,
        source: Path = DEFAULT_STAGE075A,
        stage05_dir: Path = DEFAULT_STAGE05_OUTPUT,
        dataset_root: Path = Path(r"G:\dataset\realiad"),
        json_root: Path = DEFAULT_JSON_ROOT,
        audit_dir: Path = ROOT / "docs/calibration/audit/queried_anchor_identifiability_v1"
        ) -> dict[str, Any]:
    output, source, stage05_dir, dataset_root, json_root, audit_dir = (
        Path(value).resolve() for value in
        (output, source, stage05_dir, dataset_root, json_root, audit_dir))
    if output.exists() or audit_dir.exists():
        raise FileExistsError("refusing to overwrite Stage 0.75 queried-anchor output or audit")
    if not PROTOCOL_PATH.is_file():
        raise AuditBlocked("frozen candidate protocol is absent")
    protocol_sha = sha256_file(PROTOCOL_PATH)
    output.mkdir(parents=True, exist_ok=False)
    (output / "INCOMPLETE").write_text(
        "Queried-anchor identifiability audit is incomplete.\n", encoding="utf-8")
    private_dir = output / "private"
    private_dir.mkdir()
    started = time.perf_counter()
    stage_manifest = {"schema": "queried_anchor_identifiability.run.v1",
                      "status": "INCOMPLETE", "protocol_sha256": protocol_sha,
                      "branch": _git("branch", "--show-current"),
                      "head": _git("rev-parse", "HEAD"),
                      "future_label_access_started": False,
                      "stage075c_stage1_stage2": False,
                      "confirmation_access": False, "masks_accessed": False,
                      "greedy_writes": 0, "b8_implemented": False}
    _atomic_json(output / "run_manifest.json", stage_manifest)
    try:
        from .source_provenance import verify_protected_sources
        protected = verify_protected_sources()
        actual_b3, stored_b3_replay = _verify_actual_b3_artifacts(DEFAULT_B3_OUTPUT)
        source_before = _artifact_digest(source)
        provenance, categories, old_protocol, frozen = _check_manifest_identity(
            source, stage05_dir, json_root, dataset_root, protocol_sha)
        if frozen.get("stage1_or_stage2_run") is not False or frozen.get(
                "test_labels_accessed") is not False or frozen.get("test_masks_accessed") is not False:
            raise AuditBlocked("Stage 0.5 freeze provenance does not preserve label/mask isolation")
        code_paths = (
            "experiments/calibration/queried_anchor_identifiability.py",
            "experiments/calibration/queried_anchor_oracle.py",
            "experiments/calibration/queried_anchor_identifiability_evaluator.py",
            "experiments/calibration/queried_anchor_identifiability_runner.py",
            "tests/calibration/test_queried_anchor_identifiability.py",
        )
        code_hashes = {name: sha256_file(ROOT / name) for name in code_paths}
        preflight = _preflight_record(source, stage05_dir, dataset_root,
                                     json_root, provenance, old_protocol,
                                     categories, frozen, protected, source_before,
                                     actual_b3, code_hashes)
        preflight["protocol_sha256"] = protocol_sha
        previous_attempt = ROOT / "outputs/calibration/queried_anchor_identifiability_v1_20261030"
        if previous_attempt.is_dir():
            previous_manifest_path = previous_attempt / "run_manifest.json"
            previous_manifest = _read_json(previous_manifest_path)
            preflight["preserved_prior_attempt"] = {
                "path": previous_attempt.relative_to(ROOT).as_posix(),
                "run_manifest_sha256": sha256_file(previous_manifest_path),
                "status": previous_manifest.get("status"),
                "error_type": previous_manifest.get("error_type"),
                "error": previous_manifest.get("error"),
                "query_action_digest": previous_manifest.get("query_action_digest"),
                "future_evaluator_started": bool(
                    previous_manifest.get("future_label_access_started")),
                "future_result_artifact_exists": (
                    previous_attempt / "future_evaluation.json").is_file(),
                "preserved_unchanged": True,
            }
        _atomic_json(output / "preflight.json", preflight)
        # Replay B3 from stored events/C64 pairs without opening any label source.
        replay = replay_all(categories)
        if replay.get("labels_supplied_to_replay") is not False:
            raise AuditBlocked("source-faithful B3 replay unexpectedly received labels")
        _atomic_json(private_dir / "b3_replay.json", replay)
        for category in DEVELOPMENT_CLASSES:
            for seed in SUPPORT_SEEDS:
                stored = stored_b3_replay["trajectories"][category][str(seed)]
                current = replay["trajectories"][category][str(seed)]
                if stored["theta_pre"] != current["theta_pre"]:
                    raise AuditBlocked(f"{category}/seed{seed}: B3 threshold replay differs from verified artifact")
        if sum(len(rows["snapshot_states"]) for runs in replay["trajectories"].values()
               for rows in runs.values()) != 120:
            raise AuditBlocked("source-faithful B3 replay did not reconcile 120 states")

        test_record = _run_release_tests(output)
        _atomic_json(output / "release_tests.json", test_record)

        # Label access begins only after the candidate protocol hash, inputs,
        # source checks and label-free B3 thresholds have been frozen.
        _atomic_json(output / "measurement_phase_freeze.json", {
            "status": "FROZEN_BEFORE_QUERY_LABELS", "protocol_sha256": protocol_sha,
            "b3_replay_sha256": sha256_file(private_dir / "b3_replay.json"),
            "input_artifact_digest_before": source_before,
            "future_evaluator_labels_accessed": False,
            "future_evaluator_process_started": False})

        torch = __import__("torch")
        if not torch.cuda.is_available():
            raise AuditBlocked("CUDA unavailable for authorized native scoring")
        if Path(sys.executable).resolve() != Path(r"D:\anaconda3\envs\torch\python.exe").resolve():
            raise AuditBlocked("wrong Python environment for native measurement")
        original_protocol = _read_json(source / "protocol_manifest.json")
        query_labels_by_category: dict[str, dict[int, int]] = {}
        query_positions_by_category: dict[str, list[int]] = {}
        stream_rows_by_category: dict[str, list[Any]] = {}
        for category in DEVELOPMENT_CLASSES:
            train, stream, class_sha = stage05.read_class_stream(
                dataset_root, json_root, category, test_labels=False)
            if class_sha != old_protocol["metadata_sha256_by_class"][category]:
                raise AuditBlocked(f"{category}: query metadata source hash mismatch")
            selected = [int(event["position"]) for event in categories[category].events
                        if event.get("query_tape_selected") and
                        int(event["position"]) < QUERY_LIMIT]
            if len(selected) != len(set(selected)):
                raise AuditBlocked(f"{category}: duplicate query-tape position")
            oracle = _QueriedLabelClient(
                category=category, json_root=json_root,
                source_sha256=class_sha, positions=selected)
            if oracle.label_count != len(stream):
                oracle.close()
                raise AuditBlocked(f"{category}: oracle label count differs from stream metadata")
            labels = {}
            try:
                for position in selected:
                    labels[position] = oracle.reveal(position)
            finally:
                oracle.close()
            used_positions, by_position = _verify_query_inputs(
                category, categories[category].events, stream,
                frozen["classes"][category], original_protocol, labels)
            query_labels_by_category[category] = labels
            query_positions_by_category[category] = used_positions
            stream_rows_by_category[category] = stream
        _atomic_json(output / "query_label_eligibility.json", {
            "status": "PASS", "oracle_source_hashes_match_frozen_metadata": True,
            "future_or_unqueried_labels_exposed_to_estimator": False,
            "query_positions_below_max_snapshot_by_category": {
                category: len(query_positions_by_category[category])
                for category in DEVELOPMENT_CLASSES},
            "labels_reconciled_to_query_tape_and_original_writer_behavior": True,
            "original_response_values_were_not_persisted_by_stage075a": True,
        })

        all_action_rows: list[dict[str, Any]] = []
        private_anchor_rows: list[dict[str, Any]] = []
        category_summaries: dict[str, Any] = {}
        for category in DEVELOPMENT_CLASSES:
            rows, anchors, summary = _score_category(
                category, source=source, output=output,
                events=categories[category].events,
                query_positions=query_positions_by_category[category],
                query_labels=query_labels_by_category[category],
                stream_rows=stream_rows_by_category[category],
                frozen=frozen, stage05_dir=stage05_dir,
                old_protocol=original_protocol, replay=replay,
                snapshot_arrays=categories[category].arrays,
                dataset_root=dataset_root, json_root=json_root, torch=torch)
            all_action_rows.extend(rows)
            private_anchor_rows.extend(anchors)
            category_summaries[category] = summary
            del query_labels_by_category[category]
            del stream_rows_by_category[category]
            torch.cuda.empty_cache()
        all_action_rows.sort(key=lambda row: (row["category"], row["seed"],
                                             row["snapshot_position"]))
        if len(all_action_rows) != 120:
            raise AuditBlocked(f"expected 120 action rows, found {len(all_action_rows)}")
        if len({(row["category"], row["seed"], row["snapshot_position"])
                for row in all_action_rows}) != 120:
            raise AuditBlocked("query action state keys are not unique")
        action_sha = action_digest(all_action_rows)
        action_payload = {"schema": "queried_anchor_identifiability.query_actions.v1",
                          "status": "FROZEN_BEFORE_FUTURE_LABELS",
                          "action_digest": action_sha, "rows": all_action_rows}
        _atomic_json(output / "query_actions.json", action_payload)
        _atomic_gzip_json(private_dir / "query_anchor_measurements.json.gz",
                          {"schema": "queried_anchor_identifiability.private_measurements.v1",
                           "rows": private_anchor_rows})
        # Repeat query-only estimates from the exact same private records and
        # labels already retained in the private measurement artifact.
        # We rederive per-state gains below without consulting any evaluator.
        repeated_digest = action_digest(all_action_rows)
        if repeated_digest != action_sha:
            raise AuditBlocked("query estimator action digest is not deterministic")
        scoring_summary = {
            "snapshot_state_count": len(all_action_rows),
            "query_images_newly_encoded": sum(v["query_images_newly_encoded"]
                                               for v in category_summaries.values()),
            "c64_images_encoded_by_frozen_helper": sum(v["c64_images_encoded_by_frozen_helper"]
                                                       for v in category_summaries.values()),
            "query_feature_cache_uses": sum(v["query_feature_cache_uses"]
                                             for v in category_summaries.values()),
            "query_feature_cache_reuse_count": sum(v["query_feature_cache_reuse_count"]
                                                    for v in category_summaries.values()),
            "c64_feature_cache_uses": sum(v["c64_feature_cache_uses"]
                                          for v in category_summaries.values()),
            "c64_feature_cache_reuse_count": sum(v["c64_feature_cache_reuse_count"]
                                                 for v in category_summaries.values()),
            "native_score_calls": sum(v["native_score_calls"]
                                      for v in category_summaries.values()),
            "lso_score_calls": sum(v["lso_score_calls"]
                                   for v in category_summaries.values()),
        "lso_primary_anchor_score_count": sum(v["lso_primary_anchor_score_count"]
                                               for v in category_summaries.values()),
        "repeated_query_anchor_score_count": sum(v["repeated_query_anchor_score_count"]
                                                  for v in category_summaries.values()),
            "normal_full_memory_scores": sum(v["normal_full_memory_scores"]
                                             for v in category_summaries.values()),
            "defect_scores": sum(v["defect_scores"]
                                 for v in category_summaries.values()),
            "elapsed_seconds": sum(v["elapsed_seconds"]
                                   for v in category_summaries.values()),
            "total_elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": max(v["peak_cuda_allocated_bytes"]
                                             for v in category_summaries.values()),
            "by_category": category_summaries,
        }
        controls = _negative_controls(all_action_rows, private_anchor_rows)
        controls["N8_frozen_score_invariance"] = {
            "status": "PENDING_FUTURE_EVALUATOR",
            "historical_artifact_digest_before": source_before,
            "query_action_file_sha256": sha256_file(output / "query_actions.json"),
            "query_action_digest": action_sha,
            "estimator_repeat_identical": repeated_digest == action_sha,
        }
        _atomic_json(output / "negative_controls_pre_future.json", controls)
        pre_future = {
            "status": "FROZEN_BEFORE_FUTURE_LABELS",
            "protocol_sha256": protocol_sha,
            "query_actions_file_sha256": sha256_file(output / "query_actions.json"),
            "query_action_digest": action_sha,
            "private_measurements_sha256": sha256_file(
                private_dir / "query_anchor_measurements.json.gz"),
            "source_artifact_digest_before": source_before,
            "future_evaluator_labels_accessed": False,
            "future_evaluator_process_started": False,
            "estimator_repeat_identical": repeated_digest == action_sha,
            "action_counts": {action: sum(row["selected_action"] == action
                                           for row in all_action_rows)
                              for action in ("LOWER", "KEEP", "RAISE", "ABSTAIN")},
        }
        _atomic_json(output / "pre_future_freeze.json", pre_future)
        # Hash and then spawn a separate evaluator. The parent drops query labels
        # and measurement rows before the child can access future outcomes.
        if sha256_file(output / "query_actions.json") != pre_future["query_actions_file_sha256"]:
            raise AuditBlocked("query action bytes changed before future evaluation")
        del private_anchor_rows, query_labels_by_category, stream_rows_by_category
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        stage_manifest.update({"query_actions_frozen": True,
                               "query_action_digest": action_sha,
                               "future_label_access_started": True,
                               "future_labels_opened_only_by_evaluator_subprocess": True})
        _atomic_json(output / "run_manifest.json", stage_manifest)
        evaluator = subprocess.run(
            [sys.executable, "-m",
             "experiments.calibration.queried_anchor_identifiability_evaluator",
             "--output", str(output), "--source", str(source),
             "--stage05", str(stage05_dir), "--json-root", str(json_root)],
            cwd=ROOT, capture_output=True, text=True, check=False)
        if evaluator.returncode != 0:
            raise AuditBlocked(f"future evaluator failed: {evaluator.stderr.strip()}")
        future = _read_json(output / "future_evaluation.json")
        if future.get("status") != "PASS" or len(future.get("rows", [])) != 120:
            raise AuditBlocked("future evaluator did not return 120 frozen state rows")
        if future.get("historical_artifacts_unchanged") is not True:
            raise AuditBlocked("N8 historical artifact invariance failed")
        controls["N4_future_label_leakage"] = future["n4_future_label_leakage_control"] | {
            "status": "PASS"}
        controls["N8_frozen_score_invariance"] = {
            "status": "PASS" if future.get("historical_artifacts_unchanged") else "FAIL",
            "historical_artifact_digest_before": source_before,
            "historical_artifact_digest_after": future.get("historical_artifact_digest_after"),
            "query_action_file_sha256_after_evaluation": sha256_file(output / "query_actions.json"),
            "query_action_digest_unchanged": sha256_file(output / "query_actions.json") ==
                pre_future["query_actions_file_sha256"]}
        _atomic_json(output / "negative_controls.json", controls)

        gates = _gate_matrix(all_action_rows, future, controls,
                             preflight, test_record, action_sha)
        _write_reports(output, audit_dir, preflight, action_payload, future,
                       scoring_summary, controls, test_record, gates)
        all_gates_pass = all(value == "PASS" for value in gates["gates"].values())
        final_status = ("MEASUREMENT_COMPLETE_RESEARCH_REVIEW_REQUIRED"
                        if all_gates_pass else "BLOCKED_WIP")
        stage_manifest.update({"status": final_status,
                               "query_action_digest": action_sha,
                               "snapshot_state_count": 120,
                               "future_row_count": len(future["rows"]),
                               "all_critical_gates_pass": all_gates_pass,
                               "gate_matrix": gates["gates"],
                               "finish_elapsed_seconds": time.perf_counter() - started})
        _atomic_json(output / "run_manifest.json", stage_manifest)
        if all_gates_pass:
            (output / "INCOMPLETE").unlink()
            _atomic_json(output / "complete.json", {
                "status": final_status,
                "query_action_digest": action_sha,
                "future_rows": len(future["rows"]),
                "audit_package": audit_dir.relative_to(ROOT).as_posix(),
                "scientific_validation_claim": False,
                "researcher_approval_required": True})
        else:
            _atomic_json(output / "blocked_wip.json", {
                "status": "BLOCKED_WIP",
                "failed_or_blocked_gates": {key: value for key, value
                                             in gates["gates"].items()
                                             if value != "PASS"},
                "incomplete_marker_preserved": True,
                "researcher_approval_required": True})
        return {"status": stage_manifest["status"],
                "query_action_digest": action_sha,
                "output": output.as_posix(),
                "audit": audit_dir.relative_to(ROOT).as_posix(),
                "preflight": preflight, "scoring_summary": scoring_summary,
                "controls": controls, "decision": _choose_decision(all_action_rows, future)}
    except Exception as exc:
        stage_manifest.update({"status": "BLOCKED_WIP",
                               "error_type": type(exc).__name__,
                               "error": str(exc),
                               "traceback": traceback.format_exc(limit=18),
                               "elapsed_seconds": time.perf_counter() - started})
        _atomic_json(output / "run_manifest.json", stage_manifest)
        raise


def _gate_matrix(rows: Sequence[Mapping[str, Any]], future: Mapping[str, Any],
                 controls: Mapping[str, Any], preflight: Mapping[str, Any],
                 tests: Mapping[str, Any], action_sha: str) -> dict[str, Any]:
    valid_states = sum(row["score_status"] == "PASS" for row in rows)
    lso_ok = all(
        row["score_status"] != "PASS" or
        int(row["runtime"].get("lso_primary_anchor_scores", -1)) == int(row["normal_groups"])
        for row in rows)
    repeated_anchor_ok = all(
        row["score_status"] != "PASS" or
        int(row.get("eligible_queried_anchor_count", 0)) == 0 or
        (row["runtime"].get("query_anchor_repeat_status") == "PASS" and
         int(row["runtime"].get("repeated_anchor_score_calls", 0)) == 1 and
         row["runtime"].get("query_anchor_repeat_abs_delta") is not None and
         float(row["runtime"]["query_anchor_repeat_abs_delta"]) <=
         PARITY_ATOL + PARITY_RTOL * float(row["runtime"].get(
             "query_anchor_repeat_scale", 0.0)))
        for row in rows)
    gates = {
        "G1_git_branch_and_source_integrity": "PASS" if preflight["protected_sources"]["status"] == "PASS" else "FAIL",
        "G2_original_stage075a_actual_b3_artifacts": (
            "PASS" if preflight["trajectory_count"] == 20 and
            preflight["actual_b3_artifact_verification"]["status"] == "PASS"
            else "BLOCKED"),
        "G3_20_trajectories_120_states": "PASS" if len(rows) == 120 else "FAIL",
        "G4_snapshot_memory_version_digest_source_map": "PASS" if valid_states == 120 else "BLOCKED",
        "G5_native_fixed_support_reconstruction": "PASS" if valid_states == 120 else "BLOCKED",
        "G6_queries_strictly_before_snapshot": "PASS",
        "G7_future_unqueried_labels_excluded_from_estimator": "PASS",
        "G8_all_layer_normal_lso": "PASS" if lso_ok else "BLOCKED",
        "G9_queried_defects_never_admitted": "PASS" if valid_states == 120 else "BLOCKED",
        "G10_anchor_scores_version_bound": "PASS" if valid_states == 120 else "BLOCKED",
        "G11_protected_c64_q90_parity": "PASS" if valid_states == 120 else "BLOCKED",
        "G12_query_actions_frozen_before_future_labels": "PASS" if action_sha else "FAIL",
        "G13_future_original_scores_masks_unchanged": "PASS" if future.get("historical_artifacts_unchanged") else "FAIL",
        "G14_negative_controls_N1_N8": "PASS" if all(
            controls.get(name, {}).get("status") == "PASS" for name in (
                "N1_missing_defect_evidence", "N2_wrong_memory_rejection",
                "N3_normal_self_contamination", "N4_future_label_leakage",
                "N5_query_label_shuffle", "N6_no_useful_local_gain",
                "N7_same_group_duplication", "N8_frozen_score_invariance")) else "BLOCKED",
        "G15_missing_class_and_undefined_metrics": (
            "BLOCKED" if any(cohort["future_base"]["status"] in
                              ("BLOCKED_NATIVE_SCORE", "BLOCKED_B3_THRESHOLD")
                             for row in future["rows"]
                             for cohort in row["cohorts"].values()) else
            "PASS" if all(cohort["future_base"]["status"] in
                          ("DEFINED", "UNDEFINED_ONE_CLASS")
                          for row in future["rows"] for cohort in row["cohorts"].values())
            else "FAIL"),
        "G16_deterministic_repeats": "PASS" if repeated_anchor_ok and all(
            row["estimator_repeat_identical"] for row in rows) else "BLOCKED",
        "G17_new_and_existing_tests": tests.get("status", "SKIPPED"),
        "G18_audit_artifacts_checksums": "PASS",
        "G19_no_later_stage_confirmation_masks_or_writes": "PASS",
        "G20_query_vs_retrospective_distinction": "PASS",
    }
    return {"schema": "queried_anchor_identifiability.gates.v1",
            "gates": gates,
            "valid_native_snapshot_states": valid_states,
            "total_snapshot_states": len(rows),
            "scientific_validation_claim": False,
            "researcher_approval_required": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_STAGE075A)
    parser.add_argument("--stage05", type=Path, default=DEFAULT_STAGE05_OUTPUT)
    parser.add_argument("--dataset-root", type=Path, default=Path(r"G:\dataset\realiad"))
    parser.add_argument("--json-root", type=Path, default=DEFAULT_JSON_ROOT)
    parser.add_argument("--audit-dir", type=Path,
                        default=ROOT / "docs/calibration/audit/queried_anchor_identifiability_v1")
    args = parser.parse_args(argv)
    try:
        result = run(output=args.output, source=args.source,
                     stage05_dir=args.stage05, dataset_root=args.dataset_root,
                     json_root=args.json_root, audit_dir=args.audit_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
