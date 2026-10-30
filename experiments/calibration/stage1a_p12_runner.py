"""Bounded Stage 1-A P1.2 replay using the standing researcher instruction.

The runner reuses the frozen Q-A-TS tapes, Stage 0.5 score tapes, and historical
Stage 0.75-A memory/C64 records. It opens query images lazily and obtains a
single test label only when the fixed tape requests that position. This module
does not run Stage 1-A controller comparisons or performance metrics.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping

import numpy as np

from . import stage05
from .native import NativeScoreService, SharedNativeGreedyWriter, isolate_native_bank_k256
from .profiles import DEFAULT_PROFILE, canonical_json
from .stage1a_p1_reconstruction import (
    C64Identity, FeatureKey, MemoryIdentity, P1ParityMismatch, QueryAnchor,
    ReplayEvent, VersionedAnchorEvidenceCache, sha256_json,
)
from .stage1a_p11_production import (
    CausalEvidenceRevisionAdapter, NativeAllLayerLSOAdapter,
    P11_CLASSES, P11IntegrityError, P11PrivateCheckpoint,
    P11ProductionReplayRunner, StandingP12Authorization,
    apply_historical_shared_writer, p11_source_identity,
    verify_historical_c64_artifact_links, verify_stage05_qats_artifacts,
)
from .stage1a_p12_label_provider import QueriedPositionLabelReader


ROOT = Path(__file__).resolve().parents[2]
STAGE05 = ROOT / "outputs/calibration/stage05_exploratory_20261029"
STAGE075A = ROOT / "outputs/calibration/stage075a_greedy_ranking_repaired3_20261029"
DEFAULT_DATASET_ROOT = Path(r"G:\dataset\realiad")
DEFAULT_JSON_ROOT = DEFAULT_DATASET_ROOT / "jsons" / "realiad_jsons"
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
SCORER_CONFIG_SHA256 = "1b8e9d96b121e3f1f6c1bbee13ab9f0063facee8fb08ad696dc5f9cb43dbd452"


class P12RunnerError(RuntimeError):
    """A frozen artifact, dependency, parity, or bounded-run failure."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_json(value: Any) -> str:
    return _sha_bytes(canonical_json(value).encode("utf-8"))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                          dir=str(path.parent))
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, (canonical_json(value) + "\n").encode("utf-8"))


def _new_versioned_path(parent: Path, stem: str) -> Path:
    for index in range(1, 100):
        path = parent / f"{stem}_standing_authorization_{index:02d}_20261030"
        if not path.exists():
            return path
    raise P12RunnerError(f"No free immutable namespace remains for {stem}")


def _verify_historical_package() -> dict[str, Any]:
    """Verify only frozen protocol and per-trajectory replay artifacts.

    Offline evaluation outputs are not opened or interpreted.
    """
    root_manifest = _read_json(STAGE075A / "checksums.json")
    root_entries = root_manifest.get("files", {})
    required_root = ("run_manifest.json", "protocol_manifest.json", "audit_linkage.json")
    root_hashes = {}
    for name in required_root:
        path = STAGE075A / name
        expected = root_entries.get(name)
        if not path.is_file() or not isinstance(expected, str) or _sha_file(path) != expected:
            raise P12RunnerError(f"Historical Stage 0.75-A root artifact checksum failed: {name}")
        root_hashes[name] = expected
    protocol = _read_json(STAGE075A / "protocol_manifest.json")
    run = _read_json(STAGE075A / "run_manifest.json")
    if (run.get("phase_status", {}).get("run") != "PASS" or
            run.get("online_trajectory_count") != 20 or
            run.get("stage1_or_stage2_run") is not False or
            protocol.get("support_seeds") != [0, 1, 2, 3, 4] or
            protocol.get("query_protocol") != {
                "version": "Q-A-TS-v1", "rate": 0.03, "seed": 37,
                "temporal_bins": 16, "one_query_per_official_group": True}):
        raise P12RunnerError("Historical Stage 0.75-A run/protocol identity is outside P1.2")
    if protocol.get("scorer_config_sha256") != SCORER_CONFIG_SHA256:
        raise P12RunnerError("Historical Native Scorer config digest differs from P1.2 lock")
    file_hashes: dict[str, str] = {}
    for category in P11_CLASSES:
        folder = STAGE075A / "trajectories" / category
        check_path = folder / "checksums.json"
        if not check_path.is_file():
            raise P12RunnerError(f"Missing historical trajectory checksum file: {category}")
        entries = _read_json(check_path).get("files", {})
        required = ("trajectory_manifest.json", "scores.npz",
                    "event_trace.jsonl.gz", "c64_transport.json")
        for name in required:
            path = folder / name
            expected = entries.get(name)
            if not path.is_file() or not isinstance(expected, str) or _sha_file(path) != expected:
                raise P12RunnerError(f"Historical {category} artifact checksum failed: {name}")
            file_hashes[f"{category}/{name}"] = expected
        manifest = _read_json(folder / "trajectory_manifest.json")
        if (manifest.get("category") != category or manifest.get("smoke_only") is not False or
                manifest.get("positions_scored") != stage05.EXPECTED_STREAM_LENGTHS[category] or
                manifest.get("stage1_or_stage2_run") is not False or
                manifest.get("event_trace_sha256") != entries["event_trace.jsonl.gz"] or
                manifest.get("score_artifact_sha256") != entries["scores.npz"]):
            raise P12RunnerError(f"Historical {category} trajectory manifest is not a full frozen run")
    for name, expected in protocol.get("source_sha256", {}).items():
        path = ROOT / name
        if not path.is_file() or _sha_file(path) != expected:
            raise P12RunnerError(f"Protected historical replay source changed: {name}")
    return {
        "run_manifest_sha256": root_hashes["run_manifest.json"],
        "protocol_manifest_sha256": root_hashes["protocol_manifest.json"],
        "audit_linkage_sha256": root_hashes["audit_linkage.json"],
        "trajectory_file_sha256": file_hashes,
        "historical_source_sha256": dict(protocol["source_sha256"]),
        "historical_native_source_sha256": dict(protocol["native_source_sha256"]),
        "offline_evaluation_opened": False,
    }


def _current_source_identity() -> tuple[dict[str, Any], str, str]:
    """Return complete P1.2 source provenance and Native scorer source key."""
    from .stage1a_p1_reconstruction import source_fingerprints
    current_p11 = p11_source_identity(ROOT)
    p12 = source_fingerprints(ROOT, [
        "experiments/calibration/stage1a_p12_runner.py",
        "experiments/calibration/stage1a_p12_label_provider.py",
    ])
    aggregate = _sha_json({"p11_source_identity": current_p11,
                           "p12_source_identity": p12})
    native_paths = (
        "experiments/calibration/native.py",
        "experiments/calibration/memory_state.py",
        "experiments/calibration/profiles.py",
        "experiments/p0_protocol_audit.py",
        "experiments/risk_safe/native_bridge.py",
        "experiments/risk_safe/shadow_memory.py",
        "experiments/v02_memory.py",
        "test_phase3_AIF_simplified_v22.py",
        "vmb_visionad_new.py",
    )
    raw = {name: _sha_file(ROOT / name) for name in native_paths}
    return {"p11": current_p11, "p12": p12, "native_source_files": raw}, aggregate, _sha_json(raw)


P12_EXECUTION_SOURCE_PATHS = (
    "experiments/calibration/stage1a_p1_reconstruction.py",
    "experiments/calibration/stage1a_p11_production.py",
    "experiments/calibration/stage1a_p12_label_provider.py",
    "experiments/calibration/stage1a_p12_runner.py",
)


def _capture_execution_source_sha256() -> dict[str, str]:
    return {name: _sha_file(ROOT / name) for name in P12_EXECUTION_SOURCE_PATHS}


def _source_binding_report(protocol_sources: Mapping[str, str],
                           score_sources: Mapping[str, str],
                           root: Path = ROOT) -> dict[str, Any]:
    """Compare frozen Stage 0.75-A, Stage 0.5, and current source identities."""
    entries: dict[str, Any] = {}
    all_match = True
    for name in sorted(set(protocol_sources) | set(score_sources)):
        path = root / name
        current = _sha_file(path) if path.is_file() else None
        protocol_sha = protocol_sources.get(name)
        score_sha = score_sources.get(name)
        protocol_matches = protocol_sha is not None and protocol_sha == current
        score_matches = score_sha is not None and score_sha == current
        cross_manifest_matches = (protocol_sha is not None and score_sha is not None and
                                  protocol_sha == score_sha)
        if not (protocol_matches and score_matches and cross_manifest_matches):
            all_match = False
        entries[name] = {
            "stage075a_protocol_sha256": protocol_sha,
            "stage05_score_manifest_sha256": score_sha,
            "current_source_sha256": current,
            "protocol_matches_current": protocol_matches,
            "score_manifest_matches_current": score_matches,
            "protocol_matches_score_manifest": cross_manifest_matches,
        }
    return {"status": "PASS_EXACT_SOURCE_LINKAGE" if all_match else "BLOCKED_SOURCE_LINKAGE_MISMATCH",
            "all_sources_match": all_match, "files": entries}


# Exact, immutable exception for TWO missing pre-Git Stage 0.5 source snapshots.
# This is not a claim that the historical executed bytes were recovered.
# A third mismatch or a change to either pinned current file must fail closed.
P12_MISSING_HISTORICAL_STAGE05_SOURCE = {
    "experiments/calibration/stage05.py": {
        "frozen_sha256": "af1ef435addfe2f08cb2e40a1d141b617fb15c852ed469c8aa7062663ccf8626",
        "current_sha256": "4b8aae252f5f1e096b127906794064da269aa79f7cd5514f6f738cda585c017e",
    },
    "experiments/calibration/stage05_runner.py": {
        "frozen_sha256": "1e40d9c5e48c252eef3d96686d2a315c4a87f1f325130a6c2961f7cc41cafaea",
        "current_sha256": "1d110e1ca813a064767ebeab597b609b3d863f8afd4c8085b06948ad88347e14",
    },
}


def _assess_scoped_source_reconstruction(source_linkage: Mapping[str, Any]) -> dict[str, Any]:
    """Separately assess executable reconstruction, NEVER historical byte parity.

    Only the exact two pre-Git Stage 0.5 file identities are excepted.
    All other files still require the full three-way source hash match.
    """
    entries = source_linkage.get("files", {})
    if not isinstance(entries, Mapping) or not entries:
        return {"execution_allowed": False, "status": "BLOCKED_INVALID_SOURCE_INVENTORY",
                "historical_source_bytes_recovered": False, "exception_files": []}
    exceptions: list[dict[str, str]] = []
    failed: list[str] = sorted(set(P12_MISSING_HISTORICAL_STAGE05_SOURCE) - set(entries))
    for name, row in sorted(entries.items()):
        if not isinstance(row, Mapping):
            failed.append(name)
            continue
        old = row.get("stage075a_protocol_sha256")
        manifest = row.get("stage05_score_manifest_sha256")
        current = row.get("current_source_sha256")
        if (isinstance(old, str) and old == manifest == current and
                bool(re.fullmatch(r"[0-9a-f]{64}", old))):
            continue
        pinned = P12_MISSING_HISTORICAL_STAGE05_SOURCE.get(name)
        if (pinned is None or old != pinned["frozen_sha256"] or
                manifest != pinned["frozen_sha256"] or
                current != pinned["current_sha256"]):
            failed.append(name)
            continue
        exceptions.append({"path": name, "historical_sha256": old,
                           "new_execution_sha256": current,
                           "historical_byte_identity": "UNRESOLVED"})
    all_exact = not exceptions and not failed
    allowed = not failed
    return {
        "execution_allowed": allowed,
        "status": ("PASS_EXACT_SOURCE_LINKAGE" if all_exact else
                   "PASS_SCOPED_CURRENT_SOURCE_RECONSTRUCTION" if allowed else
                   "BLOCKED_UNAPPROVED_SOURCE_MISMATCH"),
        "historical_source_bytes_recovered": all_exact,
        "original_historical_linkage_status": source_linkage.get("status"),
        "exception_files": exceptions,
        "unapproved_mismatch_files": failed,
        "required_validation": (
            "Existing exact event M0 score, writer, pre/post memory SHA, "
            "all-layer LSO, and C64 transport parity remain mandatory."),
    }


def _locked_native_config_identity() -> dict[str, Any]:
    """Bind the historical scientific-profile digest and its scorer sub-config."""
    if DEFAULT_PROFILE.scientific_sha256 != SCORER_CONFIG_SHA256:
        raise P12RunnerError("Current frozen scientific Native Scorer identity changed")
    return {
        "scorer_config_sha256": SCORER_CONFIG_SHA256,
        "scientific_profile_sha256": DEFAULT_PROFILE.scientific_sha256,
        "native_scorer_dataclass_sha256": DEFAULT_PROFILE.scorer.sha256,
        "scorer_config": asdict(DEFAULT_PROFILE.scorer),
    }


def _search_reachable_source_history(source_linkage: Mapping[str, Any]) -> dict[str, Any]:
    """Check whether frozen source bytes exist in any currently reachable commit."""
    results: dict[str, Any] = {}
    for name, row in source_linkage.get("files", {}).items():
        expected = row.get("stage075a_protocol_sha256")
        if not expected or row.get("current_source_sha256") == expected:
            continue
        listing = subprocess.run(["git", "rev-list", "--all", "--", name], cwd=ROOT,
                                 capture_output=True, text=True, check=False)
        commits = listing.stdout.splitlines() if listing.returncode == 0 else []
        matches = []
        for commit in commits:
            shown = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=ROOT,
                                   capture_output=True, check=False)
            if shown.returncode == 0 and _sha_bytes(shown.stdout) == expected:
                matches.append(commit)
        results[name] = {
            "expected_sha256": expected,
            "reachable_commits_examined": len(commits),
            "matching_commit_ids": matches,
            "expected_bytes_found_in_reachable_history": bool(matches),
        }
    return results


def _verify_native_source_identity(protocol: Mapping[str, Any], stage05_scores: Mapping[str, Any]) -> dict[str, Any]:
    frozen_native = protocol.get("native_source_sha256", {})
    score_native = stage05_scores.get("scoring_source_hashes", {})
    source_linkage = _source_binding_report(frozen_native, score_native)
    scoped_reconstruction = _assess_scoped_source_reconstruction(source_linkage)
    if not scoped_reconstruction["execution_allowed"]:
        raise P12RunnerError(
            "Unexpected Native source change outside the two exact pinned "
            "historical Stage 0.5 exceptions: "
            + canonical_json({"historical_linkage": source_linkage,
                              "execution_assessment": scoped_reconstruction}))
    current, _aggregate, scorer_sha = _current_source_identity()
    locked_config = _locked_native_config_identity()
    return {"source_fingerprints": current,
            **locked_config,
            "scorer_source_sha256": scorer_sha,
            "frozen_native_source_sha256": dict(frozen_native),
            "stage05_native_source_sha256": dict(score_native),
            "source_binding": source_linkage,
            "current_execution_source_assessment": scoped_reconstruction,
            "historical_source_byte_parity": source_linkage["all_sources_match"],
            "historical_source_exception_used": bool(scoped_reconstruction["exception_files"])}


def _write_blocked_audit(output: Path, audit: Path, error: Exception,
                         elapsed_seconds: float | None = None,
                         execution_source_sha256: Mapping[str, str] | None = None) -> bool:
    """Write a compact, immutable preflight-stop package without touching old runs."""
    if output.exists() or audit.exists():
        return False
    if not output.resolve().is_relative_to(ROOT) or not audit.resolve().is_relative_to(ROOT):
        return False

    protocol: Mapping[str, Any] = {}
    score_manifest: Mapping[str, Any] = {}
    source_linkage: Mapping[str, Any] = {}
    tape_binding: str | None = None
    tape_status = "NOT_VERIFIED"
    c64_status = "NOT_VERIFIED"
    tape_summary: Mapping[str, Any] = {}
    c64_linkage: Mapping[str, Any] = {}
    try:
        protocol = _read_json(STAGE075A / "protocol_manifest.json")
        score_manifest = _read_json(STAGE05 / "score_tape_manifest.json")
        source_linkage = _source_binding_report(
            protocol.get("native_source_sha256", {}),
            score_manifest.get("scoring_source_hashes", {}))
    except Exception as diagnostic_error:
        source_linkage = {"status": "DIAGNOSTIC_FAILED", "error": str(diagnostic_error)}
    try:
        tapes = verify_stage05_qats_artifacts(STAGE05)
        tape_binding = tapes.binding_sha256
        tape_status = ("PASS_QA_TS_V1_3PCT_SEED37" if tape_binding ==
                       "dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183"
                       else "FAIL_BINDING_MISMATCH")
        tape_summary = {
            "aggregate_binding_sha256": tape_binding,
            "bindings": tapes.authorization_bindings(),
            "source_artifact_sha256": dict(tapes.source_artifact_sha256),
        }
    except Exception as diagnostic_error:
        tape_status = f"BLOCKED:{type(diagnostic_error).__name__}:{diagnostic_error}"
    try:
        c64_linkage = verify_historical_c64_artifact_links(STAGE05, STAGE075A)
        c64_status = c64_linkage.get("status", "VERIFIED_WITHOUT_STATUS")
    except Exception as diagnostic_error:
        c64_status = f"NOT_VERIFIED:{type(diagnostic_error).__name__}:{diagnostic_error}"

    decision = "P12_FAILED_PARITY" if isinstance(error, P1ParityMismatch) else "P12_BLOCKED"
    try:
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        branch, head = None, None
    output.mkdir(parents=True, exist_ok=False)
    audit.mkdir(parents=True, exist_ok=False)
    (output / "INCOMPLETE").write_text(
        f"{decision}: stopped before real-data scoring. {type(error).__name__}: {error}\n",
        encoding="utf-8")
    failure = {
        "schema": "stage1a.p12_preflight_stop.v1",
        "decision": decision,
        "branch": branch,
        "head": head,
        "failure_type": type(error).__name__,
        "failure": str(error),
        "p12_execution_source_sha256": dict(execution_source_sha256 or {}),
        "real_image_access": False,
        "queried_label_access": False,
        "gpu_inference": False,
        "native_scoring_calls": 0,
        "event_positions_checked": 0,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(output / "P12_STOP.json", failure)
    _atomic_json(audit / "exact_native_memory_parity.json", {
        "status": "BLOCKED_BEFORE_FIRST_EVENT" if decision == "P12_BLOCKED" else "FAILED",
        "events_checked": 0, "trajectories_checked": 0,
        "source_binding": source_linkage,
        "failure": str(error),
    })
    _atomic_json(audit / "dense_anchor_evidence_manifest.json", {
        "status": "BLOCKED_NOT_GENERATED", "evidence_snapshots": 0,
        "native_anchor_score_calls": 0, "protected_c64_native_rescoring": 0,
        "reason": "Exact historical source linkage did not pass before execution."})
    _atomic_json(audit / "queried_feedback_provenance.json", {
        "authorization_basis": "explicit_user_instruction_active_task",
        "signed_or_timestamped_approval_record_created": False,
        "query_tape_binding_sha256": tape_binding,
        "verified_query_tape_identities": tape_summary,
        "query_tape_status": tape_status,
        "queried_label_reads": 0, "unqueried_labels_decoded": False,
        "real_image_bytes_opened": False, "confirmation_classes_accessed": False,
        "masks_accessed": False,
    })
    _atomic_json(audit / "native_scorer_identity.json", {
        "status": "BLOCKED_SOURCE_PROVENANCE",
        **_locked_native_config_identity(),
        "historical_native_source_linkage": source_linkage,
        "historical_c64_artifact_linkage": c64_linkage,
        "native_scorer_initialized": False,
    })
    _atomic_json(audit / "compute_accounting.json", {
        "status": "STOPPED_IN_METADATA_SOURCE_PREFLIGHT",
        "queried_label_reads": 0, "query_feature_encodings": 0,
        "static_map_native_calls": 0, "query_anchor_native_calls": 0,
        "event_positions_exactly_checked": 0, "memory_updates_checked": 0,
        "gpu_inference": False, "peak_gpu_allocated_bytes": 0,
        "peak_cpu_process_bytes": _process_peak_bytes(),
        "elapsed_seconds_to_stop": elapsed_seconds,
        "elapsed_measurement": "CLI wall clock from immediately before P1.2 run to preflight stop",
        "ceilings": {"query_feature_encodings": 459, "queried_label_reads": 459,
                     "static_map_native_calls": 2295, "query_anchor_native_calls": 75406},
    })
    _atomic_json(audit / "checkpoint_resume_verification.json", {
        "status": "NOT_RUN_BLOCKED_BEFORE_EXECUTION",
        "private_checkpoint_created": False,
        "reason": "Source identity preflight failed before trajectory initialization."})
    _atomic_json(audit / "source_provenance.json", {
        "schema": "stage1a.p12_source_provenance.v1",
        "preflight_implementation_source_sha256": {
            name: _sha_file(ROOT / name) for name in (
                "experiments/calibration/stage1a_p1_reconstruction.py",
                "experiments/calibration/stage1a_p11_production.py",
                "experiments/calibration/stage1a_p12_label_provider.py",
                "experiments/calibration/stage1a_p12_runner.py",
            )},
        "executed_preflight_source_sha256": dict(execution_source_sha256 or {}),
        "execution_source_hash_capture": "captured at P1.2 main entry before metadata preflight",
        "stage075a_native_source_sha256": protocol.get("native_source_sha256", {}),
        "stage05_score_manifest_source_sha256": score_manifest.get("scoring_source_hashes", {}),
        "source_binding_comparison": source_linkage,
        "reachable_history_search": _search_reachable_source_history(source_linkage),
        "historical_g2_status": "BLOCKED_UNCHANGED",
        "executed_source_bytes_recovered": False,
    })
    gates = [
        {"id": "P12-STANDING-AUTH", "status": "PASS_USER_STANDING_AUTHORIZATION",
         "evidence": "No separate approval record was requested or fabricated."},
        {"id": "P12-QUERY-TAPE", "status": tape_status, "evidence": tape_binding},
        {"id": "P12-SOURCE-PROVENANCE", "status": source_linkage.get("status", "BLOCKED"),
         "evidence": source_linkage},
        {"id": "P12-HISTORICAL-C64-LINKAGE", "status": c64_status,
         "evidence": "Historical artifact linkage only; no independent Native C64 scoring."},
        {"id": "P12-PHASE-A", "status": "BLOCKED_NOT_RUN", "evidence": str(error)},
        {"id": "P12-EXACT-NATIVE-MEMORY-PARITY", "status": "BLOCKED_NOT_RUN"},
        {"id": "P12-DENSE-ANCHOR-EVIDENCE", "status": "BLOCKED_NOT_RUN"},
        {"id": "P12-CHECKPOINT-RESUME", "status": "BLOCKED_NOT_RUN"},
        {"id": "P12-COMPUTE-CEILINGS", "status": "PASS_ZERO_USAGE_BELOW_CEILINGS"},
        {"id": "P12-STAGE1A-PERFORMANCE", "status": "NOT_RUN_PROHIBITED"},
        {"id": "P12-CONFIRMATION-ISOLATION", "status": "PASS_NO_ACCESS"},
        {"id": "P12-HISTORICAL-G2", "status": "BLOCKED_UNCHANGED"},
    ]
    _atomic_json(audit / "p12_gate_matrix.json", {
        "schema": "stage1a.p12_gate_matrix.v2", "decision": decision,
        "branch": branch, "head": head, "query_tape_binding_sha256": tape_binding,
        "verified_query_tapes": tape_summary,
        "gates": gates,
    })
    source_mismatch_rows = []
    for name, row in source_linkage.get("files", {}).items():
        if not (row.get("protocol_matches_current") and
                row.get("score_manifest_matches_current") and
                row.get("protocol_matches_score_manifest")):
            source_mismatch_rows.append(
                f"| `{name}` | `{row.get('stage075a_protocol_sha256')}` | "
                f"`{row.get('stage05_score_manifest_sha256')}` | "
                f"`{row.get('current_source_sha256')}` | "
                f"{row.get('protocol_matches_score_manifest')} |")
    report = f"""# Stage 1-A P1.2 bounded Native replay and dense evidence

**Decision: `{decision}`**

**Branch:** `{branch}`<br>
**HEAD:** `{head}`<br>
**Q-A-TS binding:** `{tape_binding}`

## Stop reason

The run stopped in metadata/source preflight before opening Real-IAD image bytes,
decoding any label, initializing the DINOv2 model, or issuing a Native scoring
call. The Stage 0.75-A protocol and Stage 0.5 score manifest agree with each
other on the older source hashes, but two current source files differ. The exact
three-way comparison is recorded below and in `source_provenance.json`.
Expected older source bytes were not found in reachable Git history. This is a
provenance/parity failure, not an approval-record issue; the check was not
bypassed. The P1/P1.1/P1.2 runtime source files are SHA-256 bound at runner
entry in `source_provenance.json`; this identifies the blocked P1.2 execution
code but does not recover the older historical source bytes.

| Source | Stage 0.75-A protocol | Stage 0.5 manifest | Current source | Manifests agree |
|---|---|---|---|---|
{chr(10).join(source_mismatch_rows)}

The user's standing authorization was accepted as the authorization basis;
no approval records were created. The verified query-tape aggregate is recorded
above. Actual queried-label reads, image feature encodings, static-map calls,
anchor-score calls and replay events are all zero.

## Gate status and scope

Phase A exact parity, all 20 trajectory parity, dense causal anchor evidence,
and checkpoint/resume are **BLOCKED_NOT_RUN**. Historical C64 status, if
verified, is artifact linkage only and not independent numerical rescoring.
Historical G2 remains BLOCKED. No Stage 1-A performance comparison, metric
calculation, Mint/PCB access, or mask access was performed.

The final decision is `{decision}`. Proceeding requires recovery or researcher
review of byte-identical source provenance; no historical audit evidence was
modified.
"""
    (audit / "STAGE1A_P12_NATIVE_REPLAY_REPORT.md").write_text(report, encoding="utf-8")
    checksums = {path.name: _sha_file(path) for path in sorted(audit.iterdir())
                 if path.is_file() and path.name != "checksums.json"}
    _atomic_json(audit / "checksums.json", {"algorithm": "SHA-256", "files": checksums})
    output_checksums = {path.name: _sha_file(path) for path in sorted(output.iterdir())
                        if path.is_file() and path.name != "checksums.json"}
    _atomic_json(output / "checksums.json", {
        "schema": "stage1a.p12_preflight_stop_checksums.v1",
        "algorithm": "SHA-256", "files": output_checksums,
    })
    return True


def _cpu_map(value: Any) -> Any:
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").contiguous()
    if isinstance(value, dict):
        return {key: _cpu_map(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_map(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_map(child) for child in value)
    return value


def _native_map(value: Any, device: str) -> Any:
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device=device).contiguous()
    if isinstance(value, dict):
        return {key: _native_map(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [_native_map(child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(_native_map(child, device) for child in value)
    return value


def _writer_digest(writer: SharedNativeGreedyWriter) -> str:
    state = writer.state_dict()
    return _sha_json({
        "schema": state["schema"],
        "fixed_support_sha256": state["fixed_support_sha256"],
        "theta_ref": state["theta_ref"], "sigma_ref": state["sigma_ref"],
        "protected_source_keys": state["protected_source_keys"],
        "capacities": state["capacities"],
        "memory_content_version": state["memory_content_version"],
        "legacy_greedy_version": state["legacy_greedy_version"],
        "source_revision": state["source_revision"],
        "source_map_sha256": state["source_map_sha256"],
        "memory_sha256": state["sha256"],
        "content_sha256": state["content_sha256"],
        "legacy_greedy_sha256": state["legacy_greedy_sha256"],
    })


def _load_stage05_score(category: str, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    manifest = _read_json(STAGE05 / "score_tape_manifest.json")
    entry = manifest["classes"][category]["score_tapes"][str(seed)]
    path = STAGE05 / entry["file"]
    if _sha_file(path) != entry["sha256"]:
        raise P12RunnerError(f"Stage 0.5 score tape checksum differs: {category}/seed{seed}")
    with np.load(path, allow_pickle=False) as arrays:
        static = np.asarray(arrays["test_scores"], dtype=np.float64).copy()
        c64 = np.asarray(arrays["c64_scores"], dtype=np.float64).copy()
    if (static.shape != (stage05.EXPECTED_STREAM_LENGTHS[category],) or
            c64.shape != (64,) or not np.isfinite(static).all() or
            not np.isfinite(c64).all()):
        raise P12RunnerError(f"Invalid frozen score tape shape/content: {category}/seed{seed}")
    return static, c64, entry["fixed_support_sha256"]


def _trajectory_records(category: str, seed: int, frozen: Mapping[str, Any],
                        arrays_path: Path, transport_path: Path,
                        static_tape: np.ndarray, initial_c64: np.ndarray
                        ) -> tuple[list[dict[str, Any]], dict[int, np.ndarray], dict[str, Any]]:
    """Build label-free event/state expectations from immutable historical files."""
    from .stage075a import higher_quantile
    folder = STAGE075A / "trajectories" / category
    info = frozen["classes"][category]
    stream = info["stream"]
    with np.load(arrays_path, allow_pickle=False) as arrays:
        dynamic_tape = np.asarray(arrays[f"dynamic_s{seed}"], dtype=np.float64).copy()
        historical_static = np.asarray(arrays[f"static_s{seed}"], dtype=np.float64).copy()
        c64_after = np.asarray(arrays[f"c64_pairs_after_s{seed}"], dtype=np.float64).copy()
    if (len(stream) != len(static_tape) or len(dynamic_tape) != len(stream) or
            len(historical_static) != len(stream) or
            not np.array_equal(historical_static, static_tape)):
        raise P12IntegrityError(f"{category}/seed{seed}: frozen score tape length/value mismatch")
    transport_all = _read_json(transport_path)
    transport = transport_all.get(str(seed))
    if not isinstance(transport, list) or len(transport) != len(c64_after):
        raise P12IntegrityError(f"{category}/seed{seed}: C64 transport vector count mismatch")

    with gzip.open(folder / "event_trace.jsonl.gz", "rt", encoding="utf-8") as source:
        trace = [json.loads(line) for line in source if line.strip()]
    if len(trace) != len(stream):
        raise P12IntegrityError(f"{category}/seed{seed}: historical event trace length mismatch")

    layer_count = len(trace[0]["support_seeds"][str(seed)]["writer"]["capacity_by_layer"])
    initial_sources = [[] for _ in range(layer_count)]
    source_map_sha = _sha_bytes(json.dumps(
        initial_sources, ensure_ascii=True, separators=(",", ":")).encode("ascii"))
    current_c64 = initial_c64.copy()
    current_memory_sha = str(trace[0]["support_seeds"][str(seed)]["memory_sha256_before"])
    current_version = int(trace[0]["support_seeds"][str(seed)]["memory_version_before"])
    current_source_revision = 0
    current_legacy_version = 0
    c64_by_memory_sha: dict[str, np.ndarray] = {current_memory_sha: current_c64.copy()}
    c64_q90_by_memory_sha: dict[str, float] = {
        current_memory_sha: float(higher_quantile(current_c64.tolist(), .90))}
    c64_hash_by_memory_sha: dict[str, str] = {
        current_memory_sha: _sha_json(current_c64.tolist())}
    expected_events: list[dict[str, Any]] = []
    update_index = 0
    first_update_position = None

    for position, (meta, row) in enumerate(zip(stream, trace)):
        seed_row = row.get("support_seeds", {}).get(str(seed))
        if not isinstance(seed_row, Mapping):
            raise P12IntegrityError(f"{category}/seed{seed}/t{position}: trace seed row missing")
        wr = seed_row.get("writer")
        if not isinstance(wr, Mapping):
            raise P12IntegrityError(f"{category}/seed{seed}/t{position}: writer record missing")
        expected_image = _sha_bytes(("image-identity\0" + str(meta["relative_id"])).encode())
        if (int(meta["position"]) != position or int(row.get("position", -1)) != position or
                int(meta["camera_id"]) != int(row.get("camera_id", -2)) or
                expected_image != row.get("image_identity_sha256") or
                str(meta["opaque_group"]) != row.get("official_group_sha256") or
                not math_isfinite(float(seed_row["static_score"])) or
                float(seed_row["static_score"]) != float(static_tape[position]) or
                float(seed_row["dynamic_score"]) != float(dynamic_tape[position])):
            raise P12IntegrityError(f"{category}/seed{seed}/t{position}: historical metadata/score link mismatch")
        queried = bool(row.get("query_tape_selected"))
        if bool(row.get("queried_label_disclosed")) != queried or (
                row.get("oracle_label_request_position") != (position if queried else None)):
            raise P12IntegrityError(f"{category}/t{position}: historical query timing metadata mismatch")
        if (int(wr["memory_content_version_before"]) != current_version or
                str(seed_row["memory_sha256_before"]) != current_memory_sha or
                int(wr["source_revision_before"]) != current_source_revision or
                int(wr["legacy_greedy_version_before"]) != current_legacy_version):
            raise P12IntegrityError(f"{category}/seed{seed}/t{position}: historical state chain is not contiguous")

        memory_before = MemoryIdentity(
            current_version, current_memory_sha, source_map_sha,
            current_source_revision, current_legacy_version)
        expected_after_sha = str(seed_row["memory_sha256_after"])
        expected_after_version = int(seed_row["memory_version_after"])
        after_source_map_sha = str(wr["source_map_sha256"])
        after_source_revision = int(wr["source_revision_after"])
        after_legacy_version = int(wr["legacy_greedy_version_after"])
        if (int(wr["memory_content_version_after"]) != expected_after_version or
                expected_after_version - current_version != int(bool(wr["changed"])) or
                after_source_revision - current_source_revision !=
                int(bool(wr["source_map_changed"]))):
            raise P12IntegrityError(f"{category}/seed{seed}/t{position}: historical version transition fields conflict")
        if float(seed_row["current_C64_q90_pre_event"]) != float(
                higher_quantile(current_c64.tolist(), .90)):
            raise P12IntegrityError(f"{category}/seed{seed}/t{position}: historical pre-event C64 q90 differs")

        if bool(wr["changed"]):
            if first_update_position is None:
                first_update_position = position
            if update_index >= len(transport):
                raise P12IntegrityError(f"{category}/seed{seed}/t{position}: missing C64 transport update")
            c64_row = transport[update_index]
            post_vector = c64_after[update_index]
            if (int(c64_row["position"]) != position or
                    c64_row.get("memory_before_sha256") != current_memory_sha or
                    c64_row.get("memory_after_sha256") != expected_after_sha or
                    int(c64_row["memory_version_before"]) != current_version or
                    int(c64_row["memory_version_after"]) != expected_after_version or
                    int(c64_row["source_revision_after"]) != after_source_revision or
                    int(c64_row["writer_appended"]) != int(wr["appended"]) or
                    int(c64_row["writer_replaced"]) != int(wr["replaced"]) or
                    int(c64_row["writer_skipped"]) != int(wr["skipped"]) or
                    _sha_json(post_vector.tolist()) != c64_row.get("c64_after_sha256") or
                    float(higher_quantile(post_vector.tolist(), .90)) !=
                    float(c64_row["theta_SCQ_candidate_higher_q90"])):
                raise P12IntegrityError(f"{category}/seed{seed}/t{position}: stored C64 update linkage mismatch")
            current_c64 = post_vector.copy()
            update_index += 1
        else:
            if (expected_after_sha != current_memory_sha or
                    expected_after_version != current_version):
                raise P12IntegrityError(f"{category}/seed{seed}/t{position}: no-content-change memory state changed")

        if bool(wr["changed"]):
            current_memory_sha = expected_after_sha
            current_version = expected_after_version
            prior_vector = c64_by_memory_sha.get(current_memory_sha)
            if prior_vector is not None and not np.array_equal(prior_vector, current_c64):
                raise P12IntegrityError(f"{category}/seed{seed}/t{position}: C64 identity collision")
            c64_by_memory_sha[current_memory_sha] = current_c64.copy()
            c64_q90_by_memory_sha[current_memory_sha] = float(
                higher_quantile(current_c64.tolist(), .90))
            c64_hash_by_memory_sha[current_memory_sha] = _sha_json(current_c64.tolist())
        source_map_sha = after_source_map_sha
        current_source_revision = after_source_revision
        current_legacy_version = after_legacy_version
        memory_after = MemoryIdentity(
            expected_after_version, expected_after_sha, after_source_map_sha,
            after_source_revision, after_legacy_version)
        expected_events.append({
            "position": position,
            "source_key": str(meta["sample_key"]),
            "image_identity_sha256": expected_image,
            "official_group_sha256": str(meta["opaque_group"]),
            "camera_id": int(meta["camera_id"]),
            "queried": queried,
            "static_score": float(static_tape[position]),
            "dynamic_score": float(dynamic_tape[position]),
            "memory_before": memory_before,
            "memory_after": memory_after,
            "writer_signature": dict(wr),
        })

    if update_index != len(transport) or first_update_position is None:
        raise P12IntegrityError(f"{category}/seed{seed}: C64/write transition totals do not align")
    if len(c64_by_memory_sha) != len(set(c64_by_memory_sha)):
        raise P12IntegrityError(f"{category}/seed{seed}: duplicate memory digest mapping")
    c64_states = {
        key: {"scores": vector.copy(),
              "q90": c64_q90_by_memory_sha[key],
              "transport_sha256": c64_hash_by_memory_sha[key]}
        for key, vector in c64_by_memory_sha.items()}
    return expected_events, c64_states, {"first_content_update_position": first_update_position,
                                          "content_update_count": update_index,
                                          "event_count": len(expected_events),
                                          "c64_state_count": len(c64_states)}


def math_isfinite(value: float) -> bool:
    import math
    return math.isfinite(value)


def _first_changed_position(category: str, seed: int = 0) -> int:
    path = STAGE075A / "trajectories" / category / "event_trace.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["support_seeds"][str(seed)]["writer"]["changed"]:
                return int(row["position"])
    raise P12RunnerError(f"Historical {category}/seed{seed} has no memory-content update")


class PersistingEvidenceAdapter(CausalEvidenceRevisionAdapter):
    """Persist private full EvidenceSnapshot payloads for independent audit."""

    def __init__(self, *, private_path: Path,
                 memory_probe: Any = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.private_path = private_path
        self.memory_probe = memory_probe
        self._written: set[tuple[int, str]] = set()
        if private_path.is_file():
            with private_path.open("r", encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        row = json.loads(line)
                        self._written.add((int(row["evidence_revision"]),
                                           str(row["evidence_sha256"])))

    def _persist(self, result: Any, position: int | None) -> None:
        key = (int(result.snapshot.evidence_revision), str(result.evidence_sha256))
        if key in self._written:
            return
        snapshot = result.snapshot
        row = {
            "schema": "stage1a.p12_private_evidence_snapshot.v1",
            "position": position,
            "memory_version": int(snapshot.memory_version),
            "memory_sha256": snapshot.source_exclusion_audit["memory_sha256"],
            "source_map_sha256": snapshot.source_exclusion_audit["source_map_sha256"],
            "evidence_revision": int(snapshot.evidence_revision),
            "evidence_sha256": result.evidence_sha256,
            "protected_c64_scores": list(result.protected_c64_scores),
            "normal_anchors": [asdict(anchor) for anchor in snapshot.normal_anchors],
            "defect_anchors": [asdict(anchor) for anchor in snapshot.defect_anchors],
            "source_exclusion_proofs": dict(result.source_exclusion_proofs),
            "source_exclusion_audit": dict(snapshot.source_exclusion_audit),
        }
        self.private_path.parent.mkdir(parents=True, exist_ok=True)
        with self.private_path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(canonical_json(row) + "\n")
            output.flush()
            os.fsync(output.fileno())
        self._written.add(key)

    def initialize(self, memory: MemoryIdentity):
        if self.memory_probe is not None:
            self.memory_probe("evidence_refresh_before")
        result = super().initialize(memory)
        self._persist(result, None)
        if self.memory_probe is not None:
            self.memory_probe("evidence_refresh_after")
        return result

    def after_event(self, **kwargs: Any):
        queried = bool(kwargs.get("queried"))
        if queried and self.memory_probe is not None:
            self.memory_probe("evidence_refresh_before")
        result = super().after_event(**kwargs)
        if result is not None:
            self._persist(result, int(kwargs["event"].position))
            if queried and self.memory_probe is not None:
                self.memory_probe("evidence_refresh_after")
        return result


def _prepare_historical_data(category: str, seed: int, frozen: Mapping[str, Any]):
    folder = STAGE075A / "trajectories" / category
    static, initial_c64, fixed_sha = _load_stage05_score(category, seed)
    score_path, transport_path = folder / "scores.npz", folder / "c64_transport.json"
    events, c64_states, summary = _trajectory_records(
        category, seed, frozen, score_path, transport_path, static, initial_c64)
    _attach_c64_events(events, c64_states)
    expected_fixed = _read_json(STAGE05 / "score_tape_manifest.json")["classes"][category][
        "score_tapes"][str(seed)]["fixed_support_sha256"]
    if fixed_sha != expected_fixed:
        raise P12IntegrityError(f"{category}/seed{seed}: fixed support hash differs")
    return events, c64_states, static, initial_c64, fixed_sha, summary


def _build_support_bank(category: str, frozen: Mapping[str, Any], dataset_root: Path,
                        impl: Any, cfg: Any, backbone: Any):
    from experiments.p0_protocol_audit import build_vmb, encode_reference_features
    info = frozen["classes"][category]
    prefix = info["prefix"]
    support_paths = sorted({
        stage05._absolute_image_path(dataset_root, prefix, relative)
        for seed in range(5) for relative in info["supports"][str(seed)]["images"]})
    # Only the 4x5 frozen train-normal support images are encoded here. No C64
    # or full-stream image features are regenerated.
    support_features = encode_reference_features(impl, backbone, support_paths, cfg)
    banks: dict[int, Any] = {}
    fixed: dict[int, str] = {}
    for seed in range(5):
        chosen = [stage05._absolute_image_path(dataset_root, prefix, relative)
                  for relative in info["supports"][str(seed)]["images"]]
        bank = build_vmb(impl, support_features, chosen, cfg)
        live, fixed_sha = isolate_native_bank_k256(bank)
        banks[seed] = live
        fixed[seed] = fixed_sha
    return support_features, banks, fixed


def _event_factory(events: list[dict[str, Any]], verified_tapes: Any,
                   category: str, dataset_root: Path, frozen: Mapping[str, Any],
                   image_digest_cache: dict[str, str] | None = None,
                   static_bank_identity_sha256: str = ""):
    query_set = frozenset(verified_tapes.positions(category))
    stream = frozen["classes"][category]["stream"]
    prefix = frozen["classes"][category]["prefix"]
    def factory(start: int):
        for position in range(start, len(events)):
            item = events[position]
            meta = stream[position]
            queried = position in query_set
            if (item["queried"] != queried or
                    item["image_identity_sha256"] != _sha_bytes(
                        ("image-identity\0" + str(meta["relative_id"])).encode()) or
                    item["official_group_sha256"] != str(meta["opaque_group"]) or
                    int(item["camera_id"]) != int(meta["camera_id"])):
                raise P12IntegrityError(f"{category}/t{position}: query/image/group/camera identity mismatch")
            expected_tape_group = verified_tapes.group_identity_by_class_position[
                category].get(position)
            if expected_tape_group is not None and _sha_bytes(
                    ("official-sample-group\0" + item["source_key"]).encode("utf-8")) != expected_tape_group:
                raise P12IntegrityError(
                    f"{category}/t{position}: source group differs from frozen query tape")
            image_sha = None
            if queried:
                image_path = stage05._absolute_image_path(
                    dataset_root, prefix, str(meta["relative_id"]))
                if not Path(image_path).is_file():
                    raise P12RunnerError(f"Precommitted query image is unavailable at {position}")
                digest_key = str(Path(image_path).resolve())
                if image_digest_cache is not None and digest_key in image_digest_cache:
                    image_sha = image_digest_cache[digest_key]
                else:
                    image_sha = _sha_file(Path(image_path))
                    if image_digest_cache is not None:
                        image_digest_cache[digest_key] = image_sha
            historical = item["writer_signature"]
            expected_signature = dict(historical)
            expected_signature["source_identity_sha256"] = _sha_bytes(
                item["source_key"].encode("utf-8"))
            yield ReplayEvent(
                position=position, queried=queried, source_key=item["source_key"],
                image_sha256=image_sha, static_score=item["static_score"],
                dynamic_score=item["dynamic_score"],
                memory_before=item["memory_before"], memory_after=item["memory_after"],
                c64_before=item["c64_before"], c64_after=item["c64_after"],
                expected_writer_signature=expected_signature,
                static_memory_sha256=sha256_json({
                    "fixed_support_sha256": static_bank_identity_sha256,
                    "empty_dynamic_memory_sha256": events[0]["memory_before"].memory_sha256,
                }),
                )
    return factory


def _attach_c64_events(events: list[dict[str, Any]], c64_states: Mapping[str, Any]) -> None:
    for item in events:
        try:
            item["c64_before"] = C64Identity(
                item["memory_before"].version, item["memory_before"].memory_sha256,
                c64_states[item["memory_before"].memory_sha256]["q90"],
                c64_states[item["memory_before"].memory_sha256]["transport_sha256"])
            item["c64_after"] = C64Identity(
                item["memory_after"].version, item["memory_after"].memory_sha256,
                c64_states[item["memory_after"].memory_sha256]["q90"],
                c64_states[item["memory_after"].memory_sha256]["transport_sha256"])
        except KeyError as exc:
            raise P12IntegrityError("Historical C64 state is not bound to a memory version") from exc


def _process_peak_bytes() -> int | None:
    try:
        import psutil
        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", getattr(info, "rss", 0))) or None
    except Exception:
        return None


def _private_tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


class P12ReusableNativeScoreService:
    """Reuse one fixed v22 template while installing exact versioned memory.

    This is a P1.2 call-path optimization only. It invokes the same
    ``native_bridge.native_score`` function and dynamic tensors as
    ``NativeScoreService.score``; no score equation, bank or configuration is
    changed. Reusing the static template avoids deep-copying the full fixed
    support bank for every one of the bounded anchor-score calls.
    """

    def __init__(self, service: NativeScoreService):
        from experiments.risk_safe import native_bridge
        self._bridge = native_bridge
        self._service = service
        self._bank = copy.deepcopy(service.native_template)
        self.calls = 0

    def score(self, bundle: Any, memory: Any):
        from .protocol import ScoreResult
        from .memory_state import CalibrationGreedyState
        if not isinstance(memory, CalibrationGreedyState):
            raise P12RunnerError("Native anchor score requires the exact Greedy memory state")
        layers = self._bridge.dynamic_codebooks(self._bank)
        if len(layers) != len(memory.layers):
            raise P12RunnerError("Native score template layer count differs from memory")
        for index, (codebook, tensor, capacity) in enumerate(
                zip(layers, memory.layers, memory.capacities)):
            if (int(codebook.K) != int(capacity) or
                    int(codebook.feat_dim) != int(tensor.shape[1]) or
                    codebook.protos.device != tensor.device or
                    codebook.protos.dtype != tensor.dtype or len(tensor) > int(codebook.K)):
                raise P12RunnerError(f"Native score template layer {index} is incompatible")
            codebook.protos = tensor.detach().clone().contiguous()
        if self._bridge.snapshot(self._bank, version=memory.version).sha256() != memory.sha256():
            raise P12RunnerError("Reusable Native score bank failed exact memory installation")
        image, features = bundle
        score, score_map = self._bridge.native_score(
            self._service.impl, self._bank, image, features, self._service.p0_config)
        state = {"memory": memory.sha256(), "scorer": self._service.config.sha256,
                 "scientific_profile": SCORER_CONFIG_SHA256,
                 "fixed_support": self._service.fixed_support_sha256,
                 "kind": "raw_native_score_bank"}
        self.calls += 1
        return ScoreResult(float(score), memory.version,
                           _sha_bytes(canonical_json(state).encode("utf-8")), score_map)


def _native_config(impl: Any):
    from experiments.p0_protocol_audit import Config
    scorer = DEFAULT_PROFILE.scorer
    return Config(
        device="cuda", augmentation_seed=scorer.augmentation_seed,
        bank_seed=scorer.bank_seed, fixed_capacity=scorer.fixed_capacity,
        fixed_per_image=scorer.fixed_per_image,
        augmentations=scorer.augmentations, include_identity=scorer.include_identity,
        multiplier=scorer.multiplier, feature_fuse=scorer.feature_fuse,
        topk_patches=scorer.topk_patches,
        image_aggregation=scorer.image_aggregation,
        image_topk=scorer.image_topk, image_quantile=scorer.image_quantile,
        knn=scorer.knn)


def _native_bundle(bundle: Any, device: str):
    image, features = bundle
    return image, [item.to(device=device).contiguous() for item in features]


def _group_sha(source_key: str) -> str:
    return _sha_bytes(source_key.encode("utf-8"))


def _new_writer(impl: Any, native_bank: Any, c64_groups: Iterable[str],
                initial_c64: np.ndarray) -> SharedNativeGreedyWriter:
    from .stage075a import higher_quantile
    return SharedNativeGreedyWriter(
        impl, native_bank,
        theta_ref=higher_quantile(initial_c64.tolist(), .90),
        protected_source_keys=tuple(c64_groups), sigma_ref=0.02)


def _trajectory_callbacks(*, writer: SharedNativeGreedyWriter,
                          native_service: NativeScoreService,
                          reusable_service: P12ReusableNativeScoreService,
                          impl: Any, backbone: Any,
                          m0_memory: Any, config: Any,
                          image_digest_cache: dict[str, str],
                          verified_tapes: Any, category: str,
                          dataset_root: Path, frozen: Mapping[str, Any],
                          encoder_fingerprint: str, preprocessing_fingerprint: str,
                          c64_states: Mapping[str, Any], private_evidence_path: Path,
                          scorer_config_sha256: str, scorer_source_sha256: str,
                          budget: Any, memory_probe: Any = None,
                          evidence_adapter_override: Any = None):
    from .stage1a_p1_reconstruction import shared_writer_memory_identity
    info = frozen["classes"][category]
    support_groups = info["supports"]
    c64_groups = info["c64"]["groups"]
    support_group_sha = [_group_sha(str(key)) for seed in range(5)
                         for key in support_groups[str(seed)]["groups"]]
    protected_group_sha = [_group_sha(str(key)) for key in c64_groups]
    evidence_cache = VersionedAnchorEvidenceCache(budget)
    lso = NativeAllLayerLSOAdapter(
        writer, score_fn=lambda anchor_ref, memory:
        reusable_service.score(_native_bundle(anchor_ref["bundle"], "cuda"), memory))

    def anchor_factory(event: ReplayEvent, label: int, bundle: Any) -> QueryAnchor:
        if event.image_sha256 is None:
            raise P12IntegrityError("Queried anchor lacks its image-byte identity")
        group_sha = _group_sha(event.source_key)
        anchor_id = sha256_json({
            "schema": "stage1a.p12_query_anchor.v1",
            "category": category, "position": event.position,
            "image_sha256": event.image_sha256,
            "source_group_sha256": group_sha,
        })
        return QueryAnchor(
            anchor_id, group_sha, "normal" if label == 0 else "defect",
            {"source_key": event.source_key, "bundle": bundle})

    def protected_c64(memory: MemoryIdentity) -> Mapping[str, Any]:
        try:
            state = c64_states[memory.memory_sha256]
        except KeyError as exc:
            raise P12IntegrityError("Current memory has no linked historical C64 vector") from exc
        return state

    evidence = evidence_adapter_override
    if evidence is None:
        evidence = PersistingEvidenceAdapter(
            writer=writer, cache=evidence_cache, scorer=lso,
            anchor_factory=anchor_factory, protected_c64_state_fn=protected_c64,
            scorer_config_sha256=scorer_config_sha256,
            scorer_source_sha256=scorer_source_sha256,
            protected_group_sha256s=protected_group_sha,
            deployed_support_group_sha256s=support_group_sha,
            private_path=private_evidence_path,
            memory_probe=memory_probe)

    def feature_key(event: ReplayEvent) -> FeatureKey:
        return FeatureKey(event.image_sha256, encoder_fingerprint,
                          preprocessing_fingerprint)

    def encode_feature(event: ReplayEvent):
        from experiments.p0_protocol_audit import encode_query
        record = frozen["classes"][category]["stream"][event.position]
        path = stage05._absolute_image_path(
            dataset_root, info["prefix"], str(record["relative_id"]))
        return encode_query(impl, backbone, path, config)

    def static_map(event: ReplayEvent, bundle: Any):
        score = native_service.score(_native_bundle(bundle, "cuda"), m0_memory)
        return float(score.score), _cpu_map(score.score_map)

    def writer_fn(event: ReplayEvent, queried: bool, label: int | None,
                  bundle: Any, static_amap: Any):
        from .stage1a_p11_production import apply_historical_shared_writer
        full_features = () if bundle is None else bundle[1]
        score_map = None if static_amap is None else _native_map(static_amap, "cuda")
        result = writer.apply(
            queried=queried, label=label, source_key=event.source_key,
            score=float(event.static_score), static_amap=score_map,
            full_features=full_features, step=event.position)
        from .stage1a_p11_production import historical_writer_record
        return historical_writer_record(result, writer)

    def c64_fn(memory: MemoryIdentity) -> C64Identity:
        state = protected_c64(memory)
        return C64Identity(memory.version, memory.memory_sha256,
                           float(state["q90"]), str(state["transport_sha256"]))

    return {
        "evidence_cache": evidence_cache,
        "evidence_adapter": evidence,
        "feature_key_fn": feature_key,
        "encode_feature_fn": encode_feature,
        "static_map_fn": static_map,
        "writer_fn": writer_fn,
        "state_fn": lambda: shared_writer_memory_identity(writer),
        "c64_fn": c64_fn,
        "writer_state_sha256_fn": lambda: _writer_digest(writer),
        "feature_identity": {"encoder_fingerprint_sha256": encoder_fingerprint,
                              "preprocessing_fingerprint_sha256": preprocessing_fingerprint},
    }


def _make_label_reader(category: str, tapes: Any, frozen: Mapping[str, Any],
                       json_root: Path) -> QueriedPositionLabelReader:
    info = frozen["classes"][category]
    return QueriedPositionLabelReader(
        category=category,
        path=json_root / f"{category}.json",
        expected_source_sha256=str(info["class_json_sha256"]),
        positions=tapes.positions(category),
        group_sha256_by_position=dict(
            tapes.group_identity_by_class_position[category]))


def _checkpoint_bytes(checkpoint: P11PrivateCheckpoint) -> bytes:
    return pickle.dumps(checkpoint, protocol=5)


def _write_private_checkpoint(path: Path, checkpoint: P11PrivateCheckpoint) -> str:
    payload = _checkpoint_bytes(checkpoint)
    _atomic_write(path, payload)
    return _sha_bytes(payload)


def _capture_checkpoint(*, result: Any, production_runner: P11ProductionReplayRunner,
                        writer: SharedNativeGreedyWriter, trajectory_id: str,
                        category: str, scorer_config_sha256: str,
                        scorer_source_sha256: str,
                        evidence_adapter: PersistingEvidenceAdapter,
                        evidence_cache: VersionedAnchorEvidenceCache) -> P11PrivateCheckpoint:
    return P11PrivateCheckpoint.capture(
        result=result, production_runner=production_runner, writer=writer,
        trajectory_id=trajectory_id, category=category,
        scorer_config_sha256=scorer_config_sha256,
        scorer_source_sha256=scorer_source_sha256,
        evidence_cache=evidence_cache,
        causal_evidence_adapter=evidence_adapter)


def _run_to(*, target_position: int, category: str, production_runner: P11ProductionReplayRunner,
            kwargs: dict[str, Any], checkpoint: Any = None,
            chunk_size: int = 256, progress_prefix: str = ""):
    """Run an exact event interval with resumable public digest checkpoints."""
    cursor = int(checkpoint.next_position) if checkpoint is not None else 0
    if target_position < cursor:
        raise P12RunnerError("Requested P1.2 stop position precedes replay cursor")
    result = None
    while cursor < target_position:
        count = min(chunk_size, target_position - cursor)
        result = production_runner.run(
            **kwargs, checkpoint=checkpoint, max_events=count)
        checkpoint = result.checkpoint
        cursor = result.next_position
        print(f"{progress_prefix} events={cursor}/{target_position} "
              f"queries_checked={result.query_events} status={result.status}", flush=True)
    if result is None:
        result = production_runner.run(
            **kwargs, checkpoint=checkpoint, max_events=0)
    return result


def _make_static_service(impl: Any, native_bank: Any, fixed_sha: str,
                         p0_config: Any) -> NativeScoreService:
    return NativeScoreService(impl, copy.deepcopy(native_bank), p0_config,
                              DEFAULT_PROFILE.scorer, fixed_sha)


def _make_production_runner(tapes: Any, standing: StandingP12Authorization):
    return P11ProductionReplayRunner(
        None, None, tapes, standing_authorization=standing)


def _run_one_trajectory(*, category: str, seed: int, events: list[dict[str, Any]],
                        c64_states: Mapping[str, Any], static_tape: np.ndarray,
                        initial_c64: np.ndarray, fixed_sha: str, native_bank: Any,
                        impl: Any, backbone: Any, p0_config: Any,
                        production_runner: P11ProductionReplayRunner,
                        tapes: Any, frozen: Mapping[str, Any], dataset_root: Path,
                        label_reader: QueriedPositionLabelReader,
                        image_digest_cache: dict[str, str], source_fingerprint: str,
                        scorer_config_sha256: str, scorer_source_sha256: str,
                        encoder_fingerprint: str, preprocessing_fingerprint: str,
                        output: Path, chunk_size: int, start_checkpoint: Any = None,
                        stop_at: int | None = None,
                        memory_probe: Any = None,
                        evidence_adapter_override: Any = None,
                        private_evidence_path: Path | None = None):
    info = frozen["classes"][category]
    expected_fixed = _read_json(STAGE05 / "score_tape_manifest.json")["classes"][category][
        "score_tapes"][str(seed)]["fixed_support_sha256"]
    if fixed_sha != expected_fixed:
        raise P12IntegrityError(f"{category}/seed{seed}: reconstructed fixed bank digest differs")
    writer = _new_writer(impl, native_bank, info["c64"]["groups"], initial_c64)
    m0_memory = copy.deepcopy(writer.bank)
    native_service = _make_static_service(impl, native_bank, fixed_sha, p0_config)
    reusable_service = P12ReusableNativeScoreService(native_service)
    trajectory_id = f"{category}/seed{seed}/Q-A-TS-v1-3pct-seed37"
    evidence_path = private_evidence_path or (
        output / "private" / "evidence" / f"{category}_seed{seed}.jsonl")
    callbacks = _trajectory_callbacks(
        writer=writer, native_service=native_service,
        reusable_service=reusable_service, impl=impl, backbone=backbone,
        m0_memory=m0_memory, config=p0_config,
        image_digest_cache=image_digest_cache,
        verified_tapes=tapes, category=category,
        dataset_root=dataset_root, frozen=frozen,
        encoder_fingerprint=encoder_fingerprint,
        preprocessing_fingerprint=preprocessing_fingerprint,
        c64_states=c64_states, private_evidence_path=evidence_path,
        scorer_config_sha256=scorer_config_sha256,
        scorer_source_sha256=scorer_source_sha256,
        budget=production_runner.runner.budget,
        memory_probe=memory_probe,
        evidence_adapter_override=evidence_adapter_override)
    factory = _event_factory(events, tapes, category, dataset_root,
                             frozen, image_digest_cache, fixed_sha)
    kwargs = {
        "events_factory": factory,
        "total_events": len(events),
        "trajectory_id": trajectory_id,
        "category": category,
        "source_fingerprint_sha256": source_fingerprint,
        "scorer_config_sha256": scorer_config_sha256,
        "scorer_source_sha256": scorer_source_sha256,
        "state_fn": callbacks["state_fn"],
        "c64_fn": callbacks["c64_fn"],
        "writer_state_sha256_fn": callbacks["writer_state_sha256_fn"],
        "feature_key_fn": callbacks["feature_key_fn"],
        "encode_feature_fn": callbacks["encode_feature_fn"],
        "label_reader": label_reader,
        "static_map_fn": callbacks["static_map_fn"],
        "writer_fn": callbacks["writer_fn"],
        "evidence_revision_adapter": callbacks["evidence_adapter"],
    }
    if memory_probe is not None:
        original_encode = kwargs["encode_feature_fn"]

        def measured_encode(event: ReplayEvent):
            memory_probe("feature_encoding_before")
            bundle = original_encode(event)
            memory_probe("feature_encoding_after")
            return bundle

        kwargs["encode_feature_fn"] = measured_encode
    stop = len(events) if stop_at is None else int(stop_at)
    result = _run_to(
        target_position=stop, category=category,
        production_runner=production_runner, kwargs=kwargs,
        checkpoint=start_checkpoint, chunk_size=chunk_size,
        progress_prefix=f"{category}/seed{seed}")
    return result, writer, callbacks, reusable_service, trajectory_id, kwargs


def _verify_environment(dataset_root: Path, json_root: Path,
                        stage05_run: Mapping[str, Any],
                        stage05_scores: Mapping[str, Any]) -> dict[str, Any]:
    expected_branch = EXPECTED_BRANCH
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    if branch != expected_branch:
        raise P12RunnerError(f"Expected branch {expected_branch}, found {branch}")
    if not dataset_root.is_dir() or not json_root.is_dir():
        raise P12RunnerError("Frozen Real-IAD development data or JSON root is unavailable")
    for key, path in (("dataset_root_sha256", dataset_root),
                      ("json_root_sha256", json_root)):
        actual = _sha_bytes(str(path.resolve()).encode())
        if actual != stage05_run.get(key):
            raise P12RunnerError(f"Frozen Stage 0.5 {key} identity mismatch")
    if (frozen := stage05_scores.get("model_provenance")) is None:
        raise P12RunnerError("Stage 0.5 model provenance is missing")
    import torch
    expected_python = Path(r"D:\anaconda3\envs\torch\python.exe").resolve()
    actual_python = Path(sys.executable).resolve()
    if os.name == "nt" and actual_python != expected_python:
        raise P12RunnerError(f"Wrong Python environment: {actual_python}")
    if not torch.cuda.is_available():
        raise P12RunnerError("CUDA is unavailable for the authorized Native replay")
    torch_hub = Path(torch.hub.get_dir())
    checkpoint = torch_hub / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
    source = torch_hub / "facebookresearch_dinov2_main"
    if not checkpoint.is_file() or not source.is_dir():
        raise P12RunnerError("Pinned DINOv2 checkpoint/source is unavailable")
    checkpoint_sha = _sha_file(checkpoint)
    tree_digest = hashlib.sha256()
    for item in sorted((value for value in source.rglob("*.py") if value.is_file()),
                       key=lambda value: value.relative_to(source).as_posix()):
        relative = item.relative_to(source).as_posix().encode()
        tree_digest.update(len(relative).to_bytes(8, "big"))
        tree_digest.update(relative)
        tree_digest.update(bytes.fromhex(_sha_file(item)))
    tree_sha = tree_digest.hexdigest()
    if (checkpoint_sha != frozen.get("checkpoint_sha256") or
            tree_sha != frozen.get("hub_source_tree_sha256")):
        raise P12RunnerError("Local DINOv2 model/source differs from frozen Stage 0.5")
    return {"branch": branch, "head": head,
            "python": str(actual_python), "torch": str(torch.__version__),
            "cuda_available": True,
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_device_count": torch.cuda.device_count(),
            "dataset_root_sha256": _sha_bytes(str(dataset_root.resolve()).encode()),
            "json_root_sha256": _sha_bytes(str(json_root.resolve()).encode()),
            "model_checkpoint_sha256": checkpoint_sha,
            "dinov2_source_tree_sha256": tree_sha}


def _verify_stage05_label_isolation(frozen: Mapping[str, Any],
                                   run_manifest: Mapping[str, Any],
                                   score_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate score-tape isolation without opening offline outcome artifacts."""
    if (frozen.get("test_labels_accessed") is not False or
            frozen.get("test_masks_accessed") is not False or
            run_manifest.get("stage1_or_stage2_run") is not False or
            run_manifest.get("test_masks_opened") is not False or
            score_manifest.get("controller_or_test_label_access") is not False):
        raise P12RunnerError(
            "Stage 0.5 frozen/score metadata does not establish label-free score tapes "
            "and test-mask isolation")
    return {
        "frozen_test_labels_accessed": frozen["test_labels_accessed"],
        "frozen_test_masks_accessed": frozen["test_masks_accessed"],
        "score_tape_controller_or_test_label_access":
            score_manifest["controller_or_test_label_access"],
        "stage05_run_manifest_test_labels_exposed": run_manifest.get("test_labels_exposed"),
        "stage05_run_manifest_test_masks_opened": run_manifest["test_masks_opened"],
        "prior_offline_outcome_artifacts_opened_by_p12": False,
        "p12_label_access_policy": "only current verified Q-A-TS positions, disclosed causally",
    }


def _run_all(output: Path, audit: Path, dataset_root: Path,
             json_root: Path, chunk_size: int) -> dict[str, Any]:
    """Execute Phase A, then the 20 bounded parity/evidence trajectories."""
    started = time.perf_counter()
    if subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                      capture_output=True, text=True, check=True).stdout.strip() != EXPECTED_BRANCH:
        raise P12RunnerError("Working branch differs from the authorized calibration branch")
    tapes = verify_stage05_qats_artifacts(STAGE05)
    if tapes.binding_sha256 != "dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183":
        raise P12RunnerError("Verified Q-A-TS tape aggregate differs from the frozen P1.2 binding")
    frozen = _read_json(STAGE05 / "frozen_inputs.json")
    stage05_run = _read_json(STAGE05 / "run_manifest.json")
    stage05_scores = _read_json(STAGE05 / "score_tape_manifest.json")
    historical = _verify_historical_package()
    c64_linkage = verify_historical_c64_artifact_links(STAGE05, STAGE075A)
    environment = _verify_environment(dataset_root, json_root, stage05_run, stage05_scores)
    source_identity, source_fingerprint, scorer_source = _current_source_identity()
    protocol = _read_json(STAGE075A / "protocol_manifest.json")
    native_identity = _verify_native_source_identity(protocol, stage05_scores)
    if len(P11_CLASSES) != 4 or any(
            frozen["classes"][category]["stream_images"] != stage05.EXPECTED_STREAM_LENGTHS[category]
            for category in P11_CLASSES):
        raise P12RunnerError("Frozen development class registry or stream lengths changed")
    stage05_label_isolation = _verify_stage05_label_isolation(
        frozen, stage05_run, stage05_scores)

    output.mkdir(parents=True, exist_ok=False)
    audit.mkdir(parents=True, exist_ok=False)
    (output / "INCOMPLETE").write_text(
        "P1.2 standing-authorized bounded replay is in progress.\n", encoding="utf-8")
    (output / "private").mkdir()
    (output / "private" / "evidence").mkdir()
    _atomic_json(output / "preflight_manifest.json", {
        "schema": "stage1a.p12_preflight.v2",
        "branch": environment["branch"], "base_head": environment["head"],
        "query_tape_binding_sha256": tapes.binding_sha256,
        "classes": list(P11_CLASSES), "support_seeds": [0, 1, 2, 3, 4],
        "query_rate": 0.03, "query_seed": 37,
        "historical_replay_artifacts": historical,
        "historical_c64_linkage": c64_linkage,
        "native_execution_provenance": native_identity,
        "stage05_label_isolation": stage05_label_isolation,
        "environment": environment,
        "standing_authorization_context": "explicit_user_instruction_active_task",
        "approval_record_created": False,
        "confirmation_or_mask_access": False,
    })

    standing = StandingP12Authorization.for_active_user_instruction(tapes)
    production_runner = _make_production_runner(tapes, standing)
    import torch
    from experiments.p0_protocol_audit import require_sources
    from experiments.calibration.stage1a_p1_reconstruction import source_fingerprints
    impl = require_sources()
    p0_config = _native_config(impl)
    from .native import validate_p0_config
    validate_p0_config(p0_config, DEFAULT_PROFILE.scorer)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    impl.set_seed(0)
    backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
    encoder_fingerprint = _sha_json({
        "checkpoint_sha256": environment["model_checkpoint_sha256"],
        "source_tree_sha256": environment["dinov2_source_tree_sha256"],
        "model": DEFAULT_PROFILE.scorer.model_name,
    })
    preprocessing_fingerprint = _sha_json({
        "image_size": DEFAULT_PROFILE.scorer.image_size,
        "resize_size": DEFAULT_PROFILE.scorer.resize_size,
        "feature_fuse": DEFAULT_PROFILE.scorer.feature_fuse,
        "use_fp16": DEFAULT_PROFILE.scorer.use_fp16,
        "p0_source_sha256": _sha_file(ROOT / "experiments/p0_protocol_audit.py"),
        "vmb_source_sha256": _sha_file(ROOT / "vmb_visionad_new.py"),
    })
    # P1.2 source commitment binds current code and historical source/artifacts.
    source_fingerprint = _sha_json({
        "source_identity": source_identity,
        "source_fingerprint": source_fingerprint,
        "historical_protocol_manifest_sha256": historical["protocol_manifest_sha256"],
        "historical_event_score_files": historical["trajectory_file_sha256"],
        "query_tape_binding_sha256": tapes.binding_sha256,
    })

    image_digest_cache: dict[str, str] = {}
    label_readers: dict[str, QueriedPositionLabelReader] = {}
    support_feature_cache: dict[str, dict[str, Any]] = {}
    support_paths_seen: dict[str, set[str]] = {}
    support_reference_calls = 0
    support_image_identities: set[str] = set()
    trajectory_results: list[dict[str, Any]] = []
    checkpoint_summaries: list[dict[str, Any]] = []
    exact_event_checks = 0
    query_features_shared = 0
    phase_a_checkpoint_identity: dict[str, Any] | None = None
    phase_a_end = _first_changed_position("audiojack", 0) + 1
    phase_a_split = max(0, phase_a_end // 2)
    if phase_a_split >= phase_a_end:
        phase_a_split = 0

    def support_bank(category: str, seed: int):
        nonlocal support_reference_calls
        from experiments.p0_protocol_audit import build_vmb, encode_reference_features
        cache = support_feature_cache.setdefault(category, {})
        seen = support_paths_seen.setdefault(category, set())
        info = frozen["classes"][category]
        prefix = info["prefix"]
        paths = [str(stage05._absolute_image_path(
            dataset_root, prefix, relative))
            for relative in info["supports"][str(seed)]["images"]]
        new_paths = sorted(set(paths) - seen)
        if new_paths:
            cache.update(encode_reference_features(impl, backbone, new_paths, p0_config))
            seen.update(new_paths)
            support_reference_calls += len(new_paths)
            for path in new_paths:
                support_image_identities.add(_sha_bytes(path.encode("utf-8")))
        bank = build_vmb(impl, cache, paths, p0_config)
        live, fixed_sha = isolate_native_bank_k256(bank)
        return live, fixed_sha

    def label_reader(category: str):
        if category not in label_readers:
            label_readers[category] = _make_label_reader(category, tapes, frozen, json_root)
        return label_readers[category]

    def prepare(category: str, seed: int):
        events, c64_states, static, initial_c64, fixed_sha, _summary = _prepare_historical_data(
            category, seed, frozen)
        return events, c64_states, static, initial_c64, fixed_sha

    # Phase A: one class/seed, split around the first expected content update,
    # serialize/reconstruct P11 private state, and resume through that update.
    first_category, first_seed = "audiojack", 0
    events, c64_states, static, initial_c64, fixed_sha = prepare(first_category, first_seed)
    bank, reconstructed_fixed = support_bank(first_category, first_seed)
    if reconstructed_fixed != fixed_sha:
        raise P12IntegrityError("Phase A reconstructed fixed Native support differs")
    partial_result, writer, callbacks, reusable, trajectory_id, kwargs = _run_one_trajectory(
        category=first_category, seed=first_seed, events=events,
        c64_states=c64_states, static_tape=static, initial_c64=initial_c64,
        fixed_sha=fixed_sha, native_bank=bank, impl=impl, backbone=backbone,
        p0_config=p0_config, production_runner=production_runner,
        tapes=tapes, frozen=frozen, dataset_root=dataset_root,
        label_reader=label_reader(first_category), image_digest_cache=image_digest_cache,
        source_fingerprint=source_fingerprint,
        scorer_config_sha256=SCORER_CONFIG_SHA256,
        scorer_source_sha256=scorer_source,
        encoder_fingerprint=encoder_fingerprint,
        preprocessing_fingerprint=preprocessing_fingerprint,
        output=output, chunk_size=chunk_size, stop_at=phase_a_split)
    first_private = _capture_checkpoint(
        result=partial_result, production_runner=production_runner,
        writer=writer, trajectory_id=trajectory_id, category=first_category,
        scorer_config_sha256=SCORER_CONFIG_SHA256,
        scorer_source_sha256=scorer_source,
        evidence_adapter=callbacks["evidence_adapter"],
        evidence_cache=callbacks["evidence_cache"])
    phase_checkpoint_path = output / "private" / "phase_a_reconstructed_checkpoint.pt"
    phase_checkpoint_sha = _write_private_checkpoint(phase_checkpoint_path, first_private)
    restored = pickle.loads(phase_checkpoint_path.read_bytes())
    previous_label_reader = label_readers.pop(first_category)
    previous_label_reader.close()

    # Reconstruct independent runner/writer/cache objects and let the label
    # provider skip already-disclosed tape positions without decoding them.
    production_runner = _make_production_runner(tapes, standing)
    resumed_writer = _new_writer(impl, copy.deepcopy(bank),
                                 frozen["classes"][first_category]["c64"]["groups"],
                                 initial_c64)
    resumed_m0 = copy.deepcopy(resumed_writer.bank)
    resumed_static = _make_static_service(impl, bank, fixed_sha, p0_config)
    resumed_reusable = P12ReusableNativeScoreService(resumed_static)
    # Build the restoration adapter before loading its private causal state.
    resumed_callbacks = _trajectory_callbacks(
        writer=resumed_writer, native_service=resumed_static,
        reusable_service=resumed_reusable, impl=impl, backbone=backbone,
        m0_memory=resumed_m0, config=p0_config,
        image_digest_cache=image_digest_cache,
        verified_tapes=tapes, category=first_category,
        dataset_root=dataset_root, frozen=frozen,
        encoder_fingerprint=encoder_fingerprint,
        preprocessing_fingerprint=preprocessing_fingerprint,
        c64_states=c64_states,
        private_evidence_path=output / "private" / "evidence" / "audiojack_seed0.jsonl",
        scorer_config_sha256=SCORER_CONFIG_SHA256,
        scorer_source_sha256=scorer_source,
        budget=production_runner.runner.budget)
    restored.restore(
        production_runner=production_runner, writer=resumed_writer,
        scorer_config_sha256=SCORER_CONFIG_SHA256,
        scorer_source_sha256=scorer_source,
        evidence_cache=resumed_callbacks["evidence_cache"],
        causal_evidence_adapter=resumed_callbacks["evidence_adapter"])
    phase_a_kwargs = {
        **kwargs,
        "state_fn": resumed_callbacks["state_fn"],
        "c64_fn": resumed_callbacks["c64_fn"],
        "writer_state_sha256_fn": resumed_callbacks["writer_state_sha256_fn"],
        "feature_key_fn": resumed_callbacks["feature_key_fn"],
        "encode_feature_fn": resumed_callbacks["encode_feature_fn"],
        "static_map_fn": resumed_callbacks["static_map_fn"],
        "writer_fn": resumed_callbacks["writer_fn"],
        "label_reader": label_reader(first_category),
        "evidence_revision_adapter": resumed_callbacks["evidence_adapter"],
    }
    phase_a_result = _run_to(
        target_position=phase_a_end, category=first_category,
        production_runner=production_runner, kwargs=phase_a_kwargs,
        checkpoint=restored.replay_checkpoint, chunk_size=chunk_size,
        progress_prefix="PHASE_A/audiojack/seed0")
    if (phase_a_result.next_position != phase_a_end or
            phase_a_result.checkpoint.memory.version <= 0 or
            phase_a_result.checkpoint.c64.memory_sha256 !=
            phase_a_result.checkpoint.memory.memory_sha256 or
            production_runner.runner.budget.anchor_score_calls < 1):
        raise P1ParityMismatch("Phase A did not verify first update and one real Native anchor score")
    phase_a_checkpoint_identity = {
        "split_position": phase_a_split,
        "first_content_update_position": phase_a_end - 1,
        "phase_a_end_position_exclusive": phase_a_end,
        "phase_a_event_chain_sha256": phase_a_result.event_chain_sha256,
        "phase_a_query_response_sha256": phase_a_result.query_response_sha256,
        "phase_a_memory": asdict(phase_a_result.checkpoint.memory),
        "serialized_checkpoint_sha256": phase_checkpoint_sha,
        "restored_context_sha256": restored.private_state_sha256,
        "native_anchor_score_calls_after_phase_a": production_runner.runner.budget.anchor_score_calls,
        "phase_a_exact_parity": "PASS",
    }
    from .stage1a_p11_production import verify_controller_snapshot_compatibility
    actual_snapshot_compatibility = verify_controller_snapshot_compatibility(
        resumed_callbacks["evidence_adapter"].last_result.snapshot)
    phase_a_final_checkpoint = _capture_checkpoint(
        result=phase_a_result, production_runner=production_runner,
        writer=resumed_writer, trajectory_id=trajectory_id, category=first_category,
        scorer_config_sha256=SCORER_CONFIG_SHA256,
        scorer_source_sha256=scorer_source,
        evidence_adapter=resumed_callbacks["evidence_adapter"],
        evidence_cache=resumed_callbacks["evidence_cache"])
    final_checkpoint_path = output / "private" / "latest_p1_checkpoint.pt"
    final_checkpoint_sha = _write_private_checkpoint(final_checkpoint_path,
                                                     phase_a_final_checkpoint)

    # Continue the same audiojack/seed-0 trajectory to completion. Its prefix
    # is not recomputed; P1 uses the restored exact caches and writer state.
    full_result = _run_to(
        target_position=len(events), category=first_category,
        production_runner=production_runner, kwargs=phase_a_kwargs,
        checkpoint=phase_a_result.checkpoint, chunk_size=chunk_size,
        progress_prefix="PHASE_B/audiojack/seed0")
    exact_event_checks += phase_a_end + full_result.events_checked
    trajectory_results.append({
        "trajectory_id": trajectory_id, "category": first_category,
        "support_seed": first_seed, "status": full_result.status,
        "complete": full_result.complete, "next_position": full_result.next_position,
        "events_checked_after_phase_a_resume": full_result.events_checked,
        "events_checked_total": len(events),
        "query_events_total": len(tapes.positions(first_category)),
        "memory_parity_events_total": 2 * len(events),
        "writer_parity_events_total": len(events),
        "c64_parity_events_total": 2 * len(events),
        "event_chain_sha256": full_result.event_chain_sha256,
        "query_response_sha256": full_result.query_response_sha256,
        "final_memory": asdict(full_result.checkpoint.memory),
        "evidence_snapshot_count": len(resumed_callbacks["evidence_adapter"].revisions),
        "private_evidence_sha256": _sha_file(
            output / "private" / "evidence" / "audiojack_seed0.jsonl"),
        "native_anchor_score_calls": production_runner.runner.budget.anchor_score_calls,
        "native_static_map_calls": production_runner.runner.budget.static_map_calls,
    })
    final_checkpoint = _capture_checkpoint(
        result=full_result, production_runner=production_runner,
        writer=resumed_writer, trajectory_id=trajectory_id, category=first_category,
        scorer_config_sha256=SCORER_CONFIG_SHA256,
        scorer_source_sha256=scorer_source,
        evidence_adapter=resumed_callbacks["evidence_adapter"],
        evidence_cache=resumed_callbacks["evidence_cache"])
    final_checkpoint_sha = _write_private_checkpoint(final_checkpoint_path, final_checkpoint)
    checkpoint_summaries.append({"trajectory_id": trajectory_id,
                                 **final_checkpoint.public_summary(),
                                 "private_checkpoint_sha256": final_checkpoint_sha})

    # Phase B/C: all remaining four-class/five-seed trajectories, using the
    # fixed tapes and shared feature/map/label caches. No controller metrics.
    for category in P11_CLASSES:
        for seed in range(5):
            if category == first_category and seed == first_seed:
                continue
            events, c64_states, static, initial_c64, fixed_sha = prepare(category, seed)
            bank, reconstructed_fixed = support_bank(category, seed)
            if reconstructed_fixed != fixed_sha:
                raise P1IntegrityError(f"{category}/seed{seed}: fixed support digest differs")
            result, writer, callbacks, reusable, trajectory_id, _kwargs = _run_one_trajectory(
                category=category, seed=seed, events=events,
                c64_states=c64_states, static_tape=static, initial_c64=initial_c64,
                fixed_sha=fixed_sha, native_bank=bank, impl=impl, backbone=backbone,
                p0_config=p0_config, production_runner=production_runner,
                tapes=tapes, frozen=frozen, dataset_root=dataset_root,
                label_reader=label_reader(category),
                image_digest_cache=image_digest_cache,
                source_fingerprint=source_fingerprint,
                scorer_config_sha256=SCORER_CONFIG_SHA256,
                scorer_source_sha256=scorer_source,
                encoder_fingerprint=encoder_fingerprint,
                preprocessing_fingerprint=preprocessing_fingerprint,
                output=output, chunk_size=chunk_size)
            exact_event_checks += result.events_checked
            trajectory_results.append({
                "trajectory_id": trajectory_id, "category": category,
                "support_seed": seed, "status": result.status,
                "complete": result.complete, "next_position": result.next_position,
                "events_checked": result.events_checked,
                "query_events": result.query_events,
                "memory_parity_events": result.memory_parity_events,
                "writer_parity_events": result.writer_parity_events,
                "c64_parity_events": result.c64_parity_events,
                "event_chain_sha256": result.event_chain_sha256,
                "query_response_sha256": result.query_response_sha256,
                "final_memory": asdict(result.checkpoint.memory),
                "evidence_snapshot_count": len(callbacks["evidence_adapter"].revisions),
                "private_evidence_sha256": _sha_file(
                    output / "private" / "evidence" / f"{category}_seed{seed}.jsonl"),
                "native_anchor_score_calls": production_runner.runner.budget.anchor_score_calls,
                "native_static_map_calls": production_runner.runner.budget.static_map_calls,
            })
            final_checkpoint = _capture_checkpoint(
                result=result, production_runner=production_runner,
                writer=writer, trajectory_id=trajectory_id, category=category,
                scorer_config_sha256=SCORER_CONFIG_SHA256,
                scorer_source_sha256=scorer_source,
                evidence_adapter=callbacks["evidence_adapter"],
                evidence_cache=callbacks["evidence_cache"])
            final_checkpoint_sha = _write_private_checkpoint(final_checkpoint_path,
                                                             final_checkpoint)
            checkpoint_summaries.append({"trajectory_id": trajectory_id,
                                         **final_checkpoint.public_summary(),
                                         "private_checkpoint_sha256": final_checkpoint_sha})
            _atomic_json(output / "progress_manifest.json", {
                "schema": "stage1a.p12_progress.v1",
                "completed_trajectories": len(trajectory_results),
                "total_trajectories": 20,
                "last_trajectory": trajectory_id,
                "compute_counters": asdict(production_runner.runner.budget),
            })
            del bank, writer, callbacks, reusable, result, events, c64_states

    budget = production_runner.runner.budget
    actual_label_positions = {
        category: list(label_readers[category].decoded_positions)
        for category in label_readers}
    expected_query_total = sum(row["query_count"] for row in tapes.bindings)
    if (len(trajectory_results) != 20 or
            not all(row["complete"] and row["status"] == "PASS_EXACT_PARITY"
                    for row in trajectory_results) or
            budget.queried_label_reads != expected_query_total or
            sum(len(values) for values in actual_label_positions.values()) != expected_query_total or
            len(set(image_digest_cache.values())) > budget.authorization.max_feature_encodings or
            budget.feature_encodings != len(production_runner.runner.features._values)):
        raise P1ParityMismatch("Final P1.2 trajectory/query/feature completeness checks failed")
    evidence_total = sum(int(row["evidence_snapshot_count"]) for row in trajectory_results)
    if (budget.feature_encodings > 459 or budget.static_map_calls > 2295 or
            budget.anchor_score_calls > 75406):
        raise P1ParityMismatch("Final P1.2 compute usage exceeded a frozen hard ceiling")
    for reader in label_readers.values():
        reader.close()
    elapsed = time.perf_counter() - started
    peak_cuda = int(torch.cuda.max_memory_allocated())
    compute_accounting = {
        "queried_label_reads": budget.queried_label_reads,
        "queried_label_allowlist_total": expected_query_total,
        "decoded_queried_positions_by_class": {
            category: len(values) for category, values in actual_label_positions.items()},
        "unqueried_labels_decoded": False,
        "query_feature_encodings": budget.feature_encodings,
        "query_image_hash_reads": len(image_digest_cache),
        "frozen_support_reference_image_encodings": support_reference_calls,
        "unique_support_image_identities": len(support_image_identities),
        "static_map_native_calls": budget.static_map_calls,
        "query_anchor_native_calls": budget.anchor_score_calls,
        "reusable_anchor_native_service_calls": sum(
            int(row["native_anchor_score_calls"]) for row in trajectory_results[-1:]),
        "historical_full_stream_score_calls_reused": 0,
        "historical_dynamic_score_values_reused": sum(
            stage05.EXPECTED_STREAM_LENGTHS[c] * 5 for c in P11_CLASSES),
        "historical_c64_vectors_reused": c64_linkage["class_seed_initial_c64_vectors_verified"] +
            c64_linkage["postwrite_c64_vectors_verified"],
        "new_independent_c64_score_calls": 0,
        "event_positions_exactly_checked": exact_event_checks,
        "memory_updates_exactly_checked": sum(
            int(row["final_memory"]["version"]) for row in trajectory_results),
        "evidence_snapshots_persisted": evidence_total,
        "elapsed_seconds": elapsed,
        "peak_gpu_allocated_bytes": peak_cuda,
        "peak_cpu_process_bytes": _process_peak_bytes(),
        "private_artifact_bytes": _private_tree_bytes(output / "private"),
        "ceilings": {"query_feature_encodings": 459,
                     "queried_label_reads": 459,
                     "static_map_native_calls": 2295,
                     "query_anchor_native_calls": 75406},
    }
    if (budget.static_map_calls != 2295 or
            sum(len(values) for values in actual_label_positions.values()) != expected_query_total):
        raise P12IntegrityError("P1.2 static-map or queried-label use did not match the frozen schedule")
    exact_parity = {
        "status": "PASS_EXACT_PARITY_ALL_20_TRAJECTORIES",
        "trajectory_count": len(trajectory_results),
        "event_count": exact_event_checks,
        "support_classes": list(P11_CLASSES), "support_seeds": [0, 1, 2, 3, 4],
        "memory_before_after_and_source_state_fields": "exact_all_events",
        "writer_record_schema": "historical stage075a_runner._writer_record exact fields",
        "c64_q90_and_transport": "exact_historical_linkage_no_rescore",
        "trajectories": trajectory_results,
    }
    evidence_manifest = {
        "status": "PASS_DENSE_CAUSAL_EVIDENCE_ALL_20_TRAJECTORIES",
        "evidence_snapshots": evidence_total,
        "queried_anchor_arrivals": expected_query_total * 5,
        "queried_anchor_identities": expected_query_total,
        "protected_c64_scores_per_snapshot": 64,
        "normal_policy": "actual Native all-layer source-aware LSO",
        "defect_policy": "score against current memory; never written",
        "cache_key": "memory/content/source map/anchor/scorer/LSO exact identity",
        "private_evidence_files": {
            path.name: {"sha256": _sha_file(path), "size_bytes": path.stat().st_size,
                        "raw_scores_private": True}
            for path in sorted((output / "private" / "evidence").glob("*.jsonl"))},
        "controller_snapshot_compatibility": actual_snapshot_compatibility,
    }
    queried_provenance = {
        "authorization_basis": "explicit_user_instruction_active_task",
        "signed_or_timestamped_approval_record_created": False,
        "query_tape_binding_sha256": tapes.binding_sha256,
        "query_protocol_sha256": tapes.bindings[0]["query_protocol_sha256"],
        "query_protocol": "Q-A-TS-v1",
        "query_rate": 0.03, "query_seed": 37,
        "allowed_classes": list(P11_CLASSES),
        "per_class_queried_counts": {row["class_identity"]: row["query_count"]
                                      for row in tapes.bindings},
        "queried_label_reads": budget.queried_label_reads,
        "query_response_chain_sha256_by_trajectory": {
            row["trajectory_id"]: row["query_response_sha256"]
            for row in trajectory_results},
        "unqueried_labels_decoded": False,
        "confirmation_classes_accessed": False,
        "masks_accessed": False,
    }
    checkpoint_verification = {
        "status": "PASS_SERIALIZED_PRIVATE_CHECKPOINT_RECONSTRUCTION_AND_RESUME",
        "phase_a": phase_a_checkpoint_identity,
        "phase_a_private_checkpoint_sha256": phase_checkpoint_sha,
        "resumed_prefix_event_chain_sha256": phase_a_result.event_chain_sha256,
        "resumed_prefix_response_chain_sha256": phase_a_result.query_response_sha256,
        "full_audiojack_seed0_event_chain_sha256": full_result.event_chain_sha256,
        "latest_private_checkpoint_sha256": final_checkpoint_sha,
        "private_checkpoint_payload_in_audit": False,
    }
    native_scorer_manifest = {
        **native_identity,
        "model_checkpoint_sha256": environment["model_checkpoint_sha256"],
        "dinov2_source_tree_sha256": environment["dinov2_source_tree_sha256"],
        "native_score_bundle": "(historical image, full DINOv2 features)",
        "static_map_path": "existing NativeScoreService -> native_bridge.native_score",
        "anchor_score_path": "P1.2 reusable exact Native clone -> native_bridge.native_score",
        "native_scorer_initialized": True,
        "greedy_writer_source": "unchanged SharedNativeGreedyWriter",
        "config_sha256": SCORER_CONFIG_SHA256,
        "native_scorer_dataclass_sha256": DEFAULT_PROFILE.scorer.sha256,
    }
    audit_payloads = {
        "exact_native_memory_parity.json": exact_parity,
        "dense_anchor_evidence_manifest.json": evidence_manifest,
        "queried_feedback_provenance.json": queried_provenance,
        "native_scorer_identity.json": native_scorer_manifest,
        "compute_accounting.json": compute_accounting,
        "checkpoint_resume_verification.json": checkpoint_verification,
    }
    for name, value in audit_payloads.items():
        _atomic_json(audit / name, value)
    gate = {
        "schema": "stage1a.p12_gate_matrix.v2",
        "decision": "P12_EXACT_PARITY_AND_EVIDENCE_PASS",
        "branch": environment["branch"], "base_head": environment["head"],
        "query_tape_binding_sha256": tapes.binding_sha256,
        "gates": [
            {"id": "P12-STANDING-AUTH", "status": "PASS_STANDING_USER_INSTRUCTION",
             "evidence": "Runtime authorization context bound to all four verified tapes and frozen quotas; no approval record fabricated."},
            {"id": "P12-QUERY-TAPE", "status": "PASS_QA_TS_V1_3PCT_SEED37", "evidence": tapes.binding_sha256},
            {"id": "P12-LABEL-FIREWALL", "status": "PASS_QUERIED_POSITIONS_ONLY", "evidence": f"{budget.queried_label_reads} requested and decoded; no non-tape position decoded."},
            {"id": "P12-PHASE-A", "status": "PASS_EXACT_FIRST_UPDATE_AND_ANCHOR", "evidence": phase_a_checkpoint_identity},
            {"id": "P12-EXACT-NATIVE-MEMORY-PARITY", "status": exact_parity["status"], "evidence": {"trajectories": 20, "events": exact_event_checks}},
            {"id": "P12-DENSE-ANCHOR-EVIDENCE", "status": evidence_manifest["status"], "evidence": {"snapshots": evidence_total, "native_anchor_score_calls": budget.anchor_score_calls}},
            {"id": "P12-HISTORICAL-C64-LINKAGE", "status": c64_linkage["status"], "evidence": "Stored vector/transport linkage only; no new C64 score."},
            {"id": "P12-CHECKPOINT-RESUME", "status": checkpoint_verification["status"], "evidence": checkpoint_verification},
            {"id": "P12-COMPUTE-CEILINGS", "status": "PASS_WITHIN_FROZEN_QUOTAS", "evidence": compute_accounting},
            {"id": "P12-STAGE1A-PERFORMANCE", "status": "NOT_RUN_PROHIBITED", "evidence": "No controller comparison or BA/AUROC/FPR/FNR calculation."},
            {"id": "P12-CONFIRMATION-ISOLATION", "status": "PASS_NO_MINT_PCB_MASKS", "evidence": "Only four development classes and queried labels were accessed."},
            {"id": "P12-HISTORICAL-G2", "status": "BLOCKED_UNCHANGED", "evidence": "Original G2 remains BLOCKED; independent certificate scope unchanged."},
        ],
    }
    _atomic_json(audit / "p12_gate_matrix.json", gate)
    _atomic_json(audit / "source_provenance.json", {
        "schema": "stage1a.p12_source_provenance.v1",
        "implementation_source_fingerprints": source_identity,
        "current_source_fingerprint_sha256": source_fingerprint,
        "historical_artifact_provenance": historical,
        "query_tape_source_artifact_sha256": dict(tapes.source_artifact_sha256),
        "model_and_scorer": native_scorer_manifest,
    })
    report = f"""# Stage 1-A P1.2 bounded Native replay and dense evidence

**Decision: `P12_EXACT_PARITY_AND_EVIDENCE_PASS`**

**Branch:** `{environment['branch']}`<br>
**HEAD at execution:** `{environment['head']}`<br>
**Q-A-TS binding:** `{tapes.binding_sha256}`

## Authorization and boundaries

Execution used the researcher's explicit standing authorization in the active task. No separate approval record, signature, timestamp, or synthetic approval evidence was created. The four Q-A-TS-v1 tapes, 3% budget, seed 37, and global compute ceilings were verified before image or label access. The label reader decoded `anomaly_class` only at the 459 precommitted query positions. It did not decode labels at unqueried positions; it did not open test masks, Mint, or PCB.

## Native replay and evidence

The audiojack/seed-0 smoke reached the first actual memory content update, checked the historical static M0 map scalar, writer/source map state and linked C64 transition, scored a real queried anchor, and passed after a serialized private checkpoint was reconstructed and resumed. All 20 class/seed trajectories then passed exact event-by-event memory, source attribution, writer and C64 identity checks. Dense evidence snapshots include the 64 reused protected C64 scores and all eligible queried-anchor scores. Queried-normal scores use actual all-layer source-aware LSO; queried defects use current memory and are never written.

Historical C64 vectors were checked through Stage 0.5/Stage 0.75-A artifact linkage. No new independent Native C64 scoring was performed. Private feature, label-cache, evidence-score and checkpoint payloads remain only under `{output.relative_to(ROOT).as_posix()}` and are not included in this audit package.

## Reproducibility and scope

Measured resource counts and exact trajectory/event digests are in `compute_accounting.json` and `exact_native_memory_parity.json`. The Native scorer uses the unchanged v22/VisionAD scoring path and locked configuration. Stage 1-A controller performance comparison, BA, AUROC, FPR, FNR, Stage 1-B, Stage 2, and confirmation-class analysis were not run. Original historical G2 remains BLOCKED.

The result establishes P1.2 engineering replay/evidence gates only. It does not establish empirical detector superiority, online risk guarantees, or permission to begin Stage 1-A evaluation.
"""
    (audit / "STAGE1A_P12_NATIVE_REPLAY_REPORT.md").write_text(report, encoding="utf-8")

    # The private checkpoint and evidence traces are not committed. Their
    # hashes are recorded for local completeness; audit files are compact.
    audit_checks = {path.name: _sha_file(path) for path in sorted(audit.iterdir())
                    if path.is_file() and path.name != "checksums.json"}
    _atomic_json(audit / "checksums.json", {
        "algorithm": "SHA-256", "files": audit_checks,
        "private_latest_checkpoint_sha256": final_checkpoint_sha,
        "private_evidence_hashes": evidence_manifest["private_evidence_files"],
    })
    _atomic_json(output / "complete.json", {
        "schema": "stage1a.p12_private_completion.v1",
        "decision": "P12_EXACT_PARITY_AND_EVIDENCE_PASS",
        "trajectory_count": 20, "event_count": exact_event_checks,
        "query_tape_binding_sha256": tapes.binding_sha256,
        "audit_checksums_sha256": _sha_file(audit / "checksums.json"),
        "private_checkpoint_sha256": final_checkpoint_sha,
        "stage1a_performance_evaluation": False,
        "confirmation_access": False,
    })
    (output / "INCOMPLETE").unlink(missing_ok=True)
    return {
        "decision": "P12_EXACT_PARITY_AND_EVIDENCE_PASS",
        "output": str(output), "audit": str(audit),
        "trajectory_count": len(trajectory_results),
        "event_count": exact_event_checks,
        "compute_accounting": compute_accounting,
        "tape_binding_sha256": tapes.binding_sha256,
        "audit_checksums_sha256": _sha_file(audit / "checksums.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--json-root", type=Path, default=DEFAULT_JSON_ROOT)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--audit", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.output is None or args.audit is None:
        for index in range(1, 100):
            suffix = f"_standing_authorization_{index:02d}_20261030"
            output = ROOT / "outputs/calibration" / ("stage1a_p12_native_replay_v1" + suffix)
            audit = ROOT / "docs/calibration/audit" / ("stage1a_p12_native_replay_v1" + suffix)
            if not output.exists() and not audit.exists():
                args.output = output if args.output is None else args.output
                args.audit = audit if args.audit is None else args.audit
                break
        else:
            raise SystemExit("No free Stage 1-A P1.2 immutable namespace")
    execution_source_sha256 = _capture_execution_source_sha256()
    run_started = time.perf_counter()
    try:
        result = _run_all(args.output.resolve(), args.audit.resolve(),
                          args.dataset_root.resolve(), args.json_root.resolve(),
                          args.chunk_size)
    except Exception as exc:
        print(f"P1.2_STOP {type(exc).__name__}: {exc}", flush=True)
        try:
            written = _write_blocked_audit(
                args.output.resolve(), args.audit.resolve(), exc,
                elapsed_seconds=time.perf_counter() - run_started,
                execution_source_sha256=execution_source_sha256)
            if written:
                print(f"P1.2_BLOCKED_AUDIT output={args.output.resolve()} audit={args.audit.resolve()}",
                      flush=True)
            else:
                print("P1.2_BLOCKED_AUDIT_NOT_WRITTEN namespace already exists or is outside repository",
                      flush=True)
        except Exception as audit_error:
            print(f"P1.2_BLOCKED_AUDIT_WRITE_FAILED {type(audit_error).__name__}: {audit_error}",
                  flush=True)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
