"""Bounded R1-S2 native C64 score-parity replay.

This isolated engineering check replays only the four frozen switch supports
and their 64 additional normal C64 references.  It never opens label JSON,
stream images, masks, or confirmation data.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
PROFILE_ID = "r1_s2_c64_score_parity_v1"
IDENTITY_CONTRACT = "r1-s2.source-bound.v1"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{4,63}\Z")
STAGE05 = ROOT / "outputs/calibration/stage05_exploratory_20261029"
S1_RUN_ID = "supportcheck_20261031_a1"
S1_MANIFEST = ROOT / (
    "outputs/calibration/r1_digest_contract_v1_support_only_"
    "supportcheck_20261031_a1/run_manifest.json")
FAILED_R1_DIR = ROOT / "outputs/calibration/r1_switch_s0_512_engineering_20261031"
FAILED_R1_REPORT = ROOT / (
    "docs/calibration/audit/r1_switch_s0_512_engineering_20261031/"
    "R1_ENGINEERING_REPORT.md")
SUPPORT_DIGEST_EXPECTED = (
    "8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558")
HISTORICAL_TREE_HASH_CONTRACT = (
    "stage075a_runner._tree_sha256.v1:sorted-relative-python-path-length+sha256")
CURRENT_D1_TREE_HASH_CONTRACT = (
    "r1_matched_correction_pilot._hub_source_sha256.v1:sha256-json-path-hash-list")
SCORE_ATOL = 1e-5
SCORE_RTOL = 1e-4
RELATIVE_ERROR_DENOMINATOR_FLOOR = 1e-12
MAX_GPU_SECONDS = 1200
MAX_OUTPUT_BYTES = 100 * 1024 * 1024
OUTPUT_PREFIX = "r1_s2_c64_parity_switch_s0_20261031_"
IMMUTABLE_BASELINE_PATHS = (
    "outputs/calibration/r1_digest_contract_v1_support_only_"
    "supportcheck_20261031_a1/run_manifest.json",
    "outputs/calibration/r1_s1_launcher_supportcheck_20261031_a1/stdout.log",
    "outputs/calibration/r1_s1_launcher_supportcheck_20261031_a1/stderr.log",
    "outputs/calibration/r1_switch_s0_512_engineering_20261031/run_manifest.json",
    "outputs/calibration/r1_switch_s0_512_engineering_20261031/attempt_history.jsonl",
    "docs/calibration/audit/r1_switch_s0_512_engineering_20261031/"
    "R1_ENGINEERING_REPORT.md",
)
CORE_SCORER_SOURCES = (
    "vmb_visionad_new.py",
    "test_phase3_AIF_simplified_v22.py",
    "experiments/p0_protocol_audit.py",
    "experiments/calibration/native.py",
)
FINGERPRINT_SOURCES = CORE_SCORER_SOURCES + (
    "experiments/calibration/profiles.py",
    "experiments/calibration/stage05.py",
    "experiments/calibration/stage05_runner.py",
    "experiments/calibration/stage075a_runner.py",
    "experiments/calibration/r1_matched_correction_pilot.py",
    "experiments/calibration/r1_s2_c64_score_parity.py",
)


class S2Error(RuntimeError):
    status = "INCOMPLETE"


class SourceIdentityError(S2Error):
    status = "BLOCKED_SOURCE_IDENTITY"


class InputIntegrityError(S2Error):
    status = "BLOCKED_INPUT"


class FixedBankMismatch(S2Error):
    status = "BLOCKED_FIXED_BANK"


class ScoreParityFailure(S2Error):
    status = "C64_PARITY_FAILED"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _opaque_digest(domain: str, values: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return _sha256_bytes(domain.encode("utf-8") + b"\0" + payload)


def ordered_identity_sha256(values: Sequence[str]) -> str:
    """Order-sensitive fingerprint; the historical opaque digest is set-like."""
    digest = hashlib.sha256(b"r1-s2.ordered-image-identities.v1\0")
    digest.update(len(values).to_bytes(8, "big"))
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def opaque_sample_identity(position: int, relative_id: str) -> str:
    payload = (b"r1-s2-c64-sample.v1\0" + str(position).encode("ascii") +
               b"\0" + relative_id.encode("utf-8"))
    return _sha256_bytes(payload)


def validate_execution_identity(*, branch: str, head: str,
                                 expected_source_commit: str,
                                 profile_id: str, run_id: str,
                                 authorization_id: str) -> dict[str, str]:
    if branch != EXPECTED_BRANCH:
        raise SourceIdentityError(
            f"Required branch {EXPECTED_BRANCH}; found {branch}")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", expected_source_commit):
        raise SourceIdentityError("Expected source commit must be a full Git SHA")
    if head.lower() != expected_source_commit.lower():
        raise SourceIdentityError(
            f"Source revision mismatch: expected {expected_source_commit}; found {head}")
    if profile_id != PROFILE_ID:
        raise SourceIdentityError(
            f"Incompatible R1-S2 profile: expected {PROFILE_ID}; found {profile_id}")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise InputIntegrityError("Run ID has an invalid or unsafe format")
    if not authorization_id.strip():
        raise InputIntegrityError("An authorization reference is required")
    return {
        "identity_contract": IDENTITY_CONTRACT,
        "branch": branch,
        "source_commit": head.lower(),
        "profile_id": profile_id,
        "run_id": run_id,
        "authorization_id": authorization_id,
    }


def validate_c64_metadata(*, frozen_info: Mapping[str, Any],
                          score_row: Mapping[str, Any],
                          historical_scores: Sequence[float],
                          receipt_identity_sha256: str
                          ) -> dict[str, Any]:
    """Validate frozen IDs, official groups, order and the C64 score vector."""
    support = frozen_info.get("supports", {}).get("0")
    c64 = frozen_info.get("c64")
    if not isinstance(support, Mapping) or not isinstance(c64, Mapping):
        raise InputIntegrityError("Frozen support/C64 metadata is missing")
    support_images = list(support.get("images", []))
    support_groups = list(support.get("groups", []))
    c64_images = list(c64.get("images", []))
    c64_groups = list(c64.get("groups", []))
    if (len(support_images) != 4 or len(support_groups) != 4 or
            len(set(support_images)) != 4 or len(set(support_groups)) != 4):
        raise InputIntegrityError("Frozen support order is not exactly four unique images/groups")
    if (len(c64_images) != 64 or len(c64_groups) != 64 or
            len(set(c64_images)) != 64 or len(set(c64_groups)) != 64):
        raise InputIntegrityError("Frozen C64 order is not exactly 64 unique images/groups")
    if (any(not str(image).replace("\\", "/").rsplit("/", 1)[-1].startswith(group + "_")
            for image, group in zip(support_images, support_groups)) or
            any(not str(image).replace("\\", "/").rsplit("/", 1)[-1].startswith(group + "_")
                for image, group in zip(c64_images, c64_groups))):
        raise InputIntegrityError("Frozen image/group order alignment is invalid")
    if set(support_groups) & set(c64_groups):
        raise InputIntegrityError("Frozen four supports overlap protected C64 groups")
    if support.get("c64_disjoint") is not True or frozen_info.get(
            "support_c64_disjoint") is not True:
        raise InputIntegrityError("Frozen support/C64 disjointness receipt failed")
    if (support.get("image_sha256") != _opaque_digest("support-images", support_images) or
            support.get("group_sha256") != _opaque_digest("support-groups", support_groups)):
        raise InputIntegrityError("Frozen support identity receipt does not match its ordered records")
    c64_identity = _opaque_digest("c64-images", c64_images)
    c64_group_identity = _opaque_digest("c64-groups", c64_groups)
    if c64.get("image_sha256") != c64_identity:
        raise InputIntegrityError("Frozen C64 image identity receipt is invalid")
    if c64.get("group_sha256") != c64_group_identity:
        raise InputIntegrityError("Frozen C64 official source-group receipt is invalid")
    if receipt_identity_sha256 != c64_identity:
        raise InputIntegrityError("Historical C64 score vector identity differs from frozen C64")
    scores = np.asarray(historical_scores, dtype=np.float64)
    if scores.shape != (64,) or not np.isfinite(scores).all():
        raise InputIntegrityError("Historical C64 score vector must contain 64 finite scores")
    if (int(score_row.get("c64_score_count", -1)) != 64 or
            score_row.get("c64_group_sha256") != c64_group_identity or
            score_row.get("fixed_support_sha256") != SUPPORT_DIGEST_EXPECTED or
            score_row.get("native_parity", {}).get("passed") is not True):
        raise InputIntegrityError("Historical C64 score receipt is incomplete or incompatible")
    if int(c64.get("count", -1)) != 64:
        raise InputIntegrityError("Frozen C64 count does not equal 64")
    return {
        "support_count": 4,
        "support_group_sha256": support["group_sha256"],
        "support_image_set_sha256": support["image_sha256"],
        "support_order_sha256": ordered_identity_sha256(support_images),
        "c64_count": 64,
        "c64_group_sha256": c64_group_identity,
        "c64_image_set_sha256": c64_identity,
        "c64_order_sha256": ordered_identity_sha256(c64_images),
        "historical_score_count": int(scores.size),
        "historical_score_vector_sha256": _sha256_bytes(
            np.ascontiguousarray(scores, dtype=np.float64).tobytes(order="C")),
    }


def compare_score_vectors(historical: Sequence[float], rebuilt: Sequence[float], *,
                          expected_identity_sha256: str,
                          actual_identity_sha256: str,
                          expected_order_sha256: str,
                          actual_order_sha256: str,
                          ordered_c64_images: Sequence[str]) -> dict[str, Any]:
    if expected_identity_sha256 != actual_identity_sha256:
        raise InputIntegrityError("C64 image identity/order receipt changed before comparison")
    if expected_order_sha256 != actual_order_sha256:
        raise InputIntegrityError("C64 ordered sample identity differs before comparison")
    hist = np.asarray(historical, dtype=np.float64)
    new = np.asarray(rebuilt, dtype=np.float64)
    if hist.shape != (64,) or new.shape != (64,) or len(ordered_c64_images) != 64:
        raise InputIntegrityError("Score comparison requires exactly 64 ordered pairs")
    finite = bool(np.isfinite(hist).all() and np.isfinite(new).all())
    if not finite:
        raise ScoreParityFailure("A nonfinite C64 score was produced")
    errors = np.abs(new - hist)
    tolerance = SCORE_ATOL + SCORE_RTOL * np.abs(hist)
    within = errors <= tolerance
    max_idx = int(np.argmax(errors))
    hist_bytes = np.ascontiguousarray(hist, dtype=np.float64).tobytes(order="C")
    new_bytes = np.ascontiguousarray(new, dtype=np.float64).tobytes(order="C")
    return {
        "paired_scores": 64,
        "max_absolute_error": float(errors.max()),
        "mean_absolute_error": float(errors.mean()),
        "root_mean_squared_error": float(np.sqrt(np.mean(np.square(new - hist)))),
        "relative_error_denominator_floor": RELATIVE_ERROR_DENOMINATOR_FLOOR,
        "maximum_relative_error": float(np.max(
            errors / np.maximum(np.abs(hist), RELATIVE_ERROR_DENOMINATOR_FLOOR))),
        "exact_elementwise_equality_count": int(np.count_nonzero(new == hist)),
        "numerical_tolerance_match_count": int(np.count_nonzero(within)),
        "tolerance_atol": SCORE_ATOL,
        "tolerance_rtol": SCORE_RTOL,
        "BITWISE_SCORE_PARITY": hist_bytes == new_bytes,
        "NUMERICAL_SCORE_PARITY": bool(within.all()),
        "SCORE_PARITY_FAILED": bool(not within.all()),
        "all_values_finite": finite,
        "maximum_error_position_zero_based": max_idx,
        "maximum_error_opaque_sample_identity_sha256": opaque_sample_identity(
            max_idx, str(ordered_c64_images[max_idx])),
        "historical_score_vector_sha256": _sha256_bytes(hist_bytes),
        "rebuilt_score_vector_sha256": _sha256_bytes(new_bytes),
    }


def _git_identity(expected_source_commit: str, profile_id: str,
                  run_id: str, authorization_id: str) -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()

    identity = validate_execution_identity(
        branch=git("branch", "--show-current"),
        head=git("rev-parse", "HEAD"),
        expected_source_commit=expected_source_commit,
        profile_id=profile_id, run_id=run_id,
        authorization_id=authorization_id)
    tracked = git("status", "--porcelain", "--untracked-files=no")
    if tracked:
        raise SourceIdentityError("GPU replay requires a clean tracked worktree")
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    critical = set(CORE_SCORER_SOURCES)
    shadowing = [path for path in untracked if path in critical or
                 (path.endswith(".py") and
                  (path.startswith("experiments/calibration/") or
                   path.startswith("experiments/risk_safe/")))]
    if shadowing:
        raise SourceIdentityError(
            "Untracked Python source can shadow bound imports: " +
            ", ".join(sorted(shadowing)[:8]))
    return identity


def _historical_tree_sha256(path: Path) -> str:
    """Match stage075a_runner._tree_sha256 exactly; D1 used another encoding."""
    digest = hashlib.sha256()
    files = sorted((item for item in path.rglob("*.py") if item.is_file()),
                   key=lambda item: item.relative_to(path).as_posix())
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _protected_artifact_hashes() -> dict[str, str]:
    result = {}
    for relative in IMMUTABLE_BASELINE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise InputIntegrityError(f"Protected prior artifact is missing: {relative}")
        result[relative] = sha256_file(path)
    return result


def _resolve_image(dataset_root: Path, prefix: str, relative_id: str) -> Path:
    prefix_parts = [part for part in prefix.replace("\\", "/").split("/") if part]
    image_parts = [part for part in relative_id.replace("\\", "/").split("/") if part]
    return dataset_root / "images" / Path(*prefix_parts, *image_parts)


def collect_preflight(*, expected_source_commit: str, profile_id: str,
                      run_id: str, authorization_id: str,
                      dataset_root: Path,
                      allow_existing_output_dir: bool = False) -> dict[str, Any]:
    """CPU-only identity gate. It never decodes images or reads label JSON."""
    identity = _git_identity(expected_source_commit, profile_id,
                             run_id, authorization_id)
    output_dir = ROOT / "outputs/calibration" / f"{OUTPUT_PREFIX}{run_id}"
    if output_dir.exists() and not allow_existing_output_dir:
        raise InputIntegrityError(
            f"Refusing to reuse existing R1-S2 output namespace: {output_dir.name}")

    stage_run_path = STAGE05 / "run_manifest.json"
    score_manifest_path = STAGE05 / "score_tape_manifest.json"
    checksums_path = STAGE05 / "checksums.json"
    frozen_path = STAGE05 / "frozen_inputs.json"
    for path in (stage_run_path, score_manifest_path, checksums_path, frozen_path):
        if not path.is_file():
            raise InputIntegrityError(f"Required Stage 0.5 receipt is missing: {path.name}")
    stage_run = json.loads(stage_run_path.read_text(encoding="utf-8"))
    score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))["files"]
    if stage_run.get("score_tape_manifest_sha256") != sha256_file(score_manifest_path):
        raise InputIntegrityError("Stage 0.5 score manifest checksum differs from run receipt")
    if checksums.get("frozen_inputs.json") != sha256_file(frozen_path):
        raise InputIntegrityError("Frozen input manifest checksum is invalid")
    if score_manifest.get("status") != "PASS":
        raise InputIntegrityError("Historical native C64 score manifest is not PASS")

    scorer_config = score_manifest.get("scorer_config")
    profile_record = score_manifest.get("scorer_config_sha256")
    from experiments.calibration.profiles import DEFAULT_PROFILE, NativeScorerConfig, canonical_json
    if (canonical_json(asdict(DEFAULT_PROFILE.scorer)) != canonical_json(scorer_config) or
            DEFAULT_PROFILE.scientific_sha256 != profile_record):
        raise SourceIdentityError("Current locked scorer profile differs from Stage 0.5")
    if score_manifest.get("dynamic_memory") != "empty_and_disabled":
        raise SourceIdentityError("Historical C64 profile did not disable dynamic memory")
    if score_manifest.get("controller_or_test_label_access") is not False:
        raise SourceIdentityError("Historical C64 score profile has incompatible label access")

    current_sources = {relative: sha256_file(ROOT / relative)
                       for relative in FINGERPRINT_SOURCES}
    historical_score_sources = score_manifest.get("scoring_source_hashes", {})
    missing = [relative for relative in CORE_SCORER_SOURCES
               if not historical_score_sources.get(relative)]
    if missing:
        raise SourceIdentityError(
            "Historical score manifest lacks core scorer fingerprints: " +
            ", ".join(missing))
    core_comparison = {
        relative: {
            "historical_score_time_sha256": historical_score_sources[relative],
            "current_sha256": current_sources[relative],
            "matches": current_sources[relative] == historical_score_sources[relative],
        }
        for relative in CORE_SCORER_SOURCES
    }
    if not all(row["matches"] for row in core_comparison.values()):
        raise SourceIdentityError("A core scorer source differs from historical C64 scoring")

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen_info = frozen.get("classes", {}).get("switch")
    if not isinstance(frozen_info, Mapping):
        raise InputIntegrityError("Frozen switch support/C64 manifest is missing")
    if frozen_info.get("labels_or_masks_returned_by_parser") not in (None, False):
        raise InputIntegrityError("Frozen identity parser unexpectedly returned labels or masks")
    support = frozen_info.get("supports", {}).get("0", {})
    c64 = frozen_info.get("c64", {})
    if (len(support.get("images", [])) != 4 or len(c64.get("images", [])) != 64 or
            int(c64.get("count", -1)) != 64):
        raise InputIntegrityError("Frozen support/C64 counts differ from the authorized profile")
    if frozen.get("stage1_or_stage2_run") is not False:
        raise InputIntegrityError("Frozen calibration receipt is not Stage-0-only")
    if c64.get("selection_seed") != 190021:
        raise InputIntegrityError("Frozen C64 selection identity differs from Stage 0.5")

    score_row = score_manifest.get("classes", {}).get("switch", {}).get(
        "score_tapes", {}).get("0")
    if not isinstance(score_row, Mapping):
        raise InputIntegrityError("Frozen switch/support-0 C64 score receipt is missing")
    if score_row.get("fixed_support_sha256") != SUPPORT_DIGEST_EXPECTED:
        raise InputIntegrityError("Historical C64 receipt has an incompatible fixed-bank digest")
    score_path = STAGE05 / str(score_row.get("file", ""))
    try:
        score_path.resolve().relative_to(STAGE05.resolve())
    except ValueError as exc:
        raise InputIntegrityError("C64 score receipt path escapes Stage 0.5 outputs") from exc
    if not score_path.is_file():
        raise InputIntegrityError("Historical C64 score array is missing")
    receipt_sha = sha256_file(score_path)
    if (receipt_sha != score_row.get("sha256") or
            receipt_sha != checksums.get(score_row.get("file"))):
        raise InputIntegrityError("Historical C64 score receipt checksum is invalid")
    with np.load(score_path, allow_pickle=False) as archive:
        historical_scores = np.asarray(archive["c64_scores"], dtype=np.float64).copy()
        receipt_identity = str(archive["c64_identity_sha256"].item())
    metadata = validate_c64_metadata(
        frozen_info=frozen_info, score_row=score_row,
        historical_scores=historical_scores,
        receipt_identity_sha256=receipt_identity)

    expected_checkpoint = score_manifest.get("model_provenance", {}).get(
        "checkpoint_sha256")
    historical_tree_sha = score_manifest.get("model_provenance", {}).get(
        "hub_source_tree_sha256")
    expected_python = stage_run.get("environment", {}).get("python_version")
    expected_torch = stage_run.get("environment", {}).get("torch_version")
    expected_python_exe_sha = stage_run.get("environment", {}).get(
        "python_executable_sha256")
    if not expected_checkpoint or not historical_tree_sha:
        raise SourceIdentityError("Historical DINO checkpoint/source provenance is incomplete")

    # Importing torch here is CPU-only; no CUDA property or model is queried.
    import torch
    hub_dir = Path(torch.hub.get_dir())
    checkpoint = hub_dir / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
    hub_source = hub_dir / "facebookresearch_dinov2_main"
    if not checkpoint.is_file() or not hub_source.is_dir():
        raise InputIntegrityError("Pinned local DINO checkpoint or source tree is missing")
    checkpoint_sha = sha256_file(checkpoint)
    comparable_tree_sha = _historical_tree_sha256(hub_source)
    if checkpoint_sha != expected_checkpoint:
        raise SourceIdentityError("DINO checkpoint SHA differs from Stage 0.5")
    if comparable_tree_sha != historical_tree_sha:
        raise SourceIdentityError(
            "DINO source tree differs under the historical comparable hash contract")
    if torch.__version__ != expected_torch or platform.python_version() != expected_python:
        raise SourceIdentityError("Python/Torch version differs from Stage 0.5 runtime")
    python_executable_sha = sha256_file(Path(sys.executable))

    if not S1_MANIFEST.is_file() or not (FAILED_R1_DIR / "run_manifest.json").is_file():
        raise InputIntegrityError("R1-S1 or historical failed-R1 manifest is missing")
    s1 = json.loads(S1_MANIFEST.read_text(encoding="utf-8"))
    if (s1.get("run_id") != S1_RUN_ID or s1.get("status") != "SUPPORT_ONLY_VERIFIED" or
            s1.get("source_commit") != "cf1bf2b72381633663e68050e84d91f88bf4b402" or
            s1.get("fixed_support_sha256") != SUPPORT_DIGEST_EXPECTED or
            s1.get("fixed_support_sha256_expected") != SUPPORT_DIGEST_EXPECTED or
            s1.get("fixed_bank_digest_validation", {}).get("actual_structured_algorithm") !=
            "experiments.calibration.native._fixed_bank_digest.v1:path-shape-dtype-bytes" or
            not s1.get("fixed_bank_digest_validation", {}).get(
                "matches_expected_structured_digest") or
            s1.get("c64_rescored") is not False):
        raise InputIntegrityError("Recorded R1-S1 support-only verification is incompatible")
    failed_r1 = json.loads((FAILED_R1_DIR / "run_manifest.json").read_text(
        encoding="utf-8"))
    if (failed_r1.get("status") != "BLOCKED_INPUT" or
            failed_r1.get("events_processed") != 0 or
            failed_r1.get("queried_labels_decoded") != 0):
        raise InputIntegrityError("Historical failed R1 output identity is unexpected")

    prefix = str(frozen_info.get("prefix", ""))
    ordered_support_images = list(support["images"])
    ordered_c64_images = list(c64["images"])
    all_paths = [_resolve_image(dataset_root, prefix, relative)
                 for relative in ordered_support_images + ordered_c64_images]
    missing_paths = [str(path) for path in all_paths if not path.is_file()]
    if missing_paths:
        raise InputIntegrityError(
            f"{len(missing_paths)} authorized support/C64 reference paths are unavailable")
    if (not dataset_root.is_dir() or len(all_paths) != 68 or
            len({path.resolve() for path in all_paths}) != 68):
        raise InputIntegrityError("Authorized dataset/reference path identity is invalid")

    stage_score_sources = stage_run.get("source_hashes", {})
    orchestration_differences = {}
    for relative in ("experiments/calibration/stage05.py",
                     "experiments/calibration/stage05_runner.py"):
        historical = historical_score_sources.get(relative)
        published = stage_score_sources.get(relative)
        current = current_sources.get(relative)
        if historical != current:
            orchestration_differences[relative] = {
                "historical_score_time_sha256": historical,
                "published_run_manifest_sha256": published,
                "current_sha256": current,
            }
    source_history = stage_run.get("source_hash_history", [])
    orchestration_explanations = [
        str(entry.get("reason", "")) for entry in source_history
        if any(relative in entry.get("old", {}) or relative in entry.get("new", {})
               for relative in orchestration_differences)
    ]

    return {
        "identity": identity,
        "dataset_root": str(dataset_root.resolve()),
        "authorized_reference_paths_available": 68,
        "support_count": metadata["support_count"],
        "support_group_sha256": metadata["support_group_sha256"],
        "support_image_set_sha256": metadata["support_image_set_sha256"],
        "support_order_sha256": metadata["support_order_sha256"],
        "c64_count": metadata["c64_count"],
        "c64_group_sha256": metadata["c64_group_sha256"],
        "c64_image_set_sha256": metadata["c64_image_set_sha256"],
        "c64_order_sha256": metadata["c64_order_sha256"],
        "historical_score_count": metadata["historical_score_count"],
        "historical_score_vector_sha256": metadata["historical_score_vector_sha256"],
        "fixed_support_sha256_expected": SUPPORT_DIGEST_EXPECTED,
        "historical_c64_score_receipt_sha256": receipt_sha,
        "frozen_inputs_sha256": sha256_file(frozen_path),
        "score_manifest_sha256": sha256_file(score_manifest_path),
        "profile": {
            "profile_sha256": profile_record,
            "scorer_config": scorer_config,
            "matches_historical": True,
            "dynamic_memory": score_manifest["dynamic_memory"],
        },
        "source_fingerprints": {
            relative: {
                "current_sha256": current_sources[relative],
                "historical_score_time_sha256": historical_score_sources.get(relative),
                "core_scorer_match": (relative in CORE_SCORER_SOURCES and
                                       current_sources[relative] ==
                                       historical_score_sources.get(relative)),
            }
            for relative in FINGERPRINT_SOURCES
        },
        "core_scorer_sources_match": True,
        "orchestration_source_differences": orchestration_differences,
        "orchestration_source_history_explanations": orchestration_explanations,
        "dino": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_matches_historical": True,
            "hub_source_path": str(hub_source),
            "historical_tree_hash_contract": HISTORICAL_TREE_HASH_CONTRACT,
            "historical_tree_sha256": historical_tree_sha,
            "current_tree_sha256_historical_contract": comparable_tree_sha,
            "tree_matches_historical": True,
            "d1_different_hash_contract": CURRENT_D1_TREE_HASH_CONTRACT,
            "d1_current_tree_sha256": s1.get("hub_source_tree_sha256"),
        },
        "runtime": {
            "python_version_current": platform.python_version(),
            "python_version_historical": expected_python,
            "torch_version_current": torch.__version__,
            "torch_version_historical": expected_torch,
            "cuda_version_current": torch.version.cuda,
            "python_executable": sys.executable,
            "python_executable_sha256_current": python_executable_sha,
            "python_executable_sha256_historical": expected_python_exe_sha,
            "python_executable_sha256_matches":
                python_executable_sha == expected_python_exe_sha,
            "historical_gpu_device_name": stage_run.get("environment", {}).get(
                "cuda_device_name"),
        },
        "r1_s1": {
            "run_id": S1_RUN_ID,
            "status": s1["status"],
            "source_commit": s1["source_commit"],
            "fixed_support_sha256": s1["fixed_support_sha256"],
            "c64_rescored": False,
        },
        "historical_failed_r1": {
            "status": failed_r1["status"],
            "events_processed": failed_r1["events_processed"],
            "protected_artifact_sha256": _protected_artifact_hashes(),
        },
    }


class ManifestWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.path = output_dir / "run_manifest.json"
        if self.path.exists():
            raise InputIntegrityError("Refusing to overwrite an existing run manifest")
        self.data: dict[str, Any] = {}

    def update(self, values: Mapping[str, Any]) -> None:
        self.data.update(values)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(_canonical_json(self.data) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)


def _safe_cuda_metrics(torch_module: Any) -> dict[str, int | None]:
    try:
        torch_module.cuda.synchronize()
        return {
            "peak_cuda_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
        }
    except Exception:
        return {"peak_cuda_allocated_bytes": None,
                "peak_cuda_reserved_bytes": None}


def _peak_cpu_rss_bytes() -> int | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize) if ok else None
    except Exception:
        return None


def _write_score_progress(path: Path, position: int, relative_id: str,
                          score: float) -> None:
    row = {
        "position_zero_based": position,
        "opaque_sample_identity_sha256": opaque_sample_identity(position, relative_id),
        "score": float(score),
    }
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical_json(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _execution_profile(config: Any) -> dict[str, Any]:
    return {
        "profile_id": PROFILE_ID,
        "device": config.device,
        "gpu_seconds_limit": MAX_GPU_SECONDS,
        "output_bytes_limit": MAX_OUTPUT_BYTES,
        "score_atol": SCORE_ATOL,
        "score_rtol": SCORE_RTOL,
        "relative_error_denominator_floor": RELATIVE_ERROR_DENOMINATOR_FLOOR,
        "support_count": 4,
        "c64_count": 64,
        "dynamic_greedy_memory": "empty_and_disabled",
        "correction_bank": "disabled_and_empty",
        "defect_bank": "disabled",
    }


def execute_gpu_replay(*, preflight: Mapping[str, Any], output_dir: Path,
                       manifest: ManifestWriter, dataset_root: Path,
                       expected_source_commit: str, profile_id: str,
                       run_id: str, authorization_id: str) -> dict[str, Any]:
    import torch
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration.controllers import theta0_from_c64
    from experiments.calibration.native import _fixed_bank_digest, isolate_native_bank_k256
    from experiments.calibration.profiles import DEFAULT_PROFILE, NativeScorerConfig
    from experiments.p0_protocol_audit import Config, require_sources
    from experiments.risk_safe.native_bridge import dynamic_codebooks

    started_wall = time.perf_counter()
    started_utc = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat()
    manifest.update({
        "status": "INCOMPLETE",
        "task": "R1-S2 bounded native C64 score parity",
        "identity_contract": IDENTITY_CONTRACT,
        "authorization_id": authorization_id,
        "run_id": run_id,
        "branch": preflight["identity"]["branch"],
        "source_commit": expected_source_commit,
        "source_commit_used_for_gpu": expected_source_commit,
        "profile": _execution_profile(SimpleNamespace(device="cuda")),
        "started_at_utc": started_utc,
        "preflight": dict(preflight),
        "images_decoded": 0,
        "query_labels_accessed": False,
        "stream_images_decoded": 0,
        "test_labels_or_masks_accessed": False,
        "confirmation_data_accessed": False,
        "greedy_writer_events": 0,
        "correction_updates": 0,
        "c64_scores_completed": 0,
        "dino_model_load_seconds": None,
        "support_feature_encoding_and_bank_build_seconds": None,
        "c64_encoding_seconds": None,
        "c64_scoring_seconds": None,
        "total_gpu_subprocess_wall_seconds": None,
        "actual_dino_feature_profile": None,
        "score_comparison": None,
        "threshold_compatibility": None,
        "resource_metrics": {
            "peak_cuda_allocated_bytes": None,
            "peak_cuda_reserved_bytes": None,
            "peak_cpu_rss_bytes": None,
            "unavailable_metrics_are_null_not_zero": True,
        },
    })

    cfg = Config(
        device="cuda", augmentation_seed=0, bank_seed=0,
        fixed_capacity=25000, fixed_per_image=5000, augmentations=4,
        include_identity=False, multiplier=2.5, feature_fuse="concat",
        topk_patches=32, image_aggregation="max", image_topk=64,
        image_quantile=0.95, knn=5)
    try:
        r1._validate_p0_profile(cfg, NativeScorerConfig())
        impl = require_sources()
        runtime_setup = r1._configure_historical_runtime(impl, torch)
    except Exception as exc:
        raise SourceIdentityError(
            f"Locked scorer or historical runtime initialization failed: {exc}") from exc
    if not torch.cuda.is_available():
        raise InputIntegrityError("Authorized CUDA device is unavailable")
    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    torch.cuda.reset_peak_memory_stats()
    manifest.update({
        "runtime_profile": {
            **runtime_setup,
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "python_executable_sha256": sha256_file(Path(sys.executable)),
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "cuda_device_name": device_name,
            "historical_cuda_device_name": preflight["runtime"][
                "historical_gpu_device_name"],
            "cuda_device_matches_historical": device_name == preflight["runtime"][
                "historical_gpu_device_name"],
        },
        "scorer_config_sha256": DEFAULT_PROFILE.scientific_sha256,
        "preprocess_profile": {
            "rgb": "PIL.convert(RGB)", "resize": [448, 448],
            "crop": [392, 392], "interpolation": "PIL.BICUBIC",
            "normalization": "v22 model mean/std", "feature_fuse": "concat",
            "view": "id", "historical_scoring_path":
                "p0.encode_query then p0.score_encoded",
        },
    })

    hub_dir = Path(torch.hub.get_dir())
    checkpoint = hub_dir / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != preflight["dino"]["checkpoint_sha256"]:
        raise SourceIdentityError("DINO checkpoint changed after CPU preflight")
    from experiments.calibration.r1_matched_correction_pilot import _model_state_sha256
    model_load_started = time.perf_counter()
    backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_load_started
    model_state_sha = _model_state_sha256(backbone.model)
    manifest.update({
        "dino_model_name": "dinov2_vitl14_reg4_pretrain",
        "dino_checkpoint_sha256": checkpoint_sha,
        "dino_source_tree_sha256_historical_contract":
            preflight["dino"]["current_tree_sha256_historical_contract"],
        "dino_model_state_sha256": model_state_sha,
        "dino_model_load_seconds": model_load_seconds,
        "image_decodes_before_c64_scores": 4,
        "images_decoded": 4,
    })

    from experiments.calibration.native import _fixed_bank_digest
    from experiments.p0_protocol_audit import (
        build_vmb, encode_query, encode_reference_features, score_encoded)
    frozen_path = STAGE05 / "frozen_inputs.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen_info = frozen["classes"]["switch"]
    support_rel = list(frozen_info["supports"]["0"]["images"])
    prefix = str(frozen_info["prefix"])
    support_paths = [str(_resolve_image(dataset_root, prefix, relative))
                     for relative in support_rel]
    support_started = time.perf_counter()
    support_features = encode_reference_features(impl, backbone, support_paths, cfg)
    native_bank = build_vmb(impl, support_features, support_paths, cfg)
    torch.cuda.synchronize()
    support_encode_build_seconds = time.perf_counter() - support_started
    p0_module = __import__("experiments.p0_protocol_audit",
                           fromlist=["fixed_bank_profile"])
    original_evidence = r1._fixed_bank_digest_evidence(
        native_bank, p0_module,
        SUPPORT_DIGEST_EXPECTED, _fixed_bank_digest)
    manifest.update({
        "fixed_bank_digest_validation": original_evidence,
        "fixed_support_sha256_expected": SUPPORT_DIGEST_EXPECTED,
        "fixed_support_sha256_actual": original_evidence["actual_structured_sha256"],
        "fixed_support_raw_prototype_sha256": original_evidence[
            "actual_raw_prototype_sha256"],
        "fixed_support_digest_algorithm": original_evidence["actual_structured_algorithm"],
        "fixed_support_raw_digest_algorithm": original_evidence[
            "actual_raw_prototype_algorithm"],
        "fixed_support_layer_profile": original_evidence["layer_profile"],
        "support_feature_encoding_and_bank_build_seconds": support_encode_build_seconds,
    })
    if not original_evidence["matches_expected_structured_digest"]:
        raise FixedBankMismatch(
            "Structured fixed-bank digest differs from the frozen Stage 0.5 receipt")

    isolation_started = time.perf_counter()
    bank, isolation_digest = isolate_native_bank_k256(native_bank)
    evidence = r1._fixed_bank_digest_evidence(
        bank, p0_module, SUPPORT_DIGEST_EXPECTED, _fixed_bank_digest)
    evidence["isolate_native_bank_k256_digest"] = isolation_digest
    torch.cuda.synchronize()
    bank_isolation_seconds = time.perf_counter() - isolation_started
    manifest.update({
        "fixed_bank_digest_validation": evidence,
        "fixed_support_sha256_actual": evidence["actual_structured_sha256"],
        "fixed_support_raw_prototype_sha256": evidence[
            "actual_raw_prototype_sha256"],
        "fixed_support_digest_algorithm": evidence["actual_structured_algorithm"],
        "fixed_support_raw_digest_algorithm": evidence[
            "actual_raw_prototype_algorithm"],
        "fixed_support_layer_profile": evidence["layer_profile"],
        "support_feature_encoding_and_bank_build_seconds": support_encode_build_seconds,
        "bank_isolation_and_digest_seconds": bank_isolation_seconds,
    })
    if not evidence["matches_expected_structured_digest"]:
        raise FixedBankMismatch(
            "K=256 isolated bank structured digest differs from the frozen receipt")

    dynamic = dynamic_codebooks(bank)
    if not dynamic or any(int(codebook.size()) != 0 for codebook in dynamic):
        raise SourceIdentityError("Native dynamic memory is not empty")
    if any(int(codebook.K) != 256 for codebook in dynamic):
        raise SourceIdentityError("Native dynamic memory does not have frozen K=256 capacity")
    bank.corr_cfg.enabled = False
    bank.use_defect_bank = False
    correction = bank.corr_cb["p0"]["id"][0]
    if int(correction.size()) != 0 or bool(bank.corr_cfg.enabled) or bool(
            bank.use_defect_bank):
        raise SourceIdentityError("Correction or defect memory is not disabled and empty")
    manifest.update({
        "dynamic_greedy_memory_empty": True,
        "dynamic_greedy_memory_enabled": False,
        "dynamic_codebook_count": len(dynamic),
        "dynamic_codebook_capacities": [int(codebook.K) for codebook in dynamic],
        "correction_bank_empty": True,
        "correction_bank_enabled": False,
        "defect_bank_enabled": False,
        "memory_writer_instantiated": False,
    })

    score_manifest = json.loads(
        (STAGE05 / "score_tape_manifest.json").read_text(encoding="utf-8"))
    historical_score_path = STAGE05 / score_manifest[
        "classes"]["switch"]["score_tapes"]["0"]["file"]
    with np.load(historical_score_path, allow_pickle=False) as archive:
        historical_scores = np.asarray(archive["c64_scores"], dtype=np.float64).copy()
        historical_identity = str(archive["c64_identity_sha256"].item())
    c64_rel = list(frozen_info["c64"]["images"])
    c64_paths = [_resolve_image(dataset_root, prefix, relative) for relative in c64_rel]
    current_order_sha256 = ordered_identity_sha256(c64_rel)
    if len(c64_paths) != 64 or any(not path.is_file() for path in c64_paths):
        raise InputIntegrityError("C64 reference order or path availability changed")
    if (_opaque_digest("c64-images", c64_rel) != historical_identity or
            current_order_sha256 != preflight["c64_order_sha256"]):
        raise InputIntegrityError("C64 identity changed after fixed-bank verification")

    progress_path = output_dir / "c64_score_progress.jsonl"
    if progress_path.exists():
        raise InputIntegrityError("Refusing to overwrite C64 score progress")
    rebuilt_scores: list[float] = []
    encode_seconds = 0.0
    score_seconds = 0.0
    first_feature_profile: list[dict[str, Any]] | None = None
    for position, path in enumerate(c64_paths):
        torch.cuda.synchronize()
        encode_started = time.perf_counter()
        image, features = encode_query(impl, backbone, str(path), cfg)
        torch.cuda.synchronize()
        encode_seconds += time.perf_counter() - encode_started
        current_profile = [
            {"shape": [int(dim) for dim in feature.shape],
             "dtype": str(feature.dtype)}
            for feature in features
        ]
        if first_feature_profile is None:
            first_feature_profile = current_profile
        elif current_profile != first_feature_profile:
            raise InputIntegrityError("DINO C64 feature shape/dtype changed within the frozen run")
        manifest.update({
            "images_decoded": 4 + position + 1,
            "c64_images_decoded": position + 1,
            "c64_encoding_seconds": encode_seconds,
            "actual_dino_feature_profile": first_feature_profile,
            "c64_order_sha256": ordered_identity_sha256(c64_rel),
        })

        torch.cuda.synchronize()
        score_started = time.perf_counter()
        score = float(score_encoded(impl, bank, image, features, cfg))
        torch.cuda.synchronize()
        score_seconds += time.perf_counter() - score_started
        rebuilt_scores.append(score)
        manifest.update({
            "c64_scores_completed": len(rebuilt_scores),
            "c64_encoding_seconds": encode_seconds,
            "c64_scoring_seconds": score_seconds,
            "actual_dino_feature_profile": first_feature_profile,
        })
        if not math.isfinite(score):
            with progress_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(_canonical_json({
                    "position_zero_based": position,
                    "opaque_sample_identity_sha256": opaque_sample_identity(
                        position, c64_rel[position]),
                    "score": None,
                    "score_repr": repr(score),
                    "all_values_finite": False,
                }) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            manifest.update({
                "status": "C64_PARITY_FAILED",
                "failure_reason": "Nonfinite C64 score; scoring stopped at first occurrence",
                "nonfinite_position_zero_based": position,
            })
            raise ScoreParityFailure("A nonfinite C64 score was produced")
        _write_score_progress(progress_path, position, c64_rel[position], score)

    rebuilt = np.asarray(rebuilt_scores, dtype=np.float64)
    comparison = compare_score_vectors(
        historical_scores, rebuilt,
        expected_identity_sha256=historical_identity,
        actual_identity_sha256=_opaque_digest("c64-images", c64_rel),
        expected_order_sha256=preflight["c64_order_sha256"],
        actual_order_sha256=current_order_sha256,
        ordered_c64_images=c64_rel)
    npz_path = output_dir / "c64_scores.npz"
    if npz_path.exists():
        raise InputIntegrityError("Refusing to overwrite rebuilt C64 scores")
    np.savez_compressed(
        npz_path,
        c64_scores=rebuilt,
        c64_identity_sha256=np.asarray(historical_identity),
        c64_order_sha256=np.asarray(ordered_identity_sha256(c64_rel)))

    theta_hist = float(theta0_from_c64(historical_scores.tolist()))
    theta_new = float(theta0_from_c64(rebuilt.tolist()))
    sigma_hist = max(float(historical_scores.std(ddof=1)), 0.02)
    sigma_new = max(float(rebuilt.std(ddof=1)), 0.02)
    theta_delta = theta_new - theta_hist
    sigma_delta = sigma_new - sigma_hist
    threshold_metrics = {
        "historical_theta0": theta_hist,
        "rebuilt_theta0": theta_new,
        "theta0_absolute_difference": abs(theta_delta),
        "theta0_signed_difference": theta_delta,
        "theta0_relative_difference": abs(theta_delta) / max(
            abs(theta_hist), RELATIVE_ERROR_DENOMINATOR_FLOOR),
        "historical_sigma0": sigma_hist,
        "rebuilt_sigma0": sigma_new,
        "sigma0_absolute_difference": abs(sigma_delta),
        "sigma0_signed_difference": sigma_delta,
        "sigma0_relative_difference": abs(sigma_delta) / max(
            abs(sigma_hist), RELATIVE_ERROR_DENOMINATOR_FLOOR),
        "threshold_policy": "theta0_from_c64 higher-q90; sigma0 sample std ddof=1 floored at 0.02",
        "diagnostic_only": True,
    }
    status = "C64_PARITY_PASS" if comparison["NUMERICAL_SCORE_PARITY"] else "C64_PARITY_FAILED"
    total_seconds = time.perf_counter() - started_wall
    manifest.update({
        "status": status,
        "c64_scores_completed": 64,
        "c64_encoding_seconds": encode_seconds,
        "c64_scoring_seconds": score_seconds,
        "support_feature_encoding_and_bank_build_seconds": support_encode_build_seconds,
        "total_gpu_subprocess_wall_seconds": total_seconds,
        "score_comparison": comparison,
        "threshold_compatibility": threshold_metrics,
        "rebuilt_score_array_sha256": sha256_file(npz_path),
        "rebuilt_score_array_bytes": npz_path.stat().st_size,
        "disk_footprint_bytes_before_final_manifest_update": sum(
            p.stat().st_size for p in output_dir.rglob("*") if p.is_file()),
        "resource_metrics": {
            **_safe_cuda_metrics(torch),
            "peak_cpu_rss_bytes": _peak_cpu_rss_bytes(),
            "unavailable_metrics_are_null_not_zero": True,
        },
        "finished_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
    })
    if status != "C64_PARITY_PASS":
        raise ScoreParityFailure("One or more C64 scores exceed the preregistered tolerance")
    return manifest.data


def _validate_launcher_output_dir(output_dir: Path, run_id: str) -> None:
    expected = (ROOT / "outputs/calibration" / f"{OUTPUT_PREFIX}{run_id}").resolve()
    if output_dir.resolve() != expected or not output_dir.is_dir():
        raise InputIntegrityError("Output directory is not the authorized new run namespace")
    if (output_dir / "run_manifest.json").exists() or (output_dir / "c64_scores.npz").exists():
        raise InputIntegrityError("Existing run output cannot be resumed or overwritten")
    if not (output_dir / "launcher_manifest.json").is_file():
        raise InputIntegrityError("Output namespace was not created by the bounded launcher")
    allowed = {"launcher_manifest.json", "preflight.json", "stdout.log", "stderr.log"}
    unexpected = [path.name for path in output_dir.iterdir()
                  if path.is_file() and path.name not in allowed]
    if unexpected:
        raise InputIntegrityError(
            "Output namespace contains unexpected prior artifacts: " +
            ", ".join(sorted(unexpected)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path(r"G:\dataset\realiad"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    if args.mode == "preflight":
        try:
            result = collect_preflight(
                expected_source_commit=args.expected_source_commit,
                profile_id=args.profile_id, run_id=args.run_id,
                authorization_id=args.authorization_id,
                dataset_root=args.dataset_root)
        except S2Error as exc:
            print(_canonical_json({"status": exc.status, "reason": str(exc)}))
            return 2
        except Exception as exc:
            print(_canonical_json({"status": "INCOMPLETE",
                                   "reason": f"{type(exc).__name__}: {exc}"}))
            return 2
        print(_canonical_json({"status": "PREFLIGHT_PASS", **result}))
        return 0

    if args.output_dir is None:
        parser.error("--output-dir is required for --mode run")
    output_dir = args.output_dir
    run_started_perf = time.perf_counter()
    try:
        _validate_launcher_output_dir(output_dir, args.run_id)
        manifest = ManifestWriter(output_dir)
        manifest.update({
            "schema": "r1_s2.c64_score_parity.v1",
            "status": "INCOMPLETE",
            "identity_contract": IDENTITY_CONTRACT,
            "run_id": args.run_id,
            "authorization_id": args.authorization_id,
            "source_commit": args.expected_source_commit,
            "source_commit_used_for_gpu": args.expected_source_commit,
            "started_at_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(),
            "query_labels_accessed": False,
            "stream_images_decoded": 0,
            "test_labels_or_masks_accessed": False,
            "confirmation_data_accessed": False,
        })
        preflight = collect_preflight(
            expected_source_commit=args.expected_source_commit,
            profile_id=args.profile_id, run_id=args.run_id,
            authorization_id=args.authorization_id,
            dataset_root=args.dataset_root, allow_existing_output_dir=True)
        result = execute_gpu_replay(
            preflight=preflight, output_dir=output_dir, manifest=manifest,
            dataset_root=args.dataset_root,
            expected_source_commit=args.expected_source_commit,
            profile_id=args.profile_id, run_id=args.run_id,
            authorization_id=args.authorization_id)
        manifest.update({
            "total_gpu_subprocess_wall_seconds": time.perf_counter() - run_started_perf,
            "finished_at_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(),
        })
        print(_canonical_json({"status": result["status"],
                               "scores": result.get("c64_scores_completed")}))
        return 0 if result["status"] == "C64_PARITY_PASS" else 3
    except S2Error as exc:
        if "manifest" in locals():
            manifest.update({
                "status": exc.status,
                "failure_type": type(exc).__name__,
                "failure_reason": str(exc),
                "finished_at_utc": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc).isoformat(),
                "total_gpu_subprocess_wall_seconds": time.perf_counter() - run_started_perf,
            })
        print(f"{exc.status}: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        if "manifest" in locals():
            manifest.update({
                "status": "INCOMPLETE",
                "failure_type": type(exc).__name__,
                "failure_reason": str(exc),
                "finished_at_utc": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc).isoformat(),
            })
        print(f"INCOMPLETE: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
