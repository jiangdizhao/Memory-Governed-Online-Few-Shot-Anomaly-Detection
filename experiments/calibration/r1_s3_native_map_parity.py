"""Bounded R1-S3 native dense-map parity smoke.

The module separates metadata-only preflight, one GPU replay, and a read-only
CPU tape verifier. It never imports a label provider or reads label/mask files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BRANCH = "calibration-stage0-contract-20261029"
PROFILE_ID = "r1_s3_native_map_parity_v1"
IDENTITY_CONTRACT = "r1-s3.source-bound.v1"
OUTPUT_PREFIX = "r1_s3_native_map_switch_s0_16_20261031_"
STAGE05 = ROOT / "outputs/calibration/stage05_exploratory_20261029"
S1_DIR = ROOT / "outputs/calibration/r1_digest_contract_v1_support_only_supportcheck_20261031_a1"
S2_DIR = ROOT / "outputs/calibration/r1_s2_c64_parity_switch_s0_20261031_c64parity_20261031_a1"
EXPECTED_FIXED_BANK_SHA256 = "8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558"
EXPECTED_RAW_BANK_SHA256 = "087feded0d780bbc3a56d45404b4126991b1aeb54f1ae3724bce025caf0d380e"
EXPECTED_STAGE05_SCORER_PROFILE_SHA256 = "1b8e9d96b121e3f1f6c1bbee13ab9f0063facee8fb08ad696dc5f9cb43dbd452"
EXPECTED_R1_SCORER_PROFILE_SHA256 = "ba3fa736734c2c6c3789f485f3288c5ae77177c68b6b3e24de7e257e447ced87"
EXPECTED_CHECKPOINT_SHA256 = "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
EXPECTED_HUB_TREE_SHA256 = "1de67887ca330bdfaad8ccd45128d2b9313d63aa707bbe8e4f38310e52005b24"
EXPECTED_D1_HUB_TREE_SHA256 = "b7c0037f90671c4932393bd56de8c8a205b29a6eb22dea563b6c8a88c471da15"
STRUCTURED_DIGEST_ALGORITHM = "experiments.calibration.native._fixed_bank_digest.v1:path-shape-dtype-bytes"
RAW_DIGEST_ALGORITHM = "experiments.p0_protocol_audit.fixed_bank_profile.v1:raw-prototype-bytes"
MAX_EVENTS = 16
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
SCALAR_ATOL = 1e-5
SCALAR_RTOL = 1e-4
EVENT_SCHEMA = "calibration.r1_s3_native_map_event.v1"
TAPE_SCHEMA = "calibration.r1_s3_native_map_evidence_tape.v1"
CORE_SOURCE_PATHS = (
    "test_phase3_AIF_simplified_v22.py",
    "vmb_visionad_new.py",
    "experiments/p0_protocol_audit.py",
    "experiments/calibration/native.py",
)
S1_SOURCE_PATHS = CORE_SOURCE_PATHS + (
    "experiments/a0_a1_strict4_matched.py",
    "experiments/calibration/stage05.py",
    "experiments/calibration/r1_matched_correction_pilot.py",
)


class S3Error(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    payload = (canonical_json(value) + "\n").encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_once(path: Path, payload: bytes) -> None:
    path = Path(path)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _directory_bytes(path: Path) -> int:
    if not Path(path).exists():
        return 0
    return sum(item.stat().st_size for item in Path(path).rglob("*")
               if item.is_file())


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Use the existing R1 deterministic, uncompressed NPZ tape codec."""
    from experiments.calibration.r1_matched_correction_pilot import deterministic_npz_bytes
    normalized = {str(key): np.ascontiguousarray(value)
                  for key, value in arrays.items()}
    return deterministic_npz_bytes(normalized)


def _array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = canonical_json({"shape": list(value.shape),
                             "dtype": value.dtype.str}).encode("ascii")
    return sha256_bytes(header + b"\0" + value.tobytes(order="C"))


def _raw_array_digest(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def validate_dense_map(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (784,) or result.dtype != np.dtype(np.float32):
        raise ValueError("base map must be exactly shape [784], dtype float32")
    if not np.isfinite(result).all():
        raise ValueError("base map contains nonfinite values")
    return result


def validate_top32(value: Sequence[int] | np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (32,) or not np.issubdtype(result.dtype, np.integer):
        raise ValueError("Top-32 IDs must be a length-32 integer sequence")
    result = np.ascontiguousarray(result, dtype=np.int64)
    if np.any(result < 0) or np.any(result >= 784):
        raise ValueError("Top-32 IDs fall outside the 784-patch grid")
    if np.unique(result).size != 32:
        raise ValueError("Top-32 IDs must be unique")
    return result


def max_aggregate(value: np.ndarray) -> float:
    dense = validate_dense_map(value)
    return float(np.max(dense))


def zero_correction_reconstruction(base_map: np.ndarray,
                                   residual: np.ndarray) -> np.ndarray:
    base = validate_dense_map(base_map)
    corr = np.asarray(residual)
    if corr.shape != base.shape or corr.dtype != np.dtype(np.float32):
        raise ValueError("zero-correction residual must match the float32 base map")
    if not np.isfinite(corr).all() or np.any(corr != 0):
        raise ValueError("correction-disabled profile requires an exact zero residual")
    return np.add(base, corr, dtype=np.float32)


def _reject_forbidden_event_fields(value: Any, path: str = "event") -> None:
    forbidden = ("label", "mask", "prediction", "c1", "c2", "anchor",
                 "greedy", "writer_transition")
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).lower()
            if any(token in name for token in forbidden):
                raise ValueError(f"forbidden event field: {path}.{key}")
            _reject_forbidden_event_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_event_fields(item, f"{path}[{index}]")


def _validate_profile_id(profile_id: str) -> None:
    if profile_id != PROFILE_ID:
        raise S3Error("BLOCKED_SOURCE_IDENTITY",
                      f"Unsupported S3 profile identity: {profile_id!r}")


def _validate_source_values(branch: str, actual_head: str,
                            expected_head: str, profile_id: str) -> None:
    _validate_profile_id(profile_id)
    if branch != BRANCH or not re.fullmatch(r"[0-9a-f]{40}", expected_head.lower()):
        raise S3Error("BLOCKED_SOURCE_IDENTITY", "Branch or expected source SHA is incompatible")
    if actual_head.lower() != expected_head.lower():
        raise S3Error("BLOCKED_SOURCE_IDENTITY",
                      f"Source revision mismatch: expected={expected_head}; actual={actual_head}")


def require_fresh_output_namespace(path: Path) -> None:
    if Path(path).exists():
        raise FileExistsError(f"Refusing to reuse output namespace: {path}")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        raise S3Error("BLOCKED_SOURCE_IDENTITY",
                      f"Git identity command failed: {' '.join(args)}: {exc.output}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise S3Error("BLOCKED_INPUT", f"Cannot read required JSON receipt {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise S3Error("BLOCKED_INPUT", f"Receipt is not a JSON object: {path.name}")
    return value


def _opaque_group(sample_key: str) -> str:
    return sha256_bytes(("official-sample-group\0" + str(sample_key)).encode("utf-8"))


def _resolve_image(dataset_root: Path, prefix: str, relative_id: str) -> Path:
    prefix_parts = [part for part in prefix.replace("\\", "/").split("/") if part]
    image_parts = [part for part in relative_id.replace("\\", "/").split("/") if part]
    if not image_parts or any(part in (".", "..") for part in prefix_parts + image_parts):
        raise S3Error("BLOCKED_INPUT", "Frozen image identity contains an unsafe path")
    root = (Path(dataset_root) / "images").resolve()
    result = (root / Path(*prefix_parts, *image_parts)).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise S3Error("BLOCKED_INPUT", "Frozen image path escapes the dataset image root") from exc
    return result


def _profile_receipt(stage_score_manifest: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from dataclasses import asdict
        from experiments.calibration.profiles import DEFAULT_PROFILE, NativeScorerConfig, canonical_json as profile_json
        current = NativeScorerConfig()
        recorded = stage_score_manifest["scorer_config"]
        recorded_sha = str(stage_score_manifest["scorer_config_sha256"])
        if (profile_json(asdict(DEFAULT_PROFILE.scorer)) != profile_json(recorded) or
                DEFAULT_PROFILE.scientific_sha256 != recorded_sha or
                current.sha256 != EXPECTED_R1_SCORER_PROFILE_SHA256 or
                recorded_sha != EXPECTED_STAGE05_SCORER_PROFILE_SHA256):
            raise S3Error("BLOCKED_SOURCE_IDENTITY", "Locked native scorer profile differs from Stage 0.5")
        return {"config": asdict(current), "sha256": current.sha256}
    except (KeyError, TypeError, AttributeError) as exc:
        raise S3Error("BLOCKED_INPUT", f"Stage 0.5 scorer profile receipt is incomplete: {exc}") from exc


def collect_preflight(*, expected_source_commit: str, profile_id: str,
                      run_id: str, authorization_id: str,
                      dataset_root: Path,
                      allow_existing_output: bool = False) -> dict[str, Any]:
    """Metadata-only source, receipt, and path preflight; no image bytes are read."""
    _validate_profile_id(profile_id)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{4,63}", run_id):
        raise S3Error("BLOCKED_SOURCE_IDENTITY", "Run ID has an invalid format")
    if not authorization_id or len(authorization_id) > 128:
        raise S3Error("BLOCKED_SOURCE_IDENTITY", "A separately authorized run ID is required")
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    _validate_source_values(branch, head, expected_source_commit, profile_id)
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise S3Error("BLOCKED_SOURCE_IDENTITY", "S3 execution requires a clean tracked worktree")
    untracked = _git("ls-files", "--others", "--exclude-standard").splitlines()
    shadowing = [path for path in untracked if path.endswith(".py") and
                 (path.startswith("experiments/calibration/") or
                  path.startswith("experiments/risk_safe/") or
                  path in CORE_SOURCE_PATHS)]
    if shadowing:
        raise S3Error("BLOCKED_SOURCE_IDENTITY",
                      "Untracked Python sources could shadow bound imports: " +
                      ", ".join(sorted(shadowing)[:8]))

    output_dir = ROOT / "outputs/calibration" / f"{OUTPUT_PREFIX}{run_id}"
    if output_dir.exists() and not allow_existing_output:
        raise S3Error("BLOCKED_INPUT", f"Refusing to reuse output namespace: {output_dir.name}")

    run_receipt = _load_json(STAGE05 / "run_manifest.json")
    score_receipt_path = STAGE05 / "score_tape_manifest.json"
    checksum_path = STAGE05 / "checksums.json"
    frozen_path = STAGE05 / "frozen_inputs.json"
    score_receipt = _load_json(score_receipt_path)
    checksums = _load_json(checksum_path).get("files", {})
    if (run_receipt.get("score_tape_manifest_sha256") != sha256_file(score_receipt_path) or
            checksums.get("frozen_inputs.json") != sha256_file(frozen_path) or
            score_receipt.get("status") != "PASS"):
        raise S3Error("BLOCKED_INPUT", "Stage 0.5 checksum or score receipt is invalid")
    scorer_profile = _profile_receipt(score_receipt)
    if (score_receipt.get("dynamic_memory") != "empty_and_disabled" or
            score_receipt.get("controller_or_test_label_access") is not False):
        raise S3Error("BLOCKED_SOURCE_IDENTITY", "Stage 0.5 bank or information profile is incompatible")

    frozen = _load_json(frozen_path)
    switch = frozen.get("classes", {}).get("switch")
    if not isinstance(switch, Mapping):
        raise S3Error("BLOCKED_INPUT", "Frozen switch profile is missing")
    supports = switch.get("supports", {}).get("0", {})
    c64 = switch.get("c64", {})
    stream = switch.get("stream", [])
    if (len(supports.get("images", [])) != 4 or len(supports.get("groups", [])) != 4 or
            len(c64.get("images", [])) != 64 or len(c64.get("groups", [])) != 64 or
            int(c64.get("count", -1)) != 64 or len(stream) < MAX_EVENTS or
            switch.get("labels_or_masks_returned_by_parser") not in (None, False) or
            frozen.get("stage1_or_stage2_run") is not False):
        raise S3Error("BLOCKED_INPUT", "Frozen support/C64/stream receipt is incompatible")
    if not switch.get("support_c64_disjoint") or not switch.get("train_test_groups_disjoint"):
        raise S3Error("BLOCKED_INPUT", "Frozen source-group disjointness receipt failed")

    stage_c64 = (score_receipt.get("classes", {}).get("switch", {})
                 .get("score_tapes", {}).get("0"))
    if not isinstance(stage_c64, Mapping) or stage_c64.get("fixed_support_sha256") != EXPECTED_FIXED_BANK_SHA256:
        raise S3Error("BLOCKED_INPUT", "Historical Stage 0.5 fixed-bank receipt is incompatible")
    score_path = (STAGE05 / str(stage_c64.get("file", ""))).resolve()
    try:
        score_path.relative_to(STAGE05.resolve())
    except ValueError as exc:
        raise S3Error("BLOCKED_INPUT", "Stage 0.5 C64 receipt path escapes the frozen output") from exc
    if (not score_path.is_file() or sha256_file(score_path) != stage_c64.get("sha256") or
            checksums.get(stage_c64.get("file")) != stage_c64.get("sha256")):
        raise S3Error("BLOCKED_INPUT", "Existing C64 score receipt checksum is invalid")

    s1 = _load_json(S1_DIR / "run_manifest.json")
    if (s1.get("status") != "SUPPORT_ONLY_VERIFIED" or
            s1.get("profile_id") != "r1_switch_s0_support_bank_verify_v1" or
            s1.get("branch") != BRANCH or
            s1.get("source_commit") != "cf1bf2b72381633663e68050e84d91f88bf4b402" or
            s1.get("expected_source_commit") != s1.get("source_commit") or
            s1.get("authorization_id") != "R1-S1-SUPPORT-ONLY-20261031" or
            s1.get("run_id") != "supportcheck_20261031_a1" or
            s1.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256 or
            s1.get("hub_source_tree_sha256") != EXPECTED_D1_HUB_TREE_SHA256 or
            s1.get("runtime_profile", {}).get("profile_id") != "stage05_set_seed_0_math_sdp_v1" or
            s1.get("fixed_support_sha256") != EXPECTED_FIXED_BANK_SHA256 or
            s1.get("fixed_support_sha256_expected") != EXPECTED_FIXED_BANK_SHA256 or
            s1.get("fixed_support_digest_algorithm") != STRUCTURED_DIGEST_ALGORITHM or
            s1.get("fixed_support_raw_prototype_sha256") != EXPECTED_RAW_BANK_SHA256 or
            s1.get("fixed_support_raw_digest_algorithm") != RAW_DIGEST_ALGORITHM or
            s1.get("native_scorer_profile_sha256") != EXPECTED_R1_SCORER_PROFILE_SHA256 or
            s1.get("events_processed") != 0 or s1.get("c64_rescored") is not False or
            s1.get("stream_images_decoded") != 0 or s1.get("queried_labels_decoded") != 0):
        raise S3Error("BLOCKED_INPUT", "R1-S1 support-only receipt is invalid")

    s2 = _load_json(S2_DIR / "run_manifest.json")
    launcher = _load_json(S2_DIR / "launcher_manifest.json")
    s2_compare = s2.get("score_comparison", {})
    bitwise = s2_compare.get("bitwise_score_parity",
                             s2_compare.get("BITWISE_SCORE_PARITY"))
    s2_profile = s2.get("profile", {})
    if (s2.get("status") != "C64_PARITY_PASS" or
            s2.get("branch") != BRANCH or
            s2.get("identity_contract") != "r1-s2.source-bound.v1" or
            s2.get("source_commit") != "0bb29966d5b74cda67b5b81fe0966c0f0487eda3" or
            s2.get("source_commit_used_for_gpu") != "0bb29966d5b74cda67b5b81fe0966c0f0487eda3" or
            s2.get("authorization_id") != "R1-S2-BOUNDED-C64-PARITY-20261031" or
            s2.get("run_id") != "c64parity_20261031_a1" or
            s2.get("fixed_support_sha256_expected") != EXPECTED_FIXED_BANK_SHA256 or
            s2.get("fixed_support_sha256_actual") != EXPECTED_FIXED_BANK_SHA256 or
            s2.get("fixed_support_digest_algorithm") != STRUCTURED_DIGEST_ALGORITHM or
            s2.get("dino_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256 or
            s2.get("dino_source_tree_sha256_historical_contract") != EXPECTED_HUB_TREE_SHA256 or
            s2.get("runtime_profile", {}).get("profile_id") != "stage05_set_seed_0_math_sdp_v1" or
            s2.get("c64_scores_completed") != 64 or
            s2_compare.get("exact_elementwise_equality_count") != 64 or bitwise is not True or
            s2.get("query_labels_accessed") is not False or
            s2.get("test_labels_or_masks_accessed") is not False or
            s2.get("stream_images_decoded") != 0 or
            s2.get("dynamic_greedy_memory_empty") is not True or
            s2.get("correction_bank_enabled") is not False or
            s2.get("defect_bank_enabled") is not False or
            s2.get("greedy_writer_events") != 0 or
            launcher.get("status") != "C64_PARITY_PASS" or
            launcher.get("source_commit") != s2.get("source_commit") or
            s2_profile.get("profile_id") not in (None, "r1_s2_c64_score_parity_v1")):
        raise S3Error("BLOCKED_INPUT", "R1-S2 C64 parity receipt is invalid")
    s2_array_path = S2_DIR / "c64_scores.npz"
    s2_array_sha = sha256_file(s2_array_path) if s2_array_path.is_file() else None
    if (not s2_array_sha or s2_array_sha != s2.get("rebuilt_score_array_sha256") or
            s2_array_sha != s2.get("score_array_sha256", s2_array_sha)):
        raise S3Error("BLOCKED_INPUT", "R1-S2 C64 result artifact checksum is invalid")
    for log_name, receipt_key in (("stdout.log", "stdout_sha256"),
                                  ("stderr.log", "stderr_sha256")):
        log_path = S2_DIR / log_name
        if (not log_path.is_file() or not launcher.get(receipt_key) or
                sha256_file(log_path) != launcher.get(receipt_key)):
            raise S3Error("BLOCKED_INPUT", f"R1-S2 launcher log receipt is invalid: {log_name}")
    protected = s2.get("preflight", {}).get("historical_failed_r1", {}).get(
        "protected_artifact_sha256", {})
    if not protected:
        raise S3Error("BLOCKED_INPUT", "R1-S2 protected-artifact receipt is missing")
    for relative, expected_sha in protected.items():
        protected_path = ROOT / str(relative)
        if not protected_path.is_file() or sha256_file(protected_path) != expected_sha:
            raise S3Error("BLOCKED_INPUT", f"Protected S1/failed-R1 artifact changed: {relative}")

    current_hashes = {}
    for relative in S1_SOURCE_PATHS:
        source = ROOT / relative
        if not source.is_file():
            raise S3Error("BLOCKED_SOURCE_IDENTITY", f"Bound source is missing: {relative}")
        current_hashes[relative] = sha256_file(source)
    s2_comparison = s2.get("preflight", {}).get("source_fingerprints", {})
    for relative in CORE_SOURCE_PATHS:
        row = s2_comparison.get(relative)
        if (not isinstance(row, Mapping) or
                row.get("current_sha256") != current_hashes[relative] or
                row.get("core_scorer_match") is not True):
            raise S3Error("BLOCKED_SOURCE_IDENTITY",
                          f"Core source fingerprint differs from R1-S2: {relative}")
    s1_source_hashes = s1.get("source_hashes", {})
    for relative in S1_SOURCE_PATHS:
        prior = s1_source_hashes.get(relative)
        if prior is None:
            raise S3Error("BLOCKED_SOURCE_IDENTITY",
                          f"R1-S1 receipt lacks required source fingerprint: {relative}")
        if prior != current_hashes[relative]:
            raise S3Error("BLOCKED_SOURCE_IDENTITY",
                          f"Core source fingerprint differs from R1-S1: {relative}")

    prefix = str(switch.get("prefix", ""))
    data_root = Path(dataset_root).resolve()
    support_paths = []
    for relative in supports["images"]:
        path = _resolve_image(data_root, prefix, str(relative))
        if not path.is_file():
            raise S3Error("BLOCKED_INPUT", f"Frozen support image is unavailable: {path.name}")
        support_paths.append(path)

    event_plan = []
    resolved_stream_paths = []
    support_groups = {_opaque_group(str(group)) for group in supports.get("groups", [])}
    c64_groups = {_opaque_group(str(group)) for group in c64.get("groups", [])}
    if len(support_groups) != 4 or len(c64_groups) != 64 or support_groups & c64_groups:
        raise S3Error("BLOCKED_INPUT", "Support/C64 official source-group identities are invalid")
    seen_relative_ids: set[str] = set()
    for position, row in enumerate(stream[:MAX_EVENTS]):
        if int(row.get("position", -1)) != position:
            raise S3Error("BLOCKED_INPUT", f"Frozen stream position mismatch at {position}")
        sample_key = str(row.get("sample_key", ""))
        group_sha = _opaque_group(sample_key)
        if not sample_key or row.get("opaque_group") != group_sha:
            raise S3Error("BLOCKED_INPUT", f"Frozen opaque group identity mismatch at {position}")
        if group_sha in support_groups or group_sha in c64_groups:
            raise S3Error("BLOCKED_INPUT", f"Frozen source group overlaps at event {position}")
        relative_id = str(row.get("relative_id", ""))
        if not relative_id or relative_id in seen_relative_ids:
            raise S3Error("BLOCKED_INPUT", f"Duplicate or empty stream image identity at {position}")
        seen_relative_ids.add(relative_id)
        path = _resolve_image(data_root, prefix, relative_id)
        if not path.is_file():
            raise S3Error("BLOCKED_INPUT", f"Frozen stream image is unavailable at event {position}")
        resolved_stream_paths.append(path)
        event_plan.append({
            "event_id": f"switch-s0-{position:04d}",
            "position": position,
            "opaque_group_identity_sha256": group_sha,
            "relative_id_sha256": sha256_bytes(relative_id.encode("utf-8")),
        })

    return {
        "identity_contract": IDENTITY_CONTRACT,
        "branch": branch,
        "source_commit": head,
        "profile_id": PROFILE_ID,
        "authorization_id": authorization_id,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "dataset_root": str(data_root),
        "category": "switch",
        "support_seed": 0,
        "support_count": len(support_paths),
        "event_count": MAX_EVENTS,
        "event_plan": event_plan,
        "support_relative_ids_sha256": [
            sha256_bytes(str(value).encode("utf-8")) for value in supports["images"]],
        "stage05_fixed_support_sha256": stage_c64["fixed_support_sha256"],
        "stage05_c64_receipt_sha256": stage_c64["sha256"],
        "stage05_c64_score_file_sha256": sha256_file(score_path),
        "s1_receipt_sha256": sha256_file(S1_DIR / "run_manifest.json"),
        "s2_receipt_sha256": sha256_file(S2_DIR / "run_manifest.json"),
        "s2_score_array_sha256": s2_array_sha,
        "native_scorer_profile": scorer_profile,
        "source_fingerprints": current_hashes,
        "support_paths": [str(path) for path in support_paths],
        "stream_paths": [str(path) for path in resolved_stream_paths],
        "runtime_profile_intended": "stage05_set_seed_0_math_sdp_v1",
    }


def _safe_preflight_output(preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Return a public receipt without absolute image paths or group names."""
    return {key: value for key, value in preflight.items()
            if key not in ("dataset_root", "support_paths", "stream_paths")}


def _event_arrays(*, base_map: np.ndarray, debug_map: np.ndarray,
                  adapter_ids: np.ndarray, native_ids: np.ndarray,
                  debug_ids: np.ndarray, selected_features: np.ndarray) -> dict[str, np.ndarray]:
    base = validate_dense_map(base_map)
    native_map = np.asarray(debug_map)
    if (native_map.shape != (28, 28) or native_map.dtype != np.dtype(np.float32) or
            not np.isfinite(native_map).all()):
        raise ValueError("native A_pre map must be finite float32 [28,28]")
    features = np.asarray(selected_features)
    if (features.shape != (32, 4096) or features.dtype != np.dtype(np.float32) or
            not np.isfinite(features).all()):
        raise ValueError("selected features must be finite float32 [32,4096]")
    return {
        "adapter_base_map_fp32": np.ascontiguousarray(base),
        "native_a_pre_map_fp32": np.ascontiguousarray(native_map),
        "adapter_top32_ids_i64": validate_top32(adapter_ids),
        "native_top32_ids_i64": validate_top32(native_ids),
        "debug_top32_ids_i64": validate_top32(debug_ids),
        "selected_normalized_features_fp32": np.ascontiguousarray(features),
    }


def _map_comparison(adapter: np.ndarray, native_hw: np.ndarray) -> dict[str, Any]:
    base = validate_dense_map(adapter)
    native = np.ascontiguousarray(np.asarray(native_hw).reshape(-1, order="C"))
    if native.shape != (784,) or native.dtype != np.dtype(np.float32):
        raise ValueError("native map flattening did not preserve the 784-patch FP32 grid")
    delta = np.abs(base.astype(np.float64) - native.astype(np.float64))
    equal_count = int(np.count_nonzero(base == native))
    base_bytes_sha = _raw_array_digest(base)
    native_bytes_sha = _raw_array_digest(native)
    return {
        "exact_elementwise_equality_count": equal_count,
        "exact_array_equality": bool(np.array_equal(base, native)),
        "exact_byte_equality": base_bytes_sha == native_bytes_sha,
        "adapter_map_bytes_sha256": base_bytes_sha,
        "native_map_bytes_sha256": native_bytes_sha,
        "maximum_absolute_difference": float(delta.max(initial=0.0)),
        "mean_absolute_difference": float(delta.mean()),
        "adapter_shape": list(base.shape),
        "adapter_dtype": str(base.dtype),
        "native_shape": list(native_hw.shape),
        "native_dtype": str(native_hw.dtype),
        "coordinate_transform": "native A_pre [28,28] flattened row-major to identity [784]; no interpolation",
    }


def _event_shard_paths(output_dir: Path, position: int) -> tuple[Path, Path]:
    name = f"event_{int(position):04d}"
    return (output_dir / "events" / f"{name}.npz",
            output_dir / "events" / f"{name}.json")


def _write_event(output_dir: Path, arrays: Mapping[str, np.ndarray],
                 metadata: Mapping[str, Any]) -> dict[str, Any]:
    _reject_forbidden_event_fields(metadata)
    position = int(metadata["position"])
    shard_path, event_path = _event_shard_paths(output_dir, position)
    shard_bytes = _deterministic_npz_bytes(arrays)
    array_digests = {name: _array_digest(np.asarray(value))
                     for name, value in arrays.items()}
    event = dict(metadata)
    event.update({
        "schema": EVENT_SCHEMA,
        "arrays": {name: {"shape": list(np.asarray(value).shape),
                           "dtype": np.asarray(value).dtype.str,
                           "sha256": array_digests[name]}
                   for name, value in arrays.items()},
        "shard_file": shard_path.name,
        "shard_sha256": sha256_bytes(shard_bytes),
    })
    event_bytes = (canonical_json(event) + "\n").encode("utf-8")
    if _directory_bytes(output_dir) + len(shard_bytes) + len(event_bytes) > MAX_OUTPUT_BYTES:
        raise S3Error("RESOURCE_LIMIT_REACHED", "S3 output would exceed 128 MiB")
    _write_once(shard_path, shard_bytes)
    _write_once(event_path, event_bytes)
    return event


def _load_array_shard(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    except Exception as exc:
        raise ValueError(f"Cannot read immutable event shard {path.name}: {exc}") from exc


def verify_tape(output_dir: Path) -> dict[str, Any]:
    """Read and verify only saved CPU tape files; never loads torch or images."""
    output_dir = Path(output_dir)
    manifest = _load_json(output_dir / "run_manifest.json")
    if manifest.get("schema") != TAPE_SCHEMA:
        raise ValueError("Unsupported S3 tape schema")
    plan = manifest.get("event_plan")
    if not isinstance(plan, list):
        raise ValueError("Run manifest lacks a frozen event plan")
    recorded = int(manifest.get("events_processed", 0))
    expected_positions = list(range(recorded))
    if recorded < 0 or recorded > MAX_EVENTS:
        raise ValueError("Recorded event count is outside the authorized range")
    event_dir = output_dir / "events"
    json_files = sorted(event_dir.glob("event_*.json"))
    npz_files = sorted(event_dir.glob("event_*.npz"))
    if len(json_files) != recorded or len(npz_files) != recorded:
        raise ValueError("Event metadata/shard count differs from run manifest")
    events = []
    all_positions = []
    exact_map_events = 0
    exact_top32_events = 0
    adapter_max_events = 0
    scalar_exact_events = 0
    scalar_tolerance_events = 0
    zero_reconstruction_events = 0
    dynamic_memory_events = 0
    feature_immutability_events = 0
    for position in expected_positions:
        shard_path, event_path = _event_shard_paths(output_dir, position)
        if not shard_path.is_file() or not event_path.is_file():
            raise ValueError(f"Missing event shard at position {position}")
        event = _load_json(event_path)
        _reject_forbidden_event_fields(event)
        if (event.get("schema") != EVENT_SCHEMA or
                event.get("event_id") != f"switch-s0-{position:04d}" or
                int(event.get("position", -1)) != position or
                event.get("shard_file") != shard_path.name or
                event.get("shard_sha256") != sha256_file(shard_path)):
            raise ValueError(f"Event identity or shard checksum mismatch at {position}")
        if position >= len(plan):
            raise ValueError("Event position is absent from manifest plan")
        plan_row = plan[position]
        for event_key, plan_key in (("opaque_group_identity_sha256", "opaque_group_identity_sha256"),
                                    ("relative_id_sha256", "relative_id_sha256")):
            if event.get(event_key) != plan_row.get(plan_key):
                raise ValueError(f"Source identity mismatch at event {position}: {event_key}")
        for key in ("opaque_group_identity_sha256", "relative_id_sha256",
                    "source_image_sha256", "fixed_support_sha256",
                    "fixed_support_raw_prototype_sha256", "encoder_fingerprint_sha256",
                    "preprocess_fingerprint_sha256", "scorer_profile_sha256",
                    "feature_tensor_sha256", "selected_feature_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(event.get(key, ""))):
                raise ValueError(f"Malformed event fingerprint at position {position}: {key}")
        if (event.get("fixed_support_sha256") != manifest.get("fixed_support_sha256_actual") or
                event.get("fixed_support_raw_prototype_sha256") !=
                manifest.get("fixed_support_raw_prototype_sha256")):
            raise ValueError(f"Fixed-bank identity mismatch at event {position}")
        if (not re.fullmatch(r"[0-9a-f]{64}", str(event.get("pre_dynamic_memory_sha256", ""))) or
                not re.fullmatch(r"[0-9a-f]{64}", str(event.get("post_dynamic_memory_sha256", "")))):
            raise ValueError(f"Dynamic-memory identity is malformed at event {position}")
        if manifest.get("dynamic_memory_unchanged_and_empty") is True and (
                event.get("pre_dynamic_memory_sha256") != manifest.get("dynamic_memory_sha256_before") or
                event.get("post_dynamic_memory_sha256") != manifest.get("dynamic_memory_sha256_after") or
                event.get("dynamic_memory_empty_and_unchanged") is not True):
            raise ValueError(f"Per-event dynamic-memory identity differs at event {position}")
        arrays = _load_array_shard(shard_path)
        if set(arrays) != set(event.get("arrays", {})):
            raise ValueError(f"Array inventory mismatch at event {position}")
        for name, value in arrays.items():
            receipt = event["arrays"][name]
            if (list(value.shape) != receipt.get("shape") or
                    value.dtype.str != receipt.get("dtype") or
                    _array_digest(value) != receipt.get("sha256")):
                raise ValueError(f"Array checksum/precision mismatch at {position}: {name}")
        base = validate_dense_map(arrays["adapter_base_map_fp32"])
        native_hw = arrays["native_a_pre_map_fp32"]
        ids = validate_top32(arrays["adapter_top32_ids_i64"])
        native_ids = validate_top32(arrays["native_top32_ids_i64"])
        debug_ids = validate_top32(arrays["debug_top32_ids_i64"])
        features = arrays["selected_normalized_features_fp32"]
        if (native_hw.shape != (28, 28) or native_hw.dtype != np.float32 or
                features.shape != (32, 4096) or features.dtype != np.float32 or
                not np.isfinite(native_hw).all() or not np.isfinite(features).all()):
            raise ValueError(f"Event array shape/dtype invalid at {position}")
        feature_norms = np.linalg.norm(features.astype(np.float64), axis=1)
        if not np.allclose(feature_norms, 1.0, rtol=0.0, atol=1e-5):
            raise ValueError(f"Selected feature rows are not unit-normalized at {position}")
        map_result = _map_comparison(base, native_hw)
        recorded_map = event.get("map_parity", {})
        if (recorded_map.get("adapter_map_bytes_sha256") != map_result["adapter_map_bytes_sha256"] or
                recorded_map.get("native_map_bytes_sha256") != map_result["native_map_bytes_sha256"]):
            raise ValueError(f"Recorded dense-map checksums disagree at event {position}")
        map_equal = (map_result["exact_elementwise_equality_count"] == 784 and
                     map_result["exact_byte_equality"])
        top32_equal = np.array_equal(ids, native_ids) and np.array_equal(ids, debug_ids)
        for value, key in ((ids, "adapter_top32_ids_sha256"),
                           (native_ids, "native_top32_ids_sha256"),
                           (debug_ids, "debug_top32_ids_sha256")):
            if _array_digest(value) != event.get(key):
                raise ValueError(f"Recorded Top-32 checksum disagrees at event {position}")
        if _array_digest(features) != event.get("selected_feature_sha256"):
            raise ValueError(f"Selected-feature checksum disagrees at event {position}")
        exact_map_events += int(map_equal)
        exact_top32_events += int(top32_equal)
        reconstructed = max_aggregate(base)
        adapter_max_equal = reconstructed == float(event["adapter_score"])
        adapter_max_events += int(adapter_max_equal)
        native_score = float(event["native_score"])
        scalar_exact_events += int(float(event["adapter_score"]) == native_score)
        tolerance = SCALAR_ATOL + SCALAR_RTOL * abs(native_score)
        scalar_tolerance_events += int(abs(reconstructed - native_score) <= tolerance)
        zero_reconstruction_events += int(
            event.get("zero_correction_reconstruction_exact") is True)
        dynamic_memory_events += int(
            event.get("dynamic_memory_empty_and_unchanged") is True)
        feature_immutability_events += int(
            event.get("source_feature_tensor_unchanged") is True)
        all_positions.append(position)
        events.append(event)
    if all_positions != expected_positions or len(set(all_positions)) != recorded:
        raise ValueError("Tape has duplicate, missing, or reordered event positions")
    output_bytes = _directory_bytes(output_dir)
    if output_bytes > MAX_OUTPUT_BYTES:
        raise ValueError("Saved tape exceeds the 128 MiB output ceiling")
    full_count = recorded == MAX_EVENTS
    return {
        "readback_integrity_pass": True,
        "schema": TAPE_SCHEMA,
        "events_present": recorded,
        "event_positions": all_positions,
        "dense_map_exact_readback_count": exact_map_events,
        "top32_exact_readback_count": exact_top32_events,
        "scalar_max_readback_count": adapter_max_events,
        "scalar_exact_readback_count": scalar_exact_events,
        "scalar_tolerance_readback_count": scalar_tolerance_events,
        "zero_correction_readback_count": zero_reconstruction_events,
        "dynamic_memory_readback_count": dynamic_memory_events,
        "source_feature_immutability_readback_count": feature_immutability_events,
        "source_group_and_bank_identity_match": True,
        "selected_features_shape_dtype": "[32,4096] float32" if events else None,
        "base_map_shape_dtype": "[784] float32" if events else None,
        "full_s3_event_count": full_count,
        "output_bytes": output_bytes,
    }


def _update_manifest(path: Path, updates: Mapping[str, Any]) -> dict[str, Any]:
    value = _load_json(path)
    value.update(dict(updates))
    value["output_bytes"] = _directory_bytes(path.parent)
    if value["output_bytes"] > MAX_OUTPUT_BYTES:
        raise S3Error("RESOURCE_LIMIT_REACHED", "S3 output exceeded 128 MiB")
    _atomic_json(path, value)
    return value


def _empty_memory_identity(bank: Any) -> tuple[str, int]:
    from experiments.risk_safe.native_bridge import dynamic_codebooks, snapshot
    codebooks = dynamic_codebooks(bank)
    sizes = [int(codebook.size()) for codebook in codebooks]
    identity = snapshot(bank, version=0).sha256()
    return identity, sum(sizes)


def _correction_defect_state(bank: Any) -> dict[str, Any]:
    correction_sizes = [int(cb.size()) for cls in bank.corr_cb.values()
                        for view in cls.values() for cb in view]
    defect_sizes = [int(cb.size()) for cls in bank.defect_cb.values()
                    for view in cls.values() for cb in view]
    return {
        "correction_enabled": bool(bank.corr_cfg.enabled),
        "correction_anchor_count": sum(correction_sizes),
        "defect_bank_enabled": bool(bank.use_defect_bank),
        "defect_prototype_count": sum(defect_sizes),
    }


def _run_gpu(*, expected_source_commit: str, run_id: str,
             authorization_id: str, dataset_root: Path,
             output_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    manifest_path = output_dir / "run_manifest.json"
    preflight = collect_preflight(
        expected_source_commit=expected_source_commit,
        profile_id=PROFILE_ID, run_id=run_id,
        authorization_id=authorization_id, dataset_root=dataset_root,
        allow_existing_output=True)
    expected_output = Path(preflight["output_dir"]).resolve()
    if expected_output != Path(output_dir).resolve() or not output_dir.is_dir():
        raise S3Error("BLOCKED_SOURCE_IDENTITY", "Output path differs from source-bound run namespace")
    allowed = {"run_manifest.json", "preflight.json", "launcher_manifest.json",
               "stdout.log", "stderr.log", "tape_readback.json",
               "tape_readback_stdout.log", "tape_readback_stderr.log"}
    current_entries = {item.name for item in output_dir.iterdir()}
    if not current_entries.issubset(allowed | {"events"}) or "events" not in current_entries:
        raise S3Error("BLOCKED_INPUT", "GPU output directory is not a fresh supervisor bootstrap")
    event_dir = output_dir / "events"
    if not event_dir.is_dir() or any(event_dir.iterdir()):
        raise S3Error("BLOCKED_INPUT", "S3 runner refuses to resume a nonempty tape")

    import torch
    from experiments import p0_protocol_audit as p0
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.calibration.profiles import NativeScorerConfig
    from experiments.calibration.r1_matched_correction_pilot import (
        _configure_historical_runtime, _native_dense_score,
        _source_hashes, _hub_source_sha256, _model_state_sha256,
    )

    summary = {
        "schema": TAPE_SCHEMA,
        "identity_contract": IDENTITY_CONTRACT,
        "branch": preflight["branch"],
        "source_commit": preflight["source_commit"],
        "source_commit_used_for_gpu": preflight["source_commit"],
        "profile_id": PROFILE_ID,
        "authorization_id": authorization_id,
        "run_id": run_id,
        "status": "INCOMPLETE",
        "gpu_execution_status": "STARTED",
        "category": "switch",
        "support_seed": 0,
        "support_count": 4,
        "authorized_stream_positions": list(range(MAX_EVENTS)),
        "event_plan": preflight["event_plan"],
        "events_processed": 0,
        "dense_map_exact_event_count": 0,
        "top32_exact_event_count": 0,
        "scalar_exact_event_count": 0,
        "scalar_tolerance_event_count": 0,
        "native_map_grid": [28, 28],
        "adapter_map_shape": [784],
        "adapter_map_dtype": "float32",
        "selected_feature_shape": [32, 4096],
        "selected_feature_dtype": "float32",
        "fixed_support_sha256_expected": EXPECTED_FIXED_BANK_SHA256,
        "fixed_support_digest_algorithm": STRUCTURED_DIGEST_ALGORITHM,
        "fixed_support_raw_digest_algorithm": RAW_DIGEST_ALGORITHM,
        "historical_numerical_equivalence_claimed": False,
        "runtime_profile_id": None,
        "source_fingerprints": preflight["source_fingerprints"],
        "native_scorer_profile": preflight["native_scorer_profile"],
        "support_images_encoded": 0,
        "stream_images_decoded": 0,
        "c64_images_decoded_or_rescored": False,
        "prohibited_payloads_accessed": [],
        "normal_memory_writes": 0,
        "greedy_writer_invoked": False,
        "correction_or_defect_updates": 0,
        "timings_seconds": {
            "model_initialization": None,
            "support_bank_construction": None,
            "stream_encoding": 0.0,
            "adapter_scoring": 0.0,
            "native_score_bank": 0.0,
            "diagnostic_map_scoring": 0.0,
            "tape_serialization": 0.0,
        },
        "resource_metrics": {"peak_cuda_allocated_bytes": None,
                             "peak_cuda_reserved_bytes": None,
                             "peak_cpu_working_set_bytes": None},
        "preflight_receipts": {
            "stage05_c64_score_file_sha256": preflight["stage05_c64_score_file_sha256"],
            "s1_receipt_sha256": preflight["s1_receipt_sha256"],
            "s2_receipt_sha256": preflight["s2_receipt_sha256"],
            "s2_score_array_sha256": preflight["s2_score_array_sha256"],
        },
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not manifest_path.is_file():
        _write_once(manifest_path,
                    (canonical_json(summary) + "\n").encode("utf-8"))
    else:
        boot = _load_json(manifest_path)
        if boot.get("schema") != TAPE_SCHEMA or boot.get("status") != "INCOMPLETE":
            raise S3Error("BLOCKED_INPUT", "Supervisor bootstrap manifest is incompatible")
        for key in ("supervisor_started_at_utc", "supervisor_pid",
                    "supervisor_output_limit_bytes"):
            if key in boot:
                summary[key] = boot[key]
        _atomic_json(manifest_path, summary)

    if not torch.cuda.is_available():
        raise S3Error("BLOCKED_INPUT", "Authorized CUDA device is unavailable")
    torch.cuda.reset_peak_memory_stats()
    impl = p0.require_sources()
    runtime_profile = _configure_historical_runtime(impl, torch)
    if runtime_profile.get("profile_id") != "stage05_set_seed_0_math_sdp_v1":
        raise S3Error("BLOCKED_SOURCE_IDENTITY", "Historical runtime profile is incompatible")
    summary["runtime_profile_id"] = runtime_profile["profile_id"]
    summary["runtime_profile"] = runtime_profile
    summary["python_version"] = platform.python_version()
    summary["torch_version"] = str(torch.__version__)
    summary["cuda_runtime_version"] = str(torch.version.cuda)
    summary["cuda_device_name"] = str(torch.cuda.get_device_name())
    summary["device"] = "cuda"
    checkpoint_dir = Path(torch.hub.get_dir())
    hub_path, hub_tree_sha = _hub_source_sha256(checkpoint_dir)
    checkpoint = checkpoint_dir / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
    if not checkpoint.is_file():
        raise S3Error("BLOCKED_INPUT", "Cached DINOv2 checkpoint unavailable; network is disabled")
    checkpoint_sha = sha256_file(checkpoint)
    from experiments.calibration.r1_s2_c64_score_parity import _historical_tree_sha256
    historical_hub_tree_sha = _historical_tree_sha256(Path(hub_path))
    if (checkpoint_sha != EXPECTED_CHECKPOINT_SHA256 or
            hub_tree_sha != EXPECTED_D1_HUB_TREE_SHA256 or
            historical_hub_tree_sha != EXPECTED_HUB_TREE_SHA256):
        raise S3Error("BLOCKED_SOURCE_IDENTITY", "DINOv2 source/weight fingerprint differs from S1/S2")
    summary["checkpoint_file_name"] = checkpoint.name
    summary["checkpoint_sha256"] = checkpoint_sha
    summary["hub_source_path_name"] = Path(hub_path).name
    summary["hub_source_tree_sha256"] = hub_tree_sha
    summary["hub_source_tree_sha256_historical_contract"] = historical_hub_tree_sha
    summary["runtime_source_hashes"] = _source_hashes()
    _atomic_json(manifest_path, summary)

    cfg = p0.Config(
        device="cuda", augmentation_seed=0, bank_seed=0,
        fixed_capacity=25000, fixed_per_image=5000, augmentations=4,
        include_identity=False, multiplier=2.5, feature_fuse="concat",
        topk_patches=32, image_aggregation="max", image_topk=64,
        image_quantile=0.95, knn=5)
    scorer_profile = NativeScorerConfig()
    if scorer_profile.sha256 != EXPECTED_R1_SCORER_PROFILE_SHA256:
        raise S3Error("BLOCKED_SOURCE_IDENTITY", "Runtime NativeScorerConfig profile changed")
    from experiments.calibration.r1_matched_correction_pilot import _validate_p0_profile
    _validate_p0_profile(cfg, scorer_profile)

    model_started = time.perf_counter()
    torch.cuda.synchronize()
    backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
    model_sha = _model_state_sha256(backbone.model)
    torch.cuda.synchronize()
    summary["timings_seconds"]["model_initialization"] = time.perf_counter() - model_started
    summary["loaded_model_state_sha256"] = model_sha
    summary["encoder_fingerprint_sha256"] = sha256_json({
        "model_name": scorer_profile.model_name,
        "loaded_model_state_sha256": model_sha,
        "hub_source_path_name": Path(hub_path).name,
        "hub_source_tree_sha256": hub_tree_sha,
        "source_hashes": summary["runtime_source_hashes"],
        "hook_layers": list(backbone.layers),
    })
    summary["preprocess_fingerprint_sha256"] = sha256_json({
        "profile": scorer_profile.__dict__, "rgb": "PIL.convert(RGB)",
        "resize": [448, 448], "crop": [392, 392],
        "interpolation": "PIL.BICUBIC", "normalization": "model mean/std",
        "fusion": "concat_then_l2_normalize", "view": "identity",
    })

    support_started = time.perf_counter()
    support_features = p0.encode_reference_features(
        impl, backbone, preflight["support_paths"], cfg)
    summary["support_images_encoded"] = len(preflight["support_paths"])
    bank = p0.build_vmb(impl, support_features, preflight["support_paths"], cfg)
    torch.cuda.synchronize()
    summary["timings_seconds"]["support_bank_construction"] = time.perf_counter() - support_started
    raw_before = str(p0.fixed_bank_profile(bank)["sha256"])
    structured_before = str(_fixed_bank_digest(bank))
    layer_profile = []
    for item in p0.fixed_bank_profile(bank)["layers"]:
        row = dict(item)
        row["shape"] = [int(row["n_prototypes"]), int(row["feature_dim"])]
        layer_profile.append(row)
    summary["fixed_support_raw_prototype_sha256"] = raw_before
    summary["fixed_support_layer_profile"] = layer_profile
    summary["fixed_support_sha256_actual"] = structured_before
    summary["fixed_bank_digest_validation"] = {
        "expected_structured_sha256": EXPECTED_FIXED_BANK_SHA256,
        "expected_structured_algorithm": STRUCTURED_DIGEST_ALGORITHM,
        "actual_structured_sha256": structured_before,
        "actual_structured_algorithm": STRUCTURED_DIGEST_ALGORITHM,
        "actual_raw_prototype_sha256": raw_before,
        "actual_raw_prototype_algorithm": RAW_DIGEST_ALGORITHM,
        "layer_profile": layer_profile,
        "matches_expected_structured_digest": structured_before == EXPECTED_FIXED_BANK_SHA256,
    }
    _atomic_json(manifest_path, summary)
    if structured_before != EXPECTED_FIXED_BANK_SHA256:
        raise S3Error("BLOCKED_FIXED_BANK", "Rebuilt structured fixed-bank digest differs from S1/S2 receipt")
    if raw_before != EXPECTED_RAW_BANK_SHA256:
        summary["raw_prototype_digest_matches_s1_diagnostic"] = False
    else:
        summary["raw_prototype_digest_matches_s1_diagnostic"] = True

    if bank.corr_cfg.enabled or bank.use_defect_bank:
        raise S3Error("BLOCKED_SOURCE_IDENTITY", "Correction or defect bank is enabled")
    pre_corr_defect = _correction_defect_state(bank)
    if (pre_corr_defect["correction_anchor_count"] != 0 or
            pre_corr_defect["defect_prototype_count"] != 0):
        raise S3Error("BLOCKED_SOURCE_IDENTITY", "Correction/defect codebooks are not empty")
    dynamic_before, dynamic_count = _empty_memory_identity(bank)
    if dynamic_count != 0:
        raise S3Error("MEMORY_IMMUTABILITY_FAILED", "Dynamic normal memory is not empty before replay")
    summary["fixed_support_raw_sha256_before"] = raw_before
    summary["fixed_support_structured_sha256_before"] = structured_before
    summary["dynamic_memory_sha256_before"] = dynamic_before
    summary["dynamic_memory_count_before"] = dynamic_count
    summary["correction_defect_state_before"] = pre_corr_defect
    _atomic_json(manifest_path, summary)

    from experiments.a0_a1_strict4_matched import score_bank
    stream_started = time.perf_counter()
    primary_failure: S3Error | None = None
    try:
        for position, image_path in enumerate(preflight["stream_paths"]):
            image_sha = sha256_file(Path(image_path))
            encode_started = time.perf_counter()
            torch.cuda.synchronize()
            image, features = p0.encode_query(impl, backbone, str(image_path), cfg)
            torch.cuda.synchronize()
            summary["timings_seconds"]["stream_encoding"] += time.perf_counter() - encode_started
            summary["stream_images_decoded"] += 1
            _atomic_json(manifest_path, summary)
            if len(features) != 1 or tuple(features[0].shape) != (784, 4096) or features[0].dtype != torch.float32:
                raise S3Error("BLOCKED_INPUT", f"Unexpected feature shape/dtype at event {position}")
            if not torch.isfinite(features[0]).all().item():
                raise S3Error("BLOCKED_INPUT", f"Nonfinite feature tensor at event {position}")
            feature_sha_before = hashlib.sha256(
                str((tuple(features[0].shape), str(features[0].dtype))).encode("ascii") +
                features[0].detach().to("cpu").contiguous().numpy().tobytes()).hexdigest()
            event_dynamic_before, event_dynamic_count_before = _empty_memory_identity(bank)

            adapter_started = time.perf_counter()
            adapter_map, adapter_ids, adapter_score, adapter_summary = _native_dense_score(
                impl, bank, image, features, cfg)
            torch.cuda.synchronize()
            summary["timings_seconds"]["adapter_scoring"] += time.perf_counter() - adapter_started

            native_started = time.perf_counter()
            native_score, native_amap, _ = score_bank(impl, bank, image, list(features), cfg)
            native_candidate = impl.layer_to_topk_from_amap(native_amap)
            native_ids = np.asarray(native_candidate.get(0, []), dtype=np.int64)
            torch.cuda.synchronize()
            summary["timings_seconds"]["native_score_bank"] += time.perf_counter() - native_started

            diagnostic_started = time.perf_counter()
            debug = bank.score_sample_correction_debug_maps(
                cls_name="p0", view_to_feats={"id": list(features)}, mode=0,
                views={"id": impl.ViewSpec("id", "id")},
                topk_patches=cfg.topk_patches, mode_scoring="boost",
                lam=0.8, tau_close=0.20, gamma=1.0,
                layer_fusion="mean", view_fusion="mean",
                img_agg=cfg.image_aggregation, img_topk=cfg.image_topk,
                img_quantile=cfg.image_quantile, concept_assign_view="id",
                img_hw=(image.height, image.width))
            torch.cuda.synchronize()
            summary["timings_seconds"]["diagnostic_map_scoring"] += time.perf_counter() - diagnostic_started

            base_map = validate_dense_map(np.asarray(adapter_map))
            adapter_ids = validate_top32(adapter_ids)
            native_ids = validate_top32(native_ids)
            debug_hw = np.asarray(debug["A_pre"])
            debug_ids = np.asarray(
                debug["debug"]["per_view"]["id"]["layer_summary"]["layer0"]["topk_patch_idx"],
                dtype=np.int64)
            debug_ids = validate_top32(debug_ids)
            grid_hw = list(debug["debug"]["grid_hw"])
            if grid_hw != [28, 28] or debug_hw.shape != (28, 28):
                raise S3Error("MAP_PARITY_FAILED", f"Native identity grid differs at event {position}")
            adapter_layer_ids = list(adapter_summary.keys())
            debug_view_ids = list(debug["debug"]["per_view"].keys())
            native_layer_count = len(debug["debug"]["per_view"]["id"]["layer_summary"])
            if (adapter_layer_ids != ["layer0"] or debug_view_ids != ["id"] or
                    native_layer_count != 1 or adapter_summary["layer0"].get("use_defect_bank") is not False or
                    adapter_summary["layer0"].get("corr_bank_size") != 0):
                raise S3Error("MAP_PARITY_FAILED", f"Layer/view metadata differs at event {position}")
            map_result = _map_comparison(base_map, debug_hw)
            top32_equal = bool(np.array_equal(adapter_ids, native_ids) and
                               np.array_equal(adapter_ids, debug_ids))
            adapter_score = float(adapter_score)
            native_score = float(native_score)
            score_delta = abs(adapter_score - native_score)
            scalar_tolerance = SCALAR_ATOL + SCALAR_RTOL * abs(native_score)
            scalar_tolerance_pass = score_delta <= scalar_tolerance
            scalar_exact = adapter_score == native_score
            max_reconstructed = max_aggregate(base_map)
            max_reconstruction_exact = adapter_score == max_reconstructed
            residual = np.asarray(debug["A_corr"], dtype=np.float32).reshape(-1, order="C")
            zero_reconstructed = zero_correction_reconstruction(base_map, residual)
            zero_correction_exact = bool(np.array_equal(zero_reconstructed, base_map))
            normalized_features = impl.l2_normalize(features[0].to(bank.device))
            selected_features = normalized_features[
                torch.as_tensor(adapter_ids, device=normalized_features.device,
                                dtype=torch.long)].detach().float().to("cpu").contiguous().numpy().astype(
                                    np.float32, copy=True)
            feature_sha_after = hashlib.sha256(
                str((tuple(features[0].shape), str(features[0].dtype))).encode("ascii") +
                features[0].detach().to("cpu").contiguous().numpy().tobytes()).hexdigest()
            source_features_unchanged = feature_sha_before == feature_sha_after
            event_dynamic_after, event_dynamic_count_after = _empty_memory_identity(bank)
            event_memory_unchanged = (
                event_dynamic_before == event_dynamic_after and
                event_dynamic_count_before == 0 and event_dynamic_count_after == 0)
            arrays = _event_arrays(
                base_map=base_map, debug_map=debug_hw,
                adapter_ids=adapter_ids, native_ids=native_ids,
                debug_ids=debug_ids, selected_features=selected_features)
            event_plan = preflight["event_plan"][position]
            event_meta = {
                "event_id": event_plan["event_id"],
                "position": position,
                "opaque_group_identity_sha256": event_plan["opaque_group_identity_sha256"],
                "relative_id_sha256": event_plan["relative_id_sha256"],
                "source_image_sha256": image_sha,
                "fixed_support_sha256": structured_before,
                "fixed_support_raw_prototype_sha256": raw_before,
                "pre_dynamic_memory_sha256": event_dynamic_before,
                "post_dynamic_memory_sha256": event_dynamic_after,
                "dynamic_memory_empty_and_unchanged": event_memory_unchanged,
                "encoder_fingerprint_sha256": summary["encoder_fingerprint_sha256"],
                "preprocess_fingerprint_sha256": summary["preprocess_fingerprint_sha256"],
                "scorer_profile_sha256": EXPECTED_R1_SCORER_PROFILE_SHA256,
                "feature_tensor_sha256": feature_sha_before,
                "feature_tensor_shape": [784, 4096],
                "feature_tensor_dtype": "torch.float32",
                "selected_feature_sha256": _array_digest(selected_features),
                "adapter_score": adapter_score,
                "native_score": native_score,
                "scalar_exact_equal": scalar_exact,
                "scalar_tolerance": scalar_tolerance,
                "scalar_tolerance_pass": scalar_tolerance_pass,
                "adapter_map_max": max_reconstructed,
                "adapter_score_equals_map_max": max_reconstruction_exact,
                "map_parity": map_result,
                "top32_exact_sequence_equal": top32_equal,
                "adapter_top32_ids_sha256": _array_digest(adapter_ids),
                "native_top32_ids_sha256": _array_digest(native_ids),
                "debug_top32_ids_sha256": _array_digest(debug_ids),
                "layer_view_metadata": {
                    "adapter_layers": adapter_layer_ids,
                    "native_layers": ["layer0"],
                    "adapter_view": "id",
                    "native_views": debug_view_ids,
                    "native_grid_hw": grid_hw,
                    "coordinate_order": "row-major identity patch grid",
                    "layer_fusion": "mean over one layer",
                    "view_fusion": "mean over identity view",
                    "interpolation_or_resizing": False,
                },
                "zero_correction_reconstruction_exact": zero_correction_exact,
                "source_feature_tensor_unchanged": source_features_unchanged,
            }
            serialization_started = time.perf_counter()
            event_receipt = _write_event(output_dir, arrays, event_meta)
            summary["timings_seconds"]["tape_serialization"] += time.perf_counter() - serialization_started
            summary["events_processed"] = position + 1
            summary["dense_map_exact_event_count"] += int(
                map_result["exact_elementwise_equality_count"] == 784 and
                map_result["exact_byte_equality"])
            summary["top32_exact_event_count"] += int(top32_equal)
            summary["scalar_exact_event_count"] += int(scalar_exact)
            summary["scalar_tolerance_event_count"] += int(scalar_tolerance_pass)
            summary["last_event_shard_sha256"] = event_receipt["shard_sha256"]
            _atomic_json(manifest_path, summary)
            if _directory_bytes(output_dir) > MAX_OUTPUT_BYTES:
                raise S3Error("RESOURCE_LIMIT_REACHED", "S3 output exceeded 128 MiB")
            if not source_features_unchanged or not zero_correction_exact:
                raise S3Error("MEMORY_IMMUTABILITY_FAILED",
                              f"Feature or zero-correction reconstruction changed at event {position}")
            if not event_memory_unchanged:
                raise S3Error("MEMORY_IMMUTABILITY_FAILED",
                              f"Dynamic normal memory changed at event {position}")
            if not (map_result["exact_elementwise_equality_count"] == 784 and
                    map_result["exact_byte_equality"]):
                raise S3Error("MAP_PARITY_FAILED", f"Dense-map parity failed at event {position}")
            if not top32_equal:
                raise S3Error("TOPK_PARITY_FAILED", f"Top-32 ordered IDs differ at event {position}")
            if not (scalar_tolerance_pass and max_reconstruction_exact):
                raise S3Error("SCALAR_PARITY_FAILED", f"Image-score parity failed at event {position}")
    except S3Error as exc:
        primary_failure = exc
    except Exception as exc:
        primary_failure = S3Error("INCOMPLETE", f"Replay error {type(exc).__name__}: {exc}")
    finally:
        summary["timings_seconds"]["stream_encoding"] = float(summary["timings_seconds"]["stream_encoding"])
        summary["total_gpu_subprocess_wall_seconds"] = time.monotonic() - started

    structured_after = str(_fixed_bank_digest(bank))
    raw_after = str(p0.fixed_bank_profile(bank)["sha256"])
    dynamic_after, dynamic_count_after = _empty_memory_identity(bank)
    post_corr_defect = _correction_defect_state(bank)
    summary["fixed_support_structured_sha256_after"] = structured_after
    summary["fixed_support_raw_sha256_after"] = raw_after
    summary["dynamic_memory_sha256_after"] = dynamic_after
    summary["dynamic_memory_count_after"] = dynamic_count_after
    summary["correction_defect_state_after"] = post_corr_defect
    summary["fixed_memory_unchanged"] = (structured_before == structured_after and raw_before == raw_after)
    summary["dynamic_memory_unchanged_and_empty"] = (dynamic_before == dynamic_after and
                                                       dynamic_count_after == 0)
    summary["correction_defect_state_unchanged"] = post_corr_defect == pre_corr_defect
    summary["total_gpu_subprocess_wall_seconds"] = time.monotonic() - started
    summary["output_bytes"] = _directory_bytes(output_dir)
    summary["resource_metrics"]["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    summary["resource_metrics"]["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    final_mem_ok = (summary["fixed_memory_unchanged"] and
                    summary["dynamic_memory_unchanged_and_empty"] and
                    summary["correction_defect_state_unchanged"])
    summary["gpu_execution_status"] = "GPU_REPLAY_COMPLETE_PENDING_TAPE_READBACK"
    if primary_failure is not None:
        summary["status"] = primary_failure.status
        summary["failure_message"] = str(primary_failure)
    elif not final_mem_ok:
        summary["status"] = "MEMORY_IMMUTABILITY_FAILED"
        summary["failure_message"] = "Fixed/dynamic/correction/defect state changed during scoring"
    elif summary["events_processed"] != MAX_EVENTS:
        summary["status"] = "INCOMPLETE"
        summary["failure_message"] = "Replay ended before all 16 authorized events"
    else:
        summary["status"] = "INCOMPLETE"
        summary["gpu_replay_parity_status"] = "ALL_16_PARITY_CHECKS_PASS_PENDING_TAPE_READBACK"
    summary["output_bytes"] = _directory_bytes(output_dir)
    _atomic_json(manifest_path, summary)
    cleanup = getattr(backbone, "cleanup", None)
    if callable(cleanup):
        cleanup()
    if summary["status"] not in ("INCOMPLETE",):
        return summary
    return summary


def run_preflight(args: argparse.Namespace) -> int:
    value = collect_preflight(
        expected_source_commit=args.expected_source_commit,
        profile_id=args.profile_id, run_id=args.run_id,
        authorization_id=args.authorization_id,
        dataset_root=Path(args.dataset_root))
    print(canonical_json(_safe_preflight_output(value)))
    return 0


def run_gpu(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    try:
        result = _run_gpu(
            expected_source_commit=args.expected_source_commit,
            run_id=args.run_id, authorization_id=args.authorization_id,
            dataset_root=Path(args.dataset_root), output_dir=output_dir)
        print(canonical_json({"status": result["status"],
                              "gpu_execution_status": result.get("gpu_execution_status"),
                              "events_processed": result.get("events_processed"),
                              "output_bytes": result.get("output_bytes")}))
        return 0 if result["status"] == "INCOMPLETE" and result.get("events_processed") == MAX_EVENTS else 2
    except S3Error as exc:
        if output_dir.is_dir() and (output_dir / "run_manifest.json").is_file():
            try:
                _update_manifest(output_dir / "run_manifest.json",
                                 {"status": exc.status, "failure_message": str(exc),
                                  "gpu_execution_status": "FAILED"})
            except Exception:
                pass
        print(canonical_json({"status": exc.status, "error": str(exc)}), file=sys.stderr)
        return 3
    except Exception as exc:
        if output_dir.is_dir() and (output_dir / "run_manifest.json").is_file():
            try:
                _update_manifest(output_dir / "run_manifest.json",
                                 {"status": "INCOMPLETE", "failure_type": type(exc).__name__,
                                  "failure_message": str(exc), "gpu_execution_status": "FAILED"})
            except Exception:
                pass
        print(canonical_json({"status": "INCOMPLETE", "error_type": type(exc).__name__,
                              "error": str(exc)}), file=sys.stderr)
        return 4


def run_verify(args: argparse.Namespace) -> int:
    try:
        print(canonical_json(verify_tape(Path(args.output_dir))))
        return 0
    except Exception as exc:
        print(canonical_json({"readback_integrity_pass": False,
                              "error_type": type(exc).__name__, "error": str(exc)}),
              file=sys.stderr)
        return 5


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run", "verify"), required=True)
    parser.add_argument("--profile-id", default=PROFILE_ID)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--run-id")
    parser.add_argument("--authorization-id")
    parser.add_argument("--dataset-root", default=r"G:\dataset\realiad")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    if args.mode == "verify":
        if not args.output_dir:
            parser.error("--output-dir is required for verify")
        return run_verify(args)
    for name in ("expected_source_commit", "run_id", "authorization_id"):
        if not getattr(args, name):
            parser.error(f"--{name.replace('_', '-')} is required for {args.mode}")
    try:
        _validate_profile_id(args.profile_id)
        if args.mode == "preflight":
            return run_preflight(args)
        if not args.output_dir:
            parser.error("--output-dir is required for run")
        return run_gpu(args)
    except S3Error as exc:
        print(canonical_json({"status": exc.status, "error": str(exc)}), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
