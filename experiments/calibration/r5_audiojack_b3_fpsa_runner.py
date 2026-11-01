"""R5-AUDIOJACK-B3-FPSA: one frozen, retrospective single-class mechanism test.

Stage A writes and seals all G/GC/GC-FPSA scores before any offline outcome
projection.  The only labels visible during the stream are queried feedback
disclosed after that event's score row has been durably flushed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PARENT = "dea81e097ec0bd5994eb5f59a2009c3936f320b6"
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
OUTPUT_BASE = ROOT / "outputs" / "calibration"
RUN_ID = "r5_audiojack_b3_fpsa_20261102_a1"
FAILED_SMOKE_ID = RUN_ID + "_smoke"
SMOKE_ID = RUN_ID + "_smoke_retry1"
R4_REFERENCE = OUTPUT_BASE / "r4_fpsa_audiojack_20261102_a1"
CATEGORY = "audiojack"
EVENTS = 3815
QUERIES = 114
FIRST_ANOMALOUS_QUERY = 1305
FPSA_MATCH_THROUGH = 1305
RUNTIME_SECONDS = 120 * 60
SMOKE_RUNTIME_SECONDS = 15 * 60
MIN_FREE_BYTES = 10 * 1024**3
OUTPUT_LIMIT = 4 * 1024**3
FAILURE_RESERVE = 64 * 1024
C64_COUNT = 64
B3_DELTA = 0.01
B3_MIN_NORMALS = 16
CACHE_PREFIX = "c64_cache_"
AUTHORIZATION = "R5-AUDIOJACK-B3-FPSA user-authorized single-class retrospective development mechanism validation"
RUN_SCHEMA = "r5.audiojack.b3_fpsa.run.v1"
SCORE_SCHEMA = "r5.audiojack.b3_fpsa.score.v1"
HISTORY_SCHEMA = "r5.audiojack.b3_fpsa.history.v1"
STAGE_A_SCHEMA = "r5.audiojack.b3_fpsa.stage_a_seal.v1"
STAGE_B_SCHEMA = "r5.audiojack.b3_fpsa.stage_b_seal.v1"
ATOL = 1e-5
RTOL = 1e-4
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261102
STAGE05 = OUTPUT_BASE / "stage05_exploratory_20261029"
R3_REFERENCE = OUTPUT_BASE / "r3_fullstream_2x2_20261102_a7"
DATASET_ROOT = Path(r"G:\dataset\realiad")
JSON_ROOT = Path(r"G:\dataset\realiad\jsons\realiad_jsons")
REPORT_DIR = ROOT / "docs" / "calibration" / "audit" / "r5_audiojack_b3_fpsa_20261102"
SOURCE_PATHS = (
    "experiments/calibration/r3_fullstream_2x2.py",
    "experiments/calibration/r1_matched_correction_pilot.py",
    "experiments/calibration/r2_exp2a_c1_strength.py",
    "experiments/calibration/r4_fpsa_audiojack_runner.py",
    "experiments/calibration/r4_fpsa_codebook.py",
    "experiments/calibration/r1_runtime_supervisor.py",
    "experiments/calibration/stage05.py",
    "experiments/calibration/stage1a_p12_label_provider.py",
    "experiments/calibration/profiles.py",
    "experiments/calibration/controllers.py",
    "experiments/calibration/protocol.py",
    "experiments/calibration/query_tape.py",
    "experiments/calibration/r2_feedback_repair.py",
    "experiments/p0_protocol_audit.py",
    "experiments/risk_safe/native_bridge.py",
    "experiments/v02_memory.py",
    "test_phase3_AIF_simplified_v22.py",
    "vmb_visionad_new.py",
)


class R5AudiojackError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    from experiments.calibration.r3_fullstream_2x2 import canonical_bytes
    return _sha256_bytes(canonical_bytes(value).rstrip(b"\n"))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R5AudiojackError(f"Expected JSON object at {path.name}")
    return value


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def _source_hashes() -> dict[str, str]:
    return {name: _sha256_file(ROOT / name) for name in SOURCE_PATHS}


def _repository_identity() -> dict[str, str]:
    branch, head = _git("branch", "--show-current"), _git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH or head.lower() != EXPECTED_PARENT:
        raise R5AudiojackError(
            f"Source identity mismatch: expected {EXPECTED_BRANCH}@{EXPECTED_PARENT}, "
            f"found {branch}@{head}")
    modified = _git("status", "--porcelain", "--", *SOURCE_PATHS)
    if modified:
        raise R5AudiojackError(f"Frozen dependency source files are dirty: {modified}")
    return {"branch": branch, "source_parent_commit": head}


def _source_binding(manifest: Mapping[str, Any]) -> None:
    _repository_identity()
    if (manifest.get("source_parent_commit") != EXPECTED_PARENT or
            manifest.get("branch") != EXPECTED_BRANCH or
            manifest.get("runner_source_sha256") != _sha256_file(Path(__file__).resolve()) or
            manifest.get("source_fingerprints") != _source_hashes()):
        raise R5AudiojackError("R5 source/branch identity differs from its immutable manifest")


def _output_path(run_id: str, *, existing: bool = False) -> Path:
    if run_id not in {RUN_ID, SMOKE_ID, FAILED_SMOKE_ID}:
        raise R5AudiojackError("Only the fixed R5 audiojack run and smoke IDs are authorized")
    parent = OUTPUT_BASE.resolve(strict=True)
    path = OUTPUT_BASE / run_id
    if existing:
        if path.is_symlink() or not path.is_dir() or path.resolve(strict=True).parent != parent:
            raise R5AudiojackError("Existing R5 output path is missing or escapes its namespace")
    else:
        if run_id == FAILED_SMOKE_ID:
            raise R5AudiojackError("A preserved failed R5 smoke namespace cannot be re-created")
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"R5 output exists; restart/resume is forbidden: {path.name}")
        if path.parent.resolve(strict=True) != parent:
            raise R5AudiojackError("R5 output path escaped outputs/calibration")
    return path


def _load_audiojack_contract(*, verify_images: bool) -> dict[str, Any]:
    """Load only audiojack frozen identities; never parse TEST labels here."""
    import gzip
    import numpy as np
    from experiments.calibration import stage05
    from experiments.calibration.controllers import theta0_from_c64
    from experiments.calibration.profiles import QueryProtocolConfig, canonical_json

    run = _read_json(STAGE05 / "run_manifest.json")
    frozen_path = STAGE05 / "frozen_inputs.json"
    frozen = _read_json(frozen_path)
    score_path = STAGE05 / "score_tape_manifest.json"
    score_manifest = _read_json(score_path)
    if (run.get("overall_status") != "EXPLORATORY_AUDIT_COMPLETE_NOT_STAGE1_VALIDATION" or
            run.get("confirmation_classes_opened") is not False or
            frozen.get("schema") != "stage05.frozen_inputs.v1" or
            frozen.get("freeze_verification_status") != "PASS" or
            frozen.get("stage1_or_stage2_run") is not False or
            score_manifest.get("status") != "PASS" or
            score_manifest.get("controller_or_test_label_access") is not False):
        raise R5AudiojackError("Frozen Stage 0.5 input receipt is incompatible")
    archive = STAGE05 / "query_tapes.json.gz"
    archive_sha = _sha256_file(archive)
    if archive_sha != run.get("query_tape_archive_sha256") or archive_sha != frozen.get("query_tapes_file_sha256"):
        raise R5AudiojackError("Frozen query tape archive hash mismatch")
    with gzip.open(archive, "rt", encoding="utf-8") as stream:
        tapes = json.load(stream)

    info = frozen.get("classes", {}).get(CATEGORY)
    if not isinstance(info, dict):
        raise R5AudiojackError("Frozen audiojack category receipt is absent")
    source_json = JSON_ROOT / "audiojack.json"
    if not source_json.is_file() or _sha256_file(source_json) != info.get("class_json_sha256"):
        raise R5AudiojackError("Audiojack public JSON fingerprint differs from Stage 0.5")
    train, stream_rows, class_sha = stage05.read_class_stream(
        DATASET_ROOT, JSON_ROOT, CATEGORY, test_labels=False)
    if class_sha != info.get("class_json_sha256") or len(stream_rows) != EVENTS or \
            int(info.get("stream_images", -1)) != EVENTS or len(info.get("stream", [])) != EVENTS:
        raise R5AudiojackError("Audiojack frozen stream does not contain exactly 3,815 events")
    # Reproduce only the authorized seed-0 support and its C64 calibration
    # bank. Do not calculate/read the other frozen support-seed identities.
    groups: dict[str, list[Any]] = {}
    for record in train:
        groups.setdefault(record.sample_key, []).append(record)
    normal_keys = sorted(key for key, views in groups.items()
                         if all(view.is_good for view in views))
    if len(normal_keys) < 68:
        raise R5AudiojackError("Audiojack has too few train-normal Source Groups for frozen C64")
    c64_keys = sorted(random.Random(stage05.HISTORY_SEED).sample(normal_keys, 64))
    c64_rows = [random.Random(stage05.SUPPORT_VIEW_SEED_OFFSET + index).choice(groups[key])
                for index, key in enumerate(c64_keys)]
    remaining = [key for key in normal_keys if key not in set(c64_keys)]
    support_keys = sorted(random.Random(0).sample(remaining, 4))
    support_view_rng = random.Random(stage05.SUPPORT_VIEW_SEED_OFFSET)
    support_rows = [support_view_rng.choice(groups[key]) for key in support_keys]
    support0 = {
        "seed": 0, "groups": support_keys,
        "images": [row.relative_id for row in support_rows],
        "group_sha256": stage05.opaque_digest("support-groups", support_keys),
        "image_sha256": stage05.opaque_digest(
            "support-images", [row.relative_id for row in support_rows]),
        "all_train_normal": all(row.is_good for row in support_rows),
        "c64_disjoint": not bool(set(support_keys) & set(c64_keys)),
    }
    c64 = {
        "groups": c64_keys,
        "images": [row.relative_id for row in c64_rows],
        "group_sha256": stage05.opaque_digest("c64-groups", c64_keys),
        "image_sha256": stage05.opaque_digest(
            "c64-images", [row.relative_id for row in c64_rows]),
        "count": len(c64_rows), "selection_seed": stage05.HISTORY_SEED,
    }
    if (support0 != info.get("supports", {}).get("0") or
            c64 != info.get("c64") or support0["c64_disjoint"] is not True or
            info.get("support_c64_disjoint") is not True or
            info.get("train_test_groups_disjoint") is not True):
        raise R5AudiojackError("Audiojack Support/C64 frozen selection or disjointness changed")
    frozen_rows = info["stream"]
    for position, (actual, expected) in enumerate(zip(stream_rows, frozen_rows)):
        if (actual.position != position or int(expected.get("position", -1)) != position or
                actual.relative_id != expected.get("relative_id") or
                actual.sample_key != expected.get("sample_key") or
                actual.opaque_group != expected.get("opaque_group") or
                actual.camera_id != int(expected.get("camera_id", -1))):
            raise R5AudiojackError(f"Audiojack public stream identity changed at {position}")
    tape = tapes[CATEGORY]["0.03"]["37"]
    expected_tape = frozen["tape_digests_by_seed37"][CATEGORY]["0.03"]
    tape_payload = {
        "schema": "calibration.query_tape.v1", "policy": "Q-A-TS",
        "config": asdict(QueryProtocolConfig()), "n_images": EVENTS,
        "n_groups": len({str(row.opaque_group) for row in stream_rows}),
        "requested_budget": int(tape["requested_budget"]), "early_budget": 0,
        "stream_digest": str(tape["stream_digest"]),
        "bin_edges": list(tape["bin_edges"]), "bin_quotas": list(tape["bin_quotas"]),
        "events": list(tape["events"]),
    }
    observed_tape = hashlib.sha256(canonical_json(tape_payload).encode("utf-8")).hexdigest()
    if tape.get("digest") != expected_tape or observed_tape != expected_tape:
        raise R5AudiojackError("Audiojack frozen Q-A-TS Seed 37 digest mismatch")
    positions: list[int] = []
    group_by_position: dict[int, str] = {}
    for event in tape.get("events", []):
        position = int(event.get("position", -1))
        if not 0 <= position < EVENTS or position in group_by_position:
            raise R5AudiojackError("Audiojack Q-A-TS positions are malformed/duplicated")
        row = stream_rows[position]
        identity = _sha256_bytes(("image-identity\0" + row.relative_id).encode("utf-8"))
        if (event.get("official_sample_key") != row.opaque_group or
                int(event.get("camera_id", -1)) != row.camera_id or
                event.get("relative_id") != identity):
            raise R5AudiojackError(f"Audiojack Q-A-TS identity mismatch at {position}")
        positions.append(position)
        group_by_position[position] = row.opaque_group
    expected_query_count = int(math.floor(0.03 * EVENTS + 0.5))
    if (expected_query_count != QUERIES or int(tape.get("requested_budget", -1)) != QUERIES or
            len(positions) != QUERIES or len(set(group_by_position.values())) != QUERIES):
        raise R5AudiojackError("Audiojack query count/group uniqueness differs from frozen 114")
    positions.sort()

    c64_row = score_manifest["classes"][CATEGORY]["score_tapes"]["0"]
    c64_path = (STAGE05 / str(c64_row["file"])).resolve(strict=True)
    if STAGE05.resolve(strict=True) not in c64_path.parents:
        raise R5AudiojackError("Audiojack C64 score tape escaped its frozen directory")
    if (int(c64_row.get("c64_score_count", -1)) != 64 or
            _sha256_file(c64_path) != c64_row.get("sha256") or
            c64_row.get("c64_group_sha256") != info["c64"].get("group_sha256")):
        raise R5AudiojackError("Audiojack C64 score tape fingerprint/identity failed")
    with np.load(c64_path, allow_pickle=False) as archive_data:
        c64_scores = np.asarray(archive_data["c64_scores"], dtype=np.float64)
        c64_identity = str(archive_data["c64_identity_sha256"].item())
    if (c64_scores.shape != (64,) or not np.isfinite(c64_scores).all() or
            c64_identity != info["c64"].get("image_sha256")):
        raise R5AudiojackError("Audiojack C64 score values or image identity are invalid")
    theta = float(theta0_from_c64(c64_scores.tolist()))
    sigma = max(float(c64_scores.std(ddof=1)), 0.02)

    support_groups = {_sha256_bytes(("official-sample-group\0" + str(x)).encode())
                      for x in support0["groups"]}
    c64_groups = {_sha256_bytes(("official-sample-group\0" + str(x)).encode())
                  for x in info["c64"]["groups"]}
    test_groups = {str(row.opaque_group) for row in stream_rows}
    if (len(support_groups) != 4 or len(c64_groups) != 64 or support_groups & c64_groups or
            support_groups & test_groups or c64_groups & test_groups):
        raise R5AudiojackError("Audiojack Support/C64/TEST Source Groups overlap")
    support_info = support0
    support_paths = [Path(stage05._absolute_image_path(
        DATASET_ROOT, info["prefix"], str(relative))) for relative in support_info["images"]]
    event_paths = [Path(row.image_path) for row in stream_rows]
    if verify_images and any(not path.is_file() for path in (*support_paths, *event_paths)):
        raise R5AudiojackError("An authorized audiojack Support/TEST image path is missing")
    return {
        "stream": stream_rows, "support_paths": support_paths, "event_paths": event_paths,
        "positions": tuple(positions), "query_groups": group_by_position,
        "theta": theta, "sigma": sigma, "class_json_sha256": class_sha,
        "query_tape_sha256": expected_tape,
        "query_positions_sha256": _sha256_json(positions),
        "c64_score_tape_sha256": str(c64_row["sha256"]),
        "fixed_support_sha256_expected": str(c64_row["fixed_support_sha256"]),
        "support_group_sha256": _sha256_json(sorted(support_groups)),
        "stage05_hashes": {
            "run_manifest_sha256": _sha256_file(STAGE05 / "run_manifest.json"),
            "frozen_inputs_sha256": _sha256_file(frozen_path),
            "score_tape_manifest_sha256": _sha256_file(score_path),
            "query_tape_archive_sha256": archive_sha,
        },
    }


def _load_r3_audiojack_reference(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Verify and load only the category-specific sealed R3 score prefix."""
    import experiments.calibration.r3_fullstream_2x2 as r3
    manifest = _read_json(R3_REFERENCE / "run_manifest.json")
    stage_a = _read_json(R3_REFERENCE / "stage_a_seal.json")
    stage_b = _read_json(R3_REFERENCE / "stage_b_seal.json")
    complete = _read_json(R3_REFERENCE / "complete_seal.json")
    seal_sha = _sha256_file(R3_REFERENCE / "stage_a_seal.json")
    stage_b_sha = _sha256_file(R3_REFERENCE / "stage_b_seal.json")
    category_seal = stage_a.get("category_seals", {}).get(CATEGORY, {})
    if (manifest.get("run_id") != "r3_fullstream_2x2_20261102_a7" or
            stage_a.get("status") != "SEALED_COMPLETE" or stage_b.get("status") != "SEALED_COMPLETE" or
            complete.get("status") != "COMPLETE" or complete.get("stage_a_seal_sha256") != seal_sha or
            complete.get("stage_b_seal_sha256") != stage_b_sha or
            category_seal.get("event_count") != EVENTS or
            category_seal.get("query_count") != QUERIES or
            category_seal.get("query_tape_sha256") != contract["query_tape_sha256"]):
        raise R5AudiojackError("Existing R3 audiojack reference lacks valid completion/category seals")
    path = R3_REFERENCE / "stage_a_scores.jsonl"
    refs: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8") as stream:
        for position in range(EVENTS):
            line = stream.readline()
            if not line:
                raise R5AudiojackError("R3 audiojack score prefix ended early")
            row = json.loads(line)
            if (row.get("schema") != r3.SCORE_SCHEMA or row.get("category") != CATEGORY or
                    int(row.get("position", -1)) != position or
                    any(key in row for key in ("label", "anomaly_class", "mask_path")) or
                    not {"G", "GC"}.issubset(row.get("scores", {}))):
                raise R5AudiojackError(f"R3 audiojack sealed score row {position} is invalid")
            refs.append(row)
            digest.update(r3.canonical_bytes(row))
    if digest.hexdigest() != category_seal.get("category_score_rows_sha256"):
        raise R5AudiojackError("R3 audiojack score prefix differs from its category seal")
    query_positions = [int(row["position"]) for row in refs if row.get("query_tape_decision")]
    if (query_positions != list(contract["positions"]) or
            any(str(row.get("source_group_sha256")) != contract["stream"][int(row["position"])].opaque_group
                for row in refs)):
        raise R5AudiojackError("R3 audiojack source-group/query identity differs from frozen contract")
    return refs


def _skip_json_value(data: bytes, index: int) -> int:
    from experiments.calibration.stage1a_p12_label_provider import _skip_value
    return _skip_value(data, index)


def _json_projection(data: bytes, keys: set[str]) -> dict[str, Any]:
    """Project selected JSON object members while lexically skipping all others."""
    from experiments.calibration.stage1a_p12_label_provider import _string, _ws
    index = _ws(data, 0)
    if index >= len(data) or data[index] != ord("{"):
        raise R5AudiojackError("R3 history record is not a JSON object")
    index += 1
    out: dict[str, Any] = {}
    while True:
        index = _ws(data, index)
        if index >= len(data):
            raise R5AudiojackError("Truncated R3 history object")
        if data[index] == ord("}"):
            return out
        key, index = _string(data, index)
        index = _ws(data, index)
        if index >= len(data) or data[index] != ord(":"):
            raise R5AudiojackError("Malformed R3 history member")
        start = _ws(data, index + 1)
        stop = _skip_json_value(data, start)
        if key in keys:
            out[key] = json.loads(data[start:stop].decode("utf-8"))
        index = _ws(data, stop)
        if index < len(data) and data[index] == ord(","):
            index += 1
            continue
        if index < len(data) and data[index] == ord("}"):
            return out
        raise R5AudiojackError("Malformed R3 history object delimiter")


class _R3WriterCursor:
    """Read audiojack writer receipts causally, without decoding label fields."""
    _QUERY_KEYS = {"record_type", "category", "position", "event_id_sha256",
                   "source_group_sha256", "greedy_writer_action"}
    _COMPLETE_KEYS = {"record_type", "category", "events_processed", "query_events_processed",
                      "greedy_write_events", "greedy_accepted_patch_count",
                      "greedy_skipped_patch_count"}

    def __init__(self, path: Path):
        self.stream = Path(path).open("rb")
        self.finished = False

    def _next(self) -> dict[str, Any]:
        line = self.stream.readline()
        if not line:
            raise R5AudiojackError("R3 audiojack writer reference ended unexpectedly")
        head = _json_projection(line, {"record_type", "category"})
        kind = head.get("record_type")
        keys = self._QUERY_KEYS if kind == "queried_feedback_and_updates" else self._COMPLETE_KEYS
        return _json_projection(line, keys)

    def query(self, position: int, event_id: str, group_sha: str) -> dict[str, Any]:
        while True:
            row = self._next()
            kind, category = row.get("record_type"), row.get("category")
            if category not in (None, CATEGORY):
                raise R5AudiojackError("R3 audiojack writer reference crossed into another category")
            if kind == "queried_feedback_and_updates":
                if (category != CATEGORY or int(row.get("position", -1)) != position or
                        row.get("event_id_sha256") != event_id or
                        row.get("source_group_sha256") != group_sha or
                        not isinstance(row.get("greedy_writer_action"), dict)):
                    raise R5AudiojackError(f"R3 writer receipt identity mismatch at {position}")
                return row["greedy_writer_action"]
            if kind == "category_complete" and category == CATEGORY:
                raise R5AudiojackError("R3 writer reference completed before a required query")

    def finish(self) -> dict[str, Any]:
        while True:
            row = self._next()
            if row.get("record_type") == "category_complete" and row.get("category") == CATEGORY:
                self.finished = True
                self.stream.close()
                return row
            if row.get("category") not in (None, CATEGORY):
                raise R5AudiojackError("R3 writer reference crossed category before audiojack receipt")


def _writer_parity(actual: Mapping[str, Any], expected: Mapping[str, Any], *, position: int) -> None:
    import experiments.calibration.r3_fullstream_2x2 as r3
    for key in ("input_score_source", "prediction", "candidate_ids_sha256",
                "candidate_features_sha256", "action", "eligibility_reason",
                "accepted_patches_delta", "skipped_patches_delta"):
        if actual.get(key) != expected.get(key):
            raise R5AudiojackError(f"Greedy writer parity mismatch at {position} field {key}")
    for key in ("score", "theta", "sigma"):
        if key in expected and not r3._score_tolerance(actual.get(key), expected[key]):
            raise R5AudiojackError(f"Greedy writer parity mismatch at {position} field {key}")


class R5OutputBudget:
    """R5's pre-write budget includes the explicitly named immutable C64 cache files."""

    STANDARD_FILES = {
        "budget.lock", "run_manifest.json", "stage_a_scores.jsonl",
        "attempt_history.jsonl", "c64_cache_manifest.json", "stage_a_seal.json",
        "stage_b_seal.json", "aggregate_results.json", "complete_seal.json",
        "supervisor_status.json", "failure.json",
    }

    def __init__(self, root: Path, *, create: bool = False):
        import experiments.calibration.r2_feedback_repair as repair
        self.repair = repair
        self.root = Path(os.path.abspath(root))
        self.parent = self.root.parent
        self.max_bytes = OUTPUT_LIMIT
        self.reserve = FAILURE_RESERVE
        self.max_files = 96
        self.created: set[str] = set()
        self.file_sizes: dict[str, int] = {}
        self.used_bytes = 0
        if create:
            repair._reject_reparse_chain(self.parent)
            if self.root.parent.resolve(strict=True) != self.parent.resolve(strict=True):
                raise R5AudiojackError("R5 output parent identity changed")
            self.root.mkdir(parents=False, exist_ok=False)
            with (self.root / "budget.lock").open("xb") as stream:
                stream.write(b"\0")
                stream.flush()
                os.fsync(stream.fileno())
        self._validate_root()
        self._refresh()

    def _allowed(self, name: str) -> bool:
        return (name in self.STANDARD_FILES or
                re.fullmatch(r"c64_cache_[0-9]{3}\.npz", name) is not None)

    def _validate_root(self) -> None:
        self.repair._reject_reparse_chain(self.parent)
        if (not self.root.is_dir() or self.repair._is_reparse(self.root) or
                self.root.resolve(strict=True).parent != self.parent.resolve(strict=True)):
            raise R5AudiojackError("R5 output directory is missing, redirected or escaped")

    def _inventory(self) -> tuple[dict[str, int], int]:
        sizes: dict[str, int] = {}
        used = 0
        for path in self.root.iterdir():
            if (self.repair._is_reparse(path) or not path.is_file() or
                    not self._allowed(path.name)):
                raise R5AudiojackError(f"Unaccounted R5 output path: {path.name}")
            size = path.stat().st_size
            sizes[path.name] = size
            used += size
        if len(sizes) > self.max_files or used > self.max_bytes:
            raise R5AudiojackError("R5 output exceeds its file or byte budget")
        return sizes, used

    def _refresh(self, *, allowed_change: str | None = None) -> None:
        sizes, used = self._inventory()
        names = set(sizes)
        if self.created:
            added_or_removed = names.symmetric_difference(self.created)
            resized = {name for name in names.intersection(self.created)
                       if sizes[name] != self.file_sizes[name]}
            if (added_or_removed or resized) and (
                    not allowed_change or added_or_removed - {allowed_change} or
                    resized - {allowed_change}):
                raise R5AudiojackError("Concurrent or unaccounted R5 output mutation detected")
        self.created, self.file_sizes, self.used_bytes = names, sizes, used

    @contextmanager
    def _locked(self):
        self._validate_root()
        lock = self.root / "budget.lock"
        if self.repair._is_reparse(lock) or not lock.is_file():
            raise R5AudiojackError("R5 budget lock is missing or unsafe")
        with lock.open("r+b", buffering=0) as stream:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise R5AudiojackError("Concurrent R5 output writer holds the lock") from exc
                try:
                    yield
                finally:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _write(self, name: str, payload: bytes, *, append: bool,
               failure_reserved: bool = False) -> None:
        if (not self._allowed(name) or name == "budget.lock" or
                (failure_reserved and name != "failure.json")):
            raise R5AudiojackError(f"R5 output name is not authorized: {name}")
        with self._locked():
            self._refresh()
            path = self.root / name
            if self.repair._is_reparse(path) or path.parent.resolve(strict=True) != self.root.resolve(strict=True):
                raise R5AudiojackError("R5 write path is unsafe")
            exists = path.exists()
            if not append and exists:
                raise R5AudiojackError(f"R5 output overwrite is forbidden: {name}")
            if append and (not exists or name not in self.created):
                raise R5AudiojackError(f"R5 append target is not registered: {name}")
            if not exists and len(self.created) + 1 > (self.max_files if failure_reserved else self.max_files - 1):
                raise R5AudiojackError("R5 output file budget exhausted before write")
            ceiling = self.max_bytes if failure_reserved else self.max_bytes - self.reserve
            if self.used_bytes + len(payload) > ceiling:
                raise R5AudiojackError("R5 output byte budget exhausted before write")
            mode = "ab" if append else "xb"
            with path.open(mode) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            self._refresh(allowed_change=name)

    def write_once(self, name: str, payload: bytes, *, failure_reserved: bool = False) -> None:
        self._write(name, payload, append=False, failure_reserved=failure_reserved)

    def append(self, name: str, payload: bytes) -> None:
        self._write(name, payload, append=True)

    def write_json(self, name: str, value: Any, *, failure_reserved: bool = False) -> None:
        import experiments.calibration.r3_fullstream_2x2 as r3
        self.write_once(name, r3.canonical_bytes(value), failure_reserved=failure_reserved)

    def audit(self) -> dict[str, Any]:
        self._refresh()
        return {"max_bytes": self.max_bytes, "failure_reserve_bytes": self.reserve,
                "used_bytes": self.used_bytes, "max_files": self.max_files,
                "file_count": len(self.created), "files": dict(sorted(self.file_sizes.items()))}


def _cache_encode(image: Any, features: Sequence[Any]) -> tuple[bytes, dict[str, Any]]:
    import numpy as np
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    arrays: dict[str, Any] = {"rgb_uint8": rgb}
    feature_receipts = []
    for index, tensor in enumerate(features):
        value = tensor.detach().to("cpu").contiguous().numpy()
        arrays[f"feature_{index}"] = value
        feature_receipts.append({"index": index, "shape": list(value.shape),
                                 "dtype": str(value.dtype)})
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    payload = buffer.getvalue()
    return payload, {"rgb_shape": list(rgb.shape), "feature_receipts": feature_receipts,
                     "payload_sha256": _sha256_bytes(payload), "payload_bytes": len(payload)}


def _cache_decode(payload: bytes) -> dict[str, Any]:
    import numpy as np
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        rgb = np.asarray(archive["rgb_uint8"], dtype=np.uint8).copy()
        features = [np.asarray(archive[key]).copy()
                    for key in sorted((key for key in archive.files if key.startswith("feature_")),
                                      key=lambda value: int(value.split("_")[-1]))]
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not features:
        raise R5AudiojackError("C64 cached native input/feature representation is malformed")
    if any(not np.isfinite(feature).all() for feature in features):
        raise R5AudiojackError("C64 cached native features are nonfinite")
    return {"rgb": rgb, "features": features}


def _c64_evidence(scores: Sequence[float], version: int, group_sha256: str,
                  scorer_state_sha256: str):
    from experiments.calibration.protocol import EvidenceAnchor, EvidenceSnapshot
    import math
    if (len(scores) != C64_COUNT or version < 0 or
            any(not math.isfinite(float(value)) for value in scores) or
            not re.fullmatch(r"[0-9a-f]{64}", str(group_sha256)) or
            not re.fullmatch(r"[0-9a-f]{64}", str(scorer_state_sha256))):
        raise R5AudiojackError("B3 evidence must contain exactly 64 finite current-state scores")
    anchors = tuple(EvidenceAnchor(
        opaque_anchor_id=_sha256_bytes(f"r5-c64-{index}-{group_sha256}".encode()),
        score=float(value), memory_version=int(version), protected=True,
        source_exclusion_policy="protected_no_selfwrite")
        for index, value in enumerate(scores))
    return EvidenceSnapshot(
        memory_version=int(version), normal_anchors=anchors, defect_anchors=(),
        source_exclusion_audit={"protected_c64_only": True,
                                "c64_group_sha256": group_sha256,
                                "current_scorer_state_version": int(version),
                                "current_scorer_state_sha256": scorer_state_sha256,
                                "c64_score_vector_sha256": _sha256_json(
                                    [float(value) for value in scores])},
        evidence_revision=int(version))


def _initialize_b3(controller: Any, theta0: float, scores: Sequence[float],
                   group_sha256: str, scorer_state_sha256: str) -> dict[str, Any]:
    if len(scores) != C64_COUNT:
        raise R5AudiojackError("Version-zero B3 initialization lacks all 64 C64 scores")
    before = float(controller.theta)
    snapshot = _c64_evidence(scores, 0, group_sha256, scorer_state_sha256)
    controller.on_memory_change(snapshot)
    after = float(controller.finalize_step())
    if (before != theta0 or after != theta0 or controller.evidence_version != 0 or
            controller.status != "CURRENT_C64_Q90" or
            tuple(controller.normal_scores) != tuple(float(value) for value in scores) or
            snapshot.source_exclusion_audit["current_scorer_state_sha256"] != scorer_state_sha256):
        raise R5AudiojackError("B3 version-zero initialization moved theta or failed evidence validation")
    controller._r5_state_digest = scorer_state_sha256
    return {"initial_theta": before, "theta_after_initialization": after,
            "evidence_version": int(controller.evidence_version),
            "status": str(controller.status), "movement": after - before}


def _refresh_b3(controller: Any, *, previous_digest: str, new_digest: str,
                previous_version: int, new_scores: Sequence[float],
                c64_group_sha256: str, new_state_digest: str) -> dict[str, Any] | None:
    if new_digest != new_state_digest:
        raise R5AudiojackError("B3 C64 score snapshot is bound to a different scorer-state fingerprint")
    if previous_digest == new_digest:
        return None
    if len(new_scores) != C64_COUNT or controller.evidence_version != previous_version:
        raise R5AudiojackError("B3 refresh has stale version or incomplete C64 scores")
    old_theta = float(controller.theta)
    new_version = previous_version + 1
    snapshot = _c64_evidence(new_scores, new_version, c64_group_sha256,
                              new_state_digest)
    controller.on_memory_change(snapshot)
    theta = float(controller.finalize_step())
    if (controller.evidence_version != new_version or not math.isfinite(theta) or
            tuple(controller.normal_scores) != tuple(float(value) for value in new_scores) or
            snapshot.source_exclusion_audit["current_scorer_state_sha256"] != new_state_digest or
            abs(theta - old_theta) > B3_DELTA + 1e-12):
        raise R5AudiojackError("B3 current-state update failed version or delta contract")
    controller._r5_state_digest = new_state_digest
    raw_q90 = sorted(float(value) for value in new_scores)[math.ceil(0.90 * 63)]
    return {"version_before": previous_version, "version_after": new_version,
            "old_theta": old_theta, "raw_c64_q90": raw_q90, "new_theta": theta,
            "movement": theta - old_theta,
            "binding": ("LOWER" if raw_q90 < old_theta - B3_DELTA else
                        "UPPER" if raw_q90 > old_theta + B3_DELTA else "NONE"),
            "evidence_fingerprint": _sha256_json([float(x) for x in new_scores])}


def _changed_scorers(before: Mapping[str, str], after: Mapping[str, str]) -> tuple[str, ...]:
    required = {"G", "GC", "GC-FPSA"}
    if set(before) != required or set(after) != required or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(value))
            for value in (*before.values(), *after.values())):
        raise R5AudiojackError("Scorer state identity set is incomplete or malformed")
    return tuple(name for name in ("G", "GC", "GC-FPSA")
                 if before[name] != after[name])


def _predict_b3(controller: Any, *, position: int, event_id_sha256: str,
                camera_id: int, score: float, version: int,
                state_digest: str) -> dict[str, Any]:
    from experiments.calibration.protocol import Observation
    if (int(controller.evidence_version) != int(version) or
            not re.fullmatch(r"[0-9a-f]{64}", str(state_digest)) or
            getattr(controller, "_r5_state_digest", None) != state_digest):
        raise R5AudiojackError("B3 prediction state version/fingerprint mismatch")
    observation = Observation(
        t=int(position), observation_id=str(event_id_sha256),
        camera_id=int(camera_id), score=float(score),
        memory_version=int(version), score_state_digest=str(state_digest))
    prediction = controller.predict(observation)
    expected = int(float(score) >= float(controller.theta))
    if (prediction.label != expected or prediction.threshold != controller.theta or
            not prediction.available):
        raise R5AudiojackError("Original B3 prediction violated score >= theta or availability")
    return {"prediction": int(prediction.label), "threshold": float(prediction.threshold),
            "status": str(prediction.reason or controller.status),
            "scorer_state_version": int(version), "scorer_state_sha256": state_digest}


def _frozen_writer_gate(c0_score: float, theta_c64: float) -> int:
    """The only prediction supplied to the shared writer is original frozen C0."""
    return int(float(c0_score) >= float(theta_c64))


def _assert_c64_source_groups_excluded(c64_group_sha256: str,
                                       correction_source_groups: Mapping[str, Mapping[int, set[str]]],
                                       scorer: str) -> None:
    groups = correction_source_groups.get(scorer)
    if groups is None:
        raise R5AudiojackError(f"Missing correction provenance for {scorer}")
    known = {value for per_slot in groups.values() for value in per_slot}
    if c64_group_sha256 in known:
        raise R5AudiojackError(f"Protected C64 Source Group entered {scorer} correction provenance")


class _ScoreDisclosureGate:
    """Runtime guard: labels can be requested only after the current score row flushes."""
    def __init__(self, position: int):
        self.position = int(position)
        self.flushed = False
        self.disclosed = False

    def mark_score_flushed(self, position: int) -> None:
        if int(position) != self.position or self.flushed or self.disclosed:
            raise R5AudiojackError("Invalid or repeated R5 score-flush transition")
        self.flushed = True

    def authorize_disclosure(self, position: int) -> None:
        if int(position) != self.position or not self.flushed or self.disclosed:
            raise R5AudiojackError("R5 query label requested before score seal or more than once")
        self.disclosed = True


def _defer_b3_transition(position: int, arm: str, state_digest: str,
                         transition: Mapping[str, Any]) -> dict[str, Any]:
    if arm not in {"G", "GC", "GC-FPSA"} or not re.fullmatch(
            r"[0-9a-f]{64}", str(state_digest)):
        raise R5AudiojackError("Invalid R5 B3 transition identity")
    target = int(position) + 1
    if target <= int(position) or transition.get("new_theta") is None:
        raise R5AudiojackError("R5 B3 transition is not eligible for the next event")
    return {**dict(transition), "apply_from_position": target,
            "scorer": arm, "scorer_state_sha256": state_digest,
            "c64_score_sha256": transition["evidence_fingerprint"]}


def _load_r4_audiojack_reference(contract: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import experiments.calibration.r3_fullstream_2x2 as r3
    import experiments.calibration.r4_fpsa_audiojack_runner as r4
    manifest_path = R4_REFERENCE / "run_manifest.json"
    stage_a_path = R4_REFERENCE / "stage_a_seal.json"
    stage_b_path = R4_REFERENCE / "stage_b_seal.json"
    complete_path = R4_REFERENCE / "complete_seal.json"
    scores_path = R4_REFERENCE / "stage_a_scores.jsonl"
    manifest = _read_json(manifest_path)
    stage_a, stage_b, complete = (_read_json(stage_a_path), _read_json(stage_b_path),
                                  _read_json(complete_path))
    aggregate_path = R4_REFERENCE / "aggregate_results.json"
    if (manifest.get("run_id") != "r4_fpsa_audiojack_20261102_a1" or
            manifest.get("category") != CATEGORY or
            manifest.get("source_parent_commit") != r4.EXPECTED_PARENT or
            manifest.get("runner_source_sha256") != _sha256_file(
                ROOT / "experiments/calibration/r4_fpsa_audiojack_runner.py") or
            manifest.get("source_fingerprints") != r4._source_hashes() or
            stage_a.get("status") != "SEALED_COMPLETE" or
            stage_a.get("run_id") != manifest.get("run_id") or
            stage_a.get("event_count") != EVENTS or
            stage_a.get("category") != CATEGORY or
            stage_a.get("score_rows_sha256") != _sha256_file(scores_path) or
            stage_b.get("status") != "SEALED_COMPLETE" or
            complete.get("status") != "COMPLETE" or
            complete.get("stage_a_seal_sha256") != _sha256_file(stage_a_path) or
            complete.get("stage_b_seal_sha256") != _sha256_file(stage_b_path) or
            complete.get("aggregate_results_sha256") != _sha256_file(aggregate_path)):
        raise R5AudiojackError("R4 audiojack sealed source reference failed identity/checksum verification")
    if (stage_a.get("query_tape_sha256") != contract.get("query_tape_sha256") or
            stage_a.get("query_count") != QUERIES or
            stage_a.get("unqueried_test_labels_accessed") is not False or
            stage_a.get("test_masks_opened") is not False):
        raise R5AudiojackError("R4 Stage A seal does not match the frozen audiojack query protocol")
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    with scores_path.open("r", encoding="utf-8") as stream:
        for position, line in enumerate(stream):
            row = json.loads(line)
            if (row.get("schema") != r4.SCORE_SCHEMA or row.get("category") != CATEGORY or
                    int(row.get("position", -1)) != position or
                    any(key in row for key in ("label", "anomaly_class", "mask_path"))):
                raise R5AudiojackError(f"R4 reference score row {position} is invalid or label-bearing")
            rows.append(row)
            digest.update(r3.canonical_bytes(row))
    if (len(rows) != EVENTS or digest.hexdigest() != stage_a.get("category_score_rows_sha256") or
            stage_a.get("score_rows_sha256") != _sha256_file(scores_path)):
        raise R5AudiojackError("R4 audiojack score rows do not reproduce the sealed 3,815-row digest")
    for position, (row, meta) in enumerate(zip(rows, contract["stream"])):
        if (row.get("source_group_sha256") != meta.opaque_group or
                bool(row.get("query_tape_decision")) != (position in set(contract["positions"]))):
            raise R5AudiojackError(f"R4 event/query identity differs from frozen audiojack at {position}")
    completion = None
    history_path = R4_REFERENCE / "attempt_history.jsonl"
    with history_path.open("rb") as stream:
        for line in stream:
            head = r4._json_projection(line, {"record_type", "category"})
            if head.get("record_type") == "category_complete" and head.get("category") == CATEGORY:
                completion = r4._json_projection(line, {
                    "record_type", "category", "events_processed", "query_events_processed",
                    "query_normal", "query_anomalous", "first_anomalous_query_position",
                    "fixed_support_sha256", "greedy_write_events", "greedy_accepted_patch_count",
                    "greedy_skipped_patch_count", "correction_final_sign_occupancy",
                    "fpsa_final_diagnostics", "parity", "greedy_reference_parity",
                    "loaded_model_state_sha256", "hub_source_tree_sha256", "checkpoint_sha256",
                    "runtime_profile", "actual_stream_image_encodings", "wall_seconds",
                })
                break
    if not completion or (completion.get("events_processed") != EVENTS or
            completion.get("query_events_processed") != QUERIES or
            completion.get("fixed_support_sha256") != manifest.get("fixed_support_sha256")):
        raise R5AudiojackError("R4 final audiojack scorer receipt is missing or incompatible")
    final_occ = completion["correction_final_sign_occupancy"]
    fpsa_diag = completion["fpsa_final_diagnostics"]
    if (final_occ.get("GC") != {"positive": 1, "negative": 255, "zero": 0, "total": 256} or
            final_occ.get("GC-FPSA") != {"positive": 91, "negative": 165, "zero": 0, "total": 256} or
            fpsa_diag.get("quota_borrow_opportunities") != 365 or
            fpsa_diag.get("quota_cross_sign_replacements") != 93 or
            fpsa_diag.get("quota_borrow_rejected") != 272):
        raise R5AudiojackError("R4 frozen final correction/FPSA mechanism receipt differs from task")
    return rows, {"manifest": manifest, "stage_a": stage_a, "completion": completion,
                  "stage_a_sha256": _sha256_file(stage_a_path)}


def _load_c64_inputs(contract: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    from experiments.calibration import stage05
    frozen_path = STAGE05 / "frozen_inputs.json"
    frozen = _read_json(frozen_path)
    info = frozen.get("classes", {}).get(CATEGORY)
    if not isinstance(info, dict):
        raise R5AudiojackError("Frozen audiojack C64 input receipt is absent")
    c64 = info.get("c64", {})
    if (c64.get("count") != C64_COUNT or len(c64.get("images", [])) != C64_COUNT or
            len(c64.get("groups", [])) != C64_COUNT or
            len(set(c64["groups"])) != C64_COUNT):
        raise R5AudiojackError("Protected audiojack C64 input is not exactly 64 unique groups/images")
    paths = [Path(stage05._absolute_image_path(
        DATASET_ROOT, info["prefix"], str(relative))) for relative in c64["images"]]
    if any(not path.is_file() for path in paths):
        raise R5AudiojackError("A protected audiojack C64 image is unavailable")
    score_manifest = _read_json(STAGE05 / "score_tape_manifest.json")
    receipt = score_manifest["classes"][CATEGORY]["score_tapes"]["0"]
    c64_path = (STAGE05 / str(receipt["file"])).resolve(strict=True)
    if STAGE05.resolve(strict=True) not in c64_path.parents:
        raise R5AudiojackError("Frozen audiojack C64 score path escaped Stage 0.5")
    if (_sha256_file(c64_path) != receipt.get("sha256") or
            receipt.get("c64_group_sha256") != c64.get("group_sha256") or
            receipt.get("c64_score_count") != C64_COUNT):
        raise R5AudiojackError("Frozen audiojack C64 score tape failed its identity seal")
    with np.load(c64_path, allow_pickle=False) as archive:
        scores = np.asarray(archive["c64_scores"], dtype=np.float64).copy()
        image_identity = str(archive["c64_identity_sha256"].item())
    if (scores.shape != (C64_COUNT,) or not np.isfinite(scores).all() or
            image_identity != c64.get("image_sha256") or
            _sha256_file(STAGE05 / "run_manifest.json") !=
            contract["stage05_hashes"]["run_manifest_sha256"]):
        raise R5AudiojackError("Frozen audiojack C64 scores or source identity are invalid")
    group_sha = [_sha256_bytes(("official-sample-group\0" + str(value)).encode("utf-8"))
                 for value in c64["groups"]]
    if len(set(group_sha)) != C64_COUNT or set(group_sha) & {
            str(row.opaque_group) for row in contract["stream"]}:
        raise R5AudiojackError("Protected C64 Source Groups overlap the audiojack TEST stream")
    return {"paths": paths, "group_sha256": group_sha, "scores": scores,
            "score_tape_sha256": str(receipt["sha256"]),
            "image_sha256": str(c64["image_sha256"]),
            "group_sha256_digest": str(c64["group_sha256"]),
            "c64_count": C64_COUNT}


def _build_c64_cache(output: Path, budget: R5OutputBudget, impl: Any,
                     backbone: Any, cfg: Any, c64: Mapping[str, Any], p0: Any
                     ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    cache: list[dict[str, Any]] = []
    receipts = []
    started = time.perf_counter()
    for index, path in enumerate(c64["paths"]):
        image, features = p0.encode_query(impl, backbone, str(path), cfg)
        if (len(features) != 1 or tuple(features[0].shape) != (784, 4096) or
                features[0].dtype != __import__("torch").float32 or
                not __import__("torch").isfinite(features[0]).all().item()):
            raise R5AudiojackError(f"C64 native feature contract failed at reference {index}")
        payload, detail = _cache_encode(image, features)
        name = f"{CACHE_PREFIX}{index:03d}.npz"
        budget.write_once(name, payload)
        disk = _cache_decode((output / name).read_bytes())
        memory = _cache_decode(payload)
        if (not np.array_equal(disk["rgb"], memory["rgb"]) or
                len(disk["features"]) != len(features) or
                any(not np.array_equal(disk["features"][i], features[i].detach().cpu().numpy())
                    for i in range(len(features)))):
            raise R5AudiojackError(f"C64 input/feature cache round-trip changed bytes at {index}")
        disk["cache_file"] = name
        disk["cache_sha256"] = detail["payload_sha256"]
        disk["source_group_sha256"] = c64["group_sha256"][index]
        cache.append(disk)
        receipts.append({"index": index, "file": name,
                         "file_sha256": detail["payload_sha256"],
                         "file_bytes": detail["payload_bytes"],
                         "feature_receipts": detail["feature_receipts"],
                         "source_group_sha256": c64["group_sha256"][index]})
        del image, features, payload
    manifest = {
        "schema": "r5.audiojack.c64_feature_cache.v1", "category": CATEGORY,
        "protected_train_normal_count": C64_COUNT,
        "c64_group_sha256": c64["group_sha256_digest"],
        "c64_image_sha256": c64["image_sha256"],
        "encoded_once_per_image": True, "cache_receipts": receipts,
        "cache_creation_seconds": time.perf_counter() - started,
        "test_labels_or_masks_accessed": False,
    }
    budget.write_json("c64_cache_manifest.json", manifest)
    return cache, manifest


def _score_c64_base(impl: Any, bank: Any, item: Mapping[str, Any], cfg: Any,
                    r1: Any) -> tuple[float, np.ndarray, np.ndarray, Any]:
    import numpy as np
    import torch
    from PIL import Image
    image = Image.fromarray(np.asarray(item["rgb"], dtype=np.uint8), mode="RGB")
    features = [torch.from_numpy(np.ascontiguousarray(value)).to(bank.device)
                for value in item["features"]]
    base_map, top_ids, base_score, _ = r1._native_dense_score(
        impl, bank, image, features, cfg)
    return float(base_score), base_map, top_ids, None


def _score_c64_corrected(impl: Any, bank: Any, item: Mapping[str, Any],
                         base: tuple[float, np.ndarray, np.ndarray, Any],
                         cfg_arm: Any, cb: Any,
                         source_groups: Mapping[int, set[str]],
                         position: int, r1: Any, r4: Any) -> float:
    import torch
    from experiments.calibration.r1_matched_correction_pilot import corrected_dense_map, max_score
    _base_score, base_map, top_ids, _ = base
    features = torch.from_numpy(item["features"][0]).to(bank.device)
    selected = impl.l2_normalize(features)[
        torch.as_tensor(top_ids, device=bank.device, dtype=torch.long)]
    delta, _ = r4._score_with_codebook(
        impl, bank, cfg_arm, cb, source_groups, selected,
        str(item["source_group_sha256"]), position, r1)
    return float(max_score(corrected_dense_map(base_map, top_ids, delta)))


def _scorer_state_digests(bank: Any, gc_cb: Any, fpsa_cfg: Any, fpsa_cb: Any,
                          gc_groups: Mapping[int, set[str]],
                          fpsa_groups: Mapping[int, set[str]],
                          fixed_support_sha256: str, r1: Any, r4: Any) -> dict[str, str]:
    greedy = r1._memory_content_sha256(bank)
    gc = r1._correction_state_sha256(bank, gc_groups)
    fpsa = r4._correction_state_sha256(bank, fpsa_cfg, fpsa_cb, fpsa_groups, r1)
    return {
        "G": _sha256_json({"fixed_support_sha256": fixed_support_sha256,
                           "greedy_state_sha256": greedy}),
        "GC": _sha256_json({"fixed_support_sha256": fixed_support_sha256,
                            "greedy_state_sha256": greedy,
                            "correction_state_sha256": gc}),
        "GC-FPSA": _sha256_json({"fixed_support_sha256": fixed_support_sha256,
                                 "greedy_state_sha256": greedy,
                                 "correction_state_sha256": fpsa}),
    }


def _existing_output_path(run_id: str) -> Path:
    return _output_path(run_id, existing=True)


def _verify_child_manifest(run_id: str, token: str, expected_mode: str):
    import experiments.calibration.r3_fullstream_2x2 as r3
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job
    if not current_process_is_in_job():
        raise R5AudiojackError("R5 experiment child is outside the Windows Job Object")
    output = _existing_output_path(run_id)
    manifest = r3._read_json(output / "run_manifest.json")
    _source_binding(manifest)
    if (manifest.get("run_id") != run_id or manifest.get("mode") != expected_mode or
            manifest.get("authorization") != AUTHORIZATION or
            manifest.get("authorization_token_sha256") != _sha256_bytes(token.encode("ascii")) or
            manifest.get("runner_source_sha256") != _sha256_file(Path(__file__).resolve())):
        raise R5AudiojackError("R5 child authorization/source manifest is incompatible")
    return output, manifest


def _append_history(budget: R5OutputBudget, row: Mapping[str, Any]) -> bytes:
    import experiments.calibration.r3_fullstream_2x2 as r3
    payload = r3.canonical_bytes(dict(row))
    budget.append("attempt_history.jsonl", payload)
    return payload


def _write_failure(output: Path, stage: str, exc: BaseException) -> None:
    if not output.is_dir():
        return
    try:
        budget = R5OutputBudget(output)
        payload = {
            "schema": "r5.audiojack.b3_fpsa.failure.v1", "status": "INCOMPLETE",
            "stage": stage, "run_id": output.name,
            "failure_type": type(exc).__name__, "failure_message": str(exc)[:4000],
            "automatic_restart": False,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        budget.write_json("failure.json", payload, failure_reserved=True)
    except Exception:
        pass


def _record_supervisor_result(output: Path, stage: str, result: Any) -> None:
    budget = R5OutputBudget(output)
    history_path = output / "attempt_history.jsonl"
    if not history_path.exists():
        budget.write_once("attempt_history.jsonl", b"")
    _append_history(budget, {
        "schema": "r5.audiojack.b3_fpsa.history.v1", "record_type": "supervisor_result",
        "category": CATEGORY, "stage": stage, "result": asdict(result),
        "automatic_restart": False,
    })


def _launch_child(output: Path, stage: str, token: str, *, timeout: int,
                  cuda_visibility: str | None, smoke_events: int = 0):
    import experiments.calibration.r3_fullstream_2x2 as r3
    from experiments.calibration.r1_runtime_supervisor import run_supervised_child
    command = [sys.executable, "-m", "experiments.calibration.r5_audiojack_b3_fpsa_runner",
               "--authorize-r5-audiojack-b3-fpsa", "--run-id", output.name]
    env = r3._child_environment(os.environ, stage, cuda_visibility)
    env["R5_AUDIOJACK_AUTH_TOKEN"] = token
    env["PYTHONHASHSEED"] = "0"
    if stage == "STAGE_A":
        command.append("--stage-a-child")
        if smoke_events:
            command.extend(["--smoke-events", str(smoke_events)])
    elif stage == "STAGE_B":
        command.append("--stage-b-child")
    else:
        raise R5AudiojackError(f"Unknown R5 child stage: {stage}")
    try:
        result = run_supervised_child(command, timeout_seconds=timeout, env=env,
                                      cwd=ROOT, output_dir=None)
    except BaseException as exc:
        _write_failure(output, stage, exc)
        try:
            R5OutputBudget(output).write_json("supervisor_status.json", {
                "schema": "r5.audiojack.b3_fpsa.supervisor.v1",
                "status": "SUPERVISOR_EXCEPTION", "stage": stage,
                "failure_type": type(exc).__name__, "automatic_restart": False,
            })
        except Exception:
            pass
        raise
    _record_supervisor_result(output, stage + ("_SMOKE" if smoke_events else ""), result)
    return result


def _stage_a_child(run_id: str, token: str, smoke_events: int = 0) -> dict[str, Any]:
    cuda_visibility_before = os.environ.get("CUDA_VISIBLE_DEVICES")
    import numpy as np
    import torch
    from types import SimpleNamespace
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r2_exp2a_c1_strength as exp2a
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration import r4_fpsa_audiojack_runner as r4
    from experiments.calibration.controllers import StateConsistentQuantile
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.calibration.r4_fpsa_codebook import FPSASignedCorrectionCodebook
    from experiments.risk_safe.native_bridge import dynamic_codebooks
    from v02_memory import MemoryArm

    output, manifest = _verify_child_manifest(
        run_id, token, "SMOKE_ONLY" if smoke_events else "FULL_STAGE_A")
    if smoke_events and not 0 < smoke_events <= 256:
        raise R5AudiojackError("R5 GPU smoke must be bounded to 1..256 events")
    contract = _load_audiojack_contract(verify_images=True)
    r4_rows, r4_receipt = _load_r4_audiojack_reference(contract)
    c64 = _load_c64_inputs(contract)
    budget = R5OutputBudget(output)
    r3._restore_cuda_visibility(cuda_visibility_before)
    scores_out = r3.BufferedJsonl(budget, "stage_a_scores.jsonl")
    score_digest, feedback_digest = hashlib.sha256(), hashlib.sha256()
    event_limit = min(EVENTS, smoke_events or EVENTS)
    allowed_queries = tuple(p for p in contract["positions"] if p < event_limit)
    query_set = set(allowed_queries)
    group_map = {int(p): str(contract["query_groups"][p]) for p in allowed_queries}
    from experiments.calibration.stage1a_p12_label_provider import QueriedPositionLabelReader
    label_reader = QueriedPositionLabelReader(
        category=CATEGORY, path=JSON_ROOT / "audiojack.json",
        expected_source_sha256=contract["class_json_sha256"],
        positions=allowed_queries, group_sha256_by_position=group_map) if allowed_queries else None
    baseline_cursor = r4._R3WriterCursor(R3_REFERENCE / "attempt_history.jsonl")
    backbone = None
    fpsa_cb = None
    source_groups: dict[str, dict[int, set[str]]] = {"GC": {}, "GC-FPSA": {}}
    counters = Counter()
    counters["queried_group_digests"] = set()
    counters["first_anomalous_query_position"] = None
    correction_stats: dict[str, Counter[str]] = {"GC": Counter(), "GC-FPSA": Counter()}
    greedy_memory_version = 0
    state_versions = {"G": 0, "GC": 0, "GC-FPSA": 0}
    current_c64_scores: dict[str, list[float]] = {}
    b3_controllers: dict[str, Any] = {}
    threshold_updates: dict[str, list[dict[str, Any]]] = {
        "G": [], "GC": [], "GC-FPSA": []}
    c64_rounds = Counter()
    c64_score_calls = Counter()
    c64_rescore_seconds = Counter()
    cache_state_parity_checked = False
    cache_state_parity_errors = Counter()
    parity = {"events": 0, "G_checks": 0, "GC_checks": 0,
              "FPSA_checks": 0, "G_max_abs_error": 0.0,
              "GC_max_abs_error": 0.0, "FPSA_max_abs_error": 0.0,
              "group_query_identity_checks": 0}
    category_start = time.monotonic()
    try:
        torch.set_num_threads(1)
        if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
            raise R5AudiojackError("R5 Stage A requires exactly one authorized CUDA device")
        impl = p0.require_sources()
        model_receipt = r3._get_verified_model_receipt()
        restore_hub, runtime_info = r3._runtime_and_native_setup(torch, impl, model_receipt)
        cfg, _profile = r3._native_config(
            p0, __import__("experiments.calibration.profiles", fromlist=["NativeScorerConfig"]), r1)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        setup_started = time.perf_counter()
        try:
            backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
        finally:
            restore_hub()
        loaded_model_sha = r1._model_state_sha256(backbone.model)
        if loaded_model_sha != model_receipt.get("loaded_model_state_sha256"):
            raise R5AudiojackError("R5 loaded DINO state differs from the verified local receipt")
        support_features = p0.encode_reference_features(
            impl, backbone, [str(path) for path in contract["support_paths"]], cfg)
        impl.set_seed(0)
        bank = p0.build_vmb(impl, support_features,
                            [str(path) for path in contract["support_paths"]], cfg)
        fixed_receipt = r3._check_bank_receipt(
            bank, p0, contract["fixed_support_sha256_expected"], r1)
        r3._validate_correction_config(bank)
        gc_cfg = bank.corr_cfg
        gc_cb = bank.corr_cb["p0"]["id"][0]
        if gc_cb.size() != 0 or gc_cb.K != 256:
            raise R5AudiojackError("R5 original GC Codebook did not initialize empty at K=256")
        fpsa_cfg = type(gc_cfg)(**asdict(gc_cfg))
        fpsa_cfg.enabled = False
        fpsa_cb = FPSASignedCorrectionCodebook(
            256, int(gc_cb.protos.shape[1]), gc_cb.device, fpsa_cfg)
        if gc_cb is fpsa_cb or gc_cb.protos is fpsa_cb.protos:
            raise R5AudiojackError("R5 GC and FPSA correction Codebooks share mutable state")
        greedy_cfg = SimpleNamespace(flat=256, stm=256, ltm=1, buffer=1,
                                     min_events=2, promote_every=16,
                                     cap_bytes=2 * 1024**3)
        writer = MemoryArm("greedy_coreset", bank, greedy_cfg, seed=0)
        if any(cb.size() != 0 or cb.K != 256 for cb in dynamic_codebooks(bank)):
            raise R5AudiojackError("R5 common Greedy memory did not initialize empty at K=256")
        bank.corr_cfg.enabled = False
        bank.use_defect_bank = False
        theta0, sigma0 = float(contract["theta"]), float(contract["sigma"])
        c64_cache, cache_manifest = _build_c64_cache(
            output, budget, impl, backbone, cfg, c64, p0)
        base_cache: list[tuple[float, np.ndarray, np.ndarray, Any]] = []
        initial_base_scores: list[float] = []
        for item in c64_cache:
            base = _score_c64_base(impl, bank, item, cfg, r1)
            base_cache.append(base)
            initial_base_scores.append(float(base[0]))
        initial_parity = {
            arm: {"count": C64_COUNT, "max_abs_error": 0.0}
            for arm in ("G", "GC", "GC-FPSA")}
        for index, actual in enumerate(initial_base_scores):
            expected = float(c64["scores"][index])
            error = abs(actual - expected)
            for arm in initial_parity:
                initial_parity[arm]["max_abs_error"] = max(
                    initial_parity[arm]["max_abs_error"], error)
                if not r3._score_tolerance(actual, expected):
                    raise R5AudiojackError(
                        f"Initial version-zero {arm} C64 score differs from frozen native receipt at {index}")
        initial_digests = _scorer_state_digests(
            bank, gc_cb, fpsa_cfg, fpsa_cb, source_groups["GC"], source_groups["GC-FPSA"],
            contract["fixed_support_sha256_expected"], r1, r4)
        state_digests = dict(initial_digests)
        current_c64_scores = {arm: list(initial_base_scores)
                              for arm in ("G", "GC", "GC-FPSA")}
        for arm in ("G", "GC", "GC-FPSA"):
            controller = StateConsistentQuantile(
                theta=theta0, delta=B3_DELTA, min_normals=B3_MIN_NORMALS,
                name=f"B3-{arm}")
            init = _initialize_b3(controller, theta0, current_c64_scores[arm],
                                  c64["group_sha256_digest"], initial_digests[arm])
            b3_controllers[arm] = controller
            threshold_updates[arm].append({"type": "version_zero_initialization", **init})
        if any(float(b3_controllers[arm].theta) != theta0 for arm in b3_controllers):
            raise R5AudiojackError("R5 B3 arms do not start at the common protected C64 q90")

        theta_gate = theta0
        from experiments.calibration.r1_matched_correction_pilot import (
            corrected_dense_map, max_score, common_c0_greedy_writer_action)
        for position, meta in enumerate(contract["stream"][:event_limit]):
            image_path = contract["event_paths"][position]
            if not image_path.is_file():
                raise R5AudiojackError(f"Audiojack event image disappeared at {position}")
            image_sha = r3.sha256_file(image_path)
            torch.cuda.synchronize()
            encode_start = time.perf_counter()
            image, features = p0.encode_query(impl, backbone, str(image_path), cfg)
            torch.cuda.synchronize()
            counters["actual_image_encodings"] += 1
            counters["stream_encoding_microseconds"] += int((time.perf_counter() - encode_start) * 1e6)
            if (len(features) != 1 or tuple(features[0].shape) != (784, 4096) or
                    features[0].dtype != torch.float32 or not torch.isfinite(features[0]).all().item()):
                raise R5AudiojackError(f"Audiojack native event feature contract failed at {position}")
            group_sha = str(meta.opaque_group)
            queried = position in query_set
            base_map, top_ids, base_score, native_summary = r1._native_dense_score(
                impl, bank, image, features, cfg)
            if not np.array_equal(top_ids, np.asarray(
                    native_summary["layer0"]["topk_patch_idx"], dtype=np.int64)):
                raise R5AudiojackError(f"R5 native Top-32 receipt differs at event {position}")
            if position == 0:
                ref_score, ref_ids, ref_map = r1._native_reference_and_debug(
                    impl, bank, image, features, cfg)
                if (not np.array_equal(ref_ids, top_ids) or not np.array_equal(ref_map, base_map) or
                        not r3._score_tolerance(ref_score, base_score)):
                    raise R5AudiojackError("R5 first-event direct native map/reference parity failed")
            selected = impl.l2_normalize(features[0].to(bank.device))[
                torch.as_tensor(top_ids, device=bank.device, dtype=torch.long)]
            delta_gc, diag_gc = r1._c1_score_delta(
                impl, bank, selected, source_groups["GC"], group_sha,
                event_position=position)
            delta_fpsa, diag_fpsa = r4._score_with_codebook(
                impl, bank, fpsa_cfg, fpsa_cb, source_groups["GC-FPSA"], selected,
                group_sha, position, r1)
            scores = {
                "G": float(base_score),
                "GC": float(max_score(corrected_dense_map(base_map, top_ids, delta_gc))),
                "GC-FPSA": float(max_score(corrected_dense_map(base_map, top_ids, delta_fpsa))),
            }
            if not all(math.isfinite(value) for value in scores.values()):
                raise R5AudiojackError(f"Nonfinite native scorer output at audiojack event {position}")
            ref = r4_rows[position]
            event_id = r3._event_id(
                CATEGORY, position, meta.relative_id, image_sha, group_sha)
            if (ref.get("event_id_sha256") != event_id or
                    ref.get("source_identity_sha256") != _sha256_bytes(meta.relative_id.encode()) or
                    ref.get("image_content_sha256") != image_sha or
                    ref.get("source_group_sha256") != group_sha or
                    bool(ref.get("query_tape_decision")) != queried):
                raise R5AudiojackError(f"R5 event identity differs from sealed R4 at {position}")
            parity["events"] += 1
            parity["group_query_identity_checks"] += 1
            for arm in ("G", "GC", "GC-FPSA"):
                error = abs(scores[arm] - float(ref["scores"][arm]))
                key = {"G": "G", "GC": "GC", "GC-FPSA": "FPSA"}[arm]
                parity[f"{key}_checks"] += 1
                parity[f"{key}_max_abs_error"] = max(parity[f"{key}_max_abs_error"], error)
                if not r3._score_tolerance(scores[arm], float(ref["scores"][arm])):
                    raise R5AudiojackError(
                        f"R5 frozen scorer parity failed at {position}/{arm}, abs={error:.3g}")

            state_versions_before = dict(state_versions)
            b3_predictions = {
                arm: _predict_b3(
                    b3_controllers[arm], position=position,
                    event_id_sha256=event_id, camera_id=int(meta.camera_id),
                    score=scores[arm], version=state_versions[arm],
                    state_digest=state_digests[arm])
                for arm in ("G", "GC", "GC-FPSA")}
            decisions = {}
            arm_names = {"G": "G", "GC": "GC", "GC-FPSA": "FPSA"}
            for scorer in ("G", "GC", "GC-FPSA"):
                frozen_prediction = _frozen_writer_gate(scores[scorer], theta_gate)
                decisions[f"{arm_names[scorer]}-Frozen"] = {
                    "score": scores[scorer], "threshold": theta_gate,
                    "prediction": frozen_prediction,
                    "scorer_state_version": state_versions[scorer],
                    "scorer_state_sha256": state_digests[scorer]}
                decisions[f"{arm_names[scorer]}-B3"] = {
                    "score": scores[scorer], **b3_predictions[scorer]}
            score_row = {
                "schema": SCORE_SCHEMA, "category": CATEGORY, "position": position,
                "event_id_sha256": event_id,
                "source_identity_sha256": _sha256_bytes(meta.relative_id.encode("utf-8")),
                "source_group_sha256": group_sha,
                "image_content_sha256": image_sha,
                "camera_id": int(meta.camera_id),
                "timestamp_semantics": "frozen JSON order is causal order; chronology not asserted",
                "query_tape_decision": queried, "fixed_threshold_c64_q90": theta_gate,
                "scores": scores, "decisions": decisions,
                "scorer_state_versions_before": state_versions_before,
                "greedy_memory_version_before": greedy_memory_version,
                "GC_correction_occupancy_before": int(gc_cb.size()),
                "FPSA_correction_occupancy_before": int(fpsa_cb.size()),
                "GC_same_group_excluded_slots": int(diag_gc["same_source_excluded_anchor_slots"]),
                "FPSA_same_group_excluded_slots": int(diag_fpsa["same_source_excluded_anchor_slots"]),
            }
            disclosure_gate = _ScoreDisclosureGate(position)
            encoded = scores_out.add(score_row)
            score_digest.update(encoded)
            counters["score_rows"] += 1
            if queried or (position + 1) % 256 == 0:
                scores_out.flush()
            if queried:
                disclosure_gate.mark_score_flushed(position)

            if queried:
                disclosure_gate.authorize_disclosure(position)
                label = r3._disclose_query_label_after_seal(
                    scores_out, query_decision=True, category=CATEGORY, position=position,
                    sample_key=meta.sample_key, label_reader=label_reader)
                if label not in (0, 1) or group_sha in counters["queried_group_digests"]:
                    raise R5AudiojackError("R5 queried feedback is invalid or repeats a Source Group")
                counters["queried_group_digests"].add(group_sha)
                counters["query_events"] += 1
                counters["query_normal" if label == 0 else "query_anomalous"] += 1
                if label == 1 and counters["first_anomalous_query_position"] is None:
                    counters["first_anomalous_query_position"] = position
                feedback_registration = fpsa_cb.observe_verified_feedback(
                    int(label), group_sha, position, score_sealed=True)
                correction_updates: dict[str, Any] = {}
                for arm, cfg_arm, cb_arm, branch in (
                        ("GC", gc_cfg, gc_cb, "GC"),
                        ("GC-FPSA", fpsa_cfg, fpsa_cb, "GC-FPSA")):
                    base_prediction = int(float(base_score) >= theta_gate)
                    outcome = {(0, 0): "TN", (0, 1): "FP",
                               (1, 0): "FN", (1, 1): "TP"}[(int(label), base_prediction)]
                    sign = 1 if int(label) == 1 else -1
                    _result, layer, trace_diag = exp2a._capture_update(
                        impl, bank, r1, branch, cfg_arm, cb_arm, features,
                        np.asarray(top_ids, dtype=np.int64), sign, float(base_score),
                        theta_gate, outcome, position, source_groups[branch], group_sha)
                    integer_stats = {key: int(value) for key, value in layer.items()
                                     if isinstance(value, (int, np.integer))}
                    correction_stats[branch].update(integer_stats)
                    correction_updates[arm] = {
                        "frozen_base_prediction": base_prediction,
                        "frozen_threshold": theta_gate, "outcome": outcome,
                        "sign": sign, "native_stats": integer_stats,
                        "occupancy_after": int(cb_arm.size()),
                        "native_keep_count": int(sum(bool(value) for value in
                                                       trace_diag["native_gate_keep_mask"]))}
                    if branch == "GC-FPSA":
                        correction_updates[arm]["fpsa_diagnostics"] = cb_arm.diagnostics()

                writer_start = time.perf_counter()
                audit_before = writer.audit()
                mutation = {"changed": False}
                def writer_update(*, position: int, feature_layers: Sequence[Any],
                                  candidate_ids: Any, score: float, theta: float,
                                  sigma: float) -> Mapping[str, Any]:
                    before = r1._memory_content_sha256(bank)
                    writer.update(impl, list(feature_layers), {0: candidate_ids.tolist()},
                                  position, score=score, theta=theta, sigma=sigma)
                    after = r1._memory_content_sha256(bank)
                    mutation["changed"] = before != after
                    audit_after = writer.audit()
                    return {
                        "accepted_patches_delta": int(audit_after["accepted_patches"] -
                                                      audit_before["accepted_patches"]),
                        "skipped_patches_delta": int(audit_after["skipped_patches"] -
                                                     audit_before["skipped_patches"]),
                        "_memory_content_changed": mutation["changed"],
                    }
                norm_candidates = impl.l2_normalize(features[0].to(bank.device))[
                    torch.as_tensor(top_ids, device=bank.device, dtype=torch.long)]
                writer_action = common_c0_greedy_writer_action(
                    position=position, c0_score=float(base_score), theta=theta_gate,
                    sigma=sigma0, c0_prediction=_frozen_writer_gate(base_score, theta_gate),
                    candidate_ids=np.asarray(top_ids, dtype=np.int64),
                    candidate_features=norm_candidates.detach().float().cpu().numpy(),
                    feature_layers=features, queried=True, disclosed_label=int(label),
                    eligibility=impl.select_informative_normal_write, update=writer_update)
                counters["writer_seconds_microseconds"] += int(
                    (time.perf_counter() - writer_start) * 1e6)
                changed_greedy = bool(mutation["changed"])
                writer_action.pop("_memory_content_changed", None)
                if changed_greedy:
                    greedy_memory_version += 1
                if writer_action.get("action") == "native_greedy_normal_update":
                    counters["greedy_write_events"] += 1
                    counters["greedy_accepted_patch_count"] += int(
                        writer_action.get("accepted_patches_delta", 0))
                    counters["greedy_skipped_patch_count"] += int(
                        writer_action.get("skipped_patches_delta", 0))
                expected_writer = baseline_cursor.query(position, event_id, group_sha)
                r4._writer_parity(writer_action, expected_writer, position=position)
                after_digests = _scorer_state_digests(
                    bank, gc_cb, fpsa_cfg, fpsa_cb, source_groups["GC"],
                    source_groups["GC-FPSA"], contract["fixed_support_sha256_expected"], r1, r4)
                changed = _changed_scorers(state_digests, after_digests)
                if changed:
                    if "G" in changed:
                        started = time.perf_counter()
                        torch.cuda.synchronize()
                        next_bases = []
                        next_g_scores = []
                        for item in c64_cache:
                            base = _score_c64_base(impl, bank, item, cfg, r1)
                            next_bases.append(base)
                            next_g_scores.append(float(base[0]))
                        torch.cuda.synchronize()
                        c64_rescore_seconds["G"] += time.perf_counter() - started
                        c64_score_calls["G"] += C64_COUNT
                        c64_rounds["G"] += 1
                        base_cache = next_bases
                        current_c64_scores["G"] = next_g_scores
                    for arm in ("GC", "GC-FPSA"):
                        if arm not in changed:
                            continue
                        started = time.perf_counter()
                        torch.cuda.synchronize()
                        cb_arm = gc_cb if arm == "GC" else fpsa_cb
                        cfg_arm = gc_cfg if arm == "GC" else fpsa_cfg
                        groups_arm = source_groups[arm]
                        rescored = []
                        scorer_position = EVENTS + state_versions[arm] + 1
                        for item, base in zip(c64_cache, base_cache):
                            _assert_c64_source_groups_excluded(
                                str(item["source_group_sha256"]), source_groups, arm)
                            rescored.append(_score_c64_corrected(
                                impl, bank, item, base, cfg_arm, cb_arm, groups_arm,
                                scorer_position, r1, r4))
                        torch.cuda.synchronize()
                        c64_rescore_seconds[arm] += time.perf_counter() - started
                        c64_score_calls[arm] += C64_COUNT
                        c64_rounds[arm] += 1
                        current_c64_scores[arm] = rescored

                transitions = {}
                for arm in changed:
                    old_digest = state_digests[arm]
                    old_version = state_versions[arm]
                    transition = _refresh_b3(
                        b3_controllers[arm], previous_digest=old_digest,
                        new_digest=after_digests[arm], previous_version=old_version,
                        new_scores=current_c64_scores[arm],
                        c64_group_sha256=c64["group_sha256_digest"],
                        new_state_digest=after_digests[arm])
                    if transition is None:
                        raise R5AudiojackError("Changed complete scorer state did not refresh its B3 evidence")
                    state_versions[arm] += 1
                    transition = _defer_b3_transition(
                        position, arm, after_digests[arm], transition)
                    threshold_updates[arm].append(transition)
                    transitions[arm] = transition
                state_digests = dict(after_digests)

                if changed and not cache_state_parity_checked:
                    item = c64_cache[0]
                    disk = _cache_decode((output / item["cache_file"]).read_bytes())
                    disk.update({"source_group_sha256": item["source_group_sha256"]})
                    torch.cuda.synchronize()
                    if "G" in changed:
                        direct = _score_c64_base(impl, bank, item, cfg, r1)[0]
                        reread = _score_c64_base(impl, bank, disk, cfg, r1)[0]
                        c64_score_calls["G"] += 2
                        cache_state_parity_errors["G"] = max(
                            cache_state_parity_errors["G"], abs(float(direct) - float(reread)))
                        if not r3._score_tolerance(direct, reread):
                            raise R5AudiojackError("Changed-state cached vs native C64 G score parity failed")
                    for arm in ("GC", "GC-FPSA"):
                        if arm not in changed:
                            continue
                        cb_arm = gc_cb if arm == "GC" else fpsa_cb
                        cfg_arm = gc_cfg if arm == "GC" else fpsa_cfg
                        groups_arm = source_groups[arm]
                        pseudo_pos = EVENTS + state_versions[arm]
                        _assert_c64_source_groups_excluded(
                            str(item["source_group_sha256"]), source_groups, arm)
                        memory_score = _score_c64_corrected(
                            impl, bank, item, base_cache[0], cfg_arm, cb_arm,
                            groups_arm, pseudo_pos, r1, r4)
                        disk_base = base_cache[0]
                        disk_score = _score_c64_corrected(
                            impl, bank, disk, disk_base, cfg_arm, cb_arm,
                            groups_arm, pseudo_pos, r1, r4)
                        c64_score_calls[arm] += 2
                        cache_state_parity_errors[arm] = max(
                            cache_state_parity_errors[arm], abs(float(memory_score) - float(disk_score)))
                        if not r3._score_tolerance(memory_score, disk_score):
                            raise R5AudiojackError(
                                f"Changed-state cached vs native C64 {arm} score parity failed")
                    torch.cuda.synchronize()
                    cache_state_parity_checked = True

                feedback_row = {
                    "schema": HISTORY_SCHEMA, "record_type": "queried_feedback_and_updates",
                    "category": CATEGORY, "position": position,
                    "event_id_sha256": event_id, "source_group_sha256": group_sha,
                    "disclosure_after_six_decisions_sealed": True,
                    "label": int(label), "fpsa_feedback_registration": feedback_registration,
                    "correction_updates": correction_updates,
                    "greedy_writer_action": writer_action,
                    "greedy_memory_version_after": greedy_memory_version,
                    "greedy_memory_content_changed": changed_greedy,
                    "scorer_states_changed": list(changed),
                    "scorer_state_versions_after": dict(state_versions),
                    "b3_threshold_transitions_for_next_event": transitions,
                }
                feedback_digest.update(_append_history(budget, feedback_row))

            if (position + 1) % 256 == 0:
                scores_out.flush()
                _append_history(budget, {
                    "schema": HISTORY_SCHEMA, "record_type": "diagnostic_progress_checkpoint",
                    "category": CATEGORY, "sealed_through_position": position,
                    "score_rows_written": int(counters["score_rows"]),
                    "query_feedback_count": int(counters["query_events"]),
                    "scorer_state_versions": dict(state_versions),
                    "scorer_state_sha256": dict(state_digests),
                    "thresholds_next_event": {arm: float(b3_controllers[arm].theta)
                                               for arm in b3_controllers},
                    "automatic_restart": False,
                })
                print(f"R5_AUDIOJACK_PROGRESS run={run_id} events={position+1}/{event_limit} "
                      f"queries={counters['query_events']} C64_calls="
                      f"{sum(c64_score_calls.values())}", flush=True)
            del image, features

        scores_out.flush()
        if label_reader is not None:
            if tuple(label_reader.decoded_positions) != allowed_queries:
                raise R5AudiojackError("R5 restricted query reader did not consume the exact tape prefix")
            label_reader.close()
        baseline_receipt = baseline_cursor.finish() if event_limit == EVENTS else None
        if baseline_receipt is None:
            baseline_cursor.stream.close()
        if event_limit == EVENTS and (
                counters["score_rows"] != EVENTS or counters["query_events"] != QUERIES or
                counters["query_normal"] != 74 or counters["query_anomalous"] != 40 or
                counters["first_anomalous_query_position"] != FIRST_ANOMALOUS_QUERY):
            raise R5AudiojackError("Full R5 audiojack stream/query outcome receipt differs from frozen contract")
        if counters["query_events"] and not cache_state_parity_checked:
            raise R5AudiojackError("GPU smoke/full stream failed to exercise changed-state cache parity")
        fixed_final = _fixed_bank_digest(bank)
        if fixed_final != contract["fixed_support_sha256_expected"]:
            raise R5AudiojackError("R5 shared Greedy writer altered the frozen fixed Support bank")
        final_occ = {"GC": r4._sign_occupancy(gc_cb), "GC-FPSA": r4._sign_occupancy(fpsa_cb)}
        final_fpsa = fpsa_cb.diagnostics()
        if event_limit == EVENTS:
            expected_complete = r4_receipt["completion"]
            if (final_occ != expected_complete.get("correction_final_sign_occupancy") or
                    final_fpsa.get("quota_borrow_opportunities") != 365 or
                    final_fpsa.get("quota_cross_sign_replacements") != 93 or
                    final_fpsa.get("quota_borrow_rejected") != 272 or
                    final_fpsa.get("actual_positive_occupancy") != 91 or
                    final_fpsa.get("actual_negative_occupancy") != 165):
                raise R5AudiojackError("R5 full correction/FPSA final state differs from sealed R4")
        writer_audit = writer.audit()
        writer_reference_parity = None
        if baseline_receipt is not None:
            writer_reference_parity = {}
            for key in ("greedy_write_events", "greedy_accepted_patch_count",
                        "greedy_skipped_patch_count"):
                actual = int(counters[key]) if key in counters else int(counters[
                    {"greedy_write_events": "greedy_write_events",
                     "greedy_accepted_patch_count": "greedy_accepted_patch_count",
                     "greedy_skipped_patch_count": "greedy_skipped_patch_count"}[key]])
                expected = int(baseline_receipt.get(key, -1))
                if actual != expected:
                    raise R5AudiojackError(f"R5 G-only writer differs from R3 reference at {key}")
                writer_reference_parity[key] = {"actual": actual, "reference": expected,
                                                "match": True}
        result = {
            "schema": HISTORY_SCHEMA, "record_type": "category_complete",
            "category": CATEGORY, "events_processed": int(counters["score_rows"]),
            "query_events_processed": int(counters["query_events"]),
            "query_normal": int(counters["query_normal"]),
            "query_anomalous": int(counters["query_anomalous"]),
            "first_anomalous_query_position": counters["first_anomalous_query_position"],
            "score_rows_sha256": score_digest.hexdigest(),
            "feedback_receipts_sha256": feedback_digest.hexdigest(),
            "query_tape_sha256": contract["query_tape_sha256"],
            "query_positions_sha256": _sha256_json(list(allowed_queries)),
            "fixed_threshold_c64_q90": theta_gate,
            "fixed_support_sha256": fixed_final,
            "fixed_support_digest_evidence": fixed_receipt,
            "r4_reference_stage_a_sha256": r4_receipt["stage_a_sha256"],
            "r4_reference_score_rows_sha256": r4_receipt["stage_a"].get("score_rows_sha256"),
            "r4_score_parity": parity,
            "r4_greedy_writer_reference_parity": writer_reference_parity,
            "greedy_memory_version_final": greedy_memory_version,
            "greedy_writer_audit_final": writer_audit,
            "greedy_write_events": int(counters["greedy_write_events"]),
            "greedy_accepted_patch_count": int(counters["greedy_accepted_patch_count"]),
            "greedy_skipped_patch_count": int(counters["greedy_skipped_patch_count"]),
            "correction_stats": {key: dict(value) for key, value in correction_stats.items()},
            "correction_final_sign_occupancy": final_occ,
            "fpsa_final_diagnostics": final_fpsa,
            "correction_capacity": 256,
            "scorer_state_versions_final": dict(state_versions),
            "scorer_state_sha256_final": dict(state_digests),
            "b3_threshold_updates": threshold_updates,
            "b3_update_counts": {arm: sum(1 for row in rows
                                             if row.get("type") != "version_zero_initialization")
                                 for arm, rows in threshold_updates.items()},
            "c64_rescore_rounds": {arm: int(c64_rounds[arm])
                                   for arm in ("G", "GC", "GC-FPSA")},
            "c64_rescore_score_calls": {arm: int(c64_score_calls[arm])
                                        for arm in ("G", "GC", "GC-FPSA")},
            "c64_rescore_seconds": {arm: float(c64_rescore_seconds[arm])
                                    for arm in ("G", "GC", "GC-FPSA")},
            "c64_cache": {"count": C64_COUNT, "group_sha256": c64["group_sha256_digest"],
                          "image_sha256": c64["image_sha256"],
                          "cache_manifest_sha256": _sha256_file(output / "c64_cache_manifest.json"),
                          "feature_cache_payload_bytes": sum(
                              int(row["file_bytes"]) for row in cache_manifest["cache_receipts"])},
            "c64_initial_native_score_parity": initial_parity,
            "c64_cache_changed_state_native_score_parity": {
                "passed": bool(cache_state_parity_checked),
                "representative_reference_count": 1 if cache_state_parity_checked else 0,
                "same_scoring_path_from_memory_and_disk_cache": bool(cache_state_parity_checked),
                "max_abs_error_by_scorer": {
                    arm: float(cache_state_parity_errors[arm])
                    for arm in ("G", "GC", "GC-FPSA")},
                "tolerance": {"atol": ATOL, "rtol": RTOL}},
            "runtime_profile": runtime_info["runtime_profile"],
            "hub_source_tree_sha256": runtime_info["hub_source_tree_sha256"],
            "checkpoint_sha256": runtime_info["checkpoint_sha256"],
            "loaded_model_state_sha256": loaded_model_sha,
            "actual_stream_image_encodings": int(counters["actual_image_encodings"]),
            "support_and_model_setup_seconds": time.perf_counter() - setup_started,
            "stream_encoding_microseconds": int(counters["stream_encoding_microseconds"]),
            "writer_seconds": counters["writer_seconds_microseconds"] / 1e6,
            "wall_seconds": time.monotonic() - category_start,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_cpu_rss_bytes": r1._process_rss_bytes(),
            "smoke_only": bool(smoke_events), "offline_evaluation_performed": False,
            "B3_decoupled_from_writer_and_correction": True,
        }
        _append_history(budget, result)
        backbone.cleanup()
        return result
    except BaseException as exc:
        try:
            scores_out.flush()
        except Exception:
            pass
        try:
            if label_reader is not None:
                label_reader.close()
        except Exception:
            pass
        try:
            baseline_cursor.stream.close()
        except Exception:
            pass
        try:
            if backbone is not None:
                backbone.cleanup()
        except Exception:
            pass
        _write_failure(output, "STAGE_A_CHILD", exc)
        raise


def _seal_stage_a(output: Path, contract: Mapping[str, Any],
                  r4_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import experiments.calibration.r3_fullstream_2x2 as r3
    scores_path = output / "stage_a_scores.jsonl"
    history_path = output / "attempt_history.jsonl"
    rows: list[dict[str, Any]] = []
    score_digest = hashlib.sha256()
    with scores_path.open("r", encoding="utf-8") as stream:
        for position, line in enumerate(stream):
            row = json.loads(line)
            if (row.get("schema") != SCORE_SCHEMA or row.get("category") != CATEGORY or
                    int(row.get("position", -1)) != position or
                    any(key in row for key in ("label", "anomaly_class", "mask_path")) or
                    set(row.get("decisions", {})) != {
                        "G-Frozen", "GC-Frozen", "FPSA-Frozen", "G-B3", "GC-B3", "FPSA-B3"}):
                raise R5AudiojackError(f"R5 Stage A score row {position} failed seal/schema checks")
            for arm, scorer in (("G-Frozen", "G"), ("GC-Frozen", "GC"),
                                ("FPSA-Frozen", "GC-FPSA"), ("G-B3", "G"),
                                ("GC-B3", "GC"), ("FPSA-B3", "GC-FPSA")):
                decision = row["decisions"][arm]
                if (decision.get("score") != row["scores"].get(scorer) or
                        decision.get("prediction") != int(
                            float(decision["score"]) >= float(decision["threshold"]))):
                    raise R5AudiojackError(f"R5 six-arm score/decision binding failed at {position}/{arm}")
            for scorer in ("G", "GC", "GC-FPSA"):
                actual, expected = float(row["scores"][scorer]), float(r4_rows[position]["scores"][scorer])
                if not r3._score_tolerance(actual, expected):
                    raise R5AudiojackError(f"R5 sealed score differs from R4 at {position}/{scorer}")
            rows.append(row)
            score_digest.update(r3.canonical_bytes(row))
    feedback: list[dict[str, Any]] = []
    completion = None
    with history_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("record_type") == "queried_feedback_and_updates":
                feedback.append(row)
            elif row.get("record_type") == "category_complete":
                completion = row
    if (len(rows) != EVENTS or len(feedback) != QUERIES or not completion or
            completion.get("events_processed") != EVENTS or
            completion.get("query_events_processed") != QUERIES or
            [int(row["position"]) for row in rows if row["query_tape_decision"]] !=
            list(contract["positions"]) or
            score_digest.hexdigest() != completion.get("score_rows_sha256") or
            _sha256_file(scores_path) != completion.get("score_rows_sha256")):
        raise R5AudiojackError("R5 Stage A does not contain the exact sealed 3,815-event/114-query stream")
    feedback_digest = hashlib.sha256()
    for item in feedback:
        position = int(item["position"])
        if (not 0 <= position < EVENTS or
                item.get("event_id_sha256") != rows[position].get("event_id_sha256") or
                item.get("source_group_sha256") != rows[position].get("source_group_sha256") or
                item.get("disclosure_after_six_decisions_sealed") is not True or
                item.get("scorer_states_changed") != [
                    name for name in ("G", "GC", "GC-FPSA")
                    if name in item.get("b3_threshold_transitions_for_next_event", {})] or
                int(item.get("label", -1)) not in (0, 1)):
            raise R5AudiojackError(f"R5 private queried feedback is not bound to sealed event {position}")
        if int(item["label"]) == 1 and item.get("greedy_writer_action", {}).get(
                "action") != "verified_defect_no_normal_write":
            raise R5AudiojackError(f"Queried audiojack defect entered shared normal memory at {position}")
        for arm, update in item.get("correction_updates", {}).items():
            if update.get("frozen_threshold") != float(rows[position]["fixed_threshold_c64_q90"]):
                raise R5AudiojackError(f"R5 correction writer received a B3 threshold at {position}/{arm}")
        feedback_digest.update(r3.canonical_bytes(item))
    if feedback_digest.hexdigest() != completion.get("feedback_receipts_sha256"):
        raise R5AudiojackError("R5 feedback receipt digest differs from Stage A child completion")
    for item in feedback:
        for arm, transition in item.get("b3_threshold_transitions_for_next_event", {}).items():
            target_position = int(transition.get("apply_from_position", -1))
            if target_position != int(item["position"]) + 1:
                raise R5AudiojackError("R5 B3 threshold transition is not deferred to event t+1")
            if target_position < EVENTS and float(
                    rows[target_position]["decisions"][f"{ {'G':'G','GC':'GC','GC-FPSA':'FPSA'}[arm] }-B3"]["threshold"]
                    ) != float(transition["new_theta"]):
                raise R5AudiojackError(f"R5 B3 threshold did not apply at event {target_position}/{arm}")
    if (completion.get("r4_score_parity", {}).get("events") != EVENTS or
            completion.get("r4_score_parity", {}).get("G_max_abs_error") > ATOL or
            completion.get("r4_score_parity", {}).get("GC_max_abs_error") > ATOL or
            completion.get("r4_score_parity", {}).get("FPSA_max_abs_error") > ATOL or
            completion.get("c64_initial_native_score_parity", {}).get("G", {}).get("count") != C64_COUNT or
            completion.get("c64_cache_changed_state_native_score_parity", {}).get("passed") is not True):
        raise R5AudiojackError("R5 full native or C64 cache score parity evidence is incomplete")
    writer_parity = completion.get("r4_greedy_writer_reference_parity")
    if not writer_parity or any(
            not writer_parity.get(key, {}).get("match") for key in (
                "greedy_write_events", "greedy_accepted_patch_count",
                "greedy_skipped_patch_count")):
        raise R5AudiojackError("R5 common Greedy writer differs from the frozen R3 G-only reference")
    seal = {
        "schema": STAGE_A_SCHEMA, "status": "SEALED_COMPLETE",
        "run_id": output.name, "source_parent_commit": EXPECTED_PARENT,
        "category": CATEGORY, "event_count": EVENTS, "query_count": QUERIES,
        "score_rows_sha256": _sha256_file(scores_path),
        "canonical_score_rows_sha256": score_digest.hexdigest(),
        "feedback_receipts_sha256": feedback_digest.hexdigest(),
        "query_tape_sha256": contract["query_tape_sha256"],
        "query_positions_sha256": contract["query_positions_sha256"],
        "r4_score_parity": completion["r4_score_parity"],
        "r4_greedy_writer_reference_parity": completion["r4_greedy_writer_reference_parity"],
        "c64_initial_native_score_parity": completion["c64_initial_native_score_parity"],
        "c64_cache_changed_state_native_score_parity": completion[
            "c64_cache_changed_state_native_score_parity"],
        "c64_cache_manifest_sha256": completion["c64_cache"]["cache_manifest_sha256"],
        "scorer_state_versions_final": completion["scorer_state_versions_final"],
        "b3_update_counts": completion["b3_update_counts"],
        "feedback_disclosed_after_current_six_decisions_sealed": True,
        "threshold_transitions_apply_from_next_event": True,
        "B3_is_decoupled_from_frozen_writer_and_correction_threshold": True,
        "unqueried_test_labels_accessed": False, "offline_outcome_labels_accessed": False,
        "test_masks_opened": False, "automatic_restart": False,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    R5OutputBudget(output).write_json("stage_a_seal.json", seal)
    return seal


def _binary_metrics(labels: Any, predictions: Any) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(predictions, dtype=np.int8)
    if y.shape != p.shape or not np.isin(y, (0, 1)).all() or not np.isin(p, (0, 1)).all():
        raise R5AudiojackError("Offline six-arm evaluation arrays are malformed")
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    pos, neg = tp + fn, tn + fp
    fpr = fp / neg if neg else None
    fnr = fn / pos if pos else None
    ba = (1.0 - (fpr + fnr) / 2.0) if fpr is not None and fnr is not None else None
    n = int(y.size)
    return {"events": n, "normal_events": neg, "anomalous_events": pos,
            "balanced_accuracy": ba, "fpr": fpr, "fnr": fnr,
            "accuracy": (tp + tn) / n if n else None,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}


def _bootstrap_contrasts(predictions: Mapping[str, Any], labels: Any,
                         groups: Any, comparisons: Mapping[str, tuple[str, str]]) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    group = np.asarray(groups, dtype=str)
    unique = np.unique(group)
    if len(unique) < 2:
        return {"status": "UNDEFINED_FEWER_THAN_TWO_SOURCE_GROUPS", "intervals": {}}
    inverse = np.searchsorted(unique, group)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: dict[str, list[float]] = {name: [] for name in comparisons}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(unique), size=len(unique))
        counts = np.bincount(sampled, minlength=len(unique))
        weights = counts[inverse].astype(np.float64)
        arm_ba: dict[str, float | None] = {}
        for arm, raw in predictions.items():
            p = np.asarray(raw, dtype=np.int8)
            tp = float(weights[(y == 1) & (p == 1)].sum())
            fn = float(weights[(y == 1) & (p == 0)].sum())
            fp = float(weights[(y == 0) & (p == 1)].sum())
            tn = float(weights[(y == 0) & (p == 0)].sum())
            pos, neg = tp + fn, tn + fp
            arm_ba[arm] = (1.0 - 0.5 * (fn / pos + fp / neg)) if pos and neg else None
        for name, (left, right) in comparisons.items():
            if arm_ba[left] is not None and arm_ba[right] is not None:
                samples[name].append(float(arm_ba[left] - arm_ba[right]))
    return {
        "status": "DESCRIPTIVE_SOURCE_GROUP_PAIRED_BOOTSTRAP",
        "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
        "unit": "audiojack Source Group resampled with replacement; event multiplicity follows group",
        "independent_confirmation": False,
        "intervals_2_5_97_5_percent": {
            name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            if values else None for name, values in samples.items()},
    }


def _stage_b_child(run_id: str, token: str) -> dict[str, Any]:
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "" or
            os.environ.get("R3_STAGE_B_CPU_ONLY") != "1"):
        raise R5AudiojackError("R5 Stage B must start with the supervisor's explicit CPU-only environment")
    import numpy as np
    import torch
    import experiments.calibration.r3_fullstream_2x2 as r3
    output, manifest = _verify_child_manifest(run_id, token, "FULL_STAGE_A")
    if torch.cuda.device_count() != 0 or torch.cuda.is_initialized():
        raise R5AudiojackError("R5 Stage B must run CPU-only with CUDA hidden and uninitialized")
    seal_path, scores_path = output / "stage_a_seal.json", output / "stage_a_scores.jsonl"
    seal = _read_json(seal_path)
    if (seal.get("status") != "SEALED_COMPLETE" or seal.get("category") != CATEGORY or
            seal.get("event_count") != EVENTS or seal.get("score_rows_sha256") != _sha256_file(scores_path) or
            seal.get("query_tape_sha256") != manifest.get("query_tape_sha256")):
        raise R5AudiojackError("R5 Stage B requires a verified complete audiojack Stage A seal")
    rows = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines()]
    if (len(rows) != EVENTS or [int(row.get("position", -1)) for row in rows] != list(range(EVENTS)) or
            any(row.get("schema") != SCORE_SCHEMA or row.get("category") != CATEGORY or
                "label" in row or "anomaly_class" in row or "mask_path" in row
                for row in rows)):
        raise R5AudiojackError("R5 Stage B received unsealed, unordered or label-bearing score rows")
    label_path = JSON_ROOT / "audiojack.json"
    if _sha256_file(label_path) != manifest.get("class_json_sha256"):
        raise R5AudiojackError("Audiojack TEST metadata changed after Stage A sealing")
    labels = r3._read_stage_b_label_projection(label_path, EVENTS)
    if labels.shape != (EVENTS,) or not np.isin(labels, (0, 1)).all():
        raise R5AudiojackError("Restricted audiojack Stage B outcome projection failed")
    arms = ("G-Frozen", "GC-Frozen", "FPSA-Frozen", "G-B3", "GC-B3", "FPSA-B3")
    predictions = {arm: np.asarray([int(row["decisions"][arm]["prediction"])
                                    for row in rows], dtype=np.int8) for arm in arms}
    thresholds = {arm: np.asarray([float(row["decisions"][arm]["threshold"])
                                   for row in rows], dtype=np.float64) for arm in arms}
    scores = {scorer: np.asarray([float(row["scores"][scorer]) for row in rows],
                                 dtype=np.float64)
              for scorer in ("G", "GC", "GC-FPSA")}
    fixed_map = {"G-Frozen": "G", "GC-Frozen": "GC", "FPSA-Frozen": "GC-FPSA"}
    b3_map = {"G-B3": "G", "GC-B3": "GC", "FPSA-B3": "GC-FPSA"}
    theta0 = float(manifest["fixed_threshold_c64_q90"])
    for arm in arms:
        scorer = fixed_map.get(arm, b3_map.get(arm))
        if np.any(predictions[arm] != (scores[scorer] >= thresholds[arm]).astype(np.int8)):
            raise R5AudiojackError(f"R5 Stage B prediction disagrees with score >= threshold for {arm}")
        if arm in fixed_map and np.any(thresholds[arm] != theta0):
            raise R5AudiojackError(f"R5 frozen threshold changed within stream for {arm}")
    ranking = {scorer: r3._threshold_metrics(values, labels, theta0)
               for scorer, values in scores.items()}
    metrics = {}
    for arm in arms:
        scorer = fixed_map.get(arm, b3_map.get(arm))
        decision_metrics = _binary_metrics(labels, predictions[arm])
        decision_metrics.update({"auroc": ranking[scorer]["auroc"],
                                 "average_precision": ranking[scorer]["average_precision"],
                                 "score_ranking_source": scorer})
        metrics[arm] = decision_metrics
    for frozen_arm, b3_arm in (("G-Frozen", "G-B3"),
                               ("GC-Frozen", "GC-B3"),
                               ("FPSA-Frozen", "FPSA-B3")):
        if (metrics[frozen_arm]["auroc"] != metrics[b3_arm]["auroc"] or
                metrics[frozen_arm]["average_precision"] != metrics[b3_arm]["average_precision"]):
            raise R5AudiojackError("Threshold-only B3 changed a scorer's AUROC/AP ranking")
    comparisons = {
        "FPSA-B3_minus_FPSA-Frozen": ("FPSA-B3", "FPSA-Frozen"),
        "FPSA-B3_minus_GC-B3": ("FPSA-B3", "GC-B3"),
        "FPSA-B3_minus_G-B3": ("FPSA-B3", "G-B3"),
        "GC-B3_minus_G-B3": ("GC-B3", "G-B3"),
    }
    contrasts = {}
    for name, (left, right) in comparisons.items():
        contrasts[name] = {
            "left_arm": left, "right_arm": right,
            "balanced_accuracy_difference": (
                metrics[left]["balanced_accuracy"] - metrics[right]["balanced_accuracy"]),
            "fpr_difference": metrics[left]["fpr"] - metrics[right]["fpr"],
            "fnr_difference": metrics[left]["fnr"] - metrics[right]["fnr"],
            "accuracy_difference": metrics[left]["accuracy"] - metrics[right]["accuracy"],
        }
    post_slice = slice(FIRST_ANOMALOUS_QUERY + 1, EVENTS)
    post_metrics = {arm: _binary_metrics(labels[post_slice], predictions[arm][post_slice])
                    for arm in arms}
    window_edges = list(range(0, EVENTS, 512)) + [EVENTS]
    windows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    for start, stop in zip(window_edges[:-1], window_edges[1:]):
        for arm in arms:
            values = _binary_metrics(labels[start:stop], predictions[arm][start:stop])
            windows[arm].append({"start_position": start, "end_position_exclusive": stop,
                                 "events": stop - start, "partial_tail": stop - start < 512,
                                 **values})
    worst_windows = {}
    for arm in arms:
        defined = [item for item in windows[arm] if item["balanced_accuracy"] is not None]
        complete = [item for item in defined if not item["partial_tail"]]
        key = lambda item: (item["balanced_accuracy"], -item["start_position"])
        worst_windows[arm] = {
            "worst_defined_including_partial_tail": min(defined, key=key) if defined else None,
            "worst_complete_512_event_window": min(complete, key=key) if complete else None,
        }
    threshold_trajectory = {}
    b3_arm_for_scorer = {"G": "G-B3", "GC": "GC-B3", "GC-FPSA": "FPSA-B3"}
    for scorer, arm in b3_arm_for_scorer.items():
        values = thresholds[arm]
        moved_positions = np.flatnonzero(np.diff(values) != 0.0) + 1
        threshold_trajectory[arm] = {
            "scorer": scorer, "initial_threshold": float(values[0]),
            "final_threshold": float(values[-1]), "minimum_threshold": float(values.min()),
            "maximum_threshold": float(values.max()),
            "movement_count": int(len(moved_positions)),
            "total_variation": float(np.abs(np.diff(values)).sum()),
            "state_version_final": int(rows[-1]["decisions"][arm]["scorer_state_version"]),
            "movement_event_positions": [int(value) for value in moved_positions],
        }
    group_ids = np.asarray([str(row["source_group_sha256"]) for row in rows], dtype=str)
    query_mask = np.asarray([bool(row["query_tape_decision"]) for row in rows], dtype=bool)
    history = [json.loads(line) for line in (output / "attempt_history.jsonl").read_text(
        encoding="utf-8").splitlines()]
    feedback = [row for row in history if row.get("record_type") == "queried_feedback_and_updates"]
    queried_anomalies = [int(row["position"]) for row in feedback if int(row["label"]) == 1]
    if (len(feedback) != QUERIES or len(queried_anomalies) != 40 or
            min(queried_anomalies) != FIRST_ANOMALOUS_QUERY or int(query_mask.sum()) != QUERIES):
        raise R5AudiojackError("R5 Stage B query feedback counts/first anomaly differ from frozen receipt")
    completion = next((row for row in reversed(history)
                       if row.get("record_type") == "category_complete"), None)
    if completion is None:
        raise R5AudiojackError("R5 Stage A category completion receipt is missing")
    risk_scenario = {"name": "illustrative_not_factory_approved",
                     "hypothetical_fpr_max": 0.01, "hypothetical_fnr_max": 0.05,
                     "interpretation": "descriptive example only; no researcher/factory risk budget supplied"}
    risk_scenario["arm_flags"] = {
        arm: {"meets_hypothetical_fpr": metrics[arm]["fpr"] <= risk_scenario["hypothetical_fpr_max"],
              "meets_hypothetical_fnr": metrics[arm]["fnr"] <= risk_scenario["hypothetical_fnr_max"]}
        for arm in arms}
    aggregate = {
        "schema": "r5.audiojack.b3_fpsa.aggregate.v1", "status": "COMPLETE",
        "run_id": run_id,
        "classification": "RETROSPECTIVE_EXPOSED_AUDIOJACK_DEVELOPMENT_CLASS_VALIDATION",
        "method": "B3 with complete scorer-state reference rescoring; threshold-only decoupled deployment profile",
        "dataset_protocol": {
            "category": CATEGORY, "event_count": EVENTS,
            "event_order": "frozen original Real-IAD TEST JSON order; physical chronology not asserted",
            "independent_validation": False, "test_masks_opened": False,
            "unqueried_TEST_labels_accessed_in_stage_a": False,
            "Stage_B_labels_read_after_complete_Stage_A_seal": True,
            "protected_reference_count": C64_COUNT,
            "support_count": 4, "C64_is_additional_to_supports": True,
            "query_count": int(query_mask.sum()), "query_rate": float(query_mask.mean()),
            "query_groups_unique": True,
            "first_anomalous_feedback_position": FIRST_ANOMALOUS_QUERY,
        },
        "arms": {arm: {"scorer": scorer, "threshold": threshold}
                 for arm, scorer, threshold in (
                     ("G-Frozen", "G", "frozen C64 higher-q90"),
                     ("GC-Frozen", "GC", "frozen C64 higher-q90"),
                     ("FPSA-Frozen", "GC-FPSA", "frozen C64 higher-q90"),
                     ("G-B3", "G", "G complete-state protected C64 q90"),
                     ("GC-B3", "GC", "GC complete-state protected C64 q90"),
                     ("FPSA-B3", "GC-FPSA", "FPSA complete-state protected C64 q90"))},
        "primary_endpoint": "full-stream prequential Balanced Accuracy",
        "full_stream_metrics": metrics,
        "required_decision_contrasts": contrasts,
        "post_1305_metrics": {"positions_inclusive": [1306, EVENTS - 1],
                               "event_count": EVENTS - FIRST_ANOMALOUS_QUERY - 1,
                               "metrics": post_metrics},
        "fixed_512_event_windows": windows,
        "worst_eligible_temporal_window": worst_windows,
        "threshold_trajectories": threshold_trajectory,
        "b3_controller_state_updates": completion["b3_update_counts"],
        "c64_rescoring": {
            "feature_cache_count": C64_COUNT,
            "feature_cache_bytes": completion["c64_cache"]["feature_cache_payload_bytes"],
            "cache_manifest_sha256": completion["c64_cache"]["cache_manifest_sha256"],
            "score_calls_by_scorer": completion["c64_rescore_score_calls"],
            "rescore_rounds_by_scorer": completion["c64_rescore_rounds"],
            "rescore_seconds_by_scorer": completion["c64_rescore_seconds"],
            "initial_native_score_parity": completion["c64_initial_native_score_parity"],
            "changed_state_cached_native_parity": completion[
                "c64_cache_changed_state_native_score_parity"],
        },
        "native_r4_parity": completion["r4_score_parity"],
        "greedy_writer_parity": completion["r4_greedy_writer_reference_parity"],
        "final_correction_occupancy": completion["correction_final_sign_occupancy"],
        "fpsa_final_diagnostics": completion["fpsa_final_diagnostics"],
        "source_group_count": int(len(np.unique(group_ids))),
        "source_group_paired_bootstrap": _bootstrap_contrasts(
            predictions, labels, group_ids, comparisons),
        "hypothetical_operational_tolerance_scenario": risk_scenario,
        "ranking_diagnostics_only": {
            "auroc_and_average_precision_by_arm": {
                arm: {"auroc": metrics[arm]["auroc"],
                      "average_precision": metrics[arm]["average_precision"]}
                for arm in arms},
            "threshold_paired_ranking_identical": True,
            "not_optimized": True,
        },
        "interpretation_limit": (
            "One previously exposed development category, artificial fixed 3% label-blind query tape, "
            "frozen TEST JSON order, four deployed supports plus 64 extra protected TRAIN normals; "
            "not independent production validation or factory readiness."),
        "r4_reference": {"run_id": "r4_fpsa_audiojack_20261102_a1",
                         "stage_a_seal_sha256": completion["r4_reference_stage_a_sha256"],
                         "score_rows_sha256": completion["r4_reference_score_rows_sha256"]},
        "score_rows_sha256": _sha256_file(scores_path),
        "stage_a_seal_sha256": _sha256_file(seal_path),
        "query_feedback_ledger_sha256": seal["feedback_receipts_sha256"],
        "source_parent_commit": EXPECTED_PARENT,
        "source_fingerprints": manifest["source_fingerprints"],
        "test_masks_opened": False,
        "execution_attempts": {
            "smoke_run_id": SMOKE_ID,
            "preserved_failed_smoke_run_id": FAILED_SMOKE_ID,
            "preserved_failure_reason": "attempt_history.jsonl was not initialized before first queried feedback append",
            "automatic_restart": False,
        },
    }
    R5OutputBudget(output).write_json("aggregate_results.json", aggregate)
    stage_b_seal = {
        "schema": STAGE_B_SCHEMA, "status": "SEALED_COMPLETE", "run_id": run_id,
        "stage_a_seal_sha256": _sha256_file(seal_path),
        "score_rows_sha256": _sha256_file(scores_path),
        "aggregate_results_sha256": _sha256_file(output / "aggregate_results.json"),
        "event_count": EVENTS, "audiojack_outcome_projection_after_stage_a": True,
        "test_masks_opened": False,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    budget = R5OutputBudget(output)
    budget.write_json("stage_b_seal.json", stage_b_seal)
    budget.write_json("complete_seal.json", {
        "schema": "r5.audiojack.b3_fpsa.complete.v1", "status": "COMPLETE",
        "run_id": run_id, "stage_a_seal_sha256": _sha256_file(seal_path),
        "stage_b_seal_sha256": _sha256_file(output / "stage_b_seal.json"),
        "aggregate_results_sha256": _sha256_file(output / "aggregate_results.json"),
        "event_count": EVENTS, "automatic_restart": False,
    })
    return aggregate


def _initialize_run(run_id: str, mode: str, contract: Mapping[str, Any],
                    token: str, *, free_bytes: int) -> Path:
    if mode not in {"SMOKE_ONLY", "FULL_STAGE_A"}:
        raise R5AudiojackError("R5 run initialization mode is invalid")
    output = _output_path(run_id)
    budget = R5OutputBudget(output, create=True)
    budget.write_once("attempt_history.jsonl", b"")
    manifest = {
        "schema": RUN_SCHEMA, "run_id": run_id, "mode": mode,
        "authorization": AUTHORIZATION,
        "authorization_token_sha256": _sha256_bytes(token.encode("ascii")),
        "branch": EXPECTED_BRANCH, "source_parent_commit": EXPECTED_PARENT,
        "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        "source_fingerprints": _source_hashes(),
        "category": CATEGORY, "event_count": EVENTS,
        "frozen_json_order": True, "physical_chronology_claimed": False,
        "class_json_sha256": contract["class_json_sha256"],
        "query_tape_sha256": contract["query_tape_sha256"],
        "query_positions_sha256": contract["query_positions_sha256"],
        "query_count": QUERIES, "query_rate": QUERIES / EVENTS,
        "query_policy": "Q-A-TS 3%, Seed 37; label blind",
        "support_count": 4, "support_group_sha256": contract["support_group_sha256"],
        "c64_count": C64_COUNT, "c64_is_additional_to_supports": True,
        "c64_score_tape_sha256": contract["c64_score_tape_sha256"],
        "fixed_support_sha256": contract["fixed_support_sha256_expected"],
        "fixed_threshold_c64_q90": float(contract["theta"]),
        "sigma_c64": float(contract["sigma"]),
        "stage05_fingerprints": contract["stage05_hashes"],
        "r3_reference_run_id": "r3_fullstream_2x2_20261102_a7",
        "r4_reference_run_id": "r4_fpsa_audiojack_20261102_a1",
        "b3_implementation": "experiments.calibration.controllers.StateConsistentQuantile",
        "b3_delta": B3_DELTA, "b3_min_normals": B3_MIN_NORMALS,
        "b3_state_streams": ["G", "GC", "GC-FPSA"],
        "writer_and_correction_threshold": "frozen R4 C64 higher-q90",
        "runtime_limit_seconds": SMOKE_RUNTIME_SECONDS if mode == "SMOKE_ONLY" else RUNTIME_SECONDS,
        "minimum_free_space_bytes": MIN_FREE_BYTES,
        "free_space_before_namespace_creation": int(free_bytes),
        "output_limit_bytes": OUTPUT_LIMIT, "failure_reserve_bytes": FAILURE_RESERVE,
        "output_budget_initial": budget.audit(),
        "windows_job_object_required": True, "automatic_restart": False,
        "test_masks_opened": False, "unqueried_test_labels_stage_a": False,
        "offline_outcome_evaluation_performed": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    budget.write_json("run_manifest.json", manifest)
    return output


def _write_public_deliverables(output: Path, aggregate: Mapping[str, Any],
                               context: Mapping[str, Any]) -> None:
    report_path, aggregate_path = REPORT_DIR / "REPORT.md", REPORT_DIR / "aggregate_results.json"
    if REPORT_DIR.exists() or report_path.exists() or aggregate_path.exists():
        raise FileExistsError(f"R5 tracked report path exists; refusing overwrite: {REPORT_DIR}")
    serialized = json.dumps(aggregate, sort_keys=True)
    forbidden = ("relative_id", "sample_key", "anomaly_class", "mask_path",
                 "event_id_sha256", "official_sample_key")
    if any(term in serialized for term in forbidden):
        raise R5AudiojackError("R5 public aggregate contains private event identity or label fields")
    metrics = aggregate["full_stream_metrics"]
    contrasts = aggregate["required_decision_contrasts"]
    rows = ["| Arm | BA | FPR | FNR | Accuracy | TP | TN | FP | FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("G-Frozen", "GC-Frozen", "FPSA-Frozen", "G-B3", "GC-B3", "FPSA-B3"):
        item = metrics[arm]
        confusion = item["confusion"]
        rows.append("| {arm} | {ba:.6f} | {fpr:.6f} | {fnr:.6f} | {acc:.6f} | {tp} | {tn} | {fp} | {fn} |".format(
            arm=arm, ba=item["balanced_accuracy"], fpr=item["fpr"], fnr=item["fnr"],
            acc=item["accuracy"], **confusion))
    contrast_lines = ["| Contrast | Δ BA | Δ FPR | Δ FNR |", "|---|---:|---:|---:|"]
    for name in ("FPSA-B3_minus_FPSA-Frozen", "FPSA-B3_minus_GC-B3",
                 "FPSA-B3_minus_G-B3", "GC-B3_minus_G-B3"):
        item = contrasts[name]
        contrast_lines.append(
            f"| {name.replace('_minus_', ' − ')} | "
            f"{item['balanced_accuracy_difference']:+.6f} | "
            f"{item['fpr_difference']:+.6f} | {item['fnr_difference']:+.6f} |")
    worst_lines = ["| Arm | Worst complete 512 event BA | Window | Class counts (normal/anomaly) |",
                   "|---|---:|---|---|"]
    for arm in ("G-Frozen", "GC-Frozen", "FPSA-Frozen", "G-B3", "GC-B3", "FPSA-B3"):
        item = aggregate["worst_eligible_temporal_window"][arm][
            "worst_complete_512_event_window"]
        worst_lines.append(
            f"| {arm} | {item['balanced_accuracy']:.6f} | "
            f"[{item['start_position']}, {item['end_position_exclusive']}) | "
            f"{item['normal_events']}/{item['anomalous_events']} |")
    report = f"""# R5-AUDIOJACK-B3-FPSA — Results

**Status:** COMPLETE. Retrospective exposed development-class validation; not independent validation or production-readiness evidence.

## Primary endpoint

Prequential Balanced Accuracy across all {EVENTS:,} audiojack events in frozen TEST JSON order. Physical chronology is not asserted. Query tape: {QUERIES} events ({QUERIES / EVENTS:.4%}); first anomalous queried feedback at position {FIRST_ANOMALOUS_QUERY}.

{chr(10).join(rows)}

## Required decision contrasts

{chr(10).join(contrast_lines)}

All differences are left arm minus right arm. The query tape and outcomes are retrospective development data. Thresholds do not enter Greedy memory or correction updates.

## Temporal windows and post-1305

Worst defined complete 512 event windows:

{chr(10).join(worst_lines)}

The predeclared 512 event window metrics, partial 231 event tail, undefined single-class windows, all post-1305 metrics, and full threshold trajectories are in `aggregate_results.json`.

## Parity and work performed

- R4 G/GC/GC-FPSA numerical parity: {aggregate['native_r4_parity']['events']} events; maximum absolute errors G={aggregate['native_r4_parity']['G_max_abs_error']:.9g}, GC={aggregate['native_r4_parity']['GC_max_abs_error']:.9g}, FPSA={aggregate['native_r4_parity']['FPSA_max_abs_error']:.9g}.
- Initial 64-reference C64 parity and changed-state cached/native representative parity passed. C64 rescore calls, rounds, elapsed time, and B3 state-update counts are recorded in the aggregate.
- Stage A score decisions were sealed before offline outcomes were read. Queried labels were disclosed only after each current score row flushed. No TEST masks were opened.
- B3 implementation: unchanged `StateConsistentQuantile`, delta={B3_DELTA}, min_normals={B3_MIN_NORMALS}; three independent complete scorer-state streams.
- Source-group paired bootstrap intervals are descriptive only. Seed={BOOTSTRAP_SEED}; {BOOTSTRAP_REPLICATES} resamples.

## Deployment interpretation

The hypothetical 1% FPR / 5% FNR scenario is illustrative and is not a factory-approved risk budget. This run uses one already-exposed development category, an artificial fixed label-blind 3% query tape, frozen JSON ordering, four deployed supports plus 64 additional protected TRAIN normals, and no independent production-domain data. It does not establish factory readiness or reproduce the original paper's 30-class controller protocol.

## Provenance

- Parent source commit: `{EXPECTED_PARENT}`
- Branch: `{EXPECTED_BRANCH}`
- R5 runner SHA-256: `{context['runner_sha256']}`
- CPU test file SHA-256: `{context['test_sha256']}`
- CPU tests: `{context['cpu_test_summary']}`
- 256 event GPU smoke: `{context['smoke_status']}`; output `{SMOKE_ID}`
- Failed smoke attempt `{FAILED_SMOKE_ID}` retained unchanged after the first query-history append contract error; retry used the fresh namespace `{SMOKE_ID}`.
- Full output namespace: `{output.relative_to(ROOT).as_posix()}`
- Full score rows SHA-256: `{aggregate['score_rows_sha256']}`
- Stage A seal SHA-256: `{aggregate['stage_a_seal_sha256']}`
- Complete run SHA-256: `{_sha256_file(output / 'complete_seal.json')}`
- Output byte budget: {OUTPUT_LIMIT:,} bytes with {FAILURE_RESERVE:,} bytes reserved for failure metadata.

Local commit contains this report, aggregate, runner, and focused tests only. No push performed.
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")


def run_experiment() -> dict[str, Any]:
    import experiments.calibration.r3_fullstream_2x2 as r3
    started = time.monotonic()
    repo = _repository_identity()
    if not Path(sys.executable).samefile(Path(r"D:\anaconda3\envs\torch\python.exe")):
        raise R5AudiojackError(f"R5 requires the frozen interpreter; found {sys.executable}")
    if not DATASET_ROOT.is_dir() or not JSON_ROOT.is_dir():
        raise R5AudiojackError("Authorized Real-IAD audiojack roots are unavailable")
    if REPORT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite deliverable directory {REPORT_DIR}")
    _output_path(RUN_ID)
    _output_path(SMOKE_ID)
    free_before = shutil.disk_usage(OUTPUT_BASE).free
    if free_before < MIN_FREE_BYTES:
        raise R5AudiojackError(f"Need 10 GiB free before run; found {free_before} bytes")
    contract = _load_audiojack_contract(verify_images=True)
    r3_rows = _load_r3_audiojack_reference(contract)
    r4_rows, _ = _load_r4_audiojack_reference(contract)
    if len(r3_rows) != EVENTS or len(r4_rows) != EVENTS:
        raise R5AudiojackError("R3/R4 audiojack score reference is incomplete")
    tests_path = ROOT / "tests" / "calibration" / "test_r5_audiojack_b3_fpsa_runner.py"
    if not tests_path.is_file():
        raise R5AudiojackError("Focused R5 CPU test file is missing")
    compiled = subprocess.run(
        [sys.executable, "-m", "py_compile", str(Path(__file__).resolve()), str(tests_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    if compiled.returncode:
        raise R5AudiojackError(f"R5 compilation failed: {compiled.stderr[-2500:]}")
    test_command = [sys.executable, "-m", "pytest", "-q",
                    "tests/calibration/test_r5_audiojack_b3_fpsa_runner.py",
                    "tests/calibration/test_r4_fpsa_codebook.py"]
    test_env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
                    R5_AUDIOJACK_SYNTHETIC_CPU_ONLY="1")
    test_result = subprocess.run(test_command, cwd=ROOT, env=test_env,
                                 capture_output=True, text=True, timeout=360)
    test_summary = (test_result.stdout + "\n" + test_result.stderr).strip()
    if test_result.returncode:
        raise R5AudiojackError(
            f"Focused R5 CPU tests failed ({test_result.returncode}): {test_summary[-5000:]}")
    cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    token = os.urandom(32).hex()
    smoke_free = shutil.disk_usage(OUTPUT_BASE).free
    if smoke_free < MIN_FREE_BYTES:
        raise R5AudiojackError(f"Need 10 GiB free before smoke; found {smoke_free} bytes")
    smoke = _initialize_run(SMOKE_ID, "SMOKE_ONLY", contract, token,
                            free_bytes=smoke_free)
    smoke_result = _launch_child(smoke, "STAGE_A", token,
                                 timeout=SMOKE_RUNTIME_SECONDS,
                                 cuda_visibility=cuda_visibility, smoke_events=256)
    if smoke_result.status != "COMPLETED" or smoke_result.child_exit_code != 0:
        raise R5AudiojackError(f"Bounded R5 audiojack GPU smoke failed: {asdict(smoke_result)}")
    smoke_history = [json.loads(line) for line in
                     (smoke / "attempt_history.jsonl").read_text(encoding="utf-8").splitlines()]
    smoke_receipt = next((row for row in smoke_history
                          if row.get("record_type") == "category_complete"), None)
    if (not smoke_receipt or smoke_receipt.get("events_processed") != 256 or
            smoke_receipt.get("query_events_processed", 0) <= 0 or
            smoke_receipt.get("c64_cache_changed_state_native_score_parity", {}).get(
                "passed") is not True):
        raise R5AudiojackError("R5 smoke did not exercise queried updates and C64 changed-state parity")
    R5OutputBudget(smoke).write_json("supervisor_status.json", {
        "schema": "r5.audiojack.b3_fpsa.supervisor.v1",
        "status": "GPU_SMOKE_PASS_NOT_EVALUATED", "run_id": SMOKE_ID,
        "events": 256, "query_events": smoke_receipt["query_events_processed"],
        "windows_job_object_process_tree_containment": True,
        "changed_state_c64_native_parity": True, "automatic_restart": False,
        "offline_evaluation_performed": False,
        "wall_seconds": time.monotonic() - started,
    })

    full_free = shutil.disk_usage(OUTPUT_BASE).free
    if full_free < MIN_FREE_BYTES:
        raise R5AudiojackError(f"Need 10 GiB free before full run; found {full_free} bytes")
    output = _initialize_run(RUN_ID, "FULL_STAGE_A", contract, token,
                             free_bytes=full_free)
    child = _launch_child(output, "STAGE_A", token, timeout=RUNTIME_SECONDS,
                          cuda_visibility=cuda_visibility)
    if child.status != "COMPLETED" or child.child_exit_code != 0:
        raise R5AudiojackError(f"Full R5 audiojack Stage A child failed: {asdict(child)}")
    stage_a = _seal_stage_a(output, contract, r4_rows)
    if stage_a.get("status") != "SEALED_COMPLETE":
        raise R5AudiojackError("R5 Stage A seal failed; Stage B will not run")
    stage_b = _launch_child(output, "STAGE_B", token, timeout=RUNTIME_SECONDS,
                            cuda_visibility=None)
    if stage_b.status != "COMPLETED" or stage_b.child_exit_code != 0:
        raise R5AudiojackError(f"CPU R5 audiojack Stage B failed: {asdict(stage_b)}")
    aggregate = _read_json(output / "aggregate_results.json")
    complete = _read_json(output / "complete_seal.json")
    if (complete.get("status") != "COMPLETE" or
            complete.get("stage_a_seal_sha256") != _sha256_file(output / "stage_a_seal.json") or
            complete.get("stage_b_seal_sha256") != _sha256_file(output / "stage_b_seal.json")):
        raise R5AudiojackError("R5 final Stage A/Stage B completion seal failed")
    R5OutputBudget(output).write_json("supervisor_status.json", {
        "schema": "r5.audiojack.b3_fpsa.supervisor.v1", "status": "COMPLETE",
        "run_id": RUN_ID, "stage_a_seal_sha256": _sha256_file(output / "stage_a_seal.json"),
        "stage_b_seal_sha256": _sha256_file(output / "stage_b_seal.json"),
        "runtime_limit_seconds_per_child": RUNTIME_SECONDS,
        "windows_job_object_process_tree_containment": True,
        "automatic_restart": False, "output_budget": R5OutputBudget(output).audit(),
        "minimum_free_space_bytes": MIN_FREE_BYTES,
        "free_space_before_full_launch": full_free,
    })
    _write_public_deliverables(output, aggregate, {
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "test_sha256": _sha256_file(tests_path),
        "cpu_test_summary": "PASS: " + test_result.stdout.strip().splitlines()[-1],
        "smoke_status": "GPU_SMOKE_PASS_NOT_EVALUATED",
        "failed_smoke_id": FAILED_SMOKE_ID,
        "smoke_retry_id": SMOKE_ID,
        "source_parent_commit": repo["source_parent_commit"],
        "elapsed_seconds": time.monotonic() - started,
    })
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-r5-audiojack-b3-fpsa", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stage-a-child", action="store_true")
    parser.add_argument("--stage-b-child", action="store_true")
    parser.add_argument("--smoke-events", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.authorize_r5_audiojack_b3_fpsa:
        parser.error("explicit R5 task authorization flag is required")
    if args.stage_a_child and args.stage_b_child:
        parser.error("child stage flags are mutually exclusive")
    if args.stage_a_child or args.stage_b_child:
        if not args.run_id or not os.environ.get("R5_AUDIOJACK_AUTH_TOKEN"):
            parser.error("supervised child requires run ID and ephemeral authorization token")
        if args.stage_a_child:
            _stage_a_child(args.run_id, os.environ["R5_AUDIOJACK_AUTH_TOKEN"], args.smoke_events)
        else:
            if args.smoke_events:
                parser.error("Stage B cannot receive a smoke event limit")
            _stage_b_child(args.run_id, os.environ["R5_AUDIOJACK_AUTH_TOKEN"])
        return 0
    if args.run_id is not None or args.smoke_events:
        parser.error("run ID/event limit are allowed only for supervised child mode")
    result = run_experiment()
    print(json.dumps({"status": result.get("status"),
                      "primary_endpoint": result.get("primary_endpoint"),
                      "run_id": RUN_ID}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
