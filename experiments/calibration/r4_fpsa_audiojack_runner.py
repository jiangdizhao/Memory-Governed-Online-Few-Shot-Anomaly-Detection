"""R4-FPSA-AUDIOJACK: one frozen, retrospective single-class mechanism test.

Stage A writes and seals all G/GC/GC-FPSA scores before any offline outcome
projection.  The only labels visible during the stream are queried feedback
disclosed after that event's score row has been durably flushed.
"""
from __future__ import annotations

import argparse
import hashlib
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
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PARENT = "90d15a8d3aced956fe50a43c3671b9c4ca870397"
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
RUN_ID = "r4_fpsa_audiojack_20261102_a1"
SMOKE_ID = RUN_ID + "_smoke"
CATEGORY = "audiojack"
EVENTS = 3815
QUERIES = 114
FIRST_ANOMALOUS_QUERY = 1305
FPSA_MATCH_THROUGH = 1305
RUNTIME_SECONDS = 120 * 60
SMOKE_RUNTIME_SECONDS = 15 * 60
MIN_FREE_BYTES = 10 * 1024**3
AUTHORIZATION = "R4-FPSA-AUDIOJACK user-authorized single-class retrospective development mechanism validation"
RUN_SCHEMA = "r4.fpsa.audiojack.run.v1"
SCORE_SCHEMA = "r4.fpsa.audiojack.score.v1"
HISTORY_SCHEMA = "r4.fpsa.audiojack.history.v1"
STAGE_A_SCHEMA = "r4.fpsa.audiojack.stage_a_seal.v1"
STAGE_B_SCHEMA = "r4.fpsa.audiojack.stage_b_seal.v1"
ATOL = 1e-5
RTOL = 1e-4
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261102
OUTPUT_BASE = ROOT / "outputs" / "calibration"
STAGE05 = OUTPUT_BASE / "stage05_exploratory_20261029"
R3_REFERENCE = OUTPUT_BASE / "r3_fullstream_2x2_20261102_a7"
DATASET_ROOT = Path(r"G:\dataset\realiad")
JSON_ROOT = Path(r"G:\dataset\realiad\jsons\realiad_jsons")
REPORT_DIR = ROOT / "docs" / "calibration" / "audit" / "r4_fpsa_audiojack_20261102"
SOURCE_PATHS = (
    "experiments/calibration/r3_fullstream_2x2.py",
    "experiments/calibration/r1_matched_correction_pilot.py",
    "experiments/calibration/r2_exp2a_c1_strength.py",
    "experiments/calibration/r4_fpsa_codebook.py",
    "experiments/calibration/native.py",
    "experiments/calibration/r1_runtime_supervisor.py",
    "experiments/calibration/stage05.py",
    "experiments/calibration/stage1a_p12_label_provider.py",
    "experiments/calibration/profiles.py",
    "experiments/calibration/controllers.py",
    "experiments/calibration/query_tape.py",
    "experiments/p0_protocol_audit.py",
    "experiments/risk_safe/native_bridge.py",
    "experiments/v02_memory.py",
    "test_phase3_AIF_simplified_v22.py",
    "vmb_visionad_new.py",
)


class R4AudiojackError(RuntimeError):
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
        raise R4AudiojackError(f"Expected JSON object at {path.name}")
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
        raise R4AudiojackError(
            f"Source identity mismatch: expected {EXPECTED_BRANCH}@{EXPECTED_PARENT}, "
            f"found {branch}@{head}")
    modified = _git("status", "--porcelain", "--", *SOURCE_PATHS)
    if modified:
        raise R4AudiojackError(f"Frozen dependency source files are dirty: {modified}")
    return {"branch": branch, "source_parent_commit": head}


def _source_binding(manifest: Mapping[str, Any]) -> None:
    _repository_identity()
    if (manifest.get("source_parent_commit") != EXPECTED_PARENT or
            manifest.get("branch") != EXPECTED_BRANCH or
            manifest.get("runner_source_sha256") != _sha256_file(Path(__file__).resolve()) or
            manifest.get("source_fingerprints") != _source_hashes()):
        raise R4AudiojackError("R4 source/branch identity differs from its immutable manifest")


def _output_path(run_id: str, *, existing: bool = False) -> Path:
    if run_id not in {RUN_ID, SMOKE_ID}:
        raise R4AudiojackError("Only the fixed R4 audiojack run and smoke IDs are authorized")
    parent = OUTPUT_BASE.resolve(strict=True)
    path = OUTPUT_BASE / run_id
    if existing:
        if path.is_symlink() or not path.is_dir() or path.resolve(strict=True).parent != parent:
            raise R4AudiojackError("Existing R4 output path is missing or escapes its namespace")
    else:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"R4 output exists; restart/resume is forbidden: {path.name}")
        if path.parent.resolve(strict=True) != parent:
            raise R4AudiojackError("R4 output path escaped outputs/calibration")
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
        raise R4AudiojackError("Frozen Stage 0.5 input receipt is incompatible")
    archive = STAGE05 / "query_tapes.json.gz"
    archive_sha = _sha256_file(archive)
    if archive_sha != run.get("query_tape_archive_sha256") or archive_sha != frozen.get("query_tapes_file_sha256"):
        raise R4AudiojackError("Frozen query tape archive hash mismatch")
    with gzip.open(archive, "rt", encoding="utf-8") as stream:
        tapes = json.load(stream)

    info = frozen.get("classes", {}).get(CATEGORY)
    if not isinstance(info, dict):
        raise R4AudiojackError("Frozen audiojack category receipt is absent")
    source_json = JSON_ROOT / "audiojack.json"
    if not source_json.is_file() or _sha256_file(source_json) != info.get("class_json_sha256"):
        raise R4AudiojackError("Audiojack public JSON fingerprint differs from Stage 0.5")
    train, stream_rows, class_sha = stage05.read_class_stream(
        DATASET_ROOT, JSON_ROOT, CATEGORY, test_labels=False)
    if class_sha != info.get("class_json_sha256") or len(stream_rows) != EVENTS or \
            int(info.get("stream_images", -1)) != EVENTS or len(info.get("stream", [])) != EVENTS:
        raise R4AudiojackError("Audiojack frozen stream does not contain exactly 3,815 events")
    # Reproduce only the authorized seed-0 support and its C64 calibration
    # bank. Do not calculate/read the other frozen support-seed identities.
    groups: dict[str, list[Any]] = {}
    for record in train:
        groups.setdefault(record.sample_key, []).append(record)
    normal_keys = sorted(key for key, views in groups.items()
                         if all(view.is_good for view in views))
    if len(normal_keys) < 68:
        raise R4AudiojackError("Audiojack has too few train-normal Source Groups for frozen C64")
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
        raise R4AudiojackError("Audiojack Support/C64 frozen selection or disjointness changed")
    frozen_rows = info["stream"]
    for position, (actual, expected) in enumerate(zip(stream_rows, frozen_rows)):
        if (actual.position != position or int(expected.get("position", -1)) != position or
                actual.relative_id != expected.get("relative_id") or
                actual.sample_key != expected.get("sample_key") or
                actual.opaque_group != expected.get("opaque_group") or
                actual.camera_id != int(expected.get("camera_id", -1))):
            raise R4AudiojackError(f"Audiojack public stream identity changed at {position}")
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
        raise R4AudiojackError("Audiojack frozen Q-A-TS Seed 37 digest mismatch")
    positions: list[int] = []
    group_by_position: dict[int, str] = {}
    for event in tape.get("events", []):
        position = int(event.get("position", -1))
        if not 0 <= position < EVENTS or position in group_by_position:
            raise R4AudiojackError("Audiojack Q-A-TS positions are malformed/duplicated")
        row = stream_rows[position]
        identity = _sha256_bytes(("image-identity\0" + row.relative_id).encode("utf-8"))
        if (event.get("official_sample_key") != row.opaque_group or
                int(event.get("camera_id", -1)) != row.camera_id or
                event.get("relative_id") != identity):
            raise R4AudiojackError(f"Audiojack Q-A-TS identity mismatch at {position}")
        positions.append(position)
        group_by_position[position] = row.opaque_group
    expected_query_count = int(math.floor(0.03 * EVENTS + 0.5))
    if (expected_query_count != QUERIES or int(tape.get("requested_budget", -1)) != QUERIES or
            len(positions) != QUERIES or len(set(group_by_position.values())) != QUERIES):
        raise R4AudiojackError("Audiojack query count/group uniqueness differs from frozen 114")
    positions.sort()

    c64_row = score_manifest["classes"][CATEGORY]["score_tapes"]["0"]
    c64_path = (STAGE05 / str(c64_row["file"])).resolve(strict=True)
    if STAGE05.resolve(strict=True) not in c64_path.parents:
        raise R4AudiojackError("Audiojack C64 score tape escaped its frozen directory")
    if (int(c64_row.get("c64_score_count", -1)) != 64 or
            _sha256_file(c64_path) != c64_row.get("sha256") or
            c64_row.get("c64_group_sha256") != info["c64"].get("group_sha256")):
        raise R4AudiojackError("Audiojack C64 score tape fingerprint/identity failed")
    with np.load(c64_path, allow_pickle=False) as archive_data:
        c64_scores = np.asarray(archive_data["c64_scores"], dtype=np.float64)
        c64_identity = str(archive_data["c64_identity_sha256"].item())
    if (c64_scores.shape != (64,) or not np.isfinite(c64_scores).all() or
            c64_identity != info["c64"].get("image_sha256")):
        raise R4AudiojackError("Audiojack C64 score values or image identity are invalid")
    theta = float(theta0_from_c64(c64_scores.tolist()))
    sigma = max(float(c64_scores.std(ddof=1)), 0.02)

    support_groups = {_sha256_bytes(("official-sample-group\0" + str(x)).encode())
                      for x in support0["groups"]}
    c64_groups = {_sha256_bytes(("official-sample-group\0" + str(x)).encode())
                  for x in info["c64"]["groups"]}
    test_groups = {str(row.opaque_group) for row in stream_rows}
    if (len(support_groups) != 4 or len(c64_groups) != 64 or support_groups & c64_groups or
            support_groups & test_groups or c64_groups & test_groups):
        raise R4AudiojackError("Audiojack Support/C64/TEST Source Groups overlap")
    support_info = support0
    support_paths = [Path(stage05._absolute_image_path(
        DATASET_ROOT, info["prefix"], str(relative))) for relative in support_info["images"]]
    event_paths = [Path(row.image_path) for row in stream_rows]
    if verify_images and any(not path.is_file() for path in (*support_paths, *event_paths)):
        raise R4AudiojackError("An authorized audiojack Support/TEST image path is missing")
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
        raise R4AudiojackError("Existing R3 audiojack reference lacks valid completion/category seals")
    path = R3_REFERENCE / "stage_a_scores.jsonl"
    refs: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8") as stream:
        for position in range(EVENTS):
            line = stream.readline()
            if not line:
                raise R4AudiojackError("R3 audiojack score prefix ended early")
            row = json.loads(line)
            if (row.get("schema") != r3.SCORE_SCHEMA or row.get("category") != CATEGORY or
                    int(row.get("position", -1)) != position or
                    any(key in row for key in ("label", "anomaly_class", "mask_path")) or
                    not {"G", "GC"}.issubset(row.get("scores", {}))):
                raise R4AudiojackError(f"R3 audiojack sealed score row {position} is invalid")
            refs.append(row)
            digest.update(r3.canonical_bytes(row))
    if digest.hexdigest() != category_seal.get("category_score_rows_sha256"):
        raise R4AudiojackError("R3 audiojack score prefix differs from its category seal")
    query_positions = [int(row["position"]) for row in refs if row.get("query_tape_decision")]
    if (query_positions != list(contract["positions"]) or
            any(str(row.get("source_group_sha256")) != contract["stream"][int(row["position"])].opaque_group
                for row in refs)):
        raise R4AudiojackError("R3 audiojack source-group/query identity differs from frozen contract")
    return refs


def _skip_json_value(data: bytes, index: int) -> int:
    from experiments.calibration.stage1a_p12_label_provider import _skip_value
    return _skip_value(data, index)


def _json_projection(data: bytes, keys: set[str]) -> dict[str, Any]:
    """Project selected JSON object members while lexically skipping all others."""
    from experiments.calibration.stage1a_p12_label_provider import _string, _ws
    index = _ws(data, 0)
    if index >= len(data) or data[index] != ord("{"):
        raise R4AudiojackError("R3 history record is not a JSON object")
    index += 1
    out: dict[str, Any] = {}
    while True:
        index = _ws(data, index)
        if index >= len(data):
            raise R4AudiojackError("Truncated R3 history object")
        if data[index] == ord("}"):
            return out
        key, index = _string(data, index)
        index = _ws(data, index)
        if index >= len(data) or data[index] != ord(":"):
            raise R4AudiojackError("Malformed R3 history member")
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
        raise R4AudiojackError("Malformed R3 history object delimiter")


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
            raise R4AudiojackError("R3 audiojack writer reference ended unexpectedly")
        head = _json_projection(line, {"record_type", "category"})
        kind = head.get("record_type")
        keys = self._QUERY_KEYS if kind == "queried_feedback_and_updates" else self._COMPLETE_KEYS
        return _json_projection(line, keys)

    def query(self, position: int, event_id: str, group_sha: str) -> dict[str, Any]:
        while True:
            row = self._next()
            kind, category = row.get("record_type"), row.get("category")
            if category not in (None, CATEGORY):
                raise R4AudiojackError("R3 audiojack writer reference crossed into another category")
            if kind == "queried_feedback_and_updates":
                if (category != CATEGORY or int(row.get("position", -1)) != position or
                        row.get("event_id_sha256") != event_id or
                        row.get("source_group_sha256") != group_sha or
                        not isinstance(row.get("greedy_writer_action"), dict)):
                    raise R4AudiojackError(f"R3 writer receipt identity mismatch at {position}")
                return row["greedy_writer_action"]
            if kind == "category_complete" and category == CATEGORY:
                raise R4AudiojackError("R3 writer reference completed before a required query")

    def finish(self) -> dict[str, Any]:
        while True:
            row = self._next()
            if row.get("record_type") == "category_complete" and row.get("category") == CATEGORY:
                self.finished = True
                self.stream.close()
                return row
            if row.get("category") not in (None, CATEGORY):
                raise R4AudiojackError("R3 writer reference crossed category before audiojack receipt")


def _writer_parity(actual: Mapping[str, Any], expected: Mapping[str, Any], *, position: int) -> None:
    import experiments.calibration.r3_fullstream_2x2 as r3
    for key in ("input_score_source", "prediction", "candidate_ids_sha256",
                "candidate_features_sha256", "action", "eligibility_reason",
                "accepted_patches_delta", "skipped_patches_delta"):
        if actual.get(key) != expected.get(key):
            raise R4AudiojackError(f"Greedy writer parity mismatch at {position} field {key}")
    for key in ("score", "theta", "sigma"):
        if key in expected and not r3._score_tolerance(actual.get(key), expected[key]):
            raise R4AudiojackError(f"Greedy writer parity mismatch at {position} field {key}")


def _existing_output_path(run_id: str) -> Path:
    return _output_path(run_id, existing=True)


def _write_failure(output: Path, stage: str, exc: BaseException) -> None:
    if not output.is_dir():
        return
    try:
        import experiments.calibration.r3_fullstream_2x2 as r3
        r3._write_failure(output, stage, exc, {"category": CATEGORY, "run_id": output.name})
    except Exception:
        pass


def _verify_child_manifest(run_id: str, token: str, expected_mode: str) -> tuple[Path, dict[str, Any]]:
    import experiments.calibration.r3_fullstream_2x2 as r3
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job
    if not current_process_is_in_job():
        raise R4AudiojackError("R4 experiment child must run inside the Windows Job Object")
    output = _existing_output_path(run_id)
    manifest = r3._read_json(output / "run_manifest.json")
    _source_binding(manifest)
    if (manifest.get("authorization") != AUTHORIZATION or manifest.get("mode") != expected_mode or
            manifest.get("run_id") != run_id or
            manifest.get("authorization_token_sha256") != _sha256_bytes(token.encode("ascii")) or
            manifest.get("runner_source_sha256") != _sha256_file(Path(__file__).resolve())):
        raise R4AudiojackError("Explicit R4 authorization/source manifest is incompatible")
    return output, manifest


def _stage_a_child(run_id: str, token: str, smoke_events: int = 0) -> dict[str, Any]:
    cuda_visibility_before = os.environ.get("CUDA_VISIBLE_DEVICES")
    import numpy as np
    import torch
    from types import SimpleNamespace
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r2_exp2a_c1_strength as exp2a
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.calibration.r4_fpsa_codebook import FPSASignedCorrectionCodebook
    from experiments.risk_safe.native_bridge import dynamic_codebooks
    import experiments.calibration.r3_fullstream_2x2 as r3
    from v02_memory import MemoryArm

    output, manifest = _verify_child_manifest(
        run_id, token, "SMOKE_ONLY" if smoke_events else "FULL_STAGE_A")
    if smoke_events and not 0 < smoke_events <= 256:
        raise R4AudiojackError("Audiojack GPU smoke limit must be 1..256 events")
    contract = _load_audiojack_contract(verify_images=True)
    refs = _load_r3_audiojack_reference(contract)
    output_budget = r3._budget(output)
    r3._restore_cuda_visibility(cuda_visibility_before)
    scores_out = r3.BufferedJsonl(output_budget, "stage_a_scores.jsonl")
    score_digest = hashlib.sha256()
    query_digest = hashlib.sha256()
    event_limit = min(EVENTS, smoke_events or EVENTS)
    allowed_queries = tuple(p for p in contract["positions"] if p < event_limit)
    group_map = {int(p): str(contract["query_groups"][p]) for p in allowed_queries}
    from experiments.calibration.stage1a_p12_label_provider import QueriedPositionLabelReader
    label_reader = QueriedPositionLabelReader(
        category=CATEGORY, path=JSON_ROOT / "audiojack.json",
        expected_source_sha256=contract["class_json_sha256"],
        positions=allowed_queries, group_sha256_by_position=group_map) if allowed_queries else None
    baseline_cursor = _R3WriterCursor(R3_REFERENCE / "attempt_history.jsonl")
    backbone = None
    writer = None
    counters = Counter()
    counters["queried_group_digests"] = set()
    counters["first_anomalous_query_position"] = None
    correction_stats: dict[str, Counter[str]] = {"GC": Counter(), "GC-FPSA": Counter()}
    source_groups: dict[str, dict[int, set[str]]] = {"GC": {}, "GC-FPSA": {}}
    active_peak = {"GC": 0, "GC-FPSA": 0}
    parity = {"matched_events": 0, "g_checked": 0, "gc_checked": 0,
              "g_max_abs_error": 0.0, "gc_max_abs_error": 0.0,
              "fpsa_gc_checked_through_1305": 0,
              "fpsa_gc_max_abs_error_through_1305": 0.0}
    memory_version = 0
    category_start = time.monotonic()
    try:
        torch.set_num_threads(1)
        if int(torch.cuda.device_count()) != 1 or not torch.cuda.is_available():
            raise R4AudiojackError("R4 Stage A requires exactly one visible authorized CUDA device")
        impl = p0.require_sources()
        model_receipt = r3._get_verified_model_receipt()
        restore_hub, runtime_info = r3._runtime_and_native_setup(torch, impl, model_receipt)
        cfg, _scorer_profile = r3._native_config(
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
            raise R4AudiojackError("Loaded DINO state differs from the verified local receipt")
        support_paths = [str(path) for path in contract["support_paths"]]
        support_features = p0.encode_reference_features(impl, backbone, support_paths, cfg)
        impl.set_seed(0)
        bank = p0.build_vmb(impl, support_features, support_paths, cfg)
        bank_receipt = r3._check_bank_receipt(
            bank, p0, contract["fixed_support_sha256_expected"], r1)
        r3._validate_correction_config(bank)
        gc_cfg = bank.corr_cfg
        gc_cb = bank.corr_cb["p0"]["id"][0]
        if int(gc_cb.K) != 256 or int(gc_cb.size()) != 0:
            raise R4AudiojackError("Native GC C1-Full Codebook did not start empty at K=256")
        fpsa_cfg = type(gc_cfg)(**asdict(gc_cfg))
        fpsa_cfg.enabled = False
        fpsa_cb = FPSASignedCorrectionCodebook(
            256, int(gc_cb.protos.shape[1]), gc_cb.device, fpsa_cfg)
        if gc_cb is fpsa_cb or gc_cb.protos is fpsa_cb.protos or int(fpsa_cb.size()) != 0:
            raise R4AudiojackError("GC and GC-FPSA mutable correction state is shared or nonempty")
        greedy_cfg = SimpleNamespace(flat=256, stm=256, ltm=1, buffer=1,
                                     min_events=2, promote_every=16,
                                     cap_bytes=2 * 1024**3)
        writer = MemoryArm("greedy_coreset", bank, greedy_cfg, seed=0)
        if any(int(cb.size()) != 0 or int(cb.K) != 256 for cb in dynamic_codebooks(bank)):
            raise R4AudiojackError("Common Greedy normal memory did not start empty at K=256")
        bank.corr_cfg.enabled = False
        bank.use_defect_bank = False
        theta, sigma = float(contract["theta"]), float(contract["sigma"])
        from experiments.calibration.r1_matched_correction_pilot import (
            _native_dense_score, _native_reference_and_debug, _c1_score_delta,
            corrected_dense_map, max_score, common_c0_greedy_writer_action,
        )
        score_rows = 0
        for position, meta in enumerate(contract["stream"][:event_limit]):
            image_path = contract["event_paths"][position]
            if not image_path.is_file():
                raise R4AudiojackError(f"Audiojack image disappeared at {position}")
            image_sha = r3.sha256_file(image_path)
            torch.cuda.synchronize()
            started = time.perf_counter()
            image, features = p0.encode_query(impl, backbone, str(image_path), cfg)
            torch.cuda.synchronize()
            counters["encoding_microseconds"] += int((time.perf_counter() - started) * 1e6)
            counters["actual_image_encodings"] += 1
            if (len(features) != 1 or tuple(features[0].shape) != (784, 4096) or
                    features[0].dtype != torch.float32 or not torch.isfinite(features[0]).all().item()):
                raise R4AudiojackError(f"Audiojack native feature contract failed at {position}")
            group_sha = str(meta.opaque_group)
            queried = position in set(allowed_queries)
            base_map, top_ids, base_score, native_summary = r1._native_dense_score(
                impl, bank, image, features, cfg)
            if not np.array_equal(top_ids, np.asarray(
                    native_summary["layer0"]["topk_patch_idx"], dtype=np.int64)):
                raise R4AudiojackError(f"Native Top-32 receipt differs at {position}")
            if position == 0:
                ref_score, ref_ids, ref_map = r1._native_reference_and_debug(
                    impl, bank, image, features, cfg)
                if (not np.array_equal(ref_ids, top_ids) or not np.array_equal(ref_map, base_map) or
                        not r3._score_tolerance(ref_score, base_score)):
                    raise R4AudiojackError("Native dense map adapter failed first-event reference parity")
            selected = impl.l2_normalize(features[0].to(bank.device))[
                torch.as_tensor(top_ids, device=bank.device, dtype=torch.long)]
            delta_gc, diag_gc = _score_with_codebook(
                impl, bank, gc_cfg, gc_cb, source_groups["GC"], selected,
                group_sha, position, r1)
            delta_fpsa, diag_fpsa = _score_with_codebook(
                impl, bank, fpsa_cfg, fpsa_cb, source_groups["GC-FPSA"], selected,
                group_sha, position, r1)
            counters["source_exclusion_checks"] += 2
            gc_score = float(max_score(corrected_dense_map(base_map, top_ids, delta_gc)))
            fpsa_score = float(max_score(corrected_dense_map(base_map, top_ids, delta_fpsa)))
            scores = {"G": float(base_score), "GC": gc_score, "GC-FPSA": fpsa_score}
            if not all(math.isfinite(value) for value in scores.values()):
                raise R4AudiojackError(f"Nonfinite score at audiojack event {position}")

            ref = refs[position]
            if (ref.get("source_group_sha256") != group_sha or
                    bool(ref.get("query_tape_decision")) != queried or
                    ref.get("event_id_sha256") != r3._event_id(
                        CATEGORY, position, meta.relative_id, image_sha, group_sha) or
                    ref.get("source_identity_sha256") !=
                    _sha256_bytes(meta.relative_id.encode("utf-8")) or
                    ref.get("image_content_sha256") != image_sha):
                raise R4AudiojackError(f"R3 source-group/query parity failed at {position}")
            parity["matched_events"] += 1
            for arm, key in (("G", "g"), ("GC", "gc")):
                actual, expected = scores[arm], float(ref["scores"][arm])
                error = abs(actual - expected)
                parity[f"{key}_checked"] += 1
                parity[f"{key}_max_abs_error"] = max(parity[f"{key}_max_abs_error"], error)
                if not r3._score_tolerance(actual, expected):
                    raise R4AudiojackError(
                        f"R3 {arm} score parity failed at {position}: "
                        f"actual={actual:.9g} expected={expected:.9g} abs={error:.3g}")
            if position <= FPSA_MATCH_THROUGH:
                error = abs(fpsa_score - gc_score)
                parity["fpsa_gc_checked_through_1305"] += 1
                parity["fpsa_gc_max_abs_error_through_1305"] = max(
                    parity["fpsa_gc_max_abs_error_through_1305"], error)
                if not r3._score_tolerance(fpsa_score, gc_score):
                    raise R4AudiojackError(
                        f"GC-FPSA diverged before anomalous feedback at {position}; abs={error:.3g}")
            event_id = r3._event_id(
                CATEGORY, position, meta.relative_id, image_sha, group_sha)
            score_row = {
                "schema": SCORE_SCHEMA, "category": CATEGORY, "position": position,
                "event_id_sha256": event_id,
                "source_identity_sha256": _sha256_bytes(meta.relative_id.encode("utf-8")),
                "source_group_sha256": group_sha,
                "image_content_sha256": image_sha,
                "camera_id": int(meta.camera_id),
                "prequential_order_position": position,
                "timestamp_semantics": "metadata only; physical chronology not asserted",
                "query_tape_decision": queried, "theta_c64_q90": theta,
                "scores": scores,
                "predictions_at_frozen_threshold": {
                    arm: int(value >= theta) for arm, value in scores.items()},
                "greedy_memory_version_before": memory_version,
                "GC_correction_occupancy_before": int(gc_cb.size()),
                "GC_FPSA_correction_occupancy_before": int(fpsa_cb.size()),
                "GC_same_group_excluded_slots": int(diag_gc["same_source_excluded_anchor_slots"]),
                "GC_FPSA_same_group_excluded_slots": int(diag_fpsa["same_source_excluded_anchor_slots"]),
            }
            encoded = scores_out.add(score_row)
            score_digest.update(encoded)
            score_rows += 1
            if queried or (position + 1) % 256 == 0:
                scores_out.flush()
            if queried:
                label = r3._disclose_query_label_after_seal(
                    scores_out, query_decision=True, category=CATEGORY, position=position,
                    sample_key=meta.sample_key, label_reader=label_reader)
                if label not in (0, 1):
                    raise R4AudiojackError("Queried audiojack feedback was not binary")
                if group_sha in counters["queried_group_digests"]:
                    raise R4AudiojackError("A queried Source Group was counted more than once")
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
                    base_prediction = int(base_score >= theta)
                    outcome = {(0, 0): "TN", (0, 1): "FP",
                               (1, 0): "FN", (1, 1): "TP"}[(int(label), base_prediction)]
                    sign = 1 if int(label) == 1 else -1
                    _result, layer, trace_diag = exp2a._capture_update(
                        impl, bank, r1, branch, cfg_arm, cb_arm, features,
                        np.asarray(top_ids, dtype=np.int64), sign, float(base_score),
                        theta, outcome, position, source_groups[branch], group_sha)
                    integer_stats = {key: int(value) for key, value in layer.items()
                                     if isinstance(value, (int, np.integer))}
                    correction_stats[branch].update(integer_stats)
                    active_peak[branch] = max(active_peak[branch], int(cb_arm.size()))
                    update_detail: dict[str, Any] = {
                        "base_score": float(base_score), "base_prediction": base_prediction,
                        "outcome": outcome, "sign": sign, "native_stats": integer_stats,
                        "occupancy_after": int(cb_arm.size()),
                        "native_keep_count": int(sum(bool(x) for x in
                                                       trace_diag["native_gate_keep_mask"])),
                    }
                    if branch == "GC-FPSA":
                        update_detail["fpsa_diagnostics"] = cb_arm.diagnostics()
                    correction_updates[branch] = update_detail

                writer_start = time.perf_counter()
                audit_before = writer.audit()
                change = {"changed": False}

                def writer_update(*, position: int, feature_layers: Sequence[Any],
                                  candidate_ids: Any, score: float, theta: float,
                                  sigma: float) -> Mapping[str, Any]:
                    before = r1._memory_content_sha256(bank)
                    writer.update(impl, list(feature_layers), {0: candidate_ids.tolist()},
                                  position, score=score, theta=theta, sigma=sigma)
                    after = r1._memory_content_sha256(bank)
                    change["changed"] = before != after
                    audit_after = writer.audit()
                    return {
                        "accepted_patches_delta": int(audit_after["accepted_patches"] -
                                                      audit_before["accepted_patches"]),
                        "skipped_patches_delta": int(audit_after["skipped_patches"] -
                                                     audit_before["skipped_patches"]),
                        "_memory_content_changed": change["changed"],
                    }

                norm_candidates = impl.l2_normalize(features[0].to(bank.device))[
                    torch.as_tensor(top_ids, device=bank.device, dtype=torch.long)]
                writer_action = r1.common_c0_greedy_writer_action(
                    position=position, c0_score=float(base_score), theta=theta, sigma=sigma,
                    c0_prediction=int(base_score >= theta),
                    candidate_ids=np.asarray(top_ids, dtype=np.int64),
                    candidate_features=norm_candidates.detach().float().cpu().numpy(),
                    feature_layers=features, queried=True, disclosed_label=int(label),
                    eligibility=impl.select_informative_normal_write, update=writer_update)
                counters["writer_seconds_microseconds"] += int((time.perf_counter() - writer_start) * 1e6)
                memory_changed = bool(change["changed"])
                writer_action.pop("_memory_content_changed", None)
                if memory_changed:
                    memory_version += 1
                if writer_action.get("action") == "native_greedy_normal_update":
                    counters["greedy_write_events"] += 1
                    counters["greedy_accepted_patch_count"] += int(
                        writer_action.get("accepted_patches_delta", 0))
                    counters["greedy_skipped_patch_count"] += int(
                        writer_action.get("skipped_patches_delta", 0))
                expected_writer = baseline_cursor.query(position, event_id, group_sha)
                _writer_parity(writer_action, expected_writer, position=position)
                history = {
                    "schema": HISTORY_SCHEMA, "record_type": "queried_feedback_and_updates",
                    "category": CATEGORY, "position": position,
                    "event_id_sha256": event_id, "source_group_sha256": group_sha,
                    "disclosure_after_three_score_seal": True, "label": int(label),
                    "fpsa_feedback_registration": feedback_registration,
                    "correction_updates": correction_updates,
                    "greedy_writer_action": writer_action,
                    "greedy_memory_version_after": memory_version,
                    "greedy_memory_content_changed": memory_changed,
                }
                query_digest.update(r3._append_history(output_budget, history))
            if (position + 1) % 256 == 0:
                scores_out.flush()
                _append_history_r4(output_budget, {
                    "schema": HISTORY_SCHEMA, "record_type": "diagnostic_progress_checkpoint",
                    "category": CATEGORY, "sealed_through_position": position,
                    "score_rows_written": score_rows,
                    "query_feedback_count": counters["query_events"],
                    "greedy_memory_version": memory_version,
                    "greedy_memory_state_sha256": r1._memory_state_sha256(bank, memory_version),
                    "fixed_support_sha256": _fixed_bank_digest(bank),
                    "GC_correction_state_sha256": r1._correction_state_sha256(bank, source_groups["GC"]),
                    "GC_FPSA_correction_state_sha256": _correction_state_sha256(
                        bank, fpsa_cfg, fpsa_cb, source_groups["GC-FPSA"], r1),
                    "automatic_restart": False,
                })
                print(f"R4_AUDIOJACK_PROGRESS run={run_id} events={position+1}/{event_limit} "
                      f"queries={counters['query_events']}", flush=True)
            del image, features, base_map

        scores_out.flush()
        if label_reader is not None:
            if tuple(label_reader.decoded_positions) != allowed_queries:
                raise R4AudiojackError("Audiojack queried label reader did not consume exact tape prefix")
            label_reader.close()
        baseline_receipt = baseline_cursor.finish() if event_limit == EVENTS else None
        if baseline_receipt is None:
            baseline_cursor.stream.close()
        if event_limit == EVENTS and (
                score_rows != EVENTS or counters["query_events"] != QUERIES or
                counters["query_normal"] != 74 or counters["query_anomalous"] != 40):
            raise R4AudiojackError("Full audiojack Stage A did not process 3,815 events/114 queries")
        if counters["first_anomalous_query_position"] not in (None, FIRST_ANOMALOUS_QUERY):
            raise R4AudiojackError("First anomalous queried feedback position differs from frozen task")
        if _fixed_bank_digest(bank) != contract["fixed_support_sha256_expected"]:
            raise R4AudiojackError("Common Greedy writer altered the fixed Support bank")
        if gc_cb is fpsa_cb or gc_cb.protos is fpsa_cb.protos:
            raise R4AudiojackError("GC and GC-FPSA Codebooks became shared")
        writer_final = writer.audit()
        greedy_reference_parity = None
        if baseline_receipt is not None:
            greedy_reference_parity = {}
            for key, actual_key in (("greedy_write_events", "greedy_write_events"),
                                    ("greedy_accepted_patch_count", "greedy_accepted_patch_count"),
                                    ("greedy_skipped_patch_count", "greedy_skipped_patch_count")):
                expected_value = int(baseline_receipt.get(key, -1))
                actual_value = int(counters[actual_key])
                if expected_value != actual_value:
                    raise R4AudiojackError(f"R3 final Greedy writer parity failed for {key}")
                greedy_reference_parity[key] = {"actual": actual_value,
                                                "reference": expected_value,
                                                "match": True}
        result = {
            "schema": HISTORY_SCHEMA, "record_type": "category_complete",
            "category": CATEGORY, "events_processed": score_rows,
            "query_events_processed": counters["query_events"],
            "query_normal": counters["query_normal"],
            "query_anomalous": counters["query_anomalous"],
            "first_anomalous_query_position": counters["first_anomalous_query_position"],
            "score_rows_sha256": score_digest.hexdigest(),
            "query_receipts_sha256": query_digest.hexdigest(),
            "query_tape_sha256": contract["query_tape_sha256"],
            "query_positions_sha256": _sha256_json(list(allowed_queries)),
            "fixed_support_sha256": contract["fixed_support_sha256_expected"],
            "fixed_support_digest_evidence": bank_receipt,
            "fixed_support_digest_algorithm": "experiments.calibration.native._fixed_bank_digest.v1",
            "fixed_support_raw_prototype_sha256": p0.fixed_bank_profile(bank)["sha256"],
            "theta_c64_q90": theta, "sigma_c64_floor_002": sigma,
            "greedy_memory_version_final": memory_version,
            "greedy_memory_content_sha256_final": r1._memory_content_sha256(bank),
            "greedy_writer_audit_final": writer_final,
            "greedy_write_events": int(counters["greedy_write_events"]),
            "greedy_accepted_patch_count": int(counters["greedy_accepted_patch_count"]),
            "greedy_skipped_patch_count": int(counters["greedy_skipped_patch_count"]),
            "correction_stats": {name: dict(value) for name, value in correction_stats.items()},
            "correction_final_occupancy": {"GC": int(gc_cb.size()),
                                           "GC-FPSA": int(fpsa_cb.size())},
            "correction_final_sign_occupancy": {
                "GC": _sign_occupancy(gc_cb), "GC-FPSA": _sign_occupancy(fpsa_cb)},
            "correction_max_occupancy": active_peak,
            "fpsa_final_diagnostics": fpsa_cb.diagnostics(),
            "source_exclusion_checks": int(counters["source_exclusion_checks"]),
            "source_exclusion_maps_independent": source_groups["GC"] is not source_groups["GC-FPSA"],
            "parity": parity,
            "greedy_reference_parity": greedy_reference_parity,
            "runtime_profile": runtime_info["runtime_profile"],
            "hub_source_tree_sha256": runtime_info["hub_source_tree_sha256"],
            "checkpoint_sha256": runtime_info["checkpoint_sha256"],
            "loaded_model_state_sha256": loaded_model_sha,
            "support_encoder_invocations": sum(len(x) for x in support_features.values()),
            "support_and_model_setup_seconds": time.perf_counter() - setup_started,
            "actual_stream_image_encodings": int(counters["actual_image_encodings"]),
            "writer_seconds": counters["writer_seconds_microseconds"] / 1e6,
            "wall_seconds": time.monotonic() - category_start,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_cpu_rss_bytes": r1._process_rss_bytes(),
            "smoke_only": bool(smoke_events), "offline_evaluation_performed": False,
        }
        _append_history_r4(output_budget, result)
        if backbone is not None:
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


def _score_with_codebook(impl: Any, bank: Any, cfg: Any, cb: Any,
                         groups: Mapping[int, set[str]], selected: Any,
                         group_sha: str, position: int, r1: Any):
    prior_cfg, prior_cb = bank.corr_cfg, bank.corr_cb["p0"]["id"][0]
    try:
        bank.corr_cfg, bank.corr_cb["p0"]["id"][0] = cfg, cb
        cfg.enabled = True
        return r1._c1_score_delta(impl, bank, selected, groups, group_sha,
                                  event_position=position)
    finally:
        cfg.enabled = False
        bank.corr_cfg, bank.corr_cb["p0"]["id"][0] = prior_cfg, prior_cb


def _correction_state_sha256(bank: Any, cfg: Any, cb: Any,
                             groups: Mapping[int, set[str]], r1: Any) -> str:
    prior_cfg, prior_cb = bank.corr_cfg, bank.corr_cb["p0"]["id"][0]
    try:
        bank.corr_cfg, bank.corr_cb["p0"]["id"][0] = cfg, cb
        return r1._correction_state_sha256(bank, groups)
    finally:
        bank.corr_cfg, bank.corr_cb["p0"]["id"][0] = prior_cfg, prior_cb


def _append_history_r4(budget: Any, row: Mapping[str, Any]) -> bytes:
    import experiments.calibration.r3_fullstream_2x2 as r3
    return r3._append_history(budget, row)


def _sign_occupancy(cb: Any) -> dict[str, int]:
    beta = cb.beta[:cb.size()]
    return {"positive": int((beta > 0).sum().item()),
            "negative": int((beta < 0).sum().item()),
            "zero": int((beta == 0).sum().item()), "total": int(cb.size())}


def _seal_stage_a(output: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
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
                    any(key in row for key in ("label", "anomaly_class", "mask_path"))):
                raise R4AudiojackError(f"R4 Stage A row {position} failed score/label firewall")
            rows.append(row)
            score_digest.update(r3.canonical_bytes(row))
    history: list[dict[str, Any]] = []
    with history_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("record_type") in {"queried_feedback_and_updates", "category_complete"}:
                history.append(row)
    receipt = next((row for row in reversed(history)
                    if row.get("record_type") == "category_complete"), None)
    feedback = [row for row in history if row.get("record_type") == "queried_feedback_and_updates"]
    if (len(rows) != EVENTS or [int(row["position"]) for row in rows] != list(range(EVENTS)) or
            [int(row["position"]) for row in rows if row["query_tape_decision"]] != list(contract["positions"]) or
            len(feedback) != QUERIES or receipt is None or
            receipt.get("events_processed") != EVENTS or receipt.get("query_events_processed") != QUERIES):
        raise R4AudiojackError("R4 Stage A lacks the exact complete audiojack stream/query receipt")
    query_digest = hashlib.sha256()
    for row in feedback:
        position = int(row["position"])
        score = rows[position]
        label = row.get("label")
        if isinstance(label, bool) or label not in (0, 1):
            raise R4AudiojackError(f"R4 queried feedback label is not binary at {position}")
        base_score = float(score["scores"]["G"])
        threshold = float(score["theta_c64_q90"])
        base_prediction = int(base_score >= threshold)
        expected_outcome = {(0, 0): "TN", (0, 1): "FP",
                            (1, 0): "FN", (1, 1): "TP"}[(int(label), base_prediction)]
        expected_sign = 1 if int(label) == 1 else -1
        updates = row.get("correction_updates", {})
        greedy_action = row.get("greedy_writer_action", {})
        if (row.get("event_id_sha256") != score.get("event_id_sha256") or
                row.get("source_group_sha256") != score.get("source_group_sha256") or
                row.get("disclosure_after_three_score_seal") is not True or
                score.get("query_tape_decision") is not True or
                set(updates) != {"GC", "GC-FPSA"} or
                greedy_action.get("input_score_source") != "common_C0" or
                greedy_action.get("prediction") != base_prediction or
                not r3._score_tolerance(greedy_action.get("score"), base_score)):
            raise R4AudiojackError(f"R4 queried feedback receipt does not bind sealed event {position}")
        for arm, update in updates.items():
            if (not r3._score_tolerance(update.get("base_score"), base_score) or
                    update.get("base_prediction") != base_prediction or
                    update.get("outcome") != expected_outcome or
                    update.get("sign") != expected_sign):
                raise R4AudiojackError(
                    f"R4 {arm} correction update is not bound to common C0/feedback at {position}")
        if int(label) == 1 and greedy_action.get("action") != "verified_defect_no_normal_write":
            raise R4AudiojackError(
                f"Queried audiojack defect entered normal memory at event {position}")
        query_digest.update(r3.canonical_bytes(row))
    if (score_digest.hexdigest() != receipt.get("score_rows_sha256") or
            query_digest.hexdigest() != receipt.get("query_receipts_sha256")):
        raise R4AudiojackError("R4 Stage A score/feedback ledger hash differs from its child receipt")
    seal = {
        "schema": STAGE_A_SCHEMA, "status": "SEALED_COMPLETE", "run_id": output.name,
        "source_parent_commit": EXPECTED_PARENT, "category": CATEGORY,
        "event_count": EVENTS, "query_count": QUERIES,
        "score_rows_sha256": _sha256_file(scores_path),
        "category_score_rows_sha256": score_digest.hexdigest(),
        "query_receipts_sha256": query_digest.hexdigest(),
        "query_tape_sha256": contract["query_tape_sha256"],
        "query_positions_sha256": contract["query_positions_sha256"],
        "feedback_disclosure_order_verified": True,
        "current_score_row_fsynced_before_query_label": True,
        "unqueried_test_labels_accessed": False,
        "offline_outcome_labels_accessed": False,
        "test_masks_opened": False, "automatic_restart": False,
        "r3_reference_category_digest_verified": True,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    r3._budget(output).write_json("stage_a_seal.json", seal)
    return seal


def _bootstrap_ba(scores: Mapping[str, Any], labels: Any, groups: Any,
                  threshold: float) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    group = np.asarray(groups, dtype=str)
    unique = np.unique(group)
    if len(unique) < 2:
        return {"status": "UNDEFINED_FEWER_THAN_TWO_SOURCE_GROUPS"}
    inverse = np.searchsorted(unique, group)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = {"GC-FPSA_minus_GC": [], "GC-FPSA_minus_G": [], "GC_minus_G": []}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(unique), size=len(unique))
        counts = np.bincount(sampled, minlength=len(unique))
        weights = counts[inverse].astype(np.float64)
        metrics = {arm: _threshold_metrics(scores[arm], y, threshold, weights)
                   for arm in scores}
        pairs = (("GC-FPSA", "GC", "GC-FPSA_minus_GC"),
                 ("GC-FPSA", "G", "GC-FPSA_minus_G"),
                 ("GC", "G", "GC_minus_G"))
        for left, right, name in pairs:
            a, b = metrics[left]["balanced_accuracy"], metrics[right]["balanced_accuracy"]
            if a is not None and b is not None:
                deltas[name].append(float(a - b))
    return {
        "status": "DESCRIPTIVE_SOURCE_GROUP_PAIRED_BOOTSTRAP",
        "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
        "unit": "physical Source Group, resampled with replacement; events inherit group multiplicity",
        "intervals_2_5_97_5_percent": {
            name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            if values else None for name, values in deltas.items()},
        "independent_confirmation": False,
    }


def _threshold_metrics(scores: Any, labels: Any, threshold: float,
                       weights: Any | None = None) -> dict[str, Any]:
    import experiments.calibration.r3_fullstream_2x2 as r3
    result = r3._threshold_metrics(scores, labels, threshold, weights)
    confusion = result["confusion"]
    total = sum(confusion.values())
    result["accuracy"] = ((confusion["tp"] + confusion["tn"]) / total) if total else None
    return result


def _stage_b_child(run_id: str, token: str) -> dict[str, Any]:
    import numpy as np
    import torch
    import experiments.calibration.r3_fullstream_2x2 as r3
    output, manifest = _verify_child_manifest(run_id, token, "FULL_STAGE_A")
    if torch.cuda.device_count() != 0 or torch.cuda.is_initialized():
        raise R4AudiojackError("Stage B must run CPU-only with CUDA hidden and uninitialized")
    seal_path, scores_path = output / "stage_a_seal.json", output / "stage_a_scores.jsonl"
    seal = r3._read_json(seal_path)
    if (seal.get("status") != "SEALED_COMPLETE" or seal.get("event_count") != EVENTS or
            seal.get("category") != CATEGORY or seal.get("score_rows_sha256") != _sha256_file(scores_path) or
            seal.get("query_tape_sha256") != manifest.get("query_tape_sha256")):
        raise R4AudiojackError("Stage B requires a complete verified audiojack Stage A seal")
    rows = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines()]
    if (len(rows) != EVENTS or [int(row.get("position", -1)) for row in rows] != list(range(EVENTS)) or
            any(row.get("schema") != SCORE_SCHEMA or row.get("category") != CATEGORY or
                "label" in row or "anomaly_class" in row or "mask_path" in row
                for row in rows)):
        raise R4AudiojackError("Stage B score rows violate sealed order or label firewall")
    _load_audiojack_contract(verify_images=False)
    label_path = JSON_ROOT / "audiojack.json"
    if _sha256_file(label_path) != manifest.get("class_json_sha256"):
        raise R4AudiojackError("Audiojack TEST JSON changed after Stage A preflight")
    labels = r3._read_stage_b_label_projection(label_path, EVENTS)
    if labels.shape != (EVENTS,) or not np.isin(labels, (0, 1)).all():
        raise R4AudiojackError("Restricted Stage B label projection failed")
    history = [json.loads(line) for line in (output / "attempt_history.jsonl").read_text(
        encoding="utf-8").splitlines()]
    receipts = [row for row in history if row.get("record_type") == "queried_feedback_and_updates"]
    by_position = {int(row["position"]): row for row in receipts}
    query_positions = [int(row["position"]) for row in rows if row["query_tape_decision"]]
    if set(by_position) != set(query_positions) or len(receipts) != QUERIES:
        raise R4AudiojackError("Stage B private feedback ledger differs from sealed query tape")
    for pos, receipt in by_position.items():
        if (int(receipt.get("label", -1)) != int(labels[pos]) or
                receipt.get("event_id_sha256") != rows[pos].get("event_id_sha256") or
                receipt.get("source_group_sha256") != rows[pos].get("source_group_sha256") or
                receipt.get("disclosure_after_three_score_seal") is not True):
            raise R4AudiojackError(f"Stage B queried feedback binding failed at position {pos}")
    threshold = float(rows[0]["theta_c64_q90"])
    if any(float(row["theta_c64_q90"]) != threshold for row in rows):
        raise R4AudiojackError("Stage A threshold changed within the frozen stream")
    scores = {arm: np.asarray([float(row["scores"][arm]) for row in rows], dtype=np.float64)
              for arm in ("G", "GC", "GC-FPSA")}
    groups = np.asarray([str(row["source_group_sha256"]) for row in rows], dtype=str)
    query_mask = np.asarray([bool(row["query_tape_decision"]) for row in rows], dtype=bool)
    predictions = {arm: (score >= threshold).astype(np.int8) for arm, score in scores.items()}
    full = {arm: _threshold_metrics(score, labels, threshold) for arm, score in scores.items()}
    after = slice(FIRST_ANOMALOUS_QUERY + 1, EVENTS)
    post = {arm: _threshold_metrics(score[after], labels[after], threshold)
            for arm, score in scores.items()}
    transitions = {
        "GC-FPSA_vs_GC": r3._confusion_transitions(predictions["GC"], predictions["GC-FPSA"], labels),
        "GC-FPSA_vs_G": r3._confusion_transitions(predictions["G"], predictions["GC-FPSA"], labels),
        "GC_vs_G": r3._confusion_transitions(predictions["G"], predictions["GC"], labels),
    }
    feedback_outcomes = {
        "query_count": int(query_mask.sum()),
        "normal": int(sum(int(row["label"]) == 0 for row in receipts)),
        "anomalous": int(sum(int(row["label"]) == 1 for row in receipts)),
        "first_anomalous_query_position": min(pos for pos, row in by_position.items()
                                                 if int(row["label"]) == 1),
    }
    if feedback_outcomes != {"query_count": QUERIES, "normal": 74, "anomalous": 40,
                             "first_anomalous_query_position": FIRST_ANOMALOUS_QUERY}:
        raise R4AudiojackError("Audiojack queried feedback counts/first anomaly differ from task receipt")
    # Coarsen the trajectory into fixed 512-event bins so the public aggregate
    # does not expose an event-by-event queried label sequence.
    trajectory = []
    for end in (511, 1023, 1535, 2047, 2559, 3071, 3583, 3814):
        in_bin = [row for pos, row in by_position.items()
                  if end - 511 <= pos <= end]
        last_diag = in_bin[-1].get("correction_updates", {}).get("GC-FPSA", {}).get(
            "fpsa_diagnostics") if in_bin else None
        trajectory.append({
            "event_window_end_inclusive": end,
            "queries_in_window": len(in_bin),
            "feedback_groups_cumulative": len([pos for pos in by_position if pos <= end]),
            "positive_target_capacity": (last_diag or {}).get("target_positive_capacity"),
            "negative_target_capacity": (last_diag or {}).get("target_negative_capacity"),
            "actual_positive_occupancy": (last_diag or {}).get("actual_positive_occupancy"),
            "actual_negative_occupancy": (last_diag or {}).get("actual_negative_occupancy"),
            "borrow_opportunities_cumulative": (last_diag or {}).get("quota_borrow_opportunities"),
            "successful_cross_sign_replacements_cumulative": (last_diag or {}).get("quota_cross_sign_replacements"),
            "rejected_borrowings_cumulative": (last_diag or {}).get("quota_borrow_rejected"),
        })
    score_distribution: dict[str, Any] = {}
    fixed_windows: dict[str, Any] = {}
    window_edges = [int(round(i * EVENTS / 4)) for i in range(5)]
    for arm, values in scores.items():
        score_distribution[arm] = {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "standard_deviation": float(values.std(ddof=0)),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "linear_slope_per_event": float(np.polyfit(np.arange(EVENTS), values, 1)[0]),
        }
        windows = []
        for window_index, (start, stop) in enumerate(zip(window_edges[:-1], window_edges[1:]), 1):
            window_labels = labels[start:stop]
            window_metrics = _threshold_metrics(values[start:stop], window_labels, threshold)
            windows.append({
                "window": window_index, "start_position": start,
                "end_position_exclusive": stop, "events": stop - start,
                "normal_events": int((window_labels == 0).sum()),
                "anomalous_events": int((window_labels == 1).sum()),
                "balanced_accuracy": window_metrics["balanced_accuracy"],
                "undefined_reason": (None if window_metrics["balanced_accuracy"] is not None
                                      else "window does not contain both classes"),
                "mean_score": float(values[start:stop].mean()),
                "median_score": float(np.median(values[start:stop])),
            })
        fixed_windows[arm] = windows
    final_receipt = next(row for row in reversed(history)
                         if row.get("record_type") == "category_complete")
    parity = final_receipt["parity"]
    if (parity.get("matched_events") != EVENTS or parity.get("g_checked") != EVENTS or
            parity.get("gc_checked") != EVENTS or
            parity.get("fpsa_gc_checked_through_1305") != FIRST_ANOMALOUS_QUERY + 1):
        raise R4AudiojackError("Stage A native-reference parity coverage is incomplete")
    fpsa_diag = final_receipt["fpsa_final_diagnostics"]
    delayed_trajectory_changed = any(
        item["actual_positive_occupancy"] not in (None, 0)
        for item in trajectory
        if item["event_window_end_inclusive"] >= FIRST_ANOMALOUS_QUERY + 1)
    mechanism_success = bool(
        fpsa_diag["quota_borrow_opportunities"] > 0 and
        fpsa_diag["quota_cross_sign_replacements"] > 0 and delayed_trajectory_changed)
    performance_success = bool(
        full["GC-FPSA"]["balanced_accuracy"] > full["GC"]["balanced_accuracy"])
    aggregate = {
        "schema": "r4.fpsa.audiojack.aggregate.v1",
        "status": "COMPLETE",
        "classification": "RETROSPECTIVE_SINGLE_EXPOSED_DEVELOPMENT_CLASS_MECHANISM_TEST",
        "dataset_protocol": {
            "category": CATEGORY, "events": EVENTS,
            "event_order": "frozen original TEST order; physical chronology not asserted",
            "labels_accessed_only_after_complete_stage_a_seal": True,
            "unqueried_labels_read_in_stage_a": False, "test_masks_opened": False,
            "independent_confirmation": False,
        },
        "arms": {"G": "native Greedy-only base score",
                 "GC": "same Greedy state plus original native C1-Full",
                 "GC-FPSA": "same Greedy state plus fixed R4-FPSA C1-Full Codebook"},
        "primary_endpoint": "full-stream prequential Balanced Accuracy at frozen C64 q90 threshold",
        "full_stream_metrics": full,
        "full_stream_score_distribution_and_drift": score_distribution,
        "fixed_quartile_metrics": fixed_windows,
        "full_stream_ba_contrasts": {
            "GC-FPSA_minus_GC": full["GC-FPSA"]["balanced_accuracy"] - full["GC"]["balanced_accuracy"],
            "GC-FPSA_minus_G": full["GC-FPSA"]["balanced_accuracy"] - full["G"]["balanced_accuracy"],
            "GC_minus_G": full["GC"]["balanced_accuracy"] - full["G"]["balanced_accuracy"],
        },
        "post_first_anomalous_feedback_window": {
            "event_positions_inclusive": [FIRST_ANOMALOUS_QUERY + 1, EVENTS - 1],
            "events": EVENTS - FIRST_ANOMALOUS_QUERY - 1,
            "metrics": post,
        },
        "error_transitions": transitions,
        "query_feedback_aggregate": feedback_outcomes,
        "query_rate": float(query_mask.mean()),
        "source_group_count": int(len(np.unique(groups))),
        "source_group_paired_bootstrap": _bootstrap_ba(scores, labels, groups, threshold),
        "native_reference_parity": {
            "R3_audiojack_events_matched": parity["matched_events"],
            "G_score_checks": parity["g_checked"], "G_max_abs_error": parity["g_max_abs_error"],
            "GC_score_checks": parity["gc_checked"], "GC_max_abs_error": parity["gc_max_abs_error"],
            "source_group_and_query_identity_parity": True,
            "greedy_write_event_parity": final_receipt["greedy_reference_parity"]["greedy_write_events"]["match"],
            "greedy_accepted_patch_parity": final_receipt["greedy_reference_parity"]["greedy_accepted_patch_count"]["match"],
            "fpsa_matches_gc_through_position_1305": parity["fpsa_gc_checked_through_1305"],
            "fpsa_gc_max_abs_error_through_1305": parity["fpsa_gc_max_abs_error_through_1305"],
        },
        "correction_occupancy": {
            "final": final_receipt["correction_final_sign_occupancy"],
            "maximum": final_receipt["correction_max_occupancy"],
            "fpsa_final_diagnostics": final_receipt["fpsa_final_diagnostics"],
            "binned_sign_allocation_trajectory": trajectory,
        },
        "mechanism_success": mechanism_success,
        "performance_success": performance_success,
        "interpretation_boundary": "One exposed extreme-stream development class; no independent generalization or confirmation claim.",
        "public_aggregate_redaction": {
            "raw_event_identities_present": False, "individual_test_labels_present": False,
            "source_group_identifiers_present": False, "raw_event_scores_present": False,
            "query_label_sequence_present": False,
        },
        "run_id": run_id, "source_parent_commit": EXPECTED_PARENT,
        "source_fingerprints": manifest["source_fingerprints"],
        "class_json_sha256": manifest["class_json_sha256"],
        "query_tape_sha256": manifest["query_tape_sha256"],
        "c64_score_tape_sha256": manifest["c64_score_tape_sha256"],
        "fixed_support_sha256": manifest["fixed_support_sha256"],
        "stage_a_seal_sha256": _sha256_file(seal_path),
        "score_rows_sha256": _sha256_file(scores_path),
        "feedback_ledger_sha256": seal["query_receipts_sha256"],
        "test_masks_opened": False,
    }
    budget = r3._budget(output)
    budget.write_json("aggregate_results.json", aggregate)
    stage_b = {
        "schema": STAGE_B_SCHEMA, "status": "SEALED_COMPLETE", "run_id": run_id,
        "stage_a_seal_sha256": _sha256_file(seal_path),
        "score_rows_sha256": _sha256_file(scores_path),
        "aggregate_results_sha256": _sha256_file(output / "aggregate_results.json"),
        "event_count": EVENTS, "outcome_labels_read_after_stage_a_seal": True,
        "test_masks_opened": False,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    budget.write_json("stage_b_seal.json", stage_b)
    budget.write_json("complete_seal.json", {
        "schema": "r4.fpsa.audiojack.complete.v1", "status": "COMPLETE",
        "run_id": run_id, "stage_a_seal_sha256": _sha256_file(seal_path),
        "stage_b_seal_sha256": _sha256_file(output / "stage_b_seal.json"),
        "aggregate_results_sha256": _sha256_file(output / "aggregate_results.json"),
        "event_count": EVENTS, "automatic_restart": False,
    })
    return aggregate


def _initialize_run(run_id: str, mode: str, contract: Mapping[str, Any],
                    token: str) -> Path:
    import experiments.calibration.r3_fullstream_2x2 as r3
    root = _output_path(run_id)
    source = _source_hashes()
    runner_sha = _sha256_file(Path(__file__).resolve())
    manifest = {
        "schema": RUN_SCHEMA, "run_id": run_id, "mode": mode,
        "authorization": AUTHORIZATION,
        "classification": "RETROSPECTIVE_SINGLE_EXPOSED_DEVELOPMENT_CLASS_MECHANISM_TEST",
        "status": "INCOMPLETE", "branch": EXPECTED_BRANCH,
        "source_parent_commit": EXPECTED_PARENT,
        "source_fingerprints": source, "runner_source_sha256": runner_sha,
        "authorization_token_sha256": _sha256_bytes(token.encode("ascii")),
        "category": CATEGORY, "event_count": EVENTS if mode == "FULL_STAGE_A" else 256,
        "query_count": (QUERIES if mode == "FULL_STAGE_A" else
                        sum(position < 256 for position in contract["positions"])),
        "class_json_sha256": contract["class_json_sha256"],
        "query_tape_sha256": contract["query_tape_sha256"],
        "query_positions_sha256": contract["query_positions_sha256"],
        "c64_score_tape_sha256": contract["c64_score_tape_sha256"],
        "fixed_support_sha256": contract["fixed_support_sha256_expected"],
        "stage05_hashes": contract["stage05_hashes"],
        "r3_reference": {
            "run_id": "r3_fullstream_2x2_20261102_a7",
            "stage_a_seal_sha256": _sha256_file(R3_REFERENCE / "stage_a_seal.json"),
            "audiojack_category_score_rows_sha256": _read_json(
                R3_REFERENCE / "stage_a_seal.json")["category_seals"][CATEGORY]["category_score_rows_sha256"],
        },
        "model_receipt": {
            key: r3._get_verified_model_receipt().get(key)
            for key in ("hub_source_tree_sha256", "checkpoint_sha256", "loaded_model_state_sha256", "profile_id")},
        "runtime_limit_seconds": RUNTIME_SECONDS if mode == "FULL_STAGE_A" else SMOKE_RUNTIME_SECONDS,
        "experiment_configuration": {
            "encoder": "frozen DINOv2 native encoder; one model load; one event encoding per image",
            "support_seed": 0, "support_count": 4, "additional_c64_count": 64,
            "c64_threshold": "q90", "c64_sigma_floor": 0.02,
            "greedy_normal_memory_capacity": 256, "correction_capacity": 256,
            "query_policy": "Q-A-TS 3%, seed 37, original order",
            "topk_patches": 32, "correction_config": dict(r3.CORRECTION_CONFIG),
            "fpsa_codebook_source_sha256": source[
                "experiments/calibration/r4_fpsa_codebook.py"],
            "arms": ["G", "GC", "GC-FPSA"],
            "threshold_controller": False, "parameter_search": False,
        },
        "output_storage": {"max_bytes": r3.OUTPUT_LIMIT,
                           "failure_reserve_bytes": r3.FAILURE_RESERVE,
                           "os_hard_storage_isolation": False,
                           "application_prewrite_budget": True,
                           "unique_non_overwriting_namespace": True},
        "test_masks_opened": False, "unqueried_test_labels_accessed": False,
        "offline_outcome_labels_accessed": False, "automatic_restart": False,
    }
    r3._budget(root, create=True).write_json("run_manifest.json", manifest)
    return root


def _launch_child(root: Path, stage: str, token: str, *, timeout: int,
                  authorized_cuda_visibility: str | None, smoke_events: int = 0):
    import experiments.calibration.r3_fullstream_2x2 as r3
    from experiments.calibration.r1_runtime_supervisor import run_supervised_child
    command = [sys.executable, "-m", "experiments.calibration.r4_fpsa_audiojack_runner",
               "--authorize-r4-audiojack", "--run-id", root.name]
    env = r3._child_environment(os.environ, stage, authorized_cuda_visibility)
    env["R4_FPSA_AUTH_TOKEN"] = token
    env["PYTHONHASHSEED"] = "0"
    if stage == "STAGE_A":
        command.append("--stage-a-child")
        if smoke_events:
            command.extend(["--smoke-events", str(smoke_events)])
    else:
        command.append("--stage-b-child")
    try:
        result = run_supervised_child(
            command, timeout_seconds=timeout, env=env, cwd=ROOT,
            output_dir=None)
    except BaseException as exc:
        r3._write_failure(root, stage, exc, {"category": CATEGORY,
                                             "process_tree_termination_confirmed": False,
                                             "automatic_restart": False})
        try:
            r3._record_supervisor_status(root, {
                "schema": "r4.fpsa.audiojack.supervisor.v1",
                "status": "SUPERVISOR_EXCEPTION", "stage": stage,
                "failure_type": type(exc).__name__, "automatic_restart": False,
            })
        except Exception:
            pass
        raise
    r3._append_supervisor_result(root, stage + ("_SMOKE" if smoke_events else ""),
                                 CATEGORY, result)
    return result


def run_experiment() -> dict[str, Any]:
    import experiments.calibration.r3_fullstream_2x2 as r3
    started = time.monotonic()
    repo = _repository_identity()
    if not Path(sys.executable).samefile(Path(r"D:\anaconda3\envs\torch\python.exe")):
        raise R4AudiojackError(f"R4 requires the frozen interpreter; found {sys.executable}")
    if not DATASET_ROOT.is_dir() or not JSON_ROOT.is_dir():
        raise R4AudiojackError("Authorized Real-IAD audiojack roots are unavailable")
    if REPORT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite deliverable directory {REPORT_DIR}")
    _output_path(RUN_ID)
    _output_path(SMOKE_ID)
    free_before = shutil.disk_usage(OUTPUT_BASE).free
    if free_before < MIN_FREE_BYTES:
        raise R4AudiojackError(f"Need 10 GiB free before run; found {free_before} bytes")
    contract = _load_audiojack_contract(verify_images=True)
    references = _load_r3_audiojack_reference(contract)
    if len(references) != EVENTS:
        raise R4AudiojackError("R3 audiojack score reference is incomplete")
    # CPU tests and compilation must complete before any GPU child is launched.
    tests_path = ROOT / "tests" / "calibration" / "test_r4_fpsa_audiojack_runner.py"
    compiled = subprocess.run(
        [sys.executable, "-m", "py_compile", str(Path(__file__).resolve()), str(tests_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    if compiled.returncode:
        raise R4AudiojackError(f"R4 compilation failed: {compiled.stderr[-2000:]}")
    test_command = [sys.executable, "-m", "pytest", "-q",
                    "tests/calibration/test_r4_fpsa_audiojack_runner.py",
                    "tests/calibration/test_r4_fpsa_codebook.py"]
    test_env = dict(os.environ, CUDA_VISIBLE_DEVICES="", R4_AUDIOJACK_SYNTHETIC_CPU_ONLY="1")
    test_result = subprocess.run(test_command, cwd=ROOT, env=test_env,
                                 capture_output=True, text=True, timeout=240)
    if test_result.returncode:
        raise R4AudiojackError(
            f"Focused R4 CPU tests failed ({test_result.returncode}): "
            f"{test_result.stdout[-3000:]} {test_result.stderr[-3000:]}")
    cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    token = os.urandom(32).hex()
    smoke = _initialize_run(SMOKE_ID, "SMOKE_ONLY", contract, token)
    smoke_result = _launch_child(smoke, "STAGE_A", token,
                                 timeout=SMOKE_RUNTIME_SECONDS,
                                 authorized_cuda_visibility=cuda_visibility,
                                 smoke_events=256)
    if smoke_result.status != "COMPLETED" or smoke_result.child_exit_code != 0:
        raise R4AudiojackError(f"Bounded audiojack GPU smoke failed: {asdict(smoke_result)}")
    smoke_history = [json.loads(line) for line in (smoke / "attempt_history.jsonl").read_text(
        encoding="utf-8").splitlines()]
    smoke_receipt = next((row for row in smoke_history
                          if row.get("record_type") == "category_complete"), None)
    if (not smoke_receipt or smoke_receipt.get("events_processed") != 256 or
            not smoke_history or smoke_receipt.get("query_events_processed", 0) <= 0):
        raise R4AudiojackError("Audiojack smoke did not exercise the real queried writer")
    r3._record_supervisor_status(smoke, {
        "schema": "r4.fpsa.audiojack.supervisor.v1", "status": "GPU_SMOKE_PASS_NOT_EVALUATED",
        "run_id": SMOKE_ID, "events": 256,
        "query_events": smoke_receipt["query_events_processed"],
        "windows_job_object": True, "automatic_restart": False,
        "offline_evaluation_performed": False,
        "wall_seconds": time.monotonic() - started,
    })

    # A clean, separate output namespace is created only after the smoke passes.
    full_free = shutil.disk_usage(OUTPUT_BASE).free
    if full_free < MIN_FREE_BYTES:
        raise R4AudiojackError(f"Need 10 GiB free before full run; found {full_free} bytes")
    root = _initialize_run(RUN_ID, "FULL_STAGE_A", contract, token)
    child = _launch_child(root, "STAGE_A", token, timeout=RUNTIME_SECONDS,
                          authorized_cuda_visibility=cuda_visibility)
    if child.status != "COMPLETED" or child.child_exit_code != 0:
        raise R4AudiojackError(f"Full audiojack Stage A child failed: {asdict(child)}")
    stage_a = _seal_stage_a(root, contract)
    stage_b = _launch_child(root, "STAGE_B", token, timeout=RUNTIME_SECONDS,
                            authorized_cuda_visibility=None)
    if stage_b.status != "COMPLETED" or stage_b.child_exit_code != 0:
        raise R4AudiojackError(f"CPU audiojack Stage B child failed: {asdict(stage_b)}")
    complete = _read_json(root / "complete_seal.json")
    aggregate = _read_json(root / "aggregate_results.json")
    budget = r3._budget(root)
    r3._record_supervisor_status(root, {
        "schema": "r4.fpsa.audiojack.supervisor.v1", "status": "COMPLETE",
        "run_id": RUN_ID, "stage_a_seal_sha256": _sha256_file(root / "stage_a_seal.json"),
        "stage_b_seal_sha256": _sha256_file(root / "stage_b_seal.json"),
        "full_runtime_limit_seconds": RUNTIME_SECONDS,
        "windows_job_object_process_tree_containment": True,
        "automatic_restart": False, "output_budget": budget.audit(),
        "minimum_free_space_before_full_launch": MIN_FREE_BYTES,
        "free_space_before_full_launch": full_free,
    })
    if complete.get("status") != "COMPLETE" or stage_a.get("status") != "SEALED_COMPLETE":
        raise R4AudiojackError("R4 completion seal failed after Stage B")
    _write_public_deliverables(root, aggregate, {
        "cpu_test_output": test_result.stdout.strip(),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "test_sha256": _sha256_file(tests_path),
        "smoke_run_id": SMOKE_ID,
        "smoke_events": 256,
        "source_parent_commit": repo["source_parent_commit"],
        "elapsed_seconds": time.monotonic() - started,
    })
    return aggregate


def _write_public_deliverables(output: Path, aggregate: Mapping[str, Any],
                               context: Mapping[str, Any]) -> None:
    import experiments.calibration.r3_fullstream_2x2 as r3
    report_path = REPORT_DIR / "REPORT.md"
    aggregate_path = REPORT_DIR / "aggregate_results.json"
    if REPORT_DIR.exists() or report_path.exists() or aggregate_path.exists():
        raise FileExistsError("R4 tracked report output exists; refusing overwrite")
    # Redaction guard: the public aggregate contains metrics and fingerprints,
    # never rows, image identities, labels, masks, or event-level score values.
    serialized = json.dumps(aggregate, sort_keys=True)
    forbidden = ("relative_id", "sample_key", "anomaly_class", "mask_path", "event_id_sha256")
    if any(term in serialized for term in forbidden):
        raise R4AudiojackError("R4 public aggregate contains a private identity/label field")
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    parity = aggregate["native_reference_parity"]
    metrics = aggregate["full_stream_metrics"]
    post = aggregate["post_first_anomalous_feedback_window"]["metrics"]
    occupancy = aggregate["correction_occupancy"]
    cpu = context["cpu_test_output"].splitlines()[-1] if context["cpu_test_output"] else "not recorded"
    lines = [
        "# R4-FPSA-AUDIOJACK — Single-Class Causal Mechanism Validation", "",
        "## Classification", "",
        "`RETROSPECTIVE_SINGLE_EXPOSED_DEVELOPMENT_CLASS_MECHANISM_TEST` — one frozen audiojack stream; this is not independent confirmation or generalization evidence.", "",
        "## Protocol and provenance", "",
        f"- Run: `{aggregate['run_id']}`; source parent: `{context['source_parent_commit']}`.",
        f"- Runner SHA-256: `{context['runner_sha256']}`; focused test SHA-256: `{context['test_sha256']}`.",
        f"- Audiojack class JSON SHA-256: `{aggregate['class_json_sha256']}`.",
        f"- Q-A-TS Seed 37 SHA-256: `{aggregate['query_tape_sha256']}`; C64 score tape SHA-256: `{aggregate['c64_score_tape_sha256']}`.",
        f"- Structured fixed Support SHA-256: `{aggregate['fixed_support_sha256']}`.",
        f"- Stage A seal SHA-256: `{aggregate['stage_a_seal_sha256']}`; score ledger SHA-256: `{aggregate['score_rows_sha256']}`; queried feedback ledger SHA-256: `{aggregate['feedback_ledger_sha256']}`.",
        f"- Focused CPU tests: `{cpu}`. Smoke: {context['smoke_events']} events under Windows Job Object, with queried feedback/update path exercised; no Stage B.",
        "- DINO source, checkpoint and loaded model-state fingerprints are bound in the local run manifest and Stage A receipt.",
        "- Stage A accessed only frozen audiojack metadata, Support/C64, the fixed query tape, and legal feedback after each score seal. Stage B projected audiojack image-level outcomes after the complete Stage A seal; masks were not opened.",
        "- The historical R3 manifest retains its prior top-level `INCOMPLETE` status, but its audiojack category score digest, Stage A seal, Stage B seal, and complete seal were validated for this comparison.",
        "", "## R3 numerical parity", "",
        f"- G: {parity['G_score_checks']} events; max absolute difference `{parity['G_max_abs_error']:.9g}`.",
        f"- GC: {parity['GC_score_checks']} events; max absolute difference `{parity['GC_max_abs_error']:.9g}`.",
        f"- Source-group/query identities and Greedy write/accepted-patch events: source identity parity {parity['source_group_and_query_identity_parity']}; write-event parity {parity['greedy_write_event_parity']}; accepted-patch parity {parity['greedy_accepted_patch_parity']}.",
        f"- GC-FPSA vs GC through position 1305: {parity['fpsa_matches_gc_through_position_1305']} events; max absolute difference `{parity['fpsa_gc_max_abs_error_through_1305']:.9g}`.",
        "", "## Prequential results", "",
        "Full stream (3,815 events), at the frozen C64 q90 threshold:", "",
        "| Arm | BA | FPR | FNR | TP | TN | FP | FN | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("G", "GC", "GC-FPSA"):
        m = metrics[arm]
        c = m["confusion"]
        lines.append(f"| {arm} | {m['balanced_accuracy']:.6f} | {m['fpr']:.6f} | {m['fnr']:.6f} | {c['tp']} | {c['tn']} | {c['fp']} | {c['fn']} | {m['accuracy'] if 'accuracy' in m else (c['tp']+c['tn'])/EVENTS:.6f} |")
    lines += ["", "Post-anomalous-feedback window (positions 1306–3814):", "",
              "| Arm | Events | BA | FPR | FNR | TP | TN | FP | FN |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("G", "GC", "GC-FPSA"):
        m, c = post[arm], post[arm]["confusion"]
        lines.append(f"| {arm} | {aggregate['post_first_anomalous_feedback_window']['events']} | {m['balanced_accuracy']:.6f} | {m['fpr']:.6f} | {m['fnr']:.6f} | {c['tp']} | {c['tn']} | {c['fp']} | {c['fn']} |")
    final = occupancy["final"]
    lines += ["", "## Sign allocation and borrowing", "",
              f"- Final GC occupancy: positive {final['GC']['positive']}, negative {final['GC']['negative']} (K=256).",
              f"- Final GC-FPSA occupancy: positive {final['GC-FPSA']['positive']}, negative {final['GC-FPSA']['negative']} (K=256).",
              f"- Final FPSA target: positive {occupancy['fpsa_final_diagnostics']['target_positive_capacity']}, negative {occupancy['fpsa_final_diagnostics']['target_negative_capacity']}; targets are not guaranteed occupancy.",
              f"- Borrowing opportunities: {occupancy['fpsa_final_diagnostics']['quota_borrow_opportunities']}; successful cross-sign replacements: {occupancy['fpsa_final_diagnostics']['quota_cross_sign_replacements']}; retention-rejected borrowings: {occupancy['fpsa_final_diagnostics']['quota_borrow_rejected']}.",
              "- Trajectory is reported in fixed 512-event bins to avoid publishing individual queried labels.",
              "", "## Interpretation and limitations", "",
              f"BA contrasts (GC-FPSA−GC, GC-FPSA−G, GC−G): `{aggregate['full_stream_ba_contrasts']}`.",
              "Mechanism success requires successful cross-sign borrowing and a changed delayed-feedback occupancy trajectory. Performance success requires positive full-stream BA difference over GC. Any FPR/FNR trade-off is reported as measured; this is one exposed audiojack development stream and does not establish transfer, independent significance, or safety.",
              "The 4 GiB application OutputBudget is pre-write accounting; no OS-level hard storage isolation is provided. The Windows Job Object enforces the 120-minute process-tree deadline. The run is not resumable and was launched in a fresh namespace.",
              "", "## Source-controlled artifact scope", "",
              "Only this report, its redacted aggregate, the isolated R4 runner, and its focused tests are committed. Private score/feedback ledgers remain in the local output namespace.", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-r4-audiojack", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--run-orchestrator", action="store_true")
    parser.add_argument("--stage-a-child", action="store_true")
    parser.add_argument("--stage-b-child", action="store_true")
    parser.add_argument("--smoke-events", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.authorize_r4_audiojack:
        raise R4AudiojackError("Explicit --authorize-r4-audiojack is required")
    modes = int(args.run_orchestrator) + int(args.stage_a_child) + int(args.stage_b_child)
    if modes != 1:
        raise R4AudiojackError("Choose exactly one R4 orchestrator, Stage A child, or Stage B child mode")
    if args.run_orchestrator:
        run_experiment()
        return 0
    token = os.environ.get("R4_FPSA_AUTH_TOKEN", "")
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise R4AudiojackError("Supervised child authorization token is missing")
    if args.stage_a_child:
        _stage_a_child(str(args.run_id), token, int(args.smoke_events))
    else:
        _stage_b_child(str(args.run_id), token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
