"""A4_EB68_v1: isolated equal-budget normal-reference/calibration experiment.

The module imports only verified low-level native functions and pure A3b
K-Center helpers. It never calls an A1/A2/A3 orchestration entry point.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_ID = "A4_EB68_v1"
AUTHORIZATION_ID = "EXPERIMENT_A4_EB68_AUTHORIZATION_20261107"
EXPECTED_PARENT = "84e434b7b89e892a55ad610408cd11161f7de3a5"
BRANCH = "calibration-stage0-contract-20261029"
DATASET_ROOT = Path(r"G:\dataset\realiad")
JSON_ROOT = Path(r"G:\dataset\realiad\jsons\realiad_jsons")
A1_RUN = ROOT / "outputs/calibration/a_calibration_scale_20261106_a1_full01"
A23_RUN = ROOT / "outputs/calibration/a_calibration_scale_a23_20261106_full08"
R9_R10_PROTECTED = (
    ROOT / "outputs/calibration/r9_correction_mechanism_20261103_a9",
    ROOT / "outputs/calibration/r10_signed_knn_baseline_20261103_a1",
)
RUN_ROOT = ROOT / "outputs/calibration/a4_equal_budget_20261107"
REPORT_ROOT = ROOT / "docs/calibration/audit/a4_equal_budget_20261107"
CATEGORIES = ("audiojack", "regulator", "switch", "rolled_strip_base")
EXPECTED_EVENTS = {"audiojack": 3815, "regulator": 3855, "switch": 3820,
                   "rolled_strip_base": 3805}
ARMS = ("A", "B", "C", "D")
SCORE_KEYS = {arm: {"static": f"{arm}_STATIC_M0",
                    "online": f"{arm}_PREQUENTIAL"} for arm in ARMS}
REPORT_OUTPUT_NAMES = (
    "EXPERIMENT_A4_INTEGRATED_REPORT.md", "EXPERIMENT_A4_RESULTS.csv",
    "EXPERIMENT_A4_AGGREGATE.json", "EXPERIMENT_A4_RESOURCE_ACCOUNTING.json",
    "EXPERIMENT_A4_PROVENANCE_AUDIT.md", "EXPERIMENT_A4_REVIEWER_RESPONSE_CANDIDATE.md",
    "EXPERIMENT_A4_FINAL_DECISION.md", "EXPERIMENT_A4_REPORT_MANIFEST.json",
    "figures/figure_1_static_prequential_auroc.png",
    "figures/figure_2_prequential_error_tradeoffs.png",
    "figures/figure_3_registered_contrast_decomposition.png",
    "figures/figure_4_equal_budget_memory_accounting.png",
)
FIXED_COUNT, FEATURE_DIM, FIXED_BYTES = 8192, 4096, 67_108_864
GREEDY_K, KC_CAP, KC_BLOCK_ROWS = 1024, 65_536, 4096
Q90, SIGMA_FLOOR, B3_DELTA = 0.90, 0.02, 0.01
MIN_FREE_BYTES = 30 * 1024**3
MAX_RETAINED_BYTES = 4 * 1024**3
CATEGORY_SECONDS, CAMPAIGN_SECONDS = 7200, 8 * 3600
PREEXISTING_UNTRACKED_COUNT = 6870
PREEXISTING_UNTRACKED_SHA256 = "5a04704446691302962e7cf15b97eb32d770df7ee7b5a5d7ca79f40731b6680f"

SOURCE_PATHS = (
    "AGENTS.md",
    "docs/calibration/EXPERIMENT_A_AUTHORIZATION_20261106.md",
    "docs/calibration/EXPERIMENT_A4_EB68_AUTHORIZATION.md",
    "docs/calibration/audit/a4_equal_budget_20261107/EXPERIMENT_A4_PROTOCOL.md",
    "experiments/calibration/a4_equal_budget_allocation.py",
    "experiments/calibration/a4_report_renderer.py",
    "experiments/calibration/a4_independent_verifier.py",
    "tests/calibration/test_a4_equal_budget_allocation.py",
    "experiments/calibration/a_calibration_scale.py",
    "experiments/calibration/a_calibration_continuation.py",
    "experiments/calibration/stage05.py",
    "experiments/calibration/r1_matched_correction_pilot.py",
    "experiments/calibration/r1_runtime_supervisor.py",
    "experiments/calibration/r3_fullstream_2x2.py",
    "experiments/calibration/r7_shuffled_correction_4class.py",
    "experiments/calibration/r9_correction_mechanism_falsification.py",
    "experiments/calibration/native.py",
    "experiments/calibration/profiles.py",
    "experiments/calibration/query_tape.py",
    "experiments/calibration/protocol.py",
    "experiments/risk_safe/native_bridge.py",
    "experiments/risk_safe/preflight.py",
    "experiments/v02_memory.py",
    "experiments/p0_protocol_audit.py",
    "test_phase3_AIF_simplified_v22.py",
    "vmb_visionad_new.py",
)
ALLOWED_A4_SOURCE_PATHS = frozenset({
    "AGENTS.md",
    "docs/calibration/EXPERIMENT_A4_EB68_AUTHORIZATION.md",
    "docs/calibration/audit/a4_equal_budget_20261107/EXPERIMENT_A4_PROTOCOL.md",
    "experiments/calibration/a4_equal_budget_allocation.py",
    "experiments/calibration/a4_report_renderer.py",
    "experiments/calibration/a4_independent_verifier.py",
    "tests/calibration/test_a4_equal_budget_allocation.py",
})
PROTECTED_OUTPUT_PATHS = (
    "outputs/calibration/a_calibration_scale_20261106_a1_full01",
    "outputs/calibration/a_calibration_scale_a23_20261106_full08",
    "outputs/calibration/r9_correction_mechanism_20261103_a9",
    "outputs/calibration/r10_signed_knn_baseline_20261103_a1",
)


class A4Error(RuntimeError):
    pass


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_source_bytes(value: bytes) -> str:
    """Hash source text with Git's Windows CRLF checkout normalization removed."""
    return _sha_bytes(value.replace(b"\r\n", b"\n"))


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_once(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        raise


def _fsync_file(path: Path) -> None:
    """Flush a completed temporary file through a writable Windows handle."""
    with Path(path).open("r+b") as f:
        f.flush()
        os.fsync(f.fileno())


def _write_json_once(path: Path, value: Any) -> None:
    payload = value if isinstance(value, bytes) else _json_bytes(value) + b"\n"
    _write_once(path, payload)


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def _git_paths(command: Sequence[str]) -> list[str]:
    result = subprocess.run(["git", *command], cwd=ROOT, check=True,
                            capture_output=True, text=True)
    return sorted(line.replace("\\", "/") for line in result.stdout.splitlines() if line)


def _untracked_baseline() -> dict[str, Any]:
    paths = _git_paths(("ls-files", "--others", "--exclude-standard"))
    task_paths = {"docs/calibration/EXPERIMENT_A4_EB68_AUTHORIZATION.md",
                  "docs/calibration/audit/a4_equal_budget_20261107/EXPERIMENT_A4_PROTOCOL.md",
                  "experiments/calibration/a4_equal_budget_allocation.py",
                  "experiments/calibration/a4_report_renderer.py",
                  "experiments/calibration/a4_independent_verifier.py",
                  "tests/calibration/test_a4_equal_budget_allocation.py"}
    generated = [p for p in paths if p.startswith(
        "outputs/calibration/a4_equal_budget_20261107/")]
    report_outputs = [p for p in paths if p.startswith(
        "docs/calibration/audit/a4_equal_budget_20261107/")]
    remaining = [p for p in paths if p not in task_paths and
                 not p.startswith("outputs/calibration/a4_equal_budget_20261107/") and
                 not p.startswith("docs/calibration/audit/a4_equal_budget_20261107/")]
    digest = _sha_bytes(("\n".join(remaining) + "\n").encode("utf-8")) if remaining else _sha_bytes(b"")
    return {"baseline_count": PREEXISTING_UNTRACKED_COUNT,
            "baseline_paths_sha256": PREEXISTING_UNTRACKED_SHA256,
            "observed_preexisting_count": len(remaining),
            "observed_preexisting_paths_sha256": digest,
            "task_output_untracked_count": len(generated),
            "task_report_untracked_count": len(report_outputs),
            "matches_pre_task_inventory":
                len(remaining) == PREEXISTING_UNTRACKED_COUNT and
                digest == PREEXISTING_UNTRACKED_SHA256}


def protected_file_audit(*, allow_report_files: bool = True) -> dict[str, Any]:
    """A4-specific exact-path audit; unrelated baseline untracked paths are ignored."""
    staged = _git_paths(("diff", "--cached", "--name-only"))
    unstaged = _git_paths(("diff", "--name-only"))
    report_prefix = "docs/calibration/audit/a4_equal_budget_20261107/"
    permitted = set(ALLOWED_A4_SOURCE_PATHS)
    if allow_report_files:
        permitted.update(p for p in staged + unstaged if p.startswith(report_prefix))
    bad_staged = [p for p in staged if p not in permitted]
    bad_unstaged = [p for p in unstaged if p not in permitted]
    inv = _untracked_baseline()
    if not inv["matches_pre_task_inventory"]:
        task_untracked = set(ALLOWED_A4_SOURCE_PATHS) | {
            "docs/calibration/EXPERIMENT_A4_EB68_AUTHORIZATION.md",
            "docs/calibration/audit/a4_equal_budget_20261107/EXPERIMENT_A4_PROTOCOL.md"}
        paths = _git_paths(("ls-files", "--others", "--exclude-standard"))
        residual = [p for p in paths if p not in task_untracked and not p.startswith(
            "outputs/calibration/a4_equal_budget_20261107/") and not p.startswith(
            "docs/calibration/audit/a4_equal_budget_20261107/")]
        digest = _sha_bytes(("\n".join(residual) + "\n").encode("utf-8")) if residual else _sha_bytes(b"")
        inv.update({"observed_preexisting_count": len(residual),
                    "observed_preexisting_paths_sha256": digest,
                    "matches_pre_task_inventory": len(residual) == PREEXISTING_UNTRACKED_COUNT and
                    digest == PREEXISTING_UNTRACKED_SHA256})
    if bad_staged or bad_unstaged:
        raise A4Error(f"Protected-file audit failed; staged={bad_staged}, unstaged={bad_unstaged}")
    return {"status": "PASS", "staged_tracked_changes": staged,
            "unstaged_tracked_changes": unstaged,
            "preexisting_untracked_inventory": inv,
            "method": "exact A4 allowlist plus pre-task untracked pathname digest"}


def _tree_receipt(relative_root: str) -> dict[str, Any]:
    root = ROOT / relative_root
    if not root.is_dir():
        raise A4Error(f"Required protected historical output directory is missing: {relative_root}")
    entries = []
    total = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        size = path.stat().st_size
        total += size
        entries.append((path.relative_to(root).as_posix(), size, _sha_file(path)))
    return {"relative_root": relative_root, "file_count": len(entries),
            "total_bytes": total, "tree_sha256": _sha_bytes(_json_bytes(entries))}


def _source_hashes() -> dict[str, str]:
    out = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise A4Error(f"A4 source-freeze file is missing: {relative}")
        out[relative] = _sha_source_bytes(path.read_bytes())
    return out


def _minimum_free_space() -> dict[str, int]:
    disks: dict[str, int] = {}
    for path in (ROOT, DATASET_ROOT, JSON_ROOT):
        anchor = path.anchor or str(path)
        usage = shutil.disk_usage(anchor)
        disks[anchor] = int(usage.free)
        if usage.free < MIN_FREE_BYTES:
            raise A4Error(f"Less than 30 GiB free on required volume {anchor}: {usage.free}")
    return disks


def _gpu_receipt() -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise A4Error("A4 requires exactly one available CUDA device")
    device = torch.cuda.get_device_properties(0)
    name = str(device.name)
    memory = int(device.total_memory)
    if "3090 Ti" not in name or not (23 * 1024**3 <= memory <= 25 * 1024**3):
        raise A4Error(f"A4 requires the registered 24 GB RTX 3090 Ti; found {name} with {memory} bytes")
    return {"device_count": 1, "name": name, "total_memory_bytes": memory,
            "capability": list(torch.cuda.get_device_capability(0))}


def _hash_official_group(key: str) -> str:
    return hashlib.sha256(("official-sample-group\0" + str(key)).encode("utf-8")).hexdigest()


def _identity_hash(relative_id: str) -> str:
    return hashlib.sha256(("r7-image-identity\0" + str(relative_id).replace("\\", "/")).encode("utf-8")).hexdigest()


def _category_roles(category: str, train_rows: Sequence[Any], test_rows: Sequence[Any],
                    category_json_sha: str) -> dict[str, Any]:
    from experiments.calibration import a_calibration_scale as a1
    from experiments.calibration import stage05

    if category not in CATEGORIES:
        raise A4Error(f"Unauthorized category: {category}")
    preflight = _read(A1_RUN / "preflight_private.json")
    role_path = A1_RUN / "a1" / category / "role_manifest_private.json"
    seal_path = A1_RUN / "a1" / category / "stage_a_seal.json"
    role = _read(role_path)
    a1seal = _read(seal_path)
    category_preflight = preflight["classes"][category]
    if category_json_sha != a1seal["input_digests"]["category_json"]:
        raise A4Error(f"{category}: current category JSON differs from sealed A1")
    support = category_preflight["support_manifest"]["0"]
    c64 = category_preflight["historical_C64_manifest"]
    if int(support.get("seed", -1)) != 0 or len(support["groups"]) != 4 or len(support["images"]) != 4:
        raise A4Error(f"{category}: A1 Support-0 identity receipt is invalid")
    if int(c64.get("count", -1)) != 64 or len(c64["groups"]) != 64 or len(c64["images"]) != 64:
        raise A4Error(f"{category}: historical additional C64 identity receipt is invalid")
    if role.get("category") != category or role.get("support_groups", {}).get("0") != support["groups"]:
        raise A4Error(f"{category}: private A1 role manifest does not bind Support-0 groups")
    c64_by_group = dict(zip(c64["groups"], c64["images"]))
    if len(c64_by_group) != 64 or set(c64_by_group) != set(role["historical_C64_groups"]):
        raise A4Error(f"{category}: A1 C64 historical group/image mapping is not exact")
    draw = role["draws"]["0"]
    c32_groups = list(draw["C32"])
    if len(c32_groups) != 32 or len(set(c32_groups)) != 32 or set(c32_groups) - set(c64_by_group):
        raise A4Error(f"{category}: A1 draw-0 C32 membership is not nested in C64")
    c64_groups_ordered = list(draw["C64"])
    if len(c64_groups_ordered) != 64 or set(c64_groups_ordered) != set(c64_by_group):
        raise A4Error(f"{category}: A1 draw-0 C64 order/membership is invalid")
    c32_set = set(c32_groups)
    r32_groups = [g for g in c64_groups_ordered if g not in c32_set]
    if len(r32_groups) != 32 or c32_set & set(r32_groups) or c32_set | set(r32_groups) != set(c64_by_group):
        raise A4Error(f"{category}: C32/R32 complement checks failed")
    train_by_rel: dict[str, Any] = {}
    train_by_group: dict[str, list[Any]] = {}
    for row in train_rows:
        rid = str(row.relative_id).replace("\\", "/")
        if rid in train_by_rel:
            raise A4Error(f"{category}: duplicate TRAIN relative image identity {rid}")
        train_by_rel[rid] = row
        train_by_group.setdefault(str(row.sample_key), []).append(row)
    support_items = []
    c64_items = []
    all_selected = []
    for group, rel in zip(support["groups"], support["images"]):
        rows = train_by_group.get(str(group), [])
        if rel not in train_by_rel or not rows or not all(bool(r.is_good) for r in rows):
            raise A4Error(f"{category}: Support identity is not an exact TRAIN-normal image/group")
        if train_by_rel[rel].sample_key != group:
            raise A4Error(f"{category}: Support image/group pairing changed")
        item = {"group": str(group), "group_sha256": _hash_official_group(group),
                "relative_id": str(rel).replace("\\", "/"),
                "image_path": str(train_by_rel[rel].image_path)}
        support_items.append(item)
        all_selected.append(item)
    for group in c64_groups_ordered:
        rel = c64_by_group[group]
        rows = train_by_group.get(str(group), [])
        if rel not in train_by_rel or not rows or not all(bool(r.is_good) for r in rows):
            raise A4Error(f"{category}: C64 identity is not an exact TRAIN-normal image/group")
        if train_by_rel[rel].sample_key != group:
            raise A4Error(f"{category}: C64 image/group pairing changed")
        item = {"group": str(group), "group_sha256": _hash_official_group(group),
                "relative_id": str(rel).replace("\\", "/"),
                "image_path": str(train_by_rel[rel].image_path)}
        c64_items.append(item)
        all_selected.append(item)
    group_ids = [x["group"] for x in all_selected]
    opaque_ids = [x["group_sha256"] for x in all_selected]
    if len(all_selected) != 68 or len(set(group_ids)) != 68 or len(set(opaque_ids)) != 68:
        raise A4Error(f"{category}: S4+C64 is not 68 distinct official Source Groups")
    test_groups = {str(row.sample_key) for row in test_rows}
    if set(group_ids) & test_groups:
        raise A4Error(f"{category}: TRAIN roles overlap a TEST Source Group")
    test_group_hashes = {str(row.opaque_group) for row in test_rows}
    if set(opaque_ids) & test_group_hashes:
        raise A4Error(f"{category}: opaque TRAIN/TEST Source Group overlap")
    for item in all_selected:
        if not Path(item["image_path"]).is_file():
            raise A4Error(f"{category}: registered TRAIN image is missing: {item['relative_id']}")
    if len(test_rows) != EXPECTED_EVENTS[category] or len(test_group_hashes) * 5 != len(test_rows):
        raise A4Error(f"{category}: P0 TEST event/group count differs from registration")
    identity_payload = {
        "support": [{"group": x["group"], "relative_id": x["relative_id"]} for x in support_items],
        "C64": [{"group": x["group"], "relative_id": x["relative_id"]} for x in c64_items],
        "C32_groups": c32_groups,
        "R32_groups": r32_groups,
    }
    role_digest = _sha_bytes(_json_bytes(identity_payload))
    expected_support_sha = a1seal["input_digests"]["fixed_support_seed0"]
    role_digest_a1 = _sha_file(role_path)
    result = {
        "category": category, "support_items": support_items,
        "c64_items": c64_items,
        "c32_groups": c32_groups, "r32_groups": r32_groups,
        "c32_items": [next(x for x in c64_items if x["group"] == g) for g in c32_groups],
        "r32_items": [next(x for x in c64_items if x["group"] == g) for g in r32_groups],
        "train_by_rel": train_by_rel, "train_by_group": train_by_group,
        "test_group_hashes": test_group_hashes,
        "identity_sha256": role_digest,
        "support_identity_sha256": _sha_bytes(_json_bytes(identity_payload["support"])),
        "c64_identity_sha256": _sha_bytes(_json_bytes(identity_payload["C64"])),
        "c32_identity_sha256": _sha_bytes(_json_bytes(c32_groups)),
        "r32_identity_sha256": _sha_bytes(_json_bytes(r32_groups)),
        "a1_role_manifest_sha256": role_digest_a1,
        "a1_stage_a_seal_sha256": _sha_file(seal_path),
        "a1_support0_fixed_bank_sha256": expected_support_sha,
        "category_json_sha256": category_json_sha,
        "test_event_count": len(test_rows),
        "test_group_count": len(test_group_hashes),
        "no_test_overlap": True,
        "strict_four_shot_total": False,
    }
    return result


def _query_identity_receipt(category: str, test_rows: Sequence[Any]) -> dict[str, Any]:
    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration import r9_correction_mechanism_falsification as r9

    events = r7.deterministic_group_shuffle(test_rows, seed=0)
    if len(events) != EXPECTED_EVENTS[category]:
        raise A4Error(f"{category}: P0 event shuffle count changed")
    tape = r7._make_query_tape(category, events)
    tape_receipt = r7._tape_receipt(tape)
    positions = list(tape_receipt["positions"])
    a2seal_path = A23_RUN / "stage_a" / category / "P0" / "stage_a_seal.json"
    a2seal = _read(a2seal_path)
    identities = [events[p].image_identity_sha256 for p in positions]
    groups = [events[p].source_group_sha256 for p in positions]
    identity_sha = r7._sha256_json(sorted(identities))
    group_sha = hashlib.sha256("".join(g + "\n" for g in groups).encode("ascii")).hexdigest()
    if (positions != a2seal.get("query_positions") or
            identity_sha != a2seal.get("query_identity_set_sha256") or
            group_sha != a2seal.get("query_group_set_sha256")):
        raise A4Error(f"{category}: reconstructed Q-A-TS-v1 P0 query identities differ from A2 seal")
    return {"events": events, "positions": positions,
            "query_count": len(positions), "tape_sha256": tape_receipt["tape_sha256"],
            "positions_sha256": tape_receipt["positions_sha256"],
            "identity_set_sha256": identity_sha,
            "group_set_sha256": group_sha,
            "a2_stage_a_seal_sha256": _sha_file(a2seal_path),
            "group_count": len(set(groups))}


def preflight(*, check_source_freeze: bool = False) -> dict[str, Any]:
    from experiments.calibration import stage05
    from experiments.calibration import r3_fullstream_2x2 as r3

    if _git("branch", "--show-current") != BRANCH:
        raise A4Error(f"Wrong branch; expected {BRANCH}")
    head = _git("rev-parse", "HEAD")
    if not _git("merge-base", "--is-ancestor", EXPECTED_PARENT, "HEAD") == "":
        raise A4Error("Expected A4 reference parent is not an ancestor of current HEAD")
    if not (DATASET_ROOT / "images").is_dir() or not JSON_ROOT.is_dir():
        raise A4Error("Required Real-IAD dataset or JSON directory is unavailable")
    protected = protected_file_audit()
    free = _minimum_free_space()
    gpu = _gpu_receipt()
    a1pre = _read(A1_RUN / "preflight_private.json")
    a1integrity = _read(A1_RUN / "integrity_audit.json")
    a2manifest = _read(A23_RUN / "run_manifest.json")
    if a1integrity.get("status") != "PASS_A1_ONLY":
        raise A4Error("A1 integrity receipt is not PASS_A1_ONLY")
    if a2manifest.get("protocol_id") != "a_calibration_scale_a23_20261106.v1":
        raise A4Error("A2/A3 state-provenance run manifest is incompatible")
    source_hashes = _source_hashes()
    category_receipts = {}
    for category in CATEGORIES:
        train, test, category_sha = stage05.read_class_stream(
            DATASET_ROOT, JSON_ROOT, category, test_labels=False)
        roles = _category_roles(category, train, test, category_sha)
        query = _query_identity_receipt(category, test)
        category_receipts[category] = {
            "roles": {key: roles[key] for key in (
                "identity_sha256", "support_identity_sha256", "c64_identity_sha256",
                "c32_identity_sha256", "r32_identity_sha256", "a1_role_manifest_sha256",
                "a1_stage_a_seal_sha256", "a1_support0_fixed_bank_sha256",
                "category_json_sha256", "test_event_count", "test_group_count",
                "no_test_overlap", "strict_four_shot_total")},
            "query": {key: query[key] for key in (
                "query_count", "tape_sha256", "positions_sha256", "identity_set_sha256",
                "group_set_sha256", "a2_stage_a_seal_sha256", "group_count")},
        }
    model_receipt = r3._get_verified_model_receipt()
    model = {key: model_receipt.get(key) for key in (
        "checkpoint_sha256", "loaded_model_state_sha256", "hub_source_tree_sha256",
        "runtime_profile", "status", "task", "profile_id")}
    output_collisions = [str(REPORT_ROOT / name) for name in REPORT_OUTPUT_NAMES
                         if (REPORT_ROOT / name).exists()]
    if output_collisions:
        raise A4Error(f"A4 generated report destination collision: {output_collisions}")
    source_commit = head
    tracked_a4 = set(_git_paths(("ls-files", "--cached")))
    implementation_tracked = all(p in tracked_a4 for p in SOURCE_PATHS)
    if check_source_freeze:
        if not implementation_tracked:
            raise A4Error("A4 source-freeze commit is missing one or more authorized implementation files")
        if _git("status", "--porcelain", "--untracked-files=no"):
            raise A4Error("Tracked worktree changes remain after the source-freeze commit")
        for relative, digest in source_hashes.items():
            blob = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT,
                                  check=True, capture_output=True).stdout
            if _sha_source_bytes(blob) != digest:
                raise A4Error(f"Execution-time source differs from source-freeze commit: {relative}")
    protected_outputs = [_tree_receipt(p) for p in PROTECTED_OUTPUT_PATHS]
    result = {
        "schema": "A4_EB68_v1.preflight.v1", "status": "PASS",
        "protocol_id": PROTOCOL_ID, "authorization_id": AUTHORIZATION_ID,
        "branch": BRANCH, "head": head, "expected_parent": EXPECTED_PARENT,
        "source_freeze_ready": implementation_tracked,
        "source_hashes": source_hashes, "category_receipts": category_receipts,
        "source_hash_normalization": "CRLF normalized to LF before SHA-256",
        "model_receipt": model, "free_bytes_by_volume": free,
        "gpu_receipt": gpu,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "test_feature_cache": "NOT_CREATED; stream-encode each test image once and share across arms",
        "protected_file_audit": protected,
        "protected_historical_output_trees": protected_outputs,
        "a1_preflight_sha256": _sha_file(A1_RUN / "preflight_private.json"),
        "a1_integrity_sha256": _sha_file(A1_RUN / "integrity_audit.json"),
        "a2a3_manifest_sha256": _sha_file(A23_RUN / "run_manifest.json"),
        "all_test_labels_read": False, "all_masks_opened": False,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if check_source_freeze:
        result["source_freeze_commit"] = source_commit
    return result


def _higher_q90(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise A4Error("B3 q90 requires at least two finite scores")
    index = int(math.ceil(Q90 * (len(x) - 1)))
    return float(np.partition(x, index)[index])


def _validate_fixed_profile(profile: Mapping[str, Any]) -> None:
    layers = profile.get("layers", [])
    if (int(profile.get("total_prototypes", -1)) != FIXED_COUNT or
            int(profile.get("total_feature_bytes", -1)) != FIXED_BYTES or
            len(layers) != 1 or int(layers[0].get("n_prototypes", -1)) != FIXED_COUNT or
            int(layers[0].get("feature_dim", -1)) != FEATURE_DIM or
            layers[0].get("dtype") != "float16" or
            int(layers[0].get("bytes", -1)) != FIXED_BYTES):
        raise A4Error("Fixed reference must be exactly 8,192 x 4,096 fp16 (67,108,864 bytes)")


def _window_metric_row(labels: Sequence[int], predictions: Sequence[int],
                       start: int, window: int = 512) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)[int(start):int(start) + int(window)]
    p = np.asarray(predictions, dtype=np.int8)[int(start):int(start) + int(window)]
    if len(y) != int(window) or p.shape != y.shape:
        raise A4Error("Window metric requires a complete aligned window")
    if len(np.unique(y)) < 2:
        return {"start": int(start), "end_exclusive": int(start) + int(window),
                "status": "INVALID_SINGLE_CLASS_WINDOW", "fpr": None,
                "fnr": None, "ber": None}
    fpr = float(np.mean(p[y == 0] == 1))
    fnr = float(np.mean(p[y == 1] == 0))
    return {"start": int(start), "end_exclusive": int(start) + int(window),
            "status": "VALID", "fpr": fpr, "fnr": fnr,
            "ber": (fpr + fnr) / 2}


class B3State:
    def __init__(self, scores: Sequence[float], version: int = 0):
        x = np.asarray(scores, dtype=np.float64)
        if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
            raise A4Error("B3 initialization requires finite calibration scores")
        self.theta = _higher_q90(x)
        self.sigma = max(float(x.std(ddof=1)), SIGMA_FLOOR)
        self.version = int(version)
        self.history = [{"version": self.version, "theta": self.theta,
                         "sigma": self.sigma, "kind": "initial"}]

    def refresh(self, scores: Sequence[float], version: int) -> bool:
        version = int(version)
        if version == self.version:
            return False
        if version < self.version:
            raise A4Error("B3 memory state version moved backwards")
        x = np.asarray(scores, dtype=np.float64)
        if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
            raise A4Error("B3 refresh requires finite protected scores")
        raw = _higher_q90(x)
        prior = self.theta
        self.theta = float(np.clip(raw, prior - B3_DELTA, prior + B3_DELTA))
        self.sigma = max(float(x.std(ddof=1)), SIGMA_FLOOR)
        self.version = version
        self.history.append({"version": version, "theta": self.theta,
                             "sigma": self.sigma, "raw_q90": raw,
                             "clipped_from": prior, "kind": "state_change"})
        return True


def _encode_identity_items(impl: Any, backbone: Any, cfg: Any,
                           items: Sequence[Mapping[str, Any]], category: str) -> list[dict[str, Any]]:
    import torch
    from experiments import p0_protocol_audit as p0
    output = []
    for item in items:
        image, layers = p0.encode_query(impl, backbone, item["image_path"], cfg)
        if (len(layers) != 1 or tuple(layers[0].shape) != (784, 4096) or
                layers[0].dtype != torch.float32 or not torch.isfinite(layers[0]).all().item()):
            raise A4Error(f"{category}: native identity feature contract failed")
        output.append({**item, "image": image.copy(),
                       "features": [layers[0].detach().contiguous()],
                       "feature_bytes": int(layers[0].numel() * layers[0].element_size())})
    return output


def _kcenter_candidates(category: str, groups: Sequence[Mapping[str, Any]],
                        support_cache: Mapping[str, Any], p0: Any,
                        impl: Any, backbone: Any, cfg: Any,
                        *, cap: int) -> tuple[Any, list[str], dict[str, Any]]:
    import torch
    from experiments.calibration import a_calibration_continuation as a3

    entries_by_group: dict[str, tuple[Mapping[str, Any], Sequence[Any]]] = {}
    available: dict[str, int] = {}
    extraction_start = time.perf_counter()
    for item in groups:
        cached = support_cache.get(item["image_path"])
        if cached is None:
            cached = p0.encode_reference_features(impl, backbone, [item["image_path"]], cfg)
            if set(cached) != {item["image_path"]}:
                raise A4Error(f"{category}: coreset feature extraction escaped its path")
            entries = cached[item["image_path"]]
        else:
            entries = cached
        if not entries or any(len(v) != 1 or tuple(v[0].shape) != (784, 4096) or
                              v[0].dtype != torch.float16 for v in entries):
            raise A4Error(f"{category}: registered augmentation descriptors changed")
        group_sha = str(item["group_sha256"])
        entries_by_group[group_sha] = (item, entries)
        available[group_sha] = sum(int(v[0].shape[0]) for v in entries)
    torch.cuda.synchronize()
    feature_seconds = time.perf_counter() - extraction_start
    namespace = f"a3b-fexp-kc-v1\0{category}"
    quotas = a3.a3b_candidate_quotas(list(available), cap, available, namespace)
    selected: list[tuple[str, np.ndarray]] = []
    for group_sha in sorted(entries_by_group):
        item, entries = entries_by_group[group_sha]
        candidate_ids: list[str] = []
        blocks = []
        for view_index, feature_set in enumerate(entries):
            block = feature_set[0].detach().to("cpu").contiguous().numpy()
            blocks.append(block)
            for patch_index in range(784):
                candidate_ids.append(a3.a3b_candidate_identity(
                    "a3b-fexp-kc-v1", group_sha, str(item["relative_id"]),
                    view_index, patch_index))
        matrix = np.concatenate(blocks, axis=0)
        take = int(quotas[group_sha])
        ordered = sorted(range(len(candidate_ids)), key=lambda i: candidate_ids[i])[:take]
        selected.extend((candidate_ids[i], matrix[i].copy()) for i in ordered)
    selected.sort(key=lambda pair: pair[0])
    ids = [p[0] for p in selected]
    candidates = np.stack([p[1] for p in selected]).astype(np.float16, copy=False)
    if len(ids) != min(cap, sum(available.values())) or len(set(ids)) != len(ids):
        raise A4Error(f"{category}: K-Center candidate budget/identities failed")
    if len(candidates) < FIXED_COUNT:
        raise A4Error(f"{category}: K-Center candidate pool cannot supply 8,192 prototypes")
    selection_seconds = time.perf_counter() - extraction_start - feature_seconds
    receipt = {
        "candidate_pool_source_groups": len(groups),
        "candidate_count_by_group": available,
        "candidate_quota_by_group": quotas,
        "candidate_count": int(len(ids)),
        "candidate_identity_sha256": _sha_bytes(_json_bytes(ids)),
        "candidate_matrix_sha256": _sha_bytes(candidates.tobytes(order="C")),
        "candidate_matrix_bytes": int(candidates.nbytes),
        "feature_extraction_seconds": feature_seconds,
        "candidate_selection_seconds": selection_seconds,
        "candidate_policy": "A3b identity-hash quota balanced by official Source Group",
        "kcenter_policy": "greedy max-min cosine; first/ties by sorted candidate identity SHA-256",
        "candidate_cap": int(cap),
    }
    del selected, entries_by_group
    return candidates, ids, receipt


def _build_bank(name: str, *, category: str, impl: Any, p0: Any, r1: Any,
                support_cache: Mapping[str, Any], support_paths: Sequence[str], cfg: Any,
                expected_sha: str | None, prototypes: Any | None = None) -> Any:
    import torch
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.risk_safe.native_bridge import dynamic_codebooks
    from v02_memory import MemoryArm

    impl.set_seed(0)
    bank = p0.build_vmb(impl, support_cache, list(support_paths), cfg)
    bank.corr_cfg.enabled = False
    bank.use_defect_bank = False
    if expected_sha is not None:
        evidence = r1._fixed_bank_digest_evidence(bank, p0, expected_sha, _fixed_bank_digest)
        if not evidence["matches_expected_structured_digest"]:
            raise A4Error(f"{category}/{name}: native Support-0 bank differs from A1 receipt")
    if prototypes is not None:
        fixed = bank.normal_fixed["p0"][0]["id"][0]
        fixed.set_prototypes(prototypes.to(bank.device))
    fixed_sha = _fixed_bank_digest(bank)
    profile = p0.fixed_bank_profile(bank)
    try:
        _validate_fixed_profile(profile)
    except A4Error as exc:
        raise A4Error(f"{category}/{name}: {exc}") from exc
    if prototypes is not None and tuple(prototypes.shape) != (FIXED_COUNT, FEATURE_DIM):
        raise A4Error(f"{category}/{name}: prototype shape changed")
    settings = SimpleNamespace(flat=GREEDY_K, stm=GREEDY_K, ltm=0, buffer=0,
                               min_events=2, promote_every=16,
                               cap_bytes=2 * 1024**3)
    writer = MemoryArm("greedy_coreset", bank, settings, seed=0)
    if any(int(cb.K) != GREEDY_K or cb.size() != 0 for cb in dynamic_codebooks(bank)):
        raise A4Error(f"{category}/{name}: native Greedy K=1024 did not start empty")
    bank.use_defect_bank = False
    if bank.corr_cfg.enabled:
        raise A4Error(f"{category}/{name}: Correction scoring is enabled")
    return SimpleNamespace(name=name, bank=bank, writer=writer,
                           fixed_sha256=fixed_sha, fixed_profile=profile,
                           g_version=0)


def _bank_state_sha(traj: Any, r1: Any) -> str:
    return _sha_bytes((traj.fixed_sha256 + r1._memory_content_sha256(traj.bank)).encode("ascii"))


def _native_parity(category: str, impl: Any, r1: Any, native_bridge: Any,
                   trajectory: Any, sample: Mapping[str, Any], cfg: Any,
                   parity_step: int) -> dict[str, Any]:
    import copy
    from types import SimpleNamespace
    from v02_memory import MemoryArm
    from experiments.risk_safe.native_bridge import dynamic_codebooks

    bank = trajectory.bank
    image, feature_layers = sample["image"], sample["features"]
    score_map, top_ids, direct_score, summary = r1._native_dense_score(
        impl, bank, image, feature_layers, cfg)
    score_parity = native_bridge.verify_untouched_parity(
        impl, bank, image, feature_layers, cfg)
    score_parity["direct_r1_score"] = float(direct_score)
    score_parity["top32_count"] = len(top_ids)
    disposable = copy.deepcopy(bank)
    settings = SimpleNamespace(flat=GREEDY_K, stm=GREEDY_K, ltm=0, buffer=0,
                               min_events=2, promote_every=16,
                               cap_bytes=2 * 1024**3)
    arm = MemoryArm("greedy_coreset", disposable, settings, seed=0)
    update_parity = native_bridge.verify_native_update_parity(
        impl, arm, feature_layers,
        {"per_layer_summary": {"layer0": {
            "topk_patch_idx": np.asarray(top_ids, dtype=np.int64).tolist()}}},
        score=float(direct_score), theta=float(direct_score) + max(1.0, abs(float(direct_score))),
        sigma=SIGMA_FLOOR, step=int(parity_step))
    occupancy = [int(cb.size()) for cb in dynamic_codebooks(disposable)]
    if not any(occupancy):
        raise A4Error(f"{category}/{trajectory.name}: parity writer did not perform a genuine update")
    return {"status": "PASS", "arm": trajectory.name,
            "fixed_bank_sha256": trajectory.fixed_sha256,
            "native_score_parity": score_parity,
            "native_update_parity": update_parity,
            "disposable_post_update_occupancy": occupancy,
            "evaluation_state_reused": False, "test_outcomes_accessed": False}


def _build_category_systems(category: str, roles: Mapping[str, Any], *,
                            impl: Any, p0: Any, r1: Any, p0cfg: Any,
                            backbone: Any, output: Path) -> dict[str, Any]:
    import copy
    import torch
    from experiments.calibration import a_calibration_continuation as a3
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.risk_safe import native_bridge

    support_paths = [x["image_path"] for x in roles["support_items"]]
    torch.cuda.synchronize()
    support_encoding_started = time.perf_counter()
    support_cache = p0.encode_reference_features(impl, backbone, support_paths, p0cfg)
    torch.cuda.synchronize()
    support_encoding_seconds = time.perf_counter() - support_encoding_started
    if set(support_cache) != set(support_paths):
        raise A4Error(f"{category}: Support augmentation feature cache is incomplete")
    for path in support_paths:
        if len(support_cache[path]) != int(p0cfg.augmentations):
            raise A4Error(f"{category}: native Support augmentation count changed")
    bank_a = _build_bank("A_B_NATIVE_S4", category=category, impl=impl,
        p0=p0, r1=r1, support_cache=support_cache, support_paths=support_paths,
        cfg=p0cfg, expected_sha=roles["a1_support0_fixed_bank_sha256"])
    traj_a = SimpleNamespace(name="A_B_SHARED", bank=bank_a.bank, writer=bank_a.writer,
        fixed_sha256=bank_a.fixed_sha256, fixed_profile=bank_a.fixed_profile,
        g_version=0)
    candidate_c, ids_c, receipt_c = _kcenter_candidates(
        category, roles["support_items"], support_cache, p0, impl, backbone,
        p0cfg, cap=KC_CAP)
    kcenter_c_start = time.perf_counter()
    prototypes_c, selected_c = a3.a3b_kcenter_cuda(
        candidate_c, ids_c, FIXED_COUNT, block_rows=KC_BLOCK_ROWS, return_indices=True)
    prototypes_c_repeat, selected_c_repeat = a3.a3b_kcenter_cuda(
        candidate_c, ids_c, FIXED_COUNT, block_rows=KC_BLOCK_ROWS, return_indices=True)
    selected_ids_c = [ids_c[i] for i in selected_c]
    if [ids_c[i] for i in selected_c_repeat] != selected_ids_c:
        raise A4Error(f"{category}: C K-Center selected identities are not repeatable")
    if not torch.equal(prototypes_c, prototypes_c_repeat):
        raise A4Error(f"{category}: C K-Center prototype bytes changed on repeat")
    receipt_c["kcenter_seconds_including_repeat"] = time.perf_counter() - kcenter_c_start
    receipt_c.update({"selected_identity_sha256": _sha_bytes(_json_bytes(selected_ids_c)),
                      "prototype_sha256": _sha_bytes(prototypes_c.detach().cpu().contiguous().numpy().tobytes()),
                      "repeat_selected_identity_sha256": _sha_bytes(_json_bytes([ids_c[i] for i in selected_c_repeat])),
                      "repeat_identical": True,
                      "prototype_count": int(len(selected_c)),
                      "constructor_group_identity_sha256": _sha_bytes(_json_bytes(
                          sorted(x["group_sha256"] for x in roles["support_items"]))),
                      "constructor_image_count": len(roles["support_items"]),
                      "calibration_groups_excluded": True,
                      "cuda_fp32_candidate_copy_bytes": int(len(candidate_c) * FEATURE_DIM * 4),
                      "selection_seconds": receipt_c["kcenter_seconds_including_repeat"]})
    traj_c = _build_bank("C_KCENTER_S4", category=category, impl=impl,
        p0=p0, r1=r1, support_cache=support_cache, support_paths=support_paths,
        cfg=p0cfg, expected_sha=None, prototypes=prototypes_c)
    del candidate_c, prototypes_c_repeat

    d_pool = list(roles["support_items"]) + list(roles["r32_items"])
    if len(d_pool) != 36 or {x["group"] for x in d_pool} & set(roles["c32_groups"]):
        raise A4Error(f"{category}: D constructor pool must be exactly S4 plus R32")
    candidate_d, ids_d, receipt_d = _kcenter_candidates(
        category, d_pool, support_cache, p0, impl, backbone, p0cfg, cap=KC_CAP)
    if len(candidate_d) != KC_CAP:
        raise A4Error(f"{category}: D did not fill the registered 65,536 candidate cap")
    kcenter_d_start = time.perf_counter()
    prototypes_d, selected_d = a3.a3b_kcenter_cuda(
        candidate_d, ids_d, FIXED_COUNT, block_rows=KC_BLOCK_ROWS, return_indices=True)
    prototypes_d_repeat, selected_d_repeat = a3.a3b_kcenter_cuda(
        candidate_d, ids_d, FIXED_COUNT, block_rows=KC_BLOCK_ROWS, return_indices=True)
    selected_ids_d = [ids_d[i] for i in selected_d]
    if [ids_d[i] for i in selected_d_repeat] != selected_ids_d:
        raise A4Error(f"{category}: D K-Center selected identities are not repeatable")
    if not torch.equal(prototypes_d, prototypes_d_repeat):
        raise A4Error(f"{category}: D K-Center prototype bytes changed on repeat")
    receipt_d["kcenter_seconds_including_repeat"] = time.perf_counter() - kcenter_d_start
    receipt_d.update({"selected_identity_sha256": _sha_bytes(_json_bytes(selected_ids_d)),
                      "prototype_sha256": _sha_bytes(prototypes_d.detach().cpu().contiguous().numpy().tobytes()),
                      "repeat_selected_identity_sha256": _sha_bytes(_json_bytes([ids_d[i] for i in selected_d_repeat])),
                      "repeat_identical": True,
                      "prototype_count": int(len(selected_d)),
                      "selection_seconds": receipt_d["kcenter_seconds_including_repeat"],
                      "constructor_group_identity_sha256": _sha_bytes(_json_bytes(
                          sorted(x["group_sha256"] for x in d_pool))),
                      "constructor_image_count": len(d_pool),
                      "C32_excluded_from_constructor": True,
                      "cuda_fp32_candidate_copy_bytes": int(len(candidate_d) * FEATURE_DIM * 4)})
    traj_d = _build_bank("D_KCENTER_S4_R32", category=category, impl=impl,
        p0=p0, r1=r1, support_cache=support_cache, support_paths=support_paths,
        cfg=p0cfg, expected_sha=None, prototypes=prototypes_d)
    del candidate_d, prototypes_d_repeat, prototypes_c, prototypes_d
    if len({traj_a.fixed_sha256, traj_c.fixed_sha256, traj_d.fixed_sha256}) != 3:
        raise A4Error(f"{category}: A/C/D fixed bank identities are not distinct")
    static = {"A": copy.deepcopy(traj_a.bank), "C": copy.deepcopy(traj_c.bank),
              "D": copy.deepcopy(traj_d.bank)}
    trajs = {"A": traj_a, "B": traj_a, "C": traj_c, "D": traj_d}

    c64_encoding_started = time.perf_counter()
    c64_enc = _encode_identity_items(impl, backbone, p0cfg, roles["c64_items"], category)
    torch.cuda.synchronize()
    c64_encoding_seconds = time.perf_counter() - c64_encoding_started
    c64_by_group = {item["group"]: item for item in c64_enc}
    c32_enc = [c64_by_group[g] for g in roles["c32_groups"]]
    c64_ordered = [c64_by_group[x["group"]] for x in roles["c64_items"]]
    calibrations = {
        "A": c64_ordered, "B": c32_enc, "C": c32_enc, "D": c32_enc,
    }
    states: dict[str, B3State] = {}
    score_refs: dict[str, list[float]] = {}
    writer_refs: dict[str, dict[str, float]] = {}
    calibration_seconds: dict[str, float] = {}
    for arm in ARMS:
        traj = trajs[arm]
        started = time.perf_counter()
        vals = []
        for item in calibrations[arm]:
            vals.append(float(r1._native_dense_score(
                impl, traj.bank, item["image"], item["features"], p0cfg)[2]))
        torch.cuda.synchronize()
        score_refs[arm] = vals
        states[arm] = B3State(vals, version=traj.g_version)
        calibration_seconds[arm] = time.perf_counter() - started
        if arm in ("C", "D"):
            writer_refs[arm] = {"scores": vals,
                "theta": _higher_q90(vals),
                "sigma": max(float(np.std(vals, ddof=1)), SIGMA_FLOOR)}
    # B's C32 reference scores are the common A/B writer reference on the same bank.
    writer_vals = [float(v) for v in score_refs["B"]]
    writer_refs["A_B_SHARED"] = {"scores": writer_vals,
        "theta": _higher_q90(writer_vals),
        "sigma": max(float(np.std(writer_vals, ddof=1)), SIGMA_FLOOR),
        "source": "C32 M0 scores on shared native Support-0 bank"}
    parity_sample = c64_enc[0]
    parity = {
        "A_B_SHARED": _native_parity(category, impl, r1, native_bridge, traj_a,
                                      parity_sample, p0cfg, 0),
        "C": _native_parity(category, impl, r1, native_bridge, traj_c,
                            parity_sample, p0cfg, 1),
        "D": _native_parity(category, impl, r1, native_bridge, traj_d,
                            parity_sample, p0cfg, 2),
    }
    bank_receipts = {
        "A": {"fixed_sha256": traj_a.fixed_sha256, "profile": traj_a.fixed_profile,
              "constructor_groups": 4, "constructor_role": "Support-0",
              "correction_enabled": False, "defect_bank_enabled": False,
              "online_writer": "native Greedy K=1024"},
        "B": {"fixed_sha256": traj_a.fixed_sha256, "profile": traj_a.fixed_profile,
              "constructor_groups": 4, "constructor_role": "same immutable Support-0 bank as A",
              "correction_enabled": False, "defect_bank_enabled": False,
              "online_writer": "same native Greedy K=1024 trajectory as A"},
        "C": {"fixed_sha256": traj_c.fixed_sha256, "profile": traj_c.fixed_profile,
              "constructor_groups": 4, "constructor_role": "Support-0 augmentation descriptors",
              "correction_enabled": False, "defect_bank_enabled": False,
              "online_writer": "native Greedy K=1024",
              "kcenter": receipt_c},
        "D": {"fixed_sha256": traj_d.fixed_sha256, "profile": traj_d.fixed_profile,
              "constructor_groups": 36, "constructor_role": "Support-0 plus R32 augmentation descriptors",
              "correction_enabled": False, "defect_bank_enabled": False,
              "online_writer": "native Greedy K=1024",
              "kcenter": receipt_d},
    }
    initialization = {
        "schema": "A4_EB68_v1.initialization_private.v1", "category": category,
        "calibration_scores": {arm: [float(v) for v in vals] for arm, vals in score_refs.items()},
        "calibration_counts": {arm: len(calibrations[arm]) for arm in ARMS},
        "calibration_group_counts": {arm: len({x["group"] for x in calibrations[arm]}) for arm in ARMS},
        "initial_states": {arm: {"theta": states[arm].theta, "sigma": states[arm].sigma,
                                  "version": states[arm].version,
                                  "history": states[arm].history} for arm in ARMS},
        "writer_references": {k: {key: val for key, val in value.items() if key != "scores"}
                              for k, value in writer_refs.items()},
        "writer_reference_scores": {k: [float(v) for v in value["scores"]]
                                    for k, value in writer_refs.items()},
        "fixed_banks": bank_receipts, "native_parity": parity,
        "calibration_score_seconds": calibration_seconds,
        "test_labels_read": False, "test_masks_opened": False,
    }
    # The exact C64 calibration feature set is retained; C32 arrays are shared views into it.
    unique_calibration_bytes = sum(int(x["feature_bytes"]) for x in c64_enc)
    unique_calibration_image_bytes = sum(int(np.asarray(x["image"]).nbytes) for x in c64_enc)
    support_augmentation_bytes = sum(
        int(view[0].numel() * view[0].element_size())
        for entries in support_cache.values() for view in entries)
    identity_metadata_bytes = len(_json_bytes({
        "support": [{"group": x["group"], "relative_id": x["relative_id"]}
                    for x in roles["support_items"]],
        "C64": [{"group": x["group"], "relative_id": x["relative_id"]}
                for x in roles["c64_items"]],
        "C32_groups": roles["c32_groups"], "R32_groups": roles["r32_groups"]}))
    result = {"trajectories": trajs, "static_banks": static,
              "states": states, "calibrations": calibrations,
              "calibration_items_c64": c64_ordered,
              "initialization": initialization,
              "writer_refs": writer_refs, "bank_receipts": bank_receipts,
              "support_cache": support_cache,
              "candidate_receipts": {"C": receipt_c, "D": receipt_d},
              "parity": parity,
              "calibration_score_seconds": calibration_seconds,
              "unique_calibration_feature_bytes": unique_calibration_bytes,
              "unique_calibration_image_bytes": unique_calibration_image_bytes,
              "support_augmentation_feature_bytes": support_augmentation_bytes,
              "identity_metadata_bytes": identity_metadata_bytes,
              "fixed_memory_components": {
                  "unique_online_fixed_bank_bytes": 3 * FIXED_BYTES,
                  "static_M0_shadow_bank_bytes": 3 * FIXED_BYTES,
                  "per_arm_deployed_fixed_reference_bytes": FIXED_BYTES,
              },
              "feature_access_roles": {
                  "Support4_augmentation_images": 4,
                  "C64_identity_calibration_images": 64,
                  "D_R32_augmentation_images": 32,
                  "other_train_image_identities": 0,
              },
              "encoding_seconds": {"C64_identity_calibration": c64_encoding_seconds,
                                    "Support_augmentation": support_encoding_seconds,
                                    "D_R32_augmentation": receipt_d["feature_extraction_seconds"]},
              "model": {}}
    return result


def _update_trajectory(traj: Any, *, position: int, scored: Mapping[str, Any],
                       score: float, writer_ref: Mapping[str, float],
                       impl: Any, r1: Any, a3: Any) -> dict[str, Any]:
    before_sha = r1._memory_content_sha256(traj.bank)
    action = a3._trajectory_writer_update(
        traj, position=position, feature_layers=scored["features"],
        top_ids=scored["top_ids"], selected_features=scored["selected"],
        score=float(score), theta=float(writer_ref["theta"]),
        sigma=float(writer_ref["sigma"]), queried=True, label=0,
        prediction=int(float(score) >= float(writer_ref["theta"])), impl=impl, r1=r1)
    after_sha = r1._memory_content_sha256(traj.bank)
    changed = before_sha != after_sha
    if changed:
        traj.g_version += 1
    if bool(action["state_changed"]) != changed:
        raise A4Error(f"{traj.name}: Greedy state change audit mismatch")
    action_details = action["action"]
    return {"state_changed": changed, "state_version": int(traj.g_version),
            "memory_sha256_before": before_sha, "memory_sha256_after": after_sha,
            "action": action_details, "eligibility_reason": action_details.get(
                "eligibility_reason"),
            "accepted_candidate_patches": int(action_details.get("accepted_patches_delta", 0)),
            "skipped_candidate_patches": int(action_details.get("skipped_patches_delta", 0)),
            "candidate_patch_count": int(action_details.get("keep_count", 0)),
            "greedy_occupancy": action["greedy_occupancy"],
            "allocated_state_bytes": action["allocated_state_bytes"],
            "prototype_slot_changes": action["prototype_slot_changes"]}


def _run_category_child(run_id: str, category: str, *, smoke: bool = False,
                        smoke_max_events: int = 128) -> dict[str, Any]:
    import torch
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import stage05
    from experiments.calibration import a_calibration_scale as a1
    from experiments.calibration import a_calibration_continuation as a3
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration import r9_correction_mechanism_falsification as r9
    from experiments.calibration import profiles
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job

    if category not in CATEGORIES or not current_process_is_in_job():
        raise A4Error("Category worker is outside the authorized category or Windows Job Object")
    run_root = RUN_ROOT / run_id
    category_dir = run_root / ("smoke" if smoke else "stage_a") / category
    if category_dir.exists():
        seal_path = category_dir / "stage_a_seal.json"
        if not smoke and seal_path.is_file():
            seal = _read(seal_path)
            if seal.get("status") == "SEALED_COMPLETE" and seal.get("seal_sha256") == _sha_file(seal_path):
                return {"category": category, "status": "SEALED_COMPLETE", "reused": True}
        raise A4Error(f"Refusing to overwrite partial or existing category output: {category_dir}")
    category_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    _gpu_receipt()
    torch.set_num_threads(1)
    impl = p0.require_sources()
    model_receipt = r3._get_verified_model_receipt()
    restore_hub, model_profile = r3._runtime_and_native_setup(torch, impl, model_receipt)
    p0cfg, scorer_profile = r3._native_config(p0, profiles, r1)
    backbone = None
    reader = None
    try:
        try:
            backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
        finally:
            restore_hub()
        if (list(backbone.layers) != [5, 11, 17, 23] or
                backbone.img_size != 392 or backbone.resize_size != 448 or
                backbone.model_name != "dinov2_vitl14_reg" or not backbone.use_fp16 or
                int(p0cfg.augmentations) != 4 or p0cfg.include_identity is not False or
                int(p0cfg.knn) != 5 or p0cfg.image_aggregation != "max"):
            raise A4Error("Loaded backbone/native scorer differs from the frozen A4 feature contract")
        loaded_sha = r1._model_state_sha256(backbone.model)
        if loaded_sha != model_receipt.get("loaded_model_state_sha256"):
            raise A4Error("Loaded frozen DINOv2 state differs from the verified model receipt")
        model_profile["loaded_model_state_sha256"] = loaded_sha
        model_profile["layer_indices"] = list(backbone.layers)
        model_profile["image_size"] = int(backbone.img_size)
        model_profile["resize_size"] = int(backbone.resize_size)
        model_profile["model_name"] = str(backbone.model_name)
        train_rows, test_rows, json_sha = stage05.read_class_stream(
            DATASET_ROOT, JSON_ROOT, category, test_labels=False)
        roles = _category_roles(category, train_rows, test_rows, json_sha)
        query = _query_identity_receipt(category, test_rows)
        events = query["events"]
        if [int(e.position) for e in events] != list(range(len(test_rows))):
            raise A4Error(f"{category}: shuffled P0 event positions are not contiguous")
        json_path = JSON_ROOT / f"{category}.json"
        r7.validate_image_projection(json_path, category, events)
        systems = _build_category_systems(category, roles, impl=impl, p0=p0,
            r1=r1, p0cfg=p0cfg, backbone=backbone, output=category_dir)
        systems["model"] = {"runtime_profile": model_profile,
                             "scorer_profile": asdict(scorer_profile),
                             "loaded_model_state_sha256": loaded_sha}
        # Smoke uses the real tape prefix and only its legal queried labels.
        positions = query["positions"]
        n_events = min(len(events), int(smoke_max_events)) if smoke else len(events)
        query_positions = [p for p in positions if p < n_events]
        reader = r7.IndexedQueriedLabelReader(
            json_path, category, json_sha, events, query_positions)
        query_set = set(query_positions)
        states: dict[str, B3State] = systems["states"]
        trajectories = systems["trajectories"]
        static_banks = systems["static_banks"]
        writer_refs = systems["writer_refs"]
        # Initial reference scores and exact identity receipt contain no TEST outcomes.
        _write_json_once(category_dir / "initialization_private.json", systems["initialization"])
        score_path = category_dir / "stage_a_scores.jsonl"
        feedback_path = category_dir / "private_feedback.jsonl"
        score_stream = score_path.open("xb")
        feedback_stream = feedback_path.open("xb")
        score_hash = hashlib.sha256()
        feedback_hash = hashlib.sha256()
        feedback_counts: Counter[str] = Counter()
        state_change_counts: Counter[str] = Counter()
        threshold_change_counts: Counter[str] = Counter()
        encode_seconds = 0.0
        static_score_seconds = 0.0
        online_score_seconds = 0.0
        update_seconds = 0.0
        rescore_seconds = 0.0
        queried_groups: set[str] = set()
        max_cuda_allocated = 0
        event_started = time.perf_counter()
        try:
            for position, event in enumerate(events[:n_events]):
                event_start = time.perf_counter()
                causal = r1.CausalEventOrder(position)
                enc_start = time.perf_counter()
                image, encoded = p0.encode_query(impl, backbone, str(event.image_path), p0cfg)
                torch.cuda.synchronize()
                encode_seconds += time.perf_counter() - enc_start
                features = [encoded[0].detach().contiguous()]
                if tuple(features[0].shape) != (784, 4096) or not torch.isfinite(features[0]).all().item():
                    raise A4Error(f"{category}:{position}: TEST feature contract failed")
                scores: dict[str, float] = {}
                predictions: dict[str, int] = {}
                thresholds: dict[str, float] = {}
                score_versions: dict[str, dict[str, Any]] = {}
                scored_online: dict[str, dict[str, Any]] = {}
                # A/B share one live bank and one static M0 copy. C and D are independent.
                started_static = time.perf_counter()
                for arm, static_key in (("A", "A"), ("C", "C"), ("D", "D")):
                    static_map, _ids, static_score, _summary = r1._native_dense_score(
                        impl, static_banks[static_key], image, features, p0cfg)
                    scores[SCORE_KEYS[arm]["static"]] = float(static_score)
                    scores[SCORE_KEYS["B"]["static"]] = float(static_score) if arm == "A" else scores.get(SCORE_KEYS["B"]["static"], float("nan"))
                    state = states[arm]
                    thresholds[SCORE_KEYS[arm]["static"]] = float(state.history[0]["theta"])
                    predictions[SCORE_KEYS[arm]["static"]] = int(static_score >= state.history[0]["theta"])
                    if arm == "A":
                        thresholds[SCORE_KEYS["B"]["static"]] = float(states["B"].history[0]["theta"])
                        predictions[SCORE_KEYS["B"]["static"]] = int(static_score >= states["B"].history[0]["theta"])
                torch.cuda.synchronize()
                static_score_seconds += time.perf_counter() - started_static
                started_online = time.perf_counter()
                online_by_trajectory: dict[str, dict[str, Any]] = {}
                for arm in ("A", "C", "D"):
                    traj = trajectories[arm]
                    base_map, ids, score, summary = r1._native_dense_score(
                        impl, traj.bank, image, features, p0cfg)
                    selected = impl.l2_normalize(features[0].to(traj.bank.device))[
                        torch.as_tensor(ids, device=traj.bank.device, dtype=torch.long)]
                    scored = {"base_map": base_map, "top_ids": np.asarray(ids, dtype=np.int64),
                              "base_score": float(score), "selected": selected,
                              "summary": summary, "features": features}
                    online_by_trajectory[traj.name] = scored
                    scored_online[arm] = scored
                    duplicate_arm = "B" if arm == "A" else None
                    key = SCORE_KEYS[arm]["online"]
                    scores[key] = float(score)
                    state = states[arm]
                    thresholds[key] = float(state.theta)
                    predictions[key] = int(score >= state.theta)
                    score_versions[key] = {"version": int(traj.g_version),
                                            "state_sha256": _bank_state_sha(traj, r1)}
                    if duplicate_arm:
                        bkey = SCORE_KEYS["B"]["online"]
                        scores[bkey] = float(score)
                        thresholds[bkey] = float(states["B"].theta)
                        predictions[bkey] = int(score >= states["B"].theta)
                        score_versions[bkey] = dict(score_versions[key])
                torch.cuda.synchronize()
                online_score_seconds += time.perf_counter() - started_online
                if scores[SCORE_KEYS["A"]["online"]] != scores[SCORE_KEYS["B"]["online"]]:
                    raise A4Error(f"{category}:{position}: A/B score trajectory diverged")
                expected_event_keys = {SCORE_KEYS[arm][mode] for arm in ARMS
                                       for mode in ("static", "online")}
                if (set(scores) != expected_event_keys or
                        set(predictions) != expected_event_keys or
                        set(thresholds) != expected_event_keys):
                    raise A4Error(f"{category}:{position}: four-arm score/decision map is incomplete")
                if (scores[SCORE_KEYS["A"]["static"]] != scores[SCORE_KEYS["B"]["static"]] or
                        predictions[SCORE_KEYS["A"]["online"]] != int(
                            scores[SCORE_KEYS["A"]["online"]] >= thresholds[SCORE_KEYS["A"]["online"]]) or
                        predictions[SCORE_KEYS["B"]["online"]] != int(
                            scores[SCORE_KEYS["B"]["online"]] >= thresholds[SCORE_KEYS["B"]["online"]])):
                    raise A4Error(f"{category}:{position}: A/B score or decision mapping failed")
                # Save a fully pre-feedback event row and fsync it before any label read.
                row = {"schema": "A4_EB68_v1.stage_a_event.v1", "category": category,
                    "condition": "P0", "position": position,
                    "original_json_position": int(event.original_position),
                    "event_id_sha256": str(event.event_id_sha256),
                    "image_identity_sha256": str(event.image_identity_sha256),
                    "source_group_sha256": str(event.source_group_sha256),
                    "camera_id": int(event.camera_id),
                    "query_identity_set_sha256": query["identity_set_sha256"],
                    "scores": scores, "predictions": predictions,
                    "thresholds": thresholds, "score_state_versions": score_versions,
                    "test_labels_accessed": False}
                causal.seal_scores()
                encoded_row = _json_bytes(row) + b"\n"
                score_stream.write(encoded_row)
                score_stream.flush()
                os.fsync(score_stream.fileno())
                score_hash.update(encoded_row)
                queried = causal.decide_query(position in query_set)
                feedback_counts["event_count"] += 1
                if queried:
                    label = causal.disclose(lambda: reader.label_for(
                        position, sealed_and_fsynced=True))
                    # A4 has no Correction operation; close that causal phase before
                    # allowing any native normal-memory writer call.
                    causal.corrections_updated()
                    if str(event.source_group_sha256) in queried_groups:
                        raise A4Error(f"{category}: Q-A-TS queried a Source Group twice")
                    queried_groups.add(str(event.source_group_sha256))
                    feedback_counts["queried_count"] += 1
                    feedback_counts["queried_normal"] += int(label == 0)
                    feedback_counts["queried_defect"] += int(label == 1)
                    feedback = {"schema": "A4_EB68_v1.queried_feedback.v1",
                        "category": category, "position": position,
                        "original_json_position": int(event.original_position),
                        "event_id_sha256": str(event.event_id_sha256),
                        "image_identity_sha256": str(event.image_identity_sha256),
                        "source_group_sha256": str(event.source_group_sha256),
                        "label": int(label), "score_row_fsynced_before_disclosure": True,
                        "writer_actions": {}, "state_transitions": {},
                        "threshold_refreshes": []}
                    if label == 1:
                        feedback["writer_actions"] = {
                            "A_B_SHARED": {"action": {"action": "verified_defect_no_normal_write"}},
                            "C": {"action": {"action": "verified_defect_no_normal_write"}},
                            "D": {"action": {"action": "verified_defect_no_normal_write"}}}
                        feedback_counts["verified_defect_no_write"] += 1
                    else:
                        update_started = time.perf_counter()
                        for trajectory_name, arm, writer_key in (
                                ("A_B_SHARED", "A", "A_B_SHARED"),
                                ("C_KCENTER_S4", "C", "C"),
                                ("D_KCENTER_S4_R32", "D", "D")):
                            traj = trajectories[arm]
                            scored = scored_online[arm]
                            result = _update_trajectory(
                                traj, position=position, scored=scored,
                                score=float(scored["base_score"]),
                                writer_ref=writer_refs[writer_key], impl=impl,
                                r1=r1, a3=a3)
                            feedback["writer_actions"][trajectory_name] = result
                            feedback_counts["native_writer_calls"] += 1
                            feedback_counts["candidate_patches"] += result["candidate_patch_count"]
                            feedback_counts["accepted_candidate_patches"] += result["accepted_candidate_patches"]
                            feedback_counts["skipped_candidate_patches"] += result["skipped_candidate_patches"]
                            if result["state_changed"]:
                                state_change_counts[trajectory_name] += 1
                                feedback_counts[f"state_change_{trajectory_name}"] += 1
                                refresh_arms = ("A", "B") if arm == "A" else (arm,)
                                for refresh_arm in refresh_arms:
                                    refresh_items = systems["calibrations"][refresh_arm]
                                    rescore_started = time.perf_counter()
                                    new_scores = [float(r1._native_dense_score(
                                        impl, traj.bank, item["image"], item["features"], p0cfg)[2])
                                        for item in refresh_items]
                                    torch.cuda.synchronize()
                                    rescore_seconds += time.perf_counter() - rescore_started
                                    refreshed = states[refresh_arm].refresh(new_scores, traj.g_version)
                                    if not refreshed:
                                        raise A4Error(f"{category}: genuine state change failed to refresh B3 {refresh_arm}")
                                    threshold_change_counts[refresh_arm] += int(
                                        states[refresh_arm].theta != states[refresh_arm].history[-2]["theta"])
                                    feedback["threshold_refreshes"].append({
                                        "arm": refresh_arm, "version": traj.g_version,
                                        "apply_from_position": position + 1,
                                        "theta": states[refresh_arm].theta, "sigma": states[refresh_arm].sigma,
                                        "reference_score_count": len(new_scores),
                                        "score_sha256": _sha_bytes(_json_bytes(new_scores))})
                                feedback["state_transitions"][trajectory_name] = {
                                    "state_changed": True, "version": traj.g_version,
                                    "state_sha256": _bank_state_sha(traj, r1)}
                        torch.cuda.synchronize()
                        update_seconds += time.perf_counter() - update_started
                        feedback_counts["queried_normal_count"] += 1
                        event_write_actions = sum(
                            int(x.get("action", {}).get("action") == "native_greedy_normal_update")
                            for x in feedback["writer_actions"].values())
                        feedback_counts["native_normal_write_actions"] += event_write_actions
                        feedback_counts["verified_normal_write_events"] += int(event_write_actions > 0)
                    feedback_bytes = _json_bytes(feedback) + b"\n"
                    feedback_stream.write(feedback_bytes)
                    feedback_stream.flush()
                    os.fsync(feedback_stream.fileno())
                    feedback_hash.update(feedback_bytes)
                if not queried:
                    causal.corrections_updated()
                causal.memory_updated()
                causal.finish()
                if smoke and feedback_counts["native_normal_write_actions"] > 0:
                    # A4 smoke stops after one actual queried legal normal write.
                    if feedback_counts["queried_count"] >= len(query_positions):
                        pass
                    if position + 1 >= 1:
                        break
                if not smoke and (position + 1) % 256 == 0:
                    progress = {"category": category, "position": position,
                        "event_count": position + 1,
                        "query_count": int(feedback_counts["queried_count"]),
                        "state_versions": {arm: int(states[arm].version) for arm in ARMS},
                        "automatic_restart": False}
                    p = category_dir / "progress_private.json"
                    tmp = category_dir / "progress_private.json.tmp"
                    tmp.write_bytes(_json_bytes(progress) + b"\n")
                    _fsync_file(tmp)
                    os.replace(tmp, p)
                max_cuda_allocated = max(max_cuda_allocated,
                                         int(torch.cuda.max_memory_allocated()))
                if time.perf_counter() - started > CATEGORY_SECONDS:
                    raise A4Error(f"{category}: 120-minute category deadline exceeded")
        finally:
            for stream in (score_stream, feedback_stream):
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
        if smoke:
            if feedback_counts["native_normal_write_actions"] < 1:
                raise A4Error(f"{category}: bounded smoke found no legal normal write in {n_events} events")
            expected_decoded = [p for p in query_positions if p <= position]
            if reader.decoded_positions != expected_decoded:
                raise A4Error(f"{category}: smoke disclosure sequence differs from Q-A-TS prefix")
            seal_status = "INCOMPLETE"
        else:
            reader.finish()
            if (feedback_counts["queried_count"] != len(query_positions) or
                    len(queried_groups) != len(query_positions)):
                raise A4Error(f"{category}: full query allowlist was not consumed exactly")
            seal_status = "SEALED_COMPLETE"
        score_stream_sha = _sha_file(score_path)
        feedback_stream_sha = _sha_file(feedback_path)
        if score_stream_sha != score_hash.hexdigest() or feedback_stream_sha != feedback_hash.hexdigest():
            raise A4Error(f"{category}: fsynced stream digest disagrees with writer digest")
        final_states = {arm: {"theta": states[arm].theta, "sigma": states[arm].sigma,
                              "version": states[arm].version,
                              "history": states[arm].history} for arm in ARMS}
        a_b_mem_sha = r1._memory_content_sha256(trajectories["A"].bank)
        if a_b_mem_sha != r1._memory_content_sha256(trajectories["B"].bank):
            raise A4Error(f"{category}: A/B native writer trajectory did not remain shared")
        if states["A"].version != states["B"].version:
            raise A4Error(f"{category}: A/B B3 states do not track common memory versions")
        memory_audit = {
            "A": trajectories["A"].writer.audit(),
            "B": trajectories["B"].writer.audit(),
            "C": trajectories["C"].writer.audit(),
            "D": trajectories["D"].writer.audit(),
        }
        result = {
            "schema": "A4_EB68_v1.stage_a_result.v1", "status": seal_status,
            "category": category, "condition": "P0", "event_count": int(feedback_counts["event_count"]),
            "query_count": len(query_positions), "executed_query_count": int(feedback_counts["queried_count"]),
            "verified_normal_feedback": int(feedback_counts["queried_normal"]),
            "verified_defect_feedback": int(feedback_counts["queried_defect"]),
            "verified_normal_write_events": int(feedback_counts["verified_normal_write_events"]),
            "native_normal_write_actions": int(feedback_counts["native_normal_write_actions"]),
            "query_identity_set_sha256": query["identity_set_sha256"],
            "query_positions_sha256": query["positions_sha256"],
            "score_rows_sha256": score_stream_sha,
            "feedback_rows_sha256": feedback_stream_sha,
            "state_change_counts": dict(state_change_counts),
            "threshold_change_counts": dict(threshold_change_counts),
            "accepted_candidate_patches": int(feedback_counts["accepted_candidate_patches"]),
            "candidate_patches": int(feedback_counts["candidate_patches"]),
            "final_b3": final_states,
            "online_memory_audit_by_arm": memory_audit,
            "model": systems["model"],
            "fixed_bank_receipts": systems["bank_receipts"],
            "kcenter_receipts": systems["candidate_receipts"],
            "native_parity": systems["parity"],
            "unique_calibration_feature_bytes": systems["unique_calibration_feature_bytes"],
            "unique_calibration_image_bytes": systems["unique_calibration_image_bytes"],
            "support_augmentation_feature_bytes": systems["support_augmentation_feature_bytes"],
            "identity_metadata_bytes": systems["identity_metadata_bytes"],
            "fixed_memory_components": systems["fixed_memory_components"],
            "test_query_feature_bytes_per_event": 784 * 4096 * 4,
            "initialization_metadata_json_bytes": len(
                _json_bytes(systems["initialization"])) + 1,
            "feature_access_roles": systems["feature_access_roles"],
            "writer_reference_contract": "frozen C32 M0 q90/sample std for each fixed bank",
            "writer_reference_R9_difference": "A4 uses C32 M0; historical R9/A2 used its registered C64 source",
            "cost_seconds": {"test_feature_encoding": encode_seconds,
                "support_augmentation_encoding": systems["encoding_seconds"]["Support_augmentation"],
                "C64_identity_feature_encoding": systems["encoding_seconds"]["C64_identity_calibration"],
                "D_R32_augmentation_encoding": systems["encoding_seconds"]["D_R32_augmentation"],
                "calibration_native_scoring": float(sum(systems["calibration_score_seconds"].values())),
                "Kcenter_C_candidate_staging": systems["candidate_receipts"]["C"]["candidate_selection_seconds"],
                "Kcenter_D_candidate_staging": systems["candidate_receipts"]["D"]["candidate_selection_seconds"],
                "Kcenter_C_selection_including_repeat": systems["candidate_receipts"]["C"]["selection_seconds"],
                "Kcenter_D_selection_including_repeat": systems["candidate_receipts"]["D"]["selection_seconds"],
                "static_native_scoring": static_score_seconds,
                "online_native_scoring": online_score_seconds,
                "writer_updates": update_seconds,
                "protected_reference_rescoring": rescore_seconds,
                "category_wall": time.perf_counter() - started},
            "gpu_peak_allocated_bytes": max_cuda_allocated,
            "masks_opened": False, "unqueried_labels_accessed": False,
            "test_labels_read_only_after_query": True,
        }
        _write_json_once(category_dir / "stage_a_result.json", result)
        seal = {
            "schema": "A4_EB68_v1.stage_a_seal.v1", "status": seal_status,
            "classification": "SUPERVISED_SMOKE_NOT_SCIENTIFIC_RESULT" if smoke else "REGISTERED_P0_STAGE_A",
            "category": category, "condition": "P0", "event_count": result["event_count"],
            "query_count": len(query_positions), "query_positions": query_positions,
            "query_identity_set_sha256": query["identity_set_sha256"],
            "query_group_set_sha256": query["group_set_sha256"],
            "score_rows_sha256": score_stream_sha,
            "feedback_rows_sha256": feedback_stream_sha,
            "initialization_sha256": _sha_file(category_dir / "initialization_private.json"),
            "result_sha256": _sha_file(category_dir / "stage_a_result.json"),
            "category_json_sha256": json_sha,
            "a1_role_manifest_sha256": roles["a1_role_manifest_sha256"],
            "identity_sha256": roles["identity_sha256"],
            "fixed_bank_sha256_by_arm": {arm: systems["bank_receipts"][arm]["fixed_sha256"] for arm in ARMS},
            "all_event_scores_fsynced_before_query_feedback": True,
            "queried_labels_only": reader.decoded_positions,
            "queried_label_count": len(reader.decoded_positions),
            "unqueried_labels_accessed": False, "test_masks_opened": False,
            "feedback_counts": dict(feedback_counts),
            "state_change_counts": dict(state_change_counts),
            "final_b3": final_states,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_json_once(category_dir / "stage_a_seal.json", _json_bytes(seal) + b"\n")
        seal["seal_sha256"] = _sha_file(category_dir / "stage_a_seal.json")
        _write_json_once(category_dir / "category_complete.json", _json_bytes({
            "category": category, "status": seal_status,
            "seal_sha256": seal["seal_sha256"], "result_sha256": _sha_file(category_dir / "stage_a_result.json")}) + b"\n")
        return {"category": category, "status": seal_status,
                "event_count": result["event_count"],
                "verified_normal_write_events": result["verified_normal_write_events"],
                "stage_a_seal_sha256": seal["seal_sha256"]}
    except BaseException as exc:
        try:
            _write_json_once(category_dir / "failure_private.json", _json_bytes({
                "category": category, "stage": "SMOKE" if smoke else "STAGE_A",
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc()[-14000:],
                "status": "INCOMPLETE", "automatic_restart": False}) + b"\n")
        except FileExistsError:
            pass
        raise
    finally:
        if reader is not None:
            reader.close()
        if backbone is not None:
            try:
                backbone.cleanup()
            except Exception:
                pass


def _source_freeze_manifest(run_id: str, preflight_receipt: Mapping[str, Any]) -> dict[str, Any]:
    source_commit = _git("rev-parse", "HEAD")
    if not _git_paths(("diff", "--name-only")) == [] or not _git_paths(("diff", "--cached", "--name-only")) == []:
        raise A4Error("Cannot freeze source while tracked files remain modified")
    for relative, digest in preflight_receipt["source_hashes"].items():
        content = subprocess.run(["git", "show", f"{source_commit}:{relative}"], cwd=ROOT,
                                 check=True, capture_output=True).stdout
        if _sha_source_bytes(content) != digest:
            raise A4Error(f"Source-freeze commit does not contain execution hash for {relative}")
    protected = preflight_receipt["protected_historical_output_trees"]
    contract = {
        "protocol_id": PROTOCOL_ID, "authorization_id": AUTHORIZATION_ID,
        "parent": EXPECTED_PARENT, "categories": list(CATEGORIES), "condition": "P0",
        "roles": "A1 exact Support-0 + historical C64; A1 draw-0 C32/R32",
        "arms": {"A": "native S4 fixed; B3 C64", "B": "same shared native S4 trajectory; B3 C32",
                 "C": "A3b K-Center on S4 augmentation descriptors; B3 C32",
                 "D": "same K-Center on S4+R32, 65,536 balanced candidates; B3 C32"},
        "fixed_capacity": {"prototypes": FIXED_COUNT, "dimension": FEATURE_DIM,
                           "dtype": "fp16", "bytes": FIXED_BYTES},
        "greedy_capacity": GREEDY_K, "candidate_cap": KC_CAP,
        "query_policy": "Q-A-TS-v1, seed 37, 3%, 16 bins, P0 group shuffle seed 0",
        "writer_reference": "frozen per-fixed-bank C32 M0 higher-q90; sample std ddof 1 floor .02",
        "b3": {"q": Q90, "quantile_method": "higher", "clip": B3_DELTA,
               "sigma_floor": SIGMA_FLOOR, "ddof": 1},
        "test_feature_cache": "none; one event encoding shared across all arms",
    }
    return {"schema": "A4_EB68_v1.run_manifest.v1", "run_id": run_id,
            "protocol_id": PROTOCOL_ID, "authorization_id": AUTHORIZATION_ID,
            "git": {"branch": BRANCH, "source_freeze_commit": source_commit,
                    "expected_parent": EXPECTED_PARENT},
            "source_hashes": dict(preflight_receipt["source_hashes"]),
            "source_hash_normalization": "CRLF normalized to LF before SHA-256",
            "execution_source_hashes_sha256": _sha_bytes(_json_bytes(preflight_receipt["source_hashes"])),
            "experiment_contract": contract,
            "experiment_contract_sha256": _sha_bytes(_json_bytes(contract)),
            "preflight_sha256": _sha_bytes(_json_bytes(preflight_receipt)),
            "preflight": preflight_receipt,
            "protected_historical_output_trees": protected,
            "untracked_baseline": preflight_receipt["protected_file_audit"]["preexisting_untracked_inventory"],
            "access": {"test_labels_stage_a": "queried only after current row fsync",
                       "test_labels_stage_b": "after global Stage A seal only",
                       "unqueried_labels_in_controller": False, "masks": False,
                       "confirmation_categories": False},
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _run_child(run_id: str, *, stage: str, category: str | None = None,
               smoke: bool = False, timeout_seconds: float) -> dict[str, Any]:
    from experiments.calibration import r9_correction_mechanism_falsification as r9
    from experiments.calibration import r1_runtime_supervisor as supervisor
    command = [sys.executable, "-m", "experiments.calibration.a4_equal_budget_allocation",
               "--category-child" if stage in ("CATEGORY", "SMOKE") else "--stage-b-child",
               "--run-id", run_id]
    if stage in ("CATEGORY", "SMOKE"):
        command.extend(["--category", str(category)])
    if smoke:
        command.append("--smoke-child")
    child_env = r9._child_environment("STAGE_B" if stage == "STAGE_B" else "STAGE_A")
    result = supervisor.run_supervised_child(command, timeout_seconds=float(timeout_seconds),
                                               env=child_env, cwd=ROOT)
    record = asdict(result)
    if (record.get("status") != "COMPLETED" or record.get("child_exit_code") != 0 or
            record.get("termination_confirmed") is not True or
            record.get("active_job_processes") != 0):
        raise A4Error(f"Windows Job Object child {stage}/{category} failed: {record}")
    return {"stage": stage, "category": category, "supervisor": record,
            "process_tree_containment": "Windows Job Object", "automatic_restart": False}


def _verify_category_seal(run_id: str, category: str) -> dict[str, Any]:
    path = RUN_ROOT / run_id / "stage_a" / category / "stage_a_seal.json"
    if not path.is_file():
        raise A4Error(f"Missing Stage A seal for {category}")
    seal = _read(path)
    if seal.get("status") != "SEALED_COMPLETE":
        raise A4Error(f"Stage A seal is incomplete for {category}")
    parent = path.parent
    for filename, expected in (("stage_a_scores.jsonl", seal.get("score_rows_sha256")),
                               ("private_feedback.jsonl", seal.get("feedback_rows_sha256")),
                               ("initialization_private.json", seal.get("initialization_sha256")),
                               ("stage_a_result.json", seal.get("result_sha256"))):
        if not (parent / filename).is_file() or _sha_file(parent / filename) != expected:
            raise A4Error(f"{category}: Stage A checksum mismatch for {filename}")
    if seal.get("unqueried_labels_accessed") is not False or seal.get("test_masks_opened") is not False:
        raise A4Error(f"{category}: Stage A access seal is invalid")
    seal["seal_sha256"] = _sha_file(path)
    return seal


def _seal_global_stage_a(run_id: str) -> dict[str, Any]:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    run_root = RUN_ROOT / run_id
    hashes = {}
    for category in CATEGORIES:
        seal = _verify_category_seal(run_id, category)
        hashes[category] = seal["seal_sha256"]
    path = run_root / "stage_a_global_seal.json"
    if path.exists():
        prior = _read(path)
        if prior.get("category_seal_sha256") != hashes:
            raise A4Error("Existing global Stage A seal does not match complete categories")
        prior["seal_sha256"] = _sha_file(path)
        return prior
    payload = {"schema": "A4_EB68_v1.global_stage_a_seal.v1",
        "status": "SEALED_COMPLETE", "run_id": run_id,
        "category_seal_sha256": hashes,
        "all_categories_complete_before_stage_b": True,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write_json_once(path, payload)
    payload["seal_sha256"] = _sha_file(path)
    return payload


def _run_all(run_id: str) -> dict[str, Any]:
    from experiments.calibration import a4_independent_verifier as verifier
    from experiments.calibration import a4_report_renderer as renderer

    run_id = _validate_run_id(run_id)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    run_root = RUN_ROOT / run_id
    if not run_root.exists():
        receipt = preflight(check_source_freeze=True)
        _run_engineering_tests()
        run_root.mkdir(parents=False, exist_ok=False)
        _write_json_once(run_root / "run_manifest.json", _json_bytes(
            _source_freeze_manifest(run_id, receipt)) + b"\n")
        _write_json_once(run_root / "RUN_STATE.json", _json_bytes({
            "schema": "A4_EB68_v1.run_state.v1", "run_id": run_id,
            "status": "RUNNING", "campaign_started_epoch": time.time(),
            "completed_categories": [], "automatic_restart": False}) + b"\n")
        _write_json_once(run_root / "preflight_private.json", _json_bytes(receipt) + b"\n")
        if _untracked_baseline()["matches_pre_task_inventory"] is False:
            # A4 sources have been committed, so baseline pathname inventory must again match.
            raise A4Error("Pre-existing untracked pathname inventory changed before Stage A")
        smoke = _run_child(run_id, stage="SMOKE", category="audiojack", smoke=True,
                           timeout_seconds=min(900, CATEGORY_SECONDS))
        _append_jsonl(run_root / "supervisor_status.jsonl", smoke)
        smoke_seal = _read(run_root / "smoke/audiojack/stage_a_seal.json")
        if (smoke_seal.get("status") != "INCOMPLETE" or
                smoke_seal.get("feedback_counts", {}).get("verified_normal_write_events", 0) < 1 or
                smoke_seal.get("feedback_counts", {}).get("native_normal_write_actions", 0) < 1):
            raise A4Error("Bounded smoke did not record an incomplete genuine legal normal update")
        _write_run_state(run_root, "SMOKE_PASS", {"smoke_status": "INCOMPLETE_NONSCIENTIFIC"})
    else:
        manifest = _read(run_root / "run_manifest.json")
        _validate_manifest_source(manifest)
        state = _read(run_root / "RUN_STATE.json")
        if state.get("status") == "COMPLETE":
            verified = verifier.verify_run(run_id)
            if not (REPORT_ROOT / "EXPERIMENT_A4_REPORT_MANIFEST.json").is_file():
                renderer.render_run(run_id)
            complete_path = run_root / "complete_seal.json"
            if not complete_path.exists():
                _write_json_once(complete_path, _json_bytes({
                    "status": "COMPLETE", "run_id": run_id,
                    "stage_a_global_seal_sha256": _sha_file(run_root / "stage_a_global_seal.json"),
                    "stage_b_seal_sha256": _sha_file(run_root / "stage_b_seal.json"),
                    "verification_status": verified["status"],
                    "decision": _read(REPORT_ROOT / "EXPERIMENT_A4_REPORT_MANIFEST.json")["decision"]}) + b"\n")
            return verified
        if state.get("status") not in {"RUNNING", "SMOKE_PASS", "STAGE_A_PARTIAL", "STAGE_A_COMPLETE", "STAGE_B_COMPLETE", "REPORT_PENDING"}:
            raise A4Error(f"Run is not resumable from state {state.get('status')}")
        _run_engineering_tests()
    state = _read(run_root / "RUN_STATE.json")
    campaign_start = float(state.get("campaign_started_epoch", time.time()))
    for category in CATEGORIES:
        category_seal = run_root / "stage_a" / category / "stage_a_seal.json"
        if category_seal.is_file():
            _verify_category_seal(run_id, category)
            continue
        remain = CAMPAIGN_SECONDS - (time.time() - campaign_start)
        if remain <= 0:
            raise A4Error("Eight-hour A4 campaign deadline exceeded")
        _write_run_state(run_root, "STAGE_A_PARTIAL", {"active_category": category})
        result = _run_child(run_id, stage="CATEGORY", category=category,
                            timeout_seconds=min(CATEGORY_SECONDS, remain))
        _append_jsonl(run_root / "supervisor_status.jsonl", result)
        _verify_category_seal(run_id, category)
        state = _read(run_root / "RUN_STATE.json")
        completed = sorted(set(state.get("completed_categories", [])) | {category})
        _write_run_state(run_root, "STAGE_A_PARTIAL", {"completed_categories": completed})
    stage_a = _seal_global_stage_a(run_id)
    _write_run_state(run_root, "STAGE_A_COMPLETE", {"stage_a_global_seal_sha256": stage_a["seal_sha256"]})
    if not (run_root / "stage_b_seal.json").is_file():
        remain = CAMPAIGN_SECONDS - (time.time() - campaign_start)
        if remain <= 0:
            raise A4Error("Eight-hour A4 campaign deadline exceeded before Stage B")
        result = _run_child(run_id, stage="STAGE_B", timeout_seconds=min(1800, remain))
        _append_jsonl(run_root / "supervisor_status.jsonl", result)
    verified = verifier.verify_run(run_id)
    _write_run_state(run_root, "STAGE_B_COMPLETE", {"verification": verified})
    rendered = renderer.render_run(run_id)
    _write_run_state(run_root, "COMPLETE", {"report_files": rendered["files"],
                                              "scientific_decision": rendered["decision"]})
    complete_path = run_root / "complete_seal.json"
    if not complete_path.exists():
        _write_json_once(complete_path, _json_bytes({
            "status": "COMPLETE", "run_id": run_id,
            "stage_a_global_seal_sha256": _sha_file(run_root / "stage_a_global_seal.json"),
            "stage_b_seal_sha256": _sha_file(run_root / "stage_b_seal.json"),
            "verification_status": verified["status"],
            "decision": rendered["decision"]}) + b"\n")
    return {"run_id": run_id, "status": "COMPLETE", "verification": verified,
            "report": rendered}


def _validate_manifest_source(manifest: Mapping[str, Any]) -> None:
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("authorization_id") != AUTHORIZATION_ID:
        raise A4Error("A4 run manifest protocol/authorization mismatch")
    commit = str(manifest.get("git", {}).get("source_freeze_commit", ""))
    if not commit or _git("merge-base", "--is-ancestor", commit, "HEAD") != "":
        raise A4Error("Recorded A4 source-freeze commit is not an ancestor of current HEAD")
    observed = _source_hashes()
    if observed != manifest.get("source_hashes"):
        raise A4Error("Current A4 source/dependency hashes differ from the execution manifest")


def _run_engineering_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q",
               "tests/calibration/test_a4_equal_budget_allocation.py"]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                          timeout=1800)
    result = {"command": command, "exit_code": proc.returncode,
              "stdout": proc.stdout[-16000:], "stderr": proc.stderr[-16000:]}
    if proc.returncode != 0:
        raise A4Error(f"A4 engineering tests failed: {result}")
    return result


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _json_bytes(value) + b"\n"
    with path.open("ab") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def _write_run_state(run_root: Path, status: str, details: Mapping[str, Any]) -> None:
    path = run_root / "RUN_STATE.json"
    prior = _read(path) if path.is_file() else {}
    state = {**prior, **dict(details), "status": status,
             "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    temp = path.with_suffix(".json.tmp")
    temp.write_bytes(_json_bytes(state) + b"\n")
    _fsync_file(temp)
    os.replace(temp, path)


def _validate_run_id(value: str) -> str:
    import re
    if not re.fullmatch(r"a4_eb68_20261107_[a-zA-Z0-9_-]{1,48}", str(value)):
        raise A4Error("Run ID must begin a4_eb68_20261107_ and contain only safe characters")
    return str(value)


def _stage_b_child(run_id: str) -> dict[str, Any]:
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job
    if not current_process_is_in_job():
        raise A4Error("Stage B must execute inside its Windows Job Object")
    run_root = RUN_ROOT / run_id
    global_seal_path = run_root / "stage_a_global_seal.json"
    if not global_seal_path.is_file():
        raise A4Error("Stage B blocked before the global Stage A seal")
    global_seal = _read(global_seal_path)
    if global_seal.get("status") != "SEALED_COMPLETE" or set(global_seal.get("category_seal_sha256", {})) != set(CATEGORIES):
        raise A4Error("Global Stage A seal does not contain all four categories")
    for cat in CATEGORIES:
        seal = _verify_category_seal(run_id, cat)
        if seal["seal_sha256"] != global_seal["category_seal_sha256"][cat]:
            raise A4Error(f"Stage B blocked by invalid Stage A seal: {cat}")
    from experiments.calibration import stage05
    import sklearn.metrics as skm

    attempt = run_root / ("stage_b_attempt_" + str(time.time_ns()))
    attempt.mkdir(parents=False, exist_ok=False)
    all_metrics: dict[str, Any] = {}
    projection_receipts: dict[str, Any] = {}
    for category in CATEGORIES:
        source = run_root / "stage_a" / category
        seal = _read(source / "stage_a_seal.json")
        rows = _load_jsonl(source / "stage_a_scores.jsonl")
        if len(rows) != EXPECTED_EVENTS[category] or any("label" in row for row in rows):
            raise A4Error(f"Stage B found malformed sealed score tape for {category}")
        # All unqueried TEST outcomes are first loaded here, after every seal was validated.
        labels_json_order = np.asarray(stage05.test_labels_for_class(JSON_ROOT, category), dtype=np.int8)
        if labels_json_order.shape != (len(rows),) or set(np.unique(labels_json_order)) != {0, 1}:
            raise A4Error(f"Stage B label projection is malformed or single-class for {category}")
        labels = np.asarray([labels_json_order[int(row["original_json_position"])]
                             for row in rows], dtype=np.int8)
        label_lines = []
        for row, label in zip(rows, labels):
            label_lines.append(_json_bytes({"position": row["position"],
                "original_json_position": row["original_json_position"],
                "event_id_sha256": row["event_id_sha256"], "label": int(label)}) + b"\n")
        projection_path = attempt / f"{category}_label_projection_private.jsonl"
        with projection_path.open("xb") as f:
            for line in label_lines:
                f.write(line)
            f.flush()
            os.fsync(f.fileno())
        projection_sha = _sha_file(projection_path)
        projection_receipts[category] = {"sha256": projection_sha,
                                         "count": len(label_lines),
                                         "raw_labels_private": True}
        arm_metrics: dict[str, Any] = {}
        windows_by_arm: dict[str, Any] = {}
        for arm in ARMS:
            arm_metrics[arm] = {}
            windows_by_arm[arm] = {}
            for mode in ("static", "online"):
                key = SCORE_KEYS[arm][mode]
                scores, preds, theta = _stage_b_vectors(rows, arm, mode)
                if (scores.shape != labels.shape or not np.isfinite(scores).all() or
                        preds.shape != labels.shape or not set(np.unique(preds)) <= {0, 1} or
                        not np.isfinite(theta).all()):
                    raise A4Error(f"Stage B score mapping/decision vector invalid: {category}/{key}")
                tn = int(np.sum((labels == 0) & (preds == 0)))
                fp = int(np.sum((labels == 0) & (preds == 1)))
                fn = int(np.sum((labels == 1) & (preds == 0)))
                tp = int(np.sum((labels == 1) & (preds == 1)))
                fpr = fp / (fp + tn) if fp + tn else None
                fnr = fn / (fn + tp) if fn + tp else None
                ber = (fpr + fnr) / 2 if fpr is not None and fnr is not None else None
                result = {"auroc": float(skm.roc_auc_score(labels, scores)),
                    "average_precision": float(skm.average_precision_score(labels, scores)),
                    "ber": ber, "fpr": fpr, "fnr": fnr,
                    "balanced_accuracy": 1 - ber if ber is not None else None,
                    "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
                    "score_sha256": _sha_bytes(np.asarray(scores, dtype="<f8").tobytes()),
                    "prediction_sha256": _sha_bytes(np.asarray(preds, dtype=np.uint8).tobytes()),
                    "threshold_initial": float(theta[0]), "threshold_final": float(theta[-1]),
                    "threshold_changes": int(np.count_nonzero(np.diff(theta))),
                    "event_count": int(len(labels))}
                arm_metrics[arm][mode] = result
                window_rows = []
                for start in range(0, len(labels) - 511, 512):
                    stop = start + 512
                    yw, pw = labels[start:stop], preds[start:stop]
                    window_rows.append(_window_metric_row(labels, preds, start))
                windows_by_arm[arm][mode] = window_rows
        # A/B score identity is a gate; calibration affects decisions only.
        for mode in ("static", "online"):
            key_a, key_b = SCORE_KEYS["A"][mode], SCORE_KEYS["B"][mode]
            if any(row["scores"][key_a] != row["scores"][key_b] for row in rows):
                raise A4Error(f"Stage B A/B {mode} scores are not identical for {category}")
            if arm_metrics["A"][mode]["auroc"] != arm_metrics["B"][mode]["auroc"] or arm_metrics["A"][mode]["average_precision"] != arm_metrics["B"][mode]["average_precision"]:
                raise A4Error(f"Stage B A/B {mode} AUROC/AP are not identical for {category}")
        # Actual feedback and writer counts are derived from the sealed private ledger.
        feedback_rows = _load_feedback(source / "private_feedback.jsonl")
        queried_normal = sum(int(x["label"] == 0) for x in feedback_rows)
        queried_defect = sum(int(x["label"] == 1) for x in feedback_rows)
        writer_calls = 0
        native_write_actions = 0
        candidate_patches = accepted = state_changes = 0
        for feedback in feedback_rows:
            for action in feedback.get("writer_actions", {}).values():
                action_detail = action.get("action", {})
                if isinstance(action_detail, Mapping) and action_detail.get("action") == "native_greedy_normal_update":
                    native_write_actions += 1
                candidate_patches += int(action.get("candidate_patch_count", 0))
                accepted += int(action.get("accepted_candidate_patches", 0))
                state_changes += int(action.get("state_changed", False))
        writer_calls = sum(3 for feedback in feedback_rows if feedback["label"] == 0)
        all_metrics[category] = {
            "arm_metrics": arm_metrics, "windows": windows_by_arm,
            "contrasts_online": {
                "calibration_B_minus_A": _subtract(arm_metrics["B"]["online"], arm_metrics["A"]["online"]),
                "selector_C_minus_B": _subtract(arm_metrics["C"]["online"], arm_metrics["B"]["online"]),
                "reference_D_minus_C": _subtract(arm_metrics["D"]["online"], arm_metrics["C"]["online"]),
                "budget_D_minus_A": _subtract(arm_metrics["D"]["online"], arm_metrics["A"]["online"]),
            },
            "query_count": int(seal["query_count"]), "query_rate": float(seal["query_count"] / len(rows)),
            "queried_normal": queried_normal, "queried_defect": queried_defect,
            "writer_calls": writer_calls, "candidate_patches": candidate_patches,
            "accepted_candidate_patches": accepted, "reported_state_change_actions": state_changes,
            "native_normal_write_actions": native_write_actions,
            "label_projection_sha256": projection_sha,
        }
    # Store public-compatible metrics in protected run output, labels stay in private projection files.
    metrics_payload = {"schema": "A4_EB68_v1.stage_b_metrics_private.v1",
        "run_id": run_id, "category_equal_macro": _macro_metrics(all_metrics),
        "categories": all_metrics, "label_projection_receipts": projection_receipts,
        "all_stage_a_seals_validated_before_label_load": True,
        "evaluator_cpu_only": True, "masks_opened": False,
        "created_utc": global_seal["created_utc"]}
    metrics_tmp = attempt / "stage_b_metrics_private.json"
    metrics_tmp.write_bytes(_json_bytes(metrics_payload) + b"\n")
    _fsync_file(metrics_tmp)
    # Publish verified Stage B files once, without replacing any pre-existing file.
    for cat in CATEGORIES:
        dest = run_root / "stage_b" / cat / "label_projection_private.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            temp_projection = attempt / f"{cat}_label_projection_private.jsonl"
            if _sha_file(dest) != _sha_file(temp_projection):
                raise A4Error(f"Existing Stage B projection differs from verified retry: {dest}")
            temp_projection.unlink()
        else:
            os.replace(attempt / f"{cat}_label_projection_private.jsonl", dest)
    metrics_path = run_root / "stage_b_metrics_private.json"
    if metrics_path.exists():
        if _sha_file(metrics_path) != _sha_file(metrics_tmp):
            raise A4Error("Existing Stage B metrics differ from verified retry")
        metrics_tmp.unlink()
    else:
        os.replace(metrics_tmp, metrics_path)
    seal_payload = {"schema": "A4_EB68_v1.stage_b_seal.v1", "status": "SEALED_COMPLETE",
        "run_id": run_id,
        "stage_a_global_seal_sha256": _sha_file(global_seal_path),
        "stage_a_category_seal_sha256": global_seal["category_seal_sha256"],
        "metrics_sha256": _sha_file(metrics_path),
        "label_projection_sha256": {cat: _sha_file(run_root / "stage_b" / cat / "label_projection_private.jsonl") for cat in CATEGORIES},
        "evaluator_cpu_only": True, "labels_read_after_global_stage_a_seal": True,
        "masks_opened": False, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    seal_path = run_root / "stage_b_seal.json"
    _write_json_once(seal_path, _json_bytes(seal_payload) + b"\n")
    return {"status": "SEALED_COMPLETE", "metrics_sha256": seal_payload["metrics_sha256"],
            "categories": list(CATEGORIES)}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.endswith("\n"):
                raise A4Error(f"Unsealed JSONL row {path}:{lineno}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise A4Error(f"Malformed JSONL row {path}:{lineno}")
            rows.append(row)
    return rows


def _load_feedback(path: Path) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    allowed = set()
    for row in rows:
        if (set(row) - {"schema", "category", "position", "original_json_position",
                        "event_id_sha256", "image_identity_sha256", "source_group_sha256",
                        "label", "score_row_fsynced_before_disclosure", "writer_actions",
                        "state_transitions", "threshold_refreshes"} or
                row.get("label") not in (0, 1) or row.get("score_row_fsynced_before_disclosure") is not True):
            raise A4Error("Queried feedback ledger has malformed or extra label fields")
        if row["position"] in allowed:
            raise A4Error("Duplicate queried feedback row")
        allowed.add(row["position"])
        if row["label"] == 1 and any(
                action.get("action", {}).get("action") == "native_greedy_normal_update"
                for action in row.get("writer_actions", {}).values()):
            raise A4Error("Queried defect feedback entered normal Greedy memory")
    return rows


def _stage_b_vectors(rows: Sequence[Mapping[str, Any]], arm: str,
                     mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project one explicit sealed arm/mode key; never infer from key suffixes."""
    if arm not in ARMS or mode not in ("static", "online"):
        raise A4Error("Stage B requested an unregistered arm or scoring mode")
    key = SCORE_KEYS[arm][mode]
    try:
        scores = np.asarray([row["scores"][key] for row in rows], dtype=np.float64)
        predictions = np.asarray([row["predictions"][key] for row in rows], dtype=np.int8)
        thresholds = np.asarray([row["thresholds"][key] for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise A4Error(f"Stage B sealed vector is missing explicit key {key}") from exc
    if (scores.ndim != 1 or predictions.shape != scores.shape or
            thresholds.shape != scores.shape or not np.isfinite(scores).all() or
            not np.isfinite(thresholds).all() or
            not set(np.unique(predictions)) <= {0, 1}):
        raise A4Error(f"Stage B explicit vector is malformed for {key}")
    return scores, predictions, thresholds


def _subtract(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> dict[str, float | None]:
    keys = ("auroc", "average_precision", "ber", "fpr", "fnr", "balanced_accuracy")
    return {key: (float(lhs[key]) - float(rhs[key])
                  if lhs.get(key) is not None and rhs.get(key) is not None else None)
            for key in keys}


def _macro_metrics(categories: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        result[arm] = {}
        for mode in ("static", "online"):
            result[arm][mode] = {}
            for metric in ("auroc", "average_precision", "ber", "fpr", "fnr", "balanced_accuracy"):
                values = [categories[c]["arm_metrics"][arm][mode][metric]
                          for c in CATEGORIES if categories[c]["arm_metrics"][arm][mode][metric] is not None]
                result[arm][mode][metric] = float(np.mean(values)) if values else None
    result["contrasts_online"] = {}
    for contrast in ("calibration_B_minus_A", "selector_C_minus_B",
                     "reference_D_minus_C", "budget_D_minus_A"):
        result["contrasts_online"][contrast] = {}
        for metric in ("auroc", "average_precision", "ber", "fpr", "fnr", "balanced_accuracy"):
            vals = [categories[c]["contrasts_online"][contrast][metric] for c in CATEGORIES]
            result["contrasts_online"][contrast][metric] = float(np.mean(vals))
    return result


def _dispatch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Any:
    if args.preflight:
        return preflight(check_source_freeze=False)
    if args.self_test:
        return _run_engineering_tests()
    if not args.run_id:
        parser.error("--run-id is required for this mode")
    run_id = _validate_run_id(args.run_id)
    if args.run_all:
        return _run_all(run_id)
    if args.verify_only:
        from experiments.calibration.a4_independent_verifier import verify_run
        return verify_run(run_id)
    if args.report_only:
        from experiments.calibration.a4_independent_verifier import verify_run
        from experiments.calibration.a4_report_renderer import render_run
        verified = verify_run(run_id)
        if verified.get("status") != "PASS":
            raise A4Error("Report-only reconstruction requires independent verification PASS")
        return render_run(run_id)
    if args.smoke:
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        receipt = preflight(check_source_freeze=True)
        run_root = RUN_ROOT / run_id
        if not run_root.exists():
            run_root.mkdir(parents=False, exist_ok=False)
            _write_json_once(run_root / "run_manifest.json", _json_bytes(
                _source_freeze_manifest(run_id, receipt)) + b"\n")
            _write_json_once(run_root / "RUN_STATE.json", _json_bytes({
                "run_id": run_id, "status": "RUNNING", "campaign_started_epoch": time.time()}) + b"\n")
        return _run_child(run_id, stage="SMOKE", category="audiojack", smoke=True,
                          timeout_seconds=900)
    raise A4Error("No supported mode selected")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or independently verify A4_EB68_v1")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--run-all", action="store_true")
    group.add_argument("--verify-only", action="store_true")
    group.add_argument("--report-only", action="store_true")
    group.add_argument("--category-child", action="store_true", help=argparse.SUPPRESS)
    group.add_argument("--stage-b-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-id")
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--smoke-max-events", type=int, default=128)
    args = parser.parse_args(argv)
    try:
        if args.category_child:
            if not args.run_id or not args.category:
                parser.error("Category child requires --run-id and --category")
            result = _run_category_child(_validate_run_id(args.run_id), args.category,
                                         smoke=args.smoke_child,
                                         smoke_max_events=args.smoke_max_events)
        elif args.stage_b_child:
            if not args.run_id:
                parser.error("Stage B child requires --run-id")
            result = _stage_b_child(_validate_run_id(args.run_id))
        else:
            result = _dispatch(args, parser)
    except BaseException as exc:
        if args.run_id and args.category_child:
            try:
                path = RUN_ROOT / args.run_id / ("smoke" if args.smoke_child else "stage_a") / str(args.category) / "worker_failure.json"
                if not path.exists():
                    _write_json_once(path, _json_bytes({"error_type": type(exc).__name__,
                        "error": str(exc), "traceback": traceback.format_exc()[-14000:],
                        "status": "INCOMPLETE"}) + b"\n")
            except BaseException:
                pass
        raise
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
