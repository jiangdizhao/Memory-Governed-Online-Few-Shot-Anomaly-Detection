"""Retrospective cached-data attribution of normal/anomalous C1 feedback.

This runner never extracts image features. Stage A replays the sealed EXP2A
Top-32 writer inputs on CPU and seals scores before a separate Stage B child
projects the authorized first-512 outcomes. The outer process uses the existing
Windows Job Object supervisor for one 600-second process-tree deadline.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PYTHON = Path(r"D:\anaconda3\envs\torch\python.exe")
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
EXPECTED_BASELINE = "cae6cb14822c9e41d697f7cc3d086e717d68b071"
EXP2A_REL = Path("outputs/calibration/r1_digest_contract_v1_exp2a_r2exp2a_20261101_a4")
EXP2B_REL = Path("outputs/calibration/r1_digest_contract_v1_exp2b_r2exp2b_20261101_b1")
OUTPUT_REL_PARENT = Path("outputs/calibration")
DEFAULT_RUN_ID = "r2_feedback_repair_20261102_a6"
AUTHORIZATION_ID = "user-task:R2-FEEDBACK-REPAIR-1:2026-11-02"
CATEGORIES = ("switch", "audiojack", "regulator", "rolled_strip_base")
MAIN_ARMS = ("G", "G_N", "G_A", "G_NA", "G_GLOBAL")
SECONDARY_ARMS = ("C1_NO_BOTH", "C2_A002")
ALL_OUTPUTS = frozenset({
    "budget.lock", "run_manifest.json", "attempt_history.jsonl",
    "stage_a_scores.jsonl", "stage_a_seal.json", "stage_b_seal.json",
    "stage_a_reuse_receipt.json",
    "aggregate_results.json", "complete_seal.json",
    "supervisor_status.json", "failure.json",
})
STAGE_A_EXTERNAL_SOURCE_PATHS = (
    "experiments/calibration/r1_matched_correction_pilot.py",
    "experiments/calibration/r2_exp2a_c1_strength.py",
    "experiments/calibration/r2_exp2b_memory_readout.py",
    "test_phase3_AIF_simplified_v22.py",
    "vmb_visionad_new.py",
)
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
FAILURE_RESERVE_BYTES = 64 * 1024
MAX_OUTPUT_FILES = 20
SUPERVISED_DEADLINE_SECONDS = 600
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 20261101
ATOL = 1e-5
RTOL = 1e-4
STAGE_A_SCHEMA = "r2.feedback_repair.stage_a_scores.v1"
STAGE_A_SEAL_SCHEMA = "r2.feedback_repair.stage_a_seal.v1"
STAGE_B_SEAL_SCHEMA = "r2.feedback_repair.stage_b_seal.v1"
MANIFEST_SCHEMA = "r2.feedback_repair.run_manifest.v1"


class RepairError(RuntimeError):
    pass


def _cuda_context_is_hidden(torch_module: Any) -> bool:
    """Whether no CUDA device is visible and no CUDA context was initialized."""
    return (torch_module.cuda.device_count() == 0 and
            not torch_module.cuda.is_initialized())


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _is_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        import stat
        return bool(path.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    except (AttributeError, OSError):
        return False


def _reject_reparse_chain(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    for current in (absolute, *absolute.parents):
        if current.exists() and _is_reparse(current):
            raise RepairError(f"Symlink/reparse path is not permitted: {current}")


def _safe_exact_file(root: Path, relative: str | Path) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or any(part in ("..", "") for part in rel.parts):
        raise RepairError("Input file must be a safe relative path")
    _reject_reparse_chain(root)
    base = root.resolve(strict=True)
    current = root
    for part in rel.parts:
        current = current / part
        if _is_reparse(current):
            raise RepairError(f"Reparse point in exact input path: {rel.as_posix()}")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(base) or not resolved.is_file():
        raise RepairError(f"Exact input path is missing or escaped its root: {rel.as_posix()}")
    return resolved


class OutputBudget:
    """Small, process-safe pre-write budget for the flat, allowlisted run dir."""

    def __init__(self, root: Path, *, create: bool = False,
                 max_bytes: int = MAX_OUTPUT_BYTES,
                 failure_reserve_bytes: int = FAILURE_RESERVE_BYTES,
                 max_files: int = MAX_OUTPUT_FILES):
        self.root = Path(os.path.abspath(root))
        self.parent = self.root.parent
        self.max_bytes = int(max_bytes)
        self.failure_reserve_bytes = int(failure_reserve_bytes)
        self.max_files = int(max_files)
        if (self.max_bytes <= 0 or self.failure_reserve_bytes < 0 or
                self.failure_reserve_bytes >= self.max_bytes or self.max_files < 2):
            raise RepairError("Invalid output budget configuration")
        self._thread_lock = threading.RLock()
        self.created: set[str] = set()
        self.file_sizes: dict[str, int] = {}
        self.used_bytes = 0
        if create:
            _reject_reparse_chain(self.parent)
            if self.root.parent.resolve(strict=True) != self.parent.resolve(strict=True):
                raise RepairError("Output parent identity changed")
            self.root.mkdir(parents=False, exist_ok=False)
            lock_path = self.root / "budget.lock"
            with lock_path.open("xb") as stream:
                stream.write(b"\0")
                stream.flush()
                os.fsync(stream.fileno())
        self._validate_root()
        self.created = set()
        self.file_sizes = {}
        self.used_bytes = 0
        self._refresh_inventory()

    def _validate_root(self) -> None:
        _reject_reparse_chain(self.parent)
        if not self.root.is_dir() or _is_reparse(self.root):
            raise RepairError("Run output directory is missing or unsafe")
        if self.root.resolve(strict=True).parent != self.parent.resolve(strict=True):
            raise RepairError("Output directory escaped the authorized parent")

    @contextmanager
    def _process_lock(self):
        self._validate_root()
        lock_path = self.root / "budget.lock"
        if _is_reparse(lock_path) or not lock_path.is_file():
            raise RepairError("Budget lock is missing or unsafe")
        with lock_path.open("r+b", buffering=0) as stream:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise RepairError("Concurrent output writer holds the run lock") from exc
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

    def _inventory(self) -> tuple[dict[str, int], int]:
        sizes: dict[str, int] = {}
        used = 0
        for path in self.root.iterdir():
            if _is_reparse(path) or not path.is_file() or path.name not in ALL_OUTPUTS:
                raise RepairError(f"Unaccounted output path: {path.name}")
            size = path.stat().st_size
            sizes[path.name] = size
            used += size
        if len(sizes) > self.max_files or used > self.max_bytes:
            raise RepairError("Existing output exceeds its file or byte budget")
        return sizes, used

    def _refresh_inventory(self, *, allowed_change: str | None = None) -> None:
        sizes, used = self._inventory()
        names = set(sizes)
        if self.created:
            changed_names = names.symmetric_difference(self.created)
            changed_sizes = {
                name for name in names.intersection(self.created)
                if sizes[name] != self.file_sizes[name]
            }
            if (changed_names or changed_sizes) and (
                    not allowed_change or
                    changed_names - {allowed_change} or
                    changed_sizes - {allowed_change}):
                raise RepairError("Concurrent or unaccounted output mutation detected")
        self.created, self.used_bytes = names, used
        self.file_sizes = sizes

    def _prewrite(self, name: str, size: int, *, new: bool, failure_reserved: bool) -> Path:
        if name not in ALL_OUTPUTS or name == "budget.lock":
            raise RepairError("Output filename is not in the audited allowlist")
        if failure_reserved and name != "failure.json":
            raise RepairError("Only failure.json may consume reserved output capacity")
        path = self.root / name
        if _is_reparse(path) or path.parent.resolve(strict=True) != self.root.resolve(strict=True):
            raise RepairError("Output path escaped its unique run directory")
        if new and (path.exists() or name in self.created):
            raise RepairError(f"Refusing to overwrite existing output: {name}")
        if not new and (name not in self.created or not path.is_file()):
            raise RepairError(f"Append requires registered output: {name}")
        file_ceiling = self.max_files if failure_reserved else self.max_files - 1
        if new and len(self.created) + 1 > file_ceiling:
            raise RepairError("Output file budget exhausted before write")
        byte_ceiling = self.max_bytes if failure_reserved else self.max_bytes - self.failure_reserve_bytes
        if self.used_bytes + size > byte_ceiling:
            raise RepairError("Output byte budget exhausted before write")
        return path

    def _write(self, name: str, payload: bytes, *, new: bool | None,
               failure_reserved: bool = False) -> None:
        data = bytes(payload)
        with self._thread_lock, self._process_lock():
            self._refresh_inventory()
            actual_new = (name not in self.created) if new is None else new
            path = self._prewrite(name, len(data), new=actual_new,
                                  failure_reserved=failure_reserved)
            try:
                mode = "xb" if actual_new else "ab"
                with path.open(mode) as stream:
                    count = stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                if count != len(data):
                    raise OSError("Short output write")
            except BaseException:
                # A failed filesystem write may leave a prefix. Account that
                # prefix before preserving failure metadata from the reserve.
                self._refresh_inventory(allowed_change=name)
                raise
            sizes, used = self._inventory()
            if name not in sizes or used > self.max_bytes:
                raise RepairError("Post-write verification failed")
            self.created, self.used_bytes = set(sizes), used
            self.file_sizes = sizes

    def write_once(self, name: str, payload: bytes, *, failure_reserved: bool = False) -> None:
        self._write(name, payload, new=True, failure_reserved=failure_reserved)

    def append(self, name: str, payload: bytes) -> None:
        self._write(name, payload, new=False)

    def append_jsonl(self, name: str, record: Mapping[str, Any]) -> None:
        # First row creates a file with exclusive creation. Later rows append;
        # both operations share the same lock and pre-write budget check.
        self._write(name, _canonical_bytes(dict(record)), new=None)

    def write_json(self, name: str, record: Mapping[str, Any], *,
                   failure_reserved: bool = False) -> None:
        self.write_once(name, _canonical_bytes(dict(record)),
                        failure_reserved=failure_reserved)

    def audit(self) -> dict[str, Any]:
        with self._thread_lock, self._process_lock():
            self._refresh_inventory()
            return {"file_count": len(self.created), "bytes": self.used_bytes,
                    "budget_bytes": self.max_bytes,
                    "failure_reserve_bytes": self.failure_reserve_bytes,
                    "files": sorted(self.created)}


class CausalJsonlReader:
    """One-byte-at-a-time JSONL reader; feedback cannot be read ahead."""

    def __init__(self, path: Path, description: str):
        self.path = path
        self.description = description
        self.fd: int | None = None
        self.digest = hashlib.sha256()
        self.count = 0

    def __enter__(self) -> "CausalJsonlReader":
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def read_record(self) -> dict[str, Any]:
        if self.fd is None:
            self.fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        line = bytearray()
        while True:
            byte = os.read(self.fd, 1)
            if not byte:
                if not line:
                    raise RepairError(f"Unexpected EOF in {self.description}")
                break
            line.extend(byte)
            if byte == b"\n":
                break
            if len(line) > 1024 * 1024:
                raise RepairError(f"Oversized row in {self.description}")
        self.digest.update(line)
        self.count += 1
        try:
            value = json.loads(line)
        except Exception as exc:
            raise RepairError(f"Malformed row in {self.description}") from exc
        if not isinstance(value, dict):
            raise RepairError(f"Expected JSON object in {self.description}")
        return value

    def finish(self, *, expected_count: int, expected_sha256: str) -> None:
        if self.fd is None:
            self.fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        if os.read(self.fd, 1):
            raise RepairError(f"Unexpected extra records in {self.description}")
        if self.count != expected_count or self.digest.hexdigest() != expected_sha256:
            raise RepairError(f"Sealed row count or digest mismatch: {self.description}")


@dataclass
class CorrectionBank:
    name: str
    cfg: Any
    codebook: Any
    source_groups: dict[int, set[str]] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=lambda: {
        "inserted": 0, "merged": 0, "replaced": 0, "skipped": 0,
        "same_sign_replaced": 0, "cross_sign_replaced": 0,
        "same_sign_merged": 0,
    })
    max_occupancy: int = 0
    occupancy_total: int = 0
    occupancy_steps: int = 0
    excluded_slots_total: int = 0
    eligible_slots_total: int = 0

    @property
    def occupancy(self) -> int:
        return int(self.codebook.size())

    def observe_occupancy(self) -> None:
        n = self.occupancy
        self.max_occupancy = max(self.max_occupancy, n)
        self.occupancy_total += n
        self.occupancy_steps += 1

    def note_post_update_occupancy(self) -> None:
        self.max_occupancy = max(self.max_occupancy, self.occupancy)


def _git(*args: str) -> str:
    import subprocess
    result = subprocess.run(["git", *args], cwd=ROOT, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8")
    return result.stdout.strip()


def _source_fingerprints() -> dict[str, str]:
    paths = (
        "experiments/calibration/r2_feedback_repair.py",
        "experiments/calibration/r2_exp2a_c1_strength.py",
        "experiments/calibration/r2_exp2b_memory_readout.py",
        "experiments/calibration/r2_sgrm_quick_eval.py",
        "experiments/calibration/r1_matched_correction_pilot.py",
        "experiments/calibration/r1_runtime_supervisor.py",
        "experiments/calibration/r2_perf1_stage_b.py",
        "tests/calibration/test_r2_feedback_repair.py",
        "test_phase3_AIF_simplified_v22.py",
        "vmb_visionad_new.py",
    )
    return {rel: _sha256_file(ROOT / rel) for rel in paths}


def _stage_a_reuse_source_mismatches(
        source_hashes: Mapping[str, str], current_hashes: Mapping[str, str]) -> dict[str, dict[str, str | None]]:
    """Compare only immutable native Stage A dependencies for sealed reuse.

    Stage A's own scoring/update implementation is separately bound by its
    algorithm digest. Stage B-only metric/report code may change without
    forcing a second score replay.
    """
    mismatches: dict[str, dict[str, str | None]] = {}
    for key in STAGE_A_EXTERNAL_SOURCE_PATHS:
        expected = source_hashes.get(key)
        actual = current_hashes.get(key)
        if not expected or expected != actual:
            mismatches[key] = {"expected": expected, "actual": actual}
    return mismatches


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RepairError(f"Expected JSON object: {path.name}")
    return value


def _validate_inputs_and_identity() -> dict[str, Any]:
    if os.name != "nt":
        raise RepairError("The authorized runtime requires the existing Windows Job Object supervisor")
    if sys.executable.lower() != str(PYTHON).lower():
        raise RepairError(f"Unexpected Python interpreter: {sys.executable}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RepairError("CUDA_VISIBLE_DEVICES must be empty before Torch import")
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        raise RepairError(f"Wrong branch: {branch}")
    _git("merge-base", "--is-ancestor", EXPECTED_BASELINE, head)
    exp2a_root = ROOT / EXP2A_REL
    exp2b_root = ROOT / EXP2B_REL
    manifest = _load_json(_safe_exact_file(exp2a_root, "run_manifest.json"))
    stage_a = _load_json(_safe_exact_file(exp2a_root, "stage_a_seal.json"))
    completion = _load_json(_safe_exact_file(exp2a_root, "experiment_seal.json"))
    exp2b_manifest = _load_json(_safe_exact_file(exp2b_root, "run_manifest.json"))
    exp2b_seal = _load_json(_safe_exact_file(exp2b_root, "stage_a_seal.json"))
    if (manifest.get("run_id") != "r2exp2a_20261101_a4" or
            manifest.get("phase") != "STAGE_A" or
            manifest.get("experiment_profile_id") != "r2.exp2a.c1_strength.factorial.v1"):
        raise RepairError("EXP2A manifest identity/profile is incompatible")
    if stage_a.get("status") != "SEALED_COMPLETE" or stage_a.get("total_events") != 2048:
        raise RepairError("EXP2A Stage A is not a complete 2,048-event sealed trajectory")
    if completion.get("status") != "COMPLETE" or completion.get("stage_a_status") != "SEALED_COMPLETE":
        raise RepairError("EXP2A completion record does not certify Stage A")
    stage_a_bytes = _safe_exact_file(exp2a_root, "stage_a_seal.json").read_bytes()
    completion_bytes = _safe_exact_file(exp2a_root, "experiment_seal.json").read_bytes()
    if completion.get("stage_a_seal_sha256") != _sha256_bytes(stage_a_bytes):
        raise RepairError("EXP2A completion record does not bind the Stage A seal")
    if (exp2b_seal.get("status") != "SEALED_COMPLETE" or
            exp2b_seal.get("run_id") != "r2exp2b_20261101_b1" or
            exp2b_seal.get("input_run_id") != manifest.get("run_id")):
        raise RepairError("EXP2B Stage A reference receipt is missing or incompatible")
    current_hashes = _source_fingerprints()
    old_hashes = manifest.get("source_hashes", {})
    for rel in ("experiments/calibration/r1_matched_correction_pilot.py",
                "test_phase3_AIF_simplified_v22.py", "vmb_visionad_new.py",
                "experiments/calibration/r1_runtime_supervisor.py"):
        if old_hashes.get(rel) != current_hashes[rel]:
            raise RepairError(f"Frozen replay source differs from EXP2A receipt: {rel}")
    expected_exp2a_runner_sha = manifest.get("r2_exp2a_source_sha256")
    actual_exp2a_runner_sha = current_hashes[
        "experiments/calibration/r2_exp2a_c1_strength.py"]
    # EXP2B's sealed input receipt independently binds the exact EXP2A artifacts.
    expected_seal_links = {
        "input_experiment_seal_sha256": _sha256_bytes(completion_bytes),
        "input_stage_a_seal_sha256": _sha256_bytes(stage_a_bytes),
        "input_score_file_sha256": stage_a.get("score_rows_sha256"),
        "input_transition_file_sha256": stage_a.get("transition_rows_sha256"),
        "input_strength_diagnostics_sha256": stage_a.get("strength_diagnostics_sha256"),
        "input_queried_feedback_sha256": stage_a.get("queried_feedback_sha256"),
    }
    for key, expected in expected_seal_links.items():
        if exp2b_seal.get(key) != expected:
            raise RepairError(f"EXP2B receipt does not match EXP2A sealed input: {key}")
    if exp2b_manifest.get("run_id") != "r2exp2b_20261101_b1":
        raise RepairError("EXP2B reference manifest identity is incompatible")
    return {
        "branch": branch, "head": head, "expected_baseline_ancestor": EXPECTED_BASELINE,
        "authorization_id": AUTHORIZATION_ID,
        "exp2a_run_id": manifest["run_id"],
        "exp2a_manifest_status": manifest.get("status"),
        "exp2a_stage_a_status": stage_a["status"],
        "exp2a_completion_status": completion["status"],
        "exp2a_experiment_seal_sha256": _sha256_bytes(completion_bytes),
        "exp2a_stage_a_seal_sha256": _sha256_bytes(stage_a_bytes),
        "exp2a_stage_a_input_hashes": {
            key: stage_a[key] for key in (
                "score_rows_sha256", "transition_rows_sha256",
                "strength_diagnostics_sha256", "queried_feedback_sha256")},
        "exp2b_run_id": exp2b_manifest["run_id"],
        "exp2b_stage_a_status": exp2b_seal["status"],
        "exp2b_input_receipt_match": True,
        "exp2a_source_commit": manifest.get("identity", {}).get("parent_commit"),
        "current_source_hashes": current_hashes,
        "source_fingerprints_match_for_replay": True,
        "historical_exp2a_runner_sha256_expected": expected_exp2a_runner_sha,
        "historical_exp2a_runner_sha256_actual": actual_exp2a_runner_sha,
        "historical_exp2a_runner_sha256_match": expected_exp2a_runner_sha == actual_exp2a_runner_sha,
        "runtime_profile": {
            "device": "cpu", "cuda_visible_devices": "",
            "torch_threads": 1, "supervisor": "Windows Job Object",
            "deadline_seconds": SUPERVISED_DEADLINE_SECONDS,
        },
        "stage_a_seal": stage_a,
        "exp2a_manifest": manifest,
    }


def _make_bank(name: str, v22: Any, *, capacity: int = 256,
               feature_dim: int = 4096) -> CorrectionBank:
    import torch
    cfg = v22.CorrectionConfig()
    cfg.enabled = True
    codebook = v22.SignedCorrectionCodebook(
        K=capacity, feat_dim=feature_dim, device=torch.device("cpu"), cfg=cfg)
    if codebook.protos.device.type != "cpu" or codebook.protos.dtype != torch.float32:
        raise RepairError("Native SignedCorrectionCodebook is not CPU float32")
    return CorrectionBank(name, cfg, codebook)


def _gaussian_readout(distances: Any, beta: Any, radius: Any, kappa: Any,
                      *, components: bool = False):
    """Original Codebook.apply Gaussian equation over one computed distance matrix."""
    import torch
    beta = beta.to(device="cpu", dtype=torch.float32)
    radius = radius.to(device="cpu", dtype=torch.float32).clamp_min(1e-6)
    kappa = kappa.to(device="cpu", dtype=torch.float32).clamp_min(1e-6)
    weights = torch.exp(-(distances * distances) /
                        (2.0 * (radius[None, :] ** 2))) * kappa[None, :]
    denominator = weights.sum(dim=1) + 1e-6
    full = (weights * beta[None, :]).sum(dim=1) / denominator
    if not components:
        return full, None, None
    pos_beta = torch.where(beta > 0, beta, torch.zeros_like(beta))
    neg_beta = torch.where(beta < 0, beta, torch.zeros_like(beta))
    pos = (weights * pos_beta[None, :]).sum(dim=1) / denominator
    neg = (weights * neg_beta[None, :]).sum(dim=1) / denominator
    return full, pos, neg


def _score_bank(v22: Any, r1: Any, exp2b: Any, bank: CorrectionBank,
                features: Any, group: str, position: int, *,
                components: bool = False) -> tuple[Any, Any, dict[str, int]]:
    import torch
    query = v22.l2_normalize(features.to(device="cpu", dtype=torch.float32))
    cb = bank.codebook
    n = cb.size()
    eligible = r1._validate_c1_source_groups(
        bank.source_groups, cb.last_step[:n].detach().cpu().tolist(),
        event_position=position, phase="pre_score", event_group=group)
    excluded = n - len(eligible)
    bank.eligible_slots_total += len(eligible)
    bank.excluded_slots_total += excluded
    if not eligible:
        zero = torch.zeros((query.shape[0],), dtype=torch.float32, device="cpu")
        return zero, None, {"live": n, "eligible": 0, "same_source_excluded": excluded}
    slots = torch.as_tensor(eligible, dtype=torch.long, device="cpu")
    prototypes = cb.protos[slots]
    distances = exp2b._torch_distance_matrix(query, prototypes)
    delta, pos, neg = _gaussian_readout(
        distances, cb.beta[slots], cb.radius[slots], cb.kappa[slots],
        components=components)
    if components:
        if not torch.allclose(delta, pos + neg, atol=1e-7, rtol=1e-6):
            raise RepairError("Same-state positive/negative correction decomposition failed")
        return delta, (distances, pos, neg), {
            "live": n, "eligible": len(eligible), "same_source_excluded": excluded}
    return delta, (distances if bank.name == "G_NA" else None), {
        "live": n, "eligible": len(eligible), "same_source_excluded": excluded}


def _apply_feedback_update(bank: CorrectionBank, features: Any, keep_mask: Any,
                           compact_beta: Any, *, label: int, outcome: str,
                           position: int, group: str) -> dict[str, int] | None:
    """Apply one legal C1-Full writer input to an accepting independent bank."""
    import numpy as np
    import torch
    keep = np.asarray(keep_mask, dtype=np.bool_)
    beta = np.asarray(compact_beta, dtype=np.float32).reshape(-1)
    features_np = np.asarray(features, dtype=np.float32)
    if (keep.ndim != 1 or features_np.ndim != 2 or features_np.shape[0] != keep.size or
            beta.shape != (int(keep.sum()),) or not np.isfinite(beta).all() or
            not np.isfinite(features_np).all()):
        raise RepairError("C1-Full keep/features/compact-beta writer inputs are misaligned")
    accepts = (bank.name == "G_NA" or
               (bank.name == "G_N" and label == 0) or
               (bank.name == "G_A" and label == 1))
    if not accepts:
        return None
    if (label == 0 and np.any(beta > 1e-7)) or (label == 1 and np.any(beta < -1e-7)):
        raise RepairError("C1-Full beta signs conflict with the disclosed class")
    q = torch.from_numpy(np.ascontiguousarray(features_np[keep], dtype=np.float32))
    b = torch.from_numpy(np.ascontiguousarray(beta, dtype=np.float32))
    stats = bank.codebook.update(q, b, outcome=outcome, step=int(position))
    bank.stats = {key: bank.stats.get(key, 0) + int(stats.get(key, 0))
                  for key in bank.stats}
    r1 = __import__("experiments.calibration.r1_matched_correction_pilot",
                    fromlist=["_record_c1_source_group_updates"])
    r1._record_c1_source_group_updates(
        bank.source_groups,
        last_steps=bank.codebook.last_step[:bank.occupancy].detach().cpu().tolist(),
        event_position=position, event_group=group)
    return {key: int(stats.get(key, 0)) for key in bank.stats}


def _diagnostic_identity_matches(diagnostic: Mapping[str, Any], *,
                                category: str, position: int,
                                source_group: str) -> bool:
    """Bind an EXP2A update diagnostic to its event via frozen position/group.

    The original diagnostic schema omits event IDs; the separately sealed
    feedback record binds event ID, position and the same opaque group receipt.
    """
    try:
        diagnostic_position = int(diagnostic.get("position", -1))
    except (TypeError, ValueError):
        return False
    return (diagnostic.get("category") == category and
            diagnostic_position == position and
            diagnostic.get("source_group_sha256") == source_group)


def _apply_class_conditional_feedback(banks: Mapping[str, CorrectionBank],
                                     features: Any, keep_mask: Any, compact_beta: Any,
                                     *, label: int, outcome: str, position: int,
                                     group: str, queried: bool,
                                     score_sealed: bool) -> dict[str, dict[str, int] | None]:
    if not queried:
        return {name: None for name in banks}
    if score_sealed is not True:
        raise RepairError("Current feedback cannot update any bank before the score is durably sealed")
    if label not in (0, 1):
        raise RepairError("Feedback label must be binary")
    return {
        name: _apply_feedback_update(
            bank, features, keep_mask, compact_beta, label=label,
            outcome=outcome, position=position, group=group)
        for name, bank in banks.items()
    }


def _wants_update(bank_name: str, label: int) -> bool:
    return bank_name == "G_NA" or (bank_name == "G_N" and label == 0) or (
        bank_name == "G_A" and label == 1)


def _map_score(base_map: Any, top_ids: Any, delta: Any, r1: Any) -> float:
    return float(r1.max_score(r1.corrected_dense_map(base_map, top_ids, delta)))


def _global_delta(bank: CorrectionBank, eligible_slots: Sequence[int]) -> float:
    import torch
    if not eligible_slots:
        return 0.0
    slots = torch.as_tensor(eligible_slots, dtype=torch.long, device="cpu")
    beta = bank.codebook.beta[slots]
    kappa = bank.codebook.kappa[slots].clamp_min(1e-6)
    return float((kappa * beta).sum().item() / (1e-6 + kappa.sum().item()))


def _read_npz_once(path: Path, row: Mapping[str, Any], r1: Any) -> dict[str, Any]:
    import io
    import numpy as np
    if row.get("shard_path") != f"events/event_{int(row['event_number']):06d}.npz":
        raise RepairError("Event shard path is inconsistent with its sealed event number")
    payload = path.read_bytes()
    if _sha256_bytes(payload) != row.get("shard_sha256"):
        raise RepairError(f"Event shard SHA mismatch at event {row.get('event_number')}")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if r1.arrays_sha256(arrays) != row.get("arrays_sha256"):
        raise RepairError(f"Event shard array digest mismatch at event {row.get('event_number')}")
    return arrays


def _compare_float(actual: Any, expected: Any) -> tuple[float, int, bool]:
    import numpy as np
    a = np.asarray(actual, dtype=np.float32)
    b = np.asarray(expected, dtype=np.float32)
    if a.shape != b.shape:
        return math.inf, int(max(a.size, b.size)), False
    error = np.abs(a.astype(np.float64) - b.astype(np.float64))
    maximum = float(error.max()) if error.size else 0.0
    non_bitwise = int(np.count_nonzero(a != b))
    return maximum, non_bitwise, bool(np.allclose(a, b, atol=ATOL, rtol=RTOL))


def _stats_match(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    keys = ("inserted", "merged", "replaced", "skipped", "same_sign_replaced",
            "cross_sign_replaced", "same_sign_merged")
    return all(int(actual.get(key, 0)) == int(expected.get(key, 0)) for key in keys)


def _new_category_audit() -> dict[str, Any]:
    return {
        "events": 0, "queries": 0, "feedback_normal": 0, "feedback_anomalous": 0,
        "c1_full_delta_max_abs_error": 0.0, "c1_full_delta_non_bitwise": 0,
        "c1_full_score_max_abs_error": 0.0, "c1_full_score_non_bitwise": 0,
        "c1_full_delta_checks": 0, "c1_full_score_checks": 0,
        "numeric_mismatches": 0,
        "c0_score_checks": 0, "c0_map_mismatches": 0,
        "source_exclusion_checks": 0, "source_exclusion_mismatches": 0,
        "update_stat_checks": 0, "update_stat_mismatches": 0,
        "occupancy_checks": 0, "occupancy_mismatches": 0,
        "reference_state_digest_matches": 0, "reference_state_digest_mismatch": 0,
        "global_scalar_min": None, "global_scalar_max": None,
        "events_with_eligible_full_bank": 0,
        "events_without_eligible_full_bank": 0,
        "full_patch_positive_abs": [], "full_patch_negative_abs": [],
        "full_patch_abs": [], "score_pos_effect": [], "score_neg_effect": [],
        "score_full_effect": [], "decomposition_error_max": 0.0,
        "banks": {}, "parity_status": "UNVERIFIED",
    }


def _validate_exp2a_stage_manifest(profile: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = profile["exp2a_manifest"]
    seal = profile["stage_a_seal"]
    category_inputs = manifest.get("category_inputs", {})
    if set(category_inputs) != set(CATEGORIES):
        raise RepairError("EXP2A category inventory differs from the frozen four-category protocol")
    counts = seal.get("category_event_counts", {})
    if {name: int(counts.get(name, -1)) for name in CATEGORIES} != {name: 512 for name in CATEGORIES}:
        raise RepairError("EXP2A sealed category event counts differ from 512 each")
    contract = manifest.get("common_trajectory_contract", {})
    must_be_true = (
        "C0 map/query/prediction/common Greedy writer shared",
        "C0 prediction defines all arm outcomes",
        "C1 Codebook state independent by arm",
        "one feature extraction per event shared by all arms",
        "scores sealed before queried label disclosure",
    )
    must_be_false = ("masks accessed", "unqueried labels accessed")
    if (any(contract.get(key) is not True for key in must_be_true) or
            any(contract.get(key) is not False for key in must_be_false) or
            contract.get("development outcome evaluation") != "Stage B only after full score seal"):
        raise RepairError("EXP2A common-trajectory causal contract is not fully true")
    return manifest, seal


def _read_stagea_manifest(profile: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return _validate_exp2a_stage_manifest(profile)


def _run_stage_a(run_root: Path, budget: OutputBudget, profile: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r2_exp2a_c1_strength as exp2a
    from experiments.calibration import r2_exp2b_memory_readout as exp2b
    import test_phase3_AIF_simplified_v22 as v22

    torch.set_num_threads(1)
    if torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise RepairError("CPU-only boundary failed after hiding CUDA devices")
    manifest, source_seal = _read_stagea_manifest(profile)
    input_root = ROOT / EXP2A_REL
    paths = {name: _safe_exact_file(input_root, name) for name in (
        "event_scores.jsonl", "event_transitions.jsonl",
        "strength_diagnostics.jsonl", "private_query_feedback.jsonl")}
    expected_hashes = {
        "event_scores.jsonl": source_seal["score_rows_sha256"],
        "event_transitions.jsonl": source_seal["transition_rows_sha256"],
        "strength_diagnostics.jsonl": source_seal["strength_diagnostics_sha256"],
        "private_query_feedback.jsonl": source_seal["queried_feedback_sha256"],
    }
    banks: dict[str, CorrectionBank] = {}
    final_banks: dict[str, dict[str, CorrectionBank]] = {}
    category_audit = {category: _new_category_audit() for category in CATEGORIES}
    score_reader = CausalJsonlReader(paths["event_scores.jsonl"], "EXP2A event_scores.jsonl")
    transition_reader = CausalJsonlReader(paths["event_transitions.jsonl"], "EXP2A event_transitions.jsonl")
    diagnostic_reader = CausalJsonlReader(paths["strength_diagnostics.jsonl"], "EXP2A strength_diagnostics.jsonl")
    feedback_reader = CausalJsonlReader(paths["private_query_feedback.jsonl"], "EXP2A private_query_feedback.jsonl")
    source_rows: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORIES}
    correction_stats_mismatch_examples: list[dict[str, Any]] = []
    update_digest_expected: dict[str, str] = {category: "" for category in CATEGORIES}
    query_total = 0
    row_expected_count = 0
    started = time.monotonic()
    import hashlib as _hashlib
    input_readers = (score_reader, transition_reader, diagnostic_reader, feedback_reader)
    for reader in input_readers:
        reader.__enter__()
    try:
        for cat_index, category in enumerate(CATEGORIES):
            # Correction banks are category-local trajectories, initialized empty.
            banks = {name: _make_bank(name, v22) for name in ("G_N", "G_A", "G_NA")}
            theta = float(manifest["category_inputs"][category]["theta_c64_q90"])
            if int(manifest["category_inputs"][category]["c64_count"]) != 64:
                raise RepairError(f"{category}: frozen C64 count is not 64")
            for position in range(512):
                row = score_reader.read_record()
                row_expected_count += 1
                if (row.get("schema") != "r2.exp2a.score_row.v1" or
                        row.get("category") != category or int(row.get("position", -1)) != position or
                        int(row.get("event_number", -1)) != cat_index * 512 + position):
                    raise RepairError(f"EXP2A event score identity/order mismatch at {category}:{position}")
                if int(row.get("camera_id", -1)) < 0 or not re.fullmatch(
                        r"[0-9a-f]{64}", str(row.get("event_id", ""))):
                    raise RepairError("EXP2A event identity/camera metadata is malformed")
                for identity_key in ("source_identity_sha256", "class_json_sha256"):
                    if not re.fullmatch(r"[0-9a-f]{64}", str(row.get(identity_key, ""))):
                        raise RepairError(f"EXP2A event has malformed {identity_key}")
                if str(row.get("source_group_sha256", "")) == "" or not re.fullmatch(
                        r"[0-9a-f]{64}", str(row.get("source_group_sha256", ""))):
                    raise RepairError("EXP2A Source Group identity is malformed")
                if abs(float(row["theta_c64_q90"]) - theta) > 1e-12:
                    raise RepairError(f"{category}: frozen C64 threshold changed within the stream")
                if row.get("fixed_support_sha256") != manifest["category_inputs"][category][
                        "fixed_support_sha256_expected"]:
                    raise RepairError(f"{category}: fixed support identity differs from EXP2A manifest")
                shard = _safe_exact_file(input_root, row["shard_path"])
                arrays = _read_npz_once(shard, row, r1)
                required = {"base_patch_map", "top32_ids", "top32_features", "c1_delta_full",
                            "c1_delta_no_both", "c2_delta_a002"}
                if not required.issubset(arrays):
                    raise RepairError(f"{category}:{position}: event shard lacks required C0/C1 arrays")
                base_map = np.asarray(arrays["base_patch_map"], dtype=np.float32).reshape(-1)
                top_ids = np.asarray(arrays["top32_ids"], dtype=np.int64).reshape(-1)
                top_features = np.asarray(arrays["top32_features"], dtype=np.float32)
                if (base_map.shape != (784,) or top_ids.shape != (32,) or
                        len(set(top_ids.tolist())) != 32 or np.any(top_ids < 0) or
                        np.any(top_ids >= 784) or top_features.shape != (32, 4096) or
                        not np.isfinite(base_map).all() or not np.isfinite(top_features).all()):
                    raise RepairError(f"{category}:{position}: cached C0 map/Top-32 shape or values are invalid")
                if row.get("feature_dtype") != "torch.float32" or row.get("feature_shape") != [784, 4096]:
                    raise RepairError(f"{category}:{position}: frozen feature metadata changed")
                top_hash = exp2a.sha256_json(top_ids.tolist())
                if top_hash != row.get("top32_indices_sha256"):
                    raise RepairError(f"{category}:{position}: Top-32 identity digest differs")
                c0_score = _map_score(base_map, top_ids, np.zeros(32, dtype=np.float32), r1)
                stored_c0 = float(row["scores"]["c0"])
                c0_error, _, c0_ok = _compare_float([c0_score], [stored_c0])
                audit = category_audit[category]
                audit["c0_score_checks"] += 1
                if not c0_ok:
                    audit["c0_map_mismatches"] += 1
                    audit["numeric_mismatches"] += 1
                recorded_c0_prediction = int(row["predictions_at_frozen_c64_theta"]["c0"])
                if recorded_c0_prediction != int(c0_score >= theta):
                    audit["c0_map_mismatches"] += 1
                    audit["numeric_mismatches"] += 1
                group = str(row["source_group_sha256"])
                bank_details: dict[str, Any] = {}
                scored: dict[str, float] = {"G": c0_score}
                full_decomposition = None
                full_eligible: list[int] = []
                for name in ("G_N", "G_A", "G_NA"):
                    bank = banks[name]
                    before = bank.occupancy
                    bank.observe_occupancy()
                    delta, aux, detail = _score_bank(
                        v22, r1, exp2b, bank,
                        torch.from_numpy(np.ascontiguousarray(top_features)),
                        group, position, components=(name == "G_NA"))
                    score = _map_score(base_map, top_ids, delta.detach().cpu().numpy(), r1)
                    scored[name] = score
                    bank_details[name] = {**detail, "occupancy_before": before}
                    if name == "G_NA":
                        if aux is not None:
                            distances, pos_delta, neg_delta = aux
                            n = bank.occupancy
                            eligible = r1._validate_c1_source_groups(
                                bank.source_groups,
                                bank.codebook.last_step[:n].detach().cpu().tolist(),
                                event_position=position, phase="pre_score", event_group=group)
                            full_eligible = eligible
                            audit["events_with_eligible_full_bank"] += 1
                            err = float(torch.max(torch.abs(delta - (pos_delta + neg_delta))).item()) if delta.numel() else 0.0
                            audit["decomposition_error_max"] = max(audit["decomposition_error_max"], err)
                            full_decomposition = (pos_delta.detach().cpu().numpy(),
                                                  neg_delta.detach().cpu().numpy(),
                                                  delta.detach().cpu().numpy())
                            audit["full_patch_positive_abs"].extend(np.abs(full_decomposition[0]).tolist())
                            audit["full_patch_negative_abs"].extend(np.abs(full_decomposition[1]).tolist())
                            audit["full_patch_abs"].extend(np.abs(full_decomposition[2]).tolist())
                            p_score = _map_score(base_map, top_ids, full_decomposition[0], r1)
                            n_score = _map_score(base_map, top_ids, full_decomposition[1], r1)
                            audit["score_pos_effect"].append(p_score - c0_score)
                            audit["score_neg_effect"].append(n_score - c0_score)
                            audit["score_full_effect"].append(score - c0_score)
                        else:
                            full_eligible = []
                            audit["events_without_eligible_full_bank"] += 1
                            audit["score_pos_effect"].append(0.0)
                            audit["score_neg_effect"].append(0.0)
                            audit["score_full_effect"].append(score - c0_score)
                if scored["G_N"] > c0_score + 1e-6:
                    raise RepairError(f"{category}:{position}: G_N negative-only score exceeded G")
                if scored["G_A"] < c0_score - 1e-6:
                    raise RepairError(f"{category}:{position}: G_A positive-only score fell below G")
                global_scalar = _global_delta(banks["G_NA"], full_eligible)
                audit["global_scalar_min"] = global_scalar if audit["global_scalar_min"] is None else min(
                    float(audit["global_scalar_min"]), global_scalar)
                audit["global_scalar_max"] = global_scalar if audit["global_scalar_max"] is None else max(
                    float(audit["global_scalar_max"]), global_scalar)
                scored["G_GLOBAL"] = _map_score(
                    base_map, top_ids, np.full(32, global_scalar, dtype=np.float32), r1)
                scored["C1_NO_BOTH"] = float(row["scores"]["c1_no_both"])
                scored["C2_A002"] = float(row["scores"]["c2_a002"])
                audit["events"] += 1
                audit["c0_score_checks"] += 0
                audit["source_exclusion_checks"] += 1
                stored_detail = row.get("c1_source_exclusion", {}).get("c1_full", {})
                actual_detail = bank_details["G_NA"]
                if (int(stored_detail.get("eligible_anchor_slots", -1)) != actual_detail["eligible"] or
                        int(stored_detail.get("same_source_excluded_anchor_slots", -1)) !=
                        actual_detail["same_source_excluded"]):
                    audit["source_exclusion_mismatches"] += 1
                expected_delta = np.asarray(arrays["c1_delta_full"], dtype=np.float32)
                delta_actual = (np.zeros((32,), dtype=np.float32) if full_decomposition is None
                                else full_decomposition[2])
                delta_error, delta_nonbitwise, delta_ok = _compare_float(delta_actual, expected_delta)
                score_error, score_nonbitwise, score_ok = _compare_float(
                    [scored["G_NA"]], [float(row["scores"]["c1_full"])])
                audit["c1_full_delta_checks"] += 1
                audit["c1_full_score_checks"] += 1
                audit["c1_full_delta_max_abs_error"] = max(audit["c1_full_delta_max_abs_error"], delta_error)
                audit["c1_full_delta_non_bitwise"] += delta_nonbitwise
                audit["c1_full_score_max_abs_error"] = max(audit["c1_full_score_max_abs_error"], score_error)
                audit["c1_full_score_non_bitwise"] += score_nonbitwise
                if not delta_ok or not score_ok:
                    audit["numeric_mismatches"] += 1
                score_record = {
                    "schema": STAGE_A_SCHEMA, "category": category,
                    "position": position, "event_id": row["event_id"],
                    "source_identity_sha256": row["source_identity_sha256"],
                    "source_group_sha256": group,
                    "class_json_sha256": row["class_json_sha256"],
                    "query_tape_decision": bool(row["query_tape_decision"]),
                    "theta_c64_q90": theta, "scores": scored,
                    "pre_event_occupancy": {name: bank.occupancy for name, bank in banks.items()},
                }
                source_rows[category].append({
                    "category": category, "position": position,
                    "event_id": row["event_id"],
                    "source_identity_sha256": row["source_identity_sha256"],
                    "source_group_sha256": group,
                    "class_json_sha256": row["class_json_sha256"],
                    "query_tape_decision": bool(row["query_tape_decision"]),
                })
                # This durable write is the causal score seal for the current event.
                budget.append_jsonl("stage_a_scores.jsonl", score_record)
                is_query = bool(row["query_tape_decision"])
                if is_query:
                    feedback = feedback_reader.read_record()
                    diagrow = diagnostic_reader.read_record()
                    query_total += 1
                    if (feedback.get("schema") != "r2.exp2a.queried_feedback.v1" or
                            feedback.get("category") != category or
                            int(feedback.get("position", -1)) != position or
                            feedback.get("event_id") != row["event_id"] or
                            feedback.get("source_group_sha256") != group or
                            feedback.get("disclosure_after_score_seal") is not True):
                        raise RepairError(f"{category}:{position}: queried feedback identity/order mismatch")
                    # EXP2A diagnostics bind updates by category, event
                    # position and opaque Source Group receipt; they do not
                    # duplicate the event ID stored by the feedback record.
                    if not _diagnostic_identity_matches(
                            diagrow, category=category, position=position,
                            source_group=group):
                        raise RepairError(f"{category}:{position}: update diagnostic identity mismatch")
                    label = int(feedback["label"])
                    if label not in (0, 1):
                        raise RepairError("Queried feedback label is outside binary outcome domain")
                    common_prediction = int(c0_score >= theta)
                    outcome = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[
                        (label, common_prediction)]
                    if feedback.get("outcome_from_common_c0") != outcome:
                        raise RepairError("Original queried feedback outcome is inconsistent with sealed C0")
                    diagnostic = diagrow["arms"]["c1_full"]["diagnostic"]
                    expected_stats = diagrow["arms"]["c1_full"]["native_layer_stats"]
                    keep = diagnostic["native_gate_keep_mask"]
                    beta = diagnostic["beta_patch_float32_before_codebook_clip"]
                    if (diagnostic.get("diagnostic_trace_complete") is not True or
                            len(beta) != sum(bool(value) for value in keep)):
                        raise RepairError("Actual C1-Full compact beta shape does not match keep.sum()")
                    if int(diagnostic.get("c0_prediction", -1)) != common_prediction or diagnostic.get("outcome") != outcome:
                        raise RepairError("C1-Full diagnostic outcome differs from current disclosed C0 feedback")
                    if label == 0:
                        audit["feedback_normal"] += 1
                    else:
                        audit["feedback_anomalous"] += 1
                    audit["queries"] += 1
                    update_results = _apply_class_conditional_feedback(
                        banks, top_features, keep, beta, label=label,
                        outcome=outcome, position=position, group=group,
                        queried=True, score_sealed=True)
                    for name, bank in banks.items():
                        actual = update_results[name]
                        if actual is None:
                            continue
                        bank.note_post_update_occupancy()
                        if name == "G_NA":
                            audit["update_stat_checks"] += 1
                            if not _stats_match(actual or {}, expected_stats) or not _stats_match(
                                    actual or {}, diagnostic.get("native_update_stats", {})):
                                audit["update_stat_mismatches"] += 1
                                if len(correction_stats_mismatch_examples) < 8:
                                    correction_stats_mismatch_examples.append({
                                        "category": category, "position": position,
                                        "actual": actual, "expected": expected_stats})
                            expected_occupancy = int(diagnostic["anchor_occupancy_after"])
                            audit["occupancy_checks"] += 1
                            if bank.occupancy != expected_occupancy:
                                audit["occupancy_mismatches"] += 1
                            update_digest_expected[category] = str(
                                diagnostic.get("codebook_state_sha256_after", ""))
                transition = transition_reader.read_record()
                if (transition.get("category") != category or
                        int(transition.get("position", -1)) != position or
                        transition.get("event_id") != row["event_id"]):
                    raise RepairError(f"{category}:{position}: transition identity/order mismatch")
                occupancy_after = transition.get("c1_anchor_occupancy_after", {})
                if is_query:
                    expected_occ = int(occupancy_after.get("c1_full", -1))
                    if expected_occ != banks["G_NA"].occupancy:
                        audit["occupancy_mismatches"] += 1
                audit["banks"] = {
                    name: {"occupancy": bank.occupancy,
                           "max_occupancy": bank.max_occupancy,
                           "inserts": bank.stats["inserted"],
                           "merges": bank.stats["merged"],
                           "replacements": bank.stats["replaced"],
                           "skips": bank.stats["skipped"],
                           "eligible_slot_events": bank.eligible_slots_total,
                           "same_source_excluded_slots": bank.excluded_slots_total}
                    for name, bank in banks.items()}
            final_banks[category] = banks
        expected_queries = sum(int(manifest["category_inputs"][c]["query_positions_in_prefix"])
                               for c in CATEGORIES)
        if query_total != expected_queries or query_total != 67:
            raise RepairError(f"Query tape count mismatch: {query_total} vs {expected_queries}/67")
        score_reader.finish(expected_count=2048, expected_sha256=expected_hashes["event_scores.jsonl"])
        transition_reader.finish(expected_count=2048,
                                  expected_sha256=expected_hashes["event_transitions.jsonl"])
        diagnostic_reader.finish(expected_count=query_total,
                                 expected_sha256=expected_hashes["strength_diagnostics.jsonl"])
        feedback_reader.finish(expected_count=query_total,
                               expected_sha256=expected_hashes["private_query_feedback.jsonl"])
    finally:
        for reader in input_readers:
            reader.__exit__()
    final_state_digests: dict[str, Any] = {}
    for category in CATEGORIES:
        # One digest per category at the final state; no per-event full-bank hashing.
        bank = final_banks[category]["G_NA"]
        digest_bank = type("BankState", (), {
            "corr_cfg": bank.cfg,
            "corr_cb": {"p0": {"id": [bank.codebook]}},
        })()
        actual_digest = r1._correction_state_sha256(digest_bank, bank.source_groups)
        expected_digest = update_digest_expected[category]
        final_state_digests[category] = {
            "actual_cpu_sha256": actual_digest,
            "last_recorded_c1_full_sha256": expected_digest,
            "exact_match": bool(expected_digest and actual_digest == expected_digest),
            "interpretation": "diagnostic_only_across_cpu_gpu_update_paths",
        }
        audit = category_audit[category]
        audit["reference_state_digest_matches"] = int(final_state_digests[category]["exact_match"])
        audit["reference_state_digest_mismatch"] = int(not final_state_digests[category]["exact_match"])
        critical_mismatch = any(audit[key] for key in (
            "c0_map_mismatches", "source_exclusion_mismatches",
            "update_stat_mismatches", "occupancy_mismatches"))
        numeric_ok = audit["numeric_mismatches"] == 0
        audit["parity_status"] = "PASS" if not critical_mismatch and numeric_ok else "UNVERIFIED"
        positive = np.asarray(audit.pop("full_patch_positive_abs"), dtype=np.float64)
        negative = np.asarray(audit.pop("full_patch_negative_abs"), dtype=np.float64)
        full = np.asarray(audit.pop("full_patch_abs"), dtype=np.float64)
        denom = float(np.sum(positive + negative)) if positive.size and negative.size else 0.0
        cancellation = (1.0 - float(np.sum(full)) / denom) if denom > 0 else None
        audit["positive_negative_same_state_summary"] = {
            "positive_patch_abs": _percentile_summary(positive.tolist()),
            "negative_patch_abs": _percentile_summary(negative.tolist()),
            "full_patch_abs": _percentile_summary(full.tolist()),
            "positive_only_image_score_effect_vs_G": _percentile_summary(
                audit.pop("score_pos_effect")),
            "negative_only_image_score_effect_vs_G": _percentile_summary(
                audit.pop("score_neg_effect")),
            "full_image_score_effect_vs_G": _percentile_summary(
                audit.pop("score_full_effect")),
            "patch_level_cancellation_fraction": cancellation,
            "max_abs_decomposition_error": float(audit["decomposition_error_max"]),
            "events_with_eligible_full_bank": int(audit["events_with_eligible_full_bank"]),
            "events_without_eligible_full_bank": int(audit["events_without_eligible_full_bank"]),
            "denominator": "all eligible anchors, with the same full denominator for both signs",
            "image_score_effects_are_not_additive_after_dense_map_max": True,
        }
        for name, bank in banks.items():
            audit["banks"][name] = {
                "final_occupancy": bank.occupancy,
                "max_occupancy": bank.max_occupancy,
                "average_occupancy": bank.occupancy_total / max(1, bank.occupancy_steps),
                "update_counts": dict(bank.stats),
                "eligible_slot_events": bank.eligible_slots_total,
                "same_source_excluded_slot_events": bank.excluded_slots_total,
            }
    source_rows_count = sum(len(rows) for rows in source_rows.values())
    if source_rows_count != 2048:
        raise RepairError("Stage A source identity ledger is incomplete")
    output_score_path = run_root / "stage_a_scores.jsonl"
    stage_a_output_sha = _sha256_file(output_score_path)
    seal = {
        "schema": STAGE_A_SEAL_SCHEMA, "status": "SEALED_COMPLETE",
        "run_id": run_root.name, "event_count": 2048, "query_count": query_total,
        "stage_a_algorithm_sha256": _stage_a_algorithm_sha256(),
        "score_file": "stage_a_scores.jsonl", "score_file_sha256": stage_a_output_sha,
        "source_exp2a_seal_sha256": profile["exp2a_stage_a_seal_sha256"],
        "source_row_hashes_verified": True,
        "source_input_file_hashes_verified": True,
        "category_audit": category_audit,
        "final_c1_full_state_digests": final_state_digests,
        "score_tolerances": {"atol": ATOL, "rtol": RTOL},
        "event_order": "score_and_durable_append_before_current_queried_feedback_and_update",
        "feedback_input_access": "one queried record at its event after score append; no reads ahead",
        "offline_evaluator_accessed": False,
        "feature_extraction": False, "image_decoding": False, "masks_accessed": False,
        "device": "cpu", "elapsed_seconds": time.monotonic() - started,
        "c1_full_stat_mismatch_examples": correction_stats_mismatch_examples,
    }
    budget.write_json("stage_a_seal.json", seal)
    del source_rows
    return {"status": "SEALED_COMPLETE", "seal": seal}


def _stage_a_algorithm_sha256() -> str:
    import inspect
    functions = (OutputBudget, CausalJsonlReader, _gaussian_readout,
                 _score_bank, _apply_feedback_update,
                 _diagnostic_identity_matches,
                 _apply_class_conditional_feedback, CorrectionBank,
                 _global_delta, _map_score, _read_npz_once, _stats_match,
                 _new_category_audit, _validate_exp2a_stage_manifest,
                 _read_stagea_manifest, _run_stage_a)
    payload = "\n\n".join(inspect.getsource(function) for function in functions)
    return _sha256_bytes(payload.encode("utf-8"))


def _read_sealed_stage_a(run_root: Path, manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    seal_path = _safe_exact_file(run_root, "stage_a_seal.json")
    scores_path = _safe_exact_file(run_root, "stage_a_scores.jsonl")
    seal = _load_json(seal_path)
    if (seal.get("schema") != STAGE_A_SEAL_SCHEMA or
            seal.get("status") != "SEALED_COMPLETE" or
            int(seal.get("event_count", -1)) != 2048 or
            seal.get("score_file") != "stage_a_scores.jsonl" or
            seal.get("stage_a_algorithm_sha256") != manifest.get("stage_a_algorithm_sha256") or
            _sha256_file(scores_path) != seal.get("score_file_sha256")):
        raise RepairError("Stage B refused access: new Stage A scores are not completely sealed")
    source_run_id = str(manifest.get("stage_a_source_run_id", ""))
    if seal.get("run_id") != source_run_id:
        raise RepairError("Stage A seal identity does not match this run's reuse receipt")
    rows: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORIES}
    digest = hashlib.sha256()
    with scores_path.open("rb", buffering=0) as stream:
        for index in range(2048):
            line = stream.readline()
            if not line or not line.endswith(b"\n"):
                raise RepairError("Stage A score file is truncated or not newline sealed")
            digest.update(line)
            record = json.loads(line)
            forbidden = {"label", "outcome", "anomaly_class", "mask", "private_feedback",
                         "top32_features", "base_patch_map", "raw_feature", "feature_array"}
            if forbidden.intersection(record):
                raise RepairError("Stage A public score trajectory contains a forbidden evaluator or feature field")
            category = str(record.get("category", ""))
            position = int(record.get("position", -1))
            expected_category = CATEGORIES[index // 512]
            expected_position = index % 512
            if (record.get("schema") != STAGE_A_SCHEMA or category != expected_category or
                    position != expected_position):
                raise RepairError("Stage A output event ordering/schema is invalid")
            score_map = record.get("scores")
            if not isinstance(score_map, dict) or not set(MAIN_ARMS + SECONDARY_ARMS).issubset(score_map):
                raise RepairError("Stage A output does not contain all sealed score arms")
            if any(not math.isfinite(float(value)) for value in score_map.values()):
                raise RepairError("Stage A output contains a nonfinite score")
            rows.append(record)
            by_category[category].append(record)
        if stream.read(1):
            raise RepairError("Stage A score file contains unsealed trailing records")
    if digest.hexdigest() != seal.get("score_file_sha256"):
        raise RepairError("Stage A score JSONL digest changed after sealing")
    return rows, by_category


def _replay_bank_accounting_after_seal(
        rows: Sequence[Mapping[str, Any]], run_root: Path,
        manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Rebuild category-specific writer telemetry without rescoring events.

    Stage A seals all scores first. This Stage B-only pass reads query-event
    feature shards to reproduce the independent feedback-bank updates and
    checks per-category occupancy, update statistics and source exclusions.
    Nonquery feature shards are not reopened and no score is computed.
    """
    import numpy as np
    import torch
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r2_exp2a_c1_strength as exp2a
    import test_phase3_AIF_simplified_v22 as v22

    torch.set_num_threads(1)
    # PyTorch on this Windows build may report is_available()==True when
    # CUDA_VISIBLE_DEVICES hides every device. A zero visible count and no
    # initialized CUDA context are the relevant no-GPU invariants here.
    if not _cuda_context_is_hidden(torch):
        raise RepairError("Stage B bank audit crossed the CPU-only boundary")
    input_root = ROOT / EXP2A_REL
    paths = {name: _safe_exact_file(input_root, name) for name in (
        "event_scores.jsonl", "strength_diagnostics.jsonl",
        "private_query_feedback.jsonl")}
    input_hashes = manifest.get("input_seals", {}).get("exp2a_files", {})
    readers = {
        "scores": CausalJsonlReader(paths["event_scores.jsonl"], "EXP2A event_scores.jsonl"),
        "diagnostics": CausalJsonlReader(paths["strength_diagnostics.jsonl"], "EXP2A strength_diagnostics.jsonl"),
        "feedback": CausalJsonlReader(paths["private_query_feedback.jsonl"], "EXP2A private_query_feedback.jsonl"),
    }
    for reader in readers.values():
        reader.__enter__()
    result: dict[str, dict[str, Any]] = {}
    input_index = 0
    query_count = 0
    try:
        for category in CATEGORIES:
            banks = {name: _make_bank(name, v22) for name in ("G_N", "G_A", "G_NA")}
            query_count_category = 0
            gna_native_updates_match = True
            for position in range(512):
                source_row = readers["scores"].read_record()
                sealed_row = rows[input_index]
                input_index += 1
                group = str(source_row.get("source_group_sha256", ""))
                if (source_row.get("category") != category or
                        int(source_row.get("position", -1)) != position or
                        sealed_row.get("category") != category or
                        int(sealed_row.get("position", -1)) != position or
                        source_row.get("source_identity_sha256") != sealed_row.get("source_identity_sha256") or
                        source_row.get("class_json_sha256") != sealed_row.get("class_json_sha256") or
                        group != sealed_row.get("source_group_sha256") or
                        bool(source_row.get("query_tape_decision")) != bool(sealed_row.get("query_tape_decision"))):
                    raise RepairError(f"{category}:{position}: sealed-to-source identity mismatch in bank audit")
                for bank in banks.values():
                    bank.observe_occupancy()
                    live = bank.occupancy
                    eligible = r1._validate_c1_source_groups(
                        bank.source_groups,
                        bank.codebook.last_step[:live].detach().cpu().tolist(),
                        event_position=position, phase="pre_score", event_group=group)
                    bank.eligible_slots_total += len(eligible)
                    bank.excluded_slots_total += live - len(eligible)
                if not bool(source_row.get("query_tape_decision")):
                    continue
                feedback = readers["feedback"].read_record()
                diagnostic_row = readers["diagnostics"].read_record()
                query_count += 1
                query_count_category += 1
                if (feedback.get("schema") != "r2.exp2a.queried_feedback.v1" or
                        feedback.get("category") != category or
                        int(feedback.get("position", -1)) != position or
                        feedback.get("event_id") != source_row.get("event_id") or
                        feedback.get("source_group_sha256") != group or
                        feedback.get("disclosure_after_score_seal") is not True or
                        not _diagnostic_identity_matches(
                            diagnostic_row, category=category, position=position,
                            source_group=group)):
                    raise RepairError(f"{category}:{position}: post-seal feedback receipt identity mismatch")
                theta = float(sealed_row["theta_c64_q90"])
                common_prediction = int(float(sealed_row["scores"]["G"]) >= theta)
                label = int(feedback["label"])
                outcome = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[
                    (label, common_prediction)]
                if (feedback.get("outcome_from_common_c0") != outcome or
                        int(diagnostic_row.get("c0_prediction", -1)) != common_prediction or
                        diagnostic_row.get("outcome") != outcome):
                    raise RepairError(f"{category}:{position}: feedback outcome differs from sealed C0")
                diagnostic = diagnostic_row["arms"]["c1_full"]["diagnostic"]
                keep = diagnostic["native_gate_keep_mask"]
                beta = diagnostic["beta_patch_float32_before_codebook_clip"]
                if (diagnostic.get("diagnostic_trace_complete") is not True or
                        len(beta) != sum(bool(value) for value in keep)):
                    raise RepairError(f"{category}:{position}: C1 writer inputs are misaligned")
                arrays = _read_npz_once(
                    _safe_exact_file(input_root, source_row["shard_path"]), source_row, r1)
                features = np.asarray(arrays["top32_features"], dtype=np.float32)
                updates = _apply_class_conditional_feedback(
                    banks, features, keep, beta, label=label, outcome=outcome,
                    position=position, group=group, queried=True, score_sealed=True)
                for name, bank in banks.items():
                    actual = updates[name]
                    if actual is not None:
                        bank.note_post_update_occupancy()
                    if name == "G_NA":
                        gna_native_updates_match = (
                            gna_native_updates_match and
                            _stats_match(actual or {}, diagnostic_row["arms"]["c1_full"]["native_layer_stats"]) and
                            _stats_match(actual or {}, diagnostic.get("native_update_stats", {})) and
                            bank.occupancy == int(diagnostic["anchor_occupancy_after"]))
            if query_count_category != sum(
                    bool(row["query_tape_decision"]) for row in rows[input_index - 512:input_index]):
                raise RepairError(f"{category}: bank audit query count differs from sealed schedule")
            result[category] = {"banks": {
                name: {
                    "final_occupancy": bank.occupancy,
                    "max_occupancy": bank.max_occupancy,
                    "average_occupancy": bank.occupancy_total / max(1, bank.occupancy_steps),
                    "update_counts": dict(bank.stats),
                    "eligible_slot_events": bank.eligible_slots_total,
                    "same_source_excluded_slot_events": bank.excluded_slots_total,
                } for name, bank in banks.items()}}
            result[category]["validation_status"] = (
                "PASS" if gna_native_updates_match else "UNVERIFIED_GNA_UPDATE_TELEMETRY")
            result[category]["query_count"] = query_count_category
        if input_index != 2048 or query_count != 67:
            raise RepairError("Post-seal bank audit did not consume the frozen 2,048/67 schedule")
        readers["scores"].finish(
            expected_count=2048, expected_sha256=input_hashes["score_rows_sha256"])
        readers["diagnostics"].finish(
            expected_count=67, expected_sha256=input_hashes["strength_diagnostics_sha256"])
        readers["feedback"].finish(
            expected_count=67, expected_sha256=input_hashes["queried_feedback_sha256"])
    finally:
        readers["scores"].__exit__()
        readers["diagnostics"].__exit__()
        readers["feedback"].__exit__()
    return result


def _metric_pack(perf: Any, labels: Any, scores: Any, theta: float) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    auc, ap, reason = perf._auc_ap(y, s)
    fixed = perf._fixed_threshold_metrics(y, s, theta)
    return {
        "n": int(y.size), "positive_count": int(np.count_nonzero(y == 1)),
        "negative_count": int(np.count_nonzero(y == 0)),
        "anomaly_prevalence": float(np.mean(y == 1)) if y.size else None,
        "auroc": auc, "average_precision": ap,
        "ranking_undefined_reason": reason,
        "fixed_threshold": fixed,
    }


def _unqueried_mask(rows: Sequence[Mapping[str, Any]]) -> Any:
    import numpy as np
    return np.asarray([not bool(row["query_tape_decision"]) for row in rows], dtype=bool)


def _metric_value(pack: Mapping[str, Any], name: str) -> float | None:
    if name in ("auroc", "average_precision"):
        value = pack.get(name)
    else:
        value = pack.get("fixed_threshold", {}).get(name)
    return None if value is None else float(value)


def _difference(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else float(a - b)


def _repair_ledger(labels: Any, scores: Mapping[str, Any], theta: float,
                   reference: str = "G") -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    ref_pred = np.asarray(scores[reference], dtype=np.float64) >= float(theta)
    result: dict[str, Any] = {}
    for arm, arm_scores in scores.items():
        if arm == reference:
            continue
        pred = np.asarray(arm_scores, dtype=np.float64) >= float(theta)
        repaired_fn = int(np.count_nonzero((y == 1) & ~ref_pred & pred))
        repaired_fp = int(np.count_nonzero((y == 0) & ref_pred & ~pred))
        introduced_fn = int(np.count_nonzero((y == 1) & ref_pred & ~pred))
        introduced_fp = int(np.count_nonzero((y == 0) & ~ref_pred & pred))
        base_tp = int(np.count_nonzero((y == 1) & ref_pred))
        base_fn = int(np.count_nonzero((y == 1) & ~ref_pred))
        base_fp = int(np.count_nonzero((y == 0) & ref_pred))
        base_tn = int(np.count_nonzero((y == 0) & ~ref_pred))
        candidate = {
            "tp": int(np.count_nonzero((y == 1) & pred)),
            "fn": int(np.count_nonzero((y == 1) & ~pred)),
            "fp": int(np.count_nonzero((y == 0) & pred)),
            "tn": int(np.count_nonzero((y == 0) & ~pred)),
        }
        reconstructed = {
            "tp": base_tp + repaired_fn - introduced_fn,
            "fn": base_fn - repaired_fn + introduced_fn,
            "fp": base_fp - repaired_fp + introduced_fp,
            "tn": base_tn + repaired_fp - introduced_fp,
        }
        if candidate != reconstructed:
            raise RepairError(f"Repair ledger does not conserve confusion counts for {arm}")
        result[arm] = {
            "fn_to_tp_repaired": repaired_fn,
            "fp_to_tn_repaired": repaired_fp,
            "tp_to_fn_introduced": introduced_fn,
            "tn_to_fp_introduced": introduced_fp,
            "net_fn_reduction": repaired_fn - introduced_fn,
            "net_fp_reduction": repaired_fp - introduced_fp,
            "rates": {
                "fn_repair_among_reference_fn": repaired_fn / base_fn if base_fn else None,
                "fp_repair_among_reference_fp": repaired_fp / base_fp if base_fp else None,
                "new_fn_among_reference_tp": introduced_fn / base_tp if base_tp else None,
                "new_fp_among_reference_tn": introduced_fp / base_tn if base_tn else None,
            },
            "reference_confusion": {"tp": base_tp, "fn": base_fn,
                                    "fp": base_fp, "tn": base_tn},
            "candidate_confusion": candidate,
            "confusion_reconstruction_verified": True,
        }
    return result


def _pair_credit(anomaly_score: Any, normal_score: Any) -> Any:
    import numpy as np
    delta = np.asarray(anomaly_score)[:, None] - np.asarray(normal_score)[None, :]
    return np.where(delta > 0, 1.0, np.where(delta == 0, 0.5, 0.0))


def _pair_accounting(labels: Any, groups: Sequence[str], base_scores: Any,
                     candidate_scores: Any) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    base = np.asarray(base_scores, dtype=np.float64)
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    group_array = np.asarray(groups, dtype=object)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(pos) or not len(neg):
        return {"defined": False, "all_pair_count": int(len(pos) * len(neg)),
                "cross_source_group_pair_count": 0, "reason": "one class absent"}

    def collect(p: Any, n: Any, mask: Any | None) -> dict[str, Any]:
        b = _pair_credit(base[p], base[n])
        c = _pair_credit(candidate[p], candidate[n])
        diff = c - b
        if mask is not None:
            diff = diff[np.asarray(mask, dtype=bool)]
            b = b[np.asarray(mask, dtype=bool)]
            c = c[np.asarray(mask, dtype=bool)]
        count = int(diff.size)
        if not count:
            return {"pair_count": 0, "reference_auc": None,
                    "candidate_auc": None, "net_auc_difference": None,
                    "improved_pair_credit": 0.0, "damaged_pair_credit": 0.0,
                    "pairs_improved": 0, "pairs_damaged": 0}
        return {
            "pair_count": count,
            "reference_auc": float(np.mean(b)),
            "candidate_auc": float(np.mean(c)),
            "net_auc_difference": float(np.mean(diff)),
            "improved_pair_credit": float(np.maximum(diff, 0).sum()),
            "damaged_pair_credit": float(np.maximum(-diff, 0).sum()),
            "pairs_improved": int(np.count_nonzero(diff > 0)),
            "pairs_damaged": int(np.count_nonzero(diff < 0)),
        }
    cross = group_array[pos][:, None] != group_array[neg][None, :]
    all_pairs = collect(pos, neg, None)
    cross_pairs = collect(pos, neg, cross)
    return {"defined": True, "all_pairs": all_pairs,
            "cross_source_group_pairs": cross_pairs,
            "cross_source_group_denominator_is_own_pair_count": True}


def _percentile_summary(values: Sequence[float]) -> dict[str, Any]:
    import numpy as np
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {"n": 0, "mean": None, "median": None, "p05": None,
                "p95": None, "min": None, "max": None}
    return {"n": int(a.size), "mean": float(a.mean()),
            "median": float(np.median(a)), "p05": float(np.quantile(a, .05)),
            "p95": float(np.quantile(a, .95)), "min": float(a.min()),
            "max": float(a.max())}


def _source_group_bootstrap(perf: Any, labels: Any, groups: Sequence[str],
                            score_map: Mapping[str, Any], theta: float,
                            *, iterations: int = BOOTSTRAP_ITERATIONS,
                            seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    group_array = np.asarray(groups, dtype=object)
    unique_groups = sorted(set(group_array.tolist()))
    members = {group: np.flatnonzero(group_array == group) for group in unique_groups}
    score_arrays = {arm: np.asarray(values, dtype=np.float64)
                    for arm, values in score_map.items()}
    comparisons = (("G_N", "G"), ("G_A", "G"), ("G_NA", "G"),
                   ("G_GLOBAL", "G"), ("G_NA", "G_N"),
                   ("G_NA", "G_A"), ("G_NA", "G_GLOBAL"))
    metrics = ("auroc", "average_precision", "balanced_accuracy", "fpr", "fnr")
    samples: dict[str, dict[str, list[float]]] = {
        f"{left}_minus_{right}": {metric: [] for metric in metrics}
        for left, right in comparisons}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        draw = rng.integers(0, len(unique_groups), size=len(unique_groups))
        indices = np.concatenate([members[unique_groups[int(i)]] for i in draw])
        packs = {arm: _metric_pack(perf, y[indices], values[indices], theta)
                 for arm, values in score_arrays.items()}
        for left, right in comparisons:
            name = f"{left}_minus_{right}"
            for metric in metrics:
                diff = _difference(_metric_value(packs[left], metric),
                                   _metric_value(packs[right], metric))
                if diff is not None and math.isfinite(diff):
                    samples[name][metric].append(diff)
    summary: dict[str, Any] = {}
    for name, metric_samples in samples.items():
        summary[name] = {}
        for metric, values in metric_samples.items():
            if values:
                a = np.asarray(values, dtype=np.float64)
                summary[name][metric] = {
                    "valid_replicates": int(a.size),
                    "ci95_percentile": [float(np.quantile(a, .025)),
                                        float(np.quantile(a, .975))],
                    "median": float(np.median(a)),
                }
            else:
                summary[name][metric] = {"valid_replicates": 0,
                                          "ci95_percentile": None, "median": None}
    return {"method": "paired_source_group_cluster_bootstrap",
            "iterations": int(iterations), "seed": int(seed),
            "unique_source_groups": len(unique_groups),
            "same_group_draw_reused_across_all_arms": True,
            "interval_interpretation": "conditional_on_recorded_trajectories",
            "comparisons": summary}


def _evaluate_category(perf: Any, category: str, rows: Sequence[Mapping[str, Any]],
                       labels: Any, groups: Sequence[str]) -> dict[str, Any]:
    import numpy as np
    if len(rows) != 512 or len(labels) != 512 or len(groups) != 512:
        raise RepairError(f"{category}: Stage B did not provide the fixed first-512 prefix")
    theta = float(rows[0]["theta_c64_q90"])
    scores = {arm: np.asarray([float(row["scores"][arm]) for row in rows], dtype=np.float64)
              for arm in MAIN_ARMS + SECONDARY_ARMS}
    main_scores = {arm: scores[arm] for arm in MAIN_ARMS}
    metrics = {arm: _metric_pack(perf, labels, values, theta)
               for arm, values in scores.items()}
    effects = {}
    for left, right in (("G_N", "G"), ("G_A", "G"), ("G_NA", "G"),
                        ("G_GLOBAL", "G"), ("G_NA", "G_N"),
                        ("G_NA", "G_A"), ("G_NA", "G_GLOBAL")):
        effects[f"{left}_minus_{right}"] = {
            name: _difference(_metric_value(metrics[left], name),
                              _metric_value(metrics[right], name))
            for name in ("auroc", "average_precision", "balanced_accuracy", "fpr", "fnr")}
    interaction = {}
    for name in ("auroc", "average_precision", "balanced_accuracy", "fpr", "fnr"):
        gna = _metric_value(metrics["G_NA"], name)
        gn = _metric_value(metrics["G_N"], name)
        ga = _metric_value(metrics["G_A"], name)
        g = _metric_value(metrics["G"], name)
        interaction[name] = None if any(v is None for v in (gna, gn, ga, g)) else float(gna - gn - ga + g)
    repairs = _repair_ledger(labels, main_scores, theta)
    # Add the full and geometry-free arm ledgers, with all transition counts.
    repairs.update(_repair_ledger(labels,
                                  {"G": scores["G"], "G_NA": scores["G_NA"],
                                   "G_GLOBAL": scores["G_GLOBAL"]}, theta))
    all_arms = {arm: scores[arm] for arm in MAIN_ARMS}
    pair_metrics = {arm: _pair_accounting(labels, groups, scores["G"], values)
                    for arm, values in all_arms.items() if arm != "G"}
    pair_auc_checks: dict[str, Any] = {}
    for arm, pair_record in pair_metrics.items():
        expected_delta = effects[f"{arm}_minus_G"]["auroc"]
        if pair_record.get("defined"):
            pair_delta = pair_record["all_pairs"]["net_auc_difference"]
            pair_auc_checks[arm] = {
                "pair_delta": pair_delta, "metric_delta": expected_delta,
                "absolute_error": abs(float(pair_delta) - float(expected_delta)),
                "reconstructs_auroc": abs(float(pair_delta) - float(expected_delta)) <= 1e-12,
            }
            if not pair_auc_checks[arm]["reconstructs_auroc"]:
                raise RepairError(f"{category}:{arm}: all-pair ledger does not reconstruct AUROC change")
        else:
            pair_auc_checks[arm] = {"pair_delta": None,
                                    "metric_delta": expected_delta,
                                    "reconstructs_auroc": None}
    unqueried_mask = _unqueried_mask(rows)
    unqueried_labels = np.asarray(labels, dtype=np.int8)[unqueried_mask]
    unqueried_metrics = {
        arm: _metric_pack(perf, unqueried_labels, values[unqueried_mask], theta)
        for arm, values in scores.items()}
    unqueried_metrics["subset_definition"] = "fixed original unqueried events; diagnostic only"
    bootstrap = _source_group_bootstrap(perf, labels, groups, main_scores, theta)
    feedback = rows
    # Query class counts come from the Stage A score stream only as a schedule;
    # actual disclosed feedback counts are inserted by the Stage A seal below.
    return {
        "category": category, "event_count": 512,
        "threshold_source": "original frozen C64 higher-q90",
        "theta_c64_q90": theta,
        "class_counts": {"normal": int(np.count_nonzero(np.asarray(labels) == 0)),
                         "anomalous": int(np.count_nonzero(np.asarray(labels) == 1))},
        "metrics_by_arm": metrics,
        "paired_effects": effects,
        "factorial_feedback_interaction": interaction,
        "residual_error_repair_ledger_vs_G": repairs,
        "pair_ordering_repair_damage_vs_G": pair_metrics,
        "pair_auc_reconstruction_checks": pair_auc_checks,
        "paired_source_group_bootstrap": bootstrap,
        "unqueried_event_subset": {
            "event_count": int(unqueried_mask.sum()),
            "metrics_by_arm": unqueried_metrics,
        },
        "query_count": int(sum(bool(row["query_tape_decision"]) for row in feedback)),
        "audiojack_ranking_note": (
            "AUROC/AP are undefined because the frozen prefix has no anomalies"
            if category == "audiojack" and int(np.count_nonzero(np.asarray(labels) == 1)) == 0
            else None),
    }


def _stage_b_results(profile: Mapping[str, Any], run_root: Path,
                     budget: OutputBudget) -> dict[str, Any]:
    started = time.monotonic()
    manifest = _load_json(_safe_exact_file(run_root, "run_manifest.json"))
    rows, by_category = _read_sealed_stage_a(run_root, manifest)
    stage_a_seal = _load_json(_safe_exact_file(run_root, "stage_a_seal.json"))
    # No outcome source is opened until the verified complete Stage A seal above.
    import numpy as np
    from experiments.calibration import r2_exp2a_c1_strength as exp2a
    from experiments.calibration import r2_perf1_stage_b as perf
    bank_accounting = _replay_bank_accounting_after_seal(rows, run_root, manifest)
    dataset_root = Path(r"G:\dataset\realiad")
    json_root = Path(r"G:\dataset\realiad\jsons\realiad_jsons")
    _reject_reparse_chain(dataset_root)
    _reject_reparse_chain(json_root)
    frozen_inputs = ROOT / "outputs/calibration/stage05_exploratory_20261029/frozen_inputs.json"
    _safe_exact_file(ROOT, frozen_inputs.relative_to(ROOT))
    categories: dict[str, Any] = {}
    for category in CATEGORIES:
        category_rows = by_category[category]
        labels, observed_groups = exp2a._stage_b_labels(
            category, category_rows, dataset_root, json_root)
        if len(labels) != 512 or np.any(np.asarray(labels) < 0):
            raise RepairError(f"{category}: first-512 Stage B identity projection is incomplete")
        result = _evaluate_category(perf, category, category_rows, labels, observed_groups)
        source_audit = stage_a_seal["category_audit"][category]
        result["c1_full_reconstruction_status"] = source_audit["parity_status"]
        result["c1_full_reference_auc"] = {
            "expected": {"switch": 0.843083, "regulator": 0.694623,
                         "rolled_strip_base": 0.910798}.get(category),
            "actual": result["metrics_by_arm"]["G_NA"]["auroc"],
            "status": ("UNDEFINED_EXPECTED" if category == "audiojack" else
                       "CHECKED" if source_audit["parity_status"] == "PASS" else
                       "UNVERIFIED_C1_RECONSTRUCTION"),
        }
        expected_auc = result["c1_full_reference_auc"]["expected"]
        if expected_auc is not None and source_audit["parity_status"] == "PASS":
            actual_auc = result["c1_full_reference_auc"]["actual"]
            result["c1_full_reference_auc"]["absolute_error"] = (
                None if actual_auc is None else abs(float(actual_auc) - expected_auc))
            result["c1_full_reference_auc"]["status"] = (
                "PASS" if actual_auc is not None and
                abs(float(actual_auc) - expected_auc) <= 1e-6 else "MISMATCH")
        # Add the actual queried feedback class counts, not guessed from outcomes.
        result["queried_feedback_counts"] = {
            "normal": int(source_audit["feedback_normal"]),
            "anomalous": int(source_audit["feedback_anomalous"]),
            "total": int(source_audit["queries"]),
        }
        result["correction_bank_accounting"] = bank_accounting[category]["banks"]
        result["correction_bank_accounting_validation"] = bank_accounting[category][
            "validation_status"]
        result["positive_negative_same_state_diagnostic"] = source_audit[
            "positive_negative_same_state_summary"]
        categories[category] = result
    ranking_arms = MAIN_ARMS
    defined = {arm: [categories[c]["metrics_by_arm"][arm]["auroc"]
                     for c in CATEGORIES
                     if categories[c]["metrics_by_arm"][arm]["auroc"] is not None]
               for arm in ranking_arms}
    macro = {
        arm: {
            "strict_four_category_macro_auroc": None,
            "strict_four_category_status": "UNDEFINED_AUDIOJACK_NORMAL_ONLY",
            "three_category_descriptive_mean_auroc": (
                float(np.mean(values)) if len(values) == 3 else None),
            "defined_category_count": len(values),
        } for arm, values in defined.items()}
    results = {
        "schema": "r2.feedback_repair.aggregate_results.v1",
        "scientific_classification": "RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC",
        "development_outcomes_historically_exposed": True,
        "independent_or_blind_confirmation": False,
        "task_authorization_id": AUTHORIZATION_ID,
        "run_id": run_root.name,
        "input_runs": {"exp2a": "r2exp2a_20261101_a4",
                       "exp2b_reference_only": "r2exp2b_20261101_b1"},
        "categories": categories,
        "category_aggregate": macro,
        "aggregate_caution": (
            "Three-category means are descriptive and omit normal-only audiojack; "
            "strict four-category ranking macro is undefined."),
        "main_arms": list(MAIN_ARMS),
        "secondary_reference_arms": list(SECONDARY_ARMS),
        "population": {"categories": list(CATEGORIES), "events_per_category": 512,
                       "total_events": 2048, "support_seed": 0,
                       "support_regime": "four supports plus additional C64; not strict four-shot-total",
                       "query_count": 67},
        "bootstrap": {"iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED,
                      "unit": "Source Group", "paired_across_arms": True},
        "stage_a_seal_sha256": _sha256_file(_safe_exact_file(run_root, "stage_a_seal.json")),
        "correction_bank_accounting_method": (
            "post-seal query-update replay from cached Top-32 writer inputs; no score recomputation"),
        "elapsed_stage_b_seconds": time.monotonic() - started,
    }
    budget.write_json("aggregate_results.json", results)
    stage_b_seal = {
        "schema": STAGE_B_SEAL_SCHEMA, "status": "SEALED_COMPLETE",
        "run_id": run_root.name,
        "stage_a_seal_sha256": results["stage_a_seal_sha256"],
        "aggregate_results_sha256": _sha256_file(run_root / "aggregate_results.json"),
        "labels_projected": 2048,
        "categories": list(CATEGORIES),
        "first_512_only": True,
        "masks_accessed": False, "images_decoded": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    budget.write_json("stage_b_seal.json", stage_b_seal)
    return results


def _run_stage_b_child(run_root: Path) -> int:
    run_root = _validate_child_run_root(run_root)
    if (os.environ.get("R2_FEEDBACK_REPAIR_STAGE_B_CHILD") != "1" or
            os.environ.get("R2_FEEDBACK_REPAIR_SUPERVISED_CHILD") != "1" or
            len(os.environ.get("R2_FEEDBACK_REPAIR_SUPERVISOR_NONCE", "")) < 32):
        raise RepairError("Stage B must be launched as the sealed-score child")
    budget = OutputBudget(run_root)
    try:
        _stage_b_results({}, run_root, budget)
        return 0
    except BaseException as exc:
        if "failure.json" not in budget.created:
            budget.write_json("failure.json", {
                "schema": "r2.feedback_repair.failure.v1",
                "phase": "STAGE_B", "error_type": type(exc).__name__,
                "message": str(exc)[:2000], "traceback": traceback.format_exc()[-6000:],
                "stage_a_sealed": (run_root / "stage_a_seal.json").is_file(),
            }, failure_reserved=True)
        return 1


def _run_stage_b_only_child(run_root: Path) -> int:
    run_root = _validate_child_run_root(run_root)
    if os.environ.get("R2_FEEDBACK_REPAIR_STAGE_B_ONLY_CHILD") != "1":
        raise RepairError("Stage B reuse is only callable through its supervised parent")
    os.environ["R2_FEEDBACK_REPAIR_STAGE_B_CHILD"] = "1"
    return _run_stage_b_child(run_root)


def _run_stage_a_child(run_root: Path) -> int:
    run_root = _validate_child_run_root(run_root)
    if (os.environ.get("R2_FEEDBACK_REPAIR_SUPERVISED_CHILD") != "1" or
            len(os.environ.get("R2_FEEDBACK_REPAIR_SUPERVISOR_NONCE", "")) < 32):
        raise RepairError("Stage A is only callable as a supervised child")
    budget = OutputBudget(run_root)
    try:
        profile = _validate_inputs_and_identity()
        _run_stage_a(run_root, budget, profile)
        budget.audit()
        budget = None
        child_env = dict(os.environ)
        child_env["R2_FEEDBACK_REPAIR_STAGE_B_CHILD"] = "1"
        child_env["CUDA_VISIBLE_DEVICES"] = ""
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--stage-b-child", "--run-root", str(run_root)]
        result = subprocess.run(command, cwd=ROOT, env=child_env, check=False)
        return int(result.returncode)
    except BaseException as exc:
        if budget is not None and "failure.json" not in budget.created:
            budget.write_json("failure.json", {
                "schema": "r2.feedback_repair.failure.v1",
                "phase": "STAGE_A_OR_STAGE_B_LAUNCH",
                "error_type": type(exc).__name__, "message": str(exc)[:2000],
                "traceback": traceback.format_exc()[-6000:],
            }, failure_reserved=True)
        return 1


def _validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"r2_feedback_repair_20261102_[a-z][0-9]+", run_id):
        raise RepairError("Run ID must be a fresh versioned R2-FEEDBACK-REPAIR namespace")


def _validate_child_run_root(run_root: Path) -> Path:
    root = Path(os.path.abspath(run_root))
    _validate_run_id(root.name)
    expected_parent = ROOT / OUTPUT_REL_PARENT
    if root.parent.resolve(strict=True) != expected_parent.resolve(strict=True):
        raise RepairError("Child output path is outside the authorized run namespace")
    _reject_reparse_chain(root.parent)
    return root


def _new_run_manifest(run_id: str, profile: Mapping[str, Any], *,
                      stage_a_source_run_id: str,
                      reused_from: str | None) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA, "run_id": run_id,
        "authorization_id": AUTHORIZATION_ID,
        "scientific_classification": "RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC",
        "branch": profile["branch"], "head_at_launch": profile["head"],
        "expected_baseline_ancestor": EXPECTED_BASELINE,
        "input_run_ids": {"exp2a": profile["exp2a_run_id"],
                           "exp2b_reference": profile["exp2b_run_id"]},
        "input_seals": {
            "exp2a_experiment_seal_sha256": profile["exp2a_experiment_seal_sha256"],
            "exp2a_stage_a_seal_sha256": profile["exp2a_stage_a_seal_sha256"],
            "exp2a_files": profile["exp2a_stage_a_input_hashes"],
            "exp2b_stage_a_status": profile["exp2b_stage_a_status"],
            "exp2b_receipt_match": profile["exp2b_input_receipt_match"],
        },
        "population": {"categories": list(CATEGORIES), "events_per_category": 512,
                       "total_events": 2048, "support_seed": 0,
                       "support_regime": "four supports plus additional C64; not strict four-shot-total",
                       "original_queries": 67},
        "arms": list(MAIN_ARMS), "secondary_references": list(SECONDARY_ARMS),
        "runtime": {"device": "cpu", "cuda_visible_devices": "",
                    "supervisor": "existing Windows Job Object process-tree supervisor",
                    "hard_deadline_seconds": SUPERVISED_DEADLINE_SECONDS,
                    "automatic_restart": False},
        "output_bound": {"max_bytes": MAX_OUTPUT_BYTES,
                         "failure_reserve_bytes": FAILURE_RESERVE_BYTES,
                         "max_files": MAX_OUTPUT_FILES,
                         "os_level_quota": False},
        "stage_a_algorithm_sha256": _stage_a_algorithm_sha256(),
        "source_hashes": profile["current_source_hashes"],
        "stage_a_source_run_id": stage_a_source_run_id,
        "stage_a_reused_from_run_id": reused_from,
        "source_audit": {
            "exp2a_manifest_status": profile["exp2a_manifest_status"],
            "exp2a_stage_a_status": profile["exp2a_stage_a_status"],
            "exp2a_completion_status": profile["exp2a_completion_status"],
            "replay_source_fingerprints_match": profile["source_fingerprints_match_for_replay"],
            "historical_exp2a_runner_sha256_expected": profile[
                "historical_exp2a_runner_sha256_expected"],
            "historical_exp2a_runner_sha256_actual": profile[
                "historical_exp2a_runner_sha256_actual"],
            "historical_exp2a_runner_sha256_match": profile[
                "historical_exp2a_runner_sha256_match"],
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _resolve_run_root(run_id: str) -> Path:
    _validate_run_id(run_id)
    parent = ROOT / OUTPUT_REL_PARENT
    _reject_reparse_chain(parent)
    if not parent.is_dir():
        raise RepairError("Authorized output parent is missing")
    return parent / run_id


def _run_parent(args: argparse.Namespace) -> int:
    if not args.authorize_cached_data_run:
        raise RepairError("Explicit --authorize-cached-data-run is required")
    if sys.executable.lower() != str(PYTHON).lower():
        raise RepairError(f"Use the specified interpreter: {PYTHON}")
    profile = _validate_inputs_and_identity()
    run_root = _resolve_run_root(args.run_id)
    if run_root.exists() or _is_reparse(run_root):
        raise FileExistsError(f"Refusing to reuse any existing output namespace: {run_root.name}")
    free_bytes = shutil.disk_usage(run_root.parent).free
    if free_bytes < MAX_OUTPUT_BYTES + 1024 * 1024:
        raise RepairError("Insufficient free space for the bounded output namespace")
    reused_from: str | None = None
    stage_a_source_run_id = args.run_id
    reused_payload: tuple[bytes, bytes] | None = None
    reuse_identity: dict[str, Any] | None = None
    if args.reuse_stage_a_from:
        reused_from = args.reuse_stage_a_from
        source_root = _resolve_run_root(reused_from)
        if not source_root.is_dir() or _is_reparse(source_root):
            raise RepairError("Stage A reuse source is missing or unsafe")
        source_manifest = _load_json(_safe_exact_file(source_root, "run_manifest.json"))
        rows, _ = _read_sealed_stage_a(source_root, source_manifest)
        source_seal = _load_json(_safe_exact_file(source_root, "stage_a_seal.json"))
        if source_seal.get("stage_a_algorithm_sha256") != _stage_a_algorithm_sha256():
            raise RepairError("Stage A scoring/update algorithm changed; sealed scores cannot be reused")
        mismatches = _stage_a_reuse_source_mismatches(
            source_manifest.get("source_hashes", {}),
            profile["current_source_hashes"])
        if mismatches:
            raise RepairError(f"Stage A native dependency fingerprints changed: {sorted(mismatches)}")
        reusable = set(STAGE_A_EXTERNAL_SOURCE_PATHS)
        non_stage_a_differences = {
            key: {"expected": source_manifest.get("source_hashes", {}).get(key),
                  "actual": value}
            for key, value in profile["current_source_hashes"].items()
            if key not in reusable and source_manifest.get("source_hashes", {}).get(key) != value
        }
        reuse_identity = {
            "schema": "r2.feedback_repair.stage_a_reuse_identity.v2",
            "stage_a_algorithm_sha256_match": True,
            "native_stage_a_dependencies_match": True,
            "native_dependency_paths": list(STAGE_A_EXTERNAL_SOURCE_PATHS),
            "stage_b_or_reporting_source_differences": non_stage_a_differences,
        }
        stage_a_source_run_id = str(source_seal["run_id"])
        reused_payload = (
            _safe_exact_file(source_root, "stage_a_scores.jsonl").read_bytes(),
            _safe_exact_file(source_root, "stage_a_seal.json").read_bytes(),
        )
        if len(rows) != 2048:
            raise RepairError("Stage A reuse source is not a complete 2,048-event seal")
    budget = OutputBudget(run_root, create=True)
    manifest = _new_run_manifest(args.run_id, profile,
                                 stage_a_source_run_id=stage_a_source_run_id,
                                 reused_from=reused_from)
    if reuse_identity is not None:
        manifest["stage_a_reuse_identity"] = reuse_identity
    try:
        budget.write_json("run_manifest.json", manifest)
        budget.append_jsonl("attempt_history.jsonl", {
            "attempt": 1, "phase": "STARTED",
            "mode": "STAGE_B_ONLY_REUSE" if reused_from else "FULL_STAGE_A_AND_B",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage_a_algorithm_sha256": manifest["stage_a_algorithm_sha256"],
            "automatic_restart": False,
        })
    except BaseException as exc:
        if "failure.json" not in budget.created:
            try:
                budget.write_json("failure.json", {
                    "schema": "r2.feedback_repair.failure.v1",
                    "phase": "PRE_SUPERVISOR_INITIALIZATION",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "experiment_child_started": False,
                    "stage_a_sealed": False,
                    "automatic_restart": False,
                }, failure_reserved=True)
            except Exception:
                pass
        raise
    if reused_payload is not None:
        budget.write_once("stage_a_scores.jsonl", reused_payload[0])
        budget.write_once("stage_a_seal.json", reused_payload[1])
        budget.write_json("stage_a_reuse_receipt.json", {
            "schema": "r2.feedback_repair.stage_a_reuse_receipt.v1",
            "source_run_id": reused_from,
            "stage_a_source_run_id": stage_a_source_run_id,
            "stage_a_score_sha256": _sha256_bytes(reused_payload[0]),
            "stage_a_algorithm_sha256": manifest["stage_a_algorithm_sha256"],
            "identity_contract": manifest.get("stage_a_reuse_identity"),
            "reused_scores_without_replay": True,
        })
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "R2_FEEDBACK_REPAIR_SUPERVISED_CHILD": "1",
        "R2_FEEDBACK_REPAIR_SUPERVISOR_NONCE": os.urandom(32).hex(),
    })
    child_args = [sys.executable, str(Path(__file__).resolve()),
                  "--run-root", str(run_root)]
    if reused_from:
        env["R2_FEEDBACK_REPAIR_STAGE_B_ONLY_CHILD"] = "1"
        child_args.append("--stage-b-only-child")
    else:
        child_args.append("--stage-a-child")
    from experiments.calibration.r1_runtime_supervisor import (
        SupervisorBlockingFailure, run_supervised_child)
    started = time.monotonic()
    try:
        result = run_supervised_child(
            child_args, timeout_seconds=SUPERVISED_DEADLINE_SECONDS,
            env=env, cwd=ROOT)
    except SupervisorBlockingFailure:
        # A possible surviving writer makes any parent-side output mutation unsafe.
        raise
    elapsed = time.monotonic() - started
    final_budget = OutputBudget(run_root)
    supervisor_record = {
        "schema": "r2.feedback_repair.supervisor_status.v1",
        "result": {key: getattr(result, key) for key in (
            "status", "child_pid", "child_exit_code", "elapsed_seconds",
            "timeout_seconds", "termination_confirmed", "active_job_processes")},
        "deadline_seconds": SUPERVISED_DEADLINE_SECONDS,
        "job_object_process_tree_containment": True,
        "automatic_restart": False,
        "cuda_visible_devices": "",
        "elapsed_wall_seconds": elapsed,
    }
    final_budget.write_json("supervisor_status.json", supervisor_record)
    success = (result.status == "COMPLETED" and
               result.termination_confirmed and result.child_exit_code == 0)
    if not success and "failure.json" not in final_budget.created:
        final_budget.write_json("failure.json", {
            "schema": "r2.feedback_repair.failure.v1",
            "phase": "SUPERVISED_EXECUTION",
            "supervisor_status": result.status,
            "child_exit_code": result.child_exit_code,
            "elapsed_wall_seconds": elapsed,
            "stage_a_sealed": (run_root / "stage_a_seal.json").is_file(),
        }, failure_reserved=True)
    final_budget.append_jsonl("attempt_history.jsonl", {
        "attempt": 1, "phase": "COMPLETED" if success else "FAILED",
        "supervisor_status": result.status,
        "elapsed_wall_seconds": elapsed,
        "stage_a_sealed": (run_root / "stage_a_seal.json").is_file(),
        "stage_b_sealed": (run_root / "stage_b_seal.json").is_file(),
        "automatic_restart": False,
    })
    if success:
        stage_a_seal = _load_json(_safe_exact_file(run_root, "stage_a_seal.json"))
        stage_b_seal = _load_json(_safe_exact_file(run_root, "stage_b_seal.json"))
        aggregate_path = _safe_exact_file(run_root, "aggregate_results.json")
        final_budget.write_json("complete_seal.json", {
            "schema": "r2.feedback_repair.complete_seal.v1",
            "status": "COMPLETE",
            "run_id": args.run_id,
            "stage_a_seal_sha256": _sha256_file(run_root / "stage_a_seal.json"),
            "stage_b_seal_sha256": _sha256_file(run_root / "stage_b_seal.json"),
            "aggregate_results_sha256": _sha256_file(aggregate_path),
            "stage_a_status": stage_a_seal["status"],
            "stage_b_status": stage_b_seal["status"],
            "supervisor_status": result.status,
            "elapsed_wall_seconds": elapsed,
            "output_budget_before_complete_seal": final_budget.audit(),
        })
    summary = final_budget.audit()
    print(json.dumps({"run_id": args.run_id, "supervisor_status": result.status,
                      "successful": success, "output_audit": summary},
                     sort_keys=True))
    return 0 if success else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-cached-data-run", action="store_true")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--reuse-stage-a-from")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--stage-a-child", action="store_true")
    parser.add_argument("--stage-b-child", action="store_true")
    parser.add_argument("--stage-b-only-child", action="store_true")
    args = parser.parse_args(argv)
    if sum((args.stage_a_child, args.stage_b_child, args.stage_b_only_child)) > 1:
        parser.error("Only one child phase may be selected")
    try:
        if args.stage_b_child:
            if args.run_root is None:
                parser.error("--stage-b-child requires --run-root")
            return _run_stage_b_child(args.run_root)
        if args.stage_b_only_child:
            if args.run_root is None:
                parser.error("--stage-b-only-child requires --run-root")
            return _run_stage_b_only_child(args.run_root)
        if args.stage_a_child:
            if args.run_root is None:
                parser.error("--stage-a-child requires --run-root")
            return _run_stage_a_child(args.run_root)
        return _run_parent(args)
    except BaseException as exc:
        print(f"R2-FEEDBACK-REPAIR-1 failed closed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
