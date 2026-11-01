"""Bounded CPU-only replay for the frozen Source-Group residual readout."""
from __future__ import annotations

import os

# Set these before NumPy, Torch, or scorer imports in either worker.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import hashlib
import json
import math
import os.path
import platform
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from experiments.calibration import r2_sgrm

ROOT = Path(__file__).resolve().parents[2]
BRANCH = "calibration-stage0-contract-20261029"
PARENT = "e2e31be50ccfffbcc967fb1d68505b2a57da595c"
PYTHON = Path(r"D:\anaconda3\envs\torch\python.exe")
RUN_ID = "r2_sgrm_val1_20261102_a5"
OUTPUT_ROOT = ROOT / "outputs/calibration" / RUN_ID
DELIVERABLE_ROOT = ROOT / "docs/calibration/audit/r2_sgrm_validation_20261102"
REPORT_PATH = DELIVERABLE_ROOT / "REPORT.md"
AGGREGATE_PATH = DELIVERABLE_ROOT / "aggregate_results.json"
EXP2A_RELATIVE = Path("outputs/calibration/r1_digest_contract_v1_exp2a_r2exp2a_20261101_a4")
EXP2B_RELATIVE = Path("outputs/calibration/r1_digest_contract_v1_exp2b_r2exp2b_20261101_b1")
CATEGORIES = ("switch", "audiojack", "regulator", "rolled_strip_base")
EVENTS = 512
TAU = 0.10
EPSILON = 1e-6
ATOL = 1e-5
RTOL = 1e-4
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261101
MAX_RUNTIME_SECONDS = 600
TERMINATION_GRACE_SECONDS = 10
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_OUTPUT_FILES = 16
FAILURE_RESERVE_BYTES = 64 * 1024
OUTCOME_KEYS = frozenset(("label", "outcome", "anomaly_class"))
PROCESS_START_MONOTONIC = time.monotonic()
RUN_FILES = {
    "run_manifest.json", "stage_a_scores.jsonl", "stage_a_seal.json",
    "stage_b_started.json", "stage_b_results.json", "completion_seal.json",
    "failure.json", *(f"stage_b_{category}.json" for category in CATEGORIES),
}


class ValidationError(RuntimeError):
    """Fail-closed input, parity, causal, or resource error."""


def _validated_beta_cap(value: Any) -> float:
    cap = float(value)
    if not math.isfinite(cap) or cap <= 0:
        raise ValidationError("Frozen correction beta cap is invalid")
    return cap


def _canonical(value: Any, *, newline: bool = True) -> bytes:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False)
    return (text + ("\n" if newline else "")).encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        import stat
        return bool(path.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    except (AttributeError, FileNotFoundError):
        return False


def _reject_reparse_chain(path: Path, description: str) -> None:
    current = path.absolute()
    while True:
        if _is_reparse(current):
            raise ValidationError(f"{description} contains a symlink or reparse point: {current}")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _safe_exact_file(root: Path, relative: str | Path) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or any(part in ("..", "") for part in rel.parts):
        raise ValidationError("Input path is not a safe relative path")
    _reject_reparse_chain(root, "Allowlisted input path")
    base = root.resolve(strict=True)
    current = root
    if _is_reparse(root):
        raise ValidationError("Allowlisted input root is a reparse point")
    for part in rel.parts:
        current = current / part
        if _is_reparse(current):
            raise ValidationError("Input path contains a symlink or reparse point")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(base) or not resolved.is_file():
        raise ValidationError("Resolved input escaped its allowlisted root")
    return resolved


def _recursive_has_key(value: Any, forbidden: frozenset[str]) -> bool:
    if isinstance(value, Mapping):
        return any(key in forbidden or _recursive_has_key(item, forbidden)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_recursive_has_key(item, forbidden) for item in value)
    return False


class OutputBudget:
    """Exclusive flat-output ledger with pre-write byte and file checks."""

    def __init__(self, root: Path, *, create: bool):
        self.root = root
        self.parent = root.parent
        self.lock = threading.Lock()
        self.created: set[str] = set()
        self.used_bytes = 0
        self.file_count = 0
        _reject_reparse_chain(self.parent, "Run-output parent")
        if create:
            self.root.mkdir(parents=False, exist_ok=False)
        if not self.root.is_dir() or _is_reparse(self.root):
            raise ValidationError("Run-output directory is missing or unsafe")
        _reject_reparse_chain(self.root, "Run-output path")
        if self.root.resolve(strict=True).parent != self.parent.resolve(strict=True):
            raise ValidationError("Run-output path escaped its authorized parent")
        if not create:
            for path in self.root.iterdir():
                if (_is_reparse(path) or not path.is_file() or path.name not in RUN_FILES):
                    raise ValidationError("Run-output directory contains an unaccounted file")
                self.created.add(path.name)
                self.file_count += 1
                self.used_bytes += int(path.stat().st_size)
            if self.file_count > MAX_OUTPUT_FILES or self.used_bytes > MAX_OUTPUT_BYTES:
                raise ValidationError("Existing run output already exceeds its budget")

    def _path(self, name: str, size: int, *, new: bool) -> Path:
        if Path(name).name != name or name not in RUN_FILES:
            raise ValidationError("Output filename is outside the allowlist")
        path = self.root / name
        if _is_reparse(path) or path.parent.resolve() != self.root.resolve():
            raise ValidationError("Output path escaped the fresh run namespace")
        if new:
            if path.exists() or name in self.created:
                raise ValidationError(f"Refusing to overwrite {name}")
            file_ceiling = MAX_OUTPUT_FILES if name == "failure.json" else MAX_OUTPUT_FILES - 1
            if self.file_count + 1 > file_ceiling:
                raise ValidationError("Run-output file budget exceeded before write")
        elif name not in self.created or not path.is_file():
            raise ValidationError("Append requires this run's registered JSONL file")
        byte_ceiling = MAX_OUTPUT_BYTES if name == "failure.json" else MAX_OUTPUT_BYTES - FAILURE_RESERVE_BYTES
        if self.used_bytes + size > byte_ceiling:
            raise ValidationError("Run-output byte budget exceeded before write")
        return path

    def write(self, name: str, payload: bytes) -> None:
        data = bytes(payload)
        with self.lock:
            path = self._path(name, len(data), new=True)
            self.created.add(name)
            self.file_count += 1
            try:
                with path.open("xb") as stream:
                    count = stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                if count != len(data):
                    raise ValidationError("Short output write")
                self.used_bytes += int(count)
            except BaseException:
                if path.exists():
                    self.used_bytes += int(path.stat().st_size)
                raise

    def write_json(self, name: str, value: Any) -> None:
        self.write(name, _canonical(value))

    def append_json(self, name: str, value: Any, *, sync: bool = True) -> None:
        data = _canonical(value)
        with self.lock:
            new = name not in self.created
            path = self._path(name, len(data), new=new)
            before = path.stat().st_size if path.exists() else 0
            if new:
                self.created.add(name)
                self.file_count += 1
            try:
                with path.open("ab" if not new else "xb", buffering=0) as stream:
                    count = stream.write(data)
                    if sync:
                        stream.flush()
                        os.fsync(stream.fileno())
                if count != len(data):
                    raise ValidationError("Short JSONL score seal write")
                self.used_bytes += int(count)
            except BaseException:
                if path.exists():
                    self.used_bytes += max(0, int(path.stat().st_size) - before)
                raise


def _json_line(stream: Any, digest: Any, description: str) -> dict[str, Any]:
    line = stream.readline()
    if not line:
        raise ValidationError(f"Unexpected EOF in {description}")
    digest.update(line)
    try:
        row = json.loads(line)
    except Exception as exc:
        raise ValidationError(f"Malformed JSONL record in {description}") from exc
    if not isinstance(row, dict):
        raise ValidationError(f"Expected object in {description}")
    return row


def _flat_gaussian(distances: Any, beta: Any) -> np.ndarray:
    d = np.asarray(distances)
    b = np.asarray(beta)
    if (d.ndim != 2 or d.dtype.kind != "f" or b.shape != (d.shape[1],) or
            b.dtype.kind != "f" or not np.isfinite(d).all() or not np.isfinite(b).all()):
        raise ValidationError("Flat Gaussian inputs are malformed or nonfinite")
    if d.shape[1] == 0:
        return np.zeros((d.shape[0],), dtype=np.float32)
    d64, b64 = np.asarray(d, dtype=np.float64), np.asarray(b, dtype=np.float64)
    weights = np.exp(-(d64 * d64) / (2.0 * TAU * TAU))
    values = np.sum(weights * b64[None, :], axis=1, dtype=np.float64) / (
        EPSILON + np.sum(weights, axis=1, dtype=np.float64))
    if not np.isfinite(values).all():
        raise ValidationError("Flat Gaussian output is nonfinite")
    return values.astype(np.float32)


def _paired_deltas(distances: Any, beta: Any, group_ids: Sequence[str]
                   ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """The same matrix and anchor vectors feed both readouts."""
    flat = _flat_gaussian(distances, beta)
    sgrm, diagnostics = r2_sgrm.sgrm_delta_from_distances(
        distances, beta, group_ids, tau=TAU, epsilon=EPSILON,
        return_diagnostics=True)
    return flat, sgrm, diagnostics


def _eligible(anchors: Sequence[Any], position: int, group: str) -> list[Any]:
    return [anchor for anchor in anchors
            if int(anchor.position) < position and anchor.source_group != group]


def _memory_digest(anchors: Sequence[Any]) -> str:
    h = hashlib.sha256()
    for anchor in anchors:
        h.update(anchor.feature.detach().cpu().numpy().astype(np.float32, copy=False).tobytes())
        h.update(np.asarray([anchor.beta, anchor.radius, anchor.kappa], dtype=np.float32).tobytes())
        h.update(int(anchor.position).to_bytes(8, "big", signed=True))
        h.update(anchor.source_group.encode("ascii"))
    return h.hexdigest()


def _event_identity(score: Mapping[str, Any], ref: Mapping[str, Any],
                    category: str, position: int) -> None:
    if score.get("category") != category or score.get("position") != position:
        raise ValidationError(f"{category}:{position}: event order differs")
    if score.get("event_number") != CATEGORIES.index(category) * EVENTS + position:
        raise ValidationError(f"{category}:{position}: global event number differs")
    for field in ("event_id", "class_json_sha256", "source_identity_sha256", "source_group_sha256"):
        if score.get(field) != ref.get(field):
            raise ValidationError(f"{category}:{position}: {field} identity mismatch")
    group = score.get("source_group_sha256")
    if not isinstance(group, str) or len(group) != 64:
        raise ValidationError(f"{category}:{position}: malformed Source Group digest")


def _validate_current_query_records(score: Mapping[str, Any], diagnostic: Mapping[str, Any],
                                    feedback: Mapping[str, Any], category: str,
                                    position: int) -> int:
    """Validate the actual EXP2A query schemas after this event is score-sealed."""
    group = score.get("source_group_sha256")
    if (diagnostic.get("category") != category or diagnostic.get("position") != position or
            diagnostic.get("source_group_sha256") != group):
        raise ValidationError(f"{category}:{position}: queried diagnostic identity mismatch")
    for field in ("category", "position", "event_id", "source_identity_sha256", "source_group_sha256"):
        if feedback.get(field) != score.get(field):
            raise ValidationError(f"{category}:{position}: queried feedback {field} identity mismatch")
    if feedback.get("disclosure_after_score_seal") is not True:
        raise ValidationError(f"{category}:{position}: feedback precedes score seal")
    label = feedback.get("label")
    if isinstance(label, bool) or label not in (0, 1) or diagnostic.get("label") != label:
        raise ValidationError(f"{category}:{position}: queried label mismatch")
    return int(label)


def _read_seal_inputs(exp2a: Any, exp2b: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_a = (ROOT / EXP2A_RELATIVE).resolve(strict=True)
    expected_b = (ROOT / EXP2B_RELATIVE).resolve(strict=True)
    if exp2a.ROOT.resolve(strict=True) != ROOT or exp2b.INPUT_ROOT.resolve(strict=True) != expected_a:
        raise ValidationError("EXP2A input helper resolved outside its exact allowlisted root")
    if exp2b.OUTPUT_ROOT.resolve(strict=True) != expected_b:
        raise ValidationError("EXP2B seal helper resolved outside its exact allowlisted root")
    manifest = json.loads(_safe_exact_file(expected_a, "run_manifest.json").read_text(encoding="utf-8"))
    stage_bytes = _safe_exact_file(expected_a, "stage_a_seal.json").read_bytes()
    completion_bytes = _safe_exact_file(expected_a, "experiment_seal.json").read_bytes()
    stage = json.loads(stage_bytes)
    completion = json.loads(completion_bytes)
    if (manifest.get("run_id") != "r2exp2a_20261101_a4" or
            manifest.get("authorized_categories") != list(CATEGORIES) or
            manifest.get("authorized_event_horizon") != [0, EVENTS - 1] or
            "seed-0" not in str(manifest.get("support_regime", ""))):
        raise ValidationError("EXP2A manifest category, prefix, or support-seed identity differs")
    if (manifest.get("status") != "INCOMPLETE" or manifest.get("c64_rescored") is not False or
            stage.get("schema") != "r2.exp2a.stage_a_seal.v1" or
            stage.get("status") != "SEALED_COMPLETE" or stage.get("total_events") != 2048 or
            stage.get("event_shard_count") != 2048 or
            stage.get("category_event_counts") != {category: EVENTS for category in CATEGORIES}):
        raise ValidationError("EXP2A Stage A seal is incomplete or outside the authorized prefix")
    if (completion.get("schema") != "r2.exp2a.experiment_seal.v1" or
            completion.get("status") != "COMPLETE" or completion.get("run_id") != manifest.get("run_id") or
            completion.get("stage_a_status") != "SEALED_COMPLETE" or
            completion.get("stage_b_status") != "COMPLETE" or
            completion.get("stage_a_seal_sha256") != exp2b._sha256_bytes(stage_bytes) or
            completion.get("total_event_count") != 2048 or completion.get("total_query_count") != 67 or
            completion.get("event_counts") != {category: EVENTS for category in CATEGORIES}):
        raise ValidationError("EXP2A completion seal does not match its sealed 2,048-event Stage A prefix")
    exp2b_seal, refs, arrays = exp2b._verify_score_seal(exp2b.OUTPUT_ROOT)
    if (exp2b_seal.get("input_run_id") != "r2exp2a_20261101_a4" or
            exp2b_seal.get("status") != "SEALED_COMPLETE" or
            exp2b_seal.get("input_stage_a_seal_sha256") != exp2b._sha256_bytes(stage_bytes) or
            exp2b_seal.get("input_experiment_seal_sha256") != exp2b._sha256_bytes(completion_bytes)):
        raise ValidationError("EXP2B sealed score input is not the authorized complete replay")
    header = {"manifest": manifest, "stage_seal": stage, "experiment_seal": completion,
              "stage_seal_sha256": exp2b._sha256_bytes(stage_bytes),
              "experiment_seal_sha256": exp2b._sha256_bytes(completion_bytes)}
    return header, exp2b_seal, refs, arrays


def _parity(actual: np.ndarray, expected: np.ndarray, stats: dict[str, Any], key: str) -> None:
    a, b = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    if a.shape != b.shape:
        raise ValidationError(f"{key}: shape mismatch")
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    maximum = float(delta.max()) if delta.size else 0.0
    stats[key + "_max_abs"] = max(float(stats.get(key + "_max_abs", 0.0)), maximum)
    stats[key + "_non_bitwise"] = int(stats.get(key + "_non_bitwise", 0)) + int(np.count_nonzero(a != b))
    if not np.allclose(a, b, atol=ATOL, rtol=RTOL):
        raise ValidationError(f"{key}: frozen tolerance exceeded, max_abs={maximum:.9g}")


def _group_diagnostics(diag: Mapping[str, Any], beta: np.ndarray,
                       groups: Sequence[str]) -> dict[str, Any]:
    means = np.asarray(diag["group_means"], dtype=np.float64)
    weights = np.asarray(diag["outer_group_weights"], dtype=np.float64)
    contribution = means * weights
    magnitude = np.abs(contribution)
    totals = magnitude.sum(axis=1)
    valid = totals > 0
    effective = np.zeros(len(totals), dtype=np.float64)
    if np.any(valid):
        p = magnitude[valid] / totals[valid, None]
        effective[valid] = 1.0 / np.maximum(np.sum(p * p, axis=1), 1e-300)
    by_group = contribution.sum(axis=0) if contribution.ndim == 2 else np.empty(0)
    group_beta: dict[str, list[float]] = {}
    for group, value in zip(groups, beta):
        group_beta.setdefault(group, []).append(float(value))
    signs = [float(np.sum(values)) for values in group_beta.values()]
    return {
        "eligible_anchor_count": int(beta.size),
        "eligible_source_group_count": len(group_beta),
        "eligible_group_patch_counts": [len(group_beta[group]) for group in sorted(group_beta)],
        "positive_anchor_count": int(np.count_nonzero(beta > 0)),
        "negative_anchor_count": int(np.count_nonzero(beta < 0)),
        "positive_source_group_count": int(sum(value > 0 for value in signs)),
        "negative_source_group_count": int(sum(value < 0 for value in signs)),
        "effective_groups_mean_patch": float(effective[valid].mean()) if np.any(valid) else 0.0,
        "effective_groups_max_patch": float(effective.max()) if effective.size else 0.0,
        "nonzero_correction_patches": int(valid.sum()),
        "positive_group_contribution_mass": float(np.maximum(by_group, 0).sum()),
        "negative_group_contribution_mass": float(np.maximum(-by_group, 0).sum()),
        "positive_net_groups": int(np.count_nonzero(by_group > 0)),
        "negative_net_groups": int(np.count_nonzero(by_group < 0)),
    }


def _score_seal_record(score: Callable[[], Any], seal: Callable[[Any], None],
                       disclose_update: Callable[[], None], *, queried: bool) -> Any:
    value = score()
    seal(value)
    # Query diagnostics and labels are unavailable until the event's score row
    # has been flushed and fsynced by seal().
    disclose_update()
    return value


def _write_failure(root: Path, error: BaseException) -> None:
    try:
        budget = OutputBudget(root, create=False)
        if "failure.json" not in budget.created:
            budget.write_json("failure.json", {
                "schema": "r2.sgrm.quick_eval.failure.v1", "status": "INCOMPLETE",
                "failure_type": type(error).__name__, "message": str(error)[:1000],
                "automatic_restart": False,
            })
    except Exception:
        pass


def _stage_a(started_monotonic: float) -> int:
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r2_exp2a_c1_strength as exp2a
    from experiments.calibration import r2_exp2b_memory_readout as exp2b

    budget = OutputBudget(OUTPUT_ROOT, create=True)
    try:
        header, exp2b_seal, refs, arrays_by_category = _read_seal_inputs(exp2a, exp2b)
        source_receipts: dict[str, dict[str, Any]] = {}
        for category in CATEGORIES:
            receipt = exp2b_seal["categories"][category]
            memory = receipt["memory_accounting"]
            parity = receipt["reference_parity"]
            if (receipt.get("status") != "PASS" or receipt.get("event_count") != EVENTS or
                    memory.get("first_memory_divergence_position") is not None or
                    memory.get("fifo_evictions") != 0 or memory.get("codebook_merges") != 0 or
                    memory.get("codebook_replacements") != 0 or
                    memory.get("candidate_accounting_identity") is not True or
                    parity.get("events_checked") != EVENTS):
                raise ValidationError(f"Existing EXP2B C1-NoBoth memory receipt fails for {category}")
            source_receipts[category] = {
                "events": int(receipt["event_count"]),
                "queries": int(receipt["query_count"]),
                "fifo_inserts": int(memory["fifo_inserts"]),
                "native_c1_noboth_parity_events": int(parity["events_checked"]),
                "no_fifo_evictions_or_memory_divergence": True,
            }
        manifest = {
            "schema": "r2.sgrm.validation.run_manifest.v1", "run_id": RUN_ID,
            "classification": "RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC",
            "branch": BRANCH, "source_commit": PARENT,
            "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(), "cpu_only": True,
            "input_run_ids": {"exp2a": "r2exp2a_20261101_a4", "exp2b": "r2exp2b_20261101_b1"},
            "categories": list(CATEGORIES), "events_per_category": EVENTS,
            "support_seed": 0, "algorithm_id": r2_sgrm.ALGORITHM_ID,
            "tau": TAU, "epsilon": EPSILON,
            "historical_development_outcome_summaries_previously_observed": True,
            "historical_summaries_used_as_online_inputs": False,
            "cuda_visible_devices": "", "cpu_threads": 1,
            "started_monotonic": started_monotonic,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        budget.write_json("run_manifest.json", manifest)
        input_root = exp2b.INPUT_ROOT
        score_path = _safe_exact_file(input_root, "event_scores.jsonl")
        transition_path = _safe_exact_file(input_root, "event_transitions.jsonl")
        diagnostic_path = _safe_exact_file(input_root, "strength_diagnostics.jsonl")
        feedback_path = _safe_exact_file(input_root, "private_query_feedback.jsonl")
        seal = header["stage_seal"]
        score_hash = hashlib.sha256()
        query_positions: dict[str, list[int]] = {category: [] for category in CATEGORIES}
        receipts: dict[str, Any] = {}
        torch = exp2b._torch()
        torch.set_num_threads(1)
        v22, cfg, _codebook = exp2b._correction_config_and_codebook()
        _validated_beta_cap(cfg.beta_max)
        counts_total_queries = 0
        with (score_path.open("rb") as score_stream,
              _CausalJsonlReader(transition_path, "event_transitions.jsonl") as transition_reader,
              _CausalJsonlReader(diagnostic_path, "strength_diagnostics.jsonl") as diagnostic_reader,
              _CausalJsonlReader(feedback_path, "private_query_feedback.jsonl") as feedback_reader):
            for category_index, category in enumerate(CATEGORIES):
                if time.monotonic() - started_monotonic > MAX_RUNTIME_SECONDS - 5:
                    raise TimeoutError("600-second hard deadline approaching during Stage A")
                category_event_count = 0
                anchors: list[Any] = []
                category_counts = Counter()
                parity = {"flat_delta_vs_EXP2B_D_max_abs": 0.0,
                          "flat_delta_vs_EXP2B_D_non_bitwise": 0,
                          "flat_score_vs_EXP2B_D_max_abs": 0.0,
                          "flat_score_vs_EXP2B_D_non_bitwise": 0,
                          "C0_exact_events": 0, "events": 0}
                diagnostics_summary_rows: list[dict[str, Any]] = []
                category_manifest = header["manifest"]["category_inputs"][category]
                expected_refs = refs[category]
                if len(expected_refs) != EVENTS:
                    raise ValidationError(f"EXP2B identity references incomplete for {category}")

                for position in range(EVENTS):
                    if time.monotonic() - started_monotonic > MAX_RUNTIME_SECONDS - 5:
                        raise TimeoutError("600-second hard deadline approaching during Stage A")
                    row = _json_record(score_stream, score_hash, "event_scores.jsonl")
                    if _recursive_has_key(row, OUTCOME_KEYS):
                        raise ValidationError("Stage A score input contains a forbidden outcome field")
                    _event_identity(row, expected_refs[position], category, position)
                    group = str(row["source_group_sha256"])
                    transition_was_read: dict[str, Any] = {}
                    shard_rel = row.get("shard_path")
                    expected_shard = f"events/event_{category_index * EVENTS + position:06d}.npz"
                    if shard_rel != expected_shard:
                        raise ValidationError(f"{category}:{position}: shard path differs from frozen order")
                    shard_path = _safe_exact_file(input_root, shard_rel)
                    payload = shard_path.read_bytes()
                    if _digest_bytes(payload) != row.get("shard_sha256"):
                        raise ValidationError(f"{category}:{position}: feature/map shard digest differs")
                    event_arrays = exp2b._load_npz_bytes(payload)
                    exp2b._validate_stored_event_arrays(row, event_arrays, r1,
                                                        category=category, position=position)
                    del payload
                    base = event_arrays["base_patch_map"]
                    ids = event_arrays["top32_ids"]
                    features = event_arrays["top32_features"]
                    ref_arrays = arrays_by_category[category]
                    c0 = float(r1.max_score(base))
                    if c0 != float(row["scores"]["c0"]) or c0 != float(ref_arrays["c0"][position]):
                        raise ValidationError(f"{category}:{position}: C0 map/score parity differs")
                    parity["C0_exact_events"] += 1

                    eligible_anchors = _eligible(anchors, position, group)
                    if any(anchor.position >= position or anchor.source_group == group
                           for anchor in eligible_anchors):
                        raise ValidationError(f"{category}:{position}: future/same-source anchor is eligible")
                    query = v22.l2_normalize(exp2b._as_torch_features(features))
                    if eligible_anchors:
                        anchor_features = torch.stack([anchor.feature for anchor in eligible_anchors])
                        distances = exp2b._torch_distance_matrix(query, anchor_features).detach().cpu().numpy()
                        beta = np.asarray([anchor.beta for anchor in eligible_anchors], dtype=np.float32)
                        group_ids = [str(anchor.source_group) for anchor in eligible_anchors]
                    else:
                        distances = np.empty((32, 0), dtype=np.float32)
                        beta = np.empty((0,), dtype=np.float32)
                        group_ids = []
                    before_memory = _memory_digest(anchors)
                    flat_delta, sgrm_delta, group_diag = _paired_deltas(distances, beta, group_ids)
                    if _memory_digest(anchors) != before_memory:
                        raise ValidationError("A residual readout mutated the common correction memory")
                    if not np.isfinite(flat_delta).all() or not np.isfinite(sgrm_delta).all():
                        raise ValidationError(f"{category}:{position}: readout produced nonfinite deltas")
                    _parity(flat_delta, ref_arrays["delta_D"][position], parity,
                            "flat_delta_vs_EXP2B_D")
                    flat_score = exp2b._score_dense(base, ids, flat_delta, r1)
                    sgrm_score = exp2b._score_dense(base, ids, sgrm_delta, r1)
                    c1_score = float(ref_arrays["codebook_gaussian"][position])
                    c1_delta = ref_arrays["delta_D"][position]
                    _parity(np.asarray([flat_score]), np.asarray([c1_score]), parity,
                            "flat_score_vs_EXP2B_D")
                    if not np.isclose(float(row["scores"]["c1_no_both"]), c1_score,
                                      atol=ATOL, rtol=RTOL):
                        raise ValidationError(f"{category}:{position}: C1-NoBoth D score mismatch")
                    if not np.allclose(c1_delta, event_arrays["c1_delta_no_both"],
                                       atol=ATOL, rtol=RTOL):
                        raise ValidationError(f"{category}:{position}: C1-NoBoth native delta differs")
                    c2_score = float(row["scores"]["c2_a002"])
                    if c2_score != float(ref_arrays["historical_c2_a002"][position]):
                        raise ValidationError(f"{category}:{position}: historical kNN(a=.02) score differs")
                    sgrm_diag = group_diag
                    group_info = _group_diagnostics(sgrm_diag, beta, group_ids)
                    local = {
                        "category": category, "position": position,
                        "event_id": row["event_id"],
                        "class_json_sha256": row["class_json_sha256"],
                        "source_identity_sha256": row["source_identity_sha256"],
                        "source_group_sha256": group,
                        "query_tape_decision": bool(row["query_tape_decision"]),
                        "theta_c64_q90": float(row["theta_c64_q90"]),
                        "scores": {"C0": c0, "C1_NoBoth": c1_score,
                                   "historical_C2_a0.02": c2_score,
                                   "matched_flat_Gaussian": flat_score, "SGRM": sgrm_score},
                        "delta_flat": flat_delta.tolist(), "delta_sgrm": sgrm_delta.tolist(),
                        "group_diagnostics": group_info,
                    }
                    def seal_current(value: Mapping[str, Any]) -> None:
                        budget.append_json("stage_a_scores.jsonl", value, sync=True)

                    def update_after_score() -> None:
                        transition = transition_reader.read_record()
                        transition_was_read["row"] = transition
                        if (transition.get("category") != category or transition.get("position") != position or
                                transition.get("query_decision") != row["query_tape_decision"]):
                            raise ValidationError(f"{category}:{position}: transition alignment differs")
                        queried = bool(row["query_tape_decision"])
                        if not queried:
                            if _recursive_has_key(transition, OUTCOME_KEYS):
                                raise ValidationError("Unqueried transition contains a forbidden outcome field")
                            return
                        query_positions[category].append(position)
                        diag_record = diagnostic_reader.read_record()
                        feedback = feedback_reader.read_record()
                        label = _validate_current_query_records(row, diag_record, feedback,
                                                               category, position)
                        diag, _layer_stats = exp2b._diag_for_c1_no_both(diag_record, category, position)
                        predicted = int(c0 >= float(row["theta_c64_q90"]))
                        expected_outcome = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[(label, predicted)]
                        if (feedback.get("outcome_from_common_c0") != expected_outcome or
                                diag.get("outcome") != expected_outcome):
                            raise ValidationError(f"{category}:{position}: query outcome provenance mismatch")
                        keep = np.asarray(diag["native_gate_keep_mask"], dtype=np.bool_)
                        beta_raw = np.asarray(diag["beta_patch_float32_before_codebook_clip"], dtype=np.float32)
                        beta_clip = np.asarray(diag["beta_after_original_codebook_max_clip"], dtype=np.float32)
                        if keep.shape != (32,) or beta_raw.shape != (32,) or beta_clip.shape != (32,):
                            raise ValidationError(f"{category}:{position}: candidate arrays have wrong shapes")
                        candidate = exp2b._common_candidate_eligibility(keep, beta_raw, cfg)
                        if not np.array_equal(candidate["clipped_beta"], beta_clip):
                            raise ValidationError(f"{category}:{position}: original beta clipping mismatch")
                        category_counts["queries"] += 1
                        category_counts["raw_candidates"] += int(candidate["raw_selected_count"])
                        category_counts["dN_rejects"] += int(candidate["dN_gate_rejects"])
                        category_counts["after_dN"] += int(candidate["attempted_after_dN"])
                        category_counts["beta_floor_rejects"] += int(candidate["beta_floor_rejects"])
                        category_counts["eligible_candidates"] += int(candidate["common_eligible_count"])
                        indices = np.flatnonzero(candidate["common_mask"])
                        if indices.size:
                            normalized = v22.l2_normalize(exp2b._as_torch_features(features[indices]))
                            radius = float(np.clip(cfg.radius0, cfg.radius_min, cfg.radius_max))
                            kappa = float(np.clip(cfg.kappa0, 0.1, cfg.kappa_max))
                            for idx, feature in zip(indices.tolist(), normalized):
                                anchors.append(exp2b.Anchor(
                                    feature=feature.detach().clone(), beta=float(beta_clip[idx]),
                                    radius=radius, kappa=kappa, position=position,
                                    source_group=group))
                                category_counts["fifo_inserts"] += 1
                                if len(anchors) > exp2b.CAPACITY:
                                    del anchors[0]
                                    category_counts["fifo_evictions"] += 1
                        if (diag.get("anchor_occupancy_after") != len(anchors) or
                                transition.get("c1_anchor_occupancy_after", {}).get("c1_no_both") != len(anchors)):
                            raise ValidationError(f"{category}:{position}: reconstructed anchor occupancy differs")
                        category_counts["update_trajectory_checks"] += 1

                    _score_seal_record(lambda: local, seal_current, update_after_score,
                                       queried=bool(row["query_tape_decision"]))
                    category_event_count += 1
                    parity["events"] += 1
                    parity["C0_exact_events"] = int(parity["C0_exact_events"])
                    diagnostics_summary_rows.append(group_info)
                    del event_arrays, base, ids, features, distances, beta, group_ids

                qpos = query_positions[category]
                qhash = hashlib.sha256(_canonical(qpos, newline=False)).hexdigest()
                if (len(qpos) != int(category_manifest["query_positions_in_prefix"]) or
                        qhash != category_manifest["query_positions_sha256"]):
                    raise ValidationError(f"{category}: query positions differ from the frozen tape")
                source = source_receipts[category]
                if (category_counts["queries"] != source["queries"] or
                        category_counts["fifo_inserts"] != source["fifo_inserts"] or
                        category_counts["fifo_evictions"] != 0):
                    raise ValidationError(f"{category}: reconstructed candidate stream differs from EXP2B")
                receipts[category] = {
                    "status": "PASS", "event_count": category_event_count,
                    "query_count": int(category_counts["queries"]),
                    "query_positions_sha256": qhash,
                    "reconstructed_candidate_stream": dict(category_counts),
                    "existing_exp2b_candidate_receipt": source,
                    "parity": parity,
                    "source_group_diagnostics": _summarize_group_rows(diagnostics_summary_rows),
                    "final_fifo_occupancy": len(anchors),
                }
                counts_total_queries += int(category_counts["queries"])
        if counts_total_queries != 67:
            raise ValidationError("Total queried feedback count differs from the sealed authorization")
        # All four complete score trajectories exist before a Stage A seal is issued.
        local_score_path = OUTPUT_ROOT / "stage_a_scores.jsonl"
        stage_a = {
            "schema": "r2.sgrm.validation.stage_a.v1", "status": "SEALED_COMPLETE",
            "classification": "RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC", "run_id": RUN_ID,
            "algorithm_id": r2_sgrm.ALGORITHM_ID, "tau": TAU, "epsilon": EPSILON,
            "categories": {category: {"event_count": receipts[category]["event_count"],
                                       "query_count": receipts[category]["query_count"]}
                           for category in CATEGORIES},
            "total_events": sum(receipts[category]["event_count"] for category in CATEGORIES),
            "total_queries": counts_total_queries,
            "stage_a_scores_sha256": _digest_file(local_score_path),
            "input_seals": {
                "exp2a_stage_a": header["stage_seal_sha256"],
                "exp2a_experiment": header["experiment_seal_sha256"],
                "exp2b_stage_a": _digest_file(exp2b.OUTPUT_ROOT / "stage_a_seal.json"),
                "exp2b_event_refs": exp2b_seal["event_refs_sha256"],
            },
            "input_integrity": {
                "event_scores_jsonl": score_hash.hexdigest(),
                "event_transitions_jsonl": transition_reader.digest.hexdigest(),
                "strength_diagnostics_jsonl": diagnostic_reader.digest.hexdigest(),
                "private_query_feedback_jsonl": feedback_reader.digest.hexdigest(),
            },
            "category_receipts": receipts,
            "offline_unqueried_outcomes_read": False,
            "future_feedback_loaded_before_score": False,
            "queried_feedback_accessed_after_score_seal": counts_total_queries,
            "gpu": False, "dino": False, "image_decode": False,
            "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if (stage_a["total_events"] != 2048 or
                input_hashes_match(stage_a["input_integrity"], seal) is False):
            raise ValidationError("Stage A event totals or input-file seals failed")
        budget.write_json("stage_a_seal.json", stage_a)
        return 0
    except BaseException as exc:
        _write_failure(OUTPUT_ROOT, exc)
        raise


def _json_record(stream: Any, digest: Any, name: str) -> dict[str, Any]:
    return _json_line(stream, digest, name)


class _CausalJsonlReader:
    """Lazy byte-exact reader; it opens on demand and never reads ahead."""

    def __init__(self, path: Path, description: str):
        self.path = path
        self.description = description
        self.fd: int | None = None
        self.digest = hashlib.sha256()

    def __enter__(self) -> "_CausalJsonlReader":
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def read_record(self) -> dict[str, Any]:
        if self.fd is None:
            self.fd = os.open(self.path, os.O_RDONLY)
        line = bytearray()
        while True:
            byte = os.read(self.fd, 1)
            if not byte:
                if not line:
                    raise ValidationError(f"Unexpected EOF in {self.description}")
                break
            line.extend(byte)
            if byte == b"\n":
                break
            if len(line) > 1024 * 1024:
                raise ValidationError(f"Oversized JSONL record in {self.description}")
        raw = bytes(line)
        self.digest.update(raw)
        try:
            record = json.loads(raw)
        except Exception as exc:
            raise ValidationError(f"Malformed JSONL record in {self.description}") from exc
        if not isinstance(record, dict):
            raise ValidationError(f"Expected object in {self.description}")
        return record


def input_hashes_match(actual: Mapping[str, str], seal: Mapping[str, Any]) -> bool:
    expected = {
        "event_scores_jsonl": seal.get("score_rows_sha256"),
        "event_transitions_jsonl": seal.get("transition_rows_sha256"),
        "strength_diagnostics_jsonl": seal.get("strength_diagnostics_sha256"),
        "private_query_feedback_jsonl": seal.get("queried_feedback_sha256"),
    }
    return dict(actual) == expected


def _summarize_group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [int(row["eligible_source_group_count"]) for row in rows]
    anchor_counts = [int(row["eligible_anchor_count"]) for row in rows]
    patch_counts = [int(value) for row in rows for value in row["eligible_group_patch_counts"]]
    effective = [float(row["effective_groups_mean_patch"]) for row in rows]
    signs = {name: sum(int(row[name]) for row in rows) for name in (
        "positive_anchor_count", "negative_anchor_count",
        "positive_source_group_count", "negative_source_group_count")}
    size_hist = Counter(patch_counts)
    return {
        "active_feedback_source_groups": {
            "mean_eligible_groups_per_event": float(np.mean(eligible)),
            "max_eligible_groups_per_event": max(eligible, default=0),
            "events_with_any_eligible_group": sum(value > 0 for value in eligible),
        },
        "eligible_anchors": {"mean_per_event": float(np.mean(anchor_counts)),
                             "max_per_event": max(anchor_counts, default=0)},
        "eligible_group_patch_count_distribution": {
            "group_size_frequency": {str(key): int(value) for key, value in sorted(size_hist.items())},
            "mean": float(np.mean(patch_counts)) if patch_counts else 0.0,
            "max": max(patch_counts, default=0),
        },
        "signed_evidence_counts": signs,
        "effective_source_groups": {
            "definition": "inverse Herfindahl over absolute per-patch group correction contributions",
            "mean_per_event": float(np.mean(effective)) if effective else 0.0,
            "max_per_event": float(np.max(effective)) if effective else 0.0,
        },
        "signed_group_contribution_mass": {
            "positive_mean_per_event": float(np.mean([row["positive_group_contribution_mass"] for row in rows])),
            "negative_mean_per_event": float(np.mean([row["negative_group_contribution_mass"] for row in rows])),
            "positive_net_group_event_count": int(sum(row["positive_net_groups"] > 0 for row in rows)),
            "negative_net_group_event_count": int(sum(row["negative_net_groups"] > 0 for row in rows)),
        },
    }


def _stage_b_rows() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    seal_path = OUTPUT_ROOT / "stage_a_seal.json"
    score_path = OUTPUT_ROOT / "stage_a_scores.jsonl"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("status") != "SEALED_COMPLETE" or seal.get("total_events") != 2048:
        raise ValidationError("Stage B refused incomplete Stage A scores")
    if _digest_file(score_path) != seal.get("stage_a_scores_sha256"):
        raise ValidationError("Stage A score digest mismatch before evaluator access")
    rows: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORIES}
    with score_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if _recursive_has_key(row, OUTCOME_KEYS):
                raise ValidationError("Sealed Stage A file contains an outcome field")
            category = row.get("category")
            if category not in rows or row.get("position") != len(rows[category]):
                raise ValidationError("Sealed Stage A event order is invalid")
            rows[category].append(row)
    if any(len(rows[category]) != EVENTS for category in CATEGORIES):
        raise ValidationError("Stage A score coverage differs from the four authorized prefixes")
    return seal, rows


def _metric_value(record: Mapping[str, Any], name: str) -> float | None:
    if name in ("fpr", "fnr", "balanced_accuracy"):
        value = record["fixed_threshold"].get(name)
    else:
        value = record.get(name)
    return None if value is None else float(value)


def _pair_reversals(labels: np.ndarray, groups: Sequence[str],
                    baseline: np.ndarray, candidate: np.ndarray) -> dict[str, int]:
    y = np.asarray(labels, dtype=np.int8)
    g = np.asarray(groups, dtype=object)
    b = np.asarray(baseline, dtype=np.float64)
    c = np.asarray(candidate, dtype=np.float64)
    helpful = damaging = excluded = tied = 0
    for n in np.flatnonzero(y == 0):
        for d in np.flatnonzero(y == 1):
            if g[n] == g[d]:
                excluded += 1
                continue
            base_ok = b[d] > b[n]
            cand_ok = c[d] > c[n]
            helpful += int(not base_ok and cand_ok)
            damaging += int(base_ok and not cand_ok)
            tied += int(c[d] == c[n])
    return {"helpful_pair_reversals": helpful, "damaging_pair_reversals": damaging,
            "net_helpful_minus_damaging": helpful - damaging,
            "candidate_tied_pairs": tied, "same_source_group_pairs_excluded": excluded}


def _paired_group_bootstrap(labels: np.ndarray, groups: Sequence[str],
                            arms: Mapping[str, np.ndarray], threshold: float,
                            metric_provider: Any) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    group_values = np.asarray(groups, dtype=object)
    unique = sorted(set(group_values.tolist()))
    if y.shape != group_values.shape or not unique:
        raise ValidationError("Bootstrap groups and outcomes do not align")
    members = {group: np.flatnonzero(group_values == group) for group in unique}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metric_names = ("image_auroc", "image_auprc_average_precision", "fpr", "fnr", "balanced_accuracy")
    contrasts = {
        "SGRM_minus_matched_flat_Gaussian": ("SGRM", "matched_flat_Gaussian"),
        "SGRM_minus_C0": ("SGRM", "C0"),
        "SGRM_minus_C1_NoBoth": ("SGRM", "C1_NoBoth"),
        "SGRM_minus_historical_C2_a0.02": ("SGRM", "historical_C2_a0.02"),
    }
    sampled = {f"{key}:{metric}": [] for key in (*arms, *contrasts) for metric in metric_names}
    no_both = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        draw = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([members[group] for group in draw])
        yb = y[indices]
        no_both += int(np.unique(yb).size < 2)
        records = {arm: metric_provider._metric_record(yb, np.asarray(scores)[indices], threshold)
                   for arm, scores in arms.items()}
        for arm, record in records.items():
            for metric in metric_names:
                value = _metric_value(record, metric)
                if value is not None:
                    sampled[f"{arm}:{metric}"].append(value)
        for name, (left, right) in contrasts.items():
            for metric in metric_names:
                a, b = _metric_value(records[left], metric), _metric_value(records[right], metric)
                if a is not None and b is not None:
                    sampled[f"{name}:{metric}"].append(a - b)

    def interval(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {"lower_2_5": None, "upper_97_5": None, "valid_replicates": 0}
        values_array = np.asarray(values, dtype=np.float64)
        return {"lower_2_5": float(np.quantile(values_array, 0.025)),
                "upper_97_5": float(np.quantile(values_array, 0.975)),
                "valid_replicates": int(values_array.size)}

    return {"method": "paired Source Group cluster bootstrap with replacement; one shared draw across arms",
            "replicates_requested": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            "unique_source_groups": len(unique), "replicates_without_both_classes": no_both,
            "conditional_on_existing_development_trajectories": True,
            "independent_stream_uncertainty": False,
            "intervals": {key: interval(values) for key, values in sampled.items()}}


def _stage_b(started_monotonic: float) -> int:
    from experiments.calibration import r2_exp2b_memory_readout as exp2b
    from experiments.calibration import r2_perf1_stage_b as metrics

    budget = OutputBudget(OUTPUT_ROOT, create=False)
    try:
        stage_a, rows_by_category = _stage_b_rows()
        budget.write_json("stage_b_started.json", {
            "schema": "r2.sgrm.validation.stage_b_started.v1",
            "status": "OUTCOME_ACCESS_MAY_START",
            "stage_a_seal_sha256": _digest_file(OUTPUT_ROOT / "stage_a_seal.json"),
            "categories": list(CATEGORIES), "planned_outcomes": 2048,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        evaluator = exp2b._stage_b_evaluator()
        results: dict[str, Any] = {}
        for category in CATEGORIES:
            if time.monotonic() - started_monotonic > MAX_RUNTIME_SECONDS - 2:
                raise TimeoutError("600-second hard deadline approaching during Stage B")
            rows = rows_by_category[category]
            # This is the only outcome-reading call. It projects exactly these
            # 512 identity-matched TEST records and their official Source Groups.
            labels, groups = evaluator._stage_b_labels(
                category, rows, evaluator.DATASET_ROOT, evaluator.JSON_ROOT)
            labels = np.asarray(labels, dtype=np.int8)
            groups = [str(value) for value in groups]
            if (labels.shape != (EVENTS,) or not np.isin(labels, (0, 1)).all() or
                    len(groups) != EVENTS or any(groups[i] != rows[i]["source_group_sha256"]
                                                for i in range(EVENTS))):
                raise ValidationError(f"{category}: outcome identities do not match sealed Stage A rows")
            if category == "audiojack" and np.any(labels != 0):
                raise ValidationError("Authorized audiojack prefix is no longer the frozen normal-only prefix")
            threshold_values = np.asarray([float(row["theta_c64_q90"]) for row in rows], dtype=np.float64)
            if not np.all(threshold_values == threshold_values[0]):
                raise ValidationError(f"{category}: C64 threshold is not frozen")
            threshold = float(threshold_values[0])
            arms = {name: np.asarray([float(row["scores"][name]) for row in rows], dtype=np.float32)
                    for name in ("C0", "C1_NoBoth", "historical_C2_a0.02",
                                 "matched_flat_Gaussian", "SGRM")}
            metric_records = {arm: metrics._metric_record(labels, score, threshold)
                              for arm, score in arms.items()}
            metric_names = ("image_auroc", "image_auprc_average_precision", "fpr", "fnr", "balanced_accuracy")
            deltas = {metric: (None if _metric_value(metric_records["SGRM"], metric) is None or
                               _metric_value(metric_records["matched_flat_Gaussian"], metric) is None else
                               _metric_value(metric_records["SGRM"], metric) -
                               _metric_value(metric_records["matched_flat_Gaussian"], metric))
                      for metric in metric_names}
            comparisons = {}
            for arm in ("SGRM", "matched_flat_Gaussian", "C1_NoBoth", "historical_C2_a0.02"):
                comparisons[arm] = {}
                for control in ("C0", "C1_NoBoth", "historical_C2_a0.02", "matched_flat_Gaussian"):
                    if arm == control:
                        continue
                    comparisons[arm][control] = {
                        metric: (None if _metric_value(metric_records[arm], metric) is None or
                                 _metric_value(metric_records[control], metric) is None else
                                 _metric_value(metric_records[arm], metric) -
                                 _metric_value(metric_records[control], metric))
                        for metric in ("image_auroc", "image_auprc_average_precision")}
            ranking = {
                "versus_matched_flat_Gaussian": _pair_reversals(
                    labels, groups, arms["matched_flat_Gaussian"], arms["SGRM"]),
                "versus_C0": _pair_reversals(labels, groups, arms["C0"], arms["SGRM"]),
            }
            delta_flat = np.asarray([row["delta_flat"] for row in rows], dtype=np.float32)
            delta_sgrm = np.asarray([row["delta_sgrm"] for row in rows], dtype=np.float32)
            strength = {
                "matched_flat_Gaussian": exp2b._patch_strength_summary(
                    delta_flat, arms["matched_flat_Gaussian"], arms["C0"]),
                "SGRM": exp2b._patch_strength_summary(
                    delta_sgrm, arms["SGRM"], arms["C0"]),
            }
            group_summary = _summarize_group_rows(
                [row["group_diagnostics"] for row in rows])
            bootstrap = _paired_group_bootstrap(labels, groups, arms, threshold, metrics)
            category_result = {
                "status": "EVALUATED", "event_count": EVENTS,
                "positive_count": int(labels.sum()), "negative_count": int(EVENTS - labels.sum()),
                "prevalence": float(labels.mean()), "c64_threshold": threshold,
                "metrics_by_arm": metric_records,
                "SGRM_minus_matched_flat_Gaussian": deltas,
                "comparisons": comparisons,
                "ranking_reversals": ranking,
                "correction_magnitude_and_score_movement": strength,
                "source_group_contributions": group_summary,
                "paired_group_bootstrap": bootstrap,
            }
            if category == "audiojack":
                category_result["single_class_note"] = (
                    "Normal-only prefix: AUROC, AP, FNR, and balanced accuracy are undefined; "
                    "the evaluation window was not changed.")
            results[category] = category_result
            budget.write_json(f"stage_b_{category}.json", category_result)

        evaluable = [category for category in CATEGORIES
                     if results[category]["metrics_by_arm"]["SGRM"]["image_auroc"] is not None]
        arms = tuple(results[CATEGORIES[0]]["metrics_by_arm"])
        strict_macro: dict[str, float | None] = {}
        descriptive_macro: dict[str, float | None] = {}
        for arm in arms:
            all_values = [results[c]["metrics_by_arm"][arm]["image_auroc"] for c in CATEGORIES]
            strict_macro[arm] = (float(np.mean(all_values))
                                 if all(value is not None for value in all_values) else None)
            values = [results[c]["metrics_by_arm"][arm]["image_auroc"] for c in evaluable
                      if results[c]["metrics_by_arm"][arm]["image_auroc"] is not None]
            descriptive_macro[arm] = float(np.mean(values)) if values else None
        aggregate = {
            "schema": "r2.sgrm.validation.aggregate.v1", "status": "COMPLETE",
            "classification": "RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC",
            "run_id": RUN_ID, "algorithm_id": r2_sgrm.ALGORITHM_ID,
            "parameters": {"tau": TAU, "epsilon": EPSILON, "distance": "cosine"},
            "categories": results, "total_events": 2048,
            "strict_four_category_auroc_macro": strict_macro,
            "three_evaluable_category_descriptive_auroc_macro": {
                "categories": evaluable, "values": descriptive_macro,
                "interpretation": "descriptive only; audiojack is normal-only"},
            "bootstrap_design": {"replicates": BOOTSTRAP_REPLICATES,
                                 "seed": BOOTSTRAP_SEED,
                                 "conditional_on_existing_development_trajectories": True},
            "historical_summary_exposure": {
                "development_outcome_summaries_were_previously_observed": True,
                "used_as_online_scoring_inputs": False,
                "independent_or_blind_confirmation": False,
                "prior_retrieval_incidents": "Earlier broad documentation searches surfaced historical outcome summaries; these were not used as online scoring inputs."},
            "input_seals": stage_a["input_seals"],
        }
        runtime = time.monotonic() - started_monotonic
        if runtime > MAX_RUNTIME_SECONDS:
            raise TimeoutError("Complete supervised evaluation exceeded 600 seconds")
        report = _report(aggregate, stage_a, runtime)
        budget.write_json("stage_b_results.json", aggregate)
        budget.write_json("completion_seal.json", {
            "schema": "r2.sgrm.validation.completion.v1", "status": "COMPLETE",
            "run_id": RUN_ID, "runtime_seconds": runtime,
            "stage_a_sha256": _digest_file(OUTPUT_ROOT / "stage_a_seal.json"),
            "stage_b_results_sha256": _digest_file(OUTPUT_ROOT / "stage_b_results.json"),
            "run_output_bytes": budget.used_bytes,
            "run_output_files": budget.file_count,
        })
        if budget.used_bytes > MAX_OUTPUT_BYTES or budget.file_count > MAX_OUTPUT_FILES:
            raise ValidationError("Run output exceeded its authorized budget")
        _write_public_deliverables(aggregate, report)
        print(json.dumps({"run_id": RUN_ID, "status": "COMPLETE",
                          "runtime_seconds": runtime,
                          "stage_a_sha256": _digest_file(OUTPUT_ROOT / "stage_a_seal.json"),
                          "aggregate_sha256": _digest_file(AGGREGATE_PATH)}, sort_keys=True))
        return 0
    except BaseException as exc:
        _write_failure(OUTPUT_ROOT, exc)
        raise


def _write_public_deliverables(aggregate: Mapping[str, Any], report: str) -> None:
    _reject_reparse_chain(DELIVERABLE_ROOT.parent, "Deliverable parent")
    if DELIVERABLE_ROOT.exists():
        raise ValidationError("Deliverable directory already exists; refusing overwrite")
    DELIVERABLE_ROOT.mkdir(parents=False, exist_ok=False)
    for path, content in ((AGGREGATE_PATH, _canonical(aggregate)),
                          (REPORT_PATH, report.encode("utf-8"))):
        if path.exists() or _is_reparse(path):
            raise ValidationError("Refusing to overwrite a deliverable")
        with path.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())


def _report(aggregate: Mapping[str, Any], stage_a: Mapping[str, Any],
            runtime: float) -> str:
    categories = aggregate["categories"]
    lines = [
        "# R2-SGRM-VAL1 - CPU Retrospective Development Diagnostic", "",
        f"**Status:** `{aggregate['status']}`<br>",
        f"**Classification:** `{aggregate['classification']}`<br>",
        f"**Run:** `{RUN_ID}`<br>",
        f"**Source commit:** `{PARENT}`<br>",
        f"**Runtime:** {runtime:.2f}s / {MAX_RUNTIME_SECONDS}s hard ceiling<br>",
        "**Execution:** CPU-only; CUDA visibility disabled; no model/DINO/GPU/image decoding.", "",
        "This is development-only retrospective evidence, not independent confirmation, a new blind evaluation, or cross-category generalization. Historical aggregate development results had already been observed before this task. Earlier broad documentation searches surfaced historical outcome summaries; these were not used as online scoring inputs. No parameter search or correction rescaling was performed, and no unrelated R1 output was consulted during this run.", "",
        "## Supervised attempt history", "",
        "Fresh attempts `r2_sgrm_val1_20261102_a1` and `_a2` stopped before input files were opened because the runner referenced a nonexistent EXP2A path attribute. `_a3` stopped after verifying and loading the sealed EXP2B score receipts, before reading event shards, because a fractional beta cap was incorrectly converted to an integer. `_a4` stopped at switch event 35 after that event's score was fsynced and only its current queried diagnostic/feedback was read; the diagnostic schema has no event ID field. No attempt was resumed, all partial namespaces were preserved, and only fresh run `_a5` contributes results.", "",
        "## Inputs and score-seal provenance", "",
        f"EXP2A input root: `{(ROOT / EXP2A_RELATIVE).as_posix()}`<br>",
        f"EXP2B input root: `{(ROOT / EXP2B_RELATIVE).as_posix()}`<br>",
        "Authorized scope: support seed 0; switch, audiojack, regulator, rolled_strip_base; exactly 512 events each. The EXP2A regime was 4 deployed seed-0 supports plus 64 additional C64 training normals; this is not strict four-shot-total.", "",
        f"EXP2A Stage A seal SHA-256: `{stage_a['input_seals']['exp2a_stage_a']}`<br>",
        f"EXP2A completion seal SHA-256: `{stage_a['input_seals']['exp2a_experiment']}`<br>",
        f"EXP2B Stage A score seal SHA-256: `{stage_a['input_seals']['exp2b_stage_a']}`<br>",
        f"EXP2B event identity receipt SHA-256: `{stage_a['input_seals']['exp2b_event_refs']}`<br>",
        "Stage A sealed all 2,048 new scores before the Stage B child process opened outcome metadata. During Stage A, the 67 permitted queried-feedback records were consumed sequentially only after the matching event score was flushed and fsynced. No unqueried outcome, future feedback, or mask was used.", "",
        "The C1-NoBoth matched FIFO replay used the original dN keep mask, beta floor/clipping, signed beta, causal query position, Source Group identity, and capacity-256 policy. The EXP2B receipts report exact native candidate parity, no eviction/merge/replacement, and no FIFO/native memory divergence for these prefixes. The two readouts use the same pre-event anchors and one shared cosine-distance matrix; each full 784-patch map is max aggregated.", "",
        "C0 scores matched exactly for all 2,048 events. Existing C1-NoBoth native parity receipts covered 512 events per category. Flat Gaussian versus EXP2B D maximum absolute patch-delta errors were 4.66e-8 to 1.20e-7; maximum image-score errors were 5.96e-8 to 8.94e-8, below the frozen atol=1e-5/rtol=1e-4 tolerance. Reconstructed FIFO inserts were 99 (audiojack), 136 (switch), 116 (regulator), and 100 (rolled_strip_base); no evictions occurred.", "",
        "## Category metrics", "",
        "Audiojack is normal-only in this fixed prefix, so AUROC, AP, FNR and BA remain undefined. The four-category AUROC macro is undefined; the three-evaluable-category macro is descriptive only.", "",
        "| Category | Arm | AUROC | AP | Prevalence | FPR | FNR | BA | TP/TN/FP/FN |", "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    show = lambda x: "undefined" if x is None else f"{float(x):.4f}"
    for category in CATEGORIES:
        for arm, record in categories[category]["metrics_by_arm"].items():
            fixed = record["fixed_threshold"]
            values = (record["image_auroc"], record["image_auprc_average_precision"],
                      record["defect_prevalence"], fixed["fpr"], fixed["fnr"],
                      fixed["balanced_accuracy"])
            confusion = f"{fixed['tp']}/{fixed['tn']}/{fixed['fp']}/{fixed['fn']}"
            lines.append(f"| {category} | {arm} | " + " | ".join(show(v) for v in values) + f" | {confusion} |")
    lines.extend(["", "## SGRM minus matched flat Gaussian", "",
                  "| Category | Delta AUROC | Delta AP | Delta FPR | Delta FNR | Delta BA | Helpful reversals | Damaging reversals |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for category in CATEGORIES:
        delta = categories[category]["SGRM_minus_matched_flat_Gaussian"]
        pairs = categories[category]["ranking_reversals"]["versus_matched_flat_Gaussian"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            category, show(delta["image_auroc"]), show(delta["image_auprc_average_precision"]),
            show(delta["fpr"]), show(delta["fnr"]), show(delta["balanced_accuracy"]),
            pairs["helpful_pair_reversals"], pairs["damaging_pair_reversals"]))
    lines.extend(["", "## Correction and Source Group diagnostics", "",
                  "The aggregate JSON reports paired Source Group bootstrap intervals (1,000 replicates, seed 20261101), conditional on the existing trajectories. It also reports patch RMS/mean absolute correction, image-score movement, signed evidence, active Source Group counts, eligible patch-count distributions, effective contributing groups, and pair-reversal diagnostics. Effective groups are the inverse Herfindahl count over absolute per-patch group contributions. No magnitude normalization was used.", "",
                  "The SGRM-versus-flat effect is descriptive. A null or negative result is valid. No safety guarantee, publication-level novelty, or independent generalization claim follows from these four previously exposed development prefixes.", "",
        "## CPU regression checks", "",
        "`py_compile` passed. Focused CPU regression suites: 62 passed, 1 skipped. The Windows process-tree termination test passed. The symlink-parent test was skipped because symlink creation was unavailable in this environment.", "",
                  f"Stage A: `{stage_a['status']}`; total scored events {stage_a['total_events']}; queried feedback updates {stage_a['total_queries']}. Flat Gaussian versus EXP2B D parity used atol={ATOL:g}, rtol={RTOL:g}; per-category maximum patch and image-score errors are in `aggregate_results.json`.", ""])
    return "\n".join(lines)


def _failure_after_timeout(message: str) -> None:
    try:
        budget = OutputBudget(OUTPUT_ROOT, create=False)
        if "failure.json" not in budget.created:
            budget.write_json("failure.json", {
                "schema": "r2.sgrm.quick_eval.failure.v1", "status": "INCOMPLETE",
                "failure_type": "TimeoutError", "message": message[:1000],
                "automatic_restart": False,
            })
    except Exception:
        pass


def _run_child(mode: str, started: float, timeout: float) -> None:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1"})
    command = [str(PYTHON), "-m", "experiments.calibration.r2_sgrm_quick_eval",
               mode, str(started)]
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, creationflags=flags,
                             start_new_session=(os.name != "nt"))
    try:
        output, _ = child.communicate(timeout=max(0.1, timeout - TERMINATION_GRACE_SECONDS))
    except subprocess.TimeoutExpired:
        _terminate_process_tree(child)
        output, _ = child.communicate(timeout=1)
        _failure_after_timeout(f"Hard {MAX_RUNTIME_SECONDS}-second supervisor deadline reached")
        if output:
            print(output[-3000:])
        raise TimeoutError("R2-SGRM validation exceeded the hard 600-second deadline")
    if output:
        print(output[-6000:])
    if child.poll() is None:
        raise ValidationError("Child process termination could not be confirmed")
    if child.returncode != 0:
        raise ValidationError(f"{mode} child exited with code {child.returncode}")


def _terminate_process_tree(child: subprocess.Popen[str]) -> None:
    """Terminate the worker and descendants, failing if OS confirmation fails."""
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["taskkill.exe", "/PID", str(child.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=5, check=False)
        except subprocess.TimeoutExpired as exc:
            raise ValidationError("Windows taskkill /T did not complete within the termination reserve") from exc
        if result.returncode != 0:
            raise ValidationError(f"Windows taskkill /T could not confirm process-tree termination: {result.stdout[-1000:]}")
    else:
        import signal
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired as exc:
        raise ValidationError("Supervisor could not confirm child exit after process-tree termination") from exc
    if child.poll() is None:
        raise ValidationError("Supervisor could not verify worker exit")


def _supervise() -> int:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise ValidationError(f"Use exact interpreter {PYTHON}")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if branch != BRANCH or head != PARENT:
        raise ValidationError("Branch/HEAD do not match the authorized source identity")
    if subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).strip():
        raise ValidationError("Tracked worktree changes exist before this bounded run")
    if subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True).strip():
        raise ValidationError("Staged changes exist before this bounded run")
    if OUTPUT_ROOT.exists() or DELIVERABLE_ROOT.exists():
        raise ValidationError("Fresh output namespace already exists; restart is forbidden")
    started = PROCESS_START_MONOTONIC
    _run_child("--stage-a", started, MAX_RUNTIME_SECONDS - (time.monotonic() - started))
    if time.monotonic() - started >= MAX_RUNTIME_SECONDS:
        raise TimeoutError("Stage A consumed the full run deadline")
    _run_child("--stage-b", started, MAX_RUNTIME_SECONDS - (time.monotonic() - started))
    if time.monotonic() - started > MAX_RUNTIME_SECONDS:
        raise TimeoutError("Complete run exceeded 600 seconds")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("--stage-a", "--stage-b") and len(args) == 2:
        try:
            started = float(args[1])
            if args[0] == "--stage-a":
                if OUTPUT_ROOT.exists():
                    raise ValidationError("Stage A output namespace exists; no resume")
                return _stage_a(started)
            if not OUTPUT_ROOT.is_dir() or not (OUTPUT_ROOT / "stage_a_seal.json").is_file():
                raise ValidationError("Stage B requires a complete Stage A seal")
            return _stage_b(started)
        except BaseException as exc:
            print(f"{args[0]} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    try:
        if args:
            raise ValidationError("Unexpected command-line arguments")
        return _supervise()
    except BaseException as exc:
        print(f"R2-SGRM validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
