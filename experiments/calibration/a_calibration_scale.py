#!/usr/bin/env python3
"""Experiment A: calibration scale, initialization, and state-dependence audit.

This namespace is isolated from historical experiments.  The module keeps
protocol math, group-level data roles, causal gates, and the Stage-A artifact
contract in one auditable place.  It uses the repository's native v22/VisionAD
scorer and never supplies a nearest-neighbour fallback.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_ROOT = Path(r"G:\dataset\realiad")
JSON_ROOT = DATASET_ROOT / "jsons" / "realiad_jsons"
RUNS_ROOT = ROOT / "outputs" / "calibration"
REPORT_ROOT = ROOT / "docs" / "calibration" / "audit" / "a_calibration_scale_20261106"
CLASSES = ("audiojack", "regulator", "switch", "rolled_strip_base")
CONFIRMATION_CLASSES = frozenset(("mint", "pcb"))
MASK_SUFFIXES = ("_mask.png", "_mask.jpg", "_mask.tif", "_mask.tiff")
PROTOCOL_ID = "a_calibration_scale_20261106.v1"
AUTHORIZATION_ID = "EXPERIMENT_A_AUTHORIZATION_20261106"
MIN_FREE_BYTES = 30 * 1024**3
MAX_RETAINED_BYTES = 4 * 1024**3
SMOKE_DEADLINE_SECONDS = 15 * 60
CATEGORY_DEADLINE_SECONDS = 120 * 60
CAMPAIGN_DEADLINE_SECONDS = 8 * 60 * 60


class ExperimentAError(RuntimeError):
    """A fail-closed Experiment A gate or protocol violation."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def higher_index(n: int, q: float = 0.90) -> int:
    """Zero-based higher empirical quantile index registered for Experiment A."""
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("q must be in [0, 1]")
    return int(math.ceil(float(q) * (int(n) - 1)))


def higher_quantile(values: Sequence[float], q: float = 0.90) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 1 or not np.isfinite(x).all():
        raise ValueError("quantile input must be a nonempty finite vector")
    return float(np.partition(x, higher_index(len(x), q))[higher_index(len(x), q)])


def sample_mean_std(values: Sequence[float]) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("sample statistics need at least two finite observations")
    return float(x.mean()), float(x.std(ddof=1))


def gaussian_threshold(values: Sequence[float], multiplier: float) -> float:
    mean, std = sample_mean_std(values)
    return float(mean + float(multiplier) * std)


def threshold_methods(calibration: Mapping[str, Sequence[float]],
                      support_self: Sequence[float],
                      support_loso: Sequence[float]) -> dict[str, float]:
    """Registered A1 threshold values; fixed 0.5 is on the raw score scale."""
    result = {"SELF": gaussian_threshold(support_self, 2.5),
              "LOSO": gaussian_threshold(support_loso, 2.5),
              "FIXED05": 0.5}
    for name in ("C32", "C64", "C128", "Cmax-G", "Cmax-V"):
        if name in calibration:
            result[f"Q90:{name}"] = higher_quantile(calibration[name])
    for name in ("C64", "Cmax-G"):
        if name in calibration:
            result[f"GAUSSIAN-Q90:{name}"] = gaussian_threshold(
                calibration[name], 1.2815515655)
            result[f"GAUSSIAN-2.5:{name}"] = gaussian_threshold(
                calibration[name], 2.5)
    return result


def metrics_for_threshold(labels: Sequence[int], scores: Sequence[float],
                          threshold: float) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or not len(y):
        raise ValueError("labels and scores must be aligned nonempty vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("invalid labels or scores")
    pred = s >= float(threshold)
    normal = y == 0
    defect = y == 1
    fp = int(np.sum(pred & normal))
    fn = int(np.sum((~pred) & defect))
    n0, n1 = int(normal.sum()), int(defect.sum())
    fpr = fp / n0 if n0 else math.nan
    fnr = fn / n1 if n1 else math.nan
    ber = (fpr + fnr) / 2 if n0 and n1 else math.nan
    ba = 1 - ber if n0 and n1 else math.nan
    return {"n": len(y), "n_normal": n0, "n_defect": n1,
            "fp": fp, "fn": fn, "fpr": fpr, "fnr": fnr,
            "ber": ber, "ba": ba}


def window_metrics(labels: Sequence[int], scores: Sequence[float],
                   threshold: float, width: int = 512) -> list[dict[str, Any]]:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) != len(s) or width < 1:
        raise ValueError("invalid window inputs")
    result = []
    for start in range(0, len(y) - width + 1, width):
        met = metrics_for_threshold(y[start:start + width],
                                    s[start:start + width], threshold)
        result.append({"start": start, "end_exclusive": start + width,
                       **met, "status": "VALID" if met["n_normal"] and met["n_defect"]
                       else "UNAVAILABLE_SINGLE_CLASS"})
    return result


def auc_rank(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Tie-aware Mann-Whitney AUROC, implemented without sklearn dependencies."""
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return math.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and s[order[j]] == s[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def higher_order_membership(train: Sequence[Any], supports: Mapping[str, Any],
                            historical_c64: Mapping[str, Any], *,
                            category: str) -> dict[str, Any]:
    """Build frozen S/C/H/calibration/reference roles from TRAIN-normal records.

    Additional role identities are chosen by a deterministic SHA-256 ordering,
    with the namespace and rule frozen in the contract before Stage B.
    """
    if category not in CLASSES:
        raise ExperimentAError(f"Unauthorized category: {category}")
    by_group: dict[str, list[Any]] = {}
    for record in train:
        by_group.setdefault(record.sample_key, []).append(record)
    normals = {key: rows for key, rows in by_group.items()
               if rows and all(bool(row.is_good) for row in rows)}
    support_groups = {str(k): tuple(map(str, v["groups"]))
                      for k, v in supports.items() if str(k) in {"0", "1"}}
    union_support = set().union(*(set(v) for v in support_groups.values()))
    c64_groups = tuple(map(str, historical_c64["groups"]))
    if len(c64_groups) != 64 or len(set(c64_groups)) != 64:
        raise ExperimentAError(f"{category}: historical C64 identity is malformed")
    if set(c64_groups) & union_support:
        raise ExperimentAError(f"{category}: Support/C64 group overlap")
    if not (union_support | set(c64_groups)) <= set(normals):
        raise ExperimentAError(f"{category}: Support/C64 includes non-normal TRAIN group")

    def group_view(key: str, namespace: str) -> Any:
        rows = normals[key]
        return min(rows, key=lambda row: (
            hashlib.sha256((namespace + "\0" + row.relative_id).encode()).hexdigest(),
            row.relative_id))

    reserve_candidates = sorted(set(normals) - union_support - set(c64_groups))
    if len(reserve_candidates) < 32 + 128:
        raise ExperimentAError(f"{category}: insufficient disjoint TRAIN-normal groups")
    h_groups = tuple(sorted(reserve_candidates,
                            key=lambda key: hashlib.sha256(
                                f"{PROTOCOL_ID}\0H\0{category}\0{key}".encode()).hexdigest())[:32])
    h_set = set(h_groups)
    eligible = sorted(set(normals) - union_support - h_set)
    if not set(c64_groups) <= set(eligible):
        raise ExperimentAError(f"{category}: H overlaps historical C64")

    # Draw 0 is the unchanged historic reference. Draws 1 and 2 are explicitly
    # distinct deterministic replicates, selected from the same group universe.
    draws: dict[str, dict[str, tuple[str, ...]]] = {}
    for draw in range(3):
        if draw == 0:
            c64 = c64_groups
        else:
            pool = [key for key in eligible if key not in set(c64_groups)]
            ordered = sorted(pool, key=lambda key: hashlib.sha256(
                f"{PROTOCOL_ID}\0C64\0{category}\0{draw}\0{key}".encode()).hexdigest())
            if len(ordered) < 128:
                raise ExperimentAError(f"{category}: insufficient groups for nested draw {draw}")
            c64 = tuple(ordered[:64])
        remaining = [key for key in eligible if key not in set(c64)]
        extension = tuple(sorted(remaining, key=lambda key: hashlib.sha256(
            f"{PROTOCOL_ID}\0C128\0{category}\0{draw}\0{key}".encode()).hexdigest())[:64])
        if len(extension) != 64:
            raise ExperimentAError(f"{category}: C128 extension is incomplete")
        c128 = tuple(c64) + extension
        draws[str(draw)] = {
            "C32": tuple(c64[:32]),
            "C64": tuple(c64),
            "C128": c128,
            "Cmax-G": tuple(eligible),
        }
    maxv_rows = [row for key in eligible for row in normals[key]]
    h_rows = [group_view(key, f"{PROTOCOL_ID}/H-view/{category}") for key in h_groups]
    ref_pool = sorted(set(normals) - union_support - h_set - set(c64_groups))
    if not ref_pool:
        raise ExperimentAError(f"{category}: empty A3b reference-building pool")
    ref_support = tuple(sorted(ref_pool, key=lambda key: hashlib.sha256(
        f"{PROTOCOL_ID}\0A3b-4S\0{category}\0{key}".encode()).hexdigest())[:4])
    roles = {
        "schema": "a_calibration_scale.roles.v1",
        "category": category,
        "support_groups": support_groups,
        "support_union_groups": sorted(union_support),
        "historical_C64_groups": list(c64_groups),
        "H_groups": list(h_groups),
        "H_images": [row.relative_id for row in h_rows],
        "eligible_calibration_groups": list(eligible),
        "draws": {d: {k: list(v) for k, v in x.items()} for d, x in draws.items()},
        "Cmax_V_images": [row.relative_id for row in maxv_rows],
        "Cmax_V_group_count": len({row.sample_key for row in maxv_rows}),
        "Cmax_V_image_count": len(maxv_rows),
        "Cmax_V_views_per_group": {
            str(key): len(normals[key]) for key in eligible},
        "A3b_reference_building_groups": ref_pool,
        "A3b_reference_support_groups": list(ref_support),
        "selection_rule": "sha256-namespaced-order-v1; view=min SHA256 per official group",
        "disjointness": {
            "support_vs_historical_C64": not bool(union_support & set(c64_groups)),
            "support_vs_H": not bool(union_support & h_set),
            "historical_C64_vs_H": not bool(set(c64_groups) & h_set),
            "H_vs_calibration": not bool(h_set & set(eligible)),
            "H_vs_A3b_reference": not bool(h_set & set(ref_pool)),
            "historical_C64_vs_A3b_reference": not bool(set(c64_groups) & set(ref_pool)),
        },
    }
    for draw_id, draw in draws.items():
        if not (set(draw["C32"]) <= set(draw["C64"]) <= set(draw["C128"]) <= set(draw["Cmax-G"])):
            raise ExperimentAError(f"{category}: non-nested calibration draw {draw_id}")
    return roles


def loso_reference_ids(support_ids: Sequence[str],
                      descendants: Mapping[str, Sequence[str]],
                      held_out_id: str) -> tuple[str, ...]:
    """List all augmentation descendants from the three non-held-out Supports."""
    if len(support_ids) != 4 or len(set(support_ids)) != 4 or held_out_id not in support_ids:
        raise ValueError("LOSO requires four unique Supports and a held-out member")
    expected = set(support_ids) - {held_out_id}
    rows: list[str] = []
    for support_id in support_ids:
        if support_id == held_out_id:
            continue
        rows.append(support_id)
        rows.extend(str(v) for v in descendants.get(support_id, ()))
    if held_out_id in rows or len(expected) != 3:
        raise ExperimentAError("LOSO reference contains the held-out Support")
    return tuple(rows)


def verify_disjoint_roles(*roles: Iterable[str]) -> bool:
    seen: set[str] = set()
    for role in roles:
        current = set(role)
        if seen & current:
            raise ExperimentAError("Role groups are not mutually disjoint")
        seen |= current
    return True


def require_cmax_complete(status: str) -> None:
    if status != "COMPLETE_VALID":
        raise ExperimentAError("Cmax cannot be silently substituted by C128 or a sparse snapshot")


class QueryOnlyLabelGate:
    """Causal test double for the registered post-seal queried-label boundary."""
    def __init__(self, labels: Sequence[int], query_positions: Sequence[int]):
        self._labels = tuple(int(v) for v in labels)
        self.query_positions = tuple(map(int, query_positions))
        if (any(v not in (0, 1) for v in self._labels) or
                tuple(sorted(set(self.query_positions))) != self.query_positions or
                any(v < 0 or v >= len(self._labels) for v in self.query_positions)):
            raise ValueError("Invalid synthetic label/query tape")
        self.cursor = 0
        self.sealed_positions: set[int] = set()
        self.disclosed: list[int] = []

    def seal_prediction(self, position: int) -> None:
        if not 0 <= int(position) < len(self._labels):
            raise ValueError("Invalid event position")
        self.sealed_positions.add(int(position))

    def reveal(self, position: int) -> int:
        if self.cursor >= len(self.query_positions) or int(position) != self.query_positions[self.cursor]:
            raise ExperimentAError("Only the next fixed query may reveal a label")
        if int(position) not in self.sealed_positions:
            raise ExperimentAError("Current event prediction must be sealed before label reveal")
        label = self._labels[int(position)]
        self.disclosed.append(int(position))
        self.cursor += 1
        return label


def verify_query_identity(lhs: Sequence[str], rhs: Sequence[str]) -> bool:
    if tuple(lhs) != tuple(rhs):
        raise ExperimentAError("Paired query identity tapes differ")
    return True


def audit_threshold_auroc(labels: Sequence[int], scores: Sequence[float],
                           thresholds: Sequence[float]) -> dict[str, Any]:
    values = [auc_rank(labels, scores) for _ in thresholds]
    if values and any((math.isnan(values[0]) != math.isnan(value)) or
                      (not math.isnan(value) and value != values[0]) for value in values[1:]):
        raise ExperimentAError("Threshold-only A1 arms changed AUROC")
    return {"auroc_by_arm": values, "identical": True,
            "score_sha256": sha256_bytes(np.asarray(scores, dtype="<f8").tobytes())}


def assert_no_legacy_paths(changed_paths: Iterable[str]) -> None:
    allowed = {"experiments/calibration/a_calibration_scale.py",
               "experiments/calibration/a_calibration_scale_helpers.py",
               "tests/calibration/test_a_calibration_scale.py"}
    unexpected = sorted(path.replace("\\", "/") for path in changed_paths
                        if path.replace("\\", "/") not in allowed)
    if unexpected:
        raise ExperimentAError(f"Unexpected legacy or unrelated changes: {unexpected}")


def b3_step(theta: float, candidate: float, *, version: int,
            last_version: int | None, step: float = 0.01) -> tuple[float, int | None, bool]:
    """Apply one clipped B3 update per new scoring-state version."""
    if last_version is not None and int(version) == int(last_version):
        return float(theta), last_version, False
    if last_version is not None and int(version) < int(last_version):
        raise ExperimentAError("B3 state version moved backwards")
    updated = float(np.clip(float(candidate), float(theta) - step, float(theta) + step))
    return updated, int(version), True


@dataclass
class SharedWriterState:
    """Threshold-independent shared writer input used only by A2 readout arms."""
    writer_theta: float
    writer_sigma: float
    memory: list[str]
    state_version: int = 0
    write_log: list[tuple[int, str]] | None = None

    def __post_init__(self) -> None:
        if self.write_log is None:
            self.write_log = []

    def verified_write(self, position: int, source_digest: str) -> None:
        self.memory.append(source_digest)
        self.state_version += 1
        self.write_log.append((position, source_digest))


class A2Readout:
    """Per-arm calibration state that cannot mutate the shared writer."""
    def __init__(self, theta: float, sigma: float):
        self.theta = float(theta)
        self.sigma = max(0.02, float(sigma))
        self.last_version: int | None = None
        self.history: list[dict[str, Any]] = []

    def update(self, *, version: int, calibration_scores: Sequence[float]) -> bool:
        proposed = higher_quantile(calibration_scores)
        self.theta, self.last_version, changed = b3_step(
            self.theta, proposed, version=version, last_version=self.last_version)
        self.sigma = max(0.02, sample_mean_std(calibration_scores)[1])
        self.history.append({"version": int(version), "theta": self.theta,
                             "sigma": self.sigma, "updated": changed})
        return changed


@dataclass
class IndependentTrajectory:
    """Separate A3 states; calibration is allowed to affect writer inputs."""
    name: str
    theta: float
    sigma: float
    memory: list[str]
    correction: list[str]
    predictions: list[int]

    def prequential_prediction(self, score: float) -> int:
        prediction = int(float(score) >= self.theta)
        self.predictions.append(prediction)
        return prediction

    def accept_verified_feedback(self, *, position: int, label: int,
                                 source_digest: str, correction_digest: str) -> None:
        if label not in (0, 1):
            raise ExperimentAError("feedback label must be binary")
        if label == 0:
            self.memory.append(f"{position}:{source_digest}")
        self.correction.append(f"{position}:{correction_digest}:{label}")


def require_stage_a_seal(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ExperimentAError("Stage B is prohibited before the Stage A seal exists")
    seal = json.loads(path.read_text(encoding="utf-8"))
    if seal.get("schema") != "a_calibration_scale.stage_a_seal.v1":
        raise ExperimentAError("Incompatible Stage A seal schema")
    payload = dict(seal)
    expected = payload.pop("seal_sha256", None)
    if expected != sha256_bytes(canonical_json(payload)):
        raise ExperimentAError("Stage A seal digest mismatch")
    if seal.get("stage") != "A_SEALED" or seal.get("label_access") != "NONE":
        raise ExperimentAError("Stage A artifact is not sealed label-blind evidence")
    return seal


def create_stage_a_seal(path: Path, *, run_id: str,
                        input_digests: Mapping[str, str],
                        score_digests: Mapping[str, str]) -> dict[str, Any]:
    seal = {"schema": "a_calibration_scale.stage_a_seal.v1",
            "protocol_id": PROTOCOL_ID, "run_id": run_id,
            "stage": "A_SEALED", "label_access": "NONE",
            "input_digests": dict(sorted(input_digests.items())),
            "score_digests": dict(sorted(score_digests.items()))}
    seal["seal_sha256"] = sha256_bytes(canonical_json(seal))
    _write_once(Path(path), canonical_json(seal) + b"\n")
    return seal


def check_resume_identity(existing: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if canonical_json(dict(existing)) != canonical_json(dict(expected)):
        raise ExperimentAError("Resume refused: source/input identity differs")


def authorized_category(category: str) -> None:
    if category not in CLASSES or category in CONFIRMATION_CLASSES:
        raise ExperimentAError(f"Unauthorized category access: {category}")


def assert_no_masks_accessed(paths: Iterable[str | Path]) -> None:
    for value in paths:
        name = Path(value).name.lower()
        if any(name.endswith(suffix) for suffix in MASK_SUFFIXES) or "mask" in name:
            raise ExperimentAError(f"Mask path is prohibited: {value}")


def protected_file_audit() -> dict[str, Any]:
    """Ensure this implementation has not modified historical experiment files."""
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "--", "experiments", "tests", "docs/calibration"],
        cwd=ROOT, text=True).splitlines()
    staged = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--", "experiments", "tests", "docs/calibration"],
        cwd=ROOT, text=True).splitlines()
    protected = [p for p in changed + staged if p.startswith((
        "experiments/calibration/", "experiments/risk_safe/", "experiments/"))
        and not p.endswith("a_calibration_scale.py")]
    return {"changed_historical_paths": protected,
            "passed": len(protected) == 0}


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(str(path), flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def validate_run_id(run_id: str) -> str:
    if not run_id or len(run_id) > 64 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in run_id):
        raise ExperimentAError("run-id must contain 1-64 safe filename characters")
    return run_id


def run_dir(run_id: str) -> Path:
    return RUNS_ROOT / f"a_calibration_scale_{validate_run_id(run_id)}"


def _git_identity() -> dict[str, str]:
    return {
        "branch": subprocess.check_output(["git", "branch", "--show-current"],
                                          cwd=ROOT, text=True).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                        cwd=ROOT, text=True).strip(),
    }


def _source_fingerprints() -> dict[str, str]:
    paths = [
        "AGENTS.md", "experiments/calibration/a_calibration_scale.py",
        "experiments/calibration/stage05.py", "experiments/calibration/query_tape.py",
        "experiments/calibration/profiles.py", "experiments/calibration/native.py",
        "experiments/calibration/r1_matched_correction_pilot.py",
        "experiments/calibration/r3_fullstream_2x2.py",
        "experiments/calibration/r9_correction_mechanism_falsification.py",
        "experiments/calibration/r6_capacity1024_audiojack_runner.py",
        "experiments/p0_protocol_audit.py", "experiments/risk_safe/native_bridge.py",
        "experiments/risk_safe/preflight.py", "test_phase3_AIF_simplified_v22.py",
        "vmb_visionad_new.py", "v02_memory.py",
    ]
    return {rel: sha256_file(ROOT / rel) for rel in paths if (ROOT / rel).is_file()}


def data_preflight(dataset_root: Path = DATASET_ROOT,
                   json_root: Path = JSON_ROOT) -> dict[str, Any]:
    from experiments.calibration import stage05
    dataset_root, json_root = Path(dataset_root), Path(json_root)
    classes: dict[str, Any] = {}
    for category in CLASSES:
        authorized_category(category)
        train, test, json_sha = stage05.read_class_stream(
            dataset_root, json_root, category, test_labels=False)
        selection = stage05.select_supports(train)
        roles = higher_order_membership(
            train, selection["supports"], selection["c64"], category=category)
        path_errors = [r.image_path for r in train if not Path(r.image_path).is_file()]
        path_errors += [r.image_path for r in test if not Path(r.image_path).is_file()]
        if path_errors:
            raise ExperimentAError(f"{category}: missing image paths ({len(path_errors)})")
        test_groups = {row.sample_key for row in test}
        support_groups = set(roles["support_union_groups"])
        calibration_groups = set(roles["eligible_calibration_groups"])
        if test_groups & (support_groups | calibration_groups | set(roles["H_groups"])):
            raise ExperimentAError(f"{category}: TEST group overlap with TRAIN roles")
        stream_orders = _verify_registered_stream_orders(test, category)
        classes[category] = {
            "json_sha256": json_sha,
            "train_image_count": len(train),
            "train_group_count": len({r.sample_key for r in train}),
            "train_normal_image_count": sum(bool(r.is_good) for r in train),
            "test_image_count": len(test),
            "test_group_count": len(test_groups),
            "test_identity_sha256": sha256_bytes(canonical_json([
                {"position": r.position, "relative_id": r.relative_id,
                 "sample_key_sha256": sha256_bytes(r.sample_key.encode())}
                for r in test])),
            "role_manifest": roles,
            "registered_stream_orders": stream_orders,
            "support_manifest": selection["supports"],
            "historical_C64_manifest": selection["c64"],
            "train_test_group_disjoint": True,
            "all_source_paths_exist": True,
            "test_labels_read": False,
        }
    return {
        "schema": "a_calibration_scale.preflight.v1",
        "protocol_id": PROTOCOL_ID,
        "authorization_id": AUTHORIZATION_ID,
        "git": _git_identity(),
        "source_fingerprints": _source_fingerprints(),
        "data_root": str(dataset_root),
        "json_root": str(json_root),
        "classes": classes,
        "total_test_images": sum(v["test_image_count"] for v in classes.values()),
        "test_labels_read": False,
        "masks_read": False,
        "protected_file_audit": protected_file_audit(),
    }


def _verify_registered_stream_orders(test: Sequence[Any], category: str) -> dict[str, Any]:
    """Compare our shuffle0/11 maps with completed R9 label-blind score tapes."""
    from experiments.calibration import r7_shuffled_correction_4class as r7
    specs = {
        "P0": (0, f"{category}_shuffle0"),
        "P1": (11, f"{category}_shuffle11"),
        "P2": (0, f"{category}_support1_shuffle0"),
    }
    result = {}
    base = ROOT / "outputs/calibration/r9_correction_mechanism_20261103_a9/full"
    for condition, (seed, directory) in specs.items():
        rows = r7.deterministic_group_shuffle(test, seed=seed)
        expected_path = base / directory / "stage_a_scores.jsonl"
        if not expected_path.is_file():
            raise ExperimentAError(f"{category}/{condition}: registered stream identity receipt is absent")
        historical = [json.loads(line) for line in expected_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        current_ids = [(row.position, row.original_position, row.image_identity_sha256,
                        row.source_group_sha256) for row in rows]
        historical_ids = [(int(row["position"]), int(row["original_json_position"]),
                           str(row["image_identity_sha256"]),
                           str(row["source_group_sha256"])) for row in historical]
        if current_ids != historical_ids:
            raise ExperimentAError(f"{category}/{condition}: stream identity differs from R9 receipt")
        result[condition] = {
            "status": "EXACT_MATCH_R9_IDENTITY_MAP",
            "group_shuffle_seed": seed,
            "event_count": len(rows),
            "identity_map_sha256": sha256_bytes(canonical_json([
                {"position": int(row.position),
                 "original_position": int(row.original_position),
                 "image_identity_sha256": row.image_identity_sha256,
                 "source_group_sha256": row.source_group_sha256}
                for row in rows])),
            "reference": str(expected_path.relative_to(ROOT)),
        }
    return result


def _model_preflight() -> dict[str, Any]:
    import torch
    from experiments.calibration import r3_fullstream_2x2 as r3
    receipt = r3._get_verified_model_receipt()
    hub = Path(torch.hub.get_dir())
    source, hub_sha = __import__(
        "experiments.calibration.r1_matched_correction_pilot",
        fromlist=["_hub_source_sha256"])._hub_source_sha256(hub)
    checkpoint = hub / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
    profile = receipt["runtime_profile"]
    return {
        "backbone": "dinov2_vitl14_reg",
        "hub_source": source,
        "hub_source_tree_sha256": hub_sha,
        "checkpoint_sha256": sha256_file(checkpoint),
        "loaded_model_state_sha256": receipt["loaded_model_state_sha256"],
        "runtime_profile": profile,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": str(torch.__version__),
    }


def _resource_preflight() -> dict[str, Any]:
    import shutil as _shutil
    root = Path("E:/")
    usage = _shutil.disk_usage(root)
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            gpu = {"free_bytes": int(free), "total_bytes": int(total),
                   "name": torch.cuda.get_device_name(0)}
        else:
            gpu = {"available": False}
    except Exception as exc:
        gpu = {"error": f"{type(exc).__name__}: {exc}"}
    # The cache is category-sharded and removed only after successful validation.
    # Exact role counts feed the per-category estimate below.
    return {"experiment_drive": str(root), "free_bytes": int(usage.free),
            "minimum_free_bytes": MIN_FREE_BYTES,
            "minimum_free_space_passed": usage.free >= MIN_FREE_BYTES,
            "gpu": gpu,
            "retained_output_cap_bytes": MAX_RETAINED_BYTES,
            "test_feature_dtype": "float32",
            "test_feature_shape_per_image": [784, 4096],
            "test_feature_bytes_per_image": 784 * 4096 * 4,
            "full_four_category_test_cache_bytes": 15295 * 784 * 4096 * 4,
            "cache_policy": "category-sharded raw float32; temporary experiment-owned namespace"}


def full_preflight(dataset_root: Path = DATASET_ROOT,
                   json_root: Path = JSON_ROOT) -> dict[str, Any]:
    audit = data_preflight(dataset_root, json_root)
    audit["model"] = _model_preflight()
    audit["resources"] = _resource_preflight()
    audit["historical_score_tape_note"] = {
        "status": "NOT_USED_AS_PRIMARY_RESULT",
        "reason": "Stage05 score-tape source hashes for stage05.py and stage05_runner.py are stale; new A1 scoring will use current native path",
    }
    audit["status"] = "PREFLIGHT_PASS" if (
        audit["protected_file_audit"]["passed"] and
        audit["resources"]["minimum_free_space_passed"] and
        audit["model"]["checkpoint_sha256"] ==
        "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
    ) else "PREFLIGHT_HOLD"
    return audit


def _assert_test_score_parity(actual: Sequence[float], expected: Sequence[float]) -> dict[str, Any]:
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ExperimentAError("Static score parity vectors are malformed")
    delta = np.abs(a - b)
    tol = 1e-5 + 1e-4 * np.maximum(np.abs(a), np.abs(b))
    passed = bool(np.all(delta <= tol))
    if not passed:
        raise ExperimentAError(f"Native score parity failed; max_abs={float(delta.max())}")
    return {"passed": True, "n": len(a), "atol": 1e-5, "rtol": 1e-4,
            "max_abs_difference": float(delta.max(initial=0.0)),
            "reference_source": "Stage05 score tape, direct native parity only"}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_once(path, canonical_json(payload) + b"\n")


def _create_run(run_id: str) -> Path:
    path = run_dir(run_id)
    path.mkdir(parents=False, exist_ok=False)
    _atomic_json(path / "RUN_STATE.json", {
        "schema": "a_calibration_scale.run_state.v1", "run_id": run_id,
        "status": "INCOMPLETE", "protocol_id": PROTOCOL_ID,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return path


def _write_run_state(path: Path, status: str, details: Mapping[str, Any]) -> None:
    target = path / "RUN_STATE.json"
    replacement = {"schema": "a_calibration_scale.run_state.v1",
                   "run_id": path.name.removeprefix("a_calibration_scale_"),
                   "status": status, "protocol_id": PROTOCOL_ID,
                   "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   **dict(details)}
    temp = target.with_suffix(".json.tmp")
    if temp.exists():
        raise ExperimentAError("Refusing to reuse an unfinished state temporary")
    _write_once(temp, canonical_json(replacement) + b"\n")
    os.replace(temp, target)


def _run_preflight(run_id: str) -> dict[str, Any]:
    result = full_preflight()
    out = run_dir(run_id)
    if not out.exists():
        out.mkdir(parents=False, exist_ok=False)
    _atomic_json(out / "preflight.json", result)
    return result


def _run_smoke_child(run_id: str) -> dict[str, Any]:
    """Real, non-scientific native smoke with one verified TRAIN-normal write."""
    import torch
    from types import SimpleNamespace
    from PIL import Image
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration import stage05
    from experiments.calibration.profiles import NativeScorerConfig
    from experiments.risk_safe.native_bridge import (
        dynamic_codebooks, verify_native_update_parity)
    from v02_memory import MemoryArm

    preflight = data_preflight()
    category = "audiojack"
    info = preflight["classes"][category]
    train, _test, _ = stage05.read_class_stream(DATASET_ROOT, JSON_ROOT,
                                                 category, test_labels=False)
    selection = stage05.select_supports(train)
    train_by_rel = {row.relative_id: row for row in train}
    support_paths = [train_by_rel[rel].image_path
                     for rel in selection["supports"]["0"]["images"]]
    impl = p0.require_sources()
    runtime = r1._configure_historical_runtime(impl, torch)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ExperimentAError("Smoke requires the verified single CUDA device")
    receipt = r3._get_verified_model_receipt()
    restore_hub, model_profile = r3._runtime_and_native_setup(torch, impl, receipt)
    try:
        cfg, _profile = r3._native_config(p0, __import__(
            "experiments.calibration.profiles", fromlist=["NativeScorerConfig"]), r1)
        backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
        loaded_model_sha = r1._model_state_sha256(backbone.model)
        if loaded_model_sha != receipt.get("loaded_model_state_sha256"):
            raise ExperimentAError("Smoke loaded model differs from frozen receipt")
        restore_hub()
        cache = p0.encode_reference_features(impl, backbone, support_paths, cfg)
        bank = p0.build_vmb(impl, cache, support_paths, cfg)
        manifest = json.loads((ROOT / "outputs/calibration/stage05_exploratory_20261029/score_tape_manifest.json").read_text(encoding="utf-8"))
        expected_fixed = manifest["classes"][category]["score_tapes"]["0"]["fixed_support_sha256"]
        evidence = r3._check_bank_receipt(
            bank, p0, expected_fixed, r1)
        support_row = next(row for row in train if row.is_good and
                           row.sample_key not in set(info["role_manifest"]["support_union_groups"]) and
                           row.sample_key not in set(info["role_manifest"]["H_groups"]))
        image, features = p0.encode_query(impl, backbone, support_row.image_path, cfg)
        base_map, top_ids, score, summary = r1._native_dense_score(
            impl, bank, image, features, cfg)
        settings = SimpleNamespace(flat=1024, stm=1024, ltm=1, buffer=1,
                                   min_events=2, promote_every=16,
                                   cap_bytes=MAX_RETAINED_BYTES)
        writer = MemoryArm("greedy_coreset", bank, settings, seed=0)
        if any(cb.size() != 0 for cb in dynamic_codebooks(bank)):
            raise ExperimentAError("Smoke Greedy bank did not cold-start empty")
        if any(int(cb.K) != 1024 for cb in dynamic_codebooks(bank)):
            raise ExperimentAError("Smoke native Greedy capacity is not K=1024")
        torch_module = __import__("torch")
        selected = torch_module.as_tensor(top_ids, device=features[0].device,
                                          dtype=torch_module.long)
        selected_features = impl.l2_normalize(features[0])[selected].detach().float().cpu().numpy()
        # Training records are independently verified normal by stage05 metadata.
        update = verify_native_update_parity(
            impl, writer, features,
            {"per_layer_summary": {"layer0": {"topk_patch_idx": top_ids.tolist()}}},
            score=float(score), theta=float(score) - 0.001,
            sigma=0.02, step=1)
        after = r1._native_dense_score(impl, bank, image, features, cfg)[2]
        if not any(cb.size() > 0 for cb in dynamic_codebooks(bank)):
            raise ExperimentAError("Smoke did not produce a genuine dynamic write")
        torch.cuda.synchronize()
        result = {
            "schema": "a_calibration_scale.smoke.v1", "status": "SMOKE_COMPLETE",
            "category": category, "source_group_sha256": sha256_bytes(
                support_row.sample_key.encode()),
            "input_is_train_normal": True, "masks_read": False,
            "test_labels_read": False, "runtime_profile": runtime,
            "model_profile": model_profile, "fixed_support_evidence": evidence,
            "fixed_support_sha256": expected_fixed,
            "native_score_before": float(score), "native_score_after": float(after),
            "greedy_update_parity": update,
            "dynamic_capacity": [int(cb.K) for cb in dynamic_codebooks(bank)],
            "post_write_occupancy": [int(cb.size()) for cb in dynamic_codebooks(bank)],
            "write_inputs": {"verified_normal": True, "query_policy": "not invoked; smoke only",
                             "writer_theta": float(score) - 0.001,
                             "writer_sigma": 0.02,
                             "topk_count": int(len(top_ids)),
                             "selected_features_sha256": sha256_bytes(
                                 np.ascontiguousarray(selected_features).tobytes())},
        }
        backbone.cleanup()
        return result
    finally:
        restore_hub()


def _frozen_experiment_contract(preflight: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "schema": "a_calibration_scale.contract.v1",
        "protocol_id": PROTOCOL_ID,
        "authorization_id": AUTHORIZATION_ID,
        "run_id": run_id,
        "git_parent_sha": preflight["git"]["head"],
        "source_hashes_at_contract": preflight["source_fingerprints"],
        "model": preflight.get("model"),
        "preprocessing": {"resize": 448, "center_crop": 392,
                          "feature_layers": [5, 11, 17, 23],
                          "fusion": "concatenation; L2-normalized 4096D",
                          "patch_grid": [28, 28], "view": "identity only",
                          "support_augmentations": 4, "include_identity": False,
                          "fixed_capacity_nominal": 25000,
                          "fixed_per_image_cap": 5000,
                          "knn": 5, "topk_patches": 32,
                          "image_aggregation": "spatial maximum"},
        "dynamic_configuration": {"greedy_capacity": 1024,
                                  "correction_capacity": 1024,
                                  "correction_strength": 1.0,
                                  "writer_family": "R9/R10 native writer; 8/16/24 patches",
                                  "writer_mutable_inputs": "must match registered A2/A3 profile"},
        "classes": list(CLASSES),
        "condition_profiles": {
            "P0": {"support_seed": 0, "group_shuffle_seed": 0},
            "P1": {"support_seed": 0, "group_shuffle_seed": 11},
            "P2": {"support_seed": 1, "group_shuffle_seed": 0}},
        "thresholds": {"SELF": "mean + 2.5 * sample_std, support self-score",
                       "LOSO": "mean + 2.5 * sample_std, each fold excludes held-out Support and augmentation descendants",
                       "FIXED05": 0.5,
                       "Q90": "higher; index ceil(0.90*(n-1))",
                       "GAUSSIAN_Q90": "mean + 1.2815515655 * sample_std",
                       "GAUSSIAN_2_5": "mean + 2.5 * sample_std"},
        "data_access": {"train_normal": True, "test_pixels": True,
                        "queried_labels": "registered query gate only",
                        "unqueried_labels": "Stage B after Stage A seal only",
                        "masks": False, "confirmation_classes": False},
        "query_tape": {"policy": "Q-A-TS-v1", "rate": 0.03,
                       "seed": 37, "temporal_bins": 16,
                       "max_queries_per_official_group": 1},
        "resource_limits": {"smoke_seconds": SMOKE_DEADLINE_SECONDS,
                            "category_seconds": CATEGORY_DEADLINE_SECONDS,
                            "campaign_seconds": CAMPAIGN_DEADLINE_SECONDS,
                            "minimum_free_bytes": MIN_FREE_BYTES,
                            "maximum_retained_bytes": MAX_RETAINED_BYTES},
        "estimands": {"Cmax-G": "one deterministic view per recorded official group",
                      "Cmax-V": "all eligible views; correlated views are not independent groups",
                      "H": "32 additional TRAIN-normal groups; development-only normal audit"},
        "excluded": ["Stage 1", "Stage 2 historical programs", "Experiments B/C",
                     "mint", "pcb", "all other categories", "masks"],
    }


def _stage_a_category(run_id: str, category: str) -> dict[str, Any]:
    """Fresh native M0 scoring for one category; no TEST labels are parsed."""
    from types import SimpleNamespace
    import torch
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration import stage05
    from experiments.calibration.profiles import NativeScorerConfig
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job

    authorized_category(category)
    if not current_process_is_in_job():
        raise ExperimentAError("A1 category execution must run inside a Windows Job Object")
    out = run_dir(run_id) / "a1" / category
    out.mkdir(parents=True, exist_ok=False)
    train, test, json_sha = stage05.read_class_stream(
        DATASET_ROOT, JSON_ROOT, category, test_labels=False)
    stream_orders = _verify_registered_stream_orders(test, category)
    selection = stage05.select_supports(train)
    roles = higher_order_membership(train, selection["supports"], selection["c64"],
                                    category=category)
    role_bytes = canonical_json(roles)
    _write_once(out / "role_manifest_private.json", role_bytes + b"\n")
    train_by_rel = {row.relative_id: row for row in train}
    by_group: dict[str, list[Any]] = {}
    for row in train:
        if row.is_good:
            by_group.setdefault(row.sample_key, []).append(row)

    impl = p0.require_sources()
    runtime = r1._configure_historical_runtime(impl, torch)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ExperimentAError("A1 requires the verified single CUDA device")
    model_receipt = r3._get_verified_model_receipt()
    restore_hub, model_profile = r3._runtime_and_native_setup(torch, impl, model_receipt)
    cfg, _scorer_profile = r3._native_config(
        p0, __import__("experiments.calibration.profiles", fromlist=["NativeScorerConfig"]), r1)
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    backbone = None
    try:
        backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
        loaded_model_sha = r1._model_state_sha256(backbone.model)
        if loaded_model_sha != model_receipt.get("loaded_model_state_sha256"):
            raise ExperimentAError("A1 loaded DINO state differs from verified model receipt")
        restore_hub()

        all_support_paths: list[str] = []
        support_paths_by_seed: dict[int, list[str]] = {}
        support_cache: dict[str, Any] = {}
        for seed in (0, 1):
            paths = [train_by_rel[rel].image_path
                     for rel in selection["supports"][str(seed)]["images"]]
            support_paths_by_seed[seed] = paths
            all_support_paths.extend(paths)
        all_support_paths = list(dict.fromkeys(all_support_paths))
        support_cache = p0.encode_reference_features(impl, backbone, all_support_paths, cfg)

        manifest = json.loads((ROOT / "outputs/calibration/stage05_exploratory_20261029/score_tape_manifest.json").read_text(encoding="utf-8"))
        banks: dict[int, Any] = {}
        bank_receipts: dict[int, dict[str, Any]] = {}
        support_feature_rows: dict[int, list[tuple[Any, Any]]] = {}
        for seed in (0, 1):
            paths = support_paths_by_seed[seed]
            impl.set_seed(0)
            bank = p0.build_vmb(impl, support_cache, paths, cfg)
            expected = manifest["classes"][category]["score_tapes"][str(seed)]["fixed_support_sha256"]
            evidence = r3._check_bank_receipt(bank, p0, expected, r1)
            banks[seed] = bank
            bank_receipts[seed] = evidence
            support_feature_rows[seed] = []
            for path in paths:
                image, features = p0.encode_query(impl, backbone, path, cfg)
                support_feature_rows[seed].append((image, features))

        # SELF uses the four deployed Supports against their exact augmented M0.
        support_scores: dict[str, list[float]] = {"0": [], "1": []}
        for seed in (0, 1):
            for image, features in support_feature_rows[seed]:
                support_scores[str(seed)].append(float(r1._native_dense_score(
                    impl, banks[seed], image, features, cfg)[2]))

        # LOSO rebuilds four temporary three-Support banks for each Support seed.
        loso_scores: dict[str, list[float]] = {"0": [], "1": []}
        for seed in (0, 1):
            paths = support_paths_by_seed[seed]
            for held_index, held_path in enumerate(paths):
                fold_paths = [path for i, path in enumerate(paths) if i != held_index]
                impl.set_seed(0)
                fold_bank = p0.build_vmb(impl, support_cache, fold_paths, cfg)
                held_image, held_features = support_feature_rows[seed][held_index]
                loso_scores[str(seed)].append(float(r1._native_dense_score(
                    impl, fold_bank, held_image, held_features, cfg)[2]))
                del fold_bank
                torch.cuda.empty_cache()

        # A1 calibration references are scored once per Support bank. All group
        # and view memberships are frozen in the role manifest above.
        reference_rel = list(roles["Cmax_V_images"])
        h_rel = list(roles["H_images"])
        normal_rel = list(dict.fromkeys(reference_rel + h_rel))
        reference_scores: dict[str, dict[str, float]] = {"0": {}, "1": {}}
        encode_seconds = score_seconds = 0.0
        for rel in normal_rel:
            row = train_by_rel[rel]
            encode_start = time.perf_counter()
            image, features = p0.encode_query(impl, backbone, row.image_path, cfg)
            torch.cuda.synchronize()
            encode_seconds += time.perf_counter() - encode_start
            for seed in (0, 1):
                score_start = time.perf_counter()
                value = r1._native_dense_score(impl, banks[seed], image, features, cfg)[2]
                torch.cuda.synchronize()
                score_seconds += time.perf_counter() - score_start
                reference_scores[str(seed)][rel] = float(value)

        # TEST image pixels are encoded exactly once and scored against both
        # immutable M0 banks before any label access.
        test_scores: dict[str, list[float]] = {"0": [math.nan] * len(test),
                                              "1": [math.nan] * len(test)}
        test_identity_rows = []
        test_encode_seconds = test_score_seconds = 0.0
        for row in test:
            encode_start = time.perf_counter()
            image, features = p0.encode_query(impl, backbone, row.image_path, cfg)
            torch.cuda.synchronize()
            test_encode_seconds += time.perf_counter() - encode_start
            for seed in (0, 1):
                score_start = time.perf_counter()
                value = r1._native_dense_score(impl, banks[seed], image, features, cfg)[2]
                torch.cuda.synchronize()
                test_score_seconds += time.perf_counter() - score_start
                test_scores[str(seed)][row.position] = float(value)
            test_identity_rows.append({
                "position": row.position,
                "image_identity_sha256": r7._identity_hash(row.relative_id),
                "source_group_sha256": row.opaque_group,
            })

        if any(not np.isfinite(values).all() for values in
               [np.asarray(test_scores["0"]), np.asarray(test_scores["1"]) ]):
            raise ExperimentAError("A1 produced a nonfinite TEST score")

        # Direct full-vector parity against the historic static tapes. The tape
        # source manifest has two stale hashes, so this numerical parity does
        # not replace the independent source receipt above.
        parity: dict[str, Any] = {}
        tape_dir = ROOT / "outputs/calibration/stage05_exploratory_20261029/score_tapes"
        for seed in (0, 1):
            tape_path = tape_dir / f"{category}_support{seed}.npz"
            with np.load(tape_path, allow_pickle=False) as tape:
                parity[f"test_support{seed}"] = _assert_test_score_parity(
                    test_scores[str(seed)], tape["test_scores"])
                parity[f"c64_support{seed}"] = _assert_test_score_parity(
                    [reference_scores[str(seed)][rel] for rel in selection["c64"]["images"]],
                    tape["c64_scores"])

        draw_scores: dict[str, dict[str, dict[str, list[float]]]] = {"0": {}, "1": {}}
        for seed in (0, 1):
            for draw_id, draw in roles["draws"].items():
                draw_scores[str(seed)][draw_id] = {}
                for size in ("C32", "C64", "C128", "Cmax-G"):
                    group_keys = draw[size]
                    if draw_id == "0" and size in ("C32", "C64"):
                        historical_by_group = dict(zip(selection["c64"]["groups"],
                                                       selection["c64"]["images"]))
                        rels = [historical_by_group[key] for key in group_keys]
                    else:
                        rels = [_a1_group_view(by_group[key], category) for key in group_keys]
                    draw_scores[str(seed)][draw_id][size] = [
                        reference_scores[str(seed)][rel] for rel in rels]
                draw_scores[str(seed)][draw_id]["Cmax-V"] = [
                    reference_scores[str(seed)][rel] for rel in roles["Cmax_V_images"]]

        score_payload = {
            "schema": "a_calibration_scale.a1_stage_a_scores.v1",
            "protocol_id": PROTOCOL_ID, "category": category,
            "support_seed_test_scores": test_scores,
            "support_seed_self_scores": support_scores,
            "support_seed_loso_scores": loso_scores,
            "support_seed_reference_scores": reference_scores,
            "support_seed_calibration_draw_scores": draw_scores,
            "test_identity_rows": test_identity_rows,
            "registered_stream_orders": stream_orders,
            "test_labels_read": False, "masks_read": False,
            "native_parity": parity,
            "fixed_support_receipts": bank_receipts,
        }
        score_path = out / "stage_a_scores_private.json"
        score_bytes = canonical_json(score_payload) + b"\n"
        _write_once(score_path, score_bytes)
        torch.cuda.synchronize()
        gpu_peak = {"allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "reserved_bytes": int(torch.cuda.max_memory_reserved())}
        seal = create_stage_a_seal(
            out / "stage_a_seal.json", run_id=run_id,
            input_digests={"category_json": json_sha,
                           "role_manifest": sha256_bytes(role_bytes),
                           "source_hashes": sha256_bytes(canonical_json(_source_fingerprints())),
                           "fixed_support_seed0": bank_receipts[0]["actual_structured_sha256"],
                           "fixed_support_seed1": bank_receipts[1]["actual_structured_sha256"]},
            score_digests={"stage_a_scores_private.json": sha256_bytes(score_bytes)})
        result = {
            "schema": "a_calibration_scale.a1_stage_a_result.v1",
            "status": "A_SEALED", "category": category,
            "test_count": len(test), "train_normal_groups": selection["normal_train_groups"],
            "support_image_counts": {"0": 4, "1": 4},
            "C64_count": 64, "Cmax_G_groups": len(roles["eligible_calibration_groups"]),
            "Cmax_V_groups": roles["Cmax_V_group_count"],
            "Cmax_V_images": roles["Cmax_V_image_count"],
            "H_groups": len(roles["H_groups"]),
            "fixed_bank_realized": {str(seed): bank_receipts[seed]["layer_profile"]
                                     for seed in (0, 1)},
            "native_parity": parity,
            "stage_a_seal_sha256": seal["seal_sha256"],
            "test_score_sha256": {seed: sha256_bytes(
                np.asarray(values, dtype="<f8").tobytes())
                for seed, values in test_scores.items()},
            "timing_seconds": {"total": time.perf_counter() - started,
                               "reference_encode": encode_seconds,
                               "reference_scoring": score_seconds,
                               "test_encode": test_encode_seconds,
                               "test_scoring_two_banks": test_score_seconds},
            "gpu_peak": gpu_peak,
            "runtime_profile": runtime,
            "model_profile": model_profile,
            "loaded_model_state_sha256": loaded_model_sha,
            "test_labels_read": False, "masks_read": False,
        }
        _write_once(out / "stage_a_result.json", canonical_json(result) + b"\n")
        backbone.cleanup()
        return result
    finally:
        restore_hub()


def _a1_group_view(rows: Sequence[Any], category: str) -> str:
    return min(rows, key=lambda row: (
        hashlib.sha256((f"{PROTOCOL_ID}/H-view/{category}\0" + row.relative_id).encode()).hexdigest(),
        row.relative_id)).relative_id


def _stage_b_category(run_id: str, category: str) -> dict[str, Any]:
    """Offline evaluator; only this process is allowed to request TEST labels."""
    from experiments.calibration import stage05
    from experiments.calibration import r7_shuffled_correction_4class as r7
    authorized_category(category)
    out = run_dir(run_id) / "a1" / category
    seal = require_stage_a_seal(out / "stage_a_seal.json")
    score_path = out / "stage_a_scores_private.json"
    if sha256_file(score_path) != seal["score_digests"]["stage_a_scores_private.json"]:
        raise ExperimentAError("Stage B refused: score tape digest differs from Stage A seal")
    score_payload = json.loads(score_path.read_text(encoding="utf-8"))
    train, test, _ = stage05.read_class_stream(DATASET_ROOT, JSON_ROOT,
                                                category, test_labels=False)
    sealed_ids = score_payload["test_identity_rows"]
    if len(test) != len(sealed_ids):
        raise ExperimentAError("Stage B TEST identity and sealed score counts differ")
    for row, sealed in zip(test, sealed_ids):
        if (int(sealed["position"]) != int(row.position) or
                sealed["image_identity_sha256"] != r7._identity_hash(row.relative_id) or
                sealed["source_group_sha256"] != row.opaque_group):
            raise ExperimentAError("Stage B current TEST identity differs from Stage A seal")
    labels = stage05.test_labels_for_class(JSON_ROOT, category)
    if len(labels) != len(sealed_ids):
        raise ExperimentAError("Stage B label and sealed score counts differ")
    roles = json.loads((out / "role_manifest_private.json").read_text(encoding="utf-8"))
    condition_specs = {"P0": (0, 0), "P1": (0, 11), "P2": (1, 0)}
    results: dict[str, Any] = {}
    all_auc: dict[str, Any] = {}
    for condition, (support_seed, shuffle_seed) in condition_specs.items():
        events = r7.deterministic_group_shuffle(test, seed=shuffle_seed)
        order = [int(e.original_position) for e in events]
        y = [int(labels[i]) for i in order]
        raw = score_payload["support_seed_test_scores"][str(support_seed)]
        s = [float(raw[i]) for i in order]
        initial = threshold_methods(
            score_payload["support_seed_calibration_draw_scores"][str(support_seed)]["0"],
            score_payload["support_seed_self_scores"][str(support_seed)],
            score_payload["support_seed_loso_scores"][str(support_seed)])
        calibration_arrays = score_payload["support_seed_calibration_draw_scores"][str(support_seed)]["0"]
        source_scores: dict[str, Sequence[float] | None] = {
            "SELF": score_payload["support_seed_self_scores"][str(support_seed)],
            "LOSO": score_payload["support_seed_loso_scores"][str(support_seed)],
            "FIXED05": None,
        }
        for arm_name in initial:
            if arm_name.startswith("Q90:"):
                source_scores[arm_name] = calibration_arrays[arm_name.split(":", 1)[1]]
            elif arm_name.startswith("GAUSSIAN-Q90:"):
                source_scores[arm_name] = calibration_arrays[arm_name.split(":", 1)[1]]
            elif arm_name.startswith("GAUSSIAN-2.5:"):
                source_scores[arm_name] = calibration_arrays[arm_name.split(":", 1)[1]]
        h_scores = [float(score_payload["support_seed_reference_scores"][str(support_seed)][rel])
                    for rel in roles["H_images"]]
        arm_rows = {}
        for arm, theta in initial.items():
            metrics = metrics_for_threshold(y, s, theta)
            h_exceed = sum(value >= theta for value in h_scores) / len(h_scores)
            baseline_scores = source_scores.get(arm)
            if baseline_scores is not None:
                normal_mean, normal_std = sample_mean_std(baseline_scores)
                calibration_summary = {"n": len(baseline_scores),
                                       "mean": normal_mean,
                                       "sample_std": normal_std}
            else:
                calibration_summary = None
            prefixes = {}
            for n in (512, 1024, 2048):
                prefixes[str(n)] = metrics_for_threshold(y[:min(n, len(y))],
                                                         s[:min(n, len(s))], theta)
            arm_rows[arm] = {"threshold": theta, "metrics": metrics,
                             "initial_normal_score_summary": calibration_summary,
                             "H_exceedance_fraction": h_exceed,
                             "early_prefixes": prefixes,
                             "worst_nonoverlap_512": _worst_valid_window(
                                 window_metrics(y, s, theta, 512))}
        draw_thresholds: dict[str, dict[str, float]] = {}
        for draw_id, calibration in score_payload[
                "support_seed_calibration_draw_scores"][str(support_seed)].items():
            selected = {}
            for size in ("C32", "C64", "C128", "Cmax-G", "Cmax-V"):
                selected[f"Q90:{size}"] = higher_quantile(calibration[size])
            draw_thresholds[draw_id] = selected
        threshold_stability = {}
        for arm in ("Q90:C32", "Q90:C64", "Q90:C128", "Q90:Cmax-G", "Q90:Cmax-V"):
            vals = [draw_thresholds[draw][arm] for draw in sorted(draw_thresholds)]
            threshold_stability[arm] = {"draw_values": vals,
                                        "range": max(vals) - min(vals),
                                        "sample_std": float(np.std(vals, ddof=1))}
        auc_check = audit_threshold_auroc(y, s,
                                          [v["threshold"] for v in arm_rows.values()])
        all_auc[condition] = auc_check
        results[condition] = {
            "support_seed": support_seed, "group_shuffle_seed": shuffle_seed,
            "event_identity_sha256": sha256_bytes(canonical_json([
                {"position": e.position, "original_position": e.original_position,
                 "image_identity_sha256": e.image_identity_sha256,
                 "source_group_sha256": e.source_group_sha256}
                for e in events])),
            "score_sha256": sha256_bytes(np.asarray(s, dtype="<f8").tobytes()),
            "arms": arm_rows, "threshold_stability": threshold_stability,
            "auc_parity": auc_check,
        }
    result = {"schema": "a_calibration_scale.a1_stage_b_metrics.v1",
              "status": "STAGE_B_COMPLETE", "category": category,
              "stage_a_seal_sha256": seal["seal_sha256"],
              "stage_b_label_source": "separate evaluator after Stage A seal",
              "masks_read": False, "metrics": results,
              "threshold_only_auroc_parity": all_auc}
    _write_once(out / "stage_b_metrics_private.json", canonical_json(result) + b"\n")
    return result


def _worst_valid_window(windows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in windows if row["status"] == "VALID"]
    if not valid:
        return None
    return max(valid, key=lambda row: (row["ber"], -row["ba"], row["start"]))


def _self_test() -> bool:
    import unittest
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "tests.calibration.test_a_calibration_scale")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return bool(result.wasSuccessful())


def _run_full_a1(run_id: str) -> dict[str, Any]:
    """Run preflight, CPU tests, supervised smoke, A1 Stage A, then Stage B."""
    from experiments.calibration.r1_runtime_supervisor import run_supervised_child
    from experiments.calibration.r1_runtime_supervisor import SupervisorResult

    out = _create_run(run_id)
    started = time.monotonic()
    preflight = full_preflight()
    _atomic_json(out / "preflight_private.json", preflight)
    if preflight["status"] != "PREFLIGHT_PASS":
        _write_run_state(out, "INCOMPLETE", {"blocker": "PREFLIGHT_HOLD"})
        return {"status": "PREFLIGHT_HOLD", "run_directory": str(out)}
    contract = _frozen_experiment_contract(preflight, run_id)
    _atomic_json(out / "experiment_contract.json", contract)
    if not _self_test():
        _write_run_state(out, "INCOMPLETE", {"blocker": "CPU_TEST_FAILURE"})
        return {"status": "CPU_TEST_FAILURE", "run_directory": str(out)}

    if shutil.disk_usage(RUNS_ROOT).free < MIN_FREE_BYTES:
        _write_run_state(out, "INCOMPLETE", {"blocker": "MINIMUM_FREE_DISK"})
        return {"status": "HOLD_RESOURCE_LIMIT", "run_directory": str(out)}

    smoke_command = [sys.executable, "-m", "experiments.calibration.a_calibration_scale",
                     "--smoke-child", "--run-id", run_id]
    smoke = run_supervised_child(smoke_command,
                                 timeout_seconds=SMOKE_DEADLINE_SECONDS,
                                 cwd=ROOT, output_dir=None,
                                 max_output_bytes=MAX_RETAINED_BYTES)
    if smoke.status != "COMPLETED" or smoke.child_exit_code != 0:
        _write_run_state(out, "INCOMPLETE", {"blocker": "SMOKE_FAILED",
                                             "smoke_supervisor": asdict(smoke)})
        return {"status": "SMOKE_FAILED", "run_directory": str(out)}
    _atomic_json(out / "smoke_supervisor.json", asdict(smoke))

    stage_a_results = {}
    for category in CLASSES:
        elapsed = time.monotonic() - started
        if elapsed + CATEGORY_DEADLINE_SECONDS > CAMPAIGN_DEADLINE_SECONDS:
            _write_run_state(out, "INCOMPLETE", {
                "blocker": "CAMPAIGN_DEADLINE",
                "completed_stage_a_categories": sorted(stage_a_results)})
            return {"status": "HOLD_RESOURCE_LIMIT",
                    "completed_stage_a_categories": sorted(stage_a_results),
                    "run_directory": str(out)}
        if shutil.disk_usage(RUNS_ROOT).free < MIN_FREE_BYTES:
            _write_run_state(out, "INCOMPLETE", {
                "blocker": "MINIMUM_FREE_DISK",
                "completed_stage_a_categories": sorted(stage_a_results)})
            return {"status": "HOLD_RESOURCE_LIMIT",
                    "completed_stage_a_categories": sorted(stage_a_results),
                    "run_directory": str(out)}
        command = [sys.executable, "-m", "experiments.calibration.a_calibration_scale",
                   "--stage-a-child", "--run-id", run_id, "--category", category]
        supervised = run_supervised_child(
            command, timeout_seconds=CATEGORY_DEADLINE_SECONDS,
            cwd=ROOT, output_dir=None, max_output_bytes=MAX_RETAINED_BYTES)
        if supervised.status != "COMPLETED" or supervised.child_exit_code != 0:
            _write_run_state(out, "INCOMPLETE", {
                "blocker": "STAGE_A_CATEGORY_FAILED", "category": category,
                "supervisor": asdict(supervised),
                "completed_stage_a_categories": sorted(stage_a_results)})
            return {"status": "IMPLEMENTATION_FAILED",
                    "failed_category": category,
                    "completed_stage_a_categories": sorted(stage_a_results),
                    "run_directory": str(out)}
        result_path = out / "a1" / category / "stage_a_result.json"
        if not result_path.is_file():
            raise ExperimentAError(f"{category}: supervised Stage A child omitted its result")
        stage_a_results[category] = json.loads(result_path.read_text(encoding="utf-8"))
        _atomic_json(out / "a1" / category / "supervisor_status.json", asdict(supervised))
        output_bytes = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
        if output_bytes > MAX_RETAINED_BYTES:
            _write_run_state(out, "INCOMPLETE", {"blocker": "RETAINED_OUTPUT_CAP",
                                                 "retained_bytes": output_bytes})
            return {"status": "HOLD_RESOURCE_LIMIT", "run_directory": str(out)}

    # Stage B runs in a distinct process, after every category has its own seal.
    if any(not (out / "a1" / category / "stage_a_seal.json").is_file()
           for category in CLASSES):
        _write_run_state(out, "INCOMPLETE", {"blocker": "MISSING_STAGE_A_SEAL"})
        return {"status": "IMPLEMENTATION_FAILED", "run_directory": str(out)}
    eval_command = [sys.executable, "-m", "experiments.calibration.a_calibration_scale",
                    "--stage-b-child", "--run-id", run_id]
    evaluator = run_supervised_child(eval_command, timeout_seconds=30 * 60,
                                     cwd=ROOT, output_dir=None,
                                     max_output_bytes=MAX_RETAINED_BYTES)
    if evaluator.status != "COMPLETED" or evaluator.child_exit_code != 0:
        _write_run_state(out, "INCOMPLETE", {"blocker": "STAGE_B_EVALUATOR_FAILED",
                                             "supervisor": asdict(evaluator)})
        return {"status": "IMPLEMENTATION_FAILED", "run_directory": str(out)}
    _atomic_json(out / "stage_b_supervisor.json", asdict(evaluator))
    stage_b_path = out / "stage_b_summary_private.json"
    if not stage_b_path.is_file():
        raise ExperimentAError("Stage B process omitted its summary")
    stage_b = json.loads(stage_b_path.read_text(encoding="utf-8"))
    _integrity_audit_and_reports(run_id, preflight, contract,
                                 stage_a_results, stage_b, elapsed=time.monotonic() - started)
    return {"status": "A1_COMPLETE_A2_A3_PENDING",
            "completed_components": ["preflight", "CPU_tests", "supervised_smoke",
                                     "A1_stage_a", "A1_stage_b", "integrity_audit"],
            "run_directory": str(out), "stage_a": stage_a_results,
            "elapsed_seconds": time.monotonic() - started}


def _run_stage_b_all(run_id: str) -> dict[str, Any]:
    out = run_dir(run_id)
    results = {category: _stage_b_category(run_id, category) for category in CLASSES}
    summary = {"schema": "a_calibration_scale.stage_b_summary.v1",
               "status": "STAGE_B_COMPLETE",
               "categories": {c: {
                   "stage_a_seal_sha256": results[c]["stage_a_seal_sha256"],
                   "metrics": results[c]["metrics"],
                   "threshold_only_auroc_parity": results[c]["threshold_only_auroc_parity"]}
                   for c in CLASSES},
               "test_masks_read": False}
    _write_once(out / "stage_b_summary_private.json", canonical_json(summary) + b"\n")
    return summary


def _integrity_audit_and_reports(run_id: str, preflight: Mapping[str, Any],
                                 contract: Mapping[str, Any],
                                 stage_a: Mapping[str, Any],
                                 stage_b: Mapping[str, Any], *, elapsed: float) -> None:
    out = run_dir(run_id)
    seals = {category: require_stage_a_seal(out / "a1" / category / "stage_a_seal.json")
             for category in CLASSES}
    changed_source_hashes = _source_fingerprints()
    if changed_source_hashes != contract["source_hashes_at_contract"]:
        raise ExperimentAError("Source changed during run; integrity audit refused")
    audit = {
        "schema": "a_calibration_scale.integrity_audit.v1",
        "status": "PASS_A1_ONLY",
        "run_id": run_id,
        "branch": preflight["git"]["branch"],
        "parent_sha": preflight["git"]["head"],
        "source_hashes_unchanged": True,
        "stage_a_seals": {c: seals[c]["seal_sha256"] for c in CLASSES},
        "stage_b_present_after_seals": all(
            (out / "a1" / c / "stage_b_metrics_private.json").is_file() for c in CLASSES),
        "masks_read": False,
        "confirmation_categories_accessed": False,
        "output_bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
        "output_cap_bytes": MAX_RETAINED_BYTES,
        "campaign_elapsed_seconds": elapsed,
        "completed_experiment": "A1",
        "not_run": {"A2": "not implemented/executed in this bounded pass",
                    "A3a": "not implemented/executed in this bounded pass",
                    "A3b": "HOLD_BASELINE_FIDELITY: equal-byte coreset algorithm and exact source identities are unresolved"},
    }
    if audit["output_bytes"] > MAX_RETAINED_BYTES:
        raise ExperimentAError("Output cap exceeded during integrity audit")
    _atomic_json(out / "integrity_audit.json", audit)
    _write_a1_report(run_id, preflight, stage_a, stage_b, audit)
    _write_run_state(out, "A1_COMPLETE_A2_A3_PENDING", {
        "completed_components": ["A1", "integrity_audit", "reports"],
        "audit_sha256": sha256_file(out / "integrity_audit.json"),
    })


def _write_a1_report(run_id: str, preflight: Mapping[str, Any],
                     stage_a: Mapping[str, Any], stage_b: Mapping[str, Any],
                     audit: Mapping[str, Any]) -> None:
    report_dir = REPORT_ROOT
    report_dir.mkdir(parents=True, exist_ok=True)
    stage_b_path = run_dir(run_id) / "stage_b_summary_private.json"
    summary = json.loads(stage_b_path.read_text(encoding="utf-8"))
    lines = [
        "# Experiment A: Calibration Scale, Initialization and State Dependence",
        "",
        f"Run `{run_id}` completed A1 only. This is a four-category development study, not independent confirmation.",
        "",
        "## Protocol and integrity",
        "",
        f"Branch `{preflight['git']['branch']}` at parent `{preflight['git']['head']}`. The frozen native DINOv2/VisionAD path was used; the fixed bank realized 8,192 float16 prototypes per layer. Test images were scored before Stage B; the separate Stage B process read image labels only after four category-specific Stage A seals. No masks or confirmation categories were accessed.",
        "",
        "The Stage05 static score-tape manifest contains stale hashes for `stage05.py` and `stage05_runner.py`; those tapes were used only for full-vector numerical parity against fresh native scores, never as primary results. Support/C64 fixed-bank structured digests were independently verified.",
        "",
        "## Data roles",
        "",
        "| Category | Train groups | Eligible calibration groups | H groups | Cmax-V images | Test events |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category, values in preflight["classes"].items():
        role = values["role_manifest"]
        lines.append(f"| {category} | {values['train_group_count']} | {len(role['eligible_calibration_groups'])} | {len(role['H_groups'])} | {role['Cmax_V_image_count']} | {values['test_image_count']} |")
    lines += ["", "## A1 results", "",
              "Full threshold results, threshold draw stability, normal-only H exceedance, prefixes, and 512-event windows are in the protected local run output. Threshold arms shared the same frozen score vector in each support condition. Exact per-condition AUROC equality was verified.",
              "",
              "The report does not select a calibration size by test BER. The decision candidate below uses H-tail behavior, draw stability, and resource use; test metrics remain evaluation and safety outcomes.",
              "",
              "## Component status", "",
              "- A1: `COMPLETE_VALID` for P0/P1/P2 across the four authorized development categories.",
              "- A2: not executed; shared-state B3 rescoring has not been completed.",
              "- A3a: not executed; calibration-coupled trajectories have not been completed.",
              "- A3b: `HOLD_BASELINE_FIDELITY`; a deterministic equal-byte expanded coreset constructor is not specified sufficiently to implement without adding an unregistered method.",
              "- No Stage 1/2 program, Experiment B/C, confirmation outcome, or mask was accessed.",
              "",
              "## Reproduction",
              "",
              f"`D:\\anaconda3\\envs\\torch\\python.exe -m experiments.calibration.a_calibration_scale --run --run-id {run_id}`",
              "",
              f"A1 run output: `{run_dir(run_id)}`. Audit: `{run_dir(run_id) / 'integrity_audit.json'}`.",
              ""]
    report = report_dir / "REPORT.md"
    if report.exists():
        raise ExperimentAError(f"Refusing to overwrite report: {report}")
    report.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    decision_path = report_dir / "CALIBRATION_DECISION_CANDIDATE.md"
    if decision_path.exists():
        raise ExperimentAError(f"Refusing to overwrite decision memo: {decision_path}")
    decision = ("# Calibration Decision Candidate\n\n"
                "A1 alone does not freeze the C* calibration choice. Use the normal-only H tail diagnostics, C32/C64/C128/Cmax sampling variation, and cost in the protected run output. Treat SELF and LOSO as diagnostics. Retain fixed 0.5 only if its raw-score FPR/FNR tradeoff remains acceptable under the registered evaluation. Compare empirical q90 with the Gaussian moment rules without selecting on TEST BER.\n\n"
                "A2/A3 evidence is still required before choosing a calibration configuration for Experiments B/C. A3b remains on hold until the equal-byte expanded-reference baseline can be specified and audited without adding an unregistered coreset algorithm.\n")
    decision_path.write_text(decision, encoding="utf-8", newline="\n")


def run_smoke(run_id: str) -> dict[str, Any]:
    run_id = validate_run_id(run_id)
    out = run_dir(run_id)
    if out.exists():
        raise ExperimentAError("Refusing to reuse an existing Experiment A run directory")
    free = shutil.disk_usage(RUNS_ROOT).free
    if free < MIN_FREE_BYTES:
        raise ExperimentAError("Smoke blocked: free disk is below the registered 30 GiB floor")
    out = _create_run(run_id)
    command = [sys.executable, "-m", "experiments.calibration.a_calibration_scale",
               "--smoke-child", "--run-id", run_id]
    from experiments.calibration.r1_runtime_supervisor import run_supervised_child
    started = time.monotonic()
    result = run_supervised_child(command, timeout_seconds=SMOKE_DEADLINE_SECONDS,
                                  cwd=ROOT, output_dir=None,
                                  max_output_bytes=MAX_RETAINED_BYTES)
    if result.child_exit_code != 0 or result.status != "COMPLETED":
        failure_path = out / "smoke_failure.json"
        failure = (failure_path.read_text(encoding="utf-8")
                   if failure_path.is_file() else "child did not persist an exception")
        if out.exists() and (out / "RUN_STATE.json").exists():
            _write_run_state(out, "INCOMPLETE", {"smoke_status": result.status,
                                                 "supervisor": asdict(result),
                                                 "failure": failure})
        raise ExperimentAError(f"Supervised GPU smoke failed: status={result.status}; {failure}")
    elapsed = time.monotonic() - started
    profile = json.loads((out / "smoke_profile.json").read_text(encoding="utf-8"))
    output_bytes = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    if output_bytes > MAX_RETAINED_BYTES:
        _write_run_state(out, "INCOMPLETE", {"smoke_status": "HOLD_RESOURCE_LIMIT",
                                             "retained_bytes": output_bytes})
        raise ExperimentAError("Smoke output exceeded the 4 GiB retained-output cap")
    profile["supervisor"] = asdict(result)
    profile["wall_seconds"] = elapsed
    # Replace only this run's authenticated temporary report via a separate file.
    _atomic_json(out / "smoke_supervisor.json", profile)
    status = "INCOMPLETE"
    _write_run_state(out, status, {"completed_component": "bounded_gpu_smoke",
                                   "smoke_status": profile["status"],
                                   "full_experiment_status": "NOT_STARTED"})
    return profile


def _default_run_id() -> str:
    return time.strftime("20261106_%H%M%S", time.gmtime())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-child", action="store_true", help=argparse.SUPPRESS)
    modes.add_argument("--stage-a-child", action="store_true", help=argparse.SUPPRESS)
    modes.add_argument("--stage-b-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--category", choices=CLASSES, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    run_id = validate_run_id(args.run_id or _default_run_id())
    try:
        if args.self_test:
            import unittest
            suite = unittest.defaultTestLoader.loadTestsFromName(
                "tests.calibration.test_a_calibration_scale")
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            return 0 if result.wasSuccessful() else 1
        if args.preflight:
            result = full_preflight()
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "PREFLIGHT_PASS" else 2
        if args.smoke_child:
            out = run_dir(run_id)
            if not out.is_dir() or not (out / "RUN_STATE.json").is_file():
                raise ExperimentAError("Smoke child requires a precreated, incomplete run directory")
            try:
                result = _run_smoke_child(run_id)
                _atomic_json(out / "smoke_profile.json", result)
            except BaseException as exc:
                _atomic_json(out / "smoke_failure.json", {
                    "exception_type": type(exc).__name__, "message": str(exc)})
                raise
            return 0
        if args.stage_a_child:
            if args.category is None:
                raise ExperimentAError("Stage A child requires --category")
            result = _stage_a_category(run_id, args.category)
            print(json.dumps({"status": result["status"],
                              "category": result["category"],
                              "stage_a_seal_sha256": result["stage_a_seal_sha256"]},
                             sort_keys=True))
            return 0
        if args.stage_b_child:
            result = _run_stage_b_all(run_id)
            print(json.dumps({"status": result["status"],
                              "category_count": len(result["categories"])}, sort_keys=True))
            return 0
        if args.smoke:
            result = run_smoke(run_id)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.audit_only:
            result = {"git": _git_identity(), "protected_file_audit": protected_file_audit(),
                      "run_directory": str(run_dir(run_id)),
                      "run_directory_exists": run_dir(run_id).exists()}
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["protected_file_audit"]["passed"] else 2
        if args.run:
            result = _run_full_a1(run_id)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "A1_COMPLETE_A2_A3_PENDING" else 2
    except (ExperimentAError, OSError, ValueError, RuntimeError) as exc:
        if "args" in locals() and getattr(args, "stage_a_child", False) and args.category:
            failure_dir = run_dir(run_id) / "a1" / args.category
            try:
                failure_dir.mkdir(parents=True, exist_ok=True)
                failure = failure_dir / "child_failure.json"
                if not failure.exists():
                    _atomic_json(failure, {"exception_type": type(exc).__name__,
                                           "message": str(exc)})
            except OSError:
                pass
        if "args" in locals() and getattr(args, "stage_b_child", False):
            try:
                failure = run_dir(run_id) / "stage_b_failure.json"
                if not failure.exists():
                    _atomic_json(failure, {"exception_type": type(exc).__name__,
                                           "message": str(exc)})
            except OSError:
                pass
        print(f"EXPERIMENT_A_HOLD: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
