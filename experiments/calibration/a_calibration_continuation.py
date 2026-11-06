#!/usr/bin/env python3
"""Integrated, isolated A2/A3a/A3b continuation for Experiment A.

This module imports the frozen native scorer, Greedy writer, C1-Full updater,
and A1 data-role contract.  It does not edit or replace any historical runner.
All empirical outputs are private under one new, non-overwriting run root.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.calibration import a_calibration_scale as a1

RUNS_ROOT = ROOT / "outputs" / "calibration"
REPORT_ROOT = ROOT / "docs" / "calibration" / "audit" / "a_calibration_scale_20261106"
A1_RUN_ID = "20261106_a1_full01"
RUN_PREFIX = "a_calibration_scale_"
PROTOCOL_ID = "a_calibration_scale_a23_20261106.v1"
AUTHORIZATION_ID = "EXPERIMENT_A_AUTHORIZATION_20261106"
SOURCE_PARENT = "897dabe704bbb6a14779d47f93e7e4c3f42f298a"
MAX_OUTPUT_BYTES = 4 * 1024**3
MIN_FREE_BYTES = 30 * 1024**3
CATEGORY_DEADLINE_SECONDS = 120 * 60
CAMPAIGN_DEADLINE_SECONDS = 8 * 60 * 60
CHECKPOINT_INTERVAL = 512
K_GREEDY = 1024
K_CORRECTION = 1024
CORRECTION_STRENGTH = 1.0
Q90 = 0.90
B3_DELTA = 0.01
SIGMA_FLOOR = 0.02
KC_CANDIDATE_CAP = 65_536
KC_PROTOTYPE_COUNT = 8192
KC_BLOCK_ROWS = 4096
CONDITIONS = {
    "P0": {"support_seed": 0, "order": "SHUFFLE0"},
    "P1": {"support_seed": 0, "order": "SHUFFLE11"},
    "P2": {"support_seed": 1, "order": "SUPPORT1_SHUFFLE0"},
}
PRIMARY_CATEGORIES = a1.CLASSES


class ContinuationError(RuntimeError):
    """A fail-closed Experiment A continuation protocol failure."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
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


def _write_once_or_identical(path: Path, payload: bytes) -> None:
    """Make post-seal public deliverables idempotent without replacing files."""
    path = Path(path)
    if path.exists():
        if _sha_file(path) != _sha_bytes(payload):
            raise ContinuationError(f"existing deliverable differs and cannot be overwritten: {path}")
        return
    _write_once(path, payload)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    if tmp.exists():
        raise FileExistsError(tmp)
    with tmp.open("xb") as stream:
        stream.write(_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def higher_q90(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("higher-q90 requires a nonempty finite vector")
    return float(np.partition(x, int(math.ceil(Q90 * (len(x) - 1))))[
        int(math.ceil(Q90 * (len(x) - 1)))])


@dataclass
class B3State:
    """Registered current-state q90 with one update per genuine scorer version."""

    theta: float
    sigma: float
    version: int = 0
    updates: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def initialize(cls, scores: Sequence[float], version: int = 0) -> "B3State":
        x = np.asarray(scores, dtype=np.float64)
        if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError("B3 initialization needs at least two finite normals")
        state = cls(higher_q90(x), max(float(x.std(ddof=1)), SIGMA_FLOOR), version, 0)
        state.history.append({"version": int(version), "theta": state.theta,
                              "sigma": state.sigma, "kind": "initial"})
        return state

    def refresh(self, scores: Sequence[float], version: int) -> bool:
        if int(version) <= self.version:
            if int(version) == self.version:
                return False
            raise ContinuationError("B3 scorer version moved backwards")
        x = np.asarray(scores, dtype=np.float64)
        if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
            raise ContinuationError("B3 current-state references are unavailable")
        raw = higher_q90(x)
        prior = self.theta
        self.theta = float(np.clip(raw, prior - B3_DELTA, prior + B3_DELTA))
        self.sigma = max(float(x.std(ddof=1)), SIGMA_FLOOR)
        self.version = int(version)
        self.updates += 1
        self.history.append({"version": self.version, "theta": self.theta,
                             "sigma": self.sigma, "raw_q90": raw,
                             "clipped_from": prior, "kind": "state_change"})
        return True


def a3b_candidate_quotas(group_ids: Sequence[str], total: int,
                         candidate_count_per_group: Mapping[str, int],
                         namespace: str) -> dict[str, int]:
    """Allocate a bounded candidate pool evenly, resolving remainder by hash."""
    groups = sorted(set(map(str, group_ids)))
    if not groups or total < 1 or set(groups) != set(candidate_count_per_group):
        raise ValueError("candidate quota inputs are incomplete")
    available = {g: int(candidate_count_per_group[g]) for g in groups}
    if any(v < 0 for v in available.values()):
        raise ValueError("candidate counts cannot be negative")
    target = min(int(total), sum(available.values()))
    quotas = {g: 0 for g in groups}
    remaining = target
    active = set(g for g in groups if available[g] > 0)
    while remaining and active:
        q, rem = divmod(remaining, len(active))
        order = sorted(active, key=lambda g: hashlib.sha256(
            f"{namespace}\0quota\0{g}".encode()).hexdigest())
        progressed = 0
        for i, group in enumerate(order):
            request = q + int(i < rem)
            take = min(request, available[group] - quotas[group])
            quotas[group] += take
            progressed += take
        remaining -= progressed
        active = {g for g in active if quotas[g] < available[g]}
        if progressed == 0:
            break
    if sum(quotas.values()) != target:
        raise ContinuationError("candidate allocation did not satisfy its frozen budget")
    return quotas


def a3b_candidate_identity(namespace: str, group_id: str, relative_id: str,
                            view_index: int, patch_index: int) -> str:
    token = (f"{namespace}\0{group_id}\0{relative_id}\0{int(view_index)}\0"
             f"{int(patch_index)}").encode("utf-8")
    return hashlib.sha256(token).hexdigest()


def a3b_kcenter_numpy_reference(candidates: np.ndarray,
                                identity_hashes: Sequence[str],
                                k: int) -> list[int]:
    """Small deterministic reference used in tests; production uses blockwise GPU."""
    x = np.asarray(candidates, dtype=np.float64)
    if (x.ndim != 2 or not len(x) or len(identity_hashes) != len(x) or
            not np.isfinite(x).all() or not 0 < int(k) <= len(x)):
        raise ValueError("invalid k-center candidate matrix")
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise ValueError("zero descriptor in candidate pool")
    x = x / norms[:, None]
    order = sorted(range(len(x)), key=lambda i: str(identity_hashes[i]))
    selected = [order[0]]
    min_distance = 1.0 - x @ x[selected[0]]
    used = set(selected)
    while len(selected) < int(k):
        for idx in used:
            min_distance[idx] = -math.inf
        value = float(np.max(min_distance))
        tie = [i for i in order if i not in used and min_distance[i] == value]
        chosen = tie[0]
        selected.append(chosen)
        used.add(chosen)
        min_distance = np.minimum(min_distance, 1.0 - x @ x[chosen])
    return selected


def a3b_kcenter_cuda(candidates: Any, identity_hashes: Sequence[str], k: int,
                     *, block_rows: int = KC_BLOCK_ROWS,
                     return_indices: bool = False) -> Any:
    """Exact greedy cosine k-center using one candidate-by-center vector at a time.

    The candidate matrix is O(ND); the implementation never forms an N-by-N
    distance matrix. Identity hashes are sorted before selection, so argmin's
    first-index tie rule is the preregistered deterministic tie break.
    """
    import torch
    if not torch.cuda.is_available():
        raise ContinuationError("A3b production k-center requires the authorized CUDA device")
    x = torch.as_tensor(candidates, dtype=torch.float32, device="cuda")
    if (x.ndim != 2 or len(x) == 0 or len(identity_hashes) != len(x) or
            not 0 < int(k) <= len(x) or int(block_rows) < 1):
        raise ValueError("invalid CUDA k-center candidate matrix")
    if not torch.isfinite(x).all().item():
        raise ValueError("nonfinite CUDA k-center descriptor")
    norms = torch.linalg.vector_norm(x, dim=1, keepdim=True)
    if torch.any(norms <= 0).item():
        raise ValueError("zero descriptor in CUDA k-center candidate pool")
    x = x / norms
    order = sorted(range(len(x)), key=lambda i: str(identity_hashes[i]))
    order_tensor = torch.as_tensor(order, device=x.device, dtype=torch.long)
    x = x.index_select(0, order_tensor).contiguous()
    selected = [0]
    max_similarity = torch.empty((len(x),), dtype=torch.float32, device=x.device)
    for start in range(0, len(x), int(block_rows)):
        stop = min(len(x), start + int(block_rows))
        max_similarity[start:stop] = x[start:stop] @ x[0]
    max_similarity[0] = float("inf")
    while len(selected) < int(k):
        candidate = int(torch.argmin(max_similarity).item())
        selected.append(candidate)
        for start in range(0, len(x), int(block_rows)):
            stop = min(len(x), start + int(block_rows))
            similarity = x[start:stop] @ x[candidate]
            max_similarity[start:stop] = torch.maximum(
                max_similarity[start:stop], similarity)
        max_similarity[torch.as_tensor(selected, device=x.device)] = float("inf")
    original_indices = [order[i] for i in selected]
    prototypes = torch.nn.functional.normalize(
        torch.as_tensor(candidates, dtype=torch.float32, device="cuda")[
            torch.as_tensor(original_indices, device="cuda", dtype=torch.long)],
        dim=1).to(dtype=torch.float16).contiguous()
    if return_indices:
        return prototypes, original_indices
    return prototypes


def _file_fingerprints() -> dict[str, str]:
    paths = [
        "AGENTS.md", "docs/calibration/EXPERIMENT_A_AUTHORIZATION_20261106.md",
        "experiments/calibration/a_calibration_scale.py",
        "experiments/calibration/a_calibration_continuation.py",
        "experiments/calibration/stage05.py", "experiments/calibration/profiles.py",
        "experiments/calibration/query_tape.py",
        "experiments/calibration/native.py",
        "experiments/calibration/r1_matched_correction_pilot.py",
        "experiments/calibration/r2_exp2a_c1_strength.py",
        "experiments/calibration/r3_fullstream_2x2.py",
        "experiments/calibration/r6_capacity1024_audiojack_runner.py",
        "experiments/calibration/r7_shuffled_correction_4class.py",
        "experiments/calibration/r9_correction_mechanism_falsification.py",
        "experiments/calibration/r1_runtime_supervisor.py",
        "experiments/p0_protocol_audit.py", "experiments/risk_safe/native_bridge.py",
        "experiments/risk_safe/preflight.py", "experiments/risk_safe/query_protocol.py",
        "experiments/v02_memory.py",
        "test_phase3_AIF_simplified_v22.py", "vmb_visionad_new.py", "v02_memory.py",
    ]
    return {rel: _sha_file(ROOT / rel) for rel in paths if (ROOT / rel).is_file()}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContinuationError(f"Expected JSON object: {path}")
    return value


def _verify_a1() -> dict[str, Any]:
    run = RUNS_ROOT / f"a_calibration_scale_{A1_RUN_ID}"
    state = _read_json(run / "RUN_STATE.json")
    audit = _read_json(run / "integrity_audit.json")
    contract = _read_json(run / "experiment_contract.json")
    if (state.get("status") != "A1_COMPLETE_A2_A3_PENDING" or
            audit.get("status") != "PASS_A1_ONLY" or
            audit.get("source_hashes_unchanged") is not True or
            _sha_file(run / "integrity_audit.json") != state.get("audit_sha256")):
        raise ContinuationError("A1 run state or integrity audit is not valid")
    current = a1._source_fingerprints()
    if current != contract.get("source_hashes_at_contract"):
        raise ContinuationError("A1 source fingerprints changed after its seal")
    seals = {}
    role_receipts = {}
    for category in PRIMARY_CATEGORIES:
        seal = a1.require_stage_a_seal(run / "a1" / category / "stage_a_seal.json")
        seals[category] = seal["seal_sha256"]
        role_path = run / "a1" / category / "role_manifest_private.json"
        roles = _read_json(role_path)
        role_digest = _sha_bytes(_json_bytes(roles))
        if role_digest != seal.get("input_digests", {}).get("role_manifest"):
            raise ContinuationError(f"{category}: authenticated A1 role manifest digest mismatch")
        draw0 = roles.get("draws", {}).get("0", {})
        c64 = set(draw0.get("C64", ()))
        c128 = set(draw0.get("C128", ()))
        cmax = set(draw0.get("Cmax-G", ()))
        ref = set(roles.get("A3b_reference_building_groups", ()))
        h = set(roles.get("H_groups", ()))
        if (len(c64) != 64 or len(c128) != 128 or not c64 < c128 or
                not c128 <= cmax or len(cmax) != len(roles.get("eligible_calibration_groups", ())) or
                len(h) != 32 or ref & (h | c64) or
                len(roles.get("A3b_reference_support_groups", ())) != 4 or
                not set(roles.get("A3b_reference_support_groups", ())) <= ref):
            raise ContinuationError(f"{category}: A1 calibration/A3b role structure is invalid")
        role_receipts[category] = {
            "role_manifest_sha256": role_digest,
            "C64_groups": len(c64), "C128_groups": len(c128),
            "Cmax_G_groups": len(cmax), "H_groups": len(h),
            "A3b_reference_groups": len(ref),
            "A3b_F4_groups": len(roles["A3b_reference_support_groups"]),
            "C64_subset_C128": c64 < c128,
            "C128_subset_Cmax_G": c128 <= cmax,
            "A3b_reference_disjoint_from_C64_H": not bool(ref & (h | c64)),
        }
    if seals != audit.get("stage_a_seals"):
        raise ContinuationError("A1 category seals do not match the integrity audit")
    return {"run_id": A1_RUN_ID, "run_state": state["status"],
            "integrity_status": audit["status"], "integrity_audit_sha256": state["audit_sha256"],
            "stage_a_seals": seals, "role_receipts": role_receipts,
            "source_hashes_at_contract": contract["source_hashes_at_contract"]}


def preflight() -> dict[str, Any]:
    import torch
    import numpy
    import scipy
    from experiments.calibration import r9_correction_mechanism_falsification as r9
    from experiments.calibration import r7_shuffled_correction_4class as r7
    coverage = _runner_coverage_audit()
    if sys.prefix.lower() != r"D:\anaconda3\envs\torch".lower():
        raise ContinuationError("Execution is outside the authorized existing torch environment")
    if torch.__version__ != "2.0.1+cu117":
        raise ContinuationError(f"Unregistered PyTorch version: {torch.__version__}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ContinuationError("The registered single CUDA GPU is unavailable")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT,
                                     text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()
    if branch != "calibration-stage0-contract-20261029" or head != SOURCE_PARENT:
        raise ContinuationError(f"Unexpected branch/source parent: {branch} {head}")
    protected = a1.protected_file_audit()
    if not protected.get("passed"):
        raise ContinuationError("Protected historical source audit failed")
    a1_receipt = _verify_a1()
    run_manifest_path = r9.RUN_NAMESPACE / "manifest" / "run_manifest.json"
    r9_manifest = _read_json(run_manifest_path)
    # R9 is an earlier, independently sealed run from the same branch. Its
    # parent is recorded rather than conflated with this continuation's parent;
    # event and query identities below are re-derived and compared.
    if (r9_manifest.get("branch") != "calibration-stage0-contract-20261029" or
            r9_manifest.get("schema") != "r9.correction_mechanism.run.v1" or
            not isinstance(r9_manifest.get("source_parent_commit"), str) or
            len(r9_manifest["source_parent_commit"]) != 40):
        raise ContinuationError("R9 order/query identity receipt is malformed")
    r9_seal_root = r9.RUN_NAMESPACE / "full" / "stage_seals"
    r9_stage_a = _read_json(r9_seal_root / "stage_a_seal.json")
    r9_stage_b = _read_json(r9_seal_root / "stage_b_seal.json")
    r9_complete = _read_json(r9_seal_root / "complete_seal.json")
    r9_a_sha = _sha_file(r9_seal_root / "stage_a_seal.json")
    r9_b_sha = _sha_file(r9_seal_root / "stage_b_seal.json")
    if (r9_stage_a.get("status") != "SEALED_COMPLETE" or
            r9_stage_b.get("status") != "SEALED_COMPLETE" or
            r9_complete.get("status") != "COMPLETE" or
            r9_stage_b.get("stage_a_seal_sha256") != r9_a_sha or
            r9_complete.get("stage_a_seal_sha256") != r9_a_sha or
            r9_complete.get("stage_b_seal_sha256") != r9_b_sha):
        raise ContinuationError("historical R9 native score/update parity run is not fully sealed")
    current_sources = _file_fingerprints()
    r9_sources = r9_manifest.get("source_fingerprints", {})
    parity_sources = [
        "experiments/calibration/native.py",
        "experiments/calibration/r1_matched_correction_pilot.py",
        "experiments/calibration/r2_exp2a_c1_strength.py",
        "experiments/calibration/r3_fullstream_2x2.py",
        "experiments/calibration/r6_capacity1024_audiojack_runner.py",
        "experiments/calibration/r7_shuffled_correction_4class.py",
        "experiments/calibration/r9_correction_mechanism_falsification.py",
        "experiments/p0_protocol_audit.py", "experiments/risk_safe/native_bridge.py",
        "experiments/v02_memory.py", "test_phase3_AIF_simplified_v22.py",
        "vmb_visionad_new.py"]
    if any(rel not in r9_sources or current_sources.get(rel) != r9_sources.get(rel)
           for rel in parity_sources):
        raise ContinuationError("R9 native parity receipts do not match current pinned scorer/writer sources")
    r9_identity = r9_manifest.get("identity", {})
    r7_contract = r7._frozen_contract()
    class_receipts = {}
    for category in PRIMARY_CATEGORIES:
        entry = r9_identity.get("classes", {}).get(category, {})
        orders = entry.get("orders", {})
        required = {"P0": "SHUFFLE0", "P1": "SHUFFLE11", "P2": "SUPPORT1_SHUFFLE0"}
        if not all(name in orders for name in required.values()):
            raise ContinuationError(f"{category}: registered P0/P1/P2 event maps are absent")
        if len({orders[name]["event_count"] for name in required.values()}) != 1:
            raise ContinuationError(f"{category}: condition stream lengths differ")
        if len({orders[name]["query_count"] for name in required.values()}) != 1:
            raise ContinuationError(f"{category}: condition query budgets differ")
        shuffled = r7_contract["metadata"][category]["events"]
        query_source = tuple(int(v) for v in orders["SHUFFLE0"]["query_positions"])
        q_ids = {str(shuffled[p].image_identity_sha256) for p in query_source}
        if len(q_ids) != len(query_source):
            raise ContinuationError(f"{category}: registered query tape repeats an image identity")
        for name, order in required.items():
            events = r9.make_order_events(shuffled, order)
            event_rows = [{"position": int(e.position),
                           "original_position": int(e.original_position),
                           "image_identity_sha256": str(e.image_identity_sha256),
                           "source_group_sha256": str(e.source_group_sha256),
                           "camera_id": str(e.camera_id)} for e in events]
            if r7._sha256_json(event_rows) != orders[order]["identity_map_sha256"]:
                raise ContinuationError(f"{category}/{name}: event identities do not re-derive")
            actual_q = [i for i, e in enumerate(events)
                        if str(e.image_identity_sha256) in q_ids]
            if (actual_q != list(orders[order]["query_positions"]) or
                    r7._sha256_json(actual_q) != orders[order]["query_position_sha256"]):
                raise ContinuationError(f"{category}/{name}: fixed Query Tape identity map changed")
        class_receipts[category] = {
            key: {"events": orders[order]["event_count"],
                  "queries": orders[order]["query_count"],
                  "identity_map_sha256": orders[order]["identity_map_sha256"],
                  "query_position_sha256": orders[order]["query_position_sha256"]}
            for key, order in required.items()}
    usage = shutil.disk_usage("E:/")
    if usage.free < MIN_FREE_BYTES:
        raise ContinuationError(f"free disk is below the registered 30 GiB minimum: {usage.free}")
    gpu_free, gpu_total = torch.cuda.mem_get_info(0)
    if numpy.__version__ != "1.24.1" or scipy.__version__ != "1.11.2":
        raise ContinuationError(f"Unregistered NumPy/SciPy: {numpy.__version__}/{scipy.__version__}")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090 Ti":
        raise ContinuationError(f"Unexpected CUDA device: {torch.cuda.get_device_name(0)}")
    return {
        "schema": "a_calibration_continuation.preflight.v1",
        "status": "PREFLIGHT_PASS", "protocol_id": PROTOCOL_ID,
        "authorization_id": AUTHORIZATION_ID,
        "git": {"branch": branch, "head": head},
        "runtime": {"python": sys.version, "executable": sys.executable,
                    "torch": torch.__version__, "numpy": numpy.__version__,
                    "scipy": scipy.__version__, "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(0),
                    "gpu_free_bytes": int(gpu_free), "gpu_total_bytes": int(gpu_total)},
        "disk": {"drive": "E:/", "free_bytes": int(usage.free),
                 "minimum_free_bytes": MIN_FREE_BYTES,
                 "minimum_free_passed": usage.free >= MIN_FREE_BYTES},
        "a1_receipt": a1_receipt,
        "r9_identity_manifest_sha256": _sha_file(run_manifest_path),
        "r9_source_parent_commit": r9_manifest["source_parent_commit"],
        "registered_conditions": class_receipts,
        "query_protocol": {"source": "R9 verified Q-A-TS-v1 tape",
                           "rate": 0.03, "seed": 37, "bins": 16,
                           "max_per_official_group": 1},
        "dynamic_config": {"greedy_capacity": K_GREEDY,
                           "correction_capacity": K_CORRECTION,
                           "correction_strength": CORRECTION_STRENGTH,
                           "reference_writer": "native common-C0, theta/sigma frozen at C64 M0; A3a uses its own current theta/sigma"},
        "a3b_kcenter": {"candidate_pool_cap": KC_CANDIDATE_CAP,
                        "prototype_count": KC_PROTOTYPE_COUNT,
                        "candidate_allocation": "equal per Source Group; identity-hashed ordering",
                        "tie_break": "minimum SHA-256 candidate identity",
                        "distance": "1 - cosine similarity; incremental blockwise maximum-minimum"},
        "protected_file_audit": protected,
        "source_fingerprints": current_sources,
        "native_parity_reference": {
            "r9_manifest_sha256": _sha_file(run_manifest_path),
            "r9_stage_a_seal_sha256": r9_a_sha,
            "r9_stage_b_seal_sha256": r9_b_sha,
            "r9_complete_seal_sha256": _sha_file(r9_seal_root / "complete_seal.json"),
            "compatible_sources_match": True,
            "new_disposable_train_normal_smoke_required": True,
        },
        "runner_coverage_audit": coverage,
        "masks_read": False, "confirmation_categories_accessed": False,
    }


def _opaque_group(sample_key: str) -> str:
    return hashlib.sha256(f"official-sample-group\0{sample_key}".encode("utf-8")).hexdigest()


def _training_group_views(category: str, train_rows: Sequence[Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Select the A1 deterministic one-view-per-normal-group image identities."""
    groups: dict[str, list[Any]] = {}
    for row in train_rows:
        if bool(row.is_good):
            groups.setdefault(str(row.sample_key), []).append(row)
    selected: dict[str, Any] = {}
    for group, rows in groups.items():
        relative_id = a1._a1_group_view(rows, category)
        matches = [row for row in rows if str(row.relative_id).replace("\\", "/") == relative_id]
        if len(matches) != 1:
            raise ContinuationError(f"{category}/{group}: A1 deterministic view is not unique")
        selected[group] = matches[0]
    by_opaque = {_opaque_group(group): row for group, row in selected.items()}
    return selected, by_opaque


def _encode_identity_items(impl: Any, backbone: Any, cfg: Any,
                           paths_and_groups: Sequence[tuple[str, str]], *,
                           category: str) -> list[dict[str, Any]]:
    import torch
    items: list[dict[str, Any]] = []
    for index, (path, group_sha) in enumerate(paths_and_groups):
        # Keep the native p0 wrapper as the sole preprocessing/feature entrypoint.
        from experiments import p0_protocol_audit as p0
        image, features = p0.encode_query(impl, backbone, path, cfg)
        if (len(features) != 1 or tuple(features[0].shape) != (784, 4096) or
                features[0].dtype != torch.float32 or
                not torch.isfinite(features[0]).all().item()):
            raise ContinuationError(f"{category}: calibration feature contract failed at {index}")
        items.append({"rgb": np.asarray(image.convert("RGB"), dtype=np.uint8).copy(),
                      "features": features[0].detach().to("cpu").contiguous().numpy().copy(),
                      "source_group_sha256": str(group_sha)})
    return items


def _calibration_items(category: str, support_seed: int, role_manifest: Mapping[str, Any],
                       train_rows: Sequence[Any], impl: Any, backbone: Any,
                       cfg: Any, p0: Any, r7: Any, r9: Any
                       ) -> dict[str, list[dict[str, Any]]]:
    """Build nested C64/C128/Cmax-G identity-view normal feature sets."""
    base_info = r7._frozen_contract()["metadata"][category]
    info = (base_info if support_seed == 0 else
            r9.support1_class_info(category, base_info))
    c64_items, _receipt = r9._make_r9_c64_items(
        category, info, p0, impl, backbone, cfg, r7, support_seed=support_seed)
    selected, by_opaque = _training_group_views(category, train_rows)
    draw = role_manifest["draws"]["0"]
    raw_c64 = set(role_manifest["historical_C64_groups"])
    mapped_c64 = {_opaque_group(group) for group in raw_c64}
    if len(c64_items) != 64 or {item["source_group_sha256"] for item in c64_items} != mapped_c64:
        raise ContinuationError(f"{category}: R7 C64 identities differ from authenticated A1 C64")
    path_records: dict[str, tuple[str, str]] = {}
    for size in ("C128", "Cmax-G"):
        values = list(draw[size])
        if len(values) != len(set(values)):
            raise ContinuationError(f"{category}: duplicate groups in authenticated {size}")
        for raw_group in values:
            opaque = _opaque_group(str(raw_group))
            row = by_opaque.get(opaque)
            if row is None or str(raw_group) not in selected:
                raise ContinuationError(f"{category}: missing legal normal TRAIN group {raw_group}")
            path_records[opaque] = (str(row.image_path), opaque)
    c128_groups = set(map(str, draw["C128"]))
    cmax_groups = set(map(str, draw["Cmax-G"]))
    if not raw_c64 <= c128_groups <= cmax_groups:
        raise ContinuationError(f"{category}: nested A1 C64/C128/Cmax-G memberships changed")
    c128_extra = [path_records[_opaque_group(g)] for g in draw["C128"]
                  if str(g) not in raw_c64]
    cmax_extra = [path_records[_opaque_group(g)] for g in draw["Cmax-G"]
                  if str(g) not in c128_groups]
    c128_items = list(c64_items) + _encode_identity_items(
        impl, backbone, cfg, c128_extra, category=category)
    cmax_items = c128_items + _encode_identity_items(
        impl, backbone, cfg, cmax_extra, category=category)
    if (len(c128_items) != 128 or len(cmax_items) != len(draw["Cmax-G"]) or
            {i["source_group_sha256"] for i in cmax_items} !=
            {_opaque_group(str(g)) for g in draw["Cmax-G"]}):
        raise ContinuationError(f"{category}: calibration feature set size/identity mismatch")
    return {"C64": c64_items, "C128": c128_items, "Cmax-G": cmax_items}


@dataclass
class NativeTrajectory:
    name: str
    bank: Any
    writer: Any
    fixed_sha256: str
    correction_cfg: Any
    correction_cb: Any
    source_groups: dict[int, set[str]] = field(default_factory=dict)
    g_version: int = 0
    gc_version: int = 0
    g_digest: str = ""
    gc_digest: str = ""


def _build_native_trajectory(name: str, *, impl: Any, p0: Any, r1: Any,
                             r3: Any, r6: Any, support_cache: Mapping[str, Any],
                             support_paths: Sequence[str], cfg: Any,
                             expected_fixed_sha256: str | None,
                             correction: bool = True) -> NativeTrajectory:
    import torch
    from types import SimpleNamespace
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.risk_safe.native_bridge import dynamic_codebooks
    from v02_memory import MemoryArm

    impl.set_seed(0)
    bank = p0.build_vmb(impl, support_cache, list(support_paths), cfg)
    fixed_sha = _fixed_bank_digest(bank)
    if expected_fixed_sha256 is not None:
        receipt = r3._check_bank_receipt(bank, p0, expected_fixed_sha256, r1)
        fixed_sha = str(receipt["actual_structured_sha256"])
    r3._validate_correction_config(bank)
    correction_cfg, correction_banks = r6._configure_r6_correction_capacity(bank)
    correction_cb = bank.corr_cb["p0"]["id"][0]
    if (len(correction_banks) != 1 or correction_cb is not correction_banks[0] or
            int(correction_cb.K) != K_CORRECTION or correction_cb.size() != 0):
        raise ContinuationError(f"{name}: native C1-Full K=1024 did not start empty")
    correction_cfg.enabled = False
    writer = MemoryArm("greedy_coreset", bank,
        SimpleNamespace(flat=K_GREEDY, stm=K_GREEDY, ltm=0, buffer=0,
                        min_events=2, promote_every=16,
                        cap_bytes=2 * 1024**3), seed=0)
    if any(int(cb.K) != K_GREEDY or cb.size() != 0 for cb in dynamic_codebooks(bank)):
        raise ContinuationError(f"{name}: native Greedy K=1024 did not start empty")
    bank.use_defect_bank = False
    digest = _fixed_bank_digest(bank)
    if digest != fixed_sha:
        raise ContinuationError(f"{name}: fixed normal reference changed during initialization")
    source_groups: dict[int, set[str]] = {}
    trajectory = NativeTrajectory(name=name, bank=bank, writer=writer,
        fixed_sha256=fixed_sha, correction_cfg=correction_cfg,
        correction_cb=correction_cb, source_groups=source_groups)
    trajectory.g_digest = r9_scorer_digest(fixed_sha, r1._memory_content_sha256(bank))
    trajectory.gc_digest = r9_state_digest(bank, fixed_sha, source_groups,
                                            correction_cfg, correction_cb, r1)
    return trajectory


def _verify_native_greedy_parity(*, impl: Any, p0: Any, r1: Any, r3: Any,
                                support_cache: Mapping[str, Any],
                                support_paths: Sequence[str], cfg: Any,
                                expected_fixed_sha256: str,
                                parity_row: Any, backbone: Any, category: str
                                ) -> dict[str, Any]:
    """One disposable authorized TRAIN-normal native score/update parity smoke."""
    import torch
    from types import SimpleNamespace
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.risk_safe.native_bridge import (
        dynamic_codebooks, verify_native_update_parity, verify_untouched_parity)
    from v02_memory import MemoryArm

    # Compatibility imports use the repository's canonical native bridge.
    bank = p0.build_vmb(impl, support_cache, list(support_paths), cfg)
    fixed_receipt = r3._check_bank_receipt(bank, p0, expected_fixed_sha256, r1)
    if _fixed_bank_digest(bank) != fixed_receipt["actual_structured_sha256"]:
        raise ContinuationError("parity smoke fixed-bank digest differs from native receipt")
    if bank.corr_cfg.enabled or bank.use_defect_bank:
        raise ContinuationError("parity smoke common C0 bank has correction/defect scoring enabled")
    if not bool(parity_row.is_good):
        raise ContinuationError("native Greedy parity smoke input is not a TRAIN normal")
    image, features = p0.encode_query(impl, backbone, str(parity_row.image_path), cfg)
    base_map, top_ids, score, summary = r1._native_dense_score(
        impl, bank, image, features, cfg)
    score_parity = verify_untouched_parity(impl, bank, image, features, cfg)
    settings = SimpleNamespace(flat=K_GREEDY, stm=K_GREEDY, ltm=0, buffer=0,
        min_events=2, promote_every=16, cap_bytes=2 * 1024**3)
    parity_arm = MemoryArm("greedy_coreset", bank, settings, seed=0)
    update_parity = verify_native_update_parity(
        impl, parity_arm, features,
        {"per_layer_summary": {"layer0": {
            "topk_patch_idx": np.asarray(top_ids, dtype=np.int64).tolist()} }},
        score=float(score), theta=float(score) + max(1.0, abs(float(score))),
        sigma=SIGMA_FLOOR, step=0)
    occupancy = [int(cb.size()) for cb in dynamic_codebooks(bank)]
    if not any(occupancy):
        raise ContinuationError("native Greedy parity smoke did not perform a genuine write")
    if _fixed_bank_digest(bank) != fixed_receipt["actual_structured_sha256"]:
        raise ContinuationError("native Greedy parity smoke mutated its fixed Support bank")
    return {"status": "PASS", "scope": "disposable authorized TRAIN-normal parity only",
        "category": category, "input_is_verified_train_normal": True,
        "native_score_parity": score_parity, "native_update_parity": update_parity,
        "post_write_occupancy": occupancy,
        "fixed_support_sha256": fixed_receipt["actual_structured_sha256"],
        "evaluation_trajectory_reused": False,
        "test_labels_accessed": False, "test_masks_accessed": False}


def r9_scorer_digest(fixed_sha: str, greedy_sha: str) -> str:
    from experiments.calibration.r9_correction_mechanism_falsification import _scorer_digest
    return _scorer_digest(fixed_sha, greedy_sha)


def r9_state_digest(bank: Any, fixed_sha: str, source_groups: Mapping[int, set[str]],
                    correction_cfg: Any, correction_cb: Any, r1: Any) -> str:
    from experiments.calibration import r2_exp2a_c1_strength as exp2a
    from experiments.calibration.r9_correction_mechanism_falsification import _state_digest_for_codebook
    return _state_digest_for_codebook(bank, fixed_sha, source_groups,
                                      correction_cfg, correction_cb, exp2a, r1)


def _native_event_scores(trajectory: NativeTrajectory, *, impl: Any, r1: Any,
                         exp2a: Any, r9: Any, cfg: Any, image: Any,
                         feature_layers: Sequence[Any], source_group_sha256: str,
                         position: int, use_correction: bool = True
                         ) -> dict[str, Any]:
    import numpy as np
    import torch
    base_map, top_ids, base_score, summary = r1._native_dense_score(
        impl, trajectory.bank, image, feature_layers, cfg)
    top_ids = np.asarray(top_ids, dtype=np.int64)
    if (len(top_ids) != 32 or not np.array_equal(
            top_ids, np.asarray(summary["layer0"]["topk_patch_idx"], dtype=np.int64))):
        raise ContinuationError(f"{trajectory.name}: native Top-32 selection changed")
    selected = impl.l2_normalize(feature_layers[0].to(trajectory.bank.device))[
        torch.as_tensor(top_ids, device=trajectory.bank.device, dtype=torch.long)]
    result = {"base_map": base_map, "top_ids": top_ids,
              "base_score": float(base_score), "selected": selected,
              "summary": summary}
    if use_correction:
        full_score, diagnostics = r9._score_codebook(
            impl, trajectory.bank, r1, exp2a, trajectory.correction_cfg,
            trajectory.correction_cb, trajectory.source_groups, selected,
            str(source_group_sha256), int(position), base_map, top_ids)
        result["gc_score"] = float(full_score)
        result["gc_diagnostics"] = diagnostics
    return result


def _score_reference_items(trajectory: NativeTrajectory, items: Sequence[Mapping[str, Any]],
                           *, scorer: str, impl: Any, r1: Any, exp2a: Any,
                           r9: Any, cfg: Any, event_position: int
                           ) -> tuple[list[float], float]:
    import torch
    from PIL import Image
    scores: list[float] = []
    started = time.perf_counter()
    for index, item in enumerate(items):
        image = Image.fromarray(np.asarray(item["rgb"], dtype=np.uint8), mode="RGB")
        raw_features = item["features"]
        if isinstance(raw_features, (list, tuple)):
            if len(raw_features) != 1:
                raise ContinuationError("calibration item has a nonregistered feature-layer count")
            raw_features = raw_features[0]
        feature = torch.from_numpy(np.ascontiguousarray(raw_features)).to(
            trajectory.bank.device)
        scored = _native_event_scores(trajectory, impl=impl, r1=r1,
            exp2a=exp2a, r9=r9, cfg=cfg, image=image, feature_layers=[feature],
            source_group_sha256=str(item["source_group_sha256"]),
            position=event_position, use_correction=(scorer == "GC"))
        if scorer == "G":
            scores.append(float(scored["base_score"]))
        elif scorer == "GC":
            diagnostics = scored["gc_diagnostics"]
            if int(diagnostics.get("same_source_excluded_anchor_slots", 0)):
                raise ContinuationError("calibration reference overlaps an online C1 source group")
            scores.append(float(scored["gc_score"]))
        else:
            raise ValueError(f"unsupported calibration scorer {scorer}")
    return scores, time.perf_counter() - started


def _trajectory_writer_update(trajectory: NativeTrajectory, *, position: int,
                              feature_layers: Sequence[Any], top_ids: Any,
                              selected_features: Any, score: float, theta: float,
                              sigma: float, queried: bool, label: int | None,
                              prediction: int, impl: Any, r1: Any
                              ) -> dict[str, Any]:
    """Apply the exact native common-C0 Greedy writer and audit its state change."""
    import torch
    from experiments.risk_safe.native_bridge import dynamic_codebooks
    before_sha = r1._memory_content_sha256(trajectory.bank)
    writer_audit = trajectory.writer.audit()
    before_rows = [cb.protos[:int(cb.size())].detach().to("cpu").contiguous().clone()
                   for cb in dynamic_codebooks(trajectory.bank)]
    before_sizes = [int(cb.size()) for cb in dynamic_codebooks(trajectory.bank)]

    def update(*, position: int, feature_layers: Sequence[Any], candidate_ids: Any,
               score: float, theta: float, sigma: float) -> Mapping[str, Any]:
        before = r1._memory_content_sha256(trajectory.bank)
        trajectory.writer.update(impl, list(feature_layers),
            {0: candidate_ids.tolist()}, int(position), score=float(score),
            theta=float(theta), sigma=float(sigma))
        after = r1._memory_content_sha256(trajectory.bank)
        audit_after = trajectory.writer.audit()
        return_rows = [cb.protos[:int(cb.size())].detach().to("cpu").contiguous()
                       for cb in dynamic_codebooks(trajectory.bank)]
        row_changes = []
        for layer, (old, new) in enumerate(zip(before_rows, return_rows)):
            shared = min(len(old), len(new))
            replaced = int(torch.any(old[:shared] != new[:shared], dim=1).sum().item()) if shared else 0
            row_changes.append({"layer": layer, "size_before": int(before_sizes[layer]),
                "size_after": int(len(new)), "appended": max(0, int(len(new)) - int(len(old))),
                "removed": max(0, int(len(old)) - int(len(new))),
                "changed_existing_slots": replaced,
                "admitted_or_replaced_count": max(0, int(len(new)) - int(len(old))) + replaced})
        return {"accepted_patches_delta": int(audit_after["accepted_patches"] -
                                                  writer_audit["accepted_patches"]),
                "skipped_patches_delta": int(audit_after["skipped_patches"] -
                                                 writer_audit["skipped_patches"]),
                "greedy_occupancy": [int(cb.size()) for cb in dynamic_codebooks(trajectory.bank)],
                "allocated_state_bytes": int(audit_after.get("allocated_state_bytes", 0)),
                "prototype_slot_changes": row_changes,
                "_memory_content_changed": before != after}

    action = r1.common_c0_greedy_writer_action(
        position=int(position), c0_score=float(score), theta=float(theta),
        sigma=float(sigma), c0_prediction=int(prediction),
        candidate_ids=np.asarray(top_ids, dtype=np.int64),
        candidate_features=selected_features.detach().float().cpu().numpy(),
        feature_layers=feature_layers, queried=bool(queried),
        disclosed_label=label, eligibility=impl.select_informative_normal_write,
        update=update if queried and label == 0 else _deny_writer_update)
    action.pop("_memory_content_changed", None)
    after_sha = r1._memory_content_sha256(trajectory.bank)
    changed = before_sha != after_sha
    audit_final = trajectory.writer.audit()
    occupancy_final = [int(cb.size()) for cb in dynamic_codebooks(trajectory.bank)]
    if label == 1 and queried and action.get("action") != "verified_defect_no_normal_write":
        raise ContinuationError("queried defect was admitted to Greedy normal memory")
    if not queried and action.get("action") != "no_normal_memory_write":
        raise ContinuationError("unqueried TEST event caused a Greedy memory write")
    if changed != bool(action.get("accepted_patches_delta", 0)):
        # The writer may accept a candidate that maps to an identical existing
        # prototype. The native state digest is authoritative in that case.
        if changed and not int(action.get("accepted_patches_delta", 0)):
            raise ContinuationError("native Greedy state changed without an accepted write")
    return {"action": action, "state_changed": changed,
            "memory_sha256_before": before_sha, "memory_sha256_after": after_sha,
            "greedy_occupancy": occupancy_final,
            "allocated_state_bytes": int(audit_final.get("allocated_state_bytes", 0)),
            "prototype_slot_changes": action.get("prototype_slot_changes", [])}


def _deny_writer_update(**_kwargs: Any) -> Mapping[str, Any]:
    raise ContinuationError("writer update callback invoked without queried verified-normal feedback")


def _update_native_correction(trajectory: NativeTrajectory, *, position: int,
                              feature_layers: Sequence[Any], top_ids: Any,
                              label: int, base_score: float, theta: float,
                              outcome: str, source_group_sha256: str,
                              impl: Any, r1: Any, exp2a: Any
                              ) -> dict[str, Any]:
    from experiments.calibration.r9_correction_mechanism_falsification import _state_digest_for_codebook
    before = _state_digest_for_codebook(trajectory.bank, trajectory.fixed_sha256,
        trajectory.source_groups, trajectory.correction_cfg,
        trajectory.correction_cb, exp2a, r1)
    _result, layer_stats, trace = exp2a._capture_update(
        impl, trajectory.bank, r1, "Experiment-A-C1-Full",
        trajectory.correction_cfg, trajectory.correction_cb, feature_layers,
        np.asarray(top_ids, dtype=np.int64), 1 if int(label) == 1 else -1,
        float(base_score), float(theta), str(outcome), int(position),
        trajectory.source_groups, str(source_group_sha256))
    after = _state_digest_for_codebook(trajectory.bank, trajectory.fixed_sha256,
        trajectory.source_groups, trajectory.correction_cfg,
        trajectory.correction_cb, exp2a, r1)
    correction_array_names = ("protos", "beta", "radius", "kappa", "count",
        "outcome_fp", "outcome_fn", "outcome_tn", "outcome_tp", "last_step")
    correction_storage_bytes = sum(
        int(getattr(trajectory.correction_cb, name).numel()) *
        int(getattr(trajectory.correction_cb, name).element_size())
        for name in correction_array_names)
    return {"state_changed": before != after,
            "state_sha256_before": before, "state_sha256_after": after,
            "occupancy": int(trajectory.correction_cb.size()),
            "capacity": int(trajectory.correction_cb.K),
            "allocated_state_bytes": int(correction_storage_bytes),
            "integer_stats": {str(k): int(v) for k, v in layer_stats.items()
                              if isinstance(v, (int, np.integer))},
            "trace_integer_fields": {str(k): int(v) for k, v in trace.items()
                                      if isinstance(v, (int, np.integer))}}


def _b3_prediction(state: B3State, *, score: float, state_digest: str,
                   state_version: int) -> dict[str, Any]:
    if int(state.version) != int(state_version):
        raise ContinuationError("B3 threshold version does not match the current scorer state")
    return {"prediction": int(float(score) >= float(state.theta)),
            "threshold": float(state.theta), "sigma": float(state.sigma),
            "scorer_state_version": int(state_version),
            "scorer_state_sha256": str(state_digest)}


def _record_static_c64_prediction(decisions: dict[str, Any], trajectory: Any,
                                  state: B3State, score: float) -> None:
    decisions[f"{trajectory.name}:STATIC:C64"] = _b3_prediction(
        state, score=float(score), state_digest=trajectory.g_digest,
        state_version=trajectory.g_version)


def _initialize_b3_states(trajectory: NativeTrajectory,
                          calibration: Mapping[str, Sequence[Mapping[str, Any]]],
                          *, impl: Any, r1: Any, exp2a: Any, r9: Any,
                          cfg: Any, correction: bool = True
                          ) -> tuple[dict[tuple[str, str], B3State], dict[str, Any],
                                     dict[str, list[float]]]:
    states: dict[tuple[str, str], B3State] = {}
    costs: dict[str, Any] = {}
    score_vectors: dict[str, list[float]] = {}
    scorers = ("G", "GC") if correction else ("G",)
    for size, items in calibration.items():
        for scorer in scorers:
            scores, seconds = _score_reference_items(
                trajectory, items, scorer=scorer, impl=impl, r1=r1,
                exp2a=exp2a, r9=r9, cfg=cfg, event_position=0)
            score_vectors[f"{scorer}:{size}"] = [float(v) for v in scores]
            version = trajectory.g_version if scorer == "G" else trajectory.gc_version
            digest = trajectory.g_digest if scorer == "G" else trajectory.gc_digest
            state = B3State.initialize(scores, version=version)
            states[(scorer, size)] = state
            costs[f"{scorer}:{size}"] = {
                "score_count": len(scores), "seconds": seconds,
                "calibration_reference_feature_bytes": int(
                    len(items) * 784 * 4096 * np.dtype(np.float32).itemsize),
                "score_sha256": _sha_bytes(_json_bytes(scores)),
                "state_version": int(version), "state_sha256": digest,
                "theta": state.theta, "sigma": state.sigma,
            }
    return states, costs, score_vectors


def _refresh_b3_states(trajectory: NativeTrajectory,
                       states: Mapping[tuple[str, str], B3State],
                       calibration: Mapping[str, Sequence[Mapping[str, Any]]],
                       *, g_changed: bool, gc_changed: bool,
                       impl: Any, r1: Any, exp2a: Any, r9: Any,
                       cfg: Any, event_position: int
                       ) -> tuple[list[dict[str, Any]], float]:
    refreshed: list[dict[str, Any]] = []
    total_seconds = 0.0
    for scorer, changed in (("G", g_changed), ("GC", gc_changed)):
        if not changed:
            continue
        version = trajectory.g_version if scorer == "G" else trajectory.gc_version
        digest = trajectory.g_digest if scorer == "G" else trajectory.gc_digest
        for size, items in calibration.items():
            state = states[(scorer, size)]
            scores, seconds = _score_reference_items(
                trajectory, items, scorer=scorer, impl=impl, r1=r1,
                exp2a=exp2a, r9=r9, cfg=cfg, event_position=event_position)
            changed_threshold = state.refresh(scores, version=version)
            if not changed_threshold:
                raise ContinuationError("B3 state failed to refresh for a genuine scorer state change")
            total_seconds += seconds
            refreshed.append({"trajectory": trajectory.name, "scorer": scorer,
                "calibration_size": size, "state_version": int(version),
                "state_sha256": digest, "raw_q90": higher_q90(scores),
                "theta": state.theta, "sigma": state.sigma,
                "rescore_seconds": seconds,
                "calibration_score_sha256": _sha_bytes(_json_bytes(scores)),
                "apply_from_position": int(event_position)})
    return refreshed, total_seconds


class _IndexedFeatureCache:
    """Compact memmap keyed by original JSON positions, including smoke subsets."""
    def __init__(self, array: np.ndarray, original_positions: Sequence[int]):
        self.array = array
        self.positions = {int(position): index
                          for index, position in enumerate(original_positions)}

    def __getitem__(self, original_position: int) -> np.ndarray:
        try:
            row = self.positions[int(original_position)]
        except KeyError as exc:
            raise ContinuationError(
                f"TEST feature cache does not contain JSON position {original_position}") from exc
        return self.array[row]

    def close(self) -> None:
        mmap = getattr(self.array, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _encode_test_feature_cache(category: str, events: Sequence[Any], *, impl: Any,
                               backbone: Any, cfg: Any, p0: Any,
                               scratch_dir: Path) -> tuple[_IndexedFeatureCache, dict[str, Any]]:
    """Encode the requested TEST identities once and key rows by JSON position."""
    import torch
    scratch_dir.mkdir(parents=True, exist_ok=False)
    feature_path = scratch_dir / "test_features.float32.npy"
    if feature_path.exists():
        raise FileExistsError(feature_path)
    count = len(events)
    original_positions = [int(event.original_position) for event in events]
    if (not count or len(set(original_positions)) != count or
            min(original_positions) < 0):
        raise ContinuationError(f"{category}: TEST feature identities are not unique source rows")
    mmap = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float32,
                                     shape=(count, 784, 4096))
    digest = hashlib.sha256()
    started = time.perf_counter()
    ordered_events = sorted(events, key=lambda e: int(e.original_position))
    cache_index = {int(event.original_position): index
                   for index, event in enumerate(ordered_events)}
    for event in ordered_events:
        if not Path(event.image_path).is_file():
            raise ContinuationError(f"{category}: TEST image is missing at {event.original_position}")
        _image, feature_layers = p0.encode_query(impl, backbone, str(event.image_path), cfg)
        if (len(feature_layers) != 1 or tuple(feature_layers[0].shape) != (784, 4096) or
                feature_layers[0].dtype != torch.float32 or
                not torch.isfinite(feature_layers[0]).all().item()):
            raise ContinuationError(f"{category}: TEST DINO feature contract failed")
        row = feature_layers[0].detach().to("cpu").contiguous().numpy()
        mmap[cache_index[int(event.original_position)]] = row
        digest.update(row.tobytes(order="C"))
    mmap.flush()
    del mmap
    fd = os.open(feature_path, os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    elapsed = time.perf_counter() - started
    result = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    receipt = {"path": str(feature_path), "sha256": digest.hexdigest(),
               "count": count, "original_positions_sha256": _sha_bytes(
                   _json_bytes(sorted(original_positions))),
               "shape": [count, 784, 4096], "dtype": "float32",
               "bytes": int(feature_path.stat().st_size),
               "encoding_seconds": elapsed, "encodings": count,
               "labels_read": False, "masks_read": False}
    return _IndexedFeatureCache(result, [int(event.original_position)
                                         for event in ordered_events]), receipt


def _a3b_reference_trajectories(category: str, *, impl: Any, p0: Any,
                                r1: Any, r3: Any, r6: Any, r9: Any,
                                backbone: Any, cfg: Any, train_rows: Sequence[Any],
                                role_manifest: Mapping[str, Any],
                                support_cache: Mapping[str, Any]
                                ) -> tuple[NativeTrajectory, NativeTrajectory, dict[str, Any]]:
    """Construct native F4 and equal-byte FEXP-KC reference trajectories."""
    import torch
    from experiments.calibration import r7_shuffled_correction_4class as r7
    group_rows, _by_opaque = _training_group_views(category, train_rows)
    pool_raw = list(role_manifest["A3b_reference_building_groups"])
    f4_raw = list(role_manifest["A3b_reference_support_groups"])
    if len(pool_raw) < 4 or len(f4_raw) != 4 or not set(f4_raw) <= set(pool_raw):
        raise ContinuationError(f"{category}: A3b reference-building memberships are invalid")
    pool = []
    for group in pool_raw:
        row = group_rows.get(str(group))
        if row is None:
            raise ContinuationError(f"{category}: A3b reference group is not a TRAIN normal")
        pool.append((str(group), _opaque_group(str(group)), str(row.relative_id).replace("\\", "/"),
                     str(row.image_path)))
    f4_rows = [group_rows[str(group)] for group in f4_raw]
    f4_paths = [str(row.image_path) for row in f4_rows]
    if len(set(f4_paths)) != 4:
        raise ContinuationError(f"{category}: A3b F4 does not contain four unique images")

    # The augmentation family is frozen by the native reference encoder. It is
    # deterministic per image; candidate selection uses only identity hashes.
    encoded_by_group: dict[str, dict[str, Any]] = {}
    candidate_counts: dict[str, int] = {}
    first_view_count: int | None = None
    first_features: dict[str, Any] | None = None
    candidate_feature_started = time.perf_counter()
    for group, opaque, relative_id, path in pool:
        if first_view_count is None:
            cached = p0.encode_reference_features(impl, backbone, [path], cfg)
            entries = cached[path]
            first_view_count = len(entries)
            first_features = cached
        else:
            cached = p0.encode_reference_features(impl, backbone, [path], cfg)
            entries = cached[path]
            if len(entries) != first_view_count:
                raise ContinuationError("A3b native augmentation view count varies by image")
        encoded_by_group[opaque] = {"group": group, "relative_id": relative_id,
                                    "path": path, "entries": entries}
        candidate_counts[opaque] = int(len(entries)) * 784
    if first_view_count is None or first_features is None:
        raise ContinuationError("A3b reference pool has no encodable groups")
    candidate_feature_extraction_seconds = time.perf_counter() - candidate_feature_started
    candidate_selection_started = time.perf_counter()
    quotas = a3b_candidate_quotas(
        list(candidate_counts), KC_CANDIDATE_CAP, candidate_counts,
        f"a3b-fexp-kc-v1\0{category}")
    selected_candidates: list[tuple[str, np.ndarray]] = []
    for opaque in sorted(encoded_by_group):
        item = encoded_by_group[opaque]
        identities: list[str] = []
        descriptors: list[np.ndarray] = []
        for view_index, feature_set in enumerate(item["entries"]):
            if len(feature_set) != 1:
                raise ContinuationError("A3b candidate features are not one fused 4096D layer")
            block = feature_set[0].detach().to("cpu").contiguous().numpy()
            if block.shape != (784, 4096) or block.dtype != np.float16:
                raise ContinuationError("A3b candidate feature shape/dtype changed")
            for patch_index in range(784):
                identities.append(a3b_candidate_identity(
                    "a3b-fexp-kc-v1", opaque, item["relative_id"],
                    view_index, patch_index))
            descriptors.append(block)
        matrix = np.concatenate(descriptors, axis=0)
        quota = int(quotas[opaque])
        order = sorted(range(len(identities)), key=lambda i: identities[i])[:quota]
        selected_candidates.extend((identities[i], matrix[i].copy()) for i in order)
    selected_candidates.sort(key=lambda pair: pair[0])
    if len(selected_candidates) != min(KC_CANDIDATE_CAP, sum(candidate_counts.values())):
        raise ContinuationError("A3b candidate cap was not filled exactly")
    candidate_ids = [row[0] for row in selected_candidates]
    candidate_matrix = np.stack([row[1] for row in selected_candidates], axis=0)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ContinuationError("A3b candidate identity hashes are not unique")
    if len(candidate_matrix) < KC_PROTOTYPE_COUNT:
        raise ContinuationError("A3b candidate population cannot supply 8192 prototypes")
    candidate_selection_seconds = time.perf_counter() - candidate_selection_started
    kcenter_started = time.perf_counter()
    prototypes, selected_indices = a3b_kcenter_cuda(candidate_matrix, candidate_ids,
        KC_PROTOTYPE_COUNT, return_indices=True)
    kcenter_seconds = time.perf_counter() - kcenter_started
    if tuple(prototypes.shape) != (KC_PROTOTYPE_COUNT, 4096) or prototypes.dtype != torch.float16:
        raise ContinuationError("A3b k-center did not return the exact fp16 8192x4096 budget")

    reference_build_started = time.perf_counter()
    f4 = _build_native_trajectory("A3b-F4", impl=impl, p0=p0, r1=r1,
        r3=r3, r6=r6, support_cache=support_cache, support_paths=f4_paths,
        cfg=cfg, expected_fixed_sha256=None, correction=False)
    fexp = _build_native_trajectory("A3b-FEXP-KC", impl=impl, p0=p0, r1=r1,
        r3=r3, r6=r6, support_cache=support_cache, support_paths=f4_paths,
        cfg=cfg, expected_fixed_sha256=None, correction=False)
    fixed_cb = fexp.bank.normal_fixed["p0"][0]["id"][0]
    fixed_cb.set_prototypes(prototypes.to(fexp.bank.device))
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.p0_protocol_audit import fixed_bank_profile
    f4_profile = fixed_bank_profile(f4.bank)
    fexp_profile = fixed_bank_profile(fexp.bank)
    expected_bytes = KC_PROTOTYPE_COUNT * 4096 * np.dtype(np.float16).itemsize
    if (f4_profile["total_prototypes"] != KC_PROTOTYPE_COUNT or
            fexp_profile["total_prototypes"] != KC_PROTOTYPE_COUNT or
            f4_profile["total_feature_bytes"] != expected_bytes or
            fexp_profile["total_feature_bytes"] != expected_bytes or
            f4_profile["total_feature_bytes"] != fexp_profile["total_feature_bytes"]):
        raise ContinuationError("A3b F4/FEXP fixed banks are not exactly equal-byte 8192x4096 fp16")
    fexp.fixed_sha256 = _fixed_bank_digest(fexp.bank)
    fexp.g_digest = r9_scorer_digest(fexp.fixed_sha256,
                                    r1._memory_content_sha256(fexp.bank))
    fexp.gc_digest = r9_state_digest(fexp.bank, fexp.fixed_sha256,
        fexp.source_groups, fexp.correction_cfg, fexp.correction_cb, r1)
    native_reference_build_seconds = time.perf_counter() - reference_build_started

    # The altered prototype bank must remain readable by the original native
    # scorer. Compare the bridge and adapter on the same authorized TRAIN image.
    from experiments.risk_safe.native_bridge import native_score, parity_tolerance
    native_compatibility_started = time.perf_counter()
    smoke_image, smoke_features = p0.encode_query(
        impl, backbone, str(f4_rows[0].image_path), cfg)
    score_smokes = {}
    for trajectory in (f4, fexp):
        adapter_score = float(r1._native_dense_score(
            impl, trajectory.bank, smoke_image, smoke_features, cfg)[2])
        bridge_score, _bridge_map = native_score(
            impl, trajectory.bank, smoke_image, smoke_features, cfg)
        delta = parity_tolerance(adapter_score, bridge_score)
        score_smokes[trajectory.name] = {
            "adapter_score": adapter_score, "native_bridge_score": bridge_score,
            "absolute_difference": delta,
            "tolerance": {"atol": 1e-5, "rtol": 1e-4}, "passed": True}
        if _fixed_bank_digest(trajectory.bank) != trajectory.fixed_sha256:
            raise ContinuationError(f"{trajectory.name}: native score smoke mutated fixed references")
    native_compatibility_smoke_seconds = time.perf_counter() - native_compatibility_started

    # A3b construction is frozen before Stage B. Candidate identities and the
    # bounded selected set are hashed; raw feature vectors are never retained.
    receipt = {
        "schema": "a3b.reference_construction.v1",
        "reference_pool_groups": len(pool),
        "reference_pool_group_sha256": _sha_bytes(_json_bytes(sorted(pool_raw))),
        "F4_group_sha256": _sha_bytes(_json_bytes(sorted(f4_raw))),
        "candidate_count_by_group": candidate_counts,
        "candidate_quota_by_group": quotas,
        "candidate_count": len(candidate_ids),
        "candidate_feature_extraction_seconds": candidate_feature_extraction_seconds,
        "candidate_selection_seconds": candidate_selection_seconds,
        "native_reference_bank_build_seconds": native_reference_build_seconds,
        "candidate_identity_sha256": _sha_bytes(_json_bytes(candidate_ids)),
        "candidate_selection": "identity-hash balanced per official group",
        "kcenter": "greedy maximum-minimum cosine distance; first and tie order by candidate identity SHA-256",
        "selected_prototype_identity_sha256": _sha_bytes(_json_bytes(
            [candidate_ids[i] for i in selected_indices])),
        "prototypes_sha256": hashlib.sha256(
            prototypes.detach().to("cpu").contiguous().numpy().tobytes()).hexdigest(),
        "prototype_count": int(prototypes.shape[0]),
        "prototype_dtype": str(prototypes.dtype),
        "prototype_dim": int(prototypes.shape[1]),
        "prototype_bytes": int(expected_bytes),
        "kcenter_seconds": kcenter_seconds,
        "F4_fixed_bank": f4_profile,
        "FEXP_fixed_bank": fexp_profile,
        "feature_extraction_images": len(pool),
        "feature_extraction_views": int(sum(candidate_counts.values()) // 784),
        "native_scorer": "R9/R10 p0_protocol_audit build_vmb and native r1 score adapter",
        "native_score_read_compatibility": score_smokes,
        "native_score_compatibility_smoke_seconds": native_compatibility_smoke_seconds,
        "native_compatibility_smoke_extra_train_encodings": 1,
        "test_outcomes_accessed": False,
    }
    del encoded_by_group, selected_candidates, candidate_matrix, prototypes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return f4, fexp, receipt


def _static_trajectory_clone(source: NativeTrajectory, name: str) -> NativeTrajectory:
    import copy
    from experiments.calibration.native import _fixed_bank_digest
    bank = copy.deepcopy(source.bank)
    bank.use_defect_bank = False
    bank.corr_cfg.enabled = False
    fixed_sha = _fixed_bank_digest(bank)
    if fixed_sha != source.fixed_sha256:
        raise ContinuationError(f"{name}: static clone changed the fixed native reference")
    return NativeTrajectory(name=name, bank=bank, writer=None,
        fixed_sha256=fixed_sha, correction_cfg=bank.corr_cfg,
        correction_cb=bank.corr_cb["p0"]["id"][0], source_groups={},
        g_version=0, gc_version=0,
        g_digest=r9_scorer_digest(fixed_sha,
            __import__("experiments.calibration.r1_matched_correction_pilot",
                      fromlist=["_memory_content_sha256"])._memory_content_sha256(bank)),
        gc_digest="")


def _run_condition(category: str, condition: str, order: str, support_seed: int,
                   events: Sequence[Any], query_positions: Sequence[int],
                   trajectories: Sequence[dict[str, Any]],
                   test_features: np.ndarray, *, impl: Any, r1: Any, exp2a: Any,
                   r3: Any, r7: Any, r9: Any, cfg: Any, output_dir: Path,
                   category_started: float, category_deadline: float,
                   role_manifest: Mapping[str, Any], query_identity_sha256: str,
                   cmax_items: Sequence[Mapping[str, Any]] | None = None,
                   smoke: bool = False
                   ) -> dict[str, Any]:
    """Run one sealed Stage A condition; each score is written before feedback."""
    import torch
    from PIL import Image

    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse condition output: {output_dir}")
    budget = r7._budget(output_dir, create=True)
    score_writer = r7.BudgetJsonl(budget, "stage_a_scores.jsonl")
    feedback_path = _private_ledger_path(output_dir, "feedback")
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with feedback_path.open("xb") as feedback_stream:
        feedback_stream.flush()
        os.fsync(feedback_stream.fileno())
    score_gate = r7.ScoreDisclosureGate(lambda row: score_writer.write(row, flush=True))
    feedback_digest = hashlib.sha256()
    counters: Counter[str] = Counter()
    query_groups: set[str] = set()
    query_set = set(map(int, query_positions))
    if len(query_set) != len(query_positions) or list(query_positions) != sorted(query_positions):
        raise ContinuationError(f"{category}/{condition}: query positions are malformed")

    # Initialize the B3 references at M0 before the first TEST event.
    b3_by_trajectory: dict[str, dict[tuple[str, str], B3State]] = {}
    calibration_by_trajectory: dict[str, dict[str, Sequence[Mapping[str, Any]]]] = {}
    initialization_costs: dict[str, Any] = {}
    initialization_scores: dict[str, dict[str, list[float]]] = {}
    for spec in trajectories:
        traj = spec["trajectory"]
        calibration = spec["calibration"]
        states, costs, score_vectors = _initialize_b3_states(
            traj, calibration, impl=impl, r1=r1, exp2a=exp2a,
            r9=r9, cfg=cfg, correction=bool(spec["correction"]))
        b3_by_trajectory[traj.name] = states
        calibration_by_trajectory[traj.name] = calibration
        initialization_costs[traj.name] = costs
        initialization_scores[traj.name] = score_vectors
    a2_spec = next(spec for spec in trajectories if spec["role"] == "A2")
    a2_states = b3_by_trajectory[a2_spec["trajectory"].name]
    theta0 = float(a2_states[("G", "C64")].theta)
    sigma0 = float(a2_states[("G", "C64")].sigma)
    theta0_c64 = float(a2_states[("G", "C64")].history[0]["theta"])
    if not math.isclose(theta0, theta0_c64, rel_tol=0, abs_tol=0):
        raise ContinuationError("A2 reference writer threshold changed before stream start")
    a2_initial_scores = initialization_scores[a2_spec["trajectory"].name]
    info_c64 = list(a2_spec["info"].get("c64_scores", ()))
    actual_c64 = a2_initial_scores.get("G:C64", [])
    if len(info_c64) != 64 or len(actual_c64) != 64 or any(
            not r3._score_tolerance(float(a), float(b))
            for a, b in zip(actual_c64, info_c64)):
        raise ContinuationError("A2 initial native C64 score vector fails the frozen A1/R9 parity receipt")
    initial_gc = a2_initial_scores.get("GC:C64", [])
    if len(initial_gc) != 64 or any(
            not r3._score_tolerance(float(a), float(b))
            for a, b in zip(actual_c64, initial_gc)):
        raise ContinuationError("empty native C1 changed the A2 initial C64 score vector")

    a3a_diagnostics = {
        "initial_normal_memory_parity": None,
        "first_divergence": None,
        "first_divergent_normal_write": None,
        "first_divergent_correction_update": None,
        "differing_normal_write_decisions": 0,
        "differing_normal_state_transitions": 0,
        "differing_admitted_or_replaced_prototypes": 0,
        "differing_correction_updates": 0,
        "different_G_scores": 0, "sum_abs_G_score_difference": 0.0,
        "max_abs_G_score_difference": 0.0,
    }
    a3a_c64 = next((s for s in trajectories if s["role"] == "A3a" and
                    s["calibration_name"] == "C64"), None)
    a3a_c128 = next((s for s in trajectories if s["role"] == "A3a" and
                     s["calibration_name"] == "C128"), None)
    if a3a_c64 and a3a_c128:
        t64, t128 = a3a_c64["trajectory"], a3a_c128["trajectory"]
        a3a_diagnostics["initial_normal_memory_parity"] = (
            t64.g_digest == t128.g_digest and t64.gc_digest == t128.gc_digest and
            t64.fixed_sha256 == t128.fixed_sha256)
        if not a3a_diagnostics["initial_normal_memory_parity"]:
            raise ContinuationError("A3a C64/C128 trajectories do not start in native state parity")

    reader_event_map = r9.make_order_events(
        trajectories[0]["info"]["events"], order)
    reader = r7.IndexedQueriedLabelReader(
        trajectories[0]["info"]["json_path"], category,
        trajectories[0]["info"]["contract"]["category_json_sha256"],
        reader_event_map, tuple(query_positions))
    query_group_digest = hashlib.sha256()
    state_change_counts: Counter[str] = Counter()
    rescore_seconds: Counter[str] = Counter()
    update_seconds = 0.0
    encoding_cache_receipt = {"source": "category-sharded float32 original-position mmap",
                              "encodings": 0}
    cmax_status = "NOT_REGISTERED"
    cmax_profile: dict[str, Any] | None = None
    cmax_snapshots: list[dict[str, Any]] = []
    cmax_full_enabled = any(
        "Cmax-G" in spec["calibration"] for spec in trajectories)
    if cmax_full_enabled:
        cmax_status = "PROFILE_PENDING"
    started = time.perf_counter()
    try:
        if cmax_items is not None and condition == "P0":
            # M0 is a fixed, outcome-blind snapshot. It is kept outside any
            # full-trajectory operating-point table.
            cmax_snapshots.append({"event_position": 0, "state": "M0",
                "source": "A2 shared trajectory", "full_B3_trajectory": False,
                "initial_B3": {f"{scorer}:{size}": {
                    "theta": state.theta, "sigma": state.sigma,
                    "score_sha256": initialization_costs[a2_spec["trajectory"].name][
                        f"{scorer}:{size}"]["score_sha256"]}
                    for (scorer, size), state in a2_states.items()
                    if size == "Cmax-G"}})
        for position, event in enumerate(events):
            causal = r1.CausalEventOrder(position)
            original_position = int(event.original_position)
            feature_np = np.asarray(test_features[original_position], dtype=np.float32)
            feature = torch.from_numpy(np.ascontiguousarray(feature_np)).to("cuda")
            if tuple(feature.shape) != (784, 4096) or not torch.isfinite(feature).all().item():
                raise ContinuationError(f"{category}/{condition}:{position}: cached TEST feature invalid")
            image = impl.load_rgb(str(event.image_path))
            feature_layers = [feature]
            scored_by_name: dict[str, dict[str, Any]] = {}
            score_map: dict[str, float] = {}
            decisions: dict[str, Any] = {}
            state_versions: dict[str, Any] = {}
            for spec in trajectories:
                traj: NativeTrajectory = spec["trajectory"]
                if spec.get("score_only", False):
                    scored = _native_event_scores(
                        traj, impl=impl, r1=r1, exp2a=exp2a, r9=r9, cfg=cfg,
                        image=image, feature_layers=feature_layers,
                        source_group_sha256=str(event.source_group_sha256),
                        position=position, use_correction=False)
                    scored_by_name[traj.name] = scored
                    score_map[f"{traj.name}:STATIC"] = float(scored["base_score"])
                    static_state = b3_by_trajectory[traj.name][("G", "C64")]
                    _record_static_c64_prediction(
                        decisions, traj, static_state, float(scored["base_score"]))
                    state_versions[traj.name] = {"G": {"version": 0,
                        "sha256": traj.g_digest}, "GC": None}
                    continue
                if spec.get("stopped", False):
                    continue
                scored = _native_event_scores(
                    traj, impl=impl, r1=r1, exp2a=exp2a, r9=r9, cfg=cfg,
                    image=image, feature_layers=feature_layers,
                    source_group_sha256=str(event.source_group_sha256),
                    position=position,
                    use_correction=bool(spec["correction"]))
                scored_by_name[traj.name] = scored
                score_map[f"{traj.name}:G"] = float(scored["base_score"])
                if spec["correction"]:
                    score_map[f"{traj.name}:GC"] = float(scored["gc_score"])
                state_versions[traj.name] = {
                    "G": {"version": traj.g_version, "sha256": traj.g_digest},
                    "GC": {"version": traj.gc_version, "sha256": traj.gc_digest}
                    if spec["correction"] else None}
            queried = position in query_set
            for spec in trajectories:
                traj = spec["trajectory"]
                if spec.get("score_only", False):
                    continue
                if traj.name not in scored_by_name:
                    continue
                if spec["role"] == "A2":
                    continue
                scored = scored_by_name[traj.name]
                states = b3_by_trajectory[traj.name]
                cal_name = str(spec["calibration_name"])
                g_ver, gc_ver = traj.g_version, traj.gc_version
                decisions[f"{traj.name}:G"] = _b3_prediction(
                    states[("G", cal_name)], score=float(scored["base_score"]),
                    state_digest=traj.g_digest, state_version=g_ver)
                if spec["correction"]:
                    decisions[f"{traj.name}:GC"] = _b3_prediction(
                        states[("GC", cal_name)], score=float(scored["gc_score"]),
                        state_digest=traj.gc_digest, state_version=gc_ver)
            if a2_spec["trajectory"].name in scored_by_name:
                a2_name = a2_spec["trajectory"].name
                a2_score = scored_by_name[a2_name]
                for size in ("C64", "C128"):
                    decisions[f"A2:G:{size}"] = _b3_prediction(
                        a2_states[("G", size)], score=float(a2_score["base_score"]),
                        state_digest=a2_spec["trajectory"].g_digest,
                        state_version=a2_spec["trajectory"].g_version)
                    decisions[f"A2:GC:{size}"] = _b3_prediction(
                        a2_states[("GC", size)], score=float(a2_score["gc_score"]),
                        state_digest=a2_spec["trajectory"].gc_digest,
                        state_version=a2_spec["trajectory"].gc_version)
                if cmax_status != "HOLD_RESOURCE_LIMIT" and ("G", "Cmax-G") in a2_states:
                    decisions["A2:G:Cmax-G"] = _b3_prediction(
                        a2_states[("G", "Cmax-G")], score=float(a2_score["base_score"]),
                        state_digest=a2_spec["trajectory"].g_digest,
                        state_version=a2_spec["trajectory"].g_version)
                    decisions["A2:GC:Cmax-G"] = _b3_prediction(
                        a2_states[("GC", "Cmax-G")], score=float(a2_score["gc_score"]),
                        state_digest=a2_spec["trajectory"].gc_digest,
                        state_version=a2_spec["trajectory"].gc_version)
            row = {"schema": "a_calibration_continuation.stage_a_score.v1",
                "category": category, "condition": condition, "order": order,
                "position": int(position), "original_json_position": original_position,
                "event_id_sha256": str(event.event_id_sha256),
                "image_identity_sha256": str(event.image_identity_sha256),
                "source_group_sha256": str(event.source_group_sha256),
                "camera_id": int(event.camera_id),
                "query_tape_decision": bool(queried),
                "query_identity_set_sha256": query_identity_sha256,
                "scores": score_map, "decisions": decisions,
                "score_state_versions": state_versions,
                "score_cache_key": original_position}
            causal.seal_scores()
            causal.decide_query(queried)
            score_gate.seal(position, row)
            counters["event_count"] += 1
            counters["test_feature_reuses"] += 1
            if cmax_items is not None and condition == "P0" and position > 0 and position % CHECKPOINT_INTERVAL == 0:
                if cmax_status == "HOLD_RESOURCE_LIMIT":
                    a2traj = a2_spec["trajectory"]
                    snapshot = {"event_position": position,
                                "state_version_G": a2traj.g_version,
                                "state_sha256_G": a2traj.g_digest,
                                "state_version_GC": a2traj.gc_version,
                                "state_sha256_GC": a2traj.gc_digest,
                                "full_B3_trajectory": False}
                    for scorer in ("G", "GC"):
                        values, seconds = _score_reference_items(
                            a2traj, cmax_items, scorer=scorer, impl=impl,
                            r1=r1, exp2a=exp2a, r9=r9, cfg=cfg,
                            event_position=position)
                        snapshot[scorer] = {"raw_q90_higher": higher_q90(values),
                            "sample_std_floor_002": max(float(np.std(values, ddof=1)),
                                                         SIGMA_FLOOR),
                            "score_count": len(values),
                            "score_sha256": _sha_bytes(_json_bytes(values)),
                            "rescore_seconds": seconds}
                    cmax_snapshots.append(snapshot)
            if queried:
                label = causal.disclose(lambda: reader.label_for(
                    position, sealed_and_fsynced=score_gate.is_sealed(position)))
                group_sha = str(event.source_group_sha256)
                if group_sha in query_groups:
                    raise ContinuationError("Q-A-TS tape queried one Source Group more than once")
                query_groups.add(group_sha)
                query_group_digest.update(group_sha.encode("ascii") + b"\n")
                counters["queried_label_count"] += 1
                counters["queried_normal"] += int(label == 0)
                counters["queried_defect"] += int(label == 1)
                feedback_row = {"schema": "a_calibration_continuation.private_feedback.v1",
                    "category": category, "condition": condition,
                    "position": int(position),
                    "original_json_position": original_position,
                    "event_id_sha256": str(event.event_id_sha256),
                    "source_group_sha256": group_sha, "label": int(label),
                    "score_row_sealed_before_disclosure": score_gate.is_sealed(position)}
                update_started = time.perf_counter()
                transitions: dict[str, Any] = {}
                writer_actions: dict[str, Any] = {}
                correction_updates: dict[str, Any] = {}
                a3a_write_comparison: dict[str, Any] = {}
                for spec in trajectories:
                    if spec.get("score_only", False):
                        continue
                    if spec.get("stopped", False):
                        continue
                    traj: NativeTrajectory = spec["trajectory"]
                    scored = scored_by_name.get(traj.name)
                    if scored is None:
                        continue
                    states = b3_by_trajectory[traj.name]
                    if spec["role"] == "A2":
                        theta, sigma = theta0, sigma0
                    else:
                        calibration_name = str(spec["calibration_name"])
                        writer_state = states[("G", calibration_name)]
                        theta, sigma = writer_state.theta, writer_state.sigma
                    prediction = int(float(scored["base_score"]) >= float(theta))
                    old_g_digest, old_gc_digest = traj.g_digest, traj.gc_digest
                    corr_result = None
                    if spec["correction"]:
                        outcome = {(0, 0): "TN", (0, 1): "FP",
                                   (1, 0): "FN", (1, 1): "TP"}[(int(label), prediction)]
                        corr_result = _update_native_correction(
                            traj, position=position, feature_layers=feature_layers,
                            top_ids=scored["top_ids"], label=int(label),
                            base_score=float(scored["base_score"]), theta=float(theta),
                            outcome=outcome, source_group_sha256=group_sha,
                            impl=impl, r1=r1, exp2a=exp2a)
                        correction_updates[traj.name] = corr_result
                        counters[f"{traj.name}:C1_updates"] += 1
                        counters[f"{traj.name}:C1_state_changes"] += int(corr_result["state_changed"])
                    write_result = _trajectory_writer_update(
                        traj, position=position, feature_layers=feature_layers,
                        top_ids=scored["top_ids"], selected_features=scored["selected"],
                        score=float(scored["base_score"]), theta=float(theta),
                        sigma=float(sigma), queried=True, label=int(label),
                        prediction=prediction, impl=impl, r1=r1)
                    writer_actions[traj.name] = {**write_result,
                        "theta_input": float(theta), "sigma_input": float(sigma),
                        "prediction_input": prediction,
                        "current_threshold_consumed": spec["role"] != "A2",
                        "current_sigma_consumed": spec["role"] != "A2",
                        "writer_score_source": "common_C0_native_score",
                        "candidate_patch_count": int(len(scored["top_ids"]))}
                    counters[f"{traj.name}:Greedy_state_changes"] += int(write_result["state_changed"])
                    greedy_sha = r1._memory_content_sha256(traj.bank)
                    new_g_digest = r9_scorer_digest(traj.fixed_sha256, greedy_sha)
                    new_gc_digest = (r9_state_digest(traj.bank, traj.fixed_sha256,
                        traj.source_groups, traj.correction_cfg, traj.correction_cb, r1)
                        if spec["correction"] else new_g_digest)
                    g_changed = new_g_digest != old_g_digest
                    gc_changed = (new_gc_digest != old_gc_digest) if spec["correction"] else False
                    if write_result["state_changed"] != g_changed:
                        raise ContinuationError("Greedy writer audit and native memory digest disagree")
                    if spec["correction"] and g_changed and not gc_changed:
                        raise ContinuationError("GC state digest omitted a shared Greedy state change")
                    if g_changed:
                        traj.g_version += 1
                    if gc_changed:
                        traj.gc_version += 1
                    traj.g_digest, traj.gc_digest = new_g_digest, new_gc_digest
                    state_change_counts[f"{traj.name}:G"] += int(g_changed)
                    state_change_counts[f"{traj.name}:GC"] += int(gc_changed)
                    refresh_calibration = calibration_by_trajectory[traj.name]
                    if cmax_status == "HOLD_RESOURCE_LIMIT" and spec["role"] in {"A2", "A3a"}:
                        refresh_calibration = {k: v for k, v in refresh_calibration.items()
                                               if k != "Cmax-G"}
                    refresh_rows, refresh_time = _refresh_b3_states(
                        traj, states, refresh_calibration,
                        g_changed=g_changed, gc_changed=gc_changed,
                        impl=impl, r1=r1, exp2a=exp2a, r9=r9, cfg=cfg,
                        event_position=position + 1)
                    for refresh in refresh_rows:
                        rescore_seconds[f"{traj.name}:{refresh['scorer']}:{refresh['calibration_size']}"] += float(
                            refresh["rescore_seconds"])
                    transitions[traj.name] = {
                        "greedy_writer": writer_actions[traj.name],
                        "correction_update": corr_result,
                        "g_changed": g_changed, "gc_changed": gc_changed,
                        "state_versions_after": {"G": traj.g_version, "GC": traj.gc_version},
                        "state_sha256_after": {"G": traj.g_digest, "GC": traj.gc_digest},
                        "b3_refreshes": refresh_rows}
                if a3a_c64 and a3a_c128:
                    s64, s128 = a3a_c64["trajectory"], a3a_c128["trajectory"]
                    score64 = float(scored_by_name[s64.name]["base_score"])
                    score128 = float(scored_by_name[s128.name]["base_score"])
                    delta = abs(score64 - score128)
                    a3a_diagnostics["different_G_scores"] += int(delta > 0)
                    a3a_diagnostics["sum_abs_G_score_difference"] += delta
                    a3a_diagnostics["max_abs_G_score_difference"] = max(
                        a3a_diagnostics["max_abs_G_score_difference"], delta)
                    a64 = writer_actions.get(s64.name, {}).get("action", {}).get("action")
                    a128 = writer_actions.get(s128.name, {}).get("action", {}).get("action")
                    if a64 != a128:
                        a3a_diagnostics["differing_normal_write_decisions"] += 1
                        if a3a_diagnostics["first_divergent_normal_write"] is None:
                            a3a_diagnostics["first_divergent_normal_write"] = {
                                "position": position, "cause": "native writer action differed",
                                "C64_action": a64, "C128_action": a128}
                    slots64 = writer_actions.get(s64.name, {}).get("prototype_slot_changes", [])
                    slots128 = writer_actions.get(s128.name, {}).get("prototype_slot_changes", [])
                    changed64 = sum(int(row.get("admitted_or_replaced_count", 0)) for row in slots64)
                    changed128 = sum(int(row.get("admitted_or_replaced_count", 0)) for row in slots128)
                    a3a_diagnostics["differing_admitted_or_replaced_prototypes"] += abs(
                        changed64 - changed128)
                    tr64, tr128 = transitions.get(s64.name), transitions.get(s128.name)
                    if tr64 and tr128:
                        if tr64["g_changed"] != tr128["g_changed"]:
                            a3a_diagnostics["differing_normal_state_transitions"] += 1
                        c64_update = correction_updates.get(s64.name)
                        c128_update = correction_updates.get(s128.name)
                        if c64_update and c128_update and (
                                c64_update["state_changed"] != c128_update["state_changed"] or
                                c64_update["integer_stats"] != c128_update["integer_stats"]):
                            a3a_diagnostics["differing_correction_updates"] += 1
                            if a3a_diagnostics["first_divergent_correction_update"] is None:
                                a3a_diagnostics["first_divergent_correction_update"] = {
                                    "position": position, "C64": c64_update,
                                    "C128": c128_update}
                    if a3a_diagnostics["first_divergence"] is None:
                        d64 = {k: s64.g_digest if k == "G" else s64.gc_digest for k in ("G", "GC")}
                        d128 = {k: s128.g_digest if k == "G" else s128.gc_digest for k in ("G", "GC")}
                        if d64 != d128:
                            a3a_diagnostics["first_divergence"] = {
                                "position": position,
                                "cause": "calibration-policy-specific native update produced a different state",
                                "C64_state_sha256": d64,
                                "C128_state_sha256": d128}
                feedback_row.update({"writer_actions": writer_actions,
                    "correction_updates": correction_updates,
                    "state_transitions": transitions,
                    "state_versions_after": {spec["trajectory"].name: {
                        "G": spec["trajectory"].g_version,
                        "GC": spec["trajectory"].gc_version}
                        for spec in trajectories}})
                _append_bounded_private_jsonl(
                    feedback_path, feedback_row,
                    maximum_bytes=128 * 1024**2)
                feedback_digest.update(r7._canonical_bytes(feedback_row))
                update_seconds += time.perf_counter() - update_started
                causal.corrections_updated()
                causal.memory_updated()
                causal.finish()
                # One genuine online update supplies the Cmax resource profile.
                if (cmax_status == "PROFILE_PENDING" and
                        any(v["greedy_writer"]["state_changed"]
                            for v in transitions.values())):
                    profile_costs: dict[str, float] = {}
                    for profile_spec in trajectories:
                        if not (profile_spec["role"] == "A2" or
                                (profile_spec["role"] == "A3a" and
                                 profile_spec["calibration_name"] == "Cmax-G")):
                            continue
                        profile_traj = profile_spec["trajectory"]
                        for scorer in ("G", "GC"):
                            _scores, seconds = _score_reference_items(
                                profile_traj, cmax_items, scorer=scorer,
                                impl=impl, r1=r1, exp2a=exp2a, r9=r9,
                                cfg=cfg, event_position=position + 1)
                            profile_costs[f"{profile_traj.name}:{scorer}"] = seconds
                    cmax_rescore = sum(profile_costs.values())
                    remaining_queries = sum(q > position for q in query_set)
                    estimated = cmax_rescore * max(1, remaining_queries)
                    elapsed_category = time.perf_counter() - category_started
                    remaining_deadline = max(0.0, category_deadline - elapsed_category)
                    condition_elapsed = time.perf_counter() - started
                    remaining_events = max(0, len(events) - position - 1)
                    projected_non_cmax = (condition_elapsed / max(1, position + 1)) * remaining_events
                    projected_total = estimated + projected_non_cmax
                    cmax_profile = {"profile_event_position": position,
                        "profile_after_genuine_native_write": True,
                        "measured_full_reference_rescore_seconds": cmax_rescore,
                        "measured_trajectory_rescore_seconds": profile_costs,
                        "remaining_query_upper_bound": remaining_queries,
                        "conservative_full_cmax_remaining_seconds": estimated,
                        "projected_non_cmax_remaining_seconds": projected_non_cmax,
                        "projected_total_remaining_seconds": projected_total,
                        "category_deadline_remaining_seconds": remaining_deadline,
                        "estimated_within_category_deadline": projected_total <= remaining_deadline,
                        "feature_bytes_per_Cmax_G": int(len(cmax_items) * 784 * 4096 * 4)
                            if cmax_items is not None else None,
                        "candidate_count": len(cmax_items) if cmax_items is not None else None,
                        "native_scorer_used": True}
                    if projected_total > remaining_deadline:
                        cmax_status = "HOLD_RESOURCE_LIMIT"
                        for spec in trajectories:
                            if spec.get("role") == "A3a" and spec.get("calibration_name") == "Cmax-G":
                                spec["stopped"] = True
                    else:
                        cmax_status = "FULL_TRAJECTORY_ENABLED"
            else:
                # Unqueried events can never trigger a supervised update.
                for spec in trajectories:
                    if spec.get("score_only", False):
                        continue
                    if spec.get("stopped", False):
                        continue
                    traj = spec["trajectory"]
                    scored = scored_by_name[traj.name]
                    if spec["role"] == "A2":
                        theta, sigma = theta0, sigma0
                    else:
                        state = b3_by_trajectory[traj.name][("G", spec["calibration_name"])]
                        theta, sigma = state.theta, state.sigma
                    action = _trajectory_writer_update(
                        traj, position=position, feature_layers=feature_layers,
                        top_ids=scored["top_ids"], selected_features=scored["selected"],
                        score=float(scored["base_score"]), theta=float(theta),
                        sigma=float(sigma), queried=False, label=None,
                        prediction=int(scored["base_score"] >= theta), impl=impl, r1=r1)
                    if action["state_changed"]:
                        raise ContinuationError("unqueried TEST event changed native Greedy memory")
                causal.corrections_updated()
                causal.memory_updated()
                causal.finish()
            if (position + 1) % 256 == 0:
                _append_bounded_private_jsonl(
                    _private_ledger_path(output_dir, "progress"), {
                    "category": category, "condition": condition,
                    "sealed_through_position": position,
                    "queried_feedback_count": int(counters["queried_label_count"]),
                    "state_versions": {spec["trajectory"].name: {
                        "G": spec["trajectory"].g_version,
                        "GC": spec["trajectory"].gc_version} for spec in trajectories},
                    "automatic_restart": False}, maximum_bytes=16 * 1024**2)
        score_writer.flush()
        reader.finish()
        if (len(score_gate.sealed_positions) != len(events) or
                int(counters["queried_label_count"]) != len(query_positions) or
                len(query_groups) != len(query_positions)):
            raise ContinuationError("Stage A did not seal the exact event/query allowlists")
        if cmax_status == "PROFILE_PENDING":
            cmax_status = "PROFILE_NOT_OBTAINED_IMPLEMENTATION_PRESENT"
        seal = {"schema": "a_calibration_continuation.stage_a_seal.v1",
            "status": "INCOMPLETE" if smoke else "SEALED_COMPLETE",
            "classification": "SUPERVISED_SMOKE_NOT_SCIENTIFIC_RESULT" if smoke else "REGISTERED_STAGE_A",
            "category": category,
            "condition": condition, "order": order, "support_seed": support_seed,
            "event_count": len(events), "query_count": len(query_positions),
            "query_positions": list(query_positions),
            "query_identity_set_sha256": query_identity_sha256,
            "query_group_set_sha256": query_group_digest.hexdigest(),
            "score_rows_sha256": r7._sha256_file(output_dir / "stage_a_scores.jsonl"),
            "feedback_rows_sha256": r7._sha256_file(feedback_path),
            "feedback_ledger_sha256": feedback_digest.hexdigest(),
            "all_scores_sealed_before_feedback": True,
            "unqueried_labels_accessed": False,
            "test_masks_opened": False,
            "event_state_change_counts": dict(state_change_counts),
            "initial_b3": initialization_costs,
            "final_b3_history": {name: {f"{scorer}:{size}": state.history
                for (scorer, size), state in values.items()}
                for name, values in b3_by_trajectory.items()},
            "rescore_seconds": dict(rescore_seconds),
            "update_and_rescore_seconds": update_seconds,
            "child_condition_wall_seconds": time.perf_counter() - started,
            "a3a_divergence": a3a_diagnostics,
            "cmax_G_status": cmax_status,
            "cmax_G_resource_profile": cmax_profile,
            "cmax_G_snapshot_diagnostics": cmax_snapshots,
            "greedy_capacity": K_GREEDY,
            "correction_capacity": K_CORRECTION,
            "native_correction_multiplier": CORRECTION_STRENGTH,
            "native_writer": "r1.common_c0_greedy_writer_action + MemoryArm",
            "native_score": "r1._native_dense_score and native C1-Full codebook",
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        budget.write_json("stage_a_seal.json", seal)
        seal["seal_sha256"] = r7._sha256_file(output_dir / "stage_a_seal.json")
        return seal
    except BaseException as exc:
        try:
            budget.write_json("stage_a_failure.json", {
                "schema": "a_calibration_continuation.failure.v1",
                "category": category, "condition": condition,
                "stage": "STAGE_A", "error_type": type(exc).__name__,
                "error": str(exc), "automatic_restart": False})
        except BaseException:
            pass
        raise
    finally:
        reader.close()


def _stage_a_category_child(run_id: str, category: str, *, smoke: bool = False,
                            smoke_events: int = 64) -> dict[str, Any]:
    """Supervised per-category A2/A3 Stage A execution using native R9 functions."""
    import torch
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import stage05
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r2_exp2a_c1_strength as exp2a
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration import r6_capacity1024_audiojack_runner as r6
    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration import r9_correction_mechanism_falsification as r9
    from experiments.calibration import profiles
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job

    if category not in PRIMARY_CATEGORIES:
        raise ContinuationError("category is outside the four authorized Experiment A classes")
    if not current_process_is_in_job():
        raise ContinuationError("category child is outside its Windows Job Object")
    run_root = _run_path(run_id).resolve(strict=True)
    output_root = (run_root / ("smoke" if smoke else "stage_a") / category)
    output_root.mkdir(parents=True, exist_ok=True)
    prior_category_complete = output_root / "category_complete.json"
    if prior_category_complete.is_file():
        prior = _read_json(prior_category_complete)
        if ((smoke and prior.get("status") == "INCOMPLETE") or
                (not smoke and prior.get("status") == "SEALED_COMPLETE")):
            return prior
        _archive_partial(prior_category_complete,
                         attempts_dir=output_root / "attempts")
    category_started = time.perf_counter()
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job as _in_job
    if not _in_job():
        raise ContinuationError("Windows Job Object membership was lost before model setup")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ContinuationError("category child requires the authorized single CUDA GPU")
    torch.set_num_threads(1)
    inherited_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    impl = p0.require_sources()
    model_receipt = r3._get_verified_model_receipt()
    restore_hub, runtime = r3._runtime_and_native_setup(torch, impl, model_receipt)
    cfg, scorer_profile = r3._native_config(p0, profiles, r1)
    backbone = None
    scratch_dir = run_root / ".scratch" / (category + ("_smoke" if smoke else ""))
    feature_cache = None
    try:
        try:
            backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
        finally:
            restore_hub()
        loaded_sha = r1._model_state_sha256(backbone.model)
        if loaded_sha != model_receipt.get("loaded_model_state_sha256"):
            raise ContinuationError("loaded DINOv2 state differs from the verified R9 model receipt")
        runtime["loaded_model_state_sha256"] = loaded_sha
        train_rows, _test_rows, category_json_sha = stage05.read_class_stream(
            a1.DATASET_ROOT, a1.JSON_ROOT, category, test_labels=False)
        contract = r7._frozen_contract()
        info0 = contract["metadata"][category]
        r7.validate_image_projection(Path(info0["json_path"]), category, info0["events"])
        a1_run = RUNS_ROOT / f"a_calibration_scale_{A1_RUN_ID}"
        role_manifest = _read_json(a1_run / "a1" / category / "role_manifest_private.json")
        a1_seal = _read_json(a1_run / "a1" / category / "stage_a_seal.json")
        a1_result = _read_json(a1_run / "a1" / category / "stage_a_result.json")
        if (category_json_sha != a1_seal.get("input_digests", {}).get("category_json") or
                len(info0["events"]) != int(a1_result["test_count"])):
            raise ContinuationError(f"{category}: current data metadata differs from sealed A1 inputs")
        train_by_group, _by_opaque = _training_group_views(category, train_rows)
        support_info1 = r9.support1_class_info(category, info0)
        if list(info0["c64_groups"]) != list(support_info1["c64_groups"]):
            raise ContinuationError(f"{category}: C64 identities differ across Support seeds")
        support_paths0 = [str(path) for path in info0["contract"]["support_paths"]]
        support_paths1 = [str(path) for path in support_info1["support_paths"]]
        f4_groups = list(role_manifest["A3b_reference_support_groups"])
        f4_paths = [str(train_by_group[str(group)].image_path) for group in f4_groups]
        all_ref_paths = list(dict.fromkeys(support_paths0 + support_paths1 + f4_paths))
        support_cache_started = time.perf_counter()
        support_cache = p0.encode_reference_features(
            impl, backbone, all_ref_paths, cfg)
        support_cache_seconds = time.perf_counter() - support_cache_started
        if set(support_cache) != set(all_ref_paths):
            raise ContinuationError("native support augmentation feature cache is incomplete")
        parity_row = train_by_group[str(f4_groups[0])]
        native_parity = _verify_native_greedy_parity(
            impl=impl, p0=p0, r1=r1, r3=r3, support_cache=support_cache,
            support_paths=support_paths0, cfg=cfg,
            expected_fixed_sha256=str(info0["fixed_support_sha256_expected"]),
            parity_row=parity_row, backbone=backbone, category=category)
        if not smoke:
            calibration_sets = _calibration_items(
                category, 0, role_manifest, train_rows, impl, backbone,
                cfg, p0, r7, r9)
            a3b_f4, a3b_fexp, a3b_receipt = _a3b_reference_trajectories(
                category, impl=impl, p0=p0, r1=r1, r3=r3, r6=r6,
                r9=r9, backbone=backbone, cfg=cfg, train_rows=train_rows,
                role_manifest=role_manifest, support_cache=support_cache)
        else:
            # Smoke validates P0 A2/A3a event order and update semantics. It is
            # explicitly incomplete and carries no Cmax/A3b empirical output.
            calibration_sets = _calibration_items(
                category, 0, role_manifest, train_rows, impl, backbone,
                cfg, p0, r7, r9)
            a3b_f4 = a3b_fexp = None
            a3b_receipt = {"status": "NOT_RUN_IN_SMOKE"}
        scratch_dir.parent.mkdir(parents=True, exist_ok=True)
        cache_events = (r9.make_order_events(info0["events"], "SHUFFLE0")[:int(smoke_events)]
                        if smoke else info0["events"])
        feature_cache, feature_receipt = _encode_test_feature_cache(
            category, cache_events, impl=impl, backbone=backbone,
            cfg=cfg, p0=p0, scratch_dir=scratch_dir)
        run_manifest = _read_json(run_root / "run_manifest.json")
        r9_manifest = _read_json(r9.RUN_NAMESPACE / "manifest" / "run_manifest.json")
        category_seals: dict[str, Any] = {}
        condition_plan = (("P0", "SHUFFLE0", 0),) if smoke else (
            ("P0", "SHUFFLE0", 0), ("P1", "SHUFFLE11", 0),
            ("P2", "SUPPORT1_SHUFFLE0", 1))
        for condition, order, support_seed in condition_plan:
            condition_output = output_root / condition
            if _condition_seal_is_valid(condition_output, allow_smoke=smoke):
                category_seals[condition] = _read_json(
                    condition_output / "stage_a_seal.json")
                continue
            if condition_output.exists():
                _archive_condition_ledgers(
                    condition_output, attempts_dir=output_root / "attempts")
                _archive_partial(condition_output,
                                 attempts_dir=output_root / "attempts")
            condition_info = info0 if support_seed == 0 else support_info1
            events = r9.make_order_events(info0["events"], order)
            order_receipt = r9_manifest["identity"]["classes"][category]["orders"][order]
            query_positions = [int(value) for value in order_receipt["query_positions"]]
            if smoke:
                events = events[:int(smoke_events)]
                query_positions = [value for value in query_positions if value < len(events)]
            a2_traj = _build_native_trajectory(
                f"{condition}-A2-shared", impl=impl, p0=p0, r1=r1,
                r3=r3, r6=r6, support_cache=support_cache,
                support_paths=support_paths0 if support_seed == 0 else support_paths1,
                cfg=cfg, expected_fixed_sha256=str(
                    condition_info["fixed_support_sha256_expected"]))
            a2_calibration = {"C64": calibration_sets["C64"],
                              "C128": calibration_sets["C128"]}
            if not smoke and condition == "P0":
                a2_calibration["Cmax-G"] = calibration_sets["Cmax-G"]
            specs = [{"trajectory": a2_traj, "role": "A2", "correction": True,
                      "calibration_name": "C64", "calibration": a2_calibration,
                      "info": condition_info}]
            cmax_for_condition = None
            if condition == "P0":
                for size in ("C64", "C128"):
                    traj = _build_native_trajectory(
                        f"P0-A3a-{size}", impl=impl, p0=p0, r1=r1,
                        r3=r3, r6=r6, support_cache=support_cache,
                        support_paths=support_paths0, cfg=cfg,
                        expected_fixed_sha256=str(info0["fixed_support_sha256_expected"]))
                    specs.append({"trajectory": traj, "role": "A3a",
                        "correction": True, "calibration_name": size,
                        "calibration": {size: calibration_sets[size]},
                        "info": info0})
                if not smoke:
                    traj = _build_native_trajectory(
                        "P0-A3a-Cmax-G", impl=impl, p0=p0, r1=r1,
                        r3=r3, r6=r6, support_cache=support_cache,
                        support_paths=support_paths0, cfg=cfg,
                        expected_fixed_sha256=str(info0["fixed_support_sha256_expected"]))
                    specs.append({"trajectory": traj, "role": "A3a",
                        "correction": True, "calibration_name": "Cmax-G",
                        "calibration": {"Cmax-G": calibration_sets["Cmax-G"]},
                        "info": info0})
                    specs.extend([
                        {"trajectory": a3b_f4, "role": "A3b", "correction": False,
                         "calibration_name": "C64", "calibration":
                            {"C64": calibration_sets["C64"]}, "info": info0},
                        {"trajectory": a3b_fexp, "role": "A3b", "correction": False,
                         "calibration_name": "C64", "calibration":
                            {"C64": calibration_sets["C64"]}, "info": info0},
                        {"trajectory": _static_trajectory_clone(a3b_f4,
                            "A3b-F4-STATIC"), "role": "A3b_STATIC",
                         "correction": False, "score_only": True,
                         "calibration_name": "STATIC",
                         "calibration": {"C64": calibration_sets["C64"]}, "info": info0},
                        {"trajectory": _static_trajectory_clone(a3b_fexp,
                            "A3b-FEXP-KC-STATIC"), "role": "A3b_STATIC",
                         "correction": False, "score_only": True,
                         "calibration_name": "STATIC",
                         "calibration": {"C64": calibration_sets["C64"]}, "info": info0},
                    ])
                    cmax_for_condition = calibration_sets["Cmax-G"]
            seal = _run_condition(category, condition, order, support_seed,
                events, query_positions, specs, feature_cache,
                impl=impl, r1=r1, exp2a=exp2a, r3=r3, r7=r7, r9=r9,
                cfg=cfg, output_dir=condition_output,
                category_started=category_started,
                category_deadline=CATEGORY_DEADLINE_SECONDS,
                role_manifest=role_manifest,
                query_identity_sha256=r9_manifest["identity"]["classes"][category][
                    "r7_query_identity_set_sha256"],
                cmax_items=cmax_for_condition, smoke=smoke)
            if not smoke and seal["status"] != "SEALED_COMPLETE":
                raise ContinuationError(f"{category}/{condition}: Stage A did not seal complete")
            category_seals[condition] = seal
            del specs, a2_traj
            torch.cuda.empty_cache()
        category_complete = {"schema": "a_calibration_continuation.category_complete.v1",
            "status": "INCOMPLETE" if smoke else "SEALED_COMPLETE",
            "category": category, "conditions": category_seals,
            "native_score_update_parity": native_parity,
            "native_runtime": runtime,
            "native_scorer_profile": asdict(scorer_profile),
            "native_scorer_profile_sha256": scorer_profile.sha256,
            "support_cache_seconds": support_cache_seconds,
            "test_feature_cache": feature_receipt,
            "A3b_reference_construction": a3b_receipt,
            "test_images_encoded_once": True, "test_labels_accessed": False,
            "test_masks_accessed": False,
            "child_wall_seconds": time.perf_counter() - category_started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        _write_once(output_root / "category_complete.json",
                    _json_bytes(category_complete) + b"\n")
        return category_complete
    finally:
        if feature_cache is not None:
            feature_cache.close()
            del feature_cache
        if backbone is not None:
            del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _remove_category_scratch(run_root, category, smoke=smoke)


def _nonoverlap_window_metrics(labels: Sequence[int], predictions: Sequence[int],
                               window: int = 512) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(predictions, dtype=np.int8)
    if y.shape != p.shape or y.ndim != 1:
        raise ContinuationError("nonoverlapping window metric inputs are misaligned")
    rows = []
    for start in range(0, max(0, len(y) - window + 1), window):
        stop = start + window
        yw, pw = y[start:stop], p[start:stop]
        normals, defects = int(np.sum(yw == 0)), int(np.sum(yw == 1))
        if normals == 0 or defects == 0:
            rows.append({"start": start, "end_exclusive": stop,
                         "status": "INVALID_SINGLE_CLASS_WINDOW"})
            continue
        fpr = float(np.mean(pw[yw == 0] == 1))
        fnr = float(np.mean(pw[yw == 1] == 0))
        rows.append({"start": start, "end_exclusive": stop,
            "status": "VALID", "normal_count": normals,
            "defect_count": defects, "fpr": fpr, "fnr": fnr,
            "ber": (fpr + fnr) / 2.0, "balanced_accuracy": 1.0 - (fpr + fnr) / 2.0})
    valid = [row for row in rows if row["status"] == "VALID"]
    worst = max(valid, key=lambda row: row["ber"]) if valid else None
    return {"window_events": int(window), "stride": int(window),
        "complete_windows": len(rows), "valid_windows": len(valid),
        "windows": rows, "worst_valid_window": worst}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.endswith("\n"):
                raise ContinuationError(f"{path.name}:{line_no}: unsealed JSONL row")
            row = json.loads(line)
            if not isinstance(row, dict) or "label" in row:
                raise ContinuationError("Stage A score tape contains a label or malformed row")
            rows.append(row)
    return rows


def r9_rank_metrics(labels: Any, scores: Any) -> dict[str, Any]:
    from experiments.calibration.r9_correction_mechanism_falsification import _ranking_metrics
    result = _ranking_metrics(labels, scores)
    return {"auroc": result.get("auroc"),
            "average_precision": result.get("average_precision")}


def r9_classification_metrics(labels: Any, predictions: Any) -> dict[str, Any]:
    from experiments.calibration.r9_correction_mechanism_falsification import _metric_binary
    result = _metric_binary(labels, predictions)
    fpr, fnr = result.get("fpr"), result.get("fnr")
    result["ber"] = ((float(fpr) + float(fnr)) / 2.0
                     if fpr is not None and fnr is not None else None)
    result["balanced_accuracy"] = (1.0 - result["ber"]
                                    if result["ber"] is not None else None)
    return result


def _b3_stability_metrics(seal: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for trajectory, states in seal.get("final_b3_history", {}).items():
        output[trajectory] = {}
        for key, rows in states.items():
            thresholds = np.asarray([row["theta"] for row in rows], dtype=np.float64)
            scales = np.asarray([row["sigma"] for row in rows], dtype=np.float64)
            if not len(thresholds) or not np.isfinite(thresholds).all() or not np.isfinite(scales).all():
                raise ContinuationError("sealed B3 history is empty or nonfinite")
            output[trajectory][key] = {
                "initial_threshold": float(thresholds[0]),
                "final_threshold": float(thresholds[-1]),
                "minimum_threshold": float(thresholds.min()),
                "maximum_threshold": float(thresholds.max()),
                "mean_threshold": float(thresholds.mean()),
                "threshold_total_variation": float(np.abs(np.diff(thresholds)).sum()),
                "effective_threshold_changes": int(np.count_nonzero(np.diff(thresholds))),
                "threshold_update_count": max(0, len(rows) - 1),
                "initial_sigma": float(scales[0]),
                "final_sigma": float(scales[-1]),
                "sigma_total_variation": float(np.abs(np.diff(scales)).sum()),
                "state_version_count": len({int(row["version"]) for row in rows}),
                "history": rows,
            }
    return output


def _public_a3b_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Keep group and candidate identities in private receipts only."""
    allowed = {
        "schema", "reference_pool_groups", "candidate_count",
        "candidate_feature_extraction_seconds", "candidate_selection_seconds",
        "native_reference_bank_build_seconds", "candidate_selection",
        "kcenter", "prototype_count", "prototype_dtype", "prototype_dim",
        "prototype_bytes", "kcenter_seconds", "F4_fixed_bank",
        "FEXP_fixed_bank", "feature_extraction_images",
        "feature_extraction_views", "native_scorer",
        "native_score_read_compatibility", "native_score_compatibility_smoke_seconds",
        "native_compatibility_smoke_extra_train_encodings", "test_outcomes_accessed",
    }
    result = {key: receipt[key] for key in allowed if key in receipt}
    for field_name in ("F4_fixed_bank", "FEXP_fixed_bank"):
        profile = receipt.get(field_name)
        if isinstance(profile, Mapping):
            result[field_name] = {
                "total_prototypes": profile.get("total_prototypes"),
                "total_feature_bytes": profile.get("total_feature_bytes"),
                "layers": [{key: layer.get(key) for key in (
                    "layer", "n_prototypes", "feature_dim", "capacity", "dtype", "bytes")}
                    for layer in profile.get("layers", [])],
            }
    compatibility = receipt.get("native_score_read_compatibility", {})
    result["native_score_read_compatibility"] = {
        name: {"passed": bool(value.get("passed")),
               "absolute_difference": value.get("absolute_difference"),
               "tolerance": value.get("tolerance")}
        for name, value in compatibility.items()}
    result["identities_redacted"] = True
    return result


def _stage_b_child(run_id: str) -> dict[str, Any]:
    """Outcome-isolated evaluator; refuses to read labels until every seal passes."""
    from experiments.calibration import stage05
    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job
    if not current_process_is_in_job():
        raise ContinuationError("Stage B must execute inside its Windows Job Object")
    run_root = _run_path(run_id).resolve(strict=True)
    run_state = _read_json(run_root / "RUN_STATE.json")
    if run_state.get("status") != "STAGE_A_COMPLETE":
        raise ContinuationError("Stage B blocked until all registered Stage A seals are complete")
    sealed_inputs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    seal_hashes: dict[str, dict[str, str]] = {}
    # Validate every full condition and score checksum before opening any labels.
    for category in PRIMARY_CATEGORIES:
        sealed_inputs[category] = {}
        seal_hashes[category] = {}
        for condition in ("P0", "P1", "P2"):
            leaf = run_root / "stage_a" / category / condition
            seal_path = leaf / "stage_a_seal.json"
            seal = _read_json(seal_path)
            if (seal.get("status") != "SEALED_COMPLETE" or
                    seal.get("category") != category or seal.get("condition") != condition or
                    seal.get("all_scores_sealed_before_feedback") is not True or
                    seal.get("unqueried_labels_accessed") is not False or
                    seal.get("test_masks_opened") is not False or
                    _sha_file(leaf / "stage_a_scores.jsonl") != seal.get("score_rows_sha256") or
                    _sha_file(_private_ledger_path(leaf, "feedback")) !=
                    seal.get("feedback_rows_sha256")):
                raise ContinuationError(f"Stage B refused invalid Stage A seal: {category}/{condition}")
            rows = _load_jsonl(leaf / "stage_a_scores.jsonl")
            if len(rows) != int(seal.get("event_count", -1)) or any(
                    int(row.get("position", -1)) != index for index, row in enumerate(rows)):
                raise ContinuationError(f"Stage B score tape is incomplete: {category}/{condition}")
            sealed_inputs[category][condition] = rows
            seal_hashes[category][condition] = _sha_file(seal_path)
    # Stage B outcome access begins here, after all twelve full seals validated.
    category_metrics: dict[str, Any] = {}
    resource_diagnostics: dict[str, Any] = {}
    for category in PRIMARY_CATEGORIES:
        labels_json_order = stage05.test_labels_for_class(a1.JSON_ROOT, category)
        category_metrics[category] = {}
        category_complete = _read_json(
            run_root / "stage_a" / category / "category_complete.json")
        test_cache = category_complete.get("test_feature_cache", {})
        parity = category_complete.get("native_score_update_parity", {})
        native_score_parity = parity.get("native_score_parity", {})
        native_update_parity = parity.get("native_update_parity", {})
        resource_diagnostics[category] = {
            "test_feature_cache": {key: test_cache.get(key) for key in
                ("count", "shape", "dtype", "bytes", "encoding_seconds", "encodings")},
            "support_cache_seconds": category_complete.get("support_cache_seconds"),
            "native_score_update_parity": {
                "status": parity.get("status"),
                "native_score": {key: native_score_parity.get(key) for key in
                    ("abs_difference", "max_allowed_abs", "relative_tolerance", "passed")},
                "native_update": {key: native_update_parity.get(key) for key in
                    ("keep_selected", "equal_each_layer", "passed", "comparison")},
            },
            "a3b_reference_construction": _public_a3b_receipt(
                category_complete.get("A3b_reference_construction", {})),
            "conditions": {},
        }
        for condition, rows in sealed_inputs[category].items():
            stage_a_seal = _read_json(
                run_root / "stage_a" / category / condition / "stage_a_seal.json")
            labels = np.asarray([labels_json_order[int(row["original_json_position"])]
                                 for row in rows], dtype=np.int8)
            if len(labels) != len(rows) or not np.isin(labels, [0, 1]).all():
                raise ContinuationError(f"{category}/{condition}: evaluator labels are malformed")
            score_keys = sorted({key for row in rows for key in row["scores"]})
            decision_keys = sorted({key for row in rows for key in row["decisions"]})
            condition_metrics = {"event_count": len(rows),
                "normal_count": int(np.sum(labels == 0)),
                "defect_count": int(np.sum(labels == 1)),
                "score_metrics": {}, "decision_metrics": {},
                "calibration_stability": _b3_stability_metrics(stage_a_seal),
                "a3a_divergence": stage_a_seal.get("a3a_divergence"),
                "cmax_G_status": stage_a_seal.get("cmax_G_status"),
                "cmax_G_resource_profile": stage_a_seal.get("cmax_G_resource_profile"),
                "cmax_G_snapshot_diagnostics": stage_a_seal.get(
                    "cmax_G_snapshot_diagnostics", [])}
            for key in score_keys:
                if not all(key in row["scores"] for row in rows):
                    continue
                scores = np.asarray([row["scores"][key] for row in rows], dtype=np.float64)
                if not np.isfinite(scores).all():
                    raise ContinuationError(f"{category}/{condition}:{key}: nonfinite score")
                condition_metrics["score_metrics"][key] = {
                    **r9_rank_metrics(labels, scores),
                    "score_sha256": _sha_bytes(_json_bytes(scores.tolist()))}
                if key.startswith("A3b-"):
                    cuts = (0, len(scores) // 3, (2 * len(scores)) // 3, len(scores))
                    condition_metrics.setdefault("score_segments", {})[key] = {
                        "early_middle_late": [{
                            "start": int(start), "end_exclusive": int(stop),
                            "count": int(stop - start),
                            "mean_score": float(np.mean(scores[start:stop])),
                            "median_score": float(np.median(scores[start:stop])),
                            "std_score": float(np.std(scores[start:stop], ddof=1))
                                if stop - start > 1 else 0.0,
                        } for start, stop in zip(cuts[:-1], cuts[1:]) if stop > start]}
            for key in decision_keys:
                if not all(key in row["decisions"] for row in rows):
                    continue
                predictions = np.asarray([row["decisions"][key]["prediction"]
                                          for row in rows], dtype=np.int8)
                score_key = None
                if key.startswith("A2:G:"):
                    score_key = f"{condition}-A2-shared:G"
                elif key.startswith("A2:GC:"):
                    score_key = f"{condition}-A2-shared:GC"
                elif ":STATIC:C64" in key:
                    score_key = key.split(":STATIC:C64", 1)[0] + ":STATIC"
                elif key.endswith(":G"):
                    score_key = key
                elif key.endswith(":GC"):
                    score_key = key[:-3] + ":G"
                score_metrics = (condition_metrics["score_metrics"].get(score_key, {})
                                 if score_key else {})
                condition_metrics["decision_metrics"][key] = {
                    **r9_classification_metrics(labels, predictions),
                    "ranking": {k: score_metrics.get(k) for k in
                                ("auroc", "average_precision")},
                    "nonoverlap_512": _nonoverlap_window_metrics(labels, predictions)}
            if condition == "P0":
                # Calibration isolation requires a single score vector behind
                # C64 and C128 readouts on each A2 scorer trajectory.
                for scorer in ("G", "GC"):
                    left = condition_metrics["decision_metrics"].get(f"A2:{scorer}:C64", {}).get("ranking", {})
                    right = condition_metrics["decision_metrics"].get(f"A2:{scorer}:C128", {}).get("ranking", {})
                    if (left is None or right is None or
                            (left.get("auroc"), left.get("average_precision")) !=
                            (right.get("auroc"), right.get("average_precision"))):
                        raise ContinuationError("A2 calibration isolation AUROC/AP parity failed")
            condition_metrics["stage_a_seal_sha256"] = seal_hashes[category][condition]
            category_metrics[category][condition] = condition_metrics
            feedback_path = _private_ledger_path(
                run_root / "stage_a" / category / condition, "feedback")
            dynamic_memory_bytes: dict[str, dict[str, int]] = {}
            with feedback_path.open("r", encoding="utf-8") as feedback_stream:
                for feedback_line in feedback_stream:
                    feedback = json.loads(feedback_line)
                    for trajectory, action in feedback.get("writer_actions", {}).items():
                        entry = dynamic_memory_bytes.setdefault(trajectory, {})
                        entry["greedy"] = max(entry.get("greedy", 0),
                            int(action.get("allocated_state_bytes", 0)))
                    for trajectory, update in feedback.get("correction_updates", {}).items():
                        entry = dynamic_memory_bytes.setdefault(trajectory, {})
                        entry["correction"] = max(entry.get("correction", 0),
                            int(update.get("allocated_state_bytes", 0)))
            initial_b3 = stage_a_seal.get("initial_b3", {})
            resource_diagnostics[category]["conditions"][condition] = {
                "cmax_G_status": stage_a_seal.get("cmax_G_status"),
                "cmax_G_resource_profile": stage_a_seal.get("cmax_G_resource_profile"),
                "calibration_reference_feature_bytes": {
                    trajectory: {arm: int(cost.get(
                        "calibration_reference_feature_bytes", 0))
                        for arm, cost in scores.items()}
                    for trajectory, scores in initial_b3.items()},
                "dynamic_memory_bytes_by_trajectory": dynamic_memory_bytes,
                "rescore_seconds": stage_a_seal.get("rescore_seconds", {}),
                "update_and_rescore_seconds": stage_a_seal.get(
                    "update_and_rescore_seconds"),
                "child_condition_wall_seconds": stage_a_seal.get(
                    "child_condition_wall_seconds"),
            }
    # Reuse A1's already sealed, normal-only H diagnostics for the calibration
    # size recommendation. No new TEST selection is performed here.
    a1_h = {}
    a1_root = RUNS_ROOT / f"a_calibration_scale_{A1_RUN_ID}" / "a1"
    for category in PRIMARY_CATEGORIES:
        private = _read_json(a1_root / category / "stage_b_metrics_private.json")
        h_rows = {}
        for condition, payload in private.get("metrics", {}).items():
            arms = payload.get("arms", {})
            h_rows[condition] = {name: arm.get("H_exceedance_fraction")
                for name, arm in arms.items()
                if name.startswith(("Q90:C64", "Q90:C128", "Q90:Cmax-G"))}
        a1_h[category] = h_rows
    result = {"schema": "a_calibration_continuation.stage_b_metrics.v1",
        "status": "SEALED_COMPLETE", "categories": category_metrics,
        "resource_diagnostics": resource_diagnostics,
        "a1_H_normal_only_diagnostics": a1_h,
        "stage_a_seal_hashes": seal_hashes,
        "masks_read": False, "confirmation_categories_accessed": False,
        "label_access": "Stage B only after all twelve Stage A seals verified"}
    _write_once(run_root / "stage_b_metrics_private.json",
                _json_bytes(result) + b"\n")
    stage_b_seal = {"schema": "a_calibration_continuation.stage_b_seal.v1",
        "status": "SEALED_COMPLETE",
        "stage_a_seal_hashes": seal_hashes,
        "metrics_sha256": _sha_file(run_root / "stage_b_metrics_private.json"),
        "masks_read": False, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write_once(run_root / "stage_b_seal.json", _json_bytes(stage_b_seal) + b"\n")
    return {"status": "SEALED_COMPLETE", "metrics_sha256": stage_b_seal["metrics_sha256"]}


def _write_jsonl_append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(row) + b"\n"
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _append_bounded_private_jsonl(path: Path, row: Mapping[str, Any], *,
                                  maximum_bytes: int) -> None:
    """Fsync a new continuation-private JSONL ledger under an explicit byte cap."""
    path = Path(path)
    payload = _json_bytes(row) + b"\n"
    current = int(path.stat().st_size) if path.exists() else 0
    if current + len(payload) > int(maximum_bytes):
        raise ContinuationError(f"private ledger exceeded its registered size cap: {path.name}")
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _private_ledger_path(condition_dir: Path, kind: str) -> Path:
    """Return a private category-level ledger path outside the R7 output budget."""
    if kind not in {"feedback", "progress"}:
        raise ValueError(f"unsupported private condition ledger: {kind}")
    condition_dir = Path(condition_dir)
    return condition_dir.parent / "private_ledgers" / f"{condition_dir.name}.{kind}.jsonl"


def _archive_condition_ledgers(condition_dir: Path, *, attempts_dir: Path) -> None:
    for kind in ("feedback", "progress"):
        ledger = _private_ledger_path(condition_dir, kind)
        if ledger.exists():
            _archive_partial(ledger, attempts_dir=attempts_dir)


def _archive_partial(path: Path, *, attempts_dir: Path) -> Path:
    """Preserve an incomplete attempt before retrying the same sealed stage."""
    path = Path(path)
    attempts_dir = Path(attempts_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    root = attempts_dir.parent.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if root not in resolved.parents:
        raise ContinuationError("refusing to archive an output outside its run root")
    attempts_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = attempts_dir / f"{path.name}.{stamp}.{time.time_ns()}"
    if target.exists():
        raise FileExistsError(target)
    shutil.move(str(path), str(target))
    return target


def _condition_seal_is_valid(path: Path, *, allow_smoke: bool = False) -> bool:
    seal_path = Path(path) / "stage_a_seal.json"
    if not seal_path.is_file():
        return False
    try:
        seal = _read_json(seal_path)
        expected = "INCOMPLETE" if allow_smoke else "SEALED_COMPLETE"
        return (seal.get("status") == expected and
                seal.get("score_rows_sha256") == _sha_file(Path(path) / "stage_a_scores.jsonl") and
                seal.get("feedback_rows_sha256") == _sha_file(
                    _private_ledger_path(Path(path), "feedback")))
    except (OSError, ValueError, KeyError, ContinuationError):
        return False


def _remove_category_scratch(run_root: Path, category: str, *, smoke: bool = False) -> None:
    run_root = Path(run_root).resolve(strict=True)
    suffix = "_smoke" if smoke else ""
    scratch = (run_root / ".scratch" / f"{category}{suffix}").resolve(strict=False)
    if run_root not in scratch.parents:
        raise ContinuationError("scratch path escaped its continuation run root")
    if scratch.exists():
        shutil.rmtree(scratch)


def _retained_output_bytes(run_root: Path) -> int:
    root = Path(run_root).resolve(strict=True)
    total = 0
    for path in root.rglob("*"):
        if ".scratch" in path.parts or not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        if root not in resolved.parents:
            raise ContinuationError("continuation output contains a path outside its run root")
        total += int(path.stat().st_size)
    return total


def _supervised(stage: str, run_id: str, *, category: str | None = None,
                smoke: bool = False, timeout_seconds: int) -> dict[str, Any]:
    from experiments.calibration import r9_correction_mechanism_falsification as r9
    from experiments.calibration import r1_runtime_supervisor as supervisor
    run_root = _run_path(run_id)
    if stage == "CATEGORY" and category is not None:
        # The preceding supervisor invocation has returned only after the Job
        # Object confirmed that the full child process tree exited.
        _remove_category_scratch(run_root, category, smoke=smoke)
    env = r9._child_environment("STAGE_B" if stage == "STAGE_B" else "STAGE_A")
    env["A23_RUN_ID"] = run_id
    command = [sys.executable, "-m", "experiments.calibration.a_calibration_continuation"]
    if stage == "CATEGORY":
        command.extend(["--category-child", "--run-id", run_id,
                        "--category", str(category)])
        if smoke:
            command.extend(["--smoke", "--smoke-events", "64"])
    elif stage == "STAGE_B":
        command.extend(["--stage-b-child", "--run-id", run_id])
    else:
        raise ContinuationError(f"unsupported supervised stage {stage}")
    result = supervisor.run_supervised_child(command,
        timeout_seconds=float(timeout_seconds), env=env, cwd=ROOT)
    record = asdict(result)
    if (record.get("status") != "COMPLETED" or
            record.get("child_exit_code") != 0 or
            record.get("termination_confirmed") is not True or
            record.get("active_job_processes") != 0):
        detail = None
        failures_path = run_root / "child_failures_private.jsonl"
        if failures_path.is_file():
            lines = failures_path.read_text(encoding="utf-8").splitlines()
            if lines:
                failure = json.loads(lines[-1])
                if failure.get("category") == category or stage == "STAGE_B":
                    detail = {key: failure.get(key) for key in
                              ("error_type", "error", "traceback")}
        _write_jsonl_append(run_root / "supervisor_status.jsonl", {
            "stage": stage, "category": category, "smoke": smoke,
            "supervisor": record, "child_detail": detail,
            "process_tree_containment": "Windows Job Object",
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        raise ContinuationError(
            f"supervised {stage} child failed or did not terminate cleanly: "
            f"{record}; child_detail={detail}")
    return {"stage": stage, "category": category, "smoke": smoke,
            "supervisor": record,
            "process_tree_containment": "Windows Job Object",
            "automatic_restart": False}


def _category_complete(run_root: Path, category: str) -> bool:
    complete_path = run_root / "stage_a" / category / "category_complete.json"
    if not complete_path.is_file():
        return False
    complete = _read_json(complete_path)
    if complete.get("status") != "SEALED_COMPLETE":
        return False
    conditions = complete.get("conditions", {})
    if set(conditions) != {"P0", "P1", "P2"}:
        return False
    for condition in ("P0", "P1", "P2"):
        seal_path = run_root / "stage_a" / category / condition / "stage_a_seal.json"
        seal = _read_json(seal_path)
        if (seal.get("status") != "SEALED_COMPLETE" or
                seal.get("score_rows_sha256") != _sha_file(
                    seal_path.parent / "stage_a_scores.jsonl") or
                seal.get("feedback_rows_sha256") != _sha_file(
                    _private_ledger_path(seal_path.parent, "feedback"))):
            return False
    return True


def _verify_run(run_id: str) -> dict[str, Any]:
    run_root = _run_path(run_id)
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    manifest = _read_json(run_root / "run_manifest.json")
    state = _read_json(run_root / "RUN_STATE.json")
    if manifest.get("run_id") != a1.validate_run_id(run_id):
        raise ContinuationError("run id does not match the immutable manifest")
    if (manifest.get("runner_sha256") != _sha_file(Path(__file__).resolve()) or
            manifest.get("experiment_contract_sha256") !=
            _sha_bytes(_json_bytes(_experiment_contract())) or
            manifest.get("source_fingerprints") != _file_fingerprints()):
        raise ContinuationError("current sources or frozen contract differ from this run manifest")
    result = {"run_id": run_id, "run_root": str(run_root),
        "status": state.get("status"),
        "branch": manifest.get("git", {}).get("branch"),
        "source_parent_commit": manifest.get("git", {}).get("head"),
        "completed_categories": state.get("completed_categories", []),
        "stage_a_complete": False, "stage_b_complete": False,
        "unverified_conditions": []}
    stage_a_path = run_root / "stage_a_seal.json"
    if stage_a_path.is_file():
        stage_a = _read_json(stage_a_path)
        result["stage_a_complete"] = stage_a.get("status") == "SEALED_COMPLETE"
        result["stage_a_seal_sha256"] = _sha_file(stage_a_path)
    stage_b_path = run_root / "stage_b_seal.json"
    if stage_b_path.is_file():
        stage_b = _read_json(stage_b_path)
        metrics_path = run_root / "stage_b_metrics_private.json"
        result["stage_b_complete"] = (
            stage_b.get("status") == "SEALED_COMPLETE" and metrics_path.is_file() and
            stage_b.get("metrics_sha256") == _sha_file(metrics_path))
        result["stage_b_seal_sha256"] = _sha_file(stage_b_path)
    for category in PRIMARY_CATEGORIES:
        if not _category_complete(run_root, category):
            result["unverified_conditions"].append(category)
    if result["stage_a_complete"]:
        expected = stage_a.get("category_condition_seal_hashes", {})
        actual = {category: {condition: _sha_file(
                    run_root / "stage_a" / category / condition / "stage_a_seal.json")
                    for condition in ("P0", "P1", "P2")}
                 for category in PRIMARY_CATEGORIES}
        if expected != actual:
            raise ContinuationError("global Stage A seal does not match category-condition seals")
    if result["stage_b_complete"] and result["stage_a_complete"]:
        if stage_b.get("stage_a_seal_hashes") != stage_a.get("category_condition_seal_hashes"):
            raise ContinuationError("Stage B seal does not reference the complete Stage A seal set")
    if state.get("status") == "COMPLETE":
        complete_path = run_root / "complete_seal.json"
        if not complete_path.is_file():
            raise ContinuationError("RUN_STATE says COMPLETE without a complete seal")
        complete = _read_json(complete_path)
        if (not result["stage_a_complete"] or not result["stage_b_complete"] or
                complete.get("stage_a_seal_sha256") != _sha_file(stage_a_path) or
                complete.get("stage_b_seal_sha256") != _sha_file(stage_b_path)):
            raise ContinuationError("complete seal does not bind valid Stage A/B seals")
    return result


def _validate_resume_identity(manifest: Mapping[str, Any], run_id: str,
                              preflight_receipt: Mapping[str, Any]) -> None:
    expected_contract = _sha_bytes(_json_bytes(_experiment_contract()))
    if manifest.get("run_id") != a1.validate_run_id(run_id):
        raise ContinuationError("existing run id differs from its immutable manifest")
    if manifest.get("experiment_contract_sha256") != expected_contract:
        raise ContinuationError("existing run has a different frozen experiment contract")
    if manifest.get("runner_sha256") != _sha_file(Path(__file__).resolve()):
        raise ContinuationError("existing run cannot resume after continuation source changes")
    if manifest.get("source_fingerprints") != preflight_receipt.get("source_fingerprints"):
        raise ContinuationError("existing run cannot resume after dependency fingerprints change")
    recorded_git = manifest.get("git", {})
    current_git = preflight_receipt.get("git", {})
    if (recorded_git.get("branch") != current_git.get("branch") or
            recorded_git.get("head") != current_git.get("head")):
        raise ContinuationError("existing run cannot resume from a different source parent")
    if (manifest.get("protocol_id") != PROTOCOL_ID or
            manifest.get("authorization_id") != AUTHORIZATION_ID):
        raise ContinuationError("existing run belongs to a different protocol or authorization")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or verify the sealed Experiment A A2/A3 continuation")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run-all", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--category-child", action="store_true")
    mode.add_argument("--stage-b-child", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--category", choices=PRIMARY_CATEGORIES)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-events", type=int, default=64)
    args = parser.parse_args(argv)
    try:
        result = _dispatch_cli(args, parser)
    except BaseException as exc:
        if (args.run_id and (args.category_child or args.stage_b_child)):
            try:
                run_root = _run_path(args.run_id)
                if run_root.is_dir():
                    import traceback
                    _write_jsonl_append(run_root / "child_failures_private.jsonl", {
                        "stage": "STAGE_B" if args.stage_b_child else "CATEGORY",
                        "category": args.category,
                        "smoke": bool(args.smoke),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": "".join(traceback.format_exception(exc))[-12000:],
                        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    })
            except BaseException:
                pass
        raise
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False,
                     allow_nan=False))
    return 0


def _dispatch_cli(args: Any, parser: Any) -> dict[str, Any]:
    if args.preflight:
        return preflight()
    if not args.run_id:
        parser.error("--run-id is required for this mode")
    if args.run_all:
        return _run_all(args.run_id)
    if args.verify_only:
        return _verify_run(args.run_id)
    if args.category_child:
        if not args.category:
            parser.error("--category is required with --category-child")
        return _stage_a_category_child(
            args.run_id, args.category, smoke=args.smoke,
            smoke_events=args.smoke_events)
    if args.smoke or args.category:
        parser.error("--stage-b-child does not accept --smoke or --category")
    return _stage_b_child(args.run_id)


def _run_all(run_id: str) -> dict[str, Any]:
    run_id = a1.validate_run_id(run_id)
    preflight_receipt = preflight()
    coverage_receipt = preflight_receipt["runner_coverage_audit"]
    run_root = _run_path(run_id)
    if run_root.exists():
        existing = _read_json(run_root / "run_manifest.json") if (run_root / "run_manifest.json").is_file() else {}
        state = _read_json(run_root / "RUN_STATE.json") if (run_root / "RUN_STATE.json").is_file() else {}
        _validate_resume_identity(existing, run_id, preflight_receipt)
        if state.get("status") == "COMPLETE":
            return _verify_run(run_id)
        if state.get("status") not in {"RUNNING", "INCOMPLETE_CATEGORY", "STAGE_A_COMPLETE",
                                      "INCOMPLETE_STAGE_B", "INCOMPLETE_DELIVERABLES"}:
            raise ContinuationError(f"run is not resumable from state {state.get('status')}")
    else:
        run_root.mkdir(parents=False, exist_ok=False)
        contract = _experiment_contract()
        manifest = {"schema": "a_calibration_continuation.run_manifest.v1",
            "run_id": run_id, "protocol_id": PROTOCOL_ID,
            "authorization_id": AUTHORIZATION_ID,
            "git": preflight_receipt["git"],
            "runner_sha256": _sha_file(Path(__file__).resolve()),
            "experiment_contract_sha256": _sha_bytes(_json_bytes(contract)),
            "experiment_contract": contract,
            "runner_coverage_audit": coverage_receipt,
            "a1_receipt": preflight_receipt["a1_receipt"],
            "r9_identity_manifest_sha256": preflight_receipt["r9_identity_manifest_sha256"],
            "r9_source_parent_commit": preflight_receipt["r9_source_parent_commit"],
            "source_fingerprints": preflight_receipt["source_fingerprints"],
            "condition_matrix": {
                "A2_P0": ["G", "GC", "C64", "C128", "Cmax-G resource gate"],
                "A2_P1_P2": ["G", "GC", "C64", "C128"],
                "A3a_P0": ["independent C64", "independent C128", "Cmax-G resource gate"],
                "A3b_P0": ["F4", "FEXP-KC equal-byte", "C64", "native Greedy"]},
            "access": {"TEST_images": True, "TEST_labels_during_stage_a": "queried only after score fsync",
                       "unqueried_TEST_labels": "Stage B after all seals",
                       "masks": False, "confirmation_categories": False},
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        _write_once(run_root / "run_manifest.json", _json_bytes(manifest) + b"\n")
        _write_once(run_root / "preflight.json", _json_bytes(preflight_receipt) + b"\n")
        state = {"schema": "a_calibration_continuation.run_state.v1",
            "run_id": run_id, "status": "RUNNING", "completed_categories": [],
            "runner_sha256": manifest["runner_sha256"],
            "experiment_contract_sha256": manifest["experiment_contract_sha256"],
            "manifest_sha256": _sha_file(run_root / "run_manifest.json"),
            "campaign_started_epoch": time.time(), "automatic_restart": False}
        _write_once(run_root / "RUN_STATE.json", _json_bytes(state) + b"\n")
    campaign_started = float(state.get("campaign_started_epoch", time.time()))
    statuses_path = run_root / "supervisor_status.jsonl"
    if not any(statuses_path.exists() for _ in [0]):
        statuses_path.touch(exist_ok=False)
    try:
        smoke_status_recorded = any(json.loads(line).get("stage") == "CATEGORY" and
            json.loads(line).get("smoke") is True
            for line in statuses_path.read_text(encoding="utf-8").splitlines())
        smoke_complete_path = run_root / "smoke" / "audiojack" / "category_complete.json"
        smoke_complete = (_read_json(smoke_complete_path)
                          if smoke_complete_path.is_file() else {})
        smoke_seal_valid = (smoke_complete.get("status") == "INCOMPLETE" and
            smoke_complete.get("test_labels_accessed") is False and
            smoke_complete.get("test_masks_accessed") is False and
            smoke_complete.get("conditions", {}).get("P0", {}).get("status") == "INCOMPLETE" and
            smoke_complete.get("conditions", {}).get("P0", {}).get("classification") ==
                "SUPERVISED_SMOKE_NOT_SCIENTIFIC_RESULT")
        smoke_done = smoke_status_recorded and smoke_seal_valid
        if not smoke_done:
            smoke_record = _supervised("CATEGORY", run_id, category="audiojack",
                smoke=True, timeout_seconds=15 * 60)
            _write_jsonl_append(statuses_path, smoke_record)
            smoke_complete = _read_json(smoke_complete_path)
            if (smoke_complete.get("status") != "INCOMPLETE" or
                    smoke_complete.get("conditions", {}).get("P0", {}).get("event_count") != 64 or
                    smoke_complete.get("conditions", {}).get("P0", {}).get("classification") !=
                        "SUPERVISED_SMOKE_NOT_SCIENTIFIC_RESULT"):
                raise ContinuationError("supervised smoke did not produce its required incomplete seal")
        retained_bytes = _retained_output_bytes(run_root)
        if retained_bytes > MAX_OUTPUT_BYTES:
            raise ContinuationError(
                f"retained continuation outputs exceed four GiB: {retained_bytes}")
        state = _read_json(run_root / "RUN_STATE.json")
        completed = set(state.get("completed_categories", []))
        for category in PRIMARY_CATEGORIES:
            if _category_complete(run_root, category):
                completed.add(category)
                continue
            elapsed = time.time() - campaign_started
            if elapsed >= CAMPAIGN_DEADLINE_SECONDS:
                raise ContinuationError("eight-hour campaign deadline reached before all categories completed")
            record = _supervised("CATEGORY", run_id, category=category,
                timeout_seconds=min(CATEGORY_DEADLINE_SECONDS,
                    int(CAMPAIGN_DEADLINE_SECONDS - elapsed)))
            _write_jsonl_append(statuses_path, record)
            if not _category_complete(run_root, category):
                raise ContinuationError(f"{category}: child exited without verified P0/P1/P2 seals")
            retained_bytes = _retained_output_bytes(run_root)
            if retained_bytes > MAX_OUTPUT_BYTES:
                raise ContinuationError(
                    f"retained continuation outputs exceed four GiB: {retained_bytes}")
            completed.add(category)
            state["completed_categories"] = [name for name in PRIMARY_CATEGORIES if name in completed]
            state["status"] = "RUNNING"
            _atomic_json(run_root / "RUN_STATE.json", state)
        if completed != set(PRIMARY_CATEGORIES):
            raise ContinuationError("not all authorized category Stage A seals are complete")
        stage_a_hashes = {category: {
            condition: _sha_file(run_root / "stage_a" / category / condition / "stage_a_seal.json")
            for condition in ("P0", "P1", "P2")} for category in PRIMARY_CATEGORIES}
        global_seal = {"schema": "a_calibration_continuation.stage_a_global_seal.v1",
            "status": "SEALED_COMPLETE", "run_id": run_id,
            "category_condition_seal_hashes": stage_a_hashes,
            "category_count": len(PRIMARY_CATEGORIES),
            "condition_count": 3 * len(PRIMARY_CATEGORIES),
            "test_labels_unqueried_accessed": False, "masks_read": False,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if not (run_root / "stage_a_seal.json").exists():
            _write_once(run_root / "stage_a_seal.json", _json_bytes(global_seal) + b"\n")
        state["status"] = "STAGE_A_COMPLETE"
        state["completed_categories"] = list(PRIMARY_CATEGORIES)
        _atomic_json(run_root / "RUN_STATE.json", state)
        elapsed = time.time() - campaign_started
        if elapsed >= CAMPAIGN_DEADLINE_SECONDS:
            raise ContinuationError("eight-hour campaign deadline reached before Stage B")
        stage_b_seal_path = run_root / "stage_b_seal.json"
        if stage_b_seal_path.is_file():
            existing_stage_b = _read_json(stage_b_seal_path)
            if (existing_stage_b.get("status") != "SEALED_COMPLETE" or
                    not (run_root / "stage_b_metrics_private.json").is_file() or
                    _sha_file(run_root / "stage_b_metrics_private.json") !=
                    existing_stage_b.get("metrics_sha256")):
                raise ContinuationError("existing Stage B seal or checksum is invalid")
        else:
            stage_b_record = _supervised("STAGE_B", run_id,
                timeout_seconds=min(CATEGORY_DEADLINE_SECONDS,
                    int(CAMPAIGN_DEADLINE_SECONDS - elapsed)))
            _write_jsonl_append(statuses_path, stage_b_record)
        if _sha_file(run_root / "stage_b_metrics_private.json") != _read_json(
                stage_b_seal_path).get("metrics_sha256"):
            raise ContinuationError("Stage B aggregate checksum does not match its seal")
        complete_seal_path = run_root / "complete_seal.json"
        if complete_seal_path.is_file():
            complete_seal = _read_json(complete_seal_path)
            if (complete_seal.get("status") != "COMPLETE" or
                    complete_seal.get("stage_a_seal_sha256") !=
                    _sha_file(run_root / "stage_a_seal.json") or
                    complete_seal.get("stage_b_seal_sha256") !=
                    _sha_file(run_root / "stage_b_seal.json")):
                raise ContinuationError("existing complete seal does not match Stage A/B")
        else:
            complete_seal = {"schema": "a_calibration_continuation.complete_seal.v1",
                "status": "COMPLETE", "run_id": run_id,
                "stage_a_seal_sha256": _sha_file(run_root / "stage_a_seal.json"),
                "stage_b_seal_sha256": _sha_file(run_root / "stage_b_seal.json"),
                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            _write_once(complete_seal_path, _json_bytes(complete_seal) + b"\n")
        deliverables = _write_integrated_deliverables(run_id)
        state = _read_json(run_root / "RUN_STATE.json")
        state["status"] = "COMPLETE"
        state["completed_utc"] = complete_seal["completed_utc"]
        state["deliverables"] = deliverables
        _atomic_json(run_root / "RUN_STATE.json", state)
        return {"status": "COMPLETE", "run_id": run_id,
                "complete_seal_sha256": _sha_file(run_root / "complete_seal.json"),
                "deliverables": deliverables}
    except BaseException as exc:
        state = _read_json(run_root / "RUN_STATE.json")
        if (run_root / "complete_seal.json").is_file():
            state["status"] = "INCOMPLETE_DELIVERABLES"
        elif state.get("status") not in {"STAGE_A_COMPLETE", "COMPLETE"}:
            state["status"] = "INCOMPLETE_CATEGORY"
        elif state.get("status") == "STAGE_A_COMPLETE":
            state["status"] = "INCOMPLETE_STAGE_B"
        state["blocker"] = {"type": type(exc).__name__, "message": str(exc)}
        state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _atomic_json(run_root / "RUN_STATE.json", state)
        raise


def _experiment_contract() -> dict[str, Any]:
    return {"protocol_id": PROTOCOL_ID,
        "categories": list(PRIMARY_CATEGORIES),
        "conditions": {"P0": {"support_seed": 0, "order": "SHUFFLE0"},
                       "P1": {"support_seed": 0, "order": "SHUFFLE11"},
                       "P2": {"support_seed": 1, "order": "SUPPORT1_SHUFFLE0"}},
        "score": {"backbone": "DINOv2 ViT-L/14 register",
            "blocks": [5, 11, 17, 23], "dimension": 4096,
            "identity_view": True, "grid": [28, 28], "knn_k": 5,
            "image_aggregation": "maximum patch response"},
        "greedy": {"capacity": K_GREEDY, "native_writer": True,
            "top32_candidate_order": True},
        "correction": {"enabled_for_A2_A3a": True,
            "capacity": K_CORRECTION, "multiplier": CORRECTION_STRENGTH,
            "native_C1_Full": True},
        "query_tape": {"id": "Q-A-TS-v1", "rate": 0.03,
            "seed": 37, "temporal_bins": 16,
            "one_per_official_group": True,
            "exact_identity_match_across_arms": True},
        "B3": {"quantile": 0.90, "method": "higher",
            "clip_delta": B3_DELTA, "scale": "sample_std_ddof_1",
            "sigma_floor": SIGMA_FLOOR},
        "A2": {"trajectory": "shared C64-M0 writer G/GC native trajectory",
            "P0_calibration_sizes": [64, 128, "Cmax-G"],
            "P1_P2_calibration_sizes": [64, 128],
            "Cmax_resource_gate": True},
        "A3a": {"trajectory": "independent per calibration size",
            "P0_calibration_sizes": [64, 128, "Cmax-G"],
            "Cmax_resource_gate": True},
        "A3b": {"F4_images": 4, "candidate_cap": KC_CANDIDATE_CAP,
            "kcenter_prototypes": KC_PROTOTYPE_COUNT,
            "equal_deployed_bytes": True, "candidate_group_balance": True},
        "resource_limits": {"smoke_seconds": 900,
            "category_seconds": CATEGORY_DEADLINE_SECONDS,
            "campaign_seconds": CAMPAIGN_DEADLINE_SECONDS,
            "minimum_free_bytes": MIN_FREE_BYTES,
            "maximum_retained_bytes": MAX_OUTPUT_BYTES}}


def _runner_coverage_audit() -> dict[str, Any]:
    """Fail closed unless every registered A2/A3 branch is wired into orchestration."""
    import inspect
    run_source = inspect.getsource(_run_all)
    category_source = inspect.getsource(_stage_a_category_child)
    condition_source = inspect.getsource(_run_condition)
    static_decision_source = inspect.getsource(_record_static_c64_prediction)
    stage_b_source = inspect.getsource(_stage_b_child)
    checks = {
        "A2_Greedy": "common_c0_greedy_writer_action" in condition_source,
        "A2_Greedy_plus_C1": "_update_native_correction" in condition_source,
        "A2_C64_C128_shared_readout": all(x in condition_source for x in ("C64", "C128"))
            and "A2:G:{size}" in condition_source,
        "A3a_independent_C64_C128": all(x in category_source for x in (
            "P0-A3a-{size}", "_build_native_trajectory", "calibration_sets[size]")),
        "A3b_F4_FEXP_KC_construction": all(x in category_source for x in (
            "_a3b_reference_trajectories", "A3b-F4", "A3b-FEXP-KC")),
        "A3b_static_and_prequential_evaluation": all(x in condition_source for x in (
                "score_only", "_record_static_c64_prediction")) and
                "STATIC:C64" in static_decision_source,
        "Stage_A_event_seals": all(x in condition_source for x in (
            "stage_a_seal.json", "score_gate.seal", "all_scores_sealed_before_feedback")),
        "Stage_B_after_all_twelve_seals": all(x in stage_b_source for x in (
            "P0", "P1", "P2", "SEALED_COMPLETE", "test_labels_for_class")),
        "Combined_results_and_report": "_write_integrated_deliverables(run_id)" in run_source,
        "Cmax_native_resource_gate": all(x in condition_source for x in (
            "PROFILE_PENDING", "HOLD_RESOURCE_LIMIT", "_score_reference_items")),
        "Single_invocation_category_orchestration": all(x in run_source for x in (
            "PRIMARY_CATEGORIES", "_supervised(", "STAGE_B")),
    }
    if not all(checks.values()):
        missing = [name for name, passed in checks.items() if not passed]
        raise ContinuationError(
            "IMPLEMENTATION_INCOMPLETE runner coverage audit: " + ", ".join(missing))
    return {"status": "PASS", "checks": checks,
            "condition_matrix": {
                "A2_P0": ["G", "GC", "C64", "C128", "Cmax-G gated"],
                "A2_P1_P2": ["G", "GC", "C64", "C128"],
                "A3a_P0": ["independent C64", "independent C128", "Cmax-G gated"],
                "A3b_P0": ["F4 static/prequential", "FEXP-KC static/prequential", "C64"]}}


def _integrated_performance_summaries(
        metrics: Mapping[str, Any],
        categories: Sequence[str] = PRIMARY_CATEGORIES) -> dict[str, Any]:
    """Return public, category-averaged summaries used by the integrated report."""
    def metric(category: str, arm: str, name: str) -> Any:
        value = metrics["categories"][category]["P0"]["decision_metrics"].get(arm, {})
        if name in {"auroc", "average_precision"}:
            return value.get("ranking", {}).get(name)
        return value.get(name)

    def mean_valid(values: Sequence[Any]) -> float | None:
        xs = [float(value) for value in values
              if isinstance(value, (int, float)) and math.isfinite(float(value))]
        return float(np.mean(xs)) if xs else None

    a3b_arms = {
        "F4_static": "A3b-F4-STATIC:STATIC:C64",
        "FEXP_KC_static": "A3b-FEXP-KC-STATIC:STATIC:C64",
        "F4_prequential": "A3b-F4:G",
        "FEXP_KC_prequential": "A3b-FEXP-KC:G",
    }
    a3b_summary = {
        name: {name_metric: mean_valid([
            metric(category, arm, name_metric) for category in categories])
            for name_metric in ("auroc", "average_precision", "ber", "fpr", "fnr")}
        for name, arm in a3b_arms.items()}
    a3a_ber = {
        size: mean_valid([
            metric(category, f"P0-A3a-{size}:G", "ber") for category in categories])
        for size in ("C64", "C128")}
    resource = metrics.get("resource_diagnostics", {})
    mean_encoding_seconds = mean_valid([
        resource.get(category, {}).get("test_feature_cache", {}).get("encoding_seconds")
        for category in categories])
    mean_a3b_build_seconds = mean_valid([
        sum(float(resource.get(category, {}).get("a3b_reference_construction", {}).get(key) or 0.0)
            for key in ("candidate_feature_extraction_seconds", "candidate_selection_seconds",
                        "native_reference_bank_build_seconds", "kcenter_seconds"))
        for category in categories])
    mean_cmax_profile_seconds = mean_valid([
        (resource.get(category, {}).get("conditions", {}).get("P0", {}).get(
            "cmax_G_resource_profile") or {}).get("measured_full_reference_rescore_seconds")
        for category in categories])
    return {
        "a3a_g_ber_means_p0": a3a_ber,
        "a3b_means_p0": a3b_summary,
        "mean_test_encoding_seconds_per_category": mean_encoding_seconds,
        "mean_a3b_reference_construction_seconds_per_category": mean_a3b_build_seconds,
        "mean_measured_cmax_rescore_profile_seconds": mean_cmax_profile_seconds,
    }


def _reuse_or_write_integrated_aggregate(path: Path,
                                        aggregate: Mapping[str, Any]) -> bytes:
    """Keep an existing aggregate byte-for-byte when only its report-source digest moved."""
    candidate = _json_bytes(dict(aggregate)) + b"\n"
    if not path.exists():
        _write_once_or_identical(path, candidate)
        return candidate

    existing_bytes = path.read_bytes()
    try:
        existing = json.loads(existing_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinuationError(f"existing integrated aggregate is unreadable: {path}") from exc
    candidate_object = dict(aggregate)
    previous_source_sha = existing.get("report_assembly_sha256")
    if not isinstance(previous_source_sha, str) or len(previous_source_sha) != 64:
        raise ContinuationError("existing aggregate lacks a valid report assembly digest")
    candidate_object["report_assembly_sha256"] = previous_source_sha
    if existing != candidate_object:
        raise ContinuationError(
            "existing aggregate differs from sealed metrics beyond the report-source digest")
    return existing_bytes


def _write_integrated_deliverables(run_id: str) -> dict[str, Any]:
    """Create the identity-redacted combined report after both stage seals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_root = _run_path(run_id).resolve(strict=True)
    complete = _read_json(run_root / "complete_seal.json")
    stage_a_seal = _read_json(run_root / "stage_a_seal.json")
    stage_b_seal = _read_json(run_root / "stage_b_seal.json")
    metrics_path = run_root / "stage_b_metrics_private.json"
    if (complete.get("status") != "COMPLETE" or
            stage_a_seal.get("status") != "SEALED_COMPLETE" or
            stage_b_seal.get("status") != "SEALED_COMPLETE" or
            stage_b_seal.get("metrics_sha256") != _sha_file(metrics_path)):
        raise ContinuationError("integrated deliverables require valid Stage A/B and complete seals")
    metrics = _read_json(metrics_path)
    manifest = _read_json(run_root / "run_manifest.json")
    report_dir = REPORT_ROOT
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for category, conditions in metrics["categories"].items():
        for condition, result in conditions.items():
            for arm, values in result.get("decision_metrics", {}).items():
                ranking = values.get("ranking", {})
                windows = values.get("nonoverlap_512", {})
                worst = windows.get("worst_valid_window") or {}
                rows.append({
                    "category": category, "condition": condition, "arm": arm,
                    "auroc": ranking.get("auroc"),
                    "average_precision": ranking.get("average_precision"),
                    "fpr": values.get("fpr"), "fnr": values.get("fnr"),
                    "ber": values.get("ber"),
                    "balanced_accuracy": values.get("balanced_accuracy"),
                    "valid_512_windows": windows.get("valid_windows"),
                    "worst_valid_window_ber": worst.get("ber"),
                    "cmax_G_status": result.get("cmax_G_status"),
                })
    csv_buffer = io.StringIO(newline="")
    columns = ["category", "condition", "arm", "auroc", "average_precision",
               "fpr", "fnr", "ber", "balanced_accuracy", "valid_512_windows",
               "worst_valid_window_ber", "cmax_G_status"]
    writer = csv.DictWriter(csv_buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_payload = csv_buffer.getvalue().encode("utf-8")

    def p0_metric(category: str, arm: str, metric: str) -> float | None:
        value = metrics["categories"][category]["P0"]["decision_metrics"].get(arm, {})
        if metric in {"auroc", "average_precision"}:
            return value.get("ranking", {}).get(metric)
        return value.get(metric)

    def mean_valid(values: Sequence[Any]) -> float | None:
        xs = [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(float(x))]
        return float(np.mean(xs)) if xs else None

    def save_figure(name: str, fig: Any) -> str:
        fig_payload = io.BytesIO()
        fig.savefig(fig_payload, format="png", dpi=160,
                    metadata={"Software": "Experiment A continuation"})
        plt.close(fig)
        data = fig_payload.getvalue()
        path = figures_dir / name
        _write_once_or_identical(path, data)
        return _sha_bytes(data)

    figure_hashes: dict[str, str] = {}

    # 1. Current-state B3 threshold trajectories for A2's fixed shared scorer.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
    for axis, category in zip(axes.flat, PRIMARY_CATEGORIES):
        seal = _read_json(run_root / "stage_a" / category / "P0" / "stage_a_seal.json")
        histories = seal.get("final_b3_history", {}).get("P0-A2-shared", {})
        for scorer in ("G", "GC"):
            for size in ("C64", "C128", "Cmax-G"):
                history = histories.get(f"{scorer}:{size}")
                if not history:
                    continue
                axis.plot([int(item["version"]) for item in history],
                          [float(item["theta"]) for item in history],
                          label=f"{scorer} {size}", linewidth=1.2)
        axis.set_title(category)
        axis.set_xlabel("Genuine scorer state version")
        axis.set_ylabel("B3 threshold")
        axis.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle("A2 current-state calibration thresholds, P0")
    figure_hashes["threshold_trajectories"] = save_figure(
        "a2_a3_threshold_trajectories.png", fig)

    # 2. A2 calibration-scale sensitivity for BER/FPR/FNR.
    sizes = ("C64", "C128", "Cmax-G")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), squeeze=False)
    for axis, metric in zip(axes.flat, ("ber", "fpr", "fnr")):
        for scorer, marker in (("G", "o"), ("GC", "s")):
            series = []
            for size in sizes:
                vals = [p0_metric(category, f"A2:{scorer}:{size}", metric)
                        for category in PRIMARY_CATEGORIES]
                series.append(mean_valid(vals))
            xs = [i for i, value in enumerate(series) if value is not None]
            axis.plot(xs, [series[i] for i in xs], marker=marker, label=scorer)
        axis.set_xticks(range(len(sizes)), sizes)
        axis.set_title(metric.upper())
        axis.set_ylabel("Mean across four development categories")
        axis.grid(alpha=0.25)
    axes.flat[0].legend()
    fig.suptitle("A2 calibration-scale sensitivity, P0")
    figure_hashes["a2_scale_metrics"] = save_figure(
        "a2_calibration_scale_ber_fpr_fnr.png", fig)

    # 3. G versus GC under the same shared normal-memory trajectory.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), squeeze=False)
    for axis, metric in zip(axes.flat, ("auroc", "average_precision")):
        for size, marker in (("C64", "o"), ("C128", "s")):
            g = mean_valid([p0_metric(c, f"A2:G:{size}", metric)
                            for c in PRIMARY_CATEGORIES])
            gc = mean_valid([p0_metric(c, f"A2:GC:{size}", metric)
                             for c in PRIMARY_CATEGORIES])
            if g is not None and gc is not None:
                axis.plot([0, 1], [g, gc], marker=marker, label=size)
        axis.set_xticks([0, 1], ["Greedy", "Greedy+C1"])
        axis.set_title(metric.replace("_", " ").upper())
        axis.grid(alpha=0.25)
    axes.flat[0].legend()
    fig.suptitle("A2 matched normal-memory trajectory")
    figure_hashes["a2_g_gc"] = save_figure("a2_greedy_vs_greedy_c1.png", fig)

    # 4. A3a first genuine state divergence across independent arms.
    fig, axis = plt.subplots(figsize=(9, 4.8))
    positions, labels = [], []
    for category in PRIMARY_CATEGORIES:
        seal = _read_json(run_root / "stage_a" / category / "P0" / "stage_a_seal.json")
        divergence = seal.get("a3a_divergence", {}).get("first_divergence")
        positions.append(float(divergence["position"]) if divergence else math.nan)
        labels.append(category)
    axis.bar(labels, [value if math.isfinite(value) else 0.0 for value in positions])
    for index, value in enumerate(positions):
        if not math.isfinite(value):
            axis.text(index, 0.02, "none", ha="center", va="bottom", rotation=90,
                      transform=axis.get_xaxis_transform(), fontsize=8)
    axis.set_ylabel("First event position (no divergence shown at zero)")
    axis.set_title("A3a independent C64/C128 state divergence")
    axis.grid(axis="y", alpha=0.25)
    figure_hashes["a3a_divergence"] = save_figure(
        "a3a_first_state_divergence.png", fig)

    # 5. A2 readout-only and A3a coupled-learning performance.
    fig, axis = plt.subplots(figsize=(11, 5))
    comparisons = [
        ("A2 C64", "A2:G:C64"), ("A2 C128", "A2:G:C128"),
        ("A3a C64", "P0-A3a-C64:G"), ("A3a C128", "P0-A3a-C128:G"),
    ]
    means = [mean_valid([p0_metric(c, arm, "ber") for c in PRIMARY_CATEGORIES])
             for _label, arm in comparisons]
    axis.bar([label for label, _arm in comparisons],
             [value if value is not None else 0.0 for value in means])
    axis.set_ylabel("Mean BER across four development categories")
    axis.set_title("Calibration readout versus calibration-coupled learning, G")
    axis.grid(axis="y", alpha=0.25)
    figure_hashes["a2_a3a_coupling"] = save_figure(
        "a2_readout_vs_a3a_coupled_learning.png", fig)

    # 6. Static and prequential A3b reference comparison.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), squeeze=False)
    a3b_plot_arms = [
        ("F4 static", "A3b-F4-STATIC:STATIC:C64"),
        ("FEXP-KC static", "A3b-FEXP-KC-STATIC:STATIC:C64"),
        ("F4 + Greedy", "A3b-F4:G"),
        ("FEXP-KC + Greedy", "A3b-FEXP-KC:G"),
    ]
    for axis, metric in zip(axes.flat, ("auroc", "average_precision")):
        values = [mean_valid([p0_metric(c, arm, metric) for c in PRIMARY_CATEGORIES])
                  for _label, arm in a3b_plot_arms]
        axis.bar([label for label, _arm in a3b_plot_arms],
                 [value if value is not None else 0.0 for value in values])
        axis.set_xticks(range(len(a3b_plot_arms)), [label for label, _arm in a3b_plot_arms],
                        rotation=25, ha="right")
        axis.set_title(metric.replace("_", " ").upper())
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("A3b equal-byte reference banks, P0")
    figure_hashes["a3b_references"] = save_figure(
        "a3b_f4_vs_expanded_reference.png", fig)

    # 7. Actual encoding, construction and resource-gate costs.
    resource = metrics.get("resource_diagnostics", {})
    fig, axis = plt.subplots(figsize=(13, 5.5))
    cost_labels, encoding, coreset, cmax_profiled = [], [], [], []
    for category in PRIMARY_CATEGORIES:
        receipt = resource.get(category, {})
        a3b = receipt.get("a3b_reference_construction", {})
        cache = receipt.get("test_feature_cache", {})
        p0_cond = receipt.get("conditions", {}).get("P0", {})
        profile = p0_cond.get("cmax_G_resource_profile") or {}
        cost_labels.append(category)
        encoding.append(float(cache.get("encoding_seconds") or 0.0))
        coreset.append(float(a3b.get("candidate_feature_extraction_seconds") or 0.0) +
                       float(a3b.get("candidate_selection_seconds") or 0.0) +
                       float(a3b.get("kcenter_seconds") or 0.0))
        cmax_profiled.append(float(profile.get(
            "measured_full_reference_rescore_seconds") or 0.0))
    x = np.arange(len(cost_labels))
    width = 0.25
    axis.bar(x - width, encoding, width, label="TEST DINO encoding")
    axis.bar(x, coreset, width, label="A3b reference construction")
    axis.bar(x + width, cmax_profiled, width, label="Measured Cmax rescore profile")
    axis.set_xticks(x, cost_labels)
    axis.set_ylabel("Seconds")
    axis.set_title("Measured compute costs; Cmax profile is one native rescore only")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure_hashes["compute_costs"] = save_figure(
        "calibration_and_reference_compute_cost.png", fig)

    # Conservative decision candidate: no practical-equivalence margins were
    # preregistered, so report observed effects without inventing cutoffs.
    decision_candidate = "HOLD_CALIBRATION"
    summary_a2 = {}
    for size in ("C64", "C128", "Cmax-G"):
        for scorer in ("G", "GC"):
            summary_a2[f"{scorer}:{size}"] = {
                metric: mean_valid([p0_metric(c, f"A2:{scorer}:{size}", metric)
                                    for c in PRIMARY_CATEGORIES])
                for metric in ("auroc", "average_precision", "ber", "fpr", "fnr")}
    divergence_summary = {}
    for category in PRIMARY_CATEGORIES:
        seal = _read_json(run_root / "stage_a" / category / "P0" / "stage_a_seal.json")
        divergence_summary[category] = seal.get("a3a_divergence", {})
    integrated_summaries = _integrated_performance_summaries(metrics)
    aggregate = {
        "schema": "a_calibration_continuation.integrated_aggregate.v1",
        "run_id": run_id, "protocol_id": PROTOCOL_ID,
        "authorization_id": AUTHORIZATION_ID,
        "status": "COMPLETE", "parent_commit": SOURCE_PARENT,
        "runner_sha256": manifest.get("runner_sha256"),
        "report_assembly_sha256": _sha_file(Path(__file__).resolve()),
        "experiment_contract_sha256": manifest.get("experiment_contract_sha256"),
        "stage_a_seal_sha256": _sha_file(run_root / "stage_a_seal.json"),
        "stage_b_seal_sha256": _sha_file(run_root / "stage_b_seal.json"),
        "complete_seal_sha256": _sha_file(run_root / "complete_seal.json"),
        "registered_coverage": {category: sorted(conditions)
                                 for category, conditions in metrics["categories"].items()},
        "core_a2_means_p0": summary_a2,
        **integrated_summaries,
        "a3a_divergence": divergence_summary,
        "stage_b_metrics": metrics["categories"],
        "resource_diagnostics": resource,
        "figure_sha256": figure_hashes,
        "decision_candidate": decision_candidate,
        "decision_rationale": (
            "HOLD_CALIBRATION: practical-equivalence and meaningful-improvement margins "
            "were not preregistered. The report preserves the observed category-specific "
            "effects for researcher review and does not select a calibration size using TEST BER alone."),
        "raw_test_scores_or_identities_included": False,
        "queried_feedback_included": False,
        "masks_read": False,
        "confirmation_categories_accessed": False,
    }
    aggregate_path = report_dir / "A2_A3_AGGREGATE_RESULTS.json"
    aggregate_bytes = _reuse_or_write_integrated_aggregate(aggregate_path, aggregate)
    aggregate = _read_json(aggregate_path)
    _write_once_or_identical(report_dir / "A2_A3_RESULTS_SUMMARY.csv", csv_payload)

    mean_ber64 = summary_a2["G:C64"]["ber"]
    mean_ber128 = summary_a2["G:C128"]["ber"]
    mean_tv64 = mean_valid([
        metrics["categories"][category]["P0"]["calibration_stability"].get(
            "P0-A2-shared", {}).get("G:C64", {}).get("threshold_total_variation")
        for category in PRIMARY_CATEGORIES])
    mean_tv128 = mean_valid([
        metrics["categories"][category]["P0"]["calibration_stability"].get(
            "P0-A2-shared", {}).get("G:C128", {}).get("threshold_total_variation")
        for category in PRIMARY_CATEGORIES])
    a3b_arms = {
        "F4_static": "A3b-F4-STATIC:STATIC:C64",
        "FEXP_KC_static": "A3b-FEXP-KC-STATIC:STATIC:C64",
        "F4_prequential": "A3b-F4:G",
        "FEXP_KC_prequential": "A3b-FEXP-KC:G",
    }
    category_outcome_lines = []
    for category in PRIMARY_CATEGORIES:
        values = metrics["categories"][category]["P0"]["decision_metrics"]
        get_ber = lambda key: values.get(key, {}).get("ber")
        category_outcome_lines.append(
            f"| {category} | {get_ber('A2:G:C64')} | {get_ber('A2:G:C128')} | "
            f"{get_ber('A2:GC:C64')} | {get_ber('A2:GC:C128')} | "
            f"{get_ber('P0-A3a-C64:G')} | {get_ber('P0-A3a-C128:G')} | "
            f"{get_ber(a3b_arms['F4_static'])} | "
            f"{get_ber(a3b_arms['FEXP_KC_static'])} | "
            f"{get_ber(a3b_arms['F4_prequential'])} | "
            f"{get_ber(a3b_arms['FEXP_KC_prequential'])} |")
    a3a_ber = integrated_summaries["a3a_g_ber_means_p0"]
    a3b_summary = integrated_summaries["a3b_means_p0"]
    mean_encoding_seconds = integrated_summaries["mean_test_encoding_seconds_per_category"]
    mean_a3b_build_seconds = integrated_summaries[
        "mean_a3b_reference_construction_seconds_per_category"]
    mean_cmax_profile_seconds = integrated_summaries[
        "mean_measured_cmax_rescore_profile_seconds"]
    cmax_lines = []
    for category in PRIMARY_CATEGORIES:
        p0 = metrics["categories"][category]["P0"]
        profile = p0.get("cmax_G_resource_profile") or {}
        cmax_lines.append(
            f"- **{category}:** `{p0.get('cmax_G_status')}`; "
            f"measured profile {profile.get('measured_full_reference_rescore_seconds', 'unavailable')} s; "
            f"projected remaining {profile.get('projected_total_remaining_seconds', 'unavailable')} s.")
    category_outcome_table = "\n".join(category_outcome_lines)
    category_outcome_separator = "|---" + "|---:" * 10 + "|"
    parity_lines = []
    for category in PRIMARY_CATEGORIES:
        parity = resource.get(category, {}).get("native_score_update_parity", {})
        parity_lines.append(
            f"- **{category}:** overall `{parity.get('status')}`, native score "
            f"`{(parity.get('native_score') or {}).get('passed')}`, native update "
            f"`{(parity.get('native_update') or {}).get('passed')}`.")
    report = f"""# Experiment A integrated A2/A3 results

**Status:** COMPLETE. **Run:** `{run_id}`. **Decision candidate:** `{decision_candidate}`.

## Scope and integrity

This continuation used the registered four development categories, P0/P1/P2 event orders,
Q-A-TS-v1 queried feedback, native R9/R10 scorer and update functions, and the sealed A1
role memberships. Stage A completed all 12 category-condition streams before Stage B read
unqueried image labels. No confirmation category or pixel mask was accessed. Private score,
identity, and queried-feedback tapes remain under `{run_root}`. The float32 TEST feature
cache is temporary scratch and is removed after each category completes.

Fresh disposable TRAIN-normal score/update parity checks:
""" + "\n".join(parity_lines) + f"""

The contract fixes C64, C128, and Cmax-G as distinct roles; Cmax-G is never described as
strict four-shot. A3b compares F4 with an 8,192-prototype, fp16, 4,096D FEXP-KC bank at
equal deployed fixed-reference bytes, while reporting its larger training-data access.

## Core A2 results

Mean P0 G AUROC/AP/BER/FPR/FNR across the four categories:

| Calibration | AUROC | AP | BER | FPR | FNR | Mean threshold total variation |
|---|---:|---:|---:|---:|---:|---:|
| C64 | {summary_a2['G:C64']['auroc']} | {summary_a2['G:C64']['average_precision']} | {mean_ber64} | {summary_a2['G:C64']['fpr']} | {summary_a2['G:C64']['fnr']} | {mean_tv64} |
| C128 | {summary_a2['G:C128']['auroc']} | {summary_a2['G:C128']['average_precision']} | {mean_ber128} | {summary_a2['G:C128']['fpr']} | {summary_a2['G:C128']['fnr']} | {mean_tv128} |
| Cmax-G | {summary_a2['G:Cmax-G']['auroc']} | {summary_a2['G:Cmax-G']['average_precision']} | {summary_a2['G:Cmax-G']['ber']} | {summary_a2['G:Cmax-G']['fpr']} | {summary_a2['G:Cmax-G']['fnr']} | not pooled when a full trajectory was held |

Greedy+C1 P0 means by calibration (AUROC/AP/BER/FPR/FNR): C64 {summary_a2['GC:C64']}; C128 {summary_a2['GC:C128']}; Cmax-G {summary_a2['GC:Cmax-G']} where a full trajectory was available.

## Per-category outcomes

| Category | A2 G C64 BER | A2 G C128 BER | A2 GC C64 BER | A2 GC C128 BER | A3a G C64 BER | A3a G C128 BER | A3b F4 static BER | A3b FEXP static BER | A3b F4 prequential BER | A3b FEXP prequential BER |
{category_outcome_separator}
{category_outcome_table}

Greedy+C1 P0 means by calibration are recorded in the aggregate. A2 uses a common
normal-memory trajectory and reports C64/C128 readout changes on matched score histories.
P1 and P2 evaluate C64/C128 only. Full-stream ranking scores are invariant to a threshold
readout; any ranking differences between G and GC arise from Correction scoring, not from
the choice of B3 calibration size.

## A3a coupled learning

Each P0 C64/C128/Cmax-G arm owns an independent native trajectory. The first state
divergence, differing writer actions, normal-memory changes, Correction updates, and
subsequent score changes are recorded by category in the aggregate. These results measure
the combined effect of calibration-dependent decisions and later native state updates;
they do not attribute all downstream differences to threshold quality alone.

Mean P0 A3a G BER is {a3a_ber['C64']} for C64 and {a3a_ber['C128']} for C128, compared
with A2 shared-trajectory G means {mean_ber64} and {mean_ber128}. This comparison is
descriptive: A3a changes future learning through native decisions and state updates.

## A3b expanded equal-byte reference

The paired P0 comparison reports static C64-calibrated and prequential Greedy AUROC/AP,
BER/FPR/FNR, exact deployed bytes, image access, feature extraction, k-center, and online
costs. The expanded pool uses one deterministic normal view per eligible official Source
Group, balanced identity-hashed candidate selection, and blockwise native-feature-space
k-center. It receives more predeployment data and compute than F4; equal bytes refer only
to deployed fixed-reference storage.

Mean static AUROC/AP is {a3b_summary['F4_static']['auroc']}/{a3b_summary['F4_static']['average_precision']}
for F4 and {a3b_summary['FEXP_KC_static']['auroc']}/{a3b_summary['FEXP_KC_static']['average_precision']}
for FEXP-KC. With native Greedy adaptation, prequential AUROC/AP is
{a3b_summary['F4_prequential']['auroc']}/{a3b_summary['F4_prequential']['average_precision']}
for F4 and {a3b_summary['FEXP_KC_prequential']['auroc']}/{a3b_summary['FEXP_KC_prequential']['average_precision']} for FEXP-KC.

## Cmax-G resource gate

""" + "\n".join(cmax_lines) + f"""

Snapshot-only Cmax diagnostics are identified as such in each Stage A seal and are not
included as a full B3 trajectory. The sealed resource profiles use the native scorer after
a genuine Greedy write; full Cmax branches were scheduled only by the frozen gate.

## Interpretation and decision candidate

The observed C64-to-C128 mean G BER change is {None if mean_ber64 is None or mean_ber128 is None else mean_ber128 - mean_ber64};
mean threshold total variation is {mean_tv64} for C64 and {mean_tv128} for C128. Per-category
results and normal-only A1 H diagnostics are preserved in the aggregate. A3b identifies
whether a larger legal predeployment reference changes the apparent need for online normal
adaptation. These are development-category outcomes and do not establish confirmation
performance or authorize Experiments B/C.

Mean TEST feature encoding took {mean_encoding_seconds} seconds per category. Mean A3b
reference feature extraction, selection, native bank construction, and k-center took
{mean_a3b_build_seconds} seconds per category. The measured Cmax native rescore profile
averaged {mean_cmax_profile_seconds} seconds; projected cost and dynamic Greedy/C1
allocated bytes are reported by category in the aggregate.

The candidate is **HOLD_CALIBRATION** because no practical-equivalence margin or minimum
meaningful improvement was preregistered. The evidence is reported without creating a
post hoc threshold for declaring C64, C128, or Cmax-G the winner. A researcher can review
the category effects, normal-only H stability, A3a coupling, A3b alternative reference, and
resource burden before choosing a commissioning configuration.

## Reproducibility and files

- Parent commit: `{SOURCE_PARENT}`; continuation source SHA-256: `{manifest.get('runner_sha256')}`.
- Aggregate assembly source SHA-256: `{aggregate.get('report_assembly_sha256')}`; final report renderer SHA-256: `{_sha_file(Path(__file__).resolve())}`. Report-only repairs followed the sealed run; experimental trajectories and Stage B metrics were not rerun or changed.
- Stage A seal: `{_sha_file(run_root / 'stage_a_seal.json')}`.
- Stage B seal: `{_sha_file(run_root / 'stage_b_seal.json')}`.
- Run outputs: `{run_root}`.
- Aggregate: `A2_A3_AGGREGATE_RESULTS.json`; concise CSV: `A2_A3_RESULTS_SUMMARY.csv`.
- Figures: seven validated PNGs under `figures/`.

The aggregate contains metrics and summarized costs only. Raw test scores, event identities,
queried feedback, feature caches, and masks are not included in these public audit files.
"""
    report_bytes = report.encode("utf-8")
    _write_once_or_identical(
        report_dir / "EXPERIMENT_A_A2_A3_INTEGRATED_REPORT.md", report_bytes)
    memo = f"""# Experiment A final calibration decision candidate

**Candidate: {decision_candidate}**

The integrated A2/A3 continuation completed on the four authorized development categories.
The evidence compares shared-trajectory calibration readouts (A2), independent
calibration-coupled learning (A3a), and F4 versus an expanded equal-byte native reference
(A3b). The metrics, normal-only A1 H diagnostics, resource profiles, and category-specific
failures are in `EXPERIMENT_A_A2_A3_INTEGRATED_REPORT.md` and
`A2_A3_AGGREGATE_RESULTS.json`.

No practical equivalence margin or minimum meaningful improvement was preregistered.
Therefore this memo does not designate C64, C128, or Cmax-G as the winner from TEST BER.
The result is a recommendation for researcher review and does not authorize changes to
Experiments B/C.
"""
    _write_once_or_identical(
        report_dir / "EXPERIMENT_A_FINAL_CALIBRATION_DECISION_CANDIDATE.md",
        memo.encode("utf-8"))
    return {"aggregate": str(aggregate_path),
            "aggregate_sha256": _sha_bytes(aggregate_bytes),
            "summary_csv": str(report_dir / "A2_A3_RESULTS_SUMMARY.csv"),
            "report": str(report_dir / "EXPERIMENT_A_A2_A3_INTEGRATED_REPORT.md"),
            "decision_memo": str(report_dir / "EXPERIMENT_A_FINAL_CALIBRATION_DECISION_CANDIDATE.md"),
            "figures": figure_hashes}


def _run_path(run_id: str) -> Path:
    token = a1.validate_run_id(run_id)
    return RUNS_ROOT / f"{RUN_PREFIX}{token}"


if __name__ == "__main__":
    raise SystemExit(main())
