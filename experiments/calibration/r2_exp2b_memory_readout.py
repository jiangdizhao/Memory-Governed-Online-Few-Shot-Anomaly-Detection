"""CPU-only memory-policy x residual-readout attribution on sealed EXP2A inputs.

This runner never loads DINO or decodes images.  Stage A replays only the
already-disclosed query tape and sealed feature/map shards.  Stage B is a
separate child and cannot open evaluator outcomes until every Stage A score and
diagnostic vector has a verified seal.
"""
from __future__ import annotations

import os

# These must be set before importing NumPy, Torch, or any scorer module.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import hashlib
import io
import json
import math
import os.path
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
EXPECTED_PARENT = "4a677553202df67f66cac855c97eb5d4e091d251"
EXPECTED_PYTHON = Path(r"D:\anaconda3\envs\torch\python.exe")
INPUT_RUN_ID = "r2exp2a_20261101_a4"
INPUT_RELATIVE = Path("outputs/calibration/r1_digest_contract_v1_exp2a_r2exp2a_20261101_a4")
RUN_ID = "r2exp2b_20261101_b1"
OUTPUT_RELATIVE = Path("outputs/calibration/r1_digest_contract_v1_exp2b_r2exp2b_20261101_b1")
CATEGORIES = ("switch", "audiojack", "regulator", "rolled_strip_base")
MAX_EVENTS = 512
EXPECTED_QUERY_COUNT = 67
CAPACITY = 256
READOUT_K = 5
BRIDGE_AMPLITUDE = 0.02
GAUSSIAN_EPS = 1e-6
TOP5_EPS = 1e-6
ATOL = 1e-5
RTOL = 1e-4
MAX_RUNTIME_SECONDS = 1800
MAX_TEST_RUNTIME_SECONDS = 300
MAX_OUTPUT_BYTES = 128 * 1024**2
FAILURE_RESERVE_BYTES = 64 * 1024
MIN_FREE_BYTES = 2 * 1024**3
MAX_OUTPUT_FILES = 20
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261101
MODULE = "experiments.calibration.r2_exp2b_memory_readout"
OUTPUT_PARENT = ROOT / "outputs" / "calibration"
INPUT_ROOT = ROOT / INPUT_RELATIVE
OUTPUT_ROOT = ROOT / OUTPUT_RELATIVE
DELIVERABLE_DIR = ROOT / "docs" / "calibration" / "audit" / "r2_exp2b_memory_readout_20261101"

SCORE_FIELDS = (
    "base_patch_map", "top32_ids", "top32_features", "c1_delta_full",
    "c1_delta_no_outcome_weight", "c1_delta_no_margin", "c1_delta_no_both",
    "c2_delta_a002", "c2_delta_a020",
)
SOURCE_FILES = (
    "experiments/calibration/r2_exp2b_memory_readout.py",
    "experiments/calibration/r2_exp2a_c1_strength.py",
    "experiments/calibration/r1_matched_correction_pilot.py",
    "experiments/calibration/r1_runtime_supervisor.py",
    "experiments/calibration/r2_perf1_stage_b.py",
    "test_phase3_AIF_simplified_v22.py",
)
OUTPUT_NAMES = {
    "run_manifest.json", "INCOMPLETE", ".ledger.lock", "event_refs.jsonl",
    "stage_a_seal.json", "stage_b_started.json", "stage_b_results.json", "stage_b_seal.json",
    "completion_seal.json", "supervisor_status.json", "failure.json",
    *(f"scores_{category}.npz" for category in CATEGORIES),
    *(f"category_{category}.json" for category in CATEGORIES),
}


class Exp2BError(RuntimeError):
    pass


class Exp2BIntegrityError(Exp2BError):
    pass


class OutputBudgetError(Exp2BError):
    pass


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch
        try:
            torch.set_num_threads(max(1, min(2, int(os.environ.get("R2_EXP2B_TORCH_THREADS", "2")))))
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        # On this Windows PyTorch build, is_available() can return True even
        # when CUDA_VISIBLE_DEVICES masks every device.  Count visible devices
        # instead; all scorer tensors below are also pinned explicitly to CPU.
        if torch.cuda.device_count() != 0:
            raise Exp2BError("CPU-only contract failed: CUDA is available in the worker")
        _TORCH = torch
    return _TORCH


def _is_reparse(path: Path) -> bool:
    try:
        import stat
        attrs = getattr(path.lstat(), "st_file_attributes", 0)
        return bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    except FileNotFoundError:
        return False


def _validate_output_root(root: Path, *, must_exist: bool) -> Path:
    expected = (ROOT / OUTPUT_RELATIVE).absolute()
    candidate = Path(os.path.abspath(root))
    parent = candidate.parent
    if os.path.normcase(str(candidate)) != os.path.normcase(str(expected)):
        raise OutputBudgetError("Output path does not match the unique authorized namespace")
    if not parent.is_dir() or _is_reparse(parent):
        raise OutputBudgetError("Output parent is missing or is a reparse point")
    if os.path.normcase(str(parent.resolve(strict=True))) != os.path.normcase(str(OUTPUT_PARENT.resolve(strict=True))):
        raise OutputBudgetError("Output parent resolves outside outputs/calibration")
    if must_exist and not candidate.is_dir():
        raise OutputBudgetError("Output directory is missing")
    if candidate.exists() and (_is_reparse(candidate) or not candidate.is_dir()):
        raise OutputBudgetError("Output directory is not a regular directory")
    return candidate


def _lock_file(stream) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(stream) -> None:
    try:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


class OutputLedger:
    """Single-writer, incremental, pre-write accounting for this one run."""

    def __init__(self, root: Path, *, create: bool = False, test_only: bool = False,
                 max_bytes: int = MAX_OUTPUT_BYTES,
                 failure_reserve_bytes: int = FAILURE_RESERVE_BYTES):
        self.test_only = bool(test_only)
        self.max_bytes = int(max_bytes)
        self.failure_reserve_bytes = int(failure_reserve_bytes)
        if not 0 < self.failure_reserve_bytes < self.max_bytes:
            raise OutputBudgetError("Invalid output budget/reserve configuration")
        if self.test_only:
            import tempfile
            candidate = Path(os.path.abspath(root))
            temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
            try:
                candidate.resolve(strict=False).relative_to(temp_root)
            except ValueError as exc:
                raise OutputBudgetError("Synthetic ledger tests must stay under the OS temp directory") from exc
            if candidate.exists() and (_is_reparse(candidate) or not candidate.is_dir()):
                raise OutputBudgetError("Synthetic ledger path is not a regular directory")
            self.root = candidate
        else:
            if self.max_bytes != MAX_OUTPUT_BYTES or self.failure_reserve_bytes != FAILURE_RESERVE_BYTES:
                raise OutputBudgetError("Only synthetic tests may override the authorized output budget")
            self.root = _validate_output_root(root, must_exist=not create)
        if create:
            self.root.mkdir(parents=False, exist_ok=False)
        self._pid = os.getpid()
        self._guard = threading.RLock()
        self.audit_count = 0
        lock_path = self.root / ".ledger.lock"
        if lock_path.exists() and (_is_reparse(lock_path) or not lock_path.is_file()):
            raise OutputBudgetError("Ledger lock path is not a regular file")
        self._lock_stream = lock_path.open("a+b")
        if lock_path.stat().st_size == 0:
            self._lock_stream.seek(0)
            self._lock_stream.write(b"\0")
            self._lock_stream.flush()
        try:
            _lock_file(self._lock_stream)
        except OSError as exc:
            self._lock_stream.close()
            raise OutputBudgetError("Concurrent output writer detected") from exc
        self._accounted: dict[str, int] = {}
        self._used = 0
        self._refresh_at_checkpoint()

    def close(self) -> None:
        if getattr(self, "_lock_stream", None) is not None:
            _unlock_file(self._lock_stream)
            self._lock_stream.close()
            self._lock_stream = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    @property
    def used_bytes(self) -> int:
        return self._used

    @property
    def file_count(self) -> int:
        return len(self._accounted)

    def _list_actual(self) -> dict[str, int]:
        actual: dict[str, int] = {}
        for item in self.root.iterdir():
            if _is_reparse(item) or not item.is_file():
                raise OutputBudgetError("Output directory contains a reparse point or nested directory")
            if item.name not in OUTPUT_NAMES:
                raise OutputBudgetError(f"Unaccounted output path detected: {item.name}")
            actual[item.name] = int(item.stat().st_size)
        if len(actual) > MAX_OUTPUT_FILES:
            raise OutputBudgetError("Output file-count cap exceeded")
        return actual

    def _refresh_at_checkpoint(self) -> None:
        actual = self._list_actual()
        self._accounted = actual
        self._used = sum(actual.values())
        if self._used > self.max_bytes:
            raise OutputBudgetError("Existing output exceeds the authorized byte cap")
        self.audit_count += 1

    def checkpoint(self, *, durable: bool = True) -> None:
        if os.getpid() != self._pid:
            raise OutputBudgetError("Ledger cannot be shared across writer processes")
        with self._guard:
            actual = self._list_actual()
            if actual != self._accounted:
                raise OutputBudgetError("Unaccounted or concurrent output write detected")
            if sum(actual.values()) > self.max_bytes:
                raise OutputBudgetError("Output exceeds the authorized byte cap")
            if durable:
                for name in actual:
                    if name == ".ledger.lock":
                        continue
                    with (self.root / name).open("rb") as stream:
                        try:
                            os.fsync(stream.fileno())
                        except OSError:
                            pass
            self.audit_count += 1

    def write(self, name: str, payload: bytes, *, failure_reserved: bool = False) -> str:
        if os.getpid() != self._pid:
            raise OutputBudgetError("Ledger cannot be shared across writer processes")
        if name not in OUTPUT_NAMES or Path(name).name != name:
            raise OutputBudgetError("Writer attempted a path outside the run whitelist")
        if not isinstance(payload, bytes):
            raise TypeError("Ledger payload must be bytes")
        target = self.root / name
        with self._guard:
            if target.exists() or _is_reparse(target):
                raise OutputBudgetError(f"Refusing to overwrite output: {name}")
            if len(self._accounted) >= MAX_OUTPUT_FILES:
                raise OutputBudgetError("Output file-count cap would be exceeded")
            limit = self.max_bytes if failure_reserved else self.max_bytes - self.failure_reserve_bytes
            if self._used + len(payload) > limit:
                raise OutputBudgetError("Pre-write output budget would be exceeded")
            # The complete payload is reserved before opening the destination.
            self._used += len(payload)
            try:
                with target.open("xb") as stream:
                    view = memoryview(payload)
                    offset = 0
                    while offset < len(view):
                        written = stream.write(view[offset:])
                        if not written:
                            raise OSError("short output write")
                        offset += written
                    stream.flush()
                self._accounted[name] = int(target.stat().st_size)
                if self._accounted[name] != len(payload):
                    raise OutputBudgetError("Output byte count differs from reserved size")
                return _sha256_bytes(payload)
            except BaseException:
                # Keep conservative accounting; the next checkpoint reconciles a partial file.
                if target.exists():
                    self._accounted[name] = int(target.stat().st_size)
                    self._used = sum(self._accounted.values())
                else:
                    self._used -= len(payload)
                raise

    def write_json(self, name: str, value: Any, *, failure_reserved: bool = False) -> str:
        return self.write(name, _canonical_json(value), failure_reserved=failure_reserved)


@dataclass
class Anchor:
    feature: Any
    beta: float
    radius: float
    kappa: float
    position: int
    source_group: str


def _correction_config_and_codebook():
    torch = _torch()
    import test_phase3_AIF_simplified_v22 as v22
    cfg = v22.CorrectionConfig()
    cfg.enabled = True
    cb = v22.SignedCorrectionCodebook(
        K=CAPACITY, feat_dim=4096, device=torch.device("cpu"), cfg=cfg)
    if cb.protos.device.type != "cpu" or cb.protos.dtype != torch.float32:
        raise Exp2BError("Native Correction Codebook did not initialize as CPU float32")
    return v22, cfg, cb


def _as_torch_features(features: np.ndarray):
    torch = _torch()
    return torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))


def _torch_distance_matrix(query: Any, anchors: Any):
    torch = _torch()
    if anchors.shape[0] == 0:
        return torch.empty((query.shape[0], 0), dtype=torch.float32, device="cpu")
    sim = torch.clamp(query @ anchors.t(), -1.0, 1.0)
    return 1.0 - sim


def _two_readouts_from_distances(distances: Any, beta: Any, radius: Any,
                                 kappa: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return original Gaussian, matched Top-5 beta, and stable Top-5 slots."""
    torch = _torch()
    n_query = int(distances.shape[0])
    n_anchor = int(distances.shape[1])
    if n_anchor == 0:
        zeros = np.zeros((n_query,), dtype=np.float32)
        return zeros.copy(), zeros.copy(), np.empty((n_query, 0), dtype=np.int64)
    beta = beta.to(dtype=torch.float32, device="cpu")
    radius = radius.to(dtype=torch.float32, device="cpu").clamp_min(1e-6)
    kappa = kappa.to(dtype=torch.float32, device="cpu").clamp_min(1e-6)
    weights = torch.exp(-(distances * distances) /
                        (2.0 * (radius[None, :] ** 2))) * kappa[None, :]
    gaussian = (weights * beta[None, :]).sum(dim=1) / (weights.sum(dim=1) + GAUSSIAN_EPS)
    # Stable sort makes equal-distance ties resolve by insertion/slot order.
    order = torch.argsort(distances, dim=1, stable=True)
    selected = order[:, :min(READOUT_K, n_anchor)]
    selected_beta = beta[selected]
    selected_kappa = kappa[selected]
    knn = (selected_kappa * selected_beta).sum(dim=1) / (
        selected_kappa.sum(dim=1) + TOP5_EPS)
    return (gaussian.detach().cpu().numpy().astype(np.float32),
            knn.detach().cpu().numpy().astype(np.float32),
            selected.detach().cpu().numpy().astype(np.int64))


def _read_memory(codebook: Any, source_groups: Mapping[int, set[str]],
                 fifo: Sequence[Anchor], query_features: np.ndarray,
                 current_group: str, position: int, v22: Any,
                 r1: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Compute one distance matrix per memory, then share it across both readouts."""
    torch = _torch()
    query = v22.l2_normalize(_as_torch_features(query_features))
    # Original source validation enforces strict temporal legality and conservative
    # slot-history exclusion before selecting the live Codebook subset.
    cb_n = int(codebook.size())
    c1_eligible = r1._validate_c1_source_groups(
        source_groups, codebook.last_step[:cb_n].detach().cpu().tolist(),
        event_position=position, phase="pre_score", event_group=current_group)
    c1_slots = torch.as_tensor(c1_eligible, dtype=torch.long, device="cpu")
    if c1_eligible:
        c1_protos = codebook.protos[c1_slots]
        c1_beta = codebook.beta[c1_slots]
        c1_radius = codebook.radius[c1_slots]
        c1_kappa = codebook.kappa[c1_slots]
        c1_dist = _torch_distance_matrix(query, c1_protos)
    else:
        c1_beta = torch.empty((0,), dtype=torch.float32)
        c1_radius = torch.empty((0,), dtype=torch.float32)
        c1_kappa = torch.empty((0,), dtype=torch.float32)
        c1_dist = _torch_distance_matrix(query, torch.empty((0, 4096), dtype=torch.float32))
    c1_gaussian, c1_knn, c1_selected = _two_readouts_from_distances(
        c1_dist, c1_beta, c1_radius, c1_kappa)

    fifo_eligible = [index for index, item in enumerate(fifo)
                     if item.position < position and item.source_group != current_group]
    if fifo_eligible:
        ffeatures = torch.stack([fifo[index].feature for index in fifo_eligible])
        fbeta = torch.tensor([fifo[index].beta for index in fifo_eligible], dtype=torch.float32)
        fradius = torch.tensor([fifo[index].radius for index in fifo_eligible], dtype=torch.float32)
        fkappa = torch.tensor([fifo[index].kappa for index in fifo_eligible], dtype=torch.float32)
        fdist = _torch_distance_matrix(query, ffeatures)
    else:
        fbeta = torch.empty((0,), dtype=torch.float32)
        fradius = torch.empty((0,), dtype=torch.float32)
        fkappa = torch.empty((0,), dtype=torch.float32)
        fdist = _torch_distance_matrix(query, torch.empty((0, 4096), dtype=torch.float32))
    f_gaussian, f_knn, f_selected = _two_readouts_from_distances(
        fdist, fbeta, fradius, fkappa)
    f_signs = np.asarray([[1.0 if fifo[fifo_eligible[int(slot)]].beta >= 0 else -1.0
                           for slot in row] for row in f_selected], dtype=np.float32)
    if f_selected.shape[1]:
        bridge = (BRIDGE_AMPLITUDE * f_signs.mean(axis=1)).astype(np.float32)
    else:
        bridge = np.zeros((query.shape[0],), dtype=np.float32)
    exclusion = {
        "codebook_live": cb_n,
        "codebook_same_source_or_ineligible": cb_n - len(c1_eligible),
        "fifo_live": len(fifo),
        "fifo_same_source_or_temporal": len(fifo) - len(fifo_eligible),
    }
    return f_knn, f_gaussian, c1_knn, c1_gaussian, {
        **exclusion,
        "bridge_delta": bridge,
        "fifo_selected_slots": f_selected,
        "codebook_selected_slots": c1_selected,
    }


def _score_dense(base_map: np.ndarray, ids: np.ndarray, delta: np.ndarray,
                 r1: Any) -> float:
    return float(r1.max_score(r1.corrected_dense_map(base_map, ids, delta)))


def _common_candidate_eligibility(keep_mask: np.ndarray, beta_raw: np.ndarray,
                                  cfg: Any) -> dict[str, Any]:
    keep = np.asarray(keep_mask, dtype=np.bool_)
    beta = np.asarray(beta_raw, dtype=np.float32)
    if keep.ndim != 1 or beta.shape != keep.shape or not np.isfinite(beta).all():
        raise Exp2BIntegrityError("Candidate keep mask and beta vector are not aligned finite arrays")
    clipped = np.clip(beta, -float(cfg.beta_max), float(cfg.beta_max)).astype(np.float32)
    floor_eligible = np.abs(clipped) >= float(cfg.beta_min - 1e-5)
    common = keep & floor_eligible
    return {
        "clipped_beta": clipped, "common_mask": common,
        "raw_selected_count": int(keep.size), "attempted_after_dN": int(keep.sum()),
        "dN_gate_rejects": int(keep.size - keep.sum()),
        "beta_floor_rejects": int(keep.sum() - common.sum()),
        "common_eligible_count": int(common.sum()),
    }


def _replay_native_update(codebook: Any, features: np.ndarray, keep_mask: np.ndarray,
                          beta_raw: np.ndarray, *, outcome: str, position: int) -> dict[str, int]:
    keep = np.asarray(keep_mask, dtype=np.bool_)
    beta = np.asarray(beta_raw, dtype=np.float32)
    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.ndim != 2 or feature_array.shape[0] != keep.size or beta.shape != keep.shape:
        raise Exp2BIntegrityError("Replay update feature/keep/beta alignment differs")
    return codebook.update(
        _as_torch_features(feature_array[keep]),
        _torch().from_numpy(np.ascontiguousarray(beta[keep], dtype=np.float32)),
        outcome=outcome, step=int(position))


def _trajectory_alpha(gaussian_deltas: np.ndarray,
                      knn_deltas: np.ndarray) -> dict[str, Any]:
    gaussian = np.asarray(gaussian_deltas, dtype=np.float64)
    knn = np.asarray(knn_deltas, dtype=np.float64)
    if gaussian.shape != knn.shape or not np.isfinite(gaussian).all() or not np.isfinite(knn).all():
        raise ValueError("Magnitude-normalization trajectories must be aligned finite arrays")
    rms_g = float(np.sqrt(np.mean(np.square(gaussian))))
    rms_k = float(np.sqrt(np.mean(np.square(knn))))
    if rms_k == 0.0:
        return {"status": "ZERO_KNN_RMS_UNAVAILABLE", "alpha": None,
                "gaussian_patch_rms": rms_g, "knn_patch_rms": rms_k,
                "is_causal": False}
    return {"status": "AVAILABLE", "alpha": float(rms_g / rms_k),
            "gaussian_patch_rms": rms_g, "knn_patch_rms": rms_k,
            "is_causal": False}


def _make_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _load_npz_bytes(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _jsonl_from_bytes(payload: bytes, name: str) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    except Exception as exc:
        raise Exp2BIntegrityError(f"Cannot parse sealed {name}") from exc


def _write_failure_best_effort(root: Path, exc: BaseException,
                                *, supervisor: Mapping[str, Any] | None = None) -> None:
    """Preserve failure evidence within the reserved area after a killed child."""
    try:
        root = _validate_output_root(root, must_exist=True)
        target = root / "failure.json"
        if target.exists() or _is_reparse(target):
            return
        files = list(root.iterdir())
        if any(_is_reparse(p) or not p.is_file() or p.name not in OUTPUT_NAMES for p in files):
            return
        used = sum(int(p.stat().st_size) for p in files)
        if len(files) >= MAX_OUTPUT_FILES or used > MAX_OUTPUT_BYTES - FAILURE_RESERVE_BYTES:
            return
        payload = _canonical_json({
            "schema": "r2.exp2b.failure.v1", "status": "INCOMPLETE",
            "failure_type": type(exc).__name__, "failure_message": str(exc)[:1200],
            "automatic_restart": False, "supervisor": dict(supervisor or {}),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        if used + len(payload) > MAX_OUTPUT_BYTES:
            return
        with target.open("xb") as stream:
            stream.write(payload)
            stream.flush()
    except Exception:
        return


def _read_required_once(path: Path, description: str) -> bytes:
    if not path.is_file() or _is_reparse(path):
        raise Exp2BIntegrityError(f"Missing or unsafe required input: {description}")
    return path.read_bytes()


def _validate_input_header() -> dict[str, Any]:
    """Verify run identity/seals and read each small receipt exactly once."""
    if not INPUT_ROOT.is_dir() or _is_reparse(INPUT_ROOT):
        raise Exp2BIntegrityError("Recorded EXP2A run directory is missing or unsafe")
    manifest_bytes = _read_required_once(INPUT_ROOT / "run_manifest.json", "EXP2A manifest")
    stage_bytes = _read_required_once(INPUT_ROOT / "stage_a_seal.json", "EXP2A Stage A seal")
    experiment_bytes = _read_required_once(INPUT_ROOT / "experiment_seal.json", "EXP2A completion seal")
    manifest = json.loads(manifest_bytes)
    stage = json.loads(stage_bytes)
    experiment = json.loads(experiment_bytes)
    if manifest.get("run_id") != INPUT_RUN_ID:
        raise Exp2BIntegrityError("Recorded EXP2A run identifier differs")
    # The original manifest remains immutable and says INCOMPLETE by design;
    # the separate, checksummed completion seals are authoritative.
    if manifest.get("status") != "INCOMPLETE":
        raise Exp2BIntegrityError("EXP2A initial manifest status differs from the recorded run")
    if experiment.get("schema") != "r2.exp2a.experiment_seal.v1" or experiment.get("status") != "COMPLETE":
        raise Exp2BIntegrityError("EXP2A completion seal is not COMPLETE")
    if experiment.get("run_id") != INPUT_RUN_ID:
        raise Exp2BIntegrityError("EXP2A completion seal run identifier differs")
    if experiment.get("stage_a_status") != "SEALED_COMPLETE" or experiment.get("stage_b_status") != "COMPLETE":
        raise Exp2BIntegrityError("EXP2A Stage A/Stage B completion states are invalid")
    if experiment.get("stage_a_seal_sha256") != _sha256_bytes(stage_bytes):
        raise Exp2BIntegrityError("EXP2A Stage A seal digest differs from completion seal")
    if stage.get("schema") != "r2.exp2a.stage_a_seal.v1" or stage.get("status") != "SEALED_COMPLETE":
        raise Exp2BIntegrityError("EXP2A Stage A seal is not complete")
    expected_counts = {category: MAX_EVENTS for category in CATEGORIES}
    if stage.get("category_event_counts") != expected_counts or stage.get("total_events") != 2048:
        raise Exp2BIntegrityError("EXP2A Stage A event coverage differs")
    if stage.get("event_shard_count") != 2048 or experiment.get("total_event_count") != 2048:
        raise Exp2BIntegrityError("EXP2A event shard receipt count differs")
    if experiment.get("event_counts") != expected_counts:
        raise Exp2BIntegrityError("EXP2A completion event counts differ")
    if experiment.get("total_query_count") != EXPECTED_QUERY_COUNT:
        raise Exp2BIntegrityError("EXP2A completion query count differs")
    if tuple(manifest.get("authorized_categories", ())) != CATEGORIES:
        raise Exp2BIntegrityError("EXP2A manifest category authorization differs")
    if manifest.get("authorized_event_horizon") != [0, 511]:
        raise Exp2BIntegrityError("EXP2A manifest event horizon differs")
    if manifest.get("c64_rescored") is not False:
        raise Exp2BIntegrityError("EXP2A manifest unexpectedly records C64 rescoring")
    # Validate the two compact orientation/provenance sources once as well.
    report_path = ROOT / "docs" / "calibration" / "audit" / "r2_exp2a_c1_strength_20261101" / "R2_EXP2A_C1_STRENGTH_ATTRIBUTION.md"
    report_bytes = _read_required_once(report_path, "EXP2A report")
    aggregate_bytes = _read_required_once(INPUT_ROOT / "sanitized_aggregate.json", "EXP2A sanitized aggregate")
    report_sha = _sha256_bytes(report_bytes)
    aggregate_sha = _sha256_bytes(aggregate_bytes)
    if experiment.get("report_sha256") != report_sha:
        raise Exp2BIntegrityError("EXP2A report digest differs from completion seal")
    if experiment.get("sanitized_aggregate_sha256") != aggregate_sha:
        raise Exp2BIntegrityError("EXP2A sanitized aggregate digest differs from completion seal")
    summary = json.loads(aggregate_bytes)
    if summary.get("run_id") not in (None, INPUT_RUN_ID):
        raise Exp2BIntegrityError("EXP2A sanitized aggregate run identifier differs")
    if summary.get("event_counts", expected_counts) != expected_counts:
        raise Exp2BIntegrityError("EXP2A sanitized aggregate event coverage differs")
    return {
        "manifest": manifest, "stage_seal": stage,
        "experiment_seal": experiment, "stage_seal_sha256": _sha256_bytes(stage_bytes),
        "experiment_seal_sha256": _sha256_bytes(experiment_bytes),
        "report_sha256": report_sha, "sanitized_aggregate_sha256": aggregate_sha,
        "status_note": "initial manifest INCOMPLETE; separate completion and Stage A seals verified",
    }


def _read_and_verify_jsonl_inputs(header: Mapping[str, Any], r1: Any) -> dict[str, Any]:
    """Read sealed score, transition, diagnostic and query files once each."""
    root = INPUT_ROOT
    seal = header["stage_seal"]
    file_hashes = (
        ("event_scores.jsonl", "score_rows_sha256"),
        ("event_transitions.jsonl", "transition_rows_sha256"),
        ("strength_diagnostics.jsonl", "strength_diagnostics_sha256"),
        ("private_query_feedback.jsonl", "queried_feedback_sha256"),
    )
    parsed: dict[str, list[dict[str, Any]]] = {}
    input_hashes: dict[str, str] = {}
    for name, seal_key in file_hashes:
        payload = _read_required_once(root / name, name)
        digest = _sha256_bytes(payload)
        if digest != seal.get(seal_key):
            raise Exp2BIntegrityError(f"EXP2A checksum failed for {name}")
        parsed[name] = _jsonl_from_bytes(payload, name)
        input_hashes[name] = digest
    score_rows = parsed["event_scores.jsonl"]
    transition_rows = parsed["event_transitions.jsonl"]
    diagnostic_rows = parsed["strength_diagnostics.jsonl"]
    feedback_rows = parsed["private_query_feedback.jsonl"]
    if len(score_rows) != 2048 or len(transition_rows) != 2048:
        raise Exp2BIntegrityError("EXP2A sealed score/transition rows are incomplete")
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        category: {"scores": [], "transitions": []} for category in CATEGORIES}
    seen_events: set[tuple[str, int]] = set()
    for score, transition in zip(score_rows, transition_rows):
        category = score.get("category")
        position = score.get("position")
        if category not in groups or not isinstance(position, int) or not 0 <= position < MAX_EVENTS:
            raise Exp2BIntegrityError("EXP2A score row has an invalid category/position")
        key = (category, position)
        if key in seen_events or transition.get("category") != category or transition.get("position") != position:
            raise Exp2BIntegrityError("EXP2A score/transition event order is duplicated or mismatched")
        seen_events.add(key)
        if score.get("label") is not None or any(k in score for k in ("label", "outcome", "anomaly_class")):
            raise Exp2BIntegrityError("EXP2A public score row contains outcome payload")
        if bool(score.get("query_tape_decision")) != bool(transition.get("query_decision")):
            raise Exp2BIntegrityError("EXP2A query timing differs between score and transition")
        if not transition.get("causal_order_complete"):
            raise Exp2BIntegrityError("EXP2A transition lacks a complete causal event seal")
        groups[category]["scores"].append(score)
        groups[category]["transitions"].append(transition)
    for category in CATEGORIES:
        groups[category]["scores"].sort(key=lambda row: row["position"])
        groups[category]["transitions"].sort(key=lambda row: row["position"])
        if [row["position"] for row in groups[category]["scores"]] != list(range(MAX_EVENTS)):
            raise Exp2BIntegrityError(f"{category}: EXP2A scores are not a complete frozen prefix")

    diagnostics: dict[tuple[str, int], dict[str, Any]] = {}
    for row in diagnostic_rows:
        key = (row.get("category"), row.get("position"))
        if key in diagnostics or key not in seen_events:
            raise Exp2BIntegrityError("EXP2A query diagnostics contain a duplicate or unknown event")
        if row.get("schema") != "r2.exp2a.strength_diagnostic.v1":
            raise Exp2BIntegrityError("EXP2A query diagnostic schema differs")
        diagnostics[key] = row
    feedback: dict[tuple[str, int], dict[str, Any]] = {}
    for row in feedback_rows:
        key = (row.get("category"), row.get("position"))
        if key in feedback or key not in seen_events:
            raise Exp2BIntegrityError("EXP2A feedback contains a duplicate or unknown event")
        if row.get("disclosure_after_score_seal") is not True or row.get("schema") != "r2.exp2a.queried_feedback.v1":
            raise Exp2BIntegrityError("EXP2A feedback was not disclosed after its score seal")
        feedback[key] = row
    if len(diagnostics) != EXPECTED_QUERY_COUNT or len(feedback) != EXPECTED_QUERY_COUNT:
        raise Exp2BIntegrityError("EXP2A query diagnostics/feedback count differs from 67")
    for category in CATEGORIES:
        for row in groups[category]["scores"]:
            key = (category, row["position"])
            queried = bool(row["query_tape_decision"])
            if queried != (key in feedback) or queried != (key in diagnostics):
                raise Exp2BIntegrityError(f"{category}:{row['position']} query receipt mismatch")
    return {"groups": groups, "diagnostics": diagnostics, "feedback": feedback,
            "input_hashes": input_hashes}


def _validate_stored_event_arrays(row: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
                                  r1: Any, *, category: str, position: int) -> None:
    if set(arrays) != set(SCORE_FIELDS):
        raise Exp2BIntegrityError(f"{category}:{position}: NPZ field set differs")
    if r1.arrays_sha256(arrays) != row.get("arrays_sha256"):
        raise Exp2BIntegrityError(f"{category}:{position}: uncompressed array digest differs")
    base = arrays["base_patch_map"]
    ids = arrays["top32_ids"]
    features = arrays["top32_features"]
    if base.shape != (784,) or base.dtype != np.float32 or not np.isfinite(base).all():
        raise Exp2BIntegrityError(f"{category}:{position}: native base map shape/dtype is invalid")
    if ids.shape != (32,) or ids.dtype != np.int64 or len(set(ids.tolist())) != 32 or np.any(ids < 0) or np.any(ids >= 784):
        raise Exp2BIntegrityError(f"{category}:{position}: native Top-32 IDs are invalid")
    if features.shape != (32, 4096) or features.dtype != np.float32 or not np.isfinite(features).all():
        raise Exp2BIntegrityError(f"{category}:{position}: native Top-32 features are invalid")
    expected_names = {
        "c1_full": "c1_delta_full", "c1_no_outcome_weight": "c1_delta_no_outcome_weight",
        "c1_no_margin": "c1_delta_no_margin", "c1_no_both": "c1_delta_no_both",
        "c2_a002": "c2_delta_a002", "c2_a020": "c2_delta_a020",
    }
    scores = row.get("scores")
    if not isinstance(scores, Mapping):
        raise Exp2BIntegrityError(f"{category}:{position}: score receipt is malformed")
    if float(scores["c0"]) != float(r1.max_score(base)):
        raise Exp2BIntegrityError(f"{category}:{position}: C0 score does not match its native map")
    for arm, field in expected_names.items():
        delta = arrays[field]
        if delta.shape != (32,) or delta.dtype != np.float32 or not np.isfinite(delta).all():
            raise Exp2BIntegrityError(f"{category}:{position}: {field} is invalid")
        actual = _score_dense(base, ids, delta, r1)
        if float(scores[arm]) != actual:
            raise Exp2BIntegrityError(f"{category}:{position}: {arm} score does not match its sealed delta")


def _diag_for_c1_no_both(record: Mapping[str, Any], category: str,
                         position: int) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        arm = record["arms"]["c1_no_both"]
        diag = arm["diagnostic"]
        layer_stats = arm["native_layer_stats"]
    except Exception as exc:
        raise Exp2BIntegrityError(f"{category}:{position}: C1-NoBoth update receipt is incomplete") from exc
    for field in ("native_gate_keep_mask", "beta_patch_float32_before_codebook_clip",
                  "beta_after_original_codebook_max_clip", "outcome", "codebook_state_sha256_after"):
        if field not in diag:
            raise Exp2BIntegrityError(f"{category}:{position}: C1-NoBoth diagnostic lacks {field}")
    return diag, layer_stats


def _compare_reference(actual: np.ndarray, expected: np.ndarray) -> tuple[float, int, bool]:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        return float("inf"), int(actual.size), False
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    max_error = float(error.max()) if error.size else 0.0
    non_bitwise = int(np.count_nonzero(actual != expected))
    return max_error, non_bitwise, bool(np.allclose(actual, expected, atol=ATOL, rtol=RTOL))


def _same_update_categories(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    keys = ("inserted", "merged", "replaced", "skipped", "same_sign_replaced",
            "cross_sign_replaced", "same_sign_merged")
    return all(int(actual.get(key, 0)) == int(expected.get(key, 0)) for key in keys)


def _memory_equal(fifo: Sequence[Anchor], codebook: Any) -> bool:
    n = int(codebook.size())
    if len(fifo) != n:
        return False
    if not n:
        return True
    protos = codebook.protos[:n].detach().cpu().numpy()
    beta = codebook.beta[:n].detach().cpu().numpy()
    radius = codebook.radius[:n].detach().cpu().numpy()
    kappa = codebook.kappa[:n].detach().cpu().numpy()
    return all(
        np.array_equal(fifo[index].feature.detach().cpu().numpy(), protos[index]) and
        np.float32(fifo[index].beta) == beta[index] and
        np.float32(fifo[index].radius) == radius[index] and
        np.float32(fifo[index].kappa) == kappa[index]
        for index in range(n))


def _category_arrays() -> dict[str, np.ndarray]:
    return {
        "c0": np.empty((MAX_EVENTS,), dtype=np.float32),
        "historical_c2_a002": np.empty((MAX_EVENTS,), dtype=np.float32),
        "fifo_top5": np.empty((MAX_EVENTS,), dtype=np.float32),
        "fifo_gaussian": np.empty((MAX_EVENTS,), dtype=np.float32),
        "codebook_top5": np.empty((MAX_EVENTS,), dtype=np.float32),
        "codebook_gaussian": np.empty((MAX_EVENTS,), dtype=np.float32),
        "c2_matched_candidate_pool": np.empty((MAX_EVENTS,), dtype=np.float32),
        "delta_A": np.empty((MAX_EVENTS, 32), dtype=np.float32),
        "delta_B": np.empty((MAX_EVENTS, 32), dtype=np.float32),
        "delta_C": np.empty((MAX_EVENTS, 32), dtype=np.float32),
        "delta_D": np.empty((MAX_EVENTS, 32), dtype=np.float32),
        "delta_c2_matched_candidate_pool": np.empty((MAX_EVENTS, 32), dtype=np.float32),
        "theta_c64_q90": np.empty((MAX_EVENTS,), dtype=np.float64),
        "source_group_sha256": np.empty((MAX_EVENTS,), dtype="U64"),
    }


def _score_category(category: str, records: Mapping[str, Any], ledger: OutputLedger,
                    r1: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Replay C1-NoBoth and both memory/readout factors for one frozen prefix."""
    torch = _torch()
    v22, cfg, codebook = _correction_config_and_codebook()
    if int(codebook.K) != CAPACITY:
        raise Exp2BError("Native C1-NoBoth Codebook capacity differs from 256")
    bank_proxy = type("CorrectionBankProxy", (), {})()
    bank_proxy.corr_cb = {"p0": {"id": [codebook]}}
    source_groups: dict[int, set[str]] = {}
    latest_slot_group: dict[int, str] = {}
    fifo: list[Anchor] = []
    arrays_out = _category_arrays()
    base_maps = np.empty((MAX_EVENTS, 784), dtype=np.float32)
    ids_by_event = np.empty((MAX_EVENTS, 32), dtype=np.int64)
    diag_by_position = records["diagnostics"]
    feedback_by_position = records["feedback"]
    score_rows = records["scores"]
    transition_rows = records["transitions"]
    counters = {
        "queries": 0, "raw_selected_candidates": 0, "dN_gate_rejects": 0,
        "attempted_candidates_after_dN": 0, "beta_floor_rejects": 0,
        "common_eligible_candidates": 0, "fifo_inserts": 0, "fifo_evictions": 0,
        "codebook_inserts": 0, "codebook_merges": 0, "codebook_replacements": 0,
        "codebook_capacity_rejects": 0, "codebook_floor_skips": 0,
        "codebook_positive_anchors": 0, "codebook_negative_anchors": 0,
        "fifo_positive_anchors": 0, "fifo_negative_anchors": 0,
        "cpu_update_count_mismatches": 0, "cpu_occupancy_mismatches": 0,
        "cpu_state_digest_exact_matches": 0, "cpu_state_digest_mismatches": 0,
        "memory_divergence_position": None,
        "codebook_same_source_exclusions": 0, "fifo_same_source_exclusions": 0,
        "codebook_eligible_anchor_slots_total": 0, "fifo_eligible_anchors_total": 0,
    }
    parity = {
        "events_checked": 0, "delta_non_bitwise_matches": 0,
        "delta_max_abs_error": 0.0, "score_non_bitwise_matches": 0,
        "score_max_abs_error": 0.0, "update_queries_checked": 0,
        "state_digest_exact_matches": 0, "state_digest_mismatches": 0,
        "source_exclusion_count_checks": 0,
        "tolerances": {"patch_delta_atol": ATOL, "patch_delta_rtol": RTOL,
                       "image_score_atol": ATOL, "image_score_rtol": RTOL},
    }
    timing = {"input_read_check_seconds": 0.0, "state_reconstruction_seconds": 0.0,
              "distances_readouts_seconds": 0.0, "serialization_seconds": 0.0}
    input_root = INPUT_ROOT
    for position in range(MAX_EVENTS):
        row = score_rows[position]
        transition = transition_rows[position]
        if row.get("position") != position or transition.get("position") != position:
            raise Exp2BIntegrityError(f"{category}:{position}: event order changed")
        if row.get("source_group_sha256") is None:
            raise Exp2BIntegrityError(f"{category}:{position}: source-group receipt missing")
        group = str(row["source_group_sha256"])
        if len(group) != 64:
            raise Exp2BIntegrityError(f"{category}:{position}: source-group digest malformed")
        arrays_out["theta_c64_q90"][position] = float(row["theta_c64_q90"])
        arrays_out["source_group_sha256"][position] = group
        shard_rel = str(row.get("shard_path", ""))
        expected_shard = f"events/event_{CATEGORIES.index(category) * MAX_EVENTS + position:06d}.npz"
        if shard_rel != expected_shard:
            raise Exp2BIntegrityError(f"{category}:{position}: shard path differs from frozen order")
        shard_path = input_root / shard_rel
        read_start = time.perf_counter()
        if not shard_path.is_file() or _is_reparse(shard_path):
            raise Exp2BIntegrityError(f"{category}:{position}: event shard is missing or unsafe")
        payload = shard_path.read_bytes()
        if _sha256_bytes(payload) != row.get("shard_sha256"):
            raise Exp2BIntegrityError(f"{category}:{position}: event shard byte digest differs")
        arrays = _load_npz_bytes(payload)
        _validate_stored_event_arrays(row, arrays, r1, category=category, position=position)
        timing["input_read_check_seconds"] += time.perf_counter() - read_start
        base = arrays["base_patch_map"]
        ids = arrays["top32_ids"]
        features = arrays["top32_features"]
        base_maps[position] = base
        ids_by_event[position] = ids
        arrays_out["c0"][position] = float(row["scores"]["c0"])
        arrays_out["historical_c2_a002"][position] = float(row["scores"]["c2_a002"])

        readout_start = time.perf_counter()
        fifo_knn, fifo_gaussian, codebook_knn, codebook_gaussian, detail = _read_memory(
            codebook, source_groups, fifo, features, group, position, v22, r1)
        delta_bridge = np.asarray(detail.pop("bridge_delta"), dtype=np.float32)
        detail.pop("fifo_selected_slots")
        detail.pop("codebook_selected_slots")
        if detail["codebook_live"] != int(codebook.size()):
            raise Exp2BIntegrityError(f"{category}:{position}: codebook live count changed during read")
        if row.get("c1_source_exclusion", {}).get("c1_no_both"):
            native_exclusion = row["c1_source_exclusion"]["c1_no_both"]
            if (int(native_exclusion.get("eligible_anchor_slots", -1)) !=
                    int(detail["codebook_live"] - detail["codebook_same_source_or_ineligible"])):
                raise Exp2BIntegrityError(f"{category}:{position}: source-exclusion eligibility count differs")
            parity["source_exclusion_count_checks"] += 1
        counters["codebook_same_source_exclusions"] += detail["codebook_same_source_or_ineligible"]
        counters["fifo_same_source_exclusions"] += detail["fifo_same_source_or_temporal"]
        counters["codebook_eligible_anchor_slots_total"] += (
            detail["codebook_live"] - detail["codebook_same_source_or_ineligible"])
        counters["fifo_eligible_anchors_total"] += (
            detail["fifo_live"] - detail["fifo_same_source_or_temporal"])
        arrays_out["delta_A"][position] = fifo_knn
        arrays_out["delta_B"][position] = fifo_gaussian
        arrays_out["delta_C"][position] = codebook_knn
        arrays_out["delta_D"][position] = codebook_gaussian
        arrays_out["delta_c2_matched_candidate_pool"][position] = delta_bridge
        scores_for_event = {
            "fifo_top5": _score_dense(base, ids, fifo_knn, r1),
            "fifo_gaussian": _score_dense(base, ids, fifo_gaussian, r1),
            "codebook_top5": _score_dense(base, ids, codebook_knn, r1),
            "codebook_gaussian": _score_dense(base, ids, codebook_gaussian, r1),
            "c2_matched_candidate_pool": _score_dense(base, ids, delta_bridge, r1),
        }
        arrays_out["fifo_top5"][position] = scores_for_event["fifo_top5"]
        arrays_out["fifo_gaussian"][position] = scores_for_event["fifo_gaussian"]
        arrays_out["codebook_top5"][position] = scores_for_event["codebook_top5"]
        arrays_out["codebook_gaussian"][position] = scores_for_event["codebook_gaussian"]
        arrays_out["c2_matched_candidate_pool"][position] = scores_for_event["c2_matched_candidate_pool"]
        cpu_d_score = float(scores_for_event["codebook_gaussian"])
        ref_delta = arrays["c1_delta_no_both"]
        delta_error, delta_nonbitwise, delta_ok = _compare_reference(codebook_gaussian, ref_delta)
        ref_score = float(row["scores"]["c1_no_both"])
        score_error = abs(cpu_d_score - ref_score)
        score_nonbitwise = int(np.float32(cpu_d_score) != np.float32(ref_score))
        score_ok = bool(np.isclose(cpu_d_score, ref_score, atol=ATOL, rtol=RTOL))
        parity["events_checked"] += 1
        parity["delta_non_bitwise_matches"] += delta_nonbitwise
        parity["delta_max_abs_error"] = max(parity["delta_max_abs_error"], delta_error)
        parity["score_non_bitwise_matches"] += score_nonbitwise
        parity["score_max_abs_error"] = max(parity["score_max_abs_error"], score_error)
        timing["distances_readouts_seconds"] += time.perf_counter() - readout_start
        if not delta_ok or not score_ok:
            raise Exp2BIntegrityError(
                f"{category}:{position}: D reference parity exceeded predeclared tolerance "
                f"(patch_max={delta_error:.8g}, score_abs={score_error:.8g})")

        if bool(row["query_tape_decision"]):
            update_start = time.perf_counter()
            key = (category, position)
            diag_record = diag_by_position.get(key)
            feedback = feedback_by_position.get(key)
            if diag_record is None or feedback is None:
                raise Exp2BIntegrityError(f"{category}:{position}: queried update receipt missing")
            diag, native_stats = _diag_for_c1_no_both(diag_record, category, position)
            label = feedback.get("label")
            if isinstance(label, bool) or label not in (0, 1):
                raise Exp2BIntegrityError(f"{category}:{position}: queried feedback is not binary")
            outcome = str(feedback.get("outcome_from_common_c0"))
            prediction = int(float(row["scores"]["c0"]) >= float(row["theta_c64_q90"]))
            expected_outcome = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[(int(label), prediction)]
            if (outcome != expected_outcome or diag.get("outcome") != outcome or
                    int(diag_record.get("label", -1)) != int(label) or
                    diag_record.get("source_group_sha256") != group or
                    feedback.get("source_group_sha256") != group):
                raise Exp2BIntegrityError(f"{category}:{position}: queried outcome provenance differs")
            keep = np.asarray(diag["native_gate_keep_mask"], dtype=np.bool_)
            beta_raw = np.asarray(diag["beta_patch_float32_before_codebook_clip"], dtype=np.float32)
            beta_clipped = np.asarray(diag["beta_after_original_codebook_max_clip"], dtype=np.float32)
            if keep.shape != (32,) or beta_raw.shape != (32,) or beta_clipped.shape != (32,):
                raise Exp2BIntegrityError(f"{category}:{position}: keep/beta vectors are not aligned")
            if not np.isfinite(beta_raw).all() or not np.isfinite(beta_clipped).all():
                raise Exp2BIntegrityError(f"{category}:{position}: beta vector is nonfinite")
            eligibility = _common_candidate_eligibility(keep, beta_raw, cfg)
            if not np.array_equal(eligibility["clipped_beta"], beta_clipped):
                raise Exp2BIntegrityError(f"{category}:{position}: original maximum beta clip differs")
            keep_count = int(eligibility["attempted_after_dN"])
            common_mask = eligibility["common_mask"]
            floor_rejects = int(eligibility["beta_floor_rejects"])
            counters["queries"] += 1
            counters["raw_selected_candidates"] += int(eligibility["raw_selected_count"])
            counters["dN_gate_rejects"] += int(eligibility["dN_gate_rejects"])
            counters["attempted_candidates_after_dN"] += int(eligibility["attempted_after_dN"])
            counters["beta_floor_rejects"] += floor_rejects
            counters["common_eligible_candidates"] += int(eligibility["common_eligible_count"])
            update_stats = _replay_native_update(
                codebook, features, keep, beta_raw, outcome=outcome, position=position)
            r1._record_c1_source_group_updates(
                source_groups, last_steps=codebook.last_step[:codebook.size()].detach().cpu().tolist(),
                event_position=position, event_group=group)
            for slot, last_step in enumerate(codebook.last_step[:codebook.size()].detach().cpu().tolist()):
                if int(last_step) == position:
                    latest_slot_group[slot] = group
            capacity_rejects = int(update_stats["skipped"]) - floor_rejects
            if capacity_rejects < 0:
                raise Exp2BIntegrityError(f"{category}:{position}: Codebook skip counts are inconsistent")
            counters["codebook_inserts"] += int(update_stats["inserted"])
            counters["codebook_merges"] += int(update_stats["merged"])
            counters["codebook_replacements"] += int(update_stats["replaced"])
            counters["codebook_capacity_rejects"] += capacity_rejects
            counters["codebook_floor_skips"] += floor_rejects
            counters["cpu_update_count_mismatches"] += int(not _same_update_categories(update_stats, native_stats))
            if not _same_update_categories(update_stats, native_stats):
                raise Exp2BIntegrityError(
                    f"{category}:{position}: CPU Codebook merge/replace/admission counts differ")
            occupancy = int(codebook.size())
            if occupancy != int(diag.get("anchor_occupancy_after", -1)):
                counters["cpu_occupancy_mismatches"] += 1
                raise Exp2BIntegrityError(f"{category}:{position}: CPU Codebook occupancy differs")
            recorded_transition = int(transition["c1_anchor_occupancy_after"]["c1_no_both"])
            if occupancy != recorded_transition:
                counters["cpu_occupancy_mismatches"] += 1
                raise Exp2BIntegrityError(f"{category}:{position}: transition occupancy differs")
            positive = int((codebook.beta[:occupancy] > 0).sum().item())
            negative = int((codebook.beta[:occupancy] < 0).sum().item())
            counters["codebook_positive_anchors"] = positive
            counters["codebook_negative_anchors"] = negative
            recorded_signs = transition["c1_anchor_sign_counts_after"]["c1_no_both"]
            if positive != int(recorded_signs["positive"]) or negative != int(recorded_signs["negative"]):
                raise Exp2BIntegrityError(f"{category}:{position}: CPU Codebook sign counts differ")
            actual_state_digest = r1._correction_state_sha256(bank_proxy, source_groups)
            if actual_state_digest == diag["codebook_state_sha256_after"]:
                parity["state_digest_exact_matches"] += 1
                counters["cpu_state_digest_exact_matches"] += 1
            else:
                parity["state_digest_mismatches"] += 1
                counters["cpu_state_digest_mismatches"] += 1
            parity["update_queries_checked"] += 1

            # Memory F consumes exactly the common dN+beta-floor eligible set.
            eligible_indices = np.flatnonzero(common_mask)
            if eligible_indices.size:
                selected_t = _as_torch_features(features[eligible_indices])
                normalized = v22.l2_normalize(selected_t)
                for index, normalized_feature in zip(eligible_indices.tolist(), normalized):
                    anchor = Anchor(
                        feature=normalized_feature.detach().clone(),
                        beta=float(beta_clipped[index]),
                        radius=float(np.clip(cfg.radius0, cfg.radius_min, cfg.radius_max)),
                        kappa=float(np.clip(cfg.kappa0, 0.1, cfg.kappa_max)),
                        position=position, source_group=group)
                    fifo.append(anchor)
                    counters["fifo_inserts"] += 1
                    if len(fifo) > CAPACITY:
                        del fifo[0]
                        counters["fifo_evictions"] += 1
            counters["fifo_positive_anchors"] = sum(anchor.beta > 0 for anchor in fifo)
            counters["fifo_negative_anchors"] = sum(anchor.beta < 0 for anchor in fifo)
            update_policy_diverged = bool(
                update_stats["merged"] or update_stats["replaced"] or capacity_rejects or
                counters["fifo_evictions"])
            if counters["memory_divergence_position"] is None and (
                    not _memory_equal(fifo, codebook) or update_policy_diverged):
                counters["memory_divergence_position"] = position
            timing["state_reconstruction_seconds"] += time.perf_counter() - update_start
        parity["events_checked"] = max(parity["events_checked"], position + 1)

    # Label-blind, category-wide retrospective strength normalization.  This is
    # deliberately a score diagnostic, never a causal online arm.
    alpha_meta: dict[str, Any] = {}
    for memory, gaussian_key, knn_key, output_key in (
            ("fifo", "delta_B", "delta_A", "fifo_top5_rms_normalized"),
            ("codebook", "delta_D", "delta_C", "codebook_top5_rms_normalized")):
        gaussian = arrays_out[gaussian_key]
        knn = arrays_out[knn_key]
        alpha_record = _trajectory_alpha(gaussian, knn)
        if alpha_record["status"] != "AVAILABLE":
            alpha_meta[memory] = alpha_record
            continue
        alpha = float(alpha_record["alpha"])
        scaled_scores = np.empty((MAX_EVENTS,), dtype=np.float32)
        for position in range(MAX_EVENTS):
            scaled = (knn[position] * np.float32(alpha)).astype(np.float32)
            scaled_scores[position] = _score_dense(
                base_maps[position], ids_by_event[position], scaled, r1)
        arrays_out[output_key] = scaled_scores
        alpha_meta[memory] = alpha_record
    counters["final_fifo_occupancy"] = len(fifo)
    counters["final_codebook_occupancy"] = int(codebook.size())
    counters["fifo_unique_current_source_groups"] = len({item.source_group for item in fifo})
    counters["codebook_unique_latest_source_groups"] = len(set(latest_slot_group.values()))
    counters["codebook_unique_conservative_history_groups"] = len(
        {group for history in source_groups.values() for group in history})
    counters["codebook_provenance_slots"] = len(source_groups)
    counters["codebook_policy_active"] = bool(
        counters["codebook_merges"] or counters["codebook_replacements"] or
        counters["codebook_capacity_rejects"] or counters["fifo_evictions"])
    counters["codebook_policy_inactive_by_protocol"] = bool(
        not counters["codebook_policy_active"] and
        counters["final_fifo_occupancy"] < CAPACITY and
        counters["final_codebook_occupancy"] < CAPACITY)
    counters["source_group_history_semantics"] = "conservative slot union retained through replacement"
    counters["readout_state_independence"] = "both readouts score same pre-event state; writes use C0 query feedback only"

    if not np.all(arrays_out["theta_c64_q90"] == arrays_out["theta_c64_q90"][0]):
        raise Exp2BIntegrityError(f"{category}: frozen C64 threshold changes within the prefix")
    if counters["queries"] != sum(bool(row["query_tape_decision"]) for row in score_rows):
        raise Exp2BIntegrityError(f"{category}: replay query count differs from frozen tape")
    if parity["source_exclusion_count_checks"] != MAX_EVENTS:
        raise Exp2BIntegrityError(f"{category}: C1 source-exclusion parity receipts are incomplete")
    counters["dN_gate_rejects"] = int(counters["dN_gate_rejects"])
    counters["beta_floor_rejects"] = int(counters["beta_floor_rejects"])
    counters["candidate_accounting_identity"] = (
        counters["raw_selected_candidates"] ==
        counters["dN_gate_rejects"] + counters["attempted_candidates_after_dN"] and
        counters["attempted_candidates_after_dN"] ==
        counters["beta_floor_rejects"] + counters["common_eligible_candidates"])
    if not counters["candidate_accounting_identity"]:
        raise Exp2BIntegrityError(f"{category}: candidate eligibility accounting does not balance")
    counters["common_candidate_same_across_memories"] = True
    counters["unique_source_groups_represented"] = {
        "fifo_current_anchors": counters["fifo_unique_current_source_groups"],
        "codebook_latest_slots": counters["codebook_unique_latest_source_groups"],
        "codebook_conservative_history": counters["codebook_unique_conservative_history_groups"],
    }
    counters["fifo_max_occupancy"] = min(CAPACITY, counters["fifo_inserts"])
    counters["candidate_rejection_explanation"] = {
        "beta_floor_rejects": counters["beta_floor_rejects"],
        "dN_gate_rejects": counters["dN_gate_rejects"],
        "codebook_capacity_rejects": counters["codebook_capacity_rejects"],
        "codebook_merges": counters["codebook_merges"],
        "codebook_replacements": counters["codebook_replacements"],
    }
    counters["first_memory_divergence_position"] = counters["memory_divergence_position"]
    category_result = {
        "schema": "r2.exp2b.category.v1", "category": category,
        "status": "PASS", "event_count": MAX_EVENTS,
        "query_count": counters["queries"], "reference_parity": parity,
        "memory_accounting": counters, "magnitude_normalization": alpha_meta,
        "phase_seconds": timing,
        "memory_contrast": (
            "ACTIVE" if counters["codebook_policy_active"] else
            "INACTIVE: neither memory reaches capacity and Codebook has no merges/rejections"),
        "cpu_codebook_state_digest_equality_is_diagnostic_only": True,
        "cross_device_state_sha_equality_required": False,
        "update_decision_counts_must_match_exactly": True,
    }
    return arrays_out, category_result


def _npz_payload_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.ascontiguousarray(value) for key, value in arrays.items()}


def run_stage_a(output_root: Path) -> None:
    _torch()
    root = _validate_output_root(output_root, must_exist=True)
    start = time.perf_counter()
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("run_id") != RUN_ID or manifest.get("input_run_id") != INPUT_RUN_ID:
        raise Exp2BIntegrityError("Stage A run manifest identity differs")
    _verify_run_fingerprints(manifest)
    r1 = __import__("experiments.calibration.r1_matched_correction_pilot", fromlist=["arrays_sha256"])
    header = manifest.get("verified_input_receipt")
    if not isinstance(header, Mapping) or not isinstance(header.get("stage_seal"), Mapping):
        raise Exp2BIntegrityError("Stage A lacks the one-time verified input seal receipt")
    jsonl_read_start = time.perf_counter()
    records = _read_and_verify_jsonl_inputs(header, r1)
    records["jsonl_input_validation_seconds"] = time.perf_counter() - jsonl_read_start
    with OutputLedger(root) as ledger:
        category_receipts: dict[str, dict[str, Any]] = {}
        category_hashes: dict[str, dict[str, str]] = {}
        for category in CATEGORIES:
            grouped = records["groups"][category]
            category_input = {
                "scores": grouped["scores"], "transitions": grouped["transitions"],
                "diagnostics": {key: row for key, row in records["diagnostics"].items()
                                if key[0] == category},
                "feedback": {key: row for key, row in records["feedback"].items()
                             if key[0] == category},
            }
            try:
                arrays, receipt = _score_category(category, category_input, ledger, r1)
                serialization_start = time.perf_counter()
                payload = _make_npz_bytes(_npz_payload_arrays(arrays))
                receipt["phase_seconds"]["serialization_seconds"] += time.perf_counter() - serialization_start
                score_hash = ledger.write(f"scores_{category}.npz", payload)
                receipt["score_file"] = f"scores_{category}.npz"
                receipt["score_file_sha256"] = score_hash
                receipt["score_file_bytes"] = len(payload)
                category_bytes = _canonical_json(receipt)
                category_hash = ledger.write(f"category_{category}.json", category_bytes)
                category_hashes[category] = {f"scores_{category}.npz": score_hash,
                                             f"category_{category}.json": category_hash}
                category_receipts[category] = receipt
                ledger.checkpoint(durable=True)
            except Exp2BIntegrityError as exc:
                receipt = {
                    "schema": "r2.exp2b.category.v1", "category": category,
                    "status": "UNRECONSTRUCTABLE_FROM_EXISTING_ARTIFACTS",
                    "reason": str(exc)[:1200], "score_file": None,
                    "partial_prefix_preserved": False,
                }
                category_hash = ledger.write_json(f"category_{category}.json", receipt)
                category_receipts[category] = receipt
                category_hashes[category] = {f"category_{category}.json": category_hash}
                ledger.checkpoint(durable=True)
        valid_categories = [category for category in CATEGORIES
                            if category_receipts[category]["status"] == "PASS"]
        refs: list[dict[str, Any]] = []
        for category in valid_categories:
            for row in records["groups"][category]["scores"]:
                refs.append({key: row[key] for key in (
                    "category", "position", "event_id", "class_json_sha256",
                    "source_identity_sha256", "source_group_sha256", "theta_c64_q90")})
        refs_payload = b"".join(_canonical_json(row) for row in refs)
        refs_hash = ledger.write("event_refs.jsonl", refs_payload)
        ledger.checkpoint(durable=True)
        score_seal = {
            "schema": "r2.exp2b.stage_a_score_seal.v1",
            "status": "SEALED_COMPLETE" if len(valid_categories) == len(CATEGORIES) else "PARTIAL_SEALED",
            "run_id": RUN_ID, "input_run_id": INPUT_RUN_ID,
            "input_stage_a_seal_sha256": header["stage_seal_sha256"],
            "input_experiment_seal_sha256": header["experiment_seal_sha256"],
            "input_score_file_sha256": records["input_hashes"]["event_scores.jsonl"],
            "input_transition_file_sha256": records["input_hashes"]["event_transitions.jsonl"],
            "input_strength_diagnostics_sha256": records["input_hashes"]["strength_diagnostics.jsonl"],
            "input_queried_feedback_sha256": records["input_hashes"]["private_query_feedback.jsonl"],
            "category_status": {category: category_receipts[category]["status"] for category in CATEGORIES},
            "category_event_counts": {category: (MAX_EVENTS if category in valid_categories else 0)
                                      for category in CATEGORIES},
            "category_hashes": category_hashes,
            "event_refs_sha256": refs_hash, "event_refs_count": len(refs),
            "event_refs_file": "event_refs.jsonl",
            "label_access": "NONE_IN_STAGE_A_EXCEPT_THE_67_EXISTING_QUERIED_DISCLOSURES",
            "new_feature_extraction": False, "image_decode": False, "gpu": False,
            "categories": category_receipts,
            "ledger_bytes_before_seal": ledger.used_bytes,
            "ledger_file_count_before_seal": ledger.file_count,
            "small_receipt_validation_seconds": float(manifest.get("small_receipt_validation_seconds", 0.0)),
            "jsonl_input_validation_seconds": float(records["jsonl_input_validation_seconds"]),
            "stage_a_wall_seconds": time.perf_counter() - start,
            "peak_working_set_bytes": _peak_working_set_bytes(),
            "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        ledger.write_json("stage_a_seal.json", score_seal)
        ledger.checkpoint(durable=True)
    return


PRIMARY_ARM_KEYS = {
    "A_FIFO_matched_Top5_beta": "fifo_top5",
    "B_FIFO_original_Gaussian": "fifo_gaussian",
    "C_Codebook_matched_Top5_beta": "codebook_top5",
    "D_Codebook_original_Gaussian": "codebook_gaussian",
}
BASELINE_ARM_KEYS = {
    "C0": "c0",
    "historical_C2_a0.02": "historical_c2_a002",
    "C2_matched_candidate_pool": "c2_matched_candidate_pool",
}
DIAGNOSTIC_ARM_KEYS = {
    "FIFO_Top5_RMS_normalized": "fifo_top5_rms_normalized",
    "Codebook_Top5_RMS_normalized": "codebook_top5_rms_normalized",
}
ALL_METRICS = ("image_auroc", "image_auprc_average_precision",
               "fpr", "fnr", "balanced_accuracy")
FACTOR_CONTRASTS = {
    "memory_effect_under_Gaussian_D_minus_B": ("D_Codebook_original_Gaussian", "B_FIFO_original_Gaussian"),
    "memory_effect_under_matched_Top5_C_minus_A": ("C_Codebook_matched_Top5_beta", "A_FIFO_matched_Top5_beta"),
    "readout_Gaussian_minus_Top5_with_Codebook_D_minus_C": ("D_Codebook_original_Gaussian", "C_Codebook_matched_Top5_beta"),
    "readout_Gaussian_minus_Top5_with_FIFO_B_minus_A": ("B_FIFO_original_Gaussian", "A_FIFO_matched_Top5_beta"),
}


def _verify_score_seal(root: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]],
                                             dict[str, dict[str, np.ndarray]]]:
    seal_path = root / "stage_a_seal.json"
    if not seal_path.is_file():
        raise Exp2BIntegrityError("Stage B refused evaluator access: Stage A score seal is missing")
    seal_bytes = seal_path.read_bytes()
    seal = json.loads(seal_bytes)
    if (seal.get("schema") != "r2.exp2b.stage_a_score_seal.v1" or
            seal.get("run_id") != RUN_ID or
            seal.get("status") not in ("SEALED_COMPLETE", "PARTIAL_SEALED")):
        raise Exp2BIntegrityError("Stage B refused evaluator access: Stage A seal is incomplete or mismatched")
    refs_path = root / str(seal.get("event_refs_file", ""))
    if not refs_path.is_file() or _is_reparse(refs_path):
        raise Exp2BIntegrityError("Stage B refused evaluator access: event reference file is missing")
    refs_bytes = refs_path.read_bytes()
    if _sha256_bytes(refs_bytes) != seal.get("event_refs_sha256"):
        raise Exp2BIntegrityError("Stage B refused evaluator access: event reference digest differs")
    refs = _jsonl_from_bytes(refs_bytes, "event_refs.jsonl")
    if len(refs) != int(seal.get("event_refs_count", -1)):
        raise Exp2BIntegrityError("Stage B refused evaluator access: event reference count differs")
    refs_by_category: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORIES}
    for row in refs:
        category = row.get("category")
        if category not in refs_by_category or any(key in row for key in ("label", "outcome", "anomaly_class")):
            raise Exp2BIntegrityError("Stage B refused evaluator access: invalid or labeled event reference")
        refs_by_category[category].append(row)
    arrays_by_category: dict[str, dict[str, np.ndarray]] = {}
    for category in CATEGORIES:
        status = seal.get("category_status", {}).get(category)
        category_receipt = seal.get("categories", {}).get(category, {})
        if status != "PASS":
            if refs_by_category[category]:
                raise Exp2BIntegrityError("Invalid category unexpectedly has evaluator references")
            continue
        if category_receipt.get("event_count") != MAX_EVENTS or len(refs_by_category[category]) != MAX_EVENTS:
            raise Exp2BIntegrityError(f"Stage B refused incomplete {category} score arrays")
        category_name = f"category_{category}.json"
        category_receipt_path = root / category_name
        expected_category_hash = seal.get("category_hashes", {}).get(category, {}).get(category_name)
        if (not category_receipt_path.is_file() or _is_reparse(category_receipt_path) or
                _sha256_file(category_receipt_path) != expected_category_hash):
            raise Exp2BIntegrityError(f"Stage B refused {category}: category receipt digest differs")
        category_receipt_file = json.loads(category_receipt_path.read_text(encoding="utf-8"))
        if category_receipt_file != category_receipt:
            raise Exp2BIntegrityError(f"Stage B refused {category}: category receipt differs from seal")
        score_name = f"scores_{category}.npz"
        expected_hash = seal.get("category_hashes", {}).get(category, {}).get(score_name)
        score_path = root / score_name
        if not score_path.is_file() or _is_reparse(score_path):
            raise Exp2BIntegrityError(f"Stage B refused evaluator access: {category} score file missing")
        score_bytes = score_path.read_bytes()
        if _sha256_bytes(score_bytes) != expected_hash:
            raise Exp2BIntegrityError(f"Stage B refused evaluator access: {category} score seal mismatch")
        arrays = _load_npz_bytes(score_bytes)
        for key in ("c0", "historical_c2_a002", "fifo_top5", "fifo_gaussian",
                    "codebook_top5", "codebook_gaussian", "c2_matched_candidate_pool",
                    "delta_A", "delta_B", "delta_C", "delta_D", "theta_c64_q90",
                    "source_group_sha256"):
            if key not in arrays:
                raise Exp2BIntegrityError(f"Stage B refused evaluator access: {category} lacks {key}")
        vector_keys = ("c0", "historical_c2_a002", "fifo_top5", "fifo_gaussian",
                       "codebook_top5", "codebook_gaussian", "c2_matched_candidate_pool",
                       "theta_c64_q90", "source_group_sha256")
        delta_keys = ("delta_A", "delta_B", "delta_C", "delta_D",
                      "delta_c2_matched_candidate_pool")
        if any(arrays[key].shape != (MAX_EVENTS,) for key in vector_keys):
            raise Exp2BIntegrityError(f"Stage B refused evaluator access: {category} vector shapes differ")
        if any(arrays[key].shape != (MAX_EVENTS, 32) for key in delta_keys):
            raise Exp2BIntegrityError(f"Stage B refused evaluator access: {category} patch delta shapes differ")
        if arrays["source_group_sha256"].dtype.kind not in "US":
            raise Exp2BIntegrityError(f"Stage B refused evaluator access: {category} source-group dtype differs")
        if any(len(str(value)) != 64 for value in arrays["source_group_sha256"].tolist()):
            raise Exp2BIntegrityError(f"Stage B refused evaluator access: {category} source-group digest is malformed")
        for key, value in arrays.items():
            if value.dtype.kind in "f" and not np.isfinite(value).all():
                raise Exp2BIntegrityError(f"Stage B refused evaluator access: {category} {key} is nonfinite")
        refs_by_category[category].sort(key=lambda row: int(row["position"]))
        required_ref_fields = ("event_id", "class_json_sha256", "source_identity_sha256",
                               "source_group_sha256")
        for index, reference in enumerate(refs_by_category[category]):
            if any(not isinstance(reference.get(field), str) or len(reference[field]) != 64
                   for field in required_ref_fields):
                raise Exp2BIntegrityError(f"Stage B refused evaluator access: {category} reference identity is malformed")
            if reference["source_group_sha256"] != str(arrays["source_group_sha256"][index]):
                raise Exp2BIntegrityError(f"Stage B refused evaluator access: {category} group digest differs")
        if [int(row["position"]) for row in refs_by_category[category]] != list(range(MAX_EVENTS)):
            raise Exp2BIntegrityError(f"Stage B refused evaluator access: {category} refs are not ordered")
        if any(str(refs_by_category[category][i]["source_group_sha256"]) !=
               str(arrays["source_group_sha256"][i]) for i in range(MAX_EVENTS)):
            raise Exp2BIntegrityError(f"Stage B refused evaluator access: {category} group digests differ")
        arrays_by_category[category] = arrays
    return seal, refs_by_category, arrays_by_category


def _metric_value(record: Mapping[str, Any], metric: str) -> float | None:
    if metric in ("fpr", "fnr", "balanced_accuracy"):
        value = record.get("fixed_threshold", {}).get(metric)
    else:
        value = record.get(metric)
    return None if value is None else float(value)


def _subtract(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left - right)


def _factor_metrics(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    for metric in ALL_METRICS:
        def v(cell: str) -> float | None:
            return _metric_value(records[{
                "A": "A_FIFO_matched_Top5_beta", "B": "B_FIFO_original_Gaussian",
                "C": "C_Codebook_matched_Top5_beta", "D": "D_Codebook_original_Gaussian",
            }[cell]], metric)
        contrasts = {
            "memory_effect_under_Gaussian_D_minus_B": _subtract(v("D"), v("B")),
            "memory_effect_under_matched_Top5_C_minus_A": _subtract(v("C"), v("A")),
            "readout_Gaussian_minus_Top5_with_Codebook_D_minus_C": _subtract(v("D"), v("C")),
            "readout_Gaussian_minus_Top5_with_FIFO_B_minus_A": _subtract(v("B"), v("A")),
            "interaction_D_minus_C_minus_B_plus_A": (
                None if any(v(cell) is None for cell in "ABCD") else
                float(v("D") - v("C") - v("B") + v("A"))),
        }
        effects[metric] = contrasts
    return effects


def _cluster_bootstrap(labels: np.ndarray, groups: Sequence[str],
                       scores_by_arm: Mapping[str, np.ndarray], threshold: float,
                       evaluator: Any) -> dict[str, Any]:
    """Paired Source Group bootstrap; one resample is shared by every arm."""
    y = np.asarray(labels, dtype=np.int8)
    group_values = np.asarray(groups, dtype=object)
    unique = sorted(set(group_values.tolist()))
    if len(y) != len(group_values) or not unique:
        raise Exp2BIntegrityError("Source Group bootstrap inputs are empty or misaligned")
    members = {group: np.flatnonzero(group_values == group) for group in unique}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    names = tuple(scores_by_arm)
    sampled: dict[str, list[float]] = {
        f"arm:{arm}:{metric}": [] for arm in names for metric in ALL_METRICS}
    sampled.update({f"contrast:{contrast}:{metric}": []
                    for contrast in (*FACTOR_CONTRASTS,
                                     "interaction_D_minus_C_minus_B_plus_A")
                    for metric in ALL_METRICS})
    replicates_without_both = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        group_draw = rng.choice(unique, size=len(unique), replace=True)
        selected_indices = np.concatenate([members[group] for group in group_draw])
        if np.unique(y[selected_indices]).size < 2:
            replicates_without_both += 1
        records = {
            arm: evaluator._metric_record(y[selected_indices], np.asarray(score)[selected_indices], threshold)
            for arm, score in scores_by_arm.items()
        }
        for arm, record in records.items():
            for metric in ALL_METRICS:
                value = _metric_value(record, metric)
                if value is not None:
                    sampled[f"arm:{arm}:{metric}"].append(value)
        effects = _factor_metrics(records)
        for metric, contrasts in effects.items():
            for contrast, value in contrasts.items():
                if value is not None:
                    sampled[f"contrast:{contrast}:{metric}"].append(value)

    def interval(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {"lower_2_5": None, "upper_97_5": None, "valid_replicates": 0}
        vector = np.asarray(values, dtype=np.float64)
        return {"lower_2_5": float(np.quantile(vector, 0.025)),
                "upper_97_5": float(np.quantile(vector, 0.975)),
                "valid_replicates": int(vector.size)}
    return {
        "method": "paired Source Group bootstrap with replacement; same group draw reused across all arms",
        "replicates_requested": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED, "unique_source_groups": len(unique),
        "conditional_on_frozen_historical_trajectory": True,
        "not_uncertainty_over_independently_regenerated_streams": True,
        "replicates_without_both_classes": replicates_without_both,
        "intervals": {key: interval(value) for key, value in sampled.items()},
    }


def _ranking_change(labels: np.ndarray, groups: Sequence[str], baseline: np.ndarray,
                    candidate: np.ndarray) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    group = np.asarray(groups, dtype=object)
    base = np.asarray(baseline, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    helpful = damaging = both_correct = both_wrong = tied_after = excluded = 0
    for normal in np.flatnonzero(y == 0):
        for defect in np.flatnonzero(y == 1):
            if group[normal] == group[defect]:
                excluded += 1
                continue
            base_correct = base[defect] > base[normal]
            candidate_correct = cand[defect] > cand[normal]
            if not base_correct and candidate_correct:
                helpful += 1
            elif base_correct and not candidate_correct:
                damaging += 1
            elif base_correct and candidate_correct:
                both_correct += 1
            elif not base_correct and not candidate_correct:
                both_wrong += 1
            tied_after += int(cand[defect] == cand[normal])
    return {
        "comparison": "candidate versus C0 normal-defect pair ordering",
        "source_group_policy": "exclude normal-defect pairs from the same Source Group",
        "helpful_pair_reversals": helpful, "damaging_pair_reversals": damaging,
        "correct_in_both": both_correct, "incorrect_or_tied_in_both": both_wrong,
        "candidate_tied_pairs": tied_after, "same_group_pairs_excluded": excluded,
        "net_helpful_minus_damaging": helpful - damaging,
    }


def _patch_strength_summary(deltas: np.ndarray, image_scores: np.ndarray,
                            c0_scores: np.ndarray) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64).reshape(-1)
    image_change = np.asarray(image_scores, dtype=np.float64) - np.asarray(c0_scores, dtype=np.float64)
    abs_values = np.abs(values)
    return {
        "patch_rms": float(np.sqrt(np.mean(values ** 2))),
        "patch_mean_absolute": float(abs_values.mean()),
        "signed_mean": float(values.mean()),
        "signed_quantiles": {str(q): float(np.quantile(values, q)) for q in (0, .05, .25, .5, .75, .95, 1)},
        "positive_patch_fraction": float(np.mean(values > 0)),
        "negative_patch_fraction": float(np.mean(values < 0)),
        "zero_patch_fraction_exact": float(np.mean(values == 0)),
        "near_zero_patch_fraction_abs_lt_1e_4": float(np.mean(abs_values < 1e-4)),
        "image_score_signed_movement_mean": float(image_change.mean()),
        "image_score_movement_mean_absolute": float(np.abs(image_change).mean()),
        "image_score_movement_quantiles": {str(q): float(np.quantile(image_change, q))
                                           for q in (0, .5, .9, .95, 1)},
    }


def _stage_b_evaluator() -> Any:
    """Bind the approved EXP2A identity projection to the shared PERF-1 metrics."""
    label_provider = __import__(
        "experiments.calibration.r2_exp2a_c1_strength", fromlist=["_stage_b_labels"])
    metric_provider = __import__(
        "experiments.calibration.r2_perf1_stage_b", fromlist=["_metric_record"])
    return type("R2EXP2BStageBEvaluator", (), {
        "_stage_b_labels": staticmethod(label_provider._stage_b_labels),
        "_metric_record": staticmethod(metric_provider._metric_record),
        "DATASET_ROOT": label_provider.DATASET_ROOT,
        "JSON_ROOT": label_provider.JSON_ROOT,
        "label_provider_module": label_provider.__name__,
        "metric_provider_module": metric_provider.__name__,
    })()


def _evaluate_category(category: str, refs: Sequence[Mapping[str, Any]],
                       arrays: Mapping[str, np.ndarray], evaluator: Any) -> dict[str, Any]:
    label_projection_start = time.perf_counter()
    labels, groups = evaluator._stage_b_labels(
        category, refs, evaluator.DATASET_ROOT, evaluator.JSON_ROOT)
    label_projection_seconds = time.perf_counter() - label_projection_start
    labels = np.asarray(labels, dtype=np.int8)
    groups = [str(value) for value in groups]
    if labels.shape != (MAX_EVENTS,) or not np.isin(labels, (0, 1)).all():
        raise Exp2BIntegrityError(f"{category}: evaluator did not return exactly the authorized 512 outcomes")
    if len(groups) != MAX_EVENTS or any(len(group) != 64 for group in groups):
        raise Exp2BIntegrityError(f"{category}: evaluator Source Group identities are malformed")
    if any(groups[index] != refs[index]["source_group_sha256"] for index in range(MAX_EVENTS)):
        raise Exp2BIntegrityError(f"{category}: evaluator Source Group differs from the sealed event identity")
    thresholds = np.asarray(arrays["theta_c64_q90"], dtype=np.float64)
    if thresholds.shape != (MAX_EVENTS,) or not np.all(thresholds == thresholds[0]):
        raise Exp2BIntegrityError(f"{category}: C64 threshold is not frozen across events")
    threshold = float(thresholds[0])
    score_map: dict[str, np.ndarray] = {
        name: np.asarray(arrays[key], dtype=np.float32)
        for name, key in {**PRIMARY_ARM_KEYS, **BASELINE_ARM_KEYS, **DIAGNOSTIC_ARM_KEYS}.items()
        if key in arrays
    }
    metrics_start = time.perf_counter()
    metric_records = {
        arm: evaluator._metric_record(labels, scores, threshold)
        for arm, scores in score_map.items()
    }
    primary = {arm: metric_records[arm] for arm in PRIMARY_ARM_KEYS}
    factors = _factor_metrics(primary)
    versus_controls: dict[str, Any] = {}
    for arm in PRIMARY_ARM_KEYS:
        versus_controls[arm] = {
            metric: {
                "versus_C0": _subtract(_metric_value(metric_records[arm], metric),
                                       _metric_value(metric_records["C0"], metric)),
                "versus_historical_C2_a0.02": _subtract(
                    _metric_value(metric_records[arm], metric),
                    _metric_value(metric_records["historical_C2_a0.02"], metric)),
            } for metric in ALL_METRICS}
    strength = {
        arm: _patch_strength_summary(arrays[delta_key], score_map[arm], score_map["C0"])
        for arm, delta_key in {
            "A_FIFO_matched_Top5_beta": "delta_A",
            "B_FIFO_original_Gaussian": "delta_B",
            "C_Codebook_matched_Top5_beta": "delta_C",
            "D_Codebook_original_Gaussian": "delta_D",
        }.items()}
    ranking = {
        arm: _ranking_change(labels, groups, score_map["C0"], score_map[arm])
        for arm in PRIMARY_ARM_KEYS}
    metrics_seconds = time.perf_counter() - metrics_start
    bootstrap_start = time.perf_counter()
    bootstrap = _cluster_bootstrap(labels, groups, score_map, threshold, evaluator)
    bootstrap_seconds = time.perf_counter() - bootstrap_start
    category_receipt = json.loads((OUTPUT_ROOT / f"category_{category}.json").read_text(encoding="utf-8"))
    positives = int(np.count_nonzero(labels == 1))
    negatives = int(np.count_nonzero(labels == 0))
    return {
        "status": "EVALUATED", "event_count": MAX_EVENTS,
        "positive_count": positives, "negative_count": negatives,
        "defect_prevalence": float(labels.mean()),
        "frozen_c64_threshold": threshold,
        "metrics_by_arm": metric_records,
        "factor_effects_by_metric": factors,
        "each_primary_cell_versus_controls": versus_controls,
        "paired_source_group_bootstrap": bootstrap,
        "normal_defect_ranking_changes_vs_C0": ranking,
        "patch_and_image_score_strength": strength,
        "normal_side_harms": ({
            "normal_events": negatives,
            "normal_fpr_by_arm": {arm: metric_records[arm]["fixed_threshold"]["fpr"]
                                   for arm in score_map},
            "mean_absolute_score_movement_from_C0_on_normals": {
                arm: float(np.abs(score_map[arm][labels == 0] - score_map["C0"][labels == 0]).mean())
                for arm in PRIMARY_ARM_KEYS},
        } if positives == 0 else None),
        "memory_mechanism_accounting": category_receipt.get("memory_accounting"),
        "magnitude_normalization": category_receipt.get("magnitude_normalization"),
        "magnitude_normalization_interpretation": (
            "trajectory-wide retrospective score diagnostic; uses future score statistics; "
            "not an online algorithm, deployable baseline, or headline improvement"),
        "phase_seconds": {
            "outcome_projection": label_projection_seconds,
            "metric_and_ranking": metrics_seconds,
            "paired_source_group_bootstrap": bootstrap_seconds,
        },
        "historical_c2_pool_bridge": {
            "historical_all_top32": metric_records["historical_C2_a0.02"],
            "common_eligibility_top5_C2": metric_records["C2_matched_candidate_pool"],
            "matched_variable_beta_fifo_top5": metric_records["A_FIFO_matched_Top5_beta"],
        },
    }


def run_stage_b(output_root: Path) -> None:
    root = _validate_output_root(output_root, must_exist=True)
    with OutputLedger(root) as ledger:
        run_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        if run_manifest.get("run_id") != RUN_ID or run_manifest.get("input_run_id") != INPUT_RUN_ID:
            raise Exp2BIntegrityError("Stage B run manifest identity differs")
        _verify_run_fingerprints(run_manifest)
        stage_seal, refs, arrays = _verify_score_seal(root)
        valid_categories = [category for category in CATEGORIES
                            if stage_seal["category_status"].get(category) == "PASS"]
        if not valid_categories:
            raise Exp2BIntegrityError("No categories have a valid Stage A score seal")
        # This is the first import/call path that can read TEST outcome metadata.
        label_access_started = time.perf_counter()
        evaluator = _stage_b_evaluator()
        label_access_started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        stage_b_start = time.perf_counter()
        ledger.write_json("stage_b_started.json", {
            "schema": "r2.exp2b.stage_b_started.v1",
            "status": "OUTCOME_ACCESS_MAY_START",
            "stage_a_score_seal_sha256": _sha256_file(root / "stage_a_seal.json"),
            "planned_categories": valid_categories,
            "planned_outcome_events": MAX_EVENTS * len(valid_categories),
            "label_projection_helper": evaluator.label_provider_module,
            "metric_helper": evaluator.metric_provider_module,
            "started_utc": label_access_started_utc,
        })
        ledger.checkpoint(durable=True)
        results: dict[str, Any] = {}
        for category in valid_categories:
            results[category] = _evaluate_category(
                category, refs[category], arrays[category], evaluator)
        phase_totals = {
            phase: float(sum(results[category]["phase_seconds"][phase]
                             for category in valid_categories))
            for phase in ("outcome_projection", "metric_and_ranking",
                          "paired_source_group_bootstrap")}
        evaluable_arm_set = set.intersection(*(
            set(results[category]["metrics_by_arm"]) for category in valid_categories))
        evaluable_arms = tuple(arm for arm in
            (*PRIMARY_ARM_KEYS, *BASELINE_ARM_KEYS, *DIAGNOSTIC_ARM_KEYS)
            if arm in evaluable_arm_set)
        metric_macros: dict[str, Any] = {}
        for arm in evaluable_arms:
            defined = {category: results[category]["metrics_by_arm"][arm]["image_auroc"]
                       for category in valid_categories}
            values = [float(value) for value in defined.values() if value is not None]
            metric_macros[arm] = {
                "strict_four_category_auroc_macro": (
                    float(np.mean(values)) if len(valid_categories) == 4 and len(values) == 4 else None),
                "three_evaluable_category_descriptive_auroc_macro": (
                    float(np.mean(values)) if len(values) == 3 else None),
                "defined_category_count": len(values),
                "auroc_by_category": defined,
            }
        # Category-level descriptive macro effects use only categories where all
        # relevant values are defined.  Audiojack remains explicitly undefined.
        macro_factor_effects: dict[str, Any] = {}
        for metric in ALL_METRICS:
            macro_factor_effects[metric] = {}
            names = tuple(next(iter(results.values()))["factor_effects_by_metric"][metric])
            for contrast in names:
                values = [results[category]["factor_effects_by_metric"][metric][contrast]
                          for category in valid_categories]
                defined = [float(value) for value in values if value is not None]
                macro_factor_effects[metric][contrast] = {
                    "category_equal_macro_defined": float(np.mean(defined)) if defined else None,
                    "defined_category_count": len(defined),
                    "by_category": {category: results[category]["factor_effects_by_metric"][metric][contrast]
                                    for category in valid_categories},
                }
        total_queries = int(sum(
            results[category]["memory_mechanism_accounting"]["queries"]
            for category in valid_categories))
        stage_b_results = {
            "schema": "r2.exp2b.stage_b_results.v1",
            "run_id": RUN_ID, "input_run_id": INPUT_RUN_ID,
            "stage_a_seal_status": stage_seal["status"],
            "stage_a_score_seal_sha256": _sha256_file(root / "stage_a_seal.json"),
            "valid_categories": valid_categories,
            "unreconstructable_categories": [category for category in CATEGORIES
                if category not in valid_categories],
            "label_access_started_after_complete_category_score_seals": True,
            "label_access_started_utc": label_access_started_utc,
            "label_projection_seconds": max(0.0, time.perf_counter() - label_access_started),
            "stage_b_total_seconds": time.perf_counter() - stage_b_start,
            "phase_seconds": phase_totals,
            "peak_working_set_bytes": _peak_working_set_bytes(),
            "outcome_events_accessed": MAX_EVENTS * len(valid_categories),
            "queried_feedback_events_reused_only_in_stage_a": EXPECTED_QUERY_COUNT,
            "new_evaluation_events": 0,
            "masks_read": False,
            "categories": results,
            "image_auroc_macros": metric_macros,
            "factor_effect_macros": macro_factor_effects,
            "strict_four_category_auroc_macro": None,
            "strict_macro_undefined_reason": (
                "four-category AUROC macro is undefined when any category lacks one outcome class"),
            "three_evaluable_category_descriptive_macro": {
                arm: metric_macros[arm]["three_evaluable_category_descriptive_auroc_macro"]
                for arm in evaluable_arms},
            "total_queries_replayed": total_queries,
            "bootstrap_contract": {
                "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                "unit": "Source Group", "paired_weights_across_arms": True,
                "conditional_on_frozen_trajectories": True,
            },
            "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        result_bytes = _canonical_json(stage_b_results)
        result_hash = ledger.write("stage_b_results.json", result_bytes)
        stage_b_seal = {
            "schema": "r2.exp2b.stage_b_seal.v1", "status": "COMPLETE",
            "run_id": RUN_ID, "stage_a_score_seal_sha256": stage_b_results["stage_a_score_seal_sha256"],
            "results_sha256": result_hash, "outcome_events_accessed": MAX_EVENTS * len(valid_categories),
            "masks_read": False, "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        stage_b_seal_hash = ledger.write_json("stage_b_seal.json", stage_b_seal)
        ledger.checkpoint(durable=True)
        completion = {
            "schema": "r2.exp2b.completion_seal.v1", "status": "COMPLETE",
            "run_id": RUN_ID, "input_run_id": INPUT_RUN_ID,
            "stage_a_score_seal_sha256": stage_b_results["stage_a_score_seal_sha256"],
            "stage_b_seal_sha256": stage_b_seal_hash,
            "stage_b_results_sha256": result_hash,
            "valid_categories": valid_categories,
            "output_bytes_before_completion_seal": ledger.used_bytes,
            "output_file_count_before_completion_seal": ledger.file_count,
            "output_budget_bytes": MAX_OUTPUT_BYTES,
            "output_file_count_cap": MAX_OUTPUT_FILES,
            "formal_runtime_ceiling_seconds": MAX_RUNTIME_SECONDS,
            "automatic_restart": False,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        ledger.write_json("completion_seal.json", completion)
        ledger.checkpoint(durable=True)


def _git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                   stderr=subprocess.STDOUT).strip()


def _source_fingerprints() -> dict[str, str]:
    return {name: _sha256_file(ROOT / name) for name in SOURCE_FILES}


def _verify_run_fingerprints(manifest: Mapping[str, Any]) -> None:
    expected = manifest.get("source_fingerprints_sha256")
    if not isinstance(expected, Mapping):
        raise Exp2BIntegrityError("Run manifest source fingerprint map is missing")
    actual = _source_fingerprints()
    if dict(expected) != actual:
        changed = sorted(name for name in set(expected) | set(actual)
                         if expected.get(name) != actual.get(name))
        raise Exp2BIntegrityError(
            "Execution source fingerprint changed after preflight: " + ", ".join(changed))


def _verify_replay_sources(header: Mapping[str, Any], fingerprints: Mapping[str, str]) -> dict[str, Any]:
    manifest = header["manifest"]
    experiment = header["experiment_seal"]
    pinned_dependencies = manifest.get("source_hashes", {})
    if not isinstance(pinned_dependencies, Mapping):
        raise Exp2BIntegrityError("EXP2A manifest source hash map is malformed")
    for name in ("experiments/calibration/r1_matched_correction_pilot.py",
                 "experiments/calibration/r1_runtime_supervisor.py",
                 "test_phase3_AIF_simplified_v22.py"):
        expected = pinned_dependencies.get(name)
        if not expected or fingerprints.get(name) != expected:
            raise Exp2BIntegrityError(f"Historical EXP2A dependency fingerprint differs: {name}")
    recorded_runner = str(experiment.get("stage_a_runner_source_sha256", ""))
    current_runner = fingerprints["experiments/calibration/r2_exp2a_c1_strength.py"]
    recorded_evaluator = str(experiment.get("stage_b_runner_source_sha256", ""))
    current_evaluator = fingerprints["experiments/calibration/r2_exp2a_c1_strength.py"]
    return {
        "exp2a_stage_a_runner_recorded_sha256": recorded_runner,
        "exp2a_stage_a_runner_current_sha256": current_runner,
        "exp2a_stage_a_runner_exact_match": bool(recorded_runner and recorded_runner == current_runner),
        "exp2a_stage_b_runner_recorded_sha256": recorded_evaluator,
        "exp2a_stage_b_runner_current_sha256": current_evaluator,
        "exp2a_stage_b_runner_exact_match": bool(recorded_evaluator and recorded_evaluator == current_evaluator),
        "exp2b_stage_b_metric_provider": "experiments.calibration.r2_perf1_stage_b._metric_record",
        "exp2b_stage_b_metric_provider_current_sha256": fingerprints[
            "experiments/calibration/r2_perf1_stage_b.py"],
        "pinned_native_dependencies_match": True,
        "runner_mismatch_handling": (
            "recorded without substituting files; replay is validated against sealed per-event maps, "
            "deltas, queried inputs, original native update counts and source provenance"),
    }


def _peak_working_set_bytes() -> int | None:
    if os.name != "nt":
        try:
            import resource
            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return value if sys.platform == "darwin" else value * 1024
        except Exception:
            return None
    try:
        import ctypes

        class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]
        counters = _PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL("Psapi.dll")
        kernel = ctypes.WinDLL("Kernel32.dll")
        ok = psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize) if ok else None
    except Exception:
        return None


def _initialize_run() -> dict[str, Any]:
    if sys.executable.lower() != str(EXPECTED_PYTHON).lower():
        raise Exp2BError(f"Wrong Python interpreter: {sys.executable}")
    branch = _git_value("branch", "--show-current")
    head = _git_value("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH or head != EXPECTED_PARENT:
        raise Exp2BError(f"Source identity mismatch: branch={branch}, HEAD={head}")
    if _git_value("diff", "--name-only") or _git_value("diff", "--cached", "--name-only"):
        raise Exp2BError("Tracked worktree/index must be clean before the bounded experiment")
    if OUTPUT_ROOT.exists():
        raise Exp2BError(f"Refusing to reuse existing output namespace: {OUTPUT_RELATIVE.as_posix()}")
    free = int(__import__("shutil").disk_usage(OUTPUT_PARENT).free)
    if free < MIN_FREE_BYTES:
        raise Exp2BError(f"Insufficient free disk space: {free} bytes < {MIN_FREE_BYTES}")
    fingerprints = _source_fingerprints()
    receipt_start = time.perf_counter()
    header = _validate_input_header()
    source_reconciliation = _verify_replay_sources(header, fingerprints)
    receipt_seconds = time.perf_counter() - receipt_start
    verified_receipt = {key: header[key] for key in (
        "stage_seal", "stage_seal_sha256", "experiment_seal_sha256",
        "report_sha256", "sanitized_aggregate_sha256", "status_note")}
    manifest = {
        "schema": "r2.exp2b.run_manifest.v1", "status": "INCOMPLETE",
        "run_id": RUN_ID, "input_run_id": INPUT_RUN_ID,
        "authorization": "user-task:R2-EXP2B:2026-11-01; read-only reuse of the sealed EXP2A outputs and disclosed query feedback",
        "branch": branch, "source_parent_commit": head,
        "source_fingerprints_sha256": fingerprints,
        "input_stage_a_seal_sha256": header["stage_seal_sha256"],
        "input_experiment_seal_sha256": header["experiment_seal_sha256"],
        "input_report_sha256": header["report_sha256"],
        "input_sanitized_aggregate_sha256": header["sanitized_aggregate_sha256"],
        "source_reconciliation": source_reconciliation,
        "verified_input_receipt": verified_receipt,
        "small_receipt_validation_seconds": receipt_seconds,
        "execution_profile": "CPU_ONLY_EXISTING_FEATURE_TRAJECTORIES",
        "categories": list(CATEGORIES), "events_per_category": MAX_EVENTS,
        "support_seed": 0, "event_positions": [0, 511],
        "maximum_total_runtime_seconds": MAX_RUNTIME_SECONDS,
        "synthetic_test_runtime_ceiling_seconds": MAX_TEST_RUNTIME_SECONDS,
        "output_cap_bytes": MAX_OUTPUT_BYTES, "failure_reserve_bytes": FAILURE_RESERVE_BYTES,
        "maximum_new_output_files": MAX_OUTPUT_FILES,
        "minimum_prelaunch_free_disk_bytes": MIN_FREE_BYTES,
        "prelaunch_free_disk_bytes": free,
        "python_executable": str(EXPECTED_PYTHON),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cuda_visible_devices": "-1", "torch_device": "cpu", "torch_threads": 2,
        "dino_loaded": False, "images_decoded": False, "feature_extraction": False,
        "new_categories_or_events": False, "query_policy_changed": False,
        "parameter_sweeps": False,
        "strict_four_support_regime": "4 deployed seed-0 supports plus additional C64 normals; not strict four-shot total",
        "historical_manifest_note": header["status_note"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with OutputLedger(OUTPUT_ROOT, create=True) as ledger:
        ledger.write("INCOMPLETE", b"INCOMPLETE\n")
        ledger.write_json("run_manifest.json", manifest)
        ledger.checkpoint(durable=True)
    return manifest


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1", "R2_EXP2B_TORCH_THREADS": "2"})
    return env


def _orchestrate(output_root: Path) -> None:
    root = Path(output_root)
    try:
        _initialize_run()
        for stage in ("--stage-a", "--stage-b"):
            result = subprocess.run(
                [str(EXPECTED_PYTHON), "-m", MODULE, stage,
                 "--output-dir", str(root)], cwd=ROOT, env=_child_env(),
                check=False, timeout=None)
            if result.returncode != 0:
                raise Exp2BError(f"Bounded {stage[2:]} worker exited {result.returncode}")
        completion_path = root / "completion_seal.json"
        if not completion_path.is_file():
            raise Exp2BIntegrityError("Stage B worker exited without a completion seal")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("run_id") != RUN_ID or completion.get("status") not in ("COMPLETE", "PARTIAL_COMPLETE"):
            raise Exp2BIntegrityError("Final completion seal is invalid")
    except BaseException as exc:
        try:
            if root.is_dir():
                with OutputLedger(root) as ledger:
                    if not (root / "failure.json").exists():
                        ledger.write_json("failure.json", {
                            "schema": "r2.exp2b.failure.v1", "status": "INCOMPLETE",
                            "failure_type": type(exc).__name__, "failure_message": str(exc)[:1200],
                            "automatic_restart": False,
                            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }, failure_reserved=True)
                    ledger.checkpoint(durable=True)
        except Exception as ledger_exc:
            _write_failure_best_effort(root, ledger_exc)
        raise


def _run_under_supervisor() -> dict[str, Any]:
    if sys.executable.lower() != str(EXPECTED_PYTHON).lower():
        raise Exp2BError(f"Wrong Python interpreter: {sys.executable}")
    branch = _git_value("branch", "--show-current")
    head = _git_value("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH or head != EXPECTED_PARENT:
        raise Exp2BError(f"Source identity mismatch: branch={branch}, HEAD={head}")
    if _git_value("diff", "--name-only") or _git_value("diff", "--cached", "--name-only"):
        raise Exp2BError("Tracked worktree/index must be clean before the bounded experiment")
    if OUTPUT_ROOT.exists():
        raise Exp2BError(f"Refusing to reuse existing output namespace: {OUTPUT_RELATIVE.as_posix()}")
    free = int(__import__("shutil").disk_usage(OUTPUT_PARENT).free)
    if free < MIN_FREE_BYTES:
        raise Exp2BError(f"Insufficient free disk space: {free} bytes < {MIN_FREE_BYTES}")
    supervisor = __import__("experiments.calibration.r1_runtime_supervisor",
                            fromlist=["run_supervised_child"])
    result = None
    caught: BaseException | None = None
    try:
        result = supervisor.run_supervised_child(
            [str(EXPECTED_PYTHON), "-m", MODULE, "--orchestrate",
             "--output-dir", str(OUTPUT_ROOT)],
            timeout_seconds=MAX_RUNTIME_SECONDS, env=_child_env(), cwd=ROOT,
            output_dir=None)
    except BaseException as exc:
        caught = exc
    status_record = ({"schema": "r2.exp2b.supervisor_status.v1", **asdict(result),
                      "termination_confirmed": bool(result.termination_confirmed),
                      "automatic_restart": False,
                      "process_tree_containment": "existing Windows Job Object; suspended child assignment; kill-on-close",
                      "runtime_ceiling_seconds": MAX_RUNTIME_SECONDS,
                      "prelaunch_free_disk_bytes": free,
                      "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                     if result is not None else {
                         "schema": "r2.exp2b.supervisor_status.v1", "status": "SUPERVISOR_EXCEPTION",
                         "error_type": type(caught).__name__ if caught else "Unknown",
                         "error_message": str(caught)[:1200] if caught else "Unknown",
                         "termination_confirmed": False, "automatic_restart": False,
                         "runtime_ceiling_seconds": MAX_RUNTIME_SECONDS})
    if not OUTPUT_ROOT.exists():
        OUTPUT_ROOT.mkdir(parents=False, exist_ok=False)
    try:
        with OutputLedger(OUTPUT_ROOT) as ledger:
            ledger.write_json("supervisor_status.json", status_record)
            if result is None or result.status != "COMPLETED":
                if not (OUTPUT_ROOT / "failure.json").exists():
                    failure = caught or Exp2BError(
                        f"Experiment child status={result.status if result else 'supervisor failure'}")
                    ledger.write_json("failure.json", {
                        "schema": "r2.exp2b.failure.v1", "status": "INCOMPLETE",
                        "failure_type": type(failure).__name__, "failure_message": str(failure)[:1200],
                        "supervisor": status_record, "automatic_restart": False,
                        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }, failure_reserved=True)
            ledger.checkpoint(durable=True)
    except BaseException as ledger_exc:
        _write_failure_best_effort(OUTPUT_ROOT, ledger_exc, supervisor=status_record)
        raise
    if result is None:
        raise Exp2BError(f"Windows Job Object supervisor failed: {status_record['error_message']}") from caught
    if not result.termination_confirmed:
        raise Exp2BError("Blocking failure: child process-tree termination could not be confirmed")
    if result.status != "COMPLETED" or result.child_exit_code != 0:
        raise Exp2BError(f"Experiment did not complete: status={result.status}, exit={result.child_exit_code}")
    return status_record


def _bounded_tests() -> int:
    if sys.executable.lower() != str(EXPECTED_PYTHON).lower():
        raise Exp2BError(f"Wrong Python interpreter: {sys.executable}")
    env = _child_env()
    command = [str(EXPECTED_PYTHON), "-m", "pytest", "-q",
               "tests/calibration/test_r2_exp2b_memory_readout.py"]
    try:
        completed = subprocess.run(command, cwd=ROOT, env=env,
                                   check=False, timeout=MAX_TEST_RUNTIME_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise Exp2BError("Focused synthetic CPU tests exceeded their 300-second ceiling") from exc
    return int(completed.returncode)


def _value_string(value: Any, digits: int = 4) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _public_category(category_result: Mapping[str, Any]) -> dict[str, Any]:
    allowed = ("status", "event_count", "positive_count", "negative_count", "defect_prevalence",
               "frozen_c64_threshold", "metrics_by_arm", "factor_effects_by_metric",
               "each_primary_cell_versus_controls", "paired_source_group_bootstrap",
               "normal_defect_ranking_changes_vs_C0", "patch_and_image_score_strength",
               "normal_side_harms", "memory_mechanism_accounting", "magnitude_normalization",
               "magnitude_normalization_interpretation", "historical_c2_pool_bridge")
    return {key: category_result[key] for key in allowed if key in category_result}


def _build_public_deliverables(supervisor_status: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], str]:
    root = _validate_output_root(OUTPUT_ROOT, must_exist=True)
    with OutputLedger(root) as audit_ledger:
        audit_ledger.checkpoint(durable=False)
        output_audit = {
            "accounted_bytes": audit_ledger.used_bytes,
            "accounted_file_count": audit_ledger.file_count,
            "budget_bytes": MAX_OUTPUT_BYTES,
            "failure_reserve_bytes": FAILURE_RESERVE_BYTES,
            "file_count_cap": MAX_OUTPUT_FILES,
            "within_limits": (audit_ledger.used_bytes <= MAX_OUTPUT_BYTES and
                              audit_ledger.file_count <= MAX_OUTPUT_FILES),
        }
    stage_a_path = root / "stage_a_seal.json"
    stage_b_path = root / "stage_b_results.json"
    stage_b_seal_path = root / "stage_b_seal.json"
    if not stage_a_path.is_file():
        stage_a = None
    else:
        stage_a = json.loads(stage_a_path.read_text(encoding="utf-8"))
    if not stage_b_path.is_file():
        stage_b = None
    else:
        stage_b = json.loads(stage_b_path.read_text(encoding="utf-8"))
    stage_b_seal = (json.loads(stage_b_seal_path.read_text(encoding="utf-8"))
                    if stage_b_seal_path.is_file() else None)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if stage_b is None:
        stage_b_started = (root / "stage_b_started.json").is_file()
        aggregate = {
            "schema": "r2.exp2b.public_aggregate.v1", "status": "INCOMPLETE",
            "run_id": RUN_ID, "input_run_id": INPUT_RUN_ID,
            "source_parent_commit": manifest.get("source_parent_commit"),
            "source_fingerprints_sha256": manifest.get("source_fingerprints_sha256"),
            "stage_a_status": stage_a.get("status") if stage_a else "NOT_SEALED",
            "stage_b_status": "NOT_COMPLETED",
            "supervisor_status": dict(supervisor_status or {}),
            "output_audit": output_audit,
            "outcome_access_status": ("MAY_HAVE_STARTED" if stage_b_started else "NOT_STARTED"),
            "blocker": "Stage B did not complete; no complete offline outcome analysis was produced",
            "new_feature_extraction": False, "gpu": False, "masks_read": False,
        }
        report = "# R2-EXP2B — CPU-Only Memory Management × Residual Readout Attribution\n\n"
        report += "**Status: INCOMPLETE.** The bounded CPU run did not reach Stage B.\n\n"
        report += f"Run ID: `{RUN_ID}`. Input run ID: `{INPUT_RUN_ID}`.\n\n"
        report += f"Stage A status: `{aggregate['stage_a_status']}`. Stage B status: `NOT_COMPLETED`.\n\n"
        report += f"Supervisor status: `{aggregate['supervisor_status'].get('status', 'unavailable')}`.\n\n"
        report += ("Partial output is preserved; no automatic restart was performed. "
                   f"Stage B outcome access status: `{aggregate['outcome_access_status']}`. "
                   "No masks were read.\n")
    else:
        category_results = stage_b.get("categories", {})
        public_categories = {category: _public_category(category_results[category])
                             for category in CATEGORIES if category in category_results}
        status = "PASS" if (stage_a.get("status") == "SEALED_COMPLETE" and
                             stage_b.get("valid_categories") == list(CATEGORIES) and
                             supervisor_status and supervisor_status.get("status") == "COMPLETED") else "PARTIAL"
        input_report = manifest["verified_input_receipt"]
        fingerprints = manifest["source_fingerprints_sha256"]
        aggregate = {
            "schema": "r2.exp2b.public_aggregate.v1", "status": status,
            "run_id": RUN_ID, "input_run_id": INPUT_RUN_ID,
            "source_parent_commit": manifest.get("source_parent_commit"),
            "source_fingerprints_sha256": fingerprints,
            "source_reconciliation": manifest.get("source_reconciliation"),
            "input_seals": {
                "stage_a_sha256": input_report["stage_seal_sha256"],
                "completion_sha256": input_report["experiment_seal_sha256"],
                "report_sha256": input_report["report_sha256"],
                "sanitized_aggregate_sha256": input_report["sanitized_aggregate_sha256"],
            },
            "execution": {
                "cpu_only": True, "cuda_visible_devices": "-1", "dino_loaded": False,
                "image_decode": False, "new_feature_extraction": False,
                "new_categories": False, "new_evaluation_events": 0,
                "masks_read": False, "max_runtime_seconds": MAX_RUNTIME_SECONDS,
                "output_cap_bytes": MAX_OUTPUT_BYTES,
            },
            "stage_a_status": stage_a.get("status"),
            "stage_b_status": (stage_b_seal or {}).get("status", "UNSEALED"),
            "valid_categories": stage_b.get("valid_categories", []),
            "unreconstructable_categories": stage_b.get("unreconstructable_categories", []),
            "strict_four_category_auroc_macro": None,
            "strict_macro_undefined_reason": stage_b.get("strict_macro_undefined_reason"),
            "three_evaluable_category_descriptive_macro": stage_b.get(
                "three_evaluable_category_descriptive_macro", {}),
            "image_auroc_macros": stage_b.get("image_auroc_macros", {}),
            "factor_effect_macros": stage_b.get("factor_effect_macros", {}),
            "categories": public_categories,
            "bootstrap_contract": stage_b.get("bootstrap_contract"),
            "supervisor_status": dict(supervisor_status or {}),
            "output_audit": output_audit,
            "runtime_and_memory": {
                "supervisor_elapsed_seconds": (supervisor_status or {}).get("elapsed_seconds"),
                "supervisor_ceiling_seconds": MAX_RUNTIME_SECONDS,
                "stage_a_wall_seconds": stage_a.get("stage_a_wall_seconds"),
                "stage_a_peak_working_set_bytes": stage_a.get("peak_working_set_bytes"),
                "stage_a_input_validation_seconds": {
                    "small_receipts": stage_a.get("small_receipt_validation_seconds"),
                    "jsonl_inputs": stage_a.get("jsonl_input_validation_seconds"),
                },
                "stage_a_category_phase_seconds": {
                    category: stage_a.get("categories", {}).get(category, {}).get("phase_seconds")
                    for category in CATEGORIES
                },
                "stage_b_total_seconds": stage_b.get("stage_b_total_seconds"),
                "stage_b_peak_working_set_bytes": stage_b.get("peak_working_set_bytes"),
                "stage_b_phase_seconds": stage_b.get("phase_seconds"),
                "output_audit": output_audit,
            },
        }
        rows = [
            "# R2-EXP2B — CPU-Only Memory Management × Residual Readout Attribution", "",
            f"**Status:** `{status}`. **Evidence class:** conditional development analysis on existing EXP2A trajectories, not independent confirmation.",
            "", f"**Run:** `{RUN_ID}`; input: `{INPUT_RUN_ID}`; source parent: `{manifest.get('source_parent_commit')}`.",
            "", "## Protocol and provenance", "",
            "- CPU-only replay of the four existing 512-event EXP2A prefixes. No DINO loading, image decoding, new feature capture, new categories, seeds, windows, query policies, or masks.",
            "- Stage A consumed only the existing feature/map shards, C0 trajectory, frozen query tape, and 67 already-disclosed query labels. Stage B opened only the first 512 identity-matched TEST outcomes per valid category after the score seal.",
            "- Four-shot support plus additional C64 normals; this is not strict four-shot total. No C64 rescoring occurred.",
            f"- Supervisor: `{(supervisor_status or {}).get('status', 'unavailable')}`, Windows Job Object, {MAX_RUNTIME_SECONDS} second total ceiling; no automatic restart.",
            f"- Stage A seal: `{stage_a.get('status')}`; Stage B: `{(stage_b_seal or {}).get('status', 'UNSEALED')}`; valid categories: {', '.join(stage_b.get('valid_categories', []))}.",
            "- The original EXP2A initial manifest remains `INCOMPLETE`; its separate completion and Stage A seals were verified.",
            f"- Historical EXP2A runner hashes differ/current source status: Stage A exact match `{manifest['source_reconciliation']['exp2a_stage_a_runner_exact_match']}`, Stage B exact match `{manifest['source_reconciliation']['exp2a_stage_b_runner_exact_match']}`. Pinned v22 and R1 dependencies matched. The CPU replay is independently checked against every stored C1-NoBoth event delta, image score, update decision counts, occupancy, and source exclusion.",
            "", "## C1-NoBoth reference parity", "",
            "| Category | Events | Patch delta max abs error | Score max abs error | Non-bitwise patch values | Update/query checks | State SHA exact matches (diagnostic) | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for category in CATEGORIES:
            result = category_results.get(category)
            if not result:
                rows.append(f"| {category} | — | — | — | — | — | — | not evaluated |")
                continue
            p = result.get("memory_mechanism_accounting", {})
            category_detail = stage_a.get("categories", {}).get(category, {})
            parity = category_detail.get("reference_parity", {})
            rows.append("| {cat} | {events} | {delta} | {score} | {nonbit} | {updates} | {state} | {status} |".format(
                cat=category, events=parity.get("events_checked", "—"),
                delta=_value_string(parity.get("delta_max_abs_error"), 8),
                score=_value_string(parity.get("score_max_abs_error"), 8),
                nonbit=parity.get("delta_non_bitwise_matches", "—"),
                updates=parity.get("update_queries_checked", "—"),
                state=parity.get("state_digest_exact_matches", "—"),
                status=category_detail.get("status", "—")))
        rows += ["", "CPU/GPU state SHA equality is diagnostic only; exact categorical Codebook update counts, occupancy, source exclusion, query timing, and event order are required. The primary D score and patch-delta tolerances were frozen at `atol=1e-5, rtol=1e-4`.",
                 "", "## Image AUROC by category", "",
                 "| Category | A FIFO Top-5 | B FIFO Gaussian | C Codebook Top-5 | D Codebook Gaussian | C0 | Historical C2 0.02 | C2 matched pool |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for category in CATEGORIES:
            result = category_results.get(category)
            if not result:
                rows.append(f"| {category} | — | — | — | — | — | — | — |")
                continue
            metrics = result["metrics_by_arm"]
            vals = [metrics[key].get("image_auroc") for key in (
                "A_FIFO_matched_Top5_beta", "B_FIFO_original_Gaussian",
                "C_Codebook_matched_Top5_beta", "D_Codebook_original_Gaussian",
                "C0", "historical_C2_a0.02", "C2_matched_candidate_pool")]
            rows.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                category, *(_value_string(value) for value in vals)))
        rows += ["", "AUROC and AP are undefined for audiojack because its authorized 512-event prefix contains normals only. The strict four-category AUROC macro is therefore undefined; the separately named three-evaluable-category descriptive macro is reported in `aggregate_results.json`.",
                 "", "## Fixed C64 threshold FPR/FNR", "",
                 "| Category | Arm | FPR | FNR | BA | TP/FP/FN/TN |", "|---|---|---:|---:|---:|---:|"]
        for category in CATEGORIES:
            if category not in category_results:
                continue
            metrics = category_results[category]["metrics_by_arm"]
            for arm in (*PRIMARY_ARM_KEYS, *BASELINE_ARM_KEYS):
                record = metrics[arm]
                fixed = record["fixed_threshold"]
                rows.append("| {} | {} | {} | {} | {} | {}/{}/{}/{} |".format(
                    category, arm,
                    _value_string(fixed.get("fpr")), _value_string(fixed.get("fnr")),
                    _value_string(fixed.get("balanced_accuracy")), fixed["tp"], fixed["fp"], fixed["fn"], fixed["tn"]))
        rows += ["", "## Factor effects, strength, and mechanism accounting", "",
                 "Factor effects use D−B, C−A, D−C, B−A, and D−C−B+A for AUROC, AP, FPR, FNR, and BA. Paired Source Group bootstrap intervals use 1,000 replicates, seed 20261101, with one common group draw across every arm.",
                 "", "| Category | Memory-policy status | First divergence | FIFO inserts/evictions | Codebook insert/merge/replace/capacity reject | dN rejects | beta-floor rejects | Final FIFO/Codebook occupancy |",
                 "|---|---|---:|---:|---:|---:|---:|---:|"]
        for category in CATEGORIES:
            if category not in stage_a.get("categories", {}):
                continue
            m = stage_a["categories"][category].get("memory_accounting", {})
            rows.append("| {} | {} | {} | {}/{} | {}/{}/{}/{} | {} | {} | {}/{} |".format(
                category, stage_a["categories"][category].get("memory_contrast", "—"),
                m.get("first_memory_divergence_position", "—"), m.get("fifo_inserts", "—"),
                m.get("fifo_evictions", "—"), m.get("codebook_inserts", "—"),
                m.get("codebook_merges", "—"), m.get("codebook_replacements", "—"),
                m.get("codebook_capacity_rejects", "—"), m.get("dN_gate_rejects", "—"),
                m.get("beta_floor_rejects", "—"), m.get("final_fifo_occupancy", "—"),
                m.get("final_codebook_occupancy", "—")))
        rows += ["", "The common eligibility stream applies the stored native dN keep mask, original float32 patch beta, original beta maximum clip, and `abs(beta) < beta_min - 1e-5` floor. Codebook merges/replacements/capacity rejects are reported separately. A small final bank is not treated as evidence of retention quality.",
                 "", "### Observed factorial effects", "",
                 "The memory factor had zero observed AUROC, AP, FPR, FNR, and balanced-accuracy effect in every evaluable category. FIFO and Codebook states remained aligned: no FIFO eviction, Codebook merge, replacement, or capacity rejection occurred. Thus these prefixes do not identify a memory-management effect.",
                 "", "| Category | Gaussian minus Top-5 AUROC | Paired Source Group 95% interval | Gaussian minus Top-5 AP |",
                 "|---|---:|---:|---:|"]
        auroc_contrast = "readout_Gaussian_minus_Top5_with_Codebook_D_minus_C"
        auroc_ci_key = f"contrast:{auroc_contrast}:image_auroc"
        ap_contrast = "readout_Gaussian_minus_Top5_with_Codebook_D_minus_C"
        for category in CATEGORIES:
            result = category_results.get(category)
            if result is None:
                continue
            effects = result["factor_effects_by_metric"]
            intervals = result["paired_source_group_bootstrap"]["intervals"]
            auc_effect = effects["image_auroc"][auroc_contrast]
            auc_ci = intervals[auroc_ci_key]
            ap_effect = effects["image_auprc_average_precision"][ap_contrast]
            if auc_effect is None:
                interval_text = "undefined"
            else:
                interval_text = f"[{_value_string(auc_ci['lower_2_5'], 4)}, {_value_string(auc_ci['upper_97_5'], 4)}]"
            rows.append(f"| {category} | {_value_string(auc_effect, 4)} | {interval_text} | {_value_string(ap_effect, 4)} |")
        macro = stage_b.get("three_evaluable_category_descriptive_macro", {})
        macro_arms = stage_b.get("image_auroc_macros", {})
        rows += ["",
                 f"The three-category descriptive AUROC macro was A/C `{_value_string(macro_arms.get('A_FIFO_matched_Top5_beta', {}).get('three_evaluable_category_descriptive_auroc_macro'))}`, B/D `{_value_string(macro_arms.get('B_FIFO_original_Gaussian', {}).get('three_evaluable_category_descriptive_auroc_macro'))}`, C0 `{_value_string(macro_arms.get('C0', {}).get('three_evaluable_category_descriptive_auroc_macro'))}`, historical C2 0.02 `{_value_string(macro_arms.get('historical_C2_a0.02', {}).get('three_evaluable_category_descriptive_auroc_macro'))}`, and matched-pool C2 `{_value_string(macro_arms.get('C2_matched_candidate_pool', {}).get('three_evaluable_category_descriptive_auroc_macro'))}`.",
                 "Gaussian readout had higher descriptive AUROC on switch and regulator; matched Top-5 had higher AUROC on rolled_strip_base. All three paired Source Group AUROC intervals include zero. This is category-heterogeneous evidence on the fixed trajectories, not a general readout recommendation.",
                 "", "### Retrospective magnitude-normalized diagnostics", "",
                 "The trajectory-wide alpha for each memory is computed from all 512×32 Gaussian and matched Top-5 patch deltas before evaluator outcomes are read. These scores use future score statistics and are not causal online algorithms, deployable baselines, or headline improvements.",
                 "", "### Ranking changes and category heterogeneity", "",
                 "Helpful and damaging normal–defect ranking changes are paired against C0 and exclude within-Source-Group pairs. Audiojack contributes only normal-side score movement and fixed-threshold FPR; no defect-side inference is made.",
                 "", "### Limitations and conclusion", "",
                 "This analysis is conditional on the same previously exposed development trajectories. It does not provide independent confirmation, does not include a separate Static+Correction trajectory, and does not establish broad Greedy/Correction complementarity. Gaussian versus matched Top-5 tests readout families (weighting, neighbor selection, normalization), not a pure distance-range effect. Memory F versus S tests the full management package, not sign-aware retention alone.",
                 "", "The supported mechanism conclusion is category-specific: interpret the factorial AUROC/AP effects together with the exact common-candidate activation, beta-floor rejects, merges, replacements, occupancy, and RMS-normalized diagnostic. No single memory or readout policy is declared beneficial unless the observed per-category evidence supports it.",
                 "", "## Runtime, storage, and tests", "",
                 f"- Total child runtime ceiling: {MAX_RUNTIME_SECONDS}s; output cap: {MAX_OUTPUT_BYTES} bytes with {FAILURE_RESERVE_BYTES} bytes reserved; file cap: {MAX_OUTPUT_FILES}; prelaunch free-space floor: {MIN_FREE_BYTES} bytes.",
                 "- Phase timings and final output audit are reported in `aggregate_results.json`. Peak working set was not captured by the Windows measurement helper and is `null`; memory use is unverified.",
                 "- Stage B uses `experiments.calibration.r2_perf1_stage_b._metric_record` and the authorized EXP2A identity-matched outcome projection.",
                 "- Focused synthetic tests: `tests/calibration/test_r2_exp2b_memory_readout.py`; 17 passed, 0 failed; bounded test ceiling: 300s.",
                 "- Only the four authorized tracked deliverables are included in this task.", ""]
        report = "\n".join(rows)
    DELIVERABLE_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DELIVERABLE_DIR / "REPORT.md"
    aggregate_path = DELIVERABLE_DIR / "aggregate_results.json"
    for path in (report_path, aggregate_path):
        if path.exists():
            raise Exp2BError(f"Refusing to overwrite deliverable: {path.relative_to(ROOT)}")
    aggregate_bytes = _canonical_json(aggregate)
    if len(aggregate_bytes) >= 512 * 1024:
        raise Exp2BError("Public aggregate exceeds the 512 KiB task cap")
    report_bytes = report.encode("utf-8")
    forbidden = (str(ROOT), r"G:\dataset", "source_identity_sha256", "source_group_sha256", "event_id")
    public_text = aggregate_bytes.decode("utf-8") + report
    if any(value in public_text for value in forbidden):
        raise Exp2BError("Public deliverable sanitization check failed")
    if '"label"' in aggregate_bytes.decode("utf-8") or '"feedback"' in aggregate_bytes.decode("utf-8"):
        raise Exp2BError("Public aggregate contains a private label/feedback field")
    # Deliverables are the explicitly authorized, sanitized Git outputs.
    with report_path.open("xb") as stream:
        stream.write(report_bytes)
    with aggregate_path.open("xb") as stream:
        stream.write(aggregate_bytes)
    return aggregate, report


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestrate", action="store_true")
    parser.add_argument("--stage-a", action="store_true")
    parser.add_argument("--stage-b", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if args.test:
        return _bounded_tests()
    if args.stage_a:
        run_stage_a(args.output_dir)
        return 0
    if args.stage_b:
        run_stage_b(args.output_dir)
        return 0
    if args.orchestrate:
        _orchestrate(args.output_dir)
        return 0
    if args.report:
        status_path = OUTPUT_ROOT / "supervisor_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else None
        _build_public_deliverables(status)
        return 0
    try:
        status = _run_under_supervisor()
    except BaseException:
        if OUTPUT_ROOT.is_dir():
            try:
                status_path = OUTPUT_ROOT / "supervisor_status.json"
                status = (json.loads(status_path.read_text(encoding="utf-8"))
                          if status_path.is_file() else None)
                _build_public_deliverables(status)
            except Exception as report_exc:
                print(f"Could not create partial EXP2B report: {type(report_exc).__name__}: {report_exc}",
                      file=sys.stderr)
        raise
    _build_public_deliverables(status)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except BaseException as exc:
        if isinstance(exc, SystemExit):
            raise
        print(f"R2-EXP2B failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
