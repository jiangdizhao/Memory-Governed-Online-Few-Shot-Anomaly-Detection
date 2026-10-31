"""Bounded R1 matched-trajectory Local Correction engineering pilot.

This module is intentionally isolated from historical scorer, writer,
controller, and experiment sources. It imports their native functions but
does not patch them. The pilot records engineering diagnostics only; it does
not evaluate anomaly-detection outcomes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
IDENTITY_CONTRACT_VERSION = "r1.source-bound.v1"
R1_PROFILE_ID = "r1_switch_s0_512_digest_contract_v1"
SUPPORT_ONLY_PROFILE_ID = "r1_switch_s0_support_bank_verify_v1"
OUTPUT_NAMESPACE_PREFIX = "r1_digest_contract_v1"
C2_ANCHOR_CAPACITY = 256
STRUCTURED_FIXED_BANK_DIGEST_ALGORITHM = (
    "experiments.calibration.native._fixed_bank_digest.v1:path-shape-dtype-bytes")
RAW_FIXED_BANK_DIGEST_ALGORITHM = (
    "experiments.p0_protocol_audit.fixed_bank_profile.v1:raw-prototype-bytes")
STAGE05 = ROOT / "outputs/calibration/stage05_exploratory_20261029"
DEFAULT_DATASET_ROOT = Path(r"G:\dataset\realiad")
DEFAULT_JSON_ROOT = DEFAULT_DATASET_ROOT / "jsons" / "realiad_jsons"
LEGACY_FAILED_OUTPUT = ROOT / "outputs/calibration/r1_switch_s0_512_engineering_20261031"
MAX_EVENTS = 512
MAX_OUTPUT_BYTES = 2 * 1024**3
OUTPUT_FAILURE_RESERVE_BYTES = 64 * 1024
MAX_GPU_SECONDS = 120 * 60
SMOKE_EVENTS = 16
QUERY_BINDING_SHA256 = "dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183"
QUERY_PROTOCOL_SHA256 = "6f1d8cf40e5e7b2d98dcfabe5836b49c6cd57069d3c8377122dba64e3fc2da1c"


class R1Error(RuntimeError):
    """Base error for a fail-closed R1 gate."""


class R1BlockedInput(R1Error):
    pass


class R1ParityError(R1Error):
    pass


class R1CausalityError(R1Error):
    pass


class R1ResourceLimit(R1Error):
    pass


class TapeIntegrityError(R1Error):
    pass


@dataclass(frozen=True)
class PilotConfig:
    category: str = "switch"
    support_seed: int = 0
    start_position: int = 0
    max_events: int = MAX_EVENTS
    query_rate: float = 0.03
    query_seed: int = 37
    device: str = "cuda"
    max_output_bytes: int = MAX_OUTPUT_BYTES
    max_gpu_seconds: int = MAX_GPU_SECONDS
    smoke_events: int = SMOKE_EVENTS
    c2_neighbors: int = 5
    c2_interface_scale: float = 0.20


def _validate_execution_identity(*, branch: str, head: str,
                                 expected_source_commit: str,
                                 profile_id: str,
                                 support_only: bool) -> dict[str, str]:
    expected_profile = SUPPORT_ONLY_PROFILE_ID if support_only else R1_PROFILE_ID
    if branch != EXPECTED_BRANCH:
        raise R1BlockedInput(
            f"Source-bound R1 execution requires branch {EXPECTED_BRANCH}; found {branch}")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", str(expected_source_commit)):
        raise R1BlockedInput("Expected source commit must be a full 40-character Git SHA")
    if head.lower() != expected_source_commit.lower():
        raise R1BlockedInput(
            f"Source revision mismatch: expected {expected_source_commit}; found {head}")
    if profile_id != expected_profile:
        raise R1BlockedInput(
            f"Incompatible R1 profile: expected {expected_profile}; found {profile_id}")
    return {
        "branch": branch,
        "source_commit": head.lower(),
        "expected_source_commit": expected_source_commit.lower(),
        "profile_id": profile_id,
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
    }


def _validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", str(run_id)):
        raise R1BlockedInput("run-id must be 1-64 safe filename characters")


def _validate_run_identity_fields(run_id: str, authorization_id: str) -> None:
    _validate_run_id(run_id)
    if (not str(authorization_id).strip() or len(str(authorization_id)) > 128 or
            any(ord(char) < 32 for char in str(authorization_id))):
        raise R1BlockedInput("A bounded, explicit authorization-id is required")


def _output_namespace(run_id: str, *, support_only: bool) -> Path:
    _validate_run_id(run_id)
    mode = "support_only" if support_only else "engineering"
    output = ROOT / "outputs/calibration" / f"{OUTPUT_NAMESPACE_PREFIX}_{mode}_{run_id}"
    if output.resolve() == LEGACY_FAILED_OUTPUT.resolve():
        raise R1BlockedInput("The failed legacy R1 output is immutable and cannot be reused")
    return output


def _validate_p0_profile(p0_config: Any, scorer_profile: Any) -> None:
    from experiments.calibration.native import NativeAdapterError, validate_p0_config
    try:
        validate_p0_config(p0_config, scorer_profile)
    except (NativeAdapterError, ValueError, TypeError) as exc:
        raise R1BlockedInput(f"Incompatible frozen P0/native scorer profile: {exc}") from exc


def _execution_profile_sha256(config: PilotConfig, scorer_profile: Any,
                              query_profile: Any, profile_id: str,
                              support_only: bool) -> str:
    return sha256_json({
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "profile_id": profile_id,
        "support_only": bool(support_only),
        "pilot_config": asdict(config),
        "native_scorer_profile": asdict(scorer_profile),
        "query_profile": asdict(query_profile),
        "support_regime": "four_deployed_supports_plus_additional_C64",
    })


def _fixed_bank_digest_evidence(bank: Any, p0: Any, expected_sha256: str,
                                structured_digest: Callable[[Any], str]) -> dict[str, Any]:
    raw_profile = p0.fixed_bank_profile(bank)
    actual_structured = structured_digest(bank)
    expected = str(expected_sha256).lower()
    layer_profile = []
    for layer in raw_profile["layers"]:
        item = dict(layer)
        item["shape"] = [int(item["n_prototypes"]), int(item["feature_dim"])]
        layer_profile.append(item)
    return {
        "expected_sha256": expected,
        "expected_algorithm": STRUCTURED_FIXED_BANK_DIGEST_ALGORITHM,
        "actual_structured_sha256": actual_structured,
        "actual_structured_algorithm": STRUCTURED_FIXED_BANK_DIGEST_ALGORITHM,
        "actual_raw_prototype_sha256": str(raw_profile["sha256"]),
        "actual_raw_prototype_algorithm": RAW_FIXED_BANK_DIGEST_ALGORITHM,
        "layer_profile": layer_profile,
        "matches_expected_structured_digest": actual_structured == expected,
    }


def _check_fixed_bank_receipt(bank: Any, p0: Any, expected_sha256: str,
                              structured_digest: Callable[[Any], str],
                              persist_evidence: Callable[[Mapping[str, Any]], None]
                              ) -> dict[str, Any]:
    evidence = _fixed_bank_digest_evidence(
        bank, p0, expected_sha256, structured_digest)
    # Persist both algorithm-specific digests and tensor metadata before fail-closed exit.
    persist_evidence(evidence)
    if not evidence["matches_expected_structured_digest"]:
        raise R1BlockedInput(
            "Fixed support receipt mismatch: "
            f"expected structured={evidence['expected_sha256']}; "
            f"actual structured={evidence['actual_structured_sha256']}; "
            f"raw prototype={evidence['actual_raw_prototype_sha256']}")
    return evidence


def _configure_historical_runtime(impl: Any, torch_module: Any) -> dict[str, Any]:
    """Apply and record the Stage 0.5 deterministic initialization before DINO load."""
    impl.set_seed(0)
    backends = torch_module.backends.cuda
    backends.enable_flash_sdp(False)
    backends.enable_mem_efficient_sdp(False)
    backends.enable_math_sdp(True)
    sdp = {
        "flash_enabled": bool(backends.flash_sdp_enabled()),
        "memory_efficient_enabled": bool(backends.mem_efficient_sdp_enabled()),
        "math_enabled": bool(backends.math_sdp_enabled()),
    }
    expected_sdp = {"flash_enabled": False,
                    "memory_efficient_enabled": False,
                    "math_enabled": True}
    if sdp != expected_sdp:
        raise R1BlockedInput(f"Historical math-only SDPA setup did not take effect: {sdp}")
    if not torch_module.are_deterministic_algorithms_enabled():
        raise R1BlockedInput("v22 deterministic-algorithm setup did not take effect")
    if (not torch_module.backends.cudnn.deterministic or
            torch_module.backends.cudnn.benchmark):
        raise R1BlockedInput("v22 cuDNN deterministic setup did not take effect")
    if (torch_module.backends.cudnn.allow_tf32 or
            torch_module.backends.cuda.matmul.allow_tf32):
        raise R1BlockedInput("v22 TF32-disable setup did not take effect")
    return {
        "profile_id": "stage05_set_seed_0_math_sdp_v1",
        "seed": 0,
        "sdp": sdp,
        "deterministic_algorithms": bool(
            torch_module.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch_module.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch_module.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch_module.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(
            torch_module.backends.cuda.matmul.allow_tf32),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "python_version": platform.python_version(),
        "torch_version": str(torch_module.__version__),
        "cuda_runtime_version": str(torch_module.version.cuda),
        "cuda_device_name": (str(torch_module.cuda.get_device_name())
                             if torch_module.cuda.is_available() else None),
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def opaque_sample_group(sample_key: str) -> str:
    return hashlib.sha256(
        ("official-sample-group\0" + str(sample_key)).encode("utf-8")).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to("cpu").contiguous()
    return sha256_bytes(str((tuple(value.shape), str(value.dtype))).encode("ascii") +
                        value.numpy().tobytes())


def arrays_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(value.shape), value.dtype.str)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    out = io.BytesIO()
    np.save(out, np.ascontiguousarray(array), allow_pickle=False)
    return out.getvalue()


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Stable, uncompressed NPZ bytes with fixed ZIP timestamps and key order."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(arrays[name]))
    return out.getvalue()


def corrected_dense_map(base_map: np.ndarray, selected_ids: Sequence[int],
                        selected_delta: np.ndarray) -> np.ndarray:
    """Apply a correction only at native selected IDs and return a new map."""
    base = np.asarray(base_map, dtype=np.float32)
    ids = np.asarray(selected_ids, dtype=np.int64)
    delta = np.asarray(selected_delta, dtype=np.float32)
    if base.ndim != 1 or ids.ndim != 1 or delta.shape != ids.shape:
        raise ValueError("Expected one dense map and aligned selected IDs/deltas")
    if len(set(ids.tolist())) != len(ids) or np.any(ids < 0) or np.any(ids >= len(base)):
        raise ValueError("Selected patch IDs are duplicate or out of range")
    if not np.isfinite(base).all() or not np.isfinite(delta).all():
        raise ValueError("Nonfinite base map or correction delta")
    result = base.copy()
    result[ids] = result[ids] + delta
    return result


def max_score(patch_map: np.ndarray) -> float:
    value = np.asarray(patch_map, dtype=np.float32)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ValueError("Expected a finite, nonempty dense patch map")
    return float(value.max())


def source_disjoint_anchor_indices(anchor_positions: Sequence[int],
                                   anchor_groups: Sequence[str], *,
                                   event_position: int,
                                   event_group: str) -> list[int]:
    if len(anchor_positions) != len(anchor_groups):
        raise R1CausalityError("Anchor positions and source groups differ in length")
    event_position = _causal_position(event_position, "current event position")
    event_group = _causal_group_sha256(event_group, "current source group")
    keep: list[int] = []
    for index, (position, group) in enumerate(zip(anchor_positions, anchor_groups)):
        anchor_position = _causal_position(position, f"anchor {index} position")
        anchor_group = _causal_group_sha256(group, f"anchor {index} source group")
        if anchor_position >= event_position:
            raise R1CausalityError(
                f"Pre-score anchor {index} has current/future position {anchor_position} "
                f"at event {event_position}")
        if anchor_group != event_group:
            keep.append(index)
    return keep


def _causal_position(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise R1CausalityError(f"{name} must be an integer")
    position = int(value)
    if position < 0:
        raise R1CausalityError(f"{name} must be nonnegative")
    return position


def _causal_group_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise R1CausalityError(f"{name} lacks valid SHA-256 provenance")
    return value


def _validate_c2_anchor_state(anchors: Sequence[Mapping[str, Any]], *,
                              event_position: int, phase: str,
                              event_group: str | None = None) -> list[int]:
    """Validate stored C2 provenance; return only pre-score eligible indices."""
    event_position = _causal_position(event_position, "current event position")
    if phase not in ("pre_score", "post_update"):
        raise ValueError(f"Unknown C2 anchor validation phase: {phase}")
    if phase == "pre_score":
        event_group = _causal_group_sha256(event_group, "current source group")
    eligible: list[int] = []
    required = {"position", "group_sha256", "patch_id", "sign", "feature",
                "feature_sha256"}
    for index, anchor in enumerate(anchors):
        if not isinstance(anchor, Mapping) or not required.issubset(anchor):
            raise R1CausalityError(f"C2 anchor {index} has missing/malformed provenance")
        position = _causal_position(anchor["position"], f"C2 anchor {index} position")
        group = _causal_group_sha256(anchor["group_sha256"],
                                     f"C2 anchor {index} source group")
        patch_id = anchor["patch_id"]
        sign = anchor["sign"]
        if (isinstance(patch_id, (bool, np.bool_)) or
                not isinstance(patch_id, (int, np.integer)) or
                not 0 <= int(patch_id) < 784):
            raise R1CausalityError(f"C2 anchor {index} has invalid patch identity")
        if isinstance(sign, (bool, np.bool_)) or not isinstance(sign, (int, np.integer)) or int(sign) not in (-1, 1):
            raise R1CausalityError(f"C2 anchor {index} has invalid signed-label provenance")
        feature = np.asarray(anchor["feature"])
        feature_sha = _causal_group_sha256(anchor["feature_sha256"],
                                           f"C2 anchor {index} feature digest")
        if (feature.ndim != 1 or feature.dtype != np.dtype(np.float32) or
                not np.isfinite(feature).all() or
                arrays_sha256({"feature": feature}) != feature_sha):
            raise R1CausalityError(f"C2 anchor {index} feature/provenance digest is invalid")
        if phase == "pre_score":
            if position >= event_position:
                raise R1CausalityError(
                    f"C2 pre-score state contains current/future anchor {position} "
                    f"at event {event_position}")
            if group != event_group:
                eligible.append(index)
        elif position > event_position:
            raise R1CausalityError(
                f"C2 post-update state contains future anchor {position} "
                f"after event {event_position}")
    return eligible


def signed_knn_delta(query_rows: np.ndarray, anchors: Sequence[Mapping[str, Any]], *,
                     event_position: int, event_group: str,
                     neighbors: int = 5, amplitude: float = 0.20) -> tuple[np.ndarray, int]:
    """Deterministic unit-testable signed kNN interface; no fitting/tuning."""
    q = np.asarray(query_rows, dtype=np.float32)
    if q.ndim != 2 or not np.isfinite(q).all() or neighbors < 1:
        raise ValueError("Invalid query feature matrix or neighbor count")
    eligible_indices = _validate_c2_anchor_state(
        anchors, event_position=event_position, phase="pre_score", event_group=event_group)
    eligible = [anchors[index] for index in eligible_indices]
    if not eligible:
        return np.zeros((len(q),), dtype=np.float32), 0
    bank = np.stack([np.asarray(a["feature"], dtype=np.float32) for a in eligible])
    signs = np.asarray([int(a["sign"]) for a in eligible], dtype=np.float32)
    if bank.ndim != 2 or bank.shape[1] != q.shape[1] or not np.isfinite(bank).all():
        raise ValueError("Invalid signed kNN anchor features")
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    pn = bank / np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-12)
    distances = 1.0 - np.clip(qn @ pn.T, -1.0, 1.0)
    result = np.zeros((len(q),), dtype=np.float32)
    k = min(int(neighbors), len(eligible))
    anchor_order = np.arange(len(eligible), dtype=np.int64)
    for row in range(len(q)):
        nearest = np.lexsort((anchor_order, distances[row]))[:k]
        result[row] = np.float32(float(amplitude) * float(signs[nearest].mean()))
    return result, len(eligible)


class CausalEventOrder:
    """Runtime guard for score seal -> query decision -> reveal -> updates."""
    def __init__(self, position: int):
        self.position = _causal_position(position, "current event position")
        self.queried: bool | None = None
        self.phase = "observed"

    def seal_scores(self) -> None:
        if self.phase != "observed":
            raise R1CausalityError("Scores were sealed in an invalid event phase")
        self.phase = "scores_sealed"

    def decide_query(self, queried: bool) -> bool:
        if (self.phase != "scores_sealed" or self.queried is not None or
                not isinstance(queried, (bool, np.bool_))):
            raise R1CausalityError("Query decision occurred before score seal or more than once")
        self.queried = bool(queried)
        self.phase = "query_decided"
        return self.queried

    def disclose(self, provider: Callable[[], int]) -> int:
        if self.phase != "query_decided" or not self.queried:
            raise R1CausalityError("Queried label requested before score seal or on an unqueried event")
        value = provider()
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or int(value) not in (0, 1)):
            raise R1CausalityError("Queried label provider returned a nonbinary value")
        value = int(value)
        self.phase = "label_disclosed"
        return value

    def corrections_updated(self) -> None:
        expected = "label_disclosed" if self.queried else "query_decided"
        if self.phase != expected:
            raise R1CausalityError("Correction update occurred before legal disclosure")
        self.phase = "corrections_updated"

    def memory_updated(self) -> None:
        if self.phase != "corrections_updated":
            raise R1CausalityError("Greedy write occurred before correction phase")
        self.phase = "memory_updated"

    def finish(self) -> None:
        if self.phase != "memory_updated":
            raise R1CausalityError("Event did not complete the required causal phases")
        self.phase = "complete"


def _seal_prediction_and_reveal(*, gate: CausalEventOrder, tape: Any,
                                position: int, arrays: Mapping[str, np.ndarray],
                                metadata: Mapping[str, Any],
                                query_positions: Sequence[int],
                                label_provider: Callable[[], int]
                                ) -> tuple[dict[str, Any], bool, int | None]:
    """Seal public scores before deciding whether to reveal a queried label."""
    if _causal_position(position, "event position") != gate.position:
        raise R1CausalityError("Tape seal position differs from causal event position")
    query_tape = tuple(_causal_position(value, "query tape position")
                       for value in query_positions)
    if query_tape != tuple(sorted(set(query_tape))):
        raise R1CausalityError("Query tape positions are not sorted and unique")
    sealed = tape.seal_event(position, arrays, metadata)
    gate.seal_scores()
    queried = gate.decide_query(position in set(query_tape))
    label = gate.disclose(label_provider) if queried else None
    return sealed, queried, label


class OrderedQueriedLabelReader:
    """Enforce exact frozen-tape order around the existing restricted reader."""
    def __init__(self, reader: Callable[[str, int, str], int],
                 positions: Sequence[int]):
        values = tuple(_causal_position(value, "queried position") for value in positions)
        if not values or values != tuple(sorted(set(values))):
            raise R1CausalityError("Queried position reader requires a nonempty sorted unique tape")
        self.reader = reader
        self.positions = values
        self.cursor = 0

    @property
    def decoded_positions(self) -> list[int]:
        return self.reader.decoded_positions

    def __call__(self, category: str, position: int, source_key: str) -> int:
        position = _causal_position(position, "queried label position")
        if self.cursor >= len(self.positions) or position != self.positions[self.cursor]:
            raise R1CausalityError("Queried label disclosure is not the next frozen tape position")
        value = self.reader(category, position, source_key)
        self.cursor += 1
        return value

    def close(self) -> None:
        self.reader.close()


def common_c0_greedy_writer_action(*, position: int, c0_score: float,
                                   theta: float, sigma: float,
                                   c0_prediction: int,
                                   candidate_ids: np.ndarray,
                                   candidate_features: np.ndarray,
                                   feature_layers: Sequence[Any],
                                   queried: bool, disclosed_label: int | None,
                                   eligibility: Callable[..., tuple[bool, int, str, Any]],
                                   update: Callable[..., Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the shared normal writer using only frozen C0 inputs and legal feedback."""
    position = _causal_position(position, "writer event position")
    score, threshold, sigma = float(c0_score), float(theta), float(sigma)
    if not all(math.isfinite(value) for value in (score, threshold, sigma)):
        raise R1CausalityError("Common Greedy writer received a nonfinite C0 input")
    if int(c0_prediction) not in (0, 1) or int(c0_prediction) != int(score >= threshold):
        raise R1CausalityError("Common Greedy writer prediction is not derived from C0 score/threshold")
    if not isinstance(queried, (bool, np.bool_)):
        raise R1CausalityError("Common Greedy writer query state must be an explicit boolean")
    ids = np.asarray(candidate_ids)
    features = np.asarray(candidate_features)
    if (ids.shape != (32,) or ids.dtype != np.dtype(np.int64) or
            len(set(ids.tolist())) != 32 or np.any(ids < 0) or np.any(ids >= 784)):
        raise R1CausalityError("Common Greedy writer candidate IDs are malformed")
    if (features.ndim != 2 or features.shape[0] != 32 or
            features.dtype != np.dtype(np.float32) or not np.isfinite(features).all()):
        raise R1CausalityError("Common Greedy writer candidate features are malformed")
    if queried:
        if (isinstance(disclosed_label, (bool, np.bool_)) or
                not isinstance(disclosed_label, (int, np.integer)) or
                int(disclosed_label) not in (0, 1)):
            raise R1CausalityError("Queried common writer lacks legal binary feedback")
    elif disclosed_label is not None:
        raise R1CausalityError("Unqueried event supplied supervised writer feedback")

    action: dict[str, Any] = {
        "input_score_source": "common_C0",
        "score": score,
        "theta": threshold,
        "sigma": sigma,
        "prediction": int(c0_prediction),
        "candidate_ids_sha256": sha256_json(ids.tolist()),
        "candidate_features_sha256": arrays_sha256({"top32_features": features}),
        "action": "no_normal_memory_write",
    }
    if not queried:
        action["eligibility_reason"] = "unqueried_event"
        return action
    if disclosed_label == 1:
        action["action"] = "verified_defect_no_normal_write"
        action["eligibility_reason"] = "verified_defect"
        return action

    allow, keep, reason, _info = eligibility(
        score=score, theta=threshold, sigmaN=max(sigma, 0.02),
        pred_defect=bool(c0_prediction), novelty=0.0)
    if not allow:
        action.update({"action": "native_gate_no_write",
                       "eligibility_reason": str(reason), "keep_count": int(keep)})
        return action
    result = update(position=position, feature_layers=feature_layers,
                    candidate_ids=ids, score=score, theta=threshold, sigma=sigma)
    action.update(dict(result))
    action.update({"action": "native_greedy_normal_update",
                   "eligibility_reason": str(reason), "keep_count": int(keep)})
    return action


def _contains_forbidden_label_payload(value: Any) -> bool:
    forbidden = {"label", "raw_label", "true_label", "anomaly_class", "is_good",
                 "ground_truth", "mask", "defect_mask"}
    if isinstance(value, Mapping):
        return any(str(key).lower() in forbidden or _contains_forbidden_label_payload(item)
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_label_payload(item) for item in value)
    return False


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    payload = (canonical_json(record) + "\n").encode("utf-8")
    budget = _budget_for_path(path)
    if budget is not None:
        budget.append(path, payload)
        return
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _write_once(path: Path, payload: bytes) -> None:
    budget = _budget_for_path(path)
    if budget is not None:
        budget.write_once(path, payload)
        return
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, value: Mapping[str, Any], *,
                 failure_reserved: bool = False) -> None:
    budget = _budget_for_path(path)
    if budget is not None:
        budget.atomic_json(path, value, failure_reserved=failure_reserved)
        return
    payload = (canonical_json(value) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as stream:
        temp_path = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def _directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


class OutputBudgetError(R1ResourceLimit):
    """A write was refused before it could consume unreserved output bytes."""


_ACTIVE_OUTPUT_BUDGETS: dict[str, "OutputBudget"] = {}
_ACTIVE_OUTPUT_BUDGETS_LOCK = threading.Lock()


class OutputBudget:
    """Pre-account known output writes and reserve bytes for failure records.

    This accountant rejects unknown files and serializes application writers.
    It is not an OS quota; unrelated same-user processes can bypass it, so the
    GPU pilot remains blocked until a hard filesystem quota is installed.
    """

    _FIXED_FILES = {
        "run_manifest.json", "sealed_scores.jsonl", "event_transitions.jsonl",
        "private_feedback.jsonl", "correction_anchor_events.jsonl",
        "supervisor_status.json", "attempt_history.jsonl",
    }

    def __init__(self, root: Path, *, max_bytes: int = MAX_OUTPUT_BYTES,
                 failure_reserve_bytes: int = OUTPUT_FAILURE_RESERVE_BYTES,
                 allowed_parent: Path | None = None,
                 allow_existing: bool = False):
        self.root = Path(os.path.abspath(root))
        self.root_key = os.path.normcase(str(self.root.resolve(strict=False)))
        self.max_bytes = int(max_bytes)
        self.failure_reserve_bytes = int(failure_reserve_bytes)
        if (self.max_bytes <= 0 or self.failure_reserve_bytes <= 0 or
                self.failure_reserve_bytes >= self.max_bytes):
            raise OutputBudgetError("Output budget and failure reserve must be positive and bounded")
        if allowed_parent is not None:
            parent = Path(allowed_parent).resolve(strict=True)
            actual_parent = self.root.parent.resolve(strict=True)
            if os.path.normcase(str(actual_parent)) != os.path.normcase(str(parent)):
                raise OutputBudgetError("Output directory is not a direct child of its authorized parent")
            if not self.root.name.startswith(f"{OUTPUT_NAMESPACE_PREFIX}_"):
                raise OutputBudgetError("Output directory lacks the versioned R1 namespace prefix")
        if self.root.exists() and not allow_existing:
            raise FileExistsError(f"Output namespace already exists: {self.root}")
        if self.root.exists() and not self.root.is_dir():
            raise OutputBudgetError("Output namespace exists but is not a directory")
        self._write_lock = threading.Lock()
        with _ACTIVE_OUTPUT_BUDGETS_LOCK:
            if self.root_key in _ACTIVE_OUTPUT_BUDGETS:
                raise OutputBudgetError("A writer is already active for this output directory")
            _ACTIVE_OUTPUT_BUDGETS[self.root_key] = self
        try:
            if allow_existing:
                if self._audit_tree() > self.max_bytes:
                    raise OutputBudgetError("Existing output already exceeds its authorized byte budget")
        except BaseException:
            self.close()
            raise

    @property
    def normal_limit(self) -> int:
        return self.max_bytes - self.failure_reserve_bytes

    def close(self) -> None:
        with _ACTIVE_OUTPUT_BUDGETS_LOCK:
            if _ACTIVE_OUTPUT_BUDGETS.get(self.root_key) is self:
                del _ACTIVE_OUTPUT_BUDGETS[self.root_key]

    def ensure_root(self) -> None:
        if self.root.exists():
            if not self.root.is_dir() or self.root.is_symlink():
                raise OutputBudgetError("Output root is not a plain directory")
            return
        self.root.mkdir(parents=True, exist_ok=False)

    def _is_reparse_point(self, path: Path) -> bool:
        if path.is_symlink():
            return True
        if os.name == "nt" and path.exists():
            import ctypes
            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
            if attrs == 0xFFFFFFFF:
                raise OutputBudgetError(f"Cannot inspect output path attributes: {path}")
            return bool(attrs & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT
        return False

    def _relative(self, path: Path) -> str:
        candidate = Path(os.path.abspath(path))
        try:
            relative = candidate.relative_to(self.root)
        except ValueError as exc:
            raise OutputBudgetError("Writer attempted a path outside the authorized output directory") from exc
        current = self.root
        if current.exists() and self._is_reparse_point(current):
            raise OutputBudgetError("Output root cannot be a symlink or reparse point")
        for part in relative.parts[:-1]:
            current = current / part
            if current.exists() and self._is_reparse_point(current):
                raise OutputBudgetError("Output subdirectory cannot be a symlink or reparse point")
        name = relative.as_posix()
        if (name not in self._FIXED_FILES and
                not re.fullmatch(r"events/event_[0-9]{6}\.npz", name) and
                not re.fullmatch(r"(?:run_manifest|supervisor_status)\.json\.[A-Fa-f0-9-]+\.tmp", name)):
            raise OutputBudgetError(f"Output filename is not in the audited writer allowlist: {name}")
        return name

    def _audit_tree(self) -> int:
        if not self.root.exists():
            return 0
        if not self.root.is_dir() or self._is_reparse_point(self.root):
            raise OutputBudgetError("Output namespace is not a plain directory")
        total = 0
        for path in self.root.rglob("*"):
            if path.is_dir():
                if path.relative_to(self.root).as_posix() != "events":
                    raise OutputBudgetError("Output contains an unaccounted subdirectory")
                if self._is_reparse_point(path):
                    raise OutputBudgetError("Output directory contains a reparse point")
                continue
            relative = self._relative(path)
            if not path.is_file() or self._is_reparse_point(path):
                raise OutputBudgetError(f"Output contains a non-regular file: {relative}")
            stat = path.stat()
            if getattr(stat, "st_nlink", 1) > 1:
                raise OutputBudgetError(f"Output contains a multiply-linked file: {relative}")
            total += int(stat.st_size)
        return total

    def _acquire_os_mutex(self):
        if os.name != "nt":
            raise OutputBudgetError("Cross-process output serialization requires the Windows mutex backend")
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        kernel32.ReleaseMutex.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        name = "Local\\R1OutputBudget-" + hashlib.sha256(self.root_key.encode()).hexdigest()
        handle = kernel32.CreateMutexW(None, False, name)
        if not handle:
            raise OutputBudgetError("CreateMutexW failed for output writer serialization")
        handle = ctypes.c_void_p(handle)
        status = kernel32.WaitForSingleObject(handle, 0)
        if status not in (0x00000000, 0x00000080):  # WAIT_OBJECT_0 / WAIT_ABANDONED
            kernel32.CloseHandle(handle)
            raise OutputBudgetError("Concurrent writer detected for output directory")

        def release() -> None:
            kernel32.ReleaseMutex(handle)
            kernel32.CloseHandle(handle)
        return release

    def _acquire_write_guards(self):
        if not self._write_lock.acquire(blocking=False):
            raise OutputBudgetError("Concurrent writer detected in this process")
        try:
            return self._acquire_os_mutex()
        except BaseException:
            self._write_lock.release()
            raise

    def _write(self, path: Path, payload: bytes, *, append: bool,
               failure_reserved: bool) -> None:
        if not isinstance(payload, bytes):
            raise TypeError("Budgeted payload must be bytes")
        target = Path(path)
        relative = self._relative(target)
        release_os = self._acquire_write_guards()
        try:
            current_bytes = self._audit_tree()
            if current_bytes > self.max_bytes:
                raise OutputBudgetError("Existing output already exceeds its authorized byte budget")
            exists = target.exists()
            if exists and not append:
                raise FileExistsError(f"Refusing to overwrite output file: {relative}")
            if append and exists and not target.is_file():
                raise OutputBudgetError("Append destination is not a regular file")
            limit = self.max_bytes if failure_reserved else self.normal_limit
            if current_bytes + len(payload) > limit:
                raise OutputBudgetError(
                    f"Write refused before I/O: {relative} would exceed "
                    f"{'total' if failure_reserved else 'normal'} output budget")
            flags = os.O_CREAT | os.O_WRONLY | (os.O_APPEND if append else os.O_EXCL)
            if hasattr(os, "O_BINARY"):
                flags |= os.O_BINARY
            fd = os.open(target, flags, 0o600)
            try:
                with os.fdopen(fd, "ab" if append else "wb", closefd=False) as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                os.close(fd)
            if self._audit_tree() > limit:
                raise OutputBudgetError("Observed output exceeds the pre-write budget")
        finally:
            release_os()
            self._write_lock.release()

    def write_once(self, path: Path, payload: bytes, *, failure_reserved: bool = False) -> None:
        self._write(path, payload, append=False, failure_reserved=failure_reserved)

    def append(self, path: Path, payload: bytes, *, failure_reserved: bool = False) -> None:
        self._write(path, payload, append=True, failure_reserved=failure_reserved)

    def atomic_json(self, path: Path, value: Mapping[str, Any], *,
                    failure_reserved: bool = False) -> None:
        payload = (canonical_json(value) + "\n").encode("utf-8")
        target = Path(path)
        target_relative = self._relative(target)
        if target_relative not in {"run_manifest.json", "supervisor_status.json"}:
            raise OutputBudgetError("Atomic JSON replacement is limited to status manifests")
        temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
        self._relative(temp)
        release_os = self._acquire_write_guards()
        try:
            current_bytes = self._audit_tree()
            if current_bytes > self.max_bytes:
                raise OutputBudgetError("Existing output already exceeds its authorized byte budget")
            limit = self.max_bytes if failure_reserved else self.normal_limit
            if current_bytes + len(payload) > limit:
                raise OutputBudgetError(f"Manifest replacement refused before I/O: {target_relative}")
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            if hasattr(os, "O_BINARY"):
                flags |= os.O_BINARY
            fd = os.open(temp, flags, 0o600)
            try:
                with os.fdopen(fd, "wb", closefd=False) as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                os.close(fd)
            os.replace(temp, target)
            if self._audit_tree() > limit:
                raise OutputBudgetError("Observed output exceeds the pre-write manifest budget")
        finally:
            release_os()
            self._write_lock.release()

    def write_supervisor_record(self, record: Mapping[str, Any]) -> None:
        self.ensure_root()
        self.atomic_json(self.root / "supervisor_status.json", record,
                         failure_reserved=True)


def _budget_for_path(path: Path) -> OutputBudget | None:
    candidate = Path(os.path.abspath(path))
    with _ACTIVE_OUTPUT_BUDGETS_LOCK:
        budgets = sorted(_ACTIVE_OUTPUT_BUDGETS.values(),
                         key=lambda budget: len(str(budget.root)), reverse=True)
    for budget in budgets:
        try:
            candidate.relative_to(budget.root)
        except ValueError:
            continue
        return budget
    return None


class EvidenceTapeWriter:
    """Append-only event shards; restart is refused without a full state checkpoint."""
    def __init__(self, root: Path, *, max_bytes: int = MAX_OUTPUT_BYTES,
                 resume_empty_bootstrap: bool = False,
                 budget: OutputBudget | None = None,
                 allowed_parent: Path | None = None):
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        self.next_sealed = 0
        self.next_finalized = 0
        if self.root.exists() and not resume_empty_bootstrap:
            raise FileExistsError(f"R1 output namespace already exists: {self.root}")
        self.budget = budget or OutputBudget(
            self.root, max_bytes=max_bytes, allowed_parent=allowed_parent,
            allow_existing=resume_empty_bootstrap)
        if self.budget.root != Path(os.path.abspath(self.root)):
            raise OutputBudgetError("Evidence tape and output budget roots differ")
        if self.root.exists():
            entries = {path.name for path in self.root.iterdir()}
            manifest_path = self.root / "run_manifest.json"
            if entries != {"events", "run_manifest.json"} or not (self.root / "events").is_dir():
                raise TapeIntegrityError("Explicit resume found unexpected R1 output files")
            if any((self.root / "events").iterdir()):
                raise TapeIntegrityError("Explicit empty-bootstrap resume found event shards")
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (previous.get("schema") != "calibration.r1_evidence_tape.v1" or
                    previous.get("status") != "INCOMPLETE" or
                    previous.get("sealed_events") != 0 or
                    previous.get("finalized_events") != 0 or
                    previous.get("next_position") != 0 or
                    previous.get("failure_type") != "ModuleNotFoundError" or
                    "experiments" not in str(previous.get("failure_message", ""))):
                raise TapeIntegrityError("Existing R1 state is not the known empty import-bootstrap failure")
            history_path = self.root / "attempt_history.jsonl"
            _append_jsonl(history_path, {
                "attempt": 1,
                "outcome": "EMPTY_BOOTSTRAP_FAILURE_RETAINED",
                "failure_type": previous["failure_type"],
                "failure_message": previous["failure_message"],
                "sealed_events": 0,
                "finalized_events": 0,
            })
            self.manifest = dict(previous)
            for key in ("failure_type", "failure_message", "events_processed",
                        "parity_failures", "causal_state_mismatches"):
                self.manifest.pop(key, None)
            self.manifest.update({"status": "INCOMPLETE", "retry_count": 1,
                                  "resume_mode": "explicit_empty_bootstrap_only"})
            self.update_manifest({})
        else:
            self.root.mkdir(parents=True)
            (self.root / "events").mkdir()
            self.manifest: dict[str, Any] = {
                "schema": "calibration.r1_evidence_tape.v1",
                "status": "INCOMPLETE",
                "sealed_events": 0,
                "finalized_events": 0,
                "next_position": 0,
                "output_bytes": 0,
            }
            _write_once(self.root / "run_manifest.json",
                        (canonical_json(self.manifest) + "\n").encode("utf-8"))

    def update_manifest(self, updates: Mapping[str, Any]) -> None:
        self.manifest.update(dict(updates))
        self.manifest["output_bytes"] = _directory_bytes(self.root)
        self.manifest["output_budget"] = {
            "schema": "calibration.r1_output_budget.v1",
            "max_bytes": self.budget.max_bytes,
            "failure_reserve_bytes": self.budget.failure_reserve_bytes,
            "normal_write_limit_bytes": self.budget.normal_limit,
            "enforcement": "central pre-write accountant plus Windows mutex",
            "filesystem_quota_required_for_gpu": True,
        }
        _atomic_json(self.root / "run_manifest.json", self.manifest)

    def seal_event(self, position: int, arrays: Mapping[str, np.ndarray],
                   metadata: Mapping[str, Any]) -> dict[str, Any]:
        position = int(position)
        if position != self.next_sealed:
            raise TapeIntegrityError(
                f"Expected next sealed position {self.next_sealed}, received {position}")
        if _contains_forbidden_label_payload(metadata):
            raise TapeIntegrityError("Online score tape contains a label/mask payload")
        required = {"base_patch_map", "top32_ids", "top32_features",
                    "c1_delta", "c2_delta"}
        if set(arrays) != required:
            raise TapeIntegrityError("Event shard fields differ from the frozen tape schema")
        base = np.asarray(arrays["base_patch_map"], dtype=np.float32)
        ids = np.asarray(arrays["top32_ids"], dtype=np.int64)
        features = np.asarray(arrays["top32_features"], dtype=np.float32)
        d1 = np.asarray(arrays["c1_delta"], dtype=np.float32)
        d2 = np.asarray(arrays["c2_delta"], dtype=np.float32)
        if (base.ndim != 1 or len(base) != 784 or ids.shape != (32,) or
                features.ndim != 2 or features.shape[0] != 32 or
                d1.shape != (32,) or d2.shape != (32,)):
            raise TapeIntegrityError("Event shard has an unexpected tensor shape")
        if (np.any(ids < 0) or np.any(ids >= len(base)) or len(set(ids.tolist())) != 32 or
                not all(np.isfinite(x).all() for x in (base, features, d1, d2))):
            raise TapeIntegrityError("Event shard has invalid IDs or nonfinite values")
        expected_scores = {
            "c0_score": max_score(base),
            "c1_score": max_score(corrected_dense_map(base, ids, d1)),
            "c2_score": max_score(corrected_dense_map(base, ids, d2)),
        }
        for key, expected in expected_scores.items():
            if key not in metadata or float(metadata[key]) != expected:
                raise TapeIntegrityError(f"{key} does not reconstruct exactly from the dense map")
        array_payload = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
        shard_bytes = deterministic_npz_bytes(array_payload)
        shard_name = f"events/event_{position:06d}.npz"
        _write_once(self.root / shard_name, shard_bytes)
        record = dict(metadata)
        record.update({
            "position": position,
            "array_fields": sorted(array_payload),
            "array_sha256": arrays_sha256(array_payload),
            "shard_path": shard_name,
            "shard_sha256": sha256_bytes(shard_bytes),
        })
        if _contains_forbidden_label_payload(record):
            raise TapeIntegrityError("Sealed event metadata contains a label/mask payload")
        _append_jsonl(self.root / "sealed_scores.jsonl", record)
        self.next_sealed += 1
        self.update_manifest({"sealed_events": self.next_sealed,
                              "next_position": self.next_sealed})
        return record

    def finalize_event(self, position: int, metadata: Mapping[str, Any]) -> dict[str, Any]:
        position = int(position)
        if position >= self.next_sealed:
            raise TapeIntegrityError("Cannot finalize an event before its scores are sealed")
        if position != self.next_finalized:
            raise TapeIntegrityError(
                f"Expected next finalized position {self.next_finalized}, received {position}")
        if _contains_forbidden_label_payload(metadata):
            raise TapeIntegrityError("Evidence tape contains a label/mask payload")
        record = dict(metadata)
        record["position"] = position
        _append_jsonl(self.root / "event_transitions.jsonl", record)
        self.next_finalized += 1
        self.update_manifest({"finalized_events": self.next_finalized})
        return record

    def verify(self, *, require_complete: bool = True) -> dict[str, Any]:
        sealed_path = self.root / "sealed_scores.jsonl"
        final_path = self.root / "event_transitions.jsonl"
        sealed = ([json.loads(line) for line in sealed_path.read_text(encoding="utf-8").splitlines()]
                  if sealed_path.exists() else [])
        finalized = ([json.loads(line) for line in final_path.read_text(encoding="utf-8").splitlines()]
                     if final_path.exists() else [])
        sealed_positions = [int(row["position"]) for row in sealed]
        finalized_positions = [int(row["position"]) for row in finalized]
        if sealed_positions != list(range(len(sealed_positions))):
            raise TapeIntegrityError("Sealed event positions have a duplicate or gap")
        if finalized_positions != list(range(len(finalized_positions))):
            raise TapeIntegrityError("Final transition positions have a duplicate or gap")
        if len(finalized) > len(sealed):
            raise TapeIntegrityError("More transitions than sealed score events")
        if require_complete and len(sealed) != len(finalized):
            raise TapeIntegrityError("A sealed score event lacks a post-feedback transition")
        for row in sealed:
            shard = self.root / str(row["shard_path"])
            if not shard.is_file() or sha256_file(shard) != row["shard_sha256"]:
                raise TapeIntegrityError("Event shard missing or checksum mismatch")
            with np.load(shard, allow_pickle=False) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
            if arrays_sha256(arrays) != row["array_sha256"]:
                raise TapeIntegrityError("Event array checksum mismatch")
            ids = arrays["top32_ids"].astype(np.int64, copy=False)
            if (float(row["c0_score"]) != max_score(arrays["base_patch_map"]) or
                    float(row["c1_score"]) != max_score(corrected_dense_map(
                        arrays["base_patch_map"], ids, arrays["c1_delta"])) or
                    float(row["c2_score"]) != max_score(corrected_dense_map(
                        arrays["base_patch_map"], ids, arrays["c2_delta"]))):
                raise TapeIntegrityError("Deterministic score replay does not match sealed scores")
        return {"sealed_events": len(sealed), "finalized_events": len(finalized),
                "sealed_positions_sha256": sha256_json(sealed_positions),
                "transition_positions_sha256": sha256_json(finalized_positions),
                "passed": len(sealed) == len(finalized) if require_complete else True}


def _validate_branch_and_start(*, expected_source_commit: str,
                               profile_id: str,
                               support_only: bool) -> dict[str, str]:
    branch = subprocess.check_output(["git", "branch", "--show-current"],
                                     cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=ROOT, text=True).strip()
    identity = _validate_execution_identity(
        branch=branch, head=head,
        expected_source_commit=expected_source_commit,
        profile_id=profile_id, support_only=support_only)
    tracked_changes = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True).strip()
    if tracked_changes:
        raise R1BlockedInput(
            "Source-bound R1 execution requires a clean tracked worktree")
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=ROOT, text=True).splitlines()
    critical_exact = {
        "test_phase3_AIF_simplified_v22.py", "vmb_visionad_new.py",
        "experiments/p0_protocol_audit.py", "experiments/v02_memory.py",
    }
    untracked_sources = [path for path in untracked
                         if path in critical_exact or
                         (path.endswith(".py") and
                          (path.startswith("experiments/calibration/") or
                           path.startswith("experiments/risk_safe/")))]
    if untracked_sources:
        raise R1BlockedInput(
            "Untracked Python source can shadow the bound R1 imports: " +
            ", ".join(sorted(untracked_sources)[:8]))
    return identity


def _preflight(dataset_root: Path, json_root: Path, *,
               expected_source_commit: str, profile_id: str,
               support_only: bool) -> dict[str, Any]:
    git_identity = _validate_branch_and_start(
        expected_source_commit=expected_source_commit,
        profile_id=profile_id, support_only=support_only)
    from experiments.calibration.profiles import NativeScorerConfig, QueryProtocolConfig
    from experiments.calibration.stage05 import _absolute_image_path
    from experiments.calibration.stage1a_p11_production import verify_stage05_qats_artifacts

    scorer_profile = NativeScorerConfig()
    query_profile = QueryProtocolConfig()
    tapes = verify_stage05_qats_artifacts(STAGE05)
    if tapes.binding_sha256 != QUERY_BINDING_SHA256:
        raise R1BlockedInput("Frozen Q-A-TS binding differs from the authorization")
    binding = next((dict(row) for row in tapes.bindings
                    if row.get("class_identity") == "switch"), None)
    if not binding or binding.get("query_protocol_sha256") != QUERY_PROTOCOL_SHA256:
        raise R1BlockedInput("Switch Query Tape identity is missing or incompatible")

    frozen_path = STAGE05 / "frozen_inputs.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("test_labels_accessed") is not False or frozen.get("test_masks_accessed") is not False:
        raise R1BlockedInput("Frozen Stage 0.5 provenance reports prior test-label or mask access")
    info = frozen["classes"]["switch"]
    stream = info["stream"]
    if len(stream) < MAX_EVENTS or len(stream) != int(binding["n_images"]):
        raise R1BlockedInput("Frozen switch stream is shorter than the authorized prefix")
    if len(info["supports"]["0"]["images"]) != 4:
        raise R1BlockedInput("Frozen support seed 0 does not contain exactly four supports")
    if int(info["c64"]["count"]) != 64:
        raise R1BlockedInput("Frozen protected C64 manifest is not exactly 64 references")
    if not info.get("support_c64_disjoint") or not info.get("train_test_groups_disjoint"):
        raise R1BlockedInput("Frozen support/C64 group-disjointness receipt failed")

    prefix = stream[:MAX_EVENTS]
    for position, row in enumerate(prefix):
        if int(row.get("position", -1)) != position:
            raise R1BlockedInput("Frozen stream position sequence is not the authorized prefix")
        if not isinstance(row.get("timestamp"), str) or not row["timestamp"]:
            raise R1BlockedInput(f"Missing source timestamp at event position {position}")
        sample_key = str(row["sample_key"])
        group = opaque_sample_group(sample_key)
        expected_group = tapes.group_identity_by_class_position["switch"].get(position)
        if row.get("opaque_group") != group or (expected_group is not None and expected_group != group):
            raise R1BlockedInput(f"Frozen stream/query group mismatch at event {position}")

    query_positions = tuple(position for position in tapes.positions("switch")
                            if 0 <= int(position) < MAX_EVENTS)
    query_set = frozenset(int(position) for position in query_positions)
    if any(position not in range(MAX_EVENTS) for position in query_set):
        raise R1BlockedInput("Query position falls outside the authorized event prefix")
    group_map = {position: str(tapes.group_identity_by_class_position["switch"][position])
                 for position in query_set}
    if len(set(group_map.values())) != len(group_map):
        raise R1BlockedInput("Query Tape contains duplicate official source groups in the prefix")

    support_groups = set(opaque_sample_group(x) for x in info["supports"]["0"]["groups"])
    c64_groups = set(opaque_sample_group(x) for x in info["c64"]["groups"])
    stream_groups = set(str(row["opaque_group"]) for row in prefix)
    if (support_groups & stream_groups or c64_groups & stream_groups or
            support_groups & c64_groups):
        raise R1BlockedInput("Support/C64/query-stream source-group overlap detected")

    support_paths = [Path(_absolute_image_path(dataset_root, info["prefix"], str(relative)))
                     for relative in info["supports"]["0"]["images"]]
    path_records: dict[str, Path] = {}
    for relative, path in zip(info["supports"]["0"]["images"], support_paths):
        path_records["support:" + sha256_bytes(str(relative).encode())] = path
    for position, row in enumerate(prefix):
        path_records[f"stream:{position}"] = Path(
            _absolute_image_path(dataset_root, info["prefix"], str(row["relative_id"])))
    for role, path in path_records.items():
        if not path.is_file():
            raise R1BlockedInput(f"Authorized local image file is missing for {role}")
    label_json = json_root / "switch.json"
    if not label_json.is_file():
        raise R1BlockedInput("Authorized queried-position label source is missing")

    score_manifest = json.loads((STAGE05 / "score_tape_manifest.json").read_text(
        encoding="utf-8"))
    c64_row = score_manifest["classes"]["switch"]["score_tapes"]["0"]
    c64_path = STAGE05 / str(c64_row["file"])
    if (c64_row.get("c64_score_count") != 64 or
            not c64_row.get("native_parity", {}).get("passed") or
            sha256_file(c64_path) != c64_row["sha256"]):
        raise R1BlockedInput("Frozen C64 score receipt is incomplete or checksum-invalid")
    with np.load(c64_path, allow_pickle=False) as archive:
        c64_scores = archive["c64_scores"].astype(np.float64, copy=True)
        c64_identity = str(archive["c64_identity_sha256"].item())
    if (len(c64_scores) != 64 or not np.isfinite(c64_scores).all() or
            c64_identity != info["c64"]["image_sha256"]):
        raise R1BlockedInput("Frozen C64 score/identity count differs from the protected manifest")
    if c64_row["c64_group_sha256"] != info["c64"]["group_sha256"]:
        raise R1BlockedInput("Frozen C64 score receipt has a different group identity")
    from experiments.calibration.controllers import theta0_from_c64
    theta0 = float(theta0_from_c64(c64_scores.tolist()))
    sigma0 = max(float(c64_scores.std(ddof=1)), 0.02)

    return {
        "git": git_identity,
        "scorer_profile": asdict(scorer_profile),
        "scorer_profile_sha256": scorer_profile.sha256,
        "query_profile": asdict(query_profile),
        "query_profile_config_sha256": query_profile.sha256,
        "query_binding_sha256": tapes.binding_sha256,
        "query_verification_sha256": tapes.verification_sha256,
        "query_binding": binding,
        "query_positions": sorted(query_set),
        "query_positions_sha256": sha256_json(sorted(query_set)),
        "query_positions_prefix_count": len(query_set),
        "query_group_sha256_by_position": group_map,
        "frozen_inputs_sha256": sha256_file(frozen_path),
        "class_json_sha256": str(info["class_json_sha256"]),
        "stream_count": len(stream),
        "stream_prefix_group_count": len(stream_groups),
        "stream_prefix_timestamp_policy": "frozen event position is causal order; preserve and verify source timestamps, do not require monotonic timestamps",
        "support_paths": support_paths,
        "stream_metadata": prefix,
        "prefix": str(info["prefix"]),
        "support_group_sha256": sha256_json(sorted(support_groups)),
        "c64_group_sha256": str(info["c64"]["group_sha256"]),
        "c64_image_sha256": str(info["c64"]["image_sha256"]),
        "c64_score_tape_sha256": str(c64_row["sha256"]),
        "c64_score_identity_sha256": c64_identity,
        "c64_theta0_higher_q90": theta0,
        "c64_sigma0_with_floor_0_02": sigma0,
        "fixed_support_sha256_expected": str(c64_row["fixed_support_sha256"]),
        "dataset_root": str(dataset_root),
        "json_root": str(json_root),
        "label_json_path": label_json,
        "label_json_expected_sha256": str(info["class_json_sha256"]),
    }


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(value.shape), str(value.dtype))).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _hub_source_sha256(torch_hub_dir: Path) -> tuple[str, str]:
    candidates = sorted(path for path in torch_hub_dir.glob("facebookresearch_dinov2*")
                        if path.is_dir() and (path / "hubconf.py").is_file())
    if not candidates:
        raise R1BlockedInput("Cached DINOv2 torch.hub source is unavailable; network fetch is not authorized")
    source = candidates[0]
    entries = []
    for path in sorted(source.rglob("*.py")):
        entries.append((str(path.relative_to(source)).replace("\\", "/"), sha256_file(path)))
    return str(source), sha256_json(entries)


def _native_dense_score(impl: Any, bank: Any, image: Any,
                        features: Sequence[torch.Tensor], cfg: Any
                        ) -> tuple[np.ndarray, np.ndarray, float, Mapping[str, Any]]:
    import torch
    if bank.corr_cfg.enabled or bank.use_defect_bank:
        raise R1ParityError("Common C0 scorer requires correction and defect banks disabled")
    s_maps, summary = bank.score_single_view(
        cls_name="p0", feats_per_layer=list(features), mode=0, view="id",
        topk_patches=cfg.topk_patches, mode_scoring="boost", lam=0.8,
        tau_close=0.20, gamma=1.0)
    if len(s_maps) != 1:
        raise R1ParityError("Locked concat profile must expose exactly one fused feature layer")
    ids = np.asarray(summary["layer0"]["topk_patch_idx"], dtype=np.int64)
    base_map = s_maps[0].detach().float().to("cpu").contiguous().numpy().astype(
        np.float32, copy=True)
    if base_map.shape != (784,) or ids.shape != (32,):
        raise R1ParityError(f"Unexpected native dense map/top-k shapes: {base_map.shape}/{ids.shape}")
    if not np.isfinite(base_map).all() or len(set(ids.tolist())) != 32:
        raise R1ParityError("Native dense map is nonfinite or Top-32 IDs are invalid")
    aspect = float(image.height) / float(image.width) if image.width else None
    h, w = impl._grid_hw_from_npatches(784, aspect=aspect)
    fused = bank.fuse_layers(s_maps, fusion="mean")
    aligned = impl.align_map_to_identity(
        fused, impl.ViewSpec("id", "id"), int(h), int(w))
    score = float(bank.aggregate_map(aligned, agg=cfg.image_aggregation,
                                     topk=cfg.image_topk,
                                     quantile=cfg.image_quantile))
    if not math.isfinite(score):
        raise R1ParityError("Native adapter produced a nonfinite C0 score")
    if score != max_score(base_map):
        raise R1ParityError(
            "Native image-level max does not reconstruct exactly from the dense patch map")
    return base_map, ids, score, summary


def _native_reference_and_debug(impl: Any, bank: Any, image: Any,
                                features: Sequence[torch.Tensor], cfg: Any
                                ) -> tuple[float, np.ndarray, np.ndarray]:
    from experiments.a0_a1_strict4_matched import score_bank
    score, amap = score_bank(impl, bank, image, list(features), cfg)
    candidate_map = impl.layer_to_topk_from_amap(amap)
    ids = np.asarray(candidate_map.get(0, []), dtype=np.int64)
    debug = bank.score_sample_correction_debug_maps(
        cls_name="p0", view_to_feats={"id": list(features)}, mode=0,
        views={"id": impl.ViewSpec("id", "id")},
        topk_patches=cfg.topk_patches, mode_scoring="boost",
        lam=0.8, tau_close=0.20, gamma=1.0,
        layer_fusion="mean", view_fusion="mean",
        img_agg=cfg.image_aggregation, img_topk=cfg.image_topk,
        img_quantile=cfg.image_quantile, concept_assign_view="id",
        img_hw=(image.height, image.width))
    dense = np.asarray(debug["A_pre"], dtype=np.float32).reshape(-1)
    return float(score), ids, dense.copy()


def _memory_content_sha256(bank: Any) -> str:
    from experiments.risk_safe.native_bridge import snapshot
    return snapshot(bank, version=0).sha256()


def _memory_state_sha256(bank: Any, version: int) -> str:
    from experiments.risk_safe.native_bridge import snapshot
    return snapshot(bank, version=int(version)).sha256()


def _correction_state_sha256(bank: Any,
                             source_groups: Mapping[int, set[str]]) -> str:
    cb = bank.corr_cb["p0"]["id"][0]
    n = int(cb.size())
    digest = hashlib.sha256()
    for name in ("protos", "beta", "radius", "kappa", "count",
                 "outcome_fp", "outcome_fn", "outcome_tn", "outcome_tp", "last_step"):
        value = getattr(cb, name)[:n].detach().to("cpu").contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str((tuple(value.shape), str(value.dtype))).encode("ascii"))
        digest.update(value.numpy().tobytes())
    digest.update(canonical_json({str(k): sorted(v) for k, v in sorted(source_groups.items())}).encode())
    return digest.hexdigest()


def _validate_c1_source_groups(source_groups: Mapping[int, set[str]],
                               last_steps: Sequence[int], *, event_position: int,
                               phase: str, event_group: str | None = None) -> list[int]:
    """Validate active C1 slot provenance; replacement history stays conservative."""
    event_position = _causal_position(event_position, "current event position")
    if phase not in ("pre_score", "post_update"):
        raise ValueError(f"Unknown C1 anchor validation phase: {phase}")
    if phase == "pre_score":
        event_group = _causal_group_sha256(event_group, "current source group")
    positions = [_causal_position(value, f"C1 slot {slot} last_step")
                 for slot, value in enumerate(last_steps)]
    active_count = len(positions)
    slot_keys: set[int] = set()
    for raw_slot in source_groups:
        if (isinstance(raw_slot, (bool, np.bool_)) or
                not isinstance(raw_slot, (int, np.integer)) or int(raw_slot) < 0):
            raise R1CausalityError("C1 source-group provenance has a malformed slot key")
        slot_keys.add(int(raw_slot))
    if any(slot >= active_count for slot in slot_keys):
        raise R1CausalityError("C1 source-group provenance references an inactive slot")
    eligible: list[int] = []
    for slot, last_step in enumerate(positions):
        if phase == "pre_score" and last_step >= event_position:
            raise R1CausalityError(
                f"C1 pre-score slot {slot} has current/future last_step {last_step} "
                f"at event {event_position}")
        if phase == "post_update" and last_step > event_position:
            raise R1CausalityError(
                f"C1 post-update slot {slot} has future last_step {last_step} "
                f"after event {event_position}")
        history = source_groups.get(slot)
        if not isinstance(history, (set, frozenset)) or not history:
            raise R1CausalityError(f"Active C1 slot {slot} lacks source-group provenance")
        for value in history:
            _causal_group_sha256(value, f"C1 slot {slot} source-group history")
        if phase == "pre_score" and event_group not in history:
            eligible.append(slot)
    return eligible


def _record_c1_source_group_updates(source_groups: dict[int, set[str]], *,
                                    last_steps: Sequence[int], event_position: int,
                                    event_group: str) -> None:
    """Union current provenance for touched slots; replacement is intentionally overinclusive."""
    event_position = _causal_position(event_position, "current event position")
    event_group = _causal_group_sha256(event_group, "current source group")
    positions = [_causal_position(value, f"C1 slot {slot} last_step")
                 for slot, value in enumerate(last_steps)]
    active_count = len(positions)
    for raw_slot in list(source_groups):
        if (isinstance(raw_slot, (bool, np.bool_)) or
                not isinstance(raw_slot, (int, np.integer)) or int(raw_slot) < 0):
            raise R1CausalityError("C1 source-group provenance has a malformed slot key")
        if int(raw_slot) >= active_count:
            raise R1CausalityError("C1 source-group provenance references an inactive slot")
    for slot, last_step in enumerate(positions):
        if last_step > event_position:
            raise R1CausalityError("C1 native Codebook reports an update from the future")
        if last_step == event_position:
            history = source_groups.get(slot)
            if history is None:
                history = set()
                source_groups[slot] = history
            if not isinstance(history, set):
                raise R1CausalityError(f"C1 slot {slot} provenance is not mutable exact-set state")
            for value in history:
                _causal_group_sha256(value, f"C1 slot {slot} prior source-group history")
            history.add(event_group)
    _validate_c1_source_groups(source_groups, positions,
                               event_position=event_position, phase="post_update")


def _filtered_c1_codebook(impl: Any, bank: Any, source_groups: Mapping[int, set[str]],
                          current_group: str, *, event_position: int
                          ) -> tuple[Any, list[int]]:
    import torch
    live = bank.corr_cb["p0"]["id"][0]
    n = int(live.size())
    eligible = _validate_c1_source_groups(
        source_groups, live.last_step[:n].detach().to("cpu").tolist(),
        event_position=event_position, phase="pre_score", event_group=current_group)
    cfg = bank.corr_cfg
    cfg_copy = type(cfg)(**asdict(cfg))
    cfg_copy.enabled = True
    filtered = impl.SignedCorrectionCodebook(
        max(1, len(eligible)), int(live.protos.shape[1]), live.device, cfg_copy)
    filtered.n = len(eligible)
    if eligible:
        slots = torch.as_tensor(eligible, device=live.device, dtype=torch.long)
        for name in ("protos", "beta", "radius", "kappa", "count",
                     "outcome_fp", "outcome_fn", "outcome_tn", "outcome_tp", "last_step"):
            getattr(filtered, name)[:len(eligible)].copy_(getattr(live, name)[slots])
    return filtered, eligible


def _c1_score_delta(impl: Any, bank: Any, selected_features: torch.Tensor,
                    source_groups: Mapping[int, set[str]], current_group: str,
                    *, event_position: int
                    ) -> tuple[np.ndarray, dict[str, Any]]:
    filtered, eligible = _filtered_c1_codebook(
        impl, bank, source_groups, current_group, event_position=event_position)
    if not eligible:
        return np.zeros((32,), dtype=np.float32), {
            "eligible_anchor_slots": 0, "same_source_excluded_anchor_slots": int(bank.corr_cb["p0"]["id"][0].size()),
            "far_field_patch_count": 0,
            "far_field_abs_correction_max": 0.0,
        }
    q = impl.l2_normalize(selected_features.to(bank.device))
    delta = filtered.apply(q).detach().float().to("cpu").numpy().astype(np.float32, copy=True)
    dmat = filtered._cos_dist(q, filtered.protos[:filtered.n])
    nearest_dist, nearest_slot = dmat.min(dim=1)
    nearest_radius = filtered.radius[:filtered.n][nearest_slot].clamp_min(1e-6)
    far = nearest_dist > (3.0 * nearest_radius)
    far_values = np.abs(delta[far.detach().to("cpu").numpy()]) if bool(far.any()) else np.zeros(0)
    return delta, {
        "eligible_anchor_slots": len(eligible),
        "same_source_excluded_anchor_slots": int(bank.corr_cb["p0"]["id"][0].size()) - len(eligible),
        "far_field_patch_count": int(far.sum().item()),
        "far_field_abs_correction_max": float(far_values.max()) if len(far_values) else 0.0,
    }


def _c2_state_sha256(anchors: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json([
        {"position": int(anchor["position"]),
         "group_sha256": str(anchor["group_sha256"]),
         "patch_id": int(anchor["patch_id"]), "sign": int(anchor["sign"]),
         "feature_sha256": str(anchor["feature_sha256"])}
        for anchor in anchors])


def _source_hashes() -> dict[str, str]:
    files = [
        "test_phase3_AIF_simplified_v22.py", "vmb_visionad_new.py",
        "experiments/v02_memory.py", "experiments/risk_safe/native_bridge.py",
        "experiments/a0_a1_strict4_matched.py", "experiments/p0_protocol_audit.py",
        "experiments/calibration/native.py",
        "experiments/calibration/r1_matched_correction_pilot.py",
        "experiments/calibration/r1_runtime_supervisor.py",
        "experiments/calibration/profiles.py",
        "experiments/calibration/stage1a_formal_comparison.py",
        "experiments/calibration/stage05.py",
        "experiments/calibration/stage05_runner.py",
        "experiments/calibration/stage1a_p11_production.py",
        "experiments/calibration/stage1a_p12_label_provider.py",
    ]
    return {path: sha256_file(ROOT / path) for path in files}


def _process_rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _append_private_feedback(root: Path, record: Mapping[str, Any]) -> str:
    """Restricted untracked response ledger; never referenced in the online tape payload."""
    path = root / "private_feedback.jsonl"
    payload = (canonical_json(record) + "\n").encode("utf-8")
    _append_jsonl(path, record)
    return sha256_bytes(payload)


def _require_supervised_authorization(authorization_id: str) -> None:
    if (os.environ.get("R1_PILOT_SUPERVISED_CHILD") != "1" or
            os.environ.get("R1_PILOT_SUPERVISOR_AUTHORIZATION_ID") != authorization_id or
            not re.fullmatch(r"[0-9a-f]{64}",
                             os.environ.get("R1_PILOT_SUPERVISOR_NONCE", ""))):
        raise R1BlockedInput(
            "Real R1 execution must be launched by the explicit-authorization supervisor")
    try:
        from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job
        if not current_process_is_in_job():
            raise R1BlockedInput("R1 child is not contained in a Windows Job Object")
    except R1BlockedInput:
        raise
    except Exception as exc:
        raise R1BlockedInput(f"Cannot verify supervised Windows process containment: {exc}") from exc


def run_pilot(*, expected_source_commit: str, profile_id: str,
              run_id: str, authorization_id: str,
              support_only: bool = False,
              dataset_root: Path = DEFAULT_DATASET_ROOT,
              json_root: Path = DEFAULT_JSON_ROOT) -> dict[str, Any]:
    _require_supervised_authorization(authorization_id)
    config = PilotConfig()
    _validate_run_identity_fields(run_id, authorization_id)
    expected_profile = SUPPORT_ONLY_PROFILE_ID if support_only else R1_PROFILE_ID
    if profile_id != expected_profile:
        raise R1BlockedInput(
            f"Incompatible R1 profile: expected {expected_profile}; found {profile_id}")
    output = _output_namespace(run_id, support_only=support_only)
    if output.exists():
        raise FileExistsError(
            f"Versioned R1 output namespace already exists and cannot be resumed: {output}")
    budget = OutputBudget(
        output, max_bytes=config.max_output_bytes,
        allowed_parent=ROOT / "outputs" / "calibration")
    tape = EvidenceTapeWriter(output, max_bytes=config.max_output_bytes,
                              budget=budget,
                              allowed_parent=ROOT / "outputs" / "calibration")
    tape.update_manifest({
        "task": "R1-D1 support-only verification" if support_only else "R1",
        "status": "INCOMPLETE",
        "run_id": run_id,
        "authorization_id": authorization_id,
        "profile_id": profile_id,
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "expected_source_commit": expected_source_commit,
        "support_only": bool(support_only),
        "historical_numerical_equivalence_claimed": False,
    })
    preflight: dict[str, Any] = {}
    started = time.monotonic()
    pilot_execution_started: float | None = None
    parity_count = 0
    query_count = 0
    query_labels_decoded = 0
    positive_anchor_writes = 0
    negative_anchor_writes = 0
    c1_active_positive = c1_active_negative = 0
    c2_active_anchors = 0
    normal_memory_writes = 0
    causal_mismatches = 0
    c1_far_field_count = 0
    c1_far_field_abs_max = 0.0
    source_exclusion_checks = 0
    encode_seconds = native_seconds = correction_score_seconds = 0.0
    correction_update_seconds = writer_seconds = 0.0
    peak_rss = 0
    peak_cuda_allocated = peak_cuda_reserved = 0
    c1_source_groups: dict[int, set[str]] = {}
    c2_anchors: list[dict[str, Any]] = []
    memory_version = 0
    response_chain = "0" * 64
    try:
        import torch
        from experiments import p0_protocol_audit as p0
        from experiments.calibration import stage05
        from experiments.calibration.profiles import NativeScorerConfig, QueryProtocolConfig
        from experiments.calibration.native import _fixed_bank_digest
        from experiments.risk_safe.native_bridge import dynamic_codebooks
        from v02_memory import MemoryArm

        preflight = _preflight(
            dataset_root, json_root,
            expected_source_commit=expected_source_commit,
            profile_id=profile_id, support_only=support_only)
        impl = p0.require_sources()
        runtime_profile = _configure_historical_runtime(impl, torch)
        if not torch.cuda.is_available():
            raise R1BlockedInput("Authorized CUDA device is unavailable")
        torch.cuda.reset_peak_memory_stats()
        scorer_profile = NativeScorerConfig()
        query_profile = QueryProtocolConfig()
        cfg = p0.Config(
            device=config.device, augmentation_seed=0, bank_seed=0,
            fixed_capacity=25000, fixed_per_image=5000, augmentations=4,
            include_identity=False, multiplier=2.5, feature_fuse="concat",
            topk_patches=32, image_aggregation="max", image_topk=64,
            image_quantile=0.95, knn=5)
        _validate_p0_profile(cfg, scorer_profile)
        execution_profile_sha256 = _execution_profile_sha256(
            config, scorer_profile, query_profile, profile_id, support_only)

        torch_hub_dir = Path(torch.hub.get_dir())
        hub_source_path, hub_source_sha256 = _hub_source_sha256(torch_hub_dir)
        checkpoint = torch_hub_dir / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
        if not checkpoint.is_file():
            raise R1BlockedInput(
                "Cached DINOv2 checkpoint is unavailable; network download is disabled")
        checkpoint_sha256 = sha256_file(checkpoint)
        source_hashes = _source_hashes()
        tape.update_manifest({
            "branch": preflight["git"]["branch"],
            "source_commit": preflight["git"]["source_commit"],
            "execution_profile_sha256": execution_profile_sha256,
            "native_scorer_profile_sha256": scorer_profile.sha256,
            "query_protocol_profile_sha256": query_profile.sha256,
            "runtime_profile": runtime_profile,
            "source_hashes": source_hashes,
            "dino_hub_source_path_name": Path(hub_source_path).name,
            "hub_source_tree_sha256": hub_source_sha256,
            "checkpoint_file_name": checkpoint.name,
            "checkpoint_sha256": checkpoint_sha256,
            "fixed_support_sha256_expected": preflight[
                "fixed_support_sha256_expected"],
            "fixed_support_digest_algorithm": STRUCTURED_FIXED_BANK_DIGEST_ALGORITHM,
            "historical_numerical_equivalence_claimed": False,
        })

        torch.cuda.synchronize()
        dino_started = time.perf_counter()
        pilot_execution_started = time.monotonic()
        backbone = impl.DINOv2MultiLayerBackbone(device=config.device)
        weight_state_sha256 = _model_state_sha256(backbone.model)
        tape.update_manifest({"loaded_model_state_sha256": weight_state_sha256})
        encoder_fingerprint = sha256_json({
            "model_name": scorer_profile.model_name,
            "loaded_model_state_sha256": weight_state_sha256,
            "hub_source_path": Path(hub_source_path).name,
            "hub_source_tree_sha256": hub_source_sha256,
            "vmb_source_sha256": source_hashes["vmb_visionad_new.py"],
            "v22_encoder_source_sha256": source_hashes["test_phase3_AIF_simplified_v22.py"],
            "hook_layers": list(backbone.layers),
        })
        preprocess_fingerprint = sha256_json({
            "profile": asdict(scorer_profile), "rgb": "PIL.convert(RGB)",
            "resize": [448, 448], "crop": [392, 392],
            "interpolation": "PIL.BICUBIC", "normalization": "model mean/std",
            "fusion": "concat_then_l2_normalize", "view": "id",
            "pillow_version": __import__("PIL").__version__,
        })
        support_features = p0.encode_reference_features(
            impl, backbone, [str(path) for path in preflight["support_paths"]], cfg)
        chosen = [str(path) for path in preflight["support_paths"]]
        bank = p0.build_vmb(impl, support_features, chosen, cfg)
        def persist_fixed_digest_evidence(evidence: Mapping[str, Any]) -> None:
            tape.update_manifest({
                "fixed_bank_digest_validation": dict(evidence),
                "fixed_support_sha256": evidence["actual_structured_sha256"],
                "fixed_support_digest_algorithm": evidence["actual_structured_algorithm"],
                "fixed_support_raw_prototype_sha256": evidence[
                    "actual_raw_prototype_sha256"],
                "fixed_support_raw_digest_algorithm": evidence[
                    "actual_raw_prototype_algorithm"],
                "fixed_support_layer_profile": evidence["layer_profile"],
            })

        fixed_digest_evidence = _check_fixed_bank_receipt(
            bank, p0, preflight["fixed_support_sha256_expected"],
            _fixed_bank_digest, persist_fixed_digest_evidence)
        fixed_profile = p0.fixed_bank_profile(bank)
        fixed_digest = str(fixed_digest_evidence["actual_structured_sha256"])
        fixed_raw_digest = str(fixed_digest_evidence["actual_raw_prototype_sha256"])
        if support_only:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - dino_started
            support_result = {
                "status": "SUPPORT_ONLY_VERIFIED",
                "task": "R1-D1 support-only fixed-bank verification",
                "events_processed": 0,
                "queried_labels_decoded": 0,
                "c64_rescored": False,
                "stream_images_decoded": 0,
                "fixed_support_sha256": fixed_digest,
                "fixed_support_digest_algorithm": STRUCTURED_FIXED_BANK_DIGEST_ALGORITHM,
                "fixed_support_raw_prototype_sha256": fixed_raw_digest,
                "fixed_support_raw_digest_algorithm": RAW_FIXED_BANK_DIGEST_ALGORITHM,
                "fixed_support_layer_profile": fixed_profile["layers"],
                "fixed_bank_digest_validation": fixed_digest_evidence,
                "support_and_model_seconds": elapsed,
                "loaded_model_state_sha256": weight_state_sha256,
                "hub_source_tree_sha256": hub_source_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "source_hashes": source_hashes,
                "historical_numerical_equivalence_claimed": False,
            }
            tape.update_manifest(support_result)
            backbone.cleanup()
            return support_result
        settings = SimpleNamespace(
            flat=256, stm=256, ltm=1, buffer=1, min_events=2,
            promote_every=16, cap_bytes=config.max_output_bytes)
        writer = MemoryArm("greedy_coreset", bank, settings, seed=0)
        if any(int(cb.size()) for cb in dynamic_codebooks(bank)):
            raise R1BlockedInput("Flat Greedy dynamic memory did not start empty")
        if any(int(cb.K) != 256 for cb in dynamic_codebooks(bank)):
            raise R1BlockedInput("Flat Greedy capacity differs from locked K=256")
        bank.corr_cfg.enabled = False
        bank.use_defect_bank = False
        c1_cb = bank.corr_cb["p0"]["id"][0]
        if c1_cb.size() != 0:
            raise R1BlockedInput("C1 correction state was not empty at the pilot start")
        if c1_cb.K != 256:
            raise R1BlockedInput("C1 correction capacity differs from the existing bounded default")
        torch.cuda.synchronize()
        support_and_model_seconds = time.perf_counter() - dino_started
        memory_content_digest = _memory_content_sha256(bank)
        memory_state_digest = _memory_state_sha256(bank, memory_version)
        feature_dims = int(support_features[chosen[0]][0][0].shape[-1])
        feature_grid = int(support_features[chosen[0]][0][0].shape[0])

        tape.update_manifest({
            "task": "R1", "authorization_scope": {
                "category": "switch", "support_seed": 0,
                "positions_inclusive": [0, 511], "max_events": 512,
                "query_protocol": "Q-A-TS-v1", "query_seed": 37,
                "query_rate": 0.03, "gpu_seconds_limit": MAX_GPU_SECONDS,
                "output_bytes_limit": MAX_OUTPUT_BYTES,
                "scientific_outcome_evaluation": False,
            },
            "branch": preflight["git"]["branch"],
            "source_commit": preflight["git"]["source_commit"],
            "source_hashes": source_hashes,
            "scorer_profile_sha256": preflight["scorer_profile_sha256"],
            "query_binding_sha256": preflight["query_binding_sha256"],
            "query_verification_sha256": preflight["query_verification_sha256"],
            "query_positions_sha256": preflight["query_positions_sha256"],
            "query_positions_in_prefix": preflight["query_positions_prefix_count"],
            "frozen_inputs_sha256": preflight["frozen_inputs_sha256"],
            "class_json_sha256": preflight["class_json_sha256"],
            "support_group_sha256": preflight["support_group_sha256"],
            "c64_group_sha256": preflight["c64_group_sha256"],
            "c64_image_sha256": preflight["c64_image_sha256"],
            "c64_score_tape_sha256": preflight["c64_score_tape_sha256"],
            "c64_theta0_higher_q90": preflight["c64_theta0_higher_q90"],
            "c64_sigma0_with_floor_0_02": preflight["c64_sigma0_with_floor_0_02"],
            "fixed_support_sha256": fixed_digest,
            "fixed_support_digest_algorithm": STRUCTURED_FIXED_BANK_DIGEST_ALGORITHM,
            "fixed_support_raw_prototype_sha256": fixed_raw_digest,
            "fixed_support_raw_digest_algorithm": RAW_FIXED_BANK_DIGEST_ALGORITHM,
            "fixed_support_layer_profile": fixed_profile["layers"],
            "encoder_fingerprint_sha256": encoder_fingerprint,
            "loaded_model_state_sha256": weight_state_sha256,
            "hub_source_tree_sha256": hub_source_sha256,
            "preprocess_fingerprint_sha256": preprocess_fingerprint,
            "feature_tensor_expected_shape": [feature_grid, feature_dims],
            "feature_tensor_expected_dtype": "float32",
            "fixed_theta": preflight["c64_theta0_higher_q90"],
            "writer_sigma": preflight["c64_sigma0_with_floor_0_02"],
            "method_identities": {
                "C0": {"name": "native baseline, no correction",
                       "score_source": "shared native dense patch map"},
                "C1": {"name": "C1-Greedy-adapted",
                       "historical_v22_faithful_replay": False,
                       "write_source": "existing native SignedCorrectionCodebook and update_corrections_from_topk",
                       "strength_scale": 1.0,
                       "outcome_reference": "common C0 score/prediction/fixed C64 threshold",
                       "source_group_exclusion": "filter active anchors by queried source-group provenance",
                       "reasoningbank_dependent_route": False},
                "C2": {"name": "signed cosine kNN interface fixture",
                       "neighbors": config.c2_neighbors,
                       "amplitude": config.c2_interface_scale,
                       "distance": "unit-normalized cosine distance; deterministic index tie break",
                       "capacity": C2_ANCHOR_CAPACITY, "retention": "FIFO",
                       "scientific_hyperparameters_approved": False,
                       "future_preregistration_required": [
                           "neighbor count", "correction amplitude", "distance normalization/locality",
                           "anchor capacity/retention", "equal supervision budget",
                           "simple baseline identity"]},
            },
            "output_payloads_include_private_feedback": True,
            "unqueried_labels_accessed": False,
            "masks_accessed": False,
            "confirmation_data_accessed": False,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        from experiments.calibration.stage1a_p12_label_provider import QueriedPositionLabelReader
        label_reader = QueriedPositionLabelReader(
            category="switch", path=preflight["label_json_path"],
            expected_source_sha256=preflight["label_json_expected_sha256"],
            positions=preflight["query_positions"],
            group_sha256_by_position=preflight["query_group_sha256_by_position"],
        )
        label_reader = OrderedQueriedLabelReader(
            label_reader, preflight["query_positions"])
        records = preflight["stream_metadata"]
        support_and_model_elapsed = support_and_model_seconds
        started = time.monotonic()
        tape.update_manifest({"status": "INCOMPLETE",
                              "support_and_model_seconds": support_and_model_elapsed})

        for position, meta in enumerate(records):
            gate = CausalEventOrder(position)
            elapsed = (time.monotonic() - pilot_execution_started
                       if pilot_execution_started is not None else time.monotonic() - started)
            if elapsed >= config.max_gpu_seconds:
                raise R1ResourceLimit("120-minute GPU execution wall-time ceiling reached")
            group_sha = str(meta["opaque_group"])
            image_path = Path(stage05._absolute_image_path(
                dataset_root, preflight["prefix"], str(meta["relative_id"])))
            image_sha = sha256_file(image_path)
            encode_started = time.perf_counter()
            torch.cuda.synchronize()
            image, features = p0.encode_query(impl, backbone, str(image_path), cfg)
            torch.cuda.synchronize()
            encode_seconds += time.perf_counter() - encode_started
            if len(features) != 1 or tuple(features[0].shape) != (784, 4096):
                raise R1ParityError(f"Unexpected feature shape at event {position}: "
                                    f"{[tuple(x.shape) for x in features]}")
            if features[0].dtype != torch.float32 or not torch.isfinite(features[0]).all().item():
                raise R1ParityError("Feature dtype or finiteness differs from the locked profile")
            feature_digest = tensor_sha256(features[0])
            feature_object_id = id(features[0])
            if position == 0:
                tape.update_manifest({
                    "feature_tensor_actual_shape": list(features[0].shape),
                    "feature_tensor_actual_dtype": str(features[0].dtype),
                })
            feature_key = sha256_json({
                "image_sha256": image_sha, "encoder_fingerprint_sha256": encoder_fingerprint,
                "preprocess_fingerprint_sha256": preprocess_fingerprint,
                "view": "id", "feature_shape": list(features[0].shape),
                "feature_dtype": str(features[0].dtype), "feature_sha256": feature_digest,
            })

            native_before = _memory_content_sha256(bank)
            native_started = time.perf_counter()
            base_map, top_ids, c0_score, summary = _native_dense_score(
                impl, bank, image, features, cfg)
            torch.cuda.synchronize()
            native_seconds += time.perf_counter() - native_started
            if _memory_content_sha256(bank) != native_before:
                raise R1CausalityError("Native dense-map extraction mutated Greedy memory")
            if not np.array_equal(top_ids, np.asarray(
                    summary["layer0"]["topk_patch_idx"], dtype=np.int64)):
                raise R1ParityError("Native Top-32 identity differs inside the adapter")

            if position < config.smoke_events:
                before_parity = _memory_content_sha256(bank)
                fixed_before_parity = _fixed_bank_digest(bank)
                fixed_raw_before_parity = p0.fixed_bank_profile(bank)["sha256"]
                torch.cuda.synchronize()
                ref_started = time.perf_counter()
                reference_score, reference_ids, reference_map = _native_reference_and_debug(
                    impl, bank, image, features, cfg)
                torch.cuda.synchronize()
                native_seconds += time.perf_counter() - ref_started
                if _memory_content_sha256(bank) != before_parity:
                    raise R1CausalityError("Native reference/debug score mutated Greedy memory")
                if (_fixed_bank_digest(bank) != fixed_before_parity or
                        fixed_before_parity != fixed_digest or
                        p0.fixed_bank_profile(bank)["sha256"] != fixed_raw_before_parity or
                        fixed_raw_before_parity != fixed_raw_digest):
                    raise R1CausalityError("Native parity replay mutated frozen fixed memory")
                if id(features[0]) != feature_object_id or tensor_sha256(features[0]) != feature_digest:
                    raise R1ParityError("Native parity replay changed the shared feature tensor")
                tolerance = 1e-5 + 1e-4 * abs(reference_score)
                if not np.array_equal(reference_ids, top_ids):
                    raise R1ParityError(f"Top-32 IDs differ at event {position}")
                if not np.array_equal(reference_map, base_map):
                    raise R1ParityError(f"Dense native map differs at event {position}")
                if abs(reference_score - c0_score) > tolerance:
                    raise R1ParityError(f"Native C0 score exceeds bridge tolerance at event {position}")
                parity_count += 1

            correction_score_started = time.perf_counter()
            normalized_selected = impl.l2_normalize(
                features[0].to(bank.device))[torch.as_tensor(
                    top_ids, device=bank.device, dtype=torch.long)]
            c1_delta, c1_diag = _c1_score_delta(
                impl, bank, normalized_selected, c1_source_groups, group_sha,
                event_position=position)
            c1_far_field_count += int(c1_diag["far_field_patch_count"])
            c1_far_field_abs_max = max(
                c1_far_field_abs_max, float(c1_diag["far_field_abs_correction_max"]))
            c2_delta, c2_eligible = signed_knn_delta(
                normalized_selected.detach().float().to("cpu").numpy(), c2_anchors,
                event_position=position, event_group=group_sha,
                neighbors=config.c2_neighbors, amplitude=config.c2_interface_scale)
            c2_expected_indices = source_disjoint_anchor_indices(
                [int(anchor["position"]) for anchor in c2_anchors],
                [str(anchor["group_sha256"]) for anchor in c2_anchors],
                event_position=position, event_group=group_sha)
            if len(c2_expected_indices) != c2_eligible:
                raise R1CausalityError("C2 source-group or future-anchor exclusion failed")
            source_exclusion_checks += 1
            correction_score_seconds += time.perf_counter() - correction_score_started
            corrected1 = corrected_dense_map(base_map, top_ids, c1_delta)
            corrected2 = corrected_dense_map(base_map, top_ids, c2_delta)
            c1_score, c2_score = max_score(corrected1), max_score(corrected2)
            if not (math.isfinite(c1_score) and math.isfinite(c2_score)):
                raise R1ParityError("Correction replay produced nonfinite image scores")
            theta = float(preflight["c64_theta0_higher_q90"])
            preds = {"c0": int(c0_score >= theta),
                     "c1": int(c1_score >= theta),
                     "c2": int(c2_score >= theta)}
            memory_before_version = memory_version
            memory_before_digest = memory_state_digest
            c1_before_digest = _correction_state_sha256(bank, c1_source_groups)
            c2_before_digest = _c2_state_sha256(c2_anchors)
            score_state_digest = sha256_json({
                "fixed_support_sha256": fixed_digest,
                "greedy_memory_version": memory_before_version,
                "greedy_memory_sha256": memory_before_digest,
                "scorer_profile_sha256": preflight["scorer_profile_sha256"],
            })
            event_id = sha256_json({"category": "switch", "position": position,
                                    "image_sha256": image_sha, "group_sha256": group_sha})
            arrays = {
                "base_patch_map": base_map.astype(np.float32, copy=False),
                "top32_ids": top_ids.astype(np.int64, copy=False),
                "top32_features": normalized_selected.detach().float().to("cpu").contiguous().numpy()
                .astype(np.float32, copy=True),
                "c1_delta": c1_delta.astype(np.float32, copy=False),
                "c2_delta": c2_delta.astype(np.float32, copy=False),
            }
            if arrays["top32_features"].shape != (32, 4096):
                raise R1ParityError("Stored selected feature rows have unexpected shape")
            metadata = {
                "event_id": event_id, "category": "switch", "support_seed": 0,
                "timestamp": str(meta["timestamp"]),
                "source_image_identity_sha256": sha256_bytes(str(meta["relative_id"]).encode()),
                "source_image_content_sha256": image_sha,
                "official_group_sha256": group_sha,
                "camera_id": int(meta["camera_id"]),
                "feature_key_sha256": feature_key,
                "feature_tensor_sha256": feature_digest,
                "encoder_fingerprint_sha256": encoder_fingerprint,
                "preprocess_fingerprint_sha256": preprocess_fingerprint,
                "greedy_memory_version_before": memory_before_version,
                "greedy_memory_sha256_before": memory_before_digest,
                "fixed_support_sha256": fixed_digest,
                "fixed_support_digest_algorithm": STRUCTURED_FIXED_BANK_DIGEST_ALGORITHM,
                "fixed_support_raw_prototype_sha256": fixed_raw_digest,
                "fixed_support_raw_digest_algorithm": RAW_FIXED_BANK_DIGEST_ALGORITHM,
                "score_state_sha256": score_state_digest,
                "c0_score": c0_score, "c1_score": c1_score, "c2_score": c2_score,
                "c0_prediction": preds["c0"], "c1_prediction": preds["c1"],
                "c2_prediction": preds["c2"],
                "c1_state_sha256_before": c1_before_digest,
                "c2_state_sha256_before": c2_before_digest,
                "c1_eligible_anchor_slots": int(c1_diag["eligible_anchor_slots"]),
                "c1_same_source_excluded_anchor_slots": int(c1_diag["same_source_excluded_anchor_slots"]),
                "c2_eligible_anchor_count": int(c2_eligible),
                "c1_far_field_patch_count": int(c1_diag["far_field_patch_count"]),
                "c1_far_field_abs_correction_max": float(c1_diag["far_field_abs_correction_max"]),
            }
            sealed_record, queried, label = _seal_prediction_and_reveal(
                gate=gate, tape=tape, position=position, arrays=arrays,
                metadata=metadata, query_positions=preflight["query_positions"],
                label_provider=lambda: label_reader(
                    "switch", position, str(meta["sample_key"])))
            feedback_ref: str | None = None
            correction_write_summary: dict[str, Any] = {"c1": None, "c2_added": 0}
            if queried:
                query_count += 1
                query_labels_decoded += 1
                feedback_payload = {
                    "schema": "calibration.r1_private_feedback.v1",
                    "position": position, "event_id": event_id,
                    "source_image_identity_sha256": metadata["source_image_identity_sha256"],
                    "official_group_sha256": group_sha,
                    "disclosure_after_score": True,
                    "disclosure_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "label": int(label),
                }
                feedback_sha = _append_private_feedback(output, feedback_payload)
                feedback_ref = f"private_feedback.jsonl:{position}:{feedback_sha}"
                response_chain = sha256_bytes(
                    bytes.fromhex(response_chain) +
                    canonical_json({"position": position, "group_sha256": group_sha,
                                    "label": int(label)}).encode("utf-8"))
                pred0 = preds["c0"]
                outcome = ("TN" if label == 0 and pred0 == 0 else
                           "FP" if label == 0 and pred0 == 1 else
                           "FN" if label == 1 and pred0 == 0 else "TP")
                sign = 1 if label == 1 else -1
                c1_update_started = time.perf_counter()
                old_n = int(c1_cb.size())
                bank.corr_cfg.enabled = True
                try:
                    c1_result = bank.update_corrections_from_topk(
                        cls_name="p0", feats_per_layer=list(features), mode=0, view="id",
                        layer_to_patch_idx={0: top_ids.tolist()}, sign=sign,
                        score_img=float(c0_score), theta=theta, strength_scale=1.0,
                        outcome=outcome, step=position)
                finally:
                    bank.corr_cfg.enabled = False
                c1_layer = c1_result.get("layers", {}).get("0", {})
                new_n = int(c1_cb.size())
                _record_c1_source_group_updates(
                    c1_source_groups,
                    last_steps=c1_cb.last_step[:new_n].detach().to("cpu").tolist(),
                    event_position=position, event_group=group_sha)
                kept = int(c1_layer.get("kept", 0))
                admitted = int(c1_layer.get("inserted", 0)) + int(c1_layer.get("merged", 0)) + int(c1_layer.get("replaced", 0))
                if sign > 0:
                    positive_anchor_writes += admitted
                else:
                    negative_anchor_writes += admitted
                c1_active_positive = int((c1_cb.beta[:new_n] > 0).sum().item())
                c1_active_negative = int((c1_cb.beta[:new_n] < 0).sum().item())
                c2_added = 0
                for patch_row, patch_id in enumerate(top_ids.tolist()):
                    feature_row = normalized_selected[patch_row].detach().float().to("cpu").numpy().copy()
                    feature_sha = arrays_sha256({"feature": feature_row})
                    c2_anchors.append({
                        "position": position, "group_sha256": group_sha,
                        "patch_id": int(patch_id), "sign": sign,
                        "feature": feature_row, "feature_sha256": feature_sha,
                    })
                    c2_added += 1
                if len(c2_anchors) > C2_ANCHOR_CAPACITY:
                    del c2_anchors[:len(c2_anchors) - C2_ANCHOR_CAPACITY]
                c2_active_anchors = len(c2_anchors)
                correction_update_seconds += time.perf_counter() - c1_update_started
                _append_jsonl(output / "correction_anchor_events.jsonl", {
                    "position": position, "event_id": event_id,
                    "feedback_ref": feedback_ref, "source_group_sha256": group_sha,
                    "c0_prediction": pred0, "c0_score": c0_score,
                    "c0_threshold": theta, "outcome": outcome, "sign": sign,
                    "c1_kept_by_native_gate": kept,
                    "c1_stats": {k: int(v) for k, v in c1_layer.items()
                                 if isinstance(v, (int, np.integer))},
                    "c2_added_patch_rows": c2_added,
                    "c2_retention": f"FIFO, max {C2_ANCHOR_CAPACITY} patch anchors",
                })
                correction_write_summary = {
                    "c1": {k: int(v) for k, v in c1_layer.items()
                           if isinstance(v, (int, np.integer))},
                    "c2_added": c2_added,
                }
            gate.corrections_updated()

            _validate_c1_source_groups(
                c1_source_groups,
                c1_cb.last_step[:int(c1_cb.size())].detach().to("cpu").tolist(),
                event_position=position, phase="post_update")
            _validate_c2_anchor_state(
                c2_anchors, event_position=position, phase="post_update")

            writer_started = time.perf_counter()
            memory_content_before_write = _memory_content_sha256(bank)
            def _writer_update(*, position: int, feature_layers: Sequence[Any],
                               candidate_ids: np.ndarray, score: float,
                               theta: float, sigma: float) -> Mapping[str, Any]:
                audit_before = writer.audit()
                writer.update(impl, list(feature_layers),
                              {0: candidate_ids.tolist()}, position,
                              score=score, theta=theta, sigma=sigma)
                audit_after = writer.audit()
                return {
                    "accepted_patches_delta": int(
                        audit_after["accepted_patches"] - audit_before["accepted_patches"]),
                    "skipped_patches_delta": int(
                        audit_after["skipped_patches"] - audit_before["skipped_patches"]),
                }

            writer_action = common_c0_greedy_writer_action(
                position=position, c0_score=float(c0_score), theta=theta,
                sigma=float(preflight["c64_sigma0_with_floor_0_02"]),
                c0_prediction=preds["c0"], candidate_ids=top_ids,
                candidate_features=arrays["top32_features"],
                feature_layers=features, queried=queried,
                disclosed_label=label,
                eligibility=impl.select_informative_normal_write,
                update=_writer_update)
            if writer_action["action"] == "native_greedy_normal_update":
                normal_memory_writes += 1
            writer_seconds += time.perf_counter() - writer_started
            memory_content_after_write = _memory_content_sha256(bank)
            if memory_content_after_write != memory_content_before_write:
                memory_version += 1
                memory_content_digest = memory_content_after_write
                memory_state_digest = _memory_state_sha256(bank, memory_version)
            writer_action["memory_version_after"] = memory_version
            writer_action["memory_digest_after"] = memory_state_digest
            gate.memory_updated()

            c1_after_digest = _correction_state_sha256(bank, c1_source_groups)
            c2_after_digest = _c2_state_sha256(c2_anchors)

            peak_rss = max(peak_rss, _process_rss_bytes() or 0)
            if torch.cuda.is_available():
                peak_cuda_allocated = max(peak_cuda_allocated,
                                          int(torch.cuda.max_memory_allocated()))
                peak_cuda_reserved = max(peak_cuda_reserved,
                                         int(torch.cuda.max_memory_reserved()))
            transition = {
                "query_decision": bool(queried),
                "query_reference": (f"switch:{position}" if queried else None),
                "writer_inputs_sha256": sha256_json({
                    "score": c0_score, "theta": theta,
                    "sigma": preflight["c64_sigma0_with_floor_0_02"],
                    "prediction": preds["c0"], "top32_ids": top_ids.tolist(),
                    "queried": queried,
                }),
                "writer_action": writer_action,
                "feedback_reference": feedback_ref,
                "greedy_memory_version_after": memory_version,
                "greedy_memory_sha256_after": memory_state_digest,
                "greedy_memory_content_sha256_after": memory_content_digest,
                "c1_state_sha256_after": c1_after_digest,
                "c2_state_sha256_after": c2_after_digest,
                "causal_order_complete": True,
            }
            tape.finalize_event(position, transition)
            gate.finish()
            if queried and label_reader.decoded_positions[-1] != position:
                causal_mismatches += 1
                raise R1CausalityError("Queried label reader disclosed a different position")
            if tape.manifest["output_bytes"] > config.max_output_bytes:
                raise R1ResourceLimit("R1 output exceeded the authorized storage ceiling")
            if (position + 1) % 32 == 0:
                print(f"R1_PROGRESS events={position + 1} queries={query_count} "
                      f"gpu_wall_seconds={elapsed:.1f} output_bytes={_directory_bytes(output)}",
                      flush=True)

        if query_labels_decoded != len(preflight["query_positions"]):
            raise R1CausalityError("Queried label count differs from the frozen prefix tape")
        if list(label_reader.decoded_positions) != list(preflight["query_positions"]):
            raise R1CausalityError("Queried label disclosure order differs from the fixed tape")
        final_fixed = p0.fixed_bank_profile(bank)
        final_fixed_structured_sha256 = _fixed_bank_digest(bank)
        if (final_fixed_structured_sha256 != fixed_digest or
                final_fixed["sha256"] != fixed_raw_digest):
            raise R1CausalityError("Shared Greedy writer mutated the frozen fixed support bank")
        readback = tape.verify(require_complete=True)
        observed_anchors_both_signs = c1_active_positive > 0 and c1_active_negative > 0
        status = "ENGINEERING_PASS" if observed_anchors_both_signs else "INSUFFICIENT_ANCHOR_EVIDENCE"
        final_summary = {
            "status": status, "events_processed": tape.next_finalized,
            "unique_stream_positions": tape.next_finalized,
            "smoke_parity_events": parity_count,
            "query_events": query_count,
            "queried_labels_decoded": query_labels_decoded,
            "query_positions_decoded_sha256": sha256_json(label_reader.decoded_positions),
            "c1_positive_anchor_writes": positive_anchor_writes,
            "c1_negative_anchor_writes": negative_anchor_writes,
            "c1_active_positive_anchors": c1_active_positive,
            "c1_active_negative_anchors": c1_active_negative,
            "c2_active_anchors": c2_active_anchors,
            "c1_far_field_selected_patch_observations": c1_far_field_count,
            "c1_far_field_abs_correction_max": c1_far_field_abs_max,
            "source_group_exclusion_checks": source_exclusion_checks,
            "source_group_exclusion_checks_passed": source_exclusion_checks == tape.next_sealed,
            "normal_memory_write_events": normal_memory_writes,
            "greedy_memory_version_final": memory_version,
            "greedy_memory_sha256_final": memory_state_digest,
            "fixed_support_sha256_final": final_fixed_structured_sha256,
            "fixed_support_digest_algorithm": STRUCTURED_FIXED_BANK_DIGEST_ALGORITHM,
            "fixed_support_raw_prototype_sha256_final": final_fixed["sha256"],
            "fixed_support_raw_digest_algorithm": RAW_FIXED_BANK_DIGEST_ALGORITHM,
            "causal_state_mismatches": causal_mismatches,
            "parity_failures": 0,
            "evidence_tape_readback": readback,
            "encoding_seconds": encode_seconds,
            "native_scoring_seconds": native_seconds,
            "support_and_model_seconds": support_and_model_elapsed,
            "correction_scoring_seconds": correction_score_seconds,
            "correction_update_seconds": correction_update_seconds,
            "greedy_writer_seconds": writer_seconds,
            "gpu_wall_seconds": (time.monotonic() - pilot_execution_started
                                 if pilot_execution_started is not None else 0.0),
            "peak_cpu_rss_bytes": peak_rss or None,
            "peak_cuda_allocated_bytes": peak_cuda_allocated,
            "peak_cuda_reserved_bytes": peak_cuda_reserved,
            "output_bytes": _directory_bytes(output),
            "response_chain_sha256": response_chain,
            "scientific_outcome_evaluation": False,
            "auroc_computed": False, "auprc_computed": False,
            "classification_metrics_computed": False,
            "masks_accessed": False, "unqueried_labels_accessed": False,
            "confirmation_data_accessed": False,
        }
        tape.update_manifest(final_summary)
        tape.manifest["status"] = status
        tape.update_manifest({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        label_reader.close()
        backbone.cleanup()
        return final_summary
    except Exception as exc:
        if output.exists():
            if isinstance(exc, R1ParityError):
                status = "PARITY_FAILED"
            elif isinstance(exc, R1CausalityError):
                status = "CAUSALITY_FAILED"
            elif isinstance(exc, R1ResourceLimit):
                status = "RESOURCE_LIMIT_REACHED"
            elif isinstance(exc, R1BlockedInput):
                status = "BLOCKED_INPUT"
            else:
                status = "INCOMPLETE"
            try:
                current = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
                current.update({"status": status, "failure_type": type(exc).__name__,
                                "failure_message": str(exc)[:1000],
                                "events_processed": current.get("finalized_events", 0),
                                "query_events": query_count,
                                "queried_labels_decoded": query_labels_decoded,
                                "parity_failures": int(isinstance(exc, R1ParityError)),
                                "causal_state_mismatches": causal_mismatches,
                                "gpu_wall_seconds": (time.monotonic() - pilot_execution_started
                                                     if pilot_execution_started is not None else 0.0),
                                "peak_cpu_rss_bytes": peak_rss or None,
                                "peak_cuda_allocated_bytes": peak_cuda_allocated,
                                "peak_cuda_reserved_bytes": peak_cuda_reserved,
                                "output_bytes": _directory_bytes(output)})
                _atomic_json(output / "run_manifest.json", current,
                             failure_reserved=True)
            except Exception:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--json-root", type=Path, default=DEFAULT_JSON_ROOT)
    parser.add_argument("--expected-source-commit", required=True,
                        help="Full commit SHA authorized for this execution")
    parser.add_argument("--profile-id", required=True,
                        choices=(R1_PROFILE_ID, SUPPORT_ONLY_PROFILE_ID))
    parser.add_argument("--run-id", required=True,
                        help="Unique suffix for a new non-overwriting output namespace")
    parser.add_argument("--authorization-id", required=True,
                        help="Researcher authorization record for this execution")
    parser.add_argument("--support-only", action="store_true",
                        help="Build/verify the fixed bank and stop before stream or label access")
    args = parser.parse_args(argv)
    try:
        result = run_pilot(
            dataset_root=args.dataset_root, json_root=args.json_root,
            expected_source_commit=args.expected_source_commit,
            profile_id=args.profile_id, run_id=args.run_id,
            authorization_id=args.authorization_id,
            support_only=args.support_only)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0 if result["status"] in {
            "ENGINEERING_PASS", "INSUFFICIENT_ANCHOR_EVIDENCE",
            "SUPPORT_ONLY_VERIFIED"} else 1
    except Exception as exc:
        print(f"R1_PILOT_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
