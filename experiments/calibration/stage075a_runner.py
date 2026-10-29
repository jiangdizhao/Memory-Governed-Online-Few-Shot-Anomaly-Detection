"""One-command Stage 0.75-A frozen Native Greedy development audit.

Phases are deliberately separated: ``freeze`` reads only label-blind metadata,
``smoke`` and ``run`` receive labels through a queried-position oracle process,
and ``evaluate`` opens complete labels only after all online traces are frozen.
The run scores each image once and shares its immutable feature bundle across
the five support-seed trajectories for a category.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from . import stage05
from .controllers import theta0_from_c64
from .native import SharedNativeGreedyWriter, isolate_native_bank_k256
from .profiles import DEFAULT_PROFILE, canonical_json
from .stage075a import (C64_DIAGNOSTIC_EPSILON, DEVELOPMENT_CLASSES,
                        QUERY_RATE, QUERY_SEED, SCHEMA, SNAPSHOT_NEXT_WIDTH,
                        SNAPSHOT_POSITIONS, SUPPORT_SEEDS,
                        c64_transport_diagnostics, confusion_metrics,
                        higher_quantile, paired_score_metrics,
                        prior_group_status, snapshot_cohort_mask,
                        summarize_numeric)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
DEFAULT_DATASET_ROOT = Path("G:/dataset/realiad")
DEFAULT_JSON_ROOT = DEFAULT_DATASET_ROOT / "jsons" / "realiad_jsons"
STAGE05_OUTPUT = ROOT / "outputs/calibration/stage05_exploratory_20261029"
DEFAULT_OUTPUT = ROOT / "outputs/calibration/stage075a_greedy_ranking"
TASK_ATTACHMENT = Path(
    r"C:\Users\rico\.codex\attachments\4cccda3c-2474-4ac4-b46f-7e3343383e58\Pasted text.txt")
CHECKPOINT_INTERVAL = 256
SMOKE_LENGTH = 512

SOURCE_FILES = (
    "experiments/calibration/stage075a.py",
    "experiments/calibration/stage075a_runner.py",
    "experiments/calibration/stage075a_label_oracle.py",
    "tests/calibration/test_stage075a.py",
    "experiments/calibration/stage05.py",
    "experiments/calibration/stage05_runner.py",
    "experiments/calibration/native.py",
    "experiments/calibration/memory_state.py",
    "experiments/calibration/controllers.py",
    "experiments/calibration/profiles.py",
    "experiments/p0_protocol_audit.py",
    "experiments/a0_a1_strict4_matched.py",
    "experiments/v02_memory.py",
    "experiments/risk_safe/native_bridge.py",
    "experiments/risk_safe/shadow_memory.py",
    "test_phase3_AIF_simplified_v22.py",
    "vmb_visionad_new.py",
)


class RunnerError(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted((item for item in path.rglob("*.py") if item.is_file()),
                   key=lambda item: item.relative_to(path).as_posix())
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha_file(item)))
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                dir=str(path.parent))
    os.close(fd)
    temp = Path(name)
    try:
        temp.write_bytes(data)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def _refresh_checksums(folder: Path) -> dict[str, str]:
    entries = {}
    for path in sorted(folder.rglob("*")):
        if "private_checkpoints" in path.parts:
            continue
        if (path.is_file() and path.name not in {"checksums.json", "INCOMPLETE"}
                and not path.name.endswith(".tmp")):
            entries[path.relative_to(folder).as_posix()] = _sha_file(path)
    _atomic_json(folder / "checksums.json", {"algorithm": "SHA-256", "files": entries})
    return entries


def _git_info() -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    branch, head = git("branch", "--show-current"), git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        raise RunnerError(f"Expected authorized branch {EXPECTED_BRANCH}, found {branch}")
    return {"branch": branch, "source_head": head,
            "parent_sha": head, "remote": "origin/" + EXPECTED_BRANCH}


def _environment() -> dict[str, Any]:
    import torch
    expected = Path(r"D:\anaconda3\envs\torch\python.exe").resolve()
    actual = Path(sys.executable).resolve()
    if os.name == "nt" and actual != expected:
        raise RunnerError(f"Wrong Python: {actual}; expected {expected}")
    if not torch.cuda.is_available():
        raise RunnerError("CUDA is unavailable; authorized GPU audit is BLOCKED")
    return {"python": str(actual), "python_version": sys.version.split()[0],
            "torch": str(torch.__version__), "cuda": True,
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_device_count": torch.cuda.device_count()}


def _load_stage05(dataset_root: Path, json_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not (STAGE05_OUTPUT / "run_manifest.json").is_file():
        raise RunnerError(f"Frozen Stage 0.5 output is absent: {STAGE05_OUTPUT}")
    run = _read_json(STAGE05_OUTPUT / "run_manifest.json")
    if run.get("phase_status", {}).get("score") != "PASS":
        raise RunnerError("Stage 0.5 native score phase is not PASS")
    if run.get("stage1_or_stage2_run") is not False:
        raise RunnerError("Stage 0.5 provenance does not assert Stage 1/2 isolation")
    if hashlib.sha256(str(dataset_root.resolve()).encode()).hexdigest() != run.get(
            "dataset_root_sha256"):
        raise RunnerError("Dataset-root identity differs from Stage 0.5")
    if hashlib.sha256(str(json_root.resolve()).encode()).hexdigest() != run.get(
            "json_root_sha256"):
        raise RunnerError("JSON-root identity differs from Stage 0.5")
    checks = _read_json(STAGE05_OUTPUT / "checksums.json")["files"]
    for relative, digest in checks.items():
        path = STAGE05_OUTPUT / relative
        if not path.is_file() or _sha_file(path) != digest:
            raise RunnerError(f"Stage 0.5 artifact checksum mismatch: {relative}")
    frozen, tapes = __import__(
        "experiments.calibration.stage05_runner", fromlist=["_verify_freeze"]
    )._verify_freeze(STAGE05_OUTPUT)
    score_manifest = _read_json(STAGE05_OUTPUT / "score_tape_manifest.json")
    if score_manifest.get("status") != "PASS":
        raise RunnerError("Stage 0.5 static native score manifest is not PASS")
    return frozen, tapes, score_manifest


def _verify_inputs(dataset_root: Path, json_root: Path
                   ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen, tapes, scores = _load_stage05(dataset_root, json_root)
    if tuple(frozen.get("query_rates", ())) != (0.03, 0.05, 0.07):
        raise RunnerError("Frozen Stage 0.5 rates changed")
    if tuple(frozen.get("query_seeds", ())) != tuple(stage05.QUERY_SEEDS):
        raise RunnerError("Stage 0.5 query tape seed registry changed")
    verified: dict[str, Any] = {}
    query_positions: dict[str, list[int]] = {}
    for category in DEVELOPMENT_CLASSES:
        info = frozen["classes"][category]
        if len(info["supports"]) != 5 or info["c64"]["count"] != 64:
            raise RunnerError(f"{category}: support/C64 shape differs from protocol")
        for seed in SUPPORT_SEEDS:
            support = info["supports"][str(seed)]
            if (len(support["groups"]) != 4 or len(set(support["groups"])) != 4 or
                    len(support["images"]) != 4 or not support["all_train_normal"] or
                    not support["c64_disjoint"]):
                raise RunnerError(f"{category}/seed{seed}: support contract failed")
        tape = tapes[category]["0.03"][str(QUERY_SEED)]
        query_positions[category] = sorted(int(row["position"]) for row in tape["events"])
        if len(query_positions[category]) != len(set(query_positions[category])):
            raise RunnerError(f"{category}: repeated Q-A-TS event position")
        if tape["digest"] != frozen["tape_digests_by_seed37"][category]["0.03"]:
            raise RunnerError(f"{category}: seed-37 3% Q-A-TS digest mismatch")
        class_scores = scores["classes"][category]["score_tapes"]
        for seed in SUPPORT_SEEDS:
            entry = class_scores[str(seed)]
            path = STAGE05_OUTPUT / entry["file"]
            if _sha_file(path) != entry["sha256"]:
                raise RunnerError(f"{category}/seed{seed}: Stage05 score tape checksum mismatch")
            with np.load(path, allow_pickle=False) as data:
                if (len(data["test_scores"]) != info["stream_images"] or
                        len(data["c64_scores"]) != 64 or
                        not np.isfinite(data["test_scores"]).all() or
                        not np.isfinite(data["c64_scores"]).all()):
                    raise RunnerError(f"{category}/seed{seed}: invalid static score tape")
        verified[category] = {
            "stream_images": int(info["stream_images"]),
            "official_groups": int(info["stream_official_groups"]),
            "support_group_sha256": {
                str(seed): info["supports"][str(seed)]["group_sha256"]
                for seed in SUPPORT_SEEDS},
            "support_image_sha256": {
                str(seed): info["supports"][str(seed)]["image_sha256"]
                for seed in SUPPORT_SEEDS},
            "c64_group_sha256": info["c64"]["group_sha256"],
            "c64_image_sha256": info["c64"]["image_sha256"],
            "query_tape_sha256": tape["digest"],
            "query_count": len(query_positions[category]),
            "static_score_sha256_by_seed": {
                str(seed): class_scores[str(seed)]["sha256"]
                for seed in SUPPORT_SEEDS}}
    return frozen, tapes, scores, {
        "classes": verified, "query_positions": query_positions}


def _freeze_identity(dataset_root: Path, json_root: Path) -> dict[str, Any]:
    env = _environment()
    git = _git_info()
    stage05_run = _read_json(STAGE05_OUTPUT / "run_manifest.json")
    stage05_scores = _read_json(STAGE05_OUTPUT / "score_tape_manifest.json")
    source_hashes = {name: _sha_file(ROOT / name) for name in SOURCE_FILES}
    torch_hub = Path(__import__("torch").hub.get_dir())
    checkpoint = torch_hub / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
    hub_source = torch_hub / "facebookresearch_dinov2_main"
    if not checkpoint.is_file() or not hub_source.is_dir():
        raise RunnerError("Pinned local DINOv2 checkpoint/source is unavailable")
    checkpoint_sha = _sha_file(checkpoint)
    source_tree_sha = _tree_sha256(hub_source)
    if checkpoint_sha != stage05_scores["model_provenance"]["checkpoint_sha256"]:
        raise RunnerError("Local DINOv2 checkpoint differs from Stage 0.5 provenance")
    if source_tree_sha != stage05_scores["model_provenance"]["hub_source_tree_sha256"]:
        raise RunnerError("Local DINOv2 source tree differs from Stage 0.5 provenance")
    frozen, _tapes, _scores, verified = _verify_inputs(dataset_root, json_root)
    attachment_sha = _sha_file(TASK_ATTACHMENT) if TASK_ATTACHMENT.is_file() else None
    return {
        "schema": "stage075a.frozen_protocol.v1", **git,
        "environment": env,
        "task_authorization_attachment_sha256": attachment_sha,
        "stage05_run_manifest_sha256": _sha_file(STAGE05_OUTPUT / "run_manifest.json"),
        "stage05_score_manifest_sha256": _sha_file(STAGE05_OUTPUT / "score_tape_manifest.json"),
        "dataset_root_sha256": hashlib.sha256(str(dataset_root.resolve()).encode()).hexdigest(),
        "json_root_sha256": hashlib.sha256(str(json_root.resolve()).encode()).hexdigest(),
        "metadata_sha256_by_class": stage05_run["input_metadata_sha256_by_class"],
        "source_sha256": source_hashes,
        "native_source_sha256": stage05_scores["scoring_source_hashes"],
        "scorer_config": stage05_scores["scorer_config"],
        "scorer_config_sha256": stage05_scores["scorer_config_sha256"],
        "model_provenance": stage05_scores["model_provenance"],
        "local_checkpoint_sha256": checkpoint_sha,
        "local_dinov2_source_tree_sha256": source_tree_sha,
        "class_inputs": verified["classes"],
        "query_positions": verified["query_positions"],
        "support_seeds": list(SUPPORT_SEEDS),
        "query_protocol": {"version": "Q-A-TS-v1", "rate": QUERY_RATE,
                           "seed": QUERY_SEED, "temporal_bins": 16,
                           "one_query_per_official_group": True},
        "reference_writer": {
            "source": "experiments/calibration/native.py::SharedNativeGreedyWriter",
            "theta_ref": "per-seed M0 higher-Q90 of the same protected C64 score tape",
            "higher_q90_index": "ceil(0.90*(64-1)) = 57",
            "sigma_ref": 0.02,
            "novelty": 0.0,
            "candidate_score": "frozen static M0 native score",
            "candidate_map": "native static M0 score map",
            "controller_threshold_used": False,
            "writer_modified": False},
        "memory": {"native_shared_greedy_writer": True,
                   "mutable_capacity_per_layer": 256,
                   "initial_dynamic_rows": 0,
                   "protected_c64_rows_mutable": False,
                   "content_version_changes_only_on_tensor_content": True},
        "snapshot_positions": list(SNAPSHOT_POSITIONS),
        "snapshot_next_width": SNAPSHOT_NEXT_WIDTH,
        "metrics": {"score_tie": "exact floating-point equality",
                    "prediction": "score >= threshold",
                    "quantile_q90": "higher order statistic, ceil(q*(n-1))",
                    "score_quantiles": "linear interpolation",
                    "c64_transport_epsilon": C64_DIAGNOSTIC_EPSILON,
                    "uniformity_sensitivity_ratios": [0.01, 0.05, 0.10]},
        "scope": {"stage": "0.75-A only", "stage075_b_or_c": False,
                  "stage1_or_stage2": False, "confirmation_classes": False,
                  "masks": False, "hidden_unqueried_labels_in_online_process": False,
                  "development_categories_previously_exposed_in_stage05": True},
    }


def _require_output(output: Path, *, create: bool, resume: bool) -> None:
    if create and not output.exists():
        output.mkdir(parents=True)
        (output / "INCOMPLETE").write_text("Stage 0.75-A is incomplete.\n", encoding="utf-8")
    elif not output.is_dir():
        raise RunnerError(f"Stage 0.75-A output directory is absent: {output}")
    manifest = output / "run_manifest.json"
    if create and not manifest.exists():
        return
    if not manifest.is_file():
        raise RunnerError("Stage 0.75-A output is not initialized")
    frozen = _read_json(manifest)
    if not resume and create:
        raise RunnerError(f"Refusing to overwrite an existing Stage 0.75-A output: {output}")
    if frozen.get("authorized_branch") != EXPECTED_BRANCH:
        raise RunnerError("Existing Stage 0.75-A output has another branch identity")


def _manifest_update(output: Path, phase: str, status: str,
                     updates: dict[str, Any] | None = None) -> None:
    path = output / "run_manifest.json"
    manifest = _read_json(path)
    manifest.setdefault("phase_status", {})[phase] = status
    manifest["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if updates:
        manifest.update(updates)
    _atomic_json(path, manifest)
    _refresh_checksums(output)


def _phase_freeze(output: Path, dataset_root: Path, json_root: Path,
                  resume: bool) -> None:
    if output.exists() and not resume:
        raise RunnerError(f"Refusing to overwrite existing output: {output}")
    if output.exists():
        _require_output(output, create=True, resume=resume)
        prior_validation = output / "validation_results.json"
        if not prior_validation.is_file():
            raise RunnerError("Existing output lacks its immutable preflight test record")
        validation = _read_json(prior_validation)
    else:
        validation = _run_validation()
        if validation["status"] != "PASS":
            raise RunnerError("Compilation, calibration unittests, git diff check, or protected-source verification failed")
        _require_output(output, create=True, resume=False)
        _atomic_json(output / "validation_results.json", validation)
    protocol = _freeze_identity(dataset_root, json_root)
    protocol["validation_results_sha256"] = _sha_file(output / "validation_results.json")
    protocol_sha = _sha_json(protocol)
    if (output / "protocol_manifest.json").exists():
        prior = _read_json(output / "protocol_manifest.json")
        if _sha_json(prior) != protocol_sha:
            raise RunnerError("Resume protocol/input/source fingerprints changed")
        return
    _atomic_json(output / "protocol_manifest.json", protocol)
    _atomic_json(output / "run_manifest.json", {
        "schema": "stage075a.run.v1", "authorized_branch": EXPECTED_BRANCH,
        "source_head": protocol["source_head"], "parent_sha": protocol["parent_sha"],
        "protocol_sha256": protocol_sha, "phase_status": {"freeze": "PASS"},
        "stage1_or_stage2_run": False, "confirmation_classes_accessed": False,
        "test_labels_accessed_by_online_process": "QUERIED_ONLY_ORACLE",
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    (output / "smoke").mkdir(exist_ok=True)
    (output / "trajectories").mkdir(exist_ok=True)
    (output / "private_checkpoints").mkdir(exist_ok=True)
    _refresh_checksums(output)


class _LabelOracleClient:
    def __init__(self, *, class_name: str, json_root: Path, source_sha256: str,
                 query_positions: list[int], first_position: int):
        args = [sys.executable, "-m", "experiments.calibration.stage075a_label_oracle",
                "--class-name", class_name, "--json-root", str(json_root),
                "--source-sha256", source_sha256,
                "--query-positions", ",".join(str(p) for p in query_positions),
                "--first-position", str(first_position)]
        self.proc = subprocess.Popen(args, cwd=ROOT, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, bufsize=1)
        assert self.proc.stdout is not None
        ready = json.loads(self.proc.stdout.readline())
        if ready.get("ready") is not True:
            raise RunnerError(f"Queried-label oracle failed to start: {ready}")
        self.label_count = int(ready["label_count"])
        self.requested: list[int] = []

    def reveal(self, position: int) -> int:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RunnerError("Queried-label oracle pipe is closed")
        self.proc.stdin.write(json.dumps({"position": int(position)}) + "\n")
        self.proc.stdin.flush()
        response = json.loads(self.proc.stdout.readline())
        if response.get("error") or int(response.get("position", -1)) != position:
            raise RunnerError(f"Queried-label oracle rejected position {position}")
        self.requested.append(position)
        return int(response["label"])

    def close(self) -> None:
        if self.proc.poll() is None and self.proc.stdin is not None:
            try:
                self.proc.stdin.write('{"close":true}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=15)
            except Exception:
                self.proc.terminate()
                self.proc.wait(timeout=15)
        if self.proc.returncode not in (0, None):
            stderr = self.proc.stderr.read() if self.proc.stderr is not None else ""
            raise RunnerError(f"Queried-label oracle exited {self.proc.returncode}: {stderr}")


def _stage05_score_arrays(category: str, seeds: tuple[int, ...]
                          ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    manifest = _read_json(STAGE05_OUTPUT / "score_tape_manifest.json")
    static, c64 = {}, {}
    for seed in seeds:
        path = STAGE05_OUTPUT / manifest["classes"][category]["score_tapes"][str(seed)]["file"]
        with np.load(path, allow_pickle=False) as data:
            static[seed] = np.asarray(data["test_scores"], dtype=np.float64).copy()
            c64[seed] = np.asarray(data["c64_scores"], dtype=np.float64).copy()
    return static, c64


def _build_native_category(category: str, seeds: tuple[int, ...],
                            frozen_info: dict[str, Any], dataset_root: Path,
                            json_root: Path,
                            impl: Any, cfg: Any):
    from experiments.p0_protocol_audit import (build_vmb, encode_query,
        encode_reference_features)
    from .native import _fixed_bank_digest
    train, stream, source_sha = stage05.read_class_stream(
        dataset_root, json_root, category, test_labels=False)
    if source_sha != frozen_info["class_json_sha256"]:
        raise RunnerError(f"{category}: metadata source changed after freeze")
    support_plan = stage05.select_supports(train)
    if (support_plan["c64"] != frozen_info["c64"] or
            support_plan["supports"] != frozen_info["supports"] or
            support_plan["support_c64_disjoint"] != frozen_info["support_c64_disjoint"]):
        raise RunnerError(f"{category}: support/C64 identities changed after freeze")
    train_by_rel = {row.relative_id: row for row in train}
    c64_rels = list(frozen_info["c64"]["images"])
    all_rels = sorted(set(c64_rels) | {
        relative for seed in seeds
        for relative in frozen_info["supports"][str(seed)]["images"]})
    if any(relative not in train_by_rel or not train_by_rel[relative].is_good
           for relative in all_rels):
        raise RunnerError(f"{category}: selected support/C64 image is not a train normal")
    abs_path = lambda relative: stage05._absolute_image_path(
        dataset_root, frozen_info["prefix"], relative)
    backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
    cache = encode_reference_features(impl, backbone,
                                     [abs_path(relative) for relative in all_rels], cfg)
    native_banks: dict[int, Any] = {}
    pristine_banks: dict[int, Any] = {}
    fixed_hashes: dict[int, str] = {}
    for seed in seeds:
        paths = [abs_path(relative) for relative in frozen_info["supports"][str(seed)]["images"]]
        bank = build_vmb(impl, cache, paths, cfg)
        bank, fixed_sha = isolate_native_bank_k256(bank)
        if _fixed_bank_digest(bank) != fixed_sha:
            raise RunnerError(f"{category}/seed{seed}: fixed support digest mismatch")
        native_banks[seed] = bank
        pristine_banks[seed] = copy.deepcopy(bank)
        fixed_hashes[seed] = fixed_sha
    c64_paths = [abs_path(relative) for relative in c64_rels]
    c64_bundles = [encode_query(impl, backbone, path, cfg) for path in c64_paths]
    if len(c64_bundles) != 64 or len(stream) != frozen_info["stream_images"]:
        raise RunnerError(f"{category}: native input count mismatch")
    return stream, backbone, native_banks, pristine_banks, fixed_hashes, c64_bundles


def _native_config(impl: Any):
    # The native implementation module exposes scorer primitives, while the
    # frozen Stage 0.5 runner imports Config from the P0 protocol adapter.
    # Keep this constructor identical to that verified scorer path.
    from experiments.p0_protocol_audit import Config

    scorer = DEFAULT_PROFILE.scorer
    return Config(
        device="cuda", augmentation_seed=scorer.augmentation_seed,
        bank_seed=scorer.bank_seed, fixed_capacity=scorer.fixed_capacity,
        fixed_per_image=scorer.fixed_per_image,
        augmentations=scorer.augmentations, include_identity=scorer.include_identity,
        multiplier=scorer.multiplier, feature_fuse=scorer.feature_fuse,
        topk_patches=scorer.topk_patches,
        image_aggregation=scorer.image_aggregation, image_topk=scorer.image_topk,
        image_quantile=scorer.image_quantile, knn=scorer.knn)


def _score_c64(impl: Any, bank: Any, bundles: list[Any], cfg: Any) -> np.ndarray:
    from experiments.p0_protocol_audit import score_encoded
    result = np.asarray([score_encoded(impl, bank, image, features, cfg)
                         for image, features in bundles], dtype=np.float64)
    if result.shape != (64,) or not np.isfinite(result).all():
        raise RunnerError("Native protected-C64 scoring returned invalid values")
    return result


def _memory_resident_sources(writer: SharedNativeGreedyWriter) -> set[str]:
    return {source for layer in writer.sources for source in layer
            if not source.startswith("<unknown-")}


def _memory_state_to_device(state: Any, device: str) -> Any:
    from experiments.risk_safe.shadow_memory import GreedyBank
    from .memory_state import CalibrationGreedyState
    layers = [layer.detach().to(device=device).contiguous()
              for layer in state.layers]
    legacy = GreedyBank(layers, state.capacities,
                        version=state.legacy_greedy_version)
    result = CalibrationGreedyState(legacy, state.memory_content_version)
    if (result.sha256() != state.sha256() or
            result.content_sha256() != state.content_sha256()):
        raise RunnerError("CPU checkpoint memory changed during GPU restore")
    return result


def _memory_state_to_cpu(state: Any) -> Any:
    from experiments.risk_safe.shadow_memory import GreedyBank
    from .memory_state import CalibrationGreedyState
    layers = [layer.detach().to(device="cpu").contiguous()
              for layer in state.layers]
    legacy = GreedyBank(layers, state.capacities,
                        version=state.legacy_greedy_version)
    result = CalibrationGreedyState(legacy, state.memory_content_version)
    if (result.sha256() != state.sha256() or
            result.content_sha256() != state.content_sha256()):
        raise RunnerError("GPU snapshot changed while checkpointing")
    return result


def _prototype_content_digest(bank: Any) -> str:
    """Hash prototype bytes from either native GreedyBank representation."""
    from .memory_state import prototype_content_sha256
    return prototype_content_sha256(bank.layers, bank.capacities)


def _writer_record(result: Any, writer: SharedNativeGreedyWriter) -> dict[str, Any]:
    source_rows = []
    for layer in writer.sources:
        counts = Counter(layer)
        source_rows.append({
            hashlib.sha256(("memory-source-group\0" + key).encode()).hexdigest(): int(count)
            for key, count in sorted(counts.items())
            if not key.startswith("<unknown-")})
    return {
        "eligible": bool(result.eligible), "changed": bool(result.changed),
        "keep": int(result.keep), "reason": str(result.reason),
        "appended": int(result.appended), "replaced": int(result.replaced),
        "skipped": int(result.skipped),
        "source_map_changed": bool(result.source_map_changed),
        "source_revision_before": int(result.source_revision_before),
        "source_revision_after": int(result.source_revision),
        "memory_content_version_before": int(result.memory_content_version_before),
        "memory_content_version_after": int(result.memory_content_version_after),
        "legacy_greedy_version_before": int(result.legacy_greedy_version_before),
        "legacy_greedy_version_after": int(result.legacy_greedy_version_after),
        "dynamic_rows_by_layer": [int(len(layer)) for layer in writer.bank.layers],
        "capacity_by_layer": list(writer.bank.capacities),
        "source_map_sha256": writer.source_map_sha256(),
        "resident_source_groups_by_layer_sha256": source_rows,
    }


def _run_category(*, output: Path, dataset_root: Path, json_root: Path,
                  frozen: dict[str, Any], protocol: dict[str, Any],
                  category: str, seeds: tuple[int, ...], max_events: int | None,
                  smoke: bool, resume: bool) -> dict[str, Any]:
    import torch
    run_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    from experiments.p0_protocol_audit import (encode_query, require_sources,
                                                 score_encoded)
    from experiments.risk_safe.native_bridge import (install_on_clone, native_score,
                                                      snapshot as native_snapshot)
    from .native import SharedNativeGreedyWriter
    from .stage075a import c64_transport_diagnostics

    stage05_info = frozen["classes"][category]
    stream, tapes, _scores, verified = _verify_inputs(dataset_root, json_root)
    stream_rows = [stage05.StreamImage(**row) for row in stage05_info["stream"]]
    n = len(stream_rows)
    end = n if max_events is None else min(n, int(max_events))
    if end <= 0:
        raise RunnerError("Requested trajectory length is empty")
    query_tape_obj = tapes[category]["0.03"][str(QUERY_SEED)]
    query_positions = sorted(int(row["position"]) for row in query_tape_obj["events"])
    query_set = set(query_positions)
    if len(query_set) != len(query_positions):
        raise RunnerError(f"{category}: duplicate query positions")
    target_root = output / ("smoke" if smoke else "trajectories") / category
    target_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "private_checkpoints" / (
        f"{category}_smoke.pt" if smoke else f"{category}_full.pt")
    run_identity = _sha_json({"protocol_sha256": _sha_file(output / "protocol_manifest.json"),
                              "category": category, "seeds": list(seeds),
                              "end_position": end, "smoke": smoke})
    source_sha = stage05_info["class_json_sha256"]
    score_static, score_c64 = _stage05_score_arrays(category, seeds)
    score_static = {seed: values[:end].copy() for seed, values in score_static.items()}
    initial_c64 = {seed: values.copy() for seed, values in score_c64.items()}

    scorer_cfg = _native_config(require_sources())
    impl = require_sources()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    impl.set_seed(0)
    (stream_actual, backbone, live_banks, static_banks, fixed_hashes,
     c64_bundles) = _build_native_category(
        category, seeds, stage05_info, dataset_root, json_root, impl, scorer_cfg)
    if len(stream_actual) != n:
        raise RunnerError(f"{category}: stream length changed")
    stage05_score_manifest = _read_json(STAGE05_OUTPUT / "score_tape_manifest.json")
    for seed in seeds:
        expected_fixed = stage05_score_manifest["classes"][category][
            "score_tapes"][str(seed)]["fixed_support_sha256"]
        if fixed_hashes[seed] != expected_fixed:
            raise RunnerError(f"{category}/seed{seed}: fixed native support differs from Stage05")

    writers: dict[int, SharedNativeGreedyWriter] = {}
    theta0: dict[int, float] = {}
    c64_current: dict[int, np.ndarray] = {}
    c64_state_digest: dict[int, str] = {}
    for seed in seeds:
        direct_c64 = _score_c64(impl, static_banks[seed], c64_bundles, scorer_cfg)
        if not np.array_equal(direct_c64, initial_c64[seed]):
            max_delta = float(np.max(np.abs(direct_c64 - initial_c64[seed])))
            raise RunnerError(
                f"{category}/seed{seed}: exact M0 C64 parity differs from Stage05 by {max_delta}")
        # Preserve the Stage05 score bits as the initial M0 evidence vector.
        c64_current[seed] = initial_c64[seed].copy()
        theta0[seed] = theta0_from_c64(c64_current[seed].tolist())
        writers[seed] = SharedNativeGreedyWriter(
            impl, live_banks[seed], theta_ref=theta0[seed],
            protected_source_keys=stage05_info["c64"]["groups"], sigma_ref=0.02)
        c64_state_digest[seed] = writers[seed].bank.sha256()

    score_dynamic = {seed: np.full(n, np.nan, dtype=np.float64) for seed in seeds}
    current_q90 = {seed: np.full(n, np.nan, dtype=np.float64) for seed in seeds}
    ever_group_mask = {seed: np.zeros(n, dtype=bool) for seed in seeds}
    resident_group_mask = {seed: np.zeros(n, dtype=bool) for seed in seeds}
    memory_version_before = {seed: np.full(n, -1, dtype=np.int32) for seed in seeds}
    memory_sha_before = {seed: [""] * n for seed in seeds}
    snapshot_scores: dict[int, dict[int, np.ndarray]] = {
        seed: {t: np.full(n, np.nan, dtype=np.float64)
               for t in SNAPSHOT_POSITIONS if t < end}
        for seed in seeds}
    snapshot_state: dict[int, dict[int, Any]] = {seed: {} for seed in seeds}
    snapshot_sources: dict[int, dict[int, list[list[str]]]] = {
        seed: {} for seed in seeds}
    snapshot_prior_groups: dict[int, dict[int, set[str]]] = {seed: {} for seed in seeds}
    snapshot_c64_q90: dict[int, dict[int, float]] = {seed: {} for seed in seeds}
    snapshot_banks: dict[int, dict[int, Any]] = {seed: {} for seed in seeds}
    c64_transport: dict[int, list[dict[str, Any]]] = {seed: [] for seed in seeds}
    c64_pairs_before: dict[int, list[np.ndarray]] = {seed: [] for seed in seeds}
    c64_pairs_after: dict[int, list[np.ndarray]] = {seed: [] for seed in seeds}
    event_rows: list[dict[str, Any]] = []
    ever_admitted = {seed: set() for seed in seeds}
    start_position = 0
    elapsed_prior = 0.0
    if checkpoint_path.is_file():
        if not resume:
            raise RunnerError(f"Checkpoint exists; use --resume: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("run_identity") != run_identity:
            raise RunnerError("Trajectory checkpoint does not match frozen run identity")
        start_position = int(checkpoint["next_position"])
        elapsed_prior = float(checkpoint.get("elapsed_seconds", 0.0))
        if not 0 <= start_position <= end:
            raise RunnerError("Checkpoint stream position is invalid")
        random.setstate(checkpoint["rng_state_python"])
        np.random.set_state(checkpoint["rng_state_numpy"])
        torch.set_rng_state(checkpoint["rng_state_torch"])
        if torch.cuda.is_available() and checkpoint.get("rng_state_cuda") is not None:
            torch.cuda.set_rng_state_all(checkpoint["rng_state_cuda"])
        for seed in seeds:
            writers[seed].load_state_dict(checkpoint["writers"][str(seed)])
            c64_current[seed] = checkpoint["c64_current"][str(seed)]
            c64_state_digest[seed] = checkpoint["c64_state_digest"][str(seed)]
            if c64_state_digest[seed] != writers[seed].bank.sha256():
                raise RunnerError("Checkpoint C64 evidence is keyed to a different memory digest")
            ever_admitted[seed] = set(checkpoint["ever_admitted"][str(seed)])
            score_dynamic[seed] = checkpoint["score_dynamic"][str(seed)]
            current_q90[seed] = checkpoint["current_q90"][str(seed)]
            ever_group_mask[seed] = checkpoint["ever_group_mask"][str(seed)]
            resident_group_mask[seed] = checkpoint["resident_group_mask"][str(seed)]
            memory_version_before[seed] = checkpoint["memory_version_before"][str(seed)]
            memory_sha_before[seed] = checkpoint["memory_sha_before"][str(seed)]
            c64_transport[seed] = checkpoint["c64_transport"][str(seed)]
            c64_pairs_before[seed] = checkpoint["c64_pairs_before"][str(seed)]
            c64_pairs_after[seed] = checkpoint["c64_pairs_after"][str(seed)]
            for t in snapshot_scores[seed]:
                snapshot_scores[seed][t] = checkpoint["snapshot_scores"][str(seed)][str(t)]
                if str(t) in checkpoint["snapshot_state"][str(seed)]:
                    snapshot_state[seed][t] = _memory_state_to_device(
                        checkpoint["snapshot_state"][str(seed)][str(t)], "cuda")
                    snapshot_sources[seed][t] = checkpoint["snapshot_sources"][str(seed)][str(t)]
                    snapshot_prior_groups[seed][t] = set(
                        checkpoint["snapshot_prior_groups"][str(seed)][str(t)])
                    snapshot_c64_q90[seed][t] = float(
                        checkpoint["snapshot_c64_q90"][str(seed)][str(t)])
                    snapshot_banks[seed][t] = install_on_clone(
                        static_banks[seed], snapshot_state[seed][t])
        event_rows = checkpoint["event_rows"]

    if start_position and any(
            not np.isfinite(score_dynamic[seed][:start_position]).all()
            for seed in seeds):
        raise RunnerError("Checkpoint has missing completed dynamic scores")

    # Build the immutable snapshot at t=0 on the empty M0 bank.
    if start_position == 0 and 0 in snapshot_scores[seeds[0]]:
        for seed in seeds:
            snapshot_state[seed][0] = writers[seed].bank.clone()
            snapshot_sources[seed][0] = [list(layer) for layer in writers[seed].sources]
            snapshot_prior_groups[seed][0] = set()
            snapshot_c64_q90[seed][0] = theta0[seed]
            initialize_static_snapshot(snapshot_scores[seed][0], score_static[seed], end)

    oracle: _LabelOracleClient | None = None
    if query_positions:
        oracle = _LabelOracleClient(
            class_name=category, json_root=json_root, source_sha256=source_sha,
            query_positions=[p for p in query_positions if p < end],
            first_position=start_position)
        if oracle.label_count != n:
            oracle.close()
            raise RunnerError(f"{category}: oracle label count differs from metadata")

    score_calls = {seed: 0 for seed in seeds}
    feature_count = 0
    try:
        for position in range(start_position, end):
            record = stream_rows[position]
            source_key = record.sample_key
            opaque_group = record.opaque_group

            # Fixed snapshots at t contain writes strictly before t.
            if position in snapshot_positions_for_end(end):
                for seed in seeds:
                    snap_state = writers[seed].bank.clone()
                    prior = set(ever_admitted[seed])
                    snapshot_state[seed][position] = snap_state
                    snapshot_sources[seed][position] = [list(layer)
                                                        for layer in writers[seed].sources]
                    snapshot_prior_groups[seed][position] = prior
                    snapshot_c64_q90[seed][position] = higher_quantile(
                        c64_current[seed].tolist(), .90)
                    snapshot_banks[seed][position] = install_on_clone(
                        static_banks[seed], snap_state)
                    snap_mem = set(prior) | _memory_resident_sources(writers[seed])
                    all_mask = snapshot_cohort_mask(
                        [row.sample_key for row in stream_rows[:end]],
                        snapshot_position=position, stream_length=end,
                        prior_sources=snap_mem, cohort="all_remaining")
                    next_mask = snapshot_cohort_mask(
                        [row.sample_key for row in stream_rows[:end]],
                        snapshot_position=position, stream_length=end,
                        prior_sources=snap_mem, cohort="next256")
                    values = snapshot_scores[seed][position]
                    values[np.flatnonzero(~all_mask)] = np.nan
                    # `next256` is derived from this same all-remaining vector.
                    if next_mask.sum() > SNAPSHOT_NEXT_WIDTH:
                        raise RunnerError("Snapshot next-256 mask exceeds its fixed width")

            image, features = encode_query(impl, backbone, record.image_path, scorer_cfg)
            feature_count += 1

            # Freeze both score vectors and the pre-event memory identities.
            dynamic_scores_at_t: dict[int, float] = {}
            for seed in seeds:
                writer = writers[seed]
                before_version, before_digest = writer.bank.version, writer.bank.sha256()
                memory_version_before[seed][position] = before_version
                memory_sha_before[seed][position] = before_digest
                dynamic_value = float(score_encoded(
                    impl, writer.native_bank, image, features, scorer_cfg))
                if not math.isfinite(dynamic_value):
                    raise RunnerError("Native dynamic scorer produced a nonfinite value")
                dynamic_scores_at_t[seed] = dynamic_value
                score_dynamic[seed][position] = dynamic_value
                current_q90[seed][position] = higher_quantile(c64_current[seed].tolist(), .90)
                group_flags = prior_group_status(
                    source_key, ever_admitted[seed], _memory_resident_sources(writer))
                ever_group_mask[seed][position] = not group_flags["previously_admitted"]
                resident_group_mask[seed][position] = not group_flags["currently_resident"]
                if all(len(layer) == 0 for layer in writer.bank.layers):
                    if dynamic_value != float(score_static[seed][position]):
                        raise RunnerError(
                            f"{category}/seed{seed}/t{position}: untouched M0 dynamic score differs")
                # Required full seed-0 M0 reconstruction against Stage05 for each image.
                if seed == 0:
                    reference_value = float(score_encoded(
                        impl, static_banks[seed], image, features, scorer_cfg))
                    score_calls[seed] += 1
                    if reference_value != float(score_static[seed][position]):
                        raise RunnerError(
                            f"{category}/seed0/t{position}: Stage05 M0 score parity failed")

            # Score every eligible frozen snapshot on this exact immutable feature bundle.
            for seed in seeds:
                for snap_position, snap_state in snapshot_state[seed].items():
                    if snap_position > position or not np.isnan(
                            snapshot_scores[seed][snap_position][position]):
                        continue
                    eligible = snapshot_cohort_mask(
                        [row.sample_key for row in stream_rows[:end]],
                        snapshot_position=snap_position, stream_length=end,
                        prior_sources=set(snapshot_prior_groups[seed][snap_position]),
                        cohort="all_remaining")
                    if not eligible[position]:
                        continue
                    snapshot_value = float(score_encoded(
                        impl, snapshot_banks[seed][snap_position], image, features, scorer_cfg))
                    snapshot_scores[seed][snap_position][position] = snapshot_value

            queried = position in query_set
            static_maps: dict[int, Any] = {}
            if queried:
                for seed in seeds:
                    static_value, amap = native_score(
                        impl, static_banks[seed], image, features, scorer_cfg)
                    if static_value != float(score_static[seed][position]):
                        raise RunnerError(
                            f"{category}/seed{seed}/t{position}: queried M0 score/map parity failed")
                    static_maps[seed] = amap
                if oracle is None:
                    raise RunnerError("Query tape requires a label oracle")
                current_label = oracle.reveal(position)
            else:
                current_label = None

            event_seed_rows: dict[str, Any] = {}
            for seed in seeds:
                writer = writers[seed]
                prior_flags = prior_group_status(
                    source_key, ever_admitted[seed], _memory_resident_sources(writer))
                resident_before = _memory_resident_sources(writer)
                before_c64 = c64_current[seed].copy()
                before_mem_sha = writer.bank.sha256()
                result = writer.apply(
                    queried=queried, label=current_label, source_key=source_key,
                    score=float(score_static[seed][position]),
                    static_amap=static_maps.get(seed), full_features=features,
                    step=position)
                if writer.bank.sha256() == before_mem_sha and result.changed:
                    raise RunnerError("Writer reported a content change without digest change")
                if result.changed:
                    if writer.bank.version != result.memory_content_version_before + 1:
                        raise RunnerError("Memory content version did not advance exactly once")
                    if c64_state_digest[seed] != before_mem_sha:
                        raise RunnerError("C64 evidence is not keyed to the exact pre-update memory")
                    c64_started = time.perf_counter()
                    after_c64 = _score_c64(impl, writer.native_bank, c64_bundles, scorer_cfg)
                    c64_elapsed = time.perf_counter() - c64_started
                    diagnostic = c64_transport_diagnostics(
                        before_c64.tolist(), after_c64.tolist(),
                        theta_previous=higher_quantile(before_c64.tolist(), .90),
                        epsilon=C64_DIAGNOSTIC_EPSILON)
                    diagnostic.update({
                        "position": position, "memory_before_sha256": before_mem_sha,
                        "memory_after_sha256": writer.bank.sha256(),
                        "memory_version_before": result.memory_content_version_before,
                        "memory_version_after": result.memory_content_version_after,
                        "writer_appended": result.appended,
                        "writer_replaced": result.replaced,
                        "writer_skipped": result.skipped,
                        "paired_rescore_seconds": c64_elapsed,
                        "source_revision_after": result.source_revision,
                        "c64_before_sha256": _sha_json(before_c64.tolist()),
                        "c64_after_sha256": _sha_json(after_c64.tolist())})
                    c64_transport[seed].append(diagnostic)
                    c64_pairs_before[seed].append(before_c64)
                    c64_pairs_after[seed].append(after_c64)
                    c64_current[seed] = after_c64
                    c64_state_digest[seed] = writer.bank.sha256()
                elif writer.bank.content_sha256() != \
                        _prototype_content_digest(
                            __import__("experiments.risk_safe.native_bridge",
                                       fromlist=["snapshot"]).snapshot(
                                           writer.native_bank,
                                           version=writer.bank.version)):
                    raise RunnerError("Live native bank differs from calibration memory content")
                resident_after = _memory_resident_sources(writer)
                added_sources = resident_after - resident_before
                # Source rows that were ever represented remain in the conservative history.
                ever_admitted[seed].update(
                    source for layer in writer.sources for source in layer
                    if not source.startswith("<unknown-"))
                event_seed_rows[str(seed)] = {
                    "static_score": float(score_static[seed][position]),
                    "dynamic_score": dynamic_scores_at_t[seed],
                    "theta0": theta0[seed], "current_C64_q90_pre_event": float(
                        current_q90[seed][position]),
                    "prediction_static_theta0": int(
                        score_static[seed][position] >= theta0[seed]),
                    "prediction_dynamic_theta0": int(
                        dynamic_scores_at_t[seed] >= theta0[seed]),
                    "memory_version_before": int(memory_version_before[seed][position]),
                    "memory_sha256_before": memory_sha_before[seed][position],
                    "memory_version_after": int(writer.bank.version),
                    "memory_sha256_after": writer.bank.sha256(),
                    "pre_event_previously_admitted_group": prior_flags[
                        "previously_admitted"],
                    "pre_event_currently_resident_group": prior_flags[
                        "currently_resident"],
                    "newly_resident_source_group": source_key in added_sources,
                    "writer": _writer_record(result, writer)}
            event_rows.append({
                "position": position,
                "image_identity_sha256": hashlib.sha256(
                    ("image-identity\0" + record.relative_id).encode()).hexdigest(),
                "official_group_sha256": opaque_group,
                "camera_id": record.camera_id, "time_bin": stream_time_bin(
                    query_tape_obj["bin_edges"], position), "query_tape_selected": queried,
                "queried_label_disclosed": queried,
                "oracle_label_request_position": position if queried else None,
                "support_seeds": event_seed_rows})

            del image, features
            if (position + 1) % CHECKPOINT_INTERVAL == 0 or position + 1 == end:
                _save_checkpoint(
                    checkpoint_path, run_identity=run_identity, next_position=position + 1,
                    writers=writers, c64_current=c64_current,
                    c64_state_digest=c64_state_digest, ever_admitted=ever_admitted,
                    score_dynamic=score_dynamic, current_q90=current_q90,
                    ever_group_mask=ever_group_mask,
                    resident_group_mask=resident_group_mask,
                    memory_version_before=memory_version_before,
                    memory_sha_before=memory_sha_before,
                    snapshot_scores=snapshot_scores, snapshot_state=snapshot_state,
                    snapshot_sources=snapshot_sources,
                    snapshot_prior_groups=snapshot_prior_groups,
                    snapshot_c64_q90=snapshot_c64_q90,
                    c64_transport=c64_transport,
                    c64_pairs_before=c64_pairs_before,
                    c64_pairs_after=c64_pairs_after, event_rows=event_rows,
                    query_tape_sha256=query_tape_obj["digest"],
                    elapsed_seconds=elapsed_prior + time.perf_counter() - run_started,
                    torch=torch)
                _manifest_update(output, "smoke" if smoke else "run", "INCOMPLETE",
                                 {"last_class": category,
                                  "last_completed_position": position + 1})
                print(f"{'smoke' if smoke else 'run'}: {category} {position + 1}/{end}",
                      flush=True)
    finally:
        if oracle is not None:
            oracle.close()

    elapsed = elapsed_prior + time.perf_counter() - run_started
    peak_cuda = int(torch.cuda.max_memory_allocated())
    # Every snapshot's score values must cover exactly its fixed group-clean cohort.
    for seed in seeds:
        for t in snapshot_scores[seed]:
            state = snapshot_state[seed].get(t)
            if state is None:
                raise RunnerError(f"{category}/seed{seed}/t{t}: missing frozen snapshot")
            prior = set(snapshot_prior_groups[seed][t])
            all_mask = snapshot_cohort_mask(
                [row.sample_key for row in stream_rows[:end]], snapshot_position=t,
                stream_length=end, prior_sources=prior, cohort="all_remaining")
            if not np.isfinite(snapshot_scores[seed][t][:end][all_mask]).all():
                raise RunnerError(f"{category}/seed{seed}/t{t}: missing eligible snapshot scores")
    private_memory_path = output / "private_checkpoints" / f"{category}_memory_states.pt"
    memory_artifact = {
        "schema": "stage075a.private_memory_states.v1",
        "category": category, "support_seeds": list(seeds),
        "snapshots": {str(seed): {str(t): {
            "memory": _memory_state_to_cpu(snapshot_state[seed][t]),
            "source_map": snapshot_sources[seed][t],
            "prior_admitted_groups": sorted(snapshot_prior_groups[seed][t]),
            "c64_q90": snapshot_c64_q90[seed][t]}
            for t in snapshot_state[seed]} for seed in seeds},
        "final_writers": {str(seed): writers[seed].state_dict() for seed in seeds},
    }
    _atomic_torch_save(private_memory_path, memory_artifact, torch)
    arrays_path = target_root / "scores.npz"
    arrays: dict[str, Any] = {}
    for seed in seeds:
        arrays[f"static_s{seed}"] = score_static[seed]
        arrays[f"dynamic_s{seed}"] = score_dynamic[seed]
        arrays[f"current_q90_s{seed}"] = current_q90[seed]
        arrays[f"ever_group_clean_s{seed}"] = ever_group_mask[seed]
        arrays[f"resident_group_clean_s{seed}"] = resident_group_mask[seed]
        arrays[f"memory_version_before_s{seed}"] = memory_version_before[seed]
        arrays[f"memory_sha_before_s{seed}"] = np.asarray(memory_sha_before[seed], dtype="U64")
        for t, values in snapshot_scores[seed].items():
            arrays[f"snapshot_s{seed}_t{t}"] = values
            arrays[f"snapshot_theta_s{seed}_t{t}"] = np.asarray(
                snapshot_c64_q90[seed][t], dtype=np.float64)
        arrays[f"c64_pairs_before_s{seed}"] = np.asarray(
            c64_pairs_before[seed], dtype=np.float64).reshape(-1, 64)
        arrays[f"c64_pairs_after_s{seed}"] = np.asarray(
            c64_pairs_after[seed], dtype=np.float64).reshape(-1, 64)
    _atomic_npz(arrays_path, arrays)
    trace_path = target_root / "event_trace.jsonl.gz"
    trace_data = "".join(canonical_json(row) + "\n" for row in event_rows).encode("utf-8")
    _atomic_bytes(trace_path, gzip.compress(trace_data, compresslevel=9, mtime=0))
    summary = {
        "schema": "stage075a.category_trajectory.v1", "category": category,
        "status": "SMOKE_COMPLETE" if smoke else "ONLINE_COMPLETE",
        "smoke_only": smoke, "positions_scored": end,
        "query_positions_in_run": sum(position < end for position in query_positions),
        "queried_feedback_positions": len(oracle.requested) if oracle is not None else 0,
        "queried_position_digest": _sha_json(
            oracle.requested if oracle is not None else []),
        "support_seeds": list(seeds), "feature_extractions": feature_count,
        "static_score_full_M0_seed0_parity": not smoke,
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_bytes": peak_cuda,
        "score_calls_seed0_static_reconstruction": score_calls.get(0, 0),
        "memory_by_seed": {
            str(seed): {
                "theta0": theta0[seed],
                "reference_theta": writers[seed].theta_ref,
                "reference_sigma": writers[seed].sigma_ref,
                "actual_content_changes": len(c64_transport[seed]),
                "final_memory_content_version": writers[seed].bank.version,
                "final_memory_sha256": writers[seed].bank.sha256(),
                "final_source_map_sha256": writers[seed].source_map_sha256(),
                "final_dynamic_rows_by_layer": [len(layer) for layer in writers[seed].bank.layers],
                "capacity_by_layer": list(writers[seed].bank.capacities),
                "c64_transport_events": len(c64_transport[seed]),
            } for seed in seeds},
        "snapshot_by_seed": {
            str(seed): {str(t): {
                "memory_content_version": int(snapshot_state[seed][t].version),
                "memory_sha256": snapshot_state[seed][t].sha256(),
                "memory_content_sha256": snapshot_state[seed][t].content_sha256(),
                "prior_admitted_group_count": len(snapshot_prior_groups[seed][t]),
                "prior_admitted_groups_sha256": _sha_json(sorted(
                    hashlib.sha256(("memory-source-group\0" + key).encode()).hexdigest()
                    for key in snapshot_prior_groups[seed][t])),
                "source_map_sha256": hashlib.sha256(canonical_json(
                    snapshot_sources[seed][t]).encode("utf-8")).hexdigest(),
                "c64_q90": snapshot_c64_q90[seed][t],
                "cohort_all_remaining_count": int(np.isfinite(
                    snapshot_scores[seed][t][:end]).sum()),
                "cohort_next256_count": int(np.isfinite(
                    snapshot_scores[seed][t][t:min(end, t + SNAPSHOT_NEXT_WIDTH)]).sum()),
            } for t in snapshot_scores[seed]}
            for seed in seeds},
        "input_protocol_sha256": _sha_file(output / "protocol_manifest.json"),
        "score_artifact_sha256": _sha_file(arrays_path),
        "event_trace_sha256": _sha_file(trace_path),
        "private_memory_states_sha256": _sha_file(private_memory_path),
        "test_labels_opened_for_metrics": False,
        "stage1_or_stage2_run": False,
    }
    _atomic_json(target_root / "trajectory_manifest.json", summary)
    transport_path = target_root / "c64_transport.json"
    _atomic_json(transport_path, {str(seed): c64_transport[seed] for seed in seeds})
    _refresh_checksums(target_root)
    return summary


def snapshot_positions_for_end(end: int) -> tuple[int, ...]:
    return tuple(position for position in SNAPSHOT_POSITIONS if position < end)


def initialize_static_snapshot(snapshot: np.ndarray, static_scores: np.ndarray,
                               end: int) -> None:
    """Copy the frozen M0 prefix into a t=0 cohort-sized score vector."""
    if (end < 0 or len(static_scores) != end or len(snapshot) < end):
        raise RunnerError("Static t=0 snapshot shape does not match run horizon")
    snapshot[:end] = static_scores


def stream_time_bin(bin_edges: list[int], position: int) -> int:
    import bisect
    if len(bin_edges) != 17 or not 0 <= position < bin_edges[-1]:
        raise RunnerError("Q-A-TS temporal-bin edges or stream position are invalid")
    return min(15, bisect.bisect_right(bin_edges, position) - 1)


def _save_checkpoint(path: Path, *, run_identity: str, next_position: int,
                     writers: dict[int, Any], c64_current: dict[int, Any],
                     c64_state_digest: dict[int, str], ever_admitted: dict[int, set[str]],
                     score_dynamic: dict[int, Any], current_q90: dict[int, Any],
                     ever_group_mask: dict[int, Any], resident_group_mask: dict[int, Any],
                     memory_version_before: dict[int, Any], memory_sha_before: dict[int, Any],
                     snapshot_scores: dict[int, dict[int, Any]],
                     snapshot_state: dict[int, dict[int, Any]],
                     snapshot_sources: dict[int, dict[int, list[list[str]]]],
                     snapshot_prior_groups: dict[int, dict[int, set[str]]],
                     snapshot_c64_q90: dict[int, dict[int, float]],
                     c64_transport: dict[int, list[dict[str, Any]]],
                     c64_pairs_before: dict[int, list[Any]],
                     c64_pairs_after: dict[int, list[Any]], event_rows: list[dict[str, Any]],
                     query_tape_sha256: str, elapsed_seconds: float,
                     torch: Any) -> None:
    state = {
        "schema": "stage075a.private_checkpoint.v1", "run_identity": run_identity,
        "next_position": next_position,
        "writers": {str(seed): writers[seed].state_dict() for seed in writers},
        "c64_current": {str(seed): c64_current[seed] for seed in writers},
        "c64_state_digest": {str(seed): c64_state_digest[seed] for seed in writers},
        "ever_admitted": {str(seed): sorted(ever_admitted[seed]) for seed in writers},
        "score_dynamic": {str(seed): score_dynamic[seed] for seed in writers},
        "current_q90": {str(seed): current_q90[seed] for seed in writers},
        "ever_group_mask": {str(seed): ever_group_mask[seed] for seed in writers},
        "resident_group_mask": {str(seed): resident_group_mask[seed] for seed in writers},
        "memory_version_before": {str(seed): memory_version_before[seed] for seed in writers},
        "memory_sha_before": {str(seed): memory_sha_before[seed] for seed in writers},
        "snapshot_scores": {str(seed): {str(t): values for t, values in snap.items()}
                            for seed, snap in snapshot_scores.items()},
        "snapshot_state": {str(seed): {str(t): _memory_state_to_cpu(value)
                                        for t, value in snap.items()}
                           for seed, snap in snapshot_state.items()},
        "snapshot_sources": {str(seed): {str(t): value for t, value in snap.items()}
                             for seed, snap in snapshot_sources.items()},
        "snapshot_prior_groups": {str(seed): {str(t): sorted(value)
                                              for t, value in snap.items()}
                                  for seed, snap in snapshot_prior_groups.items()},
        "snapshot_c64_q90": {str(seed): {str(t): value for t, value in snap.items()}
                             for seed, snap in snapshot_c64_q90.items()},
        "c64_transport": {str(seed): rows for seed, rows in c64_transport.items()},
        "c64_pairs_before": {str(seed): rows for seed, rows in c64_pairs_before.items()},
        "c64_pairs_after": {str(seed): rows for seed, rows in c64_pairs_after.items()},
        "event_rows": event_rows,
        "rng_state_python": random.getstate(), "rng_state_numpy": np.random.get_state(),
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "query_tape_sha256": query_tape_sha256,
        "elapsed_seconds": float(elapsed_seconds),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    temp = Path(name)
    try:
        torch.save(state, temp)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    temp = Path(name)
    try:
        np.savez_compressed(temp, **arrays)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _atomic_torch_save(path: Path, value: Any, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    temp = Path(name)
    try:
        torch.save(value, temp)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _phase_smoke(output: Path, dataset_root: Path, json_root: Path,
                 resume: bool) -> None:
    _require_output(output, create=False, resume=resume)
    manifest = _read_json(output / "run_manifest.json")
    if manifest.get("phase_status", {}).get("freeze") != "PASS":
        raise RunnerError("Protocol freeze must PASS before GPU smoke")
    protocol = _read_json(output / "protocol_manifest.json")
    frozen, _tapes, _scores, _verified = _verify_inputs(dataset_root, json_root)
    rows = []
    for category in DEVELOPMENT_CLASSES:
        path = output / "smoke" / category / "trajectory_manifest.json"
        if path.is_file():
            existing = _read_json(path)
            if existing.get("status") == "SMOKE_COMPLETE":
                if not resume:
                    raise RunnerError(f"Smoke already exists; resume requires --resume: {category}")
                rows.append(existing)
                continue
        result = _run_category(
            output=output, dataset_root=dataset_root, json_root=json_root,
            frozen=frozen, protocol=protocol, category=category, seeds=(0,),
            max_events=SMOKE_LENGTH, smoke=True, resume=resume)
        rows.append(result)
    if len(rows) != len(DEVELOPMENT_CLASSES) or any(
            row.get("status") != "SMOKE_COMPLETE" for row in rows):
        raise RunnerError("The four-category Seed-0 smoke matrix is incomplete")
    _atomic_json(output / "smoke_matrix.json", {
        "schema": "stage075a.seed0_smoke_matrix.v1", "status": "PASS",
        "predeclared_positions": SMOKE_LENGTH, "categories": rows,
        "no_metric_label_join": True, "not_used_for_method_selection": True})
    _manifest_update(output, "smoke", "PASS", {
        "smoke_category_count": len(rows), "smoke_positions": SMOKE_LENGTH})


def _phase_run(output: Path, dataset_root: Path, json_root: Path,
               resume: bool) -> None:
    _require_output(output, create=False, resume=resume)
    manifest = _read_json(output / "run_manifest.json")
    if manifest.get("phase_status", {}).get("freeze") != "PASS":
        raise RunnerError("Protocol freeze must PASS before the full matrix")
    if manifest.get("phase_status", {}).get("smoke") != "PASS":
        raise RunnerError("All four predeclared Seed-0 smoke runs must PASS")
    protocol = _read_json(output / "protocol_manifest.json")
    frozen, _tapes, _scores, _verified = _verify_inputs(dataset_root, json_root)
    rows = []
    for category in DEVELOPMENT_CLASSES:
        existing_path = output / "trajectories" / category / "trajectory_manifest.json"
        if existing_path.is_file():
            existing = _read_json(existing_path)
            if existing.get("status") == "ONLINE_COMPLETE":
                if not resume:
                    raise RunnerError(f"Trajectory exists; use --resume: {category}")
                rows.append(existing)
                continue
        result = _run_category(
            output=output, dataset_root=dataset_root, json_root=json_root,
            frozen=frozen, protocol=protocol, category=category,
            seeds=SUPPORT_SEEDS, max_events=None, smoke=False, resume=resume)
        if result.get("status") != "ONLINE_COMPLETE":
            raise RunnerError(f"{category}: full online matrix did not complete")
        rows.append(result)
        _manifest_update(output, "run", "INCOMPLETE", {
            "last_class": category, "completed_class_count": len(rows)})
    expected = {(category, seed) for category in DEVELOPMENT_CLASSES
                for seed in SUPPORT_SEEDS}
    observed = {(row["category"], int(seed)) for row in rows
                for seed in row["support_seeds"]}
    if observed != expected or len(observed) != 20:
        raise RunnerError("Full class × support-seed completion matrix is not exactly 20/20")
    completion = {
        "schema": "stage075a.online_completion.v1", "status": "PASS",
        "trajectory_count": len(observed), "categories": list(DEVELOPMENT_CLASSES),
        "support_seeds": list(SUPPORT_SEEDS), "trajectories": rows,
        "test_labels_used_for_metrics": False,
        "confirmation_classes_accessed": False, "stage1_or_stage2_run": False}
    _atomic_json(output / "online_complete.json", completion)
    _manifest_update(output, "run", "PASS", {
        "online_trajectory_count": len(observed),
        "online_completion_sha256": _sha_file(output / "online_complete.json")})


def _oracle_ba(labels: np.ndarray, scores: np.ndarray, theta0: float) -> dict[str, Any]:
    if set(np.unique(labels).tolist()) != {0, 1}:
        return {"status": "UNDEFINED_SINGLE_CLASS", "balanced_accuracy": None}
    result = stage05.retrospective_threshold_oracle(
        labels.tolist(), scores.tolist(), float(theta0))
    result["status"] = "DEFINED_RETROSPECTIVE_IN_SAMPLE_DIAGNOSTIC"
    return result


def _paired_block(labels: np.ndarray, static: np.ndarray, dynamic: np.ndarray,
                  mask: np.ndarray, theta0: float, current_theta: np.ndarray) -> dict[str, Any]:
    return paired_score_metrics(labels.tolist(), static.tolist(), dynamic.tolist(),
                                group_mask=mask.tolist(), theta0=theta0,
                                current_theta=current_theta.tolist())


def _phase_evaluate(output: Path, dataset_root: Path, json_root: Path,
                    resume: bool) -> dict[str, Any]:
    _require_output(output, create=False, resume=resume)
    _phase_verify(output)
    run_manifest = _read_json(output / "run_manifest.json")
    if run_manifest.get("phase_status", {}).get("run") != "PASS":
        raise RunnerError("Offline label evaluation requires frozen complete online traces")
    if not (output / "online_complete.json").is_file():
        raise RunnerError("Online completion manifest is missing")
    if (output / "evaluation_metrics.json").is_file():
        if not resume:
            raise RunnerError("Evaluation already exists; it is immutable")
        existing = _read_json(output / "evaluation_metrics.json")
        return existing

    classes: dict[str, Any] = {}
    all_deltas: dict[str, list[float]] = {"auroc": [], "auprc": []}
    for category in DEVELOPMENT_CLASSES:
        metadata_path = json_root / f"{category}.json"
        json_sha = stage05.sha256_file(metadata_path)
        protocol = _read_json(output / "protocol_manifest.json")
        if json_sha != protocol["metadata_sha256_by_class"][category]:
            raise RunnerError(f"{category}: label/evaluation source SHA differs from frozen run")
        labels = np.asarray(stage05.test_labels_for_class(json_root, category), dtype=np.int8)
        trajectory_dir = output / "trajectories" / category
        trajectory = _read_json(trajectory_dir / "trajectory_manifest.json")
        frozen_input_class = _read_json(STAGE05_OUTPUT / "frozen_inputs.json")[
            "classes"][category]
        group_keys = [row["sample_key"] for row in frozen_input_class["stream"]]
        with np.load(trajectory_dir / "scores.npz", allow_pickle=False) as data:
            per_seed: dict[str, Any] = {}
            for seed in SUPPORT_SEEDS:
                static = np.asarray(data[f"static_s{seed}"], dtype=np.float64)
                dynamic = np.asarray(data[f"dynamic_s{seed}"], dtype=np.float64)
                current_theta = np.asarray(data[f"current_q90_s{seed}"], dtype=np.float64)
                group_clean = np.asarray(data[f"ever_group_clean_s{seed}"], dtype=bool)
                resident_clean = np.asarray(data[f"resident_group_clean_s{seed}"], dtype=bool)
                if not (len(labels) == len(static) == len(dynamic) == len(current_theta)):
                    raise RunnerError(f"{category}/seed{seed}: offline labels/scores misaligned")
                all_mask = np.ones(len(labels), dtype=bool)
                result_all = _paired_block(labels, static, dynamic, all_mask,
                                           theta0_from_c64(
                                               _stage05_score_arrays(category, (seed,))[1][seed].tolist()),
                                           current_theta)
                theta0 = theta0_from_c64(
                    _stage05_score_arrays(category, (seed,))[1][seed].tolist())
                result_group = _paired_block(labels, static, dynamic, group_clean,
                                             theta0, current_theta)
                result_resident = _paired_block(labels, static, dynamic, resident_clean,
                                                theta0, current_theta)
                result_same_group = _paired_block(labels, static, dynamic, ~group_clean,
                                                  theta0, current_theta)
                prior_status = trajectory["memory_by_seed"][str(seed)]
                per_seed[str(seed)] = {
                    "support_group_sha256": protocol["class_inputs"][category][
                        "support_group_sha256"][str(seed)],
                    "support_image_sha256": protocol["class_inputs"][category][
                        "support_image_sha256"][str(seed)],
                    "theta0": theta0,
                    "stream_prevalence": float(labels.mean()),
                    "evaluation_denominators": {
                        "images": int(len(labels)), "normals": int(np.sum(labels == 0)),
                        "defects": int(np.sum(labels == 1))},
                    "as_operated_all_events": result_all,
                    "conservative_group_clean": result_group,
                    "resident_source_clean_sensitivity": result_resident,
                    "previously_admitted_same_group_stratum": result_same_group,
                    "score_shift_by_true_label": {
                        "normal": _metric_shift_distribution(
                            dynamic - static, labels == 0),
                        "defect": _metric_shift_distribution(
                            dynamic - static, labels == 1)},
                    "memory_activity": prior_status,
                    "snapshots": {},
                }
                if result_all["delta_auroc"] is not None:
                    all_deltas["auroc"].append(float(result_all["delta_auroc"]))
                if result_all["delta_auprc"] is not None:
                    all_deltas["auprc"].append(float(result_all["delta_auprc"]))
                for t in (int(value) for value in trajectory["snapshot_by_seed"][str(seed)]):
                    snap_scores = np.asarray(data[f"snapshot_s{seed}_t{t}"], dtype=np.float64)
                    snap_theta = float(data[f"snapshot_theta_s{seed}_t{t}"].item())
                    eligible = np.isfinite(snap_scores)
                    next_mask = eligible.copy()
                    next_mask[:t] = False
                    next_mask[min(len(next_mask), t + SNAPSHOT_NEXT_WIDTH):] = False
                    for cohort_name, mask in (("next256_group_clean", next_mask),
                                              ("all_remaining_group_clean", eligible)):
                        block = _paired_block(labels, static, snap_scores, mask,
                                              theta0, np.full(len(labels), snap_theta))
                        idx = np.flatnonzero(mask)
                        if len(idx):
                            block["static_retrospective_oracle"] = _oracle_ba(
                                labels[idx], static[idx], theta0)
                            block["snapshot_retrospective_oracle"] = _oracle_ba(
                                labels[idx], snap_scores[idx], theta0)
                            block["eligible_normal_count"] = int(np.sum(labels[mask] == 0))
                            block["eligible_defect_count"] = int(np.sum(labels[mask] == 1))
                        else:
                            block["static_retrospective_oracle"] = {
                                "status": "UNDEFINED_EMPTY_COHORT", "balanced_accuracy": None}
                            block["snapshot_retrospective_oracle"] = {
                                "status": "UNDEFINED_EMPTY_COHORT", "balanced_accuracy": None}
                            block["eligible_normal_count"] = 0
                            block["eligible_defect_count"] = 0
                        base_positions = np.zeros(len(labels), dtype=bool)
                        if cohort_name == "next256_group_clean":
                            base_positions[t:min(len(labels), t + SNAPSHOT_NEXT_WIDTH)] = True
                        else:
                            base_positions[t:] = True
                        block["candidate_position_count"] = int(base_positions.sum())
                        block["prior_source_group_exclusion_count"] = int(
                            base_positions.sum() - mask.sum())
                        block["positions_before_snapshot_excluded"] = int(t)
                        block["group_composition"] = {
                            "eligible_normals": int(np.sum(labels[mask] == 0)),
                            "eligible_defects": int(np.sum(labels[mask] == 1)),
                            "eligible_official_groups": int(len({
                                group_keys[position] for position in np.flatnonzero(mask)})),
                        }
                        per_seed[str(seed)]["snapshots"].setdefault(str(t), {
                            "snapshot_manifest": trajectory["snapshot_by_seed"][str(seed)][str(t)],
                            "snapshot_C64_q90": snap_theta})[cohort_name] = block
            classes[category] = {
                "stream_images": int(len(labels)),
                "official_sample_groups": int(protocol["class_inputs"][category][
                    "official_groups"]),
                "prevalence": float(labels.mean()),
                "prevalence_disclosure": "four development classes previously exposed in Stage 0.5",
                "support_seeds": per_seed,
                "class_seed_aggregate": {
                    "delta_auroc": summarize_numeric([
                        row["as_operated_all_events"]["delta_auroc"]
                        for row in per_seed.values()
                        if row["as_operated_all_events"]["delta_auroc"] is not None]),
                    "delta_auprc": summarize_numeric([
                        row["as_operated_all_events"]["delta_auprc"]
                        for row in per_seed.values()
                        if row["as_operated_all_events"]["delta_auprc"] is not None]),
                }}

    output_metrics = {
        "schema": "stage075a.offline_evaluation.v1", "status": "PASS",
        "label_phase_started_after_all_online_traces_frozen": True,
        "test_mask_accessed": False, "confirmation_classes_accessed": False,
        "stage1_or_stage2_run": False,
        "development_label_status": "PREVIOUSLY_EXPOSED_IN_STAGE05; USED_OFFLINE_HERE",
        "classes": classes,
        "all_20_as_operated_delta_auroc": summarize_numeric(all_deltas["auroc"]),
        "all_20_as_operated_delta_auprc": summarize_numeric(all_deltas["auprc"]),
        "ranking_scope_limitation": (
            "Dynamic full-stream scores are prequential under varying memory states; "
            "this is not ranking by one fixed learned model."),
        "independence_limitation": (
            "Five camera views are not treated as five independent manufactured specimens."),
    }
    _atomic_json(output / "evaluation_metrics.json", output_metrics)
    _manifest_update(output, "evaluate", "PASS", {
        "offline_evaluation_sha256": _sha_file(output / "evaluation_metrics.json"),
        "development_labels_joined_after_online_freeze": True,
        "test_masks_opened": False})
    return output_metrics


def _metric_shift_distribution(shifts: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(shifts[mask], dtype=np.float64)
    if not values.size:
        return {"n": 0, "quantiles_linear": None, "mean": None}
    return {"n": int(values.size), "mean": float(values.mean()),
            "quantiles_linear": {str(q): float(np.quantile(values, q, method="linear"))
                                 for q in (0, .05, .25, .5, .75, .95, 1)}}


def _transport_summary(output: Path) -> dict[str, Any]:
    by_class: dict[str, Any] = {}
    all_rows = []
    for category in DEVELOPMENT_CLASSES:
        data = _read_json(output / "trajectories" / category / "c64_transport.json")
        per_seed = {}
        for seed in SUPPORT_SEEDS:
            rows = data[str(seed)]
            all_rows.extend(rows)
            per_seed[str(seed)] = _summarize_transport_rows(rows)
        by_class[category] = per_seed
    return {"schema": "stage075a.c64_transport_summary.v1",
            "status": "PASS", "per_class_seed": by_class,
            "all_updates": _summarize_transport_rows(all_rows),
            "interpretation": {
                "theta_transport_and_theta_SCQ_are_diagnostics_only": True,
                "normal_C64_transport_does_not_establish_defect_score_transport": True,
                "translation_equals_direct_higher_q90_only_under_pure_translation_and_"
                "theta_previous_equals_higher_q90(c_before)": True,
                "B8_implemented_or_validated": False}}


def _summarize_transport_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actions = {
        "append": [row for row in rows if row["writer_appended"] > 0],
        "replacement": [row for row in rows if row["writer_replaced"] > 0],
        "skip": [row for row in rows if row["writer_skipped"] > 0],
    }
    return {
        "actual_content_change_count": len(rows),
        "median_paired_shift": summarize_numeric(
            [row["median_paired_shift"] for row in rows]),
        "shift_heterogeneity_H": summarize_numeric(
            [row["shift_heterogeneity_H_IQR"] for row in rows]),
        "theta_q90_change": summarize_numeric([row["theta_q90_change"] for row in rows]),
        "theta_q90_change_sign_counts": {
            "positive": int(sum(row["theta_q90_change"] > 0 for row in rows)),
            "zero": int(sum(row["theta_q90_change"] == 0 for row in rows)),
            "negative": int(sum(row["theta_q90_change"] < 0 for row in rows))},
        "translation_q90_disagreement": summarize_numeric(
            [row["disagreement_normalized_by_after_IQR"] for row in rows]),
        "paired_rescore_seconds": summarize_numeric(
            [row["paired_rescore_seconds"] for row in rows]),
        "rank_reordering": {
            "strict_discordant_pair_count": int(sum(
                row["rank_reordering"]["strict_discordant_pairs"] for row in rows)),
            "newly_tied_pair_count": int(sum(
                row["rank_reordering"]["newly_tied_pairs"] for row in rows)),
            "newly_ordered_pair_count": int(sum(
                row["rank_reordering"]["newly_ordered_pairs"] for row in rows)),
            "mean_anchor_rank_movement": summarize_numeric([
                row["rank_reordering"]["mean_absolute_anchor_rank_movement"]
                for row in rows])},
        "uniformity": {
            "numeric_uniform_shift_count": int(sum(
                row["uniformity_diagnostic"]["numeric_uniform_shift"] for row in rows)),
            "heterogeneity_band_counts": {
                ratio: int(sum(row["uniformity_diagnostic"][
                    "heterogeneity_to_before_IQR_cutoffs"][ratio] for row in rows))
                for ratio in ("0.01", "0.05", "0.1")}},
        "writer_activity_association": {
            "events_with_append": int(sum(row["writer_appended"] > 0 for row in rows)),
            "events_with_replacement": int(sum(row["writer_replaced"] > 0 for row in rows)),
            "events_with_skip": int(sum(row["writer_skipped"] > 0 for row in rows)),
            "appended_prototypes": int(sum(row["writer_appended"] for row in rows)),
            "replaced_prototypes": int(sum(row["writer_replaced"] for row in rows)),
            "skipped_prototypes": int(sum(row["writer_skipped"] for row in rows)),
            "median_shift_conditioned_on_action": {
                action: summarize_numeric([row["median_paired_shift"] for row in action_rows])
                for action, action_rows in actions.items()}}}


def _phase_publish(output: Path) -> dict[str, Any]:
    _require_output(output, create=False, resume=True)
    manifest = _read_json(output / "run_manifest.json")
    if manifest.get("phase_status", {}).get("evaluate") != "PASS":
        raise RunnerError("Publishing requires completed offline metrics")
    package = ROOT / "docs/calibration/audit/stage075a_greedy_ranking_v1"
    if package.exists():
        raise RunnerError(f"Refusing to overwrite immutable audit package: {package}")
    report_targets = [
        ROOT / "docs/calibration/STAGE075A_GREEDY_RANKING_FALSIFICATION_REPORT.md",
        ROOT / "docs/calibration/STAGE075A_C64_MEMORY_TRANSPORT_REPORT.md",
        ROOT / "docs/calibration/STAGE075A_RESEARCH_DECISION_CANDIDATE.md",
    ]
    existing_reports = [str(path) for path in report_targets if path.exists()]
    if existing_reports:
        raise RunnerError(f"Refusing to overwrite existing Stage 0.75-A reports: {existing_reports}")
    metrics = _read_json(output / "evaluation_metrics.json")
    completion = _read_json(output / "online_complete.json")
    protocol = _read_json(output / "protocol_manifest.json")
    transport = _transport_summary(output)
    ranking_report = _render_ranking_report(metrics, completion)
    c64_report = _render_c64_report(transport)
    decision = _render_decision_candidate(metrics, transport, completion)
    _atomic_json(output / "c64_transport_summary.json", transport)
    _atomic_bytes(output / "ranking_report.md", ranking_report.encode("utf-8"))
    _atomic_bytes(output / "c64_transport_report.md", c64_report.encode("utf-8"))
    _atomic_bytes(output / "research_decision_candidate.md", decision.encode("utf-8"))
    final_manifest = {**manifest,
                      "phase_status": {**manifest["phase_status"], "publish": "PASS"},
                      "audit_package": "docs/calibration/audit/stage075a_greedy_ranking_v1",
                      "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _atomic_json(output / "run_manifest.json", final_manifest)
    _atomic_json(output / "audit_linkage.json", {
        "schema": "stage075a.audit_linkage.v1",
        "local_output_directory": str(output.resolve()),
        "run_manifest_sha256": _sha_file(output / "run_manifest.json"),
        "protocol_manifest_sha256": _sha_file(output / "protocol_manifest.json"),
        "online_completion_sha256": _sha_file(output / "online_complete.json"),
        "evaluation_metrics_sha256": _sha_file(output / "evaluation_metrics.json"),
        "source_trace_hashes_by_class": {
            category: {
                "scores_sha256": _sha_file(output / "trajectories" / category / "scores.npz"),
                "event_trace_sha256": _sha_file(output / "trajectories" / category /
                                                 "event_trace.jsonl.gz"),
                "transport_sha256": _sha_file(output / "trajectories" / category /
                                               "c64_transport.json")}
            for category in DEVELOPMENT_CLASSES},
        "per_image_score_traces_in_public_package": False,
        "raw_labels_masks_features_images_weights_checkpoints_in_public_package": False})
    package.mkdir(parents=True)
    _atomic_json(package / "protocol_manifest.json", protocol)
    _atomic_bytes(package / "STAGE075A_EXPERIMENT_PROTOCOL_v1.md",
                  (ROOT / "docs/calibration/STAGE075A_EXPERIMENT_PROTOCOL_v1.md").read_bytes())
    _atomic_json(package / "validation_results.json",
                 _read_json(output / "validation_results.json"))
    _atomic_json(package / "online_completion_matrix.json", {
        "schema": completion["schema"], "status": completion["status"],
        "trajectory_count": completion["trajectory_count"],
        "categories": completion["categories"], "support_seeds": completion["support_seeds"],
        "trajectories": [{key: row[key] for key in (
            "category", "status", "positions_scored", "query_positions_in_run",
            "queried_feedback_positions", "support_seeds", "feature_extractions",
            "elapsed_seconds", "peak_cuda_allocated_bytes", "memory_by_seed",
            "snapshot_by_seed", "score_artifact_sha256", "event_trace_sha256")}
            for row in completion["trajectories"]]})
    _atomic_json(package / "ranking_metrics.json", metrics)
    _atomic_json(package / "c64_transport_summary.json", transport)
    _atomic_json(package / "method_fidelity_and_scope.json", {
        "native_v22_scorer": "STAGE05_SCORE_TAPE_PARITY_AND_SEED0_FULL_RECONSTRUCTION_PASS",
        "shared_native_greedy_writer": "EXISTING_IMPLEMENTATION; NO_SOURCE_CHANGES",
        "support_and_C64": "STAGE05_IDENTITIES_REVERIFIED; EXACT_M0_C64_PARITY_REQUIRED",
        "memory_update": "EXISTING_V0.2_SELECTION_AND_GREEDY_BRIDGE",
        "b8": "NOT_IMPLEMENTED_NOT_VALIDATED",
        "stage1_or_stage2": "NOT_RUN",
        "confirmation_classes": "NOT_ACCESSED",
        "historical_sources_modified": False})
    _atomic_json(package / "audit_linkage.json", _read_json(output / "audit_linkage.json"))
    _atomic_json(package / "gate_matrix.json", _gate_matrix(output, protocol, metrics))
    _atomic_bytes(package / "STAGE075A_GREEDY_RANKING_FALSIFICATION_REPORT.md",
                  ranking_report.encode("utf-8"))
    _atomic_bytes(package / "STAGE075A_C64_MEMORY_TRANSPORT_REPORT.md",
                  c64_report.encode("utf-8"))
    _atomic_bytes(package / "STAGE075A_RESEARCH_DECISION_CANDIDATE.md",
                  decision.encode("utf-8"))
    package_checksums = _refresh_checksums(package)
    _atomic_json(package / "artifact_verification.json", {
        "schema": "stage075a.artifact_verification.v1", "status": "PASS",
        "checksum_algorithm": "SHA-256", "checked_file_count": len(package_checksums),
        "private_raw_artifacts_excluded": True,
        "verification_scope": "all files in this immutable compact audit package"})
    _refresh_checksums(package)
    for target, content in zip(report_targets, (ranking_report, c64_report, decision)):
        _atomic_bytes(target, content.encode("utf-8"))
    _refresh_checksums(output)
    return {"status": "PASS", "package": str(package),
            "package_file_count": len(list(package.rglob("*"))),
            "package_checksum_manifest_sha256": _sha_file(package / "checksums.json")}


def _gate_matrix(output: Path, protocol: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    validation = _read_json(output / "validation_results.json")
    smoke = _read_json(output / "smoke_matrix.json")
    completion = _read_json(output / "online_complete.json")
    smoke_status = "PASS" if smoke.get("status") == "PASS" else "FAIL"
    run_status = "PASS" if (completion.get("status") == "PASS" and
                             completion.get("trajectory_count") == 20) else "FAIL"
    return {
        "schema": "stage075a.gates.v1",
        "gates": {
            "branch_and_parent_recorded": "PASS" if protocol.get("branch") == EXPECTED_BRANCH else "FAIL",
            "stage05_inputs_and_query_tapes_verified": "PASS" if protocol.get("class_inputs") else "FAIL",
            "python_compilation_and_calibration_unittests": validation.get("status", "FAIL"),
            "protected_historical_source_integrity": validation.get(
                "protected_source_integrity", {}).get("status", "FAIL"),
            "git_diff_check": validation.get("git_diff_check", "FAIL"),
            "seed0_full_M0_static_parity": "PASS" if all(
                row.get("static_score_full_M0_seed0_parity") is True
                for row in completion["trajectories"]) else "FAIL",
            "all_four_seed0_smokes": smoke_status,
            "full_20_trajectory_matrix": run_status,
            "causal_queried_only_feedback": "PASS" if any(
                "unqueried_labels_cannot_change_writer_or_score_trace" in test
                for test in validation.get("passed_test_ids", [])) else "FAIL",
            "native_writer_and_memory_version": "PASS" if run_status == "PASS" else "FAIL",
            "group_clean_and_snapshot_cohorts": "PASS" if run_status == "PASS" else "FAIL",
            "c64_paired_transport": "PASS" if run_status == "PASS" else "FAIL",
            "offline_metrics_after_trace_freeze": "PASS" if metrics.get("status") == "PASS" else "FAIL",
            "B8_implementation_or_validation": "NOT_RUN_BY_SCOPE",
            "stage1_or_stage2": "NOT_RUN_BY_SCOPE",
            "confirmation_class_access": "NOT_ACCESSED",
            "scientific_independent_review": "NOT_PERFORMED",
        },
        "interpretation": "Stage 0.75-A execution evidence is not independent scientific validation.",
        "protocol_manifest_sha256": _sha_file(output / "protocol_manifest.json"),
        "evaluation_metrics_sha256": _sha_file(output / "evaluation_metrics.json"),
    }


def _render_ranking_report(metrics: dict[str, Any], completion: dict[str, Any]) -> str:
    lines = [
        "# Stage 0.75-A — Native Greedy Ranking Falsification Report",
        "",
        "Status: **EXECUTED EXPLORATORY DEVELOPMENT AUDIT; INDEPENDENT RESEARCH REVIEW REQUIRED**.",
        "",
        "The four development categories were already exposed in Stage 0.5. No confirmation category, mask, Stage 1 or Stage 2 result was accessed here. Dynamic full-stream scores are prequential under changing memory states and do not describe one fixed learned model.",
        "",
        "## Full 20 trajectory results",
        "",
        "| Category | Seed | N | Defects | Static AUROC | Greedy AUROC | Δ AUROC | Static AP | Greedy AP | Δ AP | Memory changes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, cat in metrics["classes"].items():
        for seed, row in cat["support_seeds"].items():
            block = row["as_operated_all_events"]
            a, b = block["static"], block["dynamic"]
            changes = row["memory_activity"]["actual_content_changes"]
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                category, seed, block["n_included"], row["evaluation_denominators"]["defects"],
                _fmt(a["auroc"]), _fmt(b["auroc"]), _fmt(block["delta_auroc"]),
                _fmt(a["auprc_average_precision"]), _fmt(b["auprc_average_precision"]),
                _fmt(block["delta_auprc"]), changes))
        aggregate = cat["class_seed_aggregate"]
        lines.extend(["", f"### {category}", "",
                      f"Five-seed ΔAUROC: {_fmt_summary(aggregate['delta_auroc'])}; "
                      f"ΔAUPRC: {_fmt_summary(aggregate['delta_auprc'])}.", "",
                      "Per seed, as-operated, conservative group-clean, resident-source-clean, and previously admitted same-group strata are in the machine-readable metrics. Single-class strata mark AUROC and balanced accuracy undefined."])
    lines.extend(["", "## Fixed snapshots", "",
                  "For each seed, snapshots at t=0, 400, 800, 1600, 2400 and 3200 use writes strictly before t. The next-256 and all-remaining cohorts are the same frozen group-clean positions for static and snapshot scores. Metrics and cohort denominators are in `ranking_metrics.json`.", "",
                  "## Runtime and scope", "",
                  f"Four class trajectories completed; full matrix count: {completion['trajectory_count']}. Runtime, CUDA peak allocation, per-snapshot C64 thresholds, memory identities and output provenance are recorded in `online_completion_matrix.json`.", "",
                  "No superiority, population-risk guarantee, or independent validation claim is made. Five camera views are not treated as five independent manufactured products.", ""])
    return "\n".join(lines)


def _render_c64_report(transport: dict[str, Any]) -> str:
    overall = transport["all_updates"]
    return "\n".join([
        "# Stage 0.75-A — C64 Memory-Transport Audit",
        "",
        "The analysis pairs scores on the same 64 protected training-normal C64 images at every actual dynamic prototype-content change. C64 images remain immutable and never enter the mutable Greedy bank.",
        "",
        f"Actual content changes: {overall['actual_content_change_count']}.",
        f"Median paired score shift summary: {_fmt_summary(overall['median_paired_shift'])}.",
        f"Shift heterogeneity H=Q75−Q25: {_fmt_summary(overall['shift_heterogeneity_H'])}.",
        f"Normalized translation versus direct higher-q90 disagreement: {_fmt_summary(overall['translation_q90_disagreement'])}.",
        f"Paired C64 rescoring time in seconds: {_fmt_summary(overall['paired_rescore_seconds'])}.",
        f"Numeric-uniform shifts: {overall['uniformity']['numeric_uniform_shift_count']}.",
        f"Descriptive 1%, 5%, 10% of pre-update C64 IQR heterogeneity counts: {overall['uniformity']['heterogeneity_band_counts']}.",
        f"Tie-aware C64 rank reorderings: {overall['rank_reordering']}.",
        "",
        "The q90 convention is the higher order statistic at index ceil(0.90×63)=57. H uses linearly interpolated quartiles. Median is NumPy's even-sample median. The diagnostic scale floor is 1e-12. Uniformity bands are descriptive sensitivity cutoffs, not B8 hyperparameters.",
        "",
        "A scalar threshold translation and direct q90 coincide under a pure translation when the starting threshold is the same higher-q90 order statistic of the pre-update C64 scores. Heterogeneity, rank reorderings, and a starting threshold that does not equal that q90 produce disagreement.",
        "",
        "These measurements concern normal C64 scores only. They cannot establish that physical-defect scores underwent the same transformation. B8 remains unimplemented and unvalidated.",
        "",
        "Per category and support seed results are in `c64_transport_summary.json`; paired anchor scores are retained only in local raw outputs.",
        ""])


def _render_decision_candidate(metrics: dict[str, Any], transport: dict[str, Any],
                               completion: dict[str, Any]) -> str:
    classes = metrics["classes"]
    deteriorated = []
    for category, row in classes.items():
        vals = [seed["as_operated_all_events"]["delta_auroc"]
                for seed in row["support_seeds"].values()
                if seed["as_operated_all_events"]["delta_auroc"] is not None]
        if vals and float(np.mean(vals)) < 0:
            deteriorated.append(category)
    return "\n".join([
        "# Stage 0.75-A — Research Decision Candidate",
        "",
        "**Decision status: RESEARCHER_REVIEW_REQUIRED.** This report records an exploratory result for the four development categories already exposed in Stage 0.5. It is not independent scientific validation and does not authorize a subsequent stage.",
        "",
        f"The preregistered full matrix completed {completion['trajectory_count']}/20 class × seed trajectories.",
        f"Categories with negative five-seed mean as-operated ΔAUROC: {', '.join(deteriorated) if deteriorated else 'none by the descriptive mean'}.",
        f"Paired C64 native-memory content changes: {transport['all_updates']['actual_content_change_count']}.",
        "",
        "## Interpretation",
        "",
        "The Native Greedy ranking question is answered only for these four exposed development categories, the frozen four-support plus additional C64 design, and the fixed 3% seed-37 query tape. The dynamic stream score vector is produced under changing memory and temporal feedback; it is not a fixed-model ranking estimate. Group-clean and fixed-snapshot comparisons address specific same-group and chronology concerns but do not create independent product-level replication.",
        "",
        "C64 transport can describe normal-side score movement. It cannot show that defect scores move by the same amount. A measurable median shift alone is insufficient evidence for B8; rank reorderings and heterogeneity are reported separately. B8 was neither implemented nor tuned.",
        "",
        "## Requested researcher decisions",
        "",
        "1. Review whether the native writer reference policy (per-seed M0 higher-q90 C64 threshold, sigma_ref=0.02, novelty=0) remains the intended source-faithful reference for later work.",
        "2. Review negative, undefined, and group-clean results before proposing any Stage 0.75-B or Stage 1 protocol.",
        "3. Decide whether the C64 normal-transport evidence is sufficient to motivate a separately frozen B8 research design; this audit does not make that decision.",
        "4. Preserve Mint/PCB confirmation isolation and do not treat empirical comparability as theoretical equivalence.",
        "",
        "No Stage 0.75-B/C, Stage 1, Stage 2, confirmation evaluation, controller change, query-rate change, or B8 implementation was performed.",
        ""])


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def _fmt_summary(value: dict[str, Any]) -> str:
    if value.get("n", 0) == 0:
        return "undefined (n=0)"
    return (f"mean={_fmt(value['mean'])}, SD={_fmt(value['sd_sample'])}, "
            f"min={_fmt(value['min'])}, median={_fmt(value['median'])}, "
            f"max={_fmt(value['max'])}, n={value['n']}")


def _phase_verify(output: Path) -> dict[str, Any]:
    _require_output(output, create=False, resume=True)
    checks = _read_json(output / "checksums.json")["files"]
    verified = []
    for relative, expected in checks.items():
        path = output / relative
        if not path.is_file() or _sha_file(path) != expected:
            raise RunnerError(f"Stage 0.75-A artifact checksum failed: {relative}")
        verified.append(relative)
    if (output / "online_complete.json").exists():
        rows = _read_json(output / "online_complete.json")["trajectories"]
        if len(rows) != 4 or sum(len(row["support_seeds"]) for row in rows) != 20:
            raise RunnerError("Artifact verifier did not find 20 complete trajectories")
    package = ROOT / "docs/calibration/audit/stage075a_greedy_ranking_v1"
    package_checksums = package / "checksums.json"
    if package_checksums.is_file():
        for relative, expected in _read_json(package_checksums)["files"].items():
            path = package / relative
            if not path.is_file() or _sha_file(path) != expected:
                raise RunnerError(f"Compact audit package checksum failed: {relative}")
    return {"status": "PASS", "file_count": len(verified),
            "checksums_sha256": _sha_file(output / "checksums.json")}


def _run_validation() -> dict[str, Any]:
    import io
    import unittest
    from experiments.calibration.source_provenance import verify_protected_sources

    compilation_files = [ROOT / relative for relative in SOURCE_FILES]
    compile_errors = []
    for path in compilation_files:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except Exception as exc:
            compile_errors.append({"file": path.relative_to(ROOT).as_posix(),
                                   "error": f"{type(exc).__name__}: {exc}"})
    suite = unittest.TestLoader().discover(str(ROOT / "tests" / "calibration"))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    protected = verify_protected_sources()
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT,
                          capture_output=True, text=True, check=False)
    # unittest.TestResult does not expose passed cases; a PASS for the full suite
    # certifies these stable discovered Stage 0.75-A IDs.
    import importlib
    discovered = unittest.TestLoader().loadTestsFromModule(
        importlib.import_module("tests.calibration.test_stage075a"))
    stage075_ids: list[str] = []
    def walk(test):
        if isinstance(test, unittest.TestSuite):
            for child in test:
                walk(child)
        else:
            stage075_ids.append(test.id())
    walk(discovered)
    status = "PASS" if (result.wasSuccessful() and not compile_errors and
                        protected.get("status") == "PASS" and
                        diff.returncode == 0) else "FAIL"
    return {
        "schema": "stage075a.validation.v1", "status": status,
        "python_executable": sys.executable,
        "compilation": {"status": "PASS" if not compile_errors else "FAIL",
                        "files_checked": [path.relative_to(ROOT).as_posix()
                                          for path in compilation_files],
                        "errors": compile_errors},
        "calibration_unittest_suite": {
            "status": "PASS" if result.wasSuccessful() else "FAIL",
            "tests_run": result.testsRun,
            "failures": [str(test) for test, _ in result.failures],
            "errors": [str(test) for test, _ in result.errors],
            "stage075a_test_ids": stage075_ids,
            "text_output_sha256": hashlib.sha256(
                stream.getvalue().encode("utf-8")).hexdigest()},
        "passed_test_ids": stage075_ids if result.wasSuccessful() else [],
        "protected_source_integrity": protected,
        "git_diff_check": "PASS" if diff.returncode == 0 else "FAIL",
        "git_diff_check_output": diff.stdout + diff.stderr,
        "stage1_or_stage2_run": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--phase", required=True,
                        choices=("freeze", "smoke", "run", "evaluate", "publish",
                                 "verify", "all"))
    result.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    result.add_argument("--json-root", type=Path, default=DEFAULT_JSON_ROOT)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--resume", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    dataset_root, json_root = args.dataset_root, args.json_root
    if not dataset_root.is_dir() or not json_root.is_dir():
        print("BLOCKED: Real-IAD dataset/metadata roots are unavailable", file=sys.stderr)
        return 2
    try:
        _environment()
        if args.phase == "all":
            _phase_freeze(output, dataset_root, json_root, args.resume)
            _phase_smoke(output, dataset_root, json_root, args.resume)
            _phase_run(output, dataset_root, json_root, args.resume)
            _phase_evaluate(output, dataset_root, json_root, args.resume)
            _phase_publish(output)
        elif args.phase == "freeze":
            _phase_freeze(output, dataset_root, json_root, args.resume)
        elif args.phase == "smoke":
            _phase_smoke(output, dataset_root, json_root, args.resume)
        elif args.phase == "run":
            _phase_run(output, dataset_root, json_root, args.resume)
        elif args.phase == "evaluate":
            _phase_evaluate(output, dataset_root, json_root, args.resume)
        elif args.phase == "publish":
            _phase_publish(output)
        else:
            result = _phase_verify(output)
            print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        if output.exists() and (output / "run_manifest.json").is_file():
            try:
                _manifest_update(output, args.phase, "BLOCKED",
                                 {"last_error": f"{type(exc).__name__}: {exc}"})
                (output / "INCOMPLETE").write_text(
                    f"Stage 0.75-A incomplete: {type(exc).__name__}: {exc}\n",
                    encoding="utf-8")
                _refresh_checksums(output)
            except Exception:
                pass
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
