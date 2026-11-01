"""R2-EXP2A causal attribution of C1 correction-strength components.

Stage A runs one source-bound native feature capture per authorized category.
All factorial arms consume the same C0 maps, query tape and Greedy memory
trajectory, while their native SignedCorrectionCodebook states evolve
independently. Stage B opens only the first 512 category outcomes after every
Stage A score and transition has been sealed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
EXPECTED_PARENT = "4db0a1b701830339019c944d6862d7b25b6e7d14"
EXPECTED_PYTHON = Path(r"D:\anaconda3\envs\torch\python.exe")
DATASET_ROOT = Path(r"G:\dataset\realiad")
JSON_ROOT = DATASET_ROOT / "jsons" / "realiad_jsons"
STAGE05 = ROOT / "outputs/calibration/stage05_exploratory_20261029"
R1_SWITCH = ROOT / (
    "outputs/calibration/"
    "r1_digest_contract_v1_engineering_r1s4b-switch-s0-512-20261031-a1")
CATEGORIES = ("switch", "audiojack", "regulator", "rolled_strip_base")
ARMS = ("c1_full", "c1_no_outcome_weight", "c1_no_margin", "c1_no_both")
SCORE_ARMS = ("c0", *ARMS, "c2_a002", "c2_a020")
MAX_EVENTS = 512
QUERY_RATE = 0.03
QUERY_SEED = 37
SUPPORT_SEED = 0
MAX_OUTPUT_BYTES = 2 * 1024**3
FAILURE_RESERVE_BYTES = 64 * 1024
MIN_FREE_BYTES = 10 * 1024**3
MAX_RUNTIME_SECONDS = 180 * 60
OUTPUT_PREFIX = "r1_digest_contract_v1_exp2a"
PROFILE_ID = "r2.exp2a.c1_strength.factorial.v1"
RUNTIME_PROFILE_ID = "r2.exp2a.job_object.180m.researcher_override.v1"
AUTHORIZATION_ID = "user-task:R2-EXP2A:2026-11-01"
REPORT_PATH = "docs/calibration/audit/R2_EXP2A_C1_STRENGTH_ATTRIBUTION.md"
AGGREGATE_NAME = "sanitized_aggregate.json"
FIXED_BETA_MIN = 0.02


class Exp2AError(RuntimeError):
    pass


class Exp2AIntegrityError(Exp2AError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def factorial_effects(f11: float, f10: float, f01: float,
                      f00: float) -> dict[str, float]:
    """Factor index 1 is margin; index 2 is outcome weighting."""
    values = (float(f11), float(f10), float(f01), float(f00))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Factorial AUROCs must be finite")
    return {
        "outcome_effect_margin_enabled": values[0] - values[1],
        "outcome_effect_margin_disabled": values[2] - values[3],
        "margin_effect_outcome_enabled": values[0] - values[2],
        "margin_effect_outcome_disabled": values[1] - values[3],
        "interaction": values[0] - values[1] - values[2] + values[3],
    }


def category_equal_aggregate(by_category: Mapping[str, Mapping[str, Any]],
                             metrics: Sequence[str]) -> dict[str, Any]:
    if tuple(by_category) != CATEGORIES:
        raise ValueError("Category aggregate must preserve the four frozen categories")
    result: dict[str, Any] = {}
    for metric in metrics:
        values = {name: by_category[name].get(metric) for name in CATEGORIES}
        defined_categories = [
            name for name, value in values.items()
            if value is not None and math.isfinite(float(value))]
        finite = [float(values[name]) for name in defined_categories]
        result[metric] = {
            "by_category": values,
            "category_equal_macro": (float(np.mean(finite)) if len(finite) == len(CATEGORIES)
                                     else None),
            "category_equal_macro_defined": (
                float(np.mean(finite)) if finite else None),
            "defined_category_count": len(defined_categories),
            "defined_categories": defined_categories,
            "undefined_categories": [
                name for name in CATEGORIES if name not in defined_categories],
            "worst_category": (min(finite) if len(finite) == len(CATEGORIES) else None),
            "worst_defined_category": min(finite) if finite else None,
            "positive_category_count": sum(value > 0 for value in finite),
            "negative_category_count": sum(value < 0 for value in finite),
            "category_range": (max(finite) - min(finite)
                               if len(finite) == len(CATEGORIES) else None),
            "category_range_defined": max(finite) - min(finite) if finite else None,
            "category_sd_population": (float(np.std(finite))
                                       if len(finite) == len(CATEGORIES) else None),
        }
    return result


def _coverage_aware_development_verdict(
        parity_status: str, factorial_effects: Mapping[str, Mapping[str, Any]],
        activation_fractions: Sequence[float | None]) -> str:
    if parity_status != "PASS":
        return "REFERENCE_PARITY_UNVERIFIED"
    margin = factorial_effects["margin_effect_outcome_enabled"]
    outcome = factorial_effects["outcome_effect_margin_enabled"]
    evaluable = min(int(margin.get("defined_category_count", 0)),
                    int(outcome.get("defined_category_count", 0)))
    if evaluable < 3:
        return "EXPERIMENT_INCOMPLETE"
    if not any(value is not None and float(value) > 0.0
               for value in activation_fractions):
        return "INSUFFICIENT_ACTIVATION"
    margin_mean = margin.get("category_equal_macro_defined")
    outcome_mean = outcome.get("category_equal_macro_defined")
    if margin_mean is None or outcome_mean is None:
        return "EXPERIMENT_INCOMPLETE"
    if float(margin_mean) <= 0.0 and float(outcome_mean) <= 0.0:
        return "NO_CLEAR_INCREMENTAL_BENEFIT"
    if (int(margin.get("positive_category_count", 0)) < evaluable or
            int(outcome.get("positive_category_count", 0)) < evaluable):
        return "EFFECT_HETEROGENEOUS"
    return "CAUSAL_EFFECT_SUPPORTED_IN_DEVELOPMENT"


def _apply_coverage_aware_aggregation(aggregate: dict[str, Any]) -> dict[str, Any]:
    """Add defined-category summaries while keeping strict four-class macros null."""
    by_category = aggregate["outcome_metrics_by_category"]
    macro = aggregate["category_equal_aggregation"]
    macro["auroc_by_arm"] = category_equal_aggregate(
        {category: by_category[category]["auroc_by_arm"]
         for category in CATEGORIES}, SCORE_ARMS)
    macro["factorial_effects"] = category_equal_aggregate(
        {category: by_category[category]["factorial_effects"]
         for category in CATEGORIES}, (
             "outcome_effect_margin_enabled", "outcome_effect_margin_disabled",
             "margin_effect_outcome_enabled", "margin_effect_outcome_disabled",
             "interaction"))
    c1_vs_c2 = {
        category: by_category[category]["paired_auroc_differences"][
            "c1_full"]["auroc_difference_vs_c2_a002"]
        for category in CATEGORIES}
    macro["auroc_by_arm"]["c1_full_minus_c2_a002"] = category_equal_aggregate(
        {category: {"effect": c1_vs_c2[category]} for category in CATEGORIES},
        ("effect",))["effect"]
    macro["auroc_by_arm"]["c1_full_minus_c2_a002"]["by_category"] = c1_vs_c2

    questions = aggregate["scientific_questions"]
    effects = macro["factorial_effects"]
    questions.update({
        "q1_margin_adaptation_defined_category_equal_macro": effects[
            "margin_effect_outcome_enabled"]["category_equal_macro_defined"],
        "q2_outcome_weighting_defined_category_equal_macro": effects[
            "outcome_effect_margin_enabled"]["category_equal_macro_defined"],
        "q3_interaction_defined_category_equal_macro": effects[
            "interaction"]["category_equal_macro_defined"],
        "q5_c1_full_vs_c2_a002_defined_category_equal_macro": macro[
            "auroc_by_arm"]["c1_full_minus_c2_a002"][
                "category_equal_macro_defined"],
        "factorial_evaluable_category_count": effects[
            "margin_effect_outcome_enabled"]["defined_category_count"],
        "factorial_evaluable_categories": effects[
            "margin_effect_outcome_enabled"]["defined_categories"],
        "factorial_undefined_categories": effects[
            "margin_effect_outcome_enabled"]["undefined_categories"],
    })
    activation_fractions = [
        by_category[category]["mechanism_diagnostics"][
            "ablation_beta_activation_vs_full"][arm][
                "effective_activation_fraction"]
        for category in CATEGORIES
        for arm in ("c1_no_outcome_weight", "c1_no_margin")]
    previous_verdict = aggregate.get("development_verdict")
    aggregate["development_verdict_before_coverage_adjudication"] = previous_verdict
    aggregate["development_verdict"] = _coverage_aware_development_verdict(
        aggregate["c1_full_reference_parity"]["status"], effects,
        activation_fractions)
    aggregate["coverage_aware_aggregation"] = {
        "expected_category_count": len(CATEGORIES),
        "evaluable_category_count": effects[
            "margin_effect_outcome_enabled"]["defined_category_count"],
        "evaluable_categories": effects[
            "margin_effect_outcome_enabled"]["defined_categories"],
        "undefined_categories": effects[
            "margin_effect_outcome_enabled"]["undefined_categories"],
        "strict_category_equal_macro_requires_all_categories": True,
        "defined_category_macro_is_descriptive_only": True,
        "source_sha256": sha256_file(Path(__file__)),
    }
    return aggregate


def arm_parameters(*, beta_scale: float = 0.5,
                   outcome_weights: Mapping[str, float] | None = None) -> dict[str, dict[str, Any]]:
    weights = dict(outcome_weights or {"FP": 1.30, "FN": 1.20,
                                       "TN": 0.60, "TP": 0.80})
    if set(weights) != {"FP", "FN", "TN", "TP"}:
        raise ValueError("All four C0 outcome weights are required")
    return {
        "c1_full": {"beta_scale": float(beta_scale), "outcome_weights": dict(weights)},
        "c1_no_outcome_weight": {
            "beta_scale": float(beta_scale),
            "outcome_weights": {key: 1.0 for key in weights}},
        "c1_no_margin": {
            "beta_scale": 0.0, "outcome_weights": dict(weights)},
        "c1_no_both": {
            "beta_scale": 0.0,
            "outcome_weights": {key: 1.0 for key in weights}},
    }


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _validate_parent_identity(expected_parent: str = EXPECTED_PARENT) -> dict[str, str]:
    branch = _git_output("branch", "--show-current")
    head = _git_output("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH or head.lower() != expected_parent.lower():
        raise Exp2AIntegrityError(
            f"Source-bound EXP2A identity mismatch: expected "
            f"{EXPECTED_BRANCH}@{expected_parent}, found {branch}@{head}")
    return {"branch": branch, "parent_commit": head}


def _validate_interpreter() -> None:
    actual = Path(sys.executable).resolve()
    if os.name == "nt" and actual != EXPECTED_PYTHON.resolve():
        raise Exp2AIntegrityError(
            f"EXP2A requires {EXPECTED_PYTHON}; found {actual}")


def _source_hashes(r1: Any) -> dict[str, str]:
    paths = (
        "test_phase3_AIF_simplified_v22.py",
        "vmb_visionad_new.py",
        "experiments/v02_memory.py",
        "experiments/p0_protocol_audit.py",
        "experiments/risk_safe/native_bridge.py",
        "experiments/calibration/native.py",
        "experiments/calibration/profiles.py",
        "experiments/calibration/stage05.py",
        "experiments/calibration/stage05_runner.py",
        "experiments/calibration/query_tape.py",
        "experiments/calibration/stage1a_p11_production.py",
        "experiments/calibration/stage1a_p12_label_provider.py",
        "experiments/calibration/r1_matched_correction_pilot.py",
        "experiments/calibration/r1_runtime_supervisor.py",
    )
    hashes = {name: sha256_file(ROOT / name) for name in paths}
    if hashes["experiments/calibration/r1_matched_correction_pilot.py"] != \
            r1.sha256_file(ROOT / "experiments/calibration/r1_matched_correction_pilot.py"):
        raise Exp2AIntegrityError("R1 pilot source fingerprint changed during preflight")
    return hashes


def _load_frozen_preflight(dataset_root: Path, json_root: Path,
                           r1: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import stage05
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.calibration.profiles import NativeScorerConfig, QueryProtocolConfig
    from experiments.calibration.stage1a_p11_production import (
        verify_stage05_qats_artifacts)

    _validate_parent_identity()
    _validate_interpreter()
    if not torch.cuda.is_available():
        raise Exp2AIntegrityError("Authorized CUDA device is unavailable")
    tapes = verify_stage05_qats_artifacts(STAGE05)
    if tapes.binding_sha256 != r1.QUERY_BINDING_SHA256:
        raise Exp2AIntegrityError("Frozen Q-A-TS binding differs from R1 authorization")
    frozen_path = STAGE05 / "frozen_inputs.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("test_labels_accessed") is not False or \
            frozen.get("test_masks_accessed") is not False:
        raise Exp2AIntegrityError("Frozen Stage 05 manifest reports label/mask access")
    score_manifest_path = STAGE05 / "score_tape_manifest.json"
    score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    scorer_profile = NativeScorerConfig()
    query_profile = QueryProtocolConfig()
    impl = p0.require_sources()
    runtime_profile = r1._configure_historical_runtime(impl, torch)
    if not runtime_profile["sdp"] == {
            "flash_enabled": False,
            "memory_efficient_enabled": False,
            "math_enabled": True}:
        raise Exp2AIntegrityError("Historical deterministic runtime profile differs")
    cfg = p0.Config(
        device="cuda", augmentation_seed=0, bank_seed=0, fixed_capacity=25000,
        fixed_per_image=5000, augmentations=4, include_identity=False,
        multiplier=2.5, feature_fuse="concat", topk_patches=32,
        image_aggregation="max", image_topk=64, image_quantile=0.95, knn=5)
    r1._validate_p0_profile(cfg, scorer_profile)

    categories: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        info = frozen["classes"].get(category)
        if not isinstance(info, Mapping):
            raise Exp2AIntegrityError(f"{category}: missing frozen category receipt")
        train, stream, class_sha = stage05.read_class_stream(
            dataset_root, json_root, category, test_labels=False)
        if class_sha != info.get("class_json_sha256"):
            raise Exp2AIntegrityError(f"{category}: current public metadata hash changed")
        if len(stream) < MAX_EVENTS or len(stream) != int(info["stream_images"]):
            raise Exp2AIntegrityError(f"{category}: stream does not cover frozen prefix")
        support_plan = stage05.select_supports(train)
        if (support_plan["supports"] != info["supports"] or
                support_plan["c64"] != info["c64"] or
                support_plan["normal_train_groups"] != info["train_normal_group_count"] or
                not info.get("support_c64_disjoint") or
                not info.get("train_test_groups_disjoint")):
            raise Exp2AIntegrityError(f"{category}: frozen Support/C64 selection changed")
        support_info = info["supports"]["0"]
        if len(support_info["images"]) != 4 or int(info["c64"]["count"]) != 64:
            raise Exp2AIntegrityError(f"{category}: expected 4 supports plus 64 C64 normals")
        if [row.relative_id for row in stream[:MAX_EVENTS]] != [
                row["relative_id"] for row in info["stream"][:MAX_EVENTS]]:
            raise Exp2AIntegrityError(f"{category}: current event order differs from frozen stream")
        support_groups = set(r1.opaque_sample_group(x)
                             for x in support_info["groups"])
        c64_groups = set(r1.opaque_sample_group(x)
                         for x in info["c64"]["groups"])
        prefix_rows = info["stream"][:MAX_EVENTS]
        prefix_groups = [str(row["opaque_group"]) for row in prefix_rows]
        if (support_groups & c64_groups or support_groups & set(prefix_groups) or
                c64_groups & set(prefix_groups)):
            raise Exp2AIntegrityError(f"{category}: support/C64/prefix group overlap")
        binding = next((row for row in tapes.bindings
                        if row.get("class_identity") == category), None)
        if not binding or binding.get("query_protocol_sha256") != r1.QUERY_PROTOCOL_SHA256:
            raise Exp2AIntegrityError(f"{category}: Q-A-TS binding missing/incompatible")
        positions = tuple(position for position in tapes.positions(category)
                          if 0 <= int(position) < MAX_EVENTS)
        query_groups = {
            int(position): str(tapes.group_identity_by_class_position[category][position])
            for position in positions}
        if len(set(query_groups.values())) != len(query_groups):
            raise Exp2AIntegrityError(f"{category}: repeated query source group")
        for position in positions:
            if query_groups[position] != str(prefix_rows[position]["opaque_group"]):
                raise Exp2AIntegrityError(f"{category}: query group differs at {position}")
        tape_row = score_manifest["classes"][category]["score_tapes"]["0"]
        c64_path = STAGE05 / tape_row["file"]
        if (int(tape_row["c64_score_count"]) != 64 or
                not tape_row["native_parity"]["passed"] or
                sha256_file(c64_path) != tape_row["sha256"]):
            raise Exp2AIntegrityError(f"{category}: C64 score receipt failed checksum/parity")
        with np.load(c64_path, allow_pickle=False) as archive:
            c64_scores = archive["c64_scores"].astype(np.float64, copy=True)
            c64_identity = str(archive["c64_identity_sha256"].item())
        if (len(c64_scores) != 64 or not np.isfinite(c64_scores).all() or
                c64_identity != info["c64"]["image_sha256"] or
                tape_row["c64_group_sha256"] != info["c64"]["group_sha256"]):
            raise Exp2AIntegrityError(f"{category}: C64 identity/score dimensions differ")
        from experiments.calibration.controllers import theta0_from_c64
        theta = float(theta0_from_c64(c64_scores.tolist()))
        sigma = max(float(c64_scores.std(ddof=1)), 0.02)
        support_paths = [Path(stage05._absolute_image_path(
            dataset_root, info["prefix"], str(relative)))
            for relative in support_info["images"]]
        event_paths = [Path(row.image_path) for row in stream[:MAX_EVENTS]]
        if any(not path.is_file() for path in (*support_paths, *event_paths)):
            raise Exp2AIntegrityError(f"{category}: authorized source image is missing")
        categories[category] = {
            "info": info, "stream": stream[:MAX_EVENTS],
            "support_paths": support_paths, "positions": positions,
            "query_groups": query_groups, "theta": theta, "sigma": sigma,
            "c64_score_sha256": str(tape_row["sha256"]),
            "class_json_sha256": class_sha,
            "support_group_sha256": r1.sha256_json(sorted(support_groups)),
        }
    return impl, {
        "categories": categories,
        "frozen_inputs_sha256": sha256_file(frozen_path),
        "frozen_checksums_sha256": sha256_file(STAGE05 / "checksums.json"),
        "qats_binding_sha256": tapes.binding_sha256,
        "qats_verification_sha256": tapes.verification_sha256,
        "scorer_profile": asdict(scorer_profile),
        "scorer_profile_sha256": scorer_profile.sha256,
        "query_profile_sha256": query_profile.sha256,
        "runtime_profile": runtime_profile,
        "p0_config_object": cfg,
        "p0_config": asdict(cfg),
        "fixed_bank_digest_algorithm": r1.STRUCTURED_FIXED_BANK_DIGEST_ALGORITHM,
        "raw_bank_digest_algorithm": r1.RAW_FIXED_BANK_DIGEST_ALGORITHM,
        "fixed_bank_digest_function": _fixed_bank_digest,
        "p0_module": p0,
        "stage05_module": stage05,
        "source_hashes": _source_hashes(r1),
    }


def _copy_numpy_tensor(value: Any) -> np.ndarray:
    return value.detach().to("cpu").contiguous().numpy().astype(np.float32, copy=True)


def _run_traced_update(call, callback, impl):
    prior = sys.gettrace()
    method = impl.PatchVMB.update_corrections_from_topk
    target = getattr(method, "__wrapped__", method).__code__

    def trace(frame, event, _arg):
        if frame.f_code is target and event in ("line", "return"):
            callback(frame.f_locals, event)
        return trace

    sys.settrace(trace)
    try:
        return call()
    finally:
        sys.settrace(prior)


def _make_arm_configs(bank: Any, impl: Any) -> dict[str, Any]:
    from dataclasses import asdict as dataclass_asdict
    base = bank.corr_cfg
    reference = bank.corr_cb["p0"]["id"][0]
    configs = {"c1_full": base}
    codebooks = {"c1_full": reference}
    overrides = arm_parameters(
        beta_scale=float(base.beta_scale),
        outcome_weights={
            "FP": float(base.outcome_weight_fp), "FN": float(base.outcome_weight_fn),
            "TN": float(base.outcome_weight_tn), "TP": float(base.outcome_weight_tp)})
    for arm in ARMS[1:]:
        cfg = type(base)(**dataclass_asdict(base))
        cfg.beta_scale = float(overrides[arm]["beta_scale"])
        cfg.outcome_weight_fp = float(overrides[arm]["outcome_weights"]["FP"])
        cfg.outcome_weight_fn = float(overrides[arm]["outcome_weights"]["FN"])
        cfg.outcome_weight_tn = float(overrides[arm]["outcome_weights"]["TN"])
        cfg.outcome_weight_tp = float(overrides[arm]["outcome_weights"]["TP"])
        cfg.enabled = False
        configs[arm] = cfg
        codebooks[arm] = impl.SignedCorrectionCodebook(
            int(reference.K), int(reference.protos.shape[1]), reference.device, cfg)
    return configs, codebooks


def _switch_codebook(bank: Any, arm: str, cfg: Any, cb: Any):
    prior_cfg = bank.corr_cfg
    prior_cb = bank.corr_cb["p0"]["id"][0]
    bank.corr_cfg = cfg
    bank.corr_cb["p0"]["id"][0] = cb
    cfg.enabled = True
    return prior_cfg, prior_cb


def _restore_codebook(bank: Any, cfg: Any, prior_cfg: Any, prior_cb: Any) -> None:
    cfg.enabled = False
    bank.corr_cfg = prior_cfg
    bank.corr_cb["p0"]["id"][0] = prior_cb


def _score_arm(impl: Any, bank: Any, arm: str, cfg: Any, cb: Any,
               source_groups: Mapping[int, set[str]], selected: Any,
               group_sha: str, position: int):
    from experiments.calibration.r1_matched_correction_pilot import _c1_score_delta
    prior_cfg, prior_cb = _switch_codebook(bank, arm, cfg, cb)
    try:
        return _c1_score_delta(
            impl, bank, selected, source_groups, group_sha,
            event_position=position)
    finally:
        _restore_codebook(bank, cfg, prior_cfg, prior_cb)


def _state_digest_for_arm(r1: Any, bank: Any, arm: str, cfg: Any,
                          cb: Any, source_groups: Mapping[int, set[str]]) -> str:
    prior_cfg, prior_cb = _switch_codebook(bank, arm, cfg, cb)
    try:
        return r1._correction_state_sha256(bank, source_groups)
    finally:
        _restore_codebook(bank, cfg, prior_cfg, prior_cb)


def _capture_update(impl: Any, bank: Any, r1: Any, arm: str, cfg: Any, cb: Any,
                    features: Sequence[Any], top_ids: np.ndarray, sign: int,
                    c0_score: float, theta: float, outcome: str, position: int,
                    source_groups: dict[int, set[str]], group_sha: str):
    prior_cfg, prior_cb = _switch_codebook(bank, arm, cfg, cb)
    seen: dict[str, Any] = {}

    def capture(locals_: Mapping[str, Any], event: str) -> None:
        for key in ("margin", "beta_abs_base", "beta_abs", "outcome_weight",
                    "dN_sel", "keep", "dN_keep", "patch_mult", "beta_t"):
            if key not in locals_:
                continue
            value = locals_[key]
            if hasattr(value, "detach"):
                value = value.detach().to("cpu").contiguous().clone()
            elif isinstance(value, (float, int, str, bool)):
                value = value
            else:
                continue
            seen[key] = value

    try:
        cfg.enabled = True
        result = _run_traced_update(
            lambda: bank.update_corrections_from_topk(
                cls_name="p0", feats_per_layer=list(features), mode=0, view="id",
                layer_to_patch_idx={0: top_ids.tolist()}, sign=sign,
                score_img=float(c0_score), theta=float(theta), strength_scale=1.0,
                outcome=outcome, step=position),
            capture, impl)
        layer = dict(result.get("layers", {}).get("0", {}))
        new_n = int(cb.size())
        r1._record_c1_source_group_updates(
            source_groups, last_steps=cb.last_step[:new_n].detach().to("cpu").tolist(),
            event_position=position, event_group=group_sha)
        def array(name, dtype=None):
            value = seen.get(name)
            if value is None:
                return None
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=dtype).reshape(-1).tolist()
        dN = array("dN_sel", np.float32)
        keep = array("keep", bool)
        dN_keep = array("dN_keep", np.float32)
        patch_mult = array("patch_mult", np.float32)
        beta_t = array("beta_t", np.float32)
        beta_float32 = (np.asarray(beta_t, dtype=np.float32)
                        if beta_t is not None else np.zeros(0, dtype=np.float32))
        beta_bank_clipped = np.clip(
            beta_float32, -float(cfg.beta_max), float(cfg.beta_max)).astype(np.float32)
        margin = abs(float(c0_score) - float(theta))
        raw_beta = float(cfg.beta_scale) * margin
        signs = np.full(beta_float32.shape, 1 if sign >= 0 else -1, dtype=np.int8)
        trace_required = all(name in seen for name in (
            "margin", "beta_abs_base", "beta_abs", "outcome_weight",
            "dN_sel", "keep"))
        snapshot = {
            "category_position": position,
            "arm": arm,
            "outcome": outcome,
            "c0_prediction": int(c0_score >= theta),
            "c0_score": float(c0_score),
            "threshold": float(theta),
            "margin": float(margin),
            "pre_clip_beta_scale_times_margin": float(raw_beta),
            "post_base_clip_beta": float(np.clip(raw_beta, cfg.beta_min, cfg.beta_max)),
            "post_outcome_clip_beta": float(np.clip(
                np.clip(raw_beta, cfg.beta_min, cfg.beta_max) *
                float(seen.get("outcome_weight", 1.0)),
                cfg.beta_min, cfg.beta_max)),
            "outcome_weight": float(seen.get("outcome_weight", 1.0)),
            "selected_dN_before_gate": dN,
            "native_gate_keep_mask": keep,
            "dN_after_gate": dN_keep,
            "patch_multiplier": patch_mult,
            "beta_patch_float32_before_codebook_clip": beta_t,
            "beta_after_original_codebook_max_clip": beta_bank_clipped.tolist(),
            "beta_sign": signs.tolist(),
            "native_update_stats": {
                key: int(value) for key, value in layer.items()
                if isinstance(value, (int, np.integer))},
            "anchor_occupancy_after": int(cb.size()),
            "positive_anchor_count_after": int((cb.beta[:new_n] > 0).sum().item()),
            "negative_anchor_count_after": int((cb.beta[:new_n] < 0).sum().item()),
            "codebook_state_sha256_after": r1._correction_state_sha256(
                bank, source_groups),
            "diagnostic_trace_complete": trace_required and (
                bool(np.any(keep)) == all(name in seen for name in
                                          ("dN_keep", "patch_mult", "beta_t"))),
        }
        if not snapshot["diagnostic_trace_complete"]:
            raise Exp2AIntegrityError(f"{arm}: original beta locals were not captured")
        return result, layer, snapshot
    finally:
        _restore_codebook(bank, cfg, prior_cfg, prior_cb)


def _output_budget_class(r1: Any):
    fixed = set(r1.OutputBudget._FIXED_FILES) | {
        "event_scores.jsonl", "event_transitions.jsonl",
        "private_query_feedback.jsonl", "strength_diagnostics.jsonl",
        "stage_a_seal.json", AGGREGATE_NAME, "failure.json",
        "supervisor_status.json", "stage_b_supervisor_status.json",
        "experiment_seal.json",
    }

    class Exp2AOutputBudget(r1.OutputBudget):
        _FIXED_FILES = fixed

        def _relative(self, path: Path) -> str:
            self._reject_reparse_ancestors()
            candidate = Path(os.path.abspath(path))
            try:
                relative = candidate.relative_to(self.root)
            except ValueError as exc:
                raise r1.OutputBudgetError(
                    "EXP2A writer attempted a path outside the authorized run") from exc
            current = self.root
            if current.exists() and self._is_reparse_point(current):
                raise r1.OutputBudgetError("EXP2A output root is a reparse point")
            for part in relative.parts[:-1]:
                current = current / part
                if current.exists() and self._is_reparse_point(current):
                    raise r1.OutputBudgetError("EXP2A output subdirectory is a reparse point")
            name = relative.as_posix()
            if (name not in self._FIXED_FILES and
                    not re.fullmatch(r"events/event_[0-9]{6}\.npz", name) and
                    not re.fullmatch(
                        r"(?:run_manifest|supervisor_status|stage_a_seal|"
                        r"sanitized_aggregate|failure)\.json\.[A-Fa-f0-9-]+\.tmp",
                        name)):
                raise r1.OutputBudgetError(
                    f"EXP2A filename is not in the audited allowlist: {name}")
            return name

    return Exp2AOutputBudget


def _create_output(root: Path, r1: Any):
    Budget = _output_budget_class(r1)
    budget = Budget(root, max_bytes=MAX_OUTPUT_BYTES,
                    failure_reserve_bytes=FAILURE_RESERVE_BYTES,
                    allowed_parent=ROOT / "outputs" / "calibration")
    budget.ensure_root()
    (root / "events").mkdir(exist_ok=False)
    if budget._is_reparse_point(root / "events"):
        budget.close()
        raise r1.OutputBudgetError("EXP2A event directory is a reparse point")
    return budget


def _append_record(budget: Any, path: Path, record: Mapping[str, Any],
                   *, failure_reserved: bool = False) -> None:
    payload = (canonical_json(dict(record)) + "\n").encode("utf-8")
    budget.append(path, payload, failure_reserved=failure_reserved)


def _write_json(budget: Any, path: Path, record: Mapping[str, Any],
                *, failure_reserved: bool = False) -> None:
    payload = (canonical_json(dict(record)) + "\n").encode("utf-8")
    budget.write_once(path, payload, failure_reserved=failure_reserved)


def _write_run_manifest(budget: Any, root: Path, manifest: Mapping[str, Any]) -> None:
    _write_json(budget, root / "run_manifest.json", manifest)


def _event_payload(arrays: Mapping[str, np.ndarray]) -> bytes:
    from experiments.calibration.r1_matched_correction_pilot import deterministic_npz_bytes
    return deterministic_npz_bytes({
        key: np.ascontiguousarray(value) for key, value in arrays.items()})


def _event_score_row(root: Path, budget: Any, event_number: int,
                     arrays: Mapping[str, np.ndarray],
                     row: Mapping[str, Any]) -> None:
    from experiments.calibration.r1_matched_correction_pilot import (
        arrays_sha256, sha256_bytes)
    shard_name = f"events/event_{event_number:06d}.npz"
    payload = _event_payload(arrays)
    budget.write_once(root / shard_name, payload)
    event_row = dict(row)
    event_row.update({
        "event_number": int(event_number),
        "shard_path": shard_name,
        "shard_sha256": sha256_bytes(payload),
        "arrays_sha256": arrays_sha256(arrays),
        "array_fields": sorted(arrays),
    })
    _append_record(budget, root / "event_scores.jsonl", event_row)


def _check_event_score_row(row: Mapping[str, Any], root: Path,
                           expected_fields: set[str]) -> dict[str, np.ndarray]:
    from experiments.calibration.r1_matched_correction_pilot import (
        arrays_sha256, corrected_dense_map, max_score, sha256_bytes,
        sha256_file)
    shard = root / str(row["shard_path"])
    if (not shard.is_file() or sha256_file(shard) != row.get("shard_sha256") or
            not str(shard.resolve()).startswith(str((root / "events").resolve()) +
                                                os.sep)):
        raise Exp2AIntegrityError("Stage A event shard identity/checksum is invalid")
    with np.load(shard, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if set(arrays) != expected_fields or arrays_sha256(arrays) != row.get("arrays_sha256"):
        raise Exp2AIntegrityError("Stage A event shard arrays differ from the sealed receipt")
    if float(row["scores"]["c0"]) != max_score(arrays["base_patch_map"]):
        raise Exp2AIntegrityError("C0 score does not reconstruct from the sealed native map")
    ids = arrays["top32_ids"]
    c1_names = {
        "c1_full": "c1_delta_full",
        "c1_no_outcome_weight": "c1_delta_no_outcome_weight",
        "c1_no_margin": "c1_delta_no_margin",
        "c1_no_both": "c1_delta_no_both",
    }
    for arm, name in c1_names.items():
        actual = max_score(corrected_dense_map(
            arrays["base_patch_map"], ids, arrays[name]))
        if float(row["scores"][arm]) != actual:
            raise Exp2AIntegrityError(f"{arm} score does not reconstruct from its delta")
    for arm, name in (("c2_a002", "c2_delta_a002"),
                      ("c2_a020", "c2_delta_a020")):
        actual = max_score(corrected_dense_map(
            arrays["base_patch_map"], ids, arrays[name]))
        if float(row["scores"][arm]) != actual:
            raise Exp2AIntegrityError(f"{arm} score does not reconstruct from its delta")
    return arrays


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise Exp2AIntegrityError(f"Required EXP2A artifact is missing: {path.name}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line]


def _write_failure(root: Path, r1: Any, exc: BaseException, *,
                   event_counts: Mapping[str, int] | None = None) -> None:
    if not root.exists():
        return
    Budget = _output_budget_class(r1)
    budget = Budget(root, max_bytes=MAX_OUTPUT_BYTES,
                    failure_reserve_bytes=FAILURE_RESERVE_BYTES,
                    allowed_parent=ROOT / "outputs" / "calibration",
                    allow_existing=True)
    try:
        path = root / "failure.json"
        if not path.exists():
            _write_json(budget, path, {
                "schema": "r2.exp2a.failure.v1",
                "status": "INCOMPLETE",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc)[:1600],
                "partial_category_event_counts": dict(event_counts or {}),
                "automatic_restart": False,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, failure_reserved=True)
    finally:
        budget.close()


def _write_stage_a_seal(root: Path, budget: Any, counts: Mapping[str, int],
                        metadata: Mapping[str, Any]) -> dict[str, Any]:
    from experiments.calibration.r1_matched_correction_pilot import sha256_file
    scores = root / "event_scores.jsonl"
    transitions = root / "event_transitions.jsonl"
    diagnostics = root / "strength_diagnostics.jsonl"
    feedback = root / "private_query_feedback.jsonl"
    seal = {
        "schema": "r2.exp2a.stage_a_seal.v1",
        "status": ("SEALED_COMPLETE" if all(
            int(counts.get(category, 0)) == MAX_EVENTS for category in CATEGORIES)
                   else "SEALED_INCOMPLETE"),
        "category_event_counts": dict(counts),
        "total_events": int(sum(counts.values())),
        "score_rows_sha256": sha256_file(scores),
        "transition_rows_sha256": sha256_file(transitions),
        "strength_diagnostics_sha256": (
            sha256_file(diagnostics) if diagnostics.is_file() else None),
        "queried_feedback_sha256": (
            sha256_file(feedback) if feedback.is_file() else None),
        "event_shard_count": len(list((root / "events").glob("event_*.npz"))),
        "metadata": dict(metadata),
        "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(budget, root / "stage_a_seal.json", seal)
    return seal


def _capture_category(category: str, category_info: Mapping[str, Any],
                      shared: Mapping[str, Any], *, dataset_root: Path,
                      json_root: Path, root: Path, budget: Any,
                      backbone: Any, authorization_id: str,
                      run_id: str, event_counts: dict[str, int]) -> dict[str, Any]:
    import torch
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import stage05
    from experiments.calibration.r1_matched_correction_pilot import (
        C2_ANCHOR_CAPACITY, CausalEventOrder, OutputBudgetError,
        _c1_score_delta, _correction_state_sha256, _memory_content_sha256,
        _memory_state_sha256, _native_dense_score, _record_c1_source_group_updates,
        _validate_c1_source_groups, _validate_c2_anchor_state,
        _process_rss_bytes, common_c0_greedy_writer_action, signed_knn_delta,
        source_disjoint_anchor_indices, )
    from experiments.calibration.stage1a_p12_label_provider import (
        QueriedPositionLabelReader)
    from experiments.calibration.r1_matched_correction_pilot import (
        OrderedQueriedLabelReader, )
    from v02_memory import MemoryArm
    from types import SimpleNamespace

    impl = shared["impl"]
    cfg = shared["p0_config_object"]
    info = category_info["info"]
    stream = category_info["stream"]
    theta = float(category_info["theta"])
    sigma = float(category_info["sigma"])
    support_paths = category_info["support_paths"]
    support_features = p0.encode_reference_features(
        impl, backbone, [str(path) for path in support_paths], cfg)
    bank = p0.build_vmb(impl, support_features,
                        [str(path) for path in support_paths], cfg)
    from experiments.calibration.native import _fixed_bank_digest
    expected_fixed = str(
        json.loads((STAGE05 / "score_tape_manifest.json").read_text(
            encoding="utf-8"))["classes"][category]["score_tapes"]["0"]
        ["fixed_support_sha256"])
    fixed_digest = _fixed_bank_digest(bank)
    if fixed_digest != expected_fixed:
        raise Exp2AIntegrityError(
            f"{category}: actual native fixed bank differs from Stage 05 receipt")
    raw_fixed_digest = p0.fixed_bank_profile(bank)["sha256"]
    settings = SimpleNamespace(
        flat=256, stm=256, ltm=1, buffer=1, min_events=2,
        promote_every=16, cap_bytes=MAX_OUTPUT_BYTES)
    writer = MemoryArm("greedy_coreset", bank, settings, seed=0)
    if any(int(cb.size()) for cb in __import__(
            "experiments.risk_safe.native_bridge", fromlist=["dynamic_codebooks"]
    ).dynamic_codebooks(bank)):
        raise Exp2AIntegrityError(f"{category}: dynamic C0 memory did not start empty")
    if any(int(cb.K) != 256 for cb in __import__(
            "experiments.risk_safe.native_bridge", fromlist=["dynamic_codebooks"]
    ).dynamic_codebooks(bank)):
        raise Exp2AIntegrityError(f"{category}: C0 Greedy capacity changed")
    bank.corr_cfg.enabled = False
    bank.use_defect_bank = False
    configs, codebooks = _make_arm_configs(bank, impl)
    for arm in ARMS:
        if int(codebooks[arm].size()) != 0 or int(codebooks[arm].K) != 256:
            raise Exp2AIntegrityError(f"{category}:{arm} invalid initial Codebook")
    c1_source_groups: dict[str, dict[int, set[str]]] = {
        arm: {} for arm in ARMS}
    c2_anchors: list[dict[str, Any]] = []
    c0_memory_version = 0
    c0_memory_content = _memory_content_sha256(bank)
    c0_memory_state = _memory_state_sha256(bank, c0_memory_version)
    c2_updated_count = 0
    query_positions = tuple(int(x) for x in category_info["positions"])
    label_reader = None
    ordered_reader = None
    if query_positions:
        label_reader = QueriedPositionLabelReader(
            category=category,
            path=json_root / f"{category}.json",
            expected_source_sha256=str(category_info["class_json_sha256"]),
            positions=query_positions,
            group_sha256_by_position={
                int(position): str(category_info["query_groups"][position])
                for position in query_positions})
        ordered_reader = OrderedQueriedLabelReader(label_reader, query_positions)
    category_start = time.monotonic()
    category_enc_seconds = category_native_seconds = 0.0
    category_writer_seconds = category_update_seconds = 0.0
    queried_count = 0
    score_path = root / "event_scores.jsonl"
    transition_path = root / "event_transitions.jsonl"
    diag_path = root / "strength_diagnostics.jsonl"
    feedback_path = root / "private_query_feedback.jsonl"
    category_index = CATEGORIES.index(category)
    try:
        for position, meta in enumerate(stream):
            if time.monotonic() - shared["pilot_start_monotonic"] >= MAX_RUNTIME_SECONDS:
                raise Exp2AError("120-minute Job Object wall-clock budget reached")
            group_sha = str(meta.opaque_group)
            image_path = Path(meta.image_path)
            image_sha = sha256_file(image_path)
            encode_start = time.perf_counter()
            image, features = p0.encode_query(impl, backbone, str(image_path), cfg)
            torch.cuda.synchronize()
            category_enc_seconds += time.perf_counter() - encode_start
            if (len(features) != 1 or tuple(features[0].shape) != (784, 4096) or
                    features[0].dtype != torch.float32 or
                    not torch.isfinite(features[0]).all().item()):
                raise Exp2AIntegrityError(
                    f"{category}:{position} native feature shape/dtype/finiteness mismatch")
            memory_before_score = _memory_content_sha256(bank)
            native_start = time.perf_counter()
            base_map, top_ids, c0_score, summary = _native_dense_score(
                impl, bank, image, features, cfg)
            torch.cuda.synchronize()
            category_native_seconds += time.perf_counter() - native_start
            if _memory_content_sha256(bank) != memory_before_score:
                raise Exp2AIntegrityError(f"{category}:{position} C0 score mutated memory")
            if (top_ids.shape != (32,) or len(set(top_ids.tolist())) != 32 or
                    not np.array_equal(top_ids, np.asarray(
                        summary["layer0"]["topk_patch_idx"], dtype=np.int64))):
                raise Exp2AIntegrityError(f"{category}:{position} native Top-32 differs")
            normalized = impl.l2_normalize(features[0].to(bank.device))
            selected = normalized[torch.as_tensor(
                top_ids, device=bank.device, dtype=torch.long)]
            arm_deltas: dict[str, np.ndarray] = {}
            arm_diag: dict[str, dict[str, Any]] = {}
            arm_scores: dict[str, float] = {}
            correction_started = time.perf_counter()
            for arm in ARMS:
                delta, detail = _score_arm(
                    impl, bank, arm, configs[arm], codebooks[arm],
                    c1_source_groups[arm], selected, group_sha, position)
                arm_deltas[arm] = np.asarray(delta, dtype=np.float32)
                arm_diag[arm] = dict(detail)
                from experiments.calibration.r1_matched_correction_pilot import (
                    corrected_dense_map, max_score)
                arm_scores[arm] = max_score(corrected_dense_map(
                    base_map, top_ids, arm_deltas[arm]))
            c2_02, eligible_02 = signed_knn_delta(
                selected.detach().float().to("cpu").numpy(), c2_anchors,
                event_position=position, event_group=group_sha,
                neighbors=5, amplitude=0.02)
            c2_20, eligible_20 = signed_knn_delta(
                selected.detach().float().to("cpu").numpy(), c2_anchors,
                event_position=position, event_group=group_sha,
                neighbors=5, amplitude=0.20)
            expected_c2 = source_disjoint_anchor_indices(
                [int(item["position"]) for item in c2_anchors],
                [str(item["group_sha256"]) for item in c2_anchors],
                event_position=position, event_group=group_sha)
            if eligible_02 != len(expected_c2) or eligible_20 != len(expected_c2):
                raise Exp2AIntegrityError(f"{category}:{position} C2 exclusion failed")
            category_native_seconds += time.perf_counter() - correction_started
            from experiments.calibration.r1_matched_correction_pilot import (
                corrected_dense_map, max_score, arrays_sha256)
            score_map = {
                "c0": float(c0_score), **{arm: float(arm_scores[arm]) for arm in ARMS},
                "c2_a002": float(max_score(corrected_dense_map(
                    base_map, top_ids, c2_02))),
                "c2_a020": float(max_score(corrected_dense_map(
                    base_map, top_ids, c2_20))),
            }
            if not all(math.isfinite(value) for value in score_map.values()):
                raise Exp2AIntegrityError(f"{category}:{position} nonfinite arm score")
            predictions = {key: int(value >= theta) for key, value in score_map.items()}
            state_before = {
                arm: _state_digest_for_arm(
                    __import__("experiments.calibration.r1_matched_correction_pilot",
                               fromlist=["_correction_state_sha256"]),
                    bank, arm, configs[arm], codebooks[arm], c1_source_groups[arm])
                for arm in ARMS}
            event_id = sha256_json({
                "experiment": "R2-EXP2A", "category": category,
                "position": position, "image_sha256": image_sha,
                "group_sha256": group_sha})
            feature_digest = hashlib.sha256(
                features[0].detach().to("cpu").contiguous().numpy().tobytes()).hexdigest()
            selected_features = selected.detach().float().to("cpu").contiguous().numpy().copy()
            arrays = {
                "base_patch_map": np.asarray(base_map, dtype=np.float32),
                "top32_ids": np.asarray(top_ids, dtype=np.int64),
                "top32_features": selected_features,
                "c1_delta_full": arm_deltas["c1_full"],
                "c1_delta_no_outcome_weight": arm_deltas["c1_no_outcome_weight"],
                "c1_delta_no_margin": arm_deltas["c1_no_margin"],
                "c1_delta_no_both": arm_deltas["c1_no_both"],
                "c2_delta_a002": np.asarray(c2_02, dtype=np.float32),
                "c2_delta_a020": np.asarray(c2_20, dtype=np.float32),
            }
            if selected_features.shape != (32, 4096):
                raise Exp2AIntegrityError(f"{category}:{position} Top-32 feature shape changed")
            row = {
                "schema": "r2.exp2a.score_row.v1",
                "category": category, "position": position, "event_id": event_id,
                "source_identity_sha256": hashlib.sha256(
                    str(meta.relative_id).encode("utf-8")).hexdigest(),
                "class_json_sha256": category_info["class_json_sha256"],
                "source_image_sha256": image_sha,
                "source_group_sha256": group_sha,
                "timestamp": str(meta.timestamp), "camera_id": int(meta.camera_id),
                "feature_tensor_sha256": feature_digest,
                "feature_shape": [784, 4096], "feature_dtype": "torch.float32",
                "top32_indices_sha256": sha256_json(top_ids.tolist()),
                "fixed_support_sha256": fixed_digest,
                "fixed_support_raw_sha256": raw_fixed_digest,
                "c0_memory_version_before": c0_memory_version,
                "c0_memory_sha256_before": c0_memory_state,
                "c1_state_sha256_before": state_before,
                "scores": score_map, "predictions_at_frozen_c64_theta": predictions,
                "theta_c64_q90": theta, "sigma_c64": sigma,
                "query_tape_decision": position in query_positions,
                "c1_source_exclusion": arm_diag,
                "c2_eligible_anchor_count": int(eligible_02),
                "c2_source_exclusion_checks": 1,
            }
            event_number = category_index * MAX_EVENTS + position
            _event_score_row(root, budget, event_number, arrays, row)
            # All seven scores and the private feature shard are durable before
            # the query decision or any label access for this event.
            gate = CausalEventOrder(position)
            gate.seal_scores()
            queried = gate.decide_query(position in query_positions)
            label = None
            if queried:
                if ordered_reader is None:
                    raise Exp2AIntegrityError("Query Tape asks for a label without an authorized reader")
                label = gate.disclose(lambda: ordered_reader(
                    category, position, str(meta.sample_key)))
                queried_count += 1
                outcome_by_label = {
                    (0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}
                c0_prediction = predictions["c0"]
                outcome = outcome_by_label[(int(label), c0_prediction)]
                sign = 1 if int(label) == 1 else -1
                _append_record(budget, feedback_path, {
                    "schema": "r2.exp2a.queried_feedback.v1",
                    "category": category, "position": position, "event_id": event_id,
                    "source_identity_sha256": row["source_identity_sha256"],
                    "source_group_sha256": group_sha,
                    "disclosure_after_score_seal": True,
                    "label": int(label), "outcome_from_common_c0": outcome,
                })
                arm_updates = {}
                for arm in ARMS:
                    update_start = time.perf_counter()
                    result, layer, diag = _capture_update(
                        impl, bank,
                        __import__("experiments.calibration.r1_matched_correction_pilot",
                                   fromlist=["_correction_state_sha256"]),
                        arm, configs[arm], codebooks[arm], features, top_ids,
                        sign, c0_score, theta, outcome, position,
                        c1_source_groups[arm], group_sha)
                    category_update_seconds += time.perf_counter() - update_start
                    diag["category"] = category
                    diag["event_id"] = event_id
                    arm_updates[arm] = {
                        "native_layer_stats": {
                            key: int(value) for key, value in layer.items()
                            if isinstance(value, (int, np.integer))},
                        "diagnostic": diag,
                    }
                c2_added = 0
                for patch_row, patch_id in enumerate(top_ids.tolist()):
                    feature_row = selected_features[patch_row]
                    feature_sha = arrays_sha256({"feature": feature_row})
                    c2_anchors.append({
                        "position": position, "group_sha256": group_sha,
                        "patch_id": int(patch_id), "sign": sign,
                        "feature": feature_row.copy(), "feature_sha256": feature_sha})
                    c2_added += 1
                if len(c2_anchors) > C2_ANCHOR_CAPACITY:
                    del c2_anchors[:len(c2_anchors) - C2_ANCHOR_CAPACITY]
                c2_updated_count += c2_added
                _append_record(budget, diag_path, {
                    "schema": "r2.exp2a.strength_diagnostic.v1",
                    "category": category, "position": position,
                    "source_group_sha256": group_sha,
                    "c0_prediction": c0_prediction, "label": int(label),
                    "outcome": outcome, "sign": sign,
                    "arms": arm_updates,
                    "c2_added_patch_rows": c2_added,
                })
            gate.corrections_updated()
            r1_pilot = __import__(
                "experiments.calibration.r1_matched_correction_pilot",
                fromlist=["common_c0_greedy_writer_action"])
            writer_start = time.perf_counter()
            memory_before = _memory_content_sha256(bank)

            def writer_update(*, position: int, feature_layers: Sequence[Any],
                              candidate_ids: np.ndarray, score: float,
                              theta: float, sigma: float) -> Mapping[str, Any]:
                before = writer.audit()
                writer.update(impl, list(feature_layers),
                              {0: candidate_ids.tolist()}, position,
                              score=score, theta=theta, sigma=sigma)
                after = writer.audit()
                return {
                    "accepted_patches_delta": int(
                        after["accepted_patches"] - before["accepted_patches"]),
                    "skipped_patches_delta": int(
                        after["skipped_patches"] - before["skipped_patches"]),
                }
            action = common_c0_greedy_writer_action(
                position=position, c0_score=float(c0_score), theta=theta, sigma=sigma,
                c0_prediction=predictions["c0"], candidate_ids=top_ids,
                candidate_features=selected_features, feature_layers=features,
                queried=queried, disclosed_label=label,
                eligibility=impl.select_informative_normal_write,
                update=writer_update)
            category_writer_seconds += time.perf_counter() - writer_start
            memory_after = _memory_content_sha256(bank)
            if memory_after != memory_before:
                c0_memory_version += 1
                c0_memory_content = memory_after
                c0_memory_state = _memory_state_sha256(bank, c0_memory_version)
            gate.memory_updated()
            state_after = {
                arm: _state_digest_for_arm(
                    r1_pilot, bank, arm, configs[arm], codebooks[arm],
                    c1_source_groups[arm])
                for arm in ARMS}
            transition = {
                "schema": "r2.exp2a.transition.v1",
                "category": category, "position": position, "event_id": event_id,
                "query_decision": bool(queried),
                "queried_label_disclosed_after_scores": bool(queried),
                "common_c0_writer_action": action,
                "c0_memory_version_after": c0_memory_version,
                "c0_memory_sha256_after": c0_memory_state,
                "c0_memory_content_sha256_after": c0_memory_content,
                "c1_state_sha256_after": state_after,
                "c1_anchor_occupancy_after": {
                    arm: int(codebooks[arm].size()) for arm in ARMS},
                "c1_anchor_sign_counts_after": {
                    arm: {
                        "positive": int((codebooks[arm].beta[
                            :codebooks[arm].size()] > 0).sum().item()),
                        "negative": int((codebooks[arm].beta[
                            :codebooks[arm].size()] < 0).sum().item()),
                    } for arm in ARMS},
                "c1_outcome_group_counts": {
                    arm: {
                        "fp": int(codebooks[arm].outcome_fp[:codebooks[arm].size()].sum().item()),
                        "fn": int(codebooks[arm].outcome_fn[:codebooks[arm].size()].sum().item()),
                        "tn": int(codebooks[arm].outcome_tn[:codebooks[arm].size()].sum().item()),
                        "tp": int(codebooks[arm].outcome_tp[:codebooks[arm].size()].sum().item()),
                    } for arm in ARMS},
                "c2_anchor_count_after": len(c2_anchors),
                "causal_order_complete": True,
            }
            _append_record(budget, transition_path, transition)
            gate.finish()
            if queried and label_reader.decoded_positions[-1] != position:
                raise Exp2AIntegrityError("Queried label reader decoded the wrong event")
            event_counts[category] = position + 1
            if (position + 1) % 64 == 0:
                print(f"R2_EXP2A_PROGRESS category={category} events={position + 1}/512 "
                      f"queries={queried_count}", flush=True)
        if ordered_reader is not None and \
                tuple(label_reader.decoded_positions) != query_positions:
            raise Exp2AIntegrityError(f"{category}: query feedback did not match frozen tape")
        if _fixed_bank_digest(bank) != fixed_digest:
            raise Exp2AIntegrityError(f"{category}: common writer mutated fixed supports")
        return {
            "category": category, "events": len(stream),
            "query_events": queried_count,
            "query_positions_sha256": sha256_json(list(query_positions)),
            "fixed_support_sha256": fixed_digest,
            "fixed_support_raw_prototype_sha256": raw_fixed_digest,
            "c64_score_tape_sha256": category_info["c64_score_sha256"],
            "c0_memory_version_final": c0_memory_version,
            "c0_memory_state_sha256_final": c0_memory_state,
            "c0_memory_content_sha256_final": c0_memory_content,
            "c1_anchor_occupancy_final": {
                arm: int(codebooks[arm].size()) for arm in ARMS},
            "c2_anchor_count_final": len(c2_anchors),
            "shared_support_images_encoded": len(support_paths),
            "shared_query_images_encoded": len(stream),
            "shared_support_feature_views_encoded": (
                len(support_paths) * int(cfg.augmentations)),
            "shared_dino_feature_encodings": (
                len(support_paths) * int(cfg.augmentations) + len(stream)),
            "encoding_seconds": category_enc_seconds,
            "native_scoring_seconds": category_native_seconds,
            "common_writer_seconds": category_writer_seconds,
            "online_correction_update_seconds": category_update_seconds,
            "wall_seconds": time.monotonic() - category_start,
            "peak_cpu_rss_bytes": _process_rss_bytes(),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    finally:
        if label_reader is not None:
            label_reader.close()


def _verify_switch_reference(root: Path) -> dict[str, Any]:
    """Strictly compare the new Full arm with the historical 512-event R1 tape."""
    from experiments.calibration.r1_matched_correction_pilot import (
        arrays_sha256, max_score, corrected_dense_map, sha256_file)
    if not R1_SWITCH.is_dir():
        raise Exp2AIntegrityError("Historical switch R1-S4B evidence is unavailable")
    old_scores = _read_jsonl(R1_SWITCH / "sealed_scores.jsonl")
    old_transitions = _read_jsonl(R1_SWITCH / "event_transitions.jsonl")
    old_feedback = _read_jsonl(R1_SWITCH / "correction_anchor_events.jsonl")
    new_scores = [row for row in _read_jsonl(root / "event_scores.jsonl")
                  if row["category"] == "switch"]
    new_transitions = [row for row in _read_jsonl(root / "event_transitions.jsonl")
                       if row["category"] == "switch"]
    new_feedback = {
        int(row["position"]): row for row in _read_jsonl(
            root / "private_query_feedback.jsonl")
        if row["category"] == "switch"
    } if (root / "private_query_feedback.jsonl").is_file() else {}
    new_diagnostics = {
        int(row["position"]): row for row in _read_jsonl(
            root / "strength_diagnostics.jsonl")
        if row["category"] == "switch"
    } if (root / "strength_diagnostics.jsonl").is_file() else {}
    old_by_pos = {int(row["position"]): row for row in old_scores}
    old_trans_by_pos = {int(row["position"]): row for row in old_transitions}
    old_feedback_by_pos = {int(row["position"]): row for row in old_feedback}
    if (len(old_by_pos) != MAX_EVENTS or len(old_trans_by_pos) != MAX_EVENTS or
            len(new_scores) != MAX_EVENTS or len(new_transitions) != MAX_EVENTS):
        raise Exp2AIntegrityError("Historical/new switch tape does not contain 512 events")
    checked = {"events": 0, "query_events": 0, "c0_maps_equal": 0,
               "top32_features_equal": 0,
               "c1_full_deltas_equal": 0, "c1_full_scores_equal": 0,
               "c1_states_equal_before_after": 0,
               "c0_writer_states_equal": 0, "c2_a020_deltas_equal": 0,
               "query_labels_equal": 0, "update_stats_equal": 0}
    expected_fields = {
        "base_patch_map", "top32_ids", "top32_features",
        "c1_delta_full", "c1_delta_no_outcome_weight",
        "c1_delta_no_margin", "c1_delta_no_both",
        "c2_delta_a002", "c2_delta_a020",
    }
    for current in new_scores:
        position = int(current["position"])
        prior = old_by_pos[position]
        old_transition = old_trans_by_pos[position]
        current_transition = new_transitions[position]
        current_arrays = _check_event_score_row(current, root, expected_fields)
        old_shard = R1_SWITCH / str(prior["shard_path"])
        if not old_shard.is_file() or sha256_file(old_shard) != prior["shard_sha256"]:
            raise Exp2AIntegrityError(f"Historical switch event {position} checksum failed")
        with np.load(old_shard, allow_pickle=False) as archive:
            old_arrays = {name: archive[name].copy() for name in archive.files}
        if arrays_sha256(old_arrays) != prior["array_sha256"]:
            raise Exp2AIntegrityError(f"Historical switch event {position} array hash failed")
        if not np.array_equal(current_arrays["base_patch_map"], old_arrays["base_patch_map"]):
            raise Exp2AIntegrityError(f"Switch C0 dense map differs at {position}")
        if not np.array_equal(current_arrays["top32_ids"], old_arrays["top32_ids"]):
            raise Exp2AIntegrityError(f"Switch native Top-32 differs at {position}")
        if not np.array_equal(current_arrays["top32_features"],
                              old_arrays["top32_features"]):
            raise Exp2AIntegrityError(f"Switch Top-32 features differ at {position}")
        if (current.get("fixed_support_sha256") !=
                prior.get("fixed_support_sha256") or
                current.get("source_group_sha256") !=
                prior.get("official_group_sha256")):
            raise Exp2AIntegrityError(
                f"Switch fixed support or source-group identity differs at {position}")
        if not np.array_equal(current_arrays["c1_delta_full"], old_arrays["c1_delta"]):
            raise Exp2AIntegrityError(f"Switch C1-Full patch delta differs at {position}")
        if not np.array_equal(current_arrays["c2_delta_a020"], old_arrays["c2_delta"]):
            raise Exp2AIntegrityError(f"Switch C2(0.20) patch delta differs at {position}")
        checked["events"] += 1
        checked["c0_maps_equal"] += 1
        checked["top32_features_equal"] += 1
        checked["c1_full_deltas_equal"] += 1
        checked["c2_a020_deltas_equal"] += 1
        if float(current["scores"]["c1_full"]) != float(prior["c1_score"]):
            tolerance = 1e-8 + 1e-7 * abs(float(prior["c1_score"]))
            if abs(float(current["scores"]["c1_full"]) -
                   float(prior["c1_score"])) > tolerance:
                raise Exp2AIntegrityError(
                    f"Switch C1-Full image score differs beyond same-GPU tolerance at {position}")
        checked["c1_full_scores_equal"] += 1
        if (current["c1_state_sha256_before"]["c1_full"] !=
                prior["c1_state_sha256_before"] or
                current_transition["c1_state_sha256_after"]["c1_full"] !=
                old_transition["c1_state_sha256_after"]):
            raise Exp2AIntegrityError(f"Switch C1-Full Codebook state differs at {position}")
        checked["c1_states_equal_before_after"] += 1
        if (current["c0_memory_sha256_before"] !=
                prior["greedy_memory_sha256_before"] or
                current["predictions_at_frozen_c64_theta"]["c0"] !=
                prior["c0_prediction"] or
                current_transition["c0_memory_sha256_after"] !=
                old_transition["greedy_memory_sha256_after"] or
                bool(current_transition["query_decision"]) !=
                bool(old_transition["query_decision"])):
            raise Exp2AIntegrityError(f"Switch C0/Query trajectory differs at {position}")
        checked["c0_writer_states_equal"] += 1
        queried = bool(current_transition["query_decision"])
        if queried:
            checked["query_events"] += 1
            feedback = new_feedback.get(position)
            old_anchor = old_feedback_by_pos.get(position)
            diag = new_diagnostics.get(position)
            if feedback is None or old_anchor is None or diag is None:
                raise Exp2AIntegrityError(f"Switch query evidence is missing at {position}")
            old_label = 1 if old_anchor["outcome"] in ("FN", "TP") else 0
            if (int(feedback["label"]) != old_label or
                    feedback["source_group_sha256"] !=
                    old_anchor["source_group_sha256"]):
                raise Exp2AIntegrityError(f"Switch queried feedback differs at {position}")
            checked["query_labels_equal"] += 1
            if (diag["arms"]["c1_full"]["native_layer_stats"] !=
                    old_anchor["c1_stats"]):
                raise Exp2AIntegrityError(f"Switch C1-Full update stats differ at {position}")
            checked["update_stats_equal"] += 1
    if checked["query_events"] != len(old_feedback_by_pos):
        raise Exp2AIntegrityError("Switch query event count differs from historical R1")
    return {
        "status": "PASS",
        "historical_r1_output_manifest_sha256": sha256_file(
            R1_SWITCH / "run_manifest.json"),
        "event_checks": checked,
        "score_tolerance": "exact dense maps/deltas and state hashes; scalar C1 score <= 1e-8 + 1e-7*|reference|",
        "tolerance_basis": "same RTX CUDA/runtime/source; tolerance only covers scalar reduction serialization",
        "historical_auroc_reference": 0.843083,
        "auroc_recomputed_after_stage_b": False,
    }


def _stage_a_worker(*, run_id: str, authorization_id: str,
                    output: Path, dataset_root: Path, json_root: Path) -> dict[str, Any]:
    if authorization_id != AUTHORIZATION_ID:
        raise Exp2AIntegrityError("EXP2A authorization identifier differs from task record")
    if (os.environ.get("R2_EXP2A_SUPERVISED_CHILD") != "1" or
            os.environ.get("R2_EXP2A_AUTHORIZATION_ID") != authorization_id or
            not re.fullmatch(r"[0-9a-f]{64}",
                             os.environ.get("R2_EXP2A_SUPERVISOR_NONCE", ""))):
        raise Exp2AIntegrityError("Real feature capture requires explicit supervisor launch")
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job
    if not current_process_is_in_job():
        raise Exp2AIntegrityError("Native capture child is not in a Windows Job Object")
    identity = _validate_parent_identity()
    _validate_interpreter()
    free_bytes = r1._require_minimum_output_free_space(output.parent)
    if free_bytes < MIN_FREE_BYTES:
        raise Exp2AIntegrityError("Output volume has less than the required 10 GiB free")
    if output.exists():
        raise FileExistsError(f"EXP2A output namespace exists; resume is forbidden: {output}")
    import torch
    impl, preflight = _load_frozen_preflight(dataset_root, json_root, r1)
    Budget = _output_budget_class(r1)
    budget = _create_output(output, r1)
    counts = {category: 0 for category in CATEGORIES}
    started = time.monotonic()
    category_runtime: dict[str, Any] = {}
    try:
        torch_hub_dir = Path(torch.hub.get_dir())
        hub_path, hub_source_sha = r1._hub_source_sha256(torch_hub_dir)
        checkpoint = torch_hub_dir / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
        if not checkpoint.is_file():
            raise Exp2AIntegrityError(
                "Cached DINOv2 checkpoint missing; network download is disabled")
        checkpoint_sha = sha256_file(checkpoint)
        source_sha = preflight["source_hashes"]
        manifest = {
            "schema": "r2.exp2a.run.v1", "task": "R2-EXP2A",
            "status": "INCOMPLETE", "phase": "STAGE_A",
            "run_id": run_id, "authorization_id": authorization_id,
            "experiment_profile_id": PROFILE_ID,
            "identity": identity,
            "python_executable": str(Path(sys.executable).resolve()),
            "python_executable_sha256": hashlib.sha256(
                str(Path(sys.executable).resolve()).encode()).hexdigest(),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "cuda_device": torch.cuda.get_device_name(0),
            "runtime_profile": preflight["runtime_profile"],
            "source_hashes": source_sha,
            "r2_exp2a_source_sha256": sha256_file(Path(__file__)),
            "stage05_frozen_inputs_sha256": preflight["frozen_inputs_sha256"],
            "stage05_checksums_sha256": preflight["frozen_checksums_sha256"],
            "qats_binding_sha256": preflight["qats_binding_sha256"],
            "qats_verification_sha256": preflight["qats_verification_sha256"],
            "scorer_profile_sha256": preflight["scorer_profile_sha256"],
            "query_protocol": {"name": "Q-A-TS-v1",
                               "rate": QUERY_RATE, "seed": QUERY_SEED},
            "authorized_categories": list(CATEGORIES),
            "authorized_event_horizon": [0, MAX_EVENTS - 1],
            "support_regime": "4 deployed seed-0 supports plus 64 additional C64 training normals",
            "c64_rescored": False,
            "c64_threshold_source": "verified frozen Stage 05 native C64 score tape",
            "fixed_support_digest_algorithm": preflight["fixed_bank_digest_algorithm"],
            "raw_support_digest_algorithm": preflight["raw_bank_digest_algorithm"],
            "factors": {
                "index_1": "margin adaptation",
                "index_2": "FP/FN/TN/TP strength weighting",
                "parameters": arm_parameters(),
            },
            "controls": ["C0", "C2(a=0.02)", "C2(a=0.20, historical engineering reference)"],
            "common_trajectory_contract": {
                "one feature extraction per event shared by all arms": True,
                "C0 map/query/prediction/common Greedy writer shared": True,
                "C1 Codebook state independent by arm": True,
                "C0 prediction defines all arm outcomes": True,
                "scores sealed before queried label disclosure": True,
                "unqueried labels accessed": False,
                "masks accessed": False,
                "development outcome evaluation": "Stage B only after full score seal",
            },
            "feature_reuse_accounting": {
                "top32_4096d_feature_rows_persisted_once_per_event": True,
                "per_arm_dino_encodings": 0,
                "support_images_encoded_per_category": 4,
                "support_feature_views_per_category": 16,
                "query_images_encoded_per_category": MAX_EVENTS,
                "total_dino_feature_encodings_per_category": 528,
                "c64_images_encoded_or_rescored": 0,
            },
            "resource_bound": {
                "runtime_seconds": MAX_RUNTIME_SECONDS,
                "runtime_profile_id": RUNTIME_PROFILE_ID,
                "supervisor": (
                    "Windows Job Object; kill-on-close; no automatic restart; "
                    "180-minute task-specific ceiling authorized by researcher; "
                    "historical R1 supervisor source unchanged"),
                "output_max_bytes": MAX_OUTPUT_BYTES,
                "failure_reserve_bytes": FAILURE_RESERVE_BYTES,
                "minimum_free_bytes": MIN_FREE_BYTES,
                "free_bytes_before_model_initialization": free_bytes,
                "os_hard_storage_isolation": False,
            },
            "dino_hub_source_name": Path(hub_path).name,
            "dino_hub_source_tree_sha256": hub_source_sha,
            "dino_checkpoint_name": checkpoint.name,
            "dino_checkpoint_sha256": checkpoint_sha,
            "category_inputs": {
                category: {
                    "class_json_sha256": item["class_json_sha256"],
                    "query_positions_sha256": sha256_json(list(item["positions"])),
                    "query_positions_in_prefix": len(item["positions"]),
                    "c64_score_tape_sha256": item["c64_score_sha256"],
                    "support_group_sha256": item["support_group_sha256"],
                    "fixed_support_sha256_expected": str(
                        json.loads((STAGE05 / "score_tape_manifest.json").read_text(
                            encoding="utf-8"))["classes"][category]["score_tapes"]["0"]
                        ["fixed_support_sha256"]),
                    "theta_c64_q90": item["theta"],
                    "sigma_c64": item["sigma"],
                    "stream_count": len(item["stream"]),
                    "support_count": len(item["support_paths"]),
                    "c64_count": 64,
                } for category, item in preflight["categories"].items()},
            "runtime_baseline": {
                "process_cpu_seconds_start": time.process_time(),
                "process_started_monotonic": started,
            },
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_run_manifest(budget, output, manifest)
        shared: dict[str, Any] = {
            **preflight,
            "impl": impl,
            "p0_config_object": preflight["p0_config_object"],
            "pilot_start_monotonic": started,
        }
        category_summaries: dict[str, Any] = {}
        model_profiles: dict[str, Any] = {}
        for category in CATEGORIES:
            # Stage 0.5 explicitly reseeded each category before loading DINO.
            runtime_profile = r1._configure_historical_runtime(impl, torch)
            if runtime_profile["sdp"] != preflight["runtime_profile"]["sdp"]:
                raise Exp2AIntegrityError("Per-category runtime profile changed")
            torch.cuda.reset_peak_memory_stats()
            category_model_start = time.perf_counter()
            backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
            model_state_sha = r1._model_state_sha256(backbone.model)
            encoder_fingerprint = r1.sha256_json({
                "model_name": preflight["scorer_profile"]["model_name"],
                "loaded_model_state_sha256": model_state_sha,
                "hub_source_path": Path(hub_path).name,
                "hub_source_tree_sha256": hub_source_sha,
                "vmb_source_sha256": source_sha["vmb_visionad_new.py"],
                "v22_encoder_source_sha256": source_sha[
                    "test_phase3_AIF_simplified_v22.py"],
                "hook_layers": list(backbone.layers),
            })
            preprocess_fingerprint = r1.sha256_json({
                "profile": preflight["scorer_profile"], "rgb": "PIL.convert(RGB)",
                "resize": [448, 448], "crop": [392, 392],
                "interpolation": "PIL.BICUBIC", "normalization": "model mean/std",
                "fusion": "concat_then_l2_normalize", "view": "id",
                "pillow_version": __import__("PIL").__version__,
            })
            model_profiles[category] = {
                "runtime_profile": runtime_profile,
                "loaded_model_state_sha256": model_state_sha,
                "encoder_fingerprint_sha256": encoder_fingerprint,
                "preprocess_fingerprint_sha256": preprocess_fingerprint,
                "model_initialization_seconds": time.perf_counter() - category_model_start,
            }
            category_summaries[category] = _capture_category(
                category, preflight["categories"][category], shared,
                dataset_root=dataset_root, json_root=json_root,
                root=output, budget=budget, backbone=backbone,
                authorization_id=authorization_id, run_id=run_id,
                event_counts=counts)
            category_summaries[category]["loaded_model_state_sha256"] = model_state_sha
            category_summaries[category]["encoder_fingerprint_sha256"] = encoder_fingerprint
            category_summaries[category]["preprocess_fingerprint_sha256"] = preprocess_fingerprint
            backbone.cleanup()
            del backbone
            torch.cuda.empty_cache()
        expected_positions = sum(
            len(info["positions"]) for info in preflight["categories"].values())
        feedback_rows = _read_jsonl(output / "private_query_feedback.jsonl")
        if len(feedback_rows) != expected_positions:
            raise Exp2AIntegrityError("Queried-label feedback count differs from frozen tapes")
        if len(_read_jsonl(output / "event_scores.jsonl")) != len(CATEGORIES) * MAX_EVENTS:
            raise Exp2AIntegrityError("Stage A score rows are incomplete")
        if len(_read_jsonl(output / "event_transitions.jsonl")) != len(CATEGORIES) * MAX_EVENTS:
            raise Exp2AIntegrityError("Stage A transitions are incomplete")
        for row in _read_jsonl(output / "event_scores.jsonl"):
            _check_event_score_row(row, output, {
                "base_patch_map", "top32_ids", "top32_features",
                "c1_delta_full", "c1_delta_no_outcome_weight",
                "c1_delta_no_margin", "c1_delta_no_both",
                "c2_delta_a002", "c2_delta_a020"})
        parity = _verify_switch_reference(output)
        seal = _write_stage_a_seal(output, budget, counts, {
            "status": "STAGE_A_SEALED",
            "switch_r1_reference_parity": parity,
            "category_runtime": category_summaries,
            "model_profiles": model_profiles,
            "actual_query_count": expected_positions,
            "actual_query_label_count": len(feedback_rows),
            "actual_event_count": sum(counts.values()),
            "total_stage_a_seconds": time.monotonic() - started,
            "output_bytes_at_seal": budget._audit_tree(),
        })
        return {"status": "STAGE_A_SEALED",
                "stage_a_seal_sha256": sha256_file(output / "stage_a_seal.json"),
                "switch_r1_reference_parity": parity,
                "category_summaries": category_summaries,
                "total_seconds": time.monotonic() - started,
                "output_bytes": budget._audit_tree(),
                "seal": seal}
    except BaseException as exc:
        try:
            if (output / "run_manifest.json").is_file():
                _append_record(budget, output / "event_transitions.jsonl", {
                    "schema": "r2.exp2a.failure_transition.v1",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc)[:1000],
                    "category_event_counts": dict(counts),
                }, failure_reserved=True)
        except Exception:
            pass
        try:
            _write_json(budget, output / "failure.json", {
                "schema": "r2.exp2a.failure.v1",
                "status": "INCOMPLETE",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc)[:1600],
                "partial_category_event_counts": dict(counts),
                "automatic_restart": False,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, failure_reserved=True)
        except Exception:
            pass
        raise
    finally:
        budget.close()


def _quantiles(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "q05": None, "q25": None,
                "median": None, "q75": None, "q95": None}
    if not np.isfinite(array).all():
        raise Exp2AIntegrityError("Nonfinite diagnostic value")
    return {
        "count": int(array.size), "mean": float(array.mean()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
    }


def _cross_group_ranking_diagnostics(labels: np.ndarray, scores_by_arm: Mapping[str, np.ndarray],
                                     groups: Sequence[str]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    g = np.asarray(groups, dtype=object)
    defects = np.flatnonzero(y == 1)
    normals = np.flatnonzero(y == 0)
    if not len(defects) or not len(normals):
        return {"undefined_reason": "requires at least one normal and one defect",
                "denominator": 0}
    cross_group = g[defects, None] != g[normals][None, :]
    denominator = int(cross_group.sum())
    result: dict[str, Any] = {
        "cross_source_group_normal_defect_pair_denominator": denominator,
        "same_source_group_pairs_excluded": int(cross_group.size - denominator),
        "by_arm_vs_c0": {},
    }
    baseline = np.asarray(scores_by_arm["c0"], dtype=np.float64)
    base_diff = baseline[defects, None] - baseline[normals][None, :]
    for arm, raw_scores in scores_by_arm.items():
        if arm == "c0":
            continue
        candidate = np.asarray(raw_scores, dtype=np.float64)
        diff = candidate[defects, None] - candidate[normals][None, :]
        repaired = cross_group & (base_diff < 0) & (diff > 0)
        damaged = cross_group & (base_diff > 0) & (diff < 0)
        base_tie = cross_group & (base_diff == 0)
        result["by_arm_vs_c0"][arm] = {
            "strict_wrong_to_strict_correct_repaired": int(repaired.sum()),
            "strict_correct_to_strict_wrong_damaged": int(damaged.sum()),
            "net_strict_pair_repair": int(repaired.sum() - damaged.sum()),
            "baseline_ties_resolved_correct": int((base_tie & (diff > 0)).sum()),
            "baseline_ties_resolved_wrong": int((base_tie & (diff < 0)).sum()),
            "candidate_ties_from_strictly_correct": int(
                (cross_group & (base_diff > 0) & (diff == 0)).sum()),
            "candidate_ties_from_strictly_wrong": int(
                (cross_group & (base_diff < 0) & (diff == 0)).sum()),
        }
    return result


def _cross_group_pair_harm_for_arm(ranking: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Return an arm result or preserve why the category result is undefined."""
    arm_result = ranking.get("by_arm_vs_c0", {}).get(arm)
    if arm_result is None:
        return {
            "status": "UNDEFINED",
            "reason": ranking.get(
                "undefined_reason", f"ranking result is unavailable for {arm}"),
            "cross_source_group_normal_defect_pair_denominator": int(
                ranking.get("cross_source_group_normal_defect_pair_denominator", 0)),
            "same_source_group_pairs_excluded": int(
                ranking.get("same_source_group_pairs_excluded", 0)),
        }
    return {
        "status": "DEFINED",
        "cross_source_group_normal_defect_pair_denominator": int(
            ranking["cross_source_group_normal_defect_pair_denominator"]),
        **arm_result,
    }


def _prediction_damage(labels: np.ndarray, scores_by_arm: Mapping[str, np.ndarray],
                       threshold: float) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    base = np.asarray(scores_by_arm["c0"]) >= float(threshold)
    result = {}
    for arm, score in scores_by_arm.items():
        if arm == "c0":
            continue
        pred = np.asarray(score) >= float(threshold)
        result[arm] = {
            "prediction_flips_vs_c0": int(np.count_nonzero(pred != base)),
            "repaired_c0_false_positives": int(np.count_nonzero(
                base & (y == 0) & ~pred)),
            "new_false_positives": int(np.count_nonzero(
                ~base & (y == 0) & pred)),
            "repaired_c0_false_negatives": int(np.count_nonzero(
                ~base & (y == 1) & pred)),
            "new_false_negatives": int(np.count_nonzero(
                base & (y == 1) & ~pred)),
        }
    return result


def _cluster_bootstrap_category(labels: np.ndarray, groups: Sequence[str],
                                scores: Mapping[str, np.ndarray], *,
                                iterations: int, seed: int) -> dict[str, Any]:
    from experiments.calibration.r2_perf1_stage_b import _auc_ap
    y = np.asarray(labels, dtype=np.int8)
    g = np.asarray(groups, dtype=object)
    unique = sorted(set(g.tolist()))
    if not unique:
        raise Exp2AIntegrityError("Source Group bootstrap has no groups")
    member = {group: np.flatnonzero(g == group) for group in unique}
    arrays = {arm: np.asarray(value, dtype=np.float64)
              for arm, value in scores.items()}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        "c1_full_minus_c0": [], "c1_full_minus_c2_a002": [],
        "outcome_effect_margin_enabled": [],
        "outcome_effect_margin_disabled": [],
        "margin_effect_outcome_enabled": [],
        "margin_effect_outcome_disabled": [], "interaction": [],
        "c1_no_outcome_weight_minus_c0": [],
        "c1_no_margin_minus_c0": [], "c1_no_both_minus_c0": [],
        "c2_a020_minus_c0": [],
    }
    auprc_samples: dict[str, list[float]] = {arm: [] for arm in SCORE_ARMS}
    auprc_diff_samples: dict[str, list[float]] = {
        arm: [] for arm in SCORE_ARMS if arm != "c0"}
    valid_replicates = 0
    for _ in range(iterations):
        selected = rng.integers(0, len(unique), size=len(unique))
        indices = np.concatenate([member[unique[int(index)]] for index in selected])
        boot_y = y[indices]
        aucs, aps = {}, {}
        for arm, score in arrays.items():
            auc, ap, _reason = _auc_ap(boot_y, score[indices])
            aucs[arm] = auc
            aps[arm] = ap
        if any(aucs[arm] is None for arm in SCORE_ARMS):
            continue
        valid_replicates += 1
        for arm in SCORE_ARMS:
            if aps[arm] is not None:
                auprc_samples[arm].append(float(aps[arm]))
                if arm != "c0" and aps["c0"] is not None:
                    auprc_diff_samples[arm].append(float(aps[arm] - aps["c0"]))
        contrasts = factorial_effects(
            aucs["c1_full"], aucs["c1_no_outcome_weight"],
            aucs["c1_no_margin"], aucs["c1_no_both"])
        values = {
            "c1_full_minus_c0": aucs["c1_full"] - aucs["c0"],
            "c1_full_minus_c2_a002": aucs["c1_full"] - aucs["c2_a002"],
            "c1_no_outcome_weight_minus_c0":
                aucs["c1_no_outcome_weight"] - aucs["c0"],
            "c1_no_margin_minus_c0": aucs["c1_no_margin"] - aucs["c0"],
            "c1_no_both_minus_c0": aucs["c1_no_both"] - aucs["c0"],
            "c2_a020_minus_c0": aucs["c2_a020"] - aucs["c0"],
            **contrasts,
        }
        for key, value in values.items():
            samples[key].append(float(value))

    def interval(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {"lower_2_5": None, "upper_97_5": None,
                    "valid_replicates": 0}
        return {
            "lower_2_5": float(np.quantile(values, 0.025)),
            "upper_97_5": float(np.quantile(values, 0.975)),
            "valid_replicates": len(values),
        }
    return {
        "method": "paired Source Group cluster bootstrap with replacement",
        "replicates_requested": int(iterations), "valid_replicates": valid_replicates,
        "seed": int(seed), "unique_source_groups": len(unique),
        "conditional_on_fixed_historical_causal_trajectory": True,
        "not_uncertainty_over_independently_regenerated_online_streams": True,
        "intervals": {key: interval(value) for key, value in samples.items()},
        "auprc_intervals_by_arm": {
            arm: interval(values) for arm, values in auprc_samples.items()},
        "auprc_difference_vs_c0_intervals": {
            arm: interval(values) for arm, values in auprc_diff_samples.items()},
    }


def _sealed_stage_a_inputs(root: Path) -> tuple[dict[str, list[dict[str, Any]]],
                                                dict[str, list[dict[str, Any]]],
                                                dict[str, list[dict[str, Any]]]]:
    seal_path = root / "stage_a_seal.json"
    if not seal_path.is_file():
        raise Exp2AIntegrityError("Stage B cannot run without a Stage A seal")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (seal.get("schema") != "r2.exp2a.stage_a_seal.v1" or
            seal.get("status") != "SEALED_COMPLETE" or
            seal.get("category_event_counts") != {
                category: MAX_EVENTS for category in CATEGORIES}):
        raise Exp2AIntegrityError("Stage A is incomplete; outcome access is forbidden")
    from experiments.calibration.r1_matched_correction_pilot import sha256_file
    for filename, key in (
            ("event_scores.jsonl", "score_rows_sha256"),
            ("event_transitions.jsonl", "transition_rows_sha256"),
            ("strength_diagnostics.jsonl", "strength_diagnostics_sha256"),
            ("private_query_feedback.jsonl", "queried_feedback_sha256")):
        path = root / filename
        expected = seal.get(key)
        if not path.is_file() or sha256_file(path) != expected:
            raise Exp2AIntegrityError(f"Stage A seal checksum failed for {filename}")
    score_rows = _read_jsonl(root / "event_scores.jsonl")
    transition_rows = _read_jsonl(root / "event_transitions.jsonl")
    diag_rows = _read_jsonl(root / "strength_diagnostics.jsonl")
    if len(score_rows) != len(CATEGORIES) * MAX_EVENTS or \
            len(transition_rows) != len(score_rows):
        raise Exp2AIntegrityError("Stage A score/transition row counts are incomplete")
    expected_fields = {
        "base_patch_map", "top32_ids", "top32_features",
        "c1_delta_full", "c1_delta_no_outcome_weight",
        "c1_delta_no_margin", "c1_delta_no_both",
        "c2_delta_a002", "c2_delta_a020",
    }
    grouped_scores = {category: [] for category in CATEGORIES}
    grouped_transitions = {category: [] for category in CATEGORIES}
    grouped_diags = {category: [] for category in CATEGORIES}
    for row in score_rows:
        category = row.get("category")
        if category not in grouped_scores:
            raise Exp2AIntegrityError("Stage A contains an unauthorized category")
        if row.get("position") != len(grouped_scores[category]):
            raise Exp2AIntegrityError("Stage A positions are duplicated or out of order")
        if "label" in row or "outcome" in row or "anomaly_class" in row:
            raise Exp2AIntegrityError("Stage A public score row contains an outcome")
        _check_event_score_row(row, root, expected_fields)
        grouped_scores[category].append(row)
    for row in transition_rows:
        category = row.get("category")
        if category not in grouped_transitions:
            raise Exp2AIntegrityError("Stage A transition has an unauthorized category")
        if row.get("position") != len(grouped_transitions[category]):
            raise Exp2AIntegrityError("Stage A transitions are out of order")
        grouped_transitions[category].append(row)
    for row in diag_rows:
        category = row.get("category")
        if category not in grouped_diags:
            raise Exp2AIntegrityError("Strength diagnostics contain an unauthorized category")
        grouped_diags[category].append(row)
    for category in CATEGORIES:
        if (len(grouped_scores[category]) != MAX_EVENTS or
                len(grouped_transitions[category]) != MAX_EVENTS):
            raise Exp2AIntegrityError(f"{category}: Stage A coverage is incomplete")
        if any(bool(score["query_tape_decision"]) !=
               bool(transition["query_decision"])
               for score, transition in zip(
                   grouped_scores[category], grouped_transitions[category])):
            raise Exp2AIntegrityError(f"{category}: query decision differs across sealed files")
    return grouped_scores, grouped_transitions, grouped_diags


def _stage_b_labels(category: str, rows: Sequence[Mapping[str, Any]],
                    dataset_root: Path, json_root: Path) -> tuple[np.ndarray, list[str]]:
    """Project only first-512 outcomes after exact event and group identity matches."""
    from experiments.calibration.r2_perf1_stage_b import (
        StageBError, _TestPrefixProjection, _matched_test_label)
    from experiments.calibration import stage05
    from experiments.risk_safe.preflight import parse_image_name
    path = json_root / f"{category}.json"
    if not path.is_file() or sha256_file(path) != rows[0]["class_json_sha256"]:
        raise Exp2AIntegrityError(f"{category}: outcome source fingerprint changed")
    info = json.loads((STAGE05 / "frozen_inputs.json").read_text(encoding="utf-8"))[
        "classes"][category]
    expected = list(rows)
    labels = np.full(MAX_EVENTS, -1, dtype=np.int8)
    observed_groups = [""] * MAX_EVENTS
    projector = _TestPrefixProjection(path.read_text(encoding="utf-8"))

    def matched(row_index: int, image_path: str, raw_record: str) -> None:
        if row_index < 0 or row_index >= MAX_EVENTS:
            raise Exp2AIntegrityError("Stage B returned an event outside the sealed prefix")
        sealed = expected[row_index]
        absolute = stage05._absolute_image_path(
            dataset_root, info["prefix"], image_path)
        parsed = parse_image_name(absolute, expected_class=category)
        observed_identity = hashlib.sha256(
            str(parsed.rel_path).encode("utf-8")).hexdigest()
        if observed_identity != sealed["source_identity_sha256"]:
            raise Exp2AIntegrityError(f"{category}:{row_index} identity digest differs")
        from experiments.calibration.r1_matched_correction_pilot import opaque_sample_group
        group = opaque_sample_group(parsed.official_sample_key)
        if group != sealed["source_group_sha256"]:
            raise Exp2AIntegrityError(f"{category}:{row_index} source group differs")
        anomaly_class = _matched_test_label(
            raw_record, expected_identity=sealed["source_identity_sha256"],
            observed_identity=observed_identity)
        normal_class = str(projector.normal_class)
        labels[row_index] = 0 if anomaly_class == normal_class else 1
        observed_groups[row_index] = group
    try:
        projected = projector.project(expected, matched)
    except Exception as exc:
        raise Exp2AIntegrityError(f"{category}: Stage B prefix projection failed: {exc}") from exc
    if (projected != MAX_EVENTS or np.any(labels < 0) or
            any(not group for group in observed_groups)):
        raise Exp2AIntegrityError(f"{category}: Stage B did not match every prefix outcome")
    return labels, observed_groups


def _mechanism_summary(category: str, transitions: Sequence[Mapping[str, Any]],
                       diagnostics: Sequence[Mapping[str, Any]],
                       patch_values: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for arm in ARMS:
        records = [row["arms"][arm]["diagnostic"] for row in diagnostics]
        raw = [row["pre_clip_beta_scale_times_margin"] for row in records]
        base = [row["post_base_clip_beta"] for row in records]
        outcome_beta = [row["post_outcome_clip_beta"] for row in records]
        proposed = [v for row in records
                    for v in (row["beta_patch_float32_before_codebook_clip"] or [])]
        clipped = [v for row in records
                   for v in (row["beta_after_original_codebook_max_clip"] or [])]
        multipliers = [v for row in records
                       for v in (row["patch_multiplier"] or [])]
        dN_all = [v for row in records
                  for v in (row["selected_dN_before_gate"] or [])]
        dN_keep = [v for row in records
                   for v in (row["dN_after_gate"] or [])]
        outcome_counts = {key: sum(row["outcome"] == key for row in diagnostics)
                          for key in ("FP", "FN", "TN", "TP")}
        stats = {key: sum(int(row["native_layer_stats"].get(key, 0))
                          for row in (item["arms"][arm] for item in diagnostics))
                 for key in ("inserted", "merged", "replaced", "skipped",
                             "same_sign_replaced", "cross_sign_replaced",
                             "same_sign_merged")}
        outcome_weight_by_outcome = {
            outcome: _quantiles([
                float(item["arms"][arm]["diagnostic"]["outcome_weight"])
                for item in diagnostics if item["outcome"] == outcome])
            for outcome in ("FP", "FN", "TN", "TP")}
        occupancy = [int(row["c1_anchor_occupancy_after"][arm])
                     for row in transitions]
        beta_at_min = sum(abs(float(value)) <= FIXED_BETA_MIN + 1e-6
                          for value in clipped)
        beta_at_max = sum(abs(float(value)) >= 0.20 - 1e-6
                          for value in clipped)
        metrics[arm] = {
            "query_update_count": len(records),
            "outcome_query_counts": outcome_counts,
            "outcome_weight_by_outcome": outcome_weight_by_outcome,
            "native_update_counts": stats,
            "pre_clip_beta_scale_margin": _quantiles(raw),
            "post_base_clip_beta": _quantiles(base),
            "post_outcome_clip_beta": _quantiles(outcome_beta),
            "patch_beta_before_codebook_max_clip": _quantiles(proposed),
            "beta_after_original_codebook_max_clip": _quantiles(clipped),
            "patch_multiplier": _quantiles(multipliers),
            "selected_dN_before_gate": _quantiles(dN_all),
            "admitted_dN_after_gate": _quantiles(dN_keep),
            "fraction_clipped_values_at_beta_min": (
                beta_at_min / len(clipped) if clipped else None),
            "fraction_clipped_values_at_beta_max": (
                beta_at_max / len(clipped) if clipped else None),
            "selected_patch_score_correction": _quantiles(patch_values.get(arm, [])),
            "anchor_occupancy": {
                "initial": occupancy[0] if occupancy else 0,
                "final": occupancy[-1] if occupancy else 0,
                "maximum": max(occupancy) if occupancy else 0,
                "at_positions": {
                    str(position): occupancy[position]
                    for position in (0, 127, 255, 383, 511) if len(occupancy) > position},
            },
            "stored_anchor_sign_counts_final": (
                transitions[-1]["c1_anchor_sign_counts_after"][arm]
                if transitions else None),
            "stored_anchor_outcome_counts_final": (
                transitions[-1]["c1_outcome_group_counts"][arm]
                if transitions else None),
        }
    if diagnostics:
        by_position = {int(row["position"]): row for row in diagnostics}
        activation = {}
        full_rows = {position: item["arms"]["c1_full"]["diagnostic"]
                     for position, item in by_position.items()}
        for arm in ARMS[1:]:
            changed = equal = 0
            for position, item in by_position.items():
                reference = full_rows[position]
                candidate = item["arms"][arm]["diagnostic"]
                vector_reference = reference.get(
                    "beta_patch_float32_before_codebook_clip") or []
                vector_candidate = candidate.get(
                    "beta_patch_float32_before_codebook_clip") or []
                vector_equal = (
                    len(vector_reference) == len(vector_candidate) and
                    np.array_equal(np.asarray(vector_reference, dtype=np.float32),
                                   np.asarray(vector_candidate, dtype=np.float32)))
                scalar_equal = (
                    reference["post_base_clip_beta"] ==
                    candidate["post_base_clip_beta"] and
                    reference["post_outcome_clip_beta"] ==
                    candidate["post_outcome_clip_beta"])
                if vector_equal and scalar_equal:
                    equal += 1
                else:
                    changed += 1
            activation[arm] = {
                "queried_updates_with_beta_changed_vs_full": changed,
                "queried_updates_equal_to_full": equal,
                "query_updates_total": len(by_position),
                "effective_activation_fraction": (
                    changed / len(by_position) if by_position else None),
            }
    else:
        activation = {arm: {"queried_updates_with_beta_changed_vs_full": 0,
                            "queried_updates_equal_to_full": 0,
                            "query_updates_total": 0,
                            "effective_activation_fraction": None}
                      for arm in ARMS[1:]}
    return {
        "per_arm": metrics,
        "ablation_beta_activation_vs_full": activation,
        "c1_state_divergence_vs_full": {
            arm: {
                "events_with_different_state": sum(
                    row["c1_state_sha256_after"][arm] !=
                    row["c1_state_sha256_after"]["c1_full"]
                    for row in transitions),
                "events_with_different_image_score": None,
            } for arm in ARMS},
        "same_source_anchor_exclusions_total": {
            arm: None for arm in ARMS},
        "warning": (
            "A null factor estimate under rare beta activation or heavy clipping "
            "does not show that the mechanism is unnecessary."),
    }


def _evaluate_category(category: str, rows: Sequence[Mapping[str, Any]],
                       transitions: Sequence[Mapping[str, Any]],
                       diagnostics: Sequence[Mapping[str, Any]],
                       labels: np.ndarray, groups: Sequence[str],
                       root: Path, *, dataset_root: Path,
                       bootstrap_iterations: int) -> dict[str, Any]:
    from experiments.calibration.r1_matched_correction_pilot import max_score
    from experiments.calibration.r2_perf1_stage_b import _metric_record
    expected_fields = {
        "base_patch_map", "top32_ids", "top32_features",
        "c1_delta_full", "c1_delta_no_outcome_weight",
        "c1_delta_no_margin", "c1_delta_no_both",
        "c2_delta_a002", "c2_delta_a020",
    }
    score_arrays = {arm: np.empty(MAX_EVENTS, dtype=np.float64)
                    for arm in SCORE_ARMS}
    patch_values: dict[str, list[float]] = {
        arm: [] for arm in (*ARMS, "c2_a002", "c2_a020")}
    delta_field = {
        "c1_full": "c1_delta_full",
        "c1_no_outcome_weight": "c1_delta_no_outcome_weight",
        "c1_no_margin": "c1_delta_no_margin",
        "c1_no_both": "c1_delta_no_both",
        "c2_a002": "c2_delta_a002", "c2_a020": "c2_delta_a020",
    }
    for position, row in enumerate(rows):
        arrays = _check_event_score_row(row, root, expected_fields)
        for arm in SCORE_ARMS:
            score_arrays[arm][position] = float(row["scores"][arm])
        for arm, field in delta_field.items():
            patch_values[arm].extend(
                np.asarray(arrays[field], dtype=np.float64).tolist())
    score_metrics = {
        arm: _metric_record(labels.tolist(), scores.tolist(),
                            float(rows[0]["theta_c64_q90"]))
        for arm, scores in score_arrays.items()}
    aurocs = {arm: score_metrics[arm]["image_auroc"] for arm in SCORE_ARMS}
    if any(value is None for value in aurocs.values()):
        undefined = {
            arm: score_metrics[arm]["undefined_reason"]
            for arm in SCORE_ARMS if aurocs[arm] is None}
    else:
        undefined = {}
    factorial = (factorial_effects(
        aurocs["c1_full"], aurocs["c1_no_outcome_weight"],
        aurocs["c1_no_margin"], aurocs["c1_no_both"])
        if not undefined else None)
    paired = {
        arm: {
            "auroc_difference_vs_c0": (
                aurocs[arm] - aurocs["c0"]
                if aurocs[arm] is not None and aurocs["c0"] is not None else None),
            "auroc_difference_vs_c2_a002": (
                aurocs[arm] - aurocs["c2_a002"]
                if aurocs[arm] is not None and aurocs["c2_a002"] is not None else None),
        } for arm in SCORE_ARMS}
    effects = factorial or {}
    bootstrap = _cluster_bootstrap_category(
        labels, groups, score_arrays, iterations=bootstrap_iterations,
        seed=20261101 + CATEGORIES.index(category))
    prediction_diagnostics = _prediction_damage(
        labels, score_arrays, float(rows[0]["theta_c64_q90"]))
    ranking = _cross_group_ranking_diagnostics(labels, score_arrays, groups)
    image_corrections = {
        arm: _quantiles((score_arrays[arm] - score_arrays["c0"]).tolist())
        for arm in SCORE_ARMS if arm != "c0"}
    abs_image_corrections = {
        arm: _quantiles(np.abs(score_arrays[arm] - score_arrays["c0"]).tolist())
        for arm in SCORE_ARMS if arm != "c0"}
    source_group_count = len(set(groups))
    mechanisms = _mechanism_summary(
        category, transitions, diagnostics, patch_values)
    for arm in ARMS:
        mechanisms["c1_state_divergence_vs_full"][arm][
            "events_with_different_image_score"] = sum(
                rows[position]["scores"][arm] != rows[position]["scores"]["c1_full"]
                for position in range(MAX_EVENTS))
        mechanisms["same_source_anchor_exclusions_total"][arm] = sum(
            int(row["c1_source_exclusion"][arm][
                "same_source_excluded_anchor_slots"]) for row in rows)
    return {
        "events": MAX_EVENTS,
        "source_group_count": source_group_count,
        "effective_sample_denominator": source_group_count,
        "label_counts": {
            "normal": int(np.count_nonzero(labels == 0)),
            "defect": int(np.count_nonzero(labels == 1)),
        },
        "actual_query_count": int(sum(row["query_tape_decision"] for row in rows)),
        "queried_normal_count": int(sum(
            row["query_tape_decision"] and int(labels[position]) == 0
            for position, row in enumerate(rows))),
        "queried_defect_count": int(sum(
            row["query_tape_decision"] and int(labels[position]) == 1
            for position, row in enumerate(rows))),
        "metrics_by_arm": score_metrics,
        "auroc_by_arm": aurocs,
        "paired_auroc_differences": paired,
        "factorial_effects": effects,
        "factor_order": {
            "F11": "margin enabled, outcome weights enabled",
            "F10": "margin enabled, outcome weights disabled",
            "F01": "margin disabled, outcome weights enabled",
            "F00": "margin disabled, outcome weights disabled",
        },
        "undefined_metrics": undefined,
        "source_group_bootstrap": bootstrap,
        "prediction_flips_and_error_changes": prediction_diagnostics,
        "cross_group_ranking_repair_and_harm": ranking,
        "signed_image_score_correction_vs_c0": image_corrections,
        "absolute_image_score_correction_vs_c0": abs_image_corrections,
        "mechanism_diagnostics": mechanisms,
        "c2_a002_was_selected_after_switch_development_exposure": True,
        "cross_category_independent_confirmation": False,
    }


def _stage_b_worker(*, output: Path, dataset_root: Path,
                    json_root: Path, bootstrap_iterations: int = 1000) -> dict[str, Any]:
    stage_b_cpu_started = time.process_time()
    stage_b_wall_started = time.monotonic()
    from experiments.calibration import r1_matched_correction_pilot as r1
    stage_a_seal = json.loads((output / "stage_a_seal.json").read_text(
        encoding="utf-8"))
    run_manifest = json.loads((output / "run_manifest.json").read_text(
        encoding="utf-8"))
    scores, transitions, diagnostics = _sealed_stage_a_inputs(output)
    by_category: dict[str, Any] = {}
    for category in CATEGORIES:
        labels, groups = _stage_b_labels(
            category, scores[category], dataset_root, json_root)
        by_category[category] = _evaluate_category(
            category, scores[category], transitions[category], diagnostics[category],
            labels, groups, output, dataset_root=dataset_root,
            bootstrap_iterations=bootstrap_iterations)
    macro_auroc = category_equal_aggregate(
        {category: by_category[category]["auroc_by_arm"]
         for category in CATEGORIES}, SCORE_ARMS)
    macro_effects = category_equal_aggregate(
        {category: by_category[category]["factorial_effects"]
         for category in CATEGORIES}, (
             "outcome_effect_margin_enabled", "outcome_effect_margin_disabled",
             "margin_effect_outcome_enabled", "margin_effect_outcome_disabled",
             "interaction"))
    c1_vs_c2 = {
        category: by_category[category]["paired_auroc_differences"][
            "c1_full"]["auroc_difference_vs_c2_a002"]
        for category in CATEGORIES}
    c1_vs_c2_values = [value for value in c1_vs_c2.values() if value is not None]
    macro_auroc["c1_full_minus_c2_a002"] = {
        "by_category": c1_vs_c2,
        "category_equal_macro": (
            float(np.mean(c1_vs_c2_values))
            if len(c1_vs_c2_values) == len(CATEGORIES) else None),
        "worst_category": (
            min(c1_vs_c2_values)
            if len(c1_vs_c2_values) == len(CATEGORIES) else None),
        "positive_category_count": sum(value > 0 for value in c1_vs_c2_values),
        "negative_category_count": sum(value < 0 for value in c1_vs_c2_values),
        "category_range": (max(c1_vs_c2_values) - min(c1_vs_c2_values)
                           if len(c1_vs_c2_values) == len(CATEGORIES) else None),
        "category_sd_population": (float(np.std(c1_vs_c2_values))
                                  if len(c1_vs_c2_values) == len(CATEGORIES) else None),
    }
    activation_by_factor = {}
    for key, arms in (
            ("outcome_weight", ("c1_no_outcome_weight", "c1_no_both")),
            ("margin", ("c1_no_margin", "c1_no_both"))):
        for arm in arms:
            rates = {
                category: by_category[category]["mechanism_diagnostics"][
                    "ablation_beta_activation_vs_full"][arm][
                        "effective_activation_fraction"]
                for category in CATEGORIES}
            values = [value for value in rates.values() if value is not None]
            activation_by_factor[f"{key}:{arm}"] = {
                "by_category": rates,
                "category_equal_mean": float(np.mean(values)) if values else None,
                "categories_with_nonzero_activation": sum(value > 0 for value in values),
            }
    macro = {
        "auroc_by_arm": macro_auroc,
        "factorial_effects": macro_effects,
        "beta_activation": activation_by_factor,
        "unit_of_cross_category_description": "four historically exposed development categories",
        "not_an_independent_confirmation_sample": True,
    }
    parity = json.loads((output / "stage_a_seal.json").read_text(
        encoding="utf-8"))["metadata"]["switch_r1_reference_parity"]
    outcomes = {
        "q1_margin_adaptation": macro_effects[
            "margin_effect_outcome_enabled"]["category_equal_macro"],
        "q2_outcome_weighting": macro_effects[
            "outcome_effect_margin_enabled"]["category_equal_macro"],
        "q3_interaction": macro_effects["interaction"]["category_equal_macro"],
        "q4_consistency": {
            key: macro_effects[key]["positive_category_count"]
            for key in macro_effects if key != "interaction"},
        "q5_c1_full_vs_c2_a002_macro_difference": macro_auroc[
            "c1_full_minus_c2_a002"]["category_equal_macro"],
        "q6_codebook_trajectory_and_ranking": {
            category: {
                "factorial_state_divergence": {
                    arm: by_category[category]["mechanism_diagnostics"][
                        "c1_state_divergence_vs_full"][arm][
                            "events_with_different_state"]
                    for arm in ARMS},
                "cross_group_pair_harm": {
                    arm: _cross_group_pair_harm_for_arm(
                        by_category[category]["cross_group_ranking_repair_and_harm"],
                        arm)
                    for arm in (*ARMS, "c2_a002", "c2_a020")},
            } for category in CATEGORIES},
        "q7_strength_policy_central_innovation_claim": (
            "not established from four historically exposed development categories; "
            "describe mechanism only at the strength supported by the observed factorial effects"),
    }
    if parity.get("status") != "PASS":
        verdict = "REFERENCE_PARITY_UNVERIFIED"
    elif any(value is None for value in (
            outcomes["q1_margin_adaptation"], outcomes["q2_outcome_weighting"])):
        verdict = "INSUFFICIENT_ACTIVATION"
    elif (outcomes["q1_margin_adaptation"] <= 0 and
          outcomes["q2_outcome_weighting"] <= 0):
        verdict = "NO_CLEAR_INCREMENTAL_BENEFIT"
    elif (macro_effects["margin_effect_outcome_enabled"]["positive_category_count"] < 3 or
          macro_effects["outcome_effect_margin_enabled"]["positive_category_count"] < 3):
        verdict = "EFFECT_HETEROGENEOUS"
    else:
        verdict = "CAUSAL_EFFECT_SUPPORTED_IN_DEVELOPMENT"
    aggregate = {
        "schema": "r2.exp2a.sanitized_aggregate.v1",
        "experiment": "R2-EXP2A",
        "status": "COMPLETE",
        "development_verdict": verdict,
        "category_order": list(CATEGORIES),
        "event_horizon_per_category": MAX_EVENTS,
        "categories_are_historically_exposed_development_data": True,
        "outcome_metrics_by_category": by_category,
        "category_equal_aggregation": macro,
        "scientific_questions": outcomes,
        "c1_full_reference_parity": parity,
        "input_provenance": {
            "parent_branch": run_manifest["identity"]["branch"],
            "parent_commit": run_manifest["identity"]["parent_commit"],
            "source_hashes": run_manifest["source_hashes"],
            "r2_exp2a_source_sha256": run_manifest["r2_exp2a_source_sha256"],
            "stage_a_runner_source_sha256": run_manifest["r2_exp2a_source_sha256"],
            "stage_b_runner_source_sha256": r1.sha256_file(Path(__file__)),
            "stage05_frozen_inputs_sha256": run_manifest["stage05_frozen_inputs_sha256"],
            "stage05_checksums_sha256": run_manifest["stage05_checksums_sha256"],
            "qats_binding_sha256": run_manifest["qats_binding_sha256"],
            "qats_verification_sha256": run_manifest["qats_verification_sha256"],
            "scorer_profile_sha256": run_manifest["scorer_profile_sha256"],
            "dino_checkpoint_sha256": run_manifest["dino_checkpoint_sha256"],
            "category_inputs": run_manifest["category_inputs"],
            "category_model_profiles": stage_a_seal["metadata"]["model_profiles"],
            "support_regime": run_manifest["support_regime"],
        },
        "stage_a_resource_use": stage_a_seal["metadata"]["category_runtime"],
        "stage_b_resource_use": {
            "cpu_process_seconds": time.process_time() - stage_b_cpu_started,
            "wall_seconds": time.monotonic() - stage_b_wall_started,
            "gpu_used": False,
        },
        "interpretation_limits": [
            "Four categories are not independent confirmation categories.",
            "Source Group intervals condition on the fixed online trajectory.",
            "C2(a=0.02) was selected after switch development exposure.",
            "A null factorial effect under weak activation or clipping is not evidence of no mechanism.",
            "No segmentation masks, other categories, confirmation data, or MVTec data were accessed.",
        ],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Budget = _output_budget_class(r1)
    budget = Budget(output, max_bytes=MAX_OUTPUT_BYTES,
                    failure_reserve_bytes=FAILURE_RESERVE_BYTES,
                    allowed_parent=ROOT / "outputs" / "calibration",
                    allow_existing=True)
    try:
        aggregate = _apply_coverage_aware_aggregation(aggregate)
        _write_json(budget, output / AGGREGATE_NAME, aggregate)
    finally:
        budget.close()
    return aggregate


def _experiment_worker(*, run_id: str, authorization_id: str,
                       output: Path, dataset_root: Path, json_root: Path,
                       bootstrap_iterations: int) -> dict[str, Any]:
    worker_started = time.monotonic()
    worker_cpu_started = time.process_time()
    stage_a = _stage_a_worker(
        run_id=run_id, authorization_id=authorization_id, output=output,
        dataset_root=dataset_root, json_root=json_root)
    aggregate = _stage_b_worker(
        output=output, dataset_root=dataset_root, json_root=json_root,
        bootstrap_iterations=bootstrap_iterations)
    from experiments.calibration import r1_matched_correction_pilot as r1
    Budget = _output_budget_class(r1)
    budget = Budget(output, max_bytes=MAX_OUTPUT_BYTES,
                    failure_reserve_bytes=FAILURE_RESERVE_BYTES,
                    allowed_parent=ROOT / "outputs" / "calibration",
                    allow_existing=True)
    try:
        from experiments.calibration.r1_matched_correction_pilot import sha256_file
        final = {
            "schema": "r2.exp2a.experiment_seal.v1",
            "status": "COMPLETE",
            "run_id": run_id,
            "authorization_id": authorization_id,
            "stage_a_status": "SEALED_COMPLETE",
            "stage_a_seal_sha256": stage_a["stage_a_seal_sha256"],
            "stage_b_status": "COMPLETE",
            "sanitized_aggregate_sha256": sha256_file(output / AGGREGATE_NAME),
            "development_verdict": aggregate["development_verdict"],
            "event_counts": {category: MAX_EVENTS for category in CATEGORIES},
            "total_event_count": len(CATEGORIES) * MAX_EVENTS,
            "total_query_count": sum(
                int(aggregate["outcome_metrics_by_category"][category][
                    "actual_query_count"]) for category in CATEGORIES),
            "total_dino_feature_encodings": len(CATEGORIES) * 528,
            "distinct_images_encoded": len(CATEGORIES) * (4 + MAX_EVENTS),
            "shared_encodings_across_factorial_arms": True,
            "automatic_restart": False,
            "wall_seconds_in_child": time.monotonic() - worker_started,
            "cpu_process_seconds_in_child": time.process_time() - worker_cpu_started,
            "output_bytes": budget._audit_tree(),
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_json(budget, output / "experiment_seal.json", final)
    finally:
        budget.close()
    return final


def _persist_supervisor_result(output: Path, r1: Any,
                               result: Mapping[str, Any] | None,
                               error: BaseException | None) -> None:
    Budget = _output_budget_class(r1)
    if not output.exists():
        budget = Budget(output, max_bytes=MAX_OUTPUT_BYTES,
                        failure_reserve_bytes=FAILURE_RESERVE_BYTES,
                        allowed_parent=ROOT / "outputs" / "calibration")
        budget.ensure_root()
    else:
        budget = Budget(output, max_bytes=MAX_OUTPUT_BYTES,
                        failure_reserve_bytes=FAILURE_RESERVE_BYTES,
                        allowed_parent=ROOT / "outputs" / "calibration",
                        allow_existing=True)
    try:
        status = {
            "schema": "r2.exp2a.supervisor_status.v1",
            "supervisor": "Windows Job Object, suspended assignment, kill-on-close",
            "timeout_seconds": MAX_RUNTIME_SECONDS,
            "runtime_profile_id": RUNTIME_PROFILE_ID,
            "automatic_restart": False,
            "output_max_bytes": MAX_OUTPUT_BYTES,
            "failure_reserve_bytes": FAILURE_RESERVE_BYTES,
            "os_hard_storage_isolation": False,
        }
        if result is not None:
            status.update(dict(result))
        if error is not None:
            status.update({
                "status": "SUPERVISOR_OR_CHILD_FAILURE",
                "failure_type": type(error).__name__,
                "failure_message": str(error)[:1600],
            })
            failure_path = output / "failure.json"
            if not failure_path.exists():
                _write_json(budget, failure_path, {
                    "schema": "r2.exp2a.failure.v1",
                    "status": "INCOMPLETE", "failure_type": type(error).__name__,
                    "failure_message": str(error)[:1600],
                    "automatic_restart": False,
                }, failure_reserved=True)
        status_path = output / "supervisor_status.json"
        if not status_path.exists():
            _write_json(budget, status_path, status, failure_reserved=True)
    finally:
        budget.close()


def _run_exp2a_supervised_child(command: Sequence[str], *, timeout_seconds: int,
                               env: Mapping[str, str], cwd: Path,
                               output_dir: Path | None = None):
    """Use the existing Job Object supervisor with the approved task-local ceiling.

    R1's generic helper has a 120-minute guard. This narrowly scoped override
    raises only that in-memory guard for this explicitly authorized EXP2A
    process; the Job Object implementation, kill-on-close behavior, wait logic,
    and R1 source file remain unchanged.
    """
    from unittest.mock import patch
    from experiments.calibration import r1_runtime_supervisor as supervisor
    if timeout_seconds != MAX_RUNTIME_SECONDS or timeout_seconds != 180 * 60:
        raise Exp2AIntegrityError("Unexpected EXP2A task-local runtime ceiling")
    if supervisor.MAX_RUNTIME_SECONDS != 120 * 60:
        raise Exp2AIntegrityError(
            "Historical R1 supervisor source/runtime contract changed")
    with patch.object(supervisor, "MAX_RUNTIME_SECONDS", timeout_seconds):
        return supervisor.run_supervised_child(
            command, timeout_seconds=timeout_seconds, env=env, cwd=cwd,
            output_dir=output_dir)


def supervise_experiment(*, expected_parent: str, run_id: str,
                         authorization_id: str, authorized: bool,
                         dataset_root: Path, json_root: Path,
                         bootstrap_iterations: int = 1000) -> dict[str, Any]:
    if authorized is not True:
        raise Exp2AIntegrityError("Explicit --authorize-real-execution is required")
    if authorization_id != AUTHORIZATION_ID:
        raise Exp2AIntegrityError("Authorization ID is not bound to this task")
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r1_runtime_supervisor as r1_supervisor
    r1._validate_run_id(run_id)
    if (r1_supervisor.MAX_RUNTIME_SECONDS != 120 * 60 or
            MAX_RUNTIME_SECONDS != 180 * 60 or
            RUNTIME_PROFILE_ID != "r2.exp2a.job_object.180m.researcher_override.v1"):
        raise Exp2AIntegrityError("Authorized EXP2A runtime profile changed")
    identity = _validate_parent_identity(expected_parent)
    _validate_interpreter()
    if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, text=True).strip():
        raise Exp2AIntegrityError(
            "Source-bound EXP2A refuses unrelated tracked modifications")
    output = ROOT / "outputs" / "calibration" / f"{OUTPUT_PREFIX}_{run_id}"
    if output.exists():
        raise FileExistsError(
            f"EXP2A output namespace exists; automatic restart is forbidden: {output}")
    free_bytes = r1._require_minimum_output_free_space(output.parent)
    if free_bytes < MIN_FREE_BYTES:
        raise Exp2AIntegrityError("Output volume fails the minimum free-space gate")
    module = Path(__file__).resolve()
    command = [
        str(EXPECTED_PYTHON), str(module), "--worker",
        "--run-id", run_id, "--authorization-id", authorization_id,
        "--output", str(output), "--dataset-root", str(dataset_root),
        "--json-root", str(json_root),
        "--bootstrap-iterations", str(int(bootstrap_iterations)),
    ]
    child_env = dict(os.environ)
    child_env.update({
        "R2_EXP2A_SUPERVISED_CHILD": "1",
        "R2_EXP2A_AUTHORIZATION_ID": authorization_id,
        "R2_EXP2A_SUPERVISOR_NONCE": os.urandom(32).hex(),
    })
    started = time.monotonic()
    result_obj = None
    try:
        supervised = _run_exp2a_supervised_child(
            command, timeout_seconds=MAX_RUNTIME_SECONDS, env=child_env,
            cwd=ROOT, output_dir=None)
        result_obj = asdict(supervised)
        result_obj["prelaunch_free_bytes"] = free_bytes
        result_obj["source_identity"] = identity
        _persist_supervisor_result(output, r1, result_obj, None)
        if supervised.status != "COMPLETED" or supervised.child_exit_code != 0:
            raise Exp2AError(
                f"Supervised EXP2A child failed: {supervised.status} "
                f"(exit={supervised.child_exit_code})")
        final_path = output / "experiment_seal.json"
        if not final_path.is_file():
            raise Exp2AIntegrityError("Supervised child exited without experiment seal")
        final = json.loads(final_path.read_text(encoding="utf-8"))
        if final.get("status") != "COMPLETE":
            raise Exp2AIntegrityError("Experiment seal does not certify completion")
        final["supervisor_elapsed_seconds"] = time.monotonic() - started
        final["supervisor_status"] = supervised.status
        return {"output": output, "final_seal": final,
                "supervisor": result_obj}
    except BaseException as exc:
        try:
            _persist_supervisor_result(output, r1, result_obj, exc)
        except Exception:
            pass
        raise


def _worker_cli(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    expected_parent = _validate_parent_identity()
    if (os.environ.get("R2_EXP2A_SUPERVISED_CHILD") != "1" or
            os.environ.get("R2_EXP2A_AUTHORIZATION_ID") != args.authorization_id):
        raise Exp2AIntegrityError("Worker mode is restricted to supervised invocation")
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job
    if not current_process_is_in_job():
        raise Exp2AIntegrityError("Worker process is outside the Windows Job Object")
    result = _experiment_worker(
        run_id=args.run_id, authorization_id=args.authorization_id,
        output=output, dataset_root=Path(args.dataset_root),
        json_root=Path(args.json_root),
        bootstrap_iterations=args.bootstrap_iterations)
    print(canonical_json(result), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-real-execution", action="store_true")
    parser.add_argument("--expected-parent", default=EXPECTED_PARENT)
    parser.add_argument("--run-id")
    parser.add_argument("--authorization-id", default=AUTHORIZATION_ID)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--json-root", type=Path, default=JSON_ROOT)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            if not args.output:
                raise Exp2AIntegrityError("Worker output directory is required")
            return _worker_cli(args)
        if not args.authorize_real_execution:
            raise Exp2AIntegrityError(
                "Real execution requires --authorize-real-execution")
        if args.bootstrap_iterations != 1000:
            raise Exp2AIntegrityError(
                "Source Group bootstrap is frozen to 1000 replicates")
        run_id = args.run_id or (
            time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "_" +
            uuid.uuid4().hex[:10])
        result = supervise_experiment(
            expected_parent=args.expected_parent, run_id=run_id,
            authorization_id=args.authorization_id, authorized=True,
            dataset_root=args.dataset_root, json_root=args.json_root,
            bootstrap_iterations=args.bootstrap_iterations)
        print(canonical_json({
            "output": str(result["output"]),
            "final_seal": result["final_seal"],
            "supervisor": result["supervisor"],
        }), flush=True)
        return 0
    except BaseException as exc:
        print(f"R2_EXP2A_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


def _display_number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "undefined"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "undefined"
    return f"{number:.{digits}f}" if math.isfinite(number) else "undefined"


def write_integrated_deliverables(output: Path, report_dir: Path) -> tuple[Path, Path]:
    """Copy only sanitized category aggregates into the tracked audit directory."""
    aggregate = json.loads((output / AGGREGATE_NAME).read_text(encoding="utf-8"))
    if (aggregate.get("schema") != "r2.exp2a.sanitized_aggregate.v1" or
            aggregate.get("status") != "COMPLETE"):
        raise Exp2AIntegrityError("EXP2A sanitized aggregate is incomplete")
    source_aggregate_sha256 = sha256_file(output / AGGREGATE_NAME)
    aggregate = _apply_coverage_aware_aggregation(aggregate)
    aggregate["coverage_aware_aggregation"][
        "source_sanitized_aggregate_sha256"] = source_aggregate_sha256
    initial_supervisor = json.loads(
        (output / "supervisor_status.json").read_text(encoding="utf-8"))
    stage_b_supervisor = json.loads(
        (output / "stage_b_supervisor_status.json").read_text(encoding="utf-8"))
    stage_a_seal = json.loads(
        (output / "stage_a_seal.json").read_text(encoding="utf-8"))
    if (stage_b_supervisor.get("status") != "COMPLETED" or
            stage_b_supervisor.get("child_exit_code") != 0 or
            not stage_b_supervisor.get("termination_confirmed")):
        raise Exp2AIntegrityError("Stage B continuation lacks confirmed Job Object completion")
    stage_a_runtime = stage_a_seal["metadata"]["category_runtime"]
    stage_a_seconds = float(stage_a_seal["metadata"]["total_stage_a_seconds"])
    peak_cuda_allocated = max(
        int(item["peak_cuda_allocated_bytes"]) for item in stage_a_runtime.values())
    peak_cuda_reserved = max(
        int(item["peak_cuda_reserved_bytes"]) for item in stage_a_runtime.values())
    run_output_bytes = sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file())
    report_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = report_dir / AGGREGATE_NAME
    report_path = report_dir / "R2_EXP2A_C1_STRENGTH_ATTRIBUTION.md"
    if aggregate_path.exists() or report_path.exists():
        raise FileExistsError("EXP2A report directory already contains deliverables")
    aggregate_path.write_text(canonical_json(aggregate) + "\n", encoding="utf-8")

    by_category = aggregate["outcome_metrics_by_category"]
    macro = aggregate["category_equal_aggregation"]
    arms = ("c0", "c1_full", "c1_no_outcome_weight", "c1_no_margin",
            "c1_no_both", "c2_a002", "c2_a020")
    headers = ("Category", "C0", "C1-Full", "No outcome weight",
               "No margin", "No both", "C2 0.02", "C2 0.20")
    lines = [
        "# R2-EXP2A — Causal Attribution of C1 Correction Strength",
        "",
        f'**Status:** {aggregate["development_verdict"]}',
        f'**Coverage:** four historically exposed development categories, '
        f'{MAX_EVENTS} events per category ({MAX_EVENTS * len(CATEGORIES)} total).',
        "**Independent confirmation:** not provided.",
        "",
        "## Protocol and provenance",
        "",
    ]
    provenance = aggregate["input_provenance"]
    lines.extend([
        f'- Branch and parent commit: {provenance["parent_branch"]} at '
        f'{provenance["parent_commit"]}.',
        f'- Stage 05 frozen inputs SHA-256: {provenance["stage05_frozen_inputs_sha256"]}.',
        f'- Q-A-TS binding and verification SHA-256: '
        f'{provenance["qats_binding_sha256"]} and {provenance["qats_verification_sha256"]}.',
        f'- DINO checkpoint SHA-256: {provenance["dino_checkpoint_sha256"]}.',
        '- Support regime: 4 deployed seed-0 supports plus 64 additional C64 '
        'training normals; C64 score tapes were reused and not recomputed.',
        '- One shared native feature capture per category fed C0, four independent '
        'C1 Codebooks, and C2(a=0.02/0.20). Every query image was encoded once; '
        'each of four support images contributed four frozen augmentation views.',
        '- DINO feature encodings across all categories: '
        f'{sum(int(value["shared_dino_feature_encodings"]) for value in aggregate["stage_a_resource_use"].values())}. '
        'No per-arm feature extraction was performed.',
        f'- Stage A GPU: {stage_a_seal["metadata"]["model_profiles"][CATEGORIES[0]]["runtime_profile"]["cuda_device_name"]}; '
        f'{stage_a_seconds:.3f} seconds. Peak CUDA allocated/reserved: '
        f'{peak_cuda_allocated}/{peak_cuda_reserved} bytes.',
        f'- Completed Stage B CPU child: {stage_b_supervisor["elapsed_seconds"]:.3f} seconds; '
        f'combined Stage A plus successful Stage B child time: '
        f'{stage_a_seconds + float(stage_b_supervisor["elapsed_seconds"]):.3f} seconds. '
        'The initial combined child failure is recorded separately below.',
        f'- Run output size at report generation: {run_output_bytes} bytes of '
        f'{MAX_OUTPUT_BYTES}; prelaunch free space was '
        f'{initial_supervisor.get("prelaunch_free_bytes", "unrecorded")} bytes.',
        '- Query labels were disclosed online only at frozen Q-A-TS positions after '
        'all seven current-event scores were sealed. Stage B read the first 512 '
        'matching outcomes only after the complete Stage A seal. Masks were not opened.',
        '- C2(a=0.02) was selected after switch development exposure. C2(a=0.20) '
        'is an optional historical engineering reference.',
        f'- Initial combined child: {initial_supervisor["status"]}, exit '
        f'{initial_supervisor["child_exit_code"]}, after a complete Stage A seal. '
        'It stopped in the Stage B summary because the all-normal audiojack horizon '
        'has an undefined ranking result; the Q6 aggregator assumed that key existed.',
        f'- Repaired Stage B continuation: {stage_b_supervisor["status"]}, '
        f'exit {stage_b_supervisor["child_exit_code"]}, '
        f'{stage_b_supervisor["elapsed_seconds"]:.3f} seconds in a Windows Job Object. '
        'Stage A scores and failure records were preserved; Stage A was not rerun.',
        '- The coverage-aware report keeps the strict four-category macro undefined '
        'where audiojack has no defects, and separately reports descriptive estimates '
        'over the three categories with both classes.',
        '',
        "## C1-Full switch reference parity",
        "",
    ])
    parity = aggregate["c1_full_reference_parity"]
    lines.extend([
        f'- Result: {parity["status"]}; {parity["event_checks"]["events"]} events and '
        f'{parity["event_checks"]["query_events"]} query events checked against R1-S4B.',
        '- Checks included exact C0 maps and Top-32 indices, C1-Full dense-map deltas, '
        'C1 score and state hashes, native update counts, common C0 memory state, query '
        'decisions, queried labels, and C2(0.20) deltas.',
        '- Historical switch C1 AUROC was 0.843083; the recomputed value is '
        f'{_display_number(by_category["switch"]["auroc_by_arm"]["c1_full"])}.',
        '',
        "## Image AUROC",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ])
    for category in CATEGORIES:
        values = by_category[category]["auroc_by_arm"]
        lines.append("| " + " | ".join(
            [category] + [_display_number(values.get(arm)) for arm in arms]) + " |")
    macro_auroc = macro["auroc_by_arm"]
    lines.append("| Category-equal macro | " + " | ".join(
        _display_number(macro_auroc[arm]["category_equal_macro"])
        for arm in arms) + " |")
    lines.append("| Defined-category macro ("
                 f'{macro_auroc["c0"]["defined_category_count"]}/'
                 f'{len(CATEGORIES)}; descriptive) | ' + " | ".join(
        _display_number(macro_auroc[arm]["category_equal_macro_defined"])
        for arm in arms) + " |")

    contrasts = (
        "outcome_effect_margin_enabled", "outcome_effect_margin_disabled",
        "margin_effect_outcome_enabled", "margin_effect_outcome_disabled",
        "interaction")
    lines.extend([
        "",
        "## Factorial contrasts",
        "",
        "Index 1 is margin adaptation; index 2 is outcome-strength weighting. "
        "F11 is C1-Full, F10 removes outcome weights, F01 fixes base beta to 0.02, "
        "and F00 removes both components.",
        "",
        "| Contrast | Strict 4-category mean | Defined-category mean | Defined n | Positive / negative | Defined range |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    macro_effects = macro["factorial_effects"]
    for contrast in contrasts:
        value = macro_effects[contrast]
        lines.append(
            f'| {contrast} | {_display_number(value["category_equal_macro"])} | '
            f'{_display_number(value["category_equal_macro_defined"])} | '
            f'{value["defined_category_count"]}/{len(CATEGORIES)} | '
            f'{value["positive_category_count"]} / {value["negative_category_count"]} | '
            f'{_display_number(value["category_range_defined"])} |')

    lines.extend([
        "",
        "## Operating metrics by category and arm",
        "",
        "| Category | Arm | AUROC | AUPRC | Prevalence | FPR | FNR | Balanced accuracy | TP/TN/FP/FN | Queried normals/defects |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ])
    for category in CATEGORIES:
        entry = by_category[category]
        for arm in SCORE_ARMS:
            metric = entry["metrics_by_arm"][arm]
            threshold = metric["fixed_threshold"]
            counts = (f'{threshold["tp"]}/{threshold["tn"]}/'
                      f'{threshold["fp"]}/{threshold["fn"]}')
            lines.append(
                f'| {category} | {arm} | '
                f'{_display_number(metric["image_auroc"])} | '
                f'{_display_number(metric["image_auprc_average_precision"])} | '
                f'{_display_number(metric["defect_prevalence"], 4)} | '
                f'{_display_number(threshold["fpr"], 4)} | '
                f'{_display_number(threshold["fnr"], 4)} | '
                f'{_display_number(threshold["balanced_accuracy"], 4)} | {counts} | '
                f'{entry["queried_normal_count"]}/{entry["queried_defect_count"]} |')
    lines.extend([
        "",
        "## Mechanism, ranking, and uncertainty",
        "",
        "The sanitized JSON aggregate contains per-category Source Group cluster "
        "bootstrap intervals, beta distributions before and after each original "
        "clipping stage, patch multipliers, dN admission diagnostics, codebook "
        "insert/merge/replace/skip counts, occupancy by time, anchor signs, "
        "source-group exclusions, ablation activation, state and score divergence, "
        "prediction flips, false-positive/false-negative repairs and harm, and "
        "cross-source-group normal–defect pair repair/damage counts.",
        "",
        "The four categories are four historically exposed development observations, "
        "not independent confirmation. Bootstrap intervals condition on each fixed "
        "online trajectory and do not quantify regenerated-stream uncertainty. A null "
        "contrast under weak activation or heavy clipping is not evidence that the "
        "mechanism is unnecessary.",
        "",
        'audiojack contains 512 normal and 0 defect outcomes in the authorized horizon. '
        'Its AUROC, AUPRC, FNR, balanced accuracy, factorial effects, and cross-group '
        'normal-defect ranking diagnostics are undefined for that denominator; no '
        'replacement value is imputed.',
        "",
        "## Scientific interpretation",
        "",
        f'- Q1 margin effect with outcome weights enabled: '
        f'{_display_number(macro_effects["margin_effect_outcome_enabled"]["category_equal_macro_defined"])} '
        f'across {macro_effects["margin_effect_outcome_enabled"]["defined_category_count"]}/'
        f'{len(CATEGORIES)} evaluable categories; the strict four-category macro is '
        f'{_display_number(macro_effects["margin_effect_outcome_enabled"]["category_equal_macro"])}.',
        f'- Q2 outcome-weight effect with margin enabled: '
        f'{_display_number(macro_effects["outcome_effect_margin_enabled"]["category_equal_macro_defined"])} '
        f'across {macro_effects["outcome_effect_margin_enabled"]["defined_category_count"]}/'
        f'{len(CATEGORIES)} evaluable categories.',
        f'- Q3 factorial interaction: '
        f'{_display_number(macro_effects["interaction"]["category_equal_macro_defined"])} '
        f'across {macro_effects["interaction"]["defined_category_count"]}/'
        f'{len(CATEGORIES)} evaluable categories.',
        f'- Q5 C1-Full minus C2(a=0.02), defined-category mean: '
        f'{_display_number(macro_auroc["c1_full_minus_c2_a002"]["category_equal_macro_defined"])} '
        f'over {macro_auroc["c1_full_minus_c2_a002"]["defined_category_count"]}/'
        f'{len(CATEGORIES)} categories; positive/negative '
        f'{macro_auroc["c1_full_minus_c2_a002"]["positive_category_count"]}/'
        f'{macro_auroc["c1_full_minus_c2_a002"]["negative_category_count"]}.',
        f'- Q7: {aggregate["scientific_questions"]["q7_strength_policy_central_innovation_claim"]}.',
        '- Experiment 2B remains separate and was not run. These results do not '
        'authorize Gaussian-protocol changes or further evaluation.',
        "",
        "## Resource controls and reproducibility",
        "",
        f'- Windows Job Object hard deadline: {MAX_RUNTIME_SECONDS // 60} minutes; '
        f'output budget: {MAX_OUTPUT_BYTES} bytes with {FAILURE_RESERVE_BYTES} bytes '
        'reserved for failure metadata; minimum prelaunch free space: 10 GiB.',
        '- The output budget is application level; OS-level storage isolation is not provided.',
        f'- Stage A runner SHA-256: {provenance["stage_a_runner_source_sha256"]}.',
        f'- Stage B metrics runner SHA-256: {provenance["stage_b_runner_source_sha256"]}.',
        f'- Coverage-aware reporting source SHA-256: '
        f'{aggregate["coverage_aware_aggregation"]["source_sha256"]}.',
        f'- Source hash manifest: {canonical_json(provenance["source_hashes"])}.',
        f'- Machine-readable sanitized aggregate: {aggregate_path.as_posix()}.',
        '- Focused CPU regression: 21 passed; Python compilation passed. Tests include '
        'coverage with an undefined single-class category and defined/undefined '
        'cross-group ranking summaries.',
        '- CPU synthetic tests cover the original v22 update path, beta clipping and '
        'patch weighting, independent Codebooks and sign-aware retention, source-group '
        'exclusion, causal query timing, dense-map maxima, reference parity, category '
        'aggregation, and incomplete Stage B gating.',
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path, aggregate_path
