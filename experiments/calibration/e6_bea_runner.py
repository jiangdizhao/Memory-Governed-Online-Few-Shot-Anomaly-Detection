"""Explicitly gated E6-BEA-v2 Stage A entrypoint.

Importing this module and running it without the complete authorization
argument set is inert: it does not load dataset metadata, model weights, or
CUDA. The Stage B evaluator is deliberately not called here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.calibration.e6_bea_closed_loop import (
    ARM_IDS,
    E6_REGISTERED_CATEGORIES,
    E6ClosedLoopError,
    ArmRuntime,
    NormalCoverageMemory,
    OnDemandLabelReader,
    PrivateFeedbackSink,
    R7FrozenFeatureEncoder,
    ScoreLedger,
    StageAClosedLoop,
    MAX_SMOKE_TEST_EVENTS,
    _assert_module_source,
    _canonical_json,
    _reject_reparse_ancestors,
    _sha256_file,
    build_r7_native_arm_engines,
    create_e6_output_namespace,
    make_e6_event_stream,
    verify_e6_execution_identity,
    validate_smoke_event_prefix,
    validate_test_image_prefix_projection,
)
from experiments.calibration.e6_bea_policy import QueryController


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E6ClosedLoopError(f"frozen E6 metadata is missing or unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise E6ClosedLoopError(f"frozen E6 metadata has an invalid root object: {path.name}")
    return value


def _group_sha256(value: str) -> str:
    return hashlib.sha256(("official-sample-group\0" + str(value)).encode("utf-8")).hexdigest()


def load_e6_stage05_category(category: str, *, repository_root: Path,
                             verify_image_paths: bool = True,
                             smoke_prefix_events: int | None = None) -> dict[str, Any]:
    """Adapt frozen Stage 0.5 identities without opening or consuming its query tape."""
    if category not in E6_REGISTERED_CATEGORIES:
        raise E6ClosedLoopError("E6 category is outside the registered development set")
    if smoke_prefix_events is not None and (
            category != "audiojack" or int(smoke_prefix_events) != MAX_SMOKE_TEST_EVENTS):
        raise E6ClosedLoopError("bounded smoke loader accepts audiojack and exactly 20 prefix events")
    root = Path(repository_root).resolve(strict=True)
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration import stage05

    _assert_module_source(stage05, "experiments/calibration/stage05.py",
                          root=root, role="Stage 0.5 identity/path helper")
    stage05_dir = Path(r3.STAGE05).resolve(strict=True)
    run_path = stage05_dir / "run_manifest.json"
    frozen_path = stage05_dir / "frozen_inputs.json"
    scores_path = stage05_dir / "score_tape_manifest.json"
    for path in (run_path, frozen_path, scores_path):
        _reject_reparse_ancestors(path)
    run, frozen, score_manifest = (_read_json(run_path), _read_json(frozen_path),
                                   _read_json(scores_path))
    if (run.get("overall_status") != "EXPLORATORY_AUDIT_COMPLETE_NOT_STAGE1_VALIDATION" or
            run.get("confirmation_classes_opened") is not False or
            frozen.get("schema") != "stage05.frozen_inputs.v1" or
            frozen.get("freeze_verification_status") != "PASS" or
            frozen.get("stage1_or_stage2_run") is not False or
            score_manifest.get("status") != "PASS" or
            score_manifest.get("controller_or_test_label_access") is not False):
        raise E6ClosedLoopError("frozen Stage 0.5 metadata is incompatible with E6")
    # Verify that the frozen record names its query archive, but never open that
    # archive, construct a tape, or read any fixed query position.
    if (run.get("query_tape_archive_sha256") != frozen.get("query_tapes_file_sha256") or
            not isinstance(run.get("query_tape_archive_sha256"), str)):
        raise E6ClosedLoopError("frozen Stage 0.5 query archive identity receipt is inconsistent")
    category_set = set(frozen.get("classes", {}))
    if category_set != set(E6_REGISTERED_CATEGORIES):
        raise E6ClosedLoopError("frozen Stage 0.5 category inventory differs from E6 registration")

    frozen_info = frozen["classes"][category]
    raw_rows = frozen_info.get("stream")
    support_receipt = frozen_info.get("supports", {}).get("0", {})
    c64_receipt = frozen_info.get("c64", {})
    expected_events = int(r7.EXPECTED_COUNTS[category])
    if (not isinstance(raw_rows, list) or len(raw_rows) != expected_events or
            int(frozen_info.get("stream_images", -1)) != expected_events or
            frozen_info.get("support_c64_disjoint") is not True or
            frozen_info.get("train_test_groups_disjoint") is not True or
            len(support_receipt.get("images", ())) != 4 or
            int(c64_receipt.get("count", -1)) != 64):
        raise E6ClosedLoopError(f"{category}: frozen Support/C64/TEST role receipt is invalid")

    category_json = Path(r3.JSON_ROOT) / f"{category}.json"
    category_json_sha = _sha256_file(category_json)
    expected_json_sha = str(frozen_info.get("class_json_sha256", ""))
    if category_json_sha != expected_json_sha:
        raise E6ClosedLoopError(f"{category}: frozen TEST JSON source hash changed")

    stream_rows: list[dict[str, Any]] = []
    for position, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping) or int(raw.get("position", -1)) != position:
            raise E6ClosedLoopError(f"{category}: frozen TEST identity rows are not contiguous")
        relative_id = str(raw.get("relative_id", "")).replace("\\", "/")
        opaque_group = str(raw.get("opaque_group", ""))
        if len(opaque_group) != 64:
            raise E6ClosedLoopError(f"{category}: frozen TEST Source Group digest is malformed")
        image_path = stage05._absolute_image_path(r3.DATASET_ROOT,
                                                  str(frozen_info["prefix"]), relative_id)
        if (verify_image_paths and smoke_prefix_events is None and
                not Path(image_path).is_file()):
            raise E6ClosedLoopError(f"{category}:{position}: frozen TEST image path is missing")
        stream_rows.append({**dict(raw), "relative_id": relative_id,
                            "image_path": str(image_path)})
    full_events = make_e6_event_stream(stream_rows)
    if smoke_prefix_events is None:
        events = full_events
        r7.validate_image_projection(category_json, category, events)
    else:
        events = validate_smoke_event_prefix(full_events[:MAX_SMOKE_TEST_EVENTS])
        validate_test_image_prefix_projection(
            category_json, category, events, full_event_count=len(full_events))
        if verify_image_paths and any(not Path(event.image_path).is_file() for event in events):
            raise E6ClosedLoopError("a selected prefix TEST image is missing")

    support_paths = [str(stage05._absolute_image_path(
        r3.DATASET_ROOT, str(frozen_info["prefix"]), str(relative).replace("\\", "/")))
        for relative in support_receipt["images"]]
    c64_paths = [Path(stage05._absolute_image_path(
        r3.DATASET_ROOT, str(frozen_info["prefix"]), str(relative).replace("\\", "/")))
        for relative in c64_receipt.get("images", ())]
    if (len(support_paths) != 4 or len(c64_paths) != 64 or
            (verify_image_paths and any(not Path(path).is_file() for path in support_paths)) or
            (verify_image_paths and any(not path.is_file() for path in c64_paths))):
        raise E6ClosedLoopError(f"{category}: Support/C64 image path inventory is incomplete")

    raw_c64_groups = [str(value) for value in c64_receipt.get("groups", ())]
    raw_support_groups = [str(value) for value in support_receipt.get("groups", ())]
    c64_groups = [_group_sha256(value) for value in raw_c64_groups]
    support_groups = sorted({_group_sha256(value) for value in raw_support_groups})
    test_groups = {str(row["opaque_group"]) for row in stream_rows}
    if (len(support_groups) != 4 or len(c64_groups) != 64 or len(set(c64_groups)) != 64 or
            set(support_groups) & set(c64_groups) or set(support_groups) & test_groups or
            set(c64_groups) & test_groups or
            stage05.opaque_digest("support-groups", raw_support_groups) !=
            support_receipt.get("group_sha256") or
            stage05.opaque_digest("c64-groups", raw_c64_groups) !=
            c64_receipt.get("group_sha256") or
            c64_receipt.get("group_sha256") !=
            score_manifest.get("classes", {}).get(category, {}).get("score_tapes", {})
            .get("0", {}).get("c64_group_sha256", c64_receipt.get("group_sha256"))):
        raise E6ClosedLoopError(f"{category}: Support/C64/TEST Source Group identities overlap or differ")

    score_row = score_manifest.get("classes", {}).get(category, {}).get(
        "score_tapes", {}).get("0", {})
    c64_path = (stage05_dir / str(score_row.get("file", ""))).resolve(strict=True)
    if (stage05_dir not in c64_path.parents or
            _sha256_file(c64_path) != score_row.get("sha256") or
            score_row.get("c64_group_sha256") != c64_receipt.get("group_sha256")):
        raise E6ClosedLoopError(f"{category}: C64 score archive path or digest is invalid")
    with np.load(c64_path, allow_pickle=False) as archive:
        c64_scores = np.asarray(archive["c64_scores"], dtype=np.float64)
        c64_identity = str(archive["c64_identity_sha256"].item())
    if (c64_scores.shape != (64,) or not np.isfinite(c64_scores).all() or
            c64_identity != c64_receipt.get("image_sha256")):
        raise E6ClosedLoopError(f"{category}: C64 score values do not match frozen image identity")

    contract = {
        "category_json_sha256": category_json_sha,
        "stream_rows": stream_rows,
        "support_paths": support_paths,
        "support_group_sha256": r3.sha256_json(support_groups),
        "support_image_sha256": support_receipt.get("image_sha256"),
        "c64_group_sha256": c64_receipt.get("group_sha256"),
    }
    return {
        "events": events,
        "contract": contract,
        "frozen_info": frozen_info,
        "json_path": category_json,
        "support_paths": support_paths,
        "support_image_sha256": support_receipt.get("image_sha256"),
        "support_group_sha256": contract["support_group_sha256"],
        "c64_paths": c64_paths,
        "c64_groups": c64_groups,
        "c64_scores": c64_scores.tolist(),
        "c64_score_archive_sha256": str(score_row["sha256"]),
        "fixed_support_sha256_expected": str(score_row["fixed_support_sha256"]),
        "theta_c64_q90": float(np.quantile(c64_scores, 0.90, method="higher")),
        "sigma_c64_floor_002": max(float(np.std(c64_scores, ddof=1)), 0.02),
        "stage05_receipts": {
            "run_manifest_sha256": _sha256_file(run_path),
            "frozen_inputs_sha256": _sha256_file(frozen_path),
            "score_tape_manifest_sha256": _sha256_file(scores_path),
        },
    }


def _support_cls_features(impl: Any, backbone: Any, support_paths: list[str],
                          feature_fuse: str) -> np.ndarray:
    rows = []
    for path in support_paths:
        image = impl.load_rgb(path)
        _features, cls = impl.encode_pil(backbone, image, feature_fuse)
        rows.append(np.asarray(cls.detach().float().reshape(-1).cpu(), dtype=np.float32))
    if len(rows) != 4 or len({row.shape for row in rows}) != 1:
        raise E6ClosedLoopError("E6 Coverage Memory requires four compatible support CLS vectors")
    return np.stack(rows)


def _support_features_batched(impl: Any, p0: Any, backbone: Any,
                              support_paths: list[str], cfg: Any
                              ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Batch each Support's frozen augmentation refs into one backbone forward.

    The same registered P0 transforms, deterministic per-image seeds, fused
    patch features, and fp16 CPU cache are retained. Native fixed-bank digest
    verification remains the final fail-closed parity gate.
    """
    import torch

    if len(support_paths) != 4:
        raise E6ClosedLoopError("bounded smoke requires exactly four Support images")
    augmentation = impl.SupportAugConfig()
    cache: dict[str, Any] = {}
    forward_counts: list[int] = []
    refs_per_support: list[int] = []
    for path in support_paths:
        image_seed = int(p0.deterministic_subseed(cfg.augmentation_seed, path))
        random.seed(image_seed)
        np.random.seed(image_seed % (2**32 - 1))
        torch.manual_seed(image_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(image_seed)
        image = impl.load_rgb(path)
        refs = impl.make_support_refs(
            image, backbone, augmentation, cfg.augmentations,
            baseline_compat=False, include_identity=cfg.include_identity)
        if not refs:
            raise E6ClosedLoopError("native Support augmentation produced no reference views")
        # One model forward per Support, with all registered refs in its batch.
        layer_batches, _global_batch = backbone.encode_batch(refs)
        entries = []
        for ref_index in range(len(refs)):
            per_layer = [layer[ref_index].detach() for layer in layer_batches]
            fused = impl.fuse_patch_features(per_layer, cfg.feature_fuse)
            entries.append([
                impl.l2_normalize(t).detach().to("cpu", dtype=torch.float16)
                for t in fused
            ])
        cache[path] = entries
        forward_counts.append(1)
        refs_per_support.append(len(refs))
    receipt = {
        "support_images": 4,
        "support_batch_forwards": int(sum(forward_counts)),
        "support_refs_per_image": refs_per_support,
        "support_augmentation_view_inputs": int(sum(refs_per_support)),
        "support_cls_forwards": 4,
        "support_cls_image_inputs": 4,
        "support_fresh_backbone_invocations": int(sum(forward_counts) + 4),
        "support_invocation_limit": 8,
        "batching_contract": "registered P0 augmentation refs batched per Support; original native bank digest must match",
    }
    if receipt["support_fresh_backbone_invocations"] > receipt["support_invocation_limit"]:
        raise E6ClosedLoopError("Support encoding budget exceeds eight fresh backbone forwards")
    return cache, receipt


def _write_failure(run_dir: Path, stage: str, error: BaseException) -> None:
    failure = Path(run_dir) / "failure.json"
    if failure.exists():
        return
    payload = {
        "schema": "e6.bea_v2.failure.v1", "status": "INCOMPLETE",
        "stage": str(stage), "failure_type": type(error).__name__,
        "message": str(error)[:2000], "automatic_restart": False,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        with failure.open("xb") as stream:
            stream.write(_canonical_json(payload))
            stream.flush()
    except OSError:
        pass


def execute_real_stage_a(*, category: str, run_id: str,
                         approved_commit: str | None = None,
                         human_authorized: bool = False,
                         authorization_id: str | None = None,
                         repository_root: Path | None = None) -> Any:
    """Execute a future real Stage A only after separate human and exact-HEAD gates."""
    root = Path(repository_root or Path(__file__).resolve().parents[2]).resolve(strict=True)
    if human_authorized is not True or not authorization_id:
        raise E6ClosedLoopError("real E6 Stage A requires separate human authorization and an authorization ID")
    identity = verify_e6_execution_identity(str(approved_commit or ""), root=root)
    info = load_e6_stage05_category(category, repository_root=root)

    import importlib
    import torch
    from types import SimpleNamespace
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r2_exp2a_c1_strength as exp2a
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration import r6_capacity1024_audiojack_runner as r6
    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration import native
    from experiments.calibration import profiles

    impl = p0.require_sources()
    visionad = importlib.import_module("vmb_visionad_new")
    backbone, model_receipt, runtime_info = __import__(
        "experiments.calibration.e6_bea_closed_loop", fromlist=["load_r7_frozen_backbone"]
    ).load_r7_frozen_backbone(
        impl=impl, visionad=visionad, r1=r1, r3=r3,
        approved_e6_commit=identity["head"], repository_root=root)
    cfg, native_profile = r3._native_config(p0, profiles, r1)
    support_features = p0.encode_reference_features(
        impl, backbone, list(info["support_paths"]), cfg)
    support_cls = _support_cls_features(
        impl, backbone, list(info["support_paths"]), cfg.feature_fuse)
    c64_items, _cache_receipt = r7._make_c64_items(
        category, info, p0, impl, backbone, cfg)
    engines = build_r7_native_arm_engines(
        category=category, info=info, impl=impl, visionad=visionad,
        p0=p0, r1=r1, r3=r3, r6=r6, r7=r7, exp2a=exp2a,
        native_module=native, cfg=cfg, support_features=support_features,
        support_paths=info["support_paths"], c64_items=c64_items,
        approved_e6_commit=identity["head"], repository_root=root)
    events = tuple(info["events"])
    run_dir = create_e6_output_namespace(run_id, repository_root=root)
    ledger = None
    readers: list[OnDemandLabelReader] = []
    sinks: list[PrivateFeedbackSink] = []
    try:
        ledger = ScoreLedger(run_dir / "sealed_scores.jsonl")
        arms: dict[str, ArmRuntime] = {}
        for arm_id in ARM_IDS:
            reader = OnDemandLabelReader(
                info["json_path"], category, info["contract"]["category_json_sha256"],
                events, arm_id, ledger)
            readers.append(reader)
            sink = PrivateFeedbackSink(run_dir / f"private_feedback_{arm_id}.jsonl", arm_id)
            sinks.append(sink)
            arms[arm_id] = ArmRuntime(
                arm_id=arm_id, engine=engines[arm_id],
                controller=QueryController(arm_id, seed=37),
                coverage=NormalCoverageMemory(support_cls),
                label_reader=reader, private_feedback=sink)
        encoder = R7FrozenFeatureEncoder(
            impl=impl, visionad=visionad, repository_root=root, backbone=backbone,
            feature_fuse=cfg.feature_fuse, r1=r1,
            expected_model_state_sha256=str(model_receipt["loaded_model_state_sha256"]),
            runtime_info=runtime_info)
        runner = StageAClosedLoop(
            category=category, events=events, feature_encoder=encoder,
            arms=arms, ledger=ledger,
            execution_authorization_id=authorization_id)
        seal = runner.run()
        if not seal.complete:
            raise E6ClosedLoopError("E6 Stage A returned without a complete durable seal")
        return seal
    except BaseException as exc:
        if ledger is not None:
            ledger.close()
        for reader in readers:
            reader.close()
        for sink in sinks:
            sink.close()
        _write_failure(run_dir, "stage_a", exc)
        raise


def execute_bounded_smoke_stage_a(*, run_id: str, approved_commit: str,
                                  authorization_id: str, repository_root: Path,
                                  output_budget: Any) -> Any:
    """Run exactly one bounded smoke through the shared E6 StageA coordinator."""
    root = Path(repository_root).resolve(strict=True)
    if not authorization_id or not approved_commit:
        raise E6ClosedLoopError("bounded E6 smoke requires separate explicit authorization")
    identity = verify_e6_execution_identity(approved_commit, root=root)
    from experiments.calibration.e6_bea_smoke import _verify_pinned_supervisor_source
    supervisor_sha = _verify_pinned_supervisor_source(root)
    info = load_e6_stage05_category(
        "audiojack", repository_root=root, smoke_prefix_events=MAX_SMOKE_TEST_EVENTS)
    run_dir = Path(output_budget.root).resolve(strict=True)
    if (run_dir.parent != (root / "outputs" / "calibration").resolve(strict=True) or
            run_dir.name != f"e6_gpu_smoke_{run_id}"):
        raise E6ClosedLoopError("bounded smoke OutputBudget is not bound to its unique output namespace")

    import importlib
    import torch
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration import r6_capacity1024_audiojack_runner as r6
    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration import r2_exp2a_c1_strength as exp2a
    from experiments.calibration import native
    from experiments.calibration import profiles

    impl = p0.require_sources()
    visionad = importlib.import_module("vmb_visionad_new")
    properties = torch.cuda.get_device_properties(0)
    total_gpu_bytes = int(properties.total_memory)
    memory_fraction = min(1.0, (22 * 1024**3) / max(1, total_gpu_bytes))
    torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
    backbone, model_receipt, runtime_info = __import__(
        "experiments.calibration.e6_bea_closed_loop", fromlist=["load_r7_frozen_backbone"]
    ).load_r7_frozen_backbone(
        impl=impl, visionad=visionad, r1=r1, r3=r3,
        approved_e6_commit=identity["head"], repository_root=root)
    backbone.model.eval()
    for parameter in backbone.model.parameters():
        parameter.requires_grad_(False)
    if backbone.model.training or any(parameter.requires_grad for parameter in backbone.model.parameters()):
        raise E6ClosedLoopError("bounded smoke requires a frozen evaluation-mode DINO model")

    cfg, _native_profile = r3._native_config(p0, profiles, r1)
    support_features, support_encoding_receipt = _support_features_batched(
        impl, p0, backbone, list(info["support_paths"]), cfg)
    support_cls = _support_cls_features(
        impl, backbone, list(info["support_paths"]), cfg.feature_fuse)
    c64_items, c64_cache_receipt = r7._make_c64_items(
        "audiojack", info, p0, impl, backbone, cfg)
    c64_encodings = int((c64_cache_receipt or {}).get("new_c64_dino_encodings", 0))
    if (int((c64_cache_receipt or {}).get("count", 0)) != 64 or
            not 0 <= c64_encodings <= 64):
        raise E6ClosedLoopError("bounded smoke C64 receipt violates the 64-encoding limit")
    engines = build_r7_native_arm_engines(
        category="audiojack", info=info, impl=impl, visionad=visionad,
        p0=p0, r1=r1, r3=r3, r6=r6, r7=r7, exp2a=exp2a,
        native_module=native, cfg=cfg, support_features=support_features,
        support_paths=info["support_paths"], c64_items=c64_items,
        approved_e6_commit=identity["head"], repository_root=root)
    events = tuple(info["events"])
    from experiments.calibration.e6_bea_closed_loop import _canonical_json
    stream_identities = [{
        "position": event.position,
        "original_json_position": event.original_position,
        "event_id_sha256": event.event_id_sha256,
        "image_identity_sha256": event.image_identity_sha256,
        "source_group_sha256": event.source_group_sha256,
        "camera_id": event.camera_id,
    } for event in events]
    prefix_sha256 = hashlib.sha256(_canonical_json(stream_identities)).hexdigest()
    q0_engine = engines["Q0"]
    manifest = {
        "schema": "e6.bea_v2.smoke_manifest.v1",
        "artifact_kind": "bounded_engineering_smoke",
        "status_at_creation": "INCOMPLETE",
        "run_id": run_id,
        "category": "audiojack",
        "event_count": MAX_SMOKE_TEST_EVENTS,
        "stream_prefix_sha256": prefix_sha256,
        "stream_source": "R7 deterministic group shuffle, seed 0, first four complete Source Groups",
        "category_json_sha256": info["contract"]["category_json_sha256"],
        "stage05_receipts": dict(info["stage05_receipts"]),
        "support_and_c64_identity": dict(q0_engine.input_identity),
        "c64_cache_receipt": dict(c64_cache_receipt or {}),
        "support_encoding_receipt": support_encoding_receipt,
        "model_identity": {
            "loaded_model_state_sha256": model_receipt.get("loaded_model_state_sha256"),
            "hub_source_tree_sha256": model_receipt.get("hub_source_tree_sha256"),
            "checkpoint_sha256": model_receipt.get("checkpoint_sha256"),
            "runtime_profile": runtime_info.get("runtime_profile"),
            "allocator_fraction": memory_fraction,
            "allocator_limit_bytes": 22 * 1024**3,
        },
        "approved_commit": identity["head"],
        "execution_authorization_id": str(authorization_id),
        "r7_helper_source_sha256": dict(identity["r7_helper_source_sha256"]),
        "windows_job_supervisor_source_sha256": supervisor_sha,
        "configuration": {
            "support_images": 4, "c64_images": 64, "test_events": 20,
            "query_arms": list(ARM_IDS), "support_regime": "four deployed Supports plus independent C64",
            "query_seed": 37, "stream_shuffle_seed": 0,
            "controller_knows_smoke_horizon": False,
            "existing_stage05_query_tape_used": False,
            "stage_b_enabled": False,
        },
        "fresh_backbone_encoding_caps": {
            "support": 8, "support_actual": support_encoding_receipt["support_fresh_backbone_invocations"],
            "cap_unit": "backbone encode_batch forward calls; Support augmentation views are batched per registered image",
            "support_augmentation_view_inputs": support_encoding_receipt["support_augmentation_view_inputs"],
            "support_cls_image_inputs": support_encoding_receipt["support_cls_image_inputs"],
            "c64": 64, "c64_actual": c64_encodings,
            "test": 20, "test_actual_max": MAX_SMOKE_TEST_EVENTS,
            "total": 92,
        },
        "output_files": ["smoke_manifest.json", "smoke_scores.jsonl",
                         *[f"smoke_feedback_{arm}.jsonl" for arm in ARM_IDS],
                         "smoke_seal.json", "smoke_failure.json", ".output_budget.lock"],
        "output_budget": {"max_bytes": 2 * 1024**3, "failure_reserve_bytes": 64 * 1024,
                          "prewrite_accounting": True},
        "scientific_stage_a_complete": False,
        "stage_b_enabled": False,
    }
    ledger = None
    readers: list[OnDemandLabelReader] = []
    sinks: list[PrivateFeedbackSink] = []
    try:
        output_budget.write_json_once("smoke_manifest.json", manifest)
        ledger = ScoreLedger(
            run_dir / "smoke_scores.jsonl", output_budget=output_budget,
            artifact_kind="bounded_engineering_smoke")
        arms: dict[str, ArmRuntime] = {}
        for arm_id in ARM_IDS:
            reader = OnDemandLabelReader(
                info["json_path"], "audiojack", info["contract"]["category_json_sha256"],
                events, arm_id, ledger, prefix_only=True)
            readers.append(reader)
            sink = PrivateFeedbackSink(
                run_dir / f"smoke_feedback_{arm_id}.jsonl", arm_id,
                output_budget=output_budget)
            sinks.append(sink)
            arms[arm_id] = ArmRuntime(
                arm_id=arm_id, engine=engines[arm_id],
                controller=QueryController(arm_id, seed=37),
                coverage=NormalCoverageMemory(support_cls),
                label_reader=reader, private_feedback=sink)
        encoder = R7FrozenFeatureEncoder(
            impl=impl, visionad=visionad, repository_root=root, backbone=backbone,
            feature_fuse=cfg.feature_fuse, r1=r1,
            expected_model_state_sha256=str(model_receipt["loaded_model_state_sha256"]),
            runtime_info=runtime_info)
        coordinator = StageAClosedLoop(
            category="audiojack", events=events, feature_encoder=encoder,
            arms=arms, ledger=ledger, execution_authorization_id=authorization_id,
            smoke_mode=True, output_budget=output_budget)
        return coordinator.run()
    except BaseException as exc:
        if ledger is not None:
            ledger.close()
        for reader in readers:
            reader.close()
        for sink in sinks:
            sink.close()
        output_budget.write_failure({
            "schema": "e6.bea_v2.smoke_failure.v1", "status": "SMOKE_FAILED",
            "artifact_kind": "bounded_engineering_smoke", "stage": "smoke_stage_a",
            "failure_type": type(exc).__name__, "message": str(exc)[:2000],
            "automatic_restart": False, "stage_b_enabled": False,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-real-stage-a", action="store_true")
    parser.add_argument("--researcher-authorized", action="store_true")
    parser.add_argument("--authorization-id")
    parser.add_argument("--approved-commit")
    parser.add_argument("--category", choices=E6_REGISTERED_CATEGORIES)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if not args.execute_real_stage_a:
        print("E6 real Stage A is disabled; no dataset, model, or GPU was accessed.")
        return 0
    if not args.category or not args.run_id:
        parser.error("--category and --run-id are required with --execute-real-stage-a")
    try:
        execute_real_stage_a(
            category=args.category, run_id=args.run_id,
            approved_commit=args.approved_commit,
            human_authorized=args.researcher_authorized,
            authorization_id=args.authorization_id)
    except Exception as exc:
        print(f"E6 Stage A stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
