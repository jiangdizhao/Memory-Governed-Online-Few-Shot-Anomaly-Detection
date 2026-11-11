"""Source-bound, label-blind FOADS/K-NG per-category execution boundary."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .execution_support import (
    AttemptStatus, append_jsonl_fsync, append_jsonl_records_fsync, canonical_json,
    enforce_resource_limits, enforce_runtime_limits, estimate_attempt_output_bytes,
    resource_snapshot, runtime_resource_snapshot, sha256_file,
)
from .launch import (
    _campaign_output_root, _ensure_inside, _SAFE_ID,
    object_sha256, read_bound_config, read_json, registered_trajectory,
)


class FOADSExecutionError(RuntimeError):
    pass


FOADS_AUTH_SCHEMA = "final-benchmark.foads-p0-execution-authorization.v1"
FOADS_BUNDLE_SCHEMA = "final-benchmark.private-category-bundle.v1"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_EVENT_KEYS = ("position", "original_position", "event_id_sha256", "image_identity_sha256",
               "image_content_sha256", "source_group_sha256", "camera_id", "within_group_ordinal")


def build_foads_identity_projection(identity_manifest: Mapping[str, Any],
                                    role_manifest: Mapping[str, Any],
                                    stream_manifest: Mapping[str, Any],
                                    category: str) -> dict[str, Any]:
    """Recreate the metadata-only, label/mask-free projection bound in Gate 1."""
    from .foads_kng import _reject_label_mask_fields
    identity_sha = object_sha256(identity_manifest)
    if identity_manifest.get("schema") != "final-benchmark.identity-manifest.v1":
        raise FOADSExecutionError("FOADS identity source has an unsupported schema")
    role_ids = {str(row.get("image_identity_sha256")) for row in role_manifest.get("selected", [])}
    test_ids = {str(row.get("image_identity_sha256")) for row in stream_manifest.get("events", [])}
    identity_ids = role_ids | test_ids
    records = [dict(row) for row in identity_manifest.get("identity_records", [])
               if row.get("category") == category and row.get("image_identity_sha256") in identity_ids]
    if len(records) != len(identity_ids) or len(role_ids) != 68 or not test_ids:
        raise FOADSExecutionError("FOADS projection does not cover its registered 68 normals and TEST stream")
    events = []
    for event in stream_manifest.get("events", []):
        if not isinstance(event, Mapping) or set(_EVENT_KEYS) - set(event):
            raise FOADSExecutionError("registered FOADS event lacks its exact identity fields")
        events.append({key: event[key] for key in _EVENT_KEYS})
    stream = dict(stream_manifest)
    stream["events"] = events
    projection = {
        "schema": identity_manifest["schema"],
        "dataset_id": identity_manifest["dataset_id"],
        "identity_census_status": identity_manifest["identity_census_status"],
        "parent_identity_manifest_sha256": identity_sha,
        "category": category,
        "identity_records": records,
        "role_manifests": [dict(role_manifest)],
        "stream_manifests": [stream],
    }
    _reject_label_mask_fields(projection)
    return projection


def _stable_state(value: Any) -> Any:
    """Convert native Klinks state to deterministic hashable data, failing closed."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FOADSExecutionError("K-NG state contains nonfinite scalar values")
        return value
    if isinstance(value, np.generic):
        return _stable_state(value.item())
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise FOADSExecutionError("K-NG state contains nonfinite array values")
        contiguous = np.ascontiguousarray(value)
        return {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape),
                "bytes_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()}
    try:
        import torch
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            array = tensor.numpy()
            return {"dtype": str(array.dtype), "shape": list(array.shape),
                    "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return {str(key): _stable_state(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple, set)):
        rows = [_stable_state(item) for item in value]
        return sorted(rows, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))) \
            if isinstance(value, set) else rows
    # NetworkX graph is handled structurally, including all node and edge fields.
    if callable(getattr(value, "nodes", None)) and callable(getattr(value, "edges", None)):
        nodes = [[_stable_state(node_id), _stable_state(attrs)]
                 for node_id, attrs in value.nodes(data=True)]
        edges = [[_stable_state(u), _stable_state(v), _stable_state(attrs)]
                 for u, v, attrs in value.edges(data=True)]
        nodes.sort(key=lambda row: json.dumps(row[0], sort_keys=True, separators=(",", ":")))
        edges.sort(key=lambda row: json.dumps(row[:2], sort_keys=True, separators=(",", ":")))
        return {"type": type(value).__name__, "nodes": nodes, "edges": edges,
                "graph": _stable_state(getattr(value, "graph", {}))}
    raise FOADSExecutionError(f"unsupported K-NG state field type: {type(value).__name__}")


def foads_model_state_sha256(model: Any) -> str:
    """Hash node means/covariances/thresholds, node ordering and graph topology."""
    network = getattr(model, "network", None)
    node_list = getattr(model, "node_list", None)
    if network is None or node_list is None or not callable(getattr(network, "nodes", None)):
        raise FOADSExecutionError("Klinks model lacks the registered node list or graph state")
    nodes = list(network.nodes(data=True))
    if not nodes or not node_list:
        raise FOADSExecutionError("Klinks graph has no initialized nodes")
    state = {"node_list": _stable_state(list(node_list)),
             "network": _stable_state(network)}
    return hashlib.sha256(canonical_json(state)).hexdigest()


def _verify_p0_bundle_and_authorization(*, config: Mapping[str, Any], config_file_sha: str,
                                        bundle: Mapping[str, Any], identity: Mapping[str, Any],
                                        locators: Mapping[str, Any], category: str,
                                        authorization: Mapping[str, Any],
                                        source_file_sha256: Mapping[str, str],
                                        weights_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    trajectory = registered_trajectory(config, category, "P0")
    category_bundle = read_json(bundle["category_bundle_path"])
    role = category_bundle.get("role_manifest")
    stream = category_bundle.get("stream_manifest")
    release = category_bundle.get("release_identity")
    projection = build_foads_identity_projection(identity, role, stream, category)
    projection_sha = object_sha256(projection)
    if (projection_sha != trajectory.get("foads_p0_registration_projection_sha256") or
            object_sha256(role) != trajectory.get("role_manifest_sha256") or
            object_sha256(stream) != trajectory.get("stream_manifest_sha256") or
            object_sha256(identity) != config["dataset"]["identity_manifest_sha256"] or
            sha256_file(bundle["identity_manifest_path"]) != config["dataset"]["identity_manifest_file_sha256"] or
            release.get("dataset_release_sha256") != config["dataset"]["dataset_release_sha256"]):
        raise FOADSExecutionError("FOADS P0 metadata differs from the frozen identity/role/stream commitments")
    locator_sha = sha256_file(bundle["private_locator_manifest_path"])
    file_sha = sha256_file(bundle["category_bundle_path"])
    if (locators.get("schema") != "final-benchmark.private-locators.v1" or
            locators.get("identity_manifest_sha256") != config["dataset"]["identity_manifest_sha256"] or
            locators.get("categories", {}).get(category, {}).get("path_validation_status") != "PASS"):
        raise FOADSExecutionError("FOADS private locators do not match the certified category identity")
    annotation_sha = locators.get("categories", {}).get(category, {}).get("annotation_sha256")
    annotation_artifacts = [row.get("sha256") for row in identity.get("source_artifacts", [])
                            if row.get("artifact_id") == f"annotation:{category}"]
    if len(annotation_artifacts) != 1 or annotation_sha != annotation_artifacts[0]:
        raise FOADSExecutionError("FOADS annotation locator is not bound to its identity-only source hash")
    expected_auth = {
        "schema": FOADS_AUTH_SCHEMA, "status": "AUTHORIZED", "scope": "foads_kng_p0_category",
        "source_commit": config["source_identity"]["final_source_commit"],
        "campaign_run_id": config["source_identity"]["campaign_run_id"],
        "execution_plan_sha256": config_file_sha, "dataset_id": "realiad", "category": category,
        "condition_id": "P0", "identity_manifest_sha256": config["dataset"]["identity_manifest_sha256"],
        "role_manifest_sha256": trajectory["role_manifest_sha256"],
        "stream_manifest_sha256": trajectory["stream_manifest_sha256"],
        "dataset_release_sha256": config["dataset"]["dataset_release_sha256"],
        "foads_registration_projection_sha256": projection_sha,
        "category_bundle_file_sha256": file_sha,
        "private_locator_manifest_file_sha256": locator_sha,
        "upstream_commit": config["foads_kng"]["upstream_commit"],
        "upstream_source_file_sha256": dict(source_file_sha256),
        "resnet18_weights_sha256": weights_sha256,
        "expected_event_count": int(trajectory["test_event_count"]),
        "paid_labels": 0, "test_outcome_access": False,
        "test_mask_access": False, "stage_b_authorized": False,
    }
    mismatches = [key for key, value in expected_auth.items() if authorization.get(key) != value]
    if mismatches:
        raise FOADSExecutionError("FOADS authorization does not match: " + ", ".join(mismatches))
    test_locators = locators["categories"][category]["test"]
    available_train_locators = locators["categories"][category]["role_train"]
    role_ids = {row["image_identity_sha256"] for row in role["selected"]}
    if ({event["image_identity_sha256"] for event in stream["events"]} != set(test_locators) or
            not role_ids.issubset(available_train_locators)):
        raise FOADSExecutionError("FOADS locators differ from registered TEST or initial TRAIN identities")
    train_locators = {image_id: available_train_locators[image_id] for image_id in role_ids}
    identities = {
        "source_commit": expected_auth["source_commit"],
        "campaign_run_id": expected_auth["campaign_run_id"],
        "execution_plan_sha256": config_file_sha,
        "identity_manifest_sha256": expected_auth["identity_manifest_sha256"],
        "role_manifest_sha256": expected_auth["role_manifest_sha256"],
        "stream_manifest_sha256": expected_auth["stream_manifest_sha256"],
        "dataset_release_sha256": expected_auth["dataset_release_sha256"],
        "foads_registration_projection_sha256": projection_sha,
        "upstream_commit": expected_auth["upstream_commit"],
        "upstream_source_file_sha256": dict(source_file_sha256),
        "resnet18_weights_sha256": weights_sha256,
        "category_bundle_file_sha256": file_sha,
        "private_locator_manifest_file_sha256": locator_sha,
        "category": category, "condition": "P0",
    }
    return {"category_bundle": category_bundle, "projection": projection,
            "test_locators": test_locators, "train_locators": train_locators,
            "annotation_sha256": locators["categories"][category]["annotation_sha256"]}, identities


def _verify_registered_image_path(event: Mapping[str, Any], locator: Mapping[str, Any], *,
                                  category: str, split: str, annotation_sha256: str,
                                  identity_manifest_sha256: str) -> Path:
    from . import identity as identity_module
    path = Path(str(locator.get("image_path", "")))
    if not path.is_file():
        raise FOADSExecutionError("registered image file is missing")
    try:
        binding = identity_module.validate_realiad_source_path(
            path=path, category=category, split=split,
            annotation_sha256=annotation_sha256,
            official_image_key=locator["official_image_key"],
            official_group_key=locator["official_group_key"],
            image_identity_sha256=event["image_identity_sha256"],
            source_group_sha256=event["source_group_sha256"],
            camera_id=int(event.get("camera_id", 0)),
            original_position=event.get("original_position"),
            image_content_sha256=event["image_content_sha256"],
            relative_id=locator.get("relative_id"),
            identity_manifest_sha256=identity_manifest_sha256)
    except Exception as exc:
        raise FOADSExecutionError("registered image path failed official identity validation") from exc
    if (binding.get("official_group_key") != locator.get("official_group_key") or
            binding.get("camera_id") != int(event.get("camera_id", 0))):
        raise FOADSExecutionError("resolved image path differs from its registered event identity")
    if sha256_file(path) != event["image_content_sha256"]:
        raise FOADSExecutionError("registered image content hash differs from the certified identity")
    return path


def _load_rgb_feature(resnet: Any, transform: Any, source_api: Any, path: Path,
                      device: Any) -> np.ndarray:
    from PIL import Image
    import torch
    try:
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        from .foads_kng import extract_kng_embeddings
        result = extract_kng_embeddings(resnet, tensor, source_api)
        if result.shape != (1, 56, 56, 100) or not np.isfinite(result).all():
            raise FOADSExecutionError("ResNet18 produced an invalid K-NG feature map")
        return np.asarray(result[0], dtype=np.float32)
    except FOADSExecutionError:
        raise
    except Exception as exc:
        raise FOADSExecutionError("could not load or encode a registered RGB image") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FOADSExecutionError(f"required FOADS journal is missing: {path.name}")
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise FOADSExecutionError(f"FOADS journal has a truncated final line: {path.name}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise FOADSExecutionError("FOADS journal record must be an object")
            rows.append(value)
    return rows


def build_foads_seal(*, registration: Any, score_journal_path: str | Path,
                     update_journal_path: str | Path, identity: Mapping[str, Any],
                     runner_summary: Mapping[str, Any], final_state_sha256: str) -> dict[str, Any]:
    score_path, update_path = Path(score_journal_path), Path(update_journal_path)
    scores, updates = _read_jsonl(score_path), _read_jsonl(update_path)
    events = registration.stream_events
    groups = [events[start:start + 5] for start in range(0, len(events), 5)]
    if (any(len(group) != 5 for group in groups) or len(scores) != len(events) or
            runner_summary.get("scored_images") != len(events)):
        raise FOADSExecutionError("FOADS score stream is incomplete or not group aligned")
    for position, (record, event) in enumerate(zip(scores, events)):
        if (record.get("position") != position or record.get("position") != event.position or
                record.get("event_id_sha256") != event.event_id_sha256 or
                record.get("image_identity_sha256") != event.image_identity_sha256 or
                record.get("source_group_sha256") != event.source_group_sha256 or
                record.get("model_update_index") != position // 10 or
                not isinstance(record.get("raw_image_score"), (int, float)) or
                not math.isfinite(float(record["raw_image_score"])) or
                not _HEX64.fullmatch(str(record.get("model_state_before_score_sha256", "")))):
            raise FOADSExecutionError("FOADS score journal differs from registered order or model state")
    expected_updates = len(groups) // 2
    if len(updates) != expected_updates or runner_summary.get("completed_batch10_updates") != expected_updates:
        raise FOADSExecutionError("FOADS native update receipt count differs from complete Batch-10 accounting")
    for index, update in enumerate(updates):
        start, stop = index * 10, (index + 1) * 10
        expected_group_ids = [groups[index * 2][0].source_group_sha256,
                              groups[index * 2 + 1][0].source_group_sha256]
        if (update.get("schema") != "final-benchmark.foads-native-update.v1" or
                update.get("update_index") != index + 1 or update.get("start_position") != start or
                update.get("end_position_exclusive") != stop or
                update.get("event_identity_sha256s") != [event.image_identity_sha256 for event in events[start:stop]] or
                update.get("source_group_sha256s") != expected_group_ids or
                not _HEX64.fullmatch(str(update.get("model_state_after_update_sha256", "")))):
            raise FOADSExecutionError("FOADS update receipt is missing or differs from the registered Batch-10")
    final_groups = len(groups) % 2
    if (runner_summary.get("final_unupdated_groups") != final_groups or
            runner_summary.get("final_unupdated_images") != final_groups * 5 or
            runner_summary.get("paid_labels") != 0 or runner_summary.get("test_labels_or_masks_read") != 0):
        raise FOADSExecutionError("FOADS final unupdated group or zero-label accounting is invalid")
    body = {
        "schema": "final-benchmark.foads-category-seal.v1", "status": "SEALED_COMPLETE",
        "baseline": "FOADS-KNG-PAPER-CONSISTENT-PROTOCOL-ADAPTED",
        "identity": dict(identity), "expected_event_count": len(events),
        "scored_images": len(scores), "source_group_count": len(groups),
        "completed_batch10_updates": len(updates),
        "final_unupdated_groups": final_groups,
        "final_unupdated_group_sha256": groups[-1][0].source_group_sha256 if final_groups else None,
        "paid_labels": 0, "test_outcome_or_mask_access": False,
        "prediction_journal_sha256": sha256_file(score_path),
        "update_journal_sha256": sha256_file(update_path),
        "final_model_state_sha256": final_state_sha256,
    }
    body["seal_sha256"] = hashlib.sha256(canonical_json(body)).hexdigest()
    return body


def verify_foads_category_seal(root: str | Path, *, expected_identity: Mapping[str, Any]) -> dict[str, Any]:
    base = Path(root)
    try:
        status = json.loads((base / "attempt_status.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise FOADSExecutionError("FOADS attempt status receipt is missing or unreadable") from exc
    if (status.get("status") != "SEALED_COMPLETE" or status.get("identity") != dict(expected_identity) or
            status.get("execution_kind") != "foads_kng_p0"):
        raise FOADSExecutionError("FOADS attempt is incomplete or bound to different inputs")
    seal_path = base / "foads_category_seal.json"
    try:
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FOADSExecutionError("FOADS category seal is missing or unreadable") from exc
    seal_body = dict(seal)
    recorded_seal_sha = seal_body.pop("seal_sha256", None)
    actual_seal_sha = hashlib.sha256(canonical_json(seal_body)).hexdigest()
    if (status.get("completion_seal_sha256") != sha256_file(seal_path) or
            seal.get("status") != "SEALED_COMPLETE" or seal.get("identity") != dict(expected_identity) or
            recorded_seal_sha != actual_seal_sha or
            seal.get("test_outcome_or_mask_access") is not False):
        raise FOADSExecutionError("FOADS completion receipt is incomplete or bound to different inputs")
    if (seal.get("prediction_journal_sha256") != sha256_file(base / "foads_prediction_journal.jsonl") or
            seal.get("update_journal_sha256") != sha256_file(base / "foads_update_journal.jsonl")):
        raise FOADSExecutionError("FOADS prediction or update journal differs from the successful seal")
    return seal


def _verify_resnet_weights_file(expected_sha256: str) -> Path:
    try:
        import torch
        from torchvision.models import ResNet18_Weights
        filename = Path(ResNet18_Weights.IMAGENET1K_V1.url).name
        path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    except Exception as exc:
        raise FOADSExecutionError("official ResNet18 checkpoint metadata is unavailable") from exc
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise FOADSExecutionError("official ResNet18 IMAGENET1K_V1 checkpoint is missing or has a different SHA-256")
    return path


def _initialize_attempt_files(root: Path) -> tuple[Path, Path]:
    score_path = root / "foads_prediction_journal.jsonl"
    update_path = root / "foads_update_journal.jsonl"
    for path in (score_path, update_path):
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.fsync(descriptor)
        os.close(descriptor)
    return score_path, update_path


def execute_foads_one(*, config_path: str | Path, bundle_path: str | Path,
                      locator_manifest_path: str | Path, authorization_path: str | Path,
                      category: str, attempt_id: str, kng_source_root: str | Path,
                      repository: str | Path | None = None,
                      source_api: Any | None = None, resnet: Any | None = None,
                      image_loader: Any | None = None, resource_probe: Any = resource_snapshot) -> dict[str, Any]:
    if not _SAFE_ID.fullmatch(attempt_id):
        raise FOADSExecutionError("attempt ID must be a safe immutable path component")
    repository_path = Path(repository or Path.cwd()).resolve()
    config, config_file_sha, config_report = read_bound_config(
        config_path, require_source_binding=True, repository=repository_path)
    trajectory = registered_trajectory(config, category, "P0")
    bundle = read_json(bundle_path)
    if not isinstance(bundle, Mapping) or bundle.get("schema") != FOADS_BUNDLE_SCHEMA:
        raise FOADSExecutionError("FOADS launch requires the registered private category bundle")
    identity = read_json(bundle["identity_manifest_path"])
    category_bundle = read_json(bundle["category_bundle_path"])
    locators = read_json(locator_manifest_path)
    if (bundle.get("schema") != FOADS_BUNDLE_SCHEMA or
            category_bundle.get("category") != category or category_bundle.get("condition_id") != "P0" or
            category_bundle.get("schema") != "final-benchmark.category-condition-identity.v1" or
            bundle.get("private_locator_manifest_path") is None or
            Path(str(bundle["private_locator_manifest_path"])).resolve() != Path(locator_manifest_path).resolve()):
        raise FOADSExecutionError("FOADS bundle is not bound to the selected P0 category/locator file")
    try:
        from .foads_kng import load_upstream_api
        api = source_api or load_upstream_api(kng_source_root)
    except Exception as exc:
        raise FOADSExecutionError(f"pinned K-NG source validation/import failed: {exc}") from exc
    auth = read_json(authorization_path)
    if not isinstance(auth, Mapping):
        raise FOADSExecutionError("FOADS category authorization must be a JSON object")
    weights_sha = config["foads_kng"]["weights_sha256"]
    registration_payload, identities = _verify_p0_bundle_and_authorization(
        config=config, config_file_sha=config_file_sha, bundle=bundle,
        identity=identity, locators=locators, category=category, authorization=auth,
        source_file_sha256=api.verified_source.file_sha256, weights_sha256=weights_sha)
    expected_events = int(trajectory["test_event_count"])
    if len(registration_payload["category_bundle"]["stream_manifest"]["events"]) != expected_events:
        raise FOADSExecutionError("FOADS P0 event count differs from the frozen schedule")
    root = _campaign_output_root(config, repository_path) / "foads_kng" / category / "P0" / attempt_id
    _ensure_inside(root, _campaign_output_root(config, repository_path))
    if root.exists():
        raise FOADSExecutionError(f"FOADS attempt output already exists: {root}")
    estimate = estimate_attempt_output_bytes("foads", expected_events)
    initial_resources = resource_probe(root)
    enforce_resource_limits("foads", initial_resources, required_disk_bytes=estimate)
    _verify_resnet_weights_file(weights_sha)
    from .foads_kng import (
        build_registered_p0, build_pretrained_resnet18,
        initialize_registered_p0_kng, kng_resnet18_transform,
    )
    projection = registration_payload["projection"]
    registration = build_registered_p0(
        projection, category,
        expected_identity_manifest_sha256=object_sha256(projection),
        expected_role_manifest_sha256=trajectory["role_manifest_sha256"],
        expected_dataset_release_sha256=config["dataset"]["dataset_release_sha256"])
    test_locators, train_locators = registration_payload["test_locators"], registration_payload["train_locators"]
    annotation_sha = registration_payload["annotation_sha256"]
    # Authorization and source/role/stream checks are complete before the attempt begins.
    attempt = AttemptStatus(root, identity=identities, expected_events=expected_events,
                            execution_kind="foads_kng_p0")
    phase, completed = "AUTHORIZED_START", 0
    start_time = time.perf_counter()
    resnet = model = runner = maps = maps_np = train_feature_maps = None
    try:
        score_path, update_path = _initialize_attempt_files(root)
        attempt.update(phase="PINNED_SOURCE_VERIFIED", completed_events=0)
        _verify_resnet_weights_file(weights_sha)
        if resnet is None:
            resnet, actual_state_sha = build_pretrained_resnet18()
        else:
            from .foads_kng import resnet18_state_sha256
            actual_state_sha = resnet18_state_sha256(resnet)
        resnet.eval()
        import torch
        device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise FOADSExecutionError("registered GPU is not available for FOADS category execution")
        resnet.to(device)
        transform = kng_resnet18_transform()
        train_feature_maps = {}
        train_records = {row["image_identity_sha256"]: row for row in identity["identity_records"]
                         if row.get("split") == "train" and row.get("category") == category}
        for image_id in registration.initial_train_image_ids:
            record = train_records.get(image_id)
            if record is None:
                raise FOADSExecutionError("registered FOADS TRAIN-normal identity is absent from the manifest")
            path = _verify_registered_image_path(record, train_locators[image_id],
                category=category, split="train", annotation_sha256=annotation_sha,
                identity_manifest_sha256=config["dataset"]["identity_manifest_sha256"])
            if image_loader is None:
                image = _load_rgb_feature(resnet, transform, api, path, device)
            else:
                image = image_loader("train", record, train_locators[image_id], resnet, transform, api, device)
            train_feature_maps[image_id] = image
        model, initialization = initialize_registered_p0_kng(
            registration, train_feature_maps, api, distance_device=device)
        state_sha = foads_model_state_sha256(model)
        runner = __import__("experiments.calibration.final_benchmark.foads_kng", fromlist=["FOADSKNGP0Runner"]).FOADSKNGP0Runner(
            registration, model, api)
        attempt.update(phase="MODEL_INITIALIZED", completed_events=0)
        all_events = registration.stream_events
        groups = [all_events[start:start + 5] for start in range(0, expected_events, 5)]
        update_group_ids: list[str] = []
        for group_index, group in enumerate(groups):
            phase = "ENCODING_GROUP"
            maps = []
            for event in group:
                record = asdict(event)
                path = _verify_registered_image_path(record, test_locators[event.image_identity_sha256],
                    category=category, split="test", annotation_sha256=annotation_sha,
                    identity_manifest_sha256=config["dataset"]["identity_manifest_sha256"])
                if image_loader is None:
                    feature = _load_rgb_feature(resnet, transform, api, path, device)
                else:
                    feature = image_loader("test", record, test_locators[event.image_identity_sha256],
                                           resnet, transform, api, device)
                maps.append(feature)
            maps_np = np.stack(maps, axis=0)
            previous_updates = runner.update_count
            state_before = state_sha
            scored = runner.process_group(group, maps_np)
            score_records = []
            for row in scored:
                record = asdict(row)
                record["model_state_before_score_sha256"] = state_before
                score_records.append(record)
            append_jsonl_records_fsync(score_path, score_records)
            update_group_ids.append(group[0].source_group_sha256)
            completed = runner.next_position
            attempt.update(phase="GROUP_SCORES_DURABLE", completed_events=completed)
            if runner.update_count == previous_updates + 1:
                state_sha = foads_model_state_sha256(model)
                update_records = {
                    "schema": "final-benchmark.foads-native-update.v1",
                    "update_index": runner.update_count,
                    "start_position": completed - 10,
                    "end_position_exclusive": completed,
                    "event_identity_sha256s": [event.image_identity_sha256
                                               for event in all_events[completed - 10:completed]],
                    "source_group_sha256s": list(update_group_ids),
                    "model_state_after_update_sha256": state_sha,
                }
                append_jsonl_fsync(update_path, update_records)
                update_group_ids.clear()
                phase = "UPDATE_RECEIPT_DURABLE"
            if group_index % 10 == 0 or completed == expected_events:
                live = runtime_resource_snapshot(root)
                enforce_runtime_limits("foads", live,
                    required_disk_bytes=estimate_attempt_output_bytes("foads", expected_events - completed))
        summary = runner.finish()
        final_state_sha = foads_model_state_sha256(model)
        if not isinstance(actual_state_sha, str) or _HEX64.fullmatch(actual_state_sha) is None:
            raise FOADSExecutionError("ResNet18 model state fingerprint is malformed")
        seal = build_foads_seal(registration=registration,
            score_journal_path=score_path, update_journal_path=update_path,
            identity=identities, runner_summary=summary, final_state_sha256=final_state_sha)
        seal_path = root / "foads_category_seal.json"
        payload = canonical_json(seal) + b"\n"
        descriptor = os.open(seal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if (sha256_file(score_path) != seal["prediction_journal_sha256"] or
                sha256_file(update_path) != seal["update_journal_sha256"] or
                json.loads(seal_path.read_text(encoding="utf-8")) != seal):
            raise FOADSExecutionError("FOADS output journals changed while writing the completion seal")
        attempt.sealed_complete(phase="CATEGORY_SEALED", seal_path=seal_path,
                                completed_events=summary["scored_images"])
        verified = seal
        return {"status": "SEALED_COMPLETE", "attempt_root": str(root),
                "scored_images": summary["scored_images"],
                "completed_batch10_updates": summary["completed_batch10_updates"],
                "final_unupdated_groups": summary["final_unupdated_groups"],
                "elapsed_seconds": time.perf_counter() - start_time,
                "resnet_state_sha256": actual_state_sha,
                "initialization": initialization,
                "prediction_journal_sha256": verified["prediction_journal_sha256"],
                "update_journal_sha256": verified["update_journal_sha256"],
                "completion_seal_sha256": sha256_file(seal_path),
                "resources_at_start": initial_resources, "configuration": config_report}
    except BaseException as exc:
        attempt.incomplete(phase=phase, completed_events=completed, exception=exc)
        raise
    finally:
        resnet = model = runner = maps = maps_np = train_feature_maps = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

