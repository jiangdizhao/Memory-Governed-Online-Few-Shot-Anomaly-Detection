"""Gated launcher for the frozen FS4 S/CL/CE scientific campaign.

The no-argument operation is inert. Real-data staging, Support parity, Stage A
and Stage B each require a separate campaign-bound authorization record.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Mapping

from .fs4_durable_ledger import canonical, digest
from .fs4_joint_campaign_contract import (ARM_IDS, EXPECTED_CATEGORIES, EXPECTED_EVENTS,
    CampaignContractError, build_trajectory_manifests, load_spec, spec_identity,
    validate_registration, validate_spec, validate_support_preflight)

ROOT = Path(__file__).resolve().parents[2]
AUTH_SCHEMA = "FS4_JOINT_CAMPAIGN_AUTHORIZATION_V1"
CAMPAIGN_SCHEMA = "FS4_JOINT_CAMPAIGN_REGISTRATION_V1"
SCOPES = {"STAGING", "SUPPORT_PREFLIGHT", "RESOURCE_PREFLIGHT", "STAGE_A", "STAGE_B"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_new(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_new(path: Path, value: Any) -> str:
    data = canonical(value) + b"\n"
    _write_new(path, data)
    return _sha(data)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_bytes())
    except Exception as exc:
        raise ValueError(f"required campaign JSON is unreadable: {Path(path).name}") from exc


def _record_incomplete(path: Path, reason_code: str) -> None:
    marker = Path(path) / "INCOMPLETE.json"
    if marker.parent.exists() and not marker.exists():
        _write_json_new(marker, {"schema": "FS4_JOINT_CAMPAIGN_INCOMPLETE_V1",
            "status": "INCOMPLETE", "reason_code": reason_code, "spec_sha256": spec_identity()})


def _validate_authorization(path: Path, *, scope: str, campaign_id: str,
                            bindings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    auth = _read_json(path)
    binding = dict(bindings or {})
    required = {"schema", "authorization_id", "authorized_by", "authorized_at", "scope",
                "campaign_id", "scientific_spec_sha256", "authorization_statement", *binding}
    if (not isinstance(auth, dict) or set(auth) != required or auth.get("schema") != AUTH_SCHEMA or
            auth.get("scope") != scope or scope not in SCOPES or auth.get("campaign_id") != campaign_id or
            auth.get("scientific_spec_sha256") != spec_identity() or
            any(auth.get(key) != value for key, value in binding.items()) or
            any(not isinstance(auth.get(key), str) or not auth[key].strip()
                for key in ("authorization_id", "authorized_by", "authorized_at", "authorization_statement"))):
        raise CampaignContractError(f"separate {scope} authorization is missing, malformed or bound to different inputs")
    return auth


def _load_campaign(campaign_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign_root = Path(campaign_root).absolute()
    registration = _read_json(campaign_root / "registration.json")
    metadata = _read_json(campaign_root / "campaign.json")
    if (metadata.get("schema") != CAMPAIGN_SCHEMA or metadata.get("campaign_id") is None or
            metadata.get("scientific_spec_sha256") != spec_identity() or
            metadata.get("campaign_registration_sha256") != digest(registration)):
        raise CampaignContractError("campaign registration receipt or frozen specification digest differs")
    validate_registration(registration)
    return metadata, registration


def _check_committed_sources(source_commit: str) -> dict[str, str]:
    from .fs4_joint_campaign_contract import source_fingerprints
    result = source_fingerprints(ROOT)
    for relative, expected in result.items():
        committed = subprocess.run(["git", "show", f"{source_commit}:{relative}"], cwd=ROOT,
                                   capture_output=True)
        if committed.returncode or _sha(committed.stdout.replace(b"\r\n", b"\n")) != _sha((ROOT / relative).read_bytes().replace(b"\r\n", b"\n")):
            raise CampaignContractError(f"campaign source is not frozen at the registered commit: {relative}")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, "HEAD"], cwd=ROOT,
                              capture_output=True)
    if ancestor.returncode:
        raise CampaignContractError("campaign source-freeze commit is not an ancestor of HEAD")
    return result


def freeze_only() -> dict[str, Any]:
    spec = load_spec()
    return {"schema": "FS4_FINAL_JOINT_CAMPAIGN_FREEZE_CHECK_V1", "status": "FROZEN_SPEC_VALID",
            "scientific_execution_authorized": False, "scientific_spec_sha256": spec_identity(spec),
            "events": sum(EXPECTED_EVENTS.values()), "source_groups": sum(n // 5 for n in EXPECTED_EVENTS.values()),
            "trajectories": list(ARM_IDS)}


def validate_contract() -> dict[str, Any]:
    spec = load_spec()
    arms = _read_json(ROOT / "docs/calibration/fs4_joint_campaign/ARM_REGISTRY.json")
    if arms.get("schema") != "FS4_FINAL_JOINT_CAMPAIGN_ARM_REGISTRY_V1" or len(arms.get("arms", [])) != 3:
        raise CampaignContractError("human-readable arm registry is incomplete")
    actual = [(row.get("trajectory_id"), row.get("profile"), row.get("controller")) for row in arms["arms"]]
    expected = [(row["trajectory_id"], row["profile"], row["controller"]) for row in spec["arms"]]
    if actual != expected or arms.get("shared_registration", {}).get("registration_count") != 1:
        raise CampaignContractError("human and machine-readable trajectory registrations differ")
    return {"status": "PASS", "scientific_spec_sha256": spec_identity(spec),
            "category_events": EXPECTED_EVENTS, "total_events": sum(EXPECTED_EVENTS.values()),
            "arm_registry": actual, "execution_enabled": False}


def _r7_reordered_registration(original: Mapping[str, Any], original_ledger: Mapping[str, Any],
                               labels_by_original_position: Mapping[int, int]) -> tuple[dict[str, Any], dict[str, Any], dict[int, int], str, str]:
    from experiments.risk_safe.preflight import parse_image_name
    from .r7_shuffled_correction_4class import deterministic_group_shuffle

    if (original.get("schema") != "FS4_TRUSTED_EXECUTION_V1" or
            original.get("category_order") != list(EXPECTED_CATEGORIES) or
            original.get("query_seed") != 37 or original.get("model_state_sha256") != load_spec()["source_model_support"]["model"]["model_state_sha256"]):
        raise CampaignContractError("input trusted registration differs from the frozen four-category R4 population")
    if original_ledger.get("expected_events_by_category") != EXPECTED_EVENTS:
        raise CampaignContractError("input source manifest population differs from the frozen event counts")
    if set(labels_by_original_position) != set(range(sum(EXPECTED_EVENTS.values()))):
        raise CampaignContractError("private staging labels do not cover the exact original registration positions")
    categories = copy.deepcopy(original["categories"])
    result_events: dict[str, list[dict[str, Any]]] = {}
    order_receipt = []
    labels_by_new_position: dict[int, int] = {}
    global_position = 0
    for category in EXPECTED_CATEGORIES:
        rows = categories[category]["events"]
        if len(rows) != EXPECTED_EVENTS[category]:
            raise CampaignContractError(f"{category}: original registration event count differs")
        prepared = []
        view_by_group: dict[str, list[int]] = {}
        row_by_position = {}
        for row in rows:
            absolute = Path(original["dataset_root"]) / row["relative_path"]
            parsed = parse_image_name(absolute, expected_class=category)
            if row["group"] != hashlib.sha256(f"official-sample-group\0{parsed.official_sample_key}".encode()).hexdigest():
                raise CampaignContractError(f"{category}: original group identity differs from the image name")
            view_by_group.setdefault(row["group"], []).append(parsed.camera_id)
            row_by_position[row["position"]] = row
            prepared.append({"opaque_group": row["group"], "position": row["position"],
                "relative_id": row["relative_path"], "camera_id": str(parsed.camera_id),
                "timestamp": parsed.timestamp, "image_path": str(absolute)})
        if any(sorted(views) != [1, 2, 3, 4, 5] or views != sorted(views) for views in view_by_group.values()):
            raise CampaignContractError(f"{category}: a registered five-view Source Group is incomplete or reordered")
        shuffled = deterministic_group_shuffle(prepared, seed=0)
        if len(shuffled) != len(rows) or {event.original_position for event in shuffled} != set(row_by_position):
            raise CampaignContractError(f"{category}: R7 group shuffle is not a bijection")
        ordered_rows = []
        for event in shuffled:
            row = copy.deepcopy(row_by_position[event.original_position])
            row["position"] = global_position
            ordered_rows.append(row)
            labels_by_new_position[global_position] = labels_by_original_position[event.original_position]
            order_receipt.append({"category": category, "position": global_position,
                "original_position": event.original_position, "group_sha256": event.source_group_sha256,
                "view": int(event.camera_id), "image_identity_sha256": event.image_identity_sha256})
            global_position += 1
        result_events[category] = ordered_rows
    reordered = copy.deepcopy(dict(original))
    reordered_categories = {}
    for category in EXPECTED_CATEGORIES:
        reordered_categories[category] = {"supports": copy.deepcopy(categories[category]["supports"]),
                                          "events": result_events[category]}
    reordered["categories"] = reordered_categories
    reordered["dataset_identity_sha256"] = digest({"dataset_root": reordered["dataset_root"],
        "category_order": reordered["category_order"], "categories": reordered_categories})
    ledger = copy.deepcopy(dict(original_ledger))
    ledger["dataset_identity_sha256"] = reordered["dataset_identity_sha256"]
    ledger["stream_sha256_by_category"] = {category: digest([
        {"group": row["group"], "event": row["event"]} for row in result_events[category]])
        for category in EXPECTED_CATEGORIES}
    return reordered, ledger, labels_by_new_position, digest(order_receipt), digest({"registration": original, "ledger": original_ledger})


def prepare_real_data(*, campaign_id: str, campaign_root: Path, private_root: Path, image_root: Path,
                      original_registration_path: Path, original_ledger_path: Path,
                      labels_path: Path, authorization_path: Path) -> dict[str, Any]:
    _validate_authorization(authorization_path, scope="STAGING", campaign_id=campaign_id)
    campaign_root, private_root, image_root = (Path(p).absolute() for p in
        (campaign_root, private_root, image_root))
    spec = load_spec()
    for left, right in ((campaign_root, private_root), (campaign_root, image_root), (private_root, image_root)):
        if left == right or left.is_relative_to(right) or right.is_relative_to(left):
            raise CampaignContractError("campaign, mapping and image stores must be non-nested separate roots")
    if campaign_root.exists() or private_root.exists() or image_root.exists():
        raise FileExistsError("real-data campaign and blinded stores must use fresh non-overwriting roots")
    original = _read_json(original_registration_path)
    original_ledger = _read_json(original_ledger_path)
    labels_document = _read_json(labels_path)
    if set(labels_document) != {"schema", "labels_by_position"} or labels_document["schema"] != "FS4_CAMPAIGN_PRIVATE_LABEL_INPUT_V1":
        raise CampaignContractError("staging labels input has an invalid explicit schema")
    labels_by_original = {int(key): value for key, value in labels_document["labels_by_position"].items()}
    reordered, reordered_ledger, labels_by_new, order_sha, source_identity = _r7_reordered_registration(
        original, original_ledger, labels_by_original)
    from .fs4_r4_preparation import prepare_blinded_registration
    blinded, preparation_receipt = prepare_blinded_registration(original=reordered,
        original_ledger=reordered_ledger, labels_by_position=labels_by_new,
        private_root=private_root, image_root=image_root)
    validate_registration(blinded, spec=spec)
    campaign_root.mkdir(parents=False, exist_ok=False)
    campaign_meta = {"schema": CAMPAIGN_SCHEMA, "campaign_id": campaign_id,
        "scientific_spec_sha256": spec_identity(spec), "source_freeze_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True,
            text=True).stdout.strip(), "campaign_registration_sha256": digest(blinded),
        "source_mapping_sha256": blinded["source_mapping_sha256"], "image_root": str(image_root),
        "private_root": str(private_root), "dataset_root": blinded["dataset_root"],
        "r7_seed0_order_sha256": order_sha, "source_manifest_identity_sha256": source_identity,
        "preparation_receipt": preparation_receipt, "trajectory_manifests": {},
        "status": "STAGING_COMPLETE_SUPPORT_PREFLIGHT_REQUIRED"}
    _write_json_new(campaign_root / "registration.json", blinded)
    _write_json_new(campaign_root / "campaign.json", campaign_meta)
    return campaign_meta


def verify_support_preflight(*, campaign_root: Path, receipt_path: Path,
                             authorization_path: Path) -> dict[str, Any]:
    campaign_root = Path(campaign_root).absolute()
    metadata, registration = _load_campaign(campaign_root)
    _validate_authorization(authorization_path, scope="SUPPORT_PREFLIGHT", campaign_id=metadata["campaign_id"],
        bindings={"campaign_registration_sha256": metadata["campaign_registration_sha256"]})
    receipt_bytes = Path(receipt_path).read_bytes()
    receipt = validate_support_preflight(json.loads(receipt_bytes), campaign_id=metadata["campaign_id"])
    from .fs4_r2_security import WindowsPrivateStore
    support_root = campaign_root / "support-preflight"
    support_store = WindowsPrivateStore(support_root, create=True)
    support_store.write("receipt.json", receipt_bytes)
    pinned_path = support_root / "receipt.json"
    from .fs4_joint_campaign_contract import build_trajectory_manifests
    source_commit = metadata["source_freeze_commit"]
    _check_committed_sources(source_commit)
    manifests = build_trajectory_manifests(registration=registration, campaign_id=metadata["campaign_id"],
        campaign_registration_sha256=metadata["campaign_registration_sha256"],
        support_preflight_receipt=receipt, support_preflight_receipt_path=pinned_path,
        r7_seed0_order_sha256=metadata["r7_seed0_order_sha256"],
        source_manifest_identity_sha256=metadata["source_manifest_identity_sha256"],
        source_commit=source_commit)
    manifest_dir = campaign_root / "manifests"
    manifest_dir.mkdir(parents=False, exist_ok=False)
    manifest_hashes = {}
    for trajectory in ARM_IDS:
        manifest_hashes[trajectory] = _write_json_new(manifest_dir / f"{trajectory}.json", manifests[trajectory])
    metadata.update({"support_preflight_receipt_sha256": _sha(receipt_bytes),
        "support_preflight_receipt_path": str(pinned_path), "trajectory_manifests": manifest_hashes,
        "status": "SUPPORT_PREFLIGHT_VERIFIED_RESOURCE_PREFLIGHT_REQUIRED"})
    # campaign.json is an append-only receipt; the update is captured separately.
    _write_json_new(campaign_root / "support-preflight-binding.json", metadata)
    return {"status": "PASS", "campaign_id": metadata["campaign_id"],
            "campaign_registration_sha256": metadata["campaign_registration_sha256"],
            "support_preflight_receipt_sha256": _sha(receipt_bytes), "trajectory_manifest_sha256": manifest_hashes}


def _campaign_with_support(campaign_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata, registration = _load_campaign(campaign_root)
    binding_path = Path(campaign_root) / "support-preflight-binding.json"
    if not binding_path.is_file():
        raise CampaignContractError("campaign-bound four-Support preflight is not verified")
    binding = _read_json(binding_path)
    if binding.get("campaign_registration_sha256") != metadata["campaign_registration_sha256"]:
        raise CampaignContractError("Support preflight binding belongs to another registration")
    return binding, registration


def resource_preflight(*, campaign_root: Path, benchmark_receipt_path: Path | None,
                       authorization_path: Path) -> dict[str, Any]:
    campaign_root = Path(campaign_root).absolute()
    metadata, registration = _campaign_with_support(campaign_root)
    _validate_authorization(authorization_path, scope="RESOURCE_PREFLIGHT", campaign_id=metadata["campaign_id"],
        bindings={"campaign_registration_sha256": metadata["campaign_registration_sha256"],
                  "support_preflight_receipt_sha256": metadata["support_preflight_receipt_sha256"]})
    benchmark = _read_json(benchmark_receipt_path) if benchmark_receipt_path else None
    from .fs4_joint_campaign_resources import resource_preflight as check_resources
    report = check_resources(registration=registration, image_root=Path(metadata["image_root"]),
        output_root=campaign_root / "execution", private_root=Path(metadata["private_root"]),
        benchmark_receipt=benchmark)
    report["campaign_id"] = metadata["campaign_id"]
    report["support_preflight_receipt_sha256"] = metadata["support_preflight_receipt_sha256"]
    path = campaign_root / "resource-preflight.json"
    _write_json_new(path, report)
    return report


def _load_trajectory_manifest(campaign_root: Path, trajectory: str) -> dict[str, Any]:
    binding, _registration = _campaign_with_support(campaign_root)
    path = Path(campaign_root) / "manifests" / f"{trajectory}.json"
    manifest = _read_json(path)
    if _sha(canonical(manifest) + b"\n") != binding["trajectory_manifests"].get(trajectory):
        raise CampaignContractError(f"{trajectory}: bound trajectory manifest digest differs")
    return manifest


def _inspect_ledger(run_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    if not run_root.exists():
        return None
    from .fs4_research_ledger import ResearchLedger
    ledger = ResearchLedger(run_root, manifest, create=False)
    try:
        return {"position": ledger.position, "phase": ledger.phase, "current_category": ledger.current_category,
                "category_seals": dict(ledger.category_seals)}
    finally:
        ledger.close()


def _checkpoint_candidates(run_root: Path, category: str, manifest: Mapping[str, Any],
                           category_start: int, category_end: int) -> list[tuple[int, str]]:
    checkpoint_root = run_root / "checkpoints"
    if not checkpoint_root.exists():
        return []
    from .fs4_r2_security import WindowsPrivateStore, validate_private_acl
    store = WindowsPrivateStore(checkpoint_root, create=False)
    candidates = []
    for child in checkpoint_root.iterdir():
        validate_private_acl(child, directory=True)
        if not child.is_dir():
            raise CampaignContractError("checkpoint store contains an unexpected object")
        for filename in ("state.json", "tensors.npz", "COMMITTED.json"):
            validate_private_acl(child / filename, directory=False)
        marker = _read_json(child / "COMMITTED.json")
        state_bytes = (child / "state.json").read_bytes()
        tensor_bytes = (child / "tensors.npz").read_bytes()
        state = json.loads(state_bytes)
        if (marker != {"state_sha256": _sha(state_bytes), "tensor_sha256": _sha(tensor_bytes)} or
                state.get("manifest_sha256") != digest(manifest) or
                state.get("tensor_sha256") != marker["tensor_sha256"]):
            raise CampaignContractError("checkpoint inventory has a corrupt or wrong-manifest entry")
        next_position = state.get("ledger_offset", {}).get("next_position")
        if not isinstance(next_position, int) or child.name.startswith(category + "-") is False:
            continue
        if category_start <= next_position <= category_end:
            candidates.append((next_position, child.name))
    return sorted(candidates)


def _append_telemetry(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(dict(row)) + b"\n"
    if not path.exists():
        _write_new(path, data)
        return
    with path.open("ab", buffering=0) as stream:
        if stream.write(data) != len(data):
            raise OSError("short runtime telemetry append")
        os.fsync(stream.fileno())


def _start_worker(*, campaign_root: Path, metadata: Mapping[str, Any], trajectory: str,
                  manifest_path: Path, manifest: Mapping[str, Any], category: str,
                  checkpoint: str | None, create: bool):
    from .fs4_r4_worker import BrokerProcess, BlindedResearchWorker
    run_root = campaign_root / "execution" / "runs" / trajectory
    disclosure_root = campaign_root / "execution" / "disclosures" / trajectory
    run_root.parent.mkdir(parents=True, exist_ok=True)
    disclosure_root.parent.mkdir(parents=True, exist_ok=True)
    broker = BrokerProcess(private_root=Path(metadata["private_root"]), run_root=run_root,
        disclosures_root=disclosure_root, manifest_path=manifest_path)
    config = {"manifest_path": str(manifest_path.absolute()), "run_root": str(run_root.absolute()),
        "category": category, "create": create, "checkpoint": checkpoint}
    config_dir = campaign_root / "execution" / "runtime-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{trajectory}-{category}-{uuid.uuid4().hex}.json"
    _write_json_new(config_path, config)
    try:
        worker = BlindedResearchWorker(config_path, broker)
    except Exception:
        try:
            config_path.unlink()
        except OSError:
            pass
        raise
    return worker, config_path, run_root, disclosure_root


def run_stage_a(*, campaign_root: Path, authorization_path: Path, resume: bool = False) -> dict[str, Any]:
    campaign_root = Path(campaign_root).absolute()
    binding, registration = _campaign_with_support(campaign_root)
    resource_path = campaign_root / "resource-preflight.json"
    if not resource_path.is_file():
        raise CampaignContractError("resource preflight receipt is required before Stage A")
    resource = _read_json(resource_path)
    if resource.get("status") != "PASS" or resource.get("spec_sha256") != spec_identity() or resource.get(
            "registration_sha256") != digest(registration):
        raise CampaignContractError("BLOCKED_RESOURCE_PREFLIGHT: immutable resource receipt does not pass")
    manifest_hashes = binding["trajectory_manifests"]
    _validate_authorization(authorization_path, scope="STAGE_A", campaign_id=binding["campaign_id"],
        bindings={"campaign_registration_sha256": binding["campaign_registration_sha256"],
                  "support_preflight_receipt_sha256": binding["support_preflight_receipt_sha256"],
                  "resource_preflight_receipt_sha256": _sha(resource_path.read_bytes()),
                  "trajectory_manifests": manifest_hashes})
    execution_root = campaign_root / "execution"
    if not execution_root.exists():
        execution_root.mkdir(parents=False, exist_ok=False)
    elif not resume:
        raise FileExistsError("execution namespace exists; pass explicit resume authorization to continue")
    try:
        for trajectory in ARM_IDS:
            manifest_path = campaign_root / "manifests" / f"{trajectory}.json"
            manifest = _load_trajectory_manifest(campaign_root, trajectory)
            if manifest["source_commit"] != binding["source_freeze_commit"]:
                raise CampaignContractError("trajectory source freeze commit differs from campaign staging")
            run_root = execution_root / "runs" / trajectory
            existing = _inspect_ledger(run_root, manifest)
            if existing and existing["phase"] == "GLOBAL_A":
                continue
            parent_ledger = existing
            for category in EXPECTED_CATEGORIES:
                if parent_ledger and category in parent_ledger["category_seals"]:
                    continue
                events = registration["categories"][category]["events"]
                category_start, category_end = events[0]["position"], events[-1]["position"] + 1
                current = _inspect_ledger(run_root, manifest)
                if current and current["current_category"] not in (None, category):
                    raise CampaignContractError("ledger is active in a different category; refusing to switch trajectories")
                if current and current["phase"] not in ("READY",):
                    # An unfinished query intent is handled only by checkpoint replay.
                    pass
                candidates = _checkpoint_candidates(run_root, category, manifest, category_start, category_end)
                checkpoint = candidates[-1][1] if candidates else None
                if checkpoint and current and current["position"] < candidates[-1][0]:
                    raise CampaignContractError("checkpoint is ahead of its durable ledger")
                if current and current["position"] > category_start and not checkpoint:
                    raise CampaignContractError("advanced ledger has no committed category checkpoint")
                create = current is None
                worker, config_path, run_root, disclosure_root = _start_worker(
                    campaign_root=campaign_root, metadata=binding, trajectory=trajectory,
                    manifest_path=manifest_path, manifest=manifest, category=category,
                    checkpoint=checkpoint, create=create)
                telemetry_path = execution_root / "telemetry" / f"{trajectory}.jsonl"
                try:
                    if checkpoint is None:
                        worker.checkpoint(f"{category}-start")
                    if checkpoint is None:
                        position = category_start
                    else:
                        checkpoint_state = _read_json(run_root / "checkpoints" / checkpoint / "state.json")
                        position = checkpoint_state["ledger_offset"]["next_position"]
                    for event_position in range(position, category_end):
                        result = worker.step(event_position)
                        expected = events[event_position - category_start]
                        prediction = result.get("prediction", {})
                        if (prediction.get("position") != event_position or prediction.get("event") != expected["event"] or
                                prediction.get("group") != expected["group"]):
                            raise CampaignContractError("worker returned a different registered event identity")
                        completed = event_position + 1
                        if completed % 1000 == 0:
                            name = f"{category}-event-{completed:05d}"
                            if name not in {item[1] for item in _checkpoint_candidates(
                                    run_root, category, manifest, category_start, category_end)}:
                                worker.checkpoint(name)
                            _append_telemetry(telemetry_path, {"schema": "FS4_CAMPAIGN_RUNTIME_TELEMETRY_V1",
                                "trajectory_id": trajectory, "category": category, "completed_global_events": completed,
                                "captured_unix_time": time.time(), **worker.stats()})
                        # The event result is intentionally discarded after identity checks.
                        del result
                    end_name = f"{category}-end"
                    if end_name not in {item[1] for item in _checkpoint_candidates(
                            run_root, category, manifest, category_start, category_end)}:
                        worker.checkpoint(end_name)
                    worker.seal_category()
                    if category == EXPECTED_CATEGORIES[-1]:
                        worker.seal_global_a()
                    _append_telemetry(telemetry_path, {"schema": "FS4_CAMPAIGN_RUNTIME_TELEMETRY_V1",
                        "trajectory_id": trajectory, "category": category, "completed_global_events": category_end,
                        "captured_unix_time": time.time(), **worker.stats()})
                finally:
                    worker.close()
                    try:
                        config_path.unlink()
                    except OSError:
                        pass
                parent_ledger = _inspect_ledger(run_root, manifest)
        projections = {trajectory: _verify_trajectory_stage_a(campaign_root, trajectory, write_projection=True)
                       for trajectory in ARM_IDS}
        _validate_complete_stage_a(campaign_root, binding, registration, projections)
        seal = {"schema": "FS4_FINAL_JOINT_STAGE_A_CAMPAIGN_SEAL_V1", "status": "STAGE_A_COMPLETE",
            "campaign_id": binding["campaign_id"], "scientific_spec_sha256": spec_identity(),
            "campaign_registration_sha256": binding["campaign_registration_sha256"],
            "trajectory_seals": {key: {"run_id": projections[key]["run_id"],
                "manifest_sha256": projections[key]["manifest_sha256"], "stage_a_sha256": projections[key]["stage_a_sha256"]}
                for key in ARM_IDS}}
        _write_json_new(execution_root / "STAGE_A_CAMPAIGN_SEAL.json", seal)
        return seal
    except Exception:
        _record_incomplete(execution_root, "STAGE_A_INTEGRITY_OR_EXECUTION_FAILURE")
        raise


def _verify_trajectory_stage_a(campaign_root: Path, trajectory: str, *, write_projection: bool) -> dict[str, Any]:
    run_root = Path(campaign_root) / "execution" / "runs" / trajectory
    command = [sys.executable, "-m", "experiments.calibration.fs4_research_stage_a_verify", str(run_root)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
    projection = json.loads(completed.stdout)
    if projection.get("trajectory_id") != trajectory:
        raise CampaignContractError("Stage A verifier returned another trajectory")
    if write_projection:
        path = Path(campaign_root) / "execution" / "sealed-projections" / f"{trajectory}.json"
        if not path.exists():
            _write_json_new(path, projection)
        elif _read_json(path) != projection:
            raise CampaignContractError("existing sealed projection differs; refusing overwrite")
    return projection


def _validate_complete_stage_a(campaign_root: Path, binding: Mapping[str, Any], registration: Mapping[str, Any],
                               projections: Mapping[str, Mapping[str, Any]]) -> None:
    if set(projections) != set(ARM_IDS):
        raise CampaignContractError("all three Stage A trajectories are mandatory")
    reference = None
    for trajectory in ARM_IDS:
        item = projections[trajectory]
        if (item.get("campaign_id") != binding["campaign_id"] or
                item.get("campaign_registration_sha256") != binding["campaign_registration_sha256"] or
                item.get("scientific_spec_sha256") != spec_identity() or
                item.get("source_commit") != binding["source_freeze_commit"] or
                item.get("support_preflight_receipt_sha256") != binding["support_preflight_receipt_sha256"] or
                item.get("r7_seed0_order_sha256") != binding["r7_seed0_order_sha256"] or
                item.get("source_mapping_sha256") != binding["source_mapping_sha256"] or
                item.get("model_state_sha256") != load_spec()["source_model_support"]["model"]["model_state_sha256"]):
            raise CampaignContractError(f"{trajectory}: Stage A campaign identity differs")
        frozen_model = load_spec()["source_model_support"]["model"]
        if (item.get("checkpoint_sha256") != frozen_model["checkpoint_sha256"] or
                item.get("hub_source_tree_sha256") != frozen_model["hub_source_tree_sha256"] or
                item.get("scorer_config_sha256") != frozen_model["native_config_sha256"] or
                item.get("preflight_receipt_sha256") != load_spec()["source_model_support"]["historical_receipt_sha256"]):
            raise CampaignContractError(f"{trajectory}: frozen model or native Support receipt differs")
        if item.get("fixed_support_sha256_by_category") != {
                category: load_spec()["source_model_support"]["support_records_by_category"][category]["fixed_bank_sha256"]
                for category in EXPECTED_CATEGORIES}:
            raise CampaignContractError(f"{trajectory}: four-Support fixed bank identity differs")
        if item.get("stream_sha256_by_category") != {
                category: digest([{"group": row["group"], "event": row["event"]}
                                 for row in registration["categories"][category]["events"]])
                for category in EXPECTED_CATEGORIES}:
            raise CampaignContractError(f"{trajectory}: Stage A stream identity differs")
        rows = [(row["event"], row["group"], row["position"], row["view"]) for row in item["events"]]
        if reference is None:
            reference = rows
        elif rows != reference:
            raise CampaignContractError("Stage A trajectories do not share the same opaque event sequence")
        if len(item["events"]) != sum(EXPECTED_EVENTS.values()):
            raise CampaignContractError(f"{trajectory}: Stage A event count is incomplete")


def verify_stage_a(*, campaign_root: Path) -> dict[str, Any]:
    binding, registration = _campaign_with_support(campaign_root)
    projections = {trajectory: _verify_trajectory_stage_a(campaign_root, trajectory, write_projection=True)
                   for trajectory in ARM_IDS}
    _validate_complete_stage_a(campaign_root, binding, registration, projections)
    return {"status": "PASS", "campaign_id": binding["campaign_id"],
        "trajectory_seals": {key: {"stage_a_sha256": projections[key]["stage_a_sha256"],
            "manifest_sha256": projections[key]["manifest_sha256"]} for key in ARM_IDS}}


def _validate_stage_b_grant(grant_path: Path, *, binding: Mapping[str, Any],
                            projections: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    seals = {trajectory: {"manifest_sha256": projections[trajectory]["manifest_sha256"],
        "stage_a_sha256": projections[trajectory]["stage_a_sha256"],
        "run_id": projections[trajectory]["run_id"]} for trajectory in ARM_IDS}
    grant = _validate_authorization(grant_path, scope="STAGE_B", campaign_id=binding["campaign_id"],
        bindings={"campaign_registration_sha256": binding["campaign_registration_sha256"],
                  "source_mapping_sha256": binding["source_mapping_sha256"], "trajectory_seals": seals})
    return grant


def run_stage_b(*, campaign_root: Path, grant_path: Path, result_root: Path) -> dict[str, Any]:
    from .fs4_joint_campaign_evaluator import evaluate_campaign, canonical_bytes, sha256_bytes
    from .fs4_r2_security import WindowsPrivateStore
    campaign_root, result_root = Path(campaign_root).absolute(), Path(result_root).absolute()
    binding, registration = _campaign_with_support(campaign_root)
    if result_root.exists():
        raise FileExistsError("Stage B result root must be fresh and non-overwriting")
    projections = {trajectory: _verify_trajectory_stage_a(campaign_root, trajectory, write_projection=False)
                   for trajectory in ARM_IDS}
    _validate_complete_stage_a(campaign_root, binding, registration, projections)
    _validate_stage_b_grant(grant_path, binding=binding, projections=projections)
    # The full outcome map is opened only after the exact twelve sealed category
    # streams, common registration and separate Stage B authorization pass.
    stage_s = projections["S"]
    run_root = campaign_root / "execution" / "runs" / "S"
    projected = subprocess.run([sys.executable, "-m", "experiments.calibration.fs4_r4_mapping_verify",
        str(binding["private_root"]), str(run_root), stage_s["manifest_sha256"]],
        cwd=ROOT, capture_output=True, check=True)
    outcome_projection = json.loads(projected.stdout)
    labels = outcome_projection.get("labels")
    if (outcome_projection.get("source_mapping_sha256") != binding["source_mapping_sha256"] or
            not isinstance(labels, dict) or
            any(any(labels.get(event) != label for event, label in projections[trajectory]["queried_feedback"].items())
                for trajectory in ARM_IDS)):
        raise CampaignContractError("complete outcome mapping or one branch's queried feedback differs")
    telemetry_by_arm = {}
    for trajectory in ARM_IDS:
        telemetry_path = campaign_root / "execution" / "telemetry" / f"{trajectory}.jsonl"
        if not telemetry_path.is_file():
            raise CampaignContractError(f"{trajectory}: runtime/resource telemetry is missing")
        rows = [json.loads(line) for line in telemetry_path.read_bytes().splitlines()]
        if {row.get("category") for row in rows} != set(EXPECTED_CATEGORIES):
            raise CampaignContractError(f"{trajectory}: per-category runtime telemetry is incomplete")
        telemetry_by_arm[trajectory] = rows
    for other in (campaign_root, Path(binding["private_root"]).absolute(), Path(binding["image_root"]).absolute()):
        if result_root == other or result_root.is_relative_to(other) or other.is_relative_to(result_root):
            raise CampaignContractError("Stage B results must be isolated from campaign, mapping and image stores")
    result_store = WindowsPrivateStore(result_root, create=True)
    result_store.write("INCOMPLETE.json", canonical({"schema": "FS4_FINAL_JOINT_STAGE_B_INCOMPLETE_V1",
        "status": "INCOMPLETE"}) + b"\n")
    files: dict[str, str] = {}
    for trajectory in ARM_IDS:
        data = canonical_bytes(projections[trajectory]) + b"\n"
        result_store.write(f"stage_a_{trajectory}.json", data)
        files[f"stage_a_{trajectory}.json"] = sha256_bytes(data)
        telemetry_data = canonical_bytes(telemetry_by_arm[trajectory]) + b"\n"
        result_store.write(f"telemetry_{trajectory}.json", telemetry_data)
        files[f"telemetry_{trajectory}.json"] = sha256_bytes(telemetry_data)
    labels_data = canonical_bytes(outcome_projection) + b"\n"
    result_store.write("outcome_projection.json", labels_data)
    files["outcome_projection.json"] = sha256_bytes(labels_data)
    report = evaluate_campaign(projections, labels, telemetry_by_arm)
    report_data = canonical_bytes(report) + b"\n"
    result_store.write("metrics.json", report_data)
    files["metrics.json"] = sha256_bytes(report_data)
    grant_data = Path(grant_path).read_bytes()
    result_store.write("stage_b_authorization.json", grant_data)
    files["stage_b_authorization.json"] = sha256_bytes(grant_data)
    campaign_result = {"schema": "FS4_FINAL_JOINT_CAMPAIGN_RESULT_V1", "status": "COMPLETE",
        "campaign_id": binding["campaign_id"], "scientific_spec_sha256": spec_identity(),
        "campaign_registration_sha256": binding["campaign_registration_sha256"],
        "source_mapping_sha256": binding["source_mapping_sha256"],
        "trajectory_seals": {key: {"manifest_sha256": projections[key]["manifest_sha256"],
            "stage_a_sha256": projections[key]["stage_a_sha256"], "run_id": projections[key]["run_id"]}
            for key in ARM_IDS}, "metrics_sha256": sha256_bytes(report_data)}
    campaign_result_data = canonical_bytes(campaign_result) + b"\n"
    result_store.write("campaign_result.json", campaign_result_data)
    files["campaign_result.json"] = sha256_bytes(campaign_result_data)
    complete = {"schema": "FS4_FINAL_JOINT_CAMPAIGN_COMPLETE_V1", "status": "COMPLETE",
        "campaign_result_sha256": sha256_bytes(campaign_result_data), "files_sha256": files}
    result_store.write("COMPLETE.json", canonical_bytes(complete) + b"\n")
    from .fs4_r2_security import validate_private_acl
    incomplete_marker = result_root / "INCOMPLETE.json"
    validate_private_acl(incomplete_marker, directory=False)
    incomplete_marker.unlink()
    return verify_complete(result_root)


def verify_complete(result_root: Path) -> dict[str, Any]:
    from .fs4_joint_campaign_evaluator import evaluate_campaign, canonical_bytes, sha256_bytes
    from .fs4_r2_security import WindowsPrivateStore
    store = WindowsPrivateStore(Path(result_root), create=False)
    complete = json.loads(store.read("COMPLETE.json"))
    if complete.get("schema") != "FS4_FINAL_JOINT_CAMPAIGN_COMPLETE_V1" or complete.get("status") != "COMPLETE":
        raise CampaignContractError("Stage B completion seal is missing")
    expected_files = {*(f"stage_a_{arm}.json" for arm in ARM_IDS),
        *(f"telemetry_{arm}.json" for arm in ARM_IDS), "outcome_projection.json", "metrics.json",
        "stage_b_authorization.json", "campaign_result.json"}
    if set(complete.get("files_sha256", {})) != expected_files:
        raise CampaignContractError("Stage B completion seal has an unexpected file inventory")
    loaded = {}
    for name, expected in complete["files_sha256"].items():
        data = store.read(name)
        if _sha(data) != expected:
            raise CampaignContractError(f"Stage B file checksum differs: {name}")
        loaded[name] = json.loads(data)
    result = loaded["campaign_result.json"]
    if _sha(store.read("campaign_result.json")) != complete["campaign_result_sha256"]:
        raise CampaignContractError("campaign result receipt checksum differs")
    stage_a = {trajectory: loaded[f"stage_a_{trajectory}.json"] for trajectory in ARM_IDS}
    telemetry = {trajectory: loaded[f"telemetry_{trajectory}.json"] for trajectory in ARM_IDS}
    outcome_projection = loaded["outcome_projection.json"]
    labels = outcome_projection.get("labels")
    expected_seals = {trajectory: {"manifest_sha256": stage_a[trajectory]["manifest_sha256"],
        "stage_a_sha256": stage_a[trajectory]["stage_a_sha256"], "run_id": stage_a[trajectory]["run_id"]}
        for trajectory in ARM_IDS}
    if (result.get("trajectory_seals") != expected_seals or
            result.get("source_mapping_sha256") != outcome_projection.get("source_mapping_sha256") or
            outcome_projection.get("manifest_sha256") != stage_a["S"].get("manifest_sha256") or
            any(stage_a[trajectory].get("source_mapping_sha256") != result.get("source_mapping_sha256") or
                stage_a[trajectory].get("campaign_id") != result.get("campaign_id") or
                stage_a[trajectory].get("scientific_spec_sha256") != result.get("scientific_spec_sha256") or
                stage_a[trajectory].get("campaign_registration_sha256") != result.get("campaign_registration_sha256")
                for trajectory in ARM_IDS)):
        raise CampaignContractError("final outcome projection, campaign registration or Stage A seals do not agree")
    grant = loaded["stage_b_authorization.json"]
    expected_grant_keys = {"schema", "authorization_id", "authorized_by", "authorized_at", "scope",
        "campaign_id", "scientific_spec_sha256", "authorization_statement", "campaign_registration_sha256",
        "source_mapping_sha256", "trajectory_seals"}
    if (set(grant) != expected_grant_keys or grant.get("schema") != AUTH_SCHEMA or grant.get("scope") != "STAGE_B" or
            grant.get("campaign_id") != result.get("campaign_id") or
            grant.get("scientific_spec_sha256") != result.get("scientific_spec_sha256") or
            grant.get("campaign_registration_sha256") != result.get("campaign_registration_sha256") or
            grant.get("source_mapping_sha256") != result.get("source_mapping_sha256") or
            grant.get("trajectory_seals") != expected_seals or
            any(not isinstance(grant.get(key), str) or not grant[key].strip()
                for key in ("authorization_id", "authorized_by", "authorized_at", "authorization_statement"))):
        raise CampaignContractError("stored Stage B authorization no longer matches the completed campaign")
    if (not isinstance(labels, dict) or
            any(any(labels.get(event) != label for event, label in stage_a[trajectory]["queried_feedback"].items())
                for trajectory in ARM_IDS)):
        raise CampaignContractError("final outcome labels disagree with a trajectory's verified feedback")
    recomputed = evaluate_campaign(stage_a, labels, telemetry)
    if canonical_bytes(recomputed) + b"\n" != store.read("metrics.json"):
        raise CampaignContractError("stored metrics do not reproduce from the sealed Stage A projections and labels")
    if result.get("metrics_sha256") != _sha(store.read("metrics.json")):
        raise CampaignContractError("campaign result metric checksum differs")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", nargs="?", default="freeze-only", choices=("freeze-only", "validate-contract",
        "prepare-real-data", "verify-support-preflight", "resource-preflight", "run-stage-a",
        "verify-stage-a", "run-stage-b", "verify-complete"))
    parser.add_argument("--campaign-id")
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--private-root", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--original-registration", type=Path)
    parser.add_argument("--original-ledger", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--support-receipt", type=Path)
    parser.add_argument("--benchmark-receipt", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--grant", type=Path)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "freeze-only": result = freeze_only()
        elif args.operation == "validate-contract": result = validate_contract()
        elif args.operation == "prepare-real-data":
            required = (args.campaign_id, args.campaign_root, args.private_root, args.image_root,
                        args.original_registration, args.original_ledger, args.labels, args.authorization)
            if any(item is None for item in required): raise ValueError("prepare-real-data requires all staging inputs and authorization")
            result = prepare_real_data(campaign_id=args.campaign_id, campaign_root=args.campaign_root,
                private_root=args.private_root, image_root=args.image_root,
                original_registration_path=args.original_registration, original_ledger_path=args.original_ledger,
                labels_path=args.labels, authorization_path=args.authorization)
        elif args.operation == "verify-support-preflight":
            if any(item is None for item in (args.campaign_root, args.support_receipt, args.authorization)):
                raise ValueError("verify-support-preflight requires campaign root, receipt and authorization")
            result = verify_support_preflight(campaign_root=args.campaign_root, receipt_path=args.support_receipt,
                                              authorization_path=args.authorization)
        elif args.operation == "resource-preflight":
            if args.campaign_root is None or args.authorization is None: raise ValueError("resource-preflight requires campaign root and authorization")
            result = resource_preflight(campaign_root=args.campaign_root, benchmark_receipt_path=args.benchmark_receipt,
                                        authorization_path=args.authorization)
        elif args.operation == "run-stage-a":
            if args.campaign_root is None or args.authorization is None: raise ValueError("run-stage-a requires campaign root and authorization")
            result = run_stage_a(campaign_root=args.campaign_root, authorization_path=args.authorization, resume=args.resume)
        elif args.operation == "verify-stage-a":
            if args.campaign_root is None: raise ValueError("verify-stage-a requires campaign root")
            result = verify_stage_a(campaign_root=args.campaign_root)
        elif args.operation == "run-stage-b":
            if any(item is None for item in (args.campaign_root, args.grant, args.result_root)):
                raise ValueError("run-stage-b requires campaign root, separate grant and fresh result root")
            result = run_stage_b(campaign_root=args.campaign_root, grant_path=args.grant, result_root=args.result_root)
        else:
            if args.result_root is None: raise ValueError("verify-complete requires result root")
            result = verify_complete(args.result_root)
        sys.stdout.buffer.write(canonical(result) + b"\n")
        return 0
    except Exception as exc:
        print(f"FS4_CAMPAIGN_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
