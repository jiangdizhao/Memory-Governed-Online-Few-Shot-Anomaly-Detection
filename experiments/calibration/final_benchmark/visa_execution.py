"""Bounded VisA 1cls P0 registration, Stage A and post-seal Stage B commands."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from .execution_support import AttemptStatus, sha256_file
from .visa_registration import (CATEGORIES, EXPECTED_TEST_COUNTS, VisaRegistrationError,
                                build_visa_registration)


class VisaExecutionError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_exclusive(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical(payload) + b"\n"
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise VisaExecutionError(f"refusing to overwrite existing VisA artifact {path.name}") from None
    return hashlib.sha256(data).hexdigest()


def _current_head(repository: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                            check=True, capture_output=True, text=True)
    value = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise VisaExecutionError("current source identity is not a lowercase Git object ID")
    return value


def register(*, dataset_root: str | Path, identity_manifest: str | Path,
             output_root: str | Path, researcher_decision_sha256: str,
             repository: str | Path | None = None) -> dict[str, Any]:
    repo = Path(repository or Path.cwd()).resolve(strict=True)
    head = _current_head(repo)
    try:
        result = build_visa_registration(dataset_root=dataset_root,
            identity_manifest_path=identity_manifest, output_root=output_root,
            researcher_decision_sha256=researcher_decision_sha256,
            source_commit=head)
    except VisaRegistrationError as exc:
        raise VisaExecutionError(str(exc)) from exc
    return result


def create_source_bound_plan(*, registration_root: str | Path,
                             category: str = "cashew",
                             researcher_decision_sha256: str | None = None,
                             repository: str | Path | None = None) -> dict[str, Any]:
    root = Path(registration_root).resolve(strict=True)
    repo = Path(repository or Path.cwd()).resolve(strict=True)
    head = _current_head(repo)
    if category not in CATEGORIES:
        raise VisaExecutionError("source-bound VisA plan category is not in the registered 1cls inventory")
    requested_category = category
    index = json.loads((root / "registration_index.json").read_text(encoding="utf-8"))
    registration_sha = index.pop("registration_sha256", None)
    if (_sha(index) != registration_sha or index.get("source_commit") != head or
            index.get("total_test_images") != 2162 or index.get("test_counts") != {
                "candle": 200, "capsules": 160, "cashew": 150, "chewinggum": 150,
                "fryum": 150, "macaroni1": 200, "macaroni2": 200, "pcb1": 200,
                "pcb2": 200, "pcb3": 201, "pcb4": 201, "pipe_fryum": 150}):
        raise VisaExecutionError("VisA 12-category registration index is invalid or bound to another source")
    decision = json.loads((root / "private" / "researcher_decision_binding.json").read_text(encoding="utf-8"))
    if (decision.get("schema") != "final-benchmark.visa-researcher-decision-binding.v1" or
            decision.get("publisher_authentication") is not False or
            decision.get("independent_review_claimed") is not False or
            decision.get("pixel_mask_access") is not False):
        raise VisaExecutionError("researcher authorization provenance record is invalid")
    if requested_category == "cashew":
        if researcher_decision_sha256 not in (None, decision.get("researcher_decision_sha256")):
            raise VisaExecutionError("cashew plan must retain the explicitly recorded current researcher decision")
        decision_sha = decision["researcher_decision_sha256"]
    else:
        if not re.fullmatch(r"[0-9a-f]{64}", str(researcher_decision_sha256 or "")):
            raise VisaExecutionError("other VisA categories require a separate researcher decision digest")
        decision_sha = str(researcher_decision_sha256)
    category_records = []
    for item in index["categories"]:
        registered_category = item["category"]
        stream = json.loads((root / item["stream_manifest_path"]).read_text(encoding="utf-8"))
        role = json.loads((root / item["role_manifest_path"]).read_text(encoding="utf-8"))
        candidate = json.loads((root / item["unit_candidate_path"]).read_text(encoding="utf-8"))
        release = json.loads((root / item["release_identity_path"]).read_text(encoding="utf-8"))
        stream_unsigned = dict(stream)
        stream_sha = stream_unsigned.pop("stream_manifest_sha256", None)
        candidate_unsigned = dict(candidate)
        candidate_sha = candidate_unsigned.pop("candidate_sha256", None)
        if (stream_sha != _sha(stream_unsigned) or candidate_sha != _sha(candidate_unsigned) or
                stream.get("source_commit") != head or stream.get("category") != registered_category or
                stream.get("role_manifest_sha256") != _sha(role) or
                release.get("source_commit") != head):
            raise VisaExecutionError(f"VisA {registered_category} source-bound registration is inconsistent")
        category_records.append({"category": registered_category, "test_image_count": len(stream["stream_order"]),
            "role_manifest_sha256": _sha(role), "stream_manifest_sha256": stream_sha,
            "unit_candidate_sha256": candidate_sha,
            "release_identity_sha256": _sha(release)})
    plan = {"schema": "final-benchmark.visa-p0-execution-plan.v1",
        "source_commit": head, "dataset_id": "visa_1cls",
        "dataset_release_sha256": index["dataset_release_sha256"],
        "identity_manifest_sha256": index["identity_manifest_sha256"],
        "source_artifacts_sha256": index["source_artifacts_sha256"],
        "researcher_decision_sha256": decision_sha,
        "category_registrations": category_records,
        "registered_categories": list(CATEGORIES),
        "registered_test_event_count": 2162,
        "support_allocation": {"support_seed": 0, "support_count": 4, "c64_count": 64},
        "conditions": ["P0"], "query_rate": 0.10,
        "static_references": ["STATIC-4+64", "STATIC-36+32"],
        "execution_scope": {"Stage_A": [f"{requested_category}/P0"],
                            "Stage_B": [f"{requested_category}/P0/post-seal"]},
        "pixel_masks_allowed": False, "publisher_authentication_claimed": False,
        "physical_independence_claim": "NOT_ESTABLISHED",
        "authorization_provenance": "explicit researcher task; not an independent-review grant"}
    plan["execution_plan_sha256"] = _sha(plan)
    plan_path = root / "private" / requested_category / "execution_plan.json"
    plan_file_sha = _write_exclusive(plan_path, plan)
    registration = next(row for row in category_records if row["category"] == requested_category)
    result = {"execution_plan_path": str(plan_path),
        "execution_plan_sha256": plan["execution_plan_sha256"],
        "execution_plan_file_sha256": plan_file_sha,
        "source_commit": head, "plan_scope": [f"{requested_category}/P0"],
        "registered_test_images": 2162}
    if requested_category != "cashew":
        result.update({"authorization_path": None,
                       "authorization_status": "MISSING_REQUIRES_SEPARATE_RESEARCHER_AUTHORIZATION"})
        return result
    authorization = {
        "schema": "final-benchmark.visa-unit-execution-authorization.v1",
        "status": "RESEARCHER_APPROVED_VISA_ATOMIC_UNIT_EXECUTION",
        "scope": f"{requested_category}_P0_stage_a_only",
        "dataset_id": "visa_1cls", "category": requested_category, "condition_id": "P0",
        "source_commit": head, "execution_plan_sha256": plan["execution_plan_sha256"],
        "execution_plan_file_sha256": plan_file_sha,
        "identity_manifest_sha256": index["identity_manifest_sha256"],
        "role_manifest_sha256": registration["role_manifest_sha256"],
        "dataset_release_sha256": index["dataset_release_sha256"],
        "dataset_release_identity_sha256": registration["release_identity_sha256"],
        "stream_manifest_sha256": registration["stream_manifest_sha256"],
        "unit_candidate_sha256": registration["unit_candidate_sha256"],
        "researcher_decision_sha256": decision_sha,
        "physical_independence_claim": "NOT_ESTABLISHED",
        "expected_event_count": EXPECTED_TEST_COUNTS[requested_category], "query_rate": 0.10,
        "label_access": "QUERY_ONLY", "pixel_masks_allowed": False,
        "stage_b_authorized": False,
        "authorization_provenance": "explicit researcher task; not independent-review approval"}
    auth_path = root / "private" / requested_category / "execution_authorization.json"
    auth_file_sha = _write_exclusive(auth_path, authorization)
    result.update({"authorization_path": str(auth_path),
                   "authorization_file_sha256": auth_file_sha,
                   "authorization_status": authorization["status"]})
    return result


def _load_bundle(root: Path, category: str) -> tuple[dict[str, Any], ...]:
    index = json.loads((root / "registration_index.json").read_text(encoding="utf-8"))
    record = next((item for item in index["categories"] if item["category"] == category), None)
    if record is None:
        raise VisaExecutionError("requested category is absent from the VisA 12-category registration")
    def load(rel: str) -> dict[str, Any]:
        return json.loads((root / rel).read_text(encoding="utf-8"))
    identity_manifest = load("private/identity_manifest.private.json")
    return (index, record, load(record["role_manifest_path"]), load(record["release_identity_path"]),
            load(record["unit_candidate_path"]), load(record["stream_manifest_path"]),
            load(record["private_locator_path"]), identity_manifest)


def _validate_registration_runtime(root: Path, category: str, repository: Path):
    from .entrypoint import validate_release_identity
    from . import identity
    index, item, role, release, candidate, stream, locators, identity_manifest = _load_bundle(root, category)
    head = _current_head(repository)
    if head != index.get("source_commit") or head != stream.get("source_commit"):
        raise VisaExecutionError("VisA registration is not bound to the current implementation commit")
    if (stream.get("category") != category or stream.get("stream_seed") != 0 or
            stream.get("support_seed") != 0 or stream.get("dataset_id") != "visa_1cls" or
            stream.get("test_image_count") != len(stream.get("stream_order", ())) or
            stream.get("unit_candidate_sha256") != candidate.get("candidate_sha256") or
            stream.get("identity_manifest_sha256") != _sha(identity_manifest) or
            stream.get("role_manifest_sha256") != _sha(role)):
        raise VisaExecutionError("VisA P0 stream bundle is inconsistent")
    from . import identity
    expected_role = identity.prepare_visa_role_manifest(identity_manifest,
        dataset_release_sha256=index["dataset_release_sha256"], category=category, support_seed=0)
    if role != expected_role:
        raise VisaExecutionError("VisA Support/C64 identities differ from deterministic identity-only selection")
    category_rows = [row for row in identity_manifest["identity_records"]
                     if row.get("category") == category]
    selected = {row["image_identity_sha256"]: row for row in role["selected"]}
    test_rows = [row for row in category_rows if row.get("split") == "test"]
    content_counts: dict[str, int] = {}
    for row in test_rows:
        content = str(row["image_content_sha256"])
        content_counts[content] = content_counts.get(content, 0) + 1
    role_rows = []
    unit_by_image: dict[str, str] = {}
    for row in category_rows:
        image_id = str(row["image_identity_sha256"])
        if row["split"] == "test":
            role_name = "TEST"
            if content_counts[row["image_content_sha256"]] > 1:
                unit_sha = hashlib.sha256(("visa-exact-duplicate-cluster-v1\0" +
                    index["dataset_release_sha256"] + "\0" + category + "\0" +
                    row["image_content_sha256"]).encode("utf-8")).hexdigest()
            else:
                unit_sha = identity.visa_image_unit_sha256(
                    index["dataset_release_sha256"], category, image_id)
        elif image_id in selected:
            role_name = str(selected[image_id]["role"])
            unit_sha = str(selected[image_id]["source_group_sha256"])
        else:
            continue
        role_rows.append({"image_identity_sha256": image_id,
                          "image_content_sha256": row["image_content_sha256"],
                          "role": role_name})
        unit_by_image[image_id] = unit_sha
    expected_units = sorted({unit_by_image[row["image_identity_sha256"]] for row in test_rows},
        key=lambda unit: (identity._domain_hash("Gate0-stream-v1", index["dataset_release_sha256"],
                                                 category, "0", unit), unit.encode("ascii")))
    expected_candidate = identity.prepare_visa_exception_candidate(
        identity_rows=role_rows, proposed_unit_by_image=unit_by_image,
        proposed_test_unit_order=expected_units)
    if candidate != expected_candidate:
        raise VisaExecutionError("VisA atomic-unit candidate differs from deterministic identity/content rules")
    expected_stream: list[dict[str, Any]] = []
    rows_by_unit = {unit: [] for unit in expected_units}
    for row in test_rows:
        rows_by_unit[unit_by_image[row["image_identity_sha256"]]].append(row)
    for unit in expected_units:
        local = sorted(rows_by_unit[unit], key=lambda value: value["image_identity_sha256"])
        kind = "VISA_SINGLE_IMAGE" if len(local) == 1 else "VISA_EXACT_DUPLICATE_CLUSTER"
        for row in local:
            image_id = str(row["image_identity_sha256"])
            expected_stream.append({
                "dataset_id": "visa_1cls", "category": category, "role": "TEST",
                "position": len(expected_stream), "original_position": int(row["original_position"]),
                "image_identity_sha256": image_id,
                "image_content_sha256": row["image_content_sha256"],
                "source_group_sha256": unit, "atomic_unit_sha256": unit,
                "atomic_unit_kind": kind,
                "event_id_sha256": identity._domain_hash("Gate0-visa-event-v1",
                    index["dataset_release_sha256"], category, "0", unit, image_id,
                    str(row["original_position"])),
            })
    if stream.get("stream_order") != expected_stream:
        raise VisaExecutionError("VisA P0 stream order/event identities differ from the deterministic outcome-blind schedule")
    plan_path = root / "private" / category / "execution_plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        unsigned_plan = dict(plan)
        plan_sha = unsigned_plan.pop("execution_plan_sha256", None)
        if plan_sha != _sha(unsigned_plan) or plan.get("source_commit") != head:
            raise VisaExecutionError("VisA source-bound execution plan checksum or Git identity is invalid")
        planned = {row.get("category"): row for row in plan.get("category_registrations", ())}.get(category)
        if (planned is None or planned.get("role_manifest_sha256") != item.get("role_manifest_sha256") or
                planned.get("stream_manifest_sha256") != item.get("stream_manifest_sha256") or
                planned.get("unit_candidate_sha256") != item.get("unit_candidate_sha256") or
                f"{category}/P0" not in plan.get("execution_scope", {}).get("Stage_A", [])):
            raise VisaExecutionError("VisA source-bound execution plan does not bind this category registration")
    candidate_unsigned = dict(candidate)
    candidate_sha = candidate_unsigned.pop("candidate_sha256", None)
    stream_unsigned = dict(stream)
    stream_sha = stream_unsigned.pop("stream_manifest_sha256", None)
    if candidate_sha != _sha(candidate_unsigned) or stream_sha != _sha(stream_unsigned):
        raise VisaExecutionError("VisA P0 stream or candidate checksum is invalid")
    release_sha = validate_release_identity(identity_manifest=identity_manifest,
        role_manifest=role, release_identity=release)
    if release_sha != index["dataset_release_sha256"]:
        raise VisaExecutionError("VisA local snapshot release identity differs from registration index")
    try:
        fresh = identity.build_visa_1cls_identity_manifest(
            Path(next(iter(locators.values()))["dataset_root"]), workers=8)
    except Exception as exc:
        raise VisaExecutionError(f"VisA identity-only snapshot recheck failed ({type(exc).__name__})") from None
    if (fresh.get("local_snapshot_sha256") != index["dataset_snapshot_sha256"] or
            fresh.get("source_artifacts_sha256") != index["source_artifacts_sha256"] or
            fresh.get("identity_records") != identity_manifest.get("identity_records")):
        raise VisaExecutionError("current VisA image/source bytes differ from the registered local snapshot")
    return index, item, role, release, candidate, stream, locators, identity_manifest


def _expected_attempt_identity(*, category: str, head: str, index: Mapping[str, Any], item: Mapping[str, Any],
                               authorization: Mapping[str, Any], authorization_sha256: str) -> dict[str, Any]:
    return {"source_commit": head,
        "identity_manifest_sha256": index["identity_manifest_sha256"],
        "role_manifest_sha256": item["role_manifest_sha256"],
        "dataset_release_sha256": index["dataset_release_sha256"],
        "dataset_id": "visa_1cls", "category": category, "condition_id": "P0",
        "unit_candidate_sha256": item["unit_candidate_sha256"],
        "unit_approval_sha256": hashlib.sha256(_canonical(authorization)).hexdigest(),
        "stream_manifest_sha256": item["stream_manifest_sha256"],
        "execution_plan_sha256": authorization["execution_plan_sha256"],
        "execution_authorization_sha256": authorization_sha256,
        "physical_independence_claim": "NOT_ESTABLISHED"}


def run_one(*, registration_root: str | Path, dataset_root: str | Path,
            attempt_id: str, category: str = "cashew",
            authorization_path: str | Path | None = None,
            repository: str | Path | None = None) -> dict[str, Any]:
    from dataclasses import asdict
    from .entrypoint import validate_release_identity
    from .native_backend import NativeInputEncoder
    from .runner import GroupAtomicCategoryRunner, PRINCIPAL_ARMS
    from .sink import FsyncGroupSink
    from .visa_label_broker import VisaQueryLabelBroker
    from .execution_support import (enforce_resource_limits, enforce_runtime_limits,
        estimate_attempt_output_bytes, resource_snapshot, runtime_resource_snapshot)

    if category not in CATEGORIES or not re.fullmatch(rf"{re.escape(category)}_P0_attempt_[0-9]{{2}}", attempt_id):
        raise VisaExecutionError("VisA category/attempt ID must be registered and explicit")
    repo, root = Path(repository or Path.cwd()).resolve(strict=True), Path(registration_root).resolve(strict=True)
    index, item, role, release, candidate, stream, locators, identity_manifest = _validate_registration_runtime(
        root, category, repo)
    auth_path = Path(authorization_path) if authorization_path else (
        root / "private" / category / "execution_authorization.json")
    authorization = json.loads(auth_path.read_text(encoding="utf-8"))
    auth_file_sha = sha256_file(auth_path)
    actual_head = _current_head(repo)
    required_auth = {
        "schema": "final-benchmark.visa-unit-execution-authorization.v1",
        "status": "RESEARCHER_APPROVED_VISA_ATOMIC_UNIT_EXECUTION",
        "scope": f"{category}_P0_stage_a_only", "dataset_id": "visa_1cls",
        "category": category, "condition_id": "P0", "source_commit": actual_head,
        "identity_manifest_sha256": index["identity_manifest_sha256"],
        "role_manifest_sha256": item["role_manifest_sha256"],
        "dataset_release_sha256": index["dataset_release_sha256"],
        "stream_manifest_sha256": item["stream_manifest_sha256"],
        "unit_candidate_sha256": candidate["candidate_sha256"],
        "expected_event_count": EXPECTED_TEST_COUNTS[category], "query_rate": 0.10,
        "label_access": "QUERY_ONLY", "pixel_masks_allowed": False,
        "stage_b_authorized": False,
    }
    if any(authorization.get(key) != value for key, value in required_auth.items()):
        raise VisaExecutionError("VisA cashew execution grant does not match current source and data registration")
    if authorization.get("dataset_release_identity_sha256") != _sha(release):
        raise VisaExecutionError("VisA execution grant differs from the local snapshot release identity")
    plan_path = root / "private" / category / "execution_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_unsigned = dict(plan)
    plan_sha = plan_unsigned.pop("execution_plan_sha256", None)
    if (plan_sha != _sha(plan_unsigned) or authorization.get("execution_plan_sha256") != plan_sha or
            authorization.get("execution_plan_file_sha256") != sha256_file(plan_path)):
        raise VisaExecutionError("VisA authorization does not match the committed source-bound execution plan")
    if f"{category}/P0" not in plan.get("execution_scope", {}).get("Stage_A", []):
        raise VisaExecutionError("VisA source-bound execution plan does not include this category")
    if authorization.get("researcher_decision_sha256") != plan.get("researcher_decision_sha256"):
        raise VisaExecutionError("VisA execution grant is not bound to the plan's category-specific researcher decision")
    dataset_root_resolved = Path(dataset_root).resolve(strict=True)
    registered_roots = {Path(value["dataset_root"]).resolve(strict=True) for value in locators.values()}
    if registered_roots != {dataset_root_resolved}:
        raise VisaExecutionError("VisA dataset root differs from the registered local snapshot location")
    validate_release_identity(identity_manifest=identity_manifest, role_manifest=role,
                              release_identity=release)
    stream_records = list(stream["stream_order"])
    if (len(stream_records) != EXPECTED_TEST_COUNTS[category] or
            candidate.get("test_image_count") != EXPECTED_TEST_COUNTS[category]):
        raise VisaExecutionError("registered VisA P0 stream count differs from its category census")
    expected_events = EXPECTED_TEST_COUNTS[category]
    attempt_root = root.parent / category / "P0" / attempt_id
    if attempt_root.exists():
        raise VisaExecutionError("VisA attempt directory already exists; refusing overwrite")
    attempt_root.parent.mkdir(parents=True, exist_ok=True)
    expected_identity = _expected_attempt_identity(category=category, head=actual_head, index=index, item=item,
        authorization=authorization, authorization_sha256=auth_file_sha)
    status = AttemptStatus(attempt_root, identity=expected_identity,
                           expected_events=expected_events, execution_kind="principal_stage_a")
    sink = broker = None
    completed = 0
    try:
        before_resources = resource_snapshot(attempt_root)
        enforce_resource_limits("principal", before_resources,
            required_disk_bytes=estimate_attempt_output_bytes("principal", expected_events))
        runtime = NativeInputEncoder.load_authorized_runtime()
        role_ids = {str(row["image_identity_sha256"]) for row in role["selected"]}
        role_locators = {image_id: locators[image_id] for image_id in role_ids}
        commissioning = runtime.prepare_commissioning(role_manifest=role,
            private_locators=role_locators, condition_id="P0", dataset_root=dataset_root)
        backend = runtime.build_backend(category=category, condition_id="P0",
                                        commissioning=commissioning)
        static_receipt = getattr(backend, "static36_receipt", None)
        if not isinstance(static_receipt, Mapping):
            raise VisaExecutionError("registered Static-36+32 baseline did not initialize")
        first_record = stream_records[0]
        first_event = runtime.encode_registered_visa_event(
            record=first_record,
            locator=locators[first_record["image_identity_sha256"]],
            category=category, dataset_release_sha256=index["dataset_release_sha256"])
        # Bounded pre-run source/numerical repeatability check. The same first
        # event is then reused by the formal stream, so it is encoded once.
        state_before = {arm: backend.state_digest(arm) for arm in PRINCIPAL_ARMS}
        shadow_before = backend.shadow_state_digest()
        static_arms = ("STATIC-4+64", "STATIC-36+32")
        static_before = {arm: backend.static_state_digest(arm) for arm in static_arms}
        parity_rows: dict[str, Any] = {}
        packets: dict[str, Any] = {}
        for arm in PRINCIPAL_ARMS:
            packets[arm] = backend.score(arm, first_event, backend.snapshot(arm))
            packet = packets[arm]
            if not (packet.threshold is not None and all(
                    __import__("math").isfinite(float(value)) for value in (packet.score, packet.threshold))):
                raise VisaExecutionError(f"VisA bounded numerical check returned invalid {arm} values")
            parity_rows[arm] = {"score": float(packet.score), "threshold": float(packet.threshold),
                                "state_sha256": packet.state_sha256}
        shadow = backend.score_shadow(first_event, backend.snapshot("F-B"),
                                      backend.shadow_snapshot(backend.snapshot("F-B")))
        parity_rows["N-shadow"] = {"score": float(shadow.score), "state_sha256": shadow.state_sha256}
        for arm in static_arms:
            packet = backend.static_score(arm, first_event)
            parity_rows[arm] = {"score": float(packet.score), "threshold": float(packet.threshold),
                                "state_sha256": packet.state_sha256}
        for arm in PRINCIPAL_ARMS:
            repeated = backend.score(arm, first_event, backend.snapshot(arm))
            if abs(float(repeated.score) - float(packets[arm].score)) > 1e-7:
                raise VisaExecutionError(f"VisA native {arm} score is not repeatable at fixed model state")
        if ({arm: backend.state_digest(arm) for arm in PRINCIPAL_ARMS} != state_before or
                backend.shadow_state_digest() != shadow_before or
                {arm: backend.static_state_digest(arm) for arm in static_arms} != static_before):
            raise VisaExecutionError("bounded VisA numerical check changed online or static model state")
        after_resources = resource_snapshot(attempt_root)
        parity_receipt = {"schema": "final-benchmark.visa-p0-bounded-gpu-parity.v1",
            "dataset_id": "visa_1cls", "category": category, "condition_id": "P0",
            "source_commit": actual_head, "event_id_sha256": first_event.event_id_sha256,
            "registered_model_receipt": {key: runtime.model_receipt.get(key) for key in (
                "checkpoint_sha256", "loaded_model_state_sha256", "source_commit",
                "hub_source_tree_sha256", "profile_id", "execution_profile_sha256")
                if key in runtime.model_receipt},
            "scores": parity_rows, "repeatability_tolerance": 1e-7,
            "all_arm_states_unchanged": True,
            "source_model_algorithms_modified": False,
            "resource_before": before_resources, "resource_after": after_resources,
            "static36_source_receipt": dict(static_receipt)}
        _write_exclusive(attempt_root / "bounded_gpu_parity.json", parity_receipt)

        source_csv_sha = next(row["sha256"] for row in identity_manifest["source_artifacts"]
                              if row["artifact_id"] == "official-split:1cls.csv")
        sink_identity = dict(expected_identity)
        sink = FsyncGroupSink(attempt_root / "stage_a_journal.jsonl", identity=sink_identity)
        broker = VisaQueryLabelBroker(category=category, condition_id="P0",
            dataset_root=dataset_root, dataset_release_sha256=index["dataset_release_sha256"],
            source_csv_sha256=source_csv_sha, stream_records=stream_records,
            private_locators=locators, unit_candidate=candidate,
            approval=authorization, sink=sink,
            private_feedback_path=attempt_root / "private_feedback.jsonl")
        unit_order = list(candidate["test_unit_order"])
        unit_sizes = [len(candidate["test_unit_members"][unit]) for unit in unit_order]
        runner = GroupAtomicCategoryRunner(dataset_id="visa_1cls", category=category,
            condition_id="P0", support_seed=0, stream_seed=0,
            expected_event_count=expected_events, expected_group_order=unit_order,
            expected_unit_sizes=unit_sizes, rho=0.10, backend=backend, broker=broker,
            sink=sink, visa_unit_candidate=candidate,
            visa_execution_authorization=authorization)
        position = 0
        while position < len(stream_records):
            unit = stream_records[position]["atomic_unit_sha256"]
            end = position + 1
            while end < len(stream_records) and stream_records[end]["atomic_unit_sha256"] == unit:
                end += 1
            events = []
            for record in stream_records[position:end]:
                if record["position"] == 0:
                    event = first_event
                else:
                    event = runtime.encode_registered_visa_event(
                        record=record, locator=locators[record["image_identity_sha256"]],
                        category=category, dataset_release_sha256=index["dataset_release_sha256"])
                events.append(event)
            runner.process_unit(events)
            completed = end
            status.update(phase="VISA_ATOMIC_UNIT_COMMITTED", completed_events=completed)
            if completed % 10 == 0 or completed == expected_events:
                enforce_runtime_limits("principal",
                    runtime_resource_snapshot(attempt_root),
                    required_disk_bytes=estimate_attempt_output_bytes(
                        "principal", expected_events - completed))
            position = end
        seal = runner.seal()
        seal_path = attempt_root / "stage_a_category_seal.json"
        status.sealed_complete(phase="CATEGORY_SEALED", seal_path=seal_path,
                               completed_events=completed)
        report = {"schema": "final-benchmark.visa-stage-a-category-report.v1",
            **asdict(seal), "source_commit": actual_head,
            "identity_manifest_sha256": expected_identity["identity_manifest_sha256"],
            "role_manifest_sha256": expected_identity["role_manifest_sha256"],
            "dataset_release_sha256": expected_identity["dataset_release_sha256"],
            "unit_candidate_sha256": expected_identity["unit_candidate_sha256"],
            "unit_approval_sha256": expected_identity["unit_approval_sha256"],
            "stream_manifest_sha256": expected_identity["stream_manifest_sha256"],
            "query_accounting": runner.accounting(),
            "resource_report": runner.resource_report(),
            "bounded_gpu_parity_receipt_sha256": sha256_file(attempt_root / "bounded_gpu_parity.json"),
            "static36_receipt": dict(static_receipt),
            "test_labels_read_outside_legal_queries": False, "pixel_masks_accessed": False}
        _write_exclusive(attempt_root / "stage_a_category_report.json", report)
        return {"status": "SEALED_COMPLETE", "category": category, "condition": "P0",
                "completed_events": completed, "stage_a_seal_sha256": sha256_file(seal_path),
                "attempt_root": str(attempt_root), "query_accounting": runner.accounting(),
                "resource_report": runner.resource_report()}
    except BaseException as exc:
        status.incomplete(phase="VISA_STAGE_A", completed_events=completed, exception=exc)
        raise
    finally:
        if broker is not None:
            broker.close()
        if sink is not None:
            sink.close()


def evaluate_one(*, registration_root: str | Path, attempt_id: str,
                 evaluation_id: str = "stage_b_attempt_01",
                 dataset_root: str | Path, category: str = "cashew",
                 repository: str | Path | None = None) -> dict[str, Any]:
    from .execution_support import verify_attempt_receipt, ExecutionBoundaryError
    from .visa_stage_b import evaluate_visa_attempt
    repo, root = Path(repository or Path.cwd()).resolve(strict=True), Path(registration_root).resolve(strict=True)
    if category not in CATEGORIES or not re.fullmatch(rf"{re.escape(category)}_P0_attempt_[0-9]{{2}}", attempt_id):
        raise VisaExecutionError("VisA Stage B category or Stage A attempt ID is not registered")
    index, item, role, release, candidate, stream, locators, identity_manifest = _validate_registration_runtime(
        root, category, repo)
    head = _current_head(repo)
    if head != index.get("source_commit"):
        raise VisaExecutionError("VisA Stage B source commit differs from its sealed implementation plan")
    if {Path(value["dataset_root"]).resolve(strict=True) for value in locators.values()} != {
            Path(dataset_root).resolve(strict=True)}:
        raise VisaExecutionError("VisA Stage B dataset root differs from its registered local snapshot")
    auth_path = root / "private" / category / "execution_authorization.json"
    approval = json.loads(auth_path.read_text(encoding="utf-8"))
    plan_path = root / "private" / category / "execution_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    unsigned_plan = dict(plan)
    plan_sha = unsigned_plan.pop("execution_plan_sha256", None)
    if (plan_sha != _sha(unsigned_plan) or approval.get("execution_plan_sha256") != plan_sha or
            approval.get("execution_plan_file_sha256") != sha256_file(plan_path)):
        raise VisaExecutionError("VisA Stage B source-bound plan and execution authorization are inconsistent")
    approval_sha = hashlib.sha256(_canonical(approval)).hexdigest()
    attempt_root = root.parent / category / "P0" / attempt_id
    expected_identity = _expected_attempt_identity(category=category, head=head, index=index, item=item,
        authorization=approval, authorization_sha256=sha256_file(auth_path))
    try:
        receipt = verify_attempt_receipt(attempt_root / "attempt_status.json",
            expected_identity=expected_identity, expected_kind="principal_stage_a")
    except ExecutionBoundaryError as exc:
        raise VisaExecutionError("VisA Stage B cannot begin before valid Stage A completion") from exc
    from .visa_stage_b import _load_stage_a
    _load_stage_a(attempt_root, expected_identity, list(stream["stream_order"]), candidate)
    source_sha = next(row["sha256"] for row in identity_manifest["source_artifacts"]
                      if row["artifact_id"] == "official-split:1cls.csv")
    stage_b_authorization = {
        "schema": "final-benchmark.visa-stage-b-label-projection-authorization.v1",
        "status": "RESEARCHER_AUTHORIZED_POST_SEAL_VISA_STAGE_B",
        "scope": f"{category}_P0_post_seal_only", "source_commit": head,
        "dataset_id": "visa_1cls", "category": category, "condition_id": "P0",
        "stage_a_seal_sha256": sha256_file(receipt["seal_path"]),
        "unit_candidate_sha256": candidate["candidate_sha256"],
        "unit_approval_sha256": approval_sha,
        "source_csv_sha256": source_sha,
        "researcher_decision_sha256": approval.get("researcher_decision_sha256"),
        "pixel_masks_allowed": False,
        "authorization_provenance": "explicit researcher task; post-seal category analysis only",
    }
    if not re.fullmatch(r"stage_b_attempt_[0-9]{2}", evaluation_id):
        raise VisaExecutionError("Stage B evaluation ID must be explicit and non-overwriting")
    stage_b_auth_path = root / "private" / category / f"{evaluation_id}_authorization.json"
    auth_file_sha = _write_exclusive(stage_b_auth_path, stage_b_authorization)
    stream_bound = dict(stream)
    output = root.parent / category / "P0" / f"{attempt_id}_{evaluation_id}"
    result = evaluate_visa_attempt(attempt_root=attempt_root, dataset_root=dataset_root,
        identity_manifest=identity_manifest, stream_manifest=stream_bound,
        candidate=candidate, approval=approval, authorization=stage_b_authorization,
        expected_attempt_identity=expected_identity, output_directory=output)
    return {"status": "STAGE_B_COMPLETE", "output_directory": str(output),
            "stage_b_authorization_sha256": hashlib.sha256(_canonical(stage_b_authorization)).hexdigest(),
            "result": result, "stage_b_authorization_file_sha256": auth_file_sha}


def aggregate_stage_b(*, registration_root: str | Path,
                      result_paths: Sequence[str | Path], output_directory: str | Path,
                      repository: str | Path | None = None) -> dict[str, Any]:
    root, repo = Path(registration_root).resolve(strict=True), Path(repository or Path.cwd()).resolve(strict=True)
    index = json.loads((root / "registration_index.json").read_text(encoding="utf-8"))
    expected = set(CATEGORIES)
    loaded = [json.loads(Path(path).read_text(encoding="utf-8")) for path in result_paths]
    if (len(loaded) != 12 or {row.get("category") for row in loaded} != expected or
            any(row.get("dataset_id") != "visa_1cls" or row.get("condition_id") != "P0" or
                row.get("source_commit") != index.get("source_commit") for row in loaded) or
            _current_head(repo) != index.get("source_commit")):
        raise VisaExecutionError("VisA macro aggregation requires all 12 sealed P0 Stage B outputs from one source")
    methods = ["F-B", "F-M", "K-B", "K-M", "N-shadow", "STATIC-4+64", "STATIC-36+32"]
    summary: dict[str, Any] = {"schema": "final-benchmark.visa-p0-stage-b-macro.v1",
        "dataset_id": "visa_1cls", "condition_id": "P0", "source_commit": index["source_commit"],
        "category_count": 12, "methods": {}, "physical_independence_claim": "NOT_ESTABLISHED"}
    for method in methods:
        keys = ("auroc", "ap") if method == "N-shadow" else ("auroc", "ap", "ber", "fpr", "fnr", "balanced_accuracy")
        rows = []
        for category_result in loaded:
            metrics = category_result["arm_metrics"][method]
            rows.append({"category": category_result["category"], **{key: metrics[key] for key in keys}})
        aggregate = {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}
        summary["methods"][method] = {"category_equal_macro": aggregate, "per_category": rows}
    target = Path(output_directory)
    if target.exists():
        raise VisaExecutionError("VisA 12-category aggregation output already exists")
    target.mkdir(parents=True, exist_ok=False)
    _write_exclusive(target / "visa_p0_macro_metrics.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="final_benchmark.launch visa-*")
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register")
    for key in ("dataset-root", "identity-manifest", "output-root", "researcher-decision-sha256"):
        reg.add_argument("--" + key, required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--registration-root", required=True)
    plan.add_argument("--category", default="cashew")
    plan.add_argument("--researcher-decision-sha256")
    run = sub.add_parser("run-one")
    for key in ("registration-root", "dataset-root", "attempt-id"):
        run.add_argument("--" + key, required=True)
    run.add_argument("--category", default="cashew")
    run.add_argument("--authorization")
    evaluate = sub.add_parser("evaluate-one")
    for key in ("registration-root", "dataset-root", "attempt-id"):
        evaluate.add_argument("--" + key, required=True)
    evaluate.add_argument("--evaluation-id", default="stage_b_attempt_01")
    evaluate.add_argument("--category", default="cashew")
    aggregate = sub.add_parser("aggregate-stage-b")
    aggregate.add_argument("--registration-root", required=True)
    aggregate.add_argument("--result", action="append", required=True)
    aggregate.add_argument("--output-directory", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "register":
            result = register(dataset_root=args.dataset_root,
                identity_manifest=args.identity_manifest, output_root=args.output_root,
                researcher_decision_sha256=args.researcher_decision_sha256)
        elif args.command == "plan":
            result = create_source_bound_plan(registration_root=args.registration_root,
                category=args.category,
                researcher_decision_sha256=args.researcher_decision_sha256)
        elif args.command == "run-one":
            result = run_one(registration_root=args.registration_root,
                dataset_root=args.dataset_root, attempt_id=args.attempt_id,
                category=args.category, authorization_path=args.authorization)
        elif args.command == "evaluate-one":
            result = evaluate_one(registration_root=args.registration_root,
                dataset_root=args.dataset_root, attempt_id=args.attempt_id,
                evaluation_id=args.evaluation_id, category=args.category)
        else:
            result = aggregate_stage_b(registration_root=args.registration_root,
                result_paths=args.result, output_directory=args.output_directory)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
