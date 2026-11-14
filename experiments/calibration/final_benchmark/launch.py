"""Thin command-line launch boundary for the frozen final benchmark.

This module loads only registered metadata and private path/authorization
records during validation. Scoring is reachable only through explicit
``run-one``, ``run-schedule`` or ``foads-run-one`` commands.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

from .execution_support import (
    AttemptStatus, ExecutionBoundaryError, MEASURED_RESOURCE_BASELINES,
    OUTPUT_EXTRAPOLATION_MULTIPLIER, PER_ATTEMPT_FIXED_RESERVE_BYTES,
    SYSTEM_DISK_RESERVE_BYTES, canonical_json,
    enforce_resource_limits, enforce_runtime_limits, estimate_attempt_output_bytes, resource_snapshot,
    runtime_resource_snapshot,
    sha256_file, verify_attempt_receipt,
)


class LaunchError(RuntimeError):
    pass


CONFIG_SCHEMA = "final-benchmark.execution-plan-template.v1"
PRIVATE_BUNDLE_SCHEMA = "final-benchmark.private-category-bundle.v1"
CONDITIONS = {"P0": (0, 0), "P1": (0, 11), "P2": (1, 0)}
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


def read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise LaunchError(f"JSON input is unreadable: {Path(path).name}") from exc


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def current_git_head(repository: str | Path | None = None) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                            check=True, capture_output=True, text=True)
    head = result.stdout.strip()
    if not _GIT_SHA.fullmatch(head):
        raise LaunchError("current Git HEAD is not a lowercase 40-character commit ID")
    return head


def validate_config(config: Mapping[str, Any], *, current_head: str | None = None,
                    require_source_binding: bool = False) -> dict[str, Any]:
    if config.get("schema") != CONFIG_SCHEMA:
        raise LaunchError("configuration schema is not the registered final benchmark plan")
    unsigned = dict(config)
    schedule_sha = unsigned.pop("schedule_sha256", None)
    if not isinstance(schedule_sha, str) or not _HEX64.fullmatch(schedule_sha):
        raise LaunchError("configuration schedule checksum is missing or malformed")
    if object_sha256(unsigned) != schedule_sha:
        raise LaunchError("configuration checksum does not match its frozen schedule and parameters")
    dataset = config.get("dataset", {})
    principal = config.get("principal", {})
    foads = config.get("foads_kng", {})
    if (dataset.get("dataset_id") != "realiad" or dataset.get("category_count") != 30 or
            dataset.get("identity_record_count") != 151050 or
            dataset.get("dataset_release_sha256") != "52b045ea219eb270c427d30785da8e9b891c8eff2ba035b4430560bf17c962ae" or
            dataset.get("identity_manifest_sha256") != "4d6841d048cb073d3b121ee025a3afd3619572b62afcf167081acbb6aa0f9912"):
        raise LaunchError("dataset release or identity census differs from the researcher-approved Gate 1 snapshot")
    if (principal.get("arms") != ["F-B", "K-B", "F-M", "K-M"] or
            principal.get("mechanism_control") != "N-shadow" or
            principal.get("query_target") != 0.03 or
            principal.get("query_cap") != "floor(0.03*T_category)" or
            principal.get("per_arm_queries_per_source_group_max") != 1 or
            principal.get("no_forced_queries") is not True or
            principal.get("no_query_topups") is not True or
            principal.get("realized_rate_matching_tolerance") != 0.003 or
            principal.get("conditions") != {"P0": {"stream_seed": 0, "support_seed": 0},
                                             "P1": {"stream_seed": 11, "support_seed": 0},
                                             "P2": {"stream_seed": 0, "support_seed": 1}}):
        raise LaunchError("principal methods, query accounting or P0/P1/P2 seeds differ from the frozen contract")
    if (foads.get("baseline") != "FOADS-KNG-PAPER-CONSISTENT-PROTOCOL-ADAPTED" or
            foads.get("upstream_commit") != "63dd8e7c9ade5619771d4b5544f032206818bb22" or
            foads.get("weights_enum") != "ResNet18_Weights.IMAGENET1K_V1" or
            foads.get("weights_sha256") != "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec" or
            foads.get("initial_normal_count") != 68 or foads.get("initial_node_target") != 3136 or
            foads.get("fit_epochs") != 10 or foads.get("age_max") != 25 or
            foads.get("feature_dimensions") != 100 or foads.get("batch_size_images") != 10 or
            foads.get("groups_per_update") != 2 or foads.get("paid_labels") != 0):
        raise LaunchError("FOADS/K-NG source, initialization or batch protocol differs from the frozen adaptation")
    resource = config.get("resource_policy", {})
    disk_guard = resource.get("disk_guard", {})
    memory_guard = resource.get("memory_stop_rules", {})
    if (resource.get("automatic_resume") is not False or
            resource.get("full_campaign_runtime_guarantee") is not False or
            disk_guard.get("extrapolation_multiplier") != OUTPUT_EXTRAPOLATION_MULTIPLIER or
            disk_guard.get("principal_observed_bytes_per_event") != MEASURED_RESOURCE_BASELINES["principal"]["observed_bytes_per_event"] or
            disk_guard.get("foads_observed_bytes_per_event") != MEASURED_RESOURCE_BASELINES["foads"]["observed_bytes_per_event"] or
            disk_guard.get("fixed_reserve_per_attempt_bytes") != PER_ATTEMPT_FIXED_RESERVE_BYTES or
            disk_guard.get("system_free_space_reserve_bytes") != SYSTEM_DISK_RESERVE_BYTES or
            disk_guard.get("full_category_growth_measured") is not False or
            memory_guard.get("host_available_margin_over_measured_peak") != 1.25 or
            memory_guard.get("principal_measured_peak_host_rss_bytes") != MEASURED_RESOURCE_BASELINES["principal"]["peak_host_rss_bytes"] or
            memory_guard.get("principal_measured_peak_gpu_reserved_bytes") != MEASURED_RESOURCE_BASELINES["principal"]["peak_gpu_reserved_bytes"] or
            memory_guard.get("foads_measured_peak_host_rss_bytes") != MEASURED_RESOURCE_BASELINES["foads"]["peak_host_rss_bytes"] or
            memory_guard.get("foads_measured_peak_gpu_reserved_bytes") != MEASURED_RESOURCE_BASELINES["foads"]["peak_gpu_reserved_bytes"] or
            memory_guard.get("gpu_free_margin_bytes") != 1024 ** 3):
        raise LaunchError("resource stop limits differ from bounded measured evidence or enable automatic resume")
    rows = config.get("trajectories")
    if not isinstance(rows, list) or len(rows) != 90:
        raise LaunchError("principal schedule must contain exactly 90 category-condition trajectories")
    categories = {str(row.get("category")) for row in rows if isinstance(row, Mapping)}
    pairs = [(row.get("category"), row.get("condition")) for row in rows
             if isinstance(row, Mapping)]
    expected = {(category, condition) for category in categories for condition in CONDITIONS}
    if len(categories) != 30 or len(pairs) != 90 or len(set(pairs)) != 90 or set(pairs) != expected:
        raise LaunchError("schedule must register each of 30 categories exactly once for P0/P1/P2")
    category_list = config.get("foads_kng", {}).get("categories")
    if (not isinstance(category_list, list) or len(category_list) != 30 or
            len(set(category_list)) != 30 or set(category_list) != categories):
        raise LaunchError("FOADS P0 schedule must contain exactly the 30 registered categories")
    static = config.get("static_references", {}).get("Static-36+32", {})
    if (set(static.get("categories", ())) != categories or static.get("conditions") != ["P0"]):
        raise LaunchError("Static-36+32 must be registered for all categories at P0 only")
    for row in rows:
        category, condition = row["category"], row["condition"]
        if condition not in CONDITIONS:
            raise LaunchError("schedule contains an unregistered condition")
        if (not _HEX64.fullmatch(str(row.get("role_manifest_sha256", ""))) or
                not _HEX64.fullmatch(str(row.get("stream_manifest_sha256", ""))) or
                int(row.get("test_event_count", -1)) <= 0 or
                int(row.get("source_group_count", -1)) * 5 != int(row.get("test_event_count", -1))):
            raise LaunchError(f"{category}/{condition} has incomplete source/stream commitments")
        p0_only_values = (row.get("static36_schedule_sha256"),
                          row.get("foads_p0_registration_projection_sha256"))
        if (condition == "P0") != all(isinstance(value, str) and _HEX64.fullmatch(value)
                                       for value in p0_only_values):
            raise LaunchError(f"{category}/{condition} has incorrect P0-only comparator binding")
        if condition != "P0" and any(value is not None for value in p0_only_values):
            raise LaunchError(f"{category}/{condition} incorrectly enables a P0-only comparator")
    source_identity = config.get("source_identity", {})
    source_commit = source_identity.get("final_source_commit")
    run_id = source_identity.get("campaign_run_id")
    bound = isinstance(source_commit, str) and bool(_GIT_SHA.fullmatch(source_commit)) and \
        isinstance(run_id, str) and bool(_SAFE_ID.fullmatch(run_id))
    if require_source_binding and not bound:
        raise LaunchError("final Git source commit and immutable campaign run ID are not bound")
    if source_commit is not None and (not isinstance(source_commit, str) or not _GIT_SHA.fullmatch(source_commit)):
        raise LaunchError("final source identity must be an actual lowercase 40-character Git commit ID")
    if bound and current_head is not None and source_commit != current_head:
        raise LaunchError("configuration source commit differs from the current Git HEAD")
    if require_source_binding and source_identity.get("execution_authorizations_created") is not False:
        raise LaunchError("execution plan must not contain generated authorization records")
    return {"schedule_sha256": schedule_sha, "trajectory_count": 90,
            "category_count": 30, "foads_category_count": 30,
            "source_bound": bound, "source_commit": source_commit,
            "campaign_run_id": run_id}


def read_bound_config(path: str | Path, *, require_source_binding: bool = True,
                      repository: str | Path | None = None) -> tuple[dict[str, Any], str, dict[str, Any]]:
    config = read_json(path)
    if not isinstance(config, Mapping):
        raise LaunchError("execution plan root must be a JSON object")
    report = validate_config(config, current_head=current_git_head(repository),
                             require_source_binding=require_source_binding)
    return dict(config), sha256_file(path), report


def registered_trajectory(config: Mapping[str, Any], category: str, condition: str) -> dict[str, Any]:
    matches = [dict(row) for row in config["trajectories"]
               if row.get("category") == category and row.get("condition") == condition]
    if len(matches) != 1:
        raise LaunchError("selected category-condition is not uniquely present in the frozen schedule")
    return matches[0]


def _unwrap_locator_map(raw: Mapping[str, Any], category: str, key: str) -> Mapping[str, Mapping[str, str]]:
    try:
        locators = raw["categories"][category][key]
    except Exception as exc:
        raise LaunchError(f"private locator manifest lacks the registered {category}/{key} map") from exc
    if not isinstance(locators, Mapping):
        raise LaunchError("private locator map must be keyed by registered image identity")
    return locators


def load_principal_bundle(bundle_path: str | Path, *, config: Mapping[str, Any],
                          category: str, condition: str) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = read_json(bundle_path)
    if not isinstance(bundle, Mapping) or bundle.get("schema") != PRIVATE_BUNDLE_SCHEMA:
        raise LaunchError("private bundle must be a registered final-benchmark category bundle")
    required_paths = ("category_bundle_path", "identity_manifest_path", "private_locator_manifest_path")
    if any(not isinstance(bundle.get(key), str) for key in required_paths):
        raise LaunchError("private category bundle does not identify its registered Gate 1 files")
    category_bundle = read_json(bundle["category_bundle_path"])
    identity = read_json(bundle["identity_manifest_path"])
    locators = read_json(bundle["private_locator_manifest_path"])
    if not all(isinstance(value, Mapping) for value in (category_bundle, identity, locators)):
        raise LaunchError("Gate 1 data inputs must be JSON objects")
    trajectory = registered_trajectory(config, category, condition)
    dataset = config["dataset"]
    identity_sha = object_sha256(identity)
    if (category_bundle.get("category") != category or category_bundle.get("condition_id") != condition or
            category_bundle.get("schema") != "final-benchmark.category-condition-identity.v1" or
            identity_sha != dataset.get("identity_manifest_sha256") or
            sha256_file(bundle["identity_manifest_path"]) != dataset.get("identity_manifest_file_sha256") or
            category_bundle.get("identity_manifest_sha256") != identity_sha or
            category_bundle.get("identity_manifest_file_sha256") != dataset.get("identity_manifest_file_sha256") or
            category_bundle.get("role_manifest_sha256") != trajectory["role_manifest_sha256"] or
            category_bundle.get("stream_manifest_sha256") != trajectory["stream_manifest_sha256"] or
            not isinstance(category_bundle.get("release_identity"), Mapping) or
            category_bundle["release_identity"].get("dataset_release_sha256") != dataset.get("dataset_release_sha256")):
        raise LaunchError("category registration differs from the frozen source, role, stream or dataset identity")
    if (locators.get("schema") != "final-benchmark.private-locators.v1" or
            locators.get("identity_manifest_sha256") != identity_sha):
        raise LaunchError("private locators are not bound to the certified identity manifest")
    role = category_bundle.get("role_manifest")
    stream = category_bundle.get("stream_manifest")
    release = category_bundle.get("release_identity")
    if (not isinstance(role, Mapping) or object_sha256(role) != trajectory["role_manifest_sha256"] or
            not isinstance(stream, Mapping) or object_sha256(stream) != trajectory["stream_manifest_sha256"] or
            not isinstance(release, Mapping) or release.get("dataset_release_sha256") != dataset["dataset_release_sha256"]):
        raise LaunchError("category role, stream or release identity does not match the frozen schedule")
    test_locators = _unwrap_locator_map(locators, category, "test")
    available_role_locators = _unwrap_locator_map(locators, category, "role_train")
    event_ids = {str(row.get("image_identity_sha256")) for row in stream.get("events", [])}
    role_ids = {str(row.get("image_identity_sha256")) for row in role.get("selected", [])}
    if set(test_locators) != event_ids or not role_ids.issubset(available_role_locators):
        raise LaunchError("private locator identities differ from the registered stream or 4+64 role allocation")
    role_locators = {image_id: available_role_locators[image_id] for image_id in role_ids}
    if (category_bundle.get("stream_seed") != CONDITIONS[condition][1] or
            category_bundle.get("support_seed") != CONDITIONS[condition][0] or
            len(stream.get("events", [])) != trajectory["test_event_count"]):
        raise LaunchError("category registration has the wrong seed or event count")
    private_category = locators.get("categories", {}).get(category, {})
    annotation_path = private_category.get("annotation_path")
    annotation_sha = private_category.get("annotation_sha256")
    # Bind only the metadata checksum here; annotation bytes/outcomes are left to
    # the query broker after a legally executed query.
    artifacts = [row.get("sha256") for row in identity.get("source_artifacts", [])
                 if row.get("artifact_id") == f"annotation:{category}"]
    if not isinstance(annotation_path, str) or len(artifacts) != 1 or annotation_sha != artifacts[0]:
        raise LaunchError("private annotation locator is not bound to the category's registered source hash")
    payload = {
        "bundle": dict(bundle), "category_bundle": dict(category_bundle),
        "identity_manifest": dict(identity), "role_manifest": dict(role),
        "stream_manifest": dict(stream), "release_identity": dict(release),
        "test_private_locators": {str(k): dict(v) for k, v in test_locators.items()},
        "role_private_locators": {str(k): dict(v) for k, v in role_locators.items()},
        "annotation_path": annotation_path, "annotation_sha256": annotation_sha,
    }
    identities = {
        "source_commit": config["source_identity"]["final_source_commit"],
        "campaign_run_id": config["source_identity"]["campaign_run_id"],
        "execution_plan_sha256": None,
        "identity_manifest_sha256": identity_sha,
        "role_manifest_sha256": trajectory["role_manifest_sha256"],
        "stream_manifest_sha256": trajectory["stream_manifest_sha256"],
        "dataset_release_sha256": dataset["dataset_release_sha256"],
        "category_bundle_file_sha256": sha256_file(bundle["category_bundle_path"]),
        "private_locator_manifest_file_sha256": sha256_file(bundle["private_locator_manifest_path"]),
        "annotation_sha256": annotation_sha,
        "category": category, "condition": condition,
    }
    return payload, identities


def validate_principal_authorization(authorization: Mapping[str, Any], *,
                                     identities: Mapping[str, Any]) -> None:
    expected_count = authorization.get("expected_event_count")
    if (authorization.get("schema") != "final-benchmark.execution-authorization.v1" or
            authorization.get("status") != "AUTHORIZED" or
            authorization.get("scope") != "stage_a_category" or
            authorization.get("source_commit") != identities["source_commit"] or
            authorization.get("campaign_run_id") != identities["campaign_run_id"] or
            authorization.get("execution_plan_sha256") != identities["execution_plan_sha256"] or
            authorization.get("identity_manifest_sha256") != identities["identity_manifest_sha256"] or
            authorization.get("role_manifest_sha256") != identities["role_manifest_sha256"] or
            authorization.get("stream_manifest_sha256") != identities["stream_manifest_sha256"] or
            authorization.get("dataset_release_sha256") != identities["dataset_release_sha256"] or
            authorization.get("category_bundle_file_sha256") != identities["category_bundle_file_sha256"] or
            authorization.get("private_locator_manifest_file_sha256") != identities["private_locator_manifest_file_sha256"] or
            authorization.get("annotation_sha256") != identities["annotation_sha256"] or
            authorization.get("dataset_id") != "realiad" or
            authorization.get("category") != identities["category"] or
            authorization.get("condition_id") != identities["condition"] or
            authorization.get("label_access") != "QUERY_ONLY" or
            authorization.get("stage_b_authorized") is not False or
            isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0):
        raise LaunchError("Stage A authorization is missing or differs from the frozen source/data/configuration")


def _campaign_output_root(config: Mapping[str, Any], repository: str | Path) -> Path:
    run_id = config["source_identity"]["campaign_run_id"]
    relative = str(config.get("output_root", ""))
    if "<campaign_run_id>" not in relative:
        raise LaunchError("output_root must retain the registered campaign_run_id template")
    resolved = (Path(repository) / relative.replace("<campaign_run_id>", run_id)).resolve()
    return resolved


def _ensure_inside(child: Path, parent: Path) -> None:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError as exc:
        raise LaunchError("attempt output escapes the frozen campaign output root") from exc


def execute_principal_one(*, config_path: str | Path, bundle_path: str | Path,
                          authorization_path: str | Path, category: str,
                          condition: str, attempt_id: str,
                          repository: str | Path | None = None,
                          encoder: Any | None = None,
                          resource_probe: Any = resource_snapshot) -> dict[str, Any]:
    if not _SAFE_ID.fullmatch(attempt_id):
        raise LaunchError("attempt ID must be a safe immutable path component")
    repository_path = Path(repository or Path.cwd()).resolve()
    config, config_file_sha, config_report = read_bound_config(
        config_path, require_source_binding=True, repository=repository_path)
    payload, identities = load_principal_bundle(bundle_path, config=config,
                                                category=category, condition=condition)
    identities["execution_plan_sha256"] = config_file_sha
    authorization = read_json(authorization_path)
    if not isinstance(authorization, Mapping):
        raise LaunchError("Stage A authorization must be a JSON object")
    validate_principal_authorization(authorization, identities=identities)
    trajectory = registered_trajectory(config, category, condition)
    if int(authorization["expected_event_count"]) != int(trajectory["test_event_count"]):
        raise LaunchError("authorization event count differs from the registered stream")
    root = _campaign_output_root(config, repository_path)
    attempt_root = root / "stage_a" / category / condition / attempt_id
    _ensure_inside(attempt_root, root)
    if attempt_root.exists():
        raise LaunchError(f"attempt output already exists: {attempt_root}")
    estimate = estimate_attempt_output_bytes("principal", trajectory["test_event_count"])
    initial_resources = resource_probe(attempt_root)
    enforce_resource_limits("principal", initial_resources, required_disk_bytes=estimate)
    from .entrypoint import FinalBenchmarkEntrypoint
    entrypoint = FinalBenchmarkEntrypoint(
        source_commit=identities["source_commit"],
        identity_manifest=payload["identity_manifest"], role_manifest=payload["role_manifest"],
        release_identity=payload["release_identity"],
        stream_records=payload["stream_manifest"]["events"],
        test_private_locators=payload["test_private_locators"],
        role_private_locators=payload["role_private_locators"],
        category=category, condition_id=condition,
        campaign_run_id=identities["campaign_run_id"],
        execution_plan_sha256=identities["execution_plan_sha256"],
        stream_manifest_sha256=identities["stream_manifest_sha256"],
        category_bundle_file_sha256=identities["category_bundle_file_sha256"],
        private_locator_manifest_file_sha256=identities["private_locator_manifest_file_sha256"],
        annotation_sha256=identities["annotation_sha256"])
    attempt = AttemptStatus(attempt_root, identity={**identities,
        "execution_plan_file_sha256": config_file_sha,
        "authorization_file_sha256": sha256_file(authorization_path),
        "category_bundle_sha256": sha256_file(payload["bundle"]["category_bundle_path"])},
        expected_events=trajectory["test_event_count"], execution_kind="principal_stage_a")
    phase = "AUTHORIZED_START"
    completed = 0

    def progress(next_phase: str, count: int) -> None:
        nonlocal phase, completed
        phase, completed = next_phase, int(count)
        attempt.update(phase=phase, completed_events=completed)
        if next_phase in {"MODEL_INITIALIZED", "GROUP_COMMITTED", "CATEGORY_SEALED"}:
            remaining = max(0, trajectory["test_event_count"] - completed)
            required = estimate_attempt_output_bytes("principal", remaining)
            current = runtime_resource_snapshot(attempt_root)
            enforce_runtime_limits("principal", current, required_disk_bytes=required)

    try:
        seed = CONDITIONS[condition][1]
        seal, rows, _ = entrypoint.run_stage_a_category(
            authorization=authorization, encoder=encoder, output_directory=attempt_root,
            annotation_path=payload["annotation_path"],
            annotation_sha256=payload["annotation_sha256"], stream_seed=seed,
            progress_callback=progress)
        if (seal.status != "SEALED_COMPLETE" or len(rows) != trajectory["test_event_count"] or
                not (attempt_root / "stage_a_category_seal.json").is_file()):
            raise LaunchError("existing Stage A runner did not produce a full-category completion seal")
        attempt.sealed_complete(phase="CATEGORY_SEALED",
                                seal_path=attempt_root / "stage_a_category_seal.json",
                                completed_events=len(rows))
        return {"status": "SEALED_COMPLETE", "attempt_root": str(attempt_root),
                "completed_event_count": len(rows), "completion_seal_sha256": attempt.state["completion_seal_sha256"],
                "resources_at_start": initial_resources, "configuration": config_report}
    except BaseException as exc:
        attempt.incomplete(phase=phase, completed_events=completed, exception=exc)
        raise
    finally:
        del entrypoint
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _current_torch_gpu_reservation() -> int | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_reserved())
    except Exception:
        return None


def validate_sequential_authorization(record: Mapping[str, Any], *, config: Mapping[str, Any],
                                      config_sha256: str, current_head: str,
                                      schedule_sha256: str) -> None:
    if (record.get("schema") != "final-benchmark.sequential-launch-authorization.v1" or
            record.get("status") != "AUTHORIZED" or
            record.get("scope") != "stage_a_sequential_campaign" or
            record.get("source_commit") != current_head or
            record.get("campaign_run_id") != config["source_identity"]["campaign_run_id"] or
            record.get("execution_plan_sha256") != config_sha256 or
            record.get("launch_schedule_sha256") != schedule_sha256 or
            record.get("failure_policy") != "STOP_ON_FIRST_FAILURE"):
        raise LaunchError("sequential campaign authorization does not match the source-bound schedule")


def execute_sequential_schedule(*, config_path: str | Path, launch_schedule_path: str | Path,
                                authorization_path: str | Path, repository: str | Path | None = None,
                                executor: Any = execute_principal_one) -> list[dict[str, Any]]:
    repository_path = Path(repository or Path.cwd()).resolve()
    config, config_sha, _ = read_bound_config(config_path, require_source_binding=True,
                                             repository=repository_path)
    schedule = read_json(launch_schedule_path)
    authorization = read_json(authorization_path)
    if not isinstance(schedule, Mapping) or schedule.get("schema") != "final-benchmark.sequential-launch-schedule.v1":
        raise LaunchError("explicit schedule must use the registered sequential launch schedule schema")
    unsigned = dict(schedule)
    schedule_sha = unsigned.pop("schedule_sha256", None)
    if not isinstance(schedule_sha, str) or object_sha256(unsigned) != schedule_sha:
        raise LaunchError("sequential launch schedule checksum is invalid")
    validate_sequential_authorization(authorization, config=config, config_sha256=config_sha,
                                      current_head=current_git_head(repository_path),
                                      schedule_sha256=sha256_file(launch_schedule_path))
    launches = schedule.get("launches")
    expected_pairs = {(row["category"], row["condition"]) for row in config["trajectories"]}
    actual_pairs = [(row.get("category"), row.get("condition")) for row in launches or []]
    if (not isinstance(launches, list) or len(launches) != 90 or len(set(actual_pairs)) != 90 or
            set(actual_pairs) != expected_pairs):
        raise LaunchError("authorized sequential schedule must include every registered trajectory exactly once")
    if any(any(not isinstance(row.get(field), str) for field in ("bundle_path", "authorization_path", "attempt_id"))
           for row in launches):
        raise LaunchError("each sequential launch needs explicit bundle, category authorization and attempt ID paths")
    results = []
    for row in launches:
        try:
            result = executor(config_path=config_path, bundle_path=row["bundle_path"],
                              authorization_path=row["authorization_path"],
                              category=row["category"], condition=row["condition"],
                              attempt_id=row["attempt_id"], repository=repository_path)
        except Exception as exc:
            raise LaunchError(f"sequential schedule stopped on first failure at {row['category']}/{row['condition']}: {exc}") from exc
        results.append(result)
        gc.collect()
    return results


def describe_resources(config: Mapping[str, Any], output_path: str | Path) -> dict[str, Any]:
    principal_events = sum(int(row["test_event_count"]) for row in config["trajectories"])
    foads_events = sum(int(row["test_event_count"]) for row in config["trajectories"]
                       if row["condition"] == "P0")
    principal_bytes = estimate_attempt_output_bytes("principal", principal_events) + 30 * PER_ATTEMPT_FIXED_RESERVE_BYTES
    foads_bytes = estimate_attempt_output_bytes("foads", foads_events) + 30 * PER_ATTEMPT_FIXED_RESERVE_BYTES
    output_path = Path(output_path).resolve()
    disk_root = output_path
    while not disk_root.exists() and disk_root != disk_root.parent:
        disk_root = disk_root.parent
    try:
        disk_free = int(shutil.disk_usage(disk_root).free)
    except OSError:
        disk_free = None
    max_principal_events = max(int(row["test_event_count"]) for row in config["trajectories"])
    max_foads_events = max(int(row["test_event_count"]) for row in config["trajectories"]
                           if row["condition"] == "P0")
    return {
        "evidence_scope": "bounded_preflight_measured_bytes_per_event_extrapolated_10x; not a full-run guarantee",
        "principal_test_events": principal_events,
        "foads_p0_test_events": foads_events,
        "principal_category_output_stop_bytes": MEASURED_RESOURCE_BASELINES["principal"]["observed_bytes_per_event"] * OUTPUT_EXTRAPOLATION_MULTIPLIER,
        "foads_category_output_stop_bytes": MEASURED_RESOURCE_BASELINES["foads"]["observed_bytes_per_event"] * OUTPUT_EXTRAPOLATION_MULTIPLIER,
        "maximum_principal_category_reserved_output_bytes": estimate_attempt_output_bytes("principal", max_principal_events),
        "maximum_foads_category_reserved_output_bytes": estimate_attempt_output_bytes("foads", max_foads_events),
        "fixed_reserve_per_attempt_bytes": PER_ATTEMPT_FIXED_RESERVE_BYTES,
        "output_root": str(output_path),
        "output_root_collision": output_path.exists(),
        "full_campaign_guard_estimate_bytes": principal_bytes + foads_bytes,
        "system_disk_reserve_bytes": SYSTEM_DISK_RESERVE_BYTES,
        "available_disk_bytes": disk_free,
        "disk_guard_currently_satisfied": (disk_free is not None and disk_free >= principal_bytes + foads_bytes + SYSTEM_DISK_RESERVE_BYTES),
        "measured_resource_baselines": MEASURED_RESOURCE_BASELINES,
        "host_memory_margin": 1.25,
        "gpu_free_margin_bytes": 1024 ** 3,
        "limitations": ["full-category journal growth has not been measured", "no full-campaign runtime guarantee",
                        "actual authorization and post-commit source binding remain separate prerequisites"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments.calibration.final_benchmark.launch",
        description="Validate and launch only the registered final benchmark trajectories.")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate schedule/config binding without loading a model")
    validate.add_argument("--config", required=True)
    validate.add_argument("--allow-unbound-template", action="store_true",
                          help="validate schedule only; never enables execution")
    listing = sub.add_parser("list", help="print all 90 registered category-condition trajectories")
    listing.add_argument("--config", required=True)
    resources = sub.add_parser("resources", help="report bounded-measurement storage guard estimates")
    resources.add_argument("--config", required=True)
    resources.add_argument("--output-path", required=True)
    run = sub.add_parser("run-one", help="run one separately authorized Stage A trajectory")
    for arg in ("config", "bundle", "authorization", "category", "condition", "attempt-id"):
        run.add_argument("--" + arg, required=True)
    schedule = sub.add_parser("run-schedule", help="run an explicitly authorized sequential schedule; stops at first failure")
    for arg in ("config", "launch-schedule", "authorization"):
        schedule.add_argument("--" + arg, required=True)
    foads = sub.add_parser("foads-run-one", help="run a separately authorized FOADS P0 category")
    for arg in ("config", "bundle", "authorization", "category", "attempt-id", "kng-source-root"):
        foads.add_argument("--" + arg, required=True)
    foads.add_argument("--locator-manifest", required=True)
    foads_eval = sub.add_parser("foads-evaluate-one", help="evaluate one valid sealed FOADS P0 attempt in Stage B")
    for arg in ("attempt-root", "category-bundle", "annotation", "authorization", "output-dir"):
        foads_eval.add_argument("--" + arg, required=True)
    aggregate = sub.add_parser("foads-aggregate-stage-b", help="aggregate registered Real-IAD P0 FOADS Stage B results")
    aggregate.add_argument("--config", required=True)
    aggregate.add_argument("--evaluation-summary", action="append", default=[], required=True)
    aggregate.add_argument("--output-dir", required=True)
    verify = sub.add_parser("verify-attempt", help="verify a sealed attempt receipt for a later Stage B gate")
    verify.add_argument("--status", required=True)
    verify.add_argument("--expected-identity", required=True)
    verify.add_argument("--kind", choices=("principal_stage_a", "foads_kng_p0"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            config = read_json(args.config)
            report = validate_config(config,
                current_head=None if args.allow_unbound_template else current_git_head(Path.cwd()),
                require_source_binding=not args.allow_unbound_template)
            report["execution_enabled"] = bool(report["source_bound"] and not args.allow_unbound_template)
            print(json.dumps(report, sort_keys=True))
        elif args.command == "list":
            config = read_json(args.config)
            report = validate_config(config, require_source_binding=False)
            if report["source_bound"]:
                validate_config(config, current_head=current_git_head(Path.cwd()), require_source_binding=True)
            for row in config["trajectories"]:
                print(f"{row['category']}\t{row['condition']}\t{row['test_event_count']}\t{row['query_cap_per_principal_arm']}\t{row['role_manifest_sha256']}\t{row['stream_manifest_sha256']}")
        elif args.command == "resources":
            config = read_json(args.config)
            validate_config(config, require_source_binding=False)
            print(json.dumps(describe_resources(config, args.output_path), indent=2, sort_keys=True))
        elif args.command == "run-one":
            result = execute_principal_one(config_path=args.config, bundle_path=args.bundle,
                authorization_path=args.authorization, category=args.category, condition=args.condition,
                attempt_id=args.attempt_id)
            print(json.dumps(result, sort_keys=True))
        elif args.command == "run-schedule":
            result = execute_sequential_schedule(config_path=args.config,
                launch_schedule_path=args.launch_schedule, authorization_path=args.authorization)
            print(json.dumps({"status": "ALL_SEALED_COMPLETE", "count": len(result)}, sort_keys=True))
        elif args.command == "foads-run-one":
            from .foads_execution import execute_foads_one
            result = execute_foads_one(config_path=args.config, bundle_path=args.bundle,
                locator_manifest_path=args.locator_manifest, authorization_path=args.authorization,
                category=args.category, attempt_id=args.attempt_id, kng_source_root=args.kng_source_root)
            print(json.dumps(result, sort_keys=True))
        elif args.command == "foads-evaluate-one":
            from .foads_evaluation import evaluate_foads_sealed_category
            result = evaluate_foads_sealed_category(
                attempt_root=args.attempt_root, category_bundle_path=args.category_bundle,
                annotation_path=args.annotation, stage_b_authorization=args.authorization,
                output_directory=args.output_dir)
            print(json.dumps(result, sort_keys=True))
        elif args.command == "foads-aggregate-stage-b":
            from .foads_evaluation import aggregate_foads_stage_b
            result = aggregate_foads_stage_b(
                config_path=args.config, evaluation_summaries=args.evaluation_summary,
                output_directory=args.output_dir)
            print(json.dumps(result, sort_keys=True))
        elif args.command == "verify-attempt":
            identity = read_json(args.expected_identity)
            if args.kind == "foads_kng_p0":
                from .foads_execution import verify_foads_category_seal
                seal = verify_foads_category_seal(Path(args.status).parent,
                                                  expected_identity=identity)
                seal_path = Path(args.status).parent / "foads_category_seal.json"
            else:
                result = verify_attempt_receipt(args.status, expected_identity=identity,
                                                expected_kind=args.kind)
                seal, seal_path = result["seal"], result["seal_path"]
            print(json.dumps({"status": "SEALED_COMPLETE", "seal_sha256": sha256_file(seal_path)}))
        return 0
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

