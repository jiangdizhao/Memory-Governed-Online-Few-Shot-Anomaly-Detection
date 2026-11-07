"""Frozen FS4 joint-campaign contract and one-registration/three-arm builder."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .fs4_durable_ledger import canonical, digest

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "docs/calibration/fs4_joint_campaign/CAMPAIGN_SPEC.json"
ARMS_PATH = ROOT / "docs/calibration/fs4_joint_campaign/ARM_REGISTRY.json"
HISTORICAL_RECEIPT = ROOT / "docs/calibration/fs4_threshold/HARDENING_R1_GPU_PARITY_MANIFEST_V3.json"
EXPECTED_CATEGORIES = ("audiojack", "regulator", "switch", "rolled_strip_base")
EXPECTED_EVENTS = {"audiojack": 3815, "regulator": 3855, "switch": 3820, "rolled_strip_base": 3805}
EXPECTED_GROUPS = {name: n // 5 for name, n in EXPECTED_EVENTS.items()}
ARM_IDS = ("S", "CL", "CE")
SOURCE_FINGERPRINT_FILES = (
    "experiments/calibration/fs4_durable_ledger.py",
    "experiments/calibration/fs4_durable_runtime.py",
    "experiments/calibration/fs4_checkpoint.py",
    "experiments/calibration/fs4_threshold_integration.py",
    "experiments/calibration/fs4_threshold_core.py",
    "experiments/calibration/fs4_threshold_controllers.py",
    "experiments/calibration/fs4_stochastic_bea.py",
    "experiments/calibration/fs4_source_excluded_scorer.py",
    "experiments/calibration/fs4_greedy_provenance.py",
    "experiments/calibration/fs4_r2_context.py",
    "experiments/calibration/fs4_r4_blinded.py",
    "experiments/calibration/fs4_r4_broker.py",
    "experiments/calibration/fs4_r4_preparation.py",
    "experiments/calibration/fs4_r4_worker.py",
    "experiments/calibration/fs4_research_context.py",
    "experiments/calibration/fs4_research_ledger.py",
    "experiments/calibration/fs4_research_stage_a_verify.py",
    "experiments/calibration/fs4_r4_mapping_verify.py",
    "experiments/calibration/fs4_r4_evaluator.py",
    "experiments/calibration/fs4_joint_campaign_contract.py",
    "experiments/calibration/fs4_joint_campaign.py",
    "experiments/calibration/fs4_joint_campaign_evaluator.py",
    "experiments/calibration/fs4_joint_campaign_resources.py",
)
HEX = re.compile(r"[0-9a-f]{64}")


class CampaignContractError(ValueError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def spec_identity(spec: Mapping[str, Any] | None = None) -> str:
    value = copy.deepcopy(dict(spec if spec is not None else load_spec()))
    value.pop("spec_sha256", None)
    return digest(value)


def load_spec() -> dict[str, Any]:
    try:
        value = json.loads(SPEC_PATH.read_bytes())
    except Exception as exc:
        raise CampaignContractError("frozen campaign specification is unreadable") from exc
    validate_spec(value)
    return value


def validate_spec(spec: Mapping[str, Any]) -> None:
    if not isinstance(spec, Mapping) or spec.get("schema") != "FS4_FINAL_JOINT_CAMPAIGN_SPEC_V1":
        raise CampaignContractError("wrong campaign specification schema")
    if tuple(row.get("name") for row in spec.get("categories", ())) != EXPECTED_CATEGORIES:
        raise CampaignContractError("registered category identities or order changed")
    if {row["name"]: row["expected_test_events"] for row in spec["categories"]} != EXPECTED_EVENTS:
        raise CampaignContractError("registered category event counts changed")
    if {row["name"]: row["source_groups"] for row in spec["categories"]} != EXPECTED_GROUPS:
        raise CampaignContractError("registered category group counts changed")
    if spec.get("totals") != {"events": 15295, "source_groups": 3059, "views_per_group": 5}:
        raise CampaignContractError("registered population totals changed")
    arms = spec.get("arms")
    expected = (
        ("S", "SHARED_TRAJECTORY_THRESHOLD_ONLY", None),
        ("CL", "COUPLED_QUERY_ONLY_FROZEN_WRITER", "L"),
        ("CE", "COUPLED_QUERY_ONLY_FROZEN_WRITER", "E"),
    )
    if not isinstance(arms, list) or tuple((a.get("trajectory_id"), a.get("profile"), a.get("controller")) for a in arms) != expected:
        raise CampaignContractError("frozen S/CL/CE arm registry changed")
    arm_doc = json.loads(ARMS_PATH.read_bytes())
    if (arm_doc.get("schema") != "FS4_FINAL_JOINT_CAMPAIGN_ARM_REGISTRY_V1" or
            [(a.get("trajectory_id"), a.get("profile"), a.get("controller")) for a in arm_doc.get("arms", [])] !=
            [(a.get("trajectory_id"), a.get("profile"), a.get("controller")) for a in arms] or
            arm_doc.get("shared_registration", {}).get("registration_count") != 1 or
            arm_doc.get("shared_registration", {}).get("query_seed") != 37):
        raise CampaignContractError("human arm registry differs from the machine specification")
    support = spec.get("source_model_support", {})
    native = support.get("native_fixed_bank", {})
    if (support.get("support_seed") != 0 or support.get("support_count_per_category") != 4 or
            native.get("shape") != [8192, 4096] or native.get("dtype") != "torch.float16" or
            native.get("additional_normal_images") != 0 or native.get("initial_greedy_empty") is not True or
            native.get("initial_c1_empty") is not True):
        raise CampaignContractError("frozen four-Support/native fixed-bank identity changed")
    model = support.get("model", {})
    if (model.get("model_state_sha256") != "1f26464ba4055403e550f1d5c1333fb012631d98661df6993cabea730d17e3f9" or
            model.get("checkpoint_sha256") != "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51" or
            model.get("hub_source_tree_sha256") != "b7c0037f90671c4932393bd56de8c8a205b29a6eb22dea563b6c8a88c471da15" or
            model.get("native_config_sha256") != "ba3fa736734c2c6c3789f485f3288c5ae77177c68b6b3e24de7e257e447ced87"):
        raise CampaignContractError("frozen native model/configuration identities changed")
    q = spec.get("query_and_thresholds", {})
    if (q.get("policy") != "STOCHASTIC_BEA_GROUP_SLOT_V1" or q.get("seed") != 37 or
            q.get("target_image_rate") != 0.037 or q.get("max_executed_queries_per_source_group") != 1 or
            q.get("minimum_normal_source_groups") != 12 or q.get("minimum_clipped_weight_ess") != 12.0 or
            q.get("threshold_step_clip") != 0.01 or q.get("quantile_probability") != 0.9 or
            q.get("weights") != "min(1/pi, 50)"):
        raise CampaignContractError("frozen BEA or threshold estimator settings changed")
    if spec.get("temporal_analysis", {}).get("sliding_window_groups") != 50 or spec.get("temporal_analysis", {}).get("sliding_stride_groups") != 10:
        raise CampaignContractError("frozen temporal window changed")
    if spec.get("resource_controls", {}).get("checkpoint_every_completed_events") != 1000:
        raise CampaignContractError("frozen checkpoint frequency changed")
    resource = spec.get("resource_controls", {})
    if (resource.get("checkpoint_count_upper_bound") != 69 or
            resource.get("checkpoint_feature_bytes_upper_bound") != 222964482048 or
            resource.get("checkpoint_native_and_controller_overhead_upper_bound") != 18522046464 or
            resource.get("checkpoint_non_cache_allowance_bytes_per_snapshot") != 268435456 or
            resource.get("checkpoint_bytes_before_reserve") != 241486528512 or
            resource.get("checkpoint_bytes_with_25_percent_reserve") != 301858160640 or
            resource.get("additional_ledger_mapping_and_completion_reserve_bytes") != 2147483648 or
            resource.get("max_exact_e_rescore_invocations_per_e_trajectory_upper_bound") != 1171243 or
            resource.get("max_exact_e_rescore_invocations_campaign_upper_bound") != 2342486 or
            resource.get("e_cache_bytes_at_capacity") != 13153337344 or
            resource.get("resource_benchmark_required") is not True):
        raise CampaignContractError("frozen resource/checkpoint budget changed")
    identity = spec_identity(spec)
    if spec.get("spec_sha256") != identity:
        raise CampaignContractError("frozen specification digest differs")
    expected_support = spec.get("source_model_support", {}).get("support_records_by_category", {})
    if set(expected_support) != set(EXPECTED_CATEGORIES):
        raise CampaignContractError("four-Support receipt identities are incomplete")
    receipt = json.loads(HISTORICAL_RECEIPT.read_bytes())
    if _sha(HISTORICAL_RECEIPT.read_bytes()) != spec["source_model_support"]["historical_receipt_sha256"]:
        raise CampaignContractError("historical four-Support receipt file digest differs")
    if receipt.get("status") != "PASS" or receipt.get("test_images_read") != 0 or receipt.get("test_outcomes_read") != 0:
        raise CampaignContractError("historical Support reference is not an acceptable identity receipt")
    for category in EXPECTED_CATEGORIES:
        frozen = expected_support[category]
        actual = receipt["categories"][category]
        if (frozen.get("fixed_bank_sha256") != actual["fixed_evidence"]["expected_sha256"] or
                frozen.get("support_groups_sha256") != actual["support_groups_sha256"] or
                frozen.get("support_images_sha256") != actual["support_images_sha256"] or
                frozen.get("support_role_receipt_sha256") != actual["support_role_receipt_sha256"] or
                frozen.get("support_image_content_sha256") != actual["support_image_content_sha256"]):
            raise CampaignContractError(f"historical Support reference differs for {category}")


def validate_registration(registration: Mapping[str, Any], *, spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Check a completed R4 public registration against the frozen population."""
    spec = dict(spec or load_spec())
    if registration.get("schema") != "FS4_R4_BLINDED_EXECUTION_V1":
        raise CampaignContractError("one completed R4 blinded registration is required")
    if registration.get("query_seed") != 37 or registration.get("category_order") != list(EXPECTED_CATEGORIES):
        raise CampaignContractError("blinded registration category order or query seed differs")
    if set(registration.get("categories", {})) != set(EXPECTED_CATEGORIES):
        raise CampaignContractError("registration must contain exactly the four frozen categories")
    supports_by_category = spec["source_model_support"]["support_records_by_category"]
    all_events, all_groups = set(), set()
    for category in EXPECTED_CATEGORIES:
        info = registration["categories"][category]
        supports, events = info.get("supports"), info.get("events")
        if not isinstance(supports, list) or len(supports) != 4 or not isinstance(events, list):
            raise CampaignContractError(f"{category}: four Support records and full event stream required")
        support_hashes = [row.get("content_sha256") for row in supports]
        if support_hashes != supports_by_category[category]["support_image_content_sha256"]:
            raise CampaignContractError(f"{category}: Support content identity differs from frozen receipt")
        if len(events) != EXPECTED_EVENTS[category]:
            raise CampaignContractError(f"{category}: event stream is incomplete")
        if any("label" in row or "original_path" in row or "original_group" in row for row in events):
            raise CampaignContractError("public event registration contains private outcome identity")
        if len({row.get("event") for row in events}) != EXPECTED_EVENTS[category]:
            raise CampaignContractError(f"{category}: duplicated event identity")
        for start in range(0, len(events), 5):
            block = events[start:start + 5]
            if (len(block) != 5 or len({row.get("group") for row in block}) != 1 or
                    [row.get("view") for row in block] != [1, 2, 3, 4, 5]):
                raise CampaignContractError(f"{category}: stream must contain contiguous ordered five-view groups")
            group = block[0]["group"]
            if group in all_groups:
                raise CampaignContractError("Source Group identity is reused across categories")
            all_groups.add(group)
        for row in events:
            event = row.get("event")
            if event in all_events:
                raise CampaignContractError("opaque event identity is reused across categories")
            all_events.add(event)
    return copy.deepcopy(dict(registration))


def validate_support_preflight(receipt: Mapping[str, Any], *, campaign_id: str,
                               spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Check the separately generated, actual four-Support native receipt."""
    spec = dict(spec or load_spec())
    required = {"schema", "campaign_id", "spec_sha256", "model_state_sha256", "checkpoint_sha256",
                "hub_source_tree_sha256", "native_config_sha256", "categories"}
    if not isinstance(receipt, Mapping) or set(receipt) != required or receipt.get("schema") != "FS4_JOINT_SUPPORT_PREFLIGHT_V1":
        raise CampaignContractError("actual campaign four-Support preflight receipt schema differs")
    model = spec["source_model_support"]["model"]
    expected_top = {"campaign_id": campaign_id, "spec_sha256": spec_identity(spec),
                    "model_state_sha256": model["model_state_sha256"],
                    "checkpoint_sha256": model["checkpoint_sha256"],
                    "hub_source_tree_sha256": model["hub_source_tree_sha256"],
                    "native_config_sha256": model["native_config_sha256"]}
    if any(receipt.get(key) != value for key, value in expected_top.items()):
        raise CampaignContractError("actual Support preflight is for another campaign or model")
    if set(receipt["categories"]) != set(EXPECTED_CATEGORIES):
        raise CampaignContractError("Support preflight must cover all four categories")
    result = copy.deepcopy(dict(receipt))
    for category in EXPECTED_CATEGORIES:
        row = receipt["categories"][category]
        keys = {"absolute_paths", "support_groups", "support_content_sha256", "fixed_bank_sha256",
                "support_self_scores", "mean", "sample_std_ddof1", "theta0", "writer_sigma",
                "additional_normal_images", "initial_greedy_empty", "initial_c1_empty"}
        frozen = spec["source_model_support"]["support_records_by_category"][category]
        if not isinstance(row, Mapping) or set(row) != keys:
            raise CampaignContractError(f"{category}: Support preflight category schema differs")
        paths, groups = row["absolute_paths"], row["support_groups"]
        if (not isinstance(paths, list) or len(paths) != 4 or len(set(paths)) != 4 or
                any(not isinstance(path, str) or not Path(path).is_absolute() for path in paths) or
                not isinstance(groups, list) or len(groups) != 4 or len(set(groups)) != 4 or
                any(not isinstance(group, str) or HEX.fullmatch(group) is None for group in groups)):
            raise CampaignContractError(f"{category}: exact four absolute Support path identities are required")
        scores = row["support_self_scores"]
        if not isinstance(scores, list) or len(scores) != 4 or any(type(x) not in (int, float) for x in scores):
            raise CampaignContractError(f"{category}: four finite native Support self-scores are required")
        import math
        if any(not math.isfinite(float(x)) for x in scores):
            raise CampaignContractError(f"{category}: Support self-score is non-finite")
        import numpy as np
        values = np.asarray(scores, dtype=np.float64)
        mean = float(values.mean())
        sample_std = float(values.std(ddof=1))
        theta = mean + 2.5 * max(sample_std, 1e-6)
        if (type(row["additional_normal_images"]) is not int or row["additional_normal_images"] != 0 or
                row["support_content_sha256"] != frozen["support_image_content_sha256"] or
                row["fixed_bank_sha256"] != frozen["fixed_bank_sha256"] or
                row["mean"] != mean or row["sample_std_ddof1"] != sample_std or
                row["theta0"] != theta or row["writer_sigma"] != max(sample_std, 0.02) or
                row["additional_normal_images"] != 0 or row["initial_greedy_empty"] is not True or
                row["initial_c1_empty"] is not True):
            raise CampaignContractError(f"{category}: actual Support identity, digest or cold-start formula differs")
    return result


def validate_joint_execution(payload: Mapping[str, Any], ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a per-arm R4 execution view bound to one campaign registration."""
    required = {
        "schema", "campaign_id", "trajectory_id", "run_id", "profile", "controller",
        "scientific_spec_sha256", "campaign_registration_sha256", "support_preflight_receipt_sha256",
        "support_preflight_receipt_path", "support_preflight_reference_by_category", "r7_seed0_order_sha256",
        "source_manifest_identity_sha256", "protocol", "source_commit", "query_seed",
        "model_state_sha256", "checkpoint_sha256", "hub_source_tree_sha256", "scorer_config_sha256",
        "preflight_receipt_sha256", "dataset_root", "image_root", "category_order", "categories",
        "source_mapping_sha256", "dataset_identity_sha256",
    }
    if set(payload) != required or payload.get("schema") != "FS4_JOINT_CAMPAIGN_EXECUTION_V1":
        raise CampaignContractError("joint campaign execution registration has an invalid schema")
    spec = load_spec()
    trajectory = payload["trajectory_id"]
    arm = next((row for row in spec["arms"] if row["trajectory_id"] == trajectory), None)
    if arm is None or payload["profile"] != arm["profile"] or payload["controller"] != arm["controller"]:
        raise CampaignContractError("joint trajectory identity/profile/controller differs")
    if payload["run_id"] != ledger.get("run_id") or payload["profile"] != ledger.get("profile"):
        raise CampaignContractError("joint trajectory run identity differs from its ledger")
    if ledger.get("controller") != payload["controller"]:
        raise CampaignContractError("joint controller differs from its ledger")
    for key in ("campaign_registration_sha256", "support_preflight_receipt_sha256", "r7_seed0_order_sha256", "source_manifest_identity_sha256",
                "scientific_spec_sha256", "dataset_identity_sha256", "source_mapping_sha256",
                "model_state_sha256", "checkpoint_sha256", "hub_source_tree_sha256", "scorer_config_sha256",
                "preflight_receipt_sha256"):
        if not isinstance(payload[key], str) or HEX.fullmatch(payload[key]) is None:
            raise CampaignContractError("joint campaign digest is invalid")
    if payload["scientific_spec_sha256"] != spec_identity():
        raise CampaignContractError("joint execution uses a different frozen scientific specification")
    if (payload["campaign_id"] != ledger.get("campaign_id") or
            payload["trajectory_id"] != ledger.get("trajectory_id") or
            payload["scientific_spec_sha256"] != ledger.get("scientific_spec_sha256") or
            payload["campaign_registration_sha256"] != ledger.get("campaign_registration_sha256") or
            payload["support_preflight_receipt_sha256"] != ledger.get("support_preflight_receipt_sha256") or
            payload["r7_seed0_order_sha256"] != ledger.get("r7_seed0_order_sha256")):
        raise CampaignContractError("joint execution and ledger campaign identities differ")
    # The common registration is immutable and tied to image resources. The
    # per-arm wrapper changes only run/profile metadata, never opaque identities.
    from .fs4_r2_security import WindowsPrivateStore
    image_root = Path(payload["image_root"])
    registered = json.loads(WindowsPrivateStore(image_root, create=False).read("registration.json"))
    if digest(registered) != payload["campaign_registration_sha256"]:
        raise CampaignContractError("campaign registration digest differs from shared R4 resources")
    common = ("dataset_root", "image_root", "query_seed", "model_state_sha256", "checkpoint_sha256",
              "hub_source_tree_sha256", "scorer_config_sha256", "preflight_receipt_sha256",
              "category_order", "categories", "source_mapping_sha256", "dataset_identity_sha256")
    if any(payload[key] != registered[key] for key in common):
        raise CampaignContractError("per-arm execution modified common blinded registration")
    validate_registration(registered, spec=spec)
    from .fs4_r4_blinded import validate_blinded_execution
    base_ledger = copy.deepcopy(dict(ledger))
    for key in ("run_id", "protocol", "source_commit", "query_seed", "model_state_sha256", "dataset_identity_sha256"):
        base_ledger[key] = registered[key]
    base_ledger["profile"] = "SHARED_TRAJECTORY_THRESHOLD_ONLY"
    base_ledger.pop("controller", None)
    validate_blinded_execution(registered, base_ledger)
    if payload["query_seed"] != 37 or payload["category_order"] != list(EXPECTED_CATEGORIES):
        raise CampaignContractError("joint campaign must use the frozen four-category seed-37 registration")
    if payload["campaign_id"] not in payload["run_id"] or not payload["run_id"].endswith("-" + trajectory):
        raise CampaignContractError("joint run IDs must be distinct campaign-arm identities")
    if (ledger.get("scientific_spec_sha256") != spec_identity(spec) or
            ledger.get("scientific_contract_sha256") != ledger.get("source_fingerprints", {}).get(
                "docs/calibration/fs4_threshold/SCIENTIFIC_CONTRACT.md") or
            ledger.get("query_seed") != 37 or ledger.get("expected_events_by_category") != EXPECTED_EVENTS):
        raise CampaignContractError("joint ledger scientific contract or population differs")
    receipt_path = Path(payload["support_preflight_receipt_path"])
    if not receipt_path.is_absolute() or not receipt_path.is_file():
        raise CampaignContractError("campaign-bound actual Support preflight receipt is unavailable")
    receipt_bytes = receipt_path.read_bytes()
    if _sha(receipt_bytes) != payload["support_preflight_receipt_sha256"]:
        raise CampaignContractError("campaign-bound Support preflight receipt digest differs")
    support_receipt = validate_support_preflight(json.loads(receipt_bytes), campaign_id=payload["campaign_id"], spec=spec)
    expected_reference = {}
    for category in EXPECTED_CATEGORIES:
        row = support_receipt["categories"][category]
        expected_reference[category] = {"support_groups": row["support_groups"],
            "support_content_sha256": row["support_content_sha256"], "support_self_scores": row["support_self_scores"],
            "mean": row["mean"], "sample_std_ddof1": row["sample_std_ddof1"], "theta0": row["theta0"],
            "writer_sigma": row["writer_sigma"], "fixed_support_sha256": row["fixed_bank_sha256"],
            "preflight_receipt_sha256": payload["support_preflight_receipt_sha256"], "additional_normal_images": 0}
    if payload["support_preflight_reference_by_category"] != expected_reference:
        raise CampaignContractError("joint execution changed the campaign four-Support preflight evidence")
    for category in EXPECTED_CATEGORIES:
        receipt_row = support_receipt["categories"][category]
        support_rows = registered["categories"][category]["supports"]
        expected_paths = [str(Path(registered["dataset_root"]).absolute() / row["relative_path"])
                          for row in support_rows]
        if (receipt_row["support_groups"] != [row["group"] for row in support_rows] or
                receipt_row["absolute_paths"] != expected_paths):
            raise CampaignContractError(f"{category}: actual four-Support paths/groups differ from registration")
    expected_support = spec["source_model_support"]["support_records_by_category"]
    expected_fixed = {category: expected_support[category]["fixed_bank_sha256"] for category in EXPECTED_CATEGORIES}
    expected_groups = {category: [item["group"] for item in registered["categories"][category]["supports"]]
                       for category in EXPECTED_CATEGORIES}
    expected_streams = {category: digest([{"group": row["group"], "event": row["event"]}
                                          for row in registered["categories"][category]["events"]])
                        for category in EXPECTED_CATEGORIES}
    if (ledger.get("fixed_support_sha256_by_category") != expected_fixed or
            ledger.get("support_groups_by_category") != expected_groups or
            ledger.get("stream_sha256_by_category") != expected_streams):
        raise CampaignContractError("joint ledger Support or stream identities differ from the common registration")
    model = spec["source_model_support"]["model"]
    if (payload["model_state_sha256"] != model["model_state_sha256"] or
            payload["checkpoint_sha256"] != model["checkpoint_sha256"] or
            payload["hub_source_tree_sha256"] != model["hub_source_tree_sha256"] or
            payload["scorer_config_sha256"] != model["native_config_sha256"] or
            payload["preflight_receipt_sha256"] != spec["source_model_support"]["historical_receipt_sha256"]):
        raise CampaignContractError("joint model or native Support receipt differs from the freeze")
    for key in ("run_id", "source_commit", "query_seed", "model_state_sha256", "dataset_identity_sha256"):
        if payload[key] != ledger.get(key):
            raise CampaignContractError("joint blinded identity differs from its ledger")
    if payload["scorer_config_sha256"] != "ba3fa736734c2c6c3789f485f3288c5ae77177c68b6b3e24de7e257e447ced87":
        raise CampaignContractError("native scorer configuration differs from the frozen profile")
    if not Path(payload["dataset_root"]).is_absolute() or not Path(payload["image_root"]).is_absolute():
        raise CampaignContractError("joint R4 roots must be absolute")
    if digest({key: payload[key] for key in ("dataset_root", "image_root", "category_order", "categories", "source_mapping_sha256")}) != payload["dataset_identity_sha256"]:
        raise CampaignContractError("joint execution changed the shared dataset identity")
    return copy.deepcopy(dict(payload))


def source_fingerprints(root: Path = ROOT) -> dict[str, str]:
    from .fs4_durable_ledger import REQUIRED_SOURCE_PATHS
    from .fs4_research_ledger import RESEARCH_SOURCES
    required = set(SOURCE_FINGERPRINT_FILES) | set(REQUIRED_SOURCE_PATHS) | set(RESEARCH_SOURCES)
    result = {}
    for relative in sorted(required):
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise CampaignContractError(f"required source is absent from the freeze: {relative}")
        result[relative.replace("\\", "/")] = _sha(path.read_bytes())
    return result


def build_trajectory_manifests(*, registration: Mapping[str, Any], campaign_id: str,
                               campaign_registration_sha256: str, support_preflight_receipt: Mapping[str, Any],
                               support_preflight_receipt_path: Path, r7_seed0_order_sha256: str,
                               source_manifest_identity_sha256: str, source_commit: str,
                               root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """Derive three manifests from one immutable, already prepared R4 registration."""
    spec = load_spec()
    registration = validate_registration(registration, spec=spec)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{7,80}", campaign_id):
        raise CampaignContractError("campaign ID must be unique and path safe")
    source = source_fingerprints(root)
    support_preflight_receipt = validate_support_preflight(support_preflight_receipt, campaign_id=campaign_id, spec=spec)
    support_preflight_receipt_path = Path(support_preflight_receipt_path).absolute()
    support_receipt_sha = _sha(support_preflight_receipt_path.read_bytes())
    if json.loads(support_preflight_receipt_path.read_bytes()) != support_preflight_receipt:
        raise CampaignContractError("Support preflight path content differs from validated receipt")
    support = spec["source_model_support"]["support_records_by_category"]
    for category in EXPECTED_CATEGORIES:
        row = support_preflight_receipt["categories"][category]
        support_rows = registration["categories"][category]["supports"]
        expected_paths = [str(Path(registration["dataset_root"]).absolute() / item["relative_path"])
                          for item in support_rows]
        if (row["support_groups"] != [item["group"] for item in support_rows] or
                row["absolute_paths"] != expected_paths):
            raise CampaignContractError(f"{category}: actual four-Support registration differs from preflight")
    base_digest = digest(registration)
    if campaign_registration_sha256 != base_digest:
        raise CampaignContractError("campaign registration hash is not the registered R4 payload")
    streams = {category: digest([{"group": row["group"], "event": row["event"]}
                                 for row in registration["categories"][category]["events"]])
               for category in EXPECTED_CATEGORIES}
    outputs: dict[str, dict[str, Any]] = {}
    for arm in spec["arms"]:
        trajectory = arm["trajectory_id"]
        run_id = f"{campaign_id}-{trajectory}"
        execution = {key: copy.deepcopy(registration[key]) for key in (
            "query_seed", "model_state_sha256", "checkpoint_sha256", "hub_source_tree_sha256",
            "scorer_config_sha256", "preflight_receipt_sha256", "dataset_root", "image_root",
            "category_order", "categories", "source_mapping_sha256", "dataset_identity_sha256")}
        execution.update({
            "schema": "FS4_JOINT_CAMPAIGN_EXECUTION_V1", "campaign_id": campaign_id,
            "trajectory_id": trajectory, "run_id": run_id, "profile": arm["profile"],
            "controller": arm["controller"], "scientific_spec_sha256": spec_identity(spec),
            "campaign_registration_sha256": base_digest,
            "support_preflight_receipt_sha256": support_receipt_sha,
            "support_preflight_receipt_path": str(support_preflight_receipt_path),
            "support_preflight_reference_by_category": {
                c: {"support_groups": support_preflight_receipt["categories"][c]["support_groups"],
                    "support_content_sha256": support_preflight_receipt["categories"][c]["support_content_sha256"],
                    "support_self_scores": support_preflight_receipt["categories"][c]["support_self_scores"],
                    "mean": support_preflight_receipt["categories"][c]["mean"],
                    "sample_std_ddof1": support_preflight_receipt["categories"][c]["sample_std_ddof1"],
                    "theta0": support_preflight_receipt["categories"][c]["theta0"],
                    "writer_sigma": support_preflight_receipt["categories"][c]["writer_sigma"],
                    "fixed_support_sha256": support_preflight_receipt["categories"][c]["fixed_bank_sha256"],
                    "preflight_receipt_sha256": support_receipt_sha, "additional_normal_images": 0}
                for c in EXPECTED_CATEGORIES},
            "r7_seed0_order_sha256": r7_seed0_order_sha256,
            "source_manifest_identity_sha256": source_manifest_identity_sha256,
            "protocol": spec["campaign_protocol"], "source_commit": source_commit,
        })
        manifest = {
            "run_id": run_id, "campaign_id": campaign_id, "trajectory_id": trajectory,
            "source_commit": source_commit, "model_state_sha256": registration["model_state_sha256"],
            "source_fingerprints": source,
            "expected_events_by_category": dict(EXPECTED_EVENTS), "profile": arm["profile"],
            "stream_sha256_by_category": streams,
            "scientific_contract_sha256": source["docs/calibration/fs4_threshold/SCIENTIFIC_CONTRACT.md"],
            "scientific_spec_sha256": spec_identity(spec),
            "dataset_identity_sha256": registration["dataset_identity_sha256"],
            "support_groups_by_category": {c: [item["group"] for item in registration["categories"][c]["supports"]]
                                           for c in EXPECTED_CATEGORIES},
            "fixed_support_sha256_by_category": {c: support[c]["fixed_bank_sha256"] for c in EXPECTED_CATEGORIES},
            "query_seed": 37, "execution_scope": "TRUSTED_LOCAL_RESEARCH", "protocol": spec["campaign_protocol"],
            "campaign_registration_sha256": base_digest, "r7_seed0_order_sha256": r7_seed0_order_sha256,
            "support_preflight_receipt_sha256": support_receipt_sha,
            "research_execution_manifest": execution,
        }
        if arm["controller"] is not None:
            manifest["controller"] = arm["controller"]
        outputs[trajectory] = manifest
    return outputs
