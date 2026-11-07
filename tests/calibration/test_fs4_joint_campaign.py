from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.calibration.fs4_durable_ledger import digest
from experiments.calibration.fs4_joint_campaign_contract import (
    EXPECTED_CATEGORIES, EXPECTED_EVENTS, CampaignContractError, load_spec,
    spec_identity, validate_registration, validate_support_preflight, validate_spec,
)
from experiments.calibration.fs4_joint_campaign_evaluator import (
    _mean_defined, _readout_temporal, evaluate_campaign, metrics,
)
from experiments.calibration.fs4_joint_campaign_resources import estimate_campaign_resources, resource_preflight
from experiments.calibration.fs4_joint_campaign import AUTH_SCHEMA, _r7_reordered_registration, _validate_authorization, freeze_only


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _synthetic_registration() -> dict:
    spec = load_spec()
    root = str(Path.cwd().resolve())
    categories = {}
    position = 0
    for cat_index, category in enumerate(EXPECTED_CATEGORIES):
        supports = [{"group": _sha(f"support-{category}-{i}"), "relative_path": f"{category}/TRAIN/{i}.png",
            "content_sha256": content, "width": 32, "height": 32}
            for i, content in enumerate(spec["source_model_support"]["support_records_by_category"][category]
                                         ["support_image_content_sha256"])]
        events = []
        for group_index in range(EXPECTED_EVENTS[category] // 5):
            group = _sha(f"opaque-group-{category}-{group_index}")
            for view in range(1, 6):
                events.append({"group": group, "event": _sha(f"opaque-event-{category}-{group_index}-{view}"),
                    "resource": _sha(f"resource-{category}-{group_index}-{view}") + ".png",
                    "position": position, "view": view, "width": 32, "height": 32,
                    "content_sha256": _sha(f"content-{category}-{group_index}-{view}"),
                    "resource_sha256": _sha(f"bytes-{category}-{group_index}-{view}"),
                    "pixel_sha256": _sha(f"pixels-{category}-{group_index}-{view}")})
                position += 1
        categories[category] = {"supports": supports, "events": events}
    return {"schema": "FS4_R4_BLINDED_EXECUTION_V1", "run_id": "synthetic-r4-run",
        "protocol": "synthetic", "source_commit": "0" * 40, "query_seed": 37,
        "model_state_sha256": spec["source_model_support"]["model"]["model_state_sha256"],
        "checkpoint_sha256": spec["source_model_support"]["model"]["checkpoint_sha256"],
        "hub_source_tree_sha256": spec["source_model_support"]["model"]["hub_source_tree_sha256"],
        "scorer_config_sha256": spec["source_model_support"]["model"]["native_config_sha256"],
        "preflight_receipt_sha256": spec["source_model_support"]["historical_receipt_sha256"],
        "dataset_root": root, "image_root": str(Path(root) / "synthetic-images"),
        "category_order": list(EXPECTED_CATEGORIES), "categories": categories,
        "source_mapping_sha256": _sha("mapping"), "dataset_identity_sha256": _sha("dataset")}


def _support_receipt(registration: dict) -> dict:
    spec = load_spec()
    model = spec["source_model_support"]["model"]
    receipt = {"schema": "FS4_JOINT_SUPPORT_PREFLIGHT_V1", "campaign_id": "synthetic-campaign",
        "spec_sha256": spec_identity(), "model_state_sha256": model["model_state_sha256"],
        "checkpoint_sha256": model["checkpoint_sha256"], "hub_source_tree_sha256": model["hub_source_tree_sha256"],
        "native_config_sha256": model["native_config_sha256"], "categories": {}}
    for category in EXPECTED_CATEGORIES:
        scores = [0.1, 0.2, 0.3, 0.4]
        values = np.asarray(scores, dtype=np.float64)
        mean, std = float(values.mean()), float(values.std(ddof=1))
        support = registration["categories"][category]["supports"]
        fixed = spec["source_model_support"]["support_records_by_category"][category]["fixed_bank_sha256"]
        receipt["categories"][category] = {
            "absolute_paths": [str(Path(registration["dataset_root"]).absolute() / row["relative_path"]) for row in support],
            "support_groups": [row["group"] for row in support],
            "support_content_sha256": [row["content_sha256"] for row in support],
            "fixed_bank_sha256": fixed, "support_self_scores": scores, "mean": mean,
            "sample_std_ddof1": std, "theta0": mean + 2.5 * max(std, 1e-6),
            "writer_sigma": max(std, 0.02), "additional_normal_images": 0,
            "initial_greedy_empty": True, "initial_c1_empty": True,
        }
    return receipt


def test_spec_and_machine_readable_arm_registry_are_frozen():
    spec = load_spec()
    validate_spec(spec)
    assert spec_identity(spec) == spec["spec_sha256"]
    assert sum(EXPECTED_EVENTS.values()) == 15295
    assert sum(n // 5 for n in EXPECTED_EVENTS.values()) == 3059
    changed = copy.deepcopy(spec)
    changed["query_and_thresholds"]["seed"] = 38
    with pytest.raises(CampaignContractError):
        validate_spec(changed)


def test_registration_requires_the_exact_population_and_ordered_five_view_groups():
    registration = _synthetic_registration()
    assert validate_registration(registration)["schema"] == "FS4_R4_BLINDED_EXECUTION_V1"
    changed = copy.deepcopy(registration)
    changed["categories"]["audiojack"]["events"][0]["view"] = 2
    with pytest.raises(CampaignContractError):
        validate_registration(changed)
    changed = copy.deepcopy(registration)
    changed["categories"]["switch"]["events"][4]["group"] = changed["categories"]["switch"]["events"][5]["group"]
    with pytest.raises(CampaignContractError):
        validate_registration(changed)


def test_support_preflight_checks_actual_four_support_bootstrap_and_rejects_changes():
    registration = _synthetic_registration()
    receipt = _support_receipt(registration)
    assert validate_support_preflight(receipt, campaign_id="synthetic-campaign")["schema"] == "FS4_JOINT_SUPPORT_PREFLIGHT_V1"
    changed = copy.deepcopy(receipt)
    changed["categories"]["regulator"]["theta0"] += 0.01
    with pytest.raises(CampaignContractError):
        validate_support_preflight(changed, campaign_id="synthetic-campaign")
    changed = copy.deepcopy(receipt)
    changed["categories"]["audiojack"]["additional_normal_images"] = False
    with pytest.raises(CampaignContractError):
        validate_support_preflight(changed, campaign_id="synthetic-campaign")


def test_default_launcher_is_inert_and_authorizations_are_exact_scope_bound_records(tmp_path):
    assert freeze_only()["scientific_execution_authorized"] is False
    auth = {"schema": AUTH_SCHEMA, "authorization_id": "synthetic-auth-1", "authorized_by": "tester",
        "authorized_at": "2026-11-07T00:00:00Z", "scope": "STAGING", "campaign_id": "synthetic-campaign",
        "scientific_spec_sha256": spec_identity(), "authorization_statement": "Synthetic test only"}
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(auth), encoding="utf-8")
    assert _validate_authorization(path, scope="STAGING", campaign_id="synthetic-campaign")["authorization_id"] == "synthetic-auth-1"
    with pytest.raises(CampaignContractError):
        _validate_authorization(path, scope="STAGE_A", campaign_id="synthetic-campaign")


def test_r7_seed_zero_reconstruction_is_a_bijection_and_preserves_view_order():
    root = str(Path.cwd().resolve())
    categories = {}
    labels = {}
    position = 0
    for category in EXPECTED_CATEGORIES:
        supports = [{"group": _sha(f"train-group-{category}-{i}"), "relative_path": f"{category}/OK/S0001/x{i}.jpg",
            "content_sha256": _sha(f"train-image-{category}-{i}"), "width": 16, "height": 16} for i in range(4)]
        events = []
        for group_index in range(EXPECTED_EVENTS[category] // 5):
            group_key = f"{category}_{group_index + 1:04d}_OK"
            group_sha = _sha("official-sample-group\0" + group_key)
            for view in range(1, 6):
                filename = f"{category}_{group_index + 1:04d}_OK_C{view}_202001010000{view:02d}.jpg"
                events.append({"group": group_sha, "event": _sha(f"original-{category}-{group_index}-{view}"),
                    "relative_path": f"OK/S{group_index + 1:04d}/{filename}",
                    "content_sha256": _sha(f"source-{category}-{group_index}-{view}"),
                    "width": 16, "height": 16, "position": position})
                labels[position] = (group_index + view) % 2
                position += 1
        categories[category] = {"supports": supports, "events": events}
    original = {"schema": "FS4_TRUSTED_EXECUTION_V1", "run_id": "r7-synthetic", "protocol": "synthetic",
        "source_commit": "0" * 40, "query_seed": 37,
        "model_state_sha256": load_spec()["source_model_support"]["model"]["model_state_sha256"],
        "dataset_root": root, "category_order": list(EXPECTED_CATEGORIES), "categories": categories,
        "dataset_identity_sha256": _sha("before")}
    ledger = {"expected_events_by_category": dict(EXPECTED_EVENTS)}
    reordered, reordered_ledger, new_labels, order_sha, source_sha = _r7_reordered_registration(original, ledger, labels)
    assert len(new_labels) == 15295 and set(new_labels) == set(range(15295))
    assert len(order_sha) == len(source_sha) == 64
    assert reordered["categories"]["audiojack"]["events"][0]["position"] == 0
    assert reordered["categories"]["regulator"]["events"][0]["position"] == EXPECTED_EVENTS["audiojack"]
    for category in EXPECTED_CATEGORIES:
        rows = reordered["categories"][category]["events"]
        for index in range(0, len(rows), 5):
            assert [row["position"] % 5 + 1 for row in rows[index:index + 5]] == [1, 2, 3, 4, 5]
        assert reordered_ledger["expected_events_by_category"][category] == EXPECTED_EVENTS[category]


def test_metric_formulas_ties_and_undefined_one_class_inputs():
    result = metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], [0, 1, 0, 1])
    assert result["confusion"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    assert result["ber"] == 0.5 and result["balanced_accuracy"] == 0.5
    assert result["auroc"] == 0.75 and result["average_precision"] == pytest.approx(5.0 / 6.0)
    tied = metrics([0, 1, 0, 1], [0.5] * 4, [0, 0, 1, 1])
    assert tied["auroc"] == 0.5 and tied["average_precision"] == 0.5
    one_class = metrics([0, 0], [0.1, 0.2], [0, 1])
    assert one_class["ber"] is None and one_class["balanced_accuracy"] is None
    assert one_class["undefined_reasons"]["ber"] == "one ground-truth class absent"
    assert _mean_defined([0.1, 0.2, 0.3, 0.4], metric_name="BER", category_count=4)["value"] == 0.25


def test_one_class_temporal_window_keeps_ber_na():
    rows = []
    labels = {}
    for group_index in range(50):
        group = _sha(f"window-group-{group_index}")
        for view in range(1, 6):
            event = _sha(f"window-event-{group_index}-{view}")
            labels[event] = 0
            state = {"rescore_invocations": 0, "rescore_seconds_total": 0.0,
                "effective_sample_size": 0.0, "normal_evidence_groups": 0,
                "cache_bytes": 0, "cache_groups": 0, "revision": 0,
                "suppression": {}, "thresholds": {"G": 0.5, "GC": 0.5}}
            rows.append({"event": event, "group": group, "position": group_index * 5 + view - 1, "view": view,
                "category": "audiojack", "scores": {"G": 0.2, "GC": 0.2}, "decisions": {"L_G": 0},
                "thresholds": {"L_G": 0.5}, "query": {"executed": False, "candidate": 1,
                    "ordinal": view, "design_probability": None, "cost": 0.8, "utility": 0.1},
                "transition": {"audit": {"controllers": {"L": state}}}})
    result = _readout_temporal(rows, "L_G", labels)
    assert result["total_windows"] == 1 and result["valid_ber_windows"] == 0
    assert result["excluded_one_class_ber_windows"] == 1 and result["worst_window_ber"] is None
    assert result["windows"][0]["ber"] is None


def test_resource_budget_is_dimensionally_frozen_and_missing_benchmark_blocks(tmp_path):
    estimate = estimate_campaign_resources()
    assert estimate["registered_events"] == 15295
    assert estimate["trajectory_event_encodings"] == 45885
    assert estimate["e_feature_bytes_per_group"] == 784 * 4096 * 4
    assert estimate["e_feature_bytes_at_capacity"] == 13153337344
    assert estimate["checkpoint_count_upper_bound"] == 69
    assert estimate["checkpoint_native_and_controller_overhead_upper_bound"] == 69 * 256 * 1024**2
    assert estimate["checkpoint_bytes_before_reserve"] == 241486528512
    assert estimate["checkpoint_bytes_with_25_percent_reserve"] == 301858160640
    registration = _synthetic_registration()
    report = resource_preflight(registration=registration, image_root=tmp_path / "images",
        output_root=tmp_path / "output", private_root=tmp_path / "private", benchmark_receipt=None)
    assert report["status"] == "BLOCKED_RESOURCE_PREFLIGHT"
    assert "FULL_NATIVE_DIMENSION_RESOURCE_BENCHMARK_MISSING" in report["reasons"]


def _synthetic_campaign() -> tuple[dict, dict, dict]:
    stage = {}
    labels = {}
    telemetry = {key: [] for key in ("S", "CL", "CE")}
    position = 0
    for trajectory in ("S", "CL", "CE"):
        stage[trajectory] = {"trajectory_id": trajectory, "events": []}
    for category in EXPECTED_CATEGORIES:
        for group_index in range(EXPECTED_EVENTS[category] // 5):
            group = _sha(f"eval-group-{category}-{group_index}")
            for view in range(1, 6):
                label = (group_index + view) % 2
                for trajectory in ("S", "CL", "CE"):
                    event = _sha(f"eval-event-{category}-{group_index}-{view}")
                    labels[event] = label
                    score = 0.9 if label else 0.1
                    if trajectory == "S":
                        decision_map = {f"{controller}_{scorer}": int(score >= 0.5)
                            for controller in ("L", "E", "FrozenTheta") for scorer in ("G", "GC")}
                        threshold_map = {f"{controller}_{scorer}": 0.5
                            for controller in ("L", "E", "FrozenTheta") for scorer in ("G", "GC")}
                        controller_names = ("L", "E")
                    else:
                        controller = "L" if trajectory == "CL" else "E"
                        decision_map = {f"{controller}_{scorer}": int(score >= 0.5) for scorer in ("G", "GC")}
                        threshold_map = {f"{controller}_{scorer}": 0.5 for scorer in ("G", "GC")}
                        controller_names = (controller,)
                    state = {"rescore_invocations": 0, "rescore_seconds_total": 0.0,
                        "effective_sample_size": 0.0, "normal_evidence_groups": 0,
                        "cache_bytes": 0, "cache_groups": 0, "revision": 0,
                        "suppression": {}, "thresholds": {"G": 0.5, "GC": 0.5},
                        "status": {"G": "COLD_START_UNCALIBRATED", "GC": "COLD_START_UNCALIBRATED"}}
                    audit = {"controllers": {name: state for name in controller_names},
                        "normal_version": 0, "correction_version": 0,
                        "greedy_dynamic_prototypes": 0, "correction_prototypes": 0}
                    row = {"event": event, "group": group, "position": position, "view": view,
                        "category": category, "scores": {"G": score, "GC": score},
                        "decisions": decision_map, "thresholds": threshold_map,
                        "query": {"executed": False, "candidate": 1, "ordinal": view,
                            "design_probability": None, "cost": 0.8, "utility": 0.1},
                        "transition": {"audit": audit}}
                    stage[trajectory]["events"].append(row)
                position += 1
        for trajectory in ("S", "CL", "CE"):
            telemetry[trajectory].append({"category": category, "host_peak_working_set_bytes": 1,
                "cuda_peak_allocated_bytes": 1, "rescore_invocations": 0, "cache_bytes": 0})
    return stage, labels, telemetry


def test_all_arm_evaluator_reports_paired_estimands_and_equal_shared_rankings():
    stage, labels, telemetry = _synthetic_campaign()
    report = evaluate_campaign(stage, labels, telemetry)
    assert report["estimands"]["H1_shared_threshold_only"]["four_category_macro_difference"]["value"] == 0.0
    assert report["estimands"]["H2_coupled_system"]["four_category_macro_difference"]["value"] == 0.0
    assert report["arms"]["S"]["four_category_macro"]["L_GC"]["auroc"]["value"] == 1.0
    assert report["arms"]["S"]["four_category_macro"]["E_GC"]["average_precision"]["value"] == 1.0
    assert report["population"]["total_events_per_trajectory"] == 15295
