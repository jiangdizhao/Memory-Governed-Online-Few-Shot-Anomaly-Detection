from __future__ import annotations

import copy
from bisect import bisect_right
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
    _controller_update_summary, _mean_defined, _readout_temporal, evaluate_campaign, metrics,
)
from experiments.calibration.fs4_joint_campaign_resources import estimate_campaign_resources, resource_preflight
from experiments.calibration.fs4_joint_campaign import AUTH_SCHEMA, _r7_reordered_registration, _validate_authorization, freeze_only
from experiments.calibration.fs4_joint_campaign import (
    CAMPAIGN_WALL_LIMIT_SECONDS, TRAJECTORY_WALL_LIMIT_SECONDS, CampaignDeadlineExceeded,
    CampaignTimingLedger, RecoverableExecutionInterruption, _record_incomplete,
    _recovery_eligibility, _validate_complete_stage_a, _validate_terminal_state, run_stage_a,
)
from experiments.calibration import fs4_joint_campaign as campaign_module
from experiments.calibration.fs4_r4_worker import BlindedResearchWorker, ResearchWorkerTimeout


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


def test_staging_private_labels_are_bound_to_registered_official_json(tmp_path, monkeypatch):
    from experiments.calibration import fs4_joint_campaign as campaign
    from experiments.calibration.fs4_durable_ledger import canonical
    from experiments.risk_safe.preflight import parse_image_name
    from test_fs4_hardening_r2 import synthetic_registration

    category = "audiojack"
    source_root = tmp_path / "registered-source"
    ledger, original = synthetic_registration(source_root / "images")
    original["dataset_root"] = str(source_root.resolve())
    for row in [*original["categories"][category]["supports"],
                *original["categories"][category]["events"]]:
        row["relative_path"] = "images/" + row["relative_path"]
    original["dataset_identity_sha256"] = digest({
        "dataset_root": original["dataset_root"],
        "category_order": original["category_order"],
        "categories": original["categories"],
    })
    ledger["dataset_identity_sha256"] = original["dataset_identity_sha256"]
    json_root = source_root / "jsons" / "realiad_jsons"
    json_root.mkdir(parents=True)
    official_rows = []
    frozen_stream = []
    for local_position, event in enumerate(original["categories"][category]["events"]):
        source_path = source_root / Path(event["relative_path"])
        parsed = parse_image_name(source_path, expected_class=category)
        official_rows.append({"category": category, "image_path": parsed.rel_path,
                              "anomaly_class": "OK"})
        frozen_stream.append({"position": local_position, "sample_key": parsed.official_sample_key,
            "opaque_group": event["group"], "camera_id": parsed.camera_id,
            "timestamp": parsed.timestamp, "image_path": str(source_path.resolve()),
            "relative_id": parsed.rel_path})
    json_data = {"meta": {"prefix": "audiojack/", "normal_class": "OK"}, "test": official_rows}
    json_path = json_root / "audiojack.json"
    json_bytes = canonical(json_data)
    json_path.write_bytes(json_bytes)
    json_sha = hashlib.sha256(json_bytes).hexdigest()
    frozen = {"schema": "stage05.frozen_inputs.v1", "freeze_verification_status": "PASS",
        "stage1_or_stage2_run": False, "classes": {category: {
            "class_json_sha256": json_sha, "stream": frozen_stream}}}
    frozen_path = tmp_path / "frozen_inputs.json"
    frozen_bytes = canonical(frozen)
    frozen_path.write_bytes(frozen_bytes)
    ledger["frozen_source_identity"] = {
        "stage05_frozen_inputs_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "real_iad_json_sha256_by_category": {category: json_sha},
    }
    monkeypatch.setattr(campaign, "STAGE05_FROZEN_INPUTS", frozen_path)
    monkeypatch.setattr(campaign, "EXPECTED_CATEGORIES", (category,))
    monkeypatch.setattr(campaign, "EXPECTED_EVENTS", {category: 5})

    verified = campaign._validate_staging_labels_from_official_json(
        original, ledger, {position: 0 for position in range(5)})
    assert verified["verified_private_label_count"] == 5
    assert verified["official_json_sha256_by_category"] == {category: json_sha}
    with pytest.raises(CampaignContractError, match="differs from official JSON"):
        campaign._validate_staging_labels_from_official_json(
            original, ledger, {0: 1, 1: 0, 2: 0, 3: 0, 4: 0})


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
                "threshold_tokens": {"G": None, "GC": None}, "suppression": {},
                "thresholds": {"G": 0.5, "GC": 0.5}}
            rows.append({"event": event, "group": group, "position": group_index * 5 + view - 1, "view": view,
                "category": "audiojack", "scores": {"G": 0.2, "GC": 0.2}, "decisions": {"L_G": 0},
                "thresholds": {"L_G": 0.5, "L_GC": 0.5}, "query": {"executed": False, "candidate": 1,
                    "ordinal": view, "design_probability": None, "cost": 0.8, "utility": 0.1},
                "transition": {"audit": {"controllers": {"L": state}}}})
    result = _readout_temporal(rows, "L_G", labels)
    assert result["total_windows"] == 1 and result["valid_ber_windows"] == 0
    assert result["excluded_one_class_ber_windows"] == 1 and result["worst_window_ber"] is None
    assert result["windows"][0]["ber"] is None


def _update_audit_row(position, *, revision, token_g=None, token_gc=None, theta_g=0.5,
                      theta_gc=0.5, suppression=None, ess=12.0):
    group = _sha(f"update-audit-group-{position // 5}")
    event = _sha(f"update-audit-event-{position}")
    state = {"revision": revision, "threshold_tokens": {"G": token_g, "GC": token_gc},
        "thresholds": {"G": theta_g, "GC": theta_gc}, "suppression": dict(suppression or {}),
        "effective_sample_size": ess, "normal_evidence_groups": revision,
        "rescore_invocations": 0, "rescore_seconds_total": 0.0,
        "cache_bytes": 0, "cache_groups": 0}
    return {"position": position, "event": event, "group": group,
        "thresholds": {"L_G": theta_g, "L_GC": theta_gc},
        "transition": {"audit": {"controllers": {"L": state}}}}


@pytest.mark.parametrize("revision,ess", [(0, 0.0), (11, 11.0), (12, 11.5)])
def test_evidence_revision_without_accepted_token_is_not_an_update(revision, ess):
    rows = [_update_audit_row(i, revision=revision, ess=ess) for i in range(5)]
    result = _controller_update_summary(rows, "L")
    assert result["evidence_revision"] == revision
    assert result["accepted_update_count_G"] == 0
    assert result["accepted_update_count_GC"] == 0
    assert result["events_with_any_accepted_update"] == 0
    assert result["earliest_accepted_update_position_G"] is None
    assert result["earliest_accepted_update_position_GC"] is None


def test_accepted_tokens_count_per_scorer_without_threshold_movement_and_replay_duplication():
    rows = [
        _update_audit_row(10, revision=11, suppression={"INSUFFICIENT_VERIFIED_NORMALS": 22}),
        _update_audit_row(11, revision=12, token_g=[12, "G"], token_gc=[12, "GC"],
                          suppression={"INSUFFICIENT_VERIFIED_NORMALS": 22}),
        _update_audit_row(12, revision=13, token_g=[12, "G"], token_gc=[12, "GC"],
                          suppression={"INSUFFICIENT_VERIFIED_NORMALS": 22}),
        _update_audit_row(13, revision=14, token_g=[14, "G"], token_gc=[12, "GC"],
                          theta_g=0.51, suppression={"INSUFFICIENT_VERIFIED_NORMALS": 22}),
        _update_audit_row(14, revision=15, token_g=[14, "G"], token_gc=[15, "GC"],
                          theta_g=0.51, suppression={"INSUFFICIENT_VERIFIED_NORMALS": 22}),
        _update_audit_row(15, revision=15, token_g=[14, "G"], token_gc=[15, "GC"],
                          theta_g=0.51, suppression={"INSUFFICIENT_VERIFIED_NORMALS": 22}),
    ]
    result = _controller_update_summary(rows, "L")
    assert result["accepted_update_count_G"] == 2
    assert result["accepted_update_count_GC"] == 2
    assert result["earliest_accepted_update_position_G"] == 11
    assert result["earliest_accepted_update_position_GC"] == 11
    assert result["events_with_any_accepted_update"] == 3
    assert result["total_threshold_movement_G"] == pytest.approx(0.01)
    assert result["total_threshold_movement_GC"] == 0.0
    assert result["threshold_holds_by_reason"] == {}
    # A resumed ledger may repeat the exact last audit state. It is not a new token.
    replayed = copy.deepcopy(rows[-1])
    replayed["position"] += 1
    replayed["event"] = _sha("replayed-event")
    replay = _controller_update_summary([replayed], "L",
        prior_state=rows[-1]["transition"]["audit"]["controllers"]["L"])
    assert replay["accepted_update_count_G"] == replay["accepted_update_count_GC"] == 0
    assert replay["events_with_any_accepted_update"] == 0


def test_update_counts_are_correct_at_temporal_segment_and_window_boundaries():
    rows = []
    labels = {}
    for position in range(300):  # 60 groups, each with five contiguous views.
        group_index, view0 = divmod(position, 5)
        token_g = ([51, "G-first"] if position >= 99 else None)
        if position >= 249:
            token_g = [52, "G-second"]
        token_gc = ([52, "GC-first"] if position >= 100 else None)
        if position >= 250:
            token_gc = [53, "GC-second"]
        row = _update_audit_row(position, revision=53 if position >= 250 else (52 if position >= 100 else 51),
            token_g=token_g, token_gc=token_gc)
        row.update(group=_sha(f"boundary-group-{group_index}"), view=view0 + 1,
            category="audiojack", scores={"G": 0.2, "GC": 0.2}, decisions={"L_G": 0},
            thresholds={"L_G": 0.5, "L_GC": 0.5},
            query={"executed": False, "candidate": 1, "ordinal": view0 + 1,
                "design_probability": None, "cost": 0.8, "utility": 0.1})
        rows.append(row)
        labels[row["event"]] = (position + view0) % 2
    early = _controller_update_summary(rows[:100], "L")
    middle = _controller_update_summary(rows[100:200], "L", prior_state=rows[99]["transition"]["audit"]["controllers"]["L"])
    late = _controller_update_summary(rows[200:], "L", prior_state=rows[199]["transition"]["audit"]["controllers"]["L"])
    assert (early["accepted_update_count_G"], early["accepted_update_count_GC"]) == (1, 0)
    assert (middle["accepted_update_count_G"], middle["accepted_update_count_GC"]) == (0, 1)
    assert (late["accepted_update_count_G"], late["accepted_update_count_GC"]) == (1, 1)
    temporal = _readout_temporal(rows, "L_G", labels)
    first, second = temporal["windows"]
    assert first["start_group_index"] == 0 and second["start_group_index"] == 10
    assert (first["threshold_update_diagnostics"]["accepted_update_count_G"],
            first["threshold_update_diagnostics"]["accepted_update_count_GC"]) == (2, 1)
    assert (second["threshold_update_diagnostics"]["accepted_update_count_G"],
            second["threshold_update_diagnostics"]["accepted_update_count_GC"]) == (2, 2)


def test_evaluator_uses_accepted_tokens_for_controller_segments_and_windows():
    stage, labels, telemetry = _synthetic_campaign()
    rows = stage["S"]["events"]
    audio_rows = [row for row in rows if row["category"] == "audiojack"]
    for row in audio_rows:
        index = row["position"]
        state = row["transition"]["audit"]["controllers"]["L"]
        if index >= 1269:
            state["threshold_tokens"]["G"] = [12, "G-first"]
        if index >= 1270:
            state["threshold_tokens"]["GC"] = [12, "GC-first"]
        if index >= 2539:
            state["threshold_tokens"]["G"] = [13, "G-second"]
        if index >= 2540:
            state["threshold_tokens"]["GC"] = [13, "GC-second"]
    report = evaluate_campaign(stage, labels, telemetry)
    controller = report["arms"]["S"]["query_and_controller_diagnostics"]["audiojack"]["controllers"]["L"]
    assert controller["accepted_update_count_G"] == controller["accepted_update_count_GC"] == 2
    assert controller["events_with_any_accepted_update"] == 4
    segments = report["arms"]["S"]["temporal"]["audiojack"]["L_G"]["segments"]
    assert segments["early"]["accepted_update_count_G"] == 1
    assert segments["middle"]["accepted_update_count_G"] == 1
    assert segments["late"]["accepted_update_count_GC"] == 1
    windows = report["arms"]["S"]["temporal"]["audiojack"]["L_G"]["sliding_windows"]["windows"]
    crossing = next(item for item in windows if item["start_group_index"] == 210)
    assert crossing["threshold_update_diagnostics"]["accepted_update_count_G"] == 1
    assert crossing["threshold_update_diagnostics"]["accepted_update_count_GC"] == 1


class _FakeClock:
    def __init__(self):
        self.mono = 10_000_000_000
        self.wall = 1_800_000_000_000_000_000

    def monotonic_ns(self):
        return self.mono

    def time_ns(self):
        return self.wall

    def advance(self, seconds):
        delta = int(seconds * 1_000_000_000)
        self.mono += delta
        self.wall += delta


def test_campaign_timing_ledger_enforces_trajectory_limit_and_persists_restart_budget(tmp_path):
    clock = _FakeClock()
    path = tmp_path / "execution" / "TIMING_LEDGER.jsonl"
    ledger = CampaignTimingLedger(path, campaign_id="timing-test", source_commit="a" * 40, clock=clock)
    ledger.start_trajectory("S")
    clock.advance(2 * 86400)
    ledger.record("CHECKPOINT", {"name": "S-test"})
    ledger.close()
    clock.advance(3 * 86400)  # interrupted process and recovery downtime are charged
    resumed = CampaignTimingLedger(path, campaign_id="timing-test", source_commit="a" * 40, clock=clock)
    assert resumed.unclean_previous_process is False
    budget = resumed.check("S")
    assert budget["campaign_elapsed_ns"] == 5 * 86400 * 1_000_000_000
    assert budget["trajectory_elapsed_ns"]["S"] == 5 * 86400 * 1_000_000_000
    clock.advance(2 * 86400)
    with pytest.raises(CampaignDeadlineExceeded, match="TRAJECTORY_WALL_LIMIT_EXCEEDED"):
        resumed.check("S")
    resumed.close()
    with pytest.raises(CampaignDeadlineExceeded, match="TRAJECTORY_WALL_LIMIT_EXCEEDED"):
        CampaignTimingLedger(path, campaign_id="timing-test", source_commit="a" * 40, clock=clock)


def test_campaign_timing_ledger_counts_s_and_cl_before_ce_and_enforces_campaign_limit(tmp_path):
    clock = _FakeClock()
    path = tmp_path / "TIMING_LEDGER.jsonl"
    ledger = CampaignTimingLedger(path, campaign_id="timing-test", source_commit="b" * 40, clock=clock)
    for trajectory in ("S", "CL"):
        ledger.start_trajectory(trajectory)
        clock.advance(5 * 86400)
        ledger.finish_trajectory(trajectory)
        clock.advance(1.5 * 86400)  # downtime is charged to the whole campaign
        remaining = ledger.check()["campaign_remaining_ns"]
        assert remaining == (CAMPAIGN_WALL_LIMIT_SECONDS - (6.5 if trajectory == "S" else 13) * 86400) * 1_000_000_000
    ledger.start_trajectory("CE")
    assert ledger.check()["campaign_remaining_ns"] == 8 * 86400 * 1_000_000_000
    assert ledger.timeout_seconds("CE") == pytest.approx(7 * 86400)
    clock.advance(6 * 86400)
    ledger.finish_trajectory("CE")
    clock.advance(2 * 86400)
    with pytest.raises(CampaignDeadlineExceeded, match="CAMPAIGN_WALL_LIMIT_EXCEEDED"):
        ledger.check("CE")
    ledger.close()


def test_campaign_timing_ledger_fails_closed_on_tampering(tmp_path):
    clock = _FakeClock()
    path = tmp_path / "TIMING_LEDGER.jsonl"
    ledger = CampaignTimingLedger(path, campaign_id="timing-test", source_commit="c" * 40, clock=clock)
    ledger.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("CAMPAIGN_ORIGIN", "CAMPAIGN_ORIGIN_TAMPERED")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(CampaignContractError, match="timing ledger is malformed"):
        CampaignTimingLedger(path, campaign_id="timing-test", source_commit="c" * 40, clock=clock)


@pytest.mark.parametrize("classification", [
    "UNRECOVERABLE_INTEGRITY_FAILURE", "AMBIGUOUS_LABEL_DISCLOSURE", "DEADLINE_EXHAUSTION",
])
def test_fatal_ambiguous_and_expired_incomplete_states_are_not_resumable(tmp_path, classification):
    execution = tmp_path / "execution"
    execution.mkdir()
    binding = {"campaign_id": "blocked-campaign", "source_freeze_commit": "d" * 40}
    _record_incomplete(execution, "synthetic-blocker", campaign_id=binding["campaign_id"],
        source_commit=binding["source_freeze_commit"], classification=classification)
    with pytest.raises(CampaignContractError, match="fatal, ambiguous or deadline-exhausted"):
        _recovery_eligibility(execution, binding, unclean_previous_process=False)


def test_blinded_worker_timeout_is_bounded_and_terminates_worker_and_broker():
    class FakeProcess:
        def __init__(self):
            self.alive = True
            self.terminated = False
        def is_alive(self): return self.alive
        def terminate(self): self.terminated = True; self.alive = False
        def join(self, timeout=None): pass

    class FakePipe:
        def __init__(self): self.seen_timeout = None; self.closed = False
        def poll(self, timeout): self.seen_timeout = timeout; return False
        def close(self): self.closed = True

    class FakeBroker:
        def __init__(self): self.process = FakeProcess(); self.endpoint = FakePipe()

    worker = object.__new__(BlindedResearchWorker)
    worker._connection = FakePipe()
    worker._process = FakeProcess()
    worker._broker = FakeBroker()
    with pytest.raises(ResearchWorkerTimeout, match="exceeded the remaining wall budget"):
        worker._receive("OK", timeout_seconds=0.125)
    assert worker._connection.seen_timeout == pytest.approx(0.125)
    assert not worker._process.alive and worker._process.terminated
    assert not worker._broker.process.alive and worker._broker.process.terminated


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
                        "threshold_tokens": {"G": None, "GC": None},
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


def test_integrated_s_cl_ce_campaign_completion_and_checkpoint_recovery(tmp_path, monkeypatch):
    """One full-size synthetic launcher exercise: uninterrupted and recovered paths."""
    registration = _synthetic_registration()
    registration_hash = digest(registration)
    spec = load_spec()
    source_commit = "e" * 40
    binding = {"campaign_id": "synthetic-integrated-campaign",
        "campaign_registration_sha256": registration_hash, "source_freeze_commit": source_commit,
        "support_preflight_receipt_sha256": _sha("support receipt"),
        "source_mapping_sha256": registration["source_mapping_sha256"],
        "r7_seed0_order_sha256": _sha("r7 order"),
        "trajectory_manifests": {}}
    manifests = {}
    for trajectory in ("S", "CL", "CE"):
        manifests[trajectory] = {"schema": "synthetic-joint-manifest", "run_id": f"synthetic-{trajectory}",
            "trajectory_id": trajectory, "source_commit": source_commit,
            "campaign_id": binding["campaign_id"],
            "campaign_registration_sha256": registration_hash}
        binding["trajectory_manifests"][trajectory] = digest(manifests[trajectory])
    labels = {event["event"]: (event["position"] // 5) % 2
        for category in EXPECTED_CATEGORIES for event in registration["categories"][category]["events"]}
    registered_by_position = {event["position"]: event for category in EXPECTED_CATEGORIES
        for event in registration["categories"][category]["events"]}
    position_by_event = {event["event"]: event["position"] for event in registered_by_position.values()}
    policy_period = {"S": 11, "CL": 13, "CE": 17}
    def run_scenario(scenario_root: Path, *, injected: str | None):
        execution = scenario_root / "execution"
        scenario_root.mkdir(parents=True, exist_ok=True)
        (scenario_root / "manifests").mkdir()
        for arm in manifests:
            (scenario_root / "manifests" / f"{arm}.json").write_text(
                json.dumps(manifests[arm]), encoding="utf-8")
        resource = {"status": "PASS", "spec_sha256": spec_identity(),
            "registration_sha256": registration_hash}
        resource_path = scenario_root / "resource-preflight.json"
        resource_path.write_bytes(campaign_module.canonical(resource) + b"\n")
        auth = {"schema": AUTH_SCHEMA, "authorization_id": f"synthetic-{scenario_root.name}",
            "authorized_by": "synthetic-test", "authorized_at": "2026-11-07T00:00:00Z",
            "scope": "STAGE_A", "campaign_id": binding["campaign_id"],
            "scientific_spec_sha256": spec_identity(), "authorization_statement": "Synthetic integration only",
            "campaign_registration_sha256": registration_hash,
            "support_preflight_receipt_sha256": binding["support_preflight_receipt_sha256"],
            "resource_preflight_receipt_sha256": campaign_module._sha(resource_path.read_bytes()),
            "trajectory_manifests": binding["trajectory_manifests"]}
        auth_path = scenario_root / "stage-a-auth.json"
        auth_path.write_text(json.dumps(auth), encoding="utf-8")
        arm_states = {}
        checkpoint_inventory = {}
        label_reads = {arm: [] for arm in manifests}
        feedback_commits = {arm: {} for arm in manifests}
        duplicate_feedback_writes = {arm: 0 for arm in manifests}
        disclosed_roots = {}
        injection_fired = {"value": False}

        def make_state(arm, category):
            first = registration["categories"][category]["events"][0]["position"]
            return {"trajectory": arm, "position": first, "phase": "READY", "current_category": category,
                "category_seals": {}, "processed_positions": set(), "queries": {}, "feedback": {},
                "memory_updates": [], "last_hash": "0" * 64}

        class SyntheticWorker:
            def __init__(self, arm, category, run_root, checkpoint, create):
                self.arm, self.category, self.run_root = arm, category, run_root
                state = arm_states.get(arm)
                if state is None:
                    state = make_state(arm, category)
                    arm_states[arm] = state
                elif state["current_category"] is None:
                    state["current_category"] = category
                self.state = state
                self.cursor = state["position"]
                if checkpoint:
                    checkpoint_state = json.loads((run_root / "checkpoints" / checkpoint / "state.json").read_text())
                    self.cursor = checkpoint_state["ledger_offset"]["next_position"]
                self.checkpoints = checkpoint_inventory.setdefault((arm, category), [])

            def step(self, position, *, timeout_seconds=None):
                assert position == self.cursor
                registered = registered_by_position[position]
                event_id = registered["event"]
                query = registered["view"] == 1 and (position // 5) % policy_period[self.arm] == 0
                prior_query = self.state["queries"].get(event_id)
                if prior_query is not None and prior_query != query:
                    raise AssertionError("label-blind query tape changed during replay")
                self.state["queries"][event_id] = query
                if injected == "ambiguous" and self.arm == "S" and position == 0 and not injection_fired["value"]:
                    injection_fired["value"] = True
                    self.state["phase"] = "LABEL_INTENT"
                    raise ResearchWorkerTimeout("synthetic timeout during label-intent transaction")
                if query and event_id not in self.state["feedback"]:
                    label_reads[self.arm].append(event_id)
                    label = labels[event_id]  # disclosed only after this arm's query decision
                    self.state["feedback"][event_id] = label
                    feedback_commits[self.arm][event_id] = feedback_commits[self.arm].get(event_id, 0) + 1
                    if label == 0:
                        self.state["memory_updates"].append(event_id)
                elif query:
                    # Replayed legal feedback is restored from the arm's own durable map.
                    if event_id not in self.state["feedback"]:
                        duplicate_feedback_writes[self.arm] += 1
                if position not in self.state["processed_positions"]:
                    self.state["processed_positions"].add(position)
                    self.state["last_hash"] = _sha(self.state["last_hash"] + event_id)
                self.cursor = position + 1
                self.state["position"] = max(self.state["position"], self.cursor)
                if injected == "recoverable" and self.arm == "CE" and position == 1020 and not injection_fired["value"]:
                    injection_fired["value"] = True
                    raise RecoverableExecutionInterruption("synthetic interruption after committed queried-normal feedback")
                return {"prediction": {"position": position, "event": event_id, "group": registered["group"]}}

            def checkpoint(self, name, *, timeout_seconds=None):
                path = self.run_root / "checkpoints" / name
                state_bytes = json.dumps({"ledger_offset": {"next_position": self.cursor},
                    "manifest_sha256": digest(manifests[self.arm])}, sort_keys=True).encode()
                tensor_bytes = b"synthetic-native-and-controller-state"
                marker_bytes = json.dumps({"state_sha256": campaign_module._sha(state_bytes),
                    "tensor_sha256": campaign_module._sha(tensor_bytes)}, sort_keys=True).encode()
                path.mkdir(parents=True, exist_ok=True)
                for name_part, data in (("state.json", state_bytes), ("tensors.npz", tensor_bytes),
                                        ("COMMITTED.json", marker_bytes)):
                    file_path = path / name_part
                    if file_path.exists():
                        assert file_path.read_bytes() == data
                    else:
                        file_path.write_bytes(data)
                record = (self.cursor, name)
                if record not in self.checkpoints:
                    self.checkpoints.append(record)

            def seal_category(self, *, timeout_seconds=None):
                self.state["category_seals"][self.category] = _sha(f"{self.arm}:{self.category}:category-seal")
                self.state["current_category"] = None
                self.state["phase"] = "READY"
                return self.state["category_seals"][self.category]

            def seal_global_a(self, *, timeout_seconds=None):
                self.state["phase"] = "GLOBAL_A"
                return _sha(f"{self.arm}:global-a")

            def stats(self, *, timeout_seconds=None):
                return {"active_worker_seconds": 0.01, "host_peak_working_set_bytes": 1024,
                    "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None,
                    "rescore_invocations": 0, "rescore_seconds": 0.0, "cache_bytes": 0}

            def close(self, *, timeout_seconds=0.1):
                pass

        def inspect_ledger(run_root, manifest):
            arm = Path(run_root).name
            state = arm_states.get(arm)
            if state is None:
                return None
            prefix = {"run_id": manifests[arm]["run_id"], "manifest_sha256": digest(manifests[arm]),
                "position": state["position"], "phase": state["phase"], "last_hash": state["last_hash"]}
            return {"position": state["position"], "phase": state["phase"],
                "current_category": state["current_category"], "category_seals": dict(state["category_seals"]),
                "last_hash": state["last_hash"], "prefix_identity": prefix}

        def start_worker(*, campaign_root, metadata, trajectory, manifest_path, manifest, category,
                         checkpoint, create, timeout_seconds=None):
            run_root = Path(campaign_root) / "execution" / "runs" / trajectory
            run_root.mkdir(parents=True, exist_ok=True)
            disclosure_root = Path(campaign_root) / "execution" / "disclosures" / trajectory
            disclosure_root.mkdir(parents=True, exist_ok=True)
            disclosed_roots[trajectory] = disclosure_root
            worker = SyntheticWorker(trajectory, category, run_root, checkpoint, create)
            return worker, Path(campaign_root) / "synthetic-config.json", run_root, disclosure_root

        def checkpoint_candidates(run_root, category, manifest, low, high):
            arm = Path(run_root).name
            return sorted((position, name) for position, name in checkpoint_inventory.get((arm, category), [])
                if low <= position <= high)

        def projection(campaign_root, trajectory, *, write_projection, timeout_seconds=None):
            state = arm_states[trajectory]
            if write_projection and (state["phase"] != "GLOBAL_A" or
                    set(state["category_seals"]) != set(EXPECTED_CATEGORIES)):
                raise CampaignContractError("synthetic Stage A projection requires all four category seals")
            model = spec["source_model_support"]["model"]
            controller_names = ("L", "E") if trajectory == "S" else (("L",) if trajectory == "CL" else ("E",))
            memory_update_positions = sorted(position_by_event[item] for item in state["memory_updates"])
            rows = []
            for category in EXPECTED_CATEGORIES:
                for event in registration["categories"][category]["events"]:
                    position, event_id = event["position"], event["event"]
                    if position not in state["processed_positions"]:
                        continue
                    score_g = ((position * 37 + (0 if trajectory == "S" else 19 if trajectory == "CL" else 41)) % 997) / 997
                    score_gc = ((position * 53 + (0 if trajectory == "S" else 29 if trajectory == "CL" else 71)) % 991) / 991
                    theta_by_controller = {"L": 0.5, "E": 0.55, "FrozenTheta": 0.5}
                    active_controllers = controller_names if trajectory != "S" else ("L", "E")
                    scores = {"G": score_g, "GC": score_gc}
                    thresholds = {f"{name}_{scorer}": theta_by_controller[name]
                        for name in (("L", "E", "FrozenTheta") if trajectory == "S" else active_controllers)
                        for scorer in ("G", "GC")}
                    decisions = {f"{name}_{scorer}": int(scores[scorer] >= theta_by_controller[name])
                        for name in (("L", "E", "FrozenTheta") if trajectory == "S" else active_controllers)
                        for scorer in ("G", "GC")}
                    controller_audit = {}
                    normal_updates = bisect_right(memory_update_positions, position)
                    for name in controller_names:
                        theta = theta_by_controller[name]
                        controller_audit[name] = {"revision": normal_updates,
                            "threshold_tokens": {"G": None, "GC": None},
                            "thresholds": {"G": theta, "GC": theta}, "suppression": {},
                            "effective_sample_size": 0.0, "normal_evidence_groups": normal_updates,
                            "cache_bytes": 0, "cache_groups": 0, "rescore_invocations": 0,
                            "rescore_seconds_total": 0.0,
                            "status": {"G": "COLD_START_UNCALIBRATED", "GC": "COLD_START_UNCALIBRATED"}}
                    query = bool(state["queries"].get(event_id, False))
                    rows.append({"event": event_id, "group": event["group"], "position": position,
                        "view": event["view"], "category": category, "scores": scores,
                        "thresholds": thresholds, "decisions": decisions,
                        "query": {"executed": query, "candidate": (1 if trajectory == "S" else 2 if trajectory == "CL" else 3),
                            "ordinal": event["view"], "design_probability": 1 / policy_period[trajectory] if query else None,
                            "cost": 0.8, "utility": 0.1},
                        "transition": {"audit": {"controllers": controller_audit,
                            "normal_version": normal_updates, "correction_version": 0,
                            "greedy_dynamic_prototypes": normal_updates, "correction_prototypes": 0}}})
            stream_sha = {category: digest([{"group": row["group"], "event": row["event"]}
                for row in registration["categories"][category]["events"]]) for category in EXPECTED_CATEGORIES}
            fixed_support = {category: spec["source_model_support"]["support_records_by_category"][category]["fixed_bank_sha256"]
                for category in EXPECTED_CATEGORIES}
            item = {"schema": "synthetic-stage-a-projection", "run_id": manifests[trajectory]["run_id"],
                "manifest_sha256": digest(manifests[trajectory]), "source_commit": source_commit,
                "campaign_id": binding["campaign_id"], "trajectory_id": trajectory,
                "campaign_registration_sha256": registration_hash, "scientific_spec_sha256": spec_identity(),
                "support_preflight_receipt_sha256": binding["support_preflight_receipt_sha256"],
                "r7_seed0_order_sha256": binding["r7_seed0_order_sha256"],
                "source_mapping_sha256": binding["source_mapping_sha256"],
                "model_state_sha256": model["model_state_sha256"],
                "checkpoint_sha256": model["checkpoint_sha256"],
                "hub_source_tree_sha256": model["hub_source_tree_sha256"],
                "scorer_config_sha256": model["native_config_sha256"],
                "preflight_receipt_sha256": spec["source_model_support"]["historical_receipt_sha256"],
                "fixed_support_sha256_by_category": fixed_support,
                "stream_sha256_by_category": stream_sha, "stage_a_sha256": state["last_hash"],
                "category_seals": dict(state["category_seals"]), "events": rows,
                "queried_feedback": dict(state["feedback"])}
            if write_projection:
                projection_path = Path(campaign_root) / "execution" / "sealed-projections" / f"{trajectory}.json"
                projection_path.parent.mkdir(parents=True, exist_ok=True)
                if projection_path.exists():
                    assert json.loads(projection_path.read_text()) == item
                else:
                    projection_path.write_bytes(campaign_module.canonical(item) + b"\n")
            return item

        monkeypatch.setattr(campaign_module, "_campaign_with_support", lambda _root: (binding, registration))
        monkeypatch.setattr(campaign_module, "_load_trajectory_manifest", lambda _root, trajectory: manifests[trajectory])
        monkeypatch.setattr(campaign_module, "_inspect_ledger", inspect_ledger)
        monkeypatch.setattr(campaign_module, "_start_worker", start_worker)
        monkeypatch.setattr(campaign_module, "_checkpoint_candidates", checkpoint_candidates)
        monkeypatch.setattr(campaign_module, "_checkpoint_identity",
            lambda run_root, arm, category, name: {"trajectory_id": arm, "category": category,
                "checkpoint_name": name, "committed_sha256": _sha(f"{arm}:{category}:{name}"),
                "state_sha256": _sha(f"state:{arm}:{category}:{name}"), "tensor_sha256": _sha("synthetic-tensors")})
        monkeypatch.setattr(campaign_module, "_verify_trajectory_stage_a", projection)
        if injected == "deadline":
            class ExpireAtStartClock:
                def __init__(self): self.calls = 0
                def monotonic_ns(self):
                    self.calls += 1
                    return 20_000_000_000 + (CAMPAIGN_WALL_LIMIT_SECONDS * 1_000_000_000 if self.calls >= 2 else 0)
                def time_ns(self):
                    return 1_900_000_000_000_000_000 + (
                        CAMPAIGN_WALL_LIMIT_SECONDS * 1_000_000_000 if self.calls >= 2 else 0)
            expired_clock = ExpireAtStartClock()
            monkeypatch.setattr(campaign_module, "CampaignTimingLedger",
                lambda path, **kwargs: CampaignTimingLedger(path, clock=expired_clock, **kwargs))
        try:
            result = run_stage_a(campaign_root=scenario_root, authorization_path=auth_path)
            assert injected is None and result["status"] == "STAGE_A_COMPLETE"
        except CampaignDeadlineExceeded:
            assert injected == "deadline"
            result = None
        except RecoverableExecutionInterruption:
            assert injected == "recoverable" and injection_fired["value"]
            result = None
        except ResearchWorkerTimeout:
            assert injected == "ambiguous" and injection_fired["value"]
            result = None
        return {"root": scenario_root, "states": arm_states, "labels": labels,
            "label_reads": label_reads, "feedback_commits": feedback_commits,
            "duplicate_feedback_writes": duplicate_feedback_writes, "disclosed_roots": disclosed_roots,
            "injection_fired": injection_fired, "projection": projection,
            "telemetry": {arm: [{"category": cat, "host_peak_working_set_bytes": 1024,
                "cuda_peak_allocated_bytes": None, "rescore_invocations": 0, "cache_bytes": 0}
                for cat in EXPECTED_CATEGORIES] for arm in manifests}}

    normal = run_scenario(tmp_path / "normal", injected=None)
    assert sum(len(state["category_seals"]) for state in normal["states"].values()) == 12
    assert len({str(path) for path in normal["disclosed_roots"].values()}) == 3
    for arm, state in normal["states"].items():
        assert set(state["category_seals"]) == set(EXPECTED_CATEGORIES)
        assert len(state["processed_positions"]) == sum(EXPECTED_EVENTS.values())
        assert all(count == 1 for count in normal["feedback_commits"][arm].values())
        assert normal["duplicate_feedback_writes"][arm] == 0
        assert set(normal["label_reads"][arm]) == set(state["feedback"])
    cl_only_event = registration["categories"]["audiojack"]["events"][13 * 5]["event"]
    assert cl_only_event in normal["states"]["CL"]["feedback"]
    assert cl_only_event not in normal["states"]["CE"]["feedback"]
    assert normal["states"]["S"]["memory_updates"] == [event for event in normal["states"]["S"]["feedback"]
        if normal["states"]["S"]["feedback"][event] == 0]

    # A missing arm/category seal is rejected by the same admission validator used by Stage B.
    complete_projections = {arm: normal["projection"](normal["root"], arm, write_projection=False)
        for arm in manifests}
    with pytest.raises(CampaignContractError, match="all three Stage A trajectories"):
        _validate_complete_stage_a(normal["root"], binding, registration,
            {key: value for key, value in complete_projections.items() if key != "CE"})

    recovery_root = tmp_path / "recovery"
    recovery = run_scenario(recovery_root, injected="recoverable")
    marker_path = recovery_root / "execution" / "INCOMPLETE.json"
    marker_before = marker_path.read_bytes()
    assert json.loads(marker_before)["classification"] == "RECOVERABLE_EXECUTION_INTERRUPTION"
    assert not (recovery_root / "execution" / "STAGE_A_CAMPAIGN_SEAL.json").exists()
    with pytest.raises(CampaignContractError):
        campaign_module.verify_stage_a(campaign_root=recovery_root)
    # The recovery scenario's synthetic workers and durable state remain bound while the
    # real resume loop reloads its category checkpoint and replays the ledger prefix.
    seal = campaign_module.run_stage_a(campaign_root=recovery_root,
        authorization_path=recovery_root / "stage-a-auth.json", resume=True)
    assert seal["status"] == "STAGE_A_COMPLETE"
    assert marker_path.read_bytes() == marker_before
    receipt_path = recovery_root / "execution" / "RECOVERY_COMPLETION.json"
    receipt_original = receipt_path.read_bytes()
    receipt_path.unlink()
    with pytest.raises(CampaignContractError, match="no recovery-completion receipt"):
        campaign_module.verify_stage_a(campaign_root=recovery_root)
    receipt_path.write_bytes(receipt_original)
    receipt = json.loads(receipt_original)
    receipt["source_freeze_commit"] = "f" * 40
    receipt_path.write_bytes(campaign_module.canonical(receipt) + b"\n")
    with pytest.raises(CampaignContractError, match="recovery-completion receipt"):
        campaign_module.verify_stage_a(campaign_root=recovery_root)
    receipt_path.write_bytes(receipt_original)
    verified = campaign_module.verify_stage_a(campaign_root=recovery_root)
    assert verified["terminal_state"] == "VERIFIED_RECOVERY_COMPLETION"
    for arm in manifests:
        assert all(count == 1 for count in recovery["feedback_commits"][arm].values())
        assert recovery["duplicate_feedback_writes"][arm] == 0
    stage = {arm: recovery["projection"](recovery_root, arm, write_projection=False) for arm in manifests}
    report = evaluate_campaign(stage, recovery["labels"], recovery["telemetry"])
    assert report["population"]["total_events_per_trajectory"] == 15295
    assert set(report["arms"]) == {"S", "CL", "CE"}

    ambiguous = run_scenario(tmp_path / "ambiguous", injected="ambiguous")
    ambiguous_marker = json.loads((ambiguous["root"] / "execution" / "INCOMPLETE.json").read_bytes())
    assert ambiguous_marker["classification"] == "AMBIGUOUS_LABEL_DISCLOSURE"
    with pytest.raises(CampaignContractError, match="fatal, ambiguous or deadline-exhausted"):
        campaign_module.run_stage_a(campaign_root=ambiguous["root"],
            authorization_path=ambiguous["root"] / "stage-a-auth.json", resume=True)

    deadline = run_scenario(tmp_path / "deadline", injected="deadline")
    deadline_marker = json.loads((deadline["root"] / "execution" / "INCOMPLETE.json").read_bytes())
    assert deadline_marker["classification"] == "DEADLINE_EXHAUSTION"
    assert deadline_marker["reason_code"] == "CAMPAIGN_WALL_LIMIT_EXCEEDED"
    assert deadline["states"] == {}
    assert not (deadline["root"] / "execution" / "STAGE_A_CAMPAIGN_SEAL.json").exists()

    # Timing and source-fingerprint invariants are included in the synthetic integration gate.
    assert spec_identity() == "1c96077bfdaa58f9ff93a67a9349a4cc6bc8b63578ae8915ff23762b50aee896"


def test_all_arm_evaluator_reports_paired_estimands_and_equal_shared_rankings():
    stage, labels, telemetry = _synthetic_campaign()
    report = evaluate_campaign(stage, labels, telemetry)
    assert report["estimands"]["H1_shared_threshold_only"]["four_category_macro_difference"]["value"] == 0.0
    assert report["estimands"]["H2_coupled_system"]["four_category_macro_difference"]["value"] == 0.0
    assert report["arms"]["S"]["four_category_macro"]["L_GC"]["auroc"]["value"] == 1.0
    assert report["arms"]["S"]["four_category_macro"]["E_GC"]["average_precision"]["value"] == 1.0
    assert report["population"]["total_events_per_trajectory"] == 15295
