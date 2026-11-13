from __future__ import annotations

from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from dataclasses import asdict

import pytest

from experiments.calibration.bea_temporal_audit.aggregate import aggregate_campaign
from experiments.calibration.bea_temporal_audit.core import ARMS, BEA, AuditError, analyze_trajectory, validate_rows
from experiments.calibration.bea_temporal_audit.integrity import digest, load_verified
from experiments.calibration.bea_temporal_audit.output import validate_destination, write_outputs
from experiments.calibration.bea_temporal_audit import projection
from experiments.calibration.bea_temporal_audit.projection import project_after_all_seals, validate_audit_authorization, verify_metric_parity
from experiments.calibration.bea_temporal_audit.campaign import CATEGORIES, IDENTITY, NEW, OLD, RELEASE, VerifiedCampaign
from experiments.calibration.bea_temporal_audit.reference import validate_stage_b_summary
from experiments.calibration.bea_temporal_audit.cli import main as cli_main
from experiments.calibration.final_benchmark.entrypoint import FinalBenchmarkEntrypoint, EntryPointError
from experiments.calibration.final_benchmark.evaluation import LabelProjectionAuthorization, evaluate_sealed_category
from experiments.calibration.final_benchmark.execution_support import AttemptStatus
from experiments.calibration.final_benchmark.sink import FsyncGroupSink


@pytest.fixture
def audit_tmp_path():
    """Create and clean only this test's new directory inside the authorized tree."""
    parent = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="_bea_t1_r1_", dir=parent) as name:
        target = Path(name).resolve()
        assert target.parent == parent
        yield target


def sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def make_rows(groups=20, cap=3):
    counts = {a: 0 for a in ARMS}
    prices = {a: .8 for a in BEA}
    flags = {a: deque(maxlen=256) for a in BEA}
    rows = []
    for g in range(groups):
        used = {a: False for a in ARMS}
        actions = {a: [] for a in ARMS}
        for j in range(5):
            i = 5*g+j
            score = .2 if i % 2 == 0 else .9
            row = {"position": i, "event_id_sha256": sha(f"event-{i}"),
                   "image_identity_sha256": sha(f"image-{i}"), "source_group_sha256": sha(f"group-{g}"),
                   "camera_id": j+1, "arm_scores": {a: score for a in ARMS},
                   "arm_predictions": {a: int(score >= .5) for a in ARMS},
                   "arm_thresholds": {a: .5 for a in ARMS}, "query_actions": {},
                   "n_shadow_score": score, "static_scores": {"STATIC-4+64": score, "STATIC-36+32": score},
                   "static_predictions": {"STATIC-4+64": int(score >= .5), "STATIC-36+32": int(score >= .5)},
                   "static_thresholds": {"STATIC-4+64": .5, "STATIC-36+32": .5}}
            for arm in ARMS:
                proposed = True
                executed = proposed and not used[arm] and counts[arm] < cap
                group_suppressed = proposed and used[arm]
                cap_suppressed = proposed and not used[arm] and counts[arm] >= cap
                if executed:
                    used[arm] = True
                    counts[arm] += 1
                row["query_actions"][arm] = {"proposed": proposed, "executed": executed,
                    "group_suppressed": group_suppressed, "cap_suppressed": cap_suppressed,
                    "proposal_objective": 1.5 if arm in BEA else .7,
                    "proposal_price": prices[arm] if arm in BEA else None,
                    "query_receipt_sha256": sha(f"receipt-{i}-{arm}") if executed else None}
                actions[arm].append(executed)
            rows.append(row)
        for arm in BEA:
            for flag in actions[arm]:
                flags[arm].append(int(flag))
                prices[arm] = min(3, max(.05, prices[arm]+.02*(sum(flags[arm])/len(flags[arm])-.03)))
    return rows


def sealed_fixture(tmp_path, rows, stream_override=None, annotation_sha=None):
    stream = {"events": [{k: row[k] for k in ("position", "event_id_sha256", "image_identity_sha256", "source_group_sha256", "camera_id")} for row in rows]}
    if stream_override is not None: stream = stream_override
    identity = {"source_commit": "a3688e87ac165866fa4c4ddb3501f96258500534", "category": "widget", "condition": "P0",
                "identity_manifest_sha256": sha("identity"), "role_manifest_sha256": sha("role"),
                "dataset_release_sha256": sha("release"), "stream_manifest_sha256": digest(stream),
                "annotation_sha256": annotation_sha or sha("annotation"), "campaign_run_id": "synthetic"}
    root = tmp_path / "attempt_01"
    attempt = AttemptStatus(root, identity=identity, expected_events=len(rows), execution_kind="principal_stage_a")
    header = {"source_commit": identity["source_commit"], "identity_manifest_sha256": identity["identity_manifest_sha256"],
              "role_manifest_sha256": identity["role_manifest_sha256"], "dataset_release_sha256": identity["dataset_release_sha256"],
              "dataset_id": "realiad", "category": "widget", "condition_id": "P0"}
    sink = FsyncGroupSink(root / "stage_a_journal.jsonl", identity=header)
    sealed_groups = []
    for start in range(0, len(rows), 5):
        local = rows[start:start+5]
        group = {"group_index": start//5, "source_group_sha256": local[0]["source_group_sha256"],
                 "start_position": start, "end_position_exclusive": start+5,
                 "event_order_sha256": digest([r["event_id_sha256"] for r in local]),
                 "score_rows_sha256": digest(local), "status": "SEALED_BEFORE_LABEL_DISCLOSURE"}
        sealed_groups.append({**group, "durable_group_receipt_sha256": sink.append_group(group_seal=group, score_rows=local)})
    payload = {"dataset_id": "realiad", "category": "widget", "run_condition": "P0", "expected_event_count": len(rows),
               "completed_event_count": len(rows), "group_count": len(sealed_groups), "score_rows_sha256": digest(rows),
               "group_seals_sha256": digest(sealed_groups), "status": "SEALED_COMPLETE"}
    seal_sha = sink.seal_category(payload); sink.close()
    report = {"schema": "final-benchmark.stage-a-category-report.v1", **payload,
              "durable_seal_sha256": seal_sha, "source_commit": identity["source_commit"],
              "identity_manifest_sha256": identity["identity_manifest_sha256"],
              "role_manifest_sha256": identity["role_manifest_sha256"],
              "dataset_release_sha256": identity["dataset_release_sha256"]}
    (root / "stage_a_category_report.json").write_text(json.dumps(report), encoding="utf-8")
    attempt.sealed_complete(phase="CATEGORY_SEALED", seal_path=root / "stage_a_category_seal.json", completed_events=len(rows))
    record = {"category":"widget", "condition":"P0", "campaign_run_id":"synthetic", "source_commit":identity["source_commit"],
              "stage_a_category_seal_sha256":seal_sha, "metrics":{},
              "query_count":{arm: sum(int(row["query_actions"][arm]["executed"]) for row in rows) for arm in ARMS}}
    return root, identity, stream, record


def test_group_replay_cap_and_read_only_fixed_price():
    rows = make_rows(); old = copy.deepcopy(rows)
    replay = validate_rows(rows, cap=3)
    assert replay["executed"] == {a: 3 for a in ARMS}
    assert replay["first_cap_exhaustion_position"]["F-B"] == 10
    assert all(r["query_actions"]["F-B"]["proposed"] and not r["query_actions"]["F-B"]["executed"] for r in rows[11:])
    result = analyze_trajectory(rows, cap=3)
    assert result["segments"]["late"]["arms"]["F-B"]["fixed_price_proposals"] == 30
    assert result["segments"]["late"]["arms"]["F-B"]["groups_with_cap_suppression"] == 6
    assert result["component_availability"]["B"] == "NOT_IDENTIFIABLE"
    assert rows == old
    broken = copy.deepcopy(rows); broken[1]["query_actions"]["F-B"]["proposal_price"] += .01
    with pytest.raises(AuditError, match="pre-group price"):
        validate_rows(broken, cap=3)
    broken = copy.deepcopy(rows); broken[2]["camera_id"] = 1
    with pytest.raises(AuditError, match="camera"):
        validate_rows(broken, cap=3)
    zero = make_rows(groups=4, cap=0)
    zero_result = analyze_trajectory(zero, cap=0)
    assert zero_result["accounting"]["executed"]["F-B"] == 0
    assert zero_result["segments"]["early"]["arms"]["F-B"]["pre_cap_price"]["n"] == 0
    assert zero_result["segments"]["early"]["arms"]["F-B"]["post_cap_price"]["n"] > 0
    broken = copy.deepcopy(rows); broken[10]["static_thresholds"]["STATIC-4+64"] = .51
    with pytest.raises(AuditError, match="static threshold"):
        validate_rows(broken, cap=3)


def test_missing_classes_conditional_rates_and_known_metrics():
    rows = make_rows()
    result = analyze_trajectory(rows, cap=3, labels=[0]*100)
    m = result["segments"]["early"]["arms"]["F-B"]["metrics"]
    assert m["normal_count"] == 30 and m["defect_count"] == 0 and m["fpr"] == .5
    assert m["fnr"] is None and m["ber"] is None and m["auroc"] is None
    result = analyze_trajectory(rows, cap=3, labels=[i % 2 for i in range(100)])
    m = result["segments"]["late"]["arms"]["F-B"]["metrics"]
    assert (m["fpr"], m["fnr"], m["ber"], m["auroc"], m["ap"]) == (0, 0, 0, 1, 1)
    assert result["segments"]["late"]["n_shadow"]["fpr"] is None
    reverse = analyze_trajectory(rows, cap=3, labels=[1 - i % 2 for i in range(100)])
    hidden = reverse["segments"]["late"]["arms"]["F-B"]["cap_suppressed_outcomes"]
    assert (hidden["count"], hidden["defect_count"], hidden["defect_prevalence"], hidden["hidden_false_negatives"]) == (30, 15, .5, 15)
    with pytest.raises(AuditError):
        analyze_trajectory(rows, cap=3, labels=[0]*99)


def test_seal_and_identity_mismatches_are_rejected(audit_tmp_path):
    rows = make_rows(); root, identity, stream, record = sealed_fixture(audit_tmp_path, rows)
    ids = [r["event_id_sha256"] for r in rows]
    t = load_verified(status_path=root / "attempt_status.json", expected_identity=identity,
                      registered_stream_manifest=stream, expected_event_count=100, hard_cap=3, stage_b_record=record)
    assert len(t.groups) == 20
    with pytest.raises(AuditError):
        load_verified(status_path=root / "attempt_status.json", expected_identity={**identity, "category": "different"},
                      registered_stream_manifest=stream, expected_event_count=100, hard_cap=3, stage_b_record=record)
    with pytest.raises(AuditError):
        changed = copy.deepcopy(stream); changed["events"].reverse()
        load_verified(status_path=root / "attempt_status.json", expected_identity=identity,
                      registered_stream_manifest=changed, expected_event_count=100, hard_cap=3, stage_b_record=record)
    journal = root / "stage_a_journal.jsonl"
    with journal.open("ab") as f: f.write(b"{}\n")
    with pytest.raises(AuditError):
        load_verified(status_path=root / "attempt_status.json", expected_identity=identity,
                      registered_stream_manifest=stream, expected_event_count=100, hard_cap=3, stage_b_record=record)


def test_existing_stage_b_entrypoint_seal_mismatch_fails_before_loader(audit_tmp_path):
    rows = make_rows(); root, identity, stream, record = sealed_fixture(audit_tmp_path, rows)
    t = load_verified(status_path=root / "attempt_status.json", expected_identity=identity,
                      registered_stream_manifest=stream, expected_event_count=100, hard_cap=3, stage_b_record=record)
    auth = LabelProjectionAuthorization("AUTHORIZED_POST_STAGE_A_LABEL_PROJECTION", "stage_b_category_evaluation", sha("source"), "realiad", "widget", t.seal.durable_seal_sha256)
    called = []
    with pytest.raises(EntryPointError, match="receipt seal differs"):
        FinalBenchmarkEntrypoint.evaluate_stage_b(dataset_id="realiad", category="widget", rows=t.rows,
            seal=t.seal, unit_seals=t.groups, authorization=auth, attempt_status_path=t.status_path,
            expected_attempt_identity=identity, projection_loader=lambda: called.append(True))
    assert called == []


def test_cross_artifact_integrity_rejects_tampering(audit_tmp_path):
    rows = make_rows()
    for kind in ("seal", "report", "attempt", "record_source", "record_seal"):
        root, identity, stream, record = sealed_fixture(audit_tmp_path / kind, rows)
        path = root / "attempt_status.json"
        if kind == "seal":
            with (root / "stage_a_category_seal.json").open("ab") as f: f.write(b" ")
        elif kind == "report":
            report_path = root / "stage_a_category_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["durable_seal_sha256"] = sha("wrong")
            report_path.write_text(json.dumps(report), encoding="utf-8")
        elif kind == "attempt":
            status = json.loads(path.read_text(encoding="utf-8"))
            status["completion_seal_sha256"] = sha("wrong")
            path.write_text(json.dumps(status), encoding="utf-8")
        elif kind == "record_source":
            record = {**record, "source_commit": OLD}
        else:
            record = {**record, "stage_a_category_seal_sha256": sha("wrong")}
        with pytest.raises(AuditError):
            load_verified(status_path=path, expected_identity=identity, registered_stream_manifest=stream,
                          expected_event_count=100, hard_cap=3, stage_b_record=record)


def authorized_synthetic(audit_tmp_path):
    rows = make_rows(); root, identity, stream, record = sealed_fixture(audit_tmp_path, rows)
    trajectory = load_verified(status_path=root / "attempt_status.json", expected_identity=identity,
        registered_stream_manifest=stream, expected_event_count=100, hard_cap=3, stage_b_record=record)
    auth = LabelProjectionAuthorization("AUTHORIZED_POST_STAGE_A_LABEL_PROJECTION", "stage_b_category_evaluation",
        sha("source"), "realiad", "widget", trajectory.seal.durable_seal_sha256)
    labels = {row["event_id_sha256"]: i % 2 for i, row in enumerate(rows)}
    result = evaluate_sealed_category(dataset_id="realiad", category="widget", rows=trajectory.rows,
        seal=trajectory.seal, group_seals=trajectory.groups, labels_by_event_id=labels, label_projection=auth)
    record["metrics"] = result["arm_metrics"]
    record["queried_label_composition"] = result["queried_label_composition"]
    record["first_queried_defect_position"] = result["first_queried_defect_position"]
    trajectory = load_verified(status_path=root / "attempt_status.json", expected_identity=identity,
        registered_stream_manifest=stream, expected_event_count=100, hard_cap=3, stage_b_record=record)
    payload = {"schema":"realiad.stage-b-event-label-projection.v1", "dataset_id":"realiad", "category":"widget",
        "condition":"P0", "campaign_run_id":"synthetic", "source_commit":identity["source_commit"],
        "stage_a_category_seal_sha256":trajectory.seal.durable_seal_sha256,
        "annotation_sha256":identity["annotation_sha256"], "source_sha256":auth.source_sha256, "labels_by_event_id":labels}
    projection_path = audit_tmp_path / "synthetic_projection.json"
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    projection_path.write_bytes(raw)
    audit = {"schema":"bea-t1.audit-authorization.v1", "status":"RESEARCHER_AUTHORIZED_POST_SEAL_AUDIT",
             "independent_review_status":"PASS", "trajectory_count":90, "dataset_release_sha256":RELEASE,
             "identity_manifest_sha256":IDENTITY, "mask_access":False, "output_scope":"AGGREGATE_ONLY"}
    grant = {"schema":"bea-t1.label-projection-authorization.v1", "status":"RESEARCHER_AUTHORIZED_BEA_T1_LABEL_PROJECTION",
        "scope":"realiad_post_seal_event_labels", "projection_provenance":"AUTHORIZED_STAGE_B_IDENTITY_PROJECTION",
        "audit_authorization_sha256":digest(audit), "stage_b_authorization_sha256":digest(asdict(auth)),
        "dataset_id":"realiad", "category":"widget", "condition":"P0", "campaign_run_id":"synthetic",
        "source_commit":identity["source_commit"], "stage_a_category_seal_sha256":trajectory.seal.durable_seal_sha256,
        "annotation_sha256":identity["annotation_sha256"], "source_sha256":auth.source_sha256,
        "projection_file_sha256":hashlib.sha256(raw).hexdigest(), "mask_access":False}
    return trajectory, auth, audit, grant, projection_path, payload


def test_audit_only_authorized_synthetic_end_to_end(audit_tmp_path):
    t, auth, audit, grant, path, payload = authorized_synthetic(audit_tmp_path)
    assert "durable_seal_sha256" not in json.loads((t.status_path.parent / "stage_a_category_seal.json").read_text(encoding="utf-8"))
    labels = projection._evaluate_authorized_trajectory(trajectory=t, stage_b_authorization=auth,
        projection_grant=grant, audit_authorization=audit, projection_path=path)
    result = analyze_trajectory(t.rows, cap=3, labels=labels)
    assert result["segments"]["late"]["arms"]["F-B"]["metrics"]["ber"] == 0
    grouped = aggregate_campaign({("widget", p): result for p in ("P0", "P1", "P2")}, expected_categories=1)
    assert any(r["segment"] == "late" and r["arm"] == "F-B" and r["metric"] == "ber" and r["mean"] == 0 for r in grouped)
    manifest = {"schema":"bea-t1.audit-manifest.v1", "status":"COMPLETE", "source_commits":["a"*40],
        "input_artifact_sha256":{"official_metrics":sha("ref")}, "authorization_sha256":sha("grant"),
        "bootstrap_seed":20261114, "bootstrap_resamples":10000, "claim_scope":"EXPLORATORY_POST_HOC"}
    integrity = {"status":"PASS", "trajectory_count":3, "category_count":1, "event_count":300,
                 "metric_parity":"PASS_ALL_TRAJECTORIES"}
    dest = audit_tmp_path / "aggregate_output"
    write_outputs(dest, aggregate=grouped, integrity=integrity, manifest=manifest,
                  repository=Path(__file__).resolve().parents[3], synthetic_test_only=True)
    assert (dest / "audit_summary.json").is_file()


@pytest.mark.parametrize("field", ["queried_label_composition", "first_queried_defect_position"])
def test_feedback_parity_rejects_changed_registered_outcomes(audit_tmp_path, field):
    from dataclasses import replace
    t, auth, audit, grant, path, _ = authorized_synthetic(audit_tmp_path)
    changed = copy.deepcopy(t.stage_b_record)
    if field == "queried_label_composition":
        changed[field]["F-B"]["defect"] += 1
    else:
        changed[field]["F-B"] = None
    with pytest.raises(AuditError, match=field):
        projection._evaluate_authorized_trajectory(trajectory=replace(t, stage_b_record=changed),
            stage_b_authorization=auth, projection_grant=grant, audit_authorization=audit, projection_path=path)


def test_feedback_parity_null_and_types_are_exact():
    record = {"queried_label_composition": {"F-B": {"normal": 1, "defect": 0}},
              "first_queried_defect_position": {"F-B": None}}
    projection.verify_feedback_parity(record, copy.deepcopy(record))
    bad = copy.deepcopy(record); bad["first_queried_defect_position"]["F-B"] = 0
    with pytest.raises(AuditError, match="first_queried_defect_position"):
        projection.verify_feedback_parity(record, bad)
    bad = copy.deepcopy(record); bad["queried_label_composition"]["F-B"]["normal"] = True
    with pytest.raises(AuditError, match="queried_label_composition"):
        projection.verify_feedback_parity(record, bad)


def test_projection_rejected_before_label_open_and_missing_labels(audit_tmp_path, monkeypatch):
    t, auth, audit, grant, path, payload = authorized_synthetic(audit_tmp_path)
    calls = []
    original = projection._load_authorized_projection
    def monitored(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(projection, "_load_authorized_projection", monitored)
    with pytest.raises(AuditError, match="authorization"):
        projection._evaluate_authorized_trajectory(trajectory=t, stage_b_authorization=None,
            projection_grant=grant, audit_authorization=audit, projection_path=path)
    with pytest.raises(AuditError, match="authorization"):
        projection._evaluate_authorized_trajectory(trajectory=t, stage_b_authorization=auth,
            projection_grant={**grant, "stage_a_category_seal_sha256":sha("wrong")}, audit_authorization=audit, projection_path=path)
    assert calls == []
    report_path = t.status_path.parent / "stage_a_category_report.json"
    original_report = report_path.read_bytes()
    report = json.loads(original_report); report["source_commit"] = OLD
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(AuditError):
        projection._evaluate_authorized_trajectory(trajectory=t, stage_b_authorization=auth,
            projection_grant=grant, audit_authorization=audit, projection_path=path)
    assert calls == []
    report_path.write_bytes(original_report)
    payload["labels_by_event_id"].pop(next(iter(payload["labels_by_event_id"])))
    raw = json.dumps(payload, sort_keys=True).encode("utf-8"); path.write_bytes(raw)
    grant["projection_file_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(AuditError, match="missing, extra"):
        projection._evaluate_authorized_trajectory(trajectory=t, stage_b_authorization=auth,
            projection_grant=grant, audit_authorization=audit, projection_path=path)
    assert calls == [True]


def test_real_nested_stage_b_summary_coverage_and_sources():
    records = []
    for index, (category, condition) in enumerate((c, p) for c in CATEGORIES for p in ("P0", "P1", "P2")):
        source = OLD if index < 36 else NEW
        records.append({"category":category, "condition":condition,
            "campaign_run_id":"realiad-final-20261111-r1-40d61c3" if source == OLD else "synthetic-continuation",
            "source_commit":source, "stage_a_category_seal_sha256":sha(index),
            "metrics":{arm:{} for arm in (*ARMS, "N-shadow", "STATIC-4+64")}, "query_count":{arm:0 for arm in ARMS},
            "queried_label_composition":{arm:{"normal":0,"defect":0} for arm in ARMS},
            "first_queried_defect_position":{arm:None for arm in ARMS}})
    summary = {"schema":"realiad.stage-b-final-metrics.v1", "status":"STAGE_B_ALL_90_CPU_EVALUATED", "scientific_results":records}
    assert len(validate_stage_b_summary(summary)) == 90
    bad = copy.deepcopy(summary); bad["scientific_results"][36]["source_commit"] = OLD
    with pytest.raises(AuditError): validate_stage_b_summary(bad)
    bad = copy.deepcopy(summary); bad["scientific_results"][-1] = bad["scientific_results"][0]
    with pytest.raises(AuditError): validate_stage_b_summary(bad)
    bad = copy.deepcopy(summary); del bad["scientific_results"][0]["first_queried_defect_position"]
    with pytest.raises(AuditError): validate_stage_b_summary(bad)


def test_clustered_aggregation_and_missing_condition():
    r = analyze_trajectory(make_rows(), cap=3, labels=[i % 2 for i in range(100)])
    values = {(c, p): copy.deepcopy(r) for c in ("a", "b") for p in ("P0", "P1", "P2")}
    for p in ("P0", "P1", "P2"):
        values[("b", p)]["segments"]["late"]["arms"]["F-B"]["metrics"]["ber"] = .4
    out = aggregate_campaign(values, expected_categories=2)
    entry = next(x for x in out if x["segment"] == "late" and x["arm"] == "F-B" and x["metric"] == "ber")
    assert entry["mean"] == pytest.approx(.2) and entry["ci95_low"] == 0 and entry["ci95_high"] == pytest.approx(.4)
    del values[("a", "P1")]
    with pytest.raises(AuditError): aggregate_campaign(values, expected_categories=2)


def test_authorization_and_metric_parity_fail_closed():
    seals = [sha(i) for i in range(90)]; plans = [sha("a"), sha("b")]
    with pytest.raises(AuditError): validate_audit_authorization({}, audit_source_commit="a"*40, seal_hashes=seals, reference_sha256=sha("ref"), source_plan_sha256=plans, configuration_sha256=sha("config"), stage_b_authorization_sha256=sha("stageb"))
    grant = {"schema":"bea-t1.audit-authorization.v1", "status":"RESEARCHER_AUTHORIZED_POST_SEAL_AUDIT",
             "independent_review_status":"PASS", "audit_source_commit":"a"*40, "seal_set_sha256":digest(sorted(seals)),
             "source_plan_sha256":sorted(plans), "dataset_release_sha256":RELEASE, "identity_manifest_sha256":IDENTITY,
             "official_metrics_sha256":sha("ref"), "configuration_sha256":sha("config"), "stage_b_authorization_sha256":sha("stageb"), "historical_source_commits":sorted([OLD, NEW]), "outcome_access_scope":"POST_SEAL_IMAGE_LEVEL_OUTCOMES", "trajectory_count":90, "mask_access":False, "output_scope":"AGGREGATE_ONLY"}
    validate_audit_authorization(grant, audit_source_commit="a"*40, seal_hashes=seals, reference_sha256=sha("ref"), source_plan_sha256=plans, configuration_sha256=sha("config"), stage_b_authorization_sha256=sha("stageb"))
    with pytest.raises(AuditError): validate_audit_authorization(grant, audit_source_commit="a"*40, seal_hashes=seals, reference_sha256=sha("other"), source_plan_sha256=plans, configuration_sha256=sha("config"), stage_b_authorization_sha256=sha("stageb"))
    verify_metric_parity({"F-B": {"ber": .2, "n": 10}}, {"F-B": {"ber": .2, "n": 10}})
    with pytest.raises(AuditError): verify_metric_parity({"F-B": {"ber": .2}}, {"F-B": {"ber": .3}})
    with pytest.raises(AuditError): verify_metric_parity({"F-B": {"worst_256": {"worst_ber": .3}}}, {"F-B": {"worst_256": {"worst_ber": .2}}})


def test_missing_campaign_never_invokes_projection():
    with pytest.raises(AuditError): VerifiedCampaign([])
    with pytest.raises(AuditError, match="complete campaign"):
        project_after_all_seals(trajectory=None, campaign=None, audit_authorization={},
            audit_source_commit="a"*40, reference_sha256=sha("ref"), official_reference_path="unused",
            configuration_sha256=sha("config"), stage_b_authorization=None,
            projection_grant={}, projection_path="unused")


def test_cli_default_and_missing_configuration_are_blocked(capsys, audit_tmp_path):
    assert cli_main([]) == 2
    assert '"status": "BLOCKED"' in capsys.readouterr().out
    assert cli_main(["--run"]) == 2
    assert '"status": "BLOCKED"' in capsys.readouterr().out
    assert cli_main(["--run", "--config", str(audit_tmp_path / "missing.json")]) == 2
    assert '"status": "BLOCKED"' in capsys.readouterr().out
    config = audit_tmp_path / "config.json"
    config.write_text(json.dumps({"schema":"bea-t1.execution-config.v1", "audit_source_commit":"a"*40,
        "audit_authorization_path":str(audit_tmp_path / "missing-grant.json")}), encoding="utf-8")
    assert cli_main(["--run", "--config", str(config)]) == 2
    message = capsys.readouterr().out
    assert '"status": "BLOCKED"' in message and "audit configuration schema is invalid" in message


def test_real_output_requires_ignored_directory_before_creation(audit_tmp_path):
    fake_repo = audit_tmp_path / "repo"
    root = fake_repo / "outputs" / "calibration" / "bea_temporal_audit"
    root.mkdir(parents=True)
    target = root / "new_run"
    with pytest.raises(AuditError, match="not Git-ignored"):
        validate_destination(target, repository=fake_repo)
    assert not target.exists()


def test_output_privacy_and_exclusive_creation(audit_tmp_path):
    repo = Path(__file__).resolve().parents[3]
    row = {"segment":"early", "arm":"F-B", "metric":"ber", "category_count":2, "defined_category_count":2,
           "mean":.2, "ci95_low":.1, "ci95_high":.3, "status":"DEFINED"}
    manifest = {"schema":"bea-t1.audit-manifest.v1", "status":"COMPLETE", "source_commits":["a"*40],
        "input_artifact_sha256":{"official_metrics":sha("ref")}, "authorization_sha256":sha("grant"),
        "bootstrap_seed":20261114, "bootstrap_resamples":10000, "claim_scope":"EXPLORATORY_POST_HOC"}
    integrity = {"status":"PASS", "trajectory_count":90, "category_count":30, "event_count":343755,
                 "metric_parity":"PASS_ALL_TRAJECTORIES"}
    dest = audit_tmp_path / "synthetic_output"
    write_outputs(dest, aggregate=[row], integrity=integrity, manifest=manifest, repository=repo, synthetic_test_only=True)
    assert {p.name for p in dest.iterdir()} == {"integrity_checks.json", "audit_summary.json", "audit_manifest.json", "temporal_aggregate.csv", "decision_report.md"}
    with pytest.raises(FileExistsError): write_outputs(dest, aggregate=[row], integrity=integrity, manifest=manifest, repository=repo, synthetic_test_only=True)
    with pytest.raises(AuditError): write_outputs(audit_tmp_path / "rejected", aggregate=[{**row, "event_id_sha256": sha("x")}], integrity=integrity, manifest=manifest, repository=repo, synthetic_test_only=True)
    with pytest.raises(AuditError): write_outputs(audit_tmp_path / "nonfinite", aggregate=[{**row, "mean": float("nan")}], integrity=integrity, manifest=manifest, repository=repo, synthetic_test_only=True)
    external = Path(tempfile.gettempdir()) / "bea-t1-unauthorized-destination"
    with pytest.raises(AuditError, match="synthetic output"):
        validate_destination(external, repository=repo, synthetic_test_only=True)
