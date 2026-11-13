"""Synthetic-only tests for the existing-artifact adapters."""
import copy
import hashlib
import json
from types import SimpleNamespace
import pytest
from experiments.calibration.final_benchmark import identity
from experiments.calibration.bea_temporal_audit.artifacts import validate_embedded_stream
from experiments.calibration.bea_temporal_audit.campaign import RELEASE
from experiments.calibration.bea_temporal_audit.core import AuditError
from experiments.calibration.bea_temporal_audit.integrity import digest
from experiments.calibration.bea_temporal_audit.projector import _decode_annotation, project_annotation
from experiments.calibration.bea_temporal_audit.projection import verify_evaluation_parity


def fixture():
    rows, annotations = [], []
    for group_num, session in ((1, "OK"), (2, "NG")):
        group = f"widget_{group_num:04d}_{session}"
        for camera in range(1, 6):
            key = f"{group}_C{camera}_20260101010101.jpg"
            annotations.append({"image_path": key, "anomaly_class": "normal" if camera < 4 else "defect", "mask": "OPAQUE"})
            pos = len(rows)
            rows.append({"position":pos, "original_position":pos, "camera_id":camera, "within_group_ordinal":camera,
                "source_group_sha256":identity._source_group_digest(group),
                "image_identity_sha256":identity._image_identity_digest(key),
                "event_id_sha256":identity._domain_hash("Gate0-event-v1", RELEASE, "widget", "0", group, str(camera), key, str(camera))})
    # Swap complete groups to exercise original annotation positions.
    rows = rows[5:] + rows[:5]
    for i, row in enumerate(rows): row["position"] = i
    stream = {"schema":"final-benchmark.stream-manifest.v1", "dataset_id":"realiad", "dataset_release_sha256":RELEASE,
        "category":"widget", "condition_id":"P0", "support_seed":0, "stream_seed":0, "event_count":10, "events":rows}
    bundle = {"schema":"final-benchmark.category-condition-identity.v1", "dataset_id":"realiad", "category":"widget",
        "condition_id":"P0", "support_seed":0, "stream_seed":0, "stream_manifest":stream, "stream_manifest_sha256":digest(stream)}
    plan = {"category":"widget", "condition":"P0", "test_event_count":10, "source_group_count":2, "stream_manifest_sha256":digest(stream)}
    data = json.dumps({"meta":{"normal_class":"normal"}, "train":[], "test":annotations}).encode().replace(b"OPAQUE", b"\xff")
    trajectory = SimpleNamespace(category="widget", registered_stream_manifest=stream, rows=rows)
    return bundle, plan, data, trajectory


def test_embedded_stream_and_tampering():
    bundle, plan, _, _ = fixture()
    assert validate_embedded_stream(bundle=bundle, plan_row=plan)["event_count"] == 10
    for operation in ("order", "camera", "group", "seed", "digest"):
        b, p = copy.deepcopy(bundle), copy.deepcopy(plan)
        s = b["stream_manifest"]
        if operation == "order": s["events"][0], s["events"][1] = s["events"][1], s["events"][0]
        if operation == "camera": s["events"][0]["camera_id"] = 5
        if operation == "group": s["events"][0]["source_group_sha256"] = "a"*64
        if operation == "seed": s["stream_seed"] = 11
        if operation != "digest": b["stream_manifest_sha256"] = p["stream_manifest_sha256"] = digest(s)
        else: p["stream_manifest_sha256"] = "a"*64
        with pytest.raises(AuditError): validate_embedded_stream(bundle=b, plan_row=p)


def test_in_memory_projection_masks_opaque_and_identity_rejections(monkeypatch):
    from experiments.calibration import r7_shuffled_correction_4class as r7
    bundle, plan, data, t = fixture()
    decode = r7._decode_field
    def guarded(raw, bounds):
        assert b"\xff" not in raw[bounds[0]:bounds[1]]
        return decode(raw, bounds)
    monkeypatch.setattr(r7, "_decode_field", guarded)
    h = hashlib.sha256(data).hexdigest()
    labels = _decode_annotation(data, trajectory=t, annotation_sha256=h)
    assert len(labels) == 10 and sum(labels.values()) == 4
    assert labels[t.rows[0]["event_id_sha256"]] == 0
    with pytest.raises(AuditError): _decode_annotation(data, trajectory=t, annotation_sha256="0"*64)
    for field in ("original_position", "camera_id", "image_identity_sha256", "event_id_sha256"):
        bad = copy.deepcopy(t)
        bad.registered_stream_manifest["events"][0][field] = bad.registered_stream_manifest["events"][1][field]
        with pytest.raises(AuditError): _decode_annotation(data, trajectory=bad, annotation_sha256=h)


def test_projection_requires_complete_authority_and_no_output(tmp_path):
    target = tmp_path / "does-not-exist.json"
    with pytest.raises(AuditError, match="authorized complete campaign"):
        project_annotation(authority=None, trajectory=None, annotation_path=target)
    assert list(tmp_path.iterdir()) == []


def test_stage_b_parity_feedback_and_metrics():
    arms = ("F-B", "K-B", "F-M", "K-M", "N-shadow", "STATIC-4+64")
    metrics = {a:{"ber":.2} for a in arms}
    record = {"metrics":metrics, "query_count":{"F-B":1}, "queried_label_composition":{"F-B":{"normal":0,"defect":1}}, "first_queried_defect_position":{"F-B":2}}
    actual = {**record, "arm_metrics":metrics}
    verify_evaluation_parity(actual, record)
    for field, value in (("queried_label_composition", {"F-B":{"normal":1,"defect":0}}), ("first_queried_defect_position", {"F-B":None}), ("arm_metrics", {a:{"ber":.3} for a in arms})):
        bad = copy.deepcopy(actual); bad[field] = value
        with pytest.raises(AuditError): verify_evaluation_parity(bad, record)


def test_combined_grant_metadata_and_tampering(monkeypatch, tmp_path):
    from experiments.calibration.bea_temporal_audit import artifacts
    from experiments.calibration.bea_temporal_audit.campaign import CATEGORIES, OLD, NEW
    from experiments.calibration.bea_temporal_audit.integrity import digest
    pairs = [(c,p) for c in CATEGORIES for p in ("P0","P1","P2")]
    names = {OLD:"realiad-final-20261111-r1-40d61c3", NEW:"synthetic-continuation"}
    plans = {source:({"source_identity":{"campaign_run_id":name}}, digest(source)) for source,name in names.items()}
    rows = {(c,p):{"role_manifest_sha256":digest(c+p+"role"), "stream_manifest_sha256":digest(c+p+"stream"), "test_event_count":5, "source_group_count":1} for c,p in pairs}
    monkeypatch.setattr(artifacts, "registered_trajectory", lambda plan,c,p: rows[(c,p)])
    monkeypatch.setattr(artifacts, "sha256_file", lambda path: digest(path.name))
    records, entries = [], []
    for i,(c,p) in enumerate(pairs):
        source = OLD if i < 36 else NEW
        seal, annotation = digest(c+p+"seal"), digest(c+"annotation")
        records.append({"category":c,"condition":p,"source_commit":source,"campaign_run_id":names[source], "stage_a_category_seal_sha256":seal,"annotation_sha256":annotation})
        entries.append({"category":c,"condition":p,"source_commit":source,"campaign_run_id":names[source], "execution_plan_file_sha256":plans[source][1], "role_manifest_sha256":rows[c,p]["role_manifest_sha256"], "stream_manifest_sha256":rows[c,p]["stream_manifest_sha256"], "expected_event_count":5,"expected_source_groups":1,"annotation_sha256":annotation,"stage_a_category_seal_sha256":seal,"stage_a_authorization_file_sha256":digest("a"),"stage_a_journal_sha256":digest("j"), "stage_a_status_path":str(tmp_path/"outputs/calibration/final_benchmark"/names[source]/"stage_a"/c/p/"attempt_01"/"attempt_status.json")})
    grant_sha = digest("grant")
    summary = {"schema":"realiad.stage-b-final-metrics.v1","status":"STAGE_B_ALL_90_CPU_EVALUATED", "scientific_results":records,"stage_b_authorization_sha256":grant_sha}
    grant = {"schema":"realiad.stage-b-cross-campaign-authorization.v1","status":"AUTHORIZED_POST_STAGE_A_LABEL_PROJECTION", "scope":"stage_b_category_evaluation","dataset_id":"realiad", "registered_category_count":30,"registered_trajectory_count":90,"registered_test_image_events":343755,"test_mask_access":False,"stage_a_reexecution":False,"model_loading":False,"evaluation_code_commit":NEW,"evaluation_file_sha256":digest("evaluation.py"),"metrics_file_sha256":digest("metrics.py"),"scientific_execution_provenance":{v:k for k,v in names.items()},"authorizations":entries}
    def check(g, sha=grant_sha):
        return artifacts.validate_combined_grant(grant=g, grant_sha256=sha, summary=summary, plans=plans, repository=tmp_path, metadata_only=True)
    verified = check(grant)
    assert len(verified.entries) == 90
    view = artifacts.category_projection_authorization(verified, pairs[0])
    assert view.source_sha256 == entries[0]["annotation_sha256"]
    for change in ("scope", "seal", "source"):
        bad = copy.deepcopy(grant)
        if change == "scope": bad["scope"] = "other"
        if change == "seal": bad["authorizations"][0]["stage_a_category_seal_sha256"] = digest("wrong")
        if change == "source": bad["authorizations"][0]["source_commit"] = NEW
        with pytest.raises(AuditError): check(bad)
    with pytest.raises(AuditError): check(grant, digest("tampered file"))


def test_synthetic_seal_projection_evaluation_parity_without_label_files(tmp_path):
    from test_audit import make_rows, sealed_fixture
    from experiments.calibration.bea_temporal_audit.integrity import load_verified
    from experiments.calibration.final_benchmark.evaluation import LabelProjectionAuthorization, evaluate_sealed_category
    bundle, plan, data, _ = fixture()
    stream = validate_embedded_stream(bundle=bundle, plan_row=plan)
    rows = make_rows(groups=2, cap=0)
    for row, event in zip(rows, stream["events"]):
        for key in ("event_id_sha256", "image_identity_sha256", "source_group_sha256", "camera_id"):
            row[key] = event[key]
    h = hashlib.sha256(data).hexdigest()
    root, ident, stream, record = sealed_fixture(tmp_path, rows, stream_override=stream, annotation_sha=h)
    trajectory = load_verified(status_path=root/"attempt_status.json", expected_identity=ident,
        registered_stream_manifest=stream, expected_event_count=10, hard_cap=0, stage_b_record=record)
    labels = _decode_annotation(data, trajectory=trajectory, annotation_sha256=h)
    grant = LabelProjectionAuthorization("AUTHORIZED_POST_STAGE_A_LABEL_PROJECTION", "stage_b_category_evaluation", h, "realiad", "widget", trajectory.seal.durable_seal_sha256)
    actual = evaluate_sealed_category(dataset_id="realiad", category="widget", rows=trajectory.rows,
        seal=trajectory.seal, group_seals=trajectory.groups, labels_by_event_id=labels, label_projection=grant)
    reference = {"metrics":actual["arm_metrics"], "query_count":actual["query_count"],
        "queried_label_composition":actual["queried_label_composition"], "first_queried_defect_position":actual["first_queried_defect_position"]}
    verify_evaluation_parity(actual, reference)
    assert not any("projection" in p.name or "labels" in p.name for p in tmp_path.rglob("*"))


def test_failed_campaign_prerequisite_does_not_create_output(tmp_path):
    from experiments.calibration.bea_temporal_audit.orchestrate import run_verified
    destination = tmp_path / "output"
    with pytest.raises(AuditError, match="all formal campaign seals"):
        run_verified(campaign=None, audit_authorization={}, audit_source_commit="a"*40,
            official_reference_path="unused", stage_b=None, annotation_paths={}, configuration_sha256="0"*64,
            artifact_hashes={}, destination=destination, repository=tmp_path)
    assert not destination.exists()


def test_preflight_summary_scans_outcomes_opaquely(tmp_path):
    from experiments.calibration.bea_temporal_audit.campaign import CATEGORIES, OLD, NEW
    from experiments.calibration.bea_temporal_audit.reference import read_stage_b_metadata
    from experiments.calibration.bea_temporal_audit.integrity import digest
    records = []
    for i,(c,p) in enumerate((c,p) for c in CATEGORIES for p in ("P0","P1","P2")):
        source = OLD if i < 36 else NEW
        records.append({"category":c,"condition":p,"source_commit":source,
            "campaign_run_id":"realiad-final-20261111-r1-40d61c3" if i < 36 else "continuation",
            "stage_a_category_seal_sha256":digest(c+p), "annotation_sha256":digest(c),
            "metrics":"OPAQUE_OUTCOME", "first_queried_defect_position":"OPAQUE_OUTCOME"})
    payload = {"schema":"realiad.stage-b-final-metrics.v1", "status":"STAGE_B_ALL_90_CPU_EVALUATED",
        "stage_b_authorization_sha256":digest("grant"), "scientific_results":records}
    path = tmp_path / "summary.json"
    path.write_bytes(json.dumps(payload).encode().replace(b"OPAQUE_OUTCOME", b"\xff"))
    result = read_stage_b_metadata(str(path))
    assert len(result["scientific_results"]) == 90
    assert "metrics" not in result["scientific_results"][0]
    assert "first_queried_defect_position" not in result["scientific_results"][0]
