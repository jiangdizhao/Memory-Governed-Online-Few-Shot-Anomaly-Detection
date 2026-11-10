from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.calibration.final_benchmark import identity
from experiments.calibration.final_benchmark.evaluation import (
    EvaluationProtocolError, LabelProjectionAuthorization, evaluate_sealed_category,
)
from experiments.calibration.final_benchmark.metrics import (
    MetricContractError, binary_metrics, compare_query_budgets,
    paired_category_bootstrap, worst_256_window_ber,
)
from experiments.calibration.final_benchmark.query import BeaPrice, MarginRho
from experiments.calibration.final_benchmark.runner import (
    CampaignProtocolError, GroupAtomicCategoryRunner, ScorePacket, TestEvent,
)
from experiments.calibration.final_benchmark.source_audit import audit_historical_interfaces
from experiments.calibration.final_benchmark.static_baselines import (
    FIXED_BYTES, static36_threshold_c32,
)
from experiments.calibration.final_benchmark.sink import FsyncGroupSink, SinkProtocolError
from experiments.calibration.final_benchmark.label_broker import LabelBrokerError, RealIADQueryLabelBroker
from experiments.calibration import r10_signed_knn_baseline as r10


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _SyntheticBackend:
    def __init__(self):
        self.arms = {arm: 0 for arm in ("F-B", "K-B", "F-M", "K-M")}
        self.shadow = 0
        self.static = {arm: 0 for arm in ("STATIC-4+64", "STATIC-36+32")}
        self.scores_seen: list[tuple[str, int, int]] = []
        self.shadow_feedback: list[int] = []
        self.shadow_fb_state_on_feedback: list[int] = []

    def state_digest(self, arm_id):
        return sha(f"arm:{arm_id}:{self.arms[arm_id]}")

    def snapshot(self, arm_id):
        return self.arms[arm_id]

    def score(self, arm_id, event, snapshot):
        self.scores_seen.append((arm_id, event.position, snapshot))
        score = (event.position + snapshot) / 100.0
        return ScorePacket(score, int(score >= 0.5), 0.5,
                           10.0 if arm_id in ("F-B", "K-B") else 0.0,
                           self.arms[arm_id], self.state_digest(arm_id), threshold=0.5)

    def commit_feedback(self, arm_id, event, packet, label, receipt):
        if label == 0:
            self.arms[arm_id] += 1
        return {"state_version": self.arms[arm_id]}

    def shadow_snapshot(self, fb_snapshot):
        return (fb_snapshot, self.shadow)

    def shadow_state_digest(self):
        return sha(f"shadow:{self.shadow}")

    def score_shadow(self, event, fb_snapshot, shadow_snapshot):
        return ScorePacket((event.position + fb_snapshot + shadow_snapshot[1]) / 100.0,
                           0, 0.5, 0.0, self.shadow, self.shadow_state_digest())

    def commit_shadow_feedback(self, event, packet, label, receipt):
        self.shadow_feedback.append(label)
        self.shadow_fb_state_on_feedback.append(self.arms["F-B"])
        if label == 0:
            self.shadow += 1
        return {"negative_only_count": self.shadow}

    def static_state_digest(self, baseline_id):
        return sha(f"static:{baseline_id}:{self.static[baseline_id]}")

    def static_score(self, baseline_id, event):
        score = event.position / 100.0
        return ScorePacket(score, int(score >= 0.5), 0.5, 0.0, 0,
                           self.static_state_digest(baseline_id), threshold=0.5)


class _SyntheticBroker:
    def __init__(self, sink=None):
        self.disclosures: list[tuple[str, int]] = []
        self.sink = sink

    def disclose(self, *, arm_id, event, query_receipt):
        if self.sink is not None:
            assert self.sink.group_count > event.position // 5
        self.disclosures.append((arm_id, event.position))
        label = int(event.position % 2)
        return label, {"opaque_receipt_sha256": sha(f"{arm_id}:{event.position}:{label}")}


class _SyntheticSink:
    def __init__(self):
        self.group_count = 0
        self.event_count = 0
        self.groups = []

    def append_group(self, *, group_seal, score_rows):
        self.groups.append({**dict(group_seal), "rows": list(score_rows)})
        self.group_count += 1
        self.event_count += len(score_rows)
        return sha(json.dumps(self.groups[-1], sort_keys=True))

    def seal_category(self, seal):
        assert seal["group_count"] == self.group_count
        assert seal["completed_event_count"] == self.event_count
        return sha(json.dumps(dict(seal), sort_keys=True))


def event_group(start: int, group_index: int) -> list[TestEvent]:
    group = sha(f"group-{group_index}")
    return [TestEvent(start + camera - 1, sha(f"event-{start + camera - 1}"),
                      sha(f"image-{start + camera - 1}"), group, camera,
                      feature_ref=f"synthetic-feature-{start + camera - 1}")
            for camera in range(1, 6)]


class IdentityTests(unittest.TestCase):
    def test_private_locator_resolves_only_registered_identity_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = "OK/S0001/widget_0001_OK_C1_20260101000000.png"
            test = "NG/AB/S0002/widget__0002_NG_AB_C1_20260101000000.png"
            for key in (train, test):
                path = root / "images" / "widget" / key
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic")
            annotation = root / "annotation.json"
            annotation.write_text(json.dumps({"meta": {"normal_class": "OK", "prefix": "widget"},
                "train": [{"image_path": train, "anomaly_class": "OK"}],
                "test": [{"image_path": test, "anomaly_class": "SYNTHETIC_PRIVATE_OUTCOME",
                          "mask_path": "SYNTHETIC_PRIVATE_MASK"}]}), encoding="utf-8")
            digest = hashlib.sha256(annotation.read_bytes()).hexdigest()
            image_id = identity._image_identity_digest(test)
            locators = identity.resolve_realiad_private_locators(
                dataset_root=root, annotation_path=annotation, annotation_sha256=digest,
                category="widget", split="test", selected_image_ids=[image_id])
            self.assertEqual(locators[image_id]["official_image_key"], test)
            self.assertEqual(locators[image_id]["relative_id"], test)
            self.assertNotIn("anomaly_class", locators[image_id])
            with self.assertRaises(identity.IdentityAuditError):
                identity.resolve_realiad_private_locators(
                    dataset_root=root, annotation_path=annotation, annotation_sha256=sha("wrong"),
                    category="widget", split="test", selected_image_ids=[image_id])

    def test_visa_exception_candidate_stays_disabled_and_checks_units(self):
        rows = [{"image_identity_sha256": sha(f"visa-image-{i}"),
                 "image_content_sha256": sha(f"visa-content-{i if i != 69 else 68}"),
                 "role": "SUPPORT" if i < 4 else "C64" if i < 68 else "TEST"}
                for i in range(70)]
        units = {row["image_identity_sha256"]: sha(f"unit-{i if i != 69 else 68}")
                 for i, row in enumerate(rows)}
        result = identity.prepare_visa_exception_candidate(identity_rows=rows,
                                                           proposed_unit_by_image=units)
        self.assertEqual(result["status"], "RESEARCHER_DECISION_REQUIRED_NOT_EXECUTABLE")
        self.assertEqual(result["query_rate_target"], .10)
        self.assertEqual(result["test_unit_count"], 1)
        bad = dict(units)
        bad[rows[69]["image_identity_sha256"]] = sha("different-unit")
        with self.assertRaises(identity.IdentityAuditError):
            identity.prepare_visa_exception_candidate(identity_rows=rows, proposed_unit_by_image=bad)

    def test_role_seed_replay_and_static36_partition(self):
        release = sha("official-release")
        role_payload = "\0".join(("Gate0-role-v1", release, "widget", "0", "widget_group_000"))
        self.assertEqual(identity._domain_hash("Gate0-role-v1", release, "widget", "0", "widget_group_000"),
                         hashlib.sha256(role_payload.encode()).hexdigest())
        rows = []
        for group_index in range(70):
            group = f"widget_group_{group_index:03d}"
            image = f"widget_{group_index:03d}_C1_20260101000000.png"
            rows.append({"_official_group_key": group, "_official_image_key": image,
                         "image_identity_sha256": identity._image_identity_digest(image),
                         "image_content_sha256": sha(f"content-{group_index}")})
        p0 = identity._select_role_groups(
            dataset_id="realiad", category="widget", support_seed=0,
            dataset_release_sha256=release, train_normal_rows=rows, test_group_keys=set())
        p0_repeat = identity._select_role_groups(
            dataset_id="realiad", category="widget", support_seed=0,
            dataset_release_sha256=release, train_normal_rows=rows, test_group_keys=set())
        p2 = identity._select_role_groups(
            dataset_id="realiad", category="widget", support_seed=1,
            dataset_release_sha256=release, train_normal_rows=rows, test_group_keys=set())
        self.assertEqual(p0["selected"], p0_repeat["selected"])
        p1 = identity._select_role_groups(
            dataset_id="realiad", category="widget", support_seed=0,
            dataset_release_sha256=release, train_normal_rows=rows, test_group_keys=set())
        self.assertEqual(p0["selected"], p1["selected"])
        self.assertNotEqual([x["image_identity_sha256"] for x in p0["selected"][:4]],
                            [x["image_identity_sha256"] for x in p2["selected"][:4]])
        self.assertEqual(len({x["source_group_sha256"] for x in p0["selected"]}), 68)
        schedule = identity.build_static36_schedule(
            p0, dataset_release_sha256=release, category="widget", support_seed=0)
        self.assertEqual(len(schedule["reference_identity_sha256"]), 36)
        self.assertEqual(len(schedule["c32_identity_sha256"]), 32)
        self.assertEqual(set(schedule["principal_c64_identity_sha256"]),
                         {row["image_identity_sha256"] for row in p0["selected"][4:]})
        self.assertFalse(schedule["principal_roles_mutated"])
        self.assertTrue(all("static36_role" not in row for row in p0["selected"]))
        with self.assertRaises(identity.IdentityAuditError):
            identity.build_static36_schedule(
                p0, dataset_release_sha256=sha("other-release"), category="widget", support_seed=0)
        with self.assertRaises(identity.IdentityAuditError):
            identity._select_role_groups(
                dataset_id="realiad", category="widget", support_seed=0,
                dataset_release_sha256=release, train_normal_rows=rows,
                test_group_keys={"widget_group_000"})

    def test_five_camera_stream_and_missing_duplicate_group_fail_closed(self):
        release = sha("release")
        rows = []
        for group_index in range(2):
            for camera in range(1, 6):
                key = f"widget_group{group_index}_C{camera}_20260101000000.png"
                rows.append({"_official_group_key": f"widget_group{group_index}",
                             "_official_image_key": key, "camera_id": camera,
                             "original_position": len(rows),
                             "image_identity_sha256": identity._image_identity_digest(key),
                             "image_content_sha256": sha(key)})
        stream = identity._stream_order(category="widget", stream_seed=0,
                                        dataset_release_sha256=release, test_rows=rows)
        self.assertEqual([row["camera_id"] for row in stream[:5]], [1, 2, 3, 4, 5])
        self.assertEqual(sorted(row["original_position"] for row in stream), list(range(10)))
        with self.assertRaises(identity.IdentityAuditError):
            identity._stream_order(category="widget", stream_seed=0,
                                    dataset_release_sha256=release, test_rows=rows[:-1])

    def test_json_projection_drops_test_labels_and_masks(self):
        from experiments.calibration import r7_shuffled_correction_4class as r7
        decoded = []
        decoded_keys = []
        original_decode = r7._decode_field
        original_string = r7._string
        def capture(data, bounds):
            decoded.append(data[bounds[0]:bounds[1]])
            return original_decode(data, bounds)
        def capture_key(data, offset):
            result = original_string(data, offset)
            decoded_keys.append(result[0])
            return result
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "category.json"
            path.write_text(json.dumps({
                "meta": {"normal_class": "OK", "prefix": "images"},
                "train": [{"image_path": "normal.png", "anomaly_class": "OK"}],
                "test": [{"image_path": "test.png", "anomaly_class": "SYNTHETIC_OUTCOME",
                          "mask_path": "SYNTHETIC_MASK",
                          "mask": {"SYNTHETIC_SECRET_KEY": ['} ] " \\', {"inner": "MASK"}]}}],
            }), encoding="utf-8")
            with patch.object(r7, "_decode_field", side_effect=capture), \
                    patch.object(r7, "_string", side_effect=capture_key):
                meta, train, test = identity._identity_only_realiad_sections(path)
        self.assertEqual(train[0]["anomaly_class"], "OK")
        self.assertEqual(test, [{"image_path": "test.png"}])
        self.assertNotIn("anomaly_class", test[0])
        self.assertNotIn("mask_path", test[0])
        self.assertNotIn(b'"SYNTHETIC_OUTCOME"', decoded)
        self.assertNotIn(b'"SYNTHETIC_MASK"', decoded)
        self.assertNotIn("SYNTHETIC_SECRET_KEY", decoded_keys)

    def test_visa_test_projection_skips_label_and_mask_cells(self):
        row = identity._parse_csv_projected_line(
            "widget,test,defect,images/widget/test.png,mask.png",
            ["object", "split", "label", "image", "mask"],
            train_label_allowed=True)
        self.assertEqual(row, {"object": "widget", "split": "test", "image": "images/widget/test.png"})
        undecodable_outcomes = identity._parse_csv_projected_line(
            b"widget,test,\xff,images/widget/test.png,\xfe",
            ["object", "split", "label", "image", "mask"], train_label_allowed=True)
        self.assertEqual(undecodable_outcomes, row)

    def test_manifest_rejects_restricted_fields_at_any_depth(self):
        manifest = {"schema": "final-benchmark.identity-manifest.v1", "dataset_id": "realiad",
                    "identity_records": [], "nested": {"annotation": {"mask_path": "x"}}}
        with self.assertRaises(identity.IdentityAuditError):
            identity.validate_identity_manifest(manifest)


class QueryAndRunnerTests(unittest.TestCase):
    def test_margin_hash_cold_start_and_bea_price_recurrence(self):
        controller = MarginRho("realiad", "widget", "F-M", 0.03, 100, 0, 0)
        event_sha = sha("event")
        proposal = controller.propose_group_view(
            position=0, event_id_sha256=event_sha, psi=0.5, group_starts_cold=True)
        payload = "\0".join(("Gate0-Margin-rho-v1", "realiad", "widget", "F-M", "0", "0",
                             event_sha, "cold-start")).encode()
        expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / float(1 << 64)
        self.assertEqual(proposal.proposed, expected < 0.03)
        no_query_event = None
        for index in range(100):
            candidate = sha(f"cold-no-query-{index}")
            candidate_payload = "\0".join(("Gate0-Margin-rho-v1", "realiad", "widget",
                                             "F-M", "0", "0", candidate, "cold-start")).encode()
            candidate_uniform = int.from_bytes(hashlib.sha256(candidate_payload).digest()[:8], "big") / float(1 << 64)
            if candidate_uniform >= 0.03:
                no_query_event = candidate
                break
        self.assertIsNotNone(no_query_event)
        self.assertFalse(controller.propose_group_view(
            position=1, event_id_sha256=no_query_event, psi=0.4,
            group_starts_cold=True).proposed)
        controller.history.extend([0.1] * 20)
        tie = controller.propose_group_view(
            position=20, event_id_sha256=event_sha, psi=0.1, group_starts_cold=False)
        self.assertEqual(tie.threshold, 0.1)
        self.assertEqual(tie.reason, "identity_hashed_quantile_tie")
        bea = BeaPrice("F-B", 0.03)
        pre_group_price = bea.price
        self.assertTrue(bea.propose(1.0, group_price=pre_group_price).proposed)
        updated = bea.observe_final_actions([False, True])
        self.assertAlmostEqual(updated[0], 0.8 + 0.02 * (0.0 - 0.03))
        self.assertAlmostEqual(updated[1], updated[0] + 0.02 * (0.5 - 0.03))

    def test_group_atomic_barrier_caps_shadow_and_seal(self):
        backend, sink = _SyntheticBackend(), _SyntheticSink()
        broker = _SyntheticBroker(sink)
        expected_order = [sha(f"group-{group_index}") for group_index in range(20)]
        runner = GroupAtomicCategoryRunner(
            dataset_id="realiad", category="widget", condition_id="P0",
            support_seed=0, stream_seed=0, expected_event_count=100,
            expected_group_order=expected_order, rho=0.03, backend=backend, broker=broker, sink=sink)
        for group_index in range(20):
            events = event_group(group_index * 5, group_index)
            # Every score in a group saw the exact same pre-group arm state.
            rows = runner.process_group(events)
            scores_for_group = backend.scores_seen[-20:]
            for arm in ("F-B", "K-B", "F-M", "K-M"):
                self.assertEqual(len({state for seen_arm, _, state in scores_for_group if seen_arm == arm}), 1)
            self.assertTrue(all("label" not in row for row in rows))
            if group_index == 3:
                self.assertTrue(all(row["query_actions"]["F-M"]["proposal_reason"] ==
                                    "identity_hashed_cold_start" for row in rows))
            if group_index == 4:
                self.assertTrue(all(row["query_actions"]["F-M"]["proposal_reason"] !=
                                    "identity_hashed_cold_start" for row in rows))
            self.assertEqual(sum(bool(row["query_actions"]["F-B"]["executed"]) for row in rows),
                             1 if group_index < 3 else 0)
        seal = runner.seal()
        self.assertEqual(seal.status, "SEALED_COMPLETE")
        self.assertEqual(sink.group_count, 20)
        accounting = runner.accounting()
        for arm in ("F-B", "K-B"):
            self.assertLessEqual(accounting[arm]["executed"], 3)
            self.assertEqual(accounting[arm]["executed"], 3)
            self.assertEqual(accounting[arm]["query_receipt_count"], 3)
            self.assertEqual(accounting[arm]["first_cap_exhaustion_position"], 10)
            self.assertEqual(accounting[arm]["max_no_query_gap_events"], 89)
        self.assertEqual(backend.shadow_feedback, [0, 1, 0])
        self.assertEqual(backend.shadow_fb_state_on_feedback, [0, 1, 1])
        self.assertEqual(backend.arms["F-B"], 2)
        self.assertEqual(backend.arms["K-B"], 2)
        self.assertEqual(backend.shadow, 2)
        self.assertEqual(len(broker.disclosures), sum(item["executed"] for item in accounting.values()))
        for arm in ("F-M", "K-M"):
            self.assertLessEqual(accounting[arm]["executed"], 3)
        self.assertTrue(all(row["query_actions"][arm]["query_receipt_sha256"]
                            for row in runner.rows for arm in ("F-B", "K-B")
                            if row["query_actions"][arm]["executed"]))
        self.assertEqual(len({row["event_id_sha256"] for row in runner.rows}), 100)
        self.assertEqual(runner.resource_report()["resource_status"],
                         "INTERFACE_PRESENT_DEVICE_AND_MEMORY_SAMPLING_PENDING_GATE2")

    def test_invalid_group_and_stage_a_completion_fail_closed(self):
        runner = GroupAtomicCategoryRunner(
            dataset_id="realiad", category="widget", condition_id="P0",
            support_seed=0, stream_seed=0, expected_event_count=10,
            expected_group_order=[sha("group-0"), sha("group-1")],
            rho=0.03, backend=_SyntheticBackend(), broker=_SyntheticBroker(), sink=_SyntheticSink())
        with self.assertRaises(CampaignProtocolError):
            runner.process_group(event_group(0, 0)[:-1])
        with self.assertRaises(CampaignProtocolError):
            runner.seal()
        with self.assertRaises(CampaignProtocolError):
            GroupAtomicCategoryRunner(
                dataset_id="visa_1cls", category="widget", condition_id="P0",
                support_seed=0, stream_seed=0, expected_event_count=100,
                expected_group_order=[sha(f"visa-group-{i}") for i in range(20)],
                rho=0.10, backend=_SyntheticBackend(), broker=_SyntheticBroker(), sink=_SyntheticSink())

    def test_unqueried_c64_role_cannot_enter_stream(self):
        with self.assertRaises(CampaignProtocolError):
            TestEvent(0, sha("e"), sha("i"), sha("g"), 1, role="C64")


class EvaluationAndSourceTests(unittest.TestCase):
    def test_real_query_broker_requires_fsynced_action_and_reads_only_queried_views(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            release = sha("release")
            annotation = root / "synthetic.json"
            source_rows, identity_rows, locators = [], [], {}
            for camera in range(1, 6):
                image_key = f"NG/AB/S0001/widget__0001_NG_AB_C{camera}_20260101000000.png"
                group, _ = identity._realiad_identity_key("widget", image_key)
                image_id = identity._image_identity_digest(image_key)
                source_rows.append({"image_path": image_key,
                                    "anomaly_class": "OK" if camera == 1 else f"synthetic-defect-{camera}",
                                    "mask": {"SYNTHETIC_SECRET_KEY": "synthetic-unused-mask"}})
                identity_rows.append({"_official_group_key": group, "_official_image_key": image_key,
                    "camera_id": camera, "original_position": camera - 1,
                    "image_identity_sha256": image_id, "image_content_sha256": sha(image_key)})
                locators[image_id] = {"official_image_key": image_key, "official_group_key": group,
                                      "image_path": str(root / image_key)}
            annotation.write_text(json.dumps({"meta": {"normal_class": "OK"}, "test": source_rows}),
                                  encoding="utf-8")
            records = identity._stream_order(category="widget", stream_seed=0,
                                             dataset_release_sha256=release, test_rows=identity_rows)
            events = [TestEvent(row["position"], row["event_id_sha256"], row["image_identity_sha256"],
                                row["source_group_sha256"], row["camera_id"]) for row in records]
            sink = FsyncGroupSink(root / "scores.jsonl", identity={"run_id": "synthetic"})
            broker = RealIADQueryLabelBroker(
                category="widget", condition_id="P0", stream_seed=0, dataset_release_sha256=release,
                annotation_path=annotation, annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
                stream_records=records, private_locators=locators, sink=sink,
                private_feedback_path=root / "private.jsonl")
            try:
                receipts = {}
                for arm, index in (("F-B", 0), ("K-B", 1)):
                    event = events[index]
                    receipts[arm] = {"schema": "final-benchmark.query-receipt.v1", "arm_id": arm,
                        "category": "widget", "condition_id": "P0", "event_id_sha256": event.event_id_sha256,
                        "source_group_sha256": event.group_sha256, "position": event.position}
                with self.assertRaises(LabelBrokerError):
                    broker.disclose(arm_id="F-B", event=events[0], query_receipt=receipts["F-B"])
                self.assertEqual(broker.decoded_queries, [])
                rows = []
                for index, event in enumerate(events):
                    actions = {}
                    for arm in ("F-B", "K-B"):
                        executed = (arm == "F-B" and index == 0) or (arm == "K-B" and index == 1)
                        digest = hashlib.sha256(json.dumps(receipts[arm], sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest() if executed else None
                        actions[arm] = {"executed": executed, "query_receipt_sha256": digest}
                    rows.append({"position": index, "event_id_sha256": event.event_id_sha256,
                        "image_identity_sha256": event.image_identity_sha256,
                        "source_group_sha256": event.group_sha256, "camera_id": event.camera_id,
                        "query_actions": actions})
                rows_sha = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
                invalid_rows = json.loads(json.dumps(rows))
                invalid_rows[1]["query_actions"]["F-B"] = {
                    "executed": True, "query_receipt_sha256": sha("extra-query")}
                invalid_sha = hashlib.sha256(json.dumps(invalid_rows, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
                with self.assertRaises(SinkProtocolError):
                    sink.append_group(group_seal={"start_position": 0, "end_position_exclusive": 5,
                        "source_group_sha256": events[0].group_sha256, "score_rows_sha256": invalid_sha,
                        "status": "SEALED_BEFORE_LABEL_DISCLOSURE"}, score_rows=invalid_rows)
                sink.append_group(group_seal={"start_position": 0, "end_position_exclusive": 5,
                    "source_group_sha256": events[0].group_sha256, "score_rows_sha256": rows_sha,
                    "status": "SEALED_BEFORE_LABEL_DISCLOSURE"}, score_rows=rows)
                bad_rows = json.loads(json.dumps(source_rows))
                bad_rows[0]["image_path"] = "different-prefix/" + bad_rows[0]["image_path"]
                bad_annotation = root / "synthetic-mismatched.json"
                bad_annotation.write_text(json.dumps({"meta": {"normal_class": "OK"}, "test": bad_rows}),
                                          encoding="utf-8")
                bad_broker = RealIADQueryLabelBroker(
                    category="widget", condition_id="P0", stream_seed=0, dataset_release_sha256=release,
                    annotation_path=bad_annotation, annotation_sha256=hashlib.sha256(bad_annotation.read_bytes()).hexdigest(),
                    stream_records=records, private_locators=locators, sink=sink,
                    private_feedback_path=root / "private-mismatched.jsonl")
                try:
                    with self.assertRaisesRegex(LabelBrokerError, "annotation image identity"):
                        bad_broker.disclose(arm_id="F-B", event=events[0], query_receipt=receipts["F-B"])
                    self.assertEqual(bad_broker.decoded_queries, [])
                finally:
                    bad_broker.close()
                from experiments.calibration import r7_shuffled_correction_4class as r7
                decoded_values, decoded_keys = [], []
                original_decode, original_string = r7._decode_field, r7._string
                def capture_value(data, bounds):
                    decoded_values.append(bytes(data[bounds[0]:bounds[1]]))
                    return original_decode(data, bounds)
                def capture_key(data, offset):
                    result = original_string(data, offset)
                    decoded_keys.append(result[0])
                    return result
                with patch.object(r7, "_decode_field", side_effect=capture_value), \
                        patch.object(r7, "_string", side_effect=capture_key):
                    self.assertEqual(broker.disclose(arm_id="F-B", event=events[0], query_receipt=receipts["F-B"])[0], 0)
                    self.assertEqual(broker.disclose(arm_id="K-B", event=events[1], query_receipt=receipts["K-B"])[0], 1)
                self.assertIn(b'"synthetic-defect-2"', decoded_values)
                self.assertNotIn(b'"synthetic-defect-3"', decoded_values)
                self.assertNotIn(b'"synthetic-unused-mask"', decoded_values)
                self.assertNotIn("SYNTHETIC_SECRET_KEY", decoded_keys)
                self.assertEqual(broker.decoded_queries, [("F-B", 0), ("K-B", 1)])
                with self.assertRaises(LabelBrokerError):
                    broker.disclose(arm_id="F-B", event=events[0], query_receipt=receipts["F-B"])
                self.assertEqual(len((root / "private.jsonl").read_text(encoding="utf-8").splitlines()), 2)
            finally:
                broker.close()
                sink.close()

    def test_static_metrics_use_own_threshold_and_static36_is_p0_only(self):
        class StaticThresholdBackend(_SyntheticBackend):
            def __init__(self):
                super().__init__()
                self.static_seen = []

            def static_score(self, baseline_id, event):
                self.static_seen.append(baseline_id)
                threshold = {"STATIC-4+64": 0.015, "STATIC-36+32": 0.025}[baseline_id]
                score = event.position / 100.0
                return ScorePacket(score, int(score >= threshold), 0.5, 0.0, 0,
                                   self.static_state_digest(baseline_id), threshold=threshold)

        for condition, support_seed, stream_seed in (("P0", 0, 0), ("P1", 0, 11), ("P2", 1, 0)):
            backend = StaticThresholdBackend()
            runner = GroupAtomicCategoryRunner(
                dataset_id="realiad", category="widget", condition_id=condition,
                support_seed=support_seed, stream_seed=stream_seed, expected_event_count=5,
                expected_group_order=[sha("group-0")], rho=0.03, backend=backend,
                broker=_SyntheticBroker(), sink=_SyntheticSink())
            runner.process_group(event_group(0, 0))
            seal = runner.seal()
            projection = LabelProjectionAuthorization(
                "AUTHORIZED_POST_STAGE_A_LABEL_PROJECTION", "stage_b_category_evaluation",
                sha("synthetic"), "realiad", "widget", seal.durable_seal_sha256)
            result = evaluate_sealed_category(
                dataset_id="realiad", category="widget", rows=runner.rows, seal=seal,
                group_seals=runner.group_seals,
                labels_by_event_id={row["event_id_sha256"]: i % 2 for i, row in enumerate(runner.rows)},
                label_projection=projection)
            static4 = result["arm_metrics"]["STATIC-4+64"]
            self.assertAlmostEqual(static4["fpr"], 2 / 3)
            self.assertAlmostEqual(static4["fnr"], 1 / 2)
            self.assertAlmostEqual(static4["ber"], 7 / 12)
            self.assertAlmostEqual(static4["balanced_accuracy"], 5 / 12)
            self.assertIn("auroc", static4)
            self.assertIn("ap", static4)
            self.assertNotIn("ber", result["arm_metrics"]["N-shadow"])
            if condition == "P0":
                static36 = result["arm_metrics"]["STATIC-36+32"]
                self.assertAlmostEqual(static36["fpr"], 1 / 3)
                self.assertAlmostEqual(static36["fnr"], 1 / 2)
                self.assertAlmostEqual(static36["ber"], 5 / 12)
            else:
                self.assertNotIn("STATIC-36+32", result["arm_metrics"])
                self.assertNotIn("STATIC-36+32", backend.static_seen)

    def test_real_r10_corrected_c2_score_cpu_synthetic_interface(self):
        import numpy as np
        memory = r10.SignedKNNFIFO(capacity=1024)
        features = np.ones((8, 4096), dtype=np.float32)
        base_map = np.linspace(0.0, 1.0, 64, dtype=np.float32)
        top_ids = np.arange(8, dtype=np.int64)
        empty_score, empty_delta, _ = r10._corrected_c2_score(
            base_map, top_ids, features, memory, position=10,
            group_sha256=sha("group-current"))
        self.assertTrue(np.isfinite(empty_score))
        self.assertEqual(float(np.max(np.abs(empty_delta))), 0.0)
        memory.insert(features[:3], sign=1, source_group_sha256=sha("group-prior"), position=1)
        score, delta, diagnostics = r10._corrected_c2_score(
            base_map, top_ids, features, memory, position=10,
            group_sha256=sha("group-current"))
        self.assertTrue(np.isfinite(score))
        self.assertTrue(np.isfinite(delta).all())
        self.assertEqual(diagnostics["eligible_anchor_count"], 3)

    def test_metric_values_undefined_classes_and_window_rules(self):
        result = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], [0, 1, 0, 1])
        self.assertAlmostEqual(result["auroc"], 0.75)
        self.assertAlmostEqual(result["ap"], 5.0 / 6.0)
        self.assertAlmostEqual(result["ber"], 0.5)
        undefined = binary_metrics([1, 1], [0.2, 0.8], [0, 1])
        self.assertIsNone(undefined["ber"])
        self.assertEqual(undefined["status"], "UNDEFINED_MISSING_NEGATIVE_CLASS")
        with self.assertRaises(MetricContractError):
            paired_category_bootstrap({"widget": {"P0": {}}}, "H_Q", replicates=10)
        self.assertEqual(worst_256_window_ber("visa_1cls", [0, 1], [0, 1])["status"], "NOT_APPLICABLE")
        self.assertEqual(FIXED_BYTES, 67_108_864)
        self.assertEqual(static36_threshold_c32(list(range(32))), 28.0)
        unmatched = compare_query_budgets(
            dataset_id="visa_1cls", rho=0.10, total_events={"a": 100, "b": 100},
            left_queries={"a": 0, "b": 10}, right_queries={"a": 10, "b": 0})
        self.assertTrue(unmatched["equal_maximum_category_caps"])
        self.assertTrue(unmatched["category_equal_macro_realized_rate_match_pass"])
        self.assertFalse(unmatched["all_category_realized_rate_matches_pass"])
        self.assertFalse(unmatched["approximately_rate_matched_description_pass"])

    def test_stage_b_requires_matching_seal_and_projection(self):
        backend, broker = _SyntheticBackend(), _SyntheticBroker()
        runner = GroupAtomicCategoryRunner(
            dataset_id="realiad", category="widget", condition_id="P0",
            support_seed=0, stream_seed=0, expected_event_count=5,
            expected_group_order=[sha("group-0")],
            rho=0.03, backend=backend, broker=broker, sink=_SyntheticSink())
        events = event_group(0, 0)
        runner.process_group(events)
        seal = runner.seal()
        projection = LabelProjectionAuthorization(
            "AUTHORIZED_POST_STAGE_A_LABEL_PROJECTION", "stage_b_category_evaluation",
            sha("synthetic"), "realiad", "widget", seal.durable_seal_sha256)
        labels = {row["event_id_sha256"]: index % 2 for index, row in enumerate(runner.rows)}
        result = evaluate_sealed_category(
            dataset_id="realiad", category="widget", rows=runner.rows,
            seal=seal, group_seals=runner.group_seals,
            labels_by_event_id=labels, label_projection=projection)
        self.assertEqual(result["outcome_access_phase"], "STAGE_B_AFTER_VALID_STAGE_A_SEAL")
        self.assertIn("F-B", result["arm_metrics"])
        with self.assertRaises(EvaluationProtocolError):
            evaluate_sealed_category(
                dataset_id="realiad", category="widget", rows=runner.rows,
                seal=seal, group_seals=runner.group_seals, labels_by_event_id=labels,
                label_projection=LabelProjectionAuthorization("PENDING", "", sha("x"),
                                                               "realiad", "widget", seal.durable_seal_sha256))

    def test_fsync_sink_persists_group_before_completion_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            sink = FsyncGroupSink(root / "group_journal.jsonl",
                                  identity={"run_id": "synthetic", "source_commit": sha("source")})
            rows = [{"position": index, "event_id_sha256": sha(f"event-{index}"),
                     "image_identity_sha256": sha(f"image-{index}"),
                     "source_group_sha256": sha("group"), "camera_id": index + 1}
                    for index in range(5)]
            group_seal = {"start_position": 0, "end_position_exclusive": 5,
                          "source_group_sha256": sha("group"),
                          "score_rows_sha256": hashlib.sha256(json.dumps(
                              rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                              allow_nan=False).encode("utf-8")).hexdigest(),
                          "status": "SEALED_BEFORE_LABEL_DISCLOSURE"}
            sink.append_group(group_seal=group_seal, score_rows=rows)
            category_seal = {"status": "SEALED_COMPLETE", "completed_event_count": 5,
                             "group_count": 1, "score_rows_sha256": sha("rows"),
                             "group_seals_sha256": sha("groups")}
            sink.seal_category(category_seal)
            sink.close()
            self.assertEqual(len((root / "group_journal.jsonl").read_text(encoding="utf-8").splitlines()), 2)
            self.assertTrue((root / "stage_a_category_seal.json").is_file())
            with self.assertRaises(SinkProtocolError):
                FsyncGroupSink(root / "group_journal.jsonl", identity={"run_id": "overwrite"})

    def test_bootstrap_is_reproducible_and_historical_interfaces_exist(self):
        conditions = {}
        for category_index, category in enumerate(("a", "b", "c")):
            conditions[category] = {}
            for condition in ("P0", "P1", "P2"):
                value = 0.1 + 0.01 * category_index
                conditions[category][condition] = {
                    "F-M": {"ber": value + 0.02}, "F-B": {"ber": value},
                    "K-M": {"ber": value + 0.03}, "K-B": {"ber": value},
                    "N-shadow": {"auroc": 0.6}, "F-B": {"ber": value, "auroc": 0.7},
                }
        first = paired_category_bootstrap(conditions, "H_Q", replicates=100)
        second = paired_category_bootstrap(conditions, "H_Q", replicates=100)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["category_equal_estimate"], 0.02)
        audit = audit_historical_interfaces()
        self.assertIn("r10_signed_knn", audit["bindings"])
        self.assertIn("group_sha256", audit["bindings"]["r10_signed_knn"]["required_signatures"]["_corrected_c2_score"])


if __name__ == "__main__":
    unittest.main()
