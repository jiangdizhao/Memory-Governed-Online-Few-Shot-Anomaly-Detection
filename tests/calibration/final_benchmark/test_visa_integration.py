from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.calibration.final_benchmark import identity, static_baselines
from experiments.calibration.final_benchmark.entrypoint import validate_release_identity
from experiments.calibration.final_benchmark.native_backend import NativeInputEncoder
from experiments.calibration.final_benchmark.runner import TestEvent as _TestEvent
from experiments.calibration.final_benchmark.sink import FsyncGroupSink
from experiments.calibration.final_benchmark.visa_label_broker import VisaQueryLabelBroker
from experiments.calibration.final_benchmark.visa_execution import create_source_bound_plan


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


class VisaIntegrationTests(unittest.TestCase):
    def _identity_manifest(self):
        categories = ["candle", "capsules", "cashew", "chewinggum", "fryum",
                      "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
        records = []
        for index in range(68):
            image_key = f"train/cashew/{index:03d}.jpg"
            records.append({"dataset_id": "visa_1cls", "category": "cashew", "split": "train",
                "original_position": index, "image_identity_sha256": identity._image_identity_digest(image_key),
                "file_locator_sha256": sha("locator-" + image_key),
                "image_content_sha256": sha("content-" + image_key), "image_bytes": 1024,
                "source_group_sha256": None, "camera_id": None, "train_normal_eligible": True})
        manifest = {"schema": "final-benchmark.identity-manifest.v1", "dataset_id": "visa_1cls",
            "category_count": 12, "categories": [{"category": c} for c in categories],
            "identity_records": records, "local_snapshot_sha256": sha("snapshot"),
            "source_artifacts": [], "source_artifacts_sha256": sha("artifacts"),
            "official_release_sha256": None,
            "test_label_access": "NOT_ACCESSED", "mask_access": "NOT_ACCESSED"}
        return manifest

    def test_deterministic_4_plus_64_role_selection_and_local_release_validation(self):
        manifest = self._identity_manifest()
        release_id = f"visa-1cls-local-snapshot-v1:{manifest['local_snapshot_sha256']}"
        release = hashlib.sha256(("Gate0-local-release-v1\0visa_1cls\0" + release_id).encode()).hexdigest()
        manifest["identity_records"].extend([
            {"dataset_id": "visa_1cls", "category": "cashew", "split": "test",
             "original_position": 68 + index,
             "image_identity_sha256": identity._image_identity_digest(f"test/cashew/{index}.jpg"),
             "file_locator_sha256": sha(f"test-locator-{index}"),
             "image_content_sha256": sha(f"test-content-{index}"), "image_bytes": 1024,
             "source_group_sha256": None, "camera_id": None, "train_normal_eligible": None}
            for index in range(2)])
        role = identity.prepare_visa_role_manifest(manifest, dataset_release_sha256=release,
                                                   category="cashew", support_seed=0)
        self.assertEqual([row["role"] for row in role["selected"]].count("SUPPORT"), 4)
        self.assertEqual([row["role"] for row in role["selected"]].count("C64"), 64)
        self.assertEqual(len({row["image_identity_sha256"] for row in role["selected"]}), 68)
        self.assertEqual(len({row["source_group_sha256"] for row in role["selected"]}), 68)
        self.assertNotEqual(set(row["image_identity_sha256"] for row in role["selected"]),
                            set(row["image_identity_sha256"] for row in manifest["identity_records"]
                                if row["split"] == "test"))
        role_hash = hashlib.sha256(canonical(role)).hexdigest()
        identity_hash = hashlib.sha256(canonical(manifest)).hexdigest()
        release_identity = {"schema": "final-benchmark.dataset-release-identity.v1",
            "dataset_id": "visa_1cls", "status": "RESEARCHER_APPROVED_LOCAL_SNAPSHOT",
            "provenance_kind": "RESEARCHER_APPROVED_LOCAL_SNAPSHOT",
            "dataset_release_sha256": release, "local_snapshot_sha256": manifest["local_snapshot_sha256"],
            "local_snapshot_identifier": release_id,
            "source_artifacts_sha256": manifest["source_artifacts_sha256"],
            "identity_manifest_sha256": identity_hash, "role_manifest_sha256": role_hash,
            "researcher_decision_sha256": sha("decision"), "official_release_sha256": None}
        self.assertEqual(validate_release_identity(identity_manifest=manifest, role_manifest=role,
                                                  release_identity=release_identity), release)
        with self.assertRaises(identity.IdentityAuditError):
            identity.prepare_visa_role_manifest(manifest, dataset_release_sha256=release,
                                                category="cashew", support_seed=True)

    def test_identity_parser_skips_test_label_and_mask_until_explicit_query(self):
        header = ["object", "split", "image", "label", "mask"]
        row = b'cashew,test,test/cashew/a.jpg,anomaly,secret-mask-path.png\n'
        safe = identity._parse_csv_projected_line(row, header, train_label_allowed=False)
        self.assertNotIn("label", safe)
        self.assertNotIn("mask", safe)
        queried = identity._parse_csv_projected_line(row, header,
            train_label_allowed=False, queried_test_label_allowed=True)
        self.assertEqual(queried["label"], "anomaly")
        self.assertNotIn("mask", queried)

    def test_visA_registered_encoder_uses_no_invented_camera_and_checks_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "cashew" / "sample.jpg"
            image.parent.mkdir()
            image.write_bytes(b"synthetic-image-bytes")
            release = sha("release")
            image_key = "cashew/sample.jpg"
            image_id = identity._image_identity_digest(image_key)
            unit = identity.visa_image_unit_sha256(release, "cashew", image_id)
            position = 9
            event_id = identity._domain_hash("Gate0-visa-event-v1", release, "cashew", "0",
                                             unit, image_id, str(position))
            record = {"dataset_id": "visa_1cls", "category": "cashew", "role": "TEST",
                "position": 0, "original_position": position, "image_identity_sha256": image_id,
                "image_content_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "source_group_sha256": unit, "event_id_sha256": event_id,
                "atomic_unit_kind": "VISA_SINGLE_IMAGE"}
            encoded = SimpleNamespace(encoding_seconds=0.01)
            encoder = object.__new__(NativeInputEncoder)
            encoder.encode_path = lambda _path: encoded
            event = encoder.encode_registered_visa_event(record=record,
                locator={"dataset_root": str(root), "image_path": str(image),
                         "relative_id": "cashew/sample.jpg", "official_image_key": image_key},
                category="cashew", dataset_release_sha256=release)
            self.assertIsNone(event.camera_id)
            self.assertEqual(event.atomic_unit_kind, "VISA_SINGLE_IMAGE")
            self.assertEqual(event.event_id_sha256, event_id)
            image.write_bytes(b"changed-bytes")
            with self.assertRaisesRegex(RuntimeError, "image bytes"):
                encoder.encode_registered_visa_event(record=record,
                    locator={"dataset_root": str(root), "image_path": str(image),
                             "relative_id": "cashew/sample.jpg", "official_image_key": image_key},
                    category="cashew", dataset_release_sha256=release)

    def test_static36_native_adapter_accepts_visa_schedule_without_changing_selection_math(self):
        manifest = self._identity_manifest()
        release = sha("visa-release")
        role = identity.prepare_visa_role_manifest(manifest, dataset_release_sha256=release,
                                                   category="cashew", support_seed=0)
        schedule = identity.build_static36_schedule(role, dataset_release_sha256=release,
                                                    category="cashew", support_seed=0)
        rows = {row["image_identity_sha256"]: row for row in role["selected"]}
        references = []
        support_paths = []
        for image_id in schedule["reference_identity_sha256"]:
            row = rows[image_id]
            path = f"{image_id}.png"
            references.append({"image_identity_sha256": image_id,
                "group_sha256": row["source_group_sha256"], "image_path": path,
                "relative_id": path})
            if row["role"] == "SUPPORT":
                support_paths.append(path)
        c32 = [{"image_identity_sha256": image_id,
                "source_group_sha256": rows[image_id]["source_group_sha256"]}
               for image_id in schedule["c32_identity_sha256"]]
        class FakeFixed:
            def __init__(self, **kwargs):
                self.threshold = kwargs["threshold"]
            def state_digest(self):
                return sha("static")
        trajectory = SimpleNamespace(bank=object(), fixed_sha256=sha("bank"))
        with patch("experiments.calibration.a4_equal_budget_allocation._kcenter_candidates",
                   return_value=(np.empty((1, 1)), ["candidate"], {"status": "synthetic"})), \
             patch("experiments.calibration.final_benchmark.static_baselines.select_static36_bank",
                   return_value=(object(), list(range(8192)), {"status": "synthetic"})), \
             patch("experiments.calibration.a4_equal_budget_allocation._build_bank",
                   return_value=trajectory), \
             patch("experiments.calibration.r7_shuffled_correction_4class._score_c64_base",
                   return_value=(np.arange(32, dtype=float), [], 0.0)), \
             patch("experiments.calibration.final_benchmark.static_baselines.FrozenNativeStatic", FakeFixed):
            model, receipt = static_baselines.build_static36_native(
                category="cashew", schedule=schedule, reference_items=references,
                c32_items=c32, impl=object(), p0=object(), r1=object(),
                backbone=object(), cfg=object(), support_cache={}, support_paths=support_paths)
        self.assertEqual(receipt["selected_count"], 8192)
        self.assertEqual(model.threshold, static_baselines.static36_threshold_c32(np.arange(32)))
        self.assertEqual(schedule["dataset_id"], "visa_1cls")

    def test_visa_broker_discloses_only_after_fsync_for_selected_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "split_csv").mkdir()
            (root / "LICENSE-DATASET").write_bytes(b"license")
            csv_path = root / "split_csv" / "1cls.csv"
            train_rows = []
            test_rows = []
            locators = {}
            identity_rows = []
            unit_by_image = {}
            release = sha("release")
            for index in range(68):
                key = f"train/{index}.jpg"
                (root / "train").mkdir(exist_ok=True)
                (root / key).write_bytes(f"train-{index}".encode())
                image_id = identity._image_identity_digest(key)
                content = hashlib.sha256((root / key).read_bytes()).hexdigest()
                unit_by_image[image_id] = sha(f"train-unit-{index}")
                identity_rows.append({"image_identity_sha256": image_id,
                                      "image_content_sha256": content, "role": "SUPPORT" if index < 4 else "C64"})
            for index, label in enumerate(("normal", "anomaly")):
                key = f"test/{index}.jpg"
                (root / "test").mkdir(exist_ok=True)
                (root / key).write_bytes(f"test-{index}".encode())
                image_id = identity._image_identity_digest(key)
                content = hashlib.sha256((root / key).read_bytes()).hexdigest()
                unit = identity.visa_image_unit_sha256(release, "cashew", image_id)
                unit_by_image[image_id] = unit
                identity_rows.append({"image_identity_sha256": image_id,
                                      "image_content_sha256": content, "role": "TEST"})
                test_rows.append({"dataset_id": "visa_1cls", "category": "cashew", "role": "TEST",
                    "position": index, "original_position": 68 + index,
                    "image_identity_sha256": image_id, "image_content_sha256": content,
                    "source_group_sha256": unit, "atomic_unit_sha256": unit,
                    "atomic_unit_kind": "VISA_SINGLE_IMAGE",
                    "event_id_sha256": identity._domain_hash("Gate0-visa-event-v1", release,
                        "cashew", "0", unit, image_id, str(68 + index))})
                locators[image_id] = {"dataset_root": str(root.resolve()),
                    "image_path": str((root / key).resolve()), "relative_id": key,
                    "official_image_key": key, "category": "cashew", "split": "test",
                    "original_position": 68 + index, "image_content_sha256": content}
            unit_order = [row["atomic_unit_sha256"] for row in test_rows]
            candidate = identity.prepare_visa_exception_candidate(identity_rows=identity_rows,
                proposed_unit_by_image=unit_by_image, proposed_test_unit_order=unit_order)
            for index in range(68):
                train_rows.append(["cashew", "train", f"train/{index}.jpg", "normal", ""])
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["object", "split", "image", "label", "mask"])
                writer.writerows(train_rows)
                writer.writerows([["cashew", "test", f"test/{index}.jpg", label, "mask-hidden.png"]
                                  for index, label in enumerate(("normal", "anomaly"))])
            csv_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            approval = {"unit_candidate_sha256": candidate["candidate_sha256"]}
            journal = root / "journal.jsonl"
            decoded_fields = []
            original_parse = identity._parse_csv_projected_line
            def parse_spy(*args, **kwargs):
                decoded_fields.append(bool(kwargs.get("queried_test_label_allowed", False)))
                return original_parse(*args, **kwargs)
            with FsyncGroupSink(journal, identity={"dataset_id": "visa_1cls"}) as sink:
                with patch.object(identity, "_parse_csv_projected_line", side_effect=parse_spy):
                    broker = VisaQueryLabelBroker(category="cashew", condition_id="P0",
                        dataset_root=root, dataset_release_sha256=release,
                        source_csv_sha256=csv_sha, stream_records=test_rows,
                        private_locators=locators, unit_candidate=candidate,
                        approval=approval, sink=sink, private_feedback_path=root / "feedback.jsonl")
                    self.assertFalse(any(decoded_fields))
                    event = _TestEvent(position=0, event_id_sha256=test_rows[0]["event_id_sha256"],
                        image_identity_sha256=test_rows[0]["image_identity_sha256"],
                        group_sha256=test_rows[0]["source_group_sha256"], camera_id=None,
                        feature_ref=object(), atomic_unit_sha256=test_rows[0]["atomic_unit_sha256"],
                        atomic_unit_kind="VISA_SINGLE_IMAGE")
                    receipt = {"schema": "final-benchmark.query-receipt.atomic-unit.v1",
                        "arm_id": "F-B", "category": "cashew", "condition_id": "P0",
                        "event_id_sha256": event.event_id_sha256,
                        "atomic_unit_sha256": event.group_sha256,
                        "atomic_unit_kind": event.atomic_unit_kind, "position": 0}
                    rows = [{"position": 0, "event_id_sha256": event.event_id_sha256,
                        "image_identity_sha256": event.image_identity_sha256,
                        "source_group_sha256": event.group_sha256,
                        "atomic_unit_sha256": event.atomic_unit_sha256,
                        "atomic_unit_kind": event.atomic_unit_kind,
                        "query_actions": {"F-B": {"executed": True,
                            "query_receipt_sha256": hashlib.sha256(canonical(receipt)).hexdigest()}}}]
                    group = {"atomic_unit_sha256": event.group_sha256,
                        "atomic_unit_kind": event.atomic_unit_kind,
                        "unit_candidate_sha256": candidate["candidate_sha256"],
                        "unit_approval_sha256": hashlib.sha256(canonical(approval)).hexdigest(),
                        "source_group_sha256": event.group_sha256,
                        "start_position": 0, "end_position_exclusive": 1,
                        "event_order_sha256": hashlib.sha256(canonical([event.event_id_sha256])).hexdigest(),
                        "score_rows_sha256": hashlib.sha256(canonical(rows)).hexdigest(),
                        "status": "SEALED_BEFORE_LABEL_DISCLOSURE"}
                    sink.append_unit(group_seal=group, score_rows=rows)
                    label, private = broker.disclose(arm_id="F-B", event=event, query_receipt=receipt)
                    self.assertEqual(label, 0)
                    self.assertEqual(private["source_csv_sha256"], csv_sha)
                    self.assertEqual(decoded_fields.count(True), 1)
                    self.assertEqual(len(broker.decoded_queries), 1)
                    broker.close()

    def test_source_bound_plan_keeps_requested_category_and_does_not_authorize_other_categories(self):
        from experiments.calibration.final_benchmark.visa_registration import CATEGORIES
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_commit = "a" * 40
            (root / "private").mkdir()
            (root / "private" / "researcher_decision_binding.json").write_bytes(canonical({
                "schema": "final-benchmark.visa-researcher-decision-binding.v1",
                "researcher_decision_sha256": sha("cashew decision"),
                "publisher_authentication": False, "independent_review_claimed": False,
                "pixel_mask_access": False}))
            category_items = []
            counts = {"candle": 200, "capsules": 160, "cashew": 150, "chewinggum": 150,
                "fryum": 150, "macaroni1": 200, "macaroni2": 200, "pcb1": 200,
                "pcb2": 200, "pcb3": 201, "pcb4": 201, "pipe_fryum": 150}
            for category in CATEGORIES:
                folder = root / category
                folder.mkdir()
                role = {"category": category}
                role_sha = hashlib.sha256(canonical(role)).hexdigest()
                candidate_payload = {"category": category}
                candidate = {**candidate_payload,
                             "candidate_sha256": hashlib.sha256(canonical(candidate_payload)).hexdigest()}
                stream_payload = {"category": category, "source_commit": source_commit,
                    "role_manifest_sha256": role_sha, "stream_order": [0]}
                stream = {**stream_payload,
                    "stream_manifest_sha256": hashlib.sha256(canonical(stream_payload)).hexdigest()}
                release = {"category": category, "source_commit": source_commit}
                for name, value in (("role.json", role), ("candidate.json", candidate),
                                    ("stream.json", stream), ("release.json", release)):
                    (folder / name).write_bytes(canonical(value))
                category_items.append({"category": category, "role_manifest_path": f"{category}/role.json",
                    "unit_candidate_path": f"{category}/candidate.json",
                    "stream_manifest_path": f"{category}/stream.json",
                    "release_identity_path": f"{category}/release.json",
                    "private_locator_path": f"private/{category}/locators.json"})
            index = {"schema": "final-benchmark.visa-registration-index.v1",
                "source_commit": source_commit, "total_test_images": 2162,
                "test_counts": counts, "categories": category_items,
                "dataset_release_sha256": sha("release"),
                "identity_manifest_sha256": sha("identity"),
                "source_artifacts_sha256": sha("sources")}
            index["registration_sha256"] = hashlib.sha256(canonical(index)).hexdigest()
            (root / "registration_index.json").write_bytes(canonical(index))
            with patch("experiments.calibration.final_benchmark.visa_execution._current_head",
                       return_value=source_commit):
                cashew = create_source_bound_plan(registration_root=root, category="cashew")
                candle = create_source_bound_plan(registration_root=root, category="candle",
                    researcher_decision_sha256=sha("separate candle decision"))
            cashew_plan = json.loads(Path(cashew["execution_plan_path"]).read_text())
            candle_plan = json.loads(Path(candle["execution_plan_path"]).read_text())
            self.assertEqual(cashew_plan["execution_scope"]["Stage_A"], ["cashew/P0"])
            self.assertEqual(candle_plan["execution_scope"]["Stage_A"], ["candle/P0"])
            self.assertEqual(cashew["authorization_status"], "RESEARCHER_APPROVED_VISA_ATOMIC_UNIT_EXECUTION")
            self.assertIsNone(candle["authorization_path"])
            self.assertEqual(candle["authorization_status"],
                             "MISSING_REQUIRES_SEPARATE_RESEARCHER_AUTHORIZATION")


if __name__ == "__main__":
    unittest.main()
