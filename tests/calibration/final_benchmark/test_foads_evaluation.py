from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.calibration.final_benchmark import foads_evaluation as evaluation
from experiments.calibration.final_benchmark.execution_support import canonical_json, sha256_file
from experiments.calibration.final_benchmark.identity import (
    _domain_hash, _image_identity_digest, _source_group_digest,
)
from experiments.calibration.final_benchmark.launch import object_sha256


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"


def _synthetic_attempt(root: Path):
    root.mkdir()
    category = "audiojack"
    release_sha = "1" * 64
    stream_seed = 0
    # Original annotation order is the opposite of the registered stream order.
    group_specs = [(2, "NG", 0), (1, "OK", 5)]
    annotation_rows = [None] * 10
    events = []
    for group_number, session, first_position in group_specs:
        group_key = f"audiojack_{group_number:04d}_{session}"
        group_sha = _source_group_digest(group_key)
        for camera in range(1, 6):
            original_position = first_position + camera - 1
            image_key = f"{group_key}_C{camera}_20260101000000.jpg"
            anomaly_class = "good" if original_position % 2 == 0 else "scratch"
            annotation_rows[original_position] = {
                "image_path": image_key,
                "category": category,
                "anomaly_class": anomaly_class,
            }
            events.append({
                "position": len(events),
                "original_position": original_position,
                "event_id_sha256": _domain_hash(
                    "Gate0-event-v1", release_sha, category, str(stream_seed),
                    group_key, str(camera), image_key, str(camera)),
                "image_identity_sha256": _image_identity_digest(image_key),
                "image_content_sha256": "e" * 64,
                "source_group_sha256": group_sha,
                "camera_id": camera,
                "within_group_ordinal": camera,
            })
    annotation = {"meta": {"category": category, "dataset": "Real-IAD", "normal_class": "good"},
                  "train": [], "test": annotation_rows}
    annotation_bytes = _json_bytes(annotation)
    annotation_sha = hashlib.sha256(annotation_bytes).hexdigest()
    role = {
        "category": category, "condition_id": "P0", "dataset_id": "realiad",
        "stream_seed": stream_seed, "support_seed": 0,
        "annotation_sha256": annotation_sha,
        "selected": ([{"role": "SUPPORT", "id": f"s{i}"} for i in range(4)] +
                     [{"role": "C64", "id": f"c{i}"} for i in range(64)]),
    }
    stream = {"category": category, "condition_id": "P0", "dataset_id": "realiad",
              "stream_seed": stream_seed, "event_count": len(events), "events": events}
    bundle = {
        "schema": "final-benchmark.category-condition-identity.v1",
        "dataset_id": "realiad", "category": category, "condition_id": "P0",
        "support_seed": 0, "stream_seed": stream_seed,
        "release_identity": {"dataset_release_sha256": release_sha},
        "role_manifest": role, "stream_manifest": stream,
    }
    bundle_path = root / "category_bundle.json"
    bundle_path.write_bytes(_json_bytes(bundle))
    bundle_sha = sha256_file(bundle_path)
    role_sha, stream_sha = object_sha256(role), object_sha256(stream)
    scores = []
    for event in events:
        scores.append({
            "position": event["position"],
            "event_id_sha256": event["event_id_sha256"],
            "image_identity_sha256": event["image_identity_sha256"],
            "source_group_sha256": event["source_group_sha256"],
            "raw_image_score": float(event["position"]),
            "model_update_index": event["position"] // 10,
            "model_state_before_score_sha256": "a" * 64,
        })
    updates = [{
        "schema": "final-benchmark.foads-native-update.v1",
        "update_index": 1, "start_position": 0, "end_position_exclusive": 10,
        "event_identity_sha256s": [event["image_identity_sha256"] for event in events],
        "source_group_sha256s": [events[0]["source_group_sha256"], events[5]["source_group_sha256"]],
        "model_state_after_update_sha256": "b" * 64,
    }]
    (root / "foads_prediction_journal.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scores), encoding="utf-8")
    (root / "foads_update_journal.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in updates), encoding="utf-8")
    identity = {
        "category": category, "condition": "P0", "campaign_run_id": "synthetic-run",
        "source_commit": "a" * 40, "dataset_release_sha256": release_sha,
        "category_bundle_file_sha256": bundle_sha,
        "role_manifest_sha256": role_sha, "stream_manifest_sha256": stream_sha,
        "foads_registration_projection_sha256": "d" * 64,
    }
    seal = {
        "schema": "final-benchmark.foads-category-seal.v1",
        "status": "SEALED_COMPLETE",
        "baseline": evaluation.FOADS_BASELINE,
        "identity": identity,
        "expected_event_count": 10, "scored_images": 10,
        "source_group_count": 2, "completed_batch10_updates": 1,
        "final_unupdated_groups": 0, "paid_labels": 0,
        "test_outcome_or_mask_access": False,
        "prediction_journal_sha256": sha256_file(root / "foads_prediction_journal.jsonl"),
        "update_journal_sha256": sha256_file(root / "foads_update_journal.jsonl"),
        "final_model_state_sha256": "b" * 64,
    }
    seal["seal_sha256"] = hashlib.sha256(canonical_json(seal)).hexdigest()
    seal_path = root / "foads_category_seal.json"
    seal_path.write_bytes(_json_bytes(seal))
    status = {
        "schema": "final-benchmark.attempt-status.v1", "execution_kind": "foads_kng_p0",
        "status": "SEALED_COMPLETE", "identity": identity,
        "completed_event_count": 10, "expected_event_count": 10,
        "completion_seal_path": "foads_category_seal.json",
        "completion_seal_sha256": sha256_file(seal_path),
    }
    (root / "attempt_status.json").write_bytes(_json_bytes(status))
    authorization = {
        "schema": evaluation.FOADS_STAGE_B_AUTH_SCHEMA,
        "status": "AUTHORIZED_POST_STAGE_A_LABEL_PROJECTION",
        "scope": "stage_b_category_evaluation", "dataset_id": "realiad",
        "category": category, "condition_id": "P0", "campaign_run_id": "synthetic-run",
        "stage_a_source_commit": "a" * 40,
        "stage_a_seal_sha256": sha256_file(seal_path),
        "category_bundle_sha256": bundle_sha,
        "dataset_release_sha256": release_sha,
        "role_manifest_sha256": role_sha, "stream_manifest_sha256": stream_sha,
        "foads_registration_projection_sha256": "d" * 64,
        "prediction_journal_sha256": seal["prediction_journal_sha256"],
        "update_journal_sha256": seal["update_journal_sha256"],
        "source_sha256": annotation_sha,
        "test_outcomes_after_valid_seal": True,
        "test_masks_authorized": False,
        "online_outcome_access": False,
        "authorization_basis": "RESEARCHER_EXPLICIT_SYNTHETIC_TEST_ONLY",
    }
    annotation_path = root / "official_annotation.json"
    annotation_path.write_bytes(annotation_bytes)
    return {
        "attempt_root": root, "bundle_path": bundle_path,
        "annotation_path": annotation_path, "authorization": authorization,
        "annotation": annotation, "events": events, "identity": identity, "seal": seal,
    }


class FOADSEvaluationTests(unittest.TestCase):
    def test_synthetic_projection_uses_original_positions_and_historical_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _synthetic_attempt(Path(tmp) / "attempt")
            output = Path(tmp) / "stage_b"
            result = evaluation.evaluate_foads_sealed_category(
                attempt_root=fixture["attempt_root"], category_bundle_path=fixture["bundle_path"],
                annotation_path=fixture["annotation_path"],
                stage_b_authorization=fixture["authorization"], output_directory=output)
            summary = result["summary"]
            self.assertEqual(summary["stage_a_source_commit"], "a" * 40)
            self.assertEqual(summary["outcome_access_phase"], "STAGE_B_AFTER_VALID_FOADS_SEAL")
            self.assertFalse(summary["test_masks_read"])
            self.assertEqual(summary["scored_event_count"], 10)
            self.assertEqual(summary["positive_count"], 5)
            self.assertEqual(summary["negative_count"], 5)
            self.assertEqual(summary["metrics"]["crosscheck"], "PASS")

    def test_incomplete_attempt_is_rejected_before_annotation_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _synthetic_attempt(Path(tmp) / "attempt")
            status_path = fixture["attempt_root"] / "attempt_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["status"] = "INCOMPLETE"
            status_path.write_bytes(_json_bytes(status))
            with patch.object(evaluation, "_read_official_annotation_bytes",
                              side_effect=AssertionError("annotation must not be opened")):
                with self.assertRaises(evaluation.FOADSEvaluationError):
                    evaluation.evaluate_foads_sealed_category(
                        attempt_root=fixture["attempt_root"], category_bundle_path=fixture["bundle_path"],
                        annotation_path=fixture["annotation_path"],
                        stage_b_authorization=fixture["authorization"],
                        output_directory=Path(tmp) / "stage_b")

    def test_tampered_journal_is_rejected_before_annotation_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _synthetic_attempt(Path(tmp) / "attempt")
            with (fixture["attempt_root"] / "foads_prediction_journal.jsonl").open("a", encoding="utf-8") as out:
                out.write("{}\n")
            with patch.object(evaluation, "_read_official_annotation_bytes",
                              side_effect=AssertionError("annotation must not be opened")):
                with self.assertRaises(evaluation.FOADSEvaluationError):
                    evaluation.evaluate_foads_sealed_category(
                        attempt_root=fixture["attempt_root"], category_bundle_path=fixture["bundle_path"],
                        annotation_path=fixture["annotation_path"],
                        stage_b_authorization=fixture["authorization"],
                        output_directory=Path(tmp) / "stage_b")

    def test_bad_authorization_annotation_digest_is_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _synthetic_attempt(Path(tmp) / "attempt")
            authorization = dict(fixture["authorization"], source_sha256="0" * 64)
            with patch.object(evaluation, "_read_official_annotation_bytes",
                              side_effect=AssertionError("annotation must not be opened")):
                with self.assertRaises(evaluation.FOADSEvaluationError):
                    evaluation.evaluate_foads_sealed_category(
                        attempt_root=fixture["attempt_root"], category_bundle_path=fixture["bundle_path"],
                        annotation_path=fixture["annotation_path"], stage_b_authorization=authorization,
                        output_directory=Path(tmp) / "stage_b")

    def test_registered_path_mismatch_and_duplicate_original_position_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _synthetic_attempt(Path(tmp) / "attempt")
            raw = fixture["annotation_path"].read_bytes()
            annotation = json.loads(raw)
            annotation["test"][0]["image_path"] = "audiojack_0099_NG_C1_20260101000000.jpg"
            changed = _json_bytes(annotation)
            with self.assertRaises(evaluation.FOADSEvaluationError):
                evaluation._project_labels(changed, expected_sha256=hashlib.sha256(changed).hexdigest(),
                    category="audiojack", dataset_release_sha256="1" * 64,
                    stream_seed=0, events=fixture["events"])
            duplicated = [dict(event) for event in fixture["events"]]
            duplicated[1]["original_position"] = duplicated[0]["original_position"]
            with self.assertRaises(evaluation.FOADSEvaluationError):
                evaluation._project_labels(raw, expected_sha256=hashlib.sha256(raw).hexdigest(),
                    category="audiojack", dataset_release_sha256="1" * 64,
                    stream_seed=0, events=duplicated)

    def test_missing_alignment_bad_digest_and_reordered_registration_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _synthetic_attempt(Path(tmp) / "attempt")
            raw = fixture["annotation_path"].read_bytes()
            with self.assertRaises(evaluation.FOADSEvaluationError):
                evaluation._project_labels(raw, expected_sha256="0" * 64,
                    category="audiojack", dataset_release_sha256="1" * 64,
                    stream_seed=0, events=fixture["events"])
            with self.assertRaises(evaluation.FOADSEvaluationError):
                evaluation._project_labels(raw, expected_sha256=hashlib.sha256(raw).hexdigest(),
                    category="audiojack", dataset_release_sha256="1" * 64,
                    stream_seed=0, events=fixture["events"][:-1])
            bundle = json.loads(fixture["bundle_path"].read_text(encoding="utf-8"))
            bundle["stream_manifest"]["events"][0:2] = reversed(bundle["stream_manifest"]["events"][0:2])
            fixture["bundle_path"].write_bytes(_json_bytes(bundle))
            identity = dict(fixture["identity"])
            identity["category_bundle_file_sha256"] = sha256_file(fixture["bundle_path"])
            identity["stream_manifest_sha256"] = object_sha256(bundle["stream_manifest"])
            with self.assertRaises(evaluation.FOADSEvaluationError):
                evaluation._verify_category_registration(
                    fixture["attempt_root"], identity, fixture["seal"], fixture["bundle_path"])

    def test_single_class_ranking_is_explicitly_undefined(self):
        result = evaluation._calculate_ranking_metrics([0, 0, 0], [1.0, 2.0, 3.0])
        self.assertEqual(result["status"], "UNDEFINED_SINGLE_CLASS")
        self.assertIsNone(result["auroc"])
        self.assertIsNone(result["average_precision"])

    def test_aggregator_requires_all_thirty_valid_p0_results(self):
        categories = [f"cat_{i:02d}" for i in range(30)]
        registrations = {category: {
            "test_event_count": 5,
            "role_manifest_sha256": f"{i + 1:064x}",
            "stream_manifest_sha256": f"{i + 31:064x}",
            "foads_p0_registration_projection_sha256": f"{i + 61:064x}",
        } for i, category in enumerate(categories)}
        with tempfile.TemporaryDirectory() as tmp:
            summaries = []
            for i, category in enumerate(categories):
                directory = Path(tmp) / category
                directory.mkdir()
                summary = {
                    "schema": evaluation.FOADS_STAGE_B_RESULT_SCHEMA,
                    "evaluation_status": "EVALUATION_COMPLETE", "dataset_id": "realiad",
                    "category": category, "condition_id": "P0", "baseline": evaluation.FOADS_BASELINE,
                    "outcome_access_phase": "STAGE_B_AFTER_VALID_FOADS_SEAL", "test_masks_read": False,
                    "scored_event_count": 5, "role_manifest_sha256": registrations[category]["role_manifest_sha256"],
                    "stream_manifest_sha256": registrations[category]["stream_manifest_sha256"],
                    "foads_registration_projection_sha256": registrations[category]["foads_p0_registration_projection_sha256"],
                    "metric_status": "DEFINED", "metrics": {"auroc": i / 30, "average_precision": (30 - i) / 30},
                    "stage_a_source_commit": "a" * 40, "stage_a_seal_sha256": "b" * 64,
                    "annotation_sha256": "c" * 64, "post_seal_authorization_sha256": "d" * 64,
                }
                summary_bytes = _json_bytes(summary)
                report_bytes = b"synthetic aggregate fixture\n"
                (directory / "evaluation_summary.json").write_bytes(summary_bytes)
                (directory / "EVALUATION_REPORT.md").write_bytes(report_bytes)
                (directory / "evaluation_receipt.json").write_bytes(_json_bytes({
                    "schema": evaluation.FOADS_STAGE_B_RECEIPT_SCHEMA,
                    "category": category, "condition_id": "P0", "stage_a_seal_sha256": "b" * 64,
                    "annotation_sha256": "c" * 64,
                    "evaluation_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
                    "evaluation_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                }))
                summaries.append(directory / "evaluation_summary.json")
            with patch.object(evaluation, "_registered_p0_categories", return_value=registrations):
                incomplete = evaluation.aggregate_foads_stage_b(
                    config_path="unused", evaluation_summaries=summaries[:-1],
                    output_directory=Path(tmp) / "macro_incomplete")
                self.assertFalse(incomplete["complete_30_category_benchmark"])
                self.assertEqual(incomplete["valid_category_count"], 29)
                complete = evaluation.aggregate_foads_stage_b(
                    config_path="unused", evaluation_summaries=summaries,
                    output_directory=Path(tmp) / "macro_complete")
                self.assertTrue(complete["complete_30_category_benchmark"])
                self.assertAlmostEqual(complete["macro_metrics"]["category_equal_macro_auroc"], 14.5 / 30)
                self.assertAlmostEqual(complete["macro_metrics"]["category_equal_macro_average_precision"], 15.5 / 30)


if __name__ == "__main__":
    unittest.main()
