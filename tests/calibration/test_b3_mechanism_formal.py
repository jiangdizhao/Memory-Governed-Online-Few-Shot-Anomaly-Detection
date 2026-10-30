from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from experiments.calibration import stage1a_b3_mechanism_ablation as a1
from experiments.calibration import stage1a_b3_mechanism_formal as formal
from experiments.calibration.metrics import PrivateOutcomeLedger


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def history(q_values: list[float]) -> a1.VersionedC64History:
    group = digest("fixed protected C64 identity")
    refs = [a1.C64Reference.from_scores(
        version, digest(f"memory-{version}"), [q] * 64, group)
        for version, q in enumerate(q_values)]
    return a1.VersionedC64History(refs)


def event(position: int, before: int, after: int,
          score: float = 0.5) -> a1.TrajectoryEvent:
    return a1.TrajectoryEvent(
        position, score, before, digest(f"memory-{before}"),
        after, digest(f"memory-{after}"))


def prediction_row(trajectory_id: str, position: int = 0) -> dict:
    return {
        "schema": formal.PREDICTION_SCHEMA,
        "trajectory_id": trajectory_id,
        "event_position": position,
        "event_identity": {"position": position, "query_event": False},
        "score": 0.5,
        "memory_version_before": 0,
        "memory_sha256_before": digest("memory-0"),
        "memory_version_after": 0,
        "memory_sha256_after": digest("memory-0"),
        "arms": {
            arm: {"prediction": 1, "threshold_pre": 0.5,
                  "threshold_post": 0.5, "update": None}
            for arm in formal.ARMS
        },
    }


class TestB3MechanismFormal(unittest.TestCase):
    def test_matched_replay_has_identical_score_memory_and_update_positions(self):
        hist = history([0.5, 0.8, 0.1, 0.9, 0.2, 0.7])
        events = (event(0, 0, 0), event(1, 0, 1), event(2, 1, 1),
                  event(3, 1, 2), event(4, 2, 3), event(5, 3, 4),
                  event(6, 4, 5))
        result = a1.replay_matched_arms(events, hist)
        expected_positions = [row.position for row in events if row.memory_version_after != row.memory_version_before]
        for arm in formal.ARMS:
            self.assertEqual(result[arm].positions, result["A2"].positions)
            self.assertEqual(result[arm].memory_versions_before,
                             result["A2"].memory_versions_before)
            self.assertEqual(result[arm].memory_versions_after,
                             result["A2"].memory_versions_after)
            self.assertEqual([row.position for row in result[arm].updates], expected_positions)
            for update in result[arm].updates:
                self.assertLessEqual(update.reference_version, update.memory_version)
        self.assertEqual(dict(a1.ARMS), {"A2": 0, "A3": 1, "A4": 5})

    def test_transition_row_records_preupdate_prediction_and_postupdate_threshold(self):
        hist = history([0.5, 0.9])
        trajectory = a1.FrozenTrajectory(
            "audiojack", 0,
            (event(0, 0, 1, score=0.5),), hist,
            ({"t": 0, "sample_key": "synthetic-group", "camera_id": 0,
              "query_event": False,
              "methods": {"B3": {"prediction": 1, "threshold_pre": 0.5,
                                  "threshold_post": 0.51}}},),
            {"trajectory_id": "audiojack/seed0/test"})
        replay = a1.replay_matched_arms(trajectory.events, hist)
        row = formal._event_rows(trajectory, replay)[0]
        self.assertEqual(row["arms"]["A2"]["prediction"], 1)
        self.assertEqual(row["arms"]["A2"]["threshold_pre"], 0.5)
        self.assertEqual(row["arms"]["A2"]["threshold_post"], 0.51)
        self.assertEqual(row["arms"]["A3"]["update"]["reference_version"], 0)
        self.assertFalse(any(key.lower() in {"label", "true_label", "ground_truth", "outcome"}
                             for key in row))
        self.assertNotIn("labels", json.dumps(row).lower())

    def test_gzip_prediction_serialization_is_deterministic_for_resume(self):
        rows = [prediction_row("audiojack/seed0/test", 0),
                prediction_row("audiojack/seed0/test", 1)]
        first = formal._encode_jsonl_gzip(rows)
        second = formal._encode_jsonl_gzip(rows)
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl.gz"
            path.write_bytes(first)
            self.assertEqual(formal._read_prediction_rows(path, expected_count=2), rows)

    def _sealed_fixture(self, root: Path) -> tuple[Path, dict]:
        root.mkdir(parents=True, exist_ok=True)
        (root / "predictions").mkdir()
        receipts = []
        files = {}
        for category in formal.CLASSES:
            for seed in formal.SEEDS:
                trajectory_id = f"{category}/seed{seed}/synthetic"
                relative = f"predictions/{category}_seed{seed}.jsonl.gz"
                path = root / relative
                path.write_bytes(formal._encode_jsonl_gzip(
                    [prediction_row(trajectory_id)]))
                file_sha = formal._sha_file(path)
                files[relative] = file_sha
                receipts.append({
                    "class": category, "support_seed": seed,
                    "trajectory_id": trajectory_id, "event_count": 1,
                    "prediction_file": relative, "prediction_sha256": file_sha,
                })
        receipts_doc = {"trajectories": receipts}
        receipts_path = root / "prediction_receipts.json"
        receipts_path.write_text(json.dumps(receipts_doc), encoding="utf-8")
        seal = {
            "status": "PREDICTION_PHASE_COMPLETE_LABEL_FIREWALL_INTACT",
            "outcome_labels_accessed": False,
            "trajectory_count": len(receipts), "event_count": len(receipts),
            "prediction_files": files,
            "prediction_receipts_sha256": formal._sha_file(receipts_path),
        }
        (root / "prediction_phase_complete.json").write_text(
            json.dumps(seal), encoding="utf-8")
        return root, seal

    def test_seal_verifies_all_20_files_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self._sealed_fixture(Path(directory) / "run")
            seal, receipts = formal._verify_prediction_seal(
                root, expected_trajectories=20, expected_events=20)
            self.assertEqual(seal["trajectory_count"], 20)
            self.assertEqual(len(receipts["trajectories"]), 20)
            target = root / receipts["trajectories"][0]["prediction_file"]
            target.write_bytes(target.read_bytes() + b"tamper")
            with self.assertRaisesRegex(formal.FormalRunError, "failed its seal"):
                formal._verify_prediction_seal(
                    root, expected_trajectories=20, expected_events=20)

    def test_label_reader_is_not_called_without_a_valid_prediction_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "incomplete"
            root.mkdir()
            with mock.patch("experiments.calibration.stage05.test_labels_for_class") as reader:
                with self.assertRaises(formal.FormalRunError):
                    formal.run_evaluator(source_root=Path(directory), output=root,
                                         json_root=Path(directory))
                reader.assert_not_called()

    def test_one_class_denominators_remain_null_and_auc_ties_are_stable(self):
        metrics = PrivateOutcomeLedger([0, 0]).evaluate(
            [0.2, 0.2], {"A2": [0, 1]})
        method = metrics["methods"]["A2"]
        self.assertEqual(method["fpr"], 0.5)
        self.assertIsNone(method["fnr"])
        self.assertIsNone(method["balanced_accuracy"])
        self.assertEqual(method["balanced_accuracy_status"], "UNAVAILABLE_CLASS_ABSENT")
        self.assertEqual(metrics["ranking_status"], "UNAVAILABLE_CLASS_ABSENT")
        worst = formal._worst_complete_windows(method, scope="full_stream")
        self.assertIsNone(worst["balanced_accuracy"])
        self.assertEqual(worst["balanced_accuracy_status"],
                         "UNAVAILABLE_NO_COMPLETE_256_IMAGE_WINDOW")

    def test_score_ranking_does_not_depend_on_threshold_predictions(self):
        labels = [0, 1, 0, 1]
        scores = [0.1, 0.9, 0.2, 0.8]
        low_threshold = PrivateOutcomeLedger(labels).evaluate(
            scores, {"A2": [0, 1, 0, 1]})
        high_threshold = PrivateOutcomeLedger(labels).evaluate(
            scores, {"A2": [0, 0, 0, 0]})
        self.assertEqual(low_threshold["shared_score_auroc"], 1.0)
        self.assertEqual(low_threshold["shared_score_auroc"],
                         high_threshold["shared_score_auroc"])
        self.assertEqual(low_threshold["shared_score_auprc"],
                         high_threshold["shared_score_auprc"])

    def test_output_namespace_is_versioned_instead_of_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            base = repository / "outputs" / "run"
            base.mkdir(parents=True)
            selected = formal._select_output(Path("outputs/run"), resume=False,
                                             repository=repository)
            self.assertEqual(selected.name, "run_v2")
            self.assertTrue(base.is_dir())

    def test_zero_operational_divergence_is_reported_as_inconclusive(self):
        diagnostics = {
            "prediction_disagreement_events": {"A2-A3": 0, "A2-A4": 0},
            "threshold_equal_after_transition_counts": {
                "A2-A3": {"count": 1190, "total_transitions": 1190, "fraction": 1.0},
                "A2-A4": {"count": 1190, "total_transitions": 1190, "fraction": 1.0},
            },
        }
        result = formal._classify_h1({}, diagnostics)
        self.assertEqual(result["classification"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
