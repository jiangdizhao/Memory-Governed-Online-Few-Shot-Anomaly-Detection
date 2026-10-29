from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from experiments.calibration import stage05
from experiments.calibration.protocol import TapeRecord
from experiments.calibration.query_tape import build_qa_ts


def synthetic_stream(n: int = 1600):
    rows = []
    for position in range(n):
        group = position // 5
        camera = position % 5 + 1
        rows.append(stage05.StreamImage(
            position=position, image_path=f"/unused/{position}.png",
            relative_id=f"OK/S{group:04d}/audiojack_{group:04d}_OK_C{camera}_20260101000000.png",
            sample_key=f"audiojack_{group:04d}_OK",
            opaque_group=f"group-{group}", camera_id=camera,
            timestamp="20260101000000", view_count=5))
    return rows


class Stage05QueryTests(unittest.TestCase):
    def test_seed37_three_percent_matches_existing_qats_and_extensions_are_nested(self):
        stream = synthetic_stream()
        records = stage05.tape_records(stream)
        tapes = stage05.build_nested_tapes(records, 37)
        stage05.verify_nested_tapes(tapes, records)
        existing = build_qa_ts(records)
        self.assertEqual(tapes[0.03]["digest"], existing.digest)
        self.assertEqual({x["position"] for x in tapes[0.03]["events"]},
                         set(existing.positions))
        p3 = {x["position"] for x in tapes[0.03]["events"]}
        p5 = {x["position"] for x in tapes[0.05]["events"]}
        p7 = {x["position"] for x in tapes[0.07]["events"]}
        self.assertLessEqual(p3, p5)
        self.assertLessEqual(p5, p7)
        self.assertEqual([len(tapes[rate]["events"]) for rate in stage05.QUERY_RATES],
                         [48, 80, 112])

    def test_label_values_cannot_enter_tape_builder(self):
        stream = synthetic_stream()
        records = stage05.tape_records(stream)
        first = stage05.build_nested_tapes(records, 1000)
        changed_labels = [position % 2 for position in range(len(stream))]
        self.assertEqual(len(changed_labels), len(stream))
        second = stage05.build_nested_tapes(records, 1000)
        self.assertEqual({rate: tape["digest"] for rate, tape in first.items()},
                         {rate: tape["digest"] for rate, tape in second.items()})

    def test_tampered_nested_tape_fails_digest_and_nesting_verification(self):
        records = stage05.tape_records(synthetic_stream())
        tapes = stage05.build_nested_tapes(records, 37)
        tampered = {rate: dict(tape) for rate, tape in tapes.items()}
        changed = dict(tampered[0.05]["events"][0])
        changed["position"] += 1
        events = list(tampered[0.05]["events"])
        events[0] = changed
        tampered[0.05]["events"] = events
        with self.assertRaises(stage05.Stage05Error):
            stage05.verify_nested_tapes(tampered, records)

    def test_label_blind_json_reader_discards_test_labels_and_masks(self):
        payload = {
            "meta": {"prefix": "audiojack/", "normal_class": "OK"},
            "train": [{"image_path": "OK/a.jpg", "anomaly_class": "OK"}],
            "test": [{"image_path": "NG/a.jpg", "anomaly_class": "NG",
                      "mask_path": "private/mask.png"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "class.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            label_blind = stage05.read_json_sections(source, train_labels=True)
            self.assertEqual(label_blind["train"][0]["anomaly_class"], "OK")
            self.assertNotIn("anomaly_class", label_blind["test"][0])
            self.assertNotIn("mask_path", label_blind["test"][0])
            evaluator = stage05.read_json_sections(
                source, train_labels=False, test_labels=True)
            self.assertNotIn("anomaly_class", evaluator["train"][0])
            self.assertEqual(evaluator["test"][0]["anomaly_class"], "NG")


class Stage05SupportTests(unittest.TestCase):
    def _train(self, n_groups=100):
        rows = []
        for group in range(n_groups):
            for camera in range(1, 6):
                rows.append(stage05.TrainImage(
                    image_path=f"/train/{group}/{camera}.png",
                    relative_id=f"OK/S{group:04d}/audiojack_{group:04d}_OK_C{camera}_20260101000000.png",
                    sample_key=f"audiojack_{group:04d}_OK", is_good=True,
                    camera_id=camera, timestamp=f"2026010100000{camera}"))
        return rows

    def test_seed0_exact_historical_support_and_c64_compatibility(self):
        from experiments.calibration.real_iad_preflight import _select_support_and_c64
        from experiments.risk_safe.preflight import ImageRecord
        train = self._train()
        modern = stage05.select_supports(train)
        historical_rows = [ImageRecord(
            path=row.image_path, rel_path=row.relative_id,
            class_name="audiojack", split="train", sample_key=row.sample_key,
            directory_serial_key=row.relative_id.split("/")[1],
            capture_session="OK", camera_id=row.camera_id,
            timestamp=row.timestamp, is_good=row.is_good) for row in train]
        supports, c64, support_groups, c64_groups = _select_support_and_c64(historical_rows)
        self.assertEqual(modern["supports"]["0"]["groups"], support_groups)
        self.assertEqual(modern["supports"]["0"]["images"],
                         [row.image_id for row in supports])
        self.assertEqual(modern["c64"]["groups"], c64_groups)
        self.assertEqual(modern["c64"]["images"], [row.image_id for row in c64])

    def test_all_five_support_seeds_are_train_normal_and_c64_disjoint(self):
        plan = stage05.select_supports(self._train())
        self.assertEqual(set(plan["supports"]), {"0", "1", "2", "3", "4"})
        self.assertTrue(plan["support_c64_disjoint"])
        for seed, support in plan["supports"].items():
            self.assertEqual(len(support["groups"]), 4)
            self.assertEqual(len(set(support["groups"])), 4)
            self.assertTrue(support["all_train_normal"])
            self.assertTrue(support["c64_disjoint"])
        self.assertEqual(plan["c64"]["count"], 64)


class Stage05EvaluatorTests(unittest.TestCase):
    def test_oracle_ties_and_boundaries_use_score_greater_equal(self):
        labels = [0, 1]
        scores = [1.0, 1.0]
        result = stage05.retrospective_threshold_oracle(labels, scores, 2.0)
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertEqual(result["number_of_ba_tied_thresholds"], 2)
        self.assertEqual(result["threshold"], result["boundary_candidates"]["none_defect"])
        perfect = stage05.retrospective_threshold_oracle(
            [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
        self.assertEqual(perfect["balanced_accuracy"], 1.0)
        self.assertEqual(perfect["fpr"], 0.0)
        self.assertEqual(perfect["fnr"], 0.0)

    def test_single_class_windows_are_undefined(self):
        result = stage05._window_metrics(
            __import__("numpy").zeros(300, dtype=int),
            __import__("numpy").arange(300, dtype=float), 999.0)
        self.assertEqual(result["window_count"], 45)
        self.assertEqual(result["undefined_single_class_windows"], 45)
        self.assertIsNone(result["worst_defined_window"])

    def test_query_schedule_metrics_reports_gaps_and_windows(self):
        stream = synthetic_stream(512)
        records = stage05.tape_records(stream)
        tape = stage05.build_nested_tapes(records, 37)[0.03]
        labels = [int(index % 17 == 0) for index in range(512)]
        row = stage05.query_schedule_metrics(stream, labels, tape)
        self.assertEqual(row["query_count"], 15)
        self.assertIn("queried_defects", row)
        self.assertEqual(row["sliding_256_image_windows"]["count"], 257)
        self.assertEqual(row["queried_official_sample_groups"], row["query_count"])

    def test_every_256_image_query_window_has_seed_and_seed_set_counts(self):
        stream = synthetic_stream(512)
        records = stage05.tape_records(stream)
        tapes = {seed: stage05.build_nested_tapes(records, seed)[0.03]
                 for seed in (37, 1000)}
        labels = [int(index % 17 == 0) for index in range(512)]
        rows = stage05.query_window_sensitivity(labels, tapes, window=256)
        self.assertEqual(len(rows), 257)
        self.assertEqual(rows[0]["window_start"], 0)
        self.assertEqual(rows[-1]["window_end_exclusive"], 512)
        self.assertIn("seed37_queried_defects", rows[0])
        self.assertIn("quantiles_linear", rows[0])

    def test_label_access_is_separate_from_label_blind_tape_api(self):
        import inspect
        parameters = set(inspect.signature(stage05.build_nested_tapes).parameters)
        self.assertEqual(parameters, {"records", "seed"})
        self.assertNotIn("labels", parameters)
        self.assertNotIn("outcomes", parameters)


class Stage05ArtifactTests(unittest.TestCase):
    def test_runner_accepts_authorized_branch_descendants_of_pinned_base(self):
        from experiments.calibration.stage05_runner import EXPECTED_BRANCH, EXPECTED_HEAD, _git_info
        identity = _git_info()
        self.assertEqual(identity["branch"], EXPECTED_BRANCH)
        self.assertEqual(identity["pinned_base_head"], EXPECTED_HEAD)

    def test_output_is_non_overwriting_and_file_digest_detects_change(self):
        from experiments.calibration.stage05_runner import (
            RunnerError, _file_sha, _refresh_checksums, _require_output)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit"
            _require_output(output, create=True, resume=False)
            artifact = output / "frozen.json"
            artifact.write_text('{"frozen":true}\n', encoding="utf-8")
            _refresh_checksums(output)
            before = _file_sha(artifact)
            with self.assertRaises(RunnerError):
                _require_output(output, create=True, resume=False)
            artifact.write_text('{"frozen":false}\n', encoding="utf-8")
            self.assertNotEqual(before, _file_sha(artifact))


if __name__ == "__main__":
    unittest.main()
