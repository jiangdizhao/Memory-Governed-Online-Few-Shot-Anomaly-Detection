"""CPU-only contract tests for the formal Stage 1-A result adapter."""
from __future__ import annotations

import unittest
import hashlib

from experiments.calibration.metrics import PrivateOutcomeLedger
from experiments.calibration.stage1a_formal_comparison import (
    ComparisonError,
    _bind_query_source_identity,
    _enrich_method_metrics,
    _summarize_threshold_dynamics,
)


class FormalComparisonMetricTests(unittest.TestCase):
    def test_query_source_identity_reconstruction_matches_frozen_group_binding(self):
        source_key = "fixture-official-sample-key"
        expected_group = hashlib.sha256(
            ("official-sample-group\0" + source_key).encode("utf-8")).hexdigest()
        expected_source_identity = hashlib.sha256(source_key.encode("utf-8")).hexdigest()
        self.assertEqual(_bind_query_source_identity(source_key, expected_group),
                         expected_source_identity)
        with self.assertRaises(ComparisonError):
            _bind_query_source_identity(source_key, "0" * 64)

    def test_threshold_dynamics_accepts_prediction_objects_as_float_sequences(self):
        result = _summarize_threshold_dynamics(
            (None, 1.0, 1.0, 1.02), (None, 1.0, 1.01, 1.02))
        self.assertEqual(result["events"], 4)
        self.assertEqual(result["pre_available_events"], 3)
        self.assertEqual(result["threshold_change_events"], 1)
        self.assertAlmostEqual(result["absolute_threshold_movement"], 0.01)
        with self.assertRaises(ComparisonError):
            _summarize_threshold_dynamics((1.0,), (1.0, 1.1))

    def test_source_native_startup_missing_keeps_full_ba_undefined(self):
        labels = (0, 1, 0, 1)
        predictions = (0, None, 0, 1)
        availability = (True, False, True, True)
        reasons = ("SPOT_STRICT_UPPER_TAIL", "WARMUP_UNAVAILABLE",
                   "SPOT_STRICT_UPPER_TAIL", "SPOT_STRICT_UPPER_TAIL")
        base = PrivateOutcomeLedger(labels).evaluate(
            (0.1, 0.9, 0.2, 0.8), {"B5a": predictions},
            prediction_availability={"B5a": availability})["methods"]["B5a"]
        row = _enrich_method_metrics(base, labels, predictions,
                                     availability, reasons, "B5a")
        self.assertIsNone(row["BA"])
        self.assertEqual(row["balanced_accuracy_status"],
                         "UNAVAILABLE_MISSING_PREDICTIONS")
        self.assertEqual(row["startup_unavailable_count"], 1)
        self.assertEqual(row["available_abstention_count"], 0)
        self.assertEqual(row["coverage"], 0.75)

    def test_frozen_fallback_reports_native_and_fallback_decision_fractions(self):
        labels = (0, 1, 0, 1)
        predictions = (0, 1, 0, 1)
        availability = (True,) * 4
        reasons = ("STARTUP_FALLBACK", "ORIGINAL_METHOD",
                   "ABSTENTION_FALLBACK", "ORIGINAL_METHOD")
        base = PrivateOutcomeLedger(labels).evaluate(
            (0.1, 0.9, 0.2, 0.8), {"fallback": predictions})["methods"]["fallback"]
        row = _enrich_method_metrics(base, labels, predictions,
                                     availability, reasons, "B5a_FROZEN_FALLBACK")
        self.assertEqual(row["source_native_decisions"], 2)
        self.assertEqual(row["source_native_decision_fraction"], 0.5)
        self.assertEqual(row["fallback_fraction"], 0.5)
        self.assertEqual(row["original_method_decision_fraction"], 0.5)
        self.assertEqual(row["startup_fallback_fraction"], 0.25)
        self.assertEqual(row["abstention_fallback_fraction"], 0.25)

    def test_coverage_is_reported_for_each_true_class(self):
        labels = (0, 0, 1, 1)
        predictions = (0, None, 1, None)
        availability = (True, True, True, True)
        reasons = ("", "ABSTAIN", "", "ABSTAIN")
        base = PrivateOutcomeLedger(labels).evaluate(
            (0.1, 0.2, 0.8, 0.9), {"method": predictions})["methods"]["method"]
        row = _enrich_method_metrics(base, labels, predictions,
                                     availability, reasons, "B0")
        self.assertEqual(row["coverage"], 0.5)
        self.assertEqual(row["normal_coverage"], 0.5)
        self.assertEqual(row["defect_coverage"], 0.5)
        self.assertEqual(row["selective_balanced_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
