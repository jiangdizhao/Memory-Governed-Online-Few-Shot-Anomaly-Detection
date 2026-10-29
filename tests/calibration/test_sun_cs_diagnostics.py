from __future__ import annotations

import unittest

from experiments.calibration.sun_cs import (
    EMPIRICAL_EXTREMA_SENSITIVITY_POLICY, PRIMARY_ENDPOINT_POLICY,
)
from experiments.calibration.sun_cs_diagnostics import (
    b7_query_budget_feasibility, clopper_pearson_zero_error_minimum,
    endpoint_identifiability_table, first_unclipped_upper_level_n,
    make_stream, query_budgets_from_metadata_preflight, run_policy_diagnostic,
)


class SunCSDiagnosticsTests(unittest.TestCase):
    def test_registered_finite_stream_endpoints_are_exact_for_unit_ramp(self):
        rows = endpoint_identifiability_table()
        self.assertEqual([row["n"] for row in rows], [64, 256, 1050, 3815])
        self.assertEqual([row["upper_requested_level"] for row in rows],
                         [1.0, 1.0, 1.0, 1.0])
        self.assertEqual([row["lower_order_statistic_midpoint_for_sample_1_to_n"]
                          for row in rows], [17.5, 161.5, 851.5, 3415.5])
        for row in rows:
            n = row["n"]
            self.assertEqual(row["empirical_extrema_interval_for_sample_1_to_n"],
                             [row["lower_order_statistic_midpoint_for_sample_1_to_n"], n])
            self.assertEqual(
                row["conservative_extended_real_interval_for_sample_1_to_n"],
                [row["lower_order_statistic_midpoint_for_sample_1_to_n"], "+infinity"])
            self.assertFalse(row["finite_score_anomaly_possible_under_primary_endpoint"])
        horizon = first_unclipped_upper_level_n()
        self.assertEqual(horizon["first_prefix_n_with_unclipped_upper_level"], 353018)
        self.assertEqual(horizon["upper_level_at_n_minus_1"], 1.0)
        self.assertLess(horizon["upper_level_at_n"], 1.0)

    def test_b7_zero_error_cp_minimum_is_fourteen_not_eight(self):
        row = clopper_pearson_zero_error_minimum()
        self.assertEqual(row["minimum_zero_error_n"], 14)
        self.assertGreater(row["bound_at_n_minus_1"], .20)
        self.assertLessEqual(row["bound_at_n"], .20)
        self.assertFalse(row["eight_defects_satisfy_descriptive_target"])
        self.assertFalse(row["online_guarantee_claimed"])

    def test_metadata_query_budgets_and_hypothetical_defect_fractions(self):
        from pathlib import Path
        preflight = (Path(__file__).resolve().parents[2] / "docs" / "calibration" /
                     "audit" / "stage0_sixthpass_final_v1" /
                     "real_iad_metadata_preflight.json")
        metadata = query_budgets_from_metadata_preflight(preflight)
        self.assertEqual(metadata["per_class_query_budgets"], {
            "audiojack": 114, "regulator": 116,
            "switch": 115, "rolled_strip_base": 114,
        })
        self.assertEqual(metadata["total_images"], 15295)
        self.assertEqual(metadata["total_official_sample_groups"], 3059)
        self.assertEqual(metadata["total_queries"], 459)
        self.assertAlmostEqual(metadata["total_image_query_fraction"], 459 / 15295)
        self.assertAlmostEqual(metadata["total_official_sample_group_query_fraction"],
                               459 / 3059)
        self.assertFalse(metadata["labels_or_hidden_prevalence_read"])
        table = b7_query_budget_feasibility(metadata["per_class_query_budgets"])
        self.assertEqual([r["hypothetical_queried_defect_fraction"]
                          for r in table["results"]], [.05, .1, .2, .3])
        self.assertEqual(table["total_query_budget"], 459)
        for key in ("expected_development_classes_reaching_8",
                    "expected_development_classes_reaching_14"):
            values = [row[key] for row in table["results"]]
            self.assertEqual(values, sorted(values))
        for row in table["results"]:
            self.assertTrue(row["independent_bernoulli_assumption"])
            self.assertEqual(set(row["classes"]), set(metadata["per_class_query_budgets"]))
            for class_row in row["classes"].values():
                self.assertAlmostEqual(
                    class_row["probability_below_8_and_remain_normal_only_hold"],
                    1.0 - class_row["probability_queried_defects_at_least_8"])

    def test_seeded_continuous_streams_are_reproducible_and_distinct(self):
        for name in ("stationary_uniform", "stationary_lognormal",
                     "piecewise_uniform_shift"):
            first = make_stream(name, 256)
            second = make_stream(name, 256)
            self.assertEqual(first, second)
            self.assertEqual(len(first.values), 256)
            self.assertEqual(len(set(first.values)), 256)
        piecewise = make_stream("piecewise_uniform_shift", 256)
        self.assertEqual(piecewise.shift_event, 128)
        self.assertTrue(all(x < 1.0 for x in piecewise.values[:128]))
        self.assertTrue(all(2.0 < x < 3.0 for x in piecewise.values[128:]))

    def test_small_diagnostic_keeps_online_and_offline_information_separate(self):
        stream = make_stream("stationary_uniform", 128)
        for policy in (PRIMARY_ENDPOINT_POLICY,
                       EMPIRICAL_EXTREMA_SENSITIVITY_POLICY):
            result = run_policy_diagnostic(stream, policy)
            self.assertEqual(result["n_events"], 128)
            self.assertEqual(result["seed"], stream.seed)
            self.assertEqual(result["method_version"], policy.method_version)
            self.assertIn("peak_python_bytes_tracemalloc_separate_run", result)
            self.assertEqual(sum(result["prediction_counts"].values()), 128)
            self.assertEqual(result["source_target"],
                             "1[score > true synthetic Q_Ft(0.99)]")
            self.assertEqual(result["false_reset_count_stationary"],
                             result["reset_count"])
        conservative = run_policy_diagnostic(stream, PRIMARY_ENDPOINT_POLICY,
                                             measure_python_peak=False)
        self.assertEqual(conservative["prediction_counts"]["SUN_CS_ANOMALY"], 0)
        self.assertEqual(conservative["reset_count"], 0)


if __name__ == "__main__":
    unittest.main()
