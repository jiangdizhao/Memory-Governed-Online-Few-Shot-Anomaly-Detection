from __future__ import annotations

import unittest

from experiments.calibration.b7_engine_activation import run_b7_engine_activation


class B7FullEngineActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_b7_engine_activation()

    def test_named_full_engine_activation_pass(self):
        self.assertEqual(self.report["pass_name"],
                         "B7_SYNTHETIC_FULL_ENGINE_ACTIVATION_PASS-v1.0.0")
        self.assertEqual(self.report["status"], "PASS")
        self.assertFalse(self.report["stage1_or_stage2_run"])
        self.assertFalse(self.report["confirmation_outcomes_opened"])

    def test_sparse_scenario_is_honestly_underpowered(self):
        row = self.report["scenarios"]["sparse_feedback_mechanism"]
        self.assertTrue(row["checkpoint_resume_identical"])
        self.assertAlmostEqual(row["query_fraction_images"], .03, delta=.002)
        self.assertLess(row["synthetic_evaluator_queried_defect_count"], 8)
        self.assertEqual(row["B7"]["state_final"], "NORMAL_ONLY")
        self.assertEqual(row["B7"]["candidate_evaluations"], 0)
        self.assertTrue(row["c64_current_memory_version_all_64"])
        self.assertGreater(row["queried_normal_lso_source_rows_excluded"], 0)

    def test_two_sided_scenario_exercises_engine_state_and_bound_paths(self):
        row = self.report["scenarios"]["two_sided_evidence_stress"]
        b7 = row["B7"]
        self.assertTrue(row["checkpoint_resume_identical"])
        self.assertTrue(row["threshold_counterfactual_score_memory_query_invariance"])
        self.assertEqual(b7["state_final"], "BOTH")
        self.assertGreater(b7["state_event_counts"]["NORMAL_ONLY"], 0)
        self.assertGreater(b7["state_event_counts"]["BOTH"], 0)
        self.assertGreater(b7["state_event_counts"]["CONFLICT"], 0)
        self.assertGreater(b7["candidate_evaluations"], 0)
        self.assertGreater(b7["bound_evaluations"], 0)
        self.assertGreater(b7["threshold_update_count"], 0)
        self.assertLessEqual(b7["max_observed_step"], b7["configured_max_step"] + 1e-12)
        self.assertEqual(b7["optimization_event_count"],
                         len(b7["candidate_evaluations_per_optimization"]))
        self.assertTrue(b7["optimization_revisions_unique"])
        self.assertEqual(len(b7["optimization_evidence_revisions"]),
                         len(set(b7["optimization_evidence_revisions"])))
        self.assertTrue(b7["queried_only_feedback"])
        self.assertFalse(b7["score_stream_consumed_by_B7"])
        self.assertEqual(row["synthetic_evaluator_queried_defect_count"], 21)
        self.assertTrue(row["c64_current_memory_version_all_64"])
        self.assertGreater(row["queried_normal_lso_source_rows_excluded"], 0)
        self.assertGreater(row["runtime_seconds"], 0)
        self.assertGreater(row["python_peak_allocated_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
