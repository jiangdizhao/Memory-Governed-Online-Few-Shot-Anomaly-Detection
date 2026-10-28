from __future__ import annotations

import unittest

from experiments.calibration.controller_activation import _b7_synthetic_branches


class ControllerActivationTests(unittest.TestCase):
    def test_b7_activation_matrix_exercises_registered_synthetic_branches(self):
        result = _b7_synthetic_branches(0.5)
        self.assertEqual(result["BOTH"]["state"], "BOTH")
        self.assertEqual(result["NORMAL_ONLY"]["state"], "NORMAL_ONLY")
        self.assertEqual(result["CONFLICT"]["state"], "CONFLICT")
        self.assertTrue(result["BOTH_vs_SCQ"]["b7_differs_from_scq"])
        self.assertGreater(result["BOTH"]["descriptive_feasible_candidates"], 0)
        self.assertTrue(result["step_clipping"]["step_bounded"])
        self.assertTrue(result["step_clipping"]["executed_feasibility_logged"])
        self.assertTrue(result["queried_defect_only"]["unqueried_policy_rows_excluded"])
        self.assertTrue(result["current_memory_anchor_refresh"]["memory_version_advanced"])
        self.assertTrue(result["current_memory_anchor_refresh"]["protected_anchor_scores_changed"])
        self.assertTrue(result["current_memory_anchor_refresh"]["queried_normal_lso_policy_preserved"])
        self.assertTrue(result["checkpoint_equivalence"])
        self.assertTrue(result["prediction_available"])


if __name__ == "__main__":
    unittest.main()
