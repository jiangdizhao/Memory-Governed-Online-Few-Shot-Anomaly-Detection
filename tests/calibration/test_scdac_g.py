import math
import unittest

from experiments.calibration.controllers import StateConsistentQuantile
from experiments.calibration.protocol import EvidenceAnchor, EvidenceSnapshot, Observation
from experiments.calibration.scdac_g import (
    SCDACGConfig, SCDACGController, SCDACGError, _cp_one_sided_upper,
)


def snapshot(version=0, revision=1, normals=(), defects=()):
    normal_rows = tuple(EvidenceAnchor(f"n:{i}", float(score), version, True)
                        for i, score in enumerate(normals))
    defect_rows = tuple(EvidenceAnchor(f"d:{i}", float(score), version, False,
                                       source_exclusion_policy="queried_defect_not_written")
                        for i, score in enumerate(defects))
    return EvidenceSnapshot(version, normal_rows, defect_rows, {"c64_mutable": False}, revision)


class SCDACGTests(unittest.TestCase):
    def test_all_five_evidence_states_and_missing_side_hold(self):
        c = SCDACGController(.5)
        c.on_memory_change(snapshot())
        self.assertEqual(c.state, "NONE")
        self.assertEqual(c.finalize_step(), .5)

        c.on_memory_change(snapshot(revision=2, normals=[.1] * 16))
        self.assertEqual(c.state, "NORMAL_ONLY")
        self.assertEqual(c.finalize_step(), .5)
        self.assertIn("FNR", c.decision)

        c.on_memory_change(snapshot(revision=3, defects=[.9] * 8))
        self.assertEqual(c.state, "DEFECT_ONLY")
        self.assertEqual(c.finalize_step(), .5)
        self.assertIn("FPR", c.decision)

        c.on_memory_change(snapshot(revision=4, normals=[.1] * 64,
                                   defects=[.9] * 32))
        self.assertEqual(c.state, "BOTH")
        c.finalize_step()
        self.assertIn(c.state, {"BOTH", "CONFLICT"})
        self.assertIsNotNone(c.last_two_sided_supported_interval)

        c.on_memory_change(snapshot(revision=5, normals=[.5] * 32,
                                   defects=[.5] * 32))
        c.finalize_step()
        self.assertEqual(c.state, "CONFLICT")
        self.assertFalse(c.last_two_sided_supported_interval["active"])
        self.assertTrue(c.interval_stale)

    def test_b7_threshold_path_differs_from_b3_scq_under_two_sided_evidence(self):
        # C64's q90 lies above theta, so SCQ raises it. Two-sided current-state
        # evidence favors lowering the threshold to recover queried defects.
        normals = [.2] * 56 + [.9] * 8
        defects = [.5] * 32
        b3 = StateConsistentQuantile(theta=.85, delta=.01)
        b3.on_memory_change(snapshot(version=1, normals=normals))
        b3_theta = b3.finalize_step()

        b7 = SCDACGController(theta=.85)
        b7.on_memory_change(snapshot(version=1, normals=normals, defects=defects))
        b7_theta = b7.finalize_step()

        self.assertAlmostEqual(b3_theta, .86)
        self.assertAlmostEqual(b7_theta, .84)
        self.assertNotEqual(b3_theta, b7_theta)
        self.assertEqual(b7.state, "CONFLICT")
        self.assertEqual(b7.decision, "CONFLICT_MINIMUM_RISK_COMPROMISE")
        self.assertEqual(b7.diagnostics()["constraint_diagnostic_reason"],
                         "INSUFFICIENT_PRECISION")

    def test_candidate_grid_represents_score_ties_and_adjacent_decisions(self):
        c = SCDACGController(.25)
        grid = c._grid(.25, (.5, .5, .9))
        self.assertIn(.5, grid)
        self.assertIn(math.nextafter(.5, math.inf), grid)
        self.assertIn(math.nextafter(.5, -math.inf), grid)
        self.assertIn(.7, grid)
        self.assertIn(.25, grid)

    def test_both_state_reports_secondary_fpr_grid_and_cost_counts(self):
        c = SCDACGController(.25)
        c.on_memory_change(snapshot(normals=[.1] * 64, defects=[.9] * 32))
        c.finalize_step()
        diag = c.diagnostics()
        self.assertEqual([x["fpr_target"] for x in diag["risk_constrained"]],
                         [.05, .10, .20])
        self.assertTrue(all(x["objective_name"] == "fnr_upper_plus_movement_cost"
                            for x in diag["risk_constrained"] if x["status"] == "AVAILABLE"))
        self.assertGreater(diag["candidate_evaluations"], 0)
        self.assertGreater(diag["bound_evaluations"], 0)
        self.assertIn("none", diag["guarantee"])

    def test_current_version_only_and_checkpoint_roundtrip(self):
        c = SCDACGController(.3)
        stale = EvidenceAnchor("n:stale", .1, 0, True)
        with self.assertRaises(Exception):
            c.on_memory_change(EvidenceSnapshot(1, (stale,), (), {}))
        c.on_memory_change(snapshot(version=2, normals=[.1] * 64,
                                   defects=[.9] * 32))
        c.finalize_step()
        saved = c.state_dict()
        restored = SCDACGController(.3)
        restored.load_state_dict(saved)
        self.assertEqual(restored.state_dict(), saved)
        with self.assertRaises(SCDACGError):
            SCDACGController(.3).load_state_dict({**saved, "config": {"version": "wrong"}})

    def test_perfect_separation_precision_boundary_at_8_13_14_and_32_defects(self):
        self.assertAlmostEqual(_cp_one_sided_upper(0, 8, .05), .312344, places=5)
        self.assertGreater(_cp_one_sided_upper(0, 13, .05), .20)
        self.assertLessEqual(_cp_one_sided_upper(0, 14, .05), .20)
        for defect_count in (8, 13, 14, 32):
            with self.subTest(defect_count=defect_count):
                c = SCDACGController(.5)
                c.on_memory_change(snapshot(
                    revision=defect_count,
                    normals=[.1] * 64,
                    defects=[.9] * defect_count,
                ))
                c.finalize_step()
                diag = c.diagnostics()
                if defect_count in (8, 13):
                    self.assertEqual(c.state, "CONFLICT")
                    self.assertEqual(diag["constraint_diagnostic_reason"],
                                     "INSUFFICIENT_PRECISION")
                    self.assertGreater(diag["empirical_feasible_candidate_count"], 0)
                else:
                    self.assertEqual(c.state, "BOTH")
                    self.assertEqual(diag["constraint_diagnostic_reason"],
                                     "DESCRIPTIVE_FEASIBLE_SET")
                    self.assertTrue(diag["executed_threshold_in_descriptive_feasible_region"])

    def test_empirically_incompatible_overlap_is_not_misreported_as_precision(self):
        c = SCDACGController(.5)
        c.on_memory_change(snapshot(
            normals=[.1] * 48 + [.8] * 16,
            defects=[.1] * 12 + [.9] * 20,
        ))
        c.finalize_step()
        self.assertEqual(c.state, "CONFLICT")
        self.assertEqual(c.diagnostics()["constraint_diagnostic_reason"],
                         "RISK_CONSTRAINT_INCOMPATIBILITY")
        self.assertEqual(c.diagnostics()["empirical_feasible_candidate_count"], 0)

    def test_identical_empirical_score_distributions_are_unidentifiable(self):
        c = SCDACGController(.5)
        c.on_memory_change(snapshot(
            normals=[.2] * 32 + [.8] * 32,
            defects=[.2] * 16 + [.8] * 16,
        ))
        c.finalize_step()
        self.assertEqual(c.state, "CONFLICT")
        self.assertEqual(c.diagnostics()["constraint_diagnostic_reason"],
                         "UNIDENTIFIABLE_SCORE_DISTRIBUTIONS")

    def test_primary_optimizes_inside_feasible_set_then_reports_clip_violation(self):
        config = SCDACGConfig(movement_penalty=2.0)
        c = SCDACGController(.75, config)
        c.on_memory_change(snapshot(
            normals=[.1] * 52 + [.8] * 12,
            defects=[.9] * 32,
        ))
        c.finalize_step()
        diag = c.diagnostics()
        self.assertEqual(c.state, "BOTH")
        self.assertEqual(diag["selection_domain"], "descriptive_feasible_candidates")
        self.assertGreater(diag["proposed_threshold"], .8)
        self.assertTrue(diag["proposed_threshold_in_descriptive_feasible_region"])
        self.assertEqual(c.theta, .76)
        self.assertFalse(diag["executed_threshold_in_descriptive_feasible_region"])

    def test_fpr_diagnostic_minimizes_fnr_plus_movement_not_balanced_risk(self):
        c = SCDACGController(.5)
        rows = {.4: (.01, .20), .6: (.19, .05)}
        row = c._risk_diagnostics((.4, .6), rows)[-1]
        self.assertEqual(row["fpr_target"], .20)
        self.assertEqual(row["threshold"], .6)
        self.assertEqual(row["objective_name"], "fnr_upper_plus_movement_cost")

    def test_secondary_diagnostics_do_not_change_primary_threshold(self):
        evidence = snapshot(normals=[.1] * 64, defects=[.9] * 32)
        ordinary = SCDACGController(.25)
        without_report = SCDACGController(.25)
        without_report._risk_diagnostics = lambda candidates, rows: []
        ordinary.on_memory_change(evidence)
        without_report.on_memory_change(evidence)
        self.assertEqual(ordinary.finalize_step(), without_report.finalize_step())


if __name__ == "__main__":
    unittest.main()
