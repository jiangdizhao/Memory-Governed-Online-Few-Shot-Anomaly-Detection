import math
import unittest

from experiments.calibration.controllers import StateConsistentQuantile
from experiments.calibration.protocol import EvidenceAnchor, EvidenceSnapshot, Observation
from experiments.calibration.scdac_g import SCDACGController, SCDACGError


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
        self.assertEqual(b7.decision, "CONFLICT_BOUNDED_MINIMUM_DESCRIPTIVE_RISK")

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


if __name__ == "__main__":
    unittest.main()
