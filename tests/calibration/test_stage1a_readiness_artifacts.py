"""Integrity checks for the Stage 1-A readiness-only package.

These tests validate machine-readable planning records; they do not run an
experiment, inspect outcomes, or provide scientific validation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = ROOT / "docs" / "calibration"


def read_json(name: str) -> dict:
    return json.loads((CALIBRATION / name).read_text(encoding="utf-8"))


class Stage1AReadinessArtifactsTests(unittest.TestCase):
    def test_gate_fails_closed_without_execution_go(self) -> None:
        gate = read_json("stage1a_p0_gate_matrix_v1_2026-10-30.json")
        self.assertFalse(gate["execution_authorized"])
        self.assertFalse(gate["scientific_evaluation_executed"])
        self.assertEqual(gate["original_g2_status"], "BLOCKED")
        self.assertEqual(gate["recorded_stage075a_identity"]["protocol_sha256"],
                         "6f1d8cf40e5e7b2d98dcfabe5836b49c6cd57069d3c8377122dba64e3fc2da1c")
        self.assertEqual(gate["recorded_stage075a_identity"]["native_scorer_config_sha256"],
                         "1b8e9d96b121e3f1f6c1bbee13ab9f0063facee8fb08ad696dc5f9cb43dbd452")
        by_id = {row["id"]: row for row in gate["gates"]}
        for gate_id in ("P0-01", "P0-07", "P0-08", "P0-09", "P0-11"):
            self.assertIn(by_id[gate_id]["status"],
                          {"BLOCKED", "BLOCKED_FOR_HISTORICAL_RECONSTRUCTION", "NOT_RUN"})

    def test_trajectory_and_anchor_work_counts_are_consistent(self) -> None:
        plan = read_json("stage1a_reuse_and_compute_plan_v1_2026-10-30.json")
        self.assertEqual(plan["trajectory_inventory"]["trajectory_count"], 20)
        self.assertEqual(plan["trajectory_inventory"]["static_score_values"], 76475)
        self.assertEqual(plan["trajectory_inventory"]["dynamic_score_values"], 76475)
        work = plan["anchor_work_estimate"]
        self.assertEqual(work["c64_postwrite_values_reusable"],
                         work["memory_content_updates"] * 64)
        self.assertEqual(work["query_anchor_scoring_upper_bound"], 75406)
        self.assertEqual(plan["prior_snapshot_audit_measurements"]["snapshot_states"], 120)

    def test_native_warmup_counts_and_coverage(self) -> None:
        lengths = (3815, 3855, 3820, 3805)
        spot = sum(n - 1000 for n in lengths) / sum(lengths)
        dspot = sum(n - 1050 for n in lengths) / sum(lengths)
        self.assertTrue(math.isclose(spot, 0.7384766263484799, abs_tol=1e-14))
        self.assertTrue(math.isclose(dspot, 0.7254004576659039, abs_tol=1e-14))
        self.assertEqual(sum(lengths) - 4 * 1000, 11295)
        self.assertEqual(sum(lengths) - 4 * 1050, 11095)

    def test_method_registry_is_candidate_and_keeps_fallbacks_separate(self) -> None:
        matrix = read_json("stage1a_method_matrix_v1_2026-10-30.json")
        self.assertFalse(matrix["execution_authorized"])
        rows = {row["id"]: row for row in matrix["rows"]}
        for method in ("B0", "B1", "B2", "B3", "B4", "B5a", "B5b",
                       "B7", "Stage075BOffset", "B6", "B8"):
            self.assertIn(method, rows)
        self.assertIn("B5a_FROZEN_FALLBACK", rows)
        self.assertIn("B5b_FROZEN_FALLBACK", rows)
        self.assertNotEqual(rows["B5a"]["id"], rows["B5a_FROZEN_FALLBACK"]["id"])
        self.assertEqual(matrix["reporting_constraints"]["common_horizon_zero_based"], 1050)


if __name__ == "__main__":
    unittest.main()
