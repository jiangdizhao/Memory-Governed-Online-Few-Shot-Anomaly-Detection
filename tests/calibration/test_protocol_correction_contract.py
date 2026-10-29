from __future__ import annotations

from pathlib import Path
import unittest

from experiments.calibration.method_registry import METHOD_CARDS


ROOT = Path(__file__).resolve().parents[2]


class ProtocolCorrectionContractTests(unittest.TestCase):
    def test_stage1_candidate_retains_locked_scientific_protocol(self):
        text = (ROOT / "docs/calibration/STAGE1_EVALUATION_PROTOCOL_FROZEN_CANDIDATE_v2.md").read_text(
            encoding="utf-8")
        for required in (
            "RESEARCHER_APPROVAL_REQUIRED",
            "B6-SUN-CS-A3-CONSERVATIVE-EXTENDED-REAL-v2.0.0",
            "SUN_CS_A3_EMPIRICAL_EXTREMA",
            "t=1050",
            "3% Q-A-TS",
            "exactly four deployed",
            "64 *additional*",
            "K=256",
            "Mint/PCB",
            "source-native",
            "fallback",
            "TWO_SIDED_EVIDENCE_AVAILABLE",
            "DESCRIPTIVE_TWO_SIDED_CONSTRAINT_FEASIBLE",
            "14 zero-error",
            "Empirical comparability does not imply theoretical equivalence",
        ):
            self.assertIn(required, text)

    def test_gate_amendment_remains_a_candidate(self):
        text = (ROOT / "docs/calibration/STAGE0_GATE_AMENDMENT_CANDIDATE.md").read_text(
            encoding="utf-8")
        self.assertIn("RESEARCHER_APPROVAL_REQUIRED", text)
        self.assertIn("does not change the current mandatory", text)
        self.assertIn("No candidate PASS is a Stage 1 authorization", text)
        self.assertIn("conservative_extended_real", text)
        self.assertIn("no verified author-code parity", text)

    def test_b7_numerical_defaults_are_unchanged(self):
        defaults = METHOD_CARDS["B7"]["defaults"]
        self.assertEqual(defaults["min_normals"], 16)
        self.assertEqual(defaults["min_defects"], 8)
        self.assertEqual(defaults["delta"], .05)
        self.assertEqual(defaults["support_fpr"], .20)
        self.assertEqual(defaults["support_fnr"], .20)
        self.assertEqual(defaults["max_step"], .01)


if __name__ == "__main__":
    unittest.main()
