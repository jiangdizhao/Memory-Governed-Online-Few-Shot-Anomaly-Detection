"""Integrity checks for the blocked, metadata-only P1 reconstruction package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.calibration.stage1a_p1_reconstruction import source_fingerprints


PACKAGE = ROOT / "docs" / "calibration" / "audit" / "stage1a_p1_reconstruction_v1_20261030"
SOURCE_ALLOWLIST = [
    "experiments/calibration/stage075a_runner.py",
    "experiments/calibration/native.py",
    "experiments/calibration/anchor_ledger.py",
    "experiments/calibration/native_event_replay.py",
    "experiments/calibration/protocol.py",
    "experiments/calibration/profiles.py",
    "experiments/calibration/engine.py",
    "experiments/calibration/controllers.py",
    "experiments/calibration/spot_dspot.py",
    "experiments/calibration/scdac_g.py",
    "experiments/calibration/stage075b_controllers.py",
    "experiments/calibration/stage1a_p1_reconstruction.py",
    "experiments/risk_safe/native_bridge.py",
    "experiments/risk_safe/shadow_memory.py",
    "experiments/v02_memory.py",
    "test_phase3_AIF_simplified_v22.py",
    "vmb_visionad_new.py",
]


def read_json(name: str) -> dict:
    return json.loads((PACKAGE / name).read_text(encoding="utf-8"))


class Stage1AP1PackageTests(unittest.TestCase):
    def test_package_keeps_execution_and_real_parity_blocked(self) -> None:
        parity = read_json("exact_native_replay_parity.json")
        gates = read_json("p1_gate_matrix.json")
        compute = read_json("compute_accounting.json")
        self.assertEqual(parity["package_status"], "BLOCKED_BEFORE_DATA_OR_GPU_EXECUTION")
        self.assertFalse(parity["bounded_p1_native_compute_authorization_recorded"])
        self.assertFalse(parity["separate_queried_feedback_access_decision_recorded"])
        self.assertEqual(parity["historical_g2_status"], "BLOCKED")
        self.assertFalse(parity["historical_executed_source_hash_recovered"])
        self.assertFalse(parity["historical_query_response_digest_recovered"])
        self.assertFalse(gates["stage1a_performance_evaluation_executed"])
        self.assertEqual(gates["gates"][2]["status"], "BLOCKED_NOT_RECORDED")
        self.assertEqual(gates["gates"][3]["status"], "BLOCKED_NOT_RECORDED")
        self.assertEqual(compute["actual_p1_counts"]["gpu_calls"], 0)
        self.assertEqual(compute["actual_p1_counts"]["queried_label_reads"], 0)
        self.assertEqual(compute["bounded_minimum_additional_work_plan"]["full_stream_backbone_calls"], 0)

    def test_source_fingerprint_manifest_matches_current_allowlist(self) -> None:
        recorded = read_json("feature_cache_provenance.json")["source_fingerprint_allowlist"]
        actual = source_fingerprints(ROOT, SOURCE_ALLOWLIST)
        self.assertEqual(recorded, actual)

    def test_stage075b_offset_semantics_are_explicit(self) -> None:
        report = (PACKAGE / "STAGE1A_P1_RECONSTRUCTION_REPORT.md").read_text(encoding="utf-8")
        manifest = read_json("anchor_evidence_manifest.json")
        self.assertIn("does not use C64", report)
        self.assertIn("one-sided", report)
        self.assertIn("holds", report)
        self.assertEqual(manifest["anchor_rules"]["queried_normal"].split(";")[0],
                         "Score current memory with the official source group excluded from every layer")

    def test_package_checksums_cover_required_outputs(self) -> None:
        checksums = read_json("checksums.json")
        for relative, expected_sha in checksums["files"].items():
            payload = (PACKAGE / relative).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_sha)
        self.assertEqual(set(checksums["files"]), {
            "exact_native_replay_parity.json", "anchor_evidence_manifest.json",
            "feature_cache_provenance.json", "compute_accounting.json",
            "p1_gate_matrix.json", "STAGE1A_P1_RECONSTRUCTION_REPORT.md"})


if __name__ == "__main__":
    unittest.main()
