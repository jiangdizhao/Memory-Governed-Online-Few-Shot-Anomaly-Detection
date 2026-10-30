from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from experiments.calibration import stage1a_p12_runner as p12


class Stage1AP12SourceBindingTests(unittest.TestCase):
    def test_source_binding_requires_protocol_score_manifest_and_current_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "stage05.py"
            source.write_bytes(b"current source bytes")
            current = hashlib.sha256(source.read_bytes()).hexdigest()
            stale = "0" * 64

            report = p12._source_binding_report(
                {"stage05.py": stale}, {"stage05.py": current}, root)

            self.assertEqual(report["status"], "BLOCKED_SOURCE_LINKAGE_MISMATCH")
            self.assertFalse(report["all_sources_match"])
            row = report["files"]["stage05.py"]
            self.assertEqual(row["stage075a_protocol_sha256"], stale)
            self.assertEqual(row["stage05_score_manifest_sha256"], current)
            self.assertEqual(row["current_source_sha256"], current)
            self.assertFalse(row["protocol_matches_current"])
            self.assertTrue(row["score_manifest_matches_current"])
            self.assertFalse(row["protocol_matches_score_manifest"])

    def test_source_binding_passes_only_when_all_three_identities_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "native.py"
            source.write_bytes(b"identical frozen bytes")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()

            report = p12._source_binding_report(
                {"native.py": digest}, {"native.py": digest}, root)

            self.assertEqual(report["status"], "PASS_EXACT_SOURCE_LINKAGE")
            self.assertTrue(report["all_sources_match"])

    def test_two_exact_missing_pre_git_sources_can_be_reconstructed_not_recovered(self):
        pins = p12.P12_MISSING_HISTORICAL_STAGE05_SOURCE
        rows = {}
        for name, pair in pins.items():
            rows[name] = {
                "stage075a_protocol_sha256": pair["frozen_sha256"],
                "stage05_score_manifest_sha256": pair["frozen_sha256"],
                "current_source_sha256": pair["current_sha256"],
            }
        same = hashlib.sha256(b"native scorer unchanged").hexdigest()
        rows["experiments/calibration/native.py"] = {
            "stage075a_protocol_sha256": same,
            "stage05_score_manifest_sha256": same,
            "current_source_sha256": same,
        }
        report = {"status": "BLOCKED_SOURCE_LINKAGE_MISMATCH",
                  "all_sources_match": False, "files": rows}
        assessed = p12._assess_scoped_source_reconstruction(report)
        self.assertTrue(assessed["execution_allowed"])
        self.assertEqual(assessed["status"], "PASS_SCOPED_CURRENT_SOURCE_RECONSTRUCTION")
        self.assertFalse(assessed["historical_source_bytes_recovered"])
        self.assertEqual({row["path"] for row in assessed["exception_files"]}, set(pins))
        self.assertEqual(assessed["original_historical_linkage_status"],
                         "BLOCKED_SOURCE_LINKAGE_MISMATCH")

        for changed_name, key, changed in [
            ("experiments/calibration/stage05.py",
             "current_source_sha256", same),
            ("experiments/calibration/stage05_runner.py",
             "stage075a_protocol_sha256", same),
            ("experiments/calibration/stage05.py",
             "stage05_score_manifest_sha256", same),
            ("experiments/calibration/native.py",
             "current_source_sha256", "0" * 64),
        ]:
            altered = {name: dict(row) for name, row in rows.items()}
            altered[changed_name][key] = changed
            self.assertFalse(p12._assess_scoped_source_reconstruction(
                {"status": "BLOCKED_SOURCE_LINKAGE_MISMATCH",
                 "files": altered})["execution_allowed"])
        altered = {name: dict(row) for name, row in rows.items()}
        altered["new_source.py"] = {
            "stage075a_protocol_sha256": same,
            "stage05_score_manifest_sha256": same,
            "current_source_sha256": "1" * 64,
        }
        denied = p12._assess_scoped_source_reconstruction(
            {"status": "BLOCKED_SOURCE_LINKAGE_MISMATCH", "files": altered})
        self.assertFalse(denied["execution_allowed"])
        self.assertIn("new_source.py", denied["unapproved_mismatch_files"])

    def test_exact_sources_remain_exact_without_exception(self):
        value = hashlib.sha256(b"unchanged").hexdigest()
        assessed = p12._assess_scoped_source_reconstruction({
            "status": "PASS_EXACT_SOURCE_LINKAGE",
            "all_sources_match": True,
            "files": {"native.py": {
                "stage075a_protocol_sha256": value,
                "stage05_score_manifest_sha256": value,
                "current_source_sha256": value,
            }},
        })
        self.assertEqual(assessed["status"], "PASS_EXACT_SOURCE_LINKAGE")
        self.assertTrue(assessed["historical_source_bytes_recovered"])
        self.assertEqual(assessed["exception_files"], [])

    def test_missing_current_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = p12._source_binding_report(
                {"missing.py": "a" * 64}, {"missing.py": "a" * 64}, Path(temporary))
            self.assertEqual(report["status"], "BLOCKED_SOURCE_LINKAGE_MISMATCH")
            self.assertIsNone(report["files"]["missing.py"]["current_source_sha256"])


if __name__ == "__main__":
    unittest.main()
