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

    def test_missing_current_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = p12._source_binding_report(
                {"missing.py": "a" * 64}, {"missing.py": "a" * 64}, Path(temporary))
            self.assertEqual(report["status"], "BLOCKED_SOURCE_LINKAGE_MISMATCH")
            self.assertIsNone(report["files"]["missing.py"]["current_source_sha256"])


if __name__ == "__main__":
    unittest.main()
