import json
from pathlib import Path
import tempfile
import unittest

from experiments.calibration import audit_export


class AuditExportTests(unittest.TestCase):
    @staticmethod
    def _json(path: Path, value):
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8")

    def test_redaction_removes_local_paths_and_example_identities(self):
        result = audit_export._sanitize({
            "dataset_root": "G:\\dataset\\realiad",
            "example_shared_samples": ["private-sample-id"],
            "nested": {"path": "C:\\Users\\rico\\image.png",
                       "status": "PASS"},
            "citation": "https://example.org/paper",
        })
        self.assertNotIn("dataset_root", result)
        self.assertNotIn("example_shared_samples", result)
        self.assertNotIn("path", result["nested"])
        self.assertEqual(result["nested"]["status"], "PASS")
        self.assertEqual(result["citation"], "https://example.org/paper")

    def test_exported_package_and_source_artifact_link_verify(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "runner", root / "audit"
            source.mkdir()
            payloads = {
                "gates.json": {
                    "G0": {"status": "PASS", "reason": "clean"},
                    "G0_PROTECTED_SOURCE_INTEGRITY": {"status": "PASS"}},
                "test_results.json": {"tests_run": 1, "passed": 1,
                                      "failed": 0, "errors": 0, "skipped": 0,
                                      "passed_test_ids": ["sample.test"]},
                "native_parity_smoke.json": {"status": "SKIP",
                                             "reason": "cached files absent"},
                "native_greedy_parity_matrix.json": {"status": "PASS"},
                "native_event_replay.json": {"status": "SKIP"},
                "real_iad_metadata_preflight.json": {
                    "status": "PASS", "dataset_root": "G:\\dataset\\realiad",
                    "example_shared_samples": ["sample-private"]},
                "controller_method_cards.json": {"B6": {"status": "BLOCKED_FIDELITY"}},
                "controller_activation_matrix.json": {"status": "PARTIAL"},
                "sun_cs_reference_status.json": {"status": "BLOCKED_FIDELITY"},
                "protected_source_verification.json": {
                    "status": "PASS", "shadow_memory_byte_equal_to_baseline": True,
                    "unrecorded_resolved_dependencies": [],
                    "recorded_but_unresolved_dependencies": [], "failures": []},
                "protocol_manifest.json": {"profile_sha256": "a" * 64,
                                            "protocol_version": "test-v1",
                                            "support_regime": "four plus C64",
                                            "overall_status": "BLOCKED_WIP"},
                "source_hashes.json": {"source_commit": "deadbeef",
                                        "source_fingerprint_sha256": audit_export._source_fingerprint(
                                            {"x.py": "c" * 64}),
                                        "stage0_source_hashes": {"x.py": "c" * 64}},
                "run_identity.json": {"run_key": "fixture"},
            }
            for name, payload in payloads.items():
                self._json(source / name, payload)
            checksums = {path.name: audit_export._sha_file(path)
                         for path in source.iterdir() if path.is_file()}
            self._json(source / "checksums.json", checksums)

            result = audit_export.export_audit(source, output)
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["verification"]["source_artifact_link_verified"])
            verified = audit_export.verify_audit(output, source_dir=source)
            self.assertTrue(verified["valid"])
            text = "\n".join(path.read_text(encoding="utf-8")
                             for path in output.iterdir() if path.is_file())
            self.assertNotIn("G:\\dataset\\realiad", text)
            self.assertNotIn("sample-private", text)
            self.assertTrue((output / "controller_activation_matrix.json").is_file())
            self.assertTrue((output / "protected_source_verification.json").is_file())
            self.assertTrue((output / "sun_cs_reference_status.json").is_file())
            self.assertIn("Source fingerprint recomputation: **PASS**", text)

            fingerprints_path = output / "protocol_fingerprints.json"
            fingerprints = json.loads(fingerprints_path.read_text(encoding="utf-8"))
            fingerprints["stage0_source_hashes"]["x.py"] = "d" * 64
            self._json(fingerprints_path, fingerprints)
            package_checksums_path = output / "checksums.json"
            package_checksums = json.loads(
                package_checksums_path.read_text(encoding="utf-8"))
            package_checksums["files"]["protocol_fingerprints.json"] = \
                audit_export._sha_file(fingerprints_path)
            self._json(package_checksums_path, package_checksums)
            semantic_mismatch = audit_export.verify_audit(output)
            self.assertFalse(semantic_mismatch["valid"])
            self.assertIn("source fingerprint", semantic_mismatch["reason"])

            target = output / "gates.json"
            target.write_text(target.read_text(encoding="utf-8") + "tamper",
                              encoding="utf-8")
            self.assertFalse(audit_export.verify_audit(output)["valid"])


if __name__ == "__main__":
    unittest.main()
