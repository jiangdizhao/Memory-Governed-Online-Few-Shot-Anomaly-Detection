import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.calibration import PROTOCOL_VERSION, runner
from experiments.calibration.method_registry import METHOD_CARDS
from experiments.calibration.profiles import DEFAULT_PROFILE, canonical_json


class RunnerIntegrityTests(unittest.TestCase):
    def _artifact(self, directory: Path) -> None:
        source_hashes = runner._source_hashes()
        fingerprint = runner._sha(canonical_json(source_hashes).encode())
        runner._json_write(directory / "run_identity.json", {
            "schema": "calibration.stage0_run_identity.v1",
            "run_key": "fixture",
            "profile_sha256": DEFAULT_PROFILE.sha256,
            "source_fingerprint_sha256": fingerprint,
        })
        runner._json_write(directory / "protocol_manifest.json", {
            "profile_sha256": DEFAULT_PROFILE.sha256,
            "method_cards_sha256": runner._sha(
                canonical_json(METHOD_CARDS).encode()),
            "protocol_version": PROTOCOL_VERSION,
            "overall_status": "BLOCKED_WIP",
        })
        runner._json_write(directory / "source_hashes.json", {
            "stage0_source_hashes": source_hashes,
        })
        self._checksums(directory)

    @staticmethod
    def _checksums(directory: Path) -> None:
        checksums = {
            path.name: runner._hash_file(path)
            for path in sorted(directory.iterdir())
            if path.is_file() and path.name != "checksums.json"
        }
        runner._json_write(directory / "checksums.json", checksums)

    def test_valid_manifest_and_checksums_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._artifact(directory)
            result = runner._verify_artifact(directory)
            self.assertTrue(result["checksums_verified"])
            self.assertTrue(result["source_and_config_match"])
            self.assertEqual(result["artifact_status"], "BLOCKED_WIP")

    def test_corrupt_artifact_fails_checksum_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._artifact(directory)
            target = directory / "protocol_manifest.json"
            target.write_text(target.read_text(encoding="utf-8") + "corrupt",
                              encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "checksum verification failed"):
                runner._verify_artifact(directory)

    def test_source_fingerprint_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._artifact(directory)
            source_path = directory / "source_hashes.json"
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            payload["stage0_source_hashes"]["synthetic.py"] = "0" * 64
            runner._json_write(source_path, payload)
            self._checksums(directory)
            with self.assertRaisesRegex(RuntimeError, "provenance does not match"):
                runner._verify_artifact(directory)

    def test_native_smoke_reports_missing_metadata_as_skip(self):
        from experiments.calibration.native_parity_smoke import run_native_parity_smoke

        with patch("experiments.calibration.native_parity_smoke.torch.cuda.is_available",
                   return_value=True):
            result = run_native_parity_smoke(
                Path("Z:/missing-stage0-dataset"),
                Path("Z:/missing-stage0-json"))
        self.assertEqual(result["status"], "SKIP")
        self.assertIn("metadata roots are absent", result["reason"])

    def test_native_event_replay_reports_missing_metadata_as_skip(self):
        from experiments.calibration.native_event_replay import run_native_event_replay

        with patch("experiments.calibration.native_event_replay.torch.cuda.is_available",
                   return_value=True):
            result = run_native_event_replay(
                Path("Z:/missing-stage0-dataset"),
                Path("Z:/missing-stage0-json"))
        self.assertEqual(result["status"], "SKIP")
        self.assertIn("metadata roots are absent", result["reason"])


if __name__ == "__main__":
    unittest.main()
