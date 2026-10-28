from __future__ import annotations

import hashlib
import json
import subprocess
import unittest

from experiments.calibration.source_provenance import (
    PROTECTED_BASELINE_PATH,
    PROTECTED_SOURCE_COMMIT,
    ROOT,
    protected_dependency_paths,
    verify_protected_sources,
)


class ProtectedSourceProvenanceTests(unittest.TestCase):
    def test_static_baseline_covers_resolved_native_dependency_closure(self):
        baseline = json.loads(PROTECTED_BASELINE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(baseline["baseline_commit"], PROTECTED_SOURCE_COMMIT)
        self.assertEqual(set(baseline["dependency_paths"]),
                         set(protected_dependency_paths()))
        self.assertIn("experiments/risk_safe/shadow_memory.py",
                      baseline["dependency_paths"])
        self.assertIn("experiments/v02_architecture_core.py",
                      baseline["dependency_paths"])
        self.assertEqual(set(baseline["files"]), set(baseline["dependency_paths"]))

    def test_historical_shadow_memory_is_byte_identical_to_pre_fourth_pass(self):
        path = "experiments/risk_safe/shadow_memory.py"
        expected = subprocess.run(
            ["git", "show", f"{PROTECTED_SOURCE_COMMIT}:{path}"],
            cwd=ROOT, capture_output=True, check=True).stdout
        current = (ROOT / path).read_bytes()
        self.assertEqual(hashlib.sha256(current).hexdigest(),
                         hashlib.sha256(expected).hexdigest())
        self.assertTrue(verify_protected_sources()["shadow_memory_byte_equal_to_baseline"])

    def test_committed_or_worktree_historical_dependency_changes_are_detected(self):
        report = verify_protected_sources()
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["unrecorded_resolved_dependencies"], [])
        self.assertEqual(report["recorded_but_unresolved_dependencies"], [])
        self.assertEqual(report["dependency_hash_count"],
                         len(protected_dependency_paths()))


if __name__ == "__main__":
    unittest.main()
