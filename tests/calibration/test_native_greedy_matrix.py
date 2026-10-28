from __future__ import annotations

import unittest

from experiments.calibration.native_greedy_matrix import run_native_greedy_matrix


class NativeGreedyParityMatrixTests(unittest.TestCase):
    def test_append_replace_skip_and_repeated_slot_source_audit(self):
        result = run_native_greedy_matrix()
        self.assertEqual(result.get("status"), "PASS", result)
        self.assertEqual(result["operation_counts"],
                         {"APPEND": 8, "REPLACE": 10, "SKIP": 22})
        self.assertEqual(result["static_support_immutability"], "PASS")
        self.assertEqual(result["source_attribution_vs_actual_prototype_contents"],
                         "PASS")
        self.assertEqual(result["same_slot_replacement_source_audit"]["slot_indices"],
                         [[0], [0]])
        self.assertEqual(result["source_aware_lso_after_replacement"]["status"],
                         "PASS")
        self.assertTrue(all(row["exact_native_memoryarm_shadow_shared_writer"]
                            for row in result["matrix"]))
        for row in result["matrix"]:
            changed = row["memory_content_changed"]
            self.assertEqual(changed,
                             row["memory_version_after"] > row["memory_version_before"])
            self.assertEqual(row["numeric_tolerance"]["atol"], 0.0)
            self.assertEqual(row["numeric_tolerance"]["rtol"], 0.0)


if __name__ == "__main__":
    unittest.main()
