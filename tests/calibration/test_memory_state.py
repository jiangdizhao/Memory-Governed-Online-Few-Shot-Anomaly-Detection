from __future__ import annotations

import unittest

import torch

from experiments.calibration.memory_state import CalibrationGreedyState
from experiments.risk_safe.shadow_memory import GreedyBank


class CalibrationMemoryStateTests(unittest.TestCase):
    def test_content_version_is_independent_of_legacy_greedy_version(self):
        original = GreedyBank([torch.tensor([[1.0, 0.0]])], [2], version=4)
        state = CalibrationGreedyState(original, memory_content_version=7)

        same_bytes_legacy_advance = GreedyBank(
            [torch.tensor([[1.0, 0.0]])], [2], version=5)
        unchanged, changed = state.transition(same_bytes_legacy_advance)
        self.assertFalse(changed)
        self.assertEqual(unchanged.memory_content_version, 7)
        self.assertEqual(unchanged.legacy_greedy_version, 5)

        different_bytes = GreedyBank(
            [torch.tensor([[0.0, 1.0]])], [2], version=6)
        updated, changed = unchanged.transition(different_bytes)
        self.assertTrue(changed)
        self.assertEqual(updated.memory_content_version, 8)
        self.assertEqual(updated.legacy_greedy_version, 6)

    def test_state_digests_keep_content_and_version_axes_separate(self):
        first = CalibrationGreedyState(
            GreedyBank([torch.tensor([[1.0, 0.0]])], [2], version=2), 3)
        second = CalibrationGreedyState(
            GreedyBank([torch.tensor([[1.0, 0.0]])], [2], version=9), 3)
        self.assertEqual(first.content_sha256(), second.content_sha256())
        self.assertEqual(first.sha256(), second.sha256())
        self.assertNotEqual(first.legacy_bank.sha256(), second.legacy_bank.sha256())


if __name__ == "__main__":
    unittest.main()
