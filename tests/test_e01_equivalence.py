"""Offline regression for E01. No weights, dataset, or GPU required."""
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import p0_v22_equivalence as e01


class DummyTensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float16)
    def detach(self):
        return self
    def cpu(self):
        return self
    def contiguous(self):
        return self
    def numpy(self):
        return self.data


class E01Tests(unittest.TestCase):
    def test_fingerprint_change_is_detected(self):
        x = np.full((3, 4), 1, dtype=np.float16)
        bank = SimpleNamespace(normal_fixed={
            "test": {0: {"id": [SimpleNamespace(
                protos=DummyTensor(x), K=25)]}}
        })
        a = e01.memory_profile(bank, "test")
        self.assertEqual(a["total_prototypes"], 3)
        self.assertEqual(a["layers"][0]["feature_dim"], 4)
        x[:] = 2
        b = e01.memory_profile(bank, "test")
        self.assertNotEqual(a["sha256"], b["sha256"])

    def test_true_verdict_requires_all_checks(self):
        profile = {"layers": [{"n_prototypes": 3, "feature_dim": 4}],
                   "sha256": "same"}
        paired = e01.paired_score_statistics([.1, .3], [.1, .3], 1e-4)
        yes = e01.verdict(True, profile, profile, paired, 0, 1e-4, True)
        self.assertEqual(yes["status"], "equivalent_on_audited_conditions")
        no = e01.verdict(False, profile, profile, paired, 0, 1e-4, True)
        self.assertEqual(no["status"], "not_equivalent")
        self.assertIn("native_support_selection", no["failed_checks"])

    def test_paired_score_requires_nonempty_finite_values(self):
        with self.assertRaises(ValueError):
            e01.paired_score_statistics([], [], 1e-4)
        with self.assertRaises(ValueError):
            e01.paired_score_statistics([np.nan], [.1], 1e-4)
        with self.assertRaises(ValueError):
            e01.paired_score_statistics([.1], [.1, .2], 1e-4)
        self.assertFalse(e01.paired_score_statistics([.1], [.2], 1e-4)["within_tolerance"])

    def test_augmentation_call_groups(self):
        calls = [{"fingerprints":[f"hash{i}"],"count":1} for i in range(8)]
        a = e01.summarize_aug_calls(calls, 4)
        self.assertEqual(a["concept_pass"][0]["fingerprints"], ["hash0"])
        self.assertEqual(a["fixed_pass"][0]["fingerprints"], ["hash4"])
        with self.assertRaises(AssertionError):
            e01.summarize_aug_calls(calls[:7], 4)

    def test_image_hash_depends_on_bytes_and_size(self):
        class Img:
            def __init__(self, value, width):
                self.size=(width, 2)
                self.mode="RGB"
                self.value=value
            def tobytes(self):
                return bytes([self.value] * self.size[0] * self.size[1] * 3)
        self.assertEqual(e01.image_fingerprint(Img(1, 2)),
                         e01.image_fingerprint(Img(1, 2)))
        self.assertNotEqual(e01.image_fingerprint(Img(1, 2)),
                            e01.image_fingerprint(Img(2, 2)))
        self.assertNotEqual(e01.image_fingerprint(Img(1, 2)),
                            e01.image_fingerprint(Img(1, 3)))


if __name__ == "__main__":
    unittest.main()
