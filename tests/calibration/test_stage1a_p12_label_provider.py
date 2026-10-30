"""CPU-only checks for query-position-only label access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.calibration.stage1a_p12_label_provider import (
    QueriedLabelAccessError, QueriedPositionLabelReader, sha256_file,
)


def group_sha(value: str) -> str:
    return hashlib.sha256(("official-sample-group\0" + value).encode()).hexdigest()


class QueriedLabelProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "audiojack.json"
        rows = [{"image_path": f"images/{i}.jpg",
                 "mask_path": f"masks/{i}.png",
                 "anomaly_class": label,
                 "opaque_note": {"embedded": ["unqueried", i]}}
                for i, label in enumerate(("OK", "defect-a", "OK", "defect-b", "OK"))]
        self.path.write_text(json.dumps({
            "meta": {"normal_class": "OK"},
            "train": [{"anomaly_class": "train-only"}],
            "test": rows,
        }, separators=(",", ":")), encoding="utf-8")
        self.source_sha = sha256_file(self.path)
        self.positions = (1, 3)
        self.groups = {1: group_sha("group-1"), 3: group_sha("group-3")}
        self.reader = QueriedPositionLabelReader(
            category="audiojack", path=self.path,
            expected_source_sha256=self.source_sha,
            positions=self.positions, group_sha256_by_position=self.groups)

    def tearDown(self):
        self.reader.close()
        self.temp.cleanup()

    def test_decodes_only_the_requested_tape_label_at_each_call(self):
        self.assertEqual(self.reader("audiojack", 1, "group-1"), 1)
        self.assertEqual(self.reader.decoded_positions, [1])
        self.assertEqual(self.reader("audiojack", 3, "group-3"), 1)
        self.assertEqual(self.reader.decoded_positions, [1, 3])

    def test_unqueried_position_and_group_substitution_fail_closed(self):
        with self.assertRaises(QueriedLabelAccessError):
            self.reader("audiojack", 2, "group-2")
        with self.assertRaises(QueriedLabelAccessError):
            self.reader("audiojack", 1, "other-group")
        self.assertEqual(self.reader.decoded_positions, [])

    def test_restored_reader_can_skip_cached_prior_query_without_redecoding_it(self):
        self.assertEqual(self.reader("audiojack", 3, "group-3"), 1)
        self.assertEqual(self.reader.decoded_positions, [3])

    def test_source_identity_mutation_blocks_before_label_decode(self):
        self.reader.close()
        altered = QueriedPositionLabelReader(
            category="audiojack", path=self.path,
            expected_source_sha256="0" * 64,
            positions=self.positions, group_sha256_by_position=self.groups)
        with self.assertRaisesRegex(QueriedLabelAccessError, "source hash"):
            altered("audiojack", 1, "group-1")
        self.assertEqual(altered.decoded_positions, [])
        altered.close()

    def test_missing_queried_label_fails_closed(self):
        malformed_path = Path(self.temp.name) / "missing_label.json"
        malformed_path.write_text(json.dumps({
            "meta": {"normal_class": "OK"}, "test": [{"image_path": "x.jpg"}]
        }), encoding="utf-8")
        reader = QueriedPositionLabelReader(
            category="audiojack", path=malformed_path,
            expected_source_sha256=sha256_file(malformed_path),
            positions=(0,), group_sha256_by_position={0: group_sha("group-0")})
        with self.assertRaisesRegex(QueriedLabelAccessError, "no anomaly_class"):
            reader("audiojack", 0, "group-0")
        self.assertEqual(reader.decoded_positions, [])
        reader.close()


if __name__ == "__main__":
    unittest.main()
