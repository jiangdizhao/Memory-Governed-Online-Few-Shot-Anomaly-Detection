import unittest

from experiments.calibration.profiles import QueryProtocolConfig
from experiments.calibration.query_tape import build_qa_ts, build_qb
from experiments.risk_safe.preflight import ImageRecord
from experiments.calibration.real_iad_preflight import (
    _label_blind_tape_records, _select_support_and_c64,
)


def metadata_rows(*, flip_labels=False):
    rows = []
    for group in range(100):
        for camera in range(1, 6):
            is_good = not flip_labels
            name = f"audiojack_{group:04d}_OK_C{camera}_20260101000000.png"
            rel_path = f"OK/S{group:04d}/{name}"
            rows.append(ImageRecord(
                path=f"G:/synthetic/audiojack/{rel_path}", rel_path=rel_path,
                class_name="audiojack", split="train",
                sample_key=f"audiojack_{group:04d}_OK",
                directory_serial_key=f"S{group:04d}", capture_session="OK",
                camera_id=camera, timestamp=f"2026010100000{camera}",
                is_good=is_good,
            ))
    return rows


class RealIADMetadataPreflightTests(unittest.TestCase):
    def test_historical_d2_support_and_c64_are_reproducible_and_disjoint(self):
        train = metadata_rows()
        support, c64, support_keys, c64_keys = _select_support_and_c64(train)
        support2, c642, support_keys2, c64_keys2 = _select_support_and_c64(train)
        self.assertEqual(len(support), 4)
        self.assertEqual(len(c64), 64)
        self.assertEqual((support_keys, c64_keys), (support_keys2, c64_keys2))
        self.assertEqual([r.image_id for r in support], [r.image_id for r in support2])
        self.assertEqual([r.image_id for r in c64], [r.image_id for r in c642])
        self.assertFalse(set(support_keys) & set(c64_keys))
        self.assertEqual(len(set(c64_keys)), 64)
        self.assertEqual(len(set(support_keys)), 4)

    def test_query_tapes_ignore_label_fields_and_enforce_one_view_per_group(self):
        records_a = _label_blind_tape_records(metadata_rows())
        records_b = _label_blind_tape_records(metadata_rows(flip_labels=True))
        config = QueryProtocolConfig()
        for builder in (build_qa_ts, build_qb):
            tape_a = builder(records_a, config)
            tape_b = builder(records_b, config)
            self.assertEqual(tape_a.digest, tape_b.digest)
            self.assertEqual(tape_a.requested_budget, int(.03 * len(records_a) + .5))
            self.assertEqual(len(tape_a.queried_groups), len(tape_a.events))
            self.assertEqual(len(tape_a.bin_edges), 17)
        qa = build_qa_ts(records_a, config)
        self.assertEqual(sum(qa.bin_quotas), qa.requested_budget)
        self.assertEqual(sum(qa.bin_quotas), len(qa.events))
        self.assertEqual(qa.requested_budget,
                         int(0.03 * len(records_a) + 0.5))
        self.assertAlmostEqual(len(qa.events) / len(records_a),
                               qa.requested_budget / len(records_a))
        qb = build_qb(records_a, config)
        self.assertEqual(qb.early_budget, int(.20 * qb.requested_budget + .5))
        self.assertEqual(sum(event.position < 400 for event in qb.events),
                         qb.early_budget)
        self.assertEqual(sum(event.position >= 400 for event in qb.events),
                         qb.requested_budget - qb.early_budget)
        qb_bin_counts = [sum(event.time_bin == bin_index for event in qb.events)
                         for bin_index in range(16)]
        self.assertEqual(qb_bin_counts, list(qb.bin_quotas))


if __name__ == "__main__":
    unittest.main()
