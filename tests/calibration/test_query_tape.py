import unittest

from experiments.calibration.protocol import TapeRecord
from experiments.calibration.query_tape import build_qa_ts, build_qb, verify_tape


class QueryTapeTests(unittest.TestCase):
    def test_qa_ts_group_limit_bins_and_reproducibility(self):
        records = [TapeRecord(i, f"group-{i // 5}", i % 5 + 1,
                              f"image-{i:04d}") for i in range(500)]
        tape = build_qa_ts(records)
        self.assertEqual(len(tape.events), 15)
        self.assertEqual(len(tape.queried_groups), len(tape.events))
        self.assertEqual(sum(tape.bin_quotas), 15)
        self.assertEqual(tape.digest, build_qa_ts(records).digest)
        verify_tape(tape, records)
        by_group = {}
        for event in tape.events:
            by_group[event.official_sample_key] = by_group.get(event.official_sample_key, 0) + 1
        self.assertTrue(all(count <= 1 for count in by_group.values()))

    def test_qb_early_budget_is_fixed_and_label_blind(self):
        records = [TapeRecord(i, f"group-{i}", i % 5 + 1,
                              f"image-{i:04d}") for i in range(1000)]
        tape = build_qb(records)
        self.assertEqual(len(tape.events), 30)
        self.assertEqual(tape.early_budget, 6)
        self.assertEqual(sum(e.position < 400 for e in tape.events), 6)
        verify_tape(tape, records)

    def test_selection_priority_ignores_label_bearing_key_text(self):
        normal = [TapeRecord(i, f"sample-{i // 5}_ok", i % 5 + 1,
                             f"OK/{i:04d}.png") for i in range(500)]
        defect = [TapeRecord(i, f"sample-{i // 5}_ng", i % 5 + 1,
                             f"NG/{i:04d}.png") for i in range(500)]
        a, b = build_qa_ts(normal), build_qa_ts(defect)
        self.assertEqual([x.position for x in a.events],
                         [x.position for x in b.events])

    def test_infeasible_group_limited_budget_fails_closed(self):
        records = [TapeRecord(i, "one-group", i % 5 + 1, f"image-{i}")
                   for i in range(100)]
        with self.assertRaisesRegex(Exception, "infeasible"):
            build_qa_ts(records)


if __name__ == "__main__": unittest.main()
