import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.calibration.anchor_ledger import AnchorLedger, AnchorLedgerError
from experiments.calibration.controllers import FrozenTheta, RollingQ90QueriedNormal
from experiments.calibration.engine import PrequentialEngine
from experiments.calibration.metrics import PrivateOutcomeLedger
from experiments.calibration.protocol import (EvidenceAnchor, EvidenceSnapshot,
    ScoreResult, StreamEvent, TapeRecord)
from experiments.calibration.query_tape import build_qa_ts
from experiments.risk_safe.shadow_memory import GreedyBank


class EngineLedgerTests(unittest.TestCase):
    def _run(self, labels, theta=.4):
        records = [TapeRecord(i, f"s-{i}", i % 5 + 1, f"r-{i}") for i in range(80)]
        tape = build_qa_ts(records)
        memory = {"version": 0, "digest": "initial"}
        events = [StreamEvent(record, i / 100.0) for i, record in enumerate(records)]
        controllers = [FrozenTheta(theta), RollingQ90QueriedNormal(theta, window=10, min_count=2)]
        def score_fn(event):
            score = event.payload + memory["version"] * .001
            return ScoreResult(score, memory["version"], f"score-state-{memory['version']}")
        def write_fn(event, score, feedback):
            if feedback.queried and feedback.label == 0:
                memory["version"] += 1
                memory["digest"] = f"m-{memory['version']}"
            return None
        def evidence_fn(event, feedback):
            c64 = tuple(EvidenceAnchor(f"c64-{i}", i / 100.0 + memory["version"] * .001,
                                       memory["version"], True)
                        for i in range(64))
            return EvidenceSnapshot(memory["version"], c64, (), {})
        result = PrequentialEngine(tape=tape, events=events,
            outcomes=PrivateOutcomeLedger(labels), controllers=controllers,
            score_fn=score_fn,
            memory_state_fn=lambda: (memory["version"], memory["digest"]),
            write_fn=write_fn, evidence_fn=evidence_fn,
            provenance_id="unit-test-stream-v1").run()
        return result, memory

    def test_predictions_share_one_score_and_memory_trajectory(self):
        labels = [i % 2 for i in range(80)]
        result, memory = self._run(labels)
        self.assertEqual(len(result["scores"]), 80)
        self.assertEqual(len(set(x["score_state_digest"] for x in result["trace"])),
                         len(set(x["memory_version_before"] for x in result["trace"])))
        self.assertEqual(result["trace"][0]["theta_pre"]["B0"], .4)
        self.assertEqual(result["query_audit"]["n_queried_groups"],
                         result["query_audit"]["n_queried_images"])

    def test_threshold_changes_do_not_change_shared_score_or_memory(self):
        labels = [i % 2 for i in range(80)]
        low, memory_low = self._run(labels, theta=.1)
        high, memory_high = self._run(labels, theta=.9)
        self.assertEqual(low["scores"], high["scores"])
        self.assertEqual(low["tape_digest"], high["tape_digest"])
        self.assertEqual(memory_low, memory_high)

    def test_tape_is_independent_of_swapped_private_labels(self):
        records = [TapeRecord(i, f"s-{i}", 1, f"r-{i}") for i in range(80)]
        self.assertEqual(build_qa_ts(records).digest, build_qa_ts(records).digest)
        labels_a = [i % 2 for i in range(80)]
        labels_b = [1 - x for x in labels_a]
        a, _ = self._run(labels_a)
        b, _ = self._run(labels_b)
        self.assertEqual(a["tape_digest"], b["tape_digest"])

    def test_unqueried_label_swap_cannot_change_controller_or_memory_trajectory(self):
        records = [TapeRecord(i, f"s-{i}", i % 5 + 1, f"r-{i}") for i in range(80)]
        queried = build_qa_ts(records).positions
        labels_a = [i % 2 for i in range(80)]
        labels_b = [1 - y if i not in queried else y
                    for i, y in enumerate(labels_a)]
        result_a, memory_a = self._run(labels_a)
        result_b, memory_b = self._run(labels_b)
        self.assertEqual(result_a["scores"], result_b["scores"])
        self.assertEqual(result_a["predictions"], result_b["predictions"])
        self.assertEqual(memory_a, memory_b)

    def test_anchor_lso_removes_all_matching_rows_across_layers(self):
        anchors = [(f"c64-{i}", i % 5 + 1, i) for i in range(64)]
        ledger = AnchorLedger(anchors, deployed_support_keys=[f"support-{i}" for i in range(4)])
        ledger.add_queried(source_key="stream-a", camera_id=1, label=0,
                           features="normal-features", t=1, memory_version=1)
        bank = GreedyBank([
            torch.tensor([[1., 0.], [0., 1.], [1., 1.]]),
            torch.tensor([[1., 0.], [1., 1.]])], [256, 256], version=1)
        sources = [["stream-a", "stream-b", "stream-a"], ["stream-a", "stream-c"]]
        observed_sizes = []
        def scorer(features, memory):
            observed_sizes.append(tuple(len(x) for x in memory.layers))
            return float(sum(len(x) for x in memory.layers))
        evidence = ledger.refresh(memory=bank, source_rows=sources, scorer=scorer)
        query = next(a for a in evidence.normal_anchors if not a.protected)
        self.assertEqual(query.source_excluded_rows, 3)
        self.assertEqual(observed_sizes[-1], (1, 1))
        self.assertEqual(sum(a.protected for a in evidence.normal_anchors), 64)
        ledger.assert_current(evidence, 1)
        with self.assertRaisesRegex(AnchorLedgerError, "stale"):
            ledger.assert_current(evidence, 2)

    def test_protected_c64_does_not_accept_stream_overlap(self):
        ledger = AnchorLedger([(f"c64-{i}", 1, None) for i in range(64)],
                              deployed_support_keys=[f"support-{i}" for i in range(4)])
        with self.assertRaisesRegex(AnchorLedgerError, "overlaps"):
            ledger.add_queried(source_key="c64-0", camera_id=1, label=0,
                               features=None, t=1, memory_version=0)

    def test_evaluator_marks_missing_classes_and_abstentions_unavailable(self):
        metrics = PrivateOutcomeLedger([0, 0, 1]).evaluate(
            [.1, .2, .3], {"tri_state": [0, None, 1]})["methods"]["tri_state"]
        self.assertIsNone(metrics["balanced_accuracy"])
        self.assertIsNone(metrics["fpr"])
        self.assertEqual(metrics["normal_abstentions"], 1)
        absent = PrivateOutcomeLedger([0, 0]).evaluate(
            [.1, .2], {"frozen": [0, 1]})["methods"]["frozen"]
        self.assertIsNone(absent["fnr"])
        self.assertEqual(absent["windows"]["windows"][0]["status"],
                         "UNAVAILABLE_CLASS_ABSENT")

    def test_c64_is_additional_and_disjoint_from_exactly_four_supports(self):
        c64 = [(f"id-{i}", 1, None) for i in range(64)]
        with self.assertRaisesRegex(AnchorLedgerError, "four distinct deployed"):
            AnchorLedger(c64, deployed_support_keys=["s0", "s1", "s2"])
        c64[0] = ("support-0", 1, None)
        with self.assertRaisesRegex(AnchorLedgerError, "disjoint"):
            AnchorLedger(c64, deployed_support_keys=[f"support-{i}" for i in range(4)])

    def test_anchor_checkpoint_roundtrips_private_features(self):
        rows = [(f"c64-{i}", 1, torch.tensor([float(i)])) for i in range(64)]
        supports = [f"support-{i}" for i in range(4)]
        original = AnchorLedger(rows, deployed_support_keys=supports)
        original.add_queried(source_key="stream-a", camera_id=2, label=1,
                             features=torch.tensor([9.]), t=7, memory_version=0)
        state = original.state_dict()
        restored = AnchorLedger(rows, deployed_support_keys=supports)
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict()["queried_count"], 1)
        self.assertTrue(torch.equal(restored._anchors[0].features, rows[0][2]))
        self.assertTrue(torch.equal(restored._anchors[-1].features, torch.tensor([9.])))

    def test_checkpoint_resume_matches_uninterrupted_run_and_restores_rng(self):
        records = [TapeRecord(i, f"resume-{i}", i % 5 + 1, f"resume-r-{i}")
                   for i in range(100)]
        tape = build_qa_ts(records)
        labels = [i % 3 == 0 for i in range(100)]
        labels = [int(x) for x in labels]
        events = [StreamEvent(record, i / 100.) for i, record in enumerate(records)]

        class Memory:
            def __init__(self): self.version, self.digest = 0, "memory-0"
            def state_dict(self):
                return {"version": self.version, "digest": self.digest}
            def load_state_dict(self, state):
                self.version, self.digest = int(state["version"]), str(state["digest"])

        def make_engine(memory):
            controllers = [FrozenTheta(.4), RollingQ90QueriedNormal(.4, 16, 2)]
            def score_fn(event):
                return ScoreResult(float(event.payload) + np.random.random() * 1e-6 +
                                   memory.version * .001, memory.version,
                                   f"score-state-{memory.version}")
            def write_fn(event, score, feedback):
                if feedback.queried and feedback.label == 0:
                    memory.version += 1
                    memory.digest = f"memory-{memory.version}"
            return PrequentialEngine(
                tape=tape, events=events,
                outcomes=PrivateOutcomeLedger(labels), controllers=controllers,
                score_fn=score_fn,
                memory_state_fn=lambda: (memory.version, memory.digest),
                write_fn=write_fn,
                checkpoint_components={"memory": memory},
                provenance_id="synthetic-resume-test-v1")

        np.random.seed(791)
        uninterrupted_memory = Memory()
        uninterrupted = make_engine(uninterrupted_memory).run()
        np.random.seed(791)
        first_memory = Memory()
        first = make_engine(first_memory)
        partial = first.run(max_events=37)
        self.assertEqual(partial["status"], "INCOMPLETE")
        resumed_memory = Memory()
        resumed = make_engine(resumed_memory)
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "engine.pt"
            first.save_checkpoint(checkpoint_path)
            resumed.load_checkpoint(checkpoint_path)
            continued = resumed.run()
        self.assertEqual(continued["status"], "COMPLETE")
        self.assertEqual(continued["scores"], uninterrupted["scores"])
        self.assertEqual(continued["predictions"], uninterrupted["predictions"])
        self.assertEqual(continued["trace"], uninterrupted["trace"])
        self.assertEqual(resumed_memory.state_dict(), uninterrupted_memory.state_dict())

        changed_labels = labels.copy()
        changed_labels[0] = 1 - changed_labels[0]
        mismatch = PrequentialEngine(
            tape=tape, events=events, outcomes=PrivateOutcomeLedger(changed_labels),
            controllers=[FrozenTheta(.4), RollingQ90QueriedNormal(.4, 16, 2)],
            score_fn=lambda event: ScoreResult(event.payload, 0, "constant"),
            memory_state_fn=lambda: (0, "memory-0"), write_fn=lambda *args: None,
            provenance_id="synthetic-resume-test-v1")
        with self.assertRaisesRegex(Exception, "identity mismatch"):
            with tempfile.TemporaryDirectory() as temporary_dir:
                checkpoint_path = Path(temporary_dir) / "engine.pt"
                first.save_checkpoint(checkpoint_path)
                mismatch.load_checkpoint(checkpoint_path)


if __name__ == "__main__": unittest.main()
