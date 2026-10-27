"""D1-B CPU-only protocol tests (no Real-IAD images, GPU or DINOv2)."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for loc in (ROOT, ROOT / "experiments", ROOT / "tests"):
    if str(loc) not in sys.path:
        sys.path.insert(0, str(loc))

from experiments.risk_safe import preflight as pf
from experiments.risk_safe import query_protocol as qp
from experiments.risk_safe import d1b_protocol as d1b
from test_risk_safe_d1a import cached_plan_and_stream, cached_labels

CLASS = "audiojack"


class SampleOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.acquisition = cached_plan_and_stream()
        cls.labels = cached_labels()
        cls.streams = {mode: d1b.build_ordered_stream(cls.plan.phases, mode, CLASS)
                       for mode in d1b.ORDERS}

    def test_same_membership_and_phase_cohorts_for_all_modes(self):
        samples = {mode: {x.sample_key for x in stream}
                   for mode, stream in self.streams.items()}
        images = {mode: {x.rel_path for x in stream}
                  for mode, stream in self.streams.items()}
        self.assertEqual(len({frozenset(x) for x in samples.values()}), 1)
        self.assertEqual(len({frozenset(x) for x in images.values()}), 1)
        for phase in (1, 2, 3):
            cohorts = {mode: frozenset(x.sample_key for x in st if x.phase == phase)
                       for mode, st in self.streams.items()}
            self.assertEqual(len(set(cohorts.values())), 1)

    def test_random_per_phase_does_not_read_labels(self):
        flipped = tuple(tuple(replace(r, is_good=not r.is_good) for r in phase)
                        for phase in self.plan.phases)
        original = self.streams["random"]
        modified = d1b.build_ordered_stream(flipped, "random", CLASS)
        self.assertEqual(qp.stream_digest(original), qp.stream_digest(modified))
        self.assertEqual([x.rel_path for x in original],
                         [x.rel_path for x in modified])

    def test_synthetic_stress_does_depend_on_offline_labels(self):
        flipped = tuple(tuple(replace(r, is_good=not r.is_good) for r in phase)
                        for phase in self.plan.phases)
        for mode in d1b.STRESS_ORDERS:
            before = d1b.build_ordered_stream(self.plan.phases, mode, CLASS)
            after = d1b.build_ordered_stream(flipped, mode, CLASS)
            self.assertNotEqual(qp.stream_digest(before), qp.stream_digest(after))

    def test_each_synthetic_sample_is_contiguous_with_five_cameras(self):
        for mode in ("random", *d1b.STRESS_ORDERS):
            st = self.streams[mode]
            self.assertEqual(len(st) % 5, 0)
            for index in range(0, len(st), 5):
                group = st[index:index + 5]
                self.assertEqual(len({x.sample_key for x in group}), 1)
                self.assertEqual([x.camera_id for x in group], [1, 2, 3, 4, 5])

    def test_normal_first_and_defect_first_are_phase_local(self):
        for mode in ("normal_first", "defect_first"):
            st = self.streams[mode]
            for phase in (1, 2, 3):
                samples = [st[i:i + 5] for i in range(0, len(st), 5)
                           if st[i].phase == phase]
                has_defect = [any(not self.labels[x.path] for x in sample)
                              for sample in samples]
                self.assertIn(True, has_defect)
                self.assertIn(False, has_defect)
                self.assertEqual(has_defect, sorted(
                    has_defect, reverse=(mode == "defect_first")))

    def test_bursty_has_more_transitions_than_normal_first(self):
        burst = d1b.audit_order(self.streams["bursty"], self.labels)
        normal = d1b.audit_order(self.streams["normal_first"], self.labels)
        self.assertGreater(burst["total_sample_transitions"],
                           normal["total_sample_transitions"])

    def test_acquisition_order_preserves_d1a(self):
        self.assertEqual(qp.stream_digest(self.streams["acquisition_time_blocked"]),
                         qp.stream_digest(self.acquisition))

    def test_unrecognized_order_and_missing_camera_fail(self):
        with self.assertRaises(d1b.D1BProtocolError):
            d1b.build_ordered_stream(self.plan.phases, "future_unknown", CLASS)
        bad = [list(p) for p in self.plan.phases]
        bad[0].pop(0)
        with self.assertRaises(d1b.D1BProtocolError):
            d1b.build_ordered_stream(bad, "random", CLASS)

    def test_stream_order_validation_rejects_mixed_modes(self):
        st = self.streams["random"]
        mixed = list(st)
        mixed[1] = replace(st[1], stream_order="normal_first")
        with self.assertRaises(qp.QueryProtocolError):
            qp.validate_stream(mixed)


class QueryFairnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, _ = cached_plan_and_stream()
        cls.streams = {mode: d1b.build_ordered_stream(cls.plan.phases, mode, CLASS)
                       for mode in d1b.ORDERS}
        cls.universe = tuple(sorted(set(
            key for phase in cls.plan.phase_samples for key in phase)))

    def test_query_identity_fixed_across_all_orders(self):
        for q_seed in d1b.QUERY_SEEDS:
            priority = d1b.shared_query_priority(self.universe, CLASS, q_seed)
            reference = None
            for mode, stream in self.streams.items():
                tapes = d1b.build_paired_tapes(stream, d1b.MAIN_BUDGETS,
                                               q_seed, CLASS, priority)
                self.assertTrue(qp.tape_nesting_report(tapes)["all_nested"])
                current = {
                    budget: {(e.sample_key, e.rel_path, e.camera_id)
                             for e in tape.events}
                    for budget, tape in tapes.items()}
                if reference is None:
                    reference = current
                else:
                    self.assertEqual(current, reference)

    def test_different_query_seeds_have_different_selected_samples(self):
        priorities = [d1b.shared_query_priority(self.universe, CLASS, seed)
                      for seed in d1b.QUERY_SEEDS]
        self.assertEqual(len(set(priorities)), len(priorities))
        self.assertNotEqual(set(priorities[0][:32]), set(priorities[1][:32]))

    def test_selected_sample_set_does_not_change_on_label_flip(self):
        first = d1b.shared_query_priority(self.universe, CLASS, 37)
        flipped = tuple(tuple(replace(r, is_good=not r.is_good) for r in phase)
                        for phase in self.plan.phases)
        second = d1b.shared_query_priority(
            tuple(sorted(set(r.sample_key for p in flipped for r in p))),
            CLASS, 37)
        self.assertEqual(first, second)

    def test_public_prediction_handle_never_contains_real_identity(self):
        st = self.streams["random"]
        public = qp.PublicObservation.from_stream(st[0])
        self.assertNotIn(st[0].sample_key, public.observation_handle)
        self.assertFalse(hasattr(public, "rel_path"))
        self.assertFalse(hasattr(public, "sample_key"))

    def test_protocol_freeze_has_explicit_scientific_boundaries(self):
        p = d1b.protocol_block()
        self.assertEqual(p["budgets"]["primary"], [16, 32, 64, 79])
        self.assertEqual(p["q_a"]["seeds"], [37, 101, 211])
        self.assertEqual(p["stream"]["primary_order"], "random")
        self.assertEqual(p["threshold"]["T1"],
                         "existing v22 GuardedThetaSidecarConfig unchanged")
        self.assertEqual(p["q_b"]["stage"], "interface_and_rule_frozen_not_implemented")
        self.assertTrue(p["q_a"]["preselected_future_ids_are_offline_benchmark_only"])

    def test_source_manifest_and_output_guard(self):
        p = d1b.protocol_block()
        outputs = {d1b.OUTPUT_PROTOCOL: p,
                   d1b.OUTPUT_QUERIES: {"schema": "synthetic"},
                   d1b.OUTPUT_ORDERS: {"schema": "synthetic"}}
        fp = {"json_files_sha256": {}, "N_index_records": 0}
        source = d1b.make_source_manifest(outputs, fp)
        self.assertEqual(source["config_sha256"], qp.sha256_of(p))
        self.assertEqual(set(source["sources_sha256"]), set(d1b.SOURCE_FILES))
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / d1b.OUTPUT_PROTOCOL
            self.assertEqual(pf.write_json_idempotent(file, p), "created")
            self.assertEqual(pf.write_json_idempotent(file, p), "reused")
            self.assertEqual(pf.verify_json_output(file, p), "verified")
            with self.assertRaises(pf.OutputMismatch):
                pf.write_json_idempotent(file, {"changed": True})


if __name__ == "__main__":
    unittest.main()
