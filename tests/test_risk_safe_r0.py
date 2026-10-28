"""R0 safety contract tests: no Real-IAD/GPU required."""
from __future__ import annotations

import json
import tempfile
from unittest.mock import patch
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.risk_safe.r0_opportunity import (integers, offline_counts, probe_subset,
                                                    native_output_name, frozen_r0_tape,
                                                    R0TapeIntegrityError)
from experiments.risk_safe import d1b_protocol as d1b
from experiments.risk_safe import query_protocol as qp


class R0ContractTests(unittest.TestCase):
    def test_budget_integer_validation(self):
        self.assertEqual(integers("16,32,64,128"), (16, 32, 64, 128))
        self.assertEqual(integers("0,1,2", allow_zero=True), (0, 1, 2))
        for value in ("", "16,16", "0,16", "-1,4", "16,x"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    integers(value)

    def test_historical_anchor_order_excludes_current_query(self):
        events = [SimpleNamespace(path=f"p{i}", position=i,
                                  sample_key=f"key{i}", camera_id=1)
                  for i in range(1, 7)]
        labels = {"p1": True, "p2": False, "p3": True,
                  "p4": False, "p5": True, "p6": True}
        d = offline_counts(events, labels, anchors_needed=(1, 2))
        self.assertEqual(d["n_verified_normal"], 4)
        self.assertEqual(d["n_verified_defect"], 2)
        self.assertEqual(d["n_normal_with_prior_defect"], {"1": 3, "2": 2})
        self.assertEqual(d["first_verified_defect_step"], 2)
        self.assertEqual(d["events"][1]["prior_defect"], 0)
        self.assertEqual(d["events"][2]["prior_defect"], 1)
        self.assertEqual(d["events"][3]["prior_defect"], 1)
        self.assertEqual(d["events"][4]["prior_defect"], 2)

    def test_missing_authoritative_label_fails_closed(self):
        event = [SimpleNamespace(path="p1", position=1,
                                 sample_key="s", camera_id=2)]
        with self.assertRaises(RuntimeError):
            offline_counts(event, {})

    def test_probe_group_selection_uses_per_image_labels(self):
        rows = []
        for group, labels in [("a", (True,True)),
                              ("b", (False,True)),
                              ("c", (True,True)),
                              ("d", (False,False))]:
            for i, good in enumerate(labels):
                rows.append(SimpleNamespace(sample_key=group,
                                            is_good=good,
                                            rel_path=f"{group}/{i}",
                                            camera_id=i+1))
        plan = SimpleNamespace(probe_a=tuple(rows),
                               probe_a_samples=("a","b","c","d"))
        selected = probe_subset(plan, 1)
        self.assertEqual({r.sample_key for r in selected}, {"a", "b"})
        self.assertEqual([not r.is_good for r in selected
                          if r.sample_key == "b"], [True, False])

    def test_output_name_never_aliases_smoke_and_full(self):
        base = dict(out_dir=None, capacities="64,128", class_name="audiojack",
                    support_seed=0, query_seed=37, budget=128, audit_every=1)
        smoke = native_output_name(SimpleNamespace(**base, max_query_events=4))
        full = native_output_name(SimpleNamespace(**base, max_query_events=0))
        self.assertNotEqual(smoke, full)
        self.assertIn("smoke4", str(smoke))
        self.assertIn("full", str(full))
        q90 = native_output_name(SimpleNamespace(**base, max_query_events=0,
                                                 theta_mode="train64-q90"))
        self.assertNotEqual(full, q90)
        self.assertIn("train64-q90", str(q90))


    def test_r0_b128_extension_validates_all_archived_main_budgets(self):
        # Synthetic label-free stream, one Q-A query per official sample ID.
        # No Real-IAD, CUDA, or private v22 scoring required.
        stream = tuple(qp.StreamImage(
            position=i+1, sample_key=f"sample_{i//5:04d}",
            path=f"image_{i}", rel_path=f"image_{i}",
            camera_id=(i % 5)+1, timestamp="20261028000000",
            phase=1, stream_order="random") for i in range(150*5))
        universe = tuple(sorted(set(e.sample_key for e in stream)))
        priority = d1b.shared_query_priority(universe, "audiojack", 37)
        legacy = d1b.build_paired_tapes(
            stream, d1b.MAIN_BUDGETS, 37, "audiojack", priority)
        archived = {}
        for budget, tape in legacy.items():
            archived[str(budget)] = dict(
                digest=tape.digest, stream_digest=tape.stream_digest,
                N_queries=len(tape.events),
                query_positions=list(tape.positions),
                events_sha256=qp.sha256_of([e.as_manifest() for e in tape.events]),
                selected_images_sha256=qp.sha256_of(list(tape.rel_paths)),
                selected_samples_sha256=qp.sha256_of(list(tape.sample_keys)))
        base = dict(schema="risk_safe.d1b.query_manifest.v1",
                    query_seeds=[37], budgets=list(d1b.MAIN_BUDGETS),
                    class_records={"audiojack": {"random": {"37": archived}}})
        fake_plan = SimpleNamespace(phases=tuple())
        with tempfile.TemporaryDirectory() as temp, patch(
                "experiments.risk_safe.r0_opportunity.d1b.build_ordered_stream",
                return_value=stream):
            path = Path(temp) / d1b.OUTPUT_QUERIES
            def save(value):
                path.write_text(json.dumps(value), encoding="utf-8")
            save(base)
            tape, n_images, audit = frozen_r0_tape(
                fake_plan, class_name="audiojack", query_seed=37,
                budget=128, d1b_manifest_dir=temp)
            self.assertEqual(n_images, 750)
            self.assertEqual(len(tape.events), 128)
            self.assertEqual(len(set(tape.sample_keys)), 128)
            self.assertEqual(audit["source"], "derived_approved_r0_extension")
            self.assertEqual(audit["historical_validated_budgets"], [16,32,64,79])
            self.assertTrue(audit["matched_all_archived_tapes"])
            self.assertTrue(audit["nested_with_all_archived_tapes"])
            # A previously archived budget must still have the exact old digest.
            previous, _, previous_audit = frozen_r0_tape(
                fake_plan, class_name="audiojack", query_seed=37,
                budget=16, d1b_manifest_dir=temp)
            self.assertEqual(previous.digest, archived["16"]["digest"])
            self.assertEqual(previous_audit["source"], "archived_d1b")
            # The historical manifest must fail closed if tampered.
            corrupted = json.loads(json.dumps(base))
            corrupted["class_records"]["audiojack"]["random"]["37"]["16"]["digest"] = "bad"
            save(corrupted)
            with self.assertRaises(R0TapeIntegrityError):
                frozen_r0_tape(fake_plan, class_name="audiojack", query_seed=37,
                               budget=128, d1b_manifest_dir=temp)
            # A missing frozen PRIMARY budget must never become a new tape.
            missing = json.loads(json.dumps(base))
            missing["budgets"].remove(79)
            del missing["class_records"]["audiojack"]["random"]["37"]["79"]
            save(missing)
            with self.assertRaises(R0TapeIntegrityError):
                frozen_r0_tape(fake_plan, class_name="audiojack", query_seed=37,
                               budget=128, d1b_manifest_dir=temp)


if __name__ == "__main__":
    unittest.main()
