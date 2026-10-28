"""R0 safety contract tests: no Real-IAD/GPU required."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.risk_safe.r0_opportunity import (integers, offline_counts, probe_subset, native_output_name)


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


if __name__ == "__main__":
    unittest.main()
