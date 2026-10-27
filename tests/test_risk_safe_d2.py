"""D2 CPU unit tests. GPU native replay must be run separately on Windows."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import tempfile
import unittest
import torch

from experiments.risk_safe import d2_native_replay as runner
from experiments.risk_safe.d2_baselines import (
    BASELINES, BaselineState, candidate_novelty, propose_baseline)
from experiments.risk_safe.d2_metrics import (
    image_metrics, probe_deltas, risk_prediction_pairs)
from experiments.risk_safe.shadow_memory import GreedyBank
from experiments.risk_safe.native_bridge import NativeParityError
from experiments.risk_safe import preflight as pf


def row(x, y):
    return torch.tensor([[float(x), float(y)]], dtype=torch.float32)


class FakeCB:
    def __init__(self, rows):
        self.protos = rows.clone()


class NativeBank:
    def __init__(self):
        self.normal_fixed = {"p0": {0: {"id": [FakeCB(row(1, 0))]}}}
        self.normal_dyn = {"p0": {0: {"id": [FakeCB(torch.empty(0, 2))]}}}


def bank(cap=3):
    return GreedyBank((torch.empty(0, 2),), (cap,))


class AdmissionBaselines(unittest.TestCase):
    def test_shared_methods_are_registered(self):
        self.assertEqual(BASELINES,
            ("static", "append", "greedy", "reservoir", "causal_novelty"))

    def test_static_never_writes(self):
        original = bank()
        state = BaselineState("static", seed=7)
        out = propose_baseline(state, original, (row(0, 1),))
        self.assertEqual(out.decision, "reject")
        self.assertEqual(original.sha256(), out.bank.sha256())

    def test_append_strict_event_capacity_and_nonmutation(self):
        original = bank(2)
        state = BaselineState("append", seed=7)
        one = propose_baseline(state, original, (
            torch.tensor([[1., 0.], [0., 1.]]),))
        self.assertEqual(one.decision, "admit")
        self.assertEqual(one.added, 2)
        self.assertEqual(len(original.layers[0]), 0)
        original = one.bank
        old = original.sha256()
        two = propose_baseline(state, original, (row(-1, 0),))
        self.assertEqual(two.reason, "append_capacity_exceeded")
        self.assertEqual(old, original.sha256())
        self.assertEqual(old, two.bank.sha256())

    def test_greedy_uses_original_v02_slot_semantics(self):
        original = bank(2)
        state = BaselineState("greedy", seed=9)
        added = propose_baseline(state, original, (
            torch.tensor([[1., 0.], [0.99, 0.1]]),))
        self.assertEqual(added.added, 2)
        out = propose_baseline(state, added.bank, (row(0, 1),))
        self.assertEqual(out.added, 0)
        self.assertEqual(out.replaced, 1)
        self.assertEqual(len(out.bank.layers[0]), 2)

    def test_reservoir_bounded_no_static_reference_changes(self):
        native = NativeBank()
        fixed = native.normal_fixed["p0"][0]["id"][0].protos.clone()
        original = bank(2)
        state = BaselineState("reservoir", seed=4)
        for j in range(30):
            c = row(1, 0) if j % 2 else row(0, 1)
            out = propose_baseline(state, original, (c,))
            state.record(out)
            if out.decision == "admit":
                original = out.bank
        self.assertLessEqual(len(original.layers[0]), 2)
        self.assertEqual(state.seen, [30])
        self.assertTrue(torch.equal(
            native.normal_fixed["p0"][0]["id"][0].protos, fixed))

    def test_causal_novelty_no_future_labels_and_history_only(self):
        live = NativeBank()
        st = BaselineState("causal_novelty", seed=9)
        original = bank(3)
        first = propose_baseline(
            st, original, (row(0, 1),),
            raw_candidate_layers=(row(0, 1),), native_bank=live)
        self.assertEqual(first.decision, "admit")
        self.assertIsNone(first.novelty_cutoff)
        self.assertEqual(len(st.prior_novelties), 1)
        previous = st.prior_novelties[0]
        second = propose_baseline(
            st, original, (row(1, 0),),
            raw_candidate_layers=(row(1, 0),), native_bank=live)
        self.assertEqual(second.novelty_cutoff, previous)
        self.assertEqual(second.decision, "reject")
        self.assertEqual(len(st.prior_novelties), 2)

    def test_no_unverified_novelty_lookup(self):
        st = BaselineState("causal_novelty", seed=1)
        with self.assertRaises(NativeParityError):
            propose_baseline(st, bank(), (row(0, 1),))


class ProbeAndRiskMetricTests(unittest.TestCase):
    def test_image_level_fpr_fnr_and_auc(self):
        metric = image_metrics([0, 0, 1, 1],
                               [0.1, 0.8, 0.5, 0.9], theta=.6)
        self.assertEqual((metric["fp"], metric["fn"]), (1, 1))
        self.assertAlmostEqual(metric["fpr"], .5)
        self.assertAlmostEqual(metric["fnr"], .5)

    def test_probe_raw_erosion_not_equivalent_to_fnr_delta(self):
        reference = [0.1, 0.2, 0.9, 1.0]
        updated = [0.1, 0.2, 0.8, 0.95]
        old_m = image_metrics([0, 0, 1, 1], reference, .7)
        new_m = image_metrics([0, 0, 1, 1], updated, .7)
        d = probe_deltas(updated, [0, 0, 1, 1],
                         reference, new_m, old_m)
        self.assertGreater(d["defect_erosion_positive_mean"], 0)
        self.assertEqual(d["fnr_delta"], 0.)

    def test_risk_calibration_refuses_very_small_samples(self):
        self.assertEqual(risk_prediction_pairs([
            {"defect_risk": None, "independent_probe_erosion": .2},
            {"defect_risk": .5, "independent_probe_erosion": .1},
        ])["status"], "insufficient_evidence")

    def test_distinct_risk_predictors_report_exploratory_only(self):
        x = risk_prediction_pairs([
            {"defect_risk": .1, "independent_probe_erosion": .2},
            {"defect_risk": .3, "independent_probe_erosion": .6},
            {"defect_risk": .5, "independent_probe_erosion": .9},
        ])
        self.assertEqual(x["status"], "descriptive_only_not_significance_test")
        self.assertGreater(x["spearman"], .9)


class ProtocolGuardTests(unittest.TestCase):
    def test_default_gpu_pilot_and_registered_budget(self):
        args = runner.parser().parse_args([
            "--json-dir", "G:/dataset/realiad/jsons/realiad_jsons"])
        config = runner.configuration(args)
        self.assertEqual(config["budget"], 16)
        self.assertEqual(config["max_query_events"], 8)
        self.assertEqual(config["probe_checkpoints"], [0, 875, 1750, 2625])
        self.assertEqual(len(config["methods"]), 6)
        self.assertEqual(config["correction"], "disabled")
        self.assertEqual(config["stream_mode"], "random")
        self.assertTrue(config["evaluation_only_probe_prohibited_from_online_policies"])

    def test_probe_sampler_keeps_both_labels_image_level(self):
        @dataclass
        class Rec:
            sample_key: str
            camera_id: int
            is_good: bool
        selected = [
            Rec("normal-01", i, True) for i in range(1, 6)
        ] + [
            Rec("defect-01", i, i != 3) for i in range(1, 6)
        ]
        from types import SimpleNamespace
        plan = SimpleNamespace(probe_a=tuple(selected))
        records, keys = runner.probe_records(plan, 1)
        self.assertEqual(len(records), 10)
        self.assertEqual(set(keys), {"normal-01", "defect-01"})
        self.assertEqual(sum(not x.is_good for x in records), 1)

    def test_output_csv_stable_and_no_probe_feedback_coupling(self):
        s = runner.csv_content(("t", "label"), [
            {"t": 1, "label": "normal"},
            {"t": 2, "label": "defect"},
        ])
        self.assertEqual(s, "t,label\n1,normal\n2,defect\n")
        self.assertNotIn("probe_label", runner.configuration(
            runner.parser().parse_args(["--json-dir", "x"])))


if __name__ == "__main__":
    unittest.main()
