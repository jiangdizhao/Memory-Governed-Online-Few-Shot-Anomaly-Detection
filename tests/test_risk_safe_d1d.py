"""D1-D CPU contract tests. Real DINO/v22 score/update parity runs on GPU."""
from __future__ import annotations

import copy
import unittest
import torch

from experiments.risk_safe import native_bridge as b
from experiments.risk_safe import d1d_native_smoke as run
from experiments.risk_safe import d1b_protocol as d1b
from experiments.risk_safe.anchor_store import VerifiedImage
from experiments.risk_safe.shadow_memory import GreedyBank


def feature(*xs):
    return torch.tensor([list(xs)], dtype=torch.float32)


class Codebook:
    def __init__(self, features, cap):
        self.protos = features.detach().clone()
        self.K = cap
        self.feat_dim = int(features.shape[1])

    def size(self):
        return int(self.protos.shape[0])


class FakeVMB:
    """Minimal native bank with a genuine score_sample_multiview interface."""
    def __init__(self, cap=3):
        self.normal_dyn = {"p0": {0: {"id": [Codebook(feature(1, 0), cap)]}}}
        self.normal_dyn_ltm = {"p0": {0: {"id": [Codebook(
            torch.empty(0, 2), 1)]}}}
        self.normal_dyn_candidates = {"p0": {0: {"id": [Codebook(
            torch.empty(0, 2), 1)]}}}

    def score_sample_multiview(self, *, view_to_feats, **kwargs):
        x = view_to_feats["id"][0].float()
        normal = self.normal_dyn["p0"][0]["id"][0].protos.float()
        fixed = feature(1, 0)
        all_normals = torch.cat((fixed, normal), 0)
        x = torch.nn.functional.normalize(x, dim=1)
        normal = torch.nn.functional.normalize(all_normals, dim=1)
        out = float((1 - x @ normal.T).min(dim=1).values.max().item())
        amap = {"per_layer_summary": {"layer0": {
            "topk_patch_idx": list(range(len(x)))} }}
        return out, amap, {}


class FakeImpl:
    @staticmethod
    def ViewSpec(*args):
        return args

    @staticmethod
    def layer_to_topk_from_amap(amap):
        return {0: list(amap["per_layer_summary"]["layer0"]["topk_patch_idx"])}

    @staticmethod
    def select_informative_normal_write(**kwargs):
        return True, 8, "easy_but_keep_small", {}


class FakeCfg:
    topk_patches = 32
    image_aggregation = "max"
    image_topk = 64
    image_quantile = .95


class NativeBridgeContracts(unittest.TestCase):
    def setUp(self):
        self.bank = FakeVMB()
        self.impl = FakeImpl()
        self.features = (torch.tensor([[0., 1.], [-1., 0.]]),)

    def test_native_snapshot_and_independent_clone(self):
        before = b.snapshot(self.bank)
        native_shadow = b.install_on_clone(self.bank, before)
        self.assertIsNot(self.bank, native_shadow)
        self.assertEqual(b.snapshot(native_shadow).sha256(), before.sha256())
        cb = native_shadow.normal_dyn["p0"][0]["id"][0]
        cb.protos[0, 0] = 50
        self.assertEqual(b.snapshot(self.bank).sha256(), before.sha256())

    def test_native_score_uses_unmodified_native_api(self):
        before = b.snapshot(self.bank).sha256()
        value, _ = b.native_score(
            self.impl, self.bank, None, self.features, FakeCfg())
        self.assertAlmostEqual(value, 1.0, places=6)
        self.assertEqual(b.snapshot(self.bank).sha256(), before)

    def test_untouched_native_parity(self):
        audit = b.verify_untouched_parity(
            self.impl, self.bank, None, self.features, FakeCfg())
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["abs_difference"], 0)

    def test_shadow_score_passes_through_full_native_interface(self):
        current = b.snapshot(self.bank)
        shadow, _ = current.proposed((feature(0, 1),))
        adapter = b.NativeScoreAdapter(self.impl, self.bank, FakeCfg())
        original = adapter((None, self.features), current)
        after = adapter((None, self.features), shadow)
        self.assertGreater(original, after)
        self.assertEqual(b.snapshot(self.bank).sha256(), current.sha256())

    def test_native_commit_fails_on_version_or_stale_digest(self):
        snap = b.snapshot(self.bank)
        shadow, _ = snap.proposed((feature(0, 1),))
        with self.assertRaises(b.NativeParityError):
            b.install_on_live(self.bank, shadow,
                              expected_digest="wrong", version=0)
        self.assertEqual(b.snapshot(self.bank).sha256(), snap.sha256())
        b.install_on_live(self.bank, shadow,
                          expected_digest=snap.sha256(), version=0)
        self.assertEqual(b.snapshot(self.bank, version=1).sha256(),
                         shadow.sha256())

    def test_malformed_feature_dimensions_fail_closed(self):
        snap = b.snapshot(self.bank)
        bad = GreedyBank((feature(1, 0, 0),), (3,))
        with self.assertRaises(b.NativeParityError):
            b.install_on_clone(self.bank, bad)
        self.assertEqual(b.snapshot(self.bank).sha256(), snap.sha256())

    def test_explicit_score_parity_tolerance(self):
        self.assertAlmostEqual(b.parity_tolerance(1, 1 + 1e-6), 1e-6)
        with self.assertRaises(b.NativeParityError):
            b.parity_tolerance(1, 2)

    def test_candidate_indices_are_native_amap_indices(self):
        amap = {"per_layer_summary": {"layer0": {"topk_patch_idx": [1, 0]}}}
        cand, indices = b.extract_v22_candidates(
            self.impl, amap, self.features, keep=1)
        self.assertEqual(indices[0], [1, 0])
        torch.testing.assert_close(cand[0], self.features[0][1:2])

    def test_missing_native_candidates_are_not_fabricated(self):
        with self.assertRaises(b.NativeParityError):
            b.extract_v22_candidates(
                self.impl, {"per_layer_summary": {"layer0": {
                    "topk_patch_idx": []}}}, self.features, keep=8)


class NativeQueryLedgerContracts(unittest.TestCase):
    def test_strictly_prior_verified_image_only(self):
        ledger = run.NativeLedger()
        normal = VerifiedImage("q1", "im1", "sample1", 1,
                               "normal", 5, 5)
        defect = VerifiedImage("q2", "im2", "sample2", 3,
                               "defect", 10, 10)
        ledger.add(normal, None, (feature(1, 0),))
        ledger.add(defect, None, (feature(0, 1),))
        self.assertEqual((len(ledger.prior(5)[0]), len(ledger.prior(5)[1])),
                         (0, 0))
        self.assertEqual((len(ledger.prior(10)[0]), len(ledger.prior(10)[1])),
                         (1, 0))
        self.assertEqual((len(ledger.prior(11)[0]), len(ledger.prior(11)[1])),
                         (1, 1))
        with self.assertRaises(b.NativeParityError):
            ledger.add(normal, None, (feature(1, 0),))

    def test_replay_configuration_uses_frozen_q_a_not_closed_loop(self):
        p = run.parser()
        args = p.parse_args(["--json-dir", "G:/dataset/realiad/jsons/realiad_jsons"])
        conf = run.config(args)
        self.assertEqual(conf["budget"], 16)
        self.assertEqual(conf["query_seed"], 37)
        self.assertEqual(conf["max_query_events"], 4)
        self.assertEqual(conf["stream"],
                         "Q-A frozen random queried-event condensed replay")
        self.assertEqual(conf["sidecar"], "off")
        self.assertEqual(conf["signed_correction"], "off")
        self.assertEqual(conf["theta"], "strict4 LOSO frozen")
        self.assertEqual(d1b.MAIN_BUDGETS, (16, 32, 64, 79))

    def test_frozen_gates_forbid_unregistered_scientific_setting(self):
        args = run.parser().parse_args(["--budget", "8"])
        self.assertNotIn(args.budget, d1b.MAIN_BUDGETS)

    def test_no_human_label_present_in_native_candidate_interface(self):
        import inspect
        sig = inspect.signature(b.extract_v22_candidates)
        self.assertNotIn("label", sig.parameters)
        self.assertNotIn("defect_mask", sig.parameters)


if __name__ == "__main__":
    unittest.main()
