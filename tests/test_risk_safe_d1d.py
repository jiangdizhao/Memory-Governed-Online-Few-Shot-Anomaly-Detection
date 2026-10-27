"""D1-D CPU contract tests. Real DINO/v22 score/update parity runs on GPU."""
from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
import torch

from experiments.risk_safe import native_bridge as b
# native_bridge/shadow_memory configure the immutable experiments import path.
import v02_memory as native_v02_memory
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
        self.device = self.protos.device
        self.dtype = self.protos.dtype

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

    @staticmethod
    def l2_normalize(x, eps=1e-12):
        # Match the exact original v22 operation, retaining input dtype.
        return x / (x.norm(dim=-1, keepdim=True) + eps)


class FakeCfg:
    topk_patches = 32
    image_aggregation = "max"
    image_topk = 64
    image_quantile = .95


class NativeBridgeContracts(unittest.TestCase):
    def setUp(self):
        self.bank = FakeVMB()
        self.impl = FakeImpl()
        self.image = SimpleNamespace(width=32, height=32)
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
            self.impl, self.bank, self.image, self.features, FakeCfg())
        self.assertAlmostEqual(value, 2.0, places=6)
        self.assertEqual(b.snapshot(self.bank).sha256(), before)

    def test_untouched_native_parity(self):
        audit = b.verify_untouched_parity(
            self.impl, self.bank, self.image, self.features, FakeCfg())
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["abs_difference"], 0)

    def test_shadow_score_passes_through_full_native_interface(self):
        current = b.snapshot(self.bank)
        shadow, _ = current.proposed((feature(0, 1),))
        adapter = b.NativeScoreAdapter(self.impl, self.bank, FakeCfg())
        small = (feature(0, 1),)
        original = adapter((self.image, small), current)
        after = adapter((self.image, small), shadow)
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


class NativePrecisionParityRegression(unittest.TestCase):
    """Regression for the user's observed native-shadow diff=7.62939453125e-06."""

    class RealGreedyMethod:
        # Run the actual *unchanged* V0.2 update method against a bank with
        # the same codebook attributes, not a hand-written approximate oracle.
        name = "greedy_coreset"
        update = native_v02_memory.MemoryArm.update

        def __init__(self, bank):
            self.bank = bank
            self.events = self.patches = 0
            self.accepted = self.skipped = 0
            self.update_seconds = 0.

        def _check_budget(self):
            return 0

    def test_half_precision_full_matrix_native_vs_shadow_append(self):
        torch.manual_seed(91032)
        native_bank = FakeVMB(cap=16)
        cb = b.dynamic_codebooks(native_bank)[0]
        cb.protos = cb.protos.half()
        cb.dtype = cb.protos.dtype
        inputs = (torch.randn(32, 4096, dtype=torch.float16),)
        # Match the actual 4096-dimensional concatenated VisionAD features.
        cb.feat_dim = 4096
        cb.protos = inputs[0][:1].clone()
        amap = {"per_layer_summary": {"layer0": {
            "topk_patch_idx": list(range(8, 16))}}}
        arm = self.RealGreedyMethod(native_bank)
        report = b.verify_native_update_parity(
            FakeImpl(), arm, inputs, amap,
            score=.1, theta=.5, sigma=.1, step=7)
        self.assertTrue(report["passed"])
        self.assertEqual(report["greedy"]["appended"], 8)
        self.assertEqual(report["greedy"]["replaced"], 0)

    def test_half_precision_full_matrix_native_vs_shadow_saturated(self):
        torch.manual_seed(91033)
        native_bank = FakeVMB(cap=2)
        cb = b.dynamic_codebooks(native_bank)[0]
        inputs = (torch.randn(32, 4096, dtype=torch.float16),)
        cb.feat_dim = 4096
        cb.protos = FakeImpl.l2_normalize(inputs[0][:2]).detach().clone()
        cb.dtype = cb.protos.dtype
        amap = {"per_layer_summary": {"layer0": {
            "topk_patch_idx": list(range(8, 16))}}}
        arm = self.RealGreedyMethod(native_bank)
        report = b.verify_native_update_parity(
            FakeImpl(), arm, inputs, amap,
            score=.9, theta=.5, sigma=.1, step=7)
        self.assertTrue(report["passed"])
        self.assertEqual(report["greedy"]["appended"], 0)
        self.assertEqual(
            report["greedy"]["replaced"] + report["greedy"]["skipped"], 8)

    def test_normalized_native_candidates_selected_after_full_matrix(self):
        torch.manual_seed(91034)
        bank = FakeVMB(cap=16)
        cb = b.dynamic_codebooks(bank)[0]
        full = (torch.randn(32, 4096, dtype=torch.float16),)
        cb.feat_dim = 4096
        cb.protos = full[0][:1].clone()
        cb.dtype = cb.protos.dtype
        ids = {0: [11, 13, 9, 3]}
        rows = b.normalized_native_candidate_rows(
            FakeImpl(), bank, full, ids, keep=3)
        exact = FakeImpl.l2_normalize(full[0])[ids[0][:3]]
        self.assertTrue(torch.equal(rows[0], exact))
        expected_bank = b.snapshot(bank)
        shadow, audit = expected_bank.proposed(
            rows, already_normalized=True)
        self.assertTrue(torch.equal(
            shadow.layers[0][1:], exact.to(dtype=cb.dtype)))
        self.assertEqual(audit.appended, 3)


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
        with self.assertRaises(SystemExit):
            run.parser().parse_args(["--budget", "8"])

    def test_no_human_label_present_in_native_candidate_interface(self):
        import inspect
        sig = inspect.signature(b.extract_v22_candidates)
        self.assertNotIn("label", sig.parameters)
        self.assertNotIn("defect_mask", sig.parameters)


if __name__ == "__main__":
    unittest.main()
