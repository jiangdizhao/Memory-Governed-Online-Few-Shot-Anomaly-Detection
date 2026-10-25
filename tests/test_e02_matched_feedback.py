"""CPU-only E02 contract regressions. Real GPU run requires Real-IAD locally."""
import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import e02_matched_feedback as e02


class FakeController:
    def __init__(self):
        self.value = .4
        self.seen = []
    def theta(self):
        return self.value
    def observe_labeled(self, score, label):
        self.seen.append((score, label))
        self.value += .01
    def sigmaN(self):
        return .02


class FakeBank:
    def __init__(self):
        self.normal_fixed = {"c": {0: {"id": [SimpleNamespace(
            protos=SimpleNamespace(dummy=True))]}}}
        self.normal_dyn = {"c": {0: {"id": [SimpleNamespace(size=lambda: 0)]}}}
        self.use_defect_bank = False
        self.corr_cfg = SimpleNamespace(enabled=False)
    def score_sample_multiview(self, **kwargs):
        return .2, {"per_layer_summary": {"layer0": {"topk_patch_idx":[1,2]}}}, {}


class E02Tests(unittest.TestCase):
    def test_query_tape_deterministic_and_label_blind(self):
        a = e02.make_query_tape(3815, .03, 37)
        self.assertEqual(len(a), 114)
        self.assertEqual(a, e02.make_query_tape(3815, .03, 37))
        self.assertNotEqual(a, e02.make_query_tape(3815, .03, 38))
        self.assertTrue(all(1 <= t <= 3815 for t in a))
        self.assertFalse(e02.make_query_tape(100, 0, 37))

    def test_all_four_arms_same_initial_bank_and_theta(self):
        class E01:
            @staticmethod
            def memory_profile(bank, cls):
                return {"sha256":"fixed_shared_fingerprint"}
        arms, bank_hash, theta = e02.create_arms(
            FakeBank(), FakeController(), None, E01, "c"
        )
        self.assertEqual(set(arms), set(e02.ARM_FLAGS))
        self.assertEqual(bank_hash, "fixed_shared_fingerprint")
        self.assertEqual(theta, .4)
        self.assertTrue(all(a.controller.theta() == .4 for a in arms.values()))
        self.assertIsNot(arms["static"].bank, arms["stm_only"].bank)
        self.assertFalse(arms["static"].stm or arms["static"].theta_update)
        self.assertTrue(arms["stm_theta"].stm and arms["stm_theta"].theta_update)

    def test_shared_feature_scoring(self):
        class Impl:
            @staticmethod
            def ViewSpec(*args):
                return tuple(args)
        args = SimpleNamespace(topk_patches=32, image_aggregation="max")
        x, amap = e02.frozen_score(
            Impl, SimpleNamespace(bank=FakeBank()), "c",
            ["same_features"], SimpleNamespace(height=100, width=100), args
        )
        self.assertEqual(x, .2)
        self.assertEqual(amap["per_layer_summary"]["layer0"]["topk_patch_idx"], [1,2])

    def test_metrics_and_threshold_independent_auroc(self):
        y = [0,0,1,1]
        x = [.1,.3,.2,.7]
        left = e02.metrics(y, x, [0,0,0,1])
        right = e02.metrics(y, x, [0,1,0,1])
        self.assertEqual(left["auroc"], right["auroc"])
        self.assertNotEqual(left["balanced_accuracy"], right["balanced_accuracy"])

    def test_supervision_occurs_after_all_arm_predictions_in_source(self):
        """Guard the audit's fundamental prequential order at source level."""
        tree = ast.parse((ROOT / "experiments" / "e02_matched_feedback.py").read_text(encoding="utf-8"))
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=="audit")
        target = next(n for n in ast.walk(run) if isinstance(n, ast.For)
                      and isinstance(n.target, ast.Tuple)
                      and n.target.elts and isinstance(n.target.elts[0], ast.Name)
                      and n.target.elts[0].id == "t")
        stmts = target.body
        def pos(name):
            return next(i for i,s in enumerate(stmts) if name in ast.unparse(s))
        self.assertLess(pos("state = {}"), pos("y = 0 if sample.is_good"))
        self.assertLess(pos("y = 0 if sample.is_good"), pos("if asked:"))
        asked_node = next(s for s in stmts if isinstance(s, ast.If)
                          and isinstance(s.test, ast.Name) and s.test.id == "asked")
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                            and n.func.id == "write_verified_normal"
                            for n in ast.walk(asked_node)))

    def test_verified_normal_no_other_label_access(self):
        source = (ROOT / "experiments" / "e02_matched_feedback.py").read_text(encoding="utf-8")
        self.assertIn("if arm.stm and y == 0:", source)
        self.assertIn("if arm.theta_update:", source)
        self.assertIn("arm.controller.observe_labeled(state[name][\"score\"], y)", source)


if __name__ == "__main__":
    unittest.main()
