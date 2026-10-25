"""Offline E02-R1/R2 tests. No Real-IAD data, torch, GPU or weights required."""
import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import e02_r1_r2 as ext
import e02_matched_feedback as e02


def samples(n):
    return [SimpleNamespace(path=f"/data/{i}.jpg",is_good=(i%3 != 0))
            for i in range(n)]


class R1R2Tests(unittest.TestCase):
    def test_probe_selection_is_deterministic_disjoint_and_label_blind(self):
        initial=samples(100)
        one,probe=ext.split_heldout(initial,16,97)
        again,q=ext.split_heldout(initial,16,97)
        self.assertEqual([x.path for x in probe],[x.path for x in q])
        self.assertEqual([x.path for x in one],[x.path for x in again])
        self.assertTrue(set(x.path for x in one).isdisjoint(x.path for x in probe))
        # Changing all labels may not change selected paths.
        changed=[SimpleNamespace(path=z.path,is_good=not z.is_good) for z in initial]
        _,q2=ext.split_heldout(changed,16,97)
        self.assertEqual([x.path for x in probe],[x.path for x in q2])
        self.assertEqual(ext.checkpoint_steps(84),[0,21,42,63,84])
        self.assertEqual(len(e02.make_query_tape(len(one),.03,37)),3)

    def test_no_probe_overlap_or_invalid_probe_count(self):
        for k in (0,10,-1):
            with self.assertRaises(ValueError):
                ext.split_heldout(samples(10),k,1)
        with self.assertRaises(AssertionError):
            ext.split_heldout([SimpleNamespace(path="x") for _ in range(6)],2,37)

    def test_quantile_requires_verified_normals_and_causality(self):
        c=ext.RollingNormalQuantile(.55, quantile=.9, window=4,
                                    min_normals=2, alpha=.5,max_step=.01)
        self.assertEqual(c.theta(),.55)
        c.observe_labeled(.99,1)  # defects cannot enter normal-score buffer
        self.assertEqual(len(c.history),0)
        self.assertEqual(c.theta(),.55)
        c.observe_labeled(.40,0)
        self.assertEqual(c.theta(),.55)  # min 2 normals
        c.observe_labeled(.41,0)
        self.assertEqual(c.theta(),.54)  # bounded maximum negative step
        self.assertEqual(c.observed,3)
        with self.assertRaises(ValueError):
            c.observe_labeled(float("nan"),0)
        with self.assertRaises(ValueError):
            c.observe_labeled(.50,2)

    def test_mistake_feedback_direction_and_bounds(self):
        c=ext.MistakeFeedback(.50,step=.01,lower_range=.03,upper_range=.02)
        c.observe_labeled(.45,1)  # FN -> lower threshold
        self.assertAlmostEqual(c.theta(),.49)
        c.observe_labeled(.45,0)  # TN -> no update
        self.assertAlmostEqual(c.theta(),.49)
        c.observe_labeled(.60,0)  # FP -> raise
        self.assertAlmostEqual(c.theta(),.50)
        c.observe_labeled(.60,1)  # TP -> no update
        self.assertAlmostEqual(c.theta(),.50)
        for _ in range(200):
            c.observe_labeled(.4,1)
        self.assertGreaterEqual(c.theta(),.47-1e-12)
        self.assertEqual(c.observed,204)

    def test_prequential_query_gating_in_both_runners(self):
        tree=ast.parse((ROOT/"experiments"/"e02_r1_r2.py").read_text(encoding="utf-8"))
        names={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        for fn in ("run_r1","run_r2"):
            run=names[fn]
            codes=ast.unparse(run)
            self.assertIn("if t in tape:",codes) if fn=="run_r1" else self.assertIn("if asked:",codes)
            self.assertIn("sample.is_good",codes)
            self.assertIn("write_verified_normal",codes)
        self.assertIn("if y == 0:",ast.unparse(names["run_r1"]))
        self.assertIn("if y == 0:",ast.unparse(names["run_r2"]))

    def test_settings_validation(self):
        p=ext.parser()
        good=p.parse_args(["--experiment","r2","--dataset-root","A"])
        ext.init_settings(good)
        good.query_rate=1.2
        with self.assertRaises(ValueError):
            ext.init_settings(good)
        good.query_rate=.03
        good.rolling_min_normals=100
        with self.assertRaises(ValueError):
            ext.init_settings(good)

    def test_static_snapshot_metrics_independence(self):
        self.assertEqual(set(["static","stm_frozen","stm_guarded","stm_normal_quantile",
                              "stm_mistake_feedback"]),set(["static","stm_frozen",
                              "stm_guarded","stm_normal_quantile","stm_mistake_feedback"]))
        m1=e02.metrics([0,0,1,1],[.1,.2,.4,.5],[0,0,1,1])
        m2=e02.metrics([0,0,1,1],[.1,.2,.4,.5],[0,1,0,1])
        self.assertEqual(m1["auroc"],m2["auroc"])


if __name__=="__main__":
    unittest.main()
