"""P0/P1 pure contract tests; no Torch or GPU required."""
from __future__ import annotations
import unittest
from experiments.risk_safe.p01_falsification_core import (
    FalsificationProtocolError,select_matched,normal_q90,metrics,
    paired_changes,select_oracle_bits)


class TestFalsification(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(sample_key=f"sample{i:03d}",
                        static_score=.2+.01*i,
                        static_patch_novelty=.5+.05*(10-i))
                   for i in range(10)]

    def test_matched_count_all_and_reproducibility(self):
        a=select_matched(self.rows,4,seed=1701)
        b=select_matched(self.rows,4,seed=1701)
        self.assertEqual(a,b)
        self.assertEqual(a["first_k"],set(range(4)))
        self.assertEqual(a["static_score_top_k"],{6,7,8,9})
        self.assertEqual(a["static_novelty_top_k"],{0,1,2,3})
        self.assertTrue(all(len(v)==4 for v in a.values()))

    def test_duplicate_and_nonfinite_fail_closed(self):
        with self.assertRaises(FalsificationProtocolError):
            select_matched(self.rows,11,seed=1)
        dup=[dict(x) for x in self.rows]
        dup[1]["sample_key"]=dup[0]["sample_key"]
        with self.assertRaises(FalsificationProtocolError):
            select_matched(dup,4,seed=1)
        dup=[dict(x) for x in self.rows]
        dup[1]["static_score"]=float("nan")
        with self.assertRaises(FalsificationProtocolError):
            select_matched(dup,4,seed=1)

    def test_quantile_64_and_finite(self):
        self.assertAlmostEqual(normal_q90(list(range(64))),57.0)
        with self.assertRaises(FalsificationProtocolError):
            normal_q90(list(range(63)))
        with self.assertRaises(FalsificationProtocolError):
            normal_q90([float("inf")]+list(range(63)))

    def test_confusion_auc_ap_and_threshold(self):
        # Strict separability with an explicit threshold
        m=metrics([0,0,1,1],[.1,.2,.8,.9],.5)
        self.assertEqual([m[k] for k in ("tp","tn","fp","fn")],[2,2,0,0])
        self.assertAlmostEqual(m["auroc"],1)
        self.assertAlmostEqual(m["auprc"],1)
        self.assertAlmostEqual(m["balanced_accuracy"],1)
        inv=metrics([0,0,1,1],[.9,.8,.1,.2],.5)
        self.assertEqual(inv["tp"],0)
        self.assertEqual(inv["fp"],2)
        self.assertAlmostEqual(inv["auroc"],0)

    def test_tied_scores(self):
        m=metrics([0,1],[.4,.4],.4)
        self.assertEqual(m["fp"],1)
        self.assertEqual(m["tp"],1)
        self.assertAlmostEqual(m["auroc"],.5)
        self.assertAlmostEqual(m["auprc"],.5)

    def test_paired_operating_points_not_implicit_score_gain(self):
        out=paired_changes([0,0,1,1],[.7,.2,.7,.2],
                           [.4,.8,.4,.8],.5,.5)
        self.assertEqual(out["rescued_false_positives"],1)
        self.assertEqual(out["new_false_positives"],1)
        self.assertEqual(out["new_false_negatives"],1)
        self.assertEqual(out["rescued_false_negatives"],1)

    def test_oracle_must_be_complete_matching_r1(self):
        ref=dict(status="COMPLETE",globally_exhaustive=False,
            config=dict(budget=128,capacity=256),
            oracle_provenance=dict(tape_sha256="abc"),
            oracle_decision_bits="10100")
        self.assertEqual(select_oracle_bits(ref,tape_sha="abc",
                            n_normal=5,target=2,capacity=256),{0,2})
        with self.assertRaises(FalsificationProtocolError):
            select_oracle_bits(ref,tape_sha="wrong",
                            n_normal=5,target=2,capacity=256)
        with self.assertRaises(FalsificationProtocolError):
            select_oracle_bits(ref,tape_sha="abc",
                            n_normal=5,target=3,capacity=256)


if __name__=="__main__":
    unittest.main()
