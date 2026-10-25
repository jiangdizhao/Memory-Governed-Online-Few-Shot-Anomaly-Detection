"""E02-R3 offline contracts: strict budget and no-label query decisions."""
import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import e02_r3_bootstrap_sidecar as r3


class R3Tests(unittest.TestCase):
    def test_budget_exact_and_terminal_catchup(self):
        b=r3.Budget(3815,.03)
        selected=[t for t in range(1,3816) if b.decide(t,False)[0]]
        self.assertEqual(len(selected),114)
        self.assertEqual(selected[0],3702)
        self.assertEqual(b.catchup,114)
        b.check()

    def test_budget_exhaustion_suppresses_extra(self):
        b=r3.Budget(100,.03)
        sel=[t for t in range(1,101) if b.decide(t,True)[0]]
        self.assertEqual(sel,[1,2,3])
        self.assertEqual(b.used,3)
        self.assertEqual(b.suppressed,97)
        b.check()

    def test_budget_never_reads_labels(self):
        b=r3.Budget(100,.03)
        self.assertFalse(hasattr(b,"y_true"))
        self.assertEqual(b.K,3)
        self.assertEqual(r3.sha([1,2,3]),r3.sha([1,2,3]))
        self.assertNotEqual(r3.sha([1,2,3]),r3.sha([1,3,2]))

    def test_policy_calls_original_bootstrap_and_simple_aif(self):
        tree=ast.parse((ROOT/"experiments"/"e02_r3_bootstrap_sidecar.py").read_text(encoding="utf-8"))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="policy_decision")
        calls=[node.func.attr for node in ast.walk(fn) if isinstance(node,ast.Call)
               and isinstance(node.func,ast.Attribute)]
        for name in ("novelty_percentile","choose_simple_aif_bootstrap_action",
                     "select_simple_aif_action"):
            self.assertIn(name,calls)
        self.assertNotIn("is_good",ast.unparse(fn))
        step=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="step_state")
        code=ast.unparse(step)
        self.assertLess(code.index('truth = 0 if sample.is_good'),code.index('if asked:'))
        self.assertLess(code.index("for name, controller in state['controllers'].items():"),
                        code.index('truth = 0 if sample.is_good'))
        self.assertIn("state['coverage'].observe_normal(global_feat)",code)

    def test_sidecars_share_scores_and_query_tape(self):
        tree=ast.parse((ROOT/"experiments"/"e02_r3_bootstrap_sidecar.py").read_text(encoding="utf-8"))
        source=ast.unparse(tree)
        self.assertIn("balanced.cfg.enable_balanced_rescue = True",source)
        self.assertIn("'guarded': guarded",source)
        self.assertIn("'balanced_rescue': balanced",source)
        self.assertIn("ctrl.observe_labeled(score, truth)",source)
        self.assertIn("e02.write_verified_normal(",source)
        self.assertIn("e02.make_query_tape(",source)
        self.assertIn("e02.init_canonical(",source)


if __name__=="__main__":
    unittest.main()
