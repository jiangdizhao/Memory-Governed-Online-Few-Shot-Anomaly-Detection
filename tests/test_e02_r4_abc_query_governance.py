"""CPU-only R4 A/B/C causal-query-cost and score-stratum regression tests."""
import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import e02_r4_abc_query_governance as r4

class DummyAIF:
    def __init__(self):
        self.cfg=SimpleNamespace(query_cost_lr=.02,target_qrate=.03,
            min_query_cost=.05,max_query_cost=3.0,
            local_qrate_window=256,adapt_query_cost=True)
        self.query_cost_runtime=.80
        self.steps=0
        self.queries=0
        self.recent_query_flags=[]
    def observe(self,queried):
        self.steps+=1
        self.queries+=int(bool(queried))
        self.recent_query_flags.append(int(bool(queried)))
        if len(self.recent_query_flags)>256:
            self.recent_query_flags=self.recent_query_flags[-256:]
        if self.cfg.adapt_query_cost:
            rate=sum(self.recent_query_flags)/len(self.recent_query_flags)
            self.query_cost_runtime=max(.05,min(3.0,self.query_cost_runtime+.02*(rate-.03)))


class ABCContracts(unittest.TestCase):
    def test_budget_inherited_from_r3(self):
        for n in [100,3815]:
            b=r4.Budget(n,.03)
            chosen=[t for t in range(1,n+1) if b.decide(t,t%3==0)[0]]
            self.assertEqual(len(chosen),3 if n==100 else 114)
            b.check()

    def test_b_excludes_bootstrap_from_adaptation_but_counts_queries(self):
        a=DummyAIF()
        b=DummyAIF()
        b.cfg.adapt_query_cost=False
        sa={"name":"A","aif":a}
        sb={"name":"B","aif":b,"excluded_cost_steps":0,
            "excluded_cost_queries":0,"post_bootstrap_flags":[],
            "post_bootstrap_adapt_steps":0}
        for t in range(1,401):
            q=t%5==0
            r4.record_query_cost(sa,q,bootstrap=True)
            r4.record_query_cost(sb,q,bootstrap=True)
        self.assertEqual(a.queries,b.queries)
        self.assertEqual(a.steps,b.steps)
        self.assertEqual(sb["excluded_cost_steps"],400)
        self.assertEqual(sb["excluded_cost_queries"],80)
        self.assertAlmostEqual(b.query_cost_runtime,.8)
        self.assertGreater(a.query_cost_runtime,1.5)
        r4.record_query_cost(sb,False,bootstrap=False)
        self.assertAlmostEqual(b.query_cost_runtime,.8-.0006)
        self.assertEqual(sb["post_bootstrap_flags"],[0])

    def test_c_score_region_is_not_oracle_labels(self):
        allocation=r4.ScoreRegionBudget(3815,114)
        self.assertEqual(sum(allocation.quotas.values()),114)
        self.assertEqual(allocation.region(.20,.50,.05),"low_score")
        self.assertEqual(allocation.region(.49,.50,.05),"boundary")
        self.assertEqual(allocation.region(.60,.50,.05),"high_score")
        self.assertEqual(allocation.early_total_cap,46)
        # pre-registered zero-label evidence. No score-region classifier
        # is allowed to read ground truth.
        source=ast.parse((ROOT/"experiments"/"e02_r4_abc_query_governance.py").read_text(encoding="utf8"))
        c=next(x for x in source.body if isinstance(x,ast.ClassDef) and x.name=="ScoreRegionBudget")
        self.assertNotIn("is_good",ast.unparse(c))
        self.assertNotIn("y_true",ast.unparse(c))

    def test_c_low_score_priority_budget_and_caps(self):
        b=r4.Budget(100,.03)
        c=r4.ScoreRegionBudget(100,3,early_steps=40,early_total_fraction=.66,
                               low_fraction=.33,boundary_fraction=.33)
        p,reason=c.propose(1,"low_score",False,b,target_normals=16,
                           verified_normal_count=0)
        self.assertTrue(p)
        self.assertEqual(reason,"low_score_bootstrap_priority")
        yes,_=b.decide(1,p);c.observe("low_score",yes)
        self.assertTrue(yes)
        self.assertEqual(c.counts["low_score"],1)
        p,reason=c.propose(2,"low_score",True,b,target_normals=16,
                           verified_normal_count=0)
        self.assertFalse(p)
        self.assertEqual(reason,"stratum_quota_exhausted")
        # boundary region still has its independent quota.
        p,_=c.propose(3,"boundary",True,b,target_normals=16,
                       verified_normal_count=0)
        self.assertTrue(p)
        yes,_=b.decide(3,p);c.observe("boundary",yes)
        self.assertTrue(yes)
        self.assertEqual(b.used,2)
        # third query can only happen by late-stream catchup.
        for t in range(4,101):
            proposal,_=c.propose(t,"low_score",False,b,target_normals=16,
                                 verified_normal_count=0)
            yes,_=b.decide(t,proposal);c.observe("low_score",yes)
        b.check()
        self.assertEqual(sum(c.counts.values()),3)
        self.assertGreater(c.overrides["low_score"],0)

    def test_stability_metrics_report_missing_class_as_none(self):
        rows=[]
        for i in range(1,9):
            r={"t":i,"queried":int(i==1),"y_eval_only":0,
                "static_score":.2,"stm_score":.1}
            for k in ("static","frozen","guarded","balanced_rescue","mistake_feedback"):
                r[k+"_pred"]=0
            rows.append(r)
        diagnostics=r4.stability_windows(rows,stride=4)
        self.assertEqual(len(diagnostics),2)
        self.assertEqual(diagnostics[0]["n_defect"],0)
        self.assertIsNone(diagnostics[0]["balanced_rescue"]["fnr"])
        self.assertIsNone(diagnostics[0]["balanced_rescue"]["balanced_accuracy"])
        self.assertEqual(diagnostics[0]["balanced_rescue"]["fpr"],0)

    def test_prequential_label_separation(self):
        src=(ROOT/"experiments"/"e02_r4_abc_query_governance.py").read_text(encoding="utf8")
        fn=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)
                and n.name=="step_state")
        code=ast.unparse(fn)
        self.assertLess(code.index("proposed, reason = policy_decision("),
                        code.index("truth = 0 if sample.is_good"))
        self.assertLess(code.index("asked, budget_reason = state['budget'].decide"),
                        code.index("truth = 0 if sample.is_good"))
        self.assertLess(code.index("for name, controller in state['controllers'].items():"),
                        code.index("truth = 0 if sample.is_good"))
        self.assertLess(code.index("truth = 0 if sample.is_good"),
                        code.index("if asked:"))
        self.assertIn("record_query_cost(state, asked, bootstrap=bootstrap)",code)
        self.assertIn("state['strata'].observe(score_region, asked)",code)

if __name__=="__main__":
    unittest.main()
