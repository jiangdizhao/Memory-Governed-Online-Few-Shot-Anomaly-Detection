"""R5 D-only pacing regression tests: no oracle, exact C time rule, B cost semantics."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import e02_r4_abc_query_governance as r4
import e02_r5_d_temporal_pacing as r5


class FakeAif:
    def __init__(self,adapt_query_cost=False):
        self.cfg=SimpleNamespace(
            adapt_query_cost=adapt_query_cost,
            query_cost_lr=.02,target_qrate=.03,local_qrate_window=256,
            min_query_cost=.05,max_query_cost=3.0)
        self.query_cost_runtime=.8
        self.steps=0
        self.queries=0
        self.recent_query_flags=[]

    def observe(self,asked):
        self.steps+=1
        self.queries+=int(bool(asked))
        self.recent_query_flags.append(int(bool(asked)))
        if len(self.recent_query_flags)>256:
            self.recent_query_flags=self.recent_query_flags[-256:]
        if self.cfg.adapt_query_cost:
            rate=sum(self.recent_query_flags)/len(self.recent_query_flags)
            self.query_cost_runtime=max(.05,min(3,
                self.query_cost_runtime+.02*(rate-.03)))


class TimeOnlyContracts(unittest.TestCase):
    def test_d_allowed_exactly_same_as_c_every_step(self):
        n,k=3815,114
        c=r4.ScoreRegionBudget(n,k,early_steps=400,early_total_fraction=.4)
        d=r5.TimeOnlyPacing(n,k,early_steps=400,early_fraction=.4)
        self.assertEqual([d.allowed_total(t) for t in range(1,n+1)],
                         [c.allowed_total(t) for t in range(1,n+1)])
        self.assertEqual(d.allowed_total(400),46)
        self.assertEqual(d.allowed_total(n),k)
        self.assertEqual(d.snapshot()["score_quota_enforced"],False)

    def test_d_no_region_intervention(self):
        for score_region in ("low_score","boundary","high_score"):
            for count in (0,16,999):
                budget=r4.Budget(3815,.03)
                d=r5.TimeOnlyPacing(3815,budget.K)
                prop,reason=d.propose(
                    1,score_region,True,budget,
                    target_normals=16,verified_normal_count=count)
                self.assertTrue(prop)
                self.assertEqual(reason,"native_policy")
                prop,reason=d.propose(
                    1,score_region,False,budget,
                    target_normals=16,verified_normal_count=count)
                self.assertFalse(prop)
        cls=next(n for n in ast.parse(
            (ROOT/"experiments"/"e02_r5_d_temporal_pacing.py").read_text()
            ).body if isinstance(n,ast.ClassDef) and n.name=="TimeOnlyPacing")
        method=next(x for x in cls.body if isinstance(x,ast.FunctionDef)
                    and x.name=="propose")
        code=ast.unparse(method)
        self.assertIn("del region, target_normals, verified_normal_count",code)
        self.assertNotIn("y_true",code)
        self.assertNotIn("is_good",code)

    def test_d_pacing_respects_cap_and_no_free_label(self):
        n=3815
        budget=r4.Budget(n,.03)
        d=r5.TimeOnlyPacing(n,budget.K)
        requests=[]
        for t in range(1,n+1):
            prop,_=d.propose(
                t,"low_score",True,budget,
                target_normals=16,verified_normal_count=0)
            asked,source=budget.decide(t,prop)
            d.observe("low_score",asked)
            if asked:requests.append((t,source))
        self.assertEqual(len(requests),114)
        self.assertEqual(sum(t<=400 for t,_ in requests),46)
        self.assertEqual(budget.used,budget.K)
        self.assertEqual(d.queried,budget.K)
        self.assertGreater(d.blocked,0)
        self.assertEqual(d.snapshot()["low_score_priority_enabled"],False)

    def test_d_uses_same_cost_feedback_semantics_as_r4_b(self):
        def state(name):
            a=FakeAif()
            return {"name":name,"aif":a,
                    "excluded_cost_steps":0,"excluded_cost_queries":0,
                    "post_bootstrap_flags":[],"post_bootstrap_adapt_steps":0}
        b=state("B")
        d=state("D")
        for i in range(1,1000):
            asked=i%7==0
            bootstrap=i<=400
            r4.record_query_cost(b,asked,bootstrap=bootstrap)
            r4.record_query_cost(d,asked,bootstrap=bootstrap)
        self.assertEqual(b["aif"].query_cost_runtime,d["aif"].query_cost_runtime)
        self.assertEqual(b["excluded_cost_queries"],d["excluded_cost_queries"])
        self.assertEqual(b["aif"].queries,d["aif"].queries)
        self.assertAlmostEqual(b["aif"].query_cost_runtime, d["aif"].query_cost_runtime)

    def test_build_d_disables_native_adaptation_and_uses_pacer(self):
        cfg=SimpleNamespace(adapt_query_cost=True)
        dummy={"name":"D","policy_cfg":cfg,"aif":SimpleNamespace(cfg=cfg),
               "budget":r4.Budget(3815,.03)}
        args=SimpleNamespace(max_steps_effective=3815,strata_early_steps=400,
                             strata_early_budget_fraction=.4)
        with patch.object(r4,"init_arm",return_value=dummy):
            d=r5.make_d(None,None,None,None,None,None,None,args,None,None)
        self.assertIs(d,dummy)
        self.assertFalse(cfg.adapt_query_cost)
        self.assertIsInstance(d["strata"],r5.TimeOnlyPacing)

    def test_reference_checks_require_matching_r4_protocol(self):
        path=ROOT/"outputs"/"e02_r4_abc_audiojack_full_seed0"/"r4_summary.json"
        prior=json.loads(path.read_text(encoding="utf8"))
        kwargs={k:prior["parameters"][k] for k in r5.REFERENCE_KEYS}
        kwargs["reference_r4_summary"]=str(path)
        kwargs["class_name"]=prior["class_name"]
        args=SimpleNamespace(**kwargs)
        self.assertEqual(r5.read_r4_reference(args)["n_stream"],3815)
        args.query_rate=.04
        with self.assertRaisesRegex(ValueError,"query_rate"):
            r5.read_r4_reference(args)
        self.assertEqual(prior["total_budget_each"],114)

    def test_no_stale_random_tape_or_missing_output_fields(self):
        src=(ROOT/"experiments"/"e02_r5_d_temporal_pacing.py").read_text(encoding="utf8")
        parsed=ast.parse(src)
        run=next(x for x in parsed.body if isinstance(x,ast.FunctionDef) and x.name=="main_run")
        code=ast.unparse(run)
        self.assertNotIn("random_tape",code)
        self.assertIn("'total_budget_each': r4.validated_equal_budget({'D': d})",code)
        self.assertIn("check_r4_match(old, report)",code)
        self.assertIn("r4.step_state(",code)
        self.assertIn("r4.report_state(",code)


if __name__=="__main__":
    unittest.main()
