"""R1 CPU-only oracle-contract tests. No torch, Real-IAD or GPU required."""
from __future__ import annotations

import hashlib
import unittest
from types import SimpleNamespace

from experiments.risk_safe.r1_oracle_core import (
    DecisionState, R1ProtocolError, transition, evaluate_scores, run_exact,
    run_beam, frontier_by_admission_count, search_rank)


class ToyBank:
    """Immutable stand-in testing state-dependent admit/reject semantics."""
    def __init__(self, accepted=(), version=0):
        self.accepted=tuple(accepted)
        self.version=version
        self.layers=(self.accepted,)
    def sha256(self):
        return hashlib.sha256(repr((self.accepted, self.version)).encode()).hexdigest()
    def proposed(self, proposal, *, already_normalized=False):
        if not already_normalized:
            raise AssertionError("Test expects native normalized patches")
        if proposal in self.accepted:
            return ToyBank(self.accepted,self.version),SimpleNamespace(
                appended=0,replaced=0,skipped=1)
        return ToyBank(self.accepted+(proposal,),self.version+1),SimpleNamespace(
            appended=1,replaced=0,skipped=0)


class R1OracleContracts(unittest.TestCase):
    def setUp(self):
        self.labels=[0,0,1]
        self.static_scores=[.7,.4,.8]
        self.theta=.5
        self.init=DecisionState(ToyBank())
        def assess(state):
            first="rescue" in state.bank.accepted
            harmful="unsafe" in state.bank.accepted
            scores=[.3 if first else .7, .4,
                    .4 if harmful else (.7 if first else .8)]
            return evaluate_scores(self.labels,self.static_scores,scores,self.theta)
        self.assess=assess

    def test_risk_first_exact_all_four_leaves(self):
        result=run_exact(("rescue","unsafe"),self.init,self.assess)
        self.assertTrue(result["globally_exhaustive"])
        self.assertEqual(result["status"],"EXACT_FINITE_PROTOCOL_OPTIMUM")
        self.assertEqual(result["evaluated_leaves"],4)
        self.assertEqual(result["best_state"].decisions,"10")
        self.assertEqual(result["best_metrics"]["rescued_false_positives"],1)
        self.assertEqual(result["best_metrics"]["new_false_negatives"],0)
        self.assertEqual(len(result["leaf_rows"]),4)

    def test_exact_rejects_partial_approximation(self):
        with self.assertRaises(R1ProtocolError):
            run_exact(tuple(range(5)),self.init,self.assess,max_leaves=31)

    def test_beam_never_claims_oracle_bound(self):
        result=run_beam(("rescue","unsafe"),self.init,self.assess,beam_width=2)
        self.assertFalse(result["globally_exhaustive"])
        self.assertEqual(result["status"],"BEAM_BEST_FOUND_NOT_AN_UPPER_BOUND")
        self.assertEqual(len(result["stage_rows"]),2)
        self.assertEqual(result["best_state"].decisions,"10")

    def test_identical_double_admission_is_no_op_and_versions_stable(self):
        first=transition(self.init,"rescue",True)
        second=transition(first,"rescue",True)
        self.assertEqual(first.bank.version,1)
        self.assertEqual(second.bank.version,1)
        self.assertEqual(second.admissions,2)
        self.assertEqual(second.effective_updates,1)
        self.assertEqual(second.bank.sha256(),first.bank.sha256())
        rejected=transition(first,"unsafe",False)
        self.assertEqual(rejected.bank.sha256(),first.bank.sha256())
        self.assertEqual(rejected.admissions,1)

    def test_metric_outcomes_and_penalty(self):
        result=evaluate_scores([0,0,1],[.7,.4,.8],[.3,.4,.4],.5,
                               lambda_defect=2.)
        self.assertEqual(result["rescued_false_positives"],1)
        self.assertEqual(result["new_false_negatives"],1)
        self.assertAlmostEqual(result["mean_normal_score_relief"],.2)
        self.assertAlmostEqual(result["mean_defect_score_erosion"],.4)
        self.assertAlmostEqual(result["risk_adjusted_relief"],-.6)
        self.assertFalse(result["safe_no_new_fn"])

    def test_pareto_slices_are_not_confused_with_global_optimum(self):
        rows=run_exact(("rescue","unsafe"),self.init,self.assess)["leaf_rows"]
        front=frontier_by_admission_count(rows)
        self.assertEqual([r["n_admitted"] for r in front],[0,1,2])
        self.assertEqual(front[1]["decisions"],"10")

    def test_no_defect_on_search_probe_fails_closed(self):
        with self.assertRaises(R1ProtocolError):
            evaluate_scores([0,0],[.5,.6],[.4,.3],.5)

    def test_search_prioritizes_avoiding_new_fn(self):
        m_safe=self.assess(transition(self.init,"rescue",True))
        m_harm=self.assess(transition(self.init,"unsafe",True))
        self.assertGreater(search_rank(m_safe,DecisionState(ToyBank(("rescue",)))),
                           search_rank(m_harm,DecisionState(ToyBank(("unsafe",)))))


if __name__ == "__main__":
    unittest.main()
