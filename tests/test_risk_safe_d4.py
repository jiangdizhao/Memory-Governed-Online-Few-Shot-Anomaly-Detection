"""CPU regression tests for D4-A/B/C exact native dynamic-win budget."""
import copy
import hashlib
import unittest
from types import SimpleNamespace

import torch

from experiments.risk_safe.d4_retrieval_rights import (
    METHODS,REPLICATES,RHO,Evidence,ExactWinMask,make_evidence,
    stable_seed,credit,shuffled_risk,select_indices,
    spatial_priority,action_priority,native_winners)
from experiments.risk_safe.native_bridge import NativeParityError
from experiments.risk_safe.conditional_retrieval import ScopedRetrieval
from experiments.risk_safe import d4_equal_win_feedback_pilot as pilot
from test_risk_safe_d3 import Bank, FakeImpl


def examples():
    return make_evidence([
        (torch.tensor([[1.,0.],[0.,1.]]),
         torch.tensor([[.25,.25],[.75,.75]]),
         1,torch.tensor([.5,.1]),"normal-A"),
        (torch.tensor([[1.,1.]]),
         torch.tensor([[.5,.5]]),
         2,torch.tensor([.3]),"normal-B")],2)


class CreditTests(unittest.TestCase):
    def test_normal_credit_is_finite_and_context_sensitive(self):
        ex=examples()
        z=torch.tensor([[1.,0.],[0.,1.]])
        uv=torch.tensor([[.25,.25],[.75,.75]])
        out=credit(z,uv,1,ex)
        self.assertEqual(len(out),2)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(out[0].item(),out[1].item())

    def test_shuffled_defect_preserves_credit_marginals(self):
        ex=examples()
        scrambled=shuffled_risk(ex,0)
        again=shuffled_risk(ex,0)
        self.assertTrue(torch.equal(scrambled.embedding,again.embedding))
        self.assertTrue(torch.equal(scrambled.weight,ex.weight))
        self.assertEqual(scrambled.source,ex.source)
        self.assertTrue(torch.equal(scrambled.coordinate,again.coordinate))
        self.assertEqual(len(scrambled.embedding),len(ex.embedding))

    def test_no_risk_witness_becomes_identical_to_normal_only(self):
        normal=examples()
        risk=make_evidence([],2)
        q=torch.tensor([[1.,0.],[0.,1.]])
        uv=torch.tensor([[.25,.25],[.75,.75]])
        dummy=torch.tensor([[0,1,2,3,4],[0,1,2,3,4]])
        proto_uv=torch.rand(5,2)
        normal_score=action_priority(
            "normal_feedback",q,uv,1,normal,risk,
            top5=dummy,proto_uv=proto_uv)
        full_score=action_priority(
            "normal_defect_feedback",q,uv,1,normal,risk,
            top5=dummy,proto_uv=proto_uv)
        self.assertTrue(torch.equal(normal_score,full_score))

    def test_credit_shape_invalid_fails_closed(self):
        with self.assertRaises(NativeParityError):
            Evidence(torch.zeros(2,3),torch.zeros(1,2),
                     torch.zeros(2,dtype=torch.long),torch.zeros(2),
                     ("a","b"))


class EqualWinTests(unittest.TestCase):
    def test_all_methods_have_identical_native_win_count(self):
        candidate=torch.tensor([1,0,1,1,0,1,0,0,1,1,0,1],dtype=torch.bool)
        pri=torch.tensor([.3,.1,.4,.6,.2,.9,.4,.1,.5,.6,.2,.2])
        results=[]
        for method in METHODS:
            for repeat in REPLICATES if method in (
                    "random","shuffled_defect_feedback") else (0,):
                sel,k=select_indices(candidate,pri,rho=.25,method=method,
                                     image_id="heldout_C1",replicate=repeat)
                self.assertTrue(torch.all(~sel | candidate))
                self.assertEqual(sel.sum().item(),k)
                self.assertEqual(k,3)
                results.append(k)
        self.assertEqual(set(results),{3})

    def test_random_repeat_deterministic(self):
        candidate=torch.ones(30,dtype=torch.bool)
        pri=torch.arange(30,dtype=torch.float32)
        a,_=select_indices(candidate,pri,method="random",rho=.2,
                           image_id="heldout",replicate=1)
        b,_=select_indices(candidate,pri,method="random",rho=.2,
                           image_id="heldout",replicate=1)
        self.assertTrue(torch.equal(a,b))

    def test_zero_available_wins_grants_zero_permissions(self):
        s,k=select_indices(
            torch.zeros(10,dtype=torch.bool),torch.zeros(10),
            method="spatial",image_id="x")
        self.assertEqual(k,0)
        self.assertEqual(int(s.sum()),0)

    def test_mask_exactly_grants_all_original_prototypes_on_selected_patch(self):
        m=ExactWinMask()
        allowed=torch.tensor([True,False,True],dtype=torch.bool)
        m.set("imageA",allowed)
        mask=m([{}, {}, {}, {}, {}],"spatial_0125",
               query_camera=1,query_coords=torch.rand(3,2),
               image_id="imageA")
        self.assertEqual(mask.sum(dim=1).tolist(),[5,0,5])
        with self.assertRaises(NativeParityError):
            m([{}, {}, {}, {}, {}],"spatial_0125",query_camera=1,
              query_coords=torch.rand(3,2),image_id="other")

    def test_exact_win_scoped_native_distance_and_static_fallback(self):
        bank=Bank()
        bank.normal_dyn["p0"][0]["id"][0].protos=torch.tensor(
            [[0.,1.]]*5,dtype=torch.float16)
        bank.knn_k_normal=5
        bank.device=torch.device("cpu")
        query=torch.tensor([[0.,1.],[0.,1.],[0.,1.],[0.,1.]])
        # Global dynamic wins on every test patch; permission grants 2.
        global_distance,_=bank._dN_min_fixed_dyn("p0",0,"id",0,query)
        fixed=bank.normal_fixed["p0"][0]["id"][0].nearest(
            query,relax=True,k=5)[0]
        allow=ExactWinMask()
        allow.set("imageA",torch.tensor([1,0,1,0],dtype=torch.bool))
        provenance=[{"prototype_index":i,"camera_id":1,
                     "u":.25,"v":.25} for i in range(5)]
        scorer=ScopedRetrieval(FakeImpl,bank,provenance,"spatial_0125",
                               mask_provider=allow)
        scorer.set_query(1,(200,200),image_id="imageA")
        scoped,_=bank._dN_min_fixed_dyn("p0",0,"id",0,query)
        self.assertTrue(torch.allclose(scoped[[0,2]],global_distance[[0,2]],
                                       atol=1e-7))
        self.assertTrue(torch.allclose(scoped[[1,3]],fixed[[1,3]],
                                       atol=1e-7))
        self.assertEqual(scorer.last_query_stats["n_dynamic_winner_patches"],2)
        self.assertEqual(scorer.last_query_stats["n_static_fallback_patches"],2)

    def test_invalid_budget_and_candidate_mask_fails_closed(self):
        candidate=torch.tensor([True,False])
        with self.assertRaises(NativeParityError):
            select_indices(candidate,torch.tensor([.1,.2]),
                           method="random",image_id="x",rho=0)
        with self.assertRaises(NativeParityError):
            select_indices(candidate,torch.tensor([.1,.2]),
                           method="madeup",image_id="x")

    def test_spatial_priority_uses_same_topfive_source_indices(self):
        top5=torch.tensor([[0,1,2,3,4]])
        proto=torch.tensor([[.5,.5],[.5,.5],[.5,.5],[.5,.5],[.5,.5]])
        score=spatial_priority(top5,torch.tensor([[.5,.5]]),proto)
        self.assertAlmostEqual(score.item(),0.)

    def test_native_topfive_winners_preserved_with_fake_bank(self):
        bank=Bank()
        bank.normal_dyn["p0"][0]["id"][0].protos=torch.tensor(
            [[0.,1.]]*5,dtype=torch.float16)
        bank.knn_k_normal=5
        bank.device=torch.device("cpu")
        q,w,idx=native_winners(FakeImpl,bank,[torch.tensor([[0.,1.]]*4)])
        self.assertEqual(tuple(idx.shape),(4,5))
        self.assertTrue(w.all())


class ProtocolTests(unittest.TestCase):
    def test_d4_freezes_rho_and_feedback_scope(self):
        args=pilot.parser().parse_args([
            "--reference-d2a-summary","old/summary.json",
            "--threshold-registry","normal/registry.json",
            "--fixed-quantile","0.90"])
        pilot.enforce(args)
        c=pilot.config(args)
        self.assertFalse(c["probe_B_used"])
        self.assertFalse(c["policy_tuning_with_probe"])
        self.assertEqual(c["causal_scope"],
                         "post-B16 final checkpoint; NOT online incremental policy")
        self.assertEqual(RHO,.02)
        args.win_budget_fraction=.03
        with self.assertRaises(NativeParityError):
            pilot.enforce(args)

    def test_expected_output_matrix_is_720_for_9_arms_and_2_methods(self):
        n=sum(3 if k in ("random","shuffled_defect_feedback") else 1
              for k in METHODS)
        self.assertEqual(n,9)
        self.assertEqual(2*n*40,720)
        self.assertEqual(2*n,18)

    def test_no_source_feedback_implies_same_normal_full_rank(self):
        empty=make_evidence([],2)
        query=torch.tensor([[1.,0.],[0.,1.]])
        uv=torch.tensor([[.25,.25],[.75,.75]])
        proto_uv=torch.zeros(5,2)
        top5=torch.tensor([[0,1,2,3,4],[0,1,2,3,4]])
        a=action_priority("normal_feedback",query,uv,1,empty,empty,
                          top5=top5,proto_uv=proto_uv)
        b=action_priority("normal_defect_feedback",query,uv,1,empty,empty,
                          top5=top5,proto_uv=proto_uv)
        self.assertTrue(torch.equal(a,b))


if __name__=="__main__":
    unittest.main()
