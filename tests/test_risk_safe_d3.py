"""D3 CPU unit tests: scoped native top-k, provenance and fail-closed guards."""
from __future__ import annotations

from types import SimpleNamespace
import math
import unittest
import torch

from experiments.risk_safe.conditional_retrieval import (
    POLICIES, RADII, coordinates, grid_hw, assert_provenance,
    eligible_mask, conditional_dynamic_distance, ScopedRetrieval)
from experiments.risk_safe.retrieval_collision_audit import describe
from experiments.risk_safe.native_bridge import NativeParityError
from experiments.risk_safe import conditional_retrieval_pilot as pilot


class FakeImpl:
    @staticmethod
    def l2_normalize(x, eps=1e-12):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    @staticmethod
    def _grid_hw_from_npatches(n, aspect=None):
        if n != 4:
            raise ValueError(n)
        return 2, 2


class CB:
    def __init__(self, rows):
        self.protos = torch.tensor(rows, dtype=torch.float16).reshape(-1, 2)
    def size(self):
        return len(self.protos)
    def nearest(self, q, relax=False, k=5):
        if not self.size():
            return (torch.ones(len(q)), torch.full((len(q),), -1, dtype=torch.long))
        q = FakeImpl.l2_normalize(q).float()
        p = FakeImpl.l2_normalize(self.protos.float())
        sim = q @ p.T
        if relax:
            v, ids = sim.topk(k=min(k, sim.shape[1]), dim=1)
            return (1-v).mean(1), ids
        best, idx = sim.max(1)
        return 1-best, idx


class Bank:
    def __init__(self, dynamic=True):
        self.knn_k_normal = 5
        self.normal_fixed = {"p0": {0: {"id": [CB([[1, 0]])]}}}
        self.normal_dyn = {"p0": {0: {"id": [
            CB([[0, 1], [0, 1]]) if dynamic else CB([])]}}}
    def _dN_min_fixed_dyn(self, cls, mode, view, layer, q):
        f = self.normal_fixed[cls][mode][view][layer].nearest(
            q, relax=True, k=self.knn_k_normal)[0]
        d = self.normal_dyn[cls][mode][view][layer]
        if d.size():
            f = torch.minimum(f, d.nearest(
                q, relax=True, k=self.knn_k_normal)[0])
        return f, torch.full((len(q),), -1, dtype=torch.long)


def provenance():
    return [
        {"prototype_index": 0, "camera_id": 1, "u": .25, "v": .25},
        {"prototype_index": 1, "camera_id": 2, "u": .75, "v": .75},
    ]


def qpatches():
    return torch.tensor([[0., 1.]] * 4)


class ContextMaskTests(unittest.TestCase):
    def test_pre_registered_policy_matrix(self):
        self.assertEqual(len(POLICIES), 6)
        self.assertEqual(RADII["camera_spatial_0125"], .125)
        self.assertEqual(RADII["spatial_025"], .25)

    def test_native_grid_coordinates_are_aspect_aware(self):
        coords = coordinates(FakeImpl, 4, (200,200))
        self.assertTrue(torch.allclose(coords, torch.tensor([
            [.25,.25],[.25,.75],[.75,.25],[.75,.75]])))

    def test_camera_eligibility_is_only_dynamic(self):
        mask = eligible_mask(provenance(), "camera", query_camera=1,
            query_coords=coordinates(FakeImpl,4,(200,200)))
        self.assertEqual(mask.tolist(), [[True,False]]*4)

    def test_camera_spatial_eligibility_and_no_match(self):
        coords = coordinates(FakeImpl,4,(200,200))
        mask = eligible_mask(provenance(), "camera_spatial_0125",
                             query_camera=1, query_coords=coords)
        self.assertEqual(mask.tolist(), [
            [True,False],[False,False],[False,False],[False,False]])

    def test_native_mean_topfive_not_nearest_only(self):
        cb = CB([[1.,0.],[0.,1.]])
        q = torch.tensor([[1.,0.]])
        eligible = torch.tensor([[True,True]])
        actual = conditional_dynamic_distance(FakeImpl, cb, q, eligible, k=5)
        native = cb.nearest(q, relax=True,k=5)[0]
        self.assertTrue(torch.equal(actual, native))
        self.assertAlmostEqual(actual.item(), .5)
        only_first = torch.tensor([[True,False]])
        one = conditional_dynamic_distance(FakeImpl, cb, q, only_first,k=5)
        self.assertAlmostEqual(one.item(), 0., places=6)

    def test_no_eligible_dynamic_returns_infinity(self):
        d=conditional_dynamic_distance(FakeImpl,CB([[0,1]]),
                                        torch.tensor([[0.,1.]]),
                                        torch.tensor([[False]]),k=5)
        self.assertTrue(torch.isinf(d).all())

    def test_invalid_provenance_fail_closed(self):
        with self.assertRaises(NativeParityError):
            assert_provenance(provenance()[:-1],2)
        bad=provenance()
        bad[1]["camera_id"]=8
        with self.assertRaises(NativeParityError):
            assert_provenance(bad,2)

    def test_scoped_bank_does_not_modify_other_bank(self):
        original = Bank()
        other = Bank()
        scope = ScopedRetrieval(FakeImpl,other,provenance(),"camera_spatial_0125")
        scope.set_query(1,(200,200))
        before, _ = original._dN_min_fixed_dyn("p0",0,"id",0,qpatches())
        after, _ = scope.bank._dN_min_fixed_dyn("p0",0,"id",0,qpatches())
        self.assertTrue(torch.all(before <= after))
        self.assertTrue(torch.allclose(after[1:],torch.ones(3)))
        self.assertTrue(torch.allclose(before,torch.zeros(4)))
        still,_=original._dN_min_fixed_dyn("p0",0,"id",0,qpatches())
        self.assertTrue(torch.equal(before,still))

    def test_global_scope_bitwise_original_nearest(self):
        bank = Bank()
        original,_=bank._dN_min_fixed_dyn("p0",0,"id",0,qpatches())
        scoped=ScopedRetrieval(FakeImpl,bank,provenance(),"global")
        scoped.set_query(2,(200,200))
        result,_=bank._dN_min_fixed_dyn("p0",0,"id",0,qpatches())
        self.assertTrue(torch.equal(original,result))

    def test_camera_scope_returns_static_if_no_prototypes(self):
        bank=Bank(dynamic=False)
        scoped=ScopedRetrieval(FakeImpl,bank,(),"camera")
        scoped.set_query(1,(200,200))
        score,_=bank._dN_min_fixed_dyn("p0",0,"id",0,qpatches())
        self.assertTrue(torch.allclose(score,torch.ones(4)))

    def test_uncached_context_forbidden(self):
        bank = Bank()
        ScopedRetrieval(FakeImpl,bank,provenance(),"camera")
        with self.assertRaises(NativeParityError):
            bank._dN_min_fixed_dyn("p0",0,"id",0,qpatches())


class EvidenceContractTests(unittest.TestCase):
    def test_d3_argument_registry_is_mandatory(self):
        args=pilot.parser().parse_args(["--json-dir","G:/dataset/realiad/jsons/realiad_jsons"])
        with self.assertRaises(NativeParityError):
            pilot.enforce_protocol(args)
        args.threshold_registry="some/threshold_registry.json"
        args.fixed_quantile=.90
        pilot.enforce_protocol(args)

    def test_replay_preserves_frozen_b16(self):
        args=pilot.parser().parse_args(["--json-dir","x",
          "--threshold-registry","calib/threshold_registry.json",
          "--fixed-quantile","0.90"])
        cfg=pilot.config(args)
        self.assertEqual(cfg["max_query_events"],16)
        self.assertEqual(cfg["probe_groups_per_type"],4)
        self.assertFalse(cfg["retrieval_uses_test_labels"])
        self.assertFalse(cfg["probe_B_used"] if "probe_B_used" in cfg else
                         cfg["retrieval_uses_test_labels"])
        self.assertEqual(cfg["risk_gate"].split(":")[0],"NOT IMPLEMENTED")

    def test_collision_summary_explicit_topn_censoring(self):
        rows=[{"source":"greedy","label_offline_only":"1","score_erosion":".1",
               "cross_camera_top1_diagnostic":"True","image":"a","query_camera":"2"}]
        result=describe(rows,all_rows=10)
        self.assertTrue(result["top_n_censored"])
        self.assertEqual(
            result["greedy"]["top1_cross_camera_fraction_saved_defect_only"],1.)
        self.assertEqual(result["causal_novelty"]["saved_collision_patches"],0)


if __name__ == "__main__":
    unittest.main()
