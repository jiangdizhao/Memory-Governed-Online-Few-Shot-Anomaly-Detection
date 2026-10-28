"""Optional real-PyTorch CPU regression for P0/P1 native bank capacity loading.

Runs on the user's (torch) environment; skips cleanly in lightweight CI.
Exercises unchanged native_bridge.install_on_clone against a deliberately
misconfigured frozen bank K=1024 and candidate bank K=256.
"""
from __future__ import annotations
import importlib.util
import unittest
from types import SimpleNamespace


@unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                     "Requires PyTorch: run locally in the project's torch environment")
class TestP01NativeCapacity(unittest.TestCase):
    def test_clone_corrects_capacity_without_mutating_static_template(self):
        import torch
        from experiments.risk_safe.native_bridge import (
            NativeParityError, dynamic_codebooks, install_on_clone)
        from experiments.risk_safe.shadow_memory import GreedyBank
        from experiments.risk_safe.p01_falsification_native import (
            build_native_for_states, FalsificationProtocolError)
        from experiments.risk_safe.r1_oracle_core import DecisionState

        def bank_codebook(k):
            return SimpleNamespace(
                protos=torch.empty((0, 4), dtype=torch.float16),
                feat_dim=4, K=k)

        fixed = SimpleNamespace(
            normal_dyn={"p0":[{"id":[bank_codebook(1024)]}]},
            normal_dyn_ltm={"p0":[{"id":[]}]},
            normal_dyn_candidates={"p0":[{"id":[]}]})
        empty = DecisionState(
            GreedyBank((torch.empty((0,4), dtype=torch.float16),),
                       (256,)))
        with_one = DecisionState(
            GreedyBank((torch.tensor([[1.,0.,0.,0.]], dtype=torch.float16),),
                       (256,), version=1), decisions="1", admissions=1)

        # Original native-bridge defense is correct and MUST remain intact.
        with self.assertRaises(NativeParityError):
            install_on_clone(fixed, empty.bank)

        assembled = build_native_for_states(
            fixed, {"static":empty, "single_admit":with_one})
        self.assertEqual(dynamic_codebooks(fixed)[0].K, 1024)
        self.assertEqual(len(dynamic_codebooks(fixed)[0].protos), 0)
        self.assertEqual(dynamic_codebooks(assembled["static"])[0].K, 256)
        self.assertEqual(dynamic_codebooks(assembled["single_admit"])[0].K,256)
        self.assertEqual(len(dynamic_codebooks(assembled["static"])[0].protos),0)
        self.assertEqual(len(dynamic_codebooks(assembled["single_admit"])[0].protos),1)
        self.assertEqual(dynamic_codebooks(assembled["single_admit"])[0].protos.dtype,
                         torch.float16)
        self.assertTrue(torch.equal(
            dynamic_codebooks(assembled["single_admit"])[0].protos,
            with_one.bank.layers[0]))

        other = DecisionState(
            GreedyBank((torch.empty((0,4), dtype=torch.float16),),
                       (128,)))
        with self.assertRaises(FalsificationProtocolError):
            build_native_for_states(fixed, {"static":empty, "other":other})
        self.assertEqual(dynamic_codebooks(fixed)[0].K, 1024)


if __name__ == "__main__":
    unittest.main()
