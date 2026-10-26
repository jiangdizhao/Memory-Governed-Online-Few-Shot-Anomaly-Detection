"""Bridge offline reproducibility and factorial design contracts; no Real-IAD."""
from __future__ import annotations
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import bridge_memory as mem
import bridge_historical_matched as trial
import run_bridge_audit as runner

class BridgeContractTests(unittest.TestCase):
    def test_historical_config_is_not_arbitrary(self):
        x=mem.BridgeConfig()
        self.assertEqual((x.stm,x.ltm,x.buffer,x.total),(512,512,4096,5120))
        self.assertEqual((x.min_events,x.promote_every,x.patches),(2,32,24))
        self.assertEqual(mem.BridgeConfig(stm=768).total,5376)
        for wrong in (
            {"stm":256},{"ltm":64},{"buffer":128},
            {"min_events":3},{"promote_every":16},{"patches":8},
            {"evidence_mode":"oracle"},
        ):
            with self.assertRaises(ValueError):
                mem.BridgeConfig(**wrong)

    def test_eight_arms_are_complete(self):
        self.assertEqual(tuple(mem.ARM_NAMES),tuple(runner.ARMS))
        self.assertEqual(len(mem.ARM_NAMES),8)
        self.assertEqual(len(runner.conditions(runner.parser().parse_args(["--all"]))),6)
        self.assertEqual(len(runner.conditions(runner.parser().parse_args([
            "--all","--stm-capacities","512,768"]))),12)
        with self.assertRaises(ValueError):
            runner.conditions(runner.parser().parse_args([
                "--all","--stm-capacities","512,512"]))

    def test_same_bank_counterfactual_restores_flags_even_on_exception(self):
        bank=SimpleNamespace(
            enable_ltm_retrieval=True,enable_maturation_buffer_retrieval=True)
        def score():
            self.assertFalse(bank.enable_ltm_retrieval)
            self.assertTrue(bank.enable_maturation_buffer_retrieval)
            return .123
        self.assertEqual(trial.with_retrieval(
            bank,score,ltm=False,buffer=True),.123)
        self.assertTrue(bank.enable_ltm_retrieval)
        self.assertTrue(bank.enable_maturation_buffer_retrieval)
        with self.assertRaisesRegex(RuntimeError,"test"):
            trial.with_retrieval(
                bank,lambda:(_ for _ in ()).throw(RuntimeError("test")),
                ltm=False,buffer=False)
        self.assertTrue(bank.enable_ltm_retrieval)
        self.assertTrue(bank.enable_maturation_buffer_retrieval)

    def test_one_command_is_unambiguous(self):
        a=runner.parser().parse_args(["--all","--math-sdpa"])
        c=runner.command(a,"audiojack",0,512,"native",smoke=True)
        self.assertEqual(c[c.index("--max-samples")+1],"64")
        self.assertEqual(c[c.index("--stm-capacity")+1],"512")
        self.assertEqual(c[c.index("--evidence-mode")+1],"neutral")
        self.assertIn("--math-sdpa",c)
        self.assertEqual(runner.out_path(a,"switch",1,768,"native").name,
                         "switch_s1_STM768_native_full")

    def test_aulc(self):
        self.assertAlmostEqual(trial.prev.aulc([
            {"step":0,"auroc":.5},{"step":100,"auroc":.7},
            {"step":200,"auroc":.9}]),.7)

if __name__=="__main__":unittest.main()
