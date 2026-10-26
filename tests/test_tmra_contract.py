"""Offline TMRA contract regression; torch/GPU not required."""
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import tmra_recurrence_core as recurrence
import tmra_audit_core as audit
import tmra_compact_memory as compact
import run_tmra as runner

@dataclass
class Item:
    path: str

class TMRAContracts(unittest.TestCase):
    def test_source_retrieval_shapley_adds_up(self):
        x={"S0":.50,"SL":.40,"SB":.38,"SLB":.35}
        q=audit.shapley(x)
        self.assertAlmostEqual(q["phi_ltm"]+q["phi_buffer"],.15)
        self.assertAlmostEqual(q["phi_ltm"],.065)
        self.assertAlmostEqual(q["phi_buffer"],.085)
        self.assertEqual(set(audit.SOURCES),{"fixed","stm","ltm","buffer"})

    def test_retrieval_mutation_is_temporary(self):
        b=SimpleNamespace(
            enable_ltm_retrieval=True,
            enable_maturation_buffer_retrieval=True)
        def run():
            self.assertFalse(b.enable_ltm_retrieval)
            self.assertFalse(b.enable_maturation_buffer_retrieval)
            return 2.
        with audit.source_flags(b,False,False):
            self.assertEqual(run(),2.)
        self.assertTrue(b.enable_ltm_retrieval)
        self.assertTrue(b.enable_maturation_buffer_retrieval)
        with self.assertRaises(ZeroDivisionError):
            with audit.source_flags(b,False,False):1/0
        self.assertTrue(b.enable_ltm_retrieval)

    def test_partition_disjoint_and_mode_schedule(self):
        config=recurrence.Settings()
        samples=[Item(str(i)) for i in range(3800)]
        A,B,phases=recurrence.label_blind_partition(samples,config,240)
        self.assertEqual(tuple(map(len,(A,B,*phases))),
                         (96,96,900,900,900))
        all_paths=[x.path for x in A+B+sum(phases,[])]
        self.assertEqual(len(all_paths),len(set(all_paths)))
        self.assertEqual(recurrence.phase_modes("ABA"),("A","B","A"))
        self.assertEqual(recurrence.phase_modes("AAA"),("A","A","A"))
        A2,B2,phases2=recurrence.label_blind_partition(samples,config,240)
        self.assertEqual([x.path for x in A],[x.path for x in A2])
        self.assertEqual([[x.path for x in p] for p in phases],
                         [[x.path for x in p] for p in phases2])

    def test_shift_is_fixed_and_semantic_label_blind(self):
        with self.assertRaises(ValueError):
            recurrence.phase_modes("BAB")
        for changed in (dict(stm=64),dict(flat=256),dict(q_rate=.04),
                        dict(cap_bytes=5_000_000),dict(min_events=3)):
            with self.assertRaises(ValueError):
                recurrence.Settings(**changed)
        self.assertEqual(recurrence.first_recovery(
            [0.71,0.80,0.83],[0,1,3],.835,tolerance=.01),3)
        self.assertIsNone(recurrence.first_recovery(
            [.71,.80],[0,1],.835,tolerance=.01))
        self.assertIsNone(recurrence.first_recovery(
            [None],[0],None,tolerance=.01))

    def test_one_command_matrix_and_resource_contracts(self):
        args=runner.parser().parse_args(["--all","--math-sdpa"])
        args.experiment="all"
        jobs=list(runner.cases(args))
        self.assertEqual(len(jobs),11)
        self.assertEqual(sum(x[0]=="I" for x in jobs),3)
        self.assertEqual(sum(x[0]=="II" for x in jobs),8)
        self.assertEqual(compact.NAMES,
                         ("static","greedy_coreset","flat_utility","hierarchy"))
        smoke=runner.cmd(args,("II","switch",0,"ABA"),True)
        self.assertIn("--math-sdpa",smoke)
        self.assertEqual(smoke[smoke.index("--phase-length")+1],"64")

if __name__=="__main__":unittest.main()
