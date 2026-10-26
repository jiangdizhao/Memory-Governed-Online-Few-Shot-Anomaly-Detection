"""No-dataset unit verification of the one-and-only hierarchy decision gate."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import final_decision_core as d
import run_final_hierarchy_decision as runner

@dataclass
class S:
    path:str
    is_good:bool=True

def objects(n=620):
    # Five camera views per part (duplicates under filename-derived instance id).
    return [S(f"G:/images/switch_{i:04d}_OK_C{c}_{20231021123450+i:014d}.jpg",
              is_good=i%3!=0) for i in range(n) for c in range(1,6)]

class TestFinalGate(unittest.TestCase):
    def test_instance_key_and_fail_closed(self):
        self.assertEqual(d.object_id(
            r"G:\images\switch_0242_OK_C1_20231021144001.jpg"),
            "switch_0242_OK")
        self.assertEqual(d.object_id(
            "audiojack__0010_NG_ZW_C5_20231023100916.jpg"),
            "audiojack__0010_NG_ZW")
        with self.assertRaises(ValueError):
            d.object_id("switch_0242_OK_20231021144001.jpg")

    def test_frozen_memory_protocol(self):
        self.assertEqual((d.MemorySettings().stm,d.MemorySettings().ltm,
                          d.MemorySettings().buffer,d.MemorySettings().flat),
                         (128,64,64,256))
        for bad in ({"flat":384},{"promote_every":16},{"cap_bytes":3_500_000}):
            with self.assertRaises(ValueError):
                d.MemorySettings(**bad)

    def test_group_split_and_query_one_label_per_object(self):
        all_items=objects()
        chosen=[x.path for x in all_items[0:20:5]]
        ps,phases,manifest=d.make_split(all_items,chosen,seed=0)
        ids=manifest["A"]+manifest["B"]+sum(manifest["phases"],[])
        self.assertEqual(len(ids),573)
        self.assertEqual(len(ids),len(set(ids)))
        self.assertFalse(set(ids)&set(map(d.object_id,chosen)))
        self.assertEqual([len(x) for x in phases],[875,875,875])
        self.assertEqual(len(ps["A"]),120)
        selected=d.query_tape(phases)
        self.assertEqual(len(selected),round(.03*2625))
        observed=sum(phases,[])
        qkeys=[d.object_id(observed[t-1].path) for t in selected]
        self.assertEqual(len(set(qkeys)),len(qkeys))
        self.assertEqual(selected,d.query_tape(phases))
        with self.assertRaises(ValueError):
            d.query_tape(phases,rate=.2)

    def test_full_matrix_and_verdict_nonnegotiable(self):
        self.assertEqual(len(runner.cases()),8)
        negative=[{
            "hierarchy_shift_advantage":-.02,
            "hierarchy_end2_minus_greedy_end2":-.04,
            "hierarchy_B_end2_minus_greedy_B_end2":-.02,
            "A_retrieval_effect":False}]*4
        verdict=d.pass_decision(negative)
        self.assertFalse(verdict["keep_hierarchy"])
        self.assertEqual(verdict["decision"],"USE_GREEDY_WITH_RISK_SAFE_ADMISSION")
        positive=[{
            "hierarchy_shift_advantage":.025,
            "hierarchy_end2_minus_greedy_end2":.02,
            "hierarchy_B_end2_minus_greedy_B_end2":.02,
            "A_retrieval_effect":True}]*4
        self.assertTrue(d.pass_decision(positive)["keep_hierarchy"])
        self.assertFalse(d.pass_decision(positive[:3])["keep_hierarchy"])

    def test_dual_view_groups_remain_disjoint(self):
        items=objects()
        a,b,manifest=d.make_split(items,[items[0].path,items[5].path,
                                         items[10].path,items[15].path],seed=1)
        assert not set(manifest["A"])&set(manifest["B"])
        all_probe=set(manifest["A"]+manifest["B"])
        for phase in manifest["phases"]:
            self.assertFalse(all_probe&set(phase))

if __name__=="__main__":
    unittest.main()
