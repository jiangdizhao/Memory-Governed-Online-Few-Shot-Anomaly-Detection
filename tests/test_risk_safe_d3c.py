"""D3-C CPU tests: causal random mask design and frozen native control contract."""
from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.risk_safe.conditional_retrieval import eligible_mask,coordinates
from experiments.risk_safe.retrieval_random_controls import (
    REPEATS,SOURCE_CALIB_N,SOURCE_CALIB_P_GRID,CONTROL_TYPES,
    RandomEligibility,shuffled_spatial_provenance,choose_probability)
from experiments.risk_safe import d3_controlled_retrieval as ctrl
from experiments.risk_safe import conditional_retrieval_pilot as original
from experiments.risk_safe.native_bridge import NativeParityError
from experiments.risk_safe.d2_metrics import image_metrics


class NativeGrid:
    @staticmethod
    def _grid_hw_from_npatches(n, aspect=1):
        if n!=16:raise ValueError("Synthetic 4-by-4 grid only")
        return 4,4


def proto():
    return [
        {"prototype_index":i,"camera_id":i%5+1,
         "u":(i//4+.5)/4,"v":(i%4+.5)/4,
         "sample_key":f"training-{i}"} for i in range(16)]


class ContextShuffleTests(unittest.TestCase):
    def test_permutation_preserves_all_embedding_indices_and_coordinate_multiset(self):
        p=proto()
        original=copy.deepcopy(p)
        randomized=shuffled_spatial_provenance(p,method="greedy",replicate=0)
        self.assertEqual(sorted((x["u"],x["v"]) for x in randomized),
                         sorted((x["u"],x["v"]) for x in p))
        for i,x in enumerate(randomized):
            self.assertEqual(x["prototype_index"],i)
            self.assertEqual(x["camera_id"],p[i]["camera_id"])
            self.assertEqual(x["sample_key"],p[i]["sample_key"])
        self.assertEqual(p,original)
        self.assertNotEqual(randomized,p)

    def test_seed_reproducibility_and_distinct_repeats(self):
        p=proto()
        a=shuffled_spatial_provenance(p,method="greedy",replicate=1)
        b=shuffled_spatial_provenance(p,method="greedy",replicate=1)
        c=shuffled_spatial_provenance(p,method="greedy",replicate=2)
        self.assertEqual(a,b)
        self.assertNotEqual(a,c)


class RandomEligibilityTests(unittest.TestCase):
    def test_exact_per_patch_count_and_topfive_availability(self):
        p=proto()
        uv=coordinates(NativeGrid,16,(100,100))
        true_mask=eligible_mask(p,"spatial_0125",query_camera=2,
                                query_coords=uv)
        f=RandomEligibility("matched_count_random",method="greedy",replicate=0)
        mask=f(p,"spatial_0125",query_camera=2,query_coords=uv,
               image_id="real/heldout_A/C2.jpeg")
        self.assertTrue(torch.equal(true_mask.sum(1),mask.sum(1)))
        self.assertTrue(torch.equal(true_mask.sum(1)>=5,mask.sum(1)>=5))
        self.assertEqual(mask.dtype,torch.bool)
        self.assertEqual(tuple(mask.shape),(16,16))

    def test_random_control_repeatable_across_calls(self):
        p=proto()
        uv=coordinates(NativeGrid,16,(100,100))
        for kind,prob in (("matched_count_random",None),
                          ("source_calibrated_hit_random",.4)):
            f=RandomEligibility(kind,method="greedy",replicate=2,probability=prob)
            a=f(p,"spatial_0125",query_camera=1,query_coords=uv,image_id="a")
            b=f(p,"spatial_0125",query_camera=1,query_coords=uv,image_id="a")
            self.assertTrue(torch.equal(a,b))

    def test_p_one_makes_every_prototype_available(self):
        f=RandomEligibility("source_calibrated_hit_random",
                            method="causal_novelty",replicate=0,probability=1.)
        mask=f(proto(),"spatial_0125",query_camera=5,
               query_coords=coordinates(NativeGrid,16,(100,100)),
               image_id="normal_source_only")
        self.assertTrue(bool(mask.all()))

    def test_fails_closed_on_bad_source_or_probability(self):
        with self.assertRaises(NativeParityError):
            RandomEligibility("matched_count_random",method="greedy",
                              replicate=0,probability=.1)
        with self.assertRaises(NativeParityError):
            RandomEligibility("source_calibrated_hit_random",method="greedy",
                              replicate=0,probability=1.2)
        f=RandomEligibility("matched_count_random",method="greedy",replicate=0)
        with self.assertRaises(NativeParityError):
            f(proto(),"global",query_camera=1,
              query_coords=coordinates(NativeGrid,16,(100,100)),image_id="a")


class SourceOnlyHitCalibrationTests(unittest.TestCase):
    def test_tie_breaks_to_smaller_probability(self):
        vals={p:.1 for p in SOURCE_CALIB_P_GRID}
        r=choose_probability(.1,vals)
        self.assertEqual(r["selected_p"],.05)
        self.assertFalse(r["probe_used_in_selection"])
        self.assertEqual(r["source_absolute_hit_gap"],0)

    def test_calibration_uses_closed_grid(self):
        vals={p:p for p in SOURCE_CALIB_P_GRID}
        self.assertEqual(choose_probability(.37,vals)["selected_p"],.4)
        with self.assertRaises(NativeParityError):
            choose_probability(.4,{.2:.4})
        vals[.05]=float("nan")
        with self.assertRaises(NativeParityError):
            choose_probability(.4,vals)

    def test_repeats_and_source_N_are_frozen(self):
        args=ctrl.parser().parse_args([
            "--reference-d2a-summary","old/summary.json",
            "--threshold-registry","train/threshold_registry.json",
            "--fixed-quantile","0.90"])
        ctrl.enforce(args)
        self.assertEqual(REPEATS,(0,1,2))
        self.assertEqual(SOURCE_CALIB_N,8)
        self.assertEqual(len(CONTROL_TYPES),4)
        args.control_repeats=2
        with self.assertRaises(NativeParityError):
            ctrl.enforce(args)


class FrozenReferenceTests(unittest.TestCase):
    def test_reference_has_to_be_original_native_d3(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                ctrl.validate_reference_d3(Path(td))

    def test_replay_mismatch_fail_closed(self):
        with self.assertRaises((KeyError,NativeParityError)):
            ctrl.verify_replayed_d3(
                {"schema":"bad"},{"retrieval_probe_scores.csv":""},
                {"schema":"risk_safe.d3.conditional_retrieval_mechanism_pilot.v2"},
                [])

    def test_metrics_report_distinct_adaptation_and_erosion(self):
        scores=[
            {"label_offline_only":0,"score":.3,"static_score":.4,
             "eligible_top5_fraction":1.,"dynamic_win_fraction":.5},
            {"label_offline_only":0,"score":.4,"static_score":.5,
             "eligible_top5_fraction":0.,"dynamic_win_fraction":0.},
            {"label_offline_only":1,"score":.8,"static_score":.9,
             "eligible_top5_fraction":1.,"dynamic_win_fraction":.5},
            {"label_offline_only":1,"score":.7,"static_score":.7,
             "eligible_top5_fraction":0.,"dynamic_win_fraction":0.},
        ]
        m=ctrl.summary_metrics(
            scores,theta=.6,global_metrics={"static_fpr":0.5,"global_fnr":0.})
        self.assertAlmostEqual(m["normal_mean_score_gain"],.1)
        self.assertAlmostEqual(m["defect_mean_score_erosion"],.05)
        self.assertAlmostEqual(m["mean_top5_eligible_fraction"],.5)


if __name__=="__main__":
    unittest.main()
