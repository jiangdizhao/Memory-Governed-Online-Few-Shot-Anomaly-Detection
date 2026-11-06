from __future__ import annotations

from dataclasses import dataclass
import tempfile
from pathlib import Path
import unittest

import numpy as np

from experiments.calibration import a_calibration_scale as a


@dataclass(frozen=True)
class FakeTrainImage:
    sample_key: str
    relative_id: str
    is_good: bool = True


def fake_roles(group_count: int = 250):
    rows = []
    for index in range(group_count):
        group = f"g{index:03d}"
        count = 5 if index % 3 == 0 else 1
        rows.extend(FakeTrainImage(group, f"{group}/view{view}.png")
                    for view in range(count))
    supports = {
        "0": {"groups": [f"g{i:03d}" for i in range(4)]},
        "1": {"groups": [f"g{i:03d}" for i in range(4, 8)]},
    }
    c64 = {"groups": [f"g{i:03d}" for i in range(8, 72)]}
    return rows, supports, c64


class CalibrationMathTests(unittest.TestCase):
    def test_higher_quantile_index_n32(self):
        self.assertEqual(a.higher_index(32), 28)

    def test_higher_quantile_index_n64(self):
        self.assertEqual(a.higher_index(64), 57)
        self.assertEqual(a.higher_quantile(np.arange(64)), 57.0)

    def test_higher_quantile_index_n128(self):
        self.assertEqual(a.higher_index(128), 115)

    def test_sample_standard_deviation_uses_ddof1(self):
        mean, std = a.sample_mean_std([1, 2, 3])
        self.assertEqual(mean, 2.0)
        self.assertAlmostEqual(std, 1.0)

    def test_gaussian_multipliers_are_registered(self):
        values = [0.0, 1.0, 2.0]
        mean, std = a.sample_mean_std(values)
        self.assertAlmostEqual(a.gaussian_threshold(values, 1.2815515655),
                               mean + 1.2815515655 * std)
        self.assertAlmostEqual(a.gaussian_threshold(values, 2.5), mean + 2.5 * std)

    def test_fixed05_is_raw_score_threshold(self):
        thresholds = a.threshold_methods({"C64": [0.1, 0.2, 0.3]},
                                        [0.1, 0.2, 0.3, 0.4],
                                        [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(thresholds["FIXED05"], 0.5)
        self.assertEqual(sum(np.asarray([0.49, 0.5]) >= thresholds["FIXED05"]), 1)

    def test_loso_excludes_heldout_and_all_augmentation_descendants(self):
        supports = ["s0", "s1", "s2", "s3"]
        descendants = {key: [key + "-aug1", key + "-aug2"] for key in supports}
        refs = a.loso_reference_ids(supports, descendants, "s2")
        self.assertNotIn("s2", refs)
        self.assertNotIn("s2-aug1", refs)
        self.assertNotIn("s2-aug2", refs)
        self.assertEqual(len(refs), 9)

    def test_loso_uses_three_support_roots(self):
        refs = a.loso_reference_ids(["a", "b", "c", "d"], {}, "d")
        self.assertEqual(refs, ("a", "b", "c"))
        self.assertNotIn("d", refs)


class DataRoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.supports, cls.c64 = fake_roles()
        cls.roles = a.higher_order_membership(
            cls.rows, cls.supports, cls.c64, category="audiojack")

    def test_c32_c64_c128_nested_membership(self):
        draw = self.roles["draws"]["0"]
        self.assertEqual(len(draw["C32"]), 32)
        self.assertEqual(len(draw["C64"]), 64)
        self.assertEqual(len(draw["C128"]), 128)
        self.assertTrue(set(draw["C32"]) <= set(draw["C64"]) <= set(draw["C128"]))

    def test_historical_c64_preserved_exactly(self):
        self.assertEqual(tuple(self.roles["draws"]["0"]["C64"]),
                         tuple(self.c64["groups"]))

    def test_cmax_g_and_cmax_v_group_view_counts(self):
        self.assertEqual(self.roles["Cmax_V_group_count"],
                         len(self.roles["eligible_calibration_groups"]))
        self.assertGreater(self.roles["Cmax_V_image_count"],
                           self.roles["Cmax_V_group_count"])
        self.assertEqual(self.roles["Cmax_V_image_count"],
                         len(self.roles["Cmax_V_images"]))

    def test_h_is_disjoint_from_calibration_and_reference_pool(self):
        d = self.roles["disjointness"]
        self.assertTrue(d["H_vs_calibration"])
        self.assertTrue(d["H_vs_A3b_reference"])

    def test_support_calibration_and_test_groups_are_disjoint(self):
        self.assertTrue(a.verify_disjoint_roles(
            self.roles["support_union_groups"],
            self.roles["H_groups"],
            self.roles["A3b_reference_building_groups"]))

    def test_distinct_sampling_draws_have_distinct_ids(self):
        draws = self.roles["draws"]
        self.assertNotEqual(draws["0"]["C64"], draws["1"]["C64"])
        self.assertNotEqual(draws["1"]["C64"], draws["2"]["C64"])

    def test_a3b_reference_pool_excludes_support_h_and_c64(self):
        d = self.roles["disjointness"]
        self.assertTrue(d["support_vs_historical_C64"])
        self.assertTrue(d["historical_C64_vs_H"])
        self.assertTrue(d["historical_C64_vs_A3b_reference"])


class CausalAndEvaluationTests(unittest.TestCase):
    def test_a2_readout_does_not_modify_shared_writer(self):
        writer = a.SharedWriterState(0.5, 0.02, ["initial"])
        before = (list(writer.memory), writer.state_version, list(writer.write_log))
        arm = a.A2Readout(0.5, 0.01)
        arm.update(version=1, calibration_scores=np.linspace(0.1, 0.9, 64))
        self.assertEqual(before, (writer.memory, writer.state_version, writer.write_log))

    def test_a3_trajectories_can_legitimately_diverge(self):
        c64 = a.IndependentTrajectory("C64", 0.5, 0.02, [], [], [])
        cmax = a.IndependentTrajectory("Cmax-G", 0.7, 0.02, [], [], [])
        self.assertNotEqual(c64.prequential_prediction(0.6),
                            cmax.prequential_prediction(0.6))
        c64.accept_verified_feedback(position=0, label=0, source_digest="x",
                                     correction_digest="c1")
        cmax.accept_verified_feedback(position=0, label=0, source_digest="x",
                                      correction_digest="c1")
        self.assertEqual(len(c64.memory), 1)
        self.assertEqual(len(cmax.memory), 1)
        self.assertIsNot(c64.memory, cmax.memory)

    def test_b3_clips_each_update_to_plus_minus_001(self):
        value, last, did = a.b3_step(0.5, 0.9, version=1, last_version=None)
        self.assertAlmostEqual(value, 0.51)
        self.assertEqual(last, 1)
        self.assertTrue(did)

    def test_b3_updates_once_per_real_state_version(self):
        first = a.b3_step(0.5, 0.9, version=4, last_version=None)
        second = a.b3_step(first[0], 0.1, version=4, last_version=first[1])
        self.assertTrue(first[2])
        self.assertFalse(second[2])
        self.assertEqual(second[0], first[0])

    def test_prediction_is_computed_without_current_label(self):
        trajectory = a.IndependentTrajectory("P0", 0.5, 0.02, [], [], [])
        prediction = trajectory.prequential_prediction(0.6)
        self.assertEqual(prediction, 1)
        self.assertEqual(trajectory.predictions, [1])

    def test_query_gate_discloses_only_sealed_next_query(self):
        gate = a.QueryOnlyLabelGate([0, 1, 0], [1])
        with self.assertRaises(a.ExperimentAError):
            gate.reveal(1)
        gate.seal_prediction(1)
        self.assertEqual(gate.reveal(1), 1)
        self.assertEqual(gate.disclosed, [1])

    def test_fixed_query_identity_parity_is_exact(self):
        self.assertTrue(a.verify_query_identity(["q1", "q2"], ["q1", "q2"]))
        with self.assertRaises(a.ExperimentAError):
            a.verify_query_identity(["q1", "q2"], ["q2", "q1"])

    def test_threshold_only_arms_do_not_change_auroc(self):
        result = a.audit_threshold_auroc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8],
                                         [0.2, 0.5, 0.9])
        self.assertTrue(result["identical"])
        self.assertEqual(len(set(result["auroc_by_arm"])), 1)

    def test_single_class_window_is_unavailable(self):
        m = a.metrics_for_threshold([0, 0], [0.1, 0.2], 0.5)
        self.assertTrue(np.isnan(m["ber"]))
        windows = a.window_metrics([0] * 512, [0.1] * 512, 0.5)
        self.assertEqual(windows[0]["status"], "UNAVAILABLE_SINGLE_CLASS")

    def test_stage_a_seal_is_required_and_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stage_a_seal.json"
            with self.assertRaises(a.ExperimentAError):
                a.require_stage_a_seal(path)
            a.create_stage_a_seal(path, run_id="test", input_digests={"x": "a"},
                                  score_digests={"y": "b"})
            self.assertEqual(a.require_stage_a_seal(path)["stage"], "A_SEALED")

    def test_resume_refuses_changed_source_or_input_digests(self):
        a.check_resume_identity({"source": "x", "input": "y"},
                                {"source": "x", "input": "y"})
        with self.assertRaises(a.ExperimentAError):
            a.check_resume_identity({"source": "x", "input": "y"},
                                    {"source": "z", "input": "y"})

    def test_cmax_cannot_be_relabelled_as_c128(self):
        with self.assertRaises(a.ExperimentAError):
            a.require_cmax_complete("HOLD_RESOURCE_LIMIT")
        a.require_cmax_complete("COMPLETE_VALID")

    def test_only_authorized_development_categories_are_allowed(self):
        for category in a.CLASSES:
            a.authorized_category(category)
        with self.assertRaises(a.ExperimentAError):
            a.authorized_category("mint")
        with self.assertRaises(a.ExperimentAError):
            a.authorized_category("pcb")

    def test_mask_access_is_rejected(self):
        a.assert_no_masks_accessed(["train/good/image.png"])
        with self.assertRaises(a.ExperimentAError):
            a.assert_no_masks_accessed(["test/defect_mask.png"])

    def test_legacy_modification_audit_rejects_unexpected_path(self):
        a.assert_no_legacy_paths(["experiments/calibration/a_calibration_scale.py"])
        with self.assertRaises(a.ExperimentAError):
            a.assert_no_legacy_paths(["experiments/calibration/r9_runner.py"])


if __name__ == "__main__":
    unittest.main()
