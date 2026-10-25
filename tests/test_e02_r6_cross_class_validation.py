"""CPU-only preregistered R6 cross-category validation contracts."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import e02_r6_cross_class_validation as cv


class CrossClassR6Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = cv.load_profile(
            ROOT / "outputs" / "e02_r6_class_prevalence.json")

    def test_fixed_pre_registration(self):
        self.assertEqual(cv.CLASSES, ("rolled_strip_base", "switch"))
        self.assertEqual(cv.SUPPORT_SEEDS, (0, 1))
        self.assertEqual(cv.ORDERS, ("native", "normal_first", "defect_first"))
        self.assertEqual(cv.expected_budget(3805), 114)
        self.assertEqual(cv.expected_budget(3820), 115)
        chosen = self.profile["pre_registered_candidates"]
        self.assertEqual(chosen["defect_dominated"], cv.CLASSES[0])
        self.assertEqual(chosen["near_balanced"], cv.CLASSES[1])

    def test_rejects_modified_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            profile = copy.deepcopy(self.profile)
            profile["pre_registered_candidates"]["near_balanced"] = "audiojack"
            path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Pre-registration mismatch"):
                cv.load_profile(path)

    def test_run_directory_and_settings_are_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = cv.parser().parse_args([
                "--run", "--output-root", tmp, "--dataset-root", "G:/dataset/realiad"])
            args.json_dir = Path("G:/dataset/realiad/jsons/realiad_jsons")
            native = cv.build_argv(args, "switch", 1, "native")
            stress = cv.build_argv(args, "switch", 1, "defect_first")
            self.assertIn("--max-samples", native)
            self.assertEqual(native[native.index("--max-samples")+1], "0")
            self.assertEqual(native[native.index("--support-seed")+1], "1")
            self.assertEqual(native[native.index("--query-rate")+1], "0.03")
            self.assertNotIn("--reference-native-summary", native)
            self.assertIn("--reference-native-summary", stress)
            self.assertIn("--reference-native-csv", stress)
            self.assertNotIn("--reference-r4-summary", stress)
            self.assertNotIn("--require-parity", stress)
            self.assertIn("e02_r6_switch_seed1_defect_first",
                          str(cv.run_dir(Path(tmp), "switch", 1, "defect_first")))
            for wrong in (("regulator", 0, "native"),
                          ("switch", 9, "native"), ("switch", 1, "alternating_blocks")):
                with self.assertRaises(ValueError):
                    cv.run_dir(Path(tmp), *wrong)

    def _minimal_report(self, cls="switch", seed=0, order="native"):
        counts = next(x for x in self.profile["classes"]
                      if x["class_name"] == cls)
        n = counts["total"]
        k = cv.expected_budget(n)
        return {
            "schema": "e02.r6.sidecar.audit.v1",
            "class_name": cls, "stream_order": order, "n_stream": n,
            "total_budget_each": k,
            "stream_paths_sha256": "ordered_native_hash",
            "original_stream_paths_sha256": "ordered_native_hash",
            "canonical_fixed_sha256": "same_bank_hash",
            "canonical_theta0": .5, "support_paths": ["a.png","b.png","c.png","d.png"],
            "parameters": {
                "class_name": cls, "support_seed": seed, "legacy_seed": 0,
                "stream_seed": 0, "query_seed": 37, "query_rate": .03,
                "max_samples": 0, "math_sdpa": True,
                "local_saliency": True, "variants": "D"},
            "label_windows": [{
                "normal_count": counts["normal"],
                "defect_count": counts["defect"],
                "balanced_rescue": {"fnr": .6}}],
            "audit": {"n_update_attempts": 6, "accepted_steps": [1,10],
                      "outcome_counts": {"accepted": 2, "cooldown_blocked": 4}},
            "schedules": {"D": {
                "queries": k, "query_indices_1based": list(range(1,k+1)),
                "queried_normals": 10, "queried_defects": k-10,
                "budget_forced_terminal": 0, "longest_query_free_gap": 20,
                "first_400_queries": 30,
                "stm_write_calls": 10, "stm_prototypes": 300,
                "fixed_sha256": "same_bank_hash",
                "arm_metrics": {
                    "balanced_rescue": {"balanced_accuracy": .6,
                                        "fpr": .1, "fnr": .7, "theta_final": .44},
                    "mistake_feedback": {"balanced_accuracy": .62},
                    "frozen": {"auroc": .81}
                }}}

    def test_native_and_stress_lineage(self):
        counts = next(x for x in self.profile["classes"]
                      if x["class_name"] == "switch")
        native = self._minimal_report()
        n = cv._check_report(native, "switch", 0, "native", counts)
        self.assertEqual(n["query_budget"], 115)
        self.assertEqual(n["worst_defined_window_balanced_fnr"], .6)
        stress = self._minimal_report(order="normal_first")
        stress["stream_paths_sha256"] = "reordered"
        stress["stress_membership_check"] = {"passed": True}
        stress["reference_check"] = {"passed": True}
        row = cv._check_report(stress,"switch",0,"normal_first", counts,native=native)
        self.assertTrue(row["native_reference_validated"])
        stress["canonical_fixed_sha256"] = "different"
        with self.assertRaisesRegex(AssertionError, "fixed bank differs"):
            cv._check_report(stress, "switch", 0, "normal_first",
                             counts, native=native)

    def test_early_failure_and_output_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = cv.parser().parse_args([
                "--run", "--class-name", "switch",
                "--output-root", tmp])
            dest = cv.run_dir(args.output_root, "switch", 0, "native")
            dest.mkdir()
            with self.assertRaises(FileExistsError):
                cv._run(args, self.profile)

    def test_missing_class_risk_is_not_falsely_zeroed(self):
        counts = next(x for x in self.profile["classes"]
                      if x["class_name"] == "rolled_strip_base")
        report = self._minimal_report(cls="rolled_strip_base")
        report["label_windows"] = [
            {"normal_count": counts["normal"], "defect_count": 0,
             "balanced_rescue": {"fnr": None}},
            {"normal_count": 0, "defect_count": counts["defect"],
             "balanced_rescue": {"fnr": .8}},
        ]
        row = cv._check_report(
            report, "rolled_strip_base", 0, "native", counts)
        self.assertEqual(row["n_windows_without_defect"], 1)
        self.assertEqual(row["n_windows_without_normal"], 1)
        self.assertEqual(row["worst_defined_window_balanced_fnr"], .8)


if __name__ == "__main__":
    unittest.main()
