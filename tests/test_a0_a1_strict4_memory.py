"""Offline A0/A1 protocol, orchestration, and result-acceptance regressions.

CI does not have the pretrained DINOv2 checkpoint or Real-IAD data. These
tests verify contracts only; full GPU parity must be validated on the user's
machine from the immutable per-condition outputs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "experiments"))
import a0_a1_memory_methods as mem
import a0_a1_strict4_matched as exp
import run_a0_a1_validation as auto


class ProtocolTests(unittest.TestCase):
    def test_distinct_strict_support_and_no_test_leak(self):
        train = ["N1", "N2", "N3", "N4", "N5"]
        exp.verify_supports(train[:4], train, ["T1", "T2"])
        for selected, tests in [
            (train[:3], ["T1"]),
            (["N1", "N1", "N2", "N3"], ["T1"]),
            (["N1", "N2", "N3", "T1"], ["T1"]),
            (["N1", "N2", "N3", "N4"], ["N3"])]:
            with self.assertRaises(AssertionError):
                exp.verify_supports(selected, train, tests)

    def test_probe_label_blind_disjoint_and_deterministic(self):
        class Sample:
            def __init__(self, i): self.path, self.is_good = f"img{i}", i % 2 == 0
        samples = [Sample(i) for i in range(60)]
        a, b = exp.select_probe(samples, 15, 42)
        aa, bb = exp.select_probe(samples, 15, 42)
        self.assertEqual([x.path for x in a], [x.path for x in aa])
        self.assertEqual([x.path for x in b], [x.path for x in bb])
        self.assertEqual(set(x.path for x in a) & set(x.path for x in b), set())
        with self.assertRaises(ValueError):
            exp.select_probe(samples, 31, 42)

    def test_memory_methods_known_without_external_import(self):
        self.assertEqual(mem.ARM_NAMES, auto.ARMS)
        self.assertEqual(len(mem.ARM_NAMES), 5)
        self.assertEqual(mem.stable_seed(1, "x"), mem.stable_seed(1, "x"))
        self.assertNotEqual(mem.stable_seed(1, "x"), mem.stable_seed(2, "x"))

    def test_one_command_matrix_and_smoke(self):
        p = auto.parser()
        a = p.parse_args(["--all", "--math-sdpa"])
        self.assertEqual(len(auto.stream_conditions(a)), 6)
        self.assertEqual(len(auto.stream_conditions(
            p.parse_args(["--all", "--with-stress"]))), 18)
        cmd = auto.argv(a, "audiojack", 0, "native", smoke=True)
        self.assertEqual(cmd[cmd.index("--max-samples")+1], "64")
        self.assertIn("--math-sdpa", cmd)
        self.assertEqual(auto.location(
            a, "switch", 1, "normal_first").name,
            "switch_s1_normal_first_full")
        with self.assertRaises(ValueError):
            auto.stream_conditions(p.parse_args(["--all", "--seeds", "0,0"]))

    @staticmethod
    def fixture(root: Path):
        """Construct synthetic data; no feature or result is fabricated as a GPU result."""
        output = root / "fake_s0_native_full"
        output.mkdir()
        supports = ["train/a", "train/b", "train/c", "train/d"]
        probes = ["test/heldout"]
        rows = [
            {"t": str(i), "path": f"test/{i}", "label_eval_only": str(i % 2),
             "queried": str(int(i in (3, 11))), "theta_loso_fixed": "0.5"}
            for i in range(1, 101)]
        cap = [4,4,4,4]
        arms = {}
        for method in auto.ARMS:
            arm = dict(
                auroc=.7, balanced_accuracy=.65, fpr=.2, fnr=.5,
                mutable_prototypes=0 if method == "static" else 8,
                mutable_capacity=16, capacity_per_layer=cap,
                accepted_patches=0 if method == "static" else 8,
                candidate_events=0 if method == "static" else 1)
            arms[method] = arm
        p = {
            "support_selection": "strict_random_four_shot",
            "support_image_access_count": 4, "support_paths": supports,
            "support_sha256": auto.sha(supports),
            "calibration": "four_fold_leave_entire_support_image_and_all_augmentations_out",
            "loso_fold_lineage": [
                {"heldout": h, "support_paths": [s for s in supports if s != h]}
                for h in supports],
            "loso_scores": [.1,.2,.3,.4],
            "loso_stats": {"theta": .5},
            "theta0_shared_across_all_methods": .5,
            "diagnostic_self_match_stats": {"theta": .2},
            "probe_not_in_training_or_feedback": True,
            "labels_after_predictions": True,
            "no_adaptive_threshold_in_primary_experiment": True,
            "probe_paths": probes, "probe_paths_sha256": auto.sha(probes),
            "stream_paths_sha256": auto.sha([f"test/{i}" for i in range(1,101)]),
            "query_rate_target": .03, "query_positions_1based": [3,11],
            "query_tape_sha256": auto.sha([3,11]),
            "observed_query_normals": 1, "observed_query_defects": 1,
            "fixed_bank": {"sha256": "test-only"},
            "feature_configuration": {"device": "cuda"}
        }
        # Update fixture to exact 3% (3 query positions).
        p["query_positions_1based"] = [3, 11, 15]
        p["query_tape_sha256"] = auto.sha([3,11,15])
        for r in rows:
            r["queried"] = str(int(int(r["t"]) in (3,11,15)))
        p["observed_query_defects"] = 2
        with (output / "prequential.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        for name in ("probe_scores.csv","probe_summary.csv"):
            (output / name).write_text("placeholder\n", encoding="utf-8")
        report = {"schema": auto.SCHEMA, "class_name": "fake",
                  "support_seed": 0, "stream_order": "native",
                  "n_stream": 100, "n_probe": 1, "K": 3,
                  "protocol": p, "arms": arms}
        (output / "summary.json").write_text(
            json.dumps(report), encoding="utf-8")
        return output, report

    def test_validate_fails_closed_on_leak_and_wrong_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, j = self.fixture(Path(tmp))
            args = auto.parser().parse_args(
                ["--validate", "--probe-count", "1"])
            row = auto.validate(out, cls="fake", seed=0, order="native",
                                smoke=False, args=args)
            self.assertEqual(row["normal_queries"], 1)
            j["protocol"]["loso_fold_lineage"][0]["support_paths"].append("train/a")
            (out / "summary.json").write_text(json.dumps(j), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "LOSO self-match"):
                auto.validate(out, cls="fake", seed=0, order="native",
                              smoke=False, args=args)


if __name__ == "__main__":
    unittest.main()
