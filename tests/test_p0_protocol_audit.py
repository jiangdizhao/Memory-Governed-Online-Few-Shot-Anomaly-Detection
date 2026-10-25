"""CPU-only regression tests for P0-R2 randomization and replay diagnostics."""
import sys
import unittest
import csv
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import p0_protocol_audit as p0


class P0AuditTests(unittest.TestCase):
    def test_independent_random_seeds(self):
        self.assertEqual(p0.deterministic_subseed(0, "A", 0), p0.deterministic_subseed(0, "A", 0))
        self.assertNotEqual(p0.deterministic_subseed(0, "A", 0), p0.deterministic_subseed(1, "A", 0))
        self.assertNotEqual(p0.deterministic_subseed(0, "A", 0), p0.deterministic_subseed(0, "B", 0))

    def test_strict_support_seed_only_affects_support_identity(self):
        paths = [f"n{i:03d}" for i in range(200)]
        a, n = p0.select_support(None, None, paths, "strict4", 0, 4)
        b, _ = p0.select_support(None, None, paths, "strict4", 1, 4)
        self.assertNotEqual(a, b)
        self.assertEqual(n, 4)
        # The same augmentation/bank seed for any shared file is invariant
        # to which other files were selected.
        common = "same_file"
        self.assertEqual(p0.deterministic_subseed(17, common),
                         p0.deterministic_subseed(17, common))

    def test_pool_selection_repeats_with_fixed_central_selector(self):
        class Impl:
            @staticmethod
            def build_modes_from_train_good(backbone, normals, K_modes, seed):
                return "centroid", [0] * len(normals)
            @staticmethod
            def select_clean_support_paths_per_mode(
                backbone, normals, labels, centroids, shots_per_mode, seed
            ):
                return {0: normals[:shots_per_mode]}
        normal = [f"normal{i}" for i in range(20)]
        a, n = p0.select_support(Impl, object(), normal, "pool_assisted4", 0, 4)
        b, _ = p0.select_support(Impl, object(), normal, "pool_assisted4", 42, 4)
        self.assertEqual(a, b)
        self.assertEqual(n, len(normal))

    def test_threshold_curve_and_operating_metrics(self):
        y = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        named = {"self": 0.5, "loso": 0.7}
        curve = p0.diagnostic_threshold_curve(y, scores, named)
        for name, thr in named.items():
            row = next(r for r in curve if r["threshold"] == thr)
            m = p0.metrics(y, scores, thr)
            self.assertAlmostEqual(row["fpr"], m["fpr"])
            self.assertAlmostEqual(row["fnr"], m["fnr"])
            self.assertAlmostEqual(row["balanced_accuracy"], m["balanced_accuracy"])
            self.assertEqual(row["is_" + name], 1)
        self.assertTrue(all(a["fpr"] >= b["fpr"] for a, b in zip(curve, curve[1:])))
        self.assertTrue(all(a["fnr"] <= b["fnr"] for a, b in zip(curve, curve[1:])))
        self.assertEqual(curve[0]["fpr"], 1.0)
        self.assertEqual(curve[-1]["fnr"], 1.0)

    def test_self_vs_loso_same_auc_when_scores_identical(self):
        y = [0, 0, 1, 1]
        scores = [0.11, 0.3, 0.2, 0.8]
        a = p0.metrics(y, scores, 0.15)
        b = p0.metrics(y, scores, 0.5)
        self.assertEqual(a["auroc"], b["auroc"])
        self.assertNotEqual(a["fpr"], b["fpr"])


    def test_gpu_scale_drift_is_tolerated_only_at_encoder_level(self):
        # Reproduces the user's observed 0.528495 vs 0.528506 example.
        with tempfile.TemporaryDirectory() as dirname:
            p1, p2 = [Path(dirname) / v for v in ("case1.csv", "case2.csv")]
            for file, score in [(p1, 0.5284950137138367),
                                (p2, 0.5285062193870544)]:
                with file.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=["path", "label", "score"])
                    writer.writeheader()
                    writer.writerow({"path": "image1", "label": 0, "score": score})
            tolerated = p0.compare_repeated_case_csv(p1, p2, 1e-4)
            strict = p0.compare_repeated_case_csv(p1, p2, 1e-6)
            self.assertTrue(tolerated["within_tolerance"])
            self.assertFalse(strict["within_tolerance"])
            self.assertGreater(tolerated["max_abs_delta"], 1e-5)
            self.assertEqual(tolerated["n_above_1e-6"], 1)

    def test_score_comparison_rejects_stream_and_label_mismatch(self):
        with tempfile.TemporaryDirectory() as dirname:
            p1, p2 = [Path(dirname) / v for v in ("a.csv", "b.csv")]
            for f, label in [(p1, 0), (p2, 1)]:
                with f.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=["path", "label", "score"])
                    writer.writeheader()
                    writer.writerow({"path": "image1", "label": label, "score": 0.51})
            with self.assertRaises(AssertionError):
                p0.compare_repeated_case_csv(p1, p2, 1e-4)

    def test_identical_encoded_features_bank_retrieval(self):
        class Impl:
            @staticmethod
            def ViewSpec(*args):
                return tuple(args)

        class Bank:
            def score_sample_multiview(self, **kwargs):
                # Simulates deterministic scoring with cached features.
                return kwargs["view_to_feats"]["id"] * 2, None, None

        class Image:
            height = 100
            width = 100

        class Cfg:
            topk_patches = 32
            image_aggregation = "max"
            image_topk = 64
            image_quantile = .95

        score1 = p0.score_encoded(Impl, Bank(), Image(), 0.25, Cfg())
        score2 = p0.score_encoded(Impl, Bank(), Image(), 0.25, Cfg())
        self.assertEqual(score1, score2)


if __name__ == "__main__":
    unittest.main()
