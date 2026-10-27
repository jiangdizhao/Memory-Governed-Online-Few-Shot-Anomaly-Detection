"""D2-A CPU tests: frozen normal-only threshold calibration and causal barriers."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.risk_safe.d2_fixed_threshold import (
    SCHEMA, QUANTILES, fit_fixed_normal_thresholds,
    canonical_quantile_name, validate_separation, validate_registry,
    ThresholdAuditError, REGISTRY, MANIFEST)
from experiments.risk_safe import d2_native_replay as runner
from experiments.risk_safe import d2_threshold_calibration as calib
from experiments.risk_safe import preflight as pf


@dataclass
class Rec:
    sample_key: str
    rel_path: str
    is_good: bool = True
    camera_id: int = 1
    path: str = "ignored"


class Plan:
    def __init__(self):
        self.history = tuple(Rec(f"train_{i:03}", f"train_{i:03}_C1.jpg")
                             for i in range(64))
        self.support_samples_by_seed = {0: ("support0", "support1", "support2", "support3")}
        self.groups = {
            "support": set(self.support_samples_by_seed[0]),
            "mature_history": {r.sample_key for r in self.history},
            "stream": {"online1", "online2"},
            "probe_A": {"probeA"},
            "probe_B": {"probeB"},
        }

    def sample_sets(self):
        return {k: set(v) for k, v in self.groups.items()}


def registry(plan, index_hash):
    history = validate_separation(plan, plan.support_samples_by_seed[0])
    scores = [0.35 + 0.002 * i for i in range(64)]
    import numpy as np
    return {
        "schema": SCHEMA, "class_name": "audiojack",
        "support_seed": 0,
        "normal_calibration_partition": "reserved_mature_history_training_normal",
        "support_keys": sorted(plan.support_samples_by_seed[0]),
        "calibration_sample_keys": [r.sample_key for r in history],
        "calibration_paths": [r.rel_path for r in history],
        "calibration_normal_scores": scores,
        "sigma_dev_normal": float(np.std(scores, ddof=1)),
        "registered_quantiles": fit_fixed_normal_thresholds(scores),
        "realiad_class_json_sha256": index_hash,
        "probes_used_for_calibration": False,
        "online_stream_used_for_calibration": False,
        "loso_reference_theta0": 0.62,
        "loso_reference_sigma": 0.023,
    }


class FixedNormalQuantileTests(unittest.TestCase):
    def test_higher_rank_and_registered_grid(self):
        f = fit_fixed_normal_thresholds(list(range(64)))
        self.assertEqual(f["0.90"]["rank_zero_based"], 57)
        self.assertEqual(f["0.95"]["rank_zero_based"], 60)
        self.assertEqual(f["0.90"]["theta"], 57.)
        self.assertEqual(f["0.95"]["theta"], 60.)
        self.assertEqual(QUANTILES, (0.90, 0.95))

    def test_quantile_disallows_ad_hoc_probe_tuning(self):
        with self.assertRaises(ThresholdAuditError):
            canonical_quantile_name(0.97)
        with self.assertRaises(ThresholdAuditError):
            fit_fixed_normal_thresholds([0.2] * 20)
        with self.assertRaises(ThresholdAuditError):
            fit_fixed_normal_thresholds([float("nan")] * 64)

    def test_train_history_is_unique_normal_and_disjoint(self):
        p = Plan()
        self.assertEqual(len(validate_separation(p, p.support_samples_by_seed[0])), 64)
        p.history = (Rec("support0", "support0.jpg"),) + p.history[1:]
        with self.assertRaises(ThresholdAuditError):
            validate_separation(p, p.support_samples_by_seed[0])

    def test_no_probe_in_calibration_and_index_hash_match(self):
        p = Plan()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root / "audiojack.json"
            official.write_text("official-index", encoding="utf-8")
            reg = registry(p, pf.sha256_file(official))
            validated = validate_registry(reg, plan=p, class_name="audiojack",
                support_seed=0, class_json_path=official,
                registry_path=root / REGISTRY, strict_manifest=False)
            self.assertEqual(validated["n_training_normal_views"], 64)
            self.assertGreater(validated["theta_by_quantile"]["0.95"]["theta"],
                               validated["theta_by_quantile"]["0.90"]["theta"])
            altered = copy.deepcopy(reg)
            altered["probes_used_for_calibration"] = True
            with self.assertRaises(ThresholdAuditError):
                validate_registry(altered, plan=p, class_name="audiojack",
                    support_seed=0, class_json_path=official,
                    registry_path=root / REGISTRY, strict_manifest=False)
            altered = copy.deepcopy(reg)
            altered["registered_quantiles"]["0.95"]["theta"] += 1.
            with self.assertRaises(ThresholdAuditError):
                validate_registry(altered, plan=p, class_name="audiojack",
                    support_seed=0, class_json_path=official,
                    registry_path=root / REGISTRY, strict_manifest=False)

    def test_registry_source_manifest_integrity(self):
        p = Plan()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root / "audiojack.json"
            official.write_text("official-index", encoding="utf-8")
            reg = registry(p, pf.sha256_file(official))
            pf.write_json_idempotent(root / REGISTRY, reg)
            pf.write_json_idempotent(root / MANIFEST, {
                "schema": "risk_safe.d2a.fixed_threshold_source.v1",
                "source_sha256": {"some_source": "example"},
                "input_sha256": {"realiad_class_json": pf.sha256_file(official)},
                "output_sha256": {REGISTRY: pf.sha256_file(root / REGISTRY)},
            })
            v = validate_registry(reg, plan=p, class_name="audiojack",
                support_seed=0, class_json_path=official,
                registry_path=root / REGISTRY)
            self.assertEqual(v["n_training_normal_views"], 64)
            (root / REGISTRY).write_text('{"tampered":true}', encoding="utf-8")
            with self.assertRaises(ThresholdAuditError):
                validate_registry(reg, plan=p, class_name="audiojack",
                    support_seed=0, class_json_path=official,
                    registry_path=root / REGISTRY)

    def test_threshold_fit_deterministic_and_no_defect_label_input(self):
        s = [0.52 + i * 0.0001 for i in range(64)]
        self.assertEqual(fit_fixed_normal_thresholds(s),
                         fit_fixed_normal_thresholds(s))
        self.assertEqual(list(fit_fixed_normal_thresholds(s)),
                         ["0.90", "0.95"])

    def test_runner_requires_registry_and_quantile_as_pair(self):
        args = runner.parser().parse_args(["--fixed-quantile", "0.90"])
        args.json_dir = "unused"
        with self.assertRaises(ThresholdAuditError):
            runner.load_fixed_threshold(args, Plan())
        args = runner.parser().parse_args(["--threshold-registry", "unused"])
        args.json_dir = "unused"
        with self.assertRaises(ThresholdAuditError):
            runner.load_fixed_threshold(args, Plan())

    def test_runner_threshold_variant_is_explicit_and_stable(self):
        common = ["--json-dir", "G:/dataset/realiad/jsons/realiad_jsons"]
        old = runner.configuration(runner.parser().parse_args(common))
        a = runner.configuration(runner.parser().parse_args(
            common + ["--fixed-quantile", "0.90",
                      "--threshold-registry", "calib/threshold_registry.json"]))
        b = runner.configuration(runner.parser().parse_args(
            common + ["--fixed-quantile", "0.95",
                      "--threshold-registry", "calib/threshold_registry.json"]))
        self.assertNotEqual(a, b)
        self.assertEqual(a["threshold"],
            "training_normal_only_registered_fixed_quantile_no_sidecar")
        self.assertEqual(old["threshold"], "LOSO theta0 fixed, no Sidecar")
        self.assertEqual(a["stream_mode"], b["stream_mode"])

    def test_calibrator_preflight_parser_guards_fixed_support(self):
        args = calib.parser().parse_args(["--preflight-only"])
        self.assertEqual(args.support_seed, 0)
        self.assertFalse(args.verify_only)
        with self.assertRaises(SystemExit):
            calib.parser().parse_args(["--support-seed", "1"])


if __name__ == "__main__":
    unittest.main()
