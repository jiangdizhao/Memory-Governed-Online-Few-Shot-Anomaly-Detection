import unittest
from types import SimpleNamespace

import torch

from experiments.calibration.method_registry import METHOD_CARDS, build_controllers
from experiments.calibration.native import (NativeAdapterError,
    validate_mutable_source_key, validate_p0_config)
from experiments.calibration.profiles import (
    CalibrationIsolatedProfile, NativeScorerConfig,
)
from experiments.risk_safe.shadow_memory import GreedyBank


class RegistryGreedyTests(unittest.TestCase):
    def test_c64_source_is_rejected_by_memory_writer_guard(self):
        c64 = [f"c64-{i}" for i in range(64)]
        validate_mutable_source_key("stream-normal", c64)
        with self.assertRaisesRegex(NativeAdapterError, "Protected C64"):
            validate_mutable_source_key("c64-7", c64)
    def test_actual_p0_config_fields_must_match_frozen_profile(self):
        profile = NativeScorerConfig()
        names = ("augmentation_seed", "bank_seed", "fixed_capacity",
                 "fixed_per_image", "augmentations", "include_identity", "multiplier",
                 "feature_fuse", "topk_patches", "image_aggregation", "image_topk",
                 "image_quantile", "knn")
        config = SimpleNamespace(**{name: getattr(profile, name) for name in names})
        validate_p0_config(config, profile)
        config.knn = 1
        with self.assertRaisesRegex(NativeAdapterError, "differs"):
            validate_p0_config(config, profile)

    def test_execution_device_is_separate_from_scientific_profile_hash(self):
        cpu = CalibrationIsolatedProfile(execution_device="cpu")
        cuda = CalibrationIsolatedProfile(execution_device="cuda")
        self.assertEqual(cpu.scientific_sha256, cuda.scientific_sha256)
        self.assertNotEqual(cpu.sha256, cuda.sha256)
        p0_config = SimpleNamespace(device="cuda", **{
            name: getattr(cpu.scorer, name)
            for name in ("augmentation_seed", "bank_seed", "fixed_capacity",
                         "fixed_per_image", "augmentations", "include_identity",
                         "multiplier", "feature_fuse", "topk_patches",
                         "image_aggregation", "image_topk", "image_quantile", "knn")
        })
        validate_p0_config(p0_config, cpu.scorer)

    def test_unresolved_baselines_are_not_claimed_implemented(self):
        for key in ("B5a", "B5b", "B6"):
            self.assertEqual(METHOD_CARDS[key]["status"], "BLOCKED_FIDELITY")
            self.assertTrue(METHOD_CARDS[key]["blocker"])
        self.assertEqual(METHOD_CARDS["B7"]["status"], "IMPLEMENTED_PROVISIONAL")

    def test_every_method_card_has_auditable_fidelity_schema(self):
        required = {
            "paper_title", "authors", "venue", "year", "paper_url",
            "author_code_url", "pinned_source_sha", "method_label",
            "reference_alg_or_equation", "faithful_or_adapted",
            "exact_differences", "input_permissions",
            "initialization_and_warmup", "declared_parameters_and_grid",
            "random_seed_policy", "independent_unit_test_reference",
            "missing_and_insufficient_data_behavior", "numerical_edge_cases",
            "complexity_and_runtime", "claimed_guarantee",
            "invalidated_assumptions", "source_and_test_hash",
        }
        for name, card in METHOD_CARDS.items():
            self.assertTrue(required.issubset(card), name)
            self.assertTrue(card["faithful_or_adapted"], name)
            self.assertTrue(card["source_and_test_hash"]["source_hash_manifest"], name)

    def test_executable_controller_states_restore_with_matching_method_configs(self):
        scores = [i / 1000 for i in range(64)]
        original = build_controllers(.05, scores)
        restored = build_controllers(.05, scores)
        self.assertEqual([x.name for x in original], [x.name for x in restored])
        for source, target in zip(original, restored):
            state = source.state_dict()
            target.load_state_dict(state)
            self.assertEqual(target.state_dict(), state, source.name)

    def test_native_greedy_proposal_respects_capacity_and_version(self):
        initial = GreedyBank([torch.empty((0, 2), dtype=torch.float32)], [2])
        first, audit = initial.proposed([torch.tensor([[1., 0.], [.99, .1]])],
                                        already_normalized=True)
        self.assertEqual(len(first.layers[0]), 2)
        self.assertEqual(first.version, 1)
        self.assertEqual(audit.appended, 2)
        second, audit2 = first.proposed([torch.tensor([[-1., 0.]])],
                                        already_normalized=True)
        self.assertEqual(len(second.layers[0]), 2)
        self.assertLessEqual(len(second.layers[0]), 2)
        self.assertIn(audit2.replaced + audit2.skipped, (1,))


if __name__ == "__main__": unittest.main()
