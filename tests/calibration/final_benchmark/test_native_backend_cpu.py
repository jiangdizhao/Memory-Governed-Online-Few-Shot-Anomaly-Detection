"""CPU checks of the new backend and its historical numerical primitives."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.calibration import a_calibration_continuation as a3
from experiments.calibration import r1_matched_correction_pilot as r1
from experiments.calibration import r7_shuffled_correction_4class as r7
from experiments.calibration import r10_signed_knn_baseline as r10
from experiments.calibration.controllers import StateConsistentQuantile
from experiments.calibration.e6_bea_closed_loop import NormalCoverageMemory
from experiments.calibration.final_benchmark.native_backend import (
    NativeBackendError, NativeFourArmBackend, NativeInputEncoder,
)
from experiments.calibration.final_benchmark import identity
from experiments.calibration.final_benchmark.runner import ScorePacket, TestEvent
from experiments.calibration.final_benchmark.query import BeaPrice
from experiments.calibration.final_benchmark.static_baselines import (
    ensure_historical_memory_import_path, static36_threshold_c32,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class NativeBackendCPUChecks(unittest.TestCase):
    def test_registered_event_checks_identity_and_bytes_before_shared_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            image_key = "OK/S0001/widget_0001_OK_C1_20260101000000.png"
            image_path = Path(directory) / image_key
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"synthetic-image-bytes")
            group, camera = identity._realiad_identity_key("widget", image_key)
            release = sha("release")
            event_id = identity._domain_hash("Gate0-event-v1", release, "widget", "0",
                                             group, str(camera), image_key, "1")
            record = {"position": 0, "event_id_sha256": event_id,
                      "image_identity_sha256": identity._image_identity_digest(image_key),
                      "image_content_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                      "source_group_sha256": identity._source_group_digest(group),
                      "camera_id": 1, "within_group_ordinal": 1}
            locator = {"official_image_key": image_key, "official_group_key": group,
                       "image_path": str(image_path)}
            encoder = NativeInputEncoder.__new__(NativeInputEncoder)
            calls = []
            encoder.encode_path = lambda path: calls.append(path) or SimpleNamespace(encoding_seconds=.01)
            event = encoder.encode_registered_realiad_event(
                record=record, locator=locator, category="widget",
                dataset_release_sha256=release, stream_seed=0)
            self.assertEqual(event.event_id_sha256, event_id)
            self.assertEqual(len(calls), 1)
            image_path.write_bytes(b"changed-synthetic-bytes")
            with self.assertRaisesRegex(NativeBackendError, "image bytes"):
                encoder.encode_registered_realiad_event(
                    record=record, locator=locator, category="widget",
                    dataset_release_sha256=release, stream_seed=0)
            self.assertEqual(len(calls), 1)

    def test_native_constructor_resolves_sources_before_rejecting_missing_roles(self):
        from experiments import p0_protocol_audit as p0
        from experiments.calibration import profiles
        from experiments.calibration import r3_fullstream_2x2 as r3
        impl = p0.require_sources()
        cfg, _ = r3._native_config(p0, profiles, r1)
        with self.assertRaisesRegex(NativeBackendError, "four Support and 64 C64"):
            NativeFourArmBackend(category="synthetic", condition_id="P1", impl=impl, p0=p0,
                backbone=object(), cfg=cfg, support_paths=[], support_group_sha256=[],
                support_image_sha256=[], support_cache={}, support_cls=[], c64_items=[],
                c64_group_sha256=sha("c64"))

    def test_native_fb_feedback_is_identical_with_real_shadow_enabled(self):
        import copy
        from dataclasses import asdict
        import torch
        from experiments import p0_protocol_audit as p0
        from experiments.calibration import r2_exp2a_c1_strength as exp2a
        from experiments.calibration import r9_correction_mechanism_falsification as r9
        from experiments.calibration.native import _fixed_bank_digest
        ensure_historical_memory_import_path()
        from v02_memory import MemoryArm
        impl = p0.require_sources()
        torch.set_num_threads(1)
        features = torch.zeros((784, 8), dtype=torch.float32)
        features[:, 1] = 1.0
        defect_features = torch.zeros((784, 8), dtype=torch.float32)
        defect_features[:, 2] = 1.0
        original = impl.PatchVMB(device="cpu", K_normal_fixed=16, K_normal_dyn=1024,
            K_normal_ltm=1, K_concept=8, corr_cfg=impl.CorrectionConfig(enabled=False, K_correction=1024),
            use_defect_bank=False)
        original.bank_dtype = torch.float32
        original._ensure_mode_view_normals("p0", 0, "id", [features])
        fixed = torch.zeros((1, 8), dtype=torch.float32)
        fixed[:, 0] = 1.0
        original.normal_fixed["p0"][0]["id"][0].set_prototypes(fixed)
        support = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
        settings = SimpleNamespace(flat=1024, stm=1024, ltm=0, buffer=0,
                                   min_events=2, promote_every=16, cap_bytes=2 * 1024**3)

        def run(with_shadow):
            bank = copy.deepcopy(original)
            backend = NativeFourArmBackend.__new__(NativeFourArmBackend)
            backend.impl, backend.r1, backend.exp2a, backend.r10 = impl, r1, exp2a, r10
            # Only C64 rescoring is a synthetic fixture; writer, C1, state
            # digests and B3 transitions use the unchanged historical code.
            backend.r7 = SimpleNamespace(_initialize_b3=r7._initialize_b3, _refresh_b3=r7._refresh_b3,
                _score_c64_base=lambda *args: ([.5] * 64, [None] * 64, 0.0))
            backend.r9 = SimpleNamespace(_state_digest_for_codebook=r9._state_digest_for_codebook,
                _score_c64_for_codebook=lambda *args: ([.5] * 64, 0.0))
            backend.category, backend.cfg = "synthetic", object()
            backend.c64_items, backend.c64_group_sha256 = tuple({} for _ in range(64)), sha("c64")
            state = SimpleNamespace(bank=bank, writer=MemoryArm("greedy_coreset", bank, settings, seed=0),
                correction_cfg=bank.corr_cfg, correction=bank.corr_cb["p0"]["id"][0], sources={},
                knn=None, fixed_sha=_fixed_bank_digest(bank), version=0, coverage=NormalCoverageMemory(support),
                c64=[.5] * 64, base_cache=[None] * 64, theta0=.5, sigma0=.02,
                controller=StateConsistentQuantile(.5, delta=.01, min_normals=16, name="F-B"),
                detector_sha="", state_sha="")
            backend._states, backend._contexts = {"F-B": state}, {}
            state.detector_sha = backend._detector_digest("F-B")
            r7._initialize_b3(state.controller, state.c64, backend.c64_group_sha256, state.detector_sha, .5)
            backend._refresh_state_identity("F-B")
            shadow_cfg = type(bank.corr_cfg)(**asdict(bank.corr_cfg))
            shadow_cfg.enabled = False
            shadow_cb = impl.SignedCorrectionCodebook(1024, 8, bank.device, shadow_cfg)
            backend._shadow = SimpleNamespace(cfg=shadow_cfg, cb=shadow_cb, sources={}, version=0, sha="")
            backend._refresh_shadow_identity()
            results = []
            for position, label, layer in ((0, 0, features), (5, 1, defect_features)):
                event = TestEvent(position, sha(f"actual-event-{position}"), sha(f"actual-image-{position}"),
                                  sha(f"actual-group-{position}"), 1)
                backend._contexts[("F-B", event.event_id_sha256)] = {
                    "features": [layer], "selected_np": layer[:32].numpy(),
                    "top_ids": np.arange(32, dtype=np.int64), "c0_score": .49,
                    "global_cls": np.array([1, 0, 0], dtype=np.float32), "state_sha": state.state_sha}
                packet = ScorePacket(.49, 0, .5, 1.0, state.version, state.state_sha,
                                     threshold=state.controller.theta)
                if with_shadow:
                    backend.commit_shadow_feedback(event, packet, label, {})
                backend.commit_feedback("F-B", event, packet, label, {})
                results.append((state.detector_sha, r1._memory_content_sha256(bank),
                                state.version, float(state.controller.theta),
                                tuple(state.correction.beta[:state.correction.size()].tolist())))
            return results, shadow_cb.beta[:shadow_cb.size()].tolist()

        actual, shadow_beta = run(True)
        control, _ = run(False)
        self.assertEqual(actual, control)
        self.assertTrue(shadow_beta and all(value < 0 for value in shadow_beta))

    def test_backend_scores_with_native_cpu_bank_and_r10(self):
        import copy
        import torch
        from PIL import Image
        from experiments import p0_protocol_audit as p0
        from experiments.calibration import r2_exp2a_c1_strength as exp2a
        impl = p0.require_sources()
        torch.set_num_threads(1)
        cfg = impl.CorrectionConfig(enabled=False, K_correction=1024)
        bank = impl.PatchVMB(device="cpu", K_normal_fixed=16, K_normal_dyn=1024,
                             K_normal_ltm=1, K_concept=8, corr_cfg=cfg,
                             use_defect_bank=False)
        bank.bank_dtype = torch.float32
        features = torch.zeros((784, 4096), dtype=torch.float32)
        features[:, 1] = 1.0
        bank._ensure_mode_view_normals("p0", 0, "id", [features])
        fixed = torch.zeros((1, 4096), dtype=torch.float32)
        fixed[:, 0] = 1.0
        bank.normal_fixed["p0"][0]["id"][0].set_prototypes(fixed)
        bank.concept_cb["p0"][0].update(fixed)
        backend = NativeFourArmBackend.__new__(NativeFourArmBackend)
        backend.impl, backend.r1, backend.exp2a, backend.r10 = impl, r1, exp2a, r10
        backend.cfg = SimpleNamespace(topk_patches=32, image_aggregation="max", image_topk=64,
                                      image_quantile=.95)
        backend.protected_groups, backend._contexts, backend._context_group = frozenset(), {}, None
        backend.protected_images = frozenset()
        support = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
        knn = r10.SignedKNNFIFO(1024)
        knn.insert(features[:32].numpy(), sign=1, source_group_sha256=sha("prior"), position=0)
        backend._states = {}
        for arm, memory in (("F-B", None), ("K-B", knn)):
            own_bank = copy.deepcopy(bank)
            backend._states[arm] = SimpleNamespace(
                bank=own_bank, correction_cfg=own_bank.corr_cfg,
                correction=own_bank.corr_cb["p0"]["id"][0], sources={}, knn=memory,
                coverage=NormalCoverageMemory(support), c64=[.5] * 64,
                controller=SimpleNamespace(theta=.5), version=0, state_sha=sha(arm))
        encoded = SimpleNamespace(image=Image.new("RGB", (28, 28)), feature_layers=[features],
                                  global_cls=np.array([1, 0, 0], dtype=np.float32))
        event = TestEvent(5, sha("native-event"), sha("native-image"), sha("native-group"),
                          1, feature_ref=encoded)
        f = backend.score("F-B", event, backend.snapshot("F-B"))
        k = backend.score("K-B", event, backend.snapshot("K-B"))
        self.assertTrue(np.isfinite([f.score, k.score, f.bea_objective, k.bea_objective]).all())
        self.assertAlmostEqual(k.score - f.score, .02, places=5)
        self.assertNotEqual(f.bea_objective, k.bea_objective)

    def test_native_c0_writer_and_c1_signs_on_cpu_fixture(self):
        import copy
        import torch
        from experiments import p0_protocol_audit as p0
        from experiments.calibration import r2_exp2a_c1_strength as exp2a
        from experiments.calibration import r9_correction_mechanism_falsification as r9
        from experiments.calibration.native import _fixed_bank_digest
        from experiments.risk_safe import native_bridge
        ensure_historical_memory_import_path()
        from v02_memory import MemoryArm

        impl = p0.require_sources()
        torch.set_num_threads(1)
        cfg = impl.CorrectionConfig(enabled=False, K_correction=1024)
        bank = impl.PatchVMB(device="cpu", K_normal_fixed=16, K_normal_dyn=1024,
                             K_normal_ltm=1, K_concept=8, corr_cfg=cfg,
                             use_defect_bank=False)
        bank.bank_dtype = torch.float32
        features = torch.zeros((784, 8), dtype=torch.float32)
        features[:, 1] = 1.0
        bank._ensure_mode_view_normals("p0", 0, "id", [features])
        fixed = torch.zeros((1, 8), dtype=torch.float32)
        fixed[:, 0] = 1.0
        bank.normal_fixed["p0"][0]["id"][0].set_prototypes(fixed)
        parity_bank = copy.deepcopy(bank)
        settings = SimpleNamespace(flat=1024, stm=1024, ltm=0, buffer=0,
                                   min_events=2, promote_every=16, cap_bytes=2 * 1024**3)
        writer = MemoryArm("greedy_coreset", bank, settings, seed=0)
        parity_writer = MemoryArm("greedy_coreset", parity_bank, settings, seed=0)
        ids = np.arange(32, dtype=np.int64)
        parity = native_bridge.verify_native_update_parity(
            impl, parity_writer, [features],
            {"per_layer_summary": {"layer0": {"topk_patch_idx": ids.tolist()}}},
            score=.49, theta=.5, sigma=.02, step=0)
        self.assertTrue(parity["passed"])

        def update(**kwargs):
            writer.update(impl, list(kwargs["feature_layers"]), {0: kwargs["candidate_ids"].tolist()},
                          kwargs["position"], score=kwargs["score"], theta=kwargs["theta"],
                          sigma=kwargs["sigma"])
            return {}

        receipt = r1.common_c0_greedy_writer_action(
            position=0, c0_score=.49, theta=.5, sigma=.02, c0_prediction=0,
            candidate_ids=ids, candidate_features=features[:32].numpy(), feature_layers=[features],
            queried=True, disclosed_label=0, eligibility=impl.select_informative_normal_write,
            update=update)
        self.assertEqual(receipt["input_score_source"], "common_C0")
        self.assertEqual(native_bridge.snapshot(bank).sha256(), native_bridge.snapshot(parity_bank).sha256())
        before = native_bridge.snapshot(bank).sha256()
        receipt = r1.common_c0_greedy_writer_action(
            position=1, c0_score=.49, theta=.5, sigma=.02, c0_prediction=0,
            candidate_ids=ids, candidate_features=features[:32].numpy(), feature_layers=[features],
            queried=True, disclosed_label=1, eligibility=impl.select_informative_normal_write,
            update=update)
        self.assertEqual(receipt["action"], "verified_defect_no_normal_write")
        self.assertEqual(native_bridge.snapshot(bank).sha256(), before)

        c1_bank = copy.deepcopy(parity_bank)
        # C1 uses a fresh normal bank so the selected synthetic vectors remain novel.
        c1_bank.normal_dyn["p0"][0]["id"][0].clear()
        correction = c1_bank.corr_cb["p0"]["id"][0]
        sources = {}
        fixed_sha = _fixed_bank_digest(c1_bank)
        old_digest = r9._state_digest_for_codebook(c1_bank, fixed_sha, sources,
                                                    c1_bank.corr_cfg, correction, exp2a, r1)
        controller = StateConsistentQuantile(.5, delta=.01, min_normals=16, name="native-B3")
        r7._initialize_b3(controller, [.5] * 64, sha("c64"), old_digest, .5)
        exp2a._capture_update(impl, c1_bank, r1, "F-B", c1_bank.corr_cfg, correction,
            [features], ids, -1, .49, .5, "TN", 0, sources, sha("normal-group"))
        self.assertGreater(correction.size(), 0)
        self.assertTrue(torch.all(correction.beta[:correction.size()] < 0).item())
        same, _ = exp2a._score_arm(impl, c1_bank, "F-B", c1_bank.corr_cfg, correction,
                                   sources, features[:32], sha("normal-group"), 1)
        other, _ = exp2a._score_arm(impl, c1_bank, "F-B", c1_bank.corr_cfg, correction,
                                    sources, features[:32], sha("other-group"), 1)
        self.assertTrue(np.all(same == 0))
        self.assertTrue(np.all(other < 0))
        c64_feature = features.numpy()
        c64_items = [{"features": [c64_feature], "source_group_sha256": sha(f"c64-{i}")}
                     for i in range(64)]
        base_map = np.full(784, .3, dtype=np.float32)
        base_map[:32] = .5
        scores, _ = r9._score_c64_for_codebook(
            impl, c1_bank, r7, r1, exp2a, c64_items, [(base_map, ids, .5)] * 64,
            object(), c1_bank.corr_cfg, correction, sources, "synthetic", 1)
        self.assertEqual(len(scores), 64)
        self.assertTrue(all(value < .5 for value in scores))
        new_digest = r9._state_digest_for_codebook(c1_bank, fixed_sha, sources,
                                                    c1_bank.corr_cfg, correction, exp2a, r1)
        transition = r7._refresh_b3(controller, scores, sha("c64"), old_digest, new_digest, 0)
        self.assertEqual(transition["version_after"], 1)
        self.assertAlmostEqual(controller.theta, .49)
        defect_features = torch.zeros((784, 8), dtype=torch.float32)
        defect_features[:, 2] = 1.0
        exp2a._capture_update(impl, c1_bank, r1, "F-B", c1_bank.corr_cfg, correction,
            [defect_features], ids, 1, .49, .5, "FN", 1, sources, sha("defect-group"))
        self.assertTrue(torch.any(correction.beta[:correction.size()] > 0).item())

    def test_backend_shadow_does_not_change_fb_trajectory(self):
        import torch

        # Actual backend, query price, R1 writer gate and R7 B3; native bank/C1
        # numerical calls are CPU stand-ins while the v22 import is unavailable.
        class Codebook:
            def __init__(self):
                self.signs = []

            def size(self):
                return len(self.signs)

        class Writer:
            def __init__(self, bank):
                self.bank = bank

            def update(self, impl, layers, ids, position, **kwargs):
                self.bank.normal_writes += 1

        def run(with_shadow):
            patch_map = np.full(784, .3, dtype=np.float32)
            patch_map[:32] = .49
            ids = np.arange(32, dtype=np.int64)
            backend = NativeFourArmBackend.__new__(NativeFourArmBackend)
            backend.protected_groups, backend._contexts, backend._context_group = frozenset(), {}, None
            backend.protected_images = frozenset()
            backend.category, backend.cfg = "synthetic", object()
            backend.c64_items, backend.c64_group_sha256 = tuple({} for _ in range(64)), sha("c64")
            backend.impl = SimpleNamespace(
                l2_normalize=lambda value: torch.nn.functional.normalize(value, dim=-1),
                select_informative_normal_write=lambda **kwargs: (True, 32, "synthetic", None))
            backend.r1 = SimpleNamespace(
                _native_dense_score=lambda *args: (patch_map, ids, .49,
                    {"layer0": {"topk_dN_max": .5, "topk_dN_mean": .4, "topk_patch_idx": ids}}),
                corrected_dense_map=r1.corrected_dense_map, max_score=r1.max_score,
                common_c0_greedy_writer_action=r1.common_c0_greedy_writer_action,
                _memory_content_sha256=lambda bank: sha(f"normal:{bank.normal_writes}"))
            def c1_score(impl, bank, arm, cfg, codebook, *args):
                return np.full(32, sum(codebook.signs) * .01, dtype=np.float32), {}
            def c1_update(impl, bank, r1_module, arm, cfg, codebook, features, top_ids,
                          sign, score, theta, outcome, position, sources, group):
                codebook.signs.append(sign)
                return {}, {}, {}
            backend.exp2a = SimpleNamespace(_score_arm=c1_score, _capture_update=c1_update)
            backend.r9 = SimpleNamespace(
                _state_digest_for_codebook=lambda bank, fixed, sources, cfg, cb, exp2a, r1_module:
                    sha(f"{bank.normal_writes}:{cb.signs}"),
                _score_c64_for_codebook=lambda *args: ([.5] * 64, 0.0))
            backend.r7 = SimpleNamespace(_initialize_b3=r7._initialize_b3, _refresh_b3=r7._refresh_b3,
                _score_c64_base=lambda *args: ([.5] * 64, [None] * 64, 0.0))
            backend.r10 = r10
            bank = SimpleNamespace(device=torch.device("cpu"), normal_writes=0)
            support = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
            state = SimpleNamespace(bank=bank, writer=Writer(bank), correction_cfg=object(),
                correction=Codebook(), sources={}, knn=None, fixed_sha=sha("fixed"), version=0,
                coverage=NormalCoverageMemory(support), c64=[.5] * 64, base_cache=[None] * 64,
                theta0=.5, sigma0=.02,
                controller=StateConsistentQuantile(.5, delta=.01, min_normals=16, name="F-B"),
                detector_sha="", state_sha="")
            backend._states = {"F-B": state}
            state.detector_sha = backend._detector_digest("F-B")
            r7._initialize_b3(state.controller, state.c64, backend.c64_group_sha256,
                              state.detector_sha, .5)
            backend._refresh_state_identity("F-B")
            backend._shadow = SimpleNamespace(cfg=object(), cb=Codebook(), sources={}, version=0, sha="")
            backend._refresh_shadow_identity()
            features = torch.zeros((784, 4096), dtype=torch.float32)
            features[:, 0] = 1.0
            encoded = SimpleNamespace(image=object(), feature_layers=[features],
                                      global_cls=np.array([1, 0, 0], dtype=np.float32))
            policy = BeaPrice("F-B", .03)
            observed = []
            for position, label in ((0, 0), (5, 1)):
                event = TestEvent(position, sha(f"fb-event-{position}"), sha(f"fb-image-{position}"),
                                  sha(f"fb-group-{position}"), 1, feature_ref=encoded)
                snapshot = backend.snapshot("F-B")
                packet = backend.score("F-B", event, snapshot)
                proposed = policy.propose(packet.bea_objective, group_price=policy.price).proposed
                self.assertTrue(proposed)
                policy.observe_final_actions([True, False, False, False, False])
                if with_shadow:
                    shadow_snapshot = backend.shadow_snapshot(snapshot)
                    backend.score_shadow(event, snapshot, shadow_snapshot)
                    backend.commit_shadow_feedback(event, packet, label, {})
                    self.assertEqual(backend.state_digest("F-B"), snapshot)
                backend.commit_feedback("F-B", event, packet, label, {})
                observed.append((packet.score, packet.prediction, proposed,
                                 state.bank.normal_writes, tuple(state.correction.signs),
                                 state.controller.theta, backend.state_digest("F-B")))
            return observed, tuple(backend._shadow.cb.signs)

        with_shadow, shadow_signs = run(True)
        without_shadow, _ = run(False)
        self.assertEqual(with_shadow, without_shadow)
        self.assertEqual(shadow_signs, (-1,))

    def test_backend_bea_uses_each_arms_corrected_score(self):
        import torch
        from experiments.calibration.e6_bea_policy import compute_bea_components
        from experiments.calibration.exp_c_memory_evidence import d_band_utility

        # Synthetic native maps isolate backend wiring; R10 and E6 arithmetic
        # are the actual historical implementations, not a GPU parity claim.
        patch_map = np.full(784, 0.3, dtype=np.float32)
        patch_map[:32] = 0.4
        ids = np.arange(32, dtype=np.int64)
        backend = NativeFourArmBackend.__new__(NativeFourArmBackend)
        backend.protected_groups = frozenset()
        backend.protected_images = frozenset()
        backend._contexts, backend._context_group = {}, None
        backend.cfg = object()
        backend.r1 = SimpleNamespace(
            _native_dense_score=lambda *args: (patch_map, ids, .4,
                {"layer0": {"topk_dN_max": .5, "topk_dN_mean": .4, "topk_patch_idx": ids}}),
            corrected_dense_map=r1.corrected_dense_map, max_score=r1.max_score)
        backend.impl = SimpleNamespace(l2_normalize=lambda value: torch.nn.functional.normalize(value, dim=-1))
        backend.exp2a = SimpleNamespace(_score_arm=lambda *args: (np.full(32, .05, np.float32), {}))
        backend.r10 = r10
        support = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
        knn = r10.SignedKNNFIFO(1024)
        selected = np.zeros((32, 4096), dtype=np.float32)
        selected[:, 0] = 1.0
        knn.insert(selected, sign=-1, source_group_sha256=sha("prior"), position=0)
        backend._states = {}
        for arm, memory in (("F-B", None), ("K-B", knn)):
            backend._states[arm] = SimpleNamespace(
                bank=SimpleNamespace(device=torch.device("cpu")), correction_cfg=object(),
                correction=SimpleNamespace(size=lambda: 1), sources={}, knn=memory,
                coverage=NormalCoverageMemory(support), c64=[.5] * 64,
                controller=SimpleNamespace(theta=.42), version=0, state_sha=sha(arm))
        features = torch.zeros((784, 4096), dtype=torch.float32)
        features[:, 0] = 1.0
        encoded = SimpleNamespace(image=object(), feature_layers=[features],
                                  global_cls=np.array([1, 0, 0], dtype=np.float32))
        event = TestEvent(5, sha("event-5"), sha("image-5"), sha("current"), 1, feature_ref=encoded)
        f = backend.score("F-B", event, backend.snapshot("F-B"))
        k = backend.score("K-B", event, backend.snapshot("K-B"))
        self.assertAlmostEqual(f.score, .45, places=6)
        self.assertAlmostEqual(k.score, .38, places=6)
        self.assertEqual((f.prediction, k.prediction), (1, 0))
        novelty, _ = backend._states["K-B"].coverage.novelty_percentile(encoded.global_cls)
        components = compute_bea_components(
            score=k.score, threshold=.42, c64_scores=[.5] * 64, novelty_percentile=novelty,
            correction_top32=np.full(32, -.02, dtype=np.float32), correction_bank_size=32,
            topk_dN_max=.5, topk_dN_mean=.4)
        self.assertAlmostEqual(k.bea_objective, components.B + components.N + d_band_utility(components))
        self.assertNotEqual(f.bea_objective, k.bea_objective)

    def test_real_r10_and_b3_updates_remain_arm_local(self):
        # The actual R1 writer gate, R10 FIFO, R7 B3 and E6 coverage are used.
        # Bank mutation and C64 rescoring are small CPU fixtures, not GPU parity.
        class Writer:
            def __init__(self, bank):
                self.bank = bank

            def update(self, impl, layers, ids, position, **kwargs):
                self.bank.normal_writes += 1

        native_r1 = SimpleNamespace(
            common_c0_greedy_writer_action=r1.common_c0_greedy_writer_action,
            _memory_content_sha256=lambda bank: sha(f"normal:{bank.normal_writes}"))
        backend = NativeFourArmBackend.__new__(NativeFourArmBackend)
        backend.r1 = native_r1
        backend.r7 = SimpleNamespace(
            _initialize_b3=r7._initialize_b3, _refresh_b3=r7._refresh_b3,
            _score_c64_base=lambda *args: ([0.8] * 64, [None] * 64, 0.0))
        backend.r10 = SimpleNamespace(_score_c2_c64=lambda *args, **kwargs: ([0.8] * 64, 0.0, 0))
        backend.impl = SimpleNamespace(select_informative_normal_write=lambda **kwargs: (True, 32, "synthetic", None))
        backend.c64_items = tuple({"source_group_sha256": sha(f"c64-{i}")} for i in range(64))
        backend.c64_group_sha256 = sha("c64-groups")
        backend.cfg, backend.category = object(), "synthetic"
        backend._contexts = {}
        backend._states = {}
        support = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
        for arm in ("K-B", "K-M"):
            bank = SimpleNamespace(normal_writes=0)
            state = SimpleNamespace(bank=bank, writer=Writer(bank), knn=r10.SignedKNNFIFO(1024),
                fixed_sha=sha("fixed"), version=0, coverage=NormalCoverageMemory(support),
                c64=[0.5] * 64, base_cache=[None] * 64, theta0=0.5, sigma0=0.02,
                controller=StateConsistentQuantile(0.5, delta=.01, min_normals=16, name=arm),
                detector_sha="", state_sha="")
            backend._states[arm] = state
            state.detector_sha = backend._detector_digest(arm)
            r7._initialize_b3(state.controller, state.c64, backend.c64_group_sha256,
                              state.detector_sha, state.theta0)
            backend._refresh_state_identity(arm)
        unchanged = backend.state_digest("K-M")
        state = backend._states["K-B"]
        selected = np.zeros((32, 4096), dtype=np.float32)
        selected[:, 0] = 1.0
        for position, label in ((0, 1), (5, 0)):
            group = sha(f"group-{position}")
            event = TestEvent(position, sha(f"event-{position}"), sha(f"image-{position}"), group, 1)
            backend._contexts[("K-B", event.event_id_sha256)] = {
                "features": [object()], "selected_np": selected, "top_ids": np.arange(32, dtype=np.int64),
                "c0_score": 0.2, "global_cls": np.array([1, 0, 0], dtype=np.float32),
                "state_sha": state.state_sha}
            packet = ScorePacket(0.2, 0, 0.5, 0.0, state.version, state.state_sha, threshold=state.controller.theta)
            result = backend.commit_feedback("K-B", event, packet, label, {})
            self.assertEqual(result["writer_action"],
                "verified_defect_no_normal_write" if label else "native_greedy_normal_update")
            self.assertEqual(state.bank.normal_writes, 0 if label else 1)
            self.assertEqual(state.controller.evidence_version, 1 if label else 2)
            self.assertEqual(backend.state_digest("K-M"), unchanged)
            if label:
                delta, info = state.knn.score(selected, event_position=1, event_group_sha256=group)
                self.assertTrue(np.all(delta == 0))
                self.assertEqual(info["source_excluded_anchor_count"], 32)
        self.assertEqual(state.coverage.dynamic_size, 1)
        self.assertAlmostEqual(state.controller.theta, 0.52)

    def test_a3b_source_candidate_identity_and_quota_rules(self):
        groups = [sha(f"group-{i}") for i in range(36)]
        available = {group: 4 * 784 for group in groups}
        namespace = "a3b-fexp-kc-v1\0synthetic"
        quotas = a3.a3b_candidate_quotas(groups, 65536, available, namespace)
        self.assertEqual(sum(quotas.values()), 65536)
        self.assertLessEqual(max(quotas.values()) - min(quotas.values()), 1)
        first = a3.a3b_candidate_identity("a3b-fexp-kc-v1", groups[0], "OK/S1/image.png", 0, 0)
        self.assertEqual(first, a3.a3b_candidate_identity("a3b-fexp-kc-v1", groups[0], "OK/S1/image.png", 0, 0))
        self.assertNotEqual(first, a3.a3b_candidate_identity("a3b-fexp-kc-v1", groups[0], "OK/S1/image.png", 0, 1))
        self.assertEqual(static36_threshold_c32(range(32)), 28.0)


if __name__ == "__main__":
    unittest.main()
