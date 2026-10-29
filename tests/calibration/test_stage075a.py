from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from experiments.calibration.stage075a import (
    c64_transport_diagnostics,
    causal_prequential_step,
    confusion_metrics,
    higher_quantile,
    paired_score_metrics,
    prior_group_status,
    ranking_metrics,
    snapshot_cohort_mask,
)
from experiments.calibration.memory_state import CalibrationGreedyState
from experiments.risk_safe.shadow_memory import GreedyBank


class Stage075ACausalProtocolTests(unittest.TestCase):
    def test_smoke_t0_snapshot_initializes_only_scored_prefix(self):
        from experiments.calibration.stage075a_runner import initialize_static_snapshot

        snapshot = np.full(3815, np.nan, dtype=np.float64)
        static = np.linspace(0.0, 1.0, 512, dtype=np.float64)
        initialize_static_snapshot(snapshot, static, len(static))
        np.testing.assert_array_equal(snapshot[:512], static)
        self.assertTrue(np.isnan(snapshot[512:]).all())

    def test_native_legacy_bank_content_digest_uses_shared_prototype_hash(self):
        import torch
        from experiments.calibration.stage075a_runner import _prototype_content_digest

        bank = GreedyBank([torch.tensor([[1.0, 0.0], [0.0, 1.0]])], [4], version=3)
        state = CalibrationGreedyState(bank, memory_content_version=1)
        self.assertEqual(_prototype_content_digest(bank), state.content_sha256())

    def test_native_scorer_config_uses_verified_p0_config_class(self):
        from experiments.calibration.stage075a_runner import _native_config

        config = _native_config(None)
        self.assertEqual(type(config).__module__, "experiments.p0_protocol_audit")
        self.assertEqual(config.device, "cuda")

    def _replay(self, labels):
        memory = {"version": 0, "digest": "m0", "writes": []}
        requests = []
        call_log = []
        rows = []
        query_positions = {1, 3}

        def memory_state():
            return memory["version"], memory["digest"]

        for position in range(5):
            def scores(position=position):
                call_log.append((position, "score"))
                return position / 10.0, position / 10.0 + memory["version"]

            def static_map(position=position):
                call_log.append((position, "static_map"))
                return {"event": position}

            def reveal(position):
                requests.append(position)
                call_log.append((position, "reveal"))
                return labels[position]

            def write(*, queried, label, source_key, score, static_amap,
                      full_features, step):
                call_log.append((step, "writer"))
                if queried and label == 0:
                    memory["version"] += 1
                    memory["digest"] = f"m{memory['version']}"
                    memory["writes"].append(step)
                return {"queried": queried, "label_seen": label,
                        "memory_version": memory["version"]}

            result = causal_prequential_step(
                position=position, queried=position in query_positions,
                memory_state=memory_state, score_pair=scores,
                static_map=static_map, reveal_label=reveal, apply_writer=write,
                source_key=f"group-{position}", features=(position,))
            rows.append(result)
        return rows, requests, call_log, memory

    def test_unqueried_labels_cannot_change_writer_or_score_trace(self):
        first = self._replay([0, 0, 1, 1, 0])
        changed_unqueried = self._replay([1, 0, 0, 1, 1])
        self.assertEqual(first[1], [1, 3])
        self.assertEqual(changed_unqueried[1], [1, 3])
        self.assertEqual(first[3], changed_unqueried[3])
        self.assertEqual(
            [(r.static_score, r.dynamic_score, r.memory_version_before,
              r.memory_version_after) for r in first[0]],
            [(r.static_score, r.dynamic_score, r.memory_version_before,
              r.memory_version_after) for r in changed_unqueried[0]])
        self.assertEqual([r.position for r in first[0] if r.queried], [1, 3])

    def test_scores_and_native_map_precede_label_disclosure(self):
        _, _, log, _ = self._replay([0, 0, 0, 1, 0])
        for position in (1, 3):
            self.assertLess(log.index((position, "score")),
                            log.index((position, "static_map")))
            self.assertLess(log.index((position, "static_map")),
                            log.index((position, "reveal")))
            self.assertLess(log.index((position, "reveal")),
                            log.index((position, "writer")))
        for position in (0, 2, 4):
            self.assertNotIn((position, "reveal"), log)

    def test_snapshot_masks_use_only_position_and_prior_source_identity(self):
        groups = ["a", "a", "b", "c", "b", "d", "e", "a"]
        mask = snapshot_cohort_mask(
            groups, snapshot_position=2, stream_length=len(groups),
            prior_sources={"b"}, cohort="all_remaining")
        self.assertEqual(mask.tolist(),
                         [False, False, False, True, False, True, True, True])
        next_mask = snapshot_cohort_mask(
            groups, snapshot_position=2, stream_length=len(groups),
            prior_sources={"b"}, cohort="next256")
        self.assertEqual(mask.tolist(), next_mask.tolist())
        flags = prior_group_status("b", {"b"}, {"b"})
        self.assertTrue(flags["previously_admitted"])
        self.assertTrue(flags["currently_resident"])

    def test_single_class_ranking_and_balanced_accuracy_are_undefined(self):
        result = ranking_metrics([0, 0, 0], [0.1, 0.2, 0.3])
        self.assertIsNone(result["auroc"])
        self.assertEqual(result["auroc_status"], "UNDEFINED_SINGLE_CLASS")
        cm = confusion_metrics([0, 0], [0.1, 0.2], 0.15)
        self.assertIsNone(cm["balanced_accuracy"])
        self.assertIsNone(cm["fnr"])

    def test_read_only_threshold_diagnostics_do_not_change_ranking(self):
        labels = [0, 1, 0, 1, 0, 1]
        static = [0.0, 1.0, 0.2, 0.8, 0.4, 0.6]
        dynamic = [0.2, 1.2, 0.4, 1.0, 0.6, 0.8]
        before = ranking_metrics(labels, dynamic)
        paired_score_metrics(labels, static, dynamic,
                             group_mask=[True] * 6, theta0=.5,
                             current_theta=[.1, .2, .3, .4, .5, .6])
        after = ranking_metrics(labels, dynamic)
        self.assertEqual(before, after)

    def test_c64_pure_translation_and_heterogeneity_are_distinguished(self):
        before = np.linspace(0.0, 1.0, 64)
        after = before + 0.25
        theta = higher_quantile(before.tolist(), .90)
        translated = c64_transport_diagnostics(
            before, after, theta_previous=theta)
        self.assertAlmostEqual(translated["median_paired_shift"], .25)
        self.assertAlmostEqual(translated["shift_heterogeneity_H_IQR"], 0.0)
        self.assertAlmostEqual(translated["theta_transport_candidate"],
                               translated["theta_SCQ_candidate_higher_q90"])
        self.assertEqual(translated["rank_reordering"]["strict_discordant_pairs"], 0)

        heterogeneous = after.copy()
        heterogeneous[::2] += np.linspace(-0.2, 0.2, 32)
        changed = c64_transport_diagnostics(
            before, heterogeneous, theta_previous=theta)
        self.assertGreater(changed["shift_heterogeneity_H_IQR"], 0.0)
        self.assertNotEqual(changed["theta_transport_candidate"],
                            changed["theta_SCQ_candidate_higher_q90"])

    def test_higher_q90_uses_locked_order_statistic(self):
        values = list(range(64))
        self.assertEqual(higher_quantile(values, .90), values[57])

    def test_checkpoint_round_trip_preserves_scores_memory_and_provenance(self):
        import torch
        from experiments.calibration.stage075a_runner import (
            _memory_state_to_cpu, _memory_state_to_device, _save_checkpoint)

        state = CalibrationGreedyState(
            GreedyBank([torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)],
                       [4], version=2), memory_content_version=1)

        class Writer:
            def state_dict(self):
                return {"memory_sha256": state.sha256(),
                        "content_sha256": state.content_sha256(),
                        "memory_content_version": state.version}

        arrays = {0: np.asarray([0.1, 0.2, 0.3], dtype=np.float64)}
        masks = {0: np.asarray([True, False, True])}
        text_rows = [{"position": 0, "memory_sha256": state.sha256()}]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "resume.pt"
            _save_checkpoint(
                checkpoint, run_identity="frozen-run-id", next_position=1,
                writers={0: Writer()}, c64_current={0: np.arange(64, dtype=float)},
                c64_state_digest={0: state.sha256()}, ever_admitted={0: {"g0"}},
                score_dynamic=arrays, current_q90={0: np.asarray([.5, .5, .5])},
                ever_group_mask=masks, resident_group_mask=masks,
                memory_version_before={0: np.asarray([0, 1, 1])},
                memory_sha_before={0: [state.sha256()] * 3},
                snapshot_scores={0: {0: arrays[0].copy()}},
                snapshot_state={0: {0: state}},
                snapshot_sources={0: {0: [["g0", "g1"]]}},
                snapshot_prior_groups={0: {0: {"g0"}}},
                snapshot_c64_q90={0: {0: .5}}, c64_transport={0: []},
                c64_pairs_before={0: []}, c64_pairs_after={0: []},
                event_rows=text_rows, query_tape_sha256="tape-digest",
                elapsed_seconds=12.5, torch=torch)
            restored = torch.load(checkpoint, map_location="cpu")
            self.assertEqual(restored["run_identity"], "frozen-run-id")
            self.assertEqual(restored["query_tape_sha256"], "tape-digest")
            self.assertEqual(restored["next_position"], 1)
            np.testing.assert_array_equal(restored["score_dynamic"]["0"], arrays[0])
            self.assertEqual(restored["writers"]["0"]["memory_sha256"], state.sha256())
            moved = _memory_state_to_device(restored["snapshot_state"]["0"]["0"], "cpu")
            self.assertEqual(moved.sha256(), state.sha256())
            self.assertEqual(_memory_state_to_cpu(moved).content_sha256(),
                             state.content_sha256())


if __name__ == "__main__":
    unittest.main()
