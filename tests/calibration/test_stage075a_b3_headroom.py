from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from experiments.calibration.controllers import StateConsistentQuantile
from experiments.calibration.stage075a import DEVELOPMENT_CLASSES
from experiments.calibration.stage075a_b3_headroom import (
    DEFAULT_JSON_ROOT, DEFAULT_SOURCE, DEFAULT_STAGE05, CategoryInput,
    LOCAL_OFFSETS, SUPPORT_SEEDS, _band_rows, _confusion, _q90,
    _reference_recurrence, _replay_one, _snapshot_cohort, evaluate_all,
    load_verified_inputs, load_verified_labels, replay_all, sha256_json,
    query_feasibility, _read_json,
)


class ActualB3ReplayUnitTests(unittest.TestCase):
    @staticmethod
    def synthetic_trace():
        before = np.arange(64, dtype=np.float64) / 100.0
        after = before + 0.03
        q0, q1 = _q90(before), _q90(after)
        memory_hashes = ("memory-v0", "memory-v1", "memory-v2")
        versions_before, versions_after = (0, 1, 2), (1, 2, 2)
        c64_before = (before, after)
        c64_after = (after, after)
        events = []
        for pos in range(3):
            changed = pos < 2
            version_before, version_after = versions_before[pos], versions_after[pos]
            digest_before = memory_hashes[version_before]
            digest_after = memory_hashes[version_after]
            events.append({
                "position": pos, "camera_id": 1,
                "official_group_sha256": f"group-{pos}",
                "query_tape_selected": False, "queried_label_disclosed": False,
                "support_seeds": {"0": {
                    "dynamic_score": 0.575 + pos * 0.01,
                    "memory_version_before": version_before,
                    "memory_version_after": version_after,
                    "memory_sha256_before": digest_before,
                    "memory_sha256_after": digest_after,
                    "current_C64_q90_pre_event": q0 if pos == 0 else q1,
                    "writer": {"changed": changed,
                               "memory_content_version_after": version_after,
                               "appended": int(changed), "replaced": 0},
                }},
            })
        transport = []
        for pos in range(2):
            transport.append({
                "position": pos,
                "memory_version_before": pos,
                "memory_version_after": pos + 1,
                "memory_before_sha256": memory_hashes[pos],
                "memory_after_sha256": memory_hashes[pos + 1],
                "c64_before_sha256": sha256_json(c64_before[pos].tolist()),
                "c64_after_sha256": sha256_json(c64_after[pos].tolist()),
            })
        arrays = {
            "dynamic_s0": np.asarray([row["support_seeds"]["0"]["dynamic_score"]
                                      for row in events], dtype=np.float64),
            "current_q90_s0": np.asarray([q0, q1, q1], dtype=np.float64),
            "memory_version_before_s0": np.asarray([0, 1, 2], dtype=np.int64),
            "memory_sha_before_s0": np.asarray(memory_hashes[:3], dtype="U16"),
            "c64_pairs_before_s0": np.stack(c64_before),
            "c64_pairs_after_s0": np.stack(c64_after),
        }
        manifest = {"memory_by_seed": {"0": {"theta0": q0}}}
        return before, after, q0, q1, events, transport, arrays, manifest

    def test_actual_controller_is_used_and_prediction_precedes_update(self):
        before, after, q0, q1, events, transport, arrays, manifest = self.synthetic_trace()
        result = _replay_one("audiojack", 0, events, arrays, transport, manifest)
        self.assertIs(type(StateConsistentQuantile(q0)), StateConsistentQuantile)
        self.assertEqual(result["theta_pre"], [q0, q0 + 0.01, q0 + 0.02])
        self.assertEqual(result["q90_pre"], [q0, q1, q1])
        self.assertEqual(result["memory_change_count"], 2)
        self.assertEqual(result["updates"][1]["q90_changed"], False)
        self.assertAlmostEqual(result["updates"][1]["actual_movement"], 0.01)
        self.assertLessEqual(max(abs(row["actual_movement"])
                                 for row in result["updates"]), 0.01 + 1e-12)
        self.assertNotIn("label", result)
        self.assertNotIn("queried_label", result)

    def test_independent_recurrence_matches_real_b3_replay(self):
        _, _, q0, q1, events, transport, arrays, manifest = self.synthetic_trace()
        replay = _replay_one("audiojack", 0, events, arrays, transport, manifest)
        reference = _reference_recurrence(q0, [q0, q1, q1], [q1, q1, q1],
                                          [True, True, False])
        np.testing.assert_allclose(replay["theta_pre"], reference, rtol=0, atol=1e-15)

    def test_no_content_change_does_not_trigger_threshold_update(self):
        _, _, q0, q1, events, transport, arrays, manifest = self.synthetic_trace()
        result = _replay_one("audiojack", 0, events, arrays, transport, manifest)
        self.assertEqual(result["memory_change_count"], len(transport))
        self.assertEqual(result["updates"][-1]["position"], 1)
        self.assertEqual(result["threshold_update_count"], 2)


class LocalHeadroomIdentityTests(unittest.TestCase):
    def test_missing_class_has_undefined_ba(self):
        metrics = _confusion([0, 0], [0.1, 0.9], 0.5)
        self.assertIsNone(metrics["balanced_accuracy"])
        self.assertIsNone(metrics["fnr"])

    def test_oracle_includes_actual_threshold_and_headroom_nonnegative(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
        scores = np.asarray([0.1, 0.4, 0.6, 0.3, 0.5, 0.9])
        row = _snapshot_cohort(labels, scores, [f"g{i}" for i in range(6)],
                               theta_b3=0.5, theta_q90=0.45)
        self.assertTrue(row["retrospective_global_oracle"]["anchor_threshold_in_search"])
        self.assertGreaterEqual(row["h_total_oracle_minus_b3"], -1e-12)
        self.assertEqual(len(row["local_grid"]), len(LOCAL_OFFSETS))
        self.assertEqual(row["within_one_step"]["best_action"], "KEEP")

    def test_lower_and_raise_band_identities(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1, 0, 1], dtype=np.int8)
        scores = np.asarray([0.40, 0.49, 0.51, 0.48, 0.50, 0.52, 0.60, 0.44])
        rows = _band_rows(labels, scores, 0.50, [f"g{i // 2}" for i in range(8)],
                          [-0.01, 0.01])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["band_identity_pass"] for row in rows))
        for row in rows:
            self.assertAlmostEqual(row["delta_balanced_accuracy"],
                                   row["band_identity_delta_balanced_accuracy"], places=12)

    def test_query_counts_use_only_positions_strictly_before_snapshot(self):
        categories = {}
        labels = {}
        for category in DEVELOPMENT_CLASSES:
            events = []
            y = np.zeros(401, dtype=np.int8)
            y[1] = 1
            y[400] = 0
            for pos in range(401):
                selected = pos in (1, 400)
                events.append({"query_tape_selected": selected,
                               "queried_label_disclosed": selected,
                               "official_group_sha256": f"{category}-g{pos}"})
            categories[category] = CategoryInput(category, Path("."), {}, tuple(events), {}, {})
            labels[category] = y
        rows = query_feasibility(categories, labels)
        audio = [row for row in rows if row["category"] == "audiojack"]
        at400 = next(row for row in audio if row["snapshot_position"] == 400)
        at800 = next(row for row in audio if row["snapshot_position"] == 800)
        self.assertEqual(at400["queries_strictly_before_t"], 1)
        self.assertEqual(at400["distinct_queried_defect_groups"], 1)
        self.assertEqual(at800["queries_strictly_before_t"], 2)
        self.assertFalse(at800["online_local_gain_identifiable"])
        self.assertEqual(at800["status"], "ONLINE_LOCAL_GAIN_NOT_IDENTIFIABLE_FROM_EXISTING_ARTIFACTS")


class Stage075AActualB3ArtifactIntegrationTests(unittest.TestCase):
    def test_20_trajectory_replay_and_full_read_only_analysis(self):
        provenance, categories = load_verified_inputs(DEFAULT_SOURCE, DEFAULT_STAGE05)
        self.assertEqual(provenance["status"], "PASS")
        self.assertEqual(provenance["trajectory_count"], 20)
        replay = replay_all(categories)
        self.assertEqual(replay["status"], "PASS")
        self.assertFalse(replay["labels_supplied_to_replay"])
        self.assertEqual(sum(len(rows) for rows in replay["trajectories"].values()), 20)

        protocol = _read_json(DEFAULT_SOURCE / "protocol_manifest.json")
        labels, _ = load_verified_labels(DEFAULT_JSON_ROOT, DEFAULT_SOURCE, protocol)
        original_evaluation = _read_json(DEFAULT_SOURCE / "evaluation_metrics.json")
        analysis = evaluate_all(categories, replay, labels, original_evaluation)
        self.assertEqual(analysis["historical_metric_reproduction"]["status"], "PASS")
        self.assertEqual(analysis["snapshot_cohort_count"], 240)
        for trajectory in analysis["per_trajectory_metrics"]:
            for scope in trajectory["scopes"].values():
                methods = scope["methods"]
                shared = methods["Greedy_Frozen_theta0"]["ranking"]
                self.assertEqual(methods["Greedy_ReadOnly_current_C64_q90"]["ranking"], shared)
                self.assertEqual(methods["Greedy_Actual_B3"]["ranking"], shared)
        self.assertEqual(analysis["online_identifiability_status"],
                         "ONLINE_LOCAL_GAIN_NOT_IDENTIFIABLE_FROM_EXISTING_ARTIFACTS")
        replay_repeat = replay_all(categories)
        analysis_repeat = evaluate_all(categories, replay_repeat, labels, original_evaluation)
        self.assertEqual(sha256_json(replay), sha256_json(replay_repeat))
        self.assertEqual(sha256_json(analysis), sha256_json(analysis_repeat))


if __name__ == "__main__":
    unittest.main()
