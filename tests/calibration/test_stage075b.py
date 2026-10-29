from __future__ import annotations

import math
import unittest
from dataclasses import fields

from experiments.calibration.controllers import StateConsistentQuantile
from experiments.calibration.protocol import (
    EvidenceAnchor, EvidenceSnapshot, Feedback, StreamEvent, TapeRecord,
)
from experiments.calibration.scdac_g import SCDACGController, SCDACGConfig
from experiments.calibration.stage075b_controllers import (
    Stage075BController, Stage075BOffset, _cdf, _map_percentile,
)
from experiments.calibration.stage075b_synthetic import (
    FOCUSED_SEED, SCENARIOS, build_engine, checkpoint_resume_check,
    focused_mechanism_results, run_scenario,
)


class Stage075BContractTests(unittest.TestCase):
    def test_s0_actual_b3_exact_fallback(self):
        scenario = next(s for s in SCENARIOS if s.key == "S0_exact_b3_fallback")
        summary = run_scenario(scenario, 37)
        self.assertIsNone(summary["metrics"]["B3"]["balanced_accuracy"])
        engine, *_ = build_engine(scenario, 37)
        result = engine.run()
        for event in result["trace"]:
            self.assertEqual(event["theta_pre"]["B3"], event["theta_pre"]["B8-Full"])
            self.assertEqual(event["prediction_pre"]["B3"],
                             event["prediction_pre"]["B8-Full"])
            self.assertEqual(event["theta_post"]["B3"], event["theta_post"]["B8-Full"])

    def test_b3_and_b7_are_the_existing_implementations(self):
        b3 = StateConsistentQuantile(0.5, delta=0.01, min_normals=16)
        b7 = SCDACGController(0.5)
        self.assertEqual(b3.delta, 0.01)
        self.assertEqual(b3.min_normals, 16)
        self.assertIsInstance(b7, SCDACGController)
        self.assertEqual(b7.config.min_normals, 16)
        self.assertEqual(b7.config.min_defects, 8)
        self.assertEqual(b7.config.confidence_delta, 0.05)

    def test_public_observation_contains_no_label_or_group_identity(self):
        public_fields = {field.name for field in fields(__import__(
            "experiments.calibration.protocol", fromlist=["Observation"]).Observation)}
        self.assertNotIn("label", public_fields)
        self.assertNotIn("official_sample_key", public_fields)
        self.assertNotIn("path", public_fields)

    def test_predict_before_disclosure_and_shared_memory(self):
        scenario = next(s for s in SCENARIOS if s.key == "S12_shared_score_invariance")
        engine, memory, _, _, _ = build_engine(scenario, 101)
        result = engine.run()
        for row in result["trace"]:
            before, after = row["memory_version_before"], row["memory_version_after"]
            if row["write"]["label_type"] == "verified_normal":
                self.assertLessEqual(after - before, 1)
            else:
                self.assertEqual(after, before)
            self.assertEqual(row["label_disclosed"], row["query_event"])
        self.assertEqual(memory.version, result["trace"][-1]["memory_version_after"])
        self.assertTrue(all(len(layer) == memory.version for layer in memory.layers))

    def test_s12_common_scores_and_ranking_metrics(self):
        scenario = next(s for s in SCENARIOS if s.key == "S12_shared_score_invariance")
        summary = run_scenario(scenario, 2029)
        self.assertTrue(summary["score_sha256"])
        self.assertEqual(summary["ranking_status"], "DEFINED")
        # There is one score vector and one offline ranking calculation for every
        # threshold-only arm; BA is method-specific, AUROC/AP are not.
        self.assertEqual(len({summary["score_sha256"] for _ in
                              ("B3", "B7", "B3+Offset", "B8-M", "B8-S", "B8-P", "B8-Full")}), 1)
        self.assertIsNotNone(summary["shared_score_auroc"])
        self.assertIsNotNone(summary["shared_score_auprc"])

    def test_focused_s1_s2_s6_s7_s8_s10_s11_s14_and_controls(self):
        results = focused_mechanism_results()
        expected = (
            "S1_common_monotone_rank_invariance", "S2_zero_median_tail_change",
            "S6_distinct_source_ladder", "S7_staleness_separation",
            "S8_c64_mismatch_boundary", "S10_numerical_edges", "S11_lso_fixture",
            "S14_missing_label_nonidentifiability_and_control_C",
            "control_A_no_gain_fixture", "control_B_same_information",
            "control_D_candidate_selection_optimism",
        )
        for key in expected:
            self.assertEqual(results[key]["status"], "PASS", key)
        control_e = results["control_E_mechanism_removal"]
        self.assertEqual(control_e["status"], "BLOCKED")
        self.assertIn("DeltaJ undefined", control_e["component_isolation"]["evidence_gate"])
        self.assertIn("no one-toggle baseline-regularization-removal arm",
                      control_e["component_isolation"]["baseline_regularization"])
        self.assertEqual(results["S2_zero_median_tail_change"]["median_shift"], 0.0)
        self.assertNotEqual(results["S2_zero_median_tail_change"]["higher_q90_before"],
                            results["S2_zero_median_tail_change"]["higher_q90_after"])
        self.assertTrue(results["S14_missing_label_nonidentifiability_and_control_C"][
            "prefix_observations_identical"])

    def test_finite_rank_collision_keeps_raw_prediction_boundary(self):
        c64 = tuple(float(i) for i in range(64))
        candidates = (4.1, 4.9, 5.0)
        self.assertEqual(_cdf(c64, 4.1), _cdf(c64, 4.9))
        score = 4.5
        self.assertEqual(int(score >= 4.1), 1)
        self.assertEqual(int(score >= 4.9), 0)
        mapped, details = _map_percentile(c64, _cdf(c64, 4.1), 4.6, candidates)
        self.assertEqual(mapped, 4.9)
        self.assertFalse(details["raw_boundary_preserved"])
        self.assertEqual(int(score >= mapped), 0)

    def test_group_identity_rescores_do_not_create_new_evidence(self):
        scenario = next(s for s in SCENARIOS if s.key == "S11_provenance_lso")
        engine, memory, authority, _, _ = build_engine(scenario, 37, n_events=256)
        result = engine.run()
        self.assertGreater(memory.version, 0)
        self.assertTrue(all(anchor.source_excluded_rows == 3
                            for anchor in authority.last_snapshot.normal_anchors
                            if not anchor.protected))
        self.assertEqual(len(authority.normals), len({r["group"] for r in authority.normals.values()}))
        self.assertEqual(result["query_count"], len(engine.tape.events))

    def test_s11_same_group_cross_view_and_duplicate_defect_rejected(self):
        scenario = next(s for s in SCENARIOS if s.key == "S11_provenance_lso")
        engine, memory, authority, _, _ = build_engine(scenario, 37, n_events=256)
        group = "one-official-group"
        memory.write(group)
        first = StreamEvent(TapeRecord(0, group, 1, "view-1"), {"base_score": 0.4})
        snapshot = authority.build(first, Feedback(0, True, 0))
        normal = next(a for a in snapshot.normal_anchors if not a.protected)
        self.assertEqual(normal.source_excluded_rows, 3)
        second_view = StreamEvent(TapeRecord(1, group, 2, "view-2"), {"base_score": 0.5})
        with self.assertRaises(ValueError):
            authority.build(second_view, Feedback(1, True, 0))
        defect = StreamEvent(TapeRecord(2, "defect-group", 1, "defect-view"),
                             {"base_score": 1.2})
        authority.build(defect, Feedback(2, True, 1))
        with self.assertRaises(ValueError):
            authority.build(defect, Feedback(3, True, 1))

    def test_offset_updates_only_when_evidence_digest_changes(self):
        c = Stage075BOffset(0.5)
        self.assertEqual(c.finalize_step(), 0.5)
        state = c.state_dict()
        restored = Stage075BOffset(0.5)
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)

    def test_b8_rejoins_gradually_after_departure(self):
        c64 = tuple(i / 100.0 for i in range(64))
        protected = tuple(EvidenceAnchor(f"c64-{i}", x, 0, True)
                          for i, x in enumerate(c64))
        normals = tuple(EvidenceAnchor(f"n-{i}", i / 4000.0, 0, False, 3,
                                       "queried_normal_all_layer_leave_source_out")
                        for i in range(1024))
        defects = tuple(EvidenceAnchor(f"d-{i}", (0.56 if i < 342 else
                                                   0.55 if i < 683 else 0.54), 0, False, 0,
                                       "queried_defect_not_written")
                       for i in range(1024))
        controller = Stage075BController(0.57, "Full")
        revision = 1
        snapshot = EvidenceSnapshot(0, protected + normals, defects,
                                    {"all_layer_lso_applied": True}, revision)
        controller.on_memory_change(snapshot)
        controller.finalize_step()
        self.assertTrue(controller.last_diagnostics["percentile_regularization"])
        self.assertEqual(controller.last_diagnostics["b3_percentile"],
                         _cdf(tuple(sorted(c64)), controller.shadow.theta))
        for _ in range(3):
            revision += 1
            extra = EvidenceAnchor(f"d-extra-{revision}", 0.31, 0, False, 0,
                                   "queried_defect_not_written")
            snapshot = EvidenceSnapshot(0, protected + normals, defects + (extra,),
                                        {"all_layer_lso_applied": True}, revision)
            controller.on_memory_change(snapshot)
            controller.finalize_step()
        self.assertGreater(abs(controller.theta - controller.shadow.theta), 0.01)
        before = controller.theta
        revision += 1
        missing = EvidenceSnapshot(0, protected + normals, (),
                                   {"all_layer_lso_applied": True}, revision)
        controller.on_memory_change(missing)
        controller.finalize_step()
        self.assertLessEqual(abs(controller.theta - before), 0.0100000000001)
        self.assertTrue(controller.last_diagnostics["rejoin_blocked"])
        self.assertGreater(controller.last_diagnostics["remaining_b3_distance"], 0.0)

    def test_b8_full_objective_uses_current_and_transported_cdf_coordinates(self):
        controller = Stage075BController(0.50, "Full")
        current_c64 = tuple(i / 100.0 for i in range(64))
        previous_c64 = tuple(i / 80.0 for i in range(64))
        normals = (0.30, 0.42, 0.52, 0.62, 0.80)
        defects = (0.39, 0.43, 0.54, 0.59, 0.70)
        candidates = (0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70)
        theta_b3, theta_previous_b3, theta_previous_b8 = 0.45, 0.40, 0.50
        controller.c64 = current_c64
        controller.previous_c64 = previous_c64
        controller.previous_b3 = theta_previous_b3
        controller.previous_theta = theta_previous_b8
        controller.normals = normals
        controller.defects = defects

        threshold, diagnostics = controller._optimize(
            candidates, theta_b3, theta_previous_b8, 0.2, standardized=False)

        def empirical_j(value):
            fpr = sum(score >= value for score in normals) / len(normals)
            fnr = sum(score < value for score in defects) / len(defects)
            return 1.0 - fpr - fnr

        def cdf(reference, value):
            return sum(score <= value for score in reference) / 64.0

        current = tuple(sorted(current_c64))
        previous = tuple(sorted(previous_c64))
        u_b3 = cdf(current, theta_b3)
        rho_previous = (cdf(previous, theta_previous_b8) -
                        cdf(previous, theta_previous_b3))
        u_transported = min(1.0, max(0.0, u_b3 + rho_previous))
        penalty = math.sqrt(math.log(2 * len(candidates) / 0.05) /
                             (2 * len(normals)))
        penalty += math.sqrt(math.log(2 * len(candidates) / 0.05) /
                             (2 * len(defects)))
        baseline_j = empirical_j(theta_b3)
        objectives = {
            candidate: (empirical_j(candidate) - baseline_j - penalty
                        - 0.05 * abs(cdf(current, candidate) - u_b3)
                        - 0.05 * abs(cdf(current, candidate) - u_transported))
            for candidate in candidates
        }
        best_objective = max(objectives.values())
        expected_candidate = min(
            (candidate for candidate, objective in objectives.items()
             if objective == best_objective),
            key=lambda value: (abs(value - theta_b3),
                               abs(value - theta_previous_b8), value))

        self.assertEqual(diagnostics["b3_percentile"], u_b3)
        self.assertEqual(diagnostics["previous_b8_relative_percentile"], rho_previous)
        self.assertEqual(diagnostics["transported_percentile_target"], u_transported)
        self.assertAlmostEqual(diagnostics["best_penalized_objective"], best_objective)
        self.assertEqual(diagnostics["candidate_threshold"], expected_candidate)
        self.assertAlmostEqual(diagnostics["selected_candidate_percentile"],
                               cdf(current, expected_candidate))
        self.assertTrue(diagnostics["percentile_regularization"])
        self.assertEqual(threshold, theta_b3)  # The registered penalty keeps this fixture at B3.

    def test_s13_checkpoint_resume_exact_replay(self):
        result = checkpoint_resume_check()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["full_resume_identical"])
        self.assertGreaterEqual(result["checkpoint_count"], 4)


if __name__ == "__main__":
    unittest.main()
