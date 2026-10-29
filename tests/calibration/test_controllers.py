import unittest
from dataclasses import fields

from experiments.calibration.controllers import (AdaptedDualACI, FrozenTheta,
    FrozenThresholdFallback, OriginalGuardedSidecar, RollingQ90QueriedNormal,
    StateConsistentQuantile, higher_quantile,
    theta0_from_c64, _normal_fpr_lower_bound, _defect_fnr_upper_bound)
from experiments.calibration.protocol import (EvidenceAnchor, EvidenceSnapshot,
    Feedback, Observation, Prediction, ProtocolError)
from experiments.calibration.spot_dspot import (DSPOTController, POTConfig,
                                                SPOTController, DSPOT_VERSION)
from experiments.calibration.sun_cs import SunCSController


def obs(t=0, score=0.5, version=0):
    return Observation(t, f"opaque-{t}", 1, score, version, "state-digest")


class ControllerTests(unittest.TestCase):
    def test_controller_observation_and_feedback_have_no_unqueried_label_path(self):
        names = {field.name for field in fields(Observation)}
        self.assertFalse(names.intersection({"label", "official_sample_key", "path", "image", "mask"}))
        with self.assertRaises(ProtocolError):
            Feedback(0, False, 1)

    def test_operational_fallback_records_startup_and_abstention_without_changing_source(self):
        spot = SPOTController(POTConfig(n_init=4, level=.5, risk_q=.01))
        spot_hybrid = FrozenThresholdFallback(spot, .5, name="SPOT_HYBRID")
        startup = spot_hybrid.predict(obs(0, .7))
        self.assertEqual(startup.reason, "STARTUP_FALLBACK")
        self.assertEqual(startup.label, 1)
        self.assertEqual(startup.details["fallback_threshold_source"],
                         "theta0_frozen_C64")
        self.assertEqual(spot.observations, 0)
        spot_hybrid.observe_feedback(obs(0, .7), startup, Feedback(0, False))
        spot_hybrid.observe_score(obs(0, .7))
        self.assertEqual(spot.observations, 1)

        dspot = DSPOTController(POTConfig(n_init=4, level=.5, risk_q=.01,
                                          version=DSPOT_VERSION), window=2)
        dspot_hybrid = FrozenThresholdFallback(dspot, .5, name="DSPOT_HYBRID")
        dspot_startup = dspot_hybrid.predict(obs(0, .8))
        self.assertEqual(dspot_startup.reason, "STARTUP_FALLBACK")
        dspot_hybrid.observe_score(obs(0, .8))
        self.assertEqual(dspot.observations, 1)

        sun = SunCSController()
        sun_hybrid = FrozenThresholdFallback(sun, .5, name="SUN_HYBRID")
        cold = sun_hybrid.predict(obs(0, .5))
        self.assertEqual(cold.reason, "ABSTENTION_FALLBACK")
        self.assertEqual(cold.details["source_reason"], "ABSTAIN_COLD_START")
        sun_hybrid.observe_feedback(obs(0, .5), cold, Feedback(0, False))
        sun_hybrid.observe_score(obs(0, .5))
        tied = sun_hybrid.predict(obs(1, .5))
        self.assertEqual(tied.reason, "ABSTENTION_FALLBACK")
        self.assertEqual(tied.details["source_reason"], "SUN_CS_ABSTAIN")
        self.assertEqual(len(sun.history), 1)

    def test_b2_uses_historical_sidecar_with_unqueried_fallback_disabled(self):
        values = [i / 100.0 for i in range(64)]
        c = OriginalGuardedSidecar(values, theta0_from_c64(values))
        self.assertFalse(c.source.cfg.enable_normal_fallback)
        self.assertFalse(c.source.cfg.enable_fast_upward_rescue)
        self.assertFalse(c.source.cfg.enable_balanced_rescue)
        before = c.source.n_normals_labeled
        c.observe_feedback(obs(0, .2), Prediction(0, c.theta), Feedback(0, False))
        self.assertEqual(c.source.n_normals_labeled, before)
        c.observe_feedback(obs(1, .2), Prediction(0, c.theta), Feedback(1, True, 0))
        self.assertEqual(c.source.n_normals_labeled, before + 1)

    def test_higher_quantile_boundary_ties(self):
        self.assertEqual(higher_quantile([0, 1, 1, 2], .5), 1)
        self.assertEqual(higher_quantile([0, 1, 2, 3], .9), 3)

    def test_theta0_uses_d2_higher_q90_at_zero_based_rank_57(self):
        values = list(range(64))
        self.assertEqual(theta0_from_c64(values), 57)
        with self.assertRaisesRegex(ValueError, "64"):
            theta0_from_c64(values[:4])

    def test_b1_only_queried_verified_normals_enter_buffer(self):
        c = RollingQ90QueriedNormal(theta=0.25, window=4, min_count=2)
        c.observe_feedback(obs(0, .1), Prediction(0, .25), Feedback(0, False))
        c.observe_feedback(obs(1, .2), Prediction(0, .25), Feedback(1, True, 1))
        self.assertEqual(len(c.history), 0)
        c.observe_feedback(obs(2, .3), Prediction(1, .25), Feedback(2, True, 0))
        c.observe_feedback(obs(3, .4), Prediction(1, .3), Feedback(3, True, 0))
        self.assertEqual(list(c.history), [.3, .4])
        self.assertEqual(c.theta, .4)

    def test_b3_updates_once_per_memory_version(self):
        c = StateConsistentQuantile(theta=.5, delta=.01)
        anchors = tuple(EvidenceAnchor(f"c64:{i}", .8, 1, True) for i in range(64))
        snapshot = EvidenceSnapshot(1, anchors, (), {})
        c.on_memory_change(snapshot)
        self.assertEqual(c.fnr_status, "FNR_UNAVAILABLE")
        self.assertEqual(c.finalize_step(), .51)
        self.assertEqual(c.finalize_step(), .51)

    def test_b4_updates_only_the_disclosed_class_side(self):
        c = AdaptedDualACI(theta=.5)
        prediction = Prediction(1, .5)
        c.observe_feedback(obs(), prediction, Feedback(0, False))
        self.assertEqual(c.alpha_normal, .1)
        c.observe_feedback(obs(), prediction, Feedback(0, True, 0))
        self.assertAlmostEqual(c.alpha_normal, .0955)
        self.assertEqual(c.alpha_defect, .1)
        c.on_memory_change(EvidenceSnapshot(0, (), (), {}))
        self.assertEqual(c.finalize_step(), .5)
        self.assertEqual(c.last_status, "NONE")

    def test_b4_missing_side_is_not_zero_risk(self):
        c = AdaptedDualACI(theta=.5)
        defects = tuple(EvidenceAnchor(f"d:{i}", .1 + i * .01, 0, False)
                        for i in range(10))
        c.on_memory_change(EvidenceSnapshot(0, (), defects, {}))
        c.finalize_step()
        self.assertEqual(c.last_status, "DEFECT_ONLY")
        self.assertEqual(c.normal_scores, ())
        self.assertLessEqual(c.theta, .2)

    def test_b4_normal_lower_and_defect_upper_geometry_is_not_reversed(self):
        c = AdaptedDualACI(theta=.5)
        normals = tuple(EvidenceAnchor(f"n:{i}", v, 0, True)
                        for i, v in enumerate((.1, .2)))
        defects = tuple(EvidenceAnchor(f"d:{i}", v, 0, False,
                                      source_exclusion_policy="queried_defect_not_written")
                        for i, v in enumerate((.8, .9)))
        c.on_memory_change(EvidenceSnapshot(0, normals, defects, {}, 1))
        self.assertEqual(c.finalize_step(), .5)
        self.assertFalse(c.conflict)
        self.assertEqual(c.last_status, "BOTH")

    def test_b4_overlapping_scores_are_genuinely_conflicting(self):
        c = AdaptedDualACI(theta=.5)
        normals = tuple(EvidenceAnchor(f"n:{i}", v, 0, True)
                        for i, v in enumerate((.1, .9)))
        defects = tuple(EvidenceAnchor(f"d:{i}", v, 0, False,
                                      source_exclusion_policy="queried_defect_not_written")
                        for i, v in enumerate((.1, .9)))
        c.on_memory_change(EvidenceSnapshot(0, normals, defects, {}, 1))
        self.assertEqual(c.finalize_step(), .5)
        self.assertTrue(c.conflict)
        self.assertEqual(c.last_status, "CONFLICT")

    def test_b4_exact_error_count_boundaries_handle_ties(self):
        # At theta == .5 both tied normals are false positives because the
        # operational predicate is score >= theta. The lower bound moves past
        # the tie so no more than one of two normals is counted.
        lower = _normal_fpr_lower_bound((.5, .5), .5)
        self.assertGreater(lower, .5)
        self.assertEqual(sum(x >= lower for x in (.5, .5)), 0)
        upper = _defect_fnr_upper_bound((.5, .5), .5)
        self.assertEqual(upper, .5)
        self.assertEqual(sum(x < upper for x in (.5, .5)), 0)

    def test_b4_current_memory_rescore_reverses_feasibility_without_stale_scores(self):
        c = AdaptedDualACI(theta=.5)
        n0 = tuple(EvidenceAnchor(f"n:{i}", v, 0, True)
                   for i, v in enumerate((.1, .2)))
        d0 = tuple(EvidenceAnchor(f"d:{i}", v, 0, False,
                                  source_exclusion_policy="queried_defect_not_written")
                   for i, v in enumerate((.8, .9)))
        c.on_memory_change(EvidenceSnapshot(0, n0, d0, {}, 1))
        self.assertEqual(c.finalize_step(), .5)
        # The same anchor identities are rescored under M1 and now overlap.
        n1 = tuple(EvidenceAnchor(f"n:{i}", .7 + .1*i, 1, True)
                   for i in range(2))
        d1 = tuple(EvidenceAnchor(f"d:{i}", .7 + .1*i, 1, False,
                                  source_exclusion_policy="queried_defect_not_written")
                   for i in range(2))
        c.on_memory_change(EvidenceSnapshot(1, n1, d1, {}, 2))
        c.finalize_step()
        self.assertTrue(c.conflict)
        self.assertEqual(c.evidence_version, 1)

    def test_b4_refreshes_new_queried_class_evidence_without_memory_change(self):
        c = AdaptedDualACI(theta=.5)
        normals = tuple(EvidenceAnchor(f"n:{i}", .2 + i * .001, 0, True)
                        for i in range(64))
        c.on_memory_change(EvidenceSnapshot(0, normals, (), {}, 1))
        self.assertEqual(c.last_status, "NORMAL_ONLY")
        defects = (EvidenceAnchor("d:0", .3, 0, False),)
        c.on_memory_change(EvidenceSnapshot(0, normals, defects, {}, 2))
        self.assertEqual(c.last_status, "BOTH")
        self.assertEqual(c.defect_scores, (.3,))


if __name__ == "__main__": unittest.main()
