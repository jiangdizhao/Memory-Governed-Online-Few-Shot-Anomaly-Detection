import unittest
from dataclasses import fields

from experiments.calibration.controllers import (AdaptedDualACI, FrozenTheta,
    OriginalGuardedSidecar, RollingQ90QueriedNormal, StateConsistentQuantile, higher_quantile,
    theta0_from_c64)
from experiments.calibration.protocol import (EvidenceAnchor, EvidenceSnapshot,
    Feedback, Observation, Prediction, ProtocolError)


def obs(t=0, score=0.5, version=0):
    return Observation(t, f"opaque-{t}", 1, score, version, "state-digest")


class ControllerTests(unittest.TestCase):
    def test_controller_observation_and_feedback_have_no_unqueried_label_path(self):
        names = {field.name for field in fields(Observation)}
        self.assertFalse(names.intersection({"label", "official_sample_key", "path", "image", "mask"}))
        with self.assertRaises(ProtocolError):
            Feedback(0, False, 1)

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
