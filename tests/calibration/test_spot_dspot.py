from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.stats import genpareto

from experiments.calibration.protocol import Observation
from experiments.calibration.spot_dspot import (
    DSPOTController, POTConfig, SPOTController, grimshaw_fit, pot_threshold,
    DSPOT_VERSION, SPOT_VERSION,
)


def observation(t: int, score: float) -> Observation:
    return Observation(t, f"opaque-{t}", 1, float(score), 0, "synthetic-state")


class SpotDspotTests(unittest.TestCase):
    def test_grimshaw_matches_independent_scipy_gpd_mle(self):
        excesses = genpareto.rvs(c=0.2, scale=1.5, size=500,
                                 random_state=123)
        fit = grimshaw_fit(excesses, starts=10)
        ref_shape, _, ref_scale = genpareto.fit(excesses, floc=0)
        self.assertAlmostEqual(fit.shape, ref_shape, delta=5e-4)
        self.assertAlmostEqual(fit.scale, ref_scale, delta=5e-4)
        self.assertGreaterEqual(len(fit.roots), 1)
        self.assertTrue(math.isfinite(fit.log_likelihood))

    def test_pot_threshold_equation_and_near_zero_shape_limit(self):
        expected = 10.0 + 2.0 * math.log(200.0)
        self.assertAlmostEqual(
            pot_threshold(10.0, 0.0, 2.0, accepted_count=100,
                          excess_count=20, risk_q=0.001), expected, places=12)
        small_shape = 1e-10
        actual = pot_threshold(10.0, small_shape, 2.0,
                               accepted_count=100, excess_count=20,
                               risk_q=0.001)
        self.assertAlmostEqual(actual, expected, delta=1e-7)

    def test_spot_warmup_missingness_alarm_exclusion_and_peak_update(self):
        config = POTConfig(n_init=100, level=0.90, risk_q=0.005,
                           version=SPOT_VERSION)
        initial = genpareto.rvs(c=0.2, scale=1.0, size=100,
                                random_state=7)
        controller = SPOTController(config)
        for t, score in enumerate(initial):
            pred = controller.predict(observation(t, float(score)))
            self.assertIsNone(pred.label)
            self.assertFalse(pred.available)
            controller.observe_score(observation(t, float(score)))
        self.assertIsNotNone(controller.theta)
        self.assertEqual(controller.state_dict()["core"]["n"], 100)

        core = controller.core
        threshold_before = float(core.threshold)
        base = float(core.threshold_base)
        self.assertGreater(threshold_before, base)
        alarm = threshold_before + max(10.0, abs(threshold_before) * 10.0)
        prediction = controller.predict(observation(100, alarm))
        self.assertEqual(prediction.label, 1)
        old_n, old_peaks = core.n, len(core.peaks)
        controller.observe_score(observation(100, alarm))
        self.assertEqual(core.n, old_n)
        self.assertEqual(len(core.peaks), old_peaks)
        self.assertEqual(core.alarms, 1)

        peak = (base + threshold_before) / 2.0
        prediction = controller.predict(observation(101, peak))
        self.assertEqual(prediction.label, 0)
        controller.observe_score(observation(101, peak))
        self.assertEqual(core.n, old_n + 1)
        self.assertEqual(len(core.peaks), old_peaks + 1)
        self.assertEqual(core.threshold_refits, 2)

    def test_spot_checkpoint_roundtrip_is_deterministic(self):
        config = POTConfig(n_init=100, level=0.90, risk_q=0.005,
                           version=SPOT_VERSION)
        initial = genpareto.rvs(c=0.2, scale=1.0, size=100,
                                random_state=19)
        first = SPOTController(config)
        for t, value in enumerate(initial):
            first.observe_score(observation(t, float(value)))
        checkpoint = first.state_dict()
        resumed = SPOTController(config)
        resumed.load_state_dict(checkpoint)
        self.assertEqual(resumed.state_dict(), checkpoint)
        for t, value in enumerate((0.1, 0.2, 10.0, 0.4), start=100):
            left = first.predict(observation(t, value))
            right = resumed.predict(observation(t, value))
            self.assertEqual(left, right)
            first.observe_score(observation(t, value))
            resumed.observe_score(observation(t, value))
            self.assertEqual(first.state_dict(), resumed.state_dict())

    def test_dspot_uses_warmup_residuals_and_excludes_alarms_from_mean(self):
        config = POTConfig(n_init=50, level=0.90, risk_q=0.005,
                           version=DSPOT_VERSION)
        window = 4
        raw = [0.2 * t + 0.3 * math.sin(0.7 * t) +
               0.07 * math.cos(1.9 * t) for t in range(window + config.n_init)]
        controller = DSPOTController(config, window=window)
        expected_residuals = []
        accepted_window: list[float] = []
        for t, value in enumerate(raw):
            pred = controller.predict(observation(t, value))
            self.assertIsNone(pred.label)
            self.assertFalse(pred.available)
            if len(accepted_window) < window:
                accepted_window.append(value)
            else:
                expected_residuals.append(value - sum(accepted_window) / window)
                accepted_window = [*accepted_window[1:], value]
            controller.observe_score(observation(t, value))
        self.assertTrue(controller.core.ready)
        np.testing.assert_allclose(controller.initial_residuals,
                                   expected_residuals, rtol=0, atol=1e-12)
        self.assertEqual(len(controller.initial_residuals), config.n_init)
        self.assertEqual(len(controller.local_window), window)

        absolute_threshold = controller.theta
        self.assertIsNotNone(absolute_threshold)
        alarm = float(absolute_threshold) + max(100.0, abs(float(absolute_threshold)) * 10)
        pred = controller.predict(observation(len(raw), alarm))
        self.assertEqual(pred.label, 1)
        before_window = list(controller.local_window)
        before_n = controller.core.n
        controller.observe_score(observation(len(raw), alarm))
        self.assertEqual(controller.local_window, before_window)
        self.assertEqual(controller.core.n, before_n)
        self.assertEqual(controller.alarms, 1)

    def test_insufficient_peak_initialization_fails_closed(self):
        controller = SPOTController(POTConfig(n_init=10, level=0.8,
                                              risk_q=0.01))
        with self.assertRaisesRegex(ValueError, "BLOCKED_INSUFFICIENT_TAIL"):
            for t in range(10):
                controller.observe_score(observation(t, 1.0))


if __name__ == "__main__":
    unittest.main()
