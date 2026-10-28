from __future__ import annotations

import math
import unittest

from experiments.calibration.sun_cs import (
    SunCSUndefinedSourceBehavior,
    change_detected,
    classify_score,
    confidence_radius,
    empirical_quantile,
    intersect_intervals,
    quantile_confidence_set,
    unique_offline_match,
)


class SunCSReferenceTests(unittest.TestCase):
    def test_radius_matches_paper_equation_and_time_zero(self):
        self.assertEqual(confidence_radius(0, 0.5), 1.0)
        expected = min(1.0, 0.85 / math.sqrt(100) * math.sqrt(
            math.log(math.log(math.e * 100)) + 0.8 * math.log(1612 / 0.5)))
        self.assertAlmostEqual(confidence_radius(100, 0.5), expected, places=14)
        self.assertEqual(confidence_radius(1, 1e-6), 1.0)

    def test_midpoint_quantile_uses_paper_floor_ceil_ranks(self):
        values = [7.0, 1.0, 9.0, 3.0, 5.0]
        # q*n=2.75: midpoint of one-based order statistics 2 and 3.
        self.assertEqual(empirical_quantile(0.55, values), 4.0)
        # q*n=3: the same order statistic is used twice.
        self.assertEqual(empirical_quantile(0.60, values), 5.0)

    def test_equation_one_matches_hand_computed_interior_confidence_set(self):
        values = [float(i) for i in range(1, 101)]
        delta, p = 0.5, 0.5
        radius = confidence_radius(len(values), delta)
        low_q, high_q = p - 2 * radius, p + 2 * radius
        def ref(q):
            rank = q * len(values)
            low, high = math.floor(rank), math.ceil(rank)
            return (values[low - 1] + values[high - 1]) / 2
        actual = quantile_confidence_set(p, delta, values)
        self.assertAlmostEqual(actual[0], ref(low_q), places=12)
        self.assertAlmostEqual(actual[1], ref(high_q), places=12)

    def test_algorithm_one_preserves_strict_anomaly_and_abstention_edges(self):
        interval = (2.0, 4.0)
        self.assertEqual(classify_score(1.99, interval), 0)
        self.assertIsNone(classify_score(2.0, interval))
        self.assertIsNone(classify_score(4.0, interval))
        self.assertEqual(classify_score(math.nextafter(4.0, math.inf), interval), 1)

    def test_algorithm_two_intersection_change_predicate(self):
        self.assertFalse(change_detected(
            [(0.0, 2.0), (1.0, 3.0)], [(1.5, 2.5), (1.0, 4.0)]))
        self.assertTrue(change_detected(
            [(0.0, 1.0), (0.2, 0.8)], [(2.0, 3.0), (2.2, 2.8)]))
        self.assertTrue(change_detected(
            [(0.0, 0.5), (0.8, 1.0)], [(0.0, 1.0)]))

    def test_algorithm_four_uses_exactly_one_matching_offline_set(self):
        offline = [(0.0, 1.0), (2.0, 3.0), (5.0, 6.0)]
        self.assertEqual(unique_offline_match(offline, (0.5, 2.5)), None)
        self.assertEqual(unique_offline_match(offline, (2.1, 2.9)), 1)
        self.assertEqual(unique_offline_match(offline, (8.0, 9.0)), None)

    def test_undefined_empty_and_clipped_endpoint_cases_fail_closed(self):
        with self.assertRaises(SunCSUndefinedSourceBehavior):
            quantile_confidence_set(0.9, 0.1, [])
        with self.assertRaises(SunCSUndefinedSourceBehavior):
            quantile_confidence_set(0.9, 0.1, [float(i) for i in range(1, 101)])
        with self.assertRaises(SunCSUndefinedSourceBehavior):
            intersect_intervals([])


if __name__ == "__main__":
    unittest.main()
