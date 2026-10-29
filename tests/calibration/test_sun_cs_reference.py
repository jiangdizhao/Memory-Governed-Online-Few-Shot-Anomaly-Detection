from __future__ import annotations

import math
import unittest

from experiments.calibration.protocol import Observation
from experiments.calibration.engine import PrequentialEngine
from experiments.calibration.metrics import PrivateOutcomeLedger
from experiments.calibration.protocol import ScoreResult, StreamEvent, TapeRecord
from experiments.calibration.query_tape import build_qa_ts
from experiments.calibration.sun_cs import (
    EMPIRICAL_EXTREMA_SENSITIVITY_POLICY, EXTENDED_REAL_ENDPOINT_POLICY,
    PRIMARY_ENDPOINT_POLICY, SUN_CS_CHECKPOINT_SCHEMA, SunCSController,
    SunCSUndefinedSourceBehavior, algorithm2_optimized, algorithm2_reference,
    algorithm3_reference, classify_score, confidence_radius,
    empirical_quantile, encode_interval, quantile_confidence_set,
    unique_offline_match,
)


class SunCSReferenceTests(unittest.TestCase):
    def test_radius_matches_paper_equation_without_clamping(self):
        self.assertEqual(confidence_radius(0, 0.5), 1.0)
        raw = 0.85 / math.sqrt(100) * math.sqrt(
            math.log(math.log(math.e * 100)) + 0.8 * math.log(1612 / 0.5))
        self.assertAlmostEqual(confidence_radius(100, 0.5), raw, places=14)
        self.assertGreater(confidence_radius(1, 1e-6), 1.0)

    def test_empirical_quantile_midpoint_and_endpoint_conventions(self):
        values = [7.0, 1.0, 9.0, 3.0, 5.0]
        self.assertEqual(empirical_quantile(0.55, values), 4.0)
        self.assertEqual(empirical_quantile(0.60, values), 5.0)
        self.assertEqual(empirical_quantile(
            0.0, values, endpoint_policy=EMPIRICAL_EXTREMA_SENSITIVITY_POLICY), 1.0)
        self.assertEqual(empirical_quantile(
            1.0, values, endpoint_policy=EMPIRICAL_EXTREMA_SENSITIVITY_POLICY), 9.0)
        self.assertEqual(empirical_quantile(
            0.01, values, endpoint_policy=EMPIRICAL_EXTREMA_SENSITIVITY_POLICY), 1.0)
        self.assertEqual(empirical_quantile(0.0, values,
                                            endpoint_policy=EXTENDED_REAL_ENDPOINT_POLICY),
                         -math.inf)
        self.assertEqual(empirical_quantile(1.0, values,
                                            endpoint_policy=EXTENDED_REAL_ENDPOINT_POLICY),
                         math.inf)
        self.assertEqual(encode_interval((-math.inf, math.inf)),
                         [{"kind": "negative_infinity"},
                          {"kind": "positive_infinity"}])

    def test_equation_one_interior_and_clipped_endpoint_confidence_sets(self):
        values = [float(i) for i in range(1, 101)]
        delta, p = 0.5, 0.5
        radius = confidence_radius(len(values), delta)
        low_q, high_q = p - 2 * radius, p + 2 * radius
        def ref(q):
            rank = q * len(values)
            low, high = max(1, math.floor(rank)), max(1, math.ceil(rank))
            return values[low - 1] / 2 + values[high - 1] / 2
        actual = quantile_confidence_set(p, delta, values)
        self.assertAlmostEqual(actual[0], ref(low_q), places=12)
        self.assertAlmostEqual(actual[1], ref(high_q), places=12)
        small = [0.25, 0.5, 0.75]
        self.assertEqual(quantile_confidence_set(.99, .01, small,
                         endpoint_policy=EMPIRICAL_EXTREMA_SENSITIVITY_POLICY),
                         (min(small), max(small)))
        self.assertEqual(quantile_confidence_set(
            .99, .01, small, endpoint_policy=EXTENDED_REAL_ENDPOINT_POLICY),
            (-math.inf, math.inf))
        with self.assertRaises(SunCSUndefinedSourceBehavior):
            quantile_confidence_set(.99, .01, [])

    def test_algorithm_one_strict_anomaly_benign_and_inclusive_abstention(self):
        interval = (2.0, 4.0)
        self.assertEqual(classify_score(1.99, interval), 0)
        self.assertIsNone(classify_score(2.0, interval))
        self.assertIsNone(classify_score(4.0, interval))
        self.assertEqual(classify_score(math.nextafter(4.0, math.inf), interval), 1)
        self.assertEqual(classify_score(0.0, (-math.inf, math.inf)), None)

    def test_algorithm2_optimized_matches_independent_reference(self):
        sequences = [
            [0.3], [1.0, 1.0, 1.0], [0.1, 0.2, 0.9, 0.4, 0.8],
            [float((i * 17) % 23) / 23 for i in range(17)],
            [0.1 + (i % 7) * .01 for i in range(31)],
        ]
        for seq in sequences:
            for policy in (PRIMARY_ENDPOINT_POLICY,
                           EMPIRICAL_EXTREMA_SENSITIVITY_POLICY):
                ref = algorithm2_reference(seq, .99, .01, endpoint_policy=policy)
                optimized = algorithm2_optimized(seq, .99, .01,
                                                endpoint_policy=policy)
                self.assertEqual(optimized, ref, seq)

    def test_algorithm3_event_trace_matches_independent_reference(self):
        stationary = [0.5 + .1 * math.sin(i * .31) for i in range(25)]
        piecewise = [0.05 + .01 * (i % 5) for i in range(20)] + [
            0.75 + .01 * (i % 5) for i in range(20)]
        for scores in (stationary, piecewise):
            expected = algorithm3_reference(
                scores, .99, .01,
                endpoint_policy=EMPIRICAL_EXTREMA_SENSITIVITY_POLICY)
            controller = SunCSController(
                .99, .01,
                endpoint_policy=EMPIRICAL_EXTREMA_SENSITIVITY_POLICY)
            observed = []
            for t, score in enumerate(scores):
                obs = Observation(t, f"sun-{t}", t % 5 + 1, score, 0,
                                  f"fixed-score-state-{t}")
                prediction = controller.predict(obs)
                observed.append({
                    "t": t, "prediction": prediction.label,
                    "reason": prediction.reason,
                    "confidence_set": prediction.details["confidence_set"],
                    "change_detected": prediction.details["change_detected"],
                    "segment_start": prediction.details["segment_start"],
                    "history_size": prediction.details["history_size"],
                })
                controller.observe_score(obs)
            self.assertEqual(observed, expected)
        self.assertEqual(expected[0]["reason"], "ABSTAIN_COLD_START")
        self.assertTrue(any(row["reason"] == "ABSTAIN_CHANGE_DETECTED"
                            for row in algorithm3_reference(
                                piecewise, endpoint_policy=EMPIRICAL_EXTREMA_SENSITIVITY_POLICY)))

    def test_empirical_extrema_false_reset_pathology_and_conservative_control(self):
        a, b = 0.25, 0.75
        for policy in (EMPIRICAL_EXTREMA_SENSITIVITY_POLICY,
                       PRIMARY_ENDPOINT_POLICY):
            reference = algorithm2_reference([a, b], .99, .01,
                                             endpoint_policy=policy)
            optimized = algorithm2_optimized([a, b], .99, .01,
                                             endpoint_policy=policy)
            self.assertEqual(optimized, reference)
            trace = algorithm3_reference([a, b, a], .99, .01,
                                          endpoint_policy=policy)
            controller = SunCSController(.99, .01, endpoint_policy=policy)
            live = []
            for t, score in enumerate((a, b, a)):
                obs = Observation(t, f"endpoint-pathology-{policy.name}-{t}",
                                  1, score, 0, "fixed-state")
                pred = controller.predict(obs)
                live.append({
                    "t": t, "prediction": pred.label,
                    "reason": pred.reason,
                    "confidence_set": pred.details["confidence_set"],
                    "change_detected": pred.details["change_detected"],
                    "segment_start": pred.details["segment_start"],
                    "history_size": pred.details["history_size"],
                })
                controller.observe_score(obs)
            self.assertEqual(live, trace)

        empirical = EMPIRICAL_EXTREMA_SENSITIVITY_POLICY
        self.assertEqual(quantile_confidence_set(.99, .01, [a],
                         endpoint_policy=empirical), (a, a))
        self.assertEqual(quantile_confidence_set(.99, .01, [a, b],
                         endpoint_policy=empirical), (a, b))
        self.assertEqual(quantile_confidence_set(.99, .01, [b],
                         endpoint_policy=empirical), (b, b))
        self.assertTrue(algorithm2_reference(
            [a, b], .99, .01, endpoint_policy=empirical)["change_detected"])

        for values in ([a], [a, b], [b]):
            self.assertEqual(quantile_confidence_set(
                .99, .01, values, endpoint_policy=PRIMARY_ENDPOINT_POLICY),
                (-math.inf, math.inf))
        self.assertFalse(algorithm2_reference(
            [a, b], .99, .01,
            endpoint_policy=PRIMARY_ENDPOINT_POLICY)["change_detected"])
        self.assertFalse(algorithm2_optimized(
            [a, b], .99, .01,
            endpoint_policy=PRIMARY_ENDPOINT_POLICY)["change_detected"])
        self.assertNotEqual(algorithm3_reference(
            [a, b, a], .99, .01,
            endpoint_policy=PRIMARY_ENDPOINT_POLICY)[2]["reason"],
            "ABSTAIN_CHANGE_DETECTED")

    def test_conservative_upper_endpoint_clipping_precludes_anomaly(self):
        values = [float(i) for i in range(1, 3816)]
        for n in (64, 256, 1050, 3815):
            radius = confidence_radius(n, .01)
            low_q = max(.99 - 2 * radius, 0.0)
            high_q = min(.99 + 2 * radius, 1.0)
            self.assertEqual(high_q, 1.0)
            interval = quantile_confidence_set(
                .99, .01, values[:n], endpoint_policy=PRIMARY_ENDPOINT_POLICY)
            self.assertEqual(interval[1], math.inf)
            self.assertIsNone(classify_score(values[n - 1] + 1.0, interval))
            self.assertGreater(low_q, 0.0)

    def test_algorithm3_stationary_coverage_and_piecewise_reset_diagnostics(self):
        stationary = [0.5 + .08 * math.sin(i * .19) for i in range(80)]
        piecewise = [0.1 + .02 * math.sin(i * .37) for i in range(40)] + [
            0.8 + .02 * math.sin(i * .37) for i in range(40)]
        stationary_trace = algorithm3_reference(
            stationary, endpoint_policy=EMPIRICAL_EXTREMA_SENSITIVITY_POLICY)
        piecewise_trace = algorithm3_reference(
            piecewise, endpoint_policy=EMPIRICAL_EXTREMA_SENSITIVITY_POLICY)
        for trace in (stationary_trace, piecewise_trace):
            self.assertEqual(trace[0]["reason"], "ABSTAIN_COLD_START")
            covered = sum(row["prediction"] is not None for row in trace)
            abstained = len(trace) - covered
            self.assertEqual(covered + abstained, len(trace))
            self.assertGreater(covered, 0)
        self.assertTrue(any(row["change_detected"] for row in piecewise_trace))
        conservative_piecewise = algorithm3_reference(piecewise)
        self.assertFalse(any(row["change_detected"]
                             for row in conservative_piecewise))

    def test_checkpoint_restore_reproduces_remaining_event_trace(self):
        scores = [0.2 + .1 * math.sin(i * .23) for i in range(36)]
        scores[18:] = [x + .65 for x in scores[18:]]
        original = SunCSController()
        first_half = []
        for t, score in enumerate(scores[:19]):
            obs = Observation(t, f"resume-{t}", 1, score, 0, f"state-{t}")
            first_half.append(original.predict(obs).label)
            original.observe_score(obs)
        state = original.state_dict()
        restored = SunCSController()
        restored.load_state_dict(state)
        expected = []
        actual = []
        for t, score in enumerate(scores[19:], start=19):
            obs = Observation(t, f"resume-{t}", 1, score, 0, f"state-{t}")
            expected.append(original.predict(obs))
            original.observe_score(obs)
            actual.append(restored.predict(obs))
            restored.observe_score(obs)
        self.assertEqual(actual, expected)
        self.assertEqual(restored.state_dict(), original.state_dict())
        self.assertEqual(restored.state_dict()["schema"], SUN_CS_CHECKPOINT_SCHEMA)

    def test_checkpoint_v1_is_rejected_after_endpoint_candidate_versioning(self):
        state = SunCSController().state_dict()
        state["version"] = "B6-SUN-CS-A3-PAPER-EQUATIONS-DECLARED-CONVENTIONS-v1.0.0"
        state.pop("schema")
        state.pop("endpoint_policy_schema")
        with self.assertRaisesRegex(ValueError, "checkpoint identity mismatch"):
            SunCSController().load_state_dict(state)

    def test_engine_event_boundary_checkpoint_resume_is_identical(self):
        scores = [0.2 + .08 * math.sin(i * .17) for i in range(44)]
        scores[22:] = [x + .6 for x in scores[22:]]
        records = [TapeRecord(i, f"sun-engine-group-{i}", i % 5 + 1,
                              f"sun-engine-view-{i}") for i in range(len(scores))]
        tape = build_qa_ts(records)
        labels = [int(i % 3 == 0) for i in range(len(scores))]
        events = [StreamEvent(record, scores[i]) for i, record in enumerate(records)]

        def make_engine():
            return PrequentialEngine(
                tape=tape, events=events, outcomes=PrivateOutcomeLedger(labels),
                controllers=[SunCSController()],
                score_fn=lambda event: ScoreResult(float(event.payload), 0,
                                                    f"sun-state-{event.record.position}"),
                memory_state_fn=lambda: (0, "sun-fixed-memory"),
                write_fn=lambda *_: None,
                provenance_id="sun-a3-engine-checkpoint-v1")

        full = make_engine().run()
        first = make_engine().run(max_events=19)
        self.assertEqual(first["status"], "INCOMPLETE")
        resumed = make_engine()
        resumed.load_state_dict(first["checkpoint"])
        continued = resumed.run()
        for key in ("status", "cursor", "tape_digest", "scores", "predictions",
                    "prediction_availability", "trace", "metrics", "controller_states"):
            self.assertEqual(continued[key], full[key], key)

    def test_short_stream_sample_size_diagnostic_is_explicit(self):
        diagnostic = []
        for n in (64, 256, 1050, 3815):
            radius = confidence_radius(n, .01)
            diagnostic.append((n, radius, max(.99 - 2 * radius, 0),
                               min(.99 + 2 * radius, 1)))
        self.assertEqual([row[0] for row in diagnostic], [64, 256, 1050, 3815])
        self.assertTrue(all(0 < row[2] < .99 and row[3] == 1
                            for row in diagnostic))

    def test_algorithm_four_unique_intersection_helper(self):
        offline = [(0.0, 1.0), (2.0, 3.0), (5.0, 6.0)]
        self.assertIsNone(unique_offline_match(offline, (0.5, 2.5)))
        self.assertEqual(unique_offline_match(offline, (2.1, 2.9)), 1)
        self.assertIsNone(unique_offline_match(offline, (8.0, 9.0)))


if __name__ == "__main__":
    unittest.main()
