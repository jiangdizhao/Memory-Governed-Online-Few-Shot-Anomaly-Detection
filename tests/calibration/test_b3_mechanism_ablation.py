from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import unittest

from experiments.calibration.stage1a_b3_mechanism_ablation import (
    ARMS,
    AblationError,
    B3LagController,
    C64Reference,
    TrajectoryEvent,
    VersionedC64History,
    higher_quantile,
    reference_from_evidence,
    replay_arm,
    replay_matched_arms,
    verify_a0,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def history_for(q_values: list[float]) -> VersionedC64History:
    group = digest("fixed C64 identities")
    refs = []
    for version, q90 in enumerate(q_values):
        refs.append(C64Reference.from_scores(
            version, digest(f"memory-{version}"), [q90] * 64, group))
    return VersionedC64History(refs)


def event(position: int, before: int, after: int, score: float = 0.5,
          sha_before: str | None = None, sha_after: str | None = None) -> TrajectoryEvent:
    return TrajectoryEvent(
        position, score, before, sha_before or digest(f"memory-{before}"),
        after, sha_after or digest(f"memory-{after}"))


def changed_events(q_values: list[float]) -> tuple[TrajectoryEvent, ...]:
    rows = [event(0, 0, 0, score=0.5)]
    position = 1
    for version in range(1, len(q_values)):
        rows.append(event(position, version - 1, version, score=0.5))
        position += 1
        rows.append(event(position, version, version, score=0.5))
        position += 1
    return tuple(rows)


class TestB3MechanismAblation(unittest.TestCase):
    def test_higher_quantile_ties_and_nonfinite_values(self):
        self.assertEqual(higher_quantile([0, 1, 1, 1, 2], 0.5), 1)
        self.assertEqual(higher_quantile([0.0] * 58 + [1.0] * 6, 0.9), 0.0)
        self.assertEqual(higher_quantile([float(i) for i in range(10)], 0.5), 5.0)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            higher_quantile([0.0, float("nan")], 0.9)
        with self.assertRaises(ValueError):
            higher_quantile([], 0.9)

    def test_registered_lag_versions_and_version_zero_clamp(self):
        qs = [0.50, 0.70, 0.10, 0.90, 0.20, 0.80, 0.30]
        arms = replay_matched_arms(changed_events(qs), history_for(qs))
        self.assertEqual(dict(ARMS), {"A2": 0, "A3": 1, "A4": 5})
        refs = {arm: [u.reference_version for u in replay.updates]
                for arm, replay in arms.items()}
        self.assertEqual(refs["A2"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(refs["A3"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(refs["A4"], [0, 0, 0, 0, 0, 1])
        self.assertEqual(len({len(replay.updates) for replay in arms.values()}), 1)

    def test_initial_theta_has_no_artificial_first_event_update(self):
        hist = history_for([0.5, 0.9])
        with self.assertRaisesRegex(AblationError, "Delta is frozen at 0.01"):
            B3LagController("A2", hist, 0.5, delta=0.02)
        trace = replay_arm((event(0, 0, 0, score=0.5), event(1, 0, 1)), hist, "A2")
        self.assertEqual(trace.threshold_pre[0], 0.5)
        self.assertEqual(trace.threshold_post[0], 0.5)
        self.assertEqual(len(trace.updates), 1)

    def test_long_no_change_run_and_noop_writes_hold_threshold(self):
        hist = history_for([0.5])
        events = tuple(event(t, 0, 0, score=0.6) for t in range(600))
        trace = replay_arm(events, hist, "A4")
        self.assertEqual(len(trace.updates), 0)
        self.assertTrue(all(value == 0.5 for value in trace.threshold_pre + trace.threshold_post))
        with self.assertRaisesRegex(AblationError, "No-op"):
            event(600, 0, 0, sha_before=digest("memory-0"),
                  sha_after=digest("different"))

    def test_multiple_changes_clip_to_delta_and_record_reference_state(self):
        hist = history_for([0.5, 0.9, 0.1])
        trace = replay_arm((event(0, 0, 1), event(1, 1, 2)), hist, "A2")
        self.assertEqual(trace.threshold_post, (0.51, 0.50))
        self.assertAlmostEqual(trace.updates[0].applied_step, 0.01)
        self.assertAlmostEqual(trace.updates[1].applied_step, -0.01)
        self.assertEqual([u.memory_version for u in trace.updates], [1, 2])
        self.assertEqual([u.reference_version for u in trace.updates], [1, 2])
        self.assertTrue(all(abs(u.applied_step) <= 0.01 + 2 * math.ulp(0.51)
                            for u in trace.updates))

    def test_predict_before_update_and_score_greater_equal_boundary(self):
        hist = history_for([0.5, 0.9])
        trace = replay_arm((event(0, 0, 1, score=0.5),
                            event(1, 1, 1, score=0.505)), hist, "A2")
        self.assertEqual(trace.predictions, (1, 0))
        self.assertEqual(trace.threshold_pre, (0.5, 0.51))
        self.assertEqual(trace.threshold_post, (0.51, 0.51))
        controller = B3LagController("A2", hist, 0.5)
        first_event = event(0, 0, 1, score=0.5)
        with self.assertRaisesRegex(AblationError, "preceding prediction"):
            controller.after_event(first_event)
        controller.predict(first_event.score, first_event.memory_version_before,
                           first_event.memory_sha256_before)
        controller.after_event(first_event)

    def test_missing_or_mismatched_history_fails_closed(self):
        group = digest("fixed C64 identities")
        refs = [
            C64Reference.from_scores(0, digest("memory-0"), [0.5] * 64, group),
            C64Reference.from_scores(2, digest("memory-2"), [0.7] * 64, group),
        ]
        with self.assertRaisesRegex(AblationError, "complete from version 0"):
            VersionedC64History(refs)
        hist = history_for([0.5, 0.7])
        controller = B3LagController("A2", hist, 0.5)
        mismatched = event(0, 0, 1, sha_after=digest("wrong-memory-1"))
        controller.predict(mismatched.score, mismatched.memory_version_before,
                           mismatched.memory_sha256_before)
        with self.assertRaisesRegex(AblationError, "digest mismatch"):
            controller.after_event(mismatched)

    def test_frozen_q90_and_vector_digest_mismatches_fail_closed(self):
        group = digest("fixed C64 identities")
        scores = [0.0] * 58 + [1.0] * 6
        vector_sha = hashlib.sha256(json.dumps(
            scores, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
        ref = reference_from_evidence(
            0, digest("memory-0"), scores, group, 0.0, vector_sha)
        self.assertEqual(ref.q90, 0.0)
        with self.assertRaisesRegex(AblationError, "Q90_EVIDENCE_MISMATCH"):
            reference_from_evidence(
                0, digest("memory-0"), scores, group, 1.0, vector_sha)
        with self.assertRaisesRegex(AblationError, "C64_VECTOR_EVIDENCE_MISMATCH"):
            reference_from_evidence(
                0, digest("memory-0"), scores, group, 0.0, digest("wrong-vector"))

    def test_no_future_version_evidence_access(self):
        hist = history_for([0.5, 0.6, 0.7])
        with self.assertRaisesRegex(AblationError, "FUTURE_EVIDENCE_ACCESS"):
            hist.lookup(2, as_of_version=1)
        hist.lookup_log.clear()
        trace = replay_arm((event(0, 0, 1), event(1, 1, 2)), hist, "A3")
        self.assertTrue(all(version <= as_of for version, as_of in hist.lookup_log))
        self.assertTrue(all(u.reference_version <= u.memory_version for u in trace.updates))

    def test_deterministic_checkpoint_restore_matches_uninterrupted_replay(self):
        hist = history_for([0.5, 0.9, 0.2, 0.8])
        events = (event(0, 0, 0), event(1, 0, 1), event(2, 1, 1),
                  event(3, 1, 2), event(4, 2, 3))
        controller = B3LagController("A4", hist, 0.5)
        uninterrupted_pre = []
        for current in events[:3]:
            uninterrupted_pre.append(controller.theta)
            controller.predict(current.score, current.memory_version_before,
                               current.memory_sha256_before)
            controller.after_event(current)
        checkpoint = controller.state_dict()
        restored = B3LagController.from_state_dict(checkpoint, hist)
        self.assertEqual(restored.state_dict(), checkpoint)
        for current in events[3:]:
            uninterrupted_pre.append(controller.theta)
            self.assertEqual(restored.theta, controller.theta)
            self.assertEqual(
                restored.predict(current.score, current.memory_version_before,
                                 current.memory_sha256_before),
                controller.predict(current.score, current.memory_version_before,
                                   current.memory_sha256_before))
            restored.after_event(current)
            controller.after_event(current)
        self.assertEqual(restored.state_dict(), controller.state_dict())
        self.assertEqual(tuple(uninterrupted_pre),
                         replay_arm(events, hist, "A4").threshold_pre)

    def test_replay_matches_independent_reference_recurrence_exactly(self):
        qs = [0.5, 0.91, 0.12, 0.88, 0.2, 0.79, 0.35]
        hist, events = history_for(qs), changed_events(qs)
        for arm, lag in (("A2", 0), ("A3", 1), ("A4", 5)):
            theta = qs[0]
            expected_pre, expected_post, expected_predictions = [], [], []
            for row in events:
                expected_pre.append(theta)
                expected_predictions.append(int(row.score >= theta))
                if row.memory_version_after != row.memory_version_before:
                    ref_version = max(0, row.memory_version_after - lag)
                    target = qs[ref_version]
                    theta = min(max(target, theta - 0.01), theta + 0.01)
                expected_post.append(theta)
            actual = replay_arm(events, hist, arm)
            self.assertEqual(actual.threshold_pre, tuple(expected_pre))
            self.assertEqual(actual.threshold_post, tuple(expected_post))
            self.assertEqual(actual.predictions, tuple(expected_predictions))

    def test_checkpoint_rejects_wrong_arm_or_history(self):
        hist = history_for([0.5, 0.7])
        controller = B3LagController("A3", hist, 0.5)
        first = event(0, 0, 1)
        controller.predict(first.score, first.memory_version_before,
                           first.memory_sha256_before)
        controller.after_event(first)
        state = controller.state_dict()
        changed = dict(state, arm_id="A4")
        with self.assertRaisesRegex(AblationError, "configuration"):
            B3LagController.from_state_dict(changed, hist)
        other = history_for([0.5, 0.7])
        other._references = (
            other.references[0],
            C64Reference.from_scores(1, digest("other-memory-1"), [0.7] * 64,
                                     digest("different C64 identities")))
        with self.assertRaises(AblationError):
            B3LagController.from_state_dict(state, other)

    def test_historical_b3_source_file_remains_unchanged(self):
        root = Path(__file__).resolve().parents[2]
        tracked = subprocess.check_output(
            ["git", "show", "HEAD:experiments/calibration/controllers.py"],
            cwd=root)
        self.assertEqual(
            (root / "experiments/calibration/controllers.py").read_text(encoding="utf-8"),
            tracked.decode("utf-8").replace("\r\n", "\n"))

    def test_a0_and_historical_a2_parity_from_original_frozen_artifacts(self):
        frozen_root = os.environ.get("B3_A0_FROZEN_ROOT")
        if not frozen_root:
            self.skipTest("Set B3_A0_FROZEN_ROOT to the original read-only checkout")
        result = verify_a0(Path(frozen_root), Path(__file__).resolve().parents[2])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["summary"], {
            "trajectory_count": 20, "events": 76475, "memory_updates": 1190,
            "pre_threshold_mismatches": 0, "post_threshold_mismatches": 0,
            "prediction_mismatches": 0, "event_identity_mismatches": 0,
        })
        self.assertTrue(all(row["status"] == "PASS" and row["q90_history_complete"]
                            for row in result["feasibility_matrix"]))


if __name__ == "__main__":
    unittest.main()
