"""R6 audit-only, strict R5 parity, and label-separated stress-stream tests."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import e02_r6_sidecar_stability as r6


class DummyBalanced:
    """Deterministic miniature controller exercising the real audit wrapper."""
    def __init__(self):
        self.cfg = SimpleNamespace(
            accept_margin=0.001, balanced_guard_delta_fn_up=0.05,
            balanced_up_defect_slack=0.02, balanced_guard_delta_fp_down=0.2,
            balanced_guard_delta_fp_normal=0.01,
            balanced_guard_delta_fn_normal=0.01)
        self.theta_current = 0.5
        self.balanced_cooldown_remaining = 0
        self.labeled_since_update = 0
        self.bufN = [0.1]
        self.bufA = [0.9]
        self.recent_labeled = []
        self.last_eval = {}
        self.last_mode = "none"
        self.last_accept = False
        self.calls = 0

    def theta(self):
        return self.theta_current

    def _candidate_thetas_balanced(self, mode, anchors):
        return [self.theta_current, self.theta_current - 0.1]

    def _balanced_rates(self, theta):
        self.calls += 1
        return {"FPR": 0., "FNR": 1. if theta >= 0.5 else 0.}

    def _balanced_objective(self, theta, risk):
        self.calls += 1
        return risk["FNR"] + 0.01 * abs(theta - self.theta_current)

    def _maybe_balanced_update_theta(self, force_hint=None):
        if self.balanced_cooldown_remaining and force_hint is None:
            self.balanced_cooldown_remaining -= 1
            self.last_eval = {"status": "balanced_rescue_cooldown"}
            return
        current = self._balanced_rates(self.theta_current)
        start = self._balanced_objective(self.theta_current, current)
        candidates = self._candidate_thetas_balanced(
            "fast_downward_rescue", {"theta_D": 0.3})
        chosen = self.theta_current
        best = start
        for cand in candidates:
            risk = self._balanced_rates(cand)
            if risk["FPR"] > current["FPR"] + self.cfg.balanced_guard_delta_fp_down:
                continue
            obj = self._balanced_objective(cand, risk)
            if obj < best - self.cfg.accept_margin:
                best, chosen = obj, cand
        self.last_eval = {"status": "balanced_evaluated",
                          "mode": "fast_downward_rescue",
                          "current_rates": current, "current_obj": start,
                          "current_theta": self.theta_current,
                          "anchors": {"theta_D": 0.3}}
        self.theta_current = chosen
        self.last_accept = chosen != 0.5

    def observe_labeled(self, score, label):
        self.recent_labeled.append((score, label))
        self._maybe_balanced_update_theta(force_hint="downward" if label else None)


class TestR6Contracts(unittest.TestCase):
    def test_offline_only_order_keeps_same_images(self):
        xs = [SimpleNamespace(path=str(i), is_good=(i % 4 != 0))
              for i in range(19)]
        for mode in ("native", "normal_first", "defect_first", "alternating_blocks"):
            got = r6.order_stream(xs, mode, block_size=3)
            self.assertCountEqual([x.path for x in xs], [x.path for x in got])
        self.assertEqual(r6.order_stream(xs, "native"), xs)
        self.assertTrue(all(x.is_good for x in r6.order_stream(
            xs, "normal_first")[:14]))
        self.assertTrue(all(not x.is_good for x in r6.order_stream(
            xs, "defect_first")[:5]))
        with self.assertRaises(ValueError):
            r6.order_stream(xs, "unknown")

    def test_windows_use_null_for_unobserved_class(self):
        rows = []
        for label in (0, 0, 1, 1):
            row = {"y_eval_only": label, "queried": 0}
            for arm in ("frozen", "guarded", "balanced_rescue", "mistake_feedback"):
                row[f"{arm}_pred"] = 0
            rows.append(row)
        windows = r6.class_windows(rows, width=2)
        self.assertEqual(len(windows), 2)
        self.assertIsNone(windows[0]["balanced_rescue"]["fnr"])
        self.assertIsNone(windows[0]["balanced_rescue"]["balanced_accuracy"])
        self.assertIsNone(windows[1]["balanced_rescue"]["fpr"])
        self.assertEqual(windows[0]["normal_count"], 2)

    def test_instrumentation_does_not_change_controller(self):
        raw, monitored = DummyBalanced(), DummyBalanced()
        audit = r6.SidecarAudit(monitored)
        audit.step = 1
        raw.observe_labeled(0.4, 1)
        monitored.observe_labeled(0.4, 1)
        self.assertEqual(raw.theta(), monitored.theta())
        self.assertEqual(raw.calls, monitored.calls)
        self.assertEqual(len(audit.events), 1)
        self.assertEqual(audit.events[0]["outcome"], "accepted")
        self.assertEqual(audit.events[0]["force_hint"], "downward")
        self.assertGreater(len(audit.events[0]["candidate_audit"]), 0)
        raw.balanced_cooldown_remaining = 2
        monitored.balanced_cooldown_remaining = 2
        raw.observe_labeled(0.2, 0)
        monitored.observe_labeled(0.2, 0)
        self.assertEqual(raw.balanced_cooldown_remaining,
                         monitored.balanced_cooldown_remaining)
        self.assertEqual(audit.events[-1]["outcome"], "cooldown_blocked")

    def test_parity_compares_every_step_and_rejects_drift(self):
        row = {"path": "one", "query_reason": "none", "budget_reason": "native",
               "score_region": "boundary", "region_selection": "native",
               "policy": "D"}
        row.update({key: 1 for key in r6.PARITY_INTS})
        row.update({key: 0.25 for key in r6.PARITY_FLOATS})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "baseline.csv"
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            self.assertTrue(r6.compare_reference_csv([row], path)["passed"])
            different = dict(row)
            different["balanced_rescue_theta"] = 0.24
            with self.assertRaisesRegex(AssertionError, "balanced_rescue_theta"):
                r6.compare_reference_csv([different], path)

    def test_r6_accepts_only_native_r6_as_cross_class_reference(self):
        path = ROOT / "outputs" / "e02_r5_d_audiojack_full_seed0" / "r5_d_summary.json"
        old = json.loads(path.read_text(encoding="utf-8"))
        now = json.loads(json.dumps(old))
        self.assertTrue(r6.check_summary_reference(old, now, native=True)["passed"])
        r6_as_reference = dict(old)
        r6_as_reference["schema"] = "e02.r6.sidecar.audit.v1"
        r6_as_reference["stream_order"] = "native"
        self.assertTrue(r6.check_summary_reference(r6_as_reference, now, native=False)["passed"])
        r6_as_reference["stream_order"] = "defect_first"
        with self.assertRaisesRegex(AssertionError, "Stress trajectory"):
            r6.check_summary_reference(r6_as_reference, now, native=False)


if __name__ == "__main__":
    unittest.main()
