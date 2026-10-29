from __future__ import annotations

import hashlib
from types import SimpleNamespace
import unittest

import torch

from experiments.risk_safe.shadow_memory import GreedyBank
from experiments.calibration.memory_state import CalibrationGreedyState
from experiments.calibration.queried_anchor_identifiability import (
    AnchorScore, IdentifiabilityError, SECONDARY_OFFSETS, action_digest,
    defect_evidence_bin, distinct_group_count, leave_source_out,
    local_gain_table, select_primary_action,
)
from experiments.calibration.queried_anchor_identifiability_evaluator import (
    _future_best, _future_gains, _resolve_snapshot_threshold,
)
from experiments.calibration.queried_anchor_identifiability_runner import (
    _accumulate_excluded_rows, _n1_missing_defect_evidence,
    _reconstructed_prior_groups,
)


SNAPSHOT_SHA = "a" * 64


def group(name: str) -> str:
    return hashlib.sha256(name.encode("ascii")).hexdigest()


def anchor(name: str, label: int, score: float, *, version: int = 4,
           snapshot_sha: str = SNAPSHOT_SHA, policy: str = "normal_lso") -> AnchorScore:
    return AnchorScore(
        group_sha256=group(name), position=1, camera_id=1, label=label,
        score=score, memory_version=version,
        snapshot_memory_sha256=snapshot_sha,
        scoring_memory_sha256=hashlib.sha256((name + policy).encode()).hexdigest(),
        policy=policy)


class QueryAnchorIdentifiabilityTests(unittest.TestCase):
    def test_missing_class_is_unidentifiable_not_zero_gain(self):
        row = local_gain_table([anchor("n", 0, 0.2)], [], theta=0.5,
                               offsets=(-0.01, 0.0, 0.01), expected_version=4,
                               expected_snapshot_sha256=SNAPSHOT_SHA)
        self.assertEqual(row["status"], "TWO_CLASS_GAIN_UNIDENTIFIABLE")
        self.assertIsNone(row["gains"])
        self.assertEqual(select_primary_action(row)["action"], "ABSTAIN")

    def test_wrong_memory_version_or_digest_is_rejected(self):
        for row in (anchor("n", 0, 0.2, version=3),
                    anchor("n", 0, 0.2, snapshot_sha="b" * 64)):
            with self.assertRaises(IdentifiabilityError):
                local_gain_table([row], [anchor("d", 1, 0.8)], theta=0.5,
                                 offsets=(-0.01, 0.0, 0.01), expected_version=4,
                                 expected_snapshot_sha256=SNAPSHOT_SHA)

    def test_exact_positive_action_tie_abstains(self):
        normals = [anchor("n", 0, 0.505)]
        defects = [anchor("d", 1, 0.495)]
        result = local_gain_table(normals, defects, theta=0.5,
                                  offsets=(-0.01, 0.0, 0.01), expected_version=4,
                                  expected_snapshot_sha256=SNAPSHOT_SHA)
        self.assertEqual(result["gains"]["-0.010000"]["delta_j_vs_keep"], 1.0)
        self.assertEqual(result["gains"]["+0.010000"]["delta_j_vs_keep"], 1.0)
        self.assertEqual(select_primary_action(result), {
            "action": "ABSTAIN", "reason": "UNRESOLVED_POSITIVE_GAIN_TIE"})

    def test_no_useful_local_gain_selects_keep(self):
        normals = [anchor("n1", 0, 0.7), anchor("n2", 0, 0.8)]
        defects = [anchor("d1", 1, 0.2), anchor("d2", 1, 0.3)]
        result = local_gain_table(normals, defects, theta=0.5,
                                  offsets=(-0.01, 0.0, 0.01), expected_version=4,
                                  expected_snapshot_sha256=SNAPSHOT_SHA)
        self.assertEqual(select_primary_action(result)["action"], "KEEP")
        self.assertEqual(result["gains"]["+0.000000"]["delta_j_vs_keep"], 0.0)

    def test_positive_lower_gain_selected(self):
        normals = [anchor("n1", 0, 0.7), anchor("n2", 0, 0.8)]
        defects = [anchor("d1", 1, 0.495), anchor("d2", 1, 0.4)]
        result = local_gain_table(normals, defects, theta=0.5,
                                  offsets=(-0.01, 0.0, 0.01), expected_version=4,
                                  expected_snapshot_sha256=SNAPSHOT_SHA)
        self.assertEqual(select_primary_action(result)["action"], "LOWER")

    def test_duplicate_official_group_cannot_inflate_evidence(self):
        duplicate = anchor("same", 0, 0.2)
        with self.assertRaises(IdentifiabilityError):
            local_gain_table([duplicate, duplicate], [anchor("d", 1, 0.8)],
                             theta=0.5, offsets=(-0.01, 0.0, 0.01),
                             expected_version=4,
                             expected_snapshot_sha256=SNAPSHOT_SHA)
        self.assertEqual(distinct_group_count([group("same"), group("same"),
                                               group("other")]), 2)

    def test_all_layer_lso_preserves_snapshot_and_removes_all_source_rows(self):
        layers = (torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                  torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
        legacy = GreedyBank(layers, (256, 256), version=7)
        memory = CalibrationGreedyState(legacy, memory_content_version=4)
        source_rows = (("g", "h", "g"), ("g", "h"))
        original_digest = memory.sha256()
        lso, removed = leave_source_out(memory, source_rows, "g")
        self.assertEqual(removed, (2, 1))
        self.assertEqual(lso.version, 4)
        self.assertEqual(lso.legacy_greedy_version, 7)
        self.assertEqual([len(layer) for layer in lso.layers], [1, 1])
        self.assertEqual(memory.sha256(), original_digest)

    def test_lso_fails_closed_on_source_map_misalignment(self):
        memory = CalibrationGreedyState(
            GreedyBank((torch.ones((2, 2)),), (256,), version=0), 0)
        with self.assertRaises(IdentifiabilityError):
            leave_source_out(memory, (("only-one",),), "g")

    def test_lso_exclusion_counts_follow_native_layer_depth(self):
        totals = [0] * 5
        self.assertEqual(
            _accumulate_excluded_rows(totals, (1, 0, 1, 0, 2), 5),
            [1, 0, 1, 0, 2])
        with self.assertRaisesRegex(RuntimeError, "layer count"):
            _accumulate_excluded_rows([0] * 4, (1, 0, 1, 0, 2), 5)

    def test_n1_checks_all_early_audiojack_seed_position_states(self):
        rows = [{"category": "audiojack", "seed": seed,
                 "snapshot_position": position, "defect_groups": 0,
                 "primary": {"status": "TWO_CLASS_GAIN_UNIDENTIFIABLE"},
                 "selected_action": "ABSTAIN"}
                for seed in range(5) for position in (0, 400, 800)]
        result = _n1_missing_defect_evidence(rows)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["audiojack_t0_400_800_rows"], 15)
        self.assertEqual(result["expected_seed_position_rows"], 15)
        rows[-1]["selected_action"] = "KEEP"
        self.assertEqual(_n1_missing_defect_evidence(rows)["status"], "FAIL")

    def test_memory_source_key_is_distinct_from_public_group_digest(self):
        source_key = "private-metadata-sample-key"
        public_group_digest = group("same official group")
        events = [{"official_group_sha256": public_group_digest,
                   "support_seeds": {"0": {"newly_resident_source_group": True,
                                             "writer": {"appended": 1,
                                                        "replaced": 0}}}}]
        stream_rows = [SimpleNamespace(sample_key=source_key)]
        self.assertEqual(_reconstructed_prior_groups(events, 0, 1, stream_rows),
                         {source_key})
        self.assertNotEqual(source_key, public_group_digest)
        memory = CalibrationGreedyState(
            GreedyBank((torch.tensor([[1.0, 0.0], [0.0, 1.0]]),),
                       (256,), version=0), 0)
        lso, removed = leave_source_out(
            memory, ((source_key, public_group_digest),), source_key)
        self.assertEqual(removed, (1,))
        self.assertEqual(len(lso.layers[0]), 1)
        self.assertEqual(lso.layers[0][0].tolist(), [0.0, 1.0])

    def test_defect_evidence_bins_are_fixed(self):
        expected = {0: "0", 1: "1-2", 2: "1-2", 3: "3-7", 7: "3-7",
                    8: "8-13", 13: "8-13", 14: "14+", 50: "14+"}
        self.assertEqual({value: defect_evidence_bin(value) for value in expected}, expected)

    def test_action_digest_is_order_canonical(self):
        a = [{"category": "switch", "seed": 0, "snapshot_position": 400,
              "selected_action": "KEEP"},
             {"category": "audiojack", "seed": 1, "snapshot_position": 0,
              "selected_action": "ABSTAIN"}]
        self.assertEqual(action_digest(a), action_digest(list(reversed(a))))
        self.assertEqual(SECONDARY_OFFSETS, (-0.005, 0.0, 0.005))

    def test_future_metric_preserves_one_class_as_undefined(self):
        one_class = _future_gains(
            torch.tensor([0, 0]).numpy(), torch.tensor([0.1, 0.9]).numpy(),
            0.5, (-0.01, 0.0, 0.01))
        self.assertEqual(one_class["base"]["status"], "UNDEFINED_ONE_CLASS")
        self.assertIsNone(one_class["gains"]["+0.010000"]["delta_j_vs_keep"])
        self.assertFalse(_future_best(one_class["gains"])["unique"])

    def test_future_best_requires_a_unique_exact_maximum(self):
        tied = {"-0.010000": {"delta_j_vs_keep": 0.0},
                "+0.000000": {"delta_j_vs_keep": 0.0},
                "+0.010000": {"delta_j_vs_keep": -1.0}}
        self.assertFalse(_future_best(tied)["unique"])

    def test_blocked_state_with_null_recorded_threshold_remains_blocked(self):
        self.assertEqual(
            _resolve_snapshot_threshold(0.5, "BLOCKED", None),
            (0.5, "BLOCKED_NATIVE_SCORE"))
        self.assertEqual(
            _resolve_snapshot_threshold(None, "BLOCKED", None),
            (None, "BLOCKED_B3_THRESHOLD"))
        self.assertEqual(_resolve_snapshot_threshold(0.5, "PASS", 0.5),
                         (0.5, "PASS"))
        with self.assertRaises(RuntimeError):
            _resolve_snapshot_threshold(0.5, "PASS", 0.4)


if __name__ == "__main__":
    unittest.main()
