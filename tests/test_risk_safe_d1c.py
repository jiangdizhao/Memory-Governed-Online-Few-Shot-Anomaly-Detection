"""D1-C CPU-only mechanistic safety tests for counterfactual admission.

Synthetic features demonstrate causal mechanics, NOT VisionAD performance.
"""
from __future__ import annotations

import math
import unittest
import torch

from experiments.risk_safe.anchor_store import (
    AnchorLedger, AnchorProtocolError, VerifiedImage)
from experiments.risk_safe.shadow_memory import (
    GreedyBank, CosinePatchScorer, ShadowError, native_greedy_slot)
from experiments.risk_safe.admission import (
    AdmissionProtocolError, CDARG, CDARGConfig,
    heldout_counterfactual_erosion)


def row(x, y):
    return torch.tensor([[float(x), float(y)]], dtype=torch.float32)


def verified(step, label="normal", cam=1):
    return VerifiedImage(query_id=f"q{step}", image_id=f"im{step}",
                         sample_key=f"sample{step}", camera_id=cam,
                         label=label, query_decision_step=step,
                         label_revealed_step=step)


def ledger_with_anchors(normal, defect=None):
    ledger = AnchorLedger(patches_per_image=8)
    ledger.record(verified(1, "normal"), [normal])
    if defect is not None:
        ledger.record(verified(2, "defect"), [defect],
                      anomaly_patch_scores=[torch.tensor([1.0])])
    return ledger


def gate(static=None, **overrides):
    values = {"epsilon_defect": 0.02, "lambda_defect": 1.0,
              "tau_utility": 0.0, "defect_margin": 0.05,
              "sigma_normal": 1.0, "sigma_floor": 0.01}
    values.update(overrides)
    return CDARG(CDARGConfig(**values),
                 CosinePatchScorer([static if static is not None else row(1, 0)]))


class ShadowGreedyNativeParity(unittest.TestCase):
    def test_append_until_capacity(self):
        b = GreedyBank([row(1, 0)], [3])
        original = b.sha256()
        shadow, audit = b.proposed([torch.tensor([[0., 1.], [-1., 0.]])])
        self.assertEqual((audit.appended, audit.replaced, audit.skipped), (2, 0, 0))
        self.assertEqual(len(shadow.layers[0]), 3)
        self.assertEqual(len(b.layers[0]), 1)
        self.assertEqual(b.version, 0)
        self.assertEqual(b.sha256(), original)
        self.assertNotEqual(b.layers[0].data_ptr(), shadow.layers[0].data_ptr())
        b.commit(shadow, expected_version=0, expected_sha256=original)
        self.assertEqual(b.version, 1)
        self.assertEqual(len(b.layers[0]), 3)

    def test_full_bank_replacement_exact_native_v02_slot(self):
        base = torch.tensor([[1., 0.], [.99, .1]], dtype=torch.float32)
        candidate = row(0, 1)
        b = GreedyBank([base], [2])
        class Codebook:
            def __init__(self, p):
                self.protos = p
        expected_slot = native_greedy_slot(Codebook(base.clone()), candidate)
        self.assertIsNotNone(expected_slot)
        shadow, audit = b.proposed([candidate])
        self.assertEqual((audit.appended, audit.replaced), (0, 1))
        self.assertEqual(len(shadow.layers[0]), 2)
        torch.testing.assert_close(
            shadow.layers[0][expected_slot],
            torch.nn.functional.normalize(candidate, dim=1)[0])

    def test_full_bank_redundant_patch_no_op(self):
        base = torch.tensor([[1., 0.], [0., 1.]])
        b = GreedyBank([base], [2])
        shadow, audit = b.proposed([row(1, 0)])
        self.assertEqual(audit.skipped, 1)
        self.assertEqual(audit.replaced, 0)
        self.assertEqual(shadow.version, b.version)
        self.assertEqual(shadow.sha256(), b.sha256())

    def test_shadow_evaluations_repeat_identically(self):
        b = GreedyBank([row(1, 0)], [2])
        s1, a1 = b.proposed([row(0, 1)])
        s2, a2 = b.proposed([row(0, 1)])
        self.assertEqual(s1.sha256(), s2.sha256())
        self.assertEqual(a1, a2)

    def test_reject_nonfinite_invalid_shape_and_capacity(self):
        b = GreedyBank([row(1, 0)], [2])
        with self.assertRaises(ShadowError):
            b.proposed([torch.tensor([[float("nan"), 0.]])])
        with self.assertRaises(ShadowError):
            b.proposed([torch.randn(1, 3)])
        with self.assertRaises(ShadowError):
            GreedyBank([row(1, 0)], [0])

    def test_commit_fails_closed_on_live_intervening_change(self):
        b = GreedyBank([row(1, 0)], [3])
        original = b.sha256()
        shadow, _ = b.proposed([row(0, 1)])
        b.layers[0][0] = torch.tensor([0.3, 0.3])
        with self.assertRaises(ShadowError):
            b.commit(shadow, expected_version=0, expected_sha256=original)

    def test_copied_shadow_mutation_cannot_touch_live(self):
        b = GreedyBank([row(1, 0)], [3])
        before = b.sha256()
        s = b.clone()
        s.layers[0][0, :] = 0
        self.assertEqual(before, b.sha256())

    def test_multiple_layers_update_same_cap_rule(self):
        b = GreedyBank([row(1, 0), row(0, 1)], [2, 3])
        s, audit = b.proposed([row(0, 1), row(1, 0)])
        self.assertEqual(audit.appended, 2)
        self.assertEqual([len(x) for x in s.layers], [2, 2])

    def test_exact_cosine_patch_scores(self):
        b = GreedyBank([row(1, 0)], [2])
        s = CosinePatchScorer([row(1, 0)])
        self.assertAlmostEqual(s([row(-1, 0)], b), 2.0, places=5)
        shadow, _ = b.proposed([row(-1, 0)])
        self.assertAlmostEqual(s([row(-1, 0)], shadow), 0.0, places=5)


class VerifiedFeedbackLedgerTests(unittest.TestCase):
    def test_defect_requires_model_patch_ranking_not_ground_truth_mask(self):
        ledger = AnchorLedger(patches_per_image=2)
        feats = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        scores = torch.tensor([0.3, 0.9, 0.8])
        anchor = ledger.record(verified(1, "defect"), [feats],
                               anomaly_patch_scores=[scores])
        self.assertEqual(anchor.selected_patch_indices, ((1, 2),))
        self.assertEqual(anchor.features[0].shape, (2, 2))
        self.assertEqual(len(ledger.prior(1)[1]), 0)
        self.assertEqual(len(ledger.prior(2)[1]), 1)

    def test_reject_unqueried_and_label_before_query(self):
        with self.assertRaises(AnchorProtocolError):
            VerifiedImage("q", "img", "s", 1, "normal", 2, 1)
        with self.assertRaises(AnchorProtocolError):
            VerifiedImage("q", "img", "s", 1, "normal", 1, 1, queried=False)

    def test_one_image_query_does_not_label_sister_cameras(self):
        ledger = AnchorLedger()
        ledger.record(verified(1, "normal", cam=1), [row(1, 0)])
        self.assertEqual(ledger.counts["verified_query_events"], 1)
        self.assertEqual(len(ledger.prior(2)[0]), 1)
        self.assertEqual(ledger.prior(2)[0][0].event.camera_id, 1)

    def test_same_event_not_historical_anchor(self):
        ledger = AnchorLedger()
        ledger.record(verified(3, "normal"), [row(0, 1)])
        self.assertEqual(len(ledger.prior(3)[0]), 0)
        self.assertEqual(len(ledger.prior(4)[0]), 1)

    def test_duplicate_query_and_duplicate_image_forbidden(self):
        ledger = AnchorLedger()
        ledger.record(verified(1), [row(1, 0)])
        with self.assertRaises(AnchorProtocolError):
            ledger.record(verified(1), [row(1, 0)])
        other = VerifiedImage("q2", "im1", "sample1", 2, "normal", 2, 2)
        with self.assertRaises(AnchorProtocolError):
            ledger.record(other, [row(1, 0)])

    def test_defect_requires_prelabel_score(self):
        with self.assertRaises(AnchorProtocolError):
            AnchorLedger().record(verified(1, "defect"), [row(1, 0)])

    def test_anchor_features_are_copied(self):
        x = row(1, 0)
        anchor = AnchorLedger().record(verified(1), [x])
        x[0, 0] = 17
        self.assertEqual(anchor.features[0][0, 0].item(), 1.0)

    def test_capacity_fifo_only_past_evidence(self):
        ledger = AnchorLedger(max_normal_images=2)
        for i in range(1, 4):
            ledger.record(verified(i), [row(1, 0)])
        self.assertEqual(len(ledger.prior(4)[0]), 2)
        self.assertEqual([a.event.query_id for a in ledger.prior(4)[0]],
                         ["q2", "q3"])


class CDARGCausalRiskTests(unittest.TestCase):
    def setUp(self):
        self.bank = GreedyBank([row(1, 0)], [3])
        self.scorer = CosinePatchScorer([row(1, 0)])

    def test_verified_normal_harmful_to_defect_must_be_rejected(self):
        ledger = ledger_with_anchors(row(-1, 0), row(-0.96, 0.28))
        before = self.bank.sha256()
        control = gate()
        proposal = control.evaluate(
            current_verified_normal=verified(3), step=3,
            bank=self.bank, ledger=ledger,
            candidate_patches=[row(-1, 0)], theta=0.5)
        self.assertEqual(proposal.decision, "reject")
        self.assertTrue(proposal.defect_margin_crossing)
        self.assertGreater(proposal.normal_utility, 0)
        self.assertGreater(proposal.defect_risk, 0)
        self.assertFalse(control.commit_if_accepted(proposal, self.bank))
        self.assertEqual(self.bank.sha256(), before)
        self.assertEqual(self.bank.version, 0)

    def test_benign_candidate_can_admit_and_improve_normal(self):
        ledger = ledger_with_anchors(row(0, 1), row(0.8, -0.6))
        proposal = gate().evaluate(
            current_verified_normal=verified(3), step=3,
            bank=self.bank, ledger=ledger,
            candidate_patches=[row(0, 1)], theta=0.1)
        self.assertEqual(proposal.decision, "admit")
        self.assertAlmostEqual(proposal.defect_risk, 0, places=5)
        self.assertGreater(proposal.normal_utility, 0)
        self.assertFalse(proposal.defect_margin_crossing)
        self.assertTrue(CDARG.commit_if_accepted(proposal, self.bank))
        self.assertEqual(self.bank.version, 1)
        self.assertEqual(len(self.bank.layers[0]), 2)
        with self.assertRaises(AdmissionProtocolError):
            CDARG.commit_if_accepted(proposal, self.bank)

    def test_zero_defect_anchor_risk_unavailable_not_zero(self):
        ledger = ledger_with_anchors(row(0, 1), defect=None)
        before = self.bank.sha256()
        proposal = gate().evaluate(
            current_verified_normal=verified(3), step=3,
            bank=self.bank, ledger=ledger,
            candidate_patches=[row(0, 1)], theta=0.1)
        self.assertEqual(proposal.reason, "defect_risk_unavailable")
        self.assertIsNone(proposal.defect_risk)
        self.assertIsNone(proposal.defect_margin_crossing)
        self.assertFalse(CDARG.commit_if_accepted(proposal, self.bank))
        self.assertEqual(self.bank.sha256(), before)

    def test_without_normal_anchors_rejects_unavailable_utility(self):
        ledger = AnchorLedger()
        ledger.record(verified(1, "defect"), [row(0.8, -0.6)],
                      anomaly_patch_scores=[torch.tensor([0.8])])
        proposal = gate().evaluate(
            current_verified_normal=verified(3), step=3,
            bank=self.bank, ledger=ledger,
            candidate_patches=[row(0, 1)], theta=0.1)
        self.assertEqual(proposal.reason, "normal_utility_unavailable")

    def test_scorer_receives_same_theta_without_sidecar_mutation(self):
        ledger = ledger_with_anchors(row(0, 1), row(0.8, -0.6))
        before = self.bank.sha256()
        control = gate()
        proposal = control.evaluate(
            current_verified_normal=verified(3), step=3,
            bank=self.bank, ledger=ledger,
            candidate_patches=[row(0, 1)], theta=0.1)
        self.assertEqual(proposal.theta_snapshot, 0.1)
        self.assertEqual(self.bank.sha256(), before)
        self.assertEqual(proposal.greedy_audit.candidate_patches, 1)

    def test_future_anchor_not_read(self):
        ledger = ledger_with_anchors(row(0, 1))
        # Future defect query exists in trusted store but is not yet visible.
        ledger.record(verified(10, "defect"), [row(0.8, -0.6)],
                      anomaly_patch_scores=[torch.tensor([0.8])])
        proposal = gate().evaluate(
            current_verified_normal=verified(3), step=3,
            bank=self.bank, ledger=ledger,
            candidate_patches=[row(0, 1)], theta=0.1)
        self.assertEqual(proposal.reason, "defect_risk_unavailable")

    def test_gates_explicit_epsilon_and_objective(self):
        ledger = ledger_with_anchors(row(0, 1), row(0.8, -0.6))
        proposal = gate(tau_utility=100.0).evaluate(
            current_verified_normal=verified(3), step=3,
            bank=self.bank, ledger=ledger,
            candidate_patches=[row(0, 1)], theta=0.1)
        self.assertEqual(proposal.reason, "utility_risk_tradeoff_failed")

    def test_verified_defect_cannot_be_proposed_normal(self):
        with self.assertRaises(AdmissionProtocolError):
            gate().evaluate(current_verified_normal=verified(3, "defect"),
                            step=3, bank=self.bank,
                            ledger=AnchorLedger(), candidate_patches=[row(0, 1)],
                            theta=0.1)

    def test_no_op_greedy_candidate_is_not_counted_as_admission(self):
        full = GreedyBank([torch.tensor([[1., 0.], [0., 1.]])], [2])
        ledger = ledger_with_anchors(row(0, 1), row(-1, 0))
        proposal = gate().evaluate(
            current_verified_normal=verified(3), step=3,
            bank=full, ledger=ledger,
            candidate_patches=[row(1, 0)], theta=0.1)
        self.assertEqual(proposal.reason, "no_memory_delta")
        self.assertEqual(proposal.decision, "reject")

    def test_heldout_probe_only_offline_and_never_changes_gate(self):
        ledger = ledger_with_anchors(row(-1, 0), row(-0.96, 0.28))
        proposal = gate().evaluate(
            current_verified_normal=verified(3), step=3,
            bank=self.bank, ledger=ledger,
            candidate_patches=[row(-1, 0)], theta=0.5)
        independent = heldout_counterfactual_erosion(
            [[row(-0.96, 0.28)]], self.bank, proposal.shadow, self.scorer)
        self.assertGreater(independent["mean_positive_score_erosion"], 0)
        self.assertEqual(proposal.decision, "reject")

    def test_config_requires_explicit_valid_finite_values(self):
        with self.assertRaises(AdmissionProtocolError):
            CDARGConfig(float("nan"), 1, 0, .1, 1, .01)
        with self.assertRaises(AdmissionProtocolError):
            CDARGConfig(.1, 1, 0, .1, 1, 0)


class D1CSyntheticSmokeTests(unittest.TestCase):
    def test_four_scenarios_are_deterministic_and_causally_auditable(self):
        from experiments.risk_safe.d1c_cpu_smoke import build_payloads, OUTPUTS
        first = build_payloads()[OUTPUTS[0]]
        second = build_payloads()[OUTPUTS[0]]
        self.assertEqual(first, second)
        self.assertTrue(first["pass"])
        self.assertFalse(first["native_v22_image_scorer_parity_tested"])
        self.assertEqual(first["N_cases"], 4)
        self.assertEqual(
            {x["scenario"] for x in first["scenarios"]},
            {"harmful_defect_crossing", "benign_normal_adaptation",
             "no_defect_anchor", "full_capacity_harmful_replacement"})
        self.assertTrue(all(x["heldout_evaluation_only"]["offline_only_not_visible_to_gate"]
                            for x in first["scenarios"]))

    def test_synthetic_output_idempotence_guards(self):
        from pathlib import Path
        import tempfile
        from experiments.risk_safe import preflight as pf
        from experiments.risk_safe.d1c_cpu_smoke import build_payloads, OUTPUTS
        payload = build_payloads()[OUTPUTS[0]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / OUTPUTS[0]
            self.assertEqual(pf.write_json_idempotent(path, payload), "created")
            self.assertEqual(pf.write_json_idempotent(path, payload), "reused")
            self.assertEqual(pf.verify_json_output(path, payload), "verified")
            with self.assertRaises(pf.OutputMismatch):
                pf.verify_json_output(path, {"mutated": True})


if __name__ == "__main__":
    unittest.main()
