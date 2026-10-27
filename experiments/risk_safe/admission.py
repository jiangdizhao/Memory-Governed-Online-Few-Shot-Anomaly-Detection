"""D1-C Counterfactual Dual-Anchor Risk Gate (CPU mechanism implementation).

Reference: project's "Risk-Safe Memory Admission for v22: A Deployment-Oriented
Research Strategy", Priority A. The scoring callback must score identical
historical feature anchors under the live and *exactly simulated* shadow bank.
This module does not train thresholds, alter the Query Controller or Sidecar,
or see evaluation-probe labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional, Sequence, Tuple
import torch

from .anchor_store import Anchor, AnchorLedger, VerifiedImage
from .shadow_memory import GreedyAudit, GreedyBank, ShadowError


class AdmissionProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class CDARGConfig:
    """All parameters explicit: CPU examples are NOT empirically tuned values."""
    epsilon_defect: float
    lambda_defect: float
    tau_utility: float
    defect_margin: float
    sigma_normal: float
    sigma_floor: float

    def __post_init__(self) -> None:
        values = (self.epsilon_defect, self.lambda_defect,
                  self.tau_utility, self.defect_margin,
                  self.sigma_normal, self.sigma_floor)
        if not all(math.isfinite(x) for x in values):
            raise AdmissionProtocolError("Nonfinite CDARG parameter")
        if self.epsilon_defect < 0 or self.lambda_defect < 0:
            raise AdmissionProtocolError("Negative risk budget/penalty")
        if self.defect_margin < 0 or self.sigma_floor <= 0 or self.sigma_normal <= 0:
            raise AdmissionProtocolError("Invalid risk margin/scale")


@dataclass
class AdmissionProposal:
    event_id: str
    step: int
    decision: str
    reason: str
    normal_utility: Optional[float]
    defect_risk: Optional[float]
    max_defect_erosion: Optional[float]
    defect_margin_crossing: Optional[bool]
    utility_minus_penalty: Optional[float]
    normal_anchor_count: int
    defect_anchor_count: int
    prior_bank_digest: str
    shadow_bank_digest: str
    greedy_audit: GreedyAudit
    theta_snapshot: float
    shadow: GreedyBank  # PRIVATE: never feed to an online policy before commit
    applied: bool = False

    def audit(self) -> dict:
        return {
            "event_id": self.event_id, "step": self.step,
            "decision": self.decision, "reason": self.reason,
            "normal_utility": self.normal_utility,
            "defect_risk": self.defect_risk,
            "max_defect_erosion": self.max_defect_erosion,
            "defect_margin_crossing": self.defect_margin_crossing,
            "utility_minus_penalty": self.utility_minus_penalty,
            "normal_anchor_count": self.normal_anchor_count,
            "defect_anchor_count": self.defect_anchor_count,
            "bank_before_sha256": self.prior_bank_digest,
            "shadow_bank_sha256": self.shadow_bank_digest,
            "theta_snapshot": self.theta_snapshot,
            "greedy": {
                "added": self.greedy_audit.appended,
                "replaced": self.greedy_audit.replaced,
                "skipped": self.greedy_audit.skipped,
                "candidate_patches": self.greedy_audit.candidate_patches,
            },
            "committed": self.applied,
        }


class CDARG:
    """Rejects dangerous normal writes before production memory mutation.

    Constructor scorer must accept (anchor_patch_layers, GreedyBank) -> float.
    Use native VisionAD/v22 scorer adapter for real experiments. The default
    CosinePatchScorer elsewhere is ONLY for synthetic causal mechanisms.
    """
    def __init__(self, config: CDARGConfig,
                 scorer: Callable[[Sequence[torch.Tensor], GreedyBank], float]):
        self.config = config
        self.scorer = scorer

    def _score(self, anchor: Anchor, bank: GreedyBank) -> float:
        out = float(self.scorer(anchor.features, bank))
        if not math.isfinite(out):
            raise AdmissionProtocolError("Nonfinite anchor anomaly score")
        return out

    @torch.no_grad()
    def evaluate(self, *, current_verified_normal: VerifiedImage,
                 step: int, bank: GreedyBank, ledger: AnchorLedger,
                 candidate_patches: Sequence[torch.Tensor],
                 theta: float,
                 candidate_already_normalized: bool = False) -> AdmissionProposal:
        if not current_verified_normal.queried or current_verified_normal.label != "normal":
            raise AdmissionProtocolError("Normal admission requires queried image-level normal feedback")
        if step != current_verified_normal.label_revealed_step:
            raise AdmissionProtocolError("Candidate may only be evaluated after its current query disclosure")
        if not math.isfinite(theta):
            raise AdmissionProtocolError("Nonfinite fixed pre-event threshold")

        normal, defect = ledger.prior(step)  # excludes current/late evidence
        prior_digest = bank.sha256()
        shadow, greedy = bank.proposed(
            candidate_patches, already_normalized=candidate_already_normalized)
        shadow_digest = shadow.sha256()
        normal_utility: Optional[float] = None
        defect_risk: Optional[float] = None
        max_defect_erosion: Optional[float] = None
        crossing: Optional[bool] = None
        objective: Optional[float] = None
        accept = False
        reason = ""

        if greedy.appended + greedy.replaced == 0:
            reason = "no_memory_delta"
        else:
            # Compute any *available* risk and benefit before applying the
            # conservative fallback. Missing defects do not erase evidence of
            # foregone normal utility; missing normals do not erase measured
            # defect erosion.
            if normal:
                normal_delta = []
                for anchor in normal:
                    original = self._score(anchor, bank)
                    after = self._score(anchor, shadow)
                    normal_delta.append(max(original - theta, 0.0) -
                                        max(after - theta, 0.0))
                normal_utility = sum(normal_delta) / len(normal_delta)

            if defect:
                weighted_risks = []
                raw_risks = []
                crossing = False
                scale = max(self.config.sigma_normal, self.config.sigma_floor)
                for anchor in defect:
                    original = self._score(anchor, bank)
                    after = self._score(anchor, shadow)
                    erosion = max(original - after, 0.0)
                    raw_risks.append(erosion)
                    weighted_risks.append(anchor.weight * erosion / scale)
                    if (original >= theta + self.config.defect_margin and
                            after < theta + self.config.defect_margin):
                        crossing = True
                defect_risk = sum(weighted_risks) / len(weighted_risks)
                max_defect_erosion = max(raw_risks)

            if not defect:
                # Defect risk remains None (unknown), never zero.
                reason = "defect_risk_unavailable"
            elif not normal:
                reason = "normal_utility_unavailable"
            else:
                objective = normal_utility - self.config.lambda_defect * defect_risk
                if crossing:
                    reason = "protected_defect_margin_crossing"
                elif defect_risk > self.config.epsilon_defect:
                    reason = "defect_risk_budget_exceeded"
                elif objective < self.config.tau_utility:
                    reason = "utility_risk_tradeoff_failed"
                else:
                    accept = True
                    reason = "counterfactual_guard_pass"

        if bank.sha256() != prior_digest:
            raise ShadowError("Live bank was mutated during counterfactual scoring")
        return AdmissionProposal(
            event_id=current_verified_normal.query_id, step=int(step),
            decision="admit" if accept else "reject", reason=reason,
            normal_utility=normal_utility, defect_risk=defect_risk,
            max_defect_erosion=max_defect_erosion, defect_margin_crossing=crossing,
            utility_minus_penalty=objective, normal_anchor_count=len(normal),
            defect_anchor_count=len(defect), prior_bank_digest=prior_digest,
            shadow_bank_digest=shadow_digest, greedy_audit=greedy,
            theta_snapshot=float(theta), shadow=shadow)

    @staticmethod
    def commit_if_accepted(proposal: AdmissionProposal, bank: GreedyBank) -> bool:
        if proposal.applied:
            raise AdmissionProtocolError("Proposal already committed")
        if proposal.decision != "admit":
            if bank.sha256() != proposal.prior_bank_digest:
                raise ShadowError("Rejected proposal nonetheless changed memory")
            return False
        bank.commit(proposal.shadow,
                    expected_version=proposal.greedy_audit.old_version,
                    expected_sha256=proposal.prior_bank_digest)
        proposal.applied = True
        return True


@torch.no_grad()
def heldout_counterfactual_erosion(
        probe_defect_features: Sequence[Sequence[torch.Tensor]],
        live_bank: GreedyBank, shadow_bank: GreedyBank,
        scorer: Callable[[Sequence[torch.Tensor], GreedyBank], float]
        ) -> dict:
    """OFFLINE evaluation only. Never call inside CDARG.evaluate/admission.

    This entrypoint accepts probe defect features only after a decision is
    finalized. It must not write to the online AnchorLedger or QueryController.
    """
    if not probe_defect_features:
        raise AdmissionProtocolError("Empty heldout defect probe")
    erosion = []
    for image in probe_defect_features:
        before, after = float(scorer(image, live_bank)), float(scorer(image, shadow_bank))
        if not (math.isfinite(before) and math.isfinite(after)):
            raise AdmissionProtocolError("Nonfinite heldout scorer")
        erosion.append(max(before - after, 0.0))
    return {"mean_positive_score_erosion": sum(erosion) / len(erosion),
            "max_positive_score_erosion": max(erosion),
            "N_heldout_defect_images": len(erosion),
            "offline_only_not_visible_to_gate": True}
