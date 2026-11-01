"""Feedback-proportional sign allocation for the frozen v22 correction bank.

This module changes only full-bank replacement candidate selection. The native
Gaussian readout and all other correction-bank update mathematics are inherited
or copied verbatim from ``SignedCorrectionCodebook``.
"""
import re
from numbers import Integral
from typing import Any, Dict, Optional

import torch

import test_phase3_AIF_simplified_v22 as _native


class FPSAContractError(ValueError):
    """Raised when queried-feedback provenance is invalid or out of order."""


class FPSASignedCorrectionCodebook(_native.SignedCorrectionCodebook):
    """Native signed correction bank with conservative feedback-based quotas.

    ``observe_verified_feedback`` must be called once per distinct, legally
    queried source group after the current event's scores have been sealed.
    Feedback affects retention targets only; it never changes ``apply``.
    """

    def __init__(self, K: int, feat_dim: int, device: torch.device, cfg: Any):
        if isinstance(K, bool) or int(K) < 2:
            raise FPSAContractError("FPSA requires capacity K >= 2")
        super().__init__(K, feat_dim, device, cfg)
        self._queried_source_groups: set[str] = set()
        self._queried_normal_groups = 0
        self._queried_anomalous_groups = 0
        self._last_feedback_step = -1
        self.quota_borrow_opportunities = 0
        self.quota_cross_sign_replacements = 0
        self.quota_borrow_rejected = 0
        self.last_update_diagnostics: dict[str, int] = {
            "quota_borrow_opportunities": 0,
            "quota_cross_sign_replacements": 0,
            "quota_borrow_rejected": 0,
            "inserted": 0,
            "merged": 0,
            "replaced": 0,
            "skipped": 0,
        }

    def observe_verified_feedback(
        self,
        label: int,
        source_group_sha256: str,
        step: int,
        score_sealed: bool = True,
    ) -> Dict[str, Any]:
        """Record one verified binary query after its score has been sealed.

        Source group digests are counted once for the lifetime of this bank.
        A repeated digest, non-increasing step, invalid label/digest, or an
        unsealed score raises before any counters or state are changed.
        """
        if score_sealed is not True:
            raise FPSAContractError("queried feedback cannot be registered before score seal")
        if isinstance(label, bool) or not isinstance(label, Integral) or int(label) not in (0, 1):
            raise FPSAContractError("verified feedback label must be integer 0 or 1")
        if (not isinstance(source_group_sha256, str) or
                re.fullmatch(r"[0-9a-f]{64}", source_group_sha256) is None):
            raise FPSAContractError("source group identity must be a lowercase SHA-256 digest")
        if isinstance(step, bool) or not isinstance(step, Integral) or int(step) < 0:
            raise FPSAContractError("feedback step must be a nonnegative integer")
        step = int(step)
        if source_group_sha256 in self._queried_source_groups:
            raise FPSAContractError("duplicate queried Source Group feedback")
        if step <= self._last_feedback_step:
            raise FPSAContractError("queried feedback steps must be strictly increasing")

        # Commit feedback state only after every validation above succeeds.
        self._queried_source_groups.add(source_group_sha256)
        self._last_feedback_step = step
        if int(label) == 0:
            self._queried_normal_groups += 1
        else:
            self._queried_anomalous_groups += 1
        return self.diagnostics()

    def _capacity_targets(self) -> tuple[Optional[int], Optional[int]]:
        n_pos = self._queried_anomalous_groups
        n_neg = self._queried_normal_groups
        total = n_pos + n_neg
        if total == 0:
            return None, None
        if n_pos == 0:
            return 0, self.K
        if n_neg == 0:
            return self.K, 0
        # Explicit half-up rounding, not Python's ties-to-even round().
        pos = (2 * self.K * n_pos + total) // (2 * total)
        pos = min(self.K - 1, max(1, pos))
        return pos, self.K - pos

    def _live_occupancy(self) -> tuple[int, int, int]:
        beta = self.beta[: self.n]
        return (int((beta > 0).sum().item()), int((beta < 0).sum().item()),
                int((beta == 0).sum().item()))

    def diagnostics(self) -> Dict[str, Any]:
        """Return compact feedback, quota, occupancy, and native update counts."""
        target_pos, target_neg = self._capacity_targets()
        actual_pos, actual_neg, zero = self._live_occupancy()
        native = super().summary()
        return {
            "queried_normal_source_groups": int(self._queried_normal_groups),
            "queried_anomalous_source_groups": int(self._queried_anomalous_groups),
            "queried_source_groups_total": int(len(self._queried_source_groups)),
            "last_feedback_step": int(self._last_feedback_step),
            "target_positive_capacity": target_pos,
            "target_negative_capacity": target_neg,
            "actual_positive_occupancy": actual_pos,
            "actual_negative_occupancy": actual_neg,
            "zero_sign_occupancy": zero,
            "quota_borrow_opportunities": int(self.quota_borrow_opportunities),
            "quota_cross_sign_replacements": int(self.quota_cross_sign_replacements),
            "quota_borrow_rejected": int(self.quota_borrow_rejected),
            "native_inserted_total": int(native["inserted_total"]),
            "native_merged_total": int(native["merged_total"]),
            "native_replaced_total": int(native["replaced_total"]),
            "native_skipped_total": int(native["skipped_total"]),
            "native_same_sign_replaced_total": int(native["same_sign_replaced_total"]),
            "native_cross_sign_replaced_total": int(native["cross_sign_replaced_total"]),
            "native_same_sign_merge_total": int(native["same_sign_merge_total"]),
        }

    def summary(self) -> Dict[str, Any]:
        out = super().summary()
        out.update(self.diagnostics())
        return out

    @torch.no_grad()
    def update(
        self,
        q: torch.Tensor,
        beta: torch.Tensor,
        *,
        radius: Optional[torch.Tensor] = None,
        kappa: Optional[torch.Tensor] = None,
        outcome: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Dict[str, int]:
        """Native update with one FPSA-only full-bank candidate-selection branch."""
        if q.numel() == 0:
            self.last_update_diagnostics = {
                "quota_borrow_opportunities": 0,
                "quota_cross_sign_replacements": 0,
                "quota_borrow_rejected": 0,
                "inserted": 0,
                "merged": 0,
                "replaced": 0,
                "skipped": 0,
            }
            return {"inserted": 0, "merged": 0, "replaced": 0, "skipped": 0,
                    "same_sign_replaced": 0, "cross_sign_replaced": 0}

        q = _native.l2_normalize(q.to(self.device))
        beta = beta.to(self.device).float().clamp(-float(self.cfg.beta_max), float(self.cfg.beta_max))

        M = int(q.shape[0])
        if radius is None:
            radius = torch.full((M,), float(self.cfg.radius0), device=self.device, dtype=torch.float32)
        else:
            radius = radius.to(self.device).float()
        radius = radius.clamp(float(self.cfg.radius_min), float(self.cfg.radius_max))

        if kappa is None:
            kappa = torch.full((M,), float(self.cfg.kappa0), device=self.device, dtype=torch.float32)
        else:
            kappa = kappa.to(self.device).float()
        kappa = kappa.clamp(0.1, float(self.cfg.kappa_max))

        inserted = merged = replaced = skipped = 0
        same_sign_replaced = cross_sign_replaced = 0
        same_sign_merge = 0
        quota_opportunities = quota_replacements = quota_rejected = 0
        outcome_key = self._outcome_key(outcome)
        new_is_mistake = 1.0 if outcome_key in {"FP", "FN"} else 0.0

        for i in range(M):
            b_i = float(beta[i].item())
            if abs(b_i) < float(self.cfg.beta_min - 1e-5):
                skipped += 1
                continue

            if self.n == 0:
                self._write_slot(0, q[i], beta[i], radius[i], kappa[i], outcome=outcome, step=step)
                self.n = 1
                inserted += 1
                continue

            p = self.protos[: self.n]
            dist = self._cos_dist(q[i : i + 1], p).squeeze(0)
            dmin, j = torch.min(dist, dim=0)
            j = int(j.item())
            dmin_val = float(dmin.item())

            same_sign = (float(self.beta[j].item()) * b_i) > 0.0

            if dmin_val <= float(self.cfg.merge_eps) and same_sign:
                c = float(self.count[j].item())
                self.protos[j] = _native.l2_normalize((c * self.protos[j] + q[i]) / (c + 1.0))
                self.beta[j] = torch.clamp((c * self.beta[j] + beta[i]) / (c + 1.0),
                                           -float(self.cfg.beta_max), float(self.cfg.beta_max))
                self.radius[j] = torch.clamp((c * self.radius[j] + radius[i]) / (c + 1.0),
                                             float(self.cfg.radius_min), float(self.cfg.radius_max))
                self.kappa[j] = torch.clamp(self.kappa[j] + float(self.cfg.kappa_inc),
                                            0.1, float(self.cfg.kappa_max))
                self.count[j] += 1
                self._bump_outcome(j, outcome)
                if step is not None:
                    self.last_step[j] = int(step)
                merged += 1
                same_sign_merge += 1
                continue

            if self.n < self.K:
                self._write_slot(self.n, q[i], beta[i], radius[i], kappa[i], outcome=outcome, step=step)
                self.n += 1
                inserted += 1
                continue

            counts = self.count[: self.n].float()
            babs = self.beta[: self.n].abs()
            mistake_count = (self.outcome_fp[: self.n] + self.outcome_fn[: self.n]).float()
            same_mask = (self.beta[: self.n] * float(b_i)) > 0.0

            # FPSA-SPECIFIC BRANCH: borrow only from an over-target opposite
            # sign when the incoming sign is below its feedback-only target.
            quota_borrow_attempted = False
            target_pos, target_neg = self._capacity_targets()
            if target_pos is not None and target_neg is not None:
                actual_pos, actual_neg, _ = self._live_occupancy()
                if b_i > 0.0 and actual_pos < target_pos and actual_neg > target_neg:
                    quota_borrow_attempted = True
                elif b_i < 0.0 and actual_neg < target_neg and actual_pos > target_pos:
                    quota_borrow_attempted = True

            if quota_borrow_attempted:
                quota_opportunities += 1
                cand_idx = torch.nonzero((self.beta[: self.n] * float(b_i)) < 0.0,
                                         as_tuple=False).view(-1)
                if cand_idx.numel() == 0:
                    raise FPSAContractError("quota borrowing target has no opposite-sign candidate")
            elif bool(getattr(self.cfg, "enable_sign_aware_retention", True)) and bool(same_mask.any().item()):
                cand_idx = torch.nonzero(same_mask, as_tuple=False).view(-1)
            else:
                cand_idx = torch.arange(self.n, device=self.device, dtype=torch.long)

            # Original native retention formula: unchanged for either candidate set.
            retention = (
                counts[cand_idx]
                + 1.25 * mistake_count[cand_idx]
                + 0.50 * torch.clamp(babs[cand_idx] / max(float(self.cfg.beta_max), 1e-6), 0.0, 1.0)
                + 0.25 * torch.clamp(self.kappa[: self.n][cand_idx] / max(float(self.cfg.kappa_max), 1e-6), 0.0, 1.0)
            )
            pos = int(torch.argmin(retention).item())
            jrep = int(cand_idx[pos].item())
            weakest_score = float(retention[pos].item())
            new_score = float(
                1.0
                + 1.25 * new_is_mistake
                + 0.50 * min(abs(b_i) / max(float(self.cfg.beta_max), 1e-6), 1.0)
                + 0.25 * min(float(kappa[i].item()) / max(float(self.cfg.kappa_max), 1e-6), 1.0)
            )

            # Original native acceptance comparison: unchanged; FPSA never forces admission.
            replace_ok = True
            if bool(getattr(self.cfg, "enable_sign_aware_retention", True)):
                replace_ok = new_score >= weakest_score + float(getattr(self.cfg, "corr_replace_margin", 0.0))
            else:
                replace_ok = abs(b_i) > float(babs[jrep].item()) + 1e-6

            if replace_ok:
                was_same = (float(self.beta[jrep].item()) * b_i) > 0.0
                self._write_slot(jrep, q[i], beta[i], radius[i], kappa[i], outcome=outcome, step=step)
                replaced += 1
                if was_same:
                    same_sign_replaced += 1
                else:
                    cross_sign_replaced += 1
                    if quota_borrow_attempted:
                        quota_replacements += 1
            else:
                skipped += 1
                if quota_borrow_attempted:
                    quota_rejected += 1

        self.counters["inserted_total"] = int(self.counters.get("inserted_total", 0)) + int(inserted)
        self.counters["merged_total"] = int(self.counters.get("merged_total", 0)) + int(merged)
        self.counters["replaced_total"] = int(self.counters.get("replaced_total", 0)) + int(replaced)
        self.counters["skipped_total"] = int(self.counters.get("skipped_total", 0)) + int(skipped)
        self.counters["same_sign_replaced_total"] = int(self.counters.get("same_sign_replaced_total", 0)) + int(same_sign_replaced)
        self.counters["cross_sign_replaced_total"] = int(self.counters.get("cross_sign_replaced_total", 0)) + int(cross_sign_replaced)
        self.counters["same_sign_merge_total"] = int(self.counters.get("same_sign_merge_total", 0)) + int(same_sign_merge)

        self.quota_borrow_opportunities += quota_opportunities
        self.quota_cross_sign_replacements += quota_replacements
        self.quota_borrow_rejected += quota_rejected
        self.last_update_diagnostics = {
            "quota_borrow_opportunities": quota_opportunities,
            "quota_cross_sign_replacements": quota_replacements,
            "quota_borrow_rejected": quota_rejected,
            "inserted": inserted,
            "merged": merged,
            "replaced": replaced,
            "skipped": skipped,
        }

        return {
            "inserted": int(inserted), "merged": int(merged), "replaced": int(replaced), "skipped": int(skipped),
            "same_sign_replaced": int(same_sign_replaced), "cross_sign_replaced": int(cross_sign_replaced),
            "same_sign_merged": int(same_sign_merge),
        }


# Short alias for callers that register codebooks by experiment name.
FPSACorrectionCodebook = FPSASignedCorrectionCodebook
