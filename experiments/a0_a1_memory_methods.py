#!/usr/bin/env python3
"""A0/A1 fixed-capacity streaming memory baselines, without query-policy coupling.

All methods receive identical verified-normal candidate patches selected against
the common immutable reference bank. This isolates retention, not the patch
selection mechanism. The v22 utility retention is a component of the historical
system, NOT a complete BEA/STM/LTM reproduction.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any

ARM_NAMES = ("static", "fifo", "reservoir", "online_greedy_coreset",
             "v22_normal_stm", "v22_utility_stm")


def stable_seed(*items: object) -> int:
    return int(hashlib.sha256("|".join(map(str, items)).encode()).hexdigest()[:16], 16)


def _distance_table(protos):
    """Pairwise cosine distance, no cached secondary prototype bank."""
    import torch
    norm = torch.nn.functional.normalize(protos.float(), dim=1)
    d = 1.0 - torch.clamp(norm @ norm.T, -1., 1.)
    d.fill_diagonal_(float("inf"))
    return d


def kcenter_choice(prototypes, candidate):
    """Approximate online k-center swap; compare novelty to minimum pair gap.

    Return replacement index, or None to reject candidate. This is an online
    greedy coverage heuristic, NOT an offline globally optimized k-center.
    Extra O(K^2) *compute* per candidate when bank is full; O(K^2) ephemeral
    matrix memory, reported by the driver as an implementation caveat.
    """
    import torch
    if not len(prototypes):
        return 0
    p = torch.nn.functional.normalize(prototypes.float(), dim=1)
    q = torch.nn.functional.normalize(candidate.float().reshape(1, -1), dim=1)
    nearest = float((1.0 - torch.clamp(q @ p.T, -1., 1.)).min().item())
    if len(prototypes) < 2:
        return 0 if nearest > 1e-7 else None
    d = _distance_table(prototypes)
    nearest_existing = d.min(dim=1).values
    redundancy, slot = nearest_existing.min(dim=0)
    if nearest > float(redundancy.item()) + 1e-7:
        return int(slot.item())
    return None


class MemoryArm:
    def __init__(self, name: str, bank: Any, *, seed: int, cls: str = "p0"):
        if name not in ARM_NAMES:
            raise ValueError(f"Unknown memory method {name}")
        self.name, self.bank, self.cls = name, bank, cls
        self.rng = [random.Random(stable_seed(seed, name, layer))
                    for layer in range(len(bank.normal_fixed[cls][0]["id"]))]
        self.seen = [0 for _ in self.rng]
        self.fifo_cursor = [0 for _ in self.rng]
        self.candidate_events = 0
        self.candidate_patches = 0
        self.accepted_patches = 0
        self.skipped_patches = 0

        # The only retrievable mutable representation is normal_dyn. No
        # defect/correction/maturation/LTM path is allowed in this trial.
        bank.enable_ltm_retrieval = False
        bank.enable_ltm_promotion = False
        bank.enable_maturation_buffer_retrieval = False
        bank.enable_maturation_buffer = False
        bank.enable_utility_aware_stm_retention = False
        bank.use_defect_bank = False
        bank.corr_cfg.enabled = False

    def sizes(self) -> list[int]:
        return [cb.size() for cb in self.bank.normal_dyn[self.cls][0]["id"]]

    def capacities(self) -> list[int]:
        return [int(cb.K) for cb in self.bank.normal_dyn[self.cls][0]["id"]]

    def _append(self, cb, f):
        import torch
        cb.protos = torch.cat((cb.protos, f.reshape(1, -1).to(
            cb.device, dtype=cb.dtype)), dim=0)

    def observe_verified_normal(self, impl, features, patch_indices,
                                *, step: int, score: float, theta: float,
                                sigma: float):
        """Identical candidates for all dynamic arms, fed AFTER prediction."""
        if self.name == "static":
            return
        import torch
        self.candidate_events += 1
        allowed = self.bank.normal_dyn[self.cls][0]["id"]
        for layer, cb in enumerate(allowed):
            indices = list(patch_indices.get(layer, []))
            if not indices:
                continue
            q = impl.l2_normalize(features[layer].to(cb.device))
            if max(indices) >= len(q) or min(indices) < 0:
                raise AssertionError("Invalid shared patch-selection index")
            selected = q[torch.as_tensor(indices, device=cb.device, dtype=torch.long)]
            self.candidate_patches += len(selected)
            if self.name in ("v22_normal_stm", "v22_utility_stm"):
                # Preserve the v22 utility-aware FixedCapacityCodebook.update
                # primitive. Disable external maturation, LTM and correction.
                _allow, _keep, _reason, info = impl.select_informative_normal_write(
                    score=score, theta=theta, sigmaN=sigma,
                    pred_defect=score >= theta, novelty=0.0)
                report = cb.update(
                    selected, tau_insert=.15, tau_replace=.25, allow_replace=True,
                    use_utility_retention=(self.name == "v22_utility_stm"),
                    utility_score=float(info["write_value"]),
                    outcome="FP" if score >= theta else "TN",
                    boundary_value=float(info["boundary"]), event_id=step,
                    step=step, coverage_gain=0.0)
                accepted = int(report["n_added"]) + int(report["n_replaced"])
                self.accepted_patches += accepted
                self.skipped_patches += len(selected) - accepted
                continue

            for f in selected:
                self.seen[layer] += 1
                n = cb.size()
                K = int(cb.K)
                if K < 1:
                    raise AssertionError("Dynamic codebook capacity must be positive")
                if n < K:
                    self._append(cb, f)
                    self.accepted_patches += 1
                elif self.name == "fifo":
                    j = self.fifo_cursor[layer] % K
                    cb.protos[j] = f.to(dtype=cb.dtype)
                    self.fifo_cursor[layer] = (j + 1) % K
                    self.accepted_patches += 1
                elif self.name == "reservoir":
                    # Exact Algorithm R; all candidate patches form the stream.
                    j = self.rng[layer].randrange(self.seen[layer])
                    if j < K:
                        cb.protos[j] = f.to(dtype=cb.dtype)
                        self.accepted_patches += 1
                    else:
                        self.skipped_patches += 1
                elif self.name == "online_greedy_coreset":
                    j = kcenter_choice(cb.protos, f)
                    if j is None:
                        self.skipped_patches += 1
                    else:
                        cb.protos[j] = f.to(dtype=cb.dtype)
                        self.accepted_patches += 1
                else:
                    raise AssertionError(self.name)
            if cb.size() > K:
                raise AssertionError("Mutable bank overflow")

    def audit(self) -> dict:
        sizes = self.sizes()
        caps = self.capacities()
        if any(x > y for x, y in zip(sizes, caps)):
            raise AssertionError("Bank over capacity")
        # Include stored metadata overhead for utility-aware v22; other
        # methods store no extra retrievable prototypes.
        return {
            "method": self.name, "prototype_per_layer": sizes,
            "capacity_per_layer": caps,
            "mutable_prototypes": sum(sizes),
            "mutable_capacity": sum(caps),
            "candidate_events": self.candidate_events,
            "candidate_patches": self.candidate_patches,
            "accepted_patches": self.accepted_patches,
            "skipped_patches": self.skipped_patches,
            "notes": (
                "v22 utility-aware FixedCapacityCodebook.update; no LTM/maturation"
                if self.name == "v22_utility_stm" else
                "v22 normal_dyn FixedCapacityCodebook.update, original tau=0.15/0.25, no utility, LTM or maturation"
                if self.name == "v22_normal_stm" else
                "online min-pair-gap greedy replacement; not offline optimal"
                if self.name == "online_greedy_coreset" else
                "strict patch-level FIFO" if self.name == "fifo" else
                "patch-level unbiased reservoir Algorithm R"
                if self.name == "reservoir" else "no mutable memory"),
        }
