"""Machine-readable method cards and fail-closed Stage 0 controller registry."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .controllers import (AdaptedDualACI, BlockedController, FrozenTheta,
                          OriginalGuardedSidecar, RollingQ90QueriedNormal,
                          StateConsistentQuantile)


METHOD_CARDS: dict[str, dict[str, Any]] = {
    "B0": {"name": "Frozen-Theta", "status": "IMPLEMENTED",
           "equation": "theta_t=theta0; predict=1[s_t>=theta_t]",
           "information": ["shared current unlabeled score"],
           "initialization": "D2 higher q90 of 64 additional C64 normals under M0"},
    "B1": {"name": "RollingQ90QueriedNormal", "status": "IMPLEMENTED",
           "equation": "theta= higher_q90(last W queried verified-normal stale scores)",
           "information": ["shared current unlabeled score", "queried labels"],
           "defaults": {"W": 256, "n_min": 16, "q": 0.90,
                        "insufficient": "HOLD at theta0"},
           "fidelity": "Proposed frozen implementation defaults; no unlabeled scores."},
    "B2": {"name": "OriginalGuardedDualAnchorSidecar", "status": "IMPLEMENTED_ADAPTED",
           "source": "test_phase3_AIF_simplified_v22.GuardedDualAnchorThresholdController",
           "information": ["additional C64 scalar scores at M0", "queried labels"],
           "adaptations": ["disable source normal-only fallback because unqueried labels are unavailable",
                           "use source native numpy.quantile for C64 initialization; report separately from D2 theta0",
                           "balanced rescue and fast rescue remain source defaults disabled"],
           "no_guarantee": "Historical v22 behavior is wrapped for this interface; no native full-loop parity claim."},
    "B3": {"name": "StateConsistentQuantile", "status": "IMPLEMENTED_PROPOSED",
           "equation": "on M change: raw=higher_q90(C64 rescored at M); theta'=clip(raw,theta+-Delta); otherwise HOLD",
           "information": ["protected C64 current-M scores"],
           "defaults": {"Delta": 0.01, "min_normals": 16, "q": 0.90},
           "source_exclusion": "C64 protected-only primary; queried-normal LSO excluded from B3."},
    "B4": {"name": "ADAPTED_DUAL_ACI", "status": "IMPLEMENTED_ADAPTED",
           "source": "Gibbs and Candès, Adaptive Conformal Inference Under Distribution Shift, NeurIPS 2021, https://papers.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html",
           "equation": "aN+=gammaN*(alpha-1[pred=defect]) on queried y=0; aD+=gammaD*(beta-1[pred=normal]) on queried y=1; project to [Q_D(aD),Q_N(1-aN)] when consistent",
           "information": ["queried labels", "current-M normal and defect anchors"],
           "defaults": {"alpha": 0.10, "beta": 0.10, "gamma_normal": 0.005,
                        "gamma_defect": 0.005},
           "no_guarantee": "Sparse selective feedback and dependent dynamic-memory scores do not inherit ACI guarantees."},
    "B5a": {"name": "SPOT", "status": "BLOCKED_FIDELITY",
            "source": "Siffer et al., KDD 2017, https://www.eecs.yorku.ca/course_archive/2018-19/F/6412/reading/kdd17p1067.pdf",
            "equation": "zq=u+(sigma/xi)*((q*n/Nu)^(-xi)-1); xi->0: u+sigma*log(Nu/(q*n))",
            "defaults": {"n_init": 1000, "level": 0.98, "q": 0.001},
            "blocker": "No source-equivalent Grimshaw fit plus pinned author-reference trace has been implemented and verified; no substitute estimator is labeled SPOT."},
    "B5b": {"name": "dSPOT", "status": "BLOCKED_FIDELITY",
            "source": "Siffer et al., KDD 2017 (same source as B5a)",
            "equation": "SPOT on score minus causal local mean over d previous accepted normal observations; threshold=mean+zq",
            "defaults": {"n_init": 1000, "level": 0.98, "q": 0.001, "d": 50},
            "blocker": "The original source drift/update convention lacks a verified implementation trace in this package."},
    "B6": {"name": "SunCS", "status": "BLOCKED_FIDELITY",
            "source": "Sun et al., ICML 2024, https://proceedings.mlr.press/v235/sun24h.html",
            "information": ["source method uses an unsupervised score stream and emits a tri-state decision"],
            "blocker": "Exact reference algorithm test and a preregistered mapping from abstention to primary balanced accuracy are not complete; dynamic-memory assumptions also fail to transfer automatically."},
    "B7": {"name": "SCDAC-G", "status": "BLOCKED_FIDELITY",
            "source": "Internal Deep Research proposal referenced by handoff; full report absent from workspace",
            "equations": "The internal candidate outline exists in BASELINE_FIDELITY_AND_REFERENCES.md, but exact bound, minimum counts, fallback and confidence rules are not available.",
            "blocker": "Fail closed until authoritative proposal/version is supplied; no risk bound or conflict fallback is invented."},
}


def method_cards() -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in METHOD_CARDS.items()}


def build_controllers(theta0: float, c64_scores: list[float], *,
                      include_blocked: bool = False) -> list[Any]:
    if len(c64_scores) != 64:
        raise ValueError("Require 64 additional C64 source-normal scores")
    controllers: list[Any] = [
        FrozenTheta(float(theta0)),
        RollingQ90QueriedNormal(float(theta0)),
        OriginalGuardedSidecar(c64_scores, float(theta0)),
        StateConsistentQuantile(float(theta0)),
        AdaptedDualACI(float(theta0)),
    ]
    if include_blocked:
        controllers.extend(BlockedController(key, str(card.get("blocker")), theta0)
                           for key, card in METHOD_CARDS.items()
                           if card["status"] == "BLOCKED_FIDELITY")
    return controllers
