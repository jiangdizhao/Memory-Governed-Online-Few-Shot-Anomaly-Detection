"""Machine-readable method cards and fail-closed Stage 0 controller registry."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .controllers import (AdaptedDualACI, BlockedController, FrozenTheta,
                          OriginalGuardedSidecar, RollingQ90QueriedNormal,
                          StateConsistentQuantile)
from .scdac_g import SCDACGController


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
           "equation": "aN+=gammaN*(alpha-1[pred=defect]) on queried y=0; aD+=gammaD*(beta-1[pred=normal]) on queried y=1; exact empirical bounds: [normal lower, defect upper]",
           "information": ["queried labels", "current-M normal and defect anchors"],
           "defaults": {"alpha": 0.10, "beta": 0.10, "gamma_normal": 0.005,
                        "gamma_defect": 0.005},
           "no_guarantee": "Sparse selective feedback and dependent dynamic-memory scores do not inherit ACI guarantees."},
    "B5a": {"name": "SPOT", "status": "BLOCKED_FIDELITY",
            "source": "Siffer et al., KDD 2017, https://www.eecs.yorku.ca/course_archive/2018-19/F/6412/reading/kdd17p1067.pdf",
            "author_source": "https://github.com/asiffer/libspot/tree/d57e921a08930b960d07adcbec3a839bd228fe94",
            "author_source_file_blob_sha": "fdebc8c216fffeea93afcc51efab83a2d6013351",
            "equation": "zq=u+(sigma/xi)*((q*n/Nu)^(-xi)-1); xi->0: u+sigma*log(Nu/(q*n))",
            "defaults": {"n_init": 1000, "level": 0.98, "q": 0.001},
            "blocker": "Pinned author source is libspot 3.0.0 SPOT code, but no exact cross-version equation/edge-case reference trace has been verified against the KDD 2017 Grimshaw procedure; no substitute estimator is labeled SPOT."},
    "B5b": {"name": "dSPOT", "status": "BLOCKED_FIDELITY",
            "source": "Siffer et al., KDD 2017 (same source as B5a)",
            "author_source": "https://github.com/asiffer/libspot/tree/d57e921a08930b960d07adcbec3a839bd228fe94",
            "equation": "SPOT on score minus causal local mean over d previous accepted normal observations; threshold=mean+zq",
            "defaults": {"n_init": 1000, "level": 0.98, "q": 0.001, "d": 50},
            "blocker": "The pinned libspot 3.0.0 source has no dSPOT implementation; the paper describes a separate moving-average/non-anomaly window. No faithful dSPOT implementation or reference test is present."},
    "B6": {"name": "SunCS", "status": "BLOCKED_FIDELITY",
            "source": "Sun et al., ICML 2024, https://proceedings.mlr.press/v235/sun24h.html",
            "information": ["source method uses an unsupervised score stream and emits a tri-state decision"],
            "blocker": "Exact reference algorithm test and a preregistered mapping from abstention to primary balanced accuracy are not complete; dynamic-memory assumptions also fail to transfer automatically."},
    "B7": {"name": "SCDAC-G", "status": "IMPLEMENTED_PROVISIONAL",
            "source": "docs/Designing and Prioritizing Memory-Dependent _ State-Consistent Calibration for Dynamic-Memory Indust.pdf; SHA-256 5fe07256384351c1170d67183eae5a80e5aa289a03586ba767671f062f4fb977",
            "equations": "Current-state scores; empirical FPR/FNR; fixed-time one-sided Clopper-Pearson descriptive upper endpoints; minimize (U_FPR+U_FNR)/2 + lambda_move*abs(theta-theta_t)/(sigma_s+epsilon_s), then clip movement",
            "information": ["protected C64 normal anchors", "queried verified normal anchors with all-layer LSO", "queried defect anchors", "shared unlabeled stream scores are not consumed by B7"],
            "defaults": {"min_normals": 16, "min_defects": 8, "delta": 0.05, "support_fpr": 0.20, "support_fnr": 0.20, "lambda_move": 0.05, "max_step": 0.01, "fpr_diagnostic_targets": [0.05, 0.10, 0.20]},
            "adaptation": "PROVISIONAL_METHOD_SPEC; exact numerical choices are registered in the new method spec because the proposal leaves them open",
            "no_guarantee": "Descriptive only. Selective labels, serial dependence, repeated memory rescoring and threshold selection invalidate automatic transfer of binomial confidence guarantees."},
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
        SCDACGController(float(theta0)),
    ]
    if include_blocked:
        controllers.extend(BlockedController(key, str(card.get("blocker")), theta0)
                           for key, card in METHOD_CARDS.items()
                           if card["status"] == "BLOCKED_FIDELITY")
    return controllers
