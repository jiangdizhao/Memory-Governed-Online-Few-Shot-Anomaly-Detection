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
    "B7": {"name": "SCDAC-G", "version": "SCDAC-G-v0.1-THIRD-PASS-PROVISIONAL_METHOD_SPEC",
            "status": "IMPLEMENTED_PROVISIONAL",
            "source": "docs/Designing and Prioritizing Memory-Dependent _ State-Consistent Calibration for Dynamic-Memory Indust.pdf; SHA-256 5fe07256384351c1170d67183eae5a80e5aa289a03586ba767671f062f4fb977",
            "equations": "Current-state one-sided CP descriptive upper endpoints; primary min balanced upper risk plus movement over two-sided feasible candidates; if empty, CONFLICT minimum-risk compromise over all candidates; clip then report executed feasibility; FPR diagnostic minimizes FNR upper plus movement cost",
            "information": ["protected C64 normal anchors", "queried verified normal anchors with all-layer LSO", "queried defect anchors", "shared unlabeled stream scores are not consumed by B7"],
            "defaults": {"min_normals": 16, "min_defects": 8, "delta": 0.05, "support_fpr": 0.20, "support_fnr": 0.20, "lambda_move": 0.05, "max_step": 0.01, "fpr_diagnostic_targets": [0.05, 0.10, 0.20]},
            "adaptation": "PROVISIONAL_METHOD_SPEC; exact numerical choices are registered in the new method spec because the proposal leaves them open",
            "no_guarantee": "Descriptive only. Selective labels, serial dependence, repeated memory rescoring and threshold selection invalidate automatic transfer of binomial confidence guarantees."},
}

# Keep the method card schema complete even for BLOCKED_FIDELITY families.
# `None` and explicit blocked text are intentional: the card must expose what
# is unknown instead of implying a source or parameter was verified.
_CITATIONS = {
    "B0": ("Registered frozen-threshold control", None, None, None),
    "B1": ("Registered rolling-quantile control", None, None, None),
    "B2": ("Historical v22 Guarded Dual Anchor sidecar", None, None, None),
    "B3": ("Registered state-consistent quantile proposal", None, None, None),
    "B4": ("Adaptive Conformal Inference Under Distribution Shift",
           "Isaac Gibbs; Emmanuel Candès", "NeurIPS", 2021),
    "B5a": ("Anomaly Detection in Streams with Extreme Value Theory",
            "Alban Siffer; Pierre-Alain Fouque; Alexandre Termier; Christine Largouet",
            "KDD", 2017),
    "B5b": ("Anomaly Detection in Streams with Extreme Value Theory",
            "Alban Siffer; Pierre-Alain Fouque; Alexandre Termier; Christine Largouet",
            "KDD", 2017),
    "B6": ("Online Adaptive Anomaly Thresholding with Confidence Sequences",
           "Sophia Huiwen Sun; Abishek Sankararaman; Balakrishnan Murali Narayanaswamy",
           "ICML / PMLR 235", 2024),
    "B7": ("Designing and Prioritizing Memory-Dependent / State-Consistent Calibration for Dynamic-Memory Industrial Anomaly Detection",
           "Internal proposal authorship not recorded in repository metadata",
           "Internal research proposal", None),
}
_REFERENCE_TESTS = {
    "B0": ["test_theta0_uses_d2_higher_q90_at_zero_based_rank_57"],
    "B1": ["test_b1_only_queried_verified_normals_enter_buffer"],
    "B2": ["test_b2_uses_historical_sidecar_with_unqueried_fallback_disabled"],
    "B3": ["test_b3_updates_once_per_memory_version"],
    "B4": ["test_b4_exact_error_count_boundaries_handle_ties",
           "test_b4_missing_side_is_not_zero_risk"],
    "B5a": [], "B5b": [], "B6": [],
    "B7": ["tests/calibration/test_scdac_g.py"],
}
_PAPER_URLS = {
    "B4": "https://papers.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html",
    "B5a": "https://doi.org/10.1145/3097983.3098144",
    "B5b": "https://doi.org/10.1145/3097983.3098144",
    "B6": "https://proceedings.mlr.press/v235/sun24h.html",
}
_MISSING_DATA = {
    "B0": "Requires all 64 additional C64 normal scores; fail closed otherwise.",
    "B1": "Holds theta0 until n_min queried verified normals are available.",
    "B2": "Unavailable labels do not enter the adapted sidecar; source fallback is disabled.",
    "B3": "Holds theta when current-memory protected-C64 evidence is insufficient.",
    "B4": "Missing class-side evidence is unavailable and never assigned zero risk.",
    "B5a": "No executable behavior; BLOCKED_FIDELITY.",
    "B5b": "No executable behavior; BLOCKED_FIDELITY.",
    "B6": "No executable behavior; BLOCKED_FIDELITY.",
    "B7": "NONE/NORMAL_ONLY/DEFECT_ONLY hold; both-side minimums gate optimization.",
}
for _key, _card in METHOD_CARDS.items():
    _title, _authors, _venue, _year = _CITATIONS[_key]
    _card.update({
        "paper_title": _title,
        "authors": _authors,
        "venue": _venue,
        "year": _year,
        "paper_url": _PAPER_URLS.get(_key),
        "author_code_url": _card.get("author_source"),
        "pinned_source_sha": (
            "d57e921a08930b960d07adcbec3a839bd228fe94"
            if _key in ("B5a", "B5b") else None),
        "method_label": _card["name"],
        "reference_alg_or_equation": _card.get(
            "equation", _card.get("equations", _card.get("blocker"))),
        "faithful_or_adapted": _card["status"],
        "exact_differences": _card.get(
            "adaptations", _card.get("blocker", "No external adaptation; registered internal control.")),
        "input_permissions": _card.get("information", ["shared score only"]),
        "initialization_and_warmup": _card.get(
            "initialization", _card.get("defaults", "See protocol; not separately parameterized.")),
        "declared_parameters_and_grid": _card.get(
            "defaults", "No parameter grid executed in Stage 0."),
        "random_seed_policy": "No method tuning or empirical seed search in Stage 0; query/support seeds are locked in the protocol.",
        "independent_unit_test_reference": _REFERENCE_TESTS[_key],
        "missing_and_insufficient_data_behavior": _MISSING_DATA[_key],
        "numerical_edge_cases": (
            "See independent_unit_test_reference; external reference cases remain blocked."
            if _card["status"] == "BLOCKED_FIDELITY" else
            "See independent_unit_test_reference; no population guarantee is claimed."),
        "complexity_and_runtime": "Not benchmarked; per-run operation counts are emitted where implemented.",
        "claimed_guarantee": _card.get("no_guarantee", "None claimed."),
        "invalidated_assumptions": _card.get(
            "no_guarantee", "No external theorem is claimed by this registered control."),
        "source_and_test_hash": {
            "source_hash_manifest": "source_hashes.json",
            "test_results_manifest": "test_results.json",
            "reference_test_ids": _REFERENCE_TESTS[_key],
        },
    })


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
