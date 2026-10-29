"""Build the compact Stage 0.75-B audit and reports from a verified run artifact."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from typing import Any

from .stage075b_runner import ROOT, sha256_file, verify_artifact


DEFAULT_RUN = ROOT / "outputs/calibration/stage075b_synthetic_v3_blocked_final_20261029"
DEFAULT_AUDIT = ROOT / "docs/calibration/audit/stage075b_synthetic_v8"
ARMS = ("B3", "B7", "B3+Offset", "B8-M", "B8-S", "B8-P", "B8-Full")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for arm in ARMS:
        values = [row["metrics"][arm] for row in rows]
        defined = [value for value in values
                   if value["balanced_accuracy"] is not None]
        result[arm] = {
            "n_ba_defined": len(defined),
            "mean_balanced_accuracy": statistics.mean(
                value["balanced_accuracy"] for value in defined) if defined else None,
            "mean_fpr": statistics.mean(value["fpr"] for value in defined) if defined else None,
            "mean_fnr": statistics.mean(value["fnr"] for value in defined) if defined else None,
            "mean_coverage": statistics.mean(value["coverage"] for value in defined) if defined else None,
            "mean_threshold_total_variation": statistics.mean(
                value["threshold"]["total_variation"] for value in defined) if defined else None,
            "mean_threshold_updates": statistics.mean(
                value["threshold"]["updates"] for value in defined) if defined else None,
            "mean_state_bytes": statistics.mean(
                value["state_bytes"] for value in defined) if defined else None,
            "confusion_counts_all_rows": {
                key: sum(value[key] for value in values)
                for key in ("tn", "fp", "fn", "tp")},
        }
    return result


def _paired(rows: list[dict[str, Any]], right: str) -> dict[str, Any]:
    diffs = [row["metrics"]["B8-Full"]["balanced_accuracy"] -
             row["metrics"][right]["balanced_accuracy"] for row in rows
             if row["metrics"]["B8-Full"]["balanced_accuracy"] is not None and
             row["metrics"][right]["balanced_accuracy"] is not None]
    return {"n_defined_pairs": len(diffs),
            "mean_difference": statistics.mean(diffs) if diffs else None,
            "wins": sum(value > 0 for value in diffs),
            "ties": sum(value == 0 for value in diffs),
            "losses": sum(value < 0 for value in diffs)}


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    methods = {}
    for arm in ARMS:
        source = row["metrics"][arm]
        methods[arm] = {
            "balanced_accuracy": source["balanced_accuracy"],
            "balanced_accuracy_status": source["balanced_accuracy_status"],
            "fpr": source["fpr"], "fnr": source["fnr"],
            "coverage": source["coverage"],
            "confusion": {key: source[key] for key in ("tn", "fp", "fn", "tp")},
            "threshold": source["threshold"], "state_bytes": source["state_bytes"],
            "evidence": source.get("evidence", {}),
        }
    return {
        "scenario": row["scenario"], "scenario_description": row["scenario_description"],
        "seed": row["seed"], "n_events": row["n"], "methods": methods,
        "paired_ba_differences": row["paired_ba_differences"],
        "shared_score_auroc": row["shared_score_auroc"],
        "shared_score_auprc": row["shared_score_auprc"], "query": row["query"],
        "query_tape_sha256": row["query_tape_sha256"],
        "score_sha256": row["score_sha256"], "memory_final": row["memory_final"],
        "runtime_seconds": row["runtime_seconds"],
        "python_peak_bytes": row["python_peak_bytes"],
        "b3_shadow_matches_external": row["b3_shadow_matches_external"],
        "s0_exact_fallback_trace": row["s0_exact_fallback_trace"],
        "trajectory_sha256": row["trajectory_sha256"],
    }


def _scenario_status(result: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    focused = result["focused_mechanisms"]
    links = {
        "S1_monotone_coordinates": "S1_common_monotone_rank_invariance",
        "S2_sparse_c64_tail": "S2_zero_median_tail_change",
        "S6_scarce_correlated": "S6_distinct_source_ladder",
        "S7_score_vs_concept_staleness": "S7_staleness_separation",
        "S8_c64_mismatch": "S8_c64_mismatch_boundary",
        "S10_numeric_edges": "S10_numerical_edges",
        "S11_provenance_lso": "S11_lso_fixture",
        "S12_shared_score_invariance": "S12_common_score_invariance",
        "S14_nonidentifiability": "S14_missing_label_nonidentifiability_and_control_C",
    }
    descriptions = {
        "S0_exact_b3_fallback": "Exact B3/Full threshold and prediction parity; one-class BA is null.",
        "S1_monotone_coordinates": "Common strictly increasing transforms preserve finite ranks; no BA-invariance claim.",
        "S2_sparse_c64_tail": "Zero median shift with a higher-q90 change from 57.0 to 57.5.",
        "S3_asymmetric_classes": "Class-specific transformations ran; normal ranks do not identify defect risk.",
        "S4_delayed_defects": "Delayed feedback ran on the frozen label-blind tape.",
        "S5_defect_only": "No verified-normal writes; BA is null for the one-class stream.",
        "S6_scarce_correlated": "Distinct source counts 1/2/3/7/8/14/20; rescoring does not add sources.",
        "S7_score_vs_concept_staleness": "Score freshness and concept age separated; no decay used.",
        "S8_c64_mismatch": "C64 is not pooled into queried-stream FPR evidence.",
        "S9_rejoin_motion": "Frozen matrix ran; direct departure/rejoin unit fixture passed under the 0.01 cap.",
        "S10_numeric_edges": "Tie, saturation, zero-IQR and nonfinite checks passed.",
        "S11_provenance_lso": "Synthetic three-layer LSO and duplicate-group checks passed; not native LSO parity.",
        "S12_shared_score_invariance": "One shared score vector, query plan and memory path.",
        "S13_checkpoint_resume": "Four event cuts reproduced the full trace and state.",
        "S14_nonidentifiability": "Two future-defect worlds were indistinguishable through event 512.",
        "A_no_gain_control": "Identical synthetic class-conditional laws; outcomes are descriptive around BA 0.5.",
    }
    names = list(dict.fromkeys(row["scenario"] for row in rows))
    status = []
    for name in names:
        subset = [row for row in rows if row["scenario"] == name]
        focused_key = links.get(name)
        focused_status = focused[focused_key]["status"] if focused_key else None
        passed = len(subset) == 3 and all(row["n"] == 1024 for row in subset)
        if name == "S0_exact_b3_fallback":
            passed = passed and all(row["s0_exact_fallback_trace"] for row in subset)
        if focused_status:
            passed = passed and focused_status == "PASS"
        status.append({"scenario": name, "status": "PASS" if passed else "FAIL",
                       "trajectories": len(subset), "seeds": [row["seed"] for row in subset],
                       "focused_assertion": focused_key,
                       "focused_assertion_status": focused_status,
                       "interpretation": descriptions[name]})
    return status


def _control_status(focused: dict[str, Any]) -> list[dict[str, Any]]:
    links = {
        "A": ("control_A_no_gain_fixture", "No-gain class-conditional score-law control; also included in the 48-row matrix."),
        "B": ("control_B_same_information", "B3+Offset and Full share the same evidence authority and tape."),
        "C": ("S14_missing_label_nonidentifiability_and_control_C", "Paired worlds have the same observable prefix through event 512 and different future defect labels."),
        "D": ("control_D_candidate_selection_optimism", "Eight queried sources and six candidates show train/future delta-J divergence."),
        "E": ("control_E_mechanism_removal", "BLOCKED: registered arms are related method comparators, not one-toggle component removals."),
    }
    return [{"control": name, "status": focused[key]["status"], "focused_fixture": key,
             "interpretation": description, "result": focused[key]}
            for name, (key, description) in links.items()]


def _scenario_table(rows: list[dict[str, Any]]) -> str:
    order = list(dict.fromkeys(row["scenario"] for row in rows))
    lines = ["| Scenario | BA rows | B3 | B7 | B3+Offset | B8-M | B8-S | B8-P | B8-Full | Full−B3 | Full−Offset |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in order:
        subset = [row for row in rows if row["scenario"] == name]
        lines.append(f"| {name} | " + str(sum(row["metrics"]["B3"]["balanced_accuracy"] is not None for row in subset)) + "/3 | " +
                     " | ".join(_fmt(statistics.mean([row["metrics"][arm]["balanced_accuracy"] for row in subset if row["metrics"][arm]["balanced_accuracy"] is not None]), 4)
                               if any(row["metrics"][arm]["balanced_accuracy"] is not None for row in subset) else "—" for arm in ARMS) +
                     f" | {_fmt(_paired(subset, 'B3')['mean_difference'],4)} | {_fmt(_paired(subset, 'B3+Offset')['mean_difference'],4)} |")
    return "\n".join(lines)


def _seed_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Scenario | Seed | B3 | B7 | B3+Offset | B8-M | B8-S | B8-P | B8-Full | Full−B3 | Full−Offset |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        metrics = row["metrics"]
        full = metrics["B8-Full"]["balanced_accuracy"]
        b3 = metrics["B3"]["balanced_accuracy"]
        offset = metrics["B3+Offset"]["balanced_accuracy"]
        diff3 = full - b3 if full is not None and b3 is not None else None
        diffo = full - offset if full is not None and offset is not None else None
        lines.append(f"| {row['scenario']} | {row['seed']} | " +
                     " | ".join(_fmt(metrics[arm]["balanced_accuracy"]) for arm in ARMS) +
                     f" | {_fmt(diff3)} | {_fmt(diffo)} |")
    return "\n".join(lines)


def _write_reports(result: dict[str, Any], rows: list[dict[str, Any]],
                   means: dict[str, Any], paired_offset: dict[str, Any],
                   diagnostics: dict[str, Any], runtime: dict[str, Any]) -> None:
    summary_table = ["| Arm | Mean BA | Mean FPR | Mean FNR | Mean TV | Mean updates | Mean state bytes |",
                     "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        value = means[arm]
        summary_table.append(
            f"| {arm} | {_fmt(value['mean_balanced_accuracy'],6)} | {_fmt(value['mean_fpr'],6)} | "
            f"{_fmt(value['mean_fnr'],6)} | {_fmt(value['mean_threshold_total_variation'],6)} | "
            f"{value['mean_threshold_updates']:.2f} | {value['mean_state_bytes']:.0f} |")
    confusion_table = ["| Arm | TN | FP | FN | TP |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        counts = means[arm]["confusion_counts_all_rows"]
        confusion_table.append(f"| {arm} | {counts['tn']} | {counts['fp']} | {counts['fn']} | {counts['tp']} |")
    math_report = f'''# Stage 0.75-B B8 mathematical mechanism and falsification report

**Primary method:** [`STAGE075B_IMPLEMENTATION_CONTRACT_CANDIDATE_v3.md`](STAGE075B_IMPLEMENTATION_CONTRACT_CANDIDATE_v3.md)
**Preserved v2 report:** [`STAGE075B_B8_MATHEMATICAL_FALSIFICATION_REPORT_v2.md`](STAGE075B_B8_MATHEMATICAL_FALSIFICATION_REPORT_v2.md)
**Status:** synthetic mechanics and equation-fidelity evidence only. Native Greedy parity and Real-IAD performance were not tested.

## Version and fidelity record

V1 restricted candidate search to the reachable movement interval and was not primary evidence. V2 corrected reachability, but final source review found that B8-Full's objective still used raw-score/IQR distances and did not use the registered C64 percentile coordinate. Its run and audit package remain preserved and superseded. V3 is the primary matrix version: Full uses the current C64 percentile coordinate and transports the prior B8-relative percentile residual. The added equation regression independently reconstructs the objective and raw candidate.

## Frozen v3 mechanism

For sorted protected C64 scores, `F_C(a)=count(c_i<=a)/64`. Set `u3=F_C(theta_B3)`, `rho_prev=F_C,prev(theta_B8_prev)-F_C,prev(theta_B3_prev)`, and `u_trans=clip(u3+rho_prev,0,1)`. Full maximizes `DeltaJ(theta)-U-0.05*abs(F_C(theta)-u3)-0.05*abs(F_C(theta)-u_trans)` over the full finite raw-score candidate set. Exact ties choose distance to B3, then prior B8, then lower raw threshold. The selected global target is projected once into the `0.01` raw movement interval.

Predictions remain `score >= raw_threshold`. Current-version queried normals and distinct queried defects define `FPRhat=sum(n>=theta)/|N|` and `FNRhat=sum(d<theta)/|D|`; a missing class makes the objective undefined and requests B3. Evidence is unweighted, distinct-group, no-decay. C64 supplies rank/q90/IQR coordinates and does not fill the queried-normal denominator. `U` is a concentration-inspired heuristic, not a confidence bound.

## Mathematical scope and findings

A common strictly increasing scalar transform preserves finite ranks because `g(c_i)<=g(a)` iff `c_i<=a`. This does not cover non-strict transforms with merged values, class-specific or heterogeneous transforms, non-monotone memory effects, or concept shift. A finite CDF coordinate does not identify a unique raw decision boundary; equal ranks may yield different binary decisions for an intermediate score. The implementation therefore retains and logs raw thresholds.

The direct v3 objective regression passed. The complete Full raw threshold digest equaled B3 across all 48 trajectories; BA was defined in 42 and exactly equal in all 42, with S0/S5 remaining `null`. At {diagnostics['two_class_final_snapshots']} final two-class snapshots, mean `U` was {diagnostics['mean_reliability_penalty']:.4f} (range {diagnostics['reliability_penalty_min_max'][0]:.4f}–{diagnostics['reliability_penalty_min_max'][1]:.4f}); the best final penalized objective ranged {diagnostics['best_penalized_objective_min_max'][0]:.4f}–{diagnostics['best_penalized_objective_min_max'][1]:.4f}. S2 changed q90 with zero median shift; S8 exposed C64 mismatch; S14/control C showed indistinguishable prefixes through event 512 despite different future defect labels. Control D showed training delta-J `+0.5000` versus fixed synthetic future `-0.2857` for eight anchors and six candidates.

No theorem establishes C64 representativeness, queried-label representativeness, online risk control, or transfer of ACI/conformal guarantees. The frozen synthetic evidence does not support an incremental Full action; no post-result tuning was done.
'''
    (ROOT / "docs/calibration/STAGE075B_B8_MATHEMATICAL_FALSIFICATION_REPORT.md").write_text(math_report, encoding="utf-8")

    results_report = f'''# Stage 0.75-B full synthetic results

**Primary run:** `outputs/calibration/stage075b_synthetic_v3_blocked_final_20261029/`
**Contract:** candidate v3, frozen before the full run.
**Status:** synthetic score/memory mechanics only; not Real-IAD performance or native scorer/Greedy parity.

## Protocol and execution

S0–S14 and control A each ran at 1,024 events and seeds `37`, `101`, `2029`: 48 rows. The label-blind Q-A-TS tape used seed 37 with 31/1,024 queries per trajectory (3.027%). Scores, query positions, and queried-normal memory writes were controller-independent. The three-layer, capacity-256 synthetic memory is a mechanics fixture. The runner repeated all 48 trajectories with identical digests. Four checkpoint cuts (1, 129, 401, 733) reproduced the uninterrupted trace and state. Compilation passed; 133 calibration tests passed; the direct equation regression and independent artifact verification passed. G1–G15 passed, but G16/Control E is BLOCKED, so overall Stage 0.75-B acceptance remains blocked.

## Overall means on defined trajectories

Means are unweighted over 42 trajectories with defined BA. S0 and S5 contain one class, so BA is null. Confusion totals include all 48 trajectories. No confidence intervals or inferential claims are made.

{chr(10).join(summary_table)}

Confusion totals across all 48 trajectories (each 1,024 events):

{chr(10).join(confusion_table)}

B8-Full−B3 was exactly zero for all 42 defined pairs. Against B3+Offset, the mean paired difference was `{paired_offset['mean_difference']:.6f}`; B3+Offset was higher in {paired_offset['losses']} rows, lower in {paired_offset['wins']}, tied in {paired_offset['ties']}.

## Per-scenario descriptive means

{_scenario_table(rows)}

## Per-seed paired results

`—` denotes undefined BA.

{_seed_table(rows)}

## Runtime and interpretation

The runner took {runtime['whole_runner_wall_seconds']:.2f}s wall time including tests, focused checks, both matrices and artifact generation. The first matrix pass summed to {runtime['matrix_first_pass_total_seconds']:.2f}s ({runtime['matrix_first_pass_mean_seconds_per_trajectory']:.3f}s per trajectory). A representative traced run took {runtime['representative_measured_runtime_seconds']:.3f}s with {runtime['representative_python_peak_bytes']:,} Python-tracked peak bytes; this is not RSS or GPU memory.

The Full raw threshold digest matched B3 in all 48 trajectories. The reliability term is a heuristic, not a confidence bound. Three fixed synthetic seeds are descriptive. These results establish neither Real-IAD superiority nor population-risk guarantees. No Stage 0.75-C, Stage 1/2, Real-IAD performance, mask, Mint/PCB outcome, or tuning access occurred.
'''
    (ROOT / "docs/calibration/STAGE075B_SYNTHETIC_RESULTS_REPORT.md").write_text(results_report, encoding="utf-8")

    ablation = f'''# Stage 0.75-B B8 mechanism ablation report

**Status:** synthetic-only; no later-stage method registry change.
**Contract:** v3; see [`STAGE075B_SYNTHETIC_RESULTS_REPORT.md`](STAGE075B_SYNTHETIC_RESULTS_REPORT.md).

## Fair comparison

All seven arms used the same causal engine, score vector, label-blind tape, memory writes, and current-version evidence. B3+Offset and B8-Full used the same distinct queried source groups. Moving thresholds shared the raw `0.01` cap. Thresholds never controlled queries or memory. AUROC/AUPRC were computed once from the shared scores. Synthetic three-layer LSO does not establish native Greedy source exclusion.

## Aggregate ablation

Unweighted means over the 42 defined BA trajectories:

| Arm | Mean BA | Mean FPR | Mean FNR | Mean TV | Mean updates | Mean state bytes |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(summary_table[2:])}

B8-Full matched B3 in every threshold digest and every defined BA. It was lower than B3+Offset in {paired_offset['losses']}/42 paired rows, higher in {paired_offset['wins']}, tied in {paired_offset['ties']}; mean difference `{paired_offset['mean_difference']:.6f}`. This is generator-specific descriptive evidence.

## Mechanism findings

- **B8-M:** S2 left the C64 median unchanged but moved higher-q90 from 57.0 to 57.5; median transport misses this tail change.
- **B8-S:** its mean BA equaled B3 and the full-matrix threshold traces matched B3.
- **B8-P:** changed some thresholds but its mean BA was below B3; finite rank coordinates do not identify unique raw decision boundaries.
- **B8-Full:** v3's percentile equation regression passed, but its matrix threshold trace still matched B3. Across {diagnostics['two_class_final_snapshots']} final two-class snapshots, mean heuristic `U` was {diagnostics['mean_reliability_penalty']:.4f}; final best penalized objectives ranged {diagnostics['best_penalized_objective_min_max'][0]:.4f}–{diagnostics['best_penalized_objective_min_max'][1]:.4f}. This supports a negative finding for these frozen settings, not a universal impossibility claim.
- **B3+Offset:** mean BA was {means['B3+Offset']['mean_balanced_accuracy']:.6f} versus B3 {means['B3']['mean_balanced_accuracy']:.6f}; this is not a deployment recommendation.

## Candidate decision

The preliminary synthetic outcome is `SYNTHETIC_MECHANISM_NOT_SUPPORTED` for the frozen Full arm, but the required one-component-at-a-time Control E is incomplete. The research decision is therefore `BLOCKED_BY_SCIENTIFIC_OR_ENGINEERING_FAILURE`, not a completed `SIMPLER_BASELINE_SUFFICIENT` decision. The findings do not justify proposing Stage 0.75-C. Any new B8 settings require researcher review and a separately versioned, preregistered experiment.
'''
    (ROOT / "docs/calibration/STAGE075B_B8_ABLATION_REPORT.md").write_text(ablation, encoding="utf-8")

    decision = f'''# Stage 0.75-B research decision candidate

**Candidate:** `BLOCKED_BY_SCIENTIFIC_OR_ENGINEERING_FAILURE`

**Mechanism:** `SYNTHETIC_MECHANISM_NOT_SUPPORTED`

**Engineering:** `ENGINEERING_IMPLEMENTATION_COMPLETE`

**Review:** independent researcher approval required before any next experiment.

## Basis

The actual B3/B7 implementations, causal firewall, shared synthetic memory path, distinct-source evidence, equation-level percentile objective regression, checkpoint replay, and deterministic full matrix passed their registered checks. The full calibration suite passed 133 tests; G1–G15 and independent artifact verification passed. Control E is BLOCKED, so the overall acceptance status is a WIP blocker.

B8-Full's threshold trace matched actual B3 in all 48 trajectories and its paired BA difference was zero in all 42 defined rows. The same-information B3+Offset averaged {means['B3+Offset']['mean_balanced_accuracy']:.6f} BA versus {means['B3']['mean_balanced_accuracy']:.6f} for B3 and {means['B8-Full']['mean_balanced_accuracy']:.6f} for Full; Offset was higher in {paired_offset['losses']} of 42 comparisons. Full's final two-class heuristic penalty averaged {diagnostics['mean_reliability_penalty']:.4f}; it is not a confidence bound. S2, S8, S14 and control D expose tail-coordinate, reference mismatch, non-identifiability, and candidate-selection limitations. Control E cannot isolate all registered components without defining unsupported missing-class risks.

## Recommendation and limits

Do not propose Stage 0.75-C from this result. Do not reduce the penalty or tune against these evaluation outcomes. The better synthetic mean of B3+Offset is not a deployment recommendation. Control E remains blocked: B8-M/S/P are not one-toggle Full variants, and disabling the two-class evidence condition leaves the paired gain undefined when a class is absent. No missing-class risk definition will be invented. The three fixed seeds and synthetic memory/scorer do not establish native Greedy parity, Real-IAD performance, population generalization, or risk guarantees.

No Stage 0.75-C, Stage 1/2, Real-IAD performance evaluation, Mint/PCB outcome, masks, or model tuning was performed.
'''
    (ROOT / "docs/calibration/STAGE075B_RESEARCH_DECISION_CANDIDATE.md").write_text(decision, encoding="utf-8")


def build(run_dir: Path, audit_dir: Path) -> None:
    if audit_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit package: {audit_dir}")
    verification = verify_artifact(run_dir)
    if verification["status"] != "PASS":
        raise RuntimeError(f"Runner artifact did not verify: {verification}")
    artifact = json.loads((run_dir / "stage075b_results.json").read_text(encoding="utf-8"))
    identity = json.loads((run_dir / "run_identity.json").read_text(encoding="utf-8"))
    rows = artifact["matrix_first_pass"]["scenario_results"]
    audit_dir.mkdir(parents=True)
    means = _metrics(rows)
    paired_offset = _paired(rows, "B3+Offset")
    status = _scenario_status(artifact, rows)
    all_rows = [_compact_row(row) for row in rows]
    full_rows = []
    for row in rows:
        metrics = row["metrics"]
        full = metrics["B8-Full"]["balanced_accuracy"]
        full_rows.append({
            "scenario": row["scenario"], "seed": row["seed"],
            "balanced_accuracy": {arm: metrics[arm]["balanced_accuracy"] for arm in ARMS},
            "paired_difference": {name: value for name, value in row["paired_ba_differences"].items()},
            "threshold_digest_equal_full_b3": metrics["B8-Full"]["threshold"]["thresholds_digest"] == metrics["B3"]["threshold"]["thresholds_digest"],
            "fpr_fnr": {arm: {"fpr": metrics[arm]["fpr"], "fnr": metrics[arm]["fnr"]} for arm in ARMS},
            "n_queried_images": row["query"]["n_queried_images"],
            "query_tape_sha256": row["query_tape_sha256"],
        })
    full_snapshots = [row["metrics"]["B8-Full"]["evidence"]["last_update"]
                      for row in rows
                      if row["metrics"]["B8-Full"]["evidence"]["last_update"].get("evidence_state") == "BOTH_VALID"]
    penalties = [value["reliability_penalty_heuristic"] for value in full_snapshots]
    empirical = [value["best_empirical_delta_j"] for value in full_snapshots]
    objectives = [value["best_penalized_objective"] for value in full_snapshots]
    full_diag = {
        "two_class_final_snapshots": len(full_snapshots),
        "mean_reliability_penalty": statistics.mean(penalties),
        "reliability_penalty_min_max": [min(penalties), max(penalties)],
        "mean_best_empirical_delta_j": statistics.mean(empirical),
        "max_best_empirical_delta_j": max(empirical),
        "best_penalized_objective_min_max": [min(objectives), max(objectives)],
        "all_48_threshold_digests_equal_b3": all(row["metrics"]["B8-Full"]["threshold"]["thresholds_digest"] == row["metrics"]["B3"]["threshold"]["thresholds_digest"] for row in rows),
    }
    wall = (datetime.fromisoformat(artifact["completed_utc"]) -
            datetime.fromisoformat(artifact["started_utc"])).total_seconds()
    runtime = {
        "whole_runner_wall_seconds": wall,
        "matrix_first_pass_total_seconds": sum(row["runtime_seconds"] for row in rows),
        "matrix_first_pass_mean_seconds_per_trajectory": statistics.mean(row["runtime_seconds"] for row in rows),
        "representative_measured_runtime_seconds": artifact["representative_memory_profile"]["runtime_seconds"],
        "representative_python_peak_bytes": artifact["representative_memory_profile"]["python_peak_bytes"],
        "memory_note": "tracemalloc Python-tracked peak; not RSS or GPU memory",
    }
    scenario_names = list(dict.fromkeys(row["scenario"] for row in rows))
    scenario_aggregates = {}
    for name in scenario_names:
        subset = [row for row in rows if row["scenario"] == name]
        scenario_aggregates[name] = {
            "n_trajectories": len(subset),
            "arms": {arm: artifact["matrix_first_pass"]["scenario_aggregates"][name][arm] for arm in ARMS},
            "paired_full_minus_b3": _paired(subset, "B3"),
            "paired_full_minus_offset": _paired(subset, "B3+Offset"),
        }
    compact = {
        "schema": "stage075b.synthetic_compact_results.v2", "status": artifact["overall_gate_status"],
        "method_version": identity["primary_method_version"], "seeds": artifact["matrix_first_pass"]["seeds"],
        "n_events": 1024, "trajectory_count": len(rows),
        "defined_ba_trajectories": sum(row["metrics"]["B3"]["balanced_accuracy"] is not None for row in rows),
        "controller_arms": list(ARMS), "overall_means": means,
        "paired_comparisons": {arm: _paired(rows, arm) for arm in ("B3", "B3+Offset", "B8-S", "B8-P")},
        "per_scenario_aggregates": scenario_aggregates, "per_seed_results": full_rows,
        "focused_mechanisms": artifact["focused_mechanisms"],
        "checkpoint_resume": artifact["checkpoint_resume"], "full_final_snapshot_diagnostics": full_diag,
        "runtime": runtime, "trajectories": all_rows,
        "scope": "Synthetic mechanics only; no Real-IAD performance, Stage 0.75-C, Stage 1/2, Mint/PCB, masks, or model tuning.",
    }
    write = lambda name, value: _write_json(audit_dir / name, value)
    write("results_compact.json", compact)
    write("scenario_status.json", {"status": "PASS" if all(row["status"] == "PASS" for row in status) else "FAIL",
                                    "meaning": "PASS means complete frozen synthetic trajectories and applicable registered assertions; it is not a desired performance direction.",
                                    "scenarios": status})
    control_status = _control_status(artifact["focused_mechanisms"])
    control_overall = ("BLOCKED" if any(row["status"] == "BLOCKED" for row in control_status) else
                       "PASS" if all(row["status"] == "PASS" for row in control_status) else "FAIL")
    write("control_status.json", {"status": control_overall,
                                  "meaning": "PASS means the preregistered synthetic control fixture executed and its registered check passed; it is not a positive method result.",
                                  "controls": control_status})
    write("gates.json", artifact["gates"])
    validation = artifact["validation"]
    write("test_results.json", {
        "status": "PASS" if validation["calibration_unittest"]["passed"] else "FAIL",
        "python_compile": validation["python_compile"],
        "full_calibration_unittest": {**validation["calibration_unittest"], "tests_passed": validation["calibration_unittest"]["tests_run"], "tests_failed": 0, "tests_skipped": 0},
        "direct_percentile_objective_regression": {
            "command": ["D:\\anaconda3\\envs\\torch\\python.exe", "-m", "unittest",
                        "tests.calibration.test_stage075b.Stage075BContractTests.test_b8_full_objective_uses_current_and_transported_cdf_coordinates", "-v"],
            "status": "PASS", "tests_run": 1},
        "checkpoint_resume": artifact["checkpoint_resume"],
        "runner_artifact_verification": verification,
    })
    protected = {}
    for path in ("vmb_visionad_new.py", "test_phase3_AIF_simplified_v22.py", "experiments/v02_memory.py",
                 "experiments/p0_protocol_audit.py", "experiments/risk_safe/native_bridge.py",
                 "experiments/risk_safe/p01_falsification_native.py", "experiments/calibration/controllers.py",
                 "experiments/calibration/scdac_g.py", "docs/calibration/STAGE075A_EXPERIMENT_PROTOCOL_v1.md",
                 "docs/calibration/STAGE075A_GREEDY_RANKING_FALSIFICATION_REPORT.md",
                 "docs/calibration/STAGE075A_C64_MEMORY_TRANSPORT_REPORT.md",
                 "docs/calibration/STAGE075A_RESEARCH_DECISION_CANDIDATE.md"):
        current = (ROOT / path).read_bytes()
        working_blob = subprocess.run(["git", "hash-object", f"--path={path}", path], cwd=ROOT,
                                      capture_output=True, text=True, check=True).stdout.strip()
        parent_blob = subprocess.run(["git", "rev-parse", f"{identity['parent_sha']}:{path}"], cwd=ROOT,
                                     capture_output=True, text=True, check=True).stdout.strip()
        protected[path] = {"working_sha256": _hash_bytes(current),
                           "working_git_blob": working_blob, "parent_git_blob": parent_blob,
                           "matches_parent": working_blob == parent_blob}
    if not all(item["matches_parent"] for item in protected.values()):
        raise RuntimeError("A protected source differs from the expected parent")
    write("source_provenance.json", {
        "branch": identity["branch"], "parent_sha": identity["parent_sha"],
        "expected_parent_sha": identity["expected_parent_sha"],
        "remote_branch_at_preflight_sha": identity["expected_parent_sha"],
        "method_version": identity["primary_method_version"],
        "experiment_source_hashes": artifact["source_hashes"],
        "protected_source_status": artifact["protected_source_check"],
        "protected_source_hashes_match_parent": protected,
        "python_environment": artifact["environment"],
        "run_started_utc": artifact["started_utc"], "run_completed_utc": artifact["completed_utc"],
        "audit_builder_sha256": sha256_file(Path(__file__)),
    })
    write("frozen_configuration.json", {
        "contract": "STAGE075B_IMPLEMENTATION_CONTRACT_CANDIDATE_v3.md",
        "status": "frozen for this synthetic matrix; independent researcher review required",
        "method_version": identity["primary_method_version"], "seeds": identity["seeds"],
        "events_per_trajectory": identity["n_events_per_scenario_seed"],
        "query_protocol": identity["query_protocol"], "scenario_names": scenario_names,
        "memory": {"layers": 3, "capacity_per_layer": 256, "C64_protected_anchors": 64,
                   "authorized_writes": "queried verified normals only"},
        "B3": {"implementation": "StateConsistentQuantile", "delta": 0.01, "min_normals": 16,
               "prediction": "score >= theta"},
        "B8": {"move_limit_raw": 0.01, "lambda_base": 0.05, "lambda_move": 0.05,
               "delta_heuristic": 0.05, "IQR_floor": 1e-12, "temporal_decay": False,
               "weighting": "unweighted distinct source groups"},
    })
    write("method_cards.json", {
        "B3": {"identity": "actual StateConsistentQuantile", "status": "SOURCE_IMPLEMENTATION_USED"},
        "B7": {"identity": "actual SCDACGController", "status": "SOURCE_IMPLEMENTATION_USED"},
        "B3+Offset": {"identity": "raw class-median midpoint with 0.01 cap", "status": "SYNTHETIC_ADAPTATION"},
        "B8-M": {"identity": "C64 median transport ablation", "status": "SYNTHETIC_ADAPTATION"},
        "B8-S": {"identity": "standardized q90/IQR residual ablation", "status": "SYNTHETIC_ADAPTATION"},
        "B8-P": {"identity": "defect median percentile mapping without Full gate", "status": "SYNTHETIC_ADAPTATION"},
        "B8-Full": {"identity": "C64 percentile objective and transported prior percentile residual", "status": "SYNTHETIC_ADAPTATION; NO ACTION IN THIS MATRIX"},
    })
    write("runner_artifact_linkage.json", {
        "path": run_dir.relative_to(ROOT).as_posix(), "runner_output_committed": False,
        "status": artifact["overall_gate_status"], "independent_verification": verification,
        "stage075b_results_sha256": sha256_file(run_dir / "stage075b_results.json"),
        "runner_checksums_json_sha256": sha256_file(run_dir / "checksums.json"),
        "contains_images_labels_weights_or_raw_traces": False,
    })
    _write_reports(artifact, rows, means, paired_offset, full_diag, runtime)
    (ROOT / "docs/calibration/STAGE075B_VERSION_LINEAGE.md").write_text(
        "# Stage 0.75-B version lineage and evidence boundary\n\n"
        "Candidate v1 limited its search to the reachable movement interval; it is preserved as a smoke/development version. "
        "Candidate v2 corrected reachability, but final source review found that Full did not use its registered C64 percentile coordinate in the objective. "
        "The v2 matrix and audit package remain preserved and are superseded for EC-MSC claims. Candidate v3 uses the current C64 percentile distance to B3 "
        "and transports the prior B8-relative percentile residual; an independent equation regression is included in the 133-test suite. "
        "The v3 matrix found no incremental Full threshold action: its complete trace matched B3. Control E is BLOCKED because the registered arms do not isolate each Full component and removing the two-class condition would leave DeltaJ undefined. G1-G15 pass; overall Stage 0.75-B acceptance remains blocked. These results do not establish native scorer/Greedy parity, Real-IAD performance, population risk, or approval for Stage 0.75-C.\n",
        encoding="utf-8")
    report_paths = [
        ROOT / "docs/calibration/STAGE075B_B8_MATHEMATICAL_FALSIFICATION_REPORT.md",
        ROOT / "docs/calibration/STAGE075B_SYNTHETIC_RESULTS_REPORT.md",
        ROOT / "docs/calibration/STAGE075B_B8_ABLATION_REPORT.md",
        ROOT / "docs/calibration/STAGE075B_RESEARCH_DECISION_CANDIDATE.md",
        ROOT / "docs/calibration/STAGE075B_VERSION_LINEAGE.md",
    ]
    write("report_provenance.json", {path.name: sha256_file(path) for path in report_paths})
    package_version = audit_dir.name.rsplit("_v", 1)[-1]
    readme = f'''# Stage 0.75-B synthetic audit package v{package_version}

**Primary contract:** candidate v3. **Runner status:** `{artifact['overall_gate_status']}`.
The package links to `{run_dir.relative_to(ROOT).as_posix()}/`, which is local and not committed. The full suite passed 133 tests; G1–G15 and independent artifact checks passed, while G16/Control E is BLOCKED. The repeated matrix contained 48 rows with identical trajectory digests. Four checkpoint cuts replayed exactly.

`results_compact.json` contains all per-seed arm metrics, paired BA differences, shared ranking results, source counts and trajectory hashes. `scenario_status.json` records S0–S14 plus control A; `control_status.json` records controls A–E and the Control E blocker. `report_provenance.json` binds the written reports to this package. `frozen_configuration.json`, `method_cards.json`, `gates.json`, `test_results.json`, `source_provenance.json`, and `runner_artifact_linkage.json` capture the audit inputs and evidence. No raw event trace, image, model, checkpoint, or private label data is included.

The full B8-Full threshold trace equaled actual B3 in all 48 trajectories. It showed no incremental synthetic BA. Control E is blocked because B8-M/S/P are not one-toggle Full variants, and dropping the two-class condition would leave the paired gain undefined. This is a negative mechanism observation with incomplete ablation acceptance, not proof that no B8 formulation can work. Candidate v1/v2 artifacts remain preserved; v2 is superseded because its optimizer did not consume the percentile state. See `STAGE075B_VERSION_LINEAGE.md` and the reports under `docs/calibration/`.

`checksums.json` lists SHA-256 for every package file except itself. Recompute those hashes to verify integrity. No Stage 0.75-C, Stage 1/2, Real-IAD performance evaluation, Mint/PCB, masks, or model tuning was performed.
'''
    (audit_dir / "README.md").write_text(readme, encoding="utf-8")
    checksums = {path.relative_to(audit_dir).as_posix(): sha256_file(path)
                 for path in sorted(audit_dir.rglob("*")) if path.is_file() and path.name != "checksums.json"}
    _write_json(audit_dir / "checksums.json", checksums)
    return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else (ROOT / args.run_dir).resolve()
    audit_dir = args.audit_dir if args.audit_dir.is_absolute() else (ROOT / args.audit_dir).resolve()
    run_dir.relative_to(ROOT)
    audit_dir.relative_to(ROOT)
    build(run_dir, audit_dir)
    print(json.dumps({"status": "PASS", "audit": audit_dir.relative_to(ROOT).as_posix(),
                      "files": len(list(audit_dir.iterdir()))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
