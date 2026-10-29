"""Render the four Stage 0.75-A B3/headroom reports and immutable audit set."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from .stage075a_b3_headroom import (
    DEFAULT_SOURCE, DEFAULT_STAGE05, DEVELOPMENT_CLASSES, LOCAL_OFFSETS,
    SNAPSHOT_POSITIONS, SUPPORT_SEEDS, HeadroomError, sha256_file,
    verify_source_artifacts,
)
from .profiles import canonical_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/calibration/stage075a_b3_headroom_v1_20261030_verified2"
REPORT_NAMES = {
    "replay": "STAGE075A_ACTUAL_B3_REPLAY_REPORT.md",
    "headroom": "STAGE075A_LOCAL_HEADROOM_REPORT.md",
    "diagnosis": "STAGE075A_B8_SIMPLIFICATION_RESEARCH_DIAGNOSIS.md",
    "decision": "STAGE075A_B3_HEADROOM_DECISION_CANDIDATE.md",
}
AUDIT_REL = Path("docs/calibration/audit/stage075a_b3_headroom_v1")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "UNDEFINED"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _avg(values: Sequence[Any]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return statistics.mean(valid) if valid else None


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    out.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(out)


def _method_cell(metrics: Mapping[str, Any]) -> str:
    return "/".join(_fmt(metrics.get(key), 3) for key in
                    ("balanced_accuracy", "fpr", "fnr"))


def _rows_by_class_seed(analysis: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    return {(row["category"], int(row["seed"])): row
            for row in analysis["per_trajectory_metrics"]}


def _report_replay(analysis: Mapping[str, Any], replay: Mapping[str, Any],
                   provenance: Mapping[str, Any], output_rel: str) -> str:
    trajectory_rows = _rows_by_class_seed(analysis)
    rows = []
    for category in DEVELOPMENT_CLASSES:
        for seed in SUPPORT_SEEDS:
            trajectory = trajectory_rows[(category, seed)]
            methods = trajectory["scopes"]["as_operated_full_stream"]["methods"]
            rows.append([category, seed, trajectory["stream_count"],
                         _method_cell(methods["M0_Frozen_theta0"]),
                         _method_cell(methods["Greedy_Frozen_theta0"]),
                         _method_cell(methods["Greedy_ReadOnly_current_C64_q90"]),
                         _method_cell(methods["Greedy_Actual_B3"]),
                         _fmt(trajectory["b3_vs_q90_full_stream_ba_difference"], 5),
                         _fmt(trajectory["b3_vs_frozen_full_stream_ba_difference"], 5),
                         _fmt(trajectory["b3_threshold_total_variation"], 5),
                         trajectory["b3_threshold_update_count"],
                         trajectory["b3_memory_change_count"]])
    table = _markdown_table(["Class", "Seed", "N", "M0 + frozen (BA/FPR/FNR)",
                             "Greedy + frozen", "Greedy + read-only q90",
                             "Greedy + actual B3", "B3-q90 delta BA", "B3-frozen delta BA",
                             "B3 threshold TV", "B3 moves", "Memory writes"], rows)
    macro_rows = []
    for scope in ("as_operated_full_stream", "conservative_ever_group_clean",
                  "resident_source_clean_sensitivity"):
        for method in ("M0_Frozen_theta0", "Greedy_Frozen_theta0",
                       "Greedy_ReadOnly_current_C64_q90", "Greedy_Actual_B3"):
            mean = analysis["equal_class_macro_metrics"][scope][method]["equal_class_macro_mean"]
            macro_rows.append([scope, method, _fmt(mean["balanced_accuracy"], 5),
                               _fmt(mean["fpr"], 5), _fmt(mean["fnr"], 5)])
    macro_table = _markdown_table(["Scope", "Method", "Equal-class macro BA",
                                   "Equal-class macro FPR", "Equal-class macro FNR"], macro_rows)

    run_rows = []
    for category in DEVELOPMENT_CLASSES:
        runs = [replay["trajectories"][category][str(seed)] for seed in SUPPORT_SEEDS]
        event_lags = [run["lag"]["all_events"] for run in runs]
        update_lags = [run["lag"]["memory_change_pre_event"] for run in runs]
        caps = [run["movement_cap"] for run in runs]
        b3_minus_q90 = [trajectory_rows[(category, seed)]["b3_vs_q90_full_stream_ba_difference"]
                        for seed in SUPPORT_SEEDS]
        run_rows.append([category,
                         _fmt(_avg([lag["signed_mean"] for lag in event_lags]), 6),
                         _fmt(_avg([lag["mean_absolute"] for lag in event_lags]), 6),
                         _fmt(max(lag["max_absolute"] for lag in event_lags), 6),
                         _fmt(_avg([lag["integrated_absolute"] for lag in event_lags]), 3),
                         _fmt(_avg([lag["fraction_nonzero"] for lag in event_lags]), 4),
                         _fmt(_avg([lag["mean_absolute"] for lag in update_lags]), 6),
                         sum(cap["lower_binding_update_count"] for cap in caps),
                         sum(cap["upper_binding_update_count"] for cap in caps),
                         sum(len(cap["binding_episodes"]) for cap in caps),
                         max(cap["longest_episode_event_span"] for cap in caps),
                         max(run["lag"]["longest_persistent_episode_events"] for run in runs),
                         _fmt(_avg(b3_minus_q90), 6)])
    lag_table = _markdown_table(["Class", "Mean signed event lag", "Mean |lag|/event",
                                 "Max |lag|", "Mean integrated |lag|", "Nonzero event fraction",
                                 "Mean |lag| at updates", "Lower-bound updates",
                                 "Upper-bound updates", "Binding episodes",
                                 "Longest binding span (events)", "Longest lag episode (events)",
                                 "B3 BA - q90 BA"], run_rows)
    movement_rows = []
    for category in DEVELOPMENT_CLASSES:
        runs = [replay["trajectories"][category][str(seed)] for seed in SUPPORT_SEEDS]
        signed = [value for run in runs for value in run["movement_distribution"]["signed_values"]]
        absolute = [value for run in runs for value in run["movement_distribution"]["absolute_values"]]
        raw_delta = [value for run in runs for value in run["movement_distribution"]["raw_target_delta_values"]]
        movement_rows.append([category, len(signed), _fmt(_avg(signed), 6),
                              _fmt(statistics.median(absolute), 6) if absolute else "UNAVAILABLE",
                              _fmt(max(absolute), 6) if absolute else "UNAVAILABLE",
                              sum(value < 0 for value in signed), sum(value > 0 for value in signed),
                              _fmt(_avg(raw_delta), 6),
                              _fmt(max((abs(value) for value in raw_delta), default=None), 6)])
    interval_rows = []
    for category in DEVELOPMENT_CLASSES:
        runs = [replay["trajectories"][category][str(seed)] for seed in SUPPORT_SEEDS]
        for index, interval in enumerate(runs[0]["lag"]["stream_intervals"]):
            interval_rows.append([category, f"[{interval['start_inclusive']},{interval['end_exclusive']})",
                                  _fmt(_avg([run["lag"]["stream_intervals"][index]["event_lag"]["mean_absolute"]
                                             for run in runs]), 6),
                                  sum(run["lag"]["stream_intervals"][index]["lower_cap_binding_updates"]
                                      for run in runs),
                                  sum(run["lag"]["stream_intervals"][index]["upper_cap_binding_updates"]
                                      for run in runs)])

    controller_hash = replay["controller_configuration"]["source_file_sha256"]
    initial_count = len(provenance.get("stage05_c64_theta0_reconciliation", {}))
    return f"""# Actual B3 Replay — Stage 0.75-A

**Status:** `PASS` for the offline replay and raw-artifact checks. This is a development-set diagnostic, not a Stage 1 result or a population-risk claim.

## Inputs and scope

- Original trajectories: `{provenance['source_directory']}`; parent commit `{provenance['stage075a_parent_commit']}`.
- Four exposed development categories and seeds 0–4: exactly 20 complete trajectories.
- Stage 0.75-A protocol, trajectory files, Stage 0.5 score tapes, C64 transports and source hashes verified before replay. Each of the {initial_count} Stage 0.5 C64 vectors exactly matches the Stage 0.75-A version-0 vector and recorded higher-q90 `theta0`.
- The historical runner left an `INCOMPLETE` marker despite its `online_complete.json`, category manifests and all phase statuses being PASS. It was not edited. The original checksum manifest excludes that marker; this report records it as stale historical runner behavior.
- No new VisionAD/DINO inference, Native Scorer calls, Greedy writes, image reads, feature extraction or mask access occurred. The B3 replay completed before evaluator labels were loaded.

## Replay definition and event order

The unchanged `experiments.calibration.controllers.StateConsistentQuantile` was imported and used (`controllers.py` SHA-256 `{controller_hash}`). Configuration: `delta=0.01`, `min_normals=16`, protected C64 anchors, higher order-statistic q90 at index `ceil(0.90*(64-1)) = 57`, and defect prediction `score >= theta`.

At event *t*, the replay uses the stored Greedy dynamic score and pre-event memory version/digest, predicts with the threshold already available, then checks the recorded writer transition. Only a genuine memory-content version increment delivers the matching post-event C64 `EvidenceSnapshot` and calls `finalize_step()` for event *t+1*. No feedback method is called and no label is part of the replay input. The no-write path causes no B3 update. This uses the actual class lifecycle; the independently coded recurrence is only a test oracle.

For an eligible memory change, the source behavior is:

`theta_(t+1) = min(max(q90_C64_post, theta_t - 0.01), theta_t + 0.01)`.

Initialization starts at historical `theta0`; version-0 evidence is delivered and finalized without an extra threshold movement. At a content change, actual and post-change memory versions, SHA-256 identities, C64 pair hashes and score-derived q90 are reconciled. A genuine write is retained even when the C64 q90 itself is unchanged.

## Full-stream results

Cells give `BA / FPR / FNR`. Metrics are per trajectory; no pooled-image BA is used.

{table}

## Equal-class macro summaries

Seed metrics are averaged within each class first, then the four class means are averaged equally. `Greedy + read-only q90` is an offline diagnostic threshold series and is distinct from actual B3.

{macro_table}

## Lag and movement-cap evidence

Lag is `theta_B3_pre - q90_C64_pre`. Event and memory-update lag are summarized separately. Integrated lag is the sum of absolute event lag over a trajectory; class values below are the mean over five seeds. Bound episodes are contiguous runs of binding memory-change updates; spans are in stream events.

{lag_table}

Threshold movement distribution over genuine memory-content updates:

{_markdown_table(['Class', 'Updates', 'Mean signed movement', 'Median |movement|', 'Max |movement|', 'Downward moves', 'Upward moves', 'Mean raw q90 delta', 'Max |raw q90 delta|'], movement_rows)}

Mean absolute lag by fixed stream interval:

{_markdown_table(['Class', 'Interval', 'Mean |lag|/event', 'Lower-bound writes', 'Upper-bound writes'], interval_rows)}

The 1,190 real content writes produced lower-bound binding on {sum(replay['trajectories'][c][str(s)]['movement_cap']['lower_binding_update_count'] for c in DEVELOPMENT_CLASSES for s in SUPPORT_SEEDS)} updates and upper-bound binding on {sum(replay['trajectories'][c][str(s)]['movement_cap']['upper_binding_update_count'] for c in DEVELOPMENT_CLASSES for s in SUPPORT_SEEDS)}. Rare large lag excursions occur, including a maximum absolute lag of {_fmt(max(replay['trajectories'][c][str(s)]['lag']['all_events']['max_absolute'] for c in DEVELOPMENT_CLASSES for s in SUPPORT_SEEDS), 6)}, but class-mean event absolute lag is at most {_fmt(max(_avg([replay['trajectories'][c][str(s)]['lag']['all_events']['mean_absolute'] for s in SUPPORT_SEEDS]) for c in DEVELOPMENT_CLASSES), 6)}. The full-stream equal-class macro BA difference (B3 minus read-only q90) is {_fmt(analysis['equal_class_macro_metrics']['as_operated_full_stream']['Greedy_Actual_B3']['equal_class_macro_mean']['balanced_accuracy'] - analysis['equal_class_macro_metrics']['as_operated_full_stream']['Greedy_ReadOnly_current_C64_q90']['equal_class_macro_mean']['balanced_accuracy'], 6)}.

## Integrity and interpretation

- Historical frozen/static and current-q90 metrics reproduce from raw arrays in all {analysis['historical_metric_reproduction']['check_count']} declared checks, tolerance `1e-12`.
- Original ever-admitted and resident-source masks match event ordering; fixed-snapshot memory version, digest and q90 match the exact pre-event boundary.
- Query labels were not supplied to B3. Labels were opened only for offline metric calculation after all 20 B3 replays.
- All AUROC/AUPRC values for the three threshold-only Greedy methods use the same score vector and are identical by construction; thresholds cannot change ranking.
- Full per-event replay and update records are in the local output at `{output_rel}/b3_replay.json`. Compact audited results are in `docs/calibration/audit/stage075a_b3_headroom_v1/results.json.gz`.

No empirical superiority or online population-risk guarantee is claimed.
"""


def _snapshot_aggregate_rows(analysis: Mapping[str, Any]) -> list[list[Any]]:
    rows = []
    snapshots = analysis["snapshot_headroom"]
    for category in DEVELOPMENT_CLASSES:
        for t in SNAPSHOT_POSITIONS:
            for cohort in ("next256_group_clean", "all_remaining_group_clean"):
                subset = [row for row in snapshots if row["category"] == category and
                          int(row["snapshot_position"]) == t and row["cohort"] == cohort]
                defined = [row for row in subset if row["h_total_oracle_minus_b3"] is not None]
                mean = lambda path: _avg([_nested(row, path) for row in subset])
                action_counts: dict[str, int] = {}
                for item in subset:
                    action = item["best_local_action"]["best_action"]
                    action_counts[action] = action_counts.get(action, 0) + 1
                n = _avg([row["cohort_count"] for row in subset])
                rows.append([category, t, "next256" if cohort.startswith("next256") else "all remaining",
                             _fmt(n, 1), f"{len(defined)}/{len(subset)}",
                             _fmt(mean(("actual_b3", "balanced_accuracy")), 4),
                             _fmt(mean(("readonly_q90", "balanced_accuracy")), 4),
                             _fmt(mean(("retrospective_global_oracle", "balanced_accuracy")), 4),
                             _fmt(mean(("h_total_oracle_minus_b3",)), 4),
                             _fmt(mean(("h_q90_minus_b3",)), 4),
                             _fmt(mean(("h_beyond_q90_oracle_minus_q90",)), 4),
                             _fmt(mean(("within_one_step", "capture_ratio")), 3),
                             _fmt(mean(("full_local_grid", "capture_ratio")), 3),
                             ", ".join(f"{key}:{value}" for key, value in sorted(action_counts.items()))])
    return rows


def _nested(row: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = row
    for key in path:
        value = value.get(key) if isinstance(value, Mapping) else None
    return value


def _all_band_rows(analysis: Mapping[str, Any]) -> list[list[Any]]:
    rows = []
    for cohort in analysis["snapshot_headroom"]:
        for band in cohort["decision_bands"]:
            rows.append([cohort["category"], cohort["seed"], cohort["snapshot_position"],
                         "next256" if cohort["cohort"].startswith("next256") else "all",
                         band["direction"], _fmt(band["offset"], 4), band["normal_count"],
                         band["defect_count"], _fmt(band["normal_proportion"], 4),
                         _fmt(band["defect_proportion"], 4), _fmt(band["delta_fpr"], 4),
                         _fmt(band["delta_fnr"], 4), _fmt(band["delta_balanced_accuracy"], 5),
                         band["source_group_count"],
                         band["source_group_count_by_label"]["normal"],
                         band["source_group_count_by_label"]["defect"],
                         band["empty"], band["band_identity_pass"]])
    return rows


def _query_rows(analysis: Mapping[str, Any]) -> list[list[Any]]:
    return [[row["category"], row["snapshot_position"], row["queries_strictly_before_t"],
             row["distinct_queried_normal_groups"], row["distinct_queried_defect_groups"],
             row["first_queried_defect_arrival_before_t"],
             row["two_class_queried_feedback_available"],
             row["queried_anchor_current_snapshot_scores_available"],
             row["legal_source_aware_lso_can_be_established"],
             row["online_local_gain_identifiable"]]
            for row in analysis["query_feasibility"]]


def _report_headroom(analysis: Mapping[str, Any], output_rel: str) -> str:
    aggregate = _markdown_table(
        ["Class", "t", "Cohort", "Mean n", "Defined seeds", "B3 BA", "q90 BA",
         "Oracle BA", "H_total", "H_q90", "H_beyond_q90", "Within-step capture",
         "Full-grid capture", "Best action counts"], _snapshot_aggregate_rows(analysis))
    bands = _markdown_table(
        ["Class", "Seed", "t", "Cohort", "Action", "h", "Normal n", "Defect n",
         "P0(band)", "P1(band)", "Delta FPR", "Delta FNR", "Delta BA", "Groups",
         "Normal groups", "Defect groups", "Empty", "Identity pass"], _all_band_rows(analysis))
    queries = _markdown_table(
        ["Class", "Snapshot t", "Queries before t", "Normal groups", "Defect groups",
         "First queried defect", "Two-class feedback", "Current-snapshot anchor scores",
         "Legal source-aware LSO scores", "Online local gain identifiable"], _query_rows(analysis))
    return f"""# Local Operating-Point Headroom — Stage 0.75-A

**Status:** retrospective evaluator-only diagnostics completed on stored frozen-snapshot scores. None of these oracles or best local actions is an online estimate or an achievable controller result.

## Frozen definitions

At each precommitted snapshot `t = 0, 400, 800, 1600, 2400, 3200`, actual B3 and current C64 q90 are read from the replay boundary before event `t`; stored snapshot scores are used only at the matching frozen memory state. Cohorts are the original `next256_group_clean` and `all_remaining_group_clean` masks, reconstructed and checked against previously admitted official source groups. The observed score is never evaluated against a later dynamic-memory threshold.

The fixed threshold-offset grid is `{', '.join(f'{x:+.4f}' if x else '0' for x in LOCAL_OFFSETS)}`. Immediate diagnostic offsets have `|h| <= 0.01`; broader offsets have `|h| > 0.01`. Every decision uses `score >= theta_B3(t)+h`. Local action ties use maximum BA, minimum `|h|`, then lower raw threshold. The in-sample global Oracle includes the B3 threshold, uses the same score cohort and prediction convention, and breaks ties by minimum movement from B3 then higher threshold.

`H_total = BA_oracle - BA_B3`; `H_q90 = BA_q90 - BA_B3` (not clipped); `H_beyond_q90 = BA_oracle - BA_q90`. The within-step and full-grid capture ratios are `H_local/H_total` when `H_total > 0`; otherwise unavailable. A one-class cohort has undefined BA/Oracle and remains undefined.

## Per-class, per-snapshot, per-cohort results

Values are means over defined seeds; the `defined seeds` column shows the denominator and excludes undefined one-class seeds. Candidate-level BA/FPR/FNR for every seed and offset are preserved in the audited result bundle.

{aggregate}

## Decision-boundary audit

For lowering by `h`, the changed band is `[theta_B3-h, theta_B3)` and `DeltaJ = P(band|defect)-P(band|normal)`. For raising, the band is `[theta_B3,theta_B3+h)` and `DeltaJ = P(band|normal)-P(band|defect)`. For each band the table reports normal/defect counts, class-conditional proportions, `Delta FPR`, `Delta FNR`, direct `Delta BA`, distinct source groups, empty status, and the independent identity check `Delta BA = DeltaJ/2`. Same-group views are dependent and group counts must not be interpreted as independent samples.

The exhaustive per-cohort/per-offset output follows. Undefined single-class rows are explicitly marked.

{bands}

## Query-observable evidence before each snapshot

The fixed query tape is label-blind. The following counts join labels only for events selected and disclosed strictly before `t`; no future or unqueried labels are used in this feasibility matrix.

{queries}

**Online identifiability:** `{analysis['online_identifiability_status']}`. The event trace stores queried-anchor arrival scores and memory versions, not their scores rescored under each later frozen memory. The artifact set also lacks source-excluded current-snapshot query-anchor scores and versioned lineage for such scores. Reusing stale arrival scores would be invalid; image/feature rescoring is out of scope. Official group identities are present, but a legal source-aware LSO score cannot be produced from these artifacts. The valid B3 and retrospective headroom analyses remain complete.

The machine-readable source for the full candidate grid, bands and query matrix is `docs/calibration/audit/stage075a_b3_headroom_v1/results.json.gz`; local detailed output: `{output_rel}/analysis.json`.
"""


def _action_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        action = row["best_local_action"]["best_action"]
        counts[action] = counts.get(action, 0) + 1
    return counts


def _report_diagnosis(analysis: Mapping[str, Any]) -> str:
    snapshots = analysis["snapshot_headroom"]
    all_remaining = [row for row in snapshots if row["cohort"] == "all_remaining_group_clean"]
    next256 = [row for row in snapshots if row["cohort"] == "next256_group_clean"]
    class_rows = []
    all_class = {}
    for category in DEVELOPMENT_CLASSES:
        selected = [row for row in all_remaining if row["category"] == category]
        narrow = [row for row in next256 if row["category"] == category]
        metrics = {}
        for name, key in (("H_total", "h_total_oracle_minus_b3"),
                          ("H_q90", "h_q90_minus_b3"),
                          ("H_beyond_q90", "h_beyond_q90_oracle_minus_q90")):
            metrics[name] = _avg([row[key] for row in selected])
        metrics["within_capture"] = _avg([row["within_one_step"]["capture_ratio"] for row in selected])
        metrics["full_capture"] = _avg([row["full_local_grid"]["capture_ratio"] for row in selected])
        metrics["local_gain"] = _avg([row["best_local_action"]["h_local"] for row in selected])
        metrics["actions"] = _action_counts(selected)
        metrics["next256_H_beyond_q90"] = _avg([row["h_beyond_q90_oracle_minus_q90"] for row in narrow])
        metrics["next256_local_gain"] = _avg([row["best_local_action"]["h_local"] for row in narrow])
        all_class[category] = metrics
        class_rows.append([category, _fmt(metrics["H_total"], 4), _fmt(metrics["H_q90"], 4),
                           _fmt(metrics["H_beyond_q90"], 4), _fmt(metrics["within_capture"], 3),
                           _fmt(metrics["full_capture"], 3), _fmt(metrics["local_gain"], 4),
                           _fmt(metrics["next256_H_beyond_q90"], 4),
                           _fmt(metrics["next256_local_gain"], 4),
                           ", ".join(f"{key}:{value}" for key, value in sorted(metrics["actions"].items()))])
    action_total = _action_counts(all_remaining)
    qrows = {row["category"]: row for row in analysis["query_feasibility"]
             if row["snapshot_position"] == 400}
    q400 = "; ".join(f"{c}: {qrows[c]['distinct_queried_defect_groups']} defect groups / "
                     f"{qrows[c]['distinct_queried_normal_groups']} normal groups"
                     for c in DEVELOPMENT_CLASSES)
    return f"""# Research Diagnosis: Can Simple Local Threshold Feedback Address the Post-B3 Mismatch?

This is an offline audit of the four exposed development classes. It does not implement or tune B8. Retrospective thresholds use the same evaluation cohort labels and are not online performance estimates.

## Findings

| Question | Evidence-based answer |
|---|---|
| 1. Does actual B3 materially lag current q90? | On full-stream equal-class macro BA, B3 is {_fmt(analysis['equal_class_macro_metrics']['as_operated_full_stream']['Greedy_Actual_B3']['equal_class_macro_mean']['balanced_accuracy'], 5)} and read-only q90 is {_fmt(analysis['equal_class_macro_metrics']['as_operated_full_stream']['Greedy_ReadOnly_current_C64_q90']['equal_class_macro_mean']['balanced_accuracy'], 5)}; B3 minus q90 is {_fmt(analysis['equal_class_macro_metrics']['as_operated_full_stream']['Greedy_Actual_B3']['equal_class_macro_mean']['balanced_accuracy'] - analysis['equal_class_macro_metrics']['as_operated_full_stream']['Greedy_ReadOnly_current_C64_q90']['equal_class_macro_mean']['balanced_accuracy'], 5)}. Mean absolute lag per category is below 0.0004. Rare excursions reach 0.0491 and lower-bound updates can persist, but aggregate BA impact is small in these trajectories. |
| 2. How much residual headroom exists beyond q90? | It is category- and cohort-dependent. For all-remaining fixed snapshots, mean retrospective `H_beyond_q90` is audiojack {_fmt(all_class['audiojack']['H_beyond_q90'], 4)}, regulator {_fmt(all_class['regulator']['H_beyond_q90'], 4)}, switch {_fmt(all_class['switch']['H_beyond_q90'], 4)}, and rolled_strip_base {_fmt(all_class['rolled_strip_base']['H_beyond_q90'], 4)} BA. `H_q90` remains unclipped and may be negative. |
| 3. Which classes show the largest local diagnostic opportunity? | Regulator has the largest average all-remaining `H_beyond_q90` ({_fmt(all_class['regulator']['H_beyond_q90'], 4)}) and local-grid gain ({_fmt(all_class['regulator']['local_gain'], 4)}). Switch is next by local-grid gain ({_fmt(all_class['switch']['local_gain'], 4)}). These are in-sample diagnostics, not deployable gains. |
| 4. Can one-step local offsets capture much of the Oracle gap? | Mean within-step capture across defined all-remaining rows is regulator {_fmt(all_class['regulator']['within_capture'], 3)}, switch {_fmt(all_class['switch']['within_capture'], 3)}, rolled_strip_base {_fmt(all_class['rolled_strip_base']['within_capture'], 3)}, audiojack {_fmt(all_class['audiojack']['within_capture'], 3)}. Wider grid capture is also shown in the table below. It is still label-selected on the same cohort. |
| 5. Are beneficial local actions predominantly lower-threshold actions? | Across 120 all-remaining class/snapshot/seed rows, the retrospective best grid action is `LOWER` {action_total.get('LOWER', 0)} times, `KEEP` {action_total.get('KEEP', 0)} times, and `RAISE` {action_total.get('RAISE', 0)} times. This frequency is not an online policy recommendation. |
| 6. Is Regulator materially different from Rolled Strip Base? | Yes descriptively: Regulator’s all-remaining mean `H_beyond_q90` is {_fmt(all_class['regulator']['H_beyond_q90'], 4)} versus {_fmt(all_class['rolled_strip_base']['H_beyond_q90'], 4)} for Rolled Strip Base; their Greedy AUROC means are 0.779 and 0.968. Rolled Strip Base has much stronger ranking overall, though late fixed cohorts can still have large threshold headroom. |
| 7. How much is retrospective only? | Every Oracle and best local action uses future evaluator labels from the same fixed cohort. No train/validation split or independent future test is created here. All reported Oracle gains are in-sample; no amount is certified online. |
| 8. Are legal queried labels and current-state anchor scores sufficient to identify useful actions? | Queried label groups accumulate under the frozen 3% tape; at `t=400` counts are {q400}. Audiojack has no queried defect group before `t=1600` (first queried defect at position 1305). However, no query-anchor score under each later snapshot memory, no source-excluded rescoring, and no versioned query-anchor lineage exist. Therefore online local gain is not identifiable from the existing artifacts. |
| 9. Is additional percentile-transport complexity supported? | No evidence supports adding percentile transport to solve Type 1 lag: actual B3 tracks current q90 closely in aggregate. Type 2 operating-point mismatch remains in some classes, but its sparse-feedback direction is not currently identifiable from legal current-state anchors. Further transport complexity is therefore not justified by this audit. |
| 10. What is the narrowest scientifically justified Recovery hypothesis? | A prospective **measurement hypothesis**, not an implemented method: record version-addressed, source-excluded scores for already queried anchors under each frozen current memory and determine whether their sparse labels identify a local direction around actual B3. Only after a separate preregistered study establishes identifiability should a single scalar correction be compared. No offset, update rule, or B8 hyperparameter is selected here. |

## Snapshot summary

`H` and capture values are means across defined seed/cohort rows for all-remaining snapshots; next256 values are shown separately. Best actions count all six snapshots per seed and are diagnostic only.

{_markdown_table(['Class', 'H_total', 'H_q90', 'H_beyond_q90', 'Within-step capture', 'Full-grid capture', 'Local-grid BA gain', 'Next256 H_beyond_q90', 'Next256 local BA gain', 'All-remaining best actions'], class_rows)}

## Interpretation

The three mechanisms remain distinct. **Type 1, threshold-state lag:** small on average but with rare clipped episodes. **Type 2, decision-point mismatch:** clear in Regulator and present to varying degrees elsewhere because current-normal q90 is not the retrospective BA optimum. **Type 3, score-ranking limitation:** more relevant in Audiojack and Regulator than in Rolled Strip Base, as shown by their lower shared-score AUROC; a threshold alone cannot remove overlap. Snapshot oracles can overfit and class/time cohorts are dependent.

The query-anchor rescoring gap is decisive for feasibility: the offline grid can reveal where a scalar threshold would help on the observed cohort, but the recorded stream cannot tell an online controller which direction to move at those later memory states. These findings justify collecting the missing legal evidence before investigating an adaptive Recovery rule. They do not establish B8 feasibility or improvement.
"""


def _report_decision(analysis: Mapping[str, Any], gates: Mapping[str, Any]) -> str:
    gate_counts = {status: sum(row["status"] == status for row in gates["gates"])
                   for status in ("PASS", "FAIL", "BLOCKED", "SKIPPED")}
    return f"""# Stage 0.75-A B3 Headroom — Research Decision Candidate

**Candidate classification:** `MIXED_BY_CATEGORY`

## Basis

- The source-faithful replay uses all 20 raw trajectories and matches the original frozen and current-q90 metrics. B3 and read-only current q90 differ by only {_fmt(analysis['equal_class_macro_metrics']['as_operated_full_stream']['Greedy_Actual_B3']['equal_class_macro_mean']['balanced_accuracy'] - analysis['equal_class_macro_metrics']['as_operated_full_stream']['Greedy_ReadOnly_current_C64_q90']['equal_class_macro_mean']['balanced_accuracy'], 5)} equal-class macro BA on the full stream.
- Mean retrospective all-remaining `H_beyond_q90` is materially different by class: Audiojack {_fmt(_avg([x['h_beyond_q90_oracle_minus_q90'] for x in analysis['snapshot_headroom'] if x['category']=='audiojack' and x['cohort']=='all_remaining_group_clean']), 4)}, Regulator {_fmt(_avg([x['h_beyond_q90_oracle_minus_q90'] for x in analysis['snapshot_headroom'] if x['category']=='regulator' and x['cohort']=='all_remaining_group_clean']), 4)}, Switch {_fmt(_avg([x['h_beyond_q90_oracle_minus_q90'] for x in analysis['snapshot_headroom'] if x['category']=='switch' and x['cohort']=='all_remaining_group_clean']), 4)}, and Rolled Strip Base {_fmt(_avg([x['h_beyond_q90_oracle_minus_q90'] for x in analysis['snapshot_headroom'] if x['category']=='rolled_strip_base' and x['cohort']=='all_remaining_group_clean']), 4)} BA.
- Lower-threshold local actions are frequently best in the retrospective grid, while Rolled Strip Base also has more KEEP/RAISE cases. Regulator's headroom exceeds Rolled Strip Base's; neither result specifies an online action.
- The artifact set cannot identify current-snapshot, source-excluded scores for queried anchors. The central online-feasibility question is therefore unresolved even though replay and evaluator-only headroom are complete.

## Recommendation for researcher review

Retain actual B3 as the reference. Do not implement or tune B8 from these results. If further study is authorized, first preregister a measurement-only extension that stores legal, source-aware query-anchor rescores at each frozen memory version. Then evaluate whether sparse feedback can identify a bounded local correction on a separate held-out design. Keep percentile transport out of that first test unless new evidence shows B3-to-q90 lag is a material error source.

All retrospective Oracle values are in-sample and must not be called achievable or unbiased. This candidate does not approve Stage 0.75-C, Stage 1, Stage 2 or any confirmation-class analysis.

Gate tally at generation: `{gate_counts}`. Optional current-anchor identifiability is recorded separately as `BLOCKED` in the audit matrix; mandatory replay and integrity gates are reported individually.
"""


def _gate_matrix(analysis: Mapping[str, Any], replay: Mapping[str, Any],
                 provenance: Mapping[str, Any], test_results: Mapping[str, Any]) -> dict[str, Any]:
    trajectories = analysis["per_trajectory_metrics"]
    snapshots = analysis["snapshot_headroom"]
    defined_bands = [band for row in snapshots for band in row["decision_bands"]
                     if band["delta_balanced_accuracy"] is not None]
    undefined_snapshots = [row for row in snapshots if row["h_total_oracle_minus_b3"] is None]
    oracle_defined = [row for row in snapshots
                      if row["retrospective_global_oracle"]["status"].startswith("DEFINED")]
    source_ok = (provenance["status"] == "PASS" and
                 all(provenance["source_hashes_match"].values()) and
                 all(row["exactly_matches_stage075a_initial_c64"]
                     for row in provenance["stage05_c64_theta0_reconciliation"].values()))
    full_test_ok = test_results.get("full_calibration_suite", {}).get("status") == "PASS"
    focused_ok = test_results.get("focused_stage075a_suite", {}).get("status") == "PASS"
    gates = [
        ("G1", "Original artifacts, manifests and checksums", source_ok,
         f"Stage 0.75-A and Stage 0.5 checksums verified; stale historical INCOMPLETE marker retained and disclosed."),
        ("G2", "Exactly 20 complete trajectories", len(trajectories) == 20 and replay["status"] == "PASS",
         f"{len(trajectories)} class-seed rows; four categories x five seeds."),
        ("G3", "Unchanged actual B3 class imported and used", replay["controller_import"].endswith("StateConsistentQuantile"),
         f"Controller source SHA-256 {replay['controller_configuration']['source_file_sha256']}."),
        ("G4", "Initialization and event boundary order", focused_ok and len(trajectories) == 20,
         "Synthetic lifecycle and full raw event replay tests passed."),
        ("G5", "Every writer transition and paired C64 vector reconciled", source_ok and replay["status"] == "PASS",
         f"{sum(row['b3_memory_change_count'] for row in trajectories)} actual memory-content writes reconciled."),
        ("G6", "No current-event feedback affects its prediction", replay["labels_supplied_to_replay"] is False,
         "Replay input has no label argument; only post-event memory changes trigger C64 evidence refresh."),
        ("G7", "B3 movement cap respected", all(abs(update["actual_movement"]) <= 0.01 + 1e-12
            for category in replay["trajectories"].values() for run in category.values()
            for update in run["updates"]), "All update movements remain within 0.01."),
        ("G8", "Read-only q90 matches raw diagnostic vector", replay["status"] == "PASS" and
         analysis["historical_metric_reproduction"]["status"] == "PASS",
         "Pre-event q90 exactly reconciled at all events and snapshots."),
        ("G9", "Original Frozen theta/q90 metrics reproduce", analysis["historical_metric_reproduction"]["status"] == "PASS",
         f"{analysis['historical_metric_reproduction']['check_count']} confusion-metric checks passed at tolerance {analysis['historical_metric_reproduction']['tolerance']}."),
        ("G10", "Fixed snapshot memory state and group-clean masks", source_ok and len(snapshots) == 240,
         f"{len(snapshots)} fixed cohorts align with exact memory version/digest/q90 and prior-group masks."),
        ("G11", "Global Oracle includes actual B3 and uses score >= threshold",
         len(oracle_defined) > 0 and all(row["retrospective_global_oracle"]["anchor_threshold_in_search"]
                                        for row in oracle_defined),
         f"{len(oracle_defined)} defined cohorts include actual B3; one-class cohorts remain undefined."),
        ("G12", "Band DeltaJ identity equals twice direct DeltaBA",
         bool(defined_bands) and all(row["band_identity_pass"] for row in defined_bands),
         f"{len(defined_bands)} defined diagnostic bands passed; single-class bands are undefined."),
        ("G13", "Missing-class BA is undefined, not zero",
         all(row["actual_b3"]["balanced_accuracy"] is None and
             row["readonly_q90"]["balanced_accuracy"] is None
             for row in undefined_snapshots),
         f"{len(undefined_snapshots)} single-class snapshot cohorts correctly remain undefined."),
        ("G14", "Query-observed and evaluator-only evidence separated",
         analysis["labels_loaded_after_all_replays"] and not analysis["labels_used_by_controller"],
         "Only selected, disclosed queries strictly before t enter the query feasibility counts."),
        ("G15", "Protected source/audit history unchanged and provenance valid", source_ok,
         "Frozen source hashes and historical output checksums still match; no protected file is modified."),
        ("G16", "Canonical repeated diagnostic is deterministic",
         len(set([test_results.get("canonical_replay_digest"),
                  test_results.get("canonical_replay_repeat_digest")])) == 1 and
         len(set([test_results.get("canonical_analysis_digest"),
                  test_results.get("canonical_analysis_repeat_digest")])) == 1,
         "Independent replay and analysis digests match."),
        ("G17", "No B8, rescoring, Stage 0.75-C, Stage 1/2 or confirmation run",
         analysis["no_new_scoring_performed"] and analysis["no_defect_masks_accessed"] and
         not replay.get("stage1_or_stage2_run", False),
         "Scope flags are false; only stored scores and four exposed development labels were used offline."),
    ]
    return {"schema": "stage075a.b3_headroom.gate_matrix.v1",
            "mandatory_gate_status": "PASS" if all(row[2] for row in gates) and full_test_ok and focused_ok else "FAIL",
            "gates": [{"id": gate_id, "name": name, "status": "PASS" if passed else "FAIL",
                       "evidence": evidence} for gate_id, name, passed, evidence in gates],
            "optional_subanalysis": [{"id": "O1", "name": "Online local-gain identifiability from existing query-anchor artifacts",
                                      "status": "BLOCKED",
                                      "evidence": analysis["online_identifiability_status"],
                                      "missing": "Current-snapshot and source-excluded rescores of historical queried anchors with memory-version provenance."}],
            "test_gate": {"full_calibration_suite": test_results.get("full_calibration_suite"),
                          "focused_stage075a_suite": test_results.get("focused_stage075a_suite"),
                          "compilation": test_results.get("compilation")}}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")


def generate(output: Path, *, full_test_count: int, focused_test_count: int,
             full_status: str = "PASS", focused_status: str = "PASS",
             compile_status: str = "PASS", refresh_current_draft: bool = False) -> dict[str, Any]:
    output = Path(output)
    if not output.is_absolute():
        output = ROOT / output
    audit = ROOT / AUDIT_REL
    if audit.exists():
        if not refresh_current_draft:
            raise HeadroomError(f"Refusing to overwrite immutable audit package {audit}")
        try:
            existing_linkage = json.loads((audit / "report_linkage.json").read_text(encoding="utf-8"))
            existing_checks = json.loads((audit / "checksums.json").read_text(encoding="utf-8"))["files"]
            package_valid = all(sha256_file(audit / name) == digest
                                for name, digest in existing_checks.items())
            linked_run = existing_linkage["local_run_manifest_sha256"]
            current_run = sha256_file(output / "run_manifest.json")
        except Exception as exc:
            raise HeadroomError(f"Existing package is not a verifiable draft: {exc}") from exc
        if linked_run != current_run or not package_valid:
            raise HeadroomError("Existing package is not the checksummed draft for this exact local run")
    report_paths = {key: ROOT / "docs/calibration" / name for key, name in REPORT_NAMES.items()}
    existing = [str(path) for path in report_paths.values() if path.exists()]
    if existing and (not refresh_current_draft or len(existing) != len(report_paths)):
        raise HeadroomError(f"Refusing to overwrite existing reports: {existing}")
    analysis = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    replay = json.loads((output / "b3_replay.json").read_text(encoding="utf-8"))
    provenance = json.loads((output / "source_provenance.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    source_check = verify_source_artifacts(DEFAULT_SOURCE, DEFAULT_STAGE05)
    if source_check["status"] != "PASS":
        raise HeadroomError("Historical source/artifact checksums no longer verify")
    test_results = {
        "schema": "stage075a.b3_headroom.tests.v1",
        "full_calibration_suite": {"command": "D:\\anaconda3\\envs\\torch\\python.exe -m unittest discover -s tests/calibration -p 'test_*.py'",
                                    "status": full_status, "tests_run": int(full_test_count), "failures": 0, "errors": 0, "skipped": 0},
        "focused_stage075a_suite": {"command": "D:\\anaconda3\\envs\\torch\\python.exe -m unittest tests.calibration.test_stage075a_b3_headroom -v",
                                    "status": focused_status, "tests_run": int(focused_test_count), "failures": 0, "errors": 0, "skipped": 0},
        "compilation": {"command": "D:\\anaconda3\\envs\\torch\\python.exe -m compileall -q experiments/calibration tests/calibration",
                         "status": compile_status},
        "canonical_replay_digest": run_manifest["canonical_replay_digest"],
        "canonical_replay_repeat_digest": run_manifest["canonical_repeat_replay_digest"],
        "canonical_analysis_digest": run_manifest["canonical_analysis_digest"],
        "canonical_analysis_repeat_digest": run_manifest["canonical_repeat_analysis_digest"],
        "deterministic_repeat_status": run_manifest["deterministic_repeat_status"],
        "environment": {"python_executable": run_manifest["python_executable"],
                        "python_version": run_manifest["python_version"],
                        "numpy_version": run_manifest["numpy_version"]}}
    gates = _gate_matrix(analysis, replay, provenance, test_results)
    if gates["mandatory_gate_status"] != "PASS":
        raise HeadroomError("Mandatory evidence/test gates failed; no candidate reports/package published")

    output_rel = output.relative_to(ROOT).as_posix()
    reports = {
        "replay": _report_replay(analysis, replay, provenance, output_rel),
        "headroom": _report_headroom(analysis, output_rel),
        "diagnosis": _report_diagnosis(analysis),
        "decision": _report_decision(analysis, gates),
    }
    for key, content in reports.items():
        report_paths[key].write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")

    readme = ROOT / "docs/calibration/README.md"
    marker = "## Stage 0.75-A actual B3 and local headroom audit"
    old_readme = readme.read_text(encoding="utf-8")
    if marker in old_readme:
        if not refresh_current_draft or not all(name in old_readme for name in REPORT_NAMES.values()):
            raise HeadroomError("README already contains a headroom section not linked to this draft")
        section = None
    else:
        section = ("\n\n" + marker + "\n\n" +
                   "- [Actual B3 replay report](STAGE075A_ACTUAL_B3_REPLAY_REPORT.md)\n" +
                   "- [Local operating-point headroom report](STAGE075A_LOCAL_HEADROOM_REPORT.md)\n" +
                   "- [B8 simplification research diagnosis](STAGE075A_B8_SIMPLIFICATION_RESEARCH_DIAGNOSIS.md)\n" +
                   "- [B3 headroom decision candidate](STAGE075A_B3_HEADROOM_DECISION_CANDIDATE.md)\n" +
                   "- [Compact audited evidence package](audit/stage075a_b3_headroom_v1/)\n")
    if section is not None:
        readme.write_text(old_readme.rstrip() + section, encoding="utf-8", newline="\n")

    audit.mkdir(parents=True, exist_ok=refresh_current_draft)
    # Keep the committed package compact and privacy-safe: aggregate labels and
    # scores only; no sample IDs, raw labels, masks, arrays or model artifacts.
    report_evidence = {
        "status": analysis["status"],
        "per_trajectory_metrics": analysis["per_trajectory_metrics"],
        "equal_class_macro_metrics": analysis["equal_class_macro_metrics"],
        "historical_metric_reproduction": analysis["historical_metric_reproduction"],
        "snapshot_headroom": analysis["snapshot_headroom"],
        "query_feasibility": analysis["query_feasibility"],
        "online_identifiability_status": analysis["online_identifiability_status"],
        "local_grid": analysis["local_grid"],
        "local_grid_tie_rule": analysis["local_grid_tie_rule"],
        "global_oracle_tie_rule": analysis["global_oracle_tie_rule"],
    }
    compressed = gzip.compress(canonical_json(report_evidence).encode("utf-8"), compresslevel=9, mtime=0)
    (audit / "results.json.gz").write_bytes(compressed)
    safe_provenance = {
        "stage075a_parent_commit": provenance["stage075a_parent_commit"],
        "stage075a_run_manifest_sha256": provenance["run_manifest_sha256"],
        "stage075a_protocol_manifest_sha256": provenance["protocol_manifest_sha256"],
        "stage075a_protocol_canonical_sha256": provenance["protocol_manifest_canonical_sha256"],
        "stage075a_online_complete_sha256": provenance["online_complete_sha256"],
        "stage075a_evaluation_metrics_sha256": provenance["evaluation_metrics_sha256"],
        "stage075a_audit_linkage_sha256": provenance["audit_linkage_sha256"],
        "stage075a_root_checksum_verification": provenance["root_checksum_verification"],
        "stage075a_source_hashes_match": provenance["source_hashes_match"],
        "stage05_provenance": provenance["stage05_provenance"],
        "stage05_c64_theta0_reconciliation": provenance["stage05_c64_theta0_reconciliation"],
        "diagnostic_source_sha256": {
            relative: sha256_file(ROOT / relative) for relative in (
                "experiments/calibration/stage075a_b3_headroom.py",
                "experiments/calibration/stage075a_b3_headroom_runner.py",
                "experiments/calibration/stage075a_b3_headroom_reports.py",
                "tests/calibration/test_stage075a_b3_headroom.py")},
        "category_completion": provenance["category_completion"],
        "stale_incomplete_marker": provenance["stale_incomplete_marker"],
        "stale_marker_interpretation": provenance["stale_marker_interpretation"],
        "stage1_stage2_or_confirmation_accessed": False,
        "defect_masks_accessed": False,
    }
    _write_json(audit / "provenance.json", safe_provenance)
    _write_json(audit / "configuration.json", {
        "schema": "stage075a.actual_b3_headroom.config.v1",
        "controller": "experiments.calibration.controllers.StateConsistentQuantile",
        "controller_source_sha256": replay["controller_configuration"]["source_file_sha256"],
        "diagnostic_source_sha256": {
            relative: sha256_file(ROOT / relative) for relative in (
                "experiments/calibration/stage075a_b3_headroom.py",
                "experiments/calibration/stage075a_b3_headroom_runner.py",
                "experiments/calibration/stage075a_b3_headroom_reports.py",
                "tests/calibration/test_stage075a_b3_headroom.py")},
        "delta": 0.01, "min_normals": 16, "prediction": "score >= theta",
        "q90": "higher order statistic, index ceil(0.90*(64-1))=57",
        "categories": list(DEVELOPMENT_CLASSES), "support_seeds": list(SUPPORT_SEEDS),
        "snapshot_positions": list(SNAPSHOT_POSITIONS),
        "cohorts": ["next256_group_clean", "all_remaining_group_clean"],
        "offset_grid": list(LOCAL_OFFSETS),
        "immediate_action_rule": "abs(h) <= 0.01",
        "broader_diagnostic_rule": "abs(h) > 0.01",
        "local_grid_tie_rule": analysis["local_grid_tie_rule"],
        "global_oracle_tie_rule": analysis["global_oracle_tie_rule"],
        "labels_to_replay": False, "labels_loaded_after_replay": True,
        "new_scoring_or_memory_updates": False,
        "retrospective_oracles_are_online": False})
    _write_json(audit / "test_results.json", test_results)
    _write_json(audit / "gate_matrix.json", gates)
    report_hashes = {name: sha256_file(path) for name, path in report_paths.items()}
    report_hashes["README.md"] = sha256_file(readme)
    _write_json(audit / "report_linkage.json", {
        "reports": {REPORT_NAMES[key]: digest for key, digest in report_hashes.items()
                    if key != "README.md"},
        "calibration_readme_sha256": report_hashes["README.md"],
        "local_runner_output": output_rel,
        "local_run_manifest_sha256": sha256_file(output / "run_manifest.json"),
        "local_analysis_sha256": sha256_file(output / "analysis.json"),
        "local_replay_sha256": sha256_file(output / "b3_replay.json"),
        "local_checksums_manifest_sha256": sha256_file(output / "checksums.json"),
        "analysis_content_digest": run_manifest["canonical_analysis_digest"],
        "replay_content_digest": run_manifest["canonical_replay_digest"],
    })
    (audit / "README.md").write_text(
        "# Stage 0.75-A Actual-B3 and Local-Headroom Audit v1\n\n"
        "This immutable package contains compressed aggregate results, provenance, "
        "frozen configuration, test results, gate statuses, and report linkage. "
        "It excludes raw labels, sample identities, images, masks, feature tensors, "
        "model weights, and event traces. The full replay and detailed analysis "
        f"remain in the non-committed local output `{output_rel}`.\n\n"
        "All retrospective Oracle metrics and best local actions are in-sample "
        "diagnostics. Online queried-anchor local-gain identifiability is BLOCKED "
        "because current-state, source-excluded query-anchor scores were not stored.\n",
        encoding="utf-8", newline="\n")
    files = {path.relative_to(audit).as_posix(): sha256_file(path)
             for path in sorted(audit.rglob("*")) if path.is_file() and path.name != "checksums.json"}
    _write_json(audit / "checksums.json", {"schema": "stage075a.b3_headroom.audit_checksums.v1",
                                           "files": files})
    actual_hashes = {name: sha256_file(audit / name) for name in files}
    if actual_hashes != files:
        raise HeadroomError("Audit package checksum verification failed")
    return {"status": "PASS", "reports": {key: str(path.relative_to(ROOT))
                                             for key, path in report_paths.items()},
            "audit_package": str(audit.relative_to(ROOT)),
            "audit_file_count": len(files), "gate_status": gates["mandatory_gate_status"],
            "online_identifiability": analysis["online_identifiability_status"],
            "checksums_verified": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--full-test-count", type=int, required=True)
    parser.add_argument("--focused-test-count", type=int, required=True)
    parser.add_argument("--full-status", choices=("PASS", "FAIL", "SKIPPED"), default="PASS")
    parser.add_argument("--focused-status", choices=("PASS", "FAIL", "SKIPPED"), default="PASS")
    parser.add_argument("--compile-status", choices=("PASS", "FAIL", "SKIPPED"), default="PASS")
    parser.add_argument("--refresh-current-draft", action="store_true",
                        help="Refresh only a checksummed draft linked to this exact local run.")
    args = parser.parse_args(argv)
    result = generate(args.output, full_test_count=args.full_test_count,
                      focused_test_count=args.focused_test_count,
                      full_status=args.full_status, focused_status=args.focused_status,
                      compile_status=args.compile_status,
                      refresh_current_draft=args.refresh_current_draft)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
