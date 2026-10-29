"""Future-cohort metric process, launched only after query actions are frozen."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from . import stage05
from .stage075a import DEVELOPMENT_CLASSES, SNAPSHOT_POSITIONS, SNAPSHOT_NEXT_WIDTH
from .stage075a_b3_headroom import load_verified_inputs, replay_all, sha256_file
from .queried_anchor_identifiability import (
    AnchorScore, PRIMARY_OFFSETS, action_digest, canonical_json,
    defect_evidence_bin, local_gain_table, select_primary_action,
)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
DEFAULT_SOURCE = ROOT / "outputs/calibration/stage075a_greedy_ranking_repaired3_20261029"
DEFAULT_STAGE05 = ROOT / "outputs/calibration/stage05_exploratory_20261029"
DEFAULT_B3 = ROOT / "outputs/calibration/stage075a_b3_headroom_v1_20261030_verified2"
DEFAULT_JSON_ROOT = Path(r"G:\dataset\realiad\jsons\realiad_jsons")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")
    temp.replace(path)


def _labels_without_masks(json_root: Path, category: str,
                          expected_sha256: str) -> tuple[np.ndarray, str]:
    """Read evaluator labels while dropping mask_path keys during JSON decode."""
    source = json_root / f"{category}.json"
    if not source.is_file() or sha256_file(source) != expected_sha256:
        raise RuntimeError(f"{category}: evaluator metadata identity mismatch")

    def drop_mask_path(pairs):
        return {key: value for key, value in pairs if key != "mask_path"}

    decoder = json.JSONDecoder(object_pairs_hook=drop_mask_path)
    parsed = decoder.decode(source.read_text(encoding="utf-8"))
    normal_class = str(parsed.get("meta", {}).get("normal_class", "OK"))
    labels = np.asarray([
        0 if row.get("anomaly_class", normal_class) == normal_class else 1
        for row in parsed["test"]], dtype=np.int8)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise RuntimeError(f"{category}: invalid evaluator labels")
    return labels, hashlib.sha256(labels.tobytes()).hexdigest()


def _metrics(labels: np.ndarray, scores: np.ndarray, theta: float
             ) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or not np.isfinite(s).all():
        raise RuntimeError("future cohort labels/scores do not align")
    if set(np.unique(y).tolist()) != {0, 1}:
        return {"status": "UNDEFINED_ONE_CLASS", "n": int(len(y)),
                "normal_n": int(np.sum(y == 0)), "defect_n": int(np.sum(y == 1)),
                "fpr": None, "fnr": None, "j": None, "ba": None}
    pred = s >= float(theta)
    fpr = float(np.mean(pred[y == 0]))
    fnr = float(np.mean(~pred[y == 1]))
    j_value = float(1.0 - fpr - fnr)
    return {"status": "DEFINED", "n": int(len(y)),
            "normal_n": int(np.sum(y == 0)), "defect_n": int(np.sum(y == 1)),
            "fpr": fpr, "fnr": fnr, "j": j_value,
            "ba": float((1.0 + j_value) / 2.0)}


def _future_gains(labels: np.ndarray, scores: np.ndarray, theta: float,
                  offsets: Sequence[float]) -> dict[str, Any]:
    base = _metrics(labels, scores, theta)
    values: dict[str, Any] = {}
    for offset in offsets:
        result = _metrics(labels, scores, theta + float(offset))
        if base["status"] != "DEFINED" or result["status"] != "DEFINED":
            delta_j = delta_ba = None
        else:
            delta_j = float(result["j"] - base["j"])
            delta_ba = float(delta_j / 2.0)
        values[f"{float(offset):+.6f}"] = {
            **result, "offset": float(offset), "threshold": theta + float(offset),
            "delta_j_vs_keep": delta_j, "delta_ba_vs_keep": delta_ba,
            "delta_fpr": (None if base["fpr"] is None or result["fpr"] is None
                          else float(result["fpr"] - base["fpr"])),
            "delta_fnr": (None if base["fnr"] is None or result["fnr"] is None
                          else float(result["fnr"] - base["fnr"])),
        }
    return {"base": base, "gains": values}


def _future_best(gains: Mapping[str, Any]) -> dict[str, Any]:
    candidates = (("LOWER", "-0.010000"), ("KEEP", "+0.000000"),
                  ("RAISE", "+0.010000"))
    available = [(name, gains[key]["delta_j_vs_keep"])
                 for name, key in candidates
                 if gains.get(key, {}).get("delta_j_vs_keep") is not None]
    if len(available) != 3:
        return {"action": None, "unique": False, "reason": "UNDEFINED_FUTURE_METRIC"}
    best = max(value for _, value in available)
    actions = [name for name, value in available if value == best]
    return {"action": actions[0] if len(actions) == 1 else None,
            "unique": len(actions) == 1,
            "reason": "UNIQUE_MAXIMUM" if len(actions) == 1 else "EXACT_GAIN_TIE"}


def _resolve_snapshot_threshold(replay_theta: Any, score_status: str,
                                recorded_theta: Any) -> tuple[float | None, str]:
    """Fail closed on absent B3 thresholds without coercing None to float."""
    try:
        theta = float(replay_theta)
    except (TypeError, ValueError):
        return None, "BLOCKED_B3_THRESHOLD"
    if not np.isfinite(theta):
        return None, "BLOCKED_B3_THRESHOLD"
    if score_status != "PASS":
        return theta, "BLOCKED_NATIVE_SCORE"
    try:
        saved_theta = float(recorded_theta)
    except (TypeError, ValueError):
        return None, "BLOCKED_B3_THRESHOLD"
    if not np.isfinite(saved_theta) or theta != saved_theta:
        raise RuntimeError("query action B3 threshold differs from source-faithful replay")
    return theta, "PASS"


def _artifact_digest(source: Path) -> str:
    files = _read_json(source / "checksums.json")["files"]
    expected = {}
    for relative, digest in sorted(files.items()):
        path = source / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Historical Stage 0.75-A artifact changed: {relative}")
        expected[relative] = digest
    for category in DEVELOPMENT_CLASSES:
        manifest = _read_json(source / "trajectories" / category / "trajectory_manifest.json")
        private = source / "private_checkpoints" / f"{category}_memory_states.pt"
        digest = sha256_file(private)
        if digest != manifest.get("private_memory_states_sha256"):
            raise RuntimeError(f"Historical private snapshot changed: {category}")
        expected[f"private_checkpoints/{category}_memory_states.pt"] = digest
    for relative, digest in sorted(_read_json(DEFAULT_B3 / "checksums.json")["files"].items()):
        path = DEFAULT_B3 / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Actual-B3 artifact changed: {relative}")
        expected[f"actual_b3/{relative}"] = digest
    for name in ("run_manifest.json", "complete.json", "checksums.json"):
        expected[f"actual_b3/{name}"] = sha256_file(DEFAULT_B3 / name)
    return hashlib.sha256(canonical_json(expected).encode("utf-8")).hexdigest()


def _query_only_action(row: Mapping[str, Any],
                       private_rows: Sequence[Mapping[str, Any]]) -> str:
    """Recompute one action from query records only; accepts no future labels."""
    key = (str(row["category"]), int(row["seed"]), int(row["snapshot_position"]))
    records = [record for record in private_rows
               if (str(record["category"]), int(record["seed"]),
                   int(record["snapshot_position"])) == key]
    if row.get("score_status") != "PASS":
        return "ABSTAIN"
    anchors = [AnchorScore(**record["primary_anchor"]) for record in records]
    normals = [anchor for anchor in anchors if anchor.label == 0]
    defects = [anchor for anchor in anchors if anchor.label == 1]
    estimate = local_gain_table(
        normals, defects, theta=float(row["b3_threshold"]),
        offsets=PRIMARY_OFFSETS,
        expected_version=int(row["snapshot_memory_version"]),
        expected_snapshot_sha256=str(row["snapshot_memory_sha256"]))
    return select_primary_action(estimate)["action"]


def _leakage_control(action_rows: Sequence[Mapping[str, Any]],
                     private_rows: Sequence[Mapping[str, Any]],
                     labels_by_category: Mapping[str, np.ndarray]) -> dict[str, Any]:
    baseline = []
    altered = []
    checked_prefixes = 0
    for row in action_rows:
        category = str(row["category"])
        t = int(row["snapshot_position"])
        labels = labels_by_category[category]
        changed = labels.copy()
        changed[t:] = 1 - changed[t:]
        if not np.array_equal(labels[:t], changed[:t]):
            raise RuntimeError("N4 altered labels before the snapshot query prefix")
        original_action = _query_only_action(row, private_rows)
        changed_future_action = _query_only_action(row, private_rows)
        if original_action != row["selected_action"] or changed_future_action != original_action:
            raise RuntimeError("N4 future-label mutation changed a query-only action")
        baseline.append({"category": category, "seed": int(row["seed"]),
                         "snapshot_position": t, "selected_action": original_action})
        altered.append({"category": category, "seed": int(row["seed"]),
                        "snapshot_position": t, "selected_action": changed_future_action})
        checked_prefixes += 1
    return {"status": "PASS", "future_labels_flipped_at_or_after_each_snapshot": True,
            "pre_snapshot_label_prefixes_identical": True,
            "query_only_action_recomputed_for_states": checked_prefixes,
            "baseline_action_digest": action_digest(baseline),
            "altered_future_action_digest": action_digest(altered),
            "digests_identical": action_digest(baseline) == action_digest(altered)}


def run(*, output: Path, source: Path = DEFAULT_SOURCE,
        stage05_dir: Path = DEFAULT_STAGE05,
        json_root: Path = DEFAULT_JSON_ROOT) -> dict[str, Any]:
    output, source, stage05_dir, json_root = map(Path,
        (output, source, stage05_dir, json_root))
    pre_path = output / "pre_future_freeze.json"
    action_path = output / "query_actions.json"
    if not pre_path.is_file() or not action_path.is_file():
        raise RuntimeError("pre-future action freeze is absent")
    pre = _read_json(pre_path)
    actions_payload = _read_json(action_path)
    if pre.get("status") != "FROZEN_BEFORE_FUTURE_LABELS":
        raise RuntimeError("query actions do not have a frozen pre-future status")
    if sha256_file(action_path) != pre.get("query_actions_file_sha256"):
        raise RuntimeError("query action file changed after its freeze")
    if actions_payload.get("action_digest") != pre.get("query_action_digest"):
        raise RuntimeError("query action digest differs from pre-future freeze")
    if action_digest(actions_payload["rows"]) != pre.get("query_action_digest"):
        raise RuntimeError("canonical query action digest is invalid")
    if sha256_file(ROOT / "docs/calibration/QUERIED_ANCHOR_IDENTIFIABILITY_PROTOCOL_CANDIDATE_v1.md") != pre.get("protocol_sha256"):
        raise RuntimeError("frozen candidate protocol changed before future evaluation")
    branch = __import__("subprocess").run(
        ["git", "branch", "--show-current"], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout.strip()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"future evaluator branch mismatch: {branch}")

    historical_digest_before = _artifact_digest(source)
    if historical_digest_before != pre.get("source_artifact_digest_before"):
        raise RuntimeError("historical source artifact digest differs from pre-future freeze")
    # The next statements are the first evaluator-label access in this run.
    provenance, categories = load_verified_inputs(source.resolve(), stage05_dir.resolve())
    protocol075a = _read_json(source / "protocol_manifest.json")
    labels_by_category: dict[str, np.ndarray] = {}
    label_hashes: dict[str, str] = {}
    for category in DEVELOPMENT_CLASSES:
        expected = protocol075a["metadata_sha256_by_class"][category]
        labels, label_sha = _labels_without_masks(json_root, category, expected)
        if len(labels) != len(categories[category].events):
            raise RuntimeError(f"{category}: evaluator label length mismatch")
        labels_by_category[category] = labels
        label_hashes[category] = label_sha

    private_path = output / "private" / "query_anchor_measurements.json.gz"
    with gzip.open(private_path, "rt", encoding="utf-8") as stream:
        private_measurements = json.load(stream)
    if private_measurements.get("schema") != "queried_anchor_identifiability.private_measurements.v1":
        raise RuntimeError("private query-measurement schema mismatch")
    n4 = _leakage_control(actions_payload["rows"],
                          private_measurements["rows"], labels_by_category)
    if not n4["digests_identical"]:
        raise RuntimeError("N4 query action digest changed under altered future labels")

    replay = replay_all(categories)
    arrays_by_category = {category: categories[category].arrays
                          for category in DEVELOPMENT_CLASSES}
    result_rows: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = {
        category: [] for category in DEVELOPMENT_CLASSES}
    for estimate in actions_payload["rows"]:
        category = str(estimate["category"])
        seed = int(estimate["seed"])
        t = int(estimate["snapshot_position"])
        if category not in DEVELOPMENT_CLASSES or t not in SNAPSHOT_POSITIONS:
            raise RuntimeError("query action row is outside frozen evaluation design")
        n = len(labels_by_category[category])
        snapshot_scores = np.asarray(
            arrays_by_category[category][f"snapshot_s{seed}_t{t}"], dtype=np.float64)
        if snapshot_scores.shape != (n,):
            raise RuntimeError("historical frozen snapshot score length mismatch")
        if np.isfinite(snapshot_scores[:t]).any():
            raise RuntimeError("historical snapshot score vector contains pre-snapshot rows")
        finite = np.isfinite(snapshot_scores)
        cohorts: dict[str, np.ndarray] = {}
        next_mask = np.zeros(n, dtype=bool)
        next_mask[t:min(n, t + SNAPSHOT_NEXT_WIDTH)] = True
        cohorts["next256_group_clean"] = finite & next_mask
        cohorts["all_remaining_group_clean"] = finite & (np.arange(n) >= t)
        threshold, threshold_status = _resolve_snapshot_threshold(
            replay["trajectories"][category][str(seed)]["theta_pre"][t],
            str(estimate.get("score_status", "BLOCKED")),
            estimate.get("b3_threshold"))
        state_results = {}
        for cohort_name, mask in cohorts.items():
            cohort_labels = labels_by_category[category][mask]
            cohort_scores = snapshot_scores[mask]
            if threshold_status != "PASS":
                undefined = {"status": threshold_status, "n": int(mask.sum()),
                            "normal_n": None, "defect_n": None,
                            "fpr": None, "fnr": None, "j": None, "ba": None}
                empty_gains = {key: {"status": threshold_status, "offset": offset,
                                     "threshold": None, "delta_j_vs_keep": None,
                                     "delta_ba_vs_keep": None, "delta_fpr": None,
                                     "delta_fnr": None}
                               for key, offset in (("-0.010000", -0.01),
                                                  ("+0.000000", 0.0),
                                                  ("+0.010000", 0.01))}
                state_results[cohort_name] = {
                    "future_base": undefined, "future_primary_gains": empty_gains,
                    "future_secondary_gains": {},
                    "future_best_primary_action": {"action": None, "unique": False,
                                                    "reason": threshold_status},
                    "query_selected_action": "ABSTAIN",
                    "query_selected_future_delta_j": None,
                    "query_selected_future_delta_ba": None,
                    "false_activation": None,
                    "future_gain_after_label_flip_n4": empty_gains,
                    "eligible_score_count": int(mask.sum()),
                }
                continue
            primary = _future_gains(cohort_labels, cohort_scores, threshold,
                                    (-0.01, 0.0, 0.01))
            secondary = _future_gains(cohort_labels, cohort_scores, threshold,
                                      (-0.005, 0.0, 0.005))
            best = _future_best(primary["gains"])
            selected_action = str(estimate["selected_action"])
            selected_key = {"LOWER": "-0.010000", "KEEP": "+0.000000",
                            "RAISE": "+0.010000"}.get(selected_action)
            selected_delta = (None if selected_key is None else
                              primary["gains"][selected_key]["delta_j_vs_keep"])
            selected_ba = (None if selected_key is None else
                           primary["gains"][selected_key]["delta_ba_vs_keep"])
            defined_ba_gains = [primary["gains"][key]["delta_ba_vs_keep"]
                                for key in ("-0.010000", "+0.000000", "+0.010000")]
            selected_minus_best = (
                None if selected_ba is None or any(value is None for value in defined_ba_gains)
                else float(selected_ba - max(defined_ba_gains)))
            if selected_action == "ABSTAIN":
                false_activation = None
            elif selected_action in ("LOWER", "RAISE") and selected_delta is not None:
                false_activation = bool(selected_delta < 0.0)
            else:
                false_activation = False if selected_delta is not None else None
            control_bits = np.asarray(labels_by_category[category], dtype=np.int8).copy()
            control_bits[t:] = 1 - control_bits[t:]
            mutated = _future_gains(control_bits[mask], cohort_scores, threshold,
                                    (-0.01, 0.0, 0.01))
            # Future-label mutation changes evaluation metrics, while the
            # query-only estimator has already been independently recomputed.
            if (sha256_file(action_path) != pre["query_actions_file_sha256"] or
                    actions_payload["action_digest"] != pre["query_action_digest"]):
                raise RuntimeError("N4 future-label mutation altered query actions")
            state_results[cohort_name] = {
                "future_base": primary["base"],
                "future_primary_gains": primary["gains"],
                "future_secondary_gains": secondary["gains"],
                "future_best_primary_action": best,
                "query_selected_action": selected_action,
                "query_selected_future_delta_j": selected_delta,
                "query_selected_future_delta_ba": selected_ba,
                "query_selected_minus_future_best_delta_ba": selected_minus_best,
                "false_activation": false_activation,
                "future_gain_after_label_flip_n4": mutated["gains"],
                "eligible_score_count": int(mask.sum()),
            }
        primary_estimate = estimate.get("primary", {})
        row = {"category": category, "seed": seed, "snapshot_position": t,
               "query_action": estimate["selected_action"],
               "query_action_reason": estimate["selection_reason"],
               "query_estimate_status": primary_estimate.get("status", "SCORE_INCOMPLETE"),
               "query_action_identifiable": (
                   primary_estimate.get("status") == "IDENTIFIABLE_TWO_CLASS_EMPIRICAL"),
               "query_normal_groups": estimate["normal_groups"],
               "query_defect_groups": estimate["defect_groups"],
               "state_digest": estimate["snapshot_memory_sha256"],
               "cohorts": state_results}
        result_rows.append(row)
        by_category[category].append(row)

    # Report class-first summaries; snapshots/cohorts remain clustered rows.
    summary_by_class: dict[str, Any] = {}
    for category, rows in by_category.items():
        query_rows = [row for row in actions_payload["rows"]
                      if str(row["category"]) == category]
        query_identifiable_count = sum(
            row.get("primary", {}).get("status") == "IDENTIFIABLE_TWO_CLASS_EMPIRICAL"
            for row in query_rows)
        query_counts = {action: sum(row["selected_action"] == action for row in query_rows)
                        for action in ("LOWER", "KEEP", "RAISE", "ABSTAIN")}
        cohort_summaries: dict[str, Any] = {}
        for cohort_name in ("next256_group_clean", "all_remaining_group_clean"):
            defined = [row["cohorts"][cohort_name] for row in rows
                       if row["cohorts"][cohort_name]["future_base"]["status"] == "DEFINED"]
            identifiable = [value for value in defined
                            if value["query_selected_action"] != "ABSTAIN"]
            unique_best = [value for value in defined
                           if value["future_best_primary_action"]["unique"]]
            matched = [value for value in unique_best
                       if value["query_selected_action"] != "ABSTAIN"]
            false_acts = [value for value in identifiable
                          if value["query_selected_action"] in ("LOWER", "RAISE")]
            false_activated = [value for value in false_acts
                               if value["false_activation"] is True]
            selected_gains = [value["query_selected_future_delta_ba"]
                              for value in identifiable
                              if value["query_selected_future_delta_ba"] is not None]
            regret = []
            for value in identifiable:
                gains = value["future_primary_gains"]
                entries = [gains[key]["delta_ba_vs_keep"]
                           for key in ("-0.010000", "+0.000000", "+0.010000")]
                chosen = value["query_selected_future_delta_ba"]
                if chosen is not None and all(x is not None for x in entries):
                    regret.append(float(max(entries) - chosen))
            cohort_summaries[cohort_name] = {
                "future_defined_rows": len(defined),
                "future_one_class_rows": sum(
                    row["cohorts"][cohort_name]["future_base"]["status"] ==
                    "UNDEFINED_ONE_CLASS"
                    for row in rows),
                "future_blocked_native_rows": sum(
                    row["cohorts"][cohort_name]["future_base"]["status"] in
                    ("BLOCKED_NATIVE_SCORE", "BLOCKED_B3_THRESHOLD")
                    for row in rows),
                "query_identifiable_defined_rows": len(identifiable),
                "unique_future_best_rows": len(unique_best),
                "direction_agreement": (None if not matched else
                    sum(value["query_selected_action"] ==
                        value["future_best_primary_action"]["action"]
                        for value in matched) / len(matched)),
                "direction_agreement_numerator": sum(
                    value["query_selected_action"] ==
                    value["future_best_primary_action"]["action"]
                    for value in matched),
                "direction_agreement_denominator": len(matched),
                "false_activation_numerator": len(false_activated),
                "false_activation_denominator": len(false_acts),
                "false_activation_rate": (None if not false_acts else
                                           len(false_activated) / len(false_acts)),
                "harm_magnitude_conditional_mean_ba_loss": (
                    None if not false_activated else
                    float(np.mean([-value["query_selected_future_delta_ba"]
                                   for value in false_activated]))),
                "mean_selected_future_delta_ba": (None if not selected_gains else
                                                  float(np.mean(selected_gains))),
                "mean_loss_to_best_registered_ba": (None if not regret else
                                                     float(np.mean(regret))),
                "abstention_count": sum(value["query_selected_action"] == "ABSTAIN"
                                         for value in [row["cohorts"][cohort_name]
                                                       for row in rows]),
                "query_action_counts": {
                    action: sum(row["query_action"] == action for row in rows)
                    for action in ("LOWER", "KEEP", "RAISE", "ABSTAIN")},
            }
        summary_by_class[category] = {
            "snapshot_state_count": len(query_rows),
            "query_identifiable_state_count": int(query_identifiable_count),
            "query_identifiable_coverage": (
                None if not query_rows else query_identifiable_count / len(query_rows)),
            "query_action_counts": query_counts,
            "cohorts": cohort_summaries,
        }

    equal_class_macro: dict[str, Any] = {}
    for cohort_name in ("next256_group_clean", "all_remaining_group_clean"):
        class_summaries = [summary_by_class[category]["cohorts"][cohort_name]
                           for category in DEVELOPMENT_CLASSES]
        metric_names = ("direction_agreement", "false_activation_rate",
                        "harm_magnitude_conditional_mean_ba_loss",
                        "mean_selected_future_delta_ba",
                        "mean_loss_to_best_registered_ba")
        metrics: dict[str, Any] = {}
        for metric in metric_names:
            categories = [category for category in DEVELOPMENT_CLASSES
                          if summary_by_class[category]["cohorts"][cohort_name][metric]
                          is not None]
            values = [float(summary_by_class[category]["cohorts"][cohort_name][metric])
                      for category in categories]
            metrics[metric] = {
                "equal_class_mean": None if not values else float(np.mean(values)),
                "contributing_class_count": len(values),
                "categories": categories,
            }
        equal_class_macro[cohort_name] = metrics

    by_snapshot_position: dict[str, Any] = {}
    for position in SNAPSHOT_POSITIONS:
        state_rows = [row for row in result_rows
                      if row["snapshot_position"] == int(position)]
        by_cohort: dict[str, Any] = {}
        for cohort_name in ("next256_group_clean", "all_remaining_group_clean"):
            selected_gains = [row["cohorts"][cohort_name][
                "query_selected_future_delta_ba"] for row in state_rows
                if row["cohorts"][cohort_name][
                    "query_selected_future_delta_ba"] is not None]
            by_cohort[cohort_name] = {
                "future_defined_rows": sum(
                    row["cohorts"][cohort_name]["future_base"]["status"] == "DEFINED"
                    for row in state_rows),
                "query_identifiable_defined_rows": sum(
                    row["query_action_identifiable"] and
                    row["cohorts"][cohort_name]["future_base"]["status"] == "DEFINED"
                    for row in state_rows),
                "mean_selected_future_delta_ba": (
                    None if not selected_gains else float(np.mean(selected_gains))),
            }
        by_snapshot_position[str(position)] = {
            "state_count": len(state_rows),
            "query_identifiable_count": sum(row["query_action_identifiable"]
                                             for row in state_rows),
            "query_action_counts": {
                action: sum(row["query_action"] == action for row in state_rows)
                for action in ("LOWER", "KEEP", "RAISE", "ABSTAIN")},
            "by_cohort": by_cohort,
        }

    query_action_counts = {
        action: sum(row["query_action"] == action for row in result_rows)
        for action in ("LOWER", "KEEP", "RAISE", "ABSTAIN")}
    by_defect_evidence_bin: dict[str, Any] = {}
    for bin_name in ("0", "1-2", "3-7", "8-13", "14+"):
        bin_rows = [row for row in result_rows
                    if defect_evidence_bin(int(row["query_defect_groups"])) == bin_name]
        cohort_data: dict[str, Any] = {}
        for cohort_name in ("next256_group_clean", "all_remaining_group_clean"):
            defined = [row["cohorts"][cohort_name] for row in bin_rows
                       if row["cohorts"][cohort_name]["future_base"]["status"] == "DEFINED"]
            identified = [row for row in defined
                          if row["query_selected_action"] != "ABSTAIN"]
            active = [row for row in identified
                      if row["query_selected_action"] in ("LOWER", "RAISE")]
            false_activated = [row for row in active if row["false_activation"] is True]
            selected_gain = [row["query_selected_future_delta_ba"] for row in identified
                             if row["query_selected_future_delta_ba"] is not None]
            cohort_data[cohort_name] = {
                "state_count": len(bin_rows),
                "future_defined_count": len(defined),
                "query_identifiable_future_defined_count": len(identified),
                "active_action_count": len(active),
                "false_activation_count": len(false_activated),
                "false_activation_rate": (None if not active else
                                           len(false_activated) / len(active)),
                "mean_selected_future_delta_ba": (
                    None if not selected_gain else float(np.mean(selected_gain))),
                "query_action_counts": {
                    action: sum(row["query_action"] == action for row in bin_rows)
                    for action in ("LOWER", "KEEP", "RAISE", "ABSTAIN")},
            }
        by_defect_evidence_bin[bin_name] = cohort_data
    n4.update({"future_label_flip_changes_evaluation_only": True,
               "query_action_digest_unchanged": True,
               "original_query_action_file_sha256": pre["query_actions_file_sha256"]})
    historical_digest_after = _artifact_digest(source)
    if historical_digest_after != historical_digest_before:
        raise RuntimeError("historical Stage 0.75-A artifacts changed during future evaluation")
    return {"schema": "queried_anchor_identifiability.future.v1",
            "status": "PASS", "labels_accessed_after_action_freeze": True,
            "mask_paths_decoded": False, "confirmation_labels_accessed": False,
            "query_action_digest": pre["query_action_digest"],
            "n4_future_label_leakage_control": n4,
            "historical_artifact_digest_before": historical_digest_before,
            "historical_artifact_digest_after": historical_digest_after,
            "historical_artifacts_unchanged": True,
            "rows": result_rows, "summary_by_class": summary_by_class,
            "query_action_counts_120_state_rows": query_action_counts,
            "equal_class_macro": equal_class_macro,
            "by_snapshot_position": by_snapshot_position,
            "by_query_defect_evidence_bin": by_defect_evidence_bin,
            "evaluator_label_hash_by_class_private": label_hashes,
            "independence_claim": "temporally separated within previously exposed development classes; not independent test-set confirmation"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--stage05", type=Path, default=DEFAULT_STAGE05)
    parser.add_argument("--json-root", type=Path, default=DEFAULT_JSON_ROOT)
    args = parser.parse_args(argv)
    try:
        result = run(output=args.output, source=args.source,
                     stage05_dir=args.stage05, json_root=args.json_root)
        _atomic_json(args.output / "future_evaluation.json", result)
        print(json.dumps({"status": result["status"],
                          "rows": len(result["rows"]),
                          "action_digest": result["query_action_digest"],
                          "n4": result["n4_future_label_leakage_control"]["status"]},
                         sort_keys=True))
        return 0
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
