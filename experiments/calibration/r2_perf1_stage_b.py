"""Stage B: physically separate, outcome-only evaluator for sealed R2-PERF-1 scores.

The parser skips TRAIN and all non-prefix TEST values structurally. It projects
anomaly_class only from the first 512 TEST rows and only after each row's
official image identity has matched the already sealed Stage A identity at the
same position. Masks and later TEST outcomes are never materialized.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


class StageBError(RuntimeError):
    pass


class _TestPrefixProjection:
    """Subprocess-local JSON projection that never materializes TRAIN labels."""

    def __init__(self, text: str):
        from experiments.calibration.r2_l2c_metadata_census import _ProjectionParser
        self.parser = _ProjectionParser(text)
        self.meta: dict[str, Any] | None = None
        self.normal_class: str | None = None
        self.prefix: str | None = None
        self.projected_rows = 0
        self._target_records: list[tuple[int, str, int, int]] = []

    def project(self, targets: Sequence[Mapping[str, Any]], row_callback) -> int:
        p = self.parser
        p._expect("{")
        observed_top: set[str] = set()
        saw_train = False
        saw_test = False
        p._ws()
        while p.index < len(p.text) and p.text[p.index] != "}":
            key = p._read_key()
            if key in observed_top:
                raise StageBError(f"Duplicate top-level switch JSON section: {key}")
            observed_top.add(key)
            p._expect(":")
            if key == "meta":
                self.meta = p._project_object(frozenset({"normal_class", "prefix"}))
            elif key == "train":
                saw_train = True
                p._skip_value()
            elif key == "test":
                saw_test = True
                self._project_test_prefix(len(targets))
            else:
                raise StageBError(f"Unrecognized switch JSON section: {key}")
            p._ws()
            if p.index < len(p.text) and p.text[p.index] == ",":
                p.index += 1
                continue
            break
        p._expect("}")
        p._ws()
        if p.index != len(p.text):
            raise StageBError("Trailing bytes follow the official switch JSON")
        if not saw_train or not saw_test or self.meta is None:
            raise StageBError("switch JSON lacks meta, train, or test")
        self.normal_class = self.meta.get("normal_class")
        self.prefix = self.meta.get("prefix")
        if not isinstance(self.normal_class, str) or not self.normal_class:
            raise StageBError("switch metadata has no valid normal_class")
        if not isinstance(self.prefix, str) or not self.prefix:
            raise StageBError("switch metadata has no valid prefix")
        for row_index, image_path, start, end in self._target_records:
            row_callback(row_index, image_path, p.text[start:end])
            self.projected_rows += 1
        return self.projected_rows

    def _project_test_prefix(self, target_count: int) -> None:
        p = self.parser
        p._expect("[")
        row_index = 0
        p._ws()
        if p.index < len(p.text) and p.text[p.index] == "]":
            p.index += 1
            return
        while True:
            p._ws()
            if p.index >= len(p.text) or p.text[p.index] != "{":
                raise StageBError("Official TEST records must be objects")
            if row_index >= target_count:
                # Do not even materialize TEST identities or outcomes outside the
                # exact authorized 512-position prefix.
                p._skip_value()
            else:
                start = p.index
                identity = p._project_object(frozenset({"image_path"}))
                end = p.index
                image_path = identity.get("image_path")
                if not isinstance(image_path, str) or not image_path:
                    raise StageBError(f"TEST image_path is absent at position {row_index}")
                self._target_records.append((row_index, image_path, start, end))
            row_index += 1
            p._ws()
            if p.index < len(p.text) and p.text[p.index] == ",":
                p.index += 1
                continue
            p._expect("]")
            break


def _project_test_label_for_matched_record(raw_record: str) -> str:
    from experiments.calibration.r2_l2c_metadata_census import _ProjectionParser
    parser = _ProjectionParser(raw_record)
    projected = parser._project_object(frozenset({"anomaly_class"}))
    label = projected.get("anomaly_class")
    if not isinstance(label, str) or not label:
        raise StageBError("Matched TEST record has no string anomaly_class")
    return label


def _matched_test_label(raw_record: str, *, expected_identity: str,
                        observed_identity: str) -> str:
    """Decode one outcome only after exact event-identity alignment succeeds."""
    if expected_identity != observed_identity:
        raise StageBError("TEST outcome is not aligned to the sealed event identity")
    return _project_test_label_for_matched_record(raw_record)


def _average_ranks(values):
    import numpy as np
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    return ranks


def _auc_ap(labels, scores) -> tuple[float | None, float | None, str | None]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        reason = "AUROC and AUPRC require both classes" if positives == 0 or negatives == 0 else None
        return None, None, reason
    ranks = _average_ranks(s)
    auc = float((ranks[y == 1].sum() - positives * (positives + 1) / 2.0) /
                (positives * negatives))
    order = np.argsort(-s, kind="mergesort")
    sorted_labels = y[order]
    tp = np.cumsum(sorted_labels, dtype=np.int64)
    fp = np.cumsum(1 - sorted_labels, dtype=np.int64)
    # Average precision changes only at positive observations. Equal-score
    # records share a threshold; the precision for that threshold is computed
    # after consuming the full tie group.
    ap = 0.0
    prior_recall = 0.0
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and s[order[end]] == s[order[index]]:
            end += 1
        group_tp = int(tp[end - 1])
        group_fp = int(fp[end - 1])
        recall = group_tp / positives
        precision = group_tp / (group_tp + group_fp)
        ap += (recall - prior_recall) * precision
        prior_recall = recall
        index = end
    return auc, float(ap), None


def _fixed_threshold_metrics(labels, scores, threshold: float) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    pred = s >= float(threshold)
    positive = y == 1
    negative = y == 0
    tp = int(np.count_nonzero(pred & positive))
    fp = int(np.count_nonzero(pred & negative))
    fn = int(np.count_nonzero(~pred & positive))
    tn = int(np.count_nonzero(~pred & negative))
    fpr = fp / (fp + tn) if fp + tn else None
    fnr = fn / (fn + tp) if fn + tp else None
    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    return {
        "threshold": float(threshold), "threshold_source": "frozen R1 C64 theta0_higher_q90",
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fpr": fpr, "fnr": fnr,
        "balanced_accuracy": ((tpr + tnr) / 2.0 if tpr is not None and tnr is not None else None),
    }


def _metric_record(labels, scores, threshold: float) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    auc, ap, reason = _auc_ap(y, s)
    return {
        "image_auroc": auc,
        "image_auprc_average_precision": ap,
        "defect_prevalence": float(y.mean()) if len(y) else None,
        "undefined_reason": reason,
        "fixed_threshold": _fixed_threshold_metrics(y, s, threshold),
    }


def _cluster_bootstrap(labels, groups, scores_by_arm: Mapping[str, Any], *,
                       iterations: int = 1000, seed: int = 20261101) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    g = np.asarray(groups, dtype=object)
    unique_groups = sorted(set(g.tolist()))
    member_indices = {group: np.flatnonzero(g == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    auc_samples: dict[str, list[float]] = {arm: [] for arm in scores_by_arm}
    ap_samples: dict[str, list[float]] = {arm: [] for arm in scores_by_arm}
    diff_samples: dict[str, list[float]] = {arm: [] for arm in scores_by_arm if arm != "c0"}
    c3_vs_gate: list[float] = []
    arm_arrays = {arm: np.asarray(scores, dtype=np.float64)
                  for arm, scores in scores_by_arm.items()}
    for _ in range(iterations):
        selected = rng.integers(0, len(unique_groups), size=len(unique_groups))
        indices = np.concatenate([member_indices[unique_groups[int(i)]] for i in selected])
        boot_y = y[indices]
        boot_metrics: dict[str, tuple[float, float] | None] = {}
        for arm, score in arm_arrays.items():
            auc, ap, _ = _auc_ap(boot_y, score[indices])
            boot_metrics[arm] = (auc, ap) if auc is not None and ap is not None else None
            if boot_metrics[arm] is not None:
                auc_samples[arm].append(float(auc))
                ap_samples[arm].append(float(ap))
            if arm != "c0" and boot_metrics[arm] is not None and boot_metrics["c0"] is not None:
                diff_samples[arm].append(float(auc - boot_metrics["c0"][0]))
        primary_c3 = boot_metrics.get("c3_r0.10_l0.10")
        primary_gate = boot_metrics.get("c2gated_r0.10_l0.10")
        if primary_c3 is not None and primary_gate is not None:
            c3_vs_gate.append(primary_c3[0] - primary_gate[0])

    def interval(values):
        if not values:
            return {"lower_2_5": None, "upper_97_5": None, "valid_replicates": 0}
        array = np.asarray(values, dtype=np.float64)
        return {"lower_2_5": float(np.quantile(array, 0.025)),
                "upper_97_5": float(np.quantile(array, 0.975)),
                "valid_replicates": int(array.size)}

    return {
        "method": "paired Source Group cluster bootstrap with replacement",
        "replicates_requested": iterations, "seed": seed,
        "unique_source_groups": len(unique_groups),
        "conditional_on_fixed_historical_causal_trajectory": True,
        "not_uncertainty_over_independently_regenerated_online_streams": True,
        "auroc_95_percent_intervals": {arm: interval(values) for arm, values in auc_samples.items()},
        "auprc_95_percent_intervals": {arm: interval(values) for arm, values in ap_samples.items()},
        "auroc_difference_vs_c0_95_percent_intervals": {
            arm: interval(values) for arm, values in diff_samples.items()},
        "primary_c3_minus_c2_gated_auroc_95_percent_interval": interval(c3_vs_gate),
    }


def _rank_reversals(base, candidate, groups) -> dict[str, Any]:
    import numpy as np
    x = np.asarray(base, dtype=np.float64)
    y = np.asarray(candidate, dtype=np.float64)
    g = np.asarray(groups, dtype=object)
    strict = reverse = same_group = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if g[i] == g[j]:
                same_group += 1
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx != 0.0 and dy != 0.0:
                strict += 1
                reverse += int(dx * dy < 0.0)
    return {"cross_source_group_strictly_comparable_pairs": strict,
            "cross_source_group_reversed_pairs": reverse,
            "reversal_fraction": reverse / strict if strict else None,
            "same_source_group_pairs_excluded": same_group}


def _score_direction(labels, score_delta) -> dict[str, Any]:
    import numpy as np
    y = np.asarray(labels, dtype=np.int8)
    delta = np.asarray(score_delta, dtype=np.float64)
    correct = ((y == 1) & (delta > 0)) | ((y == 0) & (delta < 0))
    harmful = ((y == 1) & (delta < 0)) | ((y == 0) & (delta > 0))
    neutral = delta == 0
    return {
        "correct_direction_events": int(correct.sum()),
        "harmful_direction_events": int(harmful.sum()),
        "unchanged_score_events": int(neutral.sum()),
        "anomaly_score_delta_mean": float(delta[y == 1].mean()) if np.any(y == 1) else None,
        "normal_score_delta_mean": float(delta[y == 0].mean()) if np.any(y == 0) else None,
    }


def _quantile_interval(values):
    import numpy as np
    x = np.asarray(values, dtype=np.float64)
    if not x.size:
        return {"count": 0, "mean": None, "p2_5": None, "p97_5": None}
    return {"count": int(x.size), "mean": float(x.mean()),
            "p2_5": float(np.quantile(x, 0.025)),
            "p97_5": float(np.quantile(x, 0.975))}


def _load_and_verify_sealed_scores(output_root: Path, seal: Mapping[str, Any]):
    from experiments.calibration.r2_perf1_common import (
        EVENT_COUNT, SCORE_TABLE_SCHEMA, sha256_file)
    path = output_root / "stage_a_scores.jsonl"
    if sha256_file(path) != seal.get("score_table_sha256"):
        raise StageBError("Stage A score table SHA-256 differs from its seal")
    rows = []
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                raise StageBError("Stage A score table has a torn final record")
            row = json.loads(line)
            if row.get("schema") != "calibration.r2_perf1.score_row.v1":
                raise StageBError("Unexpected Stage A score row schema")
            rows.append(row)
    if len(rows) != EVENT_COUNT or seal.get("event_count") != EVENT_COUNT:
        raise StageBError("Stage A seal does not contain exactly 512 score records")
    required_arms = {"c0", "c1", "c2"}
    for position, row in enumerate(rows):
        scores = row.get("scores")
        if row.get("position") != position or not isinstance(scores, dict) or not required_arms.issubset(scores):
            raise StageBError(f"Malformed or misaligned Stage A score record at {position}")
        if len(row.get("source_image_identity_sha256", "")) != 64 or len(row.get("source_group_sha256", "")) != 64:
            raise StageBError(f"Malformed Stage A identity at {position}")
        if any(not math.isfinite(float(value)) for value in scores.values()):
            raise StageBError(f"Nonfinite Stage A score at {position}")
        if len(row.get("argmax_patch", {})) != len(scores):
            raise StageBError(f"Missing candidate argmax record at {position}")
    if seal.get("status") != "SEALED" or seal.get("outcome_labels_read") is not False:
        raise StageBError("Stage A has not passed the explicit label-blind score seal")
    return rows


def _read_outcomes(rows, *, test_json_path: Path):
    from experiments.calibration.r2_perf1_common import EVENT_COUNT, sha256_bytes, require_plain_path
    from experiments.risk_safe.preflight import parse_image_name

    test_json_path = require_plain_path(test_json_path, must_exist=True)
    if not test_json_path.is_file():
        raise StageBError("Authorized switch TEST metadata path is not a file")
    try:
        text = test_json_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise StageBError(f"Could not read authorized switch TEST metadata: {exc}") from exc
    target_rows = rows[:EVENT_COUNT]
    labels: list[int] = []
    groups: list[str] = []
    expected_ids = [str(row["source_image_identity_sha256"]) for row in target_rows]
    expected_groups = [str(row["source_group_sha256"]) for row in target_rows]

    def project_one(position: int, image_path: str, raw_record: str):
        parser = projection_holder[0]
        if parser.prefix is None:
            # The meta object can occur after the TEST array in valid JSON. The
            # data source used here has meta first; fail closed otherwise.
            raise StageBError("switch metadata prefix must precede its TEST array")
        absolute_identity_path = Path("G:/dataset/realiad/images") / parser.prefix / image_path
        parsed = parse_image_name(absolute_identity_path, expected_class="switch")
        identity = sha256_bytes(str(parsed.rel_path).encode("utf-8"))
        group = sha256_bytes(
            f"official-sample-group\0{parsed.official_sample_key}".encode("utf-8"))
        if identity != expected_ids[position] or group != expected_groups[position]:
            raise StageBError(
                f"TEST image identity/source group does not align with sealed event {position}")
        # The identity has now matched the sealed position. Only this single
        # target record's anomaly_class string is projected; masks and all
        # other TEST outcomes remain structurally skipped.
        anomaly_class = _matched_test_label(
            raw_record, expected_identity=expected_ids[position],
            observed_identity=identity)
        label = 0 if anomaly_class == parser.normal_class else 1
        labels.append(label)
        groups.append(group)

    projection = _TestPrefixProjection(text)
    projection_holder = [projection]
    projected = projection.project(target_rows, project_one)
    if projected != EVENT_COUNT or len(labels) != EVENT_COUNT:
        raise StageBError("Could not project exactly 512 identity-matched TEST outcomes")
    if len(set(expected_ids)) != EVENT_COUNT or len(set(expected_groups)) == 0:
        raise StageBError("Sealed event identities are duplicated or group provenance is empty")
    return labels, groups, {
        "status": "512_TEST_OUTCOMES_IDENTITY_POSITION_VERIFIED",
        "projected_target_test_outcomes": EVENT_COUNT,
        "target_test_positions": [0, EVENT_COUNT - 1],
        "test_identity_and_source_group_match": True,
        "train_labels_materialized": False,
        "test_labels_materialized_outside_target_prefix": False,
        "mask_fields_materialized": False,
        "official_test_source_sha256": None,
        "official_test_source_sha256_status": (
            "NOT_COMPUTED: full JSON hashing would read TRAIN and non-target TEST label bytes"),
    }


def evaluate(output_root: Path, *, bootstrap_replicates: int = 1000) -> dict[str, Any]:
    import numpy as np
    from experiments.calibration.r2_perf1_common import (
        EVENT_COUNT, EVIDENCE_RELATIVE, ROOT, OutputBudget, Perf1Error,
        PRIMARY_LAMBDA, PRIMARY_RADIUS, RADII, LAMBDAS, require_plain_path,
        require_worker, sha256_json, source_fingerprints, utc_now,
        verify_source_fingerprints)

    require_worker("stage_b")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    start_wall = time.monotonic()
    start_cpu = time.process_time()
    output_root = require_plain_path(output_root, must_exist=True)
    budget = OutputBudget(output_root, allow_existing=True)
    try:
        manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
        verify_source_fingerprints(manifest["source_fingerprints"])
        seal = json.loads((output_root / "stage_a_sealed.json").read_text(encoding="utf-8"))
        verify_source_fingerprints(seal["source_fingerprints"])
        if seal.get("method_configuration_sha256") != manifest.get("configuration_sha256"):
            raise StageBError("Stage A seal/configuration digest mismatch")
        rows = _load_and_verify_sealed_scores(output_root, seal)
        test_path = Path(r"G:\dataset\realiad\jsons\realiad_jsons\switch.json")
        labels, groups, alignment = _read_outcomes(rows, test_json_path=test_path)
        if len(groups) != EVENT_COUNT:
            raise StageBError("Outcome group count differs from the sealed score table")
        arm_names = list(rows[0]["scores"])
        scores_by_arm = {
            arm: np.asarray([float(row["scores"][arm]) for row in rows], dtype=np.float64)
            for arm in arm_names
        }
        label_array = np.asarray(labels, dtype=np.int8)
        group_array = np.asarray(groups, dtype=object)
        theta = float(manifest["frozen_threshold"])
        if theta != float(manifest["r1_c64_theta0_higher_q90"]):
            raise StageBError("Frozen threshold was changed between launch and evaluation")
        primary = {
            "C0": "c0", "C1": "c1", "C2": "c2",
            "C2-Gated": "c2gated_r0.10_l0.10", "C3": "c3_r0.10_l0.10",
        }
        if any(arm not in scores_by_arm for arm in primary.values()):
            raise StageBError("Required primary method score is missing")
        all_metrics = {arm: _metric_record(label_array, scores, theta)
                       for arm, scores in scores_by_arm.items()}
        bootstrap = _cluster_bootstrap(
            label_array, group_array, scores_by_arm,
            iterations=int(bootstrap_replicates), seed=20261101)
        primary_metrics: dict[str, Any] = {}
        c0_auc = all_metrics["c0"]["image_auroc"]
        for display, arm in primary.items():
            metric = dict(all_metrics[arm])
            metric["score_arm"] = arm
            metric["auroc_difference_vs_c0"] = (
                float(metric["image_auroc"] - c0_auc)
                if metric["image_auroc"] is not None and c0_auc is not None else None)
            metric["cluster_bootstrap_auroc_95_percent_interval"] = bootstrap[
                "auroc_95_percent_intervals"][arm]
            metric["cluster_bootstrap_auroc_difference_vs_c0_95_percent_interval"] = (
                None if arm == "c0" else bootstrap[
                    "auroc_difference_vs_c0_95_percent_intervals"].get(arm))
            metric["cluster_bootstrap_auprc_95_percent_interval"] = bootstrap[
                "auprc_95_percent_intervals"][arm]
            primary_metrics[display] = metric
        c3_primary = primary_metrics["C3"]["image_auroc"]
        gated_primary = primary_metrics["C2-Gated"]["image_auroc"]
        primary_delta = (float(c3_primary - gated_primary)
                         if c3_primary is not None and gated_primary is not None else None)

        sensitivity = []
        for radius in RADII:
            for regularizer in LAMBDAS:
                gated = f"c2gated_r{radius:.2f}_l{regularizer:.2f}"
                compact = f"c3_r{radius:.2f}_l{regularizer:.2f}"
                for method, arm in (("C2-Gated", gated), ("C3", compact)):
                    metric = dict(all_metrics[arm])
                    metric.update({"method": method, "R": float(radius),
                                   "lambda": float(regularizer), "score_arm": arm,
                                   "auroc_difference_vs_c0": (
                                       float(metric["image_auroc"] - c0_auc)
                                       if metric["image_auroc"] is not None and c0_auc is not None
                                       else None),
                                   "cluster_bootstrap_auroc_95_percent_interval":
                                       bootstrap["auroc_95_percent_intervals"][arm],
                                   "cluster_bootstrap_auroc_difference_vs_c0_95_percent_interval":
                                       bootstrap["auroc_difference_vs_c0_95_percent_intervals"].get(arm),
                                   "cluster_bootstrap_auprc_95_percent_interval":
                                       bootstrap["auprc_95_percent_intervals"][arm]})
                    sensitivity.append(metric)

        c2_parity = json.loads((output_root / "stage_a_c2_parity.json").read_text(encoding="utf-8"))
        mechanism = json.loads((output_root / "stage_a_diagnostics.json").read_text(encoding="utf-8"))
        conditional = {}
        rank = {}
        for display, arm in primary.items():
            score_delta = scores_by_arm[arm] - scores_by_arm["c0"]
            conditional[display] = _score_direction(label_array, score_delta)
            rank[display] = _rank_reversals(scores_by_arm["c0"], scores_by_arm[arm], group_array)
        mechanism["outcome_conditional_score_direction"] = conditional
        mechanism["primary_rank_reversals_vs_c0"] = rank
        mechanism["c3_vs_c2_gated_primary_score_change"] = {
            "c3_minus_gated_auroc": primary_delta,
            "cluster_bootstrap_95_percent_interval": bootstrap[
                "primary_c3_minus_c2_gated_auroc_95_percent_interval"],
            "c3_all_local_anchor_mean_abs_score_delta": float(np.mean(np.abs(
                scores_by_arm["c3_r0.10_l0.10"] - scores_by_arm["c0"]))),
            "gated_top5_mean_abs_score_delta": float(np.mean(np.abs(
                scores_by_arm["c2gated_r0.10_l0.10"] - scores_by_arm["c0"]))),
        }
        labels_normal = int(np.count_nonzero(label_array == 0))
        labels_defect = int(np.count_nonzero(label_array == 1))
        prevalence = labels_defect / EVENT_COUNT
        unique_group_count = len(set(group_array.tolist()))
        if labels_normal + labels_defect != EVENT_COUNT or unique_group_count != int(
                manifest["expected_source_group_count"]):
            raise StageBError("Outcome count or unique Source Group count differs from R1 tape")

        if c3_primary is None or gated_primary is None or c0_auc is None:
            case = "UNDEFINED_METRICS"
        elif c3_primary > c0_auc and c3_primary > gated_primary:
            case = "CASE_A_POSITIVE_POINT_ESTIMATE"
        elif c3_primary < gated_primary:
            case = "CASE_C_C3_WEAKER_THAN_C2_GATED"
        elif c3_primary <= c0_auc:
            case = "CASE_B_LOCALITY_WITHOUT_CLEAR_C0_GAIN"
        else:
            case = "MIXED_PRIMARY_RESULT"
        primary_direction = conditional["C3"]
        if primary_direction["harmful_direction_events"] > primary_direction["correct_direction_events"]:
            case += ";CASE_D_HARMFUL_DIRECTION_MAJORITY"
        query_total = (c2_parity["queried_label_balance"]["normal"] +
                       c2_parity["queried_label_balance"]["defect"])
        query_coverage = query_total / EVENT_COUNT
        if query_coverage < 0.05:
            case += ";CASE_E_SPARSE_QUERY_COVERAGE"

        metric_output = {
            "schema": "calibration.r2_perf1.performance_metrics.v1",
            "status": "OUTCOME_EVALUATION_COMPLETE",
            "outcome_alignment": alignment,
            "event_count": EVENT_COUNT,
            "normal_count": labels_normal,
            "defect_count": labels_defect,
            "defect_prevalence": prevalence,
            "unique_source_group_count": unique_group_count,
            "queried_feedback_balance": c2_parity["queried_label_balance"],
            "queried_feedback_event_count": query_total,
            "queried_feedback_coverage": query_coverage,
            "frozen_threshold": theta,
            "threshold_recalibrated": False,
            "primary_methods": primary_metrics,
            "c3_minus_c2_gated_primary_auroc": primary_delta,
            "bootstrap": bootstrap,
            "run_interpretation_case": case,
            "stage_b_runtime": {"wall_seconds": time.monotonic() - start_wall,
                                "cpu_seconds": time.process_time() - start_cpu},
            "source_fingerprints": source_fingerprints(),
            "no_raw_label_rows_written": True,
            "no_masks_read_or_written": True,
        }
        sensitivity_output = {
            "schema": "calibration.r2_perf1.sensitivity_grid.v1",
            "configuration_sha256": manifest["configuration_sha256"],
            "parameter_selection_on_outcomes": False,
            "rows": sensitivity,
        }
        report = {
            "schema": "calibration.r2_perf1.sanitized_report.v1",
            "status": "COMPLETE_DEVELOPMENT_ONLY_HISTORICALLY_EXPOSED_SWITCH",
            "scientific_scope": "Fixed historical 512-event switch trajectory; not independent confirmation",
            "answer_to_primary_questions": {
                "C3_vs_C0_auroc_difference": primary_metrics["C3"]["auroc_difference_vs_c0"],
                "C3_vs_C2_Gated_auroc_difference": primary_delta,
                "C3_case_classification": case,
            },
            "primary_metrics": primary_metrics,
            "sensitivity_grid_rows": len(sensitivity),
            "queried_feedback_balance": c2_parity["queried_label_balance"],
            "offline_outcome_counts": {"normal": labels_normal, "defect": labels_defect,
                                       "prevalence": prevalence},
            "source_group_count": unique_group_count,
            "fixed_threshold": theta,
            "fixed_threshold_tuned_on_outcomes": False,
            "cluster_bootstrap": bootstrap,
            "causal_interpretation": (
                "Counterfactual CPU replay conditional on the common historical C0 Greedy "
                "trajectory and fixed query disclosures; not independently adapting closed-loop systems."),
            "C1_interpretation": "Adapted native R1 Codebook system-level reference; not a pure kernel ablation.",
            "C3_interpretation": (
                "Uniform-reliability compact evidence with beta=0.20*sign and kappa=1. "
                "A C3 top-5 arm is algebraically equivalent to C2-Gated under identical support."),
            "research_conclusion_limit": (
                "A negative result applies to this fixed engineering configuration and historical trajectory; "
                "it does not falsify the broader Local Correction research direction."),
            "source_fingerprints": source_fingerprints(),
        }
        budget.write_json("stage_b_metrics.json", metric_output)
        budget.write_json("stage_b_sensitivity.json", sensitivity_output)
        budget.write_json("stage_b_diagnostics.json", mechanism)
        budget.write_json("stage_b_report.json", report)
        return report
    finally:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.output_root, bootstrap_replicates=args.bootstrap_replicates)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(f"R2_PERF1_STAGE_B_FAILURE {type(exc).__name__}: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
