"""Stage A: label-blind causal replay of R1 queried C2 anchors.

There is intentionally no dataset JSON path or TEST-label parser in this
process. Its only label-bearing input is the next record from the two exact R1
queried-feedback ledgers, read after the current candidate score line is
flushed and sealed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


def _method_grid() -> list[dict[str, Any]]:
    from experiments.calibration.r2_perf1_common import (
        AMPLITUDE, K_NEIGHBORS, LAMBDAS, RADII)
    output: list[dict[str, Any]] = []
    for radius in RADII:
        for regularizer in LAMBDAS:
            output.extend((
                {"method": "C2-Gated", "radius": float(radius),
                 "lambda": float(regularizer), "k": K_NEIGHBORS,
                 "amplitude": AMPLITUDE},
                {"method": "C3", "radius": float(radius),
                 "lambda": float(regularizer), "k": None,
                 "amplitude": AMPLITUDE},
            ))
    return output


def _configuration_id(config: Mapping[str, Any]) -> str:
    if config["method"] == "C2-Gated":
        stem = "c2gated"
    elif config["method"] == "C3":
        stem = "c3"
    else:
        raise ValueError("Unknown correction method")
    return f"{stem}_r{float(config['radius']):.2f}_l{float(config['lambda']):.2f}"


def _normalize_rows_float32(rows: Any):
    import numpy as np
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Feature rows must be a finite matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.float32(1e-12))


def vectorized_cosine_distances(query_rows: Any, anchor_rows: Any):
    """R1 float32 L2 normalization and cosine-distance convention."""
    import numpy as np
    query = _normalize_rows_float32(query_rows)
    anchors = _normalize_rows_float32(anchor_rows)
    if query.shape[1] != anchors.shape[1]:
        raise ValueError("Query/anchor feature widths differ")
    similarity = np.clip(query @ anchors.T, -1.0, 1.0)
    return 1.0 - similarity


def compact_phi(normalized_distance: Any):
    import numpy as np
    values = np.asarray(normalized_distance, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Normalized distances must be finite and nonnegative")
    result = np.zeros_like(values)
    inside = values < 1.0
    u = values[inside]
    result[inside] = (1.0 - u * u) ** 2
    return result


def candidate_deltas(distances: Any, signs: Sequence[int], *,
                     anchor_order: Sequence[int], radius: float,
                     regularizer: float, k: int | None,
                     amplitude: float = 0.20):
    """Return exact local top-k or all-local compact signed-evidence deltas."""
    import numpy as np
    d = np.asarray(distances, dtype=np.float32)
    s = np.asarray(signs, dtype=np.int8)
    order = np.asarray(anchor_order, dtype=np.int64)
    if (d.ndim != 2 or d.shape[1] != len(s) or len(s) != len(order)
            or not np.isfinite(d).all() or not np.isfinite(s).all()
            or np.any((s != -1) & (s != 1))):
        raise ValueError("Malformed distance/sign matrix")
    if not math.isfinite(radius) or radius <= 0 or not math.isfinite(regularizer) or regularizer <= 0:
        raise ValueError("Radius and regularizer must be positive finite values")
    if k is not None and (isinstance(k, bool) or int(k) < 1):
        raise ValueError("k must be positive or None")
    output = np.zeros((d.shape[0],), dtype=np.float32)
    local_count = np.zeros((d.shape[0],), dtype=np.int32)
    local_pos = np.zeros((d.shape[0],), dtype=np.int32)
    local_neg = np.zeros((d.shape[0],), dtype=np.int32)
    effective_n = np.zeros((d.shape[0],), dtype=np.float64)
    nearest_local = np.full((d.shape[0],), np.nan, dtype=np.float64)
    for row in range(d.shape[0]):
        local = np.flatnonzero(d[row] < np.float32(radius))
        local_count[row] = len(local)
        if not len(local):
            continue
        local_signs = s[local]
        local_pos[row] = int(np.count_nonzero(local_signs > 0))
        local_neg[row] = int(np.count_nonzero(local_signs < 0))
        nearest_local[row] = float(np.min(d[row, local]))
        if k is None:
            selected = local
        else:
            selected = local[np.lexsort((order[local], d[row, local]))[:int(k)]]
        weights = compact_phi(d[row, selected].astype(np.float64) / radius)
        sum_selected = math.fsum(float(value) for value in weights)
        sum_selected_sq = math.fsum(float(value) * float(value) for value in weights)
        effective_n[row] = (
            sum_selected * sum_selected / sum_selected_sq if sum_selected_sq > 0 else 0.0)
        denominator = regularizer + math.fsum(float(value) for value in weights)
        signed_sum = math.fsum(float(w) * float(s[index])
                               for w, index in zip(weights, selected))
        output[row] = np.float32(float(amplitude) * signed_sum / denominator)
    return output, {
        "local_anchor_count_per_patch": local_count,
        "local_positive_anchor_count_per_patch": local_pos,
        "local_negative_anchor_count_per_patch": local_neg,
        "effective_local_neighbor_count_per_patch": effective_n,
        "nearest_local_distance_per_patch": nearest_local,
    }


def append_queried_c2_anchors(existing: Sequence[Mapping[str, Any]], *,
                              event_position: int, event_group: str,
                              top32_ids: Any, top32_features: Any,
                              sign: int, capacity: int = 256):
    """Apply the frozen 32-row queried-event append and FIFO retention rule."""
    import numpy as np
    from experiments.calibration.r1_matched_correction_pilot import arrays_sha256
    if (isinstance(sign, bool) or not isinstance(sign, int) or sign not in (-1, 1)
            or isinstance(event_position, bool) or not isinstance(event_position, int)
            or event_position < 0 or capacity < 32):
        raise ValueError("Malformed queried C2 update identity/sign/capacity")
    ids = np.asarray(top32_ids)
    features = np.asarray(top32_features)
    if (ids.shape != (32,) or ids.dtype != np.dtype(np.int64)
            or len(set(ids.tolist())) != 32 or np.any(ids < 0) or np.any(ids >= 784)
            or features.shape != (32, 4096) or features.dtype != np.dtype(np.float32)
            or not np.isfinite(features).all()):
        raise ValueError("Queried C2 update requires the native Top-32 IDs/features")
    output = list(existing)
    for patch_row, patch_id in enumerate(ids.tolist()):
        feature = np.ascontiguousarray(features[patch_row], dtype=np.float32)
        output.append({
            "position": event_position, "group_sha256": event_group,
            "patch_id": int(patch_id), "sign": int(sign),
            "feature": feature,
            "feature_sha256": arrays_sha256({"feature": feature}),
        })
    if len(output) > capacity:
        del output[:len(output) - capacity]
    return output


def seal_then_disclose(score_record: Mapping[str, Any], *, queried: bool,
                       seal_score, disclose):
    """Enforce score persistence before a query callback can observe its label."""
    seal_score(score_record)
    if not queried:
        return None
    return disclose()


def _raw_projected_score_rows(reader: Any, field_names: frozenset[str]) -> list[dict[str, Any]]:
    from experiments.calibration.r2_0_label_blind_mechanism_audit import (
        extract_json_object_fields)
    raw = reader.read_authorized_bytes("sealed_scores.jsonl")
    rows = [extract_json_object_fields(line, field_names)
            for line in raw.splitlines() if line.strip()]
    return rows


class _QueriedLedgerReader:
    """Raw readline only: never prefetch a future query disclosure."""

    def __init__(self, feedback_path: Path, correction_path: Path):
        self.feedback = feedback_path.open("rb", buffering=0)
        self.corrections = correction_path.open("rb", buffering=0)
        self.feedback_hash = hashlib.sha256()
        self.correction_hash = hashlib.sha256()
        self.count = 0

    @staticmethod
    def _read_one(stream, name: str) -> bytes:
        line = stream.readline()
        if not line or not line.endswith(b"\n"):
            raise ValueError(f"Queried {name} ledger ended or had an incomplete line")
        return line

    def reveal(self, *, position: int, expected_event_id: str,
               expected_image_identity: str, expected_group: str,
               expected_c0_score: float, theta: float) -> tuple[dict[str, Any], dict[str, Any], int]:
        feedback_raw = self._read_one(self.feedback, "private feedback")
        correction_raw = self._read_one(self.corrections, "correction event")
        feedback = json.loads(feedback_raw)
        correction = json.loads(correction_raw)
        label = feedback.get("label")
        if (isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1)):
            raise ValueError(f"Queried disclosure at {position} has invalid binary label")
        expected_sign = 1 if label == 1 else -1
        prediction = int(expected_c0_score >= theta)
        expected_outcome = ("TN" if label == 0 and prediction == 0 else
                            "FP" if label == 0 and prediction == 1 else
                            "FN" if label == 1 and prediction == 0 else "TP")
        feedback_digest = hashlib.sha256(feedback_raw).hexdigest()
        checks = (
            feedback.get("schema") == "calibration.r1_private_feedback.v1",
            feedback.get("position") == position,
            feedback.get("event_id") == expected_event_id,
            feedback.get("source_image_identity_sha256") == expected_image_identity,
            feedback.get("official_group_sha256") == expected_group,
            feedback.get("disclosure_after_score") is True,
            correction.get("position") == position,
            correction.get("event_id") == expected_event_id,
            correction.get("source_group_sha256") == expected_group,
            correction.get("feedback_ref") ==
                f"private_feedback.jsonl:{position}:{feedback_digest}",
            correction.get("outcome") == expected_outcome,
            correction.get("sign") == expected_sign,
            correction.get("c0_prediction") == prediction,
            correction.get("c0_score") == expected_c0_score,
            correction.get("c0_threshold") == theta,
            correction.get("c2_added_patch_rows") == 32,
            correction.get("c2_retention") == "FIFO, max 256 patch anchors",
        )
        if not all(checks):
            raise ValueError(f"Queried feedback/correction provenance mismatch at {position}")
        self.feedback_hash.update(feedback_raw)
        self.correction_hash.update(correction_raw)
        self.count += 1
        return feedback, correction, feedback_digest

    def finish(self, expected_count: int) -> dict[str, Any]:
        if self.count != expected_count:
            raise ValueError("Queried ledger count differs from the frozen Query Tape")
        if self.feedback.read(1) or self.corrections.read(1):
            raise ValueError("Queried ledger contains trailing/unqueried records")
        result = {
            "queried_records_consumed": self.count,
            "private_feedback_sha256": self.feedback_hash.hexdigest(),
            "correction_event_ledger_sha256": self.correction_hash.hexdigest(),
            "read_policy": "one raw line from each ledger, only at a frozen query position",
            "future_query_records_prefetched": False,
        }
        self.feedback.close()
        self.corrections.close()
        return result


def _distance_diagnostics(distances: Any, signs: Sequence[int], radius: float):
    import numpy as np
    d = np.asarray(distances, dtype=np.float32)
    s = np.asarray(signs, dtype=np.int8)
    eligible_nearest = (np.min(d, axis=1).astype(np.float64) if d.shape[1]
                        else np.full(d.shape[0], np.nan, dtype=np.float64))
    local = d < np.float32(radius)
    local_counts = local.sum(axis=1).astype(np.int64)
    positive_counts = ((local) & (s[None, :] > 0)).sum(axis=1).astype(np.int64)
    negative_counts = ((local) & (s[None, :] < 0)).sum(axis=1).astype(np.int64)
    local_weights = compact_phi(d.astype(np.float64) / radius)
    sum_w = local_weights.sum(axis=1)
    sum_w2 = np.square(local_weights).sum(axis=1)
    effective = np.divide(sum_w * sum_w, sum_w2, out=np.zeros_like(sum_w), where=sum_w2 > 0)
    nearest_local = np.full(d.shape[0], np.nan, dtype=np.float64)
    if d.shape[1]:
        for row in range(d.shape[0]):
            selected = d[row, local[row]]
            if selected.size:
                nearest_local[row] = float(selected.min())
    return eligible_nearest, local_counts, positive_counts, negative_counts, effective, nearest_local


def _empty_aggregate(configs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return { _configuration_id(config): {
        "event_count": 0, "events_with_eligible_anchors": 0,
        "events_with_any_local_anchor": 0, "events_with_no_local_anchor": 0,
        "events_with_sign_conflict": 0, "local_positive_anchor_observations": 0,
        "local_negative_anchor_observations": 0, "local_sign_conflict_patch_count": 0,
        "eligible_anchor_observations": 0, "same_source_excluded_anchor_observations": 0,
        "nonzero_corrected_patch_count": 0, "nonzero_corrected_score_event_count": 0,
        "positive_score_change_event_count": 0, "negative_score_change_event_count": 0,
        "changed_argmax_patch_count": 0, "far_field_nonzero_correction_count": 0,
        "absolute_patch_corrections": [], "absolute_score_changes": [],
        "nearest_eligible_distance": [], "nearest_local_distance": [],
        "effective_local_neighbor_count": [],
        "anchor_pool_counts": [], "same_source_excluded_counts": [],
    } for config in configs }


def _quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    import numpy as np
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not data.size:
        return {"count": 0, "min": None, "p50": None, "p90": None,
                "p95": None, "p99": None, "max": None}
    return {"count": int(data.size), "min": float(data.min()),
            "p50": float(np.quantile(data, 0.50)),
            "p90": float(np.quantile(data, 0.90)),
            "p95": float(np.quantile(data, 0.95)),
            "p99": float(np.quantile(data, 0.99)), "max": float(data.max())}


def _finalize_diagnostics(aggregate: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, raw in aggregate.items():
        row = {key: value for key, value in raw.items()
               if key not in {"absolute_patch_corrections", "absolute_score_changes",
                              "nearest_eligible_distance", "nearest_local_distance",
                              "effective_local_neighbor_count", "anchor_pool_counts",
                              "same_source_excluded_counts"}}
        row.update({
            "absolute_patch_correction_quantiles": _quantiles(raw["absolute_patch_corrections"]),
            "absolute_score_change_quantiles": _quantiles(raw["absolute_score_changes"]),
            "nearest_eligible_anchor_distance_quantiles": _quantiles(raw["nearest_eligible_distance"]),
            "nearest_local_anchor_distance_quantiles": _quantiles(raw["nearest_local_distance"]),
            "effective_local_neighbor_count_quantiles": _quantiles(raw["effective_local_neighbor_count"]),
            "eligible_anchor_pool_size_quantiles": _quantiles(raw["anchor_pool_counts"]),
            "same_source_exclusion_count_quantiles": _quantiles(raw["same_source_excluded_counts"]),
            "nonzero_corrected_patch_fraction": raw["nonzero_corrected_patch_count"] /
                (int(raw["event_count"]) * 32) if raw["event_count"] else None,
            "nonzero_corrected_score_fraction": raw["nonzero_corrected_score_event_count"] /
                int(raw["event_count"]) if raw["event_count"] else None,
            "far_field_nonzero_correction_count": int(raw["far_field_nonzero_correction_count"]),
        })
        output[name] = row
    return output


def run_stage_a(output_root: Path) -> dict[str, Any]:
    import numpy as np
    from experiments.calibration.r1_matched_correction_pilot import (
        C2_ANCHOR_CAPACITY, _c2_state_sha256, _validate_c2_anchor_state,
        arrays_sha256, signed_knn_delta)
    from experiments.calibration.r2_0_label_blind_mechanism_audit import (
        ReadOnlyR1Evidence, corrected_dense_map, max_score, verify_public_tape)
    from experiments.calibration.r2_perf1_common import (
        AMPLITUDE, ANCHOR_CAPACITY, CONFIGURATION_SCHEMA, EVIDENCE_AUTHORIZATION_ID,
        EVIDENCE_RELATIVE, EVIDENCE_RUN_ID, EVENT_COUNT, MAX_OUTPUT_BYTES,
        PATCH_COUNT, PRIMARY_LAMBDA, PRIMARY_RADIUS, QUERY_COUNT,
        OutputBudget, Perf1Error, ROOT, canonical_json, require_plain_path,
        require_worker, sha256_bytes, sha256_file, sha256_json,
        source_fingerprints, utc_now, verify_source_fingerprints)

    require_worker("stage_a")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    output_root = require_plain_path(Path(output_root), must_exist=True)
    budget = OutputBudget(output_root, allow_existing=True)
    try:
        manifest_path = output_root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "calibration.r2_perf1.run_manifest.v1" or manifest.get("status") != "INCOMPLETE":
            raise Perf1Error("Initial R2-PERF-1 manifest is missing or incompatible")
        verify_source_fingerprints(manifest["source_fingerprints"])

        evidence_root = require_plain_path(ROOT / EVIDENCE_RELATIVE, must_exist=True)
        reader = ReadOnlyR1Evidence(evidence_root)
        evidence_manifest, evidence_manifest_raw = reader.read_manifest()
        scope = evidence_manifest.get("authorization_scope", {})
        if (evidence_manifest.get("schema") != "calibration.r1_evidence_tape.v1" or
                evidence_manifest.get("status") != "ENGINEERING_PASS" or
                evidence_manifest.get("run_id") != EVIDENCE_RUN_ID or
                evidence_manifest.get("authorization_id") != EVIDENCE_AUTHORIZATION_ID or
                evidence_manifest.get("expected_source_commit") != "c86d5b89fa0062b87d80adb324eaa6bc3c94448e" or
                scope.get("category") != "switch" or scope.get("support_seed") != 0 or
                scope.get("max_events") != EVENT_COUNT or
                evidence_manifest.get("events_processed") != EVENT_COUNT or
                evidence_manifest.get("query_events") != QUERY_COUNT or
                evidence_manifest.get("query_positions_sha256") !=
                    "7f4aedf49f3e3bffd2482bd00d9e16512553b7cef8c0af7e002b47b60ddcb17f" or
                evidence_manifest.get("confirmation_data_accessed") is not False or
                evidence_manifest.get("masks_accessed") is not False or
                evidence_manifest.get("unqueried_labels_accessed") is not False or
                evidence_manifest.get("scientific_outcome_evaluation") is not False):
            raise Perf1Error("Historical R1 evidence manifest fails the authorized switch scope")
        fixed_digest = evidence_manifest.get("fixed_support_sha256")
        events, tape_receipt = verify_public_tape(
            reader, expected_count=EVENT_COUNT,
            expected_query_digest=evidence_manifest["query_positions_sha256"],
            expected_fixed_bank_digest=fixed_digest)
        transitions, transition_hash = reader.read_transition_structure()
        sealed_fields = frozenset({
            "position", "event_id", "source_image_identity_sha256",
            "official_group_sha256", "c0_score", "c1_score", "c2_score",
            "c0_prediction", "c2_state_sha256_before", "fixed_support_sha256"})
        sealed_rows = _raw_projected_score_rows(reader, sealed_fields)
        if len(sealed_rows) != EVENT_COUNT:
            raise Perf1Error("Public sealed identity projection has the wrong event count")
        for position, (row, event, transition) in enumerate(zip(sealed_rows, events, transitions)):
            if (row.get("position") != position or row.get("official_group_sha256") !=
                    event["source_group_sha256"] or row.get("fixed_support_sha256") != fixed_digest):
                raise Perf1Error(f"Public sealed identity projection differs at event {position}")
            for key in ("event_id", "source_image_identity_sha256"):
                if not isinstance(row.get(key), str) or len(row[key]) != 64:
                    raise Perf1Error(f"Malformed sealed {key} at event {position}")
            if (not isinstance(row.get("c2_state_sha256_before"), str) or
                    len(row["c2_state_sha256_before"]) != 64):
                raise Perf1Error(f"Malformed C2 state digest at event {position}")
            if bool(transition["query_decision"]) != (position in tape_receipt["query_positions"]):
                raise Perf1Error(f"Query decision projection mismatch at event {position}")

        theta = float(evidence_manifest["c64_theta0_higher_q90"])
        if theta != float(evidence_manifest["fixed_theta"]):
            raise Perf1Error("Frozen R1 threshold fields disagree")
        configs = _method_grid()
        grid_record = {
            "schema": CONFIGURATION_SCHEMA,
            "radii": [0.02, 0.05, 0.10, 0.20, 0.40],
            "regularizers": [0.10, 1.00],
            "amplitude": AMPLITUDE, "knn": 5, "capacity": ANCHOR_CAPACITY,
            "primary": {"radius": PRIMARY_RADIUS, "lambda": PRIMARY_LAMBDA},
            "c2_gated": "strict cosine distance < R, nearest k=5, compact phi, regularized signed mean",
            "c3": "all legal local anchors, beta=0.20*sign, kappa=1, compact phi, regularized evidence",
        }
        grid_sha256 = sha256_json(grid_record)
        if (manifest.get("configuration_sha256") != grid_sha256 or
                manifest.get("configuration") != grid_record):
            raise Perf1Error("Frozen Stage A parameter grid differs from launch manifest")
        if tape_receipt["unique_source_group_count"] != 103:
            raise Perf1Error(
                "Verified source-group count differs from the pinned R2-0 observation of 103")

        query_set = set(tape_receipt["query_positions"])
        feedback_path = evidence_root / "private_feedback.jsonl"
        correction_path = evidence_root / "correction_anchor_events.jsonl"
        require_plain_path(feedback_path, must_exist=True)
        require_plain_path(correction_path, must_exist=True)
        ledgers = _QueriedLedgerReader(feedback_path, correction_path)
        anchors: list[dict[str, Any]] = []
        aggregate = _empty_aggregate(configs)
        query_labels = Counter()
        c2_delta_max_abs = 0.0
        c2_score_max_abs = 0.0
        before_state_mismatches = 0
        after_state_mismatches = 0
        eligible_count_mismatches = 0
        rows_written = 0
        peak_rss = 0
        score_sha_pre = None

        for position, (event, row, transition) in enumerate(zip(events, sealed_rows, transitions)):
            arrays = event["arrays"]
            group = str(event["source_group_sha256"])
            if _c2_state_sha256(anchors) != row["c2_state_sha256_before"]:
                before_state_mismatches += 1
                raise Perf1Error(f"C2 reconstructed pre-score state digest mismatch at {position}")
            eligible_slots = _validate_c2_anchor_state(
                anchors, event_position=position, phase="pre_score", event_group=group)
            rows = np.asarray(arrays["top32_features"], dtype=np.float32)
            historical_delta, historical_eligible_count = signed_knn_delta(
                rows, anchors, event_position=position, event_group=group,
                neighbors=5, amplitude=AMPLITUDE)
            if len(eligible_slots) != historical_eligible_count:
                eligible_count_mismatches += 1
                raise Perf1Error(f"C2 eligible-anchor count mismatch at {position}")
            delta_abs = np.abs(historical_delta.astype(np.float64) -
                               arrays["c2_delta"].astype(np.float64))
            c2_delta_max_abs = max(c2_delta_max_abs, float(delta_abs.max(initial=0.0)))
            if c2_delta_max_abs > 1e-7:
                raise Perf1Error(f"Historical C2 patch-delta parity failed at event {position}")
            reconstructed_c2_score = max_score(corrected_dense_map(
                arrays["base_patch_map"], arrays["top32_ids"], historical_delta))
            score_error = abs(reconstructed_c2_score - float(row["c2_score"]))
            c2_score_max_abs = max(c2_score_max_abs, score_error)
            if score_error > 1e-7 or reconstructed_c2_score != float(event["scores"]["c2"]):
                raise Perf1Error(f"Historical C2 score parity failed at event {position}")

            if eligible_slots:
                anchor_features = np.stack([anchors[index]["feature"]
                                            for index in eligible_slots]).astype(np.float32)
                signs = np.asarray([anchors[index]["sign"] for index in eligible_slots], dtype=np.int8)
                distances = vectorized_cosine_distances(rows, anchor_features)
                order = np.asarray(eligible_slots, dtype=np.int64)
            else:
                signs = np.empty((0,), dtype=np.int8)
                order = np.empty((0,), dtype=np.int64)
                distances = np.empty((PATCH_COUNT, 0), dtype=np.float32)

            scores: dict[str, float] = {
                "c0": float(event["scores"]["c0"]),
                "c1": float(event["scores"]["c1"]),
                "c2": float(event["scores"]["c2"]),
            }
            argmax_patch: dict[str, int] = {
                "c0": int(np.argmax(arrays["base_patch_map"])),
                "c1": int(np.argmax(event["maps"]["c1"])),
                "c2": int(np.argmax(event["maps"]["c2"])),
            }
            current_diagnostics: dict[str, dict[str, Any]] = {}
            for config in configs:
                name = _configuration_id(config)
                radius = float(config["radius"])
                delta, local_stats = candidate_deltas(
                    distances, signs, anchor_order=order, radius=radius,
                    regularizer=float(config["lambda"]), k=config["k"],
                    amplitude=AMPLITUDE)
                corrected = corrected_dense_map(arrays["base_patch_map"],
                                                arrays["top32_ids"], delta)
                score = max_score(corrected)
                scores[name] = score
                argmax_patch[name] = int(np.argmax(corrected))
                nearest_all, local_counts, local_pos, local_neg, _, nearest_local = (
                    _distance_diagnostics(distances, signs, radius))
                effective_n = local_stats["effective_local_neighbor_count_per_patch"]
                if delta.shape != (PATCH_COUNT,) or not np.isfinite(delta).all():
                    raise Perf1Error(f"Invalid correction output for {name} at {position}")
                # Strict-support assertion: a nonzero patch must have at least one
                # eligible anchor inside R for that selected native feature.
                far_mask = local_counts == 0
                far_nonzero = int(np.count_nonzero((delta != 0.0) & far_mask))
                if far_nonzero:
                    raise Perf1Error(f"Far-field correction escaped compact support: {name}")
                diag = aggregate[name]
                diag["event_count"] += 1
                diag["events_with_eligible_anchors"] += int(bool(eligible_slots))
                diag["events_with_any_local_anchor"] += int(bool(np.any(local_counts)))
                diag["events_with_no_local_anchor"] += int(not bool(np.any(local_counts)))
                diag["events_with_sign_conflict"] += int(bool(np.any((local_pos > 0) & (local_neg > 0))))
                diag["local_positive_anchor_observations"] += int(local_pos.sum())
                diag["local_negative_anchor_observations"] += int(local_neg.sum())
                diag["local_sign_conflict_patch_count"] += int(np.count_nonzero((local_pos > 0) & (local_neg > 0)))
                diag["eligible_anchor_observations"] += int(len(eligible_slots))
                excluded = len(anchors) - len(eligible_slots)
                diag["same_source_excluded_anchor_observations"] += excluded
                diag["nonzero_corrected_patch_count"] += int(np.count_nonzero(delta))
                score_change = float(score - float(event["scores"]["c0"]))
                diag["nonzero_corrected_score_event_count"] += int(score_change != 0.0)
                diag["positive_score_change_event_count"] += int(score_change > 0.0)
                diag["negative_score_change_event_count"] += int(score_change < 0.0)
                diag["changed_argmax_patch_count"] += int(argmax_patch[name] != argmax_patch["c0"])
                diag["far_field_nonzero_correction_count"] += far_nonzero
                diag["absolute_patch_corrections"].extend(np.abs(delta.astype(np.float64)).tolist())
                diag["absolute_score_changes"].append(abs(score_change))
                diag["nearest_eligible_distance"].extend(nearest_all[np.isfinite(nearest_all)].tolist())
                diag["nearest_local_distance"].extend(nearest_local[np.isfinite(nearest_local)].tolist())
                diag["effective_local_neighbor_count"].extend(effective_n.tolist())
                diag["anchor_pool_counts"].append(len(eligible_slots))
                diag["same_source_excluded_counts"].append(excluded)
                current_diagnostics[name] = {
                    "local_anchor_count_total_over_32_patches": int(local_counts.sum()),
                    "local_positive_anchor_observations": int(local_pos.sum()),
                    "local_negative_anchor_observations": int(local_neg.sum()),
                    "sign_conflict_patch_count": int(np.count_nonzero((local_pos > 0) & (local_neg > 0))),
                    "effective_local_neighbor_count_mean": float(effective_n.mean()),
                    "nearest_eligible_distance_event_min": (
                        float(np.nanmin(nearest_all)) if np.isfinite(nearest_all).any() else None),
                    "nearest_local_distance_event_min": (
                        float(np.nanmin(nearest_local)) if np.isfinite(nearest_local).any() else None),
                    "nonzero_patch_correction_count": int(np.count_nonzero(delta)),
                    "far_field_nonzero_correction_count": far_nonzero,
                }

            score_record = {
                "schema": "calibration.r2_perf1.score_row.v1",
                "position": position,
                "event_id": row["event_id"],
                "source_image_identity_sha256": row["source_image_identity_sha256"],
                "source_group_sha256": group,
                "scores": scores,
                "argmax_patch": argmax_patch,
            }
            # This fsynced append is the event-level score seal. No queried label
            # is read until after this line is durable.
            def seal_event(record):
                budget.write_json("stage_a_scores.jsonl", record, append=True)

            query_disclosure = None
            if position in query_set:
                query_disclosure = seal_then_disclose(
                    score_record, queried=True, seal_score=seal_event,
                    disclose=lambda: ledgers.reveal(
                        position=position, expected_event_id=row["event_id"],
                        expected_image_identity=row["source_image_identity_sha256"],
                        expected_group=group, expected_c0_score=float(event["scores"]["c0"]),
                        theta=theta))
            else:
                seal_then_disclose(score_record, queried=False,
                                   seal_score=seal_event,
                                   disclose=lambda: (_ for _ in ()).throw(
                                       Perf1Error("Unqueried label disclosure attempted")))
            rows_written += 1

            if position in query_set:
                feedback, _, _ = query_disclosure
                label = int(feedback["label"])
                query_labels["defect" if label == 1 else "normal"] += 1
                sign = 1 if label == 1 else -1
                anchors = append_queried_c2_anchors(
                    anchors, event_position=position, event_group=group,
                    top32_ids=arrays["top32_ids"], top32_features=rows,
                    sign=sign, capacity=C2_ANCHOR_CAPACITY)
            if _c2_state_sha256(anchors) != transition["c2_state_sha256_after"]:
                after_state_mismatches += 1
                raise Perf1Error(f"C2 reconstructed post-update state digest mismatch at {position}")
            if len(anchors) > ANCHOR_CAPACITY:
                raise Perf1Error("Reconstructed C2 FIFO exceeded capacity 256")
            try:
                import psutil
                peak_rss = max(peak_rss, int(psutil.Process().memory_info().rss))
            except Exception:
                pass
            if (position + 1) % 64 == 0:
                print(f"R2_PERF1_STAGE_A events={position + 1}/{EVENT_COUNT} queries={ledgers.count}",
                      flush=True)

        ledger_receipt = ledgers.finish(QUERY_COUNT)
        if query_labels["normal"] + query_labels["defect"] != QUERY_COUNT:
            raise Perf1Error("Queried normal/defect count differs from the frozen Query Tape")
        if rows_written != EVENT_COUNT:
            raise Perf1Error("Stage A did not seal exactly 512 candidate-score rows")
        finalized_diagnostics = _finalize_diagnostics(aggregate)
        score_file_sha = sha256_file(output_root / "stage_a_scores.jsonl")
        parity = {
            "schema": "calibration.r2_perf1.c2_reconstruction.v1",
            "status": "PASS",
            "historical_c2_parameters": {
                "k": 5, "amplitude": 0.20, "capacity": C2_ANCHOR_CAPACITY,
                "retention": "FIFO", "queried_patch_anchors": 32,
                "label_sign": {"anomalous": 1, "normal": -1},
                "exclusion": "strictly past and source-group disjoint",
            },
            "events_reconstructed": EVENT_COUNT,
            "query_events": QUERY_COUNT,
            "queried_label_balance": {"normal": query_labels["normal"],
                                       "defect": query_labels["defect"]},
            "c2_delta_max_absolute_error": c2_delta_max_abs,
            "c2_score_max_absolute_error": c2_score_max_abs,
            "patch_delta_tolerance": 1e-7,
            "score_tolerance": 1e-7,
            "before_state_mismatches": before_state_mismatches,
            "after_state_mismatches": after_state_mismatches,
            "eligible_anchor_count_mismatches": eligible_count_mismatches,
            "final_retained_anchor_count": len(anchors),
            "final_c2_state_sha256": _c2_state_sha256(anchors),
            "ledger_read_receipt": ledger_receipt,
            "scores_flushed_before_each_query_reveal": True,
            "unqueried_labels_read": False,
        }
        budget.write_json("stage_a_c2_parity.json", parity)
        budget.write_json("stage_a_diagnostics.json", {
            "schema": "calibration.r2_perf1.mechanism_diagnostics.v1",
            "event_count": EVENT_COUNT,
            "method_configuration_sha256": grid_sha256,
            "methods": finalized_diagnostics,
            "diagnostic_definitions": {
                "local_anchor_observations": "source-disjoint anchors with cosine distance < R, counted per selected patch",
                "eligible_anchor_observations": "strictly past, source-disjoint anchor pool size counted per event",
                "effective_local_neighbor_count": "(sum compact weights)^2 / sum(compact weights^2), per selected patch",
                "nonzero": "exact float32 nonzero test",
                "far_field": "nonzero selected-patch correction with no eligible anchor inside R",
            },
        })
        sealed = {
            "schema": "calibration.r2_perf1.stage_a_seal.v1",
            "status": "SEALED",
            "event_count": rows_written,
            "score_table_sha256": score_file_sha,
            "r1_evidence_manifest_sha256": sha256_bytes(evidence_manifest_raw),
            "sealed_scores_file_sha256": tape_receipt["sealed_scores_file_sha256"],
            "transition_structural_projection_sha256": transition_hash,
            "fixed_support_sha256": fixed_digest,
            "unique_source_group_count": tape_receipt["unique_source_group_count"],
            "query_positions_sha256": tape_receipt["query_positions_sha256"],
            "method_configuration_sha256": grid_sha256,
            "source_fingerprints": source_fingerprints(),
            "outcome_labels_read": False,
            "test_metadata_opened": False,
            "stage_a_complete_utc": utc_now(),
        }
        budget.write_json("stage_a_sealed.json", sealed)
        result = {
            "status": "STAGE_A_SEALED",
            "event_count": rows_written,
            "score_table_sha256": score_file_sha,
            "c2_delta_max_absolute_error": c2_delta_max_abs,
            "c2_score_max_absolute_error": c2_score_max_abs,
            "queried_labels": dict(query_labels),
            "query_count": ledgers.count,
            "source_group_count": tape_receipt["unique_source_group_count"],
            "r1_tape_receipt": tape_receipt,
            "grid_sha256": grid_sha256,
            "wall_seconds": time.monotonic() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "peak_rss_bytes": peak_rss or None,
            "output_bytes": budget._audit(),
        }
        return result
    finally:
        # The parent orchestration process writes failure metadata if this child
        # exits abnormally or is terminated by the outer Job Object.
        pass


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_stage_a(args.output_root)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(f"R2_PERF1_STAGE_A_FAILURE {type(exc).__name__}: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
