"""Read-only, CPU-only independent verifier for sealed A4_EB68_v1 runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from experiments.calibration import a4_equal_budget_allocation as a4


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _json_sha(value: Any) -> str:
    return hashlib.sha256(a4._json_bytes(value)).hexdigest()


def _equal_number(left: Any, right: Any, tol: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= tol


def _same_preexisting_untracked_inventory(actual: Any, expected: Any) -> bool:
    """Compare the frozen workspace inventory, excluding campaign output counts.

    A4 run and report files are intentionally created after the preflight snapshot,
    so their counts are expected to grow during a successful campaign.
    """
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        return False
    identity_fields = (
        "baseline_count",
        "baseline_paths_sha256",
        "observed_preexisting_count",
        "observed_preexisting_paths_sha256",
        "matches_pre_task_inventory",
    )
    return (actual.get("matches_pre_task_inventory") is True and
            expected.get("matches_pre_task_inventory") is True and
            all(actual.get(key) == expected.get(key) for key in identity_fields))


def _frozen_model_scorer_identity_matches(
        model: Any, model_receipt: Any, gpu_name: Any) -> bool:
    if not isinstance(model, Mapping) or not isinstance(model_receipt, Mapping):
        return False
    runtime = model.get("runtime_profile", {})
    scorer = model.get("scorer_profile", {})
    if not isinstance(runtime, Mapping) or not isinstance(scorer, Mapping):
        return False
    expected_scorer = {"model_name": "dinov2_vitl14_reg", "image_size": 392,
        "resize_size": 448, "use_fp16": True, "view_names": ["id"],
        "knn": 5, "image_aggregation": "max", "feature_fuse": "concat",
        "fixed_support_count": 4}
    runtime_profile = runtime.get("runtime_profile", {})
    if not isinstance(runtime_profile, Mapping):
        return False
    return (
        model.get("loaded_model_state_sha256") == model_receipt.get("loaded_model_state_sha256") and
        runtime.get("checkpoint_sha256") == model_receipt.get("checkpoint_sha256") and
        runtime.get("hub_source_tree_sha256") == model_receipt.get("hub_source_tree_sha256") and
        runtime_profile.get("cuda_device_name") == gpu_name and
        runtime.get("layer_indices") == [5, 11, 17, 23] and
        runtime.get("image_size") == 392 and runtime.get("resize_size") == 448 and
        all(scorer.get(key) == value for key, value in expected_scorer.items()))


def _recompute_one(labels: np.ndarray, scores: np.ndarray,
                   predictions: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fpr = fp / (fp + tn) if fp + tn else None
    fnr = fn / (fn + tp) if fn + tp else None
    ber = (fpr + fnr) / 2 if fpr is not None and fnr is not None else None
    return {"auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "ber": ber, "fpr": fpr, "fnr": fnr,
        "balanced_accuracy": 1 - ber if ber is not None else None,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "score_sha256": hashlib.sha256(np.asarray(scores, dtype="<f8").tobytes()).hexdigest(),
        "prediction_sha256": hashlib.sha256(np.asarray(predictions, dtype=np.uint8).tobytes()).hexdigest(),
        "event_count": int(len(labels))}


def _verify_stage_a(run_root: Path, category: str,
                    manifest: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from experiments.calibration import stage05

    directory = run_root / "stage_a" / category
    seal_path = directory / "stage_a_seal.json"
    seal = _read(seal_path)
    if seal.get("status") != "SEALED_COMPLETE" or seal.get("category") != category or seal.get("condition") != "P0":
        raise a4.A4Error(f"Independent verification: invalid Stage A seal for {category}")
    if seal.get("unqueried_labels_accessed") is not False or seal.get("test_masks_opened") is not False:
        raise a4.A4Error(f"Independent verification: access boundary failed for {category}")
    if seal.get("all_event_scores_fsynced_before_query_feedback") is not True:
        raise a4.A4Error(f"Independent verification: score/query causal receipt failed for {category}")
    for file_name, sha_key in (("stage_a_scores.jsonl", "score_rows_sha256"),
                               ("private_feedback.jsonl", "feedback_rows_sha256"),
                               ("initialization_private.json", "initialization_sha256"),
                               ("stage_a_result.json", "result_sha256")):
        path = directory / file_name
        if not path.is_file() or _sha(path) != seal.get(sha_key):
            raise a4.A4Error(f"Independent verification: {category}/{file_name} checksum failed")
    seal["seal_sha256"] = _sha(seal_path)
    complete_path = directory / "category_complete.json"
    complete = _read(complete_path)
    if (complete.get("status") != "SEALED_COMPLETE" or
            complete.get("category") != category or
            complete.get("seal_sha256") != seal["seal_sha256"] or
            complete.get("result_sha256") != seal.get("result_sha256")):
        raise a4.A4Error(f"Independent verification: category completion pointer failed for {category}")
    rows = a4._load_jsonl(directory / "stage_a_scores.jsonl")
    if len(rows) != a4.EXPECTED_EVENTS[category] or seal.get("event_count") != len(rows):
        raise a4.A4Error(f"Independent verification: wrong event count for {category}")
    train, test, json_sha = stage05.read_class_stream(
        a4.DATASET_ROOT, a4.JSON_ROOT, category, test_labels=False)
    roles = a4._category_roles(category, train, test, json_sha)
    query = a4._query_identity_receipt(category, test)
    if (seal.get("category_json_sha256") != json_sha or
            seal.get("identity_sha256") != roles["identity_sha256"] or
            seal.get("a1_role_manifest_sha256") != roles["a1_role_manifest_sha256"] or
            seal.get("query_positions") != query["positions"] or
            seal.get("query_identity_set_sha256") != query["identity_set_sha256"] or
            seal.get("query_group_set_sha256") != query["group_set_sha256"] or
            seal.get("query_count") != query["query_count"]):
        raise a4.A4Error(f"Independent verification: category identities/query tape differ for {category}")
    manifest_receipt = manifest.get("preflight", {}).get("category_receipts", {}).get(category, {})
    if any(manifest_receipt.get("roles", {}).get(key) != roles.get(key)
           for key in ("identity_sha256", "support_identity_sha256", "c64_identity_sha256",
                       "c32_identity_sha256", "r32_identity_sha256", "a1_role_manifest_sha256",
                       "a1_stage_a_seal_sha256", "a1_support0_fixed_bank_sha256",
                       "category_json_sha256", "test_event_count", "test_group_count",
                       "no_test_overlap", "strict_four_shot_total")) or any(
            manifest_receipt.get("query", {}).get(key) != query.get(key)
            for key in ("query_count", "tape_sha256", "positions_sha256",
                        "identity_set_sha256", "group_set_sha256", "a2_stage_a_seal_sha256",
                        "group_count")):
        raise a4.A4Error(f"Independent verification: preflight identity receipt differs for {category}")
    expected_score_keys = {a4.SCORE_KEYS[arm][mode] for arm in a4.ARMS
                           for mode in ("static", "online")}
    for position, row in enumerate(rows):
        event = query["events"][position]
        if (row.get("position") != position or row.get("category") != category or
                row.get("condition") != "P0" or row.get("original_json_position") != event.original_position or
                row.get("event_id_sha256") != event.event_id_sha256 or
                row.get("image_identity_sha256") != event.image_identity_sha256 or
                row.get("source_group_sha256") != event.source_group_sha256 or
                set(row.get("scores", {})) != expected_score_keys or
                set(row.get("predictions", {})) != expected_score_keys or
                set(row.get("thresholds", {})) != expected_score_keys or
                "label" in row or "anomaly_class" in row or "mask" in row or
                row.get("test_labels_accessed") is not False):
            raise a4.A4Error(f"Independent verification: malformed label-blind row {category}:{position}")
        for key in expected_score_keys:
            score = float(row["scores"][key])
            threshold = float(row["thresholds"][key])
            if (not np.isfinite(score) or not np.isfinite(threshold) or
                    row["predictions"][key] not in (0, 1) or
                    row["predictions"][key] != int(score >= threshold)):
                raise a4.A4Error(f"Independent verification: invalid score/decision {category}:{position}:{key}")
        if row["scores"][a4.SCORE_KEYS["A"]["static"]] != row["scores"][a4.SCORE_KEYS["B"]["static"]] or row["scores"][a4.SCORE_KEYS["A"]["online"]] != row["scores"][a4.SCORE_KEYS["B"]["online"]]:
            raise a4.A4Error(f"Independent verification: A/B scores diverge for {category}:{position}")
        if (row["score_state_versions"][a4.SCORE_KEYS["A"]["online"]] !=
                row["score_state_versions"][a4.SCORE_KEYS["B"]["online"]]):
            raise a4.A4Error(f"Independent verification: A/B state versions diverge for {category}:{position}")
    feedback = a4._load_feedback(directory / "private_feedback.jsonl")
    expected_q = query["positions"]
    if [int(x["position"]) for x in feedback] != expected_q:
        raise a4.A4Error(f"Independent verification: feedback positions differ from frozen tape for {category}")
    for row in feedback:
        p = int(row["position"])
        event = query["events"][p]
        if (row["event_id_sha256"] != event.event_id_sha256 or
                row["image_identity_sha256"] != event.image_identity_sha256 or
                row["source_group_sha256"] != event.source_group_sha256 or
                row.get("score_row_fsynced_before_disclosure") is not True):
            raise a4.A4Error(f"Independent verification: feedback identity/order failed for {category}:{p}")
    if (seal.get("queried_label_count") != query["query_count"] or
            seal.get("queried_labels_only") != query["positions"]):
        raise a4.A4Error(f"Independent verification: queried-label disclosure receipt differs for {category}")
    result = _read(directory / "stage_a_result.json")
    if (result.get("status") != "SEALED_COMPLETE" or
            result.get("event_count") != a4.EXPECTED_EVENTS[category] or
            result.get("query_count") != query["query_count"] or
            result.get("executed_query_count") != query["query_count"] or
            result.get("query_identity_set_sha256") != query["identity_set_sha256"] or
            result.get("query_positions_sha256") != query["positions_sha256"] or
            result.get("score_rows_sha256") != seal.get("score_rows_sha256") or
            result.get("feedback_rows_sha256") != seal.get("feedback_rows_sha256") or
            result.get("verified_normal_feedback", 0) + result.get("verified_defect_feedback", 0) !=
                query["query_count"] or
            seal.get("feedback_counts", {}).get("queried_count") != query["query_count"] or
            seal.get("feedback_counts", {}).get("verified_normal_write_events") !=
                result.get("verified_normal_write_events")):
        raise a4.A4Error(f"Independent verification: Stage A result/query count differs for {category}")
    model = result.get("model", {})
    model_receipt = manifest.get("preflight", {}).get("model_receipt", {})
    gpu_name = manifest.get("preflight", {}).get("gpu_receipt", {}).get("name")
    if not _frozen_model_scorer_identity_matches(model, model_receipt, gpu_name):
        raise a4.A4Error(f"Independent verification: frozen model/scorer identity failed for {category}")
    banks = result["fixed_bank_receipts"]
    initialization = _read(directory / "initialization_private.json")
    expected_calibration_count = {"A": 64, "B": 32, "C": 32, "D": 32}
    if (initialization.get("calibration_counts") != expected_calibration_count or
            initialization.get("calibration_group_counts") != expected_calibration_count):
        raise a4.A4Error(f"Independent verification: C64/C32 calibration sizes differ for {category}")
    for arm in a4.ARMS:
        values = np.asarray(initialization["calibration_scores"][arm], dtype=np.float64)
        initial = initialization["initial_states"][arm]
        if (len(values) != expected_calibration_count[arm] or
                initial["theta"] != a4._higher_q90(values) or
                not _equal_number(initial["sigma"], max(float(values.std(ddof=1)), a4.SIGMA_FLOOR))):
            raise a4.A4Error(f"Independent verification: initial B3 calibration receipt differs for {category}/{arm}")
    writer_scores = initialization.get("writer_reference_scores", {})
    if (writer_scores.get("A_B_SHARED") != initialization["calibration_scores"]["B"] or
            len(writer_scores.get("A_B_SHARED", [])) != 32 or
            any(writer_scores.get(arm) != initialization["calibration_scores"][arm]
                for arm in ("C", "D"))):
        raise a4.A4Error(f"Independent verification: frozen C32 writer references differ for {category}")
    final_b3 = result.get("final_b3", {})
    if set(final_b3) != set(a4.ARMS):
        raise a4.A4Error(f"Independent verification: final B3 state set is incomplete for {category}")
    history_by_arm = {}
    for arm in a4.ARMS:
        history = final_b3[arm].get("history", [])
        if (not history or history[0]["version"] != 0 or
                any(abs(float(next_row["theta"]) - float(prev_row["theta"])) >
                    a4.B3_DELTA + 1e-12 for prev_row, next_row in zip(history, history[1:]))):
            raise a4.A4Error(f"Independent verification: B3 threshold movement/history failed for {category}/{arm}")
        history_by_arm[arm] = {int(item["version"]): float(item["theta"]) for item in history}
    for row in rows:
        for arm in a4.ARMS:
            online_key = a4.SCORE_KEYS[arm]["online"]
            version = int(row["score_state_versions"][online_key]["version"])
            if (version not in history_by_arm[arm] or
                    not _equal_number(row["thresholds"][online_key], history_by_arm[arm][version])):
                raise a4.A4Error(f"Independent verification: prequential threshold state mismatch {category}/{arm}")
            static_key = a4.SCORE_KEYS[arm]["static"]
            initial_theta = float(initialization["initial_states"][arm]["theta"])
            if not _equal_number(row["thresholds"][static_key], initial_theta):
                raise a4.A4Error(f"Independent verification: static M0 threshold moved {category}/{arm}")
    expected_fixed = roles["a1_support0_fixed_bank_sha256"]
    if banks["A"]["fixed_sha256"] != expected_fixed or banks["B"]["fixed_sha256"] != expected_fixed:
        raise a4.A4Error(f"Independent verification: A/B bank not equal to A1 Support-0 for {category}")
    if banks["A"]["fixed_sha256"] != banks["B"]["fixed_sha256"]:
        raise a4.A4Error(f"Independent verification: A/B fixed banks differ for {category}")
    for arm in a4.ARMS:
        profile = banks[arm]["profile"]
        if (int(profile["total_prototypes"]) != a4.FIXED_COUNT or
                int(profile["total_feature_bytes"]) != a4.FIXED_BYTES or
                len(profile["layers"]) != 1 or int(profile["layers"][0]["feature_dim"]) != a4.FEATURE_DIM or
                profile["layers"][0]["dtype"] != "float16"):
            raise a4.A4Error(f"Independent verification: fixed-bank byte/dimension gate failed {category}/{arm}")
        if (banks[arm].get("correction_enabled") is not False or
                banks[arm].get("defect_bank_enabled") is not False or
                "native Greedy K=1024" not in banks[arm].get("online_writer", "")):
            raise a4.A4Error(f"Independent verification: non-registered writer/correction profile {category}/{arm}")
    if len({banks[x]["fixed_sha256"] for x in ("A", "C", "D")}) != 3:
        raise a4.A4Error(f"Independent verification: A/C/D fixed-bank digests are not distinct for {category}")
    for arm in ("C", "D"):
        kc = banks[arm]["kcenter"]
        if (kc.get("repeat_identical") is not True or
                kc.get("selected_identity_sha256") != kc.get("repeat_selected_identity_sha256") or
                kc.get("prototype_count") != a4.FIXED_COUNT):
            raise a4.A4Error(f"Independent verification: {arm} K-Center reproducibility failed for {category}")
    expected_c_groups = sorted(x["group_sha256"] for x in roles["support_items"])
    expected_d_groups = sorted(x["group_sha256"] for x in
                               list(roles["support_items"]) + list(roles["r32_items"]))
    kc_c, kc_d = banks["C"]["kcenter"], banks["D"]["kcenter"]
    if (kc_c.get("candidate_pool_source_groups") != 4 or
            kc_c.get("constructor_image_count") != 4 or
            kc_c.get("constructor_group_identity_sha256") != _json_sha(expected_c_groups) or
            kc_c.get("calibration_groups_excluded") is not True or
            kc_d.get("candidate_pool_source_groups") != 36 or
            kc_d.get("constructor_image_count") != 36 or
            kc_d.get("constructor_group_identity_sha256") != _json_sha(expected_d_groups) or
            kc_d.get("candidate_count") != a4.KC_CAP):
        raise a4.A4Error(f"Independent verification: K-Center construction roles/candidate cap failed for {category}")
    if len(result.get("native_parity", {})) != 3 or any(
            item.get("status") != "PASS" or
            item.get("native_score_parity", {}).get("passed") is not True or
            item.get("native_update_parity", {}).get("passed") is not True
            for item in result["native_parity"].values()):
        raise a4.A4Error(f"Independent verification: native score/update parity failed for {category}")
    if result.get("feature_access_roles", {}).get("other_train_image_identities") != 0:
        raise a4.A4Error(f"Independent verification: extra TRAIN-normal image feature access for {category}")
    expected_roles = {"Support4_augmentation_images": 4,
                      "C64_identity_calibration_images": 64,
                      "D_R32_augmentation_images": 32,
                      "other_train_image_identities": 0}
    if result.get("feature_access_roles") != expected_roles:
        raise a4.A4Error(f"Independent verification: feature access roles differ for {category}")
    expected_calibration_bytes = 64 * 784 * 4096 * 4
    if result.get("unique_calibration_feature_bytes") != expected_calibration_bytes:
        raise a4.A4Error(f"Independent verification: retained calibration feature byte count differs for {category}")
    expected_memory = {"unique_online_fixed_bank_bytes": 3 * a4.FIXED_BYTES,
                       "static_M0_shadow_bank_bytes": 3 * a4.FIXED_BYTES,
                       "per_arm_deployed_fixed_reference_bytes": a4.FIXED_BYTES}
    if result.get("fixed_memory_components") != expected_memory:
        raise a4.A4Error(f"Independent verification: fixed-memory accounting differs for {category}")
    if (int(result.get("unique_calibration_image_bytes", 0)) <= 0 or
            int(result.get("support_augmentation_feature_bytes", 0)) <= 0 or
            int(result.get("identity_metadata_bytes", 0)) <= 0 or
            result.get("test_query_feature_bytes_per_event") != 784 * 4096 * 4):
        raise a4.A4Error(f"Independent verification: runtime feature/metadata accounting is incomplete for {category}")
    return seal, rows


def _verify_stage_b(run_root: Path, global_seal: Mapping[str, Any]) -> dict[str, Any]:
    seal_path = run_root / "stage_b_seal.json"
    seal = _read(seal_path)
    if (seal.get("status") != "SEALED_COMPLETE" or
            seal.get("run_id") != run_root.name or
            seal.get("labels_read_after_global_stage_a_seal") is not True or
            seal.get("masks_opened") is not False or
            seal.get("stage_a_global_seal_sha256") != _sha(run_root / "stage_a_global_seal.json") or
            seal.get("stage_a_category_seal_sha256") != global_seal.get("category_seal_sha256")):
        raise a4.A4Error("Independent verification: Stage B seal/global Stage A linkage failed")
    metrics_path = run_root / "stage_b_metrics_private.json"
    if not metrics_path.is_file() or _sha(metrics_path) != seal.get("metrics_sha256"):
        raise a4.A4Error("Independent verification: Stage B aggregate checksum failed")
    stored = _read(metrics_path)
    if (stored.get("run_id") != run_root.name or
            stored.get("all_stage_a_seals_validated_before_label_load") is not True or
            set(stored.get("categories", {})) != set(a4.CATEGORIES)):
        raise a4.A4Error("Independent verification: Stage B metrics/run linkage failed")
    rebuilt: dict[str, Any] = {}
    labels_by_category: dict[str, np.ndarray] = {}
    rows_by_category: dict[str, list[dict[str, Any]]] = {}
    for cat in a4.CATEGORIES:
        score_rows = a4._load_jsonl(run_root / "stage_a" / cat / "stage_a_scores.jsonl")
        projection_path = run_root / "stage_b" / cat / "label_projection_private.jsonl"
        if not projection_path.is_file() or _sha(projection_path) != seal.get("label_projection_sha256", {}).get(cat):
            raise a4.A4Error(f"Independent verification: label projection checksum failed for {cat}")
        projection = a4._load_jsonl(projection_path)
        if len(projection) != len(score_rows) or len(projection) != a4.EXPECTED_EVENTS[cat]:
            raise a4.A4Error(f"Independent verification: label projection length failed for {cat}")
        labels = []
        for pos, (event, label_row) in enumerate(zip(score_rows, projection)):
            if (label_row.get("position") != pos or
                    label_row.get("original_json_position") != event.get("original_json_position") or
                    label_row.get("event_id_sha256") != event.get("event_id_sha256") or
                    label_row.get("label") not in (0, 1) or
                    set(label_row) != {"position", "original_json_position", "event_id_sha256", "label"}):
                raise a4.A4Error(f"Independent verification: label projection mapping failed {cat}:{pos}")
            labels.append(int(label_row["label"]))
        y = np.asarray(labels, dtype=np.int8)
        labels_by_category[cat] = y
        rows_by_category[cat] = score_rows
        arms: dict[str, Any] = {}
        windows: dict[str, Any] = {}
        for arm in a4.ARMS:
            arms[arm] = {}
            windows[arm] = {}
            for mode in ("static", "online"):
                key = a4.SCORE_KEYS[arm][mode]
                scores = np.asarray([r["scores"][key] for r in score_rows], dtype=np.float64)
                preds = np.asarray([r["predictions"][key] for r in score_rows], dtype=np.int8)
                calced = _recompute_one(y, scores, preds)
                old = stored["categories"][cat]["arm_metrics"][arm][mode]
                for metric in ("auroc", "average_precision", "ber", "fpr", "fnr", "balanced_accuracy"):
                    if not _equal_number(calced.get(metric), old.get(metric)):
                        raise a4.A4Error(f"Independent verification: Stage B {metric} mismatch {cat}/{arm}/{mode}")
                if calced["confusion"] != old["confusion"] or calced["score_sha256"] != old["score_sha256"] or calced["prediction_sha256"] != old["prediction_sha256"]:
                    raise a4.A4Error(f"Independent verification: Stage B vector/confusion mismatch {cat}/{arm}/{mode}")
                arms[arm][mode] = calced
                window_rows = []
                for start in range(0, len(y) - 511, 512):
                    stop = start + 512
                    yw, pw = y[start:stop], preds[start:stop]
                    if len(np.unique(yw)) != 2:
                        item = {"start": start, "end_exclusive": stop,
                                "status": "INVALID_SINGLE_CLASS_WINDOW", "fpr": None,
                                "fnr": None, "ber": None}
                    else:
                        fpr = float(np.mean(pw[yw == 0] == 1))
                        fnr = float(np.mean(pw[yw == 1] == 0))
                        item = {"start": start, "end_exclusive": stop,
                                "status": "VALID", "fpr": fpr, "fnr": fnr,
                                "ber": (fpr + fnr) / 2}
                    window_rows.append(item)
                stored_windows = stored["categories"][cat]["windows"][arm][mode]
                if len(window_rows) != len(stored_windows):
                    raise a4.A4Error(f"Independent verification: window count mismatch {cat}/{arm}/{mode}")
                for expected, actual in zip(window_rows, stored_windows):
                    if expected["start"] != actual.get("start") or expected["status"] != actual.get("status") or any(
                            not _equal_number(expected.get(m), actual.get(m))
                            for m in ("fpr", "fnr", "ber")):
                        raise a4.A4Error(f"Independent verification: 512-event window mismatch {cat}/{arm}/{mode}")
                windows[arm][mode] = window_rows
        for mode in ("static", "online"):
            if not np.array_equal(
                    np.asarray([r["scores"][a4.SCORE_KEYS["A"][mode]] for r in score_rows]),
                    np.asarray([r["scores"][a4.SCORE_KEYS["B"][mode]] for r in score_rows])):
                raise a4.A4Error(f"Independent verification: A/B score vector identity failed {cat}/{mode}")
        rebuilt[cat] = {"arm_metrics": arms, "windows": windows}
    # Independent four-category equal-weight recomputation.
    macro = {}
    for arm in a4.ARMS:
        macro[arm] = {}
        for mode in ("static", "online"):
            macro[arm][mode] = {}
            for metric in ("auroc", "average_precision", "ber", "fpr", "fnr", "balanced_accuracy"):
                values = [rebuilt[c]["arm_metrics"][arm][mode][metric] for c in a4.CATEGORIES]
                macro[arm][mode][metric] = float(np.mean(values))
    for arm in a4.ARMS:
        for mode in ("static", "online"):
            for metric, value in macro[arm][mode].items():
                if not _equal_number(value, stored["category_equal_macro"][arm][mode][metric]):
                    raise a4.A4Error(f"Independent verification: macro metric mismatch {arm}/{mode}/{metric}")
    contrast_pairs = {
        "calibration_B_minus_A": ("B", "A"),
        "selector_C_minus_B": ("C", "B"),
        "reference_D_minus_C": ("D", "C"),
        "budget_D_minus_A": ("D", "A"),
    }
    metrics = ("auroc", "average_precision", "ber", "fpr", "fnr", "balanced_accuracy")
    rebuilt_contrasts: dict[str, dict[str, dict[str, float | None]]] = {}
    for cat in a4.CATEGORIES:
        rebuilt_contrasts[cat] = {}
        for name, (left, right) in contrast_pairs.items():
            rebuilt_contrasts[cat][name] = {}
            for metric in metrics:
                lhs = rebuilt[cat]["arm_metrics"][left]["online"].get(metric)
                rhs = rebuilt[cat]["arm_metrics"][right]["online"].get(metric)
                value = float(lhs) - float(rhs) if lhs is not None and rhs is not None else None
                rebuilt_contrasts[cat][name][metric] = value
                actual = stored["categories"][cat]["contrasts_online"][name].get(metric)
                if not _equal_number(value, actual):
                    raise a4.A4Error(f"Independent verification: category contrast mismatch {cat}/{name}/{metric}")
    for name in contrast_pairs:
        for metric in metrics:
            vals = [rebuilt_contrasts[cat][name][metric] for cat in a4.CATEGORIES]
            valid = [float(v) for v in vals if v is not None]
            value = float(np.mean(valid)) if valid else None
            actual = stored["category_equal_macro"]["contrasts_online"][name].get(metric)
            if not _equal_number(value, actual):
                raise a4.A4Error(f"Independent verification: macro contrast mismatch {name}/{metric}")
    # Query and normal-memory counts are recomputed from sealed feedback, not inferred from key suffixes.
    for cat in a4.CATEGORIES:
        feedback = a4._load_feedback(run_root / "stage_a" / cat / "private_feedback.jsonl")
        if len(feedback) != stored["categories"][cat]["query_count"]:
            raise a4.A4Error(f"Independent verification: query count mismatch {cat}")
        n0 = sum(x["label"] == 0 for x in feedback)
        n1 = sum(x["label"] == 1 for x in feedback)
        target = stored["categories"][cat]
        if (n0 != target["queried_normal"] or n1 != target["queried_defect"] or
                target["query_rate"] != len(feedback) / a4.EXPECTED_EVENTS[cat]):
            raise a4.A4Error(f"Independent verification: query rate/labels mismatch {cat}")
        native_writes = 0
        candidates = accepted = state_changes = 0
        for row in feedback:
            for action in row.get("writer_actions", {}).values():
                detail = action.get("action", {})
                native_writes += int(detail.get("action") == "native_greedy_normal_update")
                candidates += int(action.get("candidate_patch_count", 0))
                accepted += int(action.get("accepted_candidate_patches", 0))
                state_changes += int(action.get("state_changed", False))
        if (native_writes != target.get("native_normal_write_actions") or
                candidates != target.get("candidate_patches") or
                accepted != target.get("accepted_candidate_patches") or
                state_changes != target.get("reported_state_change_actions") or
                3 * n0 != target.get("writer_calls")):
            raise a4.A4Error(f"Independent verification: writer count mismatch {cat}")
    return {"status": "PASS", "category_count": len(a4.CATEGORIES),
            "stage_b_seal_sha256": _sha(seal_path),
            "metrics_sha256": _sha(metrics_path),
            "label_projection_sha256": seal["label_projection_sha256"],
            "metrics_recomputed_cpu_only": True,
            "all_static_online_arm_keys_explicit": True}


def verify_run(run_id: str) -> dict[str, Any]:
    run_id = a4._validate_run_id(run_id)
    run_root = a4.RUN_ROOT / run_id
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    manifest = _read(run_root / "run_manifest.json")
    if manifest.get("run_id") != run_id or manifest.get("protocol_id") != a4.PROTOCOL_ID:
        raise a4.A4Error("Independent verification: run manifest identity mismatch")
    if manifest.get("source_hash_normalization") != "CRLF normalized to LF before SHA-256":
        raise a4.A4Error("Independent verification: source hash normalization is unspecified")
    if (_json_sha(manifest.get("preflight")) != manifest.get("preflight_sha256") or
            _json_sha(manifest.get("experiment_contract")) != manifest.get("experiment_contract_sha256") or
            manifest.get("preflight", {}).get("source_hashes") != manifest.get("source_hashes") or
            manifest.get("git", {}).get("branch") != a4.BRANCH or
            manifest.get("git", {}).get("expected_parent") != a4.EXPECTED_PARENT or
            manifest.get("protected_historical_output_trees") !=
                manifest.get("preflight", {}).get("protected_historical_output_trees")):
        raise a4.A4Error("Independent verification: run manifest preflight/contract checksum failed")
    commit = str(manifest.get("git", {}).get("source_freeze_commit", ""))
    if not commit:
        raise a4.A4Error("Independent verification: source-freeze commit missing")
    parent_to_freeze = subprocess.run(
        ["git", "merge-base", "--is-ancestor", a4.EXPECTED_PARENT, commit],
        cwd=a4.ROOT, capture_output=True)
    if (parent_to_freeze.returncode != 0 or
            manifest.get("git", {}).get("expected_parent") != a4.EXPECTED_PARENT):
        raise a4.A4Error("Independent verification: source-freeze ancestry does not bind the registered parent")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                              cwd=a4.ROOT, capture_output=True)
    if ancestor.returncode != 0:
        raise a4.A4Error("Independent verification: source-freeze commit is not an ancestor of current HEAD")
    if a4._git("branch", "--show-current") != a4.BRANCH:
        raise a4.A4Error("Independent verification: branch changed")
    committed_paths = a4._git_paths(("diff", "--name-only",
                                     f"{a4.EXPECTED_PARENT}..HEAD"))
    allowed_committed = set(a4.ALLOWED_A4_SOURCE_PATHS)
    allowed_committed.update(p for p in committed_paths if p.startswith(
        "docs/calibration/audit/a4_equal_budget_20261107/"))
    unexpected_committed = [p for p in committed_paths if p not in allowed_committed]
    if unexpected_committed:
        raise a4.A4Error(f"Independent verification: unexpected committed paths since parent: {unexpected_committed}")
    observed = a4._source_hashes()
    if observed != manifest.get("source_hashes"):
        raise a4.A4Error("Independent verification: execution-time source hashes differ")
    for relative, expected in manifest["source_hashes"].items():
        blob = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=a4.ROOT,
                              check=True, capture_output=True).stdout
        if a4._sha_source_bytes(blob) != expected:
            raise a4.A4Error(f"Independent verification: committed source hash differs for {relative}")
    protected_now = [a4._tree_receipt(path) for path in a4.PROTECTED_OUTPUT_PATHS]
    if protected_now != manifest.get("protected_historical_output_trees"):
        raise a4.A4Error("Independent verification: A1/A2/A3/R9/R10 protected output changed")
    audit = a4.protected_file_audit(allow_report_files=True)
    if not _same_preexisting_untracked_inventory(
            audit.get("preexisting_untracked_inventory"), manifest.get("untracked_baseline")):
        raise a4.A4Error("Independent verification: pre-task untracked pathname inventory changed")
    category_seals = {}
    for cat in a4.CATEGORIES:
        seal, _rows = _verify_stage_a(run_root, cat, manifest)
        category_seals[cat] = seal["seal_sha256"]
    global_path = run_root / "stage_a_global_seal.json"
    global_seal = _read(global_path)
    if (global_seal.get("status") != "SEALED_COMPLETE" or
            global_seal.get("run_id") != run_id or
            global_seal.get("all_categories_complete_before_stage_b") is not True or
            global_seal.get("category_seal_sha256") != category_seals):
        raise a4.A4Error("Independent verification: global Stage A seal does not bind all category seals")
    stage_b = _verify_stage_b(run_root, global_seal)
    complete = _read(run_root / "complete_seal.json") if (run_root / "complete_seal.json").is_file() else None
    if complete is not None and (complete.get("status") != "COMPLETE" or
            complete.get("run_id") != run_id or
            complete.get("stage_a_global_seal_sha256") != _sha(global_path) or
            complete.get("stage_b_seal_sha256") != _sha(run_root / "stage_b_seal.json")):
        raise a4.A4Error("Independent verification: complete seal is invalid")
    return {"status": "PASS", "run_id": run_id,
            "source_freeze_commit": commit,
            "source_hashes_sha256": manifest["execution_source_hashes_sha256"],
            "stage_a_global_seal_sha256": _sha(global_path),
            "category_stage_a_seal_sha256": category_seals,
            "stage_b": stage_b, "protected_file_audit": audit["status"],
            "protected_historical_output_trees": protected_now,
            "complete_seal_present": complete is not None,
            "test_masks_opened": False, "confirmation_outcomes_accessed": False}


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Independent A4 CPU-only verifier")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    result = verify_run(args.run_id)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
