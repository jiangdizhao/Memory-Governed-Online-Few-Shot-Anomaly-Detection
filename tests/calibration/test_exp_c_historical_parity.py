from __future__ import annotations

import hashlib
import json

import pytest

from experiments.calibration.exp_c_historical_parity import verify_d2_metric_reproduction
from experiments.calibration.exp_c_memory_evidence import (
    ARM_IDS,
    CATEGORIES,
    ExperimentCError,
    UNCHANGED_D2_ARMS,
)


def _metrics():
    per_category = {}
    for category in CATEGORIES:
        per_category[category] = {}
        for arm in ARM_IDS:
            per_category[category][arm] = {
                "ber": 0.2, "ba": 0.8, "fpr": 0.1, "fnr": 0.3,
                "auroc": 0.7, "ap": 0.6, "worst_256_window_ber": 0.25,
                "query_rate": 0.03 if arm in UNCHANGED_D2_ARMS else 0.031,
                "query_count": 120,
            }
    macro = {
        arm: {metric: (0.03 if metric == "query_rate" else
                       120 if metric == "query_count_sum" else
                       0.2 if metric == "ber" else
                       0.8 if metric == "ba" else
                       0.1 if metric == "fpr" else
                       0.3 if metric == "fnr" else
                       0.7 if metric == "auroc" else
                       0.6 if metric == "ap" else 0.25)
              for metric in ("ber", "ba", "fpr", "fnr", "auroc", "ap",
                             "worst_256_window_ber", "query_rate", "query_count_sum")}
        for arm in ARM_IDS}
    return {"per_category": per_category, "category_equal_macro": macro}


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8")


def _d2_evaluation(evaluation):
    return {"category_metrics": evaluation["per_category"],
            "category_equal_macro": evaluation["category_equal_macro"]}


def test_d2_metric_reproduction_uses_only_sealed_aggregate_and_current_projection(tmp_path):
    evaluation = _metrics()
    stage_b = tmp_path / "stage_b"
    stage_b.mkdir()
    evaluation_path = stage_b / "D2_evaluation.json"
    d2_evaluation = _d2_evaluation(evaluation)
    _write_json(evaluation_path, d2_evaluation)
    digest = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    _write_json(stage_b / "D2_provenance.json", {
        "status": "COMPLETE", "run_id": "d2-reference",
        "stage_b_evaluation_sha256": digest,
    })
    result = verify_d2_metric_reproduction(
        d2_run_id="d2-reference", c_evaluation=evaluation,
        d2_evaluation=d2_evaluation, d2_stage_b_dir=stage_b)
    assert result["status"] == "PASS"
    assert result["historical_label_files_opened"] is False

    changed = _metrics()
    changed["per_category"]["audiojack"]["B"]["ber"] += 0.01
    changed_d2 = _d2_evaluation(changed)
    with pytest.raises(ExperimentCError, match="recomputed D2 metric"):
        verify_d2_metric_reproduction(
            d2_run_id="d2-reference", c_evaluation=evaluation,
            d2_evaluation=changed_d2, d2_stage_b_dir=stage_b)
