"""Validation of the registered, aggregate-only Stage B summary schema."""
from __future__ import annotations

import re
from typing import Any, Mapping
from .campaign import CATEGORIES, NEW, OLD
from .core import ARMS, AuditError


def validate_stage_b_summary(summary: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    if (not isinstance(summary, Mapping) or summary.get("schema") != "realiad.stage-b-final-metrics.v1" or
            summary.get("status") != "STAGE_B_ALL_90_CPU_EVALUATED" or
            not isinstance(summary.get("scientific_results"), list) or len(summary["scientific_results"]) != 90):
        raise AuditError("official Stage B summary schema, status or count is invalid")
    expected = [(c, p) for c in CATEGORIES for p in ("P0", "P1", "P2")]
    source_by_pair = {pair: (OLD if i < 36 else NEW) for i, pair in enumerate(expected)}
    records: dict[tuple[str, str], Mapping[str, Any]] = {}
    runs = {OLD: set(), NEW: set()}
    seal_hashes = set()
    for record in summary["scientific_results"]:
        if not isinstance(record, Mapping):
            raise AuditError("Stage B scientific result is malformed")
        pair = (record.get("category"), record.get("condition"))
        source = record.get("source_commit")
        seal = record.get("stage_a_category_seal_sha256")
        metrics = record.get("metrics")
        queries = record.get("query_count")
        composition = record.get("queried_label_composition")
        first_defect = record.get("first_queried_defect_position")
        run = record.get("campaign_run_id")
        if (pair not in source_by_pair or pair in records or source != source_by_pair[pair] or
                not isinstance(run, str) or not run or not isinstance(seal, str) or
                not re.fullmatch(r"[0-9a-f]{64}", seal) or seal in seal_hashes or
                not isinstance(metrics, Mapping) or not (set(ARMS) | {"N-shadow", "STATIC-4+64"}).issubset(metrics) or
                any(not isinstance(metrics[a], Mapping) for a in set(ARMS) | {"N-shadow", "STATIC-4+64"}) or
                not isinstance(queries, Mapping) or set(queries) != set(ARMS) or
                any(type(x) is not int or x < 0 for x in queries.values()) or
                not isinstance(composition, Mapping) or set(composition) != set(ARMS) or
                any(not isinstance(composition[a], Mapping) or set(composition[a]) != {"normal", "defect"} or
                    any(type(composition[a][k]) is not int for k in ("normal", "defect")) for a in ARMS) or
                not isinstance(first_defect, Mapping) or set(first_defect) != set(ARMS) or
                any(v is not None and type(v) is not int for v in first_defect.values())):
            raise AuditError("Stage B category, source, seal, metrics or query identity is invalid")
        records[pair] = record
        runs[source].add(run)
        seal_hashes.add(seal)
    if (set(records) != set(expected) or runs[OLD] != {"realiad-final-20261111-r1-40d61c3"} or
            len(runs[NEW]) != 1 or runs[OLD] == runs[NEW]):
        raise AuditError("Stage B summary does not cover the registered cross-campaign schedule")
    return records


def validate_stage_b_metadata(summary: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    if (summary.get("schema") != "realiad.stage-b-final-metrics.v1" or
            summary.get("status") != "STAGE_B_ALL_90_CPU_EVALUATED" or
            not isinstance(summary.get("scientific_results"), list) or len(summary["scientific_results"]) != 90):
        raise AuditError("Stage B metadata schema or count is invalid")
    expected = [(c, p) for c in CATEGORIES for p in ("P0", "P1", "P2")]
    sources = {pair: OLD if i < 36 else NEW for i, pair in enumerate(expected)}
    records = {}
    runs = {OLD: set(), NEW: set()}
    seals = set()
    for record in summary["scientific_results"]:
        pair = (record.get("category"), record.get("condition"))
        source, run = record.get("source_commit"), record.get("campaign_run_id")
        seal, annotation = record.get("stage_a_category_seal_sha256"), record.get("annotation_sha256")
        if (pair not in sources or pair in records or source != sources[pair] or
                not isinstance(run, str) or not run or not isinstance(seal, str) or
                not re.fullmatch(r"[0-9a-f]{64}", seal) or seal in seals or
                not isinstance(annotation, str) or not re.fullmatch(r"[0-9a-f]{64}", annotation)):
            raise AuditError("Stage B metadata identities are invalid")
        records[pair] = record
        runs[source].add(run)
        seals.add(seal)
    if runs[OLD] != {"realiad-final-20261111-r1-40d61c3"} or len(runs[NEW]) != 1 or runs[OLD] == runs[NEW]:
        raise AuditError("Stage B metadata campaign identities are invalid")
    return records


def read_stage_b_metadata(path: str) -> dict[str, Any]:
    """Decode identity metadata only; metric and outcome values remain opaque."""
    from pathlib import Path
    from experiments.calibration.final_benchmark import identity
    from experiments.calibration.final_benchmark.label_broker import _test_record_bounds
    from experiments.calibration import r7_shuffled_correction_4class as r7
    try:
        raw = Path(path).read_bytes()
        fields = identity._identity_object_fields(raw, 0, len(raw))
        keys = ("schema", "status", "stage_b_authorization_sha256")
        result = {k: r7._decode_field(raw, fields[k]) for k in keys}
        names = ("category", "condition", "campaign_run_id", "source_commit",
                 "stage_a_category_seal_sha256", "annotation_sha256")
        result["scientific_results"] = []
        for bounds in _test_record_bounds(raw, fields["scientific_results"]):
            entry = identity._identity_object_fields(raw, *bounds)
            result["scientific_results"].append({k: r7._decode_field(raw, entry[k]) for k in names})
        validate_stage_b_metadata(result)
        return result
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError("Stage B metadata cannot be read") from exc
