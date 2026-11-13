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
        run = record.get("campaign_run_id")
        if (pair not in source_by_pair or pair in records or source != source_by_pair[pair] or
                not isinstance(run, str) or not run or not isinstance(seal, str) or
                not re.fullmatch(r"[0-9a-f]{64}", seal) or seal in seal_hashes or
                not isinstance(metrics, Mapping) or not (set(ARMS) | {"N-shadow", "STATIC-4+64"}).issubset(metrics) or
                any(not isinstance(metrics[a], Mapping) for a in set(ARMS) | {"N-shadow", "STATIC-4+64"}) or
                not isinstance(queries, Mapping) or set(queries) != set(ARMS) or
                any(type(x) is not int or x < 0 for x in queries.values())):
            raise AuditError("Stage B category, source, seal, metrics or query identity is invalid")
        records[pair] = record
        runs[source].add(run)
        seal_hashes.add(seal)
    if (set(records) != set(expected) or runs[OLD] != {"realiad-final-20261111-r1-40d61c3"} or
            len(runs[NEW]) != 1 or runs[OLD] == runs[NEW]):
        raise AuditError("Stage B summary does not cover the registered cross-campaign schedule")
    return records
