"""Metadata-only Real-IAD development protocol preflight.

This module reads only the four preregistered development-category JSON files.
It constructs the D2 historical support/C64 identities and the Stage 0 query
tapes from metadata, without opening images, masks, model weights, or confirmation
categories. Test labels are deliberately not passed to the query-tape builder.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .profiles import QueryProtocolConfig
from .protocol import TapeRecord
from .query_tape import build_qa_ts, build_qb, verify_tape


DEVELOPMENT_CLASSES = ("audiojack", "regulator", "switch", "rolled_strip_base")
HISTORY_SEED = 190021
SUPPORT_VIEW_SEED_OFFSET = 891


class RealIADPreflightError(RuntimeError):
    pass


def _opaque_id(domain: str, value: str) -> str:
    return hashlib.sha256(f"{domain}\0{value}".encode("utf-8")).hexdigest()


def _label_blind_tape_records(test_records: Sequence[Any]) -> list[TapeRecord]:
    """Build public tape fields without reading ImageRecord.is_good."""
    return [TapeRecord(
        position=index,
        official_sample_key=_opaque_id("official-sample-group", record.sample_key),
        camera_id=int(record.camera_id),
        relative_id=_opaque_id("image-identity", record.image_id),
        timestamp=str(record.timestamp),
        phase=0,
    ) for index, record in enumerate(test_records)]


def _select_support_and_c64(train_records: Sequence[Any]
                            ) -> tuple[list[Any], list[Any], list[str], list[str]]:
    """Reproduce historical D2 train-only draws without building Probe-A/B."""
    from experiments.risk_safe import preflight as historical

    config = historical.SplitConfig(support_seeds=(0,))
    train_groups = historical.group_by_sample(train_records)
    normal_groups = sorted(
        key for key, views in train_groups.items()
        if all(view.is_good for view in views)
    )
    if len(normal_groups) < config.n_history + config.n_support:
        raise RealIADPreflightError(
            f"Need {config.n_history + config.n_support} official train-normal "
            f"groups; found {len(normal_groups)}")

    c64_keys = sorted(random.Random(config.history_seed).sample(
        normal_groups, config.n_history))
    c64 = [random.Random(config.support_view_seed_offset + i).choice(
        train_groups[key]) for i, key in enumerate(c64_keys)]

    remaining = [key for key in normal_groups if key not in set(c64_keys)]
    support_keys = sorted(random.Random(0).sample(remaining, config.n_support))
    support_rng = random.Random(SUPPORT_VIEW_SEED_OFFSET)
    support = [support_rng.choice(train_groups[key]) for key in support_keys]
    return support, c64, support_keys, c64_keys


def _class_preflight(class_name: str, train: Sequence[Any], test: Sequence[Any],
                     class_json: Path,
                     query_config: QueryProtocolConfig) -> dict[str, Any]:
    from experiments.risk_safe import preflight as historical

    # This validator checks paths and official grouping identities only. Keep
    # test labels out of the query path and out of the returned audit record.
    split_audit = historical.validate_class_records(class_name, train, test)
    support, c64, support_keys, c64_keys = _select_support_and_c64(train)
    test_groups = historical.group_by_sample(test)
    group_view_counts = Counter(len(views) for views in test_groups.values())
    tape_records = _label_blind_tape_records(test)
    qa = build_qa_ts(tape_records, query_config)
    qb = build_qb(tape_records, query_config)
    verify_tape(qa, tape_records, query_config)
    verify_tape(qb, tape_records, query_config)

    support_ids = {record.image_id for record in support}
    c64_ids = {record.image_id for record in c64}
    test_ids = {record.image_id for record in test}
    train_sample_keys = set(historical.group_by_sample(train))
    test_sample_keys = set(test_groups)
    if (len(support) != 4 or len(c64) != 64 or len(support_ids) != 4 or
            len(c64_ids) != 64 or support_ids & c64_ids or
            set(support_keys) & set(c64_keys) or
            (set(support_keys) | set(c64_keys)) & test_sample_keys or
            support_ids & test_ids or c64_ids & test_ids):
        raise RealIADPreflightError(
            f"{class_name}: support/C64 identity or split separation failed")
    if len(qa.bin_edges) != 17 or sum(qa.bin_quotas) != qa.requested_budget:
        raise RealIADPreflightError(f"{class_name}: Q-A-TS bin allocation mismatch")
    if len(qa.queried_groups) != len(qa.events) or len(qb.queried_groups) != len(qb.events):
        raise RealIADPreflightError(f"{class_name}: query tape exceeds one view per group")
    if qb.early_budget != int(0.20 * qb.requested_budget + 0.5):
        raise RealIADPreflightError(f"{class_name}: Q-B warm-up rounding mismatch")

    qa_bin_counts = [0] * len(qa.bin_quotas)
    for event in qa.events:
        qa_bin_counts[event.time_bin] += 1
    qb_early_bin_counts = [0] * len(qb.bin_quotas)
    qb_later_bin_counts = [0] * len(qb.bin_quotas)
    for event in qb.events:
        target = (qb_early_bin_counts if event.position < 400
                  else qb_later_bin_counts)
        target[event.time_bin] += 1
    if tuple(qa_bin_counts) != qa.bin_quotas:
        raise RealIADPreflightError(f"{class_name}: Q-A-TS realized bin quota mismatch")
    if sum(qb_early_bin_counts) != qb.early_budget:
        raise RealIADPreflightError(f"{class_name}: Q-B early budget mismatch")

    return {
        "status": "PASS",
        "class": class_name,
        "class_json_sha256": hashlib.sha256(class_json.read_bytes()).hexdigest(),
        "metadata_only": True,
        "image_or_mask_opened": False,
        "support": {
            "count": len(support), "seed": 0,
            "identity_sha256": _opaque_id("support-identities", "\n".join(sorted(support_ids))),
            "official_group_sha256": _opaque_id("support-groups", "\n".join(support_keys)),
        },
        "c64": {
            "count": len(c64), "additional_to_support": True,
            "historical_d2_history_seed": HISTORY_SEED,
            "views_per_official_group": 1,
            "identity_sha256": _opaque_id("c64-identities", "\n".join(sorted(c64_ids))),
            "official_group_sha256": _opaque_id("c64-groups", "\n".join(c64_keys)),
            "protected_read_only": True,
        },
        "support_c64_disjoint": True,
        "train_test_official_group_disjoint": not bool(
            split_audit["official_sample_key"]["N_samples_in_both_splits"]),
        "train_test_official_groups_overlap": split_audit["official_sample_key"][
            "N_samples_in_both_splits"],
        "test_stream": {
            "images": len(test), "official_sample_groups": len(test_groups),
            "camera_view_fraction": len(test) / len(test_groups),
            "camera_views_per_group_distribution": {
                str(count): n for count, n in sorted(group_view_counts.items())},
            "one_view_per_query_group_enforced": True,
            "query_tape_label_fields": [],
            "stream_order": "Real-IAD category JSON test-record order",
        },
        "Q-A-TS": {
            "budget": qa.requested_budget, "budget_rule": "half_up(0.03*N_images)",
            "query_count": len(qa.events),
            "image_query_fraction": len(qa.events) / len(test),
            "official_sample_group_query_fraction": (
                len(qa.queried_groups) / len(test_groups)),
            "bins": 16, "bin_edges": list(qa.bin_edges),
            "bin_quotas": list(qa.bin_quotas),
            "realized_bin_counts": qa_bin_counts,
            "queried_groups": len(qa.queried_groups), "digest": qa.digest,
            "label_blind": True,
        },
        "Q-B": {
            "budget": qb.requested_budget, "early_budget": qb.early_budget,
            "early_window_images": 400, "early_fraction_of_budget": 0.20,
            "query_count": len(qb.events),
            "image_query_fraction": len(qb.events) / len(test),
            "official_sample_group_query_fraction": (
                len(qb.queried_groups) / len(test_groups)),
            "early_queries": sum(event.position < 400 for event in qb.events),
            "later_queries": sum(event.position >= 400 for event in qb.events),
            "combined_bin_quotas": list(qb.bin_quotas),
            "early_realized_bin_counts": qb_early_bin_counts,
            "later_realized_bin_counts": qb_later_bin_counts,
            "queried_groups": len(qb.queried_groups), "digest": qb.digest,
            "label_blind": True,
        },
        "query_priorities_used_test_label": False,
        "confirmation_classes_opened": False,
    }


def run_metadata_preflight(dataset_root: str | Path,
                           json_root: str | Path,
                           *, classes: Sequence[str] = DEVELOPMENT_CLASSES,
                           query_config: QueryProtocolConfig = QueryProtocolConfig()
                           ) -> dict[str, Any]:
    """Audit the four frozen development classes and no confirmation class."""
    requested = tuple(classes)
    if requested != DEVELOPMENT_CLASSES:
        raise RealIADPreflightError(
            f"Metadata preflight is locked to development classes {DEVELOPMENT_CLASSES}")
    root, json_dir = Path(dataset_root), Path(json_root)
    if not root.is_dir() or not json_dir.is_dir():
        return {"status": "SKIP", "reason": "Real-IAD dataset or JSON root is absent",
                "dataset_root_present": root.is_dir(),
                "json_root_present": json_dir.is_dir(),
                "classes": list(DEVELOPMENT_CLASSES)}

    from experiments.risk_safe import preflight as historical

    class_results = []
    for class_name in DEVELOPMENT_CLASSES:
        try:
            class_json = json_dir / f"{class_name}.json"
            if not class_json.is_file():
                raise RealIADPreflightError(f"Missing development metadata: {class_json}")
            train, test = historical.load_class_records(root, json_dir, class_name)
            class_results.append(_class_preflight(
                class_name, train, test, class_json, query_config))
        except Exception as exc:
            class_results.append({"class": class_name, "status": "BLOCKED",
                                  "reason": f"{type(exc).__name__}: {exc}"})
    status = ("BLOCKED" if any(item["status"] == "BLOCKED"
                               for item in class_results) else "PASS")
    return {
        "schema": "stage0.real_iad_metadata_preflight.v1",
        "status": status,
        "classes": class_results,
        "confirmation_classes_opened": False,
        "stage1_or_stage2_run": False,
        "source_identity": "historical experiments/risk_safe/preflight.py official_sample_key parser",
        "support_c64_selection": "historical D2 history seed 190021, support seed 0, one view per official group",
    }
