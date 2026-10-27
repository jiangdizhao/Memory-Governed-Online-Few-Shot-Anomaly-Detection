#!/usr/bin/env python3
"""D1-B protocol freeze: metadata-only, deterministic, paired query schedules.

No feature extraction, GPU allocation, online threshold adjustment or algorithm
performance. Offline ground truth is used ONLY to manufacture explicitly marked
synthetic non-IID stress orders and to audit query-label composition after tape
selection. Do not pass trusted sample metadata to a model/query controller.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from . import preflight as pf
from . import query_protocol as qp

SCHEMA = "risk_safe.d1b.protocol.v1"
PRIMARY_ORDER = "random"
STRESS_ORDERS = ("normal_first", "defect_first", "bursty")
DIAGNOSTIC_ORDER = "acquisition_time_blocked"
ORDERS = (PRIMARY_ORDER, *STRESS_ORDERS, DIAGNOSTIC_ORDER)
MAIN_BUDGETS = (16, 32, 64, 79)
SENSITIVITY_BUDGETS = (96, 128)
LOW_BUDGET_DIAGNOSTIC = (8,)
QUERY_SEEDS = (37, 101, 211)
SUPPORT_SEEDS = (0, 1, 2)
ORDER_SEED = 1701
BURST_RUN = 16
DEV_CLASSES = pf.CLASSES
INITIAL_SMOKE_CLASSES = ("audiojack", "bottle_cap")
INITIAL_SMOKE_SUPPORT_SEEDS = (0,)
INITIAL_SMOKE_QUERY_SEEDS = (37, 101)
DEFAULT_OUT_DIR = "outputs/risk_safe_sprint_20261027/d1b_protocol_v1"
OUTPUT_PROTOCOL = "protocol_v1.json"
OUTPUT_QUERIES = "query_policy_manifest.json"
OUTPUT_ORDERS = "stream_order_audit.json"
OUTPUT_SOURCE = "source_manifest.json"
ALL_OUTPUTS = (OUTPUT_PROTOCOL, OUTPUT_QUERIES, OUTPUT_ORDERS, OUTPUT_SOURCE)
SOURCE_FILES = (
    "experiments/risk_safe/__init__.py",
    "experiments/risk_safe/preflight.py",
    "experiments/risk_safe/query_protocol.py",
    "experiments/risk_safe/d1b_protocol.py",
    "experiments/final_decision_core.py",
    "experiments/v02_architecture_core.py",
    "vmb_visionad_new.py",
    "test_phase3_AIF_simplified_v22.py",
    "tests/test_risk_safe_d1a.py",
    "tests/test_risk_safe_d1b.py",
    "docs/risk_safe/D1_B_FROZEN_PROTOCOL.md",
)


class D1BProtocolError(pf.DatasetProtocolError):
    """A registered D1-B experimental protocol invariant was violated."""


def phase_groups(phase_records: Sequence[Any]) -> Dict[str, Tuple[Any, ...]]:
    grouped = pf.group_by_sample(phase_records)
    for key, records in grouped.items():
        cams = sorted(int(r.camera_id) for r in records)
        if cams != [1, 2, 3, 4, 5]:
            raise D1BProtocolError(
                f"Official sample {key} does not have exactly C1..C5: {cams}")
    return {k: tuple(sorted(v, key=lambda r: int(r.camera_id)))
            for k, v in grouped.items()}


def scenario_order(grouped: Mapping[str, Tuple[Any, ...]], mode: str,
                   class_name: str, phase: int, order_seed: int = ORDER_SEED,
                   burst_run: int = BURST_RUN) -> Tuple[str, ...]:
    """Return a permutation of one phase's official-sample keys.

    Mode random reads only IDs, never image labels. Stress modes intentionally
    use offline labels to synthesize non-IID streams. They are explicitly not
    prospective online controllers and do not grant labels at prediction time.
    """
    if mode not in ORDERS or mode == DIAGNOSTIC_ORDER:
        raise D1BProtocolError(f"Unsupported synthetic mode: {mode}")
    keys = sorted(grouped)
    rng = random.Random(qp.derived_seed(
        order_seed, "D1-B", mode, class_name, phase))
    rng.shuffle(keys)
    if mode == PRIMARY_ORDER:
        return tuple(keys)
    if burst_run < 1:
        raise D1BProtocolError("burst_run must be positive")
    normal, defective = [], []
    for key in keys:
        # This branch is exclusively for offline-constructed synthetic stress.
        (defective if any(not r.is_good for r in grouped[key]) else normal).append(key)
    if mode == "normal_first":
        return tuple(normal + defective)
    if mode == "defect_first":
        return tuple(defective + normal)
    # Fixed alternating runs, with no oracle information shown to the learner.
    order = []
    n, d, normal_turn = 0, 0, True
    while n < len(normal) or d < len(defective):
        if normal_turn and n < len(normal):
            take = normal[n:n + burst_run]
            order.extend(take)
            n += len(take)
        elif not normal_turn and d < len(defective):
            take = defective[d:d + burst_run]
            order.extend(take)
            d += len(take)
        normal_turn = not normal_turn
    if len(order) != len(keys) or set(order) != set(keys):
        raise D1BProtocolError("Bursty order dropped or duplicated samples")
    return tuple(order)


def build_ordered_stream(phases: Sequence[Sequence[Any]], mode: str,
                         class_name: str, order_seed: int = ORDER_SEED
                         ) -> Tuple[qp.StreamImage, ...]:
    if mode not in ORDERS:
        raise D1BProtocolError(f"Unknown order: {mode}")
    if mode == DIAGNOSTIC_ORDER:
        # The D1-A stream uses phase-blocked timestamp order. It is not a
        # globally chronological real deployment stream.
        return qp.build_stream(phases)
    result = []
    for phase_index, records in enumerate(phases, 1):
        groups = phase_groups(records)
        ids = scenario_order(groups, mode, class_name, phase_index, order_seed)
        for sample_key in ids:
            for record in groups[sample_key]:
                result.append(qp.StreamImage(
                    position=len(result) + 1, sample_key=sample_key,
                    path=record.path, rel_path=record.rel_path,
                    camera_id=int(record.camera_id),
                    timestamp=str(record.timestamp), phase=phase_index,
                    stream_order=mode))
    stream = tuple(result)
    qp.validate_stream(stream)
    return stream


def shared_query_priority(sample_keys: Sequence[str], class_name: str,
                          query_seed: int) -> Tuple[str, ...]:
    """One label-blind permutation per class/seed, reused across orders.

    The trusted official IDs contain condition tokens but the shuffle never
    parses or stratifies by their meaning. All controllers receive opaque
    observation handles. The sample universe is frozen independently of order.
    """
    ids = sorted(set(sample_keys))
    if len(ids) != len(sample_keys):
        raise D1BProtocolError("Duplicated official sample key universe")
    rng = random.Random(qp.derived_seed(
        query_seed, "D1-B", "shared-Q-A", class_name, qp.sha256_of(ids)))
    rng.shuffle(ids)
    return tuple(ids)


def build_paired_tapes(stream: Sequence[qp.StreamImage], budgets: Sequence[int],
                       query_seed: int, class_name: str,
                       priority: Sequence[str]) -> Dict[int, qp.QueryTape]:
    choices = qp.camera_choice_map(stream, query_seed, class_name)
    tapes = {
        b: qp.build_query_tape(stream, int(b), query_seed, class_name,
                               priority=priority, choices=choices)
        for b in sorted(set(budgets))
    }
    for smaller, larger in zip(sorted(tapes), sorted(tapes)[1:]):
        qp.assert_nested(tapes[smaller], tapes[larger])
    return tapes


def audit_order(stream: Sequence[qp.StreamImage],
                label_by_path: Mapping[str, bool]) -> Dict[str, Any]:
    """Offline-only class-sequence diagnostic; never fed to a controller."""
    phases = {}
    for item in stream:
        slot = phases.setdefault(str(item.phase),
                                 {"sample_order": [], "normal": 0, "defect": 0})
        if item.sample_key not in slot["sample_order"]:
            slot["sample_order"].append(item.sample_key)
    for phase, slot in phases.items():
        labels = []
        for sample in slot["sample_order"]:
            views = [i for i in stream if i.phase == int(phase)
                     and i.sample_key == sample]
            defective = any(not label_by_path[v.path] for v in views)
            labels.append(int(defective))
        slot["normal"] = labels.count(0)
        slot["defect"] = labels.count(1)
        slot["transitions"] = sum(x != y for x, y in zip(labels, labels[1:]))
        slot["initial_normal_prefix"] = (
            labels.index(1) if 1 in labels else len(labels))
        del slot["sample_order"]
    return {"phases": phases,
            "total_sample_transitions": sum(v["transitions"] for v in phases.values()),
            "diagnostic_only": True,
            "label_information_exposed_to_controller": False}


def protocol_block() -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "stage": "D1-B",
        "status": "frozen_before_new_admission_performance_results",
        "authority": "docs/RISK_SAFE_ADMISSION_7DAY_PLAN_2026-10-27.md",
        "identity_version": pf.IDENTITY_VERSION,
        "primary_sample_unit": pf.PRIMARY_GROUPING_UNIT,
        "label_cost_unit": "one queried camera-image label per official sample",
        "classes_development": list(DEV_CLASSES),
        "classes_confirmatory": "remaining Real-IAD categories and MVTec-AD, no tuning",
        "static_support_shots": 4,
        "support_seeds": list(SUPPORT_SEEDS),
        "mature_history": {"enabled_in_primary": False,
                           "auxiliary_verified_normal_samples": 64},
        "probe": {"A": {"normal_sample_groups": 24, "defect_sample_groups": 24},
                  "B": {"normal_sample_groups": 24, "defect_sample_groups": 24},
                  "evaluation_labels_forbidden_to_all_online_controllers": True},
        "stream": {"phases": 3, "samples_per_phase": 175,
                   "images_per_sample": 5, "total_samples": 525,
                   "total_images": 2625,
                   "phase_cohorts_fixed_across_all_orders": True,
                   "primary_order": PRIMARY_ORDER,
                   "stress_orders": list(STRESS_ORDERS),
                   "diagnostic_order": DIAGNOSTIC_ORDER,
                   "order_seed": ORDER_SEED,
                   "bursty_run_samples": BURST_RUN,
                   "random_definition": "sample shuffle per phase, cameras C1..C5 consecutive; no labels read",
                   "stress_definition": "offline any-defect-visible sample grouping within each phase; synthetic, non-deployable",
                   "acquisition_definition": "phase-blocked capture-timestamp order from D1-A; not globally chronological"},
        "budgets": {
            "primary": list(MAIN_BUDGETS),
            "sensitivity": list(SENSITIVITY_BUDGETS),
            "diagnostic_only": list(LOW_BUDGET_DIAGNOSTIC),
            "units": {"image": "N_queries/2625",
                      "sample": "N_unique_query_samples/525"},
            "budget_cap_not_guaranteed_spend_for_Q_B_Q_C": True,
            "budget_79_is_historical_reference_not_optimum": True},
        "q_a": {"seeds": list(QUERY_SEEDS),
                "mode": "shared label-blind random tape",
                "sample_priority_common_across_orders": True,
                "same_camera_image_common_across_orders": True,
                "same_query_events_across_admission_arms": True,
                "query_positions_may_vary_with_order": True,
                "nested_budgets": True,
                "preselected_future_ids_are_offline_benchmark_only": True},
        "q_b": {
            "stage": "interface_and_rule_frozen_not_implemented",
            "reference": "frozen static support scorer, same initial LOSO theta0",
            "first_query_opportunity": "C1 arrival of each official sample; one query maximum",
            "normalization": "delta=max(1e-6, MAD of support-only LOSO normal scores)",
            "score_priority": "u=1/(1+abs(s_fixed-theta0)/delta)",
            "query_probability": "min(1,(B_remaining/N_remaining)*(0.5+1.5*u))",
            "random_draw": "separate deterministic hashed U(0,1) per sample and query seed",
            "spent_budget_handling": "cap B; no future-label backfill; report unspent B",
            "prohibited": ["dynamic compared admission bank", "sidecar state",
                           "correction state", "future/hidden labels",
                           "raw label-bearing filename/sample key to controller"],
            "warning": "causal prospective rule, unlike offline preselected Q-A tape"},
        "q_c": {
            "stage": "native_v22_policy_interface_frozen_not_implemented",
            "comparison": "cap-matched AIF-inspired vs random query controller",
            "independent_trajectories": True,
            "same_cap_not_same_realised_label_count": True,
            "activation_and_cost_ledger_required": True,
            "prohibited": ["future labels", "evaluation probes", "other arm state"]},
        "threshold": {
            "T0": "fixed theta0 derived from support-only LOSO",
            "T1": "existing v22 GuardedThetaSidecarConfig unchanged",
            "T2": "future confidence-sequence sidecar, outside current freeze"},
        "ablation": {
            "A_off": "verified-normal Greedy admission",
            "A_on": "CDARG, subject to independent implementation",
            "C_off": "disable signed correction; retain defect-anchor ledger",
            "T_off": "fixed LOSO theta0",
            "Q_off": "random query at same budget cap, not zero feedback"},
        "prequential_order": [
            "trusted adapter loads opaque observation; raw scorer predicts",
            "controller decides query using only pre-feedback state",
            "if queried reveal only that image-view label",
            "verified-normal proposal evaluated against strictly prior anchors",
            "after admission, append current label to ledgers and update optional components",
            "evaluate holdout probes without exposing their labels to online controller"],
        "priority_execution": {
            "D1B": "six classes/3 query seeds/5 orders/4 primary budgets: metadata only",
            "D1C_smoke": {"classes": list(INITIAL_SMOKE_CLASSES),
                          "support_seeds": list(INITIAL_SMOKE_SUPPORT_SEEDS),
                          "query_seeds": list(INITIAL_SMOKE_QUERY_SEEDS),
                          "orders": [PRIMARY_ORDER],
                          "budgets": list(MAIN_BUDGETS)},
            "D2_primary": "run paired Q-A admission arms under random mode first",
            "D4": "shared Q-B informative tape smoke; no retroactive tuning",
            "D5": "stress orders and auxiliary high-budget conditions",
            "D6": "Q-C native AIF factorial after module isolation"},
        "scope": "metadata protocol registration only; no GPU, no CDARG, no Q-B/Q-C execution",
    }


def assemble(dataset_root: Path, json_dir: Path
             ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    base = protocol_block()
    details: Dict[str, Any] = {}
    queries: Dict[str, Any] = {}
    records_by_class: Dict[str, Any] = {}
    for cls in DEV_CLASSES:
        train, test = pf.load_class_records(dataset_root, json_dir, cls)
        records_by_class[cls] = (train, test)
        plan = pf.build_partitions(train, test, pf.SplitConfig())
        universe = tuple(sorted(set(k for phase in plan.phase_samples for k in phase)))
        if len(universe) != 525:
            raise D1BProtocolError(f"{cls}: expected 525 disjoint sample groups")
        labels = {r.path: bool(r.is_good) for r in test}
        streams = {mode: build_ordered_stream(plan.phases, mode, cls)
                   for mode in ORDERS}
        image_sets = {mode: {x.rel_path for x in st} for mode, st in streams.items()}
        if len({frozenset(v) for v in image_sets.values()}) != 1:
            raise D1BProtocolError(f"{cls}: stream membership changed with order")
        phase_sets = {
            mode: tuple(frozenset(x.sample_key for x in st if x.phase == phase)
                        for phase in (1, 2, 3))
            for mode, st in streams.items()}
        if len(set(phase_sets.values())) != 1:
            raise D1BProtocolError(f"{cls}: phase cohort changed with order")
        audit = {}
        qa = {}
        for mode, stream in streams.items():
            audit[mode] = {
                "stream_sha256": qp.stream_digest(stream),
                "N_images": len(stream),
                "N_samples": len(set(x.sample_key for x in stream)),
                **audit_order(stream, labels)}
            if audit[mode]["N_images"] != 2625:
                raise D1BProtocolError(f"{cls}/{mode}: wrong image count")
            qa[mode] = {}
        for query_seed in QUERY_SEEDS:
            priority = shared_query_priority(universe, cls, query_seed)
            same_query = {}
            for mode, stream in streams.items():
                tapes = build_paired_tapes(stream, MAIN_BUDGETS, query_seed,
                                           cls, priority)
                qa[mode][str(query_seed)] = {}
                for B, tape in tapes.items():
                    selected = tuple(sorted((e.sample_key, e.rel_path, e.camera_id)
                                            for e in tape.events))
                    if B in same_query and selected != same_query[B]:
                        raise D1BProtocolError(
                            f"{cls} seed={query_seed} B={B}: order changed Q-A labels")
                    same_query[B] = selected
                    composition = qp.label_diagnostics(stream, tape, labels)
                    qa[mode][str(query_seed)][str(B)] = {
                        "digest": tape.digest,
                        "stream_digest": tape.stream_digest,
                        "N_queries": len(tape.events),
                        "N_normal_labels_offline_only":
                            composition["N_normal_labels"],
                        "N_defect_labels_offline_only":
                            composition["N_defect_labels"],
                        "events": [e.as_manifest() for e in tape.events],
                    }
        details[cls] = {
            "N_stream_samples": len(universe),
            "order_audit": audit,
            "official_sample_disjoint": pf.audit_sample_disjointness(plan),
            "all_modes_same_sample_universe": True,
            "all_modes_same_phase_cohorts": True,
            "stress_order_labels_offline_only": True}
        queries[cls] = qa
    fp = pf.dataset_fingerprint(json_dir, DEV_CLASSES, records_by_class)
    payloads = {
        OUTPUT_PROTOCOL: base,
        OUTPUT_QUERIES: {
            "schema": "risk_safe.d1b.query_manifest.v1",
            "query_seeds": list(QUERY_SEEDS),
            "budgets": list(MAIN_BUDGETS),
            "modes": list(ORDERS),
            "class_records": queries,
            "same_queried_sample_and_camera_across_orders": True,
            "query_tapes_use_offline_preselected_universe": True},
        OUTPUT_ORDERS: {
            "schema": "risk_safe.d1b.order_audit.v1",
            "classes": details,
            "labels_only_used_for_synthetic_stress_and_offline_audit": True},
    }
    return payloads, fp


def make_source_manifest(payloads: Mapping[str, Any],
                         fingerprint: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema": "risk_safe.d1b.source_manifest.v1",
        "git": pf.git_state(),
        "sources_sha256": pf.source_hashes(SOURCE_FILES),
        "dataset": fingerprint,
        "output_sha256": {name: pf.sha256_text(pf.json_text(payloads[name]))
                          for name in (OUTPUT_PROTOCOL, OUTPUT_QUERIES,
                                       OUTPUT_ORDERS)},
        "config_sha256": qp.sha256_of(protocol_block()),
        "python_environment": pf.environment_block(),
        "R2_preflight_remains_immutable": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=pf.DEFAULT_DATASET_ROOT)
    parser.add_argument("--json-dir", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.dataset_root)
    json_dir = Path(args.json_dir) if args.json_dir else root / pf.DEFAULT_JSON_SUBDIR
    out = Path(args.out_dir)
    payloads, fp = assemble(root, json_dir)
    payloads[OUTPUT_SOURCE] = pf.preserve_committed_source_provenance(
        out / OUTPUT_SOURCE, make_source_manifest(payloads, fp))
    states = {}
    for name in ALL_OUTPUTS:
        states[name] = (pf.verify_json_output(out / name, payloads[name])
                        if args.verify_only
                        else pf.write_json_idempotent(out / name, payloads[name]))
    print(json.dumps({
        "stage": "D1-B", "output_status": states,
        "classes": len(DEV_CLASSES), "orders": list(ORDERS),
        "primary_budgets": list(MAIN_BUDGETS),
        "query_seeds": list(QUERY_SEEDS),
        "N_query_tapes": len(DEV_CLASSES) * len(ORDERS) *
                         len(QUERY_SEEDS) * len(MAIN_BUDGETS),
        "sources_sha256": payloads[OUTPUT_SOURCE]["sources_sha256"],
        "performance_metrics_computed": False,
        "gpu_used": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
