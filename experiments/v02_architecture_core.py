"""V0.2 pre-registered paired cold/mature memory architecture protocol.

All parameters fixed BEFORE observing V0.2 GPU outcomes. Historical source files
are immutable. Scores/labels of held-out probes are strictly evaluation-only.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import random
import final_decision_core as legacy

SCHEMA = "memory.v02.paired.v1"
CLASSES = ("audiojack", "bottle_cap", "eraser", "plastic_plug",
           "switch", "rolled_strip_base")
SEEDS = (0, 1, 2)
ORDERS = ("ABA", "AAA")
REGIMES = ("cold", "mature")
ARMS = ("greedy_cold", "hierarchy_cold", "greedy_mature", "hierarchy_mature")
N_HISTORY = 64
PROBE_NORMAL = 24
PROBE_DEFECT = 24
GROUPS_PER_PHASE = 175
QUERY_RATE = .03
STM_K, LTM_K, BUFFER_K = 1024, 512, 4096
PROMOTE_EVERY, MIN_EVENTS = 32, 2

object_id = legacy.object_id
group_samples = legacy.group_samples
digest = legacy.digest
auroc = legacy.auroc
query_tape = legacy.query_tape
phase_modes = legacy.phase_modes
recovery_count = legacy.recovery_count


@dataclass(frozen=True)
class MemorySettings:
    stm: int
    ltm: int
    buffer: int
    flat: int
    cap_bytes: int
    promote_every: int = PROMOTE_EVERY
    min_events: int = MIN_EVENTS


def make_memory_settings(bank):
    """Match max reserved persistent feature+metadata bytes to within 5%.

    The greedy upper bound is derived from actual concatenated feature widths,
    not a hard-coded 4096 assumption. 80 B/prototype is a conservative
    *capacity estimate*; separately log actual tensor storage and RAM/VRAM.
    """
    cb = bank.normal_dyn["p0"][0]["id"]
    dims = [int(x.feat_dim) for x in cb]
    if not dims or any(x <= 0 for x in dims):
        raise ValueError("Missing layer dimension")
    row_fp16 = sum(2 * d + 80 for d in dims)
    row_fp32 = sum(4 * d + 80 for d in dims)
    hierarchy_reserved = (STM_K + LTM_K) * row_fp16 + BUFFER_K * row_fp32
    disabled = row_fp16 + row_fp32
    greedy_k = max(1, int((hierarchy_reserved - disabled) // row_fp16))
    greedy_reserved = greedy_k * row_fp16 + disabled
    relative = abs(greedy_reserved - hierarchy_reserved) / max(greedy_reserved, hierarchy_reserved)
    if relative > .05:
        raise AssertionError("Nominal capacity parity cannot be achieved")
    # Actual PyTorch tensor storage includes auxiliary arrays, so allow explicit
    # headroom above the nominal upper-bound on feature prototypes.
    limit = math.ceil(max(greedy_reserved, hierarchy_reserved) * 1.20)
    return MemorySettings(STM_K, LTM_K, BUFFER_K, greedy_k, limit)


def select_history_and_support(train, seed):
    """Common 64 normal history objects across support seeds, then strict 4-shot.

    Only paths/metadata are inspected to select objects; no nonselected image
    is decoded or embedded for support ranking. Extra 64 verified normal
    images are *charged to mature*, never to the cold-start learner.
    """
    if seed not in SEEDS:
        raise ValueError("Unregistered support seed")
    groups = group_samples(s for s in train if s.is_good)
    ids = sorted(groups)
    if len(ids) < N_HISTORY + 4:
        raise ValueError(f"Need {N_HISTORY+4} independent normal train objects; have {len(ids)}")
    history_ids = sorted(random.Random(190021).sample(ids, N_HISTORY))
    excluded = set(history_ids)
    support_ids = sorted(random.Random(seed).sample([x for x in ids if x not in excluded], 4))
    # Sampling the camera view deterministically ensures history identity is
    # invariant across support seeds, while support identity changes.
    history = [random.Random(891+i).choice(groups[obj]) for i, obj in enumerate(history_ids)]
    rng = random.Random(seed + 891)
    supports = sorted(rng.choice(groups[obj]).path for obj in support_ids)
    if set(map(object_id, supports)) & set(history_ids):
        raise AssertionError("History/support instance leakage")
    if len(set(map(object_id, supports))) != 4:
        raise AssertionError("Supports are not four physical objects")
    return supports, history


def split_test(test):
    """One object-disjoint test split per class, independent of support seed.

    Evaluation-only A/B probe selection is stratified by *object* labels.
    Ground truth is not used to select stream queries, updates or thresholds.
    """
    groups = group_samples(test)
    normal_ids = sorted(k for k,v in groups.items() if all(x.is_good for x in v))
    defect_ids = sorted(k for k,v in groups.items() if any(not x.is_good for x in v))
    rng = random.Random(2201)
    rng.shuffle(normal_ids)
    rng.shuffle(defect_ids)
    need = 2 * PROBE_NORMAL, 2 * PROBE_DEFECT
    if len(normal_ids) < need[0] or len(defect_ids) < need[1]:
        raise ValueError(f"Insufficient probe labels: normal={len(normal_ids)}, defect={len(defect_ids)}")
    aid = sorted(normal_ids[:PROBE_NORMAL] + defect_ids[:PROBE_DEFECT])
    bid = sorted(normal_ids[PROBE_NORMAL:2*PROBE_NORMAL] +
                 defect_ids[PROBE_DEFECT:2*PROBE_DEFECT])
    excluded = set(aid + bid)
    remaining = sorted(set(groups) - excluded)
    if len(remaining) < 3 * GROUPS_PER_PHASE:
        raise ValueError(f"Need {3*GROUPS_PER_PHASE} stream physical objects; have {len(remaining)}")
    rng.shuffle(remaining)
    phase_ids = [remaining[i*GROUPS_PER_PHASE:(i+1)*GROUPS_PER_PHASE]
                 for i in range(3)]
    identity_lists = [aid, bid] + phase_ids
    flat = sum(identity_lists, [])
    if len(flat) != len(set(flat)):
        raise AssertionError("Object overlap across probes/stream")
    def images(ids):
        return [x for key in ids for x in groups[key]]
    return {"A": images(aid), "B": images(bid)}, [images(p) for p in phase_ids], {
        "A": aid, "B": bid, "phases": phase_ids, "stratified_probes_eval_only": True
    }


def decision(paired):
    """Exploratory engineering gate; no claim of population significance.

    Macro-average over six *classes*, after averaging support seeds within
    each class. Cold performance and absolute A/B detection are safety gates.
    """
    if len(paired) != len(CLASSES) * len(SEEDS):
        raise ValueError("Missing pair, no verdict")
    by_class = {}
    for row in paired:
        by_class.setdefault(row["class_name"], []).append(row)
    if set(by_class) != set(CLASSES) or any(len(v)!=len(SEEDS) for v in by_class.values()):
        raise ValueError("Unbalanced class/seed matrix")
    mean = lambda seq: sum(seq) / len(seq)
    per_class = {}
    for cl, rows in by_class.items():
        per_class[cl] = {
            "retention_advantage_mature": mean([r["retention_advantage_mature"] for r in rows]),
            "A_quality_mature": mean([r["A_quality_mature"] for r in rows]),
            "B_quality_mature": mean([r["B_quality_mature"] for r in rows]),
            "A_quality_cold": mean([r["A_quality_cold"] for r in rows]),
            "real_source_positive": any(r["source_effect_mature"] >= .005 and
                                        r["ltm_mature"] > 0 for r in rows),
        }
    avg = lambda key: mean([r[key] for r in per_class.values()])
    positive = sum(r["retention_advantage_mature"] > 0 for r in per_class.values())
    source = sum(r["real_source_positive"] for r in per_class.values())
    gates = {
        "mature_retention_gain_0.015": avg("retention_advantage_mature") >= .015,
        "positive_in_four_of_six_classes": positive >= 4,
        "mature_A_noninferiority_0.01": avg("A_quality_mature") >= -.01,
        "mature_B_noninferiority_0.01": avg("B_quality_mature") >= -.01,
        "cold_A_noninferiority_0.02": avg("A_quality_cold") >= -.02,
        "source_effect_in_two_classes": source >= 2,
    }
    keep = all(gates.values())
    return {
        "decision": ("RETAIN_HIERARCHY" if keep else "USE_GREEDY_WITH_RISK_SAFE_ADMISSION"),
        "gate_results": gates,
        "macro_retention_advantage": avg("retention_advantage_mature"),
        "positive_classes": positive, "source_effect_classes": source,
        "per_class": per_class,
        "interpretation": "Engineering research selection under tested classes, shift, and resource regime; not universal superiority.",
        "runtime_cost_gate": "Descriptive only; per-operation update/retrieval seconds and allocated tensors must be inspected before deployment.",
    }
