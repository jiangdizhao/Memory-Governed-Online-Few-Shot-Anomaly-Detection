"""Stage 0.5 label-blind tape construction and evaluator-only statistics.

This module is intentionally separate from every online controller. Query tapes
accept only label-free ``TapeRecord`` metadata. The JSON reader can discard test
label and mask fields while the frozen tapes and support identities are built.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .profiles import canonical_json
from .protocol import TapeRecord
from .query_tape import (QueryTape, QueryTapeError, _assign_bin, _bin_edges,
                         _canonical_by_group, _hash_int, _largest_remainder,
                         _select_by_bins, build_qa_ts, half_up)

DEVELOPMENT_CLASSES = ("audiojack", "regulator", "switch", "rolled_strip_base")
EXPECTED_STREAM_LENGTHS = {
    "audiojack": 3815, "regulator": 3855, "switch": 3820,
    "rolled_strip_base": 3805,
}
SUPPORT_SEEDS = (0, 1, 2, 3, 4)
QUERY_SEEDS = (37, *range(1000, 1199))
QUERY_RATES = (0.03, 0.05, 0.07)
HISTORY_SEED = 190021
SUPPORT_VIEW_SEED_OFFSET = 891


class Stage05Error(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def opaque_digest(domain: str, values: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return sha256_bytes(domain.encode("utf-8") + b"\0" + payload)


def _skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def read_json_sections(path: str | Path, *, train_labels: bool = False,
                       test_labels: bool = False) -> dict[str, Any]:
    """Read top-level JSON sections with test labels/masks dropped by default.

    JSONDecoder's object_pairs_hook discards fields before record dictionaries
    are returned to the caller. In the freeze/score phases, test
    ``anomaly_class`` and ``mask_path`` are never present in returned objects.
    Training labels are included only when explicitly required for the frozen
    normal support/C64 draw.
    """
    text = Path(path).read_text(encoding="utf-8")
    regular = json.JSONDecoder()

    def strip_test_fields(pairs):
        return {key: value for key, value in pairs
                if key not in {"anomaly_class", "mask_path"}}

    stripped = json.JSONDecoder(object_pairs_hook=strip_test_fields)
    index = _skip_ws(text, 0)
    if index >= len(text) or text[index] != "{":
        raise Stage05Error(f"Expected top-level object in {path}")
    index += 1
    result: dict[str, Any] = {}
    while True:
        index = _skip_ws(text, index)
        if index < len(text) and text[index] == "}":
            index += 1
            break
        key, index = regular.raw_decode(text, index)
        index = _skip_ws(text, index)
        if index >= len(text) or text[index] != ":":
            raise Stage05Error(f"Malformed object member in {path}")
        index = _skip_ws(text, index + 1)
        if key == "train":
            decoder = regular if train_labels else stripped
        elif key == "test":
            decoder = regular if test_labels else stripped
        else:
            decoder = regular
        result[key], index = decoder.raw_decode(text, index)
        index = _skip_ws(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "}":
            index += 1
            break
        raise Stage05Error(f"Malformed top-level JSON in {path}")
    if _skip_ws(text, index) != len(text):
        raise Stage05Error(f"Trailing JSON content in {path}")
    if not isinstance(result.get("meta"), dict):
        raise Stage05Error(f"Missing meta section in {path}")
    if not isinstance(result.get("train"), list) or not isinstance(result.get("test"), list):
        raise Stage05Error(f"Missing train/test arrays in {path}")
    return result


@dataclass(frozen=True)
class StreamImage:
    position: int
    image_path: str
    relative_id: str
    sample_key: str
    opaque_group: str
    camera_id: int
    timestamp: str
    view_count: int


@dataclass(frozen=True)
class TrainImage:
    image_path: str
    relative_id: str
    sample_key: str
    is_good: bool
    camera_id: int
    timestamp: str


def _absolute_image_path(dataset_root: str | Path, prefix: str,
                         image_path: str) -> str:
    root = Path(dataset_root) / "images"
    parts = [part for part in str(prefix).replace("\\", "/").split("/") if part]
    parts.extend(part for part in str(image_path).replace("\\", "/").split("/")
                 if part)
    return str(root.joinpath(*parts))


def _parsed_records(rows: Sequence[Mapping[str, Any]], *, class_name: str,
                    dataset_root: str | Path, prefix: str,
                    normal_class: str, labels_present: bool,
                    stream: bool) -> list[StreamImage] | list[TrainImage]:
    from experiments.risk_safe.preflight import parse_image_name

    paths: set[str] = set()
    group_counts: Counter[str] = Counter()
    parsed: list[tuple[str, Any, Mapping[str, Any]]] = []
    for row in rows:
        image_rel = row.get("image_path")
        if not isinstance(image_rel, str) or not image_rel:
            raise Stage05Error(f"{class_name}: JSON image_path is missing")
        path = _absolute_image_path(dataset_root, prefix, image_rel)
        item = parse_image_name(path, expected_class=class_name)
        if item.rel_path in paths:
            raise Stage05Error(f"{class_name}: duplicate image path {item.rel_path}")
        paths.add(item.rel_path)
        parsed.append((path, item, row))
        group_counts[item.official_sample_key] += 1
    if stream:
        result = []
        for position, (path, item, _) in enumerate(parsed):
            result.append(StreamImage(
                position=position, image_path=path, relative_id=item.rel_path,
                sample_key=item.official_sample_key,
                opaque_group=hashlib.sha256(
                    f"official-sample-group\0{item.official_sample_key}".encode()
                ).hexdigest(), camera_id=int(item.camera_id),
                timestamp=item.timestamp,
                view_count=group_counts[item.official_sample_key]))
        return result
    result = []
    for path, item, row in parsed:
        is_good = (row.get("anomaly_class", normal_class) == normal_class
                   if labels_present else False)
        result.append(TrainImage(
            image_path=path, relative_id=item.rel_path,
            sample_key=item.official_sample_key, is_good=is_good,
            camera_id=int(item.camera_id), timestamp=item.timestamp))
    return result


def read_class_stream(dataset_root: str | Path, json_root: str | Path,
                      class_name: str, *, test_labels: bool = False
                      ) -> tuple[list[TrainImage], list[StreamImage], str]:
    """Read one class, preserving JSON order and label-field access policy."""
    if class_name not in DEVELOPMENT_CLASSES:
        raise Stage05Error(f"Class outside the authorized development set: {class_name}")
    source = Path(json_root) / f"{class_name}.json"
    if not source.is_file():
        raise Stage05Error(f"Missing official category JSON: {source}")
    data = read_json_sections(source, train_labels=True, test_labels=test_labels)
    meta = data["meta"]
    prefix = str(meta.get("prefix", ""))
    normal_class = str(meta.get("normal_class", "OK"))
    train = _parsed_records(data["train"], class_name=class_name,
                            dataset_root=dataset_root, prefix=prefix,
                            normal_class=normal_class, labels_present=True,
                            stream=False)
    test = _parsed_records(data["test"], class_name=class_name,
                           dataset_root=dataset_root, prefix=prefix,
                           normal_class=normal_class, labels_present=test_labels,
                           stream=True)
    return train, test, sha256_file(source)


def test_labels_for_class(json_root: str | Path, class_name: str) -> list[int]:
    """Evaluator-only access to image-level test labels in official JSON order."""
    if class_name not in DEVELOPMENT_CLASSES:
        raise Stage05Error(f"Class outside the authorized development set: {class_name}")
    source = Path(json_root) / f"{class_name}.json"
    data = read_json_sections(source, train_labels=False, test_labels=True)
    normal_class = str(data["meta"].get("normal_class", "OK"))
    labels = [0 if row.get("anomaly_class", normal_class) == normal_class else 1
              for row in data["test"]]
    return labels


def select_supports(train: Sequence[TrainImage]) -> dict[str, Any]:
    """Reproduce historical C64 and five approved train-normal support seeds."""
    groups: dict[str, list[TrainImage]] = {}
    for record in train:
        groups.setdefault(record.sample_key, []).append(record)
    normal_keys = sorted(key for key, views in groups.items()
                         if all(view.is_good for view in views))
    if len(normal_keys) < 68:
        raise Stage05Error(f"Need 68 train-normal official groups, found {len(normal_keys)}")
    c64_keys = sorted(random.Random(HISTORY_SEED).sample(normal_keys, 64))
    c64_rows = [random.Random(SUPPORT_VIEW_SEED_OFFSET + index).choice(groups[key])
                for index, key in enumerate(c64_keys)]
    remaining = [key for key in normal_keys if key not in set(c64_keys)]
    supports: dict[str, Any] = {}
    for seed in SUPPORT_SEEDS:
        support_keys = sorted(random.Random(seed).sample(remaining, 4))
        view_rng = random.Random(seed + SUPPORT_VIEW_SEED_OFFSET)
        rows = [view_rng.choice(groups[key]) for key in support_keys]
        supports[str(seed)] = {
            "seed": seed,
            "groups": support_keys,
            "images": [row.relative_id for row in rows],
            "group_sha256": opaque_digest("support-groups", support_keys),
            "image_sha256": opaque_digest(
                "support-images", [row.relative_id for row in rows]),
            "all_train_normal": all(row.is_good for row in rows),
            "c64_disjoint": not bool(set(support_keys) & set(c64_keys)),
        }
    c64 = {
        "groups": c64_keys,
        "images": [row.relative_id for row in c64_rows],
        "group_sha256": opaque_digest("c64-groups", c64_keys),
        "image_sha256": opaque_digest(
            "c64-images", [row.relative_id for row in c64_rows]),
        "count": len(c64_rows),
        "selection_seed": HISTORY_SEED,
    }
    if len({row.relative_id for row in c64_rows}) != 64:
        raise Stage05Error("C64 image identities are not unique")
    for seed, item in supports.items():
        if len(set(item["groups"])) != 4 or len(set(item["images"])) != 4:
            raise Stage05Error(f"Support seed {seed} does not contain four distinct images/groups")
    return {"c64": c64, "supports": supports,
            "normal_train_groups": len(normal_keys),
            "support_c64_disjoint": all(item["c64_disjoint"]
                                         for item in supports.values())}


@dataclass(frozen=True)
class Stage05QueryConfig:
    """Q-A-TS-v1-compatible fields, allowing preregistered sensitivity seeds."""
    version: str = "Q-A-TS-v1"
    bins: int = 16
    fraction: float = 0.03
    early_observations: int = 400
    q_b_early_fraction: float = 0.20
    seed: int = 37
    rounding: str = "half_up_floor_0.03N_plus_0.5"
    group_unit: str = "official_sample_key"


def tape_records(stream: Sequence[StreamImage]) -> list[TapeRecord]:
    return [TapeRecord(
        position=item.position, official_sample_key=item.opaque_group,
        camera_id=item.camera_id,
        relative_id=hashlib.sha256(
            f"image-identity\0{item.relative_id}".encode()).hexdigest(),
        timestamp=item.timestamp, phase=0) for item in stream]


def _event_payload(event: Any) -> dict[str, Any]:
    return {"position": int(event.position),
            "official_sample_key": event.official_sample_key,
            "camera_id": int(event.camera_id),
            "relative_id": event.relative_id,
            "phase": int(event.phase), "time_bin": int(event.time_bin)}


def build_nested_tapes(records: Sequence[TapeRecord], seed: int
                       ) -> dict[float, dict[str, Any]]:
    """Build exact 3% Q-A-TS then deterministic nested 5%/7% extensions."""
    config = Stage05QueryConfig(seed=int(seed))
    base = build_qa_ts(records, config)
    canonical = _canonical_by_group(records, int(seed))
    edges = _bin_edges(len(records), 16)
    capacities = [0] * 16
    for item in canonical.values():
        capacities[_assign_bin(item.position, edges)] += 1
    result: dict[float, dict[str, Any]] = {
        0.03: {
            "version": "Q-A-TS-v1", "rate": 0.03, "seed": int(seed),
            "digest": base.digest, "stream_digest": base.stream_digest,
            "requested_budget": base.requested_budget,
            "bin_edges": list(base.bin_edges), "bin_quotas": list(base.bin_quotas),
            "events": [_event_payload(event) for event in base.events],
            "parent_digest": None,
        }}
    prior_events = list(base.events)
    prior_digest = base.digest
    prior_quotas = list(base.bin_quotas)
    for rate in (0.05, 0.07):
        budget = half_up(float(rate) * len(records))
        quotas = _largest_remainder(capacities, budget)
        if any(next_quota < previous for next_quota, previous in
               zip(quotas, prior_quotas)):
            raise QueryTapeError(
                f"Nested budget infeasible at {rate:.0%}: bin quotas would shrink")
        used_positions = {event.position for event in prior_events}
        used_groups = {event.official_sample_key for event in prior_events}
        current_bin_counts = Counter(event.time_bin for event in prior_events)
        additions = []
        by_bin: list[list[TapeRecord]] = [[] for _ in range(16)]
        for item in canonical.values():
            by_bin[_assign_bin(item.position, edges)].append(item)
        for bin_index, (items, quota) in enumerate(zip(by_bin, quotas)):
            needed = quota - current_bin_counts[bin_index]
            available = [item for item in items
                         if item.position not in used_positions and
                         item.official_sample_key not in used_groups]
            if needed < 0 or len(available) < needed:
                raise QueryTapeError(
                    f"Nested budget infeasible at {rate:.0%}, bin {bin_index}")
            available.sort(key=lambda item: (
                _hash_int(seed, "Q-A-TS-BudgetExtension-v1", f"{rate:.2f}",
                          prior_digest, bin_index, item.position, item.camera_id),
                item.position))
            for item in available[:needed]:
                additions.append({
                    "position": item.position,
                    "official_sample_key": item.official_sample_key,
                    "camera_id": item.camera_id,
                    "relative_id": item.relative_id,
                    "phase": item.phase,
                    "time_bin": bin_index,
                })
                used_positions.add(item.position)
                used_groups.add(item.official_sample_key)
        events = sorted([_event_payload(event) for event in prior_events] + additions,
                        key=lambda item: item["position"])
        if len(events) != budget or len(used_groups) != budget:
            raise QueryTapeError(f"Nested budget {rate:.0%} has invalid size/group count")
        payload = {
            "schema": "calibration.stage05.query_tape_extension.v1",
            "policy": "Q-A-TS-BudgetExtension-v1",
            "base_policy": "Q-A-TS-v1",
            "rate": float(rate), "seed": int(seed),
            "parent_digest": prior_digest,
            "n_images": len(records), "n_groups": len(canonical),
            "requested_budget": budget, "stream_digest": base.stream_digest,
            "bin_edges": list(edges), "bin_quotas": quotas,
            "events": events,
        }
        digest = sha256_bytes(canonical_json(payload).encode("utf-8"))
        result[float(rate)] = {**payload, "version": "Q-A-TS-BudgetExtension-v1",
                               "digest": digest}
        prior_events = [type("Event", (), event)() for event in events]
        prior_digest, prior_quotas = digest, quotas
    return result


def verify_nested_tapes(tapes: Mapping[float, Mapping[str, Any]],
                        records: Sequence[TapeRecord]) -> None:
    expected_rates = set(QUERY_RATES)
    if set(tapes) != expected_rates:
        raise Stage05Error("All three registered budget tapes are required")
    positions = {rate: {int(event["position"]) for event in tape["events"]}
                 for rate, tape in tapes.items()}
    groups_by_position = {record.position: record.official_sample_key
                          for record in records}
    for rate in QUERY_RATES:
        tape = tapes[rate]
        if rate > .03:
            digest_payload = {key: value for key, value in tape.items()
                              if key not in {"version", "digest"}}
            observed_digest = sha256_bytes(canonical_json(digest_payload).encode("utf-8"))
            if observed_digest != tape.get("digest"):
                raise Stage05Error(f"{rate:.0%} extension tape digest mismatch")
            expected_parent = tapes[.03 if rate == .05 else .05]["digest"]
            if tape.get("parent_digest") != expected_parent:
                raise Stage05Error(f"{rate:.0%} extension parent digest mismatch")
        expected = half_up(rate * len(records))
        if tape["requested_budget"] != expected or len(positions[rate]) != expected:
            raise Stage05Error(f"{rate:.0%} tape budget does not match half-up rule")
        selected_groups = [groups_by_position[position] for position in positions[rate]]
        if len(set(selected_groups)) != expected:
            raise Stage05Error(f"{rate:.0%} tape violates one-view-per-group")
        counts = [0] * 16
        for event in tape["events"]:
            counts[int(event["time_bin"])] += 1
        if counts != tape["bin_quotas"]:
            raise Stage05Error(f"{rate:.0%} tape temporal quotas do not match")
    if not positions[0.03] <= positions[0.05] <= positions[0.07]:
        raise Stage05Error("Budget extension tapes are not nested")
    seed = int(tapes[0.03]["seed"])
    base = build_qa_ts(records, Stage05QueryConfig(seed=seed))
    if tapes[0.03]["digest"] != base.digest or positions[0.03] != set(base.positions):
        raise Stage05Error("3% tape differs from the registered Q-A-TS-v1 builder")


def _confusion(labels: Sequence[int], scores: Sequence[float], threshold: float
               ) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) != len(s) or not len(y) or not np.isfinite(s).all():
        raise Stage05Error("Metric inputs must be nonempty, finite, and aligned")
    pred = s >= float(threshold)
    tp = int(np.count_nonzero(pred & (y == 1)))
    tn = int(np.count_nonzero((~pred) & (y == 0)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero((~pred) & (y == 1)))
    n0, n1 = tn + fp, tp + fn
    fpr = fp / n0 if n0 else None
    fnr = fn / n1 if n1 else None
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "n_normal": n0, "n_defect": n1,
            "balanced_accuracy": (1.0 - 0.5 * (fpr + fnr)
                                  if fpr is not None and fnr is not None else None),
            "fpr": fpr, "fnr": fnr,
            "defect_recall": 1.0 - fnr if fnr is not None else None}


def retrospective_threshold_oracle(labels: Sequence[int], scores: Sequence[float],
                                    frozen_threshold: float) -> dict[str, Any]:
    """Exact in-sample scalar threshold oracle for prediction ``score >= theta``."""
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) != len(s) or not len(y) or set(np.unique(y)) != {0, 1}:
        raise Stage05Error("Oracle requires aligned scores and both binary classes")
    if not np.isfinite(s).all() or not math.isfinite(float(frozen_threshold)):
        raise Stage05Error("Oracle inputs must be finite")
    normals = np.sort(s[y == 0])
    defects = np.sort(s[y == 1])
    candidates = set(float(value) for value in np.unique(s))
    above_max = float(np.nextafter(float(np.max(s)), math.inf))
    candidates.add(above_max)
    scored = []
    for threshold in candidates:
        tn = int(np.searchsorted(normals, threshold, side="left"))
        tp = int(len(defects) - np.searchsorted(defects, threshold, side="left"))
        # Compare exact rational balanced-accuracy numerators, avoiding float ties.
        objective = Fraction(tn, len(normals)) + Fraction(tp, len(defects))
        scored.append((objective, threshold, tn, tp))
    best_value = max(item[0] for item in scored)
    best = [item for item in scored if item[0] == best_value]
    chosen = min(best, key=lambda item: (abs(item[1] - frozen_threshold), -item[1]))
    _, threshold, tn, tp = chosen
    fp, fn = len(normals) - tn, len(defects) - tp
    return {
        "label": "IN_SAMPLE_RETROSPECTIVE_STATIC_THRESHOLD_ORACLE",
        "threshold": float(threshold),
        "balanced_accuracy": float(best_value / 2),
        "fpr": fp / len(normals), "fnr": fn / len(defects),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "number_of_ba_tied_thresholds": len(best),
        "tie_break": "minimum absolute movement from theta0, then higher threshold",
        "boundary_candidates": {"all_defect": float(np.min(s)),
                                 "none_defect": above_max},
        "prediction_convention": "score >= threshold",
        "oracle_is_online_or_unbiased": False,
    }


def query_schedule_metrics(stream: Sequence[StreamImage], labels: Sequence[int],
                           tape: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluator-only label join for one frozen query tape."""
    if len(stream) != len(labels):
        raise Stage05Error("Evaluator labels do not align with frozen stream metadata")
    queried = tape["events"]
    selected = [int(event["position"]) for event in queried]
    y = [int(value) for value in labels]
    if any(value not in (0, 1) for value in y):
        raise Stage05Error("Evaluator labels must be binary normal=0/defect=1")
    full_defects = sum(y)
    qlabels = [y[position] for position in selected]
    cameras = Counter(str(stream[position].camera_id) for position in selected)
    bins = Counter(str(int(event["time_bin"])) for event in queried)
    group_views = Counter(str(stream[position].view_count) for position in selected)
    first400 = [position for position in selected if position < min(400, len(stream))]
    first400_y = [y[position] for position in first400]
    defect_positions = [position for position, label in zip(selected, qlabels) if label == 1]
    first_activation = {}
    for threshold in (1, 8, 14):
        first_activation[str(threshold)] = (
            defect_positions[threshold - 1] if len(defect_positions) >= threshold else None)
    if defect_positions:
        gaps = [defect_positions[0]]
        gaps.extend(right - left - 1 for left, right in
                    zip(defect_positions[:-1], defect_positions[1:]))
        gaps.append(len(stream) - 1 - defect_positions[-1])
        longest_no_defect_span = max(gaps)
    else:
        longest_no_defect_span = len(stream)
    if len(stream) >= 256:
        prefix = np.concatenate(([0], np.cumsum(np.asarray(y, dtype=np.int64))))
        qmask = np.zeros(len(stream), dtype=np.int64)
        qmask[selected] = np.asarray(qlabels, dtype=np.int64)
        qprefix = np.concatenate(([0], np.cumsum(qmask)))
        # Sliding window query-defect counts, only at starts of 256-image windows.
        window_defects = (qprefix[256:] - qprefix[:-256]).astype(np.int64)
        window_count = len(window_defects)
    else:
        window_defects, window_count = np.asarray([], dtype=np.int64), 0
    return {
        "seed": int(tape["seed"]), "rate": float(tape["rate"]),
        "query_count": len(selected), "queried_normals": len(selected) - sum(qlabels),
        "queried_defects": int(sum(qlabels)),
        "queried_defect_fraction": float(np.mean(qlabels)) if qlabels else None,
        "full_stream_images": len(stream), "full_stream_normals": len(stream) - full_defects,
        "full_stream_defects": full_defects,
        "full_stream_defect_fraction": full_defects / len(stream),
        "stream_official_sample_groups": len({row.sample_key for row in stream}),
        "queried_official_sample_groups": len({event["official_sample_key"]
                                                for event in queried}),
        "query_counts_by_camera": dict(sorted(cameras.items())),
        "query_counts_by_temporal_bin": {str(i): int(bins.get(str(i), 0))
                                          for i in range(16)},
        "selected_groups_by_available_camera_views": dict(sorted(group_views.items())),
        "first_400_queries": {"count": len(first400),
                               "normals": len(first400) - sum(first400_y),
                               "defects": int(sum(first400_y))},
        "first_queried_defect_positions": first_activation,
        "longest_consecutive_image_span_without_queried_defect": int(longest_no_defect_span),
        "sliding_256_image_windows": {
            "count": int(window_count),
            "queried_defect_min": int(window_defects.min()) if window_count else None,
            "queried_defect_max": int(window_defects.max()) if window_count else None,
            "queried_defect_quantiles": ({str(q): float(np.quantile(window_defects, q))
                                           for q in (.05, .25, .5, .75, .95)}
                                          if window_count else {}),
        },
        "labels_exposed": True,
    }


def summarize_query_schedules(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray([int(row["queried_defects"]) for row in rows], dtype=np.float64)
    if not len(values):
        raise Stage05Error("No query schedule results to summarize")
    activation8 = [int(row["queried_defects"]) >= 8 for row in rows]
    activation14 = [int(row["queried_defects"]) >= 14 for row in rows]
    first8 = [row["first_queried_defect_positions"]["8"] for row in rows
              if row["first_queried_defect_positions"]["8"] is not None]
    first14 = [row["first_queried_defect_positions"]["14"] for row in rows
               if row["first_queried_defect_positions"]["14"] is not None]
    quantiles = {str(q): float(np.quantile(values, q))
                 for q in (.05, .25, .50, .75, .95)}
    return {
        "n_query_seeds": len(rows),
        "seed37_primary": next((dict(row) for row in rows if row["seed"] == 37), None),
        "queried_defect_count": {
            "minimum": int(values.min()), "maximum": int(values.max()),
            "quantiles_linear": quantiles, "mean": float(values.mean()),
            "sample_standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        },
        "probability_finite_seed_schedules_at_least_8_defects": float(np.mean(activation8)),
        "probability_finite_seed_schedules_at_least_14_defects": float(np.mean(activation14)),
        "first_activation_position": {
            "at_least_8": {"activated_schedules": len(first8),
                            "not_activated_schedules": len(rows) - len(first8),
                            "quantiles_linear": ({str(q): float(np.quantile(first8, q))
                                                  for q in (.05, .25, .50, .75, .95)}
                                                 if first8 else {})},
            "at_least_14": {"activated_schedules": len(first14),
                             "not_activated_schedules": len(rows) - len(first14),
                             "quantiles_linear": ({str(q): float(np.quantile(first14, q))
                                                   for q in (.05, .25, .50, .75, .95)}
                                                  if first14 else {})},
        },
        "finite_seed_sensitivity_only": True,
    }


def query_window_sensitivity(labels: Sequence[int],
                             tapes_by_seed: Mapping[int, Mapping[str, Any]],
                             *, window: int = 256) -> list[dict[str, Any]]:
    """Aggregate queried-defect counts for every sliding window across seeds.

    One row is returned per window start. Only aggregate finite-seed summaries
    and the primary seed-37 count are emitted; no per-image label is retained.
    """
    y = np.asarray(labels, dtype=np.int8)
    if (window < 1 or len(y) < window or set(np.unique(y)) - {0, 1} or
            37 not in tapes_by_seed):
        raise Stage05Error("Window sensitivity needs binary labels, a valid window, and seed 37")
    seeds = sorted(int(seed) for seed in tapes_by_seed)
    if len(seeds) != len(set(seeds)) or not seeds:
        raise Stage05Error("Query-window seed map is empty or duplicated")
    matrix = np.zeros((len(seeds), len(y) - window + 1), dtype=np.int32)
    for row_index, seed in enumerate(seeds):
        defect_queries = np.zeros(len(y), dtype=np.int32)
        for event in tapes_by_seed[seed]["events"]:
            position = int(event["position"])
            if not 0 <= position < len(y):
                raise Stage05Error("Query event lies outside evaluator label stream")
            defect_queries[position] = int(y[position] == 1)
        prefix = np.concatenate(([0], np.cumsum(defect_queries, dtype=np.int64)))
        matrix[row_index] = prefix[window:] - prefix[:-window]
    primary_row = seeds.index(37)
    output = []
    quantile_levels = (.05, .25, .5, .75, .95)
    for start in range(matrix.shape[1]):
        values = matrix[:, start].astype(np.float64)
        row = {"window_start": start, "window_end_exclusive": start + window,
               "seed37_queried_defects": int(matrix[primary_row, start]),
               "minimum_across_seeds": int(values.min()),
               "maximum_across_seeds": int(values.max()),
               "mean_across_seeds": float(values.mean()),
               "sample_sd_across_seeds": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
               "quantiles_linear": {str(q): float(np.quantile(values, q))
                                    for q in quantile_levels}}
        output.append(row)
    return output


def _score_distribution(values: np.ndarray) -> dict[str, Any]:
    return {"count": int(len(values)),
            "quantiles_linear": {str(q): float(np.quantile(values, q))
                                 for q in (0, .01, .05, .25, .5, .75, .95, .99, 1)}}


def _tie_count_cross(normals: np.ndarray, defects: np.ndarray) -> int:
    nvals, ncounts = np.unique(normals, return_counts=True)
    dvals, dcounts = np.unique(defects, return_counts=True)
    common, ni, di = np.intersect1d(nvals, dvals, return_indices=True)
    del common
    return int(np.sum(ncounts[ni] * dcounts[di]))


def _window_metrics(labels: np.ndarray, scores: np.ndarray,
                    threshold: float, window: int = 256) -> dict[str, Any]:
    n = len(labels)
    pred = scores >= threshold
    y = labels.astype(bool)
    p0 = np.concatenate(([0], np.cumsum((~y).astype(np.int64))))
    p1 = np.concatenate(([0], np.cumsum(y.astype(np.int64))))
    pfp = np.concatenate(([0], np.cumsum((pred & ~y).astype(np.int64))))
    pfn = np.concatenate(([0], np.cumsum((~pred & y).astype(np.int64))))
    ptp = np.concatenate(([0], np.cumsum((pred & y).astype(np.int64))))
    ptn = np.concatenate(([0], np.cumsum((~pred & ~y).astype(np.int64))))
    windows = []
    for start in range(max(0, n - window + 1)):
        end = start + window
        n0, n1 = int(p0[end] - p0[start]), int(p1[end] - p1[start])
        tn, fp = int(ptn[end] - ptn[start]), int(pfp[end] - pfp[start])
        tp, fn = int(ptp[end] - ptp[start]), int(pfn[end] - pfn[start])
        fpr = fp / n0 if n0 else None
        fnr = fn / n1 if n1 else None
        ba = 1 - .5 * (fpr + fnr) if fpr is not None and fnr is not None else None
        windows.append({"start": start, "end_exclusive": end,
                        "n_normal": n0, "n_defect": n1,
                        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                        "fpr": fpr, "fnr": fnr, "balanced_accuracy": ba})
    defined = [row for row in windows if row["balanced_accuracy"] is not None]
    worst = min(defined, key=lambda row: (row["balanced_accuracy"], row["start"])) if defined else None
    run = longest = 0
    for is_fn in (~pred & y):
        run = run + 1 if is_fn else 0
        longest = max(longest, run)
    return {
        "window_size": window, "stride": 1,
        "window_count": len(windows),
        "undefined_single_class_windows": sum(row["balanced_accuracy"] is None
                                                for row in windows),
        "worst_defined_window": worst,
        "longest_consecutive_false_negative_run": int(longest),
        "all_windows": windows,
    }


def static_score_metrics(labels: Sequence[int], scores: Sequence[float],
                         c64_scores: Sequence[float], *, support_seed: int,
                         class_name: str) -> dict[str, Any]:
    """Offline metrics after an immutable, checksummed native score tape exists."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    c64 = np.asarray(c64_scores, dtype=np.float64)
    if len(y) != len(s) or set(np.unique(y)) != {0, 1}:
        raise Stage05Error("Static scorer metrics require aligned two-class test data")
    if len(c64) != 64 or not np.isfinite(s).all() or not np.isfinite(c64).all():
        raise Stage05Error("Require 64 finite C64 and finite full-stream scores")
    rank = math.ceil(.90 * (len(c64) - 1))
    theta0 = float(np.sort(c64)[rank])
    frozen = _confusion(y, s, theta0)
    oracle = retrospective_threshold_oracle(y, s, theta0)
    normals, defects = s[y == 0], s[y == 1]
    cross_ties = _tie_count_cross(normals, defects)
    overlap_low, overlap_high = max(float(normals.min()), float(defects.min())), min(
        float(normals.max()), float(defects.max()))
    from sklearn.metrics import roc_curve
    fpr_curve, tpr_curve, _ = roc_curve(y, s, drop_intermediate=False)
    frozen_windows = _window_metrics(y, s, theta0)
    frozen_ba = frozen["balanced_accuracy"]
    headroom = float(oracle["balanced_accuracy"] - frozen_ba)
    return {
        "class": class_name, "support_seed": int(support_seed),
        "stream_images": len(s), "normal_count": int(np.count_nonzero(y == 0)),
        "defect_count": int(np.count_nonzero(y == 1)),
        "defect_prevalence": float(np.mean(y)),
        "auroc": float(roc_auc_score(y, s)),
        "auprc_average_precision": float(average_precision_score(y, s)),
        "normal_scores": _score_distribution(normals),
        "defect_scores": _score_distribution(defects),
        "score_overlap": {
            "ranges_intersect": overlap_low <= overlap_high,
            "overlap_interval": ([overlap_low, overlap_high]
                                 if overlap_low <= overlap_high else None),
            "cross_class_equal_score_pairs": cross_ties,
            "cross_class_pair_count": int(len(normals) * len(defects)),
            "cross_class_tie_fraction": cross_ties / (len(normals) * len(defects)),
            "perfect_rank_separation_normal_strictly_below_defect":
                bool(float(normals.max()) < float(defects.min())),
            "roc_curve_threshold_count": int(len(fpr_curve)),
        },
        "theta0": {"value": theta0, "source": "64 fixed C64 normals at M0",
                   "quantile": 0.90, "rank_zero_based": rank,
                   "rule": "D2 higher order statistic"},
        "frozen_threshold_metrics": frozen,
        "retrospective_oracle": oracle,
        "oracle_minus_frozen_ba": headroom,
        "signed_threshold_movement_from_theta0": oracle["threshold"] - theta0,
        "absolute_threshold_movement_from_theta0": abs(oracle["threshold"] - theta0),
        "temporal_frozen_threshold": frozen_windows,
        "FROZEN_STATIC_NEAR_CEILING": bool(frozen_ba >= .95 and headroom <= .01),
        "near_ceiling_rule_is_exploratory_only": True,
        "oracle_not_online_unbiased_or_controller_available": True,
    }


def summarize_support_seed_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    classes = sorted({str(row["class"]) for row in rows})
    output = {"classes": {}, "overall": {"configuration_count": len(rows),
                                              "near_ceiling_count": sum(
                                                  bool(row["FROZEN_STATIC_NEAR_CEILING"])
                                                  for row in rows)}}
    for class_name in classes:
        subset = [row for row in rows if row["class"] == class_name]
        fields = ("auroc", "auprc_average_precision", "frozen_threshold_metrics",
                  "oracle_minus_frozen_ba")
        metrics = {}
        for field in fields:
            if field == "frozen_threshold_metrics":
                values = [float(row[field]["balanced_accuracy"]) for row in subset]
                name = "frozen_ba"
            else:
                values = [float(row[field]) for row in subset]
                name = field
            array = np.asarray(values, dtype=np.float64)
            metrics[name] = {"mean": float(array.mean()),
                             "sample_standard_deviation": float(array.std(ddof=1)),
                             "minimum": float(array.min()), "maximum": float(array.max()),
                             "seed_values": {str(row["support_seed"]): value
                                            for row, value in zip(subset, values)}}
        oracle = np.asarray([row["retrospective_oracle"]["balanced_accuracy"]
                             for row in subset], dtype=np.float64)
        metrics["oracle_ba"] = {"mean": float(oracle.mean()),
                                "sample_standard_deviation": float(oracle.std(ddof=1)),
                                "minimum": float(oracle.min()),
                                "maximum": float(oracle.max())}
        output["classes"][class_name] = {
            "support_seed_count": len(subset), "metrics": metrics,
            "near_ceiling_seed_count": sum(
                bool(row["FROZEN_STATIC_NEAR_CEILING"]) for row in subset),
            "near_ceiling_seed_ids": [int(row["support_seed"]) for row in subset
                                      if row["FROZEN_STATIC_NEAR_CEILING"]],
        }
    return output


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(canonical_json(value) + "\n", encoding="utf-8")
