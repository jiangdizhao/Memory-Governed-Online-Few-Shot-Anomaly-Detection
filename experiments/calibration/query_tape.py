"""New label-blind, group-limited Q-A-TS and Q-B query tapes.

The archived ``experiments.risk_safe.query_protocol`` Q-A remains unchanged
and is not imported here. Selection uses public stream metadata only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Sequence

from .profiles import QueryProtocolConfig, canonical_json
from .protocol import TapeRecord


class QueryTapeError(RuntimeError):
    pass


def half_up(value: float) -> int:
    if value < 0:
        raise ValueError("half_up expects a nonnegative value")
    return int(value + 0.5)


def _hash_int(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16)


def _stream_digest(records: Sequence[TapeRecord]) -> str:
    payload = [
        {"position": r.position, "sample_key": r.official_sample_key,
         "camera_id": r.camera_id, "relative_id": r.relative_id,
         "timestamp": r.timestamp, "phase": r.phase}
        for r in records
    ]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QueryEvent:
    position: int
    official_sample_key: str
    camera_id: int
    relative_id: str
    phase: int
    time_bin: int


@dataclass(frozen=True)
class QueryTape:
    schema: str
    policy: str
    seed: int
    n_images: int
    n_groups: int
    requested_budget: int
    events: tuple[QueryEvent, ...]
    stream_digest: str
    digest: str
    bin_edges: tuple[int, ...]
    bin_quotas: tuple[int, ...]
    early_budget: int = 0

    @property
    def positions(self) -> frozenset[int]:
        return frozenset(event.position for event in self.events)

    @property
    def queried_groups(self) -> frozenset[str]:
        return frozenset(event.official_sample_key for event in self.events)


def _validate_stream(records: Sequence[TapeRecord]) -> None:
    if not records:
        raise QueryTapeError("Cannot build a tape for an empty stream")
    positions = [record.position for record in records]
    if positions != list(range(len(records))):
        raise QueryTapeError("TapeRecord positions must be contiguous and zero-based")
    ids = [record.relative_id for record in records]
    if len(set(ids)) != len(ids):
        raise QueryTapeError("Image relative ids must be unique")


def _canonical_by_group(records: Sequence[TapeRecord], seed: int
                        ) -> dict[str, TapeRecord]:
    """Choose one camera view per official group using metadata and a fixed hash."""
    groups: dict[str, list[TapeRecord]] = {}
    for record in records:
        groups.setdefault(record.official_sample_key, []).append(record)
    chosen = {}
    for key, views in groups.items():
        chosen[key] = min(
            views,
            key=lambda r: (_hash_int(seed, "camera-view", r.position,
                                    r.camera_id), r.position),
        )
    return chosen


def _bin_edges(n_images: int, bins: int) -> tuple[int, ...]:
    return tuple((index * n_images) // bins for index in range(bins + 1))


def _assign_bin(position: int, edges: Sequence[int]) -> int:
    # Fixed integer edges, no floating point boundary ambiguity.
    for index in range(len(edges) - 1):
        if edges[index] <= position < edges[index + 1]:
            return index
    return len(edges) - 2


def _largest_remainder(capacities: Sequence[int], budget: int) -> list[int]:
    """Allocate a budget in proportion to available groups, with stable ties."""
    total = sum(capacities)
    if budget < 0 or budget > total:
        raise QueryTapeError(f"Requested {budget} queries for {total} available groups")
    if budget == 0:
        return [0] * len(capacities)
    if total == 0:
        raise QueryTapeError("No eligible official sample groups")
    raw = [budget * capacity / total for capacity in capacities]
    quotas = [min(capacities[i], int(value)) for i, value in enumerate(raw)]
    remaining = budget - sum(quotas)
    order = sorted(range(len(capacities)),
                   key=lambda i: (-(raw[i] - int(raw[i])), i))
    while remaining:
        progressed = False
        for index in order:
            if quotas[index] < capacities[index]:
                quotas[index] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise QueryTapeError("Could not allocate the exact group-limited budget")
    return quotas


def _select_by_bins(canonical: dict[str, TapeRecord], edges: Sequence[int],
                    quotas: Sequence[int], seed: int, policy: str,
                    region: str = "all") -> list[tuple[int, TapeRecord]]:
    by_bin: list[list[TapeRecord]] = [[] for _ in range(len(edges) - 1)]
    for record in canonical.values():
        by_bin[_assign_bin(record.position, edges)].append(record)
    selected = []
    for bin_index, (items, quota) in enumerate(zip(by_bin, quotas)):
        if quota > len(items):
            raise QueryTapeError("Bin quota exceeds its label-blind group availability")
        ordered = sorted(items, key=lambda r: (
            _hash_int(seed, policy, region, bin_index, r.position, r.camera_id),
            r.position))
        selected.extend((bin_index, record) for record in ordered[:quota])
    return selected


def _build(records: Sequence[TapeRecord], policy: str, config: QueryProtocolConfig,
           selected: Sequence[tuple[int, TapeRecord]], budget: int,
           edges: Sequence[int], quotas: Sequence[int], early_budget: int = 0
           ) -> QueryTape:
    events = tuple(sorted((QueryEvent(
        position=record.position,
        official_sample_key=record.official_sample_key,
        camera_id=record.camera_id,
        relative_id=record.relative_id,
        phase=record.phase,
        time_bin=int(bin_index),
    ) for bin_index, record in selected), key=lambda event: event.position))
    if len(events) != budget:
        raise QueryTapeError("Selected image count differs from the registered budget")
    if len({event.official_sample_key for event in events}) != len(events):
        raise QueryTapeError("More than one camera image was selected for a sample group")
    payload = {
        "schema": "calibration.query_tape.v1",
        "policy": policy,
        "config": asdict(config),
        "n_images": len(records),
        "n_groups": len({record.official_sample_key for record in records}),
        "requested_budget": budget,
        "early_budget": early_budget,
        "stream_digest": _stream_digest(records),
        "bin_edges": list(edges),
        "bin_quotas": list(quotas),
        "events": [asdict(event) for event in events],
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return QueryTape(
        schema=payload["schema"], policy=policy, seed=config.seed,
        n_images=len(records), n_groups=payload["n_groups"],
        requested_budget=budget, events=events,
        stream_digest=payload["stream_digest"], digest=digest,
        bin_edges=tuple(edges), bin_quotas=tuple(quotas),
        early_budget=early_budget)


def build_qa_ts(records: Sequence[TapeRecord],
                config: QueryProtocolConfig = QueryProtocolConfig()) -> QueryTape:
    """Build 3%-of-images Q-A-TS, stratified across 16 fixed time bins."""
    _validate_stream(records)
    budget = half_up(config.fraction * len(records))
    canonical = _canonical_by_group(records, config.seed)
    if budget > len(canonical):
        raise QueryTapeError(
            f"Q-A-TS infeasible: B={budget} exceeds {len(canonical)} unique groups")
    edges = _bin_edges(len(records), config.bins)
    capacities = [0] * config.bins
    for record in canonical.values():
        capacities[_assign_bin(record.position, edges)] += 1
    quotas = _largest_remainder(capacities, budget)
    selected = _select_by_bins(canonical, edges, quotas, config.seed, "Q-A-TS")
    return _build(records, "Q-A-TS", config, selected, budget, edges, quotas)


def build_qb(records: Sequence[TapeRecord],
             config: QueryProtocolConfig = QueryProtocolConfig()) -> QueryTape:
    """Build Q-B with a fixed 20% early budget in the first 400 images."""
    _validate_stream(records)
    budget = half_up(config.fraction * len(records))
    canonical = _canonical_by_group(records, config.seed)
    if budget > len(canonical):
        raise QueryTapeError(f"Q-B infeasible: B={budget} exceeds unique groups")
    early_end = min(config.early_observations, len(records))
    early_budget = half_up(config.q_b_early_fraction * budget)
    early = {key: record for key, record in canonical.items()
             if record.position < early_end}
    later = {key: record for key, record in canonical.items()
             if record.position >= early_end}
    later_budget = budget - early_budget
    if early_budget > len(early) or later_budget > len(later):
        raise QueryTapeError(
            "Q-B infeasible under its registered 20%/80% split; refusing redistribution")
    edges = _bin_edges(len(records), config.bins)
    early_caps = [0] * config.bins
    later_caps = [0] * config.bins
    for record in early.values():
        early_caps[_assign_bin(record.position, edges)] += 1
    for record in later.values():
        later_caps[_assign_bin(record.position, edges)] += 1
    early_quotas = _largest_remainder(early_caps, early_budget)
    later_quotas = _largest_remainder(later_caps, later_budget)
    selected = _select_by_bins(early, edges, early_quotas, config.seed,
                               "Q-B", region="early")
    selected.extend(_select_by_bins(later, edges, later_quotas, config.seed,
                                    "Q-B", region="later"))
    quotas = [early_quotas[i] + later_quotas[i] for i in range(config.bins)]
    return _build(records, "Q-B", config, selected, budget, edges, quotas,
                  early_budget=early_budget)


def verify_tape(tape: QueryTape, records: Sequence[TapeRecord],
                config: QueryProtocolConfig = QueryProtocolConfig()) -> None:
    _validate_stream(records)
    if tape.stream_digest != _stream_digest(records):
        raise QueryTapeError("Tape stream digest does not match the supplied metadata")
    if tape.n_images != len(records) or tape.requested_budget != len(tape.events):
        raise QueryTapeError("Tape dimensions or budget are inconsistent")
    by_position = {record.position: record for record in records}
    for event in tape.events:
        record = by_position.get(event.position)
        if record is None or (record.official_sample_key, record.camera_id,
                              record.relative_id) != (
                event.official_sample_key, event.camera_id, event.relative_id):
            raise QueryTapeError("Tape event does not match the source stream record")
    expected = (build_qa_ts(records, config) if tape.policy == "Q-A-TS"
                else build_qb(records, config) if tape.policy == "Q-B" else None)
    if expected is None or expected.digest != tape.digest:
        raise QueryTapeError("Tape policy/digest mismatch")


def query_fraction_report(tape: QueryTape) -> dict[str, float | int]:
    return {
        "n_images": tape.n_images,
        "n_groups": tape.n_groups,
        "n_queried_images": len(tape.events),
        "n_queried_groups": len(tape.queried_groups),
        "query_fraction_images": (len(tape.events) / tape.n_images
                                   if tape.n_images else 0.0),
        "query_fraction_groups": (len(tape.queried_groups) / tape.n_groups
                                   if tape.n_groups else 0.0),
        "early_queries": sum(event.position < 400 for event in tape.events),
    }
