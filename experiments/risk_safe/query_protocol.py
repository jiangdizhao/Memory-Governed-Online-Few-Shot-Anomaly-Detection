"""Q-A shared label-blind random query tape plus the causal replay contract.

D1-A-R1 is metadata-only: no features, no model, no GPU.

Scientific invariants enforced here
-----------------------------------
1. **Structural label blindness.**  The tape builder accepts only
   :class:`StreamImage` objects, which have no label field.  Query selection
   therefore cannot inspect ``is_good``, ``anomaly_class``, defect masks,
   evaluation-probe labels, scores, or adaptive model state -- not even by
   accident.  Labels enter this module only through
   :func:`label_diagnostics` (offline accounting after selection) and
   :func:`replay_shared_tape` (disclosure after the query decision).
2. **Nested budgets.**  For one (class, stream, query seed) a single fixed
   random priority permutation of official samples is generated.  Budget ``B``
   selects the first ``B`` samples of that *same* permutation and the *same*
   per-sample camera image, so ``tape(B1)`` is a subset of ``tape(B2)`` for
   ``B1 < B2``.  Each budget must still be replayed independently from the same
   initial memory state; nesting is a sampling device, not permission to reuse
   learned state.
3. **One query per official sample.**  Every official sample contributes at
   most one image-level label per stream, so ``N_queries == N_unique_samples``.
4. **Causal ordering.**  The stream is a chronologically ordered sequence of
   original camera observations.  Predict first, disclose afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
#  Pre-registered Q-A constants
# ---------------------------------------------------------------------------

#: Feasibility candidates for D1-A.  These are NOT approved experimental
#: budgets; Section 2.3 of the plan requires the first-round grid to be
#: registered before any new algorithm results are read.  79 is the traceable
#: V0.2 reference (3.01% of 2,625 image observations), not a preferred value.
BUDGET_CANDIDATES: Tuple[int, ...] = (8, 16, 32, 64, 79, 96, 128)

#: Label-blind tape seed.  Matches the historical V0.2 tape seed (37) so the
#: external randomisation device is traceable; it was not tuned on outcomes.
QUERY_SEED: int = 37

QUERY_POLICY: str = "Q-A"
TAPE_SCHEMA: str = "risk_safe.q_a_tape.v2"

#: Identity contract of the grouping unit.  A tape produced under one identity
#: version must never be replayed or compared under another.
IDENTITY_VERSION: str = "realid-offical-sample-key-v2"
GROUPING_UNIT: str = "official_sample_key"

#: One image-level human label per official sample key per stream
#: (plan Section 2.1(6)).
MAX_QUERIES_PER_SAMPLE: int = 1

VALID_CAMERA_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5)


class QueryProtocolError(RuntimeError):
    """Any violated Q-A protocol invariant."""


class QueryBudgetInfeasible(QueryProtocolError):
    """Requested absolute label budget cannot be honoured without shrinking it."""


# ---------------------------------------------------------------------------
#  Deterministic seeding
# ---------------------------------------------------------------------------


def derived_seed(seed: int, *identity: Any) -> int:
    """Independent deterministic sub-seed derived from an explicit identity.

    Mirrors ``p0_protocol_audit.deterministic_subseed`` (SHA-256 prefix) so the
    same convention applies across the repository.  The seed of one draw never
    depends on the size of another draw, which is what makes nested budgets
    possible without regenerating an unrelated sample.
    """
    key = "|".join([str(seed), *(str(v) for v in identity)])
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)


def canonical_json(payload: Any) -> str:
    """Deterministic JSON used for every manifest digest in D1-A."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def sha256_of(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
#  Label-blind stream
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamImage:
    """One online image observation -- deliberately carries **no label**.

    ``sample_key`` is the physical-object grouping unit of the protocol: the
    official Real-IAD sample key (``<category>_<NNNN>_<CONDITION>``).
    ``rel_path`` is the class-relative path (portable, used in digests);
    ``path`` is the path recorded by the dataset index (used for diagnostics).
    """

    position: int
    sample_key: str
    path: str
    rel_path: str
    camera_id: int
    timestamp: str
    phase: int

    def __post_init__(self) -> None:
        if int(self.position) < 1:
            raise QueryProtocolError(f"Stream position must be >= 1: {self.position}")
        if not str(self.sample_key):
            raise QueryProtocolError("Empty official sample key")
        if int(self.camera_id) not in VALID_CAMERA_IDS:
            raise QueryProtocolError(f"Invalid camera id: {self.camera_id}")
        if len(str(self.timestamp)) != 14 or not str(self.timestamp).isdigit():
            raise QueryProtocolError(f"Invalid capture timestamp: {self.timestamp}")
        if int(self.phase) < 1:
            raise QueryProtocolError(f"Invalid phase index: {self.phase}")

    def blind_key(self) -> Dict[str, Any]:
        """Only the fields a label-blind selector may inspect."""
        return {
            "position": int(self.position),
            "sample_key": str(self.sample_key),
            "rel_path": str(self.rel_path),
            "camera_id": int(self.camera_id),
            "timestamp": str(self.timestamp),
            "phase": int(self.phase),
        }


@dataclass(frozen=True)
class PublicObservation:
    """Minimal image observation visible to a query policy before feedback.

    Real-IAD official sample keys and paths contain OK/NG tokens. Passing a
    StreamImage directly to predict() would therefore reveal a label without a
    query. The trusted data adapter may resolve observation_handle to image
    pixels/features; the controller itself must NOT see raw paths, labels, or
    sample keys. This deliberately does not provide physical-specimen identity.
    """

    position: int
    observation_handle: str
    camera_id: int
    phase: int

    @classmethod
    def from_stream(cls, image: StreamImage) -> "PublicObservation":
        return cls(position=int(image.position),
                   observation_handle=f"observation:{image.position}",
                   camera_id=int(image.camera_id), phase=int(image.phase))


def build_stream(records_by_phase: Sequence[Sequence[Any]]) -> Tuple[StreamImage, ...]:
    """Concatenate phases, ordering each block chronologically by capture metadata.

    The stream is a **phase-blocked** sequence, not one global capture timeline.
    Phases I/II/III hold physically disjoint object cohorts captured in the same
    wall-clock window, so their raw timestamps interleave by construction.  Each
    block is strictly ordered by ``(timestamp, camera_id, path)`` and the blocks
    are concatenated in the pre-registered order (I -> II -> III); global
    timestamps are therefore *not* monotone across block boundaries, and this is
    recorded as a known limitation rather than hidden.

    Records must expose ``sample_key``, ``path``, ``rel_path``, ``camera_id`` and
    ``timestamp``; any label attribute is ignored and never read.
    """
    images: List[StreamImage] = []
    for phase, records in enumerate(records_by_phase, start=1):
        ordered = sorted(records, key=lambda r: (str(r.timestamp), int(r.camera_id),
                                                 str(r.path)))
        for record in ordered:
            images.append(StreamImage(
                position=len(images) + 1,
                sample_key=str(record.sample_key),
                path=str(record.path),
                rel_path=str(record.rel_path),
                camera_id=int(record.camera_id),
                timestamp=str(record.timestamp),
                phase=phase,
            ))
    stream = tuple(images)
    validate_stream(stream)
    return stream


def validate_stream(stream: Sequence[StreamImage]) -> None:
    """Fail closed on malformed, duplicated or out-of-order stream blocks.

    Ordering rules: positions must be ``1..N``; paths and class-relative paths
    must be unique; phase indices must be non-decreasing; and inside one phase
    consecutive observations must increase in ``(timestamp, camera_id, path)``.
    """
    if not stream:
        raise QueryProtocolError("Empty stream")
    expected = list(range(1, len(stream) + 1))
    if [int(i.position) for i in stream] != expected:
        raise QueryProtocolError("Stream positions must be 1..N in sequence")
    paths = [i.path for i in stream]
    if len(set(paths)) != len(paths):
        raise QueryProtocolError("Duplicate image path in stream")
    rel = [i.rel_path for i in stream]
    if len(set(rel)) != len(rel):
        raise QueryProtocolError("Duplicate class-relative path in stream")
    for previous, current in zip(stream, stream[1:]):
        if int(current.phase) < int(previous.phase):
            raise QueryProtocolError("Phase indices must be non-decreasing")
        if int(current.phase) != int(previous.phase):
            continue
        if (current.timestamp, current.camera_id, current.path) <= (
                previous.timestamp, previous.camera_id, previous.path):
            raise QueryProtocolError(
                f"Phase {current.phase} is not chronological at positions "
                f"{previous.position}->{current.position}")


def stream_digest(stream: Sequence[StreamImage]) -> str:
    return sha256_of([i.blind_key() for i in stream])


# ---------------------------------------------------------------------------
#  Deterministic, label-blind query tapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryEvent:
    """One pre-registered human-query opportunity (label still hidden)."""

    position: int
    sample_key: str
    path: str
    rel_path: str
    camera_id: int
    phase: int

    def as_manifest(self) -> Dict[str, Any]:
        return {
            "position": int(self.position),
            "sample_key": str(self.sample_key),
            "image_path": str(self.rel_path),
            "camera_id": int(self.camera_id),
            "phase": int(self.phase),
        }


@dataclass(frozen=True)
class QueryTape:
    """A complete, digest-addressed Q-A tape for one class/stream/seed/budget."""

    class_name: str
    seed: int
    budget: int
    n_images: int
    n_samples: int
    events: Tuple[QueryEvent, ...]
    digest: str
    stream_digest: str
    priority_order: Tuple[str, ...]

    @property
    def positions(self) -> Tuple[int, ...]:
        return tuple(int(e.position) for e in self.events)

    @property
    def sample_keys(self) -> Tuple[str, ...]:
        return tuple(str(e.sample_key) for e in self.events)

    @property
    def rel_paths(self) -> Tuple[str, ...]:
        return tuple(str(e.rel_path) for e in self.events)

    @property
    def camera_ids(self) -> Tuple[int, ...]:
        return tuple(int(e.camera_id) for e in self.events)

    def manifest(self) -> Dict[str, Any]:
        """Full budget accounting required by the D1-A-R1 brief."""
        n_queries = len(self.events)
        n_unique = len(set(self.sample_keys))
        return {
            "schema": TAPE_SCHEMA,
            "identity_version": IDENTITY_VERSION,
            "grouping_unit": GROUPING_UNIT,
            "query_policy": QUERY_POLICY,
            "class_name": self.class_name,
            "query_seed": int(self.seed),
            "budget_requested": int(self.budget),
            "N_images": int(self.n_images),
            "N_samples": int(self.n_samples),
            "N_queries_selected": n_queries,
            "N_unique_queried_samples": n_unique,
            "image_query_fraction": round(n_queries / self.n_images, 6),
            "unique_sample_audit_fraction": round(n_unique / self.n_samples, 6),
            "query_positions": list(self.positions),
            "selected_image_paths": list(self.rel_paths),
            "selected_sample_keys": list(self.sample_keys),
            "selected_camera_ids": list(self.camera_ids),
            "query_manifest_sha256": self.digest,
            "stream_sha256": self.stream_digest,
            "events": [e.as_manifest() for e in self.events],
        }


def _samples_in_stream(stream: Sequence[StreamImage]) -> Tuple[str, ...]:
    return tuple(sorted({i.sample_key for i in stream}))


def sample_priority(stream: Sequence[StreamImage], seed: int,
                    class_name: str) -> Tuple[str, ...]:
    """One fixed random official-sample permutation for this stream and seed.

    Selection is over *samples*, not images: the label budget is an absolute
    number of human image-level verifications and each verification is charged
    to one distinct official sample key (plan Section 2.1(6)).
    """
    samples = list(_samples_in_stream(stream))
    random.Random(derived_seed(seed, QUERY_POLICY, class_name,
                               "sample-priority", stream_digest(stream))
                  ).shuffle(samples)
    return tuple(samples)


def camera_choice_map(stream: Sequence[StreamImage], seed: int,
                      class_name: str) -> Dict[str, StreamImage]:
    """One fixed camera observation per official sample, independent of budget.

    Each sample's draw uses its own RNG identity ``(seed, policy, class,
    sample)``, so the chosen view depends neither on the budget, nor on the
    order in which other samples were drawn, nor on any label.
    """
    candidates: Dict[str, List[StreamImage]] = {}
    for image in stream:
        candidates.setdefault(image.sample_key, []).append(image)
    chosen: Dict[str, StreamImage] = {}
    for sample_key in sorted(candidates):
        views = sorted(candidates[sample_key],
                       key=lambda i: (i.timestamp, i.camera_id, i.path))
        rng = random.Random(derived_seed(seed, QUERY_POLICY, class_name,
                                         "camera", sample_key))
        chosen[sample_key] = views[rng.randrange(len(views))]
    return chosen


def build_query_tape(stream: Sequence[StreamImage], budget: int, seed: int,
                     class_name: str = "",
                     priority: Optional[Sequence[str]] = None,
                     choices: Optional[Mapping[str, StreamImage]] = None
                     ) -> QueryTape:
    """Select the first ``budget`` samples of the fixed priority order.

    Raises :class:`QueryBudgetInfeasible` when the absolute budget exceeds the
    number of eligible official samples: the budget is never silently reduced.
    """
    validate_stream(stream)
    if not isinstance(budget, int) or isinstance(budget, bool):
        raise QueryBudgetInfeasible(f"Budget must be an integer, got {budget!r}")
    if budget <= 0:
        raise QueryBudgetInfeasible(f"Budget must be positive, got {budget}")
    n_images = len(stream)
    samples = _samples_in_stream(stream)
    n_samples = len(samples)
    if budget > n_samples:
        raise QueryBudgetInfeasible(
            f"{class_name or '<stream>'}: requested B={budget} exceeds the "
            f"{n_samples} eligible official samples; refuse to reduce")

    order = tuple(priority) if priority is not None else sample_priority(
        stream, seed, class_name)
    if set(order) != set(samples) or len(order) != n_samples:
        raise QueryProtocolError("Priority order does not match stream samples")
    selection_map = dict(choices) if choices is not None else camera_choice_map(
        stream, seed, class_name)
    if set(selection_map) != set(samples):
        raise QueryProtocolError("Camera-choice map does not match stream samples")

    selected: List[StreamImage] = []
    for sample_key in order[:budget]:
        image = selection_map[sample_key]
        if image.sample_key != sample_key:
            raise QueryProtocolError("Camera choice inconsistent with sample")
        selected.append(image)
    selected.sort(key=lambda i: i.position)

    sample_keys = [i.sample_key for i in selected]
    if len(set(sample_keys)) != len(sample_keys):
        raise QueryProtocolError("More than one query for one official sample")
    if len({i.position for i in selected}) != len(selected):
        raise QueryProtocolError("Duplicate query position")

    events = tuple(
        QueryEvent(position=int(i.position), sample_key=str(i.sample_key),
                   path=str(i.path), rel_path=str(i.rel_path),
                   camera_id=int(i.camera_id), phase=int(i.phase))
        for i in selected)

    digest = sha256_of({
        "schema": TAPE_SCHEMA,
        "identity_version": IDENTITY_VERSION,
        "grouping_unit": GROUPING_UNIT,
        "query_policy": QUERY_POLICY,
        "class_name": class_name,
        "query_seed": int(seed),
        "budget_requested": int(budget),
        "N_images": n_images,
        "N_samples": n_samples,
        "events": [e.as_manifest() for e in events],
    })
    return QueryTape(class_name=class_name, seed=int(seed), budget=int(budget),
                     n_images=n_images, n_samples=n_samples, events=events,
                     digest=digest, stream_digest=stream_digest(stream),
                     priority_order=order)


def assert_nested(smaller: QueryTape, larger: QueryTape) -> None:
    """Fail closed unless ``smaller`` is a subset of ``larger`` event by event.

    Both the query *sample* and the selected *camera image* must be identical:
    a nested budget may only add events, never re-draw the view of a sample it
    already shared with a smaller budget.
    """
    if smaller.class_name != larger.class_name or smaller.seed != larger.seed:
        raise QueryProtocolError("Cannot compare tapes of different class/seed")
    if smaller.stream_digest != larger.stream_digest:
        raise QueryProtocolError("Cannot compare tapes of different streams")
    if smaller.budget > larger.budget:
        raise QueryProtocolError("Nesting requires the smaller budget first")
    large = {e.sample_key: e for e in larger.events}
    for event in smaller.events:
        twin = large.get(event.sample_key)
        if twin is None:
            raise QueryProtocolError(
                f"Budget {smaller.budget} query sample {event.sample_key} is "
                f"absent from budget {larger.budget}")
        if (twin.position, twin.rel_path, twin.camera_id) != (
                event.position, event.rel_path, event.camera_id):
            raise QueryProtocolError(
                f"Selected image for sample {event.sample_key} changed with budget")


def build_nested_tapes(stream: Sequence[StreamImage], budgets: Sequence[int],
                       seed: int, class_name: str = ""
                       ) -> Dict[int, QueryTape]:
    """Build every budget from ONE priority order and ONE camera-choice map."""
    ordered_budgets = sorted({int(b) for b in budgets})
    priority = sample_priority(stream, seed, class_name)
    choices = camera_choice_map(stream, seed, class_name)
    tapes = {b: build_query_tape(stream, b, seed, class_name,
                                 priority=priority, choices=choices)
             for b in ordered_budgets}
    for small, large in zip(ordered_budgets, ordered_budgets[1:]):
        assert_nested(tapes[small], tapes[large])
    return tapes


def tape_nesting_report(tapes: Mapping[int, QueryTape]) -> Dict[str, Any]:
    """Numeric proof of prefix nesting for one class/stream/seed."""
    budgets = sorted(int(b) for b in tapes)
    rows = []
    for small, large in zip(budgets, budgets[1:]):
        a, b = tapes[small], tapes[large]
        assert_nested(a, b)
        large_samples = {e.sample_key: e for e in b.events}
        rows.append({
            "smaller_budget": small,
            "larger_budget": large,
            "added_events": len(b.events) - len(a.events),
            "subset": set(a.sample_keys) <= set(large_samples),
            "identical_selected_images": all(
                (large_samples[e.sample_key].position,
                 large_samples[e.sample_key].rel_path,
                 large_samples[e.sample_key].camera_id) ==
                (e.position, e.rel_path, e.camera_id) for e in a.events),
        })
    return {"budgets": budgets, "pairwise_prefix_checks": rows,
            "all_nested": all(r["subset"] and r["identical_selected_images"]
                              for r in rows)}


# ---------------------------------------------------------------------------
#  Offline diagnostics (labels AFTER selection only)
# ---------------------------------------------------------------------------


def label_diagnostics(stream: Sequence[StreamImage], tape: QueryTape,
                      labels: Mapping[str, bool]) -> Dict[str, Any]:
    """Post-hoc label composition of an already-frozen tape.

    This is an *offline diagnostic*: the plan permits offline ground truth for
    metadata checks and evaluation-only probe composition, but it must never
    influence which queries are selected.  The tape is a required argument, so
    the composition cannot even be computed before the tape exists.
    """
    del stream  # the tape fully determines the queried observations
    missing = [e.path for e in tape.events if e.path not in labels]
    if missing:
        raise QueryProtocolError(f"Missing offline labels for {len(missing)} queries")
    normal = [e for e in tape.events if bool(labels[e.path])]
    defect = [e for e in tape.events if not bool(labels[e.path])]
    per_phase: Dict[str, Dict[str, int]] = {}
    for event in tape.events:
        slot = per_phase.setdefault(str(event.phase),
                                    {"queries": 0, "normal": 0, "defect": 0})
        slot["queries"] += 1
        slot["normal" if bool(labels[event.path]) else "defect"] += 1
    return {
        "diagnostic_scope": "offline_only_never_used_for_query_selection",
        "N_normal_labels": len(normal),
        "N_defect_labels": len(defect),
        "defect_label_fraction": (round(len(defect) / len(tape.events), 6)
                                  if tape.events else 0.0),
        "queries_by_phase": per_phase,
        "queried_sample_labels": {e.sample_key: int(not bool(labels[e.path]))
                                  for e in tape.events},
    }


# ---------------------------------------------------------------------------
#  Causal replay contract (synthetic, CPU-only, no Admission algorithm)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Disclosure:
    position: int
    sample_key: str
    rel_path: str
    label: int  # 0 = verified normal, 1 = verified defect


@dataclass(frozen=True)
class ReplayResult:
    disclosures: Tuple[Disclosure, ...]
    predictions: Mapping[str, Tuple[Tuple[int, float], ...]]
    deliveries: Mapping[str, Tuple[Tuple[int, str, str, int], ...]]

    def delivery_digest(self, method: str) -> str:
        return sha256_of([list(row) for row in self.deliveries[method]])

    def parity_report(self) -> Dict[str, Any]:
        names = sorted(self.deliveries)
        digests = {name: self.delivery_digest(name) for name in names}
        positions = {name: tuple(row[0] for row in self.deliveries[name])
                     for name in names}
        return {
            "methods": names,
            "N_disclosures": len(self.disclosures),
            "delivery_sha256": digests,
            "identical_feedback_events": len(set(digests.values())) <= 1,
            "identical_query_positions": len(set(positions.values())) <= 1,
            "identical_feedback_order":
                len(set(tuple(self.deliveries[name]) for name in names)) <= 1,
        }


def replay_shared_tape(stream: Sequence[StreamImage], tape: QueryTape,
                       labels: Mapping[str, bool],
                       methods: Mapping[str, Any]) -> ReplayResult:
    """Replay one stream and deliver identical feedback to every method.

    Per time step ``t``:

    1. the image arrives; each method predicts using a sanitized observation;
    2. the pre-registered tape (not the method) decides whether ``q_t = 1``;
    3. only then is the *selected image* label revealed;
    4. the revealed label is delivered to every method in the same order;
    5. that label becomes historical evidence for ``t+1`` only.

    Methods expose ``predict(step, PublicObservation) -> float`` (never a
    label-bearing Real-IAD path) and ``observe_feedback(step, StreamImage,
    label) -> None`` ONLY after the query has legally disclosed that label.
    Raw image lookup belongs to a trusted data/feature adapter, not the query
    controller.  No Admission
    algorithm is implemented here and none is required for D1-A.
    """
    validate_stream(stream)
    if not methods:
        raise QueryProtocolError("At least one method is required")
    if tape.n_images != len(stream) or tape.stream_digest != stream_digest(stream):
        raise QueryProtocolError("Tape does not belong to this stream")
    by_position = {int(e.position): e for e in tape.events}
    available = {int(i.position): i for i in stream}
    unknown = sorted(set(by_position) - set(available))
    if unknown:
        raise QueryProtocolError(f"Tape positions outside the stream: {unknown}")

    disclosures: List[Disclosure] = []
    predictions: Dict[str, List[Tuple[int, float]]] = {n: [] for n in methods}
    deliveries: Dict[str, List[Tuple[int, str, str, int]]] = {n: [] for n in methods}
    names = sorted(methods)

    for image in stream:
        step = int(image.position)
        public_observation = PublicObservation.from_stream(image)
        for name in names:
            value = float(methods[name].predict(step, public_observation))
            if value != value or value in (float("inf"), float("-inf")):
                raise QueryProtocolError(
                    f"Method {name} produced a non-finite prediction at {step}")
            predictions[name].append((step, value))
        event = by_position.get(step)
        if event is None:
            continue
        if event.path not in labels:
            raise QueryProtocolError(f"Missing label for query at position {step}")
        if event.rel_path != image.rel_path or event.sample_key != image.sample_key:
            raise QueryProtocolError("Tape event does not match the stream image")
        label = 0 if bool(labels[event.path]) else 1
        disclosures.append(Disclosure(position=step, sample_key=image.sample_key,
                                      rel_path=image.rel_path, label=label))
        for name in names:
            methods[name].observe_feedback(step, image, label)
            deliveries[name].append((step, image.sample_key, image.rel_path, label))

    return ReplayResult(disclosures=tuple(disclosures),
                        predictions={n: tuple(v) for n, v in predictions.items()},
                        deliveries={n: tuple(v) for n, v in deliveries.items()})


class PassiveAdmission:
    """Simulated method: fixed prediction, records everything it can see.

    ``labels_visible_at_prediction`` records how many labels each prediction
    call could already see, which is exactly what the causal audit inspects.
    """

    name = "passive_admission"

    def __init__(self) -> None:
        self.predictions: List[Tuple[int, float]] = []
        self.feedback: List[Tuple[int, str, str, int]] = []
        self.labels_visible_at_prediction: List[Tuple[int, int]] = []

    def predict(self, step: int, image: PublicObservation) -> float:
        self.labels_visible_at_prediction.append((int(step), len(self.feedback)))
        self.predictions.append((int(step), 0.0))
        return 0.0

    def observe_feedback(self, step: int, image: StreamImage, label: int) -> None:
        self.feedback.append((int(step), image.sample_key, image.rel_path, int(label)))


class RunningMeanAdmission:
    """Simulated method whose prediction uses only already-disclosed labels.

    It demonstrates the legal causal direction (past labels -> current state)
    without implementing any research Admission algorithm.
    """

    name = "running_mean_admission"

    def __init__(self, prior: float = 0.5) -> None:
        self.prior = float(prior)
        self.predictions: List[Tuple[int, float]] = []
        self.feedback: List[Tuple[int, str, str, int]] = []
        self.labels_visible_at_prediction: List[Tuple[int, int]] = []

    def predict(self, step: int, image: PublicObservation) -> float:
        self.labels_visible_at_prediction.append((int(step), len(self.feedback)))
        value = (self.prior if not self.feedback
                 else sum(row[3] for row in self.feedback) / len(self.feedback))
        self.predictions.append((int(step), float(value)))
        return float(value)

    def observe_feedback(self, step: int, image: StreamImage, label: int) -> None:
        self.feedback.append((int(step), image.sample_key, image.rel_path, int(label)))


def default_simulated_methods() -> Dict[str, Any]:
    """Two simulated, non-research Admission consumers of one shared tape."""
    return {PassiveAdmission.name: PassiveAdmission(),
            RunningMeanAdmission.name: RunningMeanAdmission()}


# ---------------------------------------------------------------------------
#  Q-B / Q-C interface contracts (declared, NOT implemented in D1-A)
# ---------------------------------------------------------------------------


def q_b_interface_contract() -> Dict[str, Any]:
    """Declared shape of the future Q-B shared frozen-reference query policy.

    Q-B is deliberately **not** a reproduction of the original fully dynamic
    v22 controller: the reference query policy may read only its own legally
    acquired feedback ledger, a frozen reference scorer and a shared initial
    threshold, and never the dynamic memory state of any compared method.
    """
    return {
        "policy_id": "Q-B",
        "status": "interface_only_not_implemented_in_d1a",
        "allowed_inputs": [
            "sanitized PublicObservation: opaque handle, camera id, step and phase",
            "frozen reference scorer output for the current image",
            "the reference controller's own previously disclosed labels and spent budget",
            "shared initial theta0 and pre-registered rule constants",
        ],
        "forbidden_inputs": [
            "STM/LTM/Buffer contents of any compared Admission method",
            "Signed Correction or dynamic Sidecar state of any method",
            "future labels, evaluation-probe labels, defect masks",
            "label-conditional stratification of the query tape",
            "raw Real-IAD path / official sample key containing OK or NG tokens",
        ],
        "output_contract": ("one frozen query tape replayed unchanged to every "
                            "Admission method"),
        "budget_rule": ("cap B; unused budget is saved and reported, never "
                        "backfilled with future labels"),
    }


def q_c_interface_contract() -> Dict[str, Any]:
    """Declared shape of the future Q-C native AIF-inspired closed loop."""
    return {
        "policy_id": "Q-C",
        "status": "interface_only_not_implemented_in_d1a",
        "allowed_inputs": [
            "each method's own legal online state (memory, correction, sidecar)",
            "labels that method has itself already obtained",
            "its own remaining budget under the shared cap B",
        ],
        "forbidden_inputs": [
            "future labels, evaluation-probe labels, defect masks",
            "another method's query ledger or memory state",
        ],
        "output_contract": ("per-method query trajectory allowed; shared budget "
                            "cap and identical supervision permissions"),
        "reporting_rule": ("call it budget-cap-matched, not count-matched, "
                           "whenever actual label counts differ"),
    }
