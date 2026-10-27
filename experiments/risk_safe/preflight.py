#!/usr/bin/env python3
"""D1-A: metadata-only Real-IAD dataset and query-budget preflight.

This module establishes the *infrastructure* for Risk-Safe Memory Admission,
and nothing else:

* how many independent physical objects / original camera images exist;
* whether support / mature-history / stream / evaluation-probe partitions are
  sample-disjoint under the official Real-IAD identity;
* how Q-A shared label-blind random tapes behave under absolute human-label
  budgets;
* whether the historical V0.2 3%-of-images denominator really meant
  79 queries on 525 official samples (and not ~16);
* whether candidate budgets can *ever* satisfy the declared Sidecar gates.

It reads JSON metadata and inspects file paths only.  No feature extraction, no
DINOv2 loading, no CUDA allocation, no anomaly-detection metric.

Example
-------
``python -m experiments.risk_safe.preflight --dataset-root "G:/dataset/realiad"``
``    --json-dir "G:/dataset/realiad/jsons/realiad_jsons"``
``    --out-dir "outputs/risk_safe_sprint_20261027/d1a_preflight"``
"""
from __future__ import annotations

import argparse
import ast
import collections
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:  # package import (``python -m experiments.risk_safe.preflight``)
    from . import query_protocol as qp
except ImportError:  # direct script execution
    import query_protocol as qp  # type: ignore[no-redef]

import final_decision_core as legacy_core  # historical identity + 3% tape rule

# ---------------------------------------------------------------------------
#  Pre-registered D1-A configuration
# ---------------------------------------------------------------------------

STAGE = "D1-A"

SCHEMA_DATASET = "risk_safe.d1a.dataset_preflight.v1"
SCHEMA_BUDGET = "risk_safe.d1a.query_budget_preflight.v1"
SCHEMA_TAPE = "risk_safe.d1a.query_tape_manifest.v1"
SCHEMA_SOURCE = "risk_safe.d1a.source_manifest.v1"

OUTPUT_DATASET = "dataset_preflight.json"
OUTPUT_BUDGET = "query_budget_preflight.json"
OUTPUT_TAPE = "query_tape_manifest.json"
OUTPUT_SOURCE = "source_manifest.json"
OUTPUT_FILES = (OUTPUT_DATASET, OUTPUT_BUDGET, OUTPUT_TAPE, OUTPUT_SOURCE)

DEFAULT_DATASET_ROOT = "G:/dataset/realiad"
DEFAULT_JSON_SUBDIR = "jsons/realiad_jsons"
DEFAULT_OUT_DIR = "outputs/risk_safe_sprint_20261027/d1a_preflight"

CLASSES: Tuple[str, ...] = ("audiojack", "bottle_cap", "eraser",
                            "plastic_plug", "switch", "rolled_strip_base")

SIDECAR_SOURCE = "test_phase3_AIF_simplified_v22.py"
SIDECAR_CONFIG_CLASS = "GuardedThetaSidecarConfig"

#: Files whose SHA-256 is recorded in ``source_manifest.json``.
SOURCE_FILES: Tuple[str, ...] = (
    "experiments/risk_safe/__init__.py",
    "experiments/risk_safe/preflight.py",
    "experiments/risk_safe/query_protocol.py",
    "experiments/final_decision_core.py",
    "experiments/v02_architecture_core.py",
    "vmb_visionad_new.py",
    "test_phase3_AIF_simplified_v22.py",
    "tests/test_risk_safe_d1a.py",
)


class DatasetProtocolError(RuntimeError):
    """A dataset/split invariant was violated; the preflight must fail closed."""


class MalformedImageName(ValueError):
    """An image file name or directory does not prove physical identity."""


class InsufficientObjects(DatasetProtocolError):
    """Too few independent official samples for the requested partition."""


class OutputMismatch(RuntimeError):
    """A preserved output differs from the current code/data/configuration."""


# ---------------------------------------------------------------------------
#  Real-IAD identity: three distinct identifiers
# ---------------------------------------------------------------------------
#
# 1. ``official_sample_key`` -- the exact multi-view sample identity used by the
#    official Real-IAD evaluation.  Source:
#    https://github.com/Tencent/AnomalyDetection_Real-IAD (README, section 3),
#    ``--sample_key_pat "([a-zA-Z][a-zA-Z0-9_]*_[0-9]{4}_[A-Z][A-Z_]*[A-Z])_C[0-9]_"``
#    It is ``<category>_<NNNN>_<CONDITION>`` where CONDITION is ``OK`` or
#    ``NG[_<DEFECT>]``.  This is the *primary* grouping of D1-A-R1 and is
#    byte-identical to the repository's historical
#    ``final_decision_core.object_id`` rule (verified on all 151,050 released
#    records of the 30 Real-IAD categories).
# 2. ``directory_serial_key`` -- the ``S####`` directory name.  Retained only as
#    an independent, conservative *sensitivity* diagnostic.  It is NOT a sample
#    identity: one ``S####`` directory hosts up to five official samples
#    (the OK session plus one session per defect code) captured at different
#    times.
# 3. ``image_id`` -- the class-relative path of the exact original image.
#
# Documented dataset identity (1 and 3) is kept separate from any assumption
# about physical manufactured specimens: two different official sample keys may
# or may not be the same product, and the released metadata does not say.

#: Official Real-IAD sample-key expression (captured group only).
OFFICIAL_SAMPLE_KEY_RE = re.compile(
    r"(?P<sample_key>[a-zA-Z][a-zA-Z0-9_]*_[0-9]{4}_[A-Z][A-Z_]*[A-Z])_C[0-9]_")

#: Structural capture naming: ``<class><_|__><NNNN>_<OK|NG>[_<DEFECT>]_C<c>_<ts>``.
#: The separator differs between OK sessions (single ``_``) and NG sessions
#: (double ``__``) in the released metadata, so it is matched explicitly.
IMAGE_NAME_RE = re.compile(
    r"^(?P<cls>[a-z][a-z0-9]*(?:_[a-z0-9]+)*)"
    r"(?P<sep>_+)"
    r"(?P<sample>\d{4})_"
    r"(?P<state>OK|NG)"
    r"(?:_(?P<defect>[A-Z]{2,4}))?"
    r"_C(?P<camera>[1-5])_"
    r"(?P<timestamp>\d{14})"
    r"\.(?P<ext>jpg|jpeg|png)$"
)

#: Sample *directory*, e.g. ``S0001``.  Diagnostic identifier only.
SAMPLE_DIR_RE = re.compile(r"^S(?P<sample>\d{4})$")

TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"

#: Identity contract recorded in every output manifest.
IDENTITY_VERSION = "realid-offical-sample-key-v2"
PRIMARY_GROUPING_UNIT = "official_sample_key"
SENSITIVITY_GROUPING_UNIT = "directory_serial_key"


@dataclass(frozen=True)
class ParsedImageName:
    """Metadata provable from an image path alone (no label is read)."""

    class_name: str
    official_sample_key: str    # ``audiojack_0001_OK`` / ``audiojack__0001_NG_BX``
    directory_serial_key: str   # ``S0001`` -- diagnostic only
    image_id: str               # class-relative image path
    capture_session: str        # ``OK`` or ``NG_<DEFECT>``
    camera_id: int
    timestamp: str
    rel_path: str               # class-relative path, e.g. ``NG/BX/S0004/<name>``


def parse_image_name(path: Any, expected_class: Optional[str] = None
                     ) -> ParsedImageName:
    """Prove Real-IAD identity from the file name *and* its directory.

    Returns the three distinct identifiers (official sample key, directory
    serial, image id).  Fails closed instead of guessing: an unparseable name, a
    name the official sample-key expression cannot match, a sample directory that
    disagrees with the numeric token, a camera outside ``C1..C5``, an invalid
    capture timestamp, or a directory layout that contradicts the encoded OK/NG
    session all raise :class:`MalformedImageName`.
    """
    normalized = str(path).replace("\\", "/")
    parts = normalized.split("/")
    if len(parts) < 3:
        raise MalformedImageName(f"Path lacks Real-IAD directory structure: {path}")
    name = parts[-1]
    match = IMAGE_NAME_RE.match(name)
    if match is None:
        raise MalformedImageName(f"Cannot prove Real-IAD identity: {path}")

    class_name = match.group("cls")
    if expected_class is not None and class_name != expected_class:
        raise MalformedImageName(
            f"File name class '{class_name}' != expected '{expected_class}': {path}")

    state = match.group("state")
    defect = match.group("defect")
    if state == "OK" and defect is not None:
        raise MalformedImageName(f"OK session cannot carry a defect code: {path}")
    if state == "NG" and defect is None:
        raise MalformedImageName(f"NG session must carry a defect code: {path}")
    session = "OK" if state == "OK" else f"NG_{defect}"

    sample_dir = parts[-2]
    directory = SAMPLE_DIR_RE.match(sample_dir)
    if directory is None:
        raise MalformedImageName(f"Missing sample directory (S####) for {path}")
    if directory.group("sample") != match.group("sample"):
        raise MalformedImageName(
            f"Sample directory {sample_dir} contradicts file token "
            f"{match.group('sample')}: {path}")

    official = OFFICIAL_SAMPLE_KEY_RE.search(name)
    if official is None:
        raise MalformedImageName(
            f"Official Real-IAD sample key cannot be extracted: {path}")
    official_key = official.group("sample_key")
    suffix = f"_{match.group('sample')}_{session}"
    if not official_key.startswith(class_name) or not official_key.endswith(suffix):
        raise MalformedImageName(
            f"Official sample key '{official_key}' is inconsistent with the "
            f"class and condition in {path}")

    timestamp = match.group("timestamp")
    try:
        datetime.strptime(timestamp, TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise MalformedImageName(
            f"Invalid capture timestamp {timestamp}: {path}") from exc

    expected_dirs = (["OK", sample_dir] if state == "OK"
                     else ["NG", str(defect), sample_dir])
    observed_dirs = parts[:-1][-len(expected_dirs):]
    if observed_dirs != expected_dirs:
        raise MalformedImageName(
            f"Directory layout {observed_dirs} contradicts encoded session "
            f"{expected_dirs}: {path}")

    rel_path = "/".join(expected_dirs + [name])
    return ParsedImageName(
        class_name=class_name,
        official_sample_key=official_key,
        directory_serial_key=sample_dir,
        image_id=rel_path,
        capture_session=session,
        camera_id=int(match.group("camera")),
        timestamp=timestamp,
        rel_path=rel_path,
    )


@dataclass(frozen=True)
class ImageRecord:
    """One original camera observation with its offline label.

    ``sample_key`` is the **official Real-IAD sample key** (primary grouping).
    ``directory_serial_key`` is the ``S####`` diagnostic grouping and must never
    be used as a sample identity.  ``is_good`` is the *image-level* label taken
    from ``anomaly_class == OK``; it is used only for metadata diagnostics and
    evaluation-only probe composition, never for query selection.
    """

    path: str
    rel_path: str
    class_name: str
    split: str
    sample_key: str             # official Real-IAD sample key
    directory_serial_key: str   # S#### -- sensitivity diagnostic only
    capture_session: str
    camera_id: int
    timestamp: str
    is_good: bool

    @property
    def official_sample_key(self) -> str:
        return self.sample_key

    @property
    def image_id(self) -> str:
        return self.rel_path

    def capture_order(self) -> Tuple[str, int, str]:
        return (self.timestamp, int(self.camera_id), self.path)

    def grouping_key(self, unit: str = PRIMARY_GROUPING_UNIT) -> str:
        if unit == PRIMARY_GROUPING_UNIT:
            return self.sample_key
        if unit == SENSITIVITY_GROUPING_UNIT:
            return f"{self.class_name}/{self.directory_serial_key}"
        raise DatasetProtocolError(f"Unknown grouping unit: {unit}")


def records_from_samples(samples: Sequence[Any], class_name: str,
                         split: str) -> List[ImageRecord]:
    """Convert parser samples into validated records (labels kept offline)."""
    records: List[ImageRecord] = []
    seen_paths: set = set()
    for sample in samples:
        path = str(sample.path).replace("\\", "/")
        if path in seen_paths:
            raise DatasetProtocolError(f"Duplicate image path: {path}")
        seen_paths.add(path)
        parsed = parse_image_name(path, expected_class=class_name)
        records.append(ImageRecord(
            path=path,
            rel_path=parsed.rel_path,
            class_name=class_name,
            split=split,
            sample_key=parsed.official_sample_key,
            directory_serial_key=parsed.directory_serial_key,
            capture_session=parsed.capture_session,
            camera_id=parsed.camera_id,
            timestamp=parsed.timestamp,
            is_good=bool(sample.is_good),
        ))
    return records


def group_by_sample(records: Sequence[ImageRecord],
                    unit: str = PRIMARY_GROUPING_UNIT
                    ) -> Dict[str, List[ImageRecord]]:
    """Group camera observations by sample key, in capture order.

    ``unit`` selects the identity definition: the official sample key (primary)
    or the ``S####`` directory serial (sensitivity diagnostic only).
    """
    grouped: Dict[str, List[ImageRecord]] = {}
    for record in records:
        grouped.setdefault(record.grouping_key(unit), []).append(record)
    return {key: sorted(value, key=ImageRecord.capture_order)
            for key, value in sorted(grouped.items())}


def group_by_serial(records: Sequence[ImageRecord]) -> Dict[str, List[ImageRecord]]:
    """``S####`` grouping -- conservative diagnostic, never the primary unit."""
    return group_by_sample(records, unit=SENSITIVITY_GROUPING_UNIT)


def load_class_records(dataset_root: Any, json_dir: Any, class_name: str
                       ) -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """Read one class through the existing validated Real-IAD JSON parser."""
    import vmb_visionad_new as visionad  # existing parser; imports torch only

    index = visionad.RealIADJSONIndex(root=str(dataset_root),
                                      json_dir=str(json_dir),
                                      only_classes=[class_name])
    if class_name not in index.classes:
        raise DatasetProtocolError(
            f"Class {class_name} absent from {json_dir} (found {index.classes})")
    train = records_from_samples(
        [s for s in index.train_samples if s.cls_name == class_name],
        class_name, "train")
    test = records_from_samples(
        [s for s in index.test_samples if s.cls_name == class_name],
        class_name, "test")
    validate_class_records(class_name, train, test)
    return train, test


def validate_class_records(class_name: str, train: Sequence[ImageRecord],
                           test: Sequence[ImageRecord]) -> Dict[str, Any]:
    """Fail closed on duplicate files; report (do not 'fix') split overlap."""
    if not train or not test:
        raise DatasetProtocolError(f"{class_name}: empty train or test split")
    all_records = list(train) + list(test)
    paths = [r.path for r in all_records]
    if len(set(paths)) != len(paths):
        raise DatasetProtocolError(f"{class_name}: duplicate image path across splits")
    basenames = [r.path.rsplit("/", 1)[-1] for r in all_records]
    duplicates = [name for name, count in collections.Counter(basenames).items()
                  if count > 1]
    if duplicates:
        raise DatasetProtocolError(
            f"{class_name}: duplicate image file name(s): {sorted(duplicates)[:5]}")
    rel_paths = [r.rel_path for r in all_records]
    if len(set(rel_paths)) != len(rel_paths):
        raise DatasetProtocolError(f"{class_name}: duplicate class-relative path")

    train_samples = set(group_by_sample(train))
    test_samples = set(group_by_sample(test))
    sample_overlap = sorted(train_samples & test_samples)
    train_serials = {r.directory_serial_key for r in train}
    test_serials = {r.directory_serial_key for r in test}
    serial_overlap = sorted(train_serials & test_serials)
    return {
        "official_sample_key": {
            "N_train_samples": len(train_samples),
            "N_test_samples": len(test_samples),
            "N_samples_in_both_splits": len(sample_overlap),
            "example_shared_samples": sample_overlap[:10],
            "train_test_disjoint": not sample_overlap,
        },
        "directory_serial_key": {
            "N_train_serials": len(train_serials),
            "N_test_serials": len(test_serials),
            "N_serials_in_both_splits": len(serial_overlap),
            "example_shared_serials": serial_overlap[:10],
            "is_a_sample_identity": False,
            "interpretation": (
                "shared S#### directories do NOT by themselves constitute "
                "sample-level evaluation leakage; they are a possible "
                "physical-specimen correlation that the released metadata "
                "cannot resolve"),
        },
    }


# ---------------------------------------------------------------------------
#  Official-sample-disjoint partitioning (historical V0.2 procedure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitConfig:
    """Pre-registered partition sizes, restored to the historical V0.2 values.

    The isolation unit is the **official Real-IAD sample key**
    (``<category>_<NNNN>_<CONDITION>``), which is what the official evaluation
    and the historical V0.2 ``final_decision_core.object_id`` rule group by.
    Because the released train and test splits are disjoint at this level, the
    historical sizes (4 support, 64 mature history, 24+24 probes, 3 x 175 phases)
    remain feasible and are verified per class before anything is built.
    """

    support_seeds: Tuple[int, ...] = (0, 1, 2)
    n_support: int = 4
    n_history: int = 64
    n_probe_normal: int = 24
    n_probe_defect: int = 24
    n_phases: int = 3
    groups_per_phase: int = 175
    history_seed: int = 190021
    probe_seed: int = 2201
    support_view_seed_offset: int = 891

    def __post_init__(self) -> None:
        if self.n_support != 4:
            raise DatasetProtocolError("Strict four-shot support is frozen")
        if self.n_phases != 3:
            raise DatasetProtocolError("Three stream phases are frozen")
        if self.groups_per_phase < 1:
            raise DatasetProtocolError("Phase size must be positive")
        if not self.support_seeds:
            raise DatasetProtocolError("At least one support seed is required")

    @property
    def required_test_samples(self) -> int:
        return (2 * self.n_probe_normal + 2 * self.n_probe_defect +
                self.n_phases * self.groups_per_phase)


@dataclass(frozen=True)
class PartitionPlan:
    """One class's sample-disjoint partition, with sample-key sets for audits."""

    class_name: str
    support: Tuple[ImageRecord, ...]
    history: Tuple[ImageRecord, ...]
    phases: Tuple[Tuple[ImageRecord, ...], ...]
    probe_a: Tuple[ImageRecord, ...]
    probe_b: Tuple[ImageRecord, ...]
    support_samples_by_seed: Mapping[int, Tuple[str, ...]]
    history_samples: Tuple[str, ...]
    phase_samples: Tuple[Tuple[str, ...], ...]
    probe_a_samples: Tuple[str, ...]
    probe_b_samples: Tuple[str, ...]
    reserved_samples: Tuple[str, ...]
    excluded_shared_samples: Tuple[str, ...]

    def sample_sets(self) -> Dict[str, set]:
        return {
            "support": {o for ids in self.support_samples_by_seed.values()
                        for o in ids},
            "mature_history": set(self.history_samples),
            "stream": {o for phase in self.phase_samples for o in phase},
            "probe_A": set(self.probe_a_samples),
            "probe_B": set(self.probe_b_samples),
        }

    def partition_images(self) -> Dict[str, int]:
        return {
            "support": len(self.support),
            "mature_history": len(self.history),
            "stream": sum(len(p) for p in self.phases),
            "probe_A": len(self.probe_a),
            "probe_B": len(self.probe_b),
        }


def build_partitions(train: Sequence[ImageRecord], test: Sequence[ImageRecord],
                     config: SplitConfig = SplitConfig()) -> PartitionPlan:
    """Build an official-sample-disjoint split for one class.

    Procedure (historical V0.2 reference, same seeds and same draw order):

    1. 64 verified-normal train samples for the *mature* regime, drawn with
       seed 190021 -- one camera image each, identity-invariant across seeds;
    2. 4 strict-Cold support samples per support seed, drawn from the remaining
       normal train samples; one camera image per sample;
    3. the union of every support/history sample is *reserved*: those samples are
       removed from the test pool (a defensive guard -- the released Real-IAD
       train/test splits are already disjoint at official-sample level, so this
       removes nothing and the historical partition is reproduced exactly);
    4. 24 normal + 24 defect probe samples for probe A and probe B (sample-level
       stratification is evaluation-only);
    5. the remaining test samples form three contiguous phases of exactly
       ``groups_per_phase`` samples each, as in the historical protocol.
    """
    train_groups = group_by_sample(train)
    test_groups = group_by_sample(test)
    class_name = test[0].class_name

    normal_train = sorted(oid for oid, recs in train_groups.items()
                          if all(r.is_good for r in recs))
    if len(normal_train) < config.n_history + config.n_support:
        raise InsufficientObjects(
            f"Class {class_name}: the mature-history regime needs "
            f"{config.n_history + config.n_support} verified-normal train "
            f"samples, only {len(normal_train)} exist")

    history_samples = sorted(random.Random(config.history_seed).sample(
        normal_train, config.n_history))
    history = tuple(random.Random(config.support_view_seed_offset + i).choice(
        train_groups[oid]) for i, oid in enumerate(history_samples))

    history_set = set(history_samples)
    support_samples_by_seed: Dict[int, Tuple[str, ...]] = {}
    support_records: List[ImageRecord] = []
    for seed in sorted({int(s) for s in config.support_seeds}):
        pool = [oid for oid in normal_train if oid not in history_set]
        if len(pool) < config.n_support:
            raise InsufficientObjects(
                f"Support seed {seed}: need {config.n_support} normal samples "
                f"outside the mature-history pool, have {len(pool)}")
        chosen = sorted(random.Random(seed).sample(pool, config.n_support))
        rng = random.Random(seed + config.support_view_seed_offset)
        support_records.extend(rng.choice(train_groups[oid]) for oid in chosen)
        support_samples_by_seed[seed] = tuple(chosen)

    support_sample_union = {o for ids in support_samples_by_seed.values()
                            for o in ids}
    reserved = sorted(history_set | support_sample_union)
    reserved_set = set(reserved)
    excluded_shared = sorted(reserved_set & set(test_groups))

    available = {oid: recs for oid, recs in test_groups.items()
                 if oid not in reserved_set}
    if not available:
        raise InsufficientObjects(
            f"Class {class_name}: no test samples remain after reserving the "
            f"support/mature-history objects")
    return _finish_plan(class_name, available, support_records, history, reserved,
                        excluded_shared, support_samples_by_seed, config)


def _finish_plan(class_name: str, available: Mapping[str, List[ImageRecord]],
                 support_records: Sequence[ImageRecord],
                 history: Sequence[ImageRecord], reserved: Sequence[str],
                 excluded_shared: Sequence[str],
                 support_samples_by_seed: Mapping[int, Tuple[str, ...]],
                 config: SplitConfig) -> PartitionPlan:
    """Select evaluation-only probes and the three stream phases."""
    normal_ids = sorted(oid for oid, recs in available.items()
                        if all(r.is_good for r in recs))
    defect_ids = sorted(oid for oid, recs in available.items()
                        if any(not r.is_good for r in recs))
    rng = random.Random(config.probe_seed)
    rng.shuffle(normal_ids)
    rng.shuffle(defect_ids)
    need_normal = 2 * config.n_probe_normal
    need_defect = 2 * config.n_probe_defect
    if len(normal_ids) < need_normal or len(defect_ids) < need_defect:
        raise InsufficientObjects(
            f"Class {class_name}: evaluation-only probes need {need_normal} "
            f"normal and {need_defect} defect official samples; available: "
            f"normal={len(normal_ids)}, defect={len(defect_ids)}")

    probe_a_ids = normal_ids[:config.n_probe_normal] + \
        defect_ids[:config.n_probe_defect]
    probe_b_ids = normal_ids[config.n_probe_normal:need_normal] + \
        defect_ids[config.n_probe_defect:need_defect]
    probe_ids = set(probe_a_ids) | set(probe_b_ids)

    remaining = sorted(oid for oid in available if oid not in probe_ids)
    rng.shuffle(remaining)
    required_stream = config.n_phases * config.groups_per_phase
    if len(remaining) < required_stream:
        raise InsufficientObjects(
            f"Class {class_name}: the historical partition needs "
            f"{required_stream} stream samples ({config.n_phases} phases x "
            f"{config.groups_per_phase}), only {len(remaining)} official samples "
            f"remain after the evaluation-only probes")

    phase_ids: List[List[str]] = [
        remaining[i * config.groups_per_phase:(i + 1) * config.groups_per_phase]
        for i in range(config.n_phases)]

    def images(ids: Sequence[str]) -> Tuple[ImageRecord, ...]:
        return tuple(record for oid in ids for record in available[oid])

    plan = PartitionPlan(
        class_name=class_name,
        support=tuple(sorted(support_records, key=ImageRecord.capture_order)),
        history=tuple(history),
        phases=tuple(images(ids) for ids in phase_ids),
        probe_a=images(probe_a_ids),
        probe_b=images(probe_b_ids),
        support_samples_by_seed={k: tuple(v) for k, v in
                                 support_samples_by_seed.items()},
        history_samples=tuple(r.sample_key for r in history),
        phase_samples=tuple(tuple(ids) for ids in phase_ids),
        probe_a_samples=tuple(sorted(probe_a_ids)),
        probe_b_samples=tuple(sorted(probe_b_ids)),
        reserved_samples=tuple(reserved),
        excluded_shared_samples=tuple(excluded_shared),
    )
    audit_sample_disjointness(plan)
    return plan


def audit_sample_disjointness(plan: PartitionPlan) -> Dict[str, Any]:
    """Every pair of partitions must share zero official sample keys."""
    sets = plan.sample_sets()
    names = sorted(sets)
    overlaps: Dict[str, Any] = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            shared = sorted(sets[left] & sets[right])
            overlaps[f"{left}|{right}"] = {"N_shared": len(shared),
                                           "examples": shared[:5]}
    failures = {k: v for k, v in overlaps.items() if v["N_shared"]}
    if failures:
        raise DatasetProtocolError(
            f"{plan.class_name}: official-sample leakage between partitions: "
            f"{failures}")
    return {
        "grouping_unit": PRIMARY_GROUPING_UNIT,
        "rule": ("an official Real-IAD sample key "
                 "(<category>_<NNNN>_<CONDITION>) may appear in exactly one of "
                 "support / mature_history / stream / probe_A / probe_B"),
        "partition_sample_counts": {k: len(v) for k, v in sets.items()},
        "pairwise_shared_samples": overlaps,
        "all_partitions_disjoint": True,
    }


# ---------------------------------------------------------------------------
#  Dataset diagnostics
# ---------------------------------------------------------------------------


def partition_feasibility(class_name: str, train: Sequence[ImageRecord],
                          test: Sequence[ImageRecord],
                          config: SplitConfig) -> Dict[str, Any]:
    """Report the sample counts available for each partition before building.

    Nothing is assumed: the historical sizes are checked against the actual
    official-sample counts and the caller fails explicitly when they do not fit.
    """
    train_groups = group_by_sample(train)
    test_groups = group_by_sample(test)
    normal_train = [k for k, v in train_groups.items()
                    if all(r.is_good for r in v)]
    normal_test = [k for k, v in test_groups.items()
                   if all(r.is_good for r in v)]
    defect_test = [k for k, v in test_groups.items()
                   if any(not r.is_good for r in v)]
    return {
        "N_train_samples_all_normal": len(normal_train),
        "N_test_samples": len(test_groups),
        "N_test_samples_all_normal": len(normal_test),
        "N_test_samples_with_defect_image": len(defect_test),
        "required_train_normal_samples": config.n_history + config.n_support,
        "required_test_samples_after_probes":
            config.n_phases * config.groups_per_phase,
        "required_probe_normal_samples": 2 * config.n_probe_normal,
        "required_probe_defect_samples": 2 * config.n_probe_defect,
        "train_capacity_ok": len(normal_train) >= config.n_history + config.n_support,
        "test_capacity_ok": (len(test_groups) >= config.required_test_samples and
                             len(normal_test) >= 2 * config.n_probe_normal and
                             len(defect_test) >= 2 * config.n_probe_defect),
        "unused_test_samples_if_built": max(
            0, len(test_groups) - config.required_test_samples),
        "note": ("counts are official sample keys; historical V0.2 sizes are "
                 "verified rather than assumed"),
    }


def identity_comparison(class_name: str, train: Sequence[ImageRecord],
                        test: Sequence[ImageRecord]) -> Dict[str, Any]:
    """Compare the official sample key with the ``S####`` directory serial.

    The three findings this report separates are *not* equivalent:

    * a **verified parser error** -- grouping by ``S####`` merges distinct
      official samples (proved by the counts below);
    * a **possible physical-specimen correlation** -- one ``S####`` directory
      hosts several official samples, so two samples may be the same product
      photographed under different conditions.  The released metadata cannot
      resolve this, and it is not evidence of leakage;
    * **demonstrated evaluation leakage** -- only an overlap of *official sample
      keys* across partitions would be leakage.  It is reported, and asserted to
      be zero, for the primary protocol.
    """
    official_train = group_by_sample(train)
    official_test = group_by_sample(test)
    serial_train = group_by_serial(train)
    serial_test = group_by_serial(test)
    all_records = list(train) + list(test)

    images_per_sample = collections.Counter(
        len(v) for v in list(official_train.values()) + list(official_test.values()))
    expected_cameras = [1, 2, 3, 4, 5]
    missing_views = []
    duplicate_views = []
    for key, records in list(official_train.items()) + list(official_test.items()):
        cameras = sorted(r.camera_id for r in records)
        if cameras != expected_cameras:
            missing_views.append({"sample": key, "cameras": cameras})
        if len({r.camera_id for r in records}) != len(records):
            duplicate_views.append(key)

    by_serial: Dict[str, Dict[str, Any]] = {}
    for record in all_records:
        entry = by_serial.setdefault(record.directory_serial_key,
                                     {"samples": set(), "sessions": set(),
                                      "timestamps": set(), "paths": {}})
        entry["samples"].add(record.sample_key)
        entry["sessions"].add(record.capture_session)
        entry["timestamps"].add(record.timestamp)
        entry["paths"].setdefault(record.sample_key, record.rel_path)

    shared = {serial: entry for serial, entry in by_serial.items()
              if len(entry["samples"]) > 1}
    crossing = [s for s, e in shared.items()
                if any(x == "OK" for x in e["sessions"])
                and any(x.startswith("NG") for x in e["sessions"])]
    multi_time = [s for s, e in shared.items() if len(e["timestamps"]) > 1]
    defect_codes = sorted({session.split("_", 1)[1]
                           for entry in shared.values()
                           for session in entry["sessions"]
                           if session.startswith("NG_")})

    examples = []
    for serial in sorted(shared, key=lambda s: (-len(shared[s]["samples"]), s))[:3]:
        entry = shared[serial]
        keys = sorted(entry["samples"])
        examples.append({
            "directory_serial_key": serial,
            "official_sample_keys": keys,
            "capture_sessions": sorted(entry["sessions"]),
            "capture_timestamps": sorted(entry["timestamps"]),
            "example_image_paths": [entry["paths"][k] for k in keys],
        })

    train_serials = set(serial_train)
    test_serials = set(serial_test)
    shared_samples = sorted(set(official_train) & set(official_test))

    max_per_serial = max((len(e["samples"]) for e in by_serial.values()), default=1)
    return {
        "grouping": {
            "official_sample_key": {
                "definition": "<category>_<NNNN>_<CONDITION> via the official "
                              "Real-IAD --sample_key_pat expression",
                "N_train_groups": len(official_train),
                "N_test_groups": len(official_test),
                "is_primary": True,
            },
            "directory_serial_key": {
                "definition": "S#### directory name",
                "N_train_groups": len(serial_train),
                "N_test_groups": len(serial_test),
                "is_primary": False,
                "role": "identity-sensitivity diagnostic only",
            },
            "image_id": {"definition": "class-relative original image path",
                         "N_images": len(all_records)},
        },
        "images_per_official_sample": {str(k): v for k, v in
                                       sorted(images_per_sample.items())},
        "five_view_structure": {
            "expected_cameras": expected_cameras,
            "samples_without_exact_C1_to_C5": len(missing_views),
            "examples": missing_views[:5],
            "samples_with_duplicate_camera_id": len(duplicate_views),
            "examples_duplicate": duplicate_views[:5],
        },
        "serial_collisions": {
            "N_serials_hosting_more_than_one_official_sample": len(shared),
            "N_serials_total": len(by_serial),
            "max_official_samples_per_serial": max_per_serial,
            "N_serials_crossing_OK_and_NG": len(crossing),
            "N_serials_with_multiple_capture_timestamps": len(multi_time),
            "defect_codes_involved": defect_codes,
            "examples": examples,
        },
        "train_test_overlap": {
            "official_sample_key": {
                "N_shared": len(shared_samples),
                "examples": shared_samples[:10],
                "is_evaluation_leakage": bool(shared_samples),
            },
            "directory_serial_key": {
                "N_shared": len(train_serials & test_serials),
                "examples": sorted(train_serials & test_serials)[:10],
                "is_evaluation_leakage": False,
                "explanation": ("a shared S#### directory pairs an OK-session "
                                "sample with NG-session samples captured at "
                                "different times; the official evaluation treats "
                                "them as different samples, so this is a possible "
                                "physical-specimen correlation, NOT demonstrated "
                                "leakage"),
            },
        },
        "findings": [
            {
                "kind": "verified_parser_error",
                "statement": ("grouping by the S#### directory merges distinct "
                              "official Real-IAD samples and therefore inflates "
                              "train/test overlap"),
                "evidence": (f"N_test_groups {len(official_test)} (official) vs "
                             f"{len(serial_test)} (serial) for {class_name}"),
            },
            {
                "kind": "possible_physical_specimen_correlation",
                "statement": ("one S#### directory hosts several official samples "
                              "captured under different conditions and times; "
                              "whether they are the same manufactured specimen is "
                              "not documented"),
                "evidence": (f"{len(shared)} directories host >1 official sample "
                             f"(max {max_per_serial}); {len(crossing)} cross OK and NG"),
            },
            {
                "kind": "demonstrated_evaluation_leakage",
                "statement": ("no official sample key appears in both the train "
                              "and the test split of this class"),
                "evidence": f"N_shared_official_sample_keys = {len(shared_samples)}",
            },
        ],
    }


def _histogram(values: Sequence[int]) -> Dict[str, int]:
    counter = collections.Counter(int(v) for v in values)
    return {str(k): counter[k] for k in sorted(counter)}


def _session_histogram(records: Sequence[ImageRecord]) -> Dict[str, int]:
    counter = collections.Counter(r.capture_session for r in records)
    return {key: counter[key] for key in sorted(counter)}


def _camera_distribution(records: Sequence[ImageRecord]) -> Dict[str, Any]:
    counter = collections.Counter(r.camera_id for r in records)
    total = len(records)
    return {
        "counts": {f"C{c}": counter[c] for c in sorted(counter)},
        "fractions": ({f"C{c}": round(counter[c] / total, 6)
                       for c in sorted(counter)} if total else {}),
    }


def group_of(records: Sequence[ImageRecord], sample_key: str) -> List[ImageRecord]:
    return [r for r in records if r.sample_key == sample_key]


def label_structure(records: Sequence[ImageRecord]) -> Dict[str, Any]:
    """Describe the image-label vs official-sample-label relation.

    This is the evidence for the ``no label propagation`` rule: an official
    sample photographed in a defective condition still contains camera views
    whose *own* image-level label is normal, so propagating a sample-level defect
    label to every view would invent labels.
    """
    groups = group_by_sample(records)
    all_normal = [oid for oid, recs in groups.items() if all(r.is_good for r in recs)]
    any_defect = [oid for oid, recs in groups.items() if any(not r.is_good for r in recs)]
    normal_inside_defect = sum(1 for oid in any_defect
                               for r in groups[oid] if r.is_good)
    defect_inside_normal = sum(1 for oid in all_normal
                               for r in groups[oid] if not r.is_good)
    if defect_inside_normal:
        raise DatasetProtocolError(
            "An official sample with only normal image labels contains a "
            "defect-labelled image; identity or labels are inconsistent")
    return {
        "N_samples_all_images_normal": len(all_normal),
        "N_samples_with_at_least_one_defect_image": len(any_defect),
        "N_normal_labelled_images_inside_defect_samples": normal_inside_defect,
        "N_defect_labelled_images_inside_normal_samples": defect_inside_normal,
        "label_propagation_rule": ("image labels are never propagated to the "
                                   "other camera views of the same sample"),
        "consequence": ("a sample-level defect probe unit still contributes "
                        "normal-labelled views; evaluation keeps image-level truth"),
    }


def _partition_report(plan: PartitionPlan) -> Dict[str, Any]:
    def object_label_summary(records: Sequence[ImageRecord],
                             ids: Sequence[str]) -> Tuple[int, int]:
        normal = sum(1 for oid in ids
                     if all(r.is_good for r in group_of(records, oid)))
        return normal, len(ids) - normal

    probe_a_normal, probe_a_defect = object_label_summary(
        plan.probe_a, plan.probe_a_samples)
    probe_b_normal, probe_b_defect = object_label_summary(
        plan.probe_b, plan.probe_b_samples)

    return {
        "grouping_unit": PRIMARY_GROUPING_UNIT,
        "support": {
            "N_samples": len({o for ids in plan.support_samples_by_seed.values()
                              for o in ids}),
            "N_images": len(plan.support),
            "samples_by_support_seed": {str(k): list(v) for k, v in
                                        sorted(plan.support_samples_by_seed.items())},
            "images": [r.rel_path for r in plan.support],
        },
        "mature_history": {
            "N_samples": len(plan.history_samples),
            "N_images": len(plan.history),
            "images": [r.rel_path for r in plan.history],
            "supervision_note": ("separate history-assisted regime; never "
                                 "counted inside a strict 4-shot label budget"),
        },
        "stream": {
            "N_samples": len({o for p in plan.phase_samples for o in p}),
            "N_images": sum(len(p) for p in plan.phases),
            "phases": [{"phase": i + 1, "N_samples": len(plan.phase_samples[i]),
                        "N_images": len(plan.phases[i])}
                       for i in range(len(plan.phases))],
            "sample_ids": [oid for ids in plan.phase_samples for oid in ids],
        },
        "probe_A": {
            "N_samples": len(plan.probe_a_samples),
            "N_images": len(plan.probe_a),
            "N_normal_samples": probe_a_normal,
            "N_defect_samples": probe_a_defect,
            "evaluation_only": True,
            "sample_ids": list(plan.probe_a_samples),
        },
        "probe_B": {
            "N_samples": len(plan.probe_b_samples),
            "N_images": len(plan.probe_b),
            "N_normal_samples": probe_b_normal,
            "N_defect_samples": probe_b_defect,
            "evaluation_only": True,
            "sample_ids": list(plan.probe_b_samples),
        },
        "reserved_samples_removed_from_test_pool": len(plan.reserved_samples),
        "reserved_samples_that_existed_in_test_split":
            len(plan.excluded_shared_samples),
        "excluded_shared_sample_examples": list(plan.excluded_shared_samples[:10]),
        "historical_reference": {
            "probe_samples_per_partition": 48,
            "stream_samples_per_phase": 175,
            "note": ("historical V0.2 sizes are reproduced exactly: the official "
                     "sample key restores 175 stream samples per phase, so the "
                     "historical 525-sample / 2,625-image denominators hold"),
        },
    }


def class_diagnostics(class_name: str, train: Sequence[ImageRecord],
                      test: Sequence[ImageRecord], plan: PartitionPlan,
                      config: Optional[SplitConfig] = None
                      ) -> Dict[str, Any]:
    """Metadata report for one class (no model, no image decoding)."""
    config = config or SplitConfig()
    train_groups = group_by_sample(train)
    test_groups = group_by_sample(test)
    split_overlap = validate_class_records(class_name, train, test)
    train_mixed = [oid for oid, recs in train_groups.items()
                   if len({r.is_good for r in recs}) > 1]
    test_mixed = [oid for oid, recs in test_groups.items()
                  if len({r.is_good for r in recs}) > 1]
    multi_view = [oid for oid, recs in list(train_groups.items()) +
                  list(test_groups.items()) if len(recs) > 1]

    return {
        "identity_version": IDENTITY_VERSION,
        "counts": {
            "train_images": len(train),
            "test_images": len(test),
            "train_camera_observations": len(train),
            "test_camera_observations": len(test),
            "train_official_samples": len(train_groups),
            "test_official_samples": len(test_groups),
            "train_directory_serials": len(group_by_serial(train)),
            "test_directory_serials": len(group_by_serial(test)),
            "train_verified_normal_images": sum(1 for r in train if r.is_good),
            "train_anomalous_images": sum(1 for r in train if not r.is_good),
            "test_verified_normal_images": sum(1 for r in test if r.is_good),
            "test_anomalous_images": sum(1 for r in test if not r.is_good),
            "samples_with_multiple_camera_images": len(multi_view),
            "samples_with_mixed_image_level_labels":
                len(train_mixed) + len(test_mixed),
        },
        "camera_view_distribution": {
            "train": _camera_distribution(train),
            "test": _camera_distribution(test),
        },
        "images_per_sample_histogram": {
            "train": _histogram([len(v) for v in train_groups.values()]),
            "test": _histogram([len(v) for v in test_groups.values()]),
        },
        "capture_session_distribution": {
            "train": _session_histogram(train),
            "test": _session_histogram(test),
        },
        "label_structure": {
            "train": label_structure(train),
            "test": label_structure(test),
            "example_test_samples_with_multiple_views":
                [oid for oid, recs in test_groups.items() if len(recs) > 1][:5],
            "example_test_samples_all_normal":
                [oid for oid, recs in test_groups.items()
                 if all(r.is_good for r in recs)][:5],
        },
        "split_identity_overlap": split_overlap,
        "identity_comparison": identity_comparison(class_name, train, test),
        "partition_feasibility": partition_feasibility(class_name, train, test,
                                                       config),
        "partitions": _partition_report(plan),
        "sample_disjointness": audit_sample_disjointness(plan),
    }


# ---------------------------------------------------------------------------
#  Historical V0.2 reference reproduction (same code, same seeds)
# ---------------------------------------------------------------------------


def historical_reference_audit(train: Sequence[ImageRecord],
                               test: Sequence[ImageRecord],
                               support_seeds: Sequence[int] = (0, 1, 2)
                               ) -> Dict[str, Any]:
    """Reproduce V0.2 exactly, then audit it under *both* identity definitions.

    ``v02_architecture_core`` groups images by the token
    ``<class>_<NNNN>_<STATE>[_<DEFECT>]``, which is byte-identical to the
    official Real-IAD sample key.  This function reproduces that split unchanged
    (so 525 samples x 5 views = 2,625 observations and 79 queries are re-verified
    rather than asserted) and then contrasts the sample-level audit with the
    ``S####`` serial-level sensitivity audit.
    """
    import v02_architecture_core as v02

    path_to_sample = {r.path: r.sample_key for r in list(train) + list(test)}
    path_to_serial = {r.path: r.directory_serial_key
                      for r in list(train) + list(test)}
    path_to_good = {r.path: r.is_good for r in list(train) + list(test)}
    legacy_train = [SimpleNamespace(path=r.path, is_good=r.is_good) for r in train]
    legacy_test = [SimpleNamespace(path=r.path, is_good=r.is_good) for r in test]

    required_groups = 2 * 24 + 2 * 24 + 3 * 175
    legacy_groups = legacy_core.group_samples(legacy_test)
    if len(legacy_groups) < required_groups:
        raise DatasetProtocolError(
            f"historical V0.2 reference needs {required_groups} legacy "
            f"(sample + capture session) test groups, only {len(legacy_groups)} "
            f"exist; the 525-sample / 79-query arithmetic cannot be reproduced "
            f"for this class")

    rows: List[Dict[str, Any]] = []
    for seed in support_seeds:
        supports, history = v02.select_history_and_support(legacy_train, int(seed))
        probes, phases, groups = v02.split_test(legacy_test)
        tape = legacy_core.query_tape(phases)
        stream = sum(phases, [])
        queried = [stream[int(pos) - 1] for pos in sorted(int(t) for t in tape)]

        train_samples = {path_to_sample[p] for p in supports}
        train_samples |= {path_to_sample[h.path] for h in history}
        test_samples = {path_to_sample[s.path] for s in stream}
        test_samples |= {path_to_sample[s.path] for s in sum(probes.values(), [])}
        sample_leaks = sorted(train_samples & test_samples)

        train_serials = {path_to_serial[p] for p in supports}
        train_serials |= {path_to_serial[h.path] for h in history}
        test_serials = {path_to_serial[s.path] for s in stream}
        test_serials |= {path_to_serial[s.path] for s in sum(probes.values(), [])}

        queried_samples = [path_to_sample[s.path] for s in queried]
        queried_serials = [path_to_serial[s.path] for s in queried]
        query_labels = [int(not path_to_good[s.path]) for s in queried]
        n_stream_groups = sum(len(p) for p in groups["phases"])

        rows.append({
            "support_seed": int(seed),
            "N_support_images": len(supports),
            "N_support_samples": len({path_to_sample[p] for p in supports}),
            "N_history_images": len(history),
            "N_history_samples": len({path_to_sample[h.path] for h in history}),
            "N_probe_samples_A": len(groups["A"]),
            "N_probe_samples_B": len(groups["B"]),
            "N_stream_groups": n_stream_groups,
            "N_stream_images": len(stream),
            "N_queries": len(tape),
            "image_query_fraction": round(len(tape) / len(stream), 6),
            "unique_sample_audit_fraction_vs_stream_groups":
                round(len(tape) / n_stream_groups, 6),
            "sample_level": {
                "N_train_samples_leaking_into_test_partitions": len(sample_leaks),
                "examples": sample_leaks[:10],
                "N_queries_on_a_repeated_sample":
                    len(queried_samples) - len(set(queried_samples)),
            },
            "serial_level_sensitivity": {
                "N_distinct_serials_in_stream":
                    len({path_to_serial[s.path] for s in stream}),
                "N_train_serials_also_in_test_partitions":
                    len(train_serials & test_serials),
                "N_queries_on_a_repeated_serial":
                    len(queried_serials) - len(set(queried_serials)),
                "interpretation": ("S#### overlap is a possible "
                                   "physical-specimen correlation, not sample-level "
                                   "evaluation leakage"),
            },
            "queried_label_composition_offline_only": {
                "N_normal": query_labels.count(0),
                "N_defect": query_labels.count(1),
            },
        })

    n_images = sorted({r["N_stream_images"] for r in rows})
    n_queries = sorted({r["N_queries"] for r in rows})
    n_groups = sorted({r["N_stream_groups"] for r in rows})
    arithmetic_ok = all(
        row["N_queries"] == int(round(0.03 * row["N_stream_images"]))
        for row in rows)
    return {
        "provenance": ("experiments/v02_architecture_core.py and "
                       "experiments/final_decision_core.py imported unchanged"),
        "grouping_unit": ("historical token <class>_<NNNN>_<STATE>[_<DEFECT>] = "
                          "official Real-IAD sample key"),
        "per_support_seed": rows,
        "arithmetic_check": {
            "N_stream_groups": n_groups,
            "N_stream_images": n_images,
            "N_queries": n_queries,
            "constant_across_support_seeds":
                len(n_images) == 1 and len(n_queries) == 1 and len(n_groups) == 1,
            "queries_equal_round_3pct_of_images": arithmetic_ok,
            "image_query_fraction": round(n_queries[0] / n_images[0], 6),
            "unique_sample_audit_fraction_vs_stream_groups":
                round(n_queries[0] / n_groups[0], 6),
            "historical_3pct_means_79_queries_not_16": arithmetic_ok,
        },
        "identity_sensitivity_audit": {
            "sample_level": {
                "leakage_detected": any(
                    r["sample_level"]["N_train_samples_leaking_into_test_partitions"] > 0
                    for r in rows),
                "max_train_samples_in_test_partitions": max(
                    r["sample_level"]["N_train_samples_leaking_into_test_partitions"]
                    for r in rows),
                "max_queries_on_a_repeated_sample": max(
                    r["sample_level"]["N_queries_on_a_repeated_sample"]
                    for r in rows),
            },
            "serial_level": {
                "N_distinct_serials_in_stream":
                    sorted({r["serial_level_sensitivity"]["N_distinct_serials_in_stream"]
                            for r in rows}),
                "max_train_serials_also_in_test_partitions": max(
                    r["serial_level_sensitivity"]["N_train_serials_also_in_test_partitions"]
                    for r in rows),
                "max_queries_on_a_repeated_serial": max(
                    r["serial_level_sensitivity"]["N_queries_on_a_repeated_serial"]
                    for r in rows),
            },
            "conclusion": (
                "the historical V0.2 split is disjoint at the official sample "
                "level (which is also the level used by the official Real-IAD "
                "evaluation); the S#### serial-level overlaps are a possible "
                "physical-specimen correlation and must not be reported as "
                "demonstrated evaluation leakage"),
        },
    }


# ---------------------------------------------------------------------------
#  Sidecar activation feasibility (metadata only, thresholds untouched)
# ---------------------------------------------------------------------------


def read_dataclass_defaults(source_path: Path, class_name: str) -> Dict[str, Any]:
    """Read declared dataclass defaults by AST, without importing the module.

    Parsing the source instead of importing ``test_phase3_AIF_simplified_v22``
    keeps the preflight free of torch/backbone side effects and guarantees the
    reported values are exactly the ones declared in the audited file.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            values: Dict[str, Any] = {}
            for statement in node.body:
                if (isinstance(statement, ast.AnnAssign)
                        and isinstance(statement.target, ast.Name)
                        and statement.value is not None):
                    try:
                        values[statement.target.id] = ast.literal_eval(statement.value)
                    except (ValueError, SyntaxError):
                        continue
            if not values:
                raise DatasetProtocolError(
                    f"{class_name} in {source_path.name} declares no literal defaults")
            return values
    raise DatasetProtocolError(
        f"Class {class_name} not found in {source_path.name}")


REQUIRED_SIDECAR_KEYS: Tuple[str, ...] = (
    "min_normal_anchor", "min_defect_anchor", "min_recent_total",
    "min_recent_per_class", "update_every_labeled", "normal_capacity",
    "defect_capacity", "recent_capacity", "enable_normal_fallback",
    "enable_fast_upward_rescue", "enable_balanced_rescue",
    "min_recent_normals_fallback", "min_defect_anchor_fallback",
    "normal_fallback_fpr_trigger", "fast_rescue_min_fp_normals",
    "balanced_min_normals", "balanced_min_defects", "balanced_cooldown_steps",
)

#: Documented defaults the brief names.  A silent edit of these values must
#: break the preflight instead of silently changing the feasibility verdict.
DOCUMENTED_SIDECAR_VALUES = {
    "min_normal_anchor": 32,
    "min_defect_anchor": 16,
    "min_recent_total": 24,
    "min_recent_per_class": 6,
}


def sidecar_mechanisms(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Declare each documented activation mechanism with its real gates.

    Gates are read from the same source as the values; nothing is lowered.
    ``min_budget_for_activation`` is the smallest absolute label budget for
    which the *count* requirements could be met by some label composition.
    """
    min_recent_total = int(config["min_recent_total"])
    min_per_class = int(config["min_recent_per_class"])
    return [
        {
            "mechanism": "dual_anchor_full",
            "controller_name": "guarded_dual_anchor_v2 (default)",
            "enabled_by_default": True,
            "hard_count_gates": {
                "recent_labeled_total_min": min_recent_total,
                "recent_normals_min": min_per_class,
                "recent_defects_min": min_per_class,
            },
            "min_budget_for_activation": max(min_recent_total, 2 * min_per_class),
            "gates_a_theta_move": True,
            "code_reference": "_can_evaluate_full / _maybe_update_theta",
            "note": ("evaluated only every update_every_labeled labels; the "
                     "recent window retains at most recent_capacity labels"),
        },
        {
            "mechanism": "normal_dominant_fallback",
            "controller_name": "guarded_dual_anchor fallback (default on)",
            "enabled_by_default": bool(config["enable_normal_fallback"]),
            "hard_count_gates": {
                "recent_labeled_total_min": min_recent_total,
                "recent_normals_min": int(config["min_recent_normals_fallback"]),
                "recent_defects_max_exclusive": min_per_class,
                "defect_anchor_buffer_min": int(config["min_defect_anchor_fallback"]),
                "recent_FPR_min": float(config["normal_fallback_fpr_trigger"]),
            },
            "min_budget_for_activation": max(
                min_recent_total,
                int(config["min_recent_normals_fallback"]) +
                int(config["min_defect_anchor_fallback"])),
            "gates_a_theta_move": True,
            "code_reference": "_can_evaluate_normal_fallback",
            "note": ("needs a defect-sparse recent window AND a current FPR above "
                     "the trigger; an observed-rate condition that a budget count "
                     "alone can never guarantee"),
        },
        {
            "mechanism": "fast_upward_rescue",
            "controller_name": "v22 fast upward rescue",
            "enabled_by_default": bool(config["enable_fast_upward_rescue"]),
            "hard_count_gates": {
                "verified_FP_normals_min": int(config["fast_rescue_min_fp_normals"]),
                "recent_FPR_min": float(config.get("fast_rescue_fpr_trigger", 0.35)),
            },
            "min_budget_for_activation": int(config["fast_rescue_min_fp_normals"]),
            "gates_a_theta_move": True,
            "code_reference": "_maybe_fast_upward_rescue",
            "note": ("disabled by default; additionally needs the FPR trigger, "
                     "which no budget count guarantees"),
        },
        {
            "mechanism": "balanced_rescue",
            "controller_name": "guarded_dual_anchor_v2_balanced_rescue",
            "enabled_by_default": bool(config["enable_balanced_rescue"]),
            "hard_count_gates": {
                "normal_anchor_buffer_min": int(config["balanced_min_normals"]),
                "defect_anchor_buffer_min": int(config["balanced_min_defects"]),
                "cooldown_steps": int(config["balanced_cooldown_steps"]),
            },
            "min_budget_for_activation": (int(config["balanced_min_normals"]) +
                                          int(config["balanced_min_defects"])),
            "gates_a_theta_move": True,
            "code_reference": ("_balanced_anchor_values / "
                               "_maybe_balanced_update_theta"),
            "note": ("disabled by default; the per-label fast trigger can fire "
                     "earlier (2 verified FP normals or 1 verified FN defect) but "
                     "the guarded objective still needs both anchor groups"),
        },
    ]


def sidecar_feasibility(source_path: Path, budgets: Sequence[int]) -> Dict[str, Any]:
    """Metadata-level activation feasibility for each candidate budget."""
    config = read_dataclass_defaults(source_path, SIDECAR_CONFIG_CLASS)
    missing = [key for key in REQUIRED_SIDECAR_KEYS if key not in config]
    if missing:
        raise DatasetProtocolError(
            f"{SIDECAR_CONFIG_CLASS} is missing expected fields: {missing}")
    for key, expected in DOCUMENTED_SIDECAR_VALUES.items():
        if int(config[key]) != int(expected):
            raise DatasetProtocolError(
                f"{SIDECAR_CONFIG_CLASS}.{key} is {config[key]}, documented value "
                f"is {expected}; the preflight refuses to reinterpret a silently "
                f"changed Sidecar threshold")

    mechanisms = sidecar_mechanisms(config)
    update_every = int(config["update_every_labeled"])
    recent_capacity = int(config["recent_capacity"])
    per_budget: Dict[str, Any] = {}
    for budget in sorted(int(b) for b in budgets):
        rows = []
        for mechanism in mechanisms:
            needs = int(mechanism["min_budget_for_activation"])
            if not mechanism["enabled_by_default"]:
                verdict = "disabled_by_default_no_activation_expected"
            elif budget < needs:
                verdict = "mathematically_impossible_under_this_budget"
            else:
                verdict = "potentially_feasible_depends_on_label_composition"
            rows.append({"mechanism": mechanism["mechanism"],
                         "min_budget_for_activation": needs,
                         "verdict": verdict})
        per_budget[str(budget)] = {
            "budget": budget,
            "update_opportunities_upper_bound": budget // update_every,
            "labels_retained_in_recent_window": min(budget, recent_capacity),
            "mechanisms": rows,
            "max_possible_normal_labels": budget,
            "max_possible_defect_labels": budget,
            "can_ever_reach_min_normal_anchor":
                budget >= int(config["min_normal_anchor"]),
            "can_ever_reach_min_defect_anchor":
                budget >= int(config["min_defect_anchor"]),
            "can_ever_satisfy_both_anchor_counts_simultaneously":
                budget >= (int(config["min_normal_anchor"]) +
                           int(config["min_defect_anchor"])),
            "actual_activation": "not_established_without_online_controller_execution",
        }

    return {
        "source_file": str(source_path.name),
        "source_class": SIDECAR_CONFIG_CLASS,
        "read_method": "python ast literal_eval of declared dataclass defaults",
        "declared_values": {key: config[key] for key in sorted(config)
                            if isinstance(config[key], (int, float, bool))},
        "documented_values_match_source": True,
        "thresholds_modified": False,
        "mechanisms": mechanisms,
        "declared_threshold_roles": {
            "min_normal_anchor": ("selects the robust quantile branch of "
                                  "_candidate_thetas; the fallback branch computes "
                                  "the same quantile whenever bufN > 0, so this "
                                  "value does not by itself gate activation"),
            "min_defect_anchor": ("same as above for bufA: it selects a branch, "
                                  "not a gate, in the current implementation"),
            "min_recent_total": "hard gate on the recent labelled window",
            "min_recent_per_class": "hard gate on both label classes in the window",
        },
        "update_every_labeled": update_every,
        "recent_capacity": recent_capacity,
        "per_budget": per_budget,
        "claim_boundary": [
            "crossing a global anchor-count threshold does not guarantee activation",
            "crossing min_normal_anchor or min_defect_anchor is not sufficient",
            "actual activation requires executing the online controller",
            "no Sidecar threshold was lowered to make a budget look feasible",
        ],
    }


# ---------------------------------------------------------------------------
#  Q-A budget preflight per class
# ---------------------------------------------------------------------------


def causal_visibility_audit(result: Any, methods: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify that each prediction saw only labels disclosed at earlier steps."""
    violations: List[Dict[str, Any]] = []
    for name in sorted(methods):
        delivered_positions = [int(row[0]) for row in result.deliveries[name]]
        for step, seen in methods[name].labels_visible_at_prediction:
            expected = sum(1 for position in delivered_positions if position < step)
            if int(seen) != expected:
                violations.append({"method": name, "step": int(step),
                                   "labels_visible": int(seen),
                                   "labels_legally_visible": int(expected)})
    return {
        "rule": ("a label may only be visible to a prediction made at a strictly "
                 "later stream position than the query that revealed it"),
        "N_violations": len(violations),
        "examples": violations[:5],
        "passed": not violations,
    }


def _activation_verdict(mechanisms: Sequence[Mapping[str, Any]]) -> str:
    """Classify one budget's activation possibility for the enabled mechanisms."""
    enabled = [m for m in mechanisms
               if m["verdict"] != "disabled_by_default_no_activation_expected"]
    if not enabled:
        return "no_enabled_mechanism_declared"
    if all(str(m["verdict"]).startswith("mathematically_impossible")
           for m in enabled):
        return "mathematically_impossible_activation"
    return "potentially_feasible_activation"


def budget_preflight(class_name: str, stream: Sequence[qp.StreamImage],
                     labels: Mapping[str, bool], budgets: Sequence[int],
                     seed: int, reference_budget: int) -> Dict[str, Any]:
    """Nested Q-A tapes plus offline accounting and a synthetic replay check."""
    tapes = qp.build_nested_tapes(stream, budgets, seed, class_name)
    nesting = qp.tape_nesting_report(tapes)
    if not nesting["all_nested"]:
        raise DatasetProtocolError(f"{class_name}: nested budgets are not prefixes")

    sidecar = sidecar_feasibility(ROOT / SIDECAR_SOURCE, budgets)
    per_budget: Dict[str, Any] = {}
    for budget in sorted(tapes):
        tape = tapes[budget]
        diagnostics = qp.label_diagnostics(stream, tape, labels)
        sidecar_row = sidecar["per_budget"][str(budget)]
        per_budget[str(budget)] = {
            "budget_requested": int(budget),
            "N_images": int(tape.n_images),
            "N_samples": int(tape.n_samples),
            "N_queries_selected": len(tape.events),
            "N_unique_queried_samples": len(set(tape.sample_keys)),
            "image_query_fraction": round(len(tape.events) / tape.n_images, 6),
            "unique_sample_audit_fraction":
                round(len(set(tape.sample_keys)) / tape.n_samples, 6),
            "query_seed": int(seed),
            "query_manifest_sha256": tape.digest,
            "offline_label_composition": diagnostics,
            "normal_memory_write_opportunities": diagnostics["N_normal_labels"],
            "defect_anchor_opportunities": diagnostics["N_defect_labels"],
            "sidecar_update_opportunities_upper_bound":
                sidecar_row["update_opportunities_upper_bound"],
            "sidecar_activation_verdict":
                _activation_verdict(sidecar_row["mechanisms"]),
            "sidecar_feasibility": {
                "mechanisms": sidecar_row["mechanisms"],
                "actual_activation": sidecar_row["actual_activation"],
                "can_ever_reach_min_normal_anchor":
                    sidecar_row["can_ever_reach_min_normal_anchor"],
                "can_ever_reach_min_defect_anchor":
                    sidecar_row["can_ever_reach_min_defect_anchor"],
                "can_ever_satisfy_both_anchor_counts_simultaneously":
                    sidecar_row["can_ever_satisfy_both_anchor_counts_simultaneously"],
            },
            "declared_anchor_counts": {
                "min_normal_anchor": DOCUMENTED_SIDECAR_VALUES["min_normal_anchor"],
                "min_defect_anchor": DOCUMENTED_SIDECAR_VALUES["min_defect_anchor"],
            },
            "observed_vs_declared_anchor_counts": {
                "normal_labels_minus_min_normal_anchor":
                    diagnostics["N_normal_labels"] -
                    DOCUMENTED_SIDECAR_VALUES["min_normal_anchor"],
                "defect_labels_minus_min_defect_anchor":
                    diagnostics["N_defect_labels"] -
                    DOCUMENTED_SIDECAR_VALUES["min_defect_anchor"],
                "note": ("expected label composition of THIS tape only; it is "
                         "not an activation measurement, and in the current "
                         "controller these values select a candidate branch "
                         "rather than gating activation"),
            },
        }

    methods = qp.default_simulated_methods()
    reference = tapes[int(reference_budget)]
    replay = qp.replay_shared_tape(stream, reference, labels, methods)
    parity = replay.parity_report()
    if not (parity["identical_feedback_events"] and
            parity["identical_feedback_order"]):
        raise DatasetProtocolError(
            f"{class_name}: shared-tape replay parity failed: {parity}")
    causal = causal_visibility_audit(replay, methods)

    return {
        "stream": {
            "N_images": len(stream),
            "N_samples": len({i.sample_key for i in stream}),
            "sha256": qp.stream_digest(stream),
            "order": ("phase I -> II -> III; inside a phase ordered by "
                      "(capture timestamp, camera id, path)"),
        },
        "budgets": per_budget,
        "nesting": nesting,
        "replay_parity_at_reference_budget": {
            "budget": int(reference_budget),
            **parity,
            "causal_label_visibility": causal,
        },
        "query_policy": qp.QUERY_POLICY,
    }, tapes


# ---------------------------------------------------------------------------
#  Payload assembly
# ---------------------------------------------------------------------------

REFERENCE_BUDGET = 79  # the traceable V0.2 reference, not a preferred budget


def configuration_block(split_config: SplitConfig,
                        budgets: Sequence[int], query_seed: int) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "identity_version": IDENTITY_VERSION,
        "grouping_unit": PRIMARY_GROUPING_UNIT,
        "sensitivity_grouping_unit": SENSITIVITY_GROUPING_UNIT,
        "classes": list(CLASSES),
        "support_seeds": [int(s) for s in split_config.support_seeds],
        "strict_four_shot_support_images": int(split_config.n_support),
        "mature_history_samples": int(split_config.n_history),
        "probe_normal_samples_per_probe": int(split_config.n_probe_normal),
        "probe_defect_samples_per_probe": int(split_config.n_probe_defect),
        "stream_phases": int(split_config.n_phases),
        "stream_samples_per_phase": int(split_config.groups_per_phase),
        "history_seed": int(split_config.history_seed),
        "probe_seed": int(split_config.probe_seed),
        "support_view_seed_offset": int(split_config.support_view_seed_offset),
        "query_policy": qp.QUERY_POLICY,
        "budget_candidates": [int(b) for b in budgets],
        "reference_budget_v02": REFERENCE_BUDGET,
        "query_seed": int(query_seed),
        "max_queries_per_sample": qp.MAX_QUERIES_PER_SAMPLE,
        "stream_order": ("phases I->II->III; within a phase ordered by "
                         "(capture timestamp, camera id, path)"),
        "labels_allowed_for": [
            "evaluation-only probe composition",
            "offline metadata diagnostics and query-composition reporting",
        ],
        "labels_forbidden_for": [
            "query-sample selection",
            "query-camera selection",
            "budget selection",
            "threshold or sidecar configuration",
        ],
        "gpu_used": False,
        "features_extracted": False,
        "anomaly_detection_metrics_computed": False,
    }


def identity_rule_block() -> Dict[str, Any]:
    """The three Real-IAD identifiers and the assumptions attached to each."""
    return {
        "official_sample_key": {
            "definition": "<category>_<NNNN>_<CONDITION>",
            "example_ok": "audiojack_0001_OK",
            "example_ng": "audiojack__0001_NG_ZW",
            "authority": ("https://github.com/Tencent/AnomalyDetection_Real-IAD "
                          "README section 3, --sample_key_pat "
                          "\"([a-zA-Z][a-zA-Z0-9_]*_[0-9]{4}_[A-Z][A-Z_]*[A-Z])"
                          "_C[0-9]_\""),
            "historical_equivalence": ("byte-identical to "
                                       "final_decision_core.object_id, verified "
                                       "on all released records of the 30 "
                                       "categories"),
            "role": "primary grouping unit; the official evaluation level",
            "assumption": ("distinct sample keys are distinct official samples; "
                           "the metadata does not claim they are distinct "
                           "manufactured specimens"),
        },
        "directory_serial_key": {
            "definition": "S#### directory name",
            "role": "identity-sensitivity diagnostic only",
            "not_a_sample_identity": True,
            "why": ("one S#### directory hosts the OK session plus one session per "
                    "defect code, captured at different times"),
            "assumption": ("never used for partitioning, query uniqueness or "
                           "leakage decisions in the primary protocol"),
        },
        "image_id": {
            "definition": "class-relative original image path",
            "role": "exact original image identity for manifests and digests",
        },
        "physical_specimen_identity": {
            "status": "undocumented",
            "note": ("whether two official sample keys sharing one S#### directory "
                     "are the same manufactured specimen cannot be established "
                     "from the released metadata; this is recorded as a residual "
                     "assumption, not as leakage"),
        },
        "camera_observation": ("one original image; (sample, capture session, "
                               "camera id, timestamp)"),
        "image_label": ("anomaly_class == OK for that single image; never "
                        "propagated to sibling views"),
        "query_unit": "one image-level label per official sample key per stream",
        "validated_by": [
            "official sample-key expression must match the file name",
            "structural regex with class/sample/state/defect/camera/timestamp",
            "sample directory S#### must agree with the numeric file token",
            "directory layout must agree with the encoded OK/NG session",
            "capture timestamp must be a valid YYYYMMDDHHMMSS",
            "no duplicate image path or file name within a class",
        ],
    }


def _tape_manifest_entry(class_name: str, stream: Sequence[qp.StreamImage],
                         plan: PartitionPlan, tapes: Mapping[int, qp.QueryTape],
                         query_seed: int) -> Dict[str, Any]:
    choices = qp.camera_choice_map(stream, query_seed, class_name)
    largest = tapes[max(tapes)]
    return {
        "identity_version": IDENTITY_VERSION,
        "grouping_unit": PRIMARY_GROUPING_UNIT,
        "stream": {
            "N_images": len(stream),
            "N_samples": len({i.sample_key for i in stream}),
            "sha256": qp.stream_digest(stream),
            "phases": [{"phase": i + 1, "N_samples": len(plan.phase_samples[i]),
                        "N_images": len(plan.phases[i])}
                       for i in range(len(plan.phases))],
            "label_blind_construction": ("StreamImage carries no label field; "
                                         "positions increase inside each phase "
                                         "block"),
        },
        "priority_sample_order": list(largest.priority_order),
        "camera_choice": {
            sample_key: {"camera_id": int(choices[sample_key].camera_id),
                         "image_path": str(choices[sample_key].rel_path)}
            for sample_key in sorted(choices)
        },
        "tapes": {str(budget): tapes[budget].manifest() for budget in sorted(tapes)},
        "nesting": qp.tape_nesting_report(tapes),
    }


def _dataset_summary(dataset_classes: Mapping[str, Any],
                     historical: Mapping[str, Any]) -> Dict[str, Any]:
    counts = [row["counts"] for row in dataset_classes.values()]
    return {
        "identity_version": IDENTITY_VERSION,
        "N_classes": len(dataset_classes),
        "total_train_images": sum(c["train_images"] for c in counts),
        "total_test_images": sum(c["test_images"] for c in counts),
        "total_train_official_samples":
            sum(c["train_official_samples"] for c in counts),
        "total_test_official_samples":
            sum(c["test_official_samples"] for c in counts),
        "total_train_directory_serials":
            sum(c["train_directory_serials"] for c in counts),
        "total_test_directory_serials":
            sum(c["test_directory_serials"] for c in counts),
        "total_train_anomalous_images": sum(c["train_anomalous_images"] for c in counts),
        "stream_samples_per_class": [row["partitions"]["stream"]["N_samples"]
                                     for row in dataset_classes.values()],
        "stream_images_per_class": [row["partitions"]["stream"]["N_images"]
                                    for row in dataset_classes.values()],
        "probe_samples_per_class": [row["partitions"]["probe_A"]["N_samples"] +
                                    row["partitions"]["probe_B"]["N_samples"]
                                    for row in dataset_classes.values()],
        "official_sample_overlap_per_class": {
            name: row["identity_comparison"]["train_test_overlap"]
            ["official_sample_key"]["N_shared"]
            for name, row in dataset_classes.items()},
        "directory_serial_overlap_per_class": {
            name: row["identity_comparison"]["train_test_overlap"]
            ["directory_serial_key"]["N_shared"]
            for name, row in dataset_classes.items()},
        "all_partitions_sample_disjoint": all(
            row["sample_disjointness"]["all_partitions_disjoint"]
            for row in dataset_classes.values()),
        "all_classes_train_test_sample_disjoint": all(
            row["identity_comparison"]["train_test_overlap"]
            ["official_sample_key"]["N_shared"] == 0
            for row in dataset_classes.values()),
        "all_classes_reproduce_historical_arithmetic": all(
            row["arithmetic_check"]["queries_equal_round_3pct_of_images"]
            for row in historical.values()),
    }


def _dataset_findings(dataset_classes: Mapping[str, Any],
                      historical: Mapping[str, Any]) -> List[Dict[str, Any]]:
    serial_overlap = {name: row["identity_comparison"]["train_test_overlap"]
                      ["directory_serial_key"]["N_shared"]
                      for name, row in dataset_classes.items()}
    sample_overlap = {name: row["identity_comparison"]["train_test_overlap"]
                      ["official_sample_key"]["N_shared"]
                      for name, row in dataset_classes.items()}
    collisions = {name: row["identity_comparison"]["serial_collisions"]
                  ["N_serials_hosting_more_than_one_official_sample"]
                  for name, row in dataset_classes.items()}
    historical_sample_leaks = {name: row["identity_sensitivity_audit"]
                               ["sample_level"]["leakage_detected"]
                               for name, row in historical.items()}
    historical_serial_leaks = {name: row["identity_sensitivity_audit"]
                               ["serial_level"]
                               ["max_train_serials_also_in_test_partitions"]
                               for name, row in historical.items()}
    mixed = {name: row["label_structure"]["test"]
             ["N_normal_labelled_images_inside_defect_samples"]
             for name, row in dataset_classes.items()}
    return [
        {
            "id": "official_sample_identity_is_not_the_directory_serial",
            "kind": "verified_parser_error",
            "finding": ("one S#### directory hosts up to five official samples "
                        "(OK plus one session per defect code), so grouping by "
                        "the directory merges distinct official samples"),
            "evidence": collisions,
            "consequence": ("the primary protocol groups by the official sample "
                            "key; the D1-A first-pass serial grouping is archived "
                            "as superseded and is not used"),
        },
        {
            "id": "train_test_disjoint_at_official_sample_level",
            "kind": "verified_identity_audit",
            "finding": ("the released Real-IAD train and test splits share no "
                        "official sample key"),
            "evidence": sample_overlap,
            "consequence": ("no reservation of train samples is required and the "
                            "historical 525-sample partition is reproducible"),
        },
        {
            "id": "serial_overlap_is_specimen_correlation_not_leakage",
            "kind": "possible_physical_specimen_correlation",
            "finding": ("53-93 S#### directories appear in both the train (OK "
                        "session) and test (defect sessions) parts of a class"),
            "evidence": serial_overlap,
            "consequence": ("documented as residual uncertainty about physical "
                            "specimen identity; NOT counted as evaluation leakage"),
        },
        {
            "id": "historical_v02_is_sample_disjoint",
            "kind": "verified_identity_audit",
            "finding": ("under the official sample key the historical V0.2 split "
                        "has no train/test sample overlap"),
            "evidence": historical_sample_leaks,
            "consequence": ("the D1-A first-pass claim that V0.2 leaked physical "
                            "samples was an artefact of the wrong identity rule; "
                            "recorded serial-level coincidence is "
                            f"{historical_serial_leaks}"),
        },
        {
            "id": "ng_sessions_contain_normal_labelled_views",
            "kind": "verified_dataset_structure",
            "finding": ("images from defective capture sessions are individually "
                        "labelled OK when that camera view shows no defect"),
            "evidence": mixed,
            "consequence": ("image-level labels are never propagated across "
                            "views; probes keep image-level truth"),
        },
    ]


def _budget_summary(budget_classes: Mapping[str, Any]) -> Dict[str, Any]:
    per_budget: Dict[str, Any] = {}
    budgets = sorted(int(b) for b in
                     next(iter(budget_classes.values()))["budgets"])
    for budget in budgets:
        rows = [row["budgets"][str(budget)] for row in budget_classes.values()]
        per_budget[str(budget)] = {
            "N_classes": len(rows),
            "image_query_fraction_range": [
                min(r["image_query_fraction"] for r in rows),
                max(r["image_query_fraction"] for r in rows)],
            "unique_sample_audit_fraction_range": [
                min(r["unique_sample_audit_fraction"] for r in rows),
                max(r["unique_sample_audit_fraction"] for r in rows)],
            "normal_label_range": [
                min(r["normal_memory_write_opportunities"] for r in rows),
                max(r["normal_memory_write_opportunities"] for r in rows)],
            "defect_label_range": [
                min(r["defect_anchor_opportunities"] for r in rows),
                max(r["defect_anchor_opportunities"] for r in rows)],
            "sidecar_activation_verdicts": sorted(
                {r["sidecar_activation_verdict"] for r in rows}),
        }
    return {
        "per_budget": per_budget,
        "all_classes_nested": all(row["nesting"]["all_nested"]
                                  for row in budget_classes.values()),
        "all_classes_query_parity": all(
            row["replay_parity_at_reference_budget"]["identical_feedback_events"]
            for row in budget_classes.values()),
        "all_classes_causal_visibility_passed": all(
            row["replay_parity_at_reference_budget"]["causal_label_visibility"]["passed"]
            for row in budget_classes.values()),
        "stream_images_total": sum(row["stream"]["N_images"]
                                   for row in budget_classes.values()),
        "stream_samples_total": sum(row["stream"]["N_samples"]
                                    for row in budget_classes.values()),
    }


def run_preflight(dataset_root: Any, json_dir: Any,
                  split_config: Optional[SplitConfig] = None,
                  budgets: Sequence[int] = qp.BUDGET_CANDIDATES,
                  query_seed: int = qp.QUERY_SEED
                  ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Compute the three metadata payloads.  Pure CPU, no model, no images."""
    split_config = split_config or SplitConfig()
    budgets = tuple(int(b) for b in budgets)
    if int(query_seed) != qp.QUERY_SEED:
        raise DatasetProtocolError(
            f"Query seed {query_seed} is not the registered Q-A seed "
            f"{qp.QUERY_SEED}; register a new seed in the protocol first")
    configuration = configuration_block(split_config, budgets, query_seed)
    sidecar = sidecar_feasibility(ROOT / SIDECAR_SOURCE, budgets)

    records_by_class: Dict[str, Any] = {}
    dataset_classes: Dict[str, Any] = {}
    budget_classes: Dict[str, Any] = {}
    tape_classes: Dict[str, Any] = {}
    historical: Dict[str, Any] = {}

    for class_name in CLASSES:
        train, test = load_class_records(dataset_root, json_dir, class_name)
        records_by_class[class_name] = (train, test)
        plan = build_partitions(train, test, split_config)
        dataset_classes[class_name] = class_diagnostics(
            class_name, train, test, plan, split_config)
        historical[class_name] = historical_reference_audit(
            train, test, split_config.support_seeds)
        stream = qp.build_stream(plan.phases)
        labels = {r.path: r.is_good for r in list(train) + list(test)}
        report, tapes = budget_preflight(
            class_name, stream, labels, budgets, int(query_seed), REFERENCE_BUDGET)
        budget_classes[class_name] = report
        tape_classes[class_name] = _tape_manifest_entry(
            class_name, stream, plan, tapes, int(query_seed))
        if not report["replay_parity_at_reference_budget"][
                "causal_label_visibility"]["passed"]:
            raise DatasetProtocolError(
                f"{class_name}: causal label-visibility audit failed")
        if report["stream"]["N_images"] != 525 * 5:
            raise DatasetProtocolError(
                f"{class_name}: expected the historical 2,625 stream images, got "
                f"{report['stream']['N_images']}")

    fingerprint = dataset_fingerprint(Path(json_dir), CLASSES, records_by_class)

    dataset_payload = {
        "schema": SCHEMA_DATASET,
        "stage": STAGE,
        "identity": identity_rule_block(),
        "configuration": configuration,
        "classes": dataset_classes,
        "historical_v02_reference": historical,
        "cross_class_summary": _dataset_summary(dataset_classes, historical),
        "known_dataset_findings": _dataset_findings(dataset_classes, historical),
        "supersedes": {
            "identity_version": "D1-A serial (S####) grouping, archived",
            "path": "outputs/risk_safe_sprint_20261027/"
                    "_superseded_d1a_preflight_serial_identity/",
            "reason": ("the first-pass parser treated the S#### directory as the "
                       "physical object, which merges the OK and NG_<defect> "
                       "official samples of one directory"),
        },
        "claim_boundaries": [
            "D1-A-R1 reports metadata feasibility only; no anomaly-detection "
            "metric and no admission result is computed",
            "primary grouping is the official Real-IAD sample key; the S#### "
            "serial analysis is a sensitivity diagnostic only",
            "one image-level label is charged per official sample key per stream",
            "physical manufactured-specimen identity is undocumented and remains "
            "a residual assumption",
        ],
    }
    return {"dataset_payload": dataset_payload,
            "budget_classes": budget_classes,
            "historical": historical,
            "tape_classes": tape_classes,
            "configuration": configuration,
            "sidecar": sidecar,
            "fingerprint": fingerprint}


def build_payloads(dataset_root: Any, json_dir: Any,
                   split_config: Optional[SplitConfig] = None,
                   budgets: Sequence[int] = qp.BUDGET_CANDIDATES,
                   query_seed: int = qp.QUERY_SEED
                   ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Assemble the three metadata payloads in one deterministic pass."""
    computed = run_preflight(dataset_root, json_dir, split_config, budgets,
                             query_seed)
    configuration = computed["configuration"]
    budget_classes = computed["budget_classes"]
    historical = computed["historical"]
    sidecar = computed["sidecar"]
    budgets = tuple(int(b) for b in budgets)

    budget_payload = {
        "schema": SCHEMA_BUDGET,
        "stage": STAGE,
        "configuration": configuration,
        "units": {
            "budget": "absolute image-level human labels per class",
            "image_query_fraction": "N_queries / N_images",
            "unique_sample_audit_fraction":
                "N_unique_queried_samples / N_samples",
            "grouping_unit": PRIMARY_GROUPING_UNIT,
            "human_verification_cost": "N_queries image-level confirmations",
        },
        "historical_accounting": {
            "verified_claim": ("525 groups x 5 views = 2,625 observations at a "
                               "3% image-level rate is 79 queries, i.e. 3.01% "
                               "image-query fraction and 15.05% of the 525 "
                               "legacy groups -- never ~16 queries"),
            "per_class": historical,
            "all_classes_reproduce_525_2625_79": all(
                value["arithmetic_check"]["N_stream_groups"] == [525] and
                value["arithmetic_check"]["N_stream_images"] == [2625] and
                value["arithmetic_check"]["N_queries"] == [79]
                for value in historical.values()),
            "denominator_warning": ("those denominators belong to the legacy "
                                    "session-grouped stream; the D1-A stream has "
                                    "a different sample and image denominator"),
        },
        "sidecar_feasibility": sidecar,
        "classes": budget_classes,
        "cross_class_summary": _budget_summary(budget_classes),
        "query_policy_interfaces": {
            "Q-A": {
                "policy_id": "Q-A",
                "status": "implemented_for_d1a",
                "contract": ("one pre-registered label-blind random tape per "
                             "class/stream/seed/budget, replayed unchanged to "
                             "every Admission method"),
            },
            "Q-B": qp.q_b_interface_contract(),
            "Q-C": qp.q_c_interface_contract(),
        },
        "claim_boundaries": [
            "no Admission algorithm is implemented in D1-A",
            "Q-B and Q-C are declared interfaces only",
            "an exceeded anchor count does not establish Sidecar activation",
            "budget feasibility is a metadata statement, not a performance claim",
        ],
    }

    tape_payload = {
        "schema": SCHEMA_TAPE,
        "stage": STAGE,
        "policy": {
            "query_policy": qp.QUERY_POLICY,
            "query_seed": int(query_seed),
            "budget_candidates": [int(b) for b in budgets],
            "selection": ("one fixed random permutation of official samples per "
                          "class/stream/seed; budget B takes the first B samples; "
                          "one camera view per sample drawn from its own RNG "
                          "identity, independent of B"),
            "label_blindness": ("the selector receives StreamImage records that "
                                "carry no label field"),
            "nesting": ("tape(B1) is a subset of tape(B2) for B1 < B2; every "
                        "budget is still replayed independently from the same "
                        "initial memory state"),
        },
        "classes": computed["tape_classes"],
        "cross_class_v02_reference_budget": {
            "budget": REFERENCE_BUDGET,
            "note": ("79 labels/class is the traceable V0.2 reference; it is not "
                     "a preferred or optimal budget"),
        },
    }

    payloads = {OUTPUT_DATASET: computed["dataset_payload"],
                OUTPUT_BUDGET: budget_payload,
                OUTPUT_TAPE: tape_payload}
    return payloads, configuration, computed["fingerprint"]


def build_source_manifest(configuration: Mapping[str, Any],
                          fingerprint: Mapping[str, Any],
                          payloads: Mapping[str, Any]) -> Dict[str, Any]:
    """Provenance record written last, covering code, data and outputs."""
    return {
        "schema": SCHEMA_SOURCE,
        "stage": STAGE,
        "git": git_state(),
        "environment": environment_block(),
        "sources_sha256": source_hashes(),
        "dataset": fingerprint,
        "configuration_sha256": qp.sha256_of(configuration),
        "outputs_sha256": {
            name: sha256_text(json_text(payloads[name]))
            for name in (OUTPUT_DATASET, OUTPUT_BUDGET, OUTPUT_TAPE)},
        "output_schemas": {
            OUTPUT_DATASET: SCHEMA_DATASET,
            OUTPUT_BUDGET: SCHEMA_BUDGET,
            OUTPUT_TAPE: SCHEMA_TAPE,
            OUTPUT_SOURCE: SCHEMA_SOURCE,
        },
        "registered_defaults": {
            "classes": list(CLASSES),
            "query_seed": qp.QUERY_SEED,
            "budget_candidates": list(qp.BUDGET_CANDIDATES),
            "reference_budget_v02": REFERENCE_BUDGET,
            "sidecar_source": SIDECAR_SOURCE,
        },
        "rerun_semantics": [
            "re-running with identical code, data and configuration validates "
            "and reuses the existing outputs",
            "any difference fails explicitly instead of overwriting evidence",
            "each manifest digest covers the exact serialised bytes",
        ],
        "authorisation_scope": [
            "D1-A only: metadata preflight; no GPU work, no feature extraction",
            "no git commit, push, merge or pull request",
            "D1-B and later milestones are explicitly out of scope",
        ],
    }


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def parse_int_list(text: Any) -> Tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(text).split(",") if part.strip())


def parser() -> argparse.ArgumentParser:
    argument = argparse.ArgumentParser(
        description="D1-A metadata-only Real-IAD dataset and query-budget preflight",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    argument.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    argument.add_argument("--json-dir", default=None,
                          help=f"default: <dataset-root>/{DEFAULT_JSON_SUBDIR}")
    argument.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    argument.add_argument("--verify-only", action="store_true",
                          help="validate existing outputs without writing")
    argument.add_argument("--support-seeds", default="0,1,2")
    argument.add_argument(
        "--budgets", default=",".join(str(b) for b in qp.BUDGET_CANDIDATES),
        help="absolute label budgets; the registered candidates are the default")
    argument.add_argument("--groups-per-phase", type=int, default=175,
                          help="stream samples per phase (historical V0.2 value)")
    return argument


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    dataset_root = Path(args.dataset_root)
    json_dir = (Path(args.json_dir) if args.json_dir
                else dataset_root / DEFAULT_JSON_SUBDIR)
    out_dir = Path(args.out_dir)
    split_config = SplitConfig(support_seeds=parse_int_list(args.support_seeds),
                               groups_per_phase=int(args.groups_per_phase))
    budgets = parse_int_list(args.budgets)

    payloads, configuration, fingerprint = build_payloads(
        dataset_root, json_dir, split_config, budgets)
    payloads[OUTPUT_SOURCE] = preserve_committed_source_provenance(
        out_dir / OUTPUT_SOURCE,
        build_source_manifest(configuration, fingerprint, payloads))

    statuses = {}
    for name in OUTPUT_FILES:
        target = out_dir / name
        statuses[name] = (verify_json_output(target, payloads[name])
                          if args.verify_only
                          else write_json_idempotent(target, payloads[name]))

    dataset = payloads[OUTPUT_DATASET]
    budget = payloads[OUTPUT_BUDGET]
    summary = {
        "stage": STAGE,
        "identity_version": IDENTITY_VERSION,
        "grouping_unit": PRIMARY_GROUPING_UNIT,
        "out_dir": str(out_dir).replace("\\", "/"),
        "output_status": statuses,
        "git_head": payloads[OUTPUT_SOURCE]["git"]["head"],
        "classes": {
            name: {
                "train_images": row["counts"]["train_images"],
                "test_images": row["counts"]["test_images"],
                "train_samples": row["counts"]["train_official_samples"],
                "test_samples": row["counts"]["test_official_samples"],
                "train_serials": row["counts"]["train_directory_serials"],
                "test_serials": row["counts"]["test_directory_serials"],
                "stream_samples": row["partitions"]["stream"]["N_samples"],
                "stream_images": row["partitions"]["stream"]["N_images"],
                "probe_samples": (row["partitions"]["probe_A"]["N_samples"] +
                                  row["partitions"]["probe_B"]["N_samples"]),
                "train_test_sample_overlap":
                    row["identity_comparison"]["train_test_overlap"]
                    ["official_sample_key"]["N_shared"],
                "historical_query_check":
                    budget["historical_accounting"]["per_class"][name]
                    ["arithmetic_check"]["N_queries"],
            }
            for name, row in dataset["classes"].items()
        },
        "historical_all_classes_525_2625_79":
            budget["historical_accounting"]["all_classes_reproduce_525_2625_79"],
        "sample_disjoint":
            dataset["cross_class_summary"]["all_partitions_sample_disjoint"],
        "train_test_sample_disjoint":
            dataset["cross_class_summary"]["all_classes_train_test_sample_disjoint"],
        "nested_budgets": budget["cross_class_summary"]["all_classes_nested"],
        "query_parity": budget["cross_class_summary"]["all_classes_query_parity"],
        "causal_visibility": budget["cross_class_summary"]
        ["all_classes_causal_visibility_passed"],
        "budget_ranges": budget["cross_class_summary"]["per_budget"],
        "no_gpu": True,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
#  Deterministic, idempotent JSON outputs
# ---------------------------------------------------------------------------


def json_text(payload: Any) -> str:
    """The one serialisation used for every D1-A output file."""
    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_idempotent(path: Path, payload: Any) -> str:
    """Create the file, or validate and reuse an identical existing file.

    Never overwrites: a difference between the preserved evidence and the
    current code/data/configuration raises :class:`OutputMismatch`, and a
    corrupt existing file raises instead of being replaced.
    """
    text = json_text(payload)
    if path.exists():
        existing_text = path.read_text(encoding="utf-8")
        try:
            existing = json.loads(existing_text)
        except json.JSONDecodeError as exc:
            raise OutputMismatch(
                f"{path} exists but is not valid JSON; refusing to overwrite "
                f"incomplete or corrupt evidence: {exc}") from exc
        if existing != json.loads(text):
            raise OutputMismatch(
                f"{path} differs from the current code/data/configuration; "
                f"refusing to overwrite preserved evidence")
        if sha256_text(existing_text) != sha256_text(text):
            # Semantically identical but byte-different (e.g. edited ordering).
            raise OutputMismatch(
                f"{path} is semantically equal but not byte-identical; "
                f"restore the original file or move it aside manually")
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    # newline="" keeps the exact LF bytes the digest was computed over; the
    # default Windows translation to CRLF would break byte-level manifest parity.
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    os.replace(temporary, path)
    return "created"


def _is_ancestor_commit(older: str, current: str) -> bool:
    """True only if Git proves a fast-forward ancestry relation."""
    if not older or not current:
        return False
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", older, current],
            cwd=str(ROOT), capture_output=True, check=False)
    except OSError:
        return False
    return result.returncode == 0


def preserve_committed_source_provenance(
        path: Path, new_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Retain the original run's Git snapshot after a later commit.

    Source bytes, dataset hashes, configuration and output hashes must STILL
    match exactly. Only the historical Git metadata can be pinned, and only
    when Git proves that the original run commit is an ancestor of HEAD.
    This permits committing the run's untracked code/output files without
    invalidating an otherwise byte-identical preflight. It never silently
    overwrites evidence after code, data or protocol changes.
    """
    prospective = dict(new_manifest)
    if not path.is_file():
        return prospective
    try:
        historical = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return prospective  # normal idempotent reader will fail closed
    if not isinstance(historical, dict):
        return prospective
    old_git = historical.get("git")
    new_git = prospective.get("git")
    if not isinstance(old_git, dict) or not isinstance(new_git, dict):
        return prospective
    if _is_ancestor_commit(str(old_git.get("head") or ""),
                           str(new_git.get("head") or "")):
        prospective["git"] = old_git
    return prospective


def verify_json_output(path: Path, payload: Any) -> str:
    """Read-only variant of :func:`write_json_idempotent` for ``--verify-only``."""
    if not path.exists():
        raise OutputMismatch(f"{path} does not exist")
    text = json_text(payload)
    existing_text = path.read_text(encoding="utf-8")
    if json.loads(existing_text) != json.loads(text):
        raise OutputMismatch(f"{path} differs from the current preflight result")
    if sha256_text(existing_text) != sha256_text(text):
        raise OutputMismatch(f"{path} is not byte-identical to the current result")
    return "verified"


# ---------------------------------------------------------------------------
#  Provenance helpers
# ---------------------------------------------------------------------------


def _run_git(*arguments: str) -> Optional[str]:
    try:
        completed = subprocess.run(["git", *arguments], cwd=str(ROOT),
                                   capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def git_state() -> Dict[str, Any]:
    """Record the actual HEAD/branch; never assume the expected commit."""
    branch = _run_git("branch", "--show-current")
    head = _run_git("rev-parse", "HEAD")
    dirty = _run_git("status", "--porcelain", "--untracked-files=no")
    return {
        "branch": branch,
        "head": head,
        "head_short": head[:12] if head else None,
        "tracked_modifications": sorted(
            line[3:] for line in (dirty or "").splitlines() if line.strip()),
        "git_available": branch is not None and head is not None,
    }


def source_hashes(paths: Sequence[str] = SOURCE_FILES) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    missing = []
    for relative in paths:
        target = ROOT / relative
        if not target.is_file():
            missing.append(relative)
            continue
        hashes[relative] = sha256_file(target)
    if missing:
        raise DatasetProtocolError(
            f"Cannot hash missing source file(s): {missing}")
    return hashes


def environment_block() -> Dict[str, Any]:
    block = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.system(),
        "torch_version": None,
        "device": "cpu-only metadata preflight",
        "cuda_initialised": False,
        "features_extracted": False,
    }
    try:
        import torch  # noqa: F401  (version only; no CUDA call, no tensor)
        block["torch_version"] = str(torch.__version__)
    except ImportError:  # pragma: no cover - documented in the protocol
        block["torch_version"] = "unavailable"
    return block


def dataset_fingerprint(json_dir: Path, classes: Sequence[str],
                        records_by_class: Mapping[str, Any]) -> Dict[str, Any]:
    files = {}
    for class_name in classes:
        candidate = json_dir / f"{class_name}.json"
        if not candidate.is_file():
            raise DatasetProtocolError(f"Missing Real-IAD JSON: {candidate}")
        files[candidate.name] = sha256_file(candidate)
    semantic = []
    for class_name in classes:
        train, test = records_by_class[class_name]
        for record in list(train) + list(test):
            semantic.append([class_name, record.split, record.rel_path,
                             int(record.is_good)])
    semantic.sort()
    return {
        "json_dir": str(json_dir).replace("\\", "/"),
        "json_files_sha256": files,
        "semantic_index_sha256": qp.sha256_of(semantic),
        "N_index_records": len(semantic),
    }


if __name__ == "__main__":
    raise SystemExit(main())
