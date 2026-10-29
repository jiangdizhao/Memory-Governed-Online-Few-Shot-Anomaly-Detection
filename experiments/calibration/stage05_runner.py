"""Run the preregistered Stage 0.5 exploratory audit in isolated phases.

The phase processes are intentionally separate:
``freeze`` never returns test labels; ``query-stats`` is evaluator-only;
``score`` discards test label and mask fields; ``evaluate`` opens test labels
only after all immutable native score tapes pass checksum verification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from . import stage05 as core
from .profiles import DEFAULT_PROFILE, canonical_json

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BRANCH = "calibration-stage0-contract-20261029"
EXPECTED_HEAD = "ec78c7b6901a4651153d0696cca7373e571f6b41"
DEFAULT_DATASET_ROOT = Path("G:/dataset/realiad")
DEFAULT_JSON_ROOT = DEFAULT_DATASET_ROOT / "jsons" / "realiad_jsons"
DEFAULT_OUTPUT = ROOT / "outputs/calibration/stage05_exploratory_20261029"
REFERENCE_PREFLIGHT = ROOT / "docs/calibration/audit/stage0_final_protocol_correction_v1/real_iad_metadata_preflight.json"

SOURCE_FILES = (
    "experiments/calibration/stage05.py",
    "experiments/calibration/stage05_runner.py",
    "experiments/calibration/query_tape.py",
    "experiments/calibration/profiles.py",
    "experiments/calibration/protocol.py",
    "experiments/calibration/native.py",
    "experiments/risk_safe/preflight.py",
    "experiments/p0_protocol_audit.py",
    "experiments/risk_safe/native_bridge.py",
    "test_phase3_AIF_simplified_v22.py",
    "vmb_visionad_new.py",
)


class RunnerError(RuntimeError):
    pass


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(payload)
    os.replace(temp, path)


def _file_sha(path: Path) -> str:
    return core.sha256_file(path)


def _source_hashes() -> dict[str, str]:
    return {relative: _file_sha(ROOT / relative) for relative in SOURCE_FILES}


def _check_environment() -> dict[str, Any]:
    import torch
    expected = Path(r"D:\anaconda3\envs\torch\python.exe").resolve()
    actual = Path(sys.executable).resolve()
    if os.name == "nt" and actual != expected:
        raise RunnerError(f"Wrong Python interpreter: {actual}; expected {expected}")
    return {"python_version": sys.version.split()[0],
            "python_executable_sha256": hashlib.sha256(str(actual).encode()).hexdigest(),
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": (torch.cuda.get_device_name(0)
                                 if torch.cuda.is_available() else None)}


def _git_info() -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    branch, head = run("branch", "--show-current"), run("rev-parse", "HEAD")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_HEAD, head],
                              cwd=ROOT, capture_output=True, text=True, check=False)
    if branch != EXPECTED_BRANCH or ancestor.returncode != 0:
        raise RunnerError(
            f"Expected {EXPECTED_BRANCH} at or descended from {EXPECTED_HEAD}; "
            f"found {branch}@{head}")
    return {"branch": branch, "source_head": head,
            "pinned_base_head": EXPECTED_HEAD}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_output(output: Path, *, create: bool, resume: bool) -> None:
    if create:
        if output.exists():
            if not resume:
                raise RunnerError(f"Output exists and is non-overwriting: {output}")
            manifest = output / "run_manifest.json"
            if not manifest.is_file():
                raise RunnerError("Cannot resume without run_manifest.json")
            prior = _read_json(manifest)
            if prior.get("schema") != "stage05.exploratory.run.v1":
                raise RunnerError("Existing output is not a Stage 0.5 audit")
        else:
            output.mkdir(parents=True)
            (output / "INCOMPLETE").write_text("Stage 0.5 audit is incomplete.\n",
                                               encoding="utf-8")
    elif not output.is_dir():
        raise RunnerError(f"Stage 0.5 output does not exist: {output}")


def _check_resume_identity(output: Path, dataset_root: Path, json_root: Path) -> dict[str, Any]:
    manifest = _read_json(output / "run_manifest.json")
    expected = {"dataset_root_sha256": hashlib.sha256(
                    str(dataset_root.resolve()).encode()).hexdigest(),
                "json_root_sha256": hashlib.sha256(
                    str(json_root.resolve()).encode()).hexdigest(),
                "source_hashes": _source_hashes()}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RunnerError(f"Resume provenance mismatch: {key}")
    return manifest


def _update_manifest(output: Path, *, phase: str, status: str,
                     extra: dict[str, Any] | None = None) -> None:
    path = output / "run_manifest.json"
    manifest = _read_json(path) if path.is_file() else {}
    manifest.setdefault("schema", "stage05.exploratory.run.v1")
    manifest.setdefault("phase_status", {})[phase] = status
    manifest["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if extra:
        manifest.update(extra)
    _atomic_json(path, manifest)
    _refresh_checksums(output)


def _refresh_checksums(output: Path) -> None:
    entries = {}
    for path in sorted(output.rglob("*")):
        if (path.is_file() and path.name not in {"checksums.json", "INCOMPLETE"}
                and not path.name.endswith(".tmp")):
            entries[path.relative_to(output).as_posix()] = _file_sha(path)
    _atomic_json(output / "checksums.json", {"files": entries,
                                              "algorithm": "SHA-256"})


def _verify_freeze(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = _read_json(output / "frozen_inputs.json")
    tapes_file = output / "query_tapes.json.gz"
    if _file_sha(tapes_file) != frozen["query_tapes_file_sha256"]:
        raise RunnerError("Frozen query tape checksum mismatch")
    with gzip.open(tapes_file, "rt", encoding="utf-8") as stream:
        tapes = json.load(stream)
    for class_name, info in frozen["classes"].items():
        stream_rows = [core.StreamImage(**row) for row in info["stream"]]
        tape_records = core.tape_records(stream_rows)
        for seed in core.QUERY_SEEDS:
            class_tapes = {float(rate): tape[str(seed)]
                           for rate, tape in tapes[class_name].items()}
            core.verify_nested_tapes(class_tapes, tape_records)
        seed37 = {f"{float(rate):.2f}": tape["37"]["digest"]
                  for rate, tape in tapes[class_name].items()}
        if seed37 != frozen["tape_digests_by_seed37"][class_name]:
            raise RunnerError(f"Seed 37 tape digest mismatch for {class_name}")
    return frozen, tapes


def _phase_freeze(output: Path, dataset_root: Path, json_root: Path,
                  resume: bool) -> None:
    _require_output(output, create=True, resume=resume)
    if not (output / "run_manifest.json").is_file():
        _atomic_json(output / "run_manifest.json", {
            "schema": "stage05.exploratory.run.v1", "phase_status": {},
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **_git_info(), "environment": _check_environment(),
            "source_hashes": _source_hashes(),
            "dataset_root_sha256": hashlib.sha256(
                str(dataset_root.resolve()).encode()).hexdigest(),
            "json_root_sha256": hashlib.sha256(
                str(json_root.resolve()).encode()).hexdigest(),
            "stage1_or_stage2_run": False,
        })
    if (output / "frozen_inputs.json").is_file():
        _check_resume_identity(output, dataset_root, json_root)
        _verify_freeze(output)
        print("freeze: verified existing immutable query/support manifest")
        return
    if not REFERENCE_PREFLIGHT.is_file():
        raise RunnerError(f"Pinned metadata-only baseline preflight missing: {REFERENCE_PREFLIGHT}")
    historical = _read_json(REFERENCE_PREFLIGHT)
    if historical.get("status") != "PASS":
        raise RunnerError("Pinned metadata-only baseline preflight is not PASS")
    prior_classes = {row["class"]: row for row in historical["classes"]}
    classes: dict[str, Any] = {}
    tapes_all: dict[str, Any] = {}
    query_digests_seed37: dict[str, Any] = {}
    for class_name in core.DEVELOPMENT_CLASSES:
        train, stream, json_sha = core.read_class_stream(
            dataset_root, json_root, class_name, test_labels=False)
        if len(stream) != core.EXPECTED_STREAM_LENGTHS[class_name]:
            raise RunnerError(
                f"{class_name}: official stream length {len(stream)} differs from "
                f"frozen {core.EXPECTED_STREAM_LENGTHS[class_name]}")
        prior = prior_classes.get(class_name)
        if (not prior or json_sha != prior.get("class_json_sha256") or
                len(stream) != prior.get("test_stream", {}).get("images")):
            raise RunnerError(f"{class_name}: current official metadata differs from pinned preflight")
        selected = core.select_supports(train)
        test_groups = {item.sample_key for item in stream}
        train_groups = {item.sample_key for item in train}
        if test_groups & train_groups:
            raise RunnerError(f"{class_name}: official train/test sample groups overlap")
        if (set(selected["c64"]["groups"]) & test_groups or
                any(set(item["groups"]) & test_groups
                    for item in selected["supports"].values())):
            raise RunnerError(f"{class_name}: test group entered train support/C64")
        records = core.tape_records(stream)
        class_tapes: dict[str, Any] = {}
        for seed in core.QUERY_SEEDS:
            by_rate = core.build_nested_tapes(records, seed)
            core.verify_nested_tapes(by_rate, records)
            for rate, tape in by_rate.items():
                class_tapes.setdefault(f"{rate:.2f}", {})[str(seed)] = tape
        if class_tapes["0.03"]["37"]["digest"] != prior["Q-A-TS"]["digest"]:
            raise RunnerError(f"{class_name}: seed-37 3% digest differs from frozen Q-A-TS")
        query_digests_seed37[class_name] = {
            rate: class_tapes[f"{rate:.2f}"]["37"]["digest"]
            for rate in core.QUERY_RATES}
        classes[class_name] = {
            "class_json_sha256": json_sha,
            "stream_images": len(stream),
            "stream_official_groups": len(test_groups),
            "train_normal_group_count": selected["normal_train_groups"],
            "c64": selected["c64"], "supports": selected["supports"],
            "support_c64_disjoint": selected["support_c64_disjoint"],
            "train_test_groups_disjoint": True,
            "prefix": str((core.read_json_sections(
                json_root / f"{class_name}.json", train_labels=False,
                test_labels=False)["meta"].get("prefix", ""))),
            "stream": [asdict(row) for row in stream],
            "labels_or_masks_returned_by_parser": False,
        }
        tapes_all[class_name] = class_tapes
        print(f"freeze: {class_name}, {len(stream)} events, 200 seeds, 3 nested rates")
    tape_bytes = canonical_json(tapes_all).encode("utf-8")
    tape_compressed = gzip.compress(tape_bytes, compresslevel=9, mtime=0)
    _atomic_bytes(output / "query_tapes.json.gz", tape_compressed)
    frozen = {
        "schema": "stage05.frozen_inputs.v1",
        "classes": classes,
        "query_seeds": list(core.QUERY_SEEDS),
        "query_rates": list(core.QUERY_RATES),
        "query_tape_count": len(core.DEVELOPMENT_CLASSES) * len(core.QUERY_SEEDS) * len(core.QUERY_RATES),
        "query_tapes_file_sha256": _file_sha(output / "query_tapes.json.gz"),
        "tape_digests_by_seed37": query_digests_seed37,
        "query_builder_source_hashes": _source_hashes(),
        "pinned_stage0_preflight_sha256": _file_sha(REFERENCE_PREFLIGHT),
        "test_labels_accessed": False,
        "test_masks_accessed": False,
        "support_train_labels_used_only_for_normal_train_selection": True,
        "stage1_or_stage2_run": False,
    }
    _atomic_json(output / "frozen_inputs.json", frozen)
    env = _check_environment()
    identity = _git_info()
    _atomic_json(output / "data_exposure_ledger.json", {
        "schema": "stage05.data_exposure.v1",
        "authorized_scope": list(core.DEVELOPMENT_CLASSES),
        "confirmation_classes": "NOT_ACCESSED",
        "test_label_status": "NOT_ACCESSED",
        "train_label_status": "ACCESSED_FOR_NORMAL_SUPPORT_AND_C64_SELECTION",
        "test_mask_status": "NOT_ACCESSED",
        "test_image_status": "NOT_ACCESSED",
        "support_and_query_tapes_frozen_first": True,
        "query_tape_file_sha256": frozen["query_tapes_file_sha256"],
        "support_and_c64_identity_digests": {
            name: {"c64_group_sha256": item["c64"]["group_sha256"],
                  "c64_image_sha256": item["c64"]["image_sha256"],
                  "support_group_sha256_by_seed": {
                      seed: row["group_sha256"]
                      for seed, row in item["supports"].items()},
                  "support_image_sha256_by_seed": {
                      seed: row["image_sha256"]
                      for seed, row in item["supports"].items()}}
            for name, item in classes.items()},
        "notice": "Stage 0.5 user authorization does not authorize Stage 1/2 or Mint/PCB access.",
    })
    manifest = {
        "schema": "stage05.exploratory.run.v1", "phase_status": {"freeze": "PASS"},
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **identity, "environment": env, "source_hashes": _source_hashes(),
        "dataset_root_sha256": hashlib.sha256(str(dataset_root.resolve()).encode()).hexdigest(),
        "json_root_sha256": hashlib.sha256(str(json_root.resolve()).encode()).hexdigest(),
        "input_metadata_sha256_by_class": {
            name: info["class_json_sha256"] for name, info in classes.items()},
        "expected_stream_lengths": core.EXPECTED_STREAM_LENGTHS,
        "stage1_or_stage2_run": False,
    }
    _atomic_json(output / "run_manifest.json", manifest)
    _refresh_checksums(output)
    print(f"freeze: PASS; {frozen['query_tape_count']} tapes; labels not consulted")


def _percentile_fields(values: list[float]) -> dict[str, Any]:
    import numpy as np
    array = np.asarray(values, dtype=np.float64)
    return {"minimum": float(array.min()), "maximum": float(array.max()),
            "quantiles_linear": {str(q): float(np.quantile(array, q))
                                 for q in (.05, .25, .5, .75, .95)},
            "mean": float(array.mean()),
            "sample_standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0}


def _phase_query_stats(output: Path, dataset_root: Path, json_root: Path,
                       resume: bool) -> None:
    _require_output(output, create=False, resume=resume)
    frozen, tapes = _verify_freeze(output)
    _check_resume_identity(output, dataset_root, json_root)
    target = output / "query_schedule_aggregates.json.gz"
    if target.is_file():
        expected = _read_json(output / "query_stats_manifest.json")
        if _file_sha(target) != expected["query_schedule_aggregates_sha256"]:
            raise RunnerError("Existing query aggregates fail their checksum")
        print("query-stats: verified existing immutable evaluator output")
        return
    aggregate: dict[str, Any] = {}
    seed_rows: list[dict[str, Any]] = []
    ledger_path = output / "data_exposure_ledger.json"
    ledger = _read_json(ledger_path)
    ledger["test_label_status"] = "ACCESS_IN_PROGRESS_FOR_STAGE05_EXPLORATORY_AGGREGATES"
    ledger["test_label_access_started_at_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ledger["test_label_source_sha256_by_class"] = {}
    ledger["test_labels_accessed_classes"] = []
    _atomic_json(ledger_path, ledger)
    for class_name, info in frozen["classes"].items():
        stream = [core.StreamImage(**row) for row in info["stream"]]
        labels = core.test_labels_for_class(json_root, class_name)
        ledger["test_label_source_sha256_by_class"][class_name] = info["class_json_sha256"]
        ledger["test_labels_accessed_classes"].append(class_name)
        _atomic_json(ledger_path, ledger)
        if len(labels) != len(stream):
            raise RunnerError(f"{class_name}: test labels/metadata count mismatch")
        aggregate[class_name] = {}
        by_seed_rate: dict[float, list[dict[str, Any]]] = {rate: [] for rate in core.QUERY_RATES}
        for rate in core.QUERY_RATES:
            rate_rows = []
            for seed in core.QUERY_SEEDS:
                tape = tapes[class_name][f"{rate:.2f}"][str(seed)]
                row = core.query_schedule_metrics(stream, labels, tape)
                rate_rows.append(row)
                seed_rows.append({"class": class_name, **row})
                by_seed_rate[rate].append(row)
            summary = core.summarize_query_schedules(rate_rows)
            # Aggregate group/camera/bin and temporal outcomes over the same
            # preregistered 200 finite query schedules.
            for field in ("queried_normals", "queried_defects",
                          "longest_consecutive_image_span_without_queried_defect"):
                summary[field + "_distribution"] = _percentile_fields(
                    [float(row[field]) for row in rate_rows])
            summary["queried_defect_fraction_distribution"] = _percentile_fields(
                [float(row["queried_defect_fraction"] or 0.0) for row in rate_rows])
            summary["camera_count_mean_by_id"] = {
                str(camera): sum(row["query_counts_by_camera"].get(str(camera), 0)
                                 for row in rate_rows) / len(rate_rows)
                for camera in range(1, 6)}
            summary["temporal_bin_count_mean"] = {
                str(bin_index): sum(row["query_counts_by_temporal_bin"].get(
                    str(bin_index), 0) for row in rate_rows) / len(rate_rows)
                for bin_index in range(16)}
            summary["selected_groups_by_available_camera_views_mean"] = {
                str(views): sum(row["selected_groups_by_available_camera_views"].get(
                    str(views), 0) for row in rate_rows) / len(rate_rows)
                for views in sorted({int(k) for row in rate_rows
                                     for k in row["selected_groups_by_available_camera_views"]})}
            summary["first_400_queried_defects_distribution"] = _percentile_fields(
                [float(row["first_400_queries"]["defects"]) for row in rate_rows])
            summary["sliding_256_window_minimum_defects_distribution"] = _percentile_fields(
                [float(row["sliding_256_image_windows"]["queried_defect_min"] or 0)
                 for row in rate_rows])
            summary["sliding_256_window_maximum_defects_distribution"] = _percentile_fields(
                [float(row["sliding_256_image_windows"]["queried_defect_max"] or 0)
                 for row in rate_rows])
            summary["full_stream_composition"] = {
                "images": len(stream), "normal_images": len(stream) - sum(labels),
                "defect_images": int(sum(labels)),
                "defect_fraction": float(sum(labels) / len(labels)),
                "official_sample_groups": info["stream_official_groups"],
                "images_by_group_view_count": dict(sorted(Counter(
                    str(row.view_count) for row in stream).items())),
            }
            aggregate[class_name][f"{rate:.2f}"] = summary
        aggregate[class_name]["paired_rate_differences"] = {}
        for low, high in ((0.03, 0.05), (0.05, 0.07), (0.03, 0.07)):
            low_map = {row["seed"]: row["queried_defects"] for row in by_seed_rate[low]}
            high_map = {row["seed"]: row["queried_defects"] for row in by_seed_rate[high]}
            differences = [float(high_map[seed] - low_map[seed]) for seed in core.QUERY_SEEDS]
            aggregate[class_name]["paired_rate_differences"][f"{low:.0%}_to_{high:.0%}"] = _percentile_fields(differences)
        print(f"query-stats: {class_name}; full-stream labels joined after tape freeze")
    detail = {"schema": "stage05.query_schedule_aggregates.v1",
              "classes": aggregate, "per_seed_rows": seed_rows,
              "labels_exposed": True,
              "seed_summaries_are_finite_tape_sensitivity_not_probability_guarantees": True}
    payload = gzip.compress(canonical_json(detail).encode("utf-8"), compresslevel=9, mtime=0)
    _atomic_bytes(target, payload)
    ledger.update({
        "test_label_status": "ACCESSED_FOR_STAGE05_EXPLORATORY_AGGREGATES",
        "test_label_accessed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_label_source_sha256_by_class": {
            name: info["class_json_sha256"] for name, info in frozen["classes"].items()},
        "label_use": ["query-composition counts", "full-stream class composition",
                      "temporal/query-seed sensitivity summaries"],
        "no_masks_or_confirmation_outcomes_accessed": True,
        "blindness_consequence": "Four development-class test outcomes are exposed exploratory information and cannot be described as unseen for future method development.",
    })
    _atomic_json(ledger_path, ledger)
    _atomic_json(output / "query_stats_manifest.json", {
        "status": "PASS", "query_schedule_aggregates_sha256": _file_sha(target),
        "row_count": len(seed_rows), "labels_joined_after_frozen_tape_sha256": frozen[
            "query_tapes_file_sha256"], "class_label_count_by_class": {
                name: len(core.test_labels_for_class(json_root, name))
                for name in core.DEVELOPMENT_CLASSES},
        "raw_per_image_labels_saved": False, "masks_opened": False,
    })
    _update_manifest(output, phase="query_stats", status="PASS",
                     extra={"test_labels_exposed": True,
                            "test_masks_opened": False,
                            "confirmation_classes_opened": False})
    print("query-stats: PASS; 2,400 query schedules summarized")


def _source_scorer_config() -> dict[str, Any]:
    return asdict(DEFAULT_PROFILE.scorer)


def _score_one_class(output: Path, dataset_root: Path, json_root: Path,
                     class_name: str, frozen_info: dict[str, Any], resume: bool) -> dict[str, Any]:
    import numpy as np
    import torch
    from experiments.p0_protocol_audit import (Config, build_vmb,
        encode_query, encode_reference_features, require_sources, score_encoded)
    from experiments.risk_safe.native_bridge import native_score
    from experiments.calibration.native import isolate_native_bank_k256

    seed_order = core.SUPPORT_SEEDS
    partial_path = output / "score_partial" / f"{class_name}.npz"
    score_dir = output / "score_tapes"
    score_dir.mkdir(exist_ok=True)
    score_files = {seed: score_dir / f"{class_name}_support{seed}.npz"
                   for seed in seed_order}
    if all(path.is_file() for path in score_files.values()):
        entries = {}
        for seed, path in score_files.items():
            entries[str(seed)] = {"file": path.relative_to(output).as_posix(),
                                  "sha256": _file_sha(path)}
        return {"class": class_name, "status": "PASS", "score_tapes": entries,
                "resumed_from_complete_class": True}

    train, stream, json_sha = core.read_class_stream(
        dataset_root, json_root, class_name, test_labels=False)
    if json_sha != frozen_info["class_json_sha256"]:
        raise RunnerError(f"{class_name}: source JSON changed after tape freeze")
    support_plan = core.select_supports(train)
    if (support_plan["c64"] != frozen_info["c64"] or
            support_plan["supports"] != frozen_info["supports"] or
            support_plan["normal_train_groups"] != frozen_info["train_normal_group_count"] or
            support_plan["support_c64_disjoint"] != frozen_info["support_c64_disjoint"]):
        raise RunnerError(f"{class_name}: train-normal support selection changed")
    prefix = frozen_info["prefix"]
    by_rel = {row.relative_id: row for row in train}
    all_support_rel = sorted(set(frozen_info["c64"]["images"]) |
                             {item for row in frozen_info["supports"].values()
                              for item in row["images"]})
    train_by_rel = {row.relative_id: row for row in train}
    if not set(all_support_rel) <= set(train_by_rel):
        raise RunnerError(f"{class_name}: support/C64 image is absent from train split")
    if not all(train_by_rel[relative].is_good for relative in all_support_rel):
        raise RunnerError(f"{class_name}: non-normal training image entered support/C64")
    abs_path = lambda relative: core._absolute_image_path(dataset_root, prefix, relative)
    support_paths_by_seed = {
        seed: [abs_path(relative) for relative in frozen_info["supports"][str(seed)]["images"]]
        for seed in seed_order}
    c64_paths = [abs_path(relative) for relative in frozen_info["c64"]["images"]]
    test_paths = [item.image_path for item in stream]
    for path in all_support_rel:
        if Path(abs_path(path)).resolve() in {Path(p).resolve() for p in test_paths}:
            raise RunnerError(f"{class_name}: test image entered support/C64")
    scorer = DEFAULT_PROFILE.scorer
    cfg = Config(
        device="cuda", augmentation_seed=scorer.augmentation_seed,
        bank_seed=scorer.bank_seed, fixed_capacity=scorer.fixed_capacity,
        fixed_per_image=scorer.fixed_per_image,
        augmentations=scorer.augmentations,
        include_identity=scorer.include_identity, multiplier=scorer.multiplier,
        feature_fuse=scorer.feature_fuse, topk_patches=scorer.topk_patches,
        image_aggregation=scorer.image_aggregation, image_topk=scorer.image_topk,
        image_quantile=scorer.image_quantile, knn=scorer.knn)
    impl = require_sources()
    impl.set_seed(0)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if not torch.cuda.is_available():
        raise RunnerError("CUDA is unavailable; native Stage 0.5 scoring is BLOCKED")
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
    if not checkpoint.is_file():
        raise RunnerError("Cached original DINOv2 checkpoint is unavailable; network download disabled")
    support_rel_by_path = {abs_path(relative): relative for relative in all_support_rel}
    backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
    # One frozen feature extraction per distinct train support/C64 image. Overlap
    # across approved seeds is legal and encoded once.
    cache = encode_reference_features(impl, backbone,
                                      sorted(support_rel_by_path), cfg)
    native_banks, fixed_hashes = {}, {}
    for seed in seed_order:
        paths = support_paths_by_seed[seed]
        bank = build_vmb(impl, cache, paths, cfg)
        native_banks[seed], fixed_hashes[seed] = isolate_native_bank_k256(bank)
    n = len(test_paths)
    c64_scores = np.full((len(seed_order), len(c64_paths)), np.nan, dtype=np.float64)
    test_scores = np.full((len(seed_order), n), np.nan, dtype=np.float64)
    start_index = 0
    parity = {}
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and partial_path.is_file():
        with np.load(partial_path, allow_pickle=False) as prior:
            prior_config = str(prior["config_sha256"].item())
            if prior_config != DEFAULT_PROFILE.scientific_sha256:
                raise RunnerError(f"{class_name}: partial score checkpoint config mismatch")
            c64_scores[:] = prior["c64_scores"]
            test_scores[:] = prior["test_scores"]
            start_index = int(prior["stream_index"].item())
            parity = json.loads(str(prior["parity_json"].item()))
        if start_index < 0 or start_index > n:
            raise RunnerError(f"{class_name}: invalid partial score checkpoint position")
        if start_index and not np.isfinite(test_scores[:, :start_index]).all():
            raise RunnerError(f"{class_name}: partial score checkpoint has missing rows")
    started = time.perf_counter()
    if not np.isfinite(c64_scores).all():
        for c_index, path in enumerate(c64_paths):
            image, features = encode_query(impl, backbone, path, cfg)
            for seed_index, seed in enumerate(seed_order):
                c64_scores[seed_index, c_index] = score_encoded(
                    impl, native_banks[seed], image, features, cfg)
    if start_index == 0:
        torch.cuda.reset_peak_memory_stats()
    last_saved = start_index
    for index in range(start_index, n):
        image, features = encode_query(impl, backbone, test_paths[index], cfg)
        for seed_index, seed in enumerate(seed_order):
            value = score_encoded(impl, native_banks[seed], image, features, cfg)
            test_scores[seed_index, index] = value
            if index == 0:
                direct, _ = native_score(impl, native_banks[seed], image, features, cfg)
                delta = abs(float(direct) - float(value))
                parity[str(seed)] = {"direct_native_vs_p0_score_encoded_abs_delta": delta,
                                     "tolerance": 1e-8, "passed": delta <= 1e-8}
                if delta > 1e-8:
                    raise RunnerError(f"{class_name} seed {seed}: native scorer parity failed")
        if (index + 1) % 256 == 0 or index + 1 == n:
            _write_partial_score(partial_path, c64_scores, test_scores,
                                 index + 1, parity)
            last_saved = index + 1
            print(f"score: {class_name} {index + 1}/{n} images; five native banks")
    if not np.isfinite(c64_scores).all() or not np.isfinite(test_scores).all():
        raise RunnerError(f"{class_name}: nonfinite or incomplete score tape")
    elapsed = time.perf_counter() - started
    memory_bytes = int(torch.cuda.max_memory_allocated())
    entries = {}
    for row_index, seed in enumerate(seed_order):
        target = score_files[seed]
        if target.exists():
            raise RunnerError(f"Refusing to overwrite frozen score tape: {target}")
        buffer = tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False)
        temp = Path(buffer.name)
        buffer.close()
        try:
            np.savez_compressed(temp, support_seed=np.asarray(seed),
                                test_scores=test_scores[row_index],
                                c64_scores=c64_scores[row_index],
                                test_identity_sha256=np.asarray(core.opaque_digest(
                                    "test-stream-identities", [row.relative_id for row in stream])),
                                c64_identity_sha256=np.asarray(frozen_info["c64"]["image_sha256"]))
            os.replace(temp, target)
        finally:
            if temp.exists():
                temp.unlink()
        entries[str(seed)] = {
            "file": target.relative_to(output).as_posix(), "sha256": _file_sha(target),
            "score_count": len(test_scores[row_index]), "c64_score_count": 64,
            "fixed_support_sha256": fixed_hashes[seed],
            "support_group_sha256": frozen_info["supports"][str(seed)]["group_sha256"],
            "c64_group_sha256": frozen_info["c64"]["group_sha256"],
            "native_parity": parity[str(seed)],
        }
    del backbone
    torch.cuda.empty_cache()
    return {"class": class_name, "status": "PASS", "score_tapes": entries,
            "score_path": "native_v22_dinov2_only_dynamic_greedy_empty_disabled",
            "shared_feature_extraction_count": len(c64_paths) + n,
            "test_images_scored": n, "c64_images_scored": len(c64_paths),
            "support_features_encoded": len(support_rel_by_path),
            "elapsed_seconds": elapsed,
            "peak_cuda_allocated_bytes": memory_bytes,
            "partial_checkpoint_last_saved_index": last_saved,
            "theta0_not_used_during_score_generation": True}


def _write_partial_score(path: Path, c64_scores, test_scores, stream_index: int,
                         parity: dict[str, Any]) -> None:
    import numpy as np
    buffer = tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False)
    temp = Path(buffer.name)
    buffer.close()
    try:
        np.savez_compressed(temp, c64_scores=c64_scores, test_scores=test_scores,
                            stream_index=np.asarray(stream_index),
                            config_sha256=np.asarray(DEFAULT_PROFILE.scientific_sha256),
                            parity_json=np.asarray(canonical_json(parity)))
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _phase_score(output: Path, dataset_root: Path, json_root: Path,
                 resume: bool) -> None:
    _require_output(output, create=False, resume=resume)
    _check_resume_identity(output, dataset_root, json_root)
    frozen, _ = _verify_freeze(output)
    score_manifest_path = output / "score_tape_manifest.json"
    existing = _read_json(score_manifest_path) if score_manifest_path.is_file() else {
        "schema": "stage05.native_score_tapes.v1", "classes": {}}
    for class_name, info in frozen["classes"].items():
        result = _score_one_class(output, dataset_root, json_root, class_name,
                                  info, resume=resume)
        existing["classes"][class_name] = result
        _atomic_json(score_manifest_path, existing)
        _update_manifest(output, phase="score", status="INCOMPLETE",
                         extra={"scored_class_count": len(existing["classes"])})
    status = ("PASS" if all(row.get("status") == "PASS"
                             for row in existing["classes"].values()) and
              set(existing["classes"]) == set(core.DEVELOPMENT_CLASSES)
              else "BLOCKED")
    existing.update({
        "status": status,
        "support_seeds": list(core.SUPPORT_SEEDS),
        "scorer_config": _source_scorer_config(),
        "scorer_config_sha256": DEFAULT_PROFILE.scientific_sha256,
        "dynamic_memory": "empty_and_disabled",
        "controller_or_test_label_access": False,
        "scoring_source_hashes": _source_hashes(),
    })
    _atomic_json(score_manifest_path, existing)
    _update_manifest(output, phase="score", status=status,
                     extra={"score_tape_manifest_sha256": _file_sha(score_manifest_path)})
    if status != "PASS":
        raise RunnerError("One or more native score tapes are incomplete")
    print("score: PASS; all twenty score tapes frozen and checksummed")


def _phase_query_windows(output: Path, dataset_root: Path, json_root: Path,
                         resume: bool) -> None:
    import csv
    from io import StringIO
    _require_output(output, create=False, resume=resume)
    _check_resume_identity(output, dataset_root, json_root)
    frozen, tapes = _verify_freeze(output)
    target = output / "query_window_sensitivity.csv.gz"
    manifest_path = output / "query_window_sensitivity_manifest.json"
    if target.is_file():
        manifest = _read_json(manifest_path)
        if _file_sha(target) != manifest.get("sha256"):
            raise RunnerError("Existing query-window aggregate checksum mismatch")
        print("query-windows: verified existing immutable aggregate")
        return
    columns = ["class", "rate", "window_start", "window_end_exclusive",
               "seed37_queried_defects", "minimum_across_200_seeds",
               "p05_across_200_seeds", "p25_across_200_seeds",
               "median_across_200_seeds", "p75_across_200_seeds",
               "p95_across_200_seeds", "maximum_across_200_seeds",
               "mean_across_200_seeds", "sample_sd_across_200_seeds"]
    text_stream = StringIO()
    writer = csv.DictWriter(text_stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    labels_exposed = {}
    for class_name, info in frozen["classes"].items():
        labels = core.test_labels_for_class(json_root, class_name)
        labels_exposed[class_name] = len(labels)
        stream = [core.StreamImage(**row) for row in info["stream"]]
        if len(labels) != len(stream):
            raise RunnerError(f"{class_name}: labels do not match frozen query-window stream")
        for rate in core.QUERY_RATES:
            tapes_by_seed = {seed: tapes[class_name][f"{rate:.2f}"][str(seed)]
                             for seed in core.QUERY_SEEDS}
            rows = core.query_window_sensitivity(labels, tapes_by_seed, window=256)
            for row in rows:
                writer.writerow({
                    "class": class_name, "rate": f"{rate:.2f}",
                    "window_start": row["window_start"],
                    "window_end_exclusive": row["window_end_exclusive"],
                    "seed37_queried_defects": row["seed37_queried_defects"],
                    "minimum_across_200_seeds": row["minimum_across_seeds"],
                    "p05_across_200_seeds": row["quantiles_linear"]["0.05"],
                    "p25_across_200_seeds": row["quantiles_linear"]["0.25"],
                    "median_across_200_seeds": row["quantiles_linear"]["0.5"],
                    "p75_across_200_seeds": row["quantiles_linear"]["0.75"],
                    "p95_across_200_seeds": row["quantiles_linear"]["0.95"],
                    "maximum_across_200_seeds": row["maximum_across_seeds"],
                    "mean_across_200_seeds": row["mean_across_seeds"],
                    "sample_sd_across_200_seeds": row["sample_sd_across_seeds"],
                })
        print(f"query-windows: {class_name}; per-position summaries across 200 tapes")
    payload = gzip.compress(text_stream.getvalue().encode("utf-8"),
                            compresslevel=9, mtime=0)
    _atomic_bytes(target, payload)
    _atomic_json(manifest_path, {
        "schema": "stage05.query_window_sensitivity.v1", "status": "PASS",
        "sha256": _file_sha(target), "window_size": 256, "stride": 1,
        "rows": sum(max(0, n - 256 + 1) for n in labels_exposed.values()) * 3,
        "class_label_counts": labels_exposed,
        "summaries_are_across_fixed_finite_seed_set": True,
        "per_image_labels_saved": False,
    })
    ledger_path = output / "data_exposure_ledger.json"
    ledger = _read_json(ledger_path)
    ledger["query_window_label_analysis"] = {
        "status": "COMPLETED_AFTER_TAPE_FREEZE",
        "class_label_counts": labels_exposed,
        "window_summaries_sha256": _file_sha(target),
        "output_is_aggregate_only": True,
    }
    _atomic_json(ledger_path, ledger)
    _update_manifest(output, phase="query_windows", status="PASS",
                     extra={"query_window_sensitivity_sha256": _file_sha(target)})
    print(f"query-windows: PASS; {sum(max(0,n-255) for n in labels_exposed.values())*3} rows")


def _phase_evaluate(output: Path, dataset_root: Path, json_root: Path,
                    resume: bool) -> None:
    import numpy as np
    _require_output(output, create=False, resume=resume)
    _check_resume_identity(output, dataset_root, json_root)
    if ((output / "static_score_metrics.json").is_file() and
            (output / "run_manifest.json").is_file()):
        prior = _read_json(output / "run_manifest.json")
        if prior.get("phase_status", {}).get("evaluate") == "PASS":
            if _file_sha(output / "static_score_metrics.json") != prior.get(
                    "static_score_metrics_sha256"):
                raise RunnerError("Existing evaluator output checksum mismatch")
            print("evaluate: verified existing immutable evaluator output")
            return
    frozen, _ = _verify_freeze(output)
    score_manifest = _read_json(output / "score_tape_manifest.json")
    if score_manifest.get("status") != "PASS":
        raise RunnerError("Cannot evaluate because score phase is not complete")
    score_rows = []
    window_rows = []
    ledger_path = output / "data_exposure_ledger.json"
    ledger = _read_json(ledger_path)
    ledger["static_score_label_access_status"] = "ACCESS_IN_PROGRESS_AFTER_SCORE_FREEZE"
    ledger["static_score_evaluation_labels_accessed_classes"] = []
    ledger["score_tape_manifest_sha256"] = _file_sha(output / "score_tape_manifest.json")
    _atomic_json(ledger_path, ledger)
    for class_name, info in frozen["classes"].items():
        labels = core.test_labels_for_class(json_root, class_name)
        ledger["static_score_evaluation_labels_accessed_classes"].append(class_name)
        _atomic_json(ledger_path, ledger)
        if len(labels) != info["stream_images"]:
            raise RunnerError(f"{class_name}: evaluator labels do not match frozen stream length")
        for seed in core.SUPPORT_SEEDS:
            entry = score_manifest["classes"][class_name]["score_tapes"][str(seed)]
            path = output / entry["file"]
            if _file_sha(path) != entry["sha256"]:
                raise RunnerError(f"Score tape checksum failed: {entry['file']}")
            with np.load(path, allow_pickle=False) as tape:
                scores = tape["test_scores"].astype(np.float64)
                c64 = tape["c64_scores"].astype(np.float64)
                expected_test_digest = core.opaque_digest(
                    "test-stream-identities", [row["relative_id"] for row in info["stream"]])
                if str(tape["test_identity_sha256"].item()) != expected_test_digest:
                    raise RunnerError(f"{class_name} seed {seed}: score stream identity mismatch")
                if str(tape["c64_identity_sha256"].item()) != info["c64"]["image_sha256"]:
                    raise RunnerError(f"{class_name} seed {seed}: C64 identity mismatch")
            metrics = core.static_score_metrics(labels, scores, c64,
                                                support_seed=seed,
                                                class_name=class_name)
            metrics["score_tape_sha256"] = entry["sha256"]
            metrics["support_identity_sha256"] = entry["support_group_sha256"]
            score_rows.append(metrics)
            for window in metrics["temporal_frozen_threshold"]["all_windows"]:
                window_rows.append({"class": class_name, "support_seed": seed, **window})
        print(f"evaluate: {class_name}; labels joined only after all scores froze")
    summary = core.summarize_support_seed_metrics(score_rows)
    _atomic_json(output / "static_score_metrics.json", {
        "schema": "stage05.static_score_metrics.v1", "per_class_seed": score_rows,
        "summary": summary,
        "oracle_label": "IN_SAMPLE_RETROSPECTIVE_STATIC_THRESHOLD_ORACLE",
        "interpretation": "Exploratory in-sample static-threshold diagnostic; not online, unbiased, or a guarantee.",
        "dynamic_memory_tested": False,
        "test_labels_exposed": True,
    })
    windows_path = output / "static_score_windows.csv.gz"
    import csv
    from io import StringIO
    stream = StringIO()
    columns = ["class", "support_seed", "start", "end_exclusive", "n_normal",
               "n_defect", "tp", "tn", "fp", "fn", "fpr", "fnr",
               "balanced_accuracy"]
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(window_rows)
    _atomic_bytes(windows_path, gzip.compress(stream.getvalue().encode(),
                                              compresslevel=9, mtime=0))
    ledger.update({
        "static_score_evaluation_labels_used_for": [
            "AUROC/AUPRC", "theta0 confusion metrics", "retrospective static threshold oracle",
            "temporal 256-image window metrics"],
        "score_tapes_frozen_before_this_label_join": True,
        "score_tape_manifest_sha256": _file_sha(output / "score_tape_manifest.json"),
        "test_label_source_sha256_by_class": {
            name: info["class_json_sha256"] for name, info in frozen["classes"].items()},
        "oracle_threshold_delivered_to_any_controller": False,
        "controller_code_or_parameters_modified": False,
    })
    _atomic_json(output / "data_exposure_ledger.json", ledger)
    _update_manifest(output, phase="evaluate", status="PASS",
                     extra={"static_score_metrics_sha256": _file_sha(
                         output / "static_score_metrics.json"),
                            "window_metrics_sha256": _file_sha(windows_path),
                            "test_labels_exposed": True})
    print(f"evaluate: PASS; {len(score_rows)} class/support-seed configurations")


def _write_audit_package(output: Path) -> Path:
    audit_dir = ROOT / "docs/calibration/audit/stage05_exploratory_v1"
    if audit_dir.exists():
        raise RunnerError(f"Refusing to overwrite immutable audit package: {audit_dir}")
    audit_dir.mkdir(parents=True)
    frozen = _read_json(output / "frozen_inputs.json")
    run_manifest = _read_json(output / "run_manifest.json")
    query_data_path = output / "query_schedule_aggregates.json.gz"
    with gzip.open(query_data_path, "rt", encoding="utf-8") as stream:
        query_data = json.load(stream)
    score_data = _read_json(output / "static_score_metrics.json")
    score_manifest = _read_json(output / "score_tape_manifest.json")
    # Compact aggregate tables contain no paths, raw image labels, or score tapes.
    import csv
    from io import StringIO
    query_csv = StringIO()
    qcols = ["class", "rate", "query_count", "queried_normals", "queried_defects",
             "queried_defect_fraction", "min_defects_across_seeds",
             "p05_defects", "median_defects", "p95_defects", "max_defects",
             "P_at_least_8", "P_at_least_14", "seed37_defects",
             "seed37_first_8_position", "seed37_first_14_position",
             "mean_longest_no_defect_span"]
    qwriter = csv.DictWriter(query_csv, fieldnames=qcols, lineterminator="\n")
    qwriter.writeheader()
    for class_name, rates in query_data["classes"].items():
        for rate in ("0.03", "0.05", "0.07"):
            item = rates[rate]
            seed37 = item["seed37_primary"]
            count = item["queried_defect_count"]
            qwriter.writerow({
                "class": class_name, "rate": rate,
                "query_count": seed37["query_count"],
                "queried_normals": seed37["queried_normals"],
                "queried_defects": seed37["queried_defects"],
                "queried_defect_fraction": seed37["queried_defect_fraction"],
                "min_defects_across_seeds": count["minimum"],
                "p05_defects": count["quantiles_linear"]["0.05"],
                "median_defects": count["quantiles_linear"]["0.5"],
                "p95_defects": count["quantiles_linear"]["0.95"],
                "max_defects": count["maximum"],
                "P_at_least_8": item["probability_finite_seed_schedules_at_least_8_defects"],
                "P_at_least_14": item["probability_finite_seed_schedules_at_least_14_defects"],
                "seed37_first_8_position": seed37["first_queried_defect_positions"]["8"],
                "seed37_first_14_position": seed37["first_queried_defect_positions"]["14"],
                "mean_longest_no_defect_span": item[
                    "longest_consecutive_image_span_without_queried_defect_distribution"]["mean"],
            })
    score_csv = StringIO()
    scols = ["class", "support_seed", "AUROC", "AUPRC", "prevalence", "theta0",
             "Frozen_BA", "Frozen_FPR", "Frozen_FNR", "TP", "TN", "FP", "FN",
             "Oracle_BA", "Oracle_FPR", "Oracle_FNR", "Oracle_threshold",
             "threshold_movement", "oracle_minus_frozen_BA", "near_ceiling",
             "worst_window_start", "worst_window_BA", "undefined_windows",
             "longest_false_negative_run", "cross_class_tie_fraction"]
    swriter = csv.DictWriter(score_csv, fieldnames=scols, lineterminator="\n")
    swriter.writeheader()
    for row in score_data["per_class_seed"]:
        frozen_metrics, oracle = row["frozen_threshold_metrics"], row["retrospective_oracle"]
        worst = row["temporal_frozen_threshold"]["worst_defined_window"] or {}
        swriter.writerow({
            "class": row["class"], "support_seed": row["support_seed"],
            "AUROC": row["auroc"], "AUPRC": row["auprc_average_precision"],
            "prevalence": row["defect_prevalence"], "theta0": row["theta0"]["value"],
            "Frozen_BA": frozen_metrics["balanced_accuracy"],
            "Frozen_FPR": frozen_metrics["fpr"], "Frozen_FNR": frozen_metrics["fnr"],
            "TP": frozen_metrics["tp"], "TN": frozen_metrics["tn"],
            "FP": frozen_metrics["fp"], "FN": frozen_metrics["fn"],
            "Oracle_BA": oracle["balanced_accuracy"], "Oracle_FPR": oracle["fpr"],
            "Oracle_FNR": oracle["fnr"], "Oracle_threshold": oracle["threshold"],
            "threshold_movement": row["signed_threshold_movement_from_theta0"],
            "oracle_minus_frozen_BA": row["oracle_minus_frozen_ba"],
            "near_ceiling": row["FROZEN_STATIC_NEAR_CEILING"],
            "worst_window_start": worst.get("start"),
            "worst_window_BA": worst.get("balanced_accuracy"),
            "undefined_windows": row["temporal_frozen_threshold"]["undefined_single_class_windows"],
            "longest_false_negative_run": row["temporal_frozen_threshold"][
                "longest_consecutive_false_negative_run"],
            "cross_class_tie_fraction": row["score_overlap"]["cross_class_tie_fraction"],
        })
    (audit_dir / "query_seed_aggregate_summary.csv").write_text(
        query_csv.getvalue(), encoding="utf-8", newline="")
    compact_query = {"schema": "stage05.query_seed_sensitivity.v1", "classes": {}}
    for class_name, rates in query_data["classes"].items():
        compact_query["classes"][class_name] = {}
        for rate in ("0.03", "0.05", "0.07"):
            item = rates[rate]
            schedule_rows = [row for row in query_data["per_seed_rows"]
                             if row["class"] == class_name and row["rate"] == float(rate)]
            compact_query["classes"][class_name][rate] = {
                "seed37_primary": item["seed37_primary"],
                "across_200_fixed_seeds": {
                    key: item[key] for key in (
                        "queried_defect_count", "queried_normals_distribution",
                        "queried_defects_distribution",
                        "queried_defect_fraction_distribution",
                        "probability_finite_seed_schedules_at_least_8_defects",
                        "probability_finite_seed_schedules_at_least_14_defects",
                        "first_activation_position",
                        "longest_consecutive_image_span_without_queried_defect_distribution",
                        "camera_count_mean_by_id", "temporal_bin_count_mean",
                        "selected_groups_by_available_camera_views_mean",
                        "sliding_256_window_minimum_defects_distribution",
                        "sliding_256_window_maximum_defects_distribution",
                    )},
                "first_400_counts_across_200_fixed_seeds": {
                    "queried_normals": _percentile_fields([
                        float(row["first_400_queries"]["normals"])
                        for row in schedule_rows]),
                    "queried_defects": item["first_400_queried_defects_distribution"],
                },
                "full_stream_composition": item["full_stream_composition"],
                "seed_summaries_are_finite_schedule_sensitivity": True,
            }
    _atomic_json(audit_dir / "query_seed_sensitivity.json", compact_query)
    (audit_dir / "static_score_seed_summary.csv").write_text(
        score_csv.getvalue(), encoding="utf-8", newline="")
    distribution_csv = StringIO()
    dcols = ["class", "support_seed", "normal_count", "normal_min", "normal_q05",
             "normal_q25", "normal_median", "normal_q75", "normal_q95", "normal_max",
             "defect_count", "defect_min", "defect_q05", "defect_q25",
             "defect_median", "defect_q75", "defect_q95", "defect_max",
             "overlap_ranges_intersect", "overlap_interval_low", "overlap_interval_high",
             "cross_class_equal_score_pairs", "cross_class_tie_fraction",
             "perfect_rank_separation", "AUROC", "AUPRC", "prevalence"]
    dwriter = csv.DictWriter(distribution_csv, fieldnames=dcols, lineterminator="\n")
    dwriter.writeheader()
    for row in score_data["per_class_seed"]:
        nscore, dscore = row["normal_scores"], row["defect_scores"]
        overlap = row["score_overlap"]
        nq, dq = nscore["quantiles_linear"], dscore["quantiles_linear"]
        interval = overlap["overlap_interval"] or [None, None]
        dwriter.writerow({
            "class": row["class"], "support_seed": row["support_seed"],
            "normal_count": nscore["count"], "normal_min": nq["0"],
            "normal_q05": nq["0.05"], "normal_q25": nq["0.25"],
            "normal_median": nq["0.5"], "normal_q75": nq["0.75"],
            "normal_q95": nq["0.95"], "normal_max": nq["1"],
            "defect_count": dscore["count"], "defect_min": dq["0"],
            "defect_q05": dq["0.05"], "defect_q25": dq["0.25"],
            "defect_median": dq["0.5"], "defect_q75": dq["0.75"],
            "defect_q95": dq["0.95"], "defect_max": dq["1"],
            "overlap_ranges_intersect": overlap["ranges_intersect"],
            "overlap_interval_low": interval[0], "overlap_interval_high": interval[1],
            "cross_class_equal_score_pairs": overlap["cross_class_equal_score_pairs"],
            "cross_class_tie_fraction": overlap["cross_class_tie_fraction"],
            "perfect_rank_separation": overlap[
                "perfect_rank_separation_normal_strictly_below_defect"],
            "AUROC": row["auroc"], "AUPRC": row["auprc_average_precision"],
            "prevalence": row["defect_prevalence"],
        })
    (audit_dir / "static_score_distribution_seed_summary.csv").write_text(
        distribution_csv.getvalue(), encoding="utf-8", newline="")
    # A compressed window table reports every registered 256-image window while
    # keeping the public package aggregate-only (no image paths or per-image labels).
    shutil.copyfile(output / "static_score_windows.csv.gz",
                    audit_dir / "static_score_windows.csv.gz")
    shutil.copyfile(output / "query_window_sensitivity.csv.gz",
                    audit_dir / "query_window_sensitivity.csv.gz")
    provenance = {
        "schema": "stage05.audit_provenance.v1",
        "branch": EXPECTED_BRANCH, "source_commit": EXPECTED_HEAD,
        "class_json_sha256": {name: info["class_json_sha256"]
                               for name, info in frozen["classes"].items()},
        "query_tape_archive_sha256": frozen["query_tapes_file_sha256"],
        "query_tape_digest_seed37": frozen["tape_digests_by_seed37"],
        "support_and_c64_identity_digests": {
            name: {"c64_group_sha256": info["c64"]["group_sha256"],
                  "c64_image_sha256": info["c64"]["image_sha256"],
                  "support_group_sha256_by_seed": {
                      seed: row["group_sha256"]
                      for seed, row in info["supports"].items()},
                  "support_image_sha256_by_seed": {
                      seed: row["image_sha256"]
                      for seed, row in info["supports"].items()}}
            for name, info in frozen["classes"].items()},
        "scorer_config": score_manifest["scorer_config"],
        "scorer_config_sha256": score_manifest["scorer_config_sha256"],
        "score_tape_hashes": {name: {seed: row["sha256"] for seed, row in
                                     info["score_tapes"].items()}
                              for name, info in score_manifest["classes"].items()},
        "scoring_execution_by_class": {
            name: {key: info.get(key) for key in (
                "test_images_scored", "c64_images_scored",
                "shared_feature_extraction_count", "support_features_encoded",
                "elapsed_seconds", "peak_cuda_allocated_bytes")}
            for name, info in score_manifest["classes"].items()},
        "source_hashes_by_phase": {
            "query_tape_and_support_freeze": frozen["query_builder_source_hashes"],
            "query_composition_evaluator": run_manifest.get("phase_source_hashes", {}).get("query_stats"),
            "native_score_tape_generation": score_manifest["scoring_source_hashes"],
            "query_window_aggregates": run_manifest.get("source_hashes"),
            "static_score_evaluator": run_manifest.get("phase_source_hashes", {}).get("evaluate"),
            "audit_publisher": _source_hashes(),
        },
        "model_provenance": score_manifest.get("model_provenance"),
        "scorer_source_config_sha256": score_manifest["scorer_config_sha256"],
        "query_schedule_aggregates_sha256": _file_sha(output / "query_schedule_aggregates.json.gz"),
        "query_window_sensitivity_sha256": _file_sha(output / "query_window_sensitivity.csv.gz"),
        "test_labels_exposed_for_exploratory_statistics": True,
        "masks_opened": False,
        "Mint_PCB_or_other_confirmation_outcomes_accessed": False,
        "Stage1_or_Stage2_executed": False,
        "controllers_modified": False,
        "raw_labels_scores_images_masks_weights_committed": False,
    }
    _atomic_json(audit_dir / "source_and_protocol_provenance.json", provenance)
    exposure = _read_json(output / "data_exposure_ledger.json")
    # Keep the published ledger compact and omit local paths/individual labels.
    _atomic_json(audit_dir / "data_exposure_ledger.json", exposure)
    gates = {
        "overall": "EXPLORATORY_AUDIT_COMPLETE_NOT_STAGE1_VALIDATION",
        "query_tape_freeze_before_test_labels": "PASS",
        "query_budget_nestedness_and_seed37_parity": "PASS",
        "five_seed_support_C64_integrity": "PASS",
        "native_score_tapes_complete_and_checksummed": score_manifest["status"],
        "development_test_labels_exposed": "RECORDED_EXPLORATORY_INFORMATION",
        "native_v22_score_parity": "PASS" if all(
            item["native_parity"]["passed"]
            for cls in score_manifest["classes"].values()
            for item in cls["score_tapes"].values()) else "FAIL",
        "stage1_authorization": "HOLD",
        "stage2_authorization": "HOLD",
        "mint_pcb_access": "NOT_ACCESSED",
        "stage1_or_stage2_executed": False,
    }
    _atomic_json(audit_dir / "gate_matrix.json", gates)
    summary = _summary_markdown(query_data, score_data, gates)
    (audit_dir / "README.md").write_text(summary, encoding="utf-8")
    _write_checksums(audit_dir)
    print(f"publish: compact audit package written to {audit_dir.relative_to(ROOT)}")
    return audit_dir


def _summary_markdown(query_data: dict[str, Any], score_data: dict[str, Any],
                      gates: dict[str, Any]) -> str:
    lines = [
        "# Stage 0.5 exploratory audit package",
        "",
        "This package contains aggregate development-set statistics from the four authorized classes. It is exploratory reconnaissance, not Stage 1 validation. Development test outcomes are exposed and must not be described as unseen in future method development. Mint/PCB were untouched; no controller was changed; Stage 1/2 remain on HOLD.",
        "",
        "## Query feedback composition",
        "",
                  "See `query_seed_aggregate_summary.csv` and `query_seed_sensitivity.json` for seed-37 results, camera/bin/group-view summaries, first-400 counts, 256-image query-feedback windows, and finite-seed ranges/activation fractions. The 200 seeds are a registered finite schedule sensitivity set, not a probability distribution over all possible tapes.",
        "",
        "| Class | Rate | seed-37 queried normals / defects | seed-37 queried defect fraction | P(>=8) | P(>=14) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for class_name, rates in query_data["classes"].items():
        for rate in ("0.03", "0.05", "0.07"):
            row = rates[rate]
            primary = row["seed37_primary"]
            defect_fraction = primary["queried_defect_fraction"]
            defect_fraction_text = (f"{defect_fraction:.4f}"
                                    if defect_fraction is not None else "undefined")
            lines.append(
                f"| {class_name} | {float(rate):.0%} | "
                f"{primary['queried_normals']} / {primary['queried_defects']} | "
                f"{defect_fraction_text} | "
                f"{row['probability_finite_seed_schedules_at_least_8_defects']:.3f} | "
                f"{row['probability_finite_seed_schedules_at_least_14_defects']:.3f} |")
    lines.extend(["", "## Static native scorer", "",
                  "`query_window_sensitivity.csv.gz` reports every stride-one 256-image position, with seed-37 query-defect counts and the min/5th/25th/median/75th/95th/max/mean/SD across 200 fixed schedules. It contains no per-image labels.",
                  "",
                  "See `static_score_seed_summary.csv` for class × support-seed AUROC, AUPRC, theta0 metrics, exact in-sample retrospective static-threshold oracle, headroom, and worst-window summaries. `static_score_distribution_seed_summary.csv` reports normal/defect score quantiles, overlap, cross-class ties, and separation. The oracle is an optimistic in-sample diagnostic; it was unavailable to every controller.",
                  "",
                  "The `FROZEN_STATIC_NEAR_CEILING` flag uses only the preregistered exploratory screen (Frozen BA ≥ 0.95 and oracle headroom ≤ 0.01). It is not evidence of industrial risk control or of no need for dynamic-memory calibration.",
                  "",
                  "## Gates and interpretation", "",
                  "```json", json.dumps(gates, indent=2, sort_keys=True), "```", "",
                  "The 64 C64 images are additional protected train-normal anchors alongside four deployed supports; this is not a four-shot-total setup. The static scorer uses M0 only, with dynamic Greedy memory empty and disabled. This audit does not test adaptive memory, online threshold calibration, or controller value under changing memory.",
                  ""])
    return "\n".join(lines)


def _write_checksums(folder: Path) -> None:
    entries = {path.relative_to(folder).as_posix(): _file_sha(path)
               for path in sorted(folder.rglob("*")) if path.is_file()
               and path.name != "checksums.sha256"}
    (folder / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in entries.items()),
        encoding="utf-8")


def _phase_publish(output: Path, dataset_root: Path, json_root: Path,
                    resume: bool) -> None:
    _require_output(output, create=False, resume=resume)
    _check_resume_identity(output, dataset_root, json_root)
    prior_manifest = _read_json(output / "run_manifest.json")
    audit_dir = ROOT / "docs/calibration/audit/stage05_exploratory_v1"
    if (prior_manifest.get("phase_status", {}).get("publish") == "PASS" and
            audit_dir.is_dir()):
        if _file_sha(audit_dir / "checksums.sha256") != prior_manifest.get(
                "audit_checksums_sha256"):
            raise RunnerError("Published immutable audit checksum file changed")
        print("publish: verified existing immutable audit package")
        return
    if not (output / "query_schedule_aggregates.json.gz").is_file():
        raise RunnerError("Cannot publish before query statistics are complete")
    if not (output / "static_score_metrics.json").is_file():
        raise RunnerError("Cannot publish before static score metrics are complete")
    if not (output / "query_window_sensitivity.csv.gz").is_file():
        raise RunnerError("Cannot publish before query 256-image window aggregates are complete")
    score_manifest = _read_json(output / "score_tape_manifest.json")
    if score_manifest.get("status") != "PASS":
        raise RunnerError("Cannot publish with incomplete native score tapes")
    audit = _write_audit_package(output)
    _update_manifest(output, phase="publish", status="PASS",
                     extra={"audit_package": audit.relative_to(ROOT).as_posix(),
                            "audit_checksums_sha256": _file_sha(audit / "checksums.sha256"),
                            "overall_status": "EXPLORATORY_AUDIT_COMPLETE_NOT_STAGE1_VALIDATION"})
    marker = output / "INCOMPLETE"
    if marker.exists():
        marker.unlink()
    _refresh_checksums(output)
    print("publish: PASS; no Stage 1/2 execution or confirmation data access")


def _phase_all(output: Path, dataset_root: Path, json_root: Path,
               resume: bool) -> None:
    # Isolated child processes prevent evaluator labels from remaining in the
    # score-generation process's object graph.
    for phase in ("freeze", "query-stats", "query-windows", "score", "evaluate", "publish"):
        command = [sys.executable, "-m", "experiments.calibration.stage05_runner",
                   "--phase", phase, "--dataset-root", str(dataset_root),
                   "--json-root", str(json_root), "--output-dir", str(output), "--resume"]
        if phase == "freeze" and not output.exists():
            command.remove("--resume")
        subprocess.run(command, cwd=ROOT, check=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--phase", required=True,
                        choices=("freeze", "query-stats", "query-windows", "score",
                                 "evaluate", "publish", "all"))
    result.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    result.add_argument("--json-root", type=Path, default=DEFAULT_JSON_ROOT)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--resume", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    dataset_root = args.dataset_root
    json_root = args.json_root
    if not dataset_root.is_dir() or not json_root.is_dir():
        raise SystemExit("Real-IAD dataset/JSON root is absent; Stage 0.5 cannot run")
    try:
        _check_environment()
        if args.phase == "all":
            _phase_all(output, dataset_root, json_root, args.resume)
        elif args.phase == "freeze":
            _phase_freeze(output, dataset_root, json_root, args.resume)
        elif args.phase == "query-stats":
            _phase_query_stats(output, dataset_root, json_root, args.resume)
        elif args.phase == "score":
            _phase_score(output, dataset_root, json_root, args.resume)
        elif args.phase == "query-windows":
            _phase_query_windows(output, dataset_root, json_root, args.resume)
        elif args.phase == "evaluate":
            _phase_evaluate(output, dataset_root, json_root, args.resume)
        else:
            _phase_publish(output, dataset_root, json_root, args.resume)
        return 0
    except Exception as exc:
        if output.exists():
            _update_manifest(output, phase=args.phase, status="BLOCKED",
                             extra={"last_error": f"{type(exc).__name__}: {exc}"})
            (output / "INCOMPLETE").write_text(
                f"Stage 0.5 incomplete: {type(exc).__name__}: {exc}\n",
                encoding="utf-8")
            _refresh_checksums(output)
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
