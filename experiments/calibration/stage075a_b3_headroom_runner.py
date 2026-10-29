"""One-command offline Actual-B3 and local-headroom diagnostic runner."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import traceback
from typing import Any

from .stage075a_b3_headroom import (
    DEFAULT_JSON_ROOT, DEFAULT_SOURCE, DEFAULT_STAGE05, DEVELOPMENT_CLASSES,
    LOCAL_OFFSETS, SUPPORT_SEEDS, HeadroomError, evaluate_all,
    load_verified_inputs, load_verified_labels, replay_all, sha256_file,
    sha256_json, _read_json,
)
from .profiles import canonical_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/calibration/stage075a_b3_headroom_v1_20261030"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, canonical_json(value) + "\n")


def _checksums(folder: Path) -> dict[str, str]:
    return {path.relative_to(folder).as_posix(): sha256_file(path)
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path.name not in {"checksums.json", "complete.json"}}


def run(source_dir: Path = DEFAULT_SOURCE, stage05_dir: Path = DEFAULT_STAGE05,
        json_root: Path = DEFAULT_JSON_ROOT, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    source_dir, stage05_dir, json_root, output = map(Path,
        (source_dir, stage05_dir, json_root, output))
    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / "INCOMPLETE"
    _atomic_text(incomplete, "Actual-B3 replay/headroom diagnostic is incomplete.\n")
    started = datetime.now(timezone.utc).isoformat()
    run_manifest: dict[str, Any] = {
        "schema": "stage075a.actual_b3_headroom.run.v1",
        "status": "INCOMPLETE", "started_utc": started,
        "branch": None, "head": None,
        "scope": {"new_neural_inference": False, "new_native_scoring": False,
                  "new_greedy_updates": False, "b8_implemented": False,
                  "stage075c_stage1_stage2": False, "confirmation_access": False,
                  "defect_masks_accessed": False},
    }
    try:
        import subprocess
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        if branch != "calibration-stage0-contract-20261029" or head != "d699ad63a19e207e8b58e7cc42aee8c58abc48cc":
            raise HeadroomError(f"Expected authorized worktree/HEAD, found {branch} {head}")
        run_manifest.update({"branch": branch, "head_before_diagnostic": head,
                             "python_executable": sys.executable,
                             "python_version": platform.python_version(),
                             "numpy_version": __import__("numpy").__version__,
                             "source_directory": source_dir.as_posix(),
                             "stage05_directory": stage05_dir.as_posix(),
                             "evaluator_metadata_root": json_root.as_posix(),
                             "local_offset_grid": list(LOCAL_OFFSETS),
                             "support_seeds": list(SUPPORT_SEEDS),
                             "categories": list(DEVELOPMENT_CLASSES)})
        provenance, categories = load_verified_inputs(source_dir, stage05_dir)
        # This first replay phase is label-free. The controller receives no
        # label object and no feedback callback is invoked.
        replay = replay_all(categories)
        if replay["labels_supplied_to_replay"] is not False:
            raise HeadroomError("Replay unexpectedly reports labels as controller input")
        # Only after every trajectory replay finishes do we open the four
        # previously exposed image-level evaluator labels.
        protocol = _read_json(source_dir / "protocol_manifest.json")
        labels, label_hashes = load_verified_labels(json_root, source_dir, protocol)
        original_evaluation = _read_json(source_dir / "evaluation_metrics.json")
        analysis = evaluate_all(categories, replay, labels, original_evaluation)
        # Deterministic independent in-process rerun over the same immutable
        # artifacts. Labels remain outside both replay calls.
        replay_repeat = replay_all(categories)
        analysis_repeat = evaluate_all(categories, replay_repeat, labels, original_evaluation)
        replay_digest = sha256_json(replay)
        replay_repeat_digest = sha256_json(replay_repeat)
        analysis_digest = sha256_json(analysis)
        analysis_repeat_digest = sha256_json(analysis_repeat)
        if replay_digest != replay_repeat_digest or analysis_digest != analysis_repeat_digest:
            raise HeadroomError("Deterministic repeated diagnostic produced different canonical digests")
        # The detailed local output contains scores and aggregate metrics, but
        # no labels, images, masks, feature tensors, or private checkpoints.
        _atomic_json(output / "source_provenance.json", provenance)
        _atomic_json(output / "b3_replay.json", replay)
        _atomic_json(output / "analysis.json", analysis)
        _atomic_json(output / "run_manifest.json", {
            **run_manifest, "status": "PASS", "finished_utc": datetime.now(timezone.utc).isoformat(),
            "source_provenance_sha256": sha256_file(output / "source_provenance.json"),
            "replay_sha256": sha256_file(output / "b3_replay.json"),
            "analysis_sha256": sha256_file(output / "analysis.json"),
            "label_metadata_sha256_by_class": label_hashes,
            "canonical_replay_digest": replay_digest,
            "canonical_repeat_replay_digest": replay_repeat_digest,
            "canonical_analysis_digest": analysis_digest,
            "canonical_repeat_analysis_digest": analysis_repeat_digest,
            "deterministic_repeat_status": "PASS",
            "trajectory_count": provenance["trajectory_count"],
            "b3_replay_status": replay["status"],
            "historical_metric_reproduction_status": analysis["historical_metric_reproduction"]["status"],
            "online_local_gain_identifiability": analysis["online_identifiability_status"],
            "stale_original_incomplete_marker": provenance["stale_incomplete_marker"],
            "original_stage075a_marker_note": provenance["stale_marker_interpretation"],
            "runner_scope": analysis["no_new_scoring_performed"] and analysis["no_defect_masks_accessed"],
        })
        checksums = _checksums(output)
        _atomic_json(output / "checksums.json", {"schema": "stage075a.b3_headroom.checksums.v1",
                                                    "files": checksums})
        # Verify every file and each manifest linkage before removing only our
        # own newly-created incomplete marker.
        for relative, digest in checksums.items():
            if sha256_file(output / relative) != digest:
                raise HeadroomError(f"Output checksum mismatch: {relative}")
        incomplete.unlink()
        complete = {"status": "PASS", "output": output.as_posix(),
                    "run_manifest_sha256": sha256_file(output / "run_manifest.json"),
                    "checksums_manifest_sha256": sha256_file(output / "checksums.json"),
                    "files_checked": len(checksums)}
        _atomic_json(output / "complete.json", complete)
        # complete.json is deliberately the final committed-by-runner marker;
        # refresh checksums so the complete marker is also covered.
        checksums = _checksums(output)
        _atomic_json(output / "checksums.json", {"schema": "stage075a.b3_headroom.checksums.v1",
                                                    "files": checksums})
        if any(sha256_file(output / name) != digest for name, digest in checksums.items()):
            raise HeadroomError("Final output checksum verification failed")
        return {"status": "PASS", "output": output.as_posix(),
                "trajectory_count": provenance["trajectory_count"],
                "canonical_replay_digest": replay_digest,
                "canonical_analysis_digest": analysis_digest,
                "checksums": checksums}
    except Exception as exc:
        run_manifest.update({"status": "BLOCKED", "finished_utc": datetime.now(timezone.utc).isoformat(),
                             "error_type": type(exc).__name__, "error": str(exc),
                             "traceback": traceback.format_exc(limit=12)})
        _atomic_json(output / "run_manifest.json", run_manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--stage05", type=Path, default=DEFAULT_STAGE05)
    parser.add_argument("--json-root", type=Path, default=DEFAULT_JSON_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = run(args.source, args.stage05, args.json_root, args.output)
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
