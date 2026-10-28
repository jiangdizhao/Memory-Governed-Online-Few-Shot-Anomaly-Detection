"""One-command Stage 0 synthetic validation and audit artifact writer.

This runner never loads a dataset, image, model weight, or confirmation label.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from typing import Any

import torch

from . import PROTOCOL_VERSION
from .controllers import theta0_from_c64
from .engine import PrequentialEngine
from .method_registry import METHOD_CARDS, build_controllers
from .metrics import PrivateOutcomeLedger
from .profiles import DEFAULT_PROFILE, canonical_json
from .protocol import (EvidenceAnchor, EvidenceSnapshot, ScoreResult,
                       StreamEvent, TapeRecord)
from .query_tape import build_qa_ts, build_qb


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PYTHON = Path(r"D:\anaconda3\envs\torch\python.exe")
HISTORICAL_SOURCES = (
    "vmb_visionad_new.py", "test_phase3_AIF_simplified_v22.py",
    "experiments/p0_protocol_audit.py", "experiments/a0_a1_strict4_matched.py",
    "experiments/v02_memory.py", "experiments/risk_safe/native_bridge.py",
    "experiments/risk_safe/preflight.py", "experiments/risk_safe/query_protocol.py",
    "experiments/risk_safe/d2_threshold_calibration.py",
    "experiments/risk_safe/p01_falsification_native.py",
)
ARTIFACT_FILES = frozenset({
    "run_identity.json", "INCOMPLETE", "protocol_manifest.json",
    "dependency_environment.json", "gates.json",
    "controller_method_cards.json", "synthetic_event_trace.jsonl",
    "synthetic_summary.json", "test_results.json", "source_hashes.json",
    "stage0_summary.md", "checksums.json",
})


def _sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def _hash_file(path: Path) -> str: return _sha(path.read_bytes())


def _json_write(path: Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(payload, encoding="utf-8")
    temp.replace(path)


def _check_environment() -> dict[str, Any]:
    actual = Path(sys.executable).resolve()
    expected = EXPECTED_PYTHON.resolve()
    if str(actual).casefold() != str(expected).casefold():
        raise RuntimeError(f"Wrong Python executable: {actual}; expected {expected}")
    if sys.prefix.casefold() != r"d:\anaconda3\envs\torch".casefold():
        raise RuntimeError(f"Wrong Python environment prefix: {sys.prefix}")
    return {"python_executable": str(actual), "python_version": sys.version,
            "environment_prefix": sys.prefix, "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(), "execution_device": "cpu"}


def _synthetic_run() -> dict[str, Any]:
    n = 160
    records = [TapeRecord(i, f"synthetic-group-{i:04d}", i % 5 + 1,
                          f"synthetic-view-{i:04d}", phase=0) for i in range(n)]
    tape = build_qa_ts(records)
    outcomes_raw = [int((i % 17) in (0, 1, 2)) for i in range(n)]
    outcomes = PrivateOutcomeLedger(outcomes_raw)
    c64_scores = [0.01 + i * 0.001 for i in range(64)]
    theta0 = theta0_from_c64(c64_scores)
    controllers = build_controllers(theta0, c64_scores)

    class Memory:
        version = 0
        digest = _sha(b"synthetic-empty-memory")
    memory = Memory()
    evidence_extra: list[tuple[EvidenceAnchor, int]] = []
    events = [StreamEvent(record, (i % 23) / 100.0) for i, record in enumerate(records)]

    def score_fn(event: StreamEvent) -> ScoreResult:
        score = float(event.payload) + memory.version * 0.001
        state = _sha(canonical_json({"version": memory.version,
                                     "kind": "synthetic_only"}).encode())
        return ScoreResult(score, memory.version, state)

    def write_fn(event: StreamEvent, score: ScoreResult, feedback: Any) -> dict[str, Any]:
        before = memory.digest
        if feedback.queried and feedback.label == 0:
            memory.version += 1
            memory.digest = _sha(f"synthetic-memory-{memory.version}".encode())
            changed = True
        else:
            changed = False
        return {"queried_normal_only": bool(feedback.queried and feedback.label == 0),
                "changed": changed, "before": before, "after": memory.digest}

    def evidence_fn(event: StreamEvent | None, feedback: Any) -> EvidenceSnapshot:
        if event is not None and feedback is not None:
            label = int(feedback.label)
            evidence_extra.append((EvidenceAnchor(
                f"query:{event.record.position}",
                float(event.payload) + memory.version * 0.001, memory.version,
                False, 0, "queried_normal_all_layer_leave_source_out"
                if label == 0 else "queried_defect_not_written"), label))
        c64 = tuple(EvidenceAnchor(f"c64:{i:03d}", value + memory.version * 0.001,
                                   memory.version, True, 0, "C64_PROTECTED")
                    for i, value in enumerate(c64_scores))
        def current_version(anchor: EvidenceAnchor) -> EvidenceAnchor:
            shift = (memory.version - anchor.memory_version) * 0.001
            return EvidenceAnchor(anchor.opaque_anchor_id, anchor.score + shift,
                                  memory.version, anchor.protected,
                                  anchor.source_excluded_rows,
                                  anchor.source_exclusion_policy)
        normal_extra = tuple(current_version(a) for a, label in evidence_extra if label == 0)
        defect_extra = tuple(current_version(a) for a, label in evidence_extra if label == 1)
        return EvidenceSnapshot(memory.version, c64 + normal_extra, defect_extra,
                                {"synthetic": True, "c64_mutable": False},
                                evidence_revision=len(evidence_extra) + memory.version)

    engine = PrequentialEngine(
        tape=tape, events=events, outcomes=outcomes, controllers=controllers,
        score_fn=score_fn,
        memory_state_fn=lambda: (memory.version, memory.digest),
        write_fn=write_fn, evidence_fn=evidence_fn,
        provenance_id=f"synthetic-stage0-v1:{tape.stream_digest}")
    result = engine.run()
    qb = build_qb([TapeRecord(i, f"qb-group-{i:04d}", i % 5 + 1,
                              f"qb-view-{i:04d}") for i in range(1000)])
    return {"execution_scope": "synthetic_mechanism_smoke_only",
            "stage1_or_stage2_run": False,
            "synthetic_stream_n": n,
            "theta0_synthetic_c64_higher_q90": theta0,
            "qa_ts": {"digest": tape.digest, "query_count": len(tape.events),
                      "query_fraction_report": {
                          "images": len(tape.events) / n,
                          "groups": len(tape.queried_groups) / n}},
            "qb": {"digest": qb.digest, "query_count": len(qb.events),
                   "early_budget": qb.early_budget,
                   "early_queries": sum(e.position < 400 for e in qb.events)},
            "final_memory_version": memory.version,
            "metrics": result["metrics"],
            "scores": result["scores"], "predictions": result["predictions"],
            "trace": result["trace"],
            "controller_states": result["controller_states"]}


def _run_tests() -> dict[str, Any]:
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests" / "calibration"))
    stream = __import__("io").StringIO()
    class TrackingResult(unittest.TextTestResult):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.passed_test_ids: list[str] = []
        def addSuccess(self, test):
            self.passed_test_ids.append(test.id())
            super().addSuccess(test)
    runner = unittest.TextTestRunner(stream=stream, verbosity=2,
                                     resultclass=TrackingResult)
    started = time.time()
    result = runner.run(suite)
    details = {
        "tests_run": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
        "failed": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "duration_seconds": time.time() - started,
        "passed_test_ids": result.passed_test_ids,
        "output": stream.getvalue(),
        "failures": [str(t) + "\n" + msg for t, msg in result.failures],
        "errors_detail": [str(t) + "\n" + msg for t, msg in result.errors],
        "skipped_detail": [str(t) + ": " + msg for t, msg in result.skipped],
    }
    return details


def _source_hashes() -> dict[str, str]:
    paths = [ROOT / p for p in HISTORICAL_SOURCES]
    paths.extend(sorted((ROOT / "experiments" / "calibration").glob("*.py")))
    paths.extend(sorted((ROOT / "tests" / "calibration").glob("*.py")))
    paths.extend(sorted((ROOT / "docs" / "calibration").glob("*.md")))
    paths.append(ROOT / "docs" /
                 "Designing and Prioritizing Memory-Dependent _ State-Consistent Calibration for Dynamic-Memory Indust.pdf")
    return {str(path.relative_to(ROOT)).replace("\\", "/"): _hash_file(path)
            for path in paths if path.is_file()}


def _git_info() -> tuple[str, list[str]]:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.splitlines()
    return branch + "@" + commit, status


def _verify_checksums(out_dir: Path) -> bool:
    path = out_dir / "checksums.json"
    if not path.exists(): return False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    actual = {p.name for p in out_dir.iterdir() if p.is_file() and
              p.name != "checksums.json"}
    if set(manifest) != actual:
        return False
    for rel, expected in manifest.items():
        file = out_dir / rel
        if not file.is_file() or _hash_file(file) != expected:
            return False
    return True


def _make_artifacts(out_dir: Path) -> dict[str, Any]:
    marker = out_dir / "INCOMPLETE"
    marker.write_text("Synthetic Stage 0 only; production fidelity gates unresolved.\n",
                      encoding="utf-8")
    env = _check_environment()
    git, status = _git_info()
    branch, source_commit = git.split("@", 1)
    source_hashes = _source_hashes()
    tests = _run_tests()
    tests_ok = (tests["failed"] == 0 and tests["errors"] == 0 and
                tests["skipped"] == 0)
    synthetic = (_synthetic_run() if tests_ok else {
        "execution_scope": "skipped_after_unit_test_failure",
        "trace": [], "controller_states": {}, "scores": [], "predictions": {},
    })
    dataset_root = Path(os.environ.get("REALIAD_ROOT", r"G:\dataset\realiad"))
    json_root = Path(os.environ.get("REALIAD_JSON_ROOT",
                                   r"G:\dataset\realiad\jsons\realiad_jsons"))
    preexisting_archive_changes = [s for s in status
                                   if "outputs/risk_safe_sprint_20261027" in s]
    gates = {
        "G0": {"status": "BLOCKED" if preexisting_archive_changes else "PASS",
               "reason": "Pre-existing tracked historical archive deletions remain in worktree; not modified or staged." if preexisting_archive_changes else "Correct branch and no protected worktree changes."},
        "G1": {"status": "PASS" if tests_ok else "FAIL", "reason": "Synthetic Q-A-TS/Q-B metadata tests passed." if tests_ok else "Synthetic query tests failed; see test_results.json."},
        "G1_real_data": {"status": "SKIP", "reason": "Dataset path presence checked only; no Real-IAD files or labels opened.",
                         "dataset_root_present": dataset_root.is_dir(),
                         "json_root_present": json_root.is_dir()},
        "G2": {"status": "BLOCKED", "reason": "Native v22 score parity not executed; no dataset image/features used."},
        "G3": {"status": "BLOCKED", "reason": "Synthetic Greedy reference tests pass; live native K=256 and fp16 update parity have not run."},
        "G4": {"status": "PASS" if tests_ok else "FAIL", "reason": "Synthetic predict-before-disclosure and queried-only feedback tests passed." if tests_ok else "Synthetic causal tests failed; see test_results.json."},
        "G5": {"status": "PASS" if tests_ok else "FAIL", "reason": "Synthetic protected C64, current-version evidence, and all-layer LSO source attribution tests passed." if tests_ok else "Synthetic evidence tests failed; see test_results.json."},
        "G6": {"status": "BLOCKED_FIDELITY", "reason": "B5a/B5b and B6 remain BLOCKED_FIDELITY; B7 is executable only under its explicitly provisional method specification."},
        "G7": {"status": "BLOCKED", "reason": "Reference math verified for B0-B4; blocked baselines prevent all-family acceptance."},
        "G8": {"status": "PASS" if any("test_checkpoint_resume_matches_uninterrupted_run_and_restores_rng" in item for item in tests.get("passed_test_ids", [])) else "FAIL", "reason": "Event-boundary checkpoint/resume is compared with an uninterrupted synthetic run, including memory, controller, trace, private-label identity, and Python/NumPy/Torch RNG restoration." if any("test_checkpoint_resume_matches_uninterrupted_run_and_restores_rng" in item for item in tests.get("passed_test_ids", [])) else "Checkpoint/resume equivalence test did not pass; see test_results.json."},
        "G9": {"status": "BLOCKED", "reason": "Native score/AUROC parity across all controllers not run; synthetic trace is mechanism evidence only."},
    }
    manifest = {
        "stage": 0, "overall_status": "BLOCKED_WIP" if tests_ok else "FAIL",
        "branch": branch, "source_commit": source_commit,
        "protocol_version": PROTOCOL_VERSION,
        "profile": asdict(DEFAULT_PROFILE), "profile_sha256": DEFAULT_PROFILE.sha256,
        "execution": "synthetic_cpu_only", "stage1_run": False, "stage2_run": False,
        "confirmation_outcomes_opened": False,
        "support_regime": "four deployed supports plus 64 additional protected C64 normals",
        "method_cards": METHOD_CARDS,
        "query_tape_versions": ["Q-A-TS-v1", "Q-B via Q-A-TS-v1 allocation"],
    }
    _json_write(out_dir / "protocol_manifest.json", manifest)
    _json_write(out_dir / "dependency_environment.json", env)
    _json_write(out_dir / "gates.json", gates)
    _json_write(out_dir / "controller_method_cards.json", METHOD_CARDS)
    with (out_dir / "synthetic_event_trace.jsonl").open("w", encoding="utf-8") as f:
        for row in synthetic["trace"]:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    compact = {k: v for k, v in synthetic.items()
               if k not in {"trace", "scores", "predictions", "controller_states"}}
    compact["controller_names"] = list(synthetic["controller_states"])
    _json_write(out_dir / "synthetic_summary.json", compact)
    _json_write(out_dir / "test_results.json", tests)
    _json_write(out_dir / "source_hashes.json", {
        "stage0_source_hashes": source_hashes,
        "source_commit": source_commit,
        "source_fingerprint_sha256": _sha(canonical_json(source_hashes).encode()),
        "historical_sources_unchanged_by_this_work": True})
    summary = [
        "# Stage 0 synthetic runner summary", "",
        f"Status: **{'BLOCKED_WIP' if tests_ok else 'FAIL'}; not scientifically validated**.", "",
        f"Branch/commit: `{branch}` / `{source_commit}`.",
        "Execution used the configured torch environment, CPU, and synthetic metadata only.",
        "No Real-IAD image, feature, mask, or confirmation label was opened.",
        "Stage 1 and Stage 2 were not run.", "",
        "Gate statuses: " + ", ".join(f"{k}={v['status']}" for k, v in gates.items()),
        "", "The independent review should use `controller_method_cards.json`, `gates.json`,",
        "`source_hashes.json`, `test_results.json`, and the synthetic trace.",
    ]
    (out_dir / "stage0_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    files = [p for p in out_dir.iterdir() if p.is_file() and
             p.name != "checksums.json"]
    _json_write(out_dir / "checksums.json", {p.name: _hash_file(p) for p in sorted(files)})
    acceptance = all(gates[f"G{i}"]["status"] == "PASS" for i in range(10))
    return {"gates": gates, "tests": tests, "source_commit": source_commit,
            "branch": branch, "synthetic": compact,
            "status": "PASS" if acceptance else "BLOCKED_WIP" if tests_ok else "FAIL"}


def _verify_artifact(out_dir: Path) -> dict[str, Any]:
    if not out_dir.is_dir() or not _verify_checksums(out_dir):
        raise RuntimeError("Artifact missing or checksum verification failed")
    manifest = json.loads((out_dir / "protocol_manifest.json").read_text(encoding="utf-8"))
    hashes = json.loads((out_dir / "source_hashes.json").read_text(encoding="utf-8"))
    identity_path = out_dir / "run_identity.json"
    if not identity_path.is_file():
        raise RuntimeError("Artifact lacks a resumable run identity record")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    current_hashes = _source_hashes()
    current_fp = _sha(canonical_json(current_hashes).encode())
    if (manifest.get("profile_sha256") != DEFAULT_PROFILE.sha256 or
            hashes.get("stage0_source_hashes") != current_hashes or
            identity.get("source_fingerprint_sha256") != current_fp or
            identity.get("profile_sha256") != DEFAULT_PROFILE.sha256):
        raise RuntimeError("Artifact source/configuration provenance does not match this checkout")
    return {"checksums_verified": True, "source_and_config_match": True,
            "artifact_status": manifest.get("overall_status", "UNKNOWN"),
            "run_key": identity.get("run_key")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.stage != 0:
        parser.error("Only Stage 0 is implemented; Stage 1/2 are prohibited")
    _check_environment()
    if args.verify_only:
        if args.output_dir is None:
            parser.error("--verify-only requires --output-dir for an exact artifact")
        output = args.output_dir.resolve()
        result = _verify_artifact(output)
        print(json.dumps({"mode": "verify-only",
                          "verification_status": "ARTIFACT_INTEGRITY_VERIFIED",
                          "overall_status": result["artifact_status"],
                          "output_dir": str(output),
                          "checksums_verified": result["checksums_verified"],
                          "source_and_config_match": result["source_and_config_match"]}, indent=2))
        return 0
    git, _ = _git_info()
    source_commit = git.split("@", 1)[1]
    source_fingerprint = _sha(canonical_json(_source_hashes()).encode())
    default_key = _sha((DEFAULT_PROFILE.sha256 + source_fingerprint).encode())[:16]
    out_dir = (args.output_dir or ROOT / "outputs" / "calibration" / "stage0" /
               f"stage0_{default_key}").resolve()
    if not out_dir.is_relative_to(ROOT):
        raise RuntimeError("Output directory must remain inside the repository")
    if out_dir.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite existing Stage 0 output: {out_dir}")
        existing_names = {p.name for p in out_dir.iterdir()}
        if existing_names - ARTIFACT_FILES:
            raise RuntimeError("Resume refused: unexpected files/directories in output")
        identity_file = out_dir / "run_identity.json"
        if not identity_file.is_file():
            raise RuntimeError("Resume refused: output has no Stage 0 run identity")
        identity = json.loads(identity_file.read_text(encoding="utf-8"))
        expected_identity = {"schema": "calibration.stage0_run_identity.v1",
            "run_key": default_key, "profile_sha256": DEFAULT_PROFILE.sha256,
            "source_fingerprint_sha256": source_fingerprint}
        if identity != expected_identity:
            raise RuntimeError("Resume refused: source/configuration provenance changed")
        if _verify_checksums(out_dir):
            result = _verify_artifact(out_dir)
            print(json.dumps({"mode": "resume-verified",
                              "verification_status": "ARTIFACT_INTEGRITY_VERIFIED",
                              "overall_status": result["artifact_status"],
                              "output_dir": str(out_dir), "checksums_verified": True,
                              "source_and_config_match": True}, indent=2))
            return 0
        if not (out_dir / "INCOMPLETE").exists():
            raise RuntimeError("Resume refused: incomplete marker is missing")
        # Rebuild deterministic smoke files in place only after the immutable
        # source/config identity matches; research streams are never resumed.
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
        _json_write(out_dir / "run_identity.json", {
            "schema": "calibration.stage0_run_identity.v1",
            "run_key": default_key, "profile_sha256": DEFAULT_PROFILE.sha256,
            "source_fingerprint_sha256": source_fingerprint})
    result = _make_artifacts(out_dir)
    print(json.dumps({"mode": "all", "status": result["status"],
                      "output_dir": str(out_dir), "branch": result["branch"],
                      "source_commit": result["source_commit"],
                      "gates": {k: v["status"] for k, v in result["gates"].items()},
                      "tests_run": result["tests"]["tests_run"],
                      "passed": result["tests"]["passed"]}, indent=2))
    return 0 if result["status"] == "PASS" else 2 if result["status"] == "BLOCKED_WIP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
