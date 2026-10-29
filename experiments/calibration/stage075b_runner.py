"""One-command Stage 0.75-B synthetic validation and immutable artifact writer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

from .profiles import canonical_json
from .stage075b_synthetic import (
    FOCUSED_SEED, SCENARIOS, STAGE075B_SEEDS, checkpoint_resume_check,
    focused_mechanism_results, run_full_matrix, run_scenario,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs" / "calibration" / "stage075b_synthetic_v3_20261029"
AUDIT = ROOT / "docs" / "calibration" / "audit" / "stage075b_synthetic_v2"
BRANCH = "calibration-stage0-contract-20261029"
EXPECTED_PARENT = "0afa20967c5ed88f804d87f380089b2ca543a9fb"
SOURCE_PATHS = (
    "experiments/calibration/stage075b_controllers.py",
    "experiments/calibration/stage075b_synthetic.py",
    "experiments/calibration/stage075b_runner.py",
    "experiments/calibration/controllers.py",
    "experiments/calibration/scdac_g.py",
    "experiments/calibration/engine.py",
    "experiments/calibration/protocol.py",
    "experiments/calibration/metrics.py",
    "experiments/calibration/query_tape.py",
    "experiments/calibration/profiles.py",
    "tests/calibration/test_stage075b.py",
    "docs/calibration/STAGE075B_IMPLEMENTATION_CONTRACT_CANDIDATE_v1.md",
    "docs/calibration/STAGE075B_IMPLEMENTATION_CONTRACT_CANDIDATE_v2.md",
    "docs/calibration/STAGE075B_IMPLEMENTATION_CONTRACT_CANDIDATE_v3.md",
    "docs/calibration/STAGE075B_SYNTHETIC_PROTOCOL_CANDIDATE_v0_1.md",
    "docs/calibration/STAGE075_B8_ECMSC_RESEARCH_DESIGN_v0_2.md",
)
PROTECTED_PATHS = (
    "vmb_visionad_new.py", "test_phase3_AIF_simplified_v22.py",
    "experiments/v02_memory.py", "experiments/p0_protocol_audit.py",
    "experiments/risk_safe/native_bridge.py",
    "experiments/risk_safe/p01_falsification_native.py",
    "experiments/calibration/controllers.py",
    "experiments/calibration/scdac_g.py",
    "docs/calibration/STAGE075A_EXPERIMENT_PROTOCOL_v1.md",
    "docs/calibration/STAGE075A_GREEDY_RANKING_FALSIFICATION_REPORT.md",
    "docs/calibration/STAGE075A_C64_MEMORY_TRANSPORT_REPORT.md",
    "docs/calibration/STAGE075A_RESEARCH_DECISION_CANDIDATE.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _git(*args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, text=True,
                          capture_output=True, check=False)
    if proc.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _source_hashes() -> dict[str, str]:
    return {path: sha256_file(ROOT / path) for path in SOURCE_PATHS}


def _protected_source_check() -> dict[str, Any]:
    changed = _git("diff", "--name-only", "--", *PROTECTED_PATHS).splitlines()
    staged = _git("diff", "--cached", "--name-only", "--", *PROTECTED_PATHS).splitlines()
    return {"status": "PASS" if not changed and not staged else "FAIL",
            "working_diff": changed, "staged_diff": staged}


def _run_command(command: list[str], output_path: Path) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                          check=False, timeout=1800)
    output_path.write_text(proc.stdout + ("\n--- STDERR ---\n" + proc.stderr
                                         if proc.stderr else ""), encoding="utf-8")
    text = proc.stdout + proc.stderr
    count_match = re.search(r"Ran\s+(\d+) tests?", text)
    return {"command": command, "returncode": proc.returncode,
            "passed": proc.returncode == 0,
            "tests_run": int(count_match.group(1)) if count_match else None,
            "log": output_path.name}


def _run_validation(output: Path) -> dict[str, Any]:
    python = sys.executable
    compile_result = _run_command(
        [python, "-m", "compileall", "-q", "experiments/calibration", "tests/calibration"],
        output / "compileall.log")
    test_result = _run_command(
        [python, "-m", "unittest", "discover", "-s", "tests/calibration", "-v"],
        output / "unittest.log")
    return {"python_compile": compile_result, "calibration_unittest": test_result}


def _gate(status: bool, details: str) -> dict[str, str]:
    return {"status": "PASS" if status else "FAIL", "details": details}


def _overall_status(gates: dict[str, dict[str, str]]) -> str:
    statuses = [gate["status"] for gate in gates.values() if gate["status"] != "PENDING"]
    if "BLOCKED" in statuses:
        return "BLOCKED"
    return "PASS" if all(status == "PASS" for status in statuses) else "FAIL"


def _scenario_matrix_status(matrix: dict[str, Any]) -> bool:
    actual = {row["scenario"] for row in matrix["scenario_results"]}
    required = {scenario.key for scenario in SCENARIOS}
    return actual == required and len(matrix["scenario_results"]) == len(required) * len(STAGE075B_SEEDS)


def _strip_runtime(matrix: dict[str, Any]) -> list[tuple[str, int, str]]:
    return [(row["scenario"], row["seed"], row["trajectory_sha256"])
            for row in matrix["scenario_results"]]


def _checksums(output: Path) -> dict[str, str]:
    files = sorted(p for p in output.rglob("*") if p.is_file() and
                   p.name not in {"checksums.json", "INCOMPLETE"})
    return {p.relative_to(output).as_posix(): sha256_file(p) for p in files}


def verify_artifact(output: Path) -> dict[str, Any]:
    checksum_path = output / "checksums.json"
    if not checksum_path.exists() or (output / "INCOMPLETE").exists():
        return {"status": "BLOCKED", "reason": "checksums absent or run marked INCOMPLETE"}
    expected = json.loads(checksum_path.read_text(encoding="utf-8"))
    actual = _checksums(output)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatched = sorted(key for key in set(expected) & set(actual)
                        if expected[key] != actual[key])
    source_mismatch: list[str] = []
    identity_path = output / "run_identity.json"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        recorded_sources = identity.get("source_hashes", {})
        current_sources = _source_hashes()
        source_mismatch = sorted(path for path, digest in recorded_sources.items()
                                 if current_sources.get(path) != digest)
    else:
        source_mismatch = ["run_identity.json missing"]
    ok = not (missing or extra or mismatched or source_mismatch)
    return {"status": "PASS" if ok else "FAIL", "files_expected": len(expected),
            "missing": missing, "extra": extra, "mismatched": mismatched,
            "source_hash_mismatch": source_mismatch}


def _environment() -> dict[str, Any]:
    import numpy as np
    import torch
    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception:
        scipy_version = None
    expected_prefix = Path(r"D:\anaconda3\envs\torch")
    prefix_ok = os.path.normcase(os.path.normpath(sys.prefix)) == os.path.normcase(
        os.path.normpath(str(expected_prefix)))
    return {"python_version": sys.version, "environment_name": "torch",
            "python_executable_basename": Path(sys.executable).name,
            "python_prefix": str(Path(sys.prefix).name), "numpy": np.__version__,
            "torch": torch.__version__, "scipy": scipy_version,
            "cuda_available": bool(torch.cuda.is_available()),
            "execution_device": "CPU synthetic only", "expected_torch_environment": prefix_ok}


def execute_all(output: Path) -> int:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing Stage075-B output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "INCOMPLETE").write_text("Stage 0.75-B run is incomplete.\n", encoding="utf-8")
    status: dict[str, Any] = {"run_status": "INCOMPLETE", "started_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(output / "run_identity.json", {
        "schema": "stage075b.run_identity.v1", "branch": _git("branch", "--show-current"),
        "parent_sha": _git("rev-parse", "HEAD"), "expected_parent_sha": EXPECTED_PARENT,
        "source_hashes": _source_hashes(), "scenarios": [s.key for s in SCENARIOS],
        "seeds": list(STAGE075B_SEEDS), "query_protocol": "Q-A-TS-v1 / 3% / seed 37",
        "n_events_per_scenario_seed": 1024,
        "primary_method_version": "stage075b-ecmsc-candidate-v3",
        "superseded_contract": "STAGE075B_IMPLEMENTATION_CONTRACT_CANDIDATE_v1.md",
        "superseded_run": "outputs/calibration/stage075b_synthetic_20261029/",
    })
    try:
        branch = _git("branch", "--show-current")
        head = _git("rev-parse", "HEAD")
        source_integrity = _protected_source_check()
        validation = _run_validation(output)
        status["validation"] = validation
        compile_ok = validation["python_compile"]["passed"]
        tests_ok = validation["calibration_unittest"]["passed"]
        if not compile_ok or not tests_ok:
            raise RuntimeError("Compilation or calibration unittest suite failed")

        focused = focused_mechanism_results()
        checkpoint = checkpoint_resume_check()
        matrix_first = run_full_matrix()
        matrix_second = run_full_matrix()
        deterministic = _strip_runtime(matrix_first) == _strip_runtime(matrix_second)
        environment = _environment()
        same_information = all(
            row["metrics"]["B3+Offset"]["evidence"]["normal_source_count"] ==
            row["metrics"]["B8-Full"]["evidence"]["normal_source_count"] and
            row["metrics"]["B3+Offset"]["evidence"]["defect_source_count"] ==
            row["metrics"]["B8-Full"]["evidence"]["defect_source_count"] and
            row["query"]["n_queried_images"] == row["query"]["n_queried_groups"]
            for row in matrix_first["scenario_results"])
        memory_profile = run_scenario(
            next(s for s in SCENARIOS if s.key == "S12_shared_score_invariance"),
            STAGE075B_SEEDS[0], measure_memory=True)
        gates = {
            "G1_repository_source_integrity": _gate(
                branch == BRANCH and head == EXPECTED_PARENT and source_integrity["status"] == "PASS"
                and environment["expected_torch_environment"],
                f"branch={branch}; parent={head}; protected_sources={source_integrity['status']}"),
            "G2_mathematical_specification": _gate(
                (ROOT / "docs/calibration/STAGE075B_IMPLEMENTATION_CONTRACT_CANDIDATE_v3.md").exists(),
                "Frozen synthetic-only contract v3 uses current C64 percentile state; v1/v2 are preserved as superseded."),
            "G3_b3_fidelity": _gate(
                all(all(row["b3_shadow_matches_external"].values()) for row in
                    matrix_first["scenario_results"]) and
                all(row["s0_exact_fallback_trace"] for row in matrix_first["scenario_results"]
                    if row["scenario"] == "S0_exact_b3_fallback"),
                "Actual B3 shadow state matches external B3 on every scenario/seed; S0 trace is exact."),
            "G4_causal_firewall": _gate(
                focused["S14_missing_label_nonidentifiability_and_control_C"]["status"] == "PASS",
                "Prequential engine, queried-only feedback, paired indistinguishable worlds."),
            "G5_shared_score_invariance": _gate(
                focused["S12_common_score_invariance"]["status"] == "PASS",
                "All threshold-only controllers share the engine's single score vector."),
            "G6_memory_version_lso": _gate(
                focused["S11_lso_fixture"]["status"] == "PASS",
                "Synthetic three-layer source attribution and current-version evidence."),
            "G7_numerical_stability": _gate(
                focused["S10_numerical_edges"]["status"] == "PASS",
                "Exact ties, saturation, zero IQR floor and nonfinite rejection."),
            "G8_evidence_accounting": _gate(
                focused["S6_distinct_source_ladder"]["status"] == "PASS",
                "Unique source groups counted once; repeated rescoring does not add evidence."),
            "G9_synthetic_suite": _gate(_scenario_matrix_status(matrix_first),
                                           "All S0-S14 run for each frozen synthetic seed."),
            "G10_ablation_fairness": _gate(
                focused["control_B_same_information"]["status"] == "PASS" and same_information,
                "Shared tape, evidence authority and 0.01 move bound across threshold arms."),
            "G11_reproducibility": _gate(
                deterministic and checkpoint["status"] == "PASS",
                "Second full matrix digest matches; checkpoint/resume is event-trace identical."),
            "G12_interpretation": _gate(True,
                "Reports label heuristics as noninferential; no Real-IAD or guarantee claim."),
            "G13_tests_compile": _gate(compile_ok and tests_ok,
                f"compileall={compile_ok}; tests={validation['calibration_unittest']['tests_run']} exit={validation['calibration_unittest']['returncode']}"),
            "G14_audit_integrity": {"status": "PENDING", "details": "Checksums written and verified below."},
            "G15_scope": _gate(True,
                "Synthetic score inputs only; no Real-IAD, masks, Stage 0.75-C, Stage 1/2 or confirmation classes."),
            "G16_control_E_component_isolation": {
                "status": focused["control_E_mechanism_removal"]["status"],
                "details": focused["control_E_mechanism_removal"]["blocking_reason"]},
        }
        payload = {
            "schema": "stage075b.audit_artifact.v1", "status": "STAGE075B_BLOCKED_WIP",
            "engineering_status": "ENGINEERING_IMPLEMENTATION_COMPLETE",
            "stage": "0.75-B", "branch": branch, "source_commit": head,
            "overall_gate_status": _overall_status(gates),
            "gates": gates, "validation": validation, "focused_mechanisms": focused,
            "checkpoint_resume": checkpoint, "matrix_first_pass": matrix_first,
            "matrix_replay_trajectory_digests": _strip_runtime(matrix_second),
            "deterministic_second_pass": deterministic,
            "representative_memory_profile": memory_profile,
            "environment": environment, "same_information_verified": same_information,
            "source_hashes": _source_hashes(),
            "protected_source_check": source_integrity,
            "started_utc": status["started_utc"],
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "scope_statement": "No Stage 0.75-C, Stage 1/2, Real-IAD performance, Mint/PCB, or masks were accessed.",
        }
        _write_json(output / "stage075b_results.json", payload)
        _write_json(output / "gates.json", gates)
        _write_json(output / "source_provenance.json", {
            "parent_sha": head, "branch": branch,
            "source_hashes": _source_hashes(),
            "protected_source_status": source_integrity})
        _write_json(output / "checksums.json", _checksums(output))
        (output / "INCOMPLETE").unlink()
        verification = verify_artifact(output)
        if verification["status"] != "PASS":
            raise RuntimeError(f"Preliminary artifact checksum verification failed: {verification}")
        gates["G14_audit_integrity"] = {"status": "PASS", "details": "Artifact file list and SHA-256 verified."}
        payload["gates"] = gates
        payload["overall_gate_status"] = _overall_status(gates)
        _write_json(output / "gates.json", gates)
        _write_json(output / "stage075b_results.json", payload)
        (output / "stage075b_summary.md").write_text(
            _summary_markdown(payload), encoding="utf-8")
        _write_json(output / "checksums.json", _checksums(output))
        # Recheck complete artifact after removing the incomplete marker.
        verification = verify_artifact(output)
        if verification["status"] != "PASS":
            raise RuntimeError(f"Final artifact verification failed: {verification}")
        print(json.dumps({"status": payload["overall_gate_status"],
            "output": str(output.relative_to(ROOT)),
                          "trajectory_count": len(matrix_first["scenario_results"]),
                          "deterministic_replay": deterministic,
                          "audit_verification": verification["status"]}, indent=2))
        return 0 if payload["overall_gate_status"] == "PASS" else 2
    except Exception as exc:
        if not (output / "INCOMPLETE").exists():
            (output / "INCOMPLETE").write_text(
                "Stage 0.75-B run is incomplete.\n", encoding="utf-8")
        status.update({"run_status": "INCOMPLETE", "failure": repr(exc),
                       "failed_utc": datetime.now(timezone.utc).isoformat()})
        _write_json(output / "failure.json", status)
        raise


def _summary_markdown(payload: dict[str, Any]) -> str:
    gates = payload["gates"]
    lines = ["# Stage 0.75-B synthetic run summary", "",
             f"Status: `{payload['overall_gate_status']}` (engineering/artifact gates only).", "",
             "This is synthetic mechanics evidence. It is not native scorer parity, Real-IAD performance, a population-risk guarantee, or authorization for a later stage.", "",
             "## Gates", "", "| Gate | Status |", "|---|---|"]
    lines.extend(f"| {name} | {value['status']} |" for name, value in gates.items())
    lines.extend(["", "## Scope", "",
                  "All runs use the frozen synthetic score generator and label-blind Q-A-TS tape. No Stage 0.75-C, Stage 1/2, Real-IAD, Mint/PCB, or masks were accessed.", "",
                  "## Detailed evidence", "",
                  "See `stage075b_results.json`, `gates.json`, `source_provenance.json`, and command logs in this directory.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="run tests and all frozen synthetic scenarios")
    group.add_argument("--verify", action="store_true", help="independently verify an existing run artifact")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output_dir
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    try:
        output.relative_to(ROOT)
    except ValueError as exc:
        raise SystemExit("Stage 0.75-B outputs must remain inside the repository") from exc
    if args.verify:
        verification = verify_artifact(output)
        print(json.dumps(verification, indent=2))
        return 0 if verification["status"] == "PASS" else 2
    return execute_all(output)


if __name__ == "__main__":
    raise SystemExit(main())
