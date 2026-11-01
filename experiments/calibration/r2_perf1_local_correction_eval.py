"""Supervised launcher and Stage A/Stage B process boundary for R2-PERF-1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Sequence


def _configuration_record() -> dict[str, Any]:
    from experiments.calibration.r2_perf1_common import (
        AMPLITUDE, ANCHOR_CAPACITY, CONFIGURATION_SCHEMA, K_NEIGHBORS,
        LAMBDAS, PRIMARY_LAMBDA, PRIMARY_RADIUS, RADII)
    return {
        "schema": CONFIGURATION_SCHEMA,
        "radii": [float(value) for value in RADII],
        "regularizers": [float(value) for value in LAMBDAS],
        "amplitude": AMPLITUDE,
        "knn": K_NEIGHBORS,
        "capacity": ANCHOR_CAPACITY,
        "primary": {"radius": PRIMARY_RADIUS, "lambda": PRIMARY_LAMBDA},
        "c2_gated": "strict cosine distance < R, nearest k=5, compact phi, regularized signed mean",
        "c3": "all legal local anchors, beta=0.20*sign, kappa=1, compact phi, regularized evidence",
    }


def _worker_env(stage: str, nonce: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "R2_PERF1_WORKER": stage,
        "R2_PERF1_SUPERVISED_NONCE": nonce,
    })
    return env


def _require_orchestrator() -> None:
    from experiments.calibration.r2_perf1_common import Perf1Error, require_expected_repository
    if os.name != "nt" or os.environ.get("R2_PERF1_ORCHESTRATOR") != "1":
        raise Perf1Error("Orchestration requires the explicit Windows Job Object launcher")
    nonce = os.environ.get("R2_PERF1_SUPERVISED_NONCE", "")
    if re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        raise Perf1Error("Orchestrator lacks its supervisor-bound launch nonce")
    from experiments.calibration.r1_runtime_supervisor import current_process_is_in_job
    if not current_process_is_in_job():
        raise Perf1Error("Orchestrator is not contained in the Windows Job Object")
    require_expected_repository()


def _orchestrate(output_root: Path) -> int:
    from experiments.calibration.r2_perf1_common import (
        OutputBudget, Perf1Error, ROOT, require_plain_path, source_fingerprints,
        utc_now, verify_source_fingerprints)
    _require_orchestrator()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output_root = require_plain_path(output_root, must_exist=True)
    budget = OutputBudget(output_root, allow_existing=True)
    try:
        manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
        verify_source_fingerprints(manifest["source_fingerprints"])
        nonce = os.environ["R2_PERF1_SUPERVISED_NONCE"]
        python = Path(sys.executable).resolve()
        expected_python = Path(r"D:\anaconda3\envs\torch\python.exe").resolve()
        if os.path.normcase(str(python)) != os.path.normcase(str(expected_python)):
            raise Perf1Error(f"Unexpected orchestration interpreter: {python}")

        for stage in ("stage_a", "stage_b"):
            command = [sys.executable, "-m", f"experiments.calibration.r2_perf1_{stage}",
                       "--output-root", str(output_root)]
            if stage == "stage_b":
                command.extend(["--bootstrap-replicates", "1000"])
            completed = subprocess.run(
                command, cwd=ROOT, env=_worker_env(stage, nonce),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if completed.returncode != 0:
                raise Perf1Error(f"{stage} process failed with exit code {completed.returncode}")
            verify_source_fingerprints(manifest["source_fingerprints"])
            from experiments.calibration.r2_perf1_common import require_expected_repository
            require_expected_repository()
            if stage == "stage_a":
                seal_path = output_root / "stage_a_sealed.json"
                seal = json.loads(seal_path.read_text(encoding="utf-8"))
                from experiments.calibration.r2_perf1_common import sha256_file
                if (seal.get("status") != "SEALED" or seal.get("event_count") != 512 or
                        sha256_file(output_root / "stage_a_scores.jsonl") !=
                        seal.get("score_table_sha256") or
                        seal.get("outcome_labels_read") is not False or
                        seal.get("test_metadata_opened") is not False):
                    raise Perf1Error("Stage A score seal failed before the Stage B launch gate")
        budget.write_json("orchestrator_complete.json", {
            "schema": "calibration.r2_perf1.orchestrator_complete.v1",
            "status": "COMPLETE",
            "stage_a_sealed": True,
            "stage_b_started_after_score_seal": True,
            "source_fingerprints": source_fingerprints(),
            "completed_utc": utc_now(),
        })
        return 0
    except Exception as exc:
        try:
            budget.write_json("failure.json", {
                "schema": "calibration.r2_perf1.failure.v1",
                "status": "FAILED",
                "stage": os.environ.get("R2_PERF1_WORKER", "orchestrator"),
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "automatic_restart": False,
                "failed_utc": utc_now(),
            }, failure_reserved=True)
        except FileExistsError:
            pass
        return 2


def run_experiment(*, authorize: bool, run_id: str | None = None) -> dict[str, Any]:
    from experiments.calibration.r1_runtime_supervisor import (
        MAX_RUNTIME_SECONDS, run_supervised_child)
    from experiments.calibration.r2_perf1_common import (
        EVIDENCE_RELATIVE, FAILURE_RESERVE_BYTES, MAX_OUTPUT_BYTES,
        MAX_RUNTIME_SECONDS as PERF_RUNTIME_SECONDS, OUTPUT_PARENT,
        EXPECTED_BRANCH, EXPECTED_HEAD, MIN_FREE_BYTES, ROOT, OutputBudget,
        Perf1Error, create_output_namespace, git_identity, new_run_id,
        require_expected_repository, sha256_file, sha256_json,
        source_fingerprints, utc_now, verify_source_fingerprints)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if authorize is not True:
        raise Perf1Error("Pass --authorize-r2-perf1 for this bounded development evaluation")
    expected_python = Path(r"D:\anaconda3\envs\torch\python.exe").resolve()
    if os.path.normcase(str(Path(sys.executable).resolve())) != os.path.normcase(str(expected_python)):
        raise Perf1Error(f"R2-PERF-1 requires {expected_python}; found {sys.executable}")
    repo = require_expected_repository()
    if repo != {"branch": EXPECTED_BRANCH, "head": EXPECTED_HEAD}:
        raise Perf1Error("Repository identity changed after its expected start-state check")
    run_id = run_id or new_run_id()
    if re.fullmatch(r"[0-9]{8}T[0-9]{6}_[0-9a-f]{8}", run_id) is None:
        raise Perf1Error("Run ID must be unique timestamp/random-token format")
    output_root = create_output_namespace(run_id)
    free_bytes = int(__import__("shutil").disk_usage(OUTPUT_PARENT).free)
    config = _configuration_record()
    config_sha = sha256_json(config)
    sources = source_fingerprints()
    evidence_manifest = ROOT / EVIDENCE_RELATIVE / "run_manifest.json"
    manifest = {
        "schema": "calibration.r2_perf1.run_manifest.v1",
        "status": "INCOMPLETE",
        "run_id": run_id,
        "authorization_basis": "explicit user authorization in the R2-PERF-1 task",
        "execution_scope": "switch only; seed 0; first 512 frozen TEST positions",
        "repository": repo,
        "expected_repository": {"branch": EXPECTED_BRANCH, "head": EXPECTED_HEAD},
        "python_executable": str(expected_python),
        "cuda_visible_devices": "",
        "gpu_inference": False,
        "dino_loaded": False,
        "image_decoding": False,
        "event_count": 512,
        "expected_source_group_count": 103,
        "historical_evidence_relative_path": EVIDENCE_RELATIVE.as_posix(),
        "historical_evidence_run_id": "r1s4b-switch-s0-512-20261031-a1",
        "historical_evidence_manifest_sha256": sha256_file(evidence_manifest),
        "configuration": config,
        "configuration_sha256": config_sha,
        "primary_parameters": {"R": 0.10, "lambda": 0.10, "amplitude": 0.20,
                               "k": 5, "capacity": 256},
        "parameter_selection_on_outcomes": False,
        "frozen_threshold": 0.5165398120880127,
        "r1_c64_theta0_higher_q90": 0.5165398120880127,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "free_bytes_before_execution": free_bytes,
        "output_budget_bytes": MAX_OUTPUT_BYTES,
        "failure_reserve_bytes": FAILURE_RESERVE_BYTES,
        "runtime_deadline_seconds": PERF_RUNTIME_SECONDS,
        "supervisor": "Windows Job Object; child suspended into job; descendants inherit job; kill-on-close",
        "output_namespace_unique_and_non_overwriting": True,
        "automatic_restart": False,
        "stage_b_launch_gate": "Stage A score table SHA-256 seal must verify before Stage B process starts",
        "source_fingerprints": sources,
        "started_utc": utc_now(),
    }
    budget = OutputBudget(output_root, allow_existing=True)
    budget.write_json("run_manifest.json", manifest)
    nonce = os.urandom(32).hex()
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "R2_PERF1_ORCHESTRATOR": "1",
        "R2_PERF1_SUPERVISED_NONCE": nonce,
    })
    command = [sys.executable, "-m", "experiments.calibration.r2_perf1_local_correction_eval",
               "--orchestrate", "--output-root", str(output_root)]
    if MAX_RUNTIME_SECONDS != PERF_RUNTIME_SECONDS or MAX_RUNTIME_SECONDS != 120 * 60:
        raise Perf1Error("R1 supervisor deadline differs from the authorized 120-minute bound")
    supervisor_result: dict[str, Any]
    try:
        result = run_supervised_child(
            command, timeout_seconds=PERF_RUNTIME_SECONDS, env=env, cwd=ROOT,
            output_dir=None)
        supervisor_result = {
            "schema": "calibration.r2_perf1.supervisor_status.v1",
            **result.__dict__,
            "source_fingerprints": source_fingerprints(),
            "output_preflight_free_bytes": free_bytes,
            "minimum_free_bytes": MIN_FREE_BYTES,
            "automatic_restart": False,
            "finished_utc": utc_now(),
        }
    except Exception as exc:
        supervisor_result = {
            "schema": "calibration.r2_perf1.supervisor_status.v1",
            "status": "SUPERVISOR_FAILURE",
            "error_type": type(exc).__name__, "error": str(exc)[:2000],
            "automatic_restart": False, "finished_utc": utc_now(),
        }
    try:
        budget = OutputBudget(output_root, allow_existing=True)
        budget.write_json("supervisor_status.json", supervisor_result, failure_reserved=True)
        if supervisor_result.get("status") != "COMPLETED":
            try:
                budget.write_json("failure.json", {
                    "schema": "calibration.r2_perf1.failure.v1",
                    "status": str(supervisor_result.get("status", "SUPERVISOR_FAILURE")),
                    "error_type": str(supervisor_result.get("error_type", "ChildFailure")),
                    "error": str(supervisor_result.get("error", "Supervised execution did not complete"))[:2000],
                    "automatic_restart": False,
                    "failed_utc": utc_now(),
                }, failure_reserved=True)
            except FileExistsError:
                pass
    except Exception as exc:
        try:
            budget.write_json("failure.json", {
                "schema": "calibration.r2_perf1.failure.v1",
                "status": "SUPERVISOR_METADATA_WRITE_FAILED",
                "error_type": type(exc).__name__, "error": str(exc)[:2000],
                "automatic_restart": False,
                "failed_utc": utc_now(),
            }, failure_reserved=True)
        except Exception:
            pass
    verify_source_fingerprints(sources)
    final_path = output_root / "stage_b_report.json"
    report = json.loads(final_path.read_text(encoding="utf-8")) if final_path.is_file() else None
    return {
        "run_id": run_id,
        "output_directory": str(output_root),
        "supervisor": supervisor_result,
        "report": report,
    }


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-r2-perf1", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--orchestrate", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    if args.orchestrate:
        if args.output_root is None:
            parser.error("--orchestrate requires --output-root")
        return _orchestrate(args.output_root)
    try:
        result = run_experiment(authorize=args.authorize_r2_perf1, run_id=args.run_id)
        print(json.dumps(result, sort_keys=True), flush=True)
        supervisor = result["supervisor"]
        if supervisor.get("status") != "COMPLETED":
            return 2
        if not result.get("report"):
            return 2
        return 0
    except Exception as exc:
        print(f"R2_PERF1_LAUNCH_FAILURE {type(exc).__name__}: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
