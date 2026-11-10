"""Read-only source/interface audit for Gate 1 historical dependencies."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
HISTORICAL_BINDINGS = {
    "r7_native": "experiments/calibration/r7_shuffled_correction_4class.py",
    "r9_full_c1": "experiments/calibration/r9_correction_mechanism_falsification.py",
    "r10_signed_knn": "experiments/calibration/r10_signed_knn_baseline.py",
    "e6_bea": "experiments/calibration/e6_bea_policy.py",
    "exp_c_utilities": "experiments/calibration/exp_c_memory_evidence.py",
    "a3b_kcenter": "experiments/calibration/a_calibration_continuation.py",
    "a4_static_reference": "experiments/calibration/a4_equal_budget_allocation.py",
}
REQUIRED_FUNCTIONS = {
    "r7_native": ("_refresh_b3", "_score_c64_base"),
    "r9_full_c1": ("_score_codebook", "_state_digest_for_codebook"),
    "r10_signed_knn": ("_corrected_c2_score", "_score_c2_c64"),
    "e6_bea": ("compute_bea_components",),
    "exp_c_utilities": ("d_band_utility", "gate_decomposition"),
    "a3b_kcenter": ("a3b_candidate_quotas", "a3b_candidate_identity", "a3b_kcenter_cuda"),
    "a4_static_reference": ("_higher_q90", "_kcenter_candidates"),
}


class SourceAuditError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature_map(path: Path) -> dict[str, str]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=path.name)
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise SourceAuditError(f"historical source could not be inspected ({type(exc).__name__})") from None
    result: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = ast.unparse(node.args)
    return result


def audit_historical_interfaces(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root)
    report: dict[str, Any] = {"schema": "final-benchmark.source-audit.v1", "bindings": {}}
    for name, relative in HISTORICAL_BINDINGS.items():
        path = root / relative
        if not path.is_file():
            raise SourceAuditError(f"required historical implementation is missing: {relative}")
        signatures = _signature_map(path)
        required = REQUIRED_FUNCTIONS[name]
        missing = [function for function in required if function not in signatures]
        if missing:
            raise SourceAuditError(f"{relative} lacks a registered interface: {', '.join(missing)}")
        report["bindings"][name] = {
            "relative_path": relative,
            "sha256": _sha(path),
            "required_signatures": {function: signatures[function] for function in required},
            "status": "SOURCE_PRESENT_SIGNATURES_RESOLVED_NUMERICAL_PARITY_PENDING",
        }
    # Interface checks for known critical call contracts.
    r10 = report["bindings"]["r10_signed_knn"]["required_signatures"]["_corrected_c2_score"]
    if "group_sha256" not in r10 or "group_sha=" in r10:
        raise SourceAuditError("R10 scorer group identity keyword contract is unexpected")
    return {
        **report,
        "scientific_behavior": {
            "r7_greedy_b3": "historical source reused; no numerical Gate 2 parity executed",
            "r9_full_c1": "historical source interface present; joined native backend not installed",
            "r10_signed_knn": "historical source interface present; joined native backend not installed",
            "e6_bea_components": "source function present; new group controller implemented separately",
            "exp_c_d_band": "source function present; source numerical parity pending",
            "a3b_kcenter": "source function present; adapter binds directly; CUDA parity not run",
            "static_c32_threshold": "uses existing A4 higher empirical q90 arithmetic",
        },
        "real_inference_or_gpu_used": False,
    }
