"""Transitive local source hashing and protected historical-source baseline."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTECTED_BASELINE_PATH = ROOT / "docs" / "calibration" / "PROTECTED_SOURCE_BASELINE.json"
PROTECTED_SOURCE_COMMIT = "9d2d20df0bd94629eaea0f8cbd6a6f84ebc28a44"
EXPLICIT_DYNAMIC_MODULES = (
    "vmb_visionad_new", "test_phase3_AIF_simplified_v22",
    "p0_protocol_audit", "a0_a1_strict4_matched", "v02_memory",
)
PROTECTED_HISTORICAL_FILES = (
    "vmb_visionad_new.py", "test_phase3_AIF_simplified_v22.py",
    "experiments/p0_protocol_audit.py", "experiments/a0_a1_strict4_matched.py",
    "experiments/v02_memory.py", "experiments/v02_architecture_core.py",
    "experiments/risk_safe/native_bridge.py", "experiments/risk_safe/shadow_memory.py",
    "experiments/risk_safe/preflight.py", "experiments/risk_safe/query_protocol.py",
    "experiments/risk_safe/d2_threshold_calibration.py",
    "experiments/risk_safe/p01_falsification_native.py",
)
CALIBRATION_ENTRYPOINTS = (
    "experiments.calibration.native",
    "experiments.calibration.native_event_replay",
    "experiments.calibration.native_greedy_matrix",
    "experiments.calibration.native_parity_smoke",
    "experiments.calibration.anchor_ledger",
    "experiments.calibration.method_registry",
    "experiments.calibration.runner",
)


def _canonical_source_bytes(raw: bytes) -> bytes:
    """Normalize checkout CRLF only; Git-source identity is line-ending stable."""
    return raw.replace(b"\r\n", b"\n")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _module_candidates(module: str) -> tuple[Path, ...]:
    if not module:
        return ()
    rel = Path(*module.split("."))
    return (ROOT / rel.with_suffix(".py"), ROOT / rel / "__init__.py")


def _resolve_module(module: str) -> Path | None:
    for candidate in _module_candidates(module):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _imports(path: Path) -> set[Path]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeError):
        return set()
    result: set[Path] = set()
    module_name = path.relative_to(ROOT).with_suffix("").as_posix().replace("/", ".")
    package = (module_name if path.name == "__init__.py" else
               module_name.rsplit(".", 1)[0] if "." in module_name else "")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolved = _resolve_module(alias.name)
                if resolved is not None:
                    result.add(resolved)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                if node.level > len(base):
                    continue
                base = base[:len(base) - node.level + 1]
                module = ".".join([*base, *([node.module] if node.module else [])])
            else:
                module = node.module or ""
            resolved_base = _resolve_module(module)
            if resolved_base is not None:
                result.add(resolved_base)
            for alias in node.names:
                if alias.name == "*":
                    continue
                resolved_child = _resolve_module(".".join(x for x in (module, alias.name) if x))
                if resolved_child is not None:
                    result.add(resolved_child)
    return result


def local_dependency_paths() -> tuple[str, ...]:
    """Resolve Python-local direct and transitive imports of native entrypoints."""
    queue = [path for module in CALIBRATION_ENTRYPOINTS
             if (path := _resolve_module(module)) is not None]
    for module in EXPLICIT_DYNAMIC_MODULES:
        path = _resolve_module(module)
        if path is None and module in {"p0_protocol_audit", "a0_a1_strict4_matched", "v02_memory"}:
            path = _resolve_module("experiments." + module)
        if path is not None:
            queue.append(path)
    visited: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in visited or not path.is_file():
            continue
        visited.add(path)
        queue.extend(child for child in _imports(path) if child not in visited)
    return tuple(sorted(path.relative_to(ROOT).as_posix() for path in visited))


def protected_dependency_paths() -> tuple[str, ...]:
    resolved = {path for path in local_dependency_paths()
                if not path.startswith(("experiments/calibration/", "tests/", "docs/"))}
    resolved.update(path for path in PROTECTED_HISTORICAL_FILES
                    if (ROOT / path).is_file())
    return tuple(sorted(resolved))


def _git_show(commit: str, path: str) -> bytes:
    result = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT,
                            capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Cannot read protected baseline {commit}:{path}: " +
                           result.stderr.decode("utf-8", "replace").strip())
    return result.stdout


def generate_protected_baseline() -> dict[str, Any]:
    """Build the static manifest from the reviewed pre-fourth-pass source commit."""
    paths = protected_dependency_paths()
    files = {path: {"baseline_commit": PROTECTED_SOURCE_COMMIT,
                    "sha256": _sha(_canonical_source_bytes(_git_show(
                        PROTECTED_SOURCE_COMMIT, path)))} for path in paths}
    shadow_path = "experiments/risk_safe/shadow_memory.py"
    if shadow_path not in files:
        raise RuntimeError("Historical shadow_memory.py is absent from resolved dependency closure")
    return {
        "schema": "calibration.protected_source_baseline.v1",
        "baseline_commit": PROTECTED_SOURCE_COMMIT,
        "hash_algorithm": "sha256",
        "normalization": "Git blob bytes; worktree CRLF canonicalized to LF before comparison",
        "historical_shadow_memory_required_exact_sha256": files[shadow_path]["sha256"],
        "dependency_paths": list(paths),
        "files": files,
    }


def verify_protected_sources() -> dict[str, Any]:
    """Compare current source with the static baseline and require graph coverage."""
    report: dict[str, Any] = {
        "status": "BLOCKED", "baseline_path": "docs/calibration/PROTECTED_SOURCE_BASELINE.json",
        "baseline_commit": PROTECTED_SOURCE_COMMIT,
    }
    try:
        baseline = json.loads(PROTECTED_BASELINE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report["reason"] = f"Protected source baseline is unavailable or invalid: {exc}"
        return report
    resolved = set(protected_dependency_paths())
    recorded = set(baseline.get("dependency_paths", ()))
    files = baseline.get("files", {})
    missing = sorted(resolved - recorded)
    stale = sorted(recorded - resolved)
    failures: list[dict[str, str]] = []
    for path in sorted(recorded):
        row = files.get(path)
        if not isinstance(row, dict):
            failures.append({"path": path, "reason": "missing_static_hash"})
            continue
        try:
            current = (ROOT / Path(path)).read_bytes()
            base = _git_show(str(row.get("baseline_commit", PROTECTED_SOURCE_COMMIT)), path)
        except (OSError, RuntimeError) as exc:
            failures.append({"path": path, "reason": str(exc)})
            continue
        expected = str(row.get("sha256", ""))
        current_sha = _sha(_canonical_source_bytes(current))
        git_sha = _sha(_canonical_source_bytes(base))
        if git_sha != expected:
            failures.append({"path": path, "reason": "static_hash_does_not_match_baseline_commit"})
        if current_sha != expected:
            failures.append({"path": path, "reason": "worktree_differs_from_protected_baseline"})
    shadow = ROOT / "experiments" / "risk_safe" / "shadow_memory.py"
    shadow_raw_sha = _sha(shadow.read_bytes()) if shadow.is_file() else None
    required_shadow_sha = baseline.get("historical_shadow_memory_required_exact_sha256")
    shadow_exact = bool(shadow_raw_sha and shadow_raw_sha == required_shadow_sha)
    if not shadow_exact:
        failures.append({"path": "experiments/risk_safe/shadow_memory.py",
                         "reason": "historical_byte_level_sha256_mismatch"})
    report.update({
        "resolved_dependency_paths": sorted(resolved),
        "recorded_dependency_paths": sorted(recorded),
        "unrecorded_resolved_dependencies": missing,
        "recorded_but_unresolved_dependencies": stale,
        "failures": failures,
        "dependency_hash_count": len(recorded),
        "shadow_memory_worktree_sha256": shadow_raw_sha,
        "shadow_memory_baseline_sha256": required_shadow_sha,
        "shadow_memory_byte_equal_to_baseline": shadow_exact,
        "status": "PASS" if not (missing or stale or failures) else "BLOCKED",
    })
    return report


def source_fingerprint_paths() -> tuple[str, ...]:
    """All resolved local dependencies, including the transitive old sources."""
    return tuple(sorted(set(local_dependency_paths()) | set(protected_dependency_paths())))
