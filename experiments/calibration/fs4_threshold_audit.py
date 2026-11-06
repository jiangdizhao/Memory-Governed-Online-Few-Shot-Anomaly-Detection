"""Read-only exact-path, baseline-inventory and FS4 runtime dependency audit."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BRANCH = "calibration-stage0-contract-20261029"
START = "ecade1ec69fa040ec958a8b8ffeddd2e706cf7da"
BASELINE_UNTRACKED_COUNT = 7030
BASELINE_UNTRACKED_SHA256 = "5d036161ab992ad68c8f98e7b03e91199fc7fa7b4d170603936c57064488dd18"
SOURCES = (
    "experiments/calibration/fs4_threshold_core.py",
    "experiments/calibration/fs4_stochastic_bea.py",
    "experiments/calibration/fs4_greedy_provenance.py",
    "experiments/calibration/fs4_source_excluded_scorer.py",
    "experiments/calibration/fs4_threshold_controllers.py",
    "experiments/calibration/fs4_threshold_integration.py",
    "experiments/calibration/fs4_threshold_synthetic_smoke.py",
    "experiments/calibration/fs4_threshold_audit.py",
)
DOCS = tuple("docs/calibration/fs4_threshold/" + name for name in (
    "SCIENTIFIC_CONTRACT.md", "SELECTION_BIAS_AND_IDENTIFIABILITY.md",
    "SELF_MATCHING_AND_SOURCE_PROVENANCE.md", "NATIVE_SCORER_AND_WRITER_FIDELITY.md",
    "SYNTHETIC_TEST_RESULTS.md", "IMPLEMENTATION_REVIEW_HANDOFF.md",
    "FUTURE_INTEGRATED_EXPERIMENT_PLAN.md"))
ALLOWLIST = frozenset((*SOURCES, *DOCS, "AGENTS.md",
    "docs/calibration/FS4_THRESHOLD_IMPLEMENTATION_AUTHORIZATION.md",
    "tests/calibration/test_fs4_threshold.py"))
PROTECTED_SOURCES = (
    "test_phase3_AIF_simplified_v22.py", "vmb_visionad_new.py",
    "experiments/v02_memory.py", "experiments/p0_protocol_audit.py",
    "experiments/a0_a1_strict4_matched.py", "experiments/a0_a1_memory_methods.py",
    "experiments/tmra_compact_memory.py", "experiments/tmra_recurrence_core.py",
    "experiments/risk_safe/native_bridge.py", "experiments/calibration/native.py",
    "experiments/risk_safe/shadow_memory.py", "experiments/calibration/profiles.py",
    "experiments/calibration/memory_state.py", "experiments/calibration/protocol.py",
    "experiments/calibration/e6_bea_policy.py", "experiments/calibration/e6_bea_d2.py",
    "experiments/calibration/e6_bea_closed_loop.py", "experiments/calibration/e6_bea_d2_runner.py",
    "experiments/calibration/query_tape.py", "experiments/calibration/r1_matched_correction_pilot.py",
    "experiments/calibration/r2_exp2a_c1_strength.py", "experiments/calibration/r3_fullstream_2x2.py",
    "experiments/calibration/r7_shuffled_correction_4class.py",
    "experiments/calibration/r9_correction_mechanism_falsification.py",
    "experiments/calibration/controllers.py", "experiments/calibration/a_calibration_scale.py",
    "experiments/calibration/a_calibration_continuation.py",
)


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def audit() -> dict[str, object]:
    if git("branch", "--show-current").decode().strip() != BRANCH:
        raise RuntimeError("unexpected branch")
    subprocess.run(["git", "merge-base", "--is-ancestor", START, "HEAD"], cwd=ROOT, check=True, capture_output=True)
    changed = set(git("diff", "--name-only", START).decode("utf-8").splitlines())
    if changed - ALLOWLIST:
        raise RuntimeError(f"unexpected tracked changes: {sorted(changed - ALLOWLIST)}")
    paths = sorted(x for x in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if x)
    excluded = {p.encode("utf-8") for p in ALLOWLIST}
    residual = [p for p in paths if p not in excluded]
    digest = hashlib.sha256(b"\n".join(residual) + b"\n").hexdigest()
    if len(residual) != BASELINE_UNTRACKED_COUNT or digest != BASELINE_UNTRACKED_SHA256:
        raise RuntimeError("pre-existing untracked path inventory changed")
    checked = {}
    for name in PROTECTED_SOURCES:
        blob = git("show", f"{START}:{name}").replace(b"\r\n", b"\n")
        current = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if blob != current:
            raise RuntimeError(f"protected historical source changed: {name}")
        checked[name] = hashlib.sha256(blob).hexdigest()
    violations = []
    for name in SOURCES:
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)
        for node in ast.walk(tree):
            token = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""
            if "c64" in token.lower():
                violations.append(f"{name}:{getattr(node, 'lineno', '?')}:{token}")
    if violations:
        raise RuntimeError(f"forbidden runtime reference identifiers: {violations}")
    return {"status": "PASS", "branch": BRANCH, "starting_head": START,
            "current_head": git("rev-parse", "HEAD").decode().strip(),
            "tracked_changes": sorted(changed), "allowlist": sorted(ALLOWLIST),
            "protected_source_sha256": checked,
            "preexisting_untracked_count": len(residual), "preexisting_untracked_sha256": digest,
            "runtime_identifier_scan": "PASS", "real_data_read": False,
            "limitation": "Untracked historical content was not opened or hashed; pathname inventory is checked."}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
