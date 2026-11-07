"""Exact-path protection audit for the FS4 hardening assignment."""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess

from .fs4_threshold_audit import (ROOT, BRANCH, PROTECTED_SOURCES,
                                  BASELINE_UNTRACKED_COUNT, BASELINE_UNTRACKED_SHA256, git)

START = "3bd33076789f66f7086f8f2e165b18d789e2c7d2"
SOURCES = tuple("experiments/calibration/" + name for name in (
    "fs4_errors.py", "fs4_durable_ledger.py", "fs4_durable_runtime.py", "fs4_checkpoint.py", "fs4_resources.py",
    "fs4_query_stress.py", "fs4_hardening_audit.py", "fs4_support_preflight.py", "fs4_greedy_provenance.py",
    "fs4_source_excluded_scorer.py", "fs4_stochastic_bea.py", "fs4_threshold_controllers.py", "fs4_threshold_integration.py"))
DOCS = tuple("docs/calibration/fs4_threshold/HARDENING_R1_" + name + ".md" for name in (
    "CHANGELOG", "FATAL_ERROR_POLICY", "QUERY_STRESS_REPORT", "LEDGER_AND_RESUME_CONTRACT",
    "NATIVE_GPU_PARITY", "RESOURCE_ASSESSMENT", "TEST_MATRIX", "INDEPENDENT_REVIEW_HANDOFF"))
ALLOWLIST = frozenset((*SOURCES, *DOCS, "AGENTS.md",
    "docs/calibration/fs4_threshold/HARDENING_R1_GPU_PARITY_MANIFEST.json",
    "docs/calibration/fs4_threshold/HARDENING_R1_GPU_PARITY_MANIFEST_V2.json",
    "docs/calibration/fs4_threshold/HARDENING_R1_GPU_PARITY_MANIFEST_V3.json",
    "docs/calibration/fs4_threshold/HARDENING_R1_QUERY_STRESS.json",
    "docs/calibration/fs4_threshold/FS4_HARDENING_R1_AUTHORIZATION.md",
    "tests/calibration/test_fs4_hardening_r1.py", "tests/calibration/test_fs4_hardening_cuda.py"))


def audit() -> dict:
    if git("branch", "--show-current").decode().strip() != BRANCH:
        raise RuntimeError("unexpected branch")
    subprocess.run(["git", "merge-base", "--is-ancestor", START, "HEAD"], cwd=ROOT, check=True, capture_output=True)
    changes = set(git("diff", "--name-only", START).decode().splitlines())
    if changes - ALLOWLIST:
        raise RuntimeError(f"unexpected tracked changes: {sorted(changes - ALLOWLIST)}")
    untracked = sorted(x for x in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if x)
    remaining = [p for p in untracked if p not in {x.encode() for x in ALLOWLIST}]
    h = hashlib.sha256(b"\n".join(remaining) + b"\n").hexdigest()
    if len(remaining) != BASELINE_UNTRACKED_COUNT or h != BASELINE_UNTRACKED_SHA256:
        raise RuntimeError("unrelated untracked pathname inventory changed")
    hashes = {}
    for name in PROTECTED_SOURCES:
        baseline = git("show", f"{START}:{name}").replace(b"\r\n", b"\n")
        live = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if live != baseline:
            raise RuntimeError(f"protected native source changed: {name}")
        hashes[name] = hashlib.sha256(baseline).hexdigest()
    for name in SOURCES:
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)
        for node in ast.walk(tree):
            identifier = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""
            if "c64" in identifier.lower():
                raise RuntimeError(f"forbidden operational dependency identifier: {name}:{node.lineno}")
    return {"status": "PASS", "branch": BRANCH, "start": START, "protected_count": len(hashes),
            "protected_sha256": hashes, "tracked_changes": sorted(changes),
            "untracked_count": len(remaining), "untracked_sha256": h, "identifier_scan": "PASS"}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
