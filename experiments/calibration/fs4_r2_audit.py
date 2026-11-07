"""Exact-path R2 audit; never opens untracked historical output contents."""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess

from .fs4_threshold_audit import (ROOT, BRANCH, PROTECTED_SOURCES,
    BASELINE_UNTRACKED_COUNT, BASELINE_UNTRACKED_SHA256, git)

START = "d99c3260101564ca5b910f0522f1271ab8205efc"
SOURCES = tuple("experiments/calibration/" + name for name in (
    "fs4_r2_security.py", "fs4_r2_trust.py", "fs4_r2_context.py", "fs4_r2_stage_b.py", "fs4_r2_audit.py",
    "fs4_durable_ledger.py", "fs4_durable_runtime.py", "fs4_checkpoint.py", "fs4_threshold_integration.py"))
DOCS = tuple("docs/calibration/fs4_threshold/HARDENING_R2_" + name + ".md" for name in (
    "TRUST_AND_EXECUTION", "STAGE_B_AND_ACL", "TEST_MATRIX", "INDEPENDENT_REVIEW_HANDOFF"))
ALLOWLIST = frozenset((*SOURCES, *DOCS, "AGENTS.md",
    "docs/calibration/fs4_threshold/FS4_HARDENING_R2_AUTHORIZATION.md",
    "tests/calibration/test_fs4_hardening_r1.py", "tests/calibration/test_fs4_hardening_r2.py",
    "tests/calibration/test_fs4_hardening_r2_cuda.py"))


def audit() -> dict:
    if git("branch", "--show-current").decode().strip() != BRANCH:
        raise RuntimeError("unexpected branch")
    subprocess.run(["git", "merge-base", "--is-ancestor", START, "HEAD"], cwd=ROOT, check=True, capture_output=True)
    changes = set(git("diff", "--name-only", START).decode().splitlines())
    if changes - ALLOWLIST:
        raise RuntimeError(f"unexpected tracked changes: {sorted(changes - ALLOWLIST)}")
    paths = sorted(x for x in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if x)
    remaining = [x for x in paths if x not in {name.encode() for name in ALLOWLIST}]
    sha = hashlib.sha256(b"\n".join(remaining) + b"\n").hexdigest()
    if len(remaining) != BASELINE_UNTRACKED_COUNT or sha != BASELINE_UNTRACKED_SHA256:
        raise RuntimeError("unrelated untracked pathname inventory changed")
    checked = {}
    for name in PROTECTED_SOURCES:
        old = git("show", f"{START}:{name}").replace(b"\r\n", b"\n")
        current = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if old != current:
            raise RuntimeError(f"protected historical source changed: {name}")
        checked[name] = hashlib.sha256(old).hexdigest()
    for name in SOURCES:
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)
        for node in ast.walk(tree):
            token = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""
            if "c64" in token.lower():
                raise RuntimeError(f"forbidden operational identifier: {name}:{node.lineno}")
    return {"status": "PASS", "branch": BRANCH, "start": START, "protected": len(checked),
            "untracked_count": len(remaining), "untracked_sha256": sha,
            "runtime_identifier_scan": "PASS", "tracked_changes": sorted(changes)}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
