"""Private durable ledger for the explicitly non-adversarial local research mode."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .fs4_durable_ledger import LedgerIntegrityError, SyntheticTestLedger, canonical, digest
from .fs4_r2_context import validate_execution_manifest
from .fs4_r2_security import WindowsPrivateStore, validate_private_acl

RESEARCH_SOURCES = frozenset({
    "experiments/calibration/fs4_research_ledger.py",
    "experiments/calibration/fs4_research_context.py",
    "experiments/calibration/fs4_research_worker.py",
    "experiments/calibration/fs4_research_evaluator.py",
    "experiments/calibration/fs4_research_stage_a_verify.py",
})


class ResearchLedger(SyntheticTestLedger):
    """Research-scoped ledger; no arbitrary Stage B callback is accepted."""

    def __init__(self, root: Path, manifest: Mapping[str, Any], *, create: bool) -> None:
        supplied = dict(manifest)
        if supplied.get("execution_scope") != "TRUSTED_LOCAL_RESEARCH":
            raise LedgerIntegrityError("explicit TRUSTED_LOCAL_RESEARCH scope required")
        if not RESEARCH_SOURCES.issubset(supplied.get("source_fingerprints", {})):
            raise LedgerIntegrityError("research worker sources are not registered")
        execution = supplied.get("research_execution_manifest")
        if not isinstance(execution, dict):
            raise LedgerIntegrityError("research execution registration missing")
        self.execution_manifest = validate_execution_manifest(execution, supplied)
        self._execution_manifest = self.execution_manifest
        self._initial_manifest_digest = digest(supplied)
        self._trusted_context = None
        self._active_provenance = None
        self._reading_existing = True
        self._private = WindowsPrivateStore(Path(root), create=create)
        if create:
            self._private.write("manifest.json", canonical(supplied) + b"\n")
            self._private.write("events.jsonl", b"")
            self._private.write("writer.lock", b"0")
        else:
            for name in ("manifest.json", "events.jsonl", "writer.lock"):
                self._private.read(name)
        super().__init__(root, supplied, create=False)
        self._reading_existing = False
        self.verify_frozen_sources()

    def verify_frozen_sources(self) -> None:
        root = Path(__file__).resolve().parents[2]
        revision = self.manifest["source_commit"]
        if subprocess.run(["git", "merge-base", "--is-ancestor", revision, "HEAD"],
                          cwd=root, capture_output=True).returncode:
            raise LedgerIntegrityError("research source revision is not an ancestor of HEAD")
        for name, expected in self.manifest["source_fingerprints"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise LedgerIntegrityError("invalid frozen source path")
            value = path.read_bytes()
            committed = subprocess.run(["git", "show", f"{revision}:{name}"], cwd=root, capture_output=True)
            if (hashlib.sha256(value).hexdigest() != expected or committed.returncode or
                    committed.stdout.replace(b"\r\n", b"\n") != value.replace(b"\r\n", b"\n")):
                raise LedgerIntegrityError("research source differs from registered commit")

    def verify_live_sources(self) -> None:
        # Source is fully checked at startup, recovery and final category seal.
        if digest(self.manifest) != self._initial_manifest_digest:
            raise LedgerIntegrityError("research manifest changed")

    def _append(self, kind: str, payload: dict[str, Any]) -> str:
        self.verify_live_sources()
        validate_private_acl(self.root, directory=True)
        validate_private_acl(self.path, directory=False)
        if self._private.read("manifest.json") != canonical(self.manifest) + b"\n":
            raise LedgerIntegrityError("research manifest bytes changed")
        return super()._append(kind, payload)

    def _apply(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == "CATEGORY":
            order = self.execution_manifest["category_order"]
            if len(self.category_seals) >= len(order) or payload.get("category") != order[len(self.category_seals)]:
                raise LedgerIntegrityError("research category order differs")
        if kind == "PREDICTION":
            event = payload["event"]
            rows = self.execution_manifest["categories"][self.current_category]["events"]
            rows = [row for row in rows if row["position"] == event.get("position")]
            p = event.get("trusted_event_provenance")
            if len(rows) != 1 or not isinstance(p, dict):
                raise LedgerIntegrityError("missing registered research event provenance")
            row = rows[0]
            expected = {"schema": "FS4_TRUSTED_EVENT_V1", "dataset_identity_sha256": self.manifest["dataset_identity_sha256"],
                "manifest_sha256": digest(self.execution_manifest), "category": self.current_category,
                "image_content_sha256": row["content_sha256"], "geometry": [row["width"], row["height"]],
                "model_state_sha256": self.manifest["model_state_sha256"], "feature_sha256": event.get("feature_sha256"),
                "source_commit": self.manifest["source_commit"],
                "fixed_support_sha256": self.manifest["fixed_support_sha256_by_category"][self.current_category]}
            if (event.get("event") != row["event"] or event.get("group") != row["group"] or
                    any(p.get(key) != value for key, value in expected.items()) or
                    not isinstance(p.get("cls_sha256"), str) or len(p["cls_sha256"]) != 64 or
                    (not self._reading_existing and p != self._active_provenance)):
                raise LedgerIntegrityError("research event producer or identity differs")
        if kind in ("STAGE_B_AUTH", "STAGE_B_RESULT", "STAGE_B_SEAL", "COMPLETE"):
            raise LedgerIntegrityError("research Stage B requires the separate evaluator")
        super()._apply(kind, payload)

    def authorize_stage_b(self, *_args: Any, **_kwargs: Any) -> str:
        raise LedgerIntegrityError("research Stage B requires the separate evaluator")

    def synthetic_stage_b_result(self, _result: bytes) -> str:
        raise LedgerIntegrityError("synthetic callback result forbidden in research mode")
