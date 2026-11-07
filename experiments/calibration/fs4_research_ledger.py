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
    "experiments/calibration/fs4_r4_preparation.py",
    "experiments/calibration/fs4_r4_blinded.py",
    "experiments/calibration/fs4_r4_broker.py",
    "experiments/calibration/fs4_r4_worker.py",
    "experiments/calibration/fs4_r4_mapping_verify.py",
    "experiments/calibration/fs4_r4_evaluator.py",
    "experiments/calibration/fs4_durable_runtime.py",
    "experiments/calibration/fs4_checkpoint.py",
    "experiments/calibration/fs4_threshold_integration.py",
    "experiments/calibration/fs4_threshold_core.py",
    "experiments/calibration/fs4_threshold_controllers.py",
    "experiments/calibration/fs4_stochastic_bea.py",
    "experiments/calibration/fs4_source_excluded_scorer.py",
    "experiments/calibration/fs4_greedy_provenance.py",
    "experiments/calibration/fs4_r2_context.py",
    "experiments/calibration/fs4_joint_campaign_contract.py",
    "experiments/calibration/fs4_joint_campaign.py",
    "experiments/calibration/fs4_joint_campaign_evaluator.py",
    "experiments/calibration/fs4_joint_campaign_resources.py",
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
        if execution.get("schema") in ("FS4_R4_BLINDED_EXECUTION_V1", "FS4_JOINT_CAMPAIGN_EXECUTION_V1"):
            required = {"run_id", "source_commit", "source_fingerprints", "expected_events_by_category", "profile",
                        "stream_sha256_by_category", "scientific_contract_sha256", "dataset_identity_sha256",
                        "model_state_sha256", "support_groups_by_category", "fixed_support_sha256_by_category",
                        "query_seed", "execution_scope", "protocol", "research_execution_manifest"}
            if supplied.get("profile") == "COUPLED_QUERY_ONLY_FROZEN_WRITER": required.add("controller")
            if execution.get("schema") == "FS4_JOINT_CAMPAIGN_EXECUTION_V1":
                required.update({"campaign_id", "trajectory_id", "scientific_spec_sha256",
                                 "campaign_registration_sha256", "r7_seed0_order_sha256",
                                 "support_preflight_receipt_sha256"})
            if set(supplied) != required:
                raise LedgerIntegrityError("blinded ledger contains unregistered fields")
            from .fs4_r4_blinded import validate_blinded_execution
            self.execution_manifest = validate_blinded_execution(execution, supplied)
        else:
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
            if execution.get("schema") in ("FS4_R4_BLINDED_EXECUTION_V1", "FS4_JOINT_CAMPAIGN_EXECUTION_V1"):
                self._private.write("INCOMPLETE.json", canonical({"status": "INCOMPLETE", "stage": "A"}) + b"\n")
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
            if self.execution_manifest.get("schema") in ("FS4_R4_BLINDED_EXECUTION_V1",
                                                            "FS4_JOINT_CAMPAIGN_EXECUTION_V1"):
                expected.update({"resource_sha256": row["resource_sha256"], "pixel_sha256": row["pixel_sha256"],
                                 "view": row["view"]})
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
