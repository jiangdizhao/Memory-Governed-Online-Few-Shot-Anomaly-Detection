"""Separate-principal private Stage B projection storage; no evaluator here."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .fs4_durable_ledger import digest
from .fs4_r2_security import WindowsPrivateStore, _sid_string, current_user_sid
from .fs4_r2_trust import verify_production_signed


class StageBIsolationError(RuntimeError):
    pass


class StageBProjectionStore:
    """The authorized Stage B account owns a separate private projection store."""
    def __init__(self, *, grant: Mapping[str, Any], manifest: Mapping[str, Any],
                 global_a_sha256: str, create: bool) -> None:
        payload = verify_production_signed(grant, role="stage_b_grant")
        expected = {"run_id": manifest["run_id"], "protocol": manifest["protocol"],
            "source_commit": manifest["source_commit"], "manifest_sha256": digest(manifest),
            "dataset_identity_sha256": manifest["dataset_identity_sha256"],
            "stream_sha256_by_category": manifest["stream_sha256_by_category"],
            "stage_a_sha256": global_a_sha256, "scope": manifest["stage_b_scope"],
            "stage_a_principal_sid": manifest["stage_a_principal_sid"]}
        if any(payload.get(key) != value for key, value in expected.items()):
            raise StageBIsolationError("projection grant is not bound to the authorized sealed run")
        principal = _sid_string(current_user_sid())
        if (payload.get("stage_b_principal_sid") != principal or principal == manifest["stage_a_principal_sid"] or
                not isinstance(payload.get("projection_root"), str)):
            raise StageBIsolationError("Stage B must execute as its distinct authorized principal")
        self._store = WindowsPrivateStore(Path(payload["projection_root"]), create=create)

    def write_projection(self, name: str, value: bytes) -> Path:
        return self._store.write(name, value)

    def read_projection(self, name: str) -> bytes:
        return self._store.read(name)
