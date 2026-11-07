"""Non-adversarial, locally trusted FS4 encoder context.

This mode assumes the operator controls the worker and its Python environment.
The high-security ProductionContext remains disabled.
"""
from __future__ import annotations

from typing import Any

from .fs4_r2_context import FS4BoundaryError, ProductionContext


def require_label_blind_dataset_adapter() -> None:
    """Hold research execution until outcome-bearing event paths are isolated.

    The inherited native registration exposes OK/NG in event path names. A
    separately controlled image-only adapter must hide that mapping from the
    Stage A encoder/policy process before this hold can be removed.
    """
    raise FS4BoundaryError("RESEARCH_EXECUTION_BLOCKED: event paths expose undisclosed outcome information")


class ResearchContext(ProductionContext):
    """A context created only in the spawned local research worker."""

    def __init__(self, *, ledger: Any, category: str, controller: str | None = None) -> None:
        require_label_blind_dataset_adapter()
        from .fs4_research_ledger import ResearchLedger
        if type(ledger) is not ResearchLedger:
            raise FS4BoundaryError("research context requires its own research ledger")
        self._full_validation = True
        self._freeze_pending = True
        self._initialize(ledger=ledger, category=category, controller=controller,
                         research_payload=ledger.execution_manifest)
        self._full_validation = False

    def _verify_model(self) -> None:
        if self._freeze_pending:
            self._backbone.model.requires_grad_(False)
            self._backbone.model.eval()
            self._freeze_pending = False
        if self._full_validation:
            super()._verify_model()
            return
        model = self._backbone.model
        if (self._impl.encode_pil is not self._native_encode or
                any(module.training for module in model.modules()) or
                any(parameter.requires_grad for parameter in model.parameters())):
            raise FS4BoundaryError("research encoder is not frozen in evaluation mode")

    def validate_full(self) -> None:
        self._ledger.verify_frozen_sources()
        self._full_validation = True
        try:
            self._verify_model()
            self._verify_runtime()
            self._verify_cached_evidence()
        finally:
            self._full_validation = False

    def _encode_event(self, position: int) -> dict[str, Any]:
        event = super()._encode_event(position)
        event["event_provenance"].update({
            "source_commit": self._ledger.manifest["source_commit"],
            "fixed_support_sha256": self._ledger.manifest["fixed_support_sha256_by_category"][self._category],
        })
        return event
