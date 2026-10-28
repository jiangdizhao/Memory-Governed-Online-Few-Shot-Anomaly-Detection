"""Adapters to the unchanged v22 score and V0.2 memory implementations."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from typing import Any, Mapping, Sequence

from experiments.risk_safe import native_bridge
from experiments.risk_safe.shadow_memory import GreedyBank

from .profiles import NativeScorerConfig, canonical_json
from .protocol import ScoreResult


class NativeAdapterError(RuntimeError):
    pass


def validate_mutable_source_key(source_key: str,
                                protected_source_keys: Sequence[str]) -> None:
    if not source_key:
        raise NativeAdapterError("Verified-normal write requires an opaque source key")
    if source_key in set(protected_source_keys):
        raise NativeAdapterError("Protected C64 anchors are never eligible for memory writes")


def validate_p0_config(p0_config: Any, profile: NativeScorerConfig) -> None:
    """Fail closed unless the inherited P0 fields match the frozen profile."""
    fields = ("augmentation_seed", "bank_seed", "fixed_capacity",
              "fixed_per_image", "augmentations", "include_identity",
              "multiplier", "feature_fuse", "topk_patches",
              "image_aggregation", "image_topk", "image_quantile", "knn")
    mismatches = {}
    for name in fields:
        if not hasattr(p0_config, name):
            mismatches[name] = ("<missing>", getattr(profile, name))
        elif getattr(p0_config, name) != getattr(profile, name):
            mismatches[name] = (getattr(p0_config, name), getattr(profile, name))
    if mismatches:
        raise NativeAdapterError(f"Inherited p0.Config differs from locked profile: {mismatches}")


def _fixed_bank_digest(bank: Any) -> str:
    """Hash fixed-support prototype bytes without touching the bank."""
    h = hashlib.sha256()
    visited: set[int] = set()
    def walk(value: Any, path: str) -> None:
        if id(value) in visited:
            return
        if isinstance(value, dict):
            visited.add(id(value))
            for key in sorted(value, key=lambda x: str(x)):
                walk(value[key], f"{path}/{key}")
        elif isinstance(value, (list, tuple)):
            visited.add(id(value))
            for index, child in enumerate(value):
                walk(child, f"{path}/{index}")
        elif hasattr(value, "protos"):
            tensor = value.protos.detach().to("cpu").contiguous()
            h.update(path.encode("utf-8"))
            h.update(str((tuple(tensor.shape), str(tensor.dtype))).encode("ascii"))
            h.update(tensor.numpy().tobytes())
    fixed = getattr(bank, "normal_fixed", None)
    if fixed is None:
        raise NativeAdapterError("Native VMB lacks normal_fixed support bank")
    walk(fixed, "normal_fixed")
    return h.hexdigest()


def isolate_native_bank_k256(native_bank: Any) -> tuple[Any, str]:
    """Deep-copy P0's native bank and set only empty dynamic codebooks to K=256.

    P0 historically constructs K=1024. The static fixed support must remain
    byte-identical; nonempty or mismatched dynamic state is rejected.
    """
    before = _fixed_bank_digest(native_bank)
    clone = copy.deepcopy(native_bank)
    layers = native_bridge.dynamic_codebooks(clone)
    if not layers or any(cb.size() for cb in layers):
        raise NativeAdapterError("K=256 isolation requires an initially empty dynamic bank")
    for cb in layers:
        cb.K = 256
    if hasattr(clone, "K_normal_dyn"):
        clone.K_normal_dyn = 256
    after = _fixed_bank_digest(clone)
    if before != after or _fixed_bank_digest(native_bank) != before:
        raise NativeAdapterError("K=256 clone altered fixed or live static support")
    if any(int(cb.K) != 256 for cb in native_bridge.dynamic_codebooks(clone)):
        raise NativeAdapterError("Dynamic codebook capacity is not K=256")
    return clone, before


@dataclass
class NativeScoreService:
    """Score using P0's existing v22/VisionAD path on an isolated bank clone.

    ``bundle`` must be the historical scorer's ``(image, full_features)`` pair.
    This class intentionally supplies no fallback scorer.
    """

    impl: Any
    native_template: Any
    p0_config: Any
    config: NativeScorerConfig
    fixed_support_sha256: str

    def __post_init__(self) -> None:
        if not self.fixed_support_sha256:
            raise NativeAdapterError("Fixed support provenance hash is required")
        validate_p0_config(self.p0_config, self.config)

    def score(self, bundle: Any, memory: GreedyBank) -> ScoreResult:
        try:
            image, features = bundle
            clone = native_bridge.install_on_clone(self.native_template, memory)
            score, score_map = native_bridge.native_score(
                self.impl, clone, image, features, self.p0_config)
        except Exception as exc:
            raise NativeAdapterError(f"Native v22 score path failed: {exc}") from exc
        state = {
            "memory": memory.sha256(),
            "scorer": self.config.sha256,
            "fixed_support": self.fixed_support_sha256,
            "kind": "raw_native_score_bank",
        }
        digest = hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()
        return ScoreResult(score, memory.version, digest, score_map)


@dataclass(frozen=True)
class SharedWriteResult:
    eligible: bool
    changed: bool
    keep: int
    reason: str
    before_sha256: str
    after_sha256: str
    source_key: str
    appended: int
    replaced: int
    skipped: int


class SharedNativeGreedyWriter:
    """Single threshold-independent verified-normal writer.

    The reference threshold/sigma are fixed once, never taken from a
    calibration controller. Source rows are mirrored per layer in the same
    candidate order and replaced/append slots used by ``GreedyBank.proposed``.
    """

    def __init__(self, impl: Any, native_bank: Any, *, theta_ref: float,
                 protected_source_keys: Sequence[str], sigma_ref: float = 0.02):
        self.impl = impl
        self.native_bank = native_bank
        self.fixed_support_sha256 = _fixed_bank_digest(native_bank)
        self.theta_ref = float(theta_ref)
        self.sigma_ref = max(0.02, float(sigma_ref))
        self.protected_source_keys = frozenset(protected_source_keys)
        if len(self.protected_source_keys) != 64:
            raise NativeAdapterError("Writer must receive the 64 protected C64 source keys")
        self.bank = native_bridge.snapshot(native_bank, version=0)
        self.sources: list[list[str]] = [
            ["<unknown-preexisting>"] * len(layer) for layer in self.bank.layers]
        if any(len(layer) for layer in self.bank.layers):
            raise NativeAdapterError("Calibration memory must start empty")
        if any(cap != 256 for cap in self.bank.capacities):
            raise NativeAdapterError("Native dynamic K must be 256 per layer")

    def source_row_counts(self, source_key: str) -> tuple[int, ...]:
        return tuple(layer.count(source_key) for layer in self.sources)

    def state_dict(self) -> dict[str, Any]:
        """Private, resumable snapshot of the one live native Greedy bank."""
        return {
            "schema": "calibration.shared_native_greedy.v1",
            "fixed_support_sha256": self.fixed_support_sha256,
            "theta_ref": self.theta_ref,
            "sigma_ref": self.sigma_ref,
            "protected_source_keys": sorted(self.protected_source_keys),
            "capacities": list(self.bank.capacities),
            "version": self.bank.version,
            "layers": [layer.detach().to("cpu").contiguous().clone()
                       for layer in self.bank.layers],
            "sources": [list(rows) for rows in self.sources],
            "sha256": self.bank.sha256(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state.get("schema") != "calibration.shared_native_greedy.v1" or
                state.get("fixed_support_sha256") != self.fixed_support_sha256 or
                float(state.get("theta_ref")) != self.theta_ref or
                float(state.get("sigma_ref")) != self.sigma_ref or
                state.get("protected_source_keys") !=
                sorted(self.protected_source_keys) or
                state.get("capacities") != list(self.bank.capacities)):
            raise NativeAdapterError("Native writer checkpoint profile mismatch")
        import torch
        layers = [torch.as_tensor(x).detach().clone().contiguous()
                  for x in state.get("layers", ())]
        sources = [list(rows) for rows in state.get("sources", ())]
        if (len(layers) != len(self.bank.layers) or len(sources) != len(layers) or
                any(len(layer) != len(rows) for layer, rows in zip(layers, sources)) or
                any(len(layer) > cap for layer, cap in zip(layers, self.bank.capacities))):
            raise NativeAdapterError("Native writer checkpoint rows are misaligned")
        restored = GreedyBank(layers, self.bank.capacities,
                              version=int(state.get("version", -1)))
        if restored.version < 0 or restored.sha256() != state.get("sha256"):
            raise NativeAdapterError("Native writer checkpoint bank digest mismatch")
        live_before = native_bridge.snapshot(self.native_bank, version=self.bank.version)
        if restored.version == live_before.version:
            if restored.sha256() != live_before.sha256():
                raise NativeAdapterError("Same-version native checkpoint changes memory")
        else:
            native_bridge.install_on_live(self.native_bank, restored,
                                          expected_digest=live_before.sha256(),
                                          version=live_before.version)
        actual = native_bridge.snapshot(self.native_bank, version=restored.version)
        if (actual.sha256() != restored.sha256() or
                _fixed_bank_digest(self.native_bank) != self.fixed_support_sha256):
            raise NativeAdapterError("Native bank restoration failed parity verification")
        self.bank, self.sources = restored, sources

    def apply(self, *, queried: bool, label: int | None, source_key: str,
              score: float, static_amap: Mapping[str, Any],
              full_features: Sequence[Any], step: int) -> SharedWriteResult:
        before = self.bank
        before_digest = before.sha256()
        if not queried or label != 0:
            return SharedWriteResult(False, False, 0, "feedback_not_verified_normal",
                                     before_digest, before_digest, source_key, 0, 0, 0)
        validate_mutable_source_key(source_key, self.protected_source_keys)
        allow, keep, reason, _ = self.impl.select_informative_normal_write(
            score=float(score), theta=self.theta_ref,
            sigmaN=self.sigma_ref, pred_defect=bool(score >= self.theta_ref),
            novelty=0.0)
        if not allow or keep not in (8, 16, 24):
            return SharedWriteResult(False, False, int(keep), str(reason),
                                     before_digest, before_digest, source_key, 0, 0, 0)
        selected, selected_indices = native_bridge.extract_v22_candidates(
            self.impl, static_amap, full_features, keep=keep)
        normalized = native_bridge.normalized_native_candidate_rows(
            self.impl, self.native_bank, full_features, selected_indices, keep=keep)
        proposed, audit = before.proposed(normalized, already_normalized=True)

        # Mirror exact append/replacement slot decisions to preserve source
        # attribution for future leave-source-out evidence.
        from v02_memory import greedy_slot
        import torch
        next_sources: list[list[str]] = []
        for old_rows, candidate_rows, old_tensor, cap in zip(
                self.sources, normalized, before.layers, before.capacities):
            rows = list(old_rows)
            working = old_tensor.detach().clone().contiguous()
            for candidate in candidate_rows:
                value = candidate.reshape(1, -1).to(working.device, dtype=working.dtype)
                if len(working) < cap:
                    working = torch.cat((working, value), dim=0)
                    rows.append(source_key)
                else:
                    slot = greedy_slot(type("CB", (), {"protos": working,
                                                        "K": cap})(), value)
                    if slot is not None:
                        working[slot] = value.reshape(-1)
                        rows[slot] = source_key
            if not torch.equal(working, proposed.layers[len(next_sources)]):
                raise NativeAdapterError("Source attribution replay diverged from Greedy proposal")
            next_sources.append(rows)

        if proposed.version > before.version:
            native_bridge.install_on_live(
                self.native_bank, proposed, expected_digest=before_digest,
                version=before.version)
            actual = native_bridge.snapshot(self.native_bank, version=proposed.version)
            if actual.sha256() != proposed.sha256():
                raise NativeAdapterError("Live native bank differs from staged Greedy update")
            self.bank, self.sources = proposed, next_sources
        else:
            proposed = before
        return SharedWriteResult(True, proposed.version != before.version, keep,
                                 str(reason), before_digest, proposed.sha256(),
                                 source_key, audit.appended, audit.replaced,
                                 audit.skipped)
