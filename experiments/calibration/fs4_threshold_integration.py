"""Future FS4 event interfaces. Importing this module never starts an experiment."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import wraps
import hashlib
import json
import math
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .fs4_greedy_provenance import GreedyProvenance, _native_modules
from .fs4_source_excluded_scorer import NativeSourceExcludedScorer, _correction_digest, _feature_digest, initialize_native_support
from .fs4_stochastic_bea import SealedPrediction, StochasticBEAGroupSlot
from .fs4_threshold_core import NormalEvidence, _identity
from .fs4_threshold_controllers import SelectionAwareHistoricalThreshold, SelectionAwareSourceExcludedThreshold
from .fs4_errors import FS4FatalIntegrityError

FEATURE_CONVENTION = "native_encode_pil_concat_l2_normalized_before_scorer_normalization"


def _condition_guard(method: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(method)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(self, "failed", False):
            raise FS4FatalIntegrityError("condition was invalidated by an earlier event failure")
        try:
            return method(self, *args, **kwargs)
        except Exception:
            self.failed = True
            raise
    return guarded


def prediction_digest(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _validate_bound_inputs(provenance: Mapping[str, Any] | None, image: Any,
                           features: torch.Tensor, cls_feature: Any) -> None:
    if provenance is None:
        return
    cls = np.asarray(cls_feature, dtype=np.float32).reshape(-1)
    cls_sha = hashlib.sha256(cls.tobytes()).hexdigest()
    if (provenance.get("feature_sha256") != _feature_digest(features) or
            provenance.get("cls_sha256") != cls_sha or
            provenance.get("geometry") != [int(image.width), int(image.height)]):
        raise FS4FatalIntegrityError("trusted feature, CLS or geometry identity changed")


@dataclass(frozen=True)
class FrozenWriterReference:
    theta: float
    sigma: float


class CoverageMemory:
    """E6 support/dynamic CLS novelty geometry, initialized only from four Supports."""
    def __init__(self, support_cls: Any) -> None:
        x = np.asarray(support_cls, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] != 4 or x.shape[1] == 0 or not np.isfinite(x).all():
            raise ValueError("four finite Support CLS features required")
        self.support = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        sims = self.support @ self.support.T
        np.fill_diagonal(sims, -1.0)
        self.support_distances = np.maximum(0.0, 1.0 - sims.max(axis=1))
        self.dynamic = np.empty((0, x.shape[1]), dtype=np.float32)
        self.distances = np.empty((0,), dtype=np.float32)

    def _query(self, feature: Any) -> tuple[np.ndarray, float]:
        x = np.asarray(feature, dtype=np.float32).reshape(-1)
        if x.shape != (self.support.shape[1],) or not np.isfinite(x).all():
            raise ValueError("invalid CLS feature")
        x = x / max(float(np.linalg.norm(x)), 1e-12)
        bank = np.concatenate((self.support, self.dynamic), axis=0)
        return x, max(0.0, 1.0 - float(np.max(bank @ x)))

    def novelty(self, feature: Any) -> tuple[float, float]:
        _, distance = self._query(feature)
        refs = np.concatenate((self.support_distances, self.distances))
        return float(np.count_nonzero(refs <= distance) / len(refs)), distance

    def observe_normal(self, feature: Any) -> None:
        x, distance = self._query(feature)
        self.dynamic = np.concatenate((self.dynamic, x[None, :]), axis=0)[-1024:]
        self.distances = np.append(self.distances, np.float32(distance))[-1024:]


def build_native_from_four_support(*, impl: Any, cfg: Any, support_paths: Sequence[str],
        support_groups: Sequence[str], augmentation_cache: Mapping[str, Any],
        identity_images_and_features: Sequence[tuple[Any, torch.Tensor]], support_cls: Any,
        feature_convention: str) -> tuple[Any, dict[str, Any], CoverageMemory]:
    """Build using P0's unchanged native support builder; caller supplies only four Supports."""
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration.profiles import NativeScorerConfig
    from experiments.calibration.native import validate_p0_config
    if (len(support_paths) != 4 or len(set(support_paths)) != 4 or len(support_groups) != 4 or
            len(set(support_groups)) != 4 or set(augmentation_cache) != set(support_paths) or
            len(identity_images_and_features) != 4 or feature_convention != FEATURE_CONVENTION):
        raise ValueError("native builder accepts exactly four registered Support identities")
    for group in support_groups:
        _identity(group)
    validate_p0_config(cfg, NativeScorerConfig())
    if any(len(augmentation_cache[path]) != cfg.augmentations for path in support_paths):
        raise ValueError("Support augmentation count differs from native profile")
    if (any(not isinstance(features, torch.Tensor) or tuple(features.shape) != (784, 4096) or features.dtype != torch.float32
            for _, features in identity_images_and_features) or
            any(len(view) != 1 or not isinstance(view[0], torch.Tensor) or tuple(view[0].shape) != (784, 4096) or view[0].dtype != torch.float16
                for path in support_paths for view in augmentation_cache[path])):
        raise ValueError("native Support identity and augmentation features must be fused (784,4096)")
    impl.set_seed(0)
    bank = p0.build_vmb(impl, augmentation_cache, list(support_paths), cfg)
    r3._validate_correction_config(bank)
    if int(bank.corr_cfg.K_correction) != 256:
        raise ValueError("unexpected historical C1 capacity before K=1024 allocation")
    bank.corr_cfg.K_correction = 1024
    for views in bank.corr_cb.values():
        for view, books in views.items():
            views[view] = [type(old)(1024, int(old.protos.shape[1]), old.device, bank.corr_cfg) for old in books]
    bridge, _ = _native_modules()
    from v02_memory import MemoryArm
    writer = MemoryArm("greedy_coreset", bank, SimpleNamespace(flat=1024, stm=1024,
        ltm=0, buffer=0, min_events=2, promote_every=16, cap_bytes=2 * 1024**3), seed=0)
    correction_cb = bank.corr_cb["p0"]["id"][0]
    if len(bridge.dynamic_codebooks(bank)) != 1 or correction_cb.size() != 0:
        raise ValueError("locked fused native profile requires one empty Greedy and C1 layer")
    support = [(group, image, features) for group, (image, features) in zip(support_groups, identity_images_and_features)]
    reference = initialize_native_support(impl=impl, bank=bank, cfg=cfg,
        support=support, correction_cb=correction_cb)
    runtime = NativeRuntime(impl=impl, bank=bank, cfg=cfg, writer=writer,
        correction_cfg=bank.corr_cfg, correction_cb=correction_cb, reference=reference)
    return runtime, reference, CoverageMemory(support_cls)


class NativeRuntime:
    """Adapter to unchanged R1 native scoring, C1 updates and V0.2 Greedy writer."""
    def __init__(self, *, impl: Any, bank: Any, cfg: Any, writer: Any,
                 correction_cfg: Any, correction_cb: Any, reference: Mapping[str, Any]) -> None:
        from experiments.calibration import r3_fullstream_2x2 as r3
        checked_cfg = copy.deepcopy(correction_cfg)
        checked_cfg.K_correction = 256
        r3._validate_correction_config(SimpleNamespace(corr_cfg=checked_cfg))
        if (writer.bank is not bank or getattr(writer, "name", None) != "greedy_coreset" or
                correction_cb is not bank.corr_cb["p0"]["id"][0] or int(correction_cb.K) != 1024 or
                int(correction_cfg.K_correction) != 1024 or bank.use_defect_bank or
                float(reference["writer_reference"]["sigma"]) < 0.02):
            raise ValueError("native K=1024 Greedy/C1 or fixed writer reference is invalid")
        self.impl, self.bank, self.cfg, self.writer = impl, bank, cfg, writer
        self.correction_cfg, self.correction_cb = correction_cfg, correction_cb
        self.fixed_sha = str(reference["fixed_support_sha256"])
        theta = float(reference["writer_reference"]["theta"])
        sigma = float(reference["writer_reference"]["sigma"])
        if not math.isfinite(theta) or not math.isfinite(sigma):
            raise ValueError("frozen Support writer reference must be finite")
        self._writer_reference = FrozenWriterReference(theta, sigma)
        self.sources: dict[int, set[str]] = {}
        self.provenance = GreedyProvenance(bank)
        self.normal_version = self.correction_version = 0
        self.rescorer = NativeSourceExcludedScorer(impl=impl, bank=bank, cfg=cfg,
            correction_cfg=correction_cfg, correction_cb=correction_cb,
            correction_sources=self.sources, provenance=self.provenance,
            normalization=FEATURE_CONVENTION)
        self.states()

    @property
    def theta_ref(self) -> float:
        return self._writer_reference.theta

    @property
    def sigma_ref(self) -> float:
        return self._writer_reference.sigma

    def states(self) -> dict[str, str]:
        from experiments.calibration import r1_matched_correction_pilot as r1
        from experiments.calibration.native import _fixed_bank_digest
        bridge, _ = _native_modules()
        self.provenance.verify(self.bank)
        if _fixed_bank_digest(self.bank) != self.fixed_sha:
            raise ValueError("four-Support fixed memory changed")
        greedy = bridge.snapshot(self.bank, version=0).sha256()
        correction = _correction_digest(r1, self.correction_cb, self.sources)
        def digest(obj: Any) -> str:
            return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
        g = digest([self.fixed_sha, greedy, self.provenance.digest()])
        return {"G": g, "GC": digest([g, correction])}

    def score(self, *, group: str, position: int, image: Any, features: torch.Tensor) -> dict[str, Any]:
        from experiments.calibration import r1_matched_correction_pilot as r1, r2_exp2a_c1_strength as exp2a
        prior = self.states()
        layers = [features.to(self.bank.device)]
        base, ids, g, summary = r1._native_dense_score(self.impl, self.bank, image, layers, self.cfg)
        selected = self.impl.l2_normalize(layers[0].to(self.bank.device))[
            torch.as_tensor(ids, device=self.bank.device, dtype=torch.long)]
        delta, _ = exp2a._score_arm(self.impl, self.bank, "FS4-C1", self.correction_cfg,
            self.correction_cb, self.sources, selected, group, position)
        gc = r1.max_score(r1.corrected_dense_map(base, ids, delta))
        if self.states() != prior:
            raise ValueError("native pre-feedback score changed the live scorer")
        layer = summary["layer0"]
        return {"G": float(g), "GC": float(gc), "top_ids": ids,
                "correction_top32": delta, "correction_size": self.correction_cb.size(),
                "dmax": float(layer["topk_dN_max"]), "dmean": float(layer["topk_dN_mean"]),
                "features": layers, "states": prior}

    def feedback(self, *, receipt: Any, evidence: NormalEvidence | None, policy: StochasticBEAGroupSlot,
                 scored: Mapping[str, Any]) -> dict[str, Any]:
        from experiments.calibration import r1_matched_correction_pilot as r1, r2_exp2a_c1_strength as exp2a
        if (policy.receipts.get(receipt.prediction.position) != receipt or not receipt.executed or
                not receipt.disclosed or not receipt.query_seal or receipt.label not in (0, 1)):
            raise ValueError("native feedback requires an executed and sealed query")
        if receipt.label == 0 and (evidence is None or evidence.receipt != receipt):
            raise ValueError("verified normal requires matching query evidence")
        p = receipt.prediction
        if (scored["states"] != self.states() or
                scored["states"] != {"G": p.scorer_g, "GC": p.scorer_gc} or
                float(scored["G"]) != p.score_g or float(scored["GC"]) != p.score_gc):
            raise ValueError("native scorer changed after the sealed prediction")
        label, pos, group = receipt.label, receipt.prediction.position, receipt.prediction.group
        before = self.states()
        correction_before = _correction_digest(r1, self.correction_cb, self.sources)
        pred = int(scored["G"] >= self.theta_ref)
        outcome = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[(label, pred)]
        prior_cfg, prior_cb = exp2a._switch_codebook(self.bank, "FS4-C1", self.correction_cfg, self.correction_cb)
        try:
            self.bank.update_corrections_from_topk(cls_name="p0", feats_per_layer=list(scored["features"]),
                mode=0, view="id", layer_to_patch_idx={0: scored["top_ids"].tolist()},
                sign=1 if label else -1, score_img=float(scored["G"]), theta=self.theta_ref,
                strength_scale=1.0, outcome=outcome, step=pos)
            n = int(self.correction_cb.size())
            r1._record_c1_source_group_updates(self.sources,
                last_steps=self.correction_cb.last_step[:n].detach().cpu().tolist(),
                event_position=pos, event_group=group)
        finally:
            exp2a._restore_codebook(self.bank, self.correction_cfg, prior_cfg, prior_cb)
        writer_result = None
        if label == 0:
            writer_result = self.provenance.write_normal(writer=self.writer, impl=self.impl,
                evidence=evidence, policy=policy, features=scored["features"], candidates={0: scored["top_ids"].tolist()},
                score=float(scored["G"]), theta_ref=self.theta_ref, sigma_ref=self.sigma_ref)
        after = self.states()
        if before["G"] != after["G"]:
            self.normal_version += 1
        if correction_before != _correction_digest(r1, self.correction_cb, self.sources):
            self.correction_version += 1
        return {"before": before, "after": after, "greedy": writer_result, "outcome": outcome,
                "learning_reference_threshold": self.theta_ref, "learning_prediction": pred}


class SharedTrajectory:
    """Threshold-only comparison: one scorer, query reference, writer and feedback."""
    def __init__(self, *, runtime: Any, coverage: CoverageMemory, theta0: float,
                 policy: StochasticBEAGroupSlot, seal_prediction: Callable[[dict[str, Any]], str],
                 seal_query: Callable[[Any], str], label_provider: Callable[[str, int], int]) -> None:
        self.runtime, self.coverage, self.policy = runtime, coverage, policy
        self.L = SelectionAwareHistoricalThreshold(theta0)
        self.E = SelectionAwareSourceExcludedThreshold(theta0, runtime.rescorer)
        self._query_reference = float(theta0)
        self._learning_reference = float(getattr(runtime, "theta_ref", theta0))
        if self._learning_reference != float(theta0):
            raise ValueError("shared learning reference must equal frozen Support initialization")
        self.seal_prediction, self.seal_query, self.label_provider = seal_prediction, seal_query, label_provider
        self.mode = "SHARED_TRAJECTORY_THRESHOLD_ONLY"

    @property
    def query_reference(self) -> float:
        return self._query_reference

    @_condition_guard
    def step(self, *, position: int, group: str, event: str, image: Any,
             features: torch.Tensor, cls_feature: Any, feature_convention: str,
             event_provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if (feature_convention != FEATURE_CONVENTION or not isinstance(features, torch.Tensor) or
                tuple(features.shape) != (784, 4096) or features.dtype != torch.float32 or not torch.isfinite(features).all()):
            raise ValueError("current event must supply the complete registered fused feature tensor")
        _validate_bound_inputs(event_provenance, image, features, cls_feature)
        if float(getattr(self.runtime, "theta_ref", self._learning_reference)) != self._learning_reference:
            raise FS4FatalIntegrityError("frozen writer/C1 learning reference changed")
        scale = self.policy.prior_scale()  # obtained before the current score
        states = self.runtime.states()
        normal_version, correction_version = self.runtime.normal_version, self.runtime.correction_version
        scored = self.runtime.score(group=group, position=position, image=image, features=features)
        if scored["states"] != states:
            raise ValueError("scorer state changed before event seal")
        decisions = {name: {j: int(scored[j] >= controller.engine.theta[j]) for j in ("G", "GC")}
                     for name, controller in (("L", self.L), ("E", self.E))}
        novelty, _distance = self.coverage.novelty(cls_feature)
        query_inputs = {"novelty_percentile": novelty,
            "correction_top32": np.asarray(scored["correction_top32"], dtype=np.float64).tolist(),
            "correction_bank_size": int(scored["correction_size"]), "topk_dN_max": float(scored["dmax"]),
            "topk_dN_mean": float(scored["dmean"]), "pre_score_scale": scale}
        feature_sha = _feature_digest(features)
        top_ids = np.asarray(scored["top_ids"], dtype=np.int64).tolist()
        threshold_state = hashlib.sha256(json.dumps({"L": self.L.engine.theta, "E": self.E.engine.theta,
            "query_reference": self.query_reference}, sort_keys=True).encode()).hexdigest()
        learning_prediction = int(scored["G"] >= self._learning_reference)
        payload = {"position": position, "group": group, "event": event, "profile": self.mode,
                   "scores": {j: scored[j] for j in ("G", "GC")}, "decisions": decisions,
                   "states": states, "threshold_state": threshold_state,
                   "thresholds": {"L": dict(self.L.engine.theta), "E": dict(self.E.engine.theta)},
                   "query_reference_threshold": self.query_reference,
                   "query_inputs": query_inputs,
                   "learning_reference_threshold": self._learning_reference,
                   "learning_prediction": learning_prediction,
                   "decision_learning_disagreement": {name: {j: decisions[name][j] != learning_prediction for j in ("G", "GC")} for name in decisions},
                   "feature_sha256": feature_sha, "top32_ids": top_ids}
        if event_provenance is not None:
            payload["trusted_event_provenance"] = dict(event_provenance)
        expected_seal = prediction_digest(payload)
        seal = self.seal_prediction(payload)
        if seal != expected_seal or prediction_digest(payload) != expected_seal or _feature_digest(features) != feature_sha:
            raise ValueError("prediction ledger did not seal the exact pre-feedback payload")
        pred = SealedPrediction(position, group, event, scored["G"], scored["GC"],
                                states["G"], states["GC"], threshold_state, seal)
        receipt = self.policy.decide(pred, reference_threshold=self.query_reference, **query_inputs)
        receipt = self.policy.seal_query(receipt, self.seal_query(receipt))
        update = None
        if receipt.executed:
            _validate_bound_inputs(event_provenance, image, features, cls_feature)
            if _feature_digest(scored["features"][0]) != feature_sha:
                raise FS4FatalIntegrityError("sealed native scoring features changed before disclosure")
            receipt = self.policy.disclose(receipt, self.label_provider(event, position))
            _validate_bound_inputs(event_provenance, image, features, cls_feature)
            if _feature_digest(scored["features"][0]) != feature_sha:
                raise FS4FatalIntegrityError("sealed native scoring features changed after disclosure")
            if _feature_digest(features) != feature_sha or np.asarray(scored["top_ids"], dtype=np.int64).tolist() != top_ids:
                raise ValueError("sealed feature or Top-32 identity changed before feedback")
            evidence = (NormalEvidence.from_disclosure(self.policy, receipt,
                normal_version=normal_version, correction_version=correction_version,
                feature_handle=group) if receipt.label == 0 else None)
            if evidence is not None:
                self.E.add(evidence, features, height=int(image.height), width=int(image.width),
                           normalization=feature_convention)
                self.L.add(evidence)
            update = self.runtime.feedback(receipt=receipt, evidence=evidence, policy=self.policy, scored=scored)
            update.update({"learning_reference_threshold": self._learning_reference,
                           "learning_prediction": learning_prediction, "controller_decisions": decisions,
                           "controller_thresholds": payload["thresholds"],
                           "decision_learning_disagreement": payload["decision_learning_disagreement"]})
            if evidence is not None:
                self.coverage.observe_normal(cls_feature)
            current = self.runtime.states()
            self.E.refresh(state_g=current["G"], state_gc=current["GC"], event_position=position + 1)
        return {"sealed_prediction": payload, "query": receipt, "feedback_update": update,
                "next_thresholds": {"L": dict(self.L.engine.theta), "E": dict(self.E.engine.theta)},
                "mode": self.mode}


class CoupledTrajectory:
    """One independent system; its active controller can affect its own query policy."""
    def __init__(self, *, controller: str, runtime: Any, coverage: CoverageMemory,
                 theta0: float, policy: StochasticBEAGroupSlot,
                 seal_prediction: Callable[[dict[str, Any]], str],
                 seal_query: Callable[[Any], str], label_provider: Callable[[str, int], int]) -> None:
        if controller not in ("L", "E"):
            raise ValueError("coupled controller must be L or E")
        self.kind, self.runtime, self.coverage, self.policy = controller, runtime, coverage, policy
        self.controller = (SelectionAwareHistoricalThreshold(theta0) if controller == "L" else
                           SelectionAwareSourceExcludedThreshold(theta0, runtime.rescorer))
        self.seal_prediction, self.seal_query, self.label_provider = seal_prediction, seal_query, label_provider
        self.mode = "COUPLED_QUERY_ONLY_FROZEN_WRITER"
        self._learning_reference = float(getattr(runtime, "theta_ref", theta0))
        if self._learning_reference != float(theta0):
            raise ValueError("coupled learning reference must equal frozen Support initialization")

    @_condition_guard
    def step(self, *, position: int, group: str, event: str, image: Any,
             features: torch.Tensor, cls_feature: Any, feature_convention: str,
             event_provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if (feature_convention != FEATURE_CONVENTION or not isinstance(features, torch.Tensor) or
                tuple(features.shape) != (784, 4096) or features.dtype != torch.float32 or not torch.isfinite(features).all()):
            raise ValueError("current event must supply the complete registered fused feature tensor")
        _validate_bound_inputs(event_provenance, image, features, cls_feature)
        if float(getattr(self.runtime, "theta_ref", self._learning_reference)) != self._learning_reference:
            raise FS4FatalIntegrityError("frozen writer/C1 learning reference changed")
        scale = self.policy.prior_scale()
        states = self.runtime.states()
        normal_version, correction_version = self.runtime.normal_version, self.runtime.correction_version
        scored = self.runtime.score(group=group, position=position, image=image, features=features)
        if scored["states"] != states:
            raise ValueError("scorer state changed before event seal")
        theta = dict(self.controller.engine.theta)
        decisions = {j: int(scored[j] >= theta[j]) for j in ("G", "GC")}
        novelty, _distance = self.coverage.novelty(cls_feature)
        query_inputs = {"novelty_percentile": novelty,
            "correction_top32": np.asarray(scored["correction_top32"], dtype=np.float64).tolist(),
            "correction_bank_size": int(scored["correction_size"]), "topk_dN_max": float(scored["dmax"]),
            "topk_dN_mean": float(scored["dmean"]), "pre_score_scale": scale}
        feature_sha = _feature_digest(features)
        top_ids = np.asarray(scored["top_ids"], dtype=np.int64).tolist()
        threshold_state = hashlib.sha256(json.dumps(theta, sort_keys=True).encode()).hexdigest()
        learning_prediction = int(scored["G"] >= self._learning_reference)
        payload = {"position": position, "group": group, "event": event, "profile": self.mode,
                   "scores": {j: scored[j] for j in ("G", "GC")}, "decisions": decisions,
                   "states": states, "threshold_state": threshold_state,
                   "thresholds": theta, "query_reference_threshold": theta["GC"],
                   "query_inputs": query_inputs,
                   "learning_reference_threshold": self._learning_reference,
                   "learning_prediction": learning_prediction,
                   "decision_learning_disagreement": {j: decisions[j] != learning_prediction for j in ("G", "GC")},
                   "feature_sha256": feature_sha, "top32_ids": top_ids}
        if event_provenance is not None:
            payload["trusted_event_provenance"] = dict(event_provenance)
        expected_seal = prediction_digest(payload)
        seal = self.seal_prediction(payload)
        if seal != expected_seal or prediction_digest(payload) != expected_seal or _feature_digest(features) != feature_sha:
            raise ValueError("prediction ledger did not seal the exact pre-feedback payload")
        pred = SealedPrediction(position, group, event, scored["G"], scored["GC"],
                                states["G"], states["GC"], threshold_state, seal)
        receipt = self.policy.decide(pred, reference_threshold=theta["GC"], **query_inputs)
        receipt = self.policy.seal_query(receipt, self.seal_query(receipt))
        update = None
        if receipt.executed:
            _validate_bound_inputs(event_provenance, image, features, cls_feature)
            if _feature_digest(scored["features"][0]) != feature_sha:
                raise FS4FatalIntegrityError("sealed native scoring features changed before disclosure")
            receipt = self.policy.disclose(receipt, self.label_provider(event, position))
            _validate_bound_inputs(event_provenance, image, features, cls_feature)
            if _feature_digest(scored["features"][0]) != feature_sha:
                raise FS4FatalIntegrityError("sealed native scoring features changed after disclosure")
            if _feature_digest(features) != feature_sha or np.asarray(scored["top_ids"], dtype=np.int64).tolist() != top_ids:
                raise ValueError("sealed feature or Top-32 identity changed before feedback")
            evidence = (NormalEvidence.from_disclosure(self.policy, receipt,
                normal_version=normal_version, correction_version=correction_version,
                feature_handle=group if self.kind == "E" else None) if receipt.label == 0 else None)
            if evidence is not None:
                if self.kind == "E":
                    self.controller.add(evidence, features, height=int(image.height), width=int(image.width),
                                        normalization=feature_convention)
                else:
                    self.controller.add(evidence)
            update = self.runtime.feedback(receipt=receipt, evidence=evidence, policy=self.policy, scored=scored)
            update.update({"learning_reference_threshold": self._learning_reference,
                           "learning_prediction": learning_prediction, "controller_decisions": decisions,
                           "controller_thresholds": theta,
                           "decision_learning_disagreement": payload["decision_learning_disagreement"]})
            if evidence is not None:
                self.coverage.observe_normal(cls_feature)
            if self.kind == "E":
                current = self.runtime.states()
                self.controller.refresh(state_g=current["G"], state_gc=current["GC"], event_position=position + 1)
        return {"sealed_prediction": payload, "query": receipt, "feedback_update": update,
                "next_thresholds": dict(self.controller.engine.theta), "mode": self.mode}


def require_independent_coupled_instances(left: CoupledTrajectory, right: CoupledTrajectory) -> None:
    if (left is right or left.runtime is right.runtime or left.policy is right.policy or
            left.coverage is right.coverage or left.controller is right.controller or
            any(getattr(left.runtime, name, None) is not None and
                getattr(left.runtime, name, None) is getattr(right.runtime, name, None)
                for name in ("bank", "writer", "correction_cb", "sources", "provenance"))):
        raise ValueError("coupled systems must own independent memories, policies and controllers")
    if getattr(left.runtime, "bank", None) is not None and getattr(right.runtime, "bank", None) is not None:
        bridge, _ = _native_modules()
        def mutable_storages(runtime: Any) -> set[tuple[str, int]]:
            tensors = [cb.protos for cb in bridge.dynamic_codebooks(runtime.bank)]
            tensors.extend(getattr(runtime.correction_cb, name) for name in
                ("protos", "beta", "radius", "kappa", "count", "outcome_fp", "outcome_fn", "outcome_tn", "outcome_tp", "last_step"))
            return {(str(t.device), t.untyped_storage().data_ptr()) for t in tensors if t.untyped_storage().nbytes() > 0}
        if mutable_storages(left.runtime) & mutable_storages(right.runtime):
            raise ValueError("coupled systems share mutable native tensor storage")
