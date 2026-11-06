"""CPU-only five-view synthetic integration; never loads a dataset or model."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .fs4_stochastic_bea import StochasticBEAGroupSlot, receipt_digest
from .fs4_threshold_core import support_self_reference
from .fs4_threshold_integration import CoverageMemory, FEATURE_CONVENTION, NativeRuntime, SharedTrajectory, prediction_digest


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _offline_check_after_all_seals(sealed: list[dict[str, Any]], labels: dict[str, int],
                                   query_seal_count: int) -> int:
    if len(sealed) != query_seal_count or len(sealed) != len(labels):
        raise AssertionError("offline synthetic outcome check requires every event and query seal")
    for row in sealed:
        label = labels[row["event"]]
        if label not in (0, 1) or any(v not in (0, 1) for decisions in row["decisions"].values() for v in decisions.values()):
            raise AssertionError("invalid synthetic outcome or sealed prediction")
    return len(sealed)


class _SyntheticRescorer:
    def __init__(self, runtime: "_SyntheticRuntime") -> None:
        self.runtime = runtime
        self.calls = 0

    def score(self, record: Any, *, event_position: int) -> dict[str, float]:
        if record.evidence.group not in self.runtime.normal_groups or event_position <= record.evidence.position:
            raise ValueError("synthetic source exclusion requires a past verified normal")
        self.calls += 1
        # This test double is solely an orchestration oracle, not a native score.
        return {"G": self.runtime.theta0 + 0.20, "GC": self.runtime.theta0 + 0.25}


class _SyntheticRuntime:
    def __init__(self, theta0: float) -> None:
        self.theta0 = theta0
        self.normal_version = self.correction_version = 0
        self.normal_groups: list[str] = []
        self.correction_groups: list[tuple[str, int]] = []
        self.rescorer = _SyntheticRescorer(self)

    def states(self) -> dict[str, str]:
        g = _hash('G|' + '|'.join(self.normal_groups))
        return {"G": g, "GC": _hash(g + repr(self.correction_groups))}

    def score(self, *, group: str, position: int, image: Any, features: torch.Tensor) -> dict[str, Any]:
        return {"G": self.theta0, "GC": self.theta0, "top_ids": np.arange(32),
                "correction_top32": np.zeros(32), "correction_size": len(self.correction_groups),
                "dmax": 1.0, "dmean": 0.5, "features": [features], "states": self.states()}

    def feedback(self, *, receipt: Any, evidence: Any, policy: Any, scored: Any) -> dict[str, Any]:
        if policy.receipts.get(receipt.prediction.position) != receipt or not receipt.disclosed:
            raise ValueError("synthetic feedback was not legally disclosed")
        before = self.states()
        self.correction_groups.append((receipt.prediction.group, receipt.label))
        self.correction_version += 1
        if receipt.label == 0:
            if evidence is None:
                raise ValueError("normal feedback missing evidence")
            self.normal_groups.append(evidence.group)
            self.normal_version += 1
        return {"before": before, "after": self.states(), "normal_write": receipt.label == 0}


def run_smoke() -> dict[str, Any]:
    torch.set_num_threads(1)
    reference = support_self_reference([0.40, 0.45, 0.50, 0.55])
    theta = float(reference["theta"])
    runtime = _SyntheticRuntime(theta)
    policy = StochasticBEAGroupSlot(seed=37)
    cls_vectors = np.eye(80, dtype=np.float32)
    coverage = CoverageMemory(cls_vectors[:4])
    labels: dict[str, int] = {}
    sealed: list[dict[str, Any]] = []
    query_seals: list[int] = []
    disclosed: list[int] = []

    def seal_prediction(payload: dict[str, Any]) -> str:
        sealed.append(payload)
        return prediction_digest(payload)

    def seal_query(receipt: Any) -> str:
        if receipt.prediction.position != len(sealed) - 1:
            raise AssertionError("query preceded prediction seal")
        query_seals.append(receipt.prediction.position)
        return receipt_digest(receipt)

    def label_provider(event: str, position: int) -> int:
        if position not in query_seals or not policy.receipts[position].executed:
            raise AssertionError("label requested before query seal")
        disclosed.append(position)
        return labels[event]

    loop = SharedTrajectory(runtime=runtime, coverage=coverage, theta0=theta, policy=policy,
        seal_prediction=seal_prediction, seal_query=seal_query, label_provider=label_provider)
    image = SimpleNamespace(height=28, width=28)
    features = torch.zeros((784, 4096), dtype=torch.float32)
    results = []
    for group_index in range(14):
        group = _hash(f"synthetic-group-{group_index}")
        for ordinal in range(5):
            position = len(results)
            event = _hash(f"synthetic-event-{group_index}-{ordinal}")
            labels[event] = 1 if group_index >= 12 else 0
            results.append(loop.step(position=position, group=group, event=event,
                image=image, features=features, cls_feature=cls_vectors[4 + group_index],
                feature_convention=FEATURE_CONVENTION))
    # Evaluator sees synthetic labels only after every prediction and query is sealed.
    assert len(sealed) == len(query_seals) == len(results) == 70
    policy.validate_complete_stream()
    checked = _offline_check_after_all_seals(sealed, labels, len(query_seals))
    assert len(disclosed) == policy.diagnostics()["queries"]
    assert len(runtime.normal_groups) >= 12 and any(label == 1 for _, label in runtime.correction_groups)
    assert runtime.rescorer.calls > 0
    assert len(loop.L.evidence) == len(loop.E.evidence) == len(runtime.normal_groups)
    assert loop.L.engine.theta["G"] != loop.E.engine.theta["G"]
    assert all(row["sealed_prediction"]["scores"]["G"] == theta for row in results)
    return {"status": "PASS_SYNTHETIC_ONLY", "events": len(results),
            "queries": len(disclosed), "verified_normals": len(runtime.normal_groups),
            "verified_defects": sum(label == 1 for _, label in runtime.correction_groups),
            "e_rescore_calls": runtime.rescorer.calls,
            "post_seal_synthetic_outcomes_checked": checked,
            "mode": loop.mode, "real_data_accessed": False}


def run_native_smoke() -> dict[str, Any]:
    """Synthetic four-Support/five-view stream with actual CPU native helpers."""
    from .fs4_greedy_provenance import _native_modules
    from .fs4_source_excluded_scorer import initialize_native_support
    _native_modules()
    import test_phase3_AIF_simplified_v22 as native
    from v02_memory import MemoryArm
    torch.set_num_threads(1)
    cfg = SimpleNamespace(topk_patches=32, image_aggregation="max", image_topk=64, image_quantile=0.95)
    correction_cfg = native.CorrectionConfig(enabled=False, K_correction=1024)
    bank = native.PatchVMB(device="cpu", K_normal_fixed=8, K_normal_dyn=1024,
        K_normal_ltm=1, K_defect=1, K_concept=8, knn_k_normal=5,
        corr_cfg=correction_cfg, use_defect_bank=False)
    seed = torch.zeros((1, 4096), dtype=torch.float16); seed[0, 0] = 1
    bank._ensure_mode_view_normals("p0", 0, "id", [seed])
    bank.normal_fixed["p0"][0]["id"][0].set_prototypes(seed)
    writer = MemoryArm("greedy_coreset", bank, SimpleNamespace(flat=1024, stm=1024,
        ltm=1, buffer=1, min_events=2, promote_every=16, cap_bytes=2 * 1024**3), seed=0)
    image = SimpleNamespace(height=28, width=28)
    support_features = torch.zeros((784, 4096), dtype=torch.float32)
    support_features[:, 0] = 0.5; support_features[:, 1] = 0.866
    support = [(_hash(f"native-support-{i}"), image, support_features) for i in range(4)]
    reference = initialize_native_support(impl=native, bank=bank, cfg=cfg,
        support=support, correction_cb=bank.corr_cb["p0"]["id"][0])
    runtime = NativeRuntime(impl=native, bank=bank, cfg=cfg, writer=writer,
        correction_cfg=bank.corr_cfg, correction_cb=bank.corr_cb["p0"]["id"][0], reference=reference)
    cls_vectors = np.eye(80, dtype=np.float32)
    policy = StochasticBEAGroupSlot(seed=37)
    labels: dict[str, int] = {}
    sealed: list[dict[str, Any]] = []
    query_sealed: set[int] = set()
    disclosure_positions: list[int] = []

    def seal_prediction(payload: dict[str, Any]) -> str:
        sealed.append(payload)
        return prediction_digest(payload)

    def seal_query(receipt: Any) -> str:
        assert receipt.prediction.position == len(sealed) - 1
        query_sealed.add(receipt.prediction.position)
        return receipt_digest(receipt)

    def provide(event: str, position: int) -> int:
        assert position in query_sealed and policy.receipts[position].executed
        disclosure_positions.append(position)
        return labels[event]

    loop = SharedTrajectory(runtime=runtime, coverage=CoverageMemory(cls_vectors[:4]),
        theta0=float(reference["theta"]), policy=policy, seal_prediction=seal_prediction,
        seal_query=seal_query, label_provider=provide)
    for group_index in range(14):
        group = _hash(f"native-synthetic-group-{group_index}")
        features = torch.zeros((784, 4096), dtype=torch.float32)
        features[:, 0] = 0.5; features[:, group_index + 2] = 0.866
        for ordinal in range(5):
            position = len(sealed)
            event = _hash(f"native-synthetic-event-{group_index}-{ordinal}")
            labels[event] = 1 if group_index >= 12 else 0
            loop.step(position=position, group=group, event=event, image=image,
                      features=features, cls_feature=cls_vectors[4 + group_index],
                      feature_convention=FEATURE_CONVENTION)
    assert len(sealed) == len(query_sealed) == 70
    policy.validate_complete_stream()
    checked = _offline_check_after_all_seals(sealed, labels, len(query_sealed))
    assert len(loop.L.evidence) == len(loop.E.evidence) >= 12
    assert any(r.disclosed and r.label == 1 for r in policy.receipts.values())
    assert runtime.normal_version > 0 and runtime.correction_version > 0
    assert sum(loop.E.rescore_calls.values()) > 0
    runtime.provenance.verify(bank)
    return {"status": "PASS_SYNTHETIC_NATIVE_CPU", "events": len(sealed),
            "queries": len(disclosure_positions), "verified_normals": len(loop.L.evidence),
            "verified_defects": sum(r.disclosed and r.label == 1 for r in policy.receipts.values()),
            "greedy_slots": sum(map(len, runtime.provenance.rows)),
            "correction_slots": int(runtime.correction_cb.size()),
            "e_rescore_calls": sum(loop.E.rescore_calls.values()),
            "post_seal_synthetic_outcomes_checked": checked,
            "real_data_accessed": False}


if __name__ == "__main__":
    print(json.dumps({"orchestration_double": run_smoke(), "native_cpu": run_native_smoke()},
                     indent=2, sort_keys=True))
