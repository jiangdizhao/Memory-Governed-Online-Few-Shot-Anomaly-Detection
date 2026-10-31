"""CPU-only synthetic gates for the isolated R1 correction pilot."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.calibration.native import _fixed_bank_digest
from experiments.calibration.profiles import NativeScorerConfig
from experiments.calibration.r1_matched_correction_pilot import (
    CausalEventOrder,
    EvidenceTapeWriter,
    EXPECTED_BRANCH,
    R1BlockedInput,
    R1_PROFILE_ID,
    RAW_FIXED_BANK_DIGEST_ALGORITHM,
    STRUCTURED_FIXED_BANK_DIGEST_ALGORITHM,
    SUPPORT_ONLY_PROFILE_ID,
    R1CausalityError,
    TapeIntegrityError,
    LEGACY_FAILED_OUTPUT,
    _output_namespace,
    _check_fixed_bank_receipt,
    _configure_historical_runtime,
    _fixed_bank_digest_evidence,
    _validate_execution_identity,
    _validate_p0_profile,
    arrays_sha256,
    corrected_dense_map,
    deterministic_npz_bytes,
    max_score,
    sha256_bytes,
    signed_knn_delta,
    source_disjoint_anchor_indices,
)
from experiments.p0_protocol_audit import fixed_bank_profile


def _arrays() -> dict[str, np.ndarray]:
    base = np.linspace(0.0, 0.783, 784, dtype=np.float32)
    ids = np.arange(752, 784, dtype=np.int64)
    features = np.zeros((32, 8), dtype=np.float32)
    features[:, 0] = 1.0
    c1 = np.zeros(32, dtype=np.float32)
    c1[31] = -0.5
    c2 = np.zeros(32, dtype=np.float32)
    return {
        "base_patch_map": base,
        "top32_ids": ids,
        "top32_features": features,
        "c1_delta": c1,
        "c2_delta": c2,
    }


def _metadata(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    base = arrays["base_patch_map"]
    ids = arrays["top32_ids"]
    c1 = arrays["c1_delta"]
    c2 = arrays["c2_delta"]
    return {
        "event_id": "synthetic-event-0",
        "category": "synthetic",
        "official_group_sha256": "synthetic-group-a",
        "c0_score": max_score(base),
        "c1_score": max_score(corrected_dense_map(base, ids, c1)),
        "c2_score": max_score(corrected_dense_map(base, ids, c2)),
        "query_decision": False,
    }


def _synthetic_bank() -> SimpleNamespace:
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16)
    codebook = SimpleNamespace(protos=prototypes, K=25)
    return SimpleNamespace(normal_fixed={"p0": [{"id": [codebook]}]})


def test_raw_and_structured_fixed_bank_digests_are_distinct_algorithms():
    bank = _synthetic_bank()
    evidence = _fixed_bank_digest_evidence(
        bank, SimpleNamespace(fixed_bank_profile=fixed_bank_profile),
        _fixed_bank_digest(bank), _fixed_bank_digest)
    assert evidence["actual_structured_algorithm"] == STRUCTURED_FIXED_BANK_DIGEST_ALGORITHM
    assert evidence["actual_raw_prototype_algorithm"] == RAW_FIXED_BANK_DIGEST_ALGORITHM
    assert evidence["actual_structured_sha256"] != evidence["actual_raw_prototype_sha256"]


def test_structured_fixed_bank_digest_is_deterministic():
    bank = _synthetic_bank()
    assert _fixed_bank_digest(bank) == _fixed_bank_digest(bank)


def test_structured_fixed_bank_digest_detects_prototype_mutation():
    bank = _synthetic_bank()
    before = _fixed_bank_digest(bank)
    bank.normal_fixed["p0"][0]["id"][0].protos[0, 0] = 0.75
    assert _fixed_bank_digest(bank) != before


def test_receipt_comparison_uses_structured_digest_not_raw_profile():
    bank = _synthetic_bank()
    expected = _fixed_bank_digest(bank)
    persisted: list[dict[str, object]] = []
    evidence = _check_fixed_bank_receipt(
        bank, SimpleNamespace(fixed_bank_profile=fixed_bank_profile), expected,
        _fixed_bank_digest, persisted.append)
    assert evidence["matches_expected_structured_digest"] is True
    assert evidence["actual_raw_prototype_sha256"] != expected
    assert persisted == [evidence]


def test_receipt_mismatch_persists_both_digests_and_layer_profile_before_failure():
    bank = _synthetic_bank()
    persisted: list[dict[str, object]] = []
    with pytest.raises(R1BlockedInput) as exc_info:
        _check_fixed_bank_receipt(
            bank, SimpleNamespace(fixed_bank_profile=fixed_bank_profile), "0" * 64,
            _fixed_bank_digest, persisted.append)
    assert len(persisted) == 1
    evidence = persisted[0]
    assert evidence["matches_expected_structured_digest"] is False
    assert evidence["layer_profile"][0]["dtype"] == "float16"
    assert evidence["layer_profile"][0]["shape"] == [2, 2]
    assert evidence["layer_profile"][0]["feature_dim"] == 2
    assert evidence["expected_sha256"] in str(exc_info.value)
    assert evidence["actual_structured_sha256"] in str(exc_info.value)
    assert evidence["actual_raw_prototype_sha256"] in str(exc_info.value)


def test_source_bound_identity_rejects_wrong_revision_and_profile():
    expected = "a" * 40
    with pytest.raises(R1BlockedInput, match="Source revision mismatch"):
        _validate_execution_identity(
            branch=EXPECTED_BRANCH, head="b" * 40,
            expected_source_commit=expected,
            profile_id=SUPPORT_ONLY_PROFILE_ID, support_only=True)
    with pytest.raises(R1BlockedInput, match="Incompatible R1 profile"):
        _validate_execution_identity(
            branch=EXPECTED_BRANCH, head=expected,
            expected_source_commit=expected,
            profile_id=R1_PROFILE_ID, support_only=True)


def test_versioned_output_namespaces_cannot_select_the_failed_r1_directory():
    pilot_output = _output_namespace("case-01", support_only=False)
    support_output = _output_namespace("case-01", support_only=True)
    assert pilot_output != support_output
    assert pilot_output != LEGACY_FAILED_OUTPUT
    assert support_output != LEGACY_FAILED_OUTPUT
    assert pilot_output.name == "r1_digest_contract_v1_engineering_case-01"


def test_native_scorer_profile_mismatch_is_rejected():
    profile = NativeScorerConfig()
    p0_fields = (
        "augmentation_seed", "bank_seed", "fixed_capacity", "fixed_per_image",
        "augmentations", "include_identity", "multiplier", "feature_fuse",
        "topk_patches", "image_aggregation", "image_topk", "image_quantile", "knn",
    )
    p0_cfg = SimpleNamespace(**{name: getattr(profile, name) for name in p0_fields})
    _validate_p0_profile(p0_cfg, profile)
    p0_cfg.augmentations = 3
    with pytest.raises(R1BlockedInput, match="Incompatible frozen P0/native scorer profile"):
        _validate_p0_profile(p0_cfg, profile)


def test_historical_runtime_setup_applies_seed_and_math_only_sdp_before_load():
    calls: list[tuple[str, object]] = []

    class FakeSdp:
        def __init__(self):
            self.flash = True
            self.memory_efficient = True
            self.math = False
            self.matmul = SimpleNamespace(allow_tf32=False)

        def enable_flash_sdp(self, enabled):
            calls.append(("flash", enabled))
            self.flash = enabled

        def enable_mem_efficient_sdp(self, enabled):
            calls.append(("memory_efficient", enabled))
            self.memory_efficient = enabled

        def enable_math_sdp(self, enabled):
            calls.append(("math", enabled))
            self.math = enabled

        def flash_sdp_enabled(self):
            return self.flash

        def mem_efficient_sdp_enabled(self):
            return self.memory_efficient

        def math_sdp_enabled(self):
            return self.math

    sdp = FakeSdp()
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=sdp,
            cudnn=SimpleNamespace(deterministic=True, benchmark=False,
                                  allow_tf32=False)),
        are_deterministic_algorithms_enabled=lambda: True,
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda="synthetic"),
        __version__="synthetic",
    )
    impl = SimpleNamespace(set_seed=lambda seed: calls.append(("seed", seed)))
    profile = _configure_historical_runtime(impl, fake_torch)
    assert calls == [("seed", 0), ("flash", False),
                     ("memory_efficient", False), ("math", True)]
    assert profile["sdp"] == {
        "flash_enabled": False,
        "memory_efficient_enabled": False,
        "math_enabled": True,
    }


def test_dense_reconstruction_preserves_native_topk_and_exact_max():
    arrays = _arrays()
    before = arrays_sha256({"base_patch_map": arrays["base_patch_map"]})
    restored = corrected_dense_map(arrays["base_patch_map"],
                                   arrays["top32_ids"], arrays["c1_delta"])
    assert np.array_equal(arrays["top32_ids"], np.arange(752, 784, dtype=np.int64))
    assert restored[783] == pytest.approx(float(arrays["base_patch_map"][783]) - 0.5)
    assert max_score(restored) == float(restored[782])
    zero = corrected_dense_map(arrays["base_patch_map"], arrays["top32_ids"],
                               np.zeros(32, dtype=np.float32))
    assert np.array_equal(zero, arrays["base_patch_map"])
    assert max_score(zero) == max_score(arrays["base_patch_map"])
    assert arrays_sha256({"base_patch_map": arrays["base_patch_map"]}) == before


def test_serialized_dense_map_and_selected_ids_replay_exactly(tmp_path):
    arrays = _arrays()
    writer = EvidenceTapeWriter(tmp_path / "tape")
    writer.seal_event(0, arrays, _metadata(arrays))
    writer.finalize_event(0, {"writer_action": {"action": "no_write"},
                              "causal_order_complete": True})
    result = writer.verify()
    assert result["passed"]
    with np.load(tmp_path / "tape" / "events" / "event_000000.npz",
                 allow_pickle=False) as restored:
        assert np.array_equal(restored["top32_ids"], arrays["top32_ids"])
        assert np.array_equal(restored["base_patch_map"], arrays["base_patch_map"])
        replay = corrected_dense_map(restored["base_patch_map"], restored["top32_ids"],
                                     restored["c1_delta"])
        assert max_score(replay) == _metadata(arrays)["c1_score"]


def test_checksums_and_npz_bytes_are_deterministic():
    arrays = _arrays()
    first = deterministic_npz_bytes(arrays)
    second = deterministic_npz_bytes({key: value.copy()
                                      for key, value in reversed(list(arrays.items()))})
    assert first == second
    assert sha256_bytes(first) == sha256_bytes(second)
    assert arrays_sha256(arrays) == arrays_sha256(dict(reversed(list(arrays.items()))))


def test_query_label_provider_is_unavailable_until_score_seal():
    calls: list[str] = []
    gate = CausalEventOrder(9)
    with pytest.raises(R1CausalityError):
        gate.disclose(lambda: calls.append("read") or 0)
    assert calls == []
    gate.seal_scores()
    gate.decide_query(True)
    assert gate.disclose(lambda: calls.append("read") or 1) == 1
    gate.corrections_updated()
    gate.memory_updated()
    gate.finish()
    assert calls == ["read"]
    assert gate.phase == "complete"


def test_unqueried_event_never_requests_a_label():
    gate = CausalEventOrder(2)
    gate.seal_scores()
    gate.decide_query(False)
    with pytest.raises(R1CausalityError):
        gate.disclose(lambda: pytest.fail("unqueried label provider was called"))
    gate.corrections_updated()
    gate.memory_updated()
    gate.finish()


def test_source_and_future_anchors_are_excluded():
    import hashlib
    same, other, future = [hashlib.sha256(value.encode()).hexdigest()
                           for value in ("same", "other", "future")]
    positions = [1, 2]
    groups = [same, other]
    kept = source_disjoint_anchor_indices(positions, groups,
                                         event_position=4, event_group=same)
    assert kept == [1]
    with pytest.raises(R1CausalityError, match="current/future"):
        source_disjoint_anchor_indices([5], [future],
                                       event_position=4, event_group=same)
    anchors = [
        {"position": 1, "group_sha256": same, "sign": 1,
         "patch_id": 3, "feature": np.array([1.0, 0.0], dtype=np.float32),
         "feature_sha256": arrays_sha256({"feature": np.array([1.0, 0.0], dtype=np.float32)})},
        {"position": 2, "group_sha256": other, "sign": -1,
         "patch_id": 4, "feature": np.array([1.0, 0.0], dtype=np.float32),
         "feature_sha256": arrays_sha256({"feature": np.array([1.0, 0.0], dtype=np.float32)})},
    ]
    delta, count = signed_knn_delta(
        np.array([[1.0, 0.0]], dtype=np.float32), anchors,
        event_position=4, event_group=same, neighbors=5, amplitude=0.20)
    assert count == 1
    assert delta.tolist() == pytest.approx([-0.20])


def test_tape_refuses_duplicate_gapped_resume_and_label_payloads(tmp_path):
    arrays = _arrays()
    writer = EvidenceTapeWriter(tmp_path / "tape")
    with pytest.raises(TapeIntegrityError):
        writer.seal_event(1, arrays, _metadata(arrays))
    with pytest.raises(TapeIntegrityError):
        writer.finalize_event(0, {"writer_action": {"action": "too_early"}})
    with pytest.raises(TapeIntegrityError):
        writer.seal_event(0, arrays, {**_metadata(arrays), "label": 1})
    writer.seal_event(0, arrays, _metadata(arrays))
    with pytest.raises(TapeIntegrityError):
        writer.seal_event(0, arrays, _metadata(arrays))
    with pytest.raises(FileExistsError):
        EvidenceTapeWriter(tmp_path / "tape")
    writer.finalize_event(0, {"writer_action": {"action": "no_write"}})
    with pytest.raises(TapeIntegrityError):
        writer.finalize_event(0, {"writer_action": {"action": "duplicate"}})
    assert writer.verify()["passed"]


def test_score_replay_does_not_mutate_baseline_map_or_digest():
    arrays = _arrays()
    baseline = arrays["base_patch_map"].copy()
    digest_before = hashlib.sha256(baseline.tobytes()).hexdigest()
    _ = corrected_dense_map(baseline, arrays["top32_ids"], arrays["c1_delta"])
    _ = corrected_dense_map(baseline, arrays["top32_ids"], arrays["c2_delta"])
    assert np.array_equal(baseline, arrays["base_patch_map"])
    assert hashlib.sha256(baseline.tobytes()).hexdigest() == digest_before


def test_tape_readback_rejects_modified_shard(tmp_path):
    arrays = _arrays()
    writer = EvidenceTapeWriter(tmp_path / "tape")
    writer.seal_event(0, arrays, _metadata(arrays))
    writer.finalize_event(0, {"writer_action": {"action": "no_write"}})
    shard = tmp_path / "tape" / "events" / "event_000000.npz"
    shard.write_bytes(shard.read_bytes() + b"corruption")
    with pytest.raises(TapeIntegrityError):
        writer.verify()


def test_explicit_resume_accepts_only_the_empty_bootstrap_failure(tmp_path):
    root = tmp_path / "bootstrap-tape"
    (root / "events").mkdir(parents=True)
    (root / "run_manifest.json").write_text(json.dumps({
        "schema": "calibration.r1_evidence_tape.v1",
        "status": "INCOMPLETE", "sealed_events": 0,
        "finalized_events": 0, "next_position": 0,
        "failure_type": "ModuleNotFoundError",
        "failure_message": "No module named 'experiments'",
    }), encoding="utf-8")
    with pytest.raises(FileExistsError):
        EvidenceTapeWriter(root)
    resumed = EvidenceTapeWriter(root, resume_empty_bootstrap=True)
    assert resumed.next_sealed == resumed.next_finalized == 0
    history = json.loads((root / "attempt_history.jsonl").read_text(encoding="utf-8"))
    assert history["outcome"] == "EMPTY_BOOTSTRAP_FAILURE_RETAINED"
    arrays = _arrays()
    resumed.seal_event(0, arrays, _metadata(arrays))
    resumed.finalize_event(0, {"writer_action": {"action": "no_write"}})
    assert resumed.verify()["passed"]

    contaminated = tmp_path / "contaminated-tape"
    (contaminated / "events").mkdir(parents=True)
    (contaminated / "events" / "event_000000.npz").write_bytes(b"partial")
    (contaminated / "run_manifest.json").write_text(json.dumps({
        "schema": "calibration.r1_evidence_tape.v1",
        "status": "INCOMPLETE", "sealed_events": 0,
        "finalized_events": 0, "next_position": 0,
        "failure_type": "ModuleNotFoundError",
        "failure_message": "No module named 'experiments'",
    }), encoding="utf-8")
    with pytest.raises(TapeIntegrityError):
        EvidenceTapeWriter(contaminated, resume_empty_bootstrap=True)
