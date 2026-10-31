"""CPU-only causal protocol regressions for the isolated R1 pilot."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from experiments.calibration.r1_matched_correction_pilot import (
    C2_ANCHOR_CAPACITY,
    CausalEventOrder,
    EvidenceTapeWriter,
    OrderedQueriedLabelReader,
    PilotConfig,
    R1CausalityError,
    TapeIntegrityError,
    _seal_prediction_and_reveal,
    _c1_score_delta,
    _record_c1_source_group_updates,
    _validate_c1_source_groups,
    _validate_c2_anchor_state,
    arrays_sha256,
    common_c0_greedy_writer_action,
    signed_knn_delta,
)


def _group(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _event_arrays() -> dict[str, np.ndarray]:
    base = np.linspace(0.0, 0.783, 784, dtype=np.float32)
    ids = np.arange(752, 784, dtype=np.int64)
    return {
        "base_patch_map": base,
        "top32_ids": ids,
        "top32_features": np.ones((32, 8), dtype=np.float32),
        "c1_delta": np.zeros(32, dtype=np.float32),
        "c2_delta": np.zeros(32, dtype=np.float32),
    }


def _event_metadata(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return {"event_id": "synthetic-0", "category": "synthetic",
            "official_group_sha256": _group("event"),
            "c0_score": float(arrays["base_patch_map"].max()),
            "c1_score": float(arrays["base_patch_map"].max()),
            "c2_score": float(arrays["base_patch_map"].max())}


def _c2_anchor(position: int, group: str, patch_id: int = 0,
               sign: int = 1, vector: np.ndarray | None = None) -> dict[str, object]:
    feature = (np.array([1.0, 0.0], dtype=np.float32) if vector is None else
               np.asarray(vector, dtype=np.float32))
    return {"position": position, "group_sha256": group,
            "patch_id": patch_id, "sign": sign, "feature": feature,
            "feature_sha256": arrays_sha256({"feature": feature})}


def test_seal_precedes_query_and_failed_seal_never_reveals(tmp_path):
    arrays = _event_arrays()
    provider_calls: list[int] = []
    tape = EvidenceTapeWriter(tmp_path / "tape")
    failed_gate = CausalEventOrder(1)
    with pytest.raises(TapeIntegrityError, match="Expected next sealed position"):
        _seal_prediction_and_reveal(
            gate=failed_gate, tape=tape, position=1, arrays=arrays,
            metadata=_event_metadata(arrays), query_positions=[1],
            label_provider=lambda: provider_calls.append(1) or 0)
    assert provider_calls == []
    assert failed_gate.phase == "observed"

    gate = CausalEventOrder(0)
    sealed, queried, label = _seal_prediction_and_reveal(
        gate=gate, tape=tape, position=0, arrays=arrays,
        metadata=_event_metadata(arrays), query_positions=[0, 2],
        label_provider=lambda: provider_calls.append(0) or 0)
    assert sealed["position"] == 0
    assert queried is True and label == 0
    assert gate.phase == "label_disclosed"
    assert "query_decision" not in sealed
    gate.corrections_updated()
    gate.memory_updated()
    tape.finalize_event(0, {"query_decision": True,
                            "query_reference": "synthetic:0",
                            "writer_action": {"action": "no_write"}})
    gate.finish()
    assert gate.phase == "complete"
    assert provider_calls == [0]
    assert tape.verify()["passed"]


@pytest.mark.parametrize("label", [0, 1])
def test_queried_normal_and_defect_labels_follow_reveal_phase(label):
    gate = CausalEventOrder(3)
    gate.seal_scores()
    gate.decide_query(True)
    assert gate.disclose(lambda: label) == label
    gate.corrections_updated()
    gate.memory_updated()
    gate.finish()


@pytest.mark.parametrize("prediction", [0, 1])
def test_unqueried_normal_or_defect_looking_event_never_discloses(prediction):
    gate = CausalEventOrder(prediction)
    gate.seal_scores()
    gate.decide_query(False)
    with pytest.raises(R1CausalityError):
        gate.disclose(lambda: pytest.fail("unqueried label reader was called"))
    gate.corrections_updated()
    gate.memory_updated()
    gate.finish()


def test_causal_state_machine_rejects_early_repeat_and_out_of_order_disclosure():
    calls: list[int] = []
    early = CausalEventOrder(0)
    with pytest.raises(R1CausalityError):
        early.disclose(lambda: calls.append(0) or 0)
    early.seal_scores()
    early.decide_query(True)
    assert early.disclose(lambda: calls.append(0) or 1) == 1
    with pytest.raises(R1CausalityError):
        early.disclose(lambda: calls.append(1) or 0)
    assert calls == [0]
    with pytest.raises(R1CausalityError):
        CausalEventOrder(1).decide_query(True)


def test_real_restricted_reader_is_wrapped_with_exact_query_order(tmp_path):
    from experiments.calibration.stage1a_p12_label_provider import QueriedPositionLabelReader

    path = tmp_path / "synthetic_category.json"
    path.write_text(json.dumps({
        "meta": {"normal_class": "good"},
        "test": [{"anomaly_class": "good"}, {"anomaly_class": "bad"},
                 {"anomaly_class": "good"}],
    }), encoding="utf-8")
    keys = {0: "synthetic-a", 2: "synthetic-c"}
    groups = {position: hashlib.sha256(
        ("official-sample-group\0" + key).encode("utf-8")).hexdigest()
        for position, key in keys.items()}
    underlying = QueriedPositionLabelReader(
        category="synthetic", path=path,
        expected_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        positions=[0, 2], group_sha256_by_position=groups)
    reader = OrderedQueriedLabelReader(underlying, [0, 2])
    try:
        with pytest.raises(R1CausalityError, match="next frozen tape position"):
            reader("synthetic", 2, keys[2])
        assert reader.decoded_positions == []
        assert reader("synthetic", 0, keys[0]) == 0
        assert reader("synthetic", 2, keys[2]) == 0
        assert reader.decoded_positions == [0, 2]
        with pytest.raises(R1CausalityError, match="next frozen tape position"):
            reader("synthetic", 2, keys[2])
    finally:
        reader.close()


def test_c2_current_event_anchor_is_post_update_only_then_source_filtered():
    same, other = _group("same"), _group("other")
    current = _c2_anchor(5, same)
    assert _validate_c2_anchor_state([current], event_position=5,
                                     phase="post_update") == []
    with pytest.raises(R1CausalityError, match="current/future"):
        _validate_c2_anchor_state([current], event_position=5,
                                  phase="pre_score", event_group=other)
    assert _validate_c2_anchor_state([current], event_position=6,
                                     phase="pre_score", event_group=other) == [0]
    assert _validate_c2_anchor_state([current], event_position=6,
                                     phase="pre_score", event_group=same) == []
    with pytest.raises(R1CausalityError, match="future anchor"):
        _validate_c2_anchor_state([_c2_anchor(6, other)], event_position=5,
                                  phase="post_update")

    delta, count = signed_knn_delta(
        np.array([[1.0, 0.0]], dtype=np.float32), [current],
        event_position=6, event_group=other, neighbors=5, amplitude=0.20)
    assert count == 1 and delta.tolist() == pytest.approx([0.20])
    delta_same, count_same = signed_knn_delta(
        np.array([[1.0, 0.0]], dtype=np.float32), [current],
        event_position=6, event_group=same, neighbors=5, amplitude=0.20)
    assert count_same == 0 and delta_same.tolist() == [0.0]


def test_c2_provenance_is_fail_closed_on_missing_or_corrupt_feature_digest():
    good = _c2_anchor(0, _group("good"))
    with pytest.raises(R1CausalityError, match="missing/malformed"):
        _validate_c2_anchor_state([{"position": 0}], event_position=1,
                                  phase="pre_score", event_group=_group("query"))
    corrupt = {**good, "feature_sha256": "0" * 64}
    with pytest.raises(R1CausalityError, match="feature/provenance digest"):
        _validate_c2_anchor_state([corrupt], event_position=1,
                                  phase="pre_score", event_group=_group("query"))


def test_native_c1_insert_merge_and_different_source_union_fail_closed():
    from test_phase3_AIF_simplified_v22 import CorrectionConfig, SignedCorrectionCodebook

    config = CorrectionConfig(merge_eps=0.05)
    codebook = SignedCorrectionCodebook(2, 2, torch.device("cpu"), config)
    groups: dict[int, set[str]] = {}
    vectors = [torch.tensor([[1.0, 0.0]]),
               torch.tensor([[0.9998, 0.02]]),
               torch.tensor([[0.9995, 0.03]])]
    sources = [_group("A"), _group("B"), _group("C")]
    first = codebook.update(vectors[0], torch.tensor([0.10]), outcome="TN", step=0)
    assert first["inserted"] == 1
    _record_c1_source_group_updates(groups, last_steps=[0], event_position=0,
                                   event_group=sources[0])
    assert _validate_c1_source_groups(groups, [0], event_position=0,
                                      phase="post_update") == []
    with pytest.raises(R1CausalityError, match="current/future last_step"):
        _validate_c1_source_groups(groups, [0], event_position=0,
                                   phase="pre_score", event_group=sources[1])
    for position in (1, 2):
        result = codebook.update(vectors[position], torch.tensor([0.10]),
                                  outcome="TN", step=position)
        assert result["merged"] == 1
        _record_c1_source_group_updates(
            groups, last_steps=codebook.last_step[:codebook.size()].tolist(),
            event_position=position, event_group=sources[position])
    assert groups == {0: set(sources)}
    assert _validate_c1_source_groups(groups, codebook.last_step[:1].tolist(),
                                      event_position=3, phase="pre_score",
                                      event_group=sources[2]) == []
    assert _validate_c1_source_groups(groups, codebook.last_step[:1].tolist(),
                                      event_position=4, phase="pre_score",
                                      event_group=_group("D")) == [0]


def test_real_c1_score_adapter_rejects_same_event_and_allows_next_event():
    from types import SimpleNamespace
    import test_phase3_AIF_simplified_v22 as native

    config = native.CorrectionConfig(merge_eps=0.05)
    codebook = native.SignedCorrectionCodebook(
        2, 2, torch.device("cpu"), config)
    source = _group("current-c1")
    codebook.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0.10]),
                    outcome="TN", step=0)
    provenance = {0: {source}}
    bank = SimpleNamespace(corr_cb={"p0": {"id": [codebook]}},
                           corr_cfg=config, device=torch.device("cpu"))
    impl = SimpleNamespace(SignedCorrectionCodebook=native.SignedCorrectionCodebook,
                           l2_normalize=native.l2_normalize)
    selected = torch.tensor([[1.0, 0.0]] * 32, dtype=torch.float32)
    with pytest.raises(R1CausalityError, match="current/future last_step"):
        _c1_score_delta(impl, bank, selected, provenance, _group("other"),
                        event_position=0)
    delta, diagnostics = _c1_score_delta(
        impl, bank, selected, provenance, _group("other"), event_position=1)
    assert delta.shape == (32,)
    assert diagnostics["eligible_anchor_slots"] == 1
    excluded, diagnostics_same = _c1_score_delta(
        impl, bank, selected, provenance, source, event_position=1)
    assert np.array_equal(excluded, np.zeros(32, dtype=np.float32))
    assert diagnostics_same["eligible_anchor_slots"] == 0


def test_native_c1_slot_replacement_retains_ambiguous_old_provenance():
    from test_phase3_AIF_simplified_v22 import CorrectionConfig, SignedCorrectionCodebook

    codebook = SignedCorrectionCodebook(
        1, 2, torch.device("cpu"), CorrectionConfig(merge_eps=0.05))
    groups: dict[int, set[str]] = {}
    old_source, replacement_source = _group("old"), _group("replacement")
    codebook.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0.10]),
                    outcome="TN", step=0)
    _record_c1_source_group_updates(groups, last_steps=[0], event_position=0,
                                   event_group=old_source)
    replaced = codebook.update(torch.tensor([[0.0, 1.0]]), torch.tensor([-0.20]),
                               outcome="FN", step=1)
    assert replaced["replaced"] == 1
    _record_c1_source_group_updates(groups, last_steps=[1], event_position=1,
                                   event_group=replacement_source)
    assert groups == {0: {old_source, replacement_source}}
    assert _validate_c1_source_groups(groups, [1], event_position=2,
                                      phase="pre_score", event_group=old_source) == []
    assert _validate_c1_source_groups(groups, [1], event_position=2,
                                      phase="pre_score",
                                      event_group=replacement_source) == []


@pytest.mark.parametrize("groups,last_steps", [
    ({0: set()}, [0]),
    ({1: {_group("stale")}}, [0]),
    ({0: {"not-a-sha256"}}, [0]),
    ({0: {_group("old")}}, [-1]),
])
def test_c1_missing_stale_or_malformed_provenance_fails_closed(groups, last_steps):
    with pytest.raises(R1CausalityError):
        _validate_c1_source_groups(groups, last_steps, event_position=2,
                                   phase="pre_score", event_group=_group("query"))


def test_common_writer_is_invariant_to_c1_c2_outputs_and_uses_only_c0():
    ids = np.arange(752, 784, dtype=np.int64)
    features = np.ones((32, 8), dtype=np.float32)

    def one_run(c1_output: float, c2_output: float):
        memory: list[tuple[object, ...]] = []
        eligibility_calls: list[dict[str, object]] = []

        def eligibility(**kwargs):
            eligibility_calls.append(dict(kwargs))
            return True, 32, "synthetic_allowed", None

        def update(**kwargs):
            memory.append((kwargs["position"], kwargs["candidate_ids"].tolist(),
                           kwargs["score"], kwargs["theta"], kwargs["sigma"]))
            return {"accepted_patches_delta": 32, "skipped_patches_delta": 0}

        # Correction outputs are deliberately different and are not writer inputs.
        _ = c1_output, c2_output
        action = common_c0_greedy_writer_action(
            position=7, c0_score=0.25, theta=0.5, sigma=0.1,
            c0_prediction=0, candidate_ids=ids, candidate_features=features,
            feature_layers=["shared-synthetic-features"], queried=True,
            disclosed_label=0, eligibility=eligibility, update=update)
        return action, memory, eligibility_calls

    first = one_run(-1000.0, 1000.0)
    second = one_run(1e30, -1e30)
    assert first == second
    assert first[0]["input_score_source"] == "common_C0"
    assert first[0]["action"] == "native_greedy_normal_update"
    assert first[1] == [(7, ids.tolist(), 0.25, 0.5, 0.1)]
    assert first[2] == [{"score": 0.25, "theta": 0.5, "sigmaN": 0.1,
                         "pred_defect": False, "novelty": 0.0}]


@pytest.mark.parametrize("queried,label", [(False, None), (True, 1)])
def test_common_writer_never_writes_unqueried_or_verified_defect(queried, label):
    def forbidden(**kwargs):
        pytest.fail("eligibility/update called without queried-normal authorization")

    action = common_c0_greedy_writer_action(
        position=0, c0_score=0.8, theta=0.5, sigma=0.1,
        c0_prediction=1, candidate_ids=np.arange(752, 784, dtype=np.int64),
        candidate_features=np.ones((32, 8), dtype=np.float32),
        feature_layers=[], queried=queried, disclosed_label=label,
        eligibility=forbidden, update=forbidden)
    assert action["action"] in {"no_normal_memory_write", "verified_defect_no_normal_write"}


def test_evidence_tape_preserves_sealed_partial_state_and_refuses_implicit_resume(tmp_path):
    root = tmp_path / "partial-tape"
    writer = EvidenceTapeWriter(root)
    arrays = _event_arrays()
    writer.seal_event(0, arrays, _event_metadata(arrays))
    with pytest.raises(RuntimeError, match="injected"):
        raise RuntimeError("injected post-seal failure")
    partial = writer.verify(require_complete=False)
    assert partial == {**partial, "sealed_events": 1, "finalized_events": 0,
                       "passed": True}
    assert writer.manifest["status"] == "INCOMPLETE"
    with pytest.raises(TapeIntegrityError, match="lacks a post-feedback"):
        writer.verify()
    with pytest.raises(FileExistsError):
        EvidenceTapeWriter(root)


def test_tape_rejects_public_mask_payload_and_retains_c2_fixture_defaults(tmp_path):
    writer = EvidenceTapeWriter(tmp_path / "mask-tape")
    with pytest.raises(TapeIntegrityError, match="label/mask payload"):
        writer.seal_event(0, _event_arrays(),
                          {**_event_metadata(_event_arrays()), "defect_mask": [1]})
    config = PilotConfig()
    assert config.c2_neighbors == 5
    assert config.c2_interface_scale == pytest.approx(0.20)
    assert C2_ANCHOR_CAPACITY == 256
