from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.calibration.fs4_durable_ledger import DurableLedger, LedgerIntegrityError
from experiments.calibration.fs4_errors import (FS4FatalIntegrityError, FS4NativeScoringIntegrityError,
    FS4ProvenanceIntegrityError, FS4RecoverableAvailabilityError, FS4StateMutationError)
from experiments.calibration.fs4_query_stress import SCENARIOS, run_scenario
from experiments.calibration.fs4_stochastic_bea import SealedPrediction, StochasticBEAGroupSlot, receipt_digest
from experiments.calibration.fs4_threshold_controllers import SelectionAwareSourceExcludedThreshold
from experiments.calibration.fs4_threshold_core import NormalEvidence
from experiments.calibration.fs4_threshold_integration import CoverageMemory, FEATURE_CONVENTION, SharedTrajectory, prediction_digest
from experiments.calibration.fs4_threshold_integration import CoupledTrajectory
from experiments.calibration.fs4_threshold_synthetic_smoke import _SyntheticRuntime


def hid(text):
    return hashlib.sha256(text.encode()).hexdigest()


def evidence_records():
    p = StochasticBEAGroupSlot()
    evidence = []
    for group_number in range(12):
        for view in range(5):
            p.price = 0.05
            pos = p.last_position + 1
            group, event = hid(f"g{group_number}"), hid(f"e{group_number}-{view}")
            pred = SealedPrediction(pos, group, event, 0.5, 0.5, hid("g"), hid("gc"), hid("t"), hid("s"))
            r = p.decide(pred, reference_threshold=0.5, novelty_percentile=1, correction_top32=np.zeros(32),
                correction_bank_size=0, topk_dN_max=1, topk_dN_mean=.5, pre_score_scale=p.prior_scale())
            r = p.seal_query(r, receipt_digest(r))
            if r.executed:
                r = p.disclose(r, 0)
                evidence.append(NormalEvidence.from_disclosure(p, r, normal_version=0, correction_version=0))
    assert len(evidence) == 12
    return evidence


class Cache:
    def __init__(self):
        self.records = {}

    def add(self, evidence, *_args, **_kwargs):
        self.records[evidence.group] = SimpleNamespace(evidence=evidence)


def make_controller(scorer):
    controller = SelectionAwareSourceExcludedThreshold(.4, scorer, Cache())
    for e in evidence_records():
        controller.add(e, None, height=28, width=28, normalization="synthetic")
    return controller


@pytest.mark.parametrize("error", [FS4ProvenanceIntegrityError("provenance"),
    FS4NativeScoringIntegrityError("native"), FS4StateMutationError("mutation"),
    FS4FatalIntegrityError("fatal"), ValueError("unexpected native failure")])
def test_fatal_rescore_poisons_condition_and_never_moves_threshold(error):
    class Scorer:
        def score(self, *_args, **_kwargs):
            raise error
    controller = make_controller(Scorer())
    with pytest.raises(FS4FatalIntegrityError):
        controller.refresh(state_g="g", state_gc="gc", event_position=100)
    assert controller.engine.theta == {"G": .4, "GC": .4}
    assert controller.successful_tokens == {"G": None, "GC": None}
    with pytest.raises(FS4FatalIntegrityError):
        controller.refresh(state_g="g", state_gc="gc", event_position=101)


def test_recoverable_rescore_can_retry_same_version_once():
    class Scorer:
        unavailable = True
        def score(self, *_args, **_kwargs):
            if self.unavailable:
                raise FS4RecoverableAvailabilityError("temporary resource unavailable")
            return {"G": .7, "GC": .8}
    scorer = Scorer(); c = make_controller(scorer)
    assert c.refresh(state_g="g", state_gc="gc", event_position=100)["G"]["status"] == "CURRENT_STATE_RESCORING_UNAVAILABLE"
    assert c.successful_tokens == {"G": None, "GC": None}
    scorer.unavailable = False
    assert c.refresh(state_g="g", state_gc="gc", event_position=101)["G"]["status"] == "UPDATE_ACCEPTED"
    c.refresh(state_g="g", state_gc="gc", event_position=102)
    assert c.engine.theta == {"G": pytest.approx(.41), "GC": pytest.approx(.41)}


def test_nonfinite_and_missing_cache_are_fatal():
    class Scorer:
        def score(self, *_args, **_kwargs): return {"G": float("nan"), "GC": .5}
    c = make_controller(Scorer())
    with pytest.raises(FS4NativeScoringIntegrityError): c.refresh(state_g="g", state_gc="gc", event_position=100)
    c = make_controller(Scorer()); c.cache.records.clear()
    with pytest.raises(FS4ProvenanceIntegrityError): c.refresh(state_g="g", state_gc="gc", event_position=100)


def test_native_c1_provenance_and_mutation_before_exception_fail_fatally(monkeypatch):
    from test_fs4_threshold import native_bank_and_writer
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration.fs4_greedy_provenance import GreedyProvenance, _native_modules
    from experiments.calibration.fs4_source_excluded_scorer import NativeSourceExcludedScorer, VerifiedNormalFeatureCache
    native, bank, _writer = native_bank_and_writer(4096)
    bank.corr_cfg.corr_weight = 1.0
    sidecar = GreedyProvenance(bank); cb = bank.corr_cb["p0"]["id"][0]
    cache = VerifiedNormalFeatureCache(); evidence = evidence_records()[0]
    record = cache.add(evidence, torch.zeros((784, 4096)), height=28, width=28, normalization=FEATURE_CONVENTION)
    scorer = NativeSourceExcludedScorer(impl=native, bank=bank,
        cfg=SimpleNamespace(topk_patches=32, image_aggregation="max", image_topk=64, image_quantile=.95),
        correction_cfg=bank.corr_cfg, correction_cb=cb, correction_sources={},
        provenance=sidecar, normalization=FEATURE_CONVENTION)
    cb.n = 1; cb.last_step[0] = 0
    with pytest.raises(FS4ProvenanceIntegrityError, match="C1 source provenance"):
        scorer.score(record, event_position=100)
    cb.n = 0
    bridge, _ = _native_modules()
    def mutate_then_fail(*_args, **_kwargs):
        dynamic = bridge.dynamic_codebooks(bank)[0]
        dynamic.protos = torch.ones((1, 4096), dtype=dynamic.dtype, device=dynamic.device)
        raise RuntimeError("injected native failure after mutation")
    monkeypatch.setattr(r1, "_native_dense_score", mutate_then_fail)
    with pytest.raises(FS4StateMutationError):
        scorer.score(record, event_position=100)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_policy_stress_has_one_candidate_and_never_invents_probability(scenario):
    r = run_scenario(scenario, groups=40)
    assert r["events"] == 200 and r["candidate_slots"] == 40
    assert r["queries"] <= 40 and set(r["eligible_design_probabilities"]) <= {"0.2", "0.010000000000000002"}
    if scenario == "no_exploitation":
        assert r["verified_synthetic_normals"] == 0


def manifest():
    path = Path("experiments/v02_memory.py")
    return {"run_id": "synthetic", "source_commit": "3bd33076789f66f7086f8f2e165b18d789e2c7d2",
            "source_fingerprints": {str(path).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()},
            "expected_events_by_category": {"synthetic": 5}, "profile": "SHARED_TRAJECTORY_THRESHOLD_ONLY"}


def test_durable_order_reopen_and_stage_a_seals(tmp_path):
    m = manifest(); root = tmp_path / "run"; ledger = DurableLedger(root, m, create=True)
    ledger.select_category("synthetic")
    with pytest.raises(LedgerIntegrityError): ledger.disclose(hid("missing"), 0, lambda *_: 0)
    policy = StochasticBEAGroupSlot(); runtime = _SyntheticRuntime(.5)
    loop = SharedTrajectory(runtime=runtime, coverage=CoverageMemory(np.eye(5, dtype=np.float32)[:4]), theta0=.5,
        policy=policy, seal_prediction=ledger.seal_prediction, seal_query=ledger.seal_query,
        label_provider=lambda event, pos: ledger.disclose(event, pos, lambda *_: 0))
    image = SimpleNamespace(height=28, width=28); features = torch.zeros((784, 4096), dtype=torch.float32)
    for pos in range(5):
        row = loop.step(position=pos, group=hid("group"), event=hid(f"event{pos}"), image=image,
                        features=features, cls_feature=np.eye(5, dtype=np.float32)[4], feature_convention=FEATURE_CONVENTION)
        ledger.complete_event(position=pos, post_state_digest=prediction_digest(runtime.states()), transition={"mode": row["mode"]})
    policy.validate_complete_stream()
    with pytest.raises(LedgerIntegrityError): ledger.disclose(hid("event4"), 4, lambda *_: 0)
    ledger.seal_category("synthetic"); ledger.seal_global_stage_a(); expected = ledger.offset(); ledger.close()
    reopened = DurableLedger(root, m, create=False)
    assert reopened.offset() == expected
    authorization = {"authorization_id": "synthetic-test-only", "run_id": m["run_id"],
                     "manifest_sha256": __import__("experiments.calibration.fs4_durable_ledger", fromlist=["digest"]).digest(m),
                     "stage_a_sha256": reopened.last_hash, "scope": "FS4_STAGE_B"}
    with pytest.raises(LedgerIntegrityError):
        reopened.authorize_stage_b(authorization, verifier=lambda _: False)
    reopened.authorize_stage_b(authorization, verifier=lambda _: True); reopened.seal_complete(); reopened.close()


def test_corrupt_ledger_and_manifest_rejected(tmp_path):
    m = manifest(); root = tmp_path / "run"; ledger = DurableLedger(root, m, create=True)
    ledger.select_category("synthetic"); ledger.close()
    with (root / "events.jsonl").open("ab") as out: out.write(b"partial")
    with pytest.raises(LedgerIntegrityError, match="partial"):
        DurableLedger(root, m, create=False)
    wrong = dict(m, run_id="other")
    with pytest.raises(LedgerIntegrityError, match="manifest"):
        DurableLedger(root, wrong, create=False)


def test_durable_executor_journals_complete_transitions(tmp_path):
    from experiments.calibration.fs4_durable_runtime import DurableExecutor
    m = manifest(); ledger = DurableLedger(tmp_path / "run", m, create=True); ledger.select_category("synthetic")
    loop = SharedTrajectory(runtime=_SyntheticRuntime(.5), coverage=CoverageMemory(np.eye(5, dtype=np.float32)[:4]),
        theta0=.5, policy=StochasticBEAGroupSlot(), seal_prediction=prediction_digest,
        seal_query=receipt_digest, label_provider=lambda *_: 0)
    executor = DurableExecutor(loop, ledger, lambda *_: 0)
    features = torch.zeros((784, 4096)); image = SimpleNamespace(height=28, width=28)
    for pos in range(5):
        executor.step(position=pos, group=hid("executor-group"), event=hid(f"executor-{pos}"), image=image,
            features=features, cls_feature=np.eye(5, dtype=np.float32)[4], feature_convention=FEATURE_CONVENTION)
    executor.seal_category("synthetic")
    assert sum(row["kind"] == "TRANSITION" for row in ledger.records) == 5
    assert all(row["payload"]["transition"]["profile"] == "SHARED_TRAJECTORY_THRESHOLD_ONLY"
               for row in ledger.records if row["kind"] == "TRANSITION")
    ledger.close()


def test_coupled_query_only_records_disagreement_and_frozen_reference():
    from experiments.calibration.fs4_stochastic_bea import action_uniform, candidate_ordinal
    group = next(hid(f"candidate-{i}") for i in range(1000)
                 if candidate_ordinal(37, hid(f"candidate-{i}")) == 1 and
                 action_uniform(37, hid(f"candidate-{i}"), hid("event")) >= .05)
    def make():
        return CoupledTrajectory(controller="L", runtime=_SyntheticRuntime(.5),
            coverage=CoverageMemory(np.eye(5, dtype=np.float32)[:4]), theta0=.5,
            policy=StochasticBEAGroupSlot(), seal_prediction=prediction_digest,
            seal_query=receipt_digest, label_provider=lambda *_: 0)
    near, far = make(), make()
    near.controller.engine.theta["G"] = 1.0
    far.controller.engine.theta["GC"] = 10.0
    args = dict(position=0, group=group, event=hid("event"), image=SimpleNamespace(height=28, width=28),
                features=torch.zeros((784, 4096), dtype=torch.float32),
                cls_feature=np.eye(5, dtype=np.float32)[4], feature_convention=FEATURE_CONVENTION)
    a, b = near.step(**args), far.step(**args)
    assert a["mode"] == b["mode"] == "COUPLED_QUERY_ONLY_FROZEN_WRITER"
    assert a["query"].executed and not b["query"].executed
    p = a["sealed_prediction"]
    assert p["learning_reference_threshold"] == .5 and p["learning_prediction"] == 1
    assert p["decisions"]["G"] == 0 and p["decision_learning_disagreement"]["G"]
    assert a["feedback_update"]["learning_reference_threshold"] == .5
    assert a["feedback_update"]["learning_prediction"] == 1


def test_resource_accounting_and_preflight(tmp_path):
    from experiments.calibration.fs4_resources import BYTES_PER_NORMAL, FULL_CACHE_BYTES, preflight, rescore_projection
    assert BYTES_PER_NORMAL == 12_845_056 and FULL_CACHE_BYTES == 13_153_337_344
    r = preflight(tmp_path, cache_groups=2, required_disk_bytes=0)
    assert r["cache_tensor_bytes"] == 2 * BYTES_PER_NORMAL
    estimate = rescore_projection([.1, .2, .3], groups=10, future_refreshes=5)
    assert estimate["projected_rescore_calls"] == 50
    assert estimate["projected_seconds_at_observed_mean"] == pytest.approx(10)
    with pytest.raises(ValueError): preflight(tmp_path, cache_groups=1025, required_disk_bytes=0)


def test_native_checkpoint_boundary_restores_memory_and_policy(tmp_path):
    import shutil
    from test_fs4_threshold import native_bank_and_writer
    from experiments.calibration.fs4_checkpoint import save_checkpoint, restore_checkpoint, begin_recovery, CheckpointError
    from experiments.calibration.fs4_stochastic_bea import action_uniform, candidate_ordinal
    from experiments.calibration.fs4_threshold_integration import NativeRuntime
    from experiments.calibration.native import _fixed_bank_digest

    m = dict(manifest(), expected_events_by_category={"synthetic": 10})
    root = tmp_path / "run"; ledger = DurableLedger(root, m, create=True); ledger.select_category("synthetic")
    def make(ledger):
        native, bank, writer = native_bank_and_writer(4096)
        reference = {"fixed_support_sha256": _fixed_bank_digest(bank), "writer_reference": {"theta": 1.0, "sigma": .02}}
        cfg = SimpleNamespace(topk_patches=32, image_aggregation="max", image_topk=64, image_quantile=.95)
        runtime = NativeRuntime(impl=native, bank=bank, cfg=cfg, writer=writer, correction_cfg=bank.corr_cfg,
                                correction_cb=bank.corr_cb["p0"]["id"][0], reference=reference)
        return SharedTrajectory(runtime=runtime, coverage=CoverageMemory(np.eye(6, dtype=np.float32)[:4]), theta0=1.0,
            policy=StochasticBEAGroupSlot(), seal_prediction=ledger.seal_prediction, seal_query=ledger.seal_query,
            label_provider=lambda event, pos: ledger.disclose(event, pos, lambda *_: 0))
    first = make(ledger)
    features = torch.zeros((784, 4096)); features[:, 1] = 1
    def advance(loop, journal, positions):
        result = []
        for pos in positions:
            row = loop.step(position=pos, group=hid(f"native-{pos // 5}"), event=hid(f"native-event-{pos}"),
                image=SimpleNamespace(height=28, width=28), features=features,
                cls_feature=np.eye(6, dtype=np.float32)[4 + pos // 5], feature_convention=FEATURE_CONVENTION)
            journal.complete_event(position=pos, post_state_digest=prediction_digest(loop.runtime.states()), transition={"mode": row["mode"]})
            result.append((row["sealed_prediction"], row["query"]))
        return result
    advance(first, ledger, range(5))
    assert len(first.E.evidence) == 1 and first.runtime.correction_cb.size() > 0
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, trajectory=first, ledger=ledger)
    ledger.close()
    template_root = tmp_path / "template"
    shutil.copytree(root, template_root)
    copy_root = tmp_path / "copy"
    shutil.copytree(root, copy_root)
    ledger_a = DurableLedger(root, m, create=False); ledger_b = DurableLedger(copy_root, m, create=False)
    first.seal_prediction = ledger_a.seal_prediction; first.seal_query = ledger_a.seal_query
    first.label_provider = lambda event, pos: ledger_a.disclose(event, pos, lambda *_: 0)
    resumed = make(ledger_b)
    restore_checkpoint(checkpoint, trajectory=resumed, ledger=ledger_b)
    assert resumed.runtime.states() == first.runtime.states()
    assert resumed.E.evidence[0].issuer is resumed.policy
    assert torch.equal(resumed.E.cache.records[resumed.E.evidence[0].group].features, features)
    assert advance(first, ledger_a, range(5, 10)) == advance(resumed, ledger_b, range(5, 10))
    assert resumed.runtime.states() == first.runtime.states()
    assert resumed.L.engine.theta == first.L.engine.theta and resumed.E.engine.theta == first.E.engine.theta
    assert ledger_a.offset() == ledger_b.offset()
    ledger_a.close(); ledger_b.close()

    event = hid("crash-event")
    group = next(hid(f"crash-group-{i}") for i in range(10000)
                 if candidate_ordinal(37, hid(f"crash-group-{i}")) == 1 and
                 action_uniform(37, hid(f"crash-group-{i}"), event) < .05)
    args = dict(position=5, group=group, event=event, image=SimpleNamespace(height=28, width=28),
                features=features, cls_feature=np.eye(6, dtype=np.float32)[5], feature_convention=FEATURE_CONVENTION)
    reference_root = tmp_path / "reference"
    shutil.copytree(template_root, reference_root)
    reference_ledger = DurableLedger(reference_root, m, create=False)
    reference_loop = make(reference_ledger); restore_checkpoint(checkpoint, trajectory=reference_loop, ledger=reference_ledger)
    expected_row = reference_loop.step(**args)
    assert expected_row["query"].executed
    reference_ledger.complete_event(position=5, post_state_digest=prediction_digest(reference_loop.runtime.states()),
                                    transition={"mode": expected_row["mode"]})
    expected_states, expected_offset = reference_loop.runtime.states(), reference_ledger.offset()
    reference_ledger.close()

    for boundary in ("prediction", "query", "intent", "feedback", "correction", "greedy", "refresh", "complete"):
        crash_root = tmp_path / f"crash-{boundary}"
        shutil.copytree(template_root, crash_root)
        crash_ledger = DurableLedger(crash_root, m, create=False)
        crashing = make(crash_ledger); restore_checkpoint(checkpoint, trajectory=crashing, ledger=crash_ledger)
        if boundary == "intent":
            def failed_reader(*_args):
                raise RuntimeError("injected crash")
            crashing.label_provider = lambda event_id, pos: crash_ledger.disclose(event_id, pos, failed_reader)
        elif boundary in ("prediction", "query", "feedback"):
            attr = {"prediction": "seal_prediction", "query": "seal_query", "feedback": "label_provider"}[boundary]
            original = getattr(crashing, attr)
            def fail_after(*a, _original=original, **kw):
                _original(*a, **kw)
                raise RuntimeError("injected crash")
            setattr(crashing, attr, fail_after)
        elif boundary in ("correction", "greedy", "refresh"):
            target, attr = ((crashing.runtime.bank, "update_corrections_from_topk") if boundary == "correction" else
                            (crashing.runtime.writer, "update") if boundary == "greedy" else (crashing.E, "refresh"))
            original = getattr(target, attr)
            def fail_after_update(*a, _original=original, **kw):
                _original(*a, **kw)
                raise RuntimeError("injected crash")
            setattr(target, attr, fail_after_update)
        if boundary == "complete":
            row = crashing.step(**args)
            crash_ledger.complete_event(position=5, post_state_digest=prediction_digest(crashing.runtime.states()), transition={"mode": row["mode"]})
        else:
            with pytest.raises(RuntimeError, match="injected crash"):
                crashing.step(**args)
        crash_ledger.close()
        recovered_ledger = DurableLedger(crash_root, m, create=False)
        recovered = make(recovered_ledger); reads = []
        recovery = begin_recovery(checkpoint, trajectory=recovered, ledger=recovered_ledger,
                                  label_reader=lambda *a: (reads.append(a) or 0))
        if boundary == "intent":
            with pytest.raises(CheckpointError, match="ambiguous prior label disclosure"):
                recovered.step(**args)
            assert not reads
            recovered_ledger.close()
            continue
        recovered_row = recovered.step(**args)
        recovery.complete_event(position=5, post_state_digest=prediction_digest(recovered.runtime.states()),
                                transition={"mode": recovered_row["mode"]})
        recovery.assert_caught_up()
        assert recovered_row["sealed_prediction"] == expected_row["sealed_prediction"]
        assert recovered_row["query"] == expected_row["query"]
        assert recovered.runtime.states() == expected_states and recovered_ledger.offset() == expected_offset
        assert len(reads) == (1 if boundary in ("prediction", "query") else 0)
        recovered_ledger.close()

    with (checkpoint / "tensors.npz").open("ab") as stream: stream.write(b"x")
    ledger_c = DurableLedger(root, m, create=False)
    with pytest.raises(CheckpointError, match="checksum"):
        restore_checkpoint(checkpoint, trajectory=make(ledger_c), ledger=ledger_c)
    ledger_c.close()
