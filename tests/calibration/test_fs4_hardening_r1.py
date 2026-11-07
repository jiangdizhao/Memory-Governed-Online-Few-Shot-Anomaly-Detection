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


def manifest(streams=None):
    import subprocess
    from experiments.calibration.fs4_durable_ledger import REQUIRED_SOURCE_PATHS, digest
    if streams is None:
        streams = {"synthetic": [(hid("group"), hid(f"event{i}")) for i in range(5)]}
    fingerprints = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in REQUIRED_SOURCE_PATHS}
    return {"run_id": "synthetic", "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_fingerprints": fingerprints, "scientific_contract_sha256": fingerprints["docs/calibration/fs4_threshold/SCIENTIFIC_CONTRACT.md"],
            "dataset_identity_sha256": hid("synthetic-dataset"), "model_state_sha256": hid("synthetic-model"),
            "support_groups_by_category": {cat: [hid(f"{cat}-support-{i}") for i in range(4)] for cat in streams},
            "fixed_support_sha256_by_category": {cat: hid(f"{cat}-fixed") for cat in streams},
            "stream_sha256_by_category": {cat: digest([{"group": g, "event": e} for g, e in pairs]) for cat, pairs in streams.items()},
            "expected_events_by_category": {cat: len(pairs) for cat, pairs in streams.items()},
            "profile": "SHARED_TRAJECTORY_THRESHOLD_ONLY", "query_seed": 37}


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


def test_manifest_rejects_missing_source_fingerprints(tmp_path):
    m = manifest(); m["source_fingerprints"].pop("experiments/v02_memory.py")
    with pytest.raises(LedgerIntegrityError, match="omits required"):
        DurableLedger(tmp_path / "run", m, create=True)


def test_stage_a_rejects_mismatched_registered_stream(tmp_path):
    from experiments.calibration.fs4_durable_runtime import DurableExecutor
    m = manifest(); ledger = DurableLedger(tmp_path / "run", m, create=True); ledger.select_category("synthetic")
    loop = SharedTrajectory(runtime=_SyntheticRuntime(.5), coverage=CoverageMemory(np.eye(5, dtype=np.float32)[:4]),
        theta0=.5, policy=StochasticBEAGroupSlot(), seal_prediction=prediction_digest,
        seal_query=receipt_digest, label_provider=lambda *_: 0)
    executor = DurableExecutor(loop, ledger, lambda *_: 0)
    features = torch.zeros((784, 4096))
    for pos in range(5):
        executor.step(position=pos, group=hid("wrong-group"), event=hid(f"wrong-{pos}"),
            image=SimpleNamespace(height=28, width=28), features=features,
            cls_feature=np.eye(5, dtype=np.float32)[4], feature_convention=FEATURE_CONVENTION)
    with pytest.raises(LedgerIntegrityError, match="category Stage A"):
        executor.seal_category("synthetic")
    ledger.close()


@pytest.mark.parametrize("label,theta,expected", [(0, .6, "TN"), (0, .4, "FP"), (1, .6, "FN"), (1, .4, "TP")])
def test_native_c1_outcome_uses_frozen_writer_reference(label, theta, expected):
    from test_fs4_threshold import native_bank_and_writer, issue_for_runtime
    from experiments.calibration.fs4_threshold_integration import NativeRuntime
    from experiments.calibration.native import _fixed_bank_digest
    native, bank, writer = native_bank_and_writer(64)
    reference = {"fixed_support_sha256": _fixed_bank_digest(bank), "writer_reference": {"theta": theta, "sigma": .02}}
    cfg = SimpleNamespace(topk_patches=32, image_aggregation="max", image_topk=64, image_quantile=.95)
    runtime = NativeRuntime(impl=native, bank=bank, cfg=cfg, writer=writer, correction_cfg=bank.corr_cfg,
                            correction_cb=bank.corr_cb["p0"]["id"][0], reference=reference)
    policy = StochasticBEAGroupSlot(); evidence, receipt = issue_for_runtime(policy, 0, runtime, label)
    features = torch.zeros((32, 64)); features[:, 1] = 1
    scored = {"G": .5, "GC": .5, "states": runtime.states(), "features": [features], "top_ids": np.arange(32)}
    result = runtime.feedback(receipt=receipt, evidence=evidence, policy=policy, scored=scored)
    assert result["outcome"] == expected and result["learning_reference_threshold"] == theta


def test_durable_executor_journals_complete_transitions(tmp_path):
    from experiments.calibration.fs4_durable_runtime import DurableExecutor
    m = manifest({"synthetic": [(hid("executor-group"), hid(f"executor-{i}")) for i in range(5)]})
    ledger = DurableLedger(tmp_path / "run", m, create=True); ledger.select_category("synthetic")
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


def test_distinct_categories_use_fresh_policies_and_global_seal(tmp_path):
    from experiments.calibration.fs4_durable_runtime import DurableExecutor
    m = manifest({cat: [(hid(f"{cat}-group"), hid(f"{cat}-{pos}")) for pos in range(5 * i, 5 * i + 5)]
                  for i, cat in enumerate(("one", "two"))})
    ledger = DurableLedger(tmp_path / "run", m, create=True)
    features = torch.zeros((784, 4096)); image = SimpleNamespace(height=28, width=28)
    for index, category in enumerate(("one", "two")):
        ledger.select_category(category)
        loop = SharedTrajectory(runtime=_SyntheticRuntime(.5), coverage=CoverageMemory(np.eye(5, dtype=np.float32)[:4]),
            theta0=.5, policy=StochasticBEAGroupSlot(start_position=5 * index), seal_prediction=prediction_digest,
            seal_query=receipt_digest, label_provider=lambda *_: 0)
        executor = DurableExecutor(loop, ledger, lambda *_: 0)
        for pos in range(5 * index, 5 * index + 5):
            executor.step(position=pos, group=hid(f"{category}-group"), event=hid(f"{category}-{pos}"), image=image,
                features=features, cls_feature=np.eye(5, dtype=np.float32)[4], feature_convention=FEATURE_CONVENTION)
        executor.seal_category(category)
    ledger.seal_global_stage_a()
    assert ledger.phase == "GLOBAL_A" and ledger.position == 10
    ledger.close()


def test_ledger_rejects_forged_query_utility(tmp_path):
    from dataclasses import replace
    m = manifest(); ledger = DurableLedger(tmp_path / "run", m, create=True); ledger.select_category("synthetic")
    def forged(receipt):
        return ledger.seal_query(replace(receipt, utility=receipt.utility + .001))
    loop = SharedTrajectory(runtime=_SyntheticRuntime(.5), coverage=CoverageMemory(np.eye(5, dtype=np.float32)[:4]),
        theta0=.5, policy=StochasticBEAGroupSlot(), seal_prediction=ledger.seal_prediction,
        seal_query=forged, label_provider=lambda *_: pytest.fail("unsealed query cannot disclose"))
    with pytest.raises(LedgerIntegrityError, match="query utility"):
        loop.step(position=0, group=hid("forged-group"), event=hid("forged-event"),
            image=SimpleNamespace(height=28, width=28), features=torch.zeros((784, 4096)),
            cls_feature=np.eye(5, dtype=np.float32)[4], feature_convention=FEATURE_CONVENTION)
    assert ledger.phase == "PREDICTION"
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


def test_native_crash_during_real_e_rescoring_replays_checkpoint(tmp_path):
    import shutil
    from test_fs4_threshold import native_bank_and_writer
    from experiments.calibration.fs4_checkpoint import save_checkpoint, restore_checkpoint, begin_recovery
    from experiments.calibration.fs4_durable_runtime import DurableExecutor, post_state_digest
    from experiments.calibration.fs4_stochastic_bea import action_uniform, candidate_ordinal
    from experiments.calibration.fs4_threshold_integration import NativeRuntime
    from experiments.calibration.native import _fixed_bank_digest

    m = dict(manifest(), expected_events_by_category={"synthetic": 60})
    root = tmp_path / "run"; journal = DurableLedger(root, m, create=True); journal.select_category("synthetic")
    def make(journal):
        native, bank, writer = native_bank_and_writer(4096)
        reference = {"fixed_support_sha256": _fixed_bank_digest(bank), "writer_reference": {"theta": 1., "sigma": .02}}
        cfg = SimpleNamespace(topk_patches=32, image_aggregation="max", image_topk=64, image_quantile=.95)
        runtime = NativeRuntime(impl=native, bank=bank, cfg=cfg, writer=writer, correction_cfg=bank.corr_cfg,
                                correction_cb=bank.corr_cb["p0"]["id"][0], reference=reference)
        return SharedTrajectory(runtime=runtime, coverage=CoverageMemory(np.eye(16, dtype=np.float32)[:4]), theta0=1.,
            policy=StochasticBEAGroupSlot(), seal_prediction=journal.seal_prediction, seal_query=journal.seal_query,
            label_provider=lambda event, pos: journal.disclose(event, pos, lambda *_: 0))
    loop = make(journal); executor = DurableExecutor(loop, journal, lambda *_: 0)
    def features_for(index):
        result = torch.zeros((784, 4096)); result[:, index + 1] = 1
        return result
    image = SimpleNamespace(height=28, width=28)
    group_names = []
    for group_index in range(12):
        events = [hid(f"real-e-event-{group_index}-{i}") for i in range(5)]
        group = next(hid(f"real-e-group-{group_index}-{j}") for j in range(20000)
            if (group_index != 11 or candidate_ordinal(37, hid(f"real-e-group-{group_index}-{j}")) == 1) and
            action_uniform(37, hid(f"real-e-group-{group_index}-{j}"),
                           events[candidate_ordinal(37, hid(f"real-e-group-{group_index}-{j}")) - 1]) < .05)
        group_names.append((group, events))
    for group_index, (group, events) in enumerate(group_names[:11]):
        for ordinal, event in enumerate(events):
            executor.step(position=group_index * 5 + ordinal, group=group, event=event, image=image,
                features=features_for(group_index), cls_feature=np.eye(16, dtype=np.float32)[4 + group_index], feature_convention=FEATURE_CONVENTION)
    assert len(loop.E.evidence) == 11 and loop.E.rescore_invocations == 0
    assert all(e.receipt.design_probability == .2 for e in loop.E.evidence)
    checkpoint = tmp_path / "checkpoint"; save_checkpoint(checkpoint, trajectory=loop, ledger=journal); journal.close()
    reference_root = tmp_path / "reference"; shutil.copytree(root, reference_root)
    reference_ledger = DurableLedger(reference_root, m, create=False); reference = make(reference_ledger)
    restore_checkpoint(checkpoint, trajectory=reference, ledger=reference_ledger)
    reference_executor = DurableExecutor(reference, reference_ledger, lambda *_: 0)
    group, events = group_names[11]
    args = dict(position=55, group=group, event=events[0], image=image, features=features_for(11),
                cls_feature=np.eye(16, dtype=np.float32)[15], feature_convention=FEATURE_CONVENTION)
    expected = reference_executor.step(**args)
    assert reference.E.rescore_invocations == 12 and len(reference.E.evidence) == 12
    expected_digest, expected_offset = post_state_digest(reference), reference_ledger.offset()
    reference_ledger.close()

    failed_ledger = DurableLedger(root, m, create=False); failing = make(failed_ledger)
    restore_checkpoint(checkpoint, trajectory=failing, ledger=failed_ledger)
    failed_executor = DurableExecutor(failing, failed_ledger, lambda *_: 0)
    original = failing.E.scorer.score; count = [0]
    def fail_mid_rescore(*a, **kw):
        count[0] += 1
        if count[0] == 6:
            raise FS4NativeScoringIntegrityError("injected mid-rescore crash")
        return original(*a, **kw)
    failing.E.scorer.score = fail_mid_rescore
    before = dict(failing.E.engine.theta)
    with pytest.raises(FS4NativeScoringIntegrityError, match="mid-rescore"):
        failed_executor.step(**args)
    assert count[0] == 6 and failing.E.engine.theta == before
    failed_ledger.close()
    recovered_ledger = DurableLedger(root, m, create=False); recovered = make(recovered_ledger); reads = []
    recovery = begin_recovery(checkpoint, trajectory=recovered, ledger=recovered_ledger,
                              label_reader=lambda *a: (reads.append(a) or 0))
    resumed = DurableExecutor(recovered, recovery).step(**args)
    recovery.assert_caught_up()
    assert not reads and resumed["sealed_prediction"] == expected["sealed_prediction"]
    assert resumed["query"] == expected["query"] and post_state_digest(recovered) == expected_digest
    assert recovered_ledger.offset() == expected_offset
    recovered_ledger.close()
