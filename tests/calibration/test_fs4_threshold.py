from __future__ import annotations

from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.calibration.fs4_stochastic_bea import (
    StochasticBEAGroupSlot, SealedPrediction, action_uniform, bea_components, candidate_ordinal, receipt_digest)
from experiments.calibration.fs4_threshold_core import (
    NormalEvidence, WeightedThresholds, support_self_reference, weighted_higher_quantile, weights)
from experiments.calibration.fs4_threshold_controllers import (
    SelectionAwareHistoricalThreshold, SelectionAwareSourceExcludedThreshold)
from experiments.calibration.fs4_source_excluded_scorer import (
    NativeSourceExcludedScorer, VerifiedNormalFeatureCache)
from experiments.calibration.fs4_greedy_provenance import GreedyProvenance, _native_modules
from experiments.calibration.fs4_threshold_integration import CoverageMemory, FEATURE_CONVENTION, prediction_digest

torch.set_num_threads(1)


def hid(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def issue(policy: StochasticBEAGroupSlot, index: int, *, label: int = 0) -> NormalEvidence | None:
    group = hid(f"group-{index}")
    result = None
    for ordinal in range(5):
        pos = policy.last_position + 1
        event = hid(f"event-{index}-{ordinal}")
        prediction = SealedPrediction(pos, group, event, 0.5, 0.5, hid("G"), hid("GC"), hid("theta"), hid("seal"))
        receipt = policy.decide(prediction, reference_threshold=0.5, novelty_percentile=1.0,
            correction_top32=np.zeros(32), correction_bank_size=0,
            topk_dN_max=1.0, topk_dN_mean=0.5, pre_score_scale=policy.prior_scale())
        receipt = policy.seal_query(receipt, receipt_digest(receipt))
        if receipt.executed:
            receipt = policy.disclose(receipt, label)
            if label == 0:
                result = NormalEvidence.from_disclosure(policy, receipt, normal_version=0, correction_version=0)
    return result


def issue_exploration(policy: StochasticBEAGroupSlot, index: int) -> NormalEvidence:
    group = hid(f"group-{index}")
    candidate = candidate_ordinal(policy.seed, group)
    result = None
    for ordinal in range(1, 6):
        policy.price = 3.0
        pos = policy.last_position + 1
        trial = 0
        event = hid(f"exploration-{index}-{ordinal}-{trial}")
        while ordinal == candidate and action_uniform(policy.seed, group, event) >= 0.05:
            trial += 1
            event = hid(f"exploration-{index}-{ordinal}-{trial}")
        pred = SealedPrediction(pos, group, event, 0.5, 0.5, hid("G"), hid("GC"), hid("theta"), hid("seal"))
        receipt = policy.decide(pred, reference_threshold=0.5, novelty_percentile=1.0,
            correction_top32=np.zeros(32), correction_bank_size=0, topk_dN_max=1.0,
            topk_dN_mean=0.5, pre_score_scale=policy.prior_scale())
        receipt = policy.seal_query(receipt, receipt_digest(receipt))
        if receipt.executed:
            receipt = policy.disclose(receipt, 0)
            result = NormalEvidence.from_disclosure(policy, receipt, normal_version=0, correction_version=0)
    assert result is not None and result.receipt.design_probability == pytest.approx(0.01)
    return result


def issue_for_runtime(policy: StochasticBEAGroupSlot, index: int, runtime, label: int):
    group = hid(f"runtime-group-{index}")
    for ordinal in range(5):
        policy.price = 0.05
        pos = policy.last_position + 1
        event = hid(f"runtime-event-{index}-{ordinal}")
        states = runtime.states()
        pred = SealedPrediction(pos, group, event, 0.5, 0.5, states["G"], states["GC"], hid("theta"), hid("seal"))
        receipt = policy.decide(pred, reference_threshold=0.5, novelty_percentile=1.0,
            correction_top32=np.zeros(32), correction_bank_size=0, topk_dN_max=1.0,
            topk_dN_mean=0.5, pre_score_scale=policy.prior_scale())
        receipt = policy.seal_query(receipt, receipt_digest(receipt))
        if receipt.executed:
            receipt = policy.disclose(receipt, label)
            evidence = (NormalEvidence.from_disclosure(policy, receipt, normal_version=runtime.normal_version,
                        correction_version=runtime.correction_version) if label == 0 else None)
            return evidence, receipt
    raise AssertionError("synthetic exploit query not selected")


def test_native_bea_equations_match_historical_given_identical_scale() -> None:
    from experiments.calibration.e6_bea_policy import compute_bea_components
    # A synthetic sample with ddof=1 standard deviation exactly 0.1.
    synthetic = np.linspace(-1, 1, 64)
    synthetic = synthetic / synthetic.std(ddof=1) * 0.1
    residual = np.linspace(-0.03, 0.04, 32)
    kwargs = dict(score=0.47, threshold=0.5, novelty_percentile=0.8,
                  correction_top32=residual, correction_bank_size=3,
                  topk_dN_max=0.7, topk_dN_mean=0.3)
    old = compute_bea_components(c64_scores=synthetic, **kwargs)
    new = bea_components(scale=0.1, **kwargs)
    for field in ("B", "N", "D"):
        assert new[field] == pytest.approx(getattr(old, field), abs=1e-12)
    assert new["utility"] == pytest.approx(old.B + old.N + old.D, abs=1e-12)


def test_lagged_scale_uses_only_prior_unlabelled_scores_and_bounded_history() -> None:
    policy = StochasticBEAGroupSlot()
    assert policy.prior_scale() == 0.02
    for i, value in enumerate((0.1, 0.2, 0.4)):
        before = policy.prior_scale()
        pred = SealedPrediction(i, hid(f"scale-g-{i}"), hid(f"scale-e-{i}"), value, value,
                                hid("g"), hid("gc"), hid("t"), hid("s"))
        rec = policy.decide(pred, reference_threshold=0.3, novelty_percentile=0.5,
            correction_top32=np.zeros(32), correction_bank_size=0, topk_dN_max=0.3,
            topk_dN_mean=0.2, pre_score_scale=before)
        rec = policy.seal_query(rec, receipt_digest(rec))
        if rec.executed:
            policy.disclose(rec, 1)
    assert policy.prior_scale() == pytest.approx(1.4826 * 0.1)
    assert len(policy.history) == 3


def test_group_slot_probability_and_sealed_query_legality() -> None:
    counts = np.bincount([candidate_ordinal(seed, hid("sample")) for seed in range(5000)], minlength=6)[1:]
    assert np.all(np.abs(counts / counts.sum() - 0.2) < 0.025)
    policy = StochasticBEAGroupSlot()
    evidence = issue(policy, 0)
    assert evidence is not None
    receipt = evidence.receipt
    assert receipt.gate_probability == 1.0 and receipt.design_probability == pytest.approx(0.2)
    assert sum(r.executed for r in policy.receipts.values()) == 1
    with pytest.raises(ValueError):
        policy.disclose(receipt, 0)
    with pytest.raises(ValueError):
        NormalEvidence.from_disclosure(policy, replace(receipt, design_probability=0.0), normal_version=0, correction_version=0)
    # A historical deterministic nonquery has no fabricated positive propensity.
    noncandidate = next(r for r in policy.receipts.values() if r.ordinal != r.candidate)
    assert noncandidate.design_probability is None


def test_low_utility_exploration_probability_and_no_forced_query() -> None:
    policy = StochasticBEAGroupSlot(seed=9)
    group = hid("low-utility")
    seen = None
    for ordinal in range(1, 6):
        pos = ordinal - 1
        prediction = SealedPrediction(pos, group, hid(str(pos)), 10.0, 10.0,
            hid("g"), hid("gc"), hid("t"), hid("sealed"))
        receipt = policy.decide(prediction, reference_threshold=0.0, novelty_percentile=0,
            correction_top32=np.zeros(32), correction_bank_size=0, topk_dN_max=0.1,
            topk_dN_mean=0.1, pre_score_scale=policy.prior_scale())
        if receipt.ordinal == receipt.candidate:
            seen = receipt
        receipt = policy.seal_query(receipt, receipt_digest(receipt))
        if not receipt.executed:
            with pytest.raises(ValueError):
                policy.disclose(receipt, 0)
        else:
            policy.disclose(receipt, 1)
    assert seen is not None and not seen.exploitation
    assert seen.gate_probability == 0.05 and seen.design_probability == pytest.approx(0.01)
    assert sum(policy.flags) == sum(r.executed for r in policy.receipts.values())
    with pytest.raises(ValueError, match="more than five"):
        p = SealedPrediction(5, group, hid("six"), 0, 0, hid("g"), hid("gc"), hid("t"), hid("s"))
        policy.decide(p, reference_threshold=0, novelty_percentile=0, correction_top32=np.zeros(32),
                      correction_bank_size=0, topk_dN_max=0, topk_dN_mean=0, pre_score_scale=policy.prior_scale())


def test_weighted_quantile_ties_clipping_ess_and_initialization() -> None:
    assert weighted_higher_quantile([0, 1, 2, 3], [1, 1, 1, 1], .5) == 1
    assert weighted_higher_quantile([0, 1, 1, 2], [1, 1, 4, 1], .8) == 1
    raw, bounded, ess = weights([0.2, 0.01])
    assert raw.tolist() == [5, 100] and bounded.tolist() == [5, 50]
    assert ess == pytest.approx(55**2 / (5**2 + 50**2))
    ref = support_self_reference([1, 2, 3, 4])
    assert ref["theta"] == pytest.approx(2.5 + 2.5 * np.std([1, 2, 3, 4], ddof=1))
    assert ref["status"] == "COLD_START_UNCALIBRATED"
    with pytest.raises(ValueError):
        weights([0.0])


def test_synthetic_selection_bias_reports_stochastic_variability() -> None:
    # Independent Bernoulli draws across repeated populations; no hidden labels
    # enter the operational policy. This is a statistical diagnostic only.
    population = np.linspace(0, 1, 1000)
    selection = np.where(population > 0.6, 0.8, 0.08)
    reference = weighted_higher_quantile(population, np.ones(1000))
    unweighted, weighted, effective = [], [], []
    for seed in range(80):
        chosen = np.random.default_rng(seed).random(len(population)) < selection
        raw, bounded, ess = weights(selection[chosen])
        unweighted.append(weighted_higher_quantile(population[chosen], np.ones(chosen.sum())))
        weighted.append(weighted_higher_quantile(population[chosen], bounded))
        effective.append(ess)
    assert np.std(weighted) > 0 and min(effective) > 12
    assert abs(np.mean(weighted) - reference) < abs(np.mean(unweighted) - reference)


def test_historical_and_current_use_the_same_estimator_and_version_gates() -> None:
    policy = StochasticBEAGroupSlot(seed=37)
    records = [issue(policy, i) for i in range(12)]
    assert all(records)
    class Scorer:
        calls = 0
        def score(self, record, *, event_position):
            self.calls += 1
            return dict(record.evidence.pre_scores)
    scorer = Scorer()
    left = SelectionAwareHistoricalThreshold(0.4)
    right = SelectionAwareSourceExcludedThreshold(0.4, scorer)
    features = torch.zeros((784, 4096), dtype=torch.float32)
    for evidence in records:
        left.add(evidence)
        right.add(evidence, features, height=28, width=28, normalization=FEATURE_CONVENTION)
    result = right.refresh(state_g="g0", state_gc="gc0", event_position=100)
    assert left.engine.theta == right.engine.theta == {"G": pytest.approx(0.41), "GC": pytest.approx(0.41)}
    assert result["G"]["status"] == result["GC"]["status"] == "UPDATE_ACCEPTED"
    calls = scorer.calls
    right.refresh(state_g="g0", state_gc="gc0", event_position=101)
    assert scorer.calls == calls and right.engine.theta["G"] == pytest.approx(0.41)
    right.refresh(state_g="g0", state_gc="gc1", event_position=102)
    assert right.engine.theta["G"] == pytest.approx(0.41)
    assert right.engine.theta["GC"] == pytest.approx(0.42)
    right.refresh(state_g="g1", state_gc="gc2", event_position=103)
    assert right.engine.theta["G"] == pytest.approx(0.42)
    assert right.engine.theta["GC"] == pytest.approx(0.43)
    assert left.observe_state_change() == {"G": "THRESHOLD_HELD", "GC": "THRESHOLD_HELD"}
    detail = right.diagnostics(policy, current_position=103,
        current_scorer_states={"G": "g1", "GC": "gc2"})
    assert detail["distinct_normal_groups"] == 12
    assert detail["historical_stale_counts"] == {"G": 12, "GC": 12}
    assert sum(x["observed"] for x in detail["score_region_coverage"]["G"].values()) == len(policy.receipts)
    assert detail["cache"]["count"] == 12


def test_insufficient_and_low_ess_hold_without_rescore() -> None:
    policy = StochasticBEAGroupSlot()
    records = []
    for i in range(11):
        policy.price = 0.05
        records.append(issue(policy, i))
    engine = WeightedThresholds(0.4)
    assert engine.update("G", records[:11], [0.5] * 11, 1)["status"] == "INSUFFICIENT_VERIFIED_NORMALS"
    explored = issue_exploration(policy, 11)
    result = engine.update("G", [explored, *records], [0.5] * 12, 2)
    assert result["status"] == "LOW_EFFECTIVE_SAMPLE_SIZE" and engine.theta["G"] == 0.4
    forged = replace(records[0], receipt=replace(records[0].receipt, design_probability=0.01))
    assert engine.update("G", [forged, *records[1:], explored], [0.5] * 12, 3)["status"] == "SOURCE_PROVENANCE_INCOMPLETE"
    class FakeCache:
        def __init__(self): self.records = {}
        def add(self, e, *_args, **_kwargs): self.records[e.group] = e
    class NoRescore:
        def score(self, *_args, **_kwargs): raise AssertionError("low ESS must hold before rescoring")
    current = SelectionAwareSourceExcludedThreshold(0.4, NoRescore(), FakeCache())
    for e in [explored, *records]:
        current.add(e, None, height=28, width=28, normalization=FEATURE_CONVENTION)
    held = current.refresh(state_g="g", state_gc="gc", event_position=100)
    assert held["G"]["status"] == held["GC"]["status"] == "LOW_EFFECTIVE_SAMPLE_SIZE"
    assert current.engine.theta == {"G": 0.4, "GC": 0.4}


def test_native_coverage_geometry_matches_historical_e6() -> None:
    from experiments.calibration.e6_bea_closed_loop import NormalCoverageMemory
    support = np.eye(4, dtype=np.float32)
    ours, old = CoverageMemory(support), NormalCoverageMemory(support)
    for x in (np.array([.2, .3, .4, .5], dtype=np.float32), np.array([.5, .1, .2, .3], dtype=np.float32)):
        a, dist = ours.novelty(x)
        b, info = old.novelty_percentile(x)
        assert a == b and dist == pytest.approx(info["distance_nn"])
        ours.observe_normal(x); old.observe_normal(x)


def test_cpu_five_view_end_to_end_smoke() -> None:
    from experiments.calibration.fs4_threshold_synthetic_smoke import run_smoke
    result = run_smoke()
    assert result["status"] == "PASS_SYNTHETIC_ONLY"
    assert result["verified_normals"] >= 12 and result["verified_defects"] >= 1
    assert result["e_rescore_calls"] > 0 and not result["real_data_accessed"]
    assert result["post_seal_synthetic_outcomes_checked"] == result["events"]


def test_native_cpu_five_view_end_to_end_smoke() -> None:
    from experiments.calibration.fs4_threshold_synthetic_smoke import run_native_smoke
    result = run_native_smoke()
    assert result["status"] == "PASS_SYNTHETIC_NATIVE_CPU"
    assert result["verified_normals"] >= 12 and result["verified_defects"] > 0
    assert result["greedy_slots"] > 0 and result["correction_slots"] > 0
    assert result["e_rescore_calls"] > 0 and not result["real_data_accessed"]
    assert result["post_seal_synthetic_outcomes_checked"] == result["events"]


def test_cache_rejects_unverified_half_precision_conversion() -> None:
    policy = StochasticBEAGroupSlot()
    evidence = issue(policy, 0)
    cache = VerifiedNormalFeatureCache()
    with pytest.raises(ValueError, match="complete finite fused"):
        cache.add(evidence, torch.zeros((784, 4096), dtype=torch.float16),
                  height=28, width=28, normalization=FEATURE_CONVENTION)
    assert cache.bytes_report()["actual_tensor_bytes"] == 0


def native_bank_and_writer(dim: int):
    _native_modules()
    import test_phase3_AIF_simplified_v22 as native
    from v02_memory import MemoryArm
    cfg = native.CorrectionConfig(enabled=False, K_correction=1024)
    bank = native.PatchVMB(device="cpu", K_normal_fixed=8, K_normal_dyn=1024,
        K_normal_ltm=1, K_defect=1, K_concept=8, knn_k_normal=5,
        corr_cfg=cfg, use_defect_bank=False)
    seed = torch.zeros((1, dim), dtype=torch.float32)
    seed[0, 0] = 1
    bank._ensure_mode_view_normals("p0", 0, "id", [seed])
    bank.normal_fixed["p0"][0]["id"][0].set_prototypes(seed)
    writer = MemoryArm("greedy_coreset", bank,
        SimpleNamespace(flat=1024, stm=1024, ltm=1, buffer=1,
                        min_events=2, promote_every=16, cap_bytes=2 * 1024**3), seed=0)
    return native, bank, writer


def test_native_greedy_append_replacement_and_conservative_provenance() -> None:
    native, bank, writer = native_bank_and_writer(64)
    sidecar = GreedyProvenance(bank)
    policy = StochasticBEAGroupSlot()
    generator = torch.Generator().manual_seed(314)
    append_count = 0
    for i in range(43):
        policy.price = 0.05  # fixture makes the stochastic gate deterministically exploit
        evidence = issue(policy, i)
        assert evidence is not None
        features = torch.randn((24, 64), generator=generator)
        result = sidecar.write_normal(writer=writer, impl=native, evidence=evidence, policy=policy,
            features=[features], candidates={0: list(range(24))}, score=0.5,
            theta_ref=0.4, sigma_ref=0.02)
        append_count += result["accepted"]
    assert len(sidecar.rows[0]) == 1024 and append_count >= 1024
    replaced = [row for row in sidecar.rows[0] if len(row["history"]) > 1]
    assert replaced
    row = replaced[0]
    prior, latest = row["history"][0]["group"], row["history"][-1]["group"]
    assert prior in row["groups"] and latest in row["groups"]
    slot = sidecar.rows[0].index(row)
    assert slot not in sidecar.eligible_indices(bank, prior)[0]
    assert slot not in sidecar.eligible_indices(bank, latest)[0]
    sidecar.verify(bank)


def test_native_full_feature_source_exclusion_reselects_top32_and_preserves_live_state() -> None:
    native, bank, writer = native_bank_and_writer(4096)
    from experiments.calibration import r1_matched_correction_pilot as r1
    bridge, _ = _native_modules()
    sidecar = GreedyProvenance(bank)
    policy = StochasticBEAGroupSlot()
    evidence = issue(policy, 0)
    assert evidence is not None
    features = torch.zeros((784, 4096), dtype=torch.float32)
    features[:, 0] = 1
    features[:32, 0] = 0; features[:32, 1] = 1
    features[32:64, 0] = 0.6; features[32:64, 2] = 0.8
    result = sidecar.write_normal(writer=writer, impl=native, evidence=evidence, policy=policy,
        features=[features], candidates={0: list(range(32))}, score=0.5,
        theta_ref=0.4, sigma_ref=0.02)
    assert result["accepted"] == 24
    cb = bank.corr_cb["p0"]["id"][0]
    cb.n = 3
    cb.protos[:3].zero_(); cb.protos[:3, 1] = 1
    cb.protos[2].zero_(); cb.protos[2, 0] = 0.6; cb.protos[2, 2] = 0.8
    cb.beta[:3] = 0.1; cb.radius[:3] = 0.1; cb.kappa[:3] = 1
    cb.count[:3] = 1; cb.last_step[:3] = 0
    other = hid("other-group")
    sources = {0: {evidence.group}, 1: {evidence.group, other}, 2: {other}}
    cache = VerifiedNormalFeatureCache()
    record = cache.add(evidence, features, height=28, width=28, normalization=FEATURE_CONVENTION)
    cfg = SimpleNamespace(topk_patches=32, image_aggregation="max", image_topk=64, image_quantile=0.95)
    image = SimpleNamespace(height=28, width=28)
    _, original_ids, original_score, _ = r1._native_dense_score(native, bank, image, [features], cfg)
    before = bridge.snapshot(bank, version=0).sha256()
    correction_before = r1._correction_state_sha256(bank, sources)
    source_before = sidecar.digest()
    scorer = NativeSourceExcludedScorer(impl=native, bank=bank, cfg=cfg,
        correction_cfg=bank.corr_cfg, correction_cb=cb, correction_sources=sources,
        provenance=sidecar, normalization=FEATURE_CONVENTION)
    scored = scorer.score(record, event_position=policy.last_position + 1)
    assert scored["G"] > original_score
    assert set(scored["top32_ids"]) != set(original_ids.tolist())
    assert scored["excluded_greedy_slots"] == 24
    assert scored["correction"]["eligible_anchor_slots"] == 1
    assert scored["correction"]["same_source_excluded_anchor_slots"] == 2
    import copy
    fixed_only = copy.deepcopy(bank)
    bridge.dynamic_codebooks(fixed_only)[0].protos = bridge.dynamic_codebooks(fixed_only)[0].protos[:0].clone()
    _, _, static_score, _ = r1._native_dense_score(native, fixed_only, image, [features], cfg)
    assert scored["G"] == static_score
    assert bridge.snapshot(bank, version=0).sha256() == before
    assert r1._correction_state_sha256(bank, sources) == correction_before
    assert sidecar.digest() == source_before
    assert cache.bytes_report()["actual_tensor_bytes"] == features.numel() * features.element_size()
    other_evidence = issue(policy, 1)
    assert other_evidence is not None
    other_record = cache.add(other_evidence, features, height=28, width=28, normalization=FEATURE_CONVENTION)
    unexcluded = scorer.score(other_record, event_position=policy.last_position + 1)
    direct_bank = copy.deepcopy(bank)
    direct_bank.corr_cfg.enabled = True
    maps, _ = direct_bank.score_single_view("p0", [features], mode=0, view="id",
        topk_patches=32, mode_scoring="boost", lam=0.8, tau_close=0.2, gamma=1.0)
    assert unexcluded["GC"] == pytest.approx(float(maps[0].max()), abs=1e-5, rel=1e-4)
    assert unexcluded["correction"]["eligible_anchor_slots"] == 3
    record.features[0, 0] = 0.25
    with pytest.raises(ValueError, match="feature convention mismatch"):
        scorer.score(record, event_position=policy.last_position + 1)


def test_native_feedback_updates_c1_for_both_labels_and_greedy_only_for_normal() -> None:
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.calibration.fs4_threshold_integration import NativeRuntime
    native, bank, writer = native_bank_and_writer(64)
    cb = bank.corr_cb["p0"]["id"][0]
    reference = {"fixed_support_sha256": _fixed_bank_digest(bank),
                 "writer_reference": {"theta": 0.4, "sigma": 0.02}}
    runtime = NativeRuntime(impl=native, bank=bank,
        cfg=SimpleNamespace(topk_patches=32, image_aggregation="max", image_topk=64, image_quantile=0.95),
        writer=writer, correction_cfg=bank.corr_cfg, correction_cb=cb, reference=reference)
    policy = StochasticBEAGroupSlot()
    normal, normal_receipt = issue_for_runtime(policy, 0, runtime, 0)
    assert normal is not None
    features = torch.zeros((32, 64)); features[:, 1] = 1
    scored = {"G": 0.5, "GC": 0.5, "states": runtime.states(), "features": [features], "top_ids": np.arange(32)}
    normal_update = runtime.feedback(receipt=normal_receipt, evidence=normal, policy=policy, scored=scored)
    assert normal_update["greedy"]["accepted"] == 24
    assert cb.size() > 0 and runtime.normal_version == 1 and runtime.correction_version == 1
    normal_digest = runtime.provenance.memory_digest
    _, defect = issue_for_runtime(policy, 1, runtime, 1)
    other_features = torch.zeros((32, 64)); other_features[:, 2] = 1
    scored = {"G": 0.5, "GC": 0.5, "states": runtime.states(), "features": [other_features], "top_ids": np.arange(32)}
    defect_update = runtime.feedback(receipt=defect, evidence=None, policy=policy, scored=scored)
    assert defect_update["greedy"] is None and runtime.provenance.memory_digest == normal_digest
    assert runtime.normal_version == 1 and runtime.correction_version == 2
    assert any(defect.prediction.group in groups for groups in runtime.sources.values())


def test_sidecar_rejects_unexplained_mutation_and_cache_overflow() -> None:
    native, bank, writer = native_bank_and_writer(64)
    sidecar = GreedyProvenance(bank)
    bridge, _ = _native_modules()
    cb = bridge.dynamic_codebooks(bank)[0]
    cb.protos = torch.ones((1, 64), dtype=cb.dtype)
    with pytest.raises(ValueError, match="SOURCE_PROVENANCE_INCOMPLETE"):
        sidecar.verify(bank)
    policy = StochasticBEAGroupSlot()
    evidence = issue(policy, 0)
    cache = VerifiedNormalFeatureCache()
    cache.records = {str(i): None for i in range(1024)}
    with pytest.raises(OverflowError):
        cache.add(evidence, torch.zeros((784, 4096), dtype=torch.float32),
                  height=28, width=28, normalization=FEATURE_CONVENTION)


def test_native_support_self_initializer_and_empty_c1_agree() -> None:
    from experiments.calibration.fs4_source_excluded_scorer import initialize_native_support
    from experiments.calibration.fs4_threshold_integration import NativeRuntime
    from experiments.calibration import r1_matched_correction_pilot as r1
    native, bank, writer = native_bank_and_writer(64)
    cfg = SimpleNamespace(topk_patches=32, image_aggregation="max", image_topk=64, image_quantile=0.95)
    image = SimpleNamespace(height=28, width=28)
    vectors = []
    for i in range(4):
        feature = torch.zeros((784, 64)); feature[:, 0] = 1.0 - i * 0.1
        feature[:, i + 1] = 0.2 + i * 0.1
        vectors.append((hid(f"support-{i}"), image, feature))
    ref = initialize_native_support(impl=native, bank=bank, cfg=cfg, support=vectors,
                                    correction_cb=bank.corr_cb["p0"]["id"][0])
    direct = [r1._native_dense_score(native, bank, image, [x], cfg)[2] for _, _, x in vectors]
    assert ref["support_scores"] == direct and ref["theta_g"] == ref["theta_gc"]
    assert ref["theta"] == pytest.approx(support_self_reference(direct)["theta"])
    runtime = NativeRuntime(impl=native, bank=bank, cfg=cfg, writer=writer,
        correction_cfg=bank.corr_cfg, correction_cb=bank.corr_cb["p0"]["id"][0], reference=ref)
    score = runtime.score(group=hid("new-group"), position=0, image=image, features=vectors[0][2])
    assert score["G"] == score["GC"] and np.array_equal(score["correction_top32"], np.zeros(32))


def test_query_price_tracks_only_realized_actions_at_requested_target() -> None:
    from collections import deque
    policy = StochasticBEAGroupSlot()
    for i in range(8):
        issue(policy, i)
    price, flags = 0.80, deque(maxlen=256)
    for receipt in policy.receipts.values():
        assert receipt.cost == pytest.approx(price)
        flags.append(int(receipt.executed))
        price = min(3.0, max(0.05, price + 0.02 * (sum(flags) / len(flags) - 0.037)))
    assert policy.price == pytest.approx(price)


def test_shared_query_reference_is_independent_and_coupled_instances_are_isolated() -> None:
    from experiments.calibration.fs4_threshold_integration import (
        CoupledTrajectory, SharedTrajectory, require_independent_coupled_instances)
    from experiments.calibration.fs4_threshold_synthetic_smoke import _SyntheticRuntime
    def make_shared():
        return SharedTrajectory(runtime=_SyntheticRuntime(0.5), coverage=CoverageMemory(np.eye(4)),
            theta0=0.5, policy=StochasticBEAGroupSlot(),
            seal_prediction=prediction_digest, seal_query=receipt_digest,
            label_provider=lambda *_: 0)
    a, b = make_shared(), make_shared()
    b.L.engine.theta["GC"] = 100.0
    b.E.engine.theta["GC"] = -100.0
    args = dict(position=0, group=hid("shared-group"), event=hid("shared-event"),
                image=SimpleNamespace(height=28, width=28),
                features=torch.zeros((784, 4096), dtype=torch.float32), cls_feature=np.eye(4)[0],
                feature_convention=FEATURE_CONVENTION)
    x, y = a.step(**args), b.step(**args)
    assert x["query"].utility == y["query"].utility
    assert x["query"].executed == y["query"].executed
    assert x["sealed_prediction"]["scores"] == y["sealed_prediction"]["scores"]
    def make_coupled(kind):
        return CoupledTrajectory(controller=kind, runtime=_SyntheticRuntime(0.5),
            coverage=CoverageMemory(np.eye(4)), theta0=0.5, policy=StochasticBEAGroupSlot(),
            seal_prediction=prediction_digest, seal_query=receipt_digest,
            label_provider=lambda *_: 0)
    left, right = make_coupled("L"), make_coupled("E")
    require_independent_coupled_instances(left, right)
    assert not hasattr(left.controller, "cache") and hasattr(right.controller, "cache")
    left.controller.engine.theta["GC"] = 0.7
    right.controller.engine.theta["GC"] = 0.3
    assert left.step(**args)["query"].reference_threshold == 0.7
    assert right.step(**args)["query"].reference_threshold == 0.3
    with pytest.raises(ValueError, match="independent"):
        require_independent_coupled_instances(left, left)


def test_query_cannot_advance_before_prior_receipt_is_sealed() -> None:
    policy = StochasticBEAGroupSlot()
    assert policy.diagnostics()["realized_rate"] is None
    assert policy.diagnostics()["exploration_fraction"] is None
    with pytest.raises(ValueError, match="five views"):
        policy.validate_complete_stream()
    p0 = SealedPrediction(0, hid("g0"), hid("e0"), 0.5, 0.5, hid("g"), hid("gc"), hid("t"), hid("s"))
    policy.decide(p0, reference_threshold=0.5, novelty_percentile=1,
        correction_top32=np.zeros(32), correction_bank_size=0, topk_dN_max=1,
        topk_dN_mean=0.5, pre_score_scale=policy.prior_scale())
    with pytest.raises(ValueError, match="five views"):
        policy.validate_complete_stream()
    p1 = SealedPrediction(1, hid("g1"), hid("e1"), 0.5, 0.5, hid("g"), hid("gc"), hid("t"), hid("s"))
    with pytest.raises(ValueError, match="previous query"):
        policy.decide(p1, reference_threshold=0.5, novelty_percentile=1,
            correction_top32=np.zeros(32), correction_bank_size=0, topk_dN_max=1,
            topk_dN_mean=0.5, pre_score_scale=policy.prior_scale())


def test_native_bridge_direct_shadow_score_and_writer_parity_on_cpu() -> None:
    from experiments.calibration import r1_matched_correction_pilot as r1
    native, bank, writer = native_bank_and_writer(64)
    bridge, _ = _native_modules()
    features = torch.randn((784, 64), generator=torch.Generator().manual_seed(17))
    bank._init_concepts_if_needed("p0", [features])
    bank.concept_cb["p0"][0].update(features[:8])
    image = SimpleNamespace(height=28, width=28)
    cfg = SimpleNamespace(topk_patches=32, image_aggregation="max", image_topk=64, image_quantile=0.95)
    dense, ids, score, _ = r1._native_dense_score(native, bank, image, [features], cfg)
    direct, amap = bridge.native_score(native, bank, image, [features], cfg)
    assert direct == score == float(dense.max())
    assert bridge.verify_untouched_parity(native, bank, image, [features], cfg)["passed"] is True
    receipt = bridge.verify_native_update_parity(native, writer, [features], amap,
        score=0.5, theta=0.4, sigma=0.02, step=0)
    assert receipt["passed"] and all(receipt["equal_each_layer"])
    assert receipt["greedy"]["appended"] == 24


def test_four_support_builder_rejects_extra_reference_and_wires_native_components(monkeypatch) -> None:
    from experiments import p0_protocol_audit as p0
    from experiments.calibration.fs4_threshold_integration import build_native_from_four_support
    from experiments.calibration.profiles import NativeScorerConfig
    _native_modules()
    import test_phase3_AIF_simplified_v22 as native
    params = NativeScorerConfig()
    cfg = p0.Config(device="cpu", augmentation_seed=params.augmentation_seed,
        bank_seed=params.bank_seed, fixed_capacity=params.fixed_capacity,
        fixed_per_image=params.fixed_per_image, augmentations=params.augmentations,
        include_identity=params.include_identity, multiplier=params.multiplier,
        feature_fuse=params.feature_fuse, topk_patches=params.topk_patches,
        image_aggregation=params.image_aggregation, image_topk=params.image_topk,
        image_quantile=params.image_quantile, knn=params.knn)
    bank = native.PatchVMB(device="cpu", K_normal_fixed=8, K_normal_dyn=1024,
        K_normal_ltm=1, K_defect=1, K_concept=8, knn_k_normal=5,
        corr_cfg=native.CorrectionConfig(enabled=False, K_correction=256), use_defect_bank=False)
    seed = torch.zeros((1, 4096), dtype=torch.float32); seed[0, 0] = 1
    bank._ensure_mode_view_normals("p0", 0, "id", [seed])
    bank.normal_fixed["p0"][0]["id"][0].set_prototypes(seed)
    calls = []
    def synthetic_builder(_impl, cache, paths, _cfg):
        calls.append((set(cache), tuple(paths)))
        return bank
    monkeypatch.setattr(p0, "build_vmb", synthetic_builder)
    monkeypatch.setattr(native, "set_seed", lambda seed: torch.manual_seed(seed))
    paths = [f"synthetic-support-{i}" for i in range(4)]
    groups = [hid(p) for p in paths]
    augmented = torch.zeros((784, 4096), dtype=torch.float16); augmented[:, 0] = 1
    cache = {p: [[augmented] for _ in range(4)] for p in paths}
    identity = torch.zeros((784, 4096), dtype=torch.float32); identity[:, 0] = 1
    image = SimpleNamespace(height=28, width=28)
    runtime, ref, coverage = build_native_from_four_support(impl=native, cfg=cfg,
        support_paths=paths, support_groups=groups, augmentation_cache=cache,
        identity_images_and_features=[(image, identity)] * 4, support_cls=np.eye(4),
        feature_convention=FEATURE_CONVENTION)
    assert calls == [(set(paths), tuple(paths))]
    assert ref["initial_c1_empty"] and ref["theta_g"] == ref["theta_gc"]
    assert runtime.writer.settings.flat == 1024 and runtime.correction_cb.K == 1024
    assert coverage.support.shape[0] == 4
    with pytest.raises(ValueError, match="exactly four"):
        build_native_from_four_support(impl=native, cfg=cfg, support_paths=paths + ["extra"],
            support_groups=groups + [hid("extra")], augmentation_cache={**cache, "extra": [[augmented]] * 4},
            identity_images_and_features=[(image, identity)] * 5, support_cls=np.eye(5),
            feature_convention=FEATURE_CONVENTION)
    assert len(calls) == 1
