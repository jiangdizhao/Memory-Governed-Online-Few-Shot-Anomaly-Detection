"""Synthetic-only CUDA engineering checks; no datasets or model weights."""
from __future__ import annotations

import copy
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def make_bank(device, fixed_rows):
    from experiments.calibration.fs4_greedy_provenance import _native_modules
    _native_modules()
    import test_phase3_AIF_simplified_v22 as native
    from v02_memory import MemoryArm
    cfg = native.CorrectionConfig(enabled=False, K_correction=1024)
    bank = native.PatchVMB(device=device, K_normal_fixed=len(fixed_rows), K_normal_dyn=1024,
        K_normal_ltm=1, K_defect=1, K_concept=8, knn_k_normal=5, corr_cfg=cfg, use_defect_bank=False)
    rows = fixed_rows.to(device)
    bank._ensure_mode_view_normals("p0", 0, "id", [rows])
    bank.normal_fixed["p0"][0]["id"][0].set_prototypes(rows)
    writer = MemoryArm("greedy_coreset", bank, SimpleNamespace(flat=1024, stm=1024, ltm=1,
        buffer=1, min_events=2, promote_every=16, cap_bytes=2 * 1024**3), seed=0)
    return native, bank, writer


def test_cuda_native_source_exclusion_and_cpu_gpu_parity():
    from test_fs4_threshold import hid, issue
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration.fs4_stochastic_bea import StochasticBEAGroupSlot
    from experiments.calibration.fs4_greedy_provenance import GreedyProvenance, _native_modules
    from experiments.calibration.fs4_source_excluded_scorer import NativeSourceExcludedScorer, VerifiedNormalFeatureCache
    from experiments.calibration.fs4_threshold_integration import FEATURE_CONVENTION
    seed = torch.zeros((1, 4096)); seed[0, 0] = 1
    native, gpu, writer = make_bank("cuda", seed)
    _, cpu, _ = make_bank("cpu", seed)
    cfg = SimpleNamespace(topk_patches=32, image_aggregation="max", image_topk=64, image_quantile=.95)
    image = SimpleNamespace(height=28, width=28)
    features = torch.zeros((784, 4096)); features[:, 0] = 1
    features[:32, 0] = 0; features[:32, 1] = 1
    features[32:64, 0] = .6; features[32:64, 2] = .8
    _, _, cpu_score, _ = r1._native_dense_score(native, cpu, image, [features], cfg)
    _, _, gpu_score, _ = r1._native_dense_score(native, gpu, image, [features.cuda()], cfg)
    assert gpu_score == pytest.approx(cpu_score, abs=1e-5, rel=1e-4)
    policy = StochasticBEAGroupSlot(); evidence = issue(policy, 0); assert evidence is not None
    sidecar = GreedyProvenance(gpu)
    sidecar.write_normal(writer=writer, impl=native, evidence=evidence, policy=policy,
        features=[features.cuda()], candidates={0: list(range(32))}, score=.5, theta_ref=.4, sigma_ref=.02)
    cb = gpu.corr_cb["p0"]["id"][0]
    cb.n = 3; cb.protos[:3].zero_(); cb.protos[:3, 1] = 1
    cb.protos[2].zero_(); cb.protos[2, 0] = .6; cb.protos[2, 2] = .8
    cb.beta[:3] = .1; cb.radius[:3] = .1; cb.kappa[:3] = 1; cb.count[:3] = 1; cb.last_step[:3] = 0
    other = hid("other-group"); sources = {0: {evidence.group}, 1: {evidence.group, other}, 2: {other}}
    cache = VerifiedNormalFeatureCache()
    record = cache.add(evidence, features, height=28, width=28, normalization=FEATURE_CONVENTION)
    _, ids_before, score_before, _ = r1._native_dense_score(native, gpu, image, [features.cuda()], cfg)
    scorer = NativeSourceExcludedScorer(impl=native, bank=gpu, cfg=cfg, correction_cfg=gpu.corr_cfg,
        correction_cb=cb, correction_sources=sources, provenance=sidecar, normalization=FEATURE_CONVENTION)
    bridge, _ = _native_modules(); before = bridge.snapshot(gpu, version=0).sha256()
    result = scorer.score(record, event_position=policy.last_position + 1)
    assert result["G"] > score_before and set(result["top32_ids"]) != set(ids_before.tolist())
    assert result["excluded_greedy_slots"] == 24
    assert result["correction"]["eligible_anchor_slots"] == 1
    assert result["correction"]["same_source_excluded_anchor_slots"] == 2
    assert bridge.snapshot(gpu, version=0).sha256() == before
    clone = copy.deepcopy(gpu); bridge.dynamic_codebooks(clone)[0].protos = bridge.dynamic_codebooks(clone)[0].protos[:0].clone()
    _, _, fixed_score, _ = r1._native_dense_score(native, clone, image, [features.cuda()], cfg)
    assert result["G"] == pytest.approx(fixed_score, abs=1e-5, rel=1e-4)


def test_cuda_native_8192_prototype_resource_probe():
    from experiments.calibration import r1_matched_correction_pilot as r1
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cpu").manual_seed(19)
    fixed = torch.randn((8192, 4096), generator=generator, dtype=torch.float16)
    native, bank, _ = make_bank("cuda", fixed)
    features = torch.randn((784, 4096), generator=generator).cuda()
    cfg = SimpleNamespace(topk_patches=32, image_aggregation="max", image_topk=64, image_quantile=.95)
    torch.cuda.synchronize(); start = time.perf_counter()
    _, ids, score, _ = r1._native_dense_score(native, bank, SimpleNamespace(height=28, width=28), [features], cfg)
    torch.cuda.synchronize(); seconds = time.perf_counter() - start
    assert len(ids) == 32 and np.isfinite(score)
    assert bank.normal_fixed["p0"][0]["id"][0].protos.shape == (8192, 4096)
    print(f"synthetic_8192_native_score_seconds={seconds:.6f} gpu_peak_bytes={torch.cuda.max_memory_allocated()}")


def test_cuda_native_c1_write_and_greedy_replacement():
    from test_fs4_threshold import issue_for_runtime
    from experiments.calibration.fs4_stochastic_bea import StochasticBEAGroupSlot
    from experiments.calibration.fs4_threshold_integration import NativeRuntime
    from experiments.calibration.native import _fixed_bank_digest
    seed = torch.zeros((1, 4096)); seed[0, 0] = 1
    native, bank, writer = make_bank("cuda", seed)
    cfg = SimpleNamespace(topk_patches=32, image_aggregation="max", image_topk=64, image_quantile=.95)
    ref = {"fixed_support_sha256": _fixed_bank_digest(bank), "writer_reference": {"theta": .4, "sigma": .02}}
    runtime = NativeRuntime(impl=native, bank=bank, cfg=cfg, writer=writer,
        correction_cfg=bank.corr_cfg, correction_cb=bank.corr_cb["p0"]["id"][0], reference=ref)
    policy = StochasticBEAGroupSlot()
    evidence, receipt = issue_for_runtime(policy, 0, runtime, 0)
    generator = torch.Generator(device="cpu").manual_seed(2026)
    features = torch.randn((32, 4096), generator=generator).cuda()
    scored = {"G": .5, "GC": .5, "states": runtime.states(), "features": [features], "top_ids": np.arange(32)}
    runtime.feedback(receipt=receipt, evidence=evidence, policy=policy, scored=scored)
    assert runtime.correction_cb.size() > 0 and runtime.provenance.rows[0]
    for index in range(1, 44):
        evidence, _receipt = issue_for_runtime(policy, index, runtime, 0)
        features = torch.randn((32, 4096), generator=generator).cuda()
        runtime.provenance.write_normal(writer=writer, impl=native, evidence=evidence, policy=policy,
            features=[features], candidates={0: list(range(32))}, score=.5, theta_ref=.4, sigma_ref=.02)
    assert len(runtime.provenance.rows[0]) == 1024
    assert any(len(row["history"]) > 1 for row in runtime.provenance.rows[0])
    runtime.provenance.verify(bank)
