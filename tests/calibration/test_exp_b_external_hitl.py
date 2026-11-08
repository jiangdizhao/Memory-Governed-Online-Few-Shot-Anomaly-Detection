from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.calibration.exp_b_external_hitl import ExternalMemoryError, ExternalNormalMemory


def _unit(rows: int, dim: int, seed: int = 2):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(rows, dim, generator=generator), dim=1)


def _memory_pair(*, patches=8, dim=16, seed=2):
    fixed = _unit(14, dim, seed)
    options = {"category": "synthetic", "experiment_seed": 123,
               "device": "cpu", "_synthetic_test_mode": True,
               "_test_patch_count": patches}
    return (ExternalNormalMemory(fixed, **options),
            ExternalNormalMemory(fixed, **options))


def test_initial_state_scoring_and_c64_novelty_are_label_blind():
    e1, e2 = _memory_pair()
    assert e1.state_sha256 == e2.state_sha256
    c64 = [_unit(8, 16, i) for i in range(64)]
    original_count = e1.count
    receipt1 = e1.calibrate_novelty(c64)
    receipt2 = e2.calibrate_novelty(c64)
    assert e1.count == e2.count == original_count
    assert receipt1["threshold"] == pytest.approx(receipt2["threshold"])
    assert receipt1["c64_patch_distances"] == 64 * 8
    query = _unit(8, 16, 77)
    score = e1.score(query, source_group_sha256="a" * 64, event_position=10)
    assert np.isfinite(score.image_score)
    assert 0 <= score.anomaly_patch < 8
    assert score.eligible_prototypes == 14


def test_fp_gating_source_exclusion_and_deterministic_uniform_subsampling():
    e1, e2 = _memory_pair()
    # Synthetic banks use a deliberately low test cap to force the exact no-replacement path.
    e1.capacity = e2.capacity = 16
    c64 = [torch.eye(16)[0].repeat(8, 1) for _ in range(64)]
    e1.calibrate_novelty(c64)
    e2.calibrate_novelty(c64)
    e1.novelty_threshold = e2.novelty_threshold = 0.0
    e1.novelty_state_sha256 = e1.state_sha256
    e2.novelty_state_sha256 = e2.state_sha256
    query = torch.eye(16)[15].repeat(8, 1)
    before = e1.state_sha256
    tn = e1.false_positive_expansion(query, source_group_sha256="a" * 64,
        event_position=5, queried=True, label=0, arm_prediction=0)
    assert not tn["changed"]
    assert e1.state_sha256 == before

    # The same complete update history must yield identical random choices for both arms.
    fp1 = e1.false_positive_expansion(query, source_group_sha256="a" * 64,
        event_position=5, queried=True, label=0, arm_prediction=1)
    fp2 = e2.false_positive_expansion(query, source_group_sha256="a" * 64,
        event_position=5, queried=True, label=0, arm_prediction=1)
    assert fp1["changed"] and fp2["changed"]
    assert fp1["removed_by_uniform_subsample"] == fp2["removed_by_uniform_subsample"] == 6
    assert e1.state_sha256 == e2.state_sha256
    assert e1.count == e2.count == 16
    assert len(e1.ids) == len(set(e1.ids))
    assert len(e2.ids) == len(set(e2.ids))
    check = e1.score(query, source_group_sha256="a" * 64, event_position=6)
    assert check.source_excluded_prototypes > 0
    assert check.eligible_prototypes < e1.count


def test_fn_surgery_only_removes_nearest_five_and_never_changes_e1():
    e1, e2 = _memory_pair()
    query = torch.eye(16)[0].repeat(8, 1)
    before_e1 = e1.state_sha256
    e1_result = e1.false_negative_surgery(query, anomaly_patch=0,
        source_group_sha256="b" * 64, event_position=20,
        queried=True, label=1, arm_prediction=1)
    assert not e1_result["changed"]
    assert e1.state_sha256 == before_e1

    eligible = e2.eligible_indices(source_group_sha256="b" * 64, event_position=20)
    sims = torch.clamp(query[0] @ e2.features[eligible].T, -1, 1)
    dists = torch.sqrt(torch.clamp(2 - 2 * sims, min=0))
    expected_order = torch.argsort(dists, stable=True)[:5].tolist()
    expected_ids = [e2.ids[eligible[i]] for i in expected_order]
    result = e2.false_negative_surgery(query, anomaly_patch=0,
        source_group_sha256="b" * 64, event_position=20,
        queried=True, label=1, arm_prediction=0)
    assert result["changed"]
    assert result["removed_count"] == 5
    assert set(result["removed_ids"]) == set(expected_ids)
    assert e2.count == 9
    assert len(e2.ids) == len(set(e2.ids))


def test_noop_version_and_unqueried_label_guard():
    e1, _ = _memory_pair()
    before = e1.state_sha256
    with pytest.raises(ExternalMemoryError, match="Unqueried labels"):
        e1.apply_feedback(_unit(8, 16), source_group_sha256="c" * 64,
            event_position=1, queried=False, label=0, arm_prediction=0,
            anomaly_patch=0, enable_fn_surgery=True)
    assert e1.state_sha256 == before
    result = e1.false_negative_surgery(_unit(8, 16), anomaly_patch=0,
        source_group_sha256="c" * 64, event_position=1,
        queried=True, label=1, arm_prediction=1)
    assert not result["changed"]
    assert e1.version == 0
