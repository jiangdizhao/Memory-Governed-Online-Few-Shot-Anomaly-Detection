import math
from dataclasses import replace

import numpy as np
import pytest

from experiments.calibration.e6_bea_policy import (
    ARM_POLICIES,
    P_RANDOM_FIVE_VIEW,
    QUERY_COST_MAX,
    QUERY_COST_MIN,
    QUERY_COST_WINDOW,
    QueryController,
    compute_bea_components,
    local_saliency_from_native_stats,
)


def _components(*, score=0.3, threshold=0.4, c64=None, novelty=0.5, correction=None,
                bank_size=1, dmax=0.8, dmean=0.2):
    return compute_bea_components(
        score=score, threshold=threshold,
        c64_scores=np.asarray(c64 if c64 is not None else np.linspace(0.1, 0.4, 64)),
        novelty_percentile=novelty,
        correction_top32=np.asarray(correction if correction is not None else np.zeros(32)),
        correction_bank_size=bank_size, topk_dN_max=dmax, topk_dN_mean=dmean)


def _event_hash(index):
    import hashlib
    return hashlib.sha256(f"event-{index}".encode()).hexdigest()


def _manual_reference(score, threshold, c64, novelty_pct, corrections, dmax, dmean):
    sigma = max(float(np.std(c64, ddof=1)), 0.02)
    margin = score - threshold
    m = margin / sigma
    p = 1.0 / (1.0 + math.exp(-m))
    psi = math.exp(-0.5 * m * m)
    p_clip = min(max(p, 1e-8), 1.0 - 1e-8)
    entropy = -(p_clip * math.log(p_clip) + (1 - p_clip) * math.log(1 - p_clip))
    novelty = min(1.0, max(0.0, (novelty_pct - 0.05) / 0.90))
    corr_def = max(0.0, float(np.max(corrections)))
    corr_norm = max(0.0, -float(np.min(corrections)))
    corr_mag = max(corr_def, corr_norm)
    evidence = min(1.0, max(0.0, corr_mag / 0.05))
    saliency = min(1.0, max(0.0, (dmax - dmean) / max(abs(dmax), abs(dmean), 1e-6)))
    local_multiplier = 1 + max(0.0, saliency - 0.5)
    near, far, tau = max(0.005, 0.5 * sigma), max(0.02, 3 * sigma), max(0.005, 0.5 * sigma)
    logistic = lambda z: 1 / (1 + math.exp(-z))
    gate_far = logistic((margin + far) / tau)
    gate_near = logistic((-margin - near) / tau)
    weak_corr = 1 - evidence
    pred_defect = score >= threshold
    risk = (1 - p) if pred_defect else 2.5 * p
    u_n = psi * novelty
    u_c = psi * evidence * local_multiplier
    u_d = 0 if pred_defect else p * gate_far * gate_near * weak_corr
    return {
        "sigma": sigma, "p": p, "psi": psi, "entropy": entropy,
        "pred_defect": pred_defect, "novelty": novelty, "corr_def": corr_def,
        "corr_norm": corr_norm, "corr_mag": corr_mag, "evidence": evidence,
        "saliency": saliency, "saliency_multiplier": local_multiplier,
        "gate_far": gate_far, "gate_near": gate_near, "weak_corr": weak_corr,
        "R": risk, "H": entropy, "U_N": u_n, "U_C": u_c, "U_D": u_d,
        "B": risk + 0.30 * entropy, "N": 0.35 * u_n,
        "C": 0.15 * u_c, "D": 0.35 * u_d,
    }


def test_exact_bea_decomposition_matches_independent_golden_calculation():
    c64 = np.linspace(0.05, 0.55, 64)
    corrections = np.zeros(32, dtype=np.float64)
    corrections[3], corrections[17] = 0.018, -0.031
    expected = _manual_reference(0.29, 0.4, c64, 0.50, corrections, 0.8, 0.2)
    actual = compute_bea_components(
        score=0.29, threshold=0.4, c64_scores=c64, novelty_percentile=0.50,
        correction_top32=corrections, correction_bank_size=5,
        topk_dN_max=0.8, topk_dN_mean=0.2)
    mapping = {
        "sigma": actual.sigma, "p": actual.p_defect, "psi": actual.psi,
        "entropy": actual.entropy, "pred_defect": actual.pred_defect,
        "novelty": actual.novelty, "corr_def": actual.corr_def,
        "corr_norm": actual.corr_norm, "corr_mag": actual.corr_mag,
        "evidence": actual.correction_evidence,
        "saliency": actual.local_saliency,
        "saliency_multiplier": actual.saliency_multiplier,
        "gate_far": actual.gate_far, "gate_near": actual.gate_near,
        "weak_corr": actual.weak_correction, "R": actual.R, "H": actual.H,
        "U_N": actual.U_N, "U_C": actual.U_C, "U_D": actual.U_D,
        "B": actual.B, "N": actual.N, "C": actual.C, "D": actual.D,
    }
    for key, value in mapping.items():
        if isinstance(value, bool):
            assert value is expected[key]
        else:
            assert value == pytest.approx(expected[key], abs=1e-12)
    assert actual.margin_z == pytest.approx((0.29 - 0.4) / expected["sigma"])


def test_q6_matches_original_v22_action_and_q6_minus_q5_is_c():
    v22 = pytest.importorskip("test_phase3_AIF_simplified_v22")
    c64 = np.linspace(0.05, 0.55, 64)
    corrections = np.linspace(-0.012, 0.018, 32)
    values = dict(score=0.29, threshold=0.4, c64=c64, novelty_pct=0.60,
                  corrections=corrections, dmax=0.8, dmean=0.2)
    comp = compute_bea_components(
        score=values["score"], threshold=values["threshold"], c64_scores=c64,
        novelty_percentile=values["novelty_pct"], correction_top32=corrections,
        correction_bank_size=12, topk_dN_max=0.8, topk_dN_mean=0.2)
    cfg = v22.SimpleAIFConfig(query_cost=0.91, adapt_query_cost=False,
                              use_local_saliency=True)
    state = v22.SimpleAIFState(cfg)
    state.query_cost_runtime = 0.91
    evidence = {
        "corr_def": max(0.0, float(corrections.max())),
        "corr_norm": max(0.0, -float(corrections.min())),
        "corr_mean": float(corrections.mean()), "corr_max": float(corrections.max()),
        "corr_min": float(corrections.min()), "corr_bank_nonempty": True,
        "corr_signal_dead": False,
    }
    stats = {"per_layer_summary": {"0": {
        "topk_dN_max": 0.8, "topk_dN_mean": 0.2,
    }}}
    action, _info = v22.select_simple_aif_action(
        cfg=cfg, state=state, ablation="normal_dyn_plus_correction",
        p_defect=comp.p_defect, score=comp.score, theta=comp.threshold,
        sigmaN=comp.sigma, pred_defect=comp.pred_defect,
        novelty=comp.novelty, novelty_info={}, evidence=evidence, A_map_stats=stats)
    v22_priority = (
        _info["risk_classify"] + cfg.beta_entropy * _info["entropy"] +
        cfg.lambda_normal * _info["U_normal"] +
        cfg.lambda_correction * _info["U_correction"] +
        cfg.lambda_defect_side * _info["B_defect"])
    assert comp.objective("Q6") == pytest.approx(v22_priority, abs=1e-12)
    q6_proposed = comp.objective("Q6") > 0.91
    assert q6_proposed is (action == v22.AIFAction.QUERY_LABEL)
    assert comp.objective("Q6") - comp.objective("Q5") == pytest.approx(comp.C)
    assert comp.D > 0.0
    assert ARM_POLICIES["Q5"].objective_terms == ("B", "N", "D")
    assert ARM_POLICIES["Q4"].objective_terms == ("B",)
    assert ARM_POLICIES["Q6"].objective_terms == ("B", "N", "C", "D")


def test_empty_correction_bank_zeroes_c_but_not_predicted_normal_d():
    comp = _components(score=-0.1, threshold=0.0, c64=np.linspace(0, 1, 64),
                       correction=np.zeros(32), bank_size=0)
    assert comp.C == 0.0
    assert comp.correction_evidence == 0.0
    assert comp.pred_defect is False
    assert comp.U_D > 0.0
    assert comp.D > 0.0
    with pytest.raises(ValueError, match="empty native C1 bank"):
        _components(correction=np.r_[1.0, np.zeros(31)], bank_size=0)
    with pytest.raises(ValueError, match="nonnegative integer"):
        _components(bank_size=1.5)


@pytest.mark.parametrize("corrections, expected_def, expected_norm", [
    (np.full(32, 0.08), 0.08, 0.0),
    (np.full(32, -0.08), 0.0, 0.08),
    (np.r_[np.full(16, 0.03), np.full(16, -0.05)], 0.03, 0.05),
    (np.zeros(32), 0.0, 0.0),
])
def test_correction_signs_and_evidence_saturation(corrections, expected_def, expected_norm):
    comp = _components(score=-0.05, threshold=0.0, correction=corrections)
    assert comp.corr_def == pytest.approx(expected_def)
    assert comp.corr_norm == pytest.approx(expected_norm)
    assert comp.corr_mag == pytest.approx(max(expected_def, expected_norm))
    assert comp.correction_evidence == pytest.approx(min(1.0, comp.corr_mag / 0.05))
    if comp.correction_evidence == 1.0:
        assert comp.weak_correction == 0.0
        assert comp.U_D == 0.0


def test_local_saliency_v22_boundary_and_multiplier():
    assert local_saliency_from_native_stats(1.0, 1.0) == 0.0
    assert local_saliency_from_native_stats(0.8, 0.4) == pytest.approx(0.5)
    assert local_saliency_from_native_stats(0.8, 0.2) == pytest.approx(0.75)
    comp = _components(score=-0.1, threshold=0.0, correction=np.full(32, 0.01),
                       dmax=0.8, dmean=0.2)
    assert comp.saliency_multiplier == pytest.approx(1.25)
    assert comp.U_C == pytest.approx(comp.psi * 0.2 * 1.25)


def test_b3_threshold_and_current_c64_sample_sigma_are_used():
    c64_a = np.linspace(0.0, 1.0, 64)
    c64_b = np.linspace(0.0, 3.0, 64)
    base = _components(score=0.4, threshold=0.5, c64=c64_a)
    changed_threshold = _components(score=0.4, threshold=0.6, c64=c64_a)
    changed_c64 = _components(score=0.4, threshold=0.5, c64=c64_b)
    assert base.margin_z == pytest.approx((0.4 - 0.5) / np.std(c64_a, ddof=1))
    assert changed_threshold.margin_z != base.margin_z
    assert changed_c64.sigma == pytest.approx(np.std(c64_b, ddof=1))
    assert changed_c64.sigma != base.sigma


def test_adaptive_price_uses_only_executed_actions_and_rolls_at_256():
    controller = QueryController("Q4")
    comp = _components(score=-2.0, threshold=0.0)
    for position in range(256):
        proposal = controller.propose(position=position, event_id_sha256=_event_hash(position),
                                      components=comp)
        controller.observe_final_action(position=position, proposal=proposal,
                                        executed_query=False, group_suppressed=False)
    assert controller.query_cost_runtime == pytest.approx(0.80 - 256 * 0.02 * 0.03)
    assert len(controller.executed_flags) == QUERY_COST_WINDOW
    # The next window is filled with executed queries. The first of these evicts
    # an all-zero flag; the final price uses only the last 256 actions.
    high_priority = replace(comp, B=100.0)
    for position in range(256, 512):
        proposal = controller.propose(position=position, event_id_sha256=_event_hash(position),
                                      components=high_priority)
        controller.observe_final_action(position=position, proposal=proposal,
                                        executed_query=True, group_suppressed=False)
    assert controller.query_cost_runtime > 0.80 - 256 * 0.02 * 0.03
    assert len(controller.executed_flags) == 256
    assert sum(controller.executed_flags) == 256


def test_price_clamps_at_minimum_and_maximum():
    comp = _components(score=-2.0, threshold=0.0)
    low = QueryController("Q4", query_cost_runtime=QUERY_COST_MIN)
    high = QueryController("Q4", query_cost_runtime=QUERY_COST_MAX - 0.001)
    p0 = low.propose(position=0, event_id_sha256=_event_hash(0), components=comp)
    low.observe_final_action(position=0, proposal=p0, executed_query=False, group_suppressed=False)
    high_priority = replace(comp, B=100.0)
    p1 = high.propose(position=0, event_id_sha256=_event_hash(1), components=high_priority)
    high.observe_final_action(position=0, proposal=p1, executed_query=True, group_suppressed=False)
    assert low.query_cost_runtime == QUERY_COST_MIN
    assert high.query_cost_runtime == QUERY_COST_MAX


def test_q0_deterministic_five_view_adjustment_and_q1_periodicity():
    assert P_RANDOM_FIVE_VIEW == pytest.approx(1 - (1 - 5 * 0.03) ** (1 / 5))
    c0a, c0b = QueryController("Q0", seed=41), QueryController("Q0", seed=41)
    comp = _components()
    for position in range(50):
        event_id = _event_hash(position)
        a = c0a.propose(position=position, event_id_sha256=event_id, components=comp)
        b = c0b.propose(position=position, event_id_sha256=event_id, components=comp)
        assert a == b
        c0a.observe_final_action(position=position, proposal=a,
                                 executed_query=a.proposed, group_suppressed=False)
        c0b.observe_final_action(position=position, proposal=b,
                                 executed_query=b.proposed, group_suppressed=False)
    q1 = QueryController("Q1")
    proposals = []
    for position in range(67):
        proposal = q1.propose(position=position, event_id_sha256=_event_hash(position),
                              components=comp)
        proposals.append(position) if proposal.proposed else None
        q1.observe_final_action(position=position, proposal=proposal,
                                executed_query=proposal.proposed, group_suppressed=False)
    assert proposals == [32, 65]


@pytest.mark.parametrize("arm_id", ["Q2", "Q3"])
def test_quantile_arms_use_past_256_only_and_strict_ties(arm_id):
    controller = QueryController(arm_id, seed=3)
    comp = _components(score=-0.2, threshold=0.0, novelty=0.5)
    for position in range(64):
        proposal = controller.propose(position=position, event_id_sha256=_event_hash(position),
                                      components=comp)
        assert proposal.reason == "seeded_cold_start_bernoulli"
        controller.observe_final_action(position=position, proposal=proposal,
                                        executed_query=proposal.proposed, group_suppressed=False)
    assert len(controller.utility_history) == 64
    tied = controller.propose(position=64, event_id_sha256=_event_hash(64), components=comp)
    assert tied.threshold == pytest.approx(float(controller.utility_history[0]))
    assert tied.proposed is False
    assert tied.reason.endswith("at_or_below")


def test_controller_stream_has_no_horizon_bootstrap_or_catchup_state():
    controller = QueryController("Q6")
    comp = _components(score=-0.25, threshold=0.0)
    for position in range(3000):
        proposal = controller.propose(position=position, event_id_sha256=_event_hash(position),
                                      components=comp)
        controller.observe_final_action(position=position, proposal=proposal,
                                        executed_query=proposal.proposed, group_suppressed=False)
    forbidden = {"T", "stream_length", "remaining_events", "target_query_count",
                 "bootstrap", "catchup", "pacing_bin", "terminal_catchup"}
    assert not (forbidden & set(vars(controller)))
    assert controller.executed_queries == controller.proposals
    assert controller.events_observed == 3000


def test_strict_cost_boundary_classifies_on_equality():
    comp = _components()
    arm_id = "Q4"
    controller = QueryController(arm_id)
    cost = comp.objective(arm_id)
    controller.query_cost_runtime = cost
    proposal = controller.propose(position=0, event_id_sha256=_event_hash(0), components=comp)
    assert proposal.proposed is False
    assert proposal.reason == "bea_priority_not_above_runtime_cost"
