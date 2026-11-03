from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.calibration import r8_correction_mechanism_attribution as r8


@dataclass(frozen=True)
class Event:
    position: int
    original_position: int
    image_identity_sha256: str
    source_group_sha256: str
    event_id_sha256: str = "e" * 64
    camera_id: int = 0
    relative_id: str = ""
    timestamp: str = "t"


def event(pos: int, original: int, image: str, group: str) -> Event:
    return Event(pos, original, image * 64, group * 64)


def test_raw_and_shuffle_identity_bijection_and_r7_order_reproduction():
    shuffled = [event(i, original, chr(97 + original), chr(65 + original))
                for i, original in enumerate((2, 0, 3, 1))]
    shuffle0 = r8.make_order_events(shuffled, "SHUFFLE0")
    raw = r8.make_order_events(shuffled, "RAW")
    assert [e.image_identity_sha256 for e in shuffle0] == [e.image_identity_sha256 for e in shuffled]
    assert [e.original_position for e in raw] == [0, 1, 2, 3]
    assert [e.position for e in raw] == [0, 1, 2, 3]
    assert {e.image_identity_sha256 for e in raw} == {e.image_identity_sha256 for e in shuffled}


def test_raw_order_rejects_non_bijective_original_positions():
    rows = [event(0, 0, "a", "A"), event(1, 0, "b", "B")]
    with pytest.raises(r8.R8Error, match="bijection"):
        r8.make_order_events(rows, "RAW")


def test_query_identity_set_is_identical_across_orders():
    shuffled = [event(i, original, chr(97 + original), chr(65 + original))
                for i, original in enumerate((2, 0, 3, 1))]
    raw = r8.make_order_events(shuffled, "RAW")
    q_shuffle = r8.matched_query_positions(shuffled, (0, 2), shuffled)
    q_raw = r8.matched_query_positions(shuffled, (0, 2), raw)
    assert {shuffled[p].image_identity_sha256 for p in q_shuffle} == {
        raw[p].image_identity_sha256 for p in q_raw}
    assert q_raw == tuple(sorted(q_raw))


def test_query_budget_and_one_query_per_source_group_are_enforced():
    rows = [event(i, i, chr(97 + i), chr(65 + i)) for i in range(4)]
    positions = r8.matched_query_positions(rows, (0, 3), rows)
    assert len(positions) == 2
    duplicated = [event(0, 0, "a", "A"), event(1, 1, "b", "A")]
    with pytest.raises(r8.R8Error, match="repeats a Source Group"):
        r8.matched_query_positions(duplicated, (0, 1), duplicated)


def test_nonmonotone_original_index_outcome_projection():
    rows = [event(i, original, chr(97 + original), chr(65 + original))
            for i, original in enumerate((2, 0, 3, 1))]
    labels_original = [0, 1, 0, 1]
    assert r8.map_outcomes_to_order(labels_original, rows) == [0, 0, 1, 1]
    raw = r8.make_order_events(rows, "RAW")
    assert r8.map_outcomes_to_order(labels_original, raw) == labels_original


def test_stage_b_projection_rejects_missing_or_duplicate_identity_positions():
    rows = [event(0, 0, "a", "A"), event(1, 0, "b", "B")]
    with pytest.raises(r8.R8Error, match="bijection"):
        r8.map_outcomes_to_order([0, 1], rows)


def test_score_sealing_precedes_feedback_disclosure():
    order = []
    gate = __import__("experiments.calibration.r7_shuffled_correction_4class",
                      fromlist=["ScoreDisclosureGate"]).ScoreDisclosureGate(
                          lambda row: order.append(("fsync", row["position"])))
    gate.seal(0, {"position": 0, "scores": {"G": 0.2}})
    label = 1 if gate.is_sealed(0) else pytest.fail("feedback opened before fsync")
    order.append(("feedback", label))
    assert order == [("fsync", 0), ("feedback", 1)]


def test_score_gate_rejects_label_fields_and_nonsequential_events():
    gate = __import__("experiments.calibration.r7_shuffled_correction_4class",
                      fromlist=["ScoreDisclosureGate"]).ScoreDisclosureGate(lambda _row: None)
    with pytest.raises(Exception):
        gate.seal(0, {"position": 0, "label": 0})
    with pytest.raises(Exception):
        gate.seal(1, {"position": 1})


def test_queried_label_reader_refuses_unqueried_and_preseal_disclosure():
    reader_type = __import__("experiments.calibration.r7_shuffled_correction_4class",
                             fromlist=["IndexedQueriedLabelReader"]).IndexedQueriedLabelReader
    reader = reader_type.__new__(reader_type)
    reader.allowed = ()
    reader.allowed_set = set()
    reader.cursor = 0
    with pytest.raises(Exception, match="unqueried"):
        reader.label_for(1, sealed_and_fsynced=True)
    reader.allowed = (0,)
    reader.allowed_set = {0}
    with pytest.raises(Exception, match="fsynced"):
        reader.label_for(0, sealed_and_fsynced=False)


def test_capacity_configs_must_differ_only_by_k():
    base = {"K_correction": 256, "enabled": False, "radius0": 0.2,
            "beta_scale": 1.0, "query_rule": "frozen"}
    other = {**base, "K_correction": 1024, "enabled": True}
    r8.assert_capacity_configs_differ_only_by_k(base, other)
    with pytest.raises(r8.R8Error, match="beyond"):
        r8.assert_capacity_configs_differ_only_by_k(base, {**other, "beta_scale": 2.0})


class FakeCodebook:
    def __init__(self, K, feat_dim, device, cfg):
        self.K, self.device, self.cfg = int(K), torch.device(device), cfg
        self.protos = torch.zeros((K, feat_dim), device=self.device)
        self.beta = torch.zeros(K, device=self.device)
        self.radius = torch.ones(K, device=self.device)
        self.kappa = torch.ones(K, device=self.device)
        self.count = torch.zeros(K, dtype=torch.long, device=self.device)
        self.outcome_fp = torch.zeros_like(self.count)
        self.outcome_fn = torch.zeros_like(self.count)
        self.outcome_tn = torch.zeros_like(self.count)
        self.outcome_tp = torch.zeros_like(self.count)
        self.last_step = torch.zeros_like(self.count)
        self.n = 0

    def size(self):
        return self.n

    def apply(self, q):
        if self.n == 0:
            return torch.zeros((len(q),), dtype=torch.float32)
        distance = 1 - q @ self.protos[:self.n].T
        weight = torch.exp(-(distance**2) / (2 * self.radius[:self.n]**2)) * self.kappa[:self.n]
        return (weight * self.beta[:self.n]).sum(1) / (weight.sum(1) + 1e-6)


def synthetic_codebook():
    cfg = SimpleNamespace(enabled=True)
    cb = FakeCodebook(4, 3, "cpu", cfg)
    cb.n = 4
    cb.protos.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.],
                                  [0., 0., 1.], [-1., 0., 0.]]))
    cb.beta.copy_(torch.tensor([-.3, .2, .7, -.8]))
    cb.radius.copy_(torch.tensor([.1, .2, .3, .4]))
    cb.kappa.copy_(torch.tensor([1., 2., 3., 4.]))
    cb.count.copy_(torch.tensor([1, 2, 3, 4]))
    cb.last_step.copy_(torch.tensor([2, 3, 4, 5]))
    return cb


def test_address_bundle_permutation_is_deterministic_and_local():
    first = r8.make_address_permutation(17, "state-A", 64)
    second = r8.make_address_permutation(17, "state-A", 64)
    other_state = r8.make_address_permutation(17, "state-B", 64)
    assert first == second
    assert sorted(first) == list(range(64))
    assert first != other_state


def test_address_seeds_are_fixed_predeclared_and_do_not_share_rng():
    before = np.random.get_state()
    values = [r8.make_address_permutation(seed, "x", 32) for seed in r8.ADDRESS_SEEDS]
    after = np.random.get_state()
    assert r8.ADDRESS_SEEDS == (17, 53, 97)
    assert all(sorted(value) == list(range(32)) for value in values)
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_attribute_bundle_permutation_preserves_prototypes_and_marginals():
    cb = synthetic_codebook()
    snapshot = {name: getattr(cb, name).clone() for name in
                ("protos", "beta", "radius", "kappa", "count", "last_step")}
    shuffled, permutation = r8.permute_attribute_bundles(cb, 53, "state-X")
    assert shuffled is not cb
    assert tuple(sorted(permutation)) == tuple(range(cb.n))
    assert torch.equal(shuffled.protos[:cb.n], snapshot["protos"][:cb.n])
    old_bundles = sorted(zip(snapshot["beta"].tolist(), snapshot["radius"].tolist(),
                             snapshot["kappa"].tolist()))
    new_bundles = sorted(zip(shuffled.beta[:cb.n].tolist(), shuffled.radius[:cb.n].tolist(),
                             shuffled.kappa[:cb.n].tolist()))
    assert new_bundles == old_bundles
    assert torch.equal(shuffled.count[:cb.n], snapshot["count"][:cb.n])
    assert torch.equal(shuffled.last_step[:cb.n], snapshot["last_step"][:cb.n])
    for name, value in snapshot.items():
        assert torch.equal(getattr(cb, name), value)


def test_address_control_breaks_association_but_keeps_application_valid():
    cb = synthetic_codebook()
    clone, permutation = r8.permute_attribute_bundles(cb, 97, "address-test")
    q = torch.tensor([[.65, .30, .05], [.0, .8, .2]])
    assert torch.isfinite(clone.apply(q)).all()
    assert tuple(sorted(permutation)) == tuple(range(cb.n))
    assert not torch.equal(clone.beta[:cb.n], cb.beta[:cb.n])


def test_scalar_control_is_the_c64_mean_shift_and_has_no_state_input():
    c64_g = np.arange(64, dtype=np.float64) / 100
    c64_full = c64_g + 0.125
    score, offset = r8.scalar_control_score(0.7, c64_full, c64_g)
    assert offset == pytest.approx(0.125)
    assert score == pytest.approx(0.825)


def test_capacity_branches_are_separate_objects_with_empty_initial_state():
    cfg = SimpleNamespace(enabled=False)
    cb256, cb1024 = FakeCodebook(256, 3, "cpu", cfg), FakeCodebook(1024, 3, cfg=cfg, device="cpu")
    assert cb256 is not cb1024
    assert cb256.protos.data_ptr() != cb1024.protos.data_ptr()
    assert cb256.size() == cb1024.size() == 0
    assert cb256.K == 256 and cb1024.K == 1024


def test_common_greedy_writer_adapter_uses_c0_only_and_rejects_unqueried_write():
    from experiments.calibration.r1_matched_correction_pilot import common_c0_greedy_writer_action
    ids = np.arange(32, dtype=np.int64)
    features = np.ones((32, 3), dtype=np.float32)
    calls = []
    def update(**kwargs):
        calls.append(kwargs)
        return {"accepted": 1}
    def eligibility(**kwargs):
        return True, 32, "test", None
    result = common_c0_greedy_writer_action(
        position=0, c0_score=.2, theta=.5, sigma=.1, c0_prediction=0,
        candidate_ids=ids, candidate_features=features, feature_layers=[features],
        queried=False, disclosed_label=None, eligibility=eligibility, update=update)
    assert result["input_score_source"] == "common_C0"
    assert result["action"] == "no_normal_memory_write"
    assert not calls


def test_common_greedy_writer_never_admits_queried_defects():
    from experiments.calibration.r1_matched_correction_pilot import common_c0_greedy_writer_action
    ids = np.arange(32, dtype=np.int64)
    features = np.ones((32, 4), dtype=np.float32)
    calls = []
    result = common_c0_greedy_writer_action(
        position=1, c0_score=.8, theta=.5, sigma=.2, c0_prediction=1,
        candidate_ids=ids, candidate_features=features, feature_layers=[features],
        queried=True, disclosed_label=1,
        eligibility=lambda **_: (True, 32, "eligible", None),
        update=lambda **kw: calls.append(kw))
    assert result["action"] == "verified_defect_no_normal_write"
    assert not calls


def test_identical_c0_inputs_produce_identical_shared_memory_actions():
    from experiments.calibration.r1_matched_correction_pilot import common_c0_greedy_writer_action
    ids = np.arange(32, dtype=np.int64)
    features = np.ones((32, 2), dtype=np.float32)
    def one(_counterfactual):
        state = []
        result = common_c0_greedy_writer_action(
            position=2, c0_score=.1, theta=.5, sigma=.2, c0_prediction=0,
            candidate_ids=ids, candidate_features=features, feature_layers=[features],
            queried=True, disclosed_label=0,
            eligibility=lambda **_: (True, 32, "eligible", None),
            update=lambda **kwargs: state.append((kwargs["score"], kwargs["candidate_ids"].copy())) or {"ok": 1})
        return result, state
    full_result, full_state = one("full correction score A")
    other_result, other_state = one("different C1 correction score B")
    assert full_result == other_result
    assert len(full_state) == len(other_state) == 1
    assert full_state[0][0] == other_state[0][0] == .1
    assert np.array_equal(full_state[0][1], other_state[0][1])


def test_metrics_are_controller_invariant_for_identical_score_arrays():
    labels = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int8)
    scores = np.asarray([.1, .9, .4, .6, .8, .3])
    first = r8._ranking_metrics(labels, scores)
    second = r8._ranking_metrics(labels.copy(), scores.copy())
    assert first == second
    assert first["auroc"] == 1.0


def test_source_group_bootstrap_is_deterministic_and_paired():
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
    a = np.asarray([.1, .9, .2, .8, .4, .6])
    b = np.asarray([.2, .8, .3, .7, .5, .5])
    groups = ["a", "a", "b", "b", "c", "c"]
    first = r8._source_group_bootstrap(labels, a, b, groups, 11)
    second = r8._source_group_bootstrap(labels, a, b, groups, 11)
    assert first == second
    assert first["replicates_requested"] == 1000
    assert first["method"].startswith("paired whole-Source-Group")


def synthetic_parity_rows(count=3):
    rows, refs = [], []
    for pos in range(count):
        decisions_a, decisions_r = {}, {}
        for scorer, ref in (("G1024", "G"), ("GC1024", "GC")):
            for controller in r8.CONTROLLERS:
                decisions_a[f"{scorer}-{controller}"] = {"prediction": pos % 2,
                                                         "threshold": .5}
                decisions_r[f"{ref}-{controller}"] = {"prediction": pos % 2,
                                                       "threshold": .5}
        rows.append({"position": pos, "original_json_position": pos,
                     "image_identity_sha256": f"{pos:064x}",
                     "source_group_sha256": f"{pos+10:064x}", "camera_id": 0,
                     "structured_fixed_support_sha256": "f"*64,
                     "query_tape_decision": pos == 1,
                     "scores": {"G1024": float(pos), "GC1024": float(pos + .1)},
                     "decisions": decisions_a})
        refs.append({"position": pos, "original_json_position": pos,
                     "image_identity_sha256": f"{pos:064x}",
                     "source_group_sha256": f"{pos+10:064x}", "camera_id": 0,
                     "structured_fixed_support_sha256": "f"*64,
                     "query_tape_decision": pos == 1,
                     "scores": {"G": float(pos), "GC": float(pos + .1)},
                     "decisions": decisions_r})
    return rows, refs


def test_r7_sealed_prefix_parity_checks_all_six_controller_decisions():
    rows, refs = synthetic_parity_rows()
    result = r8._compare_r7_rows("audiojack", rows, refs)
    assert result["decision_mismatches"] == 0
    assert result["maximum_absolute_score_error"] == {"G1024": 0.0, "GC1024": 0.0}
    rows[1]["decisions"]["GC1024-B3"]["prediction"] = 0
    with pytest.raises(r8.R8Error, match="decision"):
        r8._compare_r7_rows("audiojack", rows, refs)


def test_stage_a_condition_reader_does_not_mutate_immutable_outputs(tmp_path):
    leaf = tmp_path / "audiojack_shuffle0"
    leaf.mkdir()
    rows, _ = synthetic_parity_rows(2)
    for row in rows:
        row.update({"category": "audiojack", "order": "SHUFFLE0", "label": None})
        row.pop("label")
    payload = "".join(r8._canonical(row).decode("utf-8") for row in rows)
    (leaf / "stage_a_scores.jsonl").write_bytes(payload.encode("utf-8"))
    (leaf / "run_manifest.json").write_text(json.dumps({"category": "audiojack", "order": "SHUFFLE0"}), encoding="utf-8")
    (leaf / "attempt_history.jsonl").write_text("{}\n", encoding="utf-8")
    (leaf / "stage_a_seal.json").write_text(json.dumps({
        "category": "audiojack", "order": "SHUFFLE0", "event_count": 2,
        "query_count": 1, "query_positions": [1],
        "score_rows_sha256": hashlib.sha256(payload.encode()).hexdigest()}), encoding="utf-8")
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in leaf.iterdir()}
    read, _ = r8._verify_condition_output(leaf, "audiojack", "SHUFFLE0", 2, [1])
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in leaf.iterdir()}
    assert len(read) == 2
    assert before == after


def test_public_stage_b_metrics_do_not_expose_event_vectors_or_paths():
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    assert set(r8._ranking_metrics(labels, [.1, .9, .2, .8])) == {"auroc", "average_precision"}
    assert not any(key in r8._ranking_metrics(labels, [.1, .9, .2, .8])
                   for key in ("labels", "scores", "image_path", "event_id_sha256"))


def test_address_and_scalar_controls_are_read_only_with_fixed_seed_set():
    cb = synthetic_codebook()
    before = {name: getattr(cb, name).clone() for name in
              ("protos", "beta", "radius", "kappa", "last_step")}
    r8.permute_attribute_bundles(cb, 17, "same-state")
    r8.scalar_control_score(.3, np.ones(64), np.zeros(64))
    assert all(torch.equal(getattr(cb, name), value) for name, value in before.items())
    assert r8.ADDRESS_SEEDS == (17, 53, 97)


def test_source_exclusion_happens_before_address_bundle_permutation(monkeypatch):
    live = synthetic_codebook()
    cfg = live.cfg
    bank = SimpleNamespace(corr_cfg=cfg, corr_cb={"p0": {"id": [live]}})
    calls = []

    class FakeExp2A:
        @staticmethod
        def _switch_codebook(bank, _arm, new_cfg, new_cb):
            prior = bank.corr_cfg, bank.corr_cb["p0"]["id"][0]
            bank.corr_cfg, bank.corr_cb["p0"]["id"][0] = new_cfg, new_cb
            new_cfg.enabled = True
            return prior

        @staticmethod
        def _restore_codebook(bank, new_cfg, prior_cfg, prior_cb):
            new_cfg.enabled = False
            bank.corr_cfg, bank.corr_cb["p0"]["id"][0] = prior_cfg, prior_cb

    class FakeR1:
        @staticmethod
        def _filtered_c1_codebook(_impl, active_bank, _groups, _group, *, event_position):
            calls.append(("exclude", event_position))
            assert active_bank.corr_cb["p0"]["id"][0] is live
            filtered = FakeCodebook(3, 3, "cpu", cfg)
            filtered.n = 3
            for name in ("protos", "beta", "radius", "kappa", "count", "outcome_fp",
                         "outcome_fn", "outcome_tn", "outcome_tp", "last_step"):
                getattr(filtered, name)[:3].copy_(getattr(live, name)[1:4])
            return filtered, [1, 2, 3]

    original_permute = r8.permute_attribute_bundles
    def tracked_permute(filtered, seed, state):
        calls.append(("permute", filtered.n, seed, state))
        assert filtered.n == 3
        return original_permute(filtered, seed, state)
    monkeypatch.setattr(r8, "permute_attribute_bundles", tracked_permute)
    shuffled, eligible, permutation = r8._address_control_codebook(
        object(), bank, FakeR1, FakeExp2A, cfg, live, {}, "current-group",
        9, 17, "state-identity")
    assert calls[0] == ("exclude", 9)
    assert calls[1][0:3] == ("permute", 3, 17)
    assert eligible == [1, 2, 3]
    assert len(permutation) == shuffled.size() == 3
    assert bank.corr_cb["p0"]["id"][0] is live


def test_b1_normal_query_and_b3_state_refresh_apply_after_current_prediction():
    from experiments.calibration.controllers import RollingQ90QueriedNormal, StateConsistentQuantile
    from experiments.calibration.protocol import Feedback, Observation
    from experiments.calibration import r7_shuffled_correction_4class as r7

    b1 = RollingQ90QueriedNormal(.5, window=256, min_count=16, name="test-B1")
    current_thresholds = []
    for position in range(16):
        obs = Observation(t=position, observation_id=f"e{position}", camera_id=0,
                          score=.8, memory_version=0, score_state_digest="a"*64)
        pred = b1.predict(obs)
        current_thresholds.append(pred.threshold)
        b1.observe_feedback(obs, pred, Feedback(t=position, queried=True, label=0))
        b1.finalize_step()
    assert all(value == .5 for value in current_thresholds)
    assert b1.theta == pytest.approx(.8)
    next_obs = Observation(t=16, observation_id="next", camera_id=0,
                           score=.75, memory_version=0, score_state_digest="a"*64)
    assert b1.predict(next_obs).threshold == pytest.approx(.8)

    b3 = StateConsistentQuantile(.2, delta=.01, min_normals=16, name="test-B3")
    initial = [.2] * 64
    receipt = r7._initialize_b3(b3, initial, "a"*64, "b"*64, .2)
    event_value = SimpleNamespace(position=0, event_id_sha256="id", camera_id=0)
    pre = r7._predict_b3(b3, position=0, event=event_value, score=.6,
                         version=0, state_digest="b"*64)
    assert pre["threshold"] == .2
    transition = r7._refresh_b3(b3, [.8] * 64, "a"*64,
                                "b"*64, "c"*64, 0)
    assert receipt["version"] == 0 and transition["version_after"] == 1
    assert pre["threshold"] == .2
    assert b3.theta == pytest.approx(.21)


def test_r7_runtime_contract_keeps_deterministic_bootstrap_arithmetic():
    # The same physical groups and seed produce the same paired interval; no
    # event-level rows or label vectors are returned by the summary helper.
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    groups = ["g1", "g1", "g2", "g2"]
    result = r8._source_group_bootstrap(labels, [.1, .9, .2, .8],
                                        [.2, .8, .3, .7], groups, 2026)
    assert "auroc_difference_ci_95_percentile" in result
    assert "labels" not in result and "scores" not in result


def test_stage_b_fails_closed_before_outcome_access_without_global_stage_a_seal(tmp_path, monkeypatch):
    from experiments.calibration import r8_correction_mechanism_attribution as module
    import torch
    run_root = tmp_path / "r8"
    seals = run_root / "full" / "stage_seals"
    manifest_dir = run_root / "manifest"
    seals.mkdir(parents=True)
    manifest_dir.mkdir()
    (manifest_dir / "run_manifest.json").write_text(json.dumps({"identity": {}}), encoding="utf-8")
    (seals / "stage_a_seal.json").write_text(json.dumps({"status": "INCOMPLETE"}), encoding="utf-8")
    monkeypatch.setattr(module, "RUN_NAMESPACE", run_root)
    identity = {"runner_source_sha256": module._sha256_file(Path(module.__file__).resolve()),
                "source_fingerprints": module._source_fingerprints()}
    (manifest_dir / "run_manifest.json").write_text(json.dumps({"identity": identity}), encoding="utf-8")
    monkeypatch.setattr(module, "_assert_repository_identity", lambda: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    with pytest.raises(r8.R8Error, match="refused labels"):
        module._stage_b_evaluate()


def test_stage_b_mechanism_summary_respects_shuffle_only_registration():
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    raw_rows = [{"scores": {"G1024": .1, "GC256": .2, "GC1024": .3}}
                for _ in labels]
    assert r8._stage_b_mechanism_summary("RAW", labels, raw_rows) == {
        "status": "NOT_APPLICABLE_BY_PREDECLARED_DESIGN"}

    names = ("GC1024", "Scalar", "Address17", "Address53", "Address97")
    vectors = {
        "GC1024": [.1, .9, .2, .8],
        "Scalar": [.2, .8, .3, .7],
        "Address17": [.1, .8, .2, .7],
        "Address53": [.3, .7, .4, .6],
        "Address97": [.4, .6, .3, .7],
    }
    shuffle_rows = [{"scores": {name: vectors[name][i] for name in names}}
                    for i in range(len(labels))]
    summary = r8._stage_b_mechanism_summary("SHUFFLE0", labels, shuffle_rows)
    assert summary["GC1024"]["auroc"] == 1.0
    assert "Scalar" in summary["full_minus_control"]
    assert set(summary["early_middle_late"]) == set(names)
    with pytest.raises(r8.R8Error, match="missing registered mechanism score Scalar"):
        r8._stage_b_mechanism_summary(
            "SHUFFLE0", labels, [{"scores": {"GC1024": .5}} for _ in labels])
