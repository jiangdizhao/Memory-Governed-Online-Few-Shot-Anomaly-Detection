from dataclasses import dataclass

import numpy as np
import pytest
import torch

from experiments.calibration import r9_correction_mechanism_falsification as r9
from experiments.calibration import r7_shuffled_correction_4class as r7


@dataclass(frozen=True)
class Event:
    position: int
    original_position: int
    relative_id: str
    image_identity_sha256: str
    source_group_sha256: str
    camera_id: str = "0"
    timestamp: str = "t"
    image_path: str = "x.jpg"

    @property
    def event_id_sha256(self):
        return f"{self.position:064x}"


def _event(position, original, group, image):
    return Event(position, original, image, r7._identity_hash(image), f"{group:0<64}")


def test_seeded_group_orders_are_deterministic_and_query_identity_matched():
    seed0 = [_event(i, i, "a" if i < 2 else "b", f"image-{i}") for i in range(4)]
    # All four rows are unique and the same queried image identities map in each order.
    order11a = r9.make_order_events(seed0, "SHUFFLE11")
    order11b = r9.make_order_events(seed0, "SHUFFLE11")
    order23 = r9.make_order_events(seed0, "SHUFFLE23")
    assert order11a == order11b
    q0 = (0, 3)
    q11 = r9.matched_query_positions(seed0, q0, order11a)
    q23 = r9.matched_query_positions(seed0, q0, order23)
    assert {order11a[p].image_identity_sha256 for p in q11} == {
        seed0[p].image_identity_sha256 for p in q0}
    assert {order23[p].image_identity_sha256 for p in q23} == {
        seed0[p].image_identity_sha256 for p in q0}
    for rows in (order11a, order23):
        group_positions = {}
        for row in rows:
            group_positions.setdefault(row.source_group_sha256, []).append(row.original_position)
        assert all(values == sorted(values) for values in group_positions.values())


def test_registered_conditions_include_support1_only_when_receipts_validate():
    support0 = r9._registered_condition_configuration(None)
    assert support0["orders"] == list(r9.ORDERS)
    assert support0["support_seeds"] == [0]

    support1 = r9._registered_condition_configuration({"audiojack": {"verified": True}})
    assert support1["orders"] == [*r9.ORDERS, "SUPPORT1_SHUFFLE0"]
    assert support1["support_seeds"] == [0, 1]


def test_stage_b_resource_receipt_reads_model_fingerprints_from_runtime_profile():
    required = (
        "first_normal_feedback_position", "first_defect_feedback_position",
        "c1_final_occupancy", "c1_first_full_position", "c1_update_stats",
        "simple_memory_final_occupancy", "simple_memory_insertions",
        "simple_memory_evictions", "simple_memory_first_full_position",
        "simple_memory_unavailable_sign_patch_events", "simple_memory_source_exclusions",
        "c64_rescore_costs", "mutable_memory_bytes", "timing_seconds",
        "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "peak_cpu_rss_bytes",
        "greedy_writer_audit", "c64_cache_receipt", "native_scorer_profile_sha256",
        "stream_image_encodings", "fixed_support_receipt",
    )
    completion = {key: f"value:{key}" for key in required}
    completion["runtime_profile"] = {
        "hub_source_tree_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "loaded_model_state_sha256": "c" * 64,
        "runtime_profile": {"profile_id": "test-profile"},
    }
    receipt = r9._resource_and_state_receipt(completion)
    assert receipt["hub_source_tree_sha256"] == "a" * 64
    assert receipt["checkpoint_sha256"] == "b" * 64
    assert receipt["loaded_model_state_sha256"] == "c" * 64
    assert receipt["runtime_profile"] == {"profile_id": "test-profile"}
    assert receipt["c1_final_occupancy"] == "value:c1_final_occupancy"

    completion["runtime_profile"].pop("hub_source_tree_sha256")
    with pytest.raises(r9.R8Error, match="model fingerprints"):
        r9._resource_and_state_receipt(completion)


def test_support1_c64_encoding_does_not_use_support0_bound_r7_cache(monkeypatch):
    called = []

    def encode_direct(category, info, p0, impl, backbone, cfg):
        called.append((category, info, p0, impl, backbone, cfg))
        return ["direct-stage05-c64"], {"reused": False}

    class R7:
        @staticmethod
        def _make_c64_items(*_args):
            raise AssertionError("Support 1 must not use the Support-0-bound cache")

    monkeypatch.setattr(r9, "_encode_stage05_c64_items", encode_direct)
    result = r9._make_r9_c64_items("audiojack", {}, object(), object(), object(),
                                    object(), R7, support_seed=1)
    assert result == (["direct-stage05-c64"], {"reused": False})
    assert len(called) == 1


def test_active_r9_stage_a_uses_support_aware_c64_factory():
    import inspect

    source = inspect.getsource(r9._stage_a_child)
    assert "_make_r9_c64_items(" in source
    assert "r7._make_c64_items(category, info, p0, impl, backbone, cfg)" not in source


def test_direct_stage05_c64_encoder_preserves_native_feature_contract():
    class FakeImage:
        def convert(self, _mode):
            return self

        def __array__(self, dtype=None):
            value = np.zeros((2, 3, 3), dtype=np.uint8)
            return value.astype(dtype) if dtype is not None else value

    class FakeP0:
        @staticmethod
        def encode_query(_impl, _backbone, path, _cfg):
            assert path == "c64-0.png"
            features = torch.zeros((784, 4096), dtype=torch.float32)
            return FakeImage(), [features]

    info = {
        "c64_paths": ["c64-0.png"],
        "c64_groups": ["group-hash"],
        "c64_scores": [0.25],
        "frozen_info": {"c64": {"group_sha256": "a" * 64,
                                 "image_sha256": "b" * 64}},
    }
    items, receipt = r9._encode_stage05_c64_items(
        "audiojack", info, FakeP0, None, None, None, expected_count=1)
    assert len(items) == 1
    assert items[0]["source_group_sha256"] == "group-hash"
    assert items[0]["features"][0].shape == (784, 4096)
    assert items[0]["features"][0].dtype == np.float32
    assert receipt["count"] == 1
    assert receipt["new_c64_dino_encodings"] == 1
    assert receipt["c64_image_sha256"] == "b" * 64


class FakeCodebook:
    names = ("protos", "beta", "radius", "kappa", "count", "outcome_fp",
             "outcome_fn", "outcome_tn", "outcome_tp", "last_step")

    def __init__(self, K, feat_dim, device, cfg):
        self.K, self.device, self.cfg = int(K), device, cfg
        self.n = 0
        self.protos = torch.zeros((K, feat_dim), dtype=torch.float32)
        self.beta = torch.zeros(K, dtype=torch.float32)
        self.radius = torch.ones(K, dtype=torch.float32)
        self.kappa = torch.ones(K, dtype=torch.float32)
        self.count = torch.zeros(K, dtype=torch.int64)
        self.outcome_fp = torch.zeros(K, dtype=torch.int64)
        self.outcome_fn = torch.zeros(K, dtype=torch.int64)
        self.outcome_tn = torch.zeros(K, dtype=torch.int64)
        self.outcome_tp = torch.zeros(K, dtype=torch.int64)
        self.last_step = torch.zeros(K, dtype=torch.int64)

    def apply(self, query):
        if self.n == 0:
            return torch.zeros(query.shape[0], dtype=torch.float32)
        return self.beta[:self.n].mean().expand(query.shape[0])


def _fake_codebook():
    cb = FakeCodebook(4, 2, "cpu", object())
    cb.n = 4
    cb.protos.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
    cb.beta.copy_(torch.tensor([-.2, -.1, .1, .2]))
    cb.radius.copy_(torch.tensor([.05, .08, .12, .2]))
    cb.kappa.copy_(torch.tensor([.5, .7, 1.1, 1.5]))
    for name in ("count", "outcome_fp", "outcome_fn", "outcome_tn", "outcome_tp", "last_step"):
        getattr(cb, name).copy_(torch.arange(4))
    return cb


def test_beta_only_clone_changes_only_beta_and_is_deterministic():
    original = _fake_codebook()
    before = {name: getattr(original, name).clone() for name in FakeCodebook.names}
    first, perm1 = r9.permute_beta_only(original, 17, "a" * 64)
    second, perm2 = r9.permute_beta_only(original, 17, "a" * 64)
    assert perm1 == perm2
    assert torch.equal(first.beta, second.beta)
    assert sorted(first.beta[:first.n].tolist()) == sorted(before["beta"][:original.n].tolist())
    for name in FakeCodebook.names:
        if name != "beta":
            assert torch.equal(getattr(first, name), before[name])
        assert torch.equal(getattr(original, name), before[name])
    assert not torch.equal(first.beta[:first.n], before["beta"][:original.n])


def test_beta_control_applies_source_filter_before_permutation(monkeypatch):
    order = []
    live = _fake_codebook()

    class Bank:
        device = torch.device("cpu")
        corr_cfg = object()
        corr_cb = {"p0": {"id": [live]}}

    class Exp2A:
        @staticmethod
        def _switch_codebook(*_args):
            return None, None

        @staticmethod
        def _restore_codebook(*_args):
            return None

    class R1:
        @staticmethod
        def _filtered_c1_codebook(_impl, _bank, _groups, _group, *, event_position):
            order.append(("filter", event_position))
            return live, [1, 2, 3, 4]

        @staticmethod
        def corrected_dense_map(base_map, _ids, delta):
            return np.asarray(base_map) + float(np.mean(delta))

        @staticmethod
        def max_score(value):
            return float(np.max(value))

    class Impl:
        @staticmethod
        def l2_normalize(value):
            return value

    # The clone constructor is called after _filtered_c1_codebook returns.
    orig_init = FakeCodebook.__init__

    def tracked_init(self, *args):
        order.append(("clone", None))
        orig_init(self, *args)

    monkeypatch.setattr(FakeCodebook, "__init__", tracked_init)
    score, receipt = r9.beta_only_control_score(impl=Impl(), bank=Bank(), r1=R1(),
        exp2a=Exp2A(), cfg=object(), codebook=live, source_groups={},
        group_sha256="f" * 64, position=9, selected_features=torch.ones((32, 2)),
        base_map=np.zeros(784, dtype=np.float32), top_ids=np.arange(32),
        seed=53, state_identity="c" * 64)
    assert order[0] == ("filter", 9)
    assert order[1][0] == "clone"
    assert receipt["eligible_count"] == 4
    assert np.isfinite(score)


def test_signed_exemplar_missing_sign_source_exclusion_and_fifo():
    memory = r9.SignedExemplarMemory(feature_dim=2, device="cpu")
    query = torch.tensor([[1., 0.]], dtype=torch.float32)
    empty, empty_diag = memory.residual(query, "a" * 64)
    assert empty.item() == 0.0 and empty_diag["both_signs_unavailable"]
    memory.insert(query, sign=1, source_group_sha256="a" * 64, position=0)
    excluded, diag = memory.residual(query, "a" * 64)
    assert excluded.item() == 0.0
    assert diag["same_source_excluded_positive"] == 1
    matching, diag = memory.residual(query, "b" * 64)
    assert matching.item() == pytest.approx(.2)
    assert diag["eligible_positive"] == 1 and diag["eligible_negative"] == 0
    memory.insert(torch.cat([query] * 512, dim=0), sign=1,
                  source_group_sha256="c" * 64, position=1)
    assert memory.counts[1] == 512
    assert memory.evictions[1] == 1
    assert memory.positions[0] == 1
    assert memory.counts[-1] == 0
    ledger = memory.resource_ledger()
    assert ledger["prototype_allocated_bytes"] == 1024 * 2 * 4
    assert ledger["measured_fixed_array_bytes"] == ledger["prototype_allocated_bytes"] + 1024 * 36


def test_signed_exemplar_budget_and_frozen_parameters_fail_closed():
    memory = r9.SignedExemplarMemory(feature_dim=4096, device="cpu")
    ledger = memory.resource_ledger()
    c1_allocated_budget = 1024 * 4096 * 4 + 61_440
    assert ledger["measured_fixed_array_bytes"] <= c1_allocated_budget
    with pytest.raises(r9.R8Error):
        r9.SignedExemplarMemory(feature_dim=4096, device="cpu", positive_capacity=256)
    with pytest.raises(r9.R8Error):
        r9.SignedExemplarMemory(feature_dim=4096, device="cpu", r0=.2)


def test_operating_threshold_changes_cannot_change_precomputed_ranking_inputs():
    scores = np.asarray([.1, .4, .2, .9])
    labels = np.asarray([0, 1, 0, 1])
    before = r9._ranking_metrics(labels, scores)
    predictions_a = (scores >= .3).astype(np.int8)
    predictions_b = (scores >= .8).astype(np.int8)
    assert not np.array_equal(predictions_a, predictions_b)
    assert r9._ranking_metrics(labels, scores) == before


def test_full_condition_receipt_binds_ordered_identity_and_fails_closed(tmp_path, monkeypatch):
    import json

    run_root = tmp_path / "r9"
    manifest_dir = run_root / "manifest"
    leaf = run_root / "full" / "audiojack_shuffle0"
    manifest_dir.mkdir(parents=True)
    leaf.mkdir(parents=True)
    monkeypatch.setattr(r9, "RUN_NAMESPACE", run_root)
    row = {"category": "audiojack", "order": "SHUFFLE0", "position": 0,
        "original_json_position": 8, "image_identity_sha256": "a" * 64,
        "source_group_sha256": "b" * 64, "camera_id": 3,
        "query_tape_decision": True, "scores": {"G1024": 0.1}}
    identity_rows = r9._condition_identity_rows([row])
    identity = {"identity_sha256": "c" * 64, "runner_source_sha256": "d" * 64,
        "classes": {"audiojack": {"orders": {"SHUFFLE0": {
            "identity_map_sha256": r9._sha256_json(identity_rows)}}}}}
    (manifest_dir / "run_manifest.json").write_text(json.dumps({"identity": identity}), encoding="utf-8")
    (leaf / "run_manifest.json").write_text(json.dumps({"run_id": r9.RUN_ID,
        "identity_sha256": identity["identity_sha256"],
        "runner_source_sha256": identity["runner_source_sha256"],
        "category": "audiojack", "order": "SHUFFLE0", "mode": "FULL"}), encoding="utf-8")
    (leaf / "stage_a_scores.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    seal = {"category": "audiojack", "order": "SHUFFLE0", "event_count": 1,
        "query_positions": [0], "query_identity_set_sha256": r9._sha256_json(["a" * 64]),
        "score_rows_sha256": r9._sha256_file(leaf / "stage_a_scores.jsonl")}
    (leaf / "stage_a_seal.json").write_text(json.dumps(seal), encoding="utf-8")
    (leaf / "attempt_history.jsonl").write_text("{}\n", encoding="utf-8")
    rows, _ = r9._verify_condition_output(leaf, "audiojack", "SHUFFLE0", 1, [0])
    assert len(rows) == 1

    row["source_group_sha256"] = "e" * 64
    (leaf / "stage_a_scores.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    seal["score_rows_sha256"] = r9._sha256_file(leaf / "stage_a_scores.jsonl")
    (leaf / "stage_a_seal.json").write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(r9.R8Error, match="identity checks"):
        r9._verify_condition_output(leaf, "audiojack", "SHUFFLE0", 1, [0])


def test_source_group_bootstrap_is_paired_deterministic_and_resamples_whole_groups(monkeypatch):
    monkeypatch.setattr(r9, "BOOTSTRAP_REPLICATES", 32)
    labels = [0, 1] * 6
    groups = [f"group-{i}" for i in range(6) for _ in range(2)]
    score_a = [.1, .9] * 6
    score_b = [.4, .6] * 6
    first = r9._r9_source_group_bootstrap(labels, score_a, score_b, groups, 19)
    second = r9._r9_source_group_bootstrap(labels, score_a, score_b, groups, 19)
    assert first == second
    assert first["valid_replicates"] == 32
    assert first["method"].startswith("paired whole-Source-Group")
    assert first["auroc_difference_ci_95_percentile"] == [0.0, 0.0]


def test_reference_parity_uses_registered_mixed_score_tolerance():
    decisions = {f"{scorer}-{controller}": {"prediction": 1, "threshold": .5}
        for scorer in ("G1024", "GC1024") for controller in r9.CONTROLLERS}
    actual = {"position": 0, "image_identity_sha256": "a" * 64,
        "original_json_position": 5, "source_group_sha256": "b" * 64,
        "event_id_sha256": "c" * 64, "camera_id": 1,
        "structured_fixed_support_sha256": "d" * 64,
        "query_tape_decision": False,
        "scores": {"G1024": .1, "GC1024": .2}, "decisions": decisions}
    reference = {**actual, "scores": {"G": .1, "GC": .2},
        "decisions": {f"{scorer}-{controller}": {"prediction": 1, "threshold": .5}
            for scorer in ("G", "GC") for controller in r9.CONTROLLERS}}
    receipt = r9._compare_reference_rows("audiojack", [actual], [reference], "R7")
    assert receipt["maximum_absolute_score_error"] == {"G1024": 0.0, "GC1024": 0.0}
    actual["scores"]["G1024"] += 3e-5
    with pytest.raises(r9.R8Error, match="score parity"):
        r9._compare_reference_rows("audiojack", [actual], [reference], "R7")


def test_all_r9_output_names_are_in_the_shared_budget_allowlist():
    from experiments.calibration.r2_feedback_repair import ALL_OUTPUTS

    required = {"run_manifest.json", "attempt_history.jsonl", "stage_a_scores.jsonl",
        "stage_a_seal.json", "stage_b_seal.json", "aggregate_results.json",
        "complete_seal.json", "supervisor_status.json", "failure.json"}
    assert required <= ALL_OUTPUTS
