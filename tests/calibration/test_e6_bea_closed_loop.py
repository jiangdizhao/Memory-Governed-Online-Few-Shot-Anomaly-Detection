import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import numpy as np
import pytest
import torch

from experiments.calibration.e6_bea_closed_loop import (
    ARM_IDS,
    E6_REGISTERED_CATEGORIES,
    ArmRuntime,
    EncodedEvent,
    E6ClosedLoopError,
    EventIdentity,
    FeedbackUpdate,
    NativePreEventScore,
    NormalCoverageMemory,
    OnDemandLabelReader,
    PrivateFeedbackSink,
    R7FrozenFeatureEncoder,
    ScoreLedger,
    StageAClosedLoop,
    StageASeal,
    StageBLabelSource,
    EvaluationLabelRecord,
    stage_b_evaluate,
    make_e6_event_stream,
    validate_five_view_groups,
    validate_data_role_group_disjointness,
    validate_test_image_projection,
    R7_HELPER_SOURCE_SHA256,
    verify_e6_execution_identity,
    _write_json_once,
    _read_sealed_score_log,
    create_e6_output_namespace,
    _validate_real_output_namespace,
    _validate_stage_b_identity_records,
    _decode_selected_e6_label,
    _c64_parity_receipt,
    _assert_native_module_pair,
    _row_value,
    _worst_window_ber,
    _canonical_json,
    _sha256_file,
)
from experiments.calibration.e6_bea_policy import (
    P_RANDOM_FIVE_VIEW,
    QueryController,
    _stable_uniform01,
    compute_bea_components,
)


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _events(group_count=4):
    rows = []
    position = 0
    for group_index in range(group_count):
        group = _sha(f"group-{group_index}")
        for camera in range(1, 6):
            image = _sha(f"image-{group_index}-{camera}")
            event = _sha(f"event-{position}-{image}-{group}")
            rows.append(EventIdentity(
                position=position, original_position=position,
                event_id_sha256=event, image_identity_sha256=image,
                source_group_sha256=group,
                relative_id=f"synthetic/{group_index}/{camera}.png",
                camera_id=str(camera), timestamp=f"202611040000{position:02d}",
                image_path=f"synthetic/{group_index}/{camera}.png"))
            position += 1
    return tuple(rows)


class FakeReader:
    def __init__(self, arm_id, label=0):
        self.arm_id = arm_id
        self.label = label
        self.accessed = []

    def label_for(self, event, receipt):
        assert receipt.arm_id == self.arm_id
        self.accessed.append(event.position)
        return self.label


class FakeEngine:
    def __init__(self, *, initial="a" * 64, label_threshold_shift=0.0):
        self.initial_state_sha256 = initial
        self.current_threshold = 0.0
        self.label_threshold_shift = label_threshold_shift
        self.normal_writes = 0
        self.c1_updates = 0
        self.unqueried_observations = 0
        self.state = 0
        self.seen_encoded_ids = []

    def score(self, event, encoded):
        self.seen_encoded_ids.append(encoded)
        return NativePreEventScore(
            gc_score=self.current_threshold - 0.001,
            c0_score=0.25,
            c64_scores=tuple(np.linspace(0.0, 1.0, 64)),
            correction_top32=tuple(np.zeros(32)), correction_bank_size=0,
            topk_dN_max=0.8, topk_dN_mean=0.2,
            scorer_state_version=self.state,
            scorer_state_sha256=self.initial_state_sha256)

    def observe_unqueried(self, event, encoded, score):
        self.unqueried_observations += 1
        return {"action": "no_normal_memory_write"}

    def apply_feedback(self, event, encoded, score, label):
        self.c1_updates += 1
        if label == 0:
            self.normal_writes += 1
        self.state += 1
        if self.label_threshold_shift:
            self.current_threshold += self.label_threshold_shift
        return FeedbackUpdate(
            label=label, c1_changed=True, greedy_changed=(label == 0),
            scorer_state_version=self.state, scorer_state_sha256=self.initial_state_sha256,
            diagnostics={"normal_write": label == 0})


def _coverage():
    support = np.eye(4, dtype=np.float32)
    return NormalCoverageMemory(support)


def _choose_q0_seed(events):
    return next(seed for seed in range(10000) if all(
        _stable_uniform01(seed, "Q0", event.event_id_sha256, "random") >= P_RANDOM_FIVE_VIEW
        for event in events))


def _make_runtime(tmp_path, events, *, label=0, threshold_shift=0.0, category="audiojack"):
    arms = {}
    q0_seed = _choose_q0_seed(events)
    for arm_id in ARM_IDS:
        engine = FakeEngine(label_threshold_shift=threshold_shift)
        reader = FakeReader(arm_id, label=label)
        arms[arm_id] = ArmRuntime(
            arm_id=arm_id, engine=engine,
            controller=QueryController(arm_id, seed=q0_seed),
            coverage=_coverage(), label_reader=reader)
    encoded_ids = []

    def encode(_event):
        value = EncodedEvent(image=None, feature_layers=(), global_cls=np.asarray([1, 0, 0, 0],
                                                                               dtype=np.float32))
        encoded_ids.append(id(value))
        return value

    ledger = ScoreLedger(tmp_path / "sealed_scores.jsonl")
    runner = StageAClosedLoop(category=category, events=events,
                              feature_encoder=encode, arms=arms, ledger=ledger,
                              synthetic_test_only=True)
    return runner, arms, encoded_ids


def test_five_view_preflight_rejects_wrong_group_size_and_duplicate_camera():
    events = _events(1)
    assert validate_five_view_groups(events) == {"source_groups": 1, "events": 5,
                                                  "views_per_group": 5}
    with pytest.raises(E6ClosedLoopError, match="five-distinct-view|camera-1-through-5"):
        validate_five_view_groups(events[:4])
    duplicate = list(events)
    duplicate[1] = EventIdentity(**{**duplicate[1].__dict__, "camera_id": "1"})
    with pytest.raises(E6ClosedLoopError, match="camera-1-through-5"):
        validate_five_view_groups(duplicate)


def test_support_c64_and_test_group_identities_must_be_disjoint():
    events = _events(1)
    support = [_sha(f"support-{index}") for index in range(4)]
    c64 = [_sha(f"c64-{index}") for index in range(64)]
    validate_data_role_group_disjointness(events, support, c64)
    with pytest.raises(E6ClosedLoopError, match="must be disjoint"):
        validate_data_role_group_disjointness(events, [events[0].source_group_sha256, *support[1:]], c64)
    with pytest.raises(E6ClosedLoopError, match="must be disjoint"):
        validate_data_role_group_disjointness(events, support, [events[0].source_group_sha256, *c64[1:]])
    with pytest.raises(E6ClosedLoopError, match="identities are incomplete"):
        validate_data_role_group_disjointness(events, support, c64[:-1])


def test_manifest_writer_is_exclusive_and_rejects_public_label_fields(tmp_path):
    path = tmp_path / "run_manifest.json"
    digest = _write_json_once(path, {"schema": "synthetic", "status": "INCOMPLETE"})
    assert len(digest) == 64
    with pytest.raises(FileExistsError):
        _write_json_once(path, {"schema": "replacement"})
    forbidden = tmp_path / "forbidden.json"
    with pytest.raises(E6ClosedLoopError, match="forbidden field"):
        _write_json_once(forbidden, {"label": 1})
    assert not forbidden.exists()


def test_output_namespace_creation_is_unique_and_never_reuses_existing_directory(tmp_path):
    output_parent = tmp_path / "outputs" / "calibration"
    output_parent.mkdir(parents=True)
    run_dir = create_e6_output_namespace("cpu-test-a1", repository_root=tmp_path)
    assert run_dir == output_parent / "e6_bea_v2_cpu-test-a1"
    marker = json.loads((run_dir / "run_namespace.json").read_text(encoding="utf-8"))
    assert marker == {"schema": "e6.bea_v2.run_namespace.v1",
                      "run_id": "cpu-test-a1", "status_at_creation": "INCOMPLETE"}
    with pytest.raises(FileExistsError):
        create_e6_output_namespace("cpu-test-a1", repository_root=tmp_path)
    with pytest.raises(E6ClosedLoopError, match="filesystem-safe"):
        create_e6_output_namespace("../bad", repository_root=tmp_path)


def _real_namespace_fixture(tmp_path, run_id="real-namespace-test"):
    (tmp_path / "outputs" / "calibration").mkdir(parents=True, exist_ok=True)
    run_dir = create_e6_output_namespace(run_id, repository_root=tmp_path)
    ledger = ScoreLedger(run_dir / "sealed_scores.jsonl")
    arms = {}
    for arm_id in ARM_IDS:
        sink = PrivateFeedbackSink(run_dir / f"private_feedback_{arm_id}.jsonl", arm_id)
        arms[arm_id] = SimpleNamespace(private_feedback=sink)
    return run_dir, ledger, arms


def test_genuine_real_stage_a_namespace_validator_accepts_exact_outputs(tmp_path):
    run_dir, ledger, arms = _real_namespace_fixture(tmp_path)
    try:
        receipt = _validate_real_output_namespace(
            ledger=ledger, arms=arms,
            output_parent=tmp_path / "outputs" / "calibration")
        assert receipt["run_id"] == "real-namespace-test"
        assert set(path.name for path in run_dir.iterdir()) == {
            "run_namespace.json", "sealed_scores.jsonl",
            *(f"private_feedback_{arm}.jsonl" for arm in ARM_IDS)}
    finally:
        ledger.close()
        for runtime in arms.values():
            runtime.private_feedback.close()


@pytest.mark.parametrize("malformation", ["extra_file", "ledger_alias", "renamed_private"])
def test_genuine_real_stage_a_namespace_validator_fails_closed(tmp_path, malformation):
    run_dir, ledger, arms = _real_namespace_fixture(tmp_path, f"bad-{malformation}")
    try:
        if malformation == "extra_file":
            (run_dir / "unaccounted.json").write_text("{}", encoding="utf-8")
        elif malformation == "ledger_alias":
            arms["Q1"].private_feedback.path = ledger.path
        else:
            arms["Q1"].private_feedback.path = run_dir / "private_feedback_other.jsonl"
        with pytest.raises(E6ClosedLoopError):
            _validate_real_output_namespace(
                ledger=ledger, arms=arms,
                output_parent=tmp_path / "outputs" / "calibration")
    finally:
        ledger.close()
        for runtime in arms.values():
            runtime.private_feedback.close()


def test_pinned_v22_and_vmb_modules_are_a_consistent_pair_and_swaps_fail():
    v22 = pytest.importorskip("test_phase3_AIF_simplified_v22")
    visionad = pytest.importorskip("vmb_visionad_new")
    root = Path(__file__).resolve().parents[2]
    _assert_native_module_pair(v22, visionad, root=root)
    with pytest.raises(E6ClosedLoopError, match="pinned source"):
        _assert_native_module_pair(visionad, v22, root=root)
    incompatible = SimpleNamespace(
        __file__=visionad.__file__, DINOv2MultiLayerBackbone=object())
    with pytest.raises(E6ClosedLoopError, match="backbone class"):
        _assert_native_module_pair(v22, incompatible, root=root)


def test_frozen_metadata_rows_support_mapping_and_attribute_objects():
    from dataclasses import dataclass

    @dataclass
    class Row:
        opaque_group: str

    assert _row_value({"opaque_group": "a"}, "opaque_group") == "a"
    assert _row_value(Row("b"), "opaque_group") == "b"
    assert _row_value(Row("b"), "missing", "default") == "default"


def test_e6_runner_default_is_inert_and_real_mode_requires_separate_authorization(monkeypatch, capsys):
    from experiments.calibration import e6_bea_runner as runner

    monkeypatch.setattr(runner, "load_e6_stage05_category",
                        lambda *_args, **_kwargs: pytest.fail("default path accessed metadata"))
    assert runner.main([]) == 0
    assert "no dataset, model, or GPU was accessed" in capsys.readouterr().out
    with pytest.raises(E6ClosedLoopError, match="separate human authorization"):
        runner.execute_real_stage_a(
            category="audiojack", run_id="must-not-run", human_authorized=False)
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "r7._frozen_contract(" not in source
    assert "r7._make_query_tape(" not in source


def test_c64_source_score_digest_is_distinct_from_tolerance_parity_receipt():
    from experiments.calibration import r3_fullstream_2x2 as r3

    expected = np.linspace(0.0, 1.0, 64).tolist()
    base = [value + 1.0e-5 for value in expected]
    empty_gc = [value - 1.0e-5 for value in base]
    receipt = _c64_parity_receipt(expected, base, empty_gc, r3._score_tolerance,
                                  atol=r3.ATOL, rtol=r3.RTOL)
    assert receipt["base_expected_matches"] == 64
    assert receipt["empty_gc_expected_matches"] == 64
    assert receipt["empty_gc_base_matches"] == 64
    assert receipt["expected_score_vector_sha256"] != receipt["base_score_vector_sha256"]
    with pytest.raises(E6ClosedLoopError, match="frozen R3 tolerance"):
        _c64_parity_receipt(expected, [value + 0.01 for value in expected],
                            empty_gc, r3._score_tolerance, atol=r3.ATOL, rtol=r3.RTOL)


def _build_synthetic_native_engines(monkeypatch, category_json_sha256, *,
                                    incompatible_profile=False,
                                    recomputed_base_scores=None,
                                    recomputed_gc_scores=None):
    import importlib
    from dataclasses import dataclass
    from experiments.calibration import e6_bea_closed_loop as e6
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import native
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r2_exp2a_c1_strength as exp2a
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration import r6_capacity1024_audiojack_runner as r6
    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration import profiles

    class Codebook:
        def __init__(self, protos, capacity):
            self.protos = protos
            self.K = capacity
            self.device = protos.device
            self.dtype = protos.dtype

        def size(self):
            return int(self.protos.shape[0])

    class BufferCodebook(Codebook):
        def __init__(self, capacity, feat_dim, device, **_kwargs):
            super().__init__(torch.empty((0, int(feat_dim)), dtype=torch.float32,
                                         device=device), int(capacity))
            self.feat_dim = int(feat_dim)

    class CorrectionCodebook:
        def __init__(self):
            self.K = 1024
            self.protos = torch.empty((0, 4), dtype=torch.float16)
            for name in ("beta", "radius", "kappa", "count", "outcome_fp",
                         "outcome_fn", "outcome_tn", "outcome_tp", "last_step"):
                setattr(self, name, torch.empty((0,), dtype=torch.float32))

        def size(self):
            return 0

    def make_bank():
        greedy = Codebook(torch.empty((0, 4), dtype=torch.float16), 1024)
        ltm = Codebook(torch.empty((0, 4), dtype=torch.float16), 1)
        candidates = BufferCodebook(1, 4, torch.device("cpu"))
        fixed = Codebook(torch.tensor([[1, 0, 0, 0]], dtype=torch.float16), 25000)
        corr = CorrectionCodebook()
        return SimpleNamespace(
            normal_dyn={"p0": [{"id": [greedy]}]},
            normal_dyn_ltm={"p0": [{"id": [ltm]}]},
            normal_dyn_candidates={"p0": [{"id": [candidates]}]},
            normal_fixed={"p0": {"id": [fixed]}},
            corr_cb={"p0": {"id": [corr]}},
            corr_cfg=SimpleNamespace(enabled=False, K_correction=1024),
            device=torch.device("cpu"), use_defect_bank=False)

    v22 = p0.require_sources()
    visionad = importlib.import_module("vmb_visionad_new")
    root = Path(__file__).resolve().parents[2]
    expected_head = "5ed82338e255345fee80de31e91980ca51ab8fbe"
    monkeypatch.setattr(e6, "verify_e6_execution_identity", lambda *_a, **_k: {
        "branch": e6.EXPECTED_BRANCH, "head": expected_head,
        "approved_e6_commit": expected_head, "r7_helper_source_sha256": {}})
    def synthetic_set_seed(_seed):
        return None

    # Keep the pinned module ownership metadata while avoiding CUDA seeding in
    # this CPU-only structural integration test.
    synthetic_set_seed.__module__ = v22.__name__
    monkeypatch.setattr(v22, "set_seed", synthetic_set_seed)
    monkeypatch.setattr(p0, "build_vmb", lambda *_args, **_kwargs: make_bank())
    monkeypatch.setattr(r3, "_validate_correction_config", lambda _bank: None)
    monkeypatch.setattr(r6, "_configure_r6_correction_capacity", lambda bank: (
        bank.corr_cfg, [bank.corr_cb["p0"]["id"][0]]))
    expected_scores = np.linspace(0.0, 1.0, 64).tolist()
    base_replay = list(expected_scores if recomputed_base_scores is None
                       else recomputed_base_scores)
    gc_replay = list(expected_scores if recomputed_gc_scores is None
                     else recomputed_gc_scores)
    monkeypatch.setattr(r7, "_score_c64_base", lambda *_a, **_k:
                        (list(base_replay), [None] * 64, 0.0))
    monkeypatch.setattr(r7, "_score_c64_corrected", lambda *_a, **_k:
                        (list(gc_replay), 0.0))
    fixed_digest = native._fixed_bank_digest(make_bank())
    support_paths = [f"synthetic/support-{index}.jpg" for index in range(4)]
    support_groups = [f"support-source-{index}" for index in range(4)]
    c64_source_groups = [f"c64-source-{index}" for index in range(64)]
    support_group_sha = sorted(hashlib.sha256((
        "official-sample-group\0" + value).encode()).hexdigest() for value in support_groups)
    c64_groups = [hashlib.sha256((
        "official-sample-group\0" + value).encode()).hexdigest() for value in c64_source_groups]
    c64_group_sha = _sha("ordered-c64-groups")
    category_info = {
        "contract": {
            "support_paths": support_paths,
            "category_json_sha256": category_json_sha256,
            "c64_group_sha256": c64_group_sha,
            "stream_rows": [],
        },
        "frozen_info": {
            "supports": {"0": {"groups": support_groups,
                                  "images": [f"support-{i}.jpg" for i in range(4)]}},
            "c64": {"groups": c64_source_groups,
                    "images": [f"c64-{i}.jpg" for i in range(64)],
                    "group_sha256": c64_group_sha,
                    "image_sha256": _sha("c64-images")},
        },
        "support_image_sha256": _sha("support-images"),
        "support_group_sha256": hashlib.sha256(_canonical_json(support_group_sha)).hexdigest(),
        "fixed_support_sha256_expected": fixed_digest,
        "c64_score_archive_sha256": _sha("c64-score-archive"),
        "c64_groups": c64_groups,
        "c64_scores": expected_scores,
        "theta_c64_q90": float(np.quantile(expected_scores, 0.90, method="higher")),
        "sigma_c64_floor_002": max(float(np.std(expected_scores, ddof=1)), 0.02),
    }

    @dataclass
    class MetadataRow:
        opaque_group: str

    test_group = hashlib.sha256(
        b"official-sample-group\0audiojack_0001_OK").hexdigest()
    category_info["contract"]["stream_rows"] = [MetadataRow(test_group)]
    support_features = {path: [[torch.zeros((2, 4), dtype=torch.float16)]]
                        for path in support_paths}
    c64_items = [{"source_group_sha256": value} for value in c64_groups]
    cfg, _profile = r3._native_config(p0, profiles, r1)
    if incompatible_profile:
        cfg = SimpleNamespace(**vars(cfg))
        cfg.feature_fuse = "mean"
    engines = e6.build_r7_native_arm_engines(
        category="audiojack", info=category_info, impl=v22, visionad=visionad,
        p0=p0, r1=r1, r3=r3, r6=r6, r7=r7, exp2a=exp2a,
        native_module=native, cfg=cfg, support_features=support_features,
        support_paths=support_paths, c64_items=c64_items,
        approved_e6_commit=expected_head, repository_root=root)
    return engines, category_info, support_groups, c64_groups, v22, visionad, r1


def test_real_native_factory_and_real_stage_a_initialization_use_synthetic_inputs(
        tmp_path, monkeypatch):
    import importlib
    from experiments.calibration import e6_bea_closed_loop as e6
    category_json = tmp_path / "audiojack.json"
    _events_value, test_json = _real_iad_synthetic_events()
    category_json.write_text(json.dumps(test_json), encoding="utf-8")
    category_sha = _sha_file(category_json)
    engines, info, support_groups, c64_groups, v22, visionad, r1 = (
        _build_synthetic_native_engines(monkeypatch, category_sha))
    assert len({id(engines[arm].bank) for arm in ARM_IDS}) == 7
    assert len({engine.initial_state_sha256 for engine in engines.values()}) == 1
    assert all(engine.c64_parity_receipt["base_expected_matches"] == 64
               for engine in engines.values())
    for engine in engines.values():
        expected = tuple(info["c64_scores"])
        assert tuple(engine.c64_initial_scores) == expected
        assert tuple(engine.c64_scores) == expected
        assert tuple(engine.c64_recomputed_base_scores) == expected
        assert tuple(engine.c64_recomputed_gc_scores) == expected
        assert tuple(engine.b3_controller.normal_scores) == expected
        assert engine.b3_controller.evidence_version == engine.state_version == 0
        assert engine.b3_controller.theta == engine.theta0
        assert engine.b3_controller._r7_state_digest == engine.state_digest
        assert engine.initial_b3_c64_initialization["frozen_reference_source"] == (
            "authenticated_stage05_c64_score_archive")
        assert engine.initial_b3_c64_initialization["frozen_reference_archive_sha256"] == (
            engine.input_identity["c64_score_archive_sha256"])
        assert engine.initial_b3_c64_initialization["scorer_state_sha256"] == engine.state_digest

    source_events, _ = _real_iad_synthetic_events()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    events = []
    for index, event in enumerate(source_events):
        image = image_dir / f"view-{index}.jpg"
        image.write_bytes(b"synthetic-unopened-image")
        events.append(EventIdentity(**{**event.__dict__, "image_path": str(image)}))
    events = tuple(events)
    # Bind actual Test JSON image identities to each synthetic Stage A event.
    from experiments.calibration import r7_shuffled_correction_4class as r7
    rows = []
    for event in source_events:
        rows.append({"position": event.original_position,
                     "opaque_group": event.source_group_sha256,
                     "relative_id": event.relative_id,
                     "camera_id": event.camera_id, "timestamp": event.timestamp,
                     "image_path": str(image_dir / f"view-{event.position}.jpg")})
    events = make_e6_event_stream(rows)
    events = tuple(EventIdentity(**{**event.__dict__,
                                    "image_path": str(image_dir / f"view-{event.position}.jpg")})
                   for event in events)
    info["contract"]["stream_rows"] = rows
    info["contract"]["category_json_sha256"] = category_sha
    info["c64_group_sha256"] = info["contract"]["c64_group_sha256"]
    # Replace the synthetic test-group receipt with the exact group present in this JSON.
    test_group = events[0].source_group_sha256
    info["contract"]["stream_rows"] = [SimpleNamespace(opaque_group=test_group)]
    info["c64_groups"] = c64_groups

    root = tmp_path
    (root / "outputs" / "calibration").mkdir(parents=True)
    run_dir = create_e6_output_namespace("native-init-test", repository_root=root)
    ledger = ScoreLedger(run_dir / "sealed_scores.jsonl")
    arms = {}
    reader_list, sink_list = [], []
    for arm_id in ARM_IDS:
        reader = OnDemandLabelReader(category_json, "audiojack", category_sha,
                                     events, arm_id, ledger)
        sink = PrivateFeedbackSink(run_dir / f"private_feedback_{arm_id}.jsonl", arm_id)
        reader_list.append(reader)
        sink_list.append(sink)
        arms[arm_id] = ArmRuntime(
            arm_id, engines[arm_id], QueryController(arm_id, seed=37),
            NormalCoverageMemory(np.eye(4, dtype=np.float32)), reader, sink)
    monkeypatch.setattr(e6, "__file__", str(root / "experiments" / "calibration" /
                                                "e6_bea_closed_loop.py"))
    r1_hash = _sha("synthetic-model-state")
    monkeypatch.setattr(r1, "_model_state_sha256", lambda _model: r1_hash)
    encoder = R7FrozenFeatureEncoder(
        impl=v22, visionad=visionad, repository_root=Path(__file__).resolve().parents[2],
        backbone=SimpleNamespace(model=object()), feature_fuse="concat", r1=r1,
        expected_model_state_sha256=r1_hash,
        runtime_info={"expected_model_state_sha256": r1_hash,
                      "loaded_model_state_sha256": r1_hash, "runtime_profile": "synthetic"})
    stage_a = StageAClosedLoop(category="audiojack", events=events,
                               feature_encoder=encoder, arms=arms, ledger=ledger,
                               synthetic_test_only=False,
                               execution_authorization_id="cpu-synthetic-review")
    manifest = json.loads(Path(stage_a.manifest_path).read_text(encoding="utf-8"))
    assert manifest["query_policies"]["Q0"]["seed"] == 37
    assert manifest["execution_authorization_id"] == "cpu-synthetic-review"
    b3_init = manifest["initial_b3_c64_initialization"]
    assert b3_init["b3_evidence_version"] == 0
    assert b3_init["active_initial_b3_normal_scores_sha256"] == (
        b3_init["frozen_reference_score_vector_sha256"])
    assert b3_init["scorer_state_sha256"] == manifest[
        "initial_complete_scorer_state_sha256"]
    assert set(path.name for path in run_dir.iterdir()) == {
        "run_namespace.json", "sealed_scores.jsonl", "run_manifest.json",
        *(f"private_feedback_{arm}.jsonl" for arm in ARM_IDS)}
    assert not (run_dir / "stage_a_seal.json").exists()
    ledger.close()
    for reader in reader_list:
        reader.close()
    for sink in sink_list:
        sink.close()


@pytest.mark.parametrize("perturbation", ["uniform_plus_1e6", "mixed_within_tolerance"])
def test_native_engine_initial_b3_uses_frozen_c64_reference_after_tolerated_replay(
        monkeypatch, perturbation):
    from experiments.calibration import r3_fullstream_2x2 as r3

    reference = np.linspace(0.0, 1.0, 64, dtype=np.float64)
    if perturbation == "uniform_plus_1e6":
        recomputed = reference + 1e-6
    else:
        signs = np.where(np.arange(64) % 2 == 0, 1.0, -1.0)
        recomputed = reference + signs * 1e-6
    assert all(r3._score_tolerance(float(actual), float(expected))
               for actual, expected in zip(recomputed, reference))
    recomputed_q90 = float(np.quantile(recomputed, 0.90, method="higher"))
    recomputed_sigma = max(float(np.std(recomputed, ddof=1)), 0.02)
    reference_q90 = float(np.quantile(reference, 0.90, method="higher"))
    reference_sigma = max(float(np.std(reference, ddof=1)), 0.02)
    assert recomputed_q90 != reference_q90
    if perturbation == "mixed_within_tolerance":
        assert recomputed_sigma != reference_sigma

    engines, info, *_ = _build_synthetic_native_engines(
        monkeypatch, _sha("synthetic-category-json"),
        recomputed_base_scores=recomputed.tolist(),
        recomputed_gc_scores=recomputed.tolist())
    expected = tuple(info["c64_scores"])
    for engine in engines.values():
        assert tuple(engine.c64_recomputed_base_scores) == tuple(recomputed)
        assert tuple(engine.c64_recomputed_gc_scores) == tuple(recomputed)
        assert tuple(engine.c64_scores) == expected
        assert tuple(engine.b3_controller.normal_scores) == expected
        assert engine.theta0 == reference_q90
        assert engine.sigma0 == reference_sigma
        assert engine.b3_controller.theta == engine.current_threshold == engine.theta0
        assert engine.b3_controller.evidence_version == engine.state_version == 0
        assert engine.b3_controller._r7_state_digest == engine.state_digest
        assert engine.initial_b3_c64_initialization[
            "recomputed_native_empty_gc_score_vector_sha256"] == (
                engine.c64_parity_receipt["empty_gc_score_vector_sha256"])


@pytest.mark.parametrize("component", ["base", "empty_gc"])
def test_native_engine_rejects_out_of_tolerance_c64_replay(monkeypatch, component):
    reference = np.linspace(0.0, 1.0, 64, dtype=np.float64)
    out_of_tolerance = reference + 0.01
    replay = {"recomputed_base_scores": reference.tolist(),
              "recomputed_gc_scores": reference.tolist()}
    replay["recomputed_base_scores" if component == "base" else
           "recomputed_gc_scores"] = out_of_tolerance.tolist()
    with pytest.raises(E6ClosedLoopError, match="frozen R3 tolerance"):
        _build_synthetic_native_engines(
            monkeypatch, _sha("synthetic-category-json"), **replay)


def test_real_native_factory_rejects_swapped_backend(monkeypatch):
    from experiments.calibration import e6_bea_closed_loop as e6
    v22 = pytest.importorskip("test_phase3_AIF_simplified_v22")
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(e6, "verify_e6_execution_identity", lambda *_a, **_k: {
        "branch": e6.EXPECTED_BRANCH, "head": "5ed82338e255345fee80de31e91980ca51ab8fbe",
        "approved_e6_commit": "5ed82338e255345fee80de31e91980ca51ab8fbe",
        "r7_helper_source_sha256": {}})
    with pytest.raises(E6ClosedLoopError, match="pinned source"):
        e6.build_r7_native_arm_engines(
            category="audiojack", info={}, impl=v22, visionad=v22, p0=None,
            r1=None, r3=None, r6=None, r7=None, exp2a=None,
            native_module=None, cfg=None, support_features={}, support_paths=[],
            c64_items=[], approved_e6_commit="5ed82338e255345fee80de31e91980ca51ab8fbe",
            repository_root=root)


def test_native_module_pair_rejects_foreign_function_owner(monkeypatch):
    import importlib
    from experiments.calibration import e6_bea_closed_loop as e6

    v22 = pytest.importorskip("test_phase3_AIF_simplified_v22")
    visionad = importlib.import_module("vmb_visionad_new")
    root = Path(__file__).resolve().parents[2]

    def foreign_seed(_seed):
        return None

    monkeypatch.setattr(v22, "set_seed", foreign_seed)
    with pytest.raises(E6ClosedLoopError, match="set_seed is not owned"):
        e6._assert_native_module_pair(v22, visionad, root=root)


def test_native_factory_rejects_incompatible_scorer_profile(monkeypatch):
    with pytest.raises(E6ClosedLoopError, match="native scorer config differs"):
        _build_synthetic_native_engines(
            monkeypatch, _sha("{}"), incompatible_profile=True)


@pytest.mark.parametrize("change", ["c1", "greedy", "both"])
def test_native_feedback_change_attribution_and_b3_refresh(monkeypatch, tmp_path, change):
    from experiments.calibration import e6_bea_closed_loop as e6

    category_json = tmp_path / "identity.json"
    category_json.write_text("{}", encoding="utf-8")
    engines, _info, _support, _c64, _v22, _visionad, r1 = (
        _build_synthetic_native_engines(monkeypatch, _sha_file(category_json)))
    engine = engines["Q0"]
    c1_state = {"value": "before"}
    greedy_state = {"value": "before"}
    monkeypatch.setattr(r1, "_memory_content_sha256",
                        lambda _bank: _sha(greedy_state["value"]))
    monkeypatch.setattr(r1, "_correction_state_sha256",
                        lambda _bank, _groups: _sha(c1_state["value"]))

    def capture(*_args, **_kwargs):
        if change in ("c1", "both"):
            c1_state["value"] = "after"
        return None, None, {"synthetic_c1_update": True}

    def writer(_event, _score, *, queried, label):
        assert queried and label == 0
        if change in ("greedy", "both"):
            greedy_state["value"] = "after"
        return {"action": "synthetic"}

    monkeypatch.setattr(engine.exp2a, "_capture_update", capture)
    monkeypatch.setattr(engine, "_common_writer", writer)
    base_rescore_calls = []
    rescored_c64 = [value + 0.005 for value in engine.c64_scores]

    def rescore_base(*_args, **_kwargs):
        base_rescore_calls.append(True)
        return list(engine.c64_initial_scores), list(engine.c64_base_cache), 0.0

    monkeypatch.setattr(engine.r7, "_score_c64_base", rescore_base)
    monkeypatch.setattr(engine.r7, "_score_c64_corrected", lambda *_a, **_k:
                        (list(rescored_c64), 0.0))
    before = engine._digests()["GC"]
    engine.state_digest = before
    engine.b3_controller._r7_state_digest = before
    event = _events(1)[0]
    threshold_before_event = engine.current_threshold
    score = NativePreEventScore(
        gc_score=threshold_before_event + 0.002, c0_score=0.4,
        c64_scores=engine.c64_scores,
        correction_top32=(), correction_bank_size=0,
        topk_dN_max=0.5, topk_dN_mean=0.2,
        scorer_state_version=0, scorer_state_sha256=before,
        context={"features": [], "top_ids": np.asarray([], dtype=np.int64),
                 "candidate_features": np.empty((0, 4), dtype=np.float32)})
    current_event_prediction = int(score.gc_score >= threshold_before_event)
    update = engine.apply_feedback(
        event, EncodedEvent(None, (), np.asarray([1, 0, 0, 0], dtype=np.float32)),
        score, 0)
    assert update.c1_changed is (change in ("c1", "both"))
    assert update.greedy_changed is (change in ("greedy", "both"))
    assert update.scorer_state_changed
    assert update.scorer_state_version == 1
    assert update.diagnostics["b3_transition_for_next_event"] is not None
    assert len(base_rescore_calls) == int(change in ("greedy", "both"))
    assert tuple(engine.c64_scores) == tuple(rescored_c64)
    assert tuple(engine.b3_controller.normal_scores) == tuple(rescored_c64)
    assert engine.b3_controller._r7_state_digest == engine.state_digest
    assert engine.current_threshold == pytest.approx(threshold_before_event + 0.005)
    assert score.scorer_state_sha256 == before
    assert int(score.gc_score >= threshold_before_event) == current_event_prediction == 1
    assert int(score.gc_score >= engine.current_threshold) == 0


def test_utility_ties_and_query_inactivity_are_recorded_without_changing_strict_q97():
    controller = QueryController("Q2", seed=37)
    components = compute_bea_components(
        score=0.4, threshold=0.3, c64_scores=np.linspace(0.1, 0.4, 64),
        novelty_percentile=0.4, correction_top32=np.zeros(32),
        correction_bank_size=1, topk_dN_max=0.8, topk_dN_mean=0.2)
    for position in range(257):
        proposal = controller.propose(
            position=position, event_id_sha256=_sha(f"q2-{position}"),
            components=components)
        controller.observe_final_action(
            position=position, proposal=proposal,
            executed_query=bool(proposal.proposed), group_suppressed=False)
    diagnostics = controller.diagnostics()
    assert diagnostics["quantile_comparisons"] == 193
    assert diagnostics["quantile_ties"] == 193
    assert diagnostics["quantile_tie_rate"] == 1.0
    assert diagnostics["proposals"] == diagnostics["executed_queries"]
    assert diagnostics["max_inactivity_events"] >= 0


def test_e6_event_builder_reuses_r7_shuffle_but_never_fixed_query_tape(monkeypatch):
    from experiments.calibration import r7_shuffled_correction_4class as r7

    def no_fixed_tape(*_args, **_kwargs):
        raise AssertionError("E6 must not build or consult the historical R7 fixed query tape")

    monkeypatch.setattr(r7, "_make_query_tape", no_fixed_tape)
    rows = []
    for group_index in range(2):
        group = _sha(f"source-{group_index}")
        for camera in range(1, 6):
            rows.append({"opaque_group": group, "position": len(rows),
                         "relative_id": f"synthetic/{group_index}/{camera}.png",
                         "camera_id": str(camera), "timestamp": str(camera),
                         "image_path": f"synthetic/{group_index}/{camera}.png"})
    first = make_e6_event_stream(rows)
    second = make_e6_event_stream(rows)
    assert first == second
    assert len(first) == 10
    assert validate_five_view_groups(first)["source_groups"] == 2


def test_r7_feature_encoder_returns_patch_features_and_global_cls_once():
    calls = {"load": 0, "encode": 0}

    class FakeImpl:
        @staticmethod
        def load_rgb(_path):
            calls["load"] += 1
            return object()

        @staticmethod
        def encode_pil(_backbone, _image, mode):
            calls["encode"] += 1
            assert mode == "concat"
            return [torch.ones((784, 4096), dtype=torch.float32)], torch.ones(1024)

    encoded = R7FrozenFeatureEncoder(impl=FakeImpl(), backbone=object(),
                                     feature_fuse="concat")(_events(1)[0])
    assert calls == {"load": 1, "encode": 1}
    assert encoded.feature_layers[0].shape == (784, 4096)
    assert encoded.global_cls.shape == (1024,)
    assert np.isfinite(encoded.global_cls).all()


def test_stage_a_seals_all_arms_before_feedback_and_encodes_once_per_event(tmp_path):
    events = _events(4)
    runner, arms, encoded_ids = _make_runtime(tmp_path, events)
    seal = runner.run()
    assert seal.complete and seal.event_count == len(events)
    assert len(encoded_ids) == len(events)
    assert all(len(arms[arm].engine.seen_encoded_ids) == len(events) for arm in ARM_IDS)
    for position in range(len(events)):
        ids = {id(arms[arm].engine.seen_encoded_ids[position]) for arm in ARM_IDS}
        assert len(ids) == 1
    rows = [json.loads(line) for line in Path(seal.score_log_path).read_text().splitlines()]
    assert len(rows) == len(events)
    assert all("label" not in json.dumps(row) and "anomaly_class" not in json.dumps(row)
               for row in rows)
    for arm_id in ARM_IDS:
        reader = arms[arm_id].label_reader
        assert reader.accessed == [row["position"] for row in rows
                                   if row["arm_rows"][arm_id]["query_executed"]]
        assert len(reader.accessed) == seal.arm_query_counts[arm_id]
    assert sum(seal.arm_query_counts.values()) > 0
    assert seal.arm_query_counts["Q0"] == 0
    assert seal.arm_query_counts["Q4"] == 4
    assert all(arms[arm].engine.c1_updates == seal.arm_query_counts[arm]
               for arm in ARM_IDS)
    assert seal.arm_controller_diagnostics["Q4"]["group_suppressed"] == 16
    assert rows[0]["arm_rows"]["Q5"]["components"] == rows[0]["arm_rows"]["Q6"]["components"]
    assert all(len({row["source_group_sha256"] for row in rows
                    if row["arm_rows"][arm]["query_executed"]}) ==
               seal.arm_query_counts[arm] for arm in ARM_IDS)


def test_partial_stage_a_failure_closes_writers_and_cannot_be_resumed(tmp_path):
    events = _events(1)
    runner, arms, _ = _make_runtime(tmp_path, events)
    query_seed = next(seed for seed in range(10000) if
                      _stable_uniform01(seed, "Q0", events[0].event_id_sha256, "random") <
                      P_RANDOM_FIVE_VIEW)
    arms["Q0"].controller = QueryController("Q0", seed=query_seed)

    def fail_after_query_seal(_event, _receipt):
        raise RuntimeError("synthetic disclosure-provider failure")

    arms["Q0"].label_reader.label_for = fail_after_query_seal
    with pytest.raises(RuntimeError, match="synthetic disclosure-provider failure"):
        runner.run()
    assert runner.ledger._stream.closed
    assert len(Path(runner.ledger.path).read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(FileExistsError):
        ScoreLedger(runner.ledger.path)


def test_seven_independent_engines_start_equal_then_diverge_by_query_policy(tmp_path):
    runner, arms, _ = _make_runtime(tmp_path, _events(3))
    initial = {arms[arm].engine.initial_state_sha256 for arm in ARM_IDS}
    assert initial == {"a" * 64}
    assert len({id(arms[arm].engine) for arm in ARM_IDS}) == 7
    assert len({id(arms[arm].coverage) for arm in ARM_IDS}) == 7
    runner.run()
    assert arms["Q4"].engine.normal_writes == 3
    assert arms["Q0"].engine.normal_writes == 0
    assert arms["Q4"].engine.c1_updates == 3
    assert arms["Q0"].engine.c1_updates == 0


def test_real_stage_a_rejects_synthetic_engines_unless_test_only_is_explicit(tmp_path):
    events = _events(1)
    runner, arms, _ = _make_runtime(tmp_path, events)
    with pytest.raises(E6ClosedLoopError, match="real Stage A requires"):
        StageAClosedLoop(category="audiojack", events=events,
                         feature_encoder=runner.feature_encoder,
                         arms=arms, ledger=runner.ledger)


def test_query_group_exclusion_suppresses_other_cameras_without_rescheduling(tmp_path):
    runner, arms, _ = _make_runtime(tmp_path, _events(2))
    seal = runner.run()
    rows = [json.loads(line) for line in Path(seal.score_log_path).read_text().splitlines()]
    q4 = [row["arm_rows"]["Q4"] for row in rows]
    assert [index for index, row in enumerate(q4) if row["query_executed"]] == [0, 5]
    assert sum(row["group_suppressed"] for row in q4) == 8
    assert sum(row["query_proposed"] for row in q4) == 10
    assert seal.arm_query_counts["Q4"] == 2


def test_queried_defect_updates_correction_but_never_normal_memory_or_coverage(tmp_path):
    runner, arms, _ = _make_runtime(tmp_path, _events(1), label=1)
    runner.run()
    for arm_id in ("Q4", "Q5", "Q6"):
        assert arms[arm_id].engine.c1_updates == 1
        assert arms[arm_id].engine.normal_writes == 0
        assert arms[arm_id].coverage.dynamic_size == 0


def test_unqueried_events_do_not_disclose_labels_or_supervised_write(tmp_path):
    runner, arms, _ = _make_runtime(tmp_path, _events(1))
    seal = runner.run()
    for arm_id in ("Q0", "Q1"):
        assert arms[arm_id].label_reader.accessed == []
        assert arms[arm_id].engine.c1_updates == 0
        assert arms[arm_id].engine.normal_writes == 0
        assert arms[arm_id].engine.unqueried_observations == seal.event_count


def test_queried_labels_are_written_only_to_separate_per_arm_private_files(tmp_path):
    events = _events(1)
    runner, arms, _ = _make_runtime(tmp_path, events)
    sinks = {}
    for arm_id in ARM_IDS:
        sink = PrivateFeedbackSink(tmp_path / f"private_{arm_id}.jsonl", arm_id)
        arms[arm_id].private_feedback = sink
        sinks[arm_id] = sink
    seal = runner.run()
    for sink in sinks.values():
        sink.close()
    public = Path(seal.score_log_path).read_text(encoding="utf-8")
    assert '"label"' not in public
    assert set(seal.private_feedback_sha256) == set(ARM_IDS)
    for arm_id, sink in sinks.items():
        lines = Path(sink.path).read_text(encoding="utf-8").splitlines()
        assert _sha_file(sink.path) == seal.private_feedback_sha256[arm_id]
        assert len(lines) == seal.arm_query_counts[arm_id]
        for line in lines:
            row = json.loads(line)
            assert row["arm_id"] == arm_id
            assert row["label"] == 0


def test_normal_feedback_updates_only_that_arms_coverage_and_next_threshold(tmp_path):
    runner, arms, _ = _make_runtime(tmp_path, _events(2), label=0, threshold_shift=10.0)
    seal = runner.run()
    assert arms["Q4"].coverage.dynamic_size == seal.arm_query_counts["Q4"]
    assert arms["Q0"].coverage.dynamic_size == 0
    rows = [json.loads(line) for line in Path(seal.score_log_path).read_text().splitlines()]
    assert rows[0]["arm_rows"]["Q4"]["b3_threshold"] == 0.0
    assert rows[1]["arm_rows"]["Q4"]["b3_threshold"] == 10.0
    assert rows[0]["arm_rows"]["Q4"]["prediction"] == 0


def test_stage_b_requires_all_complete_seals_before_touching_outcomes(tmp_path):
    runner, _arms, _ = _make_runtime(tmp_path, _events(1))
    seal = runner.run()
    with pytest.raises(E6ClosedLoopError, match="all four registered categories"):
        stage_b_evaluate(stage_a_seals={"audiojack": seal},
                         expected_categories=E6_REGISTERED_CATEGORIES,
                         label_sources={category: StageBLabelSource(
                             tmp_path / f"{category}.json", "a" * 64)
                             for category in E6_REGISTERED_CATEGORIES},
                         stage_b_authorized=True, authorization_id="synthetic-only")


def test_stage_b_disabled_by_default_and_never_receives_online_state(tmp_path):
    arms = None
    seals = {}
    for category in E6_REGISTERED_CATEGORIES:
        category_dir = tmp_path / category
        category_dir.mkdir()
        runner, current_arms, _ = _make_runtime(
            category_dir, _events(2), category=category)
        seals[category] = runner.run()
        if category == "audiojack":
            arms = current_arms
    with pytest.raises(E6ClosedLoopError, match="disabled"):
        stage_b_evaluate(stage_a_seals=seals,
                         expected_categories=E6_REGISTERED_CATEGORIES,
                         label_sources={category: StageBLabelSource(
                             tmp_path / f"{category}.json", "a" * 64)
                             for category in E6_REGISTERED_CATEGORIES},
                         stage_b_authorized=False)
    with pytest.raises(E6ClosedLoopError, match="private feedback seals are incomplete"):
        stage_b_evaluate(stage_a_seals=seals,
                         expected_categories=E6_REGISTERED_CATEGORIES,
                         label_sources={category: StageBLabelSource(
                             tmp_path / f"{category}.json", "a" * 64)
                             for category in E6_REGISTERED_CATEGORIES},
                         stage_b_authorized=True,
                         authorization_id="synthetic-test-only")
    assert arms["Q4"].engine.normal_writes == seals["audiojack"].arm_query_counts["Q4"]


def test_stage_b_rejects_complete_in_memory_receipt_without_persisted_manifest_or_seal(tmp_path):
    score_path = tmp_path / "sealed_scores.jsonl"
    score_path.write_bytes(b"")
    feedback = {}
    for arm in ARM_IDS:
        feedback_path = tmp_path / f"private_feedback_{arm}.jsonl"
        feedback_path.write_bytes(b"")
        feedback[arm] = _sha("")
    diagnostics = {arm: {"arm_id": arm, "events_observed": 0} for arm in ARM_IDS}
    receipt = StageASeal(
        category="audiojack", event_count=0, score_log_path=str(score_path),
        score_log_sha256=_sha(""), private_feedback_sha256=feedback,
        arm_query_counts={arm: 0 for arm in ARM_IDS},
        arm_controller_diagnostics=diagnostics, complete=True)
    with pytest.raises(E6ClosedLoopError, match="durable Stage A manifest and seal are required"):
        _read_sealed_score_log(score_path, receipt)


def test_stage_b_tamper_or_inconsistent_query_count_fails_before_labels(tmp_path):
    seals = {}
    for category in E6_REGISTERED_CATEGORIES:
        category_dir = tmp_path / category
        category_dir.mkdir()
        runner, _arms, _ = _make_runtime(category_dir, _events(1), category=category)
        seals[category] = runner.run()
    audiojack_path = Path(seals["audiojack"].score_log_path)
    audiojack_path.write_text(audiojack_path.read_text() + " ")
    with pytest.raises(E6ClosedLoopError, match="digest"):
        stage_b_evaluate(stage_a_seals=seals,
                         expected_categories=E6_REGISTERED_CATEGORIES,
                         label_sources={category: StageBLabelSource(
                             tmp_path / f"{category}.json", "a" * 64)
                             for category in E6_REGISTERED_CATEGORIES},
                         stage_b_authorized=True,
                         authorization_id="synthetic-only")


def _stage_b_json_and_seal(tmp_path, category, *, group_count=52):
    from experiments.calibration import r7_shuffled_correction_4class as r7

    category_dir = tmp_path / category
    category_dir.mkdir(parents=True, exist_ok=True)
    source_records = []
    source_labels = []
    metadata_rows = []
    for group_index in range(group_count):
        sample = group_index + 1
        is_defect = group_index % 2 == 1
        official = (f"{category}__{sample:04d}_NG_TS" if is_defect else
                    f"{category}_{sample:04d}_OK")
        group_sha = hashlib.sha256(
            ("official-sample-group\0" + official).encode("utf-8")).hexdigest()
        for camera in range(1, 6):
            original = len(source_records)
            timestamp = f"20261104{original // 60:02d}{original % 60:02d}00"
            filename = (f"{official}_C{camera}_{timestamp}.jpg")
            relative = (f"NG/TS/S{sample:04d}/{filename}" if is_defect else
                        f"OK/S{sample:04d}/{filename}")
            label_name = "TS" if is_defect else "OK"
            source_records.append({"image_path": relative, "anomaly_class": label_name})
            source_labels.append(int(is_defect))
            metadata_rows.append({"position": original, "opaque_group": group_sha,
                                  "relative_id": relative, "camera_id": camera,
                                  "timestamp": timestamp, "image_path": relative})
    json_path = category_dir / "test.json"
    json_path.write_text(json.dumps({"meta": {"normal_class": "OK"},
                                     "test": source_records}), encoding="utf-8")
    events = make_e6_event_stream(metadata_rows)
    run_dir = category_dir / f"e6_bea_v2_stage-b-{category}"
    run_dir.mkdir()
    namespace = {"schema": "e6.bea_v2.run_namespace.v1",
                 "run_id": f"stage-b-{category}", "status_at_creation": "INCOMPLETE"}
    namespace_sha = _write_json_once(run_dir / "run_namespace.json", namespace)
    feedback_hashes = {}
    for arm in ARM_IDS:
        path = run_dir / f"private_feedback_{arm}.jsonl"
        path.write_bytes(b"")
        feedback_hashes[arm] = hashlib.sha256(b"").hexdigest()
    diagnostics = {arm: {"arm_id": arm, "events_observed": len(events)} for arm in ARM_IDS}
    score_rows = []
    for event in events:
        arm_rows = {}
        for arm_index, arm in enumerate(ARM_IDS):
            arm_rows[arm] = {
                "gc_score": float((event.position * 37 % 251) / 250.0),
                "b3_threshold": float(0.25 + 0.075 * arm_index),
                "query_executed": False,
            }
        score_rows.append({
            "schema": "e6.bea_v2.stage_a_score.v1", "position": event.position,
            "original_json_position": event.original_position,
            "event_id_sha256": event.event_id_sha256,
            "image_identity_sha256": event.image_identity_sha256,
            "source_group_sha256": event.source_group_sha256,
            "arm_rows": arm_rows,
        })
    score_path = run_dir / "sealed_scores.jsonl"
    score_payload = b"".join(_canonical_json(row) for row in score_rows)
    score_path.write_bytes(score_payload)
    score_sha = hashlib.sha256(score_payload).hexdigest()
    query_counts = {arm: 0 for arm in ARM_IDS}
    manifest_record = {
        "schema": "e6.bea_v2.run_manifest.v1", "run_id": f"stage-b-{category}",
        "namespace_record_sha256": namespace_sha, "category": category,
        "registered_categories": list(E6_REGISTERED_CATEGORIES),
        "status_at_creation": "INCOMPLETE", "stream_event_count": len(events),
        "support_and_c64_identity": {"category_json_sha256": _sha256_file(json_path)},
        "stage_b_enabled": False,
        "output_files": ["run_namespace.json", "run_manifest.json", "sealed_scores.jsonl",
                          *[f"private_feedback_{arm}.jsonl" for arm in ARM_IDS],
                          "stage_a_seal.json"],
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_sha = _write_json_once(manifest_path, manifest_record)
    seal_record = {
        "schema": "e6.bea_v2.stage_a_seal.v1", "status": "COMPLETE",
        "category": category, "event_count": len(events),
        "category_json_sha256": _sha256_file(json_path),
        "manifest_sha256": manifest_sha,
        "score_log": {"file": score_path.name, "sha256": score_sha},
        "feedback_artifacts": {arm: {"file": f"private_feedback_{arm}.jsonl",
                                      "sha256": feedback_hashes[arm]} for arm in ARM_IDS},
        "arm_query_counts": query_counts,
        "arm_controller_diagnostics": diagnostics,
    }
    seal_path = run_dir / "stage_a_seal.json"
    seal_sha = _write_json_once(seal_path, seal_record)
    seal = StageASeal(
        category=category, event_count=len(events), score_log_path=str(score_path),
        score_log_sha256=score_sha, private_feedback_sha256=feedback_hashes,
        arm_query_counts=query_counts, arm_controller_diagnostics=diagnostics,
        complete=True, manifest_path=str(manifest_path), manifest_sha256=manifest_sha,
        seal_path=str(seal_path), seal_sha256=seal_sha,
        category_json_sha256=_sha256_file(json_path))
    return seal, StageBLabelSource(json_path, _sha256_file(json_path)), events, source_labels


def _independent_metrics(y, pred, score):
    negatives = [i for i, value in enumerate(y) if value == 0]
    positives = [i for i, value in enumerate(y) if value == 1]
    fpr = sum(int(pred[i]) for i in negatives) / len(negatives)
    fnr = sum(int(not pred[i]) for i in positives) / len(positives)
    auc = sum((1.0 if score[p] > score[n] else
               0.5 if score[p] == score[n] else 0.0)
              for p in positives for n in negatives) / (len(positives) * len(negatives))
    thresholds = sorted(set(score), reverse=True)
    ap = 0.0
    previous_recall = 0.0
    for threshold in thresholds:
        predicted = [i for i, value in enumerate(score) if value >= threshold]
        tp = sum(y[i] == 1 for i in predicted)
        precision = tp / len(predicted)
        recall = tp / len(positives)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return {"fpr": fpr, "fnr": fnr, "ber": (fpr + fnr) / 2,
            "ba": 1 - (fpr + fnr) / 2, "auroc": auc, "ap": ap}


def test_stage_b_positive_path_maps_original_json_labels_to_sealed_shuffled_events(tmp_path):
    seals, sources, gold = {}, {}, {}
    for category in E6_REGISTERED_CATEGORIES:
        seal, source, events, labels = _stage_b_json_and_seal(tmp_path, category)
        seals[category], sources[category], gold[category] = seal, source, (events, labels)
    result = stage_b_evaluate(
        stage_a_seals=seals, expected_categories=E6_REGISTERED_CATEGORIES,
        label_sources=sources, stage_b_authorized=True, authorization_id="cpu-synthetic")
    category = "audiojack"
    metrics = result["per_category"][category]["Q3"]
    rows = [json.loads(line) for line in Path(seals[category].score_log_path).read_text().splitlines()]
    events, original_labels = gold[category]
    y = np.asarray([original_labels[event.original_position] for event in events], dtype=np.int8)
    score = np.asarray([row["arm_rows"]["Q3"]["gc_score"] for row in rows])
    pred = score >= np.asarray([row["arm_rows"]["Q3"]["b3_threshold"] for row in rows])
    expected = _independent_metrics(y.tolist(), pred.tolist(), score.tolist())
    for name in ("ber", "ba", "fpr", "fnr", "auroc", "ap"):
        assert metrics[name] == pytest.approx(expected[name], abs=1e-12)
    assert metrics["worst_256_window_ber"] is not None


def test_stage_b_worst_256_is_null_without_a_complete_mixed_class_window(tmp_path):
    seals, sources = {}, {}
    for category in E6_REGISTERED_CATEGORIES:
        seals[category], sources[category], _events_value, _labels = (
            _stage_b_json_and_seal(tmp_path, category, group_count=20))
    result = stage_b_evaluate(
        stage_a_seals=seals, expected_categories=E6_REGISTERED_CATEGORIES,
        label_sources=sources, stage_b_authorized=True, authorization_id="cpu-synthetic")
    for category in E6_REGISTERED_CATEGORIES:
        assert result["per_category"][category]["Q0"]["worst_256_window_ber"] is None
    assert result["category_equal_macro"]["Q0"]["macro_worst_256_window_ber"] is None


def test_stage_b_rejects_unlabeled_arrays_and_permuted_sources(tmp_path):
    seals, sources = {}, {}
    for category in E6_REGISTERED_CATEGORIES:
        seals[category], sources[category], _events_value, _labels = _stage_b_json_and_seal(
            tmp_path, category)
    with pytest.raises(E6ClosedLoopError, match="hashed TEST source descriptor"):
        stage_b_evaluate(stage_a_seals=seals,
                         expected_categories=E6_REGISTERED_CATEGORIES,
                         label_sources={**sources, "audiojack": [0, 1]},
                         stage_b_authorized=True, authorization_id="cpu-synthetic")
    source = sources["audiojack"]
    payload = json.loads(source.path.read_text(encoding="utf-8"))
    payload["test"][0]["anomaly_class"], payload["test"][5]["anomaly_class"] = (
        payload["test"][5]["anomaly_class"], payload["test"][0]["anomaly_class"])
    source.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(E6ClosedLoopError, match="source digest"):
        stage_b_evaluate(stage_a_seals=seals,
                         expected_categories=E6_REGISTERED_CATEGORIES,
                         label_sources=sources, stage_b_authorized=True,
                         authorization_id="cpu-synthetic")


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "reordered", "identity"])
def test_stage_b_explicit_identity_records_reject_missing_duplicate_or_mismatched_rows(corruption):
    rows = [{"position": i, "original_json_position": 10 + i,
             "event_id_sha256": _sha(f"e{i}"),
             "image_identity_sha256": _sha(f"i{i}"),
             "source_group_sha256": _sha(f"g{i}")} for i in range(3)]
    records = [EvaluationLabelRecord(
        position=i, original_json_position=10 + i,
        event_id_sha256=rows[i]["event_id_sha256"],
        image_identity_sha256=rows[i]["image_identity_sha256"],
        source_group_sha256=rows[i]["source_group_sha256"], label=i % 2)
        for i in range(3)]
    if corruption == "missing":
        records = records[:-1]
    elif corruption == "duplicate":
        records[1] = EvaluationLabelRecord(**{**records[1].__dict__,
                                               "original_json_position": 10})
    elif corruption == "reordered":
        records[0], records[1] = records[1], records[0]
    else:
        records[1] = EvaluationLabelRecord(**{**records[1].__dict__,
                                               "image_identity_sha256": "f" * 64})
    with pytest.raises(E6ClosedLoopError, match="Stage B"):
        _validate_stage_b_identity_records(rows, records)


def test_worst_256_ignores_partial_or_single_class_windows_and_returns_unavailable():
    y = np.asarray([0, 1] * 50, dtype=np.int8)
    pred = np.zeros_like(y)
    assert _worst_window_ber(y, pred) is None
    full = np.asarray([0, 1] * 128 + [0, 1] * 2, dtype=np.int8)
    assert _worst_window_ber(full, np.zeros_like(full)) == pytest.approx(0.5)


def _real_iad_synthetic_events():
    from experiments.calibration import r7_shuffled_correction_4class as r7
    group_key = "audiojack_0001_OK"
    group_sha = hashlib.sha256(("official-sample-group\0" + group_key).encode()).hexdigest()
    events, records = [], []
    for position, camera in enumerate(range(1, 6)):
        timestamp = f"202611040000{position:02d}"
        filename = f"audiojack_0001_OK_C{camera}_{timestamp}.jpg"
        relative = f"OK/S0001/{filename}"
        image_id = r7._identity_hash(relative)
        event_id = hashlib.sha256(
            f"r7-event\0{position}\0{image_id}\0{group_sha}".encode()).hexdigest()
        events.append(EventIdentity(
            position=position, original_position=position, event_id_sha256=event_id,
            image_identity_sha256=image_id, source_group_sha256=group_sha,
            relative_id=relative, camera_id=str(camera), timestamp=timestamp,
            image_path=relative))
        records.append({"image_path": relative, "anomaly_class": "OK",
                        "mask_path": {"this": "is skipped, never decoded"}})
    return tuple(events), {"meta": {"normal_class": "OK"}, "test": records}


def _write_final_record_whitespace_json(tmp_path, *, malformed_first_label=False):
    events, payload = _real_iad_synthetic_events()
    if malformed_first_label:
        # Stage A must not inspect this unqueried outcome.
        payload["test"][0]["anomaly_class"] = {"not": "a selected label"}
    encoded = json.dumps(payload, separators=(",", ":"))
    close_array = encoded.rfind("]")
    assert close_array >= 0
    encoded = encoded[:close_array] + "   " + encoded[close_array:]
    path = tmp_path / "synthetic_test_with_final_record_whitespace.json"
    path.write_text(encoded, encoding="utf-8")
    return path, events


def test_e6_bounded_label_decode_fixes_final_record_whitespace_and_preserves_mapping(tmp_path):
    import mmap

    from experiments.calibration import r7_shuffled_correction_4class as r7
    from experiments.calibration.stage1a_p12_label_provider import (
        QueriedLabelAccessError,
        _decode_selected_label,
    )

    path, _events_value = _write_final_record_whitespace_json(tmp_path)
    raw = path.read_bytes()
    bounds = r7._test_array_boundaries(raw)
    start, stop = bounds[-1]
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
        # Reproduce the historical failure: its final whitespace scan crosses
        # the selected object's stop and reaches whitespace before the array close.
        with pytest.raises(QueriedLabelAccessError, match="Unexpected bytes after"):
            _decode_selected_label(mapping, start, stop, "OK")
        assert _decode_selected_e6_label(mapping, start, stop, "OK") == 0

    normal = b'{"anomaly_class":"OK"}'
    assert _decode_selected_e6_label(normal, 0, len(normal), "OK") == 0
    defect = b'{"anomaly_class":"TS"}'
    assert _decode_selected_e6_label(defect, 0, len(defect), "OK") == 1


@pytest.mark.parametrize("record", [b'{"other":"value"}', b'{"anomaly_class":7}'])
def test_e6_bounded_label_decode_rejects_missing_or_malformed_outcome(record):
    from experiments.calibration.stage1a_p12_label_provider import QueriedLabelAccessError

    with pytest.raises(QueriedLabelAccessError):
        _decode_selected_e6_label(record, 0, len(record), "OK")


def test_stage_a_decodes_only_queried_record_after_sealed_receipt(tmp_path, monkeypatch):
    from experiments.calibration.e6_bea_closed_loop import SealedQueryReceipt
    from experiments.calibration import stage1a_p12_label_provider as historical
    from experiments.calibration import r7_shuffled_correction_4class as r7

    path, events = _write_final_record_whitespace_json(
        tmp_path, malformed_first_label=True)
    raw = path.read_bytes()
    bounds = r7._test_array_boundaries(raw)
    decoded_records = []
    original_decoder = historical._decode_selected_label

    def observe_decoder(data, start, stop, normal_class):
        decoded_records.append((data, start, stop))
        return original_decoder(data, start, stop, normal_class)

    monkeypatch.setattr(historical, "_decode_selected_label", observe_decoder)
    ledger = ScoreLedger(tmp_path / "stage_a_scores.jsonl")
    reader = OnDemandLabelReader(path, "audiojack", _sha_file(path), events, "Q4", ledger)
    try:
        forged = SealedQueryReceipt(
            "Q4", events[-1].position, events[-1].original_position,
            events[-1].event_id_sha256, events[-1].image_identity_sha256,
            events[-1].source_group_sha256, object())
        with pytest.raises(E6ClosedLoopError, match="authorized durable query seal"):
            reader.label_for(events[-1], forged)
        assert decoded_records == []

        for event in events:
            arm_rows = {arm: {"query_executed": arm == "Q4" and event.position == 4}
                        for arm in ARM_IDS}
            ledger.seal_event(event, arm_rows)
        receipt = ledger.query_receipt("Q4", events[-1])
        assert reader.label_for(events[-1], receipt) == 0
        assert reader.decoded_positions == [events[-1].position]
        assert len(decoded_records) == 1
        decoded_data, decoded_start, decoded_stop = decoded_records[0]
        assert isinstance(decoded_data, bytes)
        assert decoded_data == raw[bounds[-1][0]:bounds[-1][1]]
        assert (decoded_start, decoded_stop) == (0, len(decoded_data))
    finally:
        reader.close()
        ledger.close()


def test_stage_b_identity_loader_decodes_final_original_record_with_trailing_array_space(
        tmp_path, monkeypatch):
    from experiments.calibration import stage1a_p12_label_provider as historical
    from experiments.calibration.e6_bea_closed_loop import _load_stage_b_identity_records

    path, events = _write_final_record_whitespace_json(tmp_path)
    source = StageBLabelSource(path, _sha_file(path))
    score_rows = [{
        "position": event.position,
        "original_json_position": event.original_position,
        "event_id_sha256": event.event_id_sha256,
        "image_identity_sha256": event.image_identity_sha256,
        "source_group_sha256": event.source_group_sha256,
    } for event in events]
    decoded_records = []
    original_decoder = historical._decode_selected_label

    def observe_decoder(data, start, stop, normal_class):
        decoded_records.append(data)
        return original_decoder(data, start, stop, normal_class)

    monkeypatch.setattr(historical, "_decode_selected_label", observe_decoder)
    records = _load_stage_b_identity_records(source, "audiojack", score_rows)
    assert [record.position for record in records] == list(range(len(events)))
    assert [record.original_json_position for record in records] == list(range(len(events)))
    assert [record.label for record in records] == [0] * len(events)
    assert len(decoded_records) == len(events)
    assert all(isinstance(record, bytes) for record in decoded_records)


def test_on_demand_label_gate_requires_seal_identity_and_discloses_once(tmp_path):
    events, test_json = _real_iad_synthetic_events()
    json_path = tmp_path / "synthetic_test.json"
    json_path.write_text(json.dumps(test_json), encoding="utf-8")
    validate_test_image_projection(json_path, "audiojack", events)
    ledger = ScoreLedger(tmp_path / "scores.jsonl")
    arms = {arm: {"query_executed": arm == "Q4"} for arm in ARM_IDS}
    with pytest.raises(E6ClosedLoopError, match="no durably sealed"):
        ledger.query_receipt("Q4", events[0])
    ledger.seal_event(events[0], arms)
    reader = OnDemandLabelReader(json_path, "audiojack", _sha_file(json_path),
                                 events, "Q4", ledger)
    with pytest.raises(E6ClosedLoopError, match="no durably sealed"):
        ledger.query_receipt("Q0", events[0])
    wrong_event = EventIdentity(**{**events[0].__dict__, "source_group_sha256": "f" * 64})
    with pytest.raises(E6ClosedLoopError, match="identity"):
        ledger.query_receipt("Q4", wrong_event)
    wrong_original = EventIdentity(**{**events[0].__dict__, "original_position": 1})
    with pytest.raises(E6ClosedLoopError, match="identity"):
        ledger.query_receipt("Q4", wrong_original)
    receipt = ledger.query_receipt("Q4", events[0])
    assert reader.label_for(events[0], receipt) == 0
    assert reader.decoded_positions == [0]
    with pytest.raises(E6ClosedLoopError, match="duplicate label"):
        reader.label_for(events[0], receipt)
    reader.close()
    ledger.close()


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_normal_coverage_uses_four_supports_then_only_queried_normals():
    coverage = _coverage()
    initial_sha = coverage.state_sha256()
    pct, info = coverage.novelty_percentile(np.asarray([1, 0, 0, 0], dtype=np.float32))
    assert 0.0 <= pct <= 1.0 and info["support_size"] == 4
    assert info["dynamic_size"] == 0
    coverage.observe_normal(np.asarray([1, 1, 0, 0], dtype=np.float32))
    assert coverage.dynamic_size == 1
    assert coverage.state_sha256() != initial_sha


def test_coverage_percentile_matches_frozen_v22_reference():
    v22 = pytest.importorskip("test_phase3_AIF_simplified_v22")
    support = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0],
                          [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    ours = NormalCoverageMemory(support)
    reference = v22.NormalCoverageMemory.from_support_feats(
        [torch.from_numpy(row.copy()) for row in support])
    for query in (np.asarray([1, 0.1, 0, 0], dtype=np.float32),
                  np.asarray([0.1, 0.1, 0.9, 0], dtype=np.float32)):
        ours_pct, _ = ours.novelty_percentile(query)
        ref_pct, _ = reference.novelty_percentile(torch.from_numpy(query.copy()))
        assert ours_pct == pytest.approx(ref_pct, abs=1e-7)
    normal = np.asarray([0.7, 0.7, 0.0, 0.0], dtype=np.float32)
    ours.observe_normal(normal)
    reference.observe_normal(torch.from_numpy(normal.copy()))
    ours_pct, _ = ours.novelty_percentile(np.asarray([0.6, 0.8, 0, 0], dtype=np.float32))
    ref_pct, _ = reference.novelty_percentile(torch.tensor([0.6, 0.8, 0, 0]))
    assert ours_pct == pytest.approx(ref_pct, abs=1e-7)


def test_execution_identity_requires_reviewed_commit_branch_and_pinned_sources(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    expected = "2a911dda634a7fe8a73347d4b2016d20b7980461"
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[1:] == ["branch", "--show-current"]:
            output = "calibration-stage0-contract-20261029\n"
        elif args[1:] == ["rev-parse", "HEAD"]:
            output = expected + "\n"
        else:
            output = ""
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(subprocess, "run", fake_run)
    receipt = verify_e6_execution_identity(expected, root=root)
    assert receipt["approved_e6_commit"] == expected
    assert receipt["r7_helper_source_sha256"] == dict(R7_HELPER_SOURCE_SHA256)
    assert len(receipt["r7_helper_source_sha256"]) == 18
    assert len(calls) == 3

    with pytest.raises(E6ClosedLoopError, match="branch/HEAD"):
        verify_e6_execution_identity("f" * 40, root=root)

    import experiments.calibration.e6_bea_closed_loop as e6_module
    original_sha_file = e6_module._sha256_file
    monkeypatch.setattr(e6_module, "_sha256_file", lambda _path: "0" * 64)
    with pytest.raises(E6ClosedLoopError, match="source SHA changed"):
        verify_e6_execution_identity(expected, root=root)
    monkeypatch.setattr(e6_module, "_sha256_file", original_sha_file)

    def wrong_branch(args, **kwargs):
        if args[1:] == ["branch", "--show-current"]:
            output = "main\n"
        elif args[1:] == ["rev-parse", "HEAD"]:
            output = expected + "\n"
        else:
            output = ""
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(subprocess, "run", wrong_branch)
    with pytest.raises(E6ClosedLoopError, match="branch/HEAD"):
        verify_e6_execution_identity(expected, root=root)

    def dirty_source(args, **kwargs):
        if args[1:] == ["branch", "--show-current"]:
            output = "calibration-stage0-contract-20261029\n"
        elif args[1:] == ["rev-parse", "HEAD"]:
            output = expected + "\n"
        else:
            output = " M experiments/calibration/e6_bea_closed_loop.py\n"
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(subprocess, "run", dirty_source)
    with pytest.raises(E6ClosedLoopError, match="uncommitted changes"):
        verify_e6_execution_identity(expected, root=root)


def test_stage05_source_is_pinned_and_mismatched_module_or_digest_fails_closed(monkeypatch):
    import experiments.calibration.e6_bea_closed_loop as e6
    from experiments.calibration import stage05

    root = Path(__file__).resolve().parents[2]
    expected = "a" * 40
    relative = "experiments/calibration/stage05.py"
    source_path = root / relative
    expected_sha = "4b8aae252f5f1e096b127906794064da269aa79f7cd5514f6f738cda585c017e"
    assert e6.R7_HELPER_SOURCE_SHA256[relative] == expected_sha
    assert _sha256_file(source_path) == expected_sha

    def fake_run(args, **_kwargs):
        if args[1:] == ["branch", "--show-current"]:
            output = "calibration-stage0-contract-20261029\n"
        elif args[1:] == ["rev-parse", "HEAD"]:
            output = expected + "\n"
        else:
            output = ""
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(subprocess, "run", fake_run)
    receipt = verify_e6_execution_identity(expected, root=root)
    assert receipt["r7_helper_source_sha256"][relative] == expected_sha
    e6._assert_module_source(stage05, relative, root=root, role="Stage 0.5 helper")
    with pytest.raises(E6ClosedLoopError, match="pinned source"):
        e6._assert_module_source(
            SimpleNamespace(__file__=str(root / "experiments/calibration/r7_shuffled_correction_4class.py")),
            relative, root=root, role="Stage 0.5 helper")

    original_sha_file = e6._sha256_file

    def mismatched_stage05(path):
        if Path(path).resolve() == source_path.resolve():
            return "0" * 64
        return original_sha_file(path)

    monkeypatch.setattr(e6, "_sha256_file", mismatched_stage05)
    with pytest.raises(E6ClosedLoopError, match="stage05.py"):
        verify_e6_execution_identity(expected, root=root)


def test_native_dense_score_and_common_greedy_writer_use_r7_helpers_on_synthetic_inputs():
    from experiments.calibration import r1_matched_correction_pilot as r1

    class SyntheticBank:
        corr_cfg = SimpleNamespace(enabled=False)
        use_defect_bank = False

        def score_single_view(self, **kwargs):
            assert kwargs["cls_name"] == "p0" and kwargs["view"] == "id"
            amap = torch.arange(784, dtype=torch.float32)
            summary = {"layer0": {"topk_patch_idx": np.arange(32, dtype=np.int64),
                                  "topk_dN_max": 0.8, "topk_dN_mean": 0.2}}
            return [amap], summary

        def fuse_layers(self, maps, fusion):
            assert fusion == "mean"
            return maps[0]

        def aggregate_map(self, amap, **kwargs):
            assert kwargs["agg"] == "max"
            return float(torch.max(amap))

    class SyntheticImpl:
        @staticmethod
        def _grid_hw_from_npatches(count, aspect=None):
            assert count == 784
            return 28, 28

        @staticmethod
        def ViewSpec(source, target):
            return (source, target)

        @staticmethod
        def align_map_to_identity(amap, view_spec, height, width):
            assert view_spec == ("id", "id") and (height, width) == (28, 28)
            return amap

    cfg = SimpleNamespace(topk_patches=32, image_aggregation="max",
                          image_topk=1, image_quantile=0.99)
    base_map, top_ids, score, summary = r1._native_dense_score(
        SyntheticImpl(), SyntheticBank(), SimpleNamespace(height=28, width=28),
        [torch.zeros((784, 3), dtype=torch.float32)], cfg)
    assert base_map.shape == (784,) and np.array_equal(top_ids, np.arange(32))
    assert score == float(np.max(base_map)) == 783.0
    assert summary["layer0"]["topk_dN_max"] == 0.8

    candidate_ids = np.arange(32, dtype=np.int64)
    candidate_features = np.ones((32, 3), dtype=np.float32)
    calls = []

    def eligibility(**kwargs):
        calls.append(("eligibility", kwargs["score"], kwargs["theta"]))
        return True, 32, "synthetic-eligible", {}

    def update(**kwargs):
        calls.append(("update", kwargs["position"], kwargs["score"]))
        return {"accepted_patches_delta": 1}

    action = r1.common_c0_greedy_writer_action(
        position=0, c0_score=0.25, theta=0.30, sigma=0.10,
        c0_prediction=0, candidate_ids=candidate_ids,
        candidate_features=candidate_features,
        feature_layers=[torch.zeros((784, 3), dtype=torch.float32)],
        queried=True, disclosed_label=0, eligibility=eligibility, update=update)
    assert action["action"] == "native_greedy_normal_update"
    assert [item[0] for item in calls] == ["eligibility", "update"]
    calls.clear()
    defect = r1.common_c0_greedy_writer_action(
        position=1, c0_score=0.25, theta=0.30, sigma=0.10,
        c0_prediction=0, candidate_ids=candidate_ids,
        candidate_features=candidate_features,
        feature_layers=[torch.zeros((784, 3), dtype=torch.float32)],
        queried=True, disclosed_label=1, eligibility=eligibility, update=update)
    assert defect["action"] == "verified_defect_no_normal_write"
    assert calls == []
    unqueried = r1.common_c0_greedy_writer_action(
        position=2, c0_score=0.25, theta=0.30, sigma=0.10,
        c0_prediction=0, candidate_ids=candidate_ids,
        candidate_features=candidate_features,
        feature_layers=[torch.zeros((784, 3), dtype=torch.float32)],
        queried=False, disclosed_label=None, eligibility=eligibility, update=update)
    assert unqueried["action"] == "no_normal_memory_write"
    assert calls == []


def test_native_c1_residuals_use_r7_causal_source_exclusion_on_cpu():
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration import r7_shuffled_correction_4class as r7
    v22 = pytest.importorskip("test_phase3_AIF_simplified_v22")

    cfg = v22.CorrectionConfig(enabled=False, K_correction=1024)
    codebook = v22.SignedCorrectionCodebook(1024, 4, torch.device("cpu"), cfg)
    codebook.n = 1
    codebook.protos[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    codebook.beta[0] = 0.10
    codebook.radius[0] = 0.10
    codebook.kappa[0] = 1.0
    codebook.count[0] = 1
    codebook.last_step[0] = 0
    bank = SimpleNamespace(corr_cb={"p0": {"id": [codebook]}},
                           corr_cfg=cfg, device=torch.device("cpu"))
    current_group = "b" * 64
    anchor_group = "a" * 64
    selected = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 32)

    delta, diagnostics = r1._c1_score_delta(
        v22, bank, selected, {0: {anchor_group}}, current_group, event_position=1)
    assert delta.shape == (32,) and np.all(delta > 0.09)
    assert diagnostics["eligible_anchor_slots"] == 1
    assert diagnostics["same_source_excluded_anchor_slots"] == 0

    excluded, diagnostics = r1._c1_score_delta(
        v22, bank, selected, {0: {current_group}}, current_group, event_position=1)
    assert np.array_equal(excluded, np.zeros(32, dtype=np.float32))
    assert diagnostics["eligible_anchor_slots"] == 0
    assert diagnostics["same_source_excluded_anchor_slots"] == 1
