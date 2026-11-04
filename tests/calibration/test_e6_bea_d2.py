import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.calibration import e6_bea_d2 as d2
from experiments.calibration.e6_bea_closed_loop import (
    ArmRuntime,
    EncodedEvent,
    EventIdentity,
    FeedbackUpdate,
    NativePreEventScore,
    NormalCoverageMemory,
    _canonical_json,
    _sha256_file,
)
from experiments.calibration.e6_bea_d2_runner import _initial_states
from experiments.calibration.e6_bea_policy import (
    QueryController,
    _stable_uniform01,
    compute_bea_components,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _components(score: float = -0.001, threshold: float = 0.0):
    return compute_bea_components(
        score=score,
        threshold=threshold,
        c64_scores=np.linspace(-0.001, 0.001, 64),
        novelty_percentile=0.95,
        correction_top32=np.zeros(32),
        correction_bank_size=0,
        topk_dN_max=0.8,
        topk_dN_mean=0.2,
    )


def _event(position: int, group: int | None = None) -> EventIdentity:
    group = position if group is None else group
    return EventIdentity(
        position=position,
        original_position=1000 + position,
        event_id_sha256=_sha(f"event-{position}"),
        image_identity_sha256=_sha(f"image-{position}"),
        source_group_sha256=_sha(f"group-{group}"),
        relative_id=f"synthetic/audiojack/{position}.png",
        camera_id=str(position % 5 + 1),
        timestamp=f"202611040000{position:02d}",
        image_path=f"synthetic/audiojack/{position}.png",
    )


def test_d2_registry_and_weighted_objective_subsets_are_exact():
    assert d2.D2_ARM_IDS == (
        "B", "BN", "BD", "BND", "BNCD", "RAND37", "PERIOD27",
        "MARGIN37", "ENTROPY37", "NOVELTY_NN37",
    )
    comp = _components()
    for arm, terms in d2.D2_BEA_TERMS.items():
        assert d2._objective(comp, arm) == pytest.approx(
            sum(getattr(comp, term) for term in terms), abs=1e-12)
    assert (comp.B, comp.N, comp.C, comp.D) == pytest.approx(
        (comp.R + 0.30 * comp.H, 0.35 * comp.U_N,
         0.15 * comp.U_C, 0.35 * comp.U_D))


@pytest.mark.parametrize(
    "d2_arm,e6_arm",
    [("B", "Q4"), ("BND", "Q5"), ("BNCD", "Q6")],
)
def test_d2_bea_price_and_executed_flag_update_match_original_e6(d2_arm, e6_arm):
    d2_controller = d2.D2PolicyController(d2_arm, seed=37)
    e6_controller = QueryController(e6_arm, seed=37)
    components = [
        _components(score=-0.001),
        _components(score=0.0),
        _components(score=0.003),
        _components(score=-0.02),
    ]
    for position, comp in enumerate(components):
        event_hash = _sha(f"policy-event-{position}")
        actual = d2_controller.propose(
            position=position, event_id_sha256=event_hash,
            components=comp, novelty_distance_nn=0.2)
        expected = e6_controller.propose(
            position=position, event_id_sha256=event_hash,
            components=comp)
        assert actual.objective == pytest.approx(expected.objective, abs=1e-12)
        assert actual.query_cost == pytest.approx(expected.query_cost, abs=1e-12)
        assert actual.proposed is expected.proposed
        assert (actual.objective > actual.query_cost) is actual.proposed
        suppressed = bool(actual.proposed and position == 1)
        executed = bool(actual.proposed and not suppressed)
        next_actual = d2_controller.observe_final_action(
            position=position, proposal=actual,
            executed_query=executed, group_suppressed=suppressed)
        next_expected = e6_controller.observe_final_action(
            position=position, proposal=expected,
            executed_query=executed, group_suppressed=suppressed)
        assert next_actual == pytest.approx(next_expected, abs=1e-12)
        assert d2_controller.query_cost_runtime == pytest.approx(
            e6_controller.query_cost_runtime, abs=1e-12)


def test_random_periodic_and_past_only_percentile_baselines():
    event_hash = _sha("rand-event")
    random_controller = d2.D2PolicyController("RAND37", seed=37)
    random_proposal = random_controller.propose(
        position=0, event_id_sha256=event_hash,
        components=_components(), novelty_distance_nn=0.0)
    expected_u = _stable_uniform01(37, "RAND37", event_hash, "random")
    assert random_proposal.random_u == expected_u
    assert random_proposal.proposed is (
        expected_u < d2.D2_RANDOM_VIEW_PROBABILITY)
    assert d2.D2_RANDOM_VIEW_PROBABILITY == pytest.approx(
        1 - (1 - 5 * 0.037) ** (1 / 5))

    periodic = d2.D2PolicyController("PERIOD27")
    decisions = []
    for position in range(27):
        proposal = periodic.propose(
            position=position, event_id_sha256=_sha(f"period-{position}"),
            components=_components(), novelty_distance_nn=0.0)
        decisions.append(proposal.proposed)
        periodic.observe_final_action(
            position=position, proposal=proposal,
            executed_query=proposal.proposed, group_suppressed=False)
    assert decisions == [position == 26 for position in range(27)]

    margin = d2.D2PolicyController("MARGIN37", seed=37)
    entropy = d2.D2PolicyController("ENTROPY37", seed=37)
    cold_comps = SimpleNamespace(psi=0.4, entropy=0.62)
    for position in range(d2.UTILITY_COLD_START_EVENTS):
        identity = _sha(f"cold-{position}")
        p_margin = margin.propose(position=position, event_id_sha256=identity,
                                  components=cold_comps, novelty_distance_nn=0.0)
        p_entropy = entropy.propose(position=position, event_id_sha256=identity,
                                    components=cold_comps, novelty_distance_nn=0.0)
        assert p_margin.proposed is p_entropy.proposed
        assert p_margin.random_u == p_entropy.random_u
        margin.observe_final_action(position=position, proposal=p_margin,
                                    executed_query=p_margin.proposed,
                                    group_suppressed=False)
        entropy.observe_final_action(position=position, proposal=p_entropy,
                                     executed_query=p_entropy.proposed,
                                     group_suppressed=False)
    assert len(margin.utility_history) == 64
    assert len(entropy.utility_history) == 64

    # At a tied all-zero or all-one maximum, the registered tie lottery keeps
    # a nonzero 3.7% chance instead of strict-max starvation.
    for arm, utility_name, utility_value in (
            ("MARGIN37", "psi", 0.0), ("ENTROPY37", "entropy", 1.0)):
        controller = d2.D2PolicyController(arm, seed=37)
        controller.last_observed_position = 255
        controller.utility_history.extend([utility_value] * 256)
        event = _sha(f"tie-{arm}")
        comp = SimpleNamespace(psi=utility_value, entropy=utility_value)
        proposal = controller.propose(
            position=256, event_id_sha256=event, components=comp,
            novelty_distance_nn=0.0)
        assert proposal.reason == "q963_identity_hashed_tie_lottery"
        assert proposal.n_eq == 256
        assert proposal.tie_probability == pytest.approx(0.037)
        assert proposal.proposed is (proposal.tie_u < 0.037)
        assert proposal.history_count == 256

    novelty = d2.D2PolicyController("NOVELTY_NN37")
    novelty.last_observed_position = 63
    novelty.utility_history.extend([0.1] * 64)
    novelty_proposal = novelty.propose(
        position=64, event_id_sha256=_sha("raw-nn"),
        components=cold_comps, novelty_distance_nn=100.0)
    assert novelty_proposal.utility == 100.0
    assert novelty_proposal.proposed is True

    past_only = d2.D2PolicyController("MARGIN37")
    past_only.last_observed_position = 63
    past_only.utility_history.extend([0.1] * 64)
    proposal = past_only.propose(
        position=64, event_id_sha256=_sha("past-only"),
        components=SimpleNamespace(psi=0.9, entropy=0.0),
        novelty_distance_nn=0.0)
    assert proposal.history_count == 64
    assert proposal.threshold == pytest.approx(0.1)
    assert len(past_only.utility_history) == 64
    past_only.observe_final_action(
        position=64, proposal=proposal, executed_query=proposal.proposed,
        group_suppressed=False)
    assert len(past_only.utility_history) == 65


class _FakeEngine:
    def __init__(self, arm: str, trace: list[tuple]):
        self.arm = arm
        self.trace = trace
        self.initial_state_sha256 = _sha("same-initial-scorer-state")
        self.fixed_support_sha256 = _sha("fixed-bank")
        self.c64_group_sha256 = _sha("c64-groups")
        self.input_identity = {
            "category_json_sha256": _sha("synthetic-category-json"),
            "c64_score_archive_sha256": _sha("c64-archive"),
        }
        self.c64_initial_scores = tuple(np.linspace(0.0, 1.0, 64))
        self.initial_b3_c64_initialization = {"synthetic": True}
        self.current_threshold = 0.0
        self.state_version = 0
        self._mutable_objects = [object() for _ in range(3)]
        self.owned_mutable_state_ids = frozenset(id(value) for value in self._mutable_objects)
        self.scored_versions: dict[int, int] = {}
        self.normal_writes = 0
        self.unqueried_calls = 0

    @property
    def state_digest(self):
        return _sha(f"{self.initial_state_sha256}-{self.state_version}")

    def score(self, event, encoded):
        self.trace.append(("score", event.position, self.arm))
        self.scored_versions[event.position] = self.state_version
        return NativePreEventScore(
            gc_score=self.current_threshold - 0.001,
            c0_score=0.2,
            c64_scores=tuple(np.linspace(-0.001, 0.001, 64)),
            correction_top32=tuple(np.zeros(32)),
            correction_bank_size=0,
            topk_dN_max=0.8,
            topk_dN_mean=0.2,
            scorer_state_version=self.state_version,
            scorer_state_sha256=self.state_digest,
        )

    def observe_unqueried(self, event, encoded, score):
        self.unqueried_calls += 1
        return {"action": "no_feedback_update"}

    def apply_feedback(self, event, encoded, score, label):
        if label == 0:
            self.normal_writes += 1
        self.state_version += 1
        self.current_threshold += 0.0001
        return FeedbackUpdate(
            label=label, c1_changed=True, greedy_changed=(label == 0),
            scorer_state_version=self.state_version,
            scorer_state_sha256=self.state_digest,
            scorer_state_changed=True,
            diagnostics={"normal_write": label == 0},
        )


class _AuditReader:
    def __init__(self, arm: str, ledger: d2.D2ScoreLedger):
        self.arm = arm
        self.ledger = ledger
        self.accessed: list[int] = []

    def label_for(self, event, receipt):
        assert receipt.arm_id == self.arm
        assert self.ledger.is_sealed(event.position)
        assert self.ledger.authorize_receipt(receipt, self.arm, event)
        rows = [json.loads(line) for line in self.ledger.path.read_text(
            encoding="utf-8").splitlines()]
        row = rows[event.position]
        assert set(row["arm_rows"]) == set(d2.D2_ARM_IDS)
        assert all("label" not in arm_row for arm_row in row["arm_rows"].values())
        self.accessed.append(event.position)
        return 0

    def close(self):
        pass


class _TracingController(d2.D2PolicyController):
    def __init__(self, arm_id: str, trace: list[tuple]):
        super().__init__(arm_id, seed=37)
        self.trace = trace

    def propose(self, **kwargs):
        self.trace.append(("proposal", kwargs["position"], self.arm_id))
        return super().propose(**kwargs)


def test_ten_arm_joint_seal_feedback_gate_and_t_plus_one_state(tmp_path, monkeypatch):
    monkeypatch.setitem(d2.D2_EXPECTED_COUNTS, "audiojack", 3)
    events = (_event(0, 0), _event(1, 0), _event(2, 2))
    support = np.eye(4, dtype=np.float32)
    trace: list[tuple] = []
    ledger = d2.D2ScoreLedger(tmp_path / "sealed_scores.jsonl")
    namespace_path = tmp_path / "run_namespace.json"
    namespace = {"run_id": "synthetic-a1", "category": "audiojack",
                 "status_at_creation": "INCOMPLETE"}
    d2._write_json_once(namespace_path, namespace)
    output_files = {
        "run_namespace.json", "run_manifest.json", "sealed_scores.jsonl",
        "stage_a_seal.json", *[f"private_feedback_{arm}.jsonl" for arm in d2.D2_ARM_IDS],
    }
    manifest = {
        "schema": d2.D2_MANIFEST_SCHEMA,
        "status_at_creation": "INCOMPLETE",
        "stage_b_enabled": False,
        "run_id": "synthetic-a1", "category": "audiojack",
        "approved_commit": "a" * 40, "contract_sha256": _sha("contract"),
        "stream_event_count": 3, "registered_arm_ids": list(d2.D2_ARM_IDS),
        "namespace_record_sha256": _sha256_file(namespace_path),
        "input_identity": {"category_json_sha256": _sha("synthetic-category-json")},
        "output_files": sorted(output_files),
    }
    manifest_path = tmp_path / "run_manifest.json"
    d2._write_json_once(manifest_path, manifest)
    manifest_sha = _sha256_file(manifest_path)
    arms = {}
    engines = {}
    readers = {}
    coverages = {}
    for arm in d2.D2_ARM_IDS:
        engine = _FakeEngine(arm, trace)
        reader = _AuditReader(arm, ledger)
        sink = d2.D2PrivateFeedbackSink(tmp_path / f"private_feedback_{arm}.jsonl", arm)
        coverage = NormalCoverageMemory(support)
        engines[arm], readers[arm], coverages[arm] = engine, reader, coverage
        arms[arm] = ArmRuntime(
            arm_id=arm, engine=engine,
            controller=_TracingController(arm, trace),
            coverage=coverage, label_reader=reader, private_feedback=sink,
        )
    initial = _initial_states(engines, coverages)
    assert len(initial) == 10
    assert len({id(engines[arm]) for arm in d2.D2_ARM_IDS}) == 10
    assert len({id(coverages[arm]) for arm in d2.D2_ARM_IDS}) == 10
    assert len({id(arms[arm].controller) for arm in d2.D2_ARM_IDS}) == 10
    assert len({row["scorer_state_sha256"] for row in initial.values()}) == 1

    encoded_events = []

    def encode(event):
        encoded = EncodedEvent(
            image=None, feature_layers=(),
            global_cls=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        encoded_events.append((event.position, encoded))
        return encoded

    coordinator = d2.D2StageAClosedLoop(
        category="audiojack", run_id="synthetic-a1", events=events,
        feature_encoder=encode, arms=arms, ledger=ledger,
        output_dir=tmp_path, manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        execution_authorization_id="SYNTHETIC-TEST-ONLY",
    )
    coordinator.run()
    assert len(encoded_events) == len(events)

    for position in range(3):
        score_indices = [i for i, row in enumerate(trace)
                         if row[0] == "score" and row[1] == position]
        proposal_indices = [i for i, row in enumerate(trace)
                            if row[0] == "proposal" and row[1] == position]
        assert len(score_indices) == len(proposal_indices) == 10
        assert max(score_indices) < min(proposal_indices)
    rows = [json.loads(line) for line in (tmp_path / "sealed_scores.jsonl").read_text(
        encoding="utf-8").splitlines()]
    for arm in d2.D2_ARM_IDS:
        expected_access = [row["position"] for row in rows
                           if row["arm_rows"][arm]["query_executed"]]
        assert readers[arm].accessed == expected_access
        assert engines[arm].normal_writes == len(expected_access)
        assert engines[arm].unqueried_calls == len(events) - len(expected_access)
        assert all(row["arm_rows"][arm]["scorer_state_version"] == 0
                   for row in rows[:1])
        assert rows[1]["arm_rows"][arm]["scorer_state_version"] == int(
            rows[0]["arm_rows"][arm]["query_executed"])
    assert any(rows[0]["arm_rows"][arm]["query_executed"] for arm in d2.D2_BEA_TERMS)
    assert any(rows[1]["arm_rows"][arm]["group_suppressed"] for arm in d2.D2_BEA_TERMS)
    assert all("label" not in row for row in rows)

    run_dir = tmp_path / "e6_bea_d2_synthetic-a1_audiojack"
    run_dir.mkdir()
    # Move just the synthetic test outputs into the namespace expected by the
    # fail-closed validator; no repository or historical artifact is touched.
    for path in list(tmp_path.iterdir()):
        if path == run_dir:
            continue
        path.rename(run_dir / path.name)
    validated = d2.validate_d2_stage_a(
        run_dir=run_dir, category="audiojack", run_id="synthetic-a1",
        expected_commit="a" * 40, contract_sha256=_sha("contract"),
        expected_events=events,
    )
    assert len(validated.rows) == 3
    assert validated.seal_sha256 == _sha256_file(run_dir / "stage_a_seal.json")
    for filename, error_match in (
            ("private_feedback_B.jsonl", "private feedback hash"),
            ("sealed_scores.jsonl", "completion seal"),
            ("run_manifest.json", "completion seal")):
        path = run_dir / filename
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with pytest.raises(d2.D2Error, match=error_match):
            d2.validate_d2_stage_a(
                run_dir=run_dir, category="audiojack", run_id="synthetic-a1",
                expected_commit="a" * 40, contract_sha256=_sha("contract"),
                expected_events=events,
            )
        path.write_bytes(original)
    seal_path = run_dir / "stage_a_seal.json"
    bad_seal = json.loads(seal_path.read_text(encoding="utf-8"))
    bad_seal["status"] = "INCOMPLETE"
    seal_path.write_bytes(_canonical_json(bad_seal))
    with pytest.raises(d2.D2Error, match="completion seal"):
        d2.validate_d2_stage_a(
            run_dir=run_dir, category="audiojack", run_id="synthetic-a1",
            expected_commit="a" * 40, contract_sha256=_sha("contract"),
            expected_events=events,
        )


def test_stage_b_rejects_missing_full_seal_before_reading_any_labels():
    incomplete = {
        "audiojack": d2.D2ValidatedRun(
            category="audiojack", run_id="partial", run_dir=Path("unused"),
            manifest={}, seal={"status": "COMPLETE"}, rows=(),
            manifest_sha256="0" * 64, seal_sha256="0" * 64,
            score_log_sha256="0" * 64),
    }
    with pytest.raises(d2.D2Error, match="exact registered authorization"):
        d2.authorize_d2_stage_b(runs=incomplete, authorization_id="")
    with pytest.raises(d2.D2Error, match="all four registered"):
        d2.authorize_d2_stage_b(
            runs=incomplete, authorization_id=d2.D2_STAGE_B_AUTHORIZATION_ID)
    with pytest.raises(d2.D2Error, match="all four registered"):
        d2.d2_stage_b_evaluate(
            runs=incomplete, labels_by_category=None,
            authorization_id=d2.D2_STAGE_B_AUTHORIZATION_ID)


def test_stage_b_reuses_e6_metrics_after_all_synthetic_seals_verify(monkeypatch, tmp_path):
    monkeypatch.setitem(d2.D2_EXPECTED_COUNTS, "audiojack", 256)
    monkeypatch.setitem(d2.D2_EXPECTED_COUNTS, "regulator", 256)
    monkeypatch.setitem(d2.D2_EXPECTED_COUNTS, "switch", 256)
    monkeypatch.setitem(d2.D2_EXPECTED_COUNTS, "rolled_strip_base", 256)
    runs = {}
    labels = {}
    for category in d2.E6_REGISTERED_CATEGORIES:
        counts = {arm: 16 for arm in d2.D2_ARM_IDS}
        diagnostics = {
            arm: {"executed_queries": counts[arm], "proposals": 18,
                  "group_suppressed": 2, "quantile_comparisons": 192,
                  "quantile_ties": 10, "tie_lottery_calls": 10,
                  "max_query_free_interval_events": 32,
                  "max_query_gap": 33, "query_cost_runtime": 0.8,
                  "query_free_interval_lengths": [16, 224]}
            for arm in d2.D2_ARM_IDS}
        rows = []
        for position in range(256):
            arm_rows = {}
            for ai, arm in enumerate(d2.D2_ARM_IDS):
                score = float(position / 255)
                threshold = 0.5
                arm_rows[arm] = {
                    "gc_score": score, "c0_score": score,
                    "b3_threshold": threshold,
                    "prediction": int(score >= threshold),
                    "correction_bank_size": position % 5,
                    "native_score_seconds": 0.001,
                    "query_executed": position % 16 == 0,
                    "query_utility": score,
                    "query_threshold": 0.5,
                }
            rows.append({"position": position, "arm_rows": arm_rows})
        # Diagnostics and ledger counts must agree for all ten arms.
        counts = {arm: 16 for arm in d2.D2_ARM_IDS}
        seal = {"status": "COMPLETE", "arm_query_counts": counts,
                "arm_controller_diagnostics": diagnostics}
        runs[category] = d2.D2ValidatedRun(
            category=category, run_id="synthetic", run_dir=tmp_path / category,
            manifest={}, seal=seal, rows=tuple(rows),
            manifest_sha256=_sha(f"{category}-manifest"),
            seal_sha256=_sha(f"{category}-seal"),
            score_log_sha256=_sha(f"{category}-score"))
        labels[category] = np.asarray([position % 2 for position in range(256)],
                                      dtype=np.int8)
    evaluation = d2.d2_stage_b_evaluate(
        runs=runs, labels_by_category=labels,
        authorization_id=d2.D2_STAGE_B_AUTHORIZATION_ID)
    assert evaluation["primary_metric"] == "category_equal_macro_prequential_ber"
    assert evaluation["category_metrics"]["audiojack"]["B"]["worst_256_window_ber"] is not None
    assert len(evaluation["category_metrics"]["audiojack"]["B"]["four_segment_temporal"]) == 4
    assert evaluation["category_metrics"]["audiojack"]["B"][
        "query_free_interval_lengths"] == [16, 224]


def test_budget_fairness_and_factorial_contrasts_are_explicit():
    categories = {}
    macro = {}
    for ci, category in enumerate(d2.E6_REGISTERED_CATEGORIES):
        categories[category] = {}
        for ai, arm in enumerate(d2.D2_ARM_IDS):
            categories[category][arm] = {
                "ber": 0.20 + 0.01 * ai + 0.001 * ci,
                "query_rate": 0.03 + 0.001 * ai,
                "query_count": 100 + ai,
                "fpr": 0.1, "fnr": 0.3, "auroc": 0.7 + ai * 0.001,
                "ap": 0.6 + ai * 0.001,
                "proposal_count": 110 + ai,
                "source_group_suppressed": 2,
                "quantile_ties": 3,
                "max_query_free_interval_events": 100,
                "query_free_interval_lengths": [10, 20],
            }
    for arm in d2.D2_ARM_IDS:
        macro[arm] = {
            key: float(np.mean([categories[cat][arm][key]
                                for cat in d2.E6_REGISTERED_CATEGORIES])
                        if key != "query_count" else
                        sum(categories[cat][arm][key]
                            for cat in d2.E6_REGISTERED_CATEGORIES))
            for key in ("ber", "query_rate", "fpr", "fnr", "auroc", "ap")
        }
        macro[arm]["query_count_sum"] = int(sum(
            categories[cat][arm]["query_count"] for cat in d2.E6_REGISTERED_CATEGORIES))
    analysis = d2.factorial_and_policy_analysis({
        "category_metrics": categories, "category_equal_macro": macro,
    })
    assert analysis["N_by_D_factorial"]["CATEGORY_EQUAL_MACRO"]["N_main_improvement"] == pytest.approx(-0.01)
    assert "classification" in analysis["N_by_D_factorial"][
        "CATEGORY_EQUAL_MACRO"]["budget_match"]["B_vs_BN"]
    assert analysis["C_on_vs_C_off"]["macro_budget_classification"]
    assert analysis["primary_BND_vs_simple"]["macro_budget_classification"]
    assert d2.classify_budget(rate_a=0.03, ber_a=0.1,
                              rate_b=0.04, ber_b=0.2) == (
        "BUDGET_DOMINANCE_EVIDENCE_IN_A_FAVOR")


def test_coverage_and_score_state_objects_are_independent_but_equivalent():
    engines = {arm: _FakeEngine(arm, []) for arm in d2.D2_ARM_IDS}
    coverages = {arm: NormalCoverageMemory(np.eye(4, dtype=np.float32))
                 for arm in d2.D2_ARM_IDS}
    states = _initial_states(engines, coverages)
    assert len(states) == 10
    assert len({id(engine._mutable_objects[0]) for engine in engines.values()}) == 10
    assert len({coverage.state_sha256() for coverage in coverages.values()}) == 1
    shared = engines["B"]._mutable_objects[0]
    engines["BN"]._mutable_objects = [shared]
    engines["BN"].owned_mutable_state_ids = frozenset({id(shared)})
    with pytest.raises(d2.D2Error, match="share a mutable native state"):
        _initial_states(engines, coverages)
