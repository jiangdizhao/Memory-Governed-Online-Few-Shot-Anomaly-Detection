"""Synthetic, CPU-only tests for bounded P1 replay/cache primitives.

No Real-IAD files, labels, masks, checkpoints, native scorer, or GPU are used.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.calibration.stage1a_p1_reconstruction import (
    AUTHORIZATION_SCHEMA, BoundedAuthorization, C64Identity, ComputeBudget,
    FeatureCache, FeatureKey, MemoryIdentity, P1Blocked, P1ComputeLimitExceeded,
    P1NativeReplayRunner, P1ParityMismatch, QueryAnchor, QueriedLabelCache,
    NativeAnchorScoreProof, ReplayEvent, StaticMapKey, VersionedAnchorEvidenceCache,
    make_p1_authorization_record, make_queried_feedback_authorization_record,
    native_score_cache_key, sha256_json, shared_writer_memory_identity,
    apply_shared_native_writer, StoredC64StateLookup,
    source_fingerprints,
)
from experiments.calibration.protocol import EvidenceAnchor, EvidenceSnapshot
from experiments.calibration.stage075b_controllers import Stage075BOffset


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def approved_fixture_records(**bounds: int) -> tuple[dict, dict]:
    label_limit = bounds.pop("max_queried_label_reads", 459)
    record = make_p1_authorization_record(
        authorization_id="synthetic-only-test",
        approved_by="synthetic-fixture",
        approved_at_utc="2026-01-01T00:00:00Z",
        **bounds)
    record["decision"] = "APPROVED"
    label_record = make_queried_feedback_authorization_record(
        authorization_id="synthetic-label-fixture",
        approved_by="synthetic-fixture",
        approved_at_utc="2026-01-01T00:00:00Z",
        max_queried_label_reads=label_limit)
    label_record["decision"] = "APPROVED"
    return record, label_record


def c64(memory: MemoryIdentity, q90: float) -> C64Identity:
    return C64Identity(memory.version, memory.memory_sha256, q90,
                       sha(f"c64:{memory.version}:{memory.memory_sha256}"))


class ToyWriter:
    def __init__(self) -> None:
        self.memory = MemoryIdentity(0, sha("mem0"), sha("sources0"), 0, 0)

    def state_sha(self) -> str:
        return sha256_json(self.memory.__dict__)


def make_events() -> list[ReplayEvent]:
    m0 = MemoryIdentity(0, sha("mem0"), sha("sources0"), 0, 0)
    m1 = MemoryIdentity(1, sha("mem1"), sha("sources1"), 1, 1)
    c0, c1 = c64(m0, 57.0), c64(m1, 58.0)
    source0, source1, source2 = "group-0", "group-1", "group-2"
    return [
        ReplayEvent(0, False, source0, None, 0.1, 0.2, m0, m0,
                    c0, c0, {"action": "NO_QUERY", "changed": False,
                             "source_identity_sha256": sha(source0)}, sha("static0")),
        ReplayEvent(1, True, source1, sha("image1"), 0.3, 0.4, m0, m0,
                    c0, c0, {"action": "QUERIED_DEFECT_NO_WRITE", "changed": False,
                             "source_identity_sha256": sha(source1)}, sha("static0")),
        ReplayEvent(2, True, source2, sha("image2"), 0.5, 0.6, m0, m1,
                    c0, c1, {"action": "NORMAL_WRITE", "changed": True,
                             "source_identity_sha256": sha(source2)}, sha("static0")),
    ]


def run_toy(events: list[ReplayEvent], *, checkpoint=None, max_events=None,
            writer: ToyWriter | None = None, budget=None, labels_cache=None):
    writer = writer or ToyWriter()
    record, label_record = approved_fixture_records()
    runner = P1NativeReplayRunner(record, label_record, budget=budget)
    label_budget = runner.budget
    labels = labels_cache or QueriedLabelCache(label_budget, {
        "audiojack": [1, 2], "regulator": [], "switch": [], "rolled_strip_base": []})
    feature_fp, preprocess_fp = sha("encoder"), sha("preprocess")
    order = []

    def apply(event, queried, label, features, amap):
        if not queried:
            return {"action": "NO_QUERY", "changed": False,
                    "source_identity_sha256": sha(event.source_key)}
        if label == 1:
            return {"action": "QUERIED_DEFECT_NO_WRITE", "changed": False,
                    "source_identity_sha256": sha(event.source_key)}
        writer.memory = event.memory_after
        return {"action": "NORMAL_WRITE", "changed": True,
                "source_identity_sha256": sha(event.source_key)}

    result = runner.run(
        events_factory=lambda start: iter(events[start:]), total_events=len(events),
        trajectory_id="synthetic", category="audiojack",
        source_fingerprint_sha256=sha("sources"),
        scorer_config_sha256=sha("scorer-config"), scorer_source_sha256=sha("scorer-source"),
        state_fn=lambda: writer.memory,
        c64_fn=lambda mem: c64(mem, 57.0 if mem.version == 0 else 58.0),
        writer_state_sha256_fn=writer.state_sha,
        feature_key_fn=lambda event: FeatureKey(event.image_sha256, feature_fp, preprocess_fp),
        encode_feature_fn=lambda event: {"feature": event.image_sha256},
        queried_labels=labels,
        label_reader=lambda category, position, group: (
            order.append(f"label:{position}"), 1 if position == 1 else 0)[1],
        static_map_fn=lambda event, feature: (
            order.append(f"map:{event.position}"),
            (event.static_score, {"map": feature}))[1],
        writer_fn=apply, checkpoint=checkpoint, max_events=max_events)
    return result, writer, runner, labels, order


class Stage1AP1ReconstructionTests(unittest.TestCase):
    def test_missing_authorization_blocks_before_any_input_provider(self) -> None:
        touched = []
        with self.assertRaises(P1Blocked):
            P1NativeReplayRunner(None, None)
        with self.assertRaises(P1Blocked):
            BoundedAuthorization.from_records({"schema": AUTHORIZATION_SCHEMA,
                                               "decision": "PENDING_RESEARCHER_DECISION"}, None)
        self.assertEqual(touched, [])

    def test_query_only_exact_replay_and_checkpoint_resume(self) -> None:
        events = make_events()
        one_shot, writer_one, runner_one, labels_one, order_one = run_toy(events)
        self.assertTrue(one_shot.complete)
        self.assertEqual(one_shot.status, "PASS_EXACT_PARITY")
        self.assertEqual(one_shot.events_checked, 3)
        self.assertEqual(one_shot.query_events, 2)
        self.assertEqual(runner_one.budget.feature_encodings, 2)
        self.assertEqual(runner_one.budget.static_map_calls, 2)
        self.assertEqual(order_one, ["map:1", "label:1", "map:2", "label:2"])
        self.assertEqual(labels_one.disclosure_count, 2)
        self.assertEqual(writer_one.memory, events[-1].memory_after)

        first, writer_split, runner_split, labels_split, _ = run_toy(events, max_events=2)
        self.assertFalse(first.complete)
        self.assertEqual(first.checkpoint.next_position, 2)
        self.assertNotIn('"label"', str(first.checkpoint.to_public_dict()))
        labels_restored = QueriedLabelCache(runner_split.budget, {
            "audiojack": [1, 2], "regulator": [], "switch": [], "rolled_strip_base": []})
        labels_restored.load_private_state_dict(labels_split.private_state_dict())
        resumed, writer_resumed, runner_resumed, labels_resumed, _ = run_toy(
            events, checkpoint=first.checkpoint, writer=writer_split,
            budget=runner_split.budget, labels_cache=labels_restored)
        self.assertTrue(resumed.complete)
        self.assertEqual(resumed.event_chain_sha256, one_shot.event_chain_sha256)
        self.assertEqual(resumed.query_response_sha256, one_shot.query_response_sha256)
        self.assertEqual(writer_resumed.memory, writer_one.memory)
        self.assertEqual(runner_resumed.budget.feature_encodings, 2)
        self.assertEqual(labels_resumed.disclosure_count, 2)

    def test_memory_mutation_stops_before_writer_and_does_not_repair(self) -> None:
        events = make_events()
        events[0] = replace(
            events[0], memory_before=replace(events[0].memory_before,
                                             memory_sha256=sha("mutated")))
        touched = []
        writer = ToyWriter()
        record, label_record = approved_fixture_records()
        runner = P1NativeReplayRunner(record, label_record)
        labels = QueriedLabelCache(runner.budget, {
            "audiojack": [1, 2], "regulator": [], "switch": [], "rolled_strip_base": []})
        with self.assertRaisesRegex(P1ParityMismatch, "pre-event memory/source identity"):
            runner.run(
                events_factory=lambda start: iter(events[start:]), total_events=3,
                trajectory_id="mutated", category="audiojack",
                source_fingerprint_sha256=sha("source"),
                scorer_config_sha256=sha("scorer-config"), scorer_source_sha256=sha("scorer-source"),
                state_fn=lambda: writer.memory,
                c64_fn=lambda mem: c64(mem, 57.0 if mem.version == 0 else 58.0),
                writer_state_sha256_fn=writer.state_sha,
                feature_key_fn=lambda event: FeatureKey(event.image_sha256, sha("e"), sha("p")),
                encode_feature_fn=lambda event: touched.append("encode"),
                queried_labels=labels,
                label_reader=lambda *args: touched.append("label"),
                static_map_fn=lambda *args: touched.append("map"),
                writer_fn=lambda event, *args: (
                    touched.append(f"writer:{event.position}"), {
                        "action": "NO_QUERY", "changed": False,
                        "source_identity_sha256": sha(event.source_key)})[1])
        self.assertEqual(touched, [])

    def test_writer_and_c64_mutations_stop_exact_replay(self) -> None:
        for mutate in (
            lambda event: replace(event, expected_writer_signature={
                **event.expected_writer_signature, "action": "MUTATED"}),
            lambda event: replace(event, c64_before=replace(
                event.c64_before, transport_sha256=sha("mutated-transport"))),
        ):
            events = make_events()
            events[0] = mutate(events[0])
            writer = ToyWriter()
            record, label_record = approved_fixture_records()
            runner = P1NativeReplayRunner(record, label_record)
            labels = QueriedLabelCache(runner.budget, {
                "audiojack": [1, 2], "regulator": [], "switch": [], "rolled_strip_base": []})
            with self.assertRaises(P1ParityMismatch):
                runner.run(
                    events_factory=lambda start: iter(events[start:]), total_events=3,
                    trajectory_id="mutation", category="audiojack",
                    source_fingerprint_sha256=sha("source"),
                    scorer_config_sha256=sha("scorer-config"), scorer_source_sha256=sha("scorer-source"),
                    state_fn=lambda: writer.memory,
                    c64_fn=lambda mem: c64(mem, 57.0 if mem.version == 0 else 58.0),
                    writer_state_sha256_fn=writer.state_sha,
                    feature_key_fn=lambda event: FeatureKey(event.image_sha256, sha("e"), sha("p")),
                    encode_feature_fn=lambda event: "feature",
                    queried_labels=labels, label_reader=lambda *args: 0,
                    static_map_fn=lambda event, feature: (event.static_score, "map"),
                    writer_fn=lambda event, *args: {
                        "action": "NO_QUERY", "changed": False,
                        "source_identity_sha256": sha(event.source_key)})

    def test_query_flag_mutation_stops_before_label_or_feature_access(self) -> None:
        events = make_events()
        events[1] = replace(events[1], queried=False)
        touched = []
        writer = ToyWriter()
        record, label_record = approved_fixture_records()
        runner = P1NativeReplayRunner(record, label_record)
        labels = QueriedLabelCache(runner.budget, {
            "audiojack": [1, 2], "regulator": [], "switch": [], "rolled_strip_base": []})
        with self.assertRaisesRegex(P1ParityMismatch, "query flag"):
            runner.run(
                events_factory=lambda start: iter(events[start:]), total_events=3,
                trajectory_id="query-mutation", category="audiojack",
                source_fingerprint_sha256=sha("source"),
                scorer_config_sha256=sha("scorer-config"), scorer_source_sha256=sha("scorer-source"),
                state_fn=lambda: writer.memory,
                c64_fn=lambda mem: c64(mem, 57.0 if mem.version == 0 else 58.0),
                writer_state_sha256_fn=writer.state_sha,
                feature_key_fn=lambda event: FeatureKey(event.image_sha256, sha("e"), sha("p")),
                encode_feature_fn=lambda event: touched.append("encode"),
                queried_labels=labels,
                label_reader=lambda *args: touched.append("label"),
                static_map_fn=lambda *args: touched.append("map"),
                writer_fn=lambda event, *args: (
                    touched.append(f"writer:{event.position}"), {
                        "action": "NO_QUERY", "changed": False,
                        "source_identity_sha256": sha(event.source_key)})[1])
        self.assertEqual(touched, ["writer:0"])

    def test_resume_checkpoint_identity_mutation_stops(self) -> None:
        events = make_events()
        first, writer, first_runner, labels, _ = run_toy(events, max_events=2)
        bad_checkpoint = replace(first.checkpoint,
                                 source_fingerprint_sha256=sha("different-source"))
        with self.assertRaisesRegex(P1ParityMismatch, "checkpoint identity"):
            run_toy(events, checkpoint=bad_checkpoint, writer=writer,
                    budget=first_runner.budget, labels_cache=labels)

    def test_compute_cap_stops_before_native_feature_call(self) -> None:
        auth = BoundedAuthorization.from_records(
            *approved_fixture_records(max_feature_encodings=0))
        budget = ComputeBudget(auth)
        cache = FeatureCache(budget)
        called = []
        key = FeatureKey(sha("image"), sha("encoder"), sha("preprocess"))
        with self.assertRaises(P1ComputeLimitExceeded):
            cache.get_or_encode(key, lambda: called.append(True))
        self.assertEqual(called, [])

    def test_query_label_cache_reads_only_precommitted_queries_once(self) -> None:
        authorization = BoundedAuthorization.from_records(
            *approved_fixture_records(max_queried_label_reads=1))
        budget = ComputeBudget(authorization)
        cache = QueriedLabelCache(budget, {
            "audiojack": [4], "regulator": [], "switch": [], "rolled_strip_base": []})
        reads = []
        reader = lambda *args: (reads.append(args), 0)[1]
        self.assertEqual(cache.get("audiojack", 4, "same-group", reader), 0)
        self.assertEqual(cache.get("audiojack", 4, "same-group", reader), 0)
        self.assertEqual(len(reads), 1)
        self.assertEqual(cache.disclosure_count, 1)
        self.assertEqual(budget.queried_label_reads, 1)
        with self.assertRaises(P1Blocked):
            cache.get("audiojack", 5, "unqueried-group", reader)
        changed = cache.private_state_dict()
        changed["values"][0][3] = 1
        restored = QueriedLabelCache(budget, {
            "audiojack": [4], "regulator": [], "switch": [], "rolled_strip_base": []})
        restored.load_private_state_dict(changed)
        self.assertNotEqual(cache.response_sha256, restored.response_sha256)

    def test_feature_and_static_map_cache_require_exact_fingerprints(self) -> None:
        auth = BoundedAuthorization.from_records(*approved_fixture_records())
        budget = ComputeBudget(auth)
        cache = FeatureCache(budget)
        image, encoder, prep = sha("image"), sha("encoder"), sha("prep")
        key = FeatureKey(image, encoder, prep)
        calls = []
        self.assertEqual(cache.get_or_encode(key, lambda: calls.append(1) or "f"), "f")
        self.assertEqual(cache.get_or_encode(key, lambda: calls.append(2) or "g"), "f")
        self.assertEqual(calls, [1])
        for mutated in (replace(key, image_sha256=sha("other-image")),
                        replace(key, encoder_fingerprint_sha256=sha("other-encoder")),
                        replace(key, preprocessing_fingerprint_sha256=sha("other-prep"))):
            self.assertNotEqual(key.sha256, mutated.sha256)
        self.assertEqual(budget.feature_encodings, 1)

        map_key = StaticMapKey(image, sha("static-bank"), sha("cfg"), sha("source"))
        for mutated in (replace(map_key, image_sha256=sha("other-image")),
                        replace(map_key, static_memory_sha256=sha("other-bank")),
                        replace(map_key, scorer_config_sha256=sha("other-cfg")),
                        replace(map_key, scorer_source_sha256=sha("other-source"))):
            self.assertNotEqual(map_key.sha256, mutated.sha256)

    def test_anchor_cache_reuses_old_scores_and_invalidates_on_memory_or_provenance(self) -> None:
        auth = BoundedAuthorization.from_records(*approved_fixture_records())
        budget = ComputeBudget(auth)
        cache = VersionedAnchorEvidenceCache(budget)
        m0 = MemoryIdentity(0, sha("m0"), sha("s0"), 0, 0)
        m1 = MemoryIdentity(1, sha("m1"), sha("s1"), 1, 1)
        anchors = [
            QueryAnchor(sha("a0"), sha("g0"), "normal", "feature0"),
            QueryAnchor(sha("a1"), sha("g1"), "defect", "feature1"),
        ]
        normal_policies = []

        def score(anchor, memory, policy):
            normal_policies.append(policy)
            return NativeAnchorScoreProof(
                float(memory.version + (1 if anchor.kind == "normal" else 2)),
                memory.version, memory.memory_sha256, memory.source_map_sha256,
                anchor.anchor_identity_sha256, anchor.source_group_sha256,
                policy, (0, 0, 0))

        kwargs = dict(protected_c64_scores=list(map(float, range(64))),
                      expected_c64_q90=57.0, scorer_config_sha256=sha("cfg"),
                      scorer_source_sha256=sha("source"),
                      protected_group_sha256s=[sha("c64")],
                      deployed_support_group_sha256s=[sha("support")],
                      score_anchor=score)
        first = cache.refresh(memory=m0, anchors=anchors,
                              c64_memory_sha256=m0.memory_sha256,
                              c64_transport_sha256=sha("transport0"), **kwargs)
        extended = anchors + [QueryAnchor(sha("a2"), sha("g2"), "defect", "feature2")]
        second = cache.refresh(memory=m0, anchors=extended,
                               c64_memory_sha256=m0.memory_sha256,
                               c64_transport_sha256=sha("transport0"), **kwargs)
        self.assertEqual(first.native_score_calls, 2)
        self.assertEqual(second.native_score_calls, 1)
        self.assertEqual(second.reused_query_scores, 2)
        self.assertEqual(normal_policies[0], "queried_normal_all_layer_leave_source_out")
        third = cache.refresh(memory=m1, anchors=extended,
                              c64_memory_sha256=m1.memory_sha256,
                              c64_transport_sha256=sha("transport1"),
                              expected_c64_q90=57.0, **{k: v for k, v in kwargs.items()
                              if k not in {"expected_c64_q90"}})
        self.assertEqual(third.native_score_calls, 3)

        with self.assertRaisesRegex(P1ParityMismatch, "official sample group"):
            cache.refresh(memory=m1,
                anchors=[anchors[0], QueryAnchor(sha("a3"), sha("g0"), "normal", "f3")],
                c64_memory_sha256=m1.memory_sha256,
                c64_transport_sha256=sha("transport1"), **kwargs)
        with self.assertRaisesRegex(P1ParityMismatch, "another memory state"):
            cache.refresh(memory=m1, anchors=extended,
                c64_memory_sha256=m0.memory_sha256,
                c64_transport_sha256=sha("transport1"), **kwargs)

    def test_native_score_cache_key_mutations_are_distinct(self) -> None:
        memory = MemoryIdentity(0, sha("m"), sha("s"), 0, 0)
        anchor = QueryAnchor(sha("a"), sha("g"), "normal", None)
        base = native_score_cache_key(memory=memory, anchor=anchor,
                                      scorer_config_sha256=sha("cfg"),
                                      scorer_source_sha256=sha("src"))
        mutations = [
            (replace(memory, version=1), anchor, sha("cfg"), sha("src")),
            (replace(memory, memory_sha256=sha("m2")), anchor, sha("cfg"), sha("src")),
            (replace(memory, source_map_sha256=sha("s2")), anchor, sha("cfg"), sha("src")),
            (memory, replace(anchor, anchor_identity_sha256=sha("a2")), sha("cfg"), sha("src")),
            (memory, replace(anchor, source_group_sha256=sha("g2")), sha("cfg"), sha("src")),
            (memory, anchor, sha("cfg2"), sha("src")),
            (memory, anchor, sha("cfg"), sha("src2")),
            (memory, replace(anchor, kind="defect"), sha("cfg"), sha("src")),
        ]
        for mem, anc, cfg, src in mutations:
            self.assertNotEqual(base, native_score_cache_key(
                memory=mem, anchor=anc, scorer_config_sha256=cfg,
                scorer_source_sha256=src))

    def test_source_fingerprints_are_deterministic_and_mutation_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.py"
            source.write_text("x = 1\n", encoding="utf-8")
            first = source_fingerprints(root, ["source.py"])
            second = source_fingerprints(root, ["source.py"])
            self.assertEqual(first, second)
            source.write_text("x = 2\n", encoding="utf-8")
            third = source_fingerprints(root, ["source.py"])
            self.assertNotEqual(first["aggregate_sha256"], third["aggregate_sha256"])

    def test_existing_shared_writer_and_c64_lookup_adapters_are_exact(self) -> None:
        event = make_events()[0]
        fake_bank = type("Bank", (), {
            "memory_content_version": 3,
            "legacy_greedy_version": 4,
            "sha256": lambda self: sha("memory")})()
        fake_writer = type("Writer", (), {
            "bank": fake_bank,
            "source_revision": 5,
            "source_map_sha256": lambda self: sha("sources"),
            "apply": lambda self, **kwargs: kwargs})()
        identity = shared_writer_memory_identity(fake_writer)
        self.assertEqual(identity, MemoryIdentity(3, sha("memory"), sha("sources"), 5, 4))
        applied = apply_shared_native_writer(fake_writer, event, False, None, None, None)
        self.assertEqual(applied["score"], event.static_score)
        self.assertEqual(applied["step"], event.position)
        self.assertEqual(applied["full_features"], ())
        c64_state = c64(identity, 12.0)
        lookup = StoredC64StateLookup({(identity.version, identity.memory_sha256): c64_state})
        self.assertEqual(lookup(identity), c64_state)
        with self.assertRaises(P1ParityMismatch):
            lookup(replace(identity, memory_sha256=sha("missing")))

    def test_stage075b_offset_holds_one_sided_and_is_not_continuous_b3_offset(self) -> None:
        controller = Stage075BOffset(theta=0.25)
        one_sided = EvidenceSnapshot(0,
            (EvidenceAnchor("n", 0.1, 0, False, 1,
                            "queried_normal_all_layer_leave_source_out"),), (), {}, 1)
        controller.on_memory_change(one_sided)
        self.assertEqual(controller.finalize_step(), 0.25)
        self.assertEqual(controller.diagnostics_last["desired"], 0.25)

        both_sides = EvidenceSnapshot(1,
            (EvidenceAnchor("n", 0.2, 1, False, 1,
                            "queried_normal_all_layer_leave_source_out"),),
            (EvidenceAnchor("d", 0.8, 1, False, 0, "queried_defect_not_written"),),
            {}, 2)
        controller.on_memory_change(both_sides)
        self.assertAlmostEqual(controller.finalize_step(), 0.26)
        # A moving B3 threshold is not added to this controller's raw median target.
        self.assertAlmostEqual(controller.diagnostics_last["desired"], 0.5)


if __name__ == "__main__":
    unittest.main()
