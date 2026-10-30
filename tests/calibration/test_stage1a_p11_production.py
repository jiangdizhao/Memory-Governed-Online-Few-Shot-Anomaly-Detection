"""CPU-only P1.1 integration tests; all data, labels and features are synthetic."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import pickle
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.calibration import stage05
from experiments.calibration.memory_state import CalibrationGreedyState
from experiments.calibration.native import SharedNativeGreedyWriter
from experiments.calibration.protocol import ScoreResult
from experiments.risk_safe.shadow_memory import GreedyBank
from experiments.calibration.stage1a_p1_reconstruction import (
    BoundedAuthorization, C64Identity, ComputeBudget, FeatureKey, MemoryIdentity,
    P1Blocked, P1ParityMismatch, P1_PROTOCOL_SHA256, QueryAnchor, ReplayEvent,
    VersionedAnchorEvidenceCache, sha256_json,
)
from experiments.calibration.stage1a_p11_production import (
    CausalEvidenceRevisionAdapter, NativeAllLayerLSOAdapter,
    P11IntegrityError, P11ProductionReplayRunner,
    P11PrivateCheckpoint, P11_AUTHORIZATION_SCHEMA,
    P11_FEEDBACK_AUTHORIZATION_SCHEMA, P11_CLASSES,
    apply_historical_shared_writer, expected_historical_replay_signature,
    make_p11_authorization_templates, refresh_complete_evidence,
    verify_controller_snapshot_compatibility, verify_qats_objects,
    query_response_chain_from_private_cache,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def synthetic_tape_set(n: int = 100):
    frozen = {"schema": "stage05.frozen_inputs.v1", "query_seeds": [37],
              "classes": {}, "tape_digests_by_seed37": {}}
    archive = {}
    for category in P11_CLASSES:
        stream = []
        records = []
        for position in range(n):
            group = f"fixture-{category}-group-{position:04d}"
            opaque_group = sha(f"official-sample-group\0{group}")
            relative = f"fixture/{category}/{position:04d}.jpg"
            stream.append({"position": position, "relative_id": relative,
                           "opaque_group": opaque_group, "camera_id": position % 5 + 1,
                           "timestamp": f"2026-01-01T00:{position % 60:02d}:00Z",
                           "image_path": "ignored-fixture-path"})
            records.append(stage05.TapeRecord(
                position, opaque_group, position % 5 + 1,
                sha(f"image-identity\0{relative}"), stream[-1]["timestamp"], 0))
        tapes = stage05.build_nested_tapes(records, 37)
        frozen["classes"][category] = {
            "stream": stream, "stream_images": n,
            "stream_official_groups": n,
        }
        frozen["tape_digests_by_seed37"][category] = {
            "0.03": tapes[0.03]["digest"]}
        archive[category] = {f"{rate:.2f}": {"37": tape}
                             for rate, tape in tapes.items()}
    return verify_qats_objects(
        frozen, archive, frozen_sha256=sha("frozen"),
        archive_sha256=sha("tapes"), checksums_sha256=sha("checksums"))


def approved_records(tapes):
    native, feedback = make_p11_authorization_templates(
        authorization_id="cpu-fixture-p1-native",
        feedback_authorization_id="cpu-fixture-p1-labels",
        verified_tapes=tapes)
    native["decision"] = "APPROVED"
    feedback["decision"] = "APPROVED"
    native["approved_by"] = feedback["approved_by"] = "synthetic-fixture"
    native["approved_at_utc"] = feedback["approved_at_utc"] = "2026-01-01T00:00:00Z"
    return native, feedback


class TinyCodebook:
    def __init__(self, rows: torch.Tensor, capacity: int):
        self.protos = rows
        self.K = int(capacity)
        self.feat_dim = int(rows.shape[1])
        self.device = rows.device

    def size(self):
        return len(self.protos)


class TinyNativeBank:
    """Only the public v22/TMRA codebook layout used by the actual writer."""
    def __init__(self, layers=2, dim=4):
        def cbs(rows, capacity):
            return [TinyCodebook(rows.clone(), capacity) for _ in range(layers)]
        fixed = cbs(torch.eye(1, dim, dtype=torch.float32), 25_000)
        empty = torch.empty((0, dim), dtype=torch.float32)
        self.normal_fixed = {"p0": [{"id": fixed}]}
        self.normal_dyn = {"p0": [{"id": cbs(empty, 256)}]}
        self.normal_dyn_ltm = {"p0": [{"id": cbs(empty, 1)}]}
        self.normal_dyn_candidates = {"p0": [{"id": cbs(empty, 1)}]}
        self.K_normal_dyn = 256


class FixtureImpl:
    def select_informative_normal_write(self, **kwargs):
        return True, 8, "cpu_fixture_accept", {"fixture": True}

    def layer_to_topk_from_amap(self, _amap):
        return {0: list(range(8)), 1: list(range(8))}

    @staticmethod
    def l2_normalize(features):
        return features / (features.norm(dim=-1, keepdim=True) + 1e-12)


def make_writer():
    protected = [f"protected-c64-{i:02d}" for i in range(64)]
    return SharedNativeGreedyWriter(FixtureImpl(), TinyNativeBank(),
                                    theta_ref=0.5,
                                    protected_source_keys=protected,
                                    sigma_ref=0.02)


def append_group(writer, source="queried-normal-group"):
    features = tuple(torch.tensor(
        [[0., 1., 0., 0.]] * 8, dtype=torch.float32) for _ in writer.bank.layers)
    event = type("WriterEvent", (), {"source_key": source,
                                    "static_score": 0.2,
                                    "position": 10})()
    record = apply_historical_shared_writer(writer, event, True, 0,
                                           features, {"synthetic": True})
    return record


def writer_state_hash(writer):
    return sha256_json({"memory": asdict(
        __import__("experiments.calibration.stage1a_p1_reconstruction",
                   fromlist=["shared_writer_memory_identity"])
        .shared_writer_memory_identity(writer)),
        "source_map": writer.source_map_sha256()})


def fixture_event_tape(writer, tapes, n=100):
    from experiments.calibration.stage1a_p1_reconstruction import (
        C64Identity, ReplayEvent, shared_writer_memory_identity,
    )
    memory = shared_writer_memory_identity(writer)
    c64 = C64Identity(memory.version, memory.memory_sha256, 57.0,
                      sha(f"c64-transport:{memory.memory_sha256}"))
    query_positions = set(tapes.positions("audiojack"))
    events = []
    for position in range(n):
        source = f"fixture-audiojack-group-{position:04d}"
        queried = position in query_positions
        event = ReplayEvent(
            position, queried, source, sha(f"fixture-image-{position}") if queried else None,
            0.2, 0.3, memory, memory, c64, c64, {}, sha("static-bank"))
        expected = historical_expected(writer, event)
        events.append(ReplayEvent(
            position, queried, source, event.image_sha256, event.static_score,
            event.dynamic_score, memory, memory, c64, c64, expected,
            event.static_memory_sha256))
    return events


def historical_expected(writer, event):
    from experiments.calibration.stage075a_runner import _writer_record
    result = writer.apply(
        queried=event.queried, label=1 if event.queried else None,
        source_key=event.source_key, score=event.static_score,
        static_amap=None, full_features=(), step=event.position)
    return expected_historical_replay_signature(
        _writer_record(result, writer), event.source_key)


def replay_kwargs(writer, events, tapes, *, checkpoint=None, max_events=None):
    from experiments.calibration.stage1a_p1_reconstruction import (
        FeatureKey, shared_writer_memory_identity,
    )
    fp, prep, cfg, source = map(sha, ("encoder", "preprocessing", "cfg", "source"))
    memory_fn = lambda: shared_writer_memory_identity(writer)
    c64_fn = lambda memory: C64Identity(
        memory.version, memory.memory_sha256, 57.0,
        sha(f"c64-transport:{memory.memory_sha256}"))
    return {
        "events_factory": lambda start: iter(events[start:]),
        "total_events": len(events), "trajectory_id": "fixture-audiojack-seed0",
        "category": "audiojack", "source_fingerprint_sha256": sha("source-tree"),
        "scorer_config_sha256": cfg, "scorer_source_sha256": source,
        "state_fn": memory_fn, "c64_fn": c64_fn,
        "writer_state_sha256_fn": lambda: writer_state_hash(writer),
        "feature_key_fn": lambda event: FeatureKey(event.image_sha256, fp, prep),
        "encode_feature_fn": lambda event: {"synthetic_feature_for": event.position},
        "label_reader": lambda _category, _position, _source: 1,
        "static_map_fn": lambda event, _features: (event.static_score, {"map": event.position}),
        "writer_fn": lambda event, queried, label, features, amap:
            apply_historical_shared_writer(writer, event, queried, label, features, amap),
        "checkpoint": checkpoint, "max_events": max_events,
    }


def resume_from_serialized_checkpoint(checkpoint_path: str) -> dict:
    """Child-process fixture: reconstruct private state and complete the tape."""
    checkpoint = pickle.loads(Path(checkpoint_path).read_bytes())
    tapes = synthetic_tape_set()
    native, feedback = approved_records(tapes)
    writer = make_writer()
    runner = P11ProductionReplayRunner(native, feedback, tapes)
    checkpoint.restore(production_runner=runner, writer=writer,
                       scorer_config_sha256=sha("cfg"),
                       scorer_source_sha256=sha("source"))
    events = fixture_event_tape(writer, tapes)
    result = runner.run(**replay_kwargs(
        writer, events, tapes, checkpoint=checkpoint.replay_checkpoint))
    return {"complete": result.complete,
            "event_chain_sha256": result.event_chain_sha256,
            "query_response_sha256": result.query_response_sha256,
            "writer_state_sha256": writer_state_hash(writer),
            "budget": {"queried_label_reads": runner.runner.budget.queried_label_reads,
                       "feature_encodings": runner.runner.budget.feature_encodings,
                       "static_map_calls": runner.runner.budget.static_map_calls,
                       "anchor_score_calls": runner.runner.budget.anchor_score_calls}}


class Stage1AP11ProductionTests(unittest.TestCase):
    def test_four_exact_qats_tapes_verify_and_tape_mutation_is_rejected(self):
        tapes = synthetic_tape_set()
        self.assertEqual(tuple(row["class_identity"] for row in tapes.bindings), P11_CLASSES)
        self.assertEqual(sum(row["query_count"] for row in tapes.bindings), 12)
        self.assertTrue(all(row["seed"] == 37 and row["official_group_unique"]
                            for row in tapes.bindings))

        frozen = {"schema": "stage05.frozen_inputs.v1", "query_seeds": [37],
                  "classes": {}, "tape_digests_by_seed37": {}}
        archive = {}
        for category in P11_CLASSES:
            rows = []
            records = []
            for position in range(100):
                group, relative = f"g-{category}-{position}", f"r-{category}-{position}"
                opaque_group = sha(f"official-sample-group\0{group}")
                rows.append({"position": position, "relative_id": relative,
                             "opaque_group": opaque_group, "camera_id": position % 5 + 1,
                             "timestamp": "", "image_path": "ignored"})
                records.append(stage05.TapeRecord(
                    position, opaque_group, position % 5 + 1,
                    sha(f"image-identity\0{relative}"), "", 0))
            nested = stage05.build_nested_tapes(records, 37)
            frozen["classes"][category] = {"stream": rows, "stream_images": 100,
                                             "stream_official_groups": 100}
            frozen["tape_digests_by_seed37"][category] = {"0.03": nested[.03]["digest"]}
            archive[category] = {f"{rate:.2f}": {"37": tape}
                                 for rate, tape in nested.items()}
        archive["switch"]["0.03"]["37"]["events"][0]["position"] += 1
        with self.assertRaises(P11IntegrityError):
            verify_qats_objects(frozen, archive, frozen_sha256=sha("f"),
                                archive_sha256=sha("t"), checksums_sha256=sha("c"))

    def test_authorization_requires_exact_verified_tape_bindings(self):
        tapes = synthetic_tape_set()
        native, feedback = approved_records(tapes)
        runner = P11ProductionReplayRunner(native, feedback, tapes)
        self.assertEqual(runner.authorization.query_protocol_sha256,
                         tapes.bindings[0]["query_protocol_sha256"])
        altered = dict(native)
        altered["query_tape_bindings"] = [dict(row) for row in native["query_tape_bindings"]]
        altered["query_tape_bindings"][0]["tape_sha256"] = sha("substituted-tape")
        with self.assertRaises(P1Blocked):
            P11ProductionReplayRunner(altered, feedback, tapes)

    def test_query_group_substitution_stops_before_feature_or_label_access(self):
        tapes = synthetic_tape_set()
        native, feedback = approved_records(tapes)
        writer = make_writer()
        events = fixture_event_tape(writer, tapes)
        position = tapes.positions("audiojack")[0]
        events[position] = replace(events[position], source_key="substituted-official-group")
        runner = P11ProductionReplayRunner(native, feedback, tapes)
        kwargs = replay_kwargs(writer, events, tapes, max_events=position + 1)
        touched = []
        kwargs["encode_feature_fn"] = lambda event: touched.append("feature")
        kwargs["label_reader"] = lambda *args: touched.append("label") or 1
        with self.assertRaisesRegex(P11IntegrityError, "official-group identity"):
            runner.run(**kwargs)
        self.assertEqual(touched, [])

    def test_historical_writer_adapter_preserves_exact_source_schema_and_counts(self):
        writer = make_writer()
        record = append_group(writer)
        expected_schema = {
            "eligible", "changed", "keep", "reason", "appended", "replaced",
            "skipped", "source_map_changed", "source_revision_before",
            "source_revision_after", "memory_content_version_before",
            "memory_content_version_after", "legacy_greedy_version_before",
            "legacy_greedy_version_after", "dynamic_rows_by_layer",
            "capacity_by_layer", "source_map_sha256",
            "resident_source_groups_by_layer_sha256",
        }
        self.assertEqual(set(record), expected_schema)
        self.assertTrue(record["eligible"] and record["changed"])
        self.assertEqual(record["appended"], 16)
        self.assertEqual(record["dynamic_rows_by_layer"], [8, 8])
        self.assertEqual(record["capacity_by_layer"], [256, 256])
        self.assertEqual(record["memory_content_version_after"], 1)
        self.assertEqual(record["legacy_greedy_version_after"], 1)
        self.assertEqual(record["source_revision_after"], 1)
        self.assertEqual(len(record["resident_source_groups_by_layer_sha256"]), 2)

        skip_event = type("WriterEvent", (), {"source_key": "defect-group",
                                              "static_score": 0.7,
                                              "position": 12})()
        skipped = apply_historical_shared_writer(writer, skip_event, True, 1, None, None)
        self.assertFalse(skipped["eligible"])
        self.assertFalse(skipped["changed"])
        self.assertEqual(skipped["reason"], "feedback_not_verified_normal")
        self.assertEqual(skipped["appended"] + skipped["replaced"], 0)

    def test_historical_writer_adapter_reports_real_replacement_and_skip_counts(self):
        writer = make_writer()
        layers = tuple(torch.tensor([[1., 0., 0., 0.]] * 256) for _ in writer.bank.layers)
        sources = [["preexisting-group"] * 256 for _ in layers]
        legacy = GreedyBank(layers, writer.bank.capacities, version=1)
        memory = CalibrationGreedyState(legacy, 1)
        state = writer.state_dict()
        source_payload = json.dumps(sources, ensure_ascii=True,
                                    separators=(",", ":")).encode("ascii")
        state.update({"layers": [x.clone() for x in layers], "sources": sources,
                      "memory_content_version": 1, "legacy_greedy_version": 1,
                      "sha256": memory.sha256(), "content_sha256": memory.content_sha256(),
                      "legacy_greedy_sha256": legacy.sha256(), "source_revision": 0,
                      "source_map_sha256": hashlib.sha256(source_payload).hexdigest()})
        writer.load_state_dict(state)
        event = type("WriterEvent", (), {"source_key": "replacement-group",
                                          "static_score": 0.2,
                                          "position": 20})()
        features = tuple(torch.tensor([[0., 1., 0., 0.]] * 8) for _ in layers)
        record = apply_historical_shared_writer(writer, event, True, 0,
                                               features, {"fixture": True})
        self.assertEqual(record["replaced"], 2)
        self.assertEqual(record["skipped"], 14)
        self.assertEqual(record["appended"], 0)
        self.assertEqual(record["dynamic_rows_by_layer"], [256, 256])
        self.assertEqual(record["source_revision_after"], 1)

    def test_native_lso_filters_actual_source_rows_in_every_layer(self):
        writer = make_writer()
        append_group(writer, "normal-source")
        memory = __import__("experiments.calibration.stage1a_p1_reconstruction",
                            fromlist=["shared_writer_memory_identity"])
        identity = memory.shared_writer_memory_identity(writer)
        anchor = QueryAnchor(sha("normal-anchor"), sha("normal-source"), "normal",
                             {"source_key": "normal-source", "bundle": "cpu-feature"})
        seen = []
        adapter = NativeAllLayerLSOAdapter(
            writer, lambda bundle, excluded: (
                seen.append((bundle, [len(layer) for layer in excluded.layers])) or
                ScoreResult(0.25, excluded.version, "synthetic-score-state")))
        proof = adapter(anchor, identity, "queried_normal_all_layer_leave_source_out")
        self.assertEqual(proof.source_group_remaining_by_layer, (0, 0))
        self.assertEqual(adapter.proofs[anchor.anchor_identity_sha256]["removed_rows_by_layer"], [8, 8])
        self.assertEqual(seen, [({"source_key": "normal-source", "bundle": "cpu-feature"}, [0, 0])])

    def test_native_lso_rejects_retained_source_row_in_one_layer(self):
        from experiments.calibration.anchor_ledger import AnchorLedger
        writer = make_writer()
        append_group(writer, "normal-source")
        memory = __import__("experiments.calibration.stage1a_p1_reconstruction",
                            fromlist=["shared_writer_memory_identity"])
        identity = memory.shared_writer_memory_identity(writer)
        anchor = QueryAnchor(sha("normal-anchor"), sha("normal-source"), "normal",
                             {"source_key": "normal-source", "bundle": "fixture"})
        original = AnchorLedger._without_source
        def faulty(bank, rows, key):
            filtered, removed = original(bank, rows, key)
            mixed = list(filtered.layers)
            mixed[1] = bank.layers[1]  # deliberately retain source rows in layer one
            return GreedyBank(mixed, bank.capacities, version=bank.version), removed
        called = []
        adapter = NativeAllLayerLSOAdapter(writer,
            lambda *_: called.append(True) or ScoreResult(0.1, identity.version, "state"))
        with patch.object(AnchorLedger, "_without_source", staticmethod(faulty)):
            with self.assertRaisesRegex(P11IntegrityError, "retained or removed rows inconsistently"):
                adapter(anchor, identity, "queried_normal_all_layer_leave_source_out")
        self.assertEqual(called, [])

    def test_complete_evidence_snapshot_is_accepted_by_b4_b7_and_offset(self):
        writer = make_writer()
        append_group(writer, "normal-source")
        identity = __import__("experiments.calibration.stage1a_p1_reconstruction",
                             fromlist=["shared_writer_memory_identity"])
        memory = identity.shared_writer_memory_identity(writer)
        normal = QueryAnchor(sha("normal-anchor"), sha("normal-source"), "normal",
                             {"source_key": "normal-source", "bundle": "normal"})
        defect = QueryAnchor(sha("defect-anchor"), sha("defect-source"), "defect",
                             {"source_key": "defect-source", "bundle": "defect"})
        c64_values = tuple(float(i) for i in range(64))
        c64_groups = [sha(f"protected-c64-group-{i}") for i in range(64)]
        scorer = NativeAllLayerLSOAdapter(
            writer, lambda payload, bank: ScoreResult(
                0.2 if payload["bundle"] == "normal" else 0.8,
                bank.version, "synthetic-current-memory-score"))
        cache = VersionedAnchorEvidenceCache(ComputeBudget(BoundedAuthorization(
            "test", "test-feedback", P11_CLASSES, P1_PROTOCOL_SHA256,
            459, 459, 2295, 75406)))
        result = refresh_complete_evidence(
            cache=cache, scorer=scorer, memory=memory, anchors=[normal, defect],
            protected_c64_scores=c64_values,
            c64_transport_sha256=sha("stored-c64-transport"),
            expected_c64_q90=57.0, scorer_config_sha256=sha("scorer-config"),
            scorer_source_sha256=sha("scorer-source"),
            protected_group_sha256s=c64_groups,
            deployed_support_group_sha256s=[sha(f"support-{i}") for i in range(4)])
        self.assertEqual(len(result.snapshot.normal_anchors), 65)
        self.assertEqual(len(result.snapshot.defect_anchors), 1)
        self.assertEqual(result.snapshot.state, "BOTH")
        self.assertEqual(result.snapshot.source_exclusion_audit["memory_sha256"], memory.memory_sha256)
        self.assertEqual(result.snapshot.source_exclusion_audit["source_revision"], memory.source_revision)
        self.assertEqual(len(result.protected_c64_scores), 64)
        self.assertEqual(verify_controller_snapshot_compatibility(result.snapshot), {
            "B4": "PASS_SYNTHETIC_EVIDENCE_INPUT",
            "B7": "PASS_SYNTHETIC_EVIDENCE_INPUT",
            "Stage075BOffset": "PASS_SYNTHETIC_EVIDENCE_INPUT"})

    def test_causal_replay_refreshes_snapshot_after_query_disclosure_and_write(self):
        tapes = synthetic_tape_set()
        native, feedback = approved_records(tapes)
        writer = make_writer()
        events = fixture_event_tape(writer, tapes)
        runner = P11ProductionReplayRunner(native, feedback, tapes)
        memory = __import__("experiments.calibration.stage1a_p1_reconstruction",
                            fromlist=["shared_writer_memory_identity"])
        scorer = NativeAllLayerLSOAdapter(
            writer, lambda payload, bank: ScoreResult(0.8, bank.version, "cpu-current-memory"))
        cache = VersionedAnchorEvidenceCache(runner.runner.budget)
        order = []

        def make_anchor(event, label, features):
            order.append("evidence")
            source = event.source_key
            return QueryAnchor(sha(f"causal-anchor-{event.position}"), sha(source),
                               "normal" if label == 0 else "defect",
                               {"source_key": source, "feature": features})

        adapter = CausalEvidenceRevisionAdapter(
            writer=writer, cache=cache, scorer=scorer, anchor_factory=make_anchor,
            protected_c64_state_fn=lambda state: {
                "scores": tuple(float(i) for i in range(64)),
                "transport_sha256": sha(f"c64:{state.memory_sha256}"), "q90": 57.0},
            scorer_config_sha256=sha("scorer-config"),
            scorer_source_sha256=sha("scorer-source"),
            protected_group_sha256s=[sha(f"c64-group-{i}") for i in range(64)],
            deployed_support_group_sha256s=[sha(f"support-group-{i}") for i in range(4)])
        first_query = tapes.positions("audiojack")[0]
        kwargs = replay_kwargs(writer, events, tapes, max_events=first_query + 1)
        base_writer_fn = kwargs["writer_fn"]
        base_encode = kwargs["encode_feature_fn"]
        base_reader = kwargs["label_reader"]
        def encode(event):
            if event.position == first_query:
                order.append("feature")
            return base_encode(event)
        def read(category, position, source):
            if position == first_query:
                order.append("label")
            return base_reader(category, position, source)
        def write(event, queried, label, features, amap):
            if event.position == first_query:
                order.append("writer")
            return base_writer_fn(event, queried, label, features, amap)
        kwargs.update({"encode_feature_fn": encode, "label_reader": read,
                       "writer_fn": write,
                       "evidence_revision_adapter": adapter})
        result = runner.run(**kwargs)
        self.assertFalse(result.complete)
        self.assertEqual(order, ["feature", "label", "writer", "evidence"])
        self.assertIsNotNone(adapter.last_result)
        self.assertEqual(adapter.last_result.snapshot.memory_version,
                         memory.shared_writer_memory_identity(writer).version)
        self.assertEqual(len(adapter.last_result.snapshot.normal_anchors), 64)
        self.assertEqual(len(adapter.last_result.snapshot.defect_anchors), 1)
        self.assertEqual(adapter.last_result.snapshot.evidence_revision, 2)
        self.assertIsNone(adapter.revisions[0]["position"])
        self.assertEqual(adapter.revisions[1]["position"], first_query)
        self.assertEqual(adapter.revisions[1]["writer_changed_memory"], False)

        private_checkpoint = P11PrivateCheckpoint.capture(
            result=result, production_runner=runner, writer=writer,
            trajectory_id=result.checkpoint.trajectory_id, category="audiojack",
            scorer_config_sha256=sha("scorer-config"),
            scorer_source_sha256=sha("scorer-source"), evidence_cache=cache,
            causal_evidence_adapter=adapter)
        resumed_runner = P11ProductionReplayRunner(native, feedback, tapes)
        resumed_writer = make_writer()
        resumed_cache = VersionedAnchorEvidenceCache(resumed_runner.runner.budget)
        resumed_scorer = NativeAllLayerLSOAdapter(
            resumed_writer, lambda payload, bank: ScoreResult(
                0.8, bank.version, "cpu-current-memory"))
        resumed_adapter = CausalEvidenceRevisionAdapter(
            writer=resumed_writer, cache=resumed_cache, scorer=resumed_scorer,
            anchor_factory=make_anchor,
            protected_c64_state_fn=lambda state: {
                "scores": tuple(float(i) for i in range(64)),
                "transport_sha256": sha(f"c64:{state.memory_sha256}"), "q90": 57.0},
            scorer_config_sha256=sha("scorer-config"),
            scorer_source_sha256=sha("scorer-source"),
            protected_group_sha256s=[sha(f"c64-group-{i}") for i in range(64)],
            deployed_support_group_sha256s=[sha(f"support-group-{i}") for i in range(4)])
        private_checkpoint.restore(
            production_runner=resumed_runner, writer=resumed_writer,
            scorer_config_sha256=sha("scorer-config"),
            scorer_source_sha256=sha("scorer-source"), evidence_cache=resumed_cache,
            causal_evidence_adapter=resumed_adapter)
        resumed_initial = resumed_adapter.initialize(
            memory.shared_writer_memory_identity(resumed_writer))
        self.assertEqual(resumed_initial.evidence_sha256, adapter.last_result.evidence_sha256)
        self.assertEqual(resumed_adapter.cache._revision, adapter.cache._revision)
        self.assertEqual(set(resumed_adapter.anchors), set(adapter.anchors))

    def test_private_checkpoint_reconstructs_writer_caches_budget_and_response_chain(self):
        tapes = synthetic_tape_set()
        native, feedback = approved_records(tapes)
        one_writer = make_writer()
        one_events = fixture_event_tape(one_writer, tapes)
        one_runner = P11ProductionReplayRunner(native, feedback, tapes)
        one_result = one_runner.run(**replay_kwargs(one_writer, one_events, tapes))

        writer_a = make_writer()
        events_a = fixture_event_tape(writer_a, tapes)
        split_runner = P11ProductionReplayRunner(native, feedback, tapes)
        first_query = tapes.positions("audiojack")[0]
        partial = split_runner.run(**replay_kwargs(
            writer_a, events_a, tapes, max_events=first_query + 1))
        self.assertFalse(partial.complete)
        private_checkpoint = P11PrivateCheckpoint.capture(
            result=partial, production_runner=split_runner, writer=writer_a,
            trajectory_id=partial.checkpoint.trajectory_id, category="audiojack",
            scorer_config_sha256=sha("cfg"), scorer_source_sha256=sha("source"))
        serialized = pickle.dumps(private_checkpoint, protocol=5)
        reconstructed_private = pickle.loads(serialized)
        self.assertNotIn("values", str(reconstructed_private.public_summary()))
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_file = Path(temp_dir) / "synthetic-private-checkpoint.bin"
            checkpoint_file.write_bytes(serialized)
            child_code = (
                "import importlib.util,json,sys; "
                "p=sys.argv[2]; s=importlib.util.spec_from_file_location('p11fixture',p); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "print(json.dumps(m.resume_from_serialized_checkpoint(sys.argv[1])))")
            child = subprocess.run(
                [sys.executable, "-c", child_code, str(checkpoint_file),
                 str(Path(__file__).resolve())],
                cwd=str(ROOT), capture_output=True, text=True, check=True)
        resumed = json.loads(child.stdout.strip().splitlines()[-1])
        self.assertTrue(resumed["complete"])
        self.assertEqual(resumed["event_chain_sha256"], one_result.event_chain_sha256)
        self.assertEqual(resumed["query_response_sha256"], one_result.query_response_sha256)
        self.assertEqual(resumed["writer_state_sha256"], writer_state_hash(one_writer))
        self.assertEqual(resumed["budget"]["queried_label_reads"],
                         one_runner.runner.budget.queried_label_reads)
        self.assertEqual(resumed["budget"]["feature_encodings"],
                         one_runner.runner.budget.feature_encodings)

    def test_private_checkpoint_rejects_label_cache_chain_mutation(self):
        tapes = synthetic_tape_set()
        native, feedback = approved_records(tapes)
        writer = make_writer()
        events = fixture_event_tape(writer, tapes)
        runner = P11ProductionReplayRunner(native, feedback, tapes)
        cursor = tapes.positions("audiojack")[0] + 1
        partial = runner.run(**replay_kwargs(writer, events, tapes, max_events=cursor))
        checkpoint = P11PrivateCheckpoint.capture(
            result=partial, production_runner=runner, writer=writer,
            trajectory_id=partial.checkpoint.trajectory_id, category="audiojack",
            scorer_config_sha256=sha("cfg"), scorer_source_sha256=sha("source"))
        mutated = pickle.loads(pickle.dumps(checkpoint, protocol=5))
        mutated.queried_label_state = dict(mutated.queried_label_state)
        mutated.queried_label_state["values"] = [list(row) for row in mutated.queried_label_state["values"]]
        mutated.queried_label_state["values"][0][3] = 0
        target_runner = P11ProductionReplayRunner(native, feedback, tapes)
        with self.assertRaisesRegex(P11IntegrityError, "response-chain digest"):
            mutated.restore(production_runner=target_runner, writer=make_writer(),
                            scorer_config_sha256=sha("cfg"),
                            scorer_source_sha256=sha("source"))


if __name__ == "__main__":
    unittest.main()
