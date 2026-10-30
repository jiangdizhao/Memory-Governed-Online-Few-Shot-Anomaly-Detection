"""Synthetic-only tests for Stage 1-A controller-only readiness."""
from __future__ import annotations

import copy
import hashlib
import math
import json
import tempfile
import unittest
from pathlib import Path

from experiments.calibration.protocol import EvidenceAnchor, EvidenceSnapshot, Observation, Prediction
from experiments.calibration.profiles import canonical_json
from experiments.calibration.stage1a_final_readiness import (
    ControllerOnlyReplay, MemoryIdentity, ReadinessError, ReplayEvent,
    _method_registry, build_controller_suite, p12_row_to_evidence_snapshot,
    synthetic_metric_contract, verify_recorded_writer_transition,
    P12_V2_REQUIRED_AUDIT_FILES, verify_p12_canonical_audit_linkage,
)


def h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def memory(version: int = 0, content: str = "m0") -> MemoryIdentity:
    return MemoryIdentity(version, h(content), h("source-" + content), version, version)


def evidence(mem: MemoryIdentity, revision: int = 0, *, event_position=None,
             defect: bool = False, evidence_digest: str | None = None,
             query_anchor_id: str | None = None) -> EvidenceSnapshot:
    protected = tuple(EvidenceAnchor(f"c64:{i}", i / 100.0, mem.version, True)
                      for i in range(64))
    query_anchor = ((EvidenceAnchor(query_anchor_id, .91 if defect else .41,
                                    mem.version, False,
                                    source_exclusion_policy=(
                                        "queried_defect_not_written" if defect else
                                        "queried_normal_all_layer_leave_source_out")),)
                    if query_anchor_id else ())
    normals = protected + (() if defect else query_anchor)
    defects = query_anchor if defect else ()
    digest = evidence_digest or h(f"evidence-{revision}")
    audit = {
        "memory_version": mem.version,
        "memory_sha256": mem.memory_sha256,
        "source_map_sha256": mem.source_map_sha256,
        "source_revision": mem.source_revision,
        "legacy_greedy_version": mem.legacy_greedy_version,
        "c64_transport_sha256": h(f"c64-{revision}"),
        "evidence_sha256": digest,
        "event_position": event_position,
        "category": "audiojack",
        "support_seed": 0,
    }
    return EvidenceSnapshot(mem.version, normals, defects, audit, revision)


class SpyController:
    def __init__(self, name: str, mutate_evidence: bool = False):
        self.name = name
        self.theta = .5
        self.labels_seen: list[tuple[int, int | None, bool]] = []
        self.scores: list[float] = []
        self.evidence_revisions: list[int] = []
        self.call_order: list[tuple] = []
        self.mutate_evidence = mutate_evidence

    def predict(self, observation):
        return Prediction(int(observation.score >= self.theta), self.theta)

    def observe_feedback(self, observation, prediction, feedback):
        self.labels_seen.append((feedback.t, feedback.label, feedback.queried))
        self.call_order.append(("feedback", feedback.t, feedback.label, feedback.queried))

    def observe_score(self, observation):
        self.scores.append(observation.score)
        self.call_order.append(("score", observation.t))

    def on_memory_change(self, item):
        self.evidence_revisions.append(item.evidence_revision)
        self.call_order.append(("evidence", item.evidence_revision))
        if self.mutate_evidence:
            item.source_exclusion_audit["mutated"] = True

    def finalize_step(self):
        return self.theta

    def state_dict(self):
        return {"name": self.name, "theta": self.theta,
                "labels_seen": list(self.labels_seen), "scores": list(self.scores),
                "evidence_revisions": list(self.evidence_revisions)}

    def load_state_dict(self, state):
        if state.get("name") != self.name:
            raise ValueError("wrong controller")
        self.theta = float(state["theta"])
        self.labels_seen = [tuple(row) for row in state["labels_seen"]]
        self.scores = list(state["scores"])
        self.evidence_revisions = list(state["evidence_revisions"])


class MutatingObservationController(SpyController):
    def predict(self, observation):
        object.__setattr__(observation, "score", 999.0)
        return Prediction(1, self.theta)


def event(t: int, *, queried=False, label=None, before=None, after=None,
          ev=None, ev_digest=None, q90_digest=None, query_anchor_id=None):
    before = before or memory()
    after = after or before
    obs = Observation(t, f"synthetic-{t}", 1, .25 + t / 10, before.version, h(f"score-{t}"))
    return ReplayEvent(t, obs, queried, label, before, after, ev, ev_digest,
                       q90_digest, "audiojack", 0,
                       query_anchor_id or (f"query:synthetic-{t}" if queried else None),
                       h(f"official-group-{t}") if queried else None)


class Stage1AFinalReadinessTests(unittest.TestCase):
    def test_p12_v2_complete_canonical_lf_inventory_and_legacy_raw_caveat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {}
            for name in P12_V2_REQUIRED_AUDIT_FILES:
                # A canonical LF manifest also accepts a CRLF checkout.
                text = "# verified audit\\n" if name.endswith(".md") else "verified\\n"
                if name in ("STAGE1A_P12_LOW_MEMORY_RECOVERY_REPORT.md",
                            "STAGE1A_P12_LOW_MEMORY_RECOVERY_REPORT_FINAL.md"):
                    text = text.replace("\\n", "\n")
                    (root / name).write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
                else:
                    text = text.replace("\\n", "\n")
                    (root / name).write_bytes(text.encode("utf-8"))
                payload = text.encode("utf-8")
                files[name] = {"hash_mode": "canonical_lf_text", "sha256":
                               hashlib.sha256(payload).hexdigest(),
                               "size_bytes": len(payload)}
            (root / "checksums_final.json").write_text(json.dumps({
                "files": {"STAGE1A_P12_LOW_MEMORY_RECOVERY_REPORT_FINAL.md": {
                    "sha256": h("missing original raw bytes")}}}), encoding="utf-8")
            # The legacy manifest is intentionally rewritten above to contain
            # an unresolved raw-byte pointer. The v2 inventory must hash those
            # FINAL fixture bytes, not its earlier placeholder payload.
            legacy_final_bytes = (root / "checksums_final.json").read_bytes()
            files["checksums_final.json"] = {
                "hash_mode": "canonical_lf_text",
                "sha256": hashlib.sha256(legacy_final_bytes).hexdigest(),
                "size_bytes": len(legacy_final_bytes),
            }
            (root / "checksums_final_v2.json").write_text(json.dumps({
                "schema": "stage1a.p12_low_memory_recovery_audit_checksums.v2",
                "file_count": 17, "self_excluded_manifest": "checksums_final_v2.json",
                "normalization": "Decode as UTF-8; replace CRLF and bare CR with LF.",
                "files": files}), encoding="utf-8")
            complete = {"final_audit_report_sha256": h("unrecovered raw report")}
            passed = verify_p12_canonical_audit_linkage(root, complete)
            self.assertEqual(passed["status"], "PASS_CANONICAL_LF_V2_COMPLETE")
            self.assertEqual(passed["verified_file_count"], 17)
            self.assertTrue(passed["legacy_raw_byte_discrepancies"])
            self.assertEqual(passed["historical_g2_status"], "BLOCKED_UNCHANGED")
            # Real semantic modification must fail closed, unlike newline changes.
            report = root / "STAGE1A_P12_LOW_MEMORY_RECOVERY_REPORT_FINAL.md"
            report.write_text("tampered content\\n", encoding="utf-8")
            with self.assertRaisesRegex(ReadinessError, "checksum differs"):
                verify_p12_canonical_audit_linkage(root, complete)
            # Deleting an inventory entry may not produce a false PASS.
            v2 = root / "checksums_final_v2.json"
            broken = json.loads(v2.read_text(encoding="utf-8"))
            broken["files"].pop("source_provenance.json")
            v2.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ReadinessError, "schema/inventory differs"):
                verify_p12_canonical_audit_linkage(root, complete)
            broken["files"] = files
            broken["schema"] = "unknown.v3"
            v2.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ReadinessError, "schema/inventory differs"):
                verify_p12_canonical_audit_linkage(root, complete)

    def test_historical_writer_record_schema_and_transition_are_checked(self):
        mem = memory()
        record = {
            "eligible": False, "changed": False, "keep": 0, "reason": "ineligible",
            "appended": 0, "replaced": 0, "skipped": 0,
            "source_map_changed": False,
            "source_revision_before": 0, "source_revision_after": 0,
            "memory_content_version_before": 0, "memory_content_version_after": 0,
            "legacy_greedy_version_before": 0, "legacy_greedy_version_after": 0,
            "dynamic_rows_by_layer": [0, 0], "capacity_by_layer": [256, 256],
            "source_map_sha256": mem.source_map_sha256,
            "resident_source_groups_by_layer_sha256": [{}, {}],
        }
        item = ReplayEvent(0, Observation(0, "writer-fixture", 1, .1, 0, h("score")),
                           False, None, mem, mem, writer_record=record)
        verify_recorded_writer_transition(item)

        missing = dict(record)
        del missing["resident_source_groups_by_layer_sha256"]
        with self.assertRaisesRegex(ReadinessError, "historical schema"):
            verify_recorded_writer_transition(ReplayEvent(
                0, item.observation, False, None, mem, mem, writer_record=missing))

        mutated = dict(record, source_map_changed=True)
        with self.assertRaisesRegex(ReadinessError, "source-map action"):
            verify_recorded_writer_transition(ReplayEvent(
                0, item.observation, False, None, mem, mem, writer_record=mutated))

    def test_query_response_chain_uses_receipt_identity_and_rejects_incomplete(self):
        mem = memory()
        anchor_id = "query:synthetic-0"
        ev = evidence(mem, 1, event_position=0, query_anchor_id=anchor_id)
        runner = ControllerOnlyReplay(controllers={"B1": SpyController("B1")},
            query_positions={0}, input_binding_sha256=h("binding"),
            initial_memory=mem, initial_evidence=evidence(mem, 0))
        with self.assertRaisesRegex(ReadinessError, "before all taped events"):
            runner.verify_query_response_chain(h("expected"))
        item = event(0, queried=True, label=0, before=mem, after=mem,
                     ev=ev, ev_digest=ev.source_exclusion_audit["evidence_sha256"],
                     q90_digest=ev.source_exclusion_audit["c64_transport_sha256"],
                     query_anchor_id=anchor_id)
        runner.step(item)
        chain = hashlib.sha256(b"stage1a-p1-query-response-v1").hexdigest()
        response = {"position": 0,
                    "source_identity_sha256": item.query_source_identity_sha256,
                    "label": 0}
        expected = hashlib.sha256(bytes.fromhex(chain) +
            canonical_json(response).encode("ascii")).hexdigest()
        self.assertEqual(runner.verify_query_response_chain(expected), expected)
        with self.assertRaisesRegex(ReadinessError, "differ from the P1.2 receipt"):
            runner.verify_query_response_chain(h("wrong"))

    def test_predict_precedes_feedback_and_future_response_cannot_change_prior(self):
        first_results = []
        for future_label in (0, 1):
            controller = SpyController("B1")
            runner = ControllerOnlyReplay(controllers={"B1": controller},
                query_positions={1}, input_binding_sha256=h("binding"), initial_memory=memory())
            pred0 = runner.step(event(0))["B1"]
            mem1 = memory(1, "m1")
            anchor_id = "query:synthetic-1"
            ev1 = evidence(mem1, 1, event_position=1, evidence_digest=h("ev1"),
                           defect=(future_label == 1), query_anchor_id=anchor_id)
            pred1 = runner.step(event(1, queried=True, label=future_label,
                before=memory(), after=mem1, ev=ev1, ev_digest=h("ev1"),
                q90_digest=ev1.source_exclusion_audit["c64_transport_sha256"],
                query_anchor_id=anchor_id))["B1"]
            self.assertEqual(pred1.label, 0)  # score=.35 is below the frozen .5 threshold
            first_results.append(pred0.label)
        self.assertEqual(first_results, [0, 0])

    def test_memory_evidence_precedes_controller_feedback_and_score_update(self):
        mem = memory()
        controller = SpyController("B4")
        runner = ControllerOnlyReplay(controllers={"B4": controller},
            query_positions={0}, input_binding_sha256=h("binding"),
            initial_memory=mem, initial_evidence=evidence(mem, 0))
        anchor_id = "query:synthetic-0"
        ev = evidence(mem, 1, event_position=0, query_anchor_id=anchor_id)
        runner.step(event(0, queried=True, label=0, before=mem, after=mem,
            ev=ev, ev_digest=ev.source_exclusion_audit["evidence_sha256"],
            q90_digest=ev.source_exclusion_audit["c64_transport_sha256"],
            query_anchor_id=anchor_id))
        self.assertEqual(controller.call_order[-3:], [
            ("evidence", 1), ("feedback", 0, 0, True), ("score", 0)])

    def test_unqueried_label_is_rejected_and_not_delivered(self):
        controller = SpyController("B0")
        runner = ControllerOnlyReplay(controllers={"B0": controller},
            query_positions=set(), input_binding_sha256=h("binding"), initial_memory=memory())
        with self.assertRaises(ReadinessError):
            runner.step(event(0, queried=False, label=1))
        self.assertEqual(controller.labels_seen, [])

    def test_unqueried_outcome_changes_are_outside_every_controller_input(self):
        states = []
        for private_outcome in (0, 1):
            # private_outcome deliberately is not passed to the replay adapter.
            _ = private_outcome
            controller = SpyController("B0")
            runner = ControllerOnlyReplay(controllers={"B0": controller},
                query_positions=set(), input_binding_sha256=h("binding"), initial_memory=memory())
            runner.step(event(0, queried=False, label=None))
            states.append(runner.state_dict())
        self.assertEqual(states[0], states[1])

    def test_one_controller_cannot_mutate_another_controllers_shared_observation(self):
        mutator = MutatingObservationController("mutator")
        observer = SpyController("observer")
        base = memory()
        runner = ControllerOnlyReplay(controllers={"mutator": mutator,
            "observer": observer}, query_positions=set(),
            input_binding_sha256=h("independent-inputs"), initial_memory=base)
        runner.step(event(0))
        self.assertEqual(observer.scores, [.25])
        self.assertEqual(runner.last_memory, base)

    def test_missing_evidence_at_a_memory_change_fails_closed(self):
        runner = ControllerOnlyReplay(controllers={"B1": SpyController("B1")},
            query_positions={0}, input_binding_sha256=h("binding"), initial_memory=memory())
        with self.assertRaisesRegex(ReadinessError, "lacks its causal evidence"):
            runner.step(event(0, queried=True, label=0, before=memory(),
                              after=memory(1, "changed")))

    def test_mutated_evidence_identity_and_wrong_boundary_fail(self):
        base = memory()
        changed = memory(1, "changed")
        wrong = evidence(changed, 1, event_position=9, evidence_digest=h("ev"),
                         query_anchor_id="query:synthetic-0")
        runner = ControllerOnlyReplay(controllers={"B4": SpyController("B4")},
            query_positions={0}, input_binding_sha256=h("binding"), initial_memory=base)
        with self.assertRaisesRegex(ReadinessError, "wrong event boundary"):
            runner.step(event(0, queried=True, label=0, before=base, after=changed,
                              ev=wrong, ev_digest=h("ev"),
                              q90_digest=wrong.source_exclusion_audit["c64_transport_sha256"],
                              query_anchor_id="query:synthetic-0"))
        mutated = evidence(changed, 1, event_position=0, evidence_digest=h("right"),
                           query_anchor_id="query:synthetic-0")
        runner2 = ControllerOnlyReplay(controllers={"B4": SpyController("B4")},
            query_positions={0}, input_binding_sha256=h("binding"), initial_memory=base)
        with self.assertRaisesRegex(ReadinessError, "content identity"):
            runner2.step(event(0, queried=True, label=0, before=base, after=changed,
                               ev=mutated, ev_digest=h("different"),
                               q90_digest=mutated.source_exclusion_audit["c64_transport_sha256"],
                               query_anchor_id="query:synthetic-0"))

    def test_p12_row_conversion_binds_category_seed_event_and_memory(self):
        mem = memory(3, "p12-memory")
        audit = {"memory_version": 3, "memory_sha256": mem.memory_sha256,
                 "source_map_sha256": mem.source_map_sha256,
                 "source_revision": 3, "legacy_greedy_version": 3,
                 "evidence_revision": 8, "evidence_sha256": h("p12-evidence"),
                 "c64_transport_sha256": h("transport")}
        protected = [{"opaque_anchor_id": f"c64:{i}", "score": i / 100,
                      "memory_version": 3, "protected": True,
                      "source_excluded_rows": 0,
                      "source_exclusion_policy": "protected_no_selfwrite"}
                     for i in range(64)]
        queried = {"opaque_anchor_id": "query:exact-anchor", "score": .42,
                   "memory_version": 3, "protected": False,
                   "source_excluded_rows": 16,
                   "source_exclusion_policy": "queried_normal_all_layer_leave_source_out"}
        row = {"schema": "stage1a.p12_private_evidence_snapshot.v1",
               "position": 17, "memory_version": 3,
               "memory_sha256": mem.memory_sha256,
               "source_map_sha256": mem.source_map_sha256,
               "evidence_revision": 8, "evidence_sha256": h("p12-evidence"),
               "source_exclusion_audit": audit,
               "normal_anchors": protected + [queried], "defect_anchors": []}
        converted = p12_row_to_evidence_snapshot(row, category="audiojack",
            support_seed=2, expected_position=17)
        self.assertEqual(converted.source_exclusion_audit["category"], "audiojack")
        self.assertEqual(converted.source_exclusion_audit["support_seed"], 2)
        self.assertEqual(converted.source_exclusion_audit["event_position"], 17)
        self.assertEqual(converted.source_exclusion_audit["memory_sha256"], mem.memory_sha256)
        self.assertEqual(len(converted.normal_anchors), 65)
        with self.assertRaisesRegex(ReadinessError, "position"):
            p12_row_to_evidence_snapshot(row, category="audiojack",
                support_seed=2, expected_position=18)

    def test_duplicate_or_stale_evidence_revision_fails(self):
        base = memory()
        c = SpyController("B4")
        initial = evidence(base, 0)
        runner = ControllerOnlyReplay(controllers={"B4": c}, query_positions={0, 1},
            input_binding_sha256=h("binding"), initial_memory=base, initial_evidence=initial)
        changed = memory(1, "changed")
        ev1 = evidence(changed, 1, event_position=0, evidence_digest=h("ev1"),
                       query_anchor_id="query:synthetic-0")
        runner.step(event(0, queried=True, label=0, before=base, after=changed,
            ev=ev1, ev_digest=h("ev1"),
            q90_digest=ev1.source_exclusion_audit["c64_transport_sha256"],
            query_anchor_id="query:synthetic-0"))
        changed2 = memory(2, "changed2")
        stale = evidence(changed2, 1, event_position=1, evidence_digest=h("stale"),
                         defect=True, query_anchor_id="query:synthetic-1")
        with self.assertRaisesRegex(ReadinessError, "duplicate or out of order"):
            runner.step(event(1, queried=True, label=1, before=changed, after=changed2,
                ev=stale, ev_digest=h("stale"),
                q90_digest=stale.source_exclusion_audit["c64_transport_sha256"],
                query_anchor_id="query:synthetic-1"))

    def test_controller_evidence_mutation_is_isolated(self):
        base = memory()
        mutator = SpyController("B4", mutate_evidence=True)
        observer = SpyController("Stage075BOffset")
        initial = evidence(base, 0)
        runner = ControllerOnlyReplay(controllers={"B4": mutator,
            "Stage075BOffset": observer}, query_positions=set(),
            input_binding_sha256=h("binding"), initial_memory=base,
            initial_evidence=initial)
        self.assertNotIn("mutated", initial.source_exclusion_audit)
        self.assertEqual(mutator.evidence_revisions, [0])
        self.assertEqual(observer.evidence_revisions, [0])
        result = runner.step(event(0))
        self.assertEqual(set(result), {"B4", "Stage075BOffset"})
        self.assertEqual(runner.trace[0]["memory_sha256_before"], base.memory_sha256)

    def test_checkpoint_resume_matches_one_shot_synthetic_replay(self):
        base = memory()
        ev0 = evidence(base, 0)
        changed = memory(1, "changed")
        ev1 = evidence(changed, 1, event_position=1, evidence_digest=h("ev1"),
                       query_anchor_id="query:synthetic-1")
        events = [
            event(0, before=base, after=base),
            event(1, queried=True, label=0, before=base, after=changed, ev=ev1,
                  ev_digest=h("ev1"),
                  q90_digest=ev1.source_exclusion_audit["c64_transport_sha256"],
                  query_anchor_id="query:synthetic-1"),
        ]
        one = ControllerOnlyReplay(controllers={"B1": SpyController("B1")},
            query_positions={1}, input_binding_sha256=h("binding"),
            initial_memory=base, initial_evidence=ev0)
        for item in events:
            one.step(item)
        split = ControllerOnlyReplay(controllers={"B1": SpyController("B1")},
            query_positions={1}, input_binding_sha256=h("binding"),
            initial_memory=base, initial_evidence=ev0)
        split.step(events[0])
        saved = copy.deepcopy(split.state_dict())
        restored = ControllerOnlyReplay(controllers={"B1": SpyController("B1")},
            query_positions={1}, input_binding_sha256=h("binding"),
            initial_memory=base, initial_evidence=ev0)
        restored.load_state_dict(saved)
        restored.step(events[1])
        self.assertEqual(restored.state_dict(), one.state_dict())
        self.assertNotIn("queried_label", saved)

    def test_actual_registered_controller_suite_smoke_and_startup_modes(self):
        c64 = [index / 100 for index in range(64)]
        suite = build_controller_suite(.58, c64)
        self.assertEqual(len(suite), 11)
        self.assertIsNot(suite["B5a"], suite["B5a_FROZEN_FALLBACK"].source)
        self.assertIsNot(suite["B5b"], suite["B5b_FROZEN_FALLBACK"].source)
        base = memory()
        runner = ControllerOnlyReplay(controllers=suite,
            query_positions={1}, input_binding_sha256=h("registry-binding"),
            initial_memory=base, initial_evidence=evidence(base, 0))
        first = runner.step(event(0))["B5a"]
        self.assertFalse(first.available)
        self.assertEqual(first.reason, "WARMUP_UNAVAILABLE")
        fallback = runner.predictions["B5a_FROZEN_FALLBACK"][0]
        self.assertTrue(fallback.available)
        self.assertEqual(fallback.reason, "STARTUP_FALLBACK")
        self.assertEqual(runner.controllers["B7"].state, "NORMAL_ONLY")
        self.assertEqual(runner.controllers["B3"].normal_scores.__len__(), 64)
        self.assertEqual(runner.controllers["B3"].fnr_status, "FNR_UNAVAILABLE")
        self.assertEqual(runner.controllers["Stage075BOffset"].theta, .58)
        self.assertEqual(runner.controllers["B1"].window, 256)
        self.assertEqual(runner.controllers["B1"].min_count, 16)
        self.assertEqual(runner.controllers["B3"].delta, .01)
        self.assertEqual(runner.controllers["B7"].config.min_defects, 8)
        self.assertEqual(runner.controllers["B7"].config.confidence_delta, .05)
        self.assertEqual(runner.controllers["B5a"].config.n_init, 1000)
        self.assertEqual(runner.controllers["B5b"].window, 50)
        changed = memory(1, "changed")
        ev1 = evidence(changed, 1, event_position=1, defect=True, evidence_digest=h("ev1"),
                       query_anchor_id="query:synthetic-1")
        second = runner.step(event(1, queried=True, label=1, before=base, after=changed,
            ev=ev1, ev_digest=h("ev1"),
            q90_digest=ev1.source_exclusion_audit["c64_transport_sha256"],
            query_anchor_id="query:synthetic-1"))
        self.assertEqual(set(second), set(suite))
        self.assertEqual(runner.controllers["Stage075BOffset"].name, "Stage075BOffset")
        self.assertEqual(runner.controllers["Stage075BOffset"].version,
                         "stage075b-b3-raw-median-offset-v1")

    def test_metric_null_denominators_startup_fallback_and_full_windows(self):
        labels = [0, 1] * 640
        scores = [float(value) for value in labels]
        n = len(labels)
        native = [None, None] + [int(score >= .5) for score in scores[2:]]
        fallback = [int(score >= .5) for score in scores]
        reasons_native = ["WARMUP_UNAVAILABLE", "WARMUP_UNAVAILABLE"] + [
            "ORIGINAL_METHOD"] * (n - 2)
        reasons_fallback = ["STARTUP_FALLBACK", "STARTUP_FALLBACK"] + [
            "ORIGINAL_METHOD"] * (n - 2)
        result = synthetic_metric_contract(labels, scores,
            {"B5a": native, "B5a_FROZEN_FALLBACK": fallback},
            availability={"B5a": [False, False] + [True] * (n - 2),
                          "B5a_FROZEN_FALLBACK": [True] * n},
            reasons={"B5a": reasons_native,
                     "B5a_FROZEN_FALLBACK": reasons_fallback})
        self.assertIsNone(result["methods"]["B5a"]["balanced_accuracy"])
        self.assertEqual(result["methods"]["B5a"]["prediction_modes"]["startup_unavailable"], 2)
        self.assertEqual(result["methods"]["B5a_FROZEN_FALLBACK"]["prediction_modes"]["fallback_fraction"], 2 / n)
        self.assertEqual(result["methods"]["B5a"]["worst_defined_full_window_balanced_accuracy"], 1.0)
        self.assertEqual(len(result["methods"]["B5a"]["full_256_image_windows_only"]), 5)
        self.assertEqual(result["methods"]["B5a_FROZEN_FALLBACK"]["defect_recall"], 1.0)
        self.assertEqual(result["shared_score_auroc"], 1.0)
        self.assertEqual(result["shared_score_auprc"], 1.0)
        self.assertEqual(result["common_horizon"]["result"]["n"], 230)
        one_class = synthetic_metric_contract([0] * 16, [float(i) for i in range(16)],
            {"B0": [0] * 16})
        self.assertIsNone(one_class["methods"]["B0"]["fnr"])
        self.assertIsNone(one_class["methods"]["B0"]["balanced_accuracy"])
        self.assertIsNone(one_class["methods"]["B0"]["defect_coverage"])

    def test_frozen_registry_keeps_applicability_and_historical_only_rows_separate(self):
        registry = _method_registry()
        self.assertEqual(set(registry["execution_scope"]), {
            "B0", "B1", "B2", "B3", "B4", "B5a", "B5a_FROZEN_FALLBACK",
            "B5b", "B5b_FROZEN_FALLBACK", "B7", "Stage075BOffset"})
        self.assertEqual(registry["applicability_only"], ["B6"])
        self.assertEqual(registry["historical_only"], ["B8"])
        self.assertTrue(registry["methods"]["Stage075BOffset"]["configuration"][
            "not_additive_b3_offset"])
        matrix = __import__(
            "experiments.calibration.stage1a_final_readiness", fromlist=["_controller_input_matrix"]
        )._controller_input_matrix(registry)
        self.assertTrue(matrix["methods"]["B7"]["score"])
        self.assertTrue(matrix["methods"]["Stage075BOffset"]["score"])

    def test_threshold_changes_do_not_change_shared_score_ranking_metrics(self):
        labels = [0, 1, 0, 1] * 300
        scores = [0.0, 1.0, 0.2, 0.8] * 300
        first = synthetic_metric_contract(labels, scores,
            {"B0": [int(x >= .5) for x in scores],
             "B1": [int(x >= .7) for x in scores]})
        second = synthetic_metric_contract(labels, scores,
            {"B0": [int(x >= .2) for x in scores],
             "B1": [int(x >= .9) for x in scores]})
        self.assertEqual((first["shared_score_auroc"], first["shared_score_auprc"]),
                         (second["shared_score_auroc"], second["shared_score_auprc"]))
        self.assertEqual(first["common_horizon"]["start_zero_based"], 1050)


if __name__ == "__main__":
    unittest.main()
