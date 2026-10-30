from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import unittest

from experiments.calibration import stage1a_p12_sharded_recovery as recovery
from experiments.calibration.stage1a_p11_production import P11_CLASSES
from experiments.calibration.stage1a_p1_reconstruction import (
    C64Identity, MemoryIdentity, ReplayEvent, sha256_json)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def receipt(category: str, seed: int, query_count: int,
            event_count: int, tape_sha: str, source_sha: str,
            scorer_sha: str) -> dict:
    memory = {
        "version": seed,
        "memory_sha256": digest(f"memory:{category}:{seed}"),
        "source_map_sha256": digest(f"source:{category}:{seed}"),
        "source_revision": seed,
        "legacy_greedy_version": seed,
    }
    return {
        "schema": recovery.RECEIPT_SCHEMA,
        "trajectory_id": f"{category}/seed{seed}/Q-A-TS-v1-3pct-seed37",
        "category": category,
        "support_seed": seed,
        "query_tape_binding_sha256": tape_sha,
        "query_positions_sha256": digest(f"positions:{category}"),
        "query_count": query_count,
        "expected_event_count": event_count,
        "event_count": event_count,
        "status": "PASS_EXACT_PARITY",
        "complete": True,
        "event_chain_sha256": digest(f"event:{category}:{seed}"),
        "query_response_chain_sha256": digest(f"response:{category}"),
        "initial_memory": {**memory, "version": 0, "source_revision": 0,
                            "legacy_greedy_version": 0},
        "final_memory": memory,
        "source_fingerprint_sha256": source_sha,
        "scorer_config_sha256": scorer_sha,
        "scorer_source_sha256": digest("scorer source"),
        "evidence_revision_count": query_count + 1,
        "evidence_file_sha256": digest(f"evidence:{category}:{seed}"),
        "evidence_source": "reused_verified_artifact",
        "last_successful_position": event_count - 1,
        "counter_delta": {
            "queried_label_reads": query_count if seed == 0 else 0,
            "feature_encodings": query_count if seed == 0 else 0,
            "static_map_calls": query_count,
            "anchor_score_calls": 0,
            "anchor_score_requests": query_count,
            "reused_evidence_score_entries": query_count,
        },
        "cumulative_class_counters": {},
        "memory_measurements": {},
    }


def evidence_row(position: int | None, revision: int,
                 query_anchor_count: int,
                 anchor_digest: str | None = None) -> dict:
    memory_sha = digest("evidence memory")
    source_sha = digest("evidence source map")
    transport_sha = digest("c64 transport")
    scorer_config = digest("config")
    scorer_source = digest("scorer")
    c64 = [float(i) for i in range(64)]
    q90 = sorted(c64)[57]
    normal = [{
        "opaque_anchor_id": f"c64:{i:03d}", "score": float(value),
        "memory_version": 0, "protected": True,
        "source_excluded_rows": 0,
        "source_exclusion_policy": "protected_no_selfwrite",
    } for i, value in enumerate(c64)]
    defects = []
    proofs = {}
    if query_anchor_count:
        anchor_digest = anchor_digest or digest("query anchor")
        opaque_id = "query:" + anchor_digest
        defects.append({
            "opaque_anchor_id": opaque_id, "score": 0.5,
            "memory_version": 0, "protected": False,
            "source_excluded_rows": 0,
            "source_exclusion_policy": "queried_defect_not_written",
        })
        proofs[anchor_digest] = {
            "memory_version": 0, "memory_sha256": memory_sha,
            "source_map_sha256": source_sha, "layer_count": 2,
            "removed_rows_by_layer": [0, 0],
            "remaining_source_rows_by_layer": [0, 0],
            "source_exclusion_semantics": "queried_defect_not_written",
        }
    audit = {
        "memory_version": 0, "memory_sha256": memory_sha,
        "source_map_sha256": source_sha, "evidence_revision": revision,
        "evidence_sha256": "", "source_revision": 0,
        "legacy_greedy_version": 0, "protected_c64_count": 64,
        "queried_normal_lso_count": 0,
        "queried_defect_current_memory_count": query_anchor_count,
        "c64_q90_higher_index_57": q90,
        "c64_transport_sha256": transport_sha,
        "scorer_config_sha256": scorer_config,
        "scorer_source_sha256": scorer_source,
        "source_exclusion_proofs": proofs,
        "raw_source_keys_or_labels_in_snapshot": False,
    }
    evidence_doc = {
        "memory_version": 0, "memory_sha256": memory_sha,
        "source_map_sha256": source_sha,
        "c64_transport_sha256": transport_sha, "c64_q90": q90,
        "normal_scores": {},
        "defect_scores": ({anchor_digest: 0.5} if query_anchor_count else {}),
    }
    evidence_sha = sha256_json(evidence_doc)
    audit["evidence_sha256"] = evidence_sha
    return {
        "schema": "stage1a.p12_private_evidence_snapshot.v1",
        "position": position, "memory_version": 0,
        "memory_sha256": memory_sha, "source_map_sha256": source_sha,
        "evidence_revision": revision, "evidence_sha256": evidence_sha,
        "protected_c64_scores": c64, "normal_anchors": normal,
        "defect_anchors": defects, "source_exclusion_proofs": proofs,
        "source_exclusion_audit": audit,
    }


class ShardedRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tape_sha = digest("tape")
        self.source_sha = digest("source fingerprint")
        self.scorer_sha = digest("scorer config")
        self.query_counts = dict(zip(P11_CLASSES, (115, 115, 115, 114)))
        self.event_counts = {category: 1000 + i
                             for i, category in enumerate(P11_CLASSES)}

    def valid_receipts(self) -> list[dict]:
        return [receipt(category, seed, self.query_counts[category],
                        self.event_counts[category], self.tape_sha,
                        self.source_sha, self.scorer_sha)
                for category in P11_CLASSES for seed in range(5)]

    def test_windows_manifest_paths_normalize_without_changing_identity(self):
        self.assertEqual(
            recovery._portable_repo_path(
                r"outputs\calibration\run\private\evidence.jsonl"),
            "outputs/calibration/run/private/evidence.jsonl")

    def test_twenty_shards_merge_deterministically_and_enforce_global_counts(self):
        merged = recovery.merge_completion_receipts(
            self.valid_receipts(), tape_binding_sha256=self.tape_sha,
            query_counts=self.query_counts,
            query_positions_sha256_by_class={
                category: digest(f"positions:{category}") for category in P11_CLASSES},
            event_counts=self.event_counts,
            expected_scorer_config_sha256=self.scorer_sha,
            expected_scorer_source_sha256=digest("scorer source"))
        self.assertEqual(merged["trajectory_count"], 20)
        self.assertEqual(merged["unique_queried_positions"], 459)
        self.assertEqual(merged["query_events_across_five_support_seeds"], 2295)
        self.assertEqual(merged["queried_label_disclosures"], 459)
        self.assertEqual(merged["physical_query_feature_encodings"], 459)
        self.assertEqual(merged["native_static_map_calls"], 2295)
        self.assertEqual(merged["trajectory_order"], [
            f"{category}/seed{seed}/Q-A-TS-v1-3pct-seed37"
            for category in P11_CLASSES for seed in range(5)])
        shuffled = list(reversed(self.valid_receipts()))
        shuffled_result = recovery.merge_completion_receipts(
            shuffled, tape_binding_sha256=self.tape_sha,
            query_counts=self.query_counts,
            query_positions_sha256_by_class={
                category: digest(f"positions:{category}") for category in P11_CLASSES},
            event_counts=self.event_counts,
            expected_scorer_config_sha256=self.scorer_sha,
            expected_scorer_source_sha256=digest("scorer source"))
        self.assertEqual(merged, shuffled_result)

    def test_duplicate_and_missing_shards_fail_closed(self):
        rows = self.valid_receipts()
        with self.assertRaisesRegex(recovery.RecoveryError, "Duplicate"):
            recovery.merge_completion_receipts(
                [*rows, rows[0]], tape_binding_sha256=self.tape_sha,
                query_counts=self.query_counts,
                query_positions_sha256_by_class={
                    category: digest(f"positions:{category}") for category in P11_CLASSES},
                event_counts=self.event_counts,
                expected_scorer_config_sha256=self.scorer_sha,
                expected_scorer_source_sha256=digest("scorer source"))
        with self.assertRaisesRegex(recovery.RecoveryError, "registry mismatch"):
            recovery.merge_completion_receipts(
                rows[:-1], tape_binding_sha256=self.tape_sha,
                query_counts=self.query_counts,
                query_positions_sha256_by_class={
                    category: digest(f"positions:{category}") for category in P11_CLASSES},
                event_counts=self.event_counts,
                expected_scorer_config_sha256=self.scorer_sha,
                expected_scorer_source_sha256=digest("scorer source"))

    def test_tape_source_scorer_and_counter_mutations_fail(self):
        mutate = self.valid_receipts()
        mutate[0]["query_positions_sha256"] = digest("mutated positions")
        with self.assertRaisesRegex(recovery.RecoveryError, "query positions"):
            recovery.merge_completion_receipts(
                mutate, tape_binding_sha256=self.tape_sha,
                query_counts=self.query_counts,
                query_positions_sha256_by_class={
                    category: digest(f"positions:{category}") for category in P11_CLASSES},
                event_counts=self.event_counts,
                expected_scorer_config_sha256=self.scorer_sha,
                expected_scorer_source_sha256=digest("scorer source"))
        mutate = self.valid_receipts()
        mutate[3]["scorer_source_sha256"] = digest("changed scorer")
        with self.assertRaisesRegex(recovery.RecoveryError, "Scorer source"):
            recovery.merge_completion_receipts(
                mutate, tape_binding_sha256=self.tape_sha,
                query_counts=self.query_counts,
                query_positions_sha256_by_class={
                    category: digest(f"positions:{category}") for category in P11_CLASSES},
                event_counts=self.event_counts,
                expected_scorer_config_sha256=self.scorer_sha,
                expected_scorer_source_sha256=digest("scorer source"))
        mutate = self.valid_receipts()
        mutate[0]["counter_delta"]["feature_encodings"] += 1
        with self.assertRaisesRegex(recovery.RecoveryError, "feature encodings"):
            recovery.merge_completion_receipts(
                mutate, tape_binding_sha256=self.tape_sha,
                query_counts=self.query_counts,
                query_positions_sha256_by_class={
                    category: digest(f"positions:{category}") for category in P11_CLASSES},
                event_counts=self.event_counts,
                expected_scorer_config_sha256=self.scorer_sha,
                expected_scorer_source_sha256=digest("scorer source"))

    def test_evidence_snapshot_integrity_and_one_byte_mutation_detection(self):
        rows = [evidence_row(None, 1, 0), evidence_row(7, 2, 1)]
        config_sha = rows[0]["source_exclusion_audit"]["scorer_config_sha256"]
        source_sha = rows[0]["source_exclusion_audit"]["scorer_source_sha256"]
        result = recovery.verify_evidence_rows(
            rows, category="audiojack", query_positions=[7],
            scorer_config_sha256=config_sha, scorer_source_sha256=source_sha)
        self.assertTrue(result["all_evidence_hashes_pass"])
        self.assertTrue(result["all_64_c64_vectors_and_q90_pass"])
        self.assertEqual(result["query_anchor_count"], 1)
        mutated = json.loads(json.dumps(rows))
        mutated[1]["protected_c64_scores"][57] += 1e-6
        with self.assertRaises(recovery.RecoveryError):
            recovery.verify_evidence_rows(
                mutated, category="audiojack", query_positions=[7],
                scorer_config_sha256=config_sha, scorer_source_sha256=source_sha)
        retained_source = json.loads(json.dumps(rows))
        anchor_digest = digest("query anchor")
        retained_source[1]["source_exclusion_audit"]["source_exclusion_proofs"][
            anchor_digest]["remaining_source_rows_by_layer"][1] = 1
        retained_source[1]["source_exclusion_proofs"] = json.loads(json.dumps(
            retained_source[1]["source_exclusion_audit"]["source_exclusion_proofs"]))
        with self.assertRaisesRegex(recovery.RecoveryError, "residual source rows"):
            recovery.verify_evidence_rows(
                retained_source, category="audiojack", query_positions=[7],
                scorer_config_sha256=config_sha, scorer_source_sha256=source_sha)
        with self.assertRaisesRegex(recovery.RecoveryError, "missing"):
            recovery.verify_evidence_rows(
                rows[:1], category="audiojack", query_positions=[7],
                scorer_config_sha256=config_sha, scorer_source_sha256=source_sha)

    def test_stored_evidence_adapter_binds_exact_replay_memory_and_query(self):
        category = "audiojack"
        position = 7
        source_key = "official-group-7"
        image_sha = digest("queried image bytes")
        source_group_sha = recovery.p12._group_sha(source_key)
        anchor_digest = sha256_json({
            "schema": "stage1a.p12_query_anchor.v1",
            "category": category, "position": position,
            "image_sha256": image_sha,
            "source_group_sha256": source_group_sha,
        })
        rows = [evidence_row(None, 1, 0),
                evidence_row(position, 2, 1, anchor_digest)]
        memory = MemoryIdentity(
            0, rows[0]["memory_sha256"], rows[0]["source_map_sha256"], 0, 0)
        c64 = C64Identity(
            0, memory.memory_sha256,
            float(rows[0]["source_exclusion_audit"]["c64_q90_higher_index_57"]),
            rows[0]["source_exclusion_audit"]["c64_transport_sha256"])
        adapter = recovery.StoredEvidenceReplayAdapter(
            category=category, rows=rows,
            c64_state_fn=lambda _memory: {
                "scores": rows[0]["protected_c64_scores"],
                "q90": c64.q90,
                "transport_sha256": c64.transport_sha256,
            })
        adapter.initialize(memory)
        adapter.initialize(memory)  # one wrapper entry per replay chunk
        event = ReplayEvent(
            position=position, queried=True, source_key=source_key,
            image_sha256=image_sha, static_score=0.1, dynamic_score=0.2,
            memory_before=memory, memory_after=memory,
            c64_before=c64, c64_after=c64,
            expected_writer_signature={}, static_memory_sha256=digest("M0"))
        adapter.after_event(
            event=event, queried=True, label=1, features=None, static_amap=None,
            writer_record={}, memory_before=memory, memory_after=memory)
        adapter.verify_complete([position])
        self.assertEqual(adapter.seen_positions, [position])

    def test_protected_native_writer_controller_and_scorer_sources_are_unchanged(self):
        root = Path(__file__).resolve().parents[2]
        protected = [
            "experiments/calibration/stage1a_p1_reconstruction.py",
            "experiments/calibration/stage1a_p11_production.py",
            "experiments/calibration/native.py",
            "experiments/calibration/memory_state.py",
            "experiments/calibration/profiles.py",
            "experiments/risk_safe/native_bridge.py",
            "experiments/risk_safe/shadow_memory.py",
            "experiments/v02_memory.py",
            "experiments/p0_protocol_audit.py",
            "test_phase3_AIF_simplified_v22.py",
            "vmb_visionad_new.py",
        ]
        diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *protected],
                              cwd=root, check=False)
        self.assertEqual(diff.returncode, 0, "protected source files changed")

    def test_memory_probe_emits_cpu_and_cuda_slots(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        class FakeTorch:
            cuda = FakeCuda()

        probe = recovery.MemoryProbe(FakeTorch())
        first = probe.sample("fixture")
        report = probe.report()["fixture"]
        self.assertGreater(first["process_rss_bytes"], 0)
        self.assertGreater(first["system_available_bytes"], 0)
        self.assertIsNone(first["cuda_allocated_bytes"])
        self.assertEqual(report["samples"], 1)
        self.assertEqual(report["first"], report["last"])


if __name__ == "__main__":
    unittest.main()
