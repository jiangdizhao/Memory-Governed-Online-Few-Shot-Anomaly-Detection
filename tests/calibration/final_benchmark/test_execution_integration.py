from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import networkx as nx
import numpy as np

from experiments.calibration.final_benchmark.execution_support import (
    AttemptStatus, ExecutionBoundaryError, append_jsonl_records_fsync,
    canonical_json, enforce_resource_limits, estimate_attempt_output_bytes,
    verify_attempt_receipt, write_json_exclusive,
)
from experiments.calibration.final_benchmark.foads_execution import (
    FOADSExecutionError, build_foads_identity_projection, build_foads_seal,
    foads_model_state_sha256, verify_foads_category_seal,
)
from experiments.calibration.final_benchmark.foads_kng import RegisteredP0Event
from experiments.calibration.final_benchmark.launch import (
    LaunchError, build_parser, current_git_head, execute_principal_one,
    execute_sequential_schedule, object_sha256, validate_config,
    validate_principal_authorization,
)


REPO = Path(__file__).resolve().parents[3]
TEMPLATE = REPO / "docs/calibration/final_benchmark_launch/FINAL_RUN_CONFIGURATION_TEMPLATE.json"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _event(position: int, group: int) -> RegisteredP0Event:
    digest = lambda ch: ch * 64
    return RegisteredP0Event(
        position=position, original_position=position,
        event_id_sha256=f"{position + 101:064x}", image_identity_sha256=(f"{position + 1:064x}"),
        image_content_sha256=digest("2"), source_group_sha256=f"{group + 1:064x}",
        camera_id=(position % 5) + 1, within_group_ordinal=(position % 5) + 1)


def _foads_fixture_files(root: Path, *, group_count: int = 2):
    events = tuple([_event(position, position // 5) for position in range(group_count * 5)])
    state_before = "c" * 64
    update_count = group_count // 2
    state_after = [hashlib.sha256(f"synthetic-post-update-{index}".encode()).hexdigest()
                   for index in range(update_count)]
    scores = [{
        "position": event.position,
        "event_id_sha256": event.event_id_sha256,
        "image_identity_sha256": event.image_identity_sha256,
        "source_group_sha256": event.source_group_sha256,
        "raw_image_score": float(event.position) / 10.0,
        "model_update_index": event.position // 10,
        "model_state_before_score_sha256": (state_before if event.position // 10 == 0
                                             else state_after[event.position // 10 - 1]),
    } for event in events]
    updates = []
    for index in range(update_count):
        start, stop = index * 10, (index + 1) * 10
        updates.append({
            "schema": "final-benchmark.foads-native-update.v1",
            "update_index": index + 1, "start_position": start, "end_position_exclusive": stop,
            "event_identity_sha256s": [event.image_identity_sha256 for event in events[start:stop]],
            "source_group_sha256s": [events[start].source_group_sha256,
                                     events[start + 5].source_group_sha256],
            "model_state_after_update_sha256": state_after[index],
        })
    registration = type("Registration", (), {"stream_events": events})()
    final_state = state_after[-1] if state_after else state_before
    summary = {"scored_images": len(events), "completed_batch10_updates": update_count,
               "final_unupdated_groups": group_count % 2,
               "final_unupdated_images": (group_count % 2) * 5,
               "paid_labels": 0, "test_labels_or_masks_read": 0}
    score_path = root / "foads_prediction_journal.jsonl"
    update_path = root / "foads_update_journal.jsonl"
    score_path.touch(exist_ok=False)
    update_path.touch(exist_ok=False)
    return registration, scores, updates, summary, score_path, update_path, final_state


class FinalLaunchIntegrationTests(unittest.TestCase):
    def test_model_state_digest_handles_multinode_numpy_list_and_hashes_centeroids(self):
        graph = nx.Graph()
        node_ids = np.array([11, 23, 47], dtype=np.int64)
        for node_id in node_ids:
            graph.add_node(int(node_id), threshold=float(node_id) / 10)
        model = SimpleNamespace(
            network=graph,
            node_list=node_ids.copy(),
            centeroid=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        )
        with self.assertRaises(ValueError):
            bool(model.node_list)  # Reproduces the ambiguous NumPy truth-value failure.
        original = foads_model_state_sha256(model)

        changed_centroid = copy.deepcopy(model)
        changed_centroid.centeroid[1, 0] += 0.25
        self.assertNotEqual(original, foads_model_state_sha256(changed_centroid))

        bad_dimensions = copy.deepcopy(model)
        bad_dimensions.centeroid = bad_dimensions.centeroid[:-1]
        with self.assertRaisesRegex(FOADSExecutionError, "centroid dimensions"):
            foads_model_state_sha256(bad_dimensions)

        reversed_graph = nx.Graph()
        for node_id in (23, 11, 47):
            reversed_graph.add_node(node_id, threshold=float(node_id) / 10)
        bad_order = SimpleNamespace(network=reversed_graph, node_list=node_ids.copy(),
                                    centeroid=model.centeroid.copy())
        with self.assertRaisesRegex(FOADSExecutionError, "ordering"):
            foads_model_state_sha256(bad_order)

    def test_frozen_config_and_exact_ninety_trajectory_schedule(self):
        config = _read(TEMPLATE)
        report = validate_config(config)
        self.assertEqual(report["trajectory_count"], 90)
        self.assertEqual(report["category_count"], 30)
        self.assertEqual(report["foads_category_count"], 30)
        pairs = {(row["category"], row["condition"]) for row in config["trajectories"]}
        self.assertEqual(len(pairs), 90)
        self.assertTrue(all((row["condition"] == "P0") ==
                            (row["foads_p0_registration_projection_sha256"] is not None)
                            for row in config["trajectories"]))
        parser = build_parser()
        subcommands = next(action for action in parser._actions if action.dest == "command").choices
        self.assertTrue({"validate", "list", "run-one", "run-schedule", "foads-run-one"}.issubset(subcommands))

    def test_source_commit_mismatch_rejects_bound_configuration(self):
        config = _read(TEMPLATE)
        config["source_identity"]["final_source_commit"] = "0" * 40
        config["source_identity"]["campaign_run_id"] = "test-run"
        with self.assertRaises(LaunchError):
            validate_config(config, current_head=current_git_head(REPO), require_source_binding=True)

    def test_principal_authorization_must_bind_exact_source_and_scope(self):
        identity = {"source_commit": "1" * 40, "campaign_run_id": "run-1",
                    "execution_plan_sha256": "2" * 64, "identity_manifest_sha256": "3" * 64,
                    "role_manifest_sha256": "4" * 64, "dataset_release_sha256": "5" * 64,
                    "stream_manifest_sha256": "6" * 64,
                    "category_bundle_file_sha256": "7" * 64,
                    "private_locator_manifest_file_sha256": "8" * 64,
                    "annotation_sha256": "9" * 64,
                    "category": "audiojack", "condition": "P0"}
        auth = {"schema": "final-benchmark.execution-authorization.v1", "status": "AUTHORIZED",
                "scope": "stage_a_category", "source_commit": "1" * 40,
                "campaign_run_id": "run-1", "execution_plan_sha256": "2" * 64,
                "identity_manifest_sha256": "3" * 64, "role_manifest_sha256": "4" * 64,
                "stream_manifest_sha256": "6" * 64,
                "dataset_release_sha256": "5" * 64,
                "category_bundle_file_sha256": "7" * 64,
                "private_locator_manifest_file_sha256": "8" * 64,
                "annotation_sha256": "9" * 64, "dataset_id": "realiad",
                "category": "audiojack", "condition_id": "P0", "label_access": "QUERY_ONLY",
                "stage_b_authorized": False, "expected_event_count": 3815}
        validate_principal_authorization(auth, identities=identity)
        auth["source_commit"] = "9" * 40
        with self.assertRaises(LaunchError):
            validate_principal_authorization(auth, identities=identity)

    def test_preauthorization_failure_creates_no_attempt_and_never_loads_model(self):
        private = REPO / "outputs/calibration/final_benchmark_gate1/attempt_05/private"
        bundle = {"schema": "final-benchmark.private-category-bundle.v1",
                  "category_bundle_path": str(private / "realiad/audiojack/P0.json"),
                  "identity_manifest_path": str(private / "realiad_identity_manifest.private.json"),
                  "private_locator_manifest_path": str(private / "realiad_private_locators.json")}
        config = _read(TEMPLATE)
        config["source_identity"]["final_source_commit"] = current_git_head(REPO)
        config["source_identity"]["campaign_run_id"] = "synthetic-preauth-test"
        unsigned = dict(config)
        unsigned.pop("schedule_sha256")
        config["schedule_sha256"] = object_sha256(unsigned)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "plan.json"
            bundle_path = root / "bundle.json"
            auth_path = root / "authorization.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            auth_path.write_text("{}", encoding="utf-8")
            resource_called = []
            with self.assertRaises(LaunchError):
                execute_principal_one(config_path=config_path, bundle_path=bundle_path,
                    authorization_path=auth_path, category="audiojack", condition="P0",
                    attempt_id="attempt_test", repository=REPO,
                    encoder=object(), resource_probe=lambda *_: resource_called.append(True))
            self.assertEqual(resource_called, [])
            expected_output = REPO / "outputs/calibration/final_benchmark/synthetic-preauth-test"
            self.assertFalse(expected_output.exists())

    def test_authorized_early_stage_a_startup_failure_leaves_incomplete_receipt(self):
        private = REPO / "outputs/calibration/final_benchmark_gate1/attempt_05/private"
        bundle = {"schema": "final-benchmark.private-category-bundle.v1",
                  "category_bundle_path": str(private / "realiad/audiojack/P0.json"),
                  "identity_manifest_path": str(private / "realiad_identity_manifest.private.json"),
                  "private_locator_manifest_path": str(private / "realiad_private_locators.json")}
        config = _read(TEMPLATE)
        config["source_identity"]["final_source_commit"] = current_git_head(REPO)
        config["source_identity"]["campaign_run_id"] = "synthetic-incomplete-test"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config["output_root"] = str(root / "out" / "<campaign_run_id>")
            unsigned = dict(config); unsigned.pop("schedule_sha256")
            config["schedule_sha256"] = object_sha256(unsigned)
            config_path = root / "plan.json"
            bundle_path = root / "bundle.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            from experiments.calibration.final_benchmark.launch import load_principal_bundle
            payload, identities = load_principal_bundle(bundle_path, config=config,
                category="audiojack", condition="P0")
            identities["execution_plan_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
            trajectory = next(row for row in config["trajectories"]
                              if row["category"] == "audiojack" and row["condition"] == "P0")
            auth = {"schema": "final-benchmark.execution-authorization.v1", "status": "AUTHORIZED",
                    "scope": "stage_a_category", "source_commit": identities["source_commit"],
                    "campaign_run_id": identities["campaign_run_id"],
                    "execution_plan_sha256": identities["execution_plan_sha256"],
                    "identity_manifest_sha256": identities["identity_manifest_sha256"],
                    "role_manifest_sha256": identities["role_manifest_sha256"],
                    "stream_manifest_sha256": identities["stream_manifest_sha256"],
                    "dataset_release_sha256": identities["dataset_release_sha256"],
                    "category_bundle_file_sha256": identities["category_bundle_file_sha256"],
                    "private_locator_manifest_file_sha256": identities["private_locator_manifest_file_sha256"],
                    "annotation_sha256": identities["annotation_sha256"],
                    "dataset_id": "realiad", "category": "audiojack", "condition_id": "P0",
                    "label_access": "QUERY_ONLY", "stage_b_authorized": False,
                    "expected_event_count": trajectory["test_event_count"]}
            auth_path = root / "authorization.json"
            auth_path.write_text(json.dumps(auth), encoding="utf-8")

            class StartupFailure:
                def prepare_commissioning(self, **kwargs):
                    raise RuntimeError("synthetic model startup failure")

            resources = {"host_available_bytes": 100 * 1024**3,
                         "process_rss_bytes": 1, "disk_free_bytes": 500 * 1024**3,
                         "gpu": {"name": "NVIDIA GeForce RTX 3090 Ti",
                                 "total_bytes": 24 * 1024**3, "free_bytes": 23 * 1024**3}}
            with self.assertRaises(RuntimeError):
                execute_principal_one(config_path=config_path, bundle_path=bundle_path,
                    authorization_path=auth_path, category="audiojack", condition="P0",
                    attempt_id="attempt_01", repository=REPO, encoder=StartupFailure(),
                    resource_probe=lambda *_: resources)
            attempt_root = root / "out/synthetic-incomplete-test/stage_a/audiojack/P0/attempt_01"
            receipt = _read(attempt_root / "attempt_status.json")
            self.assertEqual(receipt["status"], "INCOMPLETE")
            self.assertEqual(receipt["failing_phase"], "AUTHORIZED_START")
            self.assertEqual(receipt["completed_event_count"], 0)
            self.assertFalse((attempt_root / "stage_a_category_seal.json").exists())

    def test_attempt_failure_is_durable_and_existing_attempt_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "attempt_01"
            attempt = AttemptStatus(root, identity={"source_commit": "a" * 40,
                "execution_plan_sha256": "b" * 64}, expected_events=25,
                execution_kind="principal_stage_a")
            attempt.update(phase="MODEL_INITIALIZATION", completed_events=0)
            attempt.incomplete(phase="MODEL_INITIALIZATION", completed_events=0,
                               exception=RuntimeError("synthetic startup failure"))
            receipt = _read(root / "attempt_status.json")
            self.assertEqual(receipt["status"], "INCOMPLETE")
            self.assertEqual(receipt["failing_phase"], "MODEL_INITIALIZATION")
            self.assertEqual(receipt["completed_event_count"], 0)
            self.assertEqual(receipt["error_type"], "RuntimeError")
            with self.assertRaises(ExecutionBoundaryError):
                AttemptStatus(root, identity={}, expected_events=25,
                              execution_kind="principal_stage_a")

    def test_stage_b_receipt_gate_rejects_incomplete_or_mismatched_inputs(self):
        identity = {"source_commit": "a" * 40, "campaign_run_id": "test",
                    "execution_plan_sha256": "b" * 64}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "stage-a"
            attempt = AttemptStatus(root, identity=identity, expected_events=5,
                                    execution_kind="principal_stage_a")
            attempt.incomplete(phase="GROUP", completed_events=0,
                               exception=RuntimeError("partial"))
            with self.assertRaises(ExecutionBoundaryError):
                verify_attempt_receipt(root / "attempt_status.json",
                    expected_identity=identity, expected_kind="principal_stage_a")
            foads_root = Path(td) / "incomplete-foads"
            foads_attempt = AttemptStatus(foads_root, identity=identity, expected_events=10,
                                          execution_kind="foads_kng_p0")
            foads_attempt.incomplete(phase="INITIALIZATION", completed_events=0,
                                     exception=RuntimeError("synthetic FOADS failure"))
            with self.assertRaises(ExecutionBoundaryError):
                verify_attempt_receipt(foads_root / "attempt_status.json",
                    expected_identity=identity, expected_kind="foads_kng_p0")
            with self.assertRaises(FOADSExecutionError):
                verify_foads_category_seal(foads_root, expected_identity=identity)
            seal = {"status": "SEALED_COMPLETE", "expected_event_count": 5,
                    "durable_seal_sha256": "c" * 64}
            # A second directory models a real complete attempt receipt without evaluation labels.
            root2 = Path(td) / "complete-stage-a"
            complete = AttemptStatus(root2, identity=identity, expected_events=5,
                                     execution_kind="principal_stage_a")
            (root2 / "stage_a_category_seal.json").write_bytes(canonical_json(seal) + b"\n")
            complete.sealed_complete(phase="CATEGORY_SEALED",
                seal_path=root2 / "stage_a_category_seal.json", completed_events=5)
            verify_attempt_receipt(root2 / "attempt_status.json",
                expected_identity=identity, expected_kind="principal_stage_a")
            with self.assertRaises(ExecutionBoundaryError):
                verify_attempt_receipt(root2 / "attempt_status.json",
                    expected_identity={**identity, "campaign_run_id": "different"},
                    expected_kind="principal_stage_a")

    def test_measured_resource_guard_passes_and_fails_closed(self):
        snapshot = {"host_available_bytes": 40 * 1024**3,
                    "disk_free_bytes": 40 * 1024**3,
                    "gpu": {"name": "NVIDIA GeForce RTX 3090 Ti",
                            "total_bytes": 24 * 1024**3, "free_bytes": 23 * 1024**3}}
        estimate = estimate_attempt_output_bytes("principal", 3815)
        enforce_resource_limits("principal", snapshot, required_disk_bytes=estimate)
        with self.assertRaises(ExecutionBoundaryError):
            enforce_resource_limits("principal", {**snapshot, "disk_free_bytes": 2 * 1024**3},
                                    required_disk_bytes=estimate)

    def test_foads_missing_scores_or_native_update_receipt_cannot_seal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registration, scores, updates, summary, score_path, update_path, state = _foads_fixture_files(root)
            identity = {"source_commit": "a" * 40, "category": "audiojack", "condition": "P0"}
            append_jsonl_records_fsync(score_path, scores[:-1])
            with self.assertRaises(FOADSExecutionError):
                build_foads_seal(registration=registration, score_journal_path=score_path,
                    update_journal_path=update_path, identity=identity, runner_summary=summary,
                    final_state_sha256=state)
            # Restore a fresh journal with all ten scores but no Batch-10 receipt.
            score_path.unlink(); update_path.unlink()
            score_path.touch(); update_path.touch()
            append_jsonl_records_fsync(score_path, scores)
            with self.assertRaises(FOADSExecutionError):
                build_foads_seal(registration=registration, score_journal_path=score_path,
                    update_journal_path=update_path, identity=identity, runner_summary=summary,
                    final_state_sha256=state)
            append_jsonl_records_fsync(update_path, updates)
            seal = build_foads_seal(registration=registration, score_journal_path=score_path,
                update_journal_path=update_path, identity=identity, runner_summary=summary,
                final_state_sha256=state)
            self.assertEqual(seal["completed_batch10_updates"], 1)
            self.assertEqual(seal["final_unupdated_groups"], 0)

    def test_foads_state_chain_seals_multiple_batches_and_final_unupdated_group(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registration, scores, updates, summary, score_path, update_path, final_state = \
                _foads_fixture_files(root, group_count=5)
            append_jsonl_records_fsync(score_path, scores)
            append_jsonl_records_fsync(update_path, updates)
            seal = build_foads_seal(registration=registration, score_journal_path=score_path,
                update_journal_path=update_path, identity={"category": "audiojack", "condition": "P0"},
                runner_summary=summary, final_state_sha256=final_state)
            self.assertEqual(seal["completed_batch10_updates"], 2)
            self.assertEqual(seal["final_unupdated_groups"], 1)
            self.assertEqual(seal["final_unupdated_group_sha256"],
                             registration.stream_events[-5].source_group_sha256)
            self.assertEqual(seal["final_model_state_sha256"], updates[-1]["model_state_after_update_sha256"])

    def test_foads_state_chain_rejects_missing_reordered_and_inconsistent_state_records(self):
        mutations = ("missing_score_state", "missing_update_state", "within_batch_change",
                     "next_batch_discontinuity", "reordered_updates", "wrong_final_state")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                registration, scores, updates, summary, score_path, update_path, final_state = \
                    _foads_fixture_files(root, group_count=5)
                if mutation == "missing_score_state":
                    scores[0].pop("model_state_before_score_sha256")
                elif mutation == "missing_update_state":
                    updates[0].pop("model_state_after_update_sha256")
                elif mutation == "within_batch_change":
                    scores[4]["model_state_before_score_sha256"] = "e" * 64
                elif mutation == "next_batch_discontinuity":
                    scores[10]["model_state_before_score_sha256"] = "f" * 64
                elif mutation == "reordered_updates":
                    updates.reverse()
                append_jsonl_records_fsync(score_path, scores)
                append_jsonl_records_fsync(update_path, updates)
                with self.assertRaises(FOADSExecutionError):
                    build_foads_seal(registration=registration, score_journal_path=score_path,
                        update_journal_path=update_path, identity={"category": "audiojack", "condition": "P0"},
                        runner_summary=summary,
                        final_state_sha256=("9" * 64 if mutation == "wrong_final_state" else final_state))

    def test_foads_sealed_receipt_binds_both_journals_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "foads"
            attempt = AttemptStatus(root, identity={"source_commit": "a" * 40,
                "category": "audiojack", "condition": "P0"}, expected_events=10,
                execution_kind="foads_kng_p0")
            registration, scores, updates, summary, score_path, update_path, state = _foads_fixture_files(root)
            identity = {"source_commit": "a" * 40, "category": "audiojack", "condition": "P0"}
            append_jsonl_records_fsync(score_path, scores)
            append_jsonl_records_fsync(update_path, updates)
            seal = build_foads_seal(registration=registration, score_journal_path=score_path,
                update_journal_path=update_path, identity=identity, runner_summary=summary,
                final_state_sha256=state)
            seal_path = root / "foads_category_seal.json"
            write_json_exclusive(seal_path, seal)
            attempt.sealed_complete(phase="CATEGORY_SEALED", seal_path=seal_path,
                                    completed_events=10)
            verified = verify_foads_category_seal(root, expected_identity=identity)
            self.assertEqual(verified["status"], "SEALED_COMPLETE")
            with self.assertRaises(FOADSExecutionError):
                verify_foads_category_seal(root,
                    expected_identity={**identity, "category": "different"})

    def test_complete_foads_projection_hashes_match_all_registered_p0_categories(self):
        manifest_path = REPO / "outputs/calibration/final_benchmark_gate1/attempt_05/private/realiad_identity_manifest.private.json"
        root = REPO / "outputs/calibration/final_benchmark_gate1/attempt_05/private/realiad"
        identity = _read(manifest_path)
        config = _read(TEMPLATE)
        for category in config["foads_kng"]["categories"]:
            bundle = _read(root / category / "P0.json")
            projection = build_foads_identity_projection(
                identity, bundle["role_manifest"], bundle["stream_manifest"], category)
            trajectory = next(row for row in config["trajectories"]
                              if row["category"] == category and row["condition"] == "P0")
            self.assertEqual(object_sha256(projection),
                             trajectory["foads_p0_registration_projection_sha256"], category)

    def test_sequential_schedule_stops_after_first_failure(self):
        config = _read(TEMPLATE)
        config["source_identity"]["final_source_commit"] = current_git_head(REPO)
        config["source_identity"]["campaign_run_id"] = "synthetic-sequential-test"
        unsigned = dict(config); unsigned.pop("schedule_sha256")
        config["schedule_sha256"] = object_sha256(unsigned)
        launches = [{"category": row["category"], "condition": row["condition"],
                     "bundle_path": "private-bundle.json", "authorization_path": "private-auth.json",
                     "attempt_id": f"attempt_{index:02d}"}
                    for index, row in enumerate(config["trajectories"], start=1)]
        schedule = {"schema": "final-benchmark.sequential-launch-schedule.v1",
                    "failure_policy": "STOP_ON_FIRST_FAILURE", "launches": launches}
        schedule["schedule_sha256"] = object_sha256(schedule)
        calls = []
        def fail_first(**kwargs):
            calls.append((kwargs["category"], kwargs["condition"]))
            raise RuntimeError("synthetic launch failure")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            schedule_path = root / "launches.json"
            schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
            config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
            schedule_file_sha = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
            auth = {"schema": "final-benchmark.sequential-launch-authorization.v1",
                    "status": "AUTHORIZED", "scope": "stage_a_sequential_campaign",
                    "source_commit": config["source_identity"]["final_source_commit"],
                    "campaign_run_id": config["source_identity"]["campaign_run_id"],
                    "execution_plan_sha256": config_sha,
                    "launch_schedule_sha256": schedule_file_sha,
                    "failure_policy": "STOP_ON_FIRST_FAILURE"}
            auth_path = root / "schedule-auth.json"
            auth_path.write_text(json.dumps(auth), encoding="utf-8")
            with self.assertRaises(LaunchError):
                execute_sequential_schedule(config_path=config_path,
                    launch_schedule_path=schedule_path, authorization_path=auth_path,
                    repository=REPO, executor=fail_first)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()

