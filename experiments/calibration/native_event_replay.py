"""Bounded native multi-event causal parity replay (Stage 0 only).

This diagnostic uses only one development-class training-normal stream. Its
single Q-A-TS query is selected from public stream metadata before labels are
disclosed. The evaluator also receives a separately marked synthetic binary
label fixture solely to verify shared-score AUROC invariance; those labels are
never passed to controllers or the native writer.
"""
from __future__ import annotations

import hashlib
import time
import copy
from pathlib import Path
from typing import Any

import torch

from .controllers import FrozenTheta, theta0_from_c64
from .engine import PrequentialEngine
from .method_registry import METHOD_CARDS, build_controllers
from .metrics import PrivateOutcomeLedger, _ranking_metrics
from .native import (NativeScoreService, SharedNativeGreedyWriter,
                     _fixed_bank_digest, isolate_native_bank_k256)
from .profiles import DEFAULT_PROFILE, canonical_json
from .protocol import StreamEvent, TapeRecord
from .query_tape import build_qa_ts, verify_tape


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _opaque_set_digest(domain: str, values: set[str]) -> str:
    return _digest(domain.encode("utf-8") + b"\0" +
                   "\n".join(sorted(values)).encode("utf-8"))


def run_native_event_replay(dataset_root: str | Path,
                            json_root: str | Path) -> dict[str, Any]:
    """Execute a small actual-v22 trajectory, or return an honest SKIP/BLOCKED."""
    root, metadata = Path(dataset_root), Path(json_root)
    if not torch.cuda.is_available():
        return {"schema": "calibration.stage0.native_event_replay.v1",
                "status": "SKIP", "reason": "CUDA is unavailable"}
    hub = Path(torch.hub.get_dir())
    checkpoint = hub / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
    hub_source = hub / "facebookresearch_dinov2_main"
    if not root.is_dir() or not metadata.is_dir():
        return {"schema": "calibration.stage0.native_event_replay.v1",
                "status": "SKIP", "reason": "Development train metadata roots are absent"}
    if not checkpoint.is_file() or not hub_source.is_dir():
        return {"schema": "calibration.stage0.native_event_replay.v1",
                "status": "SKIP",
                "reason": "Cached DINOv2 source/checkpoint absent; network download disabled"}

    try:
        from experiments.p0_protocol_audit import (
            Config, build_vmb, encode_query, encode_reference_features,
            require_sources,
        )
        from experiments.risk_safe.preflight import group_by_sample, load_class_records
        from .anchor_ledger import AnchorLedger
        from .real_iad_preflight import _select_support_and_c64

        started = time.perf_counter()
        train, _unused_development_test_metadata = load_class_records(
            root, metadata, "audiojack")
        supports, c64, support_keys, c64_keys = _select_support_and_c64(train)
        excluded = set(support_keys) | set(c64_keys)
        groups = group_by_sample(train)
        eligible = [key for key, views in groups.items()
                    if key not in excluded and all(view.is_good for view in views)]
        ranked = sorted(eligible, key=lambda key: (
            hashlib.sha256(("stage0-native-event-replay-v1\0" + key).encode()).hexdigest(),
            key))
        if len(ranked) < 34:
            return {"schema": "calibration.stage0.native_event_replay.v1",
                    "status": "BLOCKED",
                    "reason": f"Need 34 disjoint verified train-normal groups; found {len(ranked)}"}
        stream_rows = [groups[key][0] for key in ranked[:34]]
        stream_rows.sort(key=lambda row: row.capture_order())
        records = [TapeRecord(i, row.sample_key, row.camera_id, row.rel_path,
                              timestamp=row.timestamp)
                   for i, row in enumerate(stream_rows)]
        tape = build_qa_ts(records, DEFAULT_PROFILE.query)
        verify_tape(tape, records, DEFAULT_PROFILE.query)
        if len(tape.events) != 1:
            raise RuntimeError("34-event diagnostic did not produce its registered one-query Q-A-TS tape")

        scorer = DEFAULT_PROFILE.scorer
        cfg = Config(
            device="cuda", augmentation_seed=scorer.augmentation_seed,
            bank_seed=scorer.bank_seed, fixed_capacity=scorer.fixed_capacity,
            fixed_per_image=scorer.fixed_per_image,
            augmentations=scorer.augmentations,
            include_identity=scorer.include_identity,
            multiplier=scorer.multiplier, feature_fuse=scorer.feature_fuse,
            topk_patches=scorer.topk_patches,
            image_aggregation=scorer.image_aggregation,
            image_topk=scorer.image_topk,
            image_quantile=scorer.image_quantile, knn=scorer.knn,
        )
        impl = require_sources()
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        impl.set_seed(0)
        backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
        support_features = encode_reference_features(
            impl, backbone, [row.path for row in supports], cfg)
        native_bank = build_vmb(impl, support_features,
                                [row.path for row in supports], cfg)
        support_sha = _fixed_bank_digest(native_bank)
        native_bank, isolated_support_sha = isolate_native_bank_k256(native_bank)
        if support_sha != isolated_support_sha:
            raise RuntimeError("K=256 native bank isolation altered frozen four-shot support")
        pristine_native_template = copy.deepcopy(native_bank)

        c64_bundles = [encode_query(impl, backbone, row.path, cfg) for row in c64]
        event_bundles = [encode_query(impl, backbone, row.path, cfg)
                         for row in stream_rows]
        if len(c64_bundles) != 64 or len(event_bundles) != 34:
            raise RuntimeError("Native event replay feature extraction count mismatch")

        protected = [row.sample_key for row in c64]
        writer = SharedNativeGreedyWriter(
            impl, native_bank, theta_ref=0.0,
            protected_source_keys=protected, sigma_ref=0.02)
        service = NativeScoreService(impl, native_bank, cfg, scorer,
                                     isolated_support_sha)
        ledger = AnchorLedger(
            [(row.sample_key, row.camera_id, bundle)
             for row, bundle in zip(c64, c64_bundles)],
            deployed_support_keys=support_keys)
        initial_evidence = ledger.refresh(
            memory=writer.bank, source_rows=writer.sources,
            scorer=lambda bundle, memory: service.score(bundle, memory))
        c64_scores = [anchor.score for anchor in initial_evidence.normal_anchors]
        if len(c64_scores) != 64:
            raise RuntimeError("Initial native C64 score count is not 64")
        theta0 = theta0_from_c64(c64_scores)
        writer.theta_ref = theta0
        events = [StreamEvent(record, bundle, bundle)
                  for record, bundle in zip(records, event_bundles)]
        private_training_labels = PrivateOutcomeLedger([0] * len(events))
        initial_ledger_state = ledger.state_dict()
        score_call_counts: list[dict[str, int]] = []

        def make_engine(run_writer: SharedNativeGreedyWriter,
                        run_ledger: AnchorLedger,
                        run_template: Any,
                        revisions: list[dict[str, Any]]):
            run_service = NativeScoreService(
                impl, run_template, cfg, scorer, isolated_support_sha)
            controllers = build_controllers(theta0, c64_scores)
            controllers.append(FrozenTheta(
                theta0 + max(1.0, abs(theta0)),
                name="B0_DIAGNOSTIC_HIGH_THRESHOLD"))
            for controller in controllers:
                controller.on_memory_change(initial_evidence)
            calls = {"native_score_calls": 0}

            def score(bundle: Any, memory: Any):
                calls["native_score_calls"] += 1
                return run_service.score(bundle, memory)

            def score_fn(event: StreamEvent):
                return score(event.payload, run_writer.bank)

            def write_fn(event: StreamEvent, score_result: Any, feedback: Any):
                return run_writer.apply(
                    queried=feedback.queried, label=feedback.label,
                    source_key=event.record.official_sample_key,
                    score=score_result.score, static_amap=score_result.score_map,
                    full_features=event.payload[1], step=event.record.position)

            def evidence_fn(event: StreamEvent | None, feedback: Any):
                if event is not None and feedback is not None:
                    run_ledger.add_queried(
                        source_key=event.record.official_sample_key,
                        camera_id=event.record.camera_id,
                        label=int(feedback.label), features=event.anchor_features,
                        t=event.record.position, memory_version=run_writer.bank.version)
                snapshot_now = run_ledger.refresh(
                    memory=run_writer.bank, source_rows=run_writer.sources,
                    scorer=lambda bundle, memory: score(bundle, memory))
                revisions.append({
                    "memory_version": snapshot_now.memory_version,
                    "evidence_revision": snapshot_now.evidence_revision,
                    "normal_anchor_count": len(snapshot_now.normal_anchors),
                    "protected_c64_count": sum(
                        a.protected for a in snapshot_now.normal_anchors),
                    "queried_normal_count": sum(
                        not a.protected for a in snapshot_now.normal_anchors),
                    "queried_normal_lso_rows": sum(
                        a.source_excluded_rows for a in snapshot_now.normal_anchors
                        if not a.protected),
                })
                return snapshot_now

            engine = PrequentialEngine(
                tape=tape, events=events, outcomes=private_training_labels,
                controllers=controllers, score_fn=score_fn,
                memory_state_fn=lambda: (run_writer.bank.version,
                                         run_writer.bank.content_sha256()),
                write_fn=write_fn, evidence_fn=evidence_fn,
                checkpoint_components={"writer": run_writer,
                                       "anchors": run_ledger},
                provenance_id="stage0-native-event-replay-audiojack-train-v1")
            return engine, calls

        base_revisions: list[dict[str, Any]] = []
        engine, base_calls = make_engine(writer, ledger, native_bank, base_revisions)
        uninterrupted = engine.run()
        if uninterrupted["status"] != "COMPLETE":
            raise RuntimeError("Uninterrupted native event replay did not complete")
        baseline_memory_sha = writer.bank.sha256()
        baseline_source_sha = writer.source_map_sha256()
        del engine, writer, ledger, service
        native_bank = copy.deepcopy(pristine_native_template)
        if _fixed_bank_digest(native_bank) != isolated_support_sha:
            raise RuntimeError("Fresh native replay bank changed frozen support")

        # Exercise a genuine native-bank checkpoint restore at an event boundary.
        resumed_ledger = AnchorLedger(
            [(row.sample_key, row.camera_id, bundle)
             for row, bundle in zip(c64, c64_bundles)],
            deployed_support_keys=support_keys)
        resumed_ledger.load_state_dict(copy.deepcopy(initial_ledger_state))
        resumed_writer = SharedNativeGreedyWriter(
            impl, native_bank, theta_ref=theta0,
            protected_source_keys=protected, sigma_ref=0.02)
        partial_engine, partial_calls = make_engine(
            resumed_writer, resumed_ledger, native_bank, [])
        partial = partial_engine.run(max_events=17)
        if partial["status"] != "INCOMPLETE":
            raise RuntimeError("Native event replay checkpoint fixture did not stop at event 17")
        checkpoint = partial["checkpoint"]
        del partial_engine, resumed_ledger, resumed_writer, partial
        native_bank = copy.deepcopy(pristine_native_template)
        if _fixed_bank_digest(native_bank) != isolated_support_sha:
            raise RuntimeError("Fresh resume target changed frozen support")
        restored_ledger = AnchorLedger(
            [(row.sample_key, row.camera_id, bundle)
             for row, bundle in zip(c64, c64_bundles)],
            deployed_support_keys=support_keys)
        restored_writer = SharedNativeGreedyWriter(
            impl, native_bank, theta_ref=theta0,
            protected_source_keys=protected, sigma_ref=0.02)
        restored_engine, restored_calls = make_engine(
            restored_writer, restored_ledger, native_bank, [])
        restored_engine.load_state_dict(checkpoint)
        resumed = restored_engine.run()
        if resumed["status"] != "COMPLETE":
            raise RuntimeError("Restored native event replay did not complete")

        if (resumed["scores"] != uninterrupted["scores"] or
                resumed["predictions"] != uninterrupted["predictions"] or
                resumed["prediction_availability"] !=
                uninterrupted["prediction_availability"] or
                resumed["trace"] != uninterrupted["trace"] or
                resumed["controller_states"] != uninterrupted["controller_states"] or
                restored_writer.bank.sha256() != baseline_memory_sha or
                restored_writer.source_map_sha256() != baseline_source_sha):
            raise RuntimeError("Native interrupted/resumed trace differs from uninterrupted replay")
        score_call_counts = [base_calls, partial_calls, restored_calls]
        result, writer, evidence_revisions = uninterrupted, restored_writer, base_revisions
        if result["status"] != "COMPLETE":
            raise RuntimeError("Native event replay did not complete its fixed diagnostic stream")
        if len(result["scores"]) != 34:
            raise RuntimeError("Native score trajectory is not exactly 34 events")
        versions = [row["memory_version_before"] for row in result["trace"]]
        for row in result["trace"]:
            if row["score_state_digest"] == "":
                raise RuntimeError("Native replay omitted score-state identity")
        write_rows = [row["write"] for row in result["trace"]]
        successful_writes = [row for row in write_rows if row.get("changed")]
        if len(successful_writes) != 1:
            raise RuntimeError(
                f"Expected one fixed verified-normal memory update, got {len(successful_writes)}")
        final_evidence = evidence_revisions[-1] if evidence_revisions else {}
        if (writer.bank.version != 1 or final_evidence.get("memory_version") != 1 or
                final_evidence.get("protected_c64_count") != 64 or
                final_evidence.get("queried_normal_count") != 1 or
                final_evidence.get("queried_normal_lso_rows", 0) <= 0):
            raise RuntimeError("Native anchor refresh/version/LSO invariant failed")
        if _fixed_bank_digest(native_bank) != isolated_support_sha:
            raise RuntimeError("Native event replay mutated frozen support bytes")

        values = result["scores"]
        score_vector_sha = _digest(canonical_json([float(x) for x in values]).encode())
        synthetic_labels = [int(i % 2) for i in range(len(values))]
        synthetic_auc, synthetic_auprc, ranking_status = _ranking_metrics(
            synthetic_labels, values)
        executable_names = [controller.name for controller in build_controllers(
            theta0, c64_scores)]
        blocked_method_names = [key for key, card in METHOD_CARDS.items()
                                if card["status"] == "BLOCKED_FIDELITY"]
        method_auc = {name: {"auroc": synthetic_auc, "auprc": synthetic_auprc,
                             "score_vector_sha256": score_vector_sha,
                             "label_fixture": "synthetic_binary_diagnostic_only"}
                      for name in executable_names}
        threshold_pairs = [
            (row["theta_pre"].get("B0"),
             row["theta_pre"].get("B0_DIAGNOSTIC_HIGH_THRESHOLD"))
            for row in result["trace"]]
        threshold_divergent_events = sum(
            left is not None and right is not None and left != right
            for left, right in threshold_pairs)
        if threshold_divergent_events != len(events):
            raise RuntimeError("Counterfactual threshold twin did not diverge on every event")
        query_event = next(iter(tape.events))
        query_record = records[query_event.position]
        operation_counts = {"append": 0, "replace": 0, "skip": 0}
        for row in successful_writes:
            operation_counts["append"] += int(row.get("appended", 0))
            operation_counts["replace"] += int(row.get("replaced", 0))
            operation_counts["skip"] += int(row.get("skipped", 0))
        native_score_calls = sum(
            row["native_score_calls"] for row in score_call_counts)
        return {
            "schema": "calibration.stage0.native_event_replay.v1",
            "status": "PASS",
            "scope": "bounded_audiojack_train_normal_native_v22_non_scientific_diagnostic",
            "device": "cuda", "events": len(events), "query_count": len(tape.events),
            "query_tape": {"policy": tape.policy, "version": DEFAULT_PROFILE.query.version,
                           "sha256": tape.digest,
                           "selected_group_sha256": _opaque_set_digest(
                               "native-replay-query", {
                                   query_record.official_sample_key}),
                           "selection_label_blind": True,
                           "one_view_per_official_group": True},
            "support_regime": "four_deployed_supports_plus_64_additional_C64_normals",
            "support_sha256": isolated_support_sha,
            "c64_group_sha256": _opaque_set_digest("native-replay-c64", set(c64_keys)),
            "initial_c64_current_state_count": len(initial_evidence.normal_anchors),
            "initial_theta_policy": "D2_higher_q90_of_64_additional_C64_native_scores_at_M0",
            "method_names": executable_names,
            "blocked_method_names": blocked_method_names,
            "all_controller_matrix_status": (
                "PASS_BOUNDED_NATIVE_DIAGNOSTIC" if not blocked_method_names else
                "BLOCKED_UNRESOLVED_METHOD_ROWS"),
            "score_vector_sha256": score_vector_sha,
            "score_values_emitted": False,
            "score_state_versions": sorted(set(versions)),
            "threshold_counterfactual": {
                "controller": "B0_DIAGNOSTIC_HIGH_THRESHOLD",
                "threshold_divergent_events": threshold_divergent_events,
                "query_identity_unchanged": True,
                "shared_score_vector_sha256": score_vector_sha,
                "shared_memory_hashes_unchanged_by_threshold": True,
                "basis": "one engine, one score call and one controller-independent writer per event"},
            "fixed_query_normal_update": {
                "memory_version_before_after": [0, writer.bank.version],
                "memory_content_version_before_after": [
                    0, writer.bank.memory_content_version],
                "legacy_greedy_version_after": writer.bank.legacy_greedy_version,
                "content_sha256_after": writer.bank.content_sha256(),
                "source_map_sha256": writer.source_map_sha256(),
                "source_map_revision": writer.source_revision,
                "version_axes": {
                    "memory_content_version": "calibration score/evidence version; advances iff prototype content changes",
                    "legacy_greedy_version": "unchanged historical GreedyBank semantics",
                    "source_map_revision": "source attribution changes only",
                },
                "operation_counts": operation_counts,
                "verified_normal_only": True},
            "execution_cost": {"native_score_calls": native_score_calls,
                               "feature_images_encoded": 4 + 64 + 34,
                               "checkpoint_replay_cursor": 17},
            "anchor_refreshes": evidence_revisions,
            "final_anchor_lso_rows": final_evidence.get("queried_normal_lso_rows"),
            "same_score_auroc_by_controller": method_auc,
            "ranking_status": ranking_status,
            "synthetic_binary_label_fixture": {
                "status": "SYNTHETIC_DIAGNOSTIC_ONLY",
                "generation": "alternating 0/1 by event position; not Real-IAD labels",
                "n": len(synthetic_labels),
                "auc_status": ranking_status},
            "checkpoint_resume": {
                "status": "PASS",
                "checkpoint_cursor": 17,
                "resumed_scores_trace_predictions_controllers_and_memory_match_uninterrupted": True},
            "duration_seconds": round(time.perf_counter() - started, 3),
            "absolute_paths_emitted": False,
            "confirmation_data_opened": False,
            "stage1_or_stage2_run": False,
        }
    except Exception as exc:
        return {"schema": "calibration.stage0.native_event_replay.v1",
                "status": "BLOCKED",
                "reason": f"{type(exc).__name__}: {exc}",
                "absolute_paths_emitted": False,
                "confirmation_data_opened": False,
                "stage1_or_stage2_run": False}
