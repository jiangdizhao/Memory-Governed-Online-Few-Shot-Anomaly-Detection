"""Bounded, authorization-gated entry point for Gate 1 / future Gate 2 work.

Importing this module does not load a model, read an image, open an annotation,
or authorize an experiment.  Real execution requires Gate 0-certified source
and role identities plus a separate, exact-scope authorization.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import identity
from .evaluation import LabelProjectionAuthorization, evaluate_sealed_category
from .label_broker import RealIADQueryLabelBroker
from .runner import GroupAtomicCategoryRunner, static_arms_for_condition
from .sink import FsyncGroupSink


GATE0_FREEZE_COMMIT = "a34b870fd4700bfe970e459d2963a602f04b1341"


class EntryPointError(RuntimeError):
    """A source, authorization, or bounded-execution contract failed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    result = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", result):
        raise EntryPointError(f"{name} is not a lowercase SHA-256 digest")
    return result


def validate_release_identity(*, identity_manifest: Mapping[str, Any],
                              role_manifest: Mapping[str, Any],
                              release_identity: Mapping[str, Any]) -> str:
    """Accept only a release identity explicitly certified under frozen Gate 0.

    A local snapshot digest is not silently promoted to a publisher release
    checksum.  Any future local-snapshot convention must first be approved in
    Gate 0 and represented by a Gate0-certified identity record.
    """
    try:
        identity_summary = identity.validate_identity_manifest(identity_manifest)
    except Exception as exc:
        raise EntryPointError("identity manifest failed its identity-only schema check") from exc
    if (release_identity.get("schema") != "final-benchmark.dataset-release-identity.v1" or
            release_identity.get("status") != "GATE0_CERTIFIED" or
            release_identity.get("gate0_freeze_commit") != GATE0_FREEZE_COMMIT or
            release_identity.get("dataset_id") != identity_manifest.get("dataset_id") or
            identity_summary["dataset_id"] != role_manifest.get("dataset_id") or
            role_manifest.get("dataset_id") != "realiad"):
        raise EntryPointError("real execution requires a Gate0-certified Real-IAD release identity")
    release_sha = _digest(release_identity.get("dataset_release_sha256"),
                          "certified dataset_release_sha256")
    if role_manifest.get("dataset_release_sha256") != release_sha:
        raise EntryPointError("certified release identity differs from the frozen identity/role manifests")
    identity_sha = _digest(release_identity.get("identity_manifest_sha256"),
                           "certified identity manifest digest")
    role_sha = _digest(release_identity.get("role_manifest_sha256"),
                       "certified role manifest digest")
    if identity_sha != _sha(identity_manifest) or role_sha != _sha(role_manifest):
        raise EntryPointError("release identity is not bound to these exact source manifests")
    provenance = release_identity.get("provenance_kind")
    if provenance not in {"PUBLISHER_AUTHENTICATED", "RESEARCHER_APPROVED_LOCAL_SNAPSHOT"}:
        raise EntryPointError("release provenance must distinguish publisher from approved local snapshot")
    if provenance == "PUBLISHER_AUTHENTICATED":
        publisher_sha = _digest(release_identity.get("publisher_release_sha256"),
                                 "publisher release checksum")
        if identity_manifest.get("official_release_sha256") != publisher_sha or release_sha != publisher_sha:
            raise EntryPointError("publisher release checksum differs from the registered identity manifest")
    else:
        snapshot_sha = _digest(release_identity.get("local_snapshot_sha256"),
                               "approved local snapshot digest")
        if identity_manifest.get("local_snapshot_sha256") != snapshot_sha:
            raise EntryPointError("approved local snapshot digest differs from the identity census")
        _digest(release_identity.get("researcher_decision_sha256"), "Gate 0 local-snapshot decision digest")
        local_identifier = f"realiad-local-snapshot-v1:{snapshot_sha}"
        if release_identity.get("local_snapshot_identifier") != local_identifier:
            raise EntryPointError("local release identifier does not use the proposed canonical form")
        derived = hashlib.sha256(("Gate0-local-release-v1\0realiad\0" +
                                  local_identifier).encode("ascii")).hexdigest()
        if release_sha != derived:
            raise EntryPointError("local release digest does not match the proposed domain-separated derivation")
    return release_sha


def _validate_role_manifest(role_manifest: Mapping[str, Any], *, category: str,
                            condition_id: str, release_sha256: str) -> None:
    seeds = {"P0": 0, "P1": 0, "P2": 1}
    selected = role_manifest.get("selected")
    if (role_manifest.get("dataset_id") != "realiad" or
            role_manifest.get("category") != category or
            role_manifest.get("support_seed") != seeds.get(condition_id) or
            role_manifest.get("dataset_release_sha256") != release_sha256 or
            not isinstance(selected, list) or len(selected) != 68 or
            [row.get("role") for row in selected] != ["SUPPORT"] * 4 + ["C64"] * 64):
        raise EntryPointError("role manifest differs from the registered Real-IAD 4+64 condition")
    if (len({row.get("image_identity_sha256") for row in selected}) != 68 or
            len({row.get("source_group_sha256") for row in selected}) != 68 or
            len({row.get("image_content_sha256") for row in selected}) != 68):
        raise EntryPointError("Support/C64 identity, group, or content overlap is present")


class FinalBenchmarkEntrypoint:
    """One-category coordinator for restricted preflight and separately gated stages."""

    def __init__(self, *, source_commit: str,
                 identity_manifest: Mapping[str, Any], role_manifest: Mapping[str, Any],
                 release_identity: Mapping[str, Any], stream_records: Sequence[Mapping[str, Any]],
                 test_private_locators: Mapping[str, Mapping[str, str]],
                 role_private_locators: Mapping[str, Mapping[str, str]], category: str,
                 condition_id: str):
        self.source_commit = _digest(source_commit, "source commit")
        self.identity_manifest = dict(identity_manifest)
        self.role_manifest = dict(role_manifest)
        self.release_identity = dict(release_identity)
        self.category, self.condition_id = str(category), str(condition_id)
        self.release_sha256 = validate_release_identity(
            identity_manifest=self.identity_manifest, role_manifest=self.role_manifest,
            release_identity=self.release_identity)
        _validate_role_manifest(self.role_manifest, category=self.category,
                                condition_id=self.condition_id,
                                release_sha256=self.release_sha256)
        self.identity_manifest_sha256 = _sha(self.identity_manifest)
        self.role_manifest_sha256 = _sha(self.role_manifest)
        self.stream_records = tuple(dict(row) for row in stream_records)
        self.test_private_locators = {str(key): dict(value)
                                      for key, value in test_private_locators.items()}
        self.role_private_locators = {str(key): dict(value)
                                      for key, value in role_private_locators.items()}
        if (not self.stream_records or
                [row.get("position") for row in self.stream_records] != list(range(len(self.stream_records))) or
                {row.get("image_identity_sha256") for row in self.stream_records} !=
                set(self.test_private_locators) or
                {row.get("image_identity_sha256") for row in self.role_manifest["selected"]} !=
                set(self.role_private_locators)):
            raise EntryPointError("frozen stream and separate TEST/Support-C64 private locators mismatch")
        if any(row.get("dataset_id", "realiad") != "realiad" or
               row.get("category", self.category) != self.category or
               row.get("role", "TEST") != "TEST" for row in self.stream_records):
            raise EntryPointError("entry point accepts only the selected registered Real-IAD TEST stream")
        records = [row for row in self.identity_manifest["identity_records"]
                   if row.get("category") == self.category]
        train_by_id = {row["image_identity_sha256"]: row for row in records
                       if row.get("split") == "train" and row.get("train_normal_eligible") is True}
        test_by_id = {row["image_identity_sha256"]: row for row in records
                      if row.get("split") == "test"}
        stream_ids = {row["image_identity_sha256"] for row in self.stream_records}
        if (stream_ids != set(test_by_id) or
                {row["image_identity_sha256"] for row in self.role_manifest["selected"]} - set(train_by_id) or
                {row.get("source_group_sha256") for row in self.role_manifest["selected"]} &
                {row.get("source_group_sha256") for row in self.stream_records} or
                {row.get("image_content_sha256") for row in self.role_manifest["selected"]} &
                {row.get("image_content_sha256") for row in self.stream_records}):
            raise EntryPointError("identity stream and 4+64 role allocation are not source-disjoint")
        for event in self.stream_records:
            registered = test_by_id[event["image_identity_sha256"]]
            for key in ("image_content_sha256", "source_group_sha256"):
                if event.get(key) != registered.get(key):
                    raise EntryPointError("stream event differs from its registered identity-manifest row")

    def _validate_authorization(self, authorization: Mapping[str, Any], *, scope: str) -> None:
        if (authorization.get("schema") != "final-benchmark.execution-authorization.v1" or
                authorization.get("status") != "AUTHORIZED" or
                authorization.get("scope") != scope or
                authorization.get("source_commit") != self.source_commit or
                authorization.get("identity_manifest_sha256") != self.identity_manifest_sha256 or
                authorization.get("role_manifest_sha256") != self.role_manifest_sha256 or
                authorization.get("dataset_release_sha256") != self.release_sha256 or
                authorization.get("dataset_id") != "realiad" or
                authorization.get("category") != self.category or
                authorization.get("condition_id") != self.condition_id):
            raise EntryPointError(f"separate authorization for {scope} does not match this source/data/category")

    @staticmethod
    def _load_encoder(encoder: Any | None) -> Any:
        if encoder is not None:
            return encoder
        from .native_backend import NativeInputEncoder
        return NativeInputEncoder.load_authorized_runtime()

    def run_restricted_numerical_preflight(self, *, authorization: Mapping[str, Any],
                                           max_events: int = 5, encoder: Any | None = None) -> dict[str, Any]:
        """Score at most two Real-IAD groups without creating a Stage A seal or reading labels."""
        self._validate_authorization(authorization, scope="restricted_numerical_preflight")
        if (isinstance(max_events, bool) or max_events not in (5, 10) or
                int(authorization.get("max_events", -1)) != max_events or
                authorization.get("label_access") != "NONE" or
                authorization.get("stage_b_authorized") is not False):
            raise EntryPointError("restricted preflight is limited to 5 or 10 events and no labels")
        if len(self.stream_records) < max_events or max_events % 5:
            raise EntryPointError("bounded preflight does not cover complete registered five-view groups")
        runtime = self._load_encoder(encoder)
        commissioning = runtime.prepare_commissioning(
            role_manifest=self.role_manifest, private_locators=self.role_private_locators,
            condition_id=self.condition_id)
        backend = runtime.build_backend(category=self.category, condition_id=self.condition_id,
                                        commissioning=commissioning)
        rows = []
        for record in self.stream_records[:max_events]:
            event = runtime.encode_registered_realiad_event(
                record=record, locator=self.test_private_locators[record["image_identity_sha256"]],
                category=self.category, dataset_release_sha256=self.release_sha256,
                stream_seed={"P0": 0, "P1": 11, "P2": 0}[self.condition_id])
            scores = {}
            arm_digests = {arm: backend.state_digest(arm) for arm in ("F-B", "K-B", "F-M", "K-M")}
            shadow_digest = backend.shadow_state_digest()
            static_arms = static_arms_for_condition(self.condition_id)
            static_digests = {arm: backend.static_state_digest(arm) for arm in static_arms}
            for arm in ("F-B", "K-B", "F-M", "K-M"):
                snapshot = backend.snapshot(arm)
                packet = backend.score(arm, event, snapshot)
                if (packet.threshold is None or
                        (packet.state_sha256 and packet.state_sha256 != arm_digests[arm])):
                    raise EntryPointError("native preflight score is missing its threshold or state binding")
                scores[arm] = {"score": float(packet.score),
                    "prediction": int(packet.prediction), "threshold": float(packet.threshold),
                    "state_sha256": packet.state_sha256 or arm_digests[arm]}
            shadow = backend.score_shadow(event, backend.snapshot("F-B"),
                                          backend.shadow_snapshot(backend.snapshot("F-B")))
            if shadow.state_sha256 and shadow.state_sha256 != shadow_digest:
                raise EntryPointError("N-shadow numerical preflight score differs from its state digest")
            static = {}
            for arm in static_arms:
                packet = backend.static_score(arm, event)
                if (packet.threshold is None or
                        (packet.state_sha256 and packet.state_sha256 != static_digests[arm])):
                    raise EntryPointError("static preflight score is missing its threshold or state binding")
                static[arm] = {"score": float(packet.score),
                    "prediction": int(packet.prediction), "threshold": float(packet.threshold),
                    "state_sha256": packet.state_sha256 or static_digests[arm]}
            if ({arm: backend.state_digest(arm) for arm in arm_digests} != arm_digests or
                    backend.shadow_state_digest() != shadow_digest or
                    {arm: backend.static_state_digest(arm) for arm in static_arms} != static_digests):
                raise EntryPointError("score-only numerical preflight mutated a detector trajectory")
            rows.append({"position": event.position, "event_id_sha256": event.event_id_sha256,
                         "arm_scores": scores, "n_shadow_score": float(shadow.score),
                         "static_scores": static})
        return {"status": "BOUNDED_NUMERICAL_PREFLIGHT_INCOMPLETE_NOT_STAGE_A",
                "dataset_id": "realiad", "category": self.category,
                "condition_id": self.condition_id, "event_count": len(rows),
                "source_commit": self.source_commit,
                "identity_manifest_sha256": self.identity_manifest_sha256,
                "role_manifest_sha256": self.role_manifest_sha256,
                "dataset_release_sha256": self.release_sha256,
                "score_rows": rows,
                "labels_read": False, "stage_a_seal_created": False}

    def run_stage_a_category(self, *, authorization: Mapping[str, Any],
                             encoder: Any | None = None, output_directory: str | Path,
                             annotation_path: str | Path, annotation_sha256: str,
                             stream_seed: int) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
        """Run exactly the authorized one-category Real-IAD trajectory."""
        self._validate_authorization(authorization, scope="stage_a_category")
        seed = {"P0": 0, "P1": 11, "P2": 0}[self.condition_id]
        if (stream_seed != seed or authorization.get("label_access") != "QUERY_ONLY" or
                authorization.get("stage_b_authorized") is not False or
                int(authorization.get("expected_event_count", -1)) != len(self.stream_records)):
            raise EntryPointError("Stage A authorization differs from the frozen stream and query-only scope")
        root = Path(output_directory)
        if not root.is_dir():
            raise EntryPointError("Stage A output directory must exist before execution")
        output_paths = (root / "stage_a_journal.jsonl", root / "stage_a_category_seal.json",
                        root / "private_feedback.jsonl", root / "stage_a_category_report.json")
        if any(path.exists() for path in output_paths):
            raise EntryPointError("Stage A output files must all be fresh and non-overwriting")
        runtime = self._load_encoder(encoder)
        commissioning = runtime.prepare_commissioning(
            role_manifest=self.role_manifest, private_locators=self.role_private_locators,
            condition_id=self.condition_id)
        backend = runtime.build_backend(category=self.category, condition_id=self.condition_id,
                                        commissioning=commissioning)
        groups = []
        for row in self.stream_records:
            group = str(row["source_group_sha256"])
            if not groups or groups[-1] != group:
                groups.append(group)
        if len(groups) * 5 != len(self.stream_records):
            raise EntryPointError("registered Real-IAD TEST stream is not a complete ordered five-view stream")
        output_identity = {"source_commit": self.source_commit,
            "identity_manifest_sha256": self.identity_manifest_sha256,
            "role_manifest_sha256": self.role_manifest_sha256,
            "dataset_release_sha256": self.release_sha256,
            "dataset_id": "realiad", "category": self.category,
            "condition_id": self.condition_id}
        sink = FsyncGroupSink(root / "stage_a_journal.jsonl", identity=output_identity)
        broker = None
        try:
            broker = RealIADQueryLabelBroker(category=self.category, condition_id=self.condition_id,
                stream_seed=seed, dataset_release_sha256=self.release_sha256,
                annotation_path=annotation_path, annotation_sha256=annotation_sha256,
                stream_records=self.stream_records, private_locators=self.test_private_locators,
                sink=sink, private_feedback_path=root / "private_feedback.jsonl")
            runner = GroupAtomicCategoryRunner(dataset_id="realiad", category=self.category,
                condition_id=self.condition_id, support_seed={"P0": 0, "P1": 0, "P2": 1}[self.condition_id],
                stream_seed=seed, expected_event_count=len(self.stream_records),
                expected_group_order=groups, rho=0.03, backend=backend, broker=broker, sink=sink)
            for start in range(0, len(self.stream_records), 5):
                events = []
                for record in self.stream_records[start:start + 5]:
                    event = runtime.encode_registered_realiad_event(
                        record=record, locator=self.test_private_locators[record["image_identity_sha256"]],
                        category=self.category, dataset_release_sha256=self.release_sha256,
                        stream_seed=seed)
                    events.append(event)
                runner.process_group(events)
            seal = runner.seal()
            report = {"schema": "final-benchmark.stage-a-category-report.v1",
                **asdict(seal), "source_commit": self.source_commit,
                "identity_manifest_sha256": self.identity_manifest_sha256,
                "role_manifest_sha256": self.role_manifest_sha256,
                "dataset_release_sha256": self.release_sha256,
                "query_accounting": runner.accounting(),
                "resource_report": runner.resource_report()}
            target = root / "stage_a_category_report.json"
            with target.open("xb") as stream:
                payload = _canonical(report) + b"\n"
                stream.write(payload)
                stream.flush()
                import os
                os.fsync(stream.fileno())
            return seal, list(runner.rows), list(runner.group_seals)
        finally:
            if broker is not None:
                broker.close()
            sink.close()

    @staticmethod
    def evaluate_stage_b(*, dataset_id: str, category: str, rows: Sequence[Mapping[str, Any]],
                         seal: Any, unit_seals: Sequence[Mapping[str, Any]],
                         authorization: LabelProjectionAuthorization,
                         projection_loader: Callable[[], Mapping[str, int]]) -> dict[str, Any]:
        """Invoke the label projection only after Stage A and its separate grant validate."""
        if (authorization.status != "AUTHORIZED_POST_STAGE_A_LABEL_PROJECTION" or
                authorization.scope != "stage_b_category_evaluation" or
                authorization.dataset_id != dataset_id or authorization.category != category or
                authorization.stage_a_seal_sha256 != seal.durable_seal_sha256 or
                seal.status != "SEALED_COMPLETE" or
                not re.fullmatch(r"[0-9a-f]{64}", authorization.source_sha256)):
            raise EntryPointError("Stage B label projection is not separately authorized for this seal")
        if dataset_id == "visa_1cls" and (
                authorization.unit_candidate_sha256 != getattr(seal, "unit_candidate_sha256", None) or
                authorization.unit_approval_sha256 != getattr(seal, "unit_approval_sha256", None)):
            raise EntryPointError("VisA Stage B grant does not bind the sealed atomic-unit exception")
        labels = projection_loader()
        if not isinstance(labels, Mapping):
            raise EntryPointError("authorized Stage B projection did not return an event-label mapping")
        return evaluate_sealed_category(dataset_id=dataset_id, category=category,
            rows=rows, seal=seal, group_seals=unit_seals,
            labels_by_event_id=labels, label_projection=authorization)
