from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from experiments.calibration.fs4_durable_ledger import canonical, digest
from experiments.calibration.fs4_r2_context import FS4BoundaryError
from experiments.calibration.fs4_r2_security import WindowsPrivateStore
from experiments.calibration.fs4_r4_blinded import read_blinded_image, validate_blinded_execution
from experiments.calibration.fs4_r4_preparation import blinded_ledger_manifest, prepare_blinded_registration
from experiments.calibration.fs4_research_ledger import RESEARCH_SOURCES
from test_fs4_hardening_r1 import hid
from test_fs4_hardening_r2 import synthetic_registration
from test_fs4_research_execution import TinyBackbone
from test_fs4_threshold import native_bank_and_writer


def _synthetic_r4_entry(connection, broker, config_path, broker_manifest_sha256):
    """Synthetic-only native model adapter installed in the spawned test worker."""
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import fs4_r2_context as context, profiles, r1_matched_correction_pilot as r1
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration.fs4_threshold_integration import CoverageMemory, NativeRuntime
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.calibration.fs4_r4_worker import _worker_main
    tmp = Path(config_path).parent
    native, _, _ = native_bank_and_writer(4096)
    profile = profiles.NativeScorerConfig()
    fields = ("augmentation_seed", "bank_seed", "fixed_capacity", "fixed_per_image", "augmentations",
              "include_identity", "multiplier", "feature_fuse", "topk_patches", "image_aggregation",
              "image_topk", "image_quantile", "knn")
    cfg = p0.Config(device="cpu", **{key: getattr(profile, key) for key in fields})
    def encode(_backbone, image, _mode):
        value = image.getpixel((0, 0))[0]
        features = torch.zeros((784, 4096), dtype=torch.float32); features[:, 1 + value % 32] = 1
        return [features], torch.tensor([1.0, value / 255.0], dtype=torch.float32)
    def builder(*, impl, cfg, support_cls, **kwargs):
        (tmp / "support-paths.json").write_bytes(canonical(kwargs["support_paths"]))
        _, bank, writer = native_bank_and_writer(4096)
        reference = {"fixed_support_sha256": _fixed_bank_digest(bank),
                     "writer_reference": {"theta": .5, "sigma": .02}, "theta": .5}
        runtime = NativeRuntime(impl=impl, bank=bank, cfg=cfg, writer=writer, correction_cfg=bank.corr_cfg,
            correction_cb=bank.corr_cb["p0"]["id"][0], reference=reference)
        if (tmp / "wrong-bank").exists(): runtime.fixed_sha = hid("wrong-built-bank")
        return runtime, reference, CoverageMemory(support_cls)
    native.DINOv2MultiLayerBackbone = TinyBackbone; native.encode_pil = encode
    native.make_support_refs = lambda image, *_a, **_k: [image] * 4
    p0.require_sources = lambda: native
    r3._get_verified_model_receipt = lambda: {}
    r3._runtime_and_native_setup = lambda *_a: (lambda: None, {})
    r3._native_config = lambda *_a: (cfg, profile)
    context.build_native_from_four_support = builder; context.PREFLIGHT = tmp / "preflight.json"
    original_hash = r1._model_state_sha256
    def counted(model):
        with (tmp / "model-hashes.txt").open("a") as out: out.write("1\n")
        return original_hash(model)
    r1._model_state_sha256 = counted
    _worker_main(connection, broker, config_path, broker_manifest_sha256)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.calibration.stage05 import opaque_digest
    from experiments.risk_safe.preflight import parse_image_name
    old_ledger, original = synthetic_registration(tmp_path / "original")
    _, bank, _ = native_bank_and_writer(4096)
    fixed = _fixed_bank_digest(bank)
    model_hash = r1._model_state_sha256(TinyBackbone("cpu").model)
    support = original["categories"]["audiojack"]["supports"]
    parsed = [parse_image_name(tmp_path / "original" / Path(item["relative_path"]), expected_class="audiojack") for item in support]
    preflight = {"status": "PASS", "loaded_model_state_sha256": model_hash,
        "model": {"checkpoint_sha256": original["checkpoint_sha256"], "hub_source_tree_sha256": original["hub_source_tree_sha256"]},
        "categories": {"audiojack": {"status": "PASS", "fixed_evidence": {"actual_structured_sha256": fixed},
            "support_groups_sha256": opaque_digest("support-groups", [p.official_sample_key for p in parsed]),
            "support_images_sha256": opaque_digest("support-images", [p.rel_path for p in parsed]),
            "support_image_content_sha256": [item["content_sha256"] for item in support]}}}
    data = canonical(preflight); (tmp_path / "preflight.json").write_bytes(data)
    original.update(model_state_sha256=model_hash, preflight_receipt_sha256=hashlib.sha256(data).hexdigest())
    old_ledger.update(model_state_sha256=model_hash, fixed_support_sha256_by_category={"audiojack": fixed})
    for path in RESEARCH_SOURCES:
        old_ledger["source_fingerprints"][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    # Fixed synthetic identities exercise a legal query without changing the
    # policy or using a label to select an event. Production uses secrets.
    from experiments.calibration.fs4_stochastic_bea import action_uniform, candidate_ordinal
    from experiments.calibration import fs4_r4_preparation as preparation
    group = hid("synthetic-r4-random-group")
    selected = candidate_ordinal(37, group)
    sequence = [group]
    for view in range(1, 6):
        trial = 0; event = hid(f"r4-synthetic-event-{view}-{trial}")
        while view == selected and action_uniform(37, group, event) >= .05:
            trial += 1; event = hid(f"r4-synthetic-event-{view}-{trial}")
        sequence.extend([event, hid(f"r4-synthetic-resource-{view}")])
    sequence.append(hid("r4-synthetic-private-salt"))
    identifiers = iter(sequence)
    monkeypatch.setattr(preparation.secrets, "token_hex", lambda n: next(identifiers))
    execution, receipt = prepare_blinded_registration(original=original, original_ledger=old_ledger,
        labels_by_position={i: 0 for i in range(5)}, private_root=tmp_path / "private", image_root=tmp_path / "blinded")
    manifest = blinded_ledger_manifest(old_ledger, execution)
    manifest_path = tmp_path / "public-manifest.json"; manifest_path.write_bytes(canonical(manifest))
    config = {"manifest_path": str(manifest_path), "run_root": str(tmp_path / "run"), "category": "audiojack",
              "create": True, "checkpoint": None}
    config_path = tmp_path / "worker.json"; config_path.write_bytes(canonical(config))
    return SimpleNamespace(root=tmp_path, original=original, original_ledger=old_ledger, execution=execution, manifest=manifest,
                           receipt=receipt, config=config, config_path=config_path)


def test_preparation_blinds_paths_preserves_views_and_pixels(prepared):
    execution = validate_blinded_execution(prepared.execution, prepared.manifest)
    rows = execution["categories"]["audiojack"]["events"]
    assert execution["categories"]["audiojack"]["supports"] == prepared.original["categories"]["audiojack"]["supports"]
    assert len({row["group"] for row in rows}) == 1 and [row["view"] for row in rows] == [1, 2, 3, 4, 5]
    assert len({row["event"] for row in rows}) == 5
    private = (prepared.root / "private" / "mapping.json").read_bytes()
    public_events = canonical(rows).decode()
    assert "original_path" not in public_events and "label" not in public_events and "_OK_" not in public_events and "_NG_" not in public_events
    private_path_json = json.dumps(str(prepared.root / "private"))[1:-1]
    assert private_path_json not in canonical(prepared.config).decode()
    assert private_path_json not in canonical(prepared.manifest).decode()
    for row, old in zip(rows, prepared.original["categories"]["audiojack"]["events"]):
        image = read_blinded_image(execution, row)
        with Image.open(prepared.root / "original" / Path(old["relative_path"])) as source:
            assert image.tobytes() == source.convert("RGB").tobytes()
        assert old["relative_path"].encode() not in canonical(row)
    assert b"original_path" in private
    with pytest.raises(FileNotFoundError):
        WindowsPrivateStore(prepared.root / "blinded", create=False).read("mapping.json")


def test_blinded_resource_and_identity_tampering_fails(prepared):
    rows = prepared.execution["categories"]["audiojack"]["events"]
    first, second = rows[0], rows[1]
    path = prepared.root / "blinded" / first["resource"]
    original = path.read_bytes(); path.write_bytes((prepared.root / "blinded" / second["resource"]).read_bytes())
    with pytest.raises(FS4BoundaryError): read_blinded_image(prepared.execution, first)
    path.write_bytes(original)
    forged = copy.deepcopy(first); forged["pixel_sha256"] = hid("other pixels")
    with pytest.raises(FS4BoundaryError): read_blinded_image(prepared.execution, forged)
    for field, value in (("group", hid("wrong group")), ("event", rows[1]["event"])):
        bad = copy.deepcopy(prepared.execution); bad["categories"]["audiojack"]["events"][0][field] = value
        with pytest.raises(FS4BoundaryError): validate_blinded_execution(bad, prepared.manifest)
    with pytest.raises(Exception):
        prepare_blinded_registration(original=prepared.original, original_ledger=prepared.manifest,
            labels_by_position={i: 0 for i in range(5)}, private_root=prepared.root / "private", image_root=prepared.root / "blinded")
    with pytest.raises(ValueError, match="separate"):
        prepare_blinded_registration(original=prepared.original, original_ledger=prepared.original_ledger,
            labels_by_position={i: 0 for i in range(5)}, private_root=prepared.root / "private3", image_root=prepared.root / "original" / "nested")
    source_row = prepared.original["categories"]["audiojack"]["events"][0]
    source_path = prepared.root / "original" / Path(source_row["relative_path"])
    source_path.write_bytes(b"wrong synthetic original bytes")
    with pytest.raises(ValueError, match="INCOMPLETE") as failure:
        prepare_blinded_registration(original=prepared.original, original_ledger=prepared.original_ledger,
            labels_by_position={i: 0 for i in range(5)}, private_root=prepared.root / "private2", image_root=prepared.root / "blinded2")
    assert str(source_path) not in str(failure.value)


def test_preparation_strips_label_bearing_image_metadata(prepared, monkeypatch):
    import os
    from PIL import PngImagePlugin
    from experiments.calibration import fs4_r4_preparation as preparation
    original = copy.deepcopy(prepared.original)
    source_ledger = copy.deepcopy(prepared.original_ledger)
    row = original["categories"]["audiojack"]["events"][0]
    path = prepared.root / "original" / Path(row["relative_path"])
    with Image.open(path) as image: pixels = image.convert("RGB")
    metadata = PngImagePlugin.PngInfo(); metadata.add_text("secret_label", "NG")
    stream = io.BytesIO(); pixels.save(stream, format="PNG", pnginfo=metadata)
    raw = stream.getvalue(); path.write_bytes(raw); row["content_sha256"] = hashlib.sha256(raw).hexdigest()
    original["dataset_identity_sha256"] = digest({"dataset_root": original["dataset_root"],
        "category_order": original["category_order"], "categories": original["categories"]})
    source_ledger["dataset_identity_sha256"] = original["dataset_identity_sha256"]
    monkeypatch.setattr(preparation.secrets, "token_hex", lambda n: os.urandom(n).hex())
    execution, _ = prepare_blinded_registration(original=original, original_ledger=source_ledger,
        labels_by_position={i: 0 for i in range(5)}, private_root=prepared.root / "metadata-private",
        image_root=prepared.root / "metadata-blinded")
    result = read_blinded_image(execution, execution["categories"]["audiojack"]["events"][0])
    assert result.info == {} and result.tobytes() == pixels.tobytes()


def test_worker_rejects_wrong_four_support_bank(prepared, monkeypatch):
    from experiments.calibration import fs4_r4_worker as worker_module
    from experiments.calibration.fs4_r4_worker import BlindedResearchWorker, BrokerProcess
    from experiments.calibration.fs4_research_worker import ResearchWorkerError
    monkeypatch.setattr(worker_module, "_worker_main", _synthetic_r4_entry)
    (prepared.root / "wrong-bank").write_text("synthetic", encoding="utf-8")
    broker = BrokerProcess(private_root=prepared.root / "private", run_root=prepared.root / "run",
                           disclosures_root=prepared.root / "disclosures", manifest_path=prepared.root / "public-manifest.json")
    with pytest.raises(ResearchWorkerError):
        BlindedResearchWorker(prepared.config_path, broker)


def test_broker_rejects_private_mapping_mismatch_before_worker_starts(prepared):
    from experiments.calibration.fs4_r4_worker import BrokerProcess
    from experiments.calibration.fs4_research_worker import ResearchWorkerError
    path = prepared.root / "private" / "mapping.json"
    mapping = json.loads(path.read_bytes()); mapping["events"][0]["label"] = 1
    path.write_bytes(canonical(mapping) + b"\n")
    with pytest.raises(ResearchWorkerError):
        BrokerProcess(private_root=prepared.root / "private", run_root=prepared.root / "run",
                      disclosures_root=prepared.root / "disclosures", manifest_path=prepared.root / "public-manifest.json")
    assert not (prepared.root / "run").exists()


def test_spawned_blinded_worker_broker_recovery_and_stage_b(prepared, monkeypatch):
    from experiments.calibration import fs4_r4_worker as worker_module
    from experiments.calibration.fs4_r4_worker import BlindedResearchWorker, BrokerProcess
    monkeypatch.setattr(worker_module, "_worker_main", _synthetic_r4_entry)
    def broker(): return BrokerProcess(private_root=prepared.root / "private", run_root=prepared.root / "run",
                                       disclosures_root=prepared.root / "disclosures", manifest_path=prepared.root / "public-manifest.json")
    worker = BlindedResearchWorker(prepared.config_path, broker())
    try:
        from experiments.p0_protocol_audit import deterministic_subseed
        from experiments.calibration.profiles import NativeScorerConfig
        support_paths = json.loads((prepared.root / "support-paths.json").read_bytes())
        expected_paths = [str(prepared.root / "original" / Path(row["relative_path"]))
                          for row in prepared.original["categories"]["audiojack"]["supports"]]
        assert support_paths == expected_paths
        seed = NativeScorerConfig().augmentation_seed
        assert [deterministic_subseed(seed, p) for p in support_paths] == [deterministic_subseed(seed, p) for p in expected_paths]
        worker.checkpoint("initial")
        model_hashes_before = len((prepared.root / "model-hashes.txt").read_text().splitlines())
        for pos in range(5): worker.step(pos)
        assert len((prepared.root / "model-hashes.txt").read_text().splitlines()) == model_hashes_before
    finally:
        worker.close()
    public_artifacts = [prepared.config_path, prepared.root / "public-manifest.json", prepared.root / "run" / "manifest.json",
                        prepared.root / "run" / "events.jsonl", *list((prepared.root / "run" / "checkpoints" / "initial").iterdir())]
    hidden_names = [name.encode() for row in prepared.original["categories"]["audiojack"]["events"]
                    for name in (row["relative_path"], row["relative_path"].replace("/", "\\"))]
    for artifact in public_artifacts:
        if artifact.is_file():
            body = artifact.read_bytes()
            assert all(name not in body for name in hidden_names)
            assert b'"original_path"' not in body
    from experiments.calibration.fs4_r4_broker import QueryOnlyBroker
    from experiments.calibration.fs4_durable_ledger import LedgerIntegrityError
    forbidden = broker()
    forbidden.endpoint.send_bytes(canonical({"command": "dump"}))
    assert json.loads(forbidden.endpoint.recv_bytes(4096)) == {"status": "ERROR", "message": "broker rejected disclosure"}
    forbidden.process.join(); forbidden.endpoint.close()
    rows = prepared.execution["categories"]["audiojack"]["events"]
    check = QueryOnlyBroker(private_root=prepared.root / "private", run_root=prepared.root / "run",
                            disclosures_root=prepared.root / "disclosures")
    for request in ({"event": rows[0]["event"], "group": rows[0]["group"], "position": 0, "query_sha256": hid("forged")},
                    {"event": hid("future"), "group": hid("other"), "position": 9, "query_sha256": hid("forged")}):
        with pytest.raises(LedgerIntegrityError): check.disclose(**request)
    # Reconstruct the genuine durable prefix ending at LABEL_INTENT, then test
    # independent verification while disclosure is actually legally pending.
    lines = (prepared.root / "run" / "events.jsonl").read_bytes().splitlines()
    index = next(i for i, line in enumerate(lines) if json.loads(line)["kind"] == "LABEL_INTENT")
    query_row = json.loads(lines[index - 1])["payload"]
    pred = query_row["receipt"]["prediction"]
    request = {"event": pred["event"], "group": pred["group"], "position": pred["position"],
               "query_sha256": query_row["query_sha256"]}
    probe_root = prepared.root / "broker-probe"
    probe_store = WindowsPrivateStore(probe_root, create=True)
    probe_store.write("manifest.json", (prepared.root / "run" / "manifest.json").read_bytes())
    probe_store.write("events.jsonl", b"\n".join(lines[:index + 1]) + b"\n")
    probe = QueryOnlyBroker(private_root=prepared.root / "private", run_root=probe_root,
                            disclosures_root=prepared.root / "broker-probe-disclosures")
    for bad in (dict(request, query_sha256=hid("forged-receipt")), dict(request, group=hid("wrong-group")),
                dict(request, position=request["position"] + 1), dict(request, event=hid("unqueried"))):
        with pytest.raises(LedgerIntegrityError): probe.disclose(**bad)
    assert probe.disclose(**request) == 0
    with pytest.raises(LedgerIntegrityError, match="duplicate"):
        probe.disclose(**request)
    from experiments.calibration.fs4_r4_evaluator import evaluate_blinded
    with pytest.raises(subprocess.CalledProcessError):
        evaluate_blinded(run_root=prepared.root / "run", private_root=prepared.root / "unopened-private",
                         grant_path=prepared.root / "missing-grant", result_root=prepared.root / "uncreated-result")
    # The projection helper is independently gated, even if invoked directly.
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([sys.executable, "-m", "experiments.calibration.fs4_r4_mapping_verify",
            str(prepared.root / "private"), str(prepared.root / "run"), digest(prepared.manifest)],
            check=True, capture_output=True)
    disclosures_before = sorted(p.name for p in (prepared.root / "disclosures").glob("*.json"))
    assert len(disclosures_before) == 1
    # A regenerated public identifier cannot reopen the existing run.
    regenerated = copy.deepcopy(prepared.manifest)
    regenerated["research_execution_manifest"]["categories"]["audiojack"]["events"][0]["event"] = hid("regenerated-id")
    alt_manifest = prepared.root / "regenerated.json"; alt_manifest.write_bytes(canonical(regenerated))
    bad_config = dict(prepared.config, create=False, checkpoint="initial", manifest_path=str(alt_manifest))
    prepared.config_path.write_bytes(canonical(bad_config))
    from experiments.calibration.fs4_research_worker import ResearchWorkerError
    with pytest.raises(ResearchWorkerError):
        BlindedResearchWorker(prepared.config_path, broker())
    config = dict(prepared.config, create=False, checkpoint="initial")
    prepared.config_path.write_bytes(canonical(config))
    worker = BlindedResearchWorker(prepared.config_path, broker())
    try:
        for pos in range(5): worker.step(pos)
        assert sorted(p.name for p in (prepared.root / "disclosures").glob("*.json")) == disclosures_before
        worker.seal_category(); worker.seal_global_a()
    finally:
        worker.close()
    from experiments.calibration.fs4_research_stage_a_verify import verify_stage_a
    projection = verify_stage_a(prepared.root / "run")
    assert len(projection["queried_feedback"]) == 1
    assert len(list((prepared.root / "disclosures").glob("*.json"))) == 1
    keys = ("run_id", "manifest_sha256", "source_commit", "model_state_sha256", "dataset_identity_sha256",
            "fixed_support_sha256_by_category", "stream_sha256_by_category", "stage_a_sha256")
    grant = {key: projection[key] for key in keys}
    grant.update(schema="FS4_R4_STAGE_B_GRANT_V1", scope="AGGREGATE_METRICS",
                 source_mapping_sha256=prepared.execution["source_mapping_sha256"])
    grant_path = prepared.root / "grant.json"; grant_path.write_bytes(canonical(grant))
    completed = subprocess.run([sys.executable, "-m", "experiments.calibration.fs4_r4_evaluator",
        str(prepared.root / "run"), str(prepared.root / "private"), str(grant_path), str(prepared.root / "results")],
        check=True, capture_output=True)
    result = json.loads(completed.stdout)
    assert result["metrics"]["audiojack"]
    for name, counts in result["metrics"]["audiojack"].items():
        decisions = [int(row["scores"][name.rsplit("_", 1)[1]] >= row["thresholds"][name])
                     for row in projection["events"]]
        assert counts["fp"] == sum(decisions) and counts["tn"] == len(decisions) - sum(decisions)
        assert counts["tp"] == 0 and counts["fn"] == 0 and counts["balanced_accuracy"] is None
    from experiments.calibration.fs4_r4_evaluator import verify_blinded_completion
    assert verify_blinded_completion(prepared.root / "results") == result
    output_projection = prepared.root / "results" / "outcome_projection.json"
    saved_projection = output_projection.read_bytes(); output_projection.write_bytes(b"altered")
    with pytest.raises(ValueError, match="projection checksum"):
        verify_blinded_completion(prepared.root / "results")
    output_projection.write_bytes(saved_projection)
    mapping = prepared.root / "private" / "mapping.json"
    old = mapping.read_bytes(); altered = json.loads(old); altered["events"][0]["label"] = 1
    mapping.write_bytes(canonical(altered) + b"\n")
    with pytest.raises(subprocess.CalledProcessError):
        evaluate_blinded(run_root=prepared.root / "run", private_root=prepared.root / "private",
                         grant_path=grant_path, result_root=prepared.root / "invalid-results")
    mapping.write_bytes(old)
