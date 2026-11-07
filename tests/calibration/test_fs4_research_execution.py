from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.calibration.fs4_durable_ledger import canonical
from experiments.calibration.fs4_r2_security import WindowsPrivateStore
from experiments.calibration.fs4_research_ledger import RESEARCH_SOURCES
from experiments.calibration.fs4_research_worker import ResearchWorker
from test_fs4_hardening_r1 import hid
from test_fs4_hardening_r2 import synthetic_registration
from test_fs4_threshold import native_bank_and_writer


class TinyBackbone:
    def __init__(self, device):
        self.model = torch.nn.Linear(2, 2, bias=False).eval()
        with torch.no_grad(): self.model.weight.copy_(torch.eye(2))
        self.model.num_features = 2
        self.model.blocks = torch.nn.ModuleList([torch.nn.Identity().eval() for _ in range(4)])
        self.model.eval()
        self.model_name = "dinov2_vitl14_reg"; self.img_size = 392; self.resize_size = 448
        self.use_fp16 = True; self.device = torch.device("cuda"); self.layers = [0, 1, 2, 3]
        self.mean = (.485, .456, .406); self.std = (.229, .224, .225)

    def encode_batch(self, _images):
        raise AssertionError("synthetic native encode is used")


def _synthetic_spawn_entry(connection, config_path):
    """Test-only adapter installed inside a freshly spawned Python worker."""
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import fs4_r2_context as context, profiles, r1_matched_correction_pilot as r1
    from experiments.calibration import fs4_research_context as research_context
    from experiments.calibration import r3_fullstream_2x2 as r3
    from experiments.calibration.fs4_threshold_integration import CoverageMemory, NativeRuntime
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.calibration.fs4_research_worker import _worker_main

    config = json.loads(Path(config_path).read_text())
    tmp = Path(config_path).parent
    # Synthetic-only fixture bypasses the causal hold to exercise the other
    # components. It must never be used for a scientific execution.
    research_context.require_label_blind_dataset_adapter = lambda: None
    native, _, _ = native_bank_and_writer(4096)
    profile = profiles.NativeScorerConfig()
    fields = ("augmentation_seed", "bank_seed", "fixed_capacity", "fixed_per_image", "augmentations",
              "include_identity", "multiplier", "feature_fuse", "topk_patches", "image_aggregation",
              "image_topk", "image_quantile", "knn")
    cfg = p0.Config(device="cpu", **{key: getattr(profile, key) for key in fields})

    def encode(_backbone, image, _mode):
        value = image.getpixel((0, 0))[0]
        features = torch.zeros((784, 4096), dtype=torch.float32)
        features[:, 1 + value % 32] = 1
        return [features], torch.tensor([1.0, value / 255.0], dtype=torch.float32)

    def builder(*, impl, cfg, support_cls, **_kwargs):
        _, bank, writer = native_bank_and_writer(4096)
        reference = {"fixed_support_sha256": _fixed_bank_digest(bank),
                     "writer_reference": {"theta": .5, "sigma": .02}, "theta": .5}
        runtime = NativeRuntime(impl=impl, bank=bank, cfg=cfg, writer=writer, correction_cfg=bank.corr_cfg,
            correction_cb=bank.corr_cb["p0"]["id"][0], reference=reference)
        if (tmp / "wrong_bank").exists(): runtime.fixed_sha = hid("wrong-built-bank")
        return runtime, reference, CoverageMemory(support_cls)

    class WrongModel(TinyBackbone):
        def __init__(self, device):
            super().__init__(device)
            with torch.no_grad(): self.model.weight[0, 0] += 1
    native.DINOv2MultiLayerBackbone = WrongModel if (tmp / "wrong_model").exists() else TinyBackbone
    native.encode_pil = encode
    native.make_support_refs = lambda image, *_a, **_k: [image] * 4
    p0.require_sources = lambda: native
    r3._get_verified_model_receipt = lambda: {}
    r3._runtime_and_native_setup = lambda *_a: (lambda: None, {})
    r3._native_config = lambda *_a: (cfg, profile)
    context.build_native_from_four_support = builder
    context.PREFLIGHT = tmp / "preflight.json"
    original_hash = r1._model_state_sha256
    def counted(model):
        with (tmp / "hash_count.txt").open("a") as out: out.write("1\n")
        return original_hash(model)
    r1._model_state_sha256 = counted
    _worker_main(connection, config_path)


@pytest.fixture
def synthetic_worker_registration(tmp_path):
    from experiments.calibration import r1_matched_correction_pilot as r1
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.calibration.stage05 import opaque_digest
    from experiments.risk_safe.preflight import parse_image_name
    ledger, execution = synthetic_registration(tmp_path / "dataset")
    _, bank, _ = native_bank_and_writer(4096)
    fixed = _fixed_bank_digest(bank)
    model_hash = r1._model_state_sha256(TinyBackbone("cpu").model)
    supports = execution["categories"]["audiojack"]["supports"]
    parsed = [parse_image_name(tmp_path / "dataset" / Path(row["relative_path"]), expected_class="audiojack") for row in supports]
    preflight = {"status": "PASS", "loaded_model_state_sha256": model_hash,
        "model": {"checkpoint_sha256": execution["checkpoint_sha256"], "hub_source_tree_sha256": execution["hub_source_tree_sha256"]},
        "categories": {"audiojack": {"status": "PASS", "fixed_evidence": {"actual_structured_sha256": fixed},
            "support_groups_sha256": opaque_digest("support-groups", [p.official_sample_key for p in parsed]),
            "support_images_sha256": opaque_digest("support-images", [p.rel_path for p in parsed]),
            "support_image_content_sha256": [row["content_sha256"] for row in supports]}}}
    preflight_bytes = canonical(preflight)
    (tmp_path / "preflight.json").write_bytes(preflight_bytes)
    execution.update(model_state_sha256=model_hash, preflight_receipt_sha256=hashlib.sha256(preflight_bytes).hexdigest())
    ledger.update(execution_scope="TRUSTED_LOCAL_RESEARCH", model_state_sha256=model_hash,
                  fixed_support_sha256_by_category={"audiojack": fixed}, research_execution_manifest=execution,
                  feedback_root=str(tmp_path / "feedback"))
    for name in RESEARCH_SOURCES:
        ledger["source_fingerprints"][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    store = WindowsPrivateStore(tmp_path / "feedback", create=True)
    for item in execution["categories"]["audiojack"]["events"]:
        store.write(item["event"] + ".json", canonical({"event": item["event"], "position": item["position"], "label": 0}))
    manifest_path = tmp_path / "manifest.json"; manifest_path.write_bytes(canonical(ledger))
    config = {"manifest_path": str(manifest_path), "run_root": str(tmp_path / "run"), "category": "audiojack",
              "feedback_root": str(tmp_path / "feedback"), "create": True, "checkpoint": None}
    config_path = tmp_path / "worker.json"; config_path.write_bytes(canonical(config))
    return SimpleNamespace(root=tmp_path, config=config, config_path=config_path, manifest=ledger, execution=execution)


def test_spawn_isolates_parent_monkeypatch_and_bounds_model_hashes(synthetic_worker_registration, monkeypatch):
    import experiments.calibration.fs4_research_worker as worker_module
    from experiments import p0_protocol_audit as p0
    native = p0.require_sources()
    monkeypatch.setattr(native, "encode_pil", lambda *_: (_ for _ in ()).throw(AssertionError("parent patch leaked")))
    monkeypatch.setattr(worker_module, "_worker_main", _synthetic_spawn_entry)
    data = synthetic_worker_registration
    worker = ResearchWorker(data.config_path)
    try:
        count = len((data.root / "hash_count.txt").read_text().splitlines())
        first = worker.step(0)
        assert first["prediction"]["event"] == data.execution["categories"]["audiojack"]["events"][0]["event"]
        worker.step(1)
        assert len((data.root / "hash_count.txt").read_text().splitlines()) == count
        with pytest.raises(TypeError): worker.step(torch.zeros(1))
        worker.checkpoint("after_two")
    finally:
        worker.close()
    # The verifier must reject incomplete Stage A before touching a missing
    # outcome directory or evaluation grant.
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([sys.executable, "-m", "experiments.calibration.fs4_research_evaluator",
            str(data.root / "run"), str(data.root / "missing-grant.json"), str(data.root / "missing-labels"),
            "labels.json", str(data.root / "missing-result")], check=True, capture_output=True)
    config = dict(data.config, create=False, checkpoint="after_two")
    data.config_path.write_bytes(canonical(config))
    resumed = ResearchWorker(data.config_path)
    try:
        assert resumed.step(2)["prediction"]["position"] == 2
        resumed.step(3); resumed.step(4)
        resumed.seal_category()
        resumed.seal_global_a()
    finally:
        resumed.close()
    from experiments.calibration.fs4_research_stage_a_verify import verify_stage_a
    projection = verify_stage_a(data.root / "run")
    labels = {row["event"]: row["position"] % 2 for row in projection["events"]}
    labels.update(projection["queried_feedback"])
    label_data = canonical(labels)
    label_store = WindowsPrivateStore(data.root / "outcomes", create=True)
    label_store.write("labels.json", label_data)
    keys = ("run_id", "manifest_sha256", "source_commit", "model_state_sha256", "dataset_identity_sha256",
            "fixed_support_sha256_by_category", "stream_sha256_by_category", "stage_a_sha256")
    grant = {key: projection[key] for key in keys}
    grant.update(schema="FS4_LOCAL_RESEARCH_STAGE_B_GRANT_V1", scope="AGGREGATE_METRICS",
                 labels_sha256=hashlib.sha256(label_data).hexdigest())
    grant_path = data.root / "grant.json"; grant_path.write_bytes(canonical(grant))
    evaluated = subprocess.run([sys.executable, "-m", "experiments.calibration.fs4_research_evaluator",
        str(data.root / "run"), str(grant_path), str(data.root / "outcomes"), "labels.json",
        str(data.root / "results")], check=True, capture_output=True)
    result = json.loads(evaluated.stdout)
    metrics = result["metrics"]["audiojack"]
    for name, counts in metrics.items():
        expected = [int(row["scores"][name.rsplit("_", 1)[1]] >= row["thresholds"][name]) for row in projection["events"]]
        assert counts["tp"] == sum(y == 1 and p == 1 for y, p in zip(labels.values(), expected))
        assert counts["tn"] == sum(y == 0 and p == 0 for y, p in zip(labels.values(), expected))
    from experiments.calibration.fs4_research_evaluator import verify_completion
    assert verify_completion(data.root / "results") == result
    (data.root / "results" / "result.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        verify_completion(data.root / "results")


@pytest.mark.parametrize("case", ["dataset", "category", "model", "support", "event", "seed", "source"])
def test_research_registration_rejects_mismatched_identity(synthetic_worker_registration, case):
    from experiments.calibration.fs4_r2_context import FS4BoundaryError, validate_execution_manifest
    ledger = copy.deepcopy(synthetic_worker_registration.manifest)
    execution = copy.deepcopy(ledger["research_execution_manifest"])
    if case == "dataset": execution["dataset_identity_sha256"] = hid("wrong-dataset")
    if case == "category": execution["category_order"] = ["switch"]
    if case == "model": execution["model_state_sha256"] = hid("wrong-model")
    if case == "support": execution["categories"]["audiojack"]["supports"][0]["group"] = hid("wrong-support")
    if case == "event": execution["categories"]["audiojack"]["events"][0]["event"] = hid("wrong-event")
    if case == "seed": execution["query_seed"] += 1
    if case == "source": execution["source_commit"] = "0" * 40
    with pytest.raises(FS4BoundaryError):
        validate_execution_manifest(execution, ledger)


def test_research_mode_remains_blocked_without_label_blind_adapter():
    from experiments.calibration.fs4_r2_context import FS4BoundaryError
    from experiments.calibration.fs4_research_context import require_label_blind_dataset_adapter
    with pytest.raises(FS4BoundaryError, match="event paths expose undisclosed outcome"):
        require_label_blind_dataset_adapter()


def test_real_worker_holds_before_opening_configuration(tmp_path):
    from experiments.calibration.fs4_research_worker import ResearchWorkerError
    # A missing config would raise FileNotFoundError if opened before the hold.
    with pytest.raises(ResearchWorkerError, match="event paths expose undisclosed outcome"):
        ResearchWorker(tmp_path / "not-opened.json")


def test_research_resume_reencodes_registered_image(synthetic_worker_registration, monkeypatch):
    import experiments.calibration.fs4_research_worker as worker_module
    monkeypatch.setattr(worker_module, "_worker_main", _synthetic_spawn_entry)
    data = synthetic_worker_registration
    worker = ResearchWorker(data.config_path)
    try:
        worker.checkpoint("initial")
        worker.step(0)
    finally:
        worker.close()
    row = data.execution["categories"]["audiojack"]["events"][0]
    (data.root / "dataset" / Path(row["relative_path"])).write_bytes(b"changed synthetic image")
    data.config_path.write_bytes(canonical(dict(data.config, create=False, checkpoint="initial")))
    worker = ResearchWorker(data.config_path)
    from experiments.calibration.fs4_research_worker import ResearchWorkerError
    try:
        with pytest.raises(ResearchWorkerError, match="content"):
            worker.step(0)
    finally:
        worker.close()


def test_research_ledger_rejects_feature_source_mapping_before_seal(synthetic_worker_registration):
    from experiments.calibration.fs4_durable_ledger import LedgerIntegrityError, digest
    from experiments.calibration.fs4_research_ledger import ResearchLedger
    data = synthetic_worker_registration
    ledger = ResearchLedger(data.root / "run", data.manifest, create=True)
    try:
        ledger.select_category("audiojack")
        row = data.execution["categories"]["audiojack"]["events"][0]
        p = {"schema": "FS4_TRUSTED_EVENT_V1", "dataset_identity_sha256": data.manifest["dataset_identity_sha256"],
             "manifest_sha256": digest(data.execution), "category": "audiojack", "image_content_sha256": row["content_sha256"],
             "geometry": [row["width"], row["height"]], "model_state_sha256": data.manifest["model_state_sha256"],
             "feature_sha256": hid("wrong-features"), "cls_sha256": hid("cls"), "source_commit": data.manifest["source_commit"],
             "fixed_support_sha256": data.manifest["fixed_support_sha256_by_category"]["audiojack"]}
        ledger._active_provenance = p
        event = {"position": 0, "event": row["event"], "group": row["group"],
                 "feature_sha256": hid("registered-features"), "trusted_event_provenance": p}
        with pytest.raises(LedgerIntegrityError, match="producer or identity"):
            ledger.seal_prediction(event)
        assert ledger.position == 0 and ledger.phase == "READY"
    finally:
        ledger.close()


@pytest.mark.parametrize("case, message", [("wrong_bank", "fixed bank"), ("wrong_model", "model state")])
def test_worker_rejects_actual_model_and_support_bank_mismatch(synthetic_worker_registration, monkeypatch, case, message):
    import experiments.calibration.fs4_research_worker as worker_module
    from experiments.calibration.fs4_research_worker import ResearchWorkerError
    monkeypatch.setattr(worker_module, "_worker_main", _synthetic_spawn_entry)
    data = synthetic_worker_registration
    (data.root / case).write_text("synthetic-only", encoding="utf-8")
    with pytest.raises(ResearchWorkerError, match=message):
        ResearchWorker(data.config_path)
