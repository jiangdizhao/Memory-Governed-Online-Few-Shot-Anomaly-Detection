from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.calibration.fs4_durable_ledger import (DurableLedger, LedgerIntegrityError,
    SyntheticTestLedger, digest)
from experiments.calibration.fs4_durable_runtime import DurableExecutor, SyntheticTestExecutor
from experiments.calibration.fs4_r2_context import (FS4BoundaryError, ProductionContext, _read_registered_image,
    validate_execution_manifest, verify_model_identity, verify_native_runtime)
from experiments.calibration.fs4_r2_security import (FS4AccessControlError, WindowsPrivateStore,
    _sid_string, _windows, current_user_sid, validate_private_acl)
from experiments.calibration.fs4_r2_trust import FS4TrustError, canonical, verify_signed


def hid(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sign(private: Ed25519PrivateKey, role: str, payload: dict) -> dict:
    signature = private.sign(f"FS4-R2:{role}\0".encode() + canonical(payload))
    return {"payload": payload, "signature": base64.b64encode(signature).decode()}


@pytest.fixture
def authorities(monkeypatch):
    from experiments.calibration import fs4_r2_trust as trust
    keys = {role: Ed25519PrivateKey.generate() for role in trust.ROLES}
    public = {role: key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
              for role, key in keys.items()}
    # The production API itself never accepts a caller verifier or key; this
    # fixture simulates the separately provisioned OS-protected authority file.
    monkeypatch.setattr(trust, "load_production_authorities", lambda: public)
    return keys, public


def make_images(root: Path):
    from experiments.calibration.fs4_stochastic_bea import action_uniform, candidate_ordinal
    category = "audiojack"
    supports, events = [], []
    for serial in range(1, 6):
        count = 1 if serial <= 4 else 5
        for camera in range(1, count + 1):
            relative = f"{category}/OK/S{serial:04d}/{category}_{serial:04d}_OK_C{camera}_20231021131324.jpg"
            path = root / Path(relative); path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", (16, 12), (serial * 23, camera * 13, 7))
            buffer = io.BytesIO(); image.save(buffer, format="PNG"); data = buffer.getvalue(); path.write_bytes(data)
            group = hid(f"official-sample-group\0{category}_{serial:04d}_OK")
            record = {"group": group, "relative_path": relative, "content_sha256": hashlib.sha256(data).hexdigest(),
                      "width": 16, "height": 12}
            if serial <= 4:
                supports.append(record)
            else:
                trial = 0
                event = hid(f"synthetic-event-{camera}-{trial}")
                # Select a deterministic synthetic identity that exercises an
                # actual legal query even when the exploration gate is active.
                while camera == candidate_ordinal(37, group) and action_uniform(37, group, event) >= .05:
                    trial += 1; event = hid(f"synthetic-event-{camera}-{trial}")
                events.append({**record, "event": event, "position": camera - 1})
    return supports, events


def synthetic_registration(root: Path):
    from experiments.calibration.profiles import NativeScorerConfig
    from test_fs4_hardening_r1 import manifest
    supports, events = make_images(root)
    streams = {"audiojack": [(row["group"], row["event"]) for row in events]}
    ledger = manifest(streams)
    categories = {"audiojack": {"supports": supports, "events": events}}
    dataset = {"dataset_root": str(root), "category_order": ["audiojack"], "categories": categories}
    payload = {"schema": "FS4_TRUSTED_EXECUTION_V1", "run_id": ledger["run_id"], "protocol": "FS4_SYNTHETIC",
        "source_commit": ledger["source_commit"], "query_seed": ledger["query_seed"],
        "model_state_sha256": hid("test-model"), "checkpoint_sha256": hid("test-checkpoint"),
        "hub_source_tree_sha256": hid("test-hub"), "scorer_config_sha256": NativeScorerConfig().sha256,
        "preflight_receipt_sha256": hid("test-preflight"), "dataset_root": str(root),
        "dataset_identity_sha256": digest(dataset), "category_order": ["audiojack"], "categories": categories}
    ledger.update({"protocol": payload["protocol"], "dataset_identity_sha256": payload["dataset_identity_sha256"],
        "model_state_sha256": payload["model_state_sha256"],
        "support_groups_by_category": {"audiojack": [item["group"] for item in supports]}})
    return ledger, payload


def test_signed_documents_reject_wrong_role_and_modified_contents(authorities):
    keys, public = authorities
    doc = sign(keys["stage_b_grant"], "stage_b_grant", {"run_id": "a", "scope": "x"})
    assert verify_signed(doc, role="stage_b_grant", authorities=public)["run_id"] == "a"
    with pytest.raises(FS4TrustError): verify_signed(doc, role="stage_b_result", authorities=public)
    doc["payload"]["run_id"] = "b"
    with pytest.raises(FS4TrustError): verify_signed(doc, role="stage_b_grant", authorities=public)


def test_production_trust_anchor_has_no_caller_owned_fallback(tmp_path, monkeypatch):
    from experiments.calibration import fs4_r2_trust as trust
    store = WindowsPrivateStore(tmp_path / "caller_owned", create=True)
    anchor = store.write("authorities.json", b"{}")
    with pytest.raises(FS4AccessControlError, match="owned by System"):
        trust._validate_anchor_object(anchor)
    monkeypatch.setattr(trust, "AUTHORITY_FILE", tmp_path / "absent" / "authorities.json")
    with pytest.raises(FS4TrustError, match="authority file is absent"):
        trust.load_production_authorities()


def test_windows_private_acl_rejects_untrusted_principal(tmp_path):
    if __import__("os").name != "nt": pytest.skip("Windows ACL engineering check")
    root = tmp_path / "private"; store = WindowsPrivateStore(root, create=True)
    protected_file = store.write("feedback.bin", b"synthetic")
    assert store.read("feedback.bin") == b"synthetic"
    with pytest.raises(FS4AccessControlError): store.write("feedback.bin:alternate", b"x")
    # Authz evaluates the actual Windows descriptor for the World SID without
    # requiring a second account or disclosing data to a real user.
    import ctypes
    from ctypes import wintypes
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    adv.GetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    adv.GetFileSecurityW.restype = wintypes.BOOL
    adv.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    adv.ConvertStringSidToSidW.restype = wintypes.BOOL
    needed = wintypes.DWORD(); adv.GetFileSecurityW(str(protected_file), 5, None, 0, ctypes.byref(needed))
    descriptor = ctypes.create_string_buffer(needed.value)
    assert adv.GetFileSecurityW(str(protected_file), 5, descriptor, needed, ctypes.byref(needed))
    sid = ctypes.c_void_p(); assert adv.ConvertStringSidToSidW("S-1-1-0", ctypes.byref(sid))
    class Luid(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]
    class Request(ctypes.Structure):
        _fields_ = [("DesiredAccess", wintypes.DWORD), ("PrincipalSelfSid", ctypes.c_void_p),
                    ("ObjectTypeList", ctypes.c_void_p), ("ObjectTypeListLength", wintypes.DWORD),
                    ("OptionalArguments", ctypes.c_void_p)]
    class Reply(ctypes.Structure):
        _fields_ = [("ResultListLength", wintypes.DWORD), ("GrantedAccessMask", ctypes.POINTER(wintypes.DWORD)),
                    ("SaclEvaluationResults", ctypes.POINTER(wintypes.DWORD)), ("Error", ctypes.POINTER(wintypes.DWORD))]
    authz = ctypes.WinDLL("authz", use_last_error=True)
    authz.AuthzInitializeResourceManager.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                     wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    authz.AuthzInitializeResourceManager.restype = wintypes.BOOL
    authz.AuthzInitializeContextFromSid.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                    Luid, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    authz.AuthzInitializeContextFromSid.restype = wintypes.BOOL
    authz.AuthzAccessCheck.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(Request), ctypes.c_void_p,
                                      ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Reply), ctypes.c_void_p]
    authz.AuthzAccessCheck.restype = wintypes.BOOL
    rm, context = ctypes.c_void_p(), ctypes.c_void_p()
    assert authz.AuthzInitializeResourceManager(1, None, None, None, None, ctypes.byref(rm))
    assert authz.AuthzInitializeContextFromSid(2, sid, rm, None, Luid(0, 0), None, ctypes.byref(context))
    granted, error = wintypes.DWORD(), wintypes.DWORD()
    request = Request(1, None, None, 0, None)
    reply = Reply(1, ctypes.pointer(granted), None, ctypes.pointer(error))
    assert authz.AuthzAccessCheck(0, context, ctypes.byref(request), None, descriptor, None, 0, ctypes.byref(reply), None)
    assert granted.value == 0 and error.value == 5  # ERROR_ACCESS_DENIED
    authz.AuthzFreeContext(context); authz.AuthzFreeResourceManager(rm)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]; kernel.LocalFree.restype = ctypes.c_void_p
    kernel.LocalFree(sid)
    _api, _con, security, nt = _windows()
    sd = security.GetNamedSecurityInfo(str(root), security.SE_FILE_OBJECT, security.DACL_SECURITY_INFORMATION)
    acl = sd.GetSecurityDescriptorDacl()
    everyone = security.CreateWellKnownSid(security.WinWorldSid, None)
    acl.AddAccessAllowedAce(security.ACL_REVISION, nt.FILE_GENERIC_READ, everyone)
    security.SetNamedSecurityInfo(str(root), security.SE_FILE_OBJECT,
        security.DACL_SECURITY_INFORMATION | security.PROTECTED_DACL_SECURITY_INFORMATION, None, None, acl, None)
    with pytest.raises(FS4AccessControlError): validate_private_acl(root, directory=True)
    with pytest.raises(FS4AccessControlError): store.read("feedback.bin")


@pytest.mark.parametrize("case", ["dataset", "category", "event", "source", "seed", "support", "extra_label"])
def test_registered_manifest_rejects_identity_mismatch(tmp_path, case):
    ledger, payload = synthetic_registration(tmp_path / "dataset")
    data = json.loads(json.dumps(payload))
    if case == "dataset": data["dataset_identity_sha256"] = hid("other-dataset")
    if case == "category": data["categories"] = {"switch": data["categories"]["audiojack"]}
    if case == "event": data["categories"]["audiojack"]["events"][0]["event"] = hid("wrong-event")
    if case == "source": data["source_commit"] = "0" * 40
    if case == "seed": data["query_seed"] += 1
    if case == "support": data["categories"]["audiojack"]["supports"][0]["group"] = hid("wrong-support")
    if case == "extra_label": data["categories"]["audiojack"]["events"][0]["label"] = 0
    with pytest.raises(FS4BoundaryError): validate_execution_manifest(data, ledger)


def test_synthetic_registered_image_rejects_wrong_group_content_and_geometry(tmp_path):
    root = tmp_path / "dataset"; ledger, payload = synthetic_registration(root)
    validate_execution_manifest(payload, ledger)
    event = payload["categories"]["audiojack"]["events"][0]
    _, image = _read_registered_image(root, event, category="audiojack")
    assert image.size == (16, 12)
    with pytest.raises(FS4BoundaryError): _read_registered_image(root, {**event, "group": hid("wrong")}, category="audiojack")
    with pytest.raises(FS4BoundaryError): _read_registered_image(root, {**event, "width": 17}, category="audiojack")
    (root / Path(event["relative_path"])).write_bytes(b"not the registered image")
    with pytest.raises(FS4BoundaryError, match="content"):
        _read_registered_image(root, event, category="audiojack")


def test_wrong_model_and_runtime_are_rejected():
    from test_fs4_threshold import native_bank_and_writer
    from experiments.calibration.native import _fixed_bank_digest
    from experiments.calibration.fs4_threshold_integration import NativeRuntime
    from experiments.calibration import r1_matched_correction_pilot as r1
    model = torch.nn.Linear(2, 2).eval()
    backbone = SimpleNamespace(model=model)
    encode = lambda *_: None
    impl = SimpleNamespace(encode_pil=encode)
    expected = r1._model_state_sha256(model)
    verify_model_identity(backbone, impl, encode, expected)
    with torch.no_grad(): model.weight[0, 0] += 1
    with pytest.raises(FS4BoundaryError, match="model state"): verify_model_identity(backbone, impl, encode, expected)
    native, bank, writer = native_bank_and_writer(64)
    ref = {"fixed_support_sha256": _fixed_bank_digest(bank), "writer_reference": {"theta": .4, "sigma": .02}}
    cfg = SimpleNamespace(topk_patches=32, image_aggregation="max", image_topk=64, image_quantile=.95)
    runtime = NativeRuntime(impl=native, bank=bank, cfg=cfg, writer=writer,
        correction_cfg=bank.corr_cfg, correction_cb=bank.corr_cb["p0"]["id"][0], reference=ref)
    verify_native_runtime(runtime, runtime, runtime.fixed_sha)
    with pytest.raises(FS4BoundaryError, match="unrelated"): verify_native_runtime(object(), runtime, runtime.fixed_sha)
    with pytest.raises(FS4BoundaryError, match="fixed bank"): verify_native_runtime(runtime, runtime, hid("wrong-bank"))


def test_production_executor_rejects_caller_runtime_and_tensors():
    assert SyntheticTestLedger is not DurableLedger
    with pytest.raises(ValueError, match="trusted context"):
        DurableExecutor(context=object(), ledger=object(), label_reader=lambda *_: 0)
    # Its public interface has no feature, CLS, image or runtime arguments.
    import inspect
    assert list(inspect.signature(DurableExecutor.step).parameters) == ["self", "position"]
    with pytest.raises(TypeError): DurableExecutor.step(object(), 0, features=torch.zeros(1))
    with pytest.raises(TypeError): DurableExecutor.step(object(), 0, cls_feature=np.zeros(1))


def test_sealed_feature_and_cls_swaps_fail_before_feedback():
    from experiments.calibration.fs4_errors import FS4FatalIntegrityError
    from experiments.calibration.fs4_source_excluded_scorer import _feature_digest
    from experiments.calibration.fs4_threshold_integration import _validate_bound_inputs
    image = SimpleNamespace(width=16, height=12)
    features = torch.zeros((784, 4096)); features[:, 0] = 1
    other = torch.zeros((784, 4096)); other[:, 1] = 1
    cls = np.asarray([1., 0.], dtype=np.float32)
    provenance = {"feature_sha256": _feature_digest(features), "cls_sha256": hashlib.sha256(cls.tobytes()).hexdigest(),
                  "geometry": [16, 12]}
    _validate_bound_inputs(provenance, image, features, cls)
    with pytest.raises(FS4FatalIntegrityError): _validate_bound_inputs(provenance, image, other, cls)
    with pytest.raises(FS4FatalIntegrityError): _validate_bound_inputs(provenance, image, features, np.asarray([0., 1.], dtype=np.float32))


def test_production_stage_b_methods_never_accept_callback():
    import inspect
    assert "verifier" not in inspect.signature(DurableLedger.authorize_stage_b).parameters
    with pytest.raises(TypeError): DurableLedger.authorize_stage_b(object(), {}, verifier=lambda _: True)


def production_manifest(ledger, execution, keys):
    data = json.loads(json.dumps(ledger))
    data.update({"execution_scope": "PRODUCTION", "protocol": execution["protocol"],
        "stage_b_scope": "SYNTHETIC_AGGREGATE", "stage_a_principal_sid": _sid_string(current_user_sid()),
        "r2_execution_manifest": sign(keys["experiment_manifest"], "experiment_manifest", execution)})
    authorization = {"run_id": data["run_id"], "protocol": data["protocol"], "source_commit": data["source_commit"],
        "ledger_manifest_sha256": digest(data), "dataset_identity_sha256": data["dataset_identity_sha256"],
        "stream_sha256_by_category": data["stream_sha256_by_category"], "scope": "FS4_STAGE_A"}
    data["r2_stage_a_authorization"] = sign(keys["experiment_manifest"], "experiment_manifest", authorization)
    return data


def test_signed_stage_b_requires_verified_result_and_detects_corruption(tmp_path, authorities):
    from experiments.calibration.fs4_r2_stage_b import StageBIsolationError, StageBProjectionStore
    from experiments.calibration.fs4_source_excluded_scorer import _feature_digest
    from experiments.calibration.fs4_stochastic_bea import StochasticBEAGroupSlot
    from experiments.calibration.fs4_threshold_integration import CoverageMemory, SharedTrajectory, FEATURE_CONVENTION
    from experiments.calibration.fs4_threshold_synthetic_smoke import _SyntheticRuntime
    keys, _ = authorities
    raw, execution = synthetic_registration(tmp_path / "dataset")
    m = production_manifest(raw, execution, keys)
    with pytest.raises(LedgerIntegrityError, match="unregistered fields"):
        DurableLedger(tmp_path / "invalid_run", {**m, "hidden_histogram": {"x": 1}}, create=True)
    ledger = DurableLedger(tmp_path / "private_run", m, create=True); ledger.select_category("audiojack")
    # Seed an engineering-only, signed-manifest synthetic Stage A to exercise
    # Stage B. Production inference is tested separately through its boundary.
    ledger._reading_existing = True
    runtime = _SyntheticRuntime(.5); policy = StochasticBEAGroupSlot()
    loop = SharedTrajectory(runtime=runtime, coverage=CoverageMemory(np.eye(5, dtype=np.float32)[:4]), theta0=.5,
        policy=policy, seal_prediction=ledger.seal_prediction, seal_query=ledger.seal_query,
        label_provider=lambda event, position: ledger.disclose(event, position, lambda *_: 0))
    features = torch.zeros((784, 4096)); cls = np.eye(5, dtype=np.float32)[4]
    for event in execution["categories"]["audiojack"]["events"]:
        provenance = {"schema": "FS4_TRUSTED_EVENT_V1", "dataset_identity_sha256": execution["dataset_identity_sha256"],
            "manifest_sha256": digest(execution), "category": "audiojack", "image_content_sha256": event["content_sha256"],
            "geometry": [16, 12], "model_state_sha256": execution["model_state_sha256"],
            "cls_sha256": hashlib.sha256(cls.tobytes()).hexdigest(), "feature_sha256": _feature_digest(features)}
        row = loop.step(position=event["position"], group=event["group"], event=event["event"],
            image=SimpleNamespace(width=16, height=12), features=features, cls_feature=cls,
            feature_convention=FEATURE_CONVENTION, event_provenance=provenance)
        ledger.complete_event(position=event["position"], post_state_digest=hid("state"), transition={"profile": row["mode"]})
    policy.validate_complete_stream(); ledger.seal_category("audiojack"); ledger.seal_global_stage_a()
    ledger._reading_existing = False
    stage_a = ledger.last_hash
    grant = {"authorization_id": "synthetic-grant", "run_id": m["run_id"], "protocol": m["protocol"],
        "source_commit": m["source_commit"], "manifest_sha256": digest(m),
        "dataset_identity_sha256": m["dataset_identity_sha256"],
        "stream_sha256_by_category": m["stream_sha256_by_category"], "stage_a_sha256": stage_a,
        "scope": m["stage_b_scope"], "stage_a_principal_sid": m["stage_a_principal_sid"],
        "stage_b_principal_sid": "S-1-5-21-1-2-3-1999", "projection_root": str(tmp_path / "stage_b_private")}
    bad = sign(keys["stage_b_grant"], "stage_b_grant", {**grant, "stage_a_sha256": hid("wrong")})
    with pytest.raises(LedgerIntegrityError, match="global seal"): ledger.authorize_stage_b(bad)
    signed_grant = sign(keys["stage_b_grant"], "stage_b_grant", grant)
    with pytest.raises(StageBIsolationError, match="distinct authorized principal"):
        StageBProjectionStore(grant=signed_grant, manifest=m, global_a_sha256=stage_a, create=True)
    ledger.authorize_stage_b(signed_grant)
    with pytest.raises(LedgerIntegrityError, match="complete seal"): ledger.seal_complete()
    result = {"schema": "FS4_STAGE_B_AGGREGATE_V1", "run_id": m["run_id"], "stage_a_sha256": stage_a,
              "scope": m["stage_b_scope"], "metrics": {"synthetic": .25}}
    data = canonical(result); store = ledger.stage_b_result_store(); path = store.write("result.json", data)
    receipt = {"run_id": m["run_id"], "protocol": m["protocol"], "source_commit": m["source_commit"],
        "manifest_sha256": digest(m), "stage_a_sha256": stage_a, "grant_sha256": digest(signed_grant),
        "scope": m["stage_b_scope"], "status": "EVALUATION_COMPLETE", "artifact_type": "AGGREGATE_METRICS_ONLY",
        "stage_b_principal_sid": grant["stage_b_principal_sid"], "result_file": "result.json",
        "result_size": len(data), "result_sha256": hashlib.sha256(data).hexdigest()}
    fake = sign(keys["stage_b_grant"], "stage_b_grant", receipt)
    with pytest.raises(FS4TrustError): ledger.record_stage_b_result(fake)
    signed_result = sign(keys["stage_b_result"], "stage_b_result", receipt)
    ledger.record_stage_b_result(signed_result); ledger.seal_stage_b(); ledger.seal_complete(); ledger.close()
    path.write_bytes(b"tampered")
    with pytest.raises(LedgerIntegrityError, match="checksum"):
        DurableLedger(tmp_path / "private_run", m, create=False)


@pytest.fixture
def synthetic_production(tmp_path, monkeypatch, authorities):
    """Registered synthetic images and model; no real dataset or weights."""
    from experiments import p0_protocol_audit as p0
    from experiments.calibration import fs4_r2_context as context_module
    from experiments.calibration import profiles, r1_matched_correction_pilot as r1, r3_fullstream_2x2 as r3
    from experiments.calibration.stage05 import opaque_digest
    from experiments.risk_safe.preflight import parse_image_name
    from experiments.calibration.fs4_threshold_integration import CoverageMemory, NativeRuntime
    from experiments.calibration.native import _fixed_bank_digest
    from test_fs4_threshold import native_bank_and_writer
    keys, _ = authorities
    dataset_root = tmp_path / "dataset"
    ledger_manifest, execution = synthetic_registration(dataset_root)
    native, template_bank, _ = native_bank_and_writer(4096)
    fixed_sha = _fixed_bank_digest(template_bank)
    ledger_manifest["fixed_support_sha256_by_category"]["audiojack"] = fixed_sha

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
            raise AssertionError("synthetic encode_pil is used")

    model_hash = r1._model_state_sha256(TinyBackbone("cpu").model)
    ledger_manifest["model_state_sha256"] = execution["model_state_sha256"] = model_hash
    profile = profiles.NativeScorerConfig()
    fields = ("augmentation_seed", "bank_seed", "fixed_capacity", "fixed_per_image", "augmentations",
              "include_identity", "multiplier", "feature_fuse", "topk_patches", "image_aggregation",
              "image_topk", "image_quantile", "knn")
    cfg = p0.Config(device="cpu", **{key: getattr(profile, key) for key in fields})

    def synthetic_encode(_backbone, image, _mode):
        value = image.getpixel((0, 0))[0]
        features = torch.zeros((784, 4096), dtype=torch.float32)
        features[:, 1 + value % 32] = 1
        return [features], torch.tensor([1.0, value / 255.0], dtype=torch.float32)

    monkeypatch.setattr(native, "DINOv2MultiLayerBackbone", TinyBackbone)
    monkeypatch.setattr(native, "encode_pil", synthetic_encode)
    monkeypatch.setattr(native, "make_support_refs", lambda image, *_a, **_k: [image] * 4)
    monkeypatch.setattr(p0, "require_sources", lambda: native)
    monkeypatch.setattr(r3, "_get_verified_model_receipt", lambda: {})
    monkeypatch.setattr(r3, "_runtime_and_native_setup", lambda *_a: (lambda: None, {}))
    monkeypatch.setattr(r3, "_native_config", lambda *_a: (cfg, profile))

    def synthetic_builder(*, impl, cfg, support_cls, **_kwargs):
        _native, bank, writer = native_bank_and_writer(4096)
        reference = {"fixed_support_sha256": _fixed_bank_digest(bank), "writer_reference": {"theta": .5, "sigma": .02}, "theta": .5}
        runtime = NativeRuntime(impl=impl, bank=bank, cfg=cfg, writer=writer, correction_cfg=bank.corr_cfg,
            correction_cb=bank.corr_cb["p0"]["id"][0], reference=reference)
        return runtime, reference, CoverageMemory(support_cls)
    monkeypatch.setattr(context_module, "build_native_from_four_support", synthetic_builder)
    support_paths = [dataset_root / Path(s["relative_path"]) for s in execution["categories"]["audiojack"]["supports"]]
    parsed_supports = [parse_image_name(p, expected_class="audiojack") for p in support_paths]
    preflight = {"status": "PASS", "loaded_model_state_sha256": model_hash,
        "model": {"checkpoint_sha256": execution["checkpoint_sha256"], "hub_source_tree_sha256": execution["hub_source_tree_sha256"]},
        "categories": {"audiojack": {"status": "PASS", "fixed_evidence": {"actual_structured_sha256": fixed_sha},
            "support_groups_sha256": opaque_digest("support-groups", [p.official_sample_key for p in parsed_supports]),
            "support_images_sha256": opaque_digest("support-images", [p.rel_path for p in parsed_supports]),
            "support_image_content_sha256": [s["content_sha256"] for s in execution["categories"]["audiojack"]["supports"]]}}}
    receipt_path = tmp_path / "synthetic_preflight.json"; receipt_path.write_bytes(canonical(preflight))
    monkeypatch.setattr(context_module, "PREFLIGHT", receipt_path)
    execution["preflight_receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    manifest_value = production_manifest(ledger_manifest, execution, keys)
    journal = DurableLedger(tmp_path / "protected_run", manifest_value, create=True)
    journal.select_category("audiojack")
    yield SimpleNamespace(journal=journal, execution=execution, manifest=manifest_value,
                          dataset=dataset_root, cfg=cfg, model_hash=model_hash)
    journal.close()


def test_production_boundary_synthetic_producer_and_replay(synthetic_production):
    with pytest.raises(ValueError, match="synthetic executor cannot write"):
        SyntheticTestExecutor(object(), synthetic_production.journal, lambda *_: 0)
    context = ProductionContext(ledger=synthetic_production.journal, category="audiojack")
    executor = DurableExecutor(context=context, ledger=synthetic_production.journal, label_reader=lambda *_: 0)
    executor.checkpoint("initial")
    with pytest.raises(TypeError): executor.step(0, features=torch.zeros((784, 4096)))
    with pytest.raises(TypeError): executor.step(0, cls_feature=np.zeros(2))
    with pytest.raises(FS4BoundaryError, match="next registered"):
        executor.step(1)
    row = executor.step(0)
    provenance = row["sealed_prediction"]["trusted_event_provenance"]
    assert provenance["model_state_sha256"] == synthetic_production.model_hash
    assert provenance["feature_sha256"] == row["sealed_prediction"]["feature_sha256"]
    assert len(provenance["cls_sha256"]) == 64
    journal = synthetic_production.journal
    journal._bind_context(context)
    rebuilt = ProductionContext(ledger=journal, category="audiojack")
    resumed = DurableExecutor(context=rebuilt, ledger=journal, label_reader=lambda *_: 0, checkpoint_name="initial")
    replayed = resumed.step(0)
    assert replayed["sealed_prediction"] == row["sealed_prediction"]
    assert replayed["query"] == row["query"]


def test_recovery_rejects_changed_synthetic_dataset_and_model(synthetic_production):
    journal = synthetic_production.journal
    context = ProductionContext(ledger=journal, category="audiojack")
    executor = DurableExecutor(context=context, ledger=journal, label_reader=lambda *_: 0)
    executor.checkpoint("initial"); executor.step(0)
    record = synthetic_production.execution["categories"]["audiojack"]["events"][0]
    path = synthetic_production.dataset / Path(record["relative_path"])
    path.write_bytes(b"swapped image content")
    rebuilt = ProductionContext(ledger=journal, category="audiojack")
    resumed = DurableExecutor(context=rebuilt, ledger=journal, label_reader=lambda *_: 0, checkpoint_name="initial")
    with pytest.raises(FS4BoundaryError, match="content"):
        resumed.step(0)
    with torch.no_grad(): rebuilt._backbone.model.weight[0, 0] += 1
    with pytest.raises(FS4BoundaryError, match="model state"):
        resumed.step(0)


def test_production_context_rejects_wrong_category_runtime_and_seed(synthetic_production):
    journal = synthetic_production.journal
    with pytest.raises(FS4BoundaryError, match="active production ledger"):
        ProductionContext(ledger=journal, category="switch")
    context = ProductionContext(ledger=journal, category="audiojack")
    executor = DurableExecutor(context=context, ledger=journal, label_reader=lambda *_: 0)
    original = context._trajectory.runtime
    context._trajectory.runtime = object()
    with pytest.raises(FS4BoundaryError, match="unrelated native runtime"):
        executor.step(0)
    context._trajectory.runtime = original
    context._trajectory.policy.seed += 1
    with pytest.raises(FS4BoundaryError, match="query seed"):
        executor.step(0)
    context._trajectory.policy.seed -= 1
    context._backbone.resize_size += 1
    with pytest.raises(FS4BoundaryError, match="preprocessing"):
        executor.step(0)


def test_production_context_rejects_wrong_constructed_support_bank(synthetic_production, monkeypatch):
    from experiments.calibration import fs4_r2_context as context_module
    builder = context_module.build_native_from_four_support
    def wrong_bank(**kwargs):
        runtime, reference, coverage = builder(**kwargs)
        runtime.fixed_sha = hid("incorrect-fixed-bank")
        return runtime, reference, coverage
    monkeypatch.setattr(context_module, "build_native_from_four_support", wrong_bank)
    with pytest.raises(FS4BoundaryError, match="fixed bank"):
        ProductionContext(ledger=synthetic_production.journal, category="audiojack")
    assert synthetic_production.journal.position == 0


def test_queried_feedback_cannot_mutate_bound_model(synthetic_production):
    journal = synthetic_production.journal
    context = ProductionContext(ledger=journal, category="audiojack")
    def malicious_reader(*_args):
        with torch.no_grad(): context._backbone.model.weight[0, 0] += 1
        return 0
    executor = DurableExecutor(context=context, ledger=journal, label_reader=malicious_reader)
    for position in range(5):
        try:
            executor.step(position)
        except FS4BoundaryError as exc:
            assert "model state" in str(exc)
            assert journal.phase == "FEEDBACK"
            assert not any(row["kind"] == "EVENT_COMPLETE" and row["payload"]["position"] == position for row in journal.records)
            break
    else:
        pytest.fail("the registered synthetic fixture did not exercise a legal query")
