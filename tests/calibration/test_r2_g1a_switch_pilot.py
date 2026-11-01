from __future__ import annotations

import hashlib
import weakref
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.calibration.r2_l2a_g1_geometry import (
    DescriptorBatch,
    R1_DESCRIPTOR_PROFILE,
    compare_source_disjoint,
    g1a_spatial_indices,
)
from experiments.calibration.r2_l2b_role_certification import (
    AReferenceAvailabilityReceipt,
    IndependentReviewerAttestation,
    SourceGroupMember,
    _ed25519_sign,
    canonical_sha256,
    make_reviewed_identity_manifest,
    make_reviewer_attestation,
    make_reviewer_trust_directory,
    make_role_identity_receipt,
    source_group_digest_from_key,
)
from experiments.calibration.r2_g1a_switch_pilot import (
    G1AExecutionPermit,
    G1AImageIdentity,
    G1AImageIdentityAttestation,
    G1AImageIdentityMember,
    G1AImageIdentityReceipt,
    G1AOutputError,
    G1AAuthorizationError,
    G1APilotError,
    _G1A_PERMIT_SEAL,
    _canonical_json_bytes,
    apply_g1_resource_caps,
    authorization_payload,
    compare_g1a_projected_cohort,
    encode_authorized_image,
    make_execution_permit,
    project_r1_g1a_features,
    read_only_artifact_status,
    select_bounded_groups,
    validate_a_evidence_bundle,
    validate_authorization_scope,
    validate_projected_receipt,
    verify_file_sha256,
    verify_profile_source_hashes,
)
from experiments.calibration.r2_g1a_switch_pilot import (
    MAX_OUTPUT_BYTES,
    MIN_FREE_BYTES,
)
from experiments.calibration.r1_matched_correction_pilot import (
    OutputBudget,
    OutputBudgetError,
)
from experiments.risk_safe.preflight import parse_image_name


def _group_hex(index: int) -> str:
    return hashlib.sha256(f"group-{index}".encode()).hexdigest()


def _image_hex(path: str) -> str:
    identity = parse_image_name(path, expected_class="switch")
    return hashlib.sha256(("image-identity\0" + identity.image_id).encode()).hexdigest()


def _record(camera: int = 1) -> G1AImageIdentity:
    path = rf"G:\dataset\realiad\switch\OK\S0001\switch_0001_OK_C{camera}_20200101000000.jpg"
    identity = parse_image_name(path, expected_class="switch")
    return G1AImageIdentity(
        category_id="switch",
        role="A",
        population="authorized_normal_reference_only",
        source_group_sha256=source_group_digest_from_key("switch", identity.official_sample_key),
        image_sha256=_image_hex(path),
        camera_id=camera,
        image_path=path,
        train_normal_verified=True,
    )


def _unit_full_map() -> np.ndarray:
    value = np.zeros((784, 4096), dtype=np.float32)
    value[np.arange(784), np.arange(784)] = 1.0
    return value


def _permit(output_directory: str = r"E:\repo\outputs\calibration\r2_g1_pilot\r2_g1a_switch_test_01") -> G1AExecutionPermit:
    record = _record()
    return G1AExecutionPermit(
        authorization_id="synthetic-only",
        authorization_sha256="a" * 64,
        category_id="switch",
        a_reference_receipt_sha256="b" * 64,
        image_identity_receipt_sha256="9" * 64,
        selected_source_group_sha256s=(record.source_group_sha256,),
        authorized_image_identities=((record.image_sha256, record.source_group_sha256, 1),),
        s1_image_sha256s=(record.image_sha256,),
        representation_sha256=R1_DESCRIPTOR_PROFILE.representation_sha256,
        execution_commit="c" * 40,
        output_directory=output_directory,
        max_output_bytes=MAX_OUTPUT_BYTES,
        min_free_bytes=MIN_FREE_BYTES,
        max_wall_seconds=7200,
        max_cpu_threads=4,
        max_cuda_reserved_bytes=20 * 1024**3,
        _authorization_seal=_G1A_PERMIT_SEAL,
    )


def _scope_and_trust(tmp_path):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    private = Ed25519PrivateKey.generate()
    public_hex = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    trust = make_reviewer_trust_directory(
        directory_id="r2-g1a-switch-access-synthetic-test",
        trusted_policy_approver_public_keys={"researcher-test": public_hex},
        trusted_identity_source_sha256s=(),
    )
    repo = tmp_path / "repo"
    output_parent = repo / "outputs" / "calibration" / "r2_g1_pilot"
    output_parent.mkdir(parents=True)
    scope = {
        "schema": "r2_g1a.switch_image_feature_authorization.v1",
        "operation_id": "R2_G1A_SWITCH_TRAIN_NORMAL_GEOMETRY_V1",
        "authorization_id": "synthetic-g1a-auth",
        "researcher_id": "researcher-test",
        "decision": "APPROVE_G1A_SWITCH_PILOT",
        "dataset_root": r"G:\dataset\realiad",
        "metadata_root": r"G:\dataset\realiad\jsons\realiad_jsons",
        "category_id": "switch",
        "split": "TRAIN",
        "population": "authorized_normal_reference_only",
        "identity_contract": "realid-offical-sample-key-v2",
        "grouping_unit": "official_sample_key",
        "group_digest_algorithm": "sha256:official-sample-group-nul-prefix-utf8.v1",
        "allowed_train_fields": ["image_path", "anomaly_class", "official_sample_key"],
        "authorized_stages": ["S1", "S2"],
        "allow_train_identity_projection": True,
        "allow_image_decoding": True,
        "allow_dino_feature_extraction": True,
        "allow_projected_descriptor_output": True,
        "allow_projected_cache_reuse": True,
        "allow_g1a_geometry": True,
        "allow_test_metadata_parsing": False,
        "allow_test_labels": False,
        "allow_masks": False,
        "allow_support_selection": False,
        "allow_native_scoring": False,
        "allow_g1b": False,
        "allow_g2": False,
        "sample_policy": "r2-g1-bounded-pilot-v1",
        "sample_seed": 0,
        "stage1_group_count": 10,
        "maximum_source_groups": 100,
        "camera_id": 1,
        "sampling_projection_rule": "r1_grid28x28_cell_centers_7x7_v1",
        "representation_sha256": R1_DESCRIPTOR_PROFILE.representation_sha256,
        "profile_source_commit": R1_DESCRIPTOR_PROFILE.source_commit,
        "execution_commit": "c" * 40,
        "source_fingerprint_sha256": "d" * 64,
        "trust_directory_sha256": trust.directory_sha256,
        "a_reference_receipt_sha256": "e" * 64,
        "image_identity_receipt_sha256": "f" * 64,
        "checkpoint_sha256": R1_DESCRIPTOR_PROFILE.checkpoint_sha256,
        "hub_source_tree_sha256": R1_DESCRIPTOR_PROFILE.hub_source_tree_sha256,
        "expected_loaded_model_state_sha256": R1_DESCRIPTOR_PROFILE.loaded_model_state_sha256,
        "output_directory": str(output_parent / "r2_g1a_switch_test_01"),
        "maximum_output_bytes": MAX_OUTPUT_BYTES,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "maximum_wall_seconds": 7200,
        "cpu_thread_limit": 4,
        "cuda_device": "cuda:0",
        "maximum_cuda_reserved_bytes": 20 * 1024**3,
        "signature_ed25519_hex": "",
        "authorization_sha256": "",
    }
    payload = authorization_payload(scope)
    scope["authorization_sha256"] = canonical_sha256(payload)
    scope["signature_ed25519_hex"] = private.sign(_canonical_json_bytes(payload)).hex()
    return repo, scope, trust


def _signed_synthetic_a_image_bundle():
    """Build reviewer-signed A/image identity evidence without dataset records."""
    a_key = Ed25519PrivateKey.generate()
    inventory_key = Ed25519PrivateKey.generate()

    def public_hex(key):
        return key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()

    source_manifest_sha = "1" * 64
    groups = tuple(
        SourceGroupMember(hashlib.sha256(f"synthetic-group-{i}".encode()).hexdigest(), 1)
        for i in range(100)
    )
    role_receipt = make_role_identity_receipt(
        category_id="switch", role="A", source_manifest_sha256=source_manifest_sha,
        producer_id="synthetic-a-producer", group_members=groups,
    )
    role_manifest = make_reviewed_identity_manifest(
        role_receipt, population="authorized_normal_reference_only",
        normal_evidence_manifest_sha256="2" * 64,
    )
    role_attestation = make_reviewer_attestation(
        role_receipt, role_manifest, reviewer_id="synthetic-a-reviewer",
        decision_id="a-review-01", private_key=a_key,
    )
    availability = AReferenceAvailabilityReceipt(
        category_id="switch", authorized_normal_reference_image_count=100,
        authorized_source_group_count=100,
        receipt_sha256=canonical_sha256({"synthetic": "availability"}),
        authorization_id="synthetic-test-only",
    )
    image_members = tuple(
        G1AImageIdentityMember(
            image_sha256=hashlib.sha256(f"synthetic-image-{i}".encode()).hexdigest(),
            source_group_sha256=group.group_sha256,
            camera_id=1,
        )
        for i, group in enumerate(groups)
    )
    image_receipt = G1AImageIdentityReceipt(
        schema="r2_g1a.switch_a_image_identity_inventory.v1",
        category_id="switch", role="A",
        population="authorized_normal_reference_only",
        identity_contract="realid-offical-sample-key-v2",
        group_digest_algorithm="sha256:official-sample-group-nul-prefix-utf8.v1",
        source_manifest_sha256=source_manifest_sha,
        a_reference_receipt_sha256=availability.receipt_sha256,
        producer_id="synthetic-image-inventory-producer",
        image_members=image_members, image_count=len(image_members),
        contains_raw_sample_keys=False, contains_image_paths=False,
        contains_labels_or_masks=False, receipt_sha256="",
    )
    image_payload = asdict(image_receipt)
    image_payload.pop("receipt_sha256")
    image_receipt = replace(image_receipt, receipt_sha256=canonical_sha256(image_payload))
    image_attestation = G1AImageIdentityAttestation(
        reviewer_id="synthetic-inventory-reviewer", decision_id="inventory-review-01",
        decision="CERTIFY_G1A_IMAGE_IDENTITY_INVENTORY",
        image_identity_receipt_sha256=image_receipt.receipt_sha256,
        signature_ed25519_hex="", attestation_sha256="",
    )
    image_attestation_payload = asdict(image_attestation)
    image_attestation_payload.pop("signature_ed25519_hex")
    image_attestation_payload.pop("attestation_sha256")
    signature = _ed25519_sign(inventory_key, image_attestation_payload)
    image_attestation = replace(
        image_attestation, signature_ed25519_hex=signature,
        attestation_sha256=canonical_sha256({
            "payload": image_attestation_payload,
            "signature_ed25519_hex": signature,
        }),
    )
    trust = make_reviewer_trust_directory(
        directory_id="r2-g1a-switch-synthetic-test-root",
        trusted_reviewer_public_keys={
            "synthetic-a-reviewer": public_hex(a_key),
            "synthetic-inventory-reviewer": public_hex(inventory_key),
        },
        trusted_identity_source_sha256s=(source_manifest_sha,),
        reviewed_a_availability_receipt_sha256s=(availability.receipt_sha256,),
    )
    bundle = {
        "schema": "r2_g1a.switch_a_reference_evidence_bundle.v1",
        "availability_receipt": asdict(availability),
        "identity_receipt": asdict(role_receipt),
        "reviewed_manifest": asdict(role_manifest),
        "attestation": asdict(role_attestation),
        "image_identity_receipt": asdict(image_receipt),
        "image_identity_attestation": asdict(image_attestation),
    }
    return bundle, trust


def test_fixed49_indices_descriptor_shape_and_representation_receipt():
    expected = [row * 28 + col for row in (2, 6, 10, 14, 18, 22, 26)
                for col in (2, 6, 10, 14, 18, 22, 26)]
    assert g1a_spatial_indices().tolist() == expected
    projected, receipt = project_r1_g1a_features(_unit_full_map(), record=_record())
    assert projected.shape == (49, 4096)
    assert projected.dtype == np.float32
    assert receipt.representation_sha256 == R1_DESCRIPTOR_PROFILE.representation_sha256
    assert receipt.projection_rule == "r1_grid28x28_cell_centers_7x7_v1"
    assert receipt.shape == (49, 4096)
    np.testing.assert_array_equal(projected[:, expected], np.eye(49, dtype=np.float32))


def test_projected_receipt_binds_group_image_and_source_map_without_raw_identity():
    full = _unit_full_map()
    projected, receipt = project_r1_g1a_features(full, record=_record())
    validate_projected_receipt(receipt, projected, expected_record=_record())
    assert receipt.source_group_sha256 == _record().source_group_sha256
    assert receipt.image_sha256 == _record().image_sha256
    assert receipt.full_map_source_sha256
    assert not receipt.contains_raw_sample_keys
    assert not receipt.contains_image_paths
    assert not receipt.contains_labels_or_masks
    assert not receipt.contains_private_feedback
    assert not receipt.contains_correction_anchors
    changed = full.copy()
    changed[0, 0] = 0.0
    changed[0, 1000] = 1.0
    _, changed_receipt = project_r1_g1a_features(changed, record=_record())
    assert changed_receipt.full_map_source_sha256 != receipt.full_map_source_sha256
    wrong_image = _record()
    wrong_image = G1AImageIdentity(**{**wrong_image.__dict__, "image_sha256": "f" * 64})
    with pytest.raises(G1APilotError, match="incompatible"):
        validate_projected_receipt(receipt, projected, expected_record=wrong_image)


@pytest.mark.parametrize("mutation", ["shape", "finite", "norm", "dtype"])
def test_malformed_full_maps_fail_closed(mutation):
    full = _unit_full_map()
    if mutation == "shape":
        full = full[:783]
    elif mutation == "finite":
        full[0, 0] = np.nan
    elif mutation == "norm":
        full[0, :] = 0.0
    else:
        full = full.astype(np.float16)
    with pytest.raises(G1APilotError):
        project_r1_g1a_features(full, record=_record())


def test_group_sampling_is_deterministic_and_requires_full_100():
    candidates = [_group_hex(index) for index in range(160)]
    selected_a = select_bounded_groups(candidates)
    selected_b = select_bounded_groups(reversed(candidates))
    assert selected_a == selected_b
    assert len(selected_a) == 100
    assert set(selected_a).issubset(candidates)
    with pytest.raises(G1APilotError, match="smaller"):
        select_bounded_groups(candidates[:99])


def test_camera_rule_is_checked_before_loader_and_parser_uses_c1_token():
    identity = parse_image_name(_record().image_path, expected_class="switch")
    assert identity.camera_id == 1
    record = _record(camera=2)

    class NeverLoad:
        called = False

        @staticmethod
        def load_rgb(_path):
            NeverLoad.called = True
            raise AssertionError("must not open image")

    with pytest.raises(G1AAuthorizationError, match="camera"):
        encode_authorized_image(record, permit=_permit(), implementation=NeverLoad(), backbone=object())
    assert not NeverLoad.called


def test_no_authorization_refuses_before_any_image_loader_call():
    class NeverLoad:
        called = False

        @staticmethod
        def load_rgb(_path):
            NeverLoad.called = True
            raise AssertionError("must not open image")

    with pytest.raises(G1AAuthorizationError, match="no validated"):
        encode_authorized_image(_record(), permit=None, implementation=NeverLoad(), backbone=object())
    assert not NeverLoad.called


def test_constructed_but_unvalidated_permit_is_refused_before_loader():
    invalid = G1AExecutionPermit(**{**_permit().__dict__, "_authorization_seal": object()})

    class NeverLoad:
        called = False

        @staticmethod
        def load_rgb(_path):
            NeverLoad.called = True
            raise AssertionError("unvalidated permit must not open image")

    with pytest.raises(G1AAuthorizationError, match="permit issued by signed-scope validation"):
        encode_authorized_image(
            _record(), permit=invalid, implementation=NeverLoad(), backbone=object())
    assert not NeverLoad.called


def test_exact_r1_adapter_projects_one_image_and_releases_full_map():
    import torch

    full = torch.from_numpy(_unit_full_map())

    class FakeBackbone:
        def __init__(self):
            self.held = full
            self.clears = 0

        def _clear_feats(self):
            self.held = None
            self.clears += 1

    class FakeImage:
        closed = False

        def close(self):
            self.closed = True

    image = FakeImage()

    class FakeImplementation:
        ref = weakref.ref(full)

        @staticmethod
        def load_rgb(_path):
            return image

        @staticmethod
        def encode_pil(backbone, image_value, mode):
            assert image_value is image
            assert mode == "concat"
            return [backbone.held], None

    backbone = FakeBackbone()
    projected, receipt = encode_authorized_image(
        _record(), permit=_permit(), implementation=FakeImplementation(), backbone=backbone)
    assert projected.shape == (49, 4096)
    assert receipt.representation_sha256 == R1_DESCRIPTOR_PROFILE.representation_sha256
    assert image.closed and backbone.clears == 1
    del full
    gc = __import__("gc")
    gc.collect()
    assert FakeImplementation.ref() is None


def test_s1_adapter_feeds_fixed49_rows_to_exact_source_disjoint_geometry():
    records = []
    projected = []
    receipts = []
    for i in range(1, 11):
        path = rf"G:\dataset\realiad\switch\OK\S{i:04d}\switch_{i:04d}_OK_C1_20200101000000.jpg"
        identity = parse_image_name(path, expected_class="switch")
        record = G1AImageIdentity(
            category_id="switch", role="A",
            population="authorized_normal_reference_only",
            source_group_sha256=source_group_digest_from_key("switch", identity.official_sample_key),
            image_sha256=hashlib.sha256(
                ("image-identity\0" + identity.image_id).encode("utf-8")).hexdigest(),
            camera_id=1, image_path=path, train_normal_verified=True,
        )
        values, receipt = project_r1_g1a_features(_unit_full_map(), record=record)
        records.append(record)
        projected.append(values)
        receipts.append(receipt)
    permit = G1AExecutionPermit(**{
        **_permit().__dict__,
        "selected_source_group_sha256s": tuple(x.source_group_sha256 for x in records),
        "authorized_image_identities": tuple(
            (x.image_sha256, x.source_group_sha256, x.camera_id) for x in records),
        "s1_image_sha256s": tuple(x.image_sha256 for x in records),
    })
    stats, summary = compare_g1a_projected_cohort(
        records, projected, receipts, permit=permit, stage="S1")
    assert stats.category_id == "switch"
    assert summary["stage"] == "S1"
    assert summary["query_descriptor_count"] == 490
    assert summary["query_image_count"] == 10
    assert summary["independent_query_source_group_count"] == 10
    assert summary["excluded_comparison_count"] > 0


def test_scope_signature_and_pins_validate_and_mismatches_fail_closed(tmp_path):
    repo, scope, trust = _scope_and_trust(tmp_path)
    pins = dict(
        expected_trust_directory_sha256=trust.directory_sha256,
        expected_source_fingerprint_sha256="d" * 64,
        actual_source_fingerprint_sha256="d" * 64,
        expected_execution_commit="c" * 40,
        repo_root=repo,
    )
    assert validate_authorization_scope(scope, trust, **pins) == ()
    errors = validate_authorization_scope(
        scope, trust,
        **{**pins, "actual_source_fingerprint_sha256": "f" * 64},
    )
    assert any("source-fingerprint" in error for error in errors)
    assert validate_authorization_scope(
        None, None,
        expected_trust_directory_sha256=None,
        expected_source_fingerprint_sha256=None,
        actual_source_fingerprint_sha256=None,
        expected_execution_commit=None,
        repo_root=repo,
    ) == ("signed G1-A switch authorization is missing",)


@pytest.mark.parametrize("forbidden_permission", [
    "allow_test_metadata_parsing", "allow_test_labels", "allow_masks",
    "allow_support_selection", "allow_native_scoring", "allow_g1b", "allow_g2",
])
def test_scope_rejects_prohibited_permissions_and_wrong_model_fingerprint(
        tmp_path, forbidden_permission):
    repo, scope, trust = _scope_and_trust(tmp_path)
    scope[forbidden_permission] = True
    errors = validate_authorization_scope(
        scope, trust,
        expected_trust_directory_sha256=trust.directory_sha256,
        expected_source_fingerprint_sha256="d" * 64,
        actual_source_fingerprint_sha256="d" * 64,
        expected_execution_commit="c" * 40,
        repo_root=repo,
    )
    assert any(forbidden_permission in error for error in errors)
    scope[forbidden_permission] = False
    scope["checkpoint_sha256"] = "0" * 64
    errors = validate_authorization_scope(
        scope, trust,
        expected_trust_directory_sha256=trust.directory_sha256,
        expected_source_fingerprint_sha256="d" * 64,
        actual_source_fingerprint_sha256="d" * 64,
        expected_execution_commit="c" * 40,
        repo_root=repo,
    )
    assert any("checkpoint_sha256" in error for error in errors)


def test_missing_a_receipt_cannot_issue_execution_permit():
    assert validate_a_evidence_bundle(
        None, None, expected_trust_directory_sha256=None,
    ) == ("PENDING_A_REFERENCE_RECEIPT", None, None, ())
    with pytest.raises(G1AAuthorizationError, match="missing"):
        make_execution_permit(
            None, None, None,
            expected_trust_directory_sha256=None,
            expected_source_fingerprint_sha256=None,
            actual_source_fingerprint_sha256=None,
            expected_execution_commit=None,
            repo_root=Path.cwd(),
        )


def test_read_only_preflight_lists_exact_missing_private_artifacts(tmp_path):
    report = read_only_artifact_status(tmp_path / "not-created")
    assert report["status"] == "WAITING_RESEARCHER_AUTHORIZATION"
    assert report["image_access_performed"] is False
    assert report["dataset_access_performed"] is False
    assert [Path(value).name for value in report["missing_artifacts"]] == [
        "authorization.signed.json", "trust_directory.json", "trust_directory.sha256",
        "source_fingerprint.json", "source_fingerprint.sha256", "a_reference_bundle.json",
    ]


def test_a_image_inventory_requires_valid_source_binding_and_independent_signature():
    bundle, trust = _signed_synthetic_a_image_bundle()
    status, a_sha, image_sha, members = validate_a_evidence_bundle(
        bundle, trust, expected_trust_directory_sha256=trust.directory_sha256)
    assert status == "AUTHORIZED_FOR_G1_REFERENCE_POPULATION"
    assert a_sha == bundle["availability_receipt"]["receipt_sha256"]
    assert image_sha == bundle["image_identity_receipt"]["receipt_sha256"]
    assert len(members) == 100

    missing_attestation = dict(bundle)
    missing_attestation.pop("image_identity_attestation")
    status, *_ = validate_a_evidence_bundle(
        missing_attestation, trust,
        expected_trust_directory_sha256=trust.directory_sha256)
    assert status == "PENDING_A_IMAGE_IDENTITY_REVIEW"

    tampered = dict(bundle)
    tampered_image_receipt = dict(bundle["image_identity_receipt"])
    tampered_image_receipt["image_members"] = list(tampered_image_receipt["image_members"])
    tampered_image_receipt["image_members"][0] = {
        **tampered_image_receipt["image_members"][0], "camera_id": 2}
    tampered["image_identity_receipt"] = tampered_image_receipt
    status, *_ = validate_a_evidence_bundle(
        tampered, trust, expected_trust_directory_sha256=trust.directory_sha256)
    assert status == "PENDING_A_IMAGE_IDENTITY_REVIEW"


def test_image_not_in_reviewed_inventory_is_refused_before_loader():
    record = _record()
    permit = G1AExecutionPermit(
        **{**_permit().__dict__, "authorized_image_identities": (
            ("a" * 64, record.source_group_sha256, 1),)})

    class NeverLoad:
        called = False

        @staticmethod
        def load_rgb(_path):
            NeverLoad.called = True
            raise AssertionError("unreviewed image must not be opened")

    with pytest.raises(G1AAuthorizationError, match="image identity is absent"):
        encode_authorized_image(
            record, permit=permit, implementation=NeverLoad(), backbone=object())
    assert not NeverLoad.called


def test_s1_rejects_permitted_s2_image_outside_first_ten():
    permit = _permit()
    permit = G1AExecutionPermit(
        **{**permit.__dict__, "s1_image_sha256s": ()})

    class NeverLoad:
        called = False

        @staticmethod
        def load_rgb(_path):
            NeverLoad.called = True
            raise AssertionError("S1 exclusion must happen before image open")

    with pytest.raises(G1AAuthorizationError, match="first 10 ranked groups"):
        encode_authorized_image(
            _record(), permit=permit, implementation=NeverLoad(), backbone=object(), stage="S1")
    assert not NeverLoad.called


def test_checkpoint_verification_rejects_missing_and_incomplete_files(tmp_path):
    expected = hashlib.sha256(b"complete checkpoint").hexdigest()
    missing = tmp_path / "missing.pth"
    with pytest.raises(G1APilotError, match="missing"):
        verify_file_sha256(missing, expected)
    partial = tmp_path / "partial.pth"
    partial.write_bytes(b"incomplete")
    with pytest.raises(G1APilotError, match="mismatch"):
        verify_file_sha256(partial, expected)
    partial.write_bytes(b"complete checkpoint")
    assert verify_file_sha256(partial, expected) == expected


def test_resource_caps_are_applied_before_model_initialization():
    class FakeCuda:
        available = True
        fraction = None

        @classmethod
        def is_available(cls):
            return cls.available

        @classmethod
        def set_per_process_memory_fraction(cls, value, device):
            cls.fraction = (value, device)

    class FakeTorch:
        cuda = FakeCuda
        num_threads = None
        interop_threads = None

        @classmethod
        def set_num_threads(cls, value):
            cls.num_threads = value

        @classmethod
        def set_num_interop_threads(cls, value):
            cls.interop_threads = value

    total = 24_564 * 1024**2
    fraction = apply_g1_resource_caps(FakeTorch, total_cuda_bytes=total)
    assert FakeTorch.num_threads == 4
    assert FakeTorch.interop_threads == 1
    assert FakeCuda.fraction == (fraction, 0)
    with pytest.raises(G1APilotError, match="smaller"):
        apply_g1_resource_caps(FakeTorch, total_cuda_bytes=20 * 1024**3 - 1)


def test_frozen_r1_source_hashes_match_canonical_content():
    checked = verify_profile_source_hashes(Path(__file__).resolve().parents[2])
    assert len(checked) == 5
    assert all(item["canonical_lf_sha256"] for item in checked.values())


def test_output_budget_overflow_is_refused_before_file_creation(tmp_path):
    root = tmp_path / "g1_overflow"
    budget = OutputBudget(root, max_bytes=1024, failure_reserve_bytes=128)
    try:
        budget.ensure_root()
        (root / "events").mkdir()
        target = root / "events" / "event_000001.npz"
        with pytest.raises(OutputBudgetError, match="before I/O"):
            budget.write_once(target, b"x" * 897)
        assert not target.exists()
    finally:
        budget.close()


def test_existing_g1_output_path_is_never_reused(tmp_path):
    repo = tmp_path / "repo"
    parent = repo / "outputs" / "calibration" / "r2_g1_pilot"
    target = parent / "r2_g1a_switch_test_01"
    target.mkdir(parents=True)
    with pytest.raises(G1AOutputError, match="reuse"):
        from experiments.calibration.r2_g1a_switch_pilot import create_g1a_output_store
        create_g1a_output_store(
            _permit(str(target)), repo_root=repo,
            free_bytes=lambda _path: 20 * 1024**3,
        )


def test_new_output_store_writes_only_projected_sanitized_receipt(tmp_path):
    from experiments.calibration.r2_g1a_switch_pilot import create_g1a_output_store

    repo = tmp_path / "repo"
    parent = repo / "outputs" / "calibration" / "r2_g1_pilot"
    parent.mkdir(parents=True)
    target = parent / "r2_g1a_switch_test_02"
    store = create_g1a_output_store(
        _permit(str(target)), repo_root=repo,
        free_bytes=lambda _path: 20 * 1024**3,
    )
    try:
        descriptors, receipt = project_r1_g1a_features(_unit_full_map(), record=_record())
        shard = store.write_projected(
            1, descriptors, receipt, expected_record=_record())
        assert shard.is_file()
        with np.load(shard, allow_pickle=False) as data:
            assert set(data.files) == {"projected_descriptors", "sanitized_receipt"}
            assert data["projected_descriptors"].shape == (49, 4096)
            receipt_json = bytes(data["sanitized_receipt"].tolist()).decode("ascii")
            assert '"image_path":' not in receipt_json
            assert '"official_sample_key":' not in receipt_json
            assert "contains_labels_or_masks" in receipt_json
    finally:
        store.close()


def test_missing_or_incompatible_group_or_camera_record_does_not_open_path():
    class NeverLoad:
        called = False

        @staticmethod
        def load_rgb(_path):
            NeverLoad.called = True
            raise AssertionError("must not open image")

    bad = G1AImageIdentity(**{**_record().__dict__, "train_normal_verified": False})
    with pytest.raises(G1AAuthorizationError, match="TRAIN-normal"):
        encode_authorized_image(bad, permit=_permit(), implementation=NeverLoad(), backbone=object())
    assert not NeverLoad.called
