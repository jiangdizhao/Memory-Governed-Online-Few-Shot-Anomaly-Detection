from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.calibration import r2_l2c_metadata_census as l2c
from experiments.calibration.r2_l2a_allclass_inventory import (
    CANONICAL_CATEGORY_IDS,
)
from experiments.calibration.r2_l2b_role_certification import (
    SourceGroupMember,
    canonical_sha256,
    make_reviewer_attestation,
    make_reviewer_trust_directory,
    make_reviewed_identity_manifest,
    make_role_identity_receipt,
    source_group_digest_from_key,
)


def _public_key_hex(key: Ed25519PrivateKey) -> str:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()


def _signature_bytes(payload) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def make_authorization(tmp_path, *, categories=None, max_bytes=2_000_000, **overrides):
    dataset_root = tmp_path / "realiad"
    metadata_root = dataset_root / "jsons" / "realiad_jsons"
    metadata_root.mkdir(parents=True, exist_ok=True)
    output_directory = tmp_path / "r2_l2c_output_new"
    private_key = Ed25519PrivateKey.generate()
    researcher_id = "synthetic-researcher"
    trust = make_reviewer_trust_directory(
        directory_id="synthetic-r2-l2c-trust-only",
        trusted_policy_approver_public_keys={researcher_id: _public_key_hex(private_key)},
        trusted_identity_source_sha256s=(),
    )
    if categories is None:
        categories = tuple(c for c in CANONICAL_CATEGORY_IDS if c not in {"mint", "pcb"})
    payload = {
        "schema": l2c.AUTHORIZATION_SCHEMA,
        "authorization_id": "synthetic-local-approval-not-production",
        "researcher_id": researcher_id,
        "decision": l2c.TRAIN_CENSUS_OPERATION,
        "dataset_root": str(dataset_root),
        "metadata_root": str(metadata_root),
        "approved_categories": list(categories),
        "explicitly_authorized_protected_categories": [],
        "allow_physical_json_file_bytes": True,
        "allow_test_section_bytes_encountered": True,
        "allow_train_image_path": True,
        "allow_train_anomaly_class": True,
        "allow_official_sample_key_condition_tokens": True,
        "allow_meta_normal_class": True,
        "allow_meta_prefix": True,
        "allow_directory_serial_sensitivity": True,
        "allow_private_group_receipt_output": True,
        "allow_test_section_parsing": False,
        "allow_mask_access": False,
        "allow_feature_extraction": False,
        "allow_image_decoding": False,
        "allow_support_c64_selection": False,
        "approved_output_fields": sorted(l2c.REQUIRED_PUBLIC_OUTPUT_FIELDS),
        "approved_private_receipt_fields": sorted(l2c.REQUIRED_PRIVATE_RECEIPT_FIELDS),
        "output_directory": str(output_directory),
        "maximum_output_bytes": max_bytes,
    }
    payload.update(overrides)
    digest = canonical_sha256(payload)
    signature = private_key.sign(_signature_bytes(payload)).hex()
    manifest = dict(payload)
    manifest["signature_ed25519_hex"] = signature
    manifest["authorization_sha256"] = digest
    authorization = l2c.authorization_from_mapping(manifest)
    return authorization, trust, trust.directory_sha256, dataset_root, metadata_root, output_directory


def image_record(category, sample=1, camera=1, *, state="OK", defect=None, label="OK", timestamp=None):
    serial = f"S{sample:04d}"
    timestamp = timestamp or f"202611010000{camera:02d}"
    defect_part = f"_{defect}" if defect else ""
    name = f"{category}_{sample:04d}_{state}{defect_part}_C{camera}_{timestamp}.jpg"
    if label is None:
        label = "OK" if state == "OK" else f"NG_{defect}"
    return {
        "image_path": f"{state}/{defect + '/' if defect else ''}{serial}/{name}",
        "anomaly_class": label,
        "unused_train_payload": "DO_NOT_PROJECT_TRAIN_EXTRA",
    }


def write_category_json(metadata_root, category, rows, *, test_label="PRIVATE_TEST_SENTINEL"):
    payload = {
        "meta": {
            "normal_class": "OK",
            "prefix": "synthetic_prefix",
            "unused_metadata": {"do_not": "project"},
        },
        "train": rows,
        "test": [{"anomaly_class": test_label, "mask_path": "PRIVATE_MASK_SENTINEL"}],
    }
    (metadata_root / f"{category}.json").write_text(
        json.dumps(payload), encoding="utf-8")


def test_preflight_and_execute_default_deny_without_dataset_io(monkeypatch, tmp_path):
    def forbidden(*_args, **_kwargs):
        pytest.fail("default-deny preflight touched a local path")

    monkeypatch.setattr(l2c, "_check_output_path", forbidden)
    monkeypatch.setattr(l2c, "_source_sha256", forbidden)
    report = l2c.preflight()
    denied = l2c.execute_census(None, None, None, expected_dataset_root=str(tmp_path / "never-open"))
    assert report["status"] == "R2_L2C_EXECUTOR_READY_AWAITING_AUTHORIZATION"
    assert report["local_access_authorization_status"] == "R2_L2C_LOCAL_ACCESS_NOT_AUTHORIZED"
    assert report["local_data_io_performed"] is False
    assert len(report["categories"]) == 30
    assert denied["status"] == "R2_L2C_LOCAL_ACCESS_NOT_AUTHORIZED"
    assert denied["local_data_io_performed"] is False


def test_preflight_rejects_approval_document_path_inside_dataset_without_opening(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("preflight tried to open a path inside the dataset root")

    monkeypatch.setattr(l2c.Path, "open", forbidden)
    with pytest.raises(l2c.CensusContractError, match="outside the dataset root"):
        l2c.load_authorization_and_trust(
            r"G:\dataset\realiad\jsons\realiad_jsons\authorization.json",
            r"G:\dataset\realiad\trust.json",
        )


@pytest.mark.parametrize(
    ("permission", "error_fragment"),
    [
        ("allow_physical_json_file_bytes", "physical JSON byte access"),
        ("allow_test_section_bytes_encountered", "test-section byte acknowledgment"),
        ("allow_train_image_path", "TRAIN image_path"),
        ("allow_train_anomaly_class", "TRAIN anomaly_class"),
        ("allow_official_sample_key_condition_tokens", "condition-token interpretation"),
        ("allow_meta_normal_class", "meta.normal_class"),
        ("allow_meta_prefix", "meta.prefix"),
    ],
)
def test_execute_requires_each_requested_permission(tmp_path, permission, error_fragment):
    kwargs = {permission: False}
    auth, trust, pin, root, _metadata, _output = make_authorization(tmp_path, **kwargs)
    errors = l2c.authorization_errors(
        auth, trust, pin, expected_dataset_root=str(root),
    )
    assert any(error_fragment in error for error in errors)


def test_signature_trust_pin_and_unknown_manifest_fields_fail_closed(tmp_path):
    auth, trust, pin, root, _metadata, _output = make_authorization(tmp_path)
    assert not l2c.authorization_errors(auth, trust, pin, expected_dataset_root=str(root))
    assert any("out-of-band SHA-256 pin" in error for error in l2c.authorization_errors(
        auth, trust, "0" * 64, expected_dataset_root=str(root),
    ))
    bad_signature = replace(auth, signature_ed25519_hex="0" * 128)
    assert any("signature is invalid" in error for error in l2c.authorization_errors(
        bad_signature, trust, pin, expected_dataset_root=str(root),
    ))
    with pytest.raises(l2c.CensusContractError, match="fields mismatch"):
        l2c.authorization_from_mapping({**asdict(auth), "unrecognized_permission": True})


def test_incompatible_operation_or_dataset_root_is_rejected(tmp_path):
    auth, trust, pin, root, *_ = make_authorization(tmp_path, decision="R2_L2A_GEOMETRY")
    errors = l2c.authorization_errors(auth, trust, pin, expected_dataset_root=str(root))
    assert any("does not approve the R2-L2C" in error for error in errors)

    auth, trust, pin, root, *_ = make_authorization(tmp_path / "different", categories=("audiojack",))
    errors = l2c.authorization_errors(
        auth, trust, pin, expected_dataset_root=str(root / "not-the-authorized-root"),
    )
    assert any("dataset root differs" in error for error in errors)


def test_unrecognized_output_fields_are_rejected(tmp_path):
    auth, trust, pin, root, *_ = make_authorization(
        tmp_path,
        approved_output_fields=sorted(l2c.REQUIRED_PUBLIC_OUTPUT_FIELDS | {"raw_secret_label"}),
    )
    errors = l2c.authorization_errors(auth, trust, pin, expected_dataset_root=str(root))
    assert any("output fields contain unrecognized names" in error for error in errors)


def test_projection_parser_skips_test_payload_and_projects_only_approved_fields():
    raw = (
        '{"meta":{"normal_class":"OK","prefix":"p","other":"PRIVATE_META"},'
        '"train":[{"image_path":"OK/S0001/audiojack_0001_OK_C1_20261101000001.jpg",'
        '"anomaly_class":"OK","extra":"PRIVATE_TRAIN"}],'
        '"test":[{"anomaly_class":"PRIVATE_TEST_LABEL","mask_path":"PRIVATE_MASK"}]}'
    ).encode("utf-8")
    projection = l2c.project_train_json_bytes(raw)
    serialized = json.dumps({"meta": projection.meta, "train": projection.train_rows})
    assert projection.test_section_present is True
    assert projection.meta == {"normal_class": "OK", "prefix": "p"}
    assert projection.train_rows == ({
        "image_path": "OK/S0001/audiojack_0001_OK_C1_20261101000001.jpg",
        "anomaly_class": "OK",
    },)
    assert all(secret not in serialized for secret in (
        "PRIVATE_META", "PRIVATE_TRAIN", "PRIVATE_TEST_LABEL", "PRIVATE_MASK"))


def test_census_groups_multiview_source_and_uses_approved_prefix(tmp_path):
    auth, trust, pin, root, metadata, output = make_authorization(
        tmp_path, categories=("audiojack",),
    )
    write_category_json(metadata, "audiojack", [
        image_record("audiojack", 1, 1),
        image_record("audiojack", 1, 2),
    ])
    report = l2c.execute_census(
        auth, trust, pin, expected_dataset_root=str(root),
    )
    row = next(item for item in report["categories"] if item["category_id"] == "audiojack")
    assert report["status"] == "R2_L2C_METADATA_CENSUS_COMPLETE_PARTITION_PENDING"
    assert report["metadata_files_opened"] == 1
    assert row["authorized_train_record_count"] == 2
    assert row["eligible_normal_image_count"] == 2
    assert row["distinct_normal_source_group_count"] == 1
    expected_group = source_group_digest_from_key("audiojack", "audiojack_0001_OK")
    assert row["normal_camera_view_distribution"] == {"2": 1}
    assert row["c64_candidate_pool_status"] == "CANDIDATE_POOL_TOO_SMALL_FOR_C64"
    assert row["support_and_c64_identity_status"] == "PENDING_SEPARATE_SUPPORT_C64_SELECTION_AND_REVIEW"
    private_text = (output / "private_source_group_receipts.json").read_text(encoding="utf-8")
    public_text = (output / "census_report.json").read_text(encoding="utf-8")
    assert "audiojack_0001_OK" not in private_text + public_text
    assert "OK/S0001" not in private_text + public_text
    assert "PRIVATE_TEST_SENTINEL" not in private_text + public_text
    assert "PRIVATE_MASK_SENTINEL" not in private_text + public_text
    assert report["test_section_parsed"] is False
    assert report["test_section_bytes_encountered"] is True
    assert report["geometry_performed"] is False
    public_json = json.loads(public_text)
    assert set(public_json).issubset(auth.approved_output_fields)
    private_json = json.loads(private_text)
    assert private_json["receipts"][0]["group_members"][0]["source_group_sha256"] == expected_group


def test_directory_serial_overlap_is_a_separate_authorized_sensitivity(tmp_path):
    auth, trust, pin, root, metadata, output = make_authorization(
        tmp_path, categories=("audiojack",), allow_directory_serial_sensitivity=True,
    )
    write_category_json(metadata, "audiojack", [
        image_record("audiojack", 1, 1, state="OK", label="OK"),
        image_record("audiojack", 1, 2, state="NG", defect="BX", label="NG_BX"),
    ])
    report = l2c.execute_census(auth, trust, pin, expected_dataset_root=str(root))
    row = next(item for item in report["categories"] if item["category_id"] == "audiojack")
    assert row["eligible_normal_image_count"] == 1
    assert row["distinct_normal_source_group_count"] == 1
    assert row["directory_serial_overlap_status"] == "DIRECTORY_SERIAL_OVERLAP_SENSITIVITY_REPORTED"
    assert row["directory_serial_overlap_count"] == 1
    private = json.loads((output / "private_source_group_receipts.json").read_text(encoding="utf-8"))
    assert private["receipts"][0]["directory_serial_overlap_count"] == 1


def test_duplicate_and_malformed_train_identities_are_counted_and_excluded(tmp_path):
    auth, trust, pin, root, metadata, _output = make_authorization(
        tmp_path, categories=("audiojack",),
    )
    valid = image_record("audiojack", 1, 1)
    write_category_json(metadata, "audiojack", [valid, dict(valid), {
        "image_path": "not/a/Real-IAD/image.jpg", "anomaly_class": "OK",
    }])
    report = l2c.execute_census(auth, trust, pin, expected_dataset_root=str(root))
    row = next(item for item in report["categories"] if item["category_id"] == "audiojack")
    assert row["authorized_train_record_count"] == 3
    assert row["eligible_normal_image_count"] == 1
    assert row["duplicate_image_identity_count"] == 1
    assert row["malformed_identity_count"] == 1
    assert row["distinct_normal_source_group_count"] == 1


def test_missing_category_file_is_unknown_not_zero(tmp_path):
    auth, trust, pin, root, metadata, _output = make_authorization(
        tmp_path, categories=("audiojack", "button_battery"),
    )
    write_category_json(metadata, "audiojack", [image_record("audiojack")])
    report = l2c.execute_census(auth, trust, pin, expected_dataset_root=str(root))
    missing = next(row for row in report["categories"] if row["category_id"] == "button_battery")
    assert missing["metadata_availability_status"] == "METADATA_FILE_MISSING"
    assert missing["authorized_train_record_count"] is None
    assert missing["eligible_normal_image_count"] is None
    assert "button_battery" in report["missing_or_unavailable_categories"]


def test_protected_mint_pcb_files_are_not_opened_without_individual_approval(tmp_path):
    auth, trust, pin, root, metadata, _output = make_authorization(
        tmp_path, categories=("audiojack",),
    )
    write_category_json(metadata, "audiojack", [image_record("audiojack")])
    (metadata / "mint.json").write_text("DO_NOT_OPEN_MINT", encoding="utf-8")
    (metadata / "pcb.json").write_text("DO_NOT_OPEN_PCB", encoding="utf-8")
    report = l2c.execute_census(auth, trust, pin, expected_dataset_root=str(root))
    assert report["metadata_files_opened"] == 1
    for category in ("mint", "pcb"):
        row = next(item for item in report["categories"] if item["category_id"] == category)
        assert row["authorization_status"] == "PROTECTED_NOT_AUTHORIZED"
        assert row["metadata_availability_status"] == "NOT_CHECKED_NOT_AUTHORIZED"
        assert row["protected_category_status"] == "PROTECTED_NOT_AUTHORIZED"


def test_all_thirty_categories_are_reported_without_opening_protected_files(tmp_path):
    categories = tuple(category for category in CANONICAL_CATEGORY_IDS if category not in {"mint", "pcb"})
    auth, trust, pin, root, metadata, _output = make_authorization(tmp_path, categories=categories)
    for category in categories:
        write_category_json(metadata, category, [image_record(category)])
    (metadata / "mint.json").write_text("NOT_AUTHORIZED", encoding="utf-8")
    (metadata / "pcb.json").write_text("NOT_AUTHORIZED", encoding="utf-8")
    report = l2c.execute_census(auth, trust, pin, expected_dataset_root=str(root))
    assert len(report["categories"]) == 30
    assert report["target_category_count"] == 30
    assert report["authorized_category_count"] == 28
    assert report["metadata_files_opened"] == 28
    assert report["metadata_available_category_count"] == 28
    assert report["missing_or_unavailable_categories"] == ["mint", "pcb"]


def test_h_source_partition_stays_pending_without_a_trusted_c_identity():
    reviewer_key = Ed25519PrivateKey.generate()
    category = "audiojack"
    role_evidence = {}
    source_hashes = []
    a_group = source_group_digest_from_key(category, "audiojack_0001_OK")
    for role, sample_key in (
        ("A", "audiojack_0001_OK"),
        ("B", "audiojack_0002_OK"),
        ("C", "audiojack_0003_OK"),
    ):
        source_sha = canonical_sha256({"source": role})
        source_hashes.append(source_sha)
        digest = source_group_digest_from_key(category, sample_key)
        receipt = make_role_identity_receipt(
            category_id=category, role=role,
            source_manifest_sha256=source_sha, producer_id="synthetic-producer",
            group_members=(SourceGroupMember(digest, 1),),
        )
        manifest = make_reviewed_identity_manifest(
            receipt,
            population="authorized_normal_reference_only" if role == "A" else "not_asserted",
            normal_evidence_manifest_sha256=canonical_sha256({"normal": role}) if role == "A" else None,
        )
        attestation = make_reviewer_attestation(
            receipt, manifest, reviewer_id="synthetic-reviewer",
            decision_id=f"review-{role}", private_key=reviewer_key,
        )
        role_evidence[(category, role)] = l2c.RoleEvidence(receipt, manifest, attestation)
    untrusted = make_reviewer_trust_directory(
        directory_id="empty-reviewer-trust-root",
        trusted_reviewer_public_keys={},
        trusted_identity_source_sha256s=source_hashes,
    )
    result = l2c.assess_h_source_partition(
        category, (a_group,), role_evidence, untrusted, untrusted.directory_sha256,
    )
    assert result["partition_status"] == "A_CANDIDATES_IDENTIFIED_PARTITION_NOT_CERTIFIED"
    assert result["c_identity_certificate_status"] == "C_IDENTITY_CERTIFICATE_INCOMPLETE"


def test_valid_independently_reviewed_c_identity_certifies_disjoint_partition():
    reviewer_key = Ed25519PrivateKey.generate()
    category = "audiojack"
    role_evidence = {}
    source_hashes = []
    digests = {}
    for role, sample_key in (
        ("A", "audiojack_0001_OK"),
        ("B", "audiojack_0002_OK"),
        ("C", "audiojack_0003_OK"),
    ):
        source_sha = canonical_sha256({"reviewed-source": role})
        source_hashes.append(source_sha)
        digest = source_group_digest_from_key(category, sample_key)
        digests[role] = digest
        receipt = make_role_identity_receipt(
            category_id=category, role=role,
            source_manifest_sha256=source_sha, producer_id="synthetic-producer",
            group_members=(SourceGroupMember(digest, 1),),
        )
        manifest = make_reviewed_identity_manifest(
            receipt,
            population="authorized_normal_reference_only" if role == "A" else "not_asserted",
            normal_evidence_manifest_sha256=canonical_sha256({"normal": role}) if role == "A" else None,
        )
        attestation = make_reviewer_attestation(
            receipt, manifest, reviewer_id="synthetic-reviewer",
            decision_id=f"review-{role}", private_key=reviewer_key,
        )
        role_evidence[(category, role)] = l2c.RoleEvidence(receipt, manifest, attestation)
    trust = make_reviewer_trust_directory(
        directory_id="trusted-synthetic-reviewer",
        trusted_reviewer_public_keys={"synthetic-reviewer": _public_key_hex(reviewer_key)},
        trusted_identity_source_sha256s=source_hashes,
    )
    result = l2c.assess_h_source_partition(
        category, (digests["A"],), role_evidence, trust, trust.directory_sha256,
    )
    assert result["partition_status"] == "SOURCE_GROUP_PARTITION_CERTIFIED"
    assert result["role_status"] == {
        "A": "CERTIFIED_FROM_SOURCE_EVIDENCE",
        "B": "CERTIFIED_FROM_SOURCE_EVIDENCE",
        "C": "CERTIFIED_FROM_SOURCE_EVIDENCE",
    }
    assert result["c_identity_certificate_status"] == "CERTIFIED_FROM_SOURCE_EVIDENCE"


@pytest.mark.parametrize(
    ("count", "candidate_status", "at_least_68"),
    [
        (63, "CANDIDATE_POOL_TOO_SMALL_FOR_C64", False),
        (64, "CANDIDATE_POOL_SIZE_SUFFICIENT", False),
        (67, "CANDIDATE_POOL_SIZE_SUFFICIENT", False),
        (68, "CANDIDATE_POOL_SIZE_SUFFICIENT", True),
    ],
)
def test_c64_candidate_pool_boundaries_are_not_support_c64_certification(count, candidate_status, at_least_68):
    groups = [hashlib.sha256(f"normal-group-{index}".encode()).hexdigest() for index in range(count)]
    result = l2c.support_c64_pool_status(groups)
    assert result["candidate_pool_status"] == candidate_status
    assert result["at_least_68_legal_normal_groups"] is at_least_68
    assert result["support_and_c64_identity_status"] == "PENDING_SEPARATE_SUPPORT_C64_SELECTION_AND_REVIEW"


def test_duplicate_group_invalidates_68_candidate_pool_count():
    groups = [hashlib.sha256(f"normal-group-{index}".encode()).hexdigest() for index in range(67)]
    groups.append(groups[0])
    result = l2c.support_c64_pool_status(groups)
    assert result["candidate_pool_status"] == "DUPLICATE_SOURCE_GROUPS_REJECTED"
    assert result["distinct_candidate_group_count"] == 67
    assert result["at_least_68_legal_normal_groups"] is False


def _signed_selection(category, seed, supports, c64, reviewer_key):
    partial = l2c.SupportC64SelectionReceipt(
        schema="r2_l2c.support_c64_selection.v1",
        category_id=category,
        support_seed=seed,
        support_group_sha256s=tuple(supports),
        c64_group_sha256s=tuple(c64),
        producer_id="synthetic-selection-producer",
        receipt_sha256="",
    )
    payload = asdict(partial)
    payload.pop("receipt_sha256")
    receipt = replace(partial, receipt_sha256=canonical_sha256(payload))
    review_payload = {
        "reviewer_id": "synthetic-selection-reviewer",
        "decision_id": "synthetic-selection-review-01",
        "receipt_sha256": receipt.receipt_sha256,
    }
    signature = reviewer_key.sign(_signature_bytes(review_payload)).hex()
    attestation = l2c.SupportC64SelectionAttestation(
        reviewer_id=review_payload["reviewer_id"],
        decision_id=review_payload["decision_id"],
        receipt_sha256=receipt.receipt_sha256,
        signature_ed25519_hex=signature,
    )
    return receipt, attestation


def test_68_unique_groups_can_certify_four_supports_and_separate_c64_with_review():
    category = "audiojack"
    groups = [hashlib.sha256(f"group-{index}".encode()).hexdigest() for index in range(68)]
    reviewer_key = Ed25519PrivateKey.generate()
    trust = make_reviewer_trust_directory(
        directory_id="support-review-root",
        trusted_reviewer_public_keys={"synthetic-selection-reviewer": _public_key_hex(reviewer_key)},
        trusted_identity_source_sha256s=(),
    )
    receipt, attestation = _signed_selection(category, 7, groups[:4], groups[4:], reviewer_key)
    status = l2c.assess_support_c64_selection(
        groups, receipt, attestation, trust, trust.directory_sha256,
        expected_category_id=category, expected_support_seed=7,
    )
    assert status == "SUPPORT_AND_C64_IDENTITIES_CERTIFIED"


def test_68_unique_groups_with_support_c64_overlap_fail_closed():
    category = "audiojack"
    groups = [hashlib.sha256(f"group-{index}".encode()).hexdigest() for index in range(68)]
    reviewer_key = Ed25519PrivateKey.generate()
    trust = make_reviewer_trust_directory(
        directory_id="support-review-root",
        trusted_reviewer_public_keys={"synthetic-selection-reviewer": _public_key_hex(reviewer_key)},
        trusted_identity_source_sha256s=(),
    )
    receipt, attestation = _signed_selection(category, 7, groups[:4], groups[3:67], reviewer_key)
    status = l2c.assess_support_c64_selection(
        groups, receipt, attestation, trust, trust.directory_sha256,
        expected_category_id=category, expected_support_seed=7,
    )
    assert status == "SUPPORT_C64_OVERLAP_REJECTED"


@pytest.mark.parametrize("mismatch", ("category", "seed"))
def test_support_selection_requires_correct_category_and_seed(mismatch):
    category = "audiojack"
    groups = [hashlib.sha256(f"group-{index}".encode()).hexdigest() for index in range(68)]
    reviewer_key = Ed25519PrivateKey.generate()
    trust = make_reviewer_trust_directory(
        directory_id="support-review-root",
        trusted_reviewer_public_keys={"synthetic-selection-reviewer": _public_key_hex(reviewer_key)},
        trusted_identity_source_sha256s=(),
    )
    receipt, attestation = _signed_selection(
        "switch" if mismatch == "category" else category,
        9 if mismatch == "seed" else 7,
        groups[:4], groups[4:], reviewer_key,
    )
    status = l2c.assess_support_c64_selection(
        groups, receipt, attestation, trust, trust.directory_sha256,
        expected_category_id=category, expected_support_seed=7,
    )
    assert status == "SUPPORT_CATEGORY_OR_SEED_MISMATCH"


def test_67_groups_cannot_certify_four_supports_plus_c64_even_with_receipt():
    groups = [hashlib.sha256(f"group-{index}".encode()).hexdigest() for index in range(67)]
    partial = l2c.SupportC64SelectionReceipt(
        schema="r2_l2c.support_c64_selection.v1",
        category_id="audiojack", support_seed=7,
        support_group_sha256s=tuple(groups[:4]),
        c64_group_sha256s=tuple(groups[4:]),
        producer_id="synthetic-selection-producer", receipt_sha256="0" * 64,
    )
    attestation = l2c.SupportC64SelectionAttestation(
        reviewer_id="synthetic", decision_id="synthetic", receipt_sha256="0" * 64,
        signature_ed25519_hex="0" * 128,
    )
    assert l2c.assess_support_c64_selection(
        groups, partial, attestation, None, None,
        expected_category_id="audiojack", expected_support_seed=7,
    ) == "CANDIDATE_POOL_INSUFFICIENT_FOR_4_SUPPORTS_PLUS_C64"


def test_output_budget_rejects_before_write_and_never_exceeds_limit(tmp_path):
    output = tmp_path / "budget_output"
    budget = l2c.OutputBudget(output, 32)
    try:
        with pytest.raises(l2c.OutputBudgetError, match="pre-write byte limit"):
            budget.write_json("census_report.json", {"x": "y" * 100})
        assert sum(item.stat().st_size for item in output.iterdir()) <= 32
        assert not (output / "census_report.json").exists()
    finally:
        budget.close()


def test_census_output_over_budget_is_blocked_without_overshoot(tmp_path):
    auth, trust, pin, root, metadata, output = make_authorization(
        tmp_path, categories=("audiojack",), max_bytes=1,
    )
    write_category_json(metadata, "audiojack", [image_record("audiojack")])
    report = l2c.execute_census(auth, trust, pin, expected_dataset_root=str(root))
    assert report["status"] == "R2_L2C_BLOCKED"
    if output.exists():
        assert sum(item.stat().st_size for item in output.iterdir()) <= 1


def test_executor_rejects_support_c64_selection_permission_for_census(tmp_path):
    auth, trust, pin, root, *_ = make_authorization(tmp_path, allow_support_c64_selection=True)
    errors = l2c.authorization_errors(auth, trust, pin, expected_dataset_root=str(root))
    assert any("Support/C64 selection is outside" in error for error in errors)
