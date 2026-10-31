from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

from experiments.calibration import r1_s3_native_map_parity as s3


def _synthetic_tape(tmp_path):
    root = tmp_path / "r1_s3_fixture"
    (root / "events").mkdir(parents=True)
    base = np.linspace(0.0, 1.0, 784, dtype=np.float32)
    ids = np.arange(32, dtype=np.int64)
    selected = np.zeros((32, 4096), dtype=np.float32)
    selected[:, 0] = 1.0
    plan = [{
        "event_id": "switch-s0-0000",
        "position": 0,
        "opaque_group_identity_sha256": "a" * 64,
        "relative_id_sha256": "b" * 64,
    }]
    manifest = {
        "schema": s3.TAPE_SCHEMA,
        "status": "INCOMPLETE",
        "event_plan": plan,
        "events_processed": 0,
        "fixed_support_sha256_actual": "c" * 64,
        "fixed_support_raw_prototype_sha256": "f" * 64,
        "dynamic_memory_sha256_before": "d" * 64,
        "dynamic_memory_sha256_after": "d" * 64,
    }
    (root / "run_manifest.json").write_text(
        s3.canonical_json(manifest) + "\n", encoding="utf-8")
    arrays = s3._event_arrays(
        base_map=base,
        debug_map=base.reshape(28, 28),
        adapter_ids=ids,
        native_ids=ids,
        debug_ids=ids,
        selected_features=selected,
    )
    metadata = {
        "event_id": "switch-s0-0000",
        "position": 0,
        "opaque_group_identity_sha256": "a" * 64,
        "relative_id_sha256": "b" * 64,
        "source_image_sha256": "e" * 64,
        "fixed_support_sha256": "c" * 64,
        "fixed_support_raw_prototype_sha256": "f" * 64,
        "pre_dynamic_memory_sha256": "d" * 64,
        "post_dynamic_memory_sha256": "d" * 64,
        "encoder_fingerprint_sha256": "1" * 64,
        "preprocess_fingerprint_sha256": "2" * 64,
        "scorer_profile_sha256": "3" * 64,
        "feature_tensor_sha256": "4" * 64,
        "feature_tensor_shape": [784, 4096],
        "feature_tensor_dtype": "torch.float32",
        "selected_feature_sha256": s3._array_digest(selected),
        "adapter_top32_ids_sha256": s3._array_digest(ids),
        "native_top32_ids_sha256": s3._array_digest(ids),
        "debug_top32_ids_sha256": s3._array_digest(ids),
        "map_parity": s3._map_comparison(base, base.reshape(28, 28)),
        "adapter_score": float(base.max()),
        "native_score": float(base.max()),
        "zero_correction_reconstruction_exact": True,
        "dynamic_memory_empty_and_unchanged": True,
        "source_feature_tensor_unchanged": True,
    }
    s3._write_event(root, arrays, metadata)
    manifest["events_processed"] = 1
    (root / "run_manifest.json").write_text(
        s3.canonical_json(manifest) + "\n", encoding="utf-8")
    return root


def test_deterministic_map_shard_serialization_and_readback(tmp_path):
    value = np.arange(784, dtype=np.float32)
    arrays = {"base_map_fp32": value}
    first = s3._deterministic_npz_bytes(arrays)
    second = s3._deterministic_npz_bytes(arrays)
    assert first == second
    loaded = s3._load_array_shard(_write_npz(tmp_path, first))
    np.testing.assert_array_equal(loaded["base_map_fp32"], value)
    assert loaded["base_map_fp32"].dtype == np.float32


def _write_npz(tmp_path, payload):
    path = tmp_path / "roundtrip.npz"
    path.write_bytes(payload)
    return path


def test_dense_map_shape_dtype_finiteness_and_top32_validation():
    good = np.zeros(784, dtype=np.float32)
    assert s3.validate_dense_map(good) is good
    with pytest.raises(ValueError, match="shape"):
        s3.validate_dense_map(np.zeros((28, 28), dtype=np.float32))
    with pytest.raises(ValueError, match="float32"):
        s3.validate_dense_map(np.zeros(784, dtype=np.float64))
    with pytest.raises(ValueError, match="nonfinite"):
        s3.validate_dense_map(np.full(784, np.nan, dtype=np.float32))
    ids = s3.validate_top32(np.arange(32, dtype=np.int32))
    assert ids.dtype == np.int64
    assert len(set(ids.tolist())) == 32
    with pytest.raises(ValueError, match="unique"):
        s3.validate_top32(np.zeros(32, dtype=np.int64))


def test_image_max_aggregation_and_exact_zero_correction_without_mutation():
    base = np.linspace(-1.0, 1.0, 784, dtype=np.float32)
    before = base.copy()
    base_sha = s3._array_digest(base)
    residual = np.zeros(784, dtype=np.float32)
    rebuilt = s3.zero_correction_reconstruction(base, residual)
    np.testing.assert_array_equal(rebuilt, base)
    np.testing.assert_array_equal(base, before)
    assert s3._array_digest(base) == base_sha
    assert s3.max_aggregate(base) == float(np.max(base))
    with pytest.raises(ValueError, match="exact zero"):
        s3.zero_correction_reconstruction(base, np.ones(784, dtype=np.float32))


def test_event_tape_is_immutable_checksummed_and_contains_no_label_fields(tmp_path):
    root = _synthetic_tape(tmp_path)
    receipt = s3.verify_tape(root)
    assert receipt["readback_integrity_pass"] is True
    assert receipt["events_present"] == 1
    assert receipt["dense_map_exact_readback_count"] == 1
    event = json.loads((root / "events" / "event_0000.json").read_text())
    assert "label" not in json.dumps(event).lower()
    assert "mask" not in json.dumps(event).lower()
    assert event["arrays"]["adapter_base_map_fp32"]["dtype"] == "<f4"
    assert event["arrays"]["selected_normalized_features_fp32"]["shape"] == [32, 4096]


def test_event_shards_are_write_once(tmp_path):
    root = _synthetic_tape(tmp_path)
    event_path = root / "events" / "event_0000.json"
    before = event_path.read_bytes()
    event = json.loads(before)
    arrays = s3._load_array_shard(root / "events" / "event_0000.npz")
    with pytest.raises(FileExistsError):
        s3._write_event(root, arrays, {key: value for key, value in event.items()
                                       if key not in ("schema", "arrays", "shard_file", "shard_sha256")})
    assert event_path.read_bytes() == before


def test_mismatched_event_id_and_missing_shard_are_rejected(tmp_path):
    wrong_id_root = _synthetic_tape(tmp_path / "wrong-id")
    event_path = wrong_id_root / "events" / "event_0000.json"
    event = json.loads(event_path.read_text())
    event["event_id"] = "switch-s0-0001"
    event_path.write_text(s3.canonical_json(event) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity or shard checksum"):
        s3.verify_tape(wrong_id_root)

    missing_root = _synthetic_tape(tmp_path / "missing")
    (missing_root / "events" / "event_0000.npz").unlink()
    with pytest.raises(ValueError, match="count differs"):
        s3.verify_tape(missing_root)


def test_tape_rejects_label_mask_and_controller_payload_keys():
    for payload in ({"query_label": None}, {"pixel_mask": None},
                    {"C1_prediction": 0}, {"correction_anchor": []}):
        with pytest.raises(ValueError, match="forbidden event field"):
            s3._reject_forbidden_event_fields(payload)


def test_source_commit_profile_and_output_namespace_are_fail_closed(tmp_path):
    with pytest.raises(s3.S3Error, match="Source revision mismatch"):
        s3._validate_source_values(s3.BRANCH, "0" * 40, "1" * 40, s3.PROFILE_ID)
    with pytest.raises(s3.S3Error, match="Unsupported S3 profile"):
        s3._validate_profile_id("r1-s4-or-other")
    existing = tmp_path / "already-used"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        s3.require_fresh_output_namespace(existing)


def test_incompatible_scorer_profile_is_rejected():
    from experiments.calibration.profiles import DEFAULT_PROFILE

    valid = {
        "scorer_config": asdict(DEFAULT_PROFILE.scorer),
        "scorer_config_sha256": DEFAULT_PROFILE.scientific_sha256,
    }
    assert s3._profile_receipt(valid)["sha256"] == s3.EXPECTED_R1_SCORER_PROFILE_SHA256
    invalid = dict(valid)
    invalid["scorer_config"] = dict(valid["scorer_config"])
    invalid["scorer_config"]["top_k"] = -1
    with pytest.raises(s3.S3Error, match="profile differs"):
        s3._profile_receipt(invalid)
