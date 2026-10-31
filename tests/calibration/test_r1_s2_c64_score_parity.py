from __future__ import annotations

import numpy as np
import pytest

from experiments.calibration import r1_s2_c64_score_parity as s2


def _frozen_and_receipt():
    support_groups = [f"switch_{i:04d}_OK" for i in range(4)]
    c64_groups = [f"switch_{i + 10:04d}_OK" for i in range(64)]
    support_images = [f"OK/S{i:04d}/{group}_C1_fixture.jpg"
                      for i, group in enumerate(support_groups)]
    c64_images = [f"OK/S{i + 10:04d}/{group}_C2_fixture.jpg"
                  for i, group in enumerate(c64_groups)]
    support = {
        "images": support_images,
        "groups": support_groups,
        "image_sha256": s2._opaque_digest("support-images", support_images),
        "group_sha256": s2._opaque_digest("support-groups", support_groups),
        "c64_disjoint": True,
    }
    c64 = {
        "images": c64_images,
        "groups": c64_groups,
        "image_sha256": s2._opaque_digest("c64-images", c64_images),
        "group_sha256": s2._opaque_digest("c64-groups", c64_groups),
        "count": 64,
    }
    frozen_info = {
        "supports": {"0": support},
        "c64": c64,
        "support_c64_disjoint": True,
    }
    row = {
        "c64_score_count": 64,
        "c64_group_sha256": c64["group_sha256"],
        "fixed_support_sha256": s2.SUPPORT_DIGEST_EXPECTED,
        "native_parity": {"passed": True},
    }
    return frozen_info, row, support_images, c64_images, c64["image_sha256"]


def test_c64_identity_and_frozen_image_group_order_are_checked():
    info, row, _, c64_images, identity = _frozen_and_receipt()
    hist = np.linspace(0.1, 0.9, 64, dtype=np.float64)
    checked = s2.validate_c64_metadata(
        frozen_info=info, score_row=row, historical_scores=hist,
        receipt_identity_sha256=identity)
    assert checked["c64_count"] == 64
    assert checked["c64_order_sha256"] == s2.ordered_identity_sha256(c64_images)

    reordered = dict(info)
    reordered["c64"] = dict(info["c64"])
    reordered["c64"]["images"] = list(reversed(c64_images))
    # The opaque Stage 0.5 identity sorts IDs, so order is bound separately.
    assert s2._opaque_digest("c64-images", reordered["c64"]["images"]) == identity
    assert s2.ordered_identity_sha256(reordered["c64"]["images"]) != checked[
        "c64_order_sha256"]
    with pytest.raises(s2.InputIntegrityError, match="alignment"):
        s2.validate_c64_metadata(
            frozen_info=reordered, score_row=row, historical_scores=hist,
            receipt_identity_sha256=identity)


def test_receipt_identity_mismatch_fails_closed():
    info, row, _, _, identity = _frozen_and_receipt()
    with pytest.raises(s2.InputIntegrityError, match="identity"):
        s2.validate_c64_metadata(
            frozen_info=info, score_row=row,
            historical_scores=np.ones(64),
            receipt_identity_sha256="0" * 64)


def test_score_vector_length_mismatch_fails_closed():
    info, row, _, _, identity = _frozen_and_receipt()
    with pytest.raises(s2.InputIntegrityError, match="64 finite"):
        s2.validate_c64_metadata(
            frozen_info=info, score_row=row,
            historical_scores=np.ones(63), receipt_identity_sha256=identity)


def test_score_comparison_reports_exact_and_numerical_parity():
    _, _, _, c64_images, identity = _frozen_and_receipt()
    hist = np.linspace(0.1, 0.9, 64, dtype=np.float64)
    rebuilt = hist.copy()
    rebuilt[0] += 1e-6
    result = s2.compare_score_vectors(
        hist, rebuilt, expected_identity_sha256=identity,
        actual_identity_sha256=identity,
        expected_order_sha256=s2.ordered_identity_sha256(c64_images),
        actual_order_sha256=s2.ordered_identity_sha256(c64_images),
        ordered_c64_images=c64_images)
    assert result["paired_scores"] == 64
    assert result["exact_elementwise_equality_count"] == 63
    assert result["numerical_tolerance_match_count"] == 64
    assert result["BITWISE_SCORE_PARITY"] is False
    assert result["NUMERICAL_SCORE_PARITY"] is True
    assert result["SCORE_PARITY_FAILED"] is False
    assert result["maximum_error_position_zero_based"] == 0
    assert result["maximum_error_opaque_sample_identity_sha256"] != c64_images[0]
    assert result["historical_score_vector_sha256"] != result[
        "rebuilt_score_vector_sha256"]


def test_score_order_mismatch_fails_before_numerical_comparison():
    _, _, _, c64_images, identity = _frozen_and_receipt()
    hist = np.linspace(0.1, 0.9, 64, dtype=np.float64)
    with pytest.raises(s2.InputIntegrityError, match="ordered sample identity"):
        s2.compare_score_vectors(
            hist, hist, expected_identity_sha256=identity,
            actual_identity_sha256=identity,
            expected_order_sha256=s2.ordered_identity_sha256(c64_images),
            actual_order_sha256=s2.ordered_identity_sha256(list(reversed(c64_images))),
            ordered_c64_images=c64_images)


def test_score_tolerance_failure_is_explicit():
    _, _, _, c64_images, identity = _frozen_and_receipt()
    hist = np.linspace(0.1, 0.9, 64, dtype=np.float64)
    rebuilt = hist.copy()
    rebuilt[-1] += 0.1
    result = s2.compare_score_vectors(
        hist, rebuilt, expected_identity_sha256=identity,
        actual_identity_sha256=identity,
        expected_order_sha256=s2.ordered_identity_sha256(c64_images),
        actual_order_sha256=s2.ordered_identity_sha256(c64_images),
        ordered_c64_images=c64_images)
    assert result["numerical_tolerance_match_count"] == 63
    assert result["NUMERICAL_SCORE_PARITY"] is False
    assert result["SCORE_PARITY_FAILED"] is True


def test_nonfinite_score_fails_closed():
    _, _, _, c64_images, identity = _frozen_and_receipt()
    hist = np.ones(64, dtype=np.float64)
    rebuilt = hist.copy()
    rebuilt[4] = np.nan
    with pytest.raises(s2.ScoreParityFailure, match="nonfinite"):
        s2.compare_score_vectors(
            hist, rebuilt, expected_identity_sha256=identity,
            actual_identity_sha256=identity,
            expected_order_sha256=s2.ordered_identity_sha256(c64_images),
            actual_order_sha256=s2.ordered_identity_sha256(c64_images),
            ordered_c64_images=c64_images)


def test_source_revision_and_profile_identity_are_mandatory():
    args = {
        "branch": s2.EXPECTED_BRANCH,
        "head": "a" * 40,
        "expected_source_commit": "a" * 40,
        "profile_id": s2.PROFILE_ID,
        "run_id": "c64parity_20261031_a1",
        "authorization_id": "R1-S2-BOUNDED-C64-PARITY-20261031",
    }
    assert s2.validate_execution_identity(**args)["source_commit"] == "a" * 40
    with pytest.raises(s2.SourceIdentityError, match="revision mismatch"):
        s2.validate_execution_identity(**{**args, "head": "b" * 40})
    with pytest.raises(s2.SourceIdentityError, match="profile"):
        s2.validate_execution_identity(**{**args, "profile_id": "r1_unlocked"})


def test_output_namespace_identity_is_versioned_and_bounded():
    assert s2.IDENTITY_CONTRACT == "r1-s2.source-bound.v1"
    assert s2.MAX_GPU_SECONDS == 1200
    assert s2.MAX_OUTPUT_BYTES == 100 * 1024 * 1024
    assert s2.OUTPUT_PREFIX.startswith("r1_s2_c64_parity_switch_s0_20261031_")
    launcher = (s2.ROOT / "experiments/calibration/run_r1_s2_bounded.ps1").read_text(
        encoding="utf-8")
    assert "taskkill.exe /PID $process.Id /T /F" in launcher
    assert "-RedirectStandardOutput $stdoutPath" in launcher
    assert "-RedirectStandardError $stderrPath" in launcher
    assert "$TimeoutSeconds = 1200" in launcher
    assert "$OutputLimitBytes = 100 * 1024 * 1024" in launcher
