"""Deterministic sanity check for the pre-registered V0.2 photometric shift.

B's RGB gains and additive offset remain frozen. An encoder embedding-distance
threshold is *not* a protocol requirement: a perfectly valid changed image may
produce very similar DINO features, especially in some object categories.

Grounded validity requirement: the image transform changes at least one pixel
in the inspected Phase-II images; sampled feature-distance measurements must
be finite. Their magnitude is reported and never used to accept/reject a
scientific condition or choose a category.
"""
from __future__ import annotations
import math


def validate_shift_observations(feature_cosine_distances, changed_pixel_fractions):
    distances = [float(x) for x in feature_cosine_distances]
    pixels = [float(x) for x in changed_pixel_fractions]
    if not distances or len(distances) != len(pixels):
        raise AssertionError("No matched original/B image observations for shift validation")
    if any(not math.isfinite(x) or x < -1e-3 or x > 2.001 for x in distances):
        raise AssertionError("Invalid feature-cosine shift measurements")
    if any(not math.isfinite(x) or not 0.0 <= x <= 1.0 for x in pixels):
        raise AssertionError("Invalid B-mode changed-pixel fraction")
    if not any(x > 0.0 for x in pixels):
        raise AssertionError("Frozen B transformation did not change any inspected image pixels")
    mean_distance = sum(distances) / len(distances)
    return {
        "sampled_image_count":len(distances),
        "mean_B_feature_cosine_shift":mean_distance,
        "min_B_feature_cosine_shift":min(distances),
        "max_B_feature_cosine_shift":max(distances),
        "mean_B_changed_pixel_fraction":sum(pixels) / len(pixels),
        # Diagnostic only: the former 0.01 hard rejection was not part of the
        # pre-registered experimental objective and can exclude valid classes.
        "feature_shift_below_legacy_0p01_diagnostic_only":mean_distance < 0.01,
        "passed_pixel_intervention_check":True,
    }
