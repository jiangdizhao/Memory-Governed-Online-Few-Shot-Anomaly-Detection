"""D3-C causal-label-blind context/eligibility/hit-rate random controls.

All random masks are deterministic functions of a PRE-REGISTERED repeat seed,
the method and the image identity; NO test labels, feature scores or online
ground truth are used in a mask.

1. shuffled_spatial: randomly reassign actual (u,v) provenance of the frozen
   dynamic bank, preserving its coordinate multiset and number of prototypes.
2. matched_count_random: per query PATCH, permute the true spatial_0125
   eligibility row, preserving EXACTLY the number of eligible references, and
   therefore EXACTLY the native top-5 availability for that patch.
3. source_calibrated_hit_random: Bernoulli-p dynamic eligibility where p is
   chosen using eight DISJOINT training-history NORMAL views ONLY, by matching
   the real Spatial 0.125 dynamic-win fraction ON THAT SOURCE COHORT.
   Probe A/B features/scores/labels may NOT determine p. Probe hit-rate
   differences are explicitly reported, never retroactively matched.

Important: matched-count is matched on cardinality, not on actual winners;
source-calibrated hit is NOT guaranteed to achieve exact hit-rate matching on
new probe data. Both are honest random controls, not "matched oracle" policies.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Mapping, Sequence

import torch

from .conditional_retrieval import eligible_mask
from .native_bridge import NativeParityError

CONTROL_TYPES = (
    "true_spatial_0125",
    "shuffled_spatial",
    "matched_count_random",
    "source_calibrated_hit_random",
)
SOURCE_CALIB_P_GRID = (0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00)
REPEATS = (0, 1, 2)
SOURCE_CALIB_N = 8


def deterministic_seed(*parts: object) -> int:
    text = "\u001f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big") % (2**63 - 1)


def shuffled_spatial_provenance(
        provenance: Sequence[Mapping], *,
        method: str, replicate: int) -> list[dict]:
    """Shuffle only spatial provenance, never a feature or physical camera."""
    if not provenance:
        raise NativeParityError("Shuffled control requires native dynamic memory")
    rows = [dict(p) for p in provenance]
    positions = [(float(x["u"]), float(x["v"])) for x in rows]
    rng = random.Random(deterministic_seed(
        "D3C", "shuffled_spatial", method, replicate))
    rng.shuffle(positions)
    for row, (u,v) in zip(rows,positions):
        row["u"],row["v"]=u,v
    if sorted(positions) != sorted(
            (float(x["u"]),float(x["v"])) for x in provenance):
        raise NativeParityError("Shuffled coordinates not permutation preserved")
    return rows


class RandomEligibility:
    """Call from ScopedRetrieval's label-blind mask-provider hook."""
    def __init__(self, kind: str, *, method: str, replicate: int,
                 probability: float | None = None):
        if kind not in ("matched_count_random", "source_calibrated_hit_random"):
            raise NativeParityError(f"Unsupported random eligibility kind: {kind}")
        if kind=="source_calibrated_hit_random" and (
                probability is None or
                not 0.0 <= float(probability) <= 1.0):
            raise NativeParityError("Hit-calibrated random mask needs valid p")
        if kind=="matched_count_random" and probability is not None:
            raise NativeParityError("Count-matched random has no p")
        self.kind=kind
        self.method=method
        self.replicate=int(replicate)
        self.p=probability

    def __call__(self, provenance: Sequence[Mapping], policy: str, *,
                 query_camera: int, query_coords: torch.Tensor,
                 image_id: str) -> torch.Tensor:
        if policy!="spatial_0125" or not image_id:
            raise NativeParityError("Random control must run on named spatial query")
        n,m=len(query_coords),len(provenance)
        if not m:
            raise NativeParityError("Random control without dynamic prototypes")
        generator=torch.Generator(device="cpu")
        generator.manual_seed(deterministic_seed(
            "D3C",self.kind,self.method,self.replicate,image_id))
        draws=torch.rand((n,m),generator=generator)
        if self.kind=="matched_count_random":
            true_mask=eligible_mask(
                provenance, "spatial_0125",
                query_camera=query_camera,query_coords=query_coords).cpu()
            order=draws.argsort(dim=1)
            randomized=torch.gather(true_mask,1,order)
            if not torch.equal(randomized.sum(1),true_mask.sum(1)):
                raise NativeParityError("Random eligibility counts differ")
        else:
            randomized=draws < float(self.p)
        return randomized.to(device=query_coords.device)


def choose_probability(
        target: float, grid_values: Mapping[float,float]) -> dict:
    """Fit on SOURCE training-normal dynamic hit rate; no Probe inputs."""
    if not 0 <= float(target) <= 1:
        raise NativeParityError("Source target not in [0,1]")
    if tuple(sorted(grid_values)) != SOURCE_CALIB_P_GRID:
        raise NativeParityError("Unregistered source-only Bernoulli grid")
    for p,v in grid_values.items():
        if not 0 <= v <= 1 or not math.isfinite(float(v)):
            raise NativeParityError("Invalid source-only hit measurement")
    chosen=min(SOURCE_CALIB_P_GRID,
               key=lambda p:(abs(grid_values[p]-target),p))
    return {"selected_p":chosen,
            "source_spatial_dynamic_hit_fraction":target,
            "source_random_dynamic_hit_fraction":grid_values[chosen],
            "source_absolute_hit_gap":abs(grid_values[chosen]-target),
            "grid_measurements":{str(p):grid_values[p] for p in SOURCE_CALIB_P_GRID},
            "selection": "min_absolute_source_hit_gap_ties_smallest_p",
            "probe_used_in_selection":False}
