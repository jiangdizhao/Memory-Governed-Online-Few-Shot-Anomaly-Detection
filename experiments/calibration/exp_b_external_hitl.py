"""Abbas-inspired, source-excluding external memory editors for Experiment B.

This is an explicitly adapted online protocol, not a reproduction of author
code.  The bank stores normalized DINO patch vectors, immutable IDs and source
provenance.  PyTorch is imported lazily so the state and causal rules can be
tested on CPU without loading the research backbone.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CAPACITY = 8192 + 1024
K_NEIGHBORS = 9
FN_REMOVE_COUNT = 5
PATCH_COUNT = 784
FEATURE_DIM = 4096


class ExternalMemoryError(RuntimeError):
    """Raised when an external memory violates its registered protocol."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check_group(value: str) -> str:
    if (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
        raise ExternalMemoryError("Source Group must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class ExternalScore:
    image_score: float
    patch_scores: Any
    anomaly_patch: int
    eligible_prototypes: int
    source_excluded_prototypes: int


class ExternalNormalMemory:
    """Mutable normal reference initialized by copying the actual fixed bank."""

    def __init__(self, fixed_prototypes: Any, *, category: str,
                 experiment_seed: int, capacity: int = CAPACITY,
                 device: Any = "cpu", min_scoring_neighbors: int = K_NEIGHBORS,
                 _synthetic_test_mode: bool = False,
                 _test_patch_count: int = PATCH_COUNT):
        import torch

        if int(capacity) != CAPACITY or int(min_scoring_neighbors) != K_NEIGHBORS:
            raise ExternalMemoryError("Experiment B external capacity and k=9 are frozen")
        values = torch.as_tensor(fixed_prototypes, dtype=torch.float32, device=device).detach().clone()
        expected_dim = int(values.shape[1]) if _synthetic_test_mode and values.ndim == 2 else FEATURE_DIM
        if (values.ndim != 2 or values.shape[1] != expected_dim or values.shape[0] < 1 or
                (not _synthetic_test_mode and expected_dim != FEATURE_DIM)):
            raise ExternalMemoryError("External memory must copy a nonempty fixed prototype bank")
        if not bool(torch.isfinite(values).all().item()):
            raise ExternalMemoryError("Fixed prototypes contain non-finite values")
        norms = torch.linalg.vector_norm(values, dim=1)
        if bool(torch.any(norms < 1e-12).item()):
            raise ExternalMemoryError("Fixed bank contains a zero prototype")
        values = torch.nn.functional.normalize(values, p=2, dim=1)
        self.device = torch.device(device)
        self.category = str(category)
        self.experiment_seed = int(experiment_seed)
        self.capacity = int(capacity)
        self.min_scoring_neighbors = int(min_scoring_neighbors)
        self.feature_dim = expected_dim
        self.patch_count = int(_test_patch_count) if _synthetic_test_mode else PATCH_COUNT
        if self.patch_count < 1:
            raise ExternalMemoryError("Patch count must be positive")
        self.features = values
        self.ids = [f"support:{i:05d}" for i in range(int(values.shape[0]))]
        self.source_groups: list[str | None] = [None] * len(self.ids)
        self.positions = [-1] * len(self.ids)
        self.origin_kind = ["support"] * len(self.ids)
        self.patch_indices = [-1] * len(self.ids)
        self.version = 0
        self.fp_updates = 0
        self.fn_updates = 0
        self.fn_surgery_attempts = 0
        self.fp_insertions = 0
        self.support_removed = 0
        self.online_removed = 0
        self.subsampling_events = 0
        self.subsampled_support = 0
        self.subsampled_online = 0
        self.noop_updates = 0
        self.novelty_threshold: float | None = None
        self.novelty_state_sha256: str | None = None
        self._validate_unique_ids()

    @property
    def count(self) -> int:
        return len(self.ids)

    @property
    def state_sha256(self) -> str:
        digest = hashlib.sha256(b"exp-b-external-memory-v1\0")
        digest.update(self.category.encode("utf-8"))
        digest.update(str((self.capacity, self.count, self.version)).encode("ascii"))
        x = self.features.detach().to("cpu").contiguous().numpy()
        digest.update(str((tuple(x.shape), x.dtype.str)).encode("ascii"))
        digest.update(x.tobytes())
        for prototype_id, group, position, kind, patch in zip(
                self.ids, self.source_groups, self.positions, self.origin_kind,
                self.patch_indices):
            digest.update(f"{prototype_id}\0{group or ''}\0{position}\0{kind}\0{patch}\n".encode("utf-8"))
        return digest.hexdigest()

    def _validate_unique_ids(self) -> None:
        if len(self.ids) != len(set(self.ids)):
            raise ExternalMemoryError("Duplicate external prototype IDs")
        if not (len(self.ids) == len(self.source_groups) == len(self.positions) ==
                len(self.origin_kind) == len(self.patch_indices) == int(self.features.shape[0])):
            raise ExternalMemoryError("External prototype metadata is misaligned")
        if self.count > self.capacity:
            raise ExternalMemoryError("External memory exceeded its registered cap")

    def eligible_indices(self, *, source_group_sha256: str,
                         event_position: int) -> list[int]:
        group = _check_group(source_group_sha256)
        pos = int(event_position)
        if pos < 0:
            raise ExternalMemoryError("Event position must be nonnegative")
        return [i for i, (origin, anchor_pos) in enumerate(zip(self.source_groups, self.positions))
                if origin != group and (anchor_pos < 0 or anchor_pos < pos)]

    def _nearest_distance(self, queries: Any, *, source_group_sha256: str,
                          event_position: int, neighbors: int = K_NEIGHBORS,
                          patch_chunk: int = 32,
                          prototype_chunk: int = 1024) -> tuple[Any, Any, list[int]]:
        import torch

        q = torch.as_tensor(queries, dtype=torch.float32, device=self.device)
        if q.ndim != 2 or q.shape[1] != self.feature_dim or not bool(torch.isfinite(q).all().item()):
            raise ExternalMemoryError("External query feature shape or values are invalid")
        qnorm = torch.linalg.vector_norm(q, dim=1)
        if bool(torch.any(qnorm < 1e-12).item()):
            raise ExternalMemoryError("External scoring received a zero DINO descriptor")
        q = torch.nn.functional.normalize(q, p=2, dim=1)
        eligible = self.eligible_indices(source_group_sha256=source_group_sha256,
                                         event_position=event_position)
        if not eligible:
            raise ExternalMemoryError("No source-eligible normal prototype remains for scoring")
        eligible_t = torch.as_tensor(eligible, dtype=torch.long, device=self.device)
        bank = self.features[eligible_t]
        out_d = torch.empty((q.shape[0],), dtype=torch.float32, device=self.device)
        out_i = torch.empty((q.shape[0],), dtype=torch.long, device=self.device)
        if int(neighbors) not in (1, K_NEIGHBORS):
            raise ExternalMemoryError("Only registered 1NN novelty and 9NN scoring are supported")
        k = min(int(neighbors), len(eligible))
        for q0 in range(0, int(q.shape[0]), patch_chunk):
            q1 = min(q0 + patch_chunk, int(q.shape[0]))
            block = q[q0:q1]
            best_d = torch.empty((block.shape[0], 0), dtype=torch.float32, device=self.device)
            best_i = torch.empty((block.shape[0], 0), dtype=torch.long, device=self.device)
            for b0 in range(0, len(eligible), prototype_chunk):
                b1 = min(b0 + prototype_chunk, len(eligible))
                sim = torch.clamp(block @ bank[b0:b1].T, -1.0, 1.0)
                distance = torch.sqrt(torch.clamp(2.0 - 2.0 * sim, min=0.0))
                ids = torch.arange(b0, b1, device=self.device, dtype=torch.long)
                ids = ids.expand(block.shape[0], -1)
                candidate_d = torch.cat((best_d, distance), dim=1)
                candidate_i = torch.cat((best_i, ids), dim=1)
                # Stable sorting plus ascending memory chunks resolves exact ties by bank order.
                order = torch.argsort(candidate_d, dim=1, stable=True)[:, :k]
                best_d = torch.gather(candidate_d, 1, order)
                best_i = torch.gather(candidate_i, 1, order)
            out_d[q0:q1] = best_d.mean(dim=1)
            out_i[q0:q1] = best_i[:, 0]
        return out_d, eligible_t[out_i], eligible

    def score(self, patch_features: Any, *, source_group_sha256: str,
              event_position: int) -> ExternalScore:
        import torch

        patches = torch.as_tensor(patch_features, dtype=torch.float32, device=self.device)
        if patches.shape != (self.patch_count, self.feature_dim):
            raise ExternalMemoryError("External score requires all 784 DINO patch descriptors")
        distances, nearest_global, eligible = self._nearest_distance(
            patches, source_group_sha256=source_group_sha256,
            event_position=event_position)
        score, patch = torch.max(distances, dim=0)
        return ExternalScore(float(score.item()), distances.detach(), int(patch.item()),
                             len(eligible), self.count - len(eligible))

    def _novelty_distances(self, c64_features: Sequence[Any],
                           c64_source_groups: Sequence[str]) -> Any:
        import torch

        values = []
        if len(c64_source_groups) != len(c64_features):
            raise ExternalMemoryError("C64 novelty features and Source Group identities are misaligned")
        for raw, c64_group in zip(c64_features, c64_source_groups):
            c64_group = _check_group(str(c64_group))
            features = torch.as_tensor(raw, dtype=torch.float32, device=self.device)
            if features.shape != (self.patch_count, self.feature_dim):
                raise ExternalMemoryError("C64 feature cache has an invalid patch tensor")
            nearest, _indices, _eligible = self._nearest_distance(
                features, source_group_sha256=c64_group,
                event_position=2**63 - 1, neighbors=1)
            values.append(nearest.detach())
        return torch.cat(values)

    def calibrate_novelty(self, c64_features: Sequence[Any], *,
                          c64_source_groups: Sequence[str] | None = None) -> dict[str, Any]:
        import torch

        if len(c64_features) != 64:
            raise ExternalMemoryError("Novelty calibration requires exactly 64 cached C64 feature maps")
        groups = (tuple(c64_source_groups) if c64_source_groups is not None else
                  tuple(f"{index:064x}" for index in range(64)))
        if len(groups) != 64:
            raise ExternalMemoryError("Novelty calibration requires 64 registered C64 groups")
        started = __import__("time").perf_counter()
        values = self._novelty_distances(c64_features, groups)
        threshold = float(torch.quantile(values, .5).item())
        if not math.isfinite(threshold):
            raise ExternalMemoryError("C64 novelty threshold is not finite")
        self.novelty_threshold = threshold
        self.novelty_state_sha256 = self.state_sha256
        return {"threshold": threshold, "method": "median of all C64 patch nearest-reference distances",
                "c64_images": len(c64_features), "c64_patch_distances": int(values.numel()),
                "memory_state_sha256": self.novelty_state_sha256,
                "seconds": __import__("time").perf_counter() - started}

    def _seed(self, *, event_position: int, pre_state: str) -> int:
        value = (f"exp-b-hitl-uniform-v1\0{self.category}\0{int(event_position)}\0"
                 f"{self.experiment_seed}\0{pre_state}").encode("utf-8")
        return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")

    def _store_candidates(self, vectors: Any, *, source_group_sha256: str,
                          event_position: int, patch_indices: Sequence[int]) -> dict[str, int]:
        import torch

        group = _check_group(source_group_sha256)
        positions = int(event_position)
        rows = torch.as_tensor(vectors, dtype=torch.float32, device=self.device).detach()
        if rows.ndim != 2 or rows.shape[1] != self.feature_dim or rows.shape[0] != len(patch_indices):
            raise ExternalMemoryError("External insertion rows and patch IDs are misaligned")
        if not bool(torch.isfinite(rows).all().item()):
            raise ExternalMemoryError("External insertion contains non-finite features")
        norms = torch.linalg.vector_norm(rows, dim=1)
        if bool(torch.any(norms < 1e-12).item()):
            raise ExternalMemoryError("External insertion contains a zero prototype")
        rows = torch.nn.functional.normalize(rows, p=2, dim=1)
        if rows.shape[0] == 0:
            return {"inserted": 0, "removed": 0, "support_retained": self.origin_kind.count("support"),
                    "online_retained": self.origin_kind.count("online")}
        pre = self.state_sha256
        new_ids = [f"online:{group}:{positions}:{int(index)}" for index in patch_indices]
        if len(set(new_ids)) != len(new_ids) or set(new_ids) & set(self.ids):
            raise ExternalMemoryError("An online patch generated a duplicate prototype ID")
        merged_features = torch.cat((self.features, rows), dim=0)
        merged_ids = self.ids + new_ids
        merged_groups = self.source_groups + [group] * len(new_ids)
        merged_positions = self.positions + [positions] * len(new_ids)
        merged_kinds = self.origin_kind + ["online"] * len(new_ids)
        merged_patch = self.patch_indices + [int(v) for v in patch_indices]
        inserted = len(new_ids)
        removed = 0
        removed_support = removed_online = 0
        if len(merged_ids) > self.capacity:
            rng = np.random.default_rng(self._seed(event_position=positions, pre_state=pre))
            keep = np.sort(rng.choice(len(merged_ids), size=self.capacity, replace=False))
            keep_set = set(int(i) for i in keep)
            removed_idx = [i for i in range(len(merged_ids)) if i not in keep_set]
            removed_support = sum(merged_kinds[i] == "support" for i in removed_idx)
            removed_online = sum(merged_kinds[i] == "online" for i in removed_idx)
            keep_t = torch.as_tensor(keep, dtype=torch.long, device=self.device)
            merged_features = merged_features[keep_t]
            merged_ids = [merged_ids[i] for i in keep]
            merged_groups = [merged_groups[i] for i in keep]
            merged_positions = [merged_positions[i] for i in keep]
            merged_kinds = [merged_kinds[i] for i in keep]
            merged_patch = [merged_patch[i] for i in keep]
            self.subsampling_events += 1
            self.subsampled_support += removed_support
            self.subsampled_online += removed_online
            removed = len(removed_idx)
        self.features = merged_features.detach().clone()
        self.ids = merged_ids
        self.source_groups = merged_groups
        self.positions = merged_positions
        self.origin_kind = merged_kinds
        self.patch_indices = merged_patch
        self.version += 1
        self.fp_insertions += inserted
        self._validate_unique_ids()
        return {"inserted": inserted, "removed_by_uniform_subsample": removed,
                "support_removed_by_subsample": removed_support,
                "online_removed_by_subsample": removed_online,
                "support_retained": self.origin_kind.count("support"),
                "online_retained": self.origin_kind.count("online")}

    def false_positive_expansion(self, patch_features: Any, *, source_group_sha256: str,
                                 event_position: int, queried: bool, label: int,
                                 arm_prediction: int) -> dict[str, Any]:
        import torch

        if not queried or int(label) != 0 or int(arm_prediction) != 1:
            return {"changed": False, "reason": "FP conditions not all satisfied", "inserted": 0}
        if self.novelty_threshold is None or self.novelty_state_sha256 != self.state_sha256:
            raise ExternalMemoryError("FP expansion requires a current-state C64 novelty threshold")
        patches = torch.as_tensor(patch_features, dtype=torch.float32, device=self.device)
        if patches.shape != (self.patch_count, self.feature_dim):
            raise ExternalMemoryError("FP expansion requires all 784 image patches")
        distances, _nearest, _eligible = self._nearest_distance(
            patches, source_group_sha256=source_group_sha256, event_position=event_position,
            neighbors=1)
        selected = torch.nonzero(distances > float(self.novelty_threshold), as_tuple=False).reshape(-1)
        indices = [int(i) for i in selected.detach().cpu().tolist()]
        if not indices:
            self.noop_updates += 1
            return {"changed": False, "reason": "no patch passed C64 median novelty gate", "inserted": 0}
        result = self._store_candidates(patches[selected], source_group_sha256=source_group_sha256,
                                        event_position=event_position, patch_indices=indices)
        self.fp_updates += 1
        return {"changed": True, "reason": "queried normal classified FP; novelty patches inserted",
                **result}

    def false_negative_surgery(self, patch_features: Any, *, anomaly_patch: int,
                               source_group_sha256: str, event_position: int,
                               queried: bool, label: int, arm_prediction: int) -> dict[str, Any]:
        import torch

        if not queried or int(label) != 1 or int(arm_prediction) != 0:
            return {"changed": False, "reason": "FN conditions not all satisfied", "removed_ids": []}
        self.fn_surgery_attempts += 1
        patches = torch.as_tensor(patch_features, dtype=torch.float32, device=self.device)
        if patches.shape != (self.patch_count, self.feature_dim) or not 0 <= int(anomaly_patch) < self.patch_count:
            raise ExternalMemoryError("FN surgery requires this arm's valid anomaly patch")
        group = _check_group(source_group_sha256)
        eligible = self.eligible_indices(source_group_sha256=group, event_position=event_position)
        if not eligible:
            self.noop_updates += 1
            return {"changed": False, "reason": "no source-eligible prototype", "removed_ids": []}
        if self.count - min(FN_REMOVE_COUNT, len(eligible)) < self.min_scoring_neighbors:
            self.noop_updates += 1
            return {"changed": False, "reason": "removal would violate minimum scoring bank", "removed_ids": []}
        ids_t = torch.as_tensor(eligible, dtype=torch.long, device=self.device)
        query = patches[int(anomaly_patch):int(anomaly_patch) + 1]
        sim = torch.clamp(query @ self.features[ids_t].T, -1.0, 1.0)
        distance = torch.sqrt(torch.clamp(2.0 - 2.0 * sim.reshape(-1), min=0.0))
        order = torch.argsort(distance, stable=True)[:min(FN_REMOVE_COUNT, len(eligible))]
        remove = sorted(int(v) for v in ids_t[order].detach().cpu().tolist())
        removed_ids = [self.ids[i] for i in remove]
        removed_support = sum(self.origin_kind[i] == "support" for i in remove)
        removed_online = len(remove) - removed_support
        keep = [i for i in range(self.count) if i not in set(remove)]
        keep_t = torch.as_tensor(keep, dtype=torch.long, device=self.device)
        self.features = self.features[keep_t].detach().clone()
        self.ids = [self.ids[i] for i in keep]
        self.source_groups = [self.source_groups[i] for i in keep]
        self.positions = [self.positions[i] for i in keep]
        self.origin_kind = [self.origin_kind[i] for i in keep]
        self.patch_indices = [self.patch_indices[i] for i in keep]
        self.version += 1
        self.fn_updates += 1
        self.support_removed += removed_support
        self.online_removed += removed_online
        self._validate_unique_ids()
        return {"changed": True, "reason": "queried defect classified FN; nearest mutable normal prototypes removed",
                "removed_ids": removed_ids, "support_removed": removed_support,
                "online_removed": removed_online, "removed_count": len(remove)}

    def apply_feedback(self, patch_features: Any, *, source_group_sha256: str,
                       event_position: int, queried: bool, label: int | None,
                       arm_prediction: int, anomaly_patch: int,
                       enable_fn_surgery: bool) -> dict[str, Any]:
        if not queried:
            if label is not None:
                raise ExternalMemoryError("Unqueried labels may not reach external updates")
            return {"fp": {"changed": False, "reason": "unqueried"},
                    "fn": {"changed": False, "reason": "unqueried"}}
        if label not in (0, 1):
            raise ExternalMemoryError("Queried external feedback must have a verified binary label")
        pre_version = self.version
        fp = self.false_positive_expansion(
            patch_features, source_group_sha256=source_group_sha256,
            event_position=event_position, queried=True, label=int(label),
            arm_prediction=int(arm_prediction))
        fn: dict[str, Any] = {"changed": False, "reason": "E1 has no FN surgery"}
        if enable_fn_surgery:
            fn = self.false_negative_surgery(
                patch_features, anomaly_patch=anomaly_patch,
                source_group_sha256=source_group_sha256, event_position=event_position,
                queried=True, label=int(label), arm_prediction=int(arm_prediction))
        if self.version < pre_version or self.version - pre_version > 2:
            raise ExternalMemoryError("External feedback state version changed unexpectedly")
        return {"fp": fp, "fn": fn, "state_version_before": pre_version,
                "state_version_after": self.version}

    def resource_ledger(self) -> dict[str, int | str]:
        return {"capacity": self.capacity, "occupancy": self.count,
                "prototype_bytes": int(self.features.numel() * self.features.element_size()),
                "prototype_id_utf8_bytes": sum(len(v.encode("utf-8")) for v in self.ids),
                "group_digest_bytes": sum(32 for value in self.source_groups if value is not None),
                "position_metadata_bytes": 8 * len(self.positions),
                "support_retained": self.origin_kind.count("support"),
                "online_retained": self.origin_kind.count("online"),
                "support_removed_total": self.support_removed + self.subsampled_support,
                "online_removed_total": self.online_removed + self.subsampled_online,
                "fp_insertions_total": self.fp_insertions,
                "uniform_subsampling_events": self.subsampling_events,
                "uniform_subsampling_replacements": self.subsampled_support + self.subsampled_online,
                "fn_surgery_attempts": self.fn_surgery_attempts,
                "fn_surgery_changes": self.fn_updates,
                "state_version": self.version, "state_sha256": self.state_sha256}
