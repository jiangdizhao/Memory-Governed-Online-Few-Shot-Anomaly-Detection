"""Synthetic-feature audit of the unchanged native Greedy memory operator.

The fixtures are not an anomaly scorer. They exercise the original
``MemoryArm.update``/``greedy_slot`` code, the full-matrix native bridge, and
the calibration shared writer against reproducible FP16 PatchVMB tensors.
Production scores are checked separately by ``native_parity_smoke``.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Any

import torch

from experiments.risk_safe.native_bridge import (
    dynamic_codebooks, normalized_native_candidate_rows, snapshot,
)
from experiments.risk_safe.shadow_memory import GreedyBank
from .native import SharedNativeGreedyWriter, _fixed_bank_digest


ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str((tuple(value.shape), str(value.dtype))).encode("ascii"))
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def _new_native_bank(impl, *, layers: int = 2, dim: int = 256):
    bank = impl.PatchVMB(
        device="cpu", K_normal_fixed=8, K_normal_dyn=256,
        K_normal_ltm=1, K_defect=1, K_concept=16, knn_k_normal=2,
        corr_cfg=impl.CorrectionConfig(enabled=False), use_defect_bank=False)
    features = [torch.randn((16, dim), generator=torch.Generator().manual_seed(i))
                for i in range(layers)]
    bank._ensure_mode_view_normals("p0", 0, "id", features)
    for layer, codebook in enumerate(bank.normal_fixed["p0"][0]["id"]):
        support = torch.zeros((1, dim), dtype=torch.float16)
        support[0, (dim - 1 - layer) % dim] = 1
        codebook.protos = support
    return bank


def _make_arm(impl, bank):
    from experiments.v02_architecture_core import MemorySettings
    from experiments.v02_memory import MemoryArm

    arm = MemoryArm(
        "greedy_coreset", bank,
        MemorySettings(stm=256, ltm=0, buffer=0, flat=256,
                       cap_bytes=2_000_000_000), seed=0)
    return arm


def _install_fixture_state(writer: SharedNativeGreedyWriter,
                           native_bank: Any, layers: tuple[torch.Tensor, ...],
                           sources: list[list[str]], version: int) -> dict[str, set[str]]:
    state = writer.state_dict()
    seeded = GreedyBank(layers, writer.bank.capacities, version=version)
    source_state = [list(rows) for rows in sources]
    state.update({"layers": [x.detach().clone() for x in seeded.layers],
                  "sources": source_state, "version": version,
                  "sha256": seeded.sha256(), "source_revision": 0})
    state["source_map_sha256"] = hashlib.sha256(
        __import__("json").dumps(source_state, separators=(",", ":"),
                                 ensure_ascii=True).encode("ascii")).hexdigest()
    writer.load_state_dict(state)
    allowed: dict[str, set[str]] = {}
    for layer_idx, (tensor, row_sources) in enumerate(zip(seeded.layers, sources)):
        for row, key in enumerate(row_sources):
            allowed.setdefault(f"{key}\0{layer_idx}", set()).add(
                _tensor_sha(tensor[row]))
    for cb, layer in zip(dynamic_codebooks(native_bank), seeded.layers):
        cb.protos = layer.detach().clone()
    return allowed


def _audit_source_contents(writer: SharedNativeGreedyWriter,
                           allowed: dict[str, set[str]]) -> dict[str, Any]:
    actual = snapshot(writer.native_bank, version=writer.bank.version)
    checks = 0
    for layer_idx, (tensor, row_sources) in enumerate(zip(actual.layers, writer.sources)):
        if len(tensor) != len(row_sources):
            raise AssertionError("Source attribution rows do not align with native prototypes")
        for row, key in enumerate(row_sources):
            digest = _tensor_sha(tensor[row])
            if digest not in allowed.get(f"{key}\0{layer_idx}", set()):
                raise AssertionError(
                    f"Source attribution does not match prototype bytes at layer={layer_idx}, row={row}")
            checks += 1
    return {"prototype_rows_checked": checks,
            "source_map_sha256": writer.source_map_sha256(),
            "source_revision": writer.source_revision,
            "passed": True}


def _fixture(name: str, dim: int, layers: int):
    eye = torch.eye(dim, dtype=torch.float32)
    if name == "append":
        initial = tuple(torch.empty((0, dim), dtype=torch.float16)
                        for _ in range(layers))
        candidates = [torch.roll(eye[:8], shifts=i, dims=1) for i in range(layers)]
        return initial, candidates, None
    if name == "skip":
        initial = tuple(eye.to(torch.float16).clone() for _ in range(layers))
        candidates = [eye[0:1].expand(8, -1).clone() for _ in range(layers)]
        return initial, candidates, None
    if name == "replace":
        initial = tuple(eye[0:1].expand(256, -1).to(torch.float16).clone()
                        for _ in range(layers))
        candidates = [eye[1:9].clone() for _ in range(layers)]
        return initial, candidates, None
    if name == "repeated_replace_same_slot":
        theta1, theta2 = 0.01, 0.02
        p0 = eye[0].clone()
        p1 = p0 * math.cos(0.001) + eye[1] * math.sin(0.001)
        rest = eye[2:256].clone()
        seed = torch.cat((p0[None], p1[None], rest), dim=0).to(torch.float16)
        q1 = p0 * math.cos(theta1) + eye[1] * math.sin(theta1)
        q2 = p0 * math.cos(theta2) + eye[1] * math.sin(theta2)
        fill = eye[2].expand(7, -1)
        first = torch.cat((q1[None], fill), dim=0)
        second = torch.cat((q2[None], fill), dim=0)
        return (seed.clone() for _ in range(layers)), \
            [first.clone() for _ in range(layers)], [second.clone() for _ in range(layers)]
    raise ValueError(name)


def run_native_greedy_matrix() -> dict[str, Any]:
    """Run APPEND/REPLACE/SKIP and repeated-slot cases using native V0.2 code."""
    try:
        from experiments.p0_protocol_audit import require_sources
        from experiments.v02_memory import greedy_slot
        from experiments.risk_safe.native_bridge import (
            extract_v22_candidates, normalized_native_candidate_rows)

        impl = require_sources()
        current_case = "initialization"
        # The locked concat profile exposes one native fused codebook layer.
        dim, n_layers = 256, 1
        protected = [f"fixture-c64:{i:02d}" for i in range(64)]
        static_map = {"per_layer_summary": {
            f"layer{i}": {"topk_patch_idx": list(range(8))}
            for i in range(n_layers)}}
        rows = []
        all_ok = True

        def run_case(case: str, initial_layers, event_features):
            nonlocal all_ok
            nonlocal current_case
            current_case = case
            direct_bank = _new_native_bank(impl, layers=n_layers, dim=dim)
            writer_bank = _new_native_bank(impl, layers=n_layers, dim=dim)
            direct_arm = _make_arm(impl, direct_bank)
            writer = SharedNativeGreedyWriter(
                impl, writer_bank, theta_ref=1.0,
                protected_source_keys=protected, sigma_ref=0.02)
            source_rows: list[list[str]] = []
            allowed: dict[str, set[str]] = {}
            current_version = 0
            if initial_layers is not None:
                seeded_layers = tuple(x.to(dtype=torch.float16).clone()
                                      for x in initial_layers)
                source_rows = [[f"seed:{layer}:{row}" for row in range(len(tensor))]
                               for layer, tensor in enumerate(seeded_layers)]
                allowed.update(_install_fixture_state(
                    writer, writer_bank, seeded_layers, source_rows, version=5))
                for cb, tensor in zip(dynamic_codebooks(direct_bank), seeded_layers):
                    cb.protos = tensor.clone()
                current_version = 5
            feature_runs = event_features if isinstance(event_features, list) and \
                event_features and isinstance(event_features[0], list) else [event_features]
            prior_slots: list[list[int]] = []
            results = []
            for event_no, feature_layers in enumerate(feature_runs):
                feature_tensors = [x.to(dtype=torch.float32).contiguous()
                                   for x in feature_layers]
                before_support = _fixed_bank_digest(writer_bank)
                before_bank = snapshot(writer_bank, version=current_version)
                chosen, indices = extract_v22_candidates(
                    impl, static_map, feature_tensors, keep=8)
                normalized = normalized_native_candidate_rows(
                    impl, writer_bank, feature_tensors, indices, keep=8)
                proposal, expected_audit = before_bank.proposed(
                    normalized, already_normalized=True)
                source_key = f"fixture:{case}:event:{event_no}"
                for layer, candidate in enumerate(normalized):
                    stored = candidate.to(dtype=before_bank.layers[layer].dtype)
                    allowed.setdefault(f"{source_key}\0{layer}", set()).update(
                        _tensor_sha(value) for value in stored)

                direct_arm.update(
                    impl, feature_tensors, indices, step=event_no,
                    score=0.0, theta=1.0, sigma=0.02)
                direct_after = snapshot(direct_bank,
                                        version=proposal.version)
                if direct_after.sha256() != proposal.sha256():
                    raise AssertionError("MemoryArm.update differs from GreedyBank.proposed")

                write = writer.apply(
                    queried=True, label=0, source_key=source_key, score=0.0,
                    static_amap=static_map, full_features=feature_tensors,
                    step=event_no)
                if write.appended != expected_audit.appended or \
                        write.replaced != expected_audit.replaced or \
                        write.skipped != expected_audit.skipped:
                    raise AssertionError("Shared writer operation counts differ from native proposal")
                direct_digests = [
                    _tensor_sha(row) for layer in direct_after.layers for row in layer]
                writer_digests = [
                    _tensor_sha(row) for layer in writer.bank.layers for row in layer]
                if direct_digests != writer_digests or direct_after.version != writer.bank.version:
                    raise AssertionError("Native live, shadow, and shared writer states differ")
                if _fixed_bank_digest(writer_bank) != before_support:
                    raise AssertionError("Fixed support changed during a Greedy operation")
                source_audit = _audit_source_contents(writer, allowed)

                operation = ("APPEND" if write.appended else
                             "REPLACE" if write.replaced else "SKIP")
                if case == "repeated_replace_same_slot":
                    # The deliberately close pair makes row 0 the unique first
                    # min-redundancy slot. Confirm its source after each event.
                    target = [i for i, key in enumerate(writer.sources[0])
                              if key == source_key]
                    if write.replaced and target != [0]:
                        raise AssertionError(f"Expected repeated replacement of row 0, got {target}")
                    prior_slots.append(target)
                per_layer = []
                for layer_idx, (old_layer, new_layer) in enumerate(
                        zip(before_bank.layers, writer.bank.layers)):
                    n_source = writer.sources[layer_idx].count(source_key)
                    n_added = max(0, len(new_layer) - len(old_layer))
                    per_layer.append({
                        "layer": layer_idx,
                        "append": n_added,
                        "replace": n_source - n_added,
                        "skip": 8 - n_source,
                    })
                results.append({
                    "fixture_case": case,
                    "operation": operation,
                    "counts": {"append": write.appended,
                               "replace": write.replaced,
                               "skip": write.skipped},
                    "memory_version_before": before_bank.version,
                    "memory_version_after": writer.bank.version,
                    "memory_content_changed": before_bank.content_sha256() !=
                        writer.bank.content_sha256(),
                    "dtype": str(writer.bank.layers[0].dtype),
                    "layer_shapes": [list(t.shape) for t in writer.bank.layers],
                    "exact_native_memoryarm_shadow_shared_writer": True,
                    "numeric_tolerance": {"atol": 0.0, "rtol": 0.0,
                                           "comparison": "torch.equal"},
                    "static_support_sha256": before_support,
                    "memory_sha256": writer.bank.sha256(),
                    "source_attribution_audit": source_audit,
                    "source_row_counts": list(writer.source_row_counts(source_key)),
                    "counts_per_layer": per_layer,
                    "source_map_sha256": writer.source_map_sha256(),
                    "source_revision": writer.source_revision,
                    "replaced_same_slot_row_indices": target if case ==
                        "repeated_replace_same_slot" else [],
                })
                current_version = writer.bank.version
            if case == "repeated_replace_same_slot" and prior_slots != [[0], [0]]:
                raise AssertionError(f"Expected row 0 replaced twice, got {prior_slots}")

            checkpoint = writer.state_dict()
            restored_native = _new_native_bank(impl, layers=n_layers, dim=dim)
            if _fixed_bank_digest(restored_native) != writer.fixed_support_sha256:
                raise AssertionError("Checkpoint target static support differs")
            restored_writer = SharedNativeGreedyWriter(
                impl, restored_native, theta_ref=writer.theta_ref,
                protected_source_keys=protected, sigma_ref=writer.sigma_ref)
            restored_writer.load_state_dict(checkpoint)
            if (restored_writer.bank.sha256() != writer.bank.sha256() or
                    restored_writer.sources != writer.sources or
                    restored_writer.source_map_sha256() != writer.source_map_sha256()):
                raise AssertionError("Native shared writer checkpoint did not restore exactly")
            results[-1]["checkpoint_restore"] = "PASS"

            # Exercise the production ledger's all-layer LSO filter against
            # actual post-replacement prototype tensors; score computation is
            # deliberately not substituted for native scoring here.
            if case == "repeated_replace_same_slot":
                from .anchor_ledger import AnchorLedger
                excluded, removed = AnchorLedger._without_source(
                    writer.bank, writer.sources, "fixture:repeated_replace_same_slot:event:1")
                expected_removed = n_layers
                if removed != expected_removed:
                    raise AssertionError(
                        f"LSO must remove the current source rows in all layers; got {removed}")
                if any(tuple(x.shape) != (255, dim) for x in excluded.layers):
                    raise AssertionError("LSO filtered shape differs after repeated replacement")
                lso = {"removed_rows_all_layers": removed,
                       "filtered_layer_shapes": [list(x.shape) for x in excluded.layers],
                       "memory_version": excluded.version,
                       "source_map_checked_against_prototype_bytes": True,
                       "status": "PASS"}
            else:
                lso = {"status": "NOT_APPLICABLE"}
            all_ok = all_ok and all(x["exact_native_memoryarm_shadow_shared_writer"]
                                    for x in results)
            return results, lso

        # Empty bank: eight normalized rows append independently in both layers.
        initial, candidates, _ = _fixture("append", dim, n_layers)
        append_rows, _ = run_case("append", initial, candidates)

        # Saturated duplicate bank makes the native min-gap decision replace;
        # a complete orthogonal bank plus an existing vector rejects the same
        # selected row through the original greedy_slot predicate.
        initial, candidates, _ = _fixture("replace", dim, n_layers)
        replace_rows, _ = run_case("replace", initial, candidates)
        initial, candidates, _ = _fixture("skip", dim, n_layers)
        skip_rows, _ = run_case("skip", initial, candidates)

        initial_gen, first, second = _fixture("repeated_replace_same_slot", dim, n_layers)
        initial = tuple(initial_gen)
        repeated_rows, lso = run_case(
            "repeated_replace_same_slot", initial, [first, second])

        cases = [*append_rows, *replace_rows, *skip_rows, *repeated_rows]
        expected_case_counts = {
            "append": [{"append": 8, "replace": 0, "skip": 0}],
            "replace": [{"append": 0, "replace": 8, "skip": 0}],
            "skip": [{"append": 0, "replace": 0, "skip": 8}],
            "repeated_replace_same_slot": [
                {"append": 0, "replace": 1, "skip": 7},
                {"append": 0, "replace": 1, "skip": 7}],
        }
        for fixture_case, expected_rows in expected_case_counts.items():
            observed = [row["counts"] for row in cases
                        if row["fixture_case"] == fixture_case]
            if observed != expected_rows:
                raise AssertionError(
                    f"Native {fixture_case} matrix counts did not match fixture: {observed}")
        return {
            "schema": "calibration.stage0.native_greedy_parity_matrix.v1",
            "status": "PASS" if all_ok else "BLOCKED",
            "scope": "synthetic_feature_operator_diagnostic; not an image scorer",
            "native_sources": {
                "memory_update": "experiments.v02_memory.MemoryArm.update",
                "replacement": "experiments.v02_memory.greedy_slot",
                "candidate_normalization": "native_bridge.normalized_native_candidate_rows",
                "shared_writer": "experiments.calibration.native.SharedNativeGreedyWriter",
                "codebook": "test_phase3_AIF_simplified_v22.PatchVMB",
            },
            "device": "cpu", "capacity_per_layer": 256,
            "layer_count": n_layers, "native_layer_profile": "feature_fuse=concat",
            "candidate_rows_per_layer": 8,
            "tolerance": {"atol": 0.0, "rtol": 0.0,
                          "exact": "torch.equal; FP16 prototype bytes"},
            "operation_counts": {
                "APPEND": sum(row["counts"]["append"] for row in cases),
                "REPLACE": sum(row["counts"]["replace"] for row in cases),
                "SKIP": sum(row["counts"]["skip"] for row in cases),
            },
            "static_support_immutability": "PASS",
            "source_attribution_vs_actual_prototype_contents": "PASS",
            "same_slot_replacement_source_audit": {
                "status": "PASS",
                "slot_indices": [row["replaced_same_slot_row_indices"]
                                 for row in repeated_rows],
                "source_after_second_replacement": "fixture:repeated_replace_same_slot:event:1",
            },
            "source_aware_lso_after_replacement": lso,
            "memory_version_policy": "advance once per operation iff stored prototype bytes changed",
            "matrix": cases,
            "production_native_v22_scoring": "covered separately by bounded CUDA smoke/replay",
        }
    except Exception as exc:
        return {"schema": "calibration.stage0.native_greedy_parity_matrix.v1",
                "status": "BLOCKED",
                "scope": "synthetic_feature_operator_diagnostic; not an image scorer",
                "reason": f"{type(exc).__name__}: {exc}",
                "case": locals().get("current_case", "unknown")}
