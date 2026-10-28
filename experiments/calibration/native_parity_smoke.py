"""Bounded, non-scientific smoke for the historical native score/write bridge.

Uses only audiojack train metadata: four frozen D2 supports, the separately
selected 64 additional protected C64 normals (identities only), and one other
verified train normal as a disposable Greedy-update fixture. It never reads a
test image, mask, confirmation class, or confirmation outcome. The update uses
a synthetic threshold strictly above the fixture score solely to exercise the
verified-normal write branch; no calibration or performance result is emitted.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import subprocess
from typing import Any

import torch

from .native import (NativeScoreService, _fixed_bank_digest,
                     isolate_native_bank_k256)
from .profiles import DEFAULT_PROFILE


def _opaque_digest(domain: str, values: set[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path.rglob("*.py"), key=lambda item: item.relative_to(path).as_posix())
    for source in files:
        relative = source.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(source)))
    return digest.hexdigest()


def run_native_parity_smoke(dataset_root: str | Path,
                            json_root: str | Path) -> dict[str, Any]:
    """Run actual v22 scorer and Greedy update parity, or report an honest skip."""
    root, json_dir = Path(dataset_root), Path(json_root)
    if not torch.cuda.is_available():
        return {"status": "SKIP", "reason": "CUDA is unavailable"}
    if not root.is_dir() or not json_dir.is_dir():
        return {"status": "SKIP", "reason": "Real-IAD metadata roots are absent",
                "dataset_root_present": root.is_dir(),
                "json_root_present": json_dir.is_dir()}
    hub = Path(torch.hub.get_dir())
    checkpoint = hub / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
    hub_source = hub / "facebookresearch_dinov2_main"
    if not checkpoint.is_file() or not hub_source.is_dir():
        return {"status": "SKIP",
                "reason": "Cached DINOv2 checkpoint/source is absent; network download is disabled",
                "checkpoint_present": checkpoint.is_file(),
                "hub_source_present": hub_source.is_dir()}

    try:
        from experiments.p0_protocol_audit import (
            Config, build_vmb, encode_query, encode_reference_features,
            require_sources,
        )
        from experiments.risk_safe.native_bridge import (
            native_score, snapshot, verify_native_update_parity,
            verify_untouched_parity,
        )
        from experiments.risk_safe.preflight import load_class_records
        from experiments.v02_architecture_core import MemorySettings
        from experiments.v02_memory import MemoryArm
        from .real_iad_preflight import _select_support_and_c64

        train, _ = load_class_records(root, json_dir, "audiojack")
        supports, c64, support_keys, c64_keys = _select_support_and_c64(train)
        excluded = set(support_keys) | set(c64_keys)
        extra = next((row for row in train
                      if row.is_good and row.sample_key not in excluded), None)
        if extra is None:
            return {"status": "BLOCKED",
                    "reason": "No additional verified train normal is available for the update fixture"}

        scorer = DEFAULT_PROFILE.scorer
        cfg = Config(
            device="cuda", augmentation_seed=scorer.augmentation_seed,
            bank_seed=scorer.bank_seed, fixed_capacity=scorer.fixed_capacity,
            fixed_per_image=scorer.fixed_per_image,
            augmentations=scorer.augmentations,
            include_identity=scorer.include_identity,
            multiplier=scorer.multiplier, feature_fuse=scorer.feature_fuse,
            topk_patches=scorer.topk_patches,
            image_aggregation=scorer.image_aggregation,
            image_topk=scorer.image_topk,
            image_quantile=scorer.image_quantile, knn=scorer.knn,
        )
        impl = require_sources()
        # Match the historical P0 math-SDPA reproducibility mode so native
        # support features do not depend on a memory-efficient attention kernel.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        impl.set_seed(0)
        backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
        cache = encode_reference_features(
            impl, backbone, [row.path for row in supports], cfg)
        native_bank = build_vmb(impl, cache,
                                [row.path for row in supports], cfg)
        fixed_sha = _fixed_bank_digest(native_bank)
        native_k256, fixed_k256_sha = isolate_native_bank_k256(native_bank)
        if fixed_sha != fixed_k256_sha:
            raise RuntimeError("K=256 isolation changed the static support digest")

        image, features = encode_query(impl, backbone, extra.path, cfg)
        score_parity = verify_untouched_parity(
            impl, native_k256, image, features, cfg)
        shadow = snapshot(native_k256)
        service = NativeScoreService(impl, native_k256, cfg, scorer,
                                     fixed_k256_sha)
        adapted = service.score((image, features), shadow)
        direct_score, amap = native_score(
            impl, native_k256, image, features, cfg)
        adapter_delta = abs(direct_score - adapted.score)
        if adapter_delta > 1e-8:
            raise RuntimeError(
                f"NativeScoreService score differs from direct v22 by {adapter_delta}")

        # Disposable bank only. theta_fixture forces the registered training
        # normal through the allow-write branch to test exact fp16 Greedy parity.
        disposable = copy.deepcopy(native_k256)
        settings = MemorySettings(stm=256, ltm=0, buffer=0, flat=256,
                                  cap_bytes=2_000_000_000)
        update_arm = MemoryArm("greedy_coreset", disposable, settings, seed=0)
        theta_fixture = direct_score + max(1.0, abs(direct_score))
        update_parity = verify_native_update_parity(
            impl, update_arm, features, amap, score=direct_score,
            theta=theta_fixture, sigma=0.02, step=0)
        if _fixed_bank_digest(native_k256) != fixed_k256_sha:
            raise RuntimeError("Disposable Greedy update mutated protected static support")
        try:
            hub_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=hub_source,
                capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            hub_commit = None
        return {
            "schema": "calibration.stage0.native_parity_smoke.v1",
            "status": "PASS",
            "scope": "audiojack_train_only_non_scientific_parity_smoke",
            "device": "cuda",
            "torch_version": str(torch.__version__),
            "model": scorer.model_name,
            "dinov2_hub_source_commit": hub_commit,
            "dinov2_hub_source_tree_sha256": _source_tree_sha256(hub_source),
            "checkpoint_sha256": _file_sha256(checkpoint),
            "support_count": len(supports),
            "additional_protected_c64_count": len(c64),
            "support_c64_official_groups_disjoint": not bool(
                set(support_keys) & set(c64_keys)),
            "support_group_sha256": _opaque_digest(
                "support", set(support_keys)),
            "c64_group_sha256": _opaque_digest("c64", set(c64_keys)),
            "update_fixture_group_sha256": _opaque_digest(
                "update-fixture", {extra.sample_key}),
            "dynamic_capacity_per_layer": [int(cb.K) for cb in
                __import__("experiments.risk_safe.native_bridge",
                           fromlist=["dynamic_codebooks"]).dynamic_codebooks(native_k256)],
            "static_support_sha256": fixed_k256_sha,
            "live_native_vs_untouched_clone_score_abs_delta":
                score_parity["abs_difference"],
            "direct_native_vs_calibration_adapter_score_abs_delta": adapter_delta,
            "native_greedy_update": {
                "passed": update_parity["passed"],
                "equal_each_layer": update_parity["equal_each_layer"],
                "keep_selected": update_parity["keep_selected"],
                "greedy_operator_counts": update_parity["greedy"],
                "comparison": update_parity["comparison"],
            },
            "synthetic_update_threshold_policy":
                "fixture_score + max(1, abs(fixture_score)); diagnostic only",
            "absolute_image_scores_emitted": False,
            "test_or_confirmation_images_opened": False,
            "stage1_or_stage2_run": False,
        }
    except Exception as exc:
        return {"schema": "calibration.stage0.native_parity_smoke.v1",
                "status": "BLOCKED",
                "reason": f"{type(exc).__name__}: {exc}",
                "test_or_confirmation_images_opened": False,
                "stage1_or_stage2_run": False}
