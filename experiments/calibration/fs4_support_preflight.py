"""Explicit four-Support TRAIN-only engineering preflight, never a campaign.

Requires the exact committed source revision as an argument. Only the four
seed-0 image paths recorded in the prior frozen Support receipt are opened for
each authorized category. No dataset JSON, stream images, outcomes, or masks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from .fs4_hardening_audit import ROOT, SOURCES, PROTECTED_SOURCES, audit
from .fs4_threshold_integration import FEATURE_CONVENTION, build_native_from_four_support

CATEGORIES = ("audiojack", "regulator", "switch", "rolled_strip_base")
DATASET = Path(r"G:\dataset\realiad\images")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(expected_commit: str) -> dict:
    from experiments.calibration import stage05, profiles, r1_matched_correction_pilot as r1, r3_fullstream_2x2 as r3
    from experiments import p0_protocol_audit as p0
    from experiments.risk_safe.preflight import parse_image_name
    from experiments.risk_safe.native_bridge import dynamic_codebooks
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if head != expected_commit or len(head) != 40:
        raise RuntimeError("explicit source-freeze commit does not equal HEAD")
    paths_to_check = [*SOURCES, *PROTECTED_SOURCES,
        "experiments/calibration/fs4_stochastic_bea.py", "experiments/calibration/fs4_threshold_core.py",
        "experiments/calibration/fs4_threshold_audit.py", "experiments/calibration/fs4_threshold_synthetic_smoke.py",
        "tests/calibration/test_fs4_threshold.py", "tests/calibration/test_fs4_hardening_r1.py",
        "tests/calibration/test_fs4_hardening_cuda.py"]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *paths_to_check], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout
    if dirty:
        raise RuntimeError("source freeze has tracked or untracked modifications")
    audit()
    source_hashes = {name: sha(ROOT / name) for name in paths_to_check}
    frozen_path = ROOT / "outputs/calibration/stage05_exploratory_20261029/frozen_inputs.json"
    score_manifest = ROOT / "outputs/calibration/stage05_exploratory_20261029/score_tape_manifest.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))["classes"]
    receipt_manifest = json.loads(score_manifest.read_text(encoding="utf-8"))["classes"]
    selected: dict[str, dict] = {}
    for category in CATEGORIES:
        info = frozen[category]; support = info["supports"]["0"]
        role_path = ROOT / "outputs/calibration/a_calibration_scale_20261106_a1_full01/a1" / category / "role_manifest_private.json"
        role = json.loads(role_path.read_text(encoding="utf-8"))
        groups, images = support["groups"], support["images"]
        group_digest = stage05.opaque_digest("support-groups", groups)
        if (len(groups) != 4 or len(set(groups)) != 4 or len(images) != 4 or len(set(images)) != 4 or
                not support["all_train_normal"] or groups != role["support_groups"]["0"] or
                group_digest != support["group_sha256"] or
                group_digest != receipt_manifest[category]["score_tapes"]["0"]["support_group_sha256"] or
                stage05.opaque_digest("support-images", images) != support["image_sha256"]):
            raise RuntimeError(f"{category}: independent Support receipt identity mismatch")
        paths = [(DATASET / info["prefix"] / name).resolve() for name in images]
        for path, group, image in zip(paths, groups, images):
            if not path.is_relative_to((DATASET / category).resolve()) or not path.is_file():
                raise RuntimeError(f"{category}: registered Support image is unavailable")
            parsed = parse_image_name(path, expected_class=category)
            if parsed.official_sample_key != group or parsed.rel_path != image:
                raise RuntimeError(f"{category}: Support image/group identity mismatch")
        selected[category] = {"groups": groups, "paths": paths, "group_digest": group_digest,
                              "image_digest": support["image_sha256"], "role_receipt_sha256": sha(role_path),
                              "expected_fixed": receipt_manifest[category]["score_tapes"]["0"]["fixed_support_sha256"]}
    receipt = r3._get_verified_model_receipt()
    impl = p0.require_sources()
    restore_hub, model_profile = r3._runtime_and_native_setup(torch, impl, receipt)
    torch.cuda.reset_peak_memory_stats()
    try:
        cfg, _ = r3._native_config(p0, profiles, r1)
        backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
        model_sha = r1._model_state_sha256(backbone.model)
        if model_sha != receipt["loaded_model_state_sha256"]:
            raise RuntimeError("loaded model state does not match frozen model receipt")
        restore_hub()
        output = {}
        for category in CATEGORIES:
            item = selected[category]; paths = [str(p) for p in item["paths"]]
            augmentations = p0.encode_reference_features(impl, backbone, paths, cfg)
            identities, cls_features = [], []
            for path in paths:
                image = impl.load_rgb(path)
                features, cls = impl.encode_pil(backbone, image, cfg.feature_fuse)
                identities.append((image, features[0]))
                cls_features.append(cls.detach().float().cpu().numpy().reshape(-1))
            groups = [hashlib.sha256(f"official-sample-group\0{x}".encode()).hexdigest() for x in item["groups"]]
            runtime, reference, _coverage = build_native_from_four_support(impl=impl, cfg=cfg,
                support_paths=paths, support_groups=groups, augmentation_cache=augmentations,
                identity_images_and_features=identities, support_cls=np.stack(cls_features),
                feature_convention=FEATURE_CONVENTION)
            bank = runtime.bank
            fixed_evidence = r3._check_bank_receipt(bank, p0, item["expected_fixed"], r1)
            book = bank.normal_fixed["p0"][0]["id"][0]
            if tuple(book.protos.shape) != (8192, 4096) or book.protos.dtype != torch.float16:
                raise RuntimeError(f"{category}: fixed bank geometry/dtype mismatch")
            if runtime.correction_cb.size() or any(cb.size() for cb in dynamic_codebooks(bank)):
                raise RuntimeError(f"{category}: nonempty initial native memories")
            for i, (image, features) in enumerate(identities):
                value = runtime.score(group=groups[i], position=0, image=image, features=features)
                if not np.isclose(value["G"], reference["support_scores"][i], atol=1e-5, rtol=1e-4) or value["GC"] != value["G"]:
                    raise RuntimeError(f"{category}: Support native scoring parity failed")
            output[category] = {"status": "PASS", "support_groups_sha256": item["group_digest"],
                "support_images_sha256": item["image_digest"], "support_image_content_sha256": [sha(p) for p in item["paths"]],
                "support_role_receipt_sha256": item["role_receipt_sha256"], "fixed_shape": list(book.protos.shape),
                "fixed_dtype": str(book.protos.dtype), "fixed_evidence": fixed_evidence,
                "cold_start": "four native self-scores; G=GC and C1 empty"}
            del runtime, bank, augmentations, identities, cls_features
            torch.cuda.empty_cache()
        return {"schema": "FS4_SUPPORT_PREFLIGHT_V1", "status": "PASS", "source_commit": head,
                "source_sha256": source_hashes, "frozen_input_receipt_sha256": sha(frozen_path),
                "fixed_receipt_manifest_sha256": sha(score_manifest), "model": model_profile,
                "loaded_model_state_sha256": model_sha, "categories": output,
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "additional_normal_images_read": 0, "test_images_read": 0, "test_outcomes_read": 0,
                "masks_read": 0, "scientific_metrics_computed": 0}
    finally:
        restore_hub()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.source_commit)
    data = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with args.manifest.open("xb") as out:
        out.write(data); out.flush(); os.fsync(out.fileno())
    print(f"TRAIN-only Support engineering parity: {result['status']}; manifest={args.manifest}")


if __name__ == "__main__":
    main()
