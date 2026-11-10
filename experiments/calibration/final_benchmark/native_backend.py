"""Source-bound native per-category backend for the frozen four-arm design.

Inputs are already registered Support/C64 identities and encoded commissioning
features. Real execution and GPU parity require a separate authorization.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from .runner import PRINCIPAL_ARMS, ScorePacket, static_arms_for_condition
from .static_baselines import (build_static4_native, build_static36_native,
                               ensure_historical_memory_import_path)


class NativeBackendError(RuntimeError):
    pass


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


class NativeInputEncoder:
    """The historical frozen DINO/preprocessing path, shared across all arms."""

    def __init__(self, *, impl: Any, p0: Any, backbone: Any, cfg: Any,
                 model_receipt: Mapping[str, Any]):
        self.impl, self.p0, self.backbone, self.cfg = impl, p0, backbone, cfg
        self.model_receipt = dict(model_receipt)

    @classmethod
    def load_authorized_runtime(cls) -> "NativeInputEncoder":
        """Load locally verified weights only when a later run is authorized."""
        inherited_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
        ensure_historical_memory_import_path()
        import torch
        from experiments import p0_protocol_audit as p0
        from experiments.calibration import profiles
        from experiments.calibration import r1_matched_correction_pilot as r1
        from experiments.calibration import r3_fullstream_2x2 as r3
        r3._restore_cuda_visibility(inherited_visibility)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise NativeBackendError("native runtime requires one separately authorized CUDA device")
        impl = p0.require_sources()
        model_receipt = r3._get_verified_model_receipt()
        restore, _runtime = r3._runtime_and_native_setup(torch, impl, model_receipt)
        try:
            backbone = impl.DINOv2MultiLayerBackbone(device="cuda")
        finally:
            restore()
        if r1._model_state_sha256(backbone.model) != model_receipt.get("loaded_model_state_sha256"):
            raise NativeBackendError("native backbone differs from the verified local model receipt")
        cfg, _profile = r3._native_config(p0, profiles, r1)
        return cls(impl=impl, p0=p0, backbone=backbone, cfg=cfg, model_receipt=model_receipt)

    def encode_path(self, image_path: str) -> Any:
        import torch
        from experiments.calibration.e6_bea_closed_loop import EncodedEvent
        started = time.perf_counter()
        image = self.impl.load_rgb(image_path)
        features, cls = self.impl.encode_pil(self.backbone, image, self.cfg.feature_fuse)
        if (len(features) != 1 or tuple(features[0].shape) != (784, 4096) or
                features[0].dtype != torch.float32 or not bool(torch.isfinite(features[0]).all().item())):
            raise NativeBackendError("native DINO descriptor violates the frozen feature contract")
        cls_value = torch.as_tensor(cls).detach().float().reshape(-1).cpu().numpy().copy()
        if not len(cls_value) or not np.isfinite(cls_value).all():
            raise NativeBackendError("native DINO CLS descriptor is missing or invalid")
        return EncodedEvent(image=image, feature_layers=features, global_cls=cls_value,
                            encoding_seconds=time.perf_counter() - started)

    def encode_registered_realiad_event(self, *, record: Mapping[str, Any],
                                        locator: Mapping[str, str], category: str,
                                        dataset_release_sha256: str, stream_seed: int) -> Any:
        """Bind one frozen TEST identity to its bytes before the shared encoding."""
        from experiments.risk_safe.preflight import parse_image_name
        from . import identity
        from .runner import TestEvent
        if (not re.fullmatch(r"[0-9a-f]{64}", str(dataset_release_sha256)) or
                isinstance(stream_seed, bool) or stream_seed not in (0, 11)):
            raise NativeBackendError("registered stream release or seed is invalid")
        image_key, group_key = locator["official_image_key"], locator["official_group_key"]
        parsed_group, camera = identity._realiad_identity_key(category, image_key)
        path = Path(locator["image_path"])
        parsed_path = parse_image_name(path, expected_class=category)
        ordinal = record.get("within_group_ordinal")
        event_id = identity._domain_hash(
            "Gate0-event-v1", dataset_release_sha256, category, str(stream_seed),
            group_key, str(camera), image_key, str(ordinal))
        if (parsed_group != group_key or parsed_path.official_sample_key != group_key or
                parsed_path.camera_id != camera or ordinal != camera or
                record.get("camera_id") != camera or record.get("event_id_sha256") != event_id or
                record.get("image_identity_sha256") != identity._image_identity_digest(image_key) or
                record.get("source_group_sha256") != identity._source_group_digest(group_key)):
            raise NativeBackendError("registered TEST identity does not match its official image/group/view")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != record.get("image_content_sha256"):
            raise NativeBackendError("registered TEST image bytes differ from the frozen identity")
        encoded = self.encode_path(str(path))
        return TestEvent(position=record["position"], event_id_sha256=event_id,
                         image_identity_sha256=record["image_identity_sha256"],
                         group_sha256=record["source_group_sha256"], camera_id=camera,
                         feature_ref=encoded, encoding_seconds=encoded.encoding_seconds)

    def prepare_commissioning(self, *, role_manifest: Mapping[str, Any],
                              private_locators: Mapping[str, Mapping[str, str]],
                              condition_id: str) -> dict[str, Any]:
        """Verify frozen identities before encoding only their 4+64 TRAIN normals."""
        from . import identity
        if role_manifest.get("dataset_id") != "realiad":
            raise NativeBackendError("VisA commissioning remains disabled pending its registered group exception")
        from experiments.risk_safe.preflight import parse_image_name
        rows = role_manifest.get("selected")
        release = str(role_manifest.get("dataset_release_sha256", ""))
        seed = {"P0": 0, "P1": 0, "P2": 1}.get(condition_id)
        if (seed is None or role_manifest.get("support_seed") != seed or
                not re.fullmatch(r"[0-9a-f]{64}", release) or
                not isinstance(rows, list) or len(rows) != 68 or
                [row.get("role") for row in rows] != ["SUPPORT"] * 4 + ["C64"] * 64):
            raise NativeBackendError("frozen Support/C64 role manifest is incomplete")
        ids = [str(row["image_identity_sha256"]) for row in rows]
        groups = [str(row["source_group_sha256"]) for row in rows]
        if (len(set(ids)) != 68 or len(set(groups)) != 68 or set(ids) != set(private_locators) or
                len({row["image_content_sha256"] for row in rows}) != 68):
            raise NativeBackendError("commissioning image or group identities are not disjoint")
        records = {}
        for row in rows:
            image_id = str(row["image_identity_sha256"])
            loc = private_locators[image_id]
            official_image, official_group = loc["official_image_key"], loc["official_group_key"]
            if (identity._image_identity_digest(official_image) != image_id or
                    identity._source_group_digest(official_group) != row["source_group_sha256"]):
                raise NativeBackendError("private locator does not match its frozen official identity")
            path = Path(loc["image_path"])
            parsed = parse_image_name(path, expected_class=str(role_manifest["category"]))
            if (parsed.official_sample_key != official_group or
                    parsed.rel_path != str(loc["relative_id"]).replace("\\", "/")):
                raise NativeBackendError("A4 candidate identity differs from the official path/group convention")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != row["image_content_sha256"]:
                raise NativeBackendError("commissioning image content differs from its manifest")
            records[image_id] = {"image_identity_sha256": image_id,
                "source_group_sha256": row["source_group_sha256"],
                "group_sha256": row["source_group_sha256"], "image_path": str(path),
                "relative_id": str(loc["relative_id"])}
        support_paths = [records[image_id]["image_path"] for image_id in ids[:4]]
        support_cache = self.p0.encode_reference_features(self.impl, self.backbone,
                                                           support_paths, self.cfg)
        support_cls = np.stack([self.encode_path(path).global_cls for path in support_paths])
        c64_items = []
        for image_id in ids[4:]:
            encoded = self.encode_path(records[image_id]["image_path"])
            c64_items.append({"image_identity_sha256": image_id,
                "source_group_sha256": records[image_id]["source_group_sha256"],
                "rgb": np.asarray(encoded.image.convert("RGB"), dtype=np.uint8).copy(),
                "features": [value.detach().to("cpu").contiguous().numpy().copy()
                             for value in encoded.feature_layers]})
        result = {"support_paths": support_paths, "support_group_sha256": groups[:4],
                  "support_image_sha256": ids[:4],
                  "support_cache": support_cache, "support_cls": support_cls,
                  "c64_items": c64_items, "c64_group_sha256": _hash(groups[4:])}
        if condition_id == "P0":
            schedule = identity.build_static36_schedule(
                role_manifest, dataset_release_sha256=release,
                category=str(role_manifest["category"]), support_seed=0)
            c64_by_id = {item["image_identity_sha256"]: item for item in c64_items}
            result.update({"static36_schedule": schedule,
                "static36_reference_items": [records[image_id] for image_id in schedule["reference_identity_sha256"]],
                "static36_c32_items": [c64_by_id[image_id] for image_id in schedule["c32_identity_sha256"]]})
        return result

    def build_backend(self, *, category: str, condition_id: str,
                      commissioning: Mapping[str, Any]) -> "NativeFourArmBackend":
        return NativeFourArmBackend(category=category, condition_id=condition_id,
            impl=self.impl, p0=self.p0, backbone=self.backbone, cfg=self.cfg, **dict(commissioning))


class NativeFourArmBackend:
    """Four independent native banks over one shared encoded image stream."""

    def __init__(self, *, category: str, condition_id: str, impl: Any, p0: Any,
                 backbone: Any, cfg: Any, support_paths: Sequence[str],
                 support_group_sha256: Sequence[str], support_image_sha256: Sequence[str],
                 support_cache: Mapping[str, Any],
                 support_cls: Any, c64_items: Sequence[Mapping[str, Any]],
                 c64_group_sha256: str, static36_schedule: Mapping[str, Any] | None = None,
                 static36_reference_items: Sequence[Mapping[str, Any]] | None = None,
                 static36_c32_items: Sequence[Mapping[str, Any]] | None = None):
        inherited_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
        ensure_historical_memory_import_path()
        import torch
        from experiments.calibration import r1_matched_correction_pilot as r1
        from experiments.calibration import r2_exp2a_c1_strength as exp2a
        from experiments.calibration import r3_fullstream_2x2 as r3
        from experiments.calibration import r6_capacity1024_audiojack_runner as r6
        from experiments.calibration import r7_shuffled_correction_4class as r7
        from experiments.calibration import r9_correction_mechanism_falsification as r9
        from experiments.calibration import r10_signed_knn_baseline as r10
        from experiments.calibration import profiles
        from experiments.calibration.controllers import StateConsistentQuantile
        from experiments.calibration.e6_bea_closed_loop import NormalCoverageMemory
        from experiments.calibration.native import _fixed_bank_digest
        from experiments.risk_safe.native_bridge import dynamic_codebooks
        from v02_memory import MemoryArm

        r3._restore_cuda_visibility(inherited_visibility)
        registered_static = static_arms_for_condition(condition_id)
        repository_root = Path(__file__).resolve().parents[3]
        if (Path(getattr(impl, "__file__", "")).resolve() !=
                (repository_root / "test_phase3_AIF_simplified_v22.py").resolve() or
                Path(getattr(p0, "__file__", "")).resolve() !=
                (repository_root / "experiments/p0_protocol_audit.py").resolve()):
            raise NativeBackendError("native implementation or preprocessing source is not the repository source")
        registered_cfg, _profile = r3._native_config(p0, profiles, r1)
        if asdict(cfg) != asdict(registered_cfg):
            raise NativeBackendError("native scorer configuration differs from the frozen profile")

        if (len(support_paths) != 4 or len(set(support_paths)) != 4 or
                len(support_group_sha256) != 4 or len(set(support_group_sha256)) != 4 or
                len(support_image_sha256) != 4 or len(set(support_image_sha256)) != 4 or
                len(c64_items) != 64 or set(support_paths) != set(support_cache)):
            raise NativeBackendError("four Support and 64 C64 identities are required")
        c64_groups = [str(item["source_group_sha256"]) for item in c64_items]
        c64_images = [str(item.get("image_identity_sha256", "")) for item in c64_items]
        if (len(set(c64_groups)) != 64 or set(c64_groups) & set(support_group_sha256) or
                any(not re.fullmatch(r"[0-9a-f]{64}", str(value))
                    for value in [*c64_groups, *support_group_sha256, *support_image_sha256,
                                  *c64_images, c64_group_sha256]) or
                len(set(c64_images)) != 64 or set(c64_images) & set(support_image_sha256)):
            raise NativeBackendError("Support/C64 source-group separation is invalid")
        self.category, self.condition_id = category, condition_id
        self.impl, self.p0, self.cfg = impl, p0, cfg
        self.r1, self.exp2a, self.r7, self.r9, self.r10 = r1, exp2a, r7, r9, r10
        self.c64_items = tuple(c64_items)
        self.protected_groups = frozenset(c64_groups) | frozenset(support_group_sha256)
        self.protected_images = frozenset(c64_images) | frozenset(support_image_sha256)
        self.c64_group_sha256 = c64_group_sha256
        self._states: dict[str, Any] = {}
        self._contexts: dict[tuple[str, str], Any] = {}
        self._context_group: str | None = None
        fixed_sha = None
        initial_scores = None
        initial_cache = None
        for arm in PRINCIPAL_ARMS:
            impl.set_seed(0)
            bank = p0.build_vmb(impl, support_cache, list(support_paths), cfg)
            r3._validate_correction_config(bank)
            if arm.startswith("F-"):
                corr_cfg, codebooks = r6._configure_r6_correction_capacity(bank)
                if len(codebooks) != 1 or codebooks[0].size() != 0:
                    raise NativeBackendError("native correction state must start empty at K1024")
                correction = codebooks[0]
            else:
                corr_cfg, correction = bank.corr_cfg, None
            writer = MemoryArm("greedy_coreset", bank,
                SimpleNamespace(flat=1024, stm=1024, ltm=0, buffer=0, min_events=2,
                                promote_every=16, cap_bytes=2 * 1024**3), seed=0)
            if any(int(cb.K) != 1024 or cb.size() != 0 for cb in dynamic_codebooks(bank)):
                raise NativeBackendError("native Greedy state must start empty at K1024")
            bank.use_defect_bank = False
            bank.corr_cfg.enabled = False
            profile = p0.fixed_bank_profile(bank)
            if (profile.get("total_prototypes") != 8192 or
                    profile.get("total_feature_bytes") != 64 * 1024 * 1024 or
                    len(profile.get("layers", ())) != 1 or
                    profile["layers"][0].get("feature_dim") != 4096 or
                    profile["layers"][0].get("dtype") != "float16"):
                raise NativeBackendError("native four-Support bank violates the 8192 x 4096 FP16 contract")
            actual_fixed = _fixed_bank_digest(bank)
            if fixed_sha is None:
                fixed_sha = actual_fixed
                initial_scores, initial_cache, _ = r7._score_c64_base(
                    impl, bank, self.c64_items, cfg, r1, category)
            if actual_fixed != fixed_sha:
                raise NativeBackendError("four native arms do not share the same frozen Support bank")
            knn = r10.SignedKNNFIFO(capacity=1024) if arm.startswith("K-") else None
            self._states[arm] = SimpleNamespace(
                bank=bank, writer=writer, correction_cfg=corr_cfg, correction=correction,
                sources={}, knn=knn, fixed_sha=fixed_sha, version=0,
                coverage=NormalCoverageMemory(support_cls), c64=list(initial_scores),
                base_cache=list(initial_cache), controller=None, detector_sha="", state_sha="")
        if len(initial_scores) != 64 or not np.isfinite(initial_scores).all():
            raise NativeBackendError("native commissioning requires 64 finite C64 scores")
        theta0 = float(np.quantile(np.asarray(initial_scores, dtype=np.float64), .90, method="higher"))
        sigma0 = max(float(np.std(initial_scores, ddof=1)), .02)
        for arm, state in self._states.items():
            state.theta0, state.sigma0 = theta0, sigma0
            state.detector_sha = self._detector_digest(arm)
            state.controller = StateConsistentQuantile(theta0, delta=.01, min_normals=16,
                                                        name=f"{arm}-B3")
            r7._initialize_b3(state.controller, state.c64, c64_group_sha256,
                              state.detector_sha, theta0)
            self._refresh_state_identity(arm)
        fb = self._states["F-B"]
        shadow_cfg = type(fb.correction_cfg)(**__import__("dataclasses").asdict(fb.correction_cfg))
        shadow_cfg.enabled = False
        shadow_cb = type(fb.correction)(1024, int(fb.correction.protos.shape[1]),
                                        fb.correction.device, shadow_cfg)
        self._shadow = SimpleNamespace(cfg=shadow_cfg, cb=shadow_cb, sources={}, version=0, sha="")
        self._refresh_shadow_identity()
        self._static = {"STATIC-4+64": build_static4_native(
            category=category, impl=impl, p0=p0, r1=r1, support_cache=support_cache,
            support_paths=support_paths, cfg=cfg, c64_items=self.c64_items,
            expected_fixed_sha256=fixed_sha)}
        self.static36_receipt = None
        if condition_id == "P0":
            if static36_schedule is None or static36_reference_items is None or static36_c32_items is None:
                raise NativeBackendError("P0 requires its registered Static-36+32 identity schedule")
            model, self.static36_receipt = build_static36_native(
                category=category, schedule=static36_schedule,
                reference_items=static36_reference_items, c32_items=static36_c32_items,
                impl=impl, p0=p0, r1=r1, backbone=backbone, cfg=cfg,
                support_cache=support_cache, support_paths=support_paths)
            self._static["STATIC-36+32"] = model
        if set(self._static) != set(registered_static):
            raise NativeBackendError("static bank set differs from the registered condition")

    def _detector_digest(self, arm: str) -> str:
        from experiments.calibration.exp_b_signed_feedback import state_digest
        state = self._states[arm]
        if state.knn is not None:
            return state_digest(state.fixed_sha, self.r1._memory_content_sha256(state.bank),
                                correction_sha=state.knn.state_sha256())
        return self.r9._state_digest_for_codebook(
            state.bank, state.fixed_sha, state.sources, state.correction_cfg,
            state.correction, self.exp2a, self.r1)

    def _refresh_state_identity(self, arm: str) -> None:
        state = self._states[arm]
        state.state_sha = _hash({"detector": state.detector_sha, "version": state.version,
                                 "threshold": float(state.controller.theta),
                                 "coverage": state.coverage.state_sha256()})

    def _refresh_shadow_identity(self) -> None:
        fb = self._states["F-B"]
        self._shadow.sha = self.r9._state_digest_for_codebook(
            fb.bank, fb.fixed_sha, self._shadow.sources, self._shadow.cfg,
            self._shadow.cb, self.exp2a, self.r1)

    def state_digest(self, arm_id: str) -> str:
        return self._states[arm_id].state_sha

    def snapshot(self, arm_id: str) -> str:
        return self.state_digest(arm_id)

    def _context(self, arm: str, event: Any) -> Any:
        try:
            return self._contexts[(arm, event.event_id_sha256)]
        except KeyError as exc:
            raise NativeBackendError("feedback lacks the sealed pre-group score context") from exc

    def score(self, arm_id: str, event: Any, snapshot: Any) -> ScorePacket:
        import torch
        from experiments.calibration.e6_bea_policy import compute_bea_components
        from experiments.calibration.exp_c_memory_evidence import d_band_utility
        state = self._states[arm_id]
        if (snapshot != state.state_sha or event.group_sha256 in self.protected_groups or
                event.image_identity_sha256 in self.protected_images):
            raise NativeBackendError("stale state or protected Support/C64 group in TEST stream")
        if self._context_group != event.group_sha256:
            self._contexts.clear()
            self._context_group = event.group_sha256
        encoded = event.feature_ref
        features = [value.to(state.bank.device) for value in encoded.feature_layers]
        if (len(features) != 1 or tuple(features[0].shape) != (784, 4096) or
                features[0].dtype != torch.float32):
            raise NativeBackendError("native TEST feature must have shape (784,4096)")
        base_map, top_ids, c0_score, summary = self.r1._native_dense_score(
            self.impl, state.bank, encoded.image, features, self.cfg)
        top_ids = np.asarray(top_ids, dtype=np.int64)
        if (top_ids.shape != (32,) or not np.array_equal(
                top_ids, np.asarray(summary["layer0"]["topk_patch_idx"], dtype=np.int64))):
            raise NativeBackendError("native C0 Top-32 differs from its source scoring receipt")
        selected = self.impl.l2_normalize(features[0])[torch.as_tensor(
            top_ids, device=state.bank.device, dtype=torch.long)]
        selected_np = selected.detach().float().cpu().numpy()
        if state.knn is None:
            delta, _ = self.exp2a._score_arm(
                self.impl, state.bank, arm_id, state.correction_cfg, state.correction,
                state.sources, selected, event.group_sha256, event.position)
            score = self.r1.max_score(self.r1.corrected_dense_map(base_map, top_ids, delta))
            correction_size = int(state.correction.size())
        else:
            score, delta, _ = self.r10._corrected_c2_score(
                base_map, top_ids, selected_np, state.knn, position=event.position,
                group_sha256=event.group_sha256)
            correction_size = state.knn.count
        novelty, _ = state.coverage.novelty_percentile(encoded.global_cls)
        layer = summary["layer0"]
        components = compute_bea_components(
            score=float(score), threshold=float(state.controller.theta), c64_scores=state.c64,
            novelty_percentile=novelty, correction_top32=np.asarray(delta, dtype=np.float32),
            correction_bank_size=correction_size, topk_dN_max=layer["topk_dN_max"],
            topk_dN_mean=layer["topk_dN_mean"])
        self._contexts[(arm_id, event.event_id_sha256)] = {
            "features": features, "selected": selected, "selected_np": selected_np,
            "top_ids": top_ids, "base_map": base_map, "c0_score": float(c0_score),
            "global_cls": encoded.global_cls, "state_sha": state.state_sha}
        return ScorePacket(float(score), int(float(score) >= state.controller.theta),
                           components.psi, float(components.B + components.N + d_band_utility(components)),
                           state.version, state.state_sha, threshold=float(state.controller.theta))

    def shadow_state_digest(self) -> str:
        return self._shadow.sha

    def shadow_snapshot(self, fb_snapshot: Any) -> tuple[Any, str]:
        return fb_snapshot, self._shadow.sha

    def score_shadow(self, event: Any, fb_snapshot: Any, shadow_snapshot: Any) -> ScorePacket:
        if shadow_snapshot != (fb_snapshot, self._shadow.sha) or fb_snapshot != self.state_digest("F-B"):
            raise NativeBackendError("N-shadow did not receive the pre-group F-B state")
        fb = self._states["F-B"]
        context = self._context("F-B", event)
        delta, _ = self.exp2a._score_arm(
            self.impl, fb.bank, "N-SHADOW", self._shadow.cfg, self._shadow.cb,
            self._shadow.sources, context["selected"], event.group_sha256, event.position)
        score = self.r1.max_score(self.r1.corrected_dense_map(
            context["base_map"], context["top_ids"], delta))
        return ScorePacket(float(score), 0, 0.0, 0.0, self._shadow.version, self._shadow.sha)

    def commit_shadow_feedback(self, event: Any, packet: ScorePacket, label: int,
                               fb_query_receipt: Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(label, bool) or label not in (0, 1):
            raise NativeBackendError("N-shadow accepts verified binary feedback only")
        if label == 1:
            return {"action": "defect_ignored"}
        fb, context = self._states["F-B"], self._context("F-B", event)
        before_fb = self._detector_digest("F-B")
        old = self._shadow.sha
        outcome = "FP" if context["c0_score"] >= fb.theta0 else "TN"
        self.exp2a._capture_update(
            self.impl, fb.bank, self.r1, "N-SHADOW", self._shadow.cfg, self._shadow.cb,
            context["features"], context["top_ids"], -1, context["c0_score"], fb.theta0,
            outcome, event.position, self._shadow.sources, event.group_sha256)
        if self._detector_digest("F-B") != before_fb:
            raise NativeBackendError("N-shadow changed F-B native detector state")
        self._refresh_shadow_identity()
        self._shadow.version += int(self._shadow.sha != old)
        return {"action": "normal_only_c1_update", "state_version": self._shadow.version}

    def commit_feedback(self, arm_id: str, event: Any, packet: ScorePacket,
                        label: int, query_receipt: Mapping[str, Any]) -> Mapping[str, Any]:
        state, context = self._states[arm_id], self._context(arm_id, event)
        if (isinstance(label, bool) or label not in (0, 1) or
                context["state_sha"] != state.state_sha or packet.state_sha256 != state.state_sha):
            raise NativeBackendError("feedback is not bound to the sealed arm state")
        before, greedy_before = state.detector_sha, self.r1._memory_content_sha256(state.bank)
        c0_score = context["c0_score"]
        c0_prediction = int(c0_score >= state.theta0)
        outcome = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[(label, c0_prediction)]
        if state.knn is None:
            self.exp2a._capture_update(
                self.impl, state.bank, self.r1, arm_id, state.correction_cfg, state.correction,
                context["features"], context["top_ids"], 1 if label else -1,
                c0_score, state.theta0, outcome, event.position, state.sources, event.group_sha256)
        else:
            state.knn.insert(context["selected_np"], sign=1 if label else -1,
                             source_group_sha256=event.group_sha256, position=event.position)

        def update(**kwargs: Any) -> dict[str, Any]:
            state.writer.update(self.impl, list(kwargs["feature_layers"]),
                                {0: kwargs["candidate_ids"].tolist()}, kwargs["position"],
                                score=kwargs["score"], theta=kwargs["theta"], sigma=kwargs["sigma"])
            return {}

        writer_receipt = self.r1.common_c0_greedy_writer_action(
            position=event.position, c0_score=c0_score, theta=state.theta0, sigma=state.sigma0,
            c0_prediction=c0_prediction, candidate_ids=context["top_ids"],
            candidate_features=context["selected_np"], feature_layers=context["features"],
            queried=True, disclosed_label=label, eligibility=self.impl.select_informative_normal_write,
            update=update)
        if label == 0:
            state.coverage.observe_normal(context["global_cls"])
        greedy_changed = greedy_before != self.r1._memory_content_sha256(state.bank)
        if label == 1 and greedy_changed:
            raise NativeBackendError("verified defect changed Greedy normal memory")
        after = self._detector_digest(arm_id)
        if after != before:
            if greedy_changed:
                _base, state.base_cache, _ = self.r7._score_c64_base(
                    self.impl, state.bank, self.c64_items, self.cfg, self.r1, self.category)
            if state.knn is None:
                state.c64, _ = self.r9._score_c64_for_codebook(
                    self.impl, state.bank, self.r7, self.r1, self.exp2a,
                    self.c64_items, state.base_cache, self.cfg, state.correction_cfg,
                    state.correction, state.sources, self.category, event.position + 1)
            else:
                state.c64, _, _ = self.r10._score_c2_c64(
                    self.impl, state.bank, self.c64_items, state.base_cache, state.knn,
                    event_position=event.position + 1)
            transition = self.r7._refresh_b3(state.controller, state.c64, self.c64_group_sha256,
                                             before, after, state.version)
            if transition is None:
                raise NativeBackendError("changed native detector did not refresh its B3 state")
            state.version += 1
            state.detector_sha = after
        self._refresh_state_identity(arm_id)
        if arm_id == "F-B":
            shadow_before_greedy_refresh = self._shadow.sha
            self._refresh_shadow_identity()
            self._shadow.version += int(self._shadow.sha != shadow_before_greedy_refresh)
        return {"state_version": state.version, "greedy_changed": bool(greedy_changed),
                "writer_action": writer_receipt["action"], "scorer_changed": before != after}

    def static_state_digest(self, baseline_id: str) -> str:
        return self._static[baseline_id].state_digest()

    def static_score(self, baseline_id: str, event: Any) -> ScorePacket:
        return self._static[baseline_id].score(event.feature_ref)
