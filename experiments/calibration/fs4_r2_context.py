"""FS4 production-only registered image, frozen encoder and runtime boundary.

No scientific run starts at import. A separately signed, label-blind execution
manifest and separately authorized production ledger are prerequisites.
"""
from __future__ import annotations

import copy
from collections import Counter
import hashlib
import io
import json
import random
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch

from .fs4_durable_ledger import DurableLedger, digest
from .fs4_r2_security import _reject_reparse
from .fs4_r2_trust import verify_production_signed
from .fs4_source_excluded_scorer import _feature_digest
from .fs4_stochastic_bea import StochasticBEAGroupSlot
from .fs4_threshold_integration import (FEATURE_CONVENTION, CoupledTrajectory, SharedTrajectory,
                                        NativeRuntime, build_native_from_four_support)

ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = ROOT / "docs/calibration/fs4_threshold/HARDENING_R1_GPU_PARITY_MANIFEST_V3.json"
ALLOWED_DEVELOPMENT_CATEGORIES = frozenset(("audiojack", "regulator", "switch", "rolled_strip_base"))
IMAGE_KEYS = frozenset(("group", "event", "relative_path", "content_sha256", "width", "height", "position"))
SUPPORT_KEYS = frozenset(("group", "relative_path", "content_sha256", "width", "height"))


class FS4BoundaryError(RuntimeError):
    pass


def require_isolated_worker() -> None:
    """Fail closed until an independently controlled encoder worker exists."""
    raise FS4BoundaryError("FS4_R2_BLOCKED: trusted encoder is not isolated from caller-controlled Python code")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _checked_path(root: Path, relative: str, *, category: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise FS4BoundaryError("invalid dataset-relative path")
    part = PurePosixPath(relative)
    if part.is_absolute() or not part.parts or part.parts[0] != category or any(p in (".", "..") for p in part.parts):
        raise FS4BoundaryError("dataset path escapes its registered category")
    base = Path(root).absolute()
    for parent in reversed((base, *base.parents)):
        if parent.exists():
            _reject_reparse(parent)
    current = base
    for component in part.parts:
        current = current / component
        _reject_reparse(current)
    if not current.is_file():
        raise FS4BoundaryError("registered image is unavailable")
    return current


def _read_registered_image(root: Path, record: Mapping[str, Any], *, category: str) -> tuple[Path, Image.Image]:
    from experiments.risk_safe.preflight import parse_image_name
    path = _checked_path(root, record["relative_path"], category=category)
    parsed = parse_image_name(path, expected_class=category)
    group = _sha_bytes(f"official-sample-group\0{parsed.official_sample_key}".encode())
    if group != record["group"]:
        raise FS4BoundaryError("registered official Source Group does not match the image")
    data = path.read_bytes()
    if _sha_bytes(data) != record["content_sha256"]:
        raise FS4BoundaryError("registered image content differs")
    try:
        with Image.open(io.BytesIO(data)) as source:
            image = source.convert("RGB")
            image.load()
    except Exception as exc:
        raise FS4BoundaryError("registered image could not be decoded") from exc
    if (image.width, image.height) != (record["width"], record["height"]):
        raise FS4BoundaryError("registered image geometry differs")
    return path, image


def verify_model_identity(backbone: Any, impl: Any, native_encode: Any, expected: str) -> None:
    from experiments.calibration import r1_matched_correction_pilot as r1
    if (impl.encode_pil is not native_encode or backbone.model.training or
            "encode_batch" in vars(backbone) or r1._model_state_sha256(backbone.model) != expected):
        raise FS4BoundaryError("live DINO model state or native encoder differs from its registered identity")


def verify_native_runtime(runtime: Any, expected_instance: Any, fixed_sha: str) -> None:
    if type(runtime) is not NativeRuntime or runtime is not expected_instance:
        raise FS4BoundaryError("unrelated native runtime instance")
    if runtime.states() is None or runtime.fixed_sha != fixed_sha:
        raise FS4BoundaryError("native fixed bank identity differs")


def validate_execution_manifest(payload: Mapping[str, Any], ledger_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete label-blind registration against the durable manifest."""
    from experiments.calibration.profiles import NativeScorerConfig
    required = {"schema", "run_id", "protocol", "source_commit", "query_seed", "model_state_sha256",
                "checkpoint_sha256", "hub_source_tree_sha256", "scorer_config_sha256",
                "preflight_receipt_sha256", "dataset_root", "dataset_identity_sha256", "categories", "category_order"}
    if set(payload) != required or payload["schema"] != "FS4_TRUSTED_EXECUTION_V1":
        raise FS4BoundaryError("invalid trusted execution manifest schema")
    for key in ("run_id", "protocol", "source_commit", "query_seed", "model_state_sha256", "dataset_identity_sha256"):
        if payload[key] != ledger_manifest.get(key):
            raise FS4BoundaryError(f"execution manifest {key} differs from ledger")
    if payload["scorer_config_sha256"] != NativeScorerConfig().sha256:
        raise FS4BoundaryError("native scorer configuration identity differs")
    if not isinstance(payload["dataset_root"], str) or not Path(payload["dataset_root"]).is_absolute():
        raise FS4BoundaryError("registered dataset root must be absolute")
    categories = payload["categories"]
    if (not isinstance(categories, dict) or set(categories) != set(ledger_manifest["expected_events_by_category"]) or
            not set(categories).issubset(ALLOWED_DEVELOPMENT_CATEGORIES)):
        raise FS4BoundaryError("registered category set differs")
    order = payload["category_order"]
    if not isinstance(order, list) or len(order) != len(categories) or set(order) != set(categories):
        raise FS4BoundaryError("registered category order differs")
    for name in ("model_state_sha256", "checkpoint_sha256", "hub_source_tree_sha256", "scorer_config_sha256", "preflight_receipt_sha256", "dataset_identity_sha256"):
        if not isinstance(payload[name], str) or not re.fullmatch(r"[0-9a-f]{64}", payload[name]):
            raise FS4BoundaryError("malformed frozen identity")
    position = 0
    events: set[str] = set()
    all_paths: set[str] = set()
    all_support_groups: set[str] = set()
    all_event_groups: set[str] = set()
    for category in order:
        info = categories[category]
        if set(info) != {"supports", "events"} or len(info["supports"]) != 4:
            raise FS4BoundaryError("exactly four registered Supports required")
        if any(set(record) != SUPPORT_KEYS for record in info["supports"]):
            raise FS4BoundaryError("invalid Support record")
        for item in [*info["supports"], *info["events"]]:
            if (not all(isinstance(item.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", item[key]) for key in ("group", "content_sha256")) or
                    not all(isinstance(item.get(key), int) and not isinstance(item[key], bool) and item[key] > 0 for key in ("width", "height"))):
                raise FS4BoundaryError("invalid registered image identity or geometry")
        groups = [item["group"] for item in info["supports"]]
        if len(set(groups)) != 4 or groups != ledger_manifest["support_groups_by_category"][category]:
            raise FS4BoundaryError("Support group registration differs")
        if all_support_groups.intersection(groups):
            raise FS4BoundaryError("Support group is reused across categories")
        all_support_groups.update(groups)
        for item in info["supports"]:
            if item["relative_path"] in all_paths:
                raise FS4BoundaryError("registered image path is repeated")
            all_paths.add(item["relative_path"])
        if len(info["events"]) != ledger_manifest["expected_events_by_category"][category]:
            raise FS4BoundaryError("registered event count differs")
        rows = []
        category_groups: Counter[str] = Counter()
        for item in info["events"]:
            if (set(item) != IMAGE_KEYS or item["position"] != position or item["event"] in events or
                    not isinstance(item["event"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["event"])):
                raise FS4BoundaryError("event identity or ordering differs")
            rows.append({"group": item["group"], "event": item["event"]})
            if item["relative_path"] in all_paths:
                raise FS4BoundaryError("registered image path is repeated")
            all_paths.add(item["relative_path"])
            category_groups[item["group"]] += 1; all_event_groups.add(item["group"])
            events.add(item["event"]); position += 1
        if any(count != 5 for count in category_groups.values()):
            raise FS4BoundaryError("registered Source Groups must have exactly five views")
        if digest(rows) != ledger_manifest["stream_sha256_by_category"][category]:
            raise FS4BoundaryError("registered category stream digest differs")
    if all_support_groups.intersection(all_event_groups):
        raise FS4BoundaryError("Support groups cannot occur in the event stream")
    dataset = {"dataset_root": payload["dataset_root"], "category_order": order, "categories": categories}
    if digest(dataset) != payload["dataset_identity_sha256"]:
        raise FS4BoundaryError("registered dataset identity differs")
    return copy.deepcopy(dict(payload))


class ProductionContext:
    """Owns its actual native model, registered dataset and four-Support runtime."""

    def __init__(self, *, ledger: DurableLedger, category: str, controller: str | None = None) -> None:
        require_isolated_worker()
        self._initialize(ledger=ledger, category=category, controller=controller)

    def _initialize(self, *, ledger: Any, category: str, controller: str | None = None,
                    research_payload: Mapping[str, Any] | None = None) -> None:
        from experiments import p0_protocol_audit as p0
        from experiments.calibration import profiles, r1_matched_correction_pilot as r1, r3_fullstream_2x2 as r3
        if research_payload is not None:
            from .fs4_research_context import ResearchContext
            from .fs4_research_ledger import ResearchLedger
            from .fs4_r4_worker import BlindedResearchContext
            if type(self) not in (ResearchContext, BlindedResearchContext) or type(ledger) is not ResearchLedger:
                raise FS4BoundaryError("research initialization requires the research-only context and ledger")
        if (research_payload is None and type(ledger) is not DurableLedger) or category != ledger.current_category:
            raise FS4BoundaryError("production context requires the active production ledger category")
        if ledger.manifest["profile"] == "SHARED_TRAJECTORY_THRESHOLD_ONLY":
            if controller is not None or ledger.manifest.get("controller") is not None:
                raise FS4BoundaryError("shared profile cannot select a coupled controller")
        elif controller not in ("L", "E") or controller != ledger.manifest.get("controller"):
            raise FS4BoundaryError("coupled controller differs from the signed run manifest")
        if research_payload is None:
            document = ledger.manifest.get("r2_execution_manifest")
            if not isinstance(document, dict):
                raise FS4BoundaryError("signed trusted execution manifest missing")
            signed = verify_production_signed(document, role="experiment_manifest")
        else:
            signed = research_payload
        if research_payload is not None and signed.get("schema") in (
                "FS4_R4_BLINDED_EXECUTION_V1", "FS4_JOINT_CAMPAIGN_EXECUTION_V1"):
            from .fs4_r4_blinded import validate_blinded_execution
            payload = validate_blinded_execution(signed, ledger.manifest)
        else:
            payload = validate_execution_manifest(signed, ledger.manifest)
        if category not in payload["categories"]:
            raise FS4BoundaryError("unregistered category")
        receipt_bytes = PREFLIGHT.read_bytes()
        if _sha_bytes(receipt_bytes) != payload["preflight_receipt_sha256"]:
            raise FS4BoundaryError("Support preflight receipt differs")
        receipt = json.loads(receipt_bytes)
        category_receipt = receipt["categories"][category]
        if (receipt.get("status") != "PASS" or category_receipt.get("status") != "PASS" or
                receipt["loaded_model_state_sha256"] != payload["model_state_sha256"] or
                receipt["model"]["checkpoint_sha256"] != payload["checkpoint_sha256"] or
                receipt["model"]["hub_source_tree_sha256"] != payload["hub_source_tree_sha256"] or
                category_receipt["fixed_evidence"]["actual_structured_sha256"] != ledger.manifest["fixed_support_sha256_by_category"][category] or
                category_receipt["support_image_content_sha256"] != [x["content_sha256"] for x in payload["categories"][category]["supports"]]):
            raise FS4BoundaryError("frozen model or four-Support preflight identity differs")
        self._ledger, self._category, self._manifest = ledger, category, payload
        self._manifest_digest = digest(payload)
        self._impl = p0.require_sources()
        self._native_encode = self._impl.encode_pil
        model_receipt = r3._get_verified_model_receipt()
        restore_hub, _model_profile = r3._runtime_and_native_setup(torch, self._impl, model_receipt)
        try:
            self._cfg, _ = r3._native_config(p0, profiles, r1)
            self._backbone = self._impl.DINOv2MultiLayerBackbone(device="cuda")
            self._backbone_type = type(self._backbone)
            self._backbone_batch_method = self._backbone_type.encode_batch
            self._model_type = type(self._backbone.model)
            self._model_forward_method = self._model_type.forward
        finally:
            restore_hub()
        self._verify_model()
        support_records = payload["categories"][category]["supports"]
        cache: dict[str, Any] = {}; identities = []; globals_ = []; paths = []
        official_groups, support_relative_ids = [], []
        from experiments.risk_safe.preflight import parse_image_name
        for item in support_records:
            path = _checked_path(Path(payload["dataset_root"]), item["relative_path"], category=category)
            parsed = parse_image_name(path, expected_class=category)
            if (parsed.capture_session != "OK" or
                    _sha_bytes(f"official-sample-group\0{parsed.official_sample_key}".encode()) != item["group"]):
                raise FS4BoundaryError("Support image is not a registered normal training image")
            official_groups.append(parsed.official_sample_key); support_relative_ids.append(parsed.rel_path)
        from experiments.calibration.stage05 import opaque_digest
        if (opaque_digest("support-groups", official_groups) != category_receipt["support_groups_sha256"] or
                opaque_digest("support-images", support_relative_ids) != category_receipt["support_images_sha256"]):
            raise FS4BoundaryError("Support identities differ from the independently registered preflight")
        for item in support_records:
            path, image = _read_registered_image(Path(payload["dataset_root"]), item, category=category)
            path_string = str(path); paths.append(path_string)
            image_seed = p0.deterministic_subseed(self._cfg.augmentation_seed, path_string)
            random.seed(image_seed); np.random.seed(image_seed % (2**32 - 1)); torch.manual_seed(image_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(image_seed)
            views = self._impl.make_support_refs(image, self._backbone, self._impl.SupportAugConfig(),
                self._cfg.augmentations, baseline_compat=False, include_identity=self._cfg.include_identity)
            cache[path_string] = []
            for view in views:
                features, _ = self._impl.encode_pil(self._backbone, view, self._cfg.feature_fuse)
                cache[path_string].append([self._impl.l2_normalize(t).detach().to("cpu", dtype=torch.float16) for t in features])
            features, cls = self._impl.encode_pil(self._backbone, image, self._cfg.feature_fuse)
            identities.append((image, features[0])); globals_.append(cls.detach().float().cpu().numpy().reshape(-1))
        self._verify_model()
        runtime, reference, coverage = build_native_from_four_support(impl=self._impl, cfg=self._cfg,
            support_paths=paths, support_groups=[x["group"] for x in support_records], augmentation_cache=cache,
            identity_images_and_features=identities, support_cls=np.stack(globals_), feature_convention=FEATURE_CONVENTION)
        if type(runtime) is not NativeRuntime or runtime.fixed_sha != ledger.manifest["fixed_support_sha256_by_category"][category]:
            raise FS4BoundaryError("constructed native fixed bank differs from the registered receipt")
        if signed.get("schema") == "FS4_JOINT_CAMPAIGN_EXECUTION_V1":
            joint_reference = signed["support_preflight_reference_by_category"][category]
            support_paths = [str(Path(payload["dataset_root"]).absolute() / item["relative_path"])
                             for item in support_records]
            actual_groups = [item["group"] for item in support_records]
            actual_content = [item["content_sha256"] for item in support_records]
            actual_scores = [float(value) for value in reference["support_scores"]]
            if (support_paths != json.loads(Path(signed["support_preflight_receipt_path"]).read_bytes())
                    ["categories"][category]["absolute_paths"] or
                    actual_groups != joint_reference["support_groups"] or
                    actual_content != joint_reference["support_content_sha256"] or
                    actual_scores != joint_reference["support_self_scores"] or
                    float(reference["mean"]) != joint_reference["mean"] or
                    float(reference["sample_std_ddof1"]) != joint_reference["sample_std_ddof1"] or
                    float(reference["theta"]) != joint_reference["theta0"] or
                    float(reference["writer_reference"]["sigma"]) != joint_reference["writer_sigma"] or
                    runtime.fixed_sha != joint_reference["fixed_support_sha256"]):
                raise FS4BoundaryError("constructed native four-Support bootstrap differs from campaign receipt")
        self._runtime = runtime
        self._support_reference = copy.deepcopy(reference)
        self._runtime_binding = (id(runtime), id(runtime.bank), id(runtime.writer), id(runtime.correction_cb),
                                 id(runtime.rescorer), id(runtime.impl), id(runtime.cfg), id(self._backbone.model))
        start = payload["categories"][category]["events"][0]["position"]
        policy = StochasticBEAGroupSlot(seed=payload["query_seed"], start_position=start)
        blocked = lambda *_args, **_kwargs: (_ for _ in ()).throw(FS4BoundaryError("durable callback is not bound"))
        if ledger.manifest["profile"] == "SHARED_TRAJECTORY_THRESHOLD_ONLY":
            self._trajectory = SharedTrajectory(runtime=runtime, coverage=coverage, theta0=float(reference["theta"]),
                policy=policy, seal_prediction=blocked, seal_query=blocked, label_provider=blocked,
                frozen_theta_readout=(signed.get("schema") == "FS4_JOINT_CAMPAIGN_EXECUTION_V1"))
        else:
            self._trajectory = CoupledTrajectory(controller=controller, runtime=runtime, coverage=coverage,
                theta0=float(reference["theta"]), policy=policy, seal_prediction=blocked, seal_query=blocked, label_provider=blocked)
        self._policy, self._coverage = policy, coverage
        self._coverage_support_sha256 = _sha_bytes(coverage.support.tobytes() + coverage.support_distances.tobytes())
        self._writer_reference = (runtime.theta_ref, runtime.sigma_ref)
        self._controllers = ((self._trajectory.L, self._trajectory.E) if hasattr(self._trajectory, "L")
                             else (self._trajectory.controller,))
        self._events = {item["position"]: copy.deepcopy(item) for item in payload["categories"][category]["events"]}

    def _verify_model(self) -> None:
        from experiments.calibration.profiles import NativeScorerConfig
        profile = NativeScorerConfig()
        blocks = getattr(self._backbone.model, "blocks", None)
        if blocks is None:
            raise FS4BoundaryError("DINO transformer blocks are unavailable")
        expected_layers = sorted({max(0, int(round(len(blocks) * p)) - 1) for p in (.25, .50, .75, 1.)})
        if (self._backbone.model_name != profile.model_name or self._backbone.img_size != profile.image_size or
                self._backbone.resize_size != profile.resize_size or self._backbone.use_fp16 != profile.use_fp16 or
                self._backbone.device.type != "cuda" or list(self._backbone.layers) != expected_layers or
                type(self._backbone) is not self._backbone_type or type(self._backbone).encode_batch is not self._backbone_batch_method or
                type(self._backbone.model) is not self._model_type or type(self._backbone.model).forward is not self._model_forward_method or
                "forward" in vars(self._backbone.model) or
                tuple(self._backbone.mean) != (.485, .456, .406) or tuple(self._backbone.std) != (.229, .224, .225) or
                any(module.training for module in self._backbone.model.modules())):
            raise FS4BoundaryError("frozen DINO preprocessing or feature-layer configuration differs")
        verify_model_identity(self._backbone, self._impl, self._native_encode, self._manifest["model_state_sha256"])

    def _registered_event(self, position: int) -> dict[str, Any]:
        if not isinstance(position, int) or isinstance(position, bool) or position != self._trajectory.policy.last_position + 1:
            raise FS4BoundaryError("event position is not the next registered event")
        if position not in self._events:
            raise FS4BoundaryError("event is not registered for this category")
        candidates = [item for item in self._manifest["categories"][self._category]["events"] if item["position"] == position]
        if len(candidates) != 1 or self._events[position] != candidates[0]:
            raise FS4BoundaryError("registered event table changed")
        return self._events[position]

    def _verify_runtime(self) -> None:
        from .fs4_source_excluded_scorer import NativeSourceExcludedScorer, VerifiedNormalFeatureCache
        from .fs4_threshold_controllers import SelectionAwareSourceExcludedThreshold
        from .fs4_threshold_core import WeightedThresholds
        verify_native_runtime(self._trajectory.runtime, self._runtime,
                              self._ledger.manifest["fixed_support_sha256_by_category"][self._category])
        runtime = self._runtime
        if ((id(runtime), id(runtime.bank), id(runtime.writer), id(runtime.correction_cb),
             id(runtime.rescorer), id(runtime.impl), id(runtime.cfg), id(self._backbone.model)) != self._runtime_binding or
                runtime.impl is not self._impl or runtime.cfg is not self._cfg or runtime.writer.bank is not runtime.bank or
                runtime.writer.name != "greedy_coreset" or runtime.bank.use_defect_bank or
                runtime.correction_cfg is not runtime.bank.corr_cfg or
                runtime.correction_cb is not runtime.bank.corr_cb["p0"]["id"][0] or
                int(runtime.correction_cb.K) != 1024 or self._trajectory.policy is not self._policy or
                self._trajectory.coverage is not self._coverage or self._trajectory.policy.seed != self._manifest["query_seed"] or
                (runtime.theta_ref, runtime.sigma_ref) != self._writer_reference or
                self._trajectory._learning_reference != self._writer_reference[0] or
                (hasattr(self._trajectory, "query_reference") and self._trajectory.query_reference != self._writer_reference[0]) or
                _sha_bytes(self._coverage.support.tobytes() + self._coverage.support_distances.tobytes()) != self._coverage_support_sha256 or
                ((self._trajectory.L, self._trajectory.E) if hasattr(self._trajectory, "L") else (self._trajectory.controller,)) != self._controllers):
            raise FS4BoundaryError("native runtime components or query seed changed")
        from experiments.calibration.native import validate_p0_config
        from experiments.calibration.profiles import NativeScorerConfig
        from experiments.calibration import r3_fullstream_2x2 as r3
        from experiments.risk_safe.native_bridge import dynamic_codebooks
        scorer = runtime.rescorer
        if (type(scorer) is not NativeSourceExcludedScorer or scorer.bank is not runtime.bank or
                scorer.impl is not self._impl or scorer.cfg is not self._cfg or
                scorer.correction_cfg is not runtime.correction_cfg or scorer.correction_cb is not runtime.correction_cb or
                scorer.correction_sources is not runtime.sources or scorer.provenance is not runtime.provenance or
                scorer.normalization != FEATURE_CONVENTION):
            raise FS4BoundaryError("source-excluded native scorer binding changed")
        for controller in self._controllers:
            engine = controller.engine
            if (type(engine) is not WeightedThresholds or engine.minimum_groups != 12 or
                    engine.minimum_ess != 12 or engine.w_max != 50):
                raise FS4BoundaryError("frozen weighted estimator configuration changed")
            if isinstance(controller, SelectionAwareSourceExcludedThreshold) and (
                    controller.scorer is not scorer or type(controller.cache) is not VerifiedNormalFeatureCache):
                raise FS4BoundaryError("E controller scorer or cache binding changed")
        if any(int(book.K) != 1024 for book in dynamic_codebooks(runtime.bank)):
            raise FS4BoundaryError("native Greedy capacity changed")
        checked_correction = copy.deepcopy(runtime.correction_cfg)
        checked_correction.K_correction = 256
        r3._validate_correction_config(type("FrozenConfig", (), {"corr_cfg": checked_correction})())
        validate_p0_config(self._cfg, NativeScorerConfig())

    def _verify_cached_evidence(self) -> None:
        from .fs4_threshold_controllers import SelectionAwareSourceExcludedThreshold
        predictions = {row["payload"]["event"]["position"]: row["payload"]["event"]
                       for row in self._ledger.records if row["kind"] == "PREDICTION"}
        for controller in self._controllers:
            for evidence in controller.evidence:
                try:
                    evidence.validate()
                except (ValueError, TypeError, AttributeError) as exc:
                    raise FS4BoundaryError("verified normal evidence was corrupted") from exc
                if isinstance(controller, SelectionAwareSourceExcludedThreshold):
                    record = controller.cache.records.get(evidence.group)
                    prediction = predictions.get(evidence.position)
                    provenance = prediction.get("trusted_event_provenance") if prediction else None
                    if (record is None or record.evidence != evidence or
                            not isinstance(provenance, dict) or prediction["event"] != evidence.event or
                            prediction["group"] != evidence.group or prediction["feature_sha256"] != record.feature_sha256 or
                            provenance.get("geometry") != [record.width, record.height] or
                            record.normalization != FEATURE_CONVENTION or record.dtype != str(record.features.dtype) or
                            record.features.dtype != torch.float32 or _feature_digest(record.features) != record.feature_sha256):
                        raise FS4BoundaryError("verified normal feature cache was corrupted")

    def _encode_event(self, position: int) -> dict[str, Any]:
        if digest(self._manifest) != self._manifest_digest or digest(self._ledger._execution_manifest) != self._manifest_digest:
            raise FS4BoundaryError("signed execution manifest changed")
        self._ledger.verify_live_sources()
        record = self._registered_event(position)
        self._verify_runtime()
        self._verify_model()
        _path, image = _read_registered_image(Path(self._manifest["dataset_root"]), record, category=self._category)
        features, cls = self._impl.encode_pil(self._backbone, image, self._cfg.feature_fuse)
        self._verify_model()
        if len(features) != 1 or tuple(features[0].shape) != (784, 4096) or features[0].dtype != torch.float32 or not torch.isfinite(features[0]).all():
            raise FS4BoundaryError("native encoder did not produce finite full FP32 features")
        cls_array = cls.detach().float().cpu().numpy().reshape(-1)
        if cls_array.shape != (int(self._backbone.model.num_features),) or not np.isfinite(cls_array).all():
            raise FS4BoundaryError("native encoder did not produce finite CLS features")
        provenance = {"schema": "FS4_TRUSTED_EVENT_V1", "dataset_identity_sha256": self._manifest["dataset_identity_sha256"],
            "manifest_sha256": self._manifest_digest, "category": self._category,
            "image_content_sha256": record["content_sha256"], "geometry": [image.width, image.height],
            "model_state_sha256": self._manifest["model_state_sha256"],
            "cls_sha256": _sha_bytes(cls_array.tobytes()), "feature_sha256": _feature_digest(features[0])}
        return {"position": position, "group": record["group"], "event": record["event"], "image": image,
                "features": features[0], "cls_feature": cls_array, "feature_convention": FEATURE_CONVENTION,
                "event_provenance": provenance}
