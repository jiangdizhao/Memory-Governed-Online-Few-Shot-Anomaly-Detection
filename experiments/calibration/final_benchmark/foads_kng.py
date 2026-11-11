"""Source-bound, paper-consistent FOADS/K-NG P0 protocol adapter.

The upstream K-NG repository is an immutable reference. This adapter imports
its Klinks implementation and extracts only two pure K-online helpers from the
pinned source without importing K-online's experiment driver (which hides CUDA
and reads TEST labels/masks). It restores the pinned distance methods' PyTorch
1.8 dimension-1 semantics on newer Torch and applies the approved Equation
(11) Mahalanobis quadratic form; Klinks update logic remains upstream.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import random
import re
from pathlib import Path
import sys
from types import MethodType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as torch_functional
from scipy.ndimage import gaussian_filter


BASELINE_NAME = "FOADS-KNG-PAPER-CONSISTENT-PROTOCOL-ADAPTED"
UPSTREAM_REPOSITORY = "https://github.com/Whishing/K-NG"
UPSTREAM_COMMIT = "63dd8e7c9ade5619771d4b5544f032206818bb22"
UPSTREAM_SHA256 = {
    "Konline.py": "a2bd1823cc1bb1e45b3749108bc2865816852fa493ae32ad201fcb153fef72f1",
    "Klink.py": "402f9bd969693207aedc5286029b2e51f9bd0000016eeacfab3510cfbf76d440",
    "datasets/mvtec.py": "5a1598b202e5ed3e797126e08827ef0b5215e0d530dbc8d79514fc7cc84596b6",
}

FEATURE_CHANNELS = 448
FEATURE_DIMENSIONS = 100
FEATURE_GRID = (56, 56)
IMAGE_SIZE = 224
RESIZE_SIZE = 256
INITIAL_NORMAL_IMAGE_COUNT = 68
UPSTREAM_SCRIPT_INITIAL_NORMAL_IMAGE_COUNT = 10
KNG_INIT_NODES = 56 * 56
KNG_AGE_MAX = 25
KNG_FIT_EPOCHS = 10
KNG_RANDOM_SEED = 1025
KNG_DIMENSION_SEED = 1024
KNG_IMAGE_BATCH_SIZE = 10
KNG_GROUPS_PER_UPDATE = 2
KNG_VIEWS_PER_GROUP = 5
GAUSSIAN_SIGMA = 4.0

_FORBIDDEN_KEYS = {
    "label", "labels", "test_label", "test_labels", "anomaly_class",
    "mask", "masks", "mask_path", "gt", "ground_truth", "test_outcome",
    "test_outcomes", "defect_mask", "defect_masks",
}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class FOADSKNGError(RuntimeError):
    """The pinned source, registered identity, or adapted protocol is invalid."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise FOADSKNGError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _reject_label_mask_fields(value: Any, *, where: str = "registration") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in _FORBIDDEN_KEYS or any(token in normalized for token in
                                                     ("label", "mask", "ground_truth", "outcome")):
                raise FOADSKNGError(f"{where} contains prohibited outcome field {key!r}")
            _reject_label_mask_fields(item, where=where)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_label_mask_fields(item, where=where)


@dataclass(frozen=True)
class VerifiedKNGSource:
    root: Path
    file_sha256: Mapping[str, str]
    commit: str = UPSTREAM_COMMIT


def verify_upstream_source(source_root: str | Path) -> VerifiedKNGSource:
    """Fail closed unless every source file used by this adapter matches its pin."""
    root = Path(source_root).resolve()
    actual: dict[str, str] = {}
    for relative, expected in UPSTREAM_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise FOADSKNGError(f"pinned K-NG source file is missing: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise FOADSKNGError(f"pinned K-NG source checksum mismatch: {relative}")
        actual[relative] = digest
    return VerifiedKNGSource(root=root, file_sha256=actual)


def _extract_pinned_function(path: Path, function_name: str,
                             namespace: dict[str, Any]) -> Callable[..., Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [node for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == function_name]
    if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
        raise FOADSKNGError(f"expected one top-level {function_name} in pinned source")
    module = ast.Module(body=[matches[0]], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function_name]


@dataclass(frozen=True)
class KNGSourceAPI:
    verified_source: VerifiedKNGSource
    Klinks: type
    embedding_concat: Callable[..., Any]
    upstream_matrix_mahalanobis: Callable[..., Any]
    nearest_distance_probe_shape: tuple[int, ...]


def load_upstream_api(source_root: str | Path) -> KNGSourceAPI:
    """Load Klinks from the exact source and safe, pure helpers from K-online."""
    verified = verify_upstream_source(source_root)
    klink_path = verified.root / "Klink.py"
    module_name = f"_pinned_kng_{UPSTREAM_COMMIT}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, klink_path)
        if spec is None or spec.loader is None:
            raise FOADSKNGError("could not load the pinned Klink module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    if not hasattr(module, "Klinks"):
        raise FOADSKNGError("pinned Klink.py does not export Klinks")
    online_path = verified.root / "Konline.py"
    concat = _extract_pinned_function(online_path, "embedding_concat",
                                      {"torch": torch, "F": torch_functional})
    upstream_mahalanobis = _extract_pinned_function(online_path, "Matrix_Mahalanobis",
                                                    {"np": np})
    # Klink.py calls torch.pairwise_distance((B,C,1), (1,C,N)) and then treats
    # axis 1 as N. Probe the installed Torch semantics without loading data/GPU.
    probe = torch.pairwise_distance(torch.zeros((1, 3, 1)),
                                    torch.zeros((1, 3, 4)))
    return KNGSourceAPI(verified_source=verified, Klinks=module.Klinks,
                        embedding_concat=concat,
                        upstream_matrix_mahalanobis=upstream_mahalanobis,
                        nearest_distance_probe_shape=tuple(int(x) for x in probe.shape))


def bind_pytorch18_distance_compatibility(model: Any, *,
                                          device: str | torch.device = "cuda") -> Any:
    """Restore Klinks' PyTorch 1.8 dimension-1 distance semantics explicitly.

    Only the pinned class's two distance methods are replaced on this instance.
    Its fit, threshold, pseudo-normal, topology, covariance, and online update
    methods remain the original source methods.
    """
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise FOADSKNGError("CUDA is required for the registered K-NG execution device")
    if not hasattr(model, "centeroid") or not hasattr(model, "node_list"):
        raise FOADSKNGError("distance compatibility requires a source Klinks instance")

    def _distances(self, data: np.ndarray, *, top_two: bool) -> Any:
        observation = torch.from_numpy(np.asarray(data))
        if observation.ndim != 2:
            raise FOADSKNGError("Klinks distance input must have shape (B,C)")
        batch_count, channels = observation.shape
        centroid = torch.from_numpy(np.asarray(self.centeroid)).t().unsqueeze(0)
        if centroid.shape[1] != channels or centroid.shape[2] != len(self.node_list):
            raise FOADSKNGError("Klinks centroid matrix differs from feature or node dimensions")
        centroid = centroid.to(target)
        dist_chunks = []
        index_chunks = []
        batch_size = 500
        for batch_index in range(batch_count // batch_size + 1):
            start = batch_index * batch_size
            stop = min(start + batch_size, batch_count)
            test_batch = observation[start:stop].to(target)
            # PyTorch 1.8 pairwise_distance reduced dimension 1 for these
            # 3D tensors. Preserve the epsilon, dtype, and following sqrt.
            dist_matrix = torch.linalg.vector_norm(
                test_batch.unsqueeze(-1) - centroid + 1e-6,
                ord=2,
                dim=1,
            )
            if dist_matrix.shape != (stop - start, len(self.node_list)):
                raise FOADSKNGError("legacy Klinks distance adapter did not produce (B,N)")
            if top_two:
                topk_values, topk_indexes = torch.topk(
                    dist_matrix, k=2, dim=1, largest=False, sorted=True)
                dist_chunks.append(torch.sqrt(topk_values).cpu())
                index_chunks.append(topk_indexes.cpu())
            else:
                dist_chunks.append(torch.sqrt(dist_matrix).cpu())
        if top_two:
            return torch.cat(dist_chunks), torch.cat(index_chunks)
        return torch.cat(dist_chunks)

    def gpu_cal_dist(self, data):
        return _distances(self, data, top_two=False)

    def online_gpu_cal_dist(self, data):
        return _distances(self, data, top_two=True)

    model.gpu_cal_dist = MethodType(gpu_cal_dist, model)
    model.online_gpu_cal_dist = MethodType(online_gpu_cal_dist, model)
    model._foads_pytorch18_distance_compatibility = {
        "device": str(target),
        "formula": "vector_norm(test_batch.unsqueeze(-1)-centeroid+1e-6,ord=2,dim=1)",
        "gpu_cal_dist_extra_sqrt": True,
        "online_top2_before_extra_sqrt": True,
    }
    return model


def kng_dimension_indices() -> tuple[int, ...]:
    """Match K-online's `random.seed(1024); sample(range(448), 100)` selection."""
    return tuple(random.Random(KNG_DIMENSION_SEED).sample(
        range(FEATURE_CHANNELS), FEATURE_DIMENSIONS))


def kng_resnet18_transform() -> Any:
    """Return the upstream resize/crop/ImageNet normalization transform."""
    try:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
    except Exception as exc:
        raise FOADSKNGError("torchvision transforms are unavailable") from exc
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def resnet18_state_sha256(model: Any) -> str:
    """Hash a loaded ResNet18 state for the later run identity; loads no weights."""
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(_canonical_json(list(array.shape)) + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_pretrained_resnet18() -> tuple[Any, str]:
    """Construct the official ImageNet ResNet18 without running inference."""
    try:
        from torchvision.models import ResNet18_Weights, resnet18
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=True)
    except Exception as exc:
        raise FOADSKNGError("could not load torchvision ImageNet ResNet18 weights") from exc
    model.eval()
    return model, resnet18_state_sha256(model)


def extract_kng_embeddings(model: Any, image_batch: torch.Tensor,
                           source_api: KNGSourceAPI) -> np.ndarray:
    """Apply K-online's layer1/2/3 embedding concat and fixed 100-d selection.

    The supplied tensor must already have the official 224x224 ImageNet-normalized
    transform. This function accepts image pixels only; it has no outcome inputs.
    """
    if (not isinstance(image_batch, torch.Tensor) or image_batch.ndim != 4 or
            tuple(image_batch.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE)):
        raise FOADSKNGError("K-NG image batches must have shape (B,3,224,224)")
    if getattr(model, "training", True):
        raise FOADSKNGError("ResNet18 must be in eval mode for batch-independent features")
    try:
        layers = (model.layer1[-1], model.layer2[-1], model.layer3[-1])
    except Exception as exc:
        raise FOADSKNGError("feature extractor must expose ResNet18 layer1/2/3 blocks") from exc
    captured: list[torch.Tensor] = []
    handles = [layer.register_forward_hook(lambda _m, _i, output: captured.append(output))
               for layer in layers]
    try:
        with torch.no_grad():
            model(image_batch)
        if len(captured) != 3 or any(not isinstance(item, torch.Tensor) for item in captured):
            raise FOADSKNGError("ResNet18 did not emit exactly three hooked feature maps")
        activations = [item.detach().cpu() for item in captured]
        combined = activations[0]
        combined = source_api.embedding_concat(combined, activations[1])
        combined = source_api.embedding_concat(combined, activations[2])
        if combined.ndim != 4 or combined.shape[1] != FEATURE_CHANNELS or tuple(combined.shape[2:]) != FEATURE_GRID:
            raise FOADSKNGError("ResNet18 feature pyramid differs from pinned 448x56x56 convention")
        indices = torch.tensor(kng_dimension_indices(), dtype=torch.long)
        selected = torch.index_select(combined, 1, indices)
        return selected.permute(0, 2, 3, 1).contiguous().numpy()
    finally:
        for handle in handles:
            handle.remove()


def paper_consistent_mahalanobis(features: np.ndarray, means: np.ndarray,
                                 inverse_covariances: np.ndarray) -> np.ndarray:
    """Compute Equation (11), rejecting material negative/nonfinite quadratics."""
    x = np.asarray(features, dtype=np.float64)
    mu = np.asarray(means, dtype=np.float64)
    inv = np.asarray(inverse_covariances, dtype=np.float64)
    if x.ndim != 2:
        raise FOADSKNGError("features must have shape (B,C)")
    batch, channels = x.shape
    if mu.shape != (batch, channels) or inv.shape != (batch, channels, channels):
        raise FOADSKNGError("means and inverse covariances must match feature batch and dimension")
    if not (np.isfinite(x).all() and np.isfinite(mu).all() and np.isfinite(inv).all()):
        raise FOADSKNGError("Mahalanobis inputs must be finite")
    delta = x - mu
    squared = np.einsum("bi,bij,bj->b", delta, inv, delta, optimize=True)
    if not np.isfinite(squared).all():
        raise FOADSKNGError("Mahalanobis quadratic form is nonfinite")
    abs_bound = np.einsum("bi,bij,bj->b", np.abs(delta), np.abs(inv),
                          np.abs(delta), optimize=True)
    tolerance = 64.0 * np.finfo(np.float64).eps * np.maximum(1.0, abs_bound)
    if np.any(squared < -tolerance):
        raise FOADSKNGError("Mahalanobis quadratic form is materially negative")
    squared = np.maximum(squared, 0.0)
    result = np.sqrt(squared)
    if not np.isfinite(result).all():
        raise FOADSKNGError("Mahalanobis distance is nonfinite")
    return result


def paper_consistent_raw_map(model: Any, feature_map: np.ndarray,
                             source_api: KNGSourceAPI) -> np.ndarray:
    """Return the raw 224x224, sigma-4 map using Equation (11), before min-max."""
    features = np.asarray(feature_map)
    if features.ndim != 3 or features.shape[2] != FEATURE_DIMENSIONS or not np.isfinite(features).all():
        raise FOADSKNGError("feature map must be finite with shape (H,W,100)")
    height, width, channels = features.shape
    flat = np.ascontiguousarray(features.reshape(-1, channels), dtype=np.float32)
    if not hasattr(model, "gpu_cal_dist") or not hasattr(model, "node_list"):
        raise FOADSKNGError("scoring requires the pinned Klinks nearest-node interface")
    if not getattr(model, "_foads_pytorch18_distance_compatibility", None):
        raise FOADSKNGError("Klinks must use the registered PyTorch 1.8 distance compatibility adapter")
    ranking = model.gpu_cal_dist(flat)
    if not isinstance(ranking, torch.Tensor):
        ranking = torch.as_tensor(ranking)
    expected_nodes = len(np.asarray(model.node_list))
    if (ranking.ndim != 2 or ranking.shape[0] != flat.shape[0] or
            ranking.shape[1] != expected_nodes):
        raise FOADSKNGError("legacy-compatible Klinks distance matrix must have one column per node")
    nearest_indexes = torch.topk(ranking, k=1, dim=1, largest=False,
                                 sorted=True).indices[:, 0].detach().cpu().numpy()
    node_ids = np.asarray(model.node_list)[nearest_indexes]
    means: list[np.ndarray] = []
    inverse_covariances: list[np.ndarray] = []
    try:
        for node_id in node_ids:
            node = model.network.nodes[int(node_id)]
            means.append(np.asarray(node["vector"], dtype=np.float64).reshape(channels))
            inverse_covariances.append(np.asarray(node["cov_inv"], dtype=np.float64).reshape(channels, channels))
    except Exception as exc:
        raise FOADSKNGError("Klinks nodes lack initialized means or inverse covariances") from exc
    distances = paper_consistent_mahalanobis(
        flat, np.stack(means), np.stack(inverse_covariances))
    patch_map = distances.reshape(height, width)
    interpolated = torch_functional.interpolate(
        torch.from_numpy(patch_map)[None, None], size=IMAGE_SIZE,
        mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    smoothed = gaussian_filter(interpolated, sigma=GAUSSIAN_SIGMA)
    if smoothed.shape != (IMAGE_SIZE, IMAGE_SIZE) or not np.isfinite(smoothed).all():
        raise FOADSKNGError("raw K-NG anomaly map is invalid")
    return smoothed


def paper_consistent_raw_image_score(model: Any, feature_map: np.ndarray,
                                     source_api: KNGSourceAPI) -> float:
    """Return one image's maximum raw score, independent of sibling images."""
    score = float(np.max(paper_consistent_raw_map(model, feature_map, source_api)))
    if not math.isfinite(score):
        raise FOADSKNGError("raw image anomaly score is nonfinite")
    return score


@dataclass(frozen=True)
class RegisteredP0Event:
    position: int
    original_position: int
    event_id_sha256: str
    image_identity_sha256: str
    image_content_sha256: str
    source_group_sha256: str
    camera_id: int
    within_group_ordinal: int


@dataclass(frozen=True)
class P0Registration:
    category: str
    dataset_release_sha256: str
    identity_manifest_sha256: str
    role_manifest_sha256: str
    initial_train_image_ids: tuple[str, ...]
    initial_train_roles: tuple[str, ...]
    stream_events: tuple[RegisteredP0Event, ...]


def build_registered_p0(identity_manifest: Mapping[str, Any], category: str, *,
                        expected_identity_manifest_sha256: str,
                        expected_role_manifest_sha256: str,
                        expected_dataset_release_sha256: str) -> P0Registration:
    """Bind the adapter to Gate 1's certified P0 68-image allocation and seed-0 stream."""
    _reject_label_mask_fields(identity_manifest)
    if identity_manifest.get("schema") != "final-benchmark.identity-manifest.v1" or identity_manifest.get("dataset_id") != "realiad":
        raise FOADSKNGError("FOADS-KNG requires a registered Real-IAD identity manifest")
    if identity_manifest.get("identity_census_status") != "PASS":
        raise FOADSKNGError("Real-IAD identity census is not certified PASS")
    identity_sha = _sha256(identity_manifest)
    if identity_sha != _require_digest(expected_identity_manifest_sha256, "identity manifest commitment"):
        raise FOADSKNGError("identity manifest differs from its registered commitment")
    release_sha = _require_digest(expected_dataset_release_sha256, "dataset release commitment")
    role_sha = _require_digest(expected_role_manifest_sha256, "role manifest commitment")
    roles = [item for item in identity_manifest.get("role_manifests", [])
             if item.get("category") == category and item.get("support_seed") == 0]
    streams = [item for item in identity_manifest.get("stream_manifests", [])
               if item.get("category") == category and item.get("stream_seed") == 0]
    if len(roles) != 1 or len(streams) != 1:
        raise FOADSKNGError("identity manifest lacks one registered P0 role and stream manifest")
    role = roles[0]
    stream = streams[0]
    if (role.get("dataset_id") != "realiad" or role.get("dataset_release_sha256") != release_sha or
            role.get("category") != category or role.get("support_seed") != 0 or
            _sha256(role) != role_sha):
        raise FOADSKNGError("P0 role manifest differs from its certified identity")
    selected = role.get("selected")
    if (not isinstance(selected, list) or len(selected) != INITIAL_NORMAL_IMAGE_COUNT or
            [row.get("role") for row in selected] != ["SUPPORT"] * 4 + ["C64"] * 64):
        raise FOADSKNGError("P0 initialization must use exactly four Supports plus 64 C64 identities")
    selected_ids = [str(row.get("image_identity_sha256", "")) for row in selected]
    if len(set(selected_ids)) != INITIAL_NORMAL_IMAGE_COUNT or any(_HEX64.fullmatch(item) is None for item in selected_ids):
        raise FOADSKNGError("P0 initialization identities are missing or duplicated")
    records = identity_manifest.get("identity_records")
    if not isinstance(records, list):
        raise FOADSKNGError("identity manifest has no identity-only records")
    category_records = [row for row in records if row.get("dataset_id") == "realiad" and row.get("category") == category]
    by_identity = {str(row.get("image_identity_sha256")): row for row in category_records}
    train_groups: set[str] = set()
    train_contents: set[str] = set()
    for role_row in selected:
        image_id = str(role_row["image_identity_sha256"])
        record = by_identity.get(image_id)
        if (record is None or record.get("split") != "train" or record.get("train_normal_eligible") is not True or
                record.get("image_content_sha256") != role_row.get("image_content_sha256") or
                record.get("source_group_sha256") != role_row.get("source_group_sha256")):
            raise FOADSKNGError("selected initialization image is not its registered TRAIN-normal record")
        train_groups.add(str(record["source_group_sha256"]))
        train_contents.add(str(record["image_content_sha256"]))
    if len(train_groups) != 68 or len(train_contents) != 68:
        raise FOADSKNGError("the 68 initial normals must have distinct source groups and image contents")
    if stream.get("category") != category or stream.get("stream_seed") != 0:
        raise FOADSKNGError("the registered P0 stream must use stream seed 0")
    raw_events = stream.get("events")
    if not isinstance(raw_events, list) or not raw_events or stream.get("event_count") != len(raw_events):
        raise FOADSKNGError("registered P0 TEST stream is empty or has an invalid event count")
    test_records = {str(row.get("image_identity_sha256")): row for row in category_records
                    if row.get("split") == "test"}
    if len(test_records) != len([row for row in category_records if row.get("split") == "test"]):
        raise FOADSKNGError("registered TEST identities are duplicated")
    required_event_keys = {"position", "original_position", "event_id_sha256",
                           "image_identity_sha256", "image_content_sha256",
                           "source_group_sha256", "camera_id", "within_group_ordinal"}
    parsed_events: list[RegisteredP0Event] = []
    for expected_position, row in enumerate(raw_events):
        if not isinstance(row, Mapping) or set(row) != required_event_keys:
            raise FOADSKNGError("registered P0 event schema is invalid or contains non-identity fields")
        image_id = str(row["image_identity_sha256"])
        record = test_records.get(image_id)
        if (record is None or row.get("position") != expected_position or
                row.get("image_content_sha256") != record.get("image_content_sha256") or
                row.get("source_group_sha256") != record.get("source_group_sha256") or
                row.get("camera_id") != record.get("camera_id")):
            raise FOADSKNGError("registered P0 event differs from the frozen TEST identity record")
        for key in ("event_id_sha256", "image_identity_sha256", "image_content_sha256", "source_group_sha256"):
            _require_digest(row.get(key), f"P0 event {key}")
        if row.get("source_group_sha256") in train_groups or row.get("image_content_sha256") in train_contents:
            raise FOADSKNGError("P0 TRAIN initialization overlaps a TEST group or image")
        parsed_events.append(RegisteredP0Event(
            position=int(row["position"]), original_position=int(row["original_position"]),
            event_id_sha256=str(row["event_id_sha256"]), image_identity_sha256=image_id,
            image_content_sha256=str(row["image_content_sha256"]),
            source_group_sha256=str(row["source_group_sha256"]),
            camera_id=int(row["camera_id"]), within_group_ordinal=int(row["within_group_ordinal"])))
    if len({event.image_identity_sha256 for event in parsed_events}) != len(parsed_events):
        raise FOADSKNGError("registered P0 stream repeats an image identity")
    groups: list[list[RegisteredP0Event]] = []
    for event in parsed_events:
        if not groups or groups[-1][0].source_group_sha256 != event.source_group_sha256:
            groups.append([])
        groups[-1].append(event)
    for group in groups:
        if (len(group) != KNG_VIEWS_PER_GROUP or [event.camera_id for event in group] != [1, 2, 3, 4, 5] or
                [event.within_group_ordinal for event in group] != [1, 2, 3, 4, 5]):
            raise FOADSKNGError("registered P0 stream must retain complete ordered five-view Source Groups")
    return P0Registration(category=category, dataset_release_sha256=release_sha,
                          identity_manifest_sha256=identity_sha,
                          role_manifest_sha256=role_sha,
                          initial_train_image_ids=tuple(selected_ids),
                          initial_train_roles=tuple(str(row["role"]) for row in selected),
                          stream_events=tuple(parsed_events))


def flatten_registered_initial_features(registration: P0Registration,
                                        features_by_identity: Mapping[str, np.ndarray]) -> np.ndarray:
    """Validate and flatten exactly the registered 68 normal-image feature maps."""
    expected = set(registration.initial_train_image_ids)
    if set(features_by_identity) != expected or len(features_by_identity) != 68:
        raise FOADSKNGError("initial features must contain exactly the registered 68 TRAIN-normal identities")
    arrays: list[np.ndarray] = []
    for image_id in registration.initial_train_image_ids:
        features = np.asarray(features_by_identity[image_id])
        if (features.shape != (*FEATURE_GRID, FEATURE_DIMENSIONS) or not np.isfinite(features).all()):
            raise FOADSKNGError("each initial ResNet feature map must be finite with shape (56,56,100)")
        arrays.append(np.asarray(features, dtype=np.float32).reshape(-1, FEATURE_DIMENSIONS))
    return np.ascontiguousarray(np.concatenate(arrays, axis=0), dtype=np.float32)


def seed_kng_initialization(seed: int = KNG_RANDOM_SEED) -> None:
    """Match K-online.setup_seed before the native random node selection."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True


def initialize_registered_p0_kng(registration: P0Registration,
                                 features_by_identity: Mapping[str, np.ndarray],
                                 source_api: KNGSourceAPI, *,
                                 distance_device: str | torch.device = "cuda") -> tuple[Any, dict[str, Any]]:
    """Initialize source K-NG on exactly 68 matched normal image identities."""
    initial_data = flatten_registered_initial_features(registration, features_by_identity)
    seed_kng_initialization(KNG_RANDOM_SEED)
    model = source_api.Klinks(initial_data, init_node=KNG_INIT_NODES,
                              age_max=KNG_AGE_MAX, mode=False)
    bind_pytorch18_distance_compatibility(model, device=distance_device)
    model.fit_network(epochs=KNG_FIT_EPOCHS)
    model.cluster_label()
    report = {
        "baseline": BASELINE_NAME,
        "initialization_kind": "MATCHED_68_NORMAL_PROTOCOL_ADAPTATION",
        "initial_normal_image_count": INITIAL_NORMAL_IMAGE_COUNT,
        "upstream_script_initial_normal_image_count": UPSTREAM_SCRIPT_INITIAL_NORMAL_IMAGE_COUNT,
        "initial_image_ids_sha256": _sha256(list(registration.initial_train_image_ids)),
        "initial_roles": {"SUPPORT": registration.initial_train_roles.count("SUPPORT"),
                           "C64": registration.initial_train_roles.count("C64")},
        "patch_feature_count": int(initial_data.shape[0]),
        "feature_dimensions": FEATURE_DIMENSIONS,
        "init_nodes": KNG_INIT_NODES,
        "age_max": KNG_AGE_MAX,
        "fit_epochs": KNG_FIT_EPOCHS,
        "kng_seed": KNG_RANDOM_SEED,
        "distance_compatibility": "PYTORCH_1_8_DIMENSION_1_SEMANTICS_RESTORED",
        "upstream_commit": source_api.verified_source.commit,
    }
    return model, report


@dataclass(frozen=True)
class FOADSKNGScore:
    position: int
    event_id_sha256: str
    image_identity_sha256: str
    source_group_sha256: str
    raw_image_score: float
    model_update_index: int


class FOADSKNGP0Runner:
    """Registered-order, label-free two-group scoring and native Batch-10 updates."""
    def __init__(self, registration: P0Registration, model: Any, source_api: KNGSourceAPI):
        if not getattr(model, "_foads_pytorch18_distance_compatibility", None):
            raise FOADSKNGError("Klinks must be initialized with the PyTorch 1.8 distance compatibility adapter")
        self.registration = registration
        self.model = model
        self.source_api = source_api
        self.next_position = 0
        self.update_count = 0
        self._pending_groups = 0
        self._pending_features: list[np.ndarray] = []

    def process_group(self, events: Sequence[RegisteredP0Event],
                      feature_maps: np.ndarray) -> tuple[FOADSKNGScore, ...]:
        start = self.next_position
        expected = self.registration.stream_events[start:start + KNG_VIEWS_PER_GROUP]
        if (len(expected) != KNG_VIEWS_PER_GROUP or len(events) != KNG_VIEWS_PER_GROUP or
                tuple(events) != expected):
            raise FOADSKNGError("runner accepts only the next complete registered P0 five-view group")
        maps = np.asarray(feature_maps)
        if (maps.ndim != 4 or maps.shape[0] != KNG_VIEWS_PER_GROUP or
                maps.shape[-1] != FEATURE_DIMENSIONS or not np.isfinite(maps).all()):
            raise FOADSKNGError("group feature maps must have shape (5,H,W,100) and be finite")
        group_sha = expected[0].source_group_sha256
        if any(event.source_group_sha256 != group_sha for event in expected):
            raise FOADSKNGError("a scoring batch cannot mix Source Groups")
        rows: list[FOADSKNGScore] = []
        for event, feature_map in zip(expected, maps):
            raw_score = paper_consistent_raw_image_score(self.model, feature_map, self.source_api)
            rows.append(FOADSKNGScore(
                position=event.position, event_id_sha256=event.event_id_sha256,
                image_identity_sha256=event.image_identity_sha256,
                source_group_sha256=event.source_group_sha256,
                raw_image_score=raw_score, model_update_index=self.update_count))
        self._pending_features.extend(np.asarray(item, dtype=np.float32).reshape(-1, FEATURE_DIMENSIONS)
                                      for item in maps)
        self._pending_groups += 1
        self.next_position += KNG_VIEWS_PER_GROUP
        if self._pending_groups == KNG_GROUPS_PER_UPDATE:
            # All ten event scores have been produced before the untouched source update call.
            online_data = np.ascontiguousarray(np.concatenate(self._pending_features, axis=0), dtype=np.float32)
            if len(online_data) != KNG_IMAGE_BATCH_SIZE * maps.shape[1] * maps.shape[2]:
                raise FOADSKNGError("K-NG update batch does not contain exactly ten complete images")
            self.model.online_fit(online_data)
            self.update_count += 1
            self._pending_features.clear()
            self._pending_groups = 0
        return tuple(rows)

    def finish(self) -> dict[str, int]:
        if self.next_position != len(self.registration.stream_events):
            raise FOADSKNGError("cannot finish before all registered P0 TEST images are scored")
        if self._pending_groups not in (0, 1):
            raise FOADSKNGError("invalid incomplete K-NG update batch")
        return {
            "scored_images": self.next_position,
            "completed_batch10_updates": self.update_count,
            "final_unupdated_groups": self._pending_groups,
            "final_unupdated_images": self._pending_groups * KNG_VIEWS_PER_GROUP,
            "paid_labels": 0,
            "test_labels_or_masks_read": 0,
        }


def baseline_protocol_record(registration: P0Registration,
                             source: VerifiedKNGSource,
                             resnet18_weights_sha256: str | None = None) -> dict[str, Any]:
    """Return the immutable method/source identity and explicit adaptations."""
    if resnet18_weights_sha256 is not None:
        _require_digest(resnet18_weights_sha256, "ResNet18 weights identity")
    return {
        "baseline": BASELINE_NAME,
        "source_repository": UPSTREAM_REPOSITORY,
        "source_commit": source.commit,
        "source_files_sha256": dict(source.file_sha256),
        "identity_manifest_sha256": registration.identity_manifest_sha256,
        "role_manifest_sha256": registration.role_manifest_sha256,
        "dataset_release_sha256": registration.dataset_release_sha256,
        "category": registration.category,
        "stream_seed": 0,
        "torch_version": str(torch.__version__),
        "current_pairwise_distance_probe_shape": list(
            torch.pairwise_distance(torch.zeros((1, 3, 1)), torch.zeros((1, 3, 4))).shape),
        "runtime_compatibility_restoration": {
            "name": "PYTORCH_1_8_DIMENSION_1_SEMANTICS_RESTORED",
            "formula": "vector_norm(test_batch.unsqueeze(-1)-centeroid+1e-6,ord=2,dim=1)",
            "preserves_source_sqrt_and_top2_order": True,
        },
        "resnet18_weights_sha256": resnet18_weights_sha256,
        "approved_protocol_adaptations": [
            "Paper-consistent Equation (11) quadratic form replaces the pinned source's discrepant einsum; upstream bytes remain unchanged.",
            "Per-image maximum is taken from the raw interpolated, sigma-4 map before batch-global normalization.",
            "Registered Real-IAD P0 group order replaces the upstream shuffled TEST loader; updates follow two complete five-view groups (10 images).",
            "All registered 68 TRAIN-normal identities initialize K-NG; the upstream script processes only its first 10-image training batch.",
        ],
        "upstream_components_retained": [
            "ResNet18 layer1/layer2/layer3 embedding_concat and seed-1024 dimension selection",
            "Klinks random-node initialization, 3136-node target, age_max=25, 10 fit epochs, topology, thresholds and covariance regularization",
            "Klinks.online_fit pseudo-normal screening, topology transitions and incremental covariance/mean update",
            "224px bilinear interpolation and Gaussian sigma 4 smoothing",
        ],
        "initial_normal_image_count": INITIAL_NORMAL_IMAGE_COUNT,
        "initialization_kind": "MATCHED_68_NORMAL_PROTOCOL_ADAPTATION",
        "paid_labels": 0,
        "test_label_or_mask_access": False,
    }
