from __future__ import annotations

import copy
import hashlib
import os
import random
from types import SimpleNamespace, MethodType

import numpy as np
import pytest
import torch
from torch import nn

from experiments.calibration.final_benchmark import foads_kng as kng


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha(value) -> str:
    import json
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _identity_manifest() -> dict:
    category = "synthetic_category"
    records = []
    selected = []
    for index in range(68):
        image_id = _digest(f"train-image-{index}")
        content = _digest(f"train-content-{index}")
        group = _digest(f"train-group-{index}")
        role = "SUPPORT" if index < 4 else "C64"
        records.append({
            "dataset_id": "realiad", "category": category, "split": "train",
            "original_position": index, "image_identity_sha256": image_id,
            "file_locator_sha256": _digest(f"train-locator-{index}"),
            "image_content_sha256": content, "image_bytes": 100,
            "source_group_sha256": group, "camera_id": 1,
            "train_normal_eligible": True,
        })
        selected.append({"role": role, "image_identity_sha256": image_id,
                         "image_content_sha256": content,
                         "source_group_sha256": group})
    release = _digest("registered-release")
    role = {"dataset_id": "realiad", "dataset_release_sha256": release,
            "category": category, "support_seed": 0, "selected": selected,
            "static36_c64_partition": []}
    events = []
    for group_index in range(2):
        group_sha = _digest(f"test-group-{group_index}")
        for camera in range(1, 6):
            index = group_index * 5 + camera - 1
            image_id = _digest(f"test-image-{index}")
            content = _digest(f"test-content-{index}")
            event = {
                "position": index,
                "original_position": 1000 + index,
                "event_id_sha256": _digest(f"event-{index}"),
                "image_identity_sha256": image_id,
                "image_content_sha256": content,
                "source_group_sha256": group_sha,
                "camera_id": camera,
                "within_group_ordinal": camera,
            }
            events.append(event)
            records.append({
                "dataset_id": "realiad", "category": category, "split": "test",
                "original_position": 1000 + index, "image_identity_sha256": image_id,
                "file_locator_sha256": _digest(f"test-locator-{index}"),
                "image_content_sha256": content, "image_bytes": 100,
                "source_group_sha256": group_sha, "camera_id": camera,
                "train_normal_eligible": None,
            })
    return {
        "schema": "final-benchmark.identity-manifest.v1",
        "dataset_id": "realiad", "identity_census_status": "PASS",
        "identity_records": records,
        "role_manifests": [role],
        "stream_manifests": [{"category": category, "stream_seed": 0,
                               "event_count": len(events), "events": events}],
    }


def _registration() -> kng.P0Registration:
    manifest = _identity_manifest()
    role = manifest["role_manifests"][0]
    return kng.build_registered_p0(
        manifest, "synthetic_category",
        expected_identity_manifest_sha256=_canonical_sha(manifest),
        expected_role_manifest_sha256=_canonical_sha(role),
        expected_dataset_release_sha256=role["dataset_release_sha256"],
    )


@pytest.fixture(scope="module")
def source_api() -> kng.KNGSourceAPI:
    source_root = os.environ.get("KNG_SOURCE_ROOT")
    if not source_root:
        pytest.skip("set KNG_SOURCE_ROOT to the separately obtained pinned K-NG source tree")
    return kng.load_upstream_api(source_root)


def test_equation_11_matches_linear_algebra_and_documents_upstream_difference(source_api):
    inverse = np.array([[2.0, -1.9], [-1.9, 2.0]], dtype=np.float64)
    deltas = np.array([[1.0, 1.0], [0.5, -0.5]], dtype=np.float64)
    inverses = np.repeat(inverse[None, :, :], len(deltas), axis=0)
    means = np.zeros_like(deltas)

    paper = kng.paper_consistent_mahalanobis(deltas, means, inverses)
    expected = np.array([
        np.sqrt(np.linalg.multi_dot([delta.reshape(1, 2), inverse,
                                     delta.reshape(2, 1)]).item())
        for delta in deltas
    ])
    upstream = np.asarray(source_api.upstream_matrix_mahalanobis(
        deltas, means, inverses)).reshape(-1)

    np.testing.assert_allclose(paper, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(paper, [0.447213595499958, 1.396424004376894],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(upstream, [0.447213595499958, 0.223606797749979],
                               rtol=0, atol=1e-12)
    assert np.argsort(paper).tolist() == [0, 1]
    assert np.argsort(upstream).tolist() == [1, 0]


def test_quadratic_form_clips_only_roundoff_and_rejects_material_negative():
    delta = np.array([[1.0, 0.0]])
    tiny_negative = np.array([[[-1e-15, 0.0], [0.0, 1.0]]])
    result = kng.paper_consistent_mahalanobis(delta, np.zeros_like(delta), tiny_negative)
    assert result.tolist() == [0.0]
    material_negative = np.array([[[-1e-6, 0.0], [0.0, 1.0]]])
    with pytest.raises(kng.FOADSKNGError, match="materially negative"):
        kng.paper_consistent_mahalanobis(delta, np.zeros_like(delta), material_negative)


class _FixedFeature(nn.Module):
    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.register_buffer("tensor", tensor)

    def forward(self, _x):
        return self.tensor.expand(_x.shape[0], -1, -1, -1)


class _SyntheticResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(71)
        self.layer1 = nn.Sequential(_FixedFeature(torch.randn(1, 64, 56, 56, generator=generator)))
        self.layer2 = nn.Sequential(_FixedFeature(torch.randn(1, 128, 28, 28, generator=generator)))
        self.layer3 = nn.Sequential(_FixedFeature(torch.randn(1, 256, 14, 14, generator=generator)))

    def forward(self, images):
        x = self.layer1(images)
        x = self.layer2(x)
        return self.layer3(x)


def test_resnet_embedding_and_dimension_selection_match_pinned_konline(source_api):
    random.seed(1024)
    expected_indices = tuple(random.sample(range(448), 100))
    assert kng.kng_dimension_indices() == expected_indices

    model = _SyntheticResNet18().eval()
    images = torch.zeros((1, 3, 224, 224))
    adapted = kng.extract_kng_embeddings(model, images, source_api)
    with torch.no_grad():
        activations = [layer[-1](images).cpu() for layer in
                       (model.layer1, model.layer2, model.layer3)]
    reference = source_api.embedding_concat(
        source_api.embedding_concat(activations[0], activations[1]), activations[2])
    reference = torch.index_select(reference, 1, torch.tensor(expected_indices))
    reference = reference.permute(0, 2, 3, 1).contiguous().numpy()
    assert adapted.shape == (1, 56, 56, 100)
    np.testing.assert_array_equal(adapted, reference)


class _SyntheticKlinks:
    def __init__(self, center: np.ndarray):
        self.node_list = np.array([0])
        self.centeroid = center.reshape(1, -1).copy()
        self.network = SimpleNamespace(nodes={0: {
            "vector": center.copy(), "cov_inv": np.eye(len(center)),
        }})
        self.online_calls = 0
        self._foads_pytorch18_distance_compatibility = {"device": "cpu", "synthetic": True}

    def gpu_cal_dist(self, data):
        values = torch.as_tensor(np.asarray(data), dtype=torch.float32)
        center = torch.as_tensor(self.centeroid[0], dtype=torch.float32)
        return torch.linalg.vector_norm(values - center, dim=1, keepdim=True)

    def online_fit(self, data):
        self.online_calls += 1


def _simple_features(value: float) -> np.ndarray:
    features = np.zeros((1, 1, 100), dtype=np.float32)
    features[0, 0, 0] = value
    return features


def test_raw_singleton_and_sibling_independence_with_fixed_state(source_api, monkeypatch):
    model = _SyntheticKlinks(np.zeros(100, dtype=np.float64))
    registration = _registration()
    values = (0.1, 0.5, 1.0, 1.5, 2.0)
    maps_a = np.zeros((5, *kng.FEATURE_GRID, kng.FEATURE_DIMENSIONS), dtype=np.float32)
    maps_a[..., 0] = np.asarray(values, dtype=np.float32)[:, None, None]
    maps_b = maps_a.copy()
    maps_b[1:] += 7.0

    # The raw scorer accepts one correctly shaped image at a time. The runner
    # scheduling check below substitutes only the expensive map scorer.
    singleton_map = kng.paper_consistent_raw_map(model, maps_a[0], source_api)
    assert singleton_map.shape == (224, 224)
    assert np.isfinite(singleton_map).all() and float(singleton_map.max()) > 0
    assert np.isfinite(float(singleton_map.max()))

    monkeypatch.setattr(
        kng, "paper_consistent_raw_image_score",
        lambda _model, feature_map, _source_api: float(np.asarray(feature_map)[0, 0, 0]))
    runner_a = kng.FOADSKNGP0Runner(registration, copy.deepcopy(model), source_api)
    runner_b = kng.FOADSKNGP0Runner(registration, copy.deepcopy(model), source_api)
    events = registration.stream_events[:5]
    scores_a = runner_a.process_group(events, maps_a)
    scores_b = runner_b.process_group(events, maps_b)
    assert scores_a[0].raw_image_score == scores_b[0].raw_image_score
    assert runner_a.model.online_calls == runner_b.model.online_calls == 0
    assert runner_a.update_count == 0

    raw = np.array([row.raw_image_score for row in scores_a])
    assert np.isfinite(raw).all() and raw[0] > 0
    normalized = (raw - raw.min()) / (raw.max() - raw.min())
    assert np.argsort(raw).tolist() == np.argsort(normalized).tolist()


def test_two_groups_are_scored_before_one_batch10_update_call(source_api, monkeypatch):
    registration = _registration()
    model = _SyntheticKlinks(np.zeros(100, dtype=np.float64))
    online_calls = []
    def record_online_fit(self, data):
        self.online_calls += 1
        online_calls.append(np.asarray(data).copy())
    model.online_fit = MethodType(record_online_fit, model)
    monkeypatch.setattr(
        kng, "paper_consistent_raw_image_score",
        lambda _model, feature_map, _source_api: float(np.asarray(feature_map)[0, 0, 0]))
    maps = np.zeros((10, *kng.FEATURE_GRID, kng.FEATURE_DIMENSIONS), dtype=np.float32)
    maps[..., 0] = np.arange(0.1, 1.1, 0.1, dtype=np.float32)[:, None, None]
    runner = kng.FOADSKNGP0Runner(registration, model, source_api)

    first = runner.process_group(registration.stream_events[:5], maps[:5])
    assert len(first) == 5
    assert online_calls == []
    assert runner.update_count == 0
    second = runner.process_group(registration.stream_events[5:10], maps[5:])
    assert len(second) == 5
    assert len(online_calls) == 1
    assert runner.update_count == 1
    assert all(row.model_update_index == 0 for row in first + second)

    expected_online_data = maps.reshape(-1, 100).astype(np.float32)
    np.testing.assert_array_equal(online_calls[0], expected_online_data)
    assert runner.finish() == {
        "scored_images": 10, "completed_batch10_updates": 1,
        "final_unupdated_groups": 0, "final_unupdated_images": 0,
        "paid_labels": 0, "test_labels_or_masks_read": 0,
    }


def test_feature_geometry_is_rejected_before_scoring_or_update(source_api):
    model = _SyntheticKlinks(np.zeros(100, dtype=np.float64))
    registration = _registration()
    runner = kng.FOADSKNGP0Runner(registration, model, source_api)

    with pytest.raises(kng.FOADSKNGError, match=r"\(56,56,100\)"):
        kng.paper_consistent_raw_map(model, _simple_features(1.0), source_api)
    with pytest.raises(kng.FOADSKNGError, match=r"\(5,56,56,100\)"):
        runner.process_group(registration.stream_events[:5],
                             np.zeros((5, 1, 1, 100), dtype=np.float32))
    assert model.online_calls == 0
    assert runner.next_position == runner.update_count == 0


@pytest.mark.parametrize("channels,node_count", [(3, 4), (5, 2), (2, 5)])
def test_pytorch18_distance_methods_match_explicit_formula_and_node_order(
        source_api, channels, node_count):
    generator = np.random.default_rng(400 + channels * 10 + node_count)
    training = generator.normal(size=(12, channels)).astype(np.float32)
    model = source_api.Klinks(training, init_node=node_count, age_max=25, mode=False)
    model.node_list = np.arange(node_count, dtype=np.int64) * 7 + 101
    model.centeroid = generator.normal(size=(node_count, channels)).astype(np.float32)
    kng.bind_pytorch18_distance_compatibility(model, device="cpu")
    batch = generator.normal(size=(6, channels)).astype(np.float32)

    test_batch = torch.from_numpy(batch)
    centroid = torch.from_numpy(model.centeroid).t().unsqueeze(0)
    pre_sqrt = torch.linalg.vector_norm(
        test_batch.unsqueeze(-1) - centroid + 1e-6, ord=2, dim=1)
    expected_gpu = torch.sqrt(pre_sqrt)
    observed_gpu = model.gpu_cal_dist(batch)
    assert tuple(observed_gpu.shape) == (len(batch), node_count)
    torch.testing.assert_close(observed_gpu, expected_gpu, rtol=0, atol=0)

    expected_values, expected_indexes = torch.topk(
        pre_sqrt, k=2, dim=1, largest=False, sorted=True)
    observed_values, observed_indexes = model.online_gpu_cal_dist(batch)
    assert tuple(observed_values.shape) == tuple(observed_indexes.shape) == (len(batch), 2)
    torch.testing.assert_close(observed_values, torch.sqrt(expected_values), rtol=0, atol=0)
    torch.testing.assert_close(observed_indexes, expected_indexes, rtol=0, atol=0)
    mapped_node_ids = model.node_list[observed_indexes.numpy()]
    np.testing.assert_array_equal(mapped_node_ids, model.node_list[expected_indexes.numpy()])

    # cdist is only an independent ranking reference, never the adapter value.
    reference = torch.cdist(test_batch, torch.from_numpy(model.centeroid))
    torch.testing.assert_close(torch.topk(reference, 2, dim=1, largest=False).indices,
                               expected_indexes, rtol=0, atol=0)


def _model_state(model):
    nodes = []
    for node_id in sorted(model.network.nodes):
        node = model.network.nodes[node_id]
        nodes.append((node_id, np.asarray(node["vector"]).copy(),
                      np.asarray(node["cov"]).copy(), np.asarray(node["cov_inv"]).copy(),
                      float(node["threshold"]), int(node["nodes_num"]), float(node["error"])))
    edges = sorted((min(left, right), max(left, right), int(data["age"]))
                   for left, right, data in model.network.edges(data=True))
    return nodes, edges, model.node_list.copy(), model.centeroid.copy()


def _assert_model_state_equal(actual, expected):
    assert len(actual[0]) == len(expected[0])
    for left, right in zip(actual[0], expected[0]):
        assert left[0] == right[0]
        for left_array, right_array in zip(left[1:4], right[1:4]):
            np.testing.assert_array_equal(left_array, right_array)
        assert left[4:] == right[4:]
    assert actual[1] == expected[1]
    np.testing.assert_array_equal(actual[2], expected[2])
    np.testing.assert_array_equal(actual[3], expected[3])


def _train_four_node_kng(source_api):
    generator = np.random.default_rng(88)
    centers = np.zeros((4, 100), dtype=np.float32)
    centers[:, 0] = [-12.0, -4.0, 4.0, 12.0]
    centers[:, 1] = [0.0, 8.0, 0.0, -8.0]
    training = np.concatenate([
        center + generator.normal(scale=0.12, size=(12, 100)).astype(np.float32)
        for center in centers
    ])
    # Seed 1033 samples one initial node from each synthetic cluster while
    # exercising the unmodified Klinks random initialization and fit methods.
    np.random.seed(1033)
    model = source_api.Klinks(training, init_node=4, age_max=25, mode=False)
    kng.bind_pytorch18_distance_compatibility(model, device="cpu")
    model.fit_network(epochs=2)
    model.cluster_label()
    assert len(model.network.nodes) >= 2
    assert len(model.node_list) == len(model.centeroid) == len(model.network.nodes)
    for node_id in model.node_list:
        node = model.network.nodes[int(node_id)]
        assert np.asarray(node["vector"]).shape == (100,)
        assert np.asarray(node["cov"]).shape == (100, 100)
        assert np.asarray(node["cov_inv"]).shape == (100, 100)
        assert np.isfinite(node["threshold"])
    return model, centers


def test_native_initial_assignment_and_batch10_update_match_pinned_online_fit(
        source_api, monkeypatch):
    registration = _registration()
    model, centers = _train_four_node_kng(source_api)
    reference = copy.deepcopy(model)
    kng.bind_pytorch18_distance_compatibility(reference, device="cpu")

    generator = np.random.default_rng(99)
    samples = [centers[index % 4] + generator.normal(scale=0.05, size=100).astype(np.float32)
               for index in range(8)]
    outlier_a = centers[0].copy(); outlier_a[0] += 1000
    outlier_b = centers[3].copy(); outlier_b[0] -= 1000
    samples.extend((outlier_a, outlier_b))
    online_data = np.stack(samples).astype(np.float32)
    maps = np.broadcast_to(
        online_data[:, None, None, :],
        (10, *kng.FEATURE_GRID, kng.FEATURE_DIMENSIONS)).copy()
    monkeypatch.setattr(
        kng, "paper_consistent_raw_image_score",
        lambda _model, feature_map, _source_api: float(np.asarray(feature_map)[0, 0, 0]))

    observed_pseudo_normal = {}
    source_distance = model.online_gpu_cal_dist
    def capture_online_distance(self, data):
        values, indexes = source_distance(data)
        if len(data) == 10:
            node_ids = self.node_list[indexes[:, 0].numpy()]
            thresholds = np.array([self.network.nodes[int(node_id)]["threshold"]
                                   for node_id in node_ids])
            observed_pseudo_normal["accepted"] = values[:, 0].numpy() <= thresholds
        return values, indexes
    model.online_gpu_cal_dist = MethodType(capture_online_distance, model)

    native_online_fit = model.online_fit
    runner_batches = []
    def capture_runner_online_fit(self, data):
        runner_batches.append(np.asarray(data).copy())
    model.online_fit = MethodType(capture_runner_online_fit, model)
    runner = kng.FOADSKNGP0Runner(registration, model, source_api)
    initial_state = _model_state(model)
    first = runner.process_group(registration.stream_events[:5], maps[:5])
    assert len(first) == 5 and runner.update_count == 0
    _assert_model_state_equal(_model_state(model), initial_state)
    second = runner.process_group(registration.stream_events[5:], maps[5:])
    assert len(second) == 5 and runner.update_count == 1
    assert runner.finish()["completed_batch10_updates"] == 1
    assert len(runner_batches) == 1
    np.testing.assert_array_equal(runner_batches[0], maps.reshape(-1, 100))
    _assert_model_state_equal(_model_state(model), initial_state)
    # The native Klinks update is exercised on the same ten deterministic
    # image-level synthetic observations after the runner's full-grid receipt.
    model.online_fit = native_online_fit
    model.online_fit(online_data)

    assert observed_pseudo_normal["accepted"].shape == (10,)
    reference.online_fit(online_data)
    _assert_model_state_equal(_model_state(model), _model_state(reference))
    updated_state = _model_state(model)
    assert observed_pseudo_normal["accepted"].any()
    assert (~observed_pseudo_normal["accepted"]).any()
    assert any(not np.array_equal(old[1], new[1])
               for old, new in zip(initial_state[0], updated_state[0]))


def test_registered_order_and_outcome_access_are_fail_closed():
    manifest = _identity_manifest()
    role = manifest["role_manifests"][0]
    registration = _registration()
    assert len(registration.initial_train_image_ids) == 68
    assert registration.initial_train_roles.count("SUPPORT") == 4
    assert registration.initial_train_roles.count("C64") == 64
    assert [event.position for event in registration.stream_events] == list(range(10))
    assert [event.camera_id for event in registration.stream_events] == [1, 2, 3, 4, 5] * 2

    manifest["stream_manifests"][0]["events"][0]["mask_path"] = "forbidden"
    with pytest.raises(kng.FOADSKNGError, match="prohibited outcome field"):
        kng.build_registered_p0(
            manifest, "synthetic_category",
            expected_identity_manifest_sha256=_canonical_sha(manifest),
            expected_role_manifest_sha256=_canonical_sha(role),
            expected_dataset_release_sha256=role["dataset_release_sha256"],
        )


def test_registered_stream_rejects_missing_complete_five_view_group():
    manifest = _identity_manifest()
    role = manifest["role_manifests"][0]
    stream = manifest["stream_manifests"][0]
    stream["events"] = stream["events"][:5]
    stream["event_count"] = len(stream["events"])

    with pytest.raises(kng.FOADSKNGError,
                       match="every registered TEST image identity exactly once"):
        kng.build_registered_p0(
            manifest, "synthetic_category",
            expected_identity_manifest_sha256=_canonical_sha(manifest),
            expected_role_manifest_sha256=_canonical_sha(role),
            expected_dataset_release_sha256=role["dataset_release_sha256"],
        )


def test_initial_features_require_exact_registered_identity_set():
    registration = _registration()
    with pytest.raises(kng.FOADSKNGError, match="exactly the registered 68"):
        kng.flatten_registered_initial_features(registration, {})
    assert "labels" not in kng.paper_consistent_raw_image_score.__code__.co_varnames
    assert "masks" not in kng.FOADSKNGP0Runner.process_group.__code__.co_varnames
