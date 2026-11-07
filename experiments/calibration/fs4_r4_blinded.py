"""Research-only blinded registration and pixel-preserving image reader."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, Mapping

from PIL import Image

from .fs4_durable_ledger import digest
from .fs4_r2_context import ALLOWED_DEVELOPMENT_CATEGORIES, FS4BoundaryError, SUPPORT_KEYS
from .fs4_r2_security import WindowsPrivateStore

HEX = re.compile(r"[0-9a-f]{64}")
EVENT_KEYS = frozenset(("group", "event", "resource", "position", "view", "width", "height",
                        "content_sha256", "resource_sha256", "pixel_sha256"))


def _hex(value: Any) -> bool:
    return isinstance(value, str) and HEX.fullmatch(value) is not None


def validate_blinded_execution(payload: Mapping[str, Any], ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a label-free registration and its preparation receipt."""
    from experiments.calibration.profiles import NativeScorerConfig
    required = {"schema", "run_id", "protocol", "source_commit", "query_seed", "model_state_sha256",
                "checkpoint_sha256", "hub_source_tree_sha256", "scorer_config_sha256", "preflight_receipt_sha256",
                "dataset_root", "image_root", "category_order", "categories", "source_mapping_sha256",
                "dataset_identity_sha256"}
    if set(payload) != required or payload["schema"] != "FS4_R4_BLINDED_EXECUTION_V1":
        raise FS4BoundaryError("invalid blinded research registration")
    for key in ("run_id", "protocol", "source_commit", "query_seed", "model_state_sha256", "dataset_identity_sha256"):
        if payload[key] != ledger.get(key):
            raise FS4BoundaryError("blinded registration identity differs")
    if payload["scorer_config_sha256"] != NativeScorerConfig().sha256:
        raise FS4BoundaryError("native scorer configuration differs")
    for key in ("model_state_sha256", "checkpoint_sha256", "hub_source_tree_sha256", "scorer_config_sha256",
                "preflight_receipt_sha256", "source_mapping_sha256", "dataset_identity_sha256"):
        if not _hex(payload[key]):
            raise FS4BoundaryError("invalid blinded registration digest")
    if any(not isinstance(payload[k], str) or not Path(payload[k]).is_absolute() for k in ("dataset_root", "image_root")):
        raise FS4BoundaryError("blinded registration requires absolute resource roots")
    source_root, image_root = Path(payload["dataset_root"]), Path(payload["image_root"])
    if source_root == image_root or source_root.is_relative_to(image_root) or image_root.is_relative_to(source_root):
        raise FS4BoundaryError("blinded resources must be separate from the source dataset")
    categories, order = payload["categories"], payload["category_order"]
    if (not isinstance(categories, dict) or not set(categories).issubset(ALLOWED_DEVELOPMENT_CATEGORIES) or
            set(categories) != set(ledger["expected_events_by_category"]) or not isinstance(order, list) or
            len(order) != len(categories) or set(order) != set(categories)):
        raise FS4BoundaryError("blinded categories or order differ")
    position, seen_events, seen_resources, seen_groups = 0, set(), set(), set()
    support_groups = set()
    for category in order:
        info = categories[category]
        if set(info) != {"supports", "events"} or len(info["supports"]) != 4:
            raise FS4BoundaryError("four original registered Supports are required")
        if any(set(item) != SUPPORT_KEYS for item in info["supports"]):
            raise FS4BoundaryError("invalid Support registration")
        groups = [item["group"] for item in info["supports"]]
        if len(set(groups)) != 4 or groups != ledger["support_groups_by_category"][category]:
            raise FS4BoundaryError("registered Support groups differ")
        support_groups.update(groups)
        if len(info["events"]) != ledger["expected_events_by_category"][category]:
            raise FS4BoundaryError("registered event count differs")
        views: dict[str, set[int]] = defaultdict(set)
        rows = []
        for item in info["events"]:
            if set(item) != EVENT_KEYS or item["position"] != position:
                raise FS4BoundaryError("blinded event schema or ordering differs")
            if (any(not _hex(item[k]) for k in ("group", "event", "content_sha256", "resource_sha256", "pixel_sha256")) or
                    item["event"] in seen_events or item["resource"] in seen_resources or
                    not isinstance(item["resource"], str) or not re.fullmatch(r"[0-9a-f]{64}\.png", item["resource"]) or
                    type(item["view"]) is not int or item["view"] not in range(1, 6) or
                    any(type(item[k]) is not int or item[k] <= 0 for k in ("width", "height"))):
                raise FS4BoundaryError("blinded image identity or geometry differs")
            if item["view"] in views[item["group"]]:
                raise FS4BoundaryError("duplicate blinded group view")
            views[item["group"]].add(item["view"])
            rows.append({"group": item["group"], "event": item["event"]})
            seen_events.add(item["event"]); seen_resources.add(item["resource"]); position += 1
        if any(v != {1, 2, 3, 4, 5} for v in views.values()) or seen_groups.intersection(views):
            raise FS4BoundaryError("blinded groups or camera views differ")
        seen_groups.update(views)
        if digest(rows) != ledger["stream_sha256_by_category"][category]:
            raise FS4BoundaryError("blinded stream digest differs")
    resource_ids = {name[:-4] for name in seen_resources}
    if (support_groups.intersection(seen_groups) or seen_events.intersection(seen_groups) or
            resource_ids.intersection(seen_groups) or resource_ids.intersection(seen_events)):
        raise FS4BoundaryError("Support and event identities overlap")
    identity = {k: payload[k] for k in ("dataset_root", "image_root", "category_order", "categories", "source_mapping_sha256")}
    if digest(identity) != payload["dataset_identity_sha256"]:
        raise FS4BoundaryError("blinded dataset identity differs")
    try:
        store = WindowsPrivateStore(Path(payload["image_root"]), create=False)
        registered = json.loads(store.read("registration.json"))
        receipt = json.loads(store.read("receipt.json"))
        complete = json.loads(store.read("COMPLETE.json"))
        if (registered != payload or receipt != {"schema": "FS4_R4_PREPARATION_RECEIPT_V1",
                "mapping_sha256": payload["source_mapping_sha256"], "execution_sha256": digest(payload),
                "event_count": position} or complete != {"receipt_sha256": digest(receipt)}):
            raise ValueError("registration receipt differs")
    except Exception:
        raise FS4BoundaryError("blinded registration or preparation receipt failed") from None
    return copy.deepcopy(dict(payload))


def read_blinded_image(payload: Mapping[str, Any], row: Mapping[str, Any]) -> Image.Image:
    """Read only a noninformative resource and verify bytes, pixels and geometry."""
    try:
        data = WindowsPrivateStore(Path(payload["image_root"]), create=False).read(row["resource"])
        if hashlib.sha256(data).hexdigest() != row["resource_sha256"]:
            raise ValueError("resource checksum")
        with Image.open(io.BytesIO(data)) as source:
            if source.format != "PNG" or source.info or source.mode != "RGB":
                raise ValueError("image metadata or format")
            image = source.copy()
        if (image.size != (row["width"], row["height"]) or
                hashlib.sha256(image.tobytes()).hexdigest() != row["pixel_sha256"]):
            raise ValueError("image geometry or pixels")
        return image
    except Exception:
        raise FS4BoundaryError("blinded image resource failed verification") from None
