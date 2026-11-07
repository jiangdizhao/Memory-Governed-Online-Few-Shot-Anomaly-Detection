"""Dimensionally explicit FS4 campaign storage and hardware feasibility gates."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from .fs4_durable_ledger import digest
from .fs4_joint_campaign_contract import EXPECTED_CATEGORIES, EXPECTED_EVENTS, load_spec, validate_registration

FEATURE_BYTES_PER_GROUP = 784 * 4096 * 4
CHECKPOINT_NATIVE_OVERHEAD_BYTES = 256 * 1024**2
OUTPUT_LEDGER_RESERVE_BYTES = 2 * 1024**3
RESERVE_FRACTION = 0.25


def checkpoint_schedule() -> list[dict[str, Any]]:
    """Return every frozen 1,000-event and category-boundary snapshot size."""
    result = []
    global_start = 0
    for category in EXPECTED_CATEGORIES:
        count = EXPECTED_EVENTS[category]
        global_end = global_start + count
        boundaries = {global_start, global_end}
        boundaries.update(x for x in range(((global_start // 1000) + 1) * 1000, global_end + 1, 1000))
        for completed in sorted(boundaries):
            within = completed - global_start
            groups = min(count // 5, within // 5)
            result.append({"category": category, "completed_campaign_events": completed,
                           "category_events_completed": within, "max_cached_normal_groups": groups,
                           "max_e_cache_bytes": groups * FEATURE_BYTES_PER_GROUP})
        global_start = global_end
    return result


def estimate_campaign_resources() -> dict[str, Any]:
    spec = load_spec()
    schedule = checkpoint_schedule()
    e_snapshots = sum(row["max_e_cache_bytes"] for row in schedule)
    all_checkpoints = 3 * len(schedule)
    e_trajectories = 2  # S and CE carry E; CL is L-only.
    feature_checkpoint_bytes = e_trajectories * e_snapshots
    structural_checkpoint_bytes = all_checkpoints * CHECKPOINT_NATIVE_OVERHEAD_BYTES
    checkpoint_bytes = feature_checkpoint_bytes + structural_checkpoint_bytes
    max_rescores_per_e_arm = sum(n * (n + 1) // 2 for n in (EXPECTED_EVENTS[c] // 5 for c in EXPECTED_CATEGORIES))
    return {
        "schema": "FS4_JOINT_CAMPAIGN_RESOURCE_ESTIMATE_V1",
        "registered_events": sum(EXPECTED_EVENTS.values()),
        "trajectory_event_encodings": 3 * sum(EXPECTED_EVENTS.values()),
        "trajectory_base_native_g_gc_scores": 3 * sum(EXPECTED_EVENTS.values()),
        "four_support_native_self_scores": 3 * len(EXPECTED_CATEGORIES) * 4,
        "native_fixed_bank_bytes_per_category": 8192 * 4096 * 2,
        "dynamic_greedy_prototype_bytes_per_category": 1024 * 4096 * 2,
        "dynamic_c1_prototype_bytes_per_category": 1024 * 4096 * 2,
        "max_exact_e_rescore_invocations_per_e_trajectory": max_rescores_per_e_arm,
        "max_exact_e_rescore_invocations_campaign": e_trajectories * max_rescores_per_e_arm,
        "e_feature_bytes_per_group": FEATURE_BYTES_PER_GROUP,
        "e_feature_bytes_at_capacity": 1024 * FEATURE_BYTES_PER_GROUP,
        "checkpoint_count_upper_bound": all_checkpoints,
        "checkpoint_schedule": schedule,
        "checkpoint_feature_bytes_upper_bound": feature_checkpoint_bytes,
        "checkpoint_native_and_controller_overhead_upper_bound": structural_checkpoint_bytes,
        "checkpoint_bytes_before_reserve": checkpoint_bytes,
        "checkpoint_bytes_with_25_percent_reserve": int(checkpoint_bytes * (1 + RESERVE_FRACTION)),
        "additional_ledger_mapping_and_completion_reserve": OUTPUT_LEDGER_RESERVE_BYTES,
        "host_free_ram_minimum_bytes": spec["resource_controls"]["host_free_ram_minimum_bytes"],
        "gpu_name_required": spec["resource_controls"]["gpu_name_required"],
        "gpu_total_memory_minimum_bytes": spec["resource_controls"]["gpu_total_memory_minimum_bytes"],
        "per_trajectory_wall_limit_seconds": spec["resource_controls"]["trajectory_wall_limit_seconds"],
        "campaign_wall_limit_seconds": spec["resource_controls"]["campaign_wall_limit_seconds"],
        "reference_rescore_timing_seconds": 0.175,
        "reference_timing_is_campaign_prediction": False,
    }


def _host_available_bytes() -> int | None:
    if os.name != "nt":
        return None
    class _MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    try:
        status = _MemoryStatus(); status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
    except Exception:
        pass
    return None


def _gpu_status() -> dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False, "name": None, "total_bytes": None, "free_bytes": None}
        free, total = torch.cuda.mem_get_info()
        return {"available": True, "name": torch.cuda.get_device_name(),
                "total_bytes": int(total), "free_bytes": int(free)}
    except Exception:
        return {"available": False, "name": None, "total_bytes": None, "free_bytes": None}


def resource_preflight(*, registration: Mapping[str, Any], image_root: Path,
                       output_root: Path, private_root: Path,
                       benchmark_receipt: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fail closed before Stage A when disk, RAM, GPU or full-native timing is unknown."""
    registration = validate_registration(registration)
    image_root, output_root, private_root = map(lambda p: Path(p).absolute(), (image_root, output_root, private_root))
    if output_root.exists():
        raise FileExistsError("campaign output namespace must be fresh and non-overwriting")
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for a, b in
           ((image_root, output_root), (private_root, output_root), (private_root, image_root))):
        raise ValueError("campaign resources, private mapping and output root must be separate")
    estimate = estimate_campaign_resources()
    missing_resources = []
    image_bytes = 0
    for category in EXPECTED_CATEGORIES:
        for event in registration["categories"][category]["events"]:
            path = image_root / event["resource"]
            if not path.is_file():
                missing_resources.append(event["resource"])
            else:
                image_bytes += path.stat().st_size
    reserve = estimate["checkpoint_bytes_with_25_percent_reserve"] + image_bytes + OUTPUT_LEDGER_RESERVE_BYTES
    probe = output_root.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    disk_free = shutil.disk_usage(probe).free
    host_free = _host_available_bytes()
    gpu = _gpu_status()
    reasons = []
    if missing_resources:
        reasons.append("BLINDED_RESOURCE_SET_INCOMPLETE")
    if disk_free < reserve:
        reasons.append("INSUFFICIENT_FREE_DISK")
    min_host = estimate["host_free_ram_minimum_bytes"]
    if host_free is None or host_free < min_host:
        reasons.append("HOST_RAM_PREFLIGHT_UNAVAILABLE_OR_INSUFFICIENT")
    if (not gpu["available"] or gpu["name"] != estimate["gpu_name_required"] or
            gpu["total_bytes"] is None or gpu["total_bytes"] < estimate["gpu_total_memory_minimum_bytes"]):
        reasons.append("REGISTERED_CUDA_DEVICE_UNAVAILABLE")
    if not isinstance(benchmark_receipt, Mapping):
        reasons.append("FULL_NATIVE_DIMENSION_RESOURCE_BENCHMARK_MISSING")
        benchmark = None
    else:
        benchmark = dict(benchmark_receipt)
        required = {"schema", "spec_sha256", "device_name", "native_rescore_seconds_p95",
                    "image_encode_seconds_p95", "host_peak_bytes", "gpu_peak_bytes", "synthetic_only",
                    "maximum_greedy_prototypes", "maximum_c1_prototypes", "maximum_e_cache_groups",
                    "max_checkpoint_non_cache_bytes"}
        if (set(benchmark) != required or benchmark.get("schema") != "FS4_SYNTHETIC_NATIVE_BENCHMARK_V1" or
                benchmark.get("synthetic_only") is not True or benchmark.get("spec_sha256") != load_spec()["spec_sha256"] or
                benchmark.get("device_name") != estimate["gpu_name_required"] or
                benchmark.get("maximum_greedy_prototypes") != 1024 or
                benchmark.get("maximum_c1_prototypes") != 1024 or
                benchmark.get("maximum_e_cache_groups") != 1024 or
                type(benchmark.get("max_checkpoint_non_cache_bytes")) is not int or
                benchmark["max_checkpoint_non_cache_bytes"] <= 0 or
                any(type(benchmark.get(k)) not in (int, float) or benchmark[k] <= 0
                    for k in ("native_rescore_seconds_p95", "image_encode_seconds_p95", "host_peak_bytes", "gpu_peak_bytes"))):
            reasons.append("FULL_NATIVE_DIMENSION_RESOURCE_BENCHMARK_INVALID")
        else:
            if benchmark["max_checkpoint_non_cache_bytes"] > CHECKPOINT_NATIVE_OVERHEAD_BYTES:
                reasons.append("SYNTHETIC_CHECKPOINT_OVERHEAD_EXCEEDS_FROZEN_ALLOWANCE")
            arm_rescore = estimate["max_exact_e_rescore_invocations_per_e_trajectory"] * float(benchmark["native_rescore_seconds_p95"])
            base_events = sum(EXPECTED_EVENTS.values()) * float(benchmark["image_encode_seconds_p95"])
            for label, projected in (("S", arm_rescore + base_events), ("CL", base_events), ("CE", arm_rescore + base_events)):
                if projected > estimate["per_trajectory_wall_limit_seconds"]:
                    reasons.append("TRAJECTORY_WALL_LIMIT_EXCEEDED:" + label)
            if 2 * arm_rescore + 3 * base_events > estimate["campaign_wall_limit_seconds"]:
                reasons.append("CAMPAIGN_WALL_LIMIT_EXCEEDED")
            if benchmark["host_peak_bytes"] > min_host or benchmark["gpu_peak_bytes"] > gpu.get("free_bytes", 0):
                reasons.append("SYNTHETIC_NATIVE_MEMORY_PEAK_EXCEEDS_AVAILABLE_MEMORY")
    return {"schema": "FS4_JOINT_CAMPAIGN_RESOURCE_PREFLIGHT_V1",
            "status": "PASS" if not reasons else "BLOCKED_RESOURCE_PREFLIGHT",
            "reasons": reasons, "spec_sha256": load_spec()["spec_sha256"],
            "registration_sha256": digest(registration), "image_resource_bytes": image_bytes,
            "required_free_disk_bytes": reserve, "observed_free_disk_bytes": disk_free,
            "observed_host_free_ram_bytes": host_free, "gpu": gpu,
            "estimate": estimate, "synthetic_benchmark_receipt": benchmark}


def write_resource_receipt(path: Path, report: Mapping[str, Any]) -> str:
    """Write one immutable, checksummed preflight receipt."""
    path = Path(path)
    data = json.dumps(dict(report), sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()
