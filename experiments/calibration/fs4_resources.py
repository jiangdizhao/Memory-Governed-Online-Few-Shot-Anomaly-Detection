"""Explicit FS4 resource preflight and bounded process supervisor."""
from __future__ import annotations

import math
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Any, Callable, Sequence

import numpy as np
import psutil
import torch

BYTES_PER_NORMAL = 784 * 4096 * 4
FULL_CACHE_BYTES = 1024 * BYTES_PER_NORMAL


def preflight(path: Path, *, cache_groups: int, required_disk_bytes: int,
              other_host_bytes: int = 0) -> dict[str, int]:
    if not 0 <= cache_groups <= 1024 or min(required_disk_bytes, other_host_bytes) < 0:
        raise ValueError("invalid resource budget")
    host_free = psutil.virtual_memory().available
    disk_free = shutil.disk_usage(path).free
    cache_bytes = cache_groups * BYTES_PER_NORMAL
    if host_free < cache_bytes + other_host_bytes:
        raise MemoryError("insufficient available host RAM for declared budget")
    if disk_free < required_disk_bytes:
        raise OSError("insufficient free disk for declared budget")
    return {"host_available_bytes": host_free, "disk_free_bytes": disk_free,
            "cache_groups": cache_groups, "cache_tensor_bytes": cache_bytes,
            "declared_other_host_bytes": other_host_bytes,
            "required_disk_bytes": required_disk_bytes, "full_cache_tensor_bytes": FULL_CACHE_BYTES}


def rescore_projection(samples: Sequence[float], *, groups: int, future_refreshes: int) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("finite timing observations required")
    if not 0 <= groups <= 1024 or future_refreshes < 0:
        raise ValueError("invalid projection dimensions")
    mean = float(values.mean())
    return {"samples": len(values), "mean_seconds": mean,
            "median_seconds": float(np.median(values)), "p95_seconds": float(np.quantile(values, .95)),
            "projected_rescore_calls": groups * future_refreshes,
            "projected_seconds_at_observed_mean": mean * groups * future_refreshes,
            "full_cache_one_refresh_seconds_at_observed_mean": mean * 1024}


def gpu_peak_bytes() -> int | None:
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None


def _child(conn: Any, func: Callable[..., Any], args: tuple[Any, ...]) -> None:
    try:
        conn.send((True, func(*args)))
    except BaseException as exc:
        conn.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def run_with_timeout(func: Callable[..., Any], args: tuple[Any, ...], seconds: float) -> Any:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("positive finite timeout required")
    context = mp.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(writer, func, args))
    process.start(); writer.close()
    try:
        if not reader.poll(seconds):
            process.terminate(); process.join()
            raise TimeoutError("FS4 condition exceeded wall-time limit")
        ok, result = reader.recv()
        process.join()
        if not ok or process.exitcode != 0:
            raise RuntimeError(f"FS4 condition failed: {result}")
        return result
    finally:
        reader.close()
        if process.is_alive():
            process.terminate(); process.join()
