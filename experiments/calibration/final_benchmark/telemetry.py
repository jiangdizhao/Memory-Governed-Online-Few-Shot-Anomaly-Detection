"""Small timing and byte-accounting ledger; no device probes are performed."""
from __future__ import annotations

import math
from typing import Any, Mapping


class TelemetryError(RuntimeError):
    pass


class TelemetryLedger:
    def __init__(self):
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.counters: dict[str, int] = {}
        self.memory: dict[str, dict[str, int]] = {}

    def record_duration(self, name: str, elapsed_seconds: float) -> None:
        value = float(elapsed_seconds)
        if not name or not math.isfinite(value) or value < 0:
            raise TelemetryError("timing record is invalid")
        self.seconds[name] = self.seconds.get(name, 0.0) + value
        self.calls[name] = self.calls.get(name, 0) + 1

    def increment(self, name: str, amount: int = 1) -> None:
        if not name or isinstance(amount, bool) or int(amount) != amount or amount < 0:
            raise TelemetryError("resource counter is invalid")
        self.counters[name] = self.counters.get(name, 0) + int(amount)

    def record_memory(self, name: str, *, physical_bytes: int,
                      logical_bytes: int) -> None:
        if (not name or min(int(physical_bytes), int(logical_bytes)) < 0 or
                isinstance(physical_bytes, bool) or isinstance(logical_bytes, bool)):
            raise TelemetryError("memory byte record is invalid")
        self.memory[name] = {"physical_bytes": int(physical_bytes),
                             "logical_bytes": int(logical_bytes)}

    def report(self) -> dict[str, Any]:
        return {
            "schema": "final-benchmark.resource-telemetry.v1",
            "timing_seconds": dict(sorted(self.seconds.items())),
            "timing_calls": dict(sorted(self.calls.items())),
            "counters": dict(sorted(self.counters.items())),
            "memory_components": dict(sorted(self.memory.items())),
            "gpu_peak_allocated_bytes": None,
            "gpu_peak_reserved_bytes": None,
            "host_peak_rss_bytes": None,
            "resource_status": "INTERFACE_PRESENT_DEVICE_AND_MEMORY_SAMPLING_PENDING_GATE2",
        }
