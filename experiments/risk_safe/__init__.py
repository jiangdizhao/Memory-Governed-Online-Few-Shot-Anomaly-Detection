"""Risk-Safe Memory Admission research package.

D1-A scope only: metadata-only dataset and query-budget preflight.  Nothing in
this package extracts features, builds a feature bank, loads DINOv2, allocates
CUDA tensors or evaluates anomaly-detection performance.

Modules
-------
``query_protocol``
    Q-A shared label-blind random query tape, nested absolute budgets, budget
    accounting and the causal query/feedback replay harness.
``preflight``
    Real-IAD metadata preflight, physical-object partitioning audits, the
    historical V0.2 reference check, the Sidecar activation feasibility audit
    and the deterministic JSON manifest writers.

Historically validated implementations (``vmb_visionad_new``,
``final_decision_core``, ``v02_architecture_core``, ``v02_memory``,
``test_phase3_AIF_simplified_v22``) are imported, never modified.
"""
from __future__ import annotations

SCHEMA_PREFIX = "risk_safe.d1a"
STAGE = "D1-A"

__all__ = ["SCHEMA_PREFIX", "STAGE", "preflight", "query_protocol"]
