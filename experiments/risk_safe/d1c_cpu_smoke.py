#!/usr/bin/env python3
"""D1-C controlled CPU-only counterfactual-mechanism experiment.

Scenarios are synthetic, *not Real-IAD results*, not native v22 image-scoring
parity and not evidence of deployment FNR guarantees. Data/model inputs are
fixed and generated here without accessing any test-set labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch
from . import preflight as pf
from . import query_protocol as qp
from .anchor_store import AnchorLedger, VerifiedImage
from .shadow_memory import CosinePatchScorer, GreedyBank
from .admission import CDARG, CDARGConfig, heldout_counterfactual_erosion

ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT = "outputs/risk_safe_sprint_20261027/d1c_cpu_smoke_v1"
SCHEMA = "risk_safe.d1c.synthetic_cpu_mechanism.v1"
SOURCE_FILES = (
    "experiments/risk_safe/shadow_memory.py",
    "experiments/risk_safe/anchor_store.py",
    "experiments/risk_safe/admission.py",
    "experiments/risk_safe/d1c_cpu_smoke.py",
    "tests/test_risk_safe_d1c.py",
    "experiments/v02_memory.py",
    "docs/risk_safe/D1_C_CDARG_DESIGN_GUARDRAILS_2026-10-28.md",
    "docs/risk_safe/D1_C_CPU_IMPLEMENTATION.md",
)
OUTPUTS = ("synthetic_mechanism_results.json", "source_manifest.json")


def row(x: float, y: float) -> torch.Tensor:
    return torch.tensor([[x, y]], dtype=torch.float32, device="cpu")


def label_event(t: int, y: str) -> VerifiedImage:
    return VerifiedImage(query_id=f"synthetic-query-{t}",
                         image_id=f"synthetic-image-{t}",
                         sample_key=f"synthetic-sample-{t}", camera_id=1,
                         label=y, query_decision_step=t,
                         label_revealed_step=t)


def run_case(name: str, *,
             normal: torch.Tensor, defect: torch.Tensor | None,
             candidate: torch.Tensor,
             static: torch.Tensor, dynamic: torch.Tensor,
             theta: float, capacity: int,
             probe: torch.Tensor) -> dict:
    config = CDARGConfig(epsilon_defect=0.02, lambda_defect=1.0,
                         tau_utility=0.0, defect_margin=0.05,
                         sigma_normal=1.0, sigma_floor=0.01)
    ledger = AnchorLedger(max_normal_images=64, max_defect_images=32,
                          patches_per_image=8)
    ledger.record(label_event(1, "normal"), [normal])
    if defect is not None:
        ledger.record(label_event(2, "defect"), [defect],
                      anomaly_patch_scores=[torch.tensor([1.0])])
    bank = GreedyBank([dynamic], [capacity])
    raw = CosinePatchScorer([static])
    before = bank.sha256()
    decision = CDARG(config, raw).evaluate(
        current_verified_normal=label_event(3, "normal"),
        step=3, bank=bank, ledger=ledger,
        candidate_patches=[candidate], theta=theta)

    # Held-out probe is only inspected AFTER the admission decision. The probe
    # never enters the gate, query policy or anchor ledger.
    heldout = heldout_counterfactual_erosion(
        [[probe]], bank, decision.shadow, raw)
    changed = CDARG.commit_if_accepted(decision, bank)
    if changed != (decision.decision == "admit"):
        raise AssertionError("Decision/live-memory mismatch")
    if not changed and bank.sha256() != before:
        raise AssertionError("Rejected shadow changed live memory")
    if name == "harmful_defect_crossing" and decision.reason != "protected_defect_margin_crossing":
        raise AssertionError("Dangerous verified-normal write was not rejected")
    if name == "benign_normal_adaptation" and decision.decision != "admit":
        raise AssertionError("Benign normal candidate was not admitted")
    if name == "no_defect_anchor" and decision.reason != "defect_risk_unavailable":
        raise AssertionError("No-anchor risk was incorrectly treated as zero")
    if name == "full_capacity_harmful_replacement" and (
            decision.greedy_audit.replaced == 0 or decision.decision != "reject"):
        raise AssertionError("Replacement was not safely simulated")
    return {"scenario": name, "gate": decision.audit(),
            "heldout_evaluation_only": heldout,
            "bank_after_sha256": bank.sha256(),
            "bank_after_version": bank.version,
            "bank_after_prototypes": len(bank.layers[0]),
            "normal_anchor_count": ledger.counts["verified_normal_anchor_images"],
            "defect_anchor_count": ledger.counts["verified_defect_anchor_images"]}


def synthetic_cases() -> list[dict]:
    fixed = row(1, 0)
    return [
        run_case("harmful_defect_crossing",
                 normal=row(-1, 0), defect=row(-0.96, 0.28),
                 candidate=row(-1, 0), static=fixed, dynamic=row(1, 0),
                 theta=0.5, capacity=3, probe=row(-0.92, 0.39)),
        run_case("benign_normal_adaptation",
                 normal=row(0, 1), defect=row(0.8, -0.6),
                 candidate=row(0, 1), static=fixed, dynamic=row(1, 0),
                 theta=0.1, capacity=3, probe=row(0.75, -0.66)),
        run_case("no_defect_anchor",
                 normal=row(-1, 0), defect=None,
                 candidate=row(-1, 0), static=fixed, dynamic=row(1, 0),
                 theta=0.5, capacity=3, probe=row(-0.92, 0.39)),
        run_case("full_capacity_harmful_replacement",
                 normal=row(-1, 0), defect=row(-0.96, 0.28),
                 candidate=row(-1, 0), static=fixed,
                 dynamic=torch.tensor([[1., 0.], [0.99, 0.1]]),
                 theta=0.5, capacity=2, probe=row(-0.92, 0.39)),
    ]


def build_payloads() -> dict:
    rows = synthetic_cases()
    config = {
        "epsilon_defect": 0.02, "lambda_defect": 1.0, "tau_utility": 0.0,
        "defect_margin": 0.05, "sigma_normal": 1.0, "sigma_floor": 0.01,
        "calibration_status": "DEMONSTRATION_ONLY_NOT_CALIBRATED",
        "scorer": "exact nearest cosine patch (synthetic; not v22 image score)",
        "shadow_update": "exact v02_memory.greedy_slot for full capacity",
        "sidecar": "off", "signed_correction": "off", "threshold": "fixed per case",
        "dataset": "synthetic only; no Real-IAD decoded",
    }
    payload = {
        "schema": SCHEMA, "stage": "D1-C",
        "kind": "CPU_CAUSAL_MECHANISM_SANITY_NOT_PERFORMANCE",
        "configuration": config,
        "scenarios": rows,
        "pass": all(
            (x["scenario"] == "benign_normal_adaptation" and
             x["gate"]["decision"] == "admit") or
            (x["scenario"] != "benign_normal_adaptation" and
             x["gate"]["decision"] == "reject") for x in rows),
        "N_cases": len(rows),
        "CUDA_initialised": False,
        "native_v22_image_scorer_parity_tested": False,
    }
    return {OUTPUTS[0]: payload}


def source_manifest(payloads: dict) -> dict:
    return {
        "schema": "risk_safe.d1c.source_manifest.v1",
        "producing_git": pf.git_state(),
        "source_sha256": pf.source_hashes(SOURCE_FILES),
        "output_sha256": {n: pf.sha256_text(pf.json_text(v))
                          for n, v in payloads.items()},
        "environment": pf.environment_block(),
        "status": "D1-C CPU-only synthetic smoke, not GPU/probe results",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=OUT_DEFAULT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    payloads = build_payloads()
    payloads[OUTPUTS[1]] = pf.preserve_committed_source_provenance(
        out_dir / OUTPUTS[1], source_manifest(payloads))
    status = {}
    for name in OUTPUTS:
        status[name] = (pf.verify_json_output(out_dir / name, payloads[name])
                        if args.verify_only
                        else pf.write_json_idempotent(out_dir / name, payloads[name]))
    print(json.dumps({"status": status, "cases": payloads[OUTPUTS[0]]["N_cases"],
                      "synthetic_mechanism_pass": payloads[OUTPUTS[0]]["pass"],
                      "native_v22_scoring_tested": False,
                      "gpu_used": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
