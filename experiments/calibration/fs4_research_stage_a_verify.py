"""Read-only ledger verification subprocess for a research Stage B evaluator.

It replays the durable ledger's event/query state machine without constructing
an online trajectory or reading outcome labels. Output is a score-only projection.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

from .fs4_durable_ledger import canonical, digest
from .fs4_r2_security import WindowsPrivateStore
from .fs4_research_ledger import ResearchLedger


def verify_stage_a(root: Path) -> dict:
    store = WindowsPrivateStore(root, create=False)
    manifest = json.loads(store.read("manifest.json"))
    ledger = ResearchLedger(root, manifest, create=False)
    try:
        if ledger.phase != "GLOBAL_A" or set(ledger.category_seals) != set(manifest["expected_events_by_category"]):
            raise ValueError("all category and global Stage A seals are required")
        events = []
        queried_feedback = {}
        current_event = None
        for record in ledger.records:
            if record["kind"] == "FEEDBACK":
                queried_feedback[current_event] = record["payload"]["label"]
                continue
            if record["kind"] != "PREDICTION":
                continue
            value = record["payload"]["event"]
            current_event = value["event"]
            scores, decisions, thresholds = value["scores"], value["decisions"], value["thresholds"]
            if any(isinstance(scores.get(k), bool) or not isinstance(scores.get(k), (int, float)) or
                   not math.isfinite(scores[k]) for k in ("G", "GC")):
                raise ValueError("non-finite sealed score")
            if manifest["profile"] == "SHARED_TRAJECTORY_THRESHOLD_ONLY":
                for name in ("L", "E"):
                    for kind in ("G", "GC"):
                        if decisions[name][kind] != int(scores[kind] >= thresholds[name][kind]):
                            raise ValueError("sealed decision does not match score and threshold")
                decisions = {f"{name}_{kind}": decisions[name][kind] for name in ("L", "E") for kind in ("G", "GC")}
            else:
                for kind in ("G", "GC"):
                    if decisions[kind] != int(scores[kind] >= thresholds[kind]):
                        raise ValueError("sealed decision does not match score and threshold")
                decisions = {f"{manifest['controller']}_{kind}": decisions[kind] for kind in ("G", "GC")}
            thresholds_flat = ({f"{name}_{kind}": thresholds[name][kind] for name in ("L", "E") for kind in ("G", "GC")}
                if manifest["profile"] == "SHARED_TRAJECTORY_THRESHOLD_ONLY" else
                {f"{manifest['controller']}_{kind}": thresholds[kind] for kind in ("G", "GC")})
            events.append({"event": value["event"], "position": value["position"],
                           "category": value["trusted_event_provenance"]["category"],
                           "scores": scores, "thresholds": thresholds_flat})
        return {"schema": "FS4_RESEARCH_SEALED_PROJECTION_V1", "run_id": manifest["run_id"],
                "manifest_sha256": digest(manifest), "source_commit": manifest["source_commit"],
                "model_state_sha256": manifest["model_state_sha256"],
                "dataset_identity_sha256": manifest["dataset_identity_sha256"],
                "fixed_support_sha256_by_category": manifest["fixed_support_sha256_by_category"],
                "stream_sha256_by_category": manifest["stream_sha256_by_category"],
                "stage_a_sha256": ledger.last_hash, "events": events, "queried_feedback": queried_feedback}
    finally:
        ledger.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("expected one private run directory")
    sys.stdout.buffer.write(canonical(verify_stage_a(Path(sys.argv[1]))))
