"""Read-only ledger verification subprocess for a research Stage B evaluator.

It replays the durable ledger's event/query state machine without constructing
an online trajectory or reading outcome labels. Output is a score-only projection.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import sys

import numpy as np

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
        execution = manifest["research_execution_manifest"]
        if execution.get("schema") != "FS4_JOINT_CAMPAIGN_EXECUTION_V1":
            # Preserve the pre-joint score-only projection contract for existing
            # research runs. The campaign's stricter receipts are campaign-only.
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
                    "stage_a_sha256": ledger.last_hash, "events": events,
                    "queried_feedback": queried_feedback}
        events = []
        queried_feedback = {}
        current_event = None
        current_prediction = None
        current_query = None
        current_transition = None
        category_first = {}
        category_last_thresholds = {}
        event_positions = set()
        for record in ledger.records:
            if record["kind"] == "CATEGORY":
                current_prediction = current_query = current_transition = None
                continue
            if record["kind"] == "PREDICTION":
                current_prediction = record["payload"]["event"]
                current_event = current_prediction["event"]
                current_query = current_transition = None
                continue
            if record["kind"] == "QUERY":
                current_query = record["payload"]["receipt"]
                continue
            if record["kind"] == "FEEDBACK":
                queried_feedback[current_event] = record["payload"]["label"]
                continue
            if record["kind"] == "TRANSITION":
                current_transition = record["payload"]["transition"]
                continue
            if record["kind"] != "EVENT_COMPLETE":
                continue
            value = current_prediction
            if not isinstance(value, dict) or not isinstance(current_query, dict) or not isinstance(current_transition, dict):
                raise ValueError("completed event lacks its sealed query or state transition")
            scores, decisions, thresholds = value["scores"], value["decisions"], value["thresholds"]
            if any(isinstance(scores.get(k), bool) or not isinstance(scores.get(k), (int, float)) or
                   not math.isfinite(scores[k]) for k in ("G", "GC")):
                raise ValueError("non-finite sealed score")
            provenance = value.get("trusted_event_provenance")
            category_name = provenance.get("category") if isinstance(provenance, dict) else None
            if category_name not in manifest["expected_events_by_category"] or value["position"] in event_positions:
                raise ValueError("event category or position is missing or duplicated")
            event_positions.add(value["position"])
            if not category_first.get(category_name):
                bootstrap = provenance.get("support_bootstrap")
                if not isinstance(bootstrap, dict):
                    raise ValueError("first category event lacks its four-Support bootstrap receipt")
                support = manifest["research_execution_manifest"]["categories"][category_name]["supports"]
                expected_groups = [item["group"] for item in support]
                expected_content = [item["content_sha256"] for item in support]
                support_scores = bootstrap.get("support_self_scores")
                if (len(support) != 4 or bootstrap.get("support_groups") != expected_groups or
                        bootstrap.get("support_content_sha256") != expected_content or
                        bootstrap.get("fixed_support_sha256") != manifest["fixed_support_sha256_by_category"][category_name] or
                        bootstrap.get("preflight_receipt_sha256") != manifest.get("support_preflight_receipt_sha256",
                            manifest["research_execution_manifest"]["preflight_receipt_sha256"]) or
                        bootstrap.get("additional_normal_images") != 0 or not isinstance(support_scores, list) or
                        len(support_scores) != 4 or any(not isinstance(x, (float, int)) or not math.isfinite(x) for x in support_scores)):
                    raise ValueError("four-Support bootstrap identity is inconsistent")
                values = np.asarray(support_scores, dtype=np.float64)
                mean = float(values.mean())
                sample_std = float(values.std(ddof=1))
                theta0 = mean + 2.5 * max(sample_std, 1e-6)
                if (bootstrap.get("mean") != mean or bootstrap.get("theta0") != theta0 or
                        bootstrap.get("sample_std_ddof1") != sample_std or
                        bootstrap.get("writer_sigma") != max(sample_std, 0.02)):
                    raise ValueError("four-Support cold-start formula differs")
                if manifest["research_execution_manifest"].get("schema") == "FS4_JOINT_CAMPAIGN_EXECUTION_V1":
                    registered = manifest["research_execution_manifest"]["support_preflight_reference_by_category"][category_name]
                    fields = {"support_groups": "support_groups", "support_content_sha256": "support_content_sha256",
                              "support_self_scores": "support_self_scores", "mean": "mean",
                              "sample_std_ddof1": "sample_std_ddof1", "theta0": "theta0",
                              "writer_sigma": "writer_sigma", "fixed_support_sha256": "fixed_support_sha256",
                              "preflight_receipt_sha256": "preflight_receipt_sha256",
                              "additional_normal_images": "additional_normal_images"}
                    if any(bootstrap.get(key) != registered.get(value) for key, value in fields.items()):
                        raise ValueError("event Support bootstrap differs from the campaign-bound actual receipt")
                category_first[category_name] = {"theta0": theta0, "support_bootstrap": bootstrap}
            theta0 = category_first[category_name]["theta0"]
            if current_query.get("prediction", {}).get("event") != value["event"] or current_query.get("prediction", {}).get("position") != value["position"]:
                raise ValueError("query receipt is not bound to its pre-event prediction")
            if manifest["profile"] == "SHARED_TRAJECTORY_THRESHOLD_ONLY":
                for name in ("L", "E", "FrozenTheta"):
                    for kind in ("G", "GC"):
                        if decisions[name][kind] != int(scores[kind] >= thresholds[name][kind]):
                            raise ValueError("sealed decision does not match score and threshold")
                expected_state = hashlib.sha256(json.dumps({"L": thresholds["L"], "E": thresholds["E"],
                    "FrozenTheta": thresholds["FrozenTheta"], "query_reference": value["query_reference_threshold"]},
                    sort_keys=True).encode()).hexdigest()
                if (value["query_reference_threshold"] != theta0 or
                        any(thresholds[name][j] != theta0 for name in ("L", "E", "FrozenTheta") for j in ("G", "GC"))
                        and value["position"] == min(e["position"] for e in manifest["research_execution_manifest"]["categories"][category_name]["events"])):
                    raise ValueError("shared cold-start threshold or query reference differs")
                if any(thresholds["FrozenTheta"][kind] != theta0 for kind in ("G", "GC")):
                    raise ValueError("FrozenTheta readout changed during the shared memory trajectory")
                if value["threshold_state"] != expected_state:
                    raise ValueError("shared pre-event threshold state digest differs")
                expected_next = {name: current_transition.get("next_thresholds", {}).get(name)
                                 for name in ("L", "E", "FrozenTheta")}
                if any(not isinstance(expected_next[name], dict) for name in expected_next):
                    raise ValueError("shared post-event threshold transition is missing")
                if manifest["research_execution_manifest"].get("schema") == "FS4_JOINT_CAMPAIGN_EXECUTION_V1":
                    audit = current_transition.get("audit", {})
                    controllers = audit.get("controllers", {}) if isinstance(audit, dict) else {}
                    if (any(name not in controllers or expected_next[name] != controllers[name].get("thresholds")
                            for name in ("L", "E")) or
                            expected_next["FrozenTheta"] != {kind: theta0 for kind in ("G", "GC")}):
                        raise ValueError("shared post-feedback thresholds differ from the sealed controller audit")
                decisions = {f"{name}_{kind}": decisions[name][kind]
                             for name in ("L", "E", "FrozenTheta") for kind in ("G", "GC")}
                thresholds_flat = {f"{name}_{kind}": thresholds[name][kind]
                                   for name in ("L", "E", "FrozenTheta") for kind in ("G", "GC")}
                if (value["position"] > min(e["position"] for e in manifest["research_execution_manifest"]["categories"][category_name]["events"]) and
                        category_last_thresholds[category_name] != thresholds):
                    raise ValueError("shared pre-event threshold does not match the previous transition")
                category_last_thresholds[category_name] = expected_next
            else:
                for kind in ("G", "GC"):
                    if decisions[kind] != int(scores[kind] >= thresholds[kind]):
                        raise ValueError("sealed decision does not match score and threshold")
                decisions = {f"{manifest['controller']}_{kind}": decisions[kind] for kind in ("G", "GC")}
                expected_state = hashlib.sha256(json.dumps(thresholds, sort_keys=True).encode()).hexdigest()
                if value["threshold_state"] != expected_state or value["query_reference_threshold"] != thresholds["GC"]:
                    raise ValueError("coupled pre-event threshold/query state differs")
                if value["position"] == min(e["position"] for e in manifest["research_execution_manifest"]["categories"][category_name]["events"]):
                    if thresholds["G"] != theta0 or thresholds["GC"] != theta0:
                        raise ValueError("coupled cold-start threshold differs from four-Support theta0")
                elif category_last_thresholds[category_name] != thresholds:
                    raise ValueError("coupled pre-event threshold does not match the previous transition")
                next_active = current_transition.get("next_thresholds")
                if not isinstance(next_active, dict) or set(next_active) != {"G", "GC"}:
                    raise ValueError("coupled post-event threshold transition is missing")
                if manifest["research_execution_manifest"].get("schema") == "FS4_JOINT_CAMPAIGN_EXECUTION_V1":
                    audit = current_transition.get("audit", {})
                    controllers = audit.get("controllers", {}) if isinstance(audit, dict) else {}
                    if (manifest["controller"] not in controllers or
                            next_active != controllers[manifest["controller"]].get("thresholds")):
                        raise ValueError("coupled post-feedback threshold differs from its sealed audit")
                category_last_thresholds[category_name] = next_active
                thresholds_flat = {f"{manifest['controller']}_{kind}": thresholds[kind] for kind in ("G", "GC")}
            if value["position"] < 0 or not isinstance(value.get("group"), str):
                raise ValueError("sealed event identity is invalid")
            view = provenance.get("view")
            events.append({"event": value["event"], "group": value["group"], "position": value["position"],
                           "view": view, "category": category_name, "scores": scores,
                           "decisions": decisions, "thresholds": thresholds_flat,
                           "query": current_query, "transition": current_transition})
            current_prediction = current_query = current_transition = None
        if len(events) != sum(manifest["expected_events_by_category"].values()):
            raise ValueError("sealed event count differs from campaign registration")
        return {"schema": "FS4_RESEARCH_SEALED_PROJECTION_V1", "run_id": manifest["run_id"],
                "manifest_sha256": digest(manifest), "source_commit": manifest["source_commit"],
                "campaign_id": manifest.get("campaign_id"), "trajectory_id": manifest.get("trajectory_id"),
                "profile": manifest["profile"], "controller": manifest.get("controller"),
                "campaign_registration_sha256": manifest.get("campaign_registration_sha256"),
                "r7_seed0_order_sha256": manifest.get("r7_seed0_order_sha256"),
                "source_mapping_sha256": manifest["research_execution_manifest"]["source_mapping_sha256"],
                "scientific_spec_sha256": manifest.get("scientific_spec_sha256"),
                "support_preflight_receipt_sha256": manifest.get("support_preflight_receipt_sha256"),
                "model_state_sha256": manifest["model_state_sha256"],
                "checkpoint_sha256": manifest["research_execution_manifest"]["checkpoint_sha256"],
                "hub_source_tree_sha256": manifest["research_execution_manifest"]["hub_source_tree_sha256"],
                "scorer_config_sha256": manifest["research_execution_manifest"]["scorer_config_sha256"],
                "preflight_receipt_sha256": manifest["research_execution_manifest"]["preflight_receipt_sha256"],
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
