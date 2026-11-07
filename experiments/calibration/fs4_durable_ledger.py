"""Append-only FS4 write-ahead ledger. Importing this module starts no run.

Every line is a canonical, hash-chained JSON record, flushed and fsynced before
its method returns. An interrupted or malformed tail fails closed. The label
intent record makes an unjournaled disclosure ambiguity unrecoverable.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

import numpy as np

from .fs4_stochastic_bea import (QueryReceipt, action_uniform, bea_components, candidate_ordinal,
                                 receipt_digest, TARGET_RATE)
from .fs4_threshold_integration import prediction_digest
from .fs4_hardening_audit import SOURCES, PROTECTED_SOURCES

REQUIRED_SOURCE_PATHS = frozenset((*SOURCES, *PROTECTED_SOURCES,
    "experiments/calibration/fs4_threshold_core.py",
    "experiments/calibration/fs4_threshold_audit.py",
    "docs/calibration/fs4_threshold/SCIENTIFIC_CONTRACT.md"))


class LedgerIntegrityError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


class DurableLedger:
    """One exclusive run namespace with explicitly registered category counts."""

    def __init__(self, root: Path, manifest: Mapping[str, Any], *, create: bool) -> None:
        self.root = Path(root)
        self.path = self.root / "events.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self.manifest = dict(manifest)
        required = ("run_id", "source_commit", "source_fingerprints", "expected_events_by_category", "profile",
                    "stream_sha256_by_category", "scientific_contract_sha256", "dataset_identity_sha256",
                    "model_state_sha256", "support_groups_by_category", "fixed_support_sha256_by_category")
        if any(not self.manifest.get(k) for k in required):
            raise LedgerIntegrityError("incomplete run manifest")
        if not isinstance(self.manifest.get("query_seed"), int) or isinstance(self.manifest["query_seed"], bool):
            raise LedgerIntegrityError("query seed must be registered in the run manifest")
        if not REQUIRED_SOURCE_PATHS.issubset(self.manifest["source_fingerprints"]):
            raise LedgerIntegrityError("source fingerprint set omits required native or FS4 dependencies")
        if self.manifest["scientific_contract_sha256"] != self.manifest["source_fingerprints"]["docs/calibration/fs4_threshold/SCIENTIFIC_CONTRACT.md"]:
            raise LedgerIntegrityError("scientific contract fingerprint mismatch")
        import re
        is_sha = lambda s: isinstance(s, str) and re.fullmatch(r"[0-9a-f]{64}", s) is not None
        cats = set(self.manifest["expected_events_by_category"])
        if (not all(is_sha(self.manifest[k]) for k in ("scientific_contract_sha256", "dataset_identity_sha256", "model_state_sha256")) or
                any(set(self.manifest[k]) != cats for k in ("stream_sha256_by_category", "support_groups_by_category", "fixed_support_sha256_by_category")) or
                any(not is_sha(x) for x in self.manifest["stream_sha256_by_category"].values()) or
                any(not is_sha(x) for x in self.manifest["fixed_support_sha256_by_category"].values()) or
                any(len(groups) != 4 or len(set(groups)) != 4 or not all(is_sha(x) for x in groups)
                    for groups in self.manifest["support_groups_by_category"].values())):
            raise LedgerIntegrityError("incomplete registered stream, Support, dataset or model identity")
        source_root = Path(__file__).resolve().parents[2]
        source_commit = str(self.manifest["source_commit"])
        if subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
                          cwd=source_root, capture_output=True).returncode:
            raise LedgerIntegrityError("source commit is not an ancestor")
        for name, expected in self.manifest["source_fingerprints"].items():
            path = (source_root / name).resolve()
            if not path.is_relative_to(source_root) or not path.is_file():
                raise LedgerIntegrityError("invalid source fingerprint path")
            current = path.read_bytes()
            committed = subprocess.run(["git", "show", f"{source_commit}:{name}"], cwd=source_root,
                                       capture_output=True)
            if (hashlib.sha256(current).hexdigest() != expected or committed.returncode != 0 or
                    committed.stdout.replace(b"\r\n", b"\n") != current.replace(b"\r\n", b"\n")):
                raise LedgerIntegrityError("source fingerprint mismatch")
        if self.manifest["profile"] not in ("SHARED_TRAJECTORY_THRESHOLD_ONLY", "COUPLED_QUERY_ONLY_FROZEN_WRITER"):
            raise LedgerIntegrityError("unregistered trajectory profile")
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in self.manifest["expected_events_by_category"].values()):
            raise LedgerIntegrityError("invalid registered category counts")
        if create:
            self.root.mkdir(parents=False, exist_ok=False)
            with self.manifest_path.open("xb") as out:
                out.write(canonical(self.manifest) + b"\n")
                out.flush(); os.fsync(out.fileno())
            with self.path.open("xb") as out:
                out.flush(); os.fsync(out.fileno())
        elif self.manifest_path.read_bytes() != canonical(self.manifest) + b"\n":
            raise LedgerIntegrityError("manifest identity mismatch")
        self._lock = (self.root / "writer.lock").open("a+b")
        self._lock.seek(0, os.SEEK_END)
        if self._lock.tell() == 0:
            self._lock.write(b"0"); self._lock.flush(); os.fsync(self._lock.fileno())
        self._lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lock.close()
            raise LedgerIntegrityError("another ledger writer holds the run lock") from exc
        self.records: list[dict[str, Any]] = []
        self.poisoned = False
        self.last_hash = digest(self.manifest)
        self.phase = "READY"
        self.position = 0
        self.event_ids: set[str] = set()
        self.group_counts: dict[tuple[str, str], int] = {}
        self.category_counts = {c: 0 for c in self.manifest["expected_events_by_category"]}
        self.category_streams: dict[str, list[dict[str, str]]] = {c: [] for c in self.category_counts}
        self.category_seals: dict[str, str] = {}
        self.current_category: str | None = None
        self.query_price = .8
        self.query_history: list[float] = []
        self.query_flags: list[int] = []
        self.prediction: dict[str, Any] | None = None
        self.query: dict[str, Any] | None = None
        self._read_and_verify()

    def _read_and_verify(self) -> None:
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            raise LedgerIntegrityError("partial ledger record")
        for line in data.splitlines():
            try:
                row = json.loads(line)
                if canonical(row) != line:
                    raise ValueError("noncanonical record")
                body = {k: row[k] for k in ("index", "previous", "kind", "payload")}
                if row["index"] != len(self.records) or row["previous"] != self.last_hash or row["sha256"] != digest(body):
                    raise ValueError("hash chain mismatch")
                self._apply(row["kind"], row["payload"])
                self.records.append(row); self.last_hash = row["sha256"]
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise LedgerIntegrityError("invalid ledger sequence or checksum") from exc

    def _append(self, kind: str, payload: dict[str, Any]) -> str:
        if self.poisoned or self._lock.closed:
            raise LedgerIntegrityError("ledger writer is closed or failed")
        # Validate before the irreversible append, then apply again after fsync.
        import copy
        probe = copy.copy(self)
        probe.event_ids = set(self.event_ids)
        probe.group_counts = dict(self.group_counts)
        probe.query_history = list(self.query_history)
        probe.query_flags = list(self.query_flags)
        probe.category_counts = dict(self.category_counts)
        probe.category_streams = {key: list(value) for key, value in self.category_streams.items()}
        probe.category_seals = dict(self.category_seals)
        probe._apply(kind, payload)
        body = {"index": len(self.records), "previous": self.last_hash, "kind": kind, "payload": payload}
        row = {**body, "sha256": digest(body)}
        try:
            data = canonical(row) + b"\n"
            with self.path.open("ab", buffering=0) as out:
                if out.write(data) != len(data):
                    raise OSError("short ledger write")
                out.flush(); os.fsync(out.fileno())
        except Exception:
            self.poisoned = True
            raise
        self._apply(kind, payload)
        self.records.append(row); self.last_hash = row["sha256"]
        return row["sha256"]

    def _apply(self, kind: str, p: dict[str, Any]) -> None:
        if kind == "CATEGORY":
            if self.phase != "READY" or self.current_category is not None or p["category"] not in self.category_counts or p["category"] in self.category_seals:
                raise LedgerIntegrityError("invalid category transition")
            self.current_category = p["category"]
            self.query_price = .8; self.query_history = []; self.query_flags = []
        elif kind == "PREDICTION":
            x = p["event"]
            if (self.phase != "READY" or self.current_category is None or x["position"] != self.position or
                    x["event"] in self.event_ids or x["profile"] != self.manifest["profile"] or
                    p["prediction_sha256"] != prediction_digest(x) or
                    not all(x.get(k) for k in ("group", "event", "states", "threshold_state", "thresholds", "query_inputs")) or
                    self.category_counts[self.current_category] >= self.manifest["expected_events_by_category"][self.current_category]):
                raise LedgerIntegrityError("invalid or duplicate prediction")
            self.prediction = x; self.event_ids.add(x["event"]); self.phase = "PREDICTION"
            self.category_streams[self.current_category].append({"group": x["group"], "event": x["event"]})
        elif kind == "QUERY":
            q = p["receipt"]
            pred = q["prediction"]
            key = (self.current_category, pred["group"])
            expected_ordinal = self.group_counts.get(key, 0) + 1
            if (self.phase != "PREDICTION" or self.prediction is None or
                    pred["position"] != self.position or pred["event"] != self.prediction["event"] or
                    pred["group"] != self.prediction["group"] or pred["seal"] != prediction_digest(self.prediction) or
                    pred["scorer_g"] != self.prediction["states"]["G"] or pred["scorer_gc"] != self.prediction["states"]["GC"] or
                    pred["threshold_state"] != self.prediction["threshold_state"] or
                    pred["score_g"] != self.prediction["scores"]["G"] or pred["score_gc"] != self.prediction["scores"]["GC"] or
                    q["query_seal"] is not None or q["disclosed"] or q["label"] is not None or
                    p["query_sha256"] != digest(q) or q["candidate"] != candidate_ordinal(q["seed"], pred["group"]) or
                    q["slot_probability"] != 0.2 or q["exploration_probability"] != 0.05 or
                    q["ordinal"] != expected_ordinal or expected_ordinal > 5 or
                    q["policy"] != "STOCHASTIC_BEA_GROUP_SLOT_V1" or q["seed"] != self.manifest["query_seed"]):
                raise LedgerIntegrityError("invalid query receipt")
            inputs = self.prediction["query_inputs"]
            prior = np.asarray(self.query_history, dtype=np.float64)
            scale = (max(1.4826 * float(np.median(np.abs(prior - np.median(prior)))), .02)
                     if len(prior) >= 2 else .02)
            components = bea_components(score=pred["score_gc"], threshold=self.prediction["query_reference_threshold"],
                scale=scale, novelty_percentile=inputs["novelty_percentile"],
                correction_top32=inputs["correction_top32"], correction_bank_size=inputs["correction_bank_size"],
                topk_dN_max=inputs["topk_dN_max"], topk_dN_mean=inputs["topk_dN_mean"])
            if (inputs["pre_score_scale"] != scale or q["query_scale"] != scale or
                    q["reference_threshold"] != self.prediction["query_reference_threshold"] or
                    q["utility"] != components["utility"] or q["cost"] != self.query_price):
                raise LedgerIntegrityError("query utility, scale, price or reference differs from sealed policy inputs")
            eligible = q["ordinal"] == q["candidate"]
            gate = (1.0 if q["utility"] > q["cost"] else 0.05) if eligible else None
            draw = action_uniform(q["seed"], pred["group"], pred["event"]) if eligible else None
            if (q["exploitation"] != (q["utility"] > q["cost"]) or q["gate_probability"] != gate or
                    q["random_draw"] != draw or q["design_probability"] != (0.2 * gate if gate is not None else None) or
                    q["executed"] != bool(eligible and draw < gate)):
                raise LedgerIntegrityError("query action/probability mismatch")
            self.group_counts[key] = expected_ordinal
            self.query_flags = (self.query_flags + [int(q["executed"])])[-256:]
            self.query_price = min(3.0, max(.05, self.query_price + .02 *
                (sum(self.query_flags) / len(self.query_flags) - TARGET_RATE)))
            self.query_history = (self.query_history + [pred["score_gc"]])[-256:]
            self.query = q; self.phase = "QUERY"
        elif kind == "LABEL_INTENT":
            if self.phase != "QUERY" or self.query is None or not self.query["executed"] or p["position"] != self.position:
                raise LedgerIntegrityError("unsealed or unexecuted label access")
            self.phase = "LABEL_INTENT"
        elif kind == "FEEDBACK":
            if self.phase != "LABEL_INTENT" or p["position"] != self.position or p["label"] not in (0, 1) or isinstance(p["label"], bool):
                raise LedgerIntegrityError("invalid or duplicate feedback")
            self.phase = "FEEDBACK"
        elif kind == "TRANSITION":
            if (self.query is None or self.phase != ("FEEDBACK" if self.query["executed"] else "QUERY") or
                    p["position"] != self.position or not p.get("post_state_digest")):
                raise LedgerIntegrityError("invalid state transition")
            self.phase = "TRANSITION"
        elif kind == "EVENT_COMPLETE":
            if self.phase != "TRANSITION" or p["position"] != self.position:
                raise LedgerIntegrityError("invalid event completion")
            self.category_counts[self.current_category] += 1
            self.position += 1; self.phase = "READY"; self.prediction = None; self.query = None
        elif kind == "CATEGORY_SEAL":
            c = p["category"]
            if self.phase != "READY" or c != self.current_category or c in self.category_seals or self.category_counts[c] != self.manifest["expected_events_by_category"][c] or any(n != 5 for (category, _), n in self.group_counts.items() if category == c) or digest(self.category_streams[c]) != self.manifest["stream_sha256_by_category"][c] or p["stream_sha256"] != self.last_hash:
                raise LedgerIntegrityError("invalid category Stage A seal")
            self.category_seals[c] = p["stream_sha256"]; self.current_category = None
        elif kind == "GLOBAL_A":
            if self.phase != "READY" or set(self.category_seals) != set(self.category_counts) or p["category_seals"] != self.category_seals or p["stream_sha256"] != self.last_hash:
                raise LedgerIntegrityError("invalid global Stage A seal")
            self.phase = "GLOBAL_A"
        elif kind == "STAGE_B_AUTH":
            if (self.phase != "GLOBAL_A" or not p.get("authorization_id") or
                    p.get("run_id") != self.manifest["run_id"] or
                    p.get("manifest_sha256") != digest(self.manifest) or
                    p.get("stage_a_sha256") != self.last_hash or p.get("scope") != "FS4_STAGE_B"):
                raise LedgerIntegrityError("Stage B requires a separate explicit authorization")
            self.phase = "STAGE_B_AUTH"
        elif kind == "COMPLETE":
            if self.phase != "STAGE_B_AUTH" or p["stream_sha256"] != self.last_hash:
                raise LedgerIntegrityError("invalid complete seal")
            self.phase = "COMPLETE"
        else:
            raise LedgerIntegrityError("unknown record kind")

    def select_category(self, category: str) -> None:
        self._append("CATEGORY", {"category": category})

    def seal_prediction(self, event: dict[str, Any]) -> str:
        value = prediction_digest(event)
        self._append("PREDICTION", {"event": event, "prediction_sha256": value})
        return value

    def seal_query(self, receipt: QueryReceipt) -> str:
        value = receipt_digest(receipt)
        self._append("QUERY", {"receipt": asdict(receipt), "query_sha256": value})
        return value

    def disclose(self, event: str, position: int, reader: Callable[[str, int], int]) -> int:
        if self.prediction is None or self.prediction["event"] != event or position != self.position:
            raise LedgerIntegrityError("label identity mismatch")
        self._append("LABEL_INTENT", {"position": position})
        label = reader(event, position)
        self._append("FEEDBACK", {"position": position, "label": label})
        return label

    def complete_event(self, *, position: int, post_state_digest: str, transition: dict[str, Any]) -> None:
        self._append("TRANSITION", {"position": position, "post_state_digest": post_state_digest, "transition": transition})
        self._append("EVENT_COMPLETE", {"position": position})

    def seal_category(self, category: str) -> str:
        return self._append("CATEGORY_SEAL", {"category": category, "stream_sha256": self.last_hash})

    def seal_global_stage_a(self) -> str:
        return self._append("GLOBAL_A", {"category_seals": dict(self.category_seals), "stream_sha256": self.last_hash})

    def authorize_stage_b(self, authorization: Mapping[str, Any], *, verifier: Callable[[Mapping[str, Any]], bool]) -> str:
        if not callable(verifier) or verifier(authorization) is not True:
            raise LedgerIntegrityError("Stage B authorization was not independently verified")
        return self._append("STAGE_B_AUTH", dict(authorization))

    def seal_complete(self) -> str:
        return self._append("COMPLETE", {"stream_sha256": self.last_hash})

    def offset(self) -> dict[str, Any]:
        return {"bytes": self.path.stat().st_size, "records": len(self.records), "sha256": self.last_hash,
                "next_position": self.position, "phase": self.phase}

    def close(self) -> None:
        if not self._lock.closed:
            self._lock.close()

    def __enter__(self) -> "DurableLedger":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
