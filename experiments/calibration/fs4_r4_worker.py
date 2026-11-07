"""Spawned, label-blind research worker and separately spawned feedback broker."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .fs4_durable_ledger import canonical, digest
from .fs4_r2_context import FS4BoundaryError
from .fs4_research_context import ResearchContext
from .fs4_research_worker import ResearchWorkerError
from .fs4_source_excluded_scorer import _feature_digest
from .fs4_threshold_integration import FEATURE_CONVENTION


class BlindedResearchContext(ResearchContext):
    def __init__(self, *, ledger: Any, category: str, controller: str | None = None) -> None:
        from .fs4_research_ledger import ResearchLedger
        if (multiprocessing.current_process().name == "MainProcess" or type(ledger) is not ResearchLedger or
                ledger.execution_manifest.get("schema") != "FS4_R4_BLINDED_EXECUTION_V1"):
            raise FS4BoundaryError("blinded context requires its registered research ledger")
        self._full_validation = True; self._freeze_pending = True
        self._initialize(ledger=ledger, category=category, controller=controller, research_payload=ledger.execution_manifest)
        self._full_validation = False
        from experiments.risk_safe.native_bridge import dynamic_codebooks
        if any(book.size() != 0 for book in dynamic_codebooks(self._runtime.bank)) or self._runtime.correction_cb.size() != 0:
            raise FS4BoundaryError("initial Greedy and C1 memories must be empty")

    def _encode_event(self, position: int) -> dict[str, Any]:
        from .fs4_r4_blinded import read_blinded_image
        if digest(self._manifest) != self._manifest_digest or digest(self._ledger._execution_manifest) != self._manifest_digest:
            raise FS4BoundaryError("blinded execution registration changed")
        self._ledger.verify_live_sources()
        row = self._registered_event(position)
        self._verify_runtime(); self._verify_model()
        image = read_blinded_image(self._manifest, row)
        features, cls = self._impl.encode_pil(self._backbone, image, self._cfg.feature_fuse)
        self._verify_model()
        if (len(features) != 1 or tuple(features[0].shape) != (784, 4096) or
                features[0].dtype != torch.float32 or not torch.isfinite(features[0]).all()):
            raise FS4BoundaryError("native encoder did not produce finite full FP32 features")
        cls_array = cls.detach().float().cpu().numpy().reshape(-1)
        if cls_array.shape != (int(self._backbone.model.num_features),) or not np.isfinite(cls_array).all():
            raise FS4BoundaryError("native encoder did not produce finite CLS features")
        provenance = {"schema": "FS4_TRUSTED_EVENT_V1", "dataset_identity_sha256": self._manifest["dataset_identity_sha256"],
            "manifest_sha256": self._manifest_digest, "category": self._category, "image_content_sha256": row["content_sha256"],
            "geometry": [image.width, image.height], "model_state_sha256": self._manifest["model_state_sha256"],
            "cls_sha256": hashlib.sha256(cls_array.tobytes()).hexdigest(), "feature_sha256": _feature_digest(features[0]),
            "source_commit": self._ledger.manifest["source_commit"],
            "fixed_support_sha256": self._ledger.manifest["fixed_support_sha256_by_category"][self._category],
            "resource_sha256": row["resource_sha256"], "pixel_sha256": row["pixel_sha256"], "view": row["view"]}
        return {"position": position, "group": row["group"], "event": row["event"], "image": image,
                "features": features[0], "cls_feature": cls_array, "feature_convention": FEATURE_CONVENTION,
                "event_provenance": provenance}


def _worker_main(connection: Any, broker: Any, config_path: str, broker_manifest_sha256: str) -> None:
    ledger = None
    try:
        from .fs4_checkpoint import RecoveryLedger, begin_recovery, save_checkpoint
        from .fs4_durable_runtime import _DurableCore
        from .fs4_r2_security import WindowsPrivateStore, validate_private_acl
        from .fs4_research_ledger import ResearchLedger
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if set(config) != {"manifest_path", "run_root", "category", "create", "checkpoint"} or type(config["create"]) is not bool:
            raise ResearchWorkerError("invalid blinded worker configuration")
        if any(not isinstance(config[k], str) or not Path(config[k]).is_absolute() for k in ("manifest_path", "run_root")):
            raise ResearchWorkerError("blinded worker paths must be absolute")
        manifest = json.loads(Path(config["manifest_path"]).read_text(encoding="utf-8"))
        if digest(manifest) != broker_manifest_sha256:
            raise ResearchWorkerError("broker and worker registrations differ")
        if manifest.get("research_execution_manifest", {}).get("schema") != "FS4_R4_BLINDED_EXECUTION_V1":
            raise ResearchWorkerError("worker requires a blinded execution registration")
        run_root = Path(config["run_root"])
        ledger = ResearchLedger(run_root, manifest, create=config["create"])
        if ledger.current_category is None: ledger.select_category(config["category"])
        if ledger.current_category != config["category"]:
            raise ResearchWorkerError("worker category differs")
        context = BlindedResearchContext(ledger=ledger, category=config["category"], controller=manifest.get("controller"))
        ledger._trusted_context = context

        def query_only(event: str, position: int) -> int:
            if ledger.phase != "LABEL_INTENT" or ledger.prediction is None or len(ledger.records) < 2:
                raise ResearchWorkerError("query-only feedback requires a durable intent")
            qrow = ledger.records[-2]
            if qrow["kind"] != "QUERY": raise ResearchWorkerError("durable query receipt missing")
            broker.send_bytes(canonical({"event": event, "group": ledger.prediction["group"], "position": position,
                                         "query_sha256": qrow["payload"]["query_sha256"]}))
            reply = json.loads(broker.recv_bytes(4096))
            if set(reply) != {"status", "label"} or reply["status"] != "OK" or type(reply["label"]) is not int or reply["label"] not in (0, 1):
                raise ResearchWorkerError("query-only feedback broker rejected request")
            return reply["label"]

        if config["checkpoint"] is None:
            if ledger.phase != "READY" or ledger.position != context._trajectory.policy.start_position:
                raise ResearchWorkerError("advanced ledger requires checkpoint recovery")
            selected = ledger
        else:
            store = WindowsPrivateStore(run_root / "checkpoints", create=False)
            path = store._child(config["checkpoint"], must_exist=True)
            validate_private_acl(path, directory=True)
            for name in ("state.json", "tensors.npz", "COMMITTED.json"): validate_private_acl(path / name, directory=False)
            selected = begin_recovery(path, trajectory=context._trajectory, ledger=ledger, label_reader=query_only)
            context.validate_full()
        core = _DurableCore(context._trajectory, selected, query_only)
        connection.send_bytes(canonical({"status": "READY", "mode": "TRUSTED_LOCAL_RESEARCH"}))
        while True:
            command = json.loads(connection.recv_bytes(4096))
            if not isinstance(command, dict) or set(command) != {"command", "argument"}:
                raise ResearchWorkerError("invalid blinded worker command")
            action, argument = command["command"], command["argument"]
            if action == "step" and type(argument) is int:
                event = context._encode_event(argument)
                ledger._active_provenance = dict(event["event_provenance"])
                try: row = core.step(**event)
                finally: ledger._active_provenance = None
                result = {"prediction": row["sealed_prediction"], "query": asdict(row["query"]),
                          "feedback_update": row["feedback_update"], "next_thresholds": row["next_thresholds"]}
            elif action == "checkpoint" and isinstance(argument, str):
                if isinstance(selected, RecoveryLedger): selected.assert_caught_up()
                store = WindowsPrivateStore(run_root / "checkpoints", create=not (run_root / "checkpoints").exists())
                result = save_checkpoint(store._child(argument, must_exist=False), trajectory=context._trajectory,
                                         ledger=ledger, private_store=store)
            elif action == "seal_category" and argument is None:
                context.validate_full(); result = core.seal_category(config["category"])
            elif action == "seal_global_a" and argument is None:
                context.validate_full(); result = ledger.seal_global_stage_a()
            elif action == "close" and argument is None:
                connection.send_bytes(canonical({"status": "CLOSED"})); break
            else:
                raise ResearchWorkerError("unsupported blinded worker command")
            connection.send_bytes(canonical({"status": "OK", "result": result}))
    except Exception:
        try: connection.send_bytes(canonical({"status": "ERROR", "message": "blinded worker failed closed"}))
        except Exception: pass
    finally:
        if ledger is not None: ledger.close()
        try: broker.send_bytes(canonical({"command": "close"})); broker.recv_bytes(4096)
        except Exception: pass
        broker.close(); connection.close()


class BrokerProcess:
    """Owner-side launcher; private mapping paths are never sent to the worker."""
    def __init__(self, *, private_root: Path, run_root: Path, disclosures_root: Path, manifest_path: Path) -> None:
        from .fs4_r4_broker import broker_main
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self.endpoint = parent
        self.process = context.Process(target=broker_main, args=(child,), kwargs={"private_root": str(Path(private_root).absolute()),
            "run_root": str(Path(run_root).absolute()), "disclosures_root": str(Path(disclosures_root).absolute()),
            "manifest_path": str(Path(manifest_path).absolute())})
        self.process.start(); child.close()
        reply = json.loads(self.endpoint.recv_bytes(4096))
        if set(reply) != {"status", "manifest_sha256"} or reply["status"] != "READY":
            self.process.join(); self.endpoint.close(); raise ResearchWorkerError("feedback broker did not start")
        self.manifest_sha256 = reply["manifest_sha256"]


class BlindedResearchWorker:
    def __init__(self, config_path: Path, broker: BrokerProcess) -> None:
        if type(broker) is not BrokerProcess or not Path(config_path).is_absolute():
            raise ValueError("registered configuration and separate broker required")
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self._connection = parent; self._broker = broker
        self._process = context.Process(target=_worker_main, args=(child, broker.endpoint, str(config_path), broker.manifest_sha256))
        self._process.start(); child.close(); broker.endpoint.close()
        try: self._receive("READY")
        except Exception:
            self._process.join(); broker.process.join(); self._connection.close(); raise

    def _receive(self, expected: str) -> dict[str, Any]:
        try: reply = json.loads(self._connection.recv_bytes(4 * 1024 * 1024))
        except (EOFError, OSError) as exc: raise ResearchWorkerError("blinded worker exited") from exc
        if reply.get("status") != expected: raise ResearchWorkerError(reply.get("message", "blinded worker rejected request"))
        return reply

    def _command(self, action: str, argument: Any) -> Any:
        self._connection.send_bytes(canonical({"command": action, "argument": argument}))
        return self._receive("OK")["result"]

    def step(self, position: int) -> Any:
        if type(position) is not int: raise TypeError("step accepts only an event position")
        return self._command("step", position)

    def checkpoint(self, name: str) -> Any: return self._command("checkpoint", name)
    def seal_category(self) -> str: return self._command("seal_category", None)
    def seal_global_a(self) -> str: return self._command("seal_global_a", None)

    def close(self) -> None:
        if self._process.is_alive():
            try: self._connection.send_bytes(canonical({"command": "close", "argument": None})); self._receive("CLOSED")
            except (OSError, EOFError, ResearchWorkerError): pass
        self._process.join(); self._broker.process.join(); self._connection.close()
