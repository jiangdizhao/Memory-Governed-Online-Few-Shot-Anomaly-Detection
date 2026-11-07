"""Spawned, label-blind research worker and separately spawned feedback broker."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
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
                ledger.execution_manifest.get("schema") not in ("FS4_R4_BLINDED_EXECUTION_V1",
                                                                  "FS4_JOINT_CAMPAIGN_EXECUTION_V1")):
            raise FS4BoundaryError("blinded context requires its registered research ledger")
        self._full_validation = True; self._freeze_pending = True
        self._initialize(ledger=ledger, category=category, controller=controller, research_payload=ledger.execution_manifest)
        self._full_validation = False
        from experiments.risk_safe.native_bridge import dynamic_codebooks
        if any(book.size() != 0 for book in dynamic_codebooks(self._runtime.bank)) or self._runtime.correction_cb.size() != 0:
            raise FS4BoundaryError("initial Greedy and C1 memories must be empty")
        if ledger.execution_manifest.get("schema") == "FS4_JOINT_CAMPAIGN_EXECUTION_V1":
            self._trajectory.audit_snapshot = lambda: _campaign_audit_snapshot(self._trajectory)


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
        if (position == self._manifest["categories"][self._category]["events"][0]["position"] and
                self._ledger.execution_manifest.get("schema") == "FS4_JOINT_CAMPAIGN_EXECUTION_V1"):
            supports = self._manifest["categories"][self._category]["supports"]
            reference = self._support_reference
            provenance["support_bootstrap"] = {
                "support_groups": [item["group"] for item in supports],
                "support_content_sha256": [item["content_sha256"] for item in supports],
                "support_self_scores": list(reference["support_scores"]),
                "mean": reference["mean"], "sample_std_ddof1": reference["sample_std_ddof1"],
                "theta0": reference["theta"], "writer_sigma": reference["writer_reference"]["sigma"],
                "fixed_support_sha256": reference["fixed_support_sha256"],
                "preflight_receipt_sha256": self._ledger.execution_manifest["support_preflight_receipt_sha256"],
                "additional_normal_images": 0,
            }
        return {"position": position, "group": row["group"], "event": row["event"], "image": image,
                "features": features[0], "cls_feature": cls_array, "feature_convention": FEATURE_CONVENTION,
                "event_provenance": provenance}


class ResearchWorkerTimeout(ResearchWorkerError):
    """A supervised worker or broker exceeded the caller's remaining wall budget."""


def _campaign_audit_snapshot(trajectory: Any) -> dict[str, Any]:
    """Deterministic per-event counts used for later audit and temporal summaries."""
    from .fs4_threshold_core import weights
    runtime = trajectory.runtime
    controllers = ({"L": trajectory.L, "E": trajectory.E} if hasattr(trajectory, "L") else
                   {trajectory.kind: trajectory.controller})
    rows = {}
    for name, controller in controllers.items():
        probabilities = [item.receipt.design_probability for item in controller.evidence]
        ess = weights(probabilities, controller.engine.w_max)[2] if probabilities else 0.0
        row = {"thresholds": dict(controller.engine.theta), "status": dict(controller.engine.status),
               "suppression": dict(controller.engine.suppression), "revision": controller.revision,
               "normal_evidence_groups": len({item.group for item in controller.evidence}),
               "effective_sample_size": ess, "threshold_tokens": dict(controller.engine.tokens)}
        if hasattr(controller, "cache"):
            row.update({"cache_groups": len(controller.cache.records),
                        "cache_bytes": controller.cache.bytes_report()["actual_tensor_bytes"],
                        "rescore_seconds_total": float(sum(controller.rescore_seconds)),
                        "rescore_invocations": controller.rescore_invocations,
                        "rescore_calls": dict(controller.rescore_calls)})
        rows[name] = row
    from experiments.risk_safe.native_bridge import dynamic_codebooks
    return {"controllers": rows, "normal_version": runtime.normal_version,
            "correction_version": runtime.correction_version,
            "greedy_dynamic_prototypes": sum(book.size() for book in dynamic_codebooks(runtime.bank)),
            "correction_prototypes": runtime.correction_cb.size(),
            "scorer_states": runtime.states()}


def _worker_main(connection: Any, broker: Any, config_path: str, broker_manifest_sha256: str) -> None:
    ledger = None
    worker_started = time.perf_counter()
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
        if manifest.get("research_execution_manifest", {}).get("schema") not in (
                "FS4_R4_BLINDED_EXECUTION_V1", "FS4_JOINT_CAMPAIGN_EXECUTION_V1"):
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
            elif action == "stats" and argument is None and manifest.get("trajectory_id") in ("S", "CL", "CE"):
                import ctypes
                class _Counters(ctypes.Structure):
                    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
                host_peak = None
                try:
                    counters = _Counters(); counters.cb = ctypes.sizeof(_Counters)
                    process = ctypes.windll.kernel32.GetCurrentProcess()
                    if ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
                        host_peak = int(counters.PeakWorkingSetSize)
                except Exception:
                    pass
                controller_map = ({"L": context._trajectory.L, "E": context._trajectory.E}
                                  if hasattr(context._trajectory, "L") else
                                  {context._trajectory.kind: context._trajectory.controller})
                result = {"active_worker_seconds": time.perf_counter() - worker_started,
                          "host_peak_working_set_bytes": host_peak,
                          "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                          "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
                          "rescore_invocations": sum(getattr(c, "rescore_invocations", 0) for c in controller_map.values()),
                          "rescore_seconds": sum(sum(getattr(c, "rescore_seconds", ())) for c in controller_map.values()),
                          "cache_bytes": sum(getattr(c, "cache", None).bytes_report()["actual_tensor_bytes"]
                                             for c in controller_map.values() if hasattr(c, "cache"))}
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
    def __init__(self, *, private_root: Path, run_root: Path, disclosures_root: Path,
                 manifest_path: Path, timeout_seconds: float | None = None) -> None:
        from .fs4_r4_broker import broker_main
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self.endpoint = parent
        self.process = context.Process(target=broker_main, args=(child,), kwargs={"private_root": str(Path(private_root).absolute()),
            "run_root": str(Path(run_root).absolute()), "disclosures_root": str(Path(disclosures_root).absolute()),
            "manifest_path": str(Path(manifest_path).absolute())})
        self.process.start(); child.close()
        if not self.endpoint.poll(timeout_seconds):
            self._terminate()
            raise ResearchWorkerTimeout("feedback broker startup exceeded the wall budget")
        try:
            reply = json.loads(self.endpoint.recv_bytes(4096))
        except (EOFError, OSError) as exc:
            self._terminate()
            raise ResearchWorkerError("feedback broker exited during startup") from exc
        if set(reply) != {"status", "manifest_sha256"} or reply["status"] != "READY":
            self._terminate(); raise ResearchWorkerError("feedback broker did not start")
        self.manifest_sha256 = reply["manifest_sha256"]

    def _terminate(self) -> None:
        if self.process.is_alive():
            self.process.terminate(); self.process.join(timeout=1.0)
            if self.process.is_alive() and hasattr(self.process, "kill"):
                self.process.kill(); self.process.join(timeout=1.0)
        try: self.endpoint.close()
        except OSError: pass


class BlindedResearchWorker:
    def __init__(self, config_path: Path, broker: BrokerProcess,
                 timeout_seconds: float | None = None) -> None:
        if type(broker) is not BrokerProcess or not Path(config_path).is_absolute():
            raise ValueError("registered configuration and separate broker required")
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self._connection = parent; self._broker = broker
        self._process = context.Process(target=_worker_main, args=(child, broker.endpoint, str(config_path), broker.manifest_sha256))
        self._process.start(); child.close(); broker.endpoint.close()
        try: self._receive("READY", timeout_seconds=timeout_seconds)
        except Exception:
            self._terminate(); raise

    def _terminate(self) -> None:
        for process in (self._process, self._broker.process):
            if process.is_alive():
                process.terminate(); process.join(timeout=1.0)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill(); process.join(timeout=1.0)
        try: self._connection.close()
        except OSError: pass
        try: self._broker.endpoint.close()
        except OSError: pass

    def _receive(self, expected: str, *, timeout_seconds: float | None = None) -> dict[str, Any]:
        if not self._connection.poll(timeout_seconds):
            self._terminate()
            raise ResearchWorkerTimeout("blinded worker operation exceeded the remaining wall budget")
        try: reply = json.loads(self._connection.recv_bytes(4 * 1024 * 1024))
        except (EOFError, OSError) as exc: raise ResearchWorkerError("blinded worker exited") from exc
        if reply.get("status") != expected: raise ResearchWorkerError(reply.get("message", "blinded worker rejected request"))
        return reply

    def _command(self, action: str, argument: Any, *, timeout_seconds: float | None = None) -> Any:
        self._connection.send_bytes(canonical({"command": action, "argument": argument}))
        return self._receive("OK", timeout_seconds=timeout_seconds)["result"]

    def step(self, position: int, *, timeout_seconds: float | None = None) -> Any:
        if type(position) is not int: raise TypeError("step accepts only an event position")
        return self._command("step", position, timeout_seconds=timeout_seconds)

    def checkpoint(self, name: str, *, timeout_seconds: float | None = None) -> Any:
        return self._command("checkpoint", name, timeout_seconds=timeout_seconds)
    def seal_category(self, *, timeout_seconds: float | None = None) -> str:
        return self._command("seal_category", None, timeout_seconds=timeout_seconds)
    def seal_global_a(self, *, timeout_seconds: float | None = None) -> str:
        return self._command("seal_global_a", None, timeout_seconds=timeout_seconds)
    def stats(self, *, timeout_seconds: float | None = None) -> dict[str, Any]:
        return self._command("stats", None, timeout_seconds=timeout_seconds)

    def close(self, *, timeout_seconds: float = 5.0) -> None:
        close_error: Exception | None = None
        if self._process.is_alive():
            try:
                self._connection.send_bytes(canonical({"command": "close", "argument": None}))
                self._receive("CLOSED", timeout_seconds=timeout_seconds)
            except (OSError, EOFError, ResearchWorkerError) as exc:
                close_error = exc
        self._process.join(timeout=timeout_seconds); self._broker.process.join(timeout=timeout_seconds)
        if self._process.is_alive() or self._broker.process.is_alive():
            if close_error is None:
                close_error = ResearchWorkerTimeout("worker process did not stop within bounded shutdown")
            self._terminate()
        try: self._connection.close()
        except OSError: pass
        if isinstance(close_error, ResearchWorkerTimeout):
            raise ResearchWorkerTimeout("blinded worker did not complete bounded shutdown") from close_error
        if close_error is not None and not isinstance(close_error, (OSError, EOFError)):
            raise ResearchWorkerError("blinded worker failed during shutdown") from close_error
