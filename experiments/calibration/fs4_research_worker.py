"""Persistent spawned worker for the non-adversarial local research mode.

Launch is inert until explicitly called. No scientific run is authorized by
importing or installing this module. Parent IPC contains only JSON commands.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
from typing import Any

from .fs4_durable_ledger import canonical


class ResearchWorkerError(RuntimeError):
    pass


def _worker_main(connection: Any, config_path: str) -> None:
    """Spawn target: imports native research components in a fresh interpreter."""
    ledger = None
    try:
        from .fs4_checkpoint import RecoveryLedger, begin_recovery, save_checkpoint
        from .fs4_durable_runtime import _DurableCore
        from .fs4_r2_security import WindowsPrivateStore, validate_private_acl
        from .fs4_research_context import ResearchContext, require_label_blind_dataset_adapter
        from .fs4_research_ledger import ResearchLedger

        # The inherited registration exposes outcome-bearing paths. Check the
        # hold before opening even the launch configuration or run manifest.
        require_label_blind_dataset_adapter()
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        keys = {"manifest_path", "run_root", "category", "feedback_root", "create", "checkpoint"}
        if set(config) != keys or not isinstance(config["create"], bool):
            raise ResearchWorkerError("invalid worker launch registration")
        for name in ("manifest_path", "run_root", "feedback_root"):
            if not isinstance(config[name], str) or not Path(config[name]).is_absolute():
                raise ResearchWorkerError("worker paths must be absolute")
        manifest = json.loads(Path(config["manifest_path"]).read_text(encoding="utf-8"))
        feedback_root = Path(config["feedback_root"]).absolute()
        run_root = Path(config["run_root"]).absolute()
        if manifest.get("feedback_root") != str(feedback_root):
            raise ResearchWorkerError("feedback store differs from the registered manifest")
        if feedback_root == run_root or feedback_root.is_relative_to(run_root) or run_root.is_relative_to(feedback_root):
            raise ResearchWorkerError("query feedback store must be separate from the run")
        feedback = WindowsPrivateStore(feedback_root, create=False)
        ledger = ResearchLedger(run_root, manifest, create=config["create"])
        if ledger.current_category is None:
            ledger.select_category(config["category"])
        if ledger.current_category != config["category"]:
            raise ResearchWorkerError("worker category differs from active category")
        context = ResearchContext(ledger=ledger, category=config["category"], controller=manifest.get("controller"))
        ledger._trusted_context = context

        def read_query(event: str, position: int) -> int:
            # Only called by the durable ledger after its executed-query intent.
            row = json.loads(feedback.read(event + ".json"))
            if (set(row) != {"event", "position", "label"} or row["event"] != event or
                    row["position"] != position or row["label"] not in (0, 1) or isinstance(row["label"], bool)):
                raise ResearchWorkerError("queried feedback does not match the sealed event")
            return row["label"]

        checkpoint = config["checkpoint"]
        if checkpoint is None:
            if ledger.phase != "READY" or ledger.position != context._trajectory.policy.start_position:
                raise ResearchWorkerError("advanced ledger requires checkpoint recovery")
            selected = ledger
        else:
            store = WindowsPrivateStore(run_root / "checkpoints", create=False)
            path = store._child(checkpoint, must_exist=True)
            validate_private_acl(path, directory=True)
            for name in ("state.json", "tensors.npz", "COMMITTED.json"):
                validate_private_acl(path / name, directory=False)
            selected = begin_recovery(path, trajectory=context._trajectory, ledger=ledger, label_reader=read_query)
            context.validate_full()
        core = _DurableCore(context._trajectory, selected, read_query)
        connection.send_bytes(canonical({"status": "READY", "mode": "TRUSTED_LOCAL_RESEARCH",
                                         "source_commit": manifest["source_commit"]}))
        while True:
            command = json.loads(connection.recv_bytes(4096))
            if not isinstance(command, dict) or set(command) != {"command", "argument"}:
                raise ResearchWorkerError("invalid research worker command")
            action, argument = command["command"], command["argument"]
            if action == "step":
                if not isinstance(argument, int) or isinstance(argument, bool):
                    raise ResearchWorkerError("step accepts only a registered event position")
                event = context._encode_event(argument)
                ledger._active_provenance = dict(event["event_provenance"])
                try:
                    row = core.step(**event)
                finally:
                    ledger._active_provenance = None
                result = {"prediction": row["sealed_prediction"], "query": asdict(row["query"]),
                          "feedback_update": row["feedback_update"], "next_thresholds": row["next_thresholds"]}
            elif action == "checkpoint":
                if not isinstance(argument, str):
                    raise ResearchWorkerError("checkpoint command requires a name")
                if isinstance(selected, RecoveryLedger):
                    selected.assert_caught_up()
                store = WindowsPrivateStore(run_root / "checkpoints", create=not (run_root / "checkpoints").exists())
                result = save_checkpoint(store._child(argument, must_exist=False), trajectory=context._trajectory,
                                         ledger=ledger, private_store=store)
            elif action == "seal_category" and argument is None:
                context.validate_full()
                result = core.seal_category(config["category"])
            elif action == "seal_global_a" and argument is None:
                context.validate_full()
                result = ledger.seal_global_stage_a()
            elif action == "close" and argument is None:
                connection.send_bytes(canonical({"status": "CLOSED"}))
                break
            else:
                raise ResearchWorkerError("unsupported research worker command")
            connection.send_bytes(canonical({"status": "OK", "result": result}))
    except Exception as exc:
        try:
            connection.send_bytes(canonical({"status": "ERROR", "type": type(exc).__name__, "message": str(exc)}))
        except Exception:
            pass
    finally:
        if ledger is not None:
            ledger.close()
        connection.close()


class ResearchWorker:
    """Parent interface accepts event positions or explicitly named controls only."""

    def __init__(self, config_path: Path) -> None:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            raise ValueError("absolute worker configuration path required")
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self._connection = parent
        self._process = context.Process(target=_worker_main, args=(child, str(config_path)), daemon=False)
        self._process.start()
        child.close()
        try:
            self._receive("READY")
        except Exception:
            self._process.join()
            self._connection.close()
            raise

    def _receive(self, expected: str) -> dict[str, Any]:
        try:
            reply = json.loads(self._connection.recv_bytes(4 * 1024 * 1024))
        except (EOFError, OSError) as exc:
            raise ResearchWorkerError("research worker exited without a response") from exc
        if reply.get("status") != expected:
            raise ResearchWorkerError(f"worker {reply.get('type', 'error')}: {reply.get('message', reply)}")
        return reply

    def _command(self, action: str, argument: Any) -> Any:
        self._connection.send_bytes(canonical({"command": action, "argument": argument}))
        return self._receive("OK")["result"]

    def step(self, position: int) -> Any:
        if type(position) is not int:
            raise TypeError("step accepts only an integer event position")
        return self._command("step", position)

    def checkpoint(self, name: str) -> Any:
        return self._command("checkpoint", name)

    def seal_category(self) -> str:
        return self._command("seal_category", None)

    def seal_global_a(self) -> str:
        return self._command("seal_global_a", None)

    def close(self) -> None:
        if self._process.is_alive():
            try:
                self._connection.send_bytes(canonical({"command": "close", "argument": None}))
                self._receive("CLOSED")
            except (OSError, EOFError, ResearchWorkerError):
                pass
        self._process.join()
        self._connection.close()
