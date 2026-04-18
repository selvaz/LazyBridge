"""PipelinePanel — GUI for ``LazyTool.chain`` / ``LazyTool.parallel`` tools.

Two jobs:

1. **Inspect** — topology (mode / combiner / concurrency / timeouts /
   participants).  Participant entries link to their own panels so a
   chain's topology becomes navigable inside the single tab.
2. **Test (live)** — runs the pipeline in a background thread and streams
   per-step progress events (``AGENT_START`` / ``AGENT_FINISH``) back
   into the panel state.  Clients discover updates through the existing
   sidebar polling loop — no new endpoint needed.

Progress capture is **session-aware**: if the pipeline's participants are
in a :class:`LazySession`, we attach a :class:`CallbackExporter` to its
:class:`EventLog` for the duration of the run.  Participant clones are
bound to the original session's EventLog (see ``_clone_for_invocation``),
so events emitted by clones are picked up transparently.  Session-less
pipelines still run — they just surface the final result without a
per-step timeline.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from typing import Any

from lazybridge.gui._panel import Panel


def is_pipeline_tool(tool: Any) -> bool:
    """Return ``True`` when ``tool`` is a chain/parallel/agent_tool wrapper."""
    return bool(getattr(tool, "_is_pipeline_tool", False))


_INTERESTING_EVENTS = {"agent_start", "agent_finish"}
_MAX_EVENTS = 400


class PipelinePanel(Panel):
    """Panel for a pipeline :class:`~lazybridge.lazy_tool.LazyTool`."""

    kind = "pipeline"

    def __init__(self, tool: Any) -> None:
        if not is_pipeline_tool(tool):
            raise ValueError(
                f"Tool {tool.name!r} is not a pipeline tool; use ToolPanel instead."
            )
        self._tool = tool
        self._run_lock = threading.Lock()
        self._last_run: dict[str, Any] | None = None
        self._event_buffer: deque[dict[str, Any]] = deque(maxlen=_MAX_EVENTS)

    @property
    def id(self) -> str:
        return f"pipeline-{self._tool.name}"

    @property
    def label(self) -> str:
        cfg = self._tool._pipeline
        mode = cfg.mode if cfg is not None else "pipeline"
        n = len(cfg.participants) if cfg is not None else 0
        marker = " · running" if self._is_running() else ""
        return f"{self._tool.name} · {mode} ({n}){marker}"

    # ------------------------------------------------------------------
    # Inspect
    # ------------------------------------------------------------------

    def _describe_participant(self, p: Any) -> dict[str, Any]:
        if hasattr(p, "_provider_name") and hasattr(p, "name"):
            return {
                "kind": "agent",
                "name": p.name,
                "provider": getattr(p, "_provider_name", "?"),
                "model": getattr(p, "_model_name", "?"),
                "panel_id": f"agent-{p.id}" if hasattr(p, "id") else None,
            }
        if hasattr(p, "name") and hasattr(p, "run"):
            sub_pipeline = is_pipeline_tool(p)
            return {
                "kind": "pipeline" if sub_pipeline else "tool",
                "name": p.name,
                "panel_id": (f"pipeline-{p.name}" if sub_pipeline else f"tool-{p.name}"),
            }
        return {"kind": "unknown", "name": repr(p), "panel_id": None}

    def render_state(self) -> dict[str, Any]:
        tool = self._tool
        cfg = tool._pipeline
        participants = []
        mode = combiner = concurrency_limit = step_timeout = guidance = None
        participant_names: list[str] = []
        if cfg is not None:
            mode = cfg.mode
            combiner = cfg.combiner
            concurrency_limit = cfg.concurrency_limit
            step_timeout = cfg.step_timeout
            guidance = cfg.guidance
            participants = [self._describe_participant(p) for p in cfg.participants]
            participant_names = [p["name"] for p in participants]

        state: dict[str, Any] = {
            "name": tool.name,
            "description": getattr(tool, "description", "") or "",
            "mode": mode,
            "combiner": combiner,
            "concurrency_limit": concurrency_limit,
            "step_timeout": step_timeout,
            "guidance": guidance,
            "participants": participants,
            "participant_names": participant_names,
        }
        with self._run_lock:
            if self._last_run is not None:
                state["last_run"] = dict(self._last_run)
        return state

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def handle_action(self, action: str, args: dict[str, Any]) -> dict[str, Any]:
        if action == "run":
            task = args.get("task", "")
            if not isinstance(task, str) or not task.strip():
                raise ValueError("'task' is required")
            if self._is_running():
                raise ValueError("A run is already in flight — wait or cancel first")
            return self._start_run(task)

        if action == "clear_run":
            with self._run_lock:
                self._last_run = None
                self._event_buffer.clear()
            return {"ok": True}

        return super().handle_action(action, args)

    # ------------------------------------------------------------------
    # Background execution + event capture
    # ------------------------------------------------------------------

    def _is_running(self) -> bool:
        with self._run_lock:
            return self._last_run is not None and self._last_run.get("status") == "running"

    def _session_for_capture(self) -> Any | None:
        """Return the first session we find among pipeline participants."""
        cfg = self._tool._pipeline
        if cfg is None:
            return None
        for p in cfg.participants:
            sess = getattr(p, "session", None)
            if sess is not None and hasattr(sess, "events"):
                return sess
        return None

    def _interesting_names(self) -> set[str]:
        cfg = self._tool._pipeline
        if cfg is None:
            return set()
        names: set[str] = set()
        for p in cfg.participants:
            n = getattr(p, "name", None)
            if n:
                names.add(str(n))
        return names

    def _start_run(self, task: str) -> dict[str, Any]:
        run_id = str(uuid.uuid4())[:8]
        with self._run_lock:
            self._event_buffer.clear()
            self._last_run = {
                "run_id": run_id,
                "status": "running",
                "task": task,
                "started_at": time.time(),
                "finished_at": None,
                "events": [],
                "result": None,
                "error": None,
                "captured_from_session": False,
            }
        t = threading.Thread(
            target=self._run_in_thread,
            name=f"lazybridge-gui-pipe-{self._tool.name}-{run_id}",
            daemon=True,
            args=(run_id, task),
        )
        t.start()
        return {"started": True, "run_id": run_id}

    def _run_in_thread(self, run_id: str, task: str) -> None:
        session = self._session_for_capture()
        interesting = self._interesting_names()
        exporter = None
        if session is not None and interesting:
            exporter = _PipelineEventExporter(self, interesting, run_id)
            try:
                session.events.add_exporter(exporter)
            except Exception:  # pragma: no cover - defensive
                exporter = None

            with self._run_lock:
                if self._last_run and self._last_run["run_id"] == run_id:
                    self._last_run["captured_from_session"] = True

        try:
            result = self._tool.run({"task": task})
        except Exception as exc:
            with self._run_lock:
                if self._last_run and self._last_run["run_id"] == run_id:
                    self._last_run["status"] = "error"
                    self._last_run["error"] = f"{type(exc).__name__}: {exc}"
                    self._last_run["finished_at"] = time.time()
                    self._last_run["events"] = list(self._event_buffer)
            self.notify()
            return
        finally:
            if exporter is not None:
                try:
                    session.events.remove_exporter(exporter)
                except Exception:  # pragma: no cover
                    pass

        with self._run_lock:
            if self._last_run and self._last_run["run_id"] == run_id:
                self._last_run["status"] = "done"
                self._last_run["result"] = _jsonable_result(result)
                self._last_run["finished_at"] = time.time()
                self._last_run["events"] = list(self._event_buffer)
        self.notify()

    # ------------------------------------------------------------------
    # CallbackExporter — called from the running session's thread(s)
    # ------------------------------------------------------------------

    def _record_event(self, event: dict[str, Any], run_id: str) -> None:
        event_type = str(event.get("event_type", ""))
        if event_type not in _INTERESTING_EVENTS:
            return
        with self._run_lock:
            if self._last_run is None or self._last_run.get("run_id") != run_id:
                return
            slim = {
                "event_type": event_type,
                "agent_name": event.get("agent_name"),
                "ts": event.get("ts") or time.time(),
            }
            data = event.get("data") or {}
            if event_type == "agent_finish":
                if "stop_reason" in data:
                    slim["stop_reason"] = data["stop_reason"]
                if "n_steps" in data:
                    slim["n_steps"] = data["n_steps"]
            self._event_buffer.append(slim)
            # Live mirror — reflect in last_run immediately so clients
            # polling render_state() see progress without another action.
            self._last_run["events"] = list(self._event_buffer)
        # Push a refresh notification to SSE subscribers so the client
        # redraws the timeline without waiting for the next poll tick.
        self.notify()


class _PipelineEventExporter:
    """CallbackExporter-compatible adapter that forwards filtered events."""

    __slots__ = ("_panel_ref", "_names", "_run_id")

    def __init__(self, panel: PipelinePanel, names: set[str], run_id: str) -> None:
        self._panel_ref = panel
        self._names = names
        self._run_id = run_id

    def export(self, event: dict[str, Any]) -> None:
        name = event.get("agent_name")
        if name not in self._names:
            return
        self._panel_ref._record_event(event, self._run_id)


def _jsonable_result(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    return repr(value)
