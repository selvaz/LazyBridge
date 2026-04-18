"""PipelinePanel — GUI for ``LazyTool.chain`` / ``LazyTool.parallel`` tools.

Inspect tab shows the mode ("chain" / "parallel") and the participant list
so you can see the topology at a glance.  The test tab accepts a free-form
task string and invokes the pipeline live — identical to calling
``pipeline_tool.run({"task": ...})`` from Python.

Per-step visibility (e.g. live progress of each chain step) is a planned
follow-up.  For now the test tab only surfaces the final combined output.
"""

from __future__ import annotations

from typing import Any

from lazybridge.gui._panel import Panel


def is_pipeline_tool(tool: Any) -> bool:
    """Return ``True`` when ``tool`` is a chain/parallel/agent_tool wrapper."""
    return bool(getattr(tool, "_is_pipeline_tool", False))


class PipelinePanel(Panel):
    """Panel for a pipeline :class:`~lazybridge.lazy_tool.LazyTool`."""

    kind = "pipeline"

    def __init__(self, tool: Any) -> None:
        if not is_pipeline_tool(tool):
            raise ValueError(
                f"Tool {tool.name!r} is not a pipeline tool; use ToolPanel instead."
            )
        self._tool = tool

    @property
    def id(self) -> str:
        return f"pipeline-{self._tool.name}"

    @property
    def label(self) -> str:
        cfg = self._tool._pipeline
        mode = cfg.mode if cfg is not None else "pipeline"
        n = len(cfg.participants) if cfg is not None else 0
        return f"{self._tool.name} · {mode} ({n})"

    # ------------------------------------------------------------------

    def _describe_participant(self, p: Any) -> dict[str, Any]:
        """Render a participant entry for the sidebar/topology view."""
        # LazyAgent path — has a model / provider
        if hasattr(p, "_provider_name") and hasattr(p, "name"):
            return {
                "kind": "agent",
                "name": p.name,
                "provider": getattr(p, "_provider_name", "?"),
                "model": getattr(p, "_model_name", "?"),
                "panel_id": f"agent-{p.id}" if hasattr(p, "id") else None,
            }
        # Nested LazyTool
        if hasattr(p, "name") and hasattr(p, "run"):
            sub_pipeline = is_pipeline_tool(p)
            return {
                "kind": "pipeline" if sub_pipeline else "tool",
                "name": p.name,
                "panel_id": (f"pipeline-{p.name}" if sub_pipeline else f"tool-{p.name}"),
            }
        return {"kind": "unknown", "name": repr(p), "panel_id": None}

    # ------------------------------------------------------------------

    def render_state(self) -> dict[str, Any]:
        tool = self._tool
        cfg = tool._pipeline
        participants = []
        mode = None
        combiner = None
        concurrency_limit = None
        step_timeout = None
        guidance = None
        if cfg is not None:
            mode = cfg.mode
            combiner = cfg.combiner
            concurrency_limit = cfg.concurrency_limit
            step_timeout = cfg.step_timeout
            guidance = cfg.guidance
            participants = [self._describe_participant(p) for p in cfg.participants]
        return {
            "name": tool.name,
            "description": getattr(tool, "description", "") or "",
            "mode": mode,
            "combiner": combiner,
            "concurrency_limit": concurrency_limit,
            "step_timeout": step_timeout,
            "guidance": guidance,
            "participants": participants,
        }

    # ------------------------------------------------------------------

    def handle_action(self, action: str, args: dict[str, Any]) -> dict[str, Any]:
        if action == "run":
            task = args.get("task", "")
            if not isinstance(task, str) or not task.strip():
                raise ValueError("'task' is required")
            result = self._tool.run({"task": task})
            return {"result": result if isinstance(result, (str, int, float, bool, type(None))) else repr(result)}
        return super().handle_action(action, args)
