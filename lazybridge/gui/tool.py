"""ToolPanel — GUI for ``LazyTool``.

Inspect tab is fully read-only (name, description, guidance, compiled JSON
schema).  Mutating tool metadata is discouraged because the compiled
schema is cached; use ``LazyTool.specialize`` from code instead.

Test tab generates a form from the tool's JSON Schema (one input per
parameter, typed) and invokes ``tool.run(args)`` on submit — including
for pipeline tools (chain/parallel), whose schema is ``{"task": str}``.
"""

from __future__ import annotations

from typing import Any

from lazybridge.gui._panel import Panel


class ToolPanel(Panel):
    """Panel for a :class:`~lazybridge.lazy_tool.LazyTool` instance."""

    kind = "tool"

    def __init__(self, tool: Any) -> None:
        self._tool = tool

    @property
    def id(self) -> str:
        return f"tool-{self._tool.name}"

    @property
    def label(self) -> str:
        tag = ""
        if getattr(self._tool, "_is_pipeline_tool", False):
            tag = " · pipeline"
        elif getattr(self._tool, "_delegate", None) is not None:
            tag = " · from_agent"
        return f"{self._tool.name}{tag}"

    # ------------------------------------------------------------------

    def _parameters(self) -> dict[str, Any]:
        try:
            defn = self._tool.definition()
        except Exception:  # pragma: no cover - defensive
            return {"type": "object", "properties": {}, "required": []}
        params = getattr(defn, "parameters", None) or {}
        return params if isinstance(params, dict) else {}

    def render_state(self) -> dict[str, Any]:
        tool = self._tool
        schema_mode = getattr(tool, "schema_mode", None)
        return {
            "name": tool.name,
            "description": getattr(tool, "description", None) or "",
            "guidance": getattr(tool, "guidance", None),
            "schema_mode": getattr(schema_mode, "name", str(schema_mode)) if schema_mode is not None else None,
            "strict": bool(getattr(tool, "strict", False)),
            "is_pipeline_tool": bool(getattr(tool, "_is_pipeline_tool", False)),
            "is_delegate": getattr(tool, "_delegate", None) is not None,
            "parameters": self._parameters(),
        }

    # ------------------------------------------------------------------

    def handle_action(self, action: str, args: dict[str, Any]) -> dict[str, Any]:
        if action == "invoke":
            payload = args.get("args", {})
            if not isinstance(payload, dict):
                raise ValueError("'args' must be an object")
            result = self._tool.run(payload)
            return {"result": _jsonable(result)}
        return super().handle_action(action, args)


def _jsonable(value: Any) -> Any:
    """Best-effort JSON coercion for arbitrary tool return values."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    return repr(value)
