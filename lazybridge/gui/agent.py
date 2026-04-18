"""AgentPanel — GUI for ``LazyAgent``.

Exposes three things in the browser:

1. **Inspect** — provider, model, name, description (read-only).
2. **Edit** — ``system`` prompt (mutable at runtime; reread on every call)
   and the tool set (checkbox list whose scope defaults to every tool in
   the enclosing :class:`LazySession`).
3. **Test** — live ``chat`` / ``loop`` / ``text`` invocations against the
   real provider.  Results show content + usage + cost.
"""

from __future__ import annotations

from typing import Any

from lazybridge.gui._panel import Panel


class AgentPanel(Panel):
    """Panel for a :class:`~lazybridge.lazy_agent.LazyAgent` instance."""

    kind = "agent"

    def __init__(self, agent: Any, *, available_tools: list[Any] | None = None) -> None:
        self._agent = agent
        #: Explicit tool-scope override. When ``None``, tools are drawn
        #: from the enclosing :class:`LazySession` at render time.
        self._available_tools_override = available_tools

    # ------------------------------------------------------------------
    # Panel protocol
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        return f"agent-{self._agent.id}"

    @property
    def label(self) -> str:
        prov = getattr(self._agent, "_provider_name", "?")
        return f"{self._agent.name} · {prov}"

    # ------------------------------------------------------------------

    def _available_tools(self) -> list[Any]:
        """Return the set of tools the user can enable on this agent.

        Default scope: every ``LazyTool`` currently bound to any agent in
        the same ``LazySession`` (deduped by tool name).  When the agent
        has no session, falls back to the explicit ``available_tools``
        passed to the panel constructor, else the agent's own tool list.
        """
        if self._available_tools_override is not None:
            return list(self._available_tools_override)

        session = getattr(self._agent, "session", None)
        if session is None:
            return list(getattr(self._agent, "tools", []) or [])

        seen: dict[str, Any] = {}
        for agent in list(getattr(session, "_agents", []) or []):
            for tool in list(getattr(agent, "tools", []) or []):
                name = getattr(tool, "name", None)
                if name and name not in seen:
                    seen[name] = tool
        return list(seen.values())

    def _tool_descriptor(self, tool: Any) -> dict[str, Any]:
        return {
            "name": getattr(tool, "name", repr(tool)),
            "description": getattr(tool, "description", "") or "",
        }

    def _last_output_preview(self) -> str | None:
        out = getattr(self._agent, "_last_output", None)
        if not out:
            return None
        return out if len(out) <= 2000 else out[:2000] + "…"

    def render_state(self) -> dict[str, Any]:
        agent = self._agent
        tools = [self._tool_descriptor(t) for t in (getattr(agent, "tools", None) or [])]
        available = [self._tool_descriptor(t) for t in self._available_tools()]
        return {
            "name": agent.name,
            "description": getattr(agent, "description", None) or "",
            "provider": getattr(agent, "_provider_name", "?"),
            "model": getattr(agent, "_model_name", "?"),
            "system": getattr(agent, "system", None) or "",
            "tools": tools,
            "available_tools": available,
            "has_native_tools": bool(getattr(agent, "native_tools", None)),
            "has_output_schema": getattr(agent, "output_schema", None) is not None,
            "session_id": getattr(getattr(agent, "session", None), "id", None),
            "last_output": self._last_output_preview(),
        }

    # ------------------------------------------------------------------

    def handle_action(self, action: str, args: dict[str, Any]) -> dict[str, Any]:
        if action == "update_system":
            value = str(args.get("value", ""))
            self._agent.system = value
            return {"ok": True, "system": value}

        if action == "toggle_tool":
            name = args.get("name")
            if not isinstance(name, str):
                raise ValueError("'name' is required")
            enabled = bool(args.get("enabled"))
            current = list(getattr(self._agent, "tools", None) or [])
            if enabled:
                already = any(getattr(t, "name", None) == name for t in current)
                if not already:
                    lookup = {getattr(t, "name", None): t for t in self._available_tools()}
                    tool = lookup.get(name)
                    if tool is None:
                        raise ValueError(
                            f"Tool {name!r} is not in the agent's tool scope; "
                            "add it to a session agent first."
                        )
                    current.append(tool)
            else:
                current = [t for t in current if getattr(t, "name", None) != name]
            self._agent.tools = current
            return {
                "ok": True,
                "tools": [self._tool_descriptor(t) for t in current],
            }

        if action == "test":
            mode = args.get("mode", "chat")
            message = args.get("message", "")
            if not isinstance(message, str) or not message.strip():
                raise ValueError("'message' is required")
            if mode not in {"chat", "loop", "text"}:
                raise ValueError(f"unsupported mode {mode!r} — choose chat | loop | text")
            return self._run_test(mode, message)

        return super().handle_action(action, args)

    # ------------------------------------------------------------------

    def _run_test(self, mode: str, message: str) -> dict[str, Any]:
        agent = self._agent
        if mode == "chat":
            resp = agent.chat(message)
        elif mode == "loop":
            resp = agent.loop(message)
        elif mode == "text":
            return {"content": agent.text(message)}
        else:  # pragma: no cover - validated above
            raise ValueError(f"unsupported mode {mode!r}")

        usage = getattr(resp, "usage", None)
        usage_payload: dict[str, Any] = {}
        if usage is not None:
            usage_payload = {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "thinking_tokens": getattr(usage, "thinking_tokens", 0),
                "cost_usd": getattr(usage, "cost_usd", None),
            }
        return {
            "mode": mode,
            "content": getattr(resp, "content", "") or "",
            "parsed": _safe_parsed(resp),
            "stop_reason": getattr(resp, "stop_reason", None),
            "model": getattr(resp, "model", None),
            "usage": usage_payload,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in (getattr(resp, "tool_calls", None) or [])
            ],
        }


def _safe_parsed(resp: Any) -> Any:
    """Return ``resp.parsed`` if it's JSON-safe, else ``None``.

    Pydantic instances are converted via ``.model_dump()``; anything else
    non-primitive becomes ``repr()`` so the JSON response never fails.
    """
    parsed = getattr(resp, "parsed", None)
    if parsed is None:
        return None
    dump = getattr(parsed, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            return repr(parsed)
    if isinstance(parsed, (str, int, float, bool, list, dict, type(None))):
        return parsed
    return repr(parsed)
