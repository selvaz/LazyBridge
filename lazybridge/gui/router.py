"""RouterPanel — GUI for ``LazyRouter``.

Inspect tab shows the routes table (``key → agent name``), the default
key, and the current condition callable's ``repr``.

Test tab accepts a free-form value which is passed to ``router.route()``;
the panel reports which key the condition returned, which agent was
selected, and optionally runs that agent against a user-supplied prompt.
"""

from __future__ import annotations

import inspect
from typing import Any

from lazybridge.gui._panel import Panel


class RouterPanel(Panel):
    """Panel for a :class:`~lazybridge.lazy_router.LazyRouter` instance."""

    kind = "router"

    def __init__(self, router: Any) -> None:
        self._router = router

    @property
    def id(self) -> str:
        return f"router-{self._router.id}"

    @property
    def label(self) -> str:
        return f"{self._router.name} · {len(self._router.routes)} routes"

    # ------------------------------------------------------------------

    def _condition_repr(self) -> str:
        cond = self._router.condition
        try:
            src = inspect.getsource(cond)
            return src.strip()
        except (OSError, TypeError):
            return repr(cond)

    def _route_descriptors(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for key, agent in self._router.routes.items():
            entry: dict[str, Any] = {
                "key": key,
                "agent_name": getattr(agent, "name", repr(agent)),
                "panel_id": f"agent-{agent.id}" if hasattr(agent, "id") else None,
            }
            provider = getattr(agent, "_provider_name", None)
            model = getattr(agent, "_model_name", None)
            if provider:
                entry["provider"] = provider
            if model:
                entry["model"] = model
            out.append(entry)
        return out

    def render_state(self) -> dict[str, Any]:
        return {
            "name": self._router.name,
            "default": self._router.default,
            "routes": self._route_descriptors(),
            "condition": self._condition_repr(),
        }

    # ------------------------------------------------------------------

    def handle_action(self, action: str, args: dict[str, Any]) -> dict[str, Any]:
        if action == "route":
            value = args.get("value", "")
            try:
                agent = self._router.route(value)
            except (KeyError, TypeError) as exc:
                raise ValueError(str(exc)) from exc
            # Infer the chosen key for display (may not match agent.name).
            matched_key = None
            for key, candidate in self._router.routes.items():
                if candidate is agent:
                    matched_key = key
                    break
            return {
                "matched_key": matched_key,
                "agent_name": getattr(agent, "name", None),
                "panel_id": f"agent-{agent.id}" if hasattr(agent, "id") else None,
            }

        if action == "route_and_run":
            value = args.get("value", "")
            prompt = args.get("prompt", "")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("'prompt' is required")
            agent = self._router.route(value)
            resp = agent.chat(prompt)
            usage = getattr(resp, "usage", None)
            return {
                "agent_name": getattr(agent, "name", None),
                "content": getattr(resp, "content", "") or "",
                "usage": {
                    "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                    "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
                    "cost_usd": getattr(usage, "cost_usd", None) if usage else None,
                },
            }

        return super().handle_action(action, args)
