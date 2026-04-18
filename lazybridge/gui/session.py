"""SessionPanel — read-only session overview.

Lists the session's registered agents and current store keys.  The panel
is intentionally minimal for now: deeper session inspection (event log,
graph, store editing) is planned but lives in follow-up commits.

Use the sidebar to navigate from here to an agent's own panel.
"""

from __future__ import annotations

from typing import Any

from lazybridge.gui._panel import Panel


class SessionPanel(Panel):
    """Panel for a :class:`~lazybridge.lazy_session.LazySession` instance."""

    kind = "session"

    def __init__(self, session: Any) -> None:
        self._session = session

    @property
    def id(self) -> str:
        return f"session-{self._session.id}"

    @property
    def label(self) -> str:
        n = len(list(getattr(self._session, "_agents", []) or []))
        return f"session · {n} agent(s)"

    def render_state(self) -> dict[str, Any]:
        sess = self._session
        tracking = getattr(sess, "tracking", None)
        agents = []
        for a in list(getattr(sess, "_agents", []) or []):
            agents.append(
                {
                    "id": f"agent-{a.id}",
                    "name": a.name,
                    "provider": getattr(a, "_provider_name", "?"),
                    "model": getattr(a, "_model_name", "?"),
                }
            )
        try:
            store_keys = list(getattr(sess.store, "keys", lambda: [])())
        except Exception:
            store_keys = []
        return {
            "id": sess.id,
            "tracking": getattr(tracking, "name", str(tracking)) if tracking is not None else "default",
            "agents": agents,
            "store_keys": store_keys,
        }
