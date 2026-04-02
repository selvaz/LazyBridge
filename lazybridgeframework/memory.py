"""Memory — stateful conversation history for LazyAgent.chat().

Pass a Memory instance to chat() / achat() to accumulate turns automatically
without managing a message list manually::

    mem = Memory()
    ai.chat("ciao, mi chiamo Marco", memory=mem)
    ai.chat("qual è il mio nome?", memory=mem)   # ricorda "Marco"

The same Memory object can be shared across multiple agents::

    mem = Memory()
    agent_a.chat("ricorda questo", memory=mem)
    agent_b.chat("cosa devo ricordare?", memory=mem)

For cross-session persistence, serialise via memory.history::

    import json
    sess.store.write("history", json.dumps(mem.history))
    # --- next session ---
    mem = Memory()
    mem._messages = json.loads(sess.store.read("history") or "[]")
"""

from __future__ import annotations

import threading


class Memory:
    """Accumulates conversation turns for stateful chat.

    Pass to ``chat()`` / ``achat()`` to avoid managing history manually.
    The same instance can be shared across multiple agents.
    """

    def __init__(self) -> None:
        self._messages: list[dict] = []
        self._lock = threading.Lock()

    @property
    def history(self) -> list[dict]:
        """Read-only copy of the accumulated message list."""
        with self._lock:
            return list(self._messages)

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)

    def __repr__(self) -> str:
        with self._lock:
            turns = len(self._messages) // 2
        return f"Memory(turns={turns})"

    def clear(self) -> None:
        """Reset conversation history."""
        with self._lock:
            self._messages.clear()

    # ------------------------------------------------------------------
    # Internal helpers — used by LazyAgent, not part of the public API
    # ------------------------------------------------------------------

    def _build_input(self, message: str) -> list[dict]:
        """Return full history + new user message without mutating state."""
        with self._lock:
            return self._messages + [{"role": "user", "content": message}]

    def _record(self, user_message: str, assistant_content: str) -> None:
        """Append a completed turn to history (called after a successful chat)."""
        with self._lock:
            self._messages.append({"role": "user",      "content": user_message})
            self._messages.append({"role": "assistant", "content": assistant_content})
