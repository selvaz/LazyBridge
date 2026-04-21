"""Memory — conversation history management for per-agent and shared use."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Literal

from lazybridge.core.types import Message, Role


@dataclass
class _Turn:
    user: str
    assistant: str
    token_estimate: int = 0


class Memory:
    """Conversation memory with configurable compression strategy.

    Per-agent use:  memory=Memory() on the Agent — tracks its message history.
    Shared use:     sources=[memory] on multiple Agents — live view of shared text.

    The ``text()`` method returns the current memory as a context string,
    re-read on every invocation (live view — never a stale snapshot).
    """

    def __init__(
        self,
        *,
        strategy: Literal["auto", "sliding", "summary", "none"] = "auto",
        max_tokens: int | None = 4000,
        store: Any | None = None,
    ) -> None:
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.store = store
        self._turns: list[_Turn] = []
        self._lock = threading.Lock()
        self._summary: str = ""

    def add(self, user: str, assistant: str, *, tokens: int = 0) -> None:
        with self._lock:
            self._turns.append(_Turn(user=user, assistant=assistant, token_estimate=tokens))
            self._maybe_compress()

    def _maybe_compress(self) -> None:
        if self.strategy == "none" or not self.max_tokens:
            return
        total = sum(t.token_estimate for t in self._turns)
        if self.strategy == "sliding" or (self.strategy == "auto" and total > self.max_tokens):
            if len(self._turns) > 10:
                old = self._turns[:-10]
                self._summary = self._rule_summary(old)
                self._turns = self._turns[-10:]

    def _rule_summary(self, turns: list[_Turn]) -> str:
        topics: set[str] = set()
        for t in turns:
            words = (t.user + " " + t.assistant).split()
            topics.update(w.lower() for w in words if len(w) > 5)
        return f"[Earlier conversation covered: {', '.join(sorted(topics)[:20])}]"

    def messages(self) -> list[Message]:
        """Return full message list including summary prefix if compressed."""
        with self._lock:
            result: list[Message] = []
            if self._summary:
                result.append(Message(role=Role.USER, content=f"Context from earlier: {self._summary}"))
                result.append(Message(role=Role.ASSISTANT, content="Understood."))
            for t in self._turns:
                result.append(Message(role=Role.USER, content=t.user))
                result.append(Message(role=Role.ASSISTANT, content=t.assistant))
            return result

    def text(self) -> str:
        """Return current memory as a plain-text string (live view)."""
        with self._lock:
            parts: list[str] = []
            if self._summary:
                parts.append(self._summary)
            for t in self._turns[-5:]:
                parts.append(f"User: {t.user}\nAssistant: {t.assistant}")
            return "\n\n".join(parts)

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()
            self._summary = ""
