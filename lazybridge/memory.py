"""Memory — conversation history management for per-agent and shared use."""

from __future__ import annotations

import threading
from dataclasses import dataclass
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

    LLM summarization
    -----------------
    Pass any callable (typically an ``Agent``) as ``summarizer=`` to enable
    LLM-based compression instead of the keyword-extraction fallback::

        summarizer = Agent("claude-haiku-4-5-20251001", system="Summarize conversations concisely.")
        memory = Memory(strategy="summary", summarizer=summarizer)
        agent  = Agent("claude-opus-4-7", memory=memory)

    When compression triggers the summarizer is called synchronously with a
    formatted transcript of the turns being dropped.  ``Agent.__call__``
    handles the async bridge automatically so no special wrapping is needed.
    If the summarizer raises, compression falls back to keyword extraction.
    """

    #: Hard cap on ``_turns`` when compression is disabled.  Without a
    #: cap ``strategy="none"`` (or ``max_tokens=None``) leaks memory
    #: linearly with conversation length.  The ceiling is large enough
    #: that typical interactive sessions never hit it, but prevents
    #: long-running agents from OOMing silently.  Callers who genuinely
    #: want unbounded history can pass ``max_turns=None`` explicitly.
    _DEFAULT_MAX_TURNS = 1000

    def __init__(
        self,
        *,
        strategy: Literal["auto", "sliding", "summary", "none"] = "auto",
        max_tokens: int | None = 4000,
        max_turns: int | None = _DEFAULT_MAX_TURNS,
        store: Any | None = None,
        summarizer: Any | None = None,
    ) -> None:
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.store = store
        self._turns: list[_Turn] = []
        self._lock = threading.Lock()
        self._summary: str = ""
        self._overflow_warned = False
        self._summarizer = summarizer

    def add(self, user: str, assistant: str, *, tokens: int = 0) -> None:
        with self._lock:
            self._turns.append(_Turn(user=user, assistant=assistant, token_estimate=tokens))
            self._maybe_compress()
            self._enforce_turn_cap()

    def _enforce_turn_cap(self) -> None:
        """Hard cap on total retained turns — unconditional backstop.

        Runs after ``_maybe_compress`` so strategy-specific compression
        wins when it applies.  Only fires when the caller has opted out
        of token-based compression (``strategy="none"`` or
        ``max_tokens`` is falsy) AND ``max_turns`` is set.  Drops oldest
        turns FIFO and emits a one-shot warning.
        """
        if self.max_turns is None:
            return
        if len(self._turns) <= self.max_turns:
            return
        drop = len(self._turns) - self.max_turns
        self._turns = self._turns[drop:]
        if not self._overflow_warned:
            import warnings

            warnings.warn(
                f"Memory turn count exceeded max_turns={self.max_turns}; "
                f"dropped {drop} oldest turn(s).  Pass max_turns=None to "
                f"disable this cap, or set strategy='auto'/'sliding' for "
                f"token-aware compression.",
                UserWarning,
                stacklevel=3,
            )
            self._overflow_warned = True

    def _maybe_compress(self) -> None:
        if self.strategy == "none" or not self.max_tokens:
            return
        total = sum(t.token_estimate for t in self._turns)
        # "summary" compresses like "sliding" but uses LLM summarization.
        # "sliding" always compresses when turns > window.
        # "auto"    compresses only once token budget is exceeded.
        should_compress = (
            self.strategy in ("sliding", "summary")
            or (self.strategy == "auto" and total > self.max_tokens)
        )
        if should_compress and len(self._turns) > 10:
            old = self._turns[:-10]
            if self._summarizer is not None:
                self._summary = self._llm_summary(old)
            else:
                self._summary = self._rule_summary(old)
            self._turns = self._turns[-10:]

    def _llm_summary(self, turns: list[_Turn]) -> str:
        """Call the LLM summarizer on ``turns``; fall back to _rule_summary on error."""
        assert self._summarizer is not None
        lines: list[str] = []
        for t in turns:
            lines.append(f"User: {t.user}")
            lines.append(f"Assistant: {t.assistant}")
        prompt = (
            "Write a concise summary of the following conversation. "
            "Preserve key facts, decisions, and outcomes in 2-4 sentences:\n\n"
            + "\n".join(lines)
        )
        try:
            result = self._summarizer(prompt)
            return result.text() if hasattr(result, "text") else str(result)
        except Exception:
            return self._rule_summary(turns)

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
