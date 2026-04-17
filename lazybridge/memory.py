"""Memory — stateful conversation history with smart compression.

Default behavior (``strategy="auto"``): under the token threshold, sends
everything raw. Over it, compresses older turns into a dense structured
block and keeps recent turns raw. The full history is always preserved
internally.

Quick start::

    mem = Memory()  # auto compression — just works
    ai.chat("ciao, mi chiamo Marco", memory=mem)
    ai.chat("qual è il mio nome?", memory=mem)   # ricorda "Marco"

Strategies::

    Memory()                              # auto — compresses when needed
    Memory(strategy="full")               # never compress (backward compat)
    Memory(strategy="rolling")            # sliding window + compression

With LLM compressor (more accurate)::

    compressor = LazyAgent("openai", model="gpt-4o-mini")
    mem = Memory(compressor=compressor)

The full raw history is always available via ``mem.history``.
"""

from __future__ import annotations

import threading
from typing import Any, Literal


class Memory:
    """Accumulates conversation turns with automatic context compression.

    Parameters
    ----------
    strategy:
        ``"auto"`` (default) — compresses when token estimate exceeds threshold.
        ``"full"`` — never compress (sends all turns, backward compatible).
        ``"rolling"`` — always use window + compression, regardless of size.
    max_context_tokens:
        Token budget estimate. When exceeded, older turns are compressed.
        Uses ``len(text) // 4`` approximation (no tiktoken dependency).
    window_turns:
        Number of recent turn pairs (user+assistant) to keep raw.
    compressor:
        Optional LazyAgent for LLM-powered compression. Uses a cheap model
        to extract entities, facts, decisions into a dense format.
        Default: rule-based extraction (free, no API call).
    """

    def __init__(
        self,
        *,
        strategy: Literal["full", "rolling", "auto"] = "auto",
        max_context_tokens: int = 4000,
        window_turns: int = 10,
        compressor: Any = None,
    ) -> None:
        self._messages: list[dict] = []
        self._lock = threading.Lock()
        self._strategy = strategy
        self._max_context_tokens = max_context_tokens
        self._window_size = window_turns * 2
        self._compressor = compressor
        self._compressed: str | None = None
        self._compressed_up_to: int = 0

    @classmethod
    def from_history(cls, messages: list[dict], **kwargs: Any) -> Memory:
        """Restore a Memory instance from a previously serialised history list."""
        instance = cls(**kwargs)
        instance._messages = list(messages)
        return instance

    @property
    def history(self) -> list[dict]:
        """Full raw history — never truncated, always complete."""
        with self._lock:
            return list(self._messages)

    @property
    def summary(self) -> str | None:
        """Current compressed memory block, if any."""
        with self._lock:
            return self._compressed

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)

    def __repr__(self) -> str:
        with self._lock:
            turns = len(self._messages) // 2
            compressed = "compressed" if self._compressed else "raw"
        return f"Memory(turns={turns}, {compressed})"

    def clear(self) -> None:
        """Reset conversation history and compression state."""
        with self._lock:
            self._messages.clear()
            self._compressed = None
            self._compressed_up_to = 0

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(messages: list[dict]) -> int:
        return sum(len(m.get("content", "")) for m in messages) // 4

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def _compress(self, turns: list[dict]) -> str:
        if self._compressor is not None:
            return self._llm_compress(turns)
        return self._simple_compress(turns)

    @staticmethod
    def _simple_compress(turns: list[dict]) -> str:
        n_turns = len(turns) // 2
        topics: set[str] = set()
        for t in turns:
            content = t.get("content", "")
            for word in content.split():
                if len(word) > 2 and word[0].isupper() and word.isalpha():
                    topics.add(word)
        topics_str = ", ".join(sorted(topics)[:20]) if topics else "general discussion"
        last_user = ""
        last_asst = ""
        for t in reversed(turns):
            if t.get("role") == "user" and not last_user:
                last_user = t.get("content", "")[:150]
            elif t.get("role") == "assistant" and not last_asst:
                last_asst = t.get("content", "")[:150]
            if last_user and last_asst:
                break
        lines = [f"[Memory — {n_turns} earlier turns]"]
        lines.append(f"Topics: {topics_str}")
        if last_user:
            lines.append(f"Last discussed: {last_user[:100]}")
        return "\n".join(lines)

    def _llm_compress(self, turns: list[dict]) -> str:
        sample = turns[-30:] if len(turns) > 30 else turns
        text = "\n".join(f"{t['role']}: {t['content'][:200]}" for t in sample)
        prompt = (
            "Extract key information from this conversation into a dense structured format.\n"
            "Include: entities (people, projects, tools), facts, decisions, preferences, open threads.\n"
            "Be extremely concise — use key:value format, not sentences.\n\n"
            f"Conversation ({len(turns) // 2} turns):\n{text}"
        )
        return self._compressor.text(prompt)

    # ------------------------------------------------------------------
    # Monitoring — called after each _record
    # ------------------------------------------------------------------

    def _maybe_recompress(self) -> None:
        total = len(self._messages)
        if total <= self._window_size:
            return
        older = self._messages[: -self._window_size]
        if len(older) <= self._compressed_up_to:
            return
        if self._strategy == "auto":
            est = self._estimate_tokens(self._messages)
            if est <= self._max_context_tokens:
                return
        self._compressed = self._compress(older)
        self._compressed_up_to = len(older)

    # ------------------------------------------------------------------
    # Internal helpers — used by LazyAgent
    # ------------------------------------------------------------------

    def _build_input(self, message: str) -> list[dict]:
        """Build input for provider: [compressed] + [window] + [new message]."""
        with self._lock:
            if self._strategy == "full":
                return self._messages + [{"role": "user", "content": message}]

            total = len(self._messages)

            if total <= self._window_size:
                return self._messages + [{"role": "user", "content": message}]

            window = self._messages[-self._window_size :]
            older = self._messages[: -self._window_size]

            if len(older) > self._compressed_up_to:
                self._compressed = self._compress(older)
                self._compressed_up_to = len(older)

            result: list[dict] = []
            if self._compressed:
                result.append({"role": "system", "content": self._compressed})
            result.extend(window)
            result.append({"role": "user", "content": message})
            return result

    def _record(self, user_message: str, assistant_content: str) -> None:
        """Append a completed turn and monitor for recompression."""
        with self._lock:
            self._messages.append({"role": "user", "content": user_message})
            self._messages.append({"role": "assistant", "content": assistant_content})
            if self._strategy in ("auto", "rolling"):
                self._maybe_recompress()
