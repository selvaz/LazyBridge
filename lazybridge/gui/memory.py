"""MemoryPanel — GUI for ``Memory``.

Inspect tab shows the conversation history, the current compressed block
(if any), and a token-budget meter using the same estimator the Memory
itself uses (``len(content) // 4``).

Test/edit tab lets the user:

- Clear the memory (``memory.clear()``).
- Force a recompression of the current history by calling the internal
  ``_maybe_recompress`` (works for the ``auto`` / ``rolling`` strategies).
- Export the raw history as JSON so it can be persisted and later
  restored via :meth:`Memory.from_history`.
"""

from __future__ import annotations

from typing import Any

from lazybridge.gui._panel import Panel


class MemoryPanel(Panel):
    """Panel for a :class:`~lazybridge.memory.Memory` instance."""

    kind = "memory"

    def __init__(self, memory: Any, *, label: str | None = None) -> None:
        self._memory = memory
        self._label_override = label

    @property
    def id(self) -> str:
        return f"memory-{id(self._memory):x}"

    @property
    def label(self) -> str:
        if self._label_override:
            return self._label_override
        turns = len(self._memory) // 2
        compressed = "compressed" if self._memory.summary else "raw"
        return f"memory · {turns} turn(s) · {compressed}"

    # ------------------------------------------------------------------

    def _preview(self, content: str) -> str:
        if content is None:
            return ""
        return content if len(content) <= 600 else content[:600] + "…"

    def _history_entries(self) -> list[dict[str, Any]]:
        history = self._memory.history
        return [
            {
                "role": str(m.get("role", "unknown")),
                "preview": self._preview(str(m.get("content", ""))),
                "full_length": len(str(m.get("content", ""))),
            }
            for m in history
        ]

    def render_state(self) -> dict[str, Any]:
        mem = self._memory
        history = mem.history
        summary = mem.summary
        strategy = getattr(mem, "_strategy", "auto")
        budget = int(getattr(mem, "_max_context_tokens", 0))
        token_estimate = sum(len(str(m.get("content", ""))) for m in history) // 4
        return {
            "strategy": strategy,
            "max_context_tokens": budget,
            "token_estimate": token_estimate,
            "turn_count": len(history) // 2,
            "message_count": len(history),
            "summary": summary,
            "history": self._history_entries(),
            "window_size": getattr(mem, "_window_size", None),
            "has_compressor": getattr(mem, "_compressor", None) is not None,
        }

    # ------------------------------------------------------------------

    def handle_action(self, action: str, args: dict[str, Any]) -> dict[str, Any]:
        if action == "clear":
            self._memory.clear()
            return {"ok": True}

        if action == "force_compress":
            # Memory._maybe_recompress is the private hook; call through
            # only if it exists (it does today, but be defensive).
            # Since Wave 3's M2 fix, _maybe_recompress manages the lock
            # internally (snapshot-under-lock, compress-outside,
            # publish-under-lock), so callers must NOT hold the lock.
            fn = getattr(self._memory, "_maybe_recompress", None)
            if not callable(fn):
                raise ValueError("This Memory instance does not support manual recompression.")
            fn()
            return {"ok": True, "summary": self._memory.summary}

        if action == "export_history":
            return {"history": list(self._memory.history)}

        return super().handle_action(action, args)
