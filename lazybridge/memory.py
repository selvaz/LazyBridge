"""Memory — conversation history management for per-agent and shared use."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
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

    When compression triggers the summarizer is called with a formatted
    transcript of the turns being dropped.  Three callable shapes are
    handled transparently:

    * Sync callable returning a string / Envelope / anything with
      ``.text()`` — called directly.
    * ``Agent`` — its ``__call__`` already bridges async internally.
    * Plain ``async def summarize(prompt): ...`` — the returned
      coroutine is driven to completion (in a worker thread when
      called from inside an event loop, to avoid nested-loop errors).

    If the summarizer raises, compression falls back to keyword
    extraction — never silent garbage.
    """

    #: Hard cap on ``_turns`` when compression is disabled.  Without a
    #: cap ``strategy="none"`` (or ``max_tokens=None``) leaks memory
    #: linearly with conversation length.  The ceiling is large enough
    #: that typical interactive sessions never hit it, but prevents
    #: long-running agents from OOMing silently.  Callers who genuinely
    #: want unbounded history can pass ``max_turns=None`` explicitly.
    _DEFAULT_MAX_TURNS = 1000

    #: Default deadline applied to LLM-summariser calls when the
    #: summariser returns a coroutine / awaitable.  A hung summariser
    #: must never starve the agent; on timeout the keyword-extraction
    #: fallback runs instead, never silent loss.  ``None`` disables the
    #: deadline (legacy behaviour).
    _DEFAULT_SUMMARIZER_TIMEOUT = 30.0

    def __init__(
        self,
        *,
        strategy: Literal["auto", "sliding", "summary", "none"] = "auto",
        max_tokens: int | None = 4000,
        max_turns: int | None = _DEFAULT_MAX_TURNS,
        store: Any | None = None,
        summarizer: Any | None = None,
        summarizer_timeout: float | None = _DEFAULT_SUMMARIZER_TIMEOUT,
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
        if summarizer_timeout is not None and summarizer_timeout <= 0:
            raise ValueError(f"summarizer_timeout must be > 0 or None, got {summarizer_timeout!r}")
        self.summarizer_timeout = summarizer_timeout
        # Guard against overlapping summariser calls when ``add()`` is
        # invoked concurrently.  Only one compression runs at a time;
        # other ``add()``s append turns and skip the compression branch.
        self._compressing = False
        # Separate one-shot warning flags for distinct warning sources so
        # a summariser timeout doesn't silence the turn-cap warning and
        # vice versa.
        self._summarizer_warned = False

    def add(self, user: str, assistant: str, *, tokens: int = 0) -> None:
        # Phase 1 — append + decide whether to compress, under the lock.
        with self._lock:
            self._turns.append(_Turn(user=user, assistant=assistant, token_estimate=tokens))
            plan = self._plan_compression()

        # Phase 2 — compute the summary OUTSIDE the lock so a slow or
        # hung summariser (LLM call, network) can't deadlock concurrent
        # ``add()`` callers waiting on the same Memory instance.  The
        # ``_compressing`` flag taken in ``_plan_compression`` is cleared
        # in ``finally`` so an exception path can't leave it stuck.
        if plan is not None:
            head_turns, drop_count = plan
            try:
                if self._summarizer is not None:
                    new_summary = self._llm_summary(head_turns)
                else:
                    new_summary = self._rule_summary(head_turns)
                with self._lock:
                    self._summary = new_summary
                    # Drop the first ``drop_count`` turns; any turns
                    # appended concurrently during summarisation remain
                    # in place after the cut.
                    self._turns = self._turns[drop_count:]
            finally:
                with self._lock:
                    self._compressing = False

        # Phase 3 — enforce the hard turn cap last so it can never be
        # bypassed by a long-running summariser.
        with self._lock:
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

    def _plan_compression(self) -> tuple[list[_Turn], int] | None:
        """Decide whether to compress and snapshot the head turns.

        Caller MUST hold ``self._lock``.  Returns ``(head_copy,
        drop_count)`` when compression should run, or ``None`` when no
        action is needed (already compressing / strategy disabled / not
        enough turns).  Marks ``self._compressing`` so a second
        concurrent ``add()`` skips the work.
        """
        if self._compressing:
            return None
        if self.strategy == "none":
            return None
        # "auto" needs a token budget to know when to compress; "sliding"
        # and "summary" compress by turn count so they don't need max_tokens.
        if self.strategy == "auto" and not self.max_tokens:
            return None
        total = sum(t.token_estimate for t in self._turns)
        # "summary" compresses like "sliding" but uses LLM summarization.
        # "sliding" always compresses when turns > window.
        # "auto"    compresses only once token budget is exceeded.
        should_compress = self.strategy in ("sliding", "summary") or (
            self.strategy == "auto" and total > self.max_tokens
        )
        if not (should_compress and len(self._turns) > 10):
            return None
        head = list(self._turns[:-10])  # snapshot; safe to summarise outside lock
        self._compressing = True
        return head, len(head)

    def _llm_summary(self, turns: list[_Turn]) -> str:
        """Call the LLM summarizer on ``turns``; fall back to _rule_summary on error.

        If ``summarizer`` is an async callable (returns a coroutine),
        the coroutine is driven to completion here rather than left
        dangling — without this the result is silently stringified as
        ``"<coroutine object at 0x…>"`` and the Python runtime emits a
        "coroutine was never awaited" RuntimeWarning.

        ``summarizer_timeout`` (set on the Memory) is enforced when the
        summariser returns a coroutine / awaitable — on deadline the
        keyword-extraction fallback runs.  Sync summarisers cannot be
        cancelled mid-call, so the timeout is advisory there: the
        caller must enforce it inside their own callable if needed.
        """
        assert self._summarizer is not None
        lines: list[str] = []
        for t in turns:
            lines.append(f"User: {t.user}")
            lines.append(f"Assistant: {t.assistant}")
        prompt = (
            "Write a concise summary of the following conversation. "
            "Preserve key facts, decisions, and outcomes in 2-4 sentences:\n\n" + "\n".join(lines)
        )
        try:
            result = self._summarizer(prompt)
            if inspect.iscoroutine(result) or inspect.isawaitable(result):
                result = _drive_to_completion(result, timeout=self.summarizer_timeout)
            return result.text() if hasattr(result, "text") else str(result)
        except TimeoutError:
            # Hung judge / network stall — fall back to keywords rather
            # than letting the agent run sit on a stuck future.  The
            # warning is one-shot per Memory (via _summarizer_warned, a
            # separate flag from _overflow_warned) so a degenerate
            # summariser is visible without flooding logs.
            if not self._summarizer_warned:
                self._summarizer_warned = True
                import warnings

                warnings.warn(
                    f"Memory summariser exceeded summarizer_timeout="
                    f"{self.summarizer_timeout}s; falling back to keyword "
                    f"extraction.  Raise summarizer_timeout= or pick a "
                    f"faster summariser to silence.",
                    UserWarning,
                    stacklevel=4,
                )
            return self._rule_summary(turns)
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


def _drive_to_completion(awaitable: Any, *, timeout: float | None = None) -> Any:
    """Drive an awaitable from sync context regardless of loop state.

    * No loop running → ``asyncio.run(...)``.
    * Loop running (Jupyter, FastAPI, inside an Agent's async path) →
      dispatch to a fresh loop on a worker thread so we don't nest.

    Mirrors the bridge used by :meth:`Agent.__call__` so Memory's
    compression path has identical async semantics.

    When ``timeout`` is set the awaitable is wrapped in
    ``asyncio.wait_for``; on deadline a :class:`TimeoutError` propagates
    so the caller can fall back gracefully.
    """

    async def _run() -> Any:
        coro = _ensure_coroutine(awaitable)
        if timeout is None:
            return await coro
        return await asyncio.wait_for(coro, timeout=timeout)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _run()).result()


async def _ensure_coroutine(awaitable: Any) -> Any:
    """Wrap any awaitable in a coroutine so ``asyncio.run`` accepts it."""
    return await awaitable
