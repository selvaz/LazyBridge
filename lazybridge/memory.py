"""Memory — conversation history management for per-agent and shared use."""

from __future__ import annotations

import inspect
import math
import re
import threading
import warnings
from dataclasses import dataclass
from typing import Any, Literal

from lazybridge._asyncbridge import run_coroutine_blocking
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
    LLM-based compression instead of the sentence-extraction fallback::

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

    If the summarizer raises, compression falls back to complete-sentence
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
    #: must never starve the agent; on timeout the sentence-extraction
    #: fallback runs instead, never silent loss.  ``None`` disables the
    #: deadline (legacy behaviour).
    _DEFAULT_SUMMARIZER_TIMEOUT = 30.0

    #: Recent turns are never summarized.  Besides avoiding a summary-only
    #: context, this verbatim window preserves exact identifiers and facts.
    _RECENT_TURNS = 10
    _BUDGET_TARGET = 0.60
    _SUMMARY_SHARE = 0.30

    def __init__(
        self,
        *,
        strategy: Literal["auto", "sliding", "summary", "none"] = "auto",
        max_tokens: int | None = None,
        max_turns: int | None = _DEFAULT_MAX_TURNS,
        store: Any | None = None,
        key: str | None = None,
        session_key: str = "default",
        summarizer: Any | None = None,
        summarizer_timeout: float | None = _DEFAULT_SUMMARIZER_TIMEOUT,
    ) -> None:
        """Create a Memory instance.

        Parameters
        ----------
        strategy:
            ``"auto"``    — compress when token budget is exceeded
            (requires ``max_tokens`` to be set).
            ``"sliding"`` — compress by turn count regardless of
            ``max_tokens``; does NOT require a token budget.
            ``"summary"`` — compress when the token budget is exceeded,
            using the LLM ``summarizer`` instead of extraction.
            ``"none"``    — no compression; only the hard ``max_turns``
            cap applies.
        max_tokens:
            Optional approximate token budget for remembered text. Size is
            estimated as characters / 4 and is independent of provider
            billing/cache counters. When omitted, the legacy count rules
            apply: ``"summary"`` compacts after ten turns and ``"auto"``
            does not compact.
        max_turns:
            Hard cap on retained turns after compression runs.  ``None``
            disables the cap (unbounded history — only safe for short
            sessions).  Default: :attr:`_DEFAULT_MAX_TURNS` (1000).
        store:
            Optional :class:`~lazybridge.store.Store` for persistent
            memory across process restarts.  Existing turns/summary
            under ``key`` (if any) are loaded during ``__init__``;
            every subsequent ``add()``, ``amend_last()``, and
            ``clear()`` writes the full current state back via
            :meth:`Store.write_memory`.  Requires ``key=``.
        key:
            Stable identifier (typically the owning agent's ``name``)
            used as ``agent_id`` in :meth:`Store.write_memory` /
            :meth:`Store.read_memory`.  Required when ``store`` is set —
            without it, persisted state can't be found again after a
            restart.
        session_key:
            Secondary identifier for distinguishing multiple
            conversations for the same ``key`` (e.g. per user, per
            thread).  Only meaningful together with ``store``.  Default
            ``"default"``.
        summarizer:
            Callable used by ``strategy="summary"`` — typically an
            :class:`~lazybridge.agent.Agent`.  See the class docstring
            for accepted shapes.
        summarizer_timeout:
            Deadline (seconds) applied to async summariser calls.  On
            timeout the sentence-extraction fallback runs and a one-shot
            warning is emitted via :attr:`_summarizer_warned`.  ``None``
            disables the deadline.  Default: 30 s.
        """
        if store is not None and key is None:
            raise ValueError(
                "Memory(store=...) requires key= — a stable identifier (e.g. the "
                "owning agent's name) so persisted turns can be found again after "
                "a restart. Pass key=<agent_name>."
            )
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.store = store
        self.key = key
        self.session_key = session_key
        self._turns: list[_Turn] = []
        self._lock = threading.Lock()
        self._summary: str = ""
        self._overflow_warned = False
        self._summarizer = summarizer
        if summarizer_timeout is not None and summarizer_timeout <= 0:
            raise ValueError(f"summarizer_timeout must be > 0 or None, got {summarizer_timeout!r}")
        # Phase-2 Block C: warn on too-tight timeouts.  The summariser
        # spawns a worker thread + JSON serialisation + LLM round-trip;
        # values under 5 s typically time out on every call (silent
        # always-fail trap).
        if summarizer_timeout is not None and summarizer_timeout < 5.0:
            import warnings as _warnings

            _warnings.warn(
                f"Memory(summarizer_timeout={summarizer_timeout!r}): values under 5.0 s often "
                f"trigger a timeout on every summariser invocation (worker-thread setup + LLM "
                f"round-trip).  Consider 30.0 s (the production default) unless you have data "
                f"showing your summariser is reliably faster.",
                UserWarning,
                stacklevel=2,
            )
        self.summarizer_timeout = summarizer_timeout
        # Guard against overlapping summariser calls when ``add()`` is
        # invoked concurrently.  Only one compression runs at a time;
        # other ``add()``s append turns and skip the compression branch.
        self._compressing = False
        # Separate one-shot warning flags for distinct warning sources so
        # a summariser timeout doesn't silence the turn-cap warning and
        # vice versa.
        self._summarizer_warned = False
        if self.store is not None:
            self._load_from_store()

    def _load_from_store(self) -> None:
        """Reload turns/summary previously persisted under (key, session_key).

        Called once from ``__init__``, only when ``store`` was passed.  No-op
        (fresh Memory) when nothing was ever written for this identity — a
        first run and a restart after an empty history look identical.
        """
        assert self.store is not None
        loaded = self.store.read_memory(self.key, session_key=self.session_key)
        if loaded is None:
            return
        self._turns = [
            _Turn(
                user=t["user"],
                assistant=t["assistant"],
                token_estimate=_estimate_turn_tokens(t["user"], t["assistant"]),
            )
            for t in loaded["turns"]
        ]
        self._summary = loaded["summary"]

    def _persist_locked(self) -> None:
        """Write the current state to the Store. Caller MUST hold ``self._lock``.

        Persisting *inside* the same critical section that made the mutation
        (rather than snapshotting under the lock and writing after releasing
        it) is what keeps concurrent ``add()``/``amend_last()``/``clear()``
        calls persisting in the same order they mutated in. With the
        snapshot-then-release-then-write shape, two concurrent mutations
        could race their Store writes independently of their mutation
        order — a slower write from an earlier mutation completing after a
        faster write from a later one silently overwrites the durable row
        with stale data (turns lost, or a clear() undone, on the next
        restart). A local SQLite upsert (WAL mode, no network) is fast
        enough that holding the lock across it is a non-issue — unlike the
        LLM summariser call in ``add()``'s Phase 2, which deliberately runs
        outside the lock because it can take seconds.
        """
        if self.store is None:
            return
        turns_snapshot = [
            {"user": t.user, "assistant": t.assistant, "token_estimate": t.token_estimate} for t in self._turns
        ]
        self.store.write_memory(
            self.key,
            turns=turns_snapshot,
            summary=self._summary,
            session_key=self.session_key,
        )

    def add(self, user: str, assistant: str, *, tokens: int | None = None) -> None:
        if tokens is not None:
            warnings.warn(
                "Memory.add(tokens=...) is deprecated and ignored; memory size is estimated from stored text",
                DeprecationWarning,
                stacklevel=2,
            )
        text_tokens = _estimate_turn_tokens(user, assistant)
        # Phase 1 — append + decide whether to compress, under the lock.
        with self._lock:
            self._turns.append(_Turn(user=user, assistant=assistant, token_estimate=text_tokens))
            plan = self._plan_compression()

        # Phase 2 — compute the summary OUTSIDE the lock so a slow or
        # hung summariser (LLM call, network) can't deadlock concurrent
        # ``add()`` callers waiting on the same Memory instance.  The
        # ``_compressing`` flag taken in ``_plan_compression`` is cleared
        # in ``finally`` so an exception path can't leave it stuck.
        if plan is not None:
            head_turns, drop_count = plan
            # Fold the previous summary into the new one — replacing it
            # outright would permanently discard everything the earlier
            # compression captured (the oldest context) with no warning.
            # The ``_compressing`` flag guarantees no concurrent writer
            # to ``_summary``, but read it under the lock for visibility.
            with self._lock:
                prior_summary = self._summary
                tail_over_target = self.max_tokens is not None and sum(
                    turn.token_estimate for turn in self._turns[drop_count:]
                ) > int(self.max_tokens * self._BUDGET_TARGET)
            try:
                # When the mandatory two-turn verbatim tail already fills
                # the target, calling an LLM cannot create any headroom and
                # would repeat on every long turn. Preserve the evicted head
                # with the bounded local extractor instead.
                if self._summarizer is not None and not tail_over_target:
                    new_summary = self._llm_summary(head_turns, prior=prior_summary)
                else:
                    new_summary = self._rule_summary(head_turns, prior=prior_summary)
                with self._lock:
                    self._summary = self._cap_summary(new_summary)
                    # Drop the first ``drop_count`` turns; any turns
                    # appended concurrently during summarisation remain
                    # in place after the cut.
                    self._turns = self._turns[drop_count:]
            finally:
                with self._lock:
                    self._compressing = False

        # Phase 3 — enforce the hard turn cap last so it can never be
        # bypassed by a long-running summariser, then persist the settled
        # state in the SAME critical section — see _persist_locked for why
        # that matters for write ordering under concurrent add() calls.
        with self._lock:
            self._enforce_turn_cap()
            self._persist_locked()

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

        Caller MUST hold ``self._lock`` while invoking this method *and*
        while reading the returned snapshot.  The snapshot itself
        (``head_copy``) is safe to summarise outside the lock once
        ``self._compressing`` has been set to ``True`` — that flag, not
        the lock, is the synchronisation primitive that prevents a
        second concurrent ``add()`` from re-entering the compression
        path.  Returns ``(head_copy, drop_count)`` when compression
        should run, or ``None`` when no action is needed (already
        compressing / strategy disabled / not enough turns).
        """
        if self._compressing:
            return None
        if self.strategy == "none":
            return None
        if self.strategy == "auto" and self.max_tokens is None:
            return None
        total = _estimate_text_tokens(self._summary) + sum(t.token_estimate for t in self._turns)
        # Sliding retains its count-based compatibility behaviour.  Auto and
        # summary use the stored text budget.  All strategies keep a ten-turn
        # verbatim window, so the turn threshold is only a secondary bound.
        legacy_summary = self.strategy == "summary" and self.max_tokens is None
        should_compress = (
            self.strategy == "sliding"
            or legacy_summary
            or (self.strategy in ("auto", "summary") and self.max_tokens is not None and total > self.max_tokens)
        )
        minimum_turns = self._RECENT_TURNS if self.max_tokens is None else 2
        if not (should_compress and len(self._turns) > minimum_turns):
            return None
        keep = self._RECENT_TURNS
        if self.max_tokens is not None:
            # Leave enough headroom that the capped summary plus verbatim
            # tail is at most the hysteresis target. A minimum of two turns
            # keeps the immediate exchange intact even for a tiny budget.
            tail_budget = max(0, int(self.max_tokens * (self._BUDGET_TARGET - self._SUMMARY_SHARE)))
            keep = 0
            used = 0
            for turn in reversed(self._turns):
                if keep >= 2 and used + turn.token_estimate > tail_budget:
                    break
                used += turn.token_estimate
                keep += 1
            keep = max(2, min(keep, len(self._turns) - 1))
        head = list(self._turns[:-keep])  # snapshot; safe to summarise outside lock
        self._compressing = True
        return head, len(head)

    def _cap_summary(self, summary: str) -> str:
        """Bound summaries to their budget share, preferring sentence cuts."""
        if self.max_tokens is None:
            return summary
        max_chars = max(1, int(self.max_tokens * self._SUMMARY_SHARE * 4))
        return _truncate_summary(summary, max_chars=max_chars)

    def _llm_summary(self, turns: list[_Turn], *, prior: str = "") -> str:
        """Call the LLM summarizer on ``turns``; fall back to _rule_summary on error.

        ``prior`` is the summary produced by the previous compression (if
        any).  It is included in the material to summarise so repeated
        compressions ACCUMULATE context instead of silently replacing the
        oldest history.

        If ``summarizer`` is an async callable (returns a coroutine),
        the coroutine is driven to completion here rather than left
        dangling — without this the result is silently stringified as
        ``"<coroutine object at 0x…>"`` and the Python runtime emits a
        "coroutine was never awaited" RuntimeWarning.

        ``summarizer_timeout`` (set on the Memory) is enforced when the
        summariser returns a coroutine / awaitable — the coroutine is
        wrapped in ``asyncio.wait_for``; on deadline the
        sentence-extraction fallback runs and :attr:`_summarizer_warned`
        is set (one-shot warning, separate from the turn-cap
        :attr:`_overflow_warned` flag).  Sync summarisers are called
        directly and cannot be interrupted mid-call; callers that need
        a hard deadline for sync summarisers should enforce it inside
        their own callable.
        """
        assert self._summarizer is not None
        lines: list[str] = []
        if prior:
            lines.append(f"Summary of even earlier conversation (fold this in): {prior}")
            lines.append("")
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
            # Hung judge / network stall — fall back to sentences rather
            # than letting the agent run sit on a stuck future.  The
            # warning is one-shot per Memory (via _summarizer_warned, a
            # separate flag from _overflow_warned) so a degenerate
            # summariser is visible without flooding logs.
            if not self._summarizer_warned:
                self._summarizer_warned = True
                import warnings

                warnings.warn(
                    f"Memory summariser exceeded summarizer_timeout="
                    f"{self.summarizer_timeout}s; falling back to sentence "
                    f"extraction.  Raise summarizer_timeout= or pick a "
                    f"faster summariser to silence.",
                    UserWarning,
                    stacklevel=4,
                )
            return self._rule_summary(turns, prior=prior)
        except Exception:
            return self._rule_summary(turns, prior=prior)

    def _rule_summary(self, turns: list[_Turn], *, prior: str = "") -> str:
        """Extract complete sentences rather than a lossy keyword bag."""
        sentences: list[str] = []
        if prior:
            previous = prior.removeprefix("[Earlier conversation covered: ").removesuffix("]")
            sentences.extend(_sentences(previous))
        for turn in turns:
            sentences.extend(_sentences(turn.user))
            sentences.extend(_sentences(turn.assistant))

        # Preserve order and exact spelling (notably named entities), while
        # bounding repeated fallback summaries.
        unique = list(dict.fromkeys(sentence for sentence in sentences if sentence))
        summary = f"[Earlier conversation covered: {' '.join(unique[-20:])}]"
        return self._cap_summary(summary)

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

    def amend_last(self, assistant: str) -> None:
        """Replace the assistant half of the most recent turn.

        Used by ``Agent``'s structured-output retry loop: the engine
        records the *first* response when it completes, but a correction
        retry may produce the answer actually returned to the caller.
        Amending keeps memory history consistent with the returned
        Envelope instead of preserving the rejected draft.  No-op when
        no turns are recorded (e.g. the last turn was already compressed
        away).
        """
        with self._lock:
            if self._turns:
                last = self._turns[-1]
                self._turns[-1] = _Turn(
                    user=last.user,
                    assistant=assistant,
                    token_estimate=_estimate_turn_tokens(last.user, assistant),
                )
            self._persist_locked()

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()
            self._summary = ""
            # Persist the wipe too — otherwise a restart right after clear()
            # would reload the pre-clear state from the Store.
            self._persist_locked()


def _estimate_text_tokens(text: str) -> int:
    """Conservative tokenizer-free estimate for the text Memory stores."""
    return math.ceil(len(text) / 4)


def _estimate_turn_tokens(user: str, assistant: str) -> int:
    return _estimate_text_tokens(f"User: {user}\nAssistant: {assistant}")


def _truncate_summary(summary: str, *, max_chars: int) -> str:
    """Truncate at a sentence boundary when possible, with a hard cap."""
    if len(summary) <= max_chars:
        return summary
    prefix = "[Earlier conversation covered: "
    suffix = "]"
    if summary.startswith(prefix) and summary.endswith(suffix) and max_chars > len(prefix) + len(suffix):
        inner = _truncate_summary(summary[len(prefix) : -len(suffix)], max_chars=max_chars - len(prefix) - len(suffix))
        return f"{prefix}{inner}{suffix}"
    candidates = [sentence for sentence in _sentences(summary) if len(sentence) <= max_chars]
    kept: list[str] = []
    for sentence in candidates:
        candidate = " ".join([*kept, sentence])
        if len(candidate) > max_chars:
            break
        kept.append(sentence)
    return " ".join(kept) if kept else summary[:max_chars]


def _sentences(text: str) -> list[str]:
    """Split prose at sentence boundaries without altering entity casing."""
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", text.strip()) if part.strip()]


def _drive_to_completion(awaitable: Any, *, timeout: float | None = None) -> Any:
    """Drive an awaitable from sync context regardless of loop state.

    Thin wrapper over the shared sync↔async bridge
    (:func:`lazybridge._asyncbridge.run_coroutine_blocking`) so Memory's
    compression path has semantics identical to ``Agent.__call__`` —
    including nest_asyncio (Jupyter), contextvars propagation, and
    loop-closed cleanup, none of which this path handled before.

    When ``timeout`` is set the awaitable is wrapped in
    ``asyncio.wait_for`` inside the executing loop; on deadline the
    coroutine is cancelled and a :class:`TimeoutError` propagates so the
    caller can fall back gracefully.
    """
    return run_coroutine_blocking(lambda: _ensure_coroutine(awaitable), timeout=timeout)


async def _ensure_coroutine(awaitable: Any) -> Any:
    """Wrap any awaitable in a coroutine so the bridge can await it."""
    return await awaitable
