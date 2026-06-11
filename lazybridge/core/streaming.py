"""Ambient token-sink plumbing — engine-agnostic token streaming.

``LLMEngine.stream()`` streams tokens by passing a bounded ``asyncio.Queue``
("sink") down into its own loop.  Composite engines (``Plan``,
``ReplanEngine``) cannot use that mechanism directly: the LLM calls happen
several frames below them, inside step Agents they do not construct.

This module bridges the gap with a context-local *ambient* sink:

* :func:`stream_envelope_run` — the composite engine's ``stream()`` binds a
  sink to the current context, runs its normal ``run()`` in a background
  task (the task inherits the context, and therefore the sink), and yields
  tokens as they arrive.
* :func:`consume_ambient_token_sink` — ``LLMEngine.run()`` adopts the sink
  when one is bound, streaming its turns into it while it builds the normal
  result Envelope.  Adoption *consumes* the binding for everything below
  that frame, so nested agents-as-tools stay silent — the same "tool calls
  run silently between streamed turns" contract ``LLMEngine.stream()``
  already documents.
* :func:`suppress_ambient_token_sink` / :func:`restore_ambient_token_sink` —
  composite engines unbind the sink around ``asyncio.gather`` bands so
  concurrent branches cannot interleave tokens into an unreadable stream.

Only the engine that *bound* the sink ever closes it (with a ``None``
sentinel); adopters must only ``put`` tokens.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lazybridge.envelope import Envelope

_AMBIENT_TOKEN_SINK: contextvars.ContextVar[asyncio.Queue[str | None] | None] = contextvars.ContextVar(
    "lazybridge_ambient_token_sink",
    default=None,
)

_SinkToken = contextvars.Token  # readability alias for annotations below


def consume_ambient_token_sink() -> tuple[asyncio.Queue[str | None] | None, _SinkToken[Any] | None]:
    """Return the ambient sink (or ``None``) and unbind it for this context.

    The caller streams into the returned queue and MUST pass the returned
    token to :func:`restore_ambient_token_sink` when its frame completes
    (typically in a ``finally``).  Unbinding while the caller runs is what
    keeps nested engines from double-streaming into the same queue.
    """
    sink = _AMBIENT_TOKEN_SINK.get()
    if sink is None:
        return None, None
    return sink, _AMBIENT_TOKEN_SINK.set(None)


def suppress_ambient_token_sink() -> _SinkToken[Any] | None:
    """Unbind the ambient sink for the current context, if one is bound.

    Used around ``asyncio.gather`` bands: tasks created inside the band
    snapshot the suppressed context, so concurrent branches do not
    interleave tokens.  Pass the returned token to
    :func:`restore_ambient_token_sink` afterwards (no-op on ``None``).
    """
    if _AMBIENT_TOKEN_SINK.get() is None:
        return None
    return _AMBIENT_TOKEN_SINK.set(None)


def restore_ambient_token_sink(token: _SinkToken[Any] | None) -> None:
    """Undo :func:`consume_ambient_token_sink` / :func:`suppress_ambient_token_sink`."""
    if token is not None:
        _AMBIENT_TOKEN_SINK.reset(token)


async def stream_envelope_run(
    run: Callable[[], Awaitable[Envelope[Any]]],
    *,
    buffer: int = 64,
) -> AsyncGenerator[str, None]:
    """Drive ``run()`` in a background task and yield tokens as nested
    LLM engines produce them.

    The sink is bounded (``buffer``) so a slow consumer throttles the
    producing engines via ``await sink.put(...)`` — the same backpressure
    contract as ``LLMEngine.stream()``.

    Fallback: when ``run()`` completes without any nested engine adopting
    the sink (pure-function plans, mock engines, non-LLM tools), the final
    envelope's text is yielded once — streaming degrades to the pre-token
    behaviour instead of producing an empty stream.

    Early consumer exit cancels the background run; exceptions raised by
    ``run()`` propagate to the consumer after the stream drains.
    """
    sink: asyncio.Queue[str | None] = asyncio.Queue(maxsize=buffer)
    result_box: list[Envelope[Any]] = []

    async def _runner() -> None:
        try:
            result_box.append(await run())
        finally:
            await sink.put(None)  # sentinel — run done (or failed)

    # Bind the sink only for the background task: ``create_task`` snapshots
    # the current context, so resetting immediately afterwards keeps the
    # *caller's* context clean while the plan body still sees the sink.
    bind = _AMBIENT_TOKEN_SINK.set(sink)
    try:
        task = asyncio.create_task(_runner())
    finally:
        _AMBIENT_TOKEN_SINK.reset(bind)

    yielded = False
    cancelled_by_us = False
    try:
        while True:
            token = await sink.get()
            if token is None:
                break
            yielded = True
            yield token
    finally:
        # Consumer broke early (or we finished): never leave the run
        # dangling — a cancelled stream must also stop the LLM spend.
        if not task.done():
            task.cancel()
            cancelled_by_us = True
        try:
            await task
        except asyncio.CancelledError:
            if not cancelled_by_us:
                raise

    if not yielded and result_box:
        yield result_box[0].text()
