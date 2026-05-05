"""Wave 4.1 — bounded streaming sink with backpressure.

Pre-W4.1 the ``stream()`` queue was an unbounded ``asyncio.Queue``.
A slow consumer (terminal, network, downstream blocker) caused the
queue to grow without bound while the provider kept producing
tokens — masking backpressure and risking unbounded memory growth.

W4.1 makes the sink bounded (default ``stream_buffer=64``) so a
slow consumer naturally throttles the provider via ``await
sink.put(...)`` blocking.  The buffer size is configurable and
validated at construction.
"""

from __future__ import annotations

import asyncio

import pytest

from lazybridge.engines.llm import LLMEngine

# ---------------------------------------------------------------------------
# Constructor surface — validation
# ---------------------------------------------------------------------------


def test_stream_buffer_default_is_64():
    eng = LLMEngine("claude-opus-4-7")
    assert eng.stream_buffer == 64


def test_stream_buffer_custom_value_accepted():
    eng = LLMEngine("claude-opus-4-7", stream_buffer=8)
    assert eng.stream_buffer == 8


def test_stream_buffer_zero_rejected():
    with pytest.raises(ValueError, match="stream_buffer must be >= 1"):
        LLMEngine("claude-opus-4-7", stream_buffer=0)


def test_stream_buffer_negative_rejected():
    with pytest.raises(ValueError, match="stream_buffer must be >= 1"):
        LLMEngine("claude-opus-4-7", stream_buffer=-1)


# ---------------------------------------------------------------------------
# Backpressure — producer blocks when sink saturated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bounded_queue_produces_backpressure():
    """A bounded queue blocks ``put`` when full so the producer
    can't outpace the consumer.  This guards the contract that
    LLMEngine's stream sink is bounded by ``stream_buffer``.
    """
    sink: asyncio.Queue[str | None] = asyncio.Queue(maxsize=2)

    # Fill to capacity.
    await sink.put("a")
    await sink.put("b")
    assert sink.full()

    # Now `put` must block — wrap in wait_for to detect the block.
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(sink.put("c"), timeout=0.05)

    # Drain one — put now succeeds.
    assert await sink.get() == "a"
    await asyncio.wait_for(sink.put("c"), timeout=0.05)
    assert sink.qsize() == 2


@pytest.mark.asyncio
async def test_unbounded_queue_does_not_block_for_comparison():
    """Sanity: confirms an unbounded queue (the pre-W4.1 behaviour)
    would NOT block — so the bounded test above is testing the new
    behaviour, not an asyncio quirk.
    """
    sink: asyncio.Queue[str | None] = asyncio.Queue()  # unbounded

    # Push 1000 items — none of them block.
    for i in range(1000):
        await asyncio.wait_for(sink.put(f"x{i}"), timeout=0.05)
    assert sink.qsize() == 1000


# ---------------------------------------------------------------------------
# stream() pipeline integration — bounded sink end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_method_uses_bounded_sink(monkeypatch):
    """Patch ``LLMEngine._loop`` to inspect the sink it receives:
    confirm it's a bounded queue with ``maxsize == stream_buffer``.
    """
    captured_sink: list[asyncio.Queue] = []

    async def _fake_loop(self, env, *, _stream_sink, **_kw):
        captured_sink.append(_stream_sink)
        # Push a couple of tokens then sentinel.
        await _stream_sink.put("hello")
        await _stream_sink.put(" world")
        return None

    monkeypatch.setattr(LLMEngine, "_loop", _fake_loop)

    eng = LLMEngine("claude-opus-4-7", stream_buffer=8)
    from lazybridge.envelope import Envelope

    chunks: list[str] = []
    async for tok in eng.stream(
        Envelope.from_task("hi"),
        tools=[],
        output_type=str,
        memory=None,
        session=None,
    ):
        chunks.append(tok)

    assert chunks == ["hello", " world"]
    assert len(captured_sink) == 1
    sink = captured_sink[0]
    assert isinstance(sink, asyncio.Queue)
    # The sink is bounded to stream_buffer.
    assert sink.maxsize == 8


@pytest.mark.asyncio
async def test_stream_default_buffer_is_64(monkeypatch):
    captured: list[asyncio.Queue] = []

    async def _fake_loop(self, env, *, _stream_sink, **_kw):
        captured.append(_stream_sink)
        return None

    monkeypatch.setattr(LLMEngine, "_loop", _fake_loop)

    eng = LLMEngine("claude-opus-4-7")  # default stream_buffer=64
    from lazybridge.envelope import Envelope

    async for _ in eng.stream(
        Envelope.from_task("hi"),
        tools=[],
        output_type=str,
        memory=None,
        session=None,
    ):
        pass

    assert captured[0].maxsize == 64
