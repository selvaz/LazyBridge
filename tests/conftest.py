"""Global test fixtures — mock-only, no API keys required."""
from __future__ import annotations

import logging
import sys
import asyncio
import pytest
from unittest.mock import patch, AsyncMock

# ---------------------------------------------------------------------------
# Nested event loop compatibility (Spyder / Jupyter)
#
# pytest-asyncio uses asyncio.Runner (Python 3.11+) which refuses to run when
# a loop is already active — this breaks tests inside Spyder's IPython kernel.
# nest_asyncio patches loop.run_until_complete() for nesting but does not touch
# asyncio.Runner.  This block implements the missing patch: when a loop is
# already running, hand off directly to loop.run_until_complete() which
# nest_asyncio has already made nest-safe.
# On CLI there is no running loop, so the original Runner path is used unchanged.
# ---------------------------------------------------------------------------
if sys.version_info >= (3, 11) and not getattr(asyncio.Runner.run, "_nest_patched", False):
    try:
        import nest_asyncio as _nest_asyncio  # noqa: F401 — confirm it is present

        _orig_runner_run = asyncio.Runner.run

        def _nested_runner_run(self, coro, *, context=None):
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is not None:
                _nest_asyncio.apply(running)
                return running.run_until_complete(coro)
            return _orig_runner_run(self, coro, context=context)

        _nested_runner_run._nest_patched = True
        asyncio.Runner.run = _nested_runner_run

    except ImportError:
        pass  # nest_asyncio not installed — standard Runner behaviour

from lazybridge.core.types import CompletionResponse, StreamChunk, UsageStats


# ---------------------------------------------------------------------------
# Suppress "Task exception was never retrieved: Event loop is closed"
#
# httpx's AsyncClient schedules aclose() tasks at GC time.  When those fire
# after _run_suppressed() (or pytest-asyncio) has already closed the loop,
# asyncio logs them as ERROR via the "asyncio" logger — purely cosmetic, all
# tests pass.  The filter below silences exactly that pattern.
# ---------------------------------------------------------------------------

class _SuppressLoopClosedFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "asyncio":
            return True
        msg = record.getMessage()
        return not (
            "Task exception was never retrieved" in msg
            and "Event loop is closed" in msg
        )

logging.getLogger("asyncio").addFilter(_SuppressLoopClosedFilter())


@pytest.fixture
def fake_response():
    return CompletionResponse(
        content="mock response",
        usage=UsageStats(input_tokens=10, output_tokens=5),
    )


@pytest.fixture
def fake_chunks():
    return [
        StreamChunk(delta="mock "),
        StreamChunk(
            delta="response",
            stop_reason="end_turn",
            is_final=True,
            usage=UsageStats(input_tokens=10, output_tokens=2),
        ),
    ]


@pytest.fixture
def mock_execute(fake_response):
    with patch(
        "lazybridge.core.executor.Executor.execute",
        return_value=fake_response,
    ) as m:
        yield m


@pytest.fixture
def mock_aexecute(fake_response):
    with patch(
        "lazybridge.core.executor.Executor.aexecute",
        new_callable=AsyncMock,
        return_value=fake_response,
    ) as m:
        yield m


@pytest.fixture
def mock_stream(fake_chunks):
    with patch(
        "lazybridge.core.executor.Executor.stream",
        return_value=iter(fake_chunks),
    ) as m:
        yield m
