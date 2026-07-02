"""Global test fixtures — mock-only, no API keys required."""

from __future__ import annotations

import asyncio
import logging
import sys
import types
from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap preflight — fail fast on a missing dev dependency.
#
# ~30% of the LazyBridge suite is ``async def`` tests under
# ``asyncio_mode = "auto"``.  Without ``pytest-asyncio`` installed those
# tests collect successfully but every one fails with the cryptic
# "async def functions are not natively supported" message — easy to
# misread as a framework bug.  The preflight gives a one-line install
# hint instead.
try:
    import pytest_asyncio  # noqa: F401
except ImportError as _e:  # pragma: no cover — only fires when the dev install is partial
    raise RuntimeError(
        "LazyBridge tests require ``pytest-asyncio``.\n"
        "  Install it via the test extra:\n"
        "    pip install -e '.[test]'   # or [test,all] for full coverage\n"
        "  (The project's ``[test]`` extra declares pytest-asyncio>=0.23 — "
        "your environment is missing it.)"
    ) from _e

# ---------------------------------------------------------------------------
# Google provider stub
#
# google-genai imports google.auth.crypt.es which loads a Rust extension that
# panics on this environment (pyo3_runtime.PanicException — not catchable in
# Python).  Stub the entire google.* namespace before any lazybridge import so
# that GoogleProvider silently receives _genai = None and skips initialisation.
# ---------------------------------------------------------------------------
_google_stubs = [
    "google",
    "google.auth",
    "google.auth.crypt",
    "google.auth.crypt.es",
    "google.auth.transport",
    "google.auth.transport.requests",
    "google.oauth2",
    "google.oauth2.service_account",
    "google.genai",
    "google.genai.types",
]
for _mod_name in _google_stubs:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

# ---------------------------------------------------------------------------
# OpenAI SDK stub (no-extras environments)
#
# ``lazybridge/core/providers/openai.py`` binds ``import openai`` to a
# module-global ``_openai`` at import time.  Several test modules stub
# ``sys.modules["openai"]`` at *their* import time — but by then the
# provider module may already be imported (via this conftest or an
# earlier test file), so the stub never reaches ``_openai`` and 15
# OpenAI/DeepSeek/LM Studio tests fail on ``NoneType has no attribute
# 'OpenAI'`` in environments without the SDK.  Install one MagicMock
# stub HERE, before any lazybridge import, so ``_openai`` binds to it
# consistently for the whole suite.  A real installed SDK wins.
# ---------------------------------------------------------------------------
import importlib.util as _importlib_util

if _importlib_util.find_spec("openai") is None:
    from unittest.mock import MagicMock as _MagicMock

    _openai_stub = types.ModuleType("openai")
    _openai_stub.OpenAI = _MagicMock(name="OpenAI")  # type: ignore[attr-defined]
    _openai_stub.AsyncOpenAI = _MagicMock(name="AsyncOpenAI")  # type: ignore[attr-defined]
    sys.modules["openai"] = _openai_stub

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
        import nest_asyncio as _nest_asyncio

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
        return not ("Task exception was never retrieved" in msg and "Event loop is closed" in msg)


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
