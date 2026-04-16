"""Integration tests — full pipeline flows with mocked providers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lazybridge.core.types import CompletionResponse, UsageStats
from lazybridge.lazy_agent import LazyAgent
from lazybridge.lazy_context import LazyContext
from lazybridge.lazy_session import LazySession


def _fake_response(content: str, input_tokens: int = 100, output_tokens: int = 50) -> CompletionResponse:
    return CompletionResponse(
        content=content,
        usage=UsageStats(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=0.001),
    )


def _make_session_agent(sess, name, response_content):
    """Create a real LazyAgent wired to a session with mocked executor."""
    with patch("lazybridge.core.executor.Executor.__init__", return_value=None):
        agent = LazyAgent(
            "anthropic",
            name=name,
            session=sess,
        )
    mock_exec = MagicMock()
    mock_exec.provider.get_default_max_tokens.return_value = 4096
    mock_exec.model = "test-model"
    resp = _fake_response(response_content)
    mock_exec.execute.return_value = resp
    mock_exec.aexecute = AsyncMock(return_value=resp)
    agent._executor = mock_exec
    return agent


# ---------------------------------------------------------------------------
# Full pipeline flow: session → agents → store → context → graph
# ---------------------------------------------------------------------------


def test_full_pipeline_session_to_context():
    sess = LazySession(tracking="basic")
    researcher = _make_session_agent(sess, "researcher", "Found 3 papers on LLMs")
    writer = _make_session_agent(sess, "writer", "Blog post about LLMs")

    researcher.chat("find AI papers")
    sess.store.write("findings", researcher._last_output)

    ctx = LazyContext.from_agent(researcher) + LazyContext.from_store(sess.store, keys=["findings"])
    writer.chat("write a blog post", context=ctx)

    assert researcher._last_output == "Found 3 papers on LLMs"
    assert writer._last_output == "Blog post about LLMs"
    assert sess.store.read("findings") == "Found 3 papers on LLMs"
    assert len(sess.graph.nodes()) == 2
    assert len(sess.events.get()) > 0


# ---------------------------------------------------------------------------
# Usage summary aggregation
# ---------------------------------------------------------------------------


def test_usage_summary_aggregation():
    sess = LazySession(tracking="verbose")
    a = _make_session_agent(sess, "agent_a", "output_a")
    b = _make_session_agent(sess, "agent_b", "output_b")

    a.chat("task 1")
    b.chat("task 2")
    a.chat("task 3")

    summary = sess.usage_summary()
    assert summary["total"]["input_tokens"] > 0
    assert summary["total"]["output_tokens"] > 0
    assert "agent_a" in summary["by_agent"]
    assert "agent_b" in summary["by_agent"]
    assert summary["by_agent"]["agent_a"]["input_tokens"] > summary["by_agent"]["agent_b"]["input_tokens"]


def test_usage_summary_empty_session():
    sess = LazySession(tracking="verbose")
    summary = sess.usage_summary()
    assert summary["total"]["input_tokens"] == 0
    assert summary["total"]["cost_usd"] == 0.0
    assert summary["by_agent"] == {}


# ---------------------------------------------------------------------------
# Streaming methods — dedicated API
# ---------------------------------------------------------------------------


def test_chat_stream_returns_iterator():
    from lazybridge.core.types import StreamChunk

    with patch("lazybridge.core.executor.Executor.__init__", return_value=None):
        agent = LazyAgent("anthropic", name="test")

    chunks = [
        StreamChunk(delta="hello "),
        StreamChunk(delta="world"),
        StreamChunk(delta="", is_final=True, stop_reason="end_turn"),
    ]

    mock_exec = MagicMock()
    mock_exec.provider.get_default_max_tokens.return_value = 4096
    mock_exec.model = "test-model"
    mock_exec.stream.return_value = iter(chunks)
    agent._executor = mock_exec

    result = agent.chat_stream("hello")
    collected = list(result)
    assert len(collected) == 3
    assert collected[0].delta == "hello "
    assert agent._last_output == "hello world"


# ---------------------------------------------------------------------------
# Exporter tests — StructuredLogExporter
# ---------------------------------------------------------------------------


def test_structured_log_exporter():
    import logging

    from lazybridge.exporters import StructuredLogExporter

    records = []

    class Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("test.structured")
    logger.addHandler(Handler())
    logger.setLevel(logging.DEBUG)

    exporter = StructuredLogExporter(logger_name="test.structured")
    exporter.export({"event_type": "test", "data": {"key": "value"}})

    assert len(records) == 1
    assert "test" in records[0]


# ---------------------------------------------------------------------------
# OTelExporter — import guard
# ---------------------------------------------------------------------------


def test_otel_exporter_import_guard():
    from lazybridge.exporters import OTelExporter

    try:
        import opentelemetry  # noqa: F401

        pytest.skip("opentelemetry is installed")
    except ImportError:
        with pytest.raises(ImportError, match="OpenTelemetry"):
            OTelExporter()
