"""Unit tests for Executor retry logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lazybridge.core.executor import Executor, _is_retryable
from lazybridge.core.types import CompletionRequest, CompletionResponse, Message, Role, UsageStats


def _make_request():
    return CompletionRequest(messages=[Message(role=Role.USER, content="hello")])


def _make_response():
    return CompletionResponse(content="ok", usage=UsageStats())


class FakeRetryableError(Exception):
    status_code = 429


class FakeServerError(Exception):
    status_code = 500


class FakeAuthError(Exception):
    status_code = 401


class FakeGenericError(Exception):
    pass


# ---------------------------------------------------------------------------
# _is_retryable classification
# ---------------------------------------------------------------------------


def test_is_retryable_429():
    assert _is_retryable(FakeRetryableError("rate limit"))


def test_is_retryable_500():
    assert _is_retryable(FakeServerError("internal"))


def test_is_not_retryable_401():
    assert not _is_retryable(FakeAuthError("unauthorized"))


def test_is_retryable_string_match():
    assert _is_retryable(FakeGenericError("connection reset"))


def test_is_not_retryable_generic():
    assert not _is_retryable(FakeGenericError("bad request"))


# ---------------------------------------------------------------------------
# Executor retry behavior
# ---------------------------------------------------------------------------


def test_retry_succeeds_on_second_attempt():
    mock_provider = MagicMock()
    mock_provider.complete.side_effect = [FakeRetryableError("rate limit"), _make_response()]

    with patch("lazybridge.core.executor._resolve_provider", return_value=mock_provider):
        executor = Executor("test", max_retries=1, retry_delay=0.01)

    with patch("time.sleep"):
        result = executor.execute(_make_request())

    assert result.content == "ok"
    assert mock_provider.complete.call_count == 2


def test_retry_exhausted_raises():
    mock_provider = MagicMock()
    mock_provider.complete.side_effect = FakeRetryableError("rate limit")

    with patch("lazybridge.core.executor._resolve_provider", return_value=mock_provider):
        executor = Executor("test", max_retries=2, retry_delay=0.01)

    with patch("time.sleep"), pytest.raises(FakeRetryableError):
        executor.execute(_make_request())

    assert mock_provider.complete.call_count == 3  # initial + 2 retries


def test_no_retry_on_auth_error():
    mock_provider = MagicMock()
    mock_provider.complete.side_effect = FakeAuthError("unauthorized")

    with patch("lazybridge.core.executor._resolve_provider", return_value=mock_provider):
        executor = Executor("test", max_retries=3, retry_delay=0.01)

    with pytest.raises(FakeAuthError):
        executor.execute(_make_request())

    assert mock_provider.complete.call_count == 1


def test_no_retry_when_max_retries_zero():
    mock_provider = MagicMock()
    mock_provider.complete.side_effect = FakeRetryableError("rate limit")

    with patch("lazybridge.core.executor._resolve_provider", return_value=mock_provider):
        executor = Executor("test", max_retries=0)

    with pytest.raises(FakeRetryableError):
        executor.execute(_make_request())

    assert mock_provider.complete.call_count == 1


# ---------------------------------------------------------------------------
# Async retry
# ---------------------------------------------------------------------------


async def test_async_retry_succeeds():
    from unittest.mock import AsyncMock

    mock_provider = MagicMock()
    mock_provider.acomplete = AsyncMock(side_effect=[FakeRetryableError("rate limit"), _make_response()])

    with patch("lazybridge.core.executor._resolve_provider", return_value=mock_provider):
        executor = Executor("test", max_retries=1, retry_delay=0.01)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await executor.aexecute(_make_request())

    assert result.content == "ok"
    assert mock_provider.acomplete.call_count == 2
