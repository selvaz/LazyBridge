"""Regression test for data_downloader retry jitter."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pd = pytest.importorskip("pandas", reason="data_downloader requires pandas")


def test_http_get_retry_applies_jitter():
    """Each retry sleep includes a ±25% random jitter so simultaneous
    workers don't retry in lockstep."""
    from lazybridge.ext.data_downloader import downloader as dl

    class _Cfg:
        max_retries = 4
        retry_sleep = 1.0
        request_timeout = 5

    # Force every requests.get call to raise a retryable error.
    class _FakeRequests:
        class ConnectionError(Exception):
            pass

        class Timeout(Exception):
            pass

        class HTTPError(Exception):
            pass

        @staticmethod
        def get(*_a, **_kw):
            raise _FakeRequests.ConnectionError("boom")

    sleeps: list[float] = []

    with (
        patch.object(dl, "time", new=type("t", (), {"sleep": sleeps.append})),
        patch.dict("sys.modules", {"requests": _FakeRequests}),
    ):
        try:
            dl._http_get("http://x", _Cfg())
        except _FakeRequests.ConnectionError:
            pass

    # 3 sleeps (4 attempts → 3 inter-retry sleeps).
    assert len(sleeps) == 3, sleeps

    # Each sleep is exponential-ish — expected (without jitter) would be
    # [1.0, 2.0, 4.0]. With ±25% jitter: [0.75..1.25, 1.5..2.5, 3.0..5.0].
    assert 0.75 <= sleeps[0] <= 1.25
    assert 1.5 <= sleeps[1] <= 2.5
    assert 3.0 <= sleeps[2] <= 5.0

    # Jitter produces distinct values across multiple runs.
    sleeps2: list[float] = []
    with (
        patch.object(dl, "time", new=type("t", (), {"sleep": sleeps2.append})),
        patch.dict("sys.modules", {"requests": _FakeRequests}),
    ):
        try:
            dl._http_get("http://x", _Cfg())
        except _FakeRequests.ConnectionError:
            pass
    # At least one of the three values should differ between runs
    # (probability of all three identical at float precision is ~zero).
    assert sleeps != sleeps2
