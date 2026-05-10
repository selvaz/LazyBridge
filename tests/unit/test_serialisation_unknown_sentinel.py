"""Regression test for B7 — unknown sentinel kind must raise, not silently
fall back to ``from_prev``.

Background: ``_sentinel_from_ref`` previously returned ``from_prev`` when
the ``kind`` field was unrecognised.  A round-tripped Plan with a custom
or stale sentinel kind quietly became a different Plan with no warning.
The fix raises ``ValueError`` with a typo-aware message.
"""

from __future__ import annotations

import pytest

from lazybridge.engines.plan._serialisation import _sentinel_from_ref
from lazybridge.sentinels import from_prev


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match=r"unknown kind=.*'from_lalaland'"):
        _sentinel_from_ref({"kind": "from_lalaland", "name": "x"})


def test_missing_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown kind="):
        _sentinel_from_ref({"name": "x"})


def test_known_kinds_still_round_trip() -> None:
    # Sanity: the canonical kinds resolve correctly.
    assert _sentinel_from_ref({"kind": "from_prev"}) is from_prev
    # ``None`` is the legacy default and remains lenient (means "no override").
    assert _sentinel_from_ref(None) is from_prev
