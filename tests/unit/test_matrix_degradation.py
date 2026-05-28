"""``lazybridge.matrix`` graceful-degradation tests.

``provider_capabilities()`` imports every provider class.  A single broken
optional SDK that explodes at import time must not blind the whole matrix —
the failing provider is dropped (with a warning) while the rest survive.
"""

from __future__ import annotations

import importlib

import pytest

from lazybridge import matrix


@pytest.fixture(autouse=True)
def _clear_matrix_cache():
    """The matrix is ``lru_cache``d — clear it around each test so the
    import behaviour under test is actually re-exercised."""
    matrix.provider_capabilities.cache_clear()
    yield
    matrix.provider_capabilities.cache_clear()


def test_all_providers_present_by_default() -> None:
    caps = matrix.provider_capabilities()
    expected = {name for name, _, _ in matrix._PROVIDER_IMPORTS}
    assert set(caps) == expected


def test_one_broken_provider_does_not_blind_the_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """If google's module raises at import time, the matrix drops google
    but still reports every other provider (plus a warning)."""
    broken_module = "lazybridge.core.providers.google"
    real_import_module = importlib.import_module

    def fake_import_module(name: str, *args, **kwargs):
        if name == broken_module:
            raise RuntimeError("simulated broken provider SDK")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(matrix.importlib, "import_module", fake_import_module)

    with pytest.warns(UserWarning, match="provider 'google' is unavailable"):
        caps = matrix.provider_capabilities()

    assert "google" not in caps
    # Every other provider still made it into the matrix.
    others = {name for name, _, _ in matrix._PROVIDER_IMPORTS if name != "google"}
    assert others <= set(caps)
