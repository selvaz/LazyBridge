"""Regression tests for the pre-main audit fixes."""

from __future__ import annotations

import logging

import pytest

from lazybridge.gui import close_server, install_gui_methods, open_gui
from lazybridge.gui._global import _reset_for_tests
from lazybridge.human import HumanAgent
from lazybridge.supervisor import SupervisorAgent


@pytest.fixture(autouse=True)
def _fresh_server():
    _reset_for_tests()
    install_gui_methods()
    yield
    close_server()


def test_supervisor_gui_method_raises_helpful_error():
    sup = SupervisorAgent(name="sup", input_fn=lambda prompt: "continue")
    with pytest.raises(NotImplementedError, match="panel_input_fn"):
        sup.gui()


def test_human_agent_gui_method_raises_helpful_error():
    human = HumanAgent(name="reviewer", input_fn=lambda prompt: "ok")
    with pytest.raises(NotImplementedError, match="panel_input_fn"):
        human.gui()


def test_open_gui_routes_supervisor_to_helpful_error():
    sup = SupervisorAgent(name="sup2", input_fn=lambda p: "continue")
    with pytest.raises(NotImplementedError, match="panel_input_fn"):
        open_gui(sup, open_browser=False)


def test_supervisor_achat_does_not_hang_and_preserves_behaviour():
    """The duplicate if-branch was collapsed; verify both paths
    (with/without ainput_fn) still work."""
    sup_sync = SupervisorAgent(name="s1", input_fn=lambda p: "continue")
    # Default: ainput_fn is None → falls through to sync REPL in thread.
    import asyncio

    resp = asyncio.run(sup_sync.achat("prev output"))
    assert resp.content == "prev output"


async def test_supervisor_achat_with_ainput_fn_still_works():
    """With an ainput_fn set (even if we don't natively use it yet),
    achat should still succeed — the collapse shouldn't break this path."""

    async def _ainput(prompt: str) -> str:
        return "continue"

    sup = SupervisorAgent(
        name="s2",
        input_fn=lambda p: "continue",
        ainput_fn=_ainput,
    )
    resp = await sup.achat("x")
    assert resp.content == "x"


def test_downloader_parquet_load_failure_logged(monkeypatch, tmp_path, caplog):
    """pd.read_parquet failures must produce a DEBUG log, not silent None."""
    pytest.importorskip("pandas", reason="data_downloader requires pandas")

    from lazybridge.ext.data_downloader.downloader import DataCache, _logger

    # Create a broken parquet file.
    p = tmp_path / "broken.parquet"
    p.write_bytes(b"not a parquet")

    class _FakeCache(DataCache):
        def path_for(self, ticker):  # pragma: no cover — trivial
            return str(p)

    with caplog.at_level(logging.DEBUG, logger=_logger.name):
        cache = _FakeCache.__new__(_FakeCache)
        cache._dir = tmp_path
        result = cache.load("broken")
    assert result is None
    assert any("load('broken')" in rec.getMessage() for rec in caplog.records)
