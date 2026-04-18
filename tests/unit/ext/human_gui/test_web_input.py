"""Tests for lazybridge.ext.human_gui.WebInputServer and web_input_fn."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from lazybridge.ext.human_gui import WebInputServer, web_input_fn


@pytest.fixture
def server():
    srv = WebInputServer(open_browser=False)
    try:
        yield srv
    finally:
        srv.close()


def _get(url: str, timeout: float = 2.0) -> tuple[int, bytes]:
    try:
        resp = urllib.request.urlopen(url, timeout=timeout)
        return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post_json(url: str, payload: dict, timeout: float = 2.0) -> tuple[int, dict]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_serves_with_title_and_token(server):
    status, body = _get(f"http://127.0.0.1:{server.port}/")
    assert status == 200
    page = body.decode()
    assert "<title>LazyBridge" in page
    # Token inlined as JSON string literal for the JS.
    assert json.dumps(server.token) in page


def test_healthz_public(server):
    status, body = _get(f"http://127.0.0.1:{server.port}/healthz")
    assert status == 200
    assert json.loads(body) == {"ok": True, "closed": False}


def test_prompt_requires_token(server):
    status, _ = _get(f"http://127.0.0.1:{server.port}/prompt")
    assert status == 401


def test_prompt_idle_returns_null(server):
    status, body = _get(f"http://127.0.0.1:{server.port}/prompt?t={server.token}")
    assert status == 200
    data = json.loads(body)
    assert data["prompt"] is None
    assert data["closed"] is False


def test_ask_round_trip(server):
    result: list[str] = []

    def _ask():
        result.append(server.ask("pick one", quick_commands=["continue", "retry"]))

    t = threading.Thread(target=_ask)
    t.start()
    # Give ask() a moment to publish the prompt.
    deadline = time.monotonic() + 1.0
    data = None
    while time.monotonic() < deadline:
        status, body = _get(f"http://127.0.0.1:{server.port}/prompt?t={server.token}")
        data = json.loads(body)
        if data["prompt"] is not None:
            break
        time.sleep(0.02)
    assert data and data["prompt"] == "pick one"
    assert data["quick_commands"] == ["continue", "retry"]
    seq = data["seq"]

    status, payload = _post_json(
        f"http://127.0.0.1:{server.port}/submit?t={server.token}",
        {"seq": seq, "response": "continue"},
    )
    assert status == 200
    assert payload == {"accepted": True}

    t.join(timeout=2)
    assert result == ["continue"]


def test_submit_wrong_seq_rejected(server):
    # No prompt pending -> any submit is rejected.
    status, payload = _post_json(
        f"http://127.0.0.1:{server.port}/submit?t={server.token}",
        {"seq": 999, "response": "noop"},
    )
    assert status == 200
    assert payload == {"accepted": False}


def test_submit_requires_token(server):
    status, _ = _post_json(
        f"http://127.0.0.1:{server.port}/submit",
        {"seq": 1, "response": "x"},
    )
    assert status == 401


def test_ask_timeout(server):
    with pytest.raises(TimeoutError):
        server.ask("no one will answer", timeout=0.2)


def test_close_unblocks_waiting_ask():
    srv = WebInputServer(open_browser=False)
    errors: list[Exception] = []

    def _ask():
        try:
            srv.ask("wait forever")
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=_ask)
    t.start()
    time.sleep(0.1)
    srv.close()
    t.join(timeout=2)
    assert not t.is_alive()
    assert errors and isinstance(errors[0], RuntimeError)


def test_ask_after_close_raises():
    srv = WebInputServer(open_browser=False)
    srv.close()
    with pytest.raises(RuntimeError):
        srv.ask("too late")


async def test_aask_round_trip(server):
    async def _poll_and_submit() -> None:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            status, body = _get(f"http://127.0.0.1:{server.port}/prompt?t={server.token}")
            data = json.loads(body)
            if data["prompt"] is not None:
                _post_json(
                    f"http://127.0.0.1:{server.port}/submit?t={server.token}",
                    {"seq": data["seq"], "response": "from-async"},
                )
                return
            await asyncio.sleep(0.02)

    asker = asyncio.create_task(server.aask("async prompt"))
    await _poll_and_submit()
    result = await asyncio.wait_for(asker, timeout=2)
    assert result == "from-async"


def test_input_fn_property_plugs_into_callable_signature(server):
    fn = server.input_fn
    result: list[str] = []

    def _call():
        result.append(fn("what now?"))

    t = threading.Thread(target=_call)
    t.start()
    deadline = time.monotonic() + 1.0
    seq = None
    while time.monotonic() < deadline:
        data = json.loads(_get(f"http://127.0.0.1:{server.port}/prompt?t={server.token}")[1])
        if data["prompt"] is not None:
            seq = data["seq"]
            break
        time.sleep(0.02)
    assert seq is not None
    _post_json(
        f"http://127.0.0.1:{server.port}/submit?t={server.token}",
        {"seq": seq, "response": "ok"},
    )
    t.join(timeout=2)
    assert result == ["ok"]


def test_web_input_fn_factory_carries_server_handle():
    fn = web_input_fn(open_browser=False)
    try:
        assert callable(fn)
        assert isinstance(fn.server, WebInputServer)
        assert fn.server.port > 0
    finally:
        fn.server.close()


def test_works_with_supervisor_agent_input_fn(server):
    """End-to-end integration: SupervisorAgent drives web_input_fn transparently."""
    from lazybridge import SupervisorAgent

    supervisor = SupervisorAgent(name="sup", input_fn=server.input_fn)
    result: list[object] = []

    def _run():
        result.append(supervisor.chat("research output"))

    t = threading.Thread(target=_run)
    t.start()

    # Drive one REPL prompt → "continue".
    deadline = time.monotonic() + 1.0
    seq = None
    while time.monotonic() < deadline:
        data = json.loads(_get(f"http://127.0.0.1:{server.port}/prompt?t={server.token}")[1])
        if data["prompt"] is not None:
            seq = data["seq"]
            break
        time.sleep(0.02)
    assert seq is not None
    _post_json(
        f"http://127.0.0.1:{server.port}/submit?t={server.token}",
        {"seq": seq, "response": "continue"},
    )
    t.join(timeout=2)
    assert len(result) == 1
    assert result[0].content == "research output"
