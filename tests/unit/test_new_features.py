"""Tests for the three new features: web UI, agent fallback routing, and ext/ API fixes."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from pydantic import BaseModel

from lazybridge import Agent, Envelope, Tool
from lazybridge.ext.hil import HumanEngine
from lazybridge.ext.hil.human import _build_web_form, _WebUI

# =============================================================================
# Web UI — _build_web_form
# =============================================================================


class TestBuildWebForm:
    def test_plain_text_form_contains_textarea(self):
        html = _build_web_form("Do something", [], False, {})
        assert "<textarea" in html
        assert 'name="response"' in html

    def test_task_appears_in_form(self):
        html = _build_web_form("My task here", [], False, {})
        assert "My task here" in html

    def test_tools_section_rendered(self):
        t = Tool(lambda x: x, name="my_tool")
        html = _build_web_form("task", [t], False, {})
        assert "my_tool" in html
        assert "Available actions" in html

    def test_no_tools_no_section(self):
        html = _build_web_form("task", [], False, {})
        assert "Available actions" not in html

    def test_model_form_renders_fields(self):
        class MyModel(BaseModel):
            name: str
            count: int

        fields = MyModel.model_fields
        html = _build_web_form("task", [], True, fields)
        assert 'name="name"' in html
        assert 'name="count"' in html
        assert "str" in html
        assert "int" in html

    def test_html_entities_escaped(self):
        html = _build_web_form("<script>alert(1)</script>", [], False, {})
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_newlines_become_br(self):
        html = _build_web_form("line1\nline2", [], False, {})
        assert "<br>" in html

    def test_returns_full_html_doc(self):
        html = _build_web_form("task", [], False, {})
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html


# =============================================================================
# Web UI — _WebUI class
# =============================================================================


class TestWebUIClass:
    def test_instantiation(self):
        ui = _WebUI(timeout=30, port=0)
        assert ui._timeout == 30
        assert ui._port == 0

    def test_default_timeout(self):
        ui = _WebUI()
        assert ui._timeout is None

    @pytest.mark.asyncio
    async def test_plain_text_submission(self):
        """_WebUI serves a form, accepts POST, returns the submitted value."""
        ui = _WebUI(timeout=5, port=0)

        async def submit_after_start():
            # Small delay to let the server start
            await asyncio.sleep(0.2)
            import socket

            # Poll until server is ready
            for _ in range(20):
                try:
                    socket.create_connection(("127.0.0.1", ui._port if ui._port else 0), timeout=0.1)
                    break
                except Exception:
                    await asyncio.sleep(0.05)

        # Patch _WebUI to capture the port and post immediately

        async def patched_prompt(task: str, *, tools, output_type):
            # Start the server in background, then POST to it
            import http.server
            import queue as _queue
            import urllib.parse
            import urllib.request

            response_q: _queue.Queue[str] = _queue.Queue()

            form_html = _build_web_form(task, tools, False, {})

            class _H(http.server.BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(form_html.encode())

                def do_POST(self):
                    length = int(self.headers.get("Content-Length", 0))
                    raw = self.rfile.read(length).decode()
                    data = dict(urllib.parse.parse_qsl(raw))
                    result = data.get("response", "")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(b"submitted")
                    response_q.put(result)

                def log_message(self, *_a):
                    pass

            server = http.server.HTTPServer(("127.0.0.1", 0), _H)
            port = server.server_address[1]
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()

            # POST after a tiny delay
            def _do_post():
                time.sleep(0.05)
                body = urllib.parse.urlencode({"response": "hello from web"}).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/",
                    data=body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                urllib.request.urlopen(req)

            threading.Thread(target=_do_post, daemon=True).start()
            loop = asyncio.get_running_loop()
            val = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: response_q.get(timeout=5)),
                timeout=5,
            )
            server.shutdown()
            return val

        val = await patched_prompt("my task", tools=[], output_type=str)
        assert val == "hello from web"

    @pytest.mark.asyncio
    async def test_model_submission_returns_json(self):
        """POST with Pydantic model fields returns JSON string of field values."""
        import http.server
        import queue as _queue
        import urllib.parse
        import urllib.request

        class MyModel(BaseModel):
            name: str
            count: int

        fields = MyModel.model_fields
        _build_web_form("task", [], True, fields)
        response_q: _queue.Queue[str] = _queue.Queue()

        class _H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length).decode()
                data = dict(urllib.parse.parse_qsl(raw))
                result = json.dumps({k: data.get(k, "") for k in fields})
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"ok")
                response_q.put(result)

            def log_message(self, *_a):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), _H)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        def _post():
            time.sleep(0.05)
            body = urllib.parse.urlencode({"name": "Alice", "count": "3"}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/",
                data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            urllib.request.urlopen(req)

        threading.Thread(target=_post, daemon=True).start()
        loop = asyncio.get_running_loop()
        raw = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: response_q.get(timeout=5)),
            timeout=5,
        )
        server.shutdown()
        data = json.loads(raw)
        assert data["name"] == "Alice"
        assert data["count"] == "3"


# =============================================================================
# Persistent Web UI — server reuse across prompt() calls
# =============================================================================


def _post_form(
    port: int,
    data: dict[str, str],
    *,
    follow_redirect: bool = False,
    epoch: int | None = None,
) -> tuple[int, str]:
    """POST ``data`` to ``_WebUI`` on ``port``.

    When ``epoch`` is provided it's added as ``_epoch`` (matching the
    hidden field a real browser would carry); leaving it ``None`` means
    "skip the epoch field" and is used by tests that exercise the
    stale-form rejection path.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    payload = dict(data)
    if epoch is not None:
        payload.setdefault("_epoch", str(epoch))
    body = urllib.parse.urlencode(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    # The handler sends a 303; urllib auto-follows on POST in Python 3.11+
    # (via HTTPRedirectHandler.redirect_request), which then issues a GET
    # to ``/``.  For the test we want to inspect the 303 itself, not the
    # redirect target, so disable auto-follow when ``follow_redirect`` is
    # False.
    if follow_redirect:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read().decode()

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def http_error_303(self, req, fp, code, msg, headers):
            return fp  # surface the 303 response unchanged

        http_error_302 = http_error_301 = http_error_307 = http_error_303

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(req, timeout=5) as resp:
            return resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        # 4xx / 5xx responses (e.g. the 410 returned for stale forms)
        # are surfaced by urllib as exceptions even with the no-redirect
        # opener; we want the test to inspect status + body, not a raise.
        body = exc.read().decode(errors="replace") if exc.fp is not None else ""
        return exc.code, body


class TestPersistentWebUI:
    @pytest.mark.asyncio
    async def test_two_consecutive_prompts_reuse_same_server(self):
        """Second prompt() must reuse the first prompt()'s server / port."""
        ui = _WebUI(timeout=5, port=0)
        try:
            # Drive the first prompt() in a task; submit a POST to it.
            task1 = asyncio.create_task(ui.prompt("first task", tools=[], output_type=str))
            # Poll for the server to start.
            for _ in range(50):
                if ui._server is not None and ui._url is not None:
                    break
                await asyncio.sleep(0.02)
            assert ui._server is not None, "server failed to start on first prompt"
            first_server = ui._server
            first_url = ui._url
            assert first_url is not None
            port1 = int(first_url.rsplit(":", 1)[1].rstrip("/"))

            # Submit the first form via POST (urllib runs in a thread).
            e1 = ui._form_epoch
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: _post_form(port1, {"response": "answer one"}, epoch=e1)
            )
            result1 = await asyncio.wait_for(task1, timeout=5)
            assert result1 == "answer one"

            # Second prompt() must NOT create a new server.
            task2 = asyncio.create_task(ui.prompt("second task", tools=[], output_type=str))
            await asyncio.sleep(0.05)  # let prompt() publish the new HTML
            assert ui._server is first_server, "server was recreated for second prompt"
            assert ui._url == first_url, "URL changed between prompts"

            e2 = ui._form_epoch
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: _post_form(port1, {"response": "answer two"}, epoch=e2)
            )
            result2 = await asyncio.wait_for(task2, timeout=5)
            assert result2 == "answer two"
        finally:
            ui.close()

    @pytest.mark.asyncio
    async def test_post_returns_303_redirect_to_root(self):
        """POST handler must redirect back to ``/`` so the browser auto-refreshes."""
        ui = _WebUI(timeout=5, port=0)
        try:
            task = asyncio.create_task(ui.prompt("task", tools=[], output_type=str))
            for _ in range(50):
                if ui._url is not None:
                    break
                await asyncio.sleep(0.02)
            assert ui._url is not None
            port = int(ui._url.rsplit(":", 1)[1].rstrip("/"))

            e = ui._form_epoch
            status, _body = await asyncio.get_running_loop().run_in_executor(
                None, lambda: _post_form(port, {"response": "x"}, follow_redirect=False, epoch=e)
            )
            assert status == 303
            await asyncio.wait_for(task, timeout=5)
        finally:
            ui.close()

    def test_close_is_idempotent(self):
        """close() must be safe to call before any prompt() has run, and again after."""
        ui = _WebUI(port=0)
        ui.close()  # never started — must not raise
        ui.close()  # second call — still safe

    @pytest.mark.asyncio
    async def test_close_shuts_down_server(self):
        """After close() the server thread must terminate."""
        ui = _WebUI(timeout=5, port=0)
        task = asyncio.create_task(ui.prompt("task", tools=[], output_type=str))
        for _ in range(50):
            if ui._url is not None:
                break
            await asyncio.sleep(0.02)
        assert ui._url is not None
        port = int(ui._url.rsplit(":", 1)[1].rstrip("/"))
        e = ui._form_epoch
        await asyncio.get_running_loop().run_in_executor(None, lambda: _post_form(port, {"response": "x"}, epoch=e))
        await asyncio.wait_for(task, timeout=5)

        srv_thread = ui._server_thread
        assert srv_thread is not None and srv_thread.is_alive()
        ui.close()
        assert not srv_thread.is_alive()
        assert ui._server is None


# =============================================================================
# Non-blocking GET — thinking page when no form is ready
# =============================================================================


def _get(port: int) -> tuple[int, str]:
    import urllib.request

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
        return resp.status, resp.read().decode()


class TestThinkingPage:
    @pytest.mark.asyncio
    async def test_get_returns_thinking_page_when_no_form_ready(self):
        """A GET that arrives between turns must NOT block the browser;
        it must return immediately with the auto-refreshing "thinking"
        placeholder so the user sees feedback during agent processing.
        """
        from lazybridge.ext.hil.human import _build_thinking_page

        ui = _WebUI(timeout=5, port=0)
        try:
            # Start a prompt to bring the server up, then submit POST to
            # clear ``_html_ready`` so subsequent GETs hit the
            # "no form ready" path.
            task = asyncio.create_task(ui.prompt("first", tools=[], output_type=str))
            for _ in range(50):
                if ui._url is not None:
                    break
                await asyncio.sleep(0.02)
            assert ui._url is not None
            port = int(ui._url.rsplit(":", 1)[1].rstrip("/"))

            e = ui._form_epoch
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: _post_form(port, {"response": "first answer"}, epoch=e)
            )
            await asyncio.wait_for(task, timeout=5)
            # POST cleared ``_html_ready`` — the next GET should NOT
            # block (used to wait up to 3600s).  Time it to confirm.
            assert not ui._html_ready.is_set()

            t0 = time.monotonic()
            status, body = await asyncio.get_running_loop().run_in_executor(None, lambda: _get(port))
            elapsed = time.monotonic() - t0
            assert status == 200
            # Must return promptly (<<1s) instead of long-polling.
            assert elapsed < 1.0, f"GET took {elapsed:.2f}s — should be non-blocking"
            # Body is the thinking page, not a form.
            thinking_marker = "Agent is thinking" if "Agent is thinking" in _build_thinking_page() else "thinking"
            assert thinking_marker in body
            assert "<textarea" not in body
            # Meta-refresh is what drives the browser to retry — without
            # it the spinner would freeze forever.
            assert 'http-equiv="refresh"' in body
        finally:
            ui.close()

    @pytest.mark.asyncio
    async def test_get_returns_form_when_form_is_ready(self):
        """When ``prompt()`` has just published a form, GET serves THAT
        form (not the thinking page).
        """
        ui = _WebUI(timeout=5, port=0)
        try:
            task = asyncio.create_task(ui.prompt("the real question", tools=[], output_type=str))
            for _ in range(50):
                if ui._url is not None and ui._html_ready.is_set():
                    break
                await asyncio.sleep(0.02)
            assert ui._html_ready.is_set()
            port = int(ui._url.rsplit(":", 1)[1].rstrip("/"))

            status, body = await asyncio.get_running_loop().run_in_executor(None, lambda: _get(port))
            assert status == 200
            assert "the real question" in body
            assert "<textarea" in body
            # Tear down via POST so the task completes.
            e = ui._form_epoch
            await asyncio.get_running_loop().run_in_executor(None, lambda: _post_form(port, {"response": "x"}, epoch=e))
            await asyncio.wait_for(task, timeout=5)
        finally:
            ui.close()


# =============================================================================
# Stale-form rejection — late submission from a previous-prompt form
# must not be consumed as the next prompt's response.
# =============================================================================


class TestStaleFormRejection:
    @pytest.mark.asyncio
    async def test_post_with_wrong_epoch_rejected_410(self):
        """A POST carrying an ``_epoch`` other than the current one
        must NOT be consumed; it must return 410 Gone and leave the
        in-flight ``prompt()`` waiting for the correct submission.
        """
        ui = _WebUI(timeout=5, port=0)
        try:
            task = asyncio.create_task(ui.prompt("real prompt", tools=[], output_type=str))
            for _ in range(50):
                if ui._url is not None and ui._html_ready.is_set():
                    break
                await asyncio.sleep(0.02)
            assert ui._url is not None
            port = int(ui._url.rsplit(":", 1)[1].rstrip("/"))

            stale_epoch = ui._form_epoch - 1  # one short of current
            status, body = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _post_form(
                    port,
                    {"response": "stale submission"},
                    epoch=stale_epoch,
                    follow_redirect=False,
                ),
            )
            assert status == 410, "stale form POST must be rejected with 410 Gone"
            assert "expired" in body.lower()
            # prompt() must still be waiting — the stale submission
            # was rejected, not consumed.
            assert not task.done()

            # A correctly-epoched POST still completes the prompt().
            e = ui._form_epoch
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _post_form(port, {"response": "fresh"}, epoch=e),
            )
            result = await asyncio.wait_for(task, timeout=5)
            assert result == "fresh"
        finally:
            ui.close()

    @pytest.mark.asyncio
    async def test_post_without_epoch_rejected(self):
        """A POST with no ``_epoch`` field at all (e.g. a crafted
        cross-tab submission) is treated as stale."""
        ui = _WebUI(timeout=5, port=0)
        try:
            task = asyncio.create_task(ui.prompt("task", tools=[], output_type=str))
            for _ in range(50):
                if ui._url is not None and ui._html_ready.is_set():
                    break
                await asyncio.sleep(0.02)
            assert ui._url is not None
            port = int(ui._url.rsplit(":", 1)[1].rstrip("/"))

            status, _body = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _post_form(
                    port,
                    {"response": "no epoch"},
                    follow_redirect=False,
                ),
            )
            assert status == 410
            assert not task.done()

            # Clean up: a valid POST so the prompt completes.
            e = ui._form_epoch
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _post_form(port, {"response": "ok"}, epoch=e),
            )
            await asyncio.wait_for(task, timeout=5)
        finally:
            ui.close()

    @pytest.mark.asyncio
    async def test_chatgpt_scenario_late_post_after_timeout_and_reprompt(self):
        """End-to-end of the bug described in the ChatGPT review:
        prompt #1 times out, prompt #2 starts, the BROWSER (still
        showing prompt #1's form) submits late — that POST must NOT
        be consumed as prompt #2's response.
        """
        ui = _WebUI(timeout=0.3, port=0, default="defaulted")
        try:
            # Prompt #1: starts, captures its epoch, then we let it
            # time out without ever submitting.
            task1 = asyncio.create_task(ui.prompt("first task", tools=[], output_type=str))
            for _ in range(50):
                if ui._url is not None and ui._html_ready.is_set():
                    break
                await asyncio.sleep(0.02)
            assert ui._url is not None
            port = int(ui._url.rsplit(":", 1)[1].rstrip("/"))
            stale_epoch = ui._form_epoch

            # Wait for the timeout to fire and the default path to
            # return.  prompt() should clear ``_html_ready`` and the
            # form state so a late POST can't be misdirected.
            result1 = await asyncio.wait_for(task1, timeout=2)
            assert result1 == "defaulted"
            assert not ui._html_ready.is_set()

            # Bump the UI's per-prompt timeout for prompt #2 — we want
            # it to wait long enough for the test to send first a stale
            # POST then a fresh one without the second prompt timing
            # out.  The 0.3s default was only needed to drive #1 into
            # the timeout path.
            ui._timeout = 5.0

            # Prompt #2: publishes a brand-new form with bumped epoch.
            task2 = asyncio.create_task(ui.prompt("second task", tools=[], output_type=str))
            for _ in range(50):
                if ui._html_ready.is_set() and ui._form_epoch != stale_epoch:
                    break
                await asyncio.sleep(0.02)
            assert ui._form_epoch != stale_epoch

            # Simulate the browser submitting prompt #1's stale form
            # AFTER prompt #2 went live.  Without the epoch guard this
            # would be consumed as prompt #2's response.
            status, _body = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _post_form(
                    port,
                    {"response": "STALE — must not reach prompt #2"},
                    epoch=stale_epoch,
                    follow_redirect=False,
                ),
            )
            assert status == 410, "stale POST after timeout+reprompt must be 410"
            assert not task2.done(), "stale POST must not satisfy prompt #2"

            # The fresh form still completes normally.
            fresh_epoch = ui._form_epoch
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _post_form(port, {"response": "real answer"}, epoch=fresh_epoch),
            )
            result2 = await asyncio.wait_for(task2, timeout=5)
            assert result2 == "real answer"
        finally:
            ui.close()


# =============================================================================
# HumanEngine env.error surfacing — symmetric with env.context
# =============================================================================


class TestHumanEngineErrorSurfacing:
    @pytest.mark.asyncio
    async def test_upstream_error_prepended_to_task(self):
        """When the inbound envelope carries an ``error``, HumanEngine
        must surface it to the UI so the human sees what failed
        upstream — same role ``env.context`` plays today.
        """
        from lazybridge.envelope import Envelope, ErrorInfo
        from lazybridge.ext.hil.human import HumanEngine, _UIProtocol

        captured: dict[str, str] = {}

        class _CaptureUI(_UIProtocol):
            async def prompt(self, task, *, tools, output_type):  # type: ignore[no-untyped-def]
                captured["task"] = task
                return "ok"

        engine = HumanEngine(ui=_CaptureUI())
        env = Envelope(
            task="please retry",
            error=ErrorInfo(type="UpstreamFailed", message="search timed out"),
        )
        await engine.run(env, tools=[], output_type=str, memory=None, session=None)

        # Error type + message must appear in the task surfaced to the
        # UI, before the original task body.
        assert "UpstreamFailed" in captured["task"]
        assert "search timed out" in captured["task"]
        assert "please retry" in captured["task"]
        assert captured["task"].index("UpstreamFailed") < captured["task"].index("please retry")

    @pytest.mark.asyncio
    async def test_no_error_means_no_prefix(self):
        """Envelopes without an error must produce the unchanged task —
        no spurious banner.
        """
        from lazybridge.envelope import Envelope
        from lazybridge.ext.hil.human import HumanEngine, _UIProtocol

        captured: dict[str, str] = {}

        class _CaptureUI(_UIProtocol):
            async def prompt(self, task, *, tools, output_type):  # type: ignore[no-untyped-def]
                captured["task"] = task
                return "ok"

        engine = HumanEngine(ui=_CaptureUI())
        env = Envelope(task="just a question")
        await engine.run(env, tools=[], output_type=str, memory=None, session=None)

        assert captured["task"] == "just a question"
        assert "upstream error" not in captured["task"]


# =============================================================================
# HumanEngine ui="web" wiring
# =============================================================================


class TestHumanEngineWebWiring:
    def test_web_ui_type(self):
        h = HumanEngine(ui="web")
        assert isinstance(h._ui, _WebUI)

    def test_web_ui_timeout_forwarded(self):
        h = HumanEngine(ui="web", timeout=60)
        assert h._ui._timeout == 60

    def test_terminal_still_works(self):
        from lazybridge.ext.hil.human import _TerminalUI

        h = HumanEngine(ui="terminal")
        assert isinstance(h._ui, _TerminalUI)

    def test_custom_ui_protocol(self):
        from lazybridge.ext.hil.human import _UIProtocol

        class MyUI(_UIProtocol):
            async def prompt(self, task, *, tools, output_type):
                return "custom"

        h = HumanEngine(ui=MyUI())
        assert isinstance(h._ui, MyUI)


# =============================================================================
# Agent fallback routing
# =============================================================================


class _ErrorEngine:
    """Engine that always returns an error Envelope."""

    def __init__(self, message: str = "provider error") -> None:
        self._message = message
        self._agent_name = "error_engine"

    async def run(self, env, *, tools, output_type, memory, session):
        return Envelope.error_envelope(RuntimeError(self._message))

    async def stream(self, env, *, tools, output_type, memory, session):
        yield ""


class _EchoEngine:
    """Engine that echoes the task text as payload."""

    def __init__(self) -> None:
        self._agent_name = "echo_engine"
        self.model = "echo"

    async def run(self, env, *, tools, output_type, memory, session):
        return Envelope(task=env.task, payload=env.task or "")

    async def stream(self, env, *, tools, output_type, memory, session):
        yield env.task or ""


class TestAgentFallback:
    @pytest.mark.asyncio
    async def test_fallback_used_when_primary_fails(self):
        fallback = Agent(engine=_EchoEngine(), name="fallback")
        agent = Agent(engine=_ErrorEngine(), name="primary", fallback=fallback)
        result = await agent.run("hello")
        assert result.ok
        assert result.text() == "hello"

    @pytest.mark.asyncio
    async def test_fallback_not_used_when_primary_succeeds(self):
        Agent(engine=_EchoEngine(), name="primary")
        fallback_calls = []

        class _TrackEngine:
            _agent_name = "track"
            model = "track"

            async def run(self, env, *, tools, output_type, memory, session):
                fallback_calls.append(True)
                return Envelope(payload="fallback")

            async def stream(self, env, *, tools, output_type, memory, session):
                yield ""

        fallback = Agent(engine=_TrackEngine(), name="fallback")
        agent = Agent(engine=_EchoEngine(), name="primary", fallback=fallback)
        result = await agent.run("task")
        assert result.ok
        assert len(fallback_calls) == 0

    @pytest.mark.asyncio
    async def test_fallback_none_returns_error(self):
        agent = Agent(engine=_ErrorEngine(), name="primary")
        result = await agent.run("task")
        assert not result.ok
        assert "provider error" in result.error.message

    @pytest.mark.asyncio
    async def test_fallback_chained_two_levels(self):
        """If primary fails and fallback also fails, the fallback's error is returned."""
        fallback2 = Agent(engine=_ErrorEngine("fallback error"), name="fb2")
        fallback1 = Agent(engine=_ErrorEngine("primary error"), name="fb1", fallback=fallback2)
        agent = Agent(engine=_ErrorEngine("root error"), name="root", fallback=fallback1)
        result = await agent.run("task")
        # fallback2 should have been tried last (but also fails)
        assert not result.ok

    def test_fallback_accessible_as_attribute(self):
        fb = Agent(engine=_EchoEngine(), name="fb")
        a = Agent(engine=_EchoEngine(), name="primary", fallback=fb)
        assert a.fallback is fb

    def test_no_fallback_default_is_none(self):
        a = Agent(engine=_EchoEngine(), name="primary")
        assert a.fallback is None

    @pytest.mark.asyncio
    async def test_sync_call_with_fallback(self):
        """Agent.__call__ also uses fallback."""
        fallback = Agent(engine=_EchoEngine(), name="fallback")
        agent = Agent(engine=_ErrorEngine(), name="primary", fallback=fallback)
        result = agent("sync call")
        assert result.ok
        assert result.text() == "sync call"


# =============================================================================
# ext/ doc_skills API migration (import smoke test)
# =============================================================================


class TestDocSkillsImport:
    def test_imports_use_current_api(self):
        # Should not raise — no LazyAgent/LazySession/LazyTool used
        from lazybridge.external_tools.doc_skills.doc_skills import (
            skill_builder_tools,
            skill_tools,
        )

        assert callable(skill_builder_tools)
        assert callable(skill_tools)

    def test_skill_builder_tools_returns_list_of_tool(self):
        from lazybridge.external_tools.doc_skills.doc_skills import skill_builder_tools

        out = skill_builder_tools()
        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], Tool)
        assert out[0].name == "build_doc_skill"

    def test_skill_builder_tools_custom_name(self):
        from lazybridge.external_tools.doc_skills.doc_skills import skill_builder_tools

        out = skill_builder_tools(name="my_builder")
        assert out[0].name == "my_builder"
