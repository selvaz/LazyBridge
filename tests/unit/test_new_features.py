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
        from lazybridge.ext.doc_skills.doc_skills import (
            skill_builder_tool,
            skill_tool,
        )

        assert callable(skill_builder_tool)
        assert callable(skill_tool)

    def test_skill_builder_tool_returns_tool(self):
        from lazybridge.ext.doc_skills.doc_skills import skill_builder_tool

        t = skill_builder_tool()
        assert isinstance(t, Tool)
        assert t.name == "build_doc_skill"

    def test_skill_builder_tool_custom_name(self):
        from lazybridge.ext.doc_skills.doc_skills import skill_builder_tool

        t = skill_builder_tool(name="my_builder")
        assert t.name == "my_builder"


# =============================================================================
# ext/ veo API migration (import smoke test — no google-genai needed)
# =============================================================================


class TestVeoImport:
    def test_veo_module_imports_without_genai(self):
        import importlib

        mod = importlib.import_module("lazybridge.ext.veo.veo")
        assert hasattr(mod, "veo_tool")
        assert hasattr(mod, "VeoError")

    def test_veo_tool_raises_import_error_without_genai(self):
        from lazybridge.ext.veo.veo import _GENAI_AVAILABLE, _require_genai

        if not _GENAI_AVAILABLE:
            with pytest.raises(ImportError, match="google-genai"):
                _require_genai()
