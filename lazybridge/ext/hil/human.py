"""HumanEngine — human-in-the-loop engine with terminal and web UI."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal

from lazybridge.engines.base import resolve_agent_name
from lazybridge.envelope import Envelope, EnvelopeMetadata
from lazybridge.session import EventType

if TYPE_CHECKING:
    from lazybridge.memory import Memory
    from lazybridge.session import Session
    from lazybridge.tools import Tool


def _format_attachments(images: list[Any] | None, audio: Any | None) -> str:
    """Render a one-line attachment descriptor for HIL surfaces.

    Humans can't read base64 — and showing the URL of every image is
    noisy when there are many.  This emits a short summary the user
    can rely on to decide whether to ask the agent for clarification.
    Empty inputs return an empty string so the caller can ``if hint:``
    cleanly without an extra newline.
    """
    parts: list[str] = []
    if images:
        descs: list[str] = []
        for i, img in enumerate(images):
            if getattr(img, "url", None):
                descs.append(f"#{i + 1} {img.media_type} ({img.url})")
            elif getattr(img, "base64_data", None):
                descs.append(f"#{i + 1} {img.media_type} (~{len(img.base64_data) * 3 // 4} bytes inline)")
        if descs:
            parts.append("[attached images: " + "; ".join(descs) + "]")
    if audio is not None:
        if getattr(audio, "url", None):
            parts.append(f"[attached audio: {audio.media_type} ({audio.url})]")
        elif getattr(audio, "base64_data", None):
            parts.append(f"[attached audio: {audio.media_type} (~{len(audio.base64_data) * 3 // 4} bytes inline)]")
    return "\n".join(parts)


class _UIProtocol:
    """Minimal protocol for custom UI adapters."""

    async def prompt(self, task: str, *, tools: list[Any], output_type: type) -> str:
        raise NotImplementedError


class _TerminalUI(_UIProtocol):
    def __init__(self, timeout: float | None = None, default: str | None = None) -> None:
        self._timeout = timeout
        self._default = default

    async def prompt(self, task: str, *, tools: list[Any], output_type: type) -> str:
        from pydantic import BaseModel

        print(f"\n[Human Input Required]\n{task}")

        if tools:
            tool_names = [t.name for t in tools]
            print(f"Available actions: {', '.join(tool_names)}")

        if issubclass(output_type, BaseModel) if isinstance(output_type, type) else False:
            return await self._prompt_model(output_type)

        prompt_str = "Your response: "
        # ``get_running_loop`` is the 3.10+ forward-compatible primitive;
        # ``get_event_loop`` is deprecated and errors on 3.13+ when no
        # loop is already running.  ``prompt`` is always awaited from an
        # active loop, so this is safe.
        loop = asyncio.get_running_loop()
        if self._timeout:
            try:
                fut = loop.run_in_executor(None, input, prompt_str)
                return await asyncio.wait_for(fut, timeout=self._timeout)
            except TimeoutError:
                if self._default is not None:
                    print(f"[Timeout — using default: {self._default!r}]")
                    return self._default
                raise
        else:
            return await loop.run_in_executor(None, input, prompt_str)

    async def _prompt_model(self, model_type: type) -> str:
        """Prompt each field of a Pydantic model and coerce via TypeAdapter.

        On ``ValidationError`` the field is re-prompted up to
        ``_MAX_FIELD_RETRIES`` times with the validator's error message
        shown, so a typo (``"abc"`` for an ``int``) is caught
        interactively instead of falling through to a ``BaseModel``
        validation crash at the end of the form.
        """
        import json

        print(f"Please fill in the following fields for {model_type.__name__}:")
        data: dict[str, Any] = {}
        for field_name, field_info in getattr(model_type, "model_fields", {}).items():
            annotation = field_info.annotation or str
            type_label = getattr(annotation, "__name__", str(annotation))
            data[field_name] = await self._prompt_field(field_name, annotation, type_label)
        return json.dumps(data, default=str)

    _MAX_FIELD_RETRIES = 3

    async def _prompt_field(self, field_name: str, annotation: Any, type_label: str) -> Any:
        from pydantic import ValidationError

        loop = asyncio.get_running_loop()
        last_exc: str | None = None
        for _attempt in range(self._MAX_FIELD_RETRIES):
            prefix = f"  {field_name} ({type_label}): "
            if last_exc is not None:
                prefix = f"  [invalid — {last_exc}]\n{prefix}"
            raw = await loop.run_in_executor(None, input, prefix)
            try:
                return self._coerce_field_strict(annotation, raw)
            except ValidationError as exc:
                # Compact single-line summary of the first error.
                errors = exc.errors()
                if errors:
                    err = errors[0]
                    last_exc = f"{err.get('msg', 'invalid')} ({err.get('type', '?')})"
                else:
                    last_exc = "invalid"
        # Out of retries — return whatever _coerce_field would have
        # produced in lenient mode (matches the previous behaviour).
        return self._coerce_field(annotation, raw)

    @staticmethod
    def _coerce_field_strict(annotation: Any, raw: str) -> Any:
        """Same as ``_coerce_field`` but re-raises ValidationError.

        Used by the interactive loop so the human is re-prompted with
        a readable error message instead of ending up with a string
        payload that fails Pydantic at the end of the form.
        """
        import json

        from pydantic import TypeAdapter

        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", ())
        if raw.strip() == "" and type(None) in args:
            return None
        if (origin is list or annotation is list) and not raw.lstrip().startswith("["):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            inner = args[0] if args else str
            return TypeAdapter(list[inner]).validate_python(parts)  # type: ignore[valid-type]
        trimmed = raw.strip()
        if (trimmed and trimmed[0] in "{[") or trimmed in ("true", "false", "null"):
            parsed = json.loads(trimmed)
            return TypeAdapter(annotation).validate_python(parsed)
        return TypeAdapter(annotation).validate_python(raw)

    @staticmethod
    def _coerce_field(annotation: Any, raw: str) -> Any:
        """Coerce a CLI-entered string to ``annotation`` via TypeAdapter.

        Fallback order:
        1. Empty string ⇒ ``None`` iff the annotation accepts None.
        2. ``json.loads`` if the raw string looks like JSON (``{``, ``[``,
           ``true``/``false``/``null``, or a number).  Lets users paste
           lists / nested objects verbatim.
        3. Comma-split if the annotation is a list-ish origin and the
           input doesn't start with ``[``.
        4. ``TypeAdapter(annotation).validate_python(raw)`` — Pydantic's
           native coercion (handles int / float / bool / datetime /
           Optional / Union / Enum / ...).
        5. On any failure, fall back to the raw string; the outer
           ``BaseModel(**data)`` will emit a clear ValidationError.
        """
        import json

        from pydantic import TypeAdapter, ValidationError

        # Optional + empty → None.
        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", ())
        if raw.strip() == "" and type(None) in args:
            return None

        # Comma-list sugar for list[T] when the user didn't type JSON.
        if (origin is list or annotation is list) and not raw.lstrip().startswith("["):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            inner = args[0] if args else str
            try:
                return TypeAdapter(list[inner]).validate_python(parts)  # type: ignore[valid-type]
            except ValidationError:
                return parts

        # Try JSON first for everything that could be complex.
        trimmed = raw.strip()
        if (trimmed and trimmed[0] in "{[") or trimmed in ("true", "false", "null"):
            try:
                parsed = json.loads(trimmed)
                return TypeAdapter(annotation).validate_python(parsed)
            except (json.JSONDecodeError, ValidationError, TypeError):
                pass  # fall through to plain-string validation

        # Final: let Pydantic handle the raw string (int "42", bool "yes", …).
        try:
            return TypeAdapter(annotation).validate_python(raw)
        except (ValidationError, TypeError):
            return raw


def _build_web_form(
    task: str,
    tools: list[Any],
    is_model: bool,
    fields: dict[str, Any],
    *,
    epoch: int = 0,
) -> str:
    """Return a self-contained HTML form for the human-input web UI.

    ``epoch`` is embedded as a hidden ``_epoch`` field so the POST
    handler can detect submissions from a stale form (e.g. one rendered
    for a previous ``prompt()`` whose response timed out before a later
    ``prompt()`` arrived) and reject them instead of decoding them as
    the new prompt's response.
    """
    import html as _html

    tool_section = ""
    if tools:
        names = ", ".join(t.name for t in tools if hasattr(t, "name"))
        tool_section = f"<p><strong>Available actions:</strong> {_html.escape(names)}</p>"

    if is_model:
        rows = []
        for fname, finfo in fields.items():
            ann = finfo.annotation or str
            label = getattr(ann, "__name__", str(ann))
            rows.append(
                f'<tr><td><label for="{fname}">'
                f"{_html.escape(fname)} <small>({_html.escape(label)})</small>"
                f"</label></td>"
                f'<td><input type="text" id="{fname}" name="{fname}" '
                f'style="width:400px" autocomplete="off"></td></tr>'
            )
        form_body = f"<table>{''.join(rows)}</table>"
    else:
        form_body = '<textarea name="response" rows="8" style="width:100%;max-width:700px"></textarea>'

    task_escaped = _html.escape(task).replace("\n", "<br>")
    epoch_field = f'<input type="hidden" name="_epoch" value="{int(epoch)}">'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LazyBridge — Human Input</title>
<style>
  body {{font-family:system-ui,sans-serif;max-width:800px;margin:40px auto;padding:0 20px;}}
  h2 {{color:#333;}} pre {{background:#f5f5f5;padding:12px;border-radius:4px;white-space:pre-wrap;}}
  table {{border-collapse:collapse;width:100%;}} td {{padding:6px 8px;}}
  input,textarea {{border:1px solid #ccc;padding:6px;border-radius:3px;font-size:1em;}}
  button {{margin-top:14px;padding:10px 24px;background:#0066cc;color:#fff;
           border:none;border-radius:4px;font-size:1em;cursor:pointer;}}
  button:hover {{background:#0052a3;}}
</style>
</head>
<body>
<h2>Human Input Required</h2>
<pre>{task_escaped}</pre>
{tool_section}
<form method="POST" action="/">
{epoch_field}
{form_body}
<br><button type="submit">Submit</button>
</form>
</body>
</html>"""


def _build_stale_form_page() -> str:
    """HTML returned (with 410 Gone) when a POST arrives carrying an
    ``_epoch`` that no longer matches the server's current form.

    Auto-redirects back to ``/`` so the browser refreshes onto the
    current form (or the "thinking" placeholder if no form is ready).
    The 410 status is the audit-trail signal — humans see the auto-
    redirect and the page text, but automated tests / log readers see
    a clear "stale submission" exit code.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=/">
<title>LazyBridge — form expired</title>
<style>body {font-family:system-ui,sans-serif;max-width:800px;margin:40px auto;padding:0 20px;color:#a00;}</style>
</head>
<body>
<h2>This form has expired.</h2>
<p>Redirecting you to the current prompt…</p>
</body>
</html>"""


def _build_thinking_page() -> str:
    """HTML served by GET when no form is ready (agent is processing).

    Uses ``<meta http-equiv="refresh">`` to poll every 500 ms; when the
    next ``prompt()`` publishes a form, the next refresh picks it up.
    No JavaScript — the page degrades gracefully in any browser.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0.5">
<title>LazyBridge — waiting</title>
<style>
  body {font-family:system-ui,sans-serif;max-width:800px;margin:40px auto;padding:0 20px;color:#555;}
  h2 {color:#0066cc;}
  .spinner {display:inline-block;width:18px;height:18px;border:3px solid #e0e0e0;
            border-top-color:#0066cc;border-radius:50%;animation:spin 1s linear infinite;
            vertical-align:middle;margin-right:10px;}
  @keyframes spin {to {transform:rotate(360deg);}}
</style>
</head>
<body>
<h2><span class="spinner"></span>Agent is thinking…</h2>
<p>This page will refresh automatically when the next prompt is ready.</p>
</body>
</html>"""


class _WebUI(_UIProtocol):
    """Minimal web UI for HumanEngine: serves a form on localhost, waits for submission.

    Uses only the standard library (http.server, threading, webbrowser).
    The OS picks a free port (port=0) unless an explicit port is supplied.

    The HTTP server is started lazily on the first ``prompt()`` call and
    **reused across subsequent calls** on the same instance — so the
    browser tab stays on the same URL throughout a multi-turn session
    (e.g. a ``Plan`` that routes back to a HIL step).  After each POST,
    the response is a 303 redirect to ``/``; the resulting GET long-polls
    until the next ``prompt()`` arrives, so the browser auto-refreshes
    onto the next form with no user interaction between turns.

    Call :meth:`close` to shut the server down explicitly; otherwise the
    serving thread is a daemon and dies with the process.
    """

    def __init__(self, timeout: float | None = None, port: int = 0, default: str | None = None) -> None:
        import queue as _queue
        import threading as _threading

        self._timeout = timeout
        self._port = port
        self._default = default
        # Server state — lazily created on first ``prompt()``.
        self._server: Any = None
        self._server_thread: _threading.Thread | None = None
        self._url: str | None = None
        self._opened_browser: bool = False
        # Per-turn synchronisation.  ``_pending_html`` holds the form to
        # serve on the next GET; ``_html_ready`` signals that a fresh
        # ``prompt()`` has set it.  ``_response_q`` carries POST results
        # back to the awaiting ``prompt()``.
        self._lock = _threading.Lock()
        self._pending_html: str | None = None
        self._is_model: bool = False
        self._fields: dict[str, Any] = {}
        # Monotonic per-prompt counter.  Embedded into each form as a
        # hidden ``_epoch`` field and re-checked by the POST handler so
        # a late submission from a stale form (e.g. the previous
        # ``prompt()`` timed out, then a new ``prompt()`` published a
        # different form, then the human submitted the OLD form) is
        # rejected with 410 instead of being decoded as the new
        # prompt's response.
        self._form_epoch: int = 0
        self._html_ready = _threading.Event()
        self._response_q: _queue.Queue[str] = _queue.Queue()

    def close(self) -> None:
        """Shut down the HTTP server and join its thread.

        Idempotent; safe to call if the server was never started.
        """
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:
                pass
            self._server = None
        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)
        self._server_thread = None

    async def prompt(self, task: str, *, tools: list[Any], output_type: type) -> str:
        import json
        import queue as _queue
        import threading
        import urllib.parse
        import webbrowser
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        from pydantic import BaseModel

        is_model = isinstance(output_type, type) and issubclass(output_type, BaseModel)
        fields: dict[str, Any] = getattr(output_type, "model_fields", {}) if is_model else {}

        # Publish the new form for the next GET (which may be a redirect
        # follow-up from a previous POST, already waiting on
        # ``_html_ready``).  Capture is_model/fields too so the POST
        # handler decodes against the SAME shape that was rendered.
        # Bump the epoch so any in-flight submission from a stale form
        # (rendered for a previous timed-out prompt) is rejected with
        # 410 instead of being decoded as this prompt's response.
        with self._lock:
            self._form_epoch += 1
            epoch = self._form_epoch
            self._pending_html = _build_web_form(task, tools, is_model, fields, epoch=epoch)
            self._is_model = is_model
            self._fields = fields
        # Drain any stale POST result left from a previously-timed-out
        # turn so this call never returns a response that belongs to an
        # earlier prompt.
        while not self._response_q.empty():
            try:
                self._response_q.get_nowait()
            except Exception:
                break
        self._html_ready.set()

        if self._server is None:
            ui = self  # captured by handler closure — handlers are short-lived

            class _Handler(BaseHTTPRequestHandler):
                def do_GET(self) -> None:
                    # Non-blocking: when no form is ready (agent is
                    # processing the previous turn), serve the
                    # "thinking" page with a meta-refresh that pulls
                    # the next form once it arrives.  Blocking the GET
                    # used to leave the browser with a hung request and
                    # no visible feedback during long pipeline steps.
                    if ui._html_ready.is_set():
                        with ui._lock:
                            html_now = ui._pending_html or ""
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.end_headers()
                        self.wfile.write(html_now.encode())
                    else:
                        body = _build_thinking_page().encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.end_headers()
                        self.wfile.write(body)

                def do_POST(self) -> None:
                    length = int(self.headers.get("Content-Length", 0))
                    raw = self.rfile.read(length).decode("utf-8", errors="replace")
                    data = dict(urllib.parse.parse_qsl(raw, keep_blank_values=True))
                    # Reject submissions from a stale form (epoch
                    # mismatch).  Without this guard, a POST from a
                    # form rendered for a previous ``prompt()`` whose
                    # response timed out could be decoded against the
                    # NEW prompt's ``_is_model``/``_fields`` and
                    # consumed as that prompt's response.
                    try:
                        posted_epoch = int(data.get("_epoch", "-1"))
                    except (TypeError, ValueError):
                        posted_epoch = -1
                    with ui._lock:
                        current_epoch = ui._form_epoch
                        is_model_now = ui._is_model
                        fields_now = ui._fields
                    if posted_epoch != current_epoch or not ui._html_ready.is_set():
                        body = _build_stale_form_page().encode()
                        self.send_response(410)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    if is_model_now:
                        result = json.dumps({k: data.get(k, "") for k in fields_now})
                    else:
                        result = data.get("response", "")
                    # Clear the ready-flag so the redirect GET below
                    # blocks until the next ``prompt()`` arrives.
                    ui._html_ready.clear()
                    # 303 → GET / makes the browser auto-refresh onto
                    # the next form once it's published.  No user action
                    # required between turns.
                    self.send_response(303)
                    self.send_header("Location", "/")
                    self.end_headers()
                    ui._response_q.put(result)

                def log_message(self, *_args: Any) -> None:  # suppress server logs
                    pass

            self._server = ThreadingHTTPServer(("127.0.0.1", self._port), _Handler)
            actual_port = self._server.server_address[1]
            self._url = f"http://127.0.0.1:{actual_port}/"
            self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._server_thread.start()

        if not self._opened_browser and self._url is not None:
            print(f"\n[Human Input Required — Web UI]\nOpen: {self._url}\n")
            try:
                webbrowser.open(self._url)
            except Exception:
                pass  # browser launch is best-effort
            self._opened_browser = True

        timeout = self._timeout or 3600.0
        loop = asyncio.get_running_loop()
        try:
            # Two timeouts race here: ``queue.get(timeout=)`` raises
            # ``queue.Empty`` and ``asyncio.wait_for`` raises
            # ``TimeoutError``; whichever fires first wins.  Both are
            # legitimate "no response" signals — treat them identically
            # by catching both in the except branch below.
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._response_q.get(timeout=timeout)),
                timeout=timeout,
            )
        except (TimeoutError, _queue.Empty) as exc:
            # Invalidate the current form so the persistent server
            # stops serving the now-orphaned page.  Without this clear,
            # a subsequent ``prompt()`` in the same session could be
            # answered by a late submission of the stale form, decoded
            # against the new prompt's ``_is_model``/``_fields``.  The
            # epoch check in ``do_POST`` is the second line of defence,
            # but invalidating ``_html_ready`` here means GETs in the
            # interim see the "thinking" placeholder rather than the
            # expired form.
            with self._lock:
                self._pending_html = None
                self._is_model = False
                self._fields = {}
            self._html_ready.clear()
            if self._default is not None:
                print(f"[Timeout — using default: {self._default!r}]")
                return self._default
            raise TimeoutError(f"Web UI timed out after {timeout}s without a response") from exc

        return result


class HumanEngine:
    """Presents the task to a human and returns their response as an Envelope.

    With output=PydanticModel, terminal prompts each field; web renders a form.
    Emits the same 8 event types as LLMEngine for transparent observability.
    """

    def __init__(
        self,
        *,
        timeout: float | None = None,
        ui: Literal["terminal", "web"] | _UIProtocol = "terminal",
        default: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.default = default
        if isinstance(ui, str):
            if ui == "terminal":
                self._ui: _UIProtocol = _TerminalUI(timeout=timeout, default=default)
            elif ui == "web":
                self._ui = _WebUI(timeout=timeout, default=default)
            else:
                raise ValueError(f"Unknown UI type: {ui!r}")
        else:
            self._ui = ui

    async def run(
        self,
        env: Envelope,
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
        store: Any | None = None,  # accepted-and-ignored — Plan checkpoint surface
        plan_state: Any | None = None,  # accepted-and-ignored — Plan checkpoint surface
    ) -> Envelope:
        run_id = str(uuid.uuid4())
        t_start = time.monotonic()
        agent_name = resolve_agent_name(self, "human")

        if session:
            session.emit(EventType.AGENT_START, {"agent_name": agent_name, "task": env.task}, run_id=run_id)

        try:
            task_text = env.task or env.text()
            # Surface a prior step's error to the human, symmetric with
            # ``env.context`` below: a HIL placed after a fallible step
            # in a Plan acts as the natural recovery / retry surface,
            # but only if the human can SEE that something went wrong.
            # Without this, an error envelope flowing into HIL produces
            # a form with no indication of what failed upstream.
            if env.error is not None:
                err = env.error
                err_msg = getattr(err, "message", None) or str(err)
                err_type = getattr(err, "type", None) or err.__class__.__name__
                task_text = f"[upstream error — {err_type}] {err_msg}\n\n{task_text}"
            if env.context:
                task_text = f"{task_text}\n\nContext:\n{env.context}"
            # Multimodal: humans can't view base64 in a terminal, but
            # they need to know SOMETHING is attached to make an
            # informed decision.  Append a short descriptor — same shape
            # the SupervisorEngine REPL uses, so users see consistent
            # attachment hints across HIL surfaces.
            attachment_hint = _format_attachments(env.images, env.audio)
            if attachment_hint:
                task_text = f"{task_text}\n\n{attachment_hint}"

            raw = await self._ui.prompt(task_text, tools=tools, output_type=output_type)

            payload: Any = raw
            import json

            from pydantic import BaseModel, ValidationError

            if isinstance(output_type, type) and issubclass(output_type, BaseModel):
                try:
                    data = json.loads(raw) if raw.strip().startswith("{") else {"response": raw}
                    payload = output_type(**data)
                except (json.JSONDecodeError, ValidationError, TypeError) as coerce_exc:
                    # Coercion failed — the caller asked for a typed
                    # ``output_type``, the human's input didn't match,
                    # and we must NOT silently hand back a string
                    # payload that would break ``.payload.field``
                    # access downstream.  Emit the audit-trail event
                    # AND return an explicit error envelope so ``.ok``
                    # is False and the caller can react.
                    latency_ms = (time.monotonic() - t_start) * 1000
                    err_env: Envelope[Any] = Envelope(
                        task=env.task,
                        context=env.context,
                        payload=raw,  # still available via env.payload for debug
                        metadata=EnvelopeMetadata(
                            latency_ms=latency_ms,
                            run_id=run_id,
                        ),
                        error=_coerce_error(output_type, coerce_exc),
                    )
                    if session:
                        session.emit(
                            EventType.TOOL_ERROR,
                            {
                                "agent_name": agent_name,
                                "kind": "structured_output_coercion",
                                "output_type": getattr(output_type, "__name__", str(output_type)),
                                "error_type": type(coerce_exc).__name__,
                                "error": str(coerce_exc),
                            },
                            run_id=run_id,
                        )
                        session.emit(
                            EventType.AGENT_FINISH,
                            {"agent_name": agent_name, "error": str(coerce_exc)},
                            run_id=run_id,
                        )
                    return err_env

        except Exception as exc:
            error_env = Envelope.error_envelope(exc)
            if session:
                session.emit(EventType.AGENT_FINISH, {"agent_name": agent_name, "error": str(exc)}, run_id=run_id)
            return error_env

        latency_ms = (time.monotonic() - t_start) * 1000
        result: Envelope[Any] = Envelope(
            task=env.task,
            context=env.context,
            payload=payload,
            metadata=EnvelopeMetadata(latency_ms=latency_ms, run_id=run_id),
        )

        if session:
            # Emit a HIL_DECISION for the single human input so the audit
            # trail distinguishes human answers from LLM responses.
            session.emit(
                EventType.HIL_DECISION,
                {
                    "agent_name": agent_name,
                    "kind": "input",
                    "command": env.task or "",
                    "result": result.text()[:500],
                },
                run_id=run_id,
            )
            session.emit(EventType.AGENT_FINISH, {"agent_name": agent_name, "payload": result.text()}, run_id=run_id)

        if memory:
            task_str = env.task or ""
            memory.add(task_str, result.text())

        return result

    async def stream(
        self, env: Envelope, *, tools: list, output_type: type, memory: Any, session: Any
    ) -> AsyncIterator[str]:
        env_out = await self.run(env, tools=tools, output_type=output_type, memory=memory, session=session)
        yield env_out.text()


def _coerce_error(output_type: Any, exc: BaseException) -> Any:
    """Build an :class:`ErrorInfo` for a structured-output coercion failure.

    Using a dedicated factory keeps the ``ErrorInfo`` import local (no
    module-level cost when HumanEngine is imported but never used) and
    standardises the error type name across terminal and web flows.
    """
    from lazybridge.envelope import ErrorInfo

    type_name = getattr(output_type, "__name__", str(output_type))
    return ErrorInfo(
        type="StructuredOutputCoercionError",
        message=(f"Human input could not be coerced to {type_name}: {type(exc).__name__}: {exc}"),
        retryable=False,
    )
