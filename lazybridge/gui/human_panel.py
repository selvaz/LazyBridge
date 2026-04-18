"""HumanInputPanel — human-in-the-loop prompts on the shared GuiServer.

The legacy :mod:`lazybridge.gui.human` module runs its own dedicated
``WebInputServer`` on a separate port so it is usable without importing
the rest of the GUI stack.  When you *are* using the shared GUI, a human
supervisor on its own tab is awkward — you want their prompt to show up
in the same sidebar as the agents and tools.

This module provides a :class:`HumanInputPanel` that registers on the
shared :class:`GuiServer` and a :func:`panel_input_fn` factory that
returns a drop-in ``input_fn`` for :class:`HumanAgent` /
:class:`SupervisorAgent`::

    from lazybridge import SupervisorAgent
    from lazybridge.gui.human_panel import panel_input_fn

    fn = panel_input_fn(name="reviewer")  # adds a panel to the shared server
    supervisor = SupervisorAgent(name="reviewer", input_fn=fn)
    # ... run the pipeline; the prompt shows up in the GUI sidebar.

    fn.panel.close()                       # optional: drop the panel when done
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from lazybridge.gui._global import get_server
from lazybridge.gui._panel import Panel

_logger = logging.getLogger(__name__)


class _SentinelClosed:
    """Marker pushed onto the response queue to unblock ask() on close()."""


_CLOSED = _SentinelClosed()


class HumanInputPanel(Panel):
    """Panel that turns a human-in-the-loop agent into a GUI sidebar entry."""

    kind = "human"

    def __init__(self, *, name: str, title: str | None = None, poll_interval: float = 30.0) -> None:
        self._name = name
        self._title = title or f"Human input — {name}"
        self._poll_interval = poll_interval
        self._prompt_lock = threading.Lock()
        self._current_prompt: dict[str, Any] | None = None
        self._current_seq = 0
        self._response_q: queue.Queue[str | _SentinelClosed] = queue.Queue(maxsize=1)
        self._closed = threading.Event()

    # ------------------------------------------------------------------
    # Panel protocol
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        return f"human-{self._name}"

    @property
    def label(self) -> str:
        pending = self._current_prompt is not None
        marker = " · waiting" if pending else ""
        return f"{self._name}{marker}"

    @property
    def group(self) -> str:
        return "Humans"

    def render_state(self) -> dict[str, Any]:
        with self._prompt_lock:
            if self._current_prompt is None:
                return {
                    "name": self._name,
                    "prompt": None,
                    "seq": self._current_seq,
                    "closed": self._closed.is_set(),
                }
            data = dict(self._current_prompt)
            data.update({
                "name": self._name,
                "closed": self._closed.is_set(),
            })
            return data

    def handle_action(self, action: str, args: dict[str, Any]) -> dict[str, Any]:
        if action == "submit":
            try:
                seq = int(args["seq"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("'seq' is required") from exc
            response = str(args.get("response", ""))
            accepted = self._deliver_response(seq, response)
            return {"accepted": accepted}
        return super().handle_action(action, args)

    # ------------------------------------------------------------------
    # Ask-side API (called by HumanAgent / SupervisorAgent input_fn)
    # ------------------------------------------------------------------

    def ask(
        self,
        prompt: str,
        *,
        timeout: float | None = None,
        quick_commands: list[str] | None = None,
    ) -> str:
        if self._closed.is_set():
            raise RuntimeError("HumanInputPanel is closed")

        with self._prompt_lock:
            self._current_seq += 1
            self._current_prompt = {
                "seq": self._current_seq,
                "prompt": prompt,
                "quick_commands": quick_commands or [],
            }
            try:
                while True:
                    self._response_q.get_nowait()
            except queue.Empty:
                pass

        start = time.monotonic()
        try:
            while True:
                remaining: float
                if timeout is None:
                    remaining = self._poll_interval
                else:
                    elapsed = time.monotonic() - start
                    if elapsed >= timeout:
                        raise TimeoutError(f"Web input timed out after {timeout}s")
                    remaining = min(self._poll_interval, timeout - elapsed)
                try:
                    resp = self._response_q.get(timeout=remaining)
                except queue.Empty:
                    _logger.debug("HumanInputPanel %r still waiting", self._name)
                    continue
                if isinstance(resp, _SentinelClosed):
                    raise RuntimeError("HumanInputPanel closed while waiting for input")
                return resp
        finally:
            with self._prompt_lock:
                self._current_prompt = None

    @property
    def input_fn(self) -> Callable[[str], str]:
        return lambda prompt: self.ask(prompt)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._response_q.put_nowait(_CLOSED)
        except queue.Full:
            pass

    # ------------------------------------------------------------------

    def _deliver_response(self, seq: int, response: str) -> bool:
        with self._prompt_lock:
            if self._current_prompt is None or self._current_prompt["seq"] != seq:
                return False
        try:
            self._response_q.put_nowait(response)
        except queue.Full:
            return False
        return True


def panel_input_fn(
    *,
    name: str = "human",
    title: str | None = None,
    open_browser: bool = True,
) -> Callable[[str], str]:
    """Register a :class:`HumanInputPanel` on the shared server.

    Returns a callable suitable for ``HumanAgent(input_fn=...)``.  The
    callable carries a ``.panel`` attribute pointing back at the panel so
    callers can tear it down::

        fn = panel_input_fn(name="reviewer")
        agent = HumanAgent(name="reviewer", input_fn=fn)
        ...
        fn.panel.close()
    """
    server = get_server(open_browser=open_browser)
    panel = HumanInputPanel(name=name, title=title)
    server.register(panel)

    def _fn(prompt: str) -> str:
        return panel.ask(prompt)

    _fn.panel = panel  # type: ignore[attr-defined]
    return _fn
