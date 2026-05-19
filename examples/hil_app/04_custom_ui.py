"""How to plug a custom UI into HumanEngine via the ``_UIProtocol`` hook.

The ships-in-framework web UI (``ext/hil/human.py:_WebUI``) is
deliberately minimal: stdlib only, no markdown rendering, no CSRF
tokens, no streaming, no session persistence.  For production-grade
needs, the framework's extension point is :class:`_UIProtocol` —
anything implementing ``async def prompt(self, task, *, tools,
output_type) -> str`` is accepted by ``HumanEngine(ui=...)`` and gets
wired into Plans, sessions, and routing exactly like the built-in UIs.

This example shows a "file-watched" UI: it writes the prompt to a file
and waits for the human to write the response to a sibling file.
Trivial in itself — but the shape demonstrates that you can integrate
with Slack, a custom Streamlit page, a queue, or any other surface
without touching the framework.

Run:
    python examples/hil_app/04_custom_ui.py
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from lazybridge import Agent, Plan, Step
from lazybridge.ext.hil import HumanEngine
from lazybridge.testing import MockAgent


class FileWatchUI:
    """Prompt via ``prompt.txt``; await response via ``response.txt``.

    Any object exposing ``async def prompt(task, *, tools, output_type)``
    is accepted by HumanEngine; no inheritance or registration required.
    The framework guarantees ``tools`` and ``output_type`` are passed
    in, but a custom UI is free to ignore either.
    """

    def __init__(self, prompt_file: Path, response_file: Path, poll: float = 0.5) -> None:
        self._prompt_file = prompt_file
        self._response_file = response_file
        self._poll = poll

    async def prompt(self, task: str, *, tools: list[Any], output_type: type) -> str:
        self._prompt_file.write_text(task, encoding="utf-8")
        if self._response_file.exists():
            self._response_file.unlink()

        print(f"[FileWatchUI] wrote prompt to {self._prompt_file}")
        print(f"[FileWatchUI] write your answer to {self._response_file} (then it's consumed)")

        # Poll on the executor so we don't tie up the asyncio loop.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._wait_for_response)

    def _wait_for_response(self) -> str:
        deadline = time.monotonic() + 600.0
        while time.monotonic() < deadline:
            if self._response_file.exists():
                content = self._response_file.read_text(encoding="utf-8").strip()
                self._response_file.unlink()
                return content
            time.sleep(self._poll)
        raise TimeoutError(f"No response written to {self._response_file} within 10 minutes")


def main() -> None:
    tmp = Path("/tmp/lazybridge_filewatch")
    tmp.mkdir(exist_ok=True)
    ui = FileWatchUI(prompt_file=tmp / "prompt.txt", response_file=tmp / "response.txt")

    ask = Agent(engine=HumanEngine(ui=ui), name="ask")
    answer = MockAgent(["Acknowledged."], name="answer")

    pipeline = Agent(
        engine=Plan(
            Step(ask, task="What's the human's request?"),
            Step(answer),
        ),
        name="file_watched_assistant",
    )
    result = pipeline("(human will write to /tmp/lazybridge_filewatch/response.txt)")
    print("\n→", result.text())


if __name__ == "__main__":
    main()
