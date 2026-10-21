"""One place that decides whether lazybridge may open a browser window.

The web HIL UI (``HumanEngine(ui="web")``) and the visualizer open a browser
tab as a convenience. Under a test run that convenience escapes the test
process: every server is on a throwaway port, so each run leaves a screenful
of tabs pointing at dead ``127.0.0.1:<port>`` pages (seen on the owner's
desktop after unattended suite runs). The launch is therefore skipped when:

* ``PYTEST_CURRENT_TEST`` is set (pytest sets it for every running test), or
* ``LAZYBRIDGE_NO_BROWSER`` is ``1``/``true``/``yes`` (headless hosts, jobs).

The URL is always printed by the callers, so nothing is lost.
"""

from __future__ import annotations

import os
import webbrowser
from typing import Any

_TRUTHY = {"1", "true", "yes", "on"}


def browser_launch_allowed() -> bool:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return os.environ.get("LAZYBRIDGE_NO_BROWSER", "").strip().lower() not in _TRUTHY


def open_browser(url: str, **kwargs: Any) -> bool:
    """Open ``url`` unless suppressed; never raises (the launch is best-effort)."""
    if not browser_launch_allowed():
        return False
    try:
        return bool(webbrowser.open(url, **kwargs))
    except Exception:
        return False
