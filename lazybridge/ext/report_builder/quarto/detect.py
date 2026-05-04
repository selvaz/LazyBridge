"""Detect a Quarto CLI installation on ``$PATH``.

The Quarto exporter calls :func:`find_quarto` once on construction; tests
also use it as the gate for skipping smoke tests when Quarto isn't installed.
"""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache


class QuartoNotFoundError(RuntimeError):
    """Raised when the Quarto CLI is required but not on ``$PATH``.

    The exporter catches this and either falls back to the WeasyPrint path
    (when ``backend="auto"``) or surfaces a clear install hint to the user.
    """

    INSTALL_HINT = (
        "Quarto is not installed.  Install from https://quarto.org/docs/get-started/ "
        "(macOS: 'brew install quarto'; Linux: download the .deb / .rpm).  "
        "Or use backend='weasyprint' to render via the pure-Python fallback."
    )

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.INSTALL_HINT)


@lru_cache(maxsize=1)
def find_quarto() -> str | None:
    """Return the absolute path to the ``quarto`` binary, or ``None``."""
    return shutil.which("quarto")


@lru_cache(maxsize=1)
def quarto_version() -> str | None:
    """Return the running Quarto version string, or ``None`` if absent.

    Cached because we call this once per exporter construction and it shells
    out to a process — repeating doesn't help anyone.
    """
    path = find_quarto()
    if not path:
        return None
    try:
        result = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=10, check=False
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def require_quarto() -> str:
    """Return the path to ``quarto`` or raise :class:`QuartoNotFoundError`."""
    path = find_quarto()
    if not path:
        raise QuartoNotFoundError()
    return path
