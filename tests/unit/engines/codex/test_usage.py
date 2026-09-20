"""``docs/guides/full/codex-engine.md``'s "Reading the subscription quota"
section had zero test coverage anywhere in the repo (confirmed by grep during
an ecosystem-docs-auditor pass, 20/09/2026) -- unlike its sibling guides
(``control-plane.md``, ``durable-blackboard.md``), which execute every
Python code block in the guide as a test. That gap let two real
inaccuracies ship and sit undetected until a manual re-read caught them:
``CodexUsageWindow``'s documented field list omitted the real ``kind``
field, and the doc claimed a missing/unreachable ``codex`` binary surfaces
as ``RuntimeError`` when it actually raises ``FileNotFoundError`` (that
failure happens in ``codex_executable()``/the initial
``create_subprocess_exec()``, both OUTSIDE ``fetch_codex_usage()``'s own
try/except block).

Lesson applied: "verified against source" is not the same as "ran it" --
for a claim precise enough to be wrong in a checkable way, write a test
that actually exercises the real code path instead of trusting a re-read.
These tests exercise the real ``fetch_codex_usage``/``CodexUsageWindow``
without requiring a live, authenticated Codex installation.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from lazybridge.engines.codex import usage as usage_module
from lazybridge.engines.codex.usage import CodexUsageWindow, fetch_codex_usage

_GUIDE_PATH = Path(__file__).resolve().parents[4] / "docs" / "guides" / "full" / "codex-engine.md"


def _quota_section() -> str:
    text = _GUIDE_PATH.read_text(encoding="utf-8")
    match = re.search(r"## Reading the subscription quota\n(.*?)(?:\n## |\Z)", text, re.S)
    assert match is not None, "the guide's quota section heading moved or was renamed"
    return match.group(1)


def test_the_documented_import_path_is_real() -> None:
    """The guide's runnable snippet imports ``fetch_codex_usage`` from this
    exact module path -- if that import ever moved, every reader following
    the guide would hit a fresh ``ImportError`` on line 1, invisible from a
    read-only re-check of the docstring alone."""
    section = _quota_section()
    match = re.search(r"^from ([\w.]+) import (\w+)$", section, re.M)
    assert match is not None, "no import line found in the quota section's code block"
    module_path, name = match.group(1), match.group(2)
    assert module_path == "lazybridge.engines.codex.usage"
    assert getattr(usage_module, name) is fetch_codex_usage


def test_codexusagewindow_documented_fields_match_the_real_dataclass() -> None:
    """The exact bug found live 20/09/2026: the guide's field list said five
    fields, the dataclass has six (``kind`` was missing). Compared against
    ``dataclasses.fields()`` directly, not re-typed by hand, so it cannot
    silently drift again the way the prose version did."""
    section = _quota_section()
    match = re.search(r"Each is a `CodexUsageWindow` with (.+?)\.\n", section, re.S)
    assert match is not None, "no 'Each is a CodexUsageWindow with ...' sentence found"
    documented = set(re.findall(r"`(\w+)`", match.group(1)))
    real = {f.name for f in dataclasses.fields(CodexUsageWindow)}

    assert real, "the extraction itself must not silently find nothing"
    assert documented == real, f"doc vs dataclass drift -- documented={documented} real={real}"


async def test_a_missing_codex_binary_raises_filenotfounderror_not_runtimeerror(monkeypatch) -> None:
    """The exact second bug found live 20/09/2026: the guide claimed a
    missing/unreachable ``codex`` binary surfaces as ``RuntimeError``. It
    does not -- ``codex_executable()`` runs BEFORE ``fetch_codex_usage``'s
    try/except block, so its ``FileNotFoundError`` propagates unwrapped.
    Actually calling the function proves this, rather than re-reading where
    the try block starts and hoping the read was right a second time."""

    def _never_found() -> str:
        raise FileNotFoundError("codex CLI not found on PATH or in the Codex app install directory")

    monkeypatch.setattr(usage_module, "codex_executable", _never_found)

    with pytest.raises(FileNotFoundError):
        await fetch_codex_usage()


async def test_an_explicit_nonexistent_executable_also_raises_filenotfounderror(tmp_path) -> None:
    """Same failure mode via the other entry point: passing ``executable=``
    directly (bypassing ``codex_executable()`` entirely) hits the OS's own
    ``FileNotFoundError`` from ``create_subprocess_exec``, still outside the
    try block -- confirms the guide's exception-path claim holds for both
    ways a caller can reach a missing binary, not just one."""
    missing = tmp_path / "definitely-not-codex"

    with pytest.raises(FileNotFoundError):
        await fetch_codex_usage(executable=str(missing))
