"""Architectural guard: every ``docs/*.md`` path mentioned in source
code, comments, tests, or the changelog must resolve to a real file.

Why this exists
---------------

``mkdocs build --strict`` catches dead cross-links inside ``docs/``
but cannot see paths embedded in Python source files or top-level
markdown.  We had at least one regression where
``docs/guides/core-vs-ext.md`` was referenced from
``lazybridge/ext/__init__.py``, three test files, and the changelog
— but the doc file itself was missing.  The grep below pins that.

Scope
-----

* Scans every ``*.py`` file under the project (excluding the test
  itself), plus every top-level ``*.md`` file (README, CHANGELOG,
  CONTRIBUTING, SECURITY, IMPLEMENTATION).
* Extracts every occurrence of ``docs/<path>.md`` (with optional
  ``/`` inside the path), regardless of surrounding punctuation.
* Asserts the path resolves to an existing file under the repo's
  ``docs/`` tree.

What it does NOT catch
----------------------

* HTML-style links inside ``docs/`` itself (handled by mkdocs strict).
* URLs of the form ``https://lazybridge.com/...`` — those are out of
  scope (we don't own the lifetime guarantee on the published site).
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
THIS_FILE = pathlib.Path(__file__).resolve()

# Strict: the doc path must include at least one ``/`` segment so we
# don't false-match the literal string ``foo.md`` elsewhere.  The
# pattern intentionally avoids capturing URL forms — those are
# preceded by ``://``, which the negative-lookbehind rejects.
_DOC_PATH = re.compile(r"(?<![A-Za-z0-9.:/])(docs/[A-Za-z0-9_./-]+\.md)\b")


def _scannable_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for path in REPO_ROOT.rglob("*.py"):
        # Skip ourselves and any third-party trees that might land here.
        if path == THIS_FILE:
            continue
        if any(part in {".git", "site", ".venv", "venv"} for part in path.parts):
            continue
        files.append(path)
    for name in ("README.md", "CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md", "IMPLEMENTATION.md"):
        p = REPO_ROOT / name
        if p.exists():
            files.append(p)
    return files


def _doc_path_references(path: pathlib.Path) -> list[tuple[int, str]]:
    """Yield ``(lineno, "docs/...md")`` tuples for every match.

    URL-embedded forms (``https://lazybridge.com/docs/foo.md``) are
    rejected at the regex level rather than by a line-level substring
    filter: the negative lookbehind ``(?<![A-Za-z0-9.:/])`` blocks the
    ``/`` that always precedes ``docs/`` inside a URL path, so the
    regex never matches the URL form in the first place.  A
    line-level filter would also false-negative legitimate matches
    that happen to share a line with an unrelated URL.
    """
    hits: list[tuple[int, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return hits
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in _DOC_PATH.finditer(line):
            hits.append((lineno, m.group(1)))
    return hits


def test_regex_rejects_url_embedded_doc_paths(tmp_path: pathlib.Path) -> None:
    """The lookbehind in ``_DOC_PATH`` must reject ``docs/...md`` when
    it sits inside an ``https://lazybridge.com/...`` URL — that's a
    web link, not a repo path."""
    sample = tmp_path / "fake.py"
    sample.write_text("see https://lazybridge.com/docs/foo.md and also https://x/docs/y.md\n")
    assert _doc_path_references(sample) == []


def test_regex_still_matches_real_reference_on_a_line_with_a_url(tmp_path: pathlib.Path) -> None:
    """Mixed-content lines: a real ``docs/foo.md`` reference must
    still be captured even when the same line contains an unrelated
    URL.  Pins the latent bug the v1 substring filter introduced."""
    sample = tmp_path / "fake.py"
    sample.write_text("# See docs/concepts/mental-model.md (link: https://example.com/anything)\n")
    hits = _doc_path_references(sample)
    assert hits == [(1, "docs/concepts/mental-model.md")]


def test_every_doc_path_in_source_resolves_to_a_real_file() -> None:
    failures: list[str] = []
    for path in _scannable_files():
        for lineno, doc_path in _doc_path_references(path):
            target = REPO_ROOT / doc_path
            if not target.exists():
                rel = path.relative_to(REPO_ROOT)
                failures.append(f"  {rel}:{lineno} -> {doc_path} (missing)")
    if failures:
        pytest.fail(
            "Dangling docs/*.md references found.  Either create the "
            "target file or update the reference:\n" + "\n".join(failures)
        )
