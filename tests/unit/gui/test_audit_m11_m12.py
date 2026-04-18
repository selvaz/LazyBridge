"""Regression tests for M11 (read_docs path traversal) and M12 (doc_skills symlink)."""

from __future__ import annotations

import sys

import pytest

# ---------------------------------------------------------------------------
# M11 — read_folder_docs refuses paths that escape base_dir
# ---------------------------------------------------------------------------


def _have_read_docs_deps() -> bool:
    try:
        from lazybridge.ext.read_docs import read_folder_docs  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _have_read_docs_deps(), reason="read_docs extras not installed")
def test_read_folder_docs_refuses_escape_from_base_dir(tmp_path):
    from lazybridge.ext.read_docs import read_folder_docs

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "ok.txt").write_text("hello")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("nope")

    # Path inside the sandbox works.
    ok = read_folder_docs(str(sandbox / "ok.txt"), base_dir=str(sandbox))
    assert "hello" in ok

    # Path outside the sandbox is refused.
    denied = read_folder_docs(str(outside / "secret.txt"), base_dir=str(sandbox))
    assert "refused" in denied
    assert "escapes base_dir" in denied
    # And the secret's content must not appear in the refusal.
    assert "nope" not in denied


@pytest.mark.skipif(not _have_read_docs_deps(), reason="read_docs extras not installed")
def test_read_folder_docs_without_base_dir_behaves_as_before(tmp_path):
    """No `base_dir` argument preserves the pre-fix contract."""
    from lazybridge.ext.read_docs import read_folder_docs

    (tmp_path / "greet.txt").write_text("ciao")
    out = read_folder_docs(str(tmp_path / "greet.txt"))
    assert "ciao" in out


# ---------------------------------------------------------------------------
# M12 — doc_skills.walk does not follow symlinks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="symlink behaviour is POSIX-specific here")
def test_doc_skills_iter_docs_skips_symlinks(tmp_path):
    from lazybridge.ext.doc_skills.doc_skills import _iter_docs

    root = tmp_path / "corpus"
    root.mkdir()
    (root / "real.md").write_text("real content")

    # A symlink inside the corpus pointing at a file OUTSIDE the corpus.
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.md"
    secret.write_text("secret payload")

    try:
        (root / "sym.md").symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("can't create symlinks on this filesystem")

    found = sorted(p.name for p in _iter_docs([root], include_exts=[".md"]))
    assert found == ["real.md"], f"_iter_docs should skip symlinks; got {found}"
