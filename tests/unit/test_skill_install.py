"""python -m lazybridge.skill install: copy, replace our own install, refuse others."""

from __future__ import annotations

from lazybridge.skill import __main__ as cli


def test_install_copies_the_whole_skill_without_python_files(tmp_path):
    dest = tmp_path / "skills" / "lazybridge"
    assert cli.main(["install", "--dest", str(dest)]) == 0
    assert (dest / "SKILL.md").is_file()
    assert list((dest / "references").glob("*.md"))
    assert not list(dest.rglob("*.py"))
    assert (dest / cli.MARKER).is_file()


def test_reinstall_replaces_our_own_folder(tmp_path):
    dest = tmp_path / "lazybridge"
    cli.main(["install", "--dest", str(dest)])
    (dest / "stale.md").write_text("old", encoding="utf-8")
    assert cli.main(["install", "--dest", str(dest)]) == 0
    assert not (dest / "stale.md").exists()


def test_foreign_folder_is_refused_unless_forced(tmp_path, capsys):
    dest = tmp_path / "lazybridge"
    dest.mkdir()
    (dest / "mine.md").write_text("keep", encoding="utf-8")
    assert cli.main(["install", "--dest", str(dest)]) == 1
    assert (dest / "mine.md").read_text(encoding="utf-8") == "keep"
    assert "refused" in capsys.readouterr().err
    assert cli.main(["install", "--dest", str(dest), "--force"]) == 0
    assert not (dest / "mine.md").exists()


def test_default_targets_follow_the_assistants_found(tmp_path, monkeypatch):
    homes = {"claude": tmp_path / ".claude", "codex": tmp_path / ".codex"}
    homes["codex"].mkdir()
    monkeypatch.setattr(cli, "HOMES", homes)
    assert cli.main(["install"]) == 0
    assert (homes["codex"] / "skills" / "lazybridge" / "SKILL.md").is_file()
    assert not homes["claude"].exists()
