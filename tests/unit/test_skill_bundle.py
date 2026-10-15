"""The bundled skill is a short index plus reference files.

Guards the progressive-disclosure layout of ``lazybridge/skill``: the
index stays small enough to load cheaply, its frontmatter stays valid for
skill loaders, and every route in the index points at a shipped file.
"""

from __future__ import annotations

import re

from lazybridge.skill import reference_paths, skill_path

_MAX_INDEX_LINES = 150  # the index is meant to be ~80-120 lines
_MAX_DESCRIPTION_CHARS = 1024  # Claude skill frontmatter limit


def _index_text() -> str:
    return (skill_path() / "SKILL.md").read_text(encoding="utf-8")


def test_index_exists_and_stays_short() -> None:
    lines = _index_text().splitlines()
    assert lines, "SKILL.md is empty"
    assert len(lines) <= _MAX_INDEX_LINES, f"SKILL.md has {len(lines)} lines; move detail into references/*.md"


def test_frontmatter_name_and_description() -> None:
    text = _index_text()
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "SKILL.md must start with YAML frontmatter"
    front = match.group(1)
    assert re.search(r"^name: lazybridge$", front, re.MULTILINE)
    desc = front.split("description:", 1)[1]
    desc_text = " ".join(line.strip() for line in desc.splitlines() if line.strip() not in {"|", ""})
    assert 0 < len(desc_text) <= _MAX_DESCRIPTION_CHARS


def test_every_route_points_at_a_shipped_reference() -> None:
    text = _index_text()
    routed = set(re.findall(r"references/([A-Za-z0-9_.-]+\.md)", text))
    shipped = {p.name for p in reference_paths()}
    assert shipped, "no reference files shipped under lazybridge/skill/references/"
    assert routed == shipped, f"routed-only={sorted(routed - shipped)}, unrouted={sorted(shipped - routed)}"


def test_reference_files_are_non_empty_markdown() -> None:
    for path in reference_paths():
        body = path.read_text(encoding="utf-8")
        assert body.startswith("# "), f"{path.name} should start with a level-1 heading"
        assert len(body.splitlines()) > 10, f"{path.name} looks truncated"
