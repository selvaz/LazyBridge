"""The two-minute demo: engine discovery and the two demo steps, offline."""

from __future__ import annotations

from lazybridge import demo


class _Reply:
    def __init__(self, text: str) -> None:
        self._text = text

    def text(self) -> str:
        return self._text


class _FakeAgent:
    calls: list[dict] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        _FakeAgent.calls.append(kwargs)

    def __call__(self, prompt: str) -> _Reply:
        return _Reply(f"{self.kwargs['name']} answered")


def _engines(monkeypatch, usable: dict[str, bool]) -> None:
    monkeypatch.setattr(
        demo,
        "ENGINES",
        {
            name: (lambda ok=ok: ok, f"enable {name}", lambda system=None, n=name: (n, system))
            for name, ok in usable.items()
        },
    )
    monkeypatch.setattr(demo, "Agent", _FakeAgent)
    _FakeAgent.calls = []


def test_no_engine_prints_hints_and_fails(monkeypatch, capsys):
    _engines(monkeypatch, {"a": False, "b": False})
    assert demo.main() == 1
    out = capsys.readouterr().out
    assert "enable a" in out and "enable b" in out
    assert _FakeAgent.calls == []


def test_one_engine_answers_without_review(monkeypatch, capsys):
    _engines(monkeypatch, {"a": True, "b": False})
    assert demo.main() == 0
    out = capsys.readouterr().out
    assert "a answered" in out and "Enable a second engine" in out
    assert [c["name"] for c in _FakeAgent.calls] == ["a"]
    assert _FakeAgent.calls[0]["tools"] == [demo.lazybridge_facts]


def test_two_engines_write_and_review(monkeypatch, capsys):
    _engines(monkeypatch, {"a": True, "b": True})
    assert demo.main() == 0
    out = capsys.readouterr().out
    assert "a writes, b reviews" in out
    writer = _FakeAgent.calls[-1]
    reviewer = _FakeAgent.calls[-2]
    assert writer["name"] == "a" and writer["verify"].kwargs is reviewer
    assert reviewer["engine"] == ("b", demo.REVIEW)


def test_tool_is_deterministic():
    assert "Agent = Engine + Tools + State" in demo.lazybridge_facts()
