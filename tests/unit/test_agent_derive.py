"""Tests for Agent.derive() — non-mutating clone with appended tools / overrides."""

from __future__ import annotations

from lazybridge import Agent, Step
from lazybridge.engines.llm import LLMEngine
from lazybridge.engines.plan import Plan
from lazybridge.tools import Tool


def test_derive_returns_new_agent_original_untouched():
    base = Agent(engine=LLMEngine("claude-opus-4-7"), name="base")
    extra = Tool.wrap(lambda x: x, name="echo")
    d = base.derive(tools=[extra])
    assert d is not base
    assert "echo" in d._tool_map
    assert "echo" not in base._tool_map
    assert d.name == base.name


def test_derive_appends_tools_not_replaces():
    t1 = Tool.wrap(lambda x: x, name="a")
    t2 = Tool.wrap(lambda x: x, name="b")
    base = Agent(engine=LLMEngine("claude-opus-4-7"), tools=[t1], name="base")
    d = base.derive(tools=[t2])
    assert "a" in d._tool_map and "b" in d._tool_map


def test_derive_overrides_replace():
    base = Agent(engine=LLMEngine("claude-opus-4-7"), name="base")
    d = base.derive(name="renamed")
    assert d.name == "renamed"
    assert base.name == "base"


def test_derive_revalidates_at_construction():
    r = Agent(engine=LLMEngine("claude-opus-4-7"), name="r")
    pipe = Agent(engine=Plan(Step("r")), tools=[r], name="pipe")
    d = pipe.derive(tools=[Tool.wrap(lambda x: x, name="echo")])
    assert "echo" in d._tool_map


def test_derive_no_tools_arg_is_pure_clone():
    t = Tool.wrap(lambda x: x, name="existing")
    base = Agent(engine=LLMEngine("claude-opus-4-7"), tools=[t], name="base")
    d = base.derive()
    assert d is not base
    assert "existing" in d._tool_map
    assert d.name == base.name


def test_derive_does_not_mutate_sources():
    base = Agent(engine=LLMEngine("claude-opus-4-7"), name="base", sources=["s1"])
    d = base.derive()
    d.sources.append("s2")
    assert base.sources == ["s1"]


def test_derive_original_tools_raw_untouched():
    t = Tool.wrap(lambda x: x, name="orig")
    base = Agent(engine=LLMEngine("claude-opus-4-7"), tools=[t], name="base")
    extra = Tool.wrap(lambda x: x, name="extra")
    base.derive(tools=[extra])
    assert len(base._tools_raw) == 1
    assert base._tools_raw[0] is t


def test_derive_does_not_mutate_engine_agent_name():
    base = Agent(engine=LLMEngine("claude-opus-4-7"), name="base")
    base.derive(name="alias")
    assert base.engine._agent_name == "base"


def test_derive_preserves_name_explicit_flag():
    implicit = Agent(engine=LLMEngine("claude-opus-4-7"))
    assert implicit._name_explicit is False
    d = implicit.derive()
    assert d._name_explicit is False

    explicit = Agent(engine=LLMEngine("claude-opus-4-7"), name="explicit")
    assert explicit.derive()._name_explicit is True

    assert implicit.derive(name="alias")._name_explicit is True
