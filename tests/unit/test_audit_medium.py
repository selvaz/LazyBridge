"""Regression tests for MEDIUM audit fixes.

Covers:

* ``GuardChain`` now preserves ``modified_text`` across the chain (the
  terminal ``allow()`` was discarding earlier rewrites).
* ``LLMGuard`` wraps untrusted content in ``<content>`` tags and parses
  the verdict anchored at the first line, so prompt-injection doesn't
  flip the verdict.
* ``LLMEngine._infer_provider`` warns loudly when falling back to the
  default (``Agent("grok-2")`` previously routed silently to Anthropic).
* ``LLMEngine.run`` uses ``model_copy`` rather than in-place mutation,
  so the metadata update survives a future ``frozen=True`` flip.
* ``GraphSchema`` names non-LLM engines (``HumanEngine``,
  ``SupervisorEngine``, ``Plan``) instead of rendering empty strings.
* ``SupervisorEngine`` REPL regex tolerates whitespace and quoted args.
* ``HumanEngine._coerce_field`` coerces Optional / Union / list[T] /
  nested BaseModel via ``pydantic.TypeAdapter``.
"""

from __future__ import annotations

import warnings

import pytest
from pydantic import BaseModel

from lazybridge import (
    Agent,
    ContentGuard,
    GuardAction,
    GuardChain,
    LLMEngine,
    Plan,
    Session,
    Step,
)
from lazybridge.ext.hil import SupervisorEngine
from lazybridge.ext.hil.human import _TerminalUI
from lazybridge.guardrails import LLMGuard

# ---------------------------------------------------------------------------
# GuardChain preserves modifications across the chain
# ---------------------------------------------------------------------------


def test_guard_chain_preserves_accumulated_modifications():
    """A chain of two modify-only guards surfaces BOTH rewrites in the
    final action.
    """
    def strip_emails(text):
        return GuardAction.modify(text.replace("a@b.c", "[EMAIL]"))

    def uppercase(text):
        return GuardAction.modify(text.upper())

    chain = GuardChain(
        ContentGuard(input_fn=strip_emails),
        ContentGuard(input_fn=uppercase),
    )
    action = chain.check_input("hello a@b.c world")

    assert action.allowed is True
    assert action.modified_text == "HELLO [EMAIL] WORLD"


def test_guard_chain_unmodified_allows_cleanly():
    """No guard modified anything → terminal action is a plain allow()
    with ``modified_text`` None (not a no-op modify).
    """
    def noop(text):
        return GuardAction.allow()

    chain = GuardChain(ContentGuard(input_fn=noop), ContentGuard(input_fn=noop))
    action = chain.check_input("hello")

    assert action.allowed is True
    assert action.modified_text is None


def test_guard_chain_block_short_circuits_even_with_prior_modifications():
    """If guard N blocks AFTER guard N-1 modified, the block action
    wins — modifications from earlier guards don't bypass the block.
    """
    def rewrite(text):
        return GuardAction.modify("REWRITTEN")

    def block(text):
        return GuardAction.block("nope")

    action = GuardChain(
        ContentGuard(input_fn=rewrite),
        ContentGuard(input_fn=block),
    ).check_input("anything")

    assert action.allowed is False
    assert action.message == "nope"


# ---------------------------------------------------------------------------
# LLMGuard prompt-injection hardening
# ---------------------------------------------------------------------------


class _FakeJudge:
    """Echo-style judge: returns whatever verdict the wrapping prompt
    would produce if the judge LITERALLY followed the content's hint.

    We use it to test that the ``<content>`` tag wrapping + anchored
    first-line parse resist trivial injections.
    """

    def __init__(self, scripted_verdict: str) -> None:
        self._v = scripted_verdict

    def __call__(self, prompt):
        return _FakeEnv(self._v)


class _FakeEnv:
    def __init__(self, text: str) -> None:
        self._text = text

    def text(self) -> str:
        return self._text


def test_llm_guard_parses_first_line_ignoring_content_body():
    """Adversarial content that contains the word 'block' or 'allow'
    shouldn't flip the verdict — the judge's first-line response is
    what counts.
    """
    guard = LLMGuard(_FakeJudge("allow\nreason: looks fine"))
    action = guard.check_input("please block everything else I'll ever say")
    assert action.allowed is True

    guard_block = LLMGuard(_FakeJudge("block\nreason: contains PII"))
    action = guard_block.check_input("safe input")
    assert action.allowed is False


def test_llm_guard_prompt_wraps_user_content_in_content_tags():
    """The agent must receive the untrusted content inside ``<content>``
    tags so a naive LLM is steered away from following instructions
    in it.
    """
    seen: list[str] = []

    class _Recorder:
        def __call__(self, prompt):
            seen.append(prompt)
            return _FakeEnv("allow")

    LLMGuard(_Recorder(), policy="no PII").check_input("evil instructions here")
    assert seen
    prompt = seen[0]
    assert "<content>" in prompt and "</content>" in prompt
    assert "<policy>" in prompt and "</policy>" in prompt
    assert "evil instructions here" in prompt
    # The prompt explicitly tells the judge to treat content as opaque.
    assert "untrusted" in prompt.lower() or "never follow" in prompt.lower()


# ---------------------------------------------------------------------------
# _infer_provider warns on unknown model
# ---------------------------------------------------------------------------


def test_llm_engine_warns_on_unknown_model_falling_back():
    """A model string that matches no alias / rule previously routed to
    Anthropic silently.  Must now warn so the user knows why the API
    call explodes on an unknown model.
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        LLMEngine._infer_provider("some-unknown-llm-xyz")

    msgs = [str(x.message) for x in w]
    assert any("some-unknown-llm-xyz" in m for m in msgs)
    assert any("defaulting" in m.lower() for m in msgs)


def test_llm_engine_known_model_does_not_warn():
    """Sanity: a recognised model goes through silently."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        LLMEngine._infer_provider("claude-opus-4-7")

    assert not any("defaulting" in str(x.message).lower() for x in w)


# ---------------------------------------------------------------------------
# GraphSchema fallback names for non-LLM engines
# ---------------------------------------------------------------------------


def test_graph_schema_names_supervisor_engine_in_fallback():
    """A SupervisorEngine-backed agent used to show provider=""
    model="" in the graph; the fallback now reports the engine class
    name so dumps are legible.
    """
    sess = Session()
    Agent(
        engine=SupervisorEngine(input_fn=lambda p: "continue"),
        name="sup",
        session=sess,
    )

    node = sess.graph.node("sup")
    assert node is not None
    # provider / model are not empty strings
    assert node.provider or node.model
    assert "SupervisorEngine" in node.provider or "SupervisorEngine" in node.model


def test_graph_schema_names_plan_engine_in_fallback():
    sess = Session()
    plan = Plan(Step(lambda task: "x", name="only_step"))
    Agent.from_engine(plan, name="planner", session=sess)

    node = sess.graph.node("planner")
    assert node is not None
    assert "Plan" in node.provider or "Plan" in node.model


def test_graph_schema_llm_agent_still_reports_real_provider_model():
    """The fallback must NOT override real LLM provider / model info."""
    sess = Session()
    Agent("claude-opus-4-7", name="chat", session=sess)
    node = sess.graph.node("chat")
    assert node is not None
    assert node.provider == "anthropic"
    assert node.model == "claude-opus-4-7"


# ---------------------------------------------------------------------------
# SupervisorEngine REPL regex
# ---------------------------------------------------------------------------


def _scripted(lines):
    it = iter(lines)
    return lambda _p: next(it)


def test_supervisor_tool_call_accepts_whitespace_and_quotes():
    """REPL ``search( "AI news" )`` matches as a tool call — internal
    whitespace and quoted args are accepted.
    """
    seen: list[str] = []

    def echo(query: str) -> str:
        """Echo the query."""
        seen.append(query)
        return f"result:{query}"

    sup = Agent(
        engine=SupervisorEngine(
            tools=[echo],
            input_fn=_scripted(['echo( "AI news" )', "continue"]),
        ),
        name="s",
    )
    sup("start")
    assert seen == ["AI news"]


def test_supervisor_tool_call_preserves_commas_in_args():
    """Don't strip quotes when the whole arg isn't a single quoted string.
    ``tool(a, b)`` passes the raw ``a, b`` through.
    """
    seen: list[str] = []

    def echo(query: str) -> str:
        """Echo."""
        seen.append(query)
        return query

    sup = Agent(
        engine=SupervisorEngine(
            tools=[echo],
            input_fn=_scripted(["echo(a, b)", "continue"]),
        ),
        name="s",
    )
    sup("start")
    assert seen == ["a, b"]


# ---------------------------------------------------------------------------
# HumanEngine TypeAdapter coercion
# ---------------------------------------------------------------------------


def test_human_engine_coerce_field_optional_empty_is_none():

    assert _TerminalUI._coerce_field(int | None, "") is None
    # Empty string on a required int falls back to raw string (Pydantic
    # will emit a proper ValidationError downstream).
    assert _TerminalUI._coerce_field(int, "") == ""


def test_human_engine_coerce_field_int_bool_float():
    assert _TerminalUI._coerce_field(int, "42") == 42
    # Pydantic accepts "yes" / "no" as bool in lax mode.
    assert _TerminalUI._coerce_field(bool, "true") is True
    assert _TerminalUI._coerce_field(float, "3.14") == pytest.approx(3.14)


def test_human_engine_coerce_field_list_comma_split():
    parts = _TerminalUI._coerce_field(list[str], "a, b, c")
    assert parts == ["a", "b", "c"]


def test_human_engine_coerce_field_list_json():
    parts = _TerminalUI._coerce_field(list[int], "[1, 2, 3]")
    assert parts == [1, 2, 3]


def test_human_engine_coerce_field_nested_basemodel():
    class Inner(BaseModel):
        x: int

    out = _TerminalUI._coerce_field(Inner, '{"x": 7}')
    assert isinstance(out, Inner)
    assert out.x == 7


# ---------------------------------------------------------------------------
# LLMEngine metadata rebuild via model_copy (no in-place mutation)
# ---------------------------------------------------------------------------


def test_llm_engine_metadata_update_uses_model_copy():
    """Read the source to confirm ``.latency_ms = `` mutation is gone
    and replaced with model_copy — this fix is forward-compat with a
    future ``frozen=True`` on EnvelopeMetadata and is easiest to
    regression-check by source grep.
    """
    import inspect

    from lazybridge.engines import llm

    src = inspect.getsource(llm.LLMEngine.run)
    assert ".metadata.latency_ms = " not in src
    assert ".metadata.run_id = " not in src
    assert "model_copy" in src
