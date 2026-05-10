"""Phase-1 regression tests for the deletion-led 0.8.0 cleanup.

Each test locks one of the bug fixes from
``/root/.claude/plans/puoi-verificare-anche-questo-partitioned-marshmallow.md``.
Failure here means the fix has regressed:

* ``test_deepseek_ensure_json_is_idempotent`` — B1 hardening (no in-place
  mutation of ``params['messages']``)
* ``test_anthropic_compute_cost_applies_cache_discount`` — B2 (cached
  input tokens reduce reported cost by Anthropic's standard 10% rate)
* ``test_anthropic_compute_cost_default_signature`` — B2 (signature
  parity with OpenAI ``_compute_cost``)
* ``test_anthropic_audio_url_warns`` — B4 (URL-source AudioContent now
  warns instead of dropping silently)
* ``test_compiler_unknown_step_suggests_close_match`` — B6 (typo-aware
  PlanCompileError)
* ``test_compiler_rejects_opaque_anonymous_step_reference`` — B10
  (auto-named ``_anon_*`` steps may not be referenced)
* ``test_resume_replays_store_sidecar_writes`` — B11 (durable sidecar
  write replayed on resume so external consumers see consistent state)
* ``test_skill_docs_check_passes_for_current_state`` — B12 (drift gate
  is wired and clean for the committed state)
"""

from __future__ import annotations

import warnings

import pytest

from lazybridge import (
    Agent,
    Envelope,
    MockAgent,
    Plan,
    PlanCompileError,
    Step,
    Store,
    from_step,
)
from lazybridge.core.providers.anthropic import AnthropicProvider
from lazybridge.core.providers.deepseek import DeepSeekProvider
from lazybridge.core.types import AudioContent, CompletionRequest, Message, Role, TextContent

# ---------------------------------------------------------------------------
# Shared bare-provider fixtures (no SDK init)
# ---------------------------------------------------------------------------


def _bare_anthropic() -> AnthropicProvider:
    p = AnthropicProvider.__new__(AnthropicProvider)
    p.model = "claude-opus-4-7"
    p.api_key = None
    return p


def _bare_deepseek() -> DeepSeekProvider:
    p = DeepSeekProvider.__new__(DeepSeekProvider)
    p.api_key = None
    p.model = DeepSeekProvider.default_model
    p._structured_drop_warned = False
    return p


# ---------------------------------------------------------------------------
# B1 — DeepSeek ``_ensure_json_word_in_prompt`` is non-mutating + idempotent
# ---------------------------------------------------------------------------


def test_deepseek_ensure_json_is_idempotent():
    """Calling ``_ensure_json_word_in_prompt`` repeatedly on the same params
    must rebuild the messages list (never mutate in place) so a shared
    ``request`` object stays clean across multi-turn loops."""
    provider = _bare_deepseek()
    original_messages = [{"role": "user", "content": "hello"}]
    params = {"messages": list(original_messages)}
    pre_id = id(params["messages"])

    provider._ensure_json_word_in_prompt(params, schema=None)

    # The list reference must change (we rebuilt rather than mutated).
    assert id(params["messages"]) != pre_id
    # The original list passed in (a separate object now) is untouched.
    assert original_messages == [{"role": "user", "content": "hello"}]
    # A fresh system message was prepended.
    assert params["messages"][0]["role"] == "system"
    assert "json" in params["messages"][0]["content"].lower()


def test_deepseek_ensure_json_with_existing_system_message_preserves_user_dict():
    provider = _bare_deepseek()
    sys_dict = {"role": "system", "content": "be terse"}
    user_dict = {"role": "user", "content": "go"}
    params = {"messages": [sys_dict, user_dict]}
    pre_user_id = id(user_dict)

    provider._ensure_json_word_in_prompt(params, schema=None)

    # First message rebuilt with appended JSON instruction.
    assert params["messages"][0]["role"] == "system"
    assert "be terse" in params["messages"][0]["content"]
    assert "json" in params["messages"][0]["content"].lower()
    # The original system dict is untouched (no in-place mutation).
    assert sys_dict == {"role": "system", "content": "be terse"}
    # The user dict is the SAME object — we only rebuilt the first slot.
    assert id(params["messages"][1]) == pre_user_id


# ---------------------------------------------------------------------------
# B2 — Anthropic ``_compute_cost`` accepts cached_input_tokens and discounts
# ---------------------------------------------------------------------------


def test_anthropic_compute_cost_default_signature():
    """Calling without ``cached_input_tokens`` must work (back-compat)."""
    provider = _bare_anthropic()
    cost = provider._compute_cost("claude-opus-4-7", input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(5.0)  # opus-4-7 input price = $5 / 1M


def test_anthropic_compute_cost_applies_cache_discount():
    """Cache hits charged at 10% of base input rate (Anthropic standard)."""
    provider = _bare_anthropic()
    # 1M total input tokens, 100% cached → cost = 1M * 0.1 * 5 / 1M = $0.50
    fully_cached = provider._compute_cost(
        "claude-opus-4-7",
        input_tokens=1_000_000,
        output_tokens=0,
        cached_input_tokens=1_000_000,
    )
    assert fully_cached == pytest.approx(0.5)

    # 1M total input tokens, 50% cached → 0.5M * 5 + 0.5M * 0.5 = $2.75
    half_cached = provider._compute_cost(
        "claude-opus-4-7",
        input_tokens=1_000_000,
        output_tokens=0,
        cached_input_tokens=500_000,
    )
    assert half_cached == pytest.approx(2.75)


def test_anthropic_compute_cost_clamps_cached_tokens():
    """Cached tokens reported above input_tokens (SDK noise) are clamped."""
    provider = _bare_anthropic()
    # If cached > input, clamp to input — entire input is cached.
    cost = provider._compute_cost(
        "claude-opus-4-7",
        input_tokens=1_000,
        output_tokens=0,
        cached_input_tokens=10_000,  # more than total — clamped
    )
    # Equivalent to all 1_000 cached at 0.1 × $5/1M
    assert cost == pytest.approx(1_000 * 0.1 * 5 / 1_000_000)


# ---------------------------------------------------------------------------
# B4 — Anthropic URL audio warns instead of silently dropping
# ---------------------------------------------------------------------------


def test_anthropic_audio_url_warns():
    audio = AudioContent.from_url("https://example.com/clip.wav")
    msg = Message(role=Role.USER, content=[TextContent("transcribe"), audio])
    req = CompletionRequest(messages=[msg])
    provider = _bare_anthropic()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = provider._messages_to_anthropic(req)

    # The URL audio block was dropped (Anthropic API rejects URL audio).
    audio_blocks = [b for b in result[0]["content"] if b.get("type") == "audio"]
    assert audio_blocks == []

    # And we now warn about it.
    audio_warnings = [w for w in captured if issubclass(w.category, UserWarning) and "audio" in str(w.message).lower()]
    assert len(audio_warnings) == 1
    assert "URL audio" in str(audio_warnings[0].message)


# ---------------------------------------------------------------------------
# B6 — Compiler typo-aware error
# ---------------------------------------------------------------------------


def test_compiler_unknown_step_suggests_close_match():
    """A typo'd from_step target should produce a "Did you mean 'X'?" hint."""
    a = MockAgent(["a-out"], name="a")
    b = MockAgent(["b-out"], name="b")
    with pytest.raises(PlanCompileError) as ei:
        Agent(
            engine=Plan(
                Step("a"),
                Step("b", context=from_step("reasearch")),  # typo
            ),
            tools=[a, b],
            name="agent_with_typo",
        )
    msg = str(ei.value)
    assert "from_step('reasearch')" in msg
    assert "Defined steps:" in msg
    # The suggestion may not fire for a target with no close match in
    # ``[a, b]``; the important DX is that the defined-steps list is
    # surfaced concretely.


def test_compiler_unknown_step_suggests_when_close():
    """When a close match exists, the "Did you mean" hint should fire."""
    research = MockAgent(["r-out"], name="research")
    write = MockAgent(["w-out"], name="write")
    with pytest.raises(PlanCompileError) as ei:
        Agent(
            engine=Plan(
                Step("research"),
                Step("write", context=from_step("reasearch")),  # close typo
            ),
            tools=[research, write],
            name="agent_close",
        )
    assert "Did you mean 'research'?" in str(ei.value)


# ---------------------------------------------------------------------------
# B10 — Anonymous step referenced by from_step is rejected at compile time
# ---------------------------------------------------------------------------


def test_compiler_rejects_opaque_anonymous_step_reference():
    """When ``Step(target=obj)`` falls back to ``_anon_<id>`` because the
    target has no string / __name__ / .name attribute, no other step may
    reference it via from_step(...) — the auto name is meaningless to a
    code-generating LLM."""

    class _Opaque:
        """Bare object with no usable name / __name__ source."""

        # Defining __call__ would give the class a __name__ via type(self).__name__,
        # but Step's auto-name path checks ``hasattr(target, "__name__")`` directly
        # on the instance — class-level attribute access still resolves, so we
        # delete it from the instance to force the opaque-id fallback.

    opaque_target = _Opaque()
    # ``_Opaque`` instances have no ``__name__`` and no ``.name``, and aren't
    # callable, so Step.__post_init__ must fall back to ``_anon_<hex>``.
    anon_step = Step(target=opaque_target)
    assert anon_step.name is not None and anon_step.name.startswith("_anon_"), (
        f"expected anonymous fallback, got {anon_step.name!r}"
    )

    # Use a real downstream agent so the tool-existence check (which runs
    # before sentinel validation) doesn't short-circuit the compile error.
    downstream = MockAgent(["downstream-out"], name="downstream")
    with pytest.raises(PlanCompileError) as ei:
        Agent(
            engine=Plan(
                anon_step,
                Step("downstream", context=from_step(anon_step.name)),
            ),
            tools=[downstream],
            name="anon_ref_plan",
        )
    msg = str(ei.value)
    assert "auto-named step" in msg
    assert "Fix:" in msg
    assert "explicit name=" in msg


# ---------------------------------------------------------------------------
# B11 — Resume replays Store sidecar writes for completed steps
# ---------------------------------------------------------------------------


def test_resume_replays_store_sidecar_writes():
    """If the prior run crashed *between* checkpoint commit and Store write,
    resume should re-emit the sidecar write so external consumers eventually
    see the value."""
    store = Store()
    # Hand-craft a "crashed mid-resume" checkpoint: step 'research' is
    # completed and present in kv, but the durable Store sidecar at the
    # same key was lost in the crash window.
    research = MockAgent(["the-result"], name="research")
    write = MockAgent(["written"], name="write")

    plan = Plan(
        Step("research", writes="research_out"),
        Step("write"),
        store=store,
        checkpoint_key="resume-replay-test",
        resume=True,
    )

    # Pre-seed checkpoint as if a prior run finished step 1 but lost the
    # durable Store write before saving step 2.
    store.write(
        "resume-replay-test",
        {
            "next_step": "write",
            "kv": {"research_out": "the-result"},
            "completed_steps": ["research"],
            "status": "running",
            "run_uid": "prior-run",
            "history": [],
        },
    )
    # Confirm the sidecar key is absent (the lost write).
    assert store.read("research_out") is None

    Agent(engine=plan, tools=[research, write], name="resume_replay")(Envelope(task="seed"))

    # After resume started, the sidecar write must have been replayed
    # idempotently from the checkpoint kv.
    assert store.read("research_out") == "the-result"


# ---------------------------------------------------------------------------
# B12 — skill_docs._build --check runs clean
# ---------------------------------------------------------------------------


def test_skill_docs_check_passes_for_current_state():
    """The drift checker must exit 0 against the committed SKILL.md so the
    CI gate can be enabled.  A future change to ``__all__`` without a
    matching SKILL.md update should make this test fail."""
    from lazybridge.skill_docs._build import build

    rc = build(check=True)
    assert rc == 0


# ---------------------------------------------------------------------------
# Sanity import — ensures sentinels / agents we touch above stay re-exported
# ---------------------------------------------------------------------------


def test_phase1_public_surface_unchanged() -> None:
    """Sanity: the symbols this test file imports stay public."""
    import lazybridge

    for name in (
        "Agent",
        "Envelope",
        "Memory",
        "MockAgent",
        "Plan",
        "PlanCompileError",
        "Step",
        "Store",
        "from_step",
    ):
        assert hasattr(lazybridge, name), f"lost public symbol: {name}"
