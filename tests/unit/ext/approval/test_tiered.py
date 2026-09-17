"""TieredGate — tier resolution, session-grant scoping, and audit logging.

Covers the promotion fixes from ``approval-lab``:

* session grants scoped to ``(provider, kind, name, cwd, policy fingerprint)``,
  not just ``(kind, name)`` — the same tool in two different cwds must ask
  twice;
* every decision lands in a structured :class:`AuditRecord`, including the
  responding channel's identity when a human was actually asked;
* unmatched requests are denied by default (regression pin, not an added
  behaviour).
"""

from __future__ import annotations

from lazybridge.engines.coding import ApprovalDecision, ApprovalRequest, remembering_gate
from lazybridge.ext.approval import AuditRecord, Rule, TerminalChannel, TieredGate


class FakeChannel:
    """Scripted yes/no answers; records every prompt it was asked."""

    name = "fake"

    def __init__(self, answers: list[bool] | None = None) -> None:
        self._answers = list(answers or [])
        self.prompts: list[str] = []

    async def ask(self, prompt: str) -> bool:
        self.prompts.append(prompt)
        if not self._answers:
            raise AssertionError("FakeChannel.ask called more times than scripted")
        return self._answers.pop(0)


def _request(
    *,
    name: str = "Read",
    kind: str = "tool",
    provider: str = "claude-code",
    arguments: dict | None = None,
    cwd: str | None = None,
) -> ApprovalRequest:
    return ApprovalRequest(
        provider=provider,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        name=name,
        arguments=arguments or {},
        cwd=cwd,
    )


DEFAULT_RULES = (
    Rule("allow", "Read"),
    Rule("allow", "Bash", "git status*"),
    Rule("allow", "Bash", "git add*"),
    Rule("session", "Write"),
    Rule("ask", "Bash", "git commit*"),
    Rule("deny", "Bash", "git push*"),
)


# --- tier resolution ---------------------------------------------------


async def test_allow_tier_runs_without_asking():
    channel = FakeChannel()
    gate = TieredGate(channel=channel, rules=DEFAULT_RULES)
    decision = await gate(_request(name="Read"))
    assert decision.action == "allow"
    assert channel.prompts == []  # never consulted


async def test_llm_provider_request_matches_existing_rules():
    channel = FakeChannel()
    gate = TieredGate(channel=channel, rules=(Rule("allow", "get_plan"),))

    decision = await gate(_request(provider="llm", name="get_plan"))

    assert decision.action == "allow"
    assert gate.log[0].provider == "llm"
    assert channel.prompts == []


async def test_deny_tier_never_asks():
    channel = FakeChannel()
    gate = TieredGate(channel=channel, rules=DEFAULT_RULES)
    decision = await gate(_request(name="Bash", arguments={"command": "git push origin main"}))
    assert decision.action == "deny"
    assert channel.prompts == []  # deny short-circuits, no human involved


async def test_ask_tier_asks_every_call():
    channel = FakeChannel(answers=[True, True])
    gate = TieredGate(channel=channel, rules=DEFAULT_RULES)
    req = _request(name="Bash", arguments={"command": "git commit -m x"})
    first = await gate(req)
    second = await gate(req)
    assert first.action == "allow"
    assert second.action == "allow"
    assert len(channel.prompts) == 2  # "ask" tier never remembers


async def test_ask_tier_denied_by_human():
    channel = FakeChannel(answers=[False])
    gate = TieredGate(channel=channel, rules=DEFAULT_RULES)
    decision = await gate(_request(name="Bash", arguments={"command": "git commit -m x"}))
    assert decision.action == "deny"
    assert "denied by the human approver" in decision.message


async def test_session_tier_asks_once_then_allows():
    channel = FakeChannel(answers=[True])
    gate = TieredGate(channel=channel, rules=DEFAULT_RULES)
    req = _request(name="Write", cwd="C:/repo")
    first = await gate(req)
    second = await gate(req)
    assert first.action == "allow_session"
    assert second.action == "allow"
    assert len(channel.prompts) == 1  # second call reused the grant


async def test_engine_session_wrapper_preserves_tiered_scope_and_audit():
    """The engines must not replace TieredGate's narrow key with tool name only."""
    channel = FakeChannel(answers=[True, True])
    gate = TieredGate(channel=channel, rules=DEFAULT_RULES)
    wrapped = remembering_gate(gate, set())

    first = await wrapped(_request(name="Write", cwd="C:/repo-a"))
    second = await wrapped(_request(name="Write", cwd="C:/repo-b"))
    repeated = await wrapped(_request(name="Write", cwd="C:/repo-a"))

    assert first.action == "allow_session"
    assert second.action == "allow_session"  # a separate cwd asks again
    assert repeated.action == "allow"
    assert len(channel.prompts) == 2
    assert [record.cwd for record in gate.log] == ["C:/repo-a", "C:/repo-b", "C:/repo-a"]


# --- default deny --------------------------------------------------------


async def test_default_deny_when_no_rule_matches():
    channel = FakeChannel()
    gate = TieredGate(channel=channel, rules=DEFAULT_RULES)
    decision = await gate(_request(name="WebFetch"))
    assert decision.action == "deny"
    assert "no rule" in decision.message
    assert channel.prompts == []  # unmatched never reaches a human


async def test_empty_rule_table_denies_everything():
    gate = TieredGate(channel=FakeChannel(), rules=())
    decision = await gate(_request(name="Read"))
    assert decision.action == "deny"


# --- session-grant scoping (the fixed prototype bug) ---------------------


async def test_session_grant_does_not_cross_cwd():
    channel = FakeChannel(answers=[True, True])
    gate = TieredGate(channel=channel, rules=DEFAULT_RULES)
    await gate(_request(name="Write", cwd="C:/repo-a"))
    decision = await gate(_request(name="Write", cwd="C:/repo-b"))
    assert decision.action == "allow_session"  # asked again, not silently reused
    assert len(channel.prompts) == 2


async def test_session_grant_reused_for_same_cwd_different_case():
    """Windows cwds are case-insensitive; the grant must still be recognised."""
    channel = FakeChannel(answers=[True])
    gate = TieredGate(channel=channel, rules=DEFAULT_RULES)
    await gate(_request(name="Write", cwd="C:/Repo"))
    decision = await gate(_request(name="Write", cwd="c:/repo"))
    assert decision.action == "allow"
    assert len(channel.prompts) == 1


async def test_session_grant_does_not_cross_provider():
    channel = FakeChannel(answers=[True, True])
    gate = TieredGate(channel=channel, rules=DEFAULT_RULES)
    await gate(_request(name="Write", provider="claude-code", cwd="C:/repo"))
    decision = await gate(_request(name="Write", provider="codex", cwd="C:/repo"))
    assert decision.action == "allow_session"
    assert len(channel.prompts) == 2


async def test_session_grant_does_not_cross_policy_fingerprint():
    """Same tool/cwd, but two DIFFERENT rules (different policy) must each ask.

    Matched via two ``session`` rules on the same gate that apply to disjoint
    argument patterns — the human approving ``*a.py`` must not silently cover
    a later, differently-scoped ``*b.py`` request even though both are
    "Write, session, this cwd".
    """
    channel = FakeChannel(answers=[True, True])
    mixed_rules = (Rule("session", "Write", "*a.py"), Rule("session", "Write", "*b.py"))
    gate = TieredGate(channel=channel, rules=mixed_rules)
    await gate(_request(name="Write", cwd="C:/repo", arguments={"command": "a.py"}))
    decision = await gate(_request(name="Write", cwd="C:/repo", arguments={"command": "b.py"}))
    assert decision.action == "allow_session"  # asked again: different rule fingerprint
    assert len(channel.prompts) == 2


# --- compound bash command splitting --------------------------------------


async def test_compound_command_takes_worst_segment_tier():
    channel = FakeChannel(answers=[True])
    gate = TieredGate(channel=channel, rules=DEFAULT_RULES)
    decision = await gate(_request(name="Bash", arguments={"command": "git add x && git commit -m y"}))
    # git add -> allow, git commit -> ask: the compound call must ask.
    assert decision.action == "allow"
    assert len(channel.prompts) == 1


async def test_compound_command_with_unmatched_segment_is_denied():
    channel = FakeChannel()
    gate = TieredGate(channel=channel, rules=DEFAULT_RULES)
    decision = await gate(_request(name="Bash", arguments={"command": "git status && rm -rf /"}))
    assert decision.action == "deny"
    assert channel.prompts == []


async def test_compound_command_denied_segment_wins_over_allow():
    channel = FakeChannel()
    gate = TieredGate(channel=channel, rules=DEFAULT_RULES)
    decision = await gate(_request(name="Bash", arguments={"command": "git status; git push origin main"}))
    assert decision.action == "deny"
    assert channel.prompts == []  # deny is final, no human consulted


# --- audit log -------------------------------------------------------------


async def test_audit_log_records_every_decision():
    channel = FakeChannel(answers=[True])
    gate = TieredGate(channel=channel, rules=DEFAULT_RULES)
    await gate(_request(name="Read"))
    await gate(_request(name="Write", cwd="C:/repo"))

    assert len(gate.log) == 2
    allow_record, session_record = gate.log
    assert isinstance(allow_record, AuditRecord)
    assert allow_record.tool_name == "Read"
    assert allow_record.tier == "allow"
    assert allow_record.action == "allow"
    assert allow_record.responder is None  # no human was asked

    assert session_record.tool_name == "Write"
    assert session_record.tier == "session"
    assert session_record.action == "allow_session"
    assert session_record.responder == "fake"  # channel identity captured
    assert session_record.timestamp > 0
    assert len(session_record.arguments_hash) == 64  # sha256 hex digest


async def test_audit_log_records_unmatched_default_deny():
    gate = TieredGate(channel=FakeChannel(), rules=DEFAULT_RULES)
    await gate(_request(name="SomeUnknownTool"))
    record = gate.log[-1]
    assert record.tier == "unmatched"
    assert record.action == "deny"


async def test_audit_on_record_callback_invoked():
    seen: list[AuditRecord] = []
    channel = FakeChannel()
    gate = TieredGate(channel=channel, rules=DEFAULT_RULES, on_record=seen.append)
    await gate(_request(name="Read"))
    assert len(seen) == 1
    assert seen[0].tool_name == "Read"


async def test_audit_preview_truncates_large_arguments():
    gate = TieredGate(channel=FakeChannel(), rules=(Rule("allow", "Read"),))
    huge_arg = "x" * 1000
    await gate(_request(name="Read", arguments={"content": huge_arg}))
    record = gate.log[-1]
    assert len(record.arguments_preview) <= 201  # limit + ellipsis char
    assert record.arguments_preview.endswith("…")


async def test_audit_preview_redacts_common_secret_shapes():
    gate = TieredGate(channel=FakeChannel(), rules=(Rule("allow", "Bash"),))
    await gate(
        _request(
            name="Bash",
            arguments={"command": "curl -H 'Authorization: Bearer sk-abcdef123456' https://example.com"},
        )
    )
    record = gate.log[-1]
    assert "sk-abcdef123456" not in record.arguments_preview
    assert "[redacted]" in record.arguments_preview


async def test_rendered_prompt_redacts_common_secret_shapes():
    channel = FakeChannel(answers=[False])
    gate = TieredGate(channel=channel, rules=(Rule("ask", "Bash"),))
    await gate(_request(name="Bash", arguments={"command": "export API_KEY=sk-abcdef123456"}))
    assert "sk-abcdef123456" not in channel.prompts[0]
    assert "[redacted]" in channel.prompts[0]


# --- what the approver is actually shown --------------------------------


async def test_a_long_command_is_shown_whole():
    """Two thousand characters used to be cut at four hundred.

    Measured on a live store: 155 of 397 real requests (39%) reached the
    operator truncated, and the operator approved 95% of everything. The
    budget was between eight and twenty times tighter than Telegram, which
    is the narrowest transport these prompts travel over.
    """
    channel = FakeChannel(answers=[False])
    gate = TieredGate(channel=channel, rules=(Rule("ask", "Bash"),))
    command = "echo " + ("a" * 2000)
    await gate(_request(name="Bash", arguments={"command": command}))
    assert "elided" not in channel.prompts[0]
    assert command in channel.prompts[0]


async def test_an_elided_command_still_shows_its_ending():
    """The reason this fix exists.

    In `cd repo && ... && rm -rf /tmp/gone` the consequence is LAST. Keeping
    the head and dropping the tail shows the approver the harmless opening
    and hides the part that could hurt them -- which does not merely
    inconvenience the human, it manufactures their consent.
    """
    channel = FakeChannel(answers=[False])
    gate = TieredGate(channel=channel, rules=(Rule("ask", "Bash"),))
    command = "cd /repo && " + ("x" * 6000) + " && rm -rf /tmp/gone"
    await gate(_request(name="Bash", arguments={"command": command}))
    prompt = channel.prompts[0]
    assert "cd /repo" in prompt, "the opening should still be there"
    assert "rm -rf /tmp/gone" in prompt, "the approver must see what it ends with"


async def test_the_marker_says_how_much_is_missing():
    """`...` cannot tell one dropped line from ten thousand, and that is
    exactly the judgement the reader needs to decide whether to go and look
    at the whole value."""
    import re

    channel = FakeChannel(answers=[False])
    gate = TieredGate(channel=channel, rules=(Rule("ask", "Bash"),))
    await gate(_request(name="Bash", arguments={"command": "y" * 9000}))
    match = re.search(r"\[…(\d+) characters elided…\]", channel.prompts[0])
    assert match, channel.prompts[0][:200]
    assert int(match.group(1)) > 5000


async def test_redaction_survives_into_the_tail_segment():
    """Redacting after eliding would leave a secret in the kept tail: the
    patterns would have run against a string that no longer held it."""
    channel = FakeChannel(answers=[False])
    gate = TieredGate(channel=channel, rules=(Rule("ask", "Bash"),))
    command = "export API_KEY=sk-headsecret111 && " + ("z" * 6000) + " && export API_KEY=sk-tailsecret222"
    await gate(_request(name="Bash", arguments={"command": command}))
    prompt = channel.prompts[0]
    assert "sk-headsecret111" not in prompt
    assert "sk-tailsecret222" not in prompt
    assert "[redacted]" in prompt


async def test_a_long_cwd_does_not_blow_the_rendered_message_past_budget():
    """Found by Codex review: only ``arguments`` went through ``elide()``,
    so a pathological (or just very deep) ``cwd`` could push the whole
    rendered message past Telegram's transport cap on its own -- the exact
    failure class this module exists to remove, reappearing through a field
    that had no budget at all."""
    channel = FakeChannel(answers=[False])
    gate = TieredGate(channel=channel, rules=(Rule("ask", "Bash"),))
    long_cwd = "C:\\" + ("nested-directory\\" * 400)
    await gate(_request(name="Bash", arguments={"command": "git status"}, cwd=long_cwd))
    prompt = channel.prompts[0]
    assert len(prompt) < 4096
    assert long_cwd not in prompt  # the bare, unbounded cwd never reaches the human whole
    cwd_section = prompt[prompt.index("cwd:") :]  # elide()'s marker spans its own line
    assert "elided" in cwd_section


async def test_a_short_cwd_is_shown_whole():
    channel = FakeChannel(answers=[False])
    gate = TieredGate(channel=channel, rules=(Rule("ask", "Bash"),))
    await gate(_request(name="Bash", arguments={"command": "git status"}, cwd="C:\\repo"))
    assert "cwd: C:\\repo" in channel.prompts[0]


async def test_a_short_prompt_is_untouched():
    channel = FakeChannel(answers=[False])
    gate = TieredGate(channel=channel, rules=(Rule("ask", "Bash"),))
    await gate(_request(name="Bash", arguments={"command": "git status"}))
    assert "elided" not in channel.prompts[0]
    assert '"command": "git status"' in channel.prompts[0]


# --- TerminalChannel ---------------------------------------------------


async def test_terminal_channel_accepts_yes(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    channel = TerminalChannel()
    assert await channel.ask("approve?") is True


async def test_terminal_channel_rejects_anything_else(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "nope")
    channel = TerminalChannel()
    assert await channel.ask("approve?") is False


# --- ApprovalGate structural conformance ---------------------------------


async def test_tiered_gate_satisfies_approval_gate_protocol():
    """TieredGate must work as a plain ApprovalGate callback, no wrapper needed."""
    gate = TieredGate(channel=FakeChannel(), rules=(Rule("allow", "*"),))
    result = gate(_request(name="anything"))
    decision = await result
    assert isinstance(decision, ApprovalDecision)


def test_elide_never_exceeds_the_budget_it_was_given():
    """Found by Codex review on PR #167.

    The marker costs characters too, so the first version answered a
    three-character budget with a thirty-odd-character string -- longer than
    the limit it was asked to respect, which would push a message past the
    exact transport cap the caller was defending against. Below the floor
    where both ends fit, the END is what survives: the same argument the
    rest of the module is built on.
    """
    from lazybridge._display import DISPLAY_BUDGET, elide

    text = "cd /repo && " + ("x" * 500) + " && rm -rf /tmp/gone"
    for budget in (1, 3, 10, 39, 43, 44, 100, 999, DISPLAY_BUDGET):
        assert len(elide(text, budget)) <= budget, budget
    # Nothing fits in nothing.
    assert elide(text, 0) == ""
    assert elide(text, -5) == ""
    # Where only a sliver fits, it is the sliver that names the consequence.
    assert elide(text, 20).endswith("rm -rf /tmp/gone")


# --- rules that name a kind ---------------------------------------------


async def test_a_rule_written_before_kinds_existed_still_matches_everything():
    """Every rule in every table predates this field. Defaulting to "*"
    is what keeps them meaning exactly what they meant."""
    channel = FakeChannel(answers=[False])
    gate = TieredGate(channel=channel, rules=(Rule("allow", "Bash"),))
    decision = await gate(_request(name="Bash", kind="tool"))
    assert decision.action == "allow"


async def test_a_command_can_be_spoken_about_at_all():
    """Before this, a table could say nothing about commands: a tool
    arrives as a bare identifier, a command as the whole command line, so
    tool patterns matched no command and every escalation fell into the
    default deny. Seventeen refusals in six hours in production, all of
    them Codex asking to run its own tests."""
    channel = FakeChannel(answers=[True])
    gate = TieredGate(
        channel=channel,
        rules=(Rule("ask", "*", kind_pattern="command"),),
    )
    decision = await gate(_request(name='"powershell.exe" -Command Remove-Item .test-tmp', kind="command"))
    assert decision.action == "allow"
    assert channel.prompts, "the operator has to be the one who decides"


async def test_a_command_rule_does_not_quietly_govern_tools():
    """The whole point of naming a kind is that it narrows. A catch-all
    name pattern meant for commands must not become a catch-all for tool
    calls, which would erase the default-deny the table relies on."""
    channel = FakeChannel(answers=[False])
    gate = TieredGate(channel=channel, rules=(Rule("allow", "*", kind_pattern="command"),))
    decision = await gate(_request(name="launch_specialist", kind="tool"))
    assert decision.action == "deny", "an unmatched tool still falls to the default"


async def test_a_tool_rule_does_not_quietly_govern_commands():
    channel = FakeChannel(answers=[False])
    gate = TieredGate(channel=channel, rules=(Rule("allow", "*", kind_pattern="tool"),))
    decision = await gate(_request(name="rm -rf /", kind="command"))
    assert decision.action == "deny"


def test_the_kind_is_part_of_a_rules_fingerprint():
    """Session grants are scoped by fingerprint. Two rules differing only
    in which kind they speak about must not share one, or a grant a human
    gave for a tool would silently satisfy a command of the same name."""
    tool_rule = Rule("session", "git push*", kind_pattern="tool")
    command_rule = Rule("session", "git push*", kind_pattern="command")
    assert tool_rule.fingerprint() != command_rule.fingerprint()
