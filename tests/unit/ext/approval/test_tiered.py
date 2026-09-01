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
