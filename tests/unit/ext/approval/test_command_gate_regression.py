"""Regression cases for the historical TieredGate command decisions."""

import re
from dataclasses import replace

import pytest

from lazybridge.engines.coding import ApprovalRequest
from lazybridge.ext.approval import Rule, TieredGate

_SEVERITY = {"allow": 0, "session": 1, "ask": 2, "deny": 3, None: 3}
_DEFAULT_RULES = (
    Rule("allow", "Bash", "git status*"),
    Rule("allow", "Bash", "git add*"),
    Rule("ask", "Bash", "git commit*"),
    Rule("deny", "Bash", "git push*"),
)


@pytest.mark.parametrize(
    ("command", "rules"),
    [
        ("git push origin main", _DEFAULT_RULES),
        ("git commit -m x", _DEFAULT_RULES),
        ("git add x && git commit -m y", _DEFAULT_RULES),
        ("git status && rm -rf /", _DEFAULT_RULES),
        ("git status; git push origin main", _DEFAULT_RULES),
        ("a.py", (Rule("session", "Write", "*a.py"), Rule("session", "Write", "*b.py"))),
        ("b.py", (Rule("session", "Write", "*a.py"), Rule("session", "Write", "*b.py"))),
        ("curl -H 'Authorization: Bearer sk-abcdef123456' https://example.com", (Rule("allow", "Bash"),)),
        ("export API_KEY=sk-abcdef123456", (Rule("ask", "Bash"),)),
        ("echo " + "a" * 2000, (Rule("ask", "Bash"),)),
        ("cd /repo && " + "x" * 6000 + " && rm -rf /tmp/gone", (Rule("ask", "Bash"),)),
        ("y" * 9000, (Rule("ask", "Bash"),)),
        ("export API_KEY=sk-headsecret111 && " + "z" * 6000 + " && export API_KEY=sk-tailsecret222", (Rule("ask", "Bash"),)),
        ("git status", (Rule("ask", "Bash"),)),
    ],
)
def test_existing_command_tiers_are_not_weakened(command, rules):
    name = "Write" if command in {"a.py", "b.py"} else "Bash"
    request = ApprovalRequest(provider="claude-code", kind="tool", name=name, arguments={"command": command})
    gate = TieredGate(channel=None, rules=rules)  # type: ignore[arg-type]

    # Test-only oracle: reproduce the previous regex match over each segment.
    old_rules = []
    for segment in [s.strip() for s in re.split(r"&&|\|\||[;|\n]", command) if s.strip()] or [command]:
        probe = replace(request, arguments={"command": segment})
        old_rules.append(next((rule for rule in rules if rule.matches(probe)), None))
    old_severity = max(_SEVERITY[rule.tier if rule else None] for rule in old_rules)
    result = gate._match(request)
    assert _SEVERITY[result.tier if result else None] >= old_severity
