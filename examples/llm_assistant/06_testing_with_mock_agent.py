"""Testing pattern — drive the canonical agent shape under MockAgent.

This is the recommended pytest harness for LazyBridge code: build
the agent exactly as production wires it, then swap the engine for
``MockAgent`` (or for ``Agent.from_provider(tier='cheap')`` when you
want a real but cheap model in CI).
"""

from __future__ import annotations

from lazybridge import Envelope
from lazybridge.testing import MockAgent


def write_summary(env: Envelope[str]) -> Envelope[str]:
    """Production function under test — takes an Envelope, returns one."""
    return Envelope(payload=f"Summary: {env.text()}")


def test_write_summary_with_mock() -> None:
    """A real test would live in tests/unit/ and import the actual agent."""
    research = MockAgent(["Background: LazyBridge is a Python agent framework."], name="research")
    env = research("LazyBridge")
    out = write_summary(env)
    assert out.text().startswith("Summary:")
    assert "LazyBridge" in out.text()


def main() -> None:
    test_write_summary_with_mock()
    print("ok — write_summary round-trips through MockAgent")


if __name__ == "__main__":
    main()
