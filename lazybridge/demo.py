"""Two-minute demo: one tool, three peer engines.

Run it after installing LazyBridge::

    python -m pip install "lazybridge[claude-code,deepseek]"
    python -m lazybridge.demo

The demo finds which engines this machine can use (a DeepSeek API key,
a signed-in Claude Code, a signed-in Codex), then:

1. gives every available engine the same Python tool and the same question;
2. if two or more are available, lets one write and another review with
   ``verify=``, the writer/reviewer loop.

Nothing is configured by hand: missing engines are skipped with a one-line
hint on how to enable them. This file is also the example to read.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable

from lazybridge import Agent


def lazybridge_facts() -> str:
    """Return three facts about the LazyBridge framework."""
    return (
        "Agent = Engine + Tools + State. "
        "Engines are swappable peers: provider APIs, Claude Code and Codex. "
        "An agent can be a tool of another agent."
    )


# Each factory builds one engine; ``system`` is the engine's instruction.
def _deepseek(system: str | None = None):
    from lazybridge import LLMEngine

    return LLMEngine("deepseek-flash", system=system)


def _claude(system: str | None = None):
    from lazybridge import ClaudeCodeEngine

    return ClaudeCodeEngine(model="haiku", system=system)


def _codex(system: str | None = None):
    from lazybridge import CodexEngine

    return CodexEngine(system=system)


def _codex_found() -> bool:
    try:
        from lazybridge.engines.codex import codex_executable

        codex_executable()
    except Exception:
        return False
    return True


# name -> (is it usable here?, how to enable it, factory)
ENGINES: dict[str, tuple[Callable[[], bool], str, Callable[..., object]]] = {
    "deepseek": (
        lambda: bool(os.environ.get("DEEPSEEK_API_KEY")),
        "set DEEPSEEK_API_KEY (key from platform.deepseek.com)",
        _deepseek,
    ),
    "claude-code": (
        lambda: shutil.which("claude") is not None,
        "npm install -g @anthropic-ai/claude-code, then run `claude` once to sign in",
        _claude,
    ),
    "codex": (
        _codex_found,
        "npm install -g @openai/codex, then run `codex login`",
        _codex,
    ),
}

QUESTION = "Call lazybridge_facts, then explain LazyBridge in one sentence of at most 25 words."
TASK = "Write a Python one-liner that prints the squares of 1 to 5. Reply with the code only."
REVIEW = (
    'Reply "approved" if the code is a correct Python one-liner printing 1, 4, 9, 16, 25; '
    'otherwise reply "rejected: <reason>".'
)


def show(result) -> tuple[str, bool]:
    """Text of a result, or its error: a failed engine returns an error Envelope."""
    if result.ok:
        return result.text().strip(), True
    return f"failed: {result.error.type}: {result.error.message}", False


def available() -> list[str]:
    found = []
    for name, (check, hint, _) in ENGINES.items():
        ok = check()
        print(f"  [{'x' if ok else ' '}] {name:<12}" + ("" if ok else f" -> {hint}"))
        if ok:
            found.append(name)
    return found


def main() -> int:
    print("LazyBridge demo: one tool, peer engines\n\nEngines on this machine:")
    names = available()
    if not names:
        print("\nNo engine available yet: enable at least one of the above and rerun.")
        return 1

    print(f"\n1) Same tool, same question, {len(names)} engine(s):\n   {QUESTION}\n")
    all_ok = True
    for name in names:
        agent = Agent(name=name, engine=ENGINES[name][2](), tools=[lazybridge_facts])
        try:
            text, ok = show(agent(QUESTION))
        except Exception as exc:  # a broken login must not hide the other engines
            text, ok = f"failed: {type(exc).__name__}: {exc}", False
        all_ok = all_ok and ok
        print(f"   {name:<12} {text}")

    if len(names) < 2:
        print("\nEnable a second engine to see one write and another review.")
        return 0 if all_ok else 1
    writer_name, reviewer_name = names[0], names[-1]
    print(f"\n2) {writer_name} writes, {reviewer_name} reviews (verify=):\n")
    reviewer = Agent(name=reviewer_name, engine=ENGINES[reviewer_name][2](system=REVIEW))
    writer = Agent(name=writer_name, engine=ENGINES[writer_name][2](), verify=reviewer, max_verify=2)
    try:
        text, ok = show(writer(TASK))
    except Exception as exc:
        text, ok = f"failed: {type(exc).__name__}: {exc}", False
    print(text)
    print('\nSource of this demo: python -c "import lazybridge.demo as d; print(d.__file__)"')
    return 0 if all_ok and ok else 1


if __name__ == "__main__":
    sys.exit(main())
