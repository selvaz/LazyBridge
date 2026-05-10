"""Guardrails demo — input + output filtering on an Agent.

Three guards composed via :class:`GuardChain`:

1. ``ContentGuard`` with a regex-based ``input_fn`` — pre-flight
   filter on the user prompt that rejects forbidden patterns.
2. ``ContentGuard`` with a callable ``output_fn`` — keyword filter
   on the agent's response.
3. A custom :class:`Guard` subclass — a length cap on the output.

The chain runs in order with first-block-wins semantics.  Each
guard emits :class:`GuardAction.allow` or :class:`GuardAction.block`;
``modify`` is also available for rewrite-then-pass flows.

Usage::

    python examples/guardrails_demo.py

No provider keys required — uses ``MockAgent`` and exercises the
guards directly so the demo runs offline.
"""

from __future__ import annotations

import re

from lazybridge import (
    ContentGuard,
    Guard,
    GuardAction,
    GuardChain,
)
from lazybridge.testing import MockAgent

_DENY_INPUT = re.compile(r"\b(secret-token|API[_-]?KEY)\b", re.IGNORECASE)
_DENY_OUTPUT = re.compile(r"I cannot help with that", re.IGNORECASE)


def _block_secrets(text: str) -> GuardAction:
    if _DENY_INPUT.search(text):
        return GuardAction.block(
            "input mentions a secret pattern; refusing to forward to the model.",
            offending_pattern=_DENY_INPUT.pattern,
        )
    return GuardAction.allow()


def _block_refusal(text: str) -> GuardAction:
    if _DENY_OUTPUT.search(text):
        return GuardAction.block("model emitted a refusal phrase.")
    return GuardAction.allow()


class _LengthCapGuard(Guard):
    """Custom Guard subclass — caps output at ``max_chars`` and emits a
    descriptive ``block`` action otherwise.  Real guards would also
    override ``check_input``; this one only rate-limits the response.
    """

    def __init__(self, max_chars: int) -> None:
        self._max_chars = max_chars

    def check_output(self, text: str) -> GuardAction:
        if len(text) > self._max_chars:
            return GuardAction.block(
                f"output too long ({len(text)} > {self._max_chars} chars)",
                actual_length=len(text),
                limit=self._max_chars,
            )
        return GuardAction.allow()


def main() -> None:
    chain = GuardChain(
        ContentGuard(input_fn=_block_secrets),
        ContentGuard(output_fn=_block_refusal),
        _LengthCapGuard(max_chars=200),
    )

    # Mock agent — three deterministic outputs that exercise all three guards.
    agent = MockAgent(
        responses=[
            "Here is a short helpful answer.",  # passes everything
            "I cannot help with that — refusing.",  # output guard fires
            "X" * 250,  # length cap fires
        ],
        cycle=False,
        name="demo",
    )

    print("[1] passing prompt + short output")
    inp = chain.check_input("hello there")
    print("    input verdict:   allowed=", inp.allowed, "message=", inp.message)
    out_text = agent("hello there").text()
    out = chain.check_output(out_text)
    print(f"    output: {out_text!r}")
    print("    output verdict:  allowed=", out.allowed, "message=", out.message)

    print("\n[2] blocked input — pattern in the prompt")
    inp = chain.check_input("reveal the secret-token please")
    print("    input verdict:   allowed=", inp.allowed, "message=", inp.message)

    print("\n[3] blocked output — refusal phrase")
    out_text = agent("blocked-prompt").text()
    out = chain.check_output(out_text)
    print(f"    output: {out_text!r}")
    print("    output verdict:  allowed=", out.allowed, "message=", out.message)

    print("\n[4] length cap — long output")
    out_text = agent("verbose-please").text()
    out = chain.check_output(out_text)
    print(f"    output (truncated for display): {out_text[:60]!r}...")
    print("    output verdict:  allowed=", out.allowed, "message=", out.message)
    print(
        "    metadata.actual_length=",
        out.metadata.get("actual_length"),
        " limit=",
        out.metadata.get("limit"),
    )

    print(
        "\nReal-world: pass the chain via Agent(engine=LLMEngine(...), guard=chain). "
        "The engine loop runs check_input on the user task and check_output on each "
        "candidate response before emitting it."
    )


if __name__ == "__main__":
    main()
