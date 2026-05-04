"""One-shot fragment quality pass — adds narrative + see-also where missing.

Run from repo root:  python tools/fragment_quality_pass.py
Idempotent — sections already present are left untouched.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAGMENTS = ROOT / "lazybridge" / "skill_docs" / "fragments"

# (narrative_body, see_also_body)  per fragment.
# narrative_body = "" means: don't add (already present, or doesn't apply).
# see_also_body  = "" means: don't add.
# Each block expects no leading/trailing newlines; the inserter wraps them.
PASSES: dict[str, tuple[str, str]] = {
    # ---------- Tool surface ---------------------------------------------------
    "tool_schema": (
        """**Use `mode="signature"`** (the default) for any function with type
hints and a docstring.  The schema is produced deterministically with no
LLM cost — the right choice for >95% of user code.

**Use `mode="llm"`** for legacy functions you can't easily annotate —
opaque `**kwargs`, third-party callables, auto-generated wrappers.  Pay
tokens once at construction time.

**Use `mode="hybrid"`** when the codebase is mixed: the signature path
covers what it can, the LLM only fills annotated gaps.""",
        """- [Tool](tool.md) — the wrapper that consumes a schema mode.
- [Native tools](native-tools.md) — the no-schema-needed alternative.""",
    ),
    "native_tools": (
        """**Use a native tool** when the provider already hosts the capability
(web search, code execution, file search) and your data doesn't need to
leave the provider's environment.  No code, no schema — pass an enum.

**Don't use a native tool** when you need full control over the
implementation, custom auth, or you want to swap providers without
re-testing the tool surface.  Write a regular `Tool` instead.""",
        """- [Tool](tool.md) — write your own when native isn't a fit.
- [Providers](providers.md) — which native tools each provider supports.""",
    ),
    # ---------- Composition ----------------------------------------------------
    "chain": (
        "",  # narrative already present
        """- [Agent.parallel](agent-parallel.md) — deterministic fan-out.
- [Agent.as_tool](as-tool.md) — agents wrapping agents the implicit way.
- [Plan](plan.md) — typed hand-offs and conditional routing.""",
    ),
    "as_tool": (
        """**Use `as_tool()`** when you want to expose an agent as a tool with a
specific name, description, or a `verify=` judge — the gated-call shape.

**Don't bother calling `as_tool()`** for the simple case: passing an
`Agent` directly to `tools=[...]` already wraps it for you.  Reach for
the explicit form only when overriding name / description / verify.""",
        """- [Agent.chain](chain.md) — the linear-pipeline alternative.
- [verify=](verify.md) — judge placement around a tool call.
- [Agent](agent.md) — the surface that consumes the wrapped tool.""",
    ),
    "agent_parallel": (
        "",  # narrative already present
        """- [Agent.chain](chain.md) — sequential rather than fan-out.
- [Plan](plan.md) — when you need typed steps with parallel bands.""",
    ),
    # ---------- State / observability -----------------------------------------
    "store": (
        """**Use `Store`** for cross-process or cross-run state — pipeline
checkpoints, computed artefacts a downstream agent should read live, or
shared scratch space across a fan-out.  SQLite mode is durable and
thread-safe.

**Don't use `Store`** for in-prompt conversation context — that's
`Memory`'s job.  `Store` is "what should survive a crash"; `Memory` is
"what should the model see in the next turn".""",
        """- [Memory](memory.md) — separate concept (in-prompt conversation context).
- [Checkpoint & resume](checkpoint.md) — `Plan` uses `Store` under the hood.""",
    ),
    "guards": (
        """**Use `Guard`** to filter both the input task and the output payload
before the engine runs / after it returns.  A regex-based
`ContentGuard` is essentially free; an `LLMGuard` (LLM-as-judge) costs
tokens but catches policy violations regex can't see.

**Compose with `GuardChain`**: cheap deterministic guards first, then
the LLM fallback only when needed.  First blocker wins.""",
        """- [Session](session.md) — events emitted by guard outcomes.
- [verify=](verify.md) — different placement (judge around a tool call).""",
    ),
    "human_engine": (
        """**Use `HumanEngine`** for an approval gate or a structured form: the
human types a string (or fills Pydantic fields), the agent treats it as
an LLM response.  Drop-in replacement for `LLMEngine` in any pipeline
where you want to insert a human at a specific step.

**Use `SupervisorEngine` instead** when the human needs to call tools,
retry agents with feedback, or run a real REPL — `HumanEngine` is the
lighter approval-only variant.""",
        """- [SupervisorEngine](supervisor.md) — full HIL REPL with tools and retry.
- [Agent.chain](chain.md) — typical pattern for inserting HumanEngine mid-pipeline.""",
    ),
    "evals": (
        """**Use `EvalSuite`** as a thin pytest-ish harness for an agent's
text-output behaviour: deterministic checks (`contains`, `exact_match`),
optional `llm_judge` for grading subjective outputs.

**Don't use `EvalSuite`** for fine-grained unit tests of internal
helpers — those are pytest's job.  Reach for `EvalSuite` when the unit
under test is the *agent's response*, not a function.""",
        """- [verify=](verify.md) — runtime version of a judge (gates each call).
- [Testing (MockAgent)](testing.md) — deterministic doubles for unit tests.""",
    ),
    # ---------- Plan internals -------------------------------------------------
    "envelope": (
        """**`Envelope` is the universal request/response object** — every
`agent.run()` returns one.  `payload` is typed when `output=Model` is
set, otherwise it's the raw text.  `metadata` carries cost / token /
latency telemetry, with `nested_*` buckets for Agent-of-Agents
roll-up.

**Use `.text()`** for "give me a string regardless of payload type",
**`.payload`** for the typed result, **`.ok`** to check error state.""",
        """- [Agent](agent.md) — the producer.
- [Sentinels](sentinels.md) — how `Plan` steps reference prior envelopes.""",
    ),
    "sentinels": (
        "",  # narrative already present (21 lines)
        """- [Plan](plan.md) — the engine that interprets sentinels.
- [Parallel plan steps](parallel-steps.md) — `from_parallel_all` aggregation.
- [Envelope](envelope.md) — the object sentinels carry between steps.""",
    ),
    "plan": (
        "",  # narrative-style content already in the rules block
        """- [Sentinels](sentinels.md) — `from_prev` / `from_step` / `from_parallel`.
- [Parallel plan steps](parallel-steps.md) — concurrent bands.
- [Checkpoint & resume](checkpoint.md) — crash recovery.
- [SupervisorEngine](supervisor.md) — alternative engine for HIL pipelines.
- [verify=](verify.md) — judge placement at the engine level.""",
    ),
    "parallel_steps": (
        """**Use `parallel=True` step bands** when independent steps can run
concurrently and the next step needs all of their results.
`from_parallel_all("first")` aggregates the band's outputs as a labelled
text join.  Atomicity: if any branch errors, no `writes` are applied —
a future resume re-runs the whole band cleanly.

**Use `Agent.parallel` instead** for simple deterministic fan-out at
the application layer (no Plan, no aggregation, just `list[Envelope]`).""",
        """- [Plan](plan.md) — the engine that orchestrates parallel bands.
- [Sentinels](sentinels.md) — `from_parallel_all` and `from_parallel`.""",
    ),
    "supervisor": (
        """**Use `SupervisorEngine`** for full human-in-the-loop control: a REPL
where the operator can call tools, retry agents with feedback, store
keys, or hand control back with `continue`.  It implements the same
`Engine` protocol as `LLMEngine`, so `Agent(engine=SupervisorEngine())`
slots into any pipeline.

**Use `HumanEngine` instead** for approval-only flows where the human
types one response and the pipeline moves on.""",
        """- [HumanEngine](human-engine.md) — lighter approval-only variant.
- [Plan](plan.md) — typical container for a supervisor mid-pipeline.""",
    ),
    "checkpoint": (
        """**`Plan` checkpoints** to a `Store` after every step.  A crashed
or interrupted run picks up at the failed step on the next call with
`resume=True`; a clean prior run short-circuits to the cached `writes`
bucket.

**`on_concurrent="fail"`** (default) gives single-writer semantics —
two concurrent runs sharing a `checkpoint_key` collide via CAS and the
loser raises `ConcurrentPlanRunError`.  **`on_concurrent="fork"`**
isolates each run under a per-uid suffixed key (good for fan-out
workflows; incompatible with `resume=True`).""",
        """- [Plan](plan.md) — the engine that writes checkpoints.
- [Store](store.md) — the durable layer behind checkpoints.""",
    ),
    "verify": (
        """**Use `verify=`** to wrap a run (or a tool call) in a judge/retry
loop: each output is scored, rejection feeds the judge's reason back
into the next attempt as context, capped at `max_verify`.  Two
placements:

* **Agent-level** (`Agent(..., verify=judge)`) — gates the whole run.
* **Tool-level** (`agent.as_tool(..., verify=judge)`) — gates each call
  through the wrapped agent, leaving the outer pipeline untouched.""",
        """- [Agent.as_tool](as-tool.md) — tool-level verify placement (Option B).
- [Plan](plan.md) — alternative: explicit retry steps via routing.""",
    ),
    # ---------- Graph / observability not yet covered --------------------------
    "graph_schema": (
        "",  # narrative already present
        """- [Session](session.md) — populates the graph as agents register.""",
    ),
    # ---------- Advanced -------------------------------------------------------
    "engine_protocol": (
        "",  # already has narrative
        """- [BaseProvider](base-provider.md) — the layer below an engine.
- [Plan](plan.md) — example of a non-LLM engine.""",
    ),
    "base_provider": (
        "",  # already has narrative
        """- [Engine protocol](engine-protocol.md) — the layer above a provider.
- [Provider registry](register-provider.md) — how new providers are wired in.""",
    ),
    "plan_serialize": (
        "",  # already has narrative
        """- [Plan](plan.md) — what gets serialised.""",
    ),
    "register_provider": (
        """**Use the provider registry** to make `Agent("my-model")` route to a
custom provider without forking the framework.  Two granularities:

* `register_provider_alias("mistral", "mistral")` — exact-match alias.
* `register_provider_rule("claude-opus-5", "anthropic")` — substring or
  prefix rule (later registrations win).
* `set_default_provider(None)` — disable the safety-net fallback so an
  unrecognised model raises `ValueError` at construction instead of
  routing somewhere unintended.""",
        """- [BaseProvider](base-provider.md) — what you implement for a brand-new provider.
- [Providers](providers.md) — the built-in catalogue.""",
    ),
    "core_types": (
        "",  # already has narrative
        """- [Engine protocol](engine-protocol.md) — consumes most of these types.
- [BaseProvider](base-provider.md) — produces `CompletionResponse`.""",
    ),
    "providers": (
        """**`providers.md`** is the catalogue of built-in providers and the
shape of the `tier` aliases (`super_cheap` / `cheap` / `medium` /
`expensive` / `top`) each one resolves.  Pick a model directly with
`Agent("model-id")` or by tier with
`Agent.from_provider("anthropic", tier="top")`.

For a brand-new provider see [Provider registry](register-provider.md);
to author one from scratch see [BaseProvider](base-provider.md).""",
        """- [BaseProvider](base-provider.md) — the contract every provider satisfies.
- [Provider registry](register-provider.md) — runtime aliases and rules.
- [Native tools](native-tools.md) — what each provider exposes server-side.""",
    ),
}

SECTION_RE = re.compile(r"^##\s+([a-z_-]+)\s*$", re.MULTILINE)


def section_starts(text: str) -> dict[str, int]:
    """Return {section_name: char_offset_of_##_line_start}."""
    return {m.group(1).lower(): m.start() for m in SECTION_RE.finditer(text)}


def insert_after(text: str, after: str, body: str) -> str:
    """Insert `body` as a new ## section right after the section named
    `after`. Returns text unchanged if `after` is missing."""
    starts = section_starts(text)
    if after not in starts:
        return text
    # Find the end of the `after` section: next section start, or EOF.
    matches = sorted(starts.items(), key=lambda kv: kv[1])
    after_idx = next(i for i, (n, _) in enumerate(matches) if n == after)
    end = matches[after_idx + 1][1] if after_idx + 1 < len(matches) else len(text)
    insertion = "\n" + body.rstrip() + "\n\n"
    # Trim any trailing blank lines from the section we're inserting after.
    head = text[:end].rstrip() + "\n"
    return head + insertion + text[end:].lstrip("\n")


def append_section(text: str, body: str) -> str:
    """Append a new ## section at end of file."""
    return text.rstrip() + "\n\n" + body.rstrip() + "\n"


def main() -> int:
    changed = []
    for stem, (narrative, see_also) in PASSES.items():
        path = FRAGMENTS / f"{stem}.md"
        if not path.exists():
            print(f"  ! missing: {stem}.md")
            continue
        text = path.read_text()
        sections = section_starts(text)
        new = text
        if narrative and "narrative" not in sections:
            block = "## narrative\n" + narrative.strip()
            anchor = "rules" if "rules" in sections else "signature"
            new = insert_after(new, anchor, block)
        if see_also and "see-also" not in sections:
            new = append_section(new, "## see-also\n" + see_also.strip())
        if new != text:
            path.write_text(new)
            changed.append(stem)
    print(f"Updated {len(changed)} fragment(s):")
    for s in changed:
        print(f"  + {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
