# Step 6: Cross-model verification with `verify=`

So far every output of your agent has been *whatever the model produced*. If
the model hallucinated, contradicted itself, used the wrong tone, or missed a
formatting requirement — that's what you'd ship.

LLMs are good at writing. Other LLMs are good at *finding faults* in writing.
The trick is to let one model produce, and a **different model** judge — and
to run the loop automatically when the judge isn't satisfied.

LazyBridge gives you this in one parameter: **`verify=`**.

---

## The problem in one minute

Run the writer from Step 5 on its own:

```python
writer = Agent(
    engine=LLMEngine(
        "claude-haiku-4-5",
        system="Write a one-paragraph note. 80-120 words, no jargon, "
               "active voice, third person.",
    ),
    name="writer",
)

print(writer("AI agent frameworks in 2026").text())
```

You'll get a paragraph — but maybe:

- it's 140 words (your "120 words" was a suggestion to the model, not a contract)
- one sentence is in passive voice
- it slips in jargon ("orchestration primitives") despite the instruction

The model isn't malicious — it just optimised for "good paragraph", not for
"matches every rule in the prompt". Stricter prompts help, but they don't
catch everything.

You need a **second pair of eyes**.

---

## The fix — `verify=` in one line

```python
judge = Agent(
    engine=LLMEngine(
        "gpt-5.4-mini",                       # ← different LLM family on purpose
        system=(
            "You are a strict editor. Check that the text is 80–120 words, "
            "third person, active voice, no jargon (no 'leverage', 'utilise', "
            "'orchestration', 'paradigm'). "
            "Reply 'approved' if all rules are met. "
            "Otherwise reply with the rule it violates and a one-line "
            "instruction to fix it."
        ),
    ),
    name="judge",
)

writer = Agent(
    engine=LLMEngine(
        "claude-haiku-4-5",
        system="Write a one-paragraph note. 80-120 words, no jargon, "
               "active voice, third person.",
    ),
    verify=judge,                             # ← the one new line
    max_verify=3,                             # max retries before giving up
    name="writer",
)

print(writer("AI agent frameworks in 2026").text())
```

That's the whole feature.

When `writer` produces a draft, LazyBridge passes it to `judge`. If the judge
replies with `"approved"`, the result is returned. Otherwise, the judge's
verdict is fed back to the writer as **feedback on the next attempt** — up to
`max_verify` total attempts.

---

## The judge contract

A judge can be an `Agent` *or* any `Callable[[str], Any]`. Whatever you use,
it must follow this two-rule contract:

| The judge's reply | LazyBridge does |
|---|---|
| Starts with `"approved"` (case-insensitive) | Accept the draft, return the envelope |
| Anything else | Treat as rejection; inject the verdict as feedback and retry |

That's it. No special schema, no JSON, no special framework. The simplest
useful judge is a one-line Python function:

```python
def length_check(text: str) -> str:
    words = len(text.split())
    if 80 <= words <= 120:
        return "approved"
    return f"Text is {words} words; must be 80–120. Rewrite shorter/longer."

writer = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="..."),
    verify=length_check,        # plain function as judge
    max_verify=3,
)
```

For complex judgments you'd reach for an Agent judge. For mechanical checks
(length, format, regex) a callable is plenty — and it costs zero tokens.

---

## What happens at runtime — the trace

With `verbose=True` you see every attempt:

```text
[agent ▶ writer  model=claude-haiku-4-5]
  user: AI agent frameworks in 2026
  assistant: The 2026 landscape leverages new orchestration paradigms ...
  [verify ▶ judge  model=gpt-5.4-mini]
    user: <task> + <draft>
    assistant: Rule violated: jargon ('leverage', 'orchestration', 'paradigm').
               Rewrite without those words.
  ◆ verify rejected — retry 1/3

[agent ▶ writer  retry 1]
  user: AI agent frameworks in 2026
        Feedback: Rule violated: jargon (...). Rewrite without those words.
  assistant: In 2026 the agent-framework landscape consolidates ...
  [verify ▶ judge]
    assistant: approved
  ◆ verify approved

[done] verify_attempts=2  total_cost=$0.0023
```

Three things to notice:

1. **The judge sees the original task + the draft.** It can check that the
   draft *answers* the task, not just that it follows formatting rules.
2. **The verdict becomes feedback.** The retry doesn't just rerun the same
   prompt — it gets `"Feedback: <verdict>"` appended, so the writer learns
   what to fix.
3. **Costs add up.** The judge's tokens are part of `total_cost`. Set
   `max_verify` to a value you're willing to pay for in the worst case.

---

## Why "different LLM family" matters

Models from the same family share biases. Claude tends to use certain phrases;
GPT tends to over-explain; Gemini tends to be polite-to-a-fault. When the
judge is from the same family as the writer, they'll often agree where they
shouldn't — both miss the same blindspot.

A cross-family judge catches what the writer misses. The classic pairings:

| Doer | Judge | Why |
|---|---|---|
| `gpt-5.4-mini` | `claude-haiku-4-5` | Claude is good at strict format compliance |
| `claude-haiku-4-5` | `gpt-5.4-mini` | GPT is good at catching omissions and incomplete answers |
| Any cheap model | `claude-opus-4-7` or `gpt-5.4` | High-quality judging for high-stakes output |
| Any LLM | a plain Python callable | Mechanical rules (length, regex, JSON validity) — zero token cost |

In LazyBridge swapping the judge's model is one string change:

```python
verify=Agent(engine=LLMEngine("claude-opus-4-7", system="..."), name="judge")
```

In raw SDKs (OpenAI/Anthropic/Gemini), this would be two completely different
clients, two different request shapes, two different response parsers — plus
the retry loop and feedback threading you'd write by hand.

---

## verify= vs Guards — pick the right gate

LazyBridge has a *second* kind of output gate called `Guard` (covered in the
[Guards guide](../guides/mid/guards.md)). They look similar; they're not.

| | `verify=` | `Guard` |
|---|---|---|
| On rejection | Retries with feedback (up to `max_verify`) | Hard fail — returns `GuardBlocked` envelope |
| Mental model | "Editor — gives notes, asks for a redraft" | "Bouncer — refuses entry, no second chance" |
| Use case | Format, tone, completeness, factual checks | Policy, safety, security, hard requirements |
| Cost | Multiplies cost by attempts | Single check |

If a wrong output should *fail loudly*, use a Guard. If a wrong output should
*get fixed automatically*, use `verify=`. You can use both at the same time
(verify first, then guard the verified output).

---

## Where else `verify=` works

The same `verify=judge` parameter slots into three places:

```python
# 1. On the agent (final output gate — what we've been using)
writer = Agent(engine=..., verify=judge, max_verify=3)

# 2. On a sub-agent passed as a tool (gates that specific delegation)
tools = [researcher.as_tool("research", verify=judge, max_verify=2)]

# 3. Inside a Plan step (when you only want one step verified) — see Step 9
```

Same judge contract everywhere. Same retry semantics. No new concepts to
learn.

---

## A practical recipe — cheap rules + LLM semantics

A pattern that catches a *lot* of issues for a small fraction of the cost:

```python
from lazybridge import Agent, LLMEngine


def style_rules(text: str) -> str:
    """Cheap, deterministic checks. Zero tokens."""
    words = len(text.split())
    if not (80 <= words <= 120):
        return f"Length {words} words; must be 80–120."
    forbidden = ["leverage", "utilise", "orchestration", "paradigm", "synergy"]
    found = [w for w in forbidden if w.lower() in text.lower()]
    if found:
        return f"Jargon present: {found}. Rewrite plainly."
    return "approved"


fact_judge = Agent(                                     # cross-model semantic check
    engine=LLMEngine(
        "gpt-5.4-mini",
        system="You verify factual claims against the draft's stated topic. "
               "Reply 'approved' if no claim seems implausible or contradictory. "
               "Otherwise list the suspicious claim and ask for a revision.",
    ),
    name="fact_judge",
)


def combined_judge(text: str) -> str:
    """Cheap mechanical rules first; expensive semantic check only if needed."""
    rules = style_rules(text)
    if not rules.startswith("approved"):
        return rules                              # short-circuit; no LLM cost
    return fact_judge(text).text()                # only run when rules pass


writer = Agent(
    engine=LLMEngine(
        "claude-haiku-4-5",
        system="Write a one-paragraph note. 80–120 words, plain English.",
    ),
    verify=combined_judge,
    max_verify=3,
    name="writer",
)

env = writer("AI agent frameworks in 2026")
print(env.text())
print(f"\n[total cost: ${env.metadata.cost_usd:.4f}]")
```

The cheap callable filters out 80% of cases for free. The LLM judge only
runs when the draft has already passed the mechanical rules — saving tokens.

---

## Summary

| Concept | Syntax | What it gives you |
|---|---|---|
| Judge gate | `Agent(..., verify=judge_agent)` | Automatic retry-with-feedback |
| Retry cap | `max_verify=N` (default 3) | Hard limit before returning the last draft |
| Judge contract | Reply starts with `"approved"` → accept; else → feedback | Two-rule API, no schema |
| Cross-family judging | `verify=Agent(engine=LLMEngine("<other-model>"))` | Catches single-family blindspots |
| Plain function judge | `verify=my_callable` | Free, deterministic, zero tokens |
| Two-stage gates | Cheap callable → LLM judge | Filter mechanical issues before paying for semantic ones |

You now have an agent that catches its own mistakes. The next steps go back
to **structural** composition: pipelines, parallel work, and explicit DAGs —
where `verify=` continues to slot in wherever you put it.

---

[**Step 7: Sequential pipelines with `Agent.chain` →**](07-chain.md){ .md-button .md-button--primary }

[← Step 5: Why multi-agent + sub-agent as tool](05-multi-agent.md){ .md-button }
