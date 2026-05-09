# For LLM assistants

LazyBridge ships a Claude Skill, an `llms.txt` index, and a
`llms-full.txt` corpus dump so AI coding assistants (Claude,
ChatGPT, Cursor, Windsurf, ...) can write LazyBridge code from the
authoritative current API rather than from training-data snapshots.

## What's available

| Surface | Where | Use it for |
|---|---|---|
| **Claude Skill** | Bundled with `pip install lazybridge` at `lazybridge/skill/` | Claude Code / Claude API / Claude.ai gets canonical-form guidance whenever it edits LazyBridge code |
| **`llms.txt`** | <https://docs.lazybridge.com/llms.txt> | Any tool following the [llmstxt.org](https://llmstxt.org) convention — discovers every doc page with one fetch |
| **`llms-full.txt`** | <https://docs.lazybridge.com/llms-full.txt> | The whole docs corpus concatenated; paste into a long-context model when you don't have a tool that supports `llms.txt` |

## Claude Skill

The skill is shipped *with the library*, not as a separate
download — when `pip install lazybridge` lands in your venv, the
skill lands too. See [Claude Skill install](claude-skill.md) for the
one-line symlink that makes Claude Code pick it up.

The skill teaches:

- The mental model (Engine + Tools + State).
- The canonical-first style rule (`Agent(engine=LLMEngine("…"))`,
  not the string-positional sugar).
- Every sugar form mapped to its canonical equivalent (the
  [Canonical vs sugar](../concepts/canonical-vs-sugar.md) reference,
  embedded for offline lookup).
- Anti-patterns: `asyncio.run(main())` wrapping, hand-written tool
  schemas, `.text()` on structured output, `Agent.from_model`
  boilerplate.

## `llms.txt` and `llms-full.txt`

Both are auto-generated at build time from the
[llmstxt.org-compliant](https://llmstxt.org) plugin
[`mkdocs-llmstxt`](https://pawamoy.github.io/mkdocs-llmstxt/).

- `llms.txt` is the **index** — a concise list of every doc page,
  grouped by section (Concepts / Guides / Recipes / Decisions /
  Reference / Errors). Tools that follow the convention discover
  the corpus from this single URL.
- `llms-full.txt` is the **whole corpus concatenated** —
  ~10 thousand lines, ~400 KB. Paste it into a long-context model
  when you don't have a tool that resolves `llms.txt` for you.

See [llms.txt explained](llms-txt.md) for the format spec, who's
adopting it, and what to expect when an assistant fetches it.

## Why three surfaces?

Different assistant ecosystems have different conventions:

- **Claude** ecosystem reads Skills natively. The skill is the
  highest-fidelity path: it carries enforcement rules ("write
  canonical form first", "skip asyncio.run") that a documentation
  page can't enforce.
- **OpenAI / generic LLMs** that follow the llmstxt.org convention
  fetch `llms.txt` and resolve linked pages.
- **Long-context models** (any provider) take `llms-full.txt`
  directly into the context window when the user wants the whole
  corpus available without per-page resolution.

All three are kept in sync: the skill is hand-authored from the
same code-vs-docs audit that produced the rest of the site;
`llms.txt` and `llms-full.txt` are mechanically regenerated on
every build.

## See also

- [Claude Skill install](claude-skill.md) — symlink, zip download,
  Claude.ai upload.
- [llms.txt explained](llms-txt.md) — what the format is, why
  LazyBridge ships it, and what tools consume it.
- [Canonical vs sugar](../concepts/canonical-vs-sugar.md) — the
  reference that the skill enforces and that `llms-full.txt`
  inlines.
