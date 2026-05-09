# llms.txt explained

[`llms.txt`](https://llmstxt.org) is a 2024 convention proposed by
Jeremy Howard for advertising a documentation corpus to LLM-based
tools. A single Markdown file at a known URL lists every doc page,
grouped by section, with one-line annotations.

LazyBridge generates two artefacts at build time:

| URL | Contents | Use |
|---|---|---|
| <https://docs.lazybridge.com/llms.txt> | Index — every doc page grouped by section | Tools that resolve linked pages on demand |
| <https://docs.lazybridge.com/llms-full.txt> | Whole corpus concatenated (~10K lines, ~400 KB) | Long-context models that take the corpus directly into the prompt |

## Format

Per the [llmstxt.org spec](https://llmstxt.org/#format), `llms.txt`
is a Markdown file with:

- An H1 with the project name.
- An optional blockquote with a one-line summary.
- Free-form narrative sections.
- H2-delimited sections of link lists, each with one-line
  annotations.
- An optional `## Optional` section that consumers may drop when
  context budget is tight.

LazyBridge's `llms.txt` opens with the project name + summary, then
six sections matching the docs site nav: **Concepts**, **Guides**,
**Recipes**, **Decisions**, **Reference**, **Errors**.

Example (excerpt — full version at the URL above):

```text
# LazyBridge

> Zero-boilerplate multi-provider LLM agent framework.
> Engine + Tools + State, everything is a tool.

## Concepts

- [Mental model](https://docs.lazybridge.com/concepts/mental-model/)
- [Everything is a tool](https://docs.lazybridge.com/concepts/everything-is-a-tool/)
- [Progressive complexity](https://docs.lazybridge.com/concepts/progressive-complexity/)
- [Canonical vs sugar](https://docs.lazybridge.com/concepts/canonical-vs-sugar/)

## Guides

- [Agent](https://docs.lazybridge.com/guides/basic/agent/)
- [Tool](https://docs.lazybridge.com/guides/basic/tool/)
- ...
```

## Who reads it

A growing list of AI coding tools and LLM-side fetchers, including
agents that resolve `llms.txt` automatically when they see the
domain. The convention is **best-effort**: a tool that doesn't
support `llms.txt` still works against the regular HTML pages, and
a tool that does support it gets a more economical fetch path.

Major adoptions tracked in [llmstxt.org's directory](https://llmstxt.cloud).

## When to use which

- **`llms.txt`** when the consuming tool fetches linked pages on
  demand (Claude Code, Cursor, Windsurf, Mintlify-hosted assistants,
  etc.). The index is small (~80 lines); the tool resolves only the
  pages it needs.
- **`llms-full.txt`** when the consuming tool can't dispatch
  per-page fetches. Paste the file into a long-context model
  (Anthropic 200K, Gemini 1M, OpenAI 1M tier) and ask the question
  directly.
- **Skill** (Claude only) when you want enforcement — canonical-
  form rules, anti-pattern blacklist, trigger-driven loading. See
  [Claude Skill install](claude-skill.md).

## Generation

`llms.txt` and `llms-full.txt` are produced by the
[`mkdocs-llmstxt`](https://pawamoy.github.io/mkdocs-llmstxt/)
plugin during `mkdocs build`. The plugin is configured in
`mkdocs.yml`:

```yaml
plugins:
  - llmstxt:
      full_output: llms-full.txt
      sections:
        Concepts: ["concepts/*.md"]
        Guides: ["guides/**/*.md"]
        Recipes: ["recipes/*.md"]
        Decisions: ["decisions/*.md"]
        Reference: ["reference/*.md"]
        Errors: ["errors.md"]
```

Re-runs on every push to `main` via `.github/workflows/docs.yml`.
You don't need to do anything to keep them up to date — they're
always current with the published site.

## Caveats

- **`llms-full.txt` is large.** ~400 KB at the time of writing.
  Some 8K-context models can't ingest it directly; either chunk
  it or use the per-page index.
- **The format is best-effort.** Not every consumer respects
  the `## Optional` section convention or the "skip H1 narrative"
  hint. When in doubt, fetch the regular HTML pages.
- **Versioning.** The corpus reflects the docs site at deploy
  time; if you need a stable historical version, fetch the
  GitHub-releases tarball of the docs branch.

## See also

- [For LLM assistants — overview](index.md) — three surfaces
  (Skill / llms.txt / llms-full.txt) and when to use which.
- [Claude Skill install](claude-skill.md) — the skill alternative
  for Claude environments.
- [llmstxt.org spec](https://llmstxt.org) — the upstream
  convention.
