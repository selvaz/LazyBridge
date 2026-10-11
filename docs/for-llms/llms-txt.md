# llms.txt explained

[llms.txt](https://llmstxt.org/) is a convention for an LLM-readable documentation
index. LazyBridge builds it with
[mkdocs-llmstxt](https://pawamoy.github.io/mkdocs-llmstxt/).

| URL | Content |
| --- | --- |
| [llms.txt](https://core.lazybridge.com/llms.txt) | Index of configured documentation sections |
| [llms-full.txt](https://core.lazybridge.com/llms-full.txt) | Selected pages concatenated into one corpus |

The index is Markdown with a project heading, description and sectioned links.
Consumers can fetch only the relevant guide, or read the full corpus when their
actual context budget permits. Support is a consumer convention; generating the
file does not make every assistant discover or follow it.

## Format

The [format convention](https://llmstxt.org/#format) describes a project H1,
optional summary blockquote, narrative and H2 link-list sections. An Optional
section may identify material a consumer can omit. Consumers decide how much
of this convention to implement.

This abbreviated illustration uses current LazyBridge topics, rather than
promising an exact generated section count:

```text
# LazyBridge
> Claude Code and Codex as engines; Engine + Tools + State.

## Concepts
- Mental model
- Everything is a tool
- Progressive complexity
- Canonical vs sugar

## Guides
- Coding agents
- Claude Code writes, Codex reviews
- SessionRegistry
```

The actual artifact contains linked Markdown pages with annotations. The
build also emits Markdown versions of configured pages; they are intended
for text readers alongside the normal HTML pages.

## Who reads it and which surface to use

- Use the index when an assistant or retrieval tool can resolve relevant links.
  Loading an index is not the same as loading every page it names.
- Use the full corpus when the consumer needs pages inline and has enough
  remaining context. Measure the generated file and account for prompts,
  tool output and answers; byte size is not a token-count guarantee.
- Use the bundled skill for reusable construction guidance in a compatible
  assistant environment. It can influence code generation, but cannot enforce
  runtime behavior or replace execution tests.
- Fetch HTML or checked-out Markdown when the consumer ignores the convention.
  No supported-assistant list or automatic discovery behavior is guaranteed.

## Generation and scope

mkdocs.yml configures Getting started, Category guides, Concepts, Guides,
Recipes, Decisions, Reference, Errors, Why, Comparison and For LLM assistants.
The engine guides, collaboration pattern and native registry are included.
Marketing drafts and the October audit are outside these section patterns.

Both artifacts regenerate with mkdocs build and the docs workflow on main.
The published files reflect deployment time; read the checked-out source for
branch-specific changes. A full file can be larger than a model's available
context, especially after reference generation. Check its actual size and the
consumer's remaining budget rather than relying on a fixed token estimate.

For the full configuration, read mkdocs.yml. This excerpt shows how the
plugin selects pages; additional sections are configured there:

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

## Caveats

- A corpus can exceed the consumer's context budget. Retrieve by section or
  chunk it rather than relying on a fixed model-window estimate.
- Conventions are best-effort; consumers may ignore Optional or annotations.
- Deployment is versioned separately from your checkout. For a historical
  contract, use that Git tag's source and build configuration rather than
  assuming the published site matches it.
- Retrieved documentation is evidence about an API, not authorization to
  perform a tool action or skip application acceptance checks.

## Current engine contract

Follow the [codegen contract](codegen-contract.md), which distinguishes:

- Coding engines and provider APIs behind one Agent.
- Python Memory/Store/Session from native sessions/threads and their aliases.
- Opposite-engine review from an arbitrary verify judge.
- Execution success from explicit acceptance after bounded verification.
- LazyTools integration APIs from removed core connector imports.

[Claude skill install](claude-skill.md) provides the bundled guidance path;
skills do not enforce runtime acceptance or guarantee generated code correctness.

## See also

- [For LLM assistants](index.md): surfaces and workflow.
- [Claude skill install](claude-skill.md): packaged guidance and installation.
- [Codegen contract](codegen-contract.md): current API and return contracts.
- [llmstxt.org](https://llmstxt.org/): upstream format convention.
