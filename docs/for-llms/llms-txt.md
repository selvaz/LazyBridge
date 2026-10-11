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

## Current engine contract

Follow the [codegen contract](codegen-contract.md), which distinguishes:

- Coding engines and provider APIs behind one Agent.
- Python Memory/Store/Session from native sessions/threads and their aliases.
- Opposite-engine review from an arbitrary verify judge.
- Execution success from explicit acceptance after bounded verification.
- LazyTools integration APIs from removed core connector imports.

[Claude skill install](claude-skill.md) provides the bundled guidance path;
skills do not enforce runtime acceptance or guarantee generated code correctness.
