# For LLM assistants

Use the current documentation when generating LazyBridge code. This refresh
targets 1.8.4, including the Codex transport fixes and Claude Code /
Codex engines, native aliases and opposite-engine review.

## Available surfaces

| Surface | Where | Use |
| --- | --- | --- |
| Bundled Claude skill | lazybridge/skill/ | Offline construction guidance; verify newer engine behavior in the guides |
| llms.txt | [Index](https://core.lazybridge.com/llms.txt) | Fetch relevant pages by topic |
| llms-full.txt | [Corpus](https://core.lazybridge.com/llms-full.txt) | Generated selected documentation; check size against the consuming model's budget |
| Codegen contract | [Current contract](codegen-contract.md) | Engine/config placement, return types, persistence and removed APIs |

Skills are guidance rather than runtime enforcement. The bundled skill and
llms.json remain companion resources; the current
[codegen contract](codegen-contract.md) and source signatures cover this refresh.
No code-generation behavior or model context fit is guaranteed by loading a skill.

## Bundled skill and machine-readable contract

The skill ships with the Python package at lazybridge/skill/. Follow
[Claude skill install](claude-skill.md) for the supported local installation
and packaged archive paths. It describes the mental model, canonical Agent
construction, sugar equivalents and common invalid APIs.

The companion [llms.json](https://github.com/selvaz/LazyBridge/blob/main/lazybridge/llms.json)
provides import maps, canonical patterns and anti-patterns. Use the
[canonical vs sugar reference](../concepts/canonical-vs-sugar.md) for convenience
factories. Human review, source signatures and execution tests remain necessary;
the bundled skill is guidance, not a runtime validator.

## Why several surfaces?

Compatible assistant environments can load reusable skill guidance. Link-aware
consumers can retrieve relevant pages from llms.txt. Consumers that require
inline documentation can use llms-full.txt after checking their remaining
context budget. The codegen contract is a focused current API checklist.

These formats solve different delivery problems. Neither a provider name nor
a context-window label guarantees discovery, adherence or fit. Generated files
update with each build; the hand-authored skill and JSON contract require
maintenance against source as APIs change.

## Start with the engine workflow

- [Coding agents](../guides/full/coding-agents.md): swap engines while preserving Python tools/state.
- [Claude Code writes, Codex reviews](../guides/full/claude-writes-codex-reviews.md): explicit reviewer choice, feedback and exhaustion.
- [SessionRegistry](../guides/full/session-registry.md): named native conversations across processes.
- [Permissions](../guides/full/coding-agent-config.md): native controls versus Python application tools.
- [LazyTools code bridge](https://tools.lazybridge.com/code-bridge/) and [MCP server](https://tools.lazybridge.com/mcp-server/): integration-layer APIs.

## Corpus generation

MkDocs generates llms.txt/llms-full.txt from configured sections on every build.
The index covers those sections, rather than every file in docs. Audit notes and
marketing drafts are excluded from the assistant corpus. Fetch relevant pages
when the full corpus does not fit; no fixed line count or token-budget guarantee
is maintained.

See [llms.txt explained](llms-txt.md), [Claude skill install](claude-skill.md),
[error recovery](error-recovery.md) and [reference](../reference/index.md).

## A practical generation workflow

1. Read the codegen contract and the guide for the selected engine.
2. Resolve only the additional tools/state/Plan pages the task requires.
3. Generate canonical construction with explicit names and bounded behavior.
4. Execute examples with mock model transports before trying authenticated work.
5. Use schema checks, deterministic tests and explicit acceptance for release.

For credentials, aliases and permissions, inspect the local runtime and source
version rather than filling gaps from a remembered training-data API.
