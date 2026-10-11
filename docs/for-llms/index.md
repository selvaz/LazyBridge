# For LLM assistants

Use the current documentation when generating LazyBridge code. This refresh
targets 1.8.0 plus post-release Codex transport fixes, including Claude Code /
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
