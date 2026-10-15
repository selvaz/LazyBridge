# Claude Skill install

The LazyBridge Claude Skill ships *with the library*. Three install
paths depending on which Claude surface you use.

## Layout

The skill uses progressive disclosure, so loading it costs little context:

```text
lazybridge/skill/
├── SKILL.md            # short index: mental model, golden rules, routing table
└── references/         # one file per area, read only when the task needs it
    ├── agents-and-engines.md
    ├── coding-engines.md
    ├── extensions.md
    ├── plan-and-pipelines.md
    ├── providers-and-models.md
    ├── sessions-and-memory.md
    ├── testing.md
    └── tools-and-nesting.md
```

Always install or upload the **whole directory**: `SKILL.md` routes to the
reference files by relative path. `lazybridge.skill.skill_path()` returns the
directory and `lazybridge.skill.reference_paths()` lists the reference files.
Any assistant that reads skills as a folder with a `SKILL.md` (Codex reads
`~/.codex/skills/<name>/SKILL.md`) can use the same directory.

## Path 1 — Claude Code (recommended)

Symlink the bundled skill directory into your local skills folder:

```bash
ln -s "$(python -c 'from lazybridge.skill import skill_path; print(skill_path())')" \
      ~/.claude/skills/lazybridge
```

Verify by listing your skills:

```bash
ls ~/.claude/skills/
# should show: lazybridge → /path/to/site-packages/lazybridge/skill
```

From now on, whenever Claude Code edits Python that imports from
`lazybridge`, it picks up the skill automatically. The skill's
trigger description (≤1024 chars; see `lazybridge/skill/SKILL.md`)
keys on phrases like "LazyBridge agent", "Agent.chain",
"`from_step`", "MCP server", "`HumanEngine`", "`OTelExporter`", and
"native_tools".

## Path 2 — Claude API

For programmatic upload, use the current
[Anthropic Agent Skills documentation](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview).
The bundled directory is the input artifact; SDK upload and message parameters
are an upstream API, not a LazyBridge contract. Follow that API's current
version and request format rather than the removed client.skills.create example.

## Path 3 — Claude.ai (web)

Where the current Claude.ai account exposes skill upload:

1. Build the zip as in Path 2 above (or download the release asset
   — see "Latest release" below).
2. Upload via Claude.ai → Settings → Capabilities → Skills →
   Upload.
3. The skill activates whenever a conversation matches the trigger
   description.

## Latest release

The release zip mirrors the bundled skill directory and is
attached to every tagged release on GitHub. Download from the
[Releases](https://github.com/selvaz/LazyBridge/releases) page when
you can't `pip install` (e.g. the upload-only Claude.ai path).

The release zip and the bundled directory have identical contents
— the release form just saves you the manual zip step.

## What the skill enforces

The bundled skill supplies guidance when editing LazyBridge code. It does not
enforce generated code or guarantee runtime correctness; confirm newer engine
behavior against [the current codegen contract](codegen-contract.md):

- **Canonical form first.** `Agent(engine=LLMEngine("model"), …)`
  with each constructor argument on its own line. The string-positional
  shortcut `Agent("model")` is a pure alias; everything else (deleted
  in 0.7.9: `Agent.from_model`, `Agent.from_engine`, `Agent.from_chain`,
  `Agent.from_plan`, `Agent.from_parallel`) is now part of the canonical
  ctor surface — write `Agent(engine=Plan(...))` directly.
- **No `asyncio.run(main())` wrapping.** The canonical call is
  `agent(task)`. Async forms (`await agent.run(task)`) only when
  the caller is already async.
- **No hand-written tool schemas.** Type hints + docstring drive
  the schema for `Tool(...)` (default `mode="signature"`); reach
  for `Tool.from_schema` only when a JSON schema is already known
  (MCP, OpenAPI bridges).
- **No `.text()` on structured output.** With `output=Model`,
  read `.payload` directly.
- **Public surface only.** Imports come from `lazybridge`,
  `lazybridge.ext.*`, or the explicit
  `lazybridge.core.providers.base.BaseProvider` for custom
  providers. CacheConfig is explicitly imported from lazybridge.core.types;
  the old top-level re-export is deprecated.

## Verifying the skill is loaded

Ask Claude: "Show me the canonical Agent constructor for a
LazyBridge agent that uses a Plan engine with three sequential
steps". A skill-equipped Claude produces

```python
from lazybridge import Agent, LLMEngine, Plan, Step

researcher = Agent(engine=LLMEngine("gpt-6.1-sol"), name="research")
ranker = Agent(engine=LLMEngine("gpt-6.1-sol"), name="rank")
writer = Agent(engine=LLMEngine("gpt-6.1-sol"), name="write")

pipeline = Agent(name="pipeline",
    engine=Plan(
        Step("research"),
        Step("rank"),
        Step("write"),
    ),
    tools=[researcher, ranker, writer],
)
```

A vanilla Claude (no skill) might produce
`Agent(engine=Plan(...))` or `Agent("model", ...)` — useful but not
the canonical form.

## Updating

The skill follows the library version. `pip install -U lazybridge`
updates the bundled skill directory automatically. Symlinked
installs (Path 1) pick up the update on the next process start;
API-uploaded skills (Path 2) need a re-upload.

## See also

- [For LLM assistants — overview](index.md) — what else ships
  alongside the skill.
- [llms.txt explained](llms-txt.md) — the URL-based alternative
  for non-Claude tools.
- The skill's source: `lazybridge/skill/SKILL.md` in the
  installed package.
