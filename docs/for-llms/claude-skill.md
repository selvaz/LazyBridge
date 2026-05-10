# Claude Skill install

The LazyBridge Claude Skill ships *with the library*. Three install
paths depending on which Claude surface you use.

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

## Path 2 — Claude API (`/v1/skills`)

For programmatic upload (CI, custom Claude integrations):

```bash
# Build a zip the API expects.
cd "$(python -c 'from lazybridge.skill import skill_path; print(skill_path())')"
zip -r /tmp/lazybridge-skill.zip .

# Upload via the Anthropic SDK.
python - <<'PY'
import anthropic
client = anthropic.Anthropic()
with open("/tmp/lazybridge-skill.zip", "rb") as f:
    skill = client.skills.create(file=f, name="lazybridge")
print(skill.id)
PY
```

Reference the skill ID on subsequent message calls. The skill is
loaded only when the trigger description matches; idle calls don't
pay the token cost.

## Path 3 — Claude.ai (web)

Claude.ai's Pro / Team plans accept skill zip uploads via the UI:

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

Loading the skill makes Claude follow these rules when editing
LazyBridge code:

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
  providers — never `from lazybridge.core.types import …` in
  application code.

## Verifying the skill is loaded

Ask Claude: "Show me the canonical Agent constructor for a
LazyBridge agent that uses a Plan engine with three sequential
steps". A skill-equipped Claude produces

```python
from lazybridge import Agent, LLMEngine, Plan, Step

pipeline = Agent(
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
